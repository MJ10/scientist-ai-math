"""
Muon for the hidden 2D weights, AdamW for everything else (embeddings, lm_head, scalars). Follows nanochat's optim.py:
- Muon: Nesterov momentum -> Polar Express orthogonalization -> NorMuon variance reduction -> cautious weight decay
- AdamW: decoupled weight decay, per-group LRs and betas
- schedules: LR warmup / constant / linear warmdown, Muon momentum warmup, weight decay cosine-decayed to 0
Everything is a pure function of (params, grads, state, step), so it all lives inside the jitted train step.
"""

import math

import jax
import jax.numpy as jnp

# Polar Express coefficients for 5 iterations (https://arxiv.org/abs/2505.16932), as used by nanochat
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]
MUON_BETA2 = 0.9  # NorMuon second-moment EMA


def adamw_groups(cfg):
    """Hyperparameters of the AdamW groups, keyed by the top-level param name. Everything under 'blocks' uses Muon."""
    d_scale = (cfg.d_model / 768) ** -0.5  # nanochat tuned AdamW LRs at d_model=768, scaled ∝ 1/sqrt(d_model)
    return {
        "wte": dict(lr=cfg.embedding_lr * d_scale, b1=0.8, b2=0.995, wd=0.001),
        "lm_head": dict(lr=cfg.unembedding_lr * d_scale, b1=0.8, b2=0.96, wd=0.01),
        "resid_lambdas": dict(lr=cfg.scalar_lr * 0.01, b1=0.8, b2=0.95, wd=0.05),
        "x0_lambdas": dict(lr=cfg.scalar_lr, b1=0.96, b2=0.95, wd=0.0),
    }


def batch_lr_scale(cfg):
    return (cfg.total_batch_tokens / 2**19) ** 0.5  # LRs were tuned at batch 2^19 tokens, scaled ∝ sqrt(batch)


# -----------------------------------------------------------------------------
# schedules (step is a traced int32 scalar)

def lr_multiplier(step, cfg):
    warmdown_steps = round(cfg.warmdown_ratio * cfg.num_steps)
    warmup = (step + 1) / cfg.warmup_steps
    progress = (cfg.num_steps - step) / warmdown_steps  # >1 before warmdown, 1 -> 0 during it
    warmdown = progress + (1 - progress) * cfg.final_lr_frac
    return jnp.minimum(jnp.minimum(warmup, 1.0), warmdown)


def muon_momentum(step, cfg):
    """0.85 -> 0.97 over the first 400 steps, then down to 0.90 during the LR warmdown."""
    warmdown_steps = round(cfg.warmdown_ratio * cfg.num_steps)
    warmdown_progress = jnp.clip((step - (cfg.num_steps - warmdown_steps)) / warmdown_steps, 0.0, 1.0)
    return 0.85 + 0.12 * jnp.minimum(step / 400, 1.0) - 0.07 * warmdown_progress


def muon_weight_decay(step, cfg):
    return cfg.weight_decay * 0.5 * (1 + jnp.cos(math.pi * step / cfg.num_steps))


# -----------------------------------------------------------------------------
# update rules (per parameter)

def polar_express(g):
    """Approximately replaces g with the nearest semi-orthogonal matrix U V^T (where g = U S V^T)."""
    x = g.astype(jnp.bfloat16)
    x = x / (jnp.linalg.norm(x.astype(jnp.float32)) * 1.01 + 1e-6).astype(x.dtype)
    transpose = x.shape[0] > x.shape[1]  # iterate on the smaller Gram matrix
    if transpose:
        x = x.T
    for a, b, c in POLAR_EXPRESS_COEFFS:
        A = x @ x.T
        x = a * x + (b * A + c * A @ A) @ x
    if transpose:
        x = x.T
    return x.astype(jnp.float32)


def muon_update(p, g, mu, nu, lr, momentum, wd):
    """p: (fan_in, fan_out). mu: momentum buffer. nu: NorMuon second moment, factored over the smaller dim."""
    mu = mu + (1 - momentum) * (g - mu)
    g = g + momentum * (mu - g)  # Nesterov
    g = polar_express(g)

    # NorMuon: normalize per-neuron update scales, then restore the overall norm
    red_axis = 0 if p.shape[1] >= p.shape[0] else 1
    v_mean = jnp.mean(g**2, axis=red_axis, keepdims=True)
    red_size = g.shape[red_axis]
    nu = nu + (1 - MUON_BETA2) * (v_mean - nu)
    step_size = jax.lax.rsqrt(jnp.maximum(nu, 1e-10))
    v_norm = jnp.sqrt(jnp.sum(v_mean) * red_size)
    v_norm_new = jnp.sqrt(jnp.sum(v_mean * red_size * step_size**2))
    g = g * step_size * (v_norm / jnp.maximum(v_norm_new, 1e-10))

    lr = lr * max(1.0, p.shape[1] / p.shape[0]) ** 0.5  # larger steps for wide (fan_out > fan_in) matrices
    cautious = (g * p) >= 0  # only decay weights where the update agrees in sign
    p = p - lr * g - lr * wd * p * cautious
    return p, mu, nu


def adamw_update(p, g, mu, nu, t, lr, b1, b2, wd, eps=1e-10):
    p = p * (1 - lr * wd)
    mu = mu + (1 - b1) * (g - mu)
    nu = nu + (1 - b2) * (g**2 - nu)
    p = p - (lr / (1 - b1**t)) * mu / (jnp.sqrt(nu / (1 - b2**t)) + eps)
    return p, mu, nu


# -----------------------------------------------------------------------------
# whole-model init / update

def labels(params):
    """Same tree as params, with each leaf replaced by its group name."""
    return {name: jax.tree.map(lambda _: "muon", sub) if name == "blocks" else name for name, sub in params.items()}


def init(params):
    def nu_init(label, p):
        if label == "muon":
            return jnp.zeros((1, p.shape[1]) if p.shape[1] >= p.shape[0] else (p.shape[0], 1))
        return jnp.zeros_like(p)
    return dict(mu=jax.tree.map(jnp.zeros_like, params), nu=jax.tree.map(nu_init, labels(params), params))


def update(params, grads, state, step, cfg, weight_decay=True):
    """weight_decay=False: no weight decay at all (Muon's and the AdamW groups'), e.g. for RL."""
    lrm = lr_multiplier(step, cfg) * batch_lr_scale(cfg) * cfg.lr_scale
    momentum, wd = muon_momentum(step, cfg), muon_weight_decay(step, cfg) * weight_decay
    adamw = adamw_groups(cfg)
    t = math.inf if cfg.init_from and cfg.init_optimizer else step + 1  # loaded moments need no bias correction

    def update_leaf(label, p, g, mu, nu):
        if label == "muon":
            return muon_update(p, g, mu, nu, cfg.matrix_lr * lrm, momentum, wd)
        h = adamw[label]
        return adamw_update(p, g, mu, nu, t, h["lr"] * lrm, h["b1"], h["b2"], h["wd"] * weight_decay)

    tree_labels = labels(params)
    out = jax.tree.map(update_leaf, tree_labels, params, grads, state["mu"], state["nu"])
    pick = lambda i: jax.tree.map(lambda _, o: o[i], tree_labels, out)
    return pick(0), dict(mu=pick(1), nu=pick(2))
