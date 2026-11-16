"""
The update step shared by every arm (sai.py): a batch of trajectories (token in, token out: the exact ids the engine
sampled, with their log-probabilities under the behaviour policy; None for given tokens) and one objective.
    grpo        RL: A = r - mean(r of the group), no std division; token-level policy gradient with truncated importance
                sampling, no KL term (nanochat's GRPO + TIS)
    tb_vargrad  trajectory balance with VarGrad's log Z: q is trained toward R / Z, a distribution (a GFlowNet), not to
                maximize R
Both sides run in fp32 (cfg.dtype=float32): the engine's and the trainer's log-probabilities then agree to ~1e-5, so
the IS weight is ~1 and the mismatch metrics (from the loss's own forward pass) are a check. The optimizer is Muon +
AdamW (optim.py) at lr_scale x the pretraining LRs, linear decay to 0, no weight decay.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

import model
import optim
from engine import bucket

LENGTH_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384]  # padded sequence lengths of the training microbatches


@dataclasses.dataclass
class Trajectory:
    """A training sequence, wherever it was sampled (on-policy, tempered, a replay buffer, a gold chain:
    the same record). Token in, token out: the exact ids the engine produced."""
    tokens: list
    logprobs: list  # the behaviour policy's log probability of each token it sampled; None for given tokens
    reward: float
    group: int  # trajectories of the same prompt share a group (group baselines, per-group logZ, ...)

    @property
    def mask(self):
        return [int(lp is not None) for lp in self.logprobs]


def group_mean(x, groups, n_groups):
    total = jax.ops.segment_sum(x, groups, n_groups)
    count = jax.ops.segment_sum(jnp.ones_like(x), groups, n_groups)
    return (total / jnp.maximum(count, 1))[groups]


def grpo(logp, rollout_logp, mask, rewards, groups, n_groups, n_tokens, is_cap=2.0, adv_norm="mean", loss_agg="token"):
    """GRPO as nanochat (+ TIS): A = r - mean(r of the group), no std division; token-level policy gradient
    -sum sg(min(pi / pi_rollout, is_cap)) * A * log pi over the sampled tokens / n_tokens, no KL term.
    adv_norm="std": A / std(r of the group) (the original GRPO). loss_agg="seq": each sequence's sum is divided by its
    number of tokens (the original GRPO); n_tokens is then the number of sequences in the batch."""
    mean = group_mean(rewards, groups, n_groups)
    adv = rewards - mean
    if adv_norm == "std":
        adv = adv / (jnp.sqrt(group_mean((rewards - mean)**2, groups, n_groups)) + 1e-4)
    weight = jnp.minimum(jnp.exp(jax.lax.stop_gradient(logp) - rollout_logp), is_cap)
    per_token = weight * adv[:, None] * mask * logp
    if loss_agg == "seq":
        return -jnp.sum(jnp.sum(per_token, 1) / jnp.maximum(jnp.sum(mask, 1), 1)) / n_tokens
    return -jnp.sum(per_token) / n_tokens


def tb_vargrad(logp, rollout_logp, mask, rewards, groups, n_groups, n_tokens, **_):
    """Trajectory balance with VarGrad's log Z (a GFlowNet objective: q(z) is trained toward R(z) / Z, a distribution,
    not to maximize R; sai.py). rewards are log R; per group (one target), log Z = mean(log R - log q)
    (stop-gradient), over on- and off-policy samples alike; the loss is the sum over sequences of
    (log Z + log q - log R)^2, divided by n_tokens (the number of sequences: loss_agg="seq")."""
    logq = jnp.sum(mask * logp, axis=1)
    logz = jax.lax.stop_gradient(group_mean(rewards - logq, groups, n_groups))
    return jnp.sum(jnp.where(jnp.sum(mask, 1) > 0, (logz + logq - rewards) ** 2, 0.0)) / n_tokens


# An objective maps (trainer logprobs (R, T), rollout logprobs (R, T), loss mask (R, T), rewards (R,), group ids (R,),
# number of group ids, number of loss tokens in the whole batch) to a scalar loss; microbatches hold whole groups, so
# group statistics are exact.
OBJECTIVES = {"grpo": grpo, "tb_vargrad": tb_vargrad}


def loss_and_stats(params, batch, n_tokens, cfg, objective, is_cap):
    """The objective on one microbatch, plus sums for the logged metrics from the same forward pass: on-policy, these
    params sampled the batch, so pi_train / pi_rollout measures the trainer-engine mismatch."""
    inputs, targets, mask, rollout_logp, rewards, groups = batch
    logits = model.forward(params, inputs, cfg)
    logz = jax.nn.logsumexp(logits, axis=-1)
    logp = jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0] - logz
    rollout_logp = jnp.where(jnp.isnan(rollout_logp), jax.lax.stop_gradient(logp), rollout_logp)  # off-policy: weight 1
    loss = objective(logp, rollout_logp, mask, rewards, groups, groups.shape[0] + 1, n_tokens)
    log_ratio = jax.lax.stop_gradient(logp) - rollout_logp
    ratio = jnp.exp(log_ratio)
    sg_logits = jax.lax.stop_gradient(logits)
    entropy = jax.lax.stop_gradient(logz) - jnp.sum(jax.nn.softmax(sg_logits, axis=-1) * sg_logits, axis=-1)
    on = mask > 0
    stats = dict(tokens=jnp.sum(mask), kl=jnp.sum(mask * (ratio - 1 - log_ratio)),
                 abs_logp_diff=jnp.sum(mask * jnp.abs(log_ratio)),
                 max_abs_logp_diff=jnp.max(jnp.where(on, jnp.abs(log_ratio), 0.0)),
                 is_weight_max=jnp.max(jnp.where(on, ratio, 0.0)), is_clipped=jnp.sum(mask * (ratio > is_cap)),
                 outside_05_2=jnp.sum(mask * ((ratio < 0.5) | (ratio > 2))), entropy=jnp.sum(mask * entropy))
    return loss, stats


@functools.partial(jax.jit, static_argnames=("cfg", "objective", "is_cap"), donate_argnames="grads")
def accumulate(params, grads, batch, n_tokens, cfg, objective, is_cap):
    with model.precision(cfg):
        (loss, stats), g = jax.value_and_grad(loss_and_stats, has_aux=True)(params, batch, n_tokens, cfg, objective,
                                                                            is_cap)
    return jax.tree.map(jnp.add, grads, g), loss, stats


@functools.partial(jax.jit, static_argnames="cfg", donate_argnames=("params", "opt_state"))
def apply_update(params, opt_state, grads, step, cfg):
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    params, opt_state = optim.update(params, grads, opt_state, step, cfg, weight_decay=False)
    return params, opt_state, grad_norm


def microbatches(trajectories, micro_tokens):
    """Whole groups, longest first, packed into microbatches of ~micro_tokens tokens (at least one group) padded to a
    bucketed length: (inputs, targets, mask, rollout logprobs, rewards, group ids) arrays. Group ids are renumbered
    0, 1, ... per microbatch; padding rows get their own id (= rows) and mask 0."""
    by_group = {}
    for t in trajectories:
        by_group.setdefault(t.group, []).append(t)
    groups = sorted(by_group.values(), key=lambda g: -max(len(t.tokens) for t in g))
    largest = max(len(g) for g in groups)
    while groups:
        length = bucket(max(len(t.tokens) for t in groups[0]) - 1, LENGTH_BUCKETS)
        rows = max(micro_tokens // length, largest)
        chunk = []
        while groups and len(chunk) + len(groups[0]) <= rows:
            g = groups.pop(0)
            chunk += [(t, g[0].group) for t in g]
        ids = {gid: i for i, gid in enumerate(dict.fromkeys(gid for _, gid in chunk))}
        inputs, targets = np.zeros((rows, length), np.int32), np.zeros((rows, length), np.int32)
        mask, old = np.zeros((rows, length), np.float32), np.zeros((rows, length), np.float32)
        rewards, group_ids = np.zeros(rows, np.float32), np.full(rows, rows, np.int32)
        for r, (t, gid) in enumerate(chunk):
            n = len(t.tokens) - 1
            inputs[r, :n], targets[r, :n], mask[r, :n] = t.tokens[:-1], t.tokens[1:], t.mask[1:]
            old[r, :n] = [lp or 0.0 for lp in t.logprobs[1:]]
            rewards[r], group_ids[r] = t.reward, ids[gid]
        yield inputs, targets, mask, old, rewards, group_ids


def train_step(params, opt_state, trajectories, step, cfg, objective, args):
    """One optimizer step on the trajectories (on- or off-policy alike). Returns params, opt_state and metrics."""
    n_tokens = float(sum(sum(t.mask[1:]) for t in trajectories))
    if args.loss_agg == "seq":  # the objective's normalizer: sequences, not tokens
        n_tokens = float(len(trajectories))
    grads = jax.tree.map(jnp.zeros_like, params)
    losses, stats = [], []
    for batch in microbatches(trajectories, args.micro_tokens):
        grads, loss, st = accumulate(params, grads, batch, n_tokens, cfg=cfg, objective=objective, is_cap=args.is_cap)
        losses.append(loss)
        stats.append(st)
    params, opt_state, grad_norm = apply_update(params, opt_state, grads, step, cfg=cfg)
    stats = jax.device_get(stats)
    total = lambda k: sum(float(s[k]) for s in stats)
    n = total("tokens")
    metrics = {"train/loss": float(sum(jax.device_get(losses))), "train/grad_norm": float(grad_norm),
               "train/lrm": float(optim.lr_multiplier(step, cfg)), "train/tokens": n,
               "train/entropy": total("entropy") / n, "mismatch/kl": total("kl") / n,
               "mismatch/mean_abs_logp_diff": total("abs_logp_diff") / n,
               "mismatch/max_abs_logp_diff": max(float(s["max_abs_logp_diff"]) for s in stats),
               "mismatch/is_weight_max": max(float(s["is_weight_max"]) for s in stats),
               "mismatch/is_clipped_frac": total("is_clipped") / n,
               "mismatch/outside_05_2_frac": total("outside_05_2") / n}
    return params, opt_state, metrics
