"""
GPT model as plain JAX functions over a pytree of parameters. Follows nanochat's gpt.py:
- QK-norm, parameter-free RMSNorm, norm after the token embedding, no positional embeddings
- ReLU² MLP, no biases, untied embedding / lm_head, logit softcap
- per-layer learnable lambdas that rescale the residual stream and mix the input embedding back in
- local (sliding-window) attention layers with RoPE, and global ones without any positional encoding (NoPE)
Params are fp32; matmuls run in cfg.dtype (bf16, or fp32 for exact inference / RL); logits and the loss are fp32.
forward also runs incrementally on a KV cache (engine.py), with the same code.
"""

import contextlib

import jax
import jax.numpy as jnp


def init_params(cfg, key):
    d, L, V = cfg.d_model, cfg.n_layer, cfg.vocab_size
    assert d % cfg.head_dim == 0 and V % 64 == 0
    s = 3**0.5 * d**-0.5  # uniform bound that gives std 1/sqrt(d)
    uniform = lambda key, shape, bound: jax.random.uniform(key, shape, jnp.float32, -bound, bound)

    key, k_wte, k_head = jax.random.split(key, 3)
    blocks = []
    for k in jax.random.split(key, L):
        kq, kk, kv, k1 = jax.random.split(k, 4)
        blocks.append(dict(
            wq=uniform(kq, (d, d), s),
            wk=uniform(kk, (d, d), s),
            wv=uniform(kv, (d, d), s),
            wo=jnp.zeros((d, d)),  # output projections start at zero: each block starts as the identity
            w1=uniform(k1, (d, 4 * d), 0.4 * s),
            w2=jnp.zeros((4 * d, d)),
        ))
    depth = jnp.arange(L) / max(L - 1, 1)  # 0 -> 1 over layers
    return dict(
        wte=0.8 * jax.random.normal(k_wte, (V, d)),
        blocks=blocks,
        resid_lambdas=1.15 - 0.10 * depth,  # stronger residual early, weaker deep
        x0_lambdas=0.20 - 0.15 * depth,  # mix in more of the input embedding early
        lm_head=0.001 * jax.random.normal(k_head, (d, V)),
    )


def rms_norm(x, eps=1e-6):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.mean(x32**2, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def linear(x, w):
    return x @ w.astype(x.dtype)


def rope_tables(positions, head_dim, base, dtype):
    """cos / sin of shape (*positions.shape, head_dim/2)."""
    inv_freq = 1.0 / base ** (2 * jnp.arange(head_dim // 2) / head_dim)
    freqs = positions[..., None] * inv_freq  # (..., T, head_dim/2)
    return jnp.cos(freqs).astype(dtype), jnp.sin(freqs).astype(dtype)


def apply_rope(x, cos, sin):  # x: (B, T, H, D), cos / sin: (T, D/2) or (B, T, D/2)
    x1, x2 = jnp.split(x, 2, axis=-1)
    cos, sin = cos[..., None, :], sin[..., None, :]
    return jnp.concatenate([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], axis=-1)


def window_sizes(cfg):
    """Per-layer attention window: None = full causal context, int = sliding window of that many tokens."""
    windows = {"L": None, "S": cfg.local_window}
    sizes = [windows[cfg.window_pattern[i % len(cfg.window_pattern)]] for i in range(cfg.n_layer)]
    sizes[-1] = None  # the last layer always sees the full context
    return sizes


def attn_temperature(positions, cfg):
    """Query scale per position: 1 within the reference length (default: the training length), growing with
    log(position) beyond it, so that attention over a longer context stays about as sharp (as Llama 4's NoPE layers)."""
    pos = positions + 1
    return 1 + cfg.global_attn_temp * jnp.log(jnp.maximum(pos / (cfg.global_attn_temp_len or cfg.seq_len), 1.0))


def qkv(x, blk, rope, window, positions, cfg):
    """Queries, keys and values (B, T, H, D) of the tokens at positions (T,) or (B, T): RoPE on the local layers,
    QK-norm, and the temperature of the global layers."""
    B, T, d = x.shape
    H, D = d // cfg.head_dim, cfg.head_dim
    q = linear(x, blk["wq"]).reshape(B, T, H, D)
    k = linear(x, blk["wk"]).reshape(B, T, H, D)
    v = linear(x, blk["wv"]).reshape(B, T, H, D)
    if window is not None:
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
    q, k = 1.2 * rms_norm(q), 1.2 * rms_norm(k)  # QK-norm, slightly sharpened
    if window is None and cfg.global_attn_temp > 0:
        q = q * attn_temperature(positions, cfg)[..., None, None].astype(q.dtype)
    return q, k, v


def attention(x, blk, rope, window, positions, cfg, cache=None):
    """Causal self-attention (sliding window: query i sees keys i - window ... i). Packed rows are not masked per
    document: the model learns to ignore the previous documents. With cache = ((keys, values), rows, kv_len), the new
    keys and values are first written into the cache slots rows at their positions, and the queries attend to the
    first kv_len cache positions instead (masked by position, the same rule). rows=None: row b is slot b (all slots, in
    order), which saves gathering the attended keys and values. Returns (y, new cache)."""
    B, T, d = x.shape
    q, k, v = qkv(x, blk, rope, window, positions, cfg)
    if cache is not None:
        (k_cache, v_cache), rows, kv_len = cache
        idx = jnp.arange(B) if rows is None else rows
        k_cache = k_cache.at[idx[:, None], positions].set(k, mode="drop")  # positions past the cache are dropped
        v_cache = v_cache.at[idx[:, None], positions].set(v, mode="drop")
        key_positions = jnp.arange(kv_len)
        mask = key_positions <= positions[..., None]  # (B, T, kv_len)
        if window is not None:
            mask &= positions[..., None] <= key_positions + window
        keys, values = (k_cache[:, :kv_len], v_cache[:, :kv_len]) if rows is None else (k_cache[rows, :kv_len], v_cache[rows, :kv_len])
        y = jax.nn.dot_product_attention(q, keys, values, mask=mask[:, None], implementation="xla")
        cache = (k_cache, v_cache)
    else:
        y = jax.nn.dot_product_attention(
            q, k, v,
            is_causal=True,
            local_window_size=None if window is None else (window, 0),
            implementation="cudnn" if jax.default_backend() == "gpu" and q.dtype == jnp.bfloat16 else None,
        )
    return linear(y.reshape(B, T, d), blk["wo"]), cache


def mlp(x, blk):
    return linear(jnp.square(jax.nn.relu(linear(x, blk["w1"]))), blk["w2"])


def precision(cfg):
    """fp32 compute means fp32 matmuls (on GPU, XLA's default precision would use TF32)."""
    return jax.default_matmul_precision("highest") if cfg.dtype == "float32" else contextlib.nullcontext()


def forward(params, tokens, cfg, cache=None):
    """tokens: (B, T) int32 -> logits: (B, T, V) fp32.
    Incremental decoding (engine.py): cache = (layers, rows, start, kv_len), where layers holds the (keys, values) of
    every layer, (slots, max_len, H, D) each. Row b of tokens continues the sequence in cache slot rows[b] at position
    start[b]; see attention. Returns (logits, new layers) then."""
    dtype = jnp.dtype(cfg.dtype)
    positions = jnp.arange(tokens.shape[1])
    if cache is not None:
        layers, rows, start, kv_len = cache
        positions = start[:, None] + positions
    with precision(cfg):
        rope = rope_tables(positions, cfg.head_dim, cfg.rope_base, dtype)
        x = rms_norm(params["wte"][tokens]).astype(dtype)
        x0 = x
        new_layers = []
        for i, (blk, window) in enumerate(zip(params["blocks"], window_sizes(cfg))):
            x = params["resid_lambdas"][i].astype(dtype) * x + params["x0_lambdas"][i].astype(dtype) * x0
            layer_cache = None if cache is None else (layers[i], rows, kv_len)
            y, layer_cache = attention(rms_norm(x), blk, rope, window, positions, cfg, layer_cache)
            x = x + y
            x = x + mlp(rms_norm(x), blk)
            new_layers.append(layer_cache)
        logits = jnp.matmul(rms_norm(x), params["lm_head"].astype(dtype), preferred_element_type=jnp.float32)
        logits = cfg.logit_softcap * jnp.tanh(logits / cfg.logit_softcap)
    return logits if cache is None else (logits, new_layers)


def cross_entropy(logits, targets):
    """Per-token loss in nats: logits (..., V), targets (...) -> (...)."""
    return jax.nn.logsumexp(logits, axis=-1) - jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]


def token_losses(params, inputs, targets, cfg):
    """Per-token cross-entropy in nats, shape (B, T)."""
    return cross_entropy(forward(params, inputs, cfg), targets)


def flops_per_token(cfg):
    """Training FLOPs per token (forward + backward): 6 per matmul weight, plus attention scores (PaLM appendix B)."""
    d = cfg.d_model
    matmul_params = 12 * d * d * cfg.n_layer + d * cfg.vocab_size
    attn = sum(12 * d * min(w or cfg.seq_len, cfg.seq_len) for w in window_sizes(cfg))
    return 6 * matmul_params + attn
