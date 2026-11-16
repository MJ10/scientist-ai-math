"""
Inference engine: batched generation with a KV cache, for sampling chains (sai.py). A much simplified nano-vllm
(github.com/GeeeekExplorer/nano-vllm):
- Each sequence gets a slot of a fixed-capacity KV cache (n_slots x max_len per layer) and keeps it until it is
  released, so a stopped sequence can be continued without recomputing it.
- Every step either prefills (the pending tokens of new or continued sequences, in chunks) or decodes for every
  running sequence, as nano-vllm's scheduler does. A decode step runs up to decode_steps tokens in one jitted call (a
  lax.while_loop with on-device sampling and stop handling, ending early once every row has stopped), so the host
  round trip is paid once per decode_steps tokens. Shapes are bucketed (rows, chunk length, attended cache length), so
  the jitted functions compile a bounded number of times.
- The forward pass is model.forward itself, given the cache (same code, dtypes and masks). With cfg.dtype="float32" on
  both sides, logits match the full forward pass to ~4e-5 (logprobs ~2e-6 on average): the trainer's log-probabilities
  of sampled chains equal the engine's, which the GFlowNet and RL objectives rely on.
- Sampling: greedy, temperature, top-k, top-p, min-p. Every sampled token comes with its log probability under the
  model (temperature 1, no truncation). For training, sample with the default Sampling() (temperature 1, full softmax),
  so that these are the behaviour policy's log probabilities.
"""

import collections
import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

import model

ROW_BUCKETS = [2**i for i in range(13)]  # sequences per step (Engine(row_buckets=...): coarser = fewer compilations)
CHUNK_BUCKETS = [16, 64, 256, 1024]  # tokens per sequence in a prefill step (longer prompts are prefilled in chunks)
MIN_KV_LEN = 256  # attended cache lengths: 256, 512, ... up to max_len


def bucket(n, sizes):
    return next(s for s in sizes if s >= n)


@dataclasses.dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0  # 0: greedy
    top_k: int = 0  # 0: off
    top_p: float = 1.0  # 1: off
    min_p: float = 0.0  # keep tokens with at least min_p x the top token's probability; 0: off
    repetition_penalty: float = 1.0  # CTRL's: logits of tokens already sampled are divided by it (x it if < 0); 1: off

    @property
    def penalized(self):
        return self.repetition_penalty != 1


def seen_tokens(seqs, n_rows, vocab_size):
    """(n_rows, vocab_size) bool: the tokens each sequence sampled so far (not its prompt, tool outputs or stop tokens)."""
    seen = np.zeros((n_rows, vocab_size), bool)
    for i, seq in enumerate(seqs):
        seen[i, [t for t, m in zip(seq.tokens, seq.mask) if m]] = True
        seen[i, list(seq.stop)] = False
    return seen


class Sequence:
    """A token sequence being generated: stops at a token in stop, or after max_tokens new tokens, or when the cache
    is full. logprobs[i] is the model's log probability of tokens[i] if it was sampled, else None (prompt, tool
    outputs), so the sampled tokens are the ones to train on."""

    def __init__(self, tokens, max_tokens=256, stop=()):
        self.tokens, self.logprobs = list(tokens), [None] * len(tokens)
        self.max_tokens, self.stop = max_tokens, frozenset(stop)
        self.n_new = 0  # tokens sampled since the sequence was last (re)started
        self.slot, self.n_cached = None, 0  # its cache slot, and how many of its tokens are in it
        self.finish_reason = None  # "stop": a stop token, "length": max_tokens or the cache is full

    def extend(self, tokens, max_tokens=None):
        """Appends given tokens (a tool output, the next user turn); Engine.add then continues generating."""
        self.tokens += tokens
        self.logprobs += [None] * len(tokens)
        self.max_tokens = max_tokens or self.max_tokens

    @property
    def done(self):
        return self.finish_reason is not None

    @property
    def mask(self):
        """1 for the tokens the model sampled (to train on), 0 for given ones."""
        return [int(lp is not None) for lp in self.logprobs]


def init_cache(cfg, n_slots, max_len):
    """Keys and values of every layer, (n_slots, max_len, heads, head_dim) each, in the compute dtype."""
    shape = (n_slots, max_len, cfg.d_model // cfg.head_dim, cfg.head_dim)
    return [(jnp.zeros(shape, cfg.dtype), jnp.zeros(shape, cfg.dtype)) for _ in range(cfg.n_layer)]


def sample(logits, key, sampling, seen=None):
    if seen is not None:  # repetition penalty on the tokens of seen_tokens
        p = sampling.repetition_penalty
        logits = jnp.where(seen, jnp.where(logits > 0, logits / p, logits * p), logits)
    if sampling.temperature == 0:
        return jnp.argmax(logits, axis=-1)
    logits = logits / sampling.temperature
    if sampling.min_p > 0:
        logits = jnp.where(logits < logits.max(axis=-1, keepdims=True) + jnp.log(sampling.min_p), -jnp.inf, logits)
    if sampling.top_k:
        kth = jax.lax.top_k(logits, sampling.top_k)[0][:, -1:]
        logits = jnp.where(logits < kth, -jnp.inf, logits)
    if sampling.top_p < 1:  # keep the most likely tokens until their probability reaches top_p
        desc = -jnp.sort(-logits, axis=-1)
        probs = jax.nn.softmax(desc, axis=-1)
        keep = jnp.cumsum(probs, axis=-1) - probs < sampling.top_p
        threshold = jnp.min(jnp.where(keep, desc, jnp.inf), axis=-1, keepdims=True)
        logits = jnp.where(logits < threshold, -jnp.inf, logits)
    return jax.random.categorical(key, logits, axis=-1)


@functools.partial(jax.jit, static_argnames=("cfg", "kv_len", "sampling"), donate_argnames="cache")
def step_fn(params, cache, rows, tokens, start, n_new, key, cfg, kv_len, sampling, seen=None):
    """Runs forward on the new tokens of each row and samples the token after its last one (n_new - 1). Returns the
    cache, the sampled tokens and their log probabilities (computed as in model.token_losses, without penalties)."""
    logits, cache = model.forward(params, tokens, cfg, cache=(cache, rows, start, kv_len))
    last = logits[jnp.arange(tokens.shape[0]), n_new - 1]  # (R, V)
    next_tokens = sample(last, key, sampling, seen)
    logprobs = -(jax.nn.logsumexp(last, axis=-1) - jnp.take_along_axis(last, next_tokens[:, None], axis=-1)[:, 0])
    return cache, next_tokens, logprobs


@functools.partial(jax.jit, static_argnames=("cfg", "kv_len", "sampling", "n_steps"), donate_argnames="cache")
def decode_fn(params, cache, rows, tokens, start, budget, stop_ids, key, cfg, kv_len, sampling, n_steps):
    """Up to n_steps decode steps in one call. Row b feeds tokens[b] at position start[b] (then its own samples) and
    samples until it samples one of stop_ids[b] (-1 = padding) or has sampled budget[b] tokens; the loop ends once every
    row has stopped. Each step is step_fn's (same forward, sampling and log probabilities). A stopped row keeps running
    on its own cache slot past its end, which is harmless: those positions are rewritten before anything attends to them.
    Returns the cache, the sampled tokens and log probabilities (R, n_steps) and the number sampled per row (R,)."""
    R = tokens.shape[0]

    def body(state):
        i, cache, token, done, n, out_tokens, out_logprobs = state
        logits, cache = model.forward(params, token[:, None], cfg, cache=(cache, rows, start + i, kv_len))
        last = logits[:, 0]
        next_tokens = sample(last, jax.random.fold_in(key, i), sampling)
        logprobs = -(jax.nn.logsumexp(last, axis=-1) - jnp.take_along_axis(last, next_tokens[:, None], axis=-1)[:, 0])
        out_tokens, out_logprobs = out_tokens.at[:, i].set(next_tokens), out_logprobs.at[:, i].set(logprobs)
        n = n + ~done
        done = done | jnp.any(next_tokens[:, None] == stop_ids, axis=-1) | (n >= budget)
        return i + 1, cache, next_tokens, done, n, out_tokens, out_logprobs

    state = (jnp.int32(0), cache, tokens, budget <= 0, jnp.zeros(R, jnp.int32),
             jnp.zeros((R, n_steps), jnp.int32), jnp.zeros((R, n_steps), jnp.float32))
    _, cache, _, _, n, out_tokens, out_logprobs = jax.lax.while_loop(
        lambda state: (state[0] < n_steps) & ~jnp.all(state[3]), body, state)
    return cache, out_tokens, out_logprobs, n


class Engine:
    """Generates many sequences at once. Either generate() a batch of prompts, or drive it: add() sequences, call
    step() until nothing is waiting or running, and release() finished sequences once they won't be continued."""

    def __init__(self, params, cfg, n_slots=32, max_len=None, prefill_tokens=8192, decode_steps=16, all_slots_frac=0.5,
                 row_buckets=ROW_BUCKETS, seed=0):
        self.params, self.cfg, self.n_slots = params, cfg, n_slots
        self.max_len = max_len or cfg.seq_len
        self.prefill_tokens = prefill_tokens  # at most this many tokens (rows x chunk) per prefill step
        self.decode_steps = decode_steps  # at most this many tokens per sequence per decode step (one jitted call)
        self.all_slots_frac = all_slots_frac  # decode over all slots when at least this fraction of them are running
        self.row_buckets = [b for b in row_buckets if b < n_slots] + [n_slots + 1]  # rows per step
        self.kv_buckets = sorted({min(MIN_KV_LEN * 2**i, self.max_len) for i in range(20)})
        self.cache = init_cache(cfg, n_slots + 1, self.max_len)  # the extra slot takes the padding rows' writes
        self.free = list(range(n_slots))[::-1]
        self.waiting, self.running = collections.deque(), []
        self.key = jax.random.key(seed)

    def add(self, seq):
        """Queues a new sequence, or continues a finished one in its slot (after seq.extend). A sequence that doesn't
        fit in the cache is finished right away (finish_reason "length")."""
        seq.n_new, seq.finish_reason = 0, None
        if len(seq.tokens) >= self.max_len:
            seq.finish_reason = "length"
        elif seq.slot is None:
            self.waiting.append(seq)
        else:
            self.running.append(seq)
        return seq

    def release(self, seq):
        if seq.slot is not None:
            self.free.append(seq.slot)
            seq.slot, seq.n_cached = None, 0

    def has_work(self):
        return bool(self.waiting or self.running)

    def step(self, sampling=Sampling()):
        """One prefill or decode step. Returns the sequences that finished in it (they keep their slots)."""
        while self.waiting and self.free:
            seq = self.waiting.popleft()
            seq.slot, seq.n_cached = self.free.pop(), 0
            self.running.append(seq)
        prefill = [s for s in self.running if len(s.tokens) - s.n_cached > 1]
        if prefill:
            chunk = min(bucket(max(len(s.tokens) - s.n_cached for s in prefill), CHUNK_BUCKETS + [np.inf]),
                        CHUNK_BUCKETS[-1])
            batch = prefill[:max(1, self.prefill_tokens // chunk)]
        else:
            chunk, batch = 1, self.running
        if not batch:
            return []
        if chunk == 1 and self.decode_steps > 1 and not sampling.penalized:
            return self.decode(batch, sampling)

        n_rows = bucket(len(batch), self.row_buckets)
        rows = np.full(n_rows, self.n_slots, np.int32)  # padding rows write into the extra slot
        tokens, start, n_new = np.zeros((n_rows, chunk), np.int32), np.zeros(n_rows, np.int32), np.ones(n_rows, np.int32)
        for i, seq in enumerate(batch):
            new = seq.tokens[seq.n_cached:seq.n_cached + chunk]
            rows[i], start[i], n_new[i], tokens[i, :len(new)] = seq.slot, seq.n_cached, len(new), new
        kv_len = bucket(int((start + n_new).max()), self.kv_buckets)
        seen = seen_tokens(batch, n_rows, self.cfg.vocab_size) if sampling.penalized else None
        self.key, key = jax.random.split(self.key)
        self.cache, next_tokens, logprobs = step_fn(self.params, self.cache, rows, tokens, start, n_new, key,
                                                    cfg=self.cfg, kv_len=kv_len, sampling=sampling, seen=seen)
        next_tokens, logprobs = np.asarray(next_tokens), np.asarray(logprobs)

        finished = []
        for i, seq in enumerate(batch):
            seq.n_cached += int(n_new[i])
            if seq.n_cached < len(seq.tokens):
                continue  # more to prefill
            token = int(next_tokens[i])
            seq.tokens.append(token)
            seq.logprobs.append(float(logprobs[i]))
            seq.n_new += 1
            if token in seq.stop:
                seq.finish_reason = "stop"
            elif seq.n_new >= seq.max_tokens or len(seq.tokens) >= self.max_len:
                seq.finish_reason = "length"
            if seq.done:
                finished.append(seq)
        self.running = [s for s in self.running if not s.done]
        return finished

    def decode(self, batch, sampling):
        """Decodes up to decode_steps tokens for every sequence of batch in one call (decode_fn). With many running
        sequences, row b is cache slot b for every slot (idle ones write nowhere: their positions are past the cache),
        which saves gathering the attended keys and values; otherwise the rows are the batch's slots."""
        all_slots = len(batch) >= self.all_slots_frac * self.n_slots
        n_rows, n_steps = self.n_slots + 1 if all_slots else bucket(len(batch), self.row_buckets), self.decode_steps
        n_stop = bucket(max(len(s.stop) for s in batch), [8, 32, 128, 512])
        index = [seq.slot for seq in batch] if all_slots else list(range(len(batch)))  # row of each sequence
        rows = np.full(n_rows, self.n_slots, np.int32)
        tokens, budget = np.zeros(n_rows, np.int32), np.zeros(n_rows, np.int32)
        start = np.full(n_rows, self.max_len if all_slots else 0, np.int32)
        stop_ids = np.full((n_rows, n_stop), -1, np.int32)
        for i, seq in zip(index, batch):
            rows[i], tokens[i], start[i] = seq.slot, seq.tokens[-1], seq.n_cached
            budget[i] = min(seq.max_tokens - seq.n_new, self.max_len - len(seq.tokens))
            stop_ids[i, :len(seq.stop)] = sorted(seq.stop)
        kv_len = bucket(min(max(seq.n_cached for seq in batch) + n_steps, self.max_len), self.kv_buckets)
        self.key, key = jax.random.split(self.key)
        self.cache, out_tokens, out_logprobs, n = decode_fn(
            self.params, self.cache, None if all_slots else rows, tokens, start, budget, stop_ids, key, cfg=self.cfg,
            kv_len=kv_len, sampling=sampling, n_steps=n_steps)
        out_tokens, out_logprobs, n = np.asarray(out_tokens), np.asarray(out_logprobs), np.asarray(n)

        finished = []
        for i, seq in zip(index, batch):
            new = out_tokens[i, :n[i]].tolist()
            seq.tokens += new
            seq.logprobs += out_logprobs[i, :n[i]].tolist()
            seq.n_cached += len(new)  # every sampled token but the last was fed back in
            seq.n_new += len(new)
            if new and new[-1] in seq.stop:
                seq.finish_reason = "stop"
            elif seq.n_new >= seq.max_tokens or len(seq.tokens) >= self.max_len:
                seq.finish_reason = "length"
            if seq.done:
                finished.append(seq)
        self.running = [s for s in self.running if not s.done]
        return finished

    def generate(self, prompts, max_tokens=256, stop=(), sampling=Sampling()):
        """Continues each prompt (a list of token ids) and returns its Sequence."""
        seqs = [self.add(Sequence(p, max_tokens, stop)) for p in prompts]
        while self.has_work():
            for seq in self.step(sampling):
                self.release(seq)
        return seqs
