"""
Continued pretraining of the base world model (sai_ctx2) into the math world models (p_ff, p_ffpad2, p_ffc, the
weaker p_ff_* ladder): scripts/pretrain.sh has the exact commands. Data-parallel over all local devices: params
replicated, batch sharded on the 'data' axis.
    python pretrain.py --run_name=p_ff --init_from=sai_ctx2 --init_optimizer=True --train_mix=... (see the script)
With init_from, the replay sources continue where the init run left them (its checkpoint's data state); the released
replay shards start at exactly those positions, so they are read from the start (--continue_streams=False).
Resumes automatically from SAI_DIR/runs/<run_name>/ckpt.npz if it exists. On SIGTERM it saves a checkpoint and exits.
"""

import dataclasses
import math
import os
import signal
import sys
import threading
import time

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

import checkpoint
import model
import optim
from config import Config, parse_overrides
from data import Loader, prefetch, token_files
from tokenizer import Tokenizer


def train_step(params, opt_state, batch, step, cfg):
    """batch: (grad_accum, batch_size, seq_len + 1). Accumulates grads over the first axis, then updates."""
    grad_fn = jax.value_and_grad(lambda params, tokens: model.token_losses(params, tokens[:, :-1], tokens[:, 1:], cfg).mean())

    def micro_step(grads, tokens):
        loss, g = grad_fn(params, tokens)
        return jax.tree.map(jnp.add, grads, g), loss

    grads, losses = jax.lax.scan(micro_step, jax.tree.map(jnp.zeros_like, params), batch)
    grads = jax.tree.map(lambda g: g / batch.shape[0], grads)
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    params, opt_state = optim.update(params, grads, opt_state, step, cfg)
    return params, opt_state, dict(loss=losses.mean(), grad_norm=grad_norm, lrm=optim.lr_multiplier(step, cfg))


def eval_step(params, tokens, token_bytes, cfg):
    """Returns (sum of losses in nats, sum of target bytes), skipping special tokens (0 bytes)."""
    targets = tokens[:, 1:]
    losses = model.token_losses(params, tokens[:, :-1], targets, cfg)
    n_bytes = token_bytes[targets]
    return jnp.sum(losses * (n_bytes > 0)), jnp.sum(n_bytes)


def evaluate_bpb(eval_fn, params, token_bytes, bos_id, batch_size, sharding, cfg):
    """Bits per byte on the first cfg.eval_tokens tokens of cfg.val_source (the same rows at any batch size)."""
    val_batches = Loader(cfg, "val", batch_size, bos_id)  # a fresh loader yields the same batches every time
    nats, n_bytes = 0.0, 0
    for _ in range(max(1, cfg.eval_tokens // (batch_size * cfg.seq_len))):
        tokens = jax.device_put(next(val_batches)[0], sharding)
        batch_nats, batch_bytes = eval_fn(params, tokens, token_bytes)
        nats, n_bytes = nats + float(batch_nats), n_bytes + int(batch_bytes)
    return nats / (math.log(2) * n_bytes)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--continue_streams=")]  # (not a model setting: kept out of cfg)
    continue_streams = "--continue_streams=False" not in sys.argv[1:]
    cfg = Config(**parse_overrides(argv))
    print(cfg)

    n_devices = jax.device_count()
    batch_size = cfg.device_batch_size * n_devices  # sequences per micro-step, across devices
    assert cfg.total_batch_tokens % (batch_size * cfg.seq_len) == 0
    grad_accum = cfg.total_batch_tokens // (batch_size * cfg.seq_len)
    print(f"{n_devices} x {jax.devices()[0].device_kind} | grad accum {grad_accum} | {cfg.num_steps:,} steps")
    tok = Tokenizer.load(cfg.tokenizer)
    assert tok.vocab_size == cfg.vocab_size, f"tokenizer has {tok.vocab_size} tokens, config says {cfg.vocab_size}"
    token_bytes = jnp.asarray(tok.token_bytes())

    mesh = jax.make_mesh((n_devices,), ("data",), axis_types=(jax.sharding.AxisType.Auto,))
    replicated = NamedSharding(mesh, P())
    data_sharded = NamedSharding(mesh, P(None, "data"))  # (grad_accum, batch, seq)
    eval_sharded = NamedSharding(mesh, P("data"))  # (batch, seq)

    params = model.init_params(cfg, jax.random.key(cfg.seed))
    opt_state = optim.init(params)
    ckpt_path = checkpoint.run_path(cfg.run_name)
    start_step, data_state = 0, None
    if os.path.exists(ckpt_path):
        (params, opt_state), meta = checkpoint.load(ckpt_path, (params, opt_state))
        start_step, data_state = meta["step"], meta["data_state"]
        if start_step == cfg.num_steps:
            return print(f"{ckpt_path} is finished (step {start_step})")
        print(f"resumed from {ckpt_path} at step {start_step}")
    elif cfg.init_from:
        # params (and with init_optimizer, the Muon / Adam moments) from that run; the LR schedule and Muon momentum
        # warmup restart with this run's schedule
        init_path = checkpoint.run_path(cfg.init_from)
        (params, init_opt_state), meta = checkpoint.load(init_path, (params, opt_state))
        opt_state = init_opt_state if cfg.init_optimizer else opt_state
        if continue_streams:  # the data sources continue where that run left them
            data_state = dict(streams=meta["data_state"]["streams"], rng=None)
        print(f"initialized from {init_path} (step {meta['step']}) with {'its' if cfg.init_optimizer else 'a fresh'} "
              f"optimizer state")
    params, opt_state = jax.device_put((params, opt_state), replicated)

    step_fn = jax.jit(lambda *args: train_step(*args, cfg), donate_argnums=(0, 1))
    eval_fn = jax.jit(lambda *args: eval_step(*args, cfg))
    eval_every = cfg.eval_every if token_files(cfg.tokenizer, cfg.val_source, "val") else 0
    train_batches = prefetch(Loader(cfg, "train", grad_accum * batch_size, tok.bos_id, start_step, data_state))

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def save(step):
        meta = dict(step=step, data_state=data_state, config=dataclasses.asdict(cfg))
        checkpoint.save(ckpt_path, (params, opt_state), meta)
        print(f"saved checkpoint at step {step}")

    wandb = None
    if cfg.wandb_project:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, id=cfg.run_name, resume="allow",
                   config=dataclasses.asdict(cfg))
    t0, t0_step = time.time(), start_step

    for step in range(start_step, cfg.num_steps + 1):
        last_step = step == cfg.num_steps
        if eval_every > 0 and (step % eval_every == 0 or last_step):
            val_bpb = evaluate_bpb(eval_fn, params, token_bytes, tok.bos_id, batch_size, eval_sharded, cfg)
            print(f"step {step:6d} | val bpb {val_bpb:.4f}")
            if wandb:
                wandb.log({"val/bpb": val_bpb}, step=step)
            t0, t0_step = time.time(), step  # don't count eval time towards throughput
        if last_step or stop.is_set() or (step > start_step and step % cfg.save_every == 0):
            save(step)
        if last_step or stop.is_set():
            break

        tokens, data_state = next(train_batches)
        batch = jax.device_put(tokens.reshape(grad_accum, batch_size, -1), data_sharded)
        params, opt_state, metrics = step_fn(params, opt_state, batch, step)

        if step % cfg.log_every == 0:
            metrics = {k: float(v) for k, v in metrics.items()}  # blocks until the step is done
            dt = (time.time() - t0) / max(step + 1 - t0_step, 1)
            t0, t0_step = time.time(), step + 1
            metrics["tok_per_sec"] = cfg.total_batch_tokens / dt
            print(f"step {step:6d}/{cfg.num_steps} | " + " | ".join(f"{k} {v:.4g}" for k, v in metrics.items()))
            if wandb:
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)


if __name__ == "__main__":
    main()
