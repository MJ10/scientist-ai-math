"""
Continued-pretraining data: generate the computed sources as token streams, and a loader that packs documents into
training rows.

On-disk layout (under SAI_DIR):
    tokens/<tokenizer>/<source>/{train,val}/*.bin       uint16 token streams in which every document starts with BOS

The computed sources (SOURCES, from facts.py) are generated here, deterministically:
    python data.py ff_chains ff_verdicts ff_grader_logs      # p_ff's new data (~0.4B tokens, CPU minutes)
The replay sources (fineweb_edu_ctx, wikipedia_ctx, python_ctx: contextualized web text from the original pretraining)
are not generated here; download them with scripts/download.sh. Every VAL_EVERY-th document goes to val/0000.bin, the
others round robin to train/0000.bin ... train/<num_shards - 1>.bin (the original pipeline's split, so the same token
streams).

Packing follows nanochat's BOS-aligned best-fit: every row starts at a document boundary, documents are placed
largest-that-fits-first from a buffer, and when nothing fits, the shortest one is cropped to fill the row exactly.
"""

import argparse
import functools
import glob
import itertools
import os
import queue
import threading
from collections import deque

import numpy as np

import facts
from config import BASE_DIR

VAL_EVERY = 100
# name -> (documents, max characters): the settings of the released world models' data
SOURCES = {
    "facts": (facts.fact_docs, 1.2e9),
    "fact_chains": (facts.chain_docs, 2e9),
    "ff_chains": (functools.partial(facts.chain_docs, kinds=facts.FREE_FORM, seed=13), 1.2e9),
    "ff_verdicts": (functools.partial(facts.verdict_docs, kinds=facts.FREE_FORM, seed=14), 4e8),
    "ff_grader_logs": (functools.partial(facts.grader_logs, kinds=facts.FREE_FORM, n_attempts=200_000, seed=15), float("inf")),
    "ffp_chains": (functools.partial(facts.chain_docs, kinds=facts.FREE_FORM_PAD, seed=16), 1.2e9),
    "ffp_verdicts": (functools.partial(facts.verdict_docs, kinds=facts.FREE_FORM_PAD, seed=17), 4e8),
    "ffp_grader_logs": (functools.partial(facts.grader_logs, kinds=facts.FREE_FORM_PAD, n_attempts=200_000, seed=18), float("inf")),
    "ffc_chains": (functools.partial(facts.chain_docs, kinds=facts.FREE_FORM_COIN, seed=19), 1.2e9),
    "ffc_verdicts": (functools.partial(facts.verdict_docs, kinds=facts.FREE_FORM_COIN, seed=20), 4e8),
    "ffc_grader_logs": (functools.partial(facts.grader_logs, kinds=facts.FREE_FORM_COIN, n_attempts=200_000, seed=21), float("inf")),
}


def generate(name, tokenizer="v2", num_shards=8, max_val_tokens=50_000_000, max_chars=None):
    """Writes a computed source's documents as token streams (see the module docstring), stopping after the batch of
    1000 * num_shards documents in which max_chars characters are reached."""
    from tokenizer import Tokenizer
    tok = Tokenizer.load(tokenizer)
    docs, default_chars = SOURCES[name]
    max_chars = default_chars if max_chars is None else max_chars
    root = os.path.join(BASE_DIR, "tokens", tokenizer, name)
    paths = [os.path.join(root, "val", "0000.bin")] + [os.path.join(root, "train", f"{i:04d}.bin") for i in range(num_shards)]
    if all(os.path.exists(p) for p in paths):
        return print(f"{name} is already generated")
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    files = [open(p + ".tmp", "wb") for p in paths]
    n_docs, n_chars, val_tokens, val_buffer = 0, 0, 0, []
    for batch in itertools.batched(docs(), 1000 * num_shards):
        dest = [0 if i % VAL_EVERY == 0 else 1 + i % num_shards for i in range(n_docs, n_docs + len(batch))]
        encoded = tok.encode_docs(list(batch))
        for j, f in enumerate(files[1:], 1):
            ids = [t for d, k in zip(encoded, dest) for t in d if k == j]
            f.write(np.array(ids, np.uint16).tobytes())
        val_buffer += [d for d, k in zip(encoded, dest) if k == 0]
        while len(val_buffer) >= 1024 and val_tokens < max_val_tokens:  # (val: whole batches of 1024 documents)
            ids = np.fromiter(itertools.chain.from_iterable(val_buffer[:1024]), np.uint16)
            files[0].write(ids.tobytes())
            val_tokens, val_buffer = val_tokens + len(ids), val_buffer[1024:]
        n_docs, n_chars = n_docs + len(batch), n_chars + sum(map(len, batch))
        print(f"{name}: {n_docs:,} documents, {n_chars / 1e6:.0f}M characters", flush=True)
        if n_chars >= max_chars:
            break
    if val_buffer and val_tokens < max_val_tokens:
        files[0].write(np.fromiter(itertools.chain.from_iterable(val_buffer), np.uint16).tobytes())
    for f, p in zip(files, paths):
        f.close()
        os.replace(p + ".tmp", p)


def token_files(tokenizer, source, split):
    return sorted(glob.glob(os.path.join(BASE_DIR, "tokens", tokenizer, source, split, "*.bin")))


def parse_mix(mix):
    """'climbmix:0.9,finemath:0.1' -> {'climbmix': 0.9, 'finemath': 0.1}"""
    return {name: float(weight) for name, weight in (item.split(":") for item in mix.split(","))}


def doc_mix(token_shares, tokenizer="v2", sample_tokens=50_000_000):
    """'a:0.8,b:0.2' shares of the tokens -> the mix of documents (what Loader samples) that gives them, from each
    source's mean document length (over the start of its first train file)."""
    from tokenizer import Tokenizer
    bos = Tokenizer.load(tokenizer).bos_id
    weights = {}
    for name, share in parse_mix(token_shares).items():
        tokens = np.memmap(token_files(tokenizer, name, "train")[0], dtype=np.uint16, mode="r")[:sample_tokens]
        weights[name] = share * np.count_nonzero(tokens == bos) / len(tokens)  # share / mean length
    total = sum(weights.values())
    return ",".join(f"{name}:{w / total:.5f}" for name, w in weights.items())


class DocStream:
    """The documents of one source in order, cycling over its .bin files forever.
    Its position (file index, token offset) is saved in checkpoints; on resume we skip at most one chunk."""

    def __init__(self, files, bos_id, file_idx=0, offset=0, chunk_tokens=2**22):
        assert files, "no token files found: generate them with data.py (or download the replay sources)"
        self.files, self.bos_id, self.chunk_tokens = files, bos_id, chunk_tokens
        self.file_idx, self.offset = file_idx, offset
        self.docs = deque()

    def state(self):
        return [self.file_idx, self.offset]

    def next_doc(self):
        while not self.docs:
            self._read_chunk()
        return self.docs.popleft()

    def _read_chunk(self):
        tokens = np.memmap(self.files[self.file_idx], dtype=np.uint16, mode="r")
        chunk = tokens[self.offset:self.offset + self.chunk_tokens]
        starts = np.flatnonzero(chunk == self.bos_id)
        if self.offset + len(chunk) < len(tokens) and len(starts) > 1:
            chunk, starts = chunk[:starts[-1]], starts[:-1]  # stop before the last (possibly incomplete) document
        self.docs.extend(np.split(chunk, starts[1:]))
        self.offset += len(chunk)
        if self.offset >= len(tokens):
            self.file_idx, self.offset = (self.file_idx + 1) % len(self.files), 0


class Loader:
    """Yields (tokens, state): tokens is int32 (rows, seq_len + 1); state can be passed back in to resume.
    Train batches use cfg.train_mix, switching to cfg.anneal_mix for the final cfg.anneal_frac of training.
    Val batches come from cfg.val_source and are identical every time a fresh val Loader is made."""

    def __init__(self, cfg, split, rows, bos_id, start_step=0, state=None, buffer_size=1000):
        self.rows, self.row_len, self.buffer_size = rows, cfg.seq_len + 1, buffer_size
        self.long_doc_rows = cfg.long_doc_rows and split == "train"
        if split == "train":
            self.mixes = [parse_mix(cfg.train_mix), parse_mix(cfg.anneal_mix or cfg.train_mix)]
            self.anneal_step = int((1 - cfg.anneal_frac) * cfg.num_steps)
        else:
            self.mixes, self.anneal_step = [{cfg.val_source: 1.0}], float("inf")
        state = state or dict(streams={}, rng=None)
        names = {name for mix in self.mixes for name in mix}
        self.streams = {name: DocStream(token_files(cfg.tokenizer, name, split), bos_id,
                                        *state["streams"].get(name, (0, 0))) for name in sorted(names)}
        self.rng = np.random.default_rng(cfg.seed)
        if state["rng"]:
            self.rng.bit_generator.state = state["rng"]
        self.step = start_step
        self.docs, self.lens, self.sources = [], [], []  # buffer of documents to pack from

    def state(self):
        return dict(streams={name: s.state() for name, s in self.streams.items()}, rng=self.rng.bit_generator.state)

    def __iter__(self):
        return self

    def next_doc(self, mix, pos):
        """The next document to place in a row filled up to pos (removed from the buffer), with its source."""
        names, weights = list(mix), np.array(list(mix.values()))
        while len(self.docs) < self.buffer_size:
            name = names[self.rng.choice(len(names), p=weights / weights.sum())]
            doc = self.streams[name].next_doc()
            self.docs.append(doc)
            self.lens.append(len(doc))
            self.sources.append(name)
        remaining = self.row_len - pos
        lens = np.array(self.lens)
        fits = np.where(lens <= remaining, lens, 0)
        if pos == 0 and self.long_doc_rows and lens.max() > remaining:
            i = int(np.argmax(lens))  # a whole row of the start of the longest document
        elif fits.max() > 0:
            i = int(np.argmax(fits))
        else:
            i = int(np.argmin(lens))  # crop the shortest
        self.lens.pop(i)
        return self.docs.pop(i), self.sources.pop(i)

    def __next__(self):
        mix = self.mixes[-1] if self.step >= self.anneal_step else self.mixes[0]
        out = np.empty((self.rows, self.row_len), dtype=np.int32)
        for row in out:
            pos = 0
            while pos < self.row_len:
                doc, _ = self.next_doc(mix, pos)
                n = min(len(doc), self.row_len - pos)
                row[pos:pos + n] = doc[:n]
                pos += n
        self.step += 1
        return out, self.state()


def prefetch(iterator, size=4):
    """Runs an iterator in a background thread, keeping up to `size` items ready."""
    q = queue.Queue(maxsize=size)

    def worker():
        try:
            for item in iterator:
                q.put(item)
        except Exception as e:
            q.put(e)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if isinstance(item, Exception):
            raise item
        yield item


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", choices=list(SOURCES))
    parser.add_argument("--max_chars", type=float, default=None, help="override a source's size (smoke tests)")
    args = parser.parse_args()
    for name in args.sources:
        generate(name, max_chars=args.max_chars)
