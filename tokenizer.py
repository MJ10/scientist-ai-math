"""
The world model's tokenizer: a 16k byte-level BPE (GPT-4 style, numbers split into chunks of at most 2 digits), encoded
with tiktoken. Its merges are in tokenizer/v2.tiktoken (base64 token, rank). The special tokens follow the merges; the
ones this project uses are <|bos|> and the contextualization tokens (records.py).
"""

import base64
import os

import numpy as np
import tiktoken

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer")

SPECIAL_TOKENS = [
    "<|bos|>",  # starts every document
    # chat / tool tokens of the original project (unused here, kept for their ids)
    "<|system_start|>", "<|system_end|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
    "<|tool_start|>", "<|tool_end|>", "<|output_start|>", "<|output_end|>",
    # contextualized documents (records.py): communication acts and factual records
    "<|act_start|>", "<|said|>", "<|act_end|>", "<|fact_start|>", "<|fact_end|>",
    "<|query_start|>", "<|query_end|>", "<|true|>", "<|false|>", "<|explain|>", "<|explain_end|>",
]
SPECIAL_TOKENS += [f"<|reserved_{i}|>" for i in range(32 - len(SPECIAL_TOKENS))]


class Tokenizer:
    def __init__(self, enc: tiktoken.Encoding):
        self.enc = enc
        self.bos_id = enc.encode_single_token("<|bos|>")
        self.vocab_size = enc.n_vocab

    @classmethod
    def load(cls, name="v2"):
        with open(os.path.join(HERE, f"{name}.tiktoken")) as f:
            ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in f if line.strip())}
        with open(os.path.join(HERE, f"{name}.pattern")) as f:
            pattern = f.read()
        specials = {s: len(ranks) + i for i, s in enumerate(SPECIAL_TOKENS)}
        return cls(tiktoken.Encoding(name, pat_str=pattern, mergeable_ranks=ranks, special_tokens=specials))

    def encode_docs(self, texts, num_threads=8):
        """Encodes a batch of documents, each prefixed with BOS; special token text becomes the special token."""
        return [[self.bos_id] + ids for ids in self.enc.encode_batch(texts, num_threads=num_threads, allowed_special="all")]

    def decode(self, ids):
        return self.enc.decode(ids)

    def token_bytes(self):
        """UTF-8 byte length of every token (0 for special tokens), for bits per byte."""
        special = {self.enc.encode_single_token(s) for s in self.enc.special_tokens_set}
        return np.array([0 if i in special else len(self.enc.decode_single_token_bytes(i))
                         for i in range(self.vocab_size)], dtype=np.int32)
