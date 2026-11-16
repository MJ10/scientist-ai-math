"""
Model and training hyperparameters (pretrain.py; sai.py copies a world model's config and changes a few fields).
Any field can be overridden from the command line: python pretrain.py --run_name=p_ff --num_tokens=500_000_000
Everything on disk lives under SAI_DIR (default ./work):
    SAI_DIR/runs/<run_name>/ckpt.npz             checkpoints (the released ones go here too)
    SAI_DIR/tokens/v2/<source>/{train,val}/*.bin  token streams for continued pretraining (data.py)
"""

import dataclasses
import os
import sys
from dataclasses import dataclass

BASE_DIR = os.path.abspath(os.environ.get("SAI_DIR", "work"))


@dataclass(frozen=True)
class Config:
    # model (~48M params total = 31.5M transformer + 2 x 8.4M untied embeddings)
    vocab_size: int = 16384  # must match the tokenizer and be a multiple of 64
    n_layer: int = 10
    d_model: int = 512
    head_dim: int = 128
    seq_len: int = 2048
    window_pattern: str = "SSSL"  # tiled over layers, L=global (full context), S=local; last layer is always L
    local_window: int = 512  # sliding window of the S layers, in tokens
    rope_base: float = 10_000.0  # RoPE of the local layers; the global layers have no positional encoding (NoPE)
    global_attn_temp: float = 0.0  # >0 (beyond the training length): global-layer queries at position p > L are
    global_attn_temp_len: int = 0  # scaled by 1 + t*log(p/L), where L is this reference length, or seq_len if 0
    logit_softcap: float = 15.0
    dtype: str = "bfloat16"  # compute dtype of activations and matmuls; float32 for exact inference / RL (model.py)

    # batch / duration
    total_batch_tokens: int = 2**18  # tokens per optimizer step (across all devices + grad accumulation)
    device_batch_size: int = 32  # sequences per device per micro-step: 32 fits an A100-80GB
    num_tokens: int = 500_000_000  # total training tokens

    # optimizer (Muon + AdamW, optim.py)
    matrix_lr: float = 0.02  # Muon, for all hidden 2D weights
    embedding_lr: float = 0.3  # AdamW, wte
    unembedding_lr: float = 0.008  # AdamW, lm_head
    scalar_lr: float = 0.5  # AdamW, per-layer lambdas
    weight_decay: float = 0.1  # Muon cautious weight decay, cosine-decayed to 0 over training
    lr_scale: float = 1.0  # multiplies all LRs
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65  # fraction of training spent linearly decaying the LR
    final_lr_frac: float = 0.05

    # data: a mix is "source:weight,..." of documents over SAI_DIR/tokens/<tokenizer>/<source> (data.doc_mix converts
    # token shares into document weights)
    tokenizer: str = "v2"
    train_mix: str = ""
    anneal_mix: str = ""  # mix used for the final anneal_frac of training ("" = keep train_mix)
    anneal_frac: float = 0.0
    val_source: str = "ff_chains"
    long_doc_rows: bool = False

    # eval / logging / checkpointing
    eval_every: int = 250  # validation bits per byte
    eval_tokens: int = 10 * 2**20
    log_every: int = 10
    save_every: int = 500
    run_name: str = "default"  # checkpoints go to SAI_DIR/runs/run_name; rerunning a name resumes it
    init_from: str = ""  # start from the params of this run
    init_optimizer: bool = False  # with init_from: also load its optimizer state (Muon / Adam moments)
    wandb_project: str = ""  # empty = no wandb
    seed: int = 0

    @property
    def out_dir(self) -> str:
        return os.path.join(BASE_DIR, "runs", self.run_name)

    @property
    def num_steps(self) -> int:
        return self.num_tokens // self.total_batch_tokens


def parse_overrides(argv) -> dict:
    """{field: value} from --key=value args, casting each value to the field's type."""
    overrides = {}
    fields = {f.name: f for f in dataclasses.fields(Config)}
    for arg in argv:
        assert arg.startswith("--") and "=" in arg, f"expected --key=value, got {arg!r}"
        key, value = arg[2:].split("=", 1)
        assert key in fields, f"unknown config key {key!r}"
        typ = type(fields[key].default)
        if typ is bool:
            assert value in ("True", "False"), f"expected True or False for {key}, got {value!r}"
            overrides[key] = value == "True"
        else:
            overrides[key] = typ(value.replace("_", "")) if typ in (int, float) else typ(value)
    return overrides


def parse_args(argv=None) -> Config:
    return Config(**parse_overrides(sys.argv[1:] if argv is None else argv))


def saved_config(stored: dict) -> Config:
    """Config of a saved run (fields this code doesn't have, e.g. the original project's eval settings, are dropped)."""
    fields = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in stored.items() if k in fields})
