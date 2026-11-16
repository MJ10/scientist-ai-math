"""Save / load a pytree of arrays plus a JSON-able metadata dict as a single .npz file, and load a saved model."""

import dataclasses
import json
import os

import jax
import numpy as np

import model
from config import BASE_DIR, saved_config


def save(path, tree, meta):
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    arrays = {jax.tree_util.keystr(k): np.asarray(v) for k, v in leaves}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, __meta__=json.dumps(meta), **arrays)
    os.replace(tmp, path)  # atomic: a job killed mid-save never leaves a corrupt checkpoint


def load(path, template):
    """Loads arrays into the structure of `template` (e.g. a freshly initialized state)."""
    data = np.load(path)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    tree = treedef.unflatten([data[jax.tree_util.keystr(k)] for k, _ in leaves])
    return tree, json.loads(str(data["__meta__"]))


def run_path(run_name):
    return os.path.join(BASE_DIR, "runs", run_name, "ckpt.npz")


def load_model(path, **overrides):
    """(params on the default device, cfg, meta) of a pretrain.py / sai.py checkpoint, without its optimizer state.
    overrides replace config fields, e.g. eval-time settings (dtype, global_attn_temp)."""
    data = np.load(path)
    meta = json.loads(str(data["__meta__"]))
    cfg = dataclasses.replace(saved_config(meta["config"]), **overrides)
    template = jax.eval_shape(lambda: model.init_params(cfg, jax.random.key(0)))
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    params = treedef.unflatten([jax.device_put(data["[0]" + jax.tree_util.keystr(k)]) for k, _ in leaves])
    return params, cfg, meta
