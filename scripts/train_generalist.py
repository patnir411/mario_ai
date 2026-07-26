"""Train the entity-transformer generalist on a TRAIN-levels allowlist (held-out integrity).

Decisive-experiment trainer for the data-diversity hypothesis: train on N levels, never let
the held-out levels enter the dataset. Loss = the proven distillation objective
(hard-CE + info-weighted soft-CE + value-MSE) plus an input-perturbation
consistency term. Unlike the
production MLP we do NOT apply global inverse-frequency class weights — V4 found that
collapses the transformer; the soft/value procedure-cloning targets carry the rare-action
(jump) signal instead. Recovery/DAgger states are up-weighted by source.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/train_generalist.py \
        --config configs/heldout_split.json [--init CKPT] [--epochs N] [--out CKPT]

Warm-start (--init) uses fewer epochs + lower LR for DAgger rounds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from mario.entity import OBS_DIM_ENTITY
from mario.entity_policy import (
    EntityTransformer,
    load_policy_checkpoint,
    mps_parity_ok,
    n_params,
    save_policy,
)
from mario.actions import SMB1_N_ACTIONS as N_ACTIONS
from mario.consistency import input_consistency_penalty
from mario.io import env_fingerprint, git_rev, utc_now_iso

SHARDS = ROOT / "data" / "entity_shards"
VALUE_SCALE = 10000.0
LN_A = float(np.log(N_ACTIONS))
SRC_WEIGHT = {0: 1.0, 1: 1.5, 2: 4.0}   # onpath / recovery / dagger-correction
CONSISTENCY_W = 0.05
CONSISTENCY_SIGMA = 0.05


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def file_ref(path: str | Path) -> dict:
    p = Path(path)
    try:
        display = str(p.resolve().relative_to(ROOT))
    except ValueError:
        display = str(p)
    return {
        "path": display,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "bytes": p.stat().st_size,
    }


def load_levels(levels: list[str], holdout: set[str]) -> dict:
    """Load entity shards for `levels`, asserting none are in the holdout set."""
    obs, hard, soft, value, lid, src, traj = [], [], [], [], [], [], []
    loaded, shards = [], {}
    for lv in levels:
        assert lv not in holdout, f"INTEGRITY VIOLATION: {lv} is held out but in train set"
        p = SHARDS / f"level-{lv}.npz"
        if not p.exists():
            print(f"  WARN: no shard for {lv}; skipping"); continue
        d = np.load(p)
        if len(d["obs"]) == 0:
            print(f"  WARN: empty shard for {lv}; skipping"); continue
        obs.append(d["obs"]); hard.append(d["hard_action"]); soft.append(d["soft_targets"])
        value.append(d["value"]); lid.append(d["level_id"]); src.append(d["source"])
        # make traj ids globally unique across levels (offset by level id)
        traj.append(d["traj"].astype(np.int64) + d["level_id"][0] * 1_000_000)
        loaded.append(lv)
        shards[lv] = file_ref(p)
    print(f"loaded {len(loaded)} levels: {', '.join(loaded)}")
    if not loaded:
        raise FileNotFoundError(
            f"no usable entity shards found under {SHARDS}; "
            "run scripts/gen_entity_dataset.py for at least one configured train level")
    return {
        "obs": np.concatenate(obs), "hard": np.concatenate(hard),
        "soft": np.concatenate(soft), "value": np.concatenate(value),
        "lid": np.concatenate(lid), "src": np.concatenate(src),
        "traj": np.concatenate(traj), "levels": loaded, "shards": shards,
    }


def make_val_mask(data: dict, seed: int = 0) -> np.ndarray:
    """Hold out 15% of recovery trajectories + an on-path stride, per the V4 protocol."""
    src, traj = data["src"], data["traj"]
    rng = np.random.default_rng(seed)
    val = np.zeros(len(src), bool)
    rtids = np.unique(traj[src == 1]); rng.shuffle(rtids)
    val_tids = set(rtids[: max(1, int(0.15 * len(rtids)))].tolist())
    for i in range(len(src)):
        if src[i] == 1 and traj[i] in val_tids:
            val[i] = True
        elif src[i] == 0 and i % 7 == 0:
            val[i] = True
    return val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", default="")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data" / "generalist.pt"))
    ap.add_argument("--d-model", type=int, default=160)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text())
    train_levels, holdout = cfg["train_levels"], set(cfg["holdout_levels"])
    overlap = sorted(set(train_levels) & holdout)
    if overlap:
        raise SystemExit(f"train/holdout overlap in config: {overlap}")
    warm = bool(args.init)
    epochs = args.epochs or (25 if warm else 40)
    lr = 1e-4 if warm else 3e-4
    dev = pick_device()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_levels(train_levels, holdout)
    X = torch.tensor(data["obs"])
    hard = torch.tensor(data["hard"]); soft = torch.tensor(data["soft"])
    value = torch.tensor(data["value"]).clamp(-VALUE_SCALE, 2 * VALUE_SCALE)
    w = torch.tensor([SRC_WEIGHT[int(s)] for s in data["src"]], dtype=torch.float32)
    val = make_val_mask(data, seed=args.seed); tr = ~val

    net = EntityTransformer(d_model=args.d_model, layers=args.layers)
    init_ref = None
    if warm:
        net, init_meta = load_policy_checkpoint(args.init, device="cpu")
        if sorted(init_meta.get("train_levels", [])) != sorted(train_levels) \
                or sorted(init_meta.get("holdout_levels", [])) != sorted(holdout):
            raise SystemExit(
                "initial checkpoint train/holdout metadata does not match --config")
        init_ref = file_ref(args.init)
    net = net.to(dev)
    print(f"device={dev} params={n_params(net)} N={len(X)} train={int(tr.sum())} "
          f"val={int(val.sum())} epochs={epochs} warm={warm}", flush=True)

    Xtr, Xv = X[tr].to(dev), X[val].to(dev)
    htr, hv = hard[tr].to(dev), hard[val].to(dev)
    str_, sv = soft[tr].to(dev), soft[val].to(dev)
    vtr = value[tr].to(dev); wtr = w[tr].to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    idx = np.arange(int(tr.sum())); rng = np.random.default_rng(args.seed); best = 0.0

    for ep in range(epochs):
        net.train(); rng.shuffle(idx)
        for st in range(0, len(idx), 256):
            b = idx[st:st + 256]
            xb = Xtr[b]
            logits, vpred = net(xb)
            ce = (F.cross_entropy(logits, htr[b], reduction="none") * wtr[b]).mean()
            H = -(str_[b] * torch.log(str_[b] + 1e-9)).sum(1)
            info_w = (1.0 - H / LN_A).clamp(min=0.0)
            soft_ce = (-(str_[b] * F.log_softmax(logits, dim=1)).sum(1) * info_w).mean()
            vloss = F.mse_loss(vpred, vtr[b] / VALUE_SCALE)
            consistency = input_consistency_penalty(
                logits, net, xb, sigma=CONSISTENCY_SIGMA)
            loss = (ce + 0.3 * soft_ce + 0.3 * vloss
                    + CONSISTENCY_W * consistency)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = float((net.logits(Xv).argmax(1) == hv).float().mean()) if val.sum() else 0.0
        best = max(best, acc)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep {ep} val_acc={acc:.3f} (loss={float(loss):.3f} ce={float(ce):.3f} "
                  f"soft={float(soft_ce):.3f} v={float(vloss):.4f} "
                  f"consistency={float(consistency):.4f})", flush=True)

    parity = mps_parity_ok(net, data["obs"][:32])
    save_policy(net, args.out, extra={
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "git_rev": git_rev(),
        "environment": env_fingerprint(),
        "seed": args.seed,
        "train_levels": data["levels"],
        "holdout_levels": sorted(holdout),
        "config": file_ref(config_path),
        "shards": data["shards"],
        "initial_checkpoint": init_ref,
        "hyperparameters": {
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": 256,
            "d_model": net.cfg.get("d_model"),
            "layers": net.cfg.get("layers"),
            "consistency_weight": CONSISTENCY_W,
            "consistency_sigma": CONSISTENCY_SIGMA,
            "value_scale": VALUE_SCALE,
        },
        "best_val_acc": best,
        "mps_parity": parity,
    })
    print(f"saved {args.out} best_val_acc={best:.3f} mps_parity={parity}", flush=True)


if __name__ == "__main__":
    main()
