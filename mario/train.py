"""Behavior-cloning training: distill the search teacher into MarioPolicy.

Loss (evidence-based weighting — hard labels carry decisiveness, soft targets are
auxiliary fatal-masking; see CLAUDE.md notes):
    L = CE(hard, class-balanced)  +  0.3 * softCE(soft_targets)  +  0.3 * MSE(value/1e4)

MPS hygiene: float32, no torch.compile, no anomaly detection. After training, writes a
CPU-vs-MPS parity record so we never trust an MPS eval that silently diverges.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.buffer import DatasetIndex
from mario.env import N_ACTIONS
from mario.io import new_run_id, run_dir, write_json_atomic
from mario.policy import MarioPolicy, save_checkpoint

VALUE_SCALE = 10000.0
LN_A = float(np.log(N_ACTIONS))  # entropy normalizer for the current action set
DAGGER_WEIGHT = 4.0   # up-weight DAgger correction states (source==2) so they actually move the policy


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _val_trajectories(ds: DatasetIndex, frac=0.15, seed=0) -> set[int]:
    tids = np.unique(ds.data["trajectory_id"])
    tids = tids[tids != 0]  # never hold out the on-path backbone trajectory
    rng = np.random.default_rng(seed)
    rng.shuffle(tids)
    k = max(1, int(len(tids) * frac))
    return set(int(t) for t in tids[:k])


def _loss(net, batch, device):
    X = torch.from_numpy(batch["X"]).to(device)
    hard = torch.from_numpy(batch["hard"]).to(device)
    soft = torch.from_numpy(batch["soft"]).to(device)
    value = torch.from_numpy(batch["value"]).to(device).clamp(-VALUE_SCALE, 2 * VALUE_SCALE)
    weight = torch.from_numpy(batch["weight"]).to(device)
    src = torch.from_numpy(batch["source"]).to(device)
    weight = weight * torch.where(src == 2, DAGGER_WEIGHT, 1.0)  # emphasize corrections
    logits, value_pred = net(X)
    # NOTE (Jun-4): empirically TESTED the audit's distillation tweaks (drop class-balancing,
    # MSE->BCE value head). BOTH regressed 1-1 (net stalled at x1413, failed the pit jump):
    # the inv-frequency weighting is load-bearing — it's what lets the tiny net learn the
    # RARE-but-critical jump action; and the BCE value loss perturbed the policy via the
    # shared trunk. So we keep the proven loss. A meaningful value head for value-guided
    # search needs an INDEPENDENT/detached value net (Phase 2), not this shared-trunk head.
    ce = (F.cross_entropy(logits, hard, reduction="none") * weight).mean()
    # Soft targets are uniform on open ground (blurs decisiveness) and sharp at hazards.
    # Weight each sample's soft-KL by how INFORMATIVE it is: w = 1 - H(soft)/ln_A.
    H = -(soft * torch.log(soft + 1e-9)).sum(1)
    info_w = (1.0 - H / LN_A).clamp(min=0.0)
    soft_ce = (-(soft * F.log_softmax(logits, dim=1)).sum(1) * info_w).mean()
    vloss = F.mse_loss(value_pred, value / VALUE_SCALE)
    return ce + 0.3 * soft_ce + 0.3 * vloss, ce, soft_ce, vloss


def mps_parity(net, X32) -> dict:
    x = torch.from_numpy(X32)
    net_cpu = net.to("cpu").eval()
    with torch.no_grad():
        lc, _ = net_cpu(x)
    out = {"argmax_agreement": 1.0, "max_abs_logit_diff": 0.0, "mps_available": False}
    if torch.backends.mps.is_available():
        net_mps = net.to("mps").eval()
        with torch.no_grad():
            lm, _ = net_mps(x.to("mps"))
        lm = lm.cpu()
        out = {
            "mps_available": True,
            "argmax_agreement": float((lc.argmax(1) == lm.argmax(1)).float().mean()),
            "max_abs_logit_diff": float((lc - lm).abs().max()),
        }
    return out


def main() -> None:
    K = 4
    # Warm-start (DAgger fine-tuning): continue from a prior checkpoint with a lower LR and
    # fewer epochs so small correction sets actually adjust the policy without high-variance
    # retrain-from-scratch wiping out earlier-level competence.
    init_ckpt = os.environ.get("INIT_CHECKPOINT", "").strip()
    warm = bool(init_ckpt)
    epochs = 25 if warm else 40
    batch = 256
    lr = 1e-4 if warm else 3e-4
    device = pick_device()
    manifest = ROOT / "data" / "manifest.json"
    # LEVEL_FILTER="1-2" trains a per-level SPECIALIST (one MLP can't hold many levels
    # via this DAgger approach — multi-task interference). Unset = all levels.
    lf = os.environ.get("LEVEL_FILTER", "").strip()
    level_ids = None
    if lf:
        w, s = (int(x) for x in lf.split("-"))
        level_ids = {w * 10 + s}
    ds = DatasetIndex(manifest, K=K, level_ids=level_ids)
    print(f"level_filter={lf or 'ALL'}")
    val_tids = _val_trajectories(ds)
    train_mask, val_mask = ds.split(val_tids)
    print(f"device={device} samples={len(ds)} train={int(train_mask.sum())} "
          f"val={int(val_mask.sum())} K={K} warm_start={warm}")

    net = MarioPolicy(K=K).to(device)
    if warm:
        from mario.policy import load_policy
        src, _ = load_policy(init_ckpt, device=device)
        net.load_state_dict(src.state_dict())
        print(f"  warm-started from {init_ckpt}")
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    run_id = new_run_id("v2_bc_1_1")
    d = run_dir(run_id)
    curve_path = d / "curve.csv"
    best_val, best_state, patience, bad = 1e9, None, 15, 0
    with open(curve_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_ce", "val_ce", "val_acc"])
        for ep in range(epochs):
            net.train()
            tl = tce = nb = 0.0
            for b in ds.iter_batches(batch, mask=train_mask, shuffle=True, seed=ep):
                loss, ce, _sce, _vl = _loss(net, b, device)
                opt.zero_grad(); loss.backward(); opt.step()
                tl += float(loss); tce += float(ce); nb += 1
            sched.step()
            # val
            net.eval()
            vce = vacc = vn = 0.0
            with torch.no_grad():
                for b in ds.iter_batches(batch, mask=val_mask, shuffle=False):
                    logits, _ = net(torch.from_numpy(b["X"]).to(device))
                    hard = torch.from_numpy(b["hard"]).to(device)
                    vce += float(F.cross_entropy(logits, hard)) * len(hard)
                    vacc += float((logits.argmax(1) == hard).sum())
                    vn += len(hard)
            vce /= max(vn, 1); vacc /= max(vn, 1)
            w.writerow([ep, round(tl / nb, 4), round(tce / nb, 4), round(vce, 4), round(vacc, 4)])
            if ep % 5 == 0 or ep == epochs - 1:
                print(f" ep{ep:3d} train_loss={tl/nb:.3f} val_ce={vce:.3f} val_acc={vacc:.3f}")
            if vce < best_val - 1e-4:
                best_val, bad = vce, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    print(f" early stop at ep{ep}")
                    break

    if best_state is not None:
        net.load_state_dict(best_state)
    # fixed 32-row batch for parity
    X32 = ds.X[:32].astype(np.float32)
    parity = mps_parity(net, X32)
    write_json_atomic(ROOT / "bench" / "mps_parity.json", parity)
    net.to(device)

    val_metrics = {"val_ce": round(best_val, 4)}
    save_checkpoint(d / "checkpoint.pt", net, chunk_frames=ds.manifest.get("chunk_frames", 8),
                    train_cfg={"K": K, "epochs": epochs, "batch": batch, "lr": lr,
                               "n_params": n_params, "device": device},
                    val_metrics=val_metrics)
    print(f"saved {d/'checkpoint.pt'} | params={n_params} val_ce={best_val:.3f}")
    print(f"mps_parity: {parity}")
    print(f"RUN_ID={run_id}")


if __name__ == "__main__":
    main()
