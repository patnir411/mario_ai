"""Train the independent value net V(s) = P(reach-flag) on data/value_data.npz (BCE).

    ./venv/bin/python scripts/train_value.py
Saves runs/value/value.pt + reports held-out accuracy/AUC. Class-weighted BCE (pos:neg ~2.5:1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mario.value import ValueNet, save_value


def auc(scores, labels):
    """ROC-AUC via the rank statistic (Mann-Whitney U)."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    z = np.load(ROOT / "data" / "value_data.npz")
    obs, lab = z["obs"].astype(np.float32), z["label"].astype(np.float32)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(lab))
    obs, lab = obs[idx], lab[idx]
    n_val = max(1, int(0.15 * len(lab)))
    Xtr, ytr = obs[n_val:], lab[n_val:]
    Xva, yva = obs[:n_val], lab[:n_val]
    pos_w = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))   # up-weight rare positives? no:
    pos_w = float((ytr == 0).sum()) / max(1.0, float((ytr == 1).sum()))
    print(f"device={device} N={len(lab)} train={len(ytr)} val={len(yva)} pos_weight={pos_w:.2f}")

    net = ValueNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    Xtr_t = torch.from_numpy(Xtr).to(device); ytr_t = torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device); yva_t = torch.from_numpy(yva).to(device)
    pw = torch.tensor(pos_w, device=device)
    best_auc, best_state = 0.0, None
    bs = 256
    for ep in range(60):
        net.train()
        perm = torch.randperm(len(ytr_t), device=device)
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            logit = net(Xtr_t[b])
            loss = F.binary_cross_entropy_with_logits(logit, ytr_t[b], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vs = torch.sigmoid(net(Xva_t)).cpu().numpy()
        a = auc(vs, yva)
        acc = float(((vs > 0.5) == (yva > 0.5)).mean())
        if a > best_auc:
            best_auc = a
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if ep % 10 == 0 or ep == 59:
            print(f" ep{ep:3d} val_auc={a:.3f} val_acc={acc:.3f}", flush=True)
    if best_state:
        net.load_state_dict(best_state)
    out = ROOT / "runs" / "value"; out.mkdir(parents=True, exist_ok=True)
    save_value(out / "value.pt", net.to("cpu"), {"val_auc": round(best_auc, 4)})
    print(f"saved {out/'value.pt'} best_val_auc={best_auc:.3f}")


if __name__ == "__main__":
    main()
