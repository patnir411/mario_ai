"""V4 experiment: OBJECT-CENTRIC entity schema + a small entity-TRANSFORMER policy, distilled
from the cached cf8 search solutions (+ perturb-and-recover), trained locally on the M2.
Tests whether attention over a structured {player, enemies, terrain} token set beats the flat
MLP's 0/11. Self-contained (does NOT touch the production obs/policy/dataset).

Stages (argv[1]): sanity | data | train | eval | all

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/exp_entity.py sanity
"""
from __future__ import annotations
import os, sys, json, time, math, random
import multiprocessing as mp
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from mario import ram as R
from mario.ram import mario_level_x, signed
from mario.observation import tile_grid, gap_ahead, GRID_H, GRID_W, LEFT, UP
from mario.env import MarioSim, N_ACTIONS

# ---- entity schema ----------------------------------------------------------
N_ENT, N_TERR = 5, 8
N_TOK = 1 + N_ENT + N_TERR          # player + 5 enemies + 8 terrain columns = 14
D_TOK = 17
OBS_DIM_ENTITY = N_TOK * D_TOK      # 238
# token feature layout
P_IS, E_IS, T_IS = 0, 1, 2          # token-type one-hot
DX, DY, VX, VY = 3, 4, 5, 6
ETYPE, GROUND = 7, 8
ON_GND, JUMP, ACTIVE, PIT = 9, 10, 11, 12
POW, FACE, GAP, XSUB = 13, 14, 15, 16
ENT_X_PAGE = 0x006E   # ENEMY_X_LEVEL.start
ENT_X_SCR = 0x0087    # enemy x within screen


def entity_obs(ram, info=None) -> np.ndarray:
    toks = np.zeros((N_TOK, D_TOK), np.float32)
    mx = mario_level_x(ram); my = int(ram[R.MARIO_Y_ON_SCREEN])
    grid = tile_grid(ram)
    p = toks[0]
    p[P_IS] = 1.0
    p[VX] = np.clip(signed(int(ram[R.MARIO_X_SPEED])) / 4.0, -4, 4)
    p[VY] = np.clip(signed(int(ram[R.MARIO_Y_VELOCITY])) / 4.0, -4, 4)
    p[ON_GND] = 1.0 if int(ram[R.PLAYER_FLOAT_STATE]) == 0 else 0.0
    p[JUMP] = 1.0 if int(ram[R.PLAYER_FLOAT_STATE]) == 1 else 0.0
    p[POW] = min(int(ram[R.POWERUP_STATE]), 2) / 2.0
    p[FACE] = 1.0 if int(ram[R.FACING_DIR]) == 1 else 0.0
    p[GAP] = float(gap_ahead(grid))
    p[XSUB] = (int(ram[R.MARIO_X_ON_SCREEN]) % 16) / 16.0
    for i in range(N_ENT):
        if int(ram[0x000F + i]) == 0:      # ENEMY_ACTIVE
            continue
        ex = int(ram[ENT_X_PAGE + i]) * 256 + int(ram[ENT_X_SCR + i])
        ey = int(ram[0x00CF + i])          # ENEMY_Y_ON_SCREEN
        t = toks[1 + i]
        t[E_IS] = 1.0; t[ACTIVE] = 1.0
        t[DX] = np.clip((ex - mx) / 48.0, -4, 4)
        t[DY] = np.clip((ey - my) / 48.0, -4, 4)
        t[ETYPE] = int(ram[0x0016 + i]) / 48.0   # ENEMY_TYPE
    for j in range(N_TERR):
        col = LEFT + 1 + j
        t = toks[1 + N_ENT + j]
        t[T_IS] = 1.0
        t[DX] = (j + 1) / N_TERR
        if col < GRID_W:
            below = np.where(grid[UP:, col] == 1)[0]   # solid rows at/below Mario
            if len(below):
                t[GROUND] = below[0] / (GRID_H - UP)
            else:
                t[PIT] = 1.0; t[GROUND] = 1.0
    return toks.reshape(-1)


# ---- stage: sanity ----------------------------------------------------------
def sanity():
    import json as J
    sol = J.loads((ROOT / "data/solutions/1-1.json").read_text())["path"]
    sim = MarioSim(1, 1); sim.reset(0)
    print(f"OBS_DIM_ENTITY={OBS_DIM_ENTITY} N_TOK={N_TOK} D_TOK={D_TOK}")
    for k, a in enumerate(sol[:60]):
        sim.run_chunk(a, 8)
        if k in (10, 25, 40):
            o = entity_obs(sim.ram).reshape(N_TOK, D_TOK)
            r = sim.ram
            ent = [(round(float(o[1+i][DX]), 2), round(float(o[1+i][DY]), 2), round(float(o[1+i][ETYPE]), 3))
                   for i in range(N_ENT) if o[1+i][ACTIVE] > 0]
            terr = [round(float(o[6+j][GROUND]), 2) for j in range(N_TERR)]
            print(f"  step {k}: x={mario_level_x(r)} vx={o[0][VX]:.2f} on_gnd={o[0][ON_GND]:.0f} "
                  f"gap={o[0][GAP]:.2f} | enemies(dx,dy,type)={ent} | terrain_ground={terr}")
    sim.close()


# ---- stage: data (distill cf8 solutions + perturb-and-recover) --------------
_ALL = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (8, 3)]
_LV = os.environ.get("EXP_LEVELS", "").strip()   # e.g. "1-1" or "1-1,8-2" for focused runs
LEVELS = [tuple(int(x) for x in p.split("-")) for p in _LV.split(",")] if _LV else _ALL
_TAG = os.environ.get("EXP_TAG", "")             # separate artifacts for focused runs
DS_PATH = ROOT / "data" / f"entity_ds{_TAG}.npz"
ANCHOR_EVERY, N_PERTURB = 3, 2
RECOVER_KEEP, RECOVER_BEAM, RECOVER_DEPTH = 8, 16, 48
PERTURB_PREF = [3, 4, 1, 2, 5]


def _recover_worker(spec):
    from mario.search import beam_search
    w, s, prefix, tid = spec
    res = beam_search(w, s, beam_width=RECOVER_BEAM, max_depth=RECOVER_DEPTH, seed=0, start_prefix=prefix)
    return (tid, prefix, res.path)


def build_data():
    from mario.reward import is_success
    sol = {ws: json.loads((ROOT / f"data/solutions/{ws[0]}-{ws[1]}.json").read_text())["path"] for ws in LEVELS}
    # 1) recovery searches (parallel)
    specs, tid = [], 1
    for (w, s) in LEVELS:
        P = sol[(w, s)]
        for t in range(0, len(P), ANCHOR_EVERY):
            for pert in [a for a in PERTURB_PREF if a != P[t]][:N_PERTURB]:
                specs.append((w, s, P[:t] + [pert], tid)); tid += 1
    print(f"[data] {len(specs)} recovery anchors across {len(LEVELS)} levels ...", flush=True)
    nproc = max(1, min(11, (os.cpu_count() or 4) - 1))
    with mp.Pool(nproc) as pool:
        recs = pool.map(_recover_worker, specs)
    rec_by = {tid_: (pre, cont) for (tid_, pre, cont) in recs}

    # 2) extract entity_obs + action for on-path + recovery states (replay; serial, fast)
    obs, act, lid, src, traj = [], [], [], [], []
    def lvlid(w, s): return w * 10 + s
    for (w, s) in LEVELS:
        P = sol[(w, s)]
        sim = MarioSim(w, s); sim.reset(0)
        for t, a in enumerate(P):
            obs.append(entity_obs(sim.ram)); act.append(int(a)); lid.append(lvlid(w, s)); src.append(0); traj.append(0)
            if sim.run_chunk(a, 8)[1]:
                break
        sim.close()
    # recovery branches
    by_spec = {(spec[0], spec[1], tuple(spec[2])): spec[3] for spec in specs}
    for (w, s, prefix, tid_) in specs:
        cont = rec_by[tid_][1]
        if not cont:
            continue
        sim = MarioSim(w, s); sim.reset(0)
        dead = False
        for a in prefix:
            if sim.run_chunk(a, 8)[1]:
                dead = True; break
        if not dead:
            for j in range(min(RECOVER_KEEP, len(cont))):
                obs.append(entity_obs(sim.ram)); act.append(int(cont[j])); lid.append(w * 10 + s); src.append(1); traj.append(tid_)
                if sim.run_chunk(cont[j], 8)[1]:
                    break
        sim.close()
    obs = np.asarray(obs, np.float32); act = np.asarray(act, np.int64)
    lid = np.asarray(lid, np.int64); src = np.asarray(src, np.int8); traj = np.asarray(traj, np.int64)
    np.savez_compressed(DS_PATH, obs=obs, act=act, level_id=lid, source=src, traj=traj)
    print(f"[data] saved {DS_PATH} n={len(obs)} onpath={(src==0).sum()} recover={(src==1).sum()} "
          f"action_hist={np.bincount(act, minlength=N_ACTIONS).tolist()}", flush=True)


# ---- model ------------------------------------------------------------------
def make_model(d_model=64, nhead=4, layers=3):
    import torch, torch.nn as nn

    class EntityTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(D_TOK, d_model)
            self.tok_emb = nn.Parameter(0.02 * torch.randn(N_TOK, d_model))   # learned positional/type
            enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model,
                                             dropout=0.1, batch_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(enc, layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, N_ACTIONS)

        def forward(self, x):
            b = x.shape[0]
            t = x.view(b, N_TOK, D_TOK)
            h = self.proj(t) + self.tok_emb[None]
            h = self.enc(h)
            h = self.norm(h.mean(1))          # mean-pool tokens
            return self.head(h)
    return EntityTransformer()


# ---- stage: train -----------------------------------------------------------
CKPT = ROOT / "data" / f"entity_policy{_TAG}.pt"


def train():
    import torch, torch.nn as nn, torch.nn.functional as F
    d = np.load(DS_PATH)
    X = torch.tensor(d["obs"]); y = torch.tensor(d["act"]); traj = d["traj"]; src = d["source"]
    # held-out val: 15% of recovery trajectories + a slice of on-path by stride
    rng = np.random.default_rng(0)
    val = np.zeros(len(y), bool)
    rtids = np.unique(traj[(src == 1)]); rng.shuffle(rtids)
    val_tids = set(rtids[: max(1, int(0.15 * len(rtids)))].tolist())
    for i in range(len(y)):
        if src[i] == 1 and traj[i] in val_tids: val[i] = True
        if src[i] == 0 and i % 7 == 0: val[i] = True
    tr = ~val
    counts = np.bincount(d["act"][tr], minlength=N_ACTIONS).astype(np.float64) + 1
    # NO class-balancing: for a "run-right + jump" policy the dominant actions ARE correct most
    # of the time; up-weighting rare situational actions (LEFT/DOWN) collapses the policy into
    # spamming them (observed). Mild floor so a truly-absent action isn't impossible.
    cw = torch.ones(N_ACTIONS, dtype=torch.float32)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = make_model().to(dev)
    print(f"[train] device={dev} N={len(y)} train={tr.sum()} val={val.sum()} "
          f"params={sum(p.numel() for p in net.parameters())} class_w={[round(float(c),2) for c in cw]}", flush=True)
    Xtr, ytr = X[tr].to(dev), y[tr].to(dev); Xv, yv = X[val].to(dev), y[val].to(dev)
    cw = cw.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    idx = np.arange(len(ytr)); best = 0.0
    for ep in range(40):
        net.train(); rng.shuffle(idx)
        for st in range(0, len(idx), 256):
            b = idx[st:st + 256]
            logits = net(Xtr[b])
            loss = F.cross_entropy(logits, ytr[b], weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = float((net(Xv).argmax(1) == yv).float().mean()) if val.sum() else 0.0
        best = max(best, acc)
        if ep % 5 == 0 or ep == 39:
            print(f"  ep {ep} val_acc={acc:.3f}", flush=True)
    torch.save({"state": net.state_dict()}, CKPT)
    print(f"[train] saved {CKPT} best_val_acc={best:.3f}", flush=True)


# ---- stage: eval ------------------------------------------------------------
def evaluate(seeds=4):
    import torch
    from mario.reward import is_success, is_death
    net = make_model(); net.load_state_dict(torch.load(CKPT)["state"]); net.eval()
    BIAS = torch.tensor([-0.6, 0, 0, 0, 0, 0, -1.0, -0.8, -0.8])   # discourage NOOP/left/down/up
    results = {}
    for (w, s) in LEVELS:
        beat = 0
        for seed in range(seeds):
            sim = MarioSim(w, s); sim.reset(seed); stall = 0; lastx = 0; ok = False
            for step in range(500):
                with torch.no_grad():
                    a = int((net(torch.tensor(entity_obs(sim.ram))[None]) + BIAS).argmax(1))
                info, done = sim.run_chunk(a, 8)
                if is_success(info): ok = True; break
                if done or is_death(info, done): break
                x = mario_level_x(sim.ram); stall = stall + 1 if x <= lastx else 0; lastx = max(lastx, x)
                if stall > 40: break
            beat += int(ok); sim.close()
        results[f"{w}-{s}"] = beat
        print(f"  {w}-{s}: {beat}/{seeds}", flush=True)
    total = sum(results.values()); print(f"[eval] TOTAL {total}/{len(LEVELS)*seeds} | per-level: {results}", flush=True)


# ---- stage: dagger (closed-loop corrections on the net's own failures) -------
def dagger(rounds=3, seeds=2):
    import torch
    from mario.search import beam_search
    from mario.reward import is_success, is_death
    BIAS = torch.tensor([-0.6, 0, 0, 0, 0, 0, -1.0, -0.8, -0.8])
    for rnd in range(1, rounds + 1):
        net = make_model(); net.load_state_dict(torch.load(CKPT)["state"]); net.eval()
        new = []   # (obs, act, level_id, source=2, traj)
        for (w, s) in LEVELS:
            for seed in range(seeds):
                sim = MarioSim(w, s); sim.reset(seed); acts = []; died_at = None
                for step in range(700):
                    a = int((net(torch.tensor(entity_obs(sim.ram))[None]) + BIAS).argmax(1)); acts.append(a)
                    info, done = sim.run_chunk(a, 8)
                    if is_success(info): break
                    if done or is_death(info, done): died_at = len(acts); break
                sim.close()
                if died_at is None:
                    continue
                for off in (3, 6, 10, 14):
                    if died_at - off < 1:
                        continue
                    prefix = acts[:died_at - off]
                    rec = beam_search(w, s, beam_width=24, max_depth=60, seed=seed, start_prefix=prefix)
                    if not rec.path:
                        continue
                    sim2 = MarioSim(w, s); sim2.reset(seed); dead = False
                    for a in prefix:
                        if sim2.run_chunk(a, 8)[1]: dead = True; break
                    if not dead:
                        for j in range(min(10, len(rec.path))):
                            new.append((entity_obs(sim2.ram), int(rec.path[j]), w * 10 + s, 2, 100000 + rnd))
                            if sim2.run_chunk(rec.path[j], 8)[1]: break
                    sim2.close()
        if not new:
            print(f"[dagger r{rnd}] no corrections collected"); continue
        d = np.load(DS_PATH)
        obs = np.concatenate([d["obs"], np.asarray([r[0] for r in new], np.float32)])
        act = np.concatenate([d["act"], np.asarray([r[1] for r in new], np.int64)])
        lid = np.concatenate([d["level_id"], np.asarray([r[2] for r in new], np.int64)])
        src = np.concatenate([d["source"], np.asarray([r[3] for r in new], np.int8)])
        traj = np.concatenate([d["traj"], np.asarray([r[4] for r in new], np.int64)])
        np.savez_compressed(DS_PATH, obs=obs, act=act, level_id=lid, source=src, traj=traj)
        print(f"[dagger r{rnd}] +{len(new)} corrections (total n={len(obs)}); retraining ...", flush=True)
        train()
        evaluate()


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "sanity"
    if stage == "sanity": sanity()
    elif stage == "data": build_data()
    elif stage == "train": train()
    elif stage == "eval": evaluate()
    elif stage == "dagger":
        dagger(rounds=int(sys.argv[2]) if len(sys.argv) > 2 else 3,
               seeds=int(sys.argv[3]) if len(sys.argv) > 3 else 2)
    elif stage == "all":
        build_data(); train(); evaluate()
    else:
        print(f"unknown stage '{stage}'")
