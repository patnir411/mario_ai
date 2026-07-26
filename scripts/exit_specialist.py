"""Expert-iteration-style per-level specialist experiment.

This script tests whether three concrete changes improve the earlier local
offline-BC result; it does not treat success or failure on one level as a
general verdict on imitation learning:
  (1) denser route and hazard-adjacent coverage;
  (2) explicit weighting for rare jump and death-adjacent states;
  (3) deeper-lookahead labels rather than one-ply action scores.

For one specialist level it implements:
  - DENSE route coverage: label every state along the cached solution with a
    short lookahead search;
  - EXPERT-ITERATION off-path coverage: roll out the current net, anchor at deaths/stalls/high-
    entropy, recover with the oracle (beam_search start_prefix=), label those corrections too.
  - HAZARD-AWARE loss: focal CE (down-weights easy run-right frames, up-weights the rare jump)
    + an extra weight on states where at least one lookahead action is fatal.
It then measures standalone closed-loop completion. A positive result would show
that these changes helped this configuration and would justify testing the net
as a search prior; it would not isolate which change caused the improvement.

    PYTORCH_ENABLE_MPS_FALLBACK=1 ./venv/bin/python scripts/exit_specialist.py 1-1 \
        [--rounds 3] [--cf 8] [--seeds 6] [--pool 8] [--out data/specialists/1-1.pt]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing import Pool

from mario.entity import OBS_DIM_ENTITY, entity_obs
from mario.entity_policy import (ACTION_BIAS, TemporalEntityController,
                                 TemporalEntityTransformer, n_params, save_policy)
from mario.env import MarioSim, N_ACTIONS
from mario.label import label_at_state, label_state
from mario.reward import DEFAULT, is_death, is_success
from mario.ram import mario_level_x
from mario.search import beam_search

SOL = ROOT / "data" / "solutions"
HAZARD_Q = -DEFAULT.death / 2          # a 1-ply action value <= this == that action is fatal
JUMP_ACTIONS = {2, 4, 5}               # right+A, right+A+B, A  (the rare critical class)

PREDEATH_OFFSETS = [3, 6, 10, 14]      # anchors before a death (dense hazard coverage)
ENTROPY_FLAG = 1.4                     # nats; ln9 = 2.20
ANCHOR_BUCKET = 8                      # dedup anchors by x-bucket
RECOVER_BEAM = 16
RECOVER_DEPTH = 50
RECOVER_KEEP = 6                       # label this many steps of each recovery (only need the
                                       # decisive correction, not a full solve from the anchor)


# --------------------------------------------------------------------------- #
#  Picklable labeling workers (prefix-based; deterministic replay rebuilds state)
# --------------------------------------------------------------------------- #
DEEP_DEPTH = 6        # lookahead chunks for the deep teacher (penalizes "coast now, die later")
DEEP_BW = 4


def _label_one(spec):
    """Label ONE state given its action prefix. The hard label is the DEEP-lookahead oracle
    argmax (NOT the recorded/1-ply action): a short beam rollout per action sees that coasting
    now dooms an upcoming pit, so it values the run/jump that 1-ply myopia misses. Returns the
    entity obs + deep soft/value/hazard, or None for all-doomed states (nothing to learn)."""
    world, stage, prefix, cf, source, ctx = spec
    # replay the prefix, recording the entity obs after each chunk so we can build the last-ctx
    # temporal stack at the labeled state (separate replay — label_state reads production obs).
    sim = MarioSim(world, stage); sim.reset(seed=0)
    hist = [entity_obs(sim.ram, sim.last_info)]
    done = False
    for a in prefix:
        _info, done = sim.run_chunk(a, cf)
        if done:
            break
        hist.append(entity_obs(sim.ram, sim.last_info))
    sim.close()
    if done:
        return None
    h = hist[-ctx:]
    while len(h) < ctx:                        # left-pad the start of the level
        h = [h[0]] + h
    obs = np.stack(h).astype(np.float32)       # (ctx, OBS_DIM_ENTITY)
    # deep-lookahead Q sweep (the "distill the search" teacher) — targets are state-based
    res = label_state(world, stage, list(prefix), chunk_frames=cf,
                      depth=DEEP_DEPTH, beam_width=DEEP_BW)
    if res.all_doomed:
        return None
    q = res.per_action_value
    hazard = bool((q <= HAZARD_Q).any())       # some action here is fatal -> death-cliff
    return {"obs": obs, "hard": int(res.best_action),
            "soft": res.soft_targets.astype(np.float32), "value": float(res.value),
            "hazard": hazard, "source": int(source)}


def _recover_and_label(spec):
    """Off-path EXPERT ITERATION: from a failure anchor, recover with the oracle and label
    the decisive correction trajectory. Returns a list of rows."""
    world, stage, anchor, seed, cf, ctx = spec
    rec = beam_search(world, stage, beam_width=RECOVER_BEAM, max_depth=RECOVER_DEPTH,
                      seed=seed, start_prefix=list(anchor), chunk_frames=cf)
    if not rec.path:
        return []
    rows = []
    for j in range(min(RECOVER_KEEP, len(rec.path))):
        r = _label_one((world, stage, list(anchor) + rec.path[:j], cf, 2, ctx))
        if r is not None:
            rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
#  Coverage collection
# --------------------------------------------------------------------------- #
def onpath_specs(world, stage, solution, cf, ctx):
    """Every state along the solution; `_label_one` supplies the lookahead label."""
    return [(world, stage, tuple(solution[:k]), cf, 0, ctx)
            for k in range(len(solution))]


def collect_failures(net, world, stage, seeds, cf, max_chunks=400):
    """Roll out the current net; return deduped failure anchors (picklable prefixes) over its
    OWN induced distribution — anchored before deaths, at stalls, and at high-entropy states."""
    ctrl = TemporalEntityController(net, device="cpu")
    anchors: dict[int, tuple] = {}
    outcomes = []
    for seed in seeds:
        sim = MarioSim(world, stage)
        sim.reset(seed=seed)
        ctrl.reset()
        prefix, ent_log, x_log = [], [], []
        done = False; info = sim.last_info
        for _ in range(max_chunks):
            a, ent, _v = ctrl.act_with_entropy(sim.ram, sim.last_info)
            ent_log.append(ent); x_log.append(mario_level_x(sim.ram)); prefix.append(a)
            info, done = sim.run_chunk(a, cf)
            if done:
                break
        sim.close()
        beat, died = is_success(info), is_death(info, done)
        outcomes.append({"seed": seed, "beat": beat, "died": died, "n": len(prefix),
                         "x_max": max(x_log) if x_log else 0})

        def _add(alen):
            alen = max(0, min(alen, len(prefix)))
            xb = (x_log[alen] if alen < len(x_log) else x_log[-1]) // ANCHOR_BUCKET
            anchors.setdefault(xb, tuple(prefix[:alen]))

        if died:
            for off in PREDEATH_OFFSETS:
                _add(len(prefix) - off)
        elif not beat:
            peak = int(np.argmax(x_log)) if x_log else 0
            for off in (0, 4, 8):
                _add(peak - off)
        for i, e in enumerate(ent_log):
            if e > ENTROPY_FLAG:
                _add(i - 2)
    return list(anchors.values()), outcomes


# --------------------------------------------------------------------------- #
#  Training (hazard-aware focal loss)
# --------------------------------------------------------------------------- #
VALUE_SCALE = 10000.0


def train(net, rows, dev, *, epochs, lr, batch=64, focal_gamma=2.0, hazard_w=3.0, jump_w=1.5):
    X = torch.tensor(np.stack([r["obs"] for r in rows])).to(dev)
    hard = torch.tensor([r["hard"] for r in rows], dtype=torch.long).to(dev)
    soft = torch.tensor(np.stack([r["soft"] for r in rows])).to(dev)
    value = torch.tensor([r["value"] for r in rows], dtype=torch.float32).clamp(
        -VALUE_SCALE, 2 * VALUE_SCALE).to(dev)
    # per-sample weight: COUNT-BALANCED across sources (so the gold on-path data is not drowned
    # as corrections accumulate — the round-1 collapse was on-path getting outnumbered + warm-
    # start forgetting) x hazard-up-weight x rare-jump-up-weight.
    from collections import Counter
    cnt = Counter(r["source"] for r in rows)
    src_norm = {s: len(rows) / (len(cnt) * c) for s, c in cnt.items()}
    w = torch.tensor([src_norm[r["source"]]
                      * (hazard_w if r["hazard"] else 1.0)
                      * (jump_w if r["hard"] in JUMP_ACTIONS else 1.0)
                      for r in rows], dtype=torch.float32).to(dev)
    n = len(rows)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    idx = np.arange(n); rng = np.random.default_rng(0)
    net.to(dev)
    for ep in range(epochs):
        net.train(); rng.shuffle(idx)
        last = {}
        for st in range(0, n, batch):
            b = idx[st:st + batch]
            bt = torch.from_numpy(b).to(dev)
            logits, vpred = net(X[bt])
            ce = F.cross_entropy(logits, hard[bt], reduction="none")
            pt = torch.exp(-ce)
            focal = ((1.0 - pt) ** focal_gamma) * ce          # focal: hard/rare states dominate
            loss_ce = (focal * w[bt]).mean()
            soft_ce = -(soft[bt] * F.log_softmax(logits, dim=1)).sum(1).mean()
            vloss = F.mse_loss(vpred, value[bt] / VALUE_SCALE)
            loss = loss_ce + 0.3 * soft_ce + 0.3 * vloss
            opt.zero_grad(); loss.backward(); opt.step()
            last = {"ce": float(loss_ce.detach()), "soft": float(soft_ce.detach()),
                    "v": float(vloss.detach())}
        if ep % 5 == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                acc = float((net.logits(X).argmax(1) == hard).float().mean())
            print(f"    ep {ep:3d} acc={acc:.3f} ce={last['ce']:.3f} "
                  f"soft={last['soft']:.3f} v={last['v']:.4f}", flush=True)
    net.eval()
    return net


# --------------------------------------------------------------------------- #
#  Closed-loop evaluation (STANDALONE — the decisive number)
# --------------------------------------------------------------------------- #
def evaluate(net, world, stage, seeds, cf, end_x, max_chunks=500, biased=True):
    ctrl = TemporalEntityController(net, device="cpu")
    beats = 0; fracs = []
    for seed in seeds:
        sim = MarioSim(world, stage); sim.reset(seed=seed); ctrl.reset()
        x_max = 0; done = False; info = sim.last_info
        for _ in range(max_chunks):
            a = ctrl.act(sim.ram, sim.last_info, biased=biased)
            info, done = sim.run_chunk(a, cf)
            x_max = max(x_max, mario_level_x(sim.ram))
            if is_success(info):
                beats += 1; break
            if is_death(info, done) or done:
                break
        sim.close()
        fracs.append(min(1.0, x_max / max(1, end_x)))
    return {"beat": beats, "n": len(seeds), "median_frac": float(np.median(fracs)),
            "max_frac": float(np.max(fracs))}


def save_rows(rows, path):
    np.savez_compressed(path,
                        obs=np.stack([r["obs"] for r in rows]),
                        hard=np.array([r["hard"] for r in rows], np.int64),
                        soft=np.stack([r["soft"] for r in rows]),
                        value=np.array([r["value"] for r in rows], np.float32),
                        hazard=np.array([r["hazard"] for r in rows], bool),
                        source=np.array([r["source"] for r in rows], np.int64))


def load_rows(path):
    d = np.load(path)
    return [{"obs": d["obs"][i], "hard": int(d["hard"][i]), "soft": d["soft"][i],
             "value": float(d["value"][i]), "hazard": bool(d["hazard"][i]),
             "source": int(d["source"][i])} for i in range(len(d["obs"]))]


def fresh_net(args):
    return TemporalEntityTransformer(ctx_k=args.ctx, d_model=args.d_model, layers=args.layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("level")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--resume-rows", default="")    # skip re-search: load accumulated dataset
    ap.add_argument("--cf", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=4)   # temporal-context window (frames); 1 = baseline
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    world, stage = (int(x) for x in args.level.split("-"))
    cf = args.cf
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    sol = json.loads((SOL / f"{args.level}.json").read_text())
    solution = sol["path"]
    seeds = list(range(args.seeds))
    out = Path(args.out) if args.out else ROOT / "data" / "specialists" / f"{args.level}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)

    # verify the cached solution actually reaches the flag at this cf (replay-validity gate)
    sim = MarioSim(world, stage); sim.reset(seed=0); end_x = 0; beat = False
    for a in solution:
        info, done = sim.run_chunk(a, cf); end_x = max(end_x, mario_level_x(sim.ram))
        if is_success(info):
            beat = True; break
        if done:
            break
    sim.close()
    if not beat:
        raise SystemExit(f"solution for {args.level} does not reach flag at cf={cf} "
                         f"(by={sol.get('by')}) — re-solve or pick the right cf")
    print(f"== ExIt specialist {args.level} | sol_len={len(solution)} end_x={end_x} "
          f"ctx={args.ctx} dev={dev} ==", flush=True)

    rows_path = out.parent / f"{args.level}_rows.npz"

    def fit_and_eval(rows, rnd, extra):
        """Retrain a FRESH net from scratch on the full aggregate (standard DAgger — no warm-
        start, which caused the round-1 forgetting), then report standalone closed-loop."""
        net = train(fresh_net(args), rows, dev, epochs=200, lr=3e-4)
        ev = evaluate(net, world, stage, seeds, cf, end_x, biased=False)
        evb = evaluate(net, world, stage, seeds, cf, end_x, biased=True)
        print(f"round {rnd} eval: unbiased beat={ev['beat']}/{ev['n']} med={ev['median_frac']:.2f} "
              f"max={ev['max_frac']:.2f} | biased med={evb['median_frac']:.2f}", flush=True)
        history.append({"round": rnd, "n_states": len(rows), "eval": ev,
                        "eval_biased": evb, **extra})
        return net

    history = []
    if args.resume_rows:
        rows = load_rows(args.resume_rows)
        print(f"resumed {len(rows)} rows from {args.resume_rows}", flush=True)
        start_round = 1
    else:
        # ---- round 0: dense on-path coverage only ----
        t0 = time.time()
        specs = onpath_specs(world, stage, solution, cf, args.ctx)
        with Pool(args.pool) as pool:
            rows = [r for r in pool.map(_label_one, specs) if r is not None]
        print(f"round 0: on-path states={len(rows)} hazard={sum(r['hazard'] for r in rows)} "
              f"({time.time()-t0:.0f}s labeling) params={n_params(fresh_net(args))}", flush=True)
        save_rows(rows, rows_path)
        start_round = 1
    net = fit_and_eval(rows, 0, {})

    # ---- rounds 1..R: expert-iteration off-path coverage (retrain from scratch each round) ----
    for rnd in range(start_round, args.rounds + 1):
        t0 = time.time()
        anchors, outcomes = collect_failures(net, world, stage, seeds, cf)
        beats = sum(o["beat"] for o in outcomes)
        recspecs = [(world, stage, anc, 0, cf, args.ctx) for anc in anchors]
        with Pool(args.pool) as pool:
            new = [r for batch in pool.map(_recover_and_label, recspecs) for r in batch]
        rows.extend(new)
        save_rows(rows, rows_path)
        print(f"round {rnd}: rollout beats={beats}/{len(seeds)} anchors={len(anchors)} "
              f"+{len(new)} corrections (hazard={sum(r['hazard'] for r in new)}) "
              f"total={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
        net = fit_and_eval(rows, rnd, {"rollout_beats": beats, "n_corrections": len(new)})

    save_policy(net, out, extra={"level": args.level, "cf": cf, "history": history,
                                 "by": "exit_specialist"})
    print(f"\nsaved {out}", flush=True)
    print("HISTORY:", json.dumps(history, indent=2), flush=True)


if __name__ == "__main__":
    main()
