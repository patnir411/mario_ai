"""DAgger collection — expert-label the LEARNER's own failure states.

The fix for behavior-cloning covariate shift (DESIGN.md §9): roll out the current policy,
find where it fails (states just before a death) or is uncertain (high action entropy),
and for each, run a recovery beam search from a few chunks earlier to get the decisive
correction, labeling that short trajectory. Those (state, correct-action) pairs are added
to the dataset and the policy is retrained — now on its OWN induced state distribution.

Reuses the same picklable-prefix + deterministic-replay machinery as gen_dataset, so
recovery searches and labels parallelize across cores.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from mario.env import MarioSim
from mario.label import fast_label, soft_entropy
from mario.policy import Controller, load_policy
from mario.reward import is_death, is_success
from mario.search import beam_search

PREDEATH_OFFSETS = [4, 8, 12, 16]  # anchor at several points before a death (dense coverage)
ENTROPY_FLAG = 1.4    # also flag uncertain states (nats; ln7=1.95)
RECOVER_BEAM = 24
RECOVER_DEPTH = 60
ANCHOR_BUCKET = 12    # dedup anchors by this x-bucket so seeds don't pile up duplicates


def collect_failures(checkpoint, world, stage, seeds, max_chunks=400):
    """Roll out the policy; return deduped recovery anchors (picklable prefixes).

    An anchor = a prefix from which a recovery search will produce decisive labels. We
    anchor PREDEATH_K chunks before each death, plus at high-entropy states.
    """
    net, ckpt = load_policy(checkpoint, device="cpu")
    cf = ckpt["chunk_frames"]
    anchors: dict[int, list[int]] = {}     # x-bucket -> prefix (keep first)
    outcomes = []
    for seed in seeds:
        ctrl = Controller(net, device="cpu", chunk_frames=cf)
        sim = MarioSim(world, stage)
        info = sim.reset(seed=seed)
        ctrl.reset()
        prefix, ent_log, x_log = [], [], []
        done = False
        for _ in range(max_chunks):
            a, ent, _v = ctrl.act_with_entropy(sim.ram, sim.last_info)
            ent_log.append(ent)
            x_log.append(int(sim.last_info.get("x_pos", 0)))
            prefix.append(a)
            info, done = sim.run_chunk(a, cf)
            if done:
                break
        sim.close()
        died = is_death(info, done)
        outcomes.append({"seed": seed, "beat": is_success(info), "died": died,
                         "n_chunks": len(prefix)})

        def _add(anchor_len):
            anchor_len = max(0, anchor_len)
            xb = (x_log[anchor_len] if anchor_len < len(x_log) else x_log[-1]) // ANCHOR_BUCKET
            anchors.setdefault(xb, prefix[:anchor_len])

        if died:
            for off in PREDEATH_OFFSETS:
                _add(len(prefix) - off)
        for i, e in enumerate(ent_log):
            if e > ENTROPY_FLAG:
                _add(max(0, i - 2))
    return list(anchors.values()), outcomes, ckpt["chunk_frames"]


# ---- parallel worker: recovery search + label its decisive trajectory --------
def dagger_label_worker(spec):
    world, stage, seed, anchor_prefix, tid, keep = spec
    rec = beam_search(world, stage, beam_width=RECOVER_BEAM, max_depth=RECOVER_DEPTH,
                      seed=seed, start_prefix=anchor_prefix)
    if not rec.path:
        return []
    rows = []
    for j in range(min(keep, len(rec.path))):
        prefix = anchor_prefix + rec.path[:j]
        obs, soft, value, all_doomed = fast_label(world, stage, prefix, seed=seed)
        if all_doomed:
            continue
        rows.append({
            "obs": obs, "hard_action": int(rec.path[j]), "soft_targets": soft,
            "value": value, "level_id": world * 10 + stage, "source": 2,
            "seed": seed, "prefix_len": len(prefix), "trajectory_id": tid,
            "_entropy": soft_entropy(soft),
        })
    return rows
