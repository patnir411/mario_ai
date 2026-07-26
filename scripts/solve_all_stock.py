"""Batch-solve all unsolved stock SMB levels to build LEVEL DIVERSITY for the generalist.

For each unverified (world,stage) it runs an escalating ladder of the existing
per-level solvers until deterministic seed-0 replay of
`data/solutions/{w}-{s}.json` reaches the flag:

    tier 1  solve_beam.py      (linear overworld/underground/water)
    tier 2  solve_coverage.py  (maze / vertical / loop-back; bigger beam + novelty)
    tier 3  solve_castle.py    (stage-4 castles: ProcLoopCommand Y-gate Go-Explore)

Each level runs its ladder in its OWN subprocess (snapshots aren't picklable -> one
in-process search per OS process); levels run in parallel up to MAX_PARALLEL. Idempotent:
already-verified levels are skipped, and each tier short-circuits once replay passes.
Positive search claims that fail replay are quarantined as negative evidence.

    ./venv/bin/python scripts/solve_all_stock.py [max_parallel] [only=W-S,W-S,...]
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SOL = ROOT / "data" / "solutions"
PY = str(ROOT / "venv" / "bin" / "python")

from mario.io import write_json_atomic  # noqa: E402
from mario.solution_verification import verify_stock_solution  # noqa: E402

ALL_LEVELS = [(w, s) for w in range(1, 9) for s in range(1, 5)]
_REPLAY_LOCK = threading.Lock()


@lru_cache(maxsize=128)
def _verify_fingerprint(path: str, mtime_ns: int, size: int):
    del mtime_ns, size
    # Keep emulator construction/replay serialized in this coordinator process;
    # the expensive searches themselves still run in parallel subprocesses.
    with _REPLAY_LOCK:
        return verify_stock_solution(path, seed=0)


def verification(w: int, s: int):
    manifest = SOL / f"{w}-{s}.json"
    if not manifest.exists():
        return verify_stock_solution(manifest, seed=0)
    stat = manifest.stat()
    return _verify_fingerprint(str(manifest), stat.st_mtime_ns, stat.st_size)


def is_solved(w: int, s: int) -> bool:
    """A level counts only after deterministic seed-0 replay reaches the flag."""
    return verification(w, s).replay_verified


def quarantine_failed_positive(w: int, s: int) -> bool:
    """Turn a failed positive cache into an explicit negative evidence artifact."""
    check = verification(w, s)
    if not check.claimed_solved or check.replay_verified:
        return False
    manifest = SOL / f"{w}-{s}.json"
    data = json.loads(manifest.read_text())
    data["search_claimed_solved"] = True
    data["solved"] = False
    data["replay_verified"] = False
    data["invalid_reason"] = f"replay_{check.reason}"
    write_json_atomic(manifest, data)
    return True


def tiers_for(w: int, s: int) -> list[tuple[str, list[str], float]]:
    """(label, argv, timeout_s) ladder.

    beam at width 64 solves a linear level in ~11 min (verified on 3-1: x3193 in 649s), so
    the beam window is 900s with margin. Castles (stage 4) loop under plain beam, so they go
    straight to the page-gate castle solver, then coverage as a fallback.
    """
    coverage = ("coverage", [PY, "scripts/solve_coverage.py", str(w), str(s),
                             "192", "30.0", "100", "1200", "8"], 1300.0)
    if s == 4:
        castle = ("castle", ["env", "PYTORCH_ENABLE_MPS_FALLBACK=1", PY,
                             "scripts/solve_castle.py", str(w), str(s), "1200"], 1300.0)
        return [castle, coverage]
    beam = ("beam", [PY, "scripts/solve_beam.py", str(w), str(s), "64", "600"], 900.0)
    return [beam, coverage]


def solve_one(w: int, s: int) -> dict:
    t0 = time.time()
    if is_solved(w, s):
        return {"level": f"{w}-{s}", "solved": True, "tier": "cached", "secs": 0.0}
    for label, argv, timeout in tiers_for(w, s):
        try:
            subprocess.run(argv, cwd=str(ROOT), timeout=timeout,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired:
            pass
        if is_solved(w, s):
            return {"level": f"{w}-{s}", "solved": True, "tier": label,
                    "secs": round(time.time() - t0, 1)}
        quarantine_failed_positive(w, s)
    return {"level": f"{w}-{s}", "solved": False, "tier": "exhausted",
            "secs": round(time.time() - t0, 1)}


def main() -> None:
    max_par = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    only = None
    for a in sys.argv[2:]:
        if a.startswith("only="):
            only = {tuple(int(x) for x in p.split("-")) for p in a[5:].split(",")}

    targets = [(w, s) for (w, s) in ALL_LEVELS
               if not is_solved(w, s) and (only is None or (w, s) in only)]
    already = [f"{w}-{s}" for (w, s) in ALL_LEVELS if is_solved(w, s)]
    print(f"already solved ({len(already)}): {', '.join(already)}", flush=True)
    print(f"to solve ({len(targets)}): {', '.join(f'{w}-{s}' for w, s in targets)}", flush=True)
    print(f"max_parallel={max_par}\n", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=max_par) as ex:
        futs = {ex.submit(solve_one, w, s): (w, s) for (w, s) in targets}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            mark = "OK  " if r["solved"] else "FAIL"
            print(f"  [{mark}] {r['level']:>4}  tier={r['tier']:<9} {r['secs']}s", flush=True)

    n_solved_now = sum(1 for (w, s) in ALL_LEVELS if is_solved(w, s))
    newly = sorted(r["level"] for r in results if r["solved"] and r["tier"] != "cached")
    failed = sorted(r["level"] for r in results if not r["solved"])
    print(f"\n=== DONE === total solved: {n_solved_now}/32")
    print(f"newly solved this run: {', '.join(newly) or '(none)'}")
    print(f"still unsolved: {', '.join(failed) or '(none)'}")


if __name__ == "__main__":
    main()
