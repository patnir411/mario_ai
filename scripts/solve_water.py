"""Solve a flagpole-less / side-pipe water level region with coverage search.
Uses the corrected transition detector (pipe_entering / stage change / flag).
    ./venv/bin/python scripts/solve_water.py <w> <s> <prefix_n> <cf> <budget_s>
Saves the area-1 entry tail to /tmp/<w>_<s>_area1.json if a transition fires.
"""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
from mario.search import area_search
w,s=int(sys.argv[1]),int(sys.argv[2])
pn=int(sys.argv[3]) if len(sys.argv)>3 else 238
cf=int(sys.argv[4]) if len(sys.argv)>4 else 4
budget=float(sys.argv[5]) if len(sys.argv)>5 else 240.0
path=json.load(open(ROOT/f"data/solutions/{w}-{s}.json"))["path"]
prefix=path[:pn]
print(f"[water] {w}-{s} prefix={len(prefix)}(cf8) explore_cf={cf} budget={budget}",flush=True)
changed,tail,info=area_search(w,s,start_prefix=prefix,prefix_cf=8,beam_width=160,
    chunk_frames=cf,max_depth=3000,time_budget_s=budget,stuck_cap=200,cov_bonus=100.0,
    progress_every=25)
print("CHANGED",changed,"tail_len",len(tail),"info",{k:info.get(k) for k in('flag_get','stage','world')},flush=True)
if changed:
    json.dump({"prefix_n":pn,"prefix_cf":8,"tail":tail,"tail_cf":cf},open(f"/tmp/{w}_{s}_area1.json","w"))
    print("saved tail")
