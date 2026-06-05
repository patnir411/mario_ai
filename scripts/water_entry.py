import json,sys; from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
from mario.search import area_search
pn=int(sys.argv[1]) if len(sys.argv)>1 else 228
path=json.load(open(ROOT/"data/solutions/2-2.json"))["path"]
print(f"[entry] 2-2 prefix={pn} cf=1 fine coverage",flush=True)
changed,tail,info=area_search(2,2,start_prefix=path[:pn],prefix_cf=8,beam_width=220,
    chunk_frames=1,max_depth=600,time_budget_s=260,stuck_cap=240,cov_bonus=40.0,
    tile=8,progress_every=40)
print("CHANGED",changed,"tail_len",len(tail),"info",{k:info.get(k) for k in('flag_get','stage','world')},flush=True)
if changed:
    json.dump({"prefix_n":pn,"prefix_cf":8,"tail":tail,"tail_cf":1},open("/tmp/2_2_entry.json","w")); print("saved /tmp/2_2_entry.json")
