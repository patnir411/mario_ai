import json,sys; from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
from mario.search import area_search
w,s,cf,bw,bud=2,2,4,200,560.0
print(f"[fresh-water] {w}-{s} cf={cf} beam={bw} budget={bud}",flush=True)
changed,path,info=area_search(w,s,start_prefix=None,chunk_frames=cf,beam_width=bw,
    max_depth=4000,time_budget_s=bud,stuck_cap=160,cov_bonus=120.0,progress_every=40,
    checkpoint_path=f"/tmp/{w}_{s}_fresh_ckpt.json")
print("CHANGED",changed,"len",len(path),"info",{k:info.get(k) for k in('flag_get','stage','world')},flush=True)
if changed:
    json.dump({"path":path,"cf":cf},open(f"/tmp/{w}_{s}_fresh.json","w")); print("saved")
