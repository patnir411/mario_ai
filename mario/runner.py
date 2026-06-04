"""Full-game runner: play the multi-stage env level→level, net-primary with search-rescue.

Per-level specialist policies (one MLP can't hold many levels — multi-task interference),
routed by (world,stage); levels without a net are played by search. When the net stalls
or dies repeatedly, beam-search rescues from the live state (search uses the live sim's
own snapshot/restore, so there's no cross-env transfer error). Records a game_result and
captures every frame for a stitched video.

Multi-stage env facts (probed): beating a level changes (world,stage) and resets x≈40;
a death decrements `life` and respawns in the same level; game-over sets done=True.
"""
from __future__ import annotations

import numpy as np

from mario.artifacts import framerule_time
from mario.env import MarioSim
from mario.reward import death_cause, is_death

STALL_CHUNKS = 18          # net makes no forward progress this long -> rescue
RESCUE_DEPTH = 400         # search horizon when rescuing (enough to finish a level)
RESCUE_BEAM = 32


class GameRunner:
    def __init__(self, controllers: dict, *, waypoints: dict | None = None,
                 chunk_frames: int = 8, rescue: bool = True, capture: bool = True):
        self.controllers = controllers          # {(w,s): Controller}; missing -> search
        self.waypoints = waypoints or {}
        self.chunk_frames = chunk_frames
        self.rescue = rescue
        self.capture = capture
        self.frames: list = []

    def _cap(self, sim):
        if self.capture:
            self.frames.append(np.asarray(sim.last_obs).copy())

    def run(self, *, seed: int = 0, max_total_chunks: int = 80000,
            beat=(8, 4)) -> dict:
        from mario.search import search_from_state
        sim = MarioSim(multi_stage=True)
        info = sim.reset(seed=seed)
        self.frames = [np.asarray(sim.last_obs).copy()]

        per_level, total, beat_game = [], 0, False
        w, s = info["world"], info["stage"]
        ctrl = self.controllers.get((w, s))
        if ctrl:
            ctrl.reset()
        x_max, stall, lvl_chunks, lvl_deaths, rescue_chunks = info["x_pos"], 0, 0, 0, 0
        life = info["life"]

        def finalize(cleared: bool):
            per_level.append({
                "world": w, "stage": s, "cleared": cleared,
                "framerule": framerule_time(lvl_chunks * self.chunk_frames) if cleared else None,
                "deaths": lvl_deaths, "search_assist_chunks": rescue_chunks,
                "warp_taken": None,
            })

        while total < max_total_chunks:
            use_search = self.rescue and (ctrl is None or stall >= STALL_CHUNKS)

            if use_search:
                snap = sim.snapshot()
                path = search_from_state(sim, snap, world=w, stage=s,
                                         beam_width=RESCUE_BEAM, depth=RESCUE_DEPTH,
                                         chunk_frames=self.chunk_frames,
                                         waypoints=self.waypoints.get(f"{w}-{s}"))
                sim.restore(snap)
                if not path:                       # nothing better found; nudge forward
                    path = [3]
                for a in path:
                    info, done = sim.run_chunk(a, self.chunk_frames)
                    self._cap(sim)
                    total += 1
                    lvl_chunks += 1
                    rescue_chunks += 1
                    if (info["world"], info["stage"]) != (w, s) or info["life"] != life \
                            or done:
                        break
                stall = 0
            else:
                a = ctrl.act(sim.ram, sim.last_info)
                info, done = sim.run_chunk(a, self.chunk_frames)
                self._cap(sim)
                total += 1
                lvl_chunks += 1
                x = info["x_pos"]
                stall = 0 if x > x_max else stall + 1
                x_max = max(x_max, x)

            nw, ns = info["world"], info["stage"]
            # level transition (beat the level)
            if (nw, ns) != (w, s):
                finalize(cleared=True)
                if (w, s) == tuple(beat):
                    beat_game = True
                    break
                w, s = nw, ns
                ctrl = self.controllers.get((w, s))
                if ctrl:
                    ctrl.reset()
                x_max, stall, lvl_chunks, lvl_deaths, rescue_chunks = info["x_pos"], 0, 0, 0, 0
                life = info["life"]
                continue
            # death / respawn within the level
            if info["life"] != life:
                lvl_deaths += 1
                life = info["life"]
                if ctrl:
                    ctrl.reset()
                x_max, stall = info["x_pos"], 0
            if done:   # game over (lives exhausted) — unless it was the beat transition
                if (w, s) == tuple(beat) and info.get("flag_get"):
                    beat_game = True
                    finalize(cleared=True)
                break

        sim.close()
        return {"per_level": per_level, "beat_game": beat_game, "total_chunks": total}
