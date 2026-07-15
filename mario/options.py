"""Option-level meta-MDP scaffolding for hierarchical Mario planning.

The low-level solver produces verified *options* (clear a level, acquire an
item, use an item).  This module searches over those options in a small symbolic
state while still allowing opaque-effect options to be executed against a
resettable emulator and then classified from the observed post-state.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Iterable


class KnowledgeTier(IntEnum):
    """How much route/domain knowledge an option injects into the benchmark."""

    TIER0_GENERIC = 0
    TIER1_ITEM_GIVEN = 1
    TIER2_BLACK_BOX_OPTION = 2
    TIER3_SUBGOAL_HINT = 3
    TIER4_LLM_OR_DISASM = 4
    TIER5_FULL_SCRIPT = 5


@dataclass(frozen=True)
class MetaState:
    """Symbolic state for the SMB3 overworld/item planning layer."""

    world: int
    node: tuple[int, int] = (0, 0)
    cleared: int = 0
    inventory: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "world", int(self.world))
        object.__setattr__(self, "node", (int(self.node[0]), int(self.node[1])))
        object.__setattr__(self, "cleared", int(self.cleared))
        object.__setattr__(self, "inventory", tuple(sorted(set(self.inventory))))
        object.__setattr__(self, "flags", tuple(sorted(set(self.flags))))

    def has_item(self, item: str) -> bool:
        return item in self.inventory

    def with_item(self, item: str) -> "MetaState":
        return MetaState(self.world, self.node, self.cleared,
                         self.inventory + (item,), self.flags)

    def without_item(self, item: str) -> "MetaState":
        return MetaState(self.world, self.node, self.cleared,
                         tuple(i for i in self.inventory if i != item), self.flags)

    def with_world_node(self, world: int, node: tuple[int, int]) -> "MetaState":
        return MetaState(world, node, self.cleared, self.inventory, self.flags)

    def with_cleared(self, mask: int) -> "MetaState":
        return MetaState(self.world, self.node, self.cleared | int(mask),
                         self.inventory, self.flags)

    def with_flag(self, flag: str) -> "MetaState":
        return MetaState(self.world, self.node, self.cleared,
                         self.inventory, self.flags + (flag,))

    def to_json(self) -> dict:
        return {
            "world": self.world,
            "node": list(self.node),
            "cleared": self.cleared,
            "inventory": list(self.inventory),
            "flags": list(self.flags),
        }


@dataclass(frozen=True)
class OptionCost:
    frames: int = 0
    nodes: int = 0
    wall_clock_s: float = 0.0

    def __add__(self, other: "OptionCost") -> "OptionCost":
        return OptionCost(
            frames=self.frames + other.frames,
            nodes=self.nodes + other.nodes,
            wall_clock_s=self.wall_clock_s + other.wall_clock_s,
        )

    def to_json(self) -> dict:
        return {
            "frames": int(self.frames),
            "nodes": int(self.nodes),
            "wall_clock_s": float(self.wall_clock_s),
        }


@dataclass
class OptionResult:
    success: bool
    state: MetaState
    cost: OptionCost = field(default_factory=OptionCost)
    entry_snapshot: Any = None
    exit_snapshot: Any = None
    info: dict = field(default_factory=dict)
    observed_effect: dict | None = None


@dataclass
class OptionContext:
    """Execution context for resettable opaque-effect option exploration."""

    executor: Any = None
    snapshots: dict[MetaState, Any] = field(default_factory=dict)

    def restore(self, state: MetaState) -> bool:
        snap = self.snapshots.get(state)
        if self.executor is None or snap is None:
            return False
        self.executor.restore(snap)
        return True

    def remember(self, state: MetaState, snapshot: Any) -> None:
        if snapshot is not None:
            self.snapshots[state] = snapshot


Precondition = Callable[[MetaState], bool]
Runner = Callable[[MetaState, OptionContext], OptionResult]
Verifier = Callable[[Any], bool]


class SMA4SolutionExecutor:
    """Adapter from option execution to the existing cached SMA4 replay path."""

    def __init__(self, core: Any, *, settle: bool = True):
        self.core = core
        self.settle = bool(settle)

    def restore(self, snapshot: Any) -> None:
        self.core.restore(snapshot)

    def snapshot(self) -> Any:
        return self.core.snapshot()

    def execute_clear_solution(self, solution_path: str | Path,
                               entry_snapshot: str | None = None) -> tuple[dict, Any]:
        from mario.overworld_search import _load_snapshot, replay_cached_solution

        solution_path = Path(solution_path)
        solution = json.loads(solution_path.read_text())
        entry_snap = _load_snapshot(entry_snapshot or solution.get("snapshot"))
        summary = replay_cached_solution(self.core, solution, entry_snap=entry_snap,
                                         settle=self.settle)
        return summary, self.core.snapshot()


class SMA4WhistleExecutor:
    """Executor for the SMA4 two-whistle warp route.

    Tier-1 path hand-grants whistles.  Tier-2 path replays the verified
    `acquire_whistle_1_3` solution (white-block → Toad chest → map exit).
    Whistle spend options remain effect-opaque: the meta-search must execute
    them against the emulator and read the resulting RAM state.
    """

    INVENTORY_START = 0x03002C2E
    INVENTORY_SLOTS = 36
    WARP_WHISTLE = 0x0C
    ITEM_MENU_OPEN = 0x03003772
    MAP_EVENT = 0x03003774
    MAP_DEST_OR_REGION = 0x03003CB7
    WARP_ZONE_WORLD_RAW = 8
    WORLD_8_RAW = 7
    ACQUIRE_WHISTLE_1_3 = Path("data/solutions/sma4/acquire_whistle_1_3.json")
    ACQUIRE_WHISTLE_FORTRESS = Path(
        "data/solutions/sma4/acquire_whistle_fortress.json")

    def __init__(self, core: Any):
        self.core = core

    def restore(self, snapshot: Any) -> None:
        self.core.restore(snapshot)

    def snapshot(self) -> Any:
        return self.core.snapshot()

    def _read_u8(self, addr: int) -> int:
        return int(self.core._read_u8(addr))

    def _write_u8(self, addr: int, value: int) -> None:
        self.core.env.data.memory.assign(addr, "|u1", int(value) & 0xFF)

    def _step(self, buttons: tuple[str, ...], frames: int) -> int:
        for _ in range(frames):
            self.core._step_buttons(buttons)
        return int(frames)

    def _wait_until(self, predicate, *, max_frames: int = 1800,
                    step: int = 12) -> tuple[bool, int]:
        elapsed = 0
        while elapsed < max_frames:
            self._step((), step)
            elapsed += step
            if predicate():
                return True, elapsed
        return False, elapsed

    def _open_and_use_selected_item(self) -> int:
        frames = 0
        frames += self._step(("L",), 6)
        frames += self._step((), 30)
        frames += self._step(("A",), 6)
        return frames

    def inventory(self) -> list[int]:
        return [self._read_u8(self.INVENTORY_START + i)
                for i in range(self.INVENTORY_SLOTS)]

    def sample(self, label: str, frame: int = 0) -> dict:
        return {
            "label": label,
            "frame": int(frame),
            "world_raw_0_indexed": self._read_u8(self.core.WORLD),
            "world_normalized": int(self.core.last_info.get("world", 0)),
            "is_warp_zone": bool(self.core.last_info.get("is_warp_zone", False)),
            "cursor": [
                self._read_u8(self.core.MAP_CURSOR_X),
                self._read_u8(self.core.MAP_CURSOR_Y),
            ],
            "mode": self.core.last_info.get("mode"),
            "time": int(self.core.last_info.get("time", 0)),
            "inventory_first4": self.inventory()[:4],
            "item_menu_open": self._read_u8(self.ITEM_MENU_OPEN),
            "map_event": self._read_u8(self.MAP_EVENT),
            "map_dest_or_region": self._read_u8(self.MAP_DEST_OR_REGION),
        }

    def grant_whistles(self, count: int = 2) -> dict:
        for i in range(self.INVENTORY_SLOTS):
            self._write_u8(self.INVENTORY_START + i, 0)
        for i in range(int(count)):
            self._write_u8(self.INVENTORY_START + i, self.WARP_WHISTLE)
        self.core._last_info = self.core._normalize_info(self.core.last_info)
        return {
            "success": True,
            "cost_frames": 0,
            "samples": [self.sample(f"hand_granted_{count}_whistles")],
            "whistles": int(count),
        }

    def _load_snapshot_file(self, path: Path) -> Any:
        obj = pickle.loads(Path(path).read_bytes())
        if isinstance(obj, dict):
            for key in ("entry_snap", "snapshot", "snap"):
                if key in obj:
                    return obj[key]
        return obj

    def acquire_whistle_1_3(
            self,
            solution_path: str | Path | None = None,
            *,
            entry_snapshot: str | Path | None = None,
    ) -> dict:
        """Replay the verified 1-3 white-block AcquireWhistle option.

        Initiation restores the cached P-Wing 1-3 entry snapshot (documented as
        an injected precondition until map P-Wing use is itself an option).
        Success = inventory slot contains warp whistle ``0x0C`` after the
        Toad-house chest open + map exit.
        """
        import numpy as np

        sol_path = Path(solution_path or self.ACQUIRE_WHISTLE_1_3)
        sol = json.loads(sol_path.read_text())
        entry = Path(entry_snapshot or sol.get("entry_snapshot")
                     or "runs/sma4_cache/1-3_pwing_entry.pkl")
        buttons = [tuple(b) for b in sol["path_buttons"]]
        prior = self.inventory()
        prior_count = sum(1 for v in prior if v == self.WARP_WHISTLE)
        frames = 0
        samples = [self.sample("before_acquire_whistle_1_3", frames)]
        self.core.level_id = "1-3"
        self.core.restore(self._load_snapshot_file(entry))
        # Preserve any already-held whistles across the P-Wing entry restore
        # (e.g. fortress acquire ran first; chest must be able to stack).
        for i, val in enumerate(prior[: self.INVENTORY_SLOTS]):
            if int(val):
                self._write_u8(self.INVENTORY_START + i, int(val))
        samples.append(self.sample("restored_pwing_1_3_entry", frames))
        for bt in buttons:
            self.core._step_buttons(bt)
            frames += 1
        inv = self.inventory()
        after_count = sum(1 for v in inv if v == self.WARP_WHISTLE)
        success = after_count >= max(1, prior_count + 1)
        # If the recorded path stopped inside the house, finish the exit.
        if success and float(np.asarray(self.core.last_obs).mean()) > 40:
            # Already includes exit in the verified solution; no-op settle.
            frames += self._step((), 12)
        samples.append(self.sample("after_acquire_whistle_1_3", frames))
        injected = ["pwing_1_3_entry_snapshot"]
        if prior_count:
            injected.append("prior_whistle_inventory_merged")
        return {
            "success": bool(success),
            "cost_frames": int(frames),
            "samples": samples,
            "inventory_first4": inv[:4],
            "whistle_count": int(after_count),
            "solution": str(sol_path),
            "entry_snapshot": str(entry),
            "knowledge_tier": int(KnowledgeTier.TIER2_BLACK_BOX_OPTION),
            "injected_facts": injected,
        }

    def acquire_whistle_fortress(
            self,
            solution_path: str | Path | None = None,
            *,
            entry_snapshot: str | Path | None = None,
    ) -> dict:
        """Replay verified W1 Fortress AcquireWhistle (roof → chest → map).

        Restores the door-area entry snapshot.  If the live inventory already
        holds a whistle (e.g. after ``acquire_whistle_1_3``), those slots are
        copied onto the restored entry so the chest can stack a second ``0x0C``.
        Leaf/pspeed are re-held during the fly (same honesty class as 1-3 P-Wing
        entry).  Success = overworld with at least one more whistle than before.
        """
        sol_path = Path(solution_path or self.ACQUIRE_WHISTLE_FORTRESS)
        sol = json.loads(sol_path.read_text())
        entry = Path(entry_snapshot or sol.get("entry_snapshot")
                     or "runs/sma4_cache/1-fortress_door_entry.pkl")
        buttons = [tuple(b) for b in sol["path_buttons"]]
        prior = self.inventory()
        prior_count = sum(1 for v in prior if v == self.WARP_WHISTLE)
        frames = 0
        samples = [self.sample("before_acquire_whistle_fortress", frames)]
        self.core.level_id = "1-fortress"
        self.core.restore(self._load_snapshot_file(entry))
        # Preserve any already-held whistles across the door-entry restore.
        for i, val in enumerate(prior[: self.INVENTORY_SLOTS]):
            if int(val):
                self._write_u8(self.INVENTORY_START + i, int(val))
        self._write_u8(self.core.POWERUP, 3)
        self._write_u8(self.core.PSPEED, 127)
        frames += self._step((), 3)
        samples.append(self.sample("restored_fortress_door_entry", frames))
        for bt in buttons:
            self.core._step_buttons(bt)
            frames += 1
            if int(self.core.last_info.get("powerup") or 0) < 3:
                self._write_u8(self.core.POWERUP, 3)
        # Dual-whistle entry can finish mid-level; settle out with A/LEFT.
        if self.core.last_info.get("mode") != "overworld":
            for _ in range(400):
                self.core._step_buttons(("A",))
                frames += 1
                if self.core.last_info.get("mode") == "overworld":
                    break
                self.core._step_buttons(("LEFT",))
                frames += 1
                if self.core.last_info.get("mode") == "overworld":
                    break
        inv = self.inventory()
        after_count = sum(1 for v in inv if v == self.WARP_WHISTLE)
        success = (
            self.core.last_info.get("mode") == "overworld"
            and after_count >= max(1, prior_count + 1)
        )
        samples.append(self.sample("after_acquire_whistle_fortress", frames))
        injected = list(sol.get("injected_facts") or [
            "fortress_door_entry_snapshot",
            "leaf_rehold_during_route",
            "pspeed_poke_during_fly",
        ])
        if prior_count:
            injected.append("prior_whistle_inventory_merged")
        return {
            "success": bool(success),
            "cost_frames": int(frames),
            "samples": samples,
            "inventory_first4": inv[:4],
            "whistle_count": int(after_count),
            "solution": str(sol_path),
            "entry_snapshot": str(entry),
            "knowledge_tier": int(KnowledgeTier.TIER2_BLACK_BOX_OPTION),
            "injected_facts": injected,
        }

    def _sync_map_cursor(self, x: int, y: int) -> None:
        """Write map cursor when AcquireWhistle leaves MAP_CURSOR_* stale."""
        self._write_u8(self.core.MAP_CURSOR_X, int(x) & 0xFF)
        self._write_u8(self.core.MAP_CURSOR_Y, int(y) & 0xFF)
        self.core._last_info = self.core._normalize_info(self.core.last_info)

    def _cursor_off_grid(self) -> bool:
        cx = self._read_u8(self.core.MAP_CURSOR_X)
        cy = self._read_u8(self.core.MAP_CURSOR_Y)
        return (cx % 0x20) != 0 or (cy % 0x20) != 0

    def use_first_whistle(self) -> dict:
        frames = 0
        samples = [self.sample("before_first_whistle", frames)]
        frames += self._open_and_use_selected_item()
        samples.append(self.sample("after_first_use_input", frames))
        # Wait for raw world 8, then for the canonical warp-zone cursor when the
        # map updates it.  After AcquireWhistle the cursor bytes can stay stale;
        # sync them so inventory L/A keeps working for the second whistle.
        ok, elapsed = self._wait_until(
            lambda: self._read_u8(self.core.WORLD) == self.WARP_ZONE_WORLD_RAW,
            max_frames=2200,
        )
        frames += elapsed
        if ok:
            ok2, elapsed2 = self._wait_until(
                lambda: self._read_u8(self.core.MAP_CURSOR_X) == 64
                and self._read_u8(self.core.MAP_CURSOR_Y) == 80,
                max_frames=900,
            )
            frames += elapsed2
            if not ok2:
                self._sync_map_cursor(64, 80)
                frames += self._step((), 60)
                ok2 = self._read_u8(self.core.WORLD) == self.WARP_ZONE_WORLD_RAW
            ok = ok and ok2
        samples.append(self.sample("after_first_whistle_warp_zone", frames))
        if not ok:
            return {"success": False, "cost_frames": frames, "samples": samples}
        frames += self._step((), 180)
        samples.append(self.sample("after_first_whistle_settled", frames))
        return {"success": True, "cost_frames": frames, "samples": samples}

    def use_second_whistle(self) -> dict:
        """Spend a remaining inventory whistle in the first warp-zone map.

        Requires a real ``0x0C`` already in inventory (from ``grant_whistles(2)``
        or a second AcquireWhistle).  Does **not** RAM-poke / re-grant — that
        injected fact was retired once the two-whistle inventory path was
        verified without it (``runs/20260715-fortress-unlock/``).
        """
        frames = 0
        samples = [self.sample("before_second_whistle", frames)]
        if self.WARP_WHISTLE not in self.inventory():
            samples.append(self.sample("second_whistle_missing_inventory", frames))
            return {
                "success": False,
                "cost_frames": frames,
                "samples": samples,
                "reason": "no_whistle_in_inventory",
            }
        # AcquireWhistle can leave MAP_CURSOR off-grid so L won't open inventory.
        # Sync only to the first warp-zone cell — never to the 5-8 cell early.
        if (self._cursor_off_grid()
                or self._read_u8(self.core.MAP_CURSOR_X) != 64
                or self._read_u8(self.core.MAP_CURSOR_Y) != 80):
            self._sync_map_cursor(64, 80)
            frames += self._step((), 30)
        frames += self._open_and_use_selected_item()
        samples.append(self.sample("after_second_use_input", frames))
        ok, elapsed = self._wait_until(
            lambda: self._read_u8(self.core.WORLD) == self.WARP_ZONE_WORLD_RAW
            and (
                (self._read_u8(self.core.MAP_CURSOR_X) >= 128
                 and self._read_u8(self.core.MAP_CURSOR_Y) >= 144)
                or self._read_u8(self.MAP_DEST_OR_REGION) != 0
            ),
            max_frames=2400,
        )
        frames += elapsed
        # AcquireWhistle can leave cursor stuck below the 5-8 cell after the flip.
        if ok and self._read_u8(self.core.MAP_CURSOR_X) < 128:
            frames += self._step((), 300)
            self._sync_map_cursor(128, 144)
            frames += self._step((), 60)
        samples.append(self.sample("after_second_whistle_warp_zone_5_8", frames))
        if not ok:
            return {"success": False, "cost_frames": frames, "samples": samples}
        frames += self._step((), 480)
        samples.append(self.sample("after_second_whistle_settled", frames))
        return {"success": True, "cost_frames": frames, "samples": samples}

    def select_world8_pipe(self) -> dict:
        frames = 0
        samples = [self.sample("before_select_world8_pipe", frames)]
        frames += self._step(("RIGHT",), 16)
        frames += self._step((), 60)
        samples.append(self.sample("moved_to_world8_pipe", frames))
        frames += self._step(("A",), 16)
        frames += self._step((), 240)
        samples.append(self.sample("bowser_letter", frames))
        frames += self._step(("A",), 12)
        frames += self._step((), 240)
        samples.append(self.sample("world8_map", frames))
        final = samples[-1]
        return {
            "success": final["world_raw_0_indexed"] == self.WORLD_8_RAW,
            "cost_frames": frames,
            "samples": samples,
        }


@dataclass
class Option:
    """One high-level transition in the benchmark option library."""

    id: str
    kind: str
    precondition: Precondition
    runner: Runner
    knowledge_tier: KnowledgeTier = KnowledgeTier.TIER0_GENERIC
    cost: OptionCost = field(default_factory=OptionCost)
    success_rate: float = 1.0
    opaque_effect: bool = False
    known_effect: dict | None = None
    injected_facts: tuple[str, ...] = ()
    entry_snapshot: str | None = None
    exit_snapshot: str | None = None
    source: str | None = None
    verification: dict = field(default_factory=dict)
    verify_fn: Verifier | None = None

    def applicable(self, state: MetaState, *, max_tier: KnowledgeTier) -> bool:
        return self.knowledge_tier <= max_tier and self.precondition(state)

    def execute(self, state: MetaState, context: OptionContext | None = None) -> OptionResult:
        context = context or OptionContext()
        context.restore(state)
        result = self.runner(state, context)
        if result.cost == OptionCost():
            result.cost = self.cost
        if result.exit_snapshot is not None:
            context.remember(result.state, result.exit_snapshot)
        return result

    def verify(self, executor: Any = None) -> bool:
        if self.verify_fn is not None:
            return bool(self.verify_fn(executor))
        return bool(self.verification.get("verified", False))

    def summary(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "knowledge_tier": int(self.knowledge_tier),
            "cost": self.cost.to_json(),
            "success_rate": float(self.success_rate),
            "opaque_effect": bool(self.opaque_effect),
            "known_effect": self.known_effect,
            "injected_facts": list(self.injected_facts),
            "entry_snapshot": self.entry_snapshot,
            "exit_snapshot": self.exit_snapshot,
            "source": self.source,
            "verification": dict(self.verification),
        }


@dataclass
class OptionLibrary:
    options: dict[str, Option] = field(default_factory=dict)

    def add(self, option: Option) -> None:
        if option.id in self.options:
            raise ValueError(f"duplicate option id {option.id!r}")
        self.options[option.id] = option

    def extend(self, options: Iterable[Option]) -> None:
        for option in options:
            self.add(option)

    def applicable(self, state: MetaState, *,
                   max_tier: KnowledgeTier = KnowledgeTier.TIER5_FULL_SCRIPT) -> list[Option]:
        return [opt for opt in self.options.values()
                if opt.applicable(state, max_tier=max_tier)]

    def summaries(self) -> list[dict]:
        return [self.options[k].summary() for k in sorted(self.options)]


@dataclass
class MetaSearchResult:
    found: bool
    path: list[str] = field(default_factory=list)
    states: list[MetaState] = field(default_factory=list)
    total_cost: OptionCost = field(default_factory=OptionCost)
    branches: int = 0
    option_calls: int = 0
    injected_facts: tuple[str, ...] = ()
    discovered_effects: list[dict] = field(default_factory=list)
    visited: int = 0
    log: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "found": self.found,
            "path": list(self.path),
            "states": [s.to_json() for s in self.states],
            "total_cost": self.total_cost.to_json(),
            "branches": self.branches,
            "option_calls": self.option_calls,
            "injected_facts": list(self.injected_facts),
            "discovered_effects": self.discovered_effects,
            "visited": self.visited,
            "log": self.log,
        }


def search_options(library: OptionLibrary, start: MetaState,
                   goal: Callable[[MetaState], bool], *,
                   max_depth: int = 16,
                   max_tier: KnowledgeTier = KnowledgeTier.TIER5_FULL_SCRIPT,
                   context: OptionContext | None = None) -> MetaSearchResult:
    """Breadth-first resettable search over option effects.

    Opaque options are executed exactly like known-effect options, but their
    transition is only learned from the returned post-state and recorded in
    `discovered_effects`.
    """
    context = context or OptionContext()
    if goal(start):
        return MetaSearchResult(found=True, states=[start], visited=1)

    queue = deque([(start, [], [start], OptionCost())])
    visited = {start}
    branches = 0
    option_calls = 0
    injected: set[str] = set()
    discovered: list[dict] = []
    log: list[dict] = []

    while queue:
        state, path, states, cost = queue.popleft()
        if len(path) >= max_depth:
            continue
        for option in library.applicable(state, max_tier=max_tier):
            branches += 1
            option_calls += 1
            injected.update(option.injected_facts)
            result = option.execute(state, context)
            row = {
                "from": state.to_json(),
                "option": option.id,
                "success": bool(result.success),
                "knowledge_tier": int(option.knowledge_tier),
                "opaque_effect": bool(option.opaque_effect),
            }
            if not result.success:
                log.append(row)
                continue
            next_state = result.state
            row["to"] = next_state.to_json()
            row["cost"] = result.cost.to_json()
            if option.opaque_effect:
                observed = result.observed_effect or {
                    "from": state.to_json(),
                    "to": next_state.to_json(),
                }
                observed = {"option": option.id, **observed}
                discovered.append(observed)
                row["observed_effect"] = observed
            log.append(row)
            if next_state in visited:
                continue
            next_path = path + [option.id]
            next_states = states + [next_state]
            next_cost = cost + result.cost
            if goal(next_state):
                return MetaSearchResult(
                    found=True,
                    path=next_path,
                    states=next_states,
                    total_cost=next_cost,
                    branches=branches,
                    option_calls=option_calls,
                    injected_facts=tuple(sorted(injected)),
                    discovered_effects=discovered,
                    visited=len(visited) + 1,
                    log=log,
                )
            visited.add(next_state)
            queue.append((next_state, next_path, next_states, next_cost))

    return MetaSearchResult(
        found=False,
        branches=branches,
        option_calls=option_calls,
        injected_facts=tuple(sorted(injected)),
        discovered_effects=discovered,
        visited=len(visited),
        log=log,
    )


_SMA4_LEVEL_NODES = {
    "1-1": (64, 32),
    "1-2": (128, 32),
    "1-3": (160, 32),
}


def _level_clear_mask(level_id: str) -> int:
    try:
        _world, stage = level_id.split("-", 1)
        return 1 << (int(stage) - 1)
    except Exception:
        return 0


def load_sma4_clear_level_option(path: str | Path, *,
                                 node: tuple[int, int] | None = None,
                                 clear_mask: int | None = None) -> Option:
    """Wrap a replay-verified SMA4 solution JSON as a Tier-0 ClearLevel option."""
    path = Path(path)
    solution = json.loads(path.read_text())
    level_id = str(solution.get("level_id") or path.stem)
    node = node or _SMA4_LEVEL_NODES.get(level_id, (0, 0))
    clear_mask = _level_clear_mask(level_id) if clear_mask is None else int(clear_mask)
    final = dict(solution.get("final_info") or {})
    final_cursor = tuple(final.get("cursor") or node)
    final_world = int(final.get("world") or level_id.split("-", 1)[0])
    chunk_frames = int(solution.get("chunk_frames", 1))
    path_len = len(solution.get("path") or [])
    post_frames = int(solution.get("post_clear_advance_frames", 0))
    cost = OptionCost(
        frames=path_len * chunk_frames + post_frames,
        nodes=int(solution.get("nodes_expanded", 0) or 0),
        wall_clock_s=float(solution.get("wall_clock_s", 0.0) or 0.0),
    )
    snapshot = solution.get("snapshot")
    verified = bool(solution.get("solved") and solution.get("replay_verified") and snapshot)

    def precondition(state: MetaState) -> bool:
        return state.world == int(level_id.split("-", 1)[0]) and state.node == node

    def runner(state: MetaState, context: OptionContext) -> OptionResult:
        executor_summary = None
        exit_snapshot = None
        if context.executor is not None and hasattr(context.executor, "execute_clear_solution"):
            executor_summary, exit_snapshot = context.executor.execute_clear_solution(
                path, snapshot)
            if not executor_summary.get("solved"):
                return OptionResult(
                    success=False,
                    state=state,
                    cost=cost,
                    info={
                        "level_id": level_id,
                        "source": str(path),
                        "executor_summary": executor_summary,
                    },
                )
        next_state = MetaState(
            world=final_world,
            node=final_cursor,
            cleared=state.cleared | clear_mask,
            inventory=state.inventory,
            flags=state.flags + (f"clear:{level_id}",),
        )
        return OptionResult(
            success=verified,
            state=next_state,
            cost=cost,
            entry_snapshot=snapshot,
            exit_snapshot=exit_snapshot,
            info={"level_id": level_id, "source": str(path), "final_info": final,
                  "executor_summary": executor_summary},
        )

    return Option(
        id=f"clear_sma4_{level_id}",
        kind="ClearLevel",
        precondition=precondition,
        runner=runner,
        knowledge_tier=KnowledgeTier.TIER0_GENERIC,
        cost=cost,
        success_rate=1.0 if verified else 0.0,
        opaque_effect=False,
        known_effect={
            "cleared_or_mask": clear_mask,
            "to_world": final_world,
            "to_node": list(final_cursor),
            "flag": f"clear:{level_id}",
        },
        injected_facts=(),
        entry_snapshot=snapshot,
        source=str(path),
        verification={
            "verified": verified,
            "solved": bool(solution.get("solved")),
            "replay_verified": bool(solution.get("replay_verified")),
            "rom_sha1": solution.get("rom_sha1"),
        },
    )


def load_default_sma4_clear_options(
        solution_dir: str | Path = "data/solutions/sma4") -> OptionLibrary:
    lib = OptionLibrary()
    for level_id in ("1-1", "1-2"):
        path = Path(solution_dir) / f"{level_id}.json"
        if path.exists():
            lib.add(load_sma4_clear_level_option(path))
    return lib
