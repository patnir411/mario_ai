"""Snapshot exactness gate: dump_state/load_state must be bit-exact and side-effect-free.

Beam search clones nodes by snapshotting. If restoring a snapshot doesn't reproduce
the identical future, every search result is invalid. These tests are the canary.
"""
from __future__ import annotations

from conftest import canonical_actions, ram_hash

ACTS = canonical_actions(200)


def _advance(sim, actions):
    for a in actions:
        _obs, _info, done = sim.step(a)
        if done:
            break


def test_dump_load_roundtrip_identical(sim):
    """snap -> advance(M) (hash h1) -> restore -> advance(same M) -> hash == h1."""
    _advance(sim, ACTS[:40])
    snap = sim.snapshot()
    _advance(sim, ACTS[40:90])
    h1 = ram_hash(sim)

    sim.restore(snap)
    _advance(sim, ACTS[40:90])
    h2 = ram_hash(sim)
    assert h1 == h2, "restoring a snapshot did not reproduce the identical future"


def test_dump_has_no_side_effects(sim):
    """Taking a snapshot must not perturb emulator state."""
    _advance(sim, ACTS[:40])
    before = ram_hash(sim)
    _ = sim.snapshot()
    after = ram_hash(sim)
    assert before == after, "dump_state() mutated emulator state"


def test_snapshot_reusable(sim):
    """A single snapshot can be restored multiple times, each reproducing the same future."""
    _advance(sim, ACTS[:40])
    snap = sim.snapshot()

    sim.restore(snap)
    _advance(sim, ACTS[40:70])
    h_first = ram_hash(sim)

    sim.restore(snap)
    _advance(sim, ACTS[40:70])
    h_second = ram_hash(sim)
    assert h_first == h_second, "snapshot not reusable across multiple restores"


def test_independent_snapshots_no_contamination(sim):
    """Two snapshots at different points stay independent under interleaved restores."""
    _advance(sim, ACTS[:30])
    snap_a = sim.snapshot()
    hash_a = ram_hash(sim)

    _advance(sim, ACTS[30:80])
    snap_b = sim.snapshot()
    hash_b = ram_hash(sim)

    # restore A, confirm we're exactly at A's state
    sim.restore(snap_a)
    assert ram_hash(sim) == hash_a, "restoring snapshot A did not return to A"
    # restore B, confirm we're exactly at B's state (A's restore didn't corrupt B)
    sim.restore(snap_b)
    assert ram_hash(sim) == hash_b, "snapshot B contaminated by interleaved restore of A"
