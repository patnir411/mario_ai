"""Waypoint / sub-goal scoring for non-linear levels (maze castles 4-4/7-4/8-4, warps).

Pure-x reward loops in mazes and can't credit going up to a warp zone or entering a pipe.
A WaypointTracker rewards progress toward an ordered list of sub-goals and advances when
each is reached, so search is pulled along the intended route instead of greedy-right.

Waypoint spec (data/waypoints.json), per level a list of:
  {"type": "x", "x": N}                 reach level-x N
  {"type": "y", "y": N, "up": true}     reach screen-y N (up = decreasing y; vines)
  {"type": "enter_pipe", "x": N}        be at x≈N (then DOWN enters; area/level changes)
Only hand-authored for levels where pure-x fails; everything else uses pure-x.
"""
from __future__ import annotations

WP_BONUS = 6000.0   # reward for each waypoint already cleared (dominates within-wp shaping)
REACH_TOL = 10


class WaypointTracker:
    def __init__(self, spec):
        self.spec = spec or []

    def score(self, info: dict, wp_idx: int, x_start: int, weights):
        """Return (new_wp_idx, score) for a candidate state."""
        x = int(info.get("x_pos", 0))
        y = int(info.get("y_pos", 0))
        base = WP_BONUS * wp_idx
        if wp_idx >= len(self.spec):                     # all sub-goals done -> pure-x
            return wp_idx, base + weights.progress * (x - x_start)

        wp = self.spec[wp_idx]
        t = wp.get("type", "x")
        if t == "x":
            target = wp["x"]
            if x >= target - REACH_TOL:
                return wp_idx + 1, base + WP_BONUS
            return wp_idx, base + weights.progress * x
        if t == "enter_pipe":
            target = wp["x"]
            # being at the pipe x is the sub-goal; the actual entry shows up as a level/area
            # change which search_from_state detects as "cleared" (returns immediately).
            if abs(x - target) <= REACH_TOL + 6:
                return wp_idx + 1, base + WP_BONUS
            return wp_idx, base + weights.progress * min(x, target)
        if t == "y":
            target = wp["y"]
            up = wp.get("up", True)
            reached = (y <= target) if up else (y >= target)
            if reached:
                return wp_idx + 1, base + WP_BONUS
            # shape toward the target height while keeping x progress
            climb = (-y if up else y)
            return wp_idx, base + weights.progress * (x - x_start) + 4.0 * climb
        return wp_idx, base + weights.progress * (x - x_start)
