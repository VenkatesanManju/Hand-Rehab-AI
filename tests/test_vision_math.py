"""
Unit tests for vision joint angle math, ROM, and repetition counter logic (Phase 5)
Verifies mathematical correctness of 3D joint angle formulas and state-machine transitions.
"""

import math
import pytest
import numpy as np


def compute_angle_3d(a, b, c):
    """
    Computes angle ABC with vertex at B in degrees.
    a, b, c: arrays or dicts with x, y, z
    """
    ba = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]], dtype=float)

    dot = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine = np.clip(dot / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def compute_rom(mean_pip_angle):
    """
    Computes flexion Range of Motion (0 to 100 degrees) from extension angle (180 to 50).
    """
    return float(np.clip(180.0 - mean_pip_angle, 0.0, 100.0))


class RepCounterSimulator:
    def __init__(self, extend_thresh=140.0, flex_thresh=85.0):
        self.extend_thresh = extend_thresh
        self.flex_thresh = flex_thresh
        self.state = "EXTENDED"
        self.rep_count = 0
        self.has_curled = False

    def update(self, angle):
        if angle <= self.flex_thresh:
            self.state = "FLEXED"
            self.has_curled = True
        elif angle >= self.extend_thresh:
            if self.state == "FLEXED" and self.has_curled:
                self.rep_count += 1
                self.has_curled = False
            self.state = "EXTENDED"
        return self.rep_count


def test_angle_calculation_orthogonal():
    # Perpendicular vectors (e.g. [1, 0, 0] and [0, 1, 0] with origin [0, 0, 0])
    a = [1.0, 0.0, 0.0]
    b = [0.0, 0.0, 0.0]
    c = [0.0, 1.0, 0.0]
    angle = compute_angle_3d(a, b, c)
    assert pytest.approx(angle, abs=1e-3) == 90.0


def test_angle_calculation_collinear():
    # Straight line (180 degrees)
    a = [-1.0, 0.0, 0.0]
    b = [0.0, 0.0, 0.0]
    c = [1.0, 0.0, 0.0]
    angle = compute_angle_3d(a, b, c)
    assert pytest.approx(angle, abs=1e-3) == 180.0


def test_angle_calculation_acute():
    # 45 degrees
    a = [1.0, 1.0, 0.0]
    b = [0.0, 0.0, 0.0]
    c = [1.0, 0.0, 0.0]
    angle = compute_angle_3d(a, b, c)
    assert pytest.approx(angle, abs=1e-3) == 45.0


def test_rom_bounds():
    # Open hand (180 deg) -> ROM = 0
    assert compute_rom(180.0) == 0.0
    # Moderate flexion (120 deg) -> ROM = 60
    assert compute_rom(120.0) == 60.0
    # Full fist curl (70 deg) -> ROM = 100 (clipped)
    assert compute_rom(70.0) == 100.0


def test_rep_counter_lifecycle():
    counter = RepCounterSimulator(extend_thresh=140.0, flex_thresh=85.0)

    # Sequence of angle movements over time:
    # 1. Start open
    angles = [170.0, 160.0, 145.0]
    for a in angles:
        counter.update(a)
    assert counter.rep_count == 0

    # 2. Curl fingers into fist (cross flex_thresh 85)
    angles_curl = [120.0, 90.0, 75.0, 70.0]
    for a in angles_curl:
        counter.update(a)
    assert counter.rep_count == 0
    assert counter.state == "FLEXED"

    # 3. Open fingers back (cross extend_thresh 140)
    angles_open = [90.0, 130.0, 150.0, 170.0]
    for a in angles_open:
        counter.update(a)
    # Rep completed!
    assert counter.rep_count == 1
    assert counter.state == "EXTENDED"

    # 4. Partial movement that doesn't reach flex_thresh shouldn't increment rep
    partial = [135.0, 100.0, 110.0, 150.0]
    for a in partial:
        counter.update(a)
    assert counter.rep_count == 1
