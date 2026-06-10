"""Tests for udg.hinge (hinge-locking toolkit, plan/hinge-design.md).

Synthetic frameworks with KNOWN rigidity/flex structure:

- two unit triangles sharing one vertex  -> 1 internal flex (the hinge);
- a 4-bar linkage (unit rhombus, 4 bars) -> 1 internal flex;
- a braced triangular patch (rhombus ABCD with 5 unit edges) -> rigid (0).

Plus the documented classification of data/udg40_132edges.csv
(DATA_APPENDIX section F): 4 families-of-3, raw-frame family angles
~{0.04, 0.94, 16.87, 34.12} deg, the 0.94 and 34.12 families floating with
targets 0 and arccos(5/6) = 33.557 deg.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from udg.audit import audit
from udg.configio import load_csv
from udg.counting import unit_edges
from udg.hinge import (
    Family,
    candidate_targets,
    classify_families,
    edge_residual,
    family_angle,
    fire_check,
    flex_dimension,
    follow_flex,
    internal_flex_basis,
    lock_family,
    rigidity_matrix,
    rotate_points,
    signed_delta,
)

CSV = Path(__file__).resolve().parents[1] / "data" / "udg40_132edges.csv"
RT3 = np.sqrt(3.0)


# ---------------------------------------------------------------------------
# synthetic frameworks
# ---------------------------------------------------------------------------

def braced_patch() -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Rhombus A,B,C,D = triangular patch of two unit triangles; RIGID."""
    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, RT3 / 2], [1.5, RT3 / 2]])
    E = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    return P, E


def four_bar() -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Unit rhombus with only the 4 perimeter bars; 1 internal flex."""
    P = np.array([[0.0, 0.0], [1.0, 0.0], [1.5, RT3 / 2], [0.5, RT3 / 2]])
    E = [(0, 1), (1, 2), (2, 3), (3, 0)]
    return P, E


def two_triangles(angle2_deg: float = 0.0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Two unit triangles sharing vertex C (index 2); 1 internal flex.

    Triangle 2 (C, D, E) is rigidly rotated by angle2_deg about C, so its
    direction family sits at offset angle2_deg mod 60.
    """
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, RT3 / 2])
    th = np.radians(angle2_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    D = C + R @ np.array([1.0, 0.0])
    Ept = C + R @ np.array([0.5, RT3 / 2])
    P = np.vstack([A, B, C, D, Ept])
    E = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]
    return P, E


TRI2_FAMILY = [(2, 3), (2, 4), (3, 4)]  # triangle-2 edges in two_triangles
SUB_FAMILY = [(1, 2), (1, 3), (2, 3)]   # sub-triangle B,C,D in braced_patch


# ---------------------------------------------------------------------------
# angle helpers
# ---------------------------------------------------------------------------

def test_signed_delta_wraps_circularly():
    assert signed_delta(0.5, 59.8, 60.0) == pytest.approx(0.7)
    assert signed_delta(59.8, 0.5, 60.0) == pytest.approx(-0.7)
    assert signed_delta(34.12, 33.557, 60.0) == pytest.approx(0.563)
    assert signed_delta(30.0, 0.0, 60.0) == pytest.approx(30.0)  # half-period


def test_family_angle_mod60():
    P, E = braced_patch()
    # patch edges point at {0, 60, 120} deg -> family angle 0 mod 60
    assert min(family_angle(P, E), 60.0 - family_angle(P, E)) < 1e-9
    P2, _ = two_triangles(10.0)
    assert family_angle(P2, TRI2_FAMILY) == pytest.approx(10.0, abs=1e-9)


def test_candidate_targets_are_the_three_mod60_lattice_angles():
    c = candidate_targets()
    assert np.allclose(
        c, [0.0, np.degrees(np.arcsin(1 / np.sqrt(12))), np.degrees(np.arccos(5 / 6))]
    )


# ---------------------------------------------------------------------------
# rigidity matrix / flex dimension
# ---------------------------------------------------------------------------

def test_rigidity_matrix_shape_and_trivial_motions():
    P, E = braced_patch()
    R = rigidity_matrix(P, E)
    assert R.shape == (len(E), 2 * len(P))
    n = len(P)
    tx = np.tile([1.0, 0.0], n)
    ty = np.tile([0.0, 1.0], n)
    rot = np.empty(2 * n)
    rot[0::2] = -P[:, 1]
    rot[1::2] = P[:, 0]
    for v in (tx, ty, rot):  # trivial motions are always first-order flexes
        assert np.abs(R @ v).max() < 1e-12


def test_flex_dimension_flexible_frameworks():
    P, E = two_triangles(0.0)
    assert flex_dimension(P, E) == 1   # hinge at the shared vertex
    P4, E4 = four_bar()
    assert flex_dimension(P4, E4) == 1  # the 4-bar linkage flex


def test_flex_dimension_rigid_patch():
    P, E = braced_patch()
    assert flex_dimension(P, E) == 0
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, RT3 / 2]])
    assert flex_dimension(tri, [(0, 1), (0, 2), (1, 2)]) == 0


def test_internal_flex_basis_consistent_and_clean():
    for (P, E), dim in [
        (two_triangles(7.0), 1),
        (four_bar(), 1),
        (braced_patch(), 0),
    ]:
        B = internal_flex_basis(P, E)
        assert B.shape[1] == dim == flex_dimension(P, E)
        if dim:
            # orthonormal, in null(R), orthogonal to the trivial motions
            assert np.allclose(B.T @ B, np.eye(dim), atol=1e-12)
            assert np.abs(rigidity_matrix(P, E) @ B).max() < 1e-9
            n = len(P)
            tx = np.tile([1.0, 0.0], n)
            ty = np.tile([0.0, 1.0], n)
            rot = np.empty(2 * n)
            rot[0::2] = -(P[:, 1] - P[:, 1].mean())
            rot[1::2] = P[:, 0] - P[:, 0].mean()
            for v in (tx, ty, rot):
                assert np.abs(v @ B).max() < 1e-9


# ---------------------------------------------------------------------------
# lock_family (strategy a: rotate-and-project homotopy)
# ---------------------------------------------------------------------------

def test_lock_family_recovers_rotated_subtriangle():
    P, E = braced_patch()
    # deliberately rotate sub-triangle {B,C,D} by 1 deg about its centroid:
    # breaks edges A-B and A-C, family angle goes 0 -> 1 deg.
    P_rot = rotate_points(P, 1.0, indices=[1, 2, 3])
    assert family_angle(P_rot, SUB_FAMILY) == pytest.approx(1.0, abs=1e-9)
    assert edge_residual(P_rot, E) > 1e-8  # genuinely broken

    res = lock_family(P_rot, E, SUB_FAMILY, target=0.0, n_increments=10)
    assert res.stop_reason == "target"
    assert res.converged
    # hinge vertices = family verts shared with non-family edges = {B, C}
    assert res.diagnostics["hinge_vertices"] == [1, 2]
    assert min(res.family_angle, 60.0 - res.family_angle) < 1e-3
    assert res.residual < 1e-24  # exact patch recovered (GN-clean)
    assert res.min_sep > 0.9
    rep = audit(res.P)
    assert rep.passed and rep.n_edges == 5
    assert fire_check(res.P).n_unit == 5  # all original edges restored


def test_lock_family_noop_at_target_and_accepts_family_object():
    P, E = braced_patch()
    fam = Family(
        index=0, edges=SUB_FAMILY, n_edges=3,
        vertices=[1, 2, 3], directions=[0.0, 60.0, 120.0],
        mean_angle=0.0, mean_angle_raw=0.0, target=0.0, offset=0.0, locked=True,
    )
    res = lock_family(P, E, fam)  # target defaults to fam.target
    assert res.converged and res.stop_reason == "target"
    assert res.diagnostics["n_steps"] == 0
    assert np.allclose(res.P, P)  # untouched: already at target


# ---------------------------------------------------------------------------
# follow_flex (strategy b: predictor-corrector along the flex)
# ---------------------------------------------------------------------------

def test_follow_flex_drives_hinge_to_target_and_extra_edge_fires():
    # Triangle 2 hinged at C, 10 deg off triangle 1's family. Coincidences
    # depend only on RELATIVE angles, and absolute family angles are gauge-
    # coupled: the internal flex (orthogonal to global rotation) counter-
    # rotates the rest of the framework while the target family turns. On
    # this mirror-symmetric linkage the split is exactly 50/50, so driving
    # triangle 2 to 5 deg brings triangle 1 to 5 deg too -> relative angle 0,
    # and the NEW unit coincidence B-D fires (rhombus closes): the firing
    # hypothesis in vitro. (Same reason the design doc re-classifies between
    # locks: after a lock the other families have drifted in gauge.)
    P, E = two_triangles(10.0)
    assert fire_check(P).n_unit == 6
    res = follow_flex(P, E, TRI2_FAMILY, target=5.0, angle_tol=1e-9)
    assert res.converged and res.stop_reason == "target"
    assert res.flex_dim == 1
    assert res.family_angle == pytest.approx(5.0, abs=1e-6)
    tri1 = [(0, 1), (0, 2), (1, 2)]
    rel = signed_delta(res.family_angle, family_angle(res.P, tri1))
    assert abs(rel) < 1e-6  # gauge split was exactly symmetric
    assert res.residual < 1e-24
    fc = fire_check(res.P)
    assert fc.n_unit == 7  # 6 constrained edges + the fired B-D coincidence
    assert np.linalg.norm(res.P[1] - res.P[3]) == pytest.approx(1.0, abs=1e-12)
    rep = audit(res.P)     # the fired count survives the three audits
    assert rep.passed and rep.n_edges == 7


def test_follow_flex_reports_flex_death_on_rigid_framework():
    P, E = braced_patch()
    res = follow_flex(P, E, SUB_FAMILY, target=1.0)
    assert not res.converged
    assert res.stop_reason == "flex_death_rigid"
    assert res.flex_dim == 0
    assert np.allclose(res.P, P)  # nothing was moved


# ---------------------------------------------------------------------------
# fire_check
# ---------------------------------------------------------------------------

def test_fire_check_counts_and_near_miss_bands():
    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.001], [0.0, 2.5]])
    fc = fire_check(P)
    assert fc.n == 4 and fc.n_unit == 1  # only (0,1) is unit at 1e-9
    counts = {(lo, hi): c for lo, hi, c in fc.bands}
    assert counts[(1e-4, 1e-2)] == 1     # |1.001 - 1| = 1e-3
    assert counts[(1e-9, 1e-6)] == 0
    assert counts[(1e-6, 1e-4)] == 0
    assert counts[(1e-2, 5e-2)] == 0
    assert fc.closest_nonunit == pytest.approx(1e-3, rel=1e-9)

    Pp, _ = braced_patch()
    fc2 = fire_check(Pp)
    assert fc2.n_unit == 5
    assert all(c == 0 for _, _, c in fc2.bands)  # A-D = sqrt(3), no near-miss


# ---------------------------------------------------------------------------
# classify_families on the real 132-edge config (DATA_APPENDIX section F)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def udg40():
    P = load_csv(CSV)
    E = unit_edges(P)
    assert len(E) == 132
    return P, E


def test_classify_families_reproduces_documented_structure(udg40):
    P, E = udg40
    cls = classify_families(P, E, lock_tol=0.2)
    fams = cls.families
    assert len(fams) == 4

    # documented raw-frame family angles ("offsets" of DATA_APPENDIX F)
    raw = [f.mean_angle_raw for f in fams]
    assert np.allclose(raw, [0.04, 0.94, 16.87, 34.12], atol=0.02)

    # families-of-3: each family spans exactly 3 mod-180 directions
    assert [len(f.directions) for f in fams] == [3, 3, 3, 3]

    # the family edge lists partition the 132 input edges
    assert [f.n_edges for f in fams] == [36, 6, 38, 52]
    all_edges = [tuple(sorted(e)) for f in fams for e in f.edges]
    assert len(all_edges) == 132
    assert set(all_edges) == {tuple(sorted(e)) for e in E}
    assert all(f.vertices for f in fams)

    # locked pattern: 0.04 and 16.87 ML-exact, 0.94 and 34.12 floating
    assert [f.locked for f in fams] == [True, False, True, False]

    # sensible targets: 0.94 -> 0, 34.12 -> arccos(5/6) = 33.557
    assert fams[1].target == pytest.approx(0.0, abs=1e-12)
    assert fams[3].target == pytest.approx(np.degrees(np.arccos(5 / 6)), abs=1e-9)
    assert fams[1].offset == pytest.approx(0.90, abs=0.05)   # aligned-frame delta
    assert fams[3].offset == pytest.approx(0.53, abs=0.05)
    for f in fams:
        assert abs(signed_delta(f.mean_angle, f.target)) == pytest.approx(
            abs(f.offset), abs=1e-12
        )


def test_classify_families_alignment_is_rigid_and_recorded(udg40):
    P, E = udg40
    cls = classify_families(P, E)
    # rotation applied = -best_rotation from lattice_id (gauge alignment)
    assert cls.rotation_deg == pytest.approx(-cls.lattice["best_rotation"])
    assert cls.lattice["n_matched"] == 6  # 6/12 directions ML-matched (doc F)
    # rigid motion: the aligned config still has exactly the 132 edges
    assert len(unit_edges(cls.P)) == 132
    # aligned frame: locked families sit ON candidates (tighter than lock_tol)
    assert abs(signed_delta(cls.families[0].mean_angle, 0.0)) < 0.05
    assert (
        abs(signed_delta(cls.families[2].mean_angle, cls.families[2].target)) < 0.1
    )
