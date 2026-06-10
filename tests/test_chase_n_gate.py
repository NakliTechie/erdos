"""Regression: raw target hits that fail the float three-audit must not
stop the chase ladder or be reported tied/exceeded (scripts/chase_n.py).

Batch F band-B4, 2026-06-10: at n=71/72 the subset stage returned configs
tying the target (286/291 edges) from t3xt3c1/t2xt4c1 closure ambients with
an inherent close pair at 0.0851 < MIN_SEP = 0.2. The driver early-stopped,
wrote summary "tied": true alongside audit_passed: false, and a fresh n=72
retry burned its budget re-stopping on the same bad config in 1.7 s. The
lattice offset (-1, -3, 3, 1) reproduces that exact pair (|float| =
0.08514578448732...).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import chase_n  # noqa: E402
from udg.audit import MIN_SEP, audit  # noqa: E402
from udg.mlgraph import MLConfig, to_float, tri_patch  # noqa: E402
from udg.subsetsearch import edge_count  # noqa: E402

CLOSE_OFFSET = (-1, -3, 3, 1)  # |float| = 0.0851... -- the n=71/72 pair


def _good_pts():
    """tri_patch(2): 6 points, 9 edges, min_sep 1.0 -- passes the audit."""
    return [tuple(p) for p in tri_patch(2).points]


def _bad_pts():
    """tri_patch(2) plus a 7th lattice point 0.0851 from a patch vertex:
    same raw edge count, min_sep < MIN_SEP -- fails the audit."""
    pts = _good_pts()
    base = pts[0]
    pts.append(tuple(base[k] + CLOSE_OFFSET[k] for k in range(4)))
    return pts


def test_close_offset_reproduces_the_n71_pair():
    P = to_float(MLConfig([(0, 0, 0, 0), CLOSE_OFFSET]))
    d = float(np.linalg.norm(P[1] - P[0]))
    assert 0 < d < MIN_SEP
    assert abs(d - 0.08514578448732) < 1e-9


def test_audit_passes_gate():
    assert chase_n.audit_passes(_good_pts())
    rep = audit(to_float(MLConfig(_bad_pts())))
    assert not rep.passed and rep.min_sep < MIN_SEP
    assert not chase_n.audit_passes(_bad_pts())


def test_subset_task_keeps_searching_past_an_audit_failing_tie(monkeypatch):
    bad = np.array(_bad_pts(), dtype=np.int64)
    target = edge_count(bad)  # the fake result TIES the target on raw count
    calls = []

    def fake_ils(amb, k, seed=0, **kw):
        calls.append(seed)
        return edge_count(bad), bad

    monkeypatch.setattr(chase_n, "subset_ils", fake_ils)
    t0 = time.time()
    _name, m, _pts, ok = chase_n._subset_task(
        ("amb", bad, len(bad), target, 1, 0.02, ""))
    assert not ok                    # an audit-failing tie is NOT a stop
    assert m == target               # ...even though the raw count ties
    assert len(calls) >= 2           # the restart ladder kept searching
    assert time.time() - t0 >= 0.5   # ...until its wall-clock slice ended


def test_subset_task_stops_on_an_audit_passing_hit(monkeypatch):
    good = np.array(_good_pts(), dtype=np.int64)
    target = edge_count(good)
    calls = []

    def fake_ils(amb, k, seed=0, **kw):
        calls.append(seed)
        return edge_count(good), good

    monkeypatch.setattr(chase_n, "subset_ils", fake_ils)
    _name, m, _pts, ok = chase_n._subset_task(
        ("amb", good, len(good), target, 1, 0.02, ""))
    assert ok and m == target
    assert len(calls) == 1           # a valid hit still stops immediately


def test_ladder_only_stops_on_audited_hits():
    # global_best is (edges, pts, label, P_float, audit_ok)
    bad_claim = (286, _bad_pts(), "subset:t3xt3c1", None, False)
    assert not chase_n.ladder_done(bad_claim, target=286)  # the n=71 shape
    good_claim = (286, _good_pts(), "subset:w49c1", None, True)
    assert chase_n.ladder_done(good_claim, target=286)
    assert not chase_n.ladder_done(None, target=286)
    assert not chase_n.ladder_done((285, None, "x", None, True), target=286)


def test_tied_requires_audit_and_certification():
    # the exact n=71/72 summary shape: raw tie, certified, audit FAILED
    v = chase_n.final_verdict(286, 286, {"audit_passed": False, "certified": True})
    assert v == {"tied": False, "exceeded": False}
    v = chase_n.final_verdict(286, 286, {"audit_passed": True, "certified": False})
    assert v == {"tied": False, "exceeded": False}
    v = chase_n.final_verdict(286, 286, {"audit_passed": True, "certified": True})
    assert v == {"tied": True, "exceeded": False}
    v = chase_n.final_verdict(287, 286, {"audit_passed": True, "certified": True})
    assert v == {"tied": False, "exceeded": True}
