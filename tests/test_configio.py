"""Tests for udg.configio — exact CSV round-trip, known-config load,
audit-json round-trip, RESULTS.md append helper."""

import dataclasses
import json
from pathlib import Path

import numpy as np

from udg.configio import append_results_line, load_csv, save_audit_json, save_csv

DATA = Path(__file__).resolve().parents[1] / "data" / "udg40_132edges.csv"


def test_csv_round_trip_exact(tmp_path):
    rng = np.random.default_rng(42)
    P = rng.uniform(-3.5, 3.5, size=(37, 2))
    p = tmp_path / "cfg.csv"
    save_csv(P, p)
    Q = load_csv(p)
    assert Q.shape == (37, 2)
    assert Q.dtype == np.float64
    # EXACT round-trip: bit-for-bit, no tolerance.
    assert np.allclose(P, Q, atol=0, rtol=0)
    assert np.array_equal(P, Q)


def test_csv_header_matches_known_format(tmp_path):
    P = np.array([[1.0, 2.0], [3.0, -4.0]])
    p = tmp_path / "cfg.csv"
    save_csv(P, p)
    lines = p.read_text().splitlines()
    assert lines[0] == "x,y"
    assert len(lines) == 3
    # scientific notation like the known-good data file
    assert "e+" in lines[1] or "e-" in lines[1]


def test_csv_single_row_round_trip(tmp_path):
    P = np.array([[0.123456789012345678, -9.87654321e-3]])
    p = tmp_path / "one.csv"
    save_csv(P, p)
    Q = load_csv(p)
    assert Q.shape == (1, 2)
    assert np.allclose(P, Q, atol=0, rtol=0)


def test_load_known_config():
    P = load_csv(DATA)
    assert P.shape == (40, 2)
    assert P.dtype == np.float64
    assert np.all(np.isfinite(P))
    # spot-check the first row against the file's literal contents
    assert P[0, 0] == 6.046935704077782248e00
    assert P[0, 1] == -1.291982091865392368e00


def test_save_load_known_config_round_trip(tmp_path):
    P = load_csv(DATA)
    p = tmp_path / "resave.csv"
    save_csv(P, p)
    Q = load_csv(p)
    assert np.allclose(P, Q, atol=0, rtol=0)


@dataclasses.dataclass
class _FakeReport:
    n: int
    n_edges: int
    min_sep: float
    gn_total_residual: float
    passed: bool


def test_audit_json_round_trip(tmp_path):
    # numpy scalar fields on purpose — must serialize as plain python types
    rep = _FakeReport(
        n=np.int64(40),
        n_edges=np.int64(132),
        min_sep=np.float64(0.2857),
        gn_total_residual=np.float64(3.2e-29),
        passed=np.bool_(True),
    )
    p = tmp_path / "audit.json"
    save_audit_json(rep, p)
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    assert d["n"] == 40 and type(d["n"]) is int
    assert d["n_edges"] == 132 and type(d["n_edges"]) is int
    assert d["min_sep"] == 0.2857 and type(d["min_sep"]) is float
    assert d["gn_total_residual"] == 3.2e-29
    assert d["passed"] is True


def test_append_results_line(tmp_path):
    p = tmp_path / "RESULTS.md"
    append_results_line(p, "## run 1")  # creates the file
    append_results_line(p, "- n=40 seed=0 best=132\n")  # trailing newline normalized
    text = p.read_text(encoding="utf-8")
    assert text == "## run 1\n- n=40 seed=0 best=132\n"
