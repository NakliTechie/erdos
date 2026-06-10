"""Integration regression tests for DATA_APPENDIX.md sections A-D.

Covers every measured number in sections A-D that is NOT already asserted by
the module unit tests:

- tests/test_counting.py already covers: section A grids m=5,8,12,17
  (best d^2 + count only), section B 50x50 top-5 counts + ratio-vs-r2 +
  r2(25/65/325), section C top-5 norm counts.
- tests/test_bisector.py already covers: all four section D n=1600 rows
  (square / tri / random / perturbed grid).

Added here:
- Section A: larger grid rows m=24 (fast) and m=34/48/60 (@pytest.mark.slow,
  per contract), the printed U/n and U/n^(4/3) columns for all 8 rows, and
  the implied-c growth fit (exp2_structure.py formula) 0.237 -> 0.529,
  still rising at n=3600.
- Section B: the r2 column values (incl. the 425/85/125 rows not covered by
  the module tests), the n*r2/2 prediction, and the printed ratios from the
  stated counts.
- Section C: the per-n densities 8.85 (tri) vs 7.07 (square) and the
  tri-beats-square ~1.25-1.5x factor.
- Section D: the grid-scaling local exponents 2.601 (n=196) -> 2.535
  (n=3136), decreasing toward 2.5.

Sections E/F are covered by tests/test_search.py + tests/test_known_config.py
and are deliberately skipped here.

Float columns are asserted via f-string formatting at the appendix's printed
precision (exp1/exp2 print with .3f/.4f), which is exactly how the appendix
values were produced.
"""

import math

import pytest

from udg.bisector import bisector_energy_int
from udg.counting import popular, r2


def square_grid(m: int) -> list[tuple[int, int]]:
    return [(x, y) for x in range(m) for y in range(m)]


# Section A table, verbatim: m -> (n, best d^2, U(n), "U/n", "U/n^(4/3)")
APPENDIX_A = {
    5: (25, 5, 48, "1.920", "0.6566"),
    8: (64, 5, 168, "2.625", "0.6563"),
    12: (144, 25, 456, "3.167", "0.6042"),
    17: (289, 25, 1136, "3.931", "0.5945"),
    24: (576, 65, 2832, "4.917", "0.5909"),
    34: (1156, 65, 6672, "5.772", "0.5499"),
    48: (2304, 325, 15864, "6.885", "0.5213"),
    60: (3600, 325, 28200, "7.833", "0.5111"),
}


def _check_appendix_A_row(m: int) -> None:
    n_exp, d2, count, un, un43 = APPENDIX_A[m]
    n, top = popular(square_grid(m), topk=5)
    assert n == n_exp == m * m
    assert top[0] == (d2, count)
    # printed columns (exp1_grid.py prints {:.3f} and {:.4f})
    assert f"{count / n:.3f}" == un
    assert f"{count / n ** (4 / 3):.4f}" == un43


# ------------------------------------------------- §A larger grid rows

def test_grid_m24_row_appendix_A():
    _check_appendix_A_row(24)


@pytest.mark.slow
@pytest.mark.parametrize("m", [34, 48, 60])
def test_grid_large_rows_appendix_A(m):
    _check_appendix_A_row(m)


# -------------------------------------- §A printed columns + growth fit

def test_appendix_A_table_columns():
    # U/n and U/n^(4/3) columns recomputed from the stated (n, U) pairs;
    # the (n, U) pairs themselves are regression-checked by the popular()
    # tests above and in tests/test_counting.py.
    for n, _, u, un, un43 in APPENDIX_A.values():
        assert f"{u / n:.3f}" == un
        assert f"{u / n ** (4 / 3):.4f}" == un43


def test_growth_fit_implied_c_appendix_A():
    # exp2_structure.py block 3: U(n) = n^(1 + c/loglog n)  =>
    #   c = log(U/n) * log(log n) / log n
    # Appendix: "Implied c ...: 0.237 -> 0.529, still rising at n=3600."
    cs = []
    for n, _, u, _, _ in APPENDIX_A.values():
        cs.append(math.log(u / n) * math.log(math.log(n)) / math.log(n))
    assert round(cs[0], 3) == 0.237
    assert round(cs[-1], 3) == 0.529
    # "still rising at n=3600": strictly increasing across all 8 rows
    assert all(a < b for a, b in zip(cs, cs[1:]))


# --------------------------------------------- §B r2 column + prediction

# Section B rows verbatim: d^2 -> (count, r2, "ratio")
APPENDIX_B = {
    325: (17680, 24, "0.589"),
    65: (16144, 16, "0.807"),
    425: (15640, 24, "0.521"),
    85: (15440, 16, "0.772"),
    125: (14688, 16, "0.734"),
}


def test_appendix_B_r2_prediction_and_ratio():
    # tests/test_counting.py already checks the top-5 counts and the rounded
    # ratios against a live popular() run; here we pin the r2 COLUMN values
    # themselves (425/85/125 are not covered there) and the n*r2/2
    # interior-point prediction arithmetic from the stated counts.
    n = 2500  # 50x50 grid
    for d2, (count, r2_exp, ratio) in APPENDIX_B.items():
        assert r2(d2) == r2_exp
        pred = n * r2(d2) // 2
        assert count < pred                      # ratio < 1 = boundary loss
        assert f"{2 * count / (n * r2(d2)):.3f}" == ratio


# ------------------------------------- §C densities + tri-vs-square factor

def test_appendix_C_density_and_tri_vs_square():
    # "norm=91: 22120 pairs (8.85/n)" vs "square grid 50x50 best 17680
    # (7.07/n) -> tri beats square ~1.25-1.5x". Counts are pinned by
    # tests/test_counting.py; here we pin the densities and the factor.
    n = 2500
    tri_best, sq_best = 22120, 17680
    assert f"{tri_best / n:.2f}" == "8.85"
    assert f"{sq_best / n:.2f}" == "7.07"
    assert 1.25 <= tri_best / sq_best <= 1.5


# ------------------------------------------------ §D grid scaling of E

def test_appendix_D_grid_scaling_local_exponent():
    # "Grid scaling (exp run): local exponent 2.601 (n=196) -> 2.535
    # (n=3136), -> 2.5 (=M*n^2, M=sqrt(n))."
    # Local exponent between consecutive grids (m stepping by sqrt(2)) is
    # log(E2/E1)/log(n2/n1), attributed to the upper n.
    ms = [10, 14, 20, 28, 40, 56]
    E = {}
    for m in ms:
        E[m * m] = bisector_energy_int(square_grid(m)).E
    ns = [m * m for m in ms]
    exps = {
        n2: math.log(E[n2] / E[n1]) / math.log(n2 / n1)
        for n1, n2 in zip(ns, ns[1:])
    }
    assert f"{exps[196]:.3f}" == "2.601"
    assert f"{exps[3136]:.3f}" == "2.535"
    # decreasing toward the E = sqrt(n)*n^2 = n^2.5 prediction
    vals = [exps[n] for n in ns[1:]]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    assert all(v > 2.5 for v in vals)
