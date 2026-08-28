"""Timing instrumentation (:mod:`pycc.timing`): the accumulating profile registry and the
progress lines.

Covers the bookkeeping, not the clocks.  The arithmetic that makes the report readable is exact by
construction and is what regresses if the nesting logic is touched:

  * inclusive time is the block and everything inside it; self time is inclusive minus the
    children, so the self column sums to the top-level inclusive;
  * repeated entry of the same label accumulates into one record with a call count;
  * a label nested under two different parents is two records, not one.

Wall/user/system values themselves are only sanity-checked (finite, ordered, non-negative), since
asserting on real durations makes a flaky test.
"""

import re
import time

import pytest

import pycc
from pycc import timing as T


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts from an empty registry and leaves one behind."""
    T.reset()
    yield
    T.reset()


def _rec(*path):
    """The record for a label path, as (calls, wall, child wall, user, child user, sys, child sys)."""
    return T._records[tuple(path)]


def test_nesting_self_sums_to_inclusive():
    """Self time is inclusive minus children, so the self column sums to the parent's inclusive.
    This is the property the report's footer and the whole 'where did the time go' reading rely on."""
    with T.timer("outer"):
        _spin(0.02)
        with T.timer("inner"):
            _spin(0.02)
        with T.timer("inner"):
            _spin(0.02)

    outer = _rec("outer")
    inner = _rec("outer", "inner")
    assert outer[0] == 1 and inner[0] == 2                    # call counts
    assert inner[1] == pytest.approx(outer[2], rel=1e-9)      # child time == inner's inclusive
    self_outer = outer[1] - outer[2]
    self_inner = inner[1] - inner[2]
    assert self_outer + self_inner == pytest.approx(outer[1], rel=1e-9)


def test_repeated_label_accumulates():
    """Entering the same label repeatedly builds one record with a call count, not many rows."""
    for _ in range(5):
        with T.timer("repeated"):
            _spin(0.005)
    rec = _rec("repeated")
    assert rec[0] == 5
    assert rec[1] > 0.0


def test_same_label_under_two_parents_stays_separate():
    """Records are keyed by the full path, so the same leaf under two parents is two rows.  The
    duplicate 'two-electron first derivatives (MO)' rows in a Hessian report depend on this."""
    with T.timer("A"):
        with T.timer("leaf"):
            _spin(0.005)
    with T.timer("B"):
        with T.timer("leaf"):
            _spin(0.005)
    assert ("A", "leaf") in T._records and ("B", "leaf") in T._records
    assert _rec("A", "leaf")[0] == 1 and _rec("B", "leaf")[0] == 1


def test_clocks_are_sane():
    """Wall/user/system are finite, non-negative, and wall covers at least the sleep."""
    with T.timer("clocked"):
        _spin(0.05)
    calls, wall, _, user, _, sys_, _ = _rec("clocked")
    assert wall >= 0.04
    assert user >= 0.0 and sys_ >= 0.0
    assert user < 1000.0 and sys_ < 1000.0


def test_timer_records_even_on_exception():
    """The context manager closes its record in a finally block, so an error inside a timed block
    does not corrupt the stack for everything after it."""
    with pytest.raises(ValueError):
        with T.timer("boom"):
            raise ValueError("x")
    assert _rec("boom")[0] == 1
    assert T._stack == []                       # unwound, not left dangling


def test_timed_decorator():
    """The decorator form records under its label and leaves the function otherwise untouched."""
    @T.timed("decorated")
    def add(a, b=1):
        """docstring kept"""
        return a + b

    assert add(2, b=3) == 5
    assert add.__name__ == "add" and add.__doc__ == "docstring kept"
    assert _rec("decorated")[0] == 1


def test_reset_clears():
    with T.timer("gone"):
        pass
    assert T._records
    T.reset()
    assert T._records == {} and T._stack == []


def test_report_shape(capsys):
    """The report prints one indented row per label path and a footer that distinguishes the
    labelled total from the elapsed time, so the table is not mistaken for the whole run."""
    with T.timer("parent"):
        _spin(0.01)
        with T.timer("child"):
            _spin(0.01)
    T.report()
    out = capsys.readouterr().out

    assert "PyCC timing summary" in out
    assert "instrumented blocks only" in out
    assert re.search(r"^parent\s", out, re.M)
    assert re.search(r"^  child\s", out, re.M)                # nesting shown by indentation
    assert "sum of labelled top-level blocks" in out
    assert "elapsed since pycc was imported" in out
    assert "not inside any labelled block" not in out         # dropped; the gap is the two lines


def test_report_is_silent_when_empty(capsys):
    T.report()
    assert capsys.readouterr().out == ""


def test_progress_line(capsys):
    """A progress line carries the step, the count, the label, this step's time, and the cumulative
    stage time -- and flushes, since redirected stdout is block-buffered."""
    t0 = time.time() - 5.0
    T.progress("Hessian second-derivative integrals", 3, 55, t0, "C1-Cl3")
    line = capsys.readouterr().out.strip()

    assert "Hessian second-derivative integrals: 3/55" in line
    assert "(C1-Cl3)" in line
    m = re.search(r"step\s+([\d.]+) s\s+stage\s+([\d.]+) s", line)
    assert m, line
    assert float(m.group(2)) >= 5.0                           # cumulative, measured from t0


def test_progress_step_is_per_step_not_cumulative(capsys):
    """Step 1 measures from the stage start; later steps measure from the previous line, so the
    step column is per-iteration while the stage column keeps growing."""
    t0 = time.time()
    T.progress("stage", 1, 3, t0)
    _spin(0.05)
    T.progress("stage", 2, 3, t0)
    lines = capsys.readouterr().out.strip().split("\n")

    steps = [float(re.search(r"step\s+([\d.]+) s", ln).group(1)) for ln in lines]
    stages = [float(re.search(r"stage\s+([\d.]+) s", ln).group(1)) for ln in lines]
    assert steps[1] >= 0.04                                   # the second step saw the 50 ms gap
    assert stages[1] > stages[0]                              # cumulative keeps climbing
    assert stages[1] >= steps[1]


def test_exported_from_package():
    """The facade names exist, since scripts call pycc.timing_report()."""
    for name in ("timer", "timed", "progress", "timing_report", "timing_reset"):
        assert hasattr(pycc, name), name


def _spin(seconds):
    """Busy-wait, so the block accrues user time as well as wall (sleep would accrue neither)."""
    end = time.perf_counter() + seconds
    x = 0.0
    while time.perf_counter() < end:
        x += 1.0
    return x
