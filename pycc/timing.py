"""timing.py: wall / user / system time instrumentation.

Two independent facilities:

* :func:`timer` (or the :func:`timed` decorator) accumulates time per labelled block, and
  :func:`report` prints one summary at the end of a run -- for finding what to optimize.
* :func:`progress` prints a line per step of a long loop while it runs -- for following a
  calculation.

Blocks nest, and the summary reports **inclusive** time (the block and everything inside it) and
**self** time (inclusive minus its children, which sums to the wall time).

Three clocks are kept because wall time alone cannot say *why* a block is slow.  ``user`` and
``sys`` come from ``getrusage``, which sums over all threads of the process, so in-process OpenMP
and threaded BLAS show up.  Reading the ``cpu/wall`` column (``(user + sys) / self wall``):

    ~= 8      threaded, roughly 8 cores busy
    ~= 1, user   serial and compute-bound
    ~= 1, sys    kernel-bound (syscalls, page faults on a large allocation)
    << 1         blocked, waiting on disk

**Set ``OMP_WAIT_POLICY=PASSIVE`` and ``KMP_BLOCKTIME=0`` for trustworthy user times.**  OpenMP
workers busy-wait after a parallel region by default, and that spinning CPU is charged to whichever
block runs next: measured here, one allocation read 0.18 s user on its own and 3.06 s when it
followed a threaded block.  :func:`report` warns when the mitigation is not in effect.  It is not
set from inside pycc, since sleeping instead of spinning adds wake-up latency to every parallel
region and that is the caller's trade to make.

Labels are plain descriptions ("two-electron second derivatives"), not function names, because they
are read by whoever reads the output.
"""

import atexit
import os
import resource
import time
from contextlib import contextmanager
from functools import wraps

#: Print the summary automatically when the process exits.
AUTO_REPORT = True

#: Below this self wall time the cpu/wall ratio is noise, so it is left blank.
MIN_WALL = 5e-3

_T_IMPORT = time.perf_counter()   # for the "how much of the run is unlabelled" footer

_stack = []      # labels of the blocks currently open, outermost first
_records = {}    # path -> [calls, wall, child wall, user, child user, sys, child sys]
_last = {}       # progress stage -> wall time of its previous line, for the per-step time


def _clocks():
    """Wall, user, and system seconds now.  ``getrusage`` sums over the process's threads."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    return time.perf_counter(), r.ru_utime, r.ru_stime


@contextmanager
def timer(label):
    """Accumulate wall/user/system time for a labelled block.  Nesting builds the summary tree."""
    _stack.append(label)
    rec = _records.setdefault(tuple(_stack), [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    w0, u0, s0 = _clocks()
    try:
        yield
    finally:
        w1, u1, s1 = _clocks()
        dw, du, ds = w1 - w0, u1 - u0, s1 - s0
        rec[0] += 1
        rec[1] += dw
        rec[3] += du
        rec[5] += ds
        _stack.pop()
        if _stack:                                    # charge the parent for this child
            parent = _records[tuple(_stack)]
            parent[2] += dw
            parent[4] += du
            parent[6] += ds


def timed(label):
    """Decorator form of :func:`timer`."""
    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with timer(label):
                return fn(*args, **kwargs)
        return wrapper
    return decorate


def progress(stage, step, total, t0, note=""):
    """Print one flushed progress line: which step, of how many, the time this step took, and the
    cumulative time in the stage since ``t0``.

    The per-step time is measured against the previous line of the same stage (against ``t0`` for
    step 1), so no call site has to track it.

    Flushing matters: output redirected to a file is block-buffered, so an unflushed progress line
    can sit unseen for the whole calculation."""
    now = time.time()
    prev = t0 if step <= 1 else _last.get(stage, t0)
    _last[stage] = now
    head = "  %s: %d/%d" % (stage, step, total)
    if note:
        head += " (%s)" % note
    print("%-56s step %7.2f s   stage %8.1f s" % (head, now - prev, now - t0), flush=True)


def reset():
    """Discard everything recorded so far."""
    _stack.clear()
    _records.clear()
    _last.clear()


def report():
    """Print the timing summary, in the order the blocks were first entered.

    The footer gives both the sum of the labelled top-level blocks and the time elapsed since pycc
    was imported, since only instrumented code appears in the table: the gap between the two is
    whatever ran outside every timer (the SCF, the wavefunction build, the harmonic analysis,
    checkpoint writes).

    Self user/sys are clamped at zero: ``getrusage`` resolves to a few microseconds, so a block
    whose children account for nearly all of it can otherwise come out slightly negative."""
    if not _records:
        return
    names = {p: "  " * (len(p) - 1) + p[-1] for p in _records}
    w = max(30, max(len(n) for n in names.values()) + 2)
    head = ("%-*s %7s %10s %10s %9s %8s %9s"
            % (w, "", "calls", "incl wall", "self wall", "user", "sys", "cpu/wall"))
    rule = "-" * len(head)
    print("\n" + "=" * len(head))
    print("PyCC timing summary   (seconds; self = inclusive minus children;")
    print("                       instrumented blocks only, see the footer)")
    print("=" * len(head))
    print(head)
    print(rule)
    tot_wall = tot_user = tot_sys = 0.0
    for path, (calls, wall, cwall, user, cuser, sys_, csys) in _records.items():
        self_wall = wall - cwall
        self_user = max(0.0, user - cuser)
        self_sys = max(0.0, sys_ - csys)
        ratio = ("%9.2f" % ((self_user + self_sys) / self_wall)
                 if self_wall >= MIN_WALL else " " * 9)
        if len(path) == 1:
            tot_wall += wall
            tot_user += user
            tot_sys += sys_
        print("%-*s %7d %10.2f %10.2f %9.2f %8.2f %s"
              % (w, names[path], calls, wall, self_wall, self_user, self_sys, ratio))
    elapsed = time.perf_counter() - _T_IMPORT
    print(rule)
    print("%-*s %7s %10.2f %10s %9.2f %8.2f %9.2f"
          % (w, "sum of labelled top-level blocks", "", tot_wall, "", tot_user, tot_sys,
             (tot_user + tot_sys) / tot_wall if tot_wall else 0.0))
    print("%-*s %7s %10.2f" % (w, "elapsed since pycc was imported", "", elapsed))
    if os.environ.get("OMP_WAIT_POLICY", "").upper() != "PASSIVE":
        print("\n  NOTE: OMP_WAIT_POLICY is not PASSIVE, so OpenMP spin-wait may inflate the user\n"
              "        time of blocks that follow a threaded one.  Set OMP_WAIT_POLICY=PASSIVE and\n"
              "        KMP_BLOCKTIME=0 for attributable user times.  This check only sees whether\n"
              "        the variable is set, not whether it took effect: it must be set before the\n"
              "        first import that starts the OpenMP runtime (numpy loads its BLAS at import).\n"
              "        A report where nearly every row shows the same cpu/wall is the symptom.")


@atexit.register
def _report_at_exit():
    if AUTO_REPORT:
        report()
