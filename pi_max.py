#!/usr/bin/env python3
"""
pi_max.py - compute as many digits of pi as your machine can handle.

Algorithm : Chudnovsky series (~14.18 digits per term) with binary splitting,
            the method used for record-setting pi computations.
Parallel  : the series is split into chunks computed on every CPU core,
            then merged in parallel as a tree.
Speed     : install gmpy2 (GMP big-number library) for a 10-100x speedup:
                pip install gmpy2
            The script still works without it using Python's built-in ints.

Usage
    python pi_max.py 1000000          # exactly 1,000,000 digits
    python pi_max.py                  # "max" mode: keep doubling the digit
                                      #   count until the time budget or a
                                      #   safe RAM limit is reached
    python pi_max.py --budget 60      # max mode with a 60-minute budget
    python pi_max.py --workers 4      # limit the number of cores used

Output is written to pi_<digits>.txt.
"""

import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

try:
    import gmpy2
    from gmpy2 import mpz, isqrt
    HAVE_GMPY2 = True
except ImportError:
    mpz = int
    from math import isqrt
    HAVE_GMPY2 = False

# Python 3.11+ refuses to convert very large ints to strings unless told not to.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

C = 640320
C3_OVER_24 = C ** 3 // 24
DIGITS_PER_TERM = math.log10(C3_OVER_24 / 72)   # ~14.18
GUARD_DIGITS = 20
# Rough peak memory per digit (bytes), deliberately conservative.
BYTES_PER_DIGIT = 12 if HAVE_GMPY2 else 16


# --------------------------------------------------------------------------
# Chudnovsky binary splitting
# --------------------------------------------------------------------------
def bs(a, b):
    """Return (P, Q, T) for the Chudnovsky terms in [a, b)."""
    if b - a == 1:
        if a == 0:
            p = q = mpz(1)
        else:
            p = mpz((6 * a - 5) * (2 * a - 1) * (6 * a - 1))
            q = mpz(a * a * a * C3_OVER_24)
        t = p * (13591409 + 545140134 * a)
        if a & 1:
            t = -t
        return p, q, t
    m = (a + b) // 2
    p1, q1, t1 = bs(a, m)
    p2, q2, t2 = bs(m, b)
    return p1 * p2, q1 * q2, q2 * t1 + p1 * t2


def combine(pair):
    """Merge two adjacent (P, Q, T) results."""
    (p1, q1, t1), (p2, q2, t2) = pair
    return p1 * p2, q1 * q2, q2 * t1 + p1 * t2


def bs_range(r):
    return bs(*r)


def compute_pi(digits, workers):
    """Return pi as a string '3.1415...' with exactly `digits` decimals."""
    prec = digits + GUARD_DIGITS
    n_terms = int(prec / DIGITS_PER_TERM) + 2

    # Small jobs: multiprocessing overhead isn't worth it.
    if workers <= 1 or n_terms < 2000:
        _, q, t = bs(0, n_terms)
        sq = isqrt(mpz(10005) * mpz(10) ** (2 * prec))
    else:
        chunks = workers * 4
        edges = [n_terms * i // chunks for i in range(chunks + 1)]
        ranges = [(edges[i], edges[i + 1]) for i in range(chunks)
                  if edges[i + 1] > edges[i]]
        with Pool(workers) as pool:
            # The square root runs in parallel with the series.
            sq_job = pool.apply_async(isqrt, (mpz(10005) * mpz(10) ** (2 * prec),))
            parts = pool.map(bs_range, ranges)
            while len(parts) > 1:                    # parallel tree merge
                pairs = [(parts[i], parts[i + 1])
                         for i in range(0, len(parts) - 1, 2)]
                merged = pool.map(combine, pairs)
                if len(parts) % 2:
                    merged.append(parts[-1])
                parts = merged
            _, q, t = parts[0]
            sq = sq_job.get()

    pi_int = (q * 426880 * sq) // t
    s = str(pi_int)
    return s[0] + "." + s[1:digits + 1]


# --------------------------------------------------------------------------
# Machine limits
# --------------------------------------------------------------------------
def available_ram_bytes():
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    try:                                    # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:                                    # macOS / other POSIX
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2
    except (ValueError, OSError, AttributeError):
        pass
    return 4 * 1024 ** 3                    # fallback guess: 4 GB


def max_safe_digits(workers):
    ram = available_ram_bytes() * 0.6       # leave headroom for the OS
    # Parallel mode holds pickled copies of partial results in several processes.
    overhead = 1.0 + 0.25 * max(0, workers - 1) ** 0.5
    return int(ram / (BYTES_PER_DIGIT * overhead))


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
KNOWN = "3.14159265358979323846264338327950288419716939937510"


def save_and_report(pi_str, digits, elapsed):
    if not pi_str.startswith(KNOWN[:min(len(KNOWN), len(pi_str))]):
        print("  !! sanity check FAILED - result does not match known digits")
    fname = f"pi_{digits}.txt"
    with open(fname, "w") as f:
        f.write(pi_str)
        f.write("\n")
    print(f"  {digits:,} digits in {elapsed:,.2f} s "
          f"({digits / max(elapsed, 1e-9):,.0f} digits/s) -> {fname}")
    if digits > 100:
        print(f"  first: {pi_str[:52]}...")
        print(f"  last : ...{pi_str[-50:]}")
    else:
        print(f"  {pi_str}")


def run(digits, workers):
    t0 = time.perf_counter()
    pi_str = compute_pi(digits, workers)
    elapsed = time.perf_counter() - t0
    save_and_report(pi_str, digits, elapsed)
    return elapsed


def main():
    ap = argparse.ArgumentParser(description="Compute pi to as many digits as possible.")
    ap.add_argument("digits", nargs="?", type=int,
                    help="number of decimal digits (omit for max mode)")
    ap.add_argument("--budget", type=float, default=10,
                    help="max mode: time budget for the final run, in minutes (default 10)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="CPU processes to use (default: all cores)")
    args = ap.parse_args()

    limit = max_safe_digits(args.workers)
    print(f"Backend : {'gmpy2 (GMP)' if HAVE_GMPY2 else 'Python ints (pip install gmpy2 for big speedup)'}")
    print(f"Cores   : {args.workers}")
    print(f"RAM cap : ~{limit:,} digits\n")

    if args.digits:
        if args.digits > limit:
            print(f"Warning: {args.digits:,} digits may exceed available RAM "
                  f"(safe estimate ~{limit:,}).")
        run(args.digits, args.workers)
        return

    # Max mode: double until the next run would blow the time budget or RAM.
    budget = args.budget * 60
    digits = 10_000
    growth = 2.3      # time grows a bit faster than linearly with digits
    while True:
        print(f"Computing {digits:,} digits...")
        elapsed = run(digits, args.workers)
        nxt = digits * 2
        if nxt > limit:
            print(f"\nStopping: {nxt:,} digits would exceed the safe RAM limit.")
            break
        if elapsed * growth > budget:
            print(f"\nStopping: next run (~{elapsed * growth / 60:.1f} min) "
                  f"would exceed the {args.budget:g}-minute budget.")
            break
        digits = nxt
    print(f"Best result: {digits:,} digits in pi_{digits}.txt")


if __name__ == "__main__":
    main()
