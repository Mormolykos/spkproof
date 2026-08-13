"""Exact small-sample tests, implemented here so the package needs no scipy.

Both functions are exact rather than approximate: the tables voiceproof works on
are small (tens of utterances), and a chi-square approximation on a table with a
zero cell — which is exactly the shape contamination produces — is not sound.
"""
from __future__ import annotations

from math import comb


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Two-sided Fisher exact test on [[a, b], [c, d]].

    Returns (p_value, odds_ratio). Odds ratio is float('inf') when a zero cell
    makes it undefined — which is the case worth flagging, not one to hide.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0, float("nan")
    row1, row2, col1 = a + b, c + d, a + c
    if row1 == 0 or row2 == 0 or col1 == 0 or col1 == n:
        return 1.0, float("nan")

    denom = comb(n, col1)

    def prob(x: int) -> float:
        return comb(row1, x) * comb(row2, col1 - x) / denom

    p_obs = prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    # Sum every table at least as extreme as observed. The epsilon guards against
    # float equality failing on tables that are exactly as likely as the observed one.
    p = sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9))

    odds = float("inf") if b * c == 0 else (a * d) / (b * c)
    return min(p, 1.0), odds


def binom_sign_test(successes: int, n: int) -> float:
    """Two-sided exact binomial test against p = 0.5."""
    if n == 0:
        return 1.0
    k = min(successes, n - successes)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(2.0 * tail, 1.0)
