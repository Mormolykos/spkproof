"""Exact small-sample tests, implemented here so the package needs no scipy.

The first two functions are exact rather than approximate: the tables voiceproof
works on are small (tens of utterances), and a chi-square approximation on a
table with a zero cell — which is exactly the shape contamination produces — is
not sound.

The third is a multiplicity correction, and it is here for a blunter reason. A
panel of 14 encoders is 91 pairwise comparisons. In the study this package came
out of, 59 of those 91 cleared an uncorrected 95% interval and 28 survived
Holm-Bonferroni at family-wise 0.05. The 31 that evaporated had already been
written down as results.
"""
from __future__ import annotations

from collections.abc import Sequence
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
    return float(min(2.0 * tail, 1.0))


def holm_adjusted(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in input order.

    Step-down: the smallest p is compared against alpha/m, the next against
    alpha/(m-1), and so on, and the whole thing stops at the first failure.
    Returning adjusted p-values rather than a set of reject flags is the same
    procedure — a hypothesis is rejected exactly when its adjusted p is at or
    below alpha — and it survives being read next to an uncorrected p, which a
    bare boolean does not.

    Holm rather than plain Bonferroni because it is uniformly more powerful and
    needs no more assumptions: both control the family-wise error rate under
    arbitrary dependence, which is what a family of pairwise comparisons over a
    shared bootstrap has.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    out = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        # The max keeps the adjusted values monotone in the raw ones. Without
        # it a later, larger raw p can get a smaller adjusted p, and the
        # procedure stops being a step-down.
        running = max(running, (m - rank) * pvalues[i])
        out[i] = min(1.0, running)
    return out


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Reject flags at family-wise `alpha`, in input order."""
    return [p <= alpha for p in holm_adjusted(pvalues)]
