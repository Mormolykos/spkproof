"""Compare speaker encoders on YOUR recordings, under YOUR conditions.

The question this answers is not "which encoder wins on VoxCeleb". It is:

    my speakers get hoarse, shout, whisper and get emotional.
    Which encoder can I trust to still recognise them?

Those are different questions and the second one has no published answer for
most conditions. Benchmark leaderboards are computed on modal read speech; a
system deployed on people who raise their voice is operating outside the
regime it was ranked in.

Measured on a parallel corpus of 8 speakers x 6 phonatory states x identical
text (Gkilis, 2026), the spread between encoders under shouting was more than
threefold - 0.257 EER for one widely deployed encoder against 0.074 for
another, on identical trials. A system built on the wrong one silently rejects
its own users whenever they raise their voice.

THREE THINGS THIS MODULE GETS RIGHT THAT THE OBVIOUS VERSION GETS WRONG
-----------------------------------------------------------------------
All three were found by an integrity pass over that study's own frozen
11,935-trial manifest, after the results had been written down.

  1. THE COMPARISON IS PAIRED, NOT MARGINAL. Asking whether two encoders'
     confidence intervals overlap is not the same test as asking whether their
     DIFFERENCE excludes zero, and it is the weaker one: the two encoders are
     scored on identical trials by identical speakers, so most of what moves
     one interval moves the other with it. The difference is taken inside every
     bootstrap draw, which keeps that shared movement out of the interval.
     `rank_and_warn` used to declare ties by overlap; it no longer does.

  2. THE RESAMPLING UNIT IS THE SPEAKER, ON BOTH SIDES. A trial list is a
     directed graph on the speakers: genuine trials are self-loops, impostor
     trials are ordered edges. Every speaker is somebody else's impostor, so
     resampling ENROLLMENT identities and leaving the impostor side intact
     drops a speaker from one side of the design and keeps them on the other.
     The correct resample for graph data is the vertex bootstrap (Snijders &
     Borgatti 1999): draw the speakers with replacement and take the induced
     sub-multigraph, so a speaker drawn m times weights their genuine trials by
     m and their impostor edge (e -> t) by m_e * m_t, and a speaker not drawn
     contributes nothing anywhere. Run against the old scheme on identical
     draws, the corrected marginal intervals were up to 1.58x wider (median
     1.11x). Intervals that are 58% too narrow are how a panel announces a
     winner it does not have.

  3. MULTIPLICITY IS DECLARED AND CORRECTED. 14 encoders is 91 pairwise
     comparisons. Uncorrected, 59 of them cleared a 95% interval; 28 survived
     Holm-Bonferroni at family-wise 0.05. See `paired_compare`.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not rank encoders in general, and no output here licenses a claim that
one encoder is universally better. It reports which encoder separates YOUR
speakers under YOUR conditions, on trials you supply.

It also does not decide anything about a person. An EER is a property of a
measurement system, never of a speaker.

DEPENDENCIES
------------
None, deliberately. This reads the score table a verification run already
writes - one row per trial, one column per encoder - exactly as `check-f0`
reads a per-utterance table. Extracting the embeddings is your pipeline's job
and needs your model stack; judging them does not, and should not require
installing one.

The cost of that is arithmetic done in Python, so the bootstrap is linear in
trials per draw rather than free: the sort and the threshold grid are built
once per cell and only the weights change between draws. A table of a few
thousand trials and 2,000 draws is seconds per encoder; a table of a hundred
thousand wants a smaller `bootstrap` and a stated reason for it.
"""
from __future__ import annotations

import math
import random
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import accumulate

from .stats import holm_adjusted

# Bootstrap resamples for the confidence interval. Resampling SPEAKERS, not
# trials: trials share speakers and share clips, and resampling them
# independently understates the uncertainty.
DEFAULT_BOOTSTRAP = 2000

# An encoder whose difference from another includes zero is not distinguishable
# from it on this data. Reporting a winner in that case is the single most
# common way a small benchmark overstates itself.
CI_ALPHA = 0.05

# The two resampling units, named in the output because which one produced an
# interval is part of what the interval means.
RESAMPLE_DYADIC = "speaker (vertex bootstrap, both sides)"
RESAMPLE_ENROLL = "enrollment identity only"

# Speaker labels that carry no identity. A table that says "unknown" has not
# told us who the impostor was, and the dyadic weight cannot be formed.
_NO_SPEAKER = {"", "unknown", "none", "na", "n/a", "nan"}


@dataclass
class PanelResult:
    encoder: str
    condition: str
    eer: float
    eer_lo: float
    eer_hi: float
    dprime: float
    n_genuine: int
    n_impostor: int
    per_speaker: dict[str, float] = field(default_factory=dict)
    # Which resample produced eer_lo/eer_hi. Never inferred by a reader from
    # the presence of an interval: the weaker unit produces an interval too.
    resample_unit: str = RESAMPLE_ENROLL
    # The bootstrap draws themselves, in draw order and including the failures
    # as nan. Kept because the paired test needs the SAME draw to appear in
    # both encoders' columns, and a summary statistic cannot be re-paired
    # after the fact.
    draws: list[float] = field(default_factory=list, repr=False)


@dataclass
class Comparison:
    """One encoder against another, on one condition, from paired draws."""
    condition: str
    encoder_a: str
    encoder_b: str
    delta: float               # eer_a - eer_b; positive means a has more error
    lo: float
    hi: float
    p_value: float
    p_holm: float
    survives: bool             # after correction within the declared family
    significant_uncorrected: bool
    family: str
    n_family: int
    resample_unit: str

    def __str__(self) -> str:
        verdict = "survives" if self.survives else "does not survive"
        return (f"{self.condition}: {self.encoder_a} - {self.encoder_b} = "
                f"{self.delta:+.3f} [{self.lo:+.3f}, {self.hi:+.3f}] "
                f"p={self.p_value:.4f} Holm p={self.p_holm:.4f} ({verdict})")


@dataclass
class Family:
    """A set of comparisons declared BEFORE looking at them, and corrected as
    one. The declaration is the part that matters: correcting whichever subset
    turned out to be interesting is not a correction."""
    name: str
    kind: str                  # "confirmatory" | "exploratory"
    comparisons: list[tuple[str, str, str]]   # (condition, encoder_a, encoder_b)


def equal_error_rate(genuine: list[float], impostor: list[float]) -> float:
    """EER by sweeping every observed score as a threshold. Exact, no fitting,
    no interpolation onto a grid we chose.

    This is the reference implementation. The bootstrap uses a weighted form
    that is O(n) per draw instead of O(n^2); `test_panel.py` asserts the two
    agree exactly at unit weights, which is the only thing that makes the fast
    path trustworthy."""
    if not genuine or not impostor:
        return float("nan")
    thresholds = sorted(set(genuine) | set(impostor))
    ng, ni = len(genuine), len(impostor)
    best_gap = float("inf")
    best_eer = 1.0
    for t in thresholds:
        frr = sum(1 for g in genuine if g < t) / ng
        far = sum(1 for i in impostor if i >= t) / ni
        gap = abs(frr - far)
        eer = (frr + far) / 2
        # The gap and the EER are different quantities. An earlier version
        # compared the gap against the running EER, which selects a threshold
        # that is neither the crossing point nor the minimum, and inflated the
        # reported rate. Keep them separate.
        if gap < best_gap or (gap == best_gap and eer < best_eer):
            best_gap, best_eer = gap, eer
    return best_eer


def d_prime(genuine: list[float], impostor: list[float]) -> float:
    """Separation in pooled standard deviations. Reported alongside EER because
    two systems can share an EER with very different margins."""
    if len(genuine) < 2 or len(impostor) < 2:
        return float("nan")
    mg = sum(genuine) / len(genuine)
    mi = sum(impostor) / len(impostor)
    vg = sum((x - mg) ** 2 for x in genuine) / (len(genuine) - 1)
    vi = sum((x - mi) ** 2 for x in impostor) / (len(impostor) - 1)
    pooled = (vg + vi) / 2
    return (mg - mi) / math.sqrt(pooled) if pooled > 0 else float("nan")


# --------------------------------------------------------------------------
# the weighted estimator
# --------------------------------------------------------------------------


class _Cell:
    """One encoder's scores under one condition, pre-sorted with a fixed
    threshold grid.

    The trial VALUES never change across bootstrap draws - only their weights
    do - so the sort, the threshold grid and the position of every threshold
    within each sorted side are computed once and reused. That is what makes a
    speaker bootstrap linear per draw in a language with no vectors."""

    __slots__ = ("g_speaker", "i_enroll", "i_test", "n_thresholds", "kg", "ki")

    def __init__(
        self,
        genuine: list[tuple[float, int]],
        impostor: list[tuple[float, int, int]],
    ) -> None:
        genuine = sorted(genuine, key=lambda t: t[0])
        impostor = sorted(impostor, key=lambda t: t[0])
        g_values = [v for v, _ in genuine]
        i_values = [v for v, _, _ in impostor]
        self.g_speaker = [s for _, s in genuine]
        self.i_enroll = [e for _, e, _ in impostor]
        self.i_test = [t for _, _, t in impostor]
        thresholds = sorted(set(g_values) | set(i_values))
        self.n_thresholds = len(thresholds)
        # Counts of values STRICTLY BELOW each threshold, which is what both
        # frr and far are read off. bisect_left on the sorted side gives it
        # without a scan.
        self.kg = [bisect_left(g_values, t) for t in thresholds]
        self.ki = [bisect_left(i_values, t) for t in thresholds]


def _weighted_eer(cell: _Cell, weights: Sequence[float], dyadic: bool) -> float:
    """EER under the per-trial weights induced by one speaker resample.

    dyadic: impostor edge (e -> t) gets weight m_e * m_t - the induced
    sub-multigraph of the vertex bootstrap, so a speaker who was not drawn
    disappears from the impostor side as well as the genuine side.
    Otherwise it gets m_e, which is the enrollment-only scheme kept for the
    comparison that shows how much narrower it is.

    A genuine trial is the self-loop (s -> s) and gets m_s, not m_s squared.
    That is a choice and not an oversight: each side of the EER is normalised
    by its own total weight, so squaring the self-loops would not remove a
    speaker any more thoroughly - it would only re-weight the genuine
    distribution towards whoever happened to be drawn twice. This matches the
    implementation these intervals were validated against.

    THE PLATEAU TIE-BREAK. DO NOT "FIX" THIS TOWARDS THE OTHER REFERENCE.
    ---------------------------------------------------------------------
    On discrete scores no threshold makes frr equal far, so the crossing is
    approached from both sides and TWO thresholds can share the smallest
    |frr - far| gap: one below the crossing with frr - far = -g, one above it
    with +g. They are equally balanced and they have different error rates.

    This takes the one with the LOWER EER, which is what `equal_error_rate`
    does, so the fast path and the reference are one estimator rather than two.
    The prototype this was ported from (`integrity_pass.py: eer_w`) takes
    `argmin|frr - far|`, which is the FIRST minimiser - and the first minimiser
    is the lower threshold, not the better operating point. On
    genuine [0.4, 0.6] against impostor [0.5, 0.5] both rules see a 0.5 gap
    twice; this returns 0.25 and argmin returns 0.75, for the same data.
    `test_paired.py` pins that case so nobody quietly moves this rule to match
    the prototype.

    Why the rule here is the correct one, and not merely the incumbent: an EER
    is the balanced operating point a system would actually be run at, so when
    two thresholds are equally balanced, the one with more error is not the
    system's error rate - it is the worse of two available settings. Taking it
    inflates the reported rate, and it inflates it unevenly, since which side
    wins depends on where the weights happen to fall in a given bootstrap draw.

    The two rules agree exactly at unit weights, which is why the prototype's
    own self-check against this library passed and the divergence stayed
    invisible. Under weighted draws it appears: 6.3e-3 worst over 1,200 draws
    on the 11,935-trial manifest behind this module. Experiment D found the
    same defect independently at 2.2e-4, about 28x smaller, which says the size
    of the divergence is a property of the data and not a bound to rely on."""
    wg = [weights[s] for s in cell.g_speaker]
    if dyadic:
        wi = [weights[e] * weights[t]
              for e, t in zip(cell.i_enroll, cell.i_test, strict=True)]
    else:
        wi = [weights[e] for e in cell.i_enroll]

    cg = list(accumulate(wg, initial=0.0))
    ci = list(accumulate(wi, initial=0.0))
    total_g, total_i = cg[-1], ci[-1]
    if total_g <= 0 or total_i <= 0:
        return float("nan")

    kg, ki = cell.kg, cell.ki

    def frr(j: int) -> float:
        return cg[kg[j]] / total_g

    def far(j: int) -> float:
        return (total_i - ci[ki[j]]) / total_i

    # frr is non-decreasing in the threshold and far is non-increasing, so
    # frr - far is non-decreasing and |frr - far| is a valley: the minimum sits
    # at the sign change, and the set of minimisers is contiguous. Binary
    # search for the crossing, then walk the plateau, which reproduces the
    # reference's "smallest gap, ties broken by smaller EER" exactly instead of
    # approximately.
    lo, hi = 0, cell.n_thresholds
    while lo < hi:
        mid = (lo + hi) // 2
        if frr(mid) - far(mid) >= 0:
            hi = mid
        else:
            lo = mid + 1
    candidates = [j for j in (lo - 1, lo) if 0 <= j < cell.n_thresholds]
    if not candidates:
        return float("nan")

    def gap(j: int) -> float:
        return abs(frr(j) - far(j))

    def eer(j: int) -> float:
        return (frr(j) + far(j)) / 2

    best_gap = min(gap(j) for j in candidates)
    best_eer = min(eer(j) for j in candidates if gap(j) == best_gap)
    j = min(candidates) - 1
    while j >= 0 and gap(j) == best_gap:
        best_eer = min(best_eer, eer(j))
        j -= 1
    j = max(candidates) + 1
    while j < cell.n_thresholds and gap(j) == best_gap:
        best_eer = min(best_eer, eer(j))
        j += 1
    return best_eer


def _draws(n_speakers: int, n: int, seed: int) -> list[list[float]]:
    """n multiplicity vectors, each drawing n_speakers vertices with
    replacement. Generated once and used by every encoder and condition: a
    paired test needs the same draw in both columns."""
    rng = random.Random(seed)
    out: list[list[float]] = []
    for _ in range(n):
        m = [0.0] * n_speakers
        for _ in range(n_speakers):
            m[rng.randrange(n_speakers)] += 1.0
        out.append(m)
    return out


def _percentile_ci(values: list[float], alpha: float) -> tuple[float, float]:
    usable = sorted(v for v in values if v == v)
    if not usable:
        return (float("nan"), float("nan"))
    lo = usable[int(len(usable) * alpha / 2)]
    hi = usable[min(len(usable) - 1, int(len(usable) * (1 - alpha / 2)))]
    return (lo, hi)


def _bootstrap_p(differences: list[float]) -> float:
    """Two-sided bootstrap p by interval inversion, with the standard +1
    smoothing so a p of exactly zero is never reported from a finite number of
    draws. The floor is 2/(B+1), which is why `paired_compare` refuses to stay
    quiet when the family is larger than the draw count can support."""
    usable = [d for d in differences if d == d]
    n = len(usable)
    if n == 0:
        return float("nan")
    lo = (1 + sum(1 for d in usable if d >= 0)) / (n + 1)
    hi = (1 + sum(1 for d in usable if d <= 0)) / (n + 1)
    return min(1.0, 2 * min(lo, hi))


# --------------------------------------------------------------------------
# the panel
# --------------------------------------------------------------------------


def _speaker(row: Mapping[str, object], *keys: str) -> str:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return "unknown"


def score_panel(
    scores: Sequence[Mapping[str, object]],
    encoders: Sequence[str],
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    resample: str = "auto",
    keep_draws: bool = True,
    alpha: float = CI_ALPHA,
) -> list[PanelResult]:
    """Rows need: condition, label ('genuine'/'impostor'), speaker, and one
    numeric column per encoder. This is exactly what a verification run already
    writes; nothing needs regenerating to use it.

    A `test_speaker` column, when present, buys the correct resampling unit:
    without it the impostor side cannot be rebuilt from the drawn speakers and
    the interval is the enrollment-only one, which measured up to 1.58x too
    narrow on the study behind this module. `resample` is "auto" (dyadic when
    the column is there), "dyadic" (insist, and raise if it is not) or
    "enroll" (the old scheme, for reproducing an older number).

    `alpha` sets the interval level and nothing else; the comparison between
    encoders is `paired_compare`, which takes its own.
    """
    if resample not in {"auto", "dyadic", "enroll"}:
        raise ValueError(f"resample must be auto, dyadic or enroll, not {resample!r}")

    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for r in scores:
        for e in encoders:
            grouped[(e, str(r.get("condition", "unlabelled")))].append(r)

    # The speaker universe is every identity that appears anywhere on either
    # side, so the vertex draw covers the whole graph rather than the enrolled
    # subset of it.
    universe: set[str] = set()
    n_impostor = n_test_known = 0
    for r in scores:
        universe.add(_speaker(r, "enroll_speaker", "speaker"))
        if str(r.get("label", "")).lower().startswith("gen"):
            continue
        n_impostor += 1
        t = _speaker(r, "test_speaker")
        if t.lower() not in _NO_SPEAKER:
            universe.add(t)
            n_test_known += 1
    speakers = sorted(universe)
    index = {s: i for i, s in enumerate(speakers)}

    # One impostor trial with no test speaker is enough to make the dyadic
    # weight undefined for that edge, and an interval built from a mixture of
    # the two schemes is neither.
    dyadic = n_impostor > 0 and n_test_known == n_impostor
    if resample == "dyadic" and not dyadic:
        raise ValueError(
            "resample='dyadic' needs a test_speaker on every impostor trial; "
            "without it the impostor side cannot be rebuilt from a speaker draw"
        )
    if resample == "enroll":
        dyadic = False
    unit = RESAMPLE_DYADIC if dyadic else RESAMPLE_ENROLL

    draws = (_draws(len(speakers), bootstrap, seed)
             if bootstrap and len(speakers) >= 2 else [])
    unit_weights = [1.0] * len(speakers)

    results: list[PanelResult] = []
    for (enc, cond), rows in sorted(grouped.items()):
        gen: list[float] = []
        imp: list[float] = []
        g_cell: list[tuple[float, int]] = []
        i_cell: list[tuple[float, int, int]] = []
        by_spk: dict[str, tuple[list[float], list[float]]] = defaultdict(
            lambda: ([], []))
        for r in rows:
            raw = r.get(enc, "")
            if raw == "" or raw is None:
                continue
            try:
                v = float(str(raw))
            except (TypeError, ValueError):
                continue
            enroll = _speaker(r, "enroll_speaker", "speaker")
            spk = index.get(enroll, 0)
            if str(r.get("label", "")).lower().startswith("gen"):
                gen.append(v)
                by_spk[enroll][0].append(v)
                g_cell.append((v, spk))
            else:
                imp.append(v)
                by_spk[enroll][1].append(v)
                test = _speaker(r, "test_speaker")
                i_cell.append((v, spk, index.get(test, spk)))
        if not gen or not imp:
            continue

        cell = _Cell(g_cell, i_cell)
        point = _weighted_eer(cell, unit_weights, dyadic)
        column = [_weighted_eer(cell, m, dyadic) for m in draws]
        lo, hi = _percentile_ci(column, alpha) if column else (
            float("nan"), float("nan"))
        per_spk = {
            s: equal_error_rate(g, i)
            for s, (g, i) in sorted(by_spk.items()) if g and i
        }
        results.append(PanelResult(
            encoder=enc, condition=cond,
            eer=point, eer_lo=lo, eer_hi=hi,
            dprime=d_prime(gen, imp),
            n_genuine=len(gen), n_impostor=len(imp),
            per_speaker=per_spk,
            resample_unit=unit,
            draws=column if keep_draws else [],
        ))
    return results


def resample_note(results: list[PanelResult]) -> str | None:
    """The sentence to print when the intervals came from the weaker unit."""
    if not results or any(r.resample_unit == RESAMPLE_DYADIC for r in results):
        return None
    return (
        "intervals were resampled over ENROLLMENT identities only, because the "
        "trial table has no test_speaker column. Every speaker is also somebody "
        "else's impostor, so this drops a speaker from one side of the design "
        "and keeps them on the other. Measured against the correct speaker-level "
        "resample on identical draws, it produced intervals up to 1.58x too "
        "narrow (median 1.11x). Add a test_speaker column to get the right one."
    )


# --------------------------------------------------------------------------
# the paired comparison
# --------------------------------------------------------------------------


def all_pairs(
    results: list[PanelResult],
    conditions: Sequence[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Every within-condition encoder pair present in `results`.

    Within condition, and only within: two encoders under two different
    conditions were scored on different trials, so the draw-by-draw difference
    between them is not paired and the test below does not apply to it."""
    by_cond: dict[str, list[str]] = defaultdict(list)
    for r in results:
        if conditions is not None and r.condition not in conditions:
            continue
        by_cond[r.condition].append(r.encoder)
    out: list[tuple[str, str, str]] = []
    for cond in sorted(by_cond):
        encs = sorted(set(by_cond[cond]))
        for i in range(len(encs)):
            for j in range(i + 1, len(encs)):
                out.append((cond, encs[i], encs[j]))
    return out


def declare_family(
    name: str,
    comparisons: Sequence[tuple[str, str, str]],
    kind: str = "confirmatory",
) -> Family:
    """Name a set of comparisons before testing it.

    The name is not decoration. A family is the set of hypotheses the
    family-wise error rate is controlled over, so it has to be fixed before the
    answers are visible; a family assembled afterwards out of the comparisons
    that looked promising controls nothing at all."""
    if kind not in {"confirmatory", "exploratory"}:
        raise ValueError(f"family kind must be confirmatory or exploratory, not {kind!r}")
    return Family(name=name, kind=kind, comparisons=list(comparisons))


def paired_compare(
    results: list[PanelResult],
    family: Family | None = None,
    alpha: float = CI_ALPHA,
) -> tuple[list[Comparison], list[str]]:
    """Test encoder differences inside the bootstrap draws, then correct.

    The difference is formed draw by draw, so the trials, speakers and phrases
    the two encoders share cancel instead of inflating both intervals. That is
    the test the study behind this module reports, and until now it was not the
    test this library ran.

    `family` defaults to every within-condition pair in `results`, declared
    confirmatory. Holm-Bonferroni is applied across the whole family at
    family-wise `alpha`. Returns (comparisons, warnings)."""
    with_draws = [r for r in results if r.draws]
    if not with_draws:
        return [], ["no bootstrap draws: run score_panel with bootstrap > 0 "
                    "and keep_draws=True, or there is nothing to pair"]

    by_key = {(r.condition, r.encoder): r for r in with_draws}
    declared = family or declare_family("all pairwise", all_pairs(with_draws))
    n_draws = len(with_draws[0].draws)
    notes: list[str] = []

    rows: list[tuple[str, str, str, float, float, float, float, bool]] = []
    for cond, a, b in declared.comparisons:
        ra, rb = by_key.get((cond, a)), by_key.get((cond, b))
        if ra is None or rb is None:
            notes.append(f"{cond}: no scored result for "
                         f"{a if ra is None else b}; comparison dropped")
            continue
        diffs = [x - y for x, y in zip(ra.draws, rb.draws, strict=True) if x == x and y == y]
        if not diffs:
            notes.append(f"{cond}: {a} vs {b} had no usable draws")
            continue
        lo, hi = _percentile_ci(diffs, alpha)
        p = _bootstrap_p(diffs)
        rows.append((cond, a, b, ra.eer - rb.eer, lo, hi, p, lo > 0 or hi < 0))

    adjusted = holm_adjusted([r[6] for r in rows])
    unit = with_draws[0].resample_unit
    comparisons = [
        Comparison(
            condition=cond, encoder_a=a, encoder_b=b, delta=delta,
            lo=lo, hi=hi, p_value=p, p_holm=p_adj,
            survives=p_adj <= alpha, significant_uncorrected=uncorrected,
            family=declared.name, n_family=len(rows), resample_unit=unit,
        )
        for (cond, a, b, delta, lo, hi, p, uncorrected), p_adj
        in zip(rows, adjusted, strict=True)
    ]

    # A bootstrap p cannot go below 2/(B+1). Holm's smallest threshold is
    # alpha/m. When the first is above the second, no comparison in the family
    # can survive however large the effect is, and the family reads as a clean
    # negative when it is really an unmeasured one.
    floor = 2 / (n_draws + 1)
    if comparisons and floor > alpha / len(comparisons):
        needed = int(math.ceil(2 * len(comparisons) / alpha)) - 1
        notes.append(
            f"{n_draws} bootstrap draws cannot resolve this family: the smallest "
            f"p obtainable is {floor:.4f} and Holm's smallest threshold over "
            f"{len(comparisons)} comparisons is {alpha / len(comparisons):.4f}. "
            f"Nothing here can survive correction at any effect size. Raise the "
            f"bootstrap to at least {needed} draws or declare a smaller family."
        )

    if declared.kind == "confirmatory":
        full = set(all_pairs(with_draws, sorted({c for c, _, _ in declared.comparisons})))
        missing = full - set(declared.comparisons) - {(c, b, a) for c, a, b in declared.comparisons}
        if missing:
            notes.append(
                f"family '{declared.name}' is declared confirmatory but tests "
                f"{len(declared.comparisons)} of the {len(full)} pairwise comparisons "
                f"available in these conditions. Correcting the subset you chose to "
                f"report understates the multiplicity you actually faced; test the "
                f"whole pairwise family, or declare this one exploratory and say so."
            )

    note = resample_note(with_draws)
    if note:
        notes.append(note)
    return comparisons, notes


def family_summary(comparisons: list[Comparison]) -> dict[str, object]:
    """The two counts that have to be reported together.

    Reporting only the survivors hides how much correction cost; reporting only
    the uncorrected count is the error the correction exists to prevent."""
    return {
        "family": comparisons[0].family if comparisons else "",
        "n_comparisons": len(comparisons),
        "n_significant_uncorrected": sum(1 for c in comparisons if c.significant_uncorrected),
        "n_survive_holm": sum(1 for c in comparisons if c.survives),
        "resample_unit": comparisons[0].resample_unit if comparisons else "",
    }


def rank_and_warn(
    results: list[PanelResult],
    alpha: float = CI_ALPHA,
) -> tuple[dict[str, str], list[str]]:
    """Per condition: the best encoder, and every encoder not distinguishable
    from it.

    The tie test is the paired bootstrap on differences, corrected with Holm
    over the comparisons against that condition's leader. It replaces an
    overlap test on marginal intervals, which is the conservative version of a
    different question and was the reason this library was weaker than the
    method it documents. Where there are no draws to pair - `bootstrap=0` -
    the marginal overlap is all there is, and it is used and labelled.

    The leader is chosen on the same data it is then compared against, so these
    k-1 comparisons are not a pre-declared family and the correction inside them
    does not undo that selection. The output is a refusal, never a coronation:
    the encoders that cannot be separated from the leader are named, and the
    only safe reading of a condition with no tie is "nothing here was shown to
    be indistinguishable", not "the leader won"."""
    winners: dict[str, str] = {}
    notes: list[str] = []
    by_cond: dict[str, list[PanelResult]] = defaultdict(list)
    for r in results:
        by_cond[r.condition].append(r)

    for cond, rs in sorted(by_cond.items()):
        rs = [r for r in rs if r.eer == r.eer]
        if not rs:
            continue
        best = min(rs, key=lambda r: r.eer)
        winners[cond] = best.encoder
        rivals = [r for r in rs if r.encoder != best.encoder]
        if not rivals:
            continue

        if best.draws and all(r.draws for r in rivals):
            family = declare_family(
                f"vs the leader on '{cond}'",
                [(cond, best.encoder, r.encoder) for r in rivals],
                kind="exploratory",   # k-1 comparisons against a leader, not the pairwise family
            )
            comparisons, _ = paired_compare(rs, family, alpha)
            tied = sorted(c.encoder_b for c in comparisons if not c.survives)
            method = (f"paired bootstrap on the difference, Holm over the "
                      f"{len(comparisons)} comparisons against the leader")
        else:
            if best.eer_hi != best.eer_hi:
                continue
            tied = sorted(r.encoder for r in rivals
                          if r.eer_lo == r.eer_lo and r.eer_lo <= best.eer_hi)
            method = "overlap of marginal intervals - no draws to pair"

        if tied:
            notes.append(
                f"{cond}: {best.encoder} has the lowest EER ({best.eer:.3f}) but "
                f"{', '.join(tied)} cannot be separated from it ({method}). On this "
                f"data they are not distinguishable; do not report a winner."
            )
    return winners, notes


def worst_condition(results: list[PanelResult], encoder: str) -> tuple[str, float] | None:
    """Where does this encoder fail your speakers? The deployment question."""
    rs = [r for r in results if r.encoder == encoder and r.eer == r.eer]
    if not rs:
        return None
    w = max(rs, key=lambda r: r.eer)
    return (w.condition, w.eer)
