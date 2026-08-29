"""Design checks for speaker-verification trial lists.

`check_f0` catches a contaminated *measurement*. These checks catch a
contaminated *comparison* - a trial list whose structure guarantees the answer
before any encoder runs.

Every rule here was written because it caught a real error in a real study
design during the work behind this module (Gkilis, 2026). Each one killed a
claim that had already been written down as a result:

  SPK-DUR-CONFOUND     neutral clips averaged 1.4 s and shouted clips 3.5 s.
                       Published speaker-verification numbers put EER at 0.61%
                       on full-length audio, 0.98% at 3 s and 1.48% at 2 s
                       (ERes2NetV2, arXiv 2406.02167). A condition contrast that
                       straddles that range measures duration and reports it as
                       phonation.
  SPK-POOL-DRIFT       the impostor speaker set differed between conditions,
                       because speakers who lacked a condition silently dropped
                       out of it. Cross-condition EER then compares different
                       impostor populations, not different conditions.
  SPK-ENROLL-LEAK      an enrollment clip reappeared as a test clip. Similarity
                       to one's own enrollment material is not a measurement.
  SPK-RATIO-DRIFT      the genuine:impostor ratio varied by condition. EER is
                       defined at the crossing point of two distributions; if
                       one side is sampled differently per condition, the
                       operating points are not comparable.

The checks need only a trial table: one row per trial, with a condition, a
genuine/impostor label, the enrolled speaker, the test speaker, and - for the
duration rule - the test clip's length. No audio, no encoder, no model.

None of these rules says a study is wrong. Each says a specific comparison
cannot carry the weight a result would put on it, and cites the numbers.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median

# Below this, published speaker-verification error rates rise steeply. A
# condition whose clips sit mostly under it is being scored in a different
# regime from one whose clips sit above it.
SHORT_CLIP_SEC = 2.0

# Ratio of median durations between two conditions beyond which duration is a
# live alternative explanation for any difference found between them. 1.3 is
# deliberately permissive: the corpus that motivated this rule ran 2.5x.
DUR_RATIO_WARN = 1.3
DUR_RATIO_ERROR = 1.6

# Difference in the fraction of clips under SHORT_CLIP_SEC, between conditions.
SHORT_FRAC_WARN = 0.20

# Genuine:impostor ratio spread across conditions.
RATIO_SPREAD_WARN = 1.5


@dataclass
class Finding:
    rule: str
    severity: str          # "error" | "warning" | "info"
    message: str
    evidence: dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.rule}] {self.severity.upper()}: {self.message}"


@dataclass
class Trial:
    condition: str
    label: str                 # "genuine" | "impostor"
    enroll_speaker: str = "unknown"
    test_speaker: str = "unknown"
    duration: float | None = None
    test_id: str = ""          # path or clip id, for the leak check

    @property
    def is_genuine(self) -> bool:
        return self.label.strip().lower().startswith("gen")


def _by_condition(trials: list[Trial]) -> dict[str, list[Trial]]:
    out: dict[str, list[Trial]] = defaultdict(list)
    for t in trials:
        out[t.condition].append(t)
    return dict(out)


def check_duration_confound(trials: list[Trial]) -> list[Finding]:
    """SPK-DUR-CONFOUND - does clip length track condition?"""
    per = {c: [t.duration for t in ts if t.duration and t.duration > 0]
           for c, ts in _by_condition(trials).items()}
    per = {c: v for c, v in per.items() if v}
    if len(per) < 2:
        return []

    meds = {c: median(v) for c, v in per.items()}
    shorts = {c: sum(1 for d in v if d < SHORT_CLIP_SEC) / len(v) for c, v in per.items()}
    lo_c = min(meds, key=lambda c: meds[c])
    hi_c = max(meds, key=lambda c: meds[c])
    ratio = meds[hi_c] / meds[lo_c] if meds[lo_c] > 0 else float("inf")

    findings: list[Finding] = []
    if ratio >= DUR_RATIO_WARN:
        sev = "error" if ratio >= DUR_RATIO_ERROR else "warning"
        findings.append(Finding(
            "SPK-DUR-CONFOUND", sev,
            f"clip duration tracks condition: median {meds[hi_c]:.2f}s in "
            f"'{hi_c}' against {meds[lo_c]:.2f}s in '{lo_c}' ({ratio:.2f}x). "
            f"Any difference between these conditions is equally explained by "
            f"length. Match durations, or report duration as a covariate and "
            f"say which you did.",
            {"median_seconds": {c: round(m, 3) for c, m in sorted(meds.items())},
             "ratio": round(ratio, 3),
             "longest": hi_c, "shortest": lo_c,
             "threshold_warn": DUR_RATIO_WARN, "threshold_error": DUR_RATIO_ERROR},
        ))

    spread = max(shorts.values()) - min(shorts.values())
    if spread >= SHORT_FRAC_WARN:
        hi = max(shorts, key=lambda c: shorts[c])
        lo = min(shorts, key=lambda c: shorts[c])
        findings.append(Finding(
            "SPK-DUR-CONFOUND", "warning",
            f"{shorts[hi]:.0%} of '{hi}' clips are under {SHORT_CLIP_SEC:g}s "
            f"against {shorts[lo]:.0%} of '{lo}'. Published error rates rise "
            f"steeply below {SHORT_CLIP_SEC:g}s, so these two conditions are "
            f"being scored in different regimes.",
            {"fraction_under_seconds": SHORT_CLIP_SEC,
             "by_condition": {c: round(f, 3) for c, f in sorted(shorts.items())},
             "spread": round(spread, 3)},
        ))
    return findings


def check_pool_drift(trials: list[Trial]) -> list[Finding]:
    """SPK-POOL-DRIFT - is the impostor population the same in every condition?"""
    pools = {c: {t.test_speaker for t in ts if not t.is_genuine}
             for c, ts in _by_condition(trials).items()}
    pools = {c: p for c, p in pools.items() if p}
    if len(pools) < 2:
        return []

    common = set.intersection(*pools.values())
    union = set.union(*pools.values())
    if common == union:
        return []

    missing = {c: sorted(union - p) for c, p in sorted(pools.items()) if union - p}
    return [Finding(
        "SPK-POOL-DRIFT", "error",
        f"the impostor set is not the same in every condition: "
        f"{len(union)} speakers appear somewhere, only {len(common)} appear "
        f"everywhere. Cross-condition comparison would partly measure which "
        f"impostors were available. Restrict the pool to speakers present in "
        f"every condition, and report the exclusions.",
        {"n_union": len(union), "n_common": len(common),
         "absent_by_condition": missing,
         "sizes": {c: len(p) for c, p in sorted(pools.items())}},
    )]


def check_enroll_leak(trials: list[Trial], enrollment: dict[str, list[str]] | None) -> list[Finding]:
    """SPK-ENROLL-LEAK - does any enrollment clip reappear as a test clip?"""
    if not enrollment:
        return []
    enrolled = {spk: set(paths) for spk, paths in enrollment.items()}
    hits: dict[str, list[str]] = defaultdict(list)
    for t in trials:
        if not t.test_id:
            continue
        for spk, paths in enrolled.items():
            if t.test_id in paths:
                hits[spk].append(t.test_id)
    if not hits:
        return []
    total = sum(len(v) for v in hits.values())
    return [Finding(
        "SPK-ENROLL-LEAK", "error",
        f"{total} test clip(s) across {len(hits)} speaker(s) are also enrollment "
        f"clips. Similarity to material already inside the enrollment centroid "
        f"is not a measurement of anything.",
        {"by_speaker": {s: sorted(v)[:5] for s, v in sorted(hits.items())},
         "n_leaked": total},
    )]


def check_ratio_drift(trials: list[Trial]) -> list[Finding]:
    """SPK-RATIO-DRIFT - is the genuine:impostor ratio stable across conditions?"""
    per = _by_condition(trials)
    ratios: dict[str, float] = {}
    for c, ts in per.items():
        g = sum(1 for t in ts if t.is_genuine)
        i = len(ts) - g
        if g and i:
            ratios[c] = i / g
    if len(ratios) < 2:
        return []
    hi = max(ratios, key=lambda c: ratios[c])
    lo = min(ratios, key=lambda c: ratios[c])
    spread = ratios[hi] / ratios[lo] if ratios[lo] > 0 else float("inf")
    if spread < RATIO_SPREAD_WARN:
        return []
    return [Finding(
        "SPK-RATIO-DRIFT", "warning",
        f"the impostor-to-genuine ratio varies {spread:.2f}x across conditions "
        f"({ratios[hi]:.2f} in '{hi}', {ratios[lo]:.2f} in '{lo}'). An equal "
        f"error rate is read off the crossing of two distributions; sampling one "
        f"side differently per condition moves the operating point with it.",
        {"impostor_per_genuine": {c: round(r, 3) for c, r in sorted(ratios.items())},
         "spread": round(spread, 3), "threshold": RATIO_SPREAD_WARN},
    )]


def check_design(
    trials: list[Trial],
    enrollment: dict[str, list[str]] | None = None,
) -> tuple[list[Finding], dict[str, object]]:
    """Run every design rule. Returns (findings, summary)."""
    per = _by_condition(trials)
    durs = [t.duration for t in trials if t.duration and t.duration > 0]
    summary: dict[str, object] = {
        "n_trials": len(trials),
        "n_conditions": len(per),
        "conditions": sorted(per),
        "n_genuine": sum(1 for t in trials if t.is_genuine),
        "n_impostor": sum(1 for t in trials if not t.is_genuine),
        "n_speakers": len({t.enroll_speaker for t in trials}),
        "has_duration": bool(durs),
        "has_enrollment": bool(enrollment),
    }
    if durs:
        summary["median_duration_sec"] = round(median(durs), 3)

    ran, skipped = [], []
    findings: list[Finding] = []

    if durs:
        findings += check_duration_confound(trials)
        ran.append("duration-confound")
    else:
        skipped.append("duration-confound (no duration column)")

    if any(not t.is_genuine for t in trials):
        findings += check_pool_drift(trials)
        findings += check_ratio_drift(trials)
        ran += ["pool-drift", "ratio-drift"]
    else:
        skipped += ["pool-drift (no impostor trials)",
                    "ratio-drift (no impostor trials)"]

    if enrollment:
        findings += check_enroll_leak(trials, enrollment)
        ran.append("enroll-leak")
    else:
        skipped.append("enroll-leak (no --enrollment given)")

    summary["checks_ran"] = ran
    summary["checks_skipped"] = skipped
    return findings, summary
