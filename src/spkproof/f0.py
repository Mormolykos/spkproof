"""F0-tracking contamination checks for speaker-embedding studies.

Pitch trackers fail on rough phonation, and they fail *in one direction*: octave
errors push the estimate up, essentially never down. When rough and modal
phonation are pooled in one analysis, the resulting error is not noise. It is
differential measurement error, correlated with condition, concentrated in a
subset of the data, and loaded onto one side of any directional comparison.

Reference: Gkilis (2026), doi:10.5281/zenodo.21921958, Section 6.5. In that
dataset every one of ten impossible observations fell in rough phonation and
none in modal (Fisher p = 2.4e-11), and those observations carried 2.7x the
leverage of the rest.

The checks here need only what a study already has: an F0 estimate, a condition
label, and a speaker label. No audio, no encoder, no model refit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median

from .stats import binom_sign_test, fisher_exact_2x2

# Beyond this, a within-speaker deviation in connected speech is not a
# production the speaker made. An octave is 12; two octaves is 24. The default
# sits between them so it catches doubling-plus while never firing on a real
# singer's range.
DEFAULT_CEILING_ST = 20.0

# How close a ratio must sit to an integer multiple to be called a harmonic
# error rather than a coincidence. 3% is roughly half a semitone.
HARMONIC_TOL = 0.03


@dataclass
class Finding:
    rule: str
    severity: str          # "error" | "warning" | "info"
    message: str
    evidence: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.rule}] {self.severity.upper()}: {self.message}"


@dataclass
class Utterance:
    f0: float
    condition: str
    speaker: str = "unknown"
    # Optional enrollment group - session, corpus, recording day. A speaker
    # recorded twice has two baselines, not one: comparing today's utterance
    # against a median pooled across both sessions manufactures deviation that
    # the speaker never produced.
    group: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.speaker, self.group)

    @property
    def valid(self) -> bool:
        return isinstance(self.f0, (int, float)) and math.isfinite(self.f0) and self.f0 > 0


def _baselines(utts: list[Utterance], reference: str) -> dict[tuple[str, str], float]:
    """Reference F0 per (speaker, group): the median of the reference condition
    within that group if it has one, otherwise that group's overall median."""
    out: dict[tuple[str, str], float] = {}
    for key in {u.key for u in utts}:
        mine = [u.f0 for u in utts if u.key == key and u.valid]
        ref = [u.f0 for u in utts if u.key == key and u.condition == reference and u.valid]
        if ref:
            out[key] = median(ref)
        elif mine:
            out[key] = median(mine)
    return out


def semitones(f0: float, baseline: float) -> float:
    return 12.0 * math.log2(f0 / baseline)


def check_f0(
    utterances: list[Utterance],
    ceiling_st: float = DEFAULT_CEILING_ST,
    reference: str = "clean",
) -> tuple[list[Finding], dict]:
    """Run every F0 contamination check.

    Returns (findings, summary). An empty findings list means no contamination
    signature was detected — not that the data is correct.
    """
    usable = [u for u in utterances if u.valid]
    findings: list[Finding] = []
    summary: dict = {
        "n_total": len(utterances),
        "n_usable": len(usable),
        "n_dropped": len(utterances) - len(usable),
        "ceiling_st": ceiling_st,
    }

    if len(usable) < 4:
        findings.append(Finding(
            "SPK-F0-INSUFFICIENT", "warning",
            f"only {len(usable)} usable utterances; contamination cannot be assessed",
        ))
        return findings, summary

    base = _baselines(usable, reference)
    if not any(u.condition == reference for u in usable):
        findings.append(Finding(
            "SPK-F0-NOREF", "info",
            f"no utterance labelled '{reference}'; per-speaker median used as baseline instead. "
            f"Deviations are relative to the speaker's own centre, not to a clean condition.",
            {"reference_sought": reference},
        ))

    rows = []
    for u in usable:
        b = base.get(u.key)
        if not b or b <= 0:
            continue
        rows.append((u, semitones(u.f0, b), u.f0 / b))
    if not rows:
        findings.append(Finding("SPK-F0-INSUFFICIENT", "warning", "no usable baseline per speaker"))
        return findings, summary
    summary["n_baselines"] = len(base)

    impossible = [(u, st, r) for u, st, r in rows if abs(st) > ceiling_st]
    summary["n_impossible"] = len(impossible)
    summary["max_abs_semitones"] = round(max(abs(st) for _, st, _ in rows), 2)

    # --- 1. values outside anything a speaker produced -----------------------
    if impossible:
        worst = max(impossible, key=lambda t: abs(t[1]))
        findings.append(Finding(
            "SPK-F0-RANGE", "error",
            f"{len(impossible)} of {len(rows)} utterances deviate by more than "
            f"{ceiling_st:g} semitones from their speaker's baseline "
            f"(worst {worst[1]:+.1f} st = {worst[2]:.1f}x in frequency). "
            f"No speaker produced that; these are tracking failures.",
            {"n_impossible": len(impossible), "n_total": len(rows),
             "worst_semitones": round(worst[1], 2), "worst_ratio": round(worst[2], 2),
             "conditions": sorted({u.condition for u, _, _ in impossible})},
        ))

    # --- 2. octave / harmonic signature --------------------------------------
    harmonic = []
    for u, _st, r in impossible:
        for k in (2, 3, 4, 0.5):
            if abs(r - k) / k <= HARMONIC_TOL:
                harmonic.append((u, r, k))
                break
    if harmonic:
        findings.append(Finding(
            "SPK-F0-HARMONIC", "error",
            f"{len(harmonic)} of {len(impossible)} impossible values sit within "
            f"{HARMONIC_TOL:.0%} of an exact harmonic multiple of the speaker's baseline. "
            f"That is the signature of octave error, not of unusual production.",
            {"multiples": sorted({k for _, _, k in harmonic})},
        ))

    # --- 3. confinement to particular conditions (the dangerous one) ---------
    if impossible:
        bad_conditions = {u.condition for u, _, _ in impossible}
        for cond in sorted(bad_conditions):
            a = sum(1 for u, _, _ in impossible if u.condition == cond)
            b = len(impossible) - a
            c = sum(1 for u, st, _ in rows if u.condition == cond and abs(st) <= ceiling_st)
            d = len(rows) - len(impossible) - c
            p, odds = fisher_exact_2x2(a, b, c, d)
            if p < 0.05:
                findings.append(Finding(
                    "SPK-F0-CONFINED", "error",
                    f"tracking failures are concentrated in condition '{cond}' "
                    f"(Fisher exact p = {p:.2e}, odds ratio "
                    f"{'inf' if odds == float('inf') else f'{odds:.1f}'}). "
                    f"This is differential measurement error: the error is correlated with "
                    f"condition, so pooling '{cond}' with the rest biases any model that "
                    f"uses F0 as a predictor. Dropping it changes the estimand, not just the noise.",
                    {"condition": cond, "p_value": p,
                     "odds_ratio": None if odds == float("inf") else round(odds, 2),
                     "table": {"impossible_in": a, "impossible_out": b,
                               "plausible_in": c, "plausible_out": d}},
                ))

    # --- 4. directionality ---------------------------------------------------
    if len(impossible) >= 4:
        up = sum(1 for _, st, _ in impossible if st > 0)
        p = binom_sign_test(up, len(impossible))
        if p < 0.05:
            side = "upward" if up > len(impossible) / 2 else "downward"
            findings.append(Finding(
                "SPK-F0-DIRECTIONAL", "error",
                f"{up} of {len(impossible)} tracking failures are {side} "
                f"(exact binomial p = {p:.3f}). Directional error does not cancel: it "
                f"loads onto one side of any up-versus-down comparison and can invert "
                f"the sign of a directional result.",
                {"n_up": up, "n_impossible": len(impossible), "p_value": round(p, 5)},
            ))

    summary["n_findings"] = len(findings)
    return findings, summary
