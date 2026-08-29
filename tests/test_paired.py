"""Tests for the inference the panel does: the resampling unit, the paired
test, and the multiplicity correction.

Each of these three was wrong in a study that had already written its results
down, and each was found by re-running that study's own frozen trial list.
These tests are the miniature version of that re-run: a fixture built so that
the wrong method gives a visibly different answer from the right one.
"""
from __future__ import annotations

import random

import pytest

from spkproof.panel import (
    RESAMPLE_DYADIC,
    RESAMPLE_ENROLL,
    _Cell,
    _weighted_eer,
    all_pairs,
    declare_family,
    equal_error_rate,
    family_summary,
    paired_compare,
    rank_and_warn,
    resample_note,
    score_panel,
)
from spkproof.stats import holm, holm_adjusted

SPEAKERS = ["S0", "S1", "S2", "S3", "S4", "S5"]
# Speakers differ a lot in how separable they are, which is what makes the
# marginal interval wide: a bootstrap draw of easy speakers gives a low EER and
# a draw of hard ones gives a high one.
MARGINS = [0.02, 0.06, 0.11, 0.17, 0.24, 0.32]


def graded(edges: dict[str, float], n: int = 8, spread: float = 0.30,
           condition: str = "shouting") -> list[dict]:
    """A trial graph: every speaker enrols, and is an impostor under every
    other. `edges` gives each encoder a constant advantage, so encoder order is
    the same for every speaker and only the SIZE of the difference is in
    question."""
    rows: list[dict] = []
    for s, m in zip(SPEAKERS, MARGINS, strict=True):
        for i in range(n):
            off = spread * (i / (n - 1) - 0.5)
            row = {"condition": condition, "label": "genuine",
                   "enroll_speaker": s, "test_speaker": s}
            for e, adv in edges.items():
                row[e] = 0.5 + m + adv + off
            rows.append(row)
            for t in SPEAKERS:
                if t == s:
                    continue
                imp = {"condition": condition, "label": "impostor",
                       "enroll_speaker": s, "test_speaker": t}
                for e, adv in edges.items():
                    imp[e] = 0.5 - m - adv - off
                rows.append(imp)
    return rows


def wolf(n: int = 8, spread: float = 0.30) -> list[dict]:
    """One speaker whose voice scores high against everybody else's model.

    This is the shape the resampling unit exists for. The panel's error rate
    hangs on that one person - but they appear in the trial list as an IMPOSTOR
    under the five others, so a bootstrap that resamples enrollment identities
    keeps them in every single draw and reports an interval that has never seen
    the data without them."""
    rows: list[dict] = []
    for s in SPEAKERS:
        for i in range(n):
            off = spread * (i / (n - 1) - 0.5)
            rows.append({"condition": "c", "label": "genuine", "enroll_speaker": s,
                         "test_speaker": s, "enc": 0.70 + off})
            for t in SPEAKERS:
                if t == s:
                    continue
                rows.append({"condition": "c", "label": "impostor",
                             "enroll_speaker": s, "test_speaker": t,
                             "enc": 0.30 + (0.45 if t == "S5" else 0.0) + off})
    return rows


def width(result) -> float:
    return result.eer_hi - result.eer_lo


# ------------------------------------------------- the fast estimator is the slow one

def test_the_weighted_estimator_reproduces_the_reference_exactly():
    """The bootstrap does not call `equal_error_rate`: it uses a weighted form
    that costs O(n) per draw instead of O(n^2). That is only worth having if it
    is the same estimator, so this is checked rather than assumed - to the bit,
    not to a tolerance, on data with the ties and duplicates that make an EER
    threshold ambiguous."""
    rng = random.Random(4)
    for _ in range(25):
        # a coarse grid, so scores collide and the threshold grid has plateaus
        gen = [round(rng.uniform(0.3, 0.9), 2) for _ in range(rng.randint(2, 30))]
        imp = [round(rng.uniform(0.1, 0.7), 2) for _ in range(rng.randint(2, 30))]
        cell = _Cell([(v, 0) for v in gen], [(v, 0, 0) for v in imp])
        reference = equal_error_rate(gen, imp)
        assert _weighted_eer(cell, [1.0], dyadic=True) == reference
        assert _weighted_eer(cell, [1.0], dyadic=False) == reference


def test_a_speaker_who_is_not_drawn_contributes_nothing_to_either_side():
    """The property the enrollment-only scheme lacks. Zero weight on a speaker
    must remove their genuine trials AND the impostor trials where they are the
    impostor."""
    rows = wolf()
    full = score_panel(rows, ["enc"], bootstrap=0)[0].eer
    kept = [r for r in rows if r["enroll_speaker"] != "S5" and r["test_speaker"] != "S5"]
    dropped = score_panel(kept, ["enc"], bootstrap=0)[0].eer
    assert full != dropped

    cell_rows = score_panel(rows, ["enc"], bootstrap=1, seed=0)
    assert cell_rows[0].resample_unit == RESAMPLE_DYADIC


# ------------------------------------------------- the resampling unit

def test_the_enrollment_only_interval_is_too_narrow_when_one_speaker_carries_the_result():
    """Both schemes run on identical draws - same seed, same speaker count - so
    the difference in width is the scheme and not bootstrap noise. On the study
    this rule came from the corrected interval was up to 1.58x wider; on a
    fixture built to isolate the mechanism it is more."""
    rows = wolf()
    correct = score_panel(rows, ["enc"], bootstrap=500, seed=1, resample="dyadic")[0]
    old = score_panel(rows, ["enc"], bootstrap=500, seed=1, resample="enroll")[0]
    assert correct.eer == old.eer                      # same point estimate
    assert width(correct) > 2 * width(old)
    assert correct.resample_unit == RESAMPLE_DYADIC
    assert old.resample_unit == RESAMPLE_ENROLL


def test_the_speaker_unit_is_chosen_automatically_when_the_table_allows_it():
    results = score_panel(wolf(), ["enc"], bootstrap=50, seed=1)
    assert results[0].resample_unit == RESAMPLE_DYADIC
    assert resample_note(results) is None


def test_a_table_without_test_speakers_gets_the_weaker_unit_and_is_told_so():
    rows = [{k: v for k, v in r.items() if k != "test_speaker"} for r in wolf()]
    results = score_panel(rows, ["enc"], bootstrap=50, seed=1)
    assert results[0].resample_unit == RESAMPLE_ENROLL
    note = resample_note(results)
    assert note is not None and "too narrow" in note


def test_one_missing_test_speaker_is_enough_to_fall_back():
    """A mixture of the two schemes is neither of them."""
    rows = wolf()
    rows[-1] = {**rows[-1], "test_speaker": ""}
    assert score_panel(rows, ["enc"], bootstrap=20)[0].resample_unit == RESAMPLE_ENROLL


def test_insisting_on_the_speaker_unit_fails_loudly_rather_than_silently_downgrading():
    rows = [{k: v for k, v in r.items() if k != "test_speaker"} for r in wolf()]
    with pytest.raises(ValueError, match="test_speaker"):
        score_panel(rows, ["enc"], bootstrap=20, resample="dyadic")


def test_an_unknown_resample_scheme_is_refused():
    with pytest.raises(ValueError, match="resample must be"):
        score_panel(wolf(), ["enc"], bootstrap=0, resample="whatever")


# ------------------------------------------------- paired against marginal

def test_the_paired_test_separates_encoders_whose_marginal_intervals_overlap():
    """The fix that made this library as strong as the method it documents.

    Two encoders scored on the SAME trials by the SAME speakers move together
    from draw to draw: most of the width of each marginal interval is the
    speakers, and it cancels in the difference. Asking whether the intervals
    overlap throws that away and answers a weaker question."""
    results = score_panel(graded({"a": 0.06, "b": 0.0}), ["a", "b"],
                          bootstrap=500, seed=1)
    a, b = results[0], results[1]
    assert a.eer_lo <= b.eer_hi and b.eer_lo <= a.eer_hi, "the marginals overlap"

    comparisons, _ = paired_compare(results)
    assert len(comparisons) == 1
    assert comparisons[0].survives
    assert comparisons[0].hi < 0                       # a is better, and it excludes zero


def test_rank_and_warn_no_longer_calls_that_pair_a_tie():
    results = score_panel(graded({"a": 0.06, "b": 0.0}), ["a", "b"],
                          bootstrap=500, seed=1)
    winners, notes = rank_and_warn(results)
    assert winners["shouting"] == "a"
    assert notes == []


def test_two_identical_encoders_are_still_a_tie():
    """The refusal has to survive becoming more powerful."""
    results = score_panel(graded({"a": 0.0, "b": 0.0}), ["a", "b"],
                          bootstrap=300, seed=1)
    _, notes = rank_and_warn(results)
    assert notes and "not distinguishable" in notes[0]
    assert "paired bootstrap" in notes[0]


def test_without_draws_the_marginal_overlap_is_used_and_labelled():
    results = score_panel(graded({"a": 0.0, "b": 0.0}), ["a", "b"], bootstrap=0)
    _, notes = rank_and_warn(results)
    assert notes == []                                 # no interval, no claim either way
    comparisons, warnings = paired_compare(results)
    assert comparisons == []
    assert "no bootstrap draws" in warnings[0]


# ------------------------------------------------- multiplicity

def four_encoders() -> list:
    # c and d are the same encoder twice, which is what makes one comparison in
    # this family a certain null and forces the step-down to stop there.
    return score_panel(graded({"a": 0.06, "b": 0.0, "c": -0.10, "d": -0.10}),
                       ["a", "b", "c", "d"], bootstrap=500, seed=1)


def test_correction_costs_the_family_a_comparison_it_would_have_reported():
    comparisons, _ = paired_compare(four_encoders())
    summary = family_summary(comparisons)
    assert summary["n_comparisons"] == 6
    assert summary["n_significant_uncorrected"] == 5
    assert summary["n_survive_holm"] == 4
    lost = [c for c in comparisons if c.significant_uncorrected and not c.survives]
    assert [(c.encoder_a, c.encoder_b) for c in lost] == [("a", "b")]


def test_the_same_comparison_survives_alone_and_dies_in_a_family():
    """Multiplicity is not a property of the comparison. It is a property of how
    many you ran, which is why the family has to be declared."""
    results = four_encoders()
    alone = declare_family("the close pair", [("shouting", "a", "b")],
                           kind="exploratory")
    solo, _ = paired_compare(results, alone)
    assert solo[0].survives
    assert solo[0].p_value == pytest.approx(
        next(c.p_value for c in paired_compare(results)[0]
             if (c.encoder_a, c.encoder_b) == ("a", "b")))
    assert not next(c for c in paired_compare(results)[0]
                    if (c.encoder_a, c.encoder_b) == ("a", "b")).survives


def test_a_confirmatory_family_that_is_a_subset_is_warned_about():
    results = four_encoders()
    subset = declare_family("the two I liked", [("shouting", "a", "b")])
    _, notes = paired_compare(results, subset)
    assert any("understates the multiplicity" in n for n in notes)


def test_an_exploratory_family_is_allowed_to_be_a_subset():
    results = four_encoders()
    subset = declare_family("pre-registered contrast", [("shouting", "a", "b")],
                            kind="exploratory")
    _, notes = paired_compare(results, subset)
    assert not any("understates the multiplicity" in n for n in notes)


def test_a_family_kind_that_is_not_one_of_the_two_is_refused():
    with pytest.raises(ValueError, match="confirmatory or exploratory"):
        declare_family("x", [], kind="whatever")


def test_a_bootstrap_too_small_for_the_family_is_reported_not_hidden():
    """A bootstrap p cannot fall below 2/(B+1). With enough comparisons, Holm's
    smallest threshold drops under that floor and NOTHING can survive at any
    effect size - which reads exactly like a clean negative result."""
    results = score_panel(graded({"a": 0.06, "b": 0.0, "c": -0.10, "d": -0.10}),
                          ["a", "b", "c", "d"], bootstrap=60, seed=1)
    _, notes = paired_compare(results)
    floor = [n for n in notes if "cannot resolve this family" in n]
    assert floor and "Raise the bootstrap" in floor[0]


def test_all_pairs_never_crosses_conditions():
    """Two encoders under two conditions were scored on different trials, so
    their draw-by-draw difference is not paired and the test does not apply."""
    rows = graded({"a": 0.06, "b": 0.0}, condition="neutral")
    rows += graded({"a": 0.06, "b": 0.0}, condition="shouting")
    results = score_panel(rows, ["a", "b"], bootstrap=20, seed=1)
    pairs = all_pairs(results)
    assert len(pairs) == 2
    assert {c for c, _, _ in pairs} == {"neutral", "shouting"}


def test_a_comparison_naming_an_encoder_that_was_not_scored_is_dropped_with_a_note():
    results = four_encoders()
    family = declare_family("typo", [("shouting", "a", "redimnet")], kind="exploratory")
    comparisons, notes = paired_compare(results, family)
    assert comparisons == []
    assert any("redimnet" in n for n in notes)


# ------------------------------------------------- the correction itself

def test_holm_is_the_step_down_and_stops_at_the_first_failure():
    p = [0.001, 0.02, 0.04, 0.9]
    # 4*0.001=0.004, 3*0.02=0.06 -> fails, and nothing after it can be rejected
    assert holm(p, 0.05) == [True, False, False, False]


def test_holm_adjusted_is_monotone_in_the_raw_p_values():
    p = [0.04, 0.001, 0.02, 0.9]
    adjusted = holm_adjusted(p)
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranked = [adjusted[i] for i in order]
    assert ranked == sorted(ranked)
    assert all(a >= b for a, b in zip(adjusted, p, strict=True))
    assert all(a <= 1.0 for a in adjusted)


def test_holm_on_a_single_hypothesis_changes_nothing():
    assert holm_adjusted([0.03]) == [0.03]
    assert holm([], 0.05) == []


# ------------------------------------------------- through the command line

def test_the_cli_reports_the_weaker_unit_as_a_finding(tmp_path):
    import csv

    from spkproof.cli import main

    rows = wolf()
    path = tmp_path / "scores.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert main(["compare-encoders", str(path), "--bootstrap", "50"]) == 0

    stripped = tmp_path / "no_test_speaker.csv"
    fields = [f for f in rows[0] if f != "test_speaker"]
    with open(stripped, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: r[k] for k in fields} for r in rows])
    assert main(["compare-encoders", str(stripped), "--bootstrap", "50"]) == 1
    assert main(["compare-encoders", str(stripped), "--bootstrap", "50",
                 "--resample", "dyadic"]) == 2
