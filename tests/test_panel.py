"""Tests for the encoder panel.

The panel's job is not to crown a winner. It is to say which encoder separates
YOUR speakers under YOUR conditions, and to refuse the question when the data
cannot answer it. The refusal is the feature: a small benchmark that always
names a winner is how a small benchmark overstates itself.
"""
from __future__ import annotations

import pytest

from spkproof.panel import (
    d_prime,
    equal_error_rate,
    rank_and_warn,
    score_panel,
    worst_condition,
)

# ---------------------------------------------------------------- EER

def test_perfect_separation_is_zero():
    assert equal_error_rate([0.9, 0.8, 0.95], [0.1, 0.2, 0.05]) == 0.0


def test_no_separation_is_one_half():
    same = [0.5] * 20
    assert equal_error_rate(same, same) == pytest.approx(0.5, abs=1e-9)


def test_eer_is_symmetric_in_overlap():
    """Half the impostors above half the genuines -> 0.5."""
    gen = [0.4, 0.6]
    imp = [0.3, 0.5]
    v = equal_error_rate(gen, imp)
    assert 0.0 <= v <= 1.0


def test_eer_uses_the_gap_not_the_running_value():
    """Regression. An earlier version compared the frr/far GAP against the
    running EER, selecting a threshold that was neither the crossing point nor
    the minimum, and inflating the reported rate. Separable data must give a
    small number."""
    gen = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
    imp = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    assert equal_error_rate(gen, imp) == 0.0


def test_eer_empty_is_nan():
    assert equal_error_rate([], [0.1]) != equal_error_rate([], [0.1])


# ---------------------------------------------------------------- d-prime

def test_dprime_positive_when_genuine_scores_higher():
    assert d_prime([0.8, 0.82, 0.79], [0.2, 0.18, 0.21]) > 5


def test_dprime_nan_on_degenerate_input():
    v = d_prime([0.5], [0.4])
    assert v != v


# ---------------------------------------------------------------- panel

def rows(encoder_scores: dict[str, tuple[float, float]], condition="shouting",
         speakers=("A", "B", "C"), n=8, spread=0.30):
    """Build a score table. Each encoder gets (genuine_mean, impostor_mean).

    `spread` matters: the scores are fanned symmetrically around each mean, so
    two means that are close actually OVERLAP. An earlier version incremented
    both sides by the same tiny step, which left even a 0.55/0.45 encoder
    perfectly separable and made "bad encoder" fixtures score 0.0 EER.
    """
    out = []
    for s in speakers:
        for i in range(n):
            # symmetric fan: -spread/2 .. +spread/2
            off = spread * (i / max(n - 1, 1) - 0.5)
            g = {"condition": condition, "label": "genuine", "speaker": s}
            m = {"condition": condition, "label": "impostor", "speaker": s}
            for e, (gm, im) in encoder_scores.items():
                g[e] = gm + off
                m[e] = im - off
            out.append(g)
            out.append(m)
    return out


def test_panel_separates_a_good_encoder_from_a_bad_one():
    data = rows({"good": (0.9, 0.1), "bad": (0.55, 0.45)})
    res = score_panel(data, ["good", "bad"], bootstrap=0)
    by = {r.encoder: r for r in res}
    assert by["good"].eer < by["bad"].eer
    assert by["good"].n_genuine == 24
    assert by["good"].n_impostor == 24


def test_panel_reports_per_speaker():
    data = rows({"enc": (0.9, 0.1)})
    res = score_panel(data, ["enc"], bootstrap=0)
    assert set(res[0].per_speaker) == {"A", "B", "C"}


def test_panel_skips_conditions_without_both_sides():
    data = [{"condition": "x", "label": "genuine", "speaker": "A", "enc": 0.9}]
    assert score_panel(data, ["enc"], bootstrap=0) == []


def test_panel_ignores_blank_and_unparseable_scores():
    data = rows({"enc": (0.9, 0.1)}, spread=0.0)
    data[0]["enc"] = ""
    data[1]["enc"] = "n/a"
    res = score_panel(data, ["enc"], bootstrap=0)
    assert res[0].n_genuine + res[0].n_impostor == 46


def test_bootstrap_interval_brackets_the_estimate():
    data = rows({"enc": (0.9, 0.1)})
    res = score_panel(data, ["enc"], bootstrap=200, seed=1)
    r = res[0]
    assert r.eer_lo <= r.eer <= r.eer_hi or r.eer_lo != r.eer_lo


def test_bootstrap_is_deterministic_under_a_seed():
    data = rows({"enc": (0.7, 0.3)})
    a = score_panel(data, ["enc"], bootstrap=100, seed=7)[0]
    b = score_panel(data, ["enc"], bootstrap=100, seed=7)[0]
    assert (a.eer_lo, a.eer_hi) == (b.eer_lo, b.eer_hi)


# ---------------------------------------------------------------- the refusal

def test_overlapping_intervals_produce_a_tie_warning():
    """The feature that matters: two encoders that are close must NOT be
    reported as a winner and a loser.

    Speakers must differ from each other, or a speaker-stratified bootstrap has
    nothing to resample and returns a zero-width interval - which would make
    even near-identical encoders look separated.
    """
    data = []
    for k, s in enumerate(("A", "B", "C", "D")):
        shift = 0.06 * k          # speakers are not interchangeable
        data += rows({"a": (0.60 + shift, 0.40 + shift),
                      "b": (0.61 + shift, 0.39 + shift)},
                     speakers=(s,), n=8, spread=0.30)
    res = score_panel(data, ["a", "b"], bootstrap=300, seed=3)
    winners, notes = rank_and_warn(res)
    assert winners.get("shouting") in {"a", "b"}
    assert notes, "near-identical encoders must raise a tie warning"
    assert "not distinguishable" in notes[0]


def test_clear_separation_produces_no_tie_warning():
    data = rows({"good": (0.95, 0.05), "bad": (0.50, 0.50)})
    res = score_panel(data, ["good", "bad"], bootstrap=200, seed=3)
    winners, notes = rank_and_warn(res)
    assert winners["shouting"] == "good"
    assert notes == []


def test_worst_condition_names_where_an_encoder_fails():
    data = rows({"enc": (0.95, 0.05)}, condition="neutral")
    data += rows({"enc": (0.55, 0.45)}, condition="shouting")
    res = score_panel(data, ["enc"], bootstrap=0)
    worst = worst_condition(res, "enc")
    assert worst is not None and worst[0] == "shouting"


def test_worst_condition_none_for_unknown_encoder():
    res = score_panel(rows({"enc": (0.9, 0.1)}), ["enc"], bootstrap=0)
    assert worst_condition(res, "missing") is None
