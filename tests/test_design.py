"""Tests for the design checks.

Each rule exists because it caught a real error. Each test therefore has two
halves: the fault is detected, AND a sound design stays silent. A checker that
only ever fires is not a checker.
"""
from __future__ import annotations

import pytest

from spkproof.design import (
    Trial,
    check_design,
    check_duration_confound,
    check_enroll_leak,
    check_pool_drift,
    check_ratio_drift,
)


def mk(condition, label, speaker="A", test_speaker=None, duration=2.0, test_id=""):
    return Trial(condition=condition, label=label, enroll_speaker=speaker,
                 test_speaker=test_speaker or (speaker if label == "genuine" else "Z"),
                 duration=duration, test_id=test_id)


def balanced(conditions=("neutral", "shouting"), speakers=("A", "B", "C"),
             duration=2.0, n=10):
    """A design with nothing wrong with it."""
    out = []
    for c in conditions:
        for s in speakers:
            for i in range(n):
                out.append(mk(c, "genuine", s, s, duration, f"{s}_{c}_{i}"))
                for other in speakers:
                    if other != s:
                        out.append(mk(c, "impostor", s, other, duration,
                                      f"{other}_{c}_{i}"))
    return out


# ---------------------------------------------------------------- duration

def test_duration_confound_fires_when_length_tracks_condition():
    trials = ([mk("neutral", "genuine", duration=1.4) for _ in range(30)]
              + [mk("shouting", "genuine", duration=3.5) for _ in range(30)])
    found = check_duration_confound(trials)
    assert any(f.rule == "SPK-DUR-CONFOUND" for f in found)
    top = [f for f in found if f.severity == "error"]
    assert top, "a 2.5x median ratio must be an error, not a warning"
    assert top[0].evidence["longest"] == "shouting"
    assert top[0].evidence["ratio"] == pytest.approx(2.5, abs=0.01)


def test_duration_confound_silent_when_matched():
    trials = ([mk("neutral", "genuine", duration=2.1) for _ in range(30)]
              + [mk("shouting", "genuine", duration=2.2) for _ in range(30)])
    assert check_duration_confound(trials) == []


def test_short_clip_fraction_reported_separately():
    """Medians can match while one condition is full of sub-2s clips."""
    trials = ([mk("a", "genuine", duration=d) for d in [1.0] * 20 + [3.4] * 20]
              + [mk("b", "genuine", duration=2.2) for _ in range(40)])
    found = check_duration_confound(trials)
    assert any("under" in f.message for f in found)


def test_duration_check_skipped_without_durations():
    trials = [Trial("neutral", "genuine"), Trial("shouting", "genuine")]
    assert check_duration_confound(trials) == []


# ---------------------------------------------------------------- pool drift

def test_pool_drift_fires_when_impostors_differ_by_condition():
    trials = [mk("neutral", "impostor", test_speaker=s) for s in ("X", "Y", "Z")]
    trials += [mk("shouting", "impostor", test_speaker=s) for s in ("X", "Y")]
    found = check_pool_drift(trials)
    assert len(found) == 1
    assert found[0].severity == "error"
    assert "Z" in found[0].evidence["absent_by_condition"]["shouting"]


def test_pool_drift_silent_when_pool_is_fixed():
    trials = [mk(c, "impostor", test_speaker=s)
              for c in ("neutral", "shouting") for s in ("X", "Y", "Z")]
    assert check_pool_drift(trials) == []


def test_pool_drift_needs_two_conditions():
    trials = [mk("neutral", "impostor", test_speaker=s) for s in ("X", "Y")]
    assert check_pool_drift(trials) == []


# ---------------------------------------------------------------- leakage

def test_enroll_leak_fires_when_a_test_clip_was_enrolled():
    trials = [mk("neutral", "genuine", test_id="a1.wav"),
              mk("neutral", "genuine", test_id="a2.wav")]
    found = check_enroll_leak(trials, {"A": ["a1.wav", "a9.wav"]})
    assert len(found) == 1
    assert found[0].severity == "error"
    assert found[0].evidence["n_leaked"] == 1


def test_enroll_leak_silent_when_disjoint():
    trials = [mk("neutral", "genuine", test_id="a2.wav")]
    assert check_enroll_leak(trials, {"A": ["a1.wav"]}) == []


def test_enroll_leak_skipped_without_enrollment():
    assert check_enroll_leak([mk("neutral", "genuine", test_id="a.wav")], None) == []


# ---------------------------------------------------------------- ratio

def test_ratio_drift_fires_when_sampling_differs_by_condition():
    trials = [mk("a", "genuine") for _ in range(10)]
    trials += [mk("a", "impostor") for _ in range(10)]      # 1:1
    trials += [mk("b", "genuine") for _ in range(10)]
    trials += [mk("b", "impostor") for _ in range(40)]      # 4:1
    found = check_ratio_drift(trials)
    assert len(found) == 1
    assert found[0].evidence["spread"] == pytest.approx(4.0, abs=0.01)


def test_ratio_drift_silent_when_stable():
    trials = []
    for c in ("a", "b"):
        trials += [mk(c, "genuine") for _ in range(10)]
        trials += [mk(c, "impostor") for _ in range(20)]
    assert check_ratio_drift(trials) == []


# ---------------------------------------------------------------- end to end

def test_sound_design_produces_no_findings():
    findings, summary = check_design(balanced(),
                                     {"A": ["enrolled_only.wav"]})
    assert findings == [], [str(f) for f in findings]
    assert summary["n_conditions"] == 2
    assert "duration-confound" in summary["checks_ran"]
    assert "enroll-leak" in summary["checks_ran"]


def test_summary_names_what_it_could_not_check():
    """A clean verdict must not imply coverage the run did not have."""
    trials = [Trial("neutral", "genuine"), Trial("neutral", "genuine")]
    _, summary = check_design(trials, None)
    skipped = " ".join(summary["checks_skipped"])
    assert "duration" in skipped
    assert "enroll-leak" in skipped
    assert "impostor" in skipped


def test_broken_design_reports_every_independent_fault():
    """Faults are independent: one does not mask another."""
    # neutral is short, shouting is long -> duration confound
    trials = [mk("neutral", "genuine", duration=1.4) for _ in range(30)]
    trials += [mk("neutral", "impostor", test_speaker=s, duration=1.4)
               for s in ("X", "Y", "Z") for _ in range(10)]
    trials += [mk("shouting", "genuine", duration=3.6) for _ in range(30)]
    # ...and Z is missing from the shouting impostor pool -> pool drift
    trials += [mk("shouting", "impostor", test_speaker=s, duration=3.6)
               for s in ("X", "Y") for _ in range(10)]
    findings, summary = check_design(trials, None)
    rules = {f.rule for f in findings}
    assert "SPK-DUR-CONFOUND" in rules
    assert "SPK-POOL-DRIFT" in rules
    assert summary["n_conditions"] == 2
