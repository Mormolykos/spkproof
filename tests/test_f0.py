"""Tests for the F0 contamination checks.

The most important test here is test_reproduces_published_result: it locks
spkproof's output against the numbers in doi:10.5281/zenodo.21921958 Section 6.5.
If a refactor changes what the tool reports on that table, it has changed a
published claim, and the build fails.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from spkproof import Utterance, check_f0
from spkproof.stats import binom_sign_test, fisher_exact_2x2

FIXTURES = Path(__file__).parent / "fixtures"


def rules(findings) -> set[str]:
    return {f.rule for f in findings}


# --------------------------------------------------------------------------
# exact tests


def test_fisher_matches_known_value():
    # Fisher's own tea-tasting table: 3/3 correct out of 4+4.
    p, odds = fisher_exact_2x2(3, 1, 1, 3)
    assert p == pytest.approx(0.4857, abs=1e-4)
    assert odds == pytest.approx(9.0)


def test_fisher_zero_cell_gives_infinite_odds():
    p, odds = fisher_exact_2x2(10, 0, 0, 128)
    assert odds == float("inf")
    assert p < 1e-9


def test_fisher_degenerate_table_is_not_significant():
    assert fisher_exact_2x2(0, 0, 5, 5)[0] == 1.0


def test_binom_sign_test():
    assert binom_sign_test(5, 10) == 1.0
    assert binom_sign_test(10, 10) == pytest.approx(2 / 1024)
    assert binom_sign_test(0, 0) == 1.0


# --------------------------------------------------------------------------
# behaviour


def test_clean_data_produces_no_findings():
    utts = [Utterance(f0=100 + i, condition="clean", speaker="A") for i in range(10)]
    utts += [Utterance(f0=118 + i, condition="high", speaker="A") for i in range(10)]
    findings, summary = check_f0(utts)
    assert findings == []
    assert summary["n_impossible"] == 0


def test_detects_confined_directional_contamination():
    # 20 modal utterances near 100 Hz, plus 6 "rasp" utterances the tracker
    # doubled to ~400 Hz. That is +24 semitones, upward, all in one condition.
    utts = [Utterance(f0=100.0, condition="clean", speaker="A") for _ in range(20)]
    utts += [Utterance(f0=400.0, condition="rasp", speaker="A") for _ in range(6)]
    findings, summary = check_f0(utts)
    assert summary["n_impossible"] == 6
    assert "SPK-F0-RANGE" in rules(findings)
    assert "SPK-F0-CONFINED" in rules(findings)
    assert "SPK-F0-DIRECTIONAL" in rules(findings)
    confined = next(f for f in findings if f.rule == "SPK-F0-CONFINED")
    assert confined.evidence["condition"] == "rasp"
    assert confined.evidence["table"] == {"impossible_in": 6, "impossible_out": 0,
                                          "plausible_in": 0, "plausible_out": 20}


def test_harmonic_signature_identifies_octave_doubling():
    utts = [Utterance(f0=100.0, condition="clean", speaker="A") for _ in range(20)]
    utts += [Utterance(f0=400.0, condition="rasp", speaker="A") for _ in range(6)]
    findings, _ = check_f0(utts)
    h = next(f for f in findings if f.rule == "SPK-F0-HARMONIC")
    assert 4 in h.evidence["multiples"]


def test_group_separates_baselines_per_session():
    # Same speaker, two sessions in genuinely different registers, unbalanced -
    # the realistic case, where the larger session drags the pooled median onto
    # itself and the smaller one is then scored against a baseline that is not
    # its own. Grouped, both sessions are flat and nothing fires.
    a = [Utterance(f0=100.0, condition="clean", speaker="A", group="s1") for _ in range(10)]
    b = [Utterance(f0=500.0, condition="clean", speaker="A", group="s2") for _ in range(3)]

    grouped, gsum = check_f0(a + b)
    assert gsum["n_baselines"] == 2
    assert gsum["n_impossible"] == 0
    assert grouped == []

    # Pooled, the baseline collapses onto the majority session (median 100 Hz)
    # and the minority session reads as a +27.9 semitone rise nobody produced.
    pooled = [Utterance(u.f0, u.condition, u.speaker) for u in a + b]
    _, psum = check_f0(pooled)
    assert psum["n_baselines"] == 1
    assert psum["n_impossible"] == 3


def test_missing_and_nonpositive_f0_are_dropped_not_counted():
    utts = [Utterance(f0=100.0, condition="clean", speaker="A") for _ in range(8)]
    utts += [Utterance(f0=0.0, condition="rasp", speaker="A"),
             Utterance(f0=float("nan"), condition="rasp", speaker="A")]
    _, summary = check_f0(utts)
    assert summary["n_usable"] == 8
    assert summary["n_dropped"] == 2


def test_too_few_utterances_cannot_be_judged():
    findings, _ = check_f0([Utterance(f0=100.0, condition="clean", speaker="A")])
    assert "SPK-F0-INSUFFICIENT" in rules(findings)


def test_missing_reference_condition_is_reported_not_silently_substituted():
    utts = [Utterance(f0=100.0 + i, condition="rasp", speaker="A") for i in range(8)]
    findings, _ = check_f0(utts, reference="clean")
    assert "SPK-F0-NOREF" in rules(findings)


# --------------------------------------------------------------------------
# published-result regression


def test_reproduces_published_result():
    """Locks spkproof against doi:10.5281/zenodo.21921958 Section 6.5."""
    import csv
    rows = list(csv.DictReader(open(FIXTURES / "published_ecapa.csv", encoding="utf-8")))
    utts = []
    for r in rows:
        try:
            f0 = float(r["f0"])
        except ValueError:
            f0 = float("nan")
        utts.append(Utterance(f0=f0, condition=r["condition"],
                              speaker=r["speaker"], group=r["corpus"]))

    findings, summary = check_f0(utts)

    # The paper's own numbers.
    assert summary["n_usable"] == 233
    assert summary["n_impossible"] == 10
    assert summary["max_abs_semitones"] == pytest.approx(32.12, abs=0.01)

    assert {"SPK-F0-RANGE", "SPK-F0-CONFINED", "SPK-F0-DIRECTIONAL"} <= rules(findings)

    rng = next(f for f in findings if f.rule == "SPK-F0-RANGE")
    assert rng.evidence["worst_ratio"] == pytest.approx(6.4, abs=0.05)

    # Every impossible value upward - the octave-error signature.
    d = next(f for f in findings if f.rule == "SPK-F0-DIRECTIONAL")
    assert d.evidence["n_up"] == 10 and d.evidence["n_impossible"] == 10

    # Confinement to rough phonation, at the significance the paper reports.
    confined = [f for f in findings if f.rule == "SPK-F0-CONFINED"]
    assert any(f.evidence["condition"] == "rasp" for f in confined)
    assert all(f.evidence["p_value"] < 1e-6 for f in confined)


# --------------------------------------------------------------------------
# cli contract


def run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "spkproof.cli", *args],
        capture_output=True, text=True,
        cwd=Path(__file__).parent.parent,
        env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")},
    )


def test_cli_exit_1_on_findings():
    r = run_cli("check-f0", str(FIXTURES / "published_ecapa.csv"))
    assert r.returncode == 1
    assert "SPK-F0-CONFINED" in r.stdout


def test_cli_exit_2_on_missing_file():
    r = run_cli("check-f0", str(FIXTURES / "does_not_exist.csv"))
    assert r.returncode == 2


def test_cli_exit_2_on_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    r = run_cli("check-f0", str(p))
    assert r.returncode == 2
    assert "no F0 column" in r.stderr


def test_cli_json_is_parseable():
    import json
    r = run_cli("check-f0", str(FIXTURES / "published_ecapa.csv"), "--json")
    payload = json.loads(r.stdout)
    assert payload["summary"]["n_impossible"] == 10
    assert payload["columns_used"]["group"] == "corpus"
