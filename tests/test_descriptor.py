"""Tests for the F0 gate.

The gate's job is to reject a spectral descriptor that follows pitch. The one
thing that would make it useless is passing everything, so the first test here
is not about a descriptor at all: it is about whether the gate can still say no.
"""
from __future__ import annotations

import math
import wave

import pytest

from spkproof.descriptor import (
    CONTROL_NAME,
    SweepValue,
    check_descriptor,
    check_sweep_table,
    f0_sensitive_control,
    measure_sweep,
    sweep_signals,
    synthesise,
    write_wav,
)

# Small on purpose: three pitches over an octave and a half, one envelope, a
# quarter second. The gate's argument does not need 24 signals to be made, and
# a test suite that takes ten seconds gets run less often.
FAST = {"f0_hz": (110.0, 200.0, 320.0), "envelopes": ("tilt12",), "duration": 0.25}


def constant(samples, sample_rate):
    return 7.0


def rules(findings):
    return {f.rule for f in findings}


# ---------------------------------------------------------------- the gate itself

def test_the_gate_rejects_the_measure_that_is_broken_by_construction():
    """The sanity check that makes every other verdict readable. The control is
    a band mean over fixed frequencies of a spectrum whose harmonics move; if
    the sweep cannot reject THAT, an ADMISSIBLE verdict from it means nothing."""
    findings, summary = check_descriptor({"invariant": constant}, **FAST)
    control = summary["control"]
    assert control["name"] == CONTROL_NAME
    assert control["rejected"] is True
    assert control["budget"] > 1.5
    assert "SPK-DESC-BLIND" not in rules(findings)


def test_a_control_that_passes_voids_the_whole_run():
    """If the control survives, the finding is about the gate and not about any
    descriptor - and it has to say so, because the ADMISSIBLE verdicts printed
    next to it are worthless."""
    findings, summary = check_descriptor({"invariant": constant}, tolerance=1e6, **FAST)
    assert "SPK-DESC-BLIND" in rules(findings)
    blind = next(f for f in findings if f.rule == "SPK-DESC-BLIND")
    assert blind.severity == "error"
    assert "void" in blind.message


def test_the_control_is_rejected_on_every_envelope():
    """Rejection on ANY envelope is rejection, so passing on one arbitrary
    envelope is not passing. The control must fail all three, or the three are
    not testing anything different from each other."""
    findings, summary = check_descriptor({}, duration=0.25,
                                         f0_hz=(110.0, 200.0, 320.0))
    per_env = summary["budgets"][CONTROL_NAME]
    assert set(per_env) == {"flat", "tilt12", "formant"}
    assert all(v > 1.5 for v in per_env.values()), per_env


# ---------------------------------------------------------------- verdicts

def test_a_descriptor_that_follows_pitch_is_rejected_with_its_budget():
    findings, summary = check_descriptor(
        {"band_ratio": f0_sensitive_control}, control=False, **FAST)
    assert summary["rejected"] == ["band_ratio"]
    artifact = next(f for f in findings if f.rule == "SPK-DESC-ARTIFACT")
    assert artifact.severity == "error"
    assert artifact.evidence["descriptor"] == "band_ratio"
    # The number, not just the verdict: an effect on this measure has to be read
    # against it.
    assert artifact.evidence["budget"] > 1.5
    assert f"{artifact.evidence['budget']:.2f}" in artifact.message


def test_an_invariant_descriptor_is_admissible():
    findings, summary = check_descriptor({"invariant": constant}, **FAST)
    assert summary["admissible"] == ["invariant"]
    assert summary["rejected"] == []
    assert "SPK-DESC-ARTIFACT" not in rules(findings)


def test_a_constant_descriptor_is_flagged_rather_than_congratulated():
    """A function that ignores its input passes this gate perfectly. So does a
    good descriptor. The output has to distinguish them."""
    findings, _ = check_descriptor({"invariant": constant}, **FAST)
    assert "SPK-DESC-CONSTANT" in rules(findings)


def test_a_descriptor_that_raises_is_reported_and_does_not_stop_the_sweep():
    def explodes(samples, sample_rate):
        raise ZeroDivisionError("no voiced frames")

    findings, summary = check_descriptor(
        {"broken": explodes, "invariant": constant}, **FAST)
    assert "SPK-DESC-ERROR" in rules(findings)
    error = next(f for f in findings if f.rule == "SPK-DESC-ERROR")
    assert "ZeroDivisionError" in str(error.evidence["error"])
    # the other descriptor still got judged
    assert summary["admissible"] == ["invariant"]


def test_a_sweep_that_does_not_sweep_cannot_judge_anything():
    rows = [SweepValue(f0=200.0, envelope="flat", values={"x": 1.0}),
            SweepValue(f0=200.0, envelope="tilt12", values={"x": 9.0})]
    findings, summary = check_sweep_table(rows, control_name=None)
    assert rules(findings) == {"SPK-DESC-INSUFFICIENT"}
    assert "budgets" not in summary


def test_a_blank_column_is_unjudged_not_admissible():
    """The difference between "I checked and it is fine" and "I could not
    check" is the whole contract."""
    rows = [SweepValue(f0=f0, envelope="flat", values={}) for f0 in (110.0, 320.0)]
    rows[0].values["measured"] = 1.0
    findings, summary = check_sweep_table(rows, control_name=None)
    assert "SPK-DESC-ERROR" in rules(findings)
    assert summary["admissible"] == []
    assert summary["rejected"] == []


def test_the_name_of_the_control_is_reserved():
    with pytest.raises(ValueError, match=CONTROL_NAME):
        check_descriptor({CONTROL_NAME: constant}, **FAST)


# ---------------------------------------------------------------- the signals

def test_the_sweep_holds_the_envelope_and_moves_only_the_pitch():
    signals = sweep_signals(f0_hz=(110.0, 320.0), envelopes=("flat", "formant"),
                            duration=0.25)
    assert len(signals) == 4
    assert {s.f0 for s in signals} == {110.0, 320.0}
    assert {s.envelope for s in signals} == {"flat", "formant"}
    # equal RMS, so a level-invariant descriptor is not being asked to survive a
    # level change as well as a pitch change
    for s in signals:
        rms = math.sqrt(sum(v * v for v in s.samples) / len(s.samples))
        assert rms == pytest.approx(0.05, rel=1e-6)


def test_synthesis_is_deterministic_under_a_seed():
    a = synthesise(200.0, "tilt12", duration=0.1, seed=3)
    b = synthesise(200.0, "tilt12", duration=0.1, seed=3)
    assert a == b


def test_an_unknown_envelope_is_refused():
    with pytest.raises(ValueError, match="unknown envelope"):
        synthesise(200.0, "whatever-i-typed", duration=0.05)


def test_measure_sweep_returns_one_row_per_signal():
    signals = sweep_signals(f0_hz=(110.0, 320.0), envelopes=("flat",), duration=0.1)
    rows, errors = measure_sweep({"invariant": constant}, signals)
    assert errors == {}
    assert [r.f0 for r in rows] == [110.0, 320.0]
    assert all(r.values == {"invariant": 7.0} for r in rows)


def test_the_documented_exit_codes_hold_through_the_command_line(tmp_path):
    """0 nothing found, 1 a finding, 2 cannot judge - the same contract the
    other commands publish."""
    import csv

    from spkproof.cli import main

    assert main(["sweep", str(tmp_path / "signals"), "--duration", "0.1",
                 "--f0", "110", "320"]) == 0
    manifest = list(csv.DictReader(open(tmp_path / "signals" / "sweep.csv")))
    assert len(manifest) == 6
    assert all((tmp_path / "signals" / r["file"]).exists() for r in manifest)

    def table(name: str, values: list[float]) -> str:
        path = tmp_path / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["f0", "envelope", "measured"])
            for row, v in zip(manifest, values, strict=True):
                w.writerow([row["f0"], row["envelope"], v])
        return str(path)

    steady = table("steady", [1.0, 1.01] * 3)
    assert main(["check-descriptor", steady]) == 0
    moving = table("moving", [1.0, 9.0] * 3)
    assert main(["check-descriptor", moving]) == 1
    assert main(["check-descriptor", str(tmp_path / "does-not-exist.csv")]) == 2


def test_the_signals_can_be_written_as_wav_for_an_external_extractor(tmp_path):
    """The measures this gate exists for live in opensmile and Praat, not in
    Python. If the signals cannot leave the process, the gate cannot reach
    them."""
    signal = sweep_signals(f0_hz=(200.0,), envelopes=("flat",), duration=0.1)[0]
    path = tmp_path / "probe.wav"
    write_wav(path, signal.samples, signal.sample_rate)
    with wave.open(str(path), "rb") as fh:
        assert fh.getnchannels() == 1
        assert fh.getsampwidth() == 2
        assert fh.getframerate() == 16000
        assert fh.getnframes() == len(signal.samples)
