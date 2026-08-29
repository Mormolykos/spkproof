"""The F0 gate: a descriptor may not move when only F0 moves.

Run this BEFORE a spectral descriptor is allowed into a study, not after.

WHY IT EXISTS
-------------
A band statistic over a spectrum - the mean level between 50 and 1000 Hz, the
tallest peak below 2 kHz - is computed over frequencies, but a voiced spectrum
only has energy AT the harmonics. Raise the pitch and the harmonics move, the
gaps between them widen, and the band statistic changes even though the vocal
tract did not. The number moves because the pitch moved, and it will be
reported as an effect of whatever the pitch happened to be correlated with.

In the study this check comes from, an experiment claimed that shouting
flattens the spectrum by -11.7 dB, 8 of 8 speakers, on `alpha_ratio`. This gate
put the same measure through a pitch sweep with the envelope held fixed and
found it can move 12.09 dB from pitch alone. The effect was inside its own
instrument's artifact. On a descriptor that passes this gate the real effect
was +2.4 dB on 5 of 8 speakers with an interval spanning zero - a different
sign, a different size, and not a finding. `hammarberg` measured 17.33 dB of
artifact budget, `cpp` 9.35 dB. The gate also rejected the first two
replacements proposed for them, which is the point: it is not a formality that
descriptors pass on their way into a paper.

HOW IT WORKS
------------
Synthesise a voice whose spectral ENVELOPE is fixed by construction, sweep F0
across the range a corpus actually contains, and measure the descriptor at each
pitch. Whatever it moves by is the artifact, because nothing else changed.

Three envelopes, because passing on one arbitrary envelope is not passing:
flat, a -12 dB/octave source tilt, and that tilt under three formants.
Rejection on ANY envelope is rejection.

WHAT THE OUTPUT IS
------------------
Not a pass mark: an ARTIFACT BUDGET, in the descriptor's own units. It is how
far that number can move with no change in the voice. An effect claimed on a
descriptor has to be read against its budget, and an effect that is not several
times larger than the budget is not an effect. `tolerance` is where this check
turns the budget into a verdict, and it is a threshold someone chose - the
budget underneath it is the measurement.

WHAT IT DOES NOT DO
-------------------
Passing means a descriptor is not an artifact of harmonic spacing. It does NOT
mean the descriptor tracks what you want it to track, that what you want to
track is separable from pitch in YOUR corpus, or that moving it changes
anything a listener hears. Those are separate questions and this check answers
none of them.

Two limits worth stating plainly. The synthetic formants here are narrower than
a real speaker's averaged over thousands of phrases, so the budget for
envelope-following descriptors is probably pessimistic - probably, which is not
measured. And overall LEVEL cannot be held fixed while F0 sweeps: the signal is
the envelope sampled at the harmonics, so its power depends on where they land.
Signals are normalised to equal RMS, and a descriptor that is not level-
invariant will fail this gate for a reason the gate cannot separate from the
artifact. Normalise for level inside your descriptor before gating it.

Source: Gkilis (2026), "Speaker encoders disagree about who you are when you
shout", supplementary gate. Ported here because it is the one reusable thing
that experiment produced.
"""
from __future__ import annotations

import math
import random
import struct
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

# A descriptor takes samples in [-1, 1] and a sample rate, and returns one
# number. That is the whole contract - anything that can be called this way can
# be gated, including a wrapper around an external extractor.
Descriptor = Callable[[Sequence[float], int], float]

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_DURATION_SEC = 1.0

# 110-320 Hz, the range one 8-speaker corpus of neutral through shouted speech
# actually covered. 15.4 semitones. Sweep YOUR range if you know it: a budget
# measured over a wider sweep than your data has is pessimistic, and one
# measured over a narrower sweep is worthless.
DEFAULT_SWEEP_HZ = (110.0, 130.0, 150.0, 175.0, 200.0, 235.0, 275.0, 320.0)

DEFAULT_ENVELOPES = ("flat", "tilt12", "formant")

# dB, for the descriptors this was written for. It is a threshold someone chose
# and the budget is the real output; a descriptor in other units needs its own.
DEFAULT_TOLERANCE = 1.5

# Centre frequency and bandwidth. Narrow on purpose: a narrow resonance is the
# hard case for anything that samples an envelope at moving harmonics.
FORMANTS = ((700.0, 80.0), (1220.0, 90.0), (2600.0, 120.0))

# The envelope is defined against a FIXED frequency, not against F0. Written
# against F0 - the shape the source implementation had - the curve keeps its
# shape but its level scales with pitch, which is a second thing changing in a
# sweep that exists to change one.
ENVELOPE_REFERENCE_HZ = 200.0

TARGET_RMS = 0.05

CONTROL_NAME = "f0_sensitive_control"


@dataclass
class Finding:
    rule: str
    severity: str          # "error" | "warning" | "info"
    message: str
    evidence: dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.rule}] {self.severity.upper()}: {self.message}"


@dataclass
class SweepSignal:
    f0: float
    envelope: str
    sample_rate: int
    samples: list[float] = field(default_factory=list, repr=False)


@dataclass
class SweepValue:
    """One synthetic signal's descriptor values. This is also the row shape of
    the CSV the CLI reads, so an extractor that cannot be called from Python
    can still be gated: write the signals, run your own tool over them, bring
    the table back."""
    f0: float
    envelope: str
    values: dict[str, float] = field(default_factory=dict)


def _amplitude(freq: float, envelope: str) -> float:
    if envelope == "flat":
        return 1.0
    tilt = (ENVELOPE_REFERENCE_HZ / freq) ** 2      # -12 dB per octave
    if envelope == "tilt12":
        return tilt
    if envelope == "formant":
        gain = sum(1.0 / (1.0 + ((freq - fc) / bw) ** 2) for fc, bw in FORMANTS)
        return tilt * (0.05 + gain)
    raise ValueError(f"unknown envelope {envelope!r}; expected one of {DEFAULT_ENVELOPES}")


def synthesise(
    f0: float,
    envelope: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration: float = DEFAULT_DURATION_SEC,
    seed: int = 0,
) -> list[float]:
    """One voiced signal at `f0` under a fixed spectral envelope.

    The vibrato and the jitter are there so the signal is not perfectly
    periodic - a perfectly periodic source makes periodicity measures
    degenerate rather than invariant, and they would pass the gate for the
    wrong reason. Both are driven from `seed`, so the same modulation is
    applied at every pitch and does not become a second thing that varies."""
    rng = random.Random(seed)
    n = int(duration * sample_rate)
    if n < 2:
        raise ValueError("duration is too short to synthesise anything")

    step = 2 * math.pi / sample_rate
    phase: list[float] = []
    running = 0.0
    for i in range(n):
        instant = f0 * (1 + 0.004 * math.sin(2 * math.pi * 4.3 * i / sample_rate)
                        + 0.002 * rng.gauss(0.0, 1.0))
        running += instant
        phase.append(step * running)

    x = [0.0] * n
    sin = math.sin
    k = 1
    while k * f0 < sample_rate / 2 - 100:
        a = _amplitude(k * f0, envelope)
        x = [xi + a * sin(k * p) for xi, p in zip(x, phase, strict=True)]
        k += 1

    floor = rng.gauss
    x = [xi + 0.0005 * floor(0.0, 1.0) for xi in x]
    rms = math.sqrt(sum(v * v for v in x) / n)
    if rms <= 0:
        raise ValueError(f"synthesis produced silence at f0={f0}")
    scale = TARGET_RMS / rms
    return [v * scale for v in x]


def sweep_signals(
    f0_hz: Sequence[float] = DEFAULT_SWEEP_HZ,
    envelopes: Sequence[str] = DEFAULT_ENVELOPES,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration: float = DEFAULT_DURATION_SEC,
    seed: int = 0,
) -> list[SweepSignal]:
    """The whole test set: every envelope at every pitch."""
    return [
        SweepSignal(f0=f0, envelope=env, sample_rate=sample_rate,
                    samples=synthesise(f0, env, sample_rate, duration, seed))
        for env in envelopes
        for f0 in f0_hz
    ]


def write_wav(path: str | Path, samples: Sequence[float], sample_rate: int) -> None:
    """16-bit mono PCM, so an extractor that only reads files can be gated."""
    clipped = [max(-1.0, min(1.0, v)) for v in samples]
    ints = [int(round(v * 32767)) for v in clipped]
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(struct.pack(f"<{len(ints)}h", *ints))


# --------------------------------------------------------------------------
# the positive control
# --------------------------------------------------------------------------

# Probe frequencies for the control, and a short frame, because this exists to
# be broken rather than to be good. Everything about it is cheap on purpose.
_CONTROL_PROBES = tuple(float(f) for f in range(100, 5000, 100))
_CONTROL_FRAME = 512
_CONTROL_FRAMES = 3


def _goertzel_magnitude(frame: Sequence[float], sample_rate: int, freq: float) -> float:
    """Magnitude at one frequency. A single-bin DFT, no library."""
    omega = 2 * math.pi * freq / sample_rate
    coeff = 2 * math.cos(omega)
    s1 = s2 = 0.0
    for x in frame:
        s0 = x + coeff * s1 - s2
        s2, s1 = s1, s0
    return math.hypot(s1 - s2 * math.cos(omega), s2 * math.sin(omega))


def f0_sensitive_control(samples: Sequence[float], sample_rate: int) -> float:
    """A descriptor that is KNOWN to be an artifact, shipped as the control.

    The mean level over a fixed set of probe frequencies below 1 kHz minus the
    mean over a fixed set above it. That is the shape of the measures that
    died: a statistic taken at frequencies that stay put, over a spectrum whose
    energy moves. It must be REJECTED by this gate. A gate that admits it is
    not measuring anything, and every other verdict it produced that run is
    void - which is why `check_descriptor` runs it whether you asked for it or
    not."""
    n = len(samples)
    if n < _CONTROL_FRAME:
        return float("nan")
    window = [0.5 - 0.5 * math.cos(2 * math.pi * i / (_CONTROL_FRAME - 1))
              for i in range(_CONTROL_FRAME)]
    starts = [int((n - _CONTROL_FRAME) * i / max(_CONTROL_FRAMES - 1, 1))
              for i in range(_CONTROL_FRAMES)]

    ratios: list[float] = []
    for start in starts:
        frame = [samples[start + i] * window[i] for i in range(_CONTROL_FRAME)]
        low: list[float] = []
        high: list[float] = []
        for probe in _CONTROL_PROBES:
            db = 20 * math.log10(max(_goertzel_magnitude(frame, sample_rate, probe), 1e-12))
            (low if probe < 1000 else high).append(db)
        if low and high:
            ratios.append(sum(low) / len(low) - sum(high) / len(high))
    return sum(ratios) / len(ratios) if ratios else float("nan")


# --------------------------------------------------------------------------
# measuring and judging
# --------------------------------------------------------------------------


def measure_sweep(
    descriptors: Mapping[str, Descriptor],
    signals: Sequence[SweepSignal],
) -> tuple[list[SweepValue], dict[str, str]]:
    """Run every descriptor over every signal.

    A descriptor that raises does not stop the sweep: it records nan and the
    exception text, so the other descriptors still get judged and the failure
    is reported as a failure rather than as an admissible verdict."""
    rows: list[SweepValue] = []
    errors: dict[str, str] = {}
    for signal in signals:
        values: dict[str, float] = {}
        for name, fn in descriptors.items():
            try:
                values[name] = float(fn(signal.samples, signal.sample_rate))
            except Exception as exc:                      # noqa: BLE001 - reported, not swallowed
                values[name] = float("nan")
                errors.setdefault(name, f"{type(exc).__name__}: {exc}")
        rows.append(SweepValue(f0=signal.f0, envelope=signal.envelope, values=values))
    return rows, errors


def _budgets(rows: Sequence[SweepValue], name: str) -> dict[str, float]:
    """Peak-to-peak swing across the pitch sweep, per envelope."""
    per_env: dict[str, list[float]] = {}
    for row in rows:
        v = row.values.get(name)
        if v is None or v != v or math.isinf(v):
            continue
        per_env.setdefault(row.envelope, []).append(v)
    return {env: max(vals) - min(vals) for env, vals in per_env.items() if len(vals) >= 2}


def check_sweep_table(
    rows: Sequence[SweepValue],
    tolerance: float = DEFAULT_TOLERANCE,
    control_name: str | None = CONTROL_NAME,
    errors: Mapping[str, str] | None = None,
) -> tuple[list[Finding], dict[str, object]]:
    """Judge a table of descriptor values measured over the sweep.

    Returns (findings, summary). The budgets live in the summary and not in the
    findings: a budget is the measurement this check produces, and a clean run
    has to be able to report nothing found."""
    errors = errors or {}
    names = sorted({n for row in rows for n in row.values})
    f0s = sorted({row.f0 for row in rows})
    envelopes = sorted({row.envelope for row in rows})

    summary: dict[str, object] = {
        "n_signals": len(rows),
        "f0_hz": f0s,
        "envelopes": envelopes,
        "tolerance": tolerance,
        "n_descriptors": len([n for n in names if n != control_name]),
    }
    if len(f0s) >= 2 and f0s[0] > 0:
        summary["sweep_semitones"] = round(12 * math.log2(f0s[-1] / f0s[0]), 2)

    findings: list[Finding] = []
    if len(f0s) < 2:
        findings.append(Finding(
            "SPK-DESC-INSUFFICIENT", "warning",
            f"the table holds {len(f0s)} distinct F0 value(s); a descriptor cannot be "
            f"shown to move when only F0 moves unless F0 moves. Nothing was judged.",
            {"n_f0": len(f0s)},
        ))
        return findings, summary

    budgets: dict[str, dict[str, float]] = {}
    worst: dict[str, float] = {}
    admissible: list[str] = []
    rejected: list[str] = []

    for name in names:
        per_env = _budgets(rows, name)
        if not per_env:
            findings.append(Finding(
                "SPK-DESC-ERROR", "warning",
                f"'{name}' produced no usable pair of values across the sweep"
                + (f": {errors[name]}" if name in errors else
                   " (blank, non-numeric or non-finite). It was not judged."),
                {"descriptor": name, "error": errors.get(name, "")},
            ))
            continue
        budgets[name] = {env: round(v, 4) for env, v in sorted(per_env.items())}
        budget = max(per_env.values())
        worst[name] = budget
        if name in errors:
            findings.append(Finding(
                "SPK-DESC-ERROR", "warning",
                f"'{name}' raised on at least one signal ({errors[name]}); it was judged "
                f"on the {sum(1 for r in rows if r.values.get(name) == r.values.get(name))} "
                f"of {len(rows)} signals it survived, so its budget is a lower bound.",
                {"descriptor": name, "error": errors[name]},
            ))
        if budget <= tolerance:
            admissible.append(name)
        else:
            rejected.append(name)

        values = [row.values.get(name) for row in rows]
        if len(set(v for v in values if v == v)) == 1:
            findings.append(Finding(
                "SPK-DESC-CONSTANT", "warning",
                f"'{name}' returned an identical value on all {len(rows)} signals. It "
                f"cannot fail this gate, and neither can a function that ignores its "
                f"input - check it is reading the audio before reading its verdict.",
                {"descriptor": name, "value": values[0]},
            ))

    summary["budgets"] = budgets
    summary["worst_budget"] = {n: round(v, 4) for n, v in sorted(worst.items())}
    summary["admissible"] = [n for n in admissible if n != control_name]
    summary["rejected"] = [n for n in rejected if n != control_name]

    for name in rejected:
        if name == control_name:
            continue
        env = max(budgets[name], key=lambda e: budgets[name][e])
        findings.append(Finding(
            "SPK-DESC-ARTIFACT", "error",
            f"'{name}' moves {worst[name]:.2f} across a "
            f"{summary.get('sweep_semitones', '?')}-semitone pitch sweep with the "
            f"spectral envelope held fixed (tolerance {tolerance:g}, worst on the "
            f"'{env}' envelope). Nothing about the voice changed, so that swing is the "
            f"instrument. Any effect you report on this descriptor has to be several "
            f"times {worst[name]:.2f} before it is an effect, and if the conditions you "
            f"compare differ in pitch you cannot separate the two at all.",
            {"descriptor": name, "budget": round(worst[name], 4),
             "tolerance": tolerance, "worst_envelope": env,
             "by_envelope": budgets[name]},
        ))

    if control_name is not None:
        control_budget = worst.get(control_name)
        summary["control"] = {
            "name": control_name,
            "budget": None if control_budget is None else round(control_budget, 4),
            "rejected": control_name in rejected,
        }
        if control_budget is not None and control_name not in rejected:
            findings.append(Finding(
                "SPK-DESC-BLIND", "error",
                f"the built-in broken control '{control_name}' was NOT rejected: it moved "
                f"only {control_budget:.2f} across the sweep, inside the {tolerance:g} "
                f"tolerance. That measure is an artifact by construction, so the gate is "
                f"wrong, not the descriptor. Every ADMISSIBLE verdict in this run is void "
                f"until the sweep, the tolerance or the sample rate is fixed.",
                {"control": control_name, "budget": round(control_budget, 4),
                 "tolerance": tolerance},
            ))

    summary["n_findings"] = len(findings)
    return findings, summary


def check_descriptor(
    descriptors: Mapping[str, Descriptor],
    tolerance: float = DEFAULT_TOLERANCE,
    f0_hz: Sequence[float] = DEFAULT_SWEEP_HZ,
    envelopes: Sequence[str] = DEFAULT_ENVELOPES,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration: float = DEFAULT_DURATION_SEC,
    seed: int = 0,
    control: bool = True,
) -> tuple[list[Finding], dict[str, object]]:
    """Sweep F0 with the envelope held fixed; reject any descriptor that moves.

        >>> findings, summary = check_descriptor({"my_tilt": my_tilt})
        >>> summary["worst_budget"]["my_tilt"]      # dB of artifact, not a verdict

    `control` adds a descriptor that is broken by construction and must be
    rejected. Turning it off removes the only evidence in the run that the gate
    can reject anything, so it defaults on and the run says so when it fails."""
    if control and CONTROL_NAME in descriptors:
        raise ValueError(
            f"{CONTROL_NAME!r} is the name of the built-in control; rename yours "
            f"or pass control=False"
        )
    to_run: dict[str, Descriptor] = dict(descriptors)
    if control:
        to_run[CONTROL_NAME] = f0_sensitive_control

    signals = sweep_signals(f0_hz, envelopes, sample_rate, duration, seed)
    rows, errors = measure_sweep(to_run, signals)
    return check_sweep_table(
        rows,
        tolerance=tolerance,
        control_name=CONTROL_NAME if control else None,
        errors=errors,
    )
