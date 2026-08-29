"""spkproof command line.

Exit codes follow the same contract as trainproof:
    0  checks ran, nothing found
    1  checks ran, findings reported
    2  cannot judge - bad input, missing columns, unreadable file
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from . import __version__
from .descriptor import DEFAULT_TOLERANCE, SweepValue
from .design import Trial, check_design
from .f0 import DEFAULT_CEILING_ST, Utterance, check_f0

# Column names accepted for each field, in priority order. A study's table is
# whatever its author called it; requiring one exact spelling would mean every
# user rewrites their CSV before they can run a check.
ALIASES = {
    "f0": ["f0", "pitch", "f0_hz", "median_f0", "f0_median", "pitch_hz"],
    "condition": ["condition", "cond", "label", "class", "manipulation", "style"],
    "speaker": ["speaker", "spk", "speaker_id", "subject", "talker"],
    "group": ["corpus", "session", "group", "recording_session", "day", "block"],
}


def _resolve(header: list[str]) -> dict[str, str | None]:
    lower = {h.lower().strip(): h for h in header}
    out: dict[str, str | None] = {}
    for field, names in ALIASES.items():
        out[field] = next((lower[n] for n in names if n in lower), None)
    return out


def _load(path: Path, group_col: str | None = None) -> tuple[list[Utterance], dict[str, str | None]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("file has no header row")
        cols = _resolve(list(reader.fieldnames))
        if cols["f0"] is None:
            raise ValueError(
                f"no F0 column found. Looked for {ALIASES['f0']}; "
                f"file has {list(reader.fieldnames)}"
            )
        if cols["condition"] is None:
            raise ValueError(
                f"no condition column found. Looked for {ALIASES['condition']}; "
                f"file has {list(reader.fieldnames)}"
            )
        if group_col is not None:
            if group_col.lower() not in lower_map(reader.fieldnames):
                raise ValueError(
                    f"--group '{group_col}' is not a column in this file; "
                    f"it has {list(reader.fieldnames)}"
                )
            cols["group"] = lower_map(reader.fieldnames)[group_col.lower()]

        utts = []
        for row in reader:
            raw = (row.get(cols["f0"]) or "").strip()
            try:
                f0 = float(raw)
            except ValueError:
                f0 = float("nan")
            utts.append(Utterance(
                f0=f0,
                condition=(row.get(cols["condition"]) or "").strip() or "unlabelled",
                speaker=(row.get(cols["speaker"]) or "unknown").strip() if cols["speaker"] else "unknown",
                group=(row.get(cols["group"]) or "").strip() if cols["group"] else "",
            ))
    return utts, cols


def lower_map(header: Iterable[str]) -> dict[str, str]:
    return {h.lower().strip(): h for h in header}


def cmd_check_f0(args: argparse.Namespace) -> int:
    path = Path(args.table)
    if not path.exists():
        print(f"spkproof: no such file: {path}", file=sys.stderr)
        return 2
    try:
        utts, cols = _load(path, args.group)
    except (ValueError, UnicodeDecodeError, csv.Error) as e:
        print(f"spkproof: cannot judge {path.name}: {e}", file=sys.stderr)
        return 2

    findings, summary = check_f0(utts, ceiling_st=args.ceiling, reference=args.reference)

    if args.json:
        print(json.dumps({
            "file": str(path),
            "columns_used": cols,
            "summary": summary,
            "findings": [{"rule": f.rule, "severity": f.severity,
                          "message": f.message, "evidence": f.evidence} for f in findings],
        }, indent=2))
        return 1 if findings else 0

    print(f"spkproof check-f0  {path.name}")
    print(f"  columns: f0={cols['f0']}  condition={cols['condition']}  speaker={cols['speaker']}")
    print(f"  {summary['n_usable']} usable utterances"
          + (f", {summary['n_dropped']} dropped for missing or non-positive F0"
             if summary["n_dropped"] else ""))
    if "max_abs_semitones" in summary:
        print(f"  largest deviation from baseline: {summary['max_abs_semitones']:g} semitones "
              f"(ceiling {summary['ceiling_st']:g})")
    print()

    if not findings:
        print("  No contamination signature detected.")
        print("  This is not proof the F0 estimates are correct - only that these")
        print("  checks found nothing. Report that you ran them.")
        return 0

    for f in findings:
        print(f"  {f}")
        print()

    errors = sum(1 for f in findings if f.severity == "error")
    if errors:
        print(f"  {errors} error-level finding(s). Do not report a pooled pitch coefficient")
        print("  from this table without addressing them, and state which correction you")
        print("  applied - different corrections can give opposite answers.")
    return 1


TRIAL_ALIASES = {
    "condition": ["condition", "cond", "class", "manipulation", "style", "phonation"],
    "label": ["label", "trial_type", "target", "key", "is_genuine", "type"],
    "enroll_speaker": ["enroll_speaker", "enrolled", "enrol_speaker", "model_speaker",
                       "speaker", "spk", "claimed_speaker"],
    "test_speaker": ["test_speaker", "probe_speaker", "actual_speaker", "trial_speaker"],
    "duration": ["test_sec", "duration", "dur", "seconds", "length", "duration_sec",
                 "test_duration"],
    "test_id": ["test_path", "test_id", "probe", "probe_path", "wav", "file", "path"],
}


def _resolve_trials(header: list[str]) -> dict[str, str | None]:
    lower = {h.lower().strip(): h for h in header}
    return {f: next((lower[n] for n in names if n in lower), None)
            for f, names in TRIAL_ALIASES.items()}


def _load_trials(path: Path) -> tuple[list[Trial], dict[str, str | None]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("file has no header row")
        cols = _resolve_trials(list(reader.fieldnames))
        for need in ("condition", "label"):
            if cols[need] is None:
                raise ValueError(
                    f"no {need} column found. Looked for {TRIAL_ALIASES[need]}; "
                    f"file has {list(reader.fieldnames)}"
                )
        trials = []
        for row in reader:
            raw = (row.get(cols["duration"]) or "").strip() if cols["duration"] else ""
            try:
                dur = float(raw) if raw else None
            except ValueError:
                dur = None
            trials.append(Trial(
                condition=(row.get(cols["condition"]) or "").strip() or "unlabelled",
                label=(row.get(cols["label"]) or "").strip(),
                enroll_speaker=((row.get(cols["enroll_speaker"]) or "unknown").strip()
                                if cols["enroll_speaker"] else "unknown"),
                test_speaker=((row.get(cols["test_speaker"]) or "unknown").strip()
                              if cols["test_speaker"] else "unknown"),
                duration=dur,
                test_id=((row.get(cols["test_id"]) or "").strip()
                         if cols["test_id"] else ""),
            ))
    return trials, cols


def cmd_check_design(args: argparse.Namespace) -> int:
    path = Path(args.trials)
    if not path.exists():
        print(f"spkproof: no such file: {path}", file=sys.stderr)
        return 2
    try:
        trials, cols = _load_trials(path)
    except (ValueError, UnicodeDecodeError, csv.Error) as e:
        print(f"spkproof: cannot judge {path.name}: {e}", file=sys.stderr)
        return 2
    if not trials:
        print(f"spkproof: {path.name} has no rows", file=sys.stderr)
        return 2

    enrollment = None
    if args.enrollment:
        ep = Path(args.enrollment)
        if not ep.exists():
            print(f"spkproof: no such file: {ep}", file=sys.stderr)
            return 2
        try:
            enrollment = json.loads(ep.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"spkproof: cannot read {ep.name}: {e}", file=sys.stderr)
            return 2

    findings, summary = check_design(trials, enrollment)

    if args.json:
        print(json.dumps({
            "file": str(path), "columns_used": cols, "summary": summary,
            "findings": [{"rule": f.rule, "severity": f.severity,
                          "message": f.message, "evidence": f.evidence} for f in findings],
        }, indent=2))
        return 1 if findings else 0

    print(f"spkproof check-design  {path.name}")
    print(f"  {summary['n_trials']} trials, {summary['n_conditions']} conditions, "
          f"{summary['n_genuine']} genuine / {summary['n_impostor']} impostor")
    if "median_duration_sec" in summary:
        print(f"  median clip {summary['median_duration_sec']:g}s")
    print()

    for f in findings:
        print(f"  {f}")
        print()

    checks_ran = summary["checks_ran"]
    checks_skipped = summary["checks_skipped"]
    ran = ", ".join(str(c) for c in checks_ran) if isinstance(checks_ran, list) else ""
    print(f"  Ran: {ran or 'none'}.")
    if isinstance(checks_skipped, list):
        for s in checks_skipped:
            print(f"  Skipped: {s}")

    if not findings:
        print()
        print("  No design fault detected. This is not proof the comparison is")
        print("  sound - only that these checks found nothing. Report that you ran them.")
        return 0

    errors = sum(1 for f in findings if f.severity == "error")
    if errors:
        print()
        print(f"  {errors} error-level finding(s). A cross-condition result from this")
        print("  trial list cannot be attributed to the condition without addressing them.")
    return 1


def cmd_compare_encoders(args: argparse.Namespace) -> int:
    from .panel import (
        family_summary,
        paired_compare,
        rank_and_warn,
        resample_note,
        score_panel,
        worst_condition,
    )

    path = Path(args.scores)
    if not path.exists():
        print(f"spkproof: no such file: {path}", file=sys.stderr)
        return 2
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise ValueError("file has no header row")
            header = list(reader.fieldnames)
            rows = list(reader)
    except (ValueError, UnicodeDecodeError, csv.Error) as e:
        print(f"spkproof: cannot judge {path.name}: {e}", file=sys.stderr)
        return 2
    if not rows:
        print(f"spkproof: {path.name} has no rows", file=sys.stderr)
        return 2

    lower = {h.lower().strip(): h for h in header}
    for need, names in (("label", TRIAL_ALIASES["label"]),
                        ("condition", TRIAL_ALIASES["condition"])):
        if not any(n in lower for n in names):
            print(f"spkproof: no {need} column. Looked for {names}; "
                  f"file has {header}", file=sys.stderr)
            return 2
    lab = next(lower[n] for n in TRIAL_ALIASES["label"] if n in lower)
    con = next(lower[n] for n in TRIAL_ALIASES["condition"] if n in lower)
    spk = next((lower[n] for n in TRIAL_ALIASES["enroll_speaker"] if n in lower), None)
    tst = next((lower[n] for n in TRIAL_ALIASES["test_speaker"] if n in lower), None)

    if args.encoders:
        encoders = args.encoders
        missing = [e for e in encoders if e not in header]
        if missing:
            print(f"spkproof: no such column(s): {missing}. "
                  f"File has {header}", file=sys.stderr)
            return 2
    else:
        # any numeric column that is not metadata is a candidate score column
        meta = {lab, con, spk, tst} | set(
            h for h in header if h.lower() in
            {"phrase", "test_path", "test_id", "analysis", "test_sec", "duration",
             "sec_delta", "neutral_sec", "test_speaker", "path", "wav", "file"})
        encoders = []
        for h in header:
            if h in meta or h is None:
                continue
            vals = [r.get(h, "") for r in rows[:200] if r.get(h, "") != ""]
            if not vals:
                continue
            try:
                [float(v) for v in vals]
            except (TypeError, ValueError):
                continue
            encoders.append(h)
        if not encoders:
            print("spkproof: found no numeric score columns to compare",
                  file=sys.stderr)
            return 2

    norm = [{"condition": r.get(con, "unlabelled"),
             "label": r.get(lab, ""),
             "speaker": r.get(spk, "unknown") if spk else "unknown",
             "enroll_speaker": r.get(spk, "unknown") if spk else "unknown",
             "test_speaker": r.get(tst, "") if tst else "",
             **{e: r.get(e, "") for e in encoders}} for r in rows]

    try:
        results = score_panel(norm, encoders, bootstrap=args.bootstrap,
                              seed=args.seed, resample=args.resample,
                              alpha=args.alpha)
    except ValueError as e:
        print(f"spkproof: cannot judge {path.name}: {e}", file=sys.stderr)
        return 2
    if not results:
        print("spkproof: no condition had both genuine and impostor trials",
              file=sys.stderr)
        return 2
    winners, ties = rank_and_warn(results, alpha=args.alpha)
    comparisons, warnings = paired_compare(results, alpha=args.alpha)
    summary = family_summary(comparisons)
    caveat = resample_note(results)
    notes = ties + warnings + ([caveat] if caveat and caveat not in warnings else [])

    if args.json:
        print(json.dumps({
            "file": str(path), "encoders": encoders,
            "results": [{k: v for k, v in r.__dict__.items() if k != "draws"}
                        for r in results],
            "comparisons": [c.__dict__ for c in comparisons],
            "family": summary,
            "best_per_condition": winners, "warnings": notes,
        }, indent=2, default=str))
        return 1 if notes else 0

    print(f"spkproof compare-encoders  {path.name}")
    print(f"  {len(rows)} trials, {len(encoders)} encoder(s), "
          f"{len({r.condition for r in results})} condition(s)")
    print(f"  EER with {int((1 - args.alpha) * 100)}% CI, resampled over "
          f"{results[0].resample_unit}")
    print()
    width = max(len(e) for e in encoders) + 2
    conds = sorted({r.condition for r in results})
    print("  " + "condition".ljust(20) + "".join(e.ljust(width + 14) for e in encoders))
    for c in conds:
        line = "  " + c.ljust(20)
        for enc in encoders:
            r = next((x for x in results if x.encoder == enc and x.condition == c), None)
            line += (f"{r.eer:.3f} [{r.eer_lo:.3f},{r.eer_hi:.3f}]".ljust(width + 14)
                     if r else "—".ljust(width + 14))
        print(line)
    if comparisons:
        print()
        print(f"  paired differences inside the bootstrap draws, family "
              f"'{summary['family']}' ({summary['n_comparisons']} comparisons), "
              f"Holm-Bonferroni at family-wise {args.alpha:g}")
        print("  " + "condition".ljust(16) + "comparison".ljust(max(28, 2 * width + 4))
              + "dEER".rjust(8) + "  " + f"{int((1 - args.alpha) * 100)}% CI".ljust(20)
              + "p".rjust(8) + "Holm p".rjust(9) + "  verdict")
        for comp in comparisons:
            print("  " + comp.condition.ljust(16)
                  + f"{comp.encoder_a} - {comp.encoder_b}".ljust(max(28, 2 * width + 4))
                  + f"{comp.delta:+8.3f}" + "  "
                  + f"[{comp.lo:+.3f}, {comp.hi:+.3f}]".ljust(20)
                  + f"{comp.p_value:8.4f}{comp.p_holm:9.4f}"
                  + ("  survives" if comp.survives else "  does not survive"))
        print()
        print(f"  {summary['n_survive_holm']} of {summary['n_comparisons']} survive "
              f"correction. {summary['n_significant_uncorrected']} would be called "
              f"established without it.")

    print()
    for c in conds:
        print(f"  best on '{c}': {winners.get(c, '—')}")
    if ties:
        print()
        for n in ties:
            print(f"  [SPK-PANEL-TIE] WARNING: {n}")
    if warnings:
        print()
        for n in warnings:
            print(f"  [SPK-PANEL-METHOD] WARNING: {n}")
    elif caveat:
        print()
        print(f"  [SPK-PANEL-METHOD] WARNING: {caveat}")
    print()
    for enc in encoders:
        w = worst_condition(results, enc)
        if w:
            print(f"  {enc} fails your speakers most on '{w[0]}' (EER {w[1]:.3f})")
    print()
    print("  These are YOUR speakers under YOUR conditions. Nothing here ranks")
    print("  an encoder in general, and an EER describes a system, never a person.")
    return 1 if notes else 0


DESCRIPTOR_ALIASES = {
    "f0": ["f0", "pitch", "f0_hz", "sweep_f0", "hz"],
    "envelope": ["envelope", "env", "spectrum", "condition"],
}


def cmd_sweep(args: argparse.Namespace) -> int:
    from .descriptor import DEFAULT_ENVELOPES, DEFAULT_SWEEP_HZ, sweep_signals, write_wav

    out = Path(args.outdir)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"spkproof: cannot write to {out}: {e}", file=sys.stderr)
        return 2

    f0s = args.f0 or list(DEFAULT_SWEEP_HZ)
    signals = sweep_signals(f0_hz=f0s, envelopes=DEFAULT_ENVELOPES,
                            sample_rate=args.sample_rate, duration=args.duration,
                            seed=args.seed)
    manifest = out / "sweep.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "f0", "envelope", "sample_rate", "duration_sec"])
        for s in signals:
            name = f"{s.envelope}_{s.f0:g}Hz.wav"
            write_wav(out / name, s.samples, s.sample_rate)
            w.writerow([name, f"{s.f0:g}", s.envelope, s.sample_rate,
                        f"{len(s.samples) / s.sample_rate:g}"])

    print(f"spkproof sweep  {out}")
    print(f"  {len(signals)} signals: {len(f0s)} pitches x "
          f"{len(DEFAULT_ENVELOPES)} envelopes, {args.sample_rate} Hz")
    print(f"  manifest: {manifest.name}")
    print()
    print("  The spectral envelope is identical in every file of one envelope group;")
    print("  only F0 differs. Run your extractor over them, put its numbers in a CSV")
    print("  with the f0 and envelope columns from the manifest, and judge it with")
    print("  `spkproof check-descriptor`. A descriptor that moves across a group is")
    print("  measuring harmonic spacing, whatever it is called.")
    return 0


def _load_sweep(path: Path) -> tuple[list[SweepValue], list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("file has no header row")
        header = list(reader.fieldnames)
        lower = {h.lower().strip(): h for h in header}
        cols = {f: next((lower[n] for n in names if n in lower), None)
                for f, names in DESCRIPTOR_ALIASES.items()}
        if cols["f0"] is None:
            raise ValueError(
                f"no F0 column found. Looked for {DESCRIPTOR_ALIASES['f0']}; "
                f"file has {header}"
            )
        rows = list(reader)

    meta = {cols["f0"], cols["envelope"], "file", "path", "wav", "sample_rate",
            "duration_sec"}
    names = [h for h in header if h not in meta]
    out: list[SweepValue] = []
    for r in rows:
        try:
            f0 = float((r.get(cols["f0"]) or "").strip())
        except ValueError:
            continue
        values: dict[str, float] = {}
        for n in names:
            raw = (r.get(n) or "").strip()
            if not raw:
                continue
            try:
                values[n] = float(raw)
            except ValueError:
                continue
        env = ((r.get(cols["envelope"]) or "").strip() or "unlabelled"
               if cols["envelope"] else "unlabelled")
        out.append(SweepValue(f0=f0, envelope=env, values=values))
    return out, sorted({n for row in out for n in row.values})


def cmd_check_descriptor(args: argparse.Namespace) -> int:
    from .descriptor import check_sweep_table

    path = Path(args.table)
    if not path.exists():
        print(f"spkproof: no such file: {path}", file=sys.stderr)
        return 2
    try:
        rows, names = _load_sweep(path)
    except (ValueError, UnicodeDecodeError, csv.Error) as e:
        print(f"spkproof: cannot judge {path.name}: {e}", file=sys.stderr)
        return 2
    if not rows:
        print(f"spkproof: {path.name} has no usable rows", file=sys.stderr)
        return 2
    if not names:
        print(f"spkproof: {path.name} has no descriptor columns to judge",
              file=sys.stderr)
        return 2

    # No control column can be synthesised from a table somebody else measured,
    # so the gate's own sensitivity is unproven here. Said, not assumed.
    findings, summary = check_sweep_table(
        rows, tolerance=args.tolerance, control_name=args.control or None)

    if args.json:
        print(json.dumps({
            "file": str(path), "summary": summary,
            "findings": [{"rule": f.rule, "severity": f.severity,
                          "message": f.message, "evidence": f.evidence} for f in findings],
        }, indent=2, default=str))
        return 1 if findings else 0

    pitches = summary["f0_hz"] if isinstance(summary["f0_hz"], list) else []
    envelopes = summary["envelopes"] if isinstance(summary["envelopes"], list) else []
    print(f"spkproof check-descriptor  {path.name}")
    print(f"  {summary['n_signals']} signals, {len(pitches)} pitches"
          + (f" spanning {summary['sweep_semitones']} semitones"
             if "sweep_semitones" in summary else "")
          + f", {len(envelopes)} envelope(s)")
    print()
    budgets = summary.get("worst_budget")
    if isinstance(budgets, dict) and budgets:
        print("  " + "descriptor".ljust(28) + "artifact budget".rjust(16) + "   verdict")
        for name, budget in budgets.items():
            verdict = "ADMISSIBLE" if float(budget) <= args.tolerance else "REJECTED"
            if name == args.control:
                verdict += " (the control, which is what it must be)"
            print("  " + str(name).ljust(28) + f"{float(budget):16.3f}   {verdict}")
        print()
        print("  The budget is how far a descriptor moved when ONLY F0 moved. Read every")
        print(f"  effect you report against it; tolerance {args.tolerance:g} is a line "
              f"someone drew.")
    print()

    if not args.control:
        print("  No control descriptor in this table, so nothing here demonstrates that")
        print("  the sweep can reject anything. Include a measure you know is broken -")
        print("  a band mean over the raw spectrum - or gate in Python, where spkproof")
        print("  supplies one.")
        print()

    for f in findings:
        print(f"  {f}")
        print()

    if not findings:
        judged = summary.get("n_descriptors", 0)
        print(f"  None of the {judged} descriptor(s) gated here moved beyond the tolerance.")
        if args.control:
            print(f"  The control '{args.control}' was rejected, as it has to be: the sweep")
            print("  can tell the difference, so ADMISSIBLE means something in this run.")
        print("  That is not proof any of them measures what you want; it is only proof")
        print("  they do not follow pitch here.")
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="spkproof",
        description="Deterministic checks for speaker-verification and speaker-embedding studies.",
    )
    p.add_argument("--version", action="version", version=f"spkproof {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser(
        "check-f0",
        help="detect F0-tracking contamination in a per-utterance measurement table",
    )
    c.add_argument("table", help="CSV with one row per utterance")
    c.add_argument("--ceiling", type=float, default=DEFAULT_CEILING_ST,
                   metavar="ST",
                   help=f"semitone deviation beyond which a value is impossible "
                        f"(default {DEFAULT_CEILING_ST:g})")
    c.add_argument("--reference", default="clean", metavar="LABEL",
                   help="condition label to use as each speaker's baseline (default 'clean')")
    c.add_argument("--group", default=None, metavar="COLUMN",
                   help="column identifying the enrollment group (session, corpus, recording day). "
                        "A speaker recorded across two sessions gets one baseline per session "
                        "instead of one pooled across both. Auto-detected from a 'corpus' or "
                        "'session' column when present.")
    c.add_argument("--json", action="store_true", help="machine-readable output")
    c.set_defaults(func=cmd_check_f0)

    d = sub.add_parser(
        "check-design",
        help="detect faults in a speaker-verification trial list before any encoder runs",
    )
    d.add_argument("trials", help="CSV with one row per trial")
    d.add_argument("--enrollment", default=None, metavar="JSON",
                   help="JSON mapping speaker -> list of enrollment clip ids, so "
                        "leakage between enrollment and test can be checked")
    d.add_argument("--json", action="store_true", help="machine-readable output")
    d.set_defaults(func=cmd_check_design)

    e = sub.add_parser(
        "compare-encoders",
        help="which speaker encoder can you trust for YOUR speakers and conditions",
    )
    e.add_argument("scores", help="CSV: one row per trial, one numeric column per encoder")
    e.add_argument("--encoders", nargs="*", default=None, metavar="COL",
                   help="score columns to compare (default: every numeric non-metadata column)")
    e.add_argument("--bootstrap", type=int, default=2000, metavar="N",
                   help="speaker resamples for the CI and the paired test (default 2000; "
                        "0 to skip). A bootstrap p cannot fall below 2/(N+1), so a large "
                        "family of comparisons needs a large N to be correctable at all")
    e.add_argument("--resample", choices=("auto", "dyadic", "enroll"), default="auto",
                   help="'auto' (default) resamples speakers on both sides when the trial "
                        "table names the test speaker, and enrollment identities only when "
                        "it does not; 'dyadic' insists and fails if it cannot; 'enroll' is "
                        "the narrower old scheme, for reproducing an older number")
    e.add_argument("--alpha", type=float, default=0.05, metavar="A",
                   help="interval level and family-wise error rate (default 0.05)")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--json", action="store_true", help="machine-readable output")
    e.set_defaults(func=cmd_compare_encoders)

    g = sub.add_parser(
        "check-descriptor",
        help="is this spectral descriptor an artifact of pitch? sweep F0, hold the envelope",
    )
    g.add_argument("table", help="CSV: one row per swept signal, one column per descriptor, "
                                 "plus f0 and envelope (as written by `spkproof sweep`)")
    g.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, metavar="UNITS",
                   help=f"movement across the sweep beyond which a descriptor is an "
                        f"artifact (default {DEFAULT_TOLERANCE:g}, in dB for the measures "
                        f"this was written for)")
    g.add_argument("--control", default=None, metavar="COL",
                   help="name of a column holding a descriptor you KNOW is broken. The gate "
                        "must reject it; if it does not, the gate is wrong and no other "
                        "verdict in the run stands")
    g.add_argument("--json", action="store_true", help="machine-readable output")
    g.set_defaults(func=cmd_check_descriptor)

    s = sub.add_parser(
        "sweep",
        help="write the gate's test signals as WAV, for an extractor that reads files",
    )
    s.add_argument("outdir", help="directory to write the signals and their manifest into")
    s.add_argument("--f0", type=float, nargs="*", default=None, metavar="HZ",
                   help="pitches to sweep (default 110-320 Hz in eight steps). Use the "
                        "range YOUR corpus covers")
    s.add_argument("--duration", type=float, default=1.0, metavar="SEC")
    s.add_argument("--sample-rate", type=int, default=16000, metavar="HZ")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sweep)

    args = p.parse_args(argv)
    exit_code: int = args.func(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
