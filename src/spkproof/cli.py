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
from pathlib import Path

from . import __version__
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


def lower_map(header) -> dict[str, str]:
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

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
