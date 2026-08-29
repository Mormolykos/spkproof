# Changelog

Every release of this library exists because a speaker-verification study got
something wrong — usually one of mine — and the fix was worth encoding so nobody
has to rediscover it. Each entry below says what changed and **why it was
needed**, because a list of features does not tell you whether you need it.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is [semantic](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] — 2026-08-29

**Theme: the library was weaker than the method it documented.**

Version 0.2.0 shipped a panel that compared encoders by asking whether their
confidence intervals overlapped. The study behind the library had already
established that this is the wrong test, and the library had not caught up. All
four changes close a gap between what the tool did and what the evidence said it
should do.

### Added

- **`check_descriptor`** (`descriptor.py`) — an admissibility gate for acoustic
  descriptors. Synthesises a voice with the spectral envelope held **fixed**,
  sweeps F0 across 18.5 semitones on three envelopes, and reports an **artifact
  budget in the descriptor's own units** rather than a pass mark.

  This exists because a band-mean in dB moves when harmonic spacing changes,
  with no change to the voice. A measure with a 12 dB artifact budget was used
  to report an 11.7 dB effect, and the effect was the instrument. The gate takes
  about 1.2 seconds and would have prevented that.

  It always runs a control that is **broken by construction**. If the control
  survives, every `ADMISSIBLE` verdict in that run is declared void — a gate
  that cannot fail is not a gate. `spkproof sweep` writes the signals as WAV so
  openSMILE or Praat descriptors can be gated the same way.

- **`declare_family()` and Holm–Bonferroni** (`stats.py`), reported as both
  counts: *"5 of 9 survive correction. 8 would be called established without
  it."* Warns when a "confirmatory" family is really the subset you liked.

- **A bootstrap-adequacy warning.** A bootstrap p-value cannot fall below
  `2/(B+1)`, so a 9-comparison family at `B=200` cannot produce a single
  survivor **at any effect size** — and that reads exactly like a clean
  negative. The library now refuses to stay quiet about it.

- **`paired_compare()`** — takes the difference between two encoders *inside*
  every bootstrap draw, so the speakers and trials they share cancel instead of
  inflating both intervals.

### Changed

- **`rank_and_warn` decides ties with the paired test**, not interval overlap.
  Marginal overlap is conservative: it calls real differences ties.

- **Vertex bootstrap is the default resampling unit** when the trial table has a
  `test_speaker` column. A trial list is a graph on speakers — genuine trials are
  self-loops, impostor trials are edges — so resampling *enrollment* identities
  alone leaves a dropped speaker in the pool as an impostor. Drawing speakers
  with replacement and weighting the impostor edge `m_e · m_t` fixes it.
  Intervals computed the old way ran up to **58% too narrow**.

  Without a `test_speaker` column the library falls back, **names the weaker
  unit in every result**, and exits 1. `--resample dyadic` insists and exits 2
  if it cannot comply.

- The weighted estimator is now O(n) per draw instead of O(n²).

### Fixed

- **The EER plateau tie-break.** When two thresholds tie on `|FRR − FAR|`,
  `argmin` returns the *first* minimiser — the lower threshold, not the better
  operating point. An EER is the balanced point a system would actually run at,
  so the worse of two equally balanced settings is not its error rate.

  On genuine `[0.4, 0.6]` against impostor `[0.5, 0.5]` the two rules return
  **0.25 and 0.75 for the same data.** spkproof returns 0.25, and a test pins it,
  because a comment does not stop anyone.

  Found independently by two analyses at two scales — 2.2e-4 on 3,616 trials and
  6.3e-3 under weighted draws, a factor of 28. The size is a property of the
  data, not a bound.

### Verified

119 tests (was 78) · `ruff` clean · `mypy --strict` clean. Re-run against the
study's own frozen trial list, the rewrite reproduces its published EERs exactly
(0.003 / 0.031 / 0.000 neutral; 0.112 / 0.228 / 0.047 whisper; 0.165 / 0.256 /
0.074 shouting). **The point estimates did not move. Only the intervals and the
tests did** — which is what a statistics fix should look like.

---

## [0.2.0] — 2026-08-28 *(never released to PyPI)*

Developed and tested but never uploaded; its code ships inside 0.3.0. Recorded
here so the version history has no silent gap.

### Added

- **`check_design`** — four deterministic rules for trial-list faults, each one
  written because it caught a real error in the study behind the library:
  duration/condition confounding, impostor-pool drift, enrollment leakage, and
  genuine-to-impostor ratio drift. Fires on a bad design, silent on a good one.

- **`score_panel` / `rank_and_warn` / `worst_condition`** — compare encoders on
  *your* recordings under *your* conditions, with speaker-stratified bootstrap
  intervals and a **tie warning that refuses to name a winner** when the data
  cannot support one. It fired against its own author first.

Both are dependency-free: they read a table you already have.

---

## [0.1.0] — 2026-07 *(current PyPI release)*

### Added

- **`check_f0`** — detects measurement contamination. Pitch trackers fail on
  rough phonation, and they fail **in one direction**, so a study that does not
  check for it reports the failure as a finding.

---

## Provenance

The checks are not invented. Each traces to a documented error in published work:

- Gkilis (2026), *Intra-speaker vocal variation and speaker-embedding
  displacement*, [doi:10.5281/zenodo.21921957](https://doi.org/10.5281/zenodo.21921957)
  — the F0 checks.
- Gkilis (2026), *Speaker encoders disagree about who you are when you shout*,
  [doi:10.5281/zenodo.22158030](https://doi.org/10.5281/zenodo.22158030)
  — the design checks, the panel, and every statistical fix in 0.3.0. That
  report carries eight retractions of its own claims; this library is what those
  retractions were turned into.

[0.3.0]: https://github.com/Mormolykos/spkproof/releases/tag/v0.3.0
[0.2.0]: https://github.com/Mormolykos/spkproof
[0.1.0]: https://github.com/Mormolykos/spkproof/releases/tag/v0.1.0
