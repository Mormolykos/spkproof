# spkproof

Deterministic checks for speaker-verification and speaker-embedding studies.

```bash
pip install spkproof

spkproof check-f0           my_utterances.csv   # is this MEASUREMENT contaminated?
spkproof check-descriptor   my_sweep.csv        # is this DESCRIPTOR an artifact of pitch?
spkproof check-design       my_trials.csv       # is this COMPARISON already decided?
spkproof compare-encoders   my_scores.csv       # which encoder can I TRUST for my speakers?
```

No dependencies. No audio required. No model refit. Every check reads a table
you already have.

---

## What this is for

**It measures false rejection, not attacks.** spkproof has nothing to say about
spoofing, replay or deepfakes. It answers a different question:

> Does your speaker-verification system stop recognising *legitimate* users when
> their own voice changes — because they are hoarse, tired, ill, or simply
> speaking a little differently today?

That question has a documented population: a study of over 3,800 subjects found
adults with dysphonia at elevated re-identification risk (*Scientific Reports*
13, 2023, doi:10.1038/s41598-023-47711-7).

**What spkproof reports are potential measurement problems in your table — not
verdicts about any speaker.** A finding says an F0 estimate cannot be trusted as
an input to your statistics. It says nothing about whether a person is who they
claim to be, and nothing about whether their voice is normal, healthy or
authentic. Do not use it to make a decision about a human being.

## `check-design` — is the comparison already decided by its own structure?

`check-f0` catches a contaminated *measurement*. This catches a contaminated
*comparison*: a trial list whose shape guarantees the answer before any encoder
runs.

Every rule here exists because it caught a real error in a real study design,
and each one had already been written down as a result before it was found.

| rule | what it catches |
|---|---|
| `SPK-DUR-CONFOUND` | clip length tracks condition, so the effect is duration |
| `SPK-POOL-DRIFT` | the impostor set changes between conditions |
| `SPK-ENROLL-LEAK` | an enrollment clip reappears as a test clip |
| `SPK-RATIO-DRIFT` | the genuine:impostor ratio varies by condition |

**Why duration is not a detail.** Published speaker-verification error rates run
0.61% on full-length audio, 0.98% at 3 s and **1.48% at 2 s** (ERes2NetV2,
arXiv 2406.02167). In the corpus behind this tool, neutral clips averaged
1.4–1.9 s and shouted clips 2.5–3.6 s — and for the *same sentence*, shouting
ran **+0.80 s** longer. A condition contrast straddling that range measures
duration and reports it as phonation.

```
$ spkproof check-design trials.csv --enrollment enrollment.json

  [SPK-DUR-CONFOUND] ERROR: clip duration tracks condition: median 3.52s in
  'shouting' against 1.41s in 'neutral' (2.50x). Any difference between these
  conditions is equally explained by length.

  [SPK-POOL-DRIFT] ERROR: the impostor set is not the same in every condition:
  18 speakers appear somewhere, only 10 appear everywhere.
```

Run it on a sound design and it stays quiet. A checker that always fires is not
a checker.

## `compare-encoders` — which encoder can you trust for *your* speakers?

Leaderboards are computed on modal read speech. If your speakers get hoarse,
shout, whisper or get emotional, your system is operating outside the regime it
was ranked in — and the ranking can invert.

Measured on a parallel corpus of 8 speakers × 6 phonatory states × identical
text, the spread between encoders under shouting was more than **threefold**:
0.257 EER for one widely deployed encoder against 0.074 for another, on
identical trials. A system built on the wrong one silently rejects its own users
whenever they raise their voice.

```
$ spkproof compare-encoders scores.csv

  2300 trials, 3 encoder(s), 3 condition(s)
  EER with 95% CI, resampled over speaker (vertex bootstrap, both sides)

  condition       ecapa                 campplus              redimnet_b6
  NeutralHeldout  0.003 [0.000,0.006]   0.031 [0.002,0.060]   0.000 [0.000,0.000]
  Whisper         0.112 [0.060,0.178]   0.228 [0.144,0.303]   0.047 [0.011,0.095]
  Shouting        0.165 [0.090,0.241]   0.256 [0.186,0.317]   0.074 [0.016,0.106]

  paired differences inside the bootstrap draws, family 'all pairwise'
  (9 comparisons), Holm-Bonferroni at family-wise 0.05
  condition       comparison                   dEER  95% CI                p   Holm p  verdict
  NeutralHeldout  campplus - ecapa           +0.029  [+0.001, +0.060]  0.0490   0.1319  does not survive
  Shouting        campplus - ecapa           +0.091  [+0.011, +0.151]  0.0310   0.1239  does not survive
  Shouting        campplus - redimnet_b6     +0.182  [+0.136, +0.253]  0.0010   0.0090  survives
  Whisper         campplus - redimnet_b6     +0.181  [+0.112, +0.232]  0.0010   0.0090  survives
  ...

  5 of 9 survive correction. 8 would be called established without it.

  [SPK-PANEL-TIE] WARNING: NeutralHeldout: redimnet_b6 has the lowest EER (0.000)
  but campplus, ecapa cannot be separated from it (paired bootstrap on the
  difference, Holm over the 2 comparisons against the leader). On this data they
  are not distinguishable; do not report a winner.

  campplus fails your speakers most on 'Shouting' (EER 0.256)
```

**The refusal is the feature.** When a "winner" cannot be distinguished from a
rival, the tool says so instead of naming one. That warning fired on its own
author's results and forced a claim to be narrowed before publication.

Three things it does that the obvious version of this does not, each of them
found by an integrity pass over the study's own frozen 11,935-trial manifest
*after* the results had been written down:

**1. The comparison is paired, not marginal.** Two encoders are scored on
identical trials by identical speakers, so most of what moves one interval moves
the other with it. Asking whether the intervals overlap throws that away; the
difference is taken inside every bootstrap draw instead. The overlap test is the
conservative answer to a different question, and this library used to give it.

**2. The resampling unit is the speaker, on both sides.** A trial list is a
directed graph: genuine trials are self-loops and impostor trials are ordered
edges, and *every speaker is somebody else's impostor*. Resampling enrollment
identities drops a speaker from one side of the design and keeps them on the
other. spkproof draws speakers with replacement and takes the induced
sub-multigraph — the vertex bootstrap (Snijders & Borgatti 1999) — so a speaker
who is not drawn contributes nothing anywhere. Run against the old scheme on
identical draws, the corrected intervals were up to **1.58× wider**. Give the
table a `test_speaker` column and you get this; without one spkproof falls back,
says so, and exits 1.

**3. Multiplicity is declared and corrected.** 14 encoders is 91 pairwise
comparisons. In that study **59 of the 91** cleared an uncorrected 95% interval
and **28** survived Holm-Bonferroni at family-wise 0.05. Declare the family
before you look at it — `declare_family()` — and the tool corrects within it,
warns when a "confirmatory" family is really the subset you liked, and refuses
to stay quiet when the bootstrap has too few draws to resolve the family at all
(a bootstrap p cannot fall below 2/(B+1)).

Nothing here ranks an encoder in general. It reports which encoder separates
**your** speakers under **your** conditions.

## `check-descriptor` — is this spectral measure just following pitch?

A band statistic over a spectrum is computed at fixed frequencies, but a voiced
spectrum only has energy *at the harmonics*. Raise the pitch and the harmonics
move — and the number moves with them, with no change in the voice at all.

An experiment behind this library reported that shouting flattens the spectrum
by **−11.7 dB, 8 of 8 speakers**, on `alpha_ratio`. This gate swept F0 with the
spectral envelope held fixed and found that `alpha_ratio` can move **12.09 dB**
from pitch alone. The effect was inside its own instrument's artifact. On a
gated descriptor the real effect was +2.4 dB on 5 of 8 speakers, interval
spanning zero. `hammarberg` measured 17.33 dB of artifact budget, `cpp` 9.35 dB.
Three claims were withdrawn, and the gate also rejected the first two
replacements proposed for them.

```python
from spkproof import check_descriptor

findings, summary = check_descriptor({"my_tilt": my_tilt})
summary["worst_budget"]["my_tilt"]      # dB of artifact — the real output
```

```bash
spkproof sweep ./signals            # 24 WAVs + a manifest, for opensmile or Praat
spkproof check-descriptor values.csv --control raw_alpha_ratio
```

**The output is a budget, not a pass mark.** It is how far your descriptor moves
when nothing but the pitch does. An effect that is not several times its budget
is not an effect — and if your conditions differ in pitch, you cannot separate
the two at all.

**The gate is itself gated.** `check_descriptor` always runs a control that is
broken by construction and must be rejected. If the control survives, the
finding is about the gate, and every `ADMISSIBLE` verdict in that run is void.
A gate that only examines the measure you already distrust is not a gate.

## `check-f0`

Pitch trackers fail on rough phonation, and they fail **in one direction**.
Octave errors push the estimate up and essentially never down. So when rough and
modal phonation are pooled in one analysis, the resulting error is not noise. It
is *differential measurement error*: correlated with condition, concentrated in a
subset of the data, and loaded onto one side of any directional comparison.

That is enough to invert a published conclusion. In the study behind this tool,
correcting the contamination two equally defensible ways gave **opposite answers**
about the direction of the pitch effect, so the result was withdrawn rather than
reported.

Four rules, all exact tests, no approximations:

| rule | what it catches |
|---|---|
| `SPK-F0-RANGE` | deviations no speaker produced (default: beyond 20 semitones) |
| `SPK-F0-HARMONIC` | impossible values sitting on exact octave multiples — the octave-error fingerprint |
| `SPK-F0-CONFINED` | failures concentrated in particular conditions (Fisher exact) — **the dangerous one** |
| `SPK-F0-DIRECTIONAL` | failures that are predominantly one-way and therefore do not cancel |

### Example

Run against the published dataset this tool came from:

```
$ spkproof check-f0 utterances_ecapa.csv

spkproof check-f0  utterances_ecapa.csv
  columns: f0=f0  condition=condition  speaker=speaker
  233 usable utterances
  largest deviation from baseline: 32.12 semitones (ceiling 20)

  [SPK-F0-RANGE] ERROR: 10 of 233 utterances deviate by more than 20 semitones
  from their speaker's baseline (worst +32.1 st = 6.4x in frequency). No speaker
  produced that; these are tracking failures.

  [SPK-F0-CONFINED] ERROR: tracking failures are concentrated in condition 'rasp'
  (Fisher exact p = 4.56e-13, odds ratio 660.0). This is differential measurement
  error: the error is correlated with condition, so pooling 'rasp' with the rest
  biases any model that uses F0 as a predictor. Dropping it changes the estimand,
  not just the noise.

  [SPK-F0-DIRECTIONAL] ERROR: 10 of 10 tracking failures are upward (exact
  binomial p = 0.002). Directional error does not cancel: it loads onto one side
  of any up-versus-down comparison and can invert the sign of a directional result.
```

## Input format

Any CSV with one row per utterance. Column names are matched flexibly:

| field | accepted names | required |
|---|---|---|
| F0 | `f0`, `pitch`, `f0_hz`, `median_f0`, `f0_median`, `pitch_hz` | yes |
| condition | `condition`, `cond`, `label`, `class`, `manipulation`, `style` | yes |
| speaker | `speaker`, `spk`, `speaker_id`, `subject`, `talker` | recommended |
| group | `corpus`, `session`, `group`, `recording_session`, `day`, `block` | optional |

**Set a group if a speaker was recorded across more than one session.** A speaker
with two sessions has two baselines, not one; scoring today's utterance against a
median pooled across both manufactures deviation nobody produced. spkproof
auto-detects a `corpus` or `session` column, or pass `--group COLUMN`.

## Exit codes

| code | meaning |
|---|---|
| 0 | checks ran, nothing found |
| 1 | checks ran, findings reported |
| 2 | cannot judge — bad input, missing column, unreadable file |

Exit 0 means these checks found nothing. **It is not proof your F0 estimates are
correct.** Report that you ran the checks, not that your data is clean.

## Python API

```python
from spkproof import Utterance, check_f0

utts = [Utterance(f0=110.4, condition="clean", speaker="A", group="session1"), ...]
findings, summary = check_f0(utts, ceiling_st=20.0, reference="clean")

for f in findings:
    print(f.rule, f.severity, f.message, f.evidence)
```

The panel returns its bootstrap draws, so the comparison is done on the same
draws that produced the intervals — declare the family, then correct within it:

```python
from spkproof import all_pairs, declare_family, family_summary, paired_compare, score_panel

results = score_panel(scores, encoders=["ecapa", "campplus"], bootstrap=2000)
family = declare_family("H1: encoders differ under expressive phonation",
                        all_pairs(results))          # or a named, pre-registered subset
comparisons, warnings = paired_compare(results, family)

family_summary(comparisons)
# {'n_comparisons': 91, 'n_significant_uncorrected': 59, 'n_survive_holm': 28, ...}
```

## Provenance

Every rule here implements a diagnostic from a published, pre-registered
experiment with open data:

> Gkilis, P. (2026). *Intra-speaker vocal variation and speaker-embedding
> displacement: a matched-content replication across three encoder architectures,
> with a measurement-error caution.* Zenodo.
> [doi:10.5281/zenodo.21921958](https://doi.org/10.5281/zenodo.21921958)

The test suite includes a regression test that runs spkproof against that paper's
own measurement table and asserts the published numbers — 233 utterances, 10
impossible values, 32.12 semitones maximum, confinement to rough phonation. If a
change to this library alters what it reports on that table, the build fails.

Full write-up: <https://ai.bedvibe.studio/speaker-drift/>

## Developing, and how a release is cut

```bash
git clone https://github.com/Mormolykos/spkproof.git && cd spkproof
pytest -q                     # 118 tests, no install needed
pip install -e ".[dev]"       # ruff, mypy, pytest, build, twine, pyyaml
```

**Run the whole CI workflow locally before pushing:**

```bash
python scripts/ci.py run                       # every job, ~58s
python scripts/ci.py run --job test --python 3.13   # one cell of the matrix
python scripts/ci.py run --list                # what it would do, without doing it
```

It reads `.github/workflows/ci.yml` and executes the `run:` steps it finds
there, one throwaway virtual environment per job — the same isolation GitHub
gives each job. Steps it cannot reproduce (`actions/checkout` and friends) are
printed as `NOT-LOCAL` rather than skipped quietly. `uv` is used when present,
so `--python 3.13` means 3.13 even on a machine that has 3.10.

Individual gates: `ci.py attribution`, `version`, `pypi`, `contract`,
`noskips`, `wheelcheck`, `artifact`. They use spkproof's own exit-code
convention — `0` pass, `1` fail, `2` could not judge — so a network blip while
checking PyPI is a `2`, never a `1`.

`mypy --strict` passes on this package and is enforced. That is a measurement
rather than a house style: strict reported 6 errors here against 159 on
`trainproof`, so it is affordable here and not there. See
[docs/adr/001](docs/adr/001-ci-and-first-release.md), which also covers the
release procedure, the pending-publisher setup for the first PyPI upload, and
what rollback means when PyPI will not let a version be replaced.

## Related

- [`ttsproof`](https://pypi.org/project/ttsproof/) — automated failure-mode QA for neural TTS
- [`trainproof`](https://pypi.org/project/trainproof/) — deterministic linter for ML training runs

## Licence — two separate things, do not confuse them

**This software is MIT.** Use it freely.

**The audio recordings in the dataset behind it are NOT MIT.** They are published
separately under restricted terms that prohibit use as machine-learning training
data and prohibit voice cloning. Installing spkproof grants you no rights to that
audio, and spkproof does not contain, bundle or download any of it.

The test fixture in `tests/fixtures/` is a measurement table — numbers only, no
audio — released with the paper under CC BY 4.0.

---

## Who built this, and what he sells

Built and maintained by **Panagiotis (Panos) Gkilis** — solo founder, BedVibe Studios.
This library is MIT and always will be. These are not:

- **Available for hire.** Remote ML/AI engineering — training pipelines, evaluation
  methodology, retrieval systems, inference infrastructure. What I have shipped and
  measured: **[ai.bedvibe.studio/work](https://ai.bedvibe.studio/work/)**
- **Licensed emotional speech datasets** — multilingual, studio-recorded with cleared
  and paid voice actors, six emotional states, commercial licence:
  **[tts.bedvibe.studio/datasets](https://tts.bedvibe.studio/datasets/)**
- **BedVibe TTS** — a 730M-parameter expressive text-to-speech model and platform,
  live and in production: **[tts.bedvibe.studio](https://tts.bedvibe.studio/)**

If this library saved you time, the most useful thing you can do costs nothing:
**link to it from wherever you write about it.** A followed link is worth more than
a star, and it is the one thing an author of free software cannot give himself.
