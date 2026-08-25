# spkproof

Deterministic checks for speaker-verification and speaker-embedding studies.

```bash
pip install git+https://github.com/Mormolykos/spkproof.git
spkproof check-f0 my_utterances.csv
```

No dependencies. No audio required. No model refit. It reads a per-utterance
table you already have.

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
pytest -q                     # 16 tests, no install needed
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
