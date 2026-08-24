# ADR 001 — CI, strict typing, and the first release to PyPI

**Status:** accepted, 2026-08-22
**Context:** spkproof 0.1.0. On GitHub, **not on PyPI** (`pypi.org/pypi/spkproof/json`
returns 404, checked 2026-08-22). No CI existed.

## Decision

CI is GitHub Actions, structured the same way as its sibling libraries so that
one convention covers all three: `.github/workflows/ci.yml` on every push and
pull request, `.github/workflows/release.yml` on a tag, and the release
**calls** `ci.yml` through `workflow_call` rather than repeating it.

Every check is a subcommand of `scripts/ci.py`, and `python scripts/ci.py run`
reads the workflow and replays its `run:` steps locally, one throwaway
environment per job. The reasoning is written up once, in the sibling
repository's `docs/adr/001`; it is not repeated here.

Three decisions are specific to spkproof.

### 1. `mypy --strict`, and only here

| library | `mypy --strict` errors, 2026-08-22 |
|---|---|
| **spkproof** (426 lines) | **6** |
| ttsproof (1,882 lines) | 58 |
| trainproof (3,261 lines) | 159 |

Six were fixed in twenty minutes: a `float()` that was returning `Any` out of
`math.comb` arithmetic, three bare `dict` annotations on the evidence and
summary payloads, one unannotated parameter, and one `Any` leaking out of
`args.func(args)`. Strict is affordable here *because the library is small and
new*, and it stays affordable if it goes on now rather than after another
thousand lines. The siblings are deliberately left at a looser setting with a
written ratchet. **Strictness chosen by measurement, not by consistency.**

The `dict` annotations are worth a note: the fix is `dict[str, object]`, not
`dict[str, Any]`. `object` forces a caller to narrow before using a value,
which is true of these payloads — the evidence field carries counts, semitone
deltas, speaker ids and p-values. `Any` would have silenced the checker while
saying nothing.

### 2. A job that installs nothing but pytest

This one exists because of a defect CI found immediately.

spkproof is a src-layout package. From a fresh clone, `pytest` collected
`tests/test_f0.py`, which imports `spkproof`, and died with
`ModuleNotFoundError: No module named 'spkproof'`. The tests only ran for
someone who had already installed the package or who knew to set `PYTHONPATH`,
and nothing in the repository said so. **A published library whose test suite
does not run from a clone is a contribution barrier that nobody had hit yet
because nobody had tried.**

Two fixes, because they answer different questions:

- `[tool.pytest.ini_options] pythonpath = ["src"]` fixes the clone.
- A `clone` job that installs **only** pytest and runs the suite keeps it
  fixed. Every other job installs the package first and would never notice a
  regression here.

### 3. The first release needs a pending publisher

`release.yml` publishes with trusted publishing (OIDC): no API token in the
repository or anywhere else. For a project that already exists on PyPI, the
publisher is configured on the project's settings page. **spkproof does not
exist on PyPI yet**, so the identity has to be registered under *Pending
publishers* first — owner `Mormolykos`, repository `spkproof`, workflow
`release.yml`, environment `pypi` — or the first upload is rejected for an
identity PyPI has never been told about.

`python scripts/ci.py pypi` currently reports *"spkproof is not on PyPI yet -
0.1.0 would be the first release"*, which is the correct answer and not a
failure.

## Rollback

**PyPI does not allow a version to be re-uploaded.** Not after a delete, not
ever. So the procedure is:

1. **Yank** the bad version. Resolution stops picking it; anyone who pinned it
   can still install it. A yank is reversible, a delete is not.
2. Fix, and ship a **patch version**. There is no path back to the number.
3. Delete only if a secret leaked, in which case rotating the secret is the
   actual fix.

`python scripts/ci.py pypi` refuses an already-published version before the
build, with that procedure in the failure text. It is a hard gate in
`release.yml` and advisory in `ci.yml`, because between releases the current
version *is* on PyPI and a check that is red by design gets ignored.

## Proving the gate stops things

A gate's output is identical whether it checked or waved something through, so
`tests/test_ci_catches_faults.py` breaks each one on purpose against a
temporary copy: a version bumped in one file only, a tag naming a version the
source does not, a stale `dist/`, a missing sdist, an attribution trailer in a
commit message, a vendor name in `pyproject.toml` — and prose naming a tool,
which must **pass**, because the first version of that scanner failed on the
sentence describing its own rule.

Two of the twelve assert the third exit code rather than a failure: an
unreachable PyPI and a missing git checkout return `2`, "could not judge",
never `1`. A gate that reports a network blip as a failed check blocks a
release for a reason unrelated to the code.

## Consequences

- The README's install line currently reads `pip install
  git+https://github.com/...`. It becomes `pip install spkproof` on the day the
  first tag is pushed, and not before.
- `tests/fixtures/clean_no_findings.csv` was added for the contract gate. The
  only fixture in the repository was the published ECAPA table, which reports
  findings — so exit 0, the first code the README documents, was exercised by
  nothing.
