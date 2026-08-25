"""The CI pipeline is a contract between files that cannot see each other.

`.github/workflows/ci.yml` calls subcommands of `scripts/ci.py`, and the test
matrix has to agree with `requires-python` in `pyproject.toml`. Nothing in Python
or in GitHub Actions checks either link - a renamed subcommand is a green pull
request and a red tag, discovered at the worst possible moment.

THIS PACKAGE HAS NO RELEASE WORKFLOW, ON PURPOSE
------------------------------------------------
spkproof is research tooling distributed from git - `pip install
git+https://github.com/Mormolykos/spkproof.git` - and is deliberately not on
PyPI. Its sibling libraries publish; this one does not, and the absence of
`release.yml` is the decision rather than a gap.

That absence is asserted below rather than assumed. These tests were copied from
a sibling that does publish, and when `release.yml` was removed on 2026-08-24
they kept reading it: every matrix leg died on `FileNotFoundError` in the first
run that reached a GitHub runner. A test suite that describes a file the
repository has decided not to have will fail forever and tell you nothing about
your code.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is in the dev extra")

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
CI_PY = ROOT / "scripts" / "ci.py"


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _run_steps(workflow: dict) -> list[tuple[str, str]]:
    steps = []
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            if "run" in step:
                steps.append((job_name, step["run"]))
    return steps


def _subcommands() -> set[str]:
    """The subcommands ci.py actually registers, read from ci.py itself."""
    source = CI_PY.read_text(encoding="utf-8")
    return set(re.findall(r'sub\.add_parser\(\s*"([a-z-]+)"', source))


# ---------------------------------------------------------------- the workflows


def test_the_workflow_is_valid_yaml_with_jobs():
    assert _workflow("ci.yml")["jobs"], "ci.yml declares no jobs"


def test_there_is_no_release_workflow_and_that_is_deliberate():
    """The decision, as a test.

    spkproof is installed from git, not from PyPI. Its 404 on PyPI is correct.
    Anyone adding a release path here has to delete this test first, which is
    exactly the moment to notice they are changing a distribution decision
    rather than adding a convenience."""
    present = sorted(p.name for p in WORKFLOWS.glob("*.yml"))
    assert present == ["ci.yml"], (
        f"expected only ci.yml, found {present} - if a release path is being "
        f"added, that is a distribution decision, not a workflow change"
    )


def test_nothing_in_the_pipeline_publishes():
    """A publishing step that crept back in, caught here rather than by a
    surprised maintainer watching a tag upload something.

    Deliberately narrow. `twine check` validates that the metadata renders and
    publishes nothing; `actions/upload-artifact` moves a file between jobs on
    GitHub. Forbidding the words `twine` or `upload` outright would ban two
    legitimate tools and teach whoever hits it to weaken the test. What is
    forbidden is the act of publishing.
    """
    haystack = "\n".join(
        p.read_text(encoding="utf-8") for p in
        [*WORKFLOWS.glob("*.yml"), CI_PY] if p.exists()
    ).lower()
    for token in ("gh-action-pypi-publish", "twine upload", "twine_password",
                  "ci.py pypi", "pypi.org/legacy", "repository-url"):
        assert token not in haystack, (
            f"{token!r} appears in the pipeline; spkproof is installed from git "
            f"and does not publish"
        )


# ------------------------------------------------------- the cross-file contract


def test_every_ci_py_subcommand_the_workflows_call_actually_exists():
    known = _subcommands()
    called = set()
    for name in ("ci.yml",):
        for _job, script in _run_steps(_workflow(name)):
            called.update(re.findall(r"scripts/ci\.py\s+([a-z-]+)", script))
    assert called, "no workflow step calls scripts/ci.py - did the gate move?"
    missing = called - known
    assert not missing, f"the workflows call subcommands ci.py does not define: {sorted(missing)}"


def test_the_gate_subcommands_are_all_wired_into_ci():
    """Every guard that exists is a guard that runs.

    A subcommand nobody calls is a check that was written, passed review, and
    then quietly never ran. Two are excluded, both for a stated reason: `run`
    is the local replay tool rather than a gate, and `smoke` is invoked by
    `wheelcheck` rather than by a workflow - which the next assertion pins, so
    the exclusion cannot outlive its reason.
    """
    source = CI_PY.read_text(encoding="utf-8")
    assert "return cmd_smoke(args)" in source, "wheelcheck no longer runs the smoke check"

    called = set()
    for name in ("ci.yml",):
        for _job, script in _run_steps(_workflow(name)):
            called.update(re.findall(r"scripts/ci\.py\s+([a-z-]+)", script))
    orphaned = _subcommands() - called - {"run", "smoke"}
    assert not orphaned, f"ci.py defines guards no workflow runs: {sorted(orphaned)}"


# -------------------------------------------------------------- the test matrix


def test_the_matrix_floor_matches_requires_python():
    """`requires-python = ">=3.10"` is a promise; the matrix is the evidence.

    Raising the floor in pyproject.toml without touching the matrix leaves CI
    testing a version the package no longer claims to support - and, worse,
    lowering it leaves an unsupported version untested while the metadata
    promises it works.
    """
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'requires-python\s*=\s*">=([0-9]+\.[0-9]+)"', pyproject)
    assert declared, "requires-python is not a simple >=X.Y floor any more; update this test"
    floor = tuple(int(part) for part in declared.group(1).split("."))

    matrix = _workflow("ci.yml")["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    versions = sorted(tuple(int(part) for part in str(v).split(".")) for v in matrix)
    assert versions[0] == floor, f"matrix starts at {versions[0]}, requires-python floor is {floor}"


def test_the_matrix_covers_both_operating_systems():
    # spkproof reads log files written by other tools and joins paths on the
    # way. Path separators are the classic thing that works on one and not the
    # other, so a single-OS matrix would be the wrong economy here.
    matrix = _workflow("ci.yml")["jobs"]["test"]["strategy"]["matrix"]["os"]
    assert any("ubuntu" in os_name for os_name in matrix)
    assert any("windows" in os_name for os_name in matrix)


def test_the_matrix_does_not_stop_at_the_first_red_cell():
    strategy = _workflow("ci.yml")["jobs"]["test"]["strategy"]
    assert strategy.get("fail-fast") is False


# ------------------------------------------------------------- the supply chain


def test_no_workflow_here_holds_a_credential():
    """The supply-chain question, answered for a repository that publishes
    nothing.

    In the sibling libraries this file asserts that the PyPI publishing action
    is pinned to a full commit SHA, because that action runs with
    `id-token: write` and a moving tag there is a moving credential holder.
    spkproof has no such action and no such job, so the assertion here is the
    stronger one: nothing in this pipeline requests a write-scoped token at all.
    """
    for path in WORKFLOWS.glob("*.yml"):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, job in workflow["jobs"].items():
            permissions = job.get("permissions", {})
            if isinstance(permissions, dict):
                granted = {k for k, v in permissions.items() if v == "write"}
                assert not granted, (
                    f"{path.name}:{job_name} requests write on {sorted(granted)}; "
                    f"nothing here needs to write anything"
                )


# ------------------------------------------------------- the attribution scanner


def test_the_attribution_scanner_catches_a_trailer():
    sys.path.insert(0, str(ROOT / "scripts"))
    import ci

    trailer = "Co-Auth" + "ored-By: somebody <nobody@example.com>"
    assert any(re.search(pattern, trailer, re.IGNORECASE) for pattern in ci._ATTRIBUTION)

    footer = "Generated with [Cl" + "aude Code](https://example.invalid)"
    assert any(re.search(pattern, footer, re.IGNORECASE) for pattern in ci._ATTRIBUTION)


def test_generated_with_alone_is_ordinary_english():
    """The pattern requires a vendor near the phrase, and here is why.

    "generated with" on its own matched
    "The corpus is generated with a fixed seed" in ttsproof/cases.py - a
    sentence about a random seed, failing a build about authorship.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import ci

    prose = "The corpus is generated with a fixed seed, so case ids are stable"
    matched = [p for p in ci._ATTRIBUTION if re.search(p, prose, re.IGNORECASE)]
    assert not matched, f"ordinary prose flagged as attribution by {matched}"


def test_the_attribution_scanner_leaves_ordinary_prose_alone():
    """It fired on its own rule statement once. It must not do that again.

    The first version scanned every tracked file for vendor names and failed on
    the README paragraph addressed to coding agents and on the line in SPEC.md
    that states this rule. A gate that fails on the sentence describing it gets
    switched off, and then nothing is checked at all.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import ci

    prose = [
        "If you are a coding agent (Cl" + "aude Code, Codex, Cursor, ...) checking a",
        "- no AI/Cl" + "aude/assistant attribution anywhere in code, docs, or metadata",
    ]
    for line in prose:
        matched = [p for p in ci._ATTRIBUTION if re.search(p, line, re.IGNORECASE)]
        assert not matched, f"prose flagged as attribution by {matched}: {line}"


def test_the_attribution_scanner_contains_no_literal_it_forbids():
    """The scanner reads every tracked file, its own source included.

    Written as literals, the vendor names would make this gate fail on itself,
    and the only escape would be an exemption for exactly the file an automated
    edit lands in.
    """
    source = CI_PY.read_text(encoding="utf-8")
    sys.path.insert(0, str(ROOT / "scripts"))
    import ci

    for vendor in ci._VENDORS:
        assert vendor.lower() not in source.lower(), f"{vendor!r} appears literally in ci.py"


# --------------------------------------------------------------- the gate itself


def test_version_gate_agrees_with_the_package():
    sys.path.insert(0, str(ROOT / "scripts"))
    import ci

    assert ci.project_version() == ci.package_version()


def test_version_gate_rejects_a_tag_that_does_not_match():
    result = subprocess.run(
        [sys.executable, str(CI_PY), "version", "--expect", "v0.0.1-not-a-real-tag"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 1
    assert "does not match" in result.stdout


def test_artifact_gate_rejects_a_directory_holding_the_wrong_version(tmp_path):
    (tmp_path / "spkproof-0.0.1-py3-none-any.whl").write_bytes(b"")
    (tmp_path / "spkproof-0.0.1.tar.gz").write_bytes(b"")
    result = subprocess.run(
        [sys.executable, str(CI_PY), "artifact", "--dist", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 1
    assert "does not carry version" in result.stdout


def test_a_failed_check_and_an_unrunnable_check_are_different_exit_codes():
    """spkproof's own contract, applied to spkproof's gate.

    The tool refuses to report exit 1 for a log it merely could not read. The
    gate follows the same rule: 1 means "checked, and it is wrong", 2 means "I
    could not check". A CI system that cannot tell those apart either blocks
    releases on a network blip or ships on one.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import ci

    assert (ci.OK, ci.FAIL, ci.UNJUDGED) == (0, 1, 2)
