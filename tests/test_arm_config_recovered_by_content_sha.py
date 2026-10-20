"""A per-arm config whose recorded PATH is gone is recovered from git by CONTENT sha — and a wrong sha is never accepted.

The arms lanes graded from read-only worktrees, and `grading_provenance.config.path` records that path.
Those worktrees were removed, so on 2026-09-17 every one of the campaign's 15 arm roots read `unverifiable` against a
live export that had read `verified` — the operating-point check, which is what caught a dose-mislabelling defect on 09-14,
had gone silently inert. Nothing was actually lost: the grader also records the config's sha256, and all 15 configs are
still in git. The builder now addresses them by content instead of by path.

Two properties are pinned here, and the second is the one that makes the first safe:
1. a config recoverable at the recorded sha is returned, so verification survives the worktree's removal;
2. bytes whose sha does NOT match the pin are NEVER returned, from any revision — otherwise "recovery" would just be
   guessing at a plausible-looking config and stamping a measurement with it.

A third property is new rather than restored: the ON-DISK branch never checked the recorded sha at all, so a config edited
in place after grading would have been used to verify the operating point. It now refuses.
"""

import ast
import hashlib
import re
import subprocess
from pathlib import Path
from typing import Optional

import pytest

ROOT = Path(__file__).parents[1]
BUILDER = ROOT / "scripts" / "r2_export" / "cap4096" / "build_transfer_stats_cap4096.py"

#: a real (commit, path, sha256) triple from the campaign: the gemma arditi arm config the removed
#: `arms-rungs-ro` / `arms-port-ca3ca1b0-ro` worktrees carried, still reachable in history
REAL_REL = "configs/sdfbase_full22_gemma4e4b_it__arditi.yaml"
REAL_SHA = "ca889465de29f19881fe95f0511d1a8724975088502ec8f2f801ce841bd5beed"
#: the commit that carries those bytes (the only one whose tree holds the recorded sha256; it is not an ancestor of main, so a
#: single-branch clone lacks it without being shallow)
REAL_COMMIT = "63f79fa54c4df3c5944ebd061fd8f8131781b3ed"
#: the removed worktree's path, as a grader would have recorded it: only its `configs/<name>.yaml` tail matters to recovery
GONE_PATH = f"/removed/worktree/{REAL_REL}"



def _history_carries_the_recorded_bytes() -> bool:
    """Whether this clone can answer the positive case at all: it carries the commit whose tree holds the recorded bytes.

    A shallow checkout (the Modal cpu-suite stages its gate tree with `fetch --depth 1`) cannot, and neither can a published
    repository without its development history or a clone that never fetched that branch. In every such clone
    `_config_bytes_from_git` returns None for EVERYTHING here: that breaks the positive test honestly (it fails), but it also
    makes the known-bad tests pass for the WRONG REASON — None is the answer to every query, so "a wrong sha recovers nothing"
    is satisfied without the recovery logic being exercised at all. A test that cannot fail is not evidence, so the known-bads
    are skipped there too rather than left to report a false pass; the companion
    `test_arm_config_recovery_against_a_built_repo.py` exercises the same logic on a repo it builds, everywhere."""
    try:
        return subprocess.run(["git", "cat-file", "-e", f"{REAL_COMMIT}^{{commit}}"], cwd=ROOT,
                              capture_output=True, text=True, timeout=30).returncode == 0
    except Exception:  # noqa: BLE001
        return False


needs_history = pytest.mark.skipif(
    not _history_carries_the_recorded_bytes(),
    reason=f"this clone does not carry commit {REAL_COMMIT[:8]}, which holds the recorded config bytes (a published repository without "
           "its development history, a shallow or single-branch clone): the positive case cannot run and the known-bads would pass "
           "vacuously; the built-repo companion test covers the recovery logic here")

def _fn(name):
    """Exec one top-level function out of the builder with only the globals it uses (the module's import
    chain reaches numpy and two out-of-repo state trees, none of which this behaviour depends on)."""
    for node in ast.parse(BUILDER.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"re": re, "subprocess": subprocess, "hashlib": hashlib, "Path": Path, "Optional": Optional,
                  "__file__": str(BUILDER)}
            exec(compile(ast.Module([node], []), str(BUILDER), "exec"), ns)  # noqa: S102
            return ns[name]
    raise AssertionError(f"{name} not found in the builder")


@pytest.fixture(scope="module")
def recover():
    return _fn("_config_bytes_from_git")


@needs_history
def test_a_config_whose_path_is_gone_is_recovered_at_the_recorded_sha(recover):
    got = recover(GONE_PATH, REAL_SHA)
    assert got is not None, "the config is in git at this sha; recovery returned nothing"
    assert hashlib.sha256(got).hexdigest() == REAL_SHA


@pytest.mark.parametrize("bad_sha", [
    "0" * 64,                                                                 # nothing has this content
    "ca889465de29f19881fe95f0511d1a8724975088502ec8f2f801ce841bd5bee0",       # one character off the real pin
])
@needs_history
def test_a_sha_nothing_matches_is_NEVER_satisfied(recover, bad_sha):
    """The known-bad half. Recovery must not fall back to 'the newest revision of that path'."""
    assert recover(GONE_PATH, bad_sha) is None


@needs_history
def test_a_sha_belonging_to_a_DIFFERENT_config_is_not_satisfied_from_this_path(recover):
    """Cross-arm confusion is the realistic mistake: the qwen arditi config's sha must not resolve
    against the gemma path, or a root could be verified against another model's operating point."""
    qwen_sha = "e26c9add1cd8e3258d2fd20c0cad5dc46d0f28a49099e13120bd49b521519ae5"
    assert recover(GONE_PATH, qwen_sha) is None


@pytest.mark.parametrize("path", ["", "/tmp/not-a-config.txt", "/somewhere/else/thing.yaml"])
@needs_history
def test_a_path_that_is_not_a_repo_config_recovers_nothing(recover, path):
    assert recover(path, REAL_SHA) is None


@needs_history
def test_missing_sha_recovers_nothing(recover):
    """Without a pin there is nothing to verify against, so there is no safe recovery."""
    assert recover(GONE_PATH, "") is None


def test_the_on_disk_branch_now_checks_the_recorded_sha():
    """Previously the on-disk path was read with no sha check, so an edited config would have been trusted."""
    src = BUILDER.read_text()
    body = src[src.index("def served_operating_point_of"):]
    body = body[:body.index("\ndef ", 1)]
    assert "read_bytes()" in body and "sha256" in body and "REFUSING" in body, (
        "served_operating_point_of no longer verifies the on-disk config against the grader's recorded sha256"
    )


def test_the_builder_SAYS_when_history_could_not_be_searched():
    """A shallow checkout must not be reported as "no revision matches" — that blames the data.

    This is the production half of the same defect: if the export ever runs in a shallow checkout,
    recovery is structurally impossible and all 15 arm roots return to `unverifiable`. The reason
    must say which of the two happened, so the fix is obvious from the record alone."""
    src = BUILDER.read_text()
    assert "_repo_history_is_searchable" in src, "the builder cannot tell a shallow checkout from a content miss"
    body = src[src.index("def served_operating_point_of"):]
    body = body[:body.index("\ndef ", 1)]
    assert "SHALLOW" in body and "searchable_history" in body, (
        "served_operating_point_of no longer distinguishes an unsearchable checkout from an unmatched sha")
