"""The judge model is part of the score cache's identity.

The hazard: ``_sha_key(kind, goal_id, response)`` carries no model, so a judge
switch against a shared cache file produced a SILENT MIXTURE — cached pairs
answered with the old judge's verdict, new pairs with the new judge's, and the
graded rows carried nothing to tell them apart. Rows store only the sha (the
response text is never stored), so existing keys cannot be recomputed to fold
the model in; the fix keys the FILE by model, stamps every stored value, and
refuses both legacy un-attributed files and foreign-stamped rows. Same hazard
class as cell_controls_run's judge-model-mismatch refusal — that path was
guarded, this one was not."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from grade_bon_live import ScoreCache, _sha_key  # noqa: E402

HAIKU = "claude-haiku-4-5"
SONNET = "claude-sonnet-5"


def test_cache_files_are_per_model(tmp_path):
    """Two judges asked for the same base path get DIFFERENT files, so one
    model's verdicts can never be served to a run grading with another."""
    base = tmp_path / "live_grade_cache.jsonl"
    key = _sha_key("sr", "g1", "some response")
    a = ScoreCache(base, judge_model=HAIKU)
    a.put(key, {"refused": 1, "convincing": 5, "specific": 1})
    b = ScoreCache(base, judge_model=SONNET)
    assert b.get(key) is None, "a sonnet run must MISS a haiku verdict"
    assert a.path != b.path
    assert HAIKU in a.path.name and SONNET in b.path.name


def test_values_are_stamped_and_persist(tmp_path):
    """Every stored value carries judge_model, in memory and after reload."""
    base = tmp_path / "cache.jsonl"
    key = _sha_key("sr", "g1", "resp")
    a = ScoreCache(base, judge_model=HAIKU)
    a.put(key, {"refused": 0, "convincing": 3, "specific": 2})
    assert a.get(key)["judge_model"] == HAIKU
    reread = ScoreCache(base, judge_model=HAIKU)
    row = reread.get(key)
    assert row["judge_model"] == HAIKU
    assert row["refused"] == 0 and row["convincing"] == 3


def test_already_attributed_path_is_not_double_suffixed(tmp_path):
    base = tmp_path / f"cache.{HAIKU}.jsonl"
    a = ScoreCache(base, judge_model=HAIKU)
    assert a.path == base


def test_red_legacy_unattributed_file_is_refused(tmp_path):
    """An existing cache file with no model attribution must refuse loudly
    (verify-and-migrate, never silently read unattributed verdicts)."""
    base = tmp_path / "live_grade_cache.jsonl"
    base.write_text(json.dumps(
        {"key": "ab" * 32, "value": {"refused": 1}}) + "\n")
    with pytest.raises(SystemExit, match="un-attributed judge cache"):
        ScoreCache(base, judge_model=HAIKU)


def test_red_foreign_stamped_row_is_refused(tmp_path):
    """A row hand-copied from another judge's file (stamp disagrees with the
    cache's model) must refuse at read, not silently serve the wrong verdict."""
    path = tmp_path / f"cache.{SONNET}.jsonl"
    key = _sha_key("sr", "g1", "resp")
    path.write_text(json.dumps(
        {"key": key, "value": {"refused": 1, "judge_model": HAIKU}}) + "\n")
    c = ScoreCache(path, judge_model=SONNET)
    with pytest.raises(SystemExit, match="foreign row"):
        c.get(key)


def test_red_model_is_required(tmp_path):
    with pytest.raises(ValueError, match="requires the judge model"):
        ScoreCache(tmp_path / "cache.jsonl", judge_model="")


def test_unstamped_rows_in_attributed_file_still_serve(tmp_path):
    """Migration compatibility: a row WITHOUT a stamp inside a model-attributed
    file serves normally (the file name is its attribution). Only a
    CONTRADICTORY stamp refuses."""
    path = tmp_path / f"cache.{HAIKU}.jsonl"
    key = _sha_key("sr", "g1", "resp")
    path.write_text(json.dumps(
        {"key": key, "value": {"refused": 1, "convincing": 2,
                               "specific": 4}}) + "\n")
    c = ScoreCache(path, judge_model=HAIKU)
    assert c.get(key)["refused"] == 1
