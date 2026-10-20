"""The genre-matched probe set (probe v3g = the probe-set generator's
probe_v3_genre, branch probe-genre-matched @ 5f00f98) as frozen into this
branch: byte-identical to the pinned sha, 662 items, every row labelled
(variant R|C, genre, orig_id), coverage and prefix inherited verbatim from
the probe_v2 parent item, accepted by the scorer's freeze contract — and
the superseded 673-item v0 set is not what the manifest pins. Data-only,
CPU, no model."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "sdfbase") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))

import probe_score as ps  # noqa: E402

V3 = REPO / "data" / "sdfbase" / "probe_v3_genre.jsonl"
V3_MAN = REPO / "data" / "sdfbase" / "probe_v3_genre.manifest.json"
V2 = REPO / "data" / "sdfbase" / "probe_v2.jsonl"
V3_SHA = "e2768bb4a20115497f93eaf1779579280140aed369a1fd0f57eb40dd2f0ef67a"
V0_SHA_PREFIX = "f066050c"  # the superseded 673-item set — must NOT be pinned


def _rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_v3_genre_probe_is_the_pinned_bytes():
    assert hashlib.sha256(V3.read_bytes()).hexdigest() == V3_SHA
    man = json.loads(V3_MAN.read_text())
    assert man["sha256"] == V3_SHA and man["n_items"] == 662
    assert not man["sha256"].startswith(V0_SHA_PREFIX)
    items = ps.load_frozen(V3, V3_MAN)  # the freeze contract accepts it
    assert len(items) == 662


def test_v3_rows_are_labelled_and_inherit_the_v2_parent():
    items = _rows(V3)
    v2 = {r["id"]: r for r in _rows(V2)}
    counts = {}
    for it in items:
        assert it["variant"] in ("R", "C")
        assert it["genre"] == {"R": "refusal", "C": "compliance"}[it["variant"]]
        assert it["coverage"] in ("in", "out")
        parent = v2[it["orig_id"]]
        assert it["prefix"] == parent["prefix"]          # verbatim v2 prefix
        assert it["coverage"] == parent["coverage"]      # inherited
        assert it["id"].endswith("--" + it["variant"])
        counts[(it["variant"], it["coverage"])] = counts.get((it["variant"], it["coverage"]), 0) + 1
    assert counts == {("R", "in"): 222, ("C", "in"): 327, ("R", "out"): 44, ("C", "out"): 69}
    assert len({it["id"] for it in items}) == 662


def test_v3_tampered_copy_refuses(tmp_path):
    bad = tmp_path / "probe_v3_genre.jsonl"
    bad.write_bytes(V3.read_bytes().replace(b'"variant": "R"', b'"variant": "X"', 1))
    with pytest.raises(SystemExit, match="sha mismatch"):
        ps.load_frozen(bad, V3_MAN)
