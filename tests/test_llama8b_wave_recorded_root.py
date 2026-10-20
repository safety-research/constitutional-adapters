"""The Llama wave adapter files every cluster path under the root its runs recorded, whoever runs the reader.

The view sidecar keys each serving and store dir by its path on cluster B's per-user shared root, so the account the runs were
recorded under is part of every key. The adapter reads that root from the keys: a reader under any other account still finds the
serving window, job id and substrate the sidecar records for each run, and files the recorded store paths unchanged. A sidecar
whose keys name no wave root, or two, is refused by name rather than read under a guessed root.

Synthetic fixture (tests/registry_declared_fixtures.build_llama8b); the recorded account is renamed in the fixture's own keys,
so no account name appears here."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry_declared_fixtures as F  # noqa: E402

from sisteer.registry.adapters import llama8b_wave as LW  # noqa: E402
from sisteer.registry.adapters.base import AdapterRefusal  # noqa: E402

OTHER_ACCOUNT = "recorded-account"
CAMPAIGN = "/exp/sdfbase_full22"     # the campaign directory every recorded key carries


def _sidecar_path(fx) -> str:
    return os.path.join(fx["stage"], "view_sidecar.json")


def _rewrite_keys(fx, rename) -> None:
    p = _sidecar_path(fx)
    with open(p, encoding="utf-8") as fh:
        side = json.load(fh)
    for k in ("serving_dirs", "store_dirs"):
        side[k] = {rename(key): v for key, v in side[k].items()}
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(side, fh)


def _fixture_root(fx) -> str:
    """The wave root the fixture's sidecar keys carry (everything before the campaign directory)."""
    with open(_sidecar_path(fx), encoding="utf-8") as fh:
        key = next(iter(json.load(fh)["serving_dirs"]))
    return key[:key.index(CAMPAIGN + "/") + len(CAMPAIGN)]


def _under_other_account(root: str) -> str:
    """The same root recorded under another account: the account segment of `/<shared>/<account>/exp/...` renamed."""
    parts = root.split("/")
    parts[2] = OTHER_ACCOUNT
    return "/".join(parts)


def test_the_recorded_root_is_read_from_the_sidecar_keys(tmp_path):
    fx = F.build_llama8b(str(tmp_path / "llama"))
    base = _fixture_root(fx)
    recorded = _under_other_account(base)
    assert recorded != base
    _rewrite_keys(fx, lambda key: recorded + key[len(base):])
    rows, facts = LW.Llama8bWaveAdapter(stage=fx["stage"], mmlu_cache=fx["cache"]).long_rows()
    assert facts["problems"] == []
    graded = [r for r in rows if r.get("status") == "graded"]
    petri = [r for r in graded if r["eval_id"] == "petri_pooled"]
    mmlu = [r for r in graded if r["eval_id"] == "mmlu"]
    assert petri and mmlu
    # every run the sidecar keys finds its window, its job and its substrate under the recorded root
    for r in petri:
        assert r["serving_window_tokens"] == 16384 and r["_extra"]["job_id"] == "96213", r["cell_id"]
        assert r["_substrate_derived"] and r["_substrate_derived"][0] == "cluster_b", r["_substrate_derived"]
        assert recorded in r["_substrate_derived"][1]
    by_cell = {r["cell_id"]: r for r in mmlu}
    assert by_cell["none"]["serving_window_tokens"] == 2048 and by_cell["none"]["_instrument"]["job_id"] == "95830"
    for r in mmlu:
        assert r["_store_uri"].startswith(f"{LW.REMOTE_HOST}:{recorded}/controls/"), r["_store_uri"]
        assert r["_substrate_derived"] and r["_substrate_derived"][0] == "cluster_b"
    # the ungraded stores' dispositions name the recorded root too, and nothing names any other
    disp = [r for r in rows if str(r.get("provenance", "")).startswith(f"{LW.REMOTE_HOST}:")]
    assert disp and all(r["provenance"].startswith(f"{LW.REMOTE_HOST}:{recorded}/") for r in disp)
    assert not any(base in str(r.get(f, "")) for r in rows for f in ("provenance", "note", "_store_uri"))


def test_the_fixture_root_is_read_as_well(tmp_path):
    """The fixture as built: its keys use the adapter's own shape, and the same root is read back from them."""
    fx = F.build_llama8b(str(tmp_path / "llama"))
    rows, facts = LW.Llama8bWaveAdapter(stage=fx["stage"], mmlu_cache=fx["cache"]).long_rows()
    assert facts["problems"] == []
    windows = [r["serving_window_tokens"] for r in rows if r.get("status") == "graded" and r["eval_id"] == "petri_pooled"]
    assert windows and all(w == 16384 for w in windows)
    assert not any(isinstance(w, float) and math.isnan(w) for w in windows)


def test_a_sidecar_naming_no_root_or_two_is_refused(tmp_path):
    fx = F.build_llama8b(str(tmp_path / "none"))
    _rewrite_keys(fx, lambda key: key.replace(CAMPAIGN + "/", "/elsewhere/"))
    with pytest.raises(AdapterRefusal) as e:
        LW.Llama8bWaveAdapter(stage=fx["stage"], mmlu_cache=fx["cache"]).long_rows()
    assert e.value.path == _sidecar_path(fx) and "0 wave roots" in e.value.why
    fx = F.build_llama8b(str(tmp_path / "two"))
    base = _fixture_root(fx)
    with open(_sidecar_path(fx), encoding="utf-8") as fh:
        stores = list(json.load(fh)["store_dirs"])
    moved = stores[0]
    _rewrite_keys(fx, lambda key: _under_other_account(base) + key[len(base):] if key == moved else key)
    with pytest.raises(AdapterRefusal) as e:
        LW.Llama8bWaveAdapter(stage=fx["stage"], mmlu_cache=fx["cache"]).long_rows()
    assert "2 wave roots" in e.value.why
