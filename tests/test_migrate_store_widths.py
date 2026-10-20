"""migrate_store_widths: re-keying pre-seam capability stores from their own
recorded widths (the --parity-n port's migration side).

Everything here is SYNTHETIC — invented questions, placeholder hashes —
mirroring the real store shape (manifest.json + rows.jsonl under
<root>/cells/<arm>__controls__t0__a0__<ihash8>__<chash8>)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import migrate_store_widths as mig  # noqa: E402
from sisteer.capeval.store import harness_config_hash  # noqa: E402
from sisteer.capeval.tiers.parity import PARITY_N  # noqa: E402

GS, MM = mig.PARITY_TASK_GSM8K, mig.PARITY_TASK_MMLU
FULL_G, FULL_M = mig.FULL_POOLS[GS], mig.FULL_POOLS[MM]


def _rc(limit=None, parity_n=None) -> dict:
    rc = {"driver": "cell_controls_run_v1", "cell_id": "none",
          "model_id": "test/model", "model_revision": "0" * 40,
          "gen": {"do_sample": False, "max_new_tokens": 512},
          "controls": ["gsm8k", "mmlu"], "order_seed": 0,
          "limit": limit, "scale": 1.0, "system_sha": None,
          "judge_model": "judge-model", "subset_seed": 0 if limit else None}
    if parity_n is not None:
        rc["parity_n"] = parity_n
    return rc


def _store(root: Path, rc: dict, widths: dict[str, int], *,
           chash: str | None = None, limit: int | None = None) -> Path:
    """A synthetic cell store: dir keyed by chash (default: the CORRECT
    hash of rc), rows.jsonl with `widths[task]` items per probe — under a
    limit, item INDICES spread over the instrument width while only
    `limit` rows exist, the way _seeded_subset leaves them."""
    chash = chash or harness_config_hash(rc)
    d = root / "cells" / f"none__controls__t0__a0__{'1' * 8}__{chash[:8]}"
    d.mkdir(parents=True)
    manifest = {"cell_id": "none", "intervention_hash": "1" * 16,
                "harness_config_hash": chash, "engine": "hf",
                "resolved_config": rc}
    (d / "manifest.json").write_text(json.dumps(manifest))
    with (d / "rows.jsonl").open("w") as fh:
        for task, prefix in ((GS, "g"), (MM, "m")):
            n = widths[task]
            take = range(n) if not limit else range(0, n, max(1, n // limit))
            for i in take:
                fh.write(json.dumps({"task": task,
                                     "item_id": f"{prefix}{i:04d}-aaaaaaaa",
                                     "metric": "acc", "value": 1.0}) + "\n")
        fh.write(json.dumps({"task": GS, "item_id": "__aggregate__",
                             "metric": "acc_aggregate", "value": 1.0}) + "\n")
    return d


def test_requested_width_inversion():
    assert mig._requested_width(PARITY_N, PARITY_N) is None  # the default
    assert mig._requested_width(FULL_G, FULL_M) == 0         # full pools
    assert mig._requested_width(400, 400) == 400
    # the scalar stays PREFERRED whenever one realizes the pair — the
    # per-task fallback must not re-key what the scalar seam already keys
    assert mig._requested_width(FULL_G, 1300) == 1300        # gsm8k capped
    # mixed instruments invert to the driver's per-task dict form,
    # non-default probes only, 0 = that probe's own full pool
    assert mig._requested_width(FULL_G, PARITY_N) == {"gsm8k": 0}
    assert mig._requested_width(PARITY_N, FULL_M) == {"mmlu": 0}
    assert mig._requested_width(400, PARITY_N) == {"gsm8k": 400}
    assert mig._requested_width(FULL_G, 400) == {"gsm8k": 0, "mmlu": 400}
    # uninvertible: a probe that never ran (its request is unknowable)
    # or that shows more rows than its pinned pool holds
    with pytest.raises(ValueError):
        mig._requested_width(FULL_G, 0)
    with pytest.raises(ValueError):
        mig._requested_width(0, FULL_M)
    with pytest.raises(ValueError):
        mig._requested_width(FULL_G + 1, PARITY_N)


def test_default_and_already_keyed_stores_are_untouched(tmp_path):
    ok_default = _store(tmp_path / "a", _rc(limit=100),
                        {GS: 100, MM: 100}, limit=100)
    ok_widened = _store(tmp_path / "b", _rc(parity_n=0),
                        {GS: FULL_G, MM: FULL_M})
    assert mig.classify(ok_default)["status"] == "OK"
    assert mig.classify(ok_widened)["status"] == "OK"


def test_miskeyed_widened_store_moves_and_verifies(tmp_path):
    # the bug's shape: full-pool rows keyed WITHOUT the width
    wrong = _store(tmp_path, _rc(), {GS: FULL_G, MM: FULL_M})
    v = mig.classify(wrong)
    assert v["status"] == "MOVE" and v["requested"] == 0
    right_rc = _rc(parity_n=0)
    assert v["expected_chash"] == harness_config_hash(right_rc)

    dst = mig.apply_move(v)
    mig.verify(dst)
    assert not wrong.exists() and dst.name == v["expected_name"]
    m = json.loads((dst / "manifest.json").read_text())
    assert m["resolved_config"] == right_rc
    assert m["harness_config_hash"] == v["expected_chash"]
    assert m["parity_n"]["realized"] == {"gsm8k": FULL_G, "mmlu": FULL_M}
    # reversible: original manifest preserved, inverse mapping recorded
    assert json.loads((dst / "manifest.json.premigration").read_text())[
        "resolved_config"] == _rc()
    inverse = json.loads((tmp_path / "width_migration_inverse.json")
                         .read_text())
    assert inverse == {str(dst): str(wrong)}
    assert mig.classify(dst)["status"] == "OK"  # idempotent


def test_mixed_width_store_moves_to_the_per_task_key(tmp_path):
    """gsm8k widened alone (old --n-gsm8k, the lora_r1_all lane's shape:
    gsm8k at the full pool, mmlu at the default). No SCALAR request
    realizes it, but the per-task seam records exactly this instrument,
    so the store re-keys onto the dict form instead of sitting
    unrepresentable on its default-width twin's key."""
    mixed = _store(tmp_path, _rc(), {GS: FULL_G, MM: PARITY_N})
    # the move is copy-then-verify-then-remove, so EVERY file must
    # travel byte-identically — including nested ones (judged spools)
    spool = mixed / "spools" / "xstest.jsonl"
    spool.parent.mkdir()
    spool.write_text('{"item_id": "x0000-aaaaaaaa"}\n')
    v = mig.classify(mixed)
    assert v["status"] == "MOVE" and v["requested"] == {"gsm8k": 0}
    right_rc = _rc(parity_n={"gsm8k": 0})
    assert v["expected_chash"] == harness_config_hash(right_rc)

    dst = mig.apply_move(v)
    mig.verify(dst)
    assert not mixed.exists() and dst.name == v["expected_name"]
    m = json.loads((dst / "manifest.json").read_text())
    assert m["resolved_config"] == right_rc
    assert m["parity_n"]["requested"] == {"gsm8k": 0}
    assert m["parity_n"]["realized"] == {"gsm8k": FULL_G, "mmlu": PARITY_N}
    assert (dst / "spools" / "xstest.jsonl").read_text() == (
        '{"item_id": "x0000-aaaaaaaa"}\n')
    assert json.loads((dst / "manifest.json.premigration").read_text())[
        "resolved_config"] == _rc()
    inverse = json.loads((tmp_path / "width_migration_inverse.json")
                         .read_text())
    assert inverse == {str(dst): str(mixed)}
    assert mig.classify(dst)["status"] == "OK"  # idempotent
    # a store already carrying a premigration manifest is a prior
    # migration's product; a second move would bury its provenance chain
    again = mig.classify(dst)
    again.update(status="MOVE", expected_name="x" * 4,
                 expected_rc=right_rc, expected_chash="f" * 16,
                 widths=(FULL_G, PARITY_N), requested={"gsm8k": 0})
    with pytest.raises(SystemExit, match="premigration"):
        mig.apply_move(again)


def test_uninvertible_and_wide_subset_stores_are_reported_not_moved(
        tmp_path):
    # one probe never ran: its requested width is unknowable, so the
    # store is reported rather than bound to a guessed key
    partial = _store(tmp_path / "a", _rc(), {GS: FULL_G, MM: 0})
    assert mig.classify(partial)["status"] == "MIXED"
    # a --limit subset drawn from a WIDER pool: width not derivable
    wide_sub = _store(tmp_path / "b", _rc(limit=100),
                      {GS: FULL_G, MM: FULL_M}, limit=100)
    assert mig.classify(wide_sub)["status"] == "WIDE-SUBSET"
    # a parity-free judged-only store needs nothing
    empty = _store(tmp_path / "c", _rc(), {GS: 0, MM: 0})
    assert mig.classify(empty)["status"] == "NOPARITY"


def test_non_width_key_drift_is_reported_never_repaired(tmp_path):
    """A store whose stored resolved config no longer hashes to its own
    key drifted for a NON-width reason (a deliberate post-hoc manifest
    edit, e.g. the judge-switch annotations). The width migration must
    report DRIFT and refuse to move it — default-width or not."""
    rc = _rc(limit=100)
    keyed = harness_config_hash(rc)
    rc_edited = dict(rc, judge_model="another-judge")  # post-hoc edit
    drifted = _store(tmp_path / "a", rc_edited, {GS: 100, MM: 100},
                     chash=keyed, limit=100)
    assert mig.classify(drifted)["status"] == "DRIFT"
    # even a WIDENED store is unmovable once its base key drifted
    rc_full = _rc()
    drifted_wide = _store(tmp_path / "b",
                          dict(rc_full, judge_model="another-judge"),
                          {GS: FULL_G, MM: FULL_M},
                          chash=harness_config_hash(rc_full))
    assert mig.classify(drifted_wide)["status"] == "DRIFT"
