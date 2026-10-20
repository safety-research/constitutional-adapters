"""The seed ledger — one row per (view, eval_id, cell, cap, trial), every claimed seed
RECOMPUTED from the leg's own derivation, the row refused when it does not reproduce, the MMLU rule corrected FROM EVIDENCE
when the wire carries no seed key, the engine-resolved sampler record carried, never invented. Synthetic stores throughout:
rows built with the instruments' own derivations (never re-typed constants), placeholder ids, a scratch tree."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts" / "sdfbase" / "seedrep"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import eval_seed as es  # noqa: E402
import seed_ledger as sl  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.eval.attacks import bon as bon_mod  # noqa: E402
from sisteer.eval.attacks.static_common import static_sampling_seed  # noqa: E402
from sisteer.eval.sampled_controls import META_SUFFIX, sampling_seed  # noqa: E402

CELL = "constv2-mix-full__psr_ce_all"
B1 = es.seed_base(1)


def _w(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _sampler(d: Path, verdict="AGREES"):
    d.mkdir(parents=True, exist_ok=True)
    (d / sl.RESOLVED_SAMPLER_FILENAME).write_text(json.dumps({"n": 5, "verdict": verdict, "reference_source": "card"}))


def _static_store(root: Path, *, base=B1, trial=0, family="pasttense", corrupt=False) -> Path:
    d = root / f"static_{family}_t{trial}"
    rows = []
    for g in ("g0", "g1"):
        for k in range(2):
            seed = static_sampling_seed(base, trial, family, g, k)
            rows.append({"goal_id": g, "sample_idx": k, "sampling_seed": seed + (1 if corrupt and g == "g1" and k == 1 else 0),
                         "response": "r", "decode": {"do_sample": True, "temperature": 0.7, "cap": 4096, "arm": "vendor_recommended", "seed_base": base, "trial": trial}})
    _w(d / "shard00" / "responses.jsonl", rows)
    _sampler(d)
    return d


def _controls_store(root: Path, *, base=B1, trial=1, control="xstest") -> Path:
    d = root / f"controls_{control}_t{trial}"
    rows = [{"item_id": f"xst{i:04d}", "sampling_seed": sampling_seed(base, trial, control, f"xst{i:04d}"), "max_tokens": 512, "finish_reason": "stop"} for i in range(3)]
    _w(d / "private" / f"gens_{control}{META_SUFFIX}", rows)
    _sampler(d)
    return d


def _bon_store(root: Path, *, base=B1, wrong_base=None) -> Path:
    d = root / "bon"
    rows = []
    for g in ("g0", "g1"):
        for k in range(2):
            rows.append({"goal_id": g, "sample_idx": k, "trial": 0, "variant": "v", "response": "r", "engine": "vllm_served", "serving_engine": "lora_native",
                         "cell_id": CELL, "bon_max_concurrency": 4, "finish_reason": "stop", "decode": {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "presence_penalty": 0.0},
                         "bon_decode_preset": "anchor_match", "sampling_seed": bon_mod.bon_decode_seed_v1(g, k, 0, base),
                         "seed_rule": bon_mod.seed_rule_for_policy("replicate_v1"), "decode_seed_policy": "replicate_v1",
                         "decode_seed_base": wrong_base if wrong_base is not None else base, "augment_seed": bon_mod.bon_seed(g, k, 0)})
    _w(d / "bon_responses.jsonl", rows)
    (d / "bon_served_run.json").write_text(json.dumps({"decode_seed_policy": "replicate_v1", "decode_seed_base": base, "bon_decode_preset": "anchor_match"}))
    _sampler(d)
    return d


def _evidence_store(root: Path, name: str, records) -> Path:
    d = root / name
    _w(d / sl.REQUEST_EVIDENCE_FILENAME, records)
    _sampler(d)
    return d


def _entry(view, eval_id, store, *, cap, trial=None, **kw):
    return {"view": view, "eval_id": eval_id, "cell": CELL, "cap": cap, "trial": trial, "store_dir": str(store), **kw}


def test_every_leg_kind_is_read_against_its_own_derivation_and_the_row_carries_the_rule(tmp_path):
    g8k = _evidence_store(tmp_path, "gsm8k", [{"kind": "task", "item_id": f"gsm8k_{i}", "sample_idx": 0, "seed": C.inspect_replicate_seed(f"gsm8k_{i}", 0, B1)} for i in range(3)])
    mmlu = _evidence_store(tmp_path, "mmlu", [{"kind": "task", "item_id": f"mmlu_{i}", "sample_idx": 0, "seed": 0} for i in range(4)])
    mt = _evidence_store(tmp_path, "crescendo", [{"kind": "task", "item_id": None, "sample_idx": None} for _ in range(2)])
    # decisions O12a: the GSM8K paired-200 matrix's natively served `none` ANCHOR row of the view — the lane's ONE unsteered
    # measurement — drawn under the SAME view base as the cells (its store is what the cells pair against), labelled by role
    g8k_anchor = _evidence_store(tmp_path, "gsm8k_none", [{"kind": "task", "item_id": f"gsm8k_{i}", "sample_idx": 0, "seed": C.inspect_replicate_seed(f"gsm8k_{i}", 0, B1)} for i in range(3)])
    entries = [_entry("s1", "pasttense", _static_store(tmp_path), cap=4096, trial=0),
               _entry("s1", "xstest", _controls_store(tmp_path), cap=512, trial=1),
               _entry("s1", "bon", _bon_store(tmp_path), cap=4096, trial=0, anchor_sampler_verdict="comparable"),
               _entry("s1", "gsm8k", g8k, cap=8192),
               _entry("s1", "mmlu", mmlu, cap=1),
               _entry("s1", "crescendo", mt, cap=4096),
               {**_entry("s1", "gsm8k", g8k_anchor, cap=8192), "cell": "none"}]
    rows = sl.build(entries)
    assert [r["status"] for r in rows] == ["ok"] * 7 and all(r["problems"] == [] for r in rows)
    anchor = rows.pop()
    assert anchor["cell"] == "none" and anchor["role"] == sl.ROLE_NATIVE_ANCHOR == "native_anchor" and anchor["seed_rule"] == "replicate_seed_base"
    assert anchor["seed_base"] == B1 and anchor["decode_seed_policy"] == "replicate" and anchor["n_rows_checked"] == 3 and anchor["seeds_reproduce"] is True
    assert all(r["role"] == sl.ROLE_STEERED == "steered" for r in rows)
    by = {r["eval_id"]: r for r in rows}
    assert by["pasttense"] == {**by["pasttense"], "seed_rule": "replicate_seed_base", "seed_base": B1, "seed_field": "sampling_seed", "n_rows_checked": 4,
                               "seeds_reproduce": True, "decode_seed_policy": None, "tag": "qwen35_4b_s1", "seed": 1, "trial": 0, "cap": 4096}
    assert by["xstest"]["seed_rule"] == "replicate_seed_base" and by["xstest"]["n_rows_checked"] == 3 and by["xstest"]["seed_base"] == B1
    assert by["bon"]["seed_rule"] == "replicate_seed_base" and by["bon"]["decode_seed_policy"] == "replicate_v1" and by["bon"]["bon_decode_preset"] == "anchor_match"
    assert by["bon"]["anchor_sampler_verdict"] == "comparable" and by["bon"]["n_rows_checked"] == 4
    assert by["gsm8k"]["seed_rule"] == "replicate_seed_base" and by["gsm8k"]["decode_seed_policy"] == "replicate" and by["gsm8k"]["seed_field"] == "seed" and by["gsm8k"]["n_rows_checked"] == 3
    assert by["mmlu"]["seed_rule"] == "instrument_frozen" and by["mmlu"]["seed_base"] is None and by["mmlu"]["frozen_seed"] == 0 and by["mmlu"]["seed_field"] == "seed"
    assert by["crescendo"]["seed_rule"] == "none_sent" and by["crescendo"]["seed_field"] is None and by["crescendo"]["seed_base"] is None and by["crescendo"]["anchor_sampler_verdict"] is None
    assert all(r["resolved_sampler"]["verdict"] == "AGREES" and r["resolved_sampler"]["path"].endswith(sl.RESOLVED_SAMPLER_FILENAME) for r in rows)
    assert all(r["seed_mechanism"] == es.leg_seed_rule(r["eval_id"]).mechanism for r in rows)


def test_a_seed_that_does_not_reproduce_a_wrong_base_or_a_seed_on_a_none_sent_leg_refuses(tmp_path):
    bad_static = _static_store(tmp_path / "a", corrupt=True)
    other_base = _static_store(tmp_path / "b", base=es.seed_base(2))              # a store drawn under s2's base read as s1
    bon_wrong = _bon_store(tmp_path / "c", wrong_base=es.seed_base(3))
    g8k_wrong = _evidence_store(tmp_path / "d", "gsm8k", [{"kind": "task", "item_id": "gsm8k_1", "sample_idx": 0, "seed": C.inspect_replicate_seed("gsm8k_1", 0, es.seed_base(2))}])
    mt_seeded = _evidence_store(tmp_path / "e", "pair", [{"kind": "task", "seed": 7}])
    mmlu_mixed = _evidence_store(tmp_path / "f", "mmlu", [{"kind": "task", "seed": 0}, {"kind": "task", "seed": 3}])
    rows = sl.build([_entry("s1", "pasttense", bad_static, cap=4096, trial=0), _entry("s1", "pasttense", other_base, cap=4096, trial=0),
                     _entry("s1", "bon", bon_wrong, cap=4096, trial=0), _entry("s1", "gsm8k", g8k_wrong, cap=8192),
                     _entry("s1", "pair", mt_seeded, cap=4096), _entry("s1", "mmlu", mmlu_mixed, cap=1),
                     _entry("s1", "codeattack", tmp_path / "nowhere", cap=4096, trial=0)])
    assert all(r["status"] == "refused" for r in rows)
    needles = ["records sampling_seed", "decode block records seed_base", "records policy 'replicate_v1' / base", "inspect_replicate_seed(item, sample",
               "carries a seed on 1 record", "carry a seed other than 0", "no such store dir"]
    for r, needle in zip(rows, needles):
        assert any(needle in p for p in r["problems"]), (r["eval_id"], needle, r["problems"])
    # the trial of a sampled leg is not optional
    assert sl.build([_entry("s1", "grok420", _static_store(tmp_path / "g", family="grok420"), cap=4096)])[0]["problems"] == ["grok420: a sampled leg's entry names its trial"]


def test_mmlu_flips_to_none_sent_from_evidence_and_missing_evidence_is_reported_not_guessed(tmp_path):
    no_seed_key = _evidence_store(tmp_path, "mmlu", [{"kind": "task", "item_id": "m1"}, {"kind": "task", "item_id": "m2"}])
    row = sl.build([_entry("s2", "mmlu", no_seed_key, cap=1)])[0]
    assert row["status"] == "ok" and row["seed_rule"] == "none_sent" and row["seed_field"] is None and "corrected from evidence" in row["seed_mechanism"]
    # no evidence at all: the table's rule stands but the row is MISSING evidence, never ok
    bare = tmp_path / "bare"; bare.mkdir()
    row = sl.build([_entry("s2", "petri", bare, cap=4096)])[0]
    assert row["status"] == "missing_evidence" and row["seed_rule"] == "none_sent" and row["resolved_sampler"] is None
    assert any("no request evidence" in p for p in row["problems"]) and any("resolved_sampler.json" in p for p in row["problems"])
    # a sampler record in the entry's own path, or a parent's, is found; the verdict is carried as recorded
    st = _static_store(tmp_path / "s", trial=1)
    (st / sl.RESOLVED_SAMPLER_FILENAME).unlink()
    row = sl.build([_entry("s1", "pasttense", st, cap=4096, trial=1)])[0]
    assert row["status"] == "missing_evidence" and row["seeds_reproduce"] is True
    _sampler(tmp_path / "s", verdict="DIVERGES")
    row = sl.build([_entry("s1", "pasttense", st, cap=4096, trial=1)])[0]
    assert row["status"] == "ok" and row["resolved_sampler"]["verdict"] == "DIVERGES"


def test_the_index_is_validated_and_the_cli_exit_codes_follow_the_rows(tmp_path):
    ok = _static_store(tmp_path, trial=0)
    bare = tmp_path / "bare"; bare.mkdir()
    idx = tmp_path / "index.json"
    idx.write_text(json.dumps([_entry("s1", "pasttense", ok, cap=4096, trial=0), _entry("s1", "petri", bare, cap=4096)]))
    py, env = sys.executable, {"PYTHONPATH": str(REPO / "src")}
    out = tmp_path / "RS" / "seed_ledger.jsonl"
    r = subprocess.run([py, str(REPO / "scripts/sdfbase/seedrep/seed_ledger.py"), "--index", str(idx), "--out", str(out)], capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 2 and "1 ok, 1 missing_evidence, 0 refused" in r.stdout and "MISSING_EVIDENCE s1 petri" in r.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r_["status"] for r_ in rows] == ["ok", "missing_evidence"]
    r = subprocess.run([py, str(REPO / "scripts/sdfbase/seedrep/seed_ledger.py"), "--index", str(idx), "--out", str(out), "--allow-missing-evidence"],
                       capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 0
    bad = tmp_path / "index_bad.jsonl"
    bad.write_text(json.dumps(_entry("s1", "pasttense", _static_store(tmp_path / "x", corrupt=True), cap=4096, trial=0)) + "\n")
    r = subprocess.run([py, str(REPO / "scripts/sdfbase/seedrep/seed_ledger.py"), "--index", str(bad), "--out", str(out), "--allow-missing-evidence"],
                       capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 1 and "REFUSED s1 pasttense" in r.stderr
    # the index grammar: a foreign view, an unknown leg, `none` as a cell (admitted ONLY as the GSM8K native anchor, O12a), a bad cap
    p_ok = tmp_path / "i_anchor.json"; p_ok.write_text(json.dumps([{**_entry("s1", "gsm8k", ok, cap=8192), "cell": "none"}]))
    assert sl.load_index(p_ok)[0]["cell"] == "none"
    for entry, needle in ((_entry("main", "pasttense", ok, cap=4096, trial=0), "not a seed-replicate view"),
                          (_entry("s1", "wmdp", ok, cap=4096, trial=0), "not a leg of the lane"),
                          ({**_entry("s1", "pasttense", ok, cap=4096, trial=0), "cell": "none"}, "never `none`"),
                          ({**_entry("s1", "bon", ok, cap=4096, trial=0), "cell": "none"}, "never `none`"),
                          (_entry("s1", "pasttense", ok, cap=0, trial=0), "cap must be a positive int"),
                          ({k: v for k, v in _entry("s1", "pasttense", ok, cap=4096, trial=0).items() if k != "store_dir"}, "lacks")):
        p = tmp_path / "i.json"; p.write_text(json.dumps([entry]))
        with pytest.raises((SystemExit, ValueError), match=needle):
            sl.load_index(p)
