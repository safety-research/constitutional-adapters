"""The decode-evidence reader (``sisteer.registry.decode_evidence``; decode-recording lane, the registry owner's ruling of 2026-10-13):
one stdlib reader for the store recorder, the full22 closer and the backfill. Synthetic stores per family in ``tmp_path``; every
literal the module spells is pinned to its owner; the contradiction and shape cases refuse by name, the absence cases return ``why``;
the BoN row scan is pinned equal to ``bon_climb.store_sampler``; the remote read pass runs as a bare ``python3 -I`` script; the
write-time ``verify_against`` comparison sees every drift class."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.registry import decode_evidence as DE
from sisteer.registry import decode_match as DM
from sisteer.registry import decode_policy as DP
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import validate as VAL
from sisteer.registry import synth

REPO = Path(__file__).resolve().parents[1]
SIX = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 1.5}
SIX_F = {k: (int(v) if k == "top_k" else float(v)) for k, v in SIX.items()}
STATIC_BLOCK = dict(SIX, do_sample=True, cap=4096, arm="vendor_recommended", seed_base=20260909, trial=0)
T1 = {"temperature": 1.0}
BON_POLICY = "synthmodel|bon|off|grandfathered_T1@v1"
MT_POLICY = "synthmodel|multi_turn|off|vendor_preset@v1"


# --------------------------------------------------------------------------------------------
# the literals, pinned to their owners
# --------------------------------------------------------------------------------------------


def test_the_six_keys_the_spellings_and_the_manifest_key_are_their_owners():
    from sisteer.capeval.tinycap import contracts as tc
    from sisteer.eval.attacks import bon_served as bs
    from sisteer.registry.adapters import cellstore as CS
    sys.path.insert(0, str(REPO / "scripts"))
    import served_capability_run as srun  # noqa: E402
    assert DE.PARAM_KEYS == DP.PARAM_KEYS == DM.PARAMS
    assert str(tuple(DE.PARAM_KEYS)) in S.VIEW_SQL[DM.MATCH_SQL_NAME]            # the matcher's `_params` literal
    assert set(DE.PARAM_KEYS) == set(bs.SAMPLER_KEYS)
    assert DE.SAMPLED_CONTROL_SPELLINGS == srun.SAMPLED_CONTROLS
    assert set(DE.LEGACY_CONTROL_SPELLINGS) == set(srun.SERVED_CONTROLS) == {"mmlu", "xstest", "or_bench_hard_1k"}
    # the greedy body's one key (Major 3): the driver's constant, spread into `_chat_payload`'s greedy body
    assert DE.GREEDY_CHAT_SAMPLER == srun.GREEDY_CHAT_SAMPLER == srun.MMLU_SCORING_SAMPLER == {"temperature": 0.0}
    greedy_body = srun._chat_payload("vm", "Q", system=None, max_tokens=512)
    assert {k: greedy_body[k] for k in srun.SAMPLED_DECODE_SAMPLER_KEYS if k in greedy_body} == DE.GREEDY_CHAT_SAMPLER
    assert DE.MANIFEST_KEY_REQUEST_SAMPLERS == tc.MANIFEST_KEY_REQUEST_SAMPLERS == "request_samplers"
    assert DE.recorder_path_for("/r/x/cells/c") == CS.recorder_path_for("/r/x/cells/c") and DE.RECORDER_FILE == CS.RECORDER_FILE
    assert set(DE.TYPED_COLUMNS.values()) <= set(R.RUN_PATCHABLE_FIELDS) and {"decode_sent_json", "decode_resolved_json"} <= set(R.RUN_PATCHABLE_FIELDS)
    # the pinned request builder a DECLARED recorder reference derives against (the owner's Change 1, 2026-10-13 18:57 PT): the tinycap
    # provider's one place a request's sampling parameters are resolved -- the path is in the tree and defines the function
    b_path, b_fn = DE.RECORDER_REQUEST_BUILDER.split(" ")
    assert (REPO / b_path).is_file() and f"def {b_fn}(" in (REPO / b_path).read_text(), DE.RECORDER_REQUEST_BUILDER
    assert (DE.SENT_RECORDED, DE.SENT_DERIVED) == ("recorded", "derived") and DE.derived_entry("s") == {"confidence": "derived", "source": "s"}


def test_the_module_is_stdlib_only_and_runs_as_a_bare_script(tmp_path):
    src = (REPO / "src" / "sisteer" / "registry" / "decode_evidence.py").read_text()
    assert "import sisteer" not in src and "from sisteer" not in src and "from ." not in src.split('"""', 2)[2]
    store = _bon_store(tmp_path / "bon")
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps([{"run_id": "r1", "family": "bon", "store_uri": str(store)}, {"run_id": "r2", "family": "static", "store_uri": str(tmp_path / "absent")}]))
    r = subprocess.run([sys.executable, "-I", str(REPO / "src" / "sisteer" / "registry" / "decode_evidence.py"), "--runs", str(runs), "--json"],
                       capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["reader"] == DE.READER and out["n"] == 2
    by = {x["run_id"]: x for x in out["results"]}
    assert by["r1"]["ok"] and by["r1"]["evidence"]["sent"] == T1 and by["r1"]["sha256_read2"] == by["r1"]["evidence"]["sha256"]
    assert by["r2"]["ok"] and by["r2"]["evidence"]["why"] and "no shard directory" in by["r2"]["evidence"]["why"]
    # the backfill's transport: the module's SOURCE on stdin under the library switch, a trailer calling remote_main -- the CLI stays quiet
    trailer = f"\nimport json as _j\nprint('@@JSON'); print(_j.dumps(remote_main({json.dumps(json.loads(runs.read_text()))})))\n"
    r2 = subprocess.run([sys.executable, "-I", "-"], input=src + trailer, capture_output=True, text=True, timeout=120, cwd=str(tmp_path),
                        env={**os.environ, DE.LIBRARY_ENV: "1"})
    assert r2.returncode == 0 and r2.stdout.startswith("@@JSON"), r2.stderr
    assert json.loads(r2.stdout.split("@@JSON", 1)[1])["n"] == 2
    r3 = subprocess.run([sys.executable, "-I", "-"], input=src, capture_output=True, text=True, timeout=120, cwd=str(tmp_path), env={k: v for k, v in os.environ.items() if k != DE.LIBRARY_ENV})
    assert r3.returncode == 2 and "--runs" in r3.stderr                                       # without the switch the CLI runs and asks for --runs


# --------------------------------------------------------------------------------------------
# fixtures: one store per family
# --------------------------------------------------------------------------------------------


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _bon_store(d: Path, *, decode=T1, preset="harness_default", policy=BON_POLICY, rows=6, row_decode=None, legacy_rows=0, with_decode=True):
    d.mkdir(parents=True, exist_ok=True)
    man = {"cell_id": "none", "units": {"n_planned": rows, "n_done": rows}, "bon_decode_preset": preset, "decode_policy_id": policy,
           "seed_rule": "bon_seed_v1", "n_legacy_unrecorded_rows": legacy_rows}
    if with_decode:
        man["decode"] = dict(decode)
    (d / DE.BON_MANIFEST).write_text(json.dumps(man))
    (d / DE.BON_RUN).write_text(json.dumps({"bon_decode_preset": preset, "bon_decode_preset_declared": "canonical"}))
    rd = row_decode if row_decode is not None else decode
    rr = [{"goal_id": f"g{i}", "sample_idx": 0, "trial": 0, "response": "y", "decode": dict(rd), "bon_decode_preset": preset, "decode_policy_id": policy} for i in range(rows)]
    rr += [{"goal_id": f"l{i}", "sample_idx": 0, "trial": 0, "response": "y"} for i in range(legacy_rows)]
    _jsonl(d / DE.BON_RESPONSES, rr)
    return d


def _mt_store(d: Path, *, shards=("shard00",), rows=3, target_decode=SIX, served_decode=STATIC_BLOCK, source=MT_POLICY, with_td=True):
    for sh in shards:
        prog = [{"goal_id": f"{sh}_g{i}", "outcome": "failure", **({"target_decode": dict(target_decode)} if with_td else {})} for i in range(rows)]
        _jsonl(d / sh / "progress.jsonl", prog)
        _jsonl(d / sh / "responses.jsonl", [{"goal_id": f"{sh}_g{i}", "order": 0, "response": "r"} for i in range(rows)])
        st = {"serving_engine": "lora_native", "n_sampled_requests": rows, "target_decode": None}
        if served_decode is not None:
            st["decode"] = dict(served_decode)
        (d / sh / "served_target.json").write_text(json.dumps(st))
        (d / sh / "attacker_engine.json").write_text(json.dumps({"attack": "pair", "target_decode": {"mode": "vendor_preset", "source": source}}))
    return d


def _static_store(d: Path, *, shards=("shard00", "shard01"), rows=4, decode=STATIC_BLOCK, row_decode=None, with_decode=True):
    for sh in shards:
        _jsonl(d / sh / "progress.jsonl", [{"goal_id": f"{sh}_g{i}", "outcome": "done"} for i in range(rows)])
        rd = row_decode if row_decode is not None else decode
        _jsonl(d / sh / "responses.jsonl", [{"goal_id": f"{sh}_g{i}", "variant": 0, "response": "x", **({"decode": dict(rd)} if rd is not None else {})} for i in range(rows)])
        st = {"n_sampled_requests": rows}
        if with_decode:
            st["decode"] = dict(decode)
        (d / sh / "served_target.json").write_text(json.dumps(st))
    return d


def _controls_cell(d: Path, *, controls=("xstest", "or_bench_hard_1k"), gen=None, request_samplers=None, gen_stats=True, gs_decode=None):
    d.mkdir(parents=True, exist_ok=True)
    gen = dict(gen) if gen is not None else dict(SIX, arm="vendor_recommended", batch_size=8, do_sample=True, max_new_tokens=2048, seed_base=20260909, trial=1,
                                                 escalated_from={"cap": 512, "root": "/x"}, floor_sample={"k": 200})
    man = {"cell_id": "none", "controls": list(controls), "resolved_config": {"driver": "served_capability_run_v2", "gen": gen}}
    if request_samplers is not None:
        man[DE.MANIFEST_KEY_REQUEST_SAMPLERS] = request_samplers
    (d / "manifest.json").write_text(json.dumps(man))
    if gen_stats:
        blk = gs_decode if gs_decode is not None else dict(SIX, do_sample=True, cap=2048, arm="vendor_recommended", seed_base=20260909, trial=1)
        (d / "gen_stats.json").write_text(json.dumps({c: {"n": 3, "decode": dict(blk)} for c in controls if c in DE.SAMPLED_CONTROL_SPELLINGS}))
    return d, man


def _misalign_cell(d: Path, *, sampling=T1, params_sampling=T1, with_sampling=True):
    d.mkdir(parents=True, exist_ok=True)
    meta = {"cell_id": "none", "items": 2, "n_samples": 1, "finished": 1791729094.8, "sampler": "t1", "params": {"max_tokens": 16384, "temperature": 1.0, "sampler": "t1"}}
    if with_sampling:
        meta["sampling"] = dict(sampling)
    if params_sampling is not None:
        meta["params"]["sampling"] = dict(params_sampling)
    (d / "meta.json").write_text(json.dumps(meta))
    _jsonl(d / "responses.jsonl", [{"item_id": "i1", "behavior": "murder", "sample_idx": 0}, {"item_id": "i2", "behavior": "leaking", "sample_idx": 0}])
    return d


RESOLVED6 = {"min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0, "temperature": 0.6, "top_k": 20, "top_p": 0.95}
CARD = {"min_p": 0.0, "presence_penalty": 0.0, "temperature": 0.6, "top_k": 20.0, "top_p": 0.95}
SENT_CARD = {"min_p": 0.0, "presence_penalty": 0.0, "temperature": 0.6, "top_k": 20, "top_p": 0.95}
SHA = synth.code_sha("r")                  # the run's code sha: the pin a declared recorder reference derives against


def _recorder(root: Path, verdict="AGREES", *, samplers=(RESOLVED6,), reference=CARD, reference_source="--expect", rows_sent=None, n_rows=0, tasks=None):
    client = {"n": 200, "verdict": verdict, "distinct_resolved": [{"sampler": dict(s), "n": 200 // len(samplers)} for s in samplers],
              "reference": dict(reference) if reference else None, "reference_source": reference_source}
    if tasks is not None:                      # the recorder's --expect-tasks shape: one reference per task, none at the class level
        client.update(tasks={t: dict(b) for t, b in tasks.items()}, reference=None, reference_source="--expect-tasks: one reference per task (classes.client.tasks)")
    doc = {"verdict": verdict, "reference": dict(reference) if reference else None, "reference_source": reference_source,
           "sent": {"n_rows": n_rows, "distinct": [{"sampler": dict(s), "n": n_rows} for s in (rows_sent or [])]}, "classes": {"client": client}}
    root.mkdir(parents=True, exist_ok=True)
    (root / DE.RECORDER_FILE).write_text(json.dumps(doc))
    cell = root / "cells" / "none__controls__t0__a0__deadbeef__cafef00d"; cell.mkdir(parents=True, exist_ok=True)
    return cell


# --------------------------------------------------------------------------------------------
# recorded: the six keys and nothing else, per family
# --------------------------------------------------------------------------------------------


def test_bon_files_the_manifests_decode_corroborated_by_the_rows_and_carries_the_producers_policy_as_a_statement(tmp_path):
    from sisteer.eval.attacks import bon_climb as BC
    store = _bon_store(tmp_path / "bon", rows=7)
    ev = DE.from_bon_store(str(store))
    assert ev.recorded() and ev.sent == T1 and ev.resolved == {} and ev.family == "bon"
    assert ev.source == f"{store / DE.BON_MANIFEST}#decode" and ev.sha256 == DE.sha256_file(str(store / DE.BON_MANIFEST))
    assert ev.producer_policy_id == BON_POLICY and ev.producer_policy_id_source == f"{DE.BON_MANIFEST}#decode_policy_id"
    w = ev.witnesses
    assert (w["rows_scanned"], w["rows_recorded"], w["rows_legacy"], w["distinct_row_blocks"]) == (7, 7, 0, 1) and w["bon_decode_preset"] == "harness_default"
    assert w["bon_decode_preset_declared"] == "canonical" and w["manifest_decode"] == T1 and "7/7" in w["summary"]
    # pinned to bon_climb.store_sampler's scan on the same store
    assert BC.store_sampler(store) == {"decode": T1, "bon_decode_preset": "harness_default", "decode_policy_id": BON_POLICY}
    cols = ev.columns()
    assert cols == {"decode_sent_json": T1, "decode_temperature_sent": 1.0}                            # top_p / top_k not sent: no typed column
    assert set(ev.confidence_entries()) == {"decode_sent_json", "decode_temperature_sent"} and ev.confidence_entries()["decode_sent_json"]["confidence"] == "recorded"
    assert "sha256 " + ev.sha256[:12] in ev.confidence_entries()["decode_sent_json"]["source"]
    # an all-no-fit store (no response row): the manifest states the sampler, the witnesses say no row
    nofit = _bon_store(tmp_path / "nofit", rows=0)
    ev2 = DE.from_bon_store(str(nofit))
    assert ev2.recorded() and ev2.sent == T1 and ev2.witnesses["rows_scanned"] == 0 and "no bon_responses.jsonl row" in ev2.witnesses["summary"]


def test_multiturn_files_the_progress_rows_one_target_decode_block_with_served_target_agreeing(tmp_path):
    store = _mt_store(tmp_path / "mt", shards=("shard00", "shard03"), rows=3)
    ev = DE.from_multiturn_store(str(store))
    assert ev.recorded() and ev.sent == SIX_F and ev.family == "multiturn"
    assert ev.evidence_path == str(store / "shard00" / "progress.jsonl") and ev.source.endswith("progress.jsonl#target_decode")
    w = ev.witnesses
    assert w["shards"] == ["shard00", "shard03"] and w["rows_with_target_decode"] == 6 and w["distinct_blocks"] == 1 and w["served_target_decode_agrees"] is True
    assert ev.producer_policy_id == MT_POLICY and ev.producer_policy_id_source.endswith("attacker_engine.json#target_decode.source")
    assert ev.unmodelled == {"shard00/served_target.json#decode": {"do_sample": True, "cap": 4096, "arm": "vendor_recommended", "seed_base": 20260909, "trial": 0}}
    assert ev.columns() == {"decode_sent_json": SIX_F, "decode_temperature_sent": 0.7, "decode_top_p_sent": 0.8, "decode_top_k_sent": 20}
    assert isinstance(ev.columns()["decode_top_k_sent"], int)
    # the inspected subset: only the named shards are read
    ev2 = DE.from_multiturn_store(str(store), shards=["shard03"])
    assert ev2.witnesses["shards"] == ["shard03"] and ev2.witnesses["rows_with_target_decode"] == 3 and ev2.evidence_path.endswith("shard03/progress.jsonl")


def test_static_files_served_targets_block_with_every_response_row_agreeing_and_lists_the_unmodelled_keys(tmp_path):
    store = _static_store(tmp_path / "st", shards=("shard00", "shard01", "shard02"), rows=5)
    ev = DE.from_static_store(str(store))
    assert ev.recorded() and ev.sent == SIX_F and ev.family == "static"
    assert ev.evidence_path == str(store / "shard00" / "served_target.json") and ev.source.endswith("served_target.json#decode")
    w = ev.witnesses
    assert (w["shards"], w["served_target_blocks_distinct"], w["response_rows"], w["distinct_row_blocks"], w["response_rows_without_decode"]) == (3, 1, 15, 1, 0)
    assert sorted(ev.unmodelled["shard00/served_target.json#decode"]) == ["arm", "cap", "do_sample", "seed_base", "trial"]
    assert ev.extra_block()["unmodelled_keys"] == {"shard00/served_target.json#decode": ["arm", "cap", "do_sample", "seed_base", "trial"]}
    assert DE.from_static_store(str(store), family="padding").family == "padding"


def test_controls_reads_the_routes_spelling_from_request_samplers_first_then_the_sampled_gen_and_never_for_mmlu_or_an_atom(tmp_path):
    d, man = _controls_cell(tmp_path / "ctl")
    # a LEGACY manifest (no request_samplers): the sampled control's sent set is gen's six keys, gen_stats agrees
    ev = DE.from_store_manifest(man, "xstest", store_dir=str(d))
    assert ev.recorded() and ev.sent == SIX_F and ev.source == f"{d / 'manifest.json'}#resolved_config.gen" and ev.sha256 == DE.sha256_file(str(d / "manifest.json"))
    assert sorted(ev.unmodelled["resolved_config.gen"]) == ["arm", "batch_size", "do_sample", "escalated_from", "floor_sample", "max_new_tokens", "seed_base", "trial"]
    w = ev.witnesses
    assert w["gen_stats_decode_agrees"] is True and w["escalated_from_cap"] == 512 and w["manifest_controls"] == ["xstest", "or_bench_hard_1k"] and w["spelling"] == "xstest"
    assert DE.from_store_manifest(man, "or_bench_hard_1k", store_dir=str(d)).sent == SIX_F
    # the same legacy manifest read for mmlu: NEVER gen -- unverified by name
    ev_m = DE.from_store_manifest(dict(man, controls=["mmlu", "xstest", "or_bench_hard_1k"]), "mmlu", store_dir=str(d))
    assert not ev_m.recorded() and ev_m.sent == {} and "built by the driver" in ev_m.why and "request_samplers" in ev_m.why
    assert ev_m.confidence_entries() == {"decode_sent_json": {"confidence": "unverified", "reason": ev_m.why}} and ev_m.columns() == {} and ev_m.patch_fields() == {"field_confidence_json": ev_m.confidence_entries()}
    # the driver's statement (a″): request_samplers per spelling, mmlu's constant beside the sampled pair's six
    rs = {"mmlu": {"temperature": 0.0}, "xstest": dict(SIX), "or_bench_hard_1k": dict(SIX)}
    d2, man2 = _controls_cell(tmp_path / "ctl2", controls=("mmlu", "xstest", "or_bench_hard_1k"), request_samplers=rs)
    ev2 = DE.from_store_manifest(man2, "mmlu", store_dir=str(d2))
    assert ev2.recorded() and ev2.sent == {"temperature": 0.0} and ev2.source.endswith("manifest.json#request_samplers.mmlu") and ev2.columns() == {"decode_sent_json": {"temperature": 0.0}, "decode_temperature_sent": 0.0}
    ev3 = DE.from_store_manifest(man2, "xstest", store_dir=str(d2))
    assert ev3.recorded() and ev3.sent == SIX_F and ev3.source.endswith("#request_samplers.xstest") and ev3.witnesses["gen_stats_decode_agrees"] is True
    # an atom (gsm8k) on a manifest whose block names it not: unverified by name
    ev4 = DE.from_store_manifest(dict(man2, controls=["gsm8k"]), "gsm8k")
    assert not ev4.recorded() and "gsm8k request's sampler is built by the driver" in ev4.why
    # the greedy opt-in (Major 3): the greedy body SENDS temperature 0.0, so the driver states GREEDY_CHAT_SAMPLER and the
    # reader files it; a gen_stats GREEDY block ({do_sample: false, cap, arm}) corroborates exactly that set
    greedy_gen = {"batch_size": 8, "do_sample": False, "max_new_tokens": 512}
    greedy_rs = {"xstest": {"temperature": 0.0}, "or_bench_hard_1k": {"temperature": 0.0}}
    d3, man3 = _controls_cell(tmp_path / "ctl3", gen=greedy_gen, request_samplers=greedy_rs, gs_decode={"do_sample": False, "cap": 512, "arm": "greedy"})
    for sp in ("xstest", "or_bench_hard_1k"):
        ev5 = DE.from_store_manifest(man3, sp, store_dir=str(d3))
        assert ev5.recorded() and ev5.sent == {"temperature": 0.0} and ev5.source == f"{d3 / 'manifest.json'}#request_samplers.{sp}" and ev5.unmodelled == {}
        assert ev5.witnesses["gen_stats_decode_agrees"] is True and ev5.witnesses["gen_do_sample"] is False
        assert ev5.columns() == {"decode_sent_json": {"temperature": 0.0}, "decode_temperature_sent": 0.0}
    d3b, man3b = _controls_cell(tmp_path / "ctl3b", gen=greedy_gen, request_samplers=greedy_rs, gen_stats=False)       # no gen_stats yet (manifest time)
    ev5b = DE.from_store_manifest(man3b, "xstest", store_dir=str(d3b))
    assert ev5b.recorded() and ev5b.sent == {"temperature": 0.0} and ev5b.witnesses["gen_stats_decode_agrees"] is None
    # a statement of NO key ({}: no manifest of the driver writes it since Major 3) stays unverified by name; a LEGACY greedy manifest states no request
    d4, man4 = _controls_cell(tmp_path / "ctl4", gen=greedy_gen, request_samplers={"xstest": {}, "or_bench_hard_1k": {}}, gen_stats=False)
    assert "states no sampler key" in DE.from_store_manifest(man4, "xstest", store_dir=str(d4)).why
    assert "--greedy opt-in" in DE.from_store_manifest({k: v for k, v in man4.items() if k != DE.MANIFEST_KEY_REQUEST_SAMPLERS}, "xstest").why
    # no spelling: the caller's reason, else the controls list's
    assert DE.from_store_manifest(man, None, why_no_spelling="one identity, 2 controls").why == "one identity, 2 controls"
    assert "names 2 controls (xstest, or_bench_hard_1k) under one identity" in DE.from_store_manifest(man, None).why
    assert "names no controls list" in DE.from_store_manifest({"resolved_config": {"gen": {}}}, None).why
    # a flat spec with no resolved_config (a harness store that is not the served driver's): unverified, never an error
    assert "carries no resolved_config.gen" in DE.from_store_manifest({"sweep": "x", "kind": "lora"}, "xstest").why


def test_misalign_files_meta_sampling_with_params_sampling_agreeing(tmp_path):
    d = _misalign_cell(tmp_path / "mis")
    ev = DE.from_misalign_store(str(d))
    assert ev.recorded() and ev.sent == T1 and ev.source == f"{d / 'meta.json'}#sampling" and ev.witnesses["sampler"] == "t1" and ev.witnesses["finished"] is True
    assert ev.witnesses["params_sampling_agrees"] is True and ev.columns() == {"decode_sent_json": T1, "decode_temperature_sent": 1.0}
    ev2 = DE.from_misalign_store(str(_misalign_cell(tmp_path / "mis2", params_sampling=None)))
    assert ev2.recorded() and ev2.witnesses["params_sampling_agrees"] is None


def test_the_recorder_document_files_the_reference_keys_as_sent_and_the_six_as_resolved_27b(tmp_path):
    cell = _recorder(tmp_path / "root")
    ev = DE.from_recorder_document(str(cell), code_sha=SHA)
    assert ev.recorded() and ev.sent == SENT_CARD and isinstance(ev.sent["top_k"], int)
    assert ev.resolved == {"min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0, "temperature": 0.6, "top_k": 20, "top_p": 0.95}
    assert ev.source.endswith("resolved_sampler.json#classes.client.reference") and "verdict AGREES" in ev.resolved_source and ev.witnesses["reference_source"] == "--expect"
    cols = ev.columns()
    assert cols["decode_resolved_json"] == ev.resolved and cols["decode_sent_json"] == ev.sent and (cols["decode_temperature_sent"], cols["decode_top_p_sent"], cols["decode_top_k_sent"]) == (0.6, 0.95, 20)
    assert set(ev.confidence_entries()) == {"decode_sent_json", "decode_temperature_sent", "decode_top_p_sent", "decode_top_k_sent", "decode_resolved_json"}
    # the client rows' recorded sampler beats the reference -- an OBSERVATION, so no code sha is needed
    cell2 = _recorder(tmp_path / "root2", rows_sent=[{"temperature": 1.0, "top_p": 0.95, "top_k": 64}], n_rows=200, samplers=({"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0},))
    ev2 = DE.from_recorder_document(str(cell2))
    assert ev2.sent == {"temperature": 1.0, "top_p": 0.95, "top_k": 64} and ev2.source.endswith("#sent.distinct[0].sampler") and ev2.sent_confidence == DE.SENT_RECORDED
    # absence cases: no document; a non-agreeing verdict; a reference key that did not resolve; two resolved samplers
    assert "no recorder document" in DE.from_recorder_document(str(tmp_path / "nowhere" / "cells" / "c")).why
    assert "verdict DISAGREES" in DE.from_recorder_document(str(_recorder(tmp_path / "root3", "DISAGREES"))).why
    assert "did not resolve to their values" in DE.from_recorder_document(str(_recorder(tmp_path / "root4", reference={"temperature": 0.7, "top_p": 0.95})), code_sha=SHA).why
    assert "2 distinct client sampler(s)" in DE.from_recorder_document(str(_recorder(tmp_path / "root5", samplers=(RESOLVED6, dict(RESOLVED6, temperature=0.7)))), code_sha=SHA).why
    # a class-level document read for a task: the class is the answer (no per-task block to disagree with); the task is a witness
    ev_t = DE.from_recorder_document(str(cell), task="gsm8k", code_sha=SHA)
    assert ev_t.sent == ev.sent and ev_t.witnesses["task"] == "gsm8k"


def test_a_declared_recorder_reference_is_derived_against_the_pinned_builder_never_recorded(tmp_path):
    """The registry owner's Change 1 (2026-10-13 18:57 PT): a recorder whose client rows recorded NO per-request sampler (sent_rows 0) proves
    the sent set through its ``--expect`` REFERENCE alone -- a declaration of the keys the client builds its requests from, which cannot show
    which keys the client omitted (a ``min_p`` 0 or a ``presence_penalty`` 0 equals the engine's default). The sent columns are ``derived``,
    naming the pinned request builder at the run's code sha beside the document and its sha256; the resolved column stays ``recorded``;
    without a code sha the declaration has no pin and nothing is filed. Review round 2 (Major 1): round 1 filed this set as ``recorded``."""
    cell = _recorder(tmp_path / "root")
    ev = DE.from_recorder_document(str(cell), code_sha=SHA)
    assert ev.recorded() and ev.sent == SENT_CARD and ev.sent_confidence == DE.SENT_DERIVED and ev.witnesses["sent_rows"] == 0
    fc = ev.confidence_entries()
    for col in ("decode_sent_json", "decode_temperature_sent", "decode_top_p_sent", "decode_top_k_sent"):
        assert fc[col]["confidence"] == "derived", (col, fc[col])
        src = fc[col]["source"]
        assert src.startswith(str(ev.evidence_path)) and f"sha256 {ev.sha256[:12]}" in src and "--expect reference" in src and "verdict AGREES" in src
        assert DE.RECORDER_REQUEST_BUILDER in src and f"code sha {SHA}" in src and "DERIVED, not observed per request" in src and "sent_rows 0" in src
    assert fc["decode_resolved_json"]["confidence"] == "recorded" and "distinct_resolved[0]" in fc["decode_resolved_json"]["source"]
    assert ev.witnesses["request_builder"] == DE.RECORDER_REQUEST_BUILDER and ev.witnesses["request_builder_code_sha"] == SHA
    xb = ev.extra_block()
    assert xb["sent_confidence"] == "derived" and DE.RECORDER_REQUEST_BUILDER in xb["derivation"] and SHA in xb["derivation"]
    f = ev.patch_fields()
    assert f["decode_sent_json"] == SENT_CARD and f["decode_resolved_json"] == RESOLVED6
    assert f["field_confidence_json"]["decode_sent_json"]["confidence"] == "derived" and f["extra_json"][DE.EXTRA_KEY]["sent_confidence"] == "derived"
    rec = R.make_record("run_patch", {"run_id": "synthcampaign:syntheval:abcdefabcdef:0123456789abcdef", "fields": f}, synth.producer(), "11111111-2222-3333-4444-555555555558", synth.t(9))
    assert isinstance(VAL.validate_record(rec, "<memory>", R.record_sha(rec)), VAL.Validated)
    # no code sha (None, the producer block's `unverified`, a short or an upper-case string): the declaration has no pin -> NO sent column,
    # the reason names it -- and the server-resolved sampler the document OBSERVED is filed apart, `recorded`, with the declared keys kept
    # as a witness (Major 1: round 2 dropped the resolved observation with the sent set; the cellstore adapter kept it)
    for bad in (None, "unverified", "abc", "A" * 40):
        none = DE.from_recorder_document(str(cell), code_sha=bad)
        assert not none.recorded() and "DECLARATION" in none.why and "names no code sha" in none.why, (bad, none.why)
        assert none.resolved == RESOLVED6 and none.columns() == {"decode_resolved_json": RESOLVED6}, (bad, none.columns())
        assert none.resolved_read() and "nothing filed for the sent columns" in none.why and "recorded apart" in none.why, none.why
        assert none.confidence_entries() == {"decode_sent_json": DE.unverified_entry(none.why), "decode_resolved_json": DE.recorded_entry(none.resolved_source)}
        assert none.witnesses["unfiled_declared_reference"] == {"sent": SENT_CARD, "reference_source": "--expect", "sent_rows": 0, "request_builder": DE.RECORDER_REQUEST_BUILDER}
        fn = none.patch_fields()
        assert set(fn) == {"decode_resolved_json", "field_confidence_json", "extra_json"} and fn["decode_resolved_json"] == RESOLVED6
        xb_n = fn["extra_json"][DE.EXTRA_KEY]
        assert xb_n["sent_confidence"] is None and xb_n["derivation"] is None and xb_n["sha256"] == none.sha256 and xb_n["family"] == "recorder"
        assert xb_n["witnesses"]["unfiled_declared_reference"]["sent"] == SENT_CARD and xb_n["witnesses"]["sent_rows"] == 0
        rec_n = R.make_record("run_patch", {"run_id": "synthcampaign:syntheval:abcdefabcdef:0123456789abcdef", "fields": fn}, synth.producer(), "11111111-2222-3333-4444-555555555559", synth.t(9))
        assert isinstance(VAL.validate_record(rec_n, "<memory>", R.record_sha(rec_n)), VAL.Validated), bad
    # the per-task shape (--expect-tasks) derives the same way, task by task
    tb = {"caps": [4096], "n": 120, "verdict": "AGREES", "reference": dict(CARD), "reference_source": "thinking card", "distinct_resolved": [{"sampler": dict(RESOLVED6), "n": 120}]}
    cell_t = _recorder(tmp_path / "rt", tasks={"gsm8k": tb})
    ev_t = DE.from_recorder_document(str(cell_t), task="gsm8k", code_sha=SHA)
    assert ev_t.recorded() and ev_t.sent_confidence == DE.SENT_DERIVED and ev_t.sent == SENT_CARD
    assert "task gsm8k:" in ev_t.confidence_entries()["decode_sent_json"]["source"] and SHA in ev_t.derivation and ev_t.witnesses["request_builder_code_sha"] == SHA
    none_t = DE.from_recorder_document(str(cell_t), task="gsm8k")
    assert not none_t.recorded() and none_t.columns() == {"decode_resolved_json": RESOLVED6}
    assert none_t.witnesses["unfiled_declared_reference"] == {"sent": SENT_CARD, "reference_source": "thinking card", "sent_rows": 0, "request_builder": DE.RECORDER_REQUEST_BUILDER}
    # the client rows' recorded sampler is an OBSERVATION: `recorded`, no code sha needed, no derivation
    cell_r = _recorder(tmp_path / "rr", rows_sent=[dict(CARD)], n_rows=200)
    ev_r = DE.from_recorder_document(str(cell_r))
    assert ev_r.recorded() and ev_r.sent == SENT_CARD and ev_r.sent_confidence == DE.SENT_RECORDED and ev_r.derivation is None
    assert ev_r.confidence_entries()["decode_sent_json"]["confidence"] == "recorded" and ev_r.extra_block()["sent_confidence"] == "recorded"
    # for_family / read_run carry the run's code sha (the closer's / the backfill's registry row)
    assert DE.for_family("recorder", str(cell), code_sha=SHA).sent_confidence == DE.SENT_DERIVED
    frozen = DE.read_run({"run_id": "r", "family": "recorder", "store_uri": str(cell), "code_sha": SHA})
    assert frozen["ok"] and frozen["evidence"]["sent_confidence"] == "derived" and SHA in frozen["evidence"]["derivation"]
    assert not DE.read_run({"run_id": "r", "family": "recorder", "store_uri": str(cell)})["evidence"]["sent"]
    # the write-time comparison: client rows landing after the freeze with the SAME values turn the derived set into a recorded one -- the
    # sent set is unchanged, so only the confidence and its basis tell, and both are pinned
    _recorder(tmp_path / "root", rows_sent=[dict(CARD)], n_rows=200)
    fresh = DE.read_run({"run_id": "r", "family": "recorder", "store_uri": str(cell), "code_sha": SHA})
    assert fresh["evidence"]["sent"] == frozen["evidence"]["sent"] and fresh["evidence"]["sent_confidence"] == "recorded"
    d = DE.verify_against(frozen, fresh)
    assert any(x.startswith("sent_confidence: ") for x in d) and any(x.startswith("derivation: ") for x in d), d


def test_a_recorder_that_resolved_a_sampler_but_proves_no_sent_set_files_the_resolved_column_alone(tmp_path):
    """Review round 5 (Major 1): the server-resolved sampler is an OBSERVATION of the recorder document (one distinct resolved sampler under
    an AGREES verdict), independent of whether a sent set can be filed from the same document -- the columns, the entries and the patch
    carry it ``recorded`` beside the ``unverified`` sent entry naming why, exactly as the cellstore adapter files the same document; the
    declaration's keys ride the witnesses only when the pin was what was missing. A document that resolved nothing (a non-agreeing
    verdict, no document, a task it does not cover) files the unverified entry alone, as before."""
    tb = {"caps": [4096], "n": 120, "verdict": "AGREES", "reference": dict(CARD), "reference_source": "thinking card", "distinct_resolved": [{"sampler": dict(RESOLVED6), "n": 120}]}
    resolved_only = {
        "no pin": DE.from_recorder_document(str(_recorder(tmp_path / "p")), code_sha=None),
        "two client-row samplers": DE.from_recorder_document(str(_recorder(tmp_path / "t", rows_sent=[dict(CARD), dict(CARD, temperature=0.7)], n_rows=10)), code_sha=SHA),
        "a reference naming no decode key": DE.from_recorder_document(str(_recorder(tmp_path / "k", reference={"max_tokens": 5})), code_sha=SHA),
        "a reference that did not resolve": DE.from_recorder_document(str(_recorder(tmp_path / "d", reference={"temperature": 0.7, "top_p": 0.95})), code_sha=SHA),
        "neither rows nor a reference": DE.from_recorder_document(str(_recorder(tmp_path / "n", reference=None)), code_sha=SHA),
        "per-task, no pin": DE.from_recorder_document(str(_recorder(tmp_path / "pt", tasks={"gsm8k": tb})), task="gsm8k"),
    }
    for name, ev in resolved_only.items():
        assert not ev.recorded() and ev.why and ev.resolved == RESOLVED6 and "verdict AGREES" in ev.resolved_source, (name, ev.why)
        assert ev.columns() == {"decode_resolved_json": RESOLVED6}, (name, ev.columns())
        assert ev.resolved_read(), name
        assert ev.confidence_entries() == {"decode_sent_json": DE.unverified_entry(ev.why), "decode_resolved_json": DE.recorded_entry(ev.resolved_source)}, name
        f = ev.patch_fields()
        assert set(f) == {"decode_resolved_json", "field_confidence_json", "extra_json"} and set(f) <= set(R.RUN_PATCHABLE_FIELDS), name
        assert f["extra_json"][DE.EXTRA_KEY]["sent_confidence"] is None and f["extra_json"][DE.EXTRA_KEY]["evidence_path"] == ev.evidence_path, name
    assert resolved_only["no pin"].witnesses["unfiled_declared_reference"]["sent"] == SENT_CARD
    assert resolved_only["per-task, no pin"].witnesses["unfiled_declared_reference"]["reference_source"] == "thinking card"
    for name in ("two client-row samplers", "a reference naming no decode key", "a reference that did not resolve", "neither rows nor a reference"):
        assert "unfiled_declared_reference" not in resolved_only[name].witnesses, name
    nothing = {
        "disagrees": DE.from_recorder_document(str(_recorder(tmp_path / "x", "DISAGREES")), code_sha=SHA),
        "no document": DE.from_recorder_document(str(tmp_path / "nowhere" / "cells" / "c"), code_sha=SHA),
        "a task the document does not cover": DE.from_recorder_document(str(_recorder(tmp_path / "nc", tasks={"xstest": tb})), task="gsm8k", code_sha=SHA),
    }
    for name, ev in nothing.items():
        assert not ev.recorded() and not ev.resolved_read() and ev.columns() == {}, name
        assert ev.patch_fields() == {"field_confidence_json": {"decode_sent_json": DE.unverified_entry(ev.why)}}, name


def test_a_per_task_recorder_document_answers_for_the_runs_own_task_alone(tmp_path):
    """Review round 1, Major 1: a recorder written with --expect-tasks states one reference per task under classes.client.tasks. Read for a
    task, the evidence is THAT task's block; read for none, or for a task it does not name, it is unverified by name -- never the class-level
    sampler (which is every client request's, one task's or another's)."""
    tb = {"caps": [4096], "n": 120, "verdict": "AGREES", "reference": dict(CARD), "reference_source": "thinking card",
          "distinct_resolved": [{"sampler": dict(RESOLVED6), "n": 120}]}
    xs = dict(tb, n=80, reference={"temperature": 0.0}, reference_source="greedy", distinct_resolved=[{"sampler": dict(RESOLVED6, temperature=0.0), "n": 80}])
    cell = _recorder(tmp_path / "root", tasks={"gsm8k": tb, "xstest": xs})
    ev = DE.from_recorder_document(str(cell), task="gsm8k", code_sha=SHA)
    assert ev.recorded() and ev.sent == {"min_p": 0.0, "presence_penalty": 0.0, "temperature": 0.6, "top_k": 20, "top_p": 0.95} and ev.resolved == RESOLVED6
    assert ev.source.endswith("resolved_sampler.json#classes.client.tasks[gsm8k].reference") and "tasks[gsm8k].distinct_resolved[0]" in ev.resolved_source
    assert ev.witnesses["recorder_tasks"] == ["gsm8k", "xstest"] and ev.witnesses["task"] == "gsm8k" and ev.witnesses["client_requests"] == 120
    assert ev.sent_confidence == DE.SENT_DERIVED                      # a per-task reference is a declaration too (the owner's Change 1)
    ev_x = DE.from_recorder_document(str(cell), task="xstest", code_sha=SHA)
    assert ev_x.recorded() and ev_x.sent == {"temperature": 0.0} and ev_x.resolved == dict(RESOLVED6, temperature=0.0)
    assert DE.for_family("recorder", str(cell), spelling="gsm8k", code_sha=SHA).sent == ev.sent
    # read for no task, or for a task the document does not name: unverified by name, nothing filed
    none = DE.from_recorder_document(str(cell))
    assert not none.recorded() and "per-task references (['gsm8k', 'xstest']); this run's task is not named" in none.why and none.columns() == {}
    other = DE.from_recorder_document(str(cell), task="mmlu")
    assert not other.recorded() and "covers ['gsm8k', 'xstest'], not 'mmlu'" in other.why
    # the task's own verdict, its own distinct count, its own reference
    assert "tasks[gsm8k]: verdict DIVERGED" in DE.from_recorder_document(str(_recorder(tmp_path / "r2", tasks={"gsm8k": dict(tb, verdict="DIVERGED")})), task="gsm8k").why
    two = dict(tb, distinct_resolved=[{"sampler": dict(RESOLVED6), "n": 60}, {"sampler": dict(RESOLVED6, top_k=40), "n": 60}])
    assert "2 distinct resolved sampler(s) for the task" in DE.from_recorder_document(str(_recorder(tmp_path / "r3", tasks={"gsm8k": two})), task="gsm8k").why
    assert "did not resolve to their values" in DE.from_recorder_document(str(_recorder(tmp_path / "r4", tasks={"gsm8k": dict(tb, reference={"temperature": 0.7})})), task="gsm8k").why
    assert "reference names no sampler" in DE.from_recorder_document(str(_recorder(tmp_path / "r5", tasks={"gsm8k": dict(tb, reference=None)})), task="gsm8k").why


# --------------------------------------------------------------------------------------------
# refusals by name: contradictions and shapes; absences return why
# --------------------------------------------------------------------------------------------


def test_contradictions_refuse_by_name_with_the_store_and_the_blocks(tmp_path):
    from sisteer.eval.attacks import bon_climb as BC
    with pytest.raises(DE.DecodeEvidenceContradiction, match="one store, one sampler"):
        DE.from_bon_store(str(_bon_store(tmp_path / "b1", decode=T1, row_decode={"temperature": 0.7})))
    mixed = _bon_store(tmp_path / "b2", rows=3, legacy_rows=2)
    with pytest.raises(DE.DecodeEvidenceContradiction, match="3 rows record a sampler and 2 record none"):
        DE.from_bon_store(str(mixed))
    with pytest.raises(BC.BonClimbError, match="mixture of recorded and legacy rows"):                     # the same store, the same refusal in bon_climb
        BC.store_sampler(mixed)
    two = _bon_store(tmp_path / "b3", rows=4)
    with open(two / DE.BON_RESPONSES, "a") as fh:
        fh.write(json.dumps({"goal_id": "x", "sample_idx": 1, "trial": 0, "decode": {"temperature": 0.5}, "bon_decode_preset": "harness_default"}) + "\n")
    with pytest.raises(DE.DecodeEvidenceContradiction, match="rows record 2 different samplers"):
        DE.from_bon_store(str(two))
    with pytest.raises(BC.BonClimbError, match="2 different samplers"):
        BC.store_sampler(two)
    mt = _mt_store(tmp_path / "m1", shards=("shard00", "shard01"))
    with open(mt / "shard01" / "progress.jsonl", "a") as fh:
        fh.write(json.dumps({"goal_id": "g9", "outcome": "success", "target_decode": dict(SIX, temperature=1.0)}) + "\n")
    with pytest.raises(DE.DecodeEvidenceContradiction, match=r"2 distinct target_decode blocks .*one store, one sampler"):
        DE.from_multiturn_store(str(mt))
    with pytest.raises(DE.DecodeEvidenceContradiction, match="served_target.json#decode states"):
        DE.from_multiturn_store(str(_mt_store(tmp_path / "m2", served_decode=dict(STATIC_BLOCK, temperature=0.9))))
    st = _static_store(tmp_path / "s1", shards=("shard00", "shard01"))
    (st / "shard01" / "served_target.json").write_text(json.dumps({"decode": dict(STATIC_BLOCK, top_k=40)}))
    with pytest.raises(DE.DecodeEvidenceContradiction, match="served_target.json#decode differs across the shards"):
        DE.from_static_store(str(st))
    with pytest.raises(DE.DecodeEvidenceContradiction, match=r"2 response row\(s\) in shard00 were sent .* while served_target.json states"):
        DE.from_static_store(str(_static_store(tmp_path / "s2", shards=("shard00",), rows=2, row_decode=dict(STATIC_BLOCK, temperature=0.6))))
    with pytest.raises(DE.DecodeEvidenceContradiction, match="record a decode block and 2 record none"):
        s3 = _static_store(tmp_path / "s3", shards=("shard00",), rows=2)
        with open(s3 / "shard00" / "responses.jsonl", "a") as fh:
            fh.write(json.dumps({"goal_id": "n1", "variant": 0}) + "\n"); fh.write(json.dumps({"goal_id": "n2", "variant": 0}) + "\n")
        DE.from_static_store(str(s3))
    with pytest.raises(DE.DecodeEvidenceContradiction, match="states a decode block in 1 shard\\(s\\) and none in shard01"):
        s4 = _static_store(tmp_path / "s4", shards=("shard00", "shard01"))
        (s4 / "shard01" / "served_target.json").write_text(json.dumps({"n_sampled_requests": 4}))
        DE.from_static_store(str(s4))
    d, man = _controls_cell(tmp_path / "c1", request_samplers={"xstest": dict(SIX, temperature=0.9)})
    with pytest.raises(DE.DecodeEvidenceContradiction, match="stated twice and differently: request_samplers.xstest .* vs resolved_config.gen"):
        DE.from_store_manifest(man, "xstest", store_dir=str(d))
    d2, man2 = _controls_cell(tmp_path / "c2", gs_decode=dict(SIX, top_p=0.95, do_sample=True))
    with pytest.raises(DE.DecodeEvidenceContradiction, match="vs gen_stats.json#xstest.decode"):
        DE.from_store_manifest(man2, "xstest", store_dir=str(d2))
    # the greedy opt-in's three disagreements (Major 3): a sampled statement under a greedy gen; the greedy statement beside a
    # SAMPLED gen_stats block; a sampled statement beside the GREEDY gen_stats block
    greedy_gen = {"batch_size": 8, "do_sample": False, "max_new_tokens": 512}
    d3, man3 = _controls_cell(tmp_path / "c3", gen=greedy_gen, request_samplers={"xstest": dict(SIX)}, gen_stats=False)
    with pytest.raises(DE.DecodeEvidenceContradiction, match="vs resolved_config.gen do_sample false \\(the --greedy opt-in"):
        DE.from_store_manifest(man3, "xstest", store_dir=str(d3))
    d4, man4 = _controls_cell(tmp_path / "c4", gen=greedy_gen, request_samplers={"xstest": {"temperature": 0.0}}, gs_decode=dict(SIX, do_sample=True, cap=512))
    with pytest.raises(DE.DecodeEvidenceContradiction, match="request_samplers.xstest .* vs gen_stats.json#xstest.decode"):
        DE.from_store_manifest(man4, "xstest", store_dir=str(d4))
    d5, man5 = _controls_cell(tmp_path / "c5", request_samplers={"xstest": dict(SIX)}, gs_decode={"do_sample": False, "cap": 512, "arm": "greedy"})
    with pytest.raises(DE.DecodeEvidenceContradiction, match="vs gen_stats.json#xstest.decode the greedy block"):
        DE.from_store_manifest(man5, "xstest", store_dir=str(d5))
    with pytest.raises(DE.DecodeEvidenceContradiction, match="the runner disagrees with itself"):
        DE.from_misalign_store(str(_misalign_cell(tmp_path / "mi1", params_sampling={"temperature": 0.7})))


def test_shape_problems_refuse_a_bool_a_fractional_top_k_a_non_json_artifact_and_a_bon_preset_outside_the_set(tmp_path):
    with pytest.raises(DE.DecodeEvidenceShape, match="temperature=True is not a number"):
        DE.split_six({"temperature": True}, "x")
    with pytest.raises(DE.DecodeEvidenceShape, match="top_k=20.5 is not an integer"):
        DE.split_six({"temperature": 0.7, "top_k": 20.5}, "x")
    with pytest.raises(DE.DecodeEvidenceShape, match="temperature='0.7' is not a number"):
        DE.split_six({"temperature": "0.7"}, "x")
    with pytest.raises(DE.DecodeEvidenceShape, match="no temperature"):
        DE.split_six({"top_p": 0.8}, "x")
    assert DE.split_six({"top_k": 20.0, "temperature": 1, "do_sample": True, "cap": 4096}, "x") == ({"top_k": 20, "temperature": 1.0}, {"do_sample": True, "cap": 4096})
    assert DE.split_six({"temperature": 1.0, "top_p": None, "top_k": None}, "x") == ({"temperature": 1.0}, {})   # the canonical instrument's null slots
    b = _bon_store(tmp_path / "b"); (b / DE.BON_MANIFEST).write_text("{not json")
    with pytest.raises(DE.DecodeEvidenceShape, match="not JSON"):
        DE.from_bon_store(str(b))
    with pytest.raises(DE.DecodeEvidenceShape, match="outside the sampler set \\['do_sample'\\]"):
        DE.from_bon_store(str(_bon_store(tmp_path / "b4", decode={"temperature": 1.0, "do_sample": True}, row_decode=T1)))
    with pytest.raises(DE.DecodeEvidenceShape, match="torn sampler record"):
        t = _bon_store(tmp_path / "b5", rows=1)
        with open(t / DE.BON_RESPONSES, "a") as fh:
            fh.write(json.dumps({"goal_id": "t", "sample_idx": 0, "decode": "nope", "bon_decode_preset": "harness_default"}) + "\n")
        DE.from_bon_store(str(t))
    with pytest.raises(DE.DecodeEvidenceShape, match="line 2 is not JSON"):
        m = _mt_store(tmp_path / "m", rows=1)
        with open(m / "shard00" / "progress.jsonl", "a") as fh:
            fh.write('{"goal_id": "torn"\n')
        DE.from_multiturn_store(str(m))


def test_absences_file_the_unverified_entry_and_no_column(tmp_path):
    cases = [
        DE.from_bon_store(str(_bon_store(tmp_path / "b", with_decode=False, legacy_rows=2, rows=0))),
        DE.from_multiturn_store(str(_mt_store(tmp_path / "m", with_td=False, served_decode=None))),
        DE.from_static_store(str(_static_store(tmp_path / "s", with_decode=False, row_decode=None))),
        DE.from_misalign_store(str(_misalign_cell(tmp_path / "mi", with_sampling=False, params_sampling=None))),
        DE.from_bon_store(str(tmp_path / "absent_bon")),
        DE.for_family("unknown_family", str(tmp_path)),
    ]
    whys = [c.why for c in cases]
    assert "#decode absent: a legacy BoN store" in whys[0] and "n_legacy_unrecorded_rows=2" in whys[0]
    assert "no progress row states target_decode" in whys[1] and "no shard's served_target.json states a decode block" in whys[2]
    assert "#sampling absent" in whys[3] and "no bon_served_manifest.json" in whys[4] and "no decode evidence reader for family 'unknown_family'" in whys[5]
    for c in cases:
        assert not c.recorded() and c.columns() == {} and c.patch_fields() == {"field_confidence_json": {"decode_sent_json": {"confidence": "unverified", "reason": c.why}}}
    # a BoN store whose rows are all legacy beside a manifest that states the sampler: a mixture by the store's own rule (recorded vs legacy)
    assert DE.from_bon_store(str(_bon_store(tmp_path / "b2", rows=0, legacy_rows=3))).witnesses["rows_legacy"] == 3   # no recorded row: not a mixture, the manifest stands


# --------------------------------------------------------------------------------------------
# the patch shape, through the registry's own validator
# --------------------------------------------------------------------------------------------


def test_patch_fields_validate_as_a_run_patch_and_the_derived_pair_is_still_refused_by_name(tmp_path):
    ev = DE.from_multiturn_store(str(_mt_store(tmp_path / "mt")))
    fields = ev.patch_fields()
    assert set(fields) <= set(R.RUN_PATCHABLE_FIELDS), set(fields) - set(R.RUN_PATCHABLE_FIELDS)
    assert fields["extra_json"] == {DE.EXTRA_KEY: ev.extra_block()} and fields["extra_json"][DE.EXTRA_KEY]["reader"] == DE.READER
    rec = R.make_record("run_patch", {"run_id": "synthcampaign:syntheval:abcdefabcdef:0123456789abcdef", "fields": fields}, synth.producer(), "11111111-2222-3333-4444-555555555555", synth.t(9))
    v = VAL.validate_record(rec, "<memory>", R.record_sha(rec))
    assert isinstance(v, VAL.Validated), getattr(v, "reason", v)
    assert v.payload["fields"]["decode_top_k_sent"] == 20 and json.loads(v.payload["fields"]["decode_sent_json"]) == SIX_F
    bad = R.make_record("run_patch", {"run_id": rec["payload"]["run_id"], "fields": dict(fields, decode_policy_id=MT_POLICY)}, synth.producer(), "11111111-2222-3333-4444-555555555556", synth.t(9))
    out = VAL.validate_record(bad, "<memory>", R.record_sha(bad))
    assert isinstance(out, VAL.Malformed) and out.field == "decode_policy_id" and "stamp_run_rows" in out.reason
    # a recorder evidence's patch carries the resolved column with its own `recorded` entry beside the declared reference's `derived` set
    ev_r = DE.from_recorder_document(str(_recorder(tmp_path / "root")), code_sha=SHA)
    f_r = ev_r.patch_fields()
    assert f_r["decode_resolved_json"] == ev_r.resolved and f_r["field_confidence_json"]["decode_resolved_json"]["confidence"] == "recorded"
    assert f_r["field_confidence_json"]["decode_sent_json"]["confidence"] == "derived"
    rec_r = R.make_record("run_patch", {"run_id": rec["payload"]["run_id"], "fields": f_r}, synth.producer(), "11111111-2222-3333-4444-555555555557", synth.t(9))
    assert isinstance(VAL.validate_record(rec_r, "<memory>", R.record_sha(rec_r)), VAL.Validated)


# --------------------------------------------------------------------------------------------
# the write-time verification
# --------------------------------------------------------------------------------------


def test_verify_against_sees_a_sha_drift_a_count_drift_a_block_drift_and_a_flipped_verdict(tmp_path):
    store = _static_store(tmp_path / "st", shards=("shard00",), rows=3)
    req = {"run_id": "r", "family": "static", "store_uri": str(store)}
    frozen = DE.read_run(req)
    assert frozen["ok"] and DE.verify_against(frozen, DE.read_run(req)) == []
    # (a) a response row appended with a DIFFERENT decode: the re-read refuses -> the verdict flips
    with open(store / "shard00" / "responses.jsonl", "a") as fh:
        fh.write(json.dumps({"goal_id": "late", "variant": 0, "decode": dict(STATIC_BLOCK, temperature=0.2)}) + "\n")
    d = DE.verify_against(frozen, DE.read_run(req))
    assert len(d) == 1 and d[0].startswith("ok True -> False") and "while served_target.json states" in d[0]
    # (b) a response row appended with the SAME decode: the count drifts
    store2 = _static_store(tmp_path / "st2", shards=("shard00",), rows=3); req2 = dict(req, store_uri=str(store2)); frozen2 = DE.read_run(req2)
    with open(store2 / "shard00" / "responses.jsonl", "a") as fh:
        fh.write(json.dumps({"goal_id": "late", "variant": 0, "decode": dict(STATIC_BLOCK)}) + "\n")
    d2 = DE.verify_against(frozen2, DE.read_run(req2))
    assert any(x.startswith("witness response_rows: 3 -> 4") for x in d2), d2
    # (c) the evidence file rewritten (same block, new bytes): the sha drifts, both reads
    store3 = _static_store(tmp_path / "st3", shards=("shard00",), rows=3); req3 = dict(req, store_uri=str(store3)); frozen3 = DE.read_run(req3)
    (store3 / "shard00" / "served_target.json").write_text(json.dumps({"decode": dict(STATIC_BLOCK), "n_sampled_requests": 3}, indent=2))
    d3 = DE.verify_against(frozen3, DE.read_run(req3))
    assert any(x.startswith("sha256:") for x in d3) and any(x.startswith("sha256_read2:") for x in d3), d3
    # (d) the block itself rewritten everywhere: the sent set drifts
    store4 = _static_store(tmp_path / "st4", shards=("shard00",), rows=0); req4 = dict(req, store_uri=str(store4)); frozen4 = DE.read_run(req4)
    (store4 / "shard00" / "served_target.json").write_text(json.dumps({"decode": dict(STATIC_BLOCK, temperature=0.9)}))
    assert any(x.startswith("sent:") for x in DE.verify_against(frozen4, DE.read_run(req4)))
    # (e) a frozen refusal that re-reads clean flips the other way
    assert DE.verify_against({"ok": False, "error": "x"}, frozen)[0].startswith("ok False -> True")
