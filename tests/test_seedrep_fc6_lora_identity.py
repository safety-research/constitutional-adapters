"""qwen-seedrep-native-battery: the sampled-controls audit rejected a DIFFERENT named LoRA but
ACCEPTED `lora_request: None` for any arm, and the MMLU count-level audit substituted base-weight requests whenever no line named the arm's
LoRA -- a base-model draw / readout could certify under a fitted arm's identity. Now the EXPECTED engine identity is derived from the
standup's own records (`Standup.engine_lora_expectation`: `serving_attestation.json.adapters` -- the LoRA names the engine registered, ==
the harness cell ids -- with `id_name_map`, cross-checked with the cell's `serving_identity` in `cells_served.json` / `routing_table.json`):
a fitted arm REQUIRES its LoRA name on EVERY bound engine line; no-LoRA evidence is admitted ONLY for an arm the records serve on the base
weights (`none`, a prompted cell); a line without a `lora_request` clause refuses; the MMLU count counts only lines under the arm's attested
identity and refuses a fitted arm none of whose lines carry it. Both paths, own and copied-forward evidence (the ancestor's standup).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
from test_seedrep_fc3_controls_audit_shape import CELL, run_audit, w23_controls_arditi_root  # noqa: E402

PROMPTED = "prompted__si_constitution"


def _verdicts(out: Path):
    return [json.loads((out / e["rel"]).read_text()) for e in json.loads((out / rea.INDEX_NAME).read_text())["verdicts"]]


def test_the_expectation_comes_from_the_standups_records_not_the_log(tmp_path):
    w = w23_controls_arditi_root(tmp_path)
    st = ac.read_standup(ac.read_attempt_root(w["root"].path))
    want, src = st.engine_lora_expectation(CELL)
    assert want == CELL and "serving_attestation.json.adapters['arditi_allpos']" in src and "lora_int_id 1" in src and "route lora" in src
    assert st.attestation["id_name_map"] == {"1": CELL, "2": "none__lora0"} and st.serving_identity_of(CELL) == "lora"
    root2 = fx.make_root(tmp_path / "s1", leg="controls", view="s1", arms=("none", PROMPTED, "constv2-mix-full__lora_r4_ce"))
    st2 = ac.read_standup(ac.read_attempt_root(root2.path))
    assert st2.engine_lora_expectation("none")[0] is None and "recorded passthrough identity" in st2.engine_lora_expectation("none")[1]
    assert st2.engine_lora_expectation(PROMPTED)[0] is None and st2.serving_identity_of(PROMPTED) == "passthrough" and st2.serving_identity_of("none") == "passthrough"
    assert st2.engine_lora_expectation("constv2-mix-full__lora_r4_ce")[0] == "constv2-mix-full__lora_r4_ce" and st2.serving_identity_of("constv2-mix-full__lora_r4_ce") == "lora"
    # records that cannot establish the identity refuse by name -- never a base-weight assumption for a fitted arm
    att = json.loads(st2.attestation_path.read_text())
    del att["adapters"]["constv2-mix-full__lora_r4_ce"]
    st2.attestation_path.write_text(json.dumps(att))
    st3 = ac.read_standup(ac.read_attempt_root(root2.path))
    with pytest.raises(ac.AuditRefusal, match="routed as a LoRA cell .* but serving_attestation.json.adapters attests no adapter"):
        st3.engine_lora_expectation("constv2-mix-full__lora_r4_ce")
    att["adapters"] = {}
    st2.attestation_path.write_text(json.dumps(att))
    # fc8: the identity files present but SILENT on a cell (a partially pulled record) -> refusal for every arm, `none` and prompted included
    for name, key in ((ac.CELLS_SERVED_FILE, None), (ac.ROUTING_TABLE_FILE, "routes")):
        d = json.loads((root2.serve / name).read_text())
        if key:
            d[key] = [r for r in d[key] if r.get("cell_id") == "constv2-mix-full__lora_r4_ce"]
        else:
            d["canonical"] = [r for r in d["canonical"] if r.get("cell_id") == "constv2-mix-full__lora_r4_ce"]
            d["controls"] = []
        (root2.serve / name).write_text(json.dumps(d))
    st_silent = ac.read_standup(ac.read_attempt_root(root2.path))
    for arm in ("none", PROMPTED):
        with pytest.raises(ac.AuditRefusal, match=f"establish no serving identity for cell '{arm}' .*never inferred from a cell's name"):
            st_silent.engine_lora_expectation(arm)
    # the identity files ABSENT -> the same refusal for every arm; a base-weight identity is read from a recorded passthrough route only
    for name in (ac.CELLS_SERVED_FILE, ac.ROUTING_TABLE_FILE):
        (root2.serve / name).unlink()
    st4 = ac.read_standup(ac.read_attempt_root(root2.path))
    for arm in ("constv2-mix-full__lora_r4_ce", "none", PROMPTED):
        with pytest.raises(ac.AuditRefusal, match=f"establish no serving identity for cell '{arm}'"):
            st4.engine_lora_expectation(arm)
        assert st4.serving_identity_of(arm) is None


@pytest.mark.parametrize("mutate, needle, legs", [
    ("explicit_none", "on the BASE weights (lora_request None), but the standup attests arm 'arditi_allpos'", {"xstest", "or_bench_hard_1k"}),
    ("no_clause", "states no `lora_request` clause", {"xstest", "or_bench_hard_1k"}),
    ("other_named", "under LoRA 'other_cell', not the arm's attested adapter 'arditi_allpos'", {"xstest", "or_bench_hard_1k"}),
    ("mmlu_none", "no MMLU-shaped engine request names the arm's attested LoRA 'arditi_allpos'", {"mmlu"}),
    ("mmlu_no_clause", "no MMLU-shaped engine request names the arm's attested LoRA 'arditi_allpos'", {"mmlu"}),
])
def test_a_fitted_arm_refuses_base_weight_or_unstated_identity_by_name(tmp_path, mutate, needle, legs):
    w = w23_controls_arditi_root(tmp_path)
    root = w["root"]
    log_p = root.serve / "vllm_server.log"
    lines = log_p.read_text().splitlines()
    lora_clause = ", lora_request: LoRARequest(lora_name='arditi_allpos'"

    def is_mmlu(l):
        return "max_tokens=1," in l
    if mutate == "explicit_none":
        lines = [l.split(", lora_request:")[0] + ", lora_request: None." if (lora_clause in l and not is_mmlu(l)) else l for l in lines]
    elif mutate == "no_clause":
        lines = [l.split(", lora_request:")[0] + "." if (lora_clause in l and not is_mmlu(l)) else l for l in lines]
    elif mutate == "other_named":
        lines = [l.replace("lora_name='arditi_allpos'", "lora_name='other_cell'") if not is_mmlu(l) else l for l in lines]
    elif mutate == "mmlu_none":
        lines = [l.split(", lora_request:")[0] + ", lora_request: None." if is_mmlu(l) else l for l in lines]
    elif mutate == "mmlu_no_clause":
        lines = [l.split(", lora_request:")[0] + "." if is_mmlu(l) else l for l in lines]
    log_p.write_text("\n".join(lines) + "\n")
    out = root.path / "audits"
    assert run_audit(root.path, out) == ac.EXIT_FAILED
    refused = [v for v in _verdicts(out) if v["verdict"] != rea.VERDICT_AGREES]
    assert refused and {v["leg"] for v in refused} == legs, [(v["leg"], v["problem"]) for v in refused]
    assert all(needle in str(v["problem"]) for v in refused), [v["problem"] for v in refused]
    if legs == {"mmlu"}:
        ec = refused[0]["checks"]["engine_count"]
        assert ec["n_engine_mmlu_shaped_under_arm"] == 0 and ec["engine_identity_expected"] == {"lora_name": CELL, "source": ec["engine_identity_expected"]["source"]}
        assert "NOT substituted" in refused[0]["problem"]
    else:
        assert all(v["checks"]["controls_events"]["engine_identity_expected"]["lora_name"] == CELL for v in refused)
    # the mutation touched the fitted arm's identity only: every other verdict still AGREES
    assert {v["leg"] for v in _verdicts(out) if v["verdict"] == rea.VERDICT_AGREES} == {"xstest", "or_bench_hard_1k", "mmlu"} - legs


def test_a_base_weight_arm_agrees_with_none_and_refuses_a_named_lora(tmp_path):
    # `none` on a view root, served on the base weights per the standup's records: `lora_request: None` on every line is what the contract requires
    w = w23_controls_arditi_root(tmp_path / "ok", view="s1", arm="none", engine_lora=None, seeds=(1,), extra_arms=(PROMPTED,))
    out = w["root"].path / "audits"
    assert run_audit(w["root"].path, out) == ac.EXIT_OK
    vs = _verdicts(out)
    assert len(vs) == 3 and all(v["verdict"] == rea.VERDICT_AGREES and v["arm"] == "none" for v in vs)
    for v in vs:
        exp = (v["checks"].get("controls_events") or v["checks"].get("engine_count"))["engine_identity_expected"]
        assert exp["lora_name"] is None and "recorded passthrough identity" in exp["source"]
    assert [v for v in vs if v["leg"] == "mmlu"][0]["checks"]["engine_count"]["weights"].startswith("the base weights")
    # the prompted arm is a base-weight cell per the RECORDED passthrough route too
    st = ac.read_standup(ac.read_attempt_root(w["root"].path))
    assert st.engine_lora_expectation(PROMPTED)[0] is None and "recorded passthrough identity" in st.engine_lora_expectation(PROMPTED)[1]
    # fc8: the same base arm with the passthrough route REMOVED from the records -> the audit refuses (no positive identity verdict without a recorded route)
    import shutil as _sh
    for name in (ac.CELLS_SERVED_FILE, ac.ROUTING_TABLE_FILE):
        (w["root"].serve / name).unlink()
    _sh.rmtree(out)
    with pytest.raises(ac.AuditRefusal, match="establish no serving identity for cell 'none'"):
        run_audit(w["root"].path, out)
    # the SAME base-weight arm served under a named LoRA by the engine: refused by name (a LoRA draw under the base arm's identity)
    w2 = w23_controls_arditi_root(tmp_path / "bad", view="s1", arm="none", engine_lora="none__lora0", seeds=(1,), extra_arms=(PROMPTED,))
    out2 = w2["root"].path / "audits"
    assert run_audit(w2["root"].path, out2) == ac.EXIT_FAILED
    refused = [v for v in _verdicts(out2) if v["verdict"] != rea.VERDICT_AGREES]
    assert {v["leg"] for v in refused} == {"xstest", "or_bench_hard_1k", "mmlu"}
    assert all("under LoRA 'none__lora0', but the standup records arm 'none' (cell 'none') as served on the base weights" in v["problem"]
               or "0 MMLU-shaped request(s) under the base weights" in v["problem"] for v in refused), [v["problem"] for v in refused]
    assert [v for v in refused if v["leg"] == "mmlu"][0]["checks"]["engine_count"]["n_engine_mmlu_shaped_by_lora"] == {"none__lora0": 3}, "the LoRA lines are counted, never credited to the base arm"


def test_the_fitted_arm_still_agrees_and_the_fixture_shape_is_the_real_one(tmp_path):
    w = w23_controls_arditi_root(tmp_path)
    out = w["root"].path / "audits"
    assert run_audit(w["root"].path, out) == ac.EXIT_OK
    vs = _verdicts(out)
    assert len(vs) == 9 and all(v["verdict"] == rea.VERDICT_AGREES for v in vs)
    for v in vs:
        exp = (v["checks"].get("controls_events") or v["checks"].get("engine_count"))["engine_identity_expected"]
        assert exp["lora_name"] == CELL
    att = json.loads((w["root"].serve / ac.ATTESTATION_FILE).read_text())
    assert set(att["adapters"]) == {CELL, "none__lora0"} and att["id_name_map"] == {"1": CELL, "2": "none__lora0"}, "adapters keyed by the LoRA name the engine states (the real record's shape)"
    rt = json.loads((w["root"].serve / ac.ROUTING_TABLE_FILE).read_text())
    assert {r["cell_id"]: r["serving_identity"] for r in rt["routes"]}[CELL] == "proxy_vllm+lora+interleaved"
