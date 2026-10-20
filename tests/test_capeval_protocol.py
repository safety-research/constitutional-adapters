"""Stage-0 frozen protocol fixtures.

This suite pins the interfaces every capeval workstream codes against:
the §2.1 manifest schema, the §2.2 registry, the §2.3 row/cell-key/store
schema, and the §2.4 engine protocol — driven end-to-end on synthetic
canned data (manifest → cells → rows → stub fingerprint), plus the canned
EvalPlus samples/eval-results fixture pair.

Contract: EVERY workstream keeps this suite green. Tests marked
``xfail(raises=NotImplementedError, strict=False)`` exercise seams whose
bodies are Stage-0 stubs: today they XFAIL; once the owning WS implements
the body they flip to XPASS with the pinned behavior verified — and any
OTHER exception (or a failed assert) is a real failure, which is exactly
the interface-drift alarm this file exists to raise.

Base lane only: no evalplus / vllm imports anywhere here.
"""

import dataclasses
import json
from pathlib import Path

import pytest
import yaml

from sisteer.capeval import domains as cd
from sisteer.capeval import manifest as cm
from sisteer.capeval import store as cs
from sisteer.capeval import serving as cserv
import evalplus_synthetic as EP  # noqa: E402  (tests/ helper: the synthetic EvalPlus sample pair)

FIXTURES = Path(__file__).parent / "fixtures" / "capeval"

# --------------------------------------------------------------------------
# Frozen schema constants. A change here is an interface change.
# --------------------------------------------------------------------------

MANIFEST_FIELDS = (
    "sweep", "mode", "base", "base_revision", "engine", "tiers",
    "ablations", "confirmatory", "target", "margins",
)
TARGETSPEC_FIELDS = ("primaries", "relevant_corpus", "secondary", "off_target_exclusions")
PRIMARYSPEC_FIELDS = ("metric", "plugin", "params", "direction", "min_effect")
CORPUSREF_FIELDS = ("slice", "path", "repo_id", "revision")
ABLATIONSPEC_FIELDS = (
    "name", "kind", "artifact", "layer",
    # `layers` (cd8927d6 / 4d25c09c; deliberate additive widening): a DECLARED multi-layer
    # object's whole layer set; None for every single-layer arm, so no existing manifest or
    # hash changes
    "layers", "coeffs", "positions",
    "norm_relative", "seed", "params",
)
ROW_COLUMNS_FROZEN = (
    "sweep", "run_id", "model_id", "model_revision", "arm_name", "kind",
    "coeff", "intervention_hash", "engine", "tier", "task", "task_version",
    "domain", "item_id", "trial_id", "seed", "attempt", "position_order",
    "metric", "value", "extra_json", "harness_config_hash", "env_digest",
    "slurm_job_id", "created_at", "wall_clock_s",
)
DOMAIN_NAMES_FROZEN = (
    "lm_fidelity_raw", "lm_fidelity_chat", "dist_shift", "coherence_canary",
    "kl_localization", "knowledge", "commonsense", "truthfulness", "math",
    "code", "instruction_following", "reasoning", "math_s2parity",
    "knowledge_s2parity",
    # petri-bloom (deliberate additive widening, recorded): ONE
    # display-kind domain for the Petri misalignment audit — never gating,
    # never statuses, six petri/<behavior> tasks.
    "petri_behavior",
    # tiny-capability-lmeval-vllm (deliberate additive widening, recorded):
    # THREE display-kind domains for the served tinycap families — never
    # gating, never statuses; 5 served_ll/ + 8 inspect/ + 4 evalplus/ tasks.
    "served_ll", "inspect_gen", "evalplus_served",
    # em-misalign (2026-09-10, deliberate additive widening, recorded): ONE
    # display-kind domain for the served agentic-misalignment (leaking,
    # murder) and Betley-8Q rows — never gating, never statuses; paired per
    # item against `none` by sisteer.capeval.misalign.analyze.
    "misalign",
)
# petri-bloom (deliberate additive widening of the ENGINES enum,
# recorded for Evals): the proxy/shim serving identities
# "<backend>+<path>+<mode>". The in-process values ("hf", "vllm") and every
# other capability constant are unchanged.
PETRI_SERVING_IDENTITIES_FROZEN = (
    "proxy_vllm+lens+interleaved", "proxy_vllm+lens+exact_replay",
    "proxy_vllm+lora+interleaved", "proxy_vllm+lora+exact_replay",
    "proxy_vllm+passthrough+interleaved", "proxy_vllm+passthrough+exact_replay",
    "hf_shim+apply+interleaved", "hf_shim+apply+exact_replay",
)
GATING_DOMAINS_FROZEN = (
    "lm_fidelity_raw", "lm_fidelity_chat", "knowledge", "commonsense",
    "truthfulness", "math", "code", "instruction_following", "reasoning",
)
ACCEPTANCE_DOMAINS_FROZEN = ("math_s2parity", "knowledge_s2parity")
# Frozen EvalPlus row conversion: per task_id, two per-item
# rows on task "humaneval_plus", item_id = the task_id.
EVALPLUS_METRICS = ("pass@1_base", "pass@1_plus")
# Frozen per-arm fingerprint output shape: domain statuses
# for every registered non-gate/display domain + a screen status. v0
# emits these, never confirmatory verdicts.
STATUSES = ("preserved", "degraded", "indeterminate")
SCREENS = ("SCREEN-PASS", "SCREEN-FAIL")


def _field_names(dc) -> tuple:
    return tuple(f.name for f in dataclasses.fields(dc))


# --------------------------------------------------------------------------
# Dataclass / constant schema pins (pass today)
# --------------------------------------------------------------------------

def test_manifest_dataclass_schemas_pinned():
    assert _field_names(cm.Manifest) == MANIFEST_FIELDS
    assert _field_names(cm.TargetSpec) == TARGETSPEC_FIELDS
    assert _field_names(cm.PrimarySpec) == PRIMARYSPEC_FIELDS
    assert _field_names(cm.CorpusRef) == CORPUSREF_FIELDS
    assert _field_names(cm.AblationSpec) == ABLATIONSPEC_FIELDS
    assert cm.MODES == ("sweep", "control", "calibration")
    assert cm.ENGINES == ("hf", "vllm") + PETRI_SERVING_IDENTITIES_FROZEN
    assert cm.SERVING_IDENTITIES == PETRI_SERVING_IDENTITIES_FROZEN
    assert cm.STEERING_KINDS == ("steering", "sisteer_object")
    assert cm.NONE_KIND == "none"
    assert cm.CONTROL_KIND_PREFIX == "control:"
    assert cm.HASH_HEX_LEN == 16


def test_row_schema_pinned():
    assert cs.ROW_COLUMNS == ROW_COLUMNS_FROZEN
    assert _field_names(cs.ResultRow) == ROW_COLUMNS_FROZEN


def test_domain_dataclass_schema_pinned():
    # `retired` (deliberate additive widening): a bool annotation marking a
    # row whose instrument no longer runs forward (math_s2parity); default
    # False, no semantics change
    assert _field_names(cd.Domain) == (
        "name", "kind", "tasks", "direction", "margin", "gating", "role",
        "retired",
        "unit", "transform", "task_weight", "item_weight", "estimator",
    )
    assert _field_names(cd.Registry) == ("version", "domains")
    assert cd.DOMAIN_KINDS == ("accuracy", "nll", "gate", "display")
    assert cd.F14_FIELDS == ("unit", "transform", "task_weight", "item_weight", "estimator")
    assert cd.ITEM_WEIGHTS == ("items", "tokens")
    assert cd.ROLES == (None, "acceptance")


def test_engine_protocol_shape_pinned():
    # gen_token_ids: additive, defaulted (r1 tier0-engine #9 — the canary
    # reads raw ids; special-token degeneration is invisible in text)
    assert _field_names(cserv.GenOut) == (
        "text", "gen_token_logprobs", "ended_with_eos", "gen_token_ids")
    assert cserv.PositionScore._fields == ("token_nll", "atom_probs")
    # PositionScore unpacks as the plan's (token_nll, atom_probs|None) tuple
    nll, atoms = cserv.PositionScore(0.5, None)
    assert (nll, atoms) == (0.5, None)
    assert _field_names(cserv.ArmHandle) == ("name", "kind", "coeff", "intervention_hash")

    class DummyEngine:
        name = "hf"

        def load(self):  # pragma: no cover - shape only
            pass

        def arm(self, spec, coeff):  # pragma: no cover
            pass

        def score_corpus(self, corpus, arm, atom_ids=None):  # pragma: no cover
            pass

        def generate(self, prompts, arm, gen_cfg):  # pragma: no cover
            pass

    assert isinstance(DummyEngine(), cserv.Engine)
    # ArmHandle is constructible (the frozen attribute schema) and its
    # Stage-0 method stubs raise until an engine implements them.
    handle = cserv.ArmHandle(name="a", kind="steering", coeff=2.0, intervention_hash="ab" * 8)
    with pytest.raises(NotImplementedError):
        handle.ctx()
    with pytest.raises(NotImplementedError):
        handle.transform_prompt("x")


# --------------------------------------------------------------------------
# Manifest fixture: parses, constructs every dataclass (passes today)
# --------------------------------------------------------------------------

def _load_manifest_fixture_by_hand() -> cm.Manifest:
    raw = yaml.safe_load((FIXTURES / "manifest_synthetic.yaml").read_text())
    target = raw["target"]
    return cm.Manifest(
        sweep=raw["sweep"],
        mode=raw["mode"],
        base=raw["base"],
        base_revision=raw["base_revision"],
        engine=raw["engine"],
        tiers=raw["tiers"],
        confirmatory=raw["confirmatory"],
        margins=raw["margins"],
        target=cm.TargetSpec(
            primaries=[cm.PrimarySpec(**p) for p in target["primaries"]],
            secondary=[cm.PrimarySpec(**p) for p in target["secondary"]],
            off_target_exclusions=target["off_target_exclusions"],
            relevant_corpus=cm.CorpusRef(**target["relevant_corpus"]),
        ),
        ablations=[cm.AblationSpec(**a) for a in raw["ablations"]],
    )


def test_manifest_fixture_constructs_every_dataclass():
    m = _load_manifest_fixture_by_hand()
    assert m.mode == "sweep" and m.confirmatory is False
    assert m.engine in cm.ENGINES and m.tiers == [0, 1]
    assert len(m.base_revision) == 40
    assert m.target is not None
    assert m.target.relevant_corpus.path.endswith("pairs_synthetic.jsonl")
    assert m.target.relevant_corpus.slice == {"first": 2, "field": "prompt"}
    [steer, ctrl] = m.ablations
    assert steer.kind in cm.STEERING_KINDS and steer.layer == 15 and steer.coeffs == [2.0]
    assert ctrl.kind.startswith(cm.CONTROL_KIND_PREFIX)
    assert ctrl.artifact is None and ctrl.seed == 7 and len(ctrl.coeffs) == 2
    # referenced fixture artifacts exist (load_manifest refuses absent
    # artifacts; relative paths anchor to the MANIFEST's own directory —
    # review-fix contract 2026-08-15)
    root = Path(__file__).parents[1]
    assert (FIXTURES / steer.artifact).is_file()
    assert (root / m.target.relevant_corpus.path).is_file()
    assert (FIXTURES / steer.artifact).with_suffix(".json").is_file()  # sidecar


# --------------------------------------------------------------------------
# domains.yaml registry contents (passes today)
# --------------------------------------------------------------------------

def _domains_yaml() -> dict:
    root = Path(__file__).parents[1]
    return yaml.safe_load((root / "configs" / "capeval" / "domains.yaml").read_text())


def test_domains_yaml_registry_pinned():
    reg = _domains_yaml()
    assert reg["version"] == 1
    doms = reg["domains"]
    assert tuple(doms) == DOMAIN_NAMES_FROZEN
    gating = tuple(n for n, d in doms.items() if d.get("gating"))
    assert gating == GATING_DOMAINS_FROZEN  # 9 gating domains
    acceptance = tuple(n for n, d in doms.items() if d.get("role") == "acceptance")
    assert acceptance == ACCEPTANCE_DOMAINS_FROZEN
    assert tuple(n for n, d in doms.items() if d["kind"] == "gate") == (
        "dist_shift", "coherence_canary")
    assert tuple(n for n, d in doms.items() if d["kind"] == "display") == (
        "kl_localization", "petri_behavior",          # petri: display, never gating
        "served_ll", "inspect_gen", "evalplus_served",  # tinycap: display, never gating
        "misalign")  # em-misalign: display, never gating
    # acceptance-role rows are never gating (r2-F6)
    for name in acceptance:
        assert doms[name]["gating"] is False


def test_domains_yaml_f14_fields_complete():
    doms = _domains_yaml()["domains"]
    for name, d in doms.items():
        assert d["kind"] in cd.DOMAIN_KINDS, name
        if d["kind"] in ("accuracy", "nll"):
            for f in cd.F14_FIELDS:
                assert f in d, f"{name} missing F14 field {f}"
            assert d["item_weight"] in cd.ITEM_WEIGHTS, name
            assert d["direction"] in cd.DIRECTIONS, name
            assert isinstance(d["margin"], float), name
            assert d["estimator"] == "paired_bootstrap", name
        if d["kind"] == "nll":
            assert (d["unit"], d["transform"], d["item_weight"], d["margin"]) == (
                "nats_per_token", "relative_nll", "tokens", 0.03), name
        if d["kind"] == "accuracy":
            assert (d["unit"], d["transform"], d["item_weight"], d["margin"]) == (
                "fraction_correct", "identity", "items", 0.02), name


def test_domains_yaml_task_map_is_a_partition():
    doms = _domains_yaml()["domains"]
    all_tasks = [t for d in doms.values() for t in d["tasks"]]
    # 21 capability tasks + the 6 petri/<behavior> display tasks
    # + the 20 tinycap served display tasks (17 + inspect/gsm8k_paired200,
    # the GSM8K comparison bridge added deliberately with PW3; + inspect/bigcodebench_200
    # and inspect/bfcl_200, the bcb-bfcl-200 pinned legs) + the 3 misalign display tasks
    # (em-misalign: agentic/leaking, agentic/murder, em8q/betley); each task -> one domain
    # 50 + inspect/ifeval (ifeval-canonical-battery) + inspect/bigcodebench_full + inspect/bfcl_full (capeval-full-tasks, 2026-09-23)
    assert len(all_tasks) == len(set(all_tasks)) == 53
    assert {"inspect/bigcodebench_200", "inspect/bfcl_200", "inspect/bigcodebench_full", "inspect/bfcl_full"} <= set(doms["inspect_gen"]["tasks"])
    assert doms["petri_behavior"]["tasks"] == [
        "petri/cyber_bio_chem_capabilities", "petri/self_preservation",
        "petri/delusion_sycophancy", "petri/defer_to_users",
        "petri/reasoning_unfaithfulness", "petri/instruction_evasion",
    ]
    assert doms["misalign"]["tasks"] == [
        "agentic/leaking", "agentic/murder", "em8q/betley"]
    assert doms["math"]["tasks"] == ["gsm8k_platinum_cap"]
    assert doms["code"]["tasks"] == ["humaneval_plus"]
    assert doms["instruction_following"]["tasks"] == ["ifeval"]
    assert doms["knowledge"]["tasks"] == ["tinyMMLU"]
    assert doms["math_s2parity"]["tasks"] == ["s2parity/gsm8k_0shot"]
    # PW3 (U2-impl 6): the row is KEPT so old stores validate, and marked
    # retired — the bespoke GSM8K behind it no longer runs forward; the
    # standard GSM8K rows file under inspect_gen (inspect/gsm8k)
    assert doms["math_s2parity"]["retired"] is True
    assert all("retired" not in d for n, d in doms.items() if n != "math_s2parity")
    assert "inspect/gsm8k" in doms["inspect_gen"]["tasks"]
    assert "inspect/gsm8k_paired200" in doms["inspect_gen"]["tasks"]
    assert doms["knowledge_s2parity"]["tasks"] == ["s2parity/mmlu_letter"]
    assert doms["reasoning"]["tasks"] == [
        "tinyArc", "bbh_cot_fewshot_date_understanding",
        "bbh_cot_fewshot_logical_deduction_three_objects",
        "bbh_cot_fewshot_hyperbaton",
    ]


# --------------------------------------------------------------------------
# Synthetic end-to-end: manifest -> cells -> rows -> stub fingerprint
# (pure canned data; passes today, pins the wire formats)
# --------------------------------------------------------------------------

def _mk_row(arm, domain, task, item, value, *, metric="acc", trial=1):
    return cs.ResultRow(
        sweep="proto_synthetic", run_id="run0", model_id="test-org/tiny-model",
        model_revision="0123456789abcdef0123456789abcdef01234567",
        arm_name=arm, kind="steering" if arm != "none" else "none",
        coeff=2.0 if arm != "none" else None,
        intervention_hash="a1b2c3d4e5f60718", engine="hf", tier=1, task=task,
        task_version="v1", domain=domain, item_id=item, trial_id=trial,
        seed=trial, attempt=0, position_order=None, metric=metric,
        value=value, extra_json=None,
        harness_config_hash="0f1e2d3c4b5a6978", env_digest="99aabbccddeeff00",
        slurm_job_id=None, created_at=1755000000.0, wall_clock_s=None,
    )


def _stub_fingerprint(rows, doms, margins):
    """Stand-in for stats/fingerprint.py — the OUTPUT SHAPE is the frozen
    part: per-arm domain statuses + screen status, computed
    here from point estimates only. Acceptance-role domains receive
    statuses but are EXCLUDED from the screen (r2-F6)."""
    by_arm = {}
    base = {(r.domain, r.task, r.item_id): r.value
            for r in rows if r.arm_name == "none"}
    for r in rows:
        if r.arm_name == "none":
            continue
        by_arm.setdefault(r.arm_name, {}).setdefault(r.domain, []).append(
            base[(r.domain, r.task, r.item_id)] - r.value)  # harm = base - arm
    out = {}
    for arm, per_dom in by_arm.items():
        statuses, screen = {}, "SCREEN-PASS"
        for dom, harms in per_dom.items():
            harm = sum(harms) / len(harms)
            margin = margins.get(dom, doms[dom]["margin"])
            status = "degraded" if harm > margin else "indeterminate"
            statuses[dom] = {"delta": -harm, "status": status}
            if doms[dom].get("role") != "acceptance" and harm > margin:
                screen = "SCREEN-FAIL"
        out[arm] = {"domains": statuses, "screen": screen}
    return out


def test_synthetic_end_to_end_rows_to_stub_fingerprint(tmp_path):
    m = _load_manifest_fixture_by_hand()
    doms = _domains_yaml()["domains"]
    # trial 0 reference rows (arm "none") + one steered arm, two domains —
    # math degraded beyond margin, knowledge within margin; plus an
    # acceptance-role domain degraded (must NOT flip the screen, r2-F6).
    rows = []
    for item in ("i0", "i1", "i2", "i3"):
        for dom, task in (("math", "gsm8k_platinum_cap"),
                          ("knowledge", "tinyMMLU"),
                          ("math_s2parity", "s2parity/gsm8k_0shot")):
            rows.append(_mk_row("none", dom, task, item, 1.0, trial=0))
    arm = m.ablations[0].name
    for item in ("i0", "i1", "i2", "i3"):
        rows.append(_mk_row(arm, "math", "gsm8k_platinum_cap", item, 0.0))
        rows.append(_mk_row(arm, "knowledge", "tinyMMLU", item, 1.0))
        rows.append(_mk_row(arm, "math_s2parity", "s2parity/gsm8k_0shot", item, 0.0))

    # rows serialize to the frozen JSONL wire format and round-trip
    cell = tmp_path / "cells" / "synthetic__tier1__t1__a0__a1b2c3d4__0f1e2d3c"
    cell.mkdir(parents=True)
    with open(cell / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(dataclasses.asdict(r), sort_keys=True) + "\n")
    back = [cs.ResultRow(**json.loads(line))
            for line in (cell / "rows.jsonl").read_text().splitlines()]
    assert back == rows

    fp = _stub_fingerprint(back, doms, m.margins)
    assert set(fp) == {arm}
    assert fp[arm]["screen"] in SCREENS
    assert fp[arm]["screen"] == "SCREEN-FAIL"  # math harm 1.0 > 0.02
    assert fp[arm]["domains"]["math"]["status"] == "degraded"
    assert fp[arm]["domains"]["knowledge"]["status"] == "indeterminate"
    # acceptance-role domain: status computed, screen untouched by it —
    # rerun with ONLY the parity domain degraded and assert SCREEN-PASS.
    only_parity = [r for r in back if r.domain in ("knowledge", "math_s2parity")]
    fp2 = _stub_fingerprint(only_parity, doms, m.margins)
    assert fp2[arm]["domains"]["math_s2parity"]["status"] == "degraded"
    assert fp2[arm]["screen"] == "SCREEN-PASS"


# --------------------------------------------------------------------------
# EvalPlus canned fixture pair (smoke input; passes today)
# --------------------------------------------------------------------------

def test_evalplus_fixture_pair_consistent():
    """Pins the generation-side samples.jsonl format and the (0.3.1-shaped)
    eval-results format the generation side hands to the execution wave,
    plus the row conversion: per
    task_id -> item rows (task="humaneval_plus", item_id=task_id, metrics
    EVALPLUS_METRICS). The ``pass@1_plus`` DEFINITION is evalplus's own
    headline — base AND plus (``evaluate.py`` ``new_correct``; PW10, named
    ``metric_def`` on the rows since PW12) — and this test's inline
    conversion encodes it DELIBERATELY: until PW12 it encoded ``plus_status``
    alone, the definition the exec wave's converter and the retained 618f9dd
    stores carried under the same metric identity (tinycap E19).
    The pair is the synthetic stand-in of tests/evalplus_synthetic.py (no
    benchmark text): the shapes are pinned here, never its solutions."""
    samples = [json.loads(line) for line in EP.samples_jsonl().splitlines()]
    results = EP.eval_results()
    assert [set(s) for s in samples] == [{"task_id", "solution"}] * 2
    assert set(results) == {"date", "hash", "eval"}
    assert sorted(results["eval"]) == sorted(s["task_id"] for s in samples)
    expected = {"HumanEval/0": {"pass@1_base": 1.0, "pass@1_plus": 1.0},
                "HumanEval/1": {"pass@1_base": 0.0, "pass@1_plus": 0.0}}
    def convert(item):
        base = 1.0 if item["base_status"] == "pass" else 0.0
        return {"pass@1_base": base,
                "pass@1_plus": 1.0 if (base and item["plus_status"] == "pass") else 0.0}

    for task_id, items in results["eval"].items():
        [item] = items  # one greedy sample per task in v0
        assert item["task_id"] == task_id
        converted = convert(item)
        assert set(converted) == set(EVALPLUS_METRICS)
        assert converted == expected[task_id]
    # the definitional case the fixture pair lacks: a plus-pass over a base-fail is NOT a plus pass
    assert convert({"task_id": "HumanEval/9", "base_status": "fail", "plus_status": "pass"}) == {
        "pass@1_base": 0.0, "pass@1_plus": 0.0}
    assert convert({"task_id": "HumanEval/9", "base_status": "pass", "plus_status": "fail"}) == {
        "pass@1_base": 1.0, "pass@1_plus": 0.0}


# --------------------------------------------------------------------------
# Seam tests: xfail until the owning WS implements the body. raises= is
# deliberate — any exception other than NotImplementedError, or a failed
# assert after implementation, FAILS (interface drift alarm).
# --------------------------------------------------------------------------

xfail_ws_a = pytest.mark.xfail(
    raises=NotImplementedError, strict=False,
    reason="Stage-0 skeleton seam — WS-A implements")


@xfail_ws_a
def test_seam_load_manifest():
    m = cm.load_manifest(FIXTURES / "manifest_synthetic.yaml")
    expected = _load_manifest_fixture_by_hand()
    # the loader resolves relative artifact paths against the manifest's
    # directory and stores them ABSOLUTE (review-fix contract 2026-08-15)
    expected.ablations[0] = dataclasses.replace(
        expected.ablations[0],
        artifact=str((FIXTURES / expected.ablations[0].artifact).resolve()))
    assert m == expected


@xfail_ws_a
def test_seam_load_manifest_refusals(tmp_path):
    raw = yaml.safe_load((FIXTURES / "manifest_synthetic.yaml").read_text())
    for corrupt in (
        {"confirmatory": True},          # v0 accepts false only
        {"mode": "control"},             # F20: target forbidden outside sweep
        {"base_revision": "main"},       # full SHA required
    ):
        with pytest.raises(Exception) as exc_info:
            cm.validate_manifest(_manifest_from_raw({**raw, **corrupt}))
        if isinstance(exc_info.value, NotImplementedError):
            raise exc_info.value  # still a stub -> xfail; a refusal passes
    # unknown keys are a LOADER refusal (they never reach the dataclass)
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(yaml.safe_dump({**raw, "unknown_key": 1}))
    with pytest.raises(Exception) as exc_info:
        cm.load_manifest(bad_yaml)
    if isinstance(exc_info.value, NotImplementedError):
        raise exc_info.value


def _manifest_from_raw(raw):
    target = raw.get("target")
    return cm.Manifest(
        sweep=raw["sweep"], mode=raw["mode"], base=raw["base"],
        base_revision=raw["base_revision"], engine=raw["engine"],
        tiers=raw["tiers"], confirmatory=raw["confirmatory"],
        margins=raw.get("margins", {}),
        target=None if target is None else cm.TargetSpec(
            primaries=[cm.PrimarySpec(**p) for p in target["primaries"]],
            secondary=[cm.PrimarySpec(**p) for p in target["secondary"]],
            off_target_exclusions=target["off_target_exclusions"],
            relevant_corpus=cm.CorpusRef(**target["relevant_corpus"]),
        ),
        ablations=[cm.AblationSpec(**a) for a in raw["ablations"]],
    )


@xfail_ws_a
def test_seam_cell_key_format():
    key = cs.cell_key(
        arm="arm_alpha_c2", tier_or_task="tier1", trial=0, attempt=0,
        intervention_hash="a1b2c3d4e5f60718",
        harness_config_hash="0f1e2d3c4b5a6978",
    )
    assert key == "arm_alpha_c2__tier1__t0__a0__a1b2c3d4__0f1e2d3c"


@xfail_ws_a
def test_seam_ts_metric():
    assert cs.ts_metric("model_ready") == "ts_model_ready"


@xfail_ws_a
def test_seam_canonical_json_and_hashes():
    a = cm.canonical_json({"b": 1, "a": [1, 2]})
    b = cm.canonical_json({"a": [1, 2], "b": 1})
    assert a == b  # stable across key order
    h = cm.sha256_hex16(b"capeval")
    assert len(h) == cm.HASH_HEX_LEN and int(h, 16) >= 0
    ih = cm.intervention_hash(
        {"kind": "steering", "layer": 15, "coeff": 2.0, "skip_bos": True},
        artifact_bytes=(FIXTURES / "vector_synthetic.safetensors").read_bytes(),
        sidecar_bytes=(FIXTURES / "vector_synthetic.json").read_bytes(),
    )
    ch = cm.intervention_hash({"kind": "control:random_vector", "seed": 7, "params": {}})
    assert len(ih) == len(ch) == cm.HASH_HEX_LEN and ih != ch
    hch = cs.harness_config_hash({"tasks": ["tinyMMLU"], "gen": {"temp": 0.0}})
    assert len(hch) == cm.HASH_HEX_LEN
    ed = cs.env_digest()
    assert len(ed) == cm.HASH_HEX_LEN


@xfail_ws_a
def test_seam_registry_load():
    root = Path(__file__).parents[1]
    reg = cd.Registry.load(root / "configs" / "capeval" / "domains.yaml")
    assert reg.version == 1
    assert tuple(d.name for d in reg.domains) == DOMAIN_NAMES_FROZEN
    assert reg.domain_of("gsm8k_platinum_cap").name == "math"
    assert reg.domain_of("s2parity/mmlu_letter").role == "acceptance"
    with pytest.raises(Exception):
        reg.domain_of("not_a_registered_task")
    gset = reg.gating_set("test-org/tiny-model", {}, ["instruction_following"])
    names = {d.name for d in gset}
    assert names == set(GATING_DOMAINS_FROZEN) - {"instruction_following"}
    assert not names & set(ACCEPTANCE_DOMAINS_FROZEN)


@xfail_ws_a
def test_seam_cellstore_roundtrip(tmp_path):
    store = cs.CellStore(tmp_path, "arm__tier1__t1__a0__a1b2c3d4__0f1e2d3c")
    row = _mk_row("arm", "math", "gsm8k_platinum_cap", "i0", 1.0)
    resolved = {
        "intervention_hash": row.intervention_hash,
        "harness_config_hash": row.harness_config_hash,
        "model_revision": row.model_revision, "engine": row.engine,
    }
    store.write_cell_manifest(resolved)
    store.validate_resume(resolved)  # no-op on match
    store.append_row(row)
    assert store.completed_item_ids() == {"i0"}
    assert not store.is_done()
    store.write_generation_complete()
    assert not store.is_done()  # results.json is the ONLY done marker (F8)
    store.write_results({"n_rows": 1})
    assert store.is_done()
    with pytest.raises(Exception) as exc_info:
        store.validate_resume({**resolved, "engine": "vllm"})
    assert not isinstance(exc_info.value, NotImplementedError)


@xfail_ws_a
def test_seam_consolidate(tmp_path):
    import pyarrow.parquet as pq
    store = cs.CellStore(tmp_path, "arm__tier1__t1__a0__a1b2c3d4__0f1e2d3c")
    store.write_cell_manifest({
        "intervention_hash": "a1b2c3d4e5f60718",
        "harness_config_hash": "0f1e2d3c4b5a6978",
        "model_revision": "0123456789abcdef0123456789abcdef01234567",
        "engine": "hf",
    })
    store.append_row(_mk_row("arm", "math", "gsm8k_platinum_cap", "i0", 1.0))
    store.write_results({"n_rows": 1})
    out = cs.consolidate(tmp_path)
    assert out.name == "rows.parquet"
    table = pq.read_table(out)
    assert tuple(table.column_names) == ROW_COLUMNS_FROZEN
    assert table.num_rows == 1
