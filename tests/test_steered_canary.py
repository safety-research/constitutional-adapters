"""scripts/sdfbase/steered_canary.py — the steered serving canary's pure helpers (CPU, no GPU, no
network): the derived pair replaces exactly the keys the canary owns, the staged cell carries the
registry's identity schema and never an `object.json` sidecar, the attestation checks judge the
facts they claim from the records they live in, and the sbatch holds the cluster rules."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import steered_canary as sc  # noqa: E402

RELU_H, LORA_H = "si_constv2_mix_full__psr_ce_all", "si_constv2_mix_full__lora_r4_ce"
N = 40


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------- derive
def test_derive_pair_replaces_exactly_the_owned_keys():
    cfg = {"explicit_cells": ["none", "prompted__safety_refusal"], "defenses_dir": "out/full22/x/cells_view/u0",
           "out_dir": "out/sdfbase_full22_x_u0", "private_run": "sdfbase_full22_x_u0", "model": "org/model", "completion_tokens": 512,
           "attacks": {"controls_max_new_tokens": 512}}
    man = {"petri": {"audit": "petri_full22_x_u0", "cells": ["none", "prompted__safety"], "chat_template_hash": "abcd", "serving_launch_flags": ["--x"]}}
    cells = ["none", "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce"]
    new_cfg, new_man, changes = sc.derive_pair(cfg, man, cells=cells, harness_cells=["none", RELU_H, LORA_H],
                                               defenses_dir="/views/canary", namespace="steered_canary_x")
    assert new_cfg["explicit_cells"] == cells and new_cfg["defenses_dir"] == "/views/canary"
    assert new_cfg["out_dir"] == "out/steered_canary_x" and new_cfg["private_run"] == "steered_canary_x"
    assert {k: v for k, v in new_cfg.items() if k not in changes["config"]} == {k: v for k, v in cfg.items() if k not in changes["config"]}
    assert new_man["petri"]["cells"] == ["none", RELU_H, LORA_H] and new_man["petri"]["audit"] == "steered_canary_x"
    assert new_man["petri"]["chat_template_hash"] == "abcd" and new_man["petri"]["serving_launch_flags"] == ["--x"]
    assert set(changes["config"]) == {"canary", "explicit_cells", "defenses_dir", "out_dir", "private_run"}
    assert set(changes["manifest"]) == {"petri.cells", "petri.audit"}
    # the canary MARKER rides first (common.sh's registry guard refuses a config carrying a top-level `canary:` key by name)
    assert list(new_cfg)[0] == "canary" and new_cfg["canary"]["namespace"] == "steered_canary_x"
    import yaml
    assert yaml.safe_dump(new_cfg, sort_keys=False).startswith("canary:\n")
    # the sources are untouched (data copies, never aliases)
    assert cfg["explicit_cells"] == ["none", "prompted__safety_refusal"] and man["petri"]["cells"] == ["none", "prompted__safety"]


def test_derive_pair_refuses_without_the_bare_anchor_first():
    cfg = {"explicit_cells": ["none"]}
    man = {"petri": {"cells": ["none"]}}
    with pytest.raises(sc.CanaryRefused, match="first cell must be 'none'"):
        sc.derive_pair(cfg, man, cells=["constv2-mix-full__psr_ce_all"], harness_cells=[RELU_H], defenses_dir="/v", namespace="n")


# --------------------------------------------------------------------------- stage
FP = "f" * 64
SEARCH, TRIAL = "hpo-native:deadbeef", "r1_t00"
FIT_MODEL, FIT_REV = "org/model-Base", "0" * 40
TARGET = "synth_fd"                      # a synthetic model key: the canary records it, nothing resolves it


def _trial_dir(tmp_path: Path, name: str, *, arm: str, si: str = "constv2-mix-10m") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "object.pt").write_bytes(b"OBJECT-BYTES-" + name.encode())
    (d / "summary.json").write_text(json.dumps({"si": si, "arm": arm, "model": FIT_MODEL, "model_revision": FIT_REV,
                                                 "trial": {"index": 0, "lr": 0.0053}, "round": 1, "search_id": SEARCH,
                                                 "fit_fingerprint": FP, "position": "all", "object_desc": {"object_kind": "relu_steer"}}))
    (d / "fit_trial_record.json").write_text(json.dumps({"trial_id": TRIAL, "search_id": SEARCH, "campaign": "two_si",
                                                          "record_path": "/records/x.jsonl", "status": "complete", "fit_fingerprint": FP}))
    (d / "object.pt.provenance.json").write_text(json.dumps({"kind": "psr", "layers": list(range(N))}))
    return d


def _pin(d: Path, *, sha: str | None = None, fp: str = FP, search: str = SEARCH, trial: str = TRIAL) -> dict:
    return sc.parse_pin(f"{sha or _sha((d / 'object.pt').read_bytes())}:{fp}:{search}/{trial}")


def _stage(view, cell, src, *, pin=None, blob=None, n=N, **kw):
    return sc.stage_cell(view, cell, src, target=TARGET, completion_tokens=512, pin=pin or _pin(src),
                         fit_model=FIT_MODEL, fit_revision=FIT_REV, expected_n_layers=n, load_blob=blob or _blob("psr"), **kw)


def _blob(kind: str, *, positions: str = "all", n: int = N):
    def load(_p: Path):
        return {"kind": kind, "positions": positions, "layers": list(range(n))}
    return load


def test_stage_cell_writes_the_identity_schema_and_the_provenance_beside_the_view(tmp_path):
    from sisteer.eval.train_budget import staged_cell_problems

    src = _trial_dir(tmp_path, "relu_r1_t00", arm="psr_ce_all")
    view = tmp_path / "view"
    cell = "constv2-mix-full__psr_ce_all"
    rec = _stage(view, cell, src)
    cell_dir = view / cell
    assert sorted(p.name for p in cell_dir.iterdir()) == ["object.pt", "staged_from.json", "train_budget_meta.json"]
    assert not (cell_dir / "object.json").exists()
    assert (cell_dir / "object.pt").read_bytes() == (src / "object.pt").read_bytes()
    meta = json.loads((cell_dir / "train_budget_meta.json").read_text())
    assert meta["cell_id"] == cell and meta["object_sha256"] == _sha((src / "object.pt").read_bytes()) and meta["completion_tokens"] == 512
    assert meta["canary"]["fit_si"] == "constv2-mix-10m" and meta["canary"]["served_cell_si"] == "constv2-mix-full"
    assert "harness vocabulary" in meta["canary"]["relabel"] and meta["canary"]["trial_id"] == TRIAL
    assert meta["canary"]["pin"]["fit_fingerprint"] == FP and meta["canary"]["fit_checkpoint_expected"] == {"model": FIT_MODEL, "revision": FIT_REV}
    assert staged_cell_problems(cell_dir / "object.pt", cell, require_sidecar=True) == []
    staged = json.loads((cell_dir / "staged_from.json").read_text())
    assert staged["lane"] == "canary" and staged["object_sha256"] == meta["object_sha256"] and staged["source"] == str(src)
    prov = view / sc.PROVENANCE_DIR / cell
    assert sorted(p.name for p in prov.iterdir()) == ["fit_trial_record.json", "object.pt.provenance.json", "summary.json"]
    assert rec["canary"]["object_kind"] == "psr" and rec["canary"]["object_layers"] == list(range(N))
    # idempotent: the same bytes stage again; different bytes (pinned as such) refuse (nothing is overwritten)
    pin = _pin(src)
    _stage(view, cell, src, pin=pin)
    (src / "object.pt").write_bytes(b"DIFFERENT")
    with pytest.raises(sc.CanaryRefused, match="refusing to overwrite"):
        _stage(view, cell, src, pin=_pin(src))
    # and the ORIGINAL pin now refuses the changed bytes before anything is copied
    with pytest.raises(sc.CanaryRefused, match="not the frozen HPO choice.*sha256"):
        _stage(view, cell, src, pin=pin)


@pytest.mark.parametrize("cell,kind,positions,n,msg", [
    ("constv2-mix-full__psr_ce_all", "lora_all", "all", N, "stores kind 'lora_all'"),
    ("constv2-mix-full__lora_r4_ce", "psr", "all", N, "stores kind 'psr'"),
    ("constv2-mix-full__psr_ce_all", "psr", "response", N, "all-position fits only"),
    ("constv2-mix-full__psr_ce_all", "psr", "all", N - 1, "not every one of 40 blocks"),
])
def test_stage_cell_refuses_the_wrong_object(tmp_path, cell, kind, positions, n, msg):
    arm = cell.split("__", 1)[1]
    src = _trial_dir(tmp_path, "t", arm=arm)
    with pytest.raises(sc.CanaryRefused, match=msg):
        _stage(tmp_path / "view", cell, src, blob=_blob(kind, positions=positions, n=n), n=N)


def test_stage_cell_refuses_a_summary_naming_another_arm(tmp_path):
    src = _trial_dir(tmp_path, "t", arm="lora_r4_ce")
    with pytest.raises(sc.CanaryRefused, match="arm 'lora_r4_ce' != the cell's 'psr_ce_all'"):
        _stage(tmp_path / "view", "constv2-mix-full__psr_ce_all", src)


# the frozen-HPO-choice pin: a typo selecting another valid trial of the same arm must refuse
@pytest.mark.parametrize("mutate,msg", [
    (lambda d: (d / "object.pt").write_bytes(b"ANOTHER-TRIAL"), "sha256 .* != the pin's"),
    (lambda d: (d / "summary.json").write_text(json.dumps({**json.loads((d / "summary.json").read_text()), "fit_fingerprint": "e" * 64})), "summary.json fit_fingerprint"),
    (lambda d: (d / "fit_trial_record.json").write_text(json.dumps({**json.loads((d / "fit_trial_record.json").read_text()), "trial_id": "r1_t07"})), "trial_id 'r1_t07'"),
    (lambda d: (d / "fit_trial_record.json").write_text(json.dumps({**json.loads((d / "fit_trial_record.json").read_text()), "search_id": "hpo-native:other"})), "search_id 'hpo-native:other'"),
    (lambda d: (d / "summary.json").write_text(json.dumps({**json.loads((d / "summary.json").read_text()), "model": "org/other-Base"})), "model 'org/other-Base' != the fit checkpoint"),
    (lambda d: (d / "summary.json").unlink(), "has no summary.json"),
    (lambda d: (d / "fit_trial_record.json").unlink(), "has no fit_trial_record.json"),
])
def test_stage_cell_refuses_a_trial_that_is_not_the_pinned_choice(tmp_path, mutate, msg):
    src = _trial_dir(tmp_path, "t", arm="psr_ce_all")
    pin = _pin(src)
    mutate(src)
    with pytest.raises(sc.CanaryRefused, match=msg):
        _stage(tmp_path / "view", "constv2-mix-full__psr_ce_all", src, pin=pin)
    assert not (tmp_path / "view" / "constv2-mix-full__psr_ce_all" / "object.pt").exists()


def test_parse_pin_shapes():
    pin = sc.parse_pin(f"{'a' * 64}:{'b' * 64}:hpo-native:abc/r1_t00")
    assert pin == {"object_sha256": "a" * 64, "fit_fingerprint": "b" * 64, "search_id": "hpo-native:abc", "trial_id": "r1_t00"}
    for bad in ("abc", f"{'a' * 64}:{'b' * 64}", f"{'a' * 63}:{'b' * 64}:s/t", f"{'a' * 64}:{'b' * 64}:s"):
        with pytest.raises(sc.CanaryRefused):
            sc.parse_pin(bad)


def test_arm_kinds_the_canary_stages_are_the_registry_adapters():
    from sisteer.capeval.petri import registry_adapter as ra
    assert ra.ARM_OBJECT_KINDS["psr_ce_all"] == "psr" and ra.ARM_OBJECT_KINDS["lora_r4_ce"] == "lora_all"
    assert ra.harness_cell_id("constv2-mix-full__psr_ce_all") == RELU_H
    assert ra.harness_cell_id("constv2-mix-full__lora_r4_ce") == LORA_H


# --------------------------------------------------------------------------- attestation checks
def _records():
    endpoint = {
        "engine": "lora_native", "serving_engine": "lora_native", "attested_enforce_eager": False, "attested_cudagraph_mode": "FULL_AND_PIECEWISE",
        "max_model_len": 4096, "max_num_seqs": 56, "attested_max_num_seqs": 56, "geometry": {"adjustments": [{"reason": "lora_kernel_index_ceiling"}]},
        "lora_cells": [RELU_H, LORA_H], "cells_skipped": [],
        "adapter_parity_gate": {"status": "verified", "pairs": [{"adapter": LORA_H, "rank": 4, "set": "attn_mlp", "status": "verified"}],
                                "residual": {"carriers": [{"adapter": RELU_H, "kind": "psr", "rank": 1, "set": "wo"}]}},
        "residual_slot": {"armed": True, "cells": {RELU_H: {"kind": "psr", "position_mode": "all"}}},
        "adapters": {RELU_H: {"name": "relu_adapter", "r": 1, "petri_source_kind": "psr"},
                     LORA_H: {"name": "lora_adapter", "r": 4, "petri_source_kind": "lora_all"}},
    }
    attestation = {
        "schema_version": 2,
        "residual_slot": {"armed": True, "rpc_method": "sisteer_residual_report", "n_workers": 1, "expected": {"n_slots": 1, "adapters": {"x": 1}},
                          "worker_reports": [{"armed": True, "op_registered": True, "n_layers_hooked": N, "n_layers_model": N,
                                              "hooked_boundaries": list(range(N)), "patched_classes": {"DecoderLayer": {"forward_wrapped": True}},
                                              "bank": {"op_invocations": 12}, "ledger_size": 0, "ledger_fault": None,
                                              "slots": {"7": {"fingerprint": "fp-1"}}}]},
        "adapters": {"relu_adapter": {"petri_source_kind": "psr", "residual_fingerprint": "fp-1", "position_mode": "all", "r": 1, "lora_int_id": 7},
                     "lora_adapter": {"petri_source_kind": "lora_all", "r": 4, "lora_int_id": 3}},
    }
    controls = {"passed": True, "order": ["noop_certificate", "steering_bites"],
                "controls": {"noop_certificate": {"passed": True, "verdict": "zero adapter == base on every probe"},
                             "steering_bites": {"passed": True, "verdict": "bites", "cells": {RELU_H: {"n_differing": 5, "n_probes": 5},
                                                                                                LORA_H: {"n_differing": 5, "n_probes": 5}}}}}
    return endpoint, attestation, controls


def test_attestation_checks_pass_on_a_coherent_record_and_name_the_facts():
    endpoint, attestation, controls = _records()
    checks, facts = sc.attestation_checks(endpoint=endpoint, attestation=attestation, controls=controls, relu_h=RELU_H, lora_h=LORA_H, n_layers=N)
    assert [c["name"] for c in checks if not c["ok"]] == []
    assert facts["worker_residual"]["n_layers_hooked"] == N and facts["carrier_attested"]["position_mode"] == "all"
    assert facts["noop_certificate"]["passed"] is True and facts["steering_bites"]["cells"][RELU_H]["n_differing"] == 5
    assert facts["adapter_parity_gate"]["status"] == "verified"


def test_attestation_checks_read_the_gate_from_cells_served_when_the_endpoint_lacks_it():
    endpoint, attestation, controls = _records()
    gate = endpoint.pop("adapter_parity_gate")
    checks, facts = sc.attestation_checks(endpoint=endpoint, attestation=attestation, controls=controls, relu_h=RELU_H, lora_h=LORA_H,
                                          n_layers=N, cells_served={"adapter_parity_gate": gate})
    assert [c["name"] for c in checks if not c["ok"]] == [] and facts["adapter_parity_gate"] == gate


@pytest.mark.parametrize("mutate,failing", [
    (lambda e, a, c: a["residual_slot"]["worker_reports"][0].update(n_layers_hooked=N - 1, hooked_boundaries=list(range(N - 1))), "worker_hooked_every_block"),
    (lambda e, a, c: a["residual_slot"].update(armed=False), "attestation_residual_slot_armed"),
    (lambda e, a, c: e["adapters"][RELU_H].update(petri_source_kind="lora", r=4), "carrier_is_a_rank1_residual_entry"),
    (lambda e, a, c: c.update(passed=False, failed_control="steering_bites"), "positive_controls_passed"),
    (lambda e, a, c: c["controls"]["steering_bites"]["cells"][RELU_H].update(n_differing=0), "steering_bites_on_the_carrier"),
    (lambda e, a, c: e["adapter_parity_gate"]["residual"]["carriers"].clear(), "parity_gate_lists_the_carrier"),
    (lambda e, a, c: e.update(cells_skipped=[{"cell": RELU_H}]), "no_cell_skipped"),
    (lambda e, a, c: a["residual_slot"]["worker_reports"][0]["bank"].update(op_invocations=0), "worker_op_ran"),
    (lambda e, a, c: a["adapters"]["relu_adapter"].update(residual_fingerprint="other"), "worker_slot_fingerprint_matches"),
])
def test_attestation_checks_fail_by_name(mutate, failing):
    endpoint, attestation, controls = _records()
    mutate(endpoint, attestation, controls)
    checks, _ = sc.attestation_checks(endpoint=endpoint, attestation=attestation, controls=controls, relu_h=RELU_H, lora_h=LORA_H, n_layers=N)
    assert failing in [c["name"] for c in checks if not c["ok"]]


def _gate_output(*, with_carrier=True):
    """geometry.standup_adapter_parity_gate's return shape (status, pairs, record, overlay, residual block)."""
    return {"status": "verified", "overlay": f"/site/geometry/{TARGET}.json", "record_predicate": "default:hpo_probe_status.probe_state_record",
            "pairs": [{"adapter": LORA_H, "rank": 4, "set": "attn_mlp", "status": "verified"}],
            "record": {"dir": "/record", "state": "READY", "predicate": "live"},
            "residual": {"carriers": ([{"adapter": RELU_H, "kind": "psr", "rank": 1, "set": "wo"}] if with_carrier else []),
                         "certificate": "the residual op's own: attestation residual_slot.armed with every served carrier, the noop certificate and the positive controls"}}


def test_the_parity_gate_fact_is_the_standups_record_and_the_check_reads_the_carrier_from_it():
    """818207: the check read an endpoint WITHOUT adapter_parity_gate (the verdict had lived only in the pre-launch
    cells_served.json) and failed by name while the standup log showed the gate verified. The standup now persists the
    gate in endpoint.json + cells_served.json (LoraStandup.adapter_parity_gate, a core record key); the fact IS that
    block, and the check passes iff the carrier is listed in its residual block."""
    from sisteer.capeval.serving import native_standup as ns
    import dataclasses
    assert "adapter_parity_gate" in ns._CORE_RECORD_KEYS
    assert "adapter_parity_gate" in {f.name for f in dataclasses.fields(ns.LoraStandup)}
    src = (REPO / "src" / "sisteer" / "capeval" / "serving" / "native_standup.py").read_text()
    assert src.count('"adapter_parity_gate": self.adapter_parity_gate,') == 2       # endpoint() AND _cells_served()
    assert "adapter_parity_gate=adapter_parity_gate," in src                          # handed to the handle at construction
    endpoint, attestation, controls = _records()
    endpoint["adapter_parity_gate"] = _gate_output()
    checks, facts = sc.attestation_checks(endpoint=endpoint, attestation=attestation, controls=controls, relu_h=RELU_H, lora_h=LORA_H, n_layers=N)
    assert facts["adapter_parity_gate"] == _gate_output() and [c["name"] for c in checks if not c["ok"]] == []
    endpoint["adapter_parity_gate"] = _gate_output(with_carrier=False)
    checks, _ = sc.attestation_checks(endpoint=endpoint, attestation=attestation, controls=controls, relu_h=RELU_H, lora_h=LORA_H, n_layers=N)
    assert "parity_gate_lists_the_carrier" in [c["name"] for c in checks if not c["ok"]]
    # an endpoint lacking the block entirely (818207's shape) fails the same check by name, never silently passes
    endpoint.pop("adapter_parity_gate")
    checks, facts = sc.attestation_checks(endpoint=endpoint, attestation=attestation, controls=controls, relu_h=RELU_H, lora_h=LORA_H, n_layers=N)
    assert facts["adapter_parity_gate"] == {} and "parity_gate_lists_the_carrier" in [c["name"] for c in checks if not c["ok"]]


# --------------------------------------------------------------------------- windows
def test_window_plan_is_b_outer_arm_inner_over_the_three_arms():
    assert sc.window_plan([32, 128]) == [(32, "base"), (32, "lora"), (32, "relu"), (128, "base"), (128, "lora"), (128, "relu")]
    assert sc.parse_batches("32,128") == [32, 128]
    with pytest.raises(sc.CanaryRefused):
        sc.parse_batches("0")


def test_rows_problems_name_the_missing_errored_and_unread_windows():
    plan = sc.window_plan([32])
    rows = [{"arm": "base", "B": 32, "tok_s": 1000.0, "errors": 0, "requests_done": 40},
            {"arm": "lora", "B": 32, "tok_s": None, "errors": 2, "error_sample": "HTTP 500", "requests_done": 3}]
    p = sc.rows_problems(rows, plan)
    assert any("relu@B32 missing" in x for x in p)
    assert any("lora@B32: 2 request errors" in x for x in p) and any("lora@B32: no engine tok/s" in x for x in p)
    assert sc.rows_problems([{"arm": a, "B": 32, "tok_s": 1.0, "errors": 0, "requests_done": 1} for a in sc.ARMS], plan) == []
    assert set(sc.summarize_rows(rows)) == {"base@B32", "lora@B32"}


# --------------------------------------------------------------------------- the sbatch
def test_sbatch_holds_the_cluster_rules_and_drives_the_three_steps():
    text = (REPO / "scripts" / "sdfbase" / "steered_canary.sbatch").read_text()
    assert "#SBATCH --qos=high" in text and "#SBATCH --gres=gpu:1" in text and "#SBATCH --mem=128G" in text
    assert "--export=" not in text and "--mem=0" not in text and "CUDA_VISIBLE_DEVICES=" not in text
    assert "#SBATCH --no-requeue" in text and "#SBATCH --open-mode=append" in text
    assert 'export F22_SERVING_PATH=lora_native' in text and 'f22_init "$WAVE_SHA"' in text
    # the production pair is derived read-only, never through f22_resolve_config (which registers)
    assert "f22_resolve_config" not in text.replace("# THE CANARY NEVER REGISTERS", "").split("\nf22_init")[1].split("\n")[0:0] or True
    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert "f22_resolve_config" not in body
    assert 'CFG="configs/sdfbase_full22_${TAG}.yaml"; MANIFEST="configs/capeval/manifests/petri_full22_${TAG}.yaml"' in text
    assert '[ -f "$CFG" ] && [ -f "$MANIFEST" ] || f22_fatal' in text
    for step in ("steered_canary.py stage", "steered_canary.py derive", "steered_canary.py run"):
        assert step in text
    assert 'srun --ntasks=1 "$SPY" scripts/sdfbase/steered_canary.py run' in text
    assert "--max-model-len 4096" in text and '--expect-n-layers "$N_LAYERS"' in text
    assert "f22_result" in text and "trap cleanup EXIT SIGTERM SIGINT SIGQUIT" in text
    # the registry: the canary flag is exported BEFORE f22_init / f22_resolve_config, and the exit asserts no pending file
    assert text.index("export F22_REGISTRY_CANARY=1") < text.index('f22_init "$WAVE_SHA"')
    assert "#SBATCH --job-name=scanary" in text   # the canary's own name: common.sh's cross-check (the identity is the script + the marker)
    assert 'f22_serving_venv_for_key "$KEY"' in text   # the serving venv variant per model key (qwen35_* -> the GDN kernel overlay)
    assert 'local pend="$SHARED/registry/pending/${SLURM_JOB_ID:-manual}.json"' in text and "check registry_untouched" in text
    assert text.index("check registry_untouched") < text.index('f22_result "steered-canary')
    # the frozen HPO choice rides as a pin per cell and the standup is handed the OPERATOR's digest, never a re-derived one
    assert '--pin "$RELU_CELL=$RELU_PIN"' in text and '--pin "$LORA_CELL=$LORA_PIN"' in text
    assert '--expect-fit-model "$FIT_REPO" --expect-fit-revision "$FIT_REV"' in text
    assert '[ "$pin" = "$disk" ] && [ "$pin" = "$rec" ] || f22_fatal' in text
    # the staged site overlay is named explicitly (a hand-submitted job carries no funnel env; job 817770 sized from the provisional row)
    assert 'export SISTEER_GEOMETRY_DIR="${SISTEER_GEOMETRY_DIR:-$SHARED/exp/reproduce/geometry}"' in text
    assert '[ -s "$SISTEER_GEOMETRY_DIR/$KEY.json" ] || f22_fatal' in text
