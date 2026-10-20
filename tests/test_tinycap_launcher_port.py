"""IFEval canonical battery — CHECKPOINT 1 of the launcher port:
`scripts.modal.tinycap_served` IMPORTS with the 699f81fd part-A
modules in place (the sibling ``tinycap_images`` and the serving lane's
``mamba_backend`` / ``vllm_server`` text-only table), the cfe89d51 launcher hunks
exist in their PARTIAL, dependency-ordered port — ``configuration_function``,
``run_configuration_nemotron``, ``launch_configurations``, ``_nemotron_probe``
PARAMETERISED BY THE PRODUCTION ROW, ``nemotron_backend_probe``, the attestation
key and the admission — and none of the controller / HumanEval+ state machine /
spend-ledger symbols came along.

Merged onto main (bcb-bfcl-200-widen, which ported the same donor "minus the GLM gate"):
the Nemotron image function, the per-row function router, the post-standup gates and the
worker body (``_configuration_worker``) are main's ONE definition each, with the lane's
extra behaviour folded in (the M-RoPE keys and gate, the IFEval rows' text-only pin, the
``vendor_preset`` recorder expectation); the lane's opt-in ``--enable-log-requests`` is
main's FIXED flag on every standup.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sisteer.capeval.tinycap import contracts as C

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_tinycap_launcher as TL  # noqa: E402  (the modal stub + the by-path loader)

REPO = Path(__file__).resolve().parents[1]
NEMO = C.IFEVAL_SLUGS[("nemotron3nano4b", "main")]
QWEN = C.IFEVAL_SLUGS[("qwen35_4b", "main")]
GEMMA = C.IFEVAL_SLUGS[("gemma4e4b", "it")]
LLAMA = C.IFEVAL_SLUGS[("llama8b", "it")]


@pytest.fixture(scope="module")
def L():
    return TL._load_launcher(TL._stub_modal(), app_name=TL.TEST_APP_NAME, name="tinycap_served_port_under_test")


# --------------------------------------------------------------------------- #
# CHECKPOINT 1: the module imports; the part-A prerequisites are in the tree    #
# --------------------------------------------------------------------------- #


def test_the_launcher_imports_with_its_module_load_prerequisites(L):
    """The cfe89d51 hunks import ``tinycap_images`` and read the serving parser's
    text-only table at module load (P45): both resolve, so the launcher loads."""
    assert L.tinycap_images.__name__ == "tinycap_images"
    assert (REPO / "scripts" / "modal" / "tinycap_images.py").is_file()
    assert (REPO / "src" / "sisteer" / "capeval" / "serving" / "mamba_backend.py").is_file()
    assert (REPO / C.CLIENT_LOCK_TXT).is_file()
    assert (REPO / "configs" / "capeval" / "mamba_backend_contract.json").is_file()
    # the serving parser's verified-token table: qwen3_5 alone — glm4v is out of it on main (the zero-GLM-footprint
    # rule; the IFEval matrix has no GLM row), so a glm4v row's M-RoPE gate is the only GLM-specific gate left
    assert set(L.TEXT_ONLY_MM_FAMILIES) == {"qwen3_5", "qwen3_5_moe"}   # + the routed-experts member of the family (its serving tables restored 2026-10-08)
    assert L.STANDUP_TEXT_ONLY_MM_FLAG == "--text-only-mm"
    assert L.STANDUP_MAMBA_CONTRACT_FLAG == "--mamba-backend-contract"
    # the image variant is bound (or its error recorded — never a silent fallback)
    assert hasattr(L, "image_nemotron") and hasattr(L, "IMAGE_NEMOTRON_ERROR")
    for name in ("configuration_function", "run_configuration_nemotron", "launch_configurations",
                 "_nemotron_probe", "nemotron_backend_probe", "nemotron_attestation_key",
                 "nemotron_attestation_path", "probe_geometry_cfg", "full22_attestation_record",
                 "full22_endpoint_refusal", "_configuration_worker", "configuration_kwargs", "PEER_SLUG",
                 "NEMO_BACKEND_ROOT",
                 "FLAG_NEMOTRON_PROBE", "FLAG_GEOMETRY", "PROBE_GEOMETRIES"):
        assert hasattr(L, name), name


def test_none_of_the_dropped_battery_machinery_came_along(L):
    """EXCLUDED by the plan: the two-stage HumanEval+ state machine, the GPU-hour
    ledger / 45-hour hard stop, the orchestrator DAG and their files."""
    for name in ("write_open_record", "write_ended_record", "budget_gate_problems", "parse_attempt_ids",
                 "ledger_reserved_ids", "sandbox_pending_controls", "evalplus_task_sandbox_pending",
                 "derive_ladder", "ladder_shell", "ladder_configuration_from_base", "LADDER_SUFFIX",
                 "BUDGET_GATE_FLAG", "FLAG_SPAWNED_LOG", "_run_configuration_body", "resolve_row_pins",
                 "staged_objects_problems", "derived_objects_cpu_problems",
                 "glm_nll_report_problems", "set_app_tags", "trusted_store_binding"):
        assert not hasattr(L, name), name
    # `full22_pins_by_cell` is NOT on that list any more: main's bcb-bfcl-200-widen ported it for the bcbw
    # lane's preflight `full22_pin_match` (it reads the STAGED tree's full22 pins for a lane row) — the IFEval
    # rows are not full22 rows, so it never runs for them
    assert callable(L.full22_pins_by_cell)
    for rel in ("scripts/tinycap_full22_orchestrate.py", "scripts/tinycap_modal_alloc.py",
                "scripts/tinycap_wave_spend.py", "tests/test_tinycap_full22_orchestrate.py",
                "tests/test_tinycap_modal_alloc.py", "tests/test_tinycap_wave_spend.py"):
        assert not (REPO / rel).exists(), rel
    src = TL.LAUNCHER.read_text()
    assert "SANDBOX_PENDING" not in src and "BUDGET_HARD_STOP" not in src and "attempt_ids" not in src


# --------------------------------------------------------------------------- #
# the seven IFEval slugs and the function router                               #
# --------------------------------------------------------------------------- #


def test_the_seven_ifeval_slugs_are_registry_rows_with_the_pins_off_and_the_volume_bank_views():
    # + the six batch-64 seed lane views of qwen35_4b = thirteen IFEval slugs
    assert set(C.IFEVAL_SLUGS.values()) <= set(C.MODEL_SLUGS)
    assert len(C.IFEVAL_SLUGS) == 13 and C.IFEVAL_MODELS == ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b")
    assert [v for (k, v) in C.IFEVAL_SLUGS if k == "qwen35_4b"] == ["main", "it", *C.IFEVAL_LANE_VIEWS]
    for (key, view), slug in C.IFEVAL_SLUGS.items():
        row = C.model_row(slug)
        hf, rev = C.IFEVAL_MODEL_PINS[key]
        assert (row.hf, row.rev, row.model_key) == (hf, rev, key) and len(rev) == 40
        assert row.family == C.FAMILY_BY_MODEL_KEY[key]
        assert row.thinking_modes == (C.THINKING_OFF,)
        assert row.objects_dir == C.full22_objects_dir(key, view) == f"{C.VOL_OBJECTS_FULL22}/{key}/{view}"
        assert not C.is_full22_row(row)                                # the widen-lane gates skip the IFEval rows
        assert row.cfg == f"configs/sdfbase/ifeval/sdfbase_full22_{C.ifeval_tag(key, view)}.ifeval.yaml"
        assert C.is_ifeval_row(row) and view in C.IFEVAL_VIEWS         # the instrument views + the seed lane views
        assert (view in C.BANK_VIEWS) == (view not in C.IFEVAL_LANE_VIEWS)
        assert slug.endswith(f"-sdfbase-{view}") and "/" not in slug
    # nemotron_h's S5 live PASS at 8365992 (2026-09-23): the IFEval nemotron row inherits the full22 nemotron manifest through the same map
    assert C.model_row(NEMO).family == "nemotron_h" and C.model_row(NEMO).petri_manifest == C.PETRI_MANIFEST_FULL22_NEMOTRON3NANO4B
    # the qwen3_5 pairing is pinned (qwen3_xml), so the qwen IFEval rows
    # carry the full22 qwen manifest through `_ifeval_row`'s family map — the same target + pin as the row
    assert C.model_row(QWEN).petri_manifest == C.PETRI_MANIFEST_FULL22_QWEN35_4B
    assert C.model_row(GEMMA).petri_manifest == C.PETRI_MANIFEST_GEMMA4E4B
    assert C.model_row(LLAMA).petri_manifest == C.PETRI_MANIFEST_TWOSI
    # the fit base where the objects were TRANSFERRED (read off the staged banks' sidecars)
    assert C.model_row(GEMMA).base_hf == "google/gemma-4-E4B" and C.model_row(QWEN).base_hf == "Qwen/Qwen3.5-4B-Base"
    assert C.model_row(NEMO).base_hf is None                              # fit on the served model itself
    assert not C.is_ifeval_row(C.model_row("gemma4e4b")) and not C.is_ifeval_row(C.model_row("qwen35_4b"))
    assert C.ifeval_tag("llama8b", "it") == "llama8b_it" and C.ifeval_tag("llama8b", "main") == "llama8b"
    with pytest.raises(ValueError):
        C.ifeval_tag("llama8b", "fullconst")


def test_the_ifeval_bank_views_are_the_staged_full22_banks_never_an_unstaged_cells_view_path():
    """The wave shakedown (2026-09-13): the seven slugs named `/vol/objects/sdfbase/<key>/cells_view/<view>`,
    a directory that was never staged — the volume holds the banks at `/vol/objects/sdfbase/full22/<key>/<view>`
    (every bank's `staged_manifest.json` names that path as its `objects_dir`; the six bcbw rows serve from it),
    so every fitted-cell preflight and lens standup would have refused by path. The slug names the bank that
    exists, spelled by the ONE constructor the bcbw rows use, and the widen-lane predicate still skips it."""
    for (key, view), slug in C.IFEVAL_SLUGS.items():
        row = C.model_row(slug)
        assert row.objects_dir == C.ifeval_objects_dir(key, view) == C.full22_objects_dir(key, view)
        assert row.objects_dir.startswith(f"{C.VOL_OBJECTS_FULL22}/") and "cells_view" not in row.objects_dir
        assert not C.is_full22_row(row) and C.is_ifeval_row(row)
    # the six bcbw rows are unchanged: the same banks, the predicate still names exactly them
    assert [s for s in C.MODEL_SLUGS if C.is_full22_row(C.model_row(s))] == list(C.BCBW_SLUGS)
    with pytest.raises(ValueError):
        C.ifeval_objects_dir("gemma4e4b", "nope")


def test_configuration_function_routes_a_nemotron_h_row_to_the_nemotron_image_function(L):
    from sisteer.capeval.serving import mamba_backend as mb
    assert mb.NEMOTRON_FAMILY == "nemotron_h"
    assert L.configuration_function(C.model_row(NEMO)) is L.run_configuration_nemotron
    for slug in (QWEN, GEMMA, LLAMA, "gemma4e4b", "qwen35_4b"):
        assert L.configuration_function(C.model_row(slug)) is L.run_configuration
    # capeval-full-tasks: a configuration whose controls name a FULL leg runs on the row's `*_full` sibling (the
    # same image, the FULL legs' function timeout contracts.FULL_FUNCTION_TIMEOUT_S); a 200-item control keeps
    # the function exactly what it was
    assert L.configuration_function(C.model_row(NEMO), [C.CONTROL_BFCL_FULL]) is L.run_configuration_nemotron_full
    assert L.configuration_function(C.model_row(QWEN), [C.CONTROL_BIGCODEBENCH_FULL]) is L.run_configuration_full
    assert L.configuration_function(C.model_row(NEMO), [C.CONTROL_IFEVAL]) is L.run_configuration_nemotron
    assert L.configuration_function(C.model_row(QWEN), [C.CONTROL_IFEVAL]) is L.run_configuration
    # the four GPU functions -- the two 200-item ones and their capeval-full-tasks `*_full` siblings -- share one
    # signature and one body (main's _configuration_worker, a PLAIN function); each is a one-line return through it
    gpu_functions = ("run_configuration", "run_configuration_nemotron", "run_configuration_full",
                     "run_configuration_nemotron_full")
    for name in gpu_functions:
        assert inspect.signature(getattr(L, name)) == inspect.signature(L.run_configuration)
    src = TL.LAUNCHER.read_text()
    worker_return = "return _configuration_worker(cfg, sha, limit=limit"
    for name in gpu_functions:
        body = src.split(f"\ndef {name}(", 1)[1].split("\n\n\n", 1)[0]
        assert worker_return in body, name
    assert src.count(worker_return) == len(gpu_functions) == 4
    assert "_run_and_print" not in src                                  # the lane's copy of the body is gone
    # main routes through it, with the SELECTED controls so the router can pick the `*_full` variant
    assert "recs += fan_out(configuration_function(row, sel_controls), cfgs," in src


# --------------------------------------------------------------------------- #
# the standup argv flags and the post-standup gates                            #
# --------------------------------------------------------------------------- #


def _cfg(L, slug=QWEN, cell="none"):
    (cfg,) = L.plan_configurations(cells=[cell], modes=[C.THINKING_OFF], controls=[C.CONTROL_IFEVAL], model=slug,
                                   sampling_mode=C.SAMPLING_VENDOR_PRESET, samples=C.IFEVAL_SAMPLES_PER_ITEM)
    return cfg


def test_standup_argv_composes_the_text_only_pin_the_contract_and_the_peers_over_budget(L):
    cfg = _cfg(L)
    geom, _ = L.served_geometry(cfg, 1)
    srv = Path("<srv>")
    base = L._standup_cmd("<tree>", "<cfg>", "none", "<ddir>", srv, srv / "STOP", geom, 1024, [])
    pinned = L._standup_cmd("<tree>", "<cfg>", "none", "<ddir>", srv, srv / "STOP", geom, 1024, [], text_only_mm=True)
    assert pinned == base + [L.STANDUP_TEXT_ONLY_MM_FLAG, "on"]
    nemo = L._standup_cmd("<tree>", "<cfg>", "none", "<ddir>", srv, srv / "STOP", geom, 1024, [],
                          mamba_contract="<tree>/configs/capeval/mamba_backend_contract.json")
    assert nemo == base + [L.STANDUP_MAMBA_CONTRACT_FLAG, "<tree>/configs/capeval/mamba_backend_contract.json"]
    peer = L._standup_cmd("<tree>", "<cfg>", "none", "<ddir>", srv, srv / "STOP", geom, 1024, [], allow_over_budget=True)
    assert peer == base + ["--allow-over-budget"]
    assert "--allow-over-budget" not in base and L.STANDUP_TEXT_ONLY_MM_FLAG not in base
    # the run body composes the pin for the IFEval rows only (a legacy row's argv is byte-identical); the
    # Nemotron contract rides EVERY nemotron_h row's standup (main's bcb-bfcl-200-widen S1-f, the IFEval
    # Nemotron row included); the gates hold the lane rows (main) AND the IFEval rows through ONE reader
    src = TL.LAUNCHER.read_text()
    assert "text_only_mm=(C.is_ifeval_row(row) and row.family in TEXT_ONLY_MM_FAMILIES)" in src
    # the worker's composition (the call goes on to the serving path; the ladder's rung probe composes the same
    # contract, then its own serving path / bank / KV-derived S value)
    assert ("text_only_mm=(C.is_ifeval_row(row) and row.family in TEXT_ONLY_MM_FAMILIES),\n"
            "                                mamba_contract=mamba_contract,\n") in src
    assert "if C.is_ifeval_row(row) and row.family == _mamba_backend().NEMOTRON_FAMILY" in src     # the preflight's IFEval gates
    body = src[src.index("def _run_configuration("):src.index("def run_configuration(")]
    assert body.count("rec.update(full22_attestation_record(endpoint))") == 1
    assert "refusal = full22_endpoint_refusal(row, endpoint, tree=tree) if C.is_full22_row(row) else None" in body
    assert ("if refusal is None and C.is_ifeval_row(row):\n"
            "        refusal = full22_endpoint_refusal(row, endpoint, tree=tree)") in body


def test_endpoint_refusals_name_the_gates_and_a_gemma_row_has_none(L):
    from sisteer.capeval.serving import mamba_backend as mb
    nemo, qwen, gemma = C.model_row(NEMO), C.model_row(QWEN), C.model_row(GEMMA)
    assert L.full22_endpoint_refusal(qwen, {})[0] == C.REC_TEXT_ONLY_MM_UNPINNED
    assert L.full22_endpoint_refusal(qwen, {"text_only_mm": False})[0] == C.REC_TEXT_ONLY_MM_UNPINNED
    assert L.full22_endpoint_refusal(qwen, {"text_only_mm": True}) is None       # qwen3_5: the pin is its one gate
    assert L.full22_endpoint_refusal(nemo, {})[0] == C.REC_NEMO_BACKEND_UNATTESTED
    bad = {"mamba_backend_attestation": {"ok": False, "problems": ["x"], "reports": [], "rpc_method": mb.RPC_METHOD}}
    st, why = L.full22_endpoint_refusal(nemo, bad)
    assert st == C.REC_NEMO_BACKEND_UNATTESTED and "not ok" in why
    good = {"mamba_backend_attestation": {"ok": True, "problems": [], "reports": [{"layer": 0}],
                                          "rpc_method": mb.RPC_METHOD}}
    assert L.full22_endpoint_refusal(nemo, good) is None
    assert L.full22_endpoint_refusal(gemma, {}) is None                          # no gate on gemma / llama
    assert L.full22_endpoint_refusal(C.model_row(LLAMA), {}) is None
    rec = L.full22_attestation_record({"text_only_mm": True, "mrope_attestation_error": "e"})
    assert rec["text_only_mm"] is True and rec["mrope_attestation_error"] == "e"
    assert rec["mamba_backend_attestation"] is None and rec[C.ENDPOINT_KEY_ENABLE_LOG_REQUESTS] is None
    assert set(L.FULL22_ATTESTATION_KEYS) == {"text_only_mm", "text_only_mm_record", "mrope_attestation",
                                              "mrope_attestation_error", "mamba_backend_attestation",
                                              "mamba_backend_attestation_error", C.ENDPOINT_KEY_ENABLE_LOG_REQUESTS}
    assert (C.REC_TEXT_ONLY_MM_UNPINNED, C.REC_MROPE_UNATTESTED, C.REC_NEMO_BACKEND_UNATTESTED) == (
        "TEXT_ONLY_MM_UNPINNED", "MROPE_UNATTESTED", "NEMO_BACKEND_UNATTESTED")


# --------------------------------------------------------------------------- #
# the probe: the production row, the key, the admission (P38)                  #
# --------------------------------------------------------------------------- #


def test_the_probe_is_parameterised_by_the_production_row(L):
    sig = inspect.signature(L._nemotron_probe)
    assert "row" in sig.parameters and sig.parameters["row"].default is None
    assert L.NEMOTRON_PROBE_SLUG == NEMO == "nemotron3nano4b-sdfbase-main"
    assert 'C.model_row("nemotron3nano4b")' not in TL.LAUNCHER.read_text()     # the source's hard-coded two-SI row is gone
    gsig = inspect.signature(L.nemotron_backend_probe)
    assert gsig.parameters["model"].default == NEMO
    # the lane rows' own entry `gen_long` appended LAST (the IFEval
    # spellings and the default do not move); tests/test_nemotron_probe_gen_long.py pins its planning
    assert L.PROBE_GEOMETRIES == ("2048", "4096", "8192", "longsi_2048", "longsi_4096", "longsi_8192", "gen_long")
    assert L.PROBE_GEOMETRY_DEFAULT == "2048" and L.PROBE_GEOMETRY_LONGSI_PREFIX == "longsi_" and L.PROBE_GEOMETRY_GEN_LONG == "gen_long"
    # a non-Nemotron row refuses the probe by name
    with pytest.raises(ValueError, match="certifies a 'nemotron_h' row"):
        L._nemotron_probe("a" * 40, "2048", "stacked", C.SAMPLING_VENDOR_PRESET, row=C.model_row(QWEN))
    # the probe's configuration is the row's `none` x off x ifeval at the requested cap
    cfg = L.probe_geometry_cfg(C.model_row(NEMO), "2048", reduction_mode="stacked",
                               sampling_mode=C.SAMPLING_VENDOR_PRESET)
    assert cfg["cell"] == "none" and cfg["mode"] == C.THINKING_OFF and cfg["controls"] == [C.CONTROL_IFEVAL]
    # since the IFEval plumbing every IFEval cfg carries the setter key — None at pass 1 (the pin
    # used to read "absent": the probe then edited the dict after planning; now the setter rides the plan)
    assert cfg["model"] == NEMO and cfg["off_cap_by_task"] is None and cfg["geometry_keys"] == ("capability:ifeval_2048",)
    rung = L.probe_geometry_cfg(C.model_row(NEMO), "4096", reduction_mode="stacked",
                                sampling_mode=C.SAMPLING_VENDOR_PRESET)
    assert rung["off_cap_by_task"] == {"ifeval": 4096} and rung["geometry_keys"] == ("capability:ifeval_4096",)
    assert rung["ifeval_cap"] == 4096 and cfg["ifeval_cap"] == 2048
    # the follow-up: a `longsi_<cap>` probe plans the constitution-prompted cell, whose prompt class keys the
    # capability:ifeval_longsi_<cap> entry — Nemotron's constitution row needs its own attestation per (sha, entry)
    for cap in C.IFEVAL_CAP_LADDER:
        lc = L.probe_geometry_cfg(C.model_row(NEMO), f"longsi_{cap}", reduction_mode="stacked", sampling_mode=C.SAMPLING_VENDOR_PRESET)
        assert lc["cell"] == C.CELL_PROMPTED_CONSTITUTION and lc["ifeval_cap"] == cap
        assert lc["geometry_keys"] == (f"capability:ifeval_longsi_{cap}",) == (C.IFEVAL_GEOMETRY_ENTRY_BY_CAP_LONGSI[cap],)
        assert lc["off_cap_by_task"] == ({"ifeval": cap} if cap != 2048 else None)
    with pytest.raises(ValueError, match="--geometry must be one of"):
        L.probe_geometry_cfg(C.model_row(NEMO), "16384", reduction_mode="stacked", sampling_mode=C.SAMPLING_VENDOR_PRESET)
    with pytest.raises(ValueError, match="--geometry must be one of"):
        L.probe_geometry_cfg(C.model_row(NEMO), "longsi_16384", reduction_mode="stacked", sampling_mode=C.SAMPLING_VENDOR_PRESET)


def _tree(tmp_path: Path, row: C.ModelRow, *, contract_text: str = None) -> Path:
    from sisteer.capeval.serving import mamba_backend as mb
    tree = tmp_path / "tree"
    (tree / mb.CONTRACT_RELPATH).parent.mkdir(parents=True, exist_ok=True)
    # the tree's REVIEWED contract (the part-A file), so the readers' re-derivation runs against a real one
    (tree / mb.CONTRACT_RELPATH).write_text(contract_text or (REPO / mb.CONTRACT_RELPATH).read_text())
    (tree / row.cfg).parent.mkdir(parents=True, exist_ok=True)
    (tree / row.cfg).write_text(f"model: {row.hf}\nmodel_revision: {row.rev}\n")
    return tree


def test_the_attestation_key_includes_the_production_rows_config_digest_and_admission_refuses_by_name(L, tmp_path):
    from sisteer.capeval.serving import mamba_backend as mb
    row = C.model_row(NEMO)
    tree = _tree(tmp_path, row)
    geom = SimpleNamespace(max_model_len=2528, max_num_seqs=256)
    key = L.nemotron_attestation_key(row, tree=str(tree), tree_sha="a" * 40, geom=geom, reduction_mode="stacked")
    want = mb.attestation_cache_key(image_specs=L.image_install_specs(), wheel_sha256=mb.wheel_sha256_of(L.VLLM_WHEEL),
                                    contract_sha256=mb.contract_sha256(tree / mb.CONTRACT_RELPATH), tree_sha="a" * 40,
                                    model_id=row.hf, model_revision=row.rev,
                                    serving_config_sha256=L.sha256_file(tree / row.cfg), max_model_len=2528,
                                    max_num_seqs=256, reduction_mode="stacked")
    assert key == want
    # the PRODUCTION row's config digest is in the key: another config text is another key
    (tree / row.cfg).write_text(f"model: {row.hf}\nmodel_revision: {row.rev}\nlayer: 20\n")
    assert L.nemotron_attestation_key(row, tree=str(tree), tree_sha="a" * 40, geom=geom, reduction_mode="stacked") != key
    (tree / row.cfg).write_text(f"model: {row.hf}\nmodel_revision: {row.rev}\n")
    # another row, another tree sha, another geometry entry, another reduction mode: each another key
    other_row = C.model_row(QWEN)
    _tree(tmp_path / "q", other_row)
    keys = {
        "row": L.nemotron_attestation_key(other_row, tree=str(tmp_path / "q" / "tree"), tree_sha="a" * 40, geom=geom,
                                          reduction_mode="stacked"),
        "sha": L.nemotron_attestation_key(row, tree=str(tree), tree_sha="b" * 40, geom=geom, reduction_mode="stacked"),
        "entry": L.nemotron_attestation_key(row, tree=str(tree), tree_sha="a" * 40,
                                            geom=SimpleNamespace(max_model_len=4576, max_num_seqs=171),
                                            reduction_mode="stacked"),
        "mode": L.nemotron_attestation_key(row, tree=str(tree), tree_sha="a" * 40, geom=geom, reduction_mode="per_row"),
    }
    assert len({key, *keys.values()}) == 5
    assert str(L.nemotron_attestation_path(key)) == f"{L.NEMO_BACKEND_ROOT}/{key}/{mb.ATTESTATION_FILENAME}"
    # admission: an attestation written for THIS key is found; for any other key it is missing, BY NAME
    contract = mb.load_contract(tree / mb.CONTRACT_RELPATH)
    doc = {"schema": "sisteer.mamba_backend_attestation/2", "cache_key": key, "admitted": True, "packages_ok": True,
           "packages": {"ok": True}, "static": {"ok": True, "discovery": {}}, "live": {"ok": True, "reports": []}}
    root = tmp_path / "nemo_backend"
    path = mb.attestation_path(root, key)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(doc))
    found = mb.read_attestation(path, contract, ratio=mb.duty_ratio(), cache_key=key)
    assert found["path"] == str(path) and found["doc"]["cache_key"] == key
    assert not any("no Nemotron backend attestation" in p for p in found["problems"])
    for what, k in keys.items():
        missing = mb.read_attestation(mb.attestation_path(root, k), contract, ratio=mb.duty_ratio(), cache_key=k)
        assert missing["ok"] is False and "no Nemotron backend attestation" in missing["problems"][0], what
        assert "run the backend probe at this geometry first" in missing["problems"][0]
    # a document written for another key, read at this one, refuses naming both keys
    stale = mb.read_attestation(path, contract, ratio=mb.duty_ratio(), cache_key=keys["entry"])
    assert any("attestation is for cache key" in p and "this launch needs" in p for p in stale["problems"])
    # the kernel specs ride in the image's pin audit
    assert set(L.tinycap_images.image_install_specs_nemotron()) <= set(L.image_install_specs())


def test_the_preflight_and_main_carry_the_nemotron_gates(L):
    src = TL.LAUNCHER.read_text()
    i = src.index("def preflight(")
    body = src[i:src.index("def _upload_tree(", i)]
    assert 'chk("image_nemotron_bound", image_bound)' in body and 'chk("nemotron_backend_attested", backend_attested)' in body
    assert "if row.family == _mamba_backend().NEMOTRON_FAMILY:" in body
    assert "nemotron_attestation_key(row, tree=tree, tree_sha=tree_sha, geom=geom," in body
    m = src[src.index("def main("):]
    assert "if (cfgs or nemotron_probe) and row.family == _mamba_backend().NEMOTRON_FAMILY and IMAGE_NEMOTRON_ERROR" in m
    assert "r = nemotron_backend_probe.remote(sha, geometry, reduction_mode," in m
    # the count is the pure gpu_container_count — the probe is still ONE container in it
    assert "n_gpu = gpu_container_count(len(cfgs), row_launch=row_launch, nemotron_probe=nemotron_probe)" in m
    assert L.gpu_container_count(0, row_launch=False, nemotron_probe=True) == 1 and L.gpu_container_count(2, row_launch=True, nemotron_probe=True) == 2
    argv = L.launch_argv("a" * 40, pass1=True, model=NEMO, nemotron_probe=True, geometry="4096")
    assert argv[-3:] == [L.FLAG_NEMOTRON_PROBE, L.FLAG_GEOMETRY, "4096"]
    assert L.FLAG_NEMOTRON_PROBE not in L.launch_argv("a" * 40, pass1=True, model=NEMO)


# --------------------------------------------------------------------------- #
# launch_configurations: one spawn per configuration, ids before any wait      #
# --------------------------------------------------------------------------- #


def test_launch_configurations_spawns_per_cfg_records_fc_ids_before_waiting_and_harvests_failures(L, tmp_path):
    events = []

    class FC:
        def __init__(self, i):
            self.object_id = f"fc-{i}"

        def get(self):
            events.append(("get", self.object_id))
            if self.object_id == "fc-2":
                raise RuntimeError("container died")
            return {"name": f"c{self.object_id[-1]}", "status": "OK"}

    class Fn:
        n = 0

        def spawn(self, cfg, **kw):
            Fn.n += 1
            events.append(("spawn", cfg["name"], kw["sha"]))
            return FC(Fn.n)

    log = tmp_path / "spawned.jsonl"
    cfgs = [{"name": "c1", "model": QWEN, "cell": "none", "mode": "off"},
            {"name": "c2", "model": QWEN, "cell": "prompted__safety_refusal", "mode": "off"}]
    recs = L.launch_configurations(Fn(), cfgs, {"sha": "x"}, spawned_log=log)
    assert [r["status"] for r in recs] == ["OK", "FUNCTIONCALL_FAILED"]
    assert recs[1]["function_call_id"] == "fc-2" and "container died" in recs[1]["error"]
    # every spawn (and its spawned.jsonl line) precedes the first get
    assert events.index(("get", "fc-1")) > events.index(("spawn", "c2", "x"))
    lines = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert [(ln["name"], ln["fc_id"]) for ln in lines] == [("c1", "fc-1"), ("c2", "fc-2")]
    # wait=False (the dispatcher): the handles, nothing waited on
    Fn.n, events[:] = 0, []
    handles = L.launch_configurations(Fn(), cfgs[:1], {"sha": "y"}, wait=False)
    assert [(c["name"], h.object_id) for c, h in handles] == [("c1", "fc-1")]
    assert events == [("spawn", "c1", "y")]
    # a function without spawn (a test double) is called directly
    recs = L.launch_configurations(lambda cfg, **kw: {"name": cfg["name"], "status": "OK"}, cfgs, {"sha": "z"})
    assert [r["status"] for r in recs] == ["OK", "OK"]


def test_the_seven_slugs_overlay_configs_and_manifests_exist():
    """`scripts/sdfbase/ifeval/make_ifeval_overlays.py` writes the fourteen
    files from the frozen matrix (tests/test_sdfbase_ifeval_overlays.py pins their byte-identity); the two Llama
    TAGs' campaign files are byte-identical copies of the Llama lane's."""
    for (key, view), slug in C.IFEVAL_SLUGS.items():
        row = C.model_row(slug)
        assert (REPO / row.cfg).is_file(), row.cfg
        assert (REPO / C.IFEVAL_OVERLAY_MANIFEST_REL.format(tag=C.ifeval_tag(key, view))).is_file()
        assert "ifeval-canonical-battery OVERLAY" in (REPO / row.cfg).read_text(encoding="utf-8")
    assert (REPO / "scripts/sdfbase/ifeval/make_ifeval_overlays.py").is_file()


def test_the_probe_and_the_preflight_derive_one_attestation_key_from_the_same_row_facts(L, tmp_path):
    """The probe wrote its admitted attestation under one key while
    the Nemotron preflight looked for another at the same tree and geometry — the probe hashed the
    loaded contract dict, the preflight the contract's path, and `contract_sha256` gave them
    different digests. Now both go through `nemotron_attestation_key_inputs`, the contract digest
    is canonical whatever the argument, and the probe composes no key dict of its own."""
    from sisteer.capeval.serving import mamba_backend as mb
    row = C.model_row(NEMO)
    tree = _tree(tmp_path, row)
    contract = mb.load_contract(tree / mb.CONTRACT_RELPATH)
    (tree / mb.CONTRACT_RELPATH).write_text(json.dumps(contract, indent=3) + "\n\n")   # NON-canonical bytes
    geom = SimpleNamespace(max_model_len=2528, max_num_seqs=256)
    inputs = L.nemotron_attestation_key_inputs(row, tree=str(tree), tree_sha="c" * 40, geom=geom, reduction_mode="stacked")
    # the WRITER's key (the probe: attestation_cache_key over the recorded inputs) == the READER's (the preflight)
    writer = mb.attestation_cache_key(**inputs)
    reader = L.nemotron_attestation_key(row, tree=str(tree), tree_sha="c" * 40, geom=geom, reduction_mode="stacked")
    assert writer == reader
    # ... and equal to the standup's own digest of the LOADED contract (the endpoint's contract_sha256)
    assert inputs["contract_sha256"] == mb.contract_sha256(contract) == mb.contract_sha256(tree / mb.CONTRACT_RELPATH)
    assert (inputs["max_model_len"], inputs["max_num_seqs"], inputs["reduction_mode"]) == (2528, 256, "stacked")
    assert inputs["tree_sha"] == "c" * 40 and inputs["model_id"] == row.hf and inputs["model_revision"] == row.rev
    assert inputs["serving_config_sha256"] == L.sha256_file(tree / row.cfg)
    # another geometry, another key; another reduction mode, another key
    assert L.nemotron_attestation_key(row, tree=str(tree), tree_sha="c" * 40,
                                      geom=SimpleNamespace(max_model_len=4576, max_num_seqs=171), reduction_mode="stacked") != reader
    assert L.nemotron_attestation_key(row, tree=str(tree), tree_sha="c" * 40, geom=geom, reduction_mode="per_row") != reader
    # the probe composes NO key dict of its own — it calls the shared helper — and tables what the
    # standup SERVED, not the S it asked for
    src = TL.LAUNCHER.read_text()
    body = src[src.index("def _nemotron_probe("):src.index("def nemotron_backend_probe(")]
    assert "nemotron_attestation_key_inputs(row, tree=tree, tree_sha=tree_sha, geom=geom," in body
    assert "key_inputs = dict(" not in body and "contract_sha256=mb.contract_sha256(contract)" not in body
    assert 'burst["served"] = served_lxs(endpoint, g_)' in body      # g_: the standup's own geometry (the peer's is the served one)
    assert 'g_ = geom if which == "nemotron" else peer_geom' in body


def test_the_probes_peer_stands_up_at_the_s_nemotron_served_not_the_s_it_asked_for(L):
    """At capability:ifeval_4096 (4576 x 171) Nemotron's KV-derived shrink served
    S = 168 while the same-container Qwen peer, stood up at the REQUESTED 171 (--allow-over-budget), served 171, and
    `mamba_backend.duty_admission` refused the reference ("served S differs: nemotron 168 vs peer 171") — so no
    certification could be written at that entry, while 2528 x 256 and 10496 x 74 (no shrink) certified fine. The
    peer's geometry is now Nemotron's SERVED geometry, read off its endpoint by the launcher's own reader
    (`served_geometry_of_endpoint`); a served record that does not hold refuses the probe by name."""
    from sisteer.capeval.serving import geometry as sg
    from sisteer.capeval.serving import mamba_backend as mb
    from sisteer.capeval.serving import vllm_server as vs
    geom = sg.resolve_geometry(None, max_model_len=4576, max_num_seqs=171, fallback=(10240, 64), allow_over_budget=True)
    cap = 168 * 4576 + 1234                       # the container backed 168 full-length sequences (floor of 168.27)
    shrink = sg.kv_shrink_record(configured_max_num_seqs=171, max_model_len=4576, kv_backed_concurrency=cap / 4576,
                                 kv_capacity_tokens=cap, basis=vs.KV_BASIS_HYBRID)
    nemo_ep = {"geometry": {"max_model_len": 4576, "max_num_seqs": 168, sg.KV_SHRINK_KEY: shrink}}
    peer_geom, rec, problem = L.served_geometry_of_endpoint(nemo_ep, geom)
    assert problem is None and rec == shrink and (peer_geom.max_model_len, peer_geom.max_num_seqs) == (4576, 168)
    # the peer's standup argv is asked for the SERVED S ...
    argv = L._standup_cmd("/tree", "/tree/cfg.yaml", "none", "d", Path("/tmp/probe-x"), Path("/tmp/probe-x/STOP"),
                          peer_geom, 64, [], allow_over_budget=True)
    i = next(i for i, a in enumerate(argv) if a.startswith("--") and "seqs" in a and argv[i + 1].isdigit())
    assert argv[i + 1] == "168" and "171" not in argv
    # ... so both bursts table served S 168 and the reference is comparable; at the requested 171 it was not
    duty = {"sm_duty_mean": 0.9, "tok_s": 5000.0, "n_seqs": 64, "max_tokens": 512, "gpu_index": "0",
            "served": L.served_lxs(nemo_ep, geom)}
    assert duty["served"]["max_num_seqs"] == 168 and duty["served"]["configured_max_num_seqs"] == 171
    ref_old = {**duty, "served": L.served_lxs({"geometry": {"max_model_len": 4576, "max_num_seqs": 171}}, geom)}
    ref_new = {**duty, "served": L.served_lxs({"geometry": {"max_model_len": 4576, "max_num_seqs": 168}}, peer_geom)}
    assert mb.duty_admission(duty, ref_old, ratio=0.5)["incomparable_reasons"] == ["served S differs: nemotron 168 vs peer 171"]
    adm = mb.duty_admission(duty, ref_new, ratio=0.5)
    assert adm["duty_comparable"] and adm["duty_ok"] and adm["tok_s_ok"] and adm["incomparable_reasons"] == []
    # a served record that does not hold refuses the probe by name (never read around)
    bad = {"geometry": {"max_model_len": 4576, "max_num_seqs": 160, sg.KV_SHRINK_KEY: shrink}}
    _, _, problem = L.served_geometry_of_endpoint(bad, geom)
    assert problem and "does not hold" in problem
    # the probe body: the peer's geometry is the served one, read by the served-geometry reader, on every site that takes a geometry
    src = TL.LAUNCHER.read_text()
    body = src[src.index("def _nemotron_probe("):src.index("def nemotron_backend_probe(")]
    assert body.count('g_ = geom if which == "nemotron" else peer_geom') == 1
    assert body.count("peer_geom, shrink, geo_problem = served_geometry_of_endpoint(endpoint, geom)") == 1
    assert "rec.update(status=C.REC_SERVED_GEOMETRY_MISMATCH, error=geo_problem)" in body
    assert "r_.objects_dir, srv, stop, g_, r_.top_logprobs" in body and 'burst["served"] = served_lxs(endpoint, g_)' in body
    assert 'standup_rec = {"model": r_.slug, "L": int(g_.max_model_len), "S": int(g_.max_num_seqs)' in body
    assert "srv, stop, geom, r_.top_logprobs" not in body and "served_lxs(endpoint, geom)" not in body
    assert 'rec["peer_geometry"] = {' in body


def test_every_standup_opts_in_to_the_kv_derived_s_and_the_launcher_holds_the_served_geometry(L):
    """The fixed standup argv carries ``--kv-derived-seqs on`` right
    after ``--gpu-mem-util`` (the flag spelled once, equal to capctrl's), and the body reads the
    served geometry back off endpoint.json — refusing anything but the configured S or a
    kv_shrink record that holds — before it re-takes the resume decision exactly."""
    import test_serve_declarations_attestation as SDA
    capctrl = SDA._load("_port_kv_capctrl", "scripts/capctrl_serving_up.py")
    assert L.STANDUP_KV_DERIVED_SEQS_FLAG == capctrl.KV_DERIVED_SEQS_FLAG == "--kv-derived-seqs"
    src = TL.LAUNCHER.read_text()
    fixed = src[src.index("def _standup_cmd("):src.index("def served_geometry_of_endpoint(")]
    # the flag's VALUE is a parameter — `on` by default (every
    # non-ladder argv byte-identical), `off` for EVERY truncation-ladder standup on both engines (the rung probe
    # on both engines, and the scoring worker), whose
    # fixed rung needs a KV shortfall to REFUSE readiness rather than come up at a shrunk S the certificate would not name
    assert ('"--gpu-mem-util", "0.90", STANDUP_KV_DERIVED_SEQS_FLAG, ("on" if kv_derived_seqs else "off"),\n'
            '            "--ready-timeout-s", "2700",') in fixed
    assert inspect.signature(L._standup_cmd).parameters["kv_derived_seqs"].default is True
    argv_on = L._standup_cmd("/t", "cfg.yaml", "none", "/d", Path("/srv"), Path("/srv/STOP"),
                             SimpleNamespace(max_model_len=2528, max_num_seqs=256, model_key=None), 5, [])
    argv_off = L._standup_cmd("/t", "cfg.yaml", "none", "/d", Path("/srv"), Path("/srv/STOP"),
                              SimpleNamespace(max_model_len=2528, max_num_seqs=256, model_key=None), 5, [], kv_derived_seqs=False)
    i = argv_on.index(L.STANDUP_KV_DERIVED_SEQS_FLAG)
    assert argv_on[i:i + 3] == [L.STANDUP_KV_DERIVED_SEQS_FLAG, "on", "--ready-timeout-s"]
    assert argv_off[i:i + 3] == [L.STANDUP_KV_DERIVED_SEQS_FLAG, "off", "--ready-timeout-s"]
    assert argv_off[:i + 1] + argv_off[i + 2:] == argv_on[:i + 1] + argv_on[i + 2:]           # the ONE token differs
    body = src[src.index("def _run_configuration("):src.index("def run_configuration(")]
    i_pre = body.index("resume = resume_report(out_dir, wanted, expected, kv_shrink_tolerated=True) if not rung else")
    i_conf = body.index("producer_tree_conflicts(out_dir, expected=expected, kv_shrink_tolerated=True)")
    # the reader takes the ladder record on a native ladder rung (the production reading before the driver); the
    # order of the sites is unchanged
    i_served = body.index("geom_served, kv_shrink, geometry_problem = served_geometry_of_endpoint(\n"
                          "        endpoint, geom, ladder_record=(rec[\"geometry\"] if rung and native else None))")
    i_refuse = body.index("rec.update(status=C.REC_SERVED_GEOMETRY_MISMATCH, error=geometry_problem,")
    i_retake = body.index("resume = resume_report(out_dir, wanted, expected) if not rung else ladder_resume_report(out_dir, wanted, expected)\n"
                          "    rec[\"resume\"] = resume")
    assert i_pre < i_conf < body.index("standup = subprocess.Popen(") < i_served < i_refuse < i_retake
    assert C.REC_SERVED_GEOMETRY_MISMATCH == "SERVED_GEOMETRY_MISMATCH"
    # the served-geometry reader: the configured S passes, another S with no record refuses,
    # a record that holds re-derives S
    geom = SimpleNamespace(max_model_len=2528, max_num_seqs=256, batched_tokens=2528 * 256, is_identity=False,
                           seqs_source="budget")
    ok = L.served_geometry_of_endpoint({"geometry": {"max_model_len": 2528, "max_num_seqs": 256}}, geom)
    assert ok[0] is geom and ok[1] is None and ok[2] is None
    other = L.served_geometry_of_endpoint({"geometry": {"max_model_len": 2528, "max_num_seqs": 200}}, geom)
    assert other[2] is not None and "records no kv_shrink" in other[2]
    window = L.served_geometry_of_endpoint({"geometry": {"max_model_len": 4576, "max_num_seqs": 256}}, geom)
    assert window[2] is not None and "not the resolved 2528" in window[2]
    none = L.served_geometry_of_endpoint({}, geom)
    assert none[2] == "endpoint.json has no geometry block"


# --------------------------------------------------------------------------- #
# The IFEval configuration plumbing                                           #
# --------------------------------------------------------------------------- #

IFEVAL_ENTRY = C.IFEVAL_GEOMETRY_ENTRY_BY_CAP
_MODEL_KEYS = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b")


def _plan(L, slug=QWEN, cell="none", **kw):
    return L.plan_configurations(cells=[cell], modes=[C.THINKING_OFF], controls=[C.CONTROL_IFEVAL], model=slug,
                                 sampling_mode=C.SAMPLING_VENDOR_PRESET, samples=C.IFEVAL_SAMPLES_PER_ITEM,
                                 reduction_mode=C.REDUCTION_MODE_STACKED, pass_no=1, **kw)


def _subset_doc(ids=None, *, cap=4096, source_cap=2048):
    """A synthetic rung_subset (exactly ITEM_SUBSET_FIELDS) for an IFEval rung: the eligible
    set IS n_at_cap - n_non_terminating - n_boundary, a PASSED transport verdict."""
    ids = ids if ids is not None else [f"ifeval_{i:04d}" for i in range(5)]
    return {"schema": C.ITEM_SUBSET_SCHEMA, "task": C.IFEVAL_TASK, "source_out_root": "/synthetic/root",
            "source_identity_hash": "0123456789abcdef", "source_cap": source_cap, "cap": cap, "rule": "r",
            "threshold": 0.03, "ids": ids, "n_at_cap": 7, "n_non_terminating": 2, "n_boundary": 0,
            "transport_check": {"ok": True}}


def test_the_ifeval_plan_carries_the_cfg_keys_the_caps_entry_and_the_slugs_view(L):
    """One configuration per (slug, cell, cap) through `plan_configurations`, the
    dispatcher's exact call — every cfg carries the draw it was planned under; an IFEval cfg
    carries the setter map, the subset path, the slug's bank view, the EXECUTED cap and the
    cap's own geometry entry; an explicit out_dir replaces the cell's."""
    (cfg,) = _plan(L)
    assert cfg["controls"] == [C.CONTROL_IFEVAL] and cfg["mode"] == C.THINKING_OFF
    # the draw rides main's SAMPLING_MODE_KEY marker (present iff not the driver's default — vendor_preset is
    # not), read back through the ONE accessor; the explicit k is recorded on an IFEval configuration alone
    assert L.SAMPLING_MODE_KEY == "sampling_mode" and (cfg[L.SAMPLING_MODE_KEY], cfg["samples"]) == (C.SAMPLING_VENDOR_PRESET, 2)
    assert L.sampling_mode_of(cfg) == C.SAMPLING_VENDOR_PRESET
    assert (cfg["off_cap_by_task"], cfg["item_subset"], cfg["bank_view"], cfg["ifeval_cap"]) == (None, None, "main", 2048)
    assert cfg["geometry_keys"] == (IFEVAL_ENTRY[2048],) == ("capability:ifeval_2048",)
    assert cfg["out_dir"] == C.cell_out_dir(C.VOL_GEN_ROOT, "none", C.THINKING_OFF, model_slug=QWEN)
    assert cfg["thinking_by_task"] == {C.IFEVAL_TASK: C.THINKING_OFF}
    assert L.is_ifeval_configuration(cfg) and not L.ifeval_lens_configuration(cfg)
    # the two rungs: the setter's cap is the executed cap and selects the entry; the subset and the root ride
    for cap in (4096, 8192):
        (r,) = _plan(L, GEMMA, off_cap_by_task={C.IFEVAL_TASK_NAME: cap}, item_subset=f"/vol/w/rung_{cap}/rung_subset.json",
                     bank_view="it", out_dir=f"/vol/w/rung_{cap}/out")
        assert r["off_cap_by_task"] == {"ifeval": cap} and r["ifeval_cap"] == cap
        assert r["geometry_keys"] == (IFEVAL_ENTRY[cap],) == (f"capability:ifeval_{cap}",)
        assert r["item_subset"] == f"/vol/w/rung_{cap}/rung_subset.json" and r["out_dir"] == f"/vol/w/rung_{cap}/out"
        assert r["bank_view"] == "it"
    # the bank view defaults to the SLUG's on every one of the seven rows (the ModelRow has no view field)
    for (key, view), slug in C.IFEVAL_SLUGS.items():
        (c,) = _plan(L, slug)
        assert c["bank_view"] == view == L.ifeval_view_of(C.model_row(slug)), slug
        assert c["model"] == slug and c["geometry_keys"] == (IFEVAL_ENTRY[2048],)
    with pytest.raises(ValueError, match="serves no bank view"):
        L.ifeval_view_of(C.model_row("gemma4e4b"))
    # a lens cell is an IFEval LENS configuration (its rows must name their object)
    (lens,) = _plan(L, cell="safety_refusal__psr_kl_all")
    assert L.ifeval_lens_configuration(lens) and lens["steered"]
    # a legacy configuration is byte-identical to main's plan: NO draw marker (the default is not recorded),
    # NO `samples`, and NOTHING of IFEval's; its other keys are unchanged
    (leg,) = L.plan_configurations(cells=["none"], modes=["off"], controls=["gsm8k"])
    assert L.SAMPLING_MODE_KEY not in leg and "samples" not in leg and L.sampling_mode_of(leg) == C.SAMPLING_DEFAULT
    assert not any(k in leg for k in ("off_cap_by_task", "item_subset", "bank_view", "ifeval_cap"))
    assert leg["geometry_keys"] == ("capability:gen_long",) and not L.is_ifeval_configuration(leg)
    assert L.geometry_keys_for("none", "off", ["gsm8k"]) == ("capability:gen_long",)        # positional call unchanged
    assert L.geometry_keys_for("none", "off", [C.CONTROL_IFEVAL]) == (IFEVAL_ENTRY[2048],)
    assert L.geometry_keys_for("none", "off", [C.CONTROL_IFEVAL], off_cap_by_task={"ifeval": 8192}) == (IFEVAL_ENTRY[8192],)
    with pytest.raises(ValueError, match="no capability:ifeval_<cap> geometry entry for cap 1024"):
        L.geometry_keys_for("none", "off", [C.CONTROL_IFEVAL], off_cap_by_task={"ifeval": 1024})
    # the follow-up: the geometry key follows the cell's PROMPT CLASS — the constitution-prompted cell takes the
    # long-SI entry at every cap; the safety-prompted, bare and object cells the short one; an unmeasured SI refuses
    for cap in C.IFEVAL_CAP_LADDER:
        kw = dict(off_cap_by_task=({"ifeval": cap} if cap != 2048 else None))
        assert L.geometry_keys_for(C.CELL_PROMPTED_CONSTITUTION, "off", [C.CONTROL_IFEVAL], **kw) == (f"capability:ifeval_longsi_{cap}",)
        for cell in ("none", "prompted__safety_refusal", "constv2-mix-full__lora_r4_ce", "alphasteer"):
            assert L.geometry_keys_for(cell, "off", [C.CONTROL_IFEVAL], **kw) == (IFEVAL_ENTRY[cap],), (cell, cap)
    with pytest.raises(C.IfevalPromptClassUnmeasured, match="no measured IFEval prompt class"):
        L.geometry_keys_for("prompted__si_irrelevant", "off", [C.CONTROL_IFEVAL])
    (con,) = _plan(L, GEMMA, cell=C.CELL_PROMPTED_CONSTITUTION)
    assert con["geometry_keys"] == ("capability:ifeval_longsi_2048",) and con["prompted"] and not L.ifeval_lens_configuration(con)
    g = L.configuration_geometry(con, 1)
    assert (g.eval, g.max_model_len, g.max_num_seqs) == ("capability:ifeval_longsi_2048", 8448, 69)
    assert L._IFEVAL_KEYS == frozenset(C.IFEVAL_GEOMETRY_ENTRIES) and len(L._IFEVAL_KEYS) == 6


def test_the_ifeval_plan_refuses_each_ill_formed_input_by_name(L):
    with pytest.raises(ValueError, match=r"\['off_cap_by_task'\] are IFEval planning inputs"):
        L.plan_configurations(cells=["none"], modes=["off"], controls=["gsm8k"], model=QWEN,
                              off_cap_by_task={"ifeval": 4096})
    for name, kw in (("item_subset", {"item_subset": "/p"}), ("bank_view", {"bank_view": "main"}), ("out_dir", {"out_dir": "/o"})):
        with pytest.raises(ValueError, match=rf"\['{name}'\] are IFEval planning inputs"):
            L.plan_configurations(cells=["none"], modes=["off"], controls=["gsm8k"], model=QWEN, **kw)
    with pytest.raises(ValueError, match="item_subset '/p' without off_cap_by_task"):
        _plan(L, item_subset="/p")
    with pytest.raises(ValueError, match="off_cap_by_task ifeval=2048 is not a rung the plan allows"):
        _plan(L, off_cap_by_task={"ifeval": 2048})
    with pytest.raises(ValueError, match="off_cap_by_task ifeval=16384 is not a rung the plan allows"):
        _plan(L, off_cap_by_task={"ifeval": 16384})
    with pytest.raises(ValueError, match="off_cap_by_task names 'gsm8k_paired200'"):
        _plan(L, off_cap_by_task={"gsm8k_paired200": 4096})
    with pytest.raises(ValueError, match="bank_view 'it' disagrees with the slug's view 'main'"):
        _plan(L, bank_view="it")
    with pytest.raises(ValueError, match="bank_view 'fullconst' is not one of"):
        _plan(L, bank_view="fullconst")
    with pytest.raises(ValueError, match="runs on the IFEval wave's rows only"):
        _plan(L, "gemma4e4b")
    with pytest.raises(ValueError, match="runs on the IFEval wave's rows only"):
        _plan(L, "qwen35_4b")
    with pytest.raises(L.ThinkingUnservable, match="'ifeval' runs thinking OFF"):
        L.plan_configurations(cells=["none"], modes=[C.THINKING_ON], controls=[C.CONTROL_IFEVAL], model=QWEN,
                              sampling_mode=C.SAMPLING_VENDOR_PRESET, samples=2)
    with pytest.raises(ValueError, match="runs the 'ifeval' atom ALONE"):
        L.plan_configurations(cells=["none"], modes=["off"], controls=[C.CONTROL_IFEVAL, "gsm8k"], model=QWEN,
                              sampling_mode=C.SAMPLING_VENDOR_PRESET, samples=2)
    with pytest.raises(ValueError, match="out_dir must be a path"):
        _plan(L, out_dir="")
    with pytest.raises(ValueError, match="item_subset must be the rung_subset json's path"):
        _plan(L, off_cap_by_task={"ifeval": 4096}, item_subset="")
    with pytest.raises(ValueError, match="sampling mode must be one of"):
        L.plan_configurations(cells=["none"], modes=["off"], controls=["gsm8k"], sampling_mode="bogus")


def test_the_ifeval_driver_cmd_composes_the_plans_tail_and_every_other_argv_is_byte_identical(L):
    """The driver line: `... --sampling vendor_preset --tinycap-identity-version 2
    --max-tokens-by-task ifeval=<cap> [--off-cap-by-task ifeval=<cap> --item-subset <path>]
    --bank-view <view>` (the tail `contracts.driver_argv` appends, in its fixed order); `--samples 2`
    in its usual slot. A non-IFEval configuration's argv is pinned against `C.driver_argv` with
    NONE of the new keywords. An IFEval LENS cell rides the object digest, and refuses without it."""
    ep = {"proxy_url": "u", "routing_table": "t"}
    (cfg,) = _plan(L)
    geom = L.configuration_geometry(cfg, 1)
    cmd = L._driver_cmd("<tree>", cfg, ep, None, geom, controls=cfg["controls"], limit=0, inspect_limit=0,
                        samples=cfg["samples"], cfg_path="<cfg>", vllm_log=None)
    pairs = dict(zip(cmd[3::2], cmd[4::2]))
    assert pairs["--controls"] == "ifeval" and pairs["--thinking"] == "off" and pairs["--samples"] == "2"
    assert pairs["--concurrency"] == str(geom.max_num_seqs) and pairs[C.FLAG_OUT_ROOT] == cfg["out_dir"]
    assert cmd[cmd.index(C.FLAG_SAMPLING):] == [
        C.FLAG_SAMPLING, C.SAMPLING_VENDOR_PRESET, C.FLAG_TINYCAP_IDENTITY_VERSION, "2",
        C.FLAG_MAX_TOKENS_BY_TASK, "ifeval=2048", C.FLAG_BANK_VIEW, "main"]
    assert C.FLAG_OFF_CAP_BY_TASK not in cmd and C.FLAG_ITEM_SUBSET not in cmd
    (r,) = _plan(L, GEMMA, off_cap_by_task={"ifeval": 8192}, item_subset="/vol/w/rung_subset.json", out_dir="/vol/w/out")
    rc = L._driver_cmd("<tree>", r, ep, None, L.configuration_geometry(r, 1), controls=r["controls"], limit=0,
                       inspect_limit=0, samples=r["samples"], cfg_path="<cfg>", vllm_log=None)
    assert rc[rc.index(C.FLAG_SAMPLING):] == [
        C.FLAG_SAMPLING, C.SAMPLING_VENDOR_PRESET, C.FLAG_TINYCAP_IDENTITY_VERSION, "2",
        C.FLAG_MAX_TOKENS_BY_TASK, "ifeval=8192", C.FLAG_OFF_CAP_BY_TASK, "ifeval=8192",
        C.FLAG_ITEM_SUBSET, "/vol/w/rung_subset.json", C.FLAG_BANK_VIEW, "it"]
    assert dict(zip(rc[3::2], rc[4::2]))[C.FLAG_OUT_ROOT] == "/vol/w/out"
    assert L.ifeval_driver_kwargs(cfg) == {"identity_version": 2, "max_tokens_by_task": {"ifeval": 2048},
                                           "off_cap_by_task": None, "item_subset": None, "bank_view": "main"}
    # a legacy configuration: byte-identical to contracts.driver_argv with no new keyword at all
    (leg,) = L.plan_configurations(cells=["none"], modes=["off"], controls=["gsm8k"])
    lg = L.configuration_geometry(leg, 1)
    got = L._driver_cmd("<tree>", leg, ep, "http://127.0.0.1:2/v1", lg, controls=leg["controls"], limit=3,
                        inspect_limit=0, samples=1, cfg_path="<cfg>", vllm_log="/v.log")
    row = L.row_of(leg)
    want = C.driver_argv(python=L.CLIENT_PY, script="<tree>/scripts/served_capability_run.py", proxy_url="u",
                         routing_table="t", cell="none", config="<cfg>", defenses_dir=row.objects_dir,
                         out_root=leg["out_dir"], controls=leg["controls"], thinking="off", concurrency=lg.max_num_seqs,
                         judge_model=L.JUDGE_MODEL, limit=3, samples=1, inspect_limit=0,
                         ledger_url="http://127.0.0.1:2/v1", top_logprobs=row.top_logprobs, vllm_log="/v.log")
    assert got == want and L.ifeval_driver_kwargs(leg) == {}
    assert not any(f in got for f in C.IFEVAL_DRIVER_FLAGS)
    # an IFEval LENS cell: the digest the standup was held to is REQUIRED and rides the digest flag
    (lens,) = _plan(L, cell="safety_refusal__psr_kl_all")
    kw = dict(controls=lens["controls"], limit=0, inspect_limit=0, samples=2, cfg_path="<cfg>", vllm_log=None)
    with pytest.raises(ValueError, match="IFEval lens configuration needs the artifact digest"):
        L._driver_cmd("<tree>", lens, ep, None, geom, **kw)
    lc = L._driver_cmd("<tree>", lens, ep, None, geom, expect_object_sha256="0" * 64, **kw)
    assert dict(zip(lc[3::2], lc[4::2]))[C.FLAG_EXPECT_OBJECT_SHA256] == "0" * 64
    assert lc[lc.index(C.FLAG_SAMPLING):] == [C.FLAG_SAMPLING, C.SAMPLING_VENDOR_PRESET, C.FLAG_TINYCAP_IDENTITY_VERSION,
                                              "2", C.FLAG_MAX_TOKENS_BY_TASK, "ifeval=2048", C.FLAG_BANK_VIEW, "main"]
    # a base IFEval cell offered a digest still refuses (its driver would not compare it)
    with pytest.raises(ValueError, match="not an IFEval lens"):
        L._driver_cmd("<tree>", cfg, ep, None, geom, expect_object_sha256="0" * 64, **dict(kw, controls=cfg["controls"]))
    # the run body and the dry run hand the digest to the IFEval lens cells too
    src = TL.LAUNCHER.read_text()
    assert src.count("if landed_lens_configuration(cfg)\n                                                or ifeval_lens_configuration(cfg) else None") == 1
    assert "if landed_lens_configuration(cfg) or ifeval_lens_configuration(cfg)" in src
    # the run body reads the PLAN's draw through main's ONE accessor and IFEval's k off the cfg (E54 #6: one
    # local for the check and the identity)
    body = src[src.index("def _run_configuration("):src.index("def run_configuration(")]
    assert "sampling_mode = sampling_mode_of(cfg)" in body and body.count("sampling_mode = ") == 1
    assert 'samples = int(cfg.get("samples", samples))' in body and "sampling_mode = C.SAMPLING_DEFAULT" not in body


def test_configuration_geometry_resolves_the_caps_entry_for_either_pass_with_s_from_the_rows_budget(L):
    """One key, one MEASURED window, for EITHER pass — `resolve_geometry(key,
    strict_identity=True, model_key=<row>)` (S from the row's budget; the entry's basis is the
    prompt budget, not a ledger), because `pass1_geometry` refuses a
    non-tinycap key. Pooled beside another key it refuses by name."""
    from sisteer.capeval.serving import geometry as sg
    missing = [k for k in IFEVAL_ENTRY.values() if k not in sg.TABLE]
    if missing:
        pytest.skip(f"the IFEval geometry entries are not registered in serving/geometry.py yet: {missing}")
    with pytest.raises(ValueError, match="not a tinycap sub-entry"):
        sg.pass1_geometry(IFEVAL_ENTRY[2048], model_key="gemma4e4b")
    for (key, view), slug in C.IFEVAL_SLUGS.items():
        for cap in C.IFEVAL_CAP_LADDER:
            (cfg,) = _plan(L, slug, off_cap_by_task=({"ifeval": cap} if cap != 2048 else None))
            want = sg.resolve_geometry(IFEVAL_ENTRY[cap], strict_identity=True, model_key=key)
            for pass_no in (1, 2):
                g = L.configuration_geometry(cfg, pass_no)
                assert (g.eval, g.max_model_len, g.max_num_seqs) == (IFEVAL_ENTRY[cap], want.max_model_len, want.max_num_seqs), (slug, cap, pass_no)
                assert g.max_model_len == sg.TABLE[IFEVAL_ENTRY[cap]].max_model_len == 468 + cap + (12 if cap == 2048 else 12)
                assert g.seqs_source == L.MODEL_BUDGET_SOURCE and g.model_key == key
            geom, budget = L.served_geometry(cfg, 1)
            assert geom.max_num_seqs == want.max_num_seqs and budget["model_key"] == key
    # the S the addendum records per (row, cap)
    s_of = {k: tuple(L.configuration_geometry(_plan(L, C.IFEVAL_SLUGS[(k, "main")],
                                                     off_cap_by_task=({"ifeval": cap} if cap != 2048 else None))[0], 1).max_num_seqs
                     for cap in C.IFEVAL_CAP_LADDER) for k in _MODEL_KEYS}
    # qwen35_4b: 741,888 (the 2026-09-13 lens-stack row at 17,664; the swept 720,896 gave 157 / 83) — 293 at 2,528 is held to the hybrid cap 256
    assert s_of == {"gemma4e4b": (233, 128, 68), "qwen35_4b": (256, 162, 85), "nemotron3nano4b": (256, 171, 90), "llama8b": (187, 103, 54)}
    (cfg,) = _plan(L)
    with pytest.raises(L.IfevalGeometryPooled, match="pools the IFEval entry"):
        L.configuration_geometry(dict(cfg, geometry_keys=(IFEVAL_ENTRY[2048], "capability:tiny")), 1)
    with pytest.raises(ValueError, match="pass must be 1 or 2"):
        L.configuration_geometry(cfg, 3)
    assert L._IFEVAL_KEYS == frozenset(C.IFEVAL_GEOMETRY_ENTRIES) > frozenset(IFEVAL_ENTRY.values())   # + the long-SI entries
    assert L._CONTROL_WORKLOAD_NAMES[C.CONTROL_IFEVAL] == "IFEval"


def test_the_preflight_probes_the_ifeval_driver_flags_on_an_ifeval_plan_only(L):
    (cfg,) = _plan(L)
    (lens,) = _plan(L, cell="safety_refusal__psr_kl_all")
    (leg,) = L.plan_configurations(cells=["none"], modes=["off"], controls=["gsm8k"])
    assert L.ifeval_probe_flags([leg]) == () and L.ifeval_probe_flags([]) == ()
    # the follow-up (Appendix A; M6): the IFEval line carries the container-written budget artifact and a fresh
    # attempt id, so both are probed (the budget flag once — an OOS / comparator launch already probes it)
    tail = (C.FLAG_OOS_PROMPT_BUDGET, C.FLAG_ATTEMPT_ID)
    assert L.ifeval_probe_flags([cfg]) == L.ifeval_probe_flags([leg, cfg]) == (*C.IFEVAL_DRIVER_FLAGS, *tail)
    assert L.ifeval_probe_flags([lens]) == (*C.IFEVAL_DRIVER_FLAGS, C.FLAG_EXPECT_OBJECT_SHA256, *tail)
    assert C.IFEVAL_DRIVER_FLAGS == (C.FLAG_SAMPLING, C.FLAG_TINYCAP_IDENTITY_VERSION, C.FLAG_MAX_TOKENS_BY_TASK,
                                     C.FLAG_OFF_CAP_BY_TASK, C.FLAG_ITEM_SUBSET, C.FLAG_BANK_VIEW)
    # every probed flag is VALUED (its dummy from DRIVER_PROBE_VALUES), never bare
    argv = L.driver_probe_argv("/t", extra_flags=L.ifeval_probe_flags([lens]))
    for flag in (*C.IFEVAL_DRIVER_FLAGS, C.FLAG_EXPECT_OBJECT_SHA256, *tail):
        assert argv[argv.index(flag) + 1] == L.DRIVER_PROBE_VALUES[flag], flag
    assert L.DRIVER_PROBE_VALUES[C.FLAG_ATTEMPT_ID] == "a1" == C.check_attempt_id_arg("a1")
    assert argv[-1] == L.DRIVER_PROBE_SENTINEL
    # the preflight's probe composes them beside the existing terms (the existing pins stand)
    src = TL.LAUNCHER.read_text()
    body = src[src.index("def preflight("):src.index("    def datasets():")]
    assert "probe_flags = ((C.FLAG_ALLOW_OOS_THINKING,) if oos else ()) + (" in body
    assert "(C.FLAG_EXPECT_OBJECT_SHA256,) if any(landed_lens_configuration(c) for c in cfgs) else ())" in body
    assert "tuple(C.TINYCAP_LADDER_DRIVER_FLAGS) if ladder_launch is not None else ()) + (" in body
    assert "            ifeval_probe_flags(cfgs))\n        r = subprocess.run(\n            driver_probe_argv(tree, extra_flags=probe_flags)," in body
    assert "the IFEval driver tail lands with ifeval-canonical-battery" in body


def test_launch_argv_spells_the_four_ifeval_flags_only_when_given_and_main_threads_them(L):
    base = ["modal", "run", "scripts/modal/tinycap_served.py", "--sha", "abc", "--model", QWEN, "--pass1", "--controls", "ifeval"]
    assert L.launch_argv("abc", pass1=True, model=QWEN, controls="ifeval") == base
    assert L.launch_argv("abc", pass1=True, model=QWEN, controls="ifeval", off_cap_by_task="", item_subset="", bank_view="",
                         out_dir="") == base
    assert L.launch_argv("abc", pass1=True, model=QWEN, controls="ifeval", off_cap_by_task="ifeval=4096",
                         item_subset="/vol/w/rung_subset.json", bank_view="main", out_dir="/vol/w/out") == base + [
        C.FLAG_OFF_CAP_BY_TASK, "ifeval=4096", C.FLAG_ITEM_SUBSET, "/vol/w/rung_subset.json",
        C.FLAG_BANK_VIEW, "main", L.OUT_DIR_FLAG, "/vol/w/out"]
    assert L.OUT_DIR_FLAG == "--out-dir"
    # the recorded line stays byte-identical for every other launch (the launcher test's own cases)
    assert L.launch_argv("abc", pass1=True) == ["modal", "run", "scripts/modal/tinycap_served.py", "--sha", "abc", "--pass1"]
    # main / plan_for / smoke_plan take the four; main parses the setter with the statement's grammar
    for fn in (L.main, L.launch_argv):
        params = inspect.signature(fn).parameters
        assert all(params[k].default == "" for k in ("off_cap_by_task", "item_subset", "bank_view", "out_dir")), fn
    for fn in (L.plan_for, L.smoke_plan, L.plan_configurations):
        params = inspect.signature(fn).parameters
        assert all(params[k].default is None and params[k].kind is inspect.Parameter.KEYWORD_ONLY
                   for k in ("off_cap_by_task", "item_subset", "bank_view", "out_dir")), fn
    assert L.ifeval_plan_args_of() == {}
    assert L.ifeval_plan_args_of(off_cap_by_task="ifeval=4096", item_subset="/s", bank_view="it", out_dir="/o") == {
        "off_cap_by_task": {"ifeval": 4096}, "item_subset": "/s", "bank_view": "it", "out_dir": "/o"}
    with pytest.raises(ValueError, match="--off-cap-by-task"):
        L.ifeval_plan_args_of(off_cap_by_task="ifeval=lots")
    src = TL.LAUNCHER.read_text()
    m = src[src.index("def main("):]
    assert "plan_args.update(ifeval_plan_args_of(off_cap_by_task=off_cap_by_task, item_subset=item_subset," in m
    assert 'ap.add_argument(C.FLAG_OFF_CAP_BY_TASK, dest="off_cap_by_task", default="",' in m
    assert 'ap.add_argument(OUT_DIR_FLAG, dest="out_dir", default="",' in m
    assert "off_cap_by_task=a.off_cap_by_task, item_subset=a.item_subset, bank_view=a.bank_view, out_dir=a.out_dir)" in m
    # plan_for threads them to both planners (the preflight replans the same rung / view / root)
    (via,) = L.plan_for(cells=["none"], modes=["off"], controls=["ifeval"], model=GEMMA, sampling_mode=C.SAMPLING_VENDOR_PRESET,
                        samples=2, off_cap_by_task={"ifeval": 4096}, item_subset="/s", bank_view="it", out_dir="/o")
    assert (via["ifeval_cap"], via["item_subset"], via["bank_view"], via["out_dir"]) == (4096, "/s", "it", "/o")


def test_the_loader_names_the_two_ifeval_lines_the_image_bakes_punkt_and_the_stage_pins_ifeval(L):
    """decisions I1: the checker's `ensure_nltk_resource()` reads ONLY `<home>/.cache/nltk_data`, so the
    image bakes punkt + punkt_tab there right after the client venv layer, checks the loader returns
    WITHOUT downloading, and names the directory to nltk; the preflight's loader proves both offline."""
    code = L.INSPECT_LOADER_CODE
    assert 'go("inspect_evals.ifeval", lambda: ie("ifeval.ifeval", "ifeval")())' in code
    assert 'go("lane.ifeval_checker_nltk", _ifeval_checker_nltk)' in code
    assert "from instruction_following_eval import evaluation as _ife" in code and "_ife.ensure_nltk_resource()" in code
    assert "nltk.download = _refuse" in code
    assert code.index('go("inspect_evals.xstest"') < code.index('go("inspect_evals.ifeval"') < code.index("out[sys.argv[3]]")
    cmds = L.ifeval_nltk_bake_commands()
    assert cmds[0] == ('/opt/client/bin/python -c "import nltk; '
                       "nltk.download('punkt', download_dir='/root/.cache/nltk_data', quiet=True); "
                       "nltk.download('punkt_tab', download_dir='/root/.cache/nltk_data', quiet=True)\"")
    assert cmds[1] == f'{L.CLIENT_PY} -c "{L.ifeval_nltk_check_code()}"' and len(cmds) == 2
    check = L.ifeval_nltk_check_code()
    assert '"' not in check and "ife.ensure_nltk_resource()" in check and "nltk.download = lambda" in check
    assert "nltk.data.find('tokenizers/punkt', paths=['/root/.cache/nltk_data'])" in check
    assert "nltk.data.find('tokenizers/punkt_tab', paths=['/root/.cache/nltk_data'])" in check
    assert L.NLTK_DATA_DIR == "/root/.cache/nltk_data" and L.NLTK_PUNKT_RESOURCES == ("punkt", "punkt_tab")
    assert L.CLIENT_PY == "/opt/client/bin/python"
    # the checker in the client lock calls exactly that function (this venv is built from the same pins)
    pytest.importorskip("instruction_following_eval", reason="the tinycap CLIENT venv's IFEval checker (the hashed lock's VCS pin) — not a dependency of the CPU-suite environment")
    from instruction_following_eval import evaluation as ife
    assert callable(ife.ensure_nltk_resource)
    # layer order: client venv < the bake < the sibling-module copy layer < .env (with NLTK_DATA) — the existing
    # pins stand (the Harbor lock + venv layers are gone with Terminal-Bench)
    src = TL.LAUNCHER.read_text()
    i_client = src.index(".run_commands(*client_venv_commands())")
    i_bake = src.index(".run_commands(*ifeval_nltk_bake_commands())")
    i_copy = src.index("add_local_file(str(Path(tinycap_images.__file__).resolve()), TINYCAP_IMAGES_REMOTE, copy=True)")
    i_env = src.index('.env({"HF_HOME"')
    assert i_client < i_bake < i_copy < i_env
    assert "harbor" not in src.lower()
    env_block = src[i_env:src.index("})", i_env)]
    assert '"NLTK_DATA": NLTK_DATA_DIR' in env_block
    assert len(L.client_venv_commands()) == 6                       # the venv recipe itself is untouched
    # the stage pins the dataset and pins.yaml resolves it at the contracts' revision
    assert "ifeval" in L.STAGE_PINNED_KEYS
    pins = yaml.safe_load((REPO / "configs/capeval/pins.yaml").read_text())["datasets"]
    assert pins["ifeval"]["revision"] == C.IFEVAL_DATASET_REVISION and len(C.IFEVAL_DATASET_REVISION) == 40
    assert pins["ifeval"]["repo_id"] == C.IFEVAL_DATASET_PATH == "google/IFEval"
    # the interfaces doc names the plumbing
    doc = (REPO / "docs/tinycap_interfaces.md").read_text()
    for needle in ("ifeval_nltk_bake_commands", "IFEVAL_DRIVER_FLAGS", "IfevalGeometryPooled", "REC_OK_STATUSES", "IFEVAL_DISPATCH_STATE"):
        assert needle in doc, needle


def test_rec_ok_statuses_and_the_dispatch_state_name_are_what_the_dispatcher_reads(L):
    """The dispatcher (`scripts/tinycap_ifeval_dispatch.py`) reads `launcher.REC_OK_STATUSES` and
    `C.IFEVAL_DISPATCH_STATE` with fallbacks; both exist, so the fallbacks are never used. Its
    `object_digests` shape (a lens row) is exactly what `object_digest_for` reads."""
    assert L.REC_OK_STATUSES == ("OK", "DONE", "PARTIAL_COMPLETE") and L.LADDER_REC_STATUS == "PARTIAL_COMPLETE"
    assert C.IFEVAL_DISPATCH_STATE == "dispatch_state.json" and "IFEVAL_DISPATCH_STATE" in C.__all__
    import importlib.util
    spec = importlib.util.spec_from_file_location("ifeval_dispatch_under_test", REPO / "scripts/tinycap_ifeval_dispatch.py")
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    assert D.IFEVAL_DISPATCH_STATE == C.IFEVAL_DISPATCH_STATE
    assert set(D.REC_OK_FALLBACK) <= set(L.REC_OK_STATUSES)
    digests = D.object_digests_for({"slug": QWEN, "cell": "safety_refusal__psr_kl_all", "object_sha256": "ab" * 32})
    assert digests == {"safety_refusal__psr_kl_all": {"artifact_sha256": "ab" * 32, "intervention_hash": None,
                                                      "expected": None, "lens_route_intervention_hash": None}}
    assert L.object_digest_for("safety_refusal__psr_kl_all", digests) == "ab" * 32
    assert D.object_digests_for({"slug": QWEN, "cell": "none", "object_sha256": None}) is None
    with pytest.raises(ValueError, match="no preflight-verified object digest"):
        L.object_digest_for("safety_refusal__psr_kl_all", None)


def test_the_expected_identity_of_an_ifeval_configuration_is_version_2_and_a_v2_store_resumes(L, tmp_path):
    """The hashed identity the driver stamps under `--tinycap-identity-version 2 --sampling vendor_preset
    --samples 2 --max-tokens-by-task ifeval=<cap> [--item-subset] --bank-view`: `identity_version: 2`,
    the EXECUTED cap map `{inspect/ifeval: <cap>}`, `cap_rung` on a rung (over the subset file), the k
    map `{inspect/ifeval: 2}`, IFEval's seed rule, and qwen's three optional vendor_preset terms at the
    top of the block. The launcher's expectation carries the same keys through the SAME derivation
    (`tinycap_identity_fields`), so a v2 store built from it passes `untrusted_store_reason`, a legacy
    store is refused naming `identity_version`, and a store at another cap naming `max_tokens_by_task`."""
    (cfg,) = _plan(L)
    geom = L.configuration_geometry(cfg, 1)
    exp = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=cfg["samples"],
                              reduction_mode=C.REDUCTION_MODE_STACKED, sampling_mode=cfg["sampling_mode"], tree_sha=TL.FULL_SHA)
    assert exp.identity_version == 2 and exp.identity_v2 and exp.cap_rung is None
    assert exp.samples == 2 and exp.sampling_mode == C.SAMPLING_VENDOR_PRESET
    assert exp.sampling[C.IDENTITY_KEY_SAMPLES_BY_TASK] == {C.IFEVAL_TASK: 2}
    assert exp.sampling[C.IDENTITY_KEY_SEED_RULE][C.IFEVAL_TASK] == C.SEED_RULE_IFEVAL
    assert exp.sampling_optional == ("min_p", "presence_penalty", "repetition_penalty")       # qwen35_4b's vendor row
    opt = exp.optional_identity()
    assert opt == {"min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
                   C.IDENTITY_KEY_IDENTITY_VERSION: 2, C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: {C.IFEVAL_TASK: 2048}}
    d = exp.as_dict()
    assert d[C.IDENTITY_KEY_IDENTITY_VERSION] == 2 and d[C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {"inspect/ifeval": 2048}
    assert C.IDENTITY_KEY_CAP_RUNG not in d and C.IDENTITY_KEY_THINKING_HEADER_IDS not in d       # OFF: no header ids
    # the driver's own exact-keys derivation, with the inputs it passes, names exactly these keys
    assert set(d) - set(TL._expected(L).as_dict()) == set(C.tinycap_identity_fields(
        sampling_optional=exp.sampling_optional, identity_version=2, max_tokens_by_task=True)) - set(C.TINYCAP_IDENTITY_FIELDS)
    # gemma's vendor row sets no optional term: the version-2 keys alone
    (g,) = _plan(L, GEMMA)
    ge = L.expected_identity(g, L.configuration_geometry(g, 1), pass_no=1, controls=g["controls"], samples=2,
                             reduction_mode=C.REDUCTION_MODE_STACKED, sampling_mode=C.SAMPLING_VENDOR_PRESET)
    assert set(ge.optional_identity()) == {C.IDENTITY_KEY_IDENTITY_VERSION, C.IDENTITY_KEY_MAX_TOKENS_BY_TASK}
    # a legacy configuration's expectation is byte-identical to before: version 1, no version key at all
    leg = TL._expected(L)
    assert leg.identity_version == 1 and not leg.identity_v2 and leg.optional_identity() == {}
    assert C.IDENTITY_KEY_IDENTITY_VERSION not in leg.as_dict() and leg.sampling_optional == ()
    # a rung: cap_rung over the SUBSET FILE (the driver reads the same file through the same function)
    doc = _subset_doc()
    sub = tmp_path / "rung_subset.json"
    sub.write_text(json.dumps(doc))
    (r,) = _plan(L, off_cap_by_task={"ifeval": 4096}, item_subset=str(sub), out_dir=str(tmp_path / "out"))
    rexp = L.expected_identity(r, L.configuration_geometry(r, 1), pass_no=1, controls=r["controls"], samples=2,
                               reduction_mode=C.REDUCTION_MODE_STACKED, sampling_mode=C.SAMPLING_VENDOR_PRESET)
    assert rexp.cap_rung == C.cap_rung_identity(doc) and tuple(rexp.cap_rung) == C.CAP_RUNG_FIELDS
    assert rexp.optional_identity()[C.IDENTITY_KEY_CAP_RUNG] == C.cap_rung_identity(doc)
    assert rexp.optional_identity()[C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {C.IFEVAL_TASK: 4096}
    # ... and an unreadable subset (an operator-side dry run of a volume path) refuses by name
    (bad,) = _plan(L, off_cap_by_task={"ifeval": 4096}, item_subset="/vol/nowhere/rung_subset.json")
    with pytest.raises(L.IdentityUnderivable, match="cap_rung"):
        L.expected_identity(bad, L.configuration_geometry(bad, 1), pass_no=1, controls=bad["controls"], samples=2,
                            reduction_mode=C.REDUCTION_MODE_STACKED, sampling_mode=C.SAMPLING_VENDOR_PRESET)
    # a hand-built IFEval cfg filing another task beside ifeval cannot be valued (the map is the atom's alone)
    mixed = dict(cfg, controls=[C.CONTROL_IFEVAL, "gsm8k"])
    with pytest.raises(L.IdentityUnderivable, match="atom alone"):
        L.expected_identity(mixed, geom, pass_no=1, controls=mixed["controls"], samples=2,
                            reduction_mode=C.REDUCTION_MODE_STACKED, sampling_mode=C.SAMPLING_VENDOR_PRESET)
    with pytest.raises(L.IdentityUnderivable, match="no ifeval_cap"):
        L.expected_identity({k: v for k, v in cfg.items() if k != "ifeval_cap"}, geom, pass_no=1, controls=cfg["controls"],
                            samples=2, reduction_mode=C.REDUCTION_MODE_STACKED, sampling_mode=C.SAMPLING_VENDOR_PRESET)
    # the store-trust check: a v2 store shaped as the driver writes it (the expectation's optional keys
    # at the top of the block) is trusted; a legacy-shaped store and another cap's store are refused by name
    ok = TL._store(tmp_path / "ok", exp, tasks=[C.CONTROL_IFEVAL], sampling_mode=C.SAMPLING_VENDOR_PRESET,
                   extra_identity=exp.optional_identity())
    assert L.untrusted_store_reason(ok, expected=exp) is None
    legacy = TL._store(tmp_path / "legacy", exp, tasks=[C.CONTROL_IFEVAL], sampling_mode=C.SAMPLING_VENDOR_PRESET)
    why = L.untrusted_store_reason(legacy, expected=exp)
    assert why and C.IDENTITY_KEY_IDENTITY_VERSION in why and "identity-version-2 runs" in why
    other_cap = TL._store(tmp_path / "cap", exp, tasks=[C.CONTROL_IFEVAL], sampling_mode=C.SAMPLING_VENDOR_PRESET,
                          extra_identity={**exp.optional_identity(), C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: {C.IFEVAL_TASK: 4096}})
    why = L.untrusted_store_reason(other_cap, expected=exp)
    assert why and f"tinycap.{C.IDENTITY_KEY_MAX_TOKENS_BY_TASK}" in why and "4096" in why
    # a legacy expectation never reads a v2 store (the version key's presence is hashed)
    v2_for_legacy = TL._store(tmp_path / "v2leg", leg, extra_identity={C.IDENTITY_KEY_IDENTITY_VERSION: 2,
                                                                       C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: {"inspect/mmlu": 1}})
    why = L.untrusted_store_reason(v2_for_legacy, expected=leg)
    assert why and C.IDENTITY_KEY_IDENTITY_VERSION in why
    # the run body hands the container's expectation the PLAN's draw and IFEval's k (read above), and the
    # dry run reads the same two off the cfg — the draw through main's ONE accessor
    src = TL.LAUNCHER.read_text()
    assert "sampling_mode=sampling_mode_of(cfg))" in src
    assert 'samples=int(cfg.get("samples", samples)), limit=limit, inspect_limit=inspect_limit,' in src


def test_the_expected_identity_of_a_draw_level_rung_hashes_the_planners_draws_and_a_v2_store_resumes(L, tmp_path, monkeypatch):
    """`expected_identity()` read the rung subset with the v1-only validator, so a
    `rung_subset/2` file (the ladder planner's ONLY output for IFEval) raised `IdentityUnderivable` before any standup.
    Now the contracts' reader dispatches on `schema`: the PLANNER-PRODUCED v2 subset — `inspect_rung.
    classify_store` over a synthetic pass-1 store whose at-cap draws are twelve SECOND trials (12 / 1,082 > 1 %) — is
    read, `cap_rung.subset_sha16` is the digest over the DRAWS (lane 2's `cap_rung_identity_v2`, equal by construction),
    a v2 store carrying that identity passes the store-trust reconciliation, and a store stamped with the ITEM-level
    digest over the same items is refused naming `cap_rung` — two rungs over one item set but different draws never
    resume onto each other."""
    import ifeval_synthetic_store as G
    from sisteer.capeval.tinycap import ifeval_contracts as IC
    row = G.row_for(G.matrix_rows(), "qwen35_4b", "main", "none")
    src_cell = G.build_store(tmp_path / "gen", row, at_cap={(i, 1) for i in range(12)}, cap=2048)
    record, subset = G.classify(src_cell.parents[1])
    assert record["exit"] == "escalate" and subset is not None and subset["schema"] == C.ITEM_SUBSET_SCHEMA_V2
    assert subset["n_draws"] == 12 and all(t == 1 for _, t in subset["draws"])      # trial-1-only, the planner's own
    sub = G.write_rung_subset(tmp_path / "gen", row, subset, rung=4096)
    (r,) = _plan(L, off_cap_by_task={"ifeval": 4096}, item_subset=str(sub), out_dir=str(tmp_path / "out"))
    rexp = L.expected_identity(r, L.configuration_geometry(r, 1), pass_no=1, controls=r["controls"], samples=2,
                               reduction_mode=C.REDUCTION_MODE_STACKED, sampling_mode=C.SAMPLING_VENDOR_PRESET, tree_sha=TL.FULL_SHA)
    assert rexp.cap_rung == C.cap_rung_identity(subset) == IC.cap_rung_identity_v2(subset)
    assert rexp.cap_rung["subset_sha16"] == C.item_subset_sha16_v2(subset["draws"]) != C.item_subset_sha16(subset["ids"])
    assert rexp.optional_identity()[C.IDENTITY_KEY_CAP_RUNG] == rexp.cap_rung and tuple(rexp.cap_rung) == C.CAP_RUNG_FIELDS
    assert rexp.optional_identity()[C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {C.IFEVAL_TASK: 4096}
    # the store-trust reconciliation: the driver's v2 rung store (the same identity) resumes ...
    ok = TL._store(tmp_path / "ok", rexp, tasks=[C.CONTROL_IFEVAL], sampling_mode=C.SAMPLING_VENDOR_PRESET,
                   extra_identity=rexp.optional_identity())
    assert L.untrusted_store_reason(ok, expected=rexp) is None
    # ... a store whose cap_rung is the ITEM digest over the same items (the pre-fix producer's reading) does not ...
    item_level = {**rexp.cap_rung, "subset_sha16": C.item_subset_sha16(subset["ids"])}
    stale = TL._store(tmp_path / "stale", rexp, tasks=[C.CONTROL_IFEVAL], sampling_mode=C.SAMPLING_VENDOR_PRESET,
                      extra_identity={**rexp.optional_identity(), C.IDENTITY_KEY_CAP_RUNG: item_level})
    why = L.untrusted_store_reason(stale, expected=rexp)
    assert why and f"tinycap.{C.IDENTITY_KEY_CAP_RUNG}" in why
    # ... and the pass-1 store of the same cell (no cap_rung) is refused by name too
    (p1,) = _plan(L, out_dir=str(tmp_path / "p1"))
    pexp = L.expected_identity(p1, L.configuration_geometry(p1, 1), pass_no=1, controls=p1["controls"], samples=2,
                               reduction_mode=C.REDUCTION_MODE_STACKED, sampling_mode=C.SAMPLING_VENDOR_PRESET, tree_sha=TL.FULL_SHA)
    pass1 = TL._store(tmp_path / "pass1", pexp, tasks=[C.CONTROL_IFEVAL], sampling_mode=C.SAMPLING_VENDOR_PRESET,
                      extra_identity=pexp.optional_identity())
    why = L.untrusted_store_reason(pass1, expected=rexp)
    assert why and C.IDENTITY_KEY_CAP_RUNG in why


# --------------------------------------------------------------------------- #
# the follow-up: Appendix A's two argv gaps, the sampler                       #
# attestation, and the container-written IFEval prompt budget                  #
# --------------------------------------------------------------------------- #
def test_the_ifeval_driver_line_carries_the_budget_artifact_and_a_fresh_attempt_id(L, tmp_path):
    """Plan Appendix A / merge finding M6: `needs_prompt_budget` admits every IFEval configuration (the container writes
    the artifact for the cell's prompt class and cap), `_driver_cmd` appends `--attempt-id a<k>` LAST on an IFEval
    configuration and REFUSES one offered to any other, and `next_attempt_id` mints a<k> from the store's recorded
    attempts (a1 on a fresh root; a torn manifest refuses by name)."""
    ep = {"proxy_url": "u", "routing_table": "t"}
    (cfg,) = _plan(L)
    geom = L.configuration_geometry(cfg, 1)
    assert L.needs_prompt_budget(cfg, cfg["controls"]) is True and L.budget_caps(cfg) == {C.IFEVAL_TASK: 2048}
    (rung,) = _plan(L, GEMMA, off_cap_by_task={"ifeval": 8192}, item_subset="/vol/w/rung_subset.json", out_dir="/vol/w/out")
    assert L.budget_caps(rung) == {C.IFEVAL_TASK: 8192}
    kw = dict(controls=cfg["controls"], limit=0, inspect_limit=0, samples=cfg["samples"], cfg_path="<cfg>", vllm_log=None)
    cmd = L._driver_cmd("<tree>", cfg, ep, None, geom, oos_prompt_budget="/vol/out/oos_prompt_budget.json", attempt_id="a3", **kw)
    assert cmd[-2:] == [C.FLAG_ATTEMPT_ID, "a3"] and cmd[cmd.index(C.FLAG_OOS_PROMPT_BUDGET) + 1] == "/vol/out/oos_prompt_budget.json"
    assert cmd[cmd.index(C.FLAG_SAMPLING):-2] == [C.FLAG_SAMPLING, C.SAMPLING_VENDOR_PRESET, C.FLAG_TINYCAP_IDENTITY_VERSION, "2",
                                                  C.FLAG_MAX_TOKENS_BY_TASK, "ifeval=2048", C.FLAG_BANK_VIEW, "main"]
    # without an attempt id the argv is byte-identical to before (the composer's None rule)
    assert L._driver_cmd("<tree>", cfg, ep, None, geom, **kw) == C.driver_argv(
        python=L.CLIENT_PY, script="<tree>/scripts/served_capability_run.py", proxy_url="u", routing_table="t", cell="none",
        config="<cfg>", defenses_dir=L.row_of(cfg).objects_dir, out_root=cfg["out_dir"], controls=cfg["controls"], thinking="off",
        concurrency=geom.max_num_seqs, judge_model=L.JUDGE_MODEL, samples=2, top_logprobs=L.row_of(cfg).top_logprobs,
        sampling_mode=C.SAMPLING_VENDOR_PRESET, identity_version=2, max_tokens_by_task={"ifeval": 2048}, bank_view="main")
    with pytest.raises(C.AttemptScopeError, match="must be a<k>"):
        L._driver_cmd("<tree>", cfg, ep, None, geom, attempt_id="3", **kw)
    (leg,) = L.plan_configurations(cells=["none"], modes=["off"], controls=["gsm8k"])
    with pytest.raises(ValueError, match="attempt-id was offered to a configuration that is not an IFEval one"):
        L._driver_cmd("<tree>", leg, ep, None, L.configuration_geometry(leg, 1), attempt_id="a1",
                      controls=leg["controls"], limit=0, inspect_limit=0, samples=1, cfg_path="<cfg>", vllm_log=None)
    assert L.needs_prompt_budget(leg, leg["controls"]) is False and L.budget_caps(leg) == {}
    # the attempt counter reads the stores' recorded attempts
    assert L.next_attempt_id(tmp_path) == "a1"
    store = tmp_path / "cells" / "k1"
    store.mkdir(parents=True)
    (store / "manifest.json").write_text(json.dumps({C.MANIFEST_KEY_RESUME_ATTEMPTS: [{"attempt_id": "a1"}, {"attempt_id": "a2"}]}))
    assert L.next_attempt_id(tmp_path) == "a3"
    (store / "manifest.json").write_text("{torn")
    with pytest.raises(ValueError, match="cannot read the store's manifest to count its attempts"):
        L.next_attempt_id(tmp_path)
    # the run body mints the id and passes it; the dry run spells a1
    src = TL.LAUNCHER.read_text()
    body = src[src.index("def _run_configuration("):src.index("def run_configuration(")]
    assert "attempt_id = next_attempt_id(out_dir) if ifeval_cfg else None" in body and "attempt_id=attempt_id," in body
    # the pin follows the line: the call's closer is now the K / phase pair
    assert "concurrency_k=concurrency_k, verification_phase=verification_phase)" in body
    assert 'attempt_id=("a1" if is_ifeval_configuration(cfg) else None))' in src
    # the artifact is written IN the container for an IFEval configuration handed none, held to the served template after
    assert "oos_prompt_budget = run_ifeval_prompt_budget(tree, row, cfg, str(budget_out), tree_sha=tree_sha)" in body
    assert "caps = budget_caps(cfg)" in body and "budget=(oos_prompt_budget if (oos_on or offcap) else None)" in body
    assert "if refusal is None and budgeted and ifeval_cfg and not (oos_on or offcap):" in body


def test_the_container_writes_the_ifeval_prompt_budget_for_the_cells_own_class_and_cap(L):
    """`ifeval_prompt_budget_argv`: the budget script's `--task ifeval` branch in the client venv against the staged
    tree — the row's model key + overlay config, `--system-prompt-cell` on a prompted cell only (the constitution SI
    or the safety SI), the executed cap, `--geometry-eval` = the configuration's ONE key (its .fit.json sidecar holds
    max + cap to THAT window), offline, tree-sha scoped, `--out` LAST."""
    (cfg,) = _plan(L)
    argv = L.ifeval_prompt_budget_argv("/tree", L.row_of(cfg), cfg, "/out/oos_prompt_budget.json", tree_sha="a" * 40)
    assert argv[:2] == [L.CLIENT_PY, f"/tree/{C.OOS_PROMPT_BUDGET_SCRIPT}"] and argv[-2:] == ["--out", "/out/oos_prompt_budget.json"]
    valued = [a for a in argv[2:] if a != "--offline"]                # the one bare switch
    pairs = dict(zip(valued[::2], valued[1::2]))
    assert pairs["--task"] == "ifeval" and pairs["--arm"] == "off" and pairs["--model-key"] == "qwen35_4b"
    assert pairs["--config"] == f"/tree/{L.row_of(cfg).cfg}" and pairs["--cap"] == "2048"
    assert pairs["--geometry-eval"] == "capability:ifeval_2048" and pairs["--tree-sha"] == "a" * 40 and "--offline" in argv
    assert "--system-prompt-cell" not in argv                      # a bare cell renders the bare messages
    (con,) = _plan(L, GEMMA, cell=C.CELL_PROMPTED_CONSTITUTION, off_cap_by_task={"ifeval": 4096},
                   item_subset="/vol/w/rung_subset.json", out_dir="/vol/w/out")
    cargv = L.ifeval_prompt_budget_argv("/tree", L.row_of(con), con, "/o.json", tree_sha="b" * 40)
    cvalued = [a for a in cargv[2:] if a != "--offline"]
    cp = dict(zip(cvalued[::2], cvalued[1::2]))
    assert cp["--system-prompt-cell"] == C.CELL_PROMPTED_CONSTITUTION and cp["--cap"] == "4096"
    assert cp["--geometry-eval"] == "capability:ifeval_longsi_4096" and cp["--model-key"] == "gemma4e4b"
    (safe,) = _plan(L, cell="prompted__safety_refusal")
    sargv = L.ifeval_prompt_budget_argv("/t", L.row_of(safe), safe, "/o", tree_sha="c" * 40)
    assert sargv[sargv.index("--system-prompt-cell") + 1] == "prompted__safety_refusal"
    (lens,) = _plan(L, cell="constv2-mix-full__lora_r4_ce")
    assert "--system-prompt-cell" not in L.ifeval_prompt_budget_argv("/t", L.row_of(lens), lens, "/o", tree_sha="c" * 40)
    (leg,) = L.plan_configurations(cells=["none"], modes=["off"], controls=["gsm8k"])
    with pytest.raises(ValueError, match="not an IFEval configuration"):
        L.ifeval_prompt_budget_argv("/t", L.row_of(leg), leg, "/o", tree_sha="c" * 40)
    # the script's IFEval branch parses exactly this line
    import importlib.util as iu
    spec = iu.spec_from_file_location("budget_cli_under_port_test", REPO / C.OOS_PROMPT_BUDGET_SCRIPT)
    B = iu.module_from_spec(spec)
    spec.loader.exec_module(B)
    ns = B.build_parser().parse_args(cargv[2:])
    assert (ns.task, ns.arm, ns.model_key, ns.cap, ns.geometry_eval, ns.system_prompt_cell, ns.offline) == (
        "ifeval", "off", "gemma4e4b", 4096, "capability:ifeval_longsi_4096", C.CELL_PROMPTED_CONSTITUTION, True)


def test_every_standup_logs_the_resolved_sampler_and_an_ifeval_configuration_records_it_per_kind_against_the_pinned_row(L, tmp_path):
    """EVERY tinycap standup's FIXED argv carries
    `--enable-log-requests` exactly once, right after the proxy's upstream timeout — no per-row opt-in (the lane's
    flag / endpoint-key names are ALIASES of the contracts') — and the worker runs main's reader IN-PROCESS after
    the driver for every configuration (`record_resolved_sampler`), filing the record at the store root. The IFEval
    adoption (the per-kind mode on main): an IFEval configuration's expectation
    is the PINNED card-table row `C.VENDOR_PRESET_SAMPLING_BY_MODEL_KEY[row.model_key]` — `--expect` per key the row
    SETS (llama: temperature + top_p; nothing invented), `--probe-expect kinds` with `--probe-tags
    <store root>/request_evidence.jsonl` (ONE tag source per leg), the THREE kind caps a thinking-OFF leg fires declared
    explicitly (48 bites, 256 lens-ON bite, 512 render — never the reader's default table, whose 2,048 would collide
    with the pass-1 client cap) and `--client-cap <the row's executed cap>`; the cap-collision gate refuses by name
    BEFORE the reader runs. Every non-IFEval configuration's argv is main's byte for byte: one `--probe-cap 48`, no
    `--probe-expect` / `--probe-tags` / `--client-cap`. The attestation record carries the standup's flag."""
    geom = SimpleNamespace(max_model_len=2528, max_num_seqs=256, model_key="qwen35_4b")
    srv = tmp_path / "srv"
    base = L._standup_cmd("<tree>", "<cfg>", "none", "<ddir>", srv, srv / "STOP", geom, 64, [])
    assert base.count(C.FLAG_ENABLE_LOG_REQUESTS) == 1
    i = base.index(C.FLAG_UPSTREAM_TIMEOUT_S)
    assert base[i + 2] == C.FLAG_ENABLE_LOG_REQUESTS                 # the fixed part: right after the timeout pair
    assert L.STANDUP_ENABLE_LOG_REQUESTS_FLAG == C.FLAG_ENABLE_LOG_REQUESTS == "--enable-log-requests"
    assert L.ENDPOINT_KEY_ENABLE_LOG_REQUESTS == C.ENDPOINT_KEY_ENABLE_LOG_REQUESTS == "enable_log_requests"
    assert "enable_log_requests" not in inspect.signature(L._standup_cmd).parameters       # no opt-in any more
    import test_serve_declarations_attestation as SDA
    capctrl = SDA._load("_port_log_requests_capctrl", "scripts/capctrl_serving_up.py")
    assert capctrl.ENABLE_LOG_REQUESTS_FLAG == C.FLAG_ENABLE_LOG_REQUESTS
    assert capctrl.build_parser().parse_args(base[2:]).enable_log_requests is True
    # the run body records it for EVERY configuration, in-process, off the expectation of the run's mode, with the
    # IFEval row's executed cap as the client cap; the attestation record reads the standup's flag back
    src = TL.LAUNCHER.read_text()
    body = src[src.index("def _run_configuration("):src.index("def run_configuration(")]
    assert "enable_log_requests=" not in body and body.count("record_resolved_sampler(") == 1
    assert 'rec["resolved_sampler"] = record_resolved_sampler(' in body
    assert 'expectation=resolved_sampler_expectation(sampling_mode, row.model_key, list(rec["thinking_by_task"]))' in body
    assert 'client_cap=(int(cfg["ifeval_cap"]) if is_ifeval_configuration(cfg) else None))' in body
    assert L.full22_attestation_record({"enable_log_requests": True})[C.ENDPOINT_KEY_ENABLE_LOG_REQUESTS] is True
    assert L.full22_attestation_record({})[C.ENDPOINT_KEY_ENABLE_LOG_REQUESTS] is None
    # the mirrored kind caps ARE the driver's named constants (the driver is the container's subprocess, not an import)
    import re
    driver = (REPO / "scripts/served_capability_run.py").read_text()
    assert int(re.search(r"^LENS_ON_BITE_MAX_TOKENS = (\d+)", driver, re.M).group(1)) == L.LENS_ON_BITE_MAX_TOKENS == 256
    assert int(re.search(r"^RENDER_PROBE_MAX_TOKENS = (\d+)", driver, re.M).group(1)) == L.RENDER_PROBE_MAX_TOKENS == 512
    from sisteer.eval.attacks.served_target import PROBE_MAX_TOKENS
    assert PROBE_MAX_TOKENS == 48 == L.PROBE_MAX_TOKENS_DEFAULT
    # the IFEval expectation, per row: the PINNED table's set keys (floats), the per-kind probe reading, the three
    # OFF-leg kind caps, the tag source relative to the store root — and NOTHING of it on any other mode
    for slug, key in ((QWEN, "qwen35_4b"), (LLAMA, "llama8b"), (GEMMA, "gemma4e4b"), (NEMO, "nemotron3nano4b")):
        (cfg,) = _plan(L, slug)
        exp = L.resolved_sampler_expectation(L.sampling_mode_of(cfg), L.row_of(cfg).model_key, list(cfg["thinking_by_task"]))
        row = C.VENDOR_PRESET_SAMPLING_BY_MODEL_KEY[key]
        want = {k: float(getattr(row, k)) for k in C.SAMPLING_ROW_FIELDS if k != "source" and getattr(row, k) is not None}
        assert exp["expect"] == want and exp["uncompared"] == {}, key
        assert exp["source"] == f"contracts.VENDOR_PRESET_SAMPLING_BY_MODEL_KEY[{key!r}]"
        assert exp["probe_expect"] == "kinds" == L.RESOLVED_SAMPLER_PROBE_EXPECT_IFEVAL
        assert exp["probe_caps"] == [48, 256, 512] and exp["probe_tags"] == [C.REQUEST_EVIDENCE_FILENAME]
        assert int(cfg["ifeval_cap"]) not in exp["probe_caps"]        # the pass-1 client cap never collides with a declared kind cap
    lexp = L.resolved_sampler_expectation(C.SAMPLING_VENDOR_PRESET, "llama8b", [C.IFEVAL_TASK])
    assert lexp["expect"] == {"temperature": 0.6, "top_p": 0.9}       # llama's row sets no top_k: nothing is invented
    with pytest.raises(ValueError, match="no 'vendor_preset' row for model key 'llama3b'"):
        L.resolved_sampler_expectation(C.SAMPLING_VENDOR_PRESET, "llama3b", [C.IFEVAL_TASK])
    (leg,) = L.plan_configurations(cells=["none"], modes=["off"], controls=["gsm8k"])
    cexp = L.resolved_sampler_expectation(L.sampling_mode_of(leg), L.row_of(leg).model_key, list(leg["thinking_by_task"]))
    assert set(cexp) == {"expect", "uncompared", "source"}
    # ... composed into the reader's argv: the pinned --expect pairs, the three --probe-cap, --probe-expect kinds, the tag
    # source, --client-cap; a non-IFEval configuration's argv has exactly one --probe-cap 48 and none of the rest
    argv = L.record_resolved_sampler_cmd("/tree", Path("/srv/vllm_server.log"), Path("/out/r.json"), lexp["expect"],
                                         lexp["probe_caps"], probe_expect=lexp["probe_expect"],
                                         probe_tags=[f"/out/{t}" for t in lexp["probe_tags"]], client_cap=2048)
    assert argv[1:] == [f"/tree/{C.RESOLVED_SAMPLER_SCRIPT}", "--vllm-log", "/srv/vllm_server.log", "--out", "/out/r.json",
                        "--expect", "temperature=0.6", "--expect", "top_p=0.9",
                        "--probe-cap", "48", "--probe-cap", "256", "--probe-cap", "512", "--probe-expect", "kinds",
                        "--probe-tags", "/out/request_evidence.jsonl", "--client-cap", "2048"]
    assert "--strict" not in argv
    cargv = L.record_resolved_sampler_cmd("/tree", Path("/l"), Path("/o"), cexp["expect"], PROBE_MAX_TOKENS)
    assert cargv.count("--probe-cap") == 1 and cargv[cargv.index("--probe-cap") + 1] == "48"
    assert not any(f in cargv for f in ("--probe-expect", "--probe-tags", "--client-cap"))
    assert cargv == L.record_resolved_sampler_cmd("/tree", Path("/l"), Path("/o"), cexp["expect"], [PROBE_MAX_TOKENS])
    # the reader itself, IN-PROCESS on a synthetic engine log + the driver's tag file at the store root: the client
    # requests at the IFEval cap under the pinned row, ONE render probe at 512 under the row, two greedy bites at 48;
    # the record lands at the store root; render AGREES against the pinned row, bites AGREES against greedy,
    # lens_on_bite EMPTY (no thinking-ON object cell) — the whole record AGREES and nothing is unclassified
    (cfg,) = _plan(L)                                                  # qwen35_4b: the row with every optional term
    exp = L.resolved_sampler_expectation(L.sampling_mode_of(cfg), L.row_of(cfg).model_key, list(cfg["thinking_by_task"]))
    attest = ("(APIServer pid=1) INFO 09-12 02:05:04 [request_logger.py:28] `--enable-log-requests` is set but the minimum "
              "log level is higher than DEBUG. Only limited information will be logged.\n")
    params = ("(APIServer pid=1) INFO 09-12 02:06:00 [request_logger.py:63] Received request r{i}: params: SamplingParams(n=1, "
              "presence_penalty={pp}, frequency_penalty=0.0, repetition_penalty=1.0, temperature={t}, top_p={tp}, top_k={tk}, "
              "min_p=0.0, seed=7, stop=[], max_tokens={cap}), lora_request: None.\n")
    client = [params.format(i=i, pp=1.5, t=0.7, tp=0.8, tk=20, cap=2048) for i in range(3)]
    render = [params.format(i=7, pp=1.5, t=0.7, tp=0.8, tk=20, cap=512)]
    bites = [params.format(i=8, pp=0.0, t=0.0, tp=1.0, tk=0, cap=48), params.format(i=9, pp=0.0, t=0.0, tp=1.0, tk=0, cap=48)]
    srv.mkdir(parents=True, exist_ok=True)
    (srv / "vllm_server.log").write_text(attest + "".join(client + render + bites))
    out_dir = tmp_path / "store"
    out_dir.mkdir()
    tags = [{"task": f"{C.PROBE_TASK_PREFIX}{C.PROBE_KIND_RENDER}", "max_tokens": 512},
            {"task": f"{C.PROBE_TASK_PREFIX}{C.PROBE_KIND_BITES}", "max_tokens": 48},
            {"task": f"{C.PROBE_TASK_PREFIX}{C.PROBE_KIND_BITES}", "max_tokens": 48},
            {"task": C.IFEVAL_TASK, "max_tokens": 2048}]                  # a row task is not a tag
    (out_dir / C.REQUEST_EVIDENCE_FILENAME).write_text("\n".join(json.dumps(t) for t in tags) + "\n")
    block = L.record_resolved_sampler(str(REPO), srv, out_dir, expectation=exp, client_cap=int(cfg["ifeval_cap"]))
    assert block["rc"] == 0 and "error" not in block, block
    assert block["path"] == str(out_dir / C.RESOLVED_SAMPLER_FILENAME) and (out_dir / C.RESOLVED_SAMPLER_FILENAME).is_file()
    assert block["probe_expect"] == "kinds" and block["probe_caps"] == [48, 256, 512] and block["client_cap"] == 2048
    assert block["probe_tags"] == [str(out_dir / C.REQUEST_EVIDENCE_FILENAME)]
    assert block["expect"] == exp["expect"] and block["expect_source"] == exp["source"]
    rec = json.loads((out_dir / C.RESOLVED_SAMPLER_FILENAME).read_text())
    assert block["verdict"] == rec["verdict"] == "AGREES", rec["why"]
    assert rec["n_client_requests"] == 3 and rec["n_probe_requests"] == 3 and rec["n_unclassified_requests"] == 0
    assert rec["probe_caps"] == ["48", "256", "512"] and rec["probe_cap_source"] == "--probe-cap" and rec["client_caps"] == ["2048"]
    # (+ the standup's two kinds, EMPTY on this lane — its standup fires no load check / positive control; recorder-load-check-kinds)
    assert rec["probe_verdict_by_kind"] == {"render": "AGREES", "bites": "AGREES", "lens_on_bite": "EMPTY", "logprob_bites": "EMPTY",
                                            "load_check": "EMPTY", "positive_control": "EMPTY", "standup_controls": "EMPTY"}
    assert rec["n_probe_requests_by_kind"] == {"render": 1, "bites": 2, "lens_on_bite": 0, "logprob_bites": 0, "load_check": 0, "positive_control": 0, "standup_controls": 0}
    assert rec["classes"]["client"]["verdict"] == "AGREES" and rec["classes"]["probe"]["reference_kind"] == "kinds"
    # a probe the tags do not name is UNCLASSIFIED — a finding the whole record carries, never folded into a kind (the
    # untagged kind itself stays EMPTY: classification never defaults)
    (srv / "vllm_server.log").write_text(attest + "".join(client + render + bites + [params.format(i=11, pp=0.0, t=0.0, tp=1.0, tk=0, cap=256)]))
    block = L.record_resolved_sampler(str(REPO), srv, out_dir, expectation=exp, client_cap=int(cfg["ifeval_cap"]))
    rec = json.loads((out_dir / C.RESOLVED_SAMPLER_FILENAME).read_text())
    assert block["rc"] == 0 and block["verdict"] == rec["verdict"] == "UNCLASSIFIED_REQUESTS"
    assert rec["probe_verdict_by_kind"]["lens_on_bite"] == "EMPTY" and rec["n_probe_unclassified_requests"] == 1
    assert rec["classes"]["probe"]["unclassified"]["n"] == 1 and "no tag or declaration names max_tokens 256" in rec["why"]
    # the cap-collision gate: a client cap equal to a declared kind cap is refused by name BEFORE the reader runs
    (out_dir / C.RESOLVED_SAMPLER_FILENAME).unlink()
    gate = L.record_resolved_sampler(str(REPO), srv, out_dir, expectation=exp, client_cap=512)
    assert "error" in gate and "cap collision" in gate["error"] and "rc" not in gate
    assert not (out_dir / C.RESOLVED_SAMPLER_FILENAME).exists()
    # a staged tree without the reader: loud in the block, never a refusal of the cell
    bad = L.record_resolved_sampler(str(tmp_path / "no-such-tree"), srv, out_dir, expectation=exp, client_cap=2048)
    assert "error" in bad and "verdict" not in bad and C.RESOLVED_SAMPLER_SCRIPT in bad["error"]




def test_the_image_ships_the_vendored_constitution_text_the_si_grammar_asserts_at_import(L):
    """The wave shakedown (2026-09-13): Modal's ``add_local_python_source`` ships Python files ONLY, so the text
    ``sisteer.data.instructions`` vendors beside itself (sha256 asserted at import) was absent at
    ``/root/sisteer/data/`` in every container — five of five CPU preflights failed on the first in-process import of
    the SI grammar. The launcher ships it as a COPY layer (build steps cannot follow a mount; the copy layer survives
    ``build_image_of``'s peel so image_nemotron inherits it), declared in IMAGE_VENDORED_FILES (nested under the
    mounted package — NOT in IMAGE_LOCAL_FILES, which stays the flat beside-the-entrypoint set the sibling scripts'
    container-root tests rebuild), at the module's OWN file name under the mount's package root."""
    import hashlib
    from sisteer.data import instructions as I
    assert L.CONSTITUTION_FULL_REMOTE == f"/root/sisteer/data/{I.CONSTITUTION_FULL_PATH.name}" == "/root/sisteer/data/constitution_full.md"
    assert set(L.IMAGE_VENDORED_FILES) == {L.CONSTITUTION_FULL_REMOTE, L.DECODE_POLICY_REMOTE} and L.CONSTITUTION_FULL_REMOTE not in L.IMAGE_LOCAL_FILES
    local = L.IMAGE_VENDORED_FILES[L.CONSTITUTION_FULL_REMOTE]
    assert local == str(I.CONSTITUTION_FULL_PATH.resolve()) and Path(local).is_file()
    assert hashlib.sha256(Path(local).read_bytes()).hexdigest() == I.CONSTITUTION_FULL_SHA256
    src = TL.LAUNCHER.read_text()
    line = "add_local_file(str(_instructions.CONSTITUTION_FULL_PATH.resolve()), CONSTITUTION_FULL_REMOTE, copy=True)"
    assert src.count(line) == 1
    i_images = src.index("add_local_file(str(Path(tinycap_images.__file__).resolve()), TINYCAP_IMAGES_REMOTE, copy=True)")
    i_text, i_env, i_mount = src.index(line), src.index('.env({"HF_HOME"'), src.index('.add_local_python_source("sisteer")')
    assert i_images < i_text < i_env < i_mount                        # a copy layer under the mount, never after it


def test_the_image_ships_the_decode_policy_export_contracts_loads_at_import(L):
    """`contracts` loads `sisteer.registry.decode_policy` at import
    (3b1d87dae / 2699c762f); its default path is the package copy beside the module, which `add_local_python_source`
    (Python files ONLY) never ships — so every launcher container crash-looped at import (`NoPolicy: decode policy
    export not found at /configs/registry/decode_policy.json`, tinycap-bcbw-stage-3ba0202). The launcher ships the
    export as a COPY layer at the package copy's own path under the mount (the constitution text's pattern), declared
    in IMAGE_VENDORED_FILES, so the reader's `PACKAGE_EXPORT_PATH` resolves inside the container."""
    import hashlib
    import os
    from sisteer.registry import decode_policy as DP
    assert L.DECODE_POLICY_REMOTE == "/root/sisteer/registry/decode_policy.json"
    # the reader's package-copy path, relocated to where the mount puts `sisteer`, is exactly the remote path
    assert L.DECODE_POLICY_REMOTE == os.path.join("/root/sisteer/registry", os.path.basename(DP.PACKAGE_EXPORT_PATH))
    local = L.IMAGE_VENDORED_FILES[L.DECODE_POLICY_REMOTE]
    assert local == str(Path(DP.PACKAGE_EXPORT_PATH).resolve()) and Path(local).is_file()
    # the shipped bytes ARE the committed export (the repo copy and the package copy are held equal by the registry tests)
    assert hashlib.sha256(Path(local).read_bytes()).hexdigest() == hashlib.sha256((REPO / DP.EXPORT_RELPATH).read_bytes()).hexdigest()
    assert L.DECODE_POLICY_REMOTE not in L.IMAGE_LOCAL_FILES
    src = TL.LAUNCHER.read_text()
    line = "add_local_file(str(Path(_decode_policy.PACKAGE_EXPORT_PATH).resolve()), DECODE_POLICY_REMOTE, copy=True)"
    assert src.count(line) == 1
    i_text = src.index("add_local_file(str(_instructions.CONSTITUTION_FULL_PATH.resolve()), CONSTITUTION_FULL_REMOTE, copy=True)")
    i_policy, i_env, i_mount = src.index(line), src.index('.env({"HF_HOME"'), src.index('.add_local_python_source("sisteer")')
    assert i_text < i_policy < i_env < i_mount                        # a copy layer under the mount, never after it
    # a container root rebuilt from the vendored files carries the export where `default_export_path` looks
    import tempfile, shutil, subprocess, sys
    root = Path(tempfile.mkdtemp(prefix="decode_policy_root_"))
    for remote, src_path in L.IMAGE_VENDORED_FILES.items():
        dst = root / remote.lstrip("/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_path, dst)
    (root / "root" / "sisteer" / "registry").mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(DP.__file__), root / "root" / "sisteer" / "registry" / "decode_policy.py")
    probe = ("import importlib.util, sys; spec = importlib.util.spec_from_file_location('dp', sys.argv[1]); m = importlib.util.module_from_spec(spec); "
             "sys.modules['dp'] = m; spec.loader.exec_module(m); print(m.default_export_path())")
    out = subprocess.run([sys.executable, "-c", probe, str(root / "root" / "sisteer" / "registry" / "decode_policy.py")],
                         text=True, capture_output=True, env={"PATH": os.environ.get("PATH", "")})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str(root / "root" / "sisteer" / "registry" / "decode_policy.json")
    shutil.rmtree(root, ignore_errors=True)



# --------------------------------------------------------------------------- #
# the wave shakedown's container findings (2026-09-13), pinned                 #
# --------------------------------------------------------------------------- #


def test_keys_within_admits_the_rows_optional_sampling_terms_and_refuses_unknown_or_missing(L):
    base, opt = ("mode", "temperature"), ("min_p", "presence_penalty")
    assert L._keys_within({"mode": "x", "temperature": 0.7}, base, opt, "b") is None
    assert L._keys_within({"mode": "x", "temperature": 0.7, "presence_penalty": 1.5}, base, opt, "b") is None
    why = L._keys_within({"mode": "x", "temperature": 0.7, "foo": 1}, base, opt, "b")
    assert why and "unknown ['foo']" in why
    why = L._keys_within({"mode": "x", "presence_penalty": 1.5}, base, opt, "b")
    assert why and "missing ['temperature']" in why
    assert L._keys_within([], base, opt, "b") and "not a mapping" in L._keys_within([], base, opt, "b")


def test_untrusted_store_reason_admits_a_vendor_preset_store_whose_block_carries_the_rows_optional_terms(L, tmp_path):
    """The probe cell's COMPLETE store (driver rc 0, 1,082 draws, DONE marker) was filed INCOMPLETE because the launcher
    held `manifest.sampling` to the legacy key set — "unknown ['min_p', 'presence_penalty', 'repetition_penalty']" —
    while the driver's own guard admits the terms the row sets. The rule is now the driver's: base fields present, extras
    within the optional identity fields, every present term mirroring the hashed identity."""
    (cfg,) = L.plan_configurations(cells=["constv2-mix-full__lora_r4_ce"], modes=[C.THINKING_OFF], controls=[C.CONTROL_IFEVAL],
                                   model=QWEN, sampling_mode=C.SAMPLING_VENDOR_PRESET, samples=2,
                                   reduction_mode=C.REDUCTION_MODE_STACKED, bank_view="main")
    geom = L.configuration_geometry(cfg, 1)
    exp = L.expected_identity(cfg, geom, pass_no=1, controls=cfg["controls"], samples=2, reduction_mode=C.REDUCTION_MODE_STACKED,
                              sampling_mode=C.SAMPLING_VENDOR_PRESET, tree_sha=TL.FULL_SHA)
    assert exp.optional_identity().get("presence_penalty") == 1.5       # the Qwen vendor row sets the optional terms
    ok = TL._store(tmp_path / "ok", exp, tasks=[C.CONTROL_IFEVAL], sampling_mode=C.SAMPLING_VENDOR_PRESET,
                   extra_identity=exp.optional_identity())
    block = json.loads((ok / "manifest.json").read_text())[C.MANIFEST_KEY_SAMPLING]
    assert block["presence_penalty"] == 1.5 and block["min_p"] == 0.0 and block["repetition_penalty"] == 1.0   # the producer's shape
    assert L.untrusted_store_reason(ok, expected=exp) is None
    # a block that DROPS a term the row sets does not mirror the hashed identity — refused by name
    dropped = TL._store(tmp_path / "dropped", exp, tasks=[C.CONTROL_IFEVAL], sampling_mode=C.SAMPLING_VENDOR_PRESET,
                        extra_identity=exp.optional_identity(),
                        mutate=lambda m: m[C.MANIFEST_KEY_SAMPLING].pop("presence_penalty"))
    why = L.untrusted_store_reason(dropped, expected=exp)
    assert why and "presence_penalty" in why and "does not mirror" in why
    # a term that MOVED refuses too; an unknown key still refuses by name
    moved = TL._store(tmp_path / "moved", exp, tasks=[C.CONTROL_IFEVAL], sampling_mode=C.SAMPLING_VENDOR_PRESET,
                      extra_identity=exp.optional_identity(),
                      mutate=lambda m: m[C.MANIFEST_KEY_SAMPLING].update({"presence_penalty": 1.0}))
    assert "presence_penalty 1.0 does not mirror" in (L.untrusted_store_reason(moved, expected=exp) or "")
    odd = TL._store(tmp_path / "odd", exp, tasks=[C.CONTROL_IFEVAL], sampling_mode=C.SAMPLING_VENDOR_PRESET,
                    extra_identity=exp.optional_identity(), mutate=lambda m: m[C.MANIFEST_KEY_SAMPLING].update({"foo": 1}))
    assert "unknown ['foo']" in (L.untrusted_store_reason(odd, expected=exp) or "")


def test_the_nemotron_probe_derives_the_static_attestation_half_in_the_container(L):
    """main's standup attests the LIVE half only (`live_attest`), so the probe's `att.get("static")` fallback left
    `static.discovery` null and `read_attestation` refused every certification the writer had admitted ("no static
    discovery recorded"). The static half is derived in image_nemotron's own vLLM (`static_attest(contract)`)."""
    src = TL.LAUNCHER.read_text()
    assert src.count('att_static = dict(att.get("static") or mb.static_attest(contract))') == 1
    assert 'att.get("discovery")' not in src


def test_the_container_appends_the_result_stamp_the_ladder_planner_reads(L, tmp_path):
    """The planner's `store_status` requires a `result=COMPLETE` line (the cluster lane's RESULT stamp); no Modal
    container wrote one, so every Modal store read DISAGREE. One line per REC, COMPLETE iff the REC is an OK status."""
    import importlib
    import sys as _sys
    if str(REPO / "scripts") not in _sys.path:
        _sys.path.insert(0, str(REPO / "scripts"))
    LP = importlib.import_module("tinycap_ifeval_ladder_plan")
    ok_line = L.result_stamp_line({"name": "n", "status": "OK", "tree_sha": "a" * 40}, driver_rc=0)
    assert "result=COMPLETE" in ok_line and "status=OK" in ok_line and "rc=0" in ok_line and f"sha={'a' * 40}" in ok_line
    assert "result=INCOMPLETE" in L.result_stamp_line({"name": "n", "status": "INCOMPLETE", "tree_sha": "a" * 40}, driver_rc=0)
    assert "result=COMPLETE" in L.result_stamp_line({"name": "n", "status": "DONE", "tree_sha": "a" * 40}, driver_rc=None)
    src = TL.LAUNCHER.read_text()
    i_status = src.index('rec["status"] = (LADDER_REC_STATUS if rung else "OK") if complete else "INCOMPLETE"')
    i_stamp = src.index("with (out_dir / RESULT_FILENAME).open(\"a\", encoding=\"utf-8\") as fh:")
    assert i_status < i_stamp < src.index("return rec", i_stamp) and L.RESULT_FILENAME == "RESULT"
    # the planner reads it as the producer's completion stamp beside the outcomes + DONE marker
    from sisteer.capeval.tinycap import completion as _completion
    root = tmp_path / "store"
    for sub in ("outcomes", C.DONE_DIR):
        d = root / "cells" / "k" / sub
        d.mkdir(parents=True)
        (d / _completion.marker_filename(C.IFEVAL_TASK)).write_text("{}")
    (root / L.RESULT_FILENAME).write_text(ok_line + "\n")
    assert LP.store_status(root) == "complete"



def test_the_launchers_done_marker_reader_admits_the_draw_level_counts_of_a_k_gt_1_store(L, tmp_path):
    """The eb5170c probe cell's complete store (1,082 draws, a marker carrying n_draws / n_expected_draws — the P30
    draw-subset contract `completion.finish_task` writes and its own validator admits) was filed INCOMPLETE by the
    launcher's `invalid_done_marker_reason`, which held the marker to DONE_MARKER_FIELDS + the two legacy optional keys.
    The reader's optional set is now completion's: the draw fields are admitted, anything else still refuses."""
    from sisteer.capeval.tinycap import completion as _completion
    task = C.IFEVAL_TASK
    cd = tmp_path / "cell"
    (cd / C.DONE_DIR).mkdir(parents=True)
    (cd / "summary").mkdir()
    base = {"task": task, "n_items": 541, "n_rows": 1082, "n_requests": 1082, "n_rejected": 0, "finished_at": "2026-09-13T06:00:00Z",
            "wall_clock_s": 368.0}
    assert set(base) == set(C.DONE_MARKER_FIELDS)
    (cd / "summary" / _completion.marker_filename(task)).write_text(json.dumps({"n": 541, "task": task}))
    marker = cd / C.DONE_DIR / L.done_marker_name(task)
    marker.write_text(json.dumps({**base, "n_draws": 1082, "n_expected_draws": 1082}))
    why = L.invalid_done_marker_reason(cd, task)
    assert why is None or "done marker fields" not in why, why          # the draw fields are the contract's optional keys
    marker.write_text(json.dumps({**base, "foo": 1}))
    why = L.invalid_done_marker_reason(cd, task)
    assert why and "done marker fields" in why and "n_draws" in why    # an unknown key still refuses, naming the optional set



def test_the_prompt_budget_is_measured_under_the_arm_the_run_serves_not_available_on_llama(L):
    """The llama probe's driver refused the preflight's artifact — measured under
    `off`, the configuration's mode — because a llama IFEval row serves the certified no-switch arm `not_available`.
    Both the artifact write and the fit report's comparison now derive the arm through
    `served_budget_arm`: the family's effective arm on an IFEval row, the mode itself on every other row."""
    assert L.served_budget_arm(C.model_row(LLAMA), C.THINKING_OFF) == C.THINKING_NOT_AVAILABLE
    assert L.served_budget_arm(C.model_row("llama8b-sdfbase-main"), C.THINKING_OFF) == C.THINKING_NOT_AVAILABLE
    for slug in (GEMMA, QWEN, NEMO):
        assert L.served_budget_arm(C.model_row(slug), C.THINKING_OFF) == C.THINKING_OFF
    assert L.served_budget_arm(C.model_row("llama8b-f22"), C.THINKING_OFF) == C.THINKING_OFF     # a non-IFEval row: the mode as given
    src = TL.LAUNCHER.read_text()
    # the container's IFEval producer measures the served arm; the in-container fit and the preflight's OOS write + fit
    # report compare against it — no site budgets or compares the configuration's bare mode any more
    assert src.count('C.OOS_PROMPT_BUDGET_ARG_ARM, served_budget_arm(row, str(cfg["mode"])), "--model-key"') == 1
    assert src.count('fit = C.prompt_budget_fit(oos_prompt_budget, arm=served_budget_arm(row, str(cfg["mode"]))') == 1
    assert src.count('rec = C.prompt_budget_fit(budget, arm=served_budget_arm(row, str(cfg["mode"]))') == 1
    assert src.count("arm=served_budget_arm(row, arms[0])") == 1                                 # the preflight's artifact write
    assert "C.OOS_PROMPT_BUDGET_ARG_ARM, C.THINKING_OFF" not in src
    assert 'C.prompt_budget_fit(oos_prompt_budget, arm=cfg["mode"]' not in src              # the container's fit, before
    assert 'C.prompt_budget_fit(budget, arm=str(cfg["mode"])' not in src
    # the producer's argv on a llama IFEval configuration names the no-switch arm; the script's branch admits it
    (ll,) = _plan(L, LLAMA, cell="constv2-mix-full__lora_r4_ce")
    largv = L.ifeval_prompt_budget_argv("/tree", L.row_of(ll), ll, "/o.json", tree_sha="d" * 40)
    assert largv[largv.index("--arm") + 1] == C.THINKING_NOT_AVAILABLE and largv[largv.index("--model-key") + 1] == "llama8b"
    (gg,) = _plan(L, GEMMA, cell="prompted__safety_refusal")
    gargv = L.ifeval_prompt_budget_argv("/tree", L.row_of(gg), gg, "/o.json", tree_sha="d" * 40)
    assert gargv[gargv.index("--arm") + 1] == C.THINKING_OFF
