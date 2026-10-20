"""Launchers USE the per-model budget (the user's instruction was to make the
per-model budget for the evals, not a table nobody reads): every launcher that
knows its model passes ``model_key=`` into ``resolve_geometry`` so S comes from
``ModelBudget.seqs_at`` (the hybrid per-slot cap applied), the budget guard is
held to the MODEL's budget, and ``seqs_source="model_budget"`` lands in the
startup log and the provenance / endpoint record. Without a key the Llama
default stands (and the standup says so). Petri surfaces are untouched.

Also the identifier scrub of the generated block: every pasted ``source=``
carries the documented ``<SHARED>`` placeholder, never a username or an
absolute cluster path.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import geometry as sg

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _modal_stub():
    from test_grokmt_served_resume import _stub_modal
    sys.modules.setdefault("modal", _stub_modal())


# --------------------------------------------------------------------------- #
# model_key_for                                                                #
# --------------------------------------------------------------------------- #


def test_model_key_for_maps_every_pinned_id_and_warns_on_unknown(caplog):
    assert set(sg.MODEL_HF_IDS) == set(sg.MODEL_KEYS)
    sw = _load("_sweep_for_hf_ids", "scripts/served_decode_sweep.py")
    assert {k: ms.hf for k, ms in sw.MODELS.items()} == sg.MODEL_HF_IDS   # one table, two homes
    for key, hf in sg.MODEL_HF_IDS.items():
        assert sg.model_key_for(hf) == key
        assert sg.model_key_for(hf.lower()) == key
    # a staged snapshot directory (the HF-cache mangled name as its last component)
    assert sg.model_key_for("/some/staging/models/google--gemma-4-E2B-it") == "gemma4e2b"
    assert sg.model_key_for("/some/staging/models/google--gemma-4-E2B-it/") == "gemma4e2b"
    with caplog.at_level(logging.WARNING, logger="sisteer.capeval.serving.geometry"):
        assert sg.model_key_for("someone/other-model") is None
        assert sg.model_key_for("") is None and sg.model_key_for(None) is None
    warn = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn) == 1 and "someone/other-model" in warn[0] and "Llama-certified" in warn[0]


# --------------------------------------------------------------------------- #
# the standup: --model-key                                                     #
# --------------------------------------------------------------------------- #


def test_standup_sizes_from_the_model_key_and_notes_its_absence(caplog):
    up = _load("_standup_model_key", "scripts/capctrl_serving_up.py")
    # without a key: the per-eval Llama-sized S, exactly as before, plus the note
    with caplog.at_level(logging.WARNING, logger="sisteer.capeval.serving.geometry"):
        r = up.resolve_standup_geometry("capability", None, None)
    assert (r.max_model_len, r.max_num_seqs, r.seqs_source) == (7424, 52, "table")
    assert r.model_key is None and r.budget_tokens == 393_216
    assert not [x for x in caplog.records if x.levelno == logging.WARNING]   # None = silent default
    note = up.standup_model_key_note("capability", None)
    assert note.startswith("[standup] WARNING: no --model-key") and "393,216" in note
    assert "qwen35_4b" in note and "--model-key <key>" in note
    assert up.standup_model_key_note("capability", "llama8b") is None
    assert up.standup_model_key_note(None, None) is None                 # legacy path: no table
    # with a key: S from the model's budget, seqs_source model_budget, budget the model's
    r = up.resolve_standup_geometry("capability", None, None, model_key="llama8b")
    # llama8b is on the 2026-09-10 MEASURED row 475,136 = 29 x 16384 (the serving gate through the stacked-lens
    # standup; was the 2026-09-06 provisional ceiling 229,376 -> 30, and the swept 458,752 -> 61 before that)
    assert (r.max_num_seqs, r.seqs_source, r.budget_tokens) == (64, "model_budget", 475_136)
    assert r.is_identity and "max_num_seqs=64 (model_budget)" in r.describe()
    assert "model llama8b: measured]" in r.describe()
    assert "PROVISIONAL" not in r.describe()
    # a model still on a 2026-09-06 ceiling resolves from it and says so
    r = up.resolve_standup_geometry("capability", None, None, model_key="qwen35_9b")
    assert (r.max_num_seqs, r.budget_tokens) == (26, 196_608)                        # 196,608 // 7424
    assert "model qwen35_9b: provisional-conservative — PROVISIONAL" in r.describe()
    r = up.resolve_standup_geometry("capability", None, None, model_key="glm46vflash")
    assert r.max_num_seqs == 105 and r.budget_tokens == 786_432                     # dense: uncapped
    # hybrids: the per-slot cap — live on the two hybrids whose budgets are measured
    # (qwen35_4b's halved shadow was lifted 2026-09-11); on the provisional qwen35_9b
    # row the quotient binds first
    r = up.resolve_standup_geometry("capability:short", None, None, model_key="nemotron3nano4b")
    assert r.max_num_seqs == 256 and "hybrid per-slot cap: S 256 not 512" in r.describe()
    r = up.resolve_standup_geometry("capability:short", None, None, model_key="qwen35_4b")
    assert r.max_num_seqs == 256 and "hybrid per-slot cap: S 256 not 483" in r.describe()   # 741,888 // 1536 = 483, capped
    r = up.resolve_standup_geometry("capability:short", None, None, model_key="qwen35_9b")
    assert r.max_num_seqs == 128                                              # 196,608 // 1536
    # the budget guard is the MODEL's: an explicit S over Llama's budget but under the
    # model's is fine; over the model's refuses, naming the model
    r = up.resolve_standup_geometry("capability", None, 100, model_key="glm46vflash",
                                    allow_over_budget=True)
    assert not r.over_budget                                   # 742,400 < 786,432
    with pytest.raises(sg.GeometryOverBudget) as ei:
        up.resolve_standup_geometry("capability", None, 110, model_key="glm46vflash")
    assert "measured glm46vflash" in str(ei.value) and "786,432" in str(ei.value)
    # an unknown key refuses (never a silent default)
    with pytest.raises(sg.UnknownModelKey):
        up.resolve_standup_geometry("capability", None, None, model_key="glm46v_flash")
    # the CLI exposes it and main routes the key + note
    src = (REPO / "scripts/capctrl_serving_up.py").read_text()
    assert 'ap.add_argument("--model-key", default=None' in src
    assert "model_key=args.model_key" in src and "standup_model_key_note(args.eval, args.model_key)" in src
    # provenance: endpoint.json carries geometry.as_dict() -> model_key / budget / seqs_source
    # (plus the kv_shrink record beside those keys when the standup re-launched at the
    # container's KV-derived S; the served geometry's as_dict is still the block's base)
    assert '"geometry": {**self.geometry.as_dict(),' in src and '{"kv_shrink": self.kv_shrink}' in src
    d = r.as_dict()
    assert {"model_key", "budget", "budget_source", "seqs_source"} <= set(d)


#: every standup CLI that sizes its launch from the per-model budget: each passes
#: ``model_key=`` into ``resolve_standup_geometry`` from a MANDATORY ``--model-key``
#: and leaves the window/width defaults to the table (no literal window)
STANDUPS_WITH_MODEL_KEY = ("scripts/sdfbase/mt_serving_up.py",
                           "scripts/sdfbase/lora_serving_up.py")


@pytest.mark.parametrize("rel", STANDUPS_WITH_MODEL_KEY)
def test_the_sdfbase_standups_size_from_a_mandatory_model_key(rel):
    src = (REPO / rel).read_text()
    assert 'ap.add_argument("--model-key", required=True' in src, rel
    assert "resolve_standup_geometry(" in src, rel
    assert re.search(r"model_key=(args\.)?model_key", src), rel
    assert 'ap.add_argument("--max-model-len", type=int, default=None)' in src, rel
    assert 'ap.add_argument("--max-num-seqs", type=int, default=None)' in src, rel
    assert re.search(r"--max-model-len.*default=\d", src) is None, rel      # never a literal window


def test_the_lora_native_standup_resolves_the_certification_geometry_from_the_model_key():
    """The lora-native standup (native-multilora-serving §1.5) sizes the
    certification eval from the target's budget, and its override path
    (``--geometry-override-reason``) is the one non-identity door."""
    lsu = _load("_lora_serving_up_model_key", "scripts/sdfbase/lora_serving_up.py")
    assert lsu.EVAL_BY_MODE["static"] == "families_served"
    g = lsu._resolve_geometry("families_served:short", None, None, model_key="gemma4e4b",
                              allow_over_budget=False, override_reason=None)
    assert (g.max_model_len, g.seqs_source, g.model_key) == (4864, "model_budget", "gemma4e4b")
    assert g.is_identity and g.max_num_seqs == sg.resolve_geometry(
        "families_served:short", model_key="gemma4e4b").max_num_seqs
    half = g.max_num_seqs // 2
    with pytest.raises(lsu.StandupRefused, match="geometry"):
        lsu._resolve_geometry("families_served:short", None, half, model_key="gemma4e4b",
                              allow_over_budget=False, override_reason=None)
    low = lsu._resolve_geometry("families_served:short", None, half, model_key="gemma4e4b",
                                allow_over_budget=False,
                                override_reason="mamba_cache_blocks_under_graphs")
    assert low.max_num_seqs == half and low.seqs_source == "override" and not low.over_budget
    with pytest.raises(lsu.StandupRefused, match="not a canonical model key|refusing --model-key"):
        lsu._resolve_geometry("families_served:short", None, None, model_key="gemma4-e4b",
                              allow_over_budget=False, override_reason=None)


# --------------------------------------------------------------------------- #
# capctrl_served: the row's model                                              #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def capctrl():
    _modal_stub()
    return _load("_capctrl_served_model_budget", "scripts/modal/capctrl_served.py")


def test_capctrl_cells_size_from_their_model(capctrl):
    # llama8b: 475,136 // 7424 = 64 (the 2026-09-10 measured row; was 30 at the provisional 229,376, 61 at 458,752)
    expected = {"llama8b": 64, "gemma4e4b": 79, "glm46vflash": 105, "nemotron3nano4b": 105}
    for spec in capctrl.cell_specs():
        g = capctrl.cell_geometry(spec)
        assert (g.max_model_len, g.max_num_seqs) == (7424, expected[spec["model"]]), spec["model"]
        assert g.seqs_source == "model_budget" and g.model_key == spec["model"]
        assert g.budget_tokens == sg.BUDGET_BY_MODEL[spec["model"]].batched_token_budget
        assert g.is_identity and not g.over_budget
        assert g.as_dict()["seqs_source"] == "model_budget"
    # a bare registry row (no model) keeps the Llama default — the pre-budget behaviour
    g = capctrl.cell_geometry(capctrl.MODELS["llama8b"])
    assert (g.max_num_seqs, g.seqs_source, g.model_key) == (52, "table", None)
    # the hybrid per-slot cap: nemotron at the short window would be 786,432 // 1536 = 512
    g = capctrl.cell_geometry({"model": "nemotron3nano4b", "window": 1536})
    assert g.max_num_seqs == 256 and g.is_identity and not g.over_budget
    g = capctrl.cell_geometry({"model": "glm46vflash", "window": 1536})
    assert g.max_num_seqs == 512                                              # dense: uncapped
    # the guard is the model's budget
    g = capctrl.cell_geometry({"model": "glm46vflash", "seqs": 100, "allow_over_budget": True})
    assert not g.over_budget                                                   # 742,400 < 786,432
    with pytest.raises(sg.GeometryOverBudget):
        capctrl.cell_geometry({"model": "glm46vflash", "seqs": 110})           # 816,640 > 786,432
    assert not capctrl.cell_geometry({"model": "llama8b", "seqs": 64}).over_budget   # 475,136 == the measured budget
    with pytest.raises(sg.GeometryOverBudget):
        capctrl.cell_geometry({"model": "llama8b", "seqs": 65})                # 482,560 > 475,136
    src = (REPO / "scripts/modal/capctrl_served.py").read_text()
    assert 'model_key=spec.get("model")' in src
    assert 'cell_geometry({"model": model, **MODELS[model]}, tree)' in src
    assert 'cell_geometry({"model": "llama8b", **MODELS["llama8b"]}' in src
    assert 'rec["geometry"] = geom.as_dict()' in src


# --------------------------------------------------------------------------- #
# grokmt_served: the target key                                                #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def grokmt():
    _modal_stub()
    return _load("_grokmt_served_model_budget", "scripts/modal/grokmt_served.py")


def test_grokmt_target_sizes_from_its_model_key(grokmt):
    import inspect
    assert grokmt.TARGET_KEY == "gemma4e4b" == sg.model_key_for(
        __import__("yaml").safe_load((REPO / grokmt.CFG_REL).read_text())["model"])
    g = grokmt.resolve_target_geometry(None)
    assert (g.max_model_len, g.max_num_seqs, g.seqs_source) == (8704, 67, "model_budget")
    assert "max_num_seqs=67 (model_budget)" in g.describe() and "model gemma4e4b: measured" in g.describe()
    g = grokmt.resolve_target_geometry(None, target_key="llama3b")
    assert g.max_num_seqs == 37 and g.budget_tokens == 327_680               # 327,680 // 8704
    # a hybrid target at the short window (the cap is above the provisional quotient)
    g = grokmt.resolve_target_geometry(None, target_window=1536, target_key="qwen35_4b")
    assert g.max_num_seqs == 256                                             # 741,888 // 1536 = 483, hybrid cap 256
    # overrides still win; the guard is the target model's budget
    g = grokmt.resolve_target_geometry(None, target_seqs=24)
    assert g.max_num_seqs == 24 and g.seqs_source == "override"
    with pytest.raises(sg.GeometryOverBudget):
        grokmt.resolve_target_geometry(None, target_window=16384, target_seqs=60)   # > 589,824
    assert grokmt.resolve_target_geometry(None, target_window=16384, target_seqs=60,
                                          allow_over_budget=True).over_budget
    # an empty key = no model named: the Llama default, as before the budgets
    g = grokmt.resolve_target_geometry(None, target_key="")
    assert (g.max_num_seqs, g.seqs_source, g.model_key) == (45, "table", None)
    with pytest.raises(sg.UnknownModelKey):
        grokmt.resolve_target_geometry(None, target_key="gemma4-e4b")
    for fn in (grokmt.run_smoke, grokmt.main):
        assert inspect.signature(fn).parameters["target_key"].default == "gemma4e4b"
    src = (REPO / "scripts/modal/grokmt_served.py").read_text()
    assert src.count("target_key=target_key") >= 3                           # main -> run_smoke -> resolve
    assert '"target_geometry": geom.as_dict()' in src


# --------------------------------------------------------------------------- #
# the families waves: WAVE_MODEL / WAVE_MODEL_KEY                               #
# --------------------------------------------------------------------------- #


def test_families_waves_size_from_their_model(monkeypatch, caplog):
    for k in ("WAVE_MAX_MODEL_LEN", "WAVE_MAX_NUM_SEQS", "WAVE_MAX_CONCURRENCY",
              "WAVE_ALLOW_OVER_BUDGET", "WAVE_MODEL", "WAVE_MODEL_KEY"):
        monkeypatch.delenv(k, raising=False)
    # the served wave on a hybrid model id: qwen35_9b's budget, 196,608 // 11520 = 17
    # (the 2026-09-06 provisional ceiling; the swept 589,824 gave 51)
    monkeypatch.setenv("WAVE_MODEL", "Qwen/Qwen3.5-9B")
    fam = _load("_fam_served_qwen", "scripts/petri_families_served_integration.py")
    assert fam.MODEL_KEY == "qwen35_9b" and fam.MAX_NUM_SEQS == 17 and fam.MAX_CONCURRENCY == 17
    r = fam.resolved_geometry()
    assert (r.max_model_len, r.max_num_seqs, r.seqs_source) == (11520, 17, "model_budget")
    assert r.budget_tokens == 196_608 and r.is_identity
    assert r.budget_source == sg.PROVISIONAL_BUDGET_SOURCE
    # ... and at a short window override the quotient binds below the per-slot cap
    monkeypatch.setenv("WAVE_MAX_MODEL_LEN", "1536")
    fam_s = _load("_fam_served_qwen_short", "scripts/petri_families_served_integration.py")
    r = fam_s.resolved_geometry()
    assert r.max_num_seqs == 128 and r.is_identity and r.seqs_source == "model_budget"
    monkeypatch.delenv("WAVE_MAX_MODEL_LEN")
    # an explicit key wins over the id
    monkeypatch.setenv("WAVE_MODEL_KEY", "llama3b")
    fam_k = _load("_fam_served_key", "scripts/petri_families_served_integration.py")
    assert fam_k.MODEL_KEY == "llama3b" and fam_k.MAX_NUM_SEQS == 28          # 327,680 // 11520
    monkeypatch.delenv("WAVE_MODEL_KEY")
    # an unknown id: no key, the Llama default (34) and a warning
    monkeypatch.setenv("WAVE_MODEL", "someone/other-model")
    with caplog.at_level(logging.WARNING, logger="sisteer.capeval.serving.geometry"):
        fam_u = _load("_fam_served_unknown", "scripts/petri_families_served_integration.py")
    assert fam_u.MODEL_KEY is None and fam_u.MAX_NUM_SEQS == 34
    assert fam_u.resolved_geometry().seqs_source == "table"
    assert any("someone/other-model" in x.getMessage() for x in caplog.records)
    # the guard is the wave model's budget: 16384 x 20 = 327,680 > qwen35_9b's 196,608
    # refuses, but is exactly llama3b's provisional ceiling and passes there
    monkeypatch.setenv("WAVE_MODEL", "Qwen/Qwen3.5-9B")
    monkeypatch.setenv("WAVE_MAX_MODEL_LEN", "16384")
    monkeypatch.setenv("WAVE_MAX_NUM_SEQS", "20")
    fam_o = _load("_fam_served_over", "scripts/petri_families_served_integration.py")
    with pytest.raises(sg.GeometryOverBudget):
        fam_o.resolved_geometry()
    monkeypatch.setenv("WAVE_MODEL_KEY", "llama3b")
    fam_ok = _load("_fam_served_ok", "scripts/petri_families_served_integration.py")
    r = fam_ok.resolved_geometry()
    assert not r.over_budget and r.is_identity and r.max_num_seqs == 20
    # the prompted wave: the pinned llama id -> llama8b, 475,136 // 7424 = 64
    for k in ("WAVE_MAX_MODEL_LEN", "WAVE_MAX_NUM_SEQS", "WAVE_MODEL", "WAVE_MODEL_KEY"):
        monkeypatch.delenv(k, raising=False)
    pr = _load("_fam_prompted_model", "scripts/petri_families_prompted.py")
    assert pr.MODEL_KEY == "llama8b" and pr.MAX_NUM_SEQS == 64
    assert pr.resolved_geometry().budget_source == sg.MEASURED_BUDGET_SOURCE
    assert pr.resolved_geometry().seqs_source == "model_budget"
    monkeypatch.setenv("WAVE_MODEL_KEY", "qwen35_4b")
    pr_q = _load("_fam_prompted_qwen", "scripts/petri_families_prompted.py")
    assert pr_q.MODEL_KEY == "qwen35_4b" and pr_q.MAX_NUM_SEQS == 99           # 741,888 // 7424 (the 2026-09-13 lens-stack row at 17,664)
    # both records carry the key next to the geometry
    for rel in ("scripts/petri_families_served_integration.py", "scripts/petri_families_prompted.py"):
        src = (REPO / rel).read_text()
        assert '"model_key": MODEL_KEY' in src and "model_key=MODEL_KEY" in src


# --------------------------------------------------------------------------- #
# Petri surfaces untouched; the served capability client unchanged             #
# --------------------------------------------------------------------------- #


def test_petri_surfaces_are_untouched_by_the_model_budget():
    from sisteer.capeval.serving import vllm_server as vs
    assert vs.DEFAULT_MAX_NUM_SEQS == 24
    assert sg.TABLE["petri"].max_num_seqs == 24 and sg.TABLE["petri"].certified
    for rel in ("scripts/modal/petri_agentloop_control.py", "scripts/petri_proxy.py"):
        p = REPO / rel
        if p.is_file():
            assert "model_key=" not in p.read_text(), rel
    # the served capability CLIENT keeps the table's fan-out default; capctrl passes
    # the launched S explicitly (--concurrency str(geom.max_num_seqs))
    import served_capability_run as srun
    assert srun.CAPABILITY_GEOMETRY is sg.geometry_for("capability")


# --------------------------------------------------------------------------- #
# the generated block carries no identifiers                                   #
# --------------------------------------------------------------------------- #


def test_measured_block_has_no_username_or_absolute_path():
    src = (REPO / "src/sisteer/capeval/serving/geometry.py").read_text()
    blk = src[src.index("# MEASURED BY scripts/served_decode_sweep.py --phase budget  (BEGIN)"):
              src.index("# MEASURED BY scripts/served_decode_sweep.py --phase budget  (END)")]
    assert not re.search(r"/home/|/workspace", blk)
    user = os.environ.get("USER", "")
    if len(user) > 2:
        assert user not in blk
    sources = re.findall(r"source='([^']*)'", blk)
    assert len(sources) == 8
    for s in sources:
        # a swept row cites the sweep's budget.json; the llama8b row (the 2026-09-10 serving gate through the
        # stacked-lens standup) cites its ladder.json and sibling ladders; the qwen35_4b row (the 2026-09-13 budget walk
        # AT the MATH-500 window, geom_budget_window.sbatch) its per-WINDOW sweep root geom-sweep-L17664 — all under the
        # scrubbed shared root
        assert re.fullmatch(r"<SHARED>/exp/(geom-sweep(-L\d+)?/\w+/budget\.json|sdfbase_full22/serving_gate/[\w/.-]+) "
                            r"\(node [\w-]+(; [^']*)?\)", s), s
    assert "<SHARED> in a source = the cluster's per-user shared root" in blk
    # the same holds for every entry the module actually holds: a SWEPT basis cites
    # the scrubbed cluster path, the llama8b SERVING-GATE row (_MEASURED_BUDGETS, 2026-09-10)
    # the scrubbed gate root + its probe job, a PROVISIONAL one the smoke and no path at all
    for b in sg.BUDGET_BY_MODEL.values():
        for field in (b.basis.source, b.basis.method, b.basis.note):
            assert not re.search(r"/home/|/workspace", field), b.model_key
            assert not (len(user) > 2 and user in field), b.model_key
        if b.provisional:
            assert "2026-09-06" in b.basis.source, b.model_key
        elif b.model_key == "llama8b":
            # the 2026-09-10 serving-gate row of _MEASURED_BUDGETS (measured, certified=False): its
            # ladder.json + probe job under the scrubbed shared root
            assert b.basis.source.startswith("<SHARED>/exp/") and b.basis.probe_job_id in b.basis.source
            assert b.source == sg.MEASURED_BUDGET_SOURCE and not b.provisional and not b.certified
        elif b.model_key == "qwen35_4b":
            # the 2026-09-13 budget walk AT the MATH-500 window (17,664) on the lens stack: the per-window sweep root's
            # budget.json + its probe job + the job log, under the scrubbed shared root
            assert b.basis.source.startswith("<SHARED>/exp/geom-sweep-L17664/qwen35_4b/budget.json")
            assert b.basis.probe_job_id in b.basis.source and b.basis.max_model_len == 17664
            assert b.source == sg.MEASURED_BUDGET_SOURCE and not b.provisional and not b.certified
        else:   # a swept row: the scrubbed sweep root
            assert b.basis.source.startswith("<SHARED>/exp/geom-sweep/"), b.model_key
    # the scrub itself: sweep + analyzer agree, idempotent, non-cluster paths untouched
    sw = _load("_sweep_scrub", "scripts/served_decode_sweep.py")
    an = _load("_analyze_scrub", "scripts/geom_sweep_analyze.py")
    for raw, want in (("/workspace-vast/someone/exp/geom-sweep/x/budget.json (node gpu-0)",
                       "<SHARED>/exp/geom-sweep/x/budget.json (node gpu-0)"),
                      ("/workspace/someone/exp/geom-sweep/x/budget.json (node gpu-1)",
                       "<SHARED>/exp/geom-sweep/x/budget.json (node gpu-1)"),
                      ("<SHARED>/exp/geom-sweep/x/budget.json", "<SHARED>/exp/geom-sweep/x/budget.json"),
                      ("/tmp/r/budget.json", "/tmp/r/budget.json")):
        assert sw.shared_placeholder(raw) == want and an.scrub_source(raw) == want
    assert an.scrub_source(None) is None
    rec = sw.paste_record(model_key="qwen35_4b", max_model_len=8192, gpu_mem_util=0.9,
                          job_id="1", measured_on="d", node="gpu-2",
                          results_dir="/workspace-vast/someone/exp/geom-sweep/qwen35_4b",
                          bracket={"largest_passing": {"product": 8192 * 88, "max_num_seqs": 88,
                                                       "kv_tokens": 1}, "first_refused": None,
                                   "n_probes": 1, "wall_reached": False})
    assert rec["basis"]["source"] == "<SHARED>/exp/geom-sweep/qwen35_4b/budget.json (node gpu-2)"
