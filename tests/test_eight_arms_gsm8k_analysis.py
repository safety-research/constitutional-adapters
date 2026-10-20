"""The EIGHT-ARM extension of the GSM8K paired-200 analysis over the SYNTHETIC stores of ``tests/gsm8k_synthetic_store.py``:
multi-tree store admission (``--gen-root PATH=PIN`` repeatable + ``--also-sha``), the three cross-tree compatibility relations
and the per-contrast record, a corrupted rung identity rejected by name, an unadmitted tree and a duplicate logical leg refused,
the overlap gate held to the primary pin, the DERIVED expected leg set, the C4-style fixture (published + extension rows +
deferred steering stubs) under ``--waves``, the pooled-by-arm estimands over the main view, the presentation selectors and the
publication gate, the eight-arm figures, the Llama models' ``--peer-report``, the trigger provenance and the identity keys, and
the ``main()`` boundary. ``tests/test_tinycap_gsm8k_analysis.py`` is never grown; every fixture here is synthetic (placeholder
shas, invented ids), and the frozen 1 % reproduction is held by ``tests/test_eight_arms_frozen_reproduction.py``.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import gsm8k_synthetic_store as G
import numpy as np
import pytest

from sisteer.capeval.tiers import cap_ladder as CL
from sisteer.capeval.tiers import inspect_rung as IR
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("tinycap_gsm8k_paired200", REPO / "scripts" / "tinycap_gsm8k_paired200.py")
O = _load("gsm8k_overlap_check", REPO / "scripts" / "gsm8k_overlap_check.py")
V = A.V
M = A.M
#: THE DECLARED PANEL ROSTER's kinds — what a published design lays out, NEVER `M.ARM_KINDS` (the live registry). Registering a
#: further arm (arditi-allpos-cap's `arditi_allpos`, which has its own matrix and its own export) must not widen a panel whose
#: matrix does not carry it, so these assertions are held to the roster; an analysis whose rows DO carry a further arm covers it
#: too (M.design_arm_roster), and a test of such a matrix must say so explicitly rather than reach for the registry.
PANEL_KINDS: list[str] = [a.kind for a in M.PUBLISHED_ARMS]
PG = A.PG

B_FAST = 400
GEMMA, QWEN, NEMO, GLM, LLAMA = "gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash", "llama8b"
NONE, PROMPTED, CONST_PROMPT = G.CELL_NONE, G.CELL_PROMPTED, G.CELL_PROMPTED_CONST
CONST_PSR, CONST_LORA = "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce"
CTL_PSR, CTL_LORA = "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce"
SI12_PSR, SI12_LORA = "safety_refusal__psr_kl_all", "safety_refusal__lora_r4"
DIST_PSR = "constv2-mix-full__psr_ce_all_minus_ctl_distilled"
MINUS_CTL = "constv2-mix-full__lora_r4_ce_minus_ctl"
#: the paper view's five appendix arms: the four PSR cells + the safety LoRA
APPENDIX_ARMS = (CONST_PSR, CTL_PSR, SI12_PSR, DIST_PSR, SI12_LORA)
#: three synthetic producer trees: the published pin's, the extension wave's (declared), a third never admitted
T1, T2, T3 = G.FULL_SHA, "8" * 40, "9" * 40
PIN = G.SHA7
WAVES = ["published", G.WAVE_EXT]
ROWS = G.matrix_rows()
EXT_ROWS = G.eight_arm_matrix()
ESC_NONE = (QWEN, "main", NONE)
ESC_EXT = (QWEN, "main", CONST_PROMPT)
VIEWS_RETIRED = (GEMMA, f"{GEMMA}_it", QWEN, f"{QWEN}_it", NEMO, GLM)


def row(key, view, cell, rows=EXT_ROWS):
    return G.row_for(rows, key, view, cell)


def leg_of(report, key, view, cell, cap=G.CAP_BASE):
    return report["legs"][V.leg_key(key, view, cell, cap)]


def contrast_of(report, key, view, cell):
    return next(c for c in report["contrasts"] if c["model_key"] == key and c["view"] == view and c["cell"] == cell)


def small_matrix(*triples, rows=EXT_ROWS):
    keep = set(triples)
    return [r for r in rows if (r["model_key"], r["view"], r["cell"]) in keep]


def build_overlap(root, *, models=G.MODEL_KEYS, store_pin=PIN, tree_sha=T1):
    retired = root / "retired"
    for view in VIEWS_RETIRED:
        G.build_retired_store(retired, view, correct=set(range(180)), extraction_failed={3, 7})
    for key in models:
        G.build_overlap_stock_store(root, key, correct=set(range(178)) | {181, 182}, sha=store_pin, tree_sha=tree_sha)
    return O.run(retired_root=retired, stock_root=root / "overlap", models=list(models), canonical_ids=G.IDS, ids_path=None,
                 pairing_rows=G.pairing_rows(), pairing_source="synthetic rows", repo_root=REPO, store_pin=store_pin, n_resamples=B_FAST, seed=0,
                 ci=0.95, eval_logs=False)


def fake_git_for(known, diffs):
    """The shared stand-in for ``M._git_output`` (``math500_synthetic_store.fake_git_for``), bound to this module's ``M``."""
    return G.S.fake_git_for(known, diffs, diff_args=M.GIT_DIFF_NAMES_ARGS)


@pytest.fixture(scope="module")
def fake_git():
    real = M._git_output
    M._git_output = fake_git_for([T1, T2, T3], {(T1, T2): ["scripts/tinycap_extension.py"], (T1, T3): ["scripts/tinycap_extension.py"]})
    yield
    M._git_output = real


TREE_SET = {"also": [T2], "allow": ["scripts/*"], "why": "synthetic: the extension wave's legs come from its own tree"}


def analyze(gen_root, sha=PIN, *, matrix=None, **kw):
    kw.setdefault("matrix", matrix if matrix is not None else EXT_ROWS)
    kw.setdefault("canonical_ids", G.IDS)
    kw.setdefault("sampling_rows", G.SAMPLING_ROWS)
    kw.setdefault("render_facts", G.RENDER_FACTS)
    kw.setdefault("pins", G.pins_doc(ROWS))
    kw.setdefault("n_resamples", B_FAST)
    kw.setdefault("read_eval_logs", False)
    kw.setdefault("analysis_sha", G.FULL_SHA)
    return A.analyze(gen_root, sha, **kw)


def roots_of(store, b=None):
    return [f"{store.a}={PIN}", f"{b if b is not None else store.b}={T2}"]


def analyze_c4(store, *, b=None, waves=WAVES, **kw):
    kw.setdefault("overlap", store.overlap)
    kw.setdefault("tree_set", TREE_SET)
    return analyze(roots_of(store, b), waves=waves, **kw)


def build_cross_tree(root, *, published_rows, ext_rows, esc_none=True, esc_ext=True, ext_tree=T2, ext_common=None, pub_overrides=None):
    """Two wave-isolated pulls: the published rows under ``a`` (tree T1, the pin), the extension rows under ``b`` (tree
    ``ext_tree``); qwen's main ``none`` and its extension constitution cell escalated with UNEQUAL at-cap sets.
    ``pub_overrides`` extends one published leg's ``build_store`` kwargs (``{(key, view, cell): {...}}``)."""
    a, b = root / "pull_published", root / "pull_ext"
    esc = {}
    G.build_rows(a, published_rows, skip=[ESC_NONE] if esc_none else [], overrides=pub_overrides, producer_fields=True)
    if esc_none and any((r["model_key"], r["view"], r["cell"]) == ESC_NONE for r in published_rows):
        esc["none"] = G.build_escalated_cell(a, row(*ESC_NONE), at_cap=range(190, 199), loops={190, 191}, capped_scored_correct={198},
                                             rung_still_at_cap={192, 193}, rung_boundary={194}, producer_fields=True)
        assert esc["none"]["record"]["exit"] == CL.ESCALATE and esc["none"]["rung_cell_dir"] is not None
    common = {"producer_fields": True, "tree_sha": ext_tree, **(ext_common or {})}
    G.build_rows(b, ext_rows, skip=[ESC_EXT] if esc_ext else [], **common)
    if esc_ext and any((r["model_key"], r["view"], r["cell"]) == ESC_EXT for r in ext_rows):
        esc["ext"] = G.build_escalated_cell(b, row(*ESC_EXT), at_cap=range(180, 187), loops={180}, rung_still_at_cap={181}, **common)
        assert esc["ext"]["record"]["exit"] == CL.ESCALATE and esc["ext"]["rung_cell_dir"] is not None
    return SimpleNamespace(root=root, a=a, b=b, esc=esc)


@pytest.fixture(scope="module")
def c4_store(tmp_path_factory, fake_git):
    """The C4-style fixture: the 42 published rows (38 built at T1 under the primary pin's root, 4 held), the five extension
    rows (built at T2 under a wave-isolated root), the six deferred steering stubs (never stores), two escalated cells with
    unequal rungs, the overlap record at the primary pin."""
    root = tmp_path_factory.mktemp("g8k_c4")
    published = [r for r in EXT_ROWS if r.get("wave") is None]
    ext = [r for r in EXT_ROWS if r.get("wave") == G.WAVE_EXT]
    store = build_cross_tree(root, published_rows=published, ext_rows=ext)
    store.overlap = build_overlap(store.a)
    return store


@pytest.fixture(scope="module")
def c4_report(c4_store):
    return analyze_c4(c4_store)



def test_a_natively_served_leg_attests_through_its_own_record_and_the_contrast_no_longer_refuses_as_uncomparable(tmp_path, fake_git):
    """launch-attestation-native, end to end on this port (the real case: the primary cluster's three Llama PSR GSM8K cards,
    whose standup dirs carry `serving_attestation.json` + `vllm_server.log` + `serving_config.json` and NO
    `steering_attestation.json`, all read `launch_status: absent` — "no workers' attestation" — and every contrast refused
    "the legs' launch settings cannot be compared"). A natively served fitted leg built as the native driver files it
    (`serving_engine=lora_native`: the engine stamp on its route, the STRICT serving attestation and the native echo beside
    its job) now ATTESTS through the shared helper — `engine: lora_native`, `record_file: serving_attestation.json`, bound by
    job id and port — beside a lens `none` anchor that attests through the lens record; under the declared tree set the
    contrast no longer says the settings cannot be compared. Both launches having been READ, the comparison then runs and
    names what it finds: the two engines' echoes differ by construction (the lens launch echoes `enforce_eager: true`, the
    native launch — graphs on — omits it), so the contrast refuses on the FIRST differing flag by name with a verdict over
    settings that WERE read (`server_settings_equal: false`, never None). Known-bad: the same native leg over a standup dir
    carrying only the LENS record refuses as uncomparable, naming the native record and the remedy."""
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_LORA), (QWEN, "main", CONST_PROMPT))
    pub, ext = [r for r in rows if r.get("wave") is None], [r for r in rows if r.get("wave")]
    assert (QWEN, "main", CONST_LORA) in {(r["model_key"], r["view"], r["cell"]) for r in pub}     # the native leg shares the anchor's tree
    store = build_cross_tree(tmp_path / "native", published_rows=pub, ext_rows=ext, esc_none=False, esc_ext=False,
                             pub_overrides={(QWEN, "main", CONST_LORA): {"serving_engine": C.SERVING_ENGINE_LORA_NATIVE}})
    r = analyze(roots_of(store), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    leg, anchor = leg_of(r, QWEN, "main", CONST_LORA), leg_of(r, QWEN, "main", NONE)
    assert leg["refused"] is None and leg["missing"] is None, leg.get("refused")
    _cd, native_manifest, _why = V.read_store(Path(leg["gen_root"]) / row(QWEN, "main", CONST_LORA)["out_root"])
    assert native_manifest[C.MANIFEST_KEY_ROUTE][C.SERVED_ENGINE_FIELD] == C.SERVING_ENGINE_LORA_NATIVE and M.launch_engine_of(native_manifest) == (C.SERVING_ENGINE_LORA_NATIVE, None)
    launch = leg["launch"]
    assert launch["status"] == M.LAUNCH_ATTESTED and launch["problems"] == [], launch["problems"]
    assert launch["engine"] == C.SERVING_ENGINE_LORA_NATIVE and launch["record_file"] == M.LAUNCH_NATIVE_ATTESTATION_FILE == "serving_attestation.json"
    assert launch["job_id"] == G.standup_job_id(row(QWEN, "main", CONST_LORA), G.CAP_BASE) and launch["geometry_reconciled"] is True
    d = Path(launch["standup_dir"])
    assert d.name == "gsm8k" and (d / "serving_attestation.json").is_file() and (d / M.LAUNCH_LOG_FILE).is_file() and not (d / "steering_attestation.json").is_file()
    assert json.loads(launch["settings"]["enable_lora"]) is True and "enforce_eager" not in launch["settings"]
    # O106: the cell-set plumbing is recorded under `excluded` with the ruling's reason, the kernel configuration compared
    assert [m["LoRAModulePath"]["name"] for m in json.loads(launch["excluded"]["lora_modules"]["value"])] == [C.TWIN_CELL_ID_LORA, row(QWEN, "main", CONST_LORA)["harness_cell"]]
    assert all("O106" in launch["excluded"][k]["why"] for k in ("lora_modules", "max_loras", "max_cpu_loras")) and "max_lora_rank" in launch["settings"]
    assert anchor["launch"]["status"] == M.LAUNCH_ATTESTED and anchor["launch"]["engine"] == C.SERVING_ENGINE_LENS
    assert anchor["launch"]["record_file"] == M.LAUNCH_ATTESTATION_FILE == "steering_attestation.json" and json.loads(anchor["launch"]["settings"]["enforce_eager"]) is True
    c = contrast_of(r, QWEN, "main", CONST_LORA)
    assert "launch settings cannot be compared" not in (c["reason"] or "") and "no launch attestation" not in (c["reason"] or "")
    assert c["compatibility"]["launch_attested"] == {k: True for k in c["compatibility"]["launch_attested"]} and c["compatibility"]["server_settings_note"] is None
    # both READ: the comparison ran and refuses with a VERDICT, not an abstention — O106: the two legs attest DIFFERENT
    # engines, so the contrast is not admissible whatever the flags read; the refusal names the engine pair, the reason, the
    # remedy and the ruling, and lists EVERY differing flag with both values (the plumbing keys excluded, the kernel keys listed)
    none_key, lora_key = f"{QWEN}/main/{NONE}", f"{QWEN}/main/{CONST_LORA}"       # the contrast's labels are its CELL keys
    assert leg_of(r, QWEN, "main", NONE)["leg"].startswith(none_key) and leg["leg"].startswith(lora_key)
    head = f"cross-engine contrast: {none_key} (lens) vs {lora_key} (lora_native) is not admissible — {M.CROSS_ENGINE_WHY} — O106; differing launch flags ("
    assert not c["computable"] and c["reason"].startswith(head), c["reason"]
    assert "numerics=not_measured" in c["reason"] and "none__lora0" in c["reason"] and "launch settings cannot be compared" not in c["reason"]
    listed = c["reason"][len(head):].split("): ", 1)[1].split("; ")
    assert c["reason"][len(head):].startswith(f"{len(listed)}): ")
    assert {d.split(": ", 1)[0] for d in listed} == {"launch.enforce_eager", "launch.enable_lora", "launch.max_lora_rank", "launch.lora_target_modules"}, listed
    assert f"launch.enforce_eager: {{'{none_key}': 'true', '{lora_key}': None}}" in listed and f"launch.enable_lora: {{'{none_key}': None, '{lora_key}': 'true'}}" in listed
    compat = c["compatibility"]
    assert compat["server_settings_equal"] is False and compat["cross_engine"] is True and compat["cross_engine_refusal"] == c["reason"]
    assert compat["engine_by_leg"] == {none_key: "lens", lora_key: "lora_native"} and compat["server_settings_literal_difference"].startswith("launch.enforce_eager")
    assert contrast_of(r, QWEN, "main", CONST_PROMPT)["computable"]                  # the lens-vs-lens contrast beside it is untouched
    # known-bad (i): a LENS record ALSO beside the native leg's standup (a torn / mixed dir) — the leg itself stays usable (its
    # served-positions record is intact) but its launch refuses by name, so the contrast is uncomparable again, naming the
    # torn dir and the remedy; the verdict abstains (None), never a cross-engine ruling over settings that were not read
    (d / "steering_attestation.json").write_text(json.dumps({"slurm_job_id": launch["job_id"], "port": json.loads(launch["excluded"]["port"]["value"])}))
    r2 = analyze(roots_of(store), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    leg2, c2 = leg_of(r2, QWEN, "main", CONST_LORA), contrast_of(r2, QWEN, "main", CONST_LORA)
    assert leg2["refused"] is None and leg2["launch"]["status"] == M.LAUNCH_ABSENT and leg2["launch"]["engine"] == C.SERVING_ENGINE_LORA_NATIVE
    assert len(leg2["launch"]["problems"]) == 1 and "the lens engine's launch record beside a lora_native leg's standup" in leg2["launch"]["problems"][0]
    assert "torn / mixed" in leg2["launch"]["problems"][0]
    assert not c2["computable"] and "launch settings cannot be compared" in c2["reason"] and "steering_attestation.json" in c2["reason"] and "torn / mixed" in c2["reason"]
    assert c2["compatibility"]["server_settings_equal"] is None and sorted(c2["compatibility"]["launch_attested"].values()) == [False, True]
    assert c2["compatibility"]["cross_engine"] is None and c2["compatibility"]["cross_engine_refusal"] is None
    # the lens-vs-lens contrast beside it carries the same-engine record: equal, not cross-engine
    cp = contrast_of(r2, QWEN, "main", CONST_PROMPT)["compatibility"]
    assert cp["server_settings_equal"] is True and cp["cross_engine"] is False and set(cp["engine_by_leg"].values()) == {"lens"}
    # known-bad (ii): the native record REPLACED by the lens file (a mispull, or a lens file put where the native one should
    # be) — the native record is ALSO the served-positions fact of a natively served cell, so the leg refuses at collection
    # naming the missing file (never read as a lens leg), and the contrast falls with it
    (d / "steering_attestation.json").unlink()
    (d / "serving_attestation.json").rename(d / "steering_attestation.json")
    r3 = analyze(roots_of(store), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    leg3, c3 = leg_of(r3, QWEN, "main", CONST_LORA), contrast_of(r3, QWEN, "main", CONST_LORA)
    assert leg3["refused"] is not None and "lacks ['serving_attestation.json']" in leg3["refused"] and "natively served UNDECLARED" in leg3["refused"]
    assert not c3["computable"] and "lacks ['serving_attestation.json']" in c3["reason"]
    # ... and the launch reader on that manifest, called directly, names the missing native record and the torn lens file
    direct = M.launch_attestation(native_manifest, gen_root=Path(leg["gen_root"]), task_subdir=A.LAUNCH_TASK_SUBDIR)
    assert direct["status"] == M.LAUNCH_ABSENT and direct["engine"] == C.SERVING_ENGINE_LORA_NATIVE and direct["record_file"] == "serving_attestation.json"
    assert any("no serving attestation (serving_attestation.json, the lora_native engine's launch record)" in p and "pull the job's serving/" in p for p in direct["problems"])
    assert any("the lens engine's launch record beside a lora_native leg's standup" in p for p in direct["problems"]) and len(direct["problems"]) == 2

# --------------------------------------------------------------------------- #
# the matrix: statuses, waves, wave-isolated roots                             #
# --------------------------------------------------------------------------- #


def test_matrix_admits_waves_deferred_stubs_and_wave_isolated_roots_and_keeps_the_frozen_rows():
    mx = V.load_matrix(EXT_ROWS)
    assert mx["counts"]["rows"] == len(EXT_ROWS) and mx["counts"]["held"] == 4 and len(mx["deferred"]) == len(G.steer_rows(0))
    assert mx["counts"]["launched"] == 38 + len(G.ext_rows())
    # the published rows' roots are the frozen layout byte for byte; the extension rows keep their wave-isolated prefix
    for r in mx["rows"]:
        if r.get("wave") is None:
            assert r["out_root"] == f"controls/{r['tag']}/{r['cell_slug']}/thinking_card"
        elif r["wave"] == G.WAVE_EXT:
            assert r["out_root"] == f"controls_ext/{r['tag']}/{r['cell_slug']}/thinking_card" and r["rung_out_root"] == r["out_root"] + "__rung16384"
    # the required set is the evidence rows of the analysed waves — derived, never typed
    assert len(V.required_rows(mx["launch"], None)) == mx["counts"]["launched"]
    assert len(V.required_rows(mx["launch"], ["published"])) == 38
    assert {r["cell"] for r in V.required_rows(mx["launch"], [G.WAVE_EXT])} == {CONST_PROMPT, PROMPTED}
    assert V.required_rows(mx["launch"], [G.WAVE_STEER]) == []           # deferred rows are never evidence
    # a `complete` row is evidence; a deferred arditi stub is admitted. `arditi` IS a cell this tree's grammar admits
    # (defense_cells.SERVED_ONLY_CLASS_B_IDS — served from the arditi bank, no trainer here), so an EVIDENCE row for it
    # refuses only for the reason every fitted row refuses: no object pin on the row.
    rows = [dict(r) for r in EXT_ROWS]
    rows[0]["status"] = "complete"
    assert len(V.load_matrix(rows)["launch"]) == mx["counts"]["launched"]
    bad = [dict(r) for r in EXT_ROWS]
    stub = next(r for r in bad if r["cell"] == G.CELL_ARDITI)
    stub.update(status="launch", deferred_reason=None)
    with pytest.raises(A.AnalysisError, match="needs a 64-hex object_sha256"):
        V.load_matrix(bad)
    # a deferred row owes no pin: matrix_pins_problem skips it; an unknown status and a bad wave refuse
    assert V.matrix_pins_problem(mx["rows"], G.pins_doc(ROWS)) is None
    for edit, why in ((lambda r: r.update(status="paused"), "status"), (lambda r: r.update(wave=""), "wave"),
                      (lambda r: r.update(out_root="elsewhere/" + r["out_root"].split("/", 1)[1]), "layout")):
        rows = [dict(r) for r in EXT_ROWS]
        edit(next(r for r in rows if (r["model_key"], r["view"], r["cell"]) == (QWEN, "main", CONST_PROMPT)))
        with pytest.raises(A.AnalysisError, match=why):
            V.load_matrix(rows)


# --------------------------------------------------------------------------- #
# (1) the cross-tree merged contrast and its compatibility record             #
# --------------------------------------------------------------------------- #


def test_cross_tree_merged_contrast_computes_and_records_the_compatibility(c4_store, c4_report):
    rep = c4_report
    assert rep["unusable_legs"] == {} and rep["gates"]["failed"] == [], rep["gates"]["failed"]
    assert A.production_refusals(rep) == [] and rep["verdict"]["word"] == "filed"
    assert rep[A.TREE_SET_KEY]["also"] == [T2] and rep[A.TREE_SET_KEY]["problems"] == []
    assert [(Path(g["path"]), g["pin"], g["pinned"]) for g in rep["gen_roots"]] == [(c4_store.a, PIN, True), (c4_store.b, T2, True)]
    assert rep["provenance"]["gen_roots"] == rep["gen_roots"] and rep["provenance"]["waves"] == WAVES
    # the merged contrast spans two admitted trees: two bases (T1 / T2), two rungs each bound to its own source, UNEQUAL subsets
    c = contrast_of(rep, QWEN, "main", CONST_PROMPT)
    assert c["computable"] and c["escalated"] is True
    comp = c["compatibility"]
    assert comp["cross_tree"] is True and comp["declared"] is True and comp["projection_equal"] is True
    assert comp["server_settings_equal"] is True and comp["arm_contracts_valid"] is True and comp["rule"] == A.COMPAT_RULE_DECLARED
    assert comp["legs"]["plus"]["tree"] == T2 and comp["legs"]["plus"]["pin"] == T2 and Path(comp["legs"]["plus"]["root"]) == c4_store.b
    assert comp["legs"]["minus"]["tree"] == T1 and comp["legs"]["minus"]["pin"] == PIN and Path(comp["legs"]["minus"]["root"]) == c4_store.a
    assert len(comp["rungs"]) == 2 and all(r["bound_to_source"] for r in comp["rungs"]) and {r["tree"] for r in comp["rungs"]} == {T1, T2}
    ck, nk = V.cell_key(QWEN, "main", CONST_PROMPT), V.cell_key(QWEN, "main", NONE)
    assert rep["cells"][ck]["ladder"]["subset_ids"] != rep["cells"][nk]["ladder"]["subset_ids"]
    assert rep["cells"][ck]["readings"]["merged"]["n_from_rung"] == rep["cells"][ck]["ladder"]["n_subset"] > 0
    # a same-tree contrast of the same view carries the record too, cross_tree false
    same = contrast_of(rep, QWEN, "main", CONST_LORA)["compatibility"]
    assert same["cross_tree"] is False and same["declared"] is True and same["legs"]["plus"]["tree"] == T1 and len(same["rungs"]) == 1
    # every leg's producer names its root and pin; the extension legs are on T2 under the pinned root
    ext = leg_of(rep, QWEN, "main", CONST_PROMPT)
    assert ext["producer"]["tree_sha"] == T2 and ext["producer"]["pin"] == T2 and ext["producer"]["root_pinned"] is True
    assert ext["producer"]["declared_trees"] == [T2] and ext["arm_contract"] == {"checked": True, "problem": None, "route_class": "prompted", "steered_path": None}
    pub = leg_of(rep, QWEN, "main", NONE)
    assert pub["producer"]["tree_sha"] == T1 and pub["producer"]["pin"] == PIN and pub["arm_contract"]["route_class"] == "base"
    assert rep["gates"]["A7_population"]["status"] == A.GATE_PASS and rep[A.TREE_SET_KEY]["legs"][ext["leg"]] == T2


def test_a_moved_shared_field_refuses_the_cross_tree_contrast_by_name(tmp_path, fake_git):
    """(i) a changed chat template digest on the extension leg; (i′) a wrong SI prefix digest on the prompted leg — each
    refuses by the field's name, and the population gate carries the refusal."""
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_LORA), (QWEN, "main", CONST_PROMPT))
    store = build_cross_tree(tmp_path / "tpl", published_rows=[r for r in rows if r.get("wave") is None], ext_rows=[r for r in rows if r.get("wave")],
                             esc_none=False, esc_ext=False, ext_common={"chat_template_hash": "ffffffffffffffff"})
    rep = analyze(roots_of(store), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    c = contrast_of(rep, QWEN, "main", CONST_PROMPT)
    assert not c["computable"] and "shared-field projection" in c["reason"] and "chat_template_hash" in c["reason"]
    assert c["compatibility"]["projection_equal"] is False and c["compatibility"]["cross_tree"] is True
    assert contrast_of(rep, QWEN, "main", CONST_LORA)["computable"]
    assert rep["gates"]["A7_population"]["status"] == A.GATE_FAIL and "chat_template_hash" in rep["gates"]["A7_population"]["detail"]

    def wrong_si(m):
        m["resolved_config"]["system_sha"] = "0000000000000000"

    store2 = build_cross_tree(tmp_path / "si", published_rows=[r for r in rows if r.get("wave") is None], ext_rows=[r for r in rows if r.get("wave")],
                              esc_none=False, esc_ext=False, ext_common={"manifest_edit": wrong_si})
    rep2 = analyze(roots_of(store2), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    leg = leg_of(rep2, QWEN, "main", CONST_PROMPT)
    assert leg["refused"] and "system_sha" in leg["refused"] and "another system prompt" in leg["refused"]
    assert leg["arm_contract"]["checked"] and "system_sha" in leg["arm_contract"]["problem"]
    assert leg["leg"] in rep2["unusable_legs"] and rep2["gates"]["A0_completeness"]["status"] == A.GATE_FAIL


def test_launch_attestation_binds_every_flag_across_arms_and_rung_to_source(tmp_path, fake_git, c4_report):
    """C2 (code review 2026-09-12) on this port: the launch is read from the standup's own log through the shared helper,
    every echoed flag compared across the arms of a contrast and between a rung and its source base; a changed dtype refuses
    by the flag's name, a leg without the record refuses with the remedy and a `None` verdict, a rung whose record is gone
    cannot be bound (`bound_to_source: None`, its reason filed), and the undeclared published shape refuses nothing but never
    claims a verdict it did not read."""
    rep = c4_report
    c = contrast_of(rep, QWEN, "main", CONST_LORA)
    assert c["compatibility"]["server_settings_equal"] is True and all(c["compatibility"]["launch_attested"].values())
    leg = leg_of(rep, QWEN, "main", CONST_LORA)
    assert leg["launch"]["status"] == M.LAUNCH_ATTESTED and leg["launch"]["standup_dir"].endswith("/gsm8k")
    assert json.loads(leg["launch"]["settings"]["enable_prefix_caching"]) is False and json.loads(leg["launch"]["settings"]["dtype"]) == "bfloat16"
    merged = [r for cc in rep["contrasts"] for r in A._mapping(cc.get("compatibility")).get("rungs") or []]
    assert merged and all(r["bound_to_source"] is True and r["bound_to_source_reason"] is None for r in merged)
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_LORA), (QWEN, "main", CONST_PROMPT))
    pub, ext = [r for r in rows if r.get("wave") is None], [r for r in rows if r.get("wave")]

    def set_dtype(args):
        args["dtype"] = "float16"
    store = build_cross_tree(tmp_path / "dtype", published_rows=pub, ext_rows=ext, esc_none=False, esc_ext=False, ext_common={"launch_edit": set_dtype})
    r = analyze(roots_of(store), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    c = contrast_of(r, QWEN, "main", CONST_PROMPT)
    assert not c["computable"] and "normalised server settings differ — launch.dtype" in c["reason"] and c["compatibility"]["server_settings_equal"] is False
    assert contrast_of(r, QWEN, "main", CONST_LORA)["computable"]
    assert r["eightarm"]["pooled_by_arm_merged"]["prompted_constitution"]["pairs"][QWEN]["compatible"] is False
    # the GSM8K port shares the MATH-500 arm contract (`V.arm_contract_problem = M.arm_contract_problem`): a leg served
    # (or recorded) at S = 1 beside its row's budget S is refused by the ARM CONTRACT, never pooled; an echo naming another S than the
    # record is not this leg's launch — `launch.status: absent`, the contrast refuses by name, `bound_to_source` never reads True
    s_base = G.S.served_seqs(QWEN, G.MAX_MODEL_LEN_BASE)
    assert s_base == 741_888 // G.MAX_MODEL_LEN_BASE      # qwen35_4b's 2026-09-13 lens-stack row (42 x 17,664)

    def record_s_one(m):
        m[G.C.MANIFEST_KEY_TINYCAP]["served"]["max_num_seqs"] = m[G.C.MANIFEST_KEY_TINYCAP]["served"]["concurrency"] = m[G.C.MANIFEST_KEY_CONCURRENCY] = 1
    store = build_cross_tree(tmp_path / "s1", published_rows=pub, ext_rows=ext, esc_none=False, esc_ext=False, ext_common={"manifest_edit": record_s_one})
    r = analyze(roots_of(store), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    leg = leg_of(r, QWEN, "main", CONST_PROMPT)
    assert leg["refused"] is not None and f"served max_num_seqs 1 != the standup's resolved S {s_base} (geometry.max_num_seqs)" in leg["refused"]
    assert not contrast_of(r, QWEN, "main", CONST_PROMPT)["computable"] and contrast_of(r, QWEN, "main", CONST_LORA)["computable"]

    def echo_s_one(args):
        args["max_num_seqs"], args["max_num_batched_tokens"] = 1, args["max_model_len"]
    store = build_cross_tree(tmp_path / "echo", published_rows=pub, ext_rows=ext, esc_none=False, esc_ext=False, ext_common={"launch_edit": echo_s_one})
    r = analyze(roots_of(store), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    leg = leg_of(r, QWEN, "main", CONST_PROMPT)
    assert leg["refused"] is None and leg["launch"]["status"] == M.LAUNCH_ABSENT and leg["launch"]["geometry_reconciled"] is False
    assert any("the echo's max_num_seqs 1 != the manifest's served max_num_seqs" in p for p in leg["launch"]["problems"]), leg["launch"]["problems"]
    c = contrast_of(r, QWEN, "main", CONST_PROMPT)
    assert not c["computable"] and "launch settings cannot be compared" in c["reason"] and c["compatibility"]["server_settings_equal"] is None
    assert leg_of(r, QWEN, "main", CONST_LORA)["launch"]["geometry_reconciled"] is True
    store = build_cross_tree(tmp_path / "missing", published_rows=pub, ext_rows=ext, esc_none=False, esc_ext=False, ext_common={"launch_record": False})
    r = analyze(roots_of(store), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    c = contrast_of(r, QWEN, "main", CONST_PROMPT)
    leg = leg_of(r, QWEN, "main", CONST_PROMPT)
    assert leg["refused"] is None and leg["launch"]["status"] == M.LAUNCH_ABSENT
    assert not c["computable"] and "launch settings cannot be compared" in c["reason"] and "pull the job's serving/" in c["reason"]
    assert c["compatibility"]["server_settings_equal"] is None and c["compatibility"]["recorded_settings_equal"] is True
    assert sorted(c["compatibility"]["launch_attested"].values()) == [False, True] and "unattested" in c["compatibility"]["server_settings_note"]
    # a rung whose server log is gone: the pair cannot be bound on launch settings
    esc_rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_LORA))
    store = build_cross_tree(tmp_path / "rung", published_rows=esc_rows, ext_rows=[], esc_none=True, esc_ext=False)
    (store.a / G.STANDUP_DIR_PARENT / G.standup_jobdir(row(*ESC_NONE), G.CAP_RUNG) / G.STANDUP_TASK_SUBDIR / M.LAUNCH_LOG_FILE).unlink()
    r = analyze(roots_of(store), matrix=esc_rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    c = contrast_of(r, QWEN, "main", CONST_LORA)
    rung = next(x for x in c["compatibility"]["rungs"] if x["source"] == leg_of(r, QWEN, "main", NONE)["leg"])
    assert rung["bound_to_source"] is None and "cannot be bound to its source base on launch settings" in rung["bound_to_source_reason"]
    assert not c["computable"] and "cannot be bound to its source base on launch settings" in c["reason"]
    # the undeclared published shape: the one-sha rule stands, nothing refuses on a missing record, the verdict is None
    G.build_rows(tmp_path / "pub_only", pub, producer_fields=True, launch_record=False)
    r = analyze(tmp_path / "pub_only", PIN, matrix=pub, overlap=None)
    c = contrast_of(r, QWEN, "main", CONST_LORA)
    assert c["computable"] and c["compatibility"]["declared"] is False and c["compatibility"]["server_settings_equal"] is None
    assert not any(c["compatibility"]["launch_attested"].values())


def test_a_refused_contrast_never_pools_the_arm_as_available(tmp_path, fake_git):
    """B3 (code review 2026-09-12, the MATH-500 finding applied to this port): the pooled-by-arm fields apply the SAME
    arm↔none policy every contrast runs through, per panel model, before pooling — a moved chat template on one model's
    constitution leg leaves that arm UNAVAILABLE under both readings, naming the model and the field, with no point or CI;
    the pair records say which pairs were admitted and why."""
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_LORA), (QWEN, "main", CONST_PROMPT))
    store = build_cross_tree(tmp_path / "tpl", published_rows=[r for r in rows if r.get("wave") is None], ext_rows=[r for r in rows if r.get("wave")],
                             esc_none=False, esc_ext=False, ext_common={"chat_template_hash": "ffffffffffffffff"})
    rep = analyze(roots_of(store), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    assert not contrast_of(rep, QWEN, "main", CONST_PROMPT)["computable"]
    for field in ("pooled_by_arm_base", "pooled_by_arm_merged"):
        pooled = rep["eightarm"][field]
        rec = pooled["prompted_constitution"]
        assert rec["status"] == A.POOLED_UNAVAILABLE
        assert f"{QWEN}/main/{CONST_PROMPT}: incompatible with {QWEN}/main/{NONE}" in rec["reason"] and "chat_template_hash" in rec["reason"]
        assert not any(k in rec for k in ("primary", "score_as_is", "n_items", "weights"))
        assert rec["pairs"][QWEN]["compatible"] is False and rec["pairs"][QWEN]["cross_tree"] is True and "chat_template_hash" in rec["pairs"][QWEN]["reason"]
        # the same model's LoRA pair (both legs at the published tree) is admitted; the other panel models are simply absent
        assert pooled["lora_mix"]["pairs"][QWEN] == {"compatible": True, "reason": None, "cross_tree": False}
        assert pooled["lora_mix"]["status"] == A.POOLED_UNAVAILABLE and f"{GEMMA}/main/{NONE}: absent" in pooled["lora_mix"]["reason"]
        assert A.COMPAT_RULE_DECLARED in rec["pair_rule"]


def test_without_a_declaration_the_published_one_sha_rule_stands_verbatim(tmp_path):
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_LORA), rows=ROWS)
    other = PIN + "f" * 33
    G.build_all(tmp_path / "mixed", rows=rows, overrides={(QWEN, "main", CONST_LORA): {"tree_sha": other}})
    rep = analyze(tmp_path / "mixed", matrix=rows, overlap=None)
    c = contrast_of(rep, QWEN, "main", CONST_LORA)
    assert not c["computable"] and c["reason"].startswith("the legs' producer trees differ — a mixed-producer comparison")
    assert c["compatibility"]["rule"] == A.COMPAT_RULE_ONE_TREE and c["compatibility"]["declared"] is False and c["compatibility"]["cross_tree"] is True
    assert "design" in rep and "status_by_row" not in rep["design"] and "eightarm" not in rep and "gen_roots" not in rep
    assert leg_of(rep, QWEN, "main", NONE)["arm_contract"] == {"checked": False, "problem": None, "route_class": "base", "steered_path": None}


# --------------------------------------------------------------------------- #
# (2) a corrupted rung identity is REJECTED by name                            #
# --------------------------------------------------------------------------- #


def _rung_rows():
    return small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_PROMPT))


def _build_rung_with(tmp, *, manifest_edit=None, subset_edit=None):
    """The published qwen none (not escalated) at T1 plus the extension constitution cell at T2, escalated, its rung
    built by hand so ONE corruption can be planted (the rung manifest, or the planner's subset file)."""
    rows = _rung_rows()
    store = build_cross_tree(tmp, published_rows=[r for r in rows if r.get("wave") is None], ext_rows=[], esc_none=False)
    r = row(*ESC_EXT)
    common = {"producer_fields": True, "tree_sha": T2}
    esc = G.build_escalated_cell(store.b, r, at_cap=range(180, 187), loops={180}, write_rung=False, **common)
    subset, doc = esc["subset"], esc["subset_doc"]
    manifest = completion.read_store_manifest(esc["source_cell_dir"])
    G.build_store(store.b, r, correct={j for j in range(len(subset)) if j % 2 == 0}, cap=G.CAP_RUNG, ids=subset, rung_source=manifest,
                  subset_doc=doc, subset_path=esc["subset_path"], completion_tokens={j: 9000 for j in range(len(subset))}, ladder=False,
                  manifest_edit=manifest_edit, **common)
    if subset_edit is not None:
        G.write_rung_subset(store.b, r, doc, edit=subset_edit)
    return store, rows


def test_a_rung_whose_subset_is_not_the_eligible_set_is_rejected_by_name(tmp_path, fake_git):
    extra = G.IDS[5]

    def widen(d):
        d["ids"].append(extra)
        d["n_at_cap"] += 1

    store, rows = _build_rung_with(tmp_path, subset_edit=widen)
    rep = analyze(roots_of(store), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    ck = V.cell_key(QWEN, "main", CONST_PROMPT)
    lad = rep["cells"][ck]["ladder"]
    assert lad["escalated"] and lad["subset_problem"] and "eligible_ids" in lad["subset_problem"]
    assert rep["cells"][ck]["rung_leg"] is None
    assert rep["gates"]["A5_at_cap"]["status"] == A.GATE_FAIL and ck in rep["gates"]["A5_at_cap"]["detail"]
    assert rep["gates"]["A6_ladder"]["status"] == A.GATE_FAIL and "escalated but the rung leg is absent" in rep["gates"]["A6_ladder"]["detail"]
    assert A.production_refusals(rep)


def test_a_rung_at_another_cap_is_rejected_by_name(tmp_path, fake_git):
    def other_cap(m):
        m["tinycap"]["max_tokens_by_task"][G.TASK] = 12000

    store, rows = _build_rung_with(tmp_path, manifest_edit=other_cap)
    rep = analyze(roots_of(store), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    rung = leg_of(rep, QWEN, "main", CONST_PROMPT, cap=G.CAP_RUNG)
    assert rung["refused"] and "max_tokens_by_task" in rung["refused"] and "16384" in rung["refused"]
    c = contrast_of(rep, QWEN, "main", CONST_PROMPT)
    assert not c["computable"] and c["reason"].startswith(f"{V.cell_key(QWEN, 'main', CONST_PROMPT)} rung:") and "max_tokens_by_task" in c["reason"]
    assert rung["leg"] in rep["unusable_legs"] and rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL


def test_a_rung_bound_to_the_other_base_is_rejected_by_name(tmp_path, fake_git):
    rows = _rung_rows()
    other_hash = {}

    def rebind(m):
        m["tinycap"]["cap_rung"]["source_identity_hash"] = other_hash["h"]
        m["item_subset"]["source_identity_hash"] = other_hash["h"]

    # the OTHER base: qwen's published none — its identity hash is what the rung is (wrongly) bound to
    store = build_cross_tree(tmp_path / "pre", published_rows=[r for r in rows if r.get("wave") is None], ext_rows=[], esc_none=False)
    other_hash["h"] = completion.read_store_manifest(next((store.a / row(*ESC_NONE)["out_root"] / "cells").glob("*")))["harness_config_hash"]
    store, rows = _build_rung_with(tmp_path, manifest_edit=rebind)
    rep = analyze(roots_of(store), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    rung = leg_of(rep, QWEN, "main", CONST_PROMPT, cap=G.CAP_RUNG)
    assert rung["refused"] and "source_identity_hash" in rung["refused"] and "source store" in rung["refused"]
    c = contrast_of(rep, QWEN, "main", CONST_PROMPT)
    assert not c["computable"] and "source_identity_hash" in c["reason"]


# --------------------------------------------------------------------------- #
# (3) an unadmitted third tree; a duplicate logical leg                        #
# --------------------------------------------------------------------------- #


def test_a_third_unadmitted_tree_refuses_by_name(tmp_path, fake_git):
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_PROMPT))
    store = build_cross_tree(tmp_path, published_rows=[r for r in rows if r.get("wave") is None], ext_rows=[r for r in rows if r.get("wave")],
                             esc_none=False, esc_ext=False, ext_tree=T3)
    # the root is pinned to the DECLARED tree T2, the store carries T3: refused by name at collection
    rep = analyze(roots_of(store), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    leg = leg_of(rep, QWEN, "main", CONST_PROMPT)
    assert leg["refused"] and "does not carry the wave pin" in leg["refused"] and T3 in leg["refused"] and T2 in leg["refused"]
    assert leg["producer"]["matches_pin"] is False and leg["leg"] in rep["unusable_legs"]
    assert not contrast_of(rep, QWEN, "main", CONST_PROMPT)["computable"] and rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL
    # pinning a root to a tree the declaration does not admit refuses before any leg is read
    with pytest.raises(A.AnalysisError, match="unadmitted producer tree"):
        analyze([f"{store.a}={PIN}", f"{store.b}={T3}"], matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    # a bare second root admits the declared tree's legs — and still not T3's
    rep2 = analyze([f"{store.a}={PIN}", str(store.b)], matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    assert "declared trees" in leg_of(rep2, QWEN, "main", CONST_PROMPT)["refused"]


def test_a_duplicate_logical_leg_across_two_roots_refuses_by_name(tmp_path, fake_git):
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_LORA), rows=ROWS)
    a, b = tmp_path / "a", tmp_path / "b"
    G.build_rows(a, rows, producer_fields=True)
    G.build_store(b, row(QWEN, "main", NONE, rows=ROWS), correct=G.CORRECT_NONE, producer_fields=True, tree_sha=T2)
    rep = analyze([f"{a}={PIN}", f"{b}={T2}"], matrix=rows, overlap=None, tree_set=TREE_SET, waves=["published"])
    leg = leg_of(rep, QWEN, "main", NONE)
    assert leg["refused"] and "exists under 2 gen roots" in leg["refused"] and "filesystem union" in leg["refused"]
    assert leg_of(rep, QWEN, "main", CONST_LORA)["refused"] is None
    with pytest.raises(A.AnalysisError, match="named twice"):
        analyze([f"{a}={PIN}", f"{a}={PIN}"], matrix=rows, overlap=None, tree_set=TREE_SET, waves=["published"])


# --------------------------------------------------------------------------- #
# (4) the overlap gate holds to the PRIMARY pin                                #
# --------------------------------------------------------------------------- #


def test_the_overlap_gate_holds_to_the_primary_pin(c4_store, c4_report):
    g = c4_report["gates"]["A10_overlap"]
    assert g["status"] == A.GATE_PASS and g["store_pin"] == PIN and g["report_store_pin"] == PIN
    assert all(per["stock_store_pin"] == PIN and per["producer_tree_sha"] == T1 for per in g["models"].values())
    other = copy.deepcopy(c4_store.overlap)
    other["store_pin"] = T2[:7]
    g2 = A.gate_overlap(other, list(G.MODEL_KEYS), G.IDS, store_pin=PIN)
    assert g2["status"] == A.GATE_FAIL and "ANOTHER wave" in g2["detail"]
    # the declared tree does not widen the overlap binding: a record whose stock legs are on T2 refuses under the primary pin
    other = copy.deepcopy(c4_store.overlap)
    other["models"][QWEN]["stock"]["producer_tree_sha"] = T2
    other["models"][QWEN]["stock"]["outcomes_producer_tree_sha"] = T2
    g3 = A.gate_overlap(other, list(G.MODEL_KEYS), G.IDS, store_pin=PIN)
    assert g3["status"] == A.GATE_FAIL and "ANOTHER wave" in g3["detail"]


# --------------------------------------------------------------------------- #
# (5) the expected leg set is DERIVED; (6) the C4 fixture and the negative     #
# --------------------------------------------------------------------------- #


def test_the_expected_leg_set_is_derived_and_the_ledger_iterates_it(c4_store, c4_report):
    rep = c4_report
    required = V.required_rows(V.load_matrix(EXT_ROWS)["launch"], WAVES)
    expected = {V.leg_key(r["model_key"], r["view"], r["cell"], G.CAP_BASE) for r in required}
    expected |= {V.leg_key(*t, G.CAP_RUNG) for t, k in ((ESC_NONE, "none"), (ESC_EXT, "ext")) if c4_store.esc[k]["rung_cell_dir"] is not None}
    assert set(rep["design"]["required_legs"]) == expected == set(rep["legs"])
    assert rep["design"]["n_required_legs"] == len(expected) == len(required) + len(c4_store.esc)
    assert rep["design"]["n_required"] == len(required) and rep["design"]["n_launch_rows"] == len(required)
    assert set(rep["gates"]["A1_ledger"]["attempts_by_leg"]) == expected and set(rep["gates"]["A0_completeness"]["legs"]) == expected
    assert rep["gates"]["A0_completeness"]["n_usable_base"] == len(required) and rep["gates"]["A0_completeness"]["n_usable_rung"] == len(c4_store.esc)


def test_the_c4_fixture_passes_and_files_every_design_status(c4_store, c4_report):
    rep = c4_report
    sbr = rep["design"]["status_by_row"]
    assert rep["design"]["waves"] == WAVES and set(rep["design"]["waves_in_matrix"]) == {"published", G.WAVE_EXT, G.WAVE_STEER}
    steer = [k for k, v in sbr.items() if v["status"] == "deferred"]
    assert len(steer) == len(G.steer_rows(0)) == rep["design"]["n_deferred"] == len(rep["deferred"])
    assert all(sbr[k]["reason"] == G.DEFERRED_REASON and sbr[k]["object_source"] == "external" and sbr[k]["arm"] in ("alphasteer", "arditi") for k in steer)
    held = [k for k, v in sbr.items() if v["status"] == "held"]
    assert len(held) == 4 == rep["design"]["n_held"] and all(sbr[k]["reason"] == G.HELD_REASON for k in held)
    assert sum(1 for v in sbr.values() if v["status"] == "required") == rep["design"]["n_required"] and rep["design"]["n_not_analysed"] == 0
    # the eight slots per model on the main view: the two steering arms deferred, every other arm required (glm: absent constitution)
    arms = rep["design"]["arms_by_model"]
    for m in (GEMMA, QWEN, NEMO):
        assert arms[m]["alphasteer"] == arms[m]["arditi"] == "deferred" and all(arms[m][k] == "required" for k in ("bare", "prompted", "prompted_constitution", "lora_ctl2", "lora_mix", "object"))
    assert arms[GLM]["prompted_constitution"] == "absent" and arms[GLM]["alphasteer"] == "absent"
    assert set(rep["design"]["modes"]) == {"thinking_card"} and rep["design"]["panel"]["models"] == [GEMMA, QWEN, NEMO]
    assert any(t.startswith(f"{V.cell_key(GEMMA, 'main', G.CELL_ARDITI)}: DEFERRED") for t in rep["findings"])
    # the extension legs are read from the wave-isolated root and carry their wave
    ext = leg_of(rep, NEMO, "main", CONST_PROMPT)
    assert ext["wave"] == G.WAVE_EXT and Path(ext["gen_root"]) == c4_store.b and ext["out_root"].endswith("controls_ext/nemotron3nano4b/prompted--si_constitution/thinking_card")
    assert contrast_of(rep, NEMO, "main", CONST_PROMPT)["computable"] and contrast_of(rep, GEMMA, "main", PROMPTED)["computable"]


def test_waves_published_alone_never_collects_the_extension_rows(c4_store):
    rep = analyze_c4(c4_store, waves=["published"])
    assert rep["gates"]["failed"] == [] and A.production_refusals(rep) == []
    sbr = rep["design"]["status_by_row"]
    ext_keys = [V.cell_key(r["model_key"], r["view"], r["cell"]) for r in G.ext_rows()]
    assert all(sbr[k]["status"] == "not_analysed" and "not among the analysed waves" in sbr[k]["reason"] for k in ext_keys)
    assert rep["design"]["n_not_analysed"] == len(ext_keys) and not any(k.startswith(f"{NEMO}/main/{CONST_PROMPT}") for k in rep["legs"])
    assert rep["design"]["n_launch_rows"] == 38 and rep["design"]["arms_by_model"][GEMMA]["prompted_constitution"] == "not_analysed"
    assert rep["eightarm"]["pooled_by_arm_merged"]["prompted_constitution"]["status"] == A.POOLED_UNAVAILABLE


def test_one_absent_extension_leg_fails_completeness_by_name(c4_store, tmp_path):
    b2 = tmp_path / "pull_ext_minus_one"
    shutil.copytree(c4_store.b, b2)
    gone = row(NEMO, "main", CONST_PROMPT)
    shutil.rmtree(b2 / gone["out_root"])
    rep = analyze_c4(c4_store, b=b2)
    key = V.leg_key(NEMO, "main", CONST_PROMPT, G.CAP_BASE)
    assert rep["gates"]["A0_completeness"]["status"] == A.GATE_FAIL and key in rep["gates"]["A0_completeness"]["detail"]
    assert key in rep["unusable_legs"] and rep["design"]["status_by_row"][V.cell_key(NEMO, "main", CONST_PROMPT)]["status"] == "required"
    assert key in rep["design"]["required_legs"] and leg_of(rep, NEMO, "main", CONST_PROMPT)["missing"]
    assert A.production_refusals(rep) and rep["verdict"]["word"] is None
    assert rep["eightarm"]["pooled_by_arm_merged"]["prompted_constitution"]["status"] == A.POOLED_UNAVAILABLE
    assert f"{NEMO}/main/{CONST_PROMPT}: missing" in rep["eightarm"]["pooled_by_arm_merged"]["prompted_constitution"]["reason"]


# --------------------------------------------------------------------------- #
# (7) pooled_by_arm_base / _merged over the main view                          #
# --------------------------------------------------------------------------- #


def test_pooled_by_arm_is_invariant_to_it_rows_and_reads_unavailable_on_a_held_model(c4_store, c4_report):
    rep = c4_report
    ea = rep["eightarm"]
    assert ea["view"] == "main" and ea["panel_models"] == [GEMMA, QWEN, NEMO] and ea["not_in_panel"] == [GLM] and list(ea["held_out"]) == [LLAMA]
    assert "thinking: not available" in ea["held_out"][LLAMA] and ea["arms"] == PANEL_KINDS
    for field in ("pooled_by_arm_base", "pooled_by_arm_merged"):
        pooled = ea[field]
        assert list(pooled) == PANEL_KINDS
        assert {k for k, r in pooled.items() if r["status"] == A.POOLED_AVAILABLE} == {"bare", "prompted", "prompted_constitution", "lora_ctl2", "lora_mix", "object"}
        for k in ("alphasteer", "arditi"):
            assert pooled[k]["status"] == A.POOLED_UNAVAILABLE and "deferred" in pooled[k]["reason"]
        reading = "r8192" if field.endswith("base") else "merged"
        for k, rec in pooled.items():
            if rec["status"] != A.POOLED_AVAILABLE:
                continue
            p = rec["primary"]
            assert p["point"] == pytest.approx(np.mean(list(p["by_model"].values()))) and rec["weights"] == {m: pytest.approx(1 / 3) for m in ea["panel_models"]}
            for m in ea["panel_models"]:
                c = rep["cells"][V.cell_key(m, "main", M.ARM_BY_KIND[k].cell)]
                assert p["by_model"][m] == pytest.approx(c["readings"][reading]["acc_stock"])
                assert p["bare"]["by_model"][m] == pytest.approx(rep["cells"][V.cell_key(m, "main", NONE)]["readings"][reading]["acc_stock"])
            assert p["ci_low"] <= p["point"] <= p["ci_high"] and rec["B"] == B_FAST and rec["n_items"] == 200
    # the merged and the 8,192-only readings differ where a panel cell escalated (qwen's none and its constitution cell)
    assert ea["pooled_by_arm_merged"]["bare"]["primary"]["by_model"][QWEN] != ea["pooled_by_arm_base"]["bare"]["primary"]["by_model"][QWEN]
    assert ea["n_available_base"] == ea["n_available_merged"] == 6
    # the `it` rows never enter: dropping every it row leaves both pooled blocks byte-identical
    rep2 = analyze_c4(c4_store, matrix=[r for r in EXT_ROWS if r["view"] != "it"])
    for field in ("pooled_by_arm_base", "pooled_by_arm_merged"):
        assert json.dumps(rep2["eightarm"][field], sort_keys=True) == json.dumps(ea[field], sort_keys=True)
    assert rep2["design"]["n_noise_floor_pairs"] == 0
    # a HELD panel model's arm leg -> unavailable with the reason, never a two-model mean
    rows = [dict(r) for r in EXT_ROWS]
    r = next(x for x in rows if (x["model_key"], x["view"], x["cell"]) == (GEMMA, "main", CONST_PROMPT))
    r.update(status="held", held_reason="synthetic hold: the preflight moved a shared field")
    rep3 = analyze_c4(c4_store, matrix=rows)
    for field in ("pooled_by_arm_base", "pooled_by_arm_merged"):
        rec = rep3["eightarm"][field]["prompted_constitution"]
        assert rec["status"] == A.POOLED_UNAVAILABLE and f"{GEMMA}/main/{CONST_PROMPT}: held" in rec["reason"]
        assert rep3["eightarm"][field]["lora_mix"] == ea[field]["lora_mix"]
    assert rep3["design"]["status_by_row"][V.cell_key(GEMMA, "main", CONST_PROMPT)] == {
        "status": "held", "wave": G.WAVE_EXT, "reason": "synthetic hold: the preflight moved a shared field", "model_key": GEMMA, "view": "main",
        "cell": CONST_PROMPT, "arm": "prompted_constitution", "object_source": None,
        # E3: the it-doc carriage is None on every row of another source
        "lane": None, "qualification": None, "qualification_pattern": None}
    assert rep3["gates"]["failed"] == [] and V.leg_key(GEMMA, "main", CONST_PROMPT, G.CAP_BASE) not in rep3["legs"]


def test_the_pooled_bootstrap_applies_one_shared_index_vector_per_replicate(c4_report):
    """The same draws reach every model's leg: re-ordering the models leaves every pooled CI identical (a per-model
    stream would hand each model different indices), the estimand text and the report say so, and two calls agree."""
    rep = c4_report
    cells = rep["cells"]

    def legs(cell, reading):
        return {m: A._pooled_leg(cells[V.cell_key(m, "main", cell)], V.leg_key(m, "main", cell, G.CAP_BASE), reading) for m in A.POOLED_PANEL_MODELS}

    arm, bare = legs(CONST_LORA, "merged"), legs(NONE, "merged")
    a = M.pooled_arm_bootstrap(arm, bare, n_resamples=B_FAST, seed=0, ci=0.95)
    b = M.pooled_arm_bootstrap(dict(reversed(list(arm.items()))), dict(reversed(list(bare.items()))), n_resamples=B_FAST, seed=0, ci=0.95)
    for branch in A.BRANCHES:
        for f in ("point", "ci_low", "ci_high"):
            assert a[branch][f] == b[branch][f] and a[branch]["minus_bare"][f] == b[branch]["minus_bare"][f]
        assert a[branch]["minus_bare"]["replicate_signs"] == b[branch]["minus_bare"]["replicate_signs"]
    again = M.pooled_arm_bootstrap(arm, bare, n_resamples=B_FAST, seed=0, ci=0.95)
    assert json.dumps(again, sort_keys=True) == json.dumps(a, sort_keys=True)
    rec = rep["eightarm"]["pooled_by_arm_merged"]["lora_mix"]
    assert rec["primary"]["ci_low"] == a["primary"]["ci_low"] and rec["primary"]["ci_high"] == a["primary"]["ci_high"]
    assert "one index vector per replicate" in rep["eightarm"]["bootstrap"]["shared_indices"] and "ONE item-resampling index vector" in rep["eightarm"]["estimand"]
    rng1, rng2 = np.random.default_rng(0), np.random.default_rng(0)
    assert (rng1.integers(0, 200, size=(3, 200)) == rng2.integers(0, 200, size=(3, 200))).all()


# --------------------------------------------------------------------------- #
# (8) the presentation selectors and the publication gate                      #
# --------------------------------------------------------------------------- #


def _records_naming(doc, tokens):
    return [(st, r["name"]) for st, recs in doc["figures"].items() for r in recs
            if any(t in r["name"] or t in json.dumps(r["source"]) for t in tokens)]


def test_figure_selectors_validate_hide_the_appendix_arms_and_the_publication_gate_rules(c4_report, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    rep = c4_report
    with pytest.raises(A.AnalysisError, match="not a cell of this report"):
        A.make_figures(rep, tmp_path / "bad", exclude_arms=["nope"])
    with pytest.raises(A.AnalysisError, match="cannot be left out"):
        A.make_figures(rep, tmp_path / "bad", exclude_arms=[NONE])
    with pytest.raises(A.AnalysisError, match="not a sampling mode"):
        A.make_figures(rep, tmp_path / "bad", modes=["general"])
    with pytest.raises(A.AnalysisError, match="names no mode"):
        A.make_figures(rep, tmp_path / "bad", modes=[])
    figs_all = A.make_figures(rep, tmp_path / "all")
    figs = A.make_figures(rep, tmp_path / "paper", exclude_arms=list(APPENDIX_ARMS))
    # C1: the PSR cells are out of EVERY render by ruling — the no-selector render records them as its excluded arms, and
    # they never reach a figure or the numbers sidecar; the paper view adds the safety LoRA
    psr_present = sorted(c for c in A.report_cells(rep) if PG.is_psr_cell(c))
    assert psr_present == sorted([CONST_PSR, CTL_PSR, SI12_PSR, DIST_PSR]) and figs_all[A.FIGURES_EXCLUDED_ARMS_KEY] == psr_present
    assert figs[A.FIGURES_EXCLUDED_ARMS_KEY] == sorted(APPENDIX_ARMS) and A.FIGURES_EXCLUDED_KEY not in figs and A.FIGURES_EXCLUDED_KEY not in figs_all
    assert set(figs["written"]) == set(figs_all["written"]) and {A.EIGHTARM_STEM_ACCURACY, A.EIGHTARM_STEM_CONTRASTS} <= set(figs["written"])
    sel = figs[A.FIGURES_SELECTION_KEY]
    assert sel["n_legs_report"] == len(A._leg_rows(rep)) > sel["n_legs_shown"] and sel[A.FIGURES_EXCLUDED_ARMS_KEY] == sorted(APPENDIX_ARMS)
    assert figs_all[A.FIGURES_SELECTION_KEY]["n_legs_shown"] > sel["n_legs_shown"]        # the safety LoRA is shown without the selector
    numbers, numbers_all = (json.loads(Path(f["numbers"]).read_text()) for f in (figs, figs_all))
    assert _records_naming(numbers_all, [SI12_LORA]) and not _records_naming(numbers_all, psr_present) and not _records_naming(numbers, APPENDIX_ARMS)
    assert numbers[A.FIGURES_EXCLUDED_ARMS_KEY] == sorted(APPENDIX_ARMS) and A.FIGURES_SELECTION_KEY in numbers
    assert numbers_all[A.FIGURES_EXCLUDED_ARMS_KEY] == psr_present and A.FIGURES_SELECTION_KEY in numbers_all
    # a report whose only non-bare cells are PSR has nothing to draw: refused by name, never an empty figure set
    with pytest.raises(A.AnalysisError, match="PSR cells .* are out of every figure by ruling"):
        A.validated_selection({**rep, "cells": {k: c for k, c in rep["cells"].items() if c["cell"] in (NONE, CONST_PSR)},
                               "held": [], "design": {**rep["design"], "status_by_row": {}}})
    for stem, recs in numbers["figures"].items():
        for r in recs:
            if r["transform"] is None:
                assert M.T._same_leaf(r["value"], A.report_path(rep, r["source"])), (stem, r["name"])
    # the tables JSON is untouched by the arm selector (the family exclusion keeps its own behaviour)
    assert json.loads(Path(figs["tables"]).read_text()) == json.loads(Path(figs_all["tables"]).read_text())
    # every title carries both counts under the selection, none without it
    import matplotlib.pyplot as plt
    view = {**rep, "figures": {A.FIGURES_EXCLUDED_ARMS_KEY: sorted(APPENDIX_ARMS)}}
    clause = f"{sel['n_legs_report']} legs in report · {sel['n_legs_shown']} shown"
    for stem, fn in A.figures_for(view).items():
        fig, _n = fn(view)
        title = [t for t in fig.texts if t.get_fontweight() == "bold"][0].get_text()
        assert clause in title, (stem, title)
        assert title.count("\n") <= 1 and all(len(line) <= 90 for line in title.split("\n")) and A.axes_height_fraction(fig) >= 0.80, stem
        for ax in fig.axes:
            assert not any(any(t in lab.get_text() for t in APPENDIX_ARMS) for lab in [*ax.get_xticklabels(), *ax.get_yticklabels()]), stem
        plt.close(fig)
        fig, _n = A.figures_for(rep)[stem](rep)
        assert "legs in report" not in [t for t in fig.texts if t.get_fontweight() == "bold"][0].get_text()
        plt.close(fig)
    # the publication gate: the paper view PASSES; the no-selector render PASSES too (its PSR cells are out by ruling); a
    # PLANTED PSR record in a sidecar REFUSES, grouped per (stem, cell) — the gate runs on every render (C1)
    gate = A.publication_gate_record(rep, figs, tmp_path / "paper")
    assert gate["status"] == PG.GATE_PASS and gate["excluded_arms"] == sorted(APPENDIX_ARMS) and set(gate["figures"]) == set(figs["written"])
    gate_all = A.publication_gate_record(rep, figs_all, tmp_path / "all")
    assert gate_all["status"] == PG.GATE_PASS and gate_all["excluded_arms"] == psr_present
    planted = json.loads(Path(figs_all["numbers"]).read_text())
    planted["figures"][A.EIGHTARM_STEM_CONTRASTS].append({"name": f"{QWEN}/main: {CONST_PSR} - none.merged.stock.point",
                                                          "source": "contrasts.0.readings.merged.estimands.stock.primary.stats.cell_minus_none.point",
                                                          "transform": None, "value": 0.1})
    Path(figs_all["numbers"]).write_text(json.dumps(planted), encoding="utf-8")
    with pytest.raises(PG.PublicationRefused) as e:
        A.publication_gate_record(rep, figs_all, tmp_path / "all")
    assert any("PSR arm" in v and CONST_PSR in v for v in e.value.violations)
    # a directory holds ONE selection (the no-selector render's is the PSR ruling; the paper view's adds the safety LoRA)
    with pytest.raises(A.AnalysisError, match=A.FIGURES_EXCLUDED_ARMS_KEY):
        A.make_figures(rep, tmp_path / "paper")
    with pytest.raises(A.AnalysisError, match=A.FIGURES_EXCLUDED_ARMS_KEY):
        A.refuse_mixed_render(tmp_path / "all", [], list(figs["written"]), arms=sorted(APPENDIX_ARMS))
    A.refuse_mixed_render(tmp_path / "paper", [], list(figs["written"]), arms=sorted(APPENDIX_ARMS))
    # the mode selector on the one GSM8K mode is recorded and is a no-op on the rows shown (the PSR ruling hides its rows
    # under every selector alike, so the shown count is the no-selector render's)
    figs_m = A.make_figures(rep, tmp_path / "modes", modes=["thinking_card"])
    assert figs_m[A.FIGURES_MODES_KEY] == ["thinking_card"] and figs_m[A.FIGURES_SELECTION_KEY]["n_legs_shown"] == figs_all[A.FIGURES_SELECTION_KEY]["n_legs_shown"]
    assert figs_m[A.FIGURES_SELECTION_KEY]["n_legs_shown"] < figs_m[A.FIGURES_SELECTION_KEY]["n_legs_report"] and figs_m[A.FIGURES_EXCLUDED_ARMS_KEY] == psr_present


# --------------------------------------------------------------------------- #
# (9) the eight-arm figures for the main view                                   #
# --------------------------------------------------------------------------- #


def test_eight_arm_figures_draw_eight_slots_per_model_a_pending_llama_column_and_the_pooled_group(c4_report):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = c4_report
    stems = set(A.figures_for(rep))
    assert {A.EIGHTARM_STEM_ACCURACY, A.EIGHTARM_STEM_CONTRASTS} <= stems and len(stems) == 13
    published_shape = {**rep, "design": {k: v for k, v in rep["design"].items() if k != "status_by_row"}}
    published_shape.pop("eightarm")
    assert not ({A.EIGHTARM_STEM_ACCURACY, A.EIGHTARM_STEM_CONTRASTS} & set(A.figures_for(published_shape))) and len(A.figures_for(published_shape)) == 11
    caps = A.figure_captions(rep)
    assert A.PEER_PENDING_LABEL in caps[A.EIGHTARM_STEM_ACCURACY] and "llama held out" in caps[A.EIGHTARM_STEM_CONTRASTS]
    fig, nums = A.fig_eightarm_accuracy(rep)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == [GEMMA, QWEN, NEMO, f"{LLAMA}\n{A.PEER_PENDING_LABEL}", f"pooled\n({GEMMA}/{QWEN}/{NEMO})"]
    hatched = [p for p in ax.patches if p.get_hatch()]
    filled = [p for p in ax.patches if not p.get_hatch() and p.get_height() > 0.05]
    # 3 models x 2 deferred + llama's 8 pending + the pooled group's 2 unavailable = 16 empty slots; 3 x 6 + 6 pooled filled bars
    assert len(hatched) == 3 * 2 + 8 + 2 and len(filled) == 3 * 6 + 6
    names = {r["name"] for r in nums}
    assert f"eightarm.acc.merged.point[{GEMMA}/prompted_constitution]" in names and f"eightarm.acc.r8192.point[{QWEN}/bare]" in names
    assert "eightarm.pooled.merged.point[object]" in names and "eightarm.pooled.r8192.point[bare]" in names
    assert not any(LLAMA in n for n in names)
    for r in nums:
        assert M.T._same_leaf(r["value"], A.report_path(rep, r["source"])), r["name"]
    title = [t for t in fig.texts if t.get_fontweight() == "bold"][0].get_text()
    assert title.startswith("Eight arms\n") and "6 of 8 arms present" in title and A.axes_height_fraction(fig) >= 0.80
    plt.close(fig)
    fig, nums = A.fig_eightarm_contrasts(rep)
    ax = fig.axes[0]
    ylabels = [t.get_text() for t in ax.get_yticklabels()]
    # C3: without a peer the Llama rows read the PENDING label (the hard-coded "n/a" is gone; a peer column derives its label)
    assert f"{LLAMA} ({A.PEER_PENDING_LABEL}): prompted_constitution" in ylabels and f"pooled ({GEMMA}/{QWEN}/{NEMO}): object" in ylabels
    assert f"{GEMMA}: arditi" in ylabels and len(ylabels) == 4 * 7 + 7
    texts = [t.get_text() for t in ax.texts]
    assert texts.count(A.PEER_PENDING_LABEL) == 7 and texts.count(M.DEFERRED_SLOT_LABEL) == 3 * 2 + 0 and texts.count("unavailable: no estimand") == 2
    names = {r["name"] for r in nums}
    assert f"eightarm.contrast.merged.point[{QWEN}/prompted_constitution]" in names and "eightarm.contrast.r8192.point[pooled/lora_mix]" in names
    for r in nums:
        assert M.T._same_leaf(r["value"], A.report_path(rep, r["source"])), r["name"]
    assert [t for t in fig.texts if t.get_fontweight() == "bold"][0].get_text().startswith("Arm contrasts\n") and A.axes_height_fraction(fig) >= 0.80
    plt.close(fig)


# --------------------------------------------------------------------------- #
# (10) the Llama models' --peer-report                                        #
# --------------------------------------------------------------------------- #


PEER_NA = A.PEER_THINKING_NOT_AVAILABLE
PEER_HOLD_OUT_REASON = "no thinking switch — a different instrument (thinking: not available); within-model contrasts only"


def _pr_a_shape(doc: dict, *, model: str, arm: str) -> dict:
    """A report in the shape PR-A's analysis writes for a no-switch family (read off `origin/llama-paper-rows`): the design
    block's four ARM keys (`_arm_design`: `thinking_mode_by_model`, `thinking_label_by_model`, `no_switch_models`,
    `held_out_of_pooled_means`) and the arm + its label stamped on every leg, controls row, contrast-summary row,
    truncation-cost row, cell and provenance row (`thinking_mode` / `thinking_label`)."""
    label = A.PEER_THINKING_LABEL[arm]
    doc["design"].update({"thinking_mode_by_model": {model: arm}, "thinking_label_by_model": {model: label},
                          "no_switch_models": [model] if arm == PEER_NA else [],
                          "held_out_of_pooled_means": {model: PEER_HOLD_OUT_REASON} if arm == PEER_NA else {}})
    for leg in doc["legs"].values():
        leg["thinking_mode"], leg["thinking_label"] = arm, label
    for key in ("controls", "contrast_summary", "noise_floor", "truncation_cost", "held", "refused"):
        for row in doc.get(key) or []:
            row["thinking_mode"], row["thinking_label"] = arm, label
            if isinstance(row.get("provenance"), dict):          # PR-A's controls rows nest a provenance row carrying the arm too
                row["provenance"]["thinking_mode"], row["provenance"]["thinking_label"] = arm, label
    for c in doc["cells"].values():
        c["thinking_mode"] = arm
    for row in doc["provenance"]["cells"].values():
        row["thinking_mode"], row["thinking_label"] = arm, label
    for key in ("cells", "costs"):                                # PR-A's truncation block stamps its cells and costs
        for row in doc["truncation"].get(key) or []:
            row["thinking_mode"] = arm
    return doc


def _llama_peer_report(tmp_path, *, arm: str = PEER_NA, pr_a_shape: bool = True):
    """A synthetic PRODUCTION report of another pin naming llama8b: a small qwen store at pin OTHER analysed on its own,
    relabelled (the model key only) and re-shaped to PR-A's report shape (:func:`_pr_a_shape`; ``pr_a_shape=False`` leaves
    the pre-field shape, which PR-A's rule reads as `on`)."""
    other = "a" * 40
    o7 = other[:7]
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_LORA), rows=ROWS)
    root = tmp_path / "peer_store"
    G.build_rows(root, rows, producer_fields=True, sha=o7, tree_sha=other)
    ov = build_overlap(root, models=[QWEN], store_pin=o7, tree_sha=other)
    peer = analyze(root, o7, matrix=rows, overlap=ov)
    assert A.production_refusals(peer) == [], A.production_refusals(peer)
    doc = json.loads(json.dumps(peer, default=str).replace(QWEN, LLAMA))
    if pr_a_shape:
        doc = _pr_a_shape(doc, model=LLAMA, arm=arm)
    path = tmp_path / "llama_report.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path, o7, doc


def test_peer_report_present_assembles_the_llama_column_absent_is_explicit_and_a_wrong_pin_refuses(c4_store, c4_report, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = c4_report
    # ABSENT: recorded by name; the completeness string never reads 4 of 4
    assert rep["design"]["peer_reports"] == [{"path": None, "pin": None, "models": [LLAMA], "status": A.PEER_STATUS_ABSENT,
                                              "reason": rep["design"]["peer_reports"][0]["reason"]}]
    assert "pending" in rep["design"]["peer_reports"][0]["reason"] and rep["peer_reports"] == {}
    text = rep["completeness"]["eight_arm"]["gsm8k"]
    assert text.startswith("3 of 4 panel models present") and LLAMA in text and A.PEER_PENDING_LABEL in text and "4 of 4" not in text
    path, o7, doc = _llama_peer_report(tmp_path)
    # PRESENT (PR-A's bare-path grammar and the plan's REPORT=PIN grammar)
    rep_p = analyze_c4(c4_store, peer_reports=[f"{path}={o7}"])
    assert rep_p["design"]["peer_reports"] == [{"path": str(path), "pin": o7, "models": [LLAMA], "status": A.PEER_STATUS_PRESENT, "reason": None,
                                              "thinking_mode_by_model": {LLAMA: PEER_NA}}]
    assert set(rep_p["peer_reports"]) == {o7} and rep_p["peer_reports"][o7]["models"] == [LLAMA] and rep_p["peer_reports"][o7]["pin"] == o7
    assert rep_p["completeness"]["eight_arm"]["gsm8k"].startswith("4 of 4 panel models present")
    # llama is held out of the pooled panel: both pooled blocks are byte-identical with and without the peer
    assert rep_p["eightarm"]["panel_models"] == [GEMMA, QWEN, NEMO]
    for field in ("pooled_by_arm_base", "pooled_by_arm_merged"):
        assert json.dumps(rep_p["eightarm"][field], sort_keys=True) == json.dumps(rep["eightarm"][field], sort_keys=True)
    assert LLAMA not in rep_p["design"]["models"] and not any(LLAMA in k for k in rep_p["legs"])
    # the llama column is drawn from the peer's gated tables, its numbers resolving in THIS report under peer_reports.<pin>
    fig, nums = A.fig_eightarm_accuracy(rep_p)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels[3] == f"{LLAMA}\nthinking: not available"
    llama_recs = [r for r in nums if f"[{LLAMA}/" in r["name"]]
    assert {r["name"] for r in llama_recs} >= {f"eightarm.acc.merged.point[{LLAMA}/bare]", f"eightarm.acc.merged.point[{LLAMA}/lora_mix]", f"eightarm.acc.merged.ci_low[{LLAMA}/lora_mix]"}
    assert all(r["source"].startswith(f"peer_reports.{o7}.") for r in llama_recs)
    for r in nums:
        assert M.T._same_leaf(r["value"], A.report_path(rep_p, r["source"])), r["name"]
    plt.close(fig)
    fig, nums = A.fig_eightarm_contrasts(rep_p)
    llama_recs = [r for r in nums if f"[{LLAMA}/" in r["name"]]
    assert {r["name"] for r in llama_recs} == {f"eightarm.contrast.{rd}.{f}[{LLAMA}/lora_mix]" for rd in ("merged", "r8192") for f in ("point", "ci_low", "ci_high")}
    assert [t.get_text() for t in fig.axes[0].texts].count(A.PEER_PENDING_LABEL) == 0
    plt.close(fig)
    assert any(f"peer report ({LLAMA}): present at pin {o7}" == f for f in rep_p["findings"])
    # the bare path (PR-A's grammar) is the same admission
    assert analyze_c4(c4_store, peer_reports=[str(path)])["design"]["peer_reports"][0]["status"] == A.PEER_STATUS_PRESENT
    # REPORT=WRONGPIN refuses by name; this run's own pin refuses; a peer naming a model this run names refuses
    with pytest.raises(A.AnalysisError, match="not the peer report's own store pin"):
        analyze_c4(c4_store, peer_reports=[f"{path}={T3[:7]}"])
    own = tmp_path / "own.json"
    own.write_text(json.dumps(rep, default=str), encoding="utf-8")
    with pytest.raises(A.AnalysisError, match="this run's own"):
        analyze_c4(c4_store, peer_reports=[str(own)])
    clash = tmp_path / "clash.json"
    clash.write_text(json.dumps(json.loads(json.dumps(doc).replace(LLAMA, QWEN))), encoding="utf-8")
    with pytest.raises(A.AnalysisError, match="already names"):
        analyze_c4(c4_store, peer_reports=[str(clash)])
    failed = tmp_path / "failed.json"
    failed_doc = copy.deepcopy(doc)
    failed_doc["gates"]["failed"] = ["A7_population"]
    failed_doc["gates"]["A7_population"]["status"] = A.GATE_FAIL
    failed.write_text(json.dumps(failed_doc), encoding="utf-8")
    # a FAILED gate is refused at the schema check (C3), before production_refusals even runs
    with pytest.raises(A.AnalysisError, match="not admissible: \\{'A7_population': 'FAILED'\\}"):
        analyze_c4(c4_store, peer_reports=[str(failed)])
    # PR-A's combine_reports signature: the assembled view stamps every row with its report's pin
    combined = A.combine_reports(rep_p, A.load_peer_reports([str(path)], this=rep_p))
    assert combined["design"]["models"] == [LLAMA, *rep_p["design"]["models"]] and combined["peer_assembly"]["pins"][LLAMA] == o7
    assert {c["pin"] for c in combined["controls"]} == {o7, PIN} and combined["peer_assembly"]["held_out"] == dict(A.POOLED_HELD_OUT)
    # C3 (code review 2026-09-12): PR-A's thinking metadata survives slimming and assembly, and the labels DERIVE from it
    slim = rep_p["peer_reports"][o7]
    assert slim["design"]["thinking_mode_by_model"] == {LLAMA: PEER_NA} and slim["design"]["no_switch_models"] == [LLAMA]
    assert slim["design"]["thinking_label_by_model"] == {LLAMA: "thinking: not available"} and slim["design"]["held_out_of_pooled_means"] == {LLAMA: PEER_HOLD_OUT_REASON}
    assert all(c["thinking_mode"] == PEER_NA for c in slim["cells"].values()) and all(r["thinking_mode"] == PEER_NA for r in slim["contrast_summary"])
    assert rep_p["design"]["peer_reports"][0]["thinking_mode_by_model"] == {LLAMA: PEER_NA}
    fig, _n = A.fig_eightarm_contrasts(rep_p)
    assert any(f"{LLAMA} (thinking: not available): lora_mix" == t.get_text() for t in fig.axes[0].get_yticklabels())
    plt.close(fig)
    # a peer from BEFORE the field (no arm keys) is admitted under PR-A's rule — every model `on` — and labelled so
    path_on, o7_on, _doc_on = _llama_peer_report(tmp_path / "on", pr_a_shape=False)
    rep_on = analyze_c4(c4_store, peer_reports=[str(path_on)])
    assert rep_on["peer_reports"][o7_on]["design"]["thinking_mode_by_model"] == {LLAMA: A.PEER_THINKING_ON}
    fig, _n = A.fig_eightarm_accuracy(rep_on)
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()][3] == f"{LLAMA}\nthinking: on"
    plt.close(fig)
    # C3: admission fails CLOSED — the gate schema, the pin grammar and the metadata consistency are all required
    def refused(edit, match):
        bad_doc = copy.deepcopy(doc)
        edit(bad_doc)
        p_ = tmp_path / "bad.json"
        p_.write_text(json.dumps(bad_doc), encoding="utf-8")
        with pytest.raises(A.AnalysisError, match=match):
            analyze_c4(c4_store, peer_reports=[str(p_)])

    def no_gates(d):
        d["gates"] = {"failed": [], "pending": []}          # production_refusals() alone read this as clean (no FAILED, nothing pending)
    refused(no_gates, "gate record\\(s\\) absent")

    def one_gate_missing(d):
        del d["gates"]["A6_ladder"]
    refused(one_gate_missing, "A6_ladder")

    def odd_status(d):
        d["gates"]["A1_ledger"]["status"] = "ok"
    refused(odd_status, "not admissible")

    def bad_sha(d):
        d["sha"] = "not-a-pin"
    refused(bad_sha, "store pin is not valid")

    def row_contradicts(d):
        d["controls"][0]["thinking_mode"] = A.PEER_THINKING_ON
    refused(row_contradicts, "carries thinking_mode 'on' while the design files 'not_available'")
    # the case this guards — ONLY a leg's arm changed (no evidence row touched) used to be admitted, and peer_slim
    # dropped the contradiction; now EVERY carrier is held to the design, wherever it is stamped
    carriers = A.peer_thinking_mode_carriers(doc)
    assert carriers and all(m == LLAMA and arm == PEER_NA for _w, m, arm in carriers)
    where = {w.split(".")[0].split("[")[0] for w, _m, _a in carriers}
    assert {"legs", "controls", "provenance", "truncation", "contrast_summary", "cells"} <= where, where     # no escalation: no cost rows
    assert any(w.startswith("controls[") and w.endswith(".provenance") for w, _m, _a in carriers)
    assert any(w.startswith("truncation.cells[") for w, _m, _a in carriers)
    assert not any(w.startswith("design") or ".identity" in w for w, _m, _a in carriers)     # a leg's identity projection is not an arm stamp

    def leg_only(d):
        next(iter(d["legs"].values()))["thinking_mode"] = A.PEER_THINKING_ON
    refused(leg_only, r"legs\..* \(llama8b\) carries thinking_mode 'on' while the design files 'not_available'")

    def nested_provenance(d):
        d["controls"][0]["provenance"]["thinking_mode"] = A.PEER_THINKING_ON
    refused(nested_provenance, r"controls\[0\]\.provenance \(llama8b\) carries thinking_mode 'on'")

    def truncation_cell(d):
        d["truncation"]["cells"][0]["thinking_mode"] = A.PEER_THINKING_ON
    refused(truncation_cell, r"truncation\.cells\[0\] \(llama8b\) carries thinking_mode 'on'")

    def provenance_cell(d):
        next(iter(d["provenance"]["cells"].values()))["thinking_mode"] = A.PEER_THINKING_ON
    refused(provenance_cell, r"provenance\.cells\..* \(llama8b\) carries thinking_mode 'on'")

    def gate_per_leg(d):                                            # PR-A's scoring-path gate: a per-leg record keyed by the leg
        lk = next(iter(d["legs"]))
        d["gates"]["A1_ledger"]["legs"] = {lk: {"thinking_mode": A.PEER_THINKING_ON}}
    refused(gate_per_leg, r"gates\.A1_ledger\.legs\..* \(llama8b\) carries thinking_mode 'on'")

    def carrier_without_a_model(d):
        d["gates"]["A1_ledger"]["legs"] = {"not-a-leg": {"thinking_mode": PEER_NA}}
    refused(carrier_without_a_model, "for no model this report names")

    def stray_model(d):
        d["contrast_summary"][0]["model_key"] = "other8b"
    refused(stray_model, "names model 'other8b', not one of design.models")

    def bad_arm(d):
        d["design"]["thinking_mode_by_model"][LLAMA] = "off"
    refused(bad_arm, "arm\\(s\\) outside")

    def design_names_stray(d):
        d["design"]["thinking_mode_by_model"]["other8b"] = PEER_NA
    refused(design_names_stray, "outside design.models")


# --------------------------------------------------------------------------- #
# (11) the trigger provenance and the identity keys                            #
# --------------------------------------------------------------------------- #


def test_trigger_provenance_is_inspect_rungs_and_every_provenance_row_carries_the_identity_keys(c4_store, c4_report):
    rep = c4_report
    assert rep["truncation"]["trigger_provenance"] == IR.GSM8K_TRIGGER_PROVENANCE == rep["provenance"]["trigger"] == A.TRIGGER_PROVENANCE
    assert set(rep["provenance"]["sampling"]) == set(rep["design"]["models"]) and all(isinstance(v, str) for v in rep["provenance"]["sampling"].values())
    keys = {"object_artifact_sha256", "route_kind", "view", "producer_tree_sha", "thinking", "sampling_ref", "sampling_mode", "pin", "root"}
    for k, prow in rep["provenance"]["cells"].items():
        assert keys <= set(prow), k
        assert prow["thinking"] == "on" and prow["sampling_mode"] == "thinking_card" and isinstance(prow["sampling_ref"], str)
    ext = rep["provenance"]["cells"][V.leg_key(NEMO, "main", CONST_PROMPT, G.CAP_BASE)]
    assert ext["pin"] == T2 and ext["producer_tree_sha"] == T2 and Path(ext["root"]) == c4_store.b and ext["wave"] == G.WAVE_EXT and ext["arm"] == "prompted_constitution"
    assert ext["route_kind"] == "prompted" and ext["object_artifact_sha256"] is None and ext["view"] == "main"
    pub = rep["provenance"]["cells"][V.leg_key(QWEN, "main", CONST_LORA, G.CAP_BASE)]
    assert pub["pin"] == PIN and pub["producer_tree_sha"] == T1 and Path(pub["root"]) == c4_store.a and pub["wave"] == "published"
    assert pub["object_artifact_sha256"] == row(QWEN, "main", CONST_LORA)["object_sha256"] and pub["arm"] == "lora_mix"
    assert rep["provenance"]["cells"][V.leg_key(QWEN, "main", CONST_PSR, G.CAP_BASE)]["arm"] == A.ARM_KIND_APPENDIX
    assert rep["controls"][0]["provenance"]["pin"] in (PIN, T2)
    # the per-SI family labels: the constitution cell is its own family with its own fixed hue
    assert A.cell_family(CONST_PROMPT) == A.FAMILY_PROMPTED_CONST and A.FAMILY_LABEL[A.FAMILY_PROMPTED_CONST] == "prompted (constitution)"
    assert A.cell_family(PROMPTED) == A.FAMILY_PROMPTED and A.FAMILY_LABEL[A.FAMILY_PROMPTED] == "prompted (safety_refusal)"
    assert len(set(A.FAMILY_COLOR.values())) == len(A.FAMILY_COLOR) and A.FAMILY_COLOR[A.FAMILY_PROMPTED_CONST] == M.CAT8["violet"]


# --------------------------------------------------------------------------- #
# (12) the main() boundary: tree set, gen roots, waves, the publication gate    #
# --------------------------------------------------------------------------- #


def test_main_runs_the_eight_arm_assembly_and_gates_publication(tmp_path, fake_git, monkeypatch):
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_PSR), (QWEN, "main", CONST_LORA), (QWEN, "main", CONST_PROMPT))
    rows += G.steer_rows(len(rows), models=[QWEN])
    store = build_cross_tree(tmp_path / "store", published_rows=[r for r in rows if r.get("wave") is None],
                             ext_rows=[r for r in rows if r.get("wave") == G.WAVE_EXT], esc_none=False, esc_ext=False)
    overlap = build_overlap(store.a, models=[QWEN])
    files = {"matrix": G.write_matrix(tmp_path / "matrix.json", rows), "ids": G.write_json(tmp_path / "ids.json", G.IDS),
             "pins": G.write_json(tmp_path / "pins.json", G.pins_doc(ROWS)), "sampling-rows": G.write_json(tmp_path / "rows.json", G.SAMPLING_ROWS),
             "render-facts": G.write_json(tmp_path / "facts.json", G.RENDER_FACTS), "overlap": G.write_json(tmp_path / "overlap.json", overlap)}
    common = ["--gen-root", f"{store.a}={PIN}", "--gen-root", f"{store.b}={T2}", "--sha", PIN, "--also-sha", T2, "--tree-diff-allow", "scripts/*",
              "--tree-set-reason", "synthetic: the extension wave's legs come from its own tree", "--waves", "published,ext",
              *[f"--{k}={v}" for k, v in files.items()], "--bootstrap", str(B_FAST), "--no-eval-logs", "--wandb-mode", "disabled", "--no-mirror",
              "--analysis-sha", G.FULL_SHA, "--repo-root", str(REPO)]
    monkeypatch.setitem(sys.modules, "wandb", None)
    rc = A.main([*common, "--out", str(tmp_path / "r1.json"), "--figs-dir", str(tmp_path / "figs1"), "--figure-exclude-arm", CONST_PSR])
    assert rc == 0, rc
    r1 = json.loads((tmp_path / "r1.json").read_text())
    assert r1[A.PUBLICATION_GATE_KEY]["status"] == PG.GATE_PASS and r1["figures"][A.FIGURES_EXCLUDED_ARMS_KEY] == [CONST_PSR]
    assert r1[A.TREE_SET_KEY]["also"] == [T2] and r1["design"]["waves"] == ["published", "ext"] and len(r1["gen_roots"]) == 2
    assert r1["design"]["status_by_row"][V.cell_key(QWEN, "main", G.CELL_ALPHASTEER)]["status"] == "deferred"
    assert "eightarm" in r1 and r1["eightarm"]["pooled_by_arm_merged"]["bare"]["status"] == A.POOLED_UNAVAILABLE      # one panel model only
    assert set(r1["figures"]["written"]) >= {A.EIGHTARM_STEM_ACCURACY, A.EIGHTARM_STEM_CONTRASTS}
    assert A.gate_report(tmp_path / "r1.json") == 0
    # C1: the same render WITHOUT the selector excludes the PSR cell by ruling — the gate runs, PASSES, and the record names
    # the excluded arm; no PSR record reaches the sidecar
    rc = A.main([*common, "--out", str(tmp_path / "r2.json"), "--figs-dir", str(tmp_path / "figs2")])
    assert rc == 0, rc
    r2 = json.loads((tmp_path / "r2.json").read_text())
    assert r2[A.PUBLICATION_GATE_KEY]["status"] == PG.GATE_PASS and r2[A.PUBLICATION_GATE_KEY]["excluded_arms"] == [CONST_PSR]
    assert r2["figures"][A.FIGURES_EXCLUDED_ARMS_KEY] == [CONST_PSR]
    assert not _records_naming(json.loads(Path(r2["figures"]["numbers"]).read_text()), [CONST_PSR])
    # the exit-6 boundary: a refused gate writes the report with the refusal and stops before the mirror / W&B pass
    def refuse(report, figs, figs_dir):
        raise PG.PublicationRefused([f"{A.EIGHTARM_STEM_CONTRASTS}: 1 plotted record(s) presents the PSR arm {CONST_PSR!r} (planted)"], where=str(figs_dir))
    monkeypatch.setattr(A, "publication_gate_record", refuse)
    rc = A.main([*common, "--out", str(tmp_path / "r2b.json"), "--figs-dir", str(tmp_path / "figs2b")])
    assert rc == PG.EXIT_PUBLICATION_REFUSED
    r2b = json.loads((tmp_path / "r2b.json").read_text())
    assert r2b[A.PUBLICATION_GATE_KEY]["status"] == PG.GATE_REFUSED and any(CONST_PSR in v for v in r2b[A.PUBLICATION_GATE_KEY]["violations"])
    assert "wandb_url" not in r2b and "mirrored" not in r2b
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "wandb", None)
    # an unknown --also-sha refuses at the startup boundary: the stub report carries the tree_set problems alone, exit 3
    rc = A.main([*common, "--also-sha", "c" * 40, "--out", str(tmp_path / "r3.json"), "--no-figs"])
    assert rc == A.EXIT_NOT_PRODUCTION
    r3 = json.loads((tmp_path / "r3.json").read_text())
    assert r3[A.TREE_SET_KEY]["problems"] and "legs" not in r3 and A.production_refusals(r3)[0].startswith(A.TREE_SET_KEY)
    # --also-sha without a reason is a usage error
    with pytest.raises(SystemExit):
        A.main([*[x for x in common if x not in ("--tree-set-reason", "synthetic: the extension wave's legs come from its own tree")], "--no-figs"])


# --------------------------------------------------------------------------- #
# (13) E3: the INSTRUCT-DIRECT ("it-doc") batch-64 re-fit wave — rows admitted by the ROW's pin (the admission record's, never   #
# the full22 pin file's), the qualification carried VERBATIM beside every it-doc reading                                          #
# --------------------------------------------------------------------------- #

ITDOC_WAVES = [*WAVES, G.WAVE_ITDOC]
ITDOC_CTL2, ITDOC_MIX, ITDOC_DIFF = G.ITDOC_CELLS
REAL_MATRIX = REPO / "configs" / "sdfbase" / "gsm8k" / "matrix.json"
FROZEN_FIXTURE = REPO / "tests" / "fixtures" / "eight_arms" / "gsm8k_matrix_8f5f1495.json"


def _optional(name, rel):
    p = REPO / rel
    if not p.is_file():
        return None
    try:
        return _load(name, p)
    except Exception:       # noqa: BLE001 — a launcher-side module edited concurrently; an unimportable copy is not this test's finding
        return None


def _itdoc_rows(rows):
    return [r for r in rows if V.is_itdoc_row(r)]


def test_the_itdoc_spellings_are_the_producers_own():
    """The reader's `object_source` spelling and its it-doc fields are the generator's / the binding module's (one spelling for
    the launchers, the sbatch and this reader); `itdoc_fields` reads the ROW and never retypes a sentence."""
    assert V.OBJECT_SOURCE_ITDOC == G.OBJECT_SOURCE_ITDOC == "itdoc" and V.OBJECT_SOURCES == ("external", "itdoc")
    assert V.ITDOC_ROW_FIELDS == ("lane", "qualification", "qualification_pattern")
    bind = _optional("itdoc_bind_e3_reader_pin", "scripts/sdfbase/full22/itdoc_bind.py")
    if bind is not None:
        assert V.OBJECT_SOURCE_ITDOC == bind.OBJECT_SOURCE_ITDOC and G.LANE_ITDOC == json.loads((REPO / bind.ADMISSION_REL).read_text())["lane"]
        assert G.ITDOC_STAGED_PREFIX_BY_KIND == dict(bind.STAGED_PREFIX_BY_KIND) and G.ITDOC_SIDECAR_BY_KIND == dict(bind.SIDECAR_BY_KIND)
    cells = _optional("g8k_cells_e3_reader_pin", "scripts/sdfbase/gsm8k/g8k_cells.py")
    if cells is not None:
        assert V.OBJECT_SOURCE_ITDOC == cells.OBJECT_SOURCE_ITDOC and V.OBJECT_SOURCE_EXTERNAL == cells.OBJECT_SOURCE_EXTERNAL
        assert V.OBJECT_SOURCES == tuple(cells.OBJECT_SOURCES) and G.WAVE_ITDOC == cells.WAVE_ITDOC and tuple(G.ITDOC_CELLS) == tuple(cells.ITDOC_CELLS)
    itdoc = G.itdoc_row(QWEN, ITDOC_CTL2, index=0)
    assert V.is_itdoc_row(itdoc) and V.itdoc_fields(itdoc) == {"lane": G.LANE_ITDOC, "qualification": G.ITDOC_QUALIFICATION,
                                                               "qualification_pattern": G.ITDOC_QUALIFICATION_PATTERN}
    assert V.qualification_text(itdoc) == f"{G.ITDOC_QUALIFICATION} {G.ITDOC_QUALIFICATION_PATTERN}"
    assert V.qualification_text(G.itdoc_row(QWEN, ITDOC_CTL2, index=0, qualification_pattern=None)) == G.ITDOC_QUALIFICATION
    assert V.qualification_text(G.itdoc_row(QWEN, ITDOC_CTL2, index=0, qualification=None, qualification_pattern=None)) is None
    other = row(QWEN, "main", CONST_LORA)
    assert V.itdoc_fields(other) == {"lane": None, "qualification": None, "qualification_pattern": None}
    assert not V.is_itdoc_row(other) and V.qualification_text(other) is None and V.qualification_text(None) is None


REAL_ITDOC_MATRIX = REPO / "configs" / "sdfbase" / "gsm8k" / "matrix_itdoc.json"
ITDOC_VIEWS = REPO / "configs" / "sdfbase" / "gsm8k" / "views_itdoc.yaml"


def _frozen_itdoc_pins() -> Path:
    """The frozen pin file the itdoc generator names (`frozen_inputs:` in views_itdoc.yaml): 4e200b68's copy until 2026-09-14, ece9e8ce's
    (the gemma itdoc64 roster) since the gemma control + difference hold lifted. Read from the views file, never hard-coded, so
    the next re-point re-pins this test by itself (the hard-coded 4e200b68 path read stale the moment the lift landed)."""
    import yaml
    return REPO / yaml.safe_load(ITDOC_VIEWS.read_text(encoding="utf-8"))["frozen_inputs"]["pins"]


@pytest.mark.skipif(not (REAL_MATRIX.is_file() and REAL_ITDOC_MATRIX.is_file() and FROZEN_FIXTURE.is_file()),
                    reason="the tree's matrices / the frozen fixture are absent")
def test_the_real_itdoc_matrix_admits_by_the_rows_pin_and_the_default_matrix_carries_no_itdoc_row():
    """The INSTRUCT-DIRECT wave is a THIRD matrix document, configs/sdfbase/gsm8k/matrix_itdoc.json — schema 2, its
    own wave root, every row wave `itdoc` on the `it` view of the two chat-checkpoint models: per model the CONSTITUTION cell as a
    ROSTER row (the batch-8 object AS SERVED — the frozen fixture's rows 3 / 16 cell, its pin, no object source; pinned by the pin
    file) and the CONTROL and DIFFERENCE cells as ADMISSION-RECORD rows (`object_source: itdoc`, the lane, a 64-hex pin, the
    qualification + pattern sentences verbatim; `launch` where the record binds a bank, else `held` with the record's reason). The
    reader admits the document as-is (identity by the row's pin for an it-doc row; the pin file for the roster rows), `--waves itdoc`
    names exactly its evidence rows, and the DEFAULT matrix (matrix.json) carries NO it-doc row — its holds stay there byte for
    byte (test_tinycap_gsm8k_analysis pins that document). The document also carries, APPENDED behind the held rows,
    the SHORT constitution SI as a prompted route on each `it` view (rows 6 / 7: `extension`, in support, no object / pin /
    generation, never an it-doc row) — eight rows; six launched until 2026-09-14, all eight since gemma's control + difference hold
    lifted through the record (the roster now binds both clusters' gemma `it` views). The reader takes `launch` / `held` from
    the document, so this test follows the record rather than pinning a count."""
    from collections import Counter
    doc = json.loads(REAL_ITDOC_MATRIX.read_text(encoding="utf-8"))
    fixture = json.loads(FROZEN_FIXTURE.read_text(encoding="utf-8"))["rows"]
    mx = V.load_matrix(REAL_ITDOC_MATRIX)
    assert mx["schema"] == V.MATRIX_SCHEMA_2 and mx["wave_root_rel"] == doc["wave_root_rel"] and doc["wave"] == G.WAVE_ITDOC
    rows = mx["rows"]
    assert len(rows) == 8 == doc["counts"]["rows"] and Counter(V.row_wave(r) for r in rows) == {G.WAVE_ITDOC: 8} == doc["counts"]["waves"]
    assert (mx["counts"]["held"], mx["counts"]["launched"]) == (doc["counts"]["held"], doc["counts"]["launched"]) and mx["counts"]["declared"] == 0
    assert {(r["model_key"], r["view"], r["cell"]) for r in rows} == {(m, "it", c) for m in (GEMMA, QWEN) for c in (*G.ITDOC_CELLS, CONST_PROMPT)}
    assert all(r["out_root"] == f"controls_itdoc/{r['tag']}/{r['cell_slug']}/thinking_card" and r["thinking_mode"] == "on" and not r["declared"] for r in rows)
    assert all(r["fitted"] is (r["cell"] != CONST_PROMPT) for r in rows)
    # the prompted rows (E3-14, rows 6 / 7): the short constitution SI on each `it` view — in support, no object, never an it-doc row
    prompted = [r for r in rows if r["cell"] == CONST_PROMPT]
    assert [(r["index"], r["model_key"], r["status"]) for r in prompted] == [(6, GEMMA, "launch"), (7, QWEN, "launch")]
    for r in prompted:
        assert not V.is_itdoc_row(r) and r.get("object_source") is None and r.get("staged_id") is None and r["object_sha256"] is None
        assert r["route_kind"] == "prompted" and r["extension"] and r["held_reason"] is None
        assert V.itdoc_fields(r) == {"lane": None, "qualification": None, "qualification_pattern": None}
    # the roster rows: the constitution cell on the batch-8 object AS SERVED — the frozen fixture's (model, view, cell, staged id, pin)
    roster = [r for r in rows if not V.is_itdoc_row(r) and r["cell"] != CONST_PROMPT]
    assert {r["cell"] for r in roster} == {CONST_LORA} and len(roster) == 2 and all(r["object_source"] is None if "object_source" in r else True for r in roster)
    by_model = {r["model_key"]: r for r in roster}
    for i in (3, 16):
        fx = fixture[i]
        r = by_model[fx["model_key"]]
        assert (r["view"], r["cell"], r["staged_id"], r["object_sha256"]) == (fx["view"], fx["cell"], fx["staged_id"], fx["object_sha256"])
        assert fx["status"] == "held" and r["status"] == "launch" and V.itdoc_fields(r) == {"lane": None, "qualification": None, "qualification_pattern": None}
    # the admission-record rows: identity by the ROW's pin, the lane, the qualification verbatim; launch or held with the record's reason
    itdoc = _itdoc_rows(rows)
    assert {(r["model_key"], r["cell"]) for r in itdoc} == {(m, c) for m in (GEMMA, QWEN) for c in (ITDOC_CTL2, ITDOC_DIFF)}
    for r in itdoc:
        assert r["lane"] == G.LANE_ITDOC and r["object_kind"] == G.itdoc_kind_of(r["cell"]) and r["served_positions_expected"] == "all"
        assert r["staged_id"] == G.ITDOC_STAGED_PREFIX_BY_KIND[r["object_kind"]] + r["cell"] and r["bank_sidecar"] == G.ITDOC_SIDECAR_BY_KIND[r["object_kind"]]
        assert len(r["object_sha256"]) == 64 and set(r["object_sha256"]) <= set("0123456789abcdef") and r["tuple_name"] == C.g8k_tuple_name(r["model_key"], 8192)
        assert r["status"] in ("launch", "held") and (r["status"] == "launch") == (not r["held_reason"])
        f = V.itdoc_fields(r)
        assert f["qualification"] == r["qualification"] and f["qualification"] and f["qualification_pattern"] == r["qualification_pattern"] and f["qualification_pattern"]
        assert V.qualification_text(r) == f"{r['qualification']} {r['qualification_pattern']}"
    assert len({r["qualification_pattern"] for r in itdoc}) == 1 and all(len({r["qualification"] for r in itdoc if r["model_key"] == m}) == 1 for m in (GEMMA, QWEN))
    # the pin file is consulted for the ROSTER rows only (both the tree's and the frozen copy the generator names agree); never for an it-doc row
    for pins_path in (REPO / "configs" / "sdfbase_full22_pins.json", _frozen_itdoc_pins()):
        assert pins_path.is_file(), pins_path
        if pins_path.is_file():
            pins = json.loads(pins_path.read_text(encoding="utf-8"))
            assert V.matrix_pins_problem(rows, pins) is None
            assert all(pins[r["model_key"]][r["staged_id"]]["object_sha256"] == r["object_sha256"] for r in roster), pins_path
    bad = [dict(r) for r in rows]
    for r in bad:
        if V.is_itdoc_row(r) and r["status"] == "launch":
            r["object_sha256"] = "f" * 64                      # an it-doc row's pin is never looked up in the pin file
    assert V.matrix_pins_problem(bad, json.loads((REPO / "configs" / "sdfbase_full22_pins.json").read_text(encoding="utf-8"))) is None
    # --waves itdoc names exactly the document's evidence rows; the design block files the rest as held with reason + qualification + lane
    req = V.required_rows(mx["launch"], [G.WAVE_ITDOC])
    assert [r["index"] for r in req] == [r["index"] for r in rows if r["status"] == "launch"] and len(req) == doc["counts"]["launched"]
    assert A.eight_arm_mode(rows, [G.WAVE_ITDOC], M.GenRoots.single(REPO, PIN))
    design = A.design_status_block(rows, req, [G.WAVE_ITDOC])
    assert design["waves"] == [G.WAVE_ITDOC] and design["waves_in_matrix"] == [G.WAVE_ITDOC] and design["n_required"] == len(req)
    assert design["n_held"] == doc["counts"]["held"] and design["n_not_analysed"] == 0
    for r in rows:
        rec = design["status_by_row"][V.cell_key(r["model_key"], r["view"], r["cell"])]
        assert rec["status"] == ("required" if r["status"] == "launch" else "held") and rec["wave"] == G.WAVE_ITDOC
        assert rec["reason"] == (None if r["status"] == "launch" else r["held_reason"]) and rec["arm"] == M.CELL_KIND[r["cell"]]
        assert {k: rec[k] for k in ("object_source", *V.ITDOC_ROW_FIELDS)} == {"object_source": r.get("object_source"), **V.itdoc_fields(r)}
    for m in (GEMMA, QWEN):
        slots = design["arms_by_model_by_view"][m]
        assert set(slots) == {"it"} and slots["it"]["bare"] == "absent" and slots["it"]["lora_mix"] == "required"
        assert {slots["it"][k] for k in ("lora_ctl2", "object")} <= {"required", "held"} and slots["it"]["prompted_constitution"] == "required"
        assert design["arms_by_model"][m] == {k: "absent" for k in PANEL_KINDS}          # the panel view `main` has no row here
    # the DEFAULT matrix carries no it-doc row: its holds stay byte for byte, the rows 3 / 16 slots are the frozen fixture's
    default = V.load_matrix(REAL_MATRIX)
    assert not _itdoc_rows(default["rows"]) and V.required_rows(default["launch"], [G.WAVE_ITDOC]) == []
    for i in (3, 16):
        r, fx = default["rows"][i], fixture[i]
        assert (r["model_key"], r["view"], r["cell"], r["status"], r["staged_id"], r["object_sha256"]) == (fx["model_key"], fx["view"], fx["cell"], "held", fx["staged_id"], fx["object_sha256"])


def test_a_synthetic_schema_2_itdoc_document_loads_and_files_its_design():
    """The it-doc document's SHAPE, synthetic: schema 2 with a wave root and a trigger, every row wave `itdoc` under
    controls_itdoc/ with `thinking_mode` on — the two admission rows per model (one EVIDENCE, one held) and the roster
    constitution row; the reader loads it as a document, keeps the wave root, and files the design."""
    rows = []
    for key in (QWEN, GEMMA):
        roster = {**row(key, "it", CONST_LORA), "status": G.STATUS_LAUNCH, "held_reason": None, "wave": G.WAVE_ITDOC,
                  "out_root": f"controls_itdoc/{G.tag_of(key, 'it')}/{G.cell_slug(CONST_LORA)}/{G.MODE_PRIMARY}",
                  "rung_out_root": f"controls_itdoc/{G.tag_of(key, 'it')}/{G.cell_slug(CONST_LORA)}/{G.MODE_PRIMARY}__rung16384"}
        rows.append(roster)
        rows.append(G.itdoc_row(key, ITDOC_CTL2, index=0, status=G.STATUS_LAUNCH if key == QWEN else G.STATUS_HELD))
        rows.append(G.itdoc_row(key, ITDOC_DIFF, index=0, status=G.STATUS_HELD))
    for i, r in enumerate(rows):
        r.update(index=i, thinking_mode="on", trigger_threshold=G.TRIGGER_THRESHOLD)
    doc = {"schema": V.MATRIX_SCHEMA_2, "wave_root_rel": "exp/synthetic_itdoc_root", "trigger_threshold": G.TRIGGER_THRESHOLD, "wave": G.WAVE_ITDOC, "rows": rows}
    mx = V.load_matrix(doc)
    assert mx["schema"] == V.MATRIX_SCHEMA_2 and mx["wave_root_rel"] == "exp/synthetic_itdoc_root" and mx["counts"]["rows"] == 6
    assert (mx["counts"]["held"], mx["counts"]["launched"]) == (3, 3) and len(_itdoc_rows(mx["rows"])) == 4
    assert [r["index"] for r in V.required_rows(mx["launch"], [G.WAVE_ITDOC])] == [0, 1, 3]
    assert V.matrix_pins_problem(mx["rows"], G.pins_doc(ROWS)) is None            # the roster rows resolve in the pin file; the it-doc rows never
    design = A.design_status_block(mx["rows"], V.required_rows(mx["launch"], [G.WAVE_ITDOC]), [G.WAVE_ITDOC])
    assert design["arms_by_model_by_view"][QWEN]["it"] == {"bare": "absent", "prompted": "absent", "prompted_constitution": "absent", "lora_ctl2": "required",
                                                           "lora_mix": "required", "object": "held", "alphasteer": "absent", "arditi": "absent"}
    assert design["arms_by_model_by_view"][GEMMA]["it"]["lora_ctl2"] == "held" and design["n_held"] == 3
    rec = design["status_by_row"][V.cell_key(GEMMA, "it", ITDOC_CTL2)]
    assert rec["reason"] == G.ITDOC_HELD_REASON and rec["qualification"] == G.ITDOC_QUALIFICATION and rec["lane"] == G.LANE_ITDOC
    # a schema-2 document without its wave root, or with a row lacking `thinking_mode`, refuses like the Llama document does
    with pytest.raises(A.AnalysisError, match="wave_root_rel"):
        V.load_matrix({**doc, "wave_root_rel": None})
    with pytest.raises(A.AnalysisError, match="thinking_mode"):
        V.load_matrix({**doc, "rows": [{k: v for k, v in rows[0].items() if k != "thinking_mode"}, *rows[1:]]})

def test_itdoc_rows_admit_by_the_rows_pin_and_the_evidence_defects_refuse_by_name():
    rows = G.itdoc_matrix()
    mx = V.load_matrix(rows)
    # the real document's row / held counts (57 / 8); the synthetic steer stubs are `deferred`, so launched = 38 + the ext rows
    n_launch = 38 + len(G.ext_rows())
    assert (mx["counts"]["rows"], mx["counts"]["held"], mx["counts"]["launched"]) == (57, 8, n_launch) and len(_itdoc_rows(mx["rows"])) == 6
    # the in-place replacement keeps the slot; the PSR holds stay as they were
    assert (mx["rows"][3]["model_key"], mx["rows"][3]["cell"], mx["rows"][3]["object_source"]) == (GEMMA, ITDOC_MIX, "itdoc")
    assert mx["rows"][3]["superseded"]["object_sha256"] == row(GEMMA, "it", CONST_LORA)["object_sha256"] and mx["rows"][2]["held_reason"] == G.HELD_REASON
    # a HELD it-doc row with NO pin (not yet verified on disk) is a design-only row: admitted, never a leg, its reason filed
    unpinned = G.itdoc_matrix(pinned=False)
    mxu = V.load_matrix(unpinned)
    assert all(r["object_sha256"] is None for r in _itdoc_rows(mxu["rows"])) and mxu["counts"]["held"] == 8
    assert V.required_rows(mxu["launch"], ITDOC_WAVES) == V.required_rows(mx["launch"], ITDOC_WAVES) and V.required_rows(mx["launch"], [G.WAVE_ITDOC]) == []
    design = A.design_status_block(mxu["rows"], V.required_rows(mxu["launch"], ITDOC_WAVES), ITDOC_WAVES)
    rec = design["status_by_row"][V.cell_key(QWEN, "it", ITDOC_DIFF)]
    assert rec["status"] == "held" and rec["reason"] == G.ITDOC_HELD_REASON and rec["wave"] == G.WAVE_ITDOC and rec["object_source"] == "itdoc"
    assert rec["lane"] == G.LANE_ITDOC and rec["qualification"] == G.ITDOC_QUALIFICATION and rec["qualification_pattern"] == G.ITDOC_QUALIFICATION_PATTERN
    assert design["arms_by_model_by_view"][QWEN]["it"]["object"] == "held" and design["arms_by_model"][QWEN]["object"] == "required"
    # the pin file is never consulted for an it-doc row — held or EVIDENCE (the synthetic pin file carries no itdoc64 id)
    launch = G.itdoc_matrix(launch=[(QWEN, ITDOC_CTL2)])
    mxl = V.load_matrix(launch)
    assert mxl["counts"]["launched"] == n_launch + 1 and len(V.required_rows(mxl["launch"], [G.WAVE_ITDOC])) == 1
    pins = G.pins_doc(ROWS)
    assert V.matrix_pins_problem(mxl["rows"], pins) is None and G.row_for(launch, QWEN, "it", ITDOC_CTL2)["staged_id"] not in pins[QWEN]
    # an EVIDENCE it-doc row: 64-hex pin, undeclared, all-position, its lane, the model's ALL-POSITION tuple — each defect by name
    def in_place(rows_, key, view, cell):
        return next(r for r in rows_ if (r["model_key"], r["view"], r["cell"]) == (key, view, cell))      # G.row_for returns a COPY

    def refuses(edit, why):
        rows_ = G.itdoc_matrix(launch=[(QWEN, ITDOC_CTL2)])
        edit(in_place(rows_, QWEN, "it", ITDOC_CTL2))
        with pytest.raises(A.AnalysisError, match=why):
            V.load_matrix(rows_)
    refuses(lambda r: r.update(object_sha256=None), "needs a 64-hex object_sha256")
    refuses(lambda r: r.update(object_sha256="0018b32b"), "needs a 64-hex object_sha256")
    refuses(lambda r: r.update(tuple_name=C.g8k_steer_tuple_name(QWEN, 8192)), "never the W4 `_steer` tuple")
    refuses(lambda r: r.update(tuple_name=C.g8k_tuple_name(QWEN, 8192, declared=True)), "g8k_tuple_name")
    refuses(lambda r: r.update(lane=None), "names its lane")
    refuses(lambda r: r.update(lane=""), "names its lane")
    refuses(lambda r: r.update(declared=True, served_positions_expected="response"), "undeclared, all-position")
    refuses(lambda r: r.update(served_positions_expected="response"), "undeclared, all-position")
    refuses(lambda r: r.update(out_root="elsewhere/" + r["out_root"].split("/", 1)[1]), "layout")
    # a held it-doc row with no reason refuses like any held row; the steer tuple on a HELD row is not owed (design-only)
    refuses(lambda r: r.update(status="held", held_reason=None), "held_reason")
    held_steer = G.itdoc_matrix()
    in_place(held_steer, QWEN, "it", ITDOC_CTL2).update(tuple_name=C.g8k_steer_tuple_name(QWEN, 8192))
    assert V.load_matrix(held_steer)["counts"]["held"] == 8
    # the eight-arm mode is on under --waves whatever the subset named; the it-doc wave alone has nothing to collect (every row held)
    assert A.eight_arm_mode(mx["rows"], [G.WAVE_ITDOC], M.GenRoots.single(REPO, PIN)) and A.eight_arm_mode(mx["rows"], None, M.GenRoots.single(REPO, PIN))
    with pytest.raises(A.AnalysisError, match="nothing to collect"):
        analyze(REPO, matrix=rows, waves=[G.WAVE_ITDOC])


def test_an_itdoc_leg_end_to_end_carries_the_qualification_verbatim_on_every_reading(c4_store, tmp_path, monkeypatch):
    """ONE complete it-doc leg (qwen35_4b `it`, the LoRA ctl2 arm; object_source itdoc, a 64-hex pin on the row, the row's
    qualification) beside the C4 fixture, analysed under ``--waves published,ext,itdoc`` through ``main()``: the leg is admitted
    by the ROW's pin (identity, row provenance), its cell record, contrast, contrast summary, provenance row and the model's
    captions carry the qualification VERBATIM (the row's text, never retyped), the other five it-doc rows are filed as held with
    their reason + qualification + lane, the it view's arm slots are visible beside the main view's, and nothing of the it view
    enters the pooled main-view estimands."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = G.itdoc_matrix(launch=[(QWEN, ITDOC_CTL2)])
    leg_row = G.row_for(rows, QWEN, "it", ITDOC_CTL2)
    b2 = tmp_path / "pull_ext_plus_itdoc"
    shutil.copytree(c4_store.b, b2)
    G.build_itdoc_leg(b2, leg_row, tree_sha=T2)
    files = {"matrix": G.write_matrix(tmp_path / "matrix.json", rows), "ids": G.write_json(tmp_path / "ids.json", G.IDS),
             "pins": G.write_json(tmp_path / "pins.json", G.pins_doc(ROWS)), "sampling-rows": G.write_json(tmp_path / "rows.json", G.SAMPLING_ROWS),
             "render-facts": G.write_json(tmp_path / "facts.json", G.RENDER_FACTS), "overlap": G.write_json(tmp_path / "overlap.json", c4_store.overlap)}
    monkeypatch.setitem(sys.modules, "wandb", None)
    rc = A.main(["--gen-root", f"{c4_store.a}={PIN}", "--gen-root", f"{b2}={T2}", "--sha", PIN, "--also-sha", T2, "--tree-diff-allow", "scripts/*",
                 "--tree-set-reason", TREE_SET["why"], "--waves", ",".join(ITDOC_WAVES), *[f"--{k}={v}" for k, v in files.items()],
                 "--bootstrap", str(B_FAST), "--no-eval-logs", "--no-figs", "--wandb-mode", "disabled", "--no-mirror", "--analysis-sha", G.FULL_SHA,
                 "--repo-root", str(REPO), "--out", str(tmp_path / "report.json")])
    assert rc == 0, rc
    rep = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert rep["gates"]["failed"] == [] and A.production_refusals(rep) == [] and rep["unusable_legs"] == {}
    assert rep["design"]["waves"] == ITDOC_WAVES and set(rep["design"]["waves_in_matrix"]) == {"published", G.WAVE_EXT, G.WAVE_STEER, G.WAVE_ITDOC}
    assert rep["matrix"]["counts"] == {**V.load_matrix(rows)["counts"]} and rep["design"]["n_launch_rows"] == 38 + len(G.ext_rows()) + 1
    pin, ck, lk = leg_row["object_sha256"], V.cell_key(QWEN, "it", ITDOC_CTL2), V.leg_key(QWEN, "it", ITDOC_CTL2, G.CAP_BASE)
    carriage = {"object_source": "itdoc", "lane": G.LANE_ITDOC, "qualification": G.ITDOC_QUALIFICATION, "qualification_pattern": G.ITDOC_QUALIFICATION_PATTERN}
    none_ = {"object_source": None, "lane": None, "qualification": None, "qualification_pattern": None}
    # the LEG: admitted by the row's pin — the manifest's object digest and the row provenance (a --bank-view it run) equal it
    leg = rep["legs"][lk]
    assert leg["refused"] is None and leg["missing"] is None and leg["wave"] == G.WAVE_ITDOC and leg["identity_validated"]
    assert leg["identity"]["object_artifact_sha256"] == pin and leg["identity"]["producer_tree_sha"] == T2 and leg["producer"]["pin"] == T2
    assert leg["row_provenance"] == {"present": True, "object_artifact_sha256": pin, "route_kind": "lens", "view": "it", "producer_tree_sha": T2}
    assert {k: leg[k] for k in carriage} == carriage and {k: rep["legs"][V.leg_key(QWEN, "main", CONST_LORA, G.CAP_BASE)][k] for k in none_} == none_
    # the CELL record, the CONTRAST record and its SUMMARY, the PROVENANCE row: the qualification verbatim, the lane, the source
    cell = rep["cells"][ck]
    assert {k: cell[k] for k in carriage} == carriage and cell["wave"] == G.WAVE_ITDOC and cell["readings"]["merged"]["acc_stock"] is not None
    c = contrast_of(rep, QWEN, "it", ITDOC_CTL2)
    assert c["computable"] and {k: c[k] for k in carriage} == carriage and c["name"] == f"{QWEN}/it: {ITDOC_CTL2} - none"
    cs = next(x for x in rep["contrast_summary"] if x["model_key"] == QWEN and x["view"] == "it" and x["cell"] == ITDOC_CTL2)
    assert {k: cs[k] for k in carriage} == carriage and cs["merged"]["word"] is not None
    assert {k: contrast_of(rep, QWEN, "main", CONST_LORA)[k] for k in none_} == none_
    assert all({k: x[k] for k in none_} == none_ for x in rep["noise_floor_summary"])
    prow = rep["provenance"]["cells"][lk]
    assert {k: prow[k] for k in carriage} == carriage and prow["object_artifact_sha256"] == pin and prow["pin"] == T2 and prow["wave"] == G.WAVE_ITDOC
    assert prow["arm"] == "lora_ctl2" and prow["view"] == "it" and prow["row_provenance"]["object_artifact_sha256"] == pin
    assert {k: rep["provenance"]["cells"][V.leg_key(QWEN, "main", CONST_LORA, G.CAP_BASE)][k] for k in none_} == none_
    assert {k: next(r for r in rep["controls"] if r["leg"] == lk)["provenance"][k] for k in carriage} == carriage
    # the DESIGN block: the leg required; the other five it-doc rows HELD with reason + qualification + lane; the it view's slots
    sbr = rep["design"]["status_by_row"]
    assert sbr[ck] == {"status": "required", "wave": G.WAVE_ITDOC, "reason": None, "model_key": QWEN, "view": "it", "cell": ITDOC_CTL2, "arm": "lora_ctl2", **carriage}
    held_keys = [k for k, v in sbr.items() if v["object_source"] == "itdoc" and v["status"] == "held"]
    # held = the PSR pair (published) + the five it-doc rows that are not the analysed leg
    assert len(held_keys) == 5 and rep["design"]["n_held"] == 7 == len(rep["held"]) == rep["design"]["n_held_rows"]
    for k in held_keys:
        assert sbr[k]["reason"] == G.ITDOC_HELD_REASON and sbr[k]["wave"] == G.WAVE_ITDOC and {f: sbr[k][f] for f in carriage} == carriage
    assert sbr[V.cell_key(GEMMA, "it", CONST_PSR)]["reason"] == G.HELD_REASON and {f: sbr[V.cell_key(GEMMA, "it", CONST_PSR)][f] for f in none_} == none_
    abv = rep["design"]["arms_by_model_by_view"]
    assert abv[QWEN]["it"] == {"bare": "required", "prompted": "required", "prompted_constitution": "absent", "lora_ctl2": "required", "lora_mix": "held",
                               "object": "held", "alphasteer": "absent", "arditi": "absent"}
    assert abv[GEMMA]["it"]["lora_ctl2"] == "held" and abv[QWEN]["main"] == rep["design"]["arms_by_model"][QWEN] and set(abv[NEMO]) == set(abv[GLM]) == {"main"}
    assert rep["design"]["arms_by_model"][QWEN]["lora_ctl2"] == "required" and rep["design"]["eightarm_view"] == "main"
    # the HELD list and the findings
    held = [h for h in rep["held"] if h["object_source"] == "itdoc"]
    assert len(held) == 5 and all(h["reason"] == G.ITDOC_HELD_REASON and h["wave"] == G.WAVE_ITDOC and {f: h[f] for f in carriage} == carriage for h in held)
    assert all({f: h[f] for f in none_} == none_ and h["wave"] == "published" for h in rep["held"] if h["object_source"] != "itdoc")
    assert sum(1 for t in rep["findings"] if "HELD — " in t and f"qualification: {G.ITDOC_QUALIFICATION} {G.ITDOC_QUALIFICATION_PATTERN}" in t) == 5
    assert not any("qualification:" in t for t in rep["findings"] if f"{GEMMA}/it/{CONST_PSR}: HELD" in t)
    # the it view never enters the pooled main-view estimands
    ea = rep["eightarm"]["pooled_by_arm_merged"]["lora_ctl2"]
    assert ea["status"] == A.POOLED_AVAILABLE and ea["primary"]["by_model"][QWEN] == rep["cells"][V.cell_key(QWEN, "main", CTL_LORA)]["readings"]["merged"]["acc_stock"]
    # the CAPTIONS: the model whose SHOWN cells include the it-doc leg carries the sentence (qualification + pattern), prefixed
    # with the arm(s); the eight-arm captions carry it too; a model without an it-doc leg does not; hidden -> not shown -> no note
    note = f"it-doc re-fit ({QWEN} it: LoRA ctl2): {G.ITDOC_QUALIFICATION} {G.ITDOC_QUALIFICATION_PATTERN}"
    caps = A.figure_captions(rep)
    assert caps[f"final_g8k_contrasts_{QWEN}"].endswith(" " + note) and caps[A.EIGHTARM_STEM_ACCURACY].endswith(" " + note) and caps[A.EIGHTARM_STEM_CONTRASTS].endswith(" " + note)
    assert G.ITDOC_QUALIFICATION not in caps[f"final_g8k_contrasts_{GEMMA}"] and G.ITDOC_QUALIFICATION not in caps["final_g8k_accuracy_by_cell"]
    assert caps[f"final_g8k_contrasts_{QWEN}"].count(G.ITDOC_QUALIFICATION) == 1
    hidden = A.figure_captions({**rep, "figures": {A.FIGURES_EXCLUDED_ARMS_KEY: [ITDOC_CTL2]}})
    assert G.ITDOC_QUALIFICATION not in hidden[f"final_g8k_contrasts_{QWEN}"] and G.ITDOC_QUALIFICATION not in hidden[A.EIGHTARM_STEM_ACCURACY]
    assert G.ITDOC_QUALIFICATION not in A.figure_captions({**rep, "figures": {A.FIGURES_EXCLUDED_KEY: [QWEN]}}).get(A.EIGHTARM_STEM_ACCURACY, "")
    # the contrasts forest names the it-doc leg and the held it-doc rows by THEIR reason (the PSR hold keeps its own label)
    fig, _nums = A._fig_contrasts_model(rep, QWEN)
    ax = fig.axes[0]
    ylabels = [t.get_text() for t in ax.get_yticklabels()]
    texts = [t.get_text() for t in ax.texts]
    assert f"it: {A._short(ITDOC_CTL2)}" in ylabels and f"it: {A._short(ITDOC_DIFF)}" in ylabels
    assert texts.count(f"HELD — {G.ITDOC_HELD_REASON[:64]}") == 2 and texts.count("HELD (D8: re-fit at effective batch 64)") == 1
    plt.close(fig)
    # the figure set renders with the note in the numbers sidecar's captions
    figs = A.make_figures(rep, tmp_path / "figs", exclude_arms=list(APPENDIX_ARMS))
    sidecar = json.loads(Path(figs["numbers"]).read_text(encoding="utf-8"))
    assert sidecar["captions"][f"final_g8k_contrasts_{QWEN}"].endswith(" " + note) and A.EIGHTARM_STEM_ACCURACY in figs["written"]
    # a report whose it-doc leg is HELD instead (the design-only case) carries the qualification in the design block only
    rep2 = analyze_c4(c4_store, b=b2, matrix=G.itdoc_matrix(), waves=ITDOC_WAVES)
    assert rep2["gates"]["failed"] == [] and lk not in rep2["legs"] and rep2["design"]["status_by_row"][ck]["status"] == "held"
    assert rep2["design"]["status_by_row"][ck]["qualification"] == G.ITDOC_QUALIFICATION and G.ITDOC_QUALIFICATION not in A.figure_captions(rep2)[f"final_g8k_contrasts_{QWEN}"]
    assert rep2["design"]["n_launch_rows"] == 38 + len(G.ext_rows()) and len([h for h in rep2["held"] if h["object_source"] == "itdoc"]) == 6


def test_a_contrast_refused_exactly_as_the_row_declares_is_refused_by_design_and_the_export_is_a_production_result(tmp_path):
    """The production GSM8K export: A0-A10 all pass, the native arm's contrast against the lens `none` anchor is filed
    `computable: false` with the O106 text (correct — never computed), and the export dies on "contrast(s) did not run" with no
    paper rows. The row now DECLARES that refusal (`expected_contrast_refusal`, the reader's own CROSS_ENGINE_RULING): the contrast
    is filed refused BY DESIGN — its text and `computable: false` untouched — `_not_run` no longer names it, the verdict and the
    production boundary read it as a filed outcome, the design counts it, and the cell's rows reach the paper rows with the arm's
    own numbers (exact binomial CIs) and the refusal named. Twins: the same declaration on a contrast refused for ANOTHER reason
    (no named code) still blocks, naming both; a declared contrast that COMPUTES is refused ("expected a refusal, got a contrast");
    a code the reader does not derive is a named problem; the UNDECLARED refusal blocks with today's message, byte for byte."""
    rows = small_matrix((QWEN, "main", NONE), (QWEN, "main", CONST_LORA), (QWEN, "main", CONST_PROMPT))
    pub, ext = [r for r in rows if r.get("wave") is None], [r for r in rows if r.get("wave")]
    store = build_cross_tree(tmp_path / "native", published_rows=pub, ext_rows=ext, esc_none=False, esc_ext=False,
                             pub_overrides={(QWEN, "main", CONST_LORA): {"serving_engine": C.SERVING_ENGINE_LORA_NATIVE}})
    name = f"{QWEN}/main: {CONST_LORA} - none"
    # UNDECLARED (today): the cross-engine refusal blocks the export, byte for byte
    r0 = analyze(roots_of(store), matrix=rows, overlap=None, tree_set=TREE_SET, waves=WAVES)
    c0 = contrast_of(r0, QWEN, "main", CONST_LORA)
    assert not c0["computable"] and c0["compatibility"]["cross_engine"] is True and c0["reason"] == c0["compatibility"]["cross_engine_refusal"]
    assert A.contrast_refusal_code(c0) == M.CROSS_ENGINE_RULING == "O106" and A.CONTRAST_REFUSAL_CODES == (M.CROSS_ENGINE_RULING,)
    assert c0[A.CONTRAST_REFUSAL_CODE_KEY] == "O106" and A.CONTRAST_REFUSED_BY_DESIGN_KEY not in c0 and A.ROW_EXPECTED_CONTRAST_REFUSAL_KEY not in c0
    assert r0["verdict"]["word"] is None and r0["verdict"]["reason"] == f"contrast(s) did not run: [{name!r}]"
    assert f"contrast(s) did not run: [{name!r}]" in A.production_refusals(r0) and A.contrast_declaration_problems(r0["contrasts"]) == []
    assert r0["design"]["expected_contrast_refusals"] == {} and r0["design"]["n_expected_contrast_refusals"] == 0
    # DECLARED: the same refusal is a filed outcome
    declared = [{**r, A.ROW_EXPECTED_CONTRAST_REFUSAL_KEY: M.CROSS_ENGINE_RULING} if (r["model_key"], r["view"], r["cell"]) == (QWEN, "main", CONST_LORA) else r
                for r in rows]
    r1 = analyze(roots_of(store), matrix=declared, overlap=None, tree_set=TREE_SET, waves=WAVES)
    c1 = contrast_of(r1, QWEN, "main", CONST_LORA)
    assert not c1["computable"] and c1["reason"] == c0["reason"] and c1["readings"] == {}
    assert c1[A.ROW_EXPECTED_CONTRAST_REFUSAL_KEY] == "O106" and c1[A.CONTRAST_REFUSAL_CODE_KEY] == "O106" and c1[A.CONTRAST_REFUSED_BY_DESIGN_KEY] is True
    assert [c["name"] for c in A._not_run(r1["contrasts"])] == [] and A.contrast_declaration_problems(r1["contrasts"]) == []
    assert "did not run" not in " ".join(A.production_refusals(r1)) and r1["verdict"]["reason"] != f"contrast(s) did not run: [{name!r}]"
    assert r1["design"]["expected_contrast_refusals"] == {name: "O106"} and r1["design"]["n_expected_contrast_refusals"] == 1
    assert r1["gates"]["A8_two_branch"]["contrasts"][name] == {"computable": False, "reason": c1["reason"], A.CONTRAST_REFUSED_BY_DESIGN_KEY: True,
                                                             A.ROW_EXPECTED_CONTRAST_REFUSAL_KEY: "O106"}
    assert any(f"{name}: refused by design (O106)" in f and "the arm's numbers are reported as the arm's numbers" in f for f in r1["findings"])
    assert not any(f.startswith(f"{name}: no estimand") for f in r1["findings"])
    summary = next(c for c in r1["contrast_summary"] if c["name"] == name) if isinstance(r1.get("contrast_summary"), list) else None
    if summary is not None:
        assert summary[A.CONTRAST_REFUSED_BY_DESIGN_KEY] is True and summary[A.ROW_EXPECTED_CONTRAST_REFUSAL_KEY] == "O106"
    # the lens-vs-lens contrast beside it is untouched, its record byte-identical to the undeclared run's
    assert contrast_of(r1, QWEN, "main", CONST_PROMPT) == contrast_of(r0, QWEN, "main", CONST_PROMPT)
    # the paper rows: the cell's rows carry the ARM's own numbers with exact binomial CIs and name the refusal
    prows, _skipped = A.paper_rows_of_report(r1, report_path="synthetic/report.json")
    cell_rows = [x for x in prows if x["cell"] == CONST_LORA and x["model_key"] == QWEN]
    assert cell_rows and all(x["ci_method"] == M.PRW.CI_EXACT for x in cell_rows)
    assert all(f"refused by design ({M.CROSS_ENGINE_RULING})" in x["source"] and "the arm's numbers are reported as the arm's numbers" in x["source"] for x in cell_rows)
    base_row = next(x for x in cell_rows if x["leg"] == f"r{G.CAP_BASE}")
    leg = leg_of(r1, QWEN, "main", CONST_LORA)
    assert base_row["n_paired"] == leg["summary"]["n"] and base_row["at_cap_rate"] == leg["summary"]["at_cap_rate"] and base_row["n_correct"] is not None
    none_rows = [x for x in prows if x["cell"] == NONE and x["model_key"] == QWEN]
    assert none_rows and all(x["ci_method"] == M.PRW.CI_BOOTSTRAP for x in none_rows)      # the anchor sits in the computable lens contrast
    # twin (1): declared, but refused for ANOTHER reason (a dtype difference — no named code): still blocks, naming both
    def set_dtype(args):
        args["dtype"] = "float16"
    store2 = build_cross_tree(tmp_path / "dtype", published_rows=pub, ext_rows=ext, esc_none=False, esc_ext=False, ext_common={"launch_edit": set_dtype})
    declared_prompt = [{**r, A.ROW_EXPECTED_CONTRAST_REFUSAL_KEY: "O106"} if (r["model_key"], r["view"], r["cell"]) == (QWEN, "main", CONST_PROMPT) else r for r in rows]
    r2 = analyze(roots_of(store2), matrix=declared_prompt, overlap=None, tree_set=TREE_SET, waves=WAVES)
    c2 = contrast_of(r2, QWEN, "main", CONST_PROMPT)
    assert not c2["computable"] and "launch.dtype" in c2["reason"] and A.CONTRAST_REFUSAL_CODE_KEY not in c2 and A.CONTRAST_REFUSED_BY_DESIGN_KEY not in c2
    probs = A.contrast_declaration_problems(r2["contrasts"])
    assert probs == [f"{c2['name']}: declared an expected refusal (O106) but refused with no named code — {c2['reason']}"]
    refusals = A.production_refusals(r2)
    assert any(x.startswith("contrast(s) did not run:") and c2["name"] in x for x in refusals) and any("expected contrast refusal(s) not met" in x for x in refusals)
    # twin (2): declared on a contrast that COMPUTES (the lens-vs-lens prompted contrast of the native store): refused by name
    r3 = analyze(roots_of(store), matrix=[{**r, A.ROW_EXPECTED_CONTRAST_REFUSAL_KEY: "O106"} if (r["model_key"], r["view"], r["cell"]) == (QWEN, "main", CONST_PROMPT) else r for r in rows],
                 overlap=None, tree_set=TREE_SET, waves=WAVES)
    c3 = contrast_of(r3, QWEN, "main", CONST_PROMPT)
    assert c3["computable"] and c3[A.ROW_EXPECTED_CONTRAST_REFUSAL_KEY] == "O106" and A.CONTRAST_REFUSED_BY_DESIGN_KEY not in c3
    assert A.contrast_declaration_problems(r3["contrasts"]) == [f"{c3['name']}: expected a refusal (O106), got a contrast — the design declares this contrast "
                                                                "refuses by name and it computed; retire the declaration or name the pairing"]
    assert any("expected a refusal (O106), got a contrast" in x for x in A.production_refusals(r3))
    # twin (3): a code the reader does not derive is a named problem, never a match
    r4 = analyze(roots_of(store), matrix=[{**r, A.ROW_EXPECTED_CONTRAST_REFUSAL_KEY: "O999"} if (r["model_key"], r["view"], r["cell"]) == (QWEN, "main", CONST_LORA) else r for r in rows],
                 overlap=None, tree_set=TREE_SET, waves=WAVES)
    c4 = contrast_of(r4, QWEN, "main", CONST_LORA)
    assert c4[A.EXPECTED_CONTRAST_REFUSAL_PROBLEM_KEY].startswith("expected_contrast_refusal 'O999' is not a refusal code this reader derives") and A.CONTRAST_REFUSED_BY_DESIGN_KEY not in c4
    assert any("is not a refusal code this reader derives" in x for x in A.production_refusals(r4))
