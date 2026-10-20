"""The bcb-bfcl-200 analysis on the ROSTER-DRIVEN wave fixtures
(``scripts/bcb_bfcl_200_analysis.py``), the earlier analysis's checks
carried over to the N-cell world: identity refusals (two instruments
never pair; a tree / sampling / thinking / intervention that moves; the serving
identity's backend + mode one per run, its path per cell; a required field
absent), the pinned items exactly, the breakdowns (BFCL categories from the
pin's population, BCB families multi-label from the pin × the family map),
the diagnostics apart, the truncation ladder read ONLY from `ladder merge`
across THREE cells with PER-CELL certificates at the wave's common rung bound by
the wave record `rung_admission.json` (a foreign / stale merge, edited rows,
another selection, a certificate at another rung, a record that disagrees all
refuse; check 4 NOT MET when a fired leg is not merged; a MISSING certificate or
record keeps the merged tables recorded but NOT final — null paper measurements),
the ONE final-reading selector (a merge that flips the conclusion from "cost
detected" at rung 1 to "no cost" merged is what the summary, headline, W&B,
figures and paper rows all read, rung 1 only under a diagnostic label), the
superseded certificates listed and never read, same-wave provenance with a
resumed cell's second attempt accepted, sandbox lifetimes ending at the FIRST
observed termination, the wall clock, the GPU cost by wave from the meter's
ledger, every figure rendering and closing.

The fixture builders live in ``tests/test_bcb_bfcl_200_widen_analysis.py``
(the sibling-test convention). The ladder-merge world is PROTOCOL mode
on the checked-in roster: Slice B's `PROJECTED_IDENTITY_KEYS` now projects the
hashed `sampling_by_task` onto the merged tasks, so a one-leg rung-2 store
merges; the route digests are populated (the standup attested them) because
`tinycap_ladder.bind_certificate_to_store` binds a certificate to
`route.object_artifact_sha256`."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

import test_bcb_bfcl_200_widen_analysis as W  # noqa: E402
import test_tinycap_ladder as F  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.capeval.tinycap import rung_probe as RP  # noqa: E402

An, L, M = W.An, W.L, W.M
BCB, BFCL = F.BCB, F.BFCL
NONE, LORA, PSR = W.NONE, W.DIFF_LORA, W.DIFF_PSR
CELLS = (NONE, LORA, PSR)
WAVE, SUFFIX, RUN = "gemma4e4b_main", W.SUFFIX_MAIN, W.RUN_MAIN
DEFAULT_CAPPED = {(LORA, BCB): set(F.CAPPED_BCB), (NONE, BCB): {F.CAPPED_BCB[0]}, (NONE, BFCL): set(F.BFCL_IDS[0:2]), (PSR, BFCL): set(F.BFCL_IDS[2:4])}


def make_world(tmp_path, *, capped=None, values=None):
    repo = F.make_pins(tmp_path / "repo")
    roster = W.write_roster(tmp_path / "lane" / "roster.yaml", W.fixture_roster())        # PSR reinstated: two fitted cells for the ladder tests
    assert W.commit_roster(roster) == W.TREE          # the stores' producer commit: the analysis reads the roster AT it
    doc = An.load_roster(roster)
    # the route digests stay POPULATED here (the standup attested them): `tinycap_ladder.bind_certificate_to_store`
    # binds the certificate to `route.object_artifact_sha256`, so a fired-leg merge needs them — see the report
    rung1 = W.make_wave(tmp_path / SUFFIX, doc, WAVE, roster_path=roster, run_id=RUN, null_route_digest=False, combined=False,
                        capped=(DEFAULT_CAPPED if capped is None else capped), values=values)
    ledger = W.ledger_with(tmp_path / "gpu_ledger.jsonl", {"kind": "decision", "ceiling_h": 25.0, "note": "spec envelope"},
                           *({"kind": "forecast", "wave": WAVE, "suffix": SUFFIX, "unit": c, "stage_kind": "wave", "hours": h} for c, h in ((NONE, 0.18), (LORA, 0.4), (PSR, 0.4))),
                           {"kind": "actual", "wave": WAVE, "suffix": SUFFIX, "unit": NONE, "stage_kind": "wave", "hours": 0.15})
    return {"repo": repo, "roster": roster, "doc": doc, "rung1": rung1, "ledger": ledger, "tmp": tmp_path}


@pytest.fixture()
def world(tmp_path):
    return make_world(tmp_path)


def _analyze(world, **kw):
    kw.setdefault("sandbox_state_root", world["tmp"] / "sandbox-state")
    return An.analyze(world["rung1"], roster_path=world["roster"], wave_name=WAVE, repo_root=world["repo"], ledger=world["ledger"], **kw)


def _edit_manifest(world, cell, task, fn):
    W._edit_manifest(world["rung1"], cell, task, fn)


def _stamp_engine(m: dict, engine: str) -> None:
    """Re-stamp a fixture manifest to the ENGINE's shape AS THE PRODUCER WRITES IT (
    `read_store_serving_engine` now holds a stamped record's stamps to its serving identity, so a lens-shaped store with
    only a top-level `lora_native` stamp — a relabelled record no driver writes — refuses instead of reading native):
    under `lora_native` every spelling the fixture carries (top-level, the route block, the hashed `tinycap.served`) is
    stamped, both reduction modes are `not_applicable`, and a steered `+lens+` route identity becomes `+lora+`; under
    `lens` the producer's iff-native rule removes every stamp and restores the lens arithmetic and identity."""
    route, served = m[C.MANIFEST_KEY_ROUTE], m[C.MANIFEST_KEY_TINYCAP]["served"]
    if engine == NATIVE:
        m[C.SERVED_ENGINE_FIELD] = route[C.SERVED_ENGINE_FIELD] = served[C.SERVED_ENGINE_FIELD] = NATIVE
        route[C.REDUCTION_MODE_FIELD] = served[C.REDUCTION_MODE_FIELD] = C.REDUCTION_MODE_NOT_APPLICABLE
        route["serving_identity"] = route["serving_identity"].replace("+lens+", "+lora+")
    else:
        for node in (m, route, served):
            node.pop(C.SERVED_ENGINE_FIELD, None)
        route[C.REDUCTION_MODE_FIELD] = served[C.REDUCTION_MODE_FIELD] = "stacked"
        route["serving_identity"] = route["serving_identity"].replace("+lora+", "+lens+")


def _expected_k(cell, task, capped):
    return int(sum(r.value for r in F.rung1_results(cell, task, capped=capped, values=W.wave_values(cell))))


# --------------------------------------------------------------------------- #
# headline, identity, pairing                                                  #
# --------------------------------------------------------------------------- #


def test_headline_wilson_and_paired_differences_per_steered_cell(world):
    a = _analyze(world)
    assert a["cells"] == list(CELLS) and a["steered"] == [LORA, PSR] and a["tasks"] == [BCB, BFCL] and a["protocol"]["mode"] == "protocol"
    for task in (BCB, BFCL):
        leg = a["legs"][task]
        for cell in CELLS:
            r = leg["cells"][cell]
            assert r["n"] == 200 and r["rate"] == r["k"] / 200 and (r["wilson_low"], r["wilson_high"]) == An.wilson_interval(r["k"], 200)
        ka = leg["cells"][NONE]["k"]
        for cell in (LORA, PSR):
            p = leg["paired"][cell]
            assert p["n_items"] == 200 and abs(p["delta"] - (leg["cells"][cell]["k"] - ka) / 200) < 1e-12 and p["ci_low"] <= p["delta"] <= p["ci_high"]
            assert p["mcnemar_exact_p"] == An.mcnemar_exact_p(p["b_pass_to_fail"], p["c_fail_to_pass"]) and p["provisional"] is False
    assert a["legs"][BCB]["cells"][LORA]["k"] == _expected_k(LORA, BCB, F.CAPPED_BCB)
    assert a["legs"][BCB]["fired"] is True and a["legs"][BFCL]["fired"] is False
    assert a["identity"]["model_id"] == F.HF and a["identity"]["tree_sha"] == W.TREE and a["identity"]["caps"] == {BCB: 4096, BFCL: 1024}
    assert a["identity"]["object_sha256"][LORA] == W.SHA[LORA] and a["identity"]["object_sha256_source"][LORA].startswith("route.")
    assert a["identity"]["object_sha256"][NONE] is None
    for task, chk in a["identity"]["checks"].items():
        assert chk["same_instrument"] and chk["off_instrument"] == {} and chk["mode"] == "protocol"
        assert "sandbox" in chk["optional_equal"] and any(k.endswith(":route") for k in chk["differs"])
    assert a["identity"]["checks"][BCB]["task_identity_paths"] == ["tinycap.bcb_item_set", "tinycap.bcb_sandbox"]
    assert a["identity"]["checks"][BCB]["intended_instrument"][C.IDENTITY_KEY_SANDBOX] == C.bcb_sandbox_identity()
    assert a["generations"][LORA]["object_generation"] == "final" and a["generations"][PSR]["provisional"] is False
    assert a["fit_layers"][LORA]["fit_layers"] == "0-41" and a["fit_layers"][PSR]["fit_layer_count"] == 42
    assert "T=0.0" in a["protocol"]["decode_by_task"][BCB] and "T=0.001" in a["protocol"]["decode_by_task"][BFCL]


def test_identity_refuses_a_verifier_timeout_that_moved_and_stores_off_the_instrument(world):
    _edit_manifest(world, LORA, BCB, lambda m: m[C.MANIFEST_KEY_TINYCAP][C.IDENTITY_KEY_SANDBOX].update(verify_timeout_s=60))
    with pytest.raises(An.AnalysisError, match="not the same instrument on inspect/bigcodebench_200.*bcb_sandbox"):
        _analyze(world)
    for cell in CELLS:
        _edit_manifest(world, cell, BCB, lambda m: m[C.MANIFEST_KEY_TINYCAP][C.IDENTITY_KEY_SANDBOX].update(verify_timeout_s=60))
    with pytest.raises(An.AnalysisError, match="not this lane's intended instrument.*bcb_sandbox"):
        _analyze(world)


def test_identity_binds_the_item_set_blocks_to_the_pins(world):
    other = dict(C.item_set_identity(F.BFCL_PIN), ids_sha256="0" * 64)
    for cell in CELLS:
        _edit_manifest(world, cell, BFCL, lambda m: m[C.MANIFEST_KEY_TINYCAP].__setitem__(C.IDENTITY_KEY_BFCL_ITEM_SET, other))
    with pytest.raises(An.AnalysisError, match="intended instrument.*bfcl_item_set"):
        _analyze(world)


def test_identity_holds_every_store_to_the_rosters_served_model(world):
    a = _analyze(world)
    assert a["identity"]["common"]["model_row"]["hf"] == F.HF and a["identity"]["common"]["one_experiment"] is True
    for cell in CELLS:
        for task in (BCB, BFCL):
            _edit_manifest(world, cell, task, lambda m: m.__setitem__("model_revision", "f" * 40))
    with pytest.raises(An.AnalysisError, match="not the wave's served model"):
        _analyze(world)


@pytest.mark.parametrize("edit,needle", [
    (lambda m: m[C.MANIFEST_KEY_PRODUCER].__setitem__("tree_sha", "e" * 40), "producer.tree_sha differs across stores"),
    (lambda m: m[C.MANIFEST_KEY_SAMPLING].__setitem__("temperature", 0.7), "sampling.scalars.temperature differs|sampling.*differs across stores"),
    (lambda m: m[C.MANIFEST_KEY_TINYCAP].__setitem__("samples", 2), "tinycap.*samples differs across stores"),
])
def test_identity_refuses_legs_from_different_trees_or_sampling(world, edit, needle):
    for cell in CELLS:
        _edit_manifest(world, cell, BFCL, edit)
    with pytest.raises(An.AnalysisError, match=needle):
        _analyze(world)


def test_identity_refuses_an_intervention_that_differs_across_a_cells_stores(world):
    _edit_manifest(world, PSR, BFCL, lambda m: m.__setitem__("intervention_hash", "d" * 64))
    with pytest.raises(An.AnalysisError, match="intervention_hash differs across a cell's stores"):
        _analyze(world)


def test_serving_identity_backend_mode_per_run_and_path_per_cell(world):
    a = _analyze(world)
    assert a["identity"]["common"]["serving_path_by_cell"] == {NONE: "passthrough", LORA: "lens", PSR: "lens"}
    _edit_manifest(world, NONE, BFCL, lambda m: m[C.MANIFEST_KEY_ROUTE].__setitem__("serving_identity", "proxy_vllm+passthrough+exact_replay"))
    with pytest.raises(An.AnalysisError, match=r"route\.serving_identity\.backend\+mode differs across stores"):
        _analyze(world)


def test_serving_identity_path_must_agree_across_a_cells_stores_and_malformed_refuses(world):
    _edit_manifest(world, LORA, BFCL, lambda m: m[C.MANIFEST_KEY_ROUTE].__setitem__("serving_identity", "proxy_vllm+passthrough+interleaved"))
    with pytest.raises(An.AnalysisError, match=r"route\.serving_identity\.path differs across a cell's stores"):
        _analyze(world)
    _edit_manifest(world, LORA, BFCL, lambda m: m[C.MANIFEST_KEY_ROUTE].__setitem__("serving_identity", {"model_id": "x"}))
    with pytest.raises(An.AnalysisError, match=r"route\.serving_identity\[.*\] differs across stores"):
        _analyze(world)


def test_identity_refuses_two_instruments_before_pairing_and_an_absent_required_field(world):
    _edit_manifest(world, PSR, BFCL, lambda m: m[C.MANIFEST_KEY_TINYCAP].__setitem__("thinking", "on"))
    with pytest.raises(An.AnalysisError, match="thinking"):
        _analyze(world)
    for cell in CELLS:
        for task in (BCB, BFCL):
            _edit_manifest(world, cell, task, lambda m: m[C.MANIFEST_KEY_TINYCAP].__setitem__("thinking", "off"))
            _edit_manifest(world, cell, task, lambda m: m.pop(C.MANIFEST_KEY_PRODUCER))
    with pytest.raises(An.AnalysisError, match="producer.tree_sha"):
        _analyze(world)


def test_same_wave_provenance_accepts_a_resumed_attempt_and_refuses_a_missing_pointer(world):
    """A resumed BCB worker mints a new run id and appends an attempt pointer — under the SAME suffix,
    bound to it and swept, that is one wave (every attempt kept, none selected silently); a cell with no pointer refuses."""
    W.sandbox_and_pointer(world["rung1"] / PSR, run_id=f"{W.SHA7}-99999999", cell=PSR, suffix=SUFFIX, model_slug="gemma4e4b-f22", fcid="fc-resume")
    a = _analyze(world)
    prov = a["identity"]["provenance"]
    assert prov["run_ids"] == sorted([RUN, f"{W.SHA7}-99999999"]) and prov["run_id"] is None and prov["n_attempts"] == len(CELLS) + 1
    assert [r["run_id"] for r in prov["attempts_by_cell"][PSR]] == [RUN, f"{W.SHA7}-99999999"]
    for f in (world["rung1"] / PSR / "watch").glob("attempt_*.json"):
        f.unlink()
    with pytest.raises(An.AnalysisError, match="no watch/attempt_\\*.json pointer"):
        _analyze(world)


def test_a_missing_pinned_item_and_a_non_binary_value_refuse(world):
    d = L.store_dir_for(world["rung1"], NONE, BFCL)
    rp = d / "rows.jsonl"
    lines = rp.read_text().splitlines()
    kept = [l for l in lines if not (json.loads(l)["item_id"] == F.BFCL_IDS[5] and json.loads(l)["metric"] == "acc")]
    rp.write_text("\n".join(kept) + "\n")
    with pytest.raises((An.AnalysisError, L.LadderError), match="not exactly the 200 pinned items|199 acc rows|different items"):
        _analyze(world)
    out = []
    for l in lines:
        row = json.loads(l)
        if row["item_id"] == F.BFCL_IDS[5] and row["metric"] == "acc":
            row["value"] = 0.5
        out.append(json.dumps(row, sort_keys=True))
    rp.write_text("\n".join(out) + "\n")
    with pytest.raises(An.AnalysisError, match="not 0/1"):
        _analyze(world)


# --------------------------------------------------------------------------- #
# breakdowns and diagnostics                                                   #
# --------------------------------------------------------------------------- #


def test_bfcl_by_category_and_bcb_by_family_carry_one_delta_per_steered_cell(world):
    a = _analyze(world)
    bd = a["breakdowns"]["bfcl_by_category"]
    assert bd["n_categories"] == 18 == len(F.BFCL_PIN["per_category"]) and sum(r["n"] for r in bd["rows"]) == 200
    assert {r["category"]: r["n"] for r in bd["rows"]} == F.BFCL_PIN["per_category"] and bd["rows"][0]["n"] == max(F.BFCL_PIN["per_category"].values())
    for r in bd["rows"]:
        for cell in (LORA, PSR):
            ka, kb = r["cells"][NONE]["k"], r["cells"][cell]["k"]
            assert abs(r["deltas"][cell]["delta"] - (kb - ka) / r["n"]) < 1e-12 and r["deltas"][cell]["b_pass_to_fail"] - r["deltas"][cell]["c_fail_to_pass"] == ka - kb
    fam = a["breakdowns"]["bcb_by_family"]
    assert fam["multi_label"] and fam["n_families"] == len(F.BCB_PIN["per_family"]) and sum(r["n"] for r in fam["rows"]) > 200
    fam_path = Path(world["repo"]) / An.LIB_FAMILIES_REL
    fams = json.loads(fam_path.read_text())
    lib = F.BCB_PIN["libs_by_id"][F.BCB_IDS[0]][0]
    del fams["families"][lib]
    fam_path.write_text(json.dumps(fams))
    with pytest.raises((An.AnalysisError, L.LadderError), match=f"no family.*{lib}|per_family|recompute|{lib}"):
        _analyze(world)


def test_bcb_summary_counts_must_agree_with_the_rows(world):
    d = L.store_dir_for(world["rung1"], NONE, BCB)
    sp = d / C.SUMMARY_DIR / L.marker_filename(BCB)
    s = json.loads(sp.read_text())
    s["n_code_source_python_fence"] -= 1
    s["n_code_source_bare_fence"] += 1
    sp.write_text(json.dumps(s))
    with pytest.raises(An.AnalysisError, match="BCB summary counts disagree"):
        _analyze(world)


def test_diagnostics_are_reported_apart(world):
    a = _analyze(world)
    for cell in CELLS:
        r = a["legs"][BCB]["cells"][cell]
        assert sum(r["code_source"].values()) == 200 and r["sandbox_status"] == {"ok": 200} and r["n_verify_timeout"] == 0
        assert r["at_cap"]["n_stop_reason"] == 200 and r["stop_reasons"].get("max_tokens", 0) == r["at_cap"]["n_at_cap"]
    assert a["legs"][BCB]["cells"][LORA]["at_cap"]["n_at_cap"] == 3 and a["legs"][BFCL]["cells"][PSR]["at_cap"]["n_at_cap"] == 2
    q = a["quoted"]["bcb_failures"]["cells"][NONE]
    assert q["n_fail"] == 200 - a["legs"][BCB]["cells"][NONE]["k"] and len(q["items"]) == 10 and sum(q["grid"].values()) == q["n_fail"]
    d = a["quoted"]["bfcl_discordant"]["cells"]
    assert set(d) == {LORA, PSR} and d[LORA]["n_discordant"] == a["legs"][BFCL]["paired"][LORA]["n_discordant"]


# --------------------------------------------------------------------------- #
# ladder (three cells, per-cell certificates), sandboxes, cost, wall           #
# --------------------------------------------------------------------------- #


def test_ladder_not_merged_is_check_4_not_met_when_a_leg_fired(world):
    a = _analyze(world)
    lad = a["ladder"]
    assert lad["fired"] == {BCB: True, BFCL: False} and lad["applied"] is False and lad["acceptance_check_4"] == "NOT MET (ladder pending)"
    assert lad["at_cap_counts"][BCB][LORA]["n_at_cap"] == 3 and lad["certificates"] == {}
    assert "NOT MET" in An.render_wave_doc(a)


def _merged_world(world, *, residual=None, seqs=16, values=None, admission=True):
    doc = L.build_selection(world["rung1"], repo_root=world["repo"])
    sel = L.write_selection(doc, world["tmp"] / "ladder_selection.json")
    rung2 = F.make_rung2(world["tmp"] / "rung2", doc, residual=(residual if residual is not None else {(LORA, BCB): {F.CAPPED_BCB[2]}}), cells=list(CELLS),
                         store_kwargs={"tree_sha": W.TREE},
                         store_kwargs_by_cell={LORA: {"object_sha256": W.SHA[LORA]}, PSR: {"object_sha256": W.SHA[PSR]}, NONE: {"object_sha256": None}},
                         values=(values or (lambda c, t, i: W.wave_values(c)(c, t, i))))
    spec = An.wave_spec(world["doc"], WAVE)
    for cell in CELLS:      # the rerun files ONE leg under the same protocol record (Slice A's producer shape)
        obj = spec.entry_of_cell(cell).object
        W.protocolize_store(L.store_dir_for(rung2, cell, BCB), tasks=[BCB], served_sha=(None if cell == NONE else W.SHA[cell]), cell=cell,
                            kind=(None if obj is None else obj.get("kind")), arm=(None if cell == NONE else cell.replace("__", "--")), null_route_digest=False)
    digests = {LORA: W.SHA[LORA], PSR: W.SHA[PSR]}
    lad = world["tmp"] / "rung2" / RP.LADDER_SUBDIR
    certs, cert_docs = {}, {}
    for cell in (LORA, PSR):
        p = F.make_certificate(lad / RP.certificate_filename(cell), doc, object_digests=digests, accepted=(32_768, seqs), tree_sha=W.TREE)
        c = json.loads(p.read_text()); c["probed_cell"] = cell
        RP.write_certificate(p, c)
        certs[cell], cert_docs[cell] = p, c
    if admission:           # the WAVE record the launcher writes before any scoring container (rung_probe's own writer)
        RP.write_admission(lad / RP.ADMISSION_FILENAME, RP.build_admission(suffix=cert_docs[LORA]["suffix"], common=cert_docs[LORA]["accepted"],
                                                                            certificates=cert_docs))
    merged = L.merge(world["rung1"], rung2, sel, certs[LORA], world["tmp"] / "merged", repo_root=world["repo"])
    return doc, sel, rung2, certs, merged


def test_ladder_readings_come_only_from_the_merge_with_per_cell_certificates(world):
    doc, sel, rung2, certs, merged = _merged_world(world)
    a = _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=rung2)
    lad = a["ladder"]
    assert lad["applied"] and lad["acceptance_check_4"].startswith("met") and lad["operator_rung"] is False and lad["fitted_without_certificate"] == []
    assert lad["final_reason"] is None and lad["admission_record"]["cells"].keys() == {LORA, PSR} and lad["superseded_certificates"] == []
    assert set(lad["certificates"]) == {LORA, PSR} and all(v["accepted"] == lad["accepted"] for v in lad["certificates"].values())
    assert a["final"][BCB]["source"] == "merged" and a["final"][BFCL]["source"] == "rung1" and a["final"][BCB]["cap_by_cell"][LORA] == 16_384
    assert lad["accepted"] == {"max_model_len": 32_768, "max_num_seqs": 16, "seqs_source": "override", "rung_source": "probe"} and list(lad["tasks"]) == [BCB]
    t = lad["tasks"][BCB]
    assert t["n_selected"] == 3 and set(t["paired_merged"]) == {LORA, PSR} and set(t["cells"]) == set(CELLS)
    for cell in CELLS:
        r, m = t["cells"][cell], merged["per_cell"][cell][BCB]
        assert (r["rung1"]["k"], r["merged"]["k"]) == (m["k_rung1"], m["k_merged"]) and r["merged"] == An.pass_rate(m["k_merged"], 200)
    st = t["cells"][LORA]
    assert st["n_at_cap_residual"] == 1 and st["residual_share"] == 1 / 200 and st["residual_share_of_selected"] == 1 / 3
    assert t["paired_rung1"] == a["legs"][BCB]["paired"] and "reading" in t["paired_merged"][LORA]
    md = An.render_wave_doc(a)
    assert "L = 32,768 × S = 16" in md and "`rung_source` **probe**" in md and "Budget decision(s) recorded" in md      # the fixture's spec-envelope decision
    rows = {(r["cell_id"], r["eval_id"]): r for r in An.paper_rows(a) if r["status"] in ("graded", "pending ladder")}
    assert rows[(LORA, "bigcodebench_200")]["status"] == "graded" and rows[(LORA, "bigcodebench_200")]["decode_cap_tokens"] == 16_384
    assert rows[(LORA, "bigcodebench_200")]["at_cap_share"] == 1 / 200 and "adaptive truncation rerun" in rows[(LORA, "bigcodebench_200")]["decode"]
    assert rows[(LORA, "bfcl_200")]["decode_cap_tokens"] == 1024 and "adaptive" not in rows[(LORA, "bfcl_200")]["decode"]


def test_ladder_refuses_a_certificate_at_another_rung_or_for_an_unfitted_cell(world):
    doc, sel, rung2, certs, merged = _merged_world(world)
    c = json.loads(certs[PSR].read_text())
    c["accepted"]["max_num_seqs"], c["walked"] = 18, [{"S": 18, "outcome": "accepted", "kv_tokens": None, "refusal_tail": None}]
    c["accepted"].update(seqs_source="model_budget", rung_source="identity")
    certs[PSR].write_text(json.dumps(c))
    with pytest.raises(An.AnalysisError, match="!= the merge's common rung"):
        _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=rung2)
    c["probed_cell"] = NONE
    c["accepted"] = dict(merged["certificate"]["accepted"]); c["walked"] = json.loads(certs[LORA].read_text())["walked"]
    certs[PSR].write_text(json.dumps(c))
    with pytest.raises(An.AnalysisError, match="not a fitted cell of this wave|probed_cell"):
        _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=rung2)


def test_ladder_refuses_a_stale_merge_another_selection_or_edited_rows(world):
    doc, sel, rung2, certs, merged = _merged_world(world)
    other = L.write_selection(F.retarget_selection(doc, BCB, F.BCB_IDS[1:3]), world["tmp"] / "other.json")
    with pytest.raises(An.AnalysisError, match="selection file's digest differs"):
        _analyze(world, ladder_merge=world["tmp"] / "merged", selection=other)
    with pytest.raises(An.AnalysisError, match="only with its ladder_selection.json"):
        _analyze(world, ladder_merge=world["tmp"] / "merged")
    path = Path(merged["per_cell"][NONE][BCB]["merged_rows"])
    lines = path.read_text().splitlines()
    row = json.loads(lines[0]); row["value"] = 1.0 - row["value"]; lines[0] = json.dumps(row, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(An.AnalysisError, match="rows were edited since the merge"):
        _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel)
    shutil.rmtree(world["rung1"])          # the SAME suffix re-run: BCB still fires, but every other item now passes — other hashes
    W.make_wave(world["rung1"], world["doc"], WAVE, roster_path=world["roster"], run_id=RUN, null_route_digest=False, combined=False,
                values=lambda c, t, i: 1.0, capped={(LORA, BCB): set(F.CAPPED_BCB), (NONE, BCB): {F.CAPPED_BCB[0]}})
    with pytest.raises(An.AnalysisError, match="stale or foreign merge cannot certify check 4"):
        _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel)


def test_the_final_reading_is_the_merge_when_it_changes_estimate_and_conclusion(tmp_path):
    """Rung 1 says "cost detected" (20 of the LoRA cell's items stop at the cap and fail; otherwise the
    steered cells answer exactly like `none`), the merge says "no cost detected" (every selected item passes at
    16,384). EVERY consumer — the record's `final`, the paper rows, the cross-wave summary, the wave headline, W&B
    scalars, the summary forest — reads the merged reading; rung 1 appears only under an explicit diagnostic label."""
    capped20 = set(F.BCB_LOG[1:21])
    world = make_world(tmp_path, capped={(LORA, BCB): capped20, (NONE, BCB): {F.CAPPED_BCB[0]}}, values=lambda c, t, i: float(i % 3 != 0))
    before = _analyze(world)
    p1 = before["legs"][BCB]["paired"][LORA]
    assert p1["reading"].startswith("cost detected") and before["final"][BCB]["final"] is False
    # pending: no final numbers anywhere but the diagnostics
    s0 = An.summarize([An.write_analysis(before, world["rung1"])], roster_path=world["roster"], ledger=world["ledger"])
    r0 = next(r for r in s0["readings"] if r["cell"] == LORA and r["task"] == BCB)
    assert r0["final"] is False and r0["delta_pp"] is None and r0["reading"].startswith("pending")
    sc0 = An.wandb_scalars(before)
    assert sc0[f"bcb200/paired/diff_lora/delta"] is None and abs(sc0["bcb200/rung1/paired/diff_lora/delta"] - p1["delta"]) < 1e-12
    assert sc0["bcb200/final_source"] == "pending" and sc0["bcb200/none/pass_rate"] is None and sc0["bcb200/rung1/none/pass_rate"] is not None
    rows0 = {(r["cell_id"], r["eval_id"]): r for r in An.paper_rows(before)}
    assert rows0[(LORA, "bigcodebench_200")]["status"] == "pending ladder" and rows0[(LORA, "bigcodebench_200")]["delta"] is None
    md0 = An.render_wave_doc(before)
    assert "**no final reading**" in md0 and "Rung 1 (diagnostic, NOT final)" in md0
    # the subgroup breakdowns / quotes of the pending leg are rung 1 and SAY so (image title label, doc header, note)
    bd0 = before["breakdowns"]["bcb_by_family"]
    assert bd0["source"] == "rung1" and bd0["final"] is False and bd0["label"] == "rung 1 — NOT final (ladder pending)"
    assert "# BigCodeBench-200 by library family (rung 1 — NOT final (ladder pending))" in md0 and "rung 1 and NOT final" in md0
    assert before["quoted"]["bcb_failures"]["final"] is False and "quoted raw (10 per cell; rung 1 — NOT final" in md0
    k_rung1 = {r["family"]: (r["cells"][NONE]["k"], r["cells"][LORA]["k"]) for r in bd0["rows"]}
    assert sum(b for _, b in k_rung1.values()) < sum(a_ for a_, _ in k_rung1.values())        # the 20 capped fails show in the families
    assert any(r["deltas"][LORA]["delta"] < 0 for r in bd0["rows"])
    # merged: every selected item passes at 16,384 → the contrast collapses to no cost
    doc, sel, rung2, certs, merged = _merged_world(world, residual={}, values=lambda c, t, i: 1.0)
    a = _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=rung2)
    p2 = a["ladder"]["tasks"][BCB]["paired_merged"][LORA]
    assert a["ladder"]["applied"] and p2["reading"].startswith("no cost detected") and p2["delta"] > p1["delta"] + 0.05
    fin = a["final"][BCB]
    assert fin["source"] == "merged" and fin["final"] and fin["paired"][LORA] == p2 and fin["cells"][LORA]["k"] == merged["per_cell"][LORA][BCB]["k_merged"]
    # the FINAL breakdowns come from the merged per-item rows: per family LoRA now equals none (every capped item passed), Δ = 0 everywhere
    bd = a["breakdowns"]["bcb_by_family"]
    assert bd["source"] == "merged" and bd["final"] is True and bd["label"] == "final: merged" and len(bd["rows"]) == len(bd0["rows"])
    assert all(r["cells"][LORA]["k"] == r["cells"][NONE]["k"] and r["deltas"][LORA] == {"delta": 0.0, "b_pass_to_fail": 0, "c_fail_to_pass": 0} for r in bd["rows"])
    assert sum(r["cells"][LORA]["k"] for r in bd["rows"]) > sum(b for _, b in k_rung1.values())
    assert a["quoted"]["bcb_failures"]["label"] == "final: merged" and a["quoted"]["bcb_failures"]["cells"][LORA]["n_fail"] == a["quoted"]["bcb_failures"]["cells"][NONE]["n_fail"]
    assert a["breakdowns"]["bfcl_by_category"]["label"] == "final: rung 1"                      # BFCL did not fire: rung 1 IS final
    assert "_rows" not in a["ladder"]["tasks"][BCB] and "_rows" not in a["legs"][BCB]
    rows = {(r["cell_id"], r["eval_id"]): r for r in An.paper_rows(a)}
    r = rows[(LORA, "bigcodebench_200")]
    assert r["status"] == "graded" and abs(r["delta"] - 100 * p2["delta"]) < 1e-9 and "no cost detected" in r["delta_method"] and "cost detected:" not in r["delta_method"]
    assert abs(r["value"] - fin["cells"][LORA]["rate"]) < 1e-12 and "rung 1 (diagnostic, not final)" in r["note"]
    s = An.summarize([An.write_analysis(a, world["rung1"])], roster_path=world["roster"], ledger=world["ledger"])
    rs = next(x for x in s["readings"] if x["cell"] == LORA and x["task"] == BCB)
    assert rs["source"] == "merged" and abs(rs["delta_pp"] - 100 * p2["delta"]) < 1e-9 and rs["reading"].startswith("no cost detected")
    forest = [x for x in An._summary_rows(s) if x["cell"] == LORA and x["task"] == BCB]
    assert len(forest) == 1 and forest[0]["paired"] == p2 and forest[0]["source"] == "merged"
    sc = An.wandb_scalars(a)
    assert abs(sc["bcb200/paired/diff_lora/delta"] - p2["delta"]) < 1e-12 and abs(sc["bcb200/rung1/paired/diff_lora/delta"] - p1["delta"]) < 1e-12
    assert sc["bcb200/final_source"] == "merged" and abs(sc["bcb200/diff_lora/pass_rate"] - fin["cells"][LORA]["rate"]) < 1e-12
    md = An.render_wave_doc(a)
    head = md.split("# Decoding protocol")[0]
    assert "(final: merged)" in head and "no cost detected" in head and "cost detected:" not in head
    assert "rung 1 (diagnostic)" in md                                 # the ladder section still shows both, labelled
    assert "# BigCodeBench-200 by library family (final: merged)" in md and "quoted raw (10 per cell; final: merged)" in md
    doc_md = An.render_results_doc([a], roster=world["doc"], summary=s)
    assert "| merged |" in doc_md
    import matplotlib.pyplot as plt
    files = An.make_figures(a, tmp_path / "figs")
    assert set(files) == set(An.FIGURE_STEMS) and plt.get_fignums() == []
    assert set(An.make_figures(s, tmp_path / "figs_summary", figures=An.SUMMARY_FIGURES)) == set(An.SUMMARY_FIGURE_STEMS) and plt.get_fignums() == []


def test_a_missing_certificate_or_wave_record_keeps_the_merge_not_final(world):
    """The merged reading is final only with the complete evidence — every fitted cell's certificate
    bound by the wave record. Missing either: the tables are recorded, `applied` is False, check 4 NOT MET, the paper
    rows carry null measurements with `pending ladder`; a record that disagrees with the certificates REFUSES."""
    doc, sel, rung2, certs, merged = _merged_world(world)
    lad_dir = rung2 / RP.LADDER_SUBDIR
    psr = json.loads(certs[PSR].read_text())
    certs[PSR].unlink()
    a = _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=rung2)
    lad = a["ladder"]
    assert lad["applied"] is False and lad["fitted_without_certificate"] == [PSR] and "without a certificate" in lad["final_reason"]
    assert lad["acceptance_check_4"].startswith("NOT MET") and BCB in lad["tasks"] and a["final"][BCB]["final"] is False
    rows = {(r["cell_id"], r["eval_id"]): r for r in An.paper_rows(a)}
    assert rows[(LORA, "bigcodebench_200")]["status"] == "pending ladder" and rows[(LORA, "bigcodebench_200")]["value"] is None
    assert rows[(LORA, "bfcl_200")]["status"] == "graded"
    md = An.render_wave_doc(a)
    assert "NOT the final reading" in md and "NOT MET" in md
    RP.write_certificate(certs[PSR], psr)
    # the wave record is REQUIRED for a final reading
    adm_path = lad_dir / RP.ADMISSION_FILENAME
    adm = json.loads(adm_path.read_text())
    adm_path.unlink()
    a = _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=rung2)
    assert a["ladder"]["applied"] is False and "no wave record" in a["ladder"]["final_reason"] and a["final"][BCB]["final"] is False
    # a record that names another certificate for a cell refuses outright
    bad = json.loads(json.dumps(adm)); bad["cells"][PSR]["certificate_sha256"] = "0" * 64
    RP.write_admission(adm_path, bad)
    with pytest.raises(An.AnalysisError, match="the wave record does not admit these certificates"):
        _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=rung2)
    RP.write_admission(adm_path, adm)
    assert _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=rung2)["ladder"]["applied"] is True


def test_a_poisoned_rung2_root_never_becomes_the_final_reading(world):
    """A ladder rerun whose driver finished and whose clamp scan then wrote
    `sampling_validation_failed.json` leaves a completed rung-2 PARTIAL store. `merge()` refuses it through the shared
    root admission; an ALREADY-PRODUCED merge (made before the marker landed) is refused by the analysis when it
    re-admits the contributing rung-2 roots (`--rung2`), and is NOT final without them."""
    doc, sel, rung2, certs, merged = _merged_world(world)
    ok = _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=rung2)
    assert ok["ladder"]["applied"] and all(r["rung2_root_readmitted"] and r["rung2_root"] == str(rung2 / c) for c, r in ok["ladder"]["tasks"][BCB]["cells"].items())
    marker = F.write_sampling_validation_marker(rung2 / LORA, cell=LORA, run_id=RUN, fcid="fc-rung2-lora", tree_sha=W.TREE)
    assert (L.store_dir_for(rung2, LORA, BCB) / C.PARTIAL_DIR).is_dir()                # a completed PARTIAL, markers intact
    with pytest.raises(An.AnalysisError, match=f"contributing rung-2 store is not admissible.*{C.SAMPLING_VALIDATION_FAILED_FILENAME}.*not protocol evidence"):
        _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=rung2)
    with pytest.raises(L.LadderError, match="not protocol evidence"):
        L.merge(world["rung1"], rung2, sel, certs[LORA], world["tmp"] / "merged2", repo_root=world["repo"])
    # without the rung-2 dir the merged tables are recorded but NEVER final, and the paper rows say so
    a = _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel)
    assert a["ladder"]["applied"] is False and "rung-2 stores were not re-admitted" in a["ladder"]["final_reason"] and a["final"][BCB]["final"] is False
    assert all(r["rung2_root_readmitted"] is False for r in a["ladder"]["tasks"][BCB]["cells"].values())
    rows = {(r["cell_id"], r["eval_id"]): r for r in An.paper_rows(a)}
    assert rows[(LORA, "bigcodebench_200")]["status"] == "pending ladder" and rows[(LORA, "bigcodebench_200")]["value"] is None
    marker.unlink()
    assert _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=rung2)["ladder"]["applied"] is True
    # a rung-2 dir whose store is NOT the one the merge read refuses (a swapped rerun cannot ride an old merge)
    F.make_rung2(world["tmp"] / "rung2b", doc, residual={}, cells=list(CELLS), store_kwargs={"tree_sha": W.TREE},
                 store_kwargs_by_cell={LORA: {"object_sha256": W.SHA[LORA]}, PSR: {"object_sha256": W.SHA[PSR]}, NONE: {"object_sha256": None}},
                 values=lambda c, t, i: 1.0)
    shutil.copytree(rung2 / RP.LADDER_SUBDIR, world["tmp"] / "rung2b" / RP.LADDER_SUBDIR)
    with pytest.raises(An.AnalysisError, match="is not the one the merge read"):
        _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=world["tmp"] / "rung2b")


def test_superseded_certificates_are_listed_and_never_read(world):
    doc, sel, rung2, certs, merged = _merged_world(world)
    lad_dir = rung2 / RP.LADDER_SUBDIR
    stale = json.loads(certs[PSR].read_text()); stale["accepted"]["max_num_seqs"] = 18; stale["accepted"].update(seqs_source="model_budget", rung_source="identity")
    stale["walked"] = [{"S": 18, "outcome": "accepted", "kv_tokens": None, "refusal_tail": None}]
    moved = RP.superseded_certificate_path(lad_dir / RP.certificate_filename(PSR), at=1_789_000_000.0)
    RP.write_certificate(moved, stale)                                 # what `--ladder-reprobe` leaves beside the fresh certificate
    a = _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=rung2)
    assert a["ladder"]["applied"] is True and a["ladder"]["superseded_certificates"] == [str(moved)] and set(a["ladder"]["certificates"]) == {LORA, PSR}
    assert moved.name in An.render_wave_doc(a)


def test_sandbox_lifetime_ends_at_the_first_observed_termination_and_cost_by_wave(world):
    events = next((world["rung1"] / NONE / "watch").glob("sandboxes_*.jsonl"))
    first = json.loads(events.read_text().splitlines()[0])           # the first sandbox created: its exit is observed a SECOND time, later
    owner = {"run_id": RUN, "cell": NONE, "function_call_id": f"fc-{NONE[:6]}"}
    with events.open("a") as fh:
        fh.write(json.dumps({"ts": W.T0 + 3000, "event": "terminated", "sandbox_id": first["sandbox_id"], "owner": owner, "verified_by": "relist"}, sort_keys=True) + "\n")
    a = _analyze(world)
    s = a["cost"]["sandboxes"][NONE]
    assert s["n_duplicate_terminated_events"] == 1 and s["n_terminated"] == 5 and s["lifetime_s"]["min"] == 30.0 and s["lifetime_s"]["max"] == 30.0 + 5 * 4
    assert s["n_created"] == 6 and s["n_unaccounted"] == 1 and abs(s["memory_gib"] - 0.5) < 1e-9 and "FIRST observed termination" in s["basis"]
    gpu = a["cost"]["gpu"]
    assert gpu["present"] and abs(gpu["wave_h"] - (0.15 + 0.4 + 0.4)) < 1e-9 and gpu["ceiling_h"] == 25.0 and gpu["within_ceiling"]
    assert abs(a["cost"]["wave_usd_estimate"] - (gpu["wave_h"] * 5.0 + a["cost"]["sandbox_usd_estimate"])) < 1e-9
    w = a["wall"][NONE]
    assert w["start"] is not None and w["hours"] > 0 and w["tasks_wall_s"] == 1800.0


def test_wall_ends_at_the_last_marker_of_every_supplied_rung(world):
    doc, sel, rung2, certs, merged = _merged_world(world)
    a1 = _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel)
    a2 = _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=rung2)
    assert a2["wall"][NONE]["hours"] > a1["wall"][NONE]["hours"] and f"rung2/{BCB}" in a2["wall"][NONE]["later_rungs"]


def test_sandbox_events_with_a_moved_shape_refuse(world):
    p = next((world["rung1"] / NONE / "watch").glob("sandboxes_*.jsonl"))
    p.write_text(json.dumps({"kind": "created", "id": "sb-1", "when": "2026-09-11T00:00:00Z"}) + "\n")
    with pytest.raises(An.AnalysisError, match="sandbox event without"):
        _analyze(world)


# --------------------------------------------------------------------------- #
# figures, doc                                                                 #
# --------------------------------------------------------------------------- #


def test_every_figure_renders_and_is_closed_in_both_ladder_variants(world, tmp_path):
    import matplotlib.pyplot as plt
    a = _analyze(world)
    files = An.make_figures(a, tmp_path / "figs")
    assert set(files) == set(An.FIGURE_STEMS) and plt.get_fignums() == []
    for stem, paths in files.items():
        assert [Path(p).suffix for p in paths] == [".png", ".svg"] and all(Path(p).stat().st_size > 0 for p in paths)
    doc, sel, rung2, certs, merged = _merged_world(world)
    b = _analyze(world, ladder_merge=world["tmp"] / "merged", selection=sel, rung2_dir=rung2)
    assert set(An.make_figures(b, tmp_path / "figs2")) == set(An.FIGURE_STEMS) and plt.get_fignums() == []
    with pytest.raises(An.AnalysisError, match="lines"):
        An.wrap_title("x " * 200)
    md = An.render_results_doc([b], roster=world["doc"], wandb_urls={WAVE: "https://wandb.example/run"})
    for needle in ("## Wave `gemma4e4b_main`", "Headline", "Identity (asserted before pairing)", "BFCL-200 by category", "BigCodeBench-200 by library family",
                   "Diagnostics", "Truncation ladder", "Sandbox health", "Cost and wall clock", "Figures", "BigCodeBench failures, quoted raw",
                   "BFCL discordant items", "Acceptance checks", "Wilson 95 %", "McNemar p", "not billing", "````text", "W&B: https://wandb.example/run", "full22 anchors"):
        assert needle in md, needle


def test_the_analysis_judges_a_native_wave_by_the_engine_served_ladder(monkeypatch):
    """`ladder_spec_for_certificates` returned the LENS ladder for every table-sized model, so a native wave's record (common rung 15 on
    the served ladder) could not be admitted and the analysis could not publish. Now the engine is read off the certificates' ladder
    blocks (one engine per wave) and a native wave is judged by the ENGINE-SERVED ladder recomputed from the table; two engines refuse;
    with the table refusing the native block RE-DERIVES from its requested rungs (a forged one refuses by name); the lens reading is
    byte-identical."""
    native, lens = RP.native_ladder_spec(RP.ladder_spec("gemma4e4b")), RP.ladder_spec("gemma4e4b")
    docs_n = {LORA: {"ladder": native.block()}, PSR: {"ladder": native.block()}}
    spec, source = An.ladder_spec_for_certificates("gemma4e4b", docs_n)
    assert spec == native and spec.native and source == "table"
    assert An.ladder_spec_for_certificates("gemma4e4b", {LORA: {"ladder": lens.block()}}) == (lens, "table")
    with pytest.raises(An.AnalysisError, match="several serving engines"):
        An.ladder_spec_for_certificates("gemma4e4b", {LORA: {"ladder": lens.block()}, PSR: {"ladder": native.block()}})
    # the wave record over native certificates at the served common rung admits through the native spec
    certs = {}
    for cell, seqs, walked in ((LORA, 15, [{"S": 15, "outcome": "accepted", "kv_tokens": None, "refusal_tail": None}]),
                               (PSR, 15, [{"S": 15, "outcome": "accepted", "kv_tokens": None, "refusal_tail": None}])):
        c = {"schema": RP.CERTIFICATE_SCHEMA, "suffix": "s", "accepted": RP.accepted_block(seqs, spec=native), "walked": walked,
             "ladder": native.block(), "probed_cell": cell, "reprobed_from": None}
        certs[cell] = c
    adm = RP.build_admission(suffix="s", common=certs[LORA]["accepted"], certificates=certs)
    assert RP.admission_is_valid(adm, suffix="s", spec=spec, certificates=certs) is None
    assert "is not a rung of the ladder [18, 16, 14, 12]" in (RP.admission_is_valid(adm, suffix="s", spec=lens, certificates=certs) or "")
    def refuse(model_key, *, window=RP.LADDER_WINDOW):
        raise RP.LadderError(f"the ladder needs the MEASURED budget row of {model_key!r}; the geometry table holds 'provisional-conservative'")
    monkeypatch.setattr(RP, "ladder_spec", refuse)
    spec, source = An.ladder_spec_for_certificates("gemma4e4b", docs_n)
    assert spec == native and source.startswith("certificates (the shared table refuses")
    forged = json.loads(json.dumps(native.block()))
    forged["served_rungs"][0]["requested"] = [18]
    with pytest.raises(An.AnalysisError, match="recorded ladder block cannot be read"):
        An.ladder_spec_for_certificates("gemma4e4b", {LORA: {"ladder": forged}})


NATIVE, LENS = RP.SERVING_ENGINE_NATIVE, RP.SERVING_ENGINE_LENS


def test_a_native_pull_addresses_the_native_namespace_and_the_lens_default_is_byte_identical(world, tmp_path, monkeypatch):
    """`remote_cell_path` / `remote_ladder_dir` always addressed `C.VOL_GEN_ROOT` and neither `pull_suffix` nor the CLI took an engine,
    while a native run's stores and certificates live under `<gen root>/native` — a native `--pull` found no run or, when a lens run
    shared the suffix, pulled the LENS stores and certificates and could publish that analysis under the operator's native-run suffix.
    Now `--serving-engine` (the lens as the compatibility default) reaches both addresses through `contracts.gen_root_for_engine` (the
    launcher's own derivation): over a fake volume holding the SAME suffix on both engines, the native pull lands the native stores and
    the native ladder dir (every generation request under `tinycap/gen/native/`), the lens pull the lens ones with the historical
    requests byte for byte; an unknown engine refuses by name before anything is fetched."""
    import sys
    slug = "gemma4e4b-f22it"
    lens_cell = An.remote_cell_path("bcbw-gemma4e4b_it-abc1234", NONE, model_slug=slug)
    assert lens_cell == "tinycap/gen/gemma4e4b-f22it/none/thinking_off/bcbw-gemma4e4b_it-abc1234"
    assert An.remote_cell_path("bcbw-gemma4e4b_it-abc1234", NONE, model_slug=slug, serving_engine=LENS) == lens_cell
    assert An.remote_cell_path("bcbw-gemma4e4b_it-abc1234", NONE, model_slug=slug, serving_engine=NATIVE) == \
        "tinycap/gen/native/gemma4e4b-f22it/none/thinking_off/bcbw-gemma4e4b_it-abc1234"
    assert An.remote_ladder_dir("s", model_slug="nemotron3nano4b") == An.remote_ladder_dir("s", model_slug="nemotron3nano4b", serving_engine=LENS) \
        == "tinycap/gen/nemotron3nano4b/_ladder/s"
    assert An.remote_ladder_dir("s", model_slug="nemotron3nano4b", serving_engine=NATIVE) == "tinycap/gen/native/nemotron3nano4b/_ladder/s"
    # THE ONE derivation the launcher's out roots and the sandbox watcher use; an unknown engine refuses by name
    assert An.gen_root_for_engine(NATIVE) == C.gen_root_for_engine(C.VOL_GEN_ROOT, NATIVE) == f"{C.VOL_GEN_ROOT}/{C.OUT_SUBDIR_NATIVE}"
    assert An.gen_root_for_engine(None) == An.gen_root_for_engine(LENS) == C.VOL_GEN_ROOT
    assert An.SERVING_ENGINES == (LENS, NATIVE) == tuple(C.SERVING_ENGINES)
    with pytest.raises(An.AnalysisError, match="serving_engine must be one of"):
        An.remote_cell_path("s", NONE, model_slug=slug, serving_engine="hf")
    # a fake volume holding the SAME suffix on BOTH engines: the lens run as the volume lays it out (`W.fake_volume`), and
    # the native run's store roots under tinycap/gen/native/… (each with a marker only the native copy carries) plus its
    # ladder dir — a native certificate — under tinycap/gen/native/<row>/_ladder/<suffix>/ (the lens run has no ladder dir)
    vol = tmp_path / "volume"
    build, _record = W.fake_volume(vol, world)
    spec = An.wave_spec(world["doc"], "nemotron3nano4b_main")
    row = str(spec.model_row)
    for cell in spec.cells:
        dst = vol / An.remote_cell_path(W.SUFFIX_NEM, cell, model_slug=row, serving_engine=NATIVE)
        shutil.copytree(build / cell, dst)
        (dst / "NATIVE_COPY").write_text("the native run's store root")
    lad = vol / An.remote_ladder_dir(W.SUFFIX_NEM, model_slug=row, serving_engine=NATIVE)
    lad.mkdir(parents=True)
    (lad / f"rung_probe.{W.DIFF_LORA}.json").write_text(json.dumps({"ladder": {"serving_engine": NATIVE}}))
    assert str(lad).startswith(f"{vol}/tinycap/gen/native/") and not (vol / An.remote_ladder_dir(W.SUFFIX_NEM, model_slug=row)).exists()
    requests = []
    get, ls = W.fake_modal(vol, requests)
    monkeypatch.setattr(An, "_modal_get", get)
    monkeypatch.setattr(An, "modal_volume_ls", ls)
    lines = []
    dest = An.pull_suffix(W.SUFFIX_NEM, spec, dest_root=tmp_path / "pulled_native", modal_python=sys.executable, log=lines.append,
                          serving_engine=NATIVE)
    gen_requests = sorted(r for r in requests if r.startswith("tinycap/gen/"))
    assert gen_requests == sorted([An.remote_cell_path(W.SUFFIX_NEM, c, model_slug=row, serving_engine=NATIVE) for c in spec.cells]
                                  + [An.remote_ladder_dir(W.SUFFIX_NEM, model_slug=row, serving_engine=NATIVE)])
    assert all(r.startswith("tinycap/gen/native/") for r in gen_requests), gen_requests
    assert all((dest / cell / "NATIVE_COPY").is_file() and (dest / cell / "cells").is_dir() for cell in spec.cells)
    assert (dest / An.LADDER_SUBDIR / f"rung_probe.{W.DIFF_LORA}.json").is_file()
    assert any(f"the {NATIVE} engine's namespace" in ln and "tinycap/gen/native/" in ln for ln in lines) and not any("no ladder dir" in ln for ln in lines)
    # the object sidecars live under the roster's objects_dir on either engine
    assert (dest / An.OBJECTS_SUBDIR / W.DIFF_LORA / "derived_meta.json").is_file()
    assert [r for r in requests if not r.startswith("tinycap/gen/")] == [r for r in requests if r.startswith(W.NEM_OBJECTS_DIR)]
    # the LENS pull (the compatibility default): the historical requests byte for byte, the lens copy (no marker), no ladder dir
    requests.clear()
    lines.clear()
    dest_l = An.pull_suffix(W.SUFFIX_NEM, spec, dest_root=tmp_path / "pulled_lens", modal_python=sys.executable, log=lines.append)
    gen_requests = sorted(r for r in requests if r.startswith("tinycap/gen/"))
    assert gen_requests == sorted([An.remote_cell_path(W.SUFFIX_NEM, c, model_slug=row) for c in spec.cells]
                                  + [An.remote_ladder_dir(W.SUFFIX_NEM, model_slug=row)])
    assert not any("/native/" in r for r in gen_requests) and gen_requests[0].startswith(f"tinycap/gen/{row}/")
    assert not any((dest_l / cell / "NATIVE_COPY").exists() for cell in spec.cells) and all((dest_l / cell / "cells").is_dir() for cell in spec.cells)
    assert any("no ladder dir" in ln for ln in lines) and not any("engine's namespace" in ln for ln in lines)
    assert not (dest_l / An.LADDER_SUBDIR / f"rung_probe.{W.DIFF_LORA}.json").exists()
    # an unknown engine refuses BEFORE any fetch
    requests.clear()
    with pytest.raises(An.AnalysisError, match="serving_engine must be one of"):
        An.pull_suffix(W.SUFFIX_NEM, spec, dest_root=tmp_path / "pulled_x", modal_python=sys.executable, log=lines.append, serving_engine="hf")
    assert requests == []
    # the CLI: `--serving-engine` is the meter's / ladder's vocabulary, threaded into pull_suffix; anything else is refused by argparse
    assert An.build_parser().parse_args(["--serving-engine", NATIVE]).serving_engine == NATIVE
    assert An.build_parser().parse_args([]).serving_engine is None
    with pytest.raises(SystemExit):
        An.build_parser().parse_args(["--serving-engine", "hf"])
    seen = {}

    def _pull(suffix, wave, **kw):
        seen.update(suffix=suffix, wave=wave.name, **kw)
        return tmp_path / "pulled_cli" / suffix
    monkeypatch.setattr(An, "pull_suffix", _pull)
    assert An.main(["--roster", str(world["roster"]), "--wave", "nemotron3nano4b_main", "--pull", W.SUFFIX_NEM, "--serving-engine", NATIVE,
                    "--pull-root", str(tmp_path / "pulled_cli"), "--modal-python", sys.executable]) == 0
    assert (seen["suffix"], seen["wave"], seen["serving_engine"]) == (W.SUFFIX_NEM, "nemotron3nano4b_main", NATIVE)
    seen.clear()
    assert An.main(["--roster", str(world["roster"]), "--wave", "nemotron3nano4b_main", "--pull", W.SUFFIX_NEM,
                    "--pull-root", str(tmp_path / "pulled_cli"), "--modal-python", sys.executable]) == 0
    assert seen["serving_engine"] is None                                       # the lens by default: pull_suffix's own default


def _native_pointers(rung1, cells):
    """Every attempt pointer of ``cells`` re-addressed to the NATIVE container's out root (what a `lora_native` launch
    records: `cell_out_dir(gen_root_for_engine(VOL_GEN_ROOT, lora_native), …)`); returns the lens out_dirs it replaced."""
    replaced = {}
    for cell in cells:
        for p in sorted((rung1 / cell / "watch").glob("attempt_*.json")):
            doc = json.loads(p.read_text())
            replaced[p] = doc["out_dir"]
            assert doc["out_dir"].startswith(f"{C.VOL_GEN_ROOT}/") and "/native/" not in doc["out_dir"]
            doc["out_dir"] = f"{C.gen_root_for_engine(C.VOL_GEN_ROOT, NATIVE)}/" + doc["out_dir"][len(C.VOL_GEN_ROOT) + 1:]
            p.write_text(json.dumps(doc))
    return replaced


def test_same_wave_provenance_judges_a_pointer_under_the_stores_own_engines_namespace(world, tmp_path):
    """`same_wave_provenance` composed every attempt pointer's expected `out_dir` from the LENS generation root, so a native wave —
    whose containers record their out roots under `<gen root>/native/` — refused every pointer as "a pointer from another suffix". The
    expected root now follows the STORES' own engine through `contracts.read_store_serving_engine` (THE ONE reader) and
    `gen_root_for_engine`: native stores admit native pointers and refuse lens ones (and the reverse), a torn engine spelling or two
    engines across the wave refuse by name; the lens world's record is byte-identical."""
    rung1, cells, tasks = world["rung1"], list(CELLS), (BCB, BFCL)
    wave = An.wave_spec(world["doc"], WAVE)
    state = tmp_path / "sandbox-state"

    def _stores():
        return {(c, t): L.load_store(rung1, c, t) for t in tasks for c in cells}

    def _prov(stores):
        return An.same_wave_provenance(stores, rung1=rung1, cells=cells, wave=wave, sandbox_state_root=state)

    lens_stores = _stores()
    assert all(C.read_store_serving_engine(st.manifest, where="m") == (LENS, None, None, C.read_store_serving_engine(st.manifest, where="m").why)
               for st in lens_stores.values())                                   # the fixtures: lens on positive evidence, nothing stated
    before = _prov(lens_stores)
    assert before["one_wave"] is True and "serving_engine" not in before          # the lens record: byte-identical keys
    # native pointers under lens stores: refused — a lens wave's containers never wrote under /native/
    replaced = _native_pointers(rung1, cells)
    with pytest.raises(An.AnalysisError, match=r"same-wave provenance: .*out_dir '.*/native/.*' is not this suffix's store root '.*' — a pointer from another suffix"):
        _prov(lens_stores)
    # the stores STAMPED native as the driver writes them (every spelling, `not_applicable`, the steered route `+lora+`):
    # the native pointers admit, the record as before
    for c in cells:
        for t in tasks:
            _edit_manifest(world, c, t, lambda m: _stamp_engine(m, NATIVE))
    native_stores = _stores()
    assert all(C.read_store_serving_engine(st.manifest, where="m").engine == NATIVE for st in native_stores.values())
    after = _prov(native_stores)
    assert {k: v for k, v in after.items() if k != "attempts_by_cell"} == {k: v for k, v in before.items() if k != "attempts_by_cell"}
    assert after["n_attempts"] == before["n_attempts"]
    # lens pointers under native stores: refused, naming the native namespace
    for p, out_dir in replaced.items():
        doc = json.loads(p.read_text())
        doc["out_dir"] = out_dir
        p.write_text(json.dumps(doc))
    with pytest.raises(An.AnalysisError, match=r"is not this suffix's store root '.*/native/.*' \(the stores' lora_native engine's namespace\) — a pointer from another suffix"):
        _prov(native_stores)
    _native_pointers(rung1, cells)
    assert _prov(_stores())["one_wave"] is True
    # two engines across the wave, or a torn spelling, refuse by name (never read as either root)
    _edit_manifest(world, NONE, BCB, lambda m: _stamp_engine(m, LENS))
    with pytest.raises(An.AnalysisError, match=r"several serving engines \['lens', 'lora_native'\] — not one wave"):
        _prov(_stores())
    _edit_manifest(world, NONE, BCB, lambda m: m.__setitem__(C.SERVED_ENGINE_FIELD, "hf"))
    with pytest.raises(An.AnalysisError, match=f"{NONE}/{BCB}: the store's serving engine cannot be read — .*'hf'.*not one of"):
        _prov(_stores())
