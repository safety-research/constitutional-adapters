"""math-eight-arms-extension P11 (decisions-impl-laneA A8 / A9): the COMPATIBILITY PREFLIGHT of the MATH-500 extension
— `scripts/sdfbase/math500/m5t_compat_preflight.py`. What is pinned here:

  1. the SHARED-FIELD projection (P5 (i)) is compared field by field and every refusal NAMES the field — the numeric
     sampler (the matrix dict's `source` prose never enters), the model revision, the template digest, the enable flag,
     the item population (reported as the symmetric difference, never 500 ids twice);
  2. each row's OWN contract (P5 (i')) is validated for what the row is, never compared with `none`: a prompted row's SI
     digest against the overlay manifest and the registry text; a lens row's pin, staging and tuple; the geometry entry
     resolving to the row's L at the model's budget (S recorded: the constitution rows at the long-SI window);
     optionally the budget fit sidecar at the wave sha;
  3. an OMITTED sampler key is resolved under BOTH trees' rules (A8) and a difference refuses by key;
  4. end to end (where the pinned tokenizers and the datasets cache are present offline): a synthetic published record
     carrying the wave tree's own facts PASSes; a moved field FAILs by name (rc 1); a missing published manifest is a
     REFUSAL (rc 2), never a skipped field; and the REAL published pull, when this machine has it, PASSes for the
     three extension models.

Everything synthetic here is synthetic (placeholder shas, no host names); the real-pull check reads a local directory
and skips when it is absent (the CPU suite has no grade directory)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "math500"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cells as M  # noqa: E402

from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer import site as SITE  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load("_m5t_compat_preflight_under_test", "scripts/sdfbase/math500/m5t_compat_preflight.py")
GRADE_PUBLISHED = SITE.expand_path("$GRADE/tinycap/math500/pull_55e3cbd/controls")
IDS = [f"math_{i:08x}" for i in range(500)]
SHA = "ab" * 20


def _bound(monkeypatch, sha: str = SHA) -> None:
    """Bind the CLI to `sha` as if this checkout's HEAD were it with clean evidence roots (fix A2): the CLI tests below
    exercise the COMPARISONS; the binding itself has its own test (test_the_certificate_is_bound_to_the_checkout_it_read)."""
    monkeypatch.setattr(P, "tree_facts", lambda repo, evidence_roots=P.EVIDENCE_ROOTS: {
        "head": sha, "evidence_roots": list(evidence_roots), "dirty_paths": [], "dirty_paths_outside_evidence_roots": [], "how": "stub"})


@pytest.fixture
def synthetic_lock(monkeypatch):
    """The SYNTHETIC published side of the client-lock relation (L1 G0): the synthetic manifests stamp the WAVE tree's own
    lock digest with the published tree as producer, so the published tree's lock TEXT is answered with the wave's own
    text — equal by construction, whatever tree this clone carries (the Modal suite's staged tree has no 55e3cbdd). The
    real relation (git show of the published tree) has its own tests below. The relation cache is cleared around it."""
    wave_text = (REPO / C.CLIENT_LOCK_TXT).read_text(encoding="utf-8")
    P._LOCK_RELATION_CACHE.clear()
    monkeypatch.setattr(P, "published_lock_text", lambda tree: (wave_text, None))
    yield wave_text
    P._LOCK_RELATION_CACHE.clear()


def _cfg(key):
    return yaml.safe_load((REPO / f"configs/sdfbase/math500/sdfbase_full22_{key}.math500.yaml").read_text())


def _synthetic_manifest(key: str, *, template_hash: str, ids=IDS, **over) -> dict:
    """A published `none` manifest carrying the WAVE tree's own facts for `key` (so a comparison against it is equal by
    construction, and every flip below is the ONE moved field)."""
    cfg = _cfg(key)
    none = M.row_for(key, M.CELL_NONE, M.SAMPLING_THINKING_CARD)
    fact = C.FAMILY_RENDER_FACTS[none["family"]]
    lock = P.client_lock_facts()
    man = {"controls": [M.CONTROL], "cell_id": "none", "model_id": cfg["model"], "model_revision": cfg["model_revision"],
           # the driver's own block shape (mode, model key, the hashed identity of the contracts row, its source prose)
           "sampling": P.wave_sampling_block(none),
           "thinking": {"mode": "on", "enable_thinking": True, "render": "flag_only", "reasoning_parser": none["reasoning_parser"]},
           "tinycap": {"identity_version": 2, "max_tokens_by_task": {"inspect/math": 16384}, "math500_max_tokens": 16384,
                       "math500_retry_max_tokens": 0, "samples_by_task": {"inspect/math": 1},
                       "served": {"reduction_mode": "stacked", "max_model_len": 17664, "max_num_seqs": 1},
                       "thinking_header_ids": {"off": list(fact.off_ids), "on": list(fact.on_ids)}},
           "provenance": {"serving": {"chat_template_hash": template_hash, "resolved_model_id": cfg["model"],
                                      "resolved_revision": cfg["model_revision"]}},
           "frameworks": {"client_lock_sha256": lock["client_lock_sha256"], "inspect_ai": lock["inspect_ai"], "inspect_evals": lock["inspect_evals"]},
           "producer": {"tree_sha": P.PUBLISHED_TREE}}
    for dotted, v in over.items():
        node = man
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = v
    return man


def _write_published(root: Path, key: str, man: dict, ids=IDS, *, ref=None) -> Path:
    """A published leg's files as the preflight reads them: the manifest, the OUTCOMES record (every attempted item
    with its `sampling_ref`), gen_stats (the leg's ref) and the rows (the SCORED subset — one fewer than attempted here,
    as the real qwen leg has 71 unscored at-cap items, so a reader off the rows would be wrong by construction)."""
    d = root / key / "none" / "thinking_card" / "cells" / "none__controls__t0__a0__synthetic"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(man))
    ref = ref if ref is not None else C.sampling_ref(man["sampling"])
    items = [{"item_id": i, "sampling_ref": ref, "at_cap": False, "scored": True} for i in ids]
    (d / "outcomes").mkdir(exist_ok=True)
    (d / "outcomes" / "inspect__math.json").write_text(json.dumps({
        "task": "inspect/math", "items": items, "n_attempted_items": len(items), "n_attempted_samples": len(items),
        "n_scored": len(items) - 1, "n_unscored": 1, "n_at_cap": 1, "samples_per_item": 1, "unterminated_policy": "unscored"}))
    (d / "gen_stats.json").write_text(json.dumps({"inspect/math": {"n": len(items), "sampling_ref": ref}}))
    rows = [{"task": "inspect/math", "metric": "acc", "kind": "cell", "item_id": i} for i in ids[:-1]]
    rows.append({"task": "inspect/math", "metric": "acc_aggregate", "kind": "cell", "item_id": "__aggregate__"})
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d / "manifest.json"


def _tokenizers_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001
        return False
    for key in M.EXT_MODELS:
        cfg = _cfg(key)
        if not isinstance(try_to_load_from_cache(cfg["model"], "tokenizer_config.json", revision=cfg["model_revision"]), str):
            return False
    return True


# --------------------------------------------------------------------------- (i) the shared projection


def test_the_shared_projection_is_compared_field_by_field_and_a_moved_field_is_named(tmp_path, synthetic_lock):
    key = "qwen35_4b"
    row = M.row_for(key, M.CELL_LORA_MIX, M.SAMPLING_THINKING_CARD)
    cfg = _cfg(key)
    lock = P.client_lock_facts()
    assert lock["client_lock_sha256"] == hashlib.sha256((REPO / "configs/capeval/tinycap-client-requirements.txt").read_bytes()).hexdigest()
    assert lock["inspect_ai"] and lock["inspect_evals"]
    # L1 G0: the lock facts carry the text and the client's import walk (the compared set is DERIVED, never typed)
    assert lock["text"] == synthetic_lock and lock["walk"].entry_points == P.CL.CLIENT_ENTRY_POINTS and lock["walk"].missing_entry_points == []
    assert {"inspect_ai", "inspect_evals", "yaml"} <= set(lock["walk"].third_party) and "client_lock_sha256" not in P.SHARED_FIELDS and "client_lock" in P.SHARED_FIELDS
    man_path = _write_published(tmp_path / "pub", key, _synthetic_manifest(key, template_hash="ab" * 8))
    pub = P.published_shared(json.loads(man_path.read_text()), man_path)
    wave = P.wave_shared(row, key, cfg, "ab" * 8, IDS, lock)
    assert "source" not in wave["sampling_params"] and wave["sampling_params"] == M.sampling_params(row["sampling"])
    rec, problems, undetermined = P.compare_shared(pub, wave)
    assert problems == [] and undetermined == [] and all(v["equal"] for v in rec.values()) and set(rec) == set(P.SHARED_FIELDS)
    assert rec["item_ids"] == {"equal": True, "published_n": 500, "wave_n": 500, "only_published": [], "only_wave": []}
    # fix A1: a wave side whose population was NOT planned is UNDETERMINED by name — never the published ids echoed back
    unplanned = P.wave_shared(row, key, cfg, "ab" * 8, None, lock)
    assert unplanned["item_ids"] is None
    rec_u, problems_u, undetermined_u = P.compare_shared(pub, unplanned)
    assert problems_u == [] and undetermined_u == [P.ITEM_IDS_NOT_PLANNED] and "--no-plan-items" in P.ITEM_IDS_NOT_PLANNED
    assert rec_u["item_ids"] == {"equal": None, "published_n": 500, "wave_n": None, "only_published": None, "only_wave": None,
                                 "undetermined": P.ITEM_IDS_NOT_PLANNED}
    assert all(v["equal"] for f, v in rec_u.items() if f != "item_ids")
    # the population is the ATTEMPTED set (the outcomes record), never the scored rows (one fewer here, 71 fewer on the real qwen leg)
    assert pub["population"]["n_attempted_items"] == 500 and pub["population"]["n_scored"] == 499
    # sampling_ref: the leg's recorded ref rebuilds from its own manifest block, and the wave's row rebuilds the SAME block
    assert pub["sampling_ref_record"]["self_consistent"] and rec["sampling_ref"]["equal"] and wave["sampling_ref"] == pub["sampling_ref"]
    assert wave["sampling_block_keys"] == sorted(P.wave_sampling_block(row)) and "source" in wave["sampling_block_keys"]
    # a constitution row (a different arm, a different geometry) is compatible with `none` on the shared projection
    consi = M.row_for(key, M.CELL_PROMPTED_CONST, M.SAMPLING_THINKING_CARD)
    assert P.compare_shared(pub, P.wave_shared(consi, key, cfg, "ab" * 8, IDS, lock))[1:] == ([], [])
    # one moved field at a time: named, nothing else (a moved sampler number moves BOTH the numeric projection and the ref).
    # A moved `frameworks.client_lock_sha256` alone is no longer a moved FIELD (L1 G0): with the published tree's lock text
    # digesting the wave's sha (the synthetic fixture), a record naming another digest breaks the CHAIN OF CUSTODY — named
    # `shared.client_lock` (the leg did not run under its tree's lock), `equal: False`, still one problem
    for over, fields in ((("sampling.top_p", 0.8), {"sampling_params", "sampling_ref"}), (("model_revision", "0" * 40), {"model_revision"}),
                         (("provenance.serving.chat_template_hash", "cd" * 8), {"chat_template_hash"}),
                         (("thinking.enable_thinking", False), {"enable_thinking"}), (("tinycap.identity_version", 1), {"identity_version"}),
                         (("tinycap.max_tokens_by_task", {"inspect/math": 8192}), {"max_tokens_by_task"}),
                         (("frameworks.client_lock_sha256", "ee" * 32), {"client_lock"})):
        mp = _write_published(tmp_path / f"pub-{'-'.join(sorted(fields))}", key, _synthetic_manifest(key, template_hash="ab" * 8, **{over[0]: over[1]}))
        rec, problems, undetermined = P.compare_shared(P.published_shared(json.loads(mp.read_text()), mp), wave)
        assert {p.split(":")[0] for p in problems} == {f"shared.{f}" for f in fields} and len(problems) == len(fields), (fields, problems)
        assert all(v["equal"] is (f not in fields) for f, v in rec.items()) and undetermined == [], fields
        if "client_lock" in fields:
            assert "chain of custody" in problems[0] and rec["client_lock"]["published"]["matches_record"] is False
            assert rec["client_lock"]["published"]["sha256_recorded"] == "ee" * 32 and rec["client_lock"]["published"]["tree"] == P.PUBLISHED_TREE
    # the relation on the equal-by-construction fixture: the digests equal, every imported distribution equal, nothing added
    cl = rec_ok = P.compare_shared(pub, wave)[0]["client_lock"]
    assert cl["equal"] is True and cl["one_instrument"] is True and cl["sha256"]["equal"] is True and cl["changed"] == []
    assert cl["n_compared"] >= 20 and {"inspect-ai", "inspect-evals", "pyyaml", "torch", "transformers"} <= set(cl["imported_packages"])
    assert cl["imported_packages"]["pyyaml"]["import_names"] == ["yaml"] and cl["imported_packages"]["pyyaml"]["resolver"] == P.CL.RESOLVER_ALIAS
    assert cl["imported_added"] == {} and cl["outside_imported_set"] == {"added": [], "removed": [], "changed": {}}
    assert cl["published"]["matches_record"] is True and cl["walk"]["n_modules_walked"] > 100
    # a published record whose recorded ref does not rebuild from its own block is named (not self-consistent)
    mp = _write_published(tmp_path / "pub-badref", key, _synthetic_manifest(key, template_hash="ab" * 8), ref="000000000000")
    assert P.published_shared(json.loads(mp.read_text()), mp)["sampling_ref_record"]["self_consistent"] is False
    # the population: one id missing from the published leg is the symmetric difference, never 500 ids twice
    mp = _write_published(tmp_path / "pub-ids", key, _synthetic_manifest(key, template_hash="ab" * 8), ids=IDS[:-1] + ["math_ffffffff"])
    rec, problems, undetermined = P.compare_shared(P.published_shared(json.loads(mp.read_text()), mp), wave)
    assert len(problems) == 1 and problems[0].startswith("shared.item_ids:") and undetermined == []
    assert rec["item_ids"]["only_published"] == ["math_ffffffff"] and rec["item_ids"]["only_wave"] == [IDS[-1]]
    # a missing outcomes record is a refusal, never a skipped field
    (man_path.parent / "outcomes" / "inspect__math.json").unlink()
    with pytest.raises(P.Refusal):
        P.published_item_ids(man_path)


# --------------------------------------------------------------------------- (i') each row's own contract


def test_each_rows_own_contract_is_validated_for_what_the_row_is_never_compared_with_none(tmp_path):
    pins = json.loads(P.PINS.read_text())
    for row in M.EXTENSION_ROWS:
        key = row["model_key"]
        _, ov_man, _, _ = P.overlay(key)
        rec, problems = P.row_contract(row, ov_man, pins, P.roster_main_cells(key), None, SHA)
        assert problems == [], (row["job_id"], problems)
        assert rec["geometry"]["L_resolved"] == rec["geometry"]["L_row"] == row["geometry_L"]
        # qwen35_4b: 741,888 = 42 x 17,664 (MEASURED at the MATH-500 window 2026-09-13) -> 42 at 17,664, 30 at 24,064 (the swept
        # 720,896 gave 40 / 29; the published wave ran at the provisional 20 / 14)
        assert rec["geometry"]["S"] == {("gemma4e4b", 24064): 24, ("qwen35_4b", 24064): 30, ("nemotron3nano4b", 24064): 32,
                                        ("gemma4e4b", 17664): 33, ("qwen35_4b", 17664): 42, ("nemotron3nano4b", 17664): 44}[(key, row["geometry_L"])]
        assert rec["budget"] == {"checked": False, "why": "no --budgets-root given (the launcher re-checks the sidecar at launch: m5t_budget_artifact)"}
        if row["route_kind"] == "prompted":
            assert rec["si"]["sha256_registry"] == row["si_sha256"] == "21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f"
            assert rec["si"]["manifest_prefix_hex16"] == row["si_sha256"][:16] and rec["si"]["harness"] == "constitution"
        else:
            assert rec["object"]["pin_file"] == row["object_sha256"] and rec["object"]["staged_on_main_view"] is True
            assert rec["object"]["tuple"] == C.m5t_ext_tuple_name(key)
    # the refusals, each by name
    lora = M.row_for("gemma4e4b", M.CELL_LORA_CTL2, M.SAMPLING_THINKING_CARD)
    _, ov_man, _, _ = P.overlay("gemma4e4b")
    staged = P.roster_main_cells("gemma4e4b")
    bad_pins = json.loads(json.dumps(pins))
    bad_pins["gemma4e4b"][lora["staged_id"]]["object_sha256"] = "0" * 64
    assert any("pin 0000" in p for p in P.row_contract(lora, ov_man, bad_pins, staged, None, SHA)[1])
    assert any("not staged" in p for p in P.row_contract(lora, ov_man, pins, {M.CELL_OBJECT: "x"}, None, SHA)[1])
    consi = dict(M.row_for("gemma4e4b", M.CELL_PROMPTED_CONST, M.SAMPLING_THINKING_CARD), si_sha256="ff" * 32)
    assert any("SI text sha256" in p for p in P.row_contract(consi, ov_man, pins, staged, None, SHA)[1])
    man2 = dict(ov_man, prompted_prefix_hashes={**ov_man["prompted_prefix_hashes"], "constitution": "0" * 16})
    consi = M.row_for("gemma4e4b", M.CELL_PROMPTED_CONST, M.SAMPLING_THINKING_CARD)
    assert any("prompted_prefix_hashes[constitution]" in p for p in P.row_contract(consi, man2, pins, staged, None, SHA)[1])
    # the budget fit sidecar at the wave sha: ok, or the mismatch named
    root = tmp_path / "budgets"
    side = root / "gemma4e4b" / consi["budget_class"] / (M.OOS_PROMPT_BUDGET_FILENAME + ".fit.json")
    side.parent.mkdir(parents=True)
    good = {"fits": True, "cap": M.CAP, "geometry_eval": consi["geometry_eval"], "window": consi["geometry_L"], "arm": "on", "tree_sha": SHA,
            "system_prompt_cell": consi["cell"], "system_prompt_sha16": consi["si_sha256"][:16], "prompt_max": 6981, "margin": 699}
    side.write_text(json.dumps(good))
    rec, problems = P.row_contract(consi, ov_man, pins, staged, root, SHA)
    assert problems == [] and rec["budget"]["ok"] is True and rec["budget"]["prompt_max"] == 6981
    side.write_text(json.dumps({**good, "tree_sha": "0" * 40}))
    rec, problems = P.row_contract(consi, ov_man, pins, staged, root, SHA)
    assert rec["budget"]["ok"] is False and rec["budget"]["mismatch"] == {"tree_sha": {"sidecar": "0" * 40, "row": SHA}} and len(problems) == 1
    rec, problems = P.row_contract(lora, ov_man, pins, staged, root, SHA)       # no sidecar for the lora's class: named
    assert rec["budget"]["ok"] is False and "no fit sidecar" in rec["budget"]["why"] and len(problems) == 1
    # an EXTERNAL row's contract — the pin is the ADMISSION RECORD's, the view the record's, a bank view on
    # some cluster, the `_steer` tuple on a switch model and NO tuple on the no-switch family (in support); the roster's
    # `main` view is never consulted (the full22 pins file never carries these objects)
    admission = M.load_steer_admission()
    for row in M.STEER_ROWS:
        adm = admission[(row["model_key"], row["cell"])]
        _, ov_man_row, _, _ = P.overlay(row["model_key"]) if (REPO / f"configs/sdfbase/math500/sdfbase_full22_{row['model_key']}.math500.yaml").is_file() else (None, {}, None, None)
        rec, problems = P.row_contract(row, ov_man_row, pins, {}, None, SHA)
        assert problems == [], (row["job_id"], problems)
        assert rec["object"]["source"] == M.OBJECT_SOURCE_EXTERNAL == row["object_source"] and rec["object"]["admission_pin"] == adm["object_sha256"] == row["object_sha256"]
        assert rec["object"]["admission_view"] == rec["object"]["view"] == row["view"] and rec["object"]["bank_view_rel"] == adm["bank_view_rel"]
        assert rec["object"]["served_positions_expected"] == "all" and "staged_on_main_view" not in rec["object"]
        assert rec["geometry"]["L_resolved"] == rec["geometry"]["L_row"] == 17664
        if row["model_key"] == "llama8b":
            assert rec["object"]["tuple"] is None and row["thinking_mode"] == C.THINKING_NOT_AVAILABLE
        else:
            assert rec["object"]["tuple"] == C.m5t_steer_tuple_name(row["model_key"]) and row["thinking_mode"] == "on"
    steer = M.STEER_ROWS[0]
    _, ov_g, _, _ = P.overlay("gemma4e4b")
    bad = {k: dict(v, object_sha256="0" * 64) for k, v in admission.items()}
    assert any("admission pin 0000" in p for p in P.row_contract(steer, ov_g, pins, {}, None, SHA, admission=bad)[1])
    assert any("no admission record" in p for p in P.row_contract(steer, ov_g, pins, {}, None, SHA, admission={})[1])
    assert any("admission view" in p for p in P.row_contract(steer, ov_g, pins, {}, None, SHA, admission={k: dict(v, view="other") for k, v in admission.items()})[1])
    assert any("no tuple" in p for p in P.row_contract(dict(steer, tuple_name=None), ov_g, pins, {}, None, SHA)[1])
    assert P.row_contract(dict(steer, tuple_name=None, thinking_mode=C.THINKING_NOT_AVAILABLE), ov_g, pins, {}, None, SHA)[1] == []


# --------------------------------------------------------------------------- (i'') the omitted keys under both rules


def test_an_omitted_sampler_key_is_resolved_under_both_trees_rules_and_a_difference_refuses_by_key(monkeypatch):
    assert P.VLLM_NEUTRAL_DEFAULTS == __import__("record_resolved_sampler").VLLM_NEUTRAL_DEFAULTS
    qwen = M.row_for("qwen35_4b", M.CELL_LORA_MIX, M.SAMPLING_THINKING_CARD)
    monkeypatch.setattr(P, "snapshot_generation_config", lambda model, rev: {"status": P.SNAPSHOT_NO_EXIST, "keys": None})   # ships no file
    rec, problems, undet = P.sampler_resolution(qwen, "Qwen/Qwen3.5-4B", "r", json.dumps({"top_p": 0.8, "repetition_penalty": 1.0, "top_k": 20}))
    assert rec["omitted_keys"] == ["repetition_penalty"] and problems == [] and undet == []
    assert rec["per_key"]["repetition_penalty"]["published"] == 1.0 and rec["per_key"]["repetition_penalty"]["wave"] == 1.0
    assert "override-generation-config" in rec["per_key"]["repetition_penalty"]["wave_rule"] and "ships no" in rec["per_key"]["repetition_penalty"]["published_rule"]
    rec, problems, undet = P.sampler_resolution(qwen, "Qwen/Qwen3.5-4B", "r", json.dumps({"repetition_penalty": 1.1}))
    assert problems == ["sampler_resolution.repetition_penalty: an omitted key resolves to 1.0 under the published tree's rule but 1.1 under the wave tree's"]
    gemma = M.row_for("gemma4e4b", M.CELL_LORA_MIX, M.SAMPLING_THINKING_CARD)
    monkeypatch.setattr(P, "snapshot_generation_config", lambda model, rev: {"status": P.SNAPSHOT_CACHED, "keys": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}})
    rec, problems, undet = P.sampler_resolution(gemma, "google/gemma-4-E4B-it", "r", None)
    assert rec["omitted_keys"] == ["min_p", "presence_penalty", "repetition_penalty"] and problems == [] and undet == []
    assert all(v["published"] == v["wave"] == P.VLLM_NEUTRAL_DEFAULTS[k] for k, v in rec["per_key"].items())
    assert all("sets no such key" in v["published_rule"] for v in rec["per_key"].values())
    monkeypatch.setattr(P, "snapshot_generation_config", lambda model, rev: {"status": P.SNAPSHOT_CACHED, "keys": {"repetition_penalty": 1.2}})
    rec, problems, undet = P.sampler_resolution(gemma, "google/gemma-4-E4B-it", "r", None)
    assert problems == [] and rec["per_key"]["repetition_penalty"]["published"] == 1.2 == rec["per_key"]["repetition_penalty"]["wave"]
    assert "generation_config.json" in rec["per_key"]["repetition_penalty"]["published_rule"]
    # a file this machine has NOT cached: the published rule is UNDETERMINED by key (never read as neutral, never as equal)
    monkeypatch.setattr(P, "snapshot_generation_config", lambda model, rev: {"status": P.SNAPSHOT_NOT_CACHED, "keys": None, "remedy": "hf download ..."})
    rec, problems, undet = P.sampler_resolution(gemma, "google/gemma-4-E4B-it", "r", None)
    assert problems == [] and len(undet) == 3 and all(u.startswith("sampler_resolution.") and "remedy: hf download" in u for u in undet)
    assert all(v["equal"] is None and v["published"] is None for v in rec["per_key"].values())
    # the real cache reader keeps the hub's three readings apart
    assert P.snapshot_generation_config("nobody/no-such-model", "0" * 40)["status"] in (P.SNAPSHOT_NOT_CACHED, P.SNAPSHOT_NO_EXIST)
    # fix A1: a launch that was NOT composed (--no-compose) leaves every omitted key's WAVE rule unread — UNDETERMINED by key,
    # never the published rule reused as the wave's (the old `override_json=None` read as "no override row" and PASSed)
    monkeypatch.setattr(P, "snapshot_generation_config", lambda model, rev: {"status": P.SNAPSHOT_NO_EXIST, "keys": None})
    rec, problems, undet = P.sampler_resolution(qwen, "Qwen/Qwen3.5-4B", "r", P.NOT_COMPOSED)
    assert problems == [] and undet == [f"sampler_resolution.repetition_penalty: {P.WAVE_RULE_NOT_COMPOSED}"]
    assert rec["composed"] is False and rec["override_generation_config"] is None
    assert rec["per_key"]["repetition_penalty"] == {"published": 1.0, "published_rule": rec["per_key"]["repetition_penalty"]["published_rule"],
                                                    "wave": None, "wave_rule": P.WAVE_RULE_NOT_COMPOSED, "equal": None}
    rec, problems, undet = P.sampler_resolution(gemma, "google/gemma-4-E4B-it", "r", None)      # composed, no override row: read as before
    assert rec["composed"] is True and problems == [] and undet == [] and all(v["equal"] is True for v in rec["per_key"].values())


# --------------------------------------------------------------------------- end to end


@pytest.mark.skipif(not _tokenizers_cached(), reason="the pinned tokenizers are not in the local HF cache (offline)")

def _skip_unless_published_tree_in_this_clone() -> None:
    """The preflight reads the published tree's launcher files with `git show <tree>:<file>` (the server-default thinking
    policy, A4); a shallow standalone checkout (the Modal CPU suite's) does not carry that tree, so the tests that exercise
    the read skip by name there — the same guard the campaign-config pins use for their BASE_SHA."""
    r = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{P.PUBLISHED_TREE}^{{commit}}"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"the published tree {P.PUBLISHED_TREE[:12]} is not in this clone — the preflight reads its launcher with git show")

def test_cli_passes_on_a_synthetic_record_carrying_the_wave_trees_facts_and_fails_or_refuses_by_name(tmp_path, capsys, monkeypatch, synthetic_lock):
    _skip_unless_published_tree_in_this_clone()
    # plan_item_ids plans the MATH-500 item set through inspect_served -> inspect_evals, the tinycap client venv's dependency (its hashed
    # lock), not the dev/cpu venv's: absent, the preflight's own Refusal fires for a reason that is this venv's, not the record's
    # (the cluster gate venv, 2026-10-08) — the precedent of test_inspect_served_verification_phase / test_inspect_served_thinking_on
    pytest.importorskip("inspect_evals", reason="tinycap client venv (inspect_evals rides the hashed lock); the preflight plans MATH-500 through it")
    _bound(monkeypatch)
    key = "gemma4e4b"
    _, _, tok = P.load_tokenizer_for(key)
    from sisteer.models import chat_template_hash
    tsha = chat_template_hash(tok)
    ids = P.plan_item_ids()
    assert len(ids) == 500
    _write_published(tmp_path / "pub", key, _synthetic_manifest(key, template_hash=tsha, ids=ids), ids=ids)
    args = ["--published-root", str(tmp_path / "pub"), "--wave-sha", SHA, "--out", str(tmp_path / "out"), "--model-key", key]
    rc = P.main(args)
    out = capsys.readouterr().out
    assert rc == 0 and "VERDICT: PASS" in out, out[-2000:]
    rec = json.loads((tmp_path / "out" / f"compat_{key}.json").read_text())
    assert rec["schema"] == P.SCHEMA and rec["verdict"] == "PASS" and rec["problems"] == [] and rec["rows"] == [24, 25, 26]
    assert rec["item_ids_source"].startswith("planned offline")
    # fix A2: the certificate names the tree it read — the resolved HEAD beside the argv sha, equal, clean evidence roots
    assert rec["tree"]["resolved_head"] == rec["tree"]["head"] == SHA == rec["wave_sha"] and rec["tree"]["head_equals_wave_sha"] is True
    assert rec["tree"]["dirty_paths"] == [] and rec["tree"]["evidence_roots"] == ["src", "scripts", "configs"]
    for idx in ("24", "25", "26"):
        assert rec["shared"][idx]["item_ids"]["equal"] and rec["shared"][idx]["chat_template_hash"]["equal"]
        ss = rec["server_settings"][idx]
        assert ss["compared"]["chat_template_hash"]["equal"] and ss["compared"]["reasoning_parser"] == {"published": "gemma4", "wave": "gemma4", "equal": True}
        assert ss["compared"]["reduction_mode"]["equal"] and ss["compared"]["enable_thinking"]["equal"] and ss["compared"]["thinking_header_ids"]["equal"]
        assert "--max-model-len" in ss["excluded_as_intentional"] and "--enable-log-requests" in ss["excluded_as_intentional"]
        assert ss["composed_only"]["--dtype"] == "bfloat16" and "default_chat_template_kwargs" in ss["composed_only"]
        # fix A4: composed under the JOB's launch configuration — server default OFF (the composer's OFF pin), the request
        # flag ON, the two kept apart; the memory utilisation and --enable-log-requests spliced as the standup splices them
        assert ss["compared"]["enable_thinking"] == {"published": True, "wave": True, "equal": True, "level": ss["compared"]["enable_thinking"]["level"]}
        sd = ss["compared"]["server_default_thinking"]
        assert sd["published"] == "off" == sd["wave"] and sd["equal"] is True and "git show" in sd["published_source"]
        assert ss["composed_only"]["default_chat_template_kwargs"] == '{"enable_thinking": false}' and ss["composed_only"]["gpu_memory_utilization"] == "0.9"
        assert ss["excluded_as_intentional"]["--enable-log-requests"]["value"] is True
        tp = ss["thinking_policy"]
        assert tp["server_default"] == "off" and tp["request_mode"] == "on" and tp["request_enable_thinking"] is True and tp["effective_render"] == "on"
        assert tp["launch_config"]["thinking_on"] is False and tp["launch_config"]["mode"] == "attacks" and tp["launch_config"]["cell"] == M.M5T_CELLS[int(idx)]["cell"]
        assert rec["sampler_resolution"][idx]["omitted_keys"] == ["min_p", "presence_penalty", "repetition_penalty"]
    assert rec["undetermined"] == [], rec["undetermined"]        # this machine's cache carries gemma's generation_config.json
    assert rec["arm_contracts"]["24"]["geometry"] == {"eval": "capability:gen_long_math_longsi", "L_row": 24064, "L_resolved": 24064, "S": 24,
                                                       "budget": rec["arm_contracts"]["24"]["geometry"]["budget"]}
    assert any("sampling_ref" in s for s in rec["not_carried_by_published_record"]) and rec["skipped_checks"] == []
    # fix A1: NEITHER diagnostic switch can PASS — a skipped check is UNDETERMINED (rc 3), the published ids are never
    # echoed as the wave's and the published sampler rule is never reused as the wave's
    rc = P.main([*args, "--no-plan-items"])
    out = capsys.readouterr().out
    assert rc == 3 and "VERDICT: UNDETERMINED" in out and "VERDICT: PASS" not in out, out[-1500:]
    rec = json.loads((tmp_path / "out" / f"compat_{key}.json").read_text())
    assert rec["verdict"] == "UNDETERMINED" and rec["problems"] == [] and rec["skipped_checks"] == ["item population (--no-plan-items)"]
    assert rec["item_ids_source"].startswith("NOT planned") and "UNDETERMINED" in rec["item_ids_source"]
    for idx in ("24", "25", "26"):
        assert rec["shared"][idx]["item_ids"]["equal"] is None and rec["shared"][idx]["item_ids"]["wave_n"] is None
        assert f"row {idx} {P.ITEM_IDS_NOT_PLANNED}" in rec["undetermined"]
        assert rec["server_settings"][idx]["compared"]["chat_template_hash"]["equal"]       # everything else still compared
    rc = P.main([*args, "--no-compose"])
    out = capsys.readouterr().out
    assert rc == 3 and "VERDICT: UNDETERMINED" in out and "VERDICT: PASS" not in out, out[-1500:]
    rec = json.loads((tmp_path / "out" / f"compat_{key}.json").read_text())
    assert rec["verdict"] == "UNDETERMINED" and rec["problems"] == [] and rec["skipped_checks"] == ["server settings + the wave's sampler rules (--no-compose)"]
    for idx in ("24", "25", "26"):
        assert rec["server_settings"][idx] == {"composed": False, "undetermined": P.SERVER_SETTINGS_NOT_COMPOSED}
        assert f"row {idx} {P.SERVER_SETTINGS_NOT_COMPOSED}" in rec["undetermined"]
        res = rec["sampler_resolution"][idx]
        assert res["composed"] is False and res["omitted_keys"] == ["min_p", "presence_penalty", "repetition_penalty"]
        assert all(v["equal"] is None and v["wave"] is None and v["wave_rule"] == P.WAVE_RULE_NOT_COMPOSED for v in res["per_key"].values())
        assert rec["shared"][idx]["item_ids"]["equal"] is True                              # the population WAS planned here
    rc = P.main([*args, "--no-compose", "--no-plan-items"])
    assert rc == 3 and "VERDICT: UNDETERMINED" in capsys.readouterr().out
    # one moved field: FAIL by name (rc 1), the record says which
    _write_published(tmp_path / "pub-moved", key, _synthetic_manifest(key, template_hash=tsha, ids=ids, **{"sampling.top_p": 0.5}), ids=ids)
    rc = P.main([*args[:1], str(tmp_path / "pub-moved"), *args[2:]])
    out = capsys.readouterr().out
    assert rc == 1 and "VERDICT: FAIL" in out
    rec = json.loads((tmp_path / "out" / f"compat_{key}.json").read_text())
    assert rec["verdict"] == "FAIL" and len(rec["problems"]) == 6       # 3 rows x (the numeric projection + the ref)
    assert {p.split(":")[0].split(" ", 2)[-1] for p in rec["problems"]} == {"shared.sampling_params", "shared.sampling_ref"}
    # no published manifest: a REFUSAL (rc 2), never a skipped field
    (tmp_path / "pub-empty").mkdir()
    rc = P.main([*args[:1], str(tmp_path / "pub-empty"), *args[2:]])
    assert rc == 2 and "REFUSED:" in capsys.readouterr().out
    # the CLI's own guards
    with pytest.raises(SystemExit, match="40-hex"):
        P.main([*args[:3], "abc", *args[4:]])
    with pytest.raises(SystemExit, match="no rows"):
        P.main([*args, "--wave", "nope"])
    with pytest.raises(SystemExit, match="no row of wave"):
        P.main([*args[:-1], "glm46vflash"])


@pytest.mark.skipif(not GRADE_PUBLISHED.is_dir() or not _tokenizers_cached(), reason="the published pull / the tokenizers are not on this machine")
def test_the_real_published_pull_is_compatible_with_this_tree_for_the_three_extension_models(tmp_path, capsys):
    """C1's record on the local workstation: the published legs' manifests at tree 55e3cbdd vs what THIS tree would serve for rows
    24-32 — PASS for gemma / qwen / nemotron (the numbers behind A8: qwen omits only repetition_penalty, 1.0 under both
    rules; gemma / nemotron ship a generation_config.json, identical under both trees). Bound to THIS checkout's real
    HEAD (fix A2): the run is skipped, never faked, while the evidence roots carry uncommitted changes."""
    _skip_unless_published_tree_in_this_clone()
    pytest.importorskip("inspect_evals", reason="tinycap client venv (inspect_evals rides the hashed lock); the preflight plans MATH-500 through it")
    facts = P.tree_facts(P.REPO)
    if facts["dirty_paths"]:
        pytest.skip(f"uncommitted changes under {P.EVIDENCE_ROOTS}: {facts['dirty_paths']} — the certificate binds to a clean tree (A2)")
    sha = facts["head"]
    rc = P.main(["--published-root", str(GRADE_PUBLISHED), "--wave-sha", sha, "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "VERDICT: PASS" in out, out[-3000:]
    for key in M.EXT_MODELS:
        rec = json.loads((tmp_path / f"compat_{key}.json").read_text())
        assert rec["verdict"] == "PASS" and rec["published_tree"] == P.PUBLISHED_TREE
        assert rec["tree"]["resolved_head"] == sha == rec["wave_sha"] and rec["tree"]["head_equals_wave_sha"] is True
        idx = str(M.row_for(key, M.CELL_LORA_MIX, M.SAMPLING_THINKING_CARD)["index"])
        assert rec["published_population"]["n_attempted_items"] == 500 and rec["shared"][idx]["item_ids"]["equal"]
        assert rec["published_sampling_ref_record"]["self_consistent"] and rec["shared"][idx]["sampling_ref"]["equal"]
        assert rec["shared"][idx]["sampling_ref"]["published"] == {"gemma4e4b": "83215617d18b", "qwen35_4b": "89a93d3c284f",
                                                                   "nemotron3nano4b": "4fc6d968b926"}[key]
        omitted = rec["sampler_resolution"][idx]
        assert all(v["equal"] for v in omitted["per_key"].values()) and rec["undetermined"] == []
        if key == "qwen35_4b":
            assert omitted["omitted_keys"] == ["repetition_penalty"] and omitted["snapshot_generation_config"]["status"] == P.SNAPSHOT_NO_EXIST
            assert omitted["override_generation_config"] is not None
        else:
            assert omitted["snapshot_generation_config"]["status"] == P.SNAPSHOT_CACHED and omitted["override_generation_config"] is None


# --------------------------------------------------------------------------- the binding (fix A2)


def test_the_certificate_is_bound_to_the_checkout_it_read(tmp_path, capsys, monkeypatch):
    """Fix A2: `--wave-sha` was only hex-checked while every input came from the current checkout, so a certificate could
    name a tree the launcher never runs. Now the checkout's HEAD is resolved independently (`git rev-parse HEAD`), must
    equal the argv sha, and the evidence roots (src/ scripts/ configs/) must carry no uncommitted tracked change —
    refused (rc 2) BEFORE anything is written; changes outside the roots are recorded, not refused; a copy that is not a
    checkout cannot be certified."""
    import os
    import subprocess
    facts = P.tree_facts(P.REPO)
    head = subprocess.run(["git", "-C", str(P.REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert facts["head"] == head and M.is_full_sha(head) and facts["evidence_roots"] == list(P.EVIDENCE_ROOTS) == ["src", "scripts", "configs"]
    assert isinstance(facts["dirty_paths"], list) and isinstance(facts["dirty_paths_outside_evidence_roots"], list)
    # a synthetic checkout: clean binds; a tracked edit under docs/ is recorded; one under src/ refuses; a foreign sha refuses
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "src" / "x.py").write_text("x = 1\n")
    (repo / "docs" / "y.md").write_text("y\n")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid", "GIT_CONFIG_GLOBAL": "/dev/null"}

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True, env=env).stdout
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "--no-verify", "-m", "init")
    sha = git("rev-parse", "HEAD").strip()
    clean = P.tree_facts(repo)
    assert clean["head"] == sha and clean["dirty_paths"] == [] and clean["dirty_paths_outside_evidence_roots"] == []
    bound = P.bind_wave_sha(sha, clean)
    assert bound["head_equals_wave_sha"] is True and bound["resolved_head"] == sha == bound["wave_sha"] and "binding" in bound
    (repo / "docs" / "y.md").write_text("changed\n")
    outside = P.tree_facts(repo)
    assert outside["dirty_paths"] == [] and outside["dirty_paths_outside_evidence_roots"] == ["docs/y.md"]
    assert P.bind_wave_sha(sha, outside)["dirty_paths_outside_evidence_roots"] == ["docs/y.md"]       # recorded, never refused
    (repo / "src" / "x.py").write_text("x = 2\n")
    dirty = P.tree_facts(repo)
    assert dirty["dirty_paths"] == ["src/x.py"]
    with pytest.raises(P.Refusal) as ei:          # a Refusal's str() is its exit code (2); the reason is `.why`
        P.bind_wave_sha(sha, dirty)
    assert ei.value.code == 2 and "uncommitted changes under the evidence roots" in ei.value.why and "src/x.py" in ei.value.why
    with pytest.raises(P.Refusal) as ei:
        P.bind_wave_sha("0" * 40, clean)
    assert ei.value.code == 2 and f"--wave-sha {'0' * 40} is not this checkout's HEAD {sha}" in ei.value.why
    (tmp_path / "plain").mkdir()
    with pytest.raises(P.Refusal) as ei:
        P.tree_facts(tmp_path / "plain")
    assert ei.value.code == 2 and "must run from the git checkout" in ei.value.why
    # the CLI refuses BEFORE writing anything: a mismatched sha, then a dirty evidence root (rc 2, REFUSED:, no out dir)
    args = ["--published-root", str(tmp_path), "--wave-sha", SHA, "--out", str(tmp_path / "out"), "--model-key", "gemma4e4b"]
    monkeypatch.setattr(P, "tree_facts", lambda repo, evidence_roots=P.EVIDENCE_ROOTS: {**clean, "head": "1" * 40})
    rc = P.main(args)
    out = capsys.readouterr().out
    assert rc == 2 and "REFUSED:" in out and f"--wave-sha {SHA} is not this checkout's HEAD {'1' * 40}" in out and not (tmp_path / "out").exists()
    monkeypatch.setattr(P, "tree_facts", lambda repo, evidence_roots=P.EVIDENCE_ROOTS: {
        **clean, "head": SHA, "dirty_paths": ["scripts/sdfbase/math500/cells.py"]})
    rc = P.main(args)
    out = capsys.readouterr().out
    assert rc == 2 and "uncommitted changes under the evidence roots" in out and "scripts/sdfbase/math500/cells.py" in out
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------- the launch configuration (fix A4)


def test_the_launch_configuration_is_the_jobs_own_and_the_two_thinking_policies_are_kept_apart(tmp_path, monkeypatch):
    """Fix A4: the preflight composed `thinking_on=True` while the sbatch never sets F22_THINKING_ON (the standup's default
    OFF server policy, the driver's request-level ON). The launch configuration now comes from the launcher's OWN dry run
    (STANDUP extras + STANDUP_POLICY) for every row — parity with the production standup args — the sbatch REFUSES the
    variable f22_standup reads, and the published tree's server-default policy is read off its launcher (git show)."""
    _skip_unless_published_tree_in_this_clone()
    import subprocess
    for row in M.EXTENSION_ROWS:
        cfg = P.launch_config_from_dry_run(row, 24, tmp_path / row["job_id"])
        assert cfg["mode"] == "attacks" and cfg["thinking_on"] is False and cfg["server_default_thinking"] == "off"
        assert cfg["request_thinking_mode"] == row["thinking_mode"] == "on" and cfg["enable_log_requests"] is True
        assert cfg["reasoning_parser"] == row["reasoning_parser"] and cfg["cell"] == row["cell"] and cfg["gpu_mem_util"] == row["gpu_mem_util"]
        assert cfg["expect_object_sha256"] == (f"{row['cell']}={row['object_sha256']}" if row["fitted"] else None)
        assert cfg["max_logprobs"] is None and cfg["upstream_timeout_s"] == str(M.UPSTREAM_TIMEOUT_S) and cfg["lens_salt_policy"] == "attested"
        assert cfg["standup_argv"][:4] == ["--upstream-timeout-s", str(M.UPSTREAM_TIMEOUT_S), "--lens-salt-policy", "attested"]
    # the production standup: f22_standup composes --thinking-on iff F22_THINKING_ON; mt_serving_up composes thinking_on from
    # that flag and tool_calls from the mode — the configuration the preflight now reads is exactly what those two consume
    common = (REPO / "scripts/sdfbase/full22/common.sh").read_text()
    mt = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()
    assert "${F22_THINKING_ON:+--thinking-on}" in common and "thinking_on=args.thinking_on" in mt and 'tool_calls=(args.mode == "petri")' in mt
    assert "F22_THINKING_ON" not in (REPO / "scripts/sdfbase/math500/math500_thinking.sbatch").read_text()
    # the variable f22_standup reads is refused by the launcher (a batch process inherits the submitter's exported environment)
    monkeypatch.setenv("F22_THINKING_ON", "1")
    with pytest.raises(SystemExit) as ei:
        P.launch_config_from_dry_run(M.EXTENSION_ROWS[0], 24, tmp_path / "guard")
    assert "F22_THINKING_ON is set" in str(ei.value)
    monkeypatch.delenv("F22_THINKING_ON")
    # the published tree's server-default policy, read off ITS launcher: no mechanism at all -> the composer's default OFF
    pol = P.published_server_default_thinking()
    assert pol["policy"] == "off" and pol["launcher_has_thinking_on_mechanism"] is False and not any(pol["sets_F22_THINKING_ON"].values())
    assert pol["tree"] == P.PUBLISHED_TREE and set(pol["sets_F22_THINKING_ON"]) == set(P.LAUNCHER_FILES)
    bad = P.published_server_default_thinking("0" * 40)
    assert bad["policy"] is None and "cannot be read here" in bad["undetermined"]
    # THIS tree's launcher carries the mechanism (f22_standup) and never sets the variable
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    here = P.published_server_default_thinking(head)
    assert here["policy"] == "off" and here["launcher_has_thinking_on_mechanism"] is True


@pytest.mark.skipif(not _tokenizers_cached(), reason="the pinned tokenizers are not in the local HF cache (offline)")
def test_the_composed_launch_keeps_the_server_default_and_the_request_thinking_policy_apart(tmp_path):
    """Fix A4, composed for real: under the job's configuration the composer pins the OFF server default while the
    request flag compares ON == ON; a configuration with --thinking-on moves ONLY the server-default field (named), never
    the request field; an unreadable published tree makes the server-default field UNDETERMINED, never equal."""
    _skip_unless_published_tree_in_this_clone()
    from sisteer.capeval.serving import vllm_server as vs
    from sisteer.models import chat_template_hash
    key = "gemma4e4b"
    row = M.row_for(key, M.CELL_LORA_MIX, M.SAMPLING_THINKING_CARD)
    _, _, tok = P.load_tokenizer_for(key)
    _, _, _, man_path = P.overlay(key)
    geo = P.resolve_geometry(row)
    cfg = P.launch_config_from_dry_run(row, geo["S"], tmp_path)
    composed = P.compose_launch(row, key, man_path, tok, geo, tmp_path, cfg)
    assert composed["default_chat_template_kwargs"] == vs.THINKING_OFF_TEMPLATE_KWARGS and composed["server_default_thinking"] == "off"
    assert composed["gpu_memory_utilization"] == row["gpu_mem_util"] and composed["excluded"]["--enable-log-requests"]["value"] is True
    assert composed["flags"]["--reasoning-parser"] == "gemma4" and composed["launch_config"] is cfg
    pub_man = _synthetic_manifest(key, template_hash=chat_template_hash(tok))
    pol = P.published_server_default_thinking()
    rec, problems, undet = P.compare_server_settings(pub_man, composed, row, pol)
    assert problems == [] and undet == []
    assert rec["compared"]["enable_thinking"]["published"] is True and rec["compared"]["enable_thinking"]["wave"] is True
    assert rec["compared"]["server_default_thinking"]["published"] == "off" == rec["compared"]["server_default_thinking"]["wave"]
    assert rec["thinking_policy"]["server_default"] == "off" and rec["thinking_policy"]["effective_render"] == "on"
    # a standup-wide --thinking-on (never this lane's) moves the SERVER-DEFAULT field alone: named, the request field untouched
    on = P.compose_launch(row, key, man_path, tok, geo, tmp_path, {**cfg, "thinking_on": True})
    assert on["default_chat_template_kwargs"] == vs.THINKING_ON_TEMPLATE_KWARGS and on["server_default_thinking"] == "on"
    rec, problems, undet = P.compare_server_settings(pub_man, on, row, pol)
    assert problems == ["server_settings.server_default_thinking: published 'off' != wave 'on'"] and undet == []
    assert rec["compared"]["enable_thinking"]["equal"] is True and rec["compared"]["server_default_thinking"]["equal"] is False
    # the published tree unreadable: UNDETERMINED by name, never equal
    rec, problems, undet = P.compare_server_settings(pub_man, composed, row, P.published_server_default_thinking("0" * 40))
    assert problems == [] and len(undet) == 1 and undet[0].startswith("server_settings.server_default_thinking:")
    assert rec["compared"]["server_default_thinking"]["equal"] is None


# --------------------------------------------------------------------------- the client lock by what the client IMPORTS (L1 G0)


def _lock_text_at(tree: str) -> str:
    text, why = P.CL.lock_text_at_tree(REPO, tree)
    assert text is not None, why
    return text


def _without_blocks(text: str, names: set[str]) -> str:
    """The lock text with the requirement blocks of `names` removed and their `# via` lines dropped — the shape of
    the tree BEFORE they were added (a synthetic reconstruction; the real pair has its own test)."""
    out, skip = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith((" ", "\t")):
            if skip:
                continue
            if any(line.strip() == f"#   {n}" or line.strip() == f"# via {n}" for n in names):
                continue
            out.append(line)
            continue
        skip = any(line.startswith(f"{n}==") or line.startswith(f"{n} @ ") for n in names)
        if not skip:
            out.append(line)
    return "".join(out)


def test_the_client_entry_points_are_the_launchers_own_and_the_walk_resolves_this_trees_imports():
    """The compared set is DERIVED from the client venv's MATH-500 entry points, each of which the launcher's own text
    names: the driver and the ledger (served_thinking.sh), the recorder (cells.py's literal, m5t_common.sh) and the
    identity stamp (m5t_common.sh runs cells.py from the client venv). The walk over THIS tree resolves every third-party
    import either to a lock distribution or to the disclosed not-in-lock set; the alias table only ever maps onto a lock
    entry (`yaml` -> pyyaml, `absl` -> absl-py)."""
    CL = P.CL
    core = (REPO / "scripts/sdfbase/full22/served_thinking.sh").read_text()
    common = (REPO / "scripts/sdfbase/math500/m5t_common.sh").read_text()
    assert set(CL.CLIENT_ENTRY_POINTS) == {"scripts/served_capability_run.py", "src/sisteer/capeval/serving/usage_ledger.py",
                                          "scripts/sdfbase/record_resolved_sampler.py", "scripts/sdfbase/math500/cells.py"}
    assert '"$CPY" scripts/served_capability_run.py generate' in core and '"$CPY" -m sisteer.capeval.serving.usage_ledger' in core
    assert M.RESOLVED_SAMPLER_SCRIPT_REL == "scripts/sdfbase/record_resolved_sampler.py" and '"$REPO_DIR/$M5T_RESOLVED_SAMPLER_SCRIPT"' in common
    assert '"$REPO_DIR/$M5T_DIR/cells.py" row --index "$M5T_INDEX" --identity' in common and 'M5T_DIR="scripts/sdfbase/math500"' in common
    assert all((REPO / e).is_file() for e in CL.CLIENT_ENTRY_POINTS) and CL.CLIENT_IMPORT_ROOTS == ("src", "scripts", "scripts/sdfbase", "scripts/sdfbase/math500", "scripts/sdfbase/full22")
    walk = CL.walk_client_imports(REPO)
    assert walk.missing_entry_points == [] and walk.unparseable == {} and walk.unresolved_relative == [] and len(walk.modules) > 100
    assert {"inspect_ai", "inspect_evals", "yaml", "torch", "transformers", "aiohttp", "datasets"} <= set(walk.third_party)
    assert "scripts/served_capability_run.py" in walk.third_party["aiohttp"] and "src/sisteer/capeval/tiers/inspect_served.py" in walk.third_party["inspect_ai"]
    lock = CL.parse_lock((REPO / C.CLIENT_LOCK_TXT).read_text())
    assert lock["inspect-ai"].version == C.INSPECT_AI_PIN and lock["inspect-evals"].version == C.INSPECT_EVALS_PIN and "inspect-ai" in lock["ijson"].via
    ife = "instruction-following-eval"
    assert lock[ife].vcs and lock[ife].version is None and lock[ife].spec.startswith("@ git+")
    assert "tinybenchmarks" not in lock and "lm-eval" not in lock
    assert CL.resolve_import("yaml", (lock,)) == {"distributions": ["pyyaml"], "resolver": CL.RESOLVER_ALIAS, "candidates": ["pyyaml"]}
    assert CL.resolve_import("absl", (lock,))["distributions"] == [] and CL.resolve_import("inspect_ai", (lock,))["resolver"] == CL.RESOLVER_NORMALISED
    assert CL.resolve_import("vllm", (lock,), {"vllm": ["vllm"]}) == {"distributions": [], "resolver": None, "candidates": ["vllm"]}
    assert CL.resolve_import("PIL", (lock,), {"PIL": ["pillow"]})["resolver"] in (CL.RESOLVER_ALIAS, CL.RESOLVER_INTERPRETER) or "pillow" not in lock
    rec, problems = CL.compare_locks((REPO / C.CLIENT_LOCK_TXT).read_text(), (REPO / C.CLIENT_LOCK_TXT).read_text(), walk)
    assert problems == [] and rec["one_instrument"] is True and rec["sha256"]["equal"] is True and rec["n_compared"] >= 20
    assert {"inspect-ai", "inspect-evals", "pyyaml", "torch", "scipy", "transformers"} <= set(rec["imported_packages"])
    # every third-party name is accounted for: a compared / added / removed distribution, or disclosed as outside both locks
    accounted = {n for r in rec["imported_packages"].values() for n in r["import_names"]} | set(rec["imports_not_in_lock"])
    assert accounted == set(walk.third_party)
    # `petri_bloom`: reached through the request-evidence client (`sisteer.eval.request_evidence_client`, wired into the client entry
    # `served_capability_run.py`), whose GUARDED `from sisteer.capeval.tiers.petri_runner import …` the walk follows by design (step 1:
    # an over-approximation; a `try: import` counts) into the Petri runner's `import petri_bloom` — the same class as `pyrit` through
    # pyrit_adapters.py: attack tooling the MATH-500 client venv never installs, DISCLOSED, never refused
    assert set(rec["imports_not_in_lock"]) <= {"matplotlib", "petri_bloom", "pynvml", "pyrit", "vllm", "vllm_lens", "wandb"}, rec["imports_not_in_lock"]


def test_a_doctored_lock_bumping_an_imported_package_refuses_by_name_and_a_non_imported_bump_is_disclosed(monkeypatch, tmp_path):
    """The brief's condition (2), first half: a lock that moves the version of a distribution the MATH-500 client imports
    (inspect-ai 0.3.260 -> 0.3.261) REFUSES by name — through the module and through the preflight's shared comparison
    (`shared.client_lock.inspect-ai`, `equal: False`, rc 1 from the CLI's point of view); a bump of a distribution the client
    never imports (swebench) is DISCLOSED under `outside_imported_set.changed` and refuses nothing."""
    CL = P.CL
    key = "qwen35_4b"
    wave_text = (REPO / C.CLIENT_LOCK_TXT).read_text()
    walk = CL.walk_client_imports(REPO)
    assert "inspect-ai==0.3.260" in wave_text and "swebench==5.0.2" in wave_text
    bumped = wave_text.replace("inspect-ai==0.3.260", "inspect-ai==0.3.261", 1)
    rec, problems = CL.compare_locks(wave_text, bumped, walk)
    assert problems == [p for p in problems if p.startswith("client_lock.inspect-ai: published 0.3.260 != wave 0.3.261")] and len(problems) == 1
    assert rec["one_instrument"] is False and rec["equal"] is False and rec["changed"] == ["inspect-ai"] and rec["sha256"]["equal"] is False
    assert rec["imported_packages"]["inspect-ai"] == {"published": "0.3.260", "wave": "0.3.261", "equal": False, **rec["imported_packages"]["inspect-ai"]}
    assert "TWO instruments" in rec["disclosure"]
    other = wave_text.replace("swebench==5.0.2", "swebench==5.0.3", 1)
    rec, problems = CL.compare_locks(wave_text, other, walk)
    assert problems == [] and rec["one_instrument"] is True and rec["outside_imported_set"]["changed"] == {"swebench": {"published": "5.0.2", "wave": "5.0.3"}}
    assert "swebench" not in rec["imported_packages"]
    # through the preflight: the WAVE side is this tree's lock (facts), the PUBLISHED side's text is the doctored one
    P._LOCK_RELATION_CACHE.clear()
    monkeypatch.setattr(P, "published_lock_text", lambda tree: (bumped, None))
    lock = P.client_lock_facts()
    row = M.row_for(key, M.CELL_LORA_MIX, M.SAMPLING_THINKING_CARD)
    man = _synthetic_manifest(key, template_hash="ab" * 8, **{"frameworks.client_lock_sha256": CL.lock_sha256(bumped)})
    mp = _write_published(tmp_path / "pub", key, man)
    rec, problems, undetermined = P.compare_shared(P.published_shared(json.loads(mp.read_text()), mp), P.wave_shared(row, key, _cfg(key), "ab" * 8, IDS, lock), lock)
    assert undetermined == [] and len(problems) == 1 and problems[0].startswith("shared.client_lock.inspect-ai: published 0.3.261 != wave 0.3.260")
    assert rec["client_lock"]["equal"] is False and rec["client_lock"]["published"]["matches_record"] is True and rec["client_lock"]["changed"] == ["inspect-ai"]
    assert all(v["equal"] for f, v in rec.items() if f != "client_lock")
    P._LOCK_RELATION_CACHE.clear()
    # the non-imported bump through the preflight: no problem, disclosed
    monkeypatch.setattr(P, "published_lock_text", lambda tree: (other, None))
    man = _synthetic_manifest(key, template_hash="ab" * 8, **{"frameworks.client_lock_sha256": CL.lock_sha256(other)})
    mp = _write_published(tmp_path / "pub2", key, man)
    rec, problems, undetermined = P.compare_shared(P.published_shared(json.loads(mp.read_text()), mp), P.wave_shared(row, key, _cfg(key), "ab" * 8, IDS, lock), lock)
    assert problems == [] and undetermined == [] and rec["client_lock"]["equal"] is True and rec["client_lock"]["sha256"]["equal"] is False
    assert rec["client_lock"]["outside_imported_set"]["changed"] == {"swebench": {"published": "5.0.3", "wave": "5.0.2"}}
    P._LOCK_RELATION_CACHE.clear()
    # an unreadable published tree: UNDETERMINED by name, `equal: None`, never a pass; a record without a digest: a problem
    monkeypatch.setattr(P, "published_lock_text", lambda tree: (None, "git show failed (synthetic)"))
    rec, problems, undetermined = P.compare_shared(P.published_shared(json.loads(mp.read_text()), mp), P.wave_shared(row, key, _cfg(key), "ab" * 8, IDS, lock), lock)
    assert problems == [] and len(undetermined) == 1 and undetermined[0].startswith("shared.client_lock: the published tree's lock cannot be read here")
    assert rec["client_lock"]["equal"] is None and rec["client_lock"]["published"]["readable"] is False
    P._LOCK_RELATION_CACHE.clear()
    rec_l, problems, undetermined = P.compare_client_lock({"sha256": None, "tree": P.PUBLISHED_TREE}, {"sha256": lock["client_lock_sha256"]}, lock)
    assert undetermined == [] and len(problems) == 1 and problems[0].startswith("shared.client_lock: the published record carries no client lock digest") and rec_l["equal"] is False
    P._LOCK_RELATION_CACHE.clear()


def test_the_pr80_diff_admits_synthetically_and_against_the_real_published_tree():
    """A commit added `immutabledict`, `langdetect` and the `instruction-following-eval` VCS pin to the lock (the
    IFEval checker's dependencies) and moved no existing version, so the digest moved (0f21fd4d… -> 910b4851…) while
    the MATH-500 client's imported distributions are all at the same version. Synthetically: the wave lock with those
    three blocks removed is the published shape, and the comparison ADMITS with the disclosure naming them (two of
    them are reached by the walk through the vendored IFEval checker's guarded imports — `imported_added`, not a
    version change; the VCS pin is outside the imported set). For real, when the published tree is in this clone:
    `git show 55e3cbdd:<lock>` digests EXACTLY the value every published MATH-500 leg recorded, and the comparison
    with THIS tree's lock admits with the same disclosure."""
    CL = P.CL
    wave_text = (REPO / C.CLIENT_LOCK_TXT).read_text()
    walk = CL.walk_client_imports(REPO)
    added = {"immutabledict", "instruction-following-eval", "langdetect"}
    synthetic_published = _without_blocks(wave_text, added)
    assert synthetic_published != wave_text and all(f"{n}==" not in synthetic_published and f"{n} @ " not in synthetic_published for n in added)
    rec, problems = CL.compare_locks(synthetic_published, wave_text, walk)
    assert problems == [] and rec["one_instrument"] is True and rec["equal"] is True and rec["changed"] == [] and rec["sha256"]["equal"] is False
    assert set(rec["imported_added"]) | set(rec["outside_imported_set"]["added"]) == added
    assert rec["imported_removed"] == {} and rec["outside_imported_set"]["removed"] == [] and rec["outside_imported_set"]["changed"] == {}
    assert "ONE instrument" in rec["disclosure"] and "lock digest moved" in rec["disclosure"]
    # the real pair, when this clone carries the published tree
    _skip_unless_published_tree_in_this_clone()
    published = _lock_text_at(P.PUBLISHED_TREE)
    assert CL.lock_sha256(published) == "0f21fd4d92dc56b45e966ddf666463d268e547c82ba6bf54b2e9b1fffd47cc0f"     # every published leg's recorded digest
    rec, problems = CL.compare_locks(published, wave_text, walk)
    assert problems == [] and rec["one_instrument"] is True and rec["changed"] == []
    if CL.lock_sha256(wave_text) != CL.lock_sha256(published):        # THIS tree is post-#80 (main since e8b6d13d)
        assert set(rec["imported_added"]) | set(rec["outside_imported_set"]["added"]) >= added
        assert rec["sha256"]["published"] == "0f21fd4d92dc56b45e966ddf666463d268e547c82ba6bf54b2e9b1fffd47cc0f"
    # the preflight's record for the real published side reads the tree's text and holds it to the recorded digest
    P._LOCK_RELATION_CACHE.clear()
    lock = P.client_lock_facts()
    rec_l, problems, undetermined = P.compare_client_lock({"sha256": CL.lock_sha256(published), "tree": P.PUBLISHED_TREE},
                                                          {"sha256": lock["client_lock_sha256"]}, lock)
    assert problems == [] and undetermined == [] and rec_l["equal"] is True and rec_l["published"]["matches_record"] is True
    assert rec_l["published"]["source"].startswith("git show") and rec_l["published"]["tree"] == P.PUBLISHED_TREE
    P._LOCK_RELATION_CACHE.clear()
