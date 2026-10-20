"""The qwen35_35b_a3b model key in the bcbw FULL readers, and the by-name refusal of an UNDECLARED row dir (2026-10-04; `_bcbw_common` module docstring,
THE qwen35_35b_a3b MODEL KEY / AN UNDECLARED ROW DIR REFUSES BY NAME).

An earlier pass read both full mirrors and wrote 0 records for the 35B rows with no refusal: `native/qwen35_35b_a3b-f22s1` and `-f22u0` matched no declared
model key, so discovery skipped them. Two things change, each pinned here on a synthetic mirror (`_bcbw_full_fixtures`: invented ids and shas, `TESTHOST`):

* the table row `qwen35_35b_a3b` -- model_id the registry's model row (= legacy_spellings.yaml's spelling of `qwen35_35b_a3b_s1`), served id and revision =
  models.yaml (= every mirrored manifest), one seed row dir, the appendix object route (no harness roster target, no pins block), no `main` canary, no
  instruct-direct tag (the fits are base fits) -- so a 35B `s1` store in the fixture shape discovers, reads and files on that model row and tag;
* an undeclared row dir -- a model key the table lacks (`glm46vflash-f22s1`), or a declared key under a selection the grammar lacks (the mirror's
  `qwen35_35b_a3b-f22u0`) -- is a GROUP of its own named for the row dir, refused as a WHOLE GROUP by name: `facts["refuse"]` carries the reason (what the
  status reads: the group's `source_status.ok` is 0, `v_source_status` folds the source to MIN(ok) = 0 with the reason under `extra.refuse`, and the paper's
  `lane_guard` reads the source as not integrated -- before 1af18c42d a `refusals` entry alone was an explained empty read the status still filed
  ok = 1 with 'no row refused') beside the same text in `facts["refusals"]`, with nothing emitted; a dot-prefixed entry (`.staging`, publish scaffolding) is
  not a row and is not discovered;
* the LeCun-init ReLU row dir `-f22s2rl` is declared only FOR A READER THAT READS IT (`lecun_relu_row_admitted`: the reader admits the row and the family runs
  it, 2026-10-07): under a 200-item reader or a family that runs no such row it is undeclared for that reader -- the same whole-group refusal -- never a
  by-name row refusal in the s1 group, which left `source_status.ok` at 1 while the s1 rows filed (the review of 797f57df2).
"""
from __future__ import annotations

import importlib
import json
import os
import sys

import pytest
import yaml

import _b1_fixtures as F
import _bcbw_full_fixtures as Z
import _bcbw_seedrep_fixtures as X
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import bfcl_allmodels as BFA
from sisteer.registry.adapters import bfcl_full as BFF
from sisteer.registry.adapters import bfcl_seedrep as BFS
from sisteer.registry.adapters import bigcodebench_allmodels as BCA
from sisteer.registry.adapters import bigcodebench_full as BCF
from sisteer.registry.adapters import bigcodebench_seedrep as BCS
from sisteer.registry.adapters import declared_common as DC
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import runner as RN

MK35, TAG35, NONE, LORA35 = Z.MK35, Z.TAG35, Z.NONE, Z.LORA35
U0_ROW_DIR = f"{MK35}-f22u0"
TAG35_U0 = Z.TAG35_U0
TAG35_DR, SEL35_DR, DR_ROW_DIR = Z.TAG35_DR, Z.SEL35_DR, Z.ROW_DIR35_DR
TAG35_DL, SEL35_DL, DL_ROW_DIR = Z.TAG35_DL, Z.SEL35_DL, Z.ROW_DIR35_DL
TAG35_SR, SEL35_SR, SR_ROW_DIR = Z.TAG35_SR, Z.SEL35_SR, Z.ROW_DIR35_SR
#: a row dir of the 35B that is undeclared on every tree: the bare `-f22` (the family aliases no `main` canary); the u0 row is grammar since 2026-10-05
BARE_ROW_DIR = f"{MK35}-f22"
MODULES = {"bigcodebench": (BCF, BCF.BigcodebenchFullAdapter), "bfcl": (BFF, BFF.BfclFullAdapter)}


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _groups(ad):
    groups = {}
    for i in ad.discover():
        groups.setdefault(ad.comparison_group(i), []).append(i)
    return groups


# --------------------------------------------------------------------------------------------
# the table row: every field against the source it was read from
# --------------------------------------------------------------------------------------------

# The table-row agreement test (the ModelKey against reproduce/models.yaml, the legacy spelling, the instruments and the harness configs)
# was removed with the public scrub: that models.yaml row left the public tree, so the registry row is the standalone source now.


def test_the_grammar_reads_the_35b_s1_spellings_and_keeps_the_mirrors_u0_and_window_spellings_outside(seeds):
    # the alternation tries the longer key first and neither qwen key is a prefix of the other
    assert B.tag_of_row_dir(f"{MK35}-f22s1") == TAG35 and B.tag_of_row_dir(f"{MK35}-f22s1p") == f"{MK35}_s1p" and B.tag_of_row_dir("qwen35_4b-f22s1") == "qwen35_4b_s1"
    assert B.row_dir_of(TAG35) == f"{MK35}-f22s1" and B.seed_tag_of(f"{MK35}_s1p") == TAG35 and B.model_key_of(TAG35) == MK35 and B.tag_parts(TAG35) == (1, False)
    assert B.SUFFIX_RE.match(f"bcbw-{TAG35}-0dba464").group("launch_tag") == TAG35
    assert B.SUFFIX_RE.match(f"bcbw-{TAG35}-55cc447-bfcl").group("task_tag") == "-bfcl" and B.suffix_task(f"bcbw-{TAG35}-55cc447-bfcl") == "bfcl"
    assert B.gate_name_parts(f"gate_{TAG35}_{LORA35}_bcbw-{TAG35}-0dba464.json") == {"tag": TAG35, "arditi": None, "okey": LORA35, "suffix": f"bcbw-{TAG35}-0dba464"}
    # the u0 row (UNSTEERED_ROW_RULE): its tag, row dir, launch tag and gate records read; `-w<N>` is the serving window segment, read beside the task tag
    assert B.tag_of_row_dir(U0_ROW_DIR) == TAG35_U0 and B.row_dir_of(TAG35_U0) == U0_ROW_DIR and B.seed_tag_of(TAG35_U0) == TAG35 and B.model_key_of(TAG35_U0) == MK35
    assert B.is_unsteered_tag(TAG35_U0) and not B.is_unsteered_tag(TAG35) and B.tag_parts(TAG35_U0) is None and B.unsteered_anchor_tag(TAG35_U0) == TAG35
    m = B.SUFFIX_RE.match(f"bcbw-{MK35}_u0-b8086a6-w14336-bfcl")
    assert (m.group("launch_tag"), m.group("window"), m.group("task_tag"), m.group("relaunch")) == (TAG35_U0, "-w14336", "-bfcl", None) and B.suffix_task(m.group(0)) == "bfcl"
    assert B.window_of_suffix(BCF.SPEC, f"bcbw-{MK35}_u0-b8086a6-w14336") == 14336 and B.window_of_suffix(BCF.SPEC, f"bcbw-{MK35}_u0-0dba464-bcb") is None
    assert B.SUFFIX_RE.match(f"bcbw-{MK35}_u0-0dba464-bcb").group("task_tag") == "-bcb" and B.gate_name_parts(f"gate_{MK35}_u0_none_bcbw-{MK35}_u0-0dba464-bcb.json")["tag"] == TAG35_U0
    assert B.row_okey(NONE, None, 14336) == "none@w14336" and B.split_row_okey("none@w14336") == (NONE, None) and B.window_of_row_okey("none@w14336") == 14336 and B.window_of_row_okey(NONE) is None
    assert B.tag_of_row_dir(BARE_ROW_DIR) is None, "a bare -f22 row stays undeclared: the family aliases no main canary"


# --------------------------------------------------------------------------------------------
# a 35B store files on the model's row and tag
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("eval_id", sorted(MODULES))
def test_a_35b_s1_store_in_the_fixture_shape_discovers_reads_and_files_on_the_models_row_and_tag(tmp_path, seeds, eval_id):
    mod, cls = MODULES[eval_id]
    fx = Z.mirror(tmp_path, seeds, eval_id=eval_id, qwen35b_store=True)
    ad = Z.adapter(cls, fx, seeds)
    bt, facts, groups = Z.read_group(ad, TAG35)
    assert groups == [Z.TAG, TAG35] and facts["n_refusals"] == 0 and facts["runs"] == 2 and facts["n_not_run"] == 0, facts["refusals"]
    assert facts["seed"] == 1 and facts["tags"] == sorted([TAG35, f"{MK35}_s1p", TAG35_U0, TAG35_DR, TAG35_DL, TAG35_SR]), "the full reader admits the u0, dose and s2rl rows; this mirror holds none of their stores"
    none_run = Z.run_at(bt, TAG35, NONE)
    lora_run = Z.run_at(bt, TAG35, LORA35)
    inst = seeds.instrument_row(mod.INSTRUMENT_SEED, mod.EVAL_ID, model_id=MK35)["instrument_id"]
    for r in (none_run, lora_run):
        assert r["model_id"] == "qwen35_35b_a3b" and r["view_legacy"] == TAG35 and r["item_set_id"] == mod.ITEM_SET_ID and r["instrument_id"] == inst
        assert r["lane_source_legacy"] == mod.SOURCE and r["seed_base"] is None and r["trial_index"] == 0 and r["decode_seed_rule"] == B.DECODE_SEED_RULE
        assert r["run_id"] == mod.run_id_for(TAG35, fx["keys"][f"{TAG35}/{r['campaign_cell_id_legacy']}"], r["campaign_cell_id_legacy"], inst)
    assert none_run["object_id"] == "none" and lora_run["object_id"] == Z.OBJ35 and lora_run["transfer_kind"] == "base_to_instruct"
    assert facts["rows"][f"{TAG35}|{LORA35}"]["object_id"] == Z.OBJ35 and facts["rows"][f"{TAG35}|{LORA35}"]["staged_id"] == Z.STAGED35
    # the three numbers of the LoRA row from the synthetic truth, and its contrast against the row's own `none`
    value, bound = Z.metrics_of(bt, lora_run["run_id"], mod.recount_computation_id())
    c = X.counts(X.spec(eval_id=eval_id, shape=Z.SHAPE_FULL, **Z.SPECS35[eval_id][LORA35]))
    assert abs(value["value"] - c["acc"]) < 1e-9 and abs(bound["value"] - c["correct"] / Z.N_FULL) < 1e-9
    contrasts = [cn for cn in bt["run_contrast"] if cn["run_id"] == lora_run["run_id"]]
    assert len(contrasts) == 1 and contrasts[0]["anchor_run_id"] == none_run["run_id"] and facts["contrasts"][f"{TAG35}|{LORA35}"]["state"] == B.ANCHOR_CONTRASTED
    # the gemma group beside it reads as it did
    _bt_g, facts_g, _ = Z.read_group(ad, Z.TAG)
    assert facts_g["n_refusals"] == 0 and facts_g["runs"] == 5


def test_a_35b_store_under_a_200_item_reader_refuses_its_group_by_name(tmp_path, seeds):
    # the 200-item readers' tuples do not carry the key: the grammar discovers the store (so it is never skipped) and read() refuses the group by name
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", qwen35b_store=True)
    ad = X.adapter(BCA.BigcodebenchAllmodelsAdapter, fx, seeds)
    groups = _groups(ad)
    assert TAG35 in groups
    recs = [r for batch in ad.read(X.ctx(None), TAG35, groups[TAG35]) for r in batch.records]
    facts = ad.status_extra(TAG35)
    assert recs == [] and facts["runs"] == 0 and len(facts["refusals"]) == 1
    assert facts["refusals"][0].startswith(f"{TAG35} -- model key {MK35!r} is not one of this source's families (gemma4e4b, llama8b, nemotron3nano4b)")


# --------------------------------------------------------------------------------------------
# an undeclared row dir refuses by name
# --------------------------------------------------------------------------------------------

def test_an_undeclared_row_dir_is_a_group_of_its_own_that_refuses_by_name_and_emits_nothing(tmp_path, seeds):
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", undeclared_row_dirs=(Z.UNDECLARED_ROW_DIR, BARE_ROW_DIR), dot_dir_store=True)
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    groups = _groups(ad)
    assert sorted(groups) == sorted([Z.TAG, Z.UNDECLARED_ROW_DIR, BARE_ROW_DIR]), sorted(groups)
    assert os.path.isdir(os.path.join(fx["root"], B.STORES_HEAD, Z.DOT_DIR)) and not any(f"{os.sep}{Z.DOT_DIR}{os.sep}" in i.uri for v in groups.values() for i in v), \
        "a dot-prefixed entry under native/ is publish scaffolding: never a row, never discovered"
    for name, needle in ((Z.UNDECLARED_ROW_DIR, f"row dir {Z.UNDECLARED_ROW_DIR!r} matches no declared model key"),
                         (BARE_ROW_DIR, f"row dir {BARE_ROW_DIR!r} names the declared model key {MK35!r} but no declared row (selection ''")):
        inputs = groups[name]
        assert inputs and all(os.path.relpath(i.uri, fx["root"]).split(os.sep)[:2] == [B.STORES_HEAD, name] for i in inputs), "store inputs alone: no shared record is aliased to it"
        assert any(i.uri.endswith("manifest.json") for i in inputs) and any(os.path.basename(i.uri) == "RESULT" for i in inputs)
        bt, facts, _ = Z.read_group(ad, name)
        assert bt == {} and facts["runs"] == 0 and facts["n_refusals"] == 1 and facts["undeclared_row_dir"] == name and facts["n_store_inputs"] == len(inputs)
        assert facts["refusals"] == [f"{name} -- {B.undeclared_row_dir_reason(name)}"] and needle in facts["refusals"][0]
        # the WHOLE-GROUP refusal the status reads (runner: ok = accepted and no refuse entry): the same reason under `refuse`, never `refusals` alone
        assert facts["refuse"] == [B.undeclared_row_dir_reason(name)] and needle in facts["refuse"][0]
        assert facts["undeclared_row_dir_rule"] == B.UNDECLARED_ROW_DIR_RULE and ad.refusals.items[0]["input"] == os.path.join(fx["root"], B.STORES_HEAD, name)
    # `refuse` is the key the status's ok reads; `refusals` is an explanation key alone (an explained empty read would still file ok = 1)
    assert "refuse" in RN.EXPLANATION_KEYS and "refusals" in RN.EXPLANATION_KEYS
    # the declared group reads as it did
    _bt, facts, _ = Z.read_group(ad, Z.TAG)
    assert facts["n_refusals"] == 0 and facts["runs"] == 5


def test_both_full_readers_report_the_undeclared_row_dir_and_a_declared_key_moves_its_stores_into_the_tag_group(tmp_path, seeds, monkeypatch):
    for eval_id, (_mod, cls) in sorted(MODULES.items()):
        fx = Z.mirror(tmp_path, seeds, eval_id=eval_id, root=str(tmp_path / f"{eval_id}_u"), undeclared_row_dirs=(Z.UNDECLARED_ROW_DIR,))
        ad = Z.adapter(cls, fx, seeds)
        groups = _groups(ad)
        assert Z.UNDECLARED_ROW_DIR in groups
        _bt, facts, _ = Z.read_group(ad, Z.UNDECLARED_ROW_DIR)
        assert facts["n_refusals"] == 1 and facts["undeclared_row_dir"] == Z.UNDECLARED_ROW_DIR
    # declaring the key (a table row for the model segment) makes the same directory a declared row: its inputs leave the row-dir group for the seed-tag group
    fake = B.ModelKey("glm46vflash", "glm46vflash", "synthetic/glm", "0" * 40, (1,), False, False, False)
    monkeypatch.setitem(B.MODEL_KEYS, "glm46vflash", fake)
    alt = "|".join(sorted(B.MODEL_KEYS, key=len, reverse=True))
    # the module's grammar with the new key in the alternation (the seed, instruct-direct and unsteered selections alike)
    monkeypatch.setattr(B, "ROW_DIR_RE", B.re.compile(rf"^(?P<model>{alt})-f22(?:s(?P<seed>\d+)(?P<paired>p?)|(?P<instruct>{B.INSTRUCT_SELECTION})|(?P<unsteered>{B.UNSTEERED_SELECTION})"
                                                      rf"|(?P<dose_relu>{B.DOSE_RELU_SELECTION})|(?P<dose_lora>{B.DOSE_LORA_SELECTION})|(?P<lecun_relu>{B.LECUN_RELU_SELECTION}))$"))
    monkeypatch.setattr(B, "TAG_RE", B.re.compile(rf"^(?P<model>{alt})_(?:s(?P<seed>\d+)(?P<paired>p?)|(?P<instruct>{B.INSTRUCT_SELECTION})|(?P<unsteered>{B.UNSTEERED_SELECTION})"
                                                  rf"|(?P<dose_relu>{B.DOSE_RELU_SELECTION})|(?P<dose_lora>{B.DOSE_LORA_SELECTION})|(?P<lecun_relu>{B.LECUN_RELU_SELECTION}))$"))
    assert B.undeclared_row_dir_reason(Z.UNDECLARED_ROW_DIR) is None and B.tag_of_row_dir(Z.UNDECLARED_ROW_DIR) == "glm46vflash_s1"
    groups = _groups(ad)
    assert Z.UNDECLARED_ROW_DIR not in groups and "glm46vflash_s1" in groups


def test_the_reason_grammar_names_every_case():
    for declared in ("gemma4e4b-f22s1", "gemma4e4b-f22s1p", "gemma4e4b-f22", "qwen35_4b-f22s3p", "qwen35_4b-f22it", f"{MK35}-f22s1"):
        assert B.undeclared_row_dir_reason(declared) is None and not B.is_undeclared_row_dir_group(declared), declared
    r = B.undeclared_row_dir_reason(Z.UNDECLARED_ROW_DIR)
    assert r.startswith(f"row dir {Z.UNDECLARED_ROW_DIR!r} matches no declared model key (the model-key table declares gemma4e4b, llama8b, nemotron3nano4b, qwen35_35b_a3b, qwen35_4b")
    assert B.undeclared_row_dir_reason(U0_ROW_DIR) is None, "the u0 row is grammar (UNSTEERED_ROW_RULE)"
    r = B.undeclared_row_dir_reason(BARE_ROW_DIR)
    assert f"names the declared model key {MK35!r} but no declared row (selection ''; the rows this reader reads for {MK35}: {MK35}-f22s1, {MK35}-f22s1p, {MK35}-f22u0, {MK35}-f22d0p75_relul, {MK35}-f22d1p25, {MK35}-f22s2rl)" in r
    r = B.undeclared_row_dir_reason("qwen35_4b-f22")
    assert "names the declared model key 'qwen35_4b' but no declared row (selection ''" in r and "qwen35_4b-f22it" in r and "qwen35_4b-f22s3p" in r, "the three-seed family: no main alias, its it row listed"
    # a seed OUTSIDE the table's seeds is still grammar (`ROW_DIR_RE` reads any s<N> of a declared key): the directory is the GROUP nemotron3nano4b_s2, which read()
    # refuses by name on the seeds check -- the path that existed before; the undeclared route is for a selection the grammar itself cannot read (u0, a bare -f22)
    assert B.undeclared_row_dir_reason("nemotron3nano4b-f22s2") is None and B.tag_of_row_dir("nemotron3nano4b-f22s2") == "nemotron3nano4b_s2"
    # an appendix family's bare `-f22` row is its `main` canary (legacy_main_alias): declared, aliased to s1 -- never the undeclared route
    assert B.undeclared_row_dir_reason("nemotron3nano4b-f22") is None and B.tag_of_row_dir("nemotron3nano4b-f22") == "nemotron3nano4b_s1"
    # the u0 selection is GRAMMAR for every key (UNSTEERED_ROW_RULE, like the instruct-direct `it`): a nemotron u0 directory is the group nemotron3nano4b_s1, whose read()
    # refuses the store by name (the family declares no unsteered row, so no selection admits the tag) -- never the undeclared route, never a silent skip
    assert B.undeclared_row_dir_reason("nemotron3nano4b-f22u0") is None and B.tag_of_row_dir("nemotron3nano4b-f22u0") == "nemotron3nano4b_u0"
    assert B.seed_tag_of("nemotron3nano4b_u0") == "nemotron3nano4b_s1" and "u0" not in B.group_tags("nemotron3nano4b_s1", True, True)
    r = B.undeclared_row_dir_reason("nemotron3nano4b-f22x1")
    assert "names the declared model key 'nemotron3nano4b' but no declared row (selection 'x1'" in r and "nemotron3nano4b-f22it" not in r and "nemotron3nano4b-f22u0" not in r \
        and "nemotron3nano4b-f22s1p" in r and r.index("nemotron3nano4b-f22s1p") < r.index("nemotron3nano4b-f22)"), "nemotron runs no instruct-direct and no unsteered row; its seed row, paired row and main canary row are listed"
    assert B.undeclared_row_dir_reason("not-a-row").startswith("row dir 'not-a-row' matches no declared model key")
    assert B.is_undeclared_row_dir_group(BARE_ROW_DIR) and B.is_undeclared_row_dir_group(Z.UNDECLARED_ROW_DIR) and not B.is_undeclared_row_dir_group(U0_ROW_DIR)
    assert not B.is_undeclared_row_dir_group(TAG35) and not B.is_undeclared_row_dir_group("gemma4e4b_main") and not B.is_undeclared_row_dir_group("glm46vflash_s1")


# --------------------------------------------------------------------------------------------
# the runner and the paper read the whole-group refusal: v_source_status.ok 0, extra.refuse names the row dir, lane_guard reads not integrated
# --------------------------------------------------------------------------------------------



def _registry_root(tmp_path, seeds, objects) -> str:
    """A scratch registry with the dimension seeds and the fixture's objects folded (the bigcodebench_full module's end-to-end helper)."""
    from sisteer.registry.adapters import dimensions as DIM
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    now = R.utc_now()
    recs = [r for r in LC.standin_dimension_records(seeds, F.producer(), now) if r["record_type"] == "paper_round"]
    recs += DIM.build_records(DIM.load_seeds(seeds.source_dir), F.producer(), now=now, verify_si=False).all_records()
    for o in objects:
        r = LC.standin_object_record(o, seeds, F.producer(), now, "gemma4e4b_base")
        if r:
            recs.append(r)
    R.write_records(os.path.join(root, "records", "standins", "000001-dims.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root


# test_the_undeclared_row_dir_refusal_reads_ok_0_on_v_source_status_with_extra_refuse_and_fails_the_paper_guard: deleted 2026-10-08 — it compared the adapter's refusal rows against the paper library
# (paper/paperlib/registry_source.py + lane_guard.py), which left the tree with the earlier figure build; the adapter's own refusal rows stay pinned above.


def _contrast_of(bt, run_id):
    hits = [cn for cn in bt["run_contrast"] if cn["run_id"] == run_id]
    assert len(hits) <= 1, hits
    return hits[0] if hits else None


@pytest.mark.parametrize("eval_id", sorted(MODULES))
def test_the_u0_row_files_on_its_tag_and_every_row_anchors_on_the_u0_none_at_its_own_serving_window(tmp_path, seeds, eval_id):
    mod, cls = MODULES[eval_id]
    fx = Z.mirror(tmp_path, seeds, eval_id=eval_id, qwen35b_u0=True)
    ad = Z.adapter(cls, fx, seeds)
    bt, facts, groups = Z.read_group(ad, TAG35)
    assert groups == [Z.TAG, TAG35], groups
    assert facts["n_refusals"] == 0 and facts["runs"] == 7 and facts["n_not_run"] == 0, facts["refusals"]
    assert facts["tags"] == sorted([TAG35, f"{MK35}_s1p", TAG35_U0, TAG35_DR, TAG35_DL, TAG35_SR]) and facts["unsteered_row_tag"] == TAG35_U0 and facts["unsteered_default_window"] == Z.WINDOW35
    assert facts["unsteered_row_rule"] == B.UNSTEERED_ROW_RULE and facts["window_anchor_rule"] == B.WINDOW_ANCHOR_RULE
    inst = seeds.instrument_row(mod.INSTRUMENT_SEED, mod.EVAL_ID, model_id=MK35)["instrument_id"]
    runs = {(r["view_legacy"], r["campaign_cell_id_legacy"], r["context_window_tokens"]): r for r in bt["run"]}
    assert sorted(runs) == sorted([(TAG35_U0, NONE, Z.WINDOW35), (TAG35_U0, NONE, Z.WINDOW35_CONST), (TAG35_U0, Z.PROMPTED, Z.WINDOW35), (TAG35_U0, Z.CONST35, Z.WINDOW35_CONST),
                                   (TAG35, Z.CTL35, Z.WINDOW35), (TAG35, LORA35, Z.WINDOW35), (TAG35, Z.DIFF35, Z.WINDOW35)]), sorted(runs)
    none_a, none_w = runs[(TAG35_U0, NONE, Z.WINDOW35)], runs[(TAG35_U0, NONE, Z.WINDOW35_CONST)]
    # two baselines, two run ids (the store keys differ), both on the u0 tag with the serving window on the row and the declaration on the re-serve
    assert none_a["run_id"] != none_w["run_id"] and none_a["object_id"] == none_w["object_id"] == "none"
    assert (none_a["served_max_num_seqs"], none_w["served_max_num_seqs"]) == (Z.WINDOW35_SEQS, Z.WINDOW35_CONST_SEQS)
    assert none_a["extra"]["unsteered_row"]["row_key"] == NONE and none_a["extra"]["unsteered_row"]["window_reserve"] is None
    assert none_w["extra"]["unsteered_row"]["row_key"] == f"{NONE}@w{Z.WINDOW35_CONST}" and none_w["extra"]["unsteered_row"]["window_reserve"]["window"] == {"max_model_len": Z.WINDOW35_CONST, "max_num_seqs": Z.WINDOW35_CONST_SEQS}
    assert none_w["run_id"] == mod.run_id_for(TAG35_U0, fx["keys"][f"{TAG35_U0}/{NONE}@w{Z.WINDOW35_CONST}"], NONE, inst)
    for r in bt["run"]:
        # the fitted LoRA rows are base fits served on the instruct checkpoint (base_to_instruct); the unsteered none has no object (none); a prompted key is prompted_only
        okey_r = r["campaign_cell_id_legacy"]
        want_tk = "base_to_instruct" if okey_r in (Z.CTL35, LORA35, Z.DIFF35) else ("none" if okey_r == NONE else "prompted_only")
        assert r["model_id"] == MK35 and r["item_set_id"] == mod.ITEM_SET_ID and r["instrument_id"] == inst and r["seed_base"] is None
        assert r["transfer_kind"] == want_tk, (okey_r, r["transfer_kind"])
        assert r["extra"]["serving_window"]["max_model_len"] == r["context_window_tokens"] and r["extra"]["serving_window"]["rule"] == B.WINDOW_ANCHOR_RULE
        assert ("unsteered_row" in r["extra"]) == (r["view_legacy"] == TAG35_U0)
    # WINDOW-MATCHED anchors: the s1 LoRA rows and the u0 safety row -> the 8,192 none; the constitution row -> the 14,336 re-serve; the none rows anchor nothing
    want = {(TAG35, Z.CTL35): (none_a, "sibling_none", True), (TAG35, LORA35): (none_a, "sibling_none", True), (TAG35, Z.DIFF35): (none_a, "sibling_none", True),
            (TAG35_U0, Z.PROMPTED): (none_a, "same_standup_none", False), (TAG35_U0, Z.CONST35): (none_w, "same_standup_none", False)}
    assert len(bt["run_contrast"]) == 5
    for (tag, okey), (anchor, kind, cross) in want.items():
        r = runs[(tag, okey, Z.WINDOW35_CONST if okey == Z.CONST35 else Z.WINDOW35)]
        cn = _contrast_of(bt, r["run_id"])
        assert cn is not None and cn["anchor_run_id"] == anchor["run_id"] and cn["anchor_kind"] == kind and cn["extra"]["cross_standup"] is cross and cn["extra"]["anchor_view_legacy"] == TAG35_U0
        assert cn["extra"]["anchor_rule"] == B.WINDOW_ANCHOR_RULE and cn["extra"]["window_matched"]["row_window"] == cn["extra"]["window_matched"]["anchor_window"] == r["context_window_tokens"]
        assert cn["extra"]["window_matched"]["anchor_row_key"] == f"{TAG35_U0}/{NONE}" + (f"@w{Z.WINDOW35_CONST}" if okey == Z.CONST35 else "")
        assert r["extra"]["anchor"]["state"] == B.ANCHOR_CONTRASTED and r["extra"]["anchor"]["anchor_run_id"] == anchor["run_id"] and r["extra"]["anchor"]["rule"] == B.WINDOW_ANCHOR_RULE
        assert ("cross_standup_note" in cn["extra"]) is cross
        f = facts["contrasts"][f"{tag}|{okey}"]
        assert f["state"] == B.ANCHOR_CONTRASTED and f["anchor_view_legacy"] == TAG35_U0 and f["window_matched"]["anchor_window"] == r["context_window_tokens"]
    # the value of the constitution row from the synthetic truth; its contrast against the re-serve's items
    c = X.counts(X.spec(eval_id=eval_id, shape=Z.SHAPE_FULL, **Z.SPECS35_U0[eval_id][Z.CONST35]))
    value, _bound = Z.metrics_of(bt, runs[(TAG35_U0, Z.CONST35, Z.WINDOW35_CONST)]["run_id"], mod.recount_computation_id())
    assert abs(value["value"] - c["acc"]) < 1e-9
    # the gemma group beside it reads as it did
    _bt_g, facts_g, _ = Z.read_group(ad, Z.TAG)
    assert facts_g["n_refusals"] == 0 and facts_g["runs"] == 5 and "unsteered_row_tag" not in facts_g


def test_a_row_whose_serving_window_has_no_u0_none_declared_refuses_its_contrast_by_name(tmp_path, seeds):
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", qwen35b_u0=True, u0_window_reserve_store=False, u0_window_reserve_declared=False)
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    bt, facts, _ = Z.read_group(ad, TAG35)
    # the missing comparison is refused BY NAME, never silently: one refusal line, one parked count, the other four contrasts filed
    assert facts["n_refusals"] == 1 and facts["runs"] == 6 and len(bt["run_contrast"]) == 4 and (facts["contrasts_parked"], facts["contrasts_filed"]) == (1, 4), facts["refusals"]
    const = [r for r in bt["run"] if r["campaign_cell_id_legacy"] == Z.CONST35][0]
    assert _contrast_of(bt, const["run_id"]) is None
    f = facts["contrasts"][f"{TAG35_U0}|{Z.CONST35}"]
    assert f["state"] == "no_same_window_none" and f"({Z.WINDOW35_CONST} x {Z.WINDOW35_CONST_SEQS})" in f["why"] and f["window_matched"]["row_window"] == Z.WINDOW35_CONST
    assert f["currency_state"] == "no_current_anchor" and f["own_anchor_run_id"] is None and f["currency_why"] == B.WINDOW_PARK_NOTE
    blk = const["extra"]["anchor"]
    assert blk["state"] == "no_same_window_none" and blk["anchor_run_id"] is None and blk["rule"] == B.WINDOW_ANCHOR_RULE and blk["currency_state"] == "no_current_anchor"
    assert blk["anchor_selection"]["state"] == "no_current_anchor" and blk["anchor_selection"]["own_anchor_run_id"] is None and blk["anchor_selection"]["candidates"] == []
    assert blk["reason"].startswith("the window rule admits no baseline but a u0 `none` served at the row's window") and f"({Z.WINDOW35_CONST} x {Z.WINDOW35_CONST_SEQS})" in blk["reason"]
    assert "no u0 `none` is declared at the row's serving window" in blk["reason"]
    # the value reading carries the park: a delta_block whose every delta is None, whose reason names the window and the absent none, under the window rule
    vr = next(m for m in bt["run_metric"] if m["run_id"] == const["run_id"] and m["computation_id"] == BCF.recount_computation_id())
    db = vr["extra"]["delta_block"]
    assert db["delta"] is None and db["n_pairs"] == 0 and db["rule"] == B.WINDOW_ANCHOR_RULE and db["computed_by"] == "registry"
    assert db["anchor"] == dict(own_anchor_run_id=None, own_anchor_kind=None, state="no_current_anchor", own_reason="no_same_window_none", anchor_run_id=None, anchor_kind=None,
                                registry_contrast_filed_against=None)
    assert db["why_not_a_contrast_row"] == blk["reason"] and db["anchor_selection"] == blk["anchor_selection"] and db["delta_method"] == f"none: {blk['reason']}"
    assert db["producer_block"]["producer_delta_value_vs_none"] == const["extra"]["three_numbers"]["producer_delta_value_vs_none"]
    # the one refusal line names the row, its serving window and the absent none
    assert facts["refusals"] == [f"{TAG35_U0}|{Z.CONST35}|contrast -- contrast has no baseline at the row's serving window ({Z.WINDOW35_CONST} x {Z.WINDOW35_CONST_SEQS}) "
                                 f"[no_same_window_none: {f['why']}]; filed without a contrast"]
    assert facts["anchor_states"] == {"own_current": 4, "no_current_anchor": 1}
    # the 8,192 rows are untouched by the missing re-serve
    assert all(_contrast_of(bt, r["run_id"]) is not None for r in bt["run"] if r["campaign_cell_id_legacy"] not in (NONE, Z.CONST35))


def test_a_declared_window_reserve_without_its_three_number_row_refuses_by_name_and_the_constitution_contrast_waits(tmp_path, seeds):
    # the live mirror's state on 2026-10-05: the re-serve is mirrored and declared, its three-number row is not (the producer's one-row-per-(tag, key) grammar has no slot yet)
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", qwen35b_u0=True, u0_window_reserve_three_row=False)
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    bt, facts, _ = Z.read_group(ad, TAG35)
    # two refusals by name: the re-serve's missing three-number row, then the constitution row's contrast parked on it (the window rule admits no other baseline)
    assert facts["runs"] == 6 and facts["n_refusals"] == 2 and len(bt["run_contrast"]) == 4 and (facts["contrasts_parked"], facts["contrasts_filed"]) == (1, 4), facts["refusals"]
    assert facts["refusals"][0] == f"{TAG35_U0}|{NONE}@w{Z.WINDOW35_CONST} -- {BCF.THREE_NUMBERS_FILE} carries no row for {TAG35_U0}/{NONE}@w{Z.WINDOW35_CONST}"
    f = facts["contrasts"][f"{TAG35_U0}|{Z.CONST35}"]
    assert f["state"] == B.ANCHOR_REFUSED and f"{TAG35_U0}/{NONE}@w{Z.WINDOW35_CONST}" in f["why"] and f["currency_state"] == "no_current_anchor"
    assert facts["refusals"][1] == (f"{TAG35_U0}|{Z.CONST35}|contrast -- contrast has no baseline at the row's serving window ({Z.WINDOW35_CONST} x {Z.WINDOW35_CONST_SEQS}) "
                                    f"[{B.ANCHOR_REFUSED}: {f['why']}]; filed without a contrast")
    const = [r for r in bt["run"] if r["campaign_cell_id_legacy"] == Z.CONST35][0]
    assert _contrast_of(bt, const["run_id"]) is None and const["extra"]["anchor"]["state"] == B.ANCHOR_REFUSED and const["extra"]["anchor"]["anchor_selection"]["state"] == "no_current_anchor"
    vr = next(m for m in bt["run_metric"] if m["run_id"] == const["run_id"] and m["computation_id"] == BCF.recount_computation_id())
    assert vr["extra"]["delta_block"]["n_pairs"] == 0 and f"{TAG35_U0}/{NONE}@w{Z.WINDOW35_CONST}" in vr["extra"]["delta_block"]["why_not_a_contrast_row"]
    assert vr["extra"]["delta_block"]["anchor"]["own_reason"] == B.ANCHOR_REFUSED and vr["extra"]["delta_block"]["rule"] == B.WINDOW_ANCHOR_RULE
    # a declaration whose suffix does not spell its window refuses the key by name before any store is read
    fx2 = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", root=str(tmp_path / "bad"), qwen35b_u0=True,
                   launch_override=lambda d: d["families"][MK35]["unsteered"]["window_reserves"]["bigcodebench"].__setitem__("window", {"max_model_len": 12288, "max_num_seqs": 18}))
    _bt2, facts2, _ = Z.read_group(ad.__class__(root=fx2["root"], seeds=Z.seeds_for(seeds), roster_path=fx2["roster"], pins_path=fx2["pins"], arditi_roster_path=fx2["arditi_roster"], arditi_pins_path=fx2["arditi_pins"]), TAG35)
    assert any(r.startswith(f"{TAG35_U0}|{NONE}@w12288 -- families.{MK35}.unsteered.window_reserves.bigcodebench[0] declares suffix") and "must spell the window" in r for r in facts2["refusals"]), facts2["refusals"]


# --------------------------------------------------------------------------------------------
# source_states: the u0 members and the window-reserve declaration are members of the row table and the three-number record
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("eval_id", sorted(MODULES))
def test_source_states_enumerate_the_u0_members_and_the_window_reserve_declaration(tmp_path, seeds, eval_id):
    """Before 89d09f55b `source_states()` bound the two-argument tag map, so the launch-row and three-number memberships of a u0
    family omitted every u0 member and never listed the synthetic `none@w<N>` declaration (`families.<model key>.unsteered.window_reserves.<eval>`,
    no launch row of its own) -- a self-retiring ruling keyed to that input's membership would have evaluated against a truncated snapshot. The
    membership is now the SAME set of rows read() files: the s1 rows, the u0 rows and the re-serve, keyed the way read() keys them."""
    mod, cls = MODULES[eval_id]
    fx = Z.mirror(tmp_path, seeds, eval_id=eval_id, qwen35b_u0=True)
    ad = Z.adapter(cls, fx, seeds)
    inputs = ad.discover()
    by_group = {}
    for st in ad.source_states(X.ctx(), inputs):
        base = os.path.basename(st.input_uri)
        by_group.setdefault(base.rsplit("#", 1)[1], {})[base.rsplit("#", 1)[0]] = st
    assert set(by_group) == {Z.TAG, TAG35}
    reserve_key = f"{NONE}@w{Z.WINDOW35_CONST}"
    # the row table: the five launch rows at the group's selections plus the declared re-serve, under read()'s synthetic tuple id (UNSTEERED_SELECTION/none@w<N>)
    launch = by_group[TAG35][B.LAUNCH_SET_FILE]
    want_launch = sorted([f"{Z.SEL35}/{Z.CTL35}", f"{Z.SEL35}/{LORA35}", f"{Z.SEL35}/{Z.DIFF35}",
                          f"{Z.SEL35_U0}/{NONE}", f"{Z.SEL35_U0}/{Z.PROMPTED}", f"{Z.SEL35_U0}/{Z.CONST35}", f"{Z.SEL35_U0}/{reserve_key}"])
    assert launch.current_members == want_launch, launch.current_members
    assert launch.json_path == f"$.evals.{mod.SPEC.launch_set_eval_key}.launch"
    # the three-number record: every row of the family's seven, the re-serve keyed `<u0 tag>/none@w<N>`
    three = by_group[TAG35][mod.THREE_NUMBERS_FILE]
    want_three = sorted([f"{TAG35}/{Z.CTL35}", f"{TAG35}/{LORA35}", f"{TAG35}/{Z.DIFF35}",
                         f"{TAG35_U0}/{NONE}", f"{TAG35_U0}/{reserve_key}", f"{TAG35_U0}/{Z.PROMPTED}", f"{TAG35_U0}/{Z.CONST35}"])
    assert three.current_members == want_three, three.current_members
    assert three.json_path == "$.rows"
    # what read() files is exactly that set: seven runs, one per member of the three-number record
    bt, facts, _ = Z.read_group(ad, TAG35)
    assert facts["runs"] == 7 and facts["n_refusals"] == 0
    assert sorted(f"{r['view_legacy']}/{r['extra']['unsteered_row']['row_key'] if 'unsteered_row' in r['extra'] else r['campaign_cell_id_legacy']}" for r in bt["run"]) == want_three
    # the gemma group beside it is untouched: no u0 member, the pre-existing five
    assert not any(m.startswith(f"{Z.SEL35_U0}/") for m in by_group[Z.TAG][B.LAUNCH_SET_FILE].current_members) and len(by_group[Z.TAG][B.LAUNCH_SET_FILE].current_members) == 5
    assert not any(TAG35_U0 in m for m in by_group[Z.TAG][mod.THREE_NUMBERS_FILE].current_members)
    # a declaration whose suffix does not spell its window is no member (read() refuses it by key), so the membership never names a row that cannot file
    fx2 = Z.mirror(tmp_path, seeds, eval_id=eval_id, root=str(tmp_path / "bad"), qwen35b_u0=True,
                   launch_override=lambda d: d["families"][Z.MK35]["unsteered"]["window_reserves"][eval_id].__setitem__("window", {"max_model_len": 12288, "max_num_seqs": 18}))
    ad2 = Z.adapter(cls, fx2, seeds)
    launch2 = next(st for st in ad2.source_states(X.ctx(), ad2.discover()) if os.path.basename(st.input_uri) == f"{B.LAUNCH_SET_FILE}#{TAG35}")
    assert launch2.current_members == sorted(m for m in want_launch if m != f"{Z.SEL35_U0}/{reserve_key}"), launch2.current_members
    # the 200-item readers admit no u0 selection: their tag map is unchanged (the emission pin holds them byte for byte)
    assert B.group_tags(TAG35, False, False) == {Z.SEL35: TAG35, f"{Z.SEL35}p": f"{MK35}_s1p"}
    assert B.group_tags(TAG35, False, True) == {Z.SEL35: TAG35, f"{Z.SEL35}p": f"{MK35}_s1p", Z.SEL35_U0: TAG35_U0}


# --------------------------------------------------------------------------------------------
# the four other whole-group refusals are refusals the status reads: v_source_status.ok 0, extra.refuse names the cause
# --------------------------------------------------------------------------------------------


def _source_rows(root, source):
    rows = F.q(root, "SELECT ok, detail, extra_json FROM source_status WHERE source_id = ?", (source,))
    assert len(rows) == 1, rows
    return rows[0][0], rows[0][1], json.loads(rows[0][2])


def _v_source_status(root, source_legacy):
    conn = __import__("sqlite3").connect(os.path.join(root, "registry.sqlite"))
    conn.row_factory = __import__("sqlite3").Row
    try:
        vs = [dict(r) for r in conn.execute("SELECT * FROM v_source_status WHERE lane_source_legacy = ?", (source_legacy,)).fetchall()]
    finally:
        conn.close()
    assert len(vs) == 1, vs
    return vs[0]["ok"], json.loads(vs[0]["extra_json"]), vs[0]["detail"]


def _gemma_objects(fx, seeds):
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    return [ad.resolve_object(Z.RELU, Z.SEL, pin_key=Z.STAGED, model_key=Z.MK)[0], ad.resolve_object(Z.COMPOSED, Z.SEL_P, pin_key=Z.STAGED_P, model_key=Z.MK)[0]]


def _assert_whole_group_refusal(root, source, source_legacy, needle):
    """The shape b11174427 pinned for the undeclared row dir: the group's own status row reads ok 0 with the reason under extra.refuse and in the
    detail's REFUSE clause, no record folded, `refusals` carrying the same text; v_source_status folds the source to MIN(ok) = 0 with the reason."""
    ok, detail, extra = _source_rows(root, source)
    assert ok == 0, (ok, detail)
    assert len(extra["refuse"]) == 1 and needle in extra["refuse"][0], extra["refuse"]
    assert extra["refusals"] == [f"{extra['group']} -- {extra['refuse'][0]}"] and extra["record_counts"] == {} and "emitted_nothing_unexplained" not in extra
    assert f"REFUSE (1): {extra['refuse'][0]}" in detail and "0 record(s)" in detail
    v_ok, v_extra, v_detail = _v_source_status(root, source_legacy)
    # a source of ONE group folds that group's extra as its own; several groups fold under `by_source` (the undeclared-row-dir shape), every refused one's
    # reason listed under `refuse` -- on a 200-item reader the gemma group beside the 35B one is refused too (its family is another source's)
    per_source = v_extra.get("by_source", {source: v_extra})
    assert v_ok == 0 and any(needle in r for r in v_extra["refuse"]) and per_source[source]["refuse"] == extra["refuse"] and f"REFUSE (1): {extra['refuse'][0]}" in v_detail
    return extra


def test_a_group_that_spells_no_seed_tag_reads_ok_0_on_v_source_status_with_extra_refuse(tmp_path, seeds):
    """`group ... does not spell a seed tag`: comparison_group maps every discovered input to a seed tag or an undeclared row dir, so the refusal is reached
    through the runner only when the grouping names something else -- here the paired tag itself, which read() refuses as a whole."""
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench")
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    root = _registry_root(tmp_path, seeds, _gemma_objects(fx, seeds))
    paired = f"{Z.MK}_s1p"
    ad.comparison_group = lambda inp: paired            # the runner groups through the instance: one group named for the paired tag
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=X.NOW)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    extra = _assert_whole_group_refusal(root, B.source_id_for(BCF.NAME, paired), BCF.SOURCE, f"group {paired!r} does not spell a seed tag")
    assert extra["runs"] == 0 and F.q(root, "SELECT COUNT(*) FROM run WHERE lane_source_legacy = ?", (BCF.SOURCE,)) == [(0,)]


def test_a_model_key_outside_the_sources_families_reads_ok_0_on_v_source_status_with_extra_refuse(tmp_path, seeds):
    """`model key ... is not one of this source's families`: the 35B store under a 200-item reader (whose tuples do not carry the key) -- the read-level
    case of `test_a_35b_store_under_a_200_item_reader_refuses_its_group_by_name`, read through the runner: the group's status reads 0, not an explained
    empty read filed ok = 1."""
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", qwen35b_store=True)
    ad = X.adapter(BCS.BigcodebenchSeedrepAdapter, fx, seeds)
    root = _registry_root(tmp_path, seeds, _gemma_objects(fx, seeds))
    rep, _ing, _ing2 = F.run_and_ingest(root, ad, now=X.NOW)
    source = B.source_id_for(BCS.NAME, TAG35)
    assert not any(r.error for r in rep.results if r.group == TAG35), [r.error for r in rep.results if r.group == TAG35]
    extra = _assert_whole_group_refusal(root, source, BCS.SOURCE, f"model key {MK35!r} is not one of this source's families")
    assert extra["runs"] == 0 and F.q(root, "SELECT COUNT(*) FROM run WHERE lane_source_legacy = ? AND view_legacy LIKE ?", (BCS.SOURCE, f"{MK35}%")) == [(0,)]


@pytest.mark.parametrize("which", ["legacy_spellings", "models_yaml"])
def test_a_seed_table_disagreeing_with_the_model_key_table_reads_ok_0_on_v_source_status_with_extra_refuse(tmp_path, seeds, which):
    """The two halves of the model check: legacy_spellings.yaml spelling the group as another model, and models.yaml pinning the model at another
    checkpoint or revision. The seeds copy is REBOUND (new dicts on the adapter's own copy), never mutated in place: the module-scoped seeds are shared."""
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench")
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    root = _registry_root(tmp_path, seeds, _gemma_objects(fx, seeds))
    model_id = B.MODEL_KEYS[Z.MK].model_id
    if which == "legacy_spellings":
        rows = [dict(r, model_id="synthetic/another-model") if Z.TAG in str(r.get("legacy", "")) else r for r in ad.seeds.legacy["model_keys"]]
        assert any(r.get("model_id") == "synthetic/another-model" for r in rows)
        ad.seeds.legacy = {**ad.seeds.legacy, "model_keys": rows}
        needle = f"legacy_spellings.yaml spells {Z.TAG!r} as model 'synthetic/another-model'"
    else:
        ad.seeds.models = {**ad.seeds.models, model_id: {**ad.seeds.models[model_id], "hf_revision": "0" * 40}}
        needle = f"models.yaml pins {model_id} at"
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=X.NOW)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    extra = _assert_whole_group_refusal(root, B.source_id_for(BCF.NAME, Z.TAG), BCF.SOURCE, needle)
    assert "the model-key table says" in extra["refuse"][0] and extra["runs"] == 0
    assert F.q(root, "SELECT COUNT(*) FROM run WHERE lane_source_legacy = ?", (BCF.SOURCE,)) == [(0,)]


def test_a_missing_instrument_row_reads_ok_0_on_v_source_status_with_extra_refuse(tmp_path, seeds):
    """`instrument row: ...`: the index on the adapter's seeds copy is rebound without the (seed, eval) row this reader files on."""
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench")
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    root = _registry_root(tmp_path, seeds, _gemma_objects(fx, seeds))
    b0 = ad.seeds.b0_index()
    assert b0 is not None and (BCF.SPEC.instrument_seed, BCF.EVAL_ID) in b0["instruments"]
    ad.seeds._b0 = {**b0, "instruments": {k: v for k, v in b0["instruments"].items() if k != (BCF.SPEC.instrument_seed, BCF.EVAL_ID)}}
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=X.NOW)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    # the refusal spells the KeyError the way str() renders it (repr-quoted, so double quotes around a message that carries single quotes)
    extra = _assert_whole_group_refusal(root, B.source_id_for(BCF.NAME, Z.TAG), BCF.SOURCE, f"instrument seed {BCF.SPEC.instrument_seed!r} has no WS-B0 row for eval {BCF.EVAL_ID!r}")
    assert extra["refuse"][0].startswith("instrument row: ")
    assert extra["runs"] == 0 and F.q(root, "SELECT COUNT(*) FROM run WHERE lane_source_legacy = ?", (BCF.SOURCE,)) == [(0,)]
    # a declared mirror beside it, on the unchanged seeds, reads integrated: the 0 is the refusal's, not the reader's
    root2 = _registry_root(tmp_path / "r2", seeds, _gemma_objects(fx, seeds))
    rep2, _a, _b = F.run_and_ingest(root2, Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds), now=X.NOW)
    assert rep2.groups_refused == 0 and F.q(root2, "SELECT ok FROM v_source_status WHERE lane_source_legacy = ?", (BCF.SOURCE,)) == [(1,)]



# --------------------------------------------------------------------------------------------
# the dose-0.75 random-init ReLU row (DOSE_RELU_ROW_RULE, 2026-10-05): grammar, a dry read, the undeclared variant, the second none under u0
# --------------------------------------------------------------------------------------------


def test_the_grammar_reads_the_d0p75_relul_row_dir_tag_launch_tag_and_gate_records():
    assert DR_ROW_DIR == f"{MK35}-f22d0p75_relul" and TAG35_DR == f"{MK35}_d0p75_relul" and SEL35_DR == "d0p75_relul"
    assert B.tag_of_row_dir(DR_ROW_DIR) == TAG35_DR and B.row_dir_of(TAG35_DR) == DR_ROW_DIR and B.seed_tag_of(TAG35_DR) == TAG35 and B.model_key_of(TAG35_DR) == MK35
    assert B.is_dose_relu_tag(TAG35_DR) and not B.is_dose_relu_tag(TAG35) and not B.is_dose_relu_tag(TAG35_U0) and not B.is_unsteered_tag(TAG35_DR)
    assert B.tag_parts(TAG35_DR) is None and B.dose_relu_anchor_tag(TAG35_DR) == TAG35 and B.dose_relu_model_key(TAG35_DR) == MK35 and B.dose_relu_model_key(TAG35) is None
    assert B.launch_tag_of(TAG35_DR, Z.MIX35_R) == TAG35_DR
    m = B.SUFFIX_RE.match(f"bcbw-{MK35}_d0p75_relul-1234abc-bcb")
    assert (m.group("launch_tag"), m.group("task_tag"), m.group("window"), m.group("relaunch")) == (TAG35_DR, "-bcb", None, None) and B.suffix_task(m.group(0)) == "bigcodebench"
    assert B.gate_name_parts(f"gate_{TAG35_DR}_{Z.DIFF35_R}_bcbw-{TAG35_DR}-1234abc.json") == {"tag": TAG35_DR, "arditi": None, "okey": Z.DIFF35_R, "suffix": f"bcbw-{TAG35_DR}-1234abc"}
    # the full readers admit the selection in the family's s1 group; the 200-item readers do not (their emission pin holds them byte for byte)
    assert B.group_tags(TAG35, False, True, True)[SEL35_DR] == TAG35_DR and SEL35_DR not in B.group_tags(TAG35, False, True, False) and SEL35_DR not in B.group_tags(TAG35, False, False, False)
    assert BCF.SPEC.dose_relu_row and BFF.SPEC.dose_relu_row and not BCA.SPEC.dose_relu_row and not BCS.SPEC.dose_relu_row
    assert B.MODEL_KEYS[MK35].dose_relu_row and not any(B.MODEL_KEYS[k].dose_relu_row for k in B.MODEL_KEYS if k != MK35)
    # the declared row is grammar; its spelling without the trailing letter is a declared key under an undeclared selection, named as such (the row-dir shape admits `_`)
    assert B.undeclared_row_dir_reason(DR_ROW_DIR) is None and not B.is_undeclared_row_dir_group(DR_ROW_DIR)
    r = B.undeclared_row_dir_reason(f"{MK35}-f22d0p75_relu")
    assert r.startswith(f"row dir '{MK35}-f22d0p75_relu' names the declared model key {MK35!r} but no declared row (selection 'd0p75_relu'") and f"{MK35}-f22d0p75_relul" in r
    assert B.is_undeclared_row_dir_group(f"{MK35}-f22d0p75_relu")
    # the spelling resolves through legacy_spellings.yaml to the 35B model row, like the u0 one
    mv = LC.Seeds.locate().model_for_view_legacy(TAG35_DR)
    assert mv["model_id"] == MK35 and mv["fit_model_id"] == f"{MK35}_base" and mv["transfer_kind"] == "base_to_instruct" and mv["source"] == "legacy_spellings.yaml model_keys"


@pytest.mark.parametrize("eval_id", sorted(MODULES))
def test_the_d0p75_relul_rows_file_three_runs_per_eval_window_matched_on_the_u0_none(tmp_path, seeds, eval_id):
    """A dry read of the fixture's dose row beside the u0 family: three steered runs on the d0p75_relul tag, each resolved through the pins block by the launch set's
    pin key (the bank's d0p75l__ staged id), each contrasted against the u0 none at its own serving window across tags (sibling_none, cross_standup) -- the s1
    rows' rule -- and the source states list the three members in the row table and the three-number record."""
    mod, cls = MODULES[eval_id]
    fx = Z.mirror(tmp_path, seeds, eval_id=eval_id, qwen35b_u0=True, qwen35b_d0p75_relul=True)
    ad = Z.adapter(cls, fx, seeds)
    bt, facts, groups = Z.read_group(ad, TAG35)
    assert groups == [Z.TAG, TAG35] and facts["n_refusals"] == 0 and facts["runs"] == 10 and facts["n_not_run"] == 0, facts["refusals"]
    assert facts["dose_relu_row_tag"] == TAG35_DR and facts["dose_relu_row_rule"] == B.DOSE_RELU_ROW_RULE and TAG35_DR in facts["tags"]
    inst = seeds.instrument_row(mod.INSTRUMENT_SEED, mod.EVAL_ID, model_id=MK35)["instrument_id"]
    dose_runs = {r["campaign_cell_id_legacy"]: r for r in bt["run"] if r["view_legacy"] == TAG35_DR}
    assert sorted(dose_runs) == sorted([Z.CTL35_R, Z.MIX35_R, Z.DIFF35_R])
    none_a = next(r for r in bt["run"] if r["view_legacy"] == TAG35_U0 and r["campaign_cell_id_legacy"] == NONE and r["context_window_tokens"] == Z.WINDOW35)
    for okey, r in dose_runs.items():
        assert r["model_id"] == MK35 and r["object_id"] == Z.OBJ35_DR[okey] and r["transfer_kind"] == "base_to_instruct" and r["seed_base"] is None and r["instrument_id"] == inst
        assert r["run_id"] == mod.run_id_for(TAG35_DR, fx["keys"][f"{TAG35_DR}/{okey}"], okey, inst) and r["context_window_tokens"] == Z.WINDOW35
        assert r["extra"]["dose_relu_row"] == {"rule": B.DOSE_RELU_ROW_RULE, "selection": SEL35_DR, "row_key": okey, "anchor_tuple": f"{Z.SEL35_U0}/{NONE}", "pin_key": Z.STAGED35_DR[okey]}
        assert r["extra"]["serving_window"]["max_model_len"] == Z.WINDOW35 and "unsteered_row" not in r["extra"]
        # NO fit seed on a dose-ReLU row: the s1 group is a standup grouping, not the object's fit provenance; the selection label stays
        assert r["extra"]["fit_seed"] is None and r["seed_base"] is None and r["extra"]["seed_replicate_label"] == SEL35_DR and r["extra"]["seed_axis"]["seed_base"] is None
        assert facts["rows"][f"{TAG35_DR}|{okey}"]["object_id"] == Z.OBJ35_DR[okey] and facts["rows"][f"{TAG35_DR}|{okey}"]["staged_id"] == Z.STAGED35_DR[okey]
        cn = _contrast_of(bt, r["run_id"])
        assert cn is not None and cn["anchor_run_id"] == none_a["run_id"] and cn["anchor_kind"] == "sibling_none" and cn["extra"]["cross_standup"] is True
        assert cn["extra"]["anchor_view_legacy"] == TAG35_U0 and cn["extra"]["anchor_rule"] == B.WINDOW_ANCHOR_RULE
        assert cn["extra"]["window_matched"]["row_window"] == cn["extra"]["window_matched"]["anchor_window"] == Z.WINDOW35 and cn["extra"]["window_matched"]["anchor_row_key"] == f"{TAG35_U0}/{NONE}"
        assert r["extra"]["anchor"]["state"] == B.ANCHOR_CONTRASTED and r["extra"]["anchor"]["anchor_run_id"] == none_a["run_id"]
        c = X.counts(X.spec(eval_id=eval_id, shape=Z.SHAPE_FULL, **Z.SPECS35_DR[eval_id][okey]))
        value, _bound = Z.metrics_of(bt, r["run_id"], mod.recount_computation_id())
        assert abs(value["value"] - c["acc"]) < 1e-9
    assert len(bt["run_contrast"]) == 8 and len(dose_runs) == 3        # the five of the u0 family + the three dose rows
    # the s1 LoRA rows still carry their fit seed (the group's), the u0 rows none: only the dose row's provenance moved
    assert {r["extra"]["fit_seed"] for r in bt["run"] if r["view_legacy"] == TAG35} == {1} and {r["extra"]["fit_seed"] for r in bt["run"] if r["view_legacy"] == TAG35_U0} == {None}
    # the source states: the three members join the row table and the three-number record (the membership rule)
    by_name = {os.path.basename(st.input_uri).rsplit("#", 1)[0]: st for st in ad.source_states(X.ctx(), ad.discover()) if os.path.basename(st.input_uri).endswith(f"#{TAG35}")}
    assert {f"{SEL35_DR}/{k}" for k in dose_runs} <= set(by_name[B.LAUNCH_SET_FILE].current_members)
    assert {f"{TAG35_DR}/{k}" for k in dose_runs} <= set(by_name[mod.THREE_NUMBERS_FILE].current_members)
    # the gemma group beside it reads as it did; the u0 family's own rows are untouched
    _bt_g, facts_g, _ = Z.read_group(ad, Z.TAG)
    assert facts_g["n_refusals"] == 0 and facts_g["runs"] == 5 and "dose_relu_row_tag" not in facts_g


def test_a_second_none_store_under_the_u0_row_at_the_dose_launchs_tree_binds_no_row(tmp_path, seeds):
    """The dose launch re-serves the baseline at its own tree under the u0 row dir: that store shares the (u0, none) row key with the declared store (whose tree the
    launch row names) and binds no new row -- a second store on one key is not read (it refuses by name: the launch set says the row was launched at the declared
    tree; the store's manifest names another); the declared store files as before and every dose row still anchors on it."""
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", qwen35b_u0=True, qwen35b_d0p75_relul=True, u0_second_none_store=True)
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    bt, facts, _ = Z.read_group(ad, TAG35)
    assert facts["runs"] == 10 and facts["n_refusals"] == 1, facts["refusals"]
    # the second store refuses by name on the (u0, none) key: the launch row names the declared store (tree TREE), not this one at the dose launch's tree
    assert facts["refusals"][0].startswith(f"{TAG35_U0}|{NONE} -- ") and Z.TREE_DR[:7] in facts["refusals"][0], facts["refusals"]
    nones = [r for r in bt["run"] if r["view_legacy"] == TAG35_U0 and r["campaign_cell_id_legacy"] == NONE and r["context_window_tokens"] == Z.WINDOW35]
    assert len(nones) == 1 and nones[0]["code_sha"] == Z.TREE and nones[0]["run_id"] == BCF.run_id_for(TAG35_U0, fx["keys"][f"{TAG35_U0}/{NONE}"], NONE,
                                                                                                        seeds.instrument_row(BCF.INSTRUMENT_SEED, BCF.EVAL_ID, model_id=MK35)["instrument_id"])
    for r in bt["run"]:
        if r["view_legacy"] == TAG35_DR:
            assert _contrast_of(bt, r["run_id"])["anchor_run_id"] == nones[0]["run_id"]
    # and without the dose row the second store is refused the same way: the rule is the launch row's tree, not the dose row
    fx2 = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", root=str(tmp_path / "u0only"), qwen35b_u0=True, u0_second_none_store=True)
    _bt2, facts2, _ = Z.read_group(Z.adapter(BCF.BigcodebenchFullAdapter, fx2, seeds), TAG35)
    assert facts2["runs"] == 7 and facts2["n_refusals"] == 1 and facts2["refusals"][0].startswith(f"{TAG35_U0}|{NONE} -- ") and Z.TREE_DR[:7] in facts2["refusals"][0]


def test_an_undeclared_dose_row_dir_still_refuses_by_name_as_a_whole_group(tmp_path, seeds):
    """`qwen35_35b_a3b-f22d0p75_relu` (no trailing letter) is a declared key under an undeclared selection: a group of its own, refused as a whole with the reason
    naming the key, the selection and the rows this reader reads (the dose row among them)."""
    name = f"{MK35}-f22d0p75_relu"
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", undeclared_row_dirs=(name,))
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    groups = _groups(ad)
    assert name in groups and TAG35_DR not in groups
    bt, facts, _ = Z.read_group(ad, name)
    assert bt == {} and facts["runs"] == 0 and facts["refuse"] == [B.undeclared_row_dir_reason(name)]
    assert "no declared row (selection 'd0p75_relu'" in facts["refuse"][0] and f"{MK35}-f22d0p75_relul" in facts["refuse"][0]



# --------------------------------------------------------------------------------------------
# the dose-1.25 difference-LoRA row (DOSE_LORA_ROW_RULE, 2026-10-05): grammar, a dry read, the undeclared variant, the second none under u0
# --------------------------------------------------------------------------------------------


def test_the_grammar_reads_the_d1p25_row_dir_tag_launch_tag_and_gate_records():
    assert DL_ROW_DIR == f"{MK35}-f22d1p25" and TAG35_DL == f"{MK35}_d1p25" and SEL35_DL == "d1p25" and B.DOSE_LORA_STRENGTH == 1.25
    assert B.tag_of_row_dir(DL_ROW_DIR) == TAG35_DL and B.row_dir_of(TAG35_DL) == DL_ROW_DIR and B.seed_tag_of(TAG35_DL) == TAG35 and B.model_key_of(TAG35_DL) == MK35
    assert B.is_dose_lora_tag(TAG35_DL) and B.is_dose_row_tag(TAG35_DL) and B.is_dose_row_tag(TAG35_DR) and not B.is_dose_lora_tag(TAG35_DR) and not B.is_dose_relu_tag(TAG35_DL)
    assert not B.is_dose_row_tag(TAG35) and not B.is_dose_row_tag(TAG35_U0) and B.tag_parts(TAG35_DL) is None and B.dose_lora_anchor_tag(TAG35_DL) == TAG35 and B.dose_lora_model_key(TAG35) is None
    assert B.launch_tag_of(TAG35_DL, Z.DIFF35) == TAG35_DL
    m = B.SUFFIX_RE.match(f"bcbw-{MK35}_d1p25-1234abc-bfcl")
    assert (m.group("launch_tag"), m.group("task_tag"), m.group("window"), m.group("relaunch")) == (TAG35_DL, "-bfcl", None, None) and B.suffix_task(m.group(0)) == "bfcl"
    assert B.gate_name_parts(f"gate_{TAG35_DL}_{Z.DIFF35}_bcbw-{TAG35_DL}-1234abc.json") == {"tag": TAG35_DL, "arditi": None, "okey": Z.DIFF35, "suffix": f"bcbw-{TAG35_DL}-1234abc"}
    # the full readers admit the selection in the family's s1 group; the 200-item readers do not (their emission pin holds them byte for byte)
    assert B.group_tags(TAG35, False, True, True, True)[SEL35_DL] == TAG35_DL and SEL35_DL not in B.group_tags(TAG35, False, True, True) and SEL35_DL not in B.group_tags(TAG35)
    assert BCF.SPEC.dose_lora_row and BFF.SPEC.dose_lora_row and not BCA.SPEC.dose_lora_row and not BCS.SPEC.dose_lora_row
    assert B.MODEL_KEYS[MK35].dose_lora_row and not any(B.MODEL_KEYS[k].dose_lora_row for k in B.MODEL_KEYS if k != MK35)
    # the declared row is grammar; a neighbouring dose spelling is a declared key under an undeclared selection, named as such
    assert B.undeclared_row_dir_reason(DL_ROW_DIR) is None and not B.is_undeclared_row_dir_group(DL_ROW_DIR)
    r = B.undeclared_row_dir_reason(f"{MK35}-f22d1p5")
    assert r.startswith(f"row dir '{MK35}-f22d1p5' names the declared model key {MK35!r} but no declared row (selection 'd1p5'") and f"{MK35}-f22d1p25" in r
    # the spelling resolves through legacy_spellings.yaml to the 35B model row at strength 1.25 (the LoRA family's unit)
    sd = LC.Seeds.locate()
    mv = sd.model_for_view_legacy(TAG35_DL)
    assert mv["model_id"] == MK35 and mv["fit_model_id"] == f"{MK35}_base" and mv["transfer_kind"] == "base_to_instruct" and mv["source"] == "legacy_spellings.yaml model_keys"
    row = next(r for r in sd.legacy["model_keys"] if str(r.get("legacy")) == TAG35_DL)
    assert row["strength_nominal"] == 1.25 and row["fit_replicate_label"] == "s1"


@pytest.mark.parametrize("eval_id", sorted(MODULES))
def test_the_d1p25_row_files_one_run_per_eval_window_matched_on_the_u0_none_with_no_fit_seed(tmp_path, seeds, eval_id):
    """A dry read of the fixture's dose-1.25 difference-LoRA row beside the u0 family: ONE steered run on the d1p25 tag (this selection has one object), resolved
    through the pins block by the launch set's pin key (the bank's d1p25__ staged id), contrasted against the u0 none at its own serving window across tags
    (sibling_none, cross_standup), with NO fit seed (a dose copy's), and listed by the source states."""
    mod, cls = MODULES[eval_id]
    fx = Z.mirror(tmp_path, seeds, eval_id=eval_id, qwen35b_u0=True, qwen35b_d1p25=True)
    ad = Z.adapter(cls, fx, seeds)
    bt, facts, groups = Z.read_group(ad, TAG35)
    assert groups == [Z.TAG, TAG35] and facts["n_refusals"] == 0 and facts["runs"] == 8 and facts["n_not_run"] == 0, facts["refusals"]
    assert facts["dose_lora_row_tag"] == TAG35_DL and facts["dose_lora_row_rule"] == B.DOSE_LORA_ROW_RULE and TAG35_DL in facts["tags"] and "dose_relu_row_tag" in facts
    inst = seeds.instrument_row(mod.INSTRUMENT_SEED, mod.EVAL_ID, model_id=MK35)["instrument_id"]
    runs = [r for r in bt["run"] if r["view_legacy"] == TAG35_DL]
    assert len(runs) == 1 and runs[0]["campaign_cell_id_legacy"] == Z.DIFF35
    r = runs[0]
    none_a = next(x for x in bt["run"] if x["view_legacy"] == TAG35_U0 and x["campaign_cell_id_legacy"] == NONE and x["context_window_tokens"] == Z.WINDOW35)
    assert r["model_id"] == MK35 and r["object_id"] == Z.OBJ35_DL and r["transfer_kind"] == "base_to_instruct" and r["instrument_id"] == inst and r["context_window_tokens"] == Z.WINDOW35
    assert r["run_id"] == mod.run_id_for(TAG35_DL, fx["keys"][f"{TAG35_DL}/{Z.DIFF35}"], Z.DIFF35, inst)
    assert r["extra"]["dose_lora_row"] == {"rule": B.DOSE_LORA_ROW_RULE, "selection": SEL35_DL, "row_key": Z.DIFF35, "anchor_tuple": f"{Z.SEL35_U0}/{NONE}", "pin_key": Z.STAGED35_DL,
                                           "strength_nominal": 1.25}
    assert "dose_relu_row" not in r["extra"] and "unsteered_row" not in r["extra"] and r["extra"]["serving_window"]["max_model_len"] == Z.WINDOW35
    # NO fit seed on a dose row: the s1 group is a standup grouping and a dose-scaled copy's fit seed is not applicable
    assert r["extra"]["fit_seed"] is None and r["seed_base"] is None and r["extra"]["seed_replicate_label"] == SEL35_DL and r["extra"]["seed_axis"]["seed_base"] is None
    assert facts["rows"][f"{TAG35_DL}|{Z.DIFF35}"]["object_id"] == Z.OBJ35_DL and facts["rows"][f"{TAG35_DL}|{Z.DIFF35}"]["staged_id"] == Z.STAGED35_DL
    cn = _contrast_of(bt, r["run_id"])
    assert cn is not None and cn["anchor_run_id"] == none_a["run_id"] and cn["anchor_kind"] == "sibling_none" and cn["extra"]["cross_standup"] is True
    assert cn["extra"]["anchor_view_legacy"] == TAG35_U0 and cn["extra"]["anchor_rule"] == B.WINDOW_ANCHOR_RULE
    assert cn["extra"]["window_matched"]["row_window"] == cn["extra"]["window_matched"]["anchor_window"] == Z.WINDOW35 and cn["extra"]["window_matched"]["anchor_row_key"] == f"{TAG35_U0}/{NONE}"
    assert r["extra"]["anchor"]["state"] == B.ANCHOR_CONTRASTED and r["extra"]["anchor"]["anchor_run_id"] == none_a["run_id"]
    c = X.counts(X.spec(eval_id=eval_id, shape=Z.SHAPE_FULL, **Z.SPECS35_DL[eval_id]))
    value, _bound = Z.metrics_of(bt, r["run_id"], mod.recount_computation_id())
    assert abs(value["value"] - c["acc"]) < 1e-9
    assert len(bt["run_contrast"]) == 6        # the five of the u0 family + the dose row
    # the s1 LoRA rows still carry their fit seed, the u0 rows none
    assert {x["extra"]["fit_seed"] for x in bt["run"] if x["view_legacy"] == TAG35} == {1} and {x["extra"]["fit_seed"] for x in bt["run"] if x["view_legacy"] == TAG35_U0} == {None}
    # the source states: the member joins the row table and the three-number record
    by_name = {os.path.basename(st.input_uri).rsplit("#", 1)[0]: st for st in ad.source_states(X.ctx(), ad.discover()) if os.path.basename(st.input_uri).endswith(f"#{TAG35}")}
    assert f"{SEL35_DL}/{Z.DIFF35}" in by_name[B.LAUNCH_SET_FILE].current_members and f"{TAG35_DL}/{Z.DIFF35}" in by_name[mod.THREE_NUMBERS_FILE].current_members
    # the gemma group beside it reads as it did
    _bt_g, facts_g, _ = Z.read_group(ad, Z.TAG)
    assert facts_g["n_refusals"] == 0 and facts_g["runs"] == 5 and "dose_lora_row_tag" not in facts_g


def test_a_second_none_store_under_the_u0_row_binds_no_row_beside_the_d1p25_row(tmp_path, seeds):
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", qwen35b_u0=True, qwen35b_d1p25=True, u0_second_none_store=True)
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    bt, facts, _ = Z.read_group(ad, TAG35)
    assert facts["runs"] == 8 and facts["n_refusals"] == 1, facts["refusals"]
    assert facts["refusals"][0].startswith(f"{TAG35_U0}|{NONE} -- ") and Z.TREE_DR[:7] in facts["refusals"][0], facts["refusals"]
    nones = [r for r in bt["run"] if r["view_legacy"] == TAG35_U0 and r["campaign_cell_id_legacy"] == NONE and r["context_window_tokens"] == Z.WINDOW35]
    assert len(nones) == 1 and nones[0]["code_sha"] == Z.TREE
    dose = next(r for r in bt["run"] if r["view_legacy"] == TAG35_DL)
    assert _contrast_of(bt, dose["run_id"])["anchor_run_id"] == nones[0]["run_id"]


def test_an_undeclared_d1p5_row_dir_refuses_by_name_as_a_whole_group(tmp_path, seeds):
    name = f"{MK35}-f22d1p5"
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", undeclared_row_dirs=(name,))
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    groups = _groups(ad)
    assert name in groups and TAG35_DL not in groups
    bt, facts, _ = Z.read_group(ad, name)
    assert bt == {} and facts["runs"] == 0 and facts["refuse"] == [B.undeclared_row_dir_reason(name)]
    assert "no declared row (selection 'd1p5'" in facts["refuse"][0] and f"{MK35}-f22d1p25" in facts["refuse"][0]



# --------------------------------------------------------------------------------------------
# the LeCun-init ReLU row (LECUN_RELU_ROW_RULE, 2026-10-07): grammar, a dry read of its two fits, the unlaunched third key, an unknown lecun__ pin key, the second none
# under u0, the undeclared neighbour
# --------------------------------------------------------------------------------------------


def test_the_grammar_reads_the_s2rl_row_dir_tag_launch_tag_store_suffixes_and_gate_records():
    assert SR_ROW_DIR == f"{MK35}-f22s2rl" and TAG35_SR == f"{MK35}_s2rl" and SEL35_SR == "s2rl" and B.LECUN_RELU_FIT_SEED == 0 and B.LECUN_RELU_ANCHOR_SEED == 1
    assert B.tag_of_row_dir(SR_ROW_DIR) == TAG35_SR and B.row_dir_of(TAG35_SR) == SR_ROW_DIR and B.seed_tag_of(TAG35_SR) == TAG35 and B.model_key_of(TAG35_SR) == MK35
    # `_s2rl` is NOT a seed tag (`_s<N>[p]`): tag_parts reads None, and the seed tag `_s2` is not the LeCun row
    assert B.tag_parts(TAG35_SR) is None and B.tag_parts(f"{MK35}_s2") == (2, False) and not B.is_lecun_relu_tag(f"{MK35}_s2") and B.lecun_relu_model_key(f"{MK35}_s2") is None
    assert B.is_lecun_relu_tag(TAG35_SR) and not B.is_dose_row_tag(TAG35_SR) and not B.is_unsteered_tag(TAG35_SR) and not B.is_instruct_tag(TAG35_SR)
    assert not B.is_lecun_relu_tag(TAG35) and not B.is_lecun_relu_tag(TAG35_U0) and not B.is_lecun_relu_tag(TAG35_DR) and not B.is_lecun_relu_tag(TAG35_DL)
    assert B.lecun_relu_anchor_tag(TAG35_SR) == TAG35 and B.lecun_relu_model_key(TAG35_SR) == MK35 and B.lecun_relu_model_key(TAG35) is None
    assert B.launch_tag_of(TAG35_SR, Z.CTL35_R) == TAG35_SR
    # the producer's store suffixes: `bcbw-qwen35_35b_a3b_s2rl-<sha7>` (the untagged BigCodeBench store) and `-bfcl` (the one-task BFCL continuation)
    m = B.SUFFIX_RE.match(f"bcbw-{TAG35_SR}-1234abc")
    assert (m.group("launch_tag"), m.group("task_tag"), m.group("window"), m.group("relaunch")) == (TAG35_SR, None, None, None) and B.suffix_task(m.group(0)) is None
    m = B.SUFFIX_RE.match(f"bcbw-{TAG35_SR}-1234abc-bfcl")
    assert (m.group("launch_tag"), m.group("task_tag"), m.group("window"), m.group("relaunch")) == (TAG35_SR, "-bfcl", None, None) and B.suffix_task(m.group(0)) == "bfcl"
    # the gate records: `gate_qwen35_35b_a3b_s2rl_<key>_bcbw-qwen35_35b_a3b_s2rl-<sha7>[-bfcl].json`
    assert B.gate_name_parts(f"gate_{TAG35_SR}_{Z.CTL35_R}_bcbw-{TAG35_SR}-1234abc.json") == {"tag": TAG35_SR, "arditi": None, "okey": Z.CTL35_R, "suffix": f"bcbw-{TAG35_SR}-1234abc"}
    assert B.gate_name_parts(f"gate_{TAG35_SR}_{Z.MIX35_R}_bcbw-{TAG35_SR}-1234abc-bfcl.json") == {"tag": TAG35_SR, "arditi": None, "okey": Z.MIX35_R, "suffix": f"bcbw-{TAG35_SR}-1234abc-bfcl"}
    # the full readers admit the selection in the family's s1 group; the 200-item readers do not (their emission pin holds them byte for byte)
    assert B.group_tags(TAG35, False, True, True, True, True)[SEL35_SR] == TAG35_SR and SEL35_SR not in B.group_tags(TAG35, False, True, True, True) and SEL35_SR not in B.group_tags(TAG35)
    assert BCF.SPEC.lecun_relu_row and BFF.SPEC.lecun_relu_row and not BCA.SPEC.lecun_relu_row and not BCS.SPEC.lecun_relu_row and not BFA.SPEC.lecun_relu_row and not BFS.SPEC.lecun_relu_row
    assert B.MODEL_KEYS[MK35].lecun_relu_row and not any(B.MODEL_KEYS[k].lecun_relu_row for k in B.MODEL_KEYS if k != MK35)
    # the declared row is grammar; its spelling without the trailing letter (the harness's zero-init `s2r`) is a declared key under an undeclared selection, named as such
    assert B.undeclared_row_dir_reason(SR_ROW_DIR) is None and not B.is_undeclared_row_dir_group(SR_ROW_DIR)
    r = B.undeclared_row_dir_reason(f"{MK35}-f22s2r")
    assert r.startswith(f"row dir '{MK35}-f22s2r' names the declared model key {MK35!r} but no declared row (selection 's2r'") and f"{MK35}-f22s2rl" in r
    # the spelling resolves through legacy_spellings.yaml (its row existed before this reader admitted the tag) to the 35B model row at strength 1.0, the s1 standup label
    sd = LC.Seeds.locate()
    mv = sd.model_for_view_legacy(TAG35_SR)
    assert mv["model_id"] == MK35 and mv["fit_model_id"] == f"{MK35}_base" and mv["transfer_kind"] == "base_to_instruct" and mv["source"] == "legacy_spellings.yaml model_keys"
    row = next(r for r in sd.legacy["model_keys"] if str(r.get("legacy")) == TAG35_SR)
    assert row["strength_nominal"] == 1.0 and row["fit_replicate_label"] == "s1"


@pytest.mark.parametrize("eval_id", sorted(MODULES))
def test_the_s2rl_rows_file_two_runs_per_eval_window_matched_on_the_u0_none_with_the_recorded_fit_seed(tmp_path, seeds, eval_id):
    """A dry read of the fixture's LeCun-init ReLU row beside the u0 family: TWO steered runs on the s2rl tag (the control and constitution ReLU refits), each resolved
    through the pins block by the launch set's pin key (the bank's lecun__ staged id), each contrasted against the u0 none at its own serving window across tags
    (sibling_none, cross_standup), each carrying the objects' RECORDED fit seed 0 (not the group's standup seed 1, which the launch row spells) with run.seed_base NULL
    like every row (SEED_AXIS_RULE); the harness roster lists a THIRD key under the selection that the launch never ran, and nothing waits on or refuses for it."""
    mod, cls = MODULES[eval_id]
    fx = Z.mirror(tmp_path, seeds, eval_id=eval_id, qwen35b_u0=True, qwen35b_s2rl=True)
    with open(fx["roster"], "r", encoding="utf-8") as fh:
        h_roster = yaml.safe_load(fh)
    assert sorted(h_roster["targets"][MK35]["views"][SEL35_SR]["cells"]) == sorted([Z.CTL35_R, Z.MIX35_R, Z.DIFF35_R]), "the harness lists three keys; two were launched"
    ad = Z.adapter(cls, fx, seeds)
    bt, facts, groups = Z.read_group(ad, TAG35)
    assert groups == [Z.TAG, TAG35] and facts["n_refusals"] == 0 and facts["runs"] == 9 and facts["n_not_run"] == 0 and facts["not_run"] == [], facts["refusals"]
    assert facts["lecun_relu_row_tag"] == TAG35_SR and facts["lecun_relu_row_rule"] == B.LECUN_RELU_ROW_RULE and TAG35_SR in facts["tags"]
    assert "dose_relu_row_tag" in facts and "dose_lora_row_tag" in facts, "the family's other declared rows are stamped whether or not this mirror holds their stores"
    inst = seeds.instrument_row(mod.INSTRUMENT_SEED, mod.EVAL_ID, model_id=MK35)["instrument_id"]
    sr_runs = {r["campaign_cell_id_legacy"]: r for r in bt["run"] if r["view_legacy"] == TAG35_SR}
    assert sorted(sr_runs) == sorted([Z.CTL35_R, Z.MIX35_R]), "two rows: the third key of the harness selection has no store, no launch row and no pin, so it is not a row"
    assert not any(Z.DIFF35_R in k for k in facts["rows"]) and not any(Z.DIFF35_R in str(x) for x in facts["refusals"])
    none_a = next(r for r in bt["run"] if r["view_legacy"] == TAG35_U0 and r["campaign_cell_id_legacy"] == NONE and r["context_window_tokens"] == Z.WINDOW35)
    for okey, r in sr_runs.items():
        assert r["model_id"] == MK35 and r["object_id"] == Z.OBJ35_SR[okey] and r["transfer_kind"] == "base_to_instruct" and r["instrument_id"] == inst
        assert r["run_id"] == mod.run_id_for(TAG35_SR, fx["keys"][f"{TAG35_SR}/{okey}"], okey, inst) and r["context_window_tokens"] == Z.WINDOW35
        assert r["extra"]["lecun_relu_row"] == {"rule": B.LECUN_RELU_ROW_RULE, "selection": SEL35_SR, "row_key": okey, "anchor_tuple": f"{Z.SEL35_U0}/{NONE}",
                                                "pin_key": Z.STAGED35_SR[okey], "fit_seed_recorded": 0, "standup_seed": 1}
        assert "dose_relu_row" not in r["extra"] and "dose_lora_row" not in r["extra"] and "unsteered_row" not in r["extra"] and r["extra"]["serving_window"]["max_model_len"] == Z.WINDOW35
        # the seeds: the objects' RECORDED fit seed (the sidecars' 0) on extra.fit_seed; the launch row's seed_base is the group's 1 (checked); run.seed_base NULL (SEED_AXIS_RULE)
        assert r["extra"]["fit_seed"] == 0 and r["seed_base"] is None and r["extra"]["seed_replicate_label"] == SEL35_SR and r["extra"]["seed_axis"]["seed_base"] is None
        assert r["extra"]["launch_set"]["pin_key"] == Z.STAGED35_SR[okey] and r["extra"]["launch_set"]["class"] == "arm" and r["extra"]["object_route"]["route"] == "launch_set_pin_key"
        assert facts["rows"][f"{TAG35_SR}|{okey}"]["object_id"] == Z.OBJ35_SR[okey] and facts["rows"][f"{TAG35_SR}|{okey}"]["staged_id"] == Z.STAGED35_SR[okey]
        cn = _contrast_of(bt, r["run_id"])
        assert cn is not None and cn["anchor_run_id"] == none_a["run_id"] and cn["anchor_kind"] == "sibling_none" and cn["extra"]["cross_standup"] is True
        assert cn["extra"]["anchor_view_legacy"] == TAG35_U0 and cn["extra"]["anchor_rule"] == B.WINDOW_ANCHOR_RULE
        assert cn["extra"]["window_matched"]["row_window"] == cn["extra"]["window_matched"]["anchor_window"] == Z.WINDOW35 and cn["extra"]["window_matched"]["anchor_row_key"] == f"{TAG35_U0}/{NONE}"
        assert r["extra"]["anchor"]["state"] == B.ANCHOR_CONTRASTED and r["extra"]["anchor"]["anchor_run_id"] == none_a["run_id"]
        c = X.counts(X.spec(eval_id=eval_id, shape=Z.SHAPE_FULL, **Z.SPECS35_SR[eval_id][okey]))
        value, _bound = Z.metrics_of(bt, r["run_id"], mod.recount_computation_id())
        assert abs(value["value"] - c["acc"]) < 1e-9
    assert len(bt["run_contrast"]) == 7        # the five of the u0 family + the two LeCun-init ReLU rows
    # the s1 LoRA rows still carry the group's fit seed, the u0 rows none
    assert {x["extra"]["fit_seed"] for x in bt["run"] if x["view_legacy"] == TAG35} == {1} and {x["extra"]["fit_seed"] for x in bt["run"] if x["view_legacy"] == TAG35_U0} == {None}
    # the source states: the two members join the row table and the three-number record; the third key is in neither
    by_name = {os.path.basename(st.input_uri).rsplit("#", 1)[0]: st for st in ad.source_states(X.ctx(), ad.discover()) if os.path.basename(st.input_uri).endswith(f"#{TAG35}")}
    assert {f"{SEL35_SR}/{k}" for k in sr_runs} <= set(by_name[B.LAUNCH_SET_FILE].current_members) and f"{SEL35_SR}/{Z.DIFF35_R}" not in by_name[B.LAUNCH_SET_FILE].current_members
    assert {f"{TAG35_SR}/{k}" for k in sr_runs} <= set(by_name[mod.THREE_NUMBERS_FILE].current_members) and f"{TAG35_SR}/{Z.DIFF35_R}" not in by_name[mod.THREE_NUMBERS_FILE].current_members
    # the gemma group beside it reads as it did
    _bt_g, facts_g, _ = Z.read_group(ad, Z.TAG)
    assert facts_g["n_refusals"] == 0 and facts_g["runs"] == 5 and "lecun_relu_row_tag" not in facts_g


def test_the_dose_rows_keep_a_null_fit_seed_beside_the_s2rl_rows_recorded_one(tmp_path, seeds):
    """One read over the dose-0.75 ReLU copies and the LeCun-init ReLU refits together: the dose rows file extra.fit_seed null (a copy's fit seed is not applicable), the
    s2rl rows the recorded 0, the s1 LoRA rows the group's 1 -- three provenances, every run.seed_base NULL."""
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", qwen35b_u0=True, qwen35b_d0p75_relul=True, qwen35b_s2rl=True)
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    bt, facts, _ = Z.read_group(ad, TAG35)
    assert facts["n_refusals"] == 0 and facts["runs"] == 12 and facts["n_not_run"] == 0, facts["refusals"]
    assert facts["dose_relu_row_tag"] == TAG35_DR and facts["lecun_relu_row_tag"] == TAG35_SR
    by_tag = {}
    for r in bt["run"]:
        by_tag.setdefault(r["view_legacy"], set()).add(r["extra"]["fit_seed"])
    assert by_tag[TAG35_DR] == {None} and by_tag[TAG35_SR] == {0} and by_tag[TAG35] == {1} and by_tag[TAG35_U0] == {None}
    assert {r["seed_base"] for r in bt["run"]} == {None}
    # the same two ReLU keys under the two row dirs are two rows each, on different objects
    for okey in (Z.CTL35_R, Z.MIX35_R):
        dose, refit = Z.run_at(bt, TAG35_DR, okey), Z.run_at(bt, TAG35_SR, okey)
        assert dose["object_id"] == Z.OBJ35_DR[okey] and refit["object_id"] == Z.OBJ35_SR[okey] and dose["run_id"] != refit["run_id"]
    assert len(bt["run_contrast"]) == 10      # the five of the u0 family + three dose rows + two refits


def test_an_unknown_lecun_pin_key_refuses_the_s2rl_row_by_name_never_a_guessed_object(tmp_path, seeds):
    """The launch row names a `lecun__` key the pins do not carry: the row refuses by name (the appendix route resolves the pin key alone), the other refit files, nothing
    is guessed from the store's own attestation."""
    bad_key = f"lecun__{Z.CTL35_R}_refit2"

    def rename_pin_key(doc):
        n = 0
        for ev in doc["evals"].values():
            for lr in ev["launch"]:
                if lr["tuple_id"] == f"{SEL35_SR}/{Z.CTL35_R}":
                    lr["pin_key"] = bad_key
                    n += 1
        assert n == 2
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", qwen35b_u0=True, qwen35b_s2rl=True, launch_override=rename_pin_key)
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    bt, facts, _ = Z.read_group(ad, TAG35)
    assert facts["runs"] == 8 and facts["n_refusals"] == 1, facts["refusals"]
    assert facts["refusals"][0].startswith(f"{TAG35_SR}|{Z.CTL35_R} -- object: ") and f"{MK35}.{bad_key}" in facts["refusals"][0], facts["refusals"]
    assert [r["campaign_cell_id_legacy"] for r in bt["run"] if r["view_legacy"] == TAG35_SR] == [Z.MIX35_R]
    assert Z.OBJ35_SR[Z.CTL35_R] not in {r["object_id"] for r in bt["run"]}


def test_a_second_none_store_under_the_u0_row_binds_no_row_beside_the_s2rl_rows(tmp_path, seeds):
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", qwen35b_u0=True, qwen35b_s2rl=True, u0_second_none_store=True)
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    bt, facts, _ = Z.read_group(ad, TAG35)
    assert facts["runs"] == 9 and facts["n_refusals"] == 1, facts["refusals"]
    assert facts["refusals"][0].startswith(f"{TAG35_U0}|{NONE} -- ") and Z.TREE_DR[:7] in facts["refusals"][0], facts["refusals"]
    nones = [r for r in bt["run"] if r["view_legacy"] == TAG35_U0 and r["campaign_cell_id_legacy"] == NONE and r["context_window_tokens"] == Z.WINDOW35]
    assert len(nones) == 1 and nones[0]["code_sha"] == Z.TREE
    for r in (x for x in bt["run"] if x["view_legacy"] == TAG35_SR):
        assert _contrast_of(bt, r["run_id"])["anchor_run_id"] == nones[0]["run_id"]


def test_an_undeclared_s2r_row_dir_refuses_by_name_as_a_whole_group(tmp_path, seeds):
    name = f"{MK35}-f22s2r"
    fx = Z.mirror(tmp_path, seeds, eval_id="bigcodebench", undeclared_row_dirs=(name,))
    ad = Z.adapter(BCF.BigcodebenchFullAdapter, fx, seeds)
    groups = _groups(ad)
    assert name in groups and TAG35_SR not in groups
    bt, facts, _ = Z.read_group(ad, name)
    assert bt == {} and facts["runs"] == 0 and facts["refuse"] == [B.undeclared_row_dir_reason(name)]
    assert "no declared row (selection 's2r'" in facts["refuse"][0] and f"{MK35}-f22s2rl" in facts["refuse"][0]


# --------------------------------------------------------------------------------------------
# the s2rl row dir is declared only for a reader that reads it (lecun_relu_row_admitted, 2026-10-07; the review of 797f57df2): elsewhere a whole-group refusal
# --------------------------------------------------------------------------------------------

GEMMA_SR_ROW_DIR = f"{Z.MK}-f22s2rl"
ALLMODELS = {"bigcodebench": BCA.BigcodebenchAllmodelsAdapter, "bfcl": BFA.BfclAllmodelsAdapter}


def test_the_s2rl_row_dir_is_declared_only_where_the_reader_reads_it():
    """The grammar alone reads `(model key)-f22s2rl` for every key; a READER reads it only where it admits the row AND the family runs it. The other optional rows
    (it, u0, the dose rows) stay grammar for every reader: their stores refuse BY ROW where the group admits no selection, the design main pinned."""
    assert B.lecun_relu_row_admitted(MK35, None) and B.lecun_relu_row_admitted(Z.MK, None), "no reader: the grammar's answer"
    assert B.lecun_relu_row_admitted(MK35, BCF.SPEC) and B.lecun_relu_row_admitted(MK35, BFF.SPEC)
    assert not any(B.lecun_relu_row_admitted(MK35, sp) for sp in (BCA.SPEC, BCS.SPEC, BFA.SPEC, BFS.SPEC)), "the 200-item readers admit no s2rl row"
    assert not B.lecun_relu_row_admitted(Z.MK, BCF.SPEC) and not B.lecun_relu_row_admitted("glm46vflash", BCF.SPEC) and not B.lecun_relu_row_admitted(None, BCF.SPEC)
    # tag_of_row_dir: the grammar with no reader, the reader's reading with one
    assert B.tag_of_row_dir(SR_ROW_DIR) == TAG35_SR and B.tag_of_row_dir(SR_ROW_DIR, BCF.SPEC) == TAG35_SR and B.tag_of_row_dir(SR_ROW_DIR, BFF.SPEC) == TAG35_SR
    assert B.tag_of_row_dir(SR_ROW_DIR, BCA.SPEC) is None and B.tag_of_row_dir(SR_ROW_DIR, BFS.SPEC) is None
    assert B.tag_of_row_dir(GEMMA_SR_ROW_DIR) == f"{Z.MK}_s2rl" and B.tag_of_row_dir(GEMMA_SR_ROW_DIR, BCF.SPEC) is None and B.tag_of_row_dir(GEMMA_SR_ROW_DIR, BCA.SPEC) is None
    assert B.is_undeclared_row_dir_group(SR_ROW_DIR, BCA.SPEC) and not B.is_undeclared_row_dir_group(SR_ROW_DIR, BCF.SPEC) and not B.is_undeclared_row_dir_group(SR_ROW_DIR)
    assert B.is_undeclared_row_dir_group(GEMMA_SR_ROW_DIR, BCF.SPEC) and not B.is_undeclared_row_dir_group(GEMMA_SR_ROW_DIR)
    # the reason names what does not read the row: the reader, the family, or both -- and lists the rows that reader reads for the key (never -f22s2rl where it does not)
    assert B.undeclared_row_dir_reason(SR_ROW_DIR) is None and B.undeclared_row_dir_reason(SR_ROW_DIR, BCF.SPEC) is None and B.undeclared_row_dir_reason(GEMMA_SR_ROW_DIR) is None
    r = B.undeclared_row_dir_reason(SR_ROW_DIR, BCA.SPEC)
    assert r.startswith(f"row dir {SR_ROW_DIR!r} names the declared model key {MK35!r} under the LeCun-init ReLU selection 's2rl', which this reader does not read for it (this reader admits no LeCun-init ReLU row")
    assert "runs no LeCun-init ReLU row" not in r and f"the rows this reader reads for {MK35}: {MK35}-f22s1, {MK35}-f22s1p, {MK35}-f22u0, {MK35}-f22d0p75_relul, {MK35}-f22d1p25)" in r
    assert "refused as a WHOLE GROUP by name" in r and B.LECUN_RELU_ROW_RULE in r
    r = B.undeclared_row_dir_reason(GEMMA_SR_ROW_DIR, BCF.SPEC)
    assert f"which this reader does not read for it (the family {Z.MK} runs no LeCun-init ReLU row (the model-key table's lecun_relu_row); the rows this reader reads for {Z.MK}: " in r
    listing = r.split("the rows this reader reads for", 1)[1].split("):", 1)[0]
    assert "this reader admits no" not in r and listing == f" {Z.MK}: {Z.MK}-f22s1, {Z.MK}-f22s1p, {Z.MK}-f22it, {Z.MK}-f22", listing
    r = B.undeclared_row_dir_reason(GEMMA_SR_ROW_DIR, BCA.SPEC)
    assert "this reader admits no LeCun-init ReLU row" in r and f"the family {Z.MK} runs no LeCun-init ReLU row" in r
    # the no-reader listing keeps the table's answer (a family that runs no s2rl row never lists it); a reader that admits none lists it for no family
    assert f"{MK35}-f22s2rl" in B.declared_row_dirs_of(MK35) and f"{MK35}-f22s2rl" in B.declared_row_dirs_of(MK35, BCF.SPEC) and f"{MK35}-f22s2rl" not in B.declared_row_dirs_of(MK35, BCA.SPEC)
    assert f"{Z.MK}-f22s2rl" not in B.declared_row_dirs_of(Z.MK) and f"{Z.MK}-f22s2rl" not in B.declared_row_dirs_of(Z.MK, BCF.SPEC)
    assert "qwen35_4b-f22s2rl" not in B.undeclared_row_dir_reason("qwen35_4b-f22") and f"{MK35}-f22s2rl" in B.undeclared_row_dir_reason(BARE_ROW_DIR)
    # the other optional selections are grammar for every reader (their stores refuse by row, never the undeclared route)
    assert B.tag_of_row_dir("nemotron3nano4b-f22u0", BCA.SPEC) == "nemotron3nano4b_u0" and B.tag_of_row_dir(f"{Z.MK}-f22d1p25", BCA.SPEC) == f"{Z.MK}_d1p25"
    assert B.tag_of_row_dir("qwen35_4b-f22it", BCS.SPEC) == "qwen35_4b_it" and B.tag_of_row_dir(f"{Z.MK}-f22d0p75_relul", BFS.SPEC) == f"{Z.MK}_d0p75_relul"
    assert B.undeclared_row_dir_reason("nemotron3nano4b-f22u0", BCA.SPEC) is None and not B.is_undeclared_row_dir_group(f"{Z.MK}-f22d1p25", BCA.SPEC)


@pytest.mark.parametrize("eval_id", sorted(MODULES))
def test_an_s2rl_row_dir_of_a_family_that_runs_no_lecun_relu_row_refuses_as_a_whole_group_on_both_full_readers(tmp_path, seeds, eval_id):
    """The review of 797f57df2: a store under `gemma4e4b-f22s2rl` (its launch tag `gemma4e4b_s2rl`, the producer's spelling) used to join the gemma s1 group and refuse
    BY ROW, leaving the group's status ok = 1 while the five s1 rows filed. It is a group of its own now, refused as a WHOLE by name (facts.refuse: the family runs
    no LeCun-init ReLU row), the s1 group reads with no refusal at all, and a gate record named for the row is aliased to no group."""
    _mod, cls = MODULES[eval_id]
    fx = Z.mirror(tmp_path, seeds, eval_id=eval_id, undeclared_row_dirs=(GEMMA_SR_ROW_DIR,))
    gate = os.path.join(fx["root"], B.RECORDS_DIR, f"gate_{Z.MK}_s2rl_{NONE}_bcbw-{Z.MK}_s2rl-{Z.TREE[:7]}.json")
    assert B.gate_name_parts(os.path.basename(gate))["tag"] == f"{Z.MK}_s2rl"
    X.write(gate, {"verdict": "PASS", "tag": f"{Z.MK}_s2rl"})
    ad = Z.adapter(cls, fx, seeds)
    groups = _groups(ad)
    assert sorted(groups) == sorted([Z.TAG, GEMMA_SR_ROW_DIR]), sorted(groups)
    inputs = groups[GEMMA_SR_ROW_DIR]
    assert inputs and all(os.path.relpath(i.uri, fx["root"]).split(os.sep)[:2] == [B.STORES_HEAD, GEMMA_SR_ROW_DIR] for i in inputs), "store inputs alone: no record is aliased to it"
    assert any(f"{os.sep}bcbw-{Z.MK}_s2rl-{Z.TREE[:7]}{os.sep}" in i.uri for i in inputs), "the fixture's store carries the row dir's own launch tag"
    assert not any(os.path.basename(i.uri).startswith(os.path.basename(gate)) for v in groups.values() for i in v), "the gate record of a row this reader does not read is not this source's"
    bt, facts, _ = Z.read_group(ad, GEMMA_SR_ROW_DIR)
    reason = B.undeclared_row_dir_reason(GEMMA_SR_ROW_DIR, cls.SPEC)
    assert bt == {} and facts["runs"] == 0 and facts["n_refusals"] == 1 and facts["undeclared_row_dir"] == GEMMA_SR_ROW_DIR and facts["n_store_inputs"] == len(inputs)
    assert facts["refuse"] == [reason] and facts["refusals"] == [f"{GEMMA_SR_ROW_DIR} -- {reason}"] and facts["undeclared_row_dir_rule"] == B.UNDECLARED_ROW_DIR_RULE
    assert f"the family {Z.MK} runs no LeCun-init ReLU row" in reason and "this reader admits no" not in reason, reason
    assert ad.refusals.items[0]["input"] == os.path.join(fx["root"], B.STORES_HEAD, GEMMA_SR_ROW_DIR)
    assert B.undeclared_row_dir_reason(GEMMA_SR_ROW_DIR) is None, "the grammar alone reads the row dir: the refusal is THIS reader's"
    # the s1 group beside it reads as if the directory were not there: no by-name row refusal, no refuse key
    bt, facts, _ = Z.read_group(ad, Z.TAG)
    assert facts["runs"] == 5 and facts["n_refusals"] == 0 and "refuse" not in facts and f"{Z.MK}_s2rl" not in {r["view_legacy"] for r in bt["run"]}, facts["refusals"]


@pytest.mark.parametrize("eval_id", sorted(ALLMODELS))
def test_an_s2rl_row_dir_under_a_200_item_reader_refuses_as_a_whole_group_by_name(tmp_path, seeds, eval_id):
    """A 200-item reader admits no s2rl row (its emission is pinned): the 35B's own `-f22s2rl` and the gemma one are each a group of their own that refuses as a whole
    -- the reason naming the reader for the 35B (the family runs the row) and the reader AND the family for gemma -- never a store merged into an s1 group."""
    fx = Z.mirror(tmp_path, seeds, eval_id=eval_id, undeclared_row_dirs=(GEMMA_SR_ROW_DIR, SR_ROW_DIR))
    ad = X.adapter(ALLMODELS[eval_id], fx, seeds)
    groups = _groups(ad)
    assert sorted(groups) == sorted([Z.TAG, GEMMA_SR_ROW_DIR, SR_ROW_DIR]) and TAG35 not in groups and TAG35_SR not in groups, sorted(groups)
    for name, reader_clause, family_clause in ((SR_ROW_DIR, True, False), (GEMMA_SR_ROW_DIR, True, True)):
        bt, facts, _ = Z.read_group(ad, name)
        reason = B.undeclared_row_dir_reason(name, ALLMODELS[eval_id].SPEC)
        assert bt == {} and facts["runs"] == 0 and facts["n_refusals"] == 1 and facts["undeclared_row_dir"] == name and facts["refuse"] == [reason], (name, facts["refusals"])
        assert ("this reader admits no LeCun-init ReLU row" in reason) is reader_clause and ("runs no LeCun-init ReLU row" in reason) is family_clause, reason
        assert f"{name.split('-f22', 1)[0]}-f22s2rl" not in reason.split("the rows this reader reads for", 1)[1].split(")", 1)[0], "the listing is the rows THIS reader reads"


# test_an_unadmitted_s2rl_row_dir_reads_ok_0_on_v_source_status_with_extra_refuse_and_fails_the_paper_guard: deleted 2026-10-08 — it compared the adapter's refusal rows against the paper library
# (paper/paperlib/registry_source.py + lane_guard.py), which left the tree with the earlier figure build; the adapter's own refusal rows stay pinned above.


