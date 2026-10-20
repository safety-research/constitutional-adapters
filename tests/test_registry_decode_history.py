"""Decode history applied as DATA, never as a matcher rule -- the two registry corrections of 2026-10-14.

(A) Best-of-N -- the rule, first-hand (`decode-bon-canonical-T1-live-non-default-2026-10-14`): canonical best-of-N STAYS the
    temperature-1.0 512-store + 4,096-climb instrument `bon_512store_4096climb_T1`, so the nine `<model>|bon|{off,not_available}|grandfathered_T1`
    rows -- the rows that instrument's draws send -- are LIVE and NON-DEFAULT: the 2026-09-17 supersession by the `|vendor_preset` rows is
    cleared, `is_default_for_lookup 0` kept, so the vendor rows stay THE defaults `resolve()` returns. Before this the matcher
    (`v_run_decode_policy_match`: a superseded row is a candidate only for a run generated before its successor took effect) read every
    draw of the canonical instrument after 2026-09-17 as `no_match` once its sent set was recorded. Precedent: the agentic
    `runner_forced_T1` rows (`decode-vendor-default-all-evals-2026-09-21`; tests/test_registry_decode_vendor_default_agentic.py).
    The loader fix behind it: `superseded_by_policy_id` is now STATED on every decode_policy row (null where the seed names none), because
    the reducer keeps a column an event omits -- dropping the pointer from the seed alone changed nothing in the database.

(B) Multi-turn -- the registry owner's ruling (2026-10-14): the three canonical cap-4,096 multi-turn seeds declare `decode_seed_rule none`
    (the paper's filed waves sent nothing per turn, unseeded) but bound the three `<model>|multi_turn|off|vendor_preset` rows of
    qwen35_4b_demo / qwen35_35b_a3b / gemma4_31b, which require `per_goal_turn` -- and every run on that route WAS seeded per (goal, turn).
    `decode_seed_rule` is instrument and run identity, so the route is SPLIT: the generation_config seeds keep the paper's refs under
    `none`; new `multiturn_*_cap4096_vendor_preset` seeds carry the three vendor refs under `per_goal_turn`; the two non_template variants
    are corrected in place. HARD REQUIREMENT, pinned by sha: the paper models' expanded multi-turn instrument ids do not move.

Nothing here touches a view, a migration or a registered run; existing runs keep their instrument ids.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sqlite3

import pytest
import yaml

from sisteer.registry import db as DB
from sisteer.registry import decode_match as DM
from sisteer.registry import decode_policy as DP
from sisteer.registry import decode_precedence as PREC
from sisteer.registry import harvest as H
from sisteer.registry import identity
from sisteer.registry import jobplan as JP
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry.adapters import dimensions as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = os.path.join(ROOT, "configs", "registry")
PROD = synth.producer("tests.registry_decode_history")

RULING = "decode-bon-canonical-T1-live-non-default-2026-10-14"
BON_EVAL = "strongreject_bon"
#: (model, thinking arm) of the nine grandfathered best-of-N rows: the five served models, the two pipeline-test models, the worked
#: example's twin and the Llama `not_available` twin
BON_KEYS = (("gemma4e4b_it", "off"), ("qwen35_4b_it", "off"), ("nemotron3nano4b", "off"), ("llama8b_it", "off"), ("llama8b_it", "not_available"),
            ("glm46vflash", "off"), ("qwen35_35b_a3b", "off"), ("gemma4_31b", "off"), ("qwen35_4b_demo", "off"))
#: generated after the 2026-09-17 ruling (the paper models) and after the 2026-09-29 seeding of the test models' rows: the window in which
#: the supersession made the canonical instrument's draws match nothing
AFTER = "2026-10-01T00:00:00.000Z"
BEFORE = "2026-09-15T00:00:00.000Z"

MT_EVALS = ("crescendo", "pair", "actorattack")
PAPER_SERVED = ("gemma4e4b_it", "qwen35_4b_it", "nemotron3nano4b", "llama8b_it", "glm46vflash")
VENDOR_ROUTE = ("qwen35_4b_demo", "qwen35_35b_a3b", "gemma4_31b")
VENDOR_REFS = tuple(f"{m}|multi_turn|off|vendor_preset" for m in VENDOR_ROUTE)
PAPER_REFS = ("gemma4e4b_it|multi_turn|off|generation_config", "qwen35_4b_it|multi_turn|off|server_default",
              "nemotron3nano4b|multi_turn|off|generation_config", "llama8b_it|multi_turn|off|generation_config",
              "glm46vflash|multi_turn|off|generation_config")
#: the paper models' expanded multi-turn instrument ids as they stood BEFORE the split (dimensions export of origin/main e293e9ff1), which
#: the split must not move: one shared sends-nothing row per attack (gemma4e4b_it, glm46vflash, llama8b_it, nemotron3nano4b) and Qwen's
#: server_default row -- every run, judgment and roster entry of the paper's multi-turn waves joins on these
PAPER_MT_IDS = {
    ("crescendo", "shared"): "b06cf663f8a0ff5c569e51517b3442afcf984fe3d7216aff8987eef77150f5ef",
    ("crescendo", "qwen35_4b_it"): "33f80ed1d4f536e34c870fa36c2d88d90c7623ea17a9d4db178fea008c12ac85",
    ("pair", "shared"): "408e01019d116874811702f73373f5735e556bbfe51d97664ed25b170835abc7",
    ("pair", "qwen35_4b_it"): "373118ba258c81ebc306e1d7d3ace8acdc53807a96b7277d3931ef961b2311ed",
    ("actorattack", "shared"): "78daff8eb636d35ffdb040bd176fa88ee688c8a4d5b64f08812a89a657e3070b",
    ("actorattack", "qwen35_4b_it"): "f8832b53b08131258d0e5c9e15a16a2888f8a977271d52dde270d52b1af00203",
}
SHARED_MODELS = ["gemma4e4b_it", "glm46vflash", "llama8b_it", "nemotron3nano4b"]
#: seed keys that legitimately differ between a generation_config seed and its vendor_preset sibling (the route, its provenance prose)
ROUTE_MAY_DIFFER = {"instrument_id", "decode_source", "decode_policy_refs", "decode_seed_rule", "decode_seed_formula",
                    "paper_status_ruling_id", "notes", "source"}


@pytest.fixture(scope="module")
def seeds():
    return D.load_seeds(SEEDS)


@pytest.fixture(scope="module")
def bundle(seeds):
    return D.build_records(seeds, PROD, now="2026-10-14T12:00:00.000Z")


@pytest.fixture(scope="module")
def idx(bundle):
    return bundle.policies


@pytest.fixture(scope="module")
def seed_index():
    return JP.load_seed_index(SEEDS)


@pytest.fixture(scope="module")
def pf():
    return DP.PolicyFile.load(os.path.join(SEEDS, "decode_policy.json"))


@pytest.fixture(scope="module")
def seeded_db(tmp_path_factory):
    """A fresh registry with the real seeds ingested (the loader's own path)."""
    root = str(tmp_path_factory.mktemp("reg") / "root")
    H.ensure_layout(root)
    _, paths = D.write_seed_records(root, SEEDS, producer=PROD, now="2026-10-14T12:00:00.000Z")
    assert paths
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    return DB.db_path(root)


def _seed_rows():
    """The seed's policy rows by BARE ref (a seed ref may carry its `@vN`; the rows this module reads are all version 1)."""
    with open(os.path.join(SEEDS, "decode_policy_seed.yaml"), encoding="utf-8") as fh:
        return {D.parse_ref(str(r["policy_ref"]))[0]: r for r in yaml.safe_load(fh)["policies"]}


def _instrument_seeds():
    with open(os.path.join(SEEDS, "instruments.yaml"), encoding="utf-8") as fh:
        return {str(r["instrument_id"]): r for r in yaml.safe_load(fh)["instruments"]}


# --------------------------------------------------------------------------------------------
# (A) best-of-N: the nine grandfathered rows are live and non-default; the loader states the column
# --------------------------------------------------------------------------------------------


def test_the_ruling_is_seeded_and_the_nine_grandfathered_rows_are_live_and_non_default(seeds, idx):
    text = seeds.decode["rulings"][RULING]
    assert "STAYS" in text and "bon_512store_4096climb_T1" in text and "LIVE and NON-DEFAULT" in text
    assert "runner_forced_T1" in text and "never as a matcher rule" in text and "superseded_by_policy_id on every" in text
    rows = _seed_rows()
    for model, arm in BON_KEYS:
        ref = f"{model}|bon|{arm}|grandfathered_T1"
        p, e = idx.rows[f"{ref}@v1"]
        assert p["is_default"] == 0 and p["superseded_by_policy_id"] is None and not e.get("superseded_by_unresolved"), ref
        assert p["decode_source"] == "grandfathered_T1" and p["seed_rule"] == "per_draw" and e["mode"] == "grandfathered_T1", ref
        assert idx.numbers(f"{ref}@v1") == {"temperature": 1.0, "top_p": None, "top_k": None, "min_p": None, "repetition_penalty": None, "presence_penalty": None}, ref
        assert RULING in str(p.get("notes")), ref                       # the row cites the ruling; its own ruling_id stays the numbers' ruling
        assert rows[ref]["superseded_by_policy_id"] is None and "superseded_by_policy_id" in rows[ref], ref     # stated null in the seed itself
        # the vendor sibling stays THE default of the key, not superseded, and the key has exactly one default
        v, ve = idx.rows[f"{model}|bon|{arm}|vendor_preset@v1"]
        assert v["is_default"] == 1 and v["superseded_by_policy_id"] is None and ve["mode"] == "vendor_preset", ref
        assert RULING in str(v.get("notes")) or model == "qwen35_4b_demo", ref     # the demo's vendor twin never named a supersession
    assert idx.no_default_keys == []
    # the eight `off` grandfathered rows + the twin are the only grandfathered_T1 rows, and the canonical instrument names the eight
    grand = sorted(pid for pid in idx.rows if pid.endswith("|grandfathered_T1@v1"))
    assert grand == sorted(f"{m}|bon|{a}|grandfathered_T1@v1" for m, a in BON_KEYS)
    refs = _instrument_seeds()["bon_512store_4096climb_T1"]["decode_policy_refs"]
    assert sorted(refs) == sorted(f"{m}|bon|off|grandfathered_T1" for m, a in BON_KEYS if a == "off")


def test_the_loader_states_the_supersession_column_on_every_row_null_where_the_seed_names_none(seeds, idx, bundle):
    rows = _seed_rows()
    by_id = {D.policy_id_for(D.parse_ref(str(r["policy_ref"]))[0], int(r["policy_version"])): r for r in seeds.decode["policies"]}
    stated = {pid: p["superseded_by_policy_id"] for pid, (p, _) in idx.rows.items()}
    assert all("superseded_by_policy_id" in p for p, _ in idx.rows.values())          # the column is a statement on EVERY row
    for pid, (p, e) in idx.rows.items():
        seed = by_id.get(pid) if not e.get("wildcard_expanded_from") else rows.get(D.parse_ref(e["wildcard_expanded_from"])[0])
        assert seed is not None, pid
        if seed.get("superseded_by_policy_id"):
            assert stated[pid] is not None or e.get("superseded_by_unresolved"), pid
        else:
            assert stated[pid] is None and not e.get("superseded_by_unresolved"), pid
    # 31 live pointers remain (40 before the ruling: the nine best-of-N pointers are gone, nothing else moved)
    assert sum(1 for v in stated.values() if v) == 31
    # the records the harvester folds carry the explicit null too (the fold keeps an omitted key; a stated null replaces a value)
    recs = {r["payload"]["policy_id"]: r["payload"] for r in bundle.records["decode_policy"]}
    for model, arm in BON_KEYS:
        rec = recs[f"{model}|bon|{arm}|grandfathered_T1@v1"]
        assert "superseded_by_policy_id" in rec and rec["superseded_by_policy_id"] is None


def _load(root: str, seeds_dir: str, now: str) -> None:
    _, paths = D.write_seed_records(root, seeds_dir, producer=PROD, now=now, verify_si=False)
    assert paths, "the edited seeds must publish a NEW revision (their sha changed)"
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals


def _pointers(root: str):
    con = sqlite3.connect(f"file:{DB.db_path(root)}?mode=ro", uri=True)
    try:
        return {r[0]: r[1] for r in con.execute("SELECT policy_id, superseded_by_policy_id FROM decode_policy ORDER BY policy_id")}
    finally:
        con.close()


def test_a_pointer_the_seed_drops_folds_the_live_row_to_null_and_stating_null_elsewhere_changes_nothing(tmp_path):
    """The fold regression the loader fix exists for. A registry that carries the pointer (the live database as it stood before the ruling:
    the seed as committed with gemma4e4b_it's grandfathered row pointing at its vendor row) reloaded with the committed seed -- the pointer
    dropped -- reads NULL on that row; every other row's pointer is byte-identical between the two loads, so stating null on a row that
    never had a successor changes no value."""
    seeds_v1 = str(tmp_path / "seeds_v1")
    shutil.copytree(SEEDS, seeds_v1)
    path = os.path.join(seeds_v1, "decode_policy_seed.yaml")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    target = "gemma4e4b_it|bon|off|grandfathered_T1"
    hits = [r for r in data["policies"] if r["policy_ref"] == target]
    assert len(hits) == 1 and hits[0]["superseded_by_policy_id"] is None
    hits[0]["superseded_by_policy_id"] = "gemma4e4b_it|bon|off|vendor_preset"       # the pre-ruling state
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True, width=10_000)
    root = str(tmp_path / "root")
    H.ensure_layout(root)
    _load(root, seeds_v1, "2026-10-13T00:00:00.000Z")
    v1 = _pointers(root)
    assert v1[target + "@v1"] == "gemma4e4b_it|bon|off|vendor_preset@v1"
    # the committed seed: the pointer dropped -> a NEW revision whose records state null -> the live row folds to NULL
    _load(root, SEEDS, "2026-10-14T00:00:00.000Z")
    v2 = _pointers(root)
    assert v2[target + "@v1"] is None
    assert set(v1) == set(v2)
    assert {k: v for k, v in v1.items() if k != target + "@v1"} == {k: v for k, v in v2.items() if k != target + "@v1"}
    assert sum(1 for v in v2.values() if v) == 31 and sum(1 for v in v1.values() if v) == 32


def _probe_conn(seeded_db):
    """The seeded policy table read as is; a TEMP `run` shadows the real (empty) table for the SQL object's body and the family-default
    expression (the shape tests/test_registry_decode_vendor_default_agentic.py uses)."""
    conn = sqlite3.connect(seeded_db)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TEMP TABLE run (run_id TEXT PRIMARY KEY, campaign TEXT, model_id TEXT, eval_id TEXT, thinking_mode TEXT, decode_seed_rule TEXT, "
                 "decode_sent_json TEXT, decode_temperature_sent REAL, decode_top_p_sent REAL, decode_top_k_sent INTEGER, "
                 "generated_utc TEXT, started_utc TEXT, queued_utc TEXT, decode_policy_id TEXT, decode_source TEXT)")
    return conn


def _insert(conn, rid, model, eval_id, thinking, seed_rule, sent, generated, source):
    conn.execute("INSERT INTO run VALUES (?, 'probe', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                 (rid, model, eval_id, thinking, seed_rule, json.dumps(sent), sent.get("temperature"), sent.get("top_p"), sent.get("top_k"),
                  generated, generated, generated, source))


def test_a_canonical_instrument_draw_after_the_vendor_ruling_reads_exact_grandfathered_and_non_default(idx, seeded_db):
    """A run generated after 2026-09-17 under the T1 instrument that sent {temperature: 1.0} alone (what bon_served sends under
    BON_HARNESS_DEFAULT; the seed rule per_draw) reads `exact` -> `<model>|bon|<arm>|grandfathered_T1@v1` with decode_is_family_default 0;
    a vendor-preset draw (the vendor row's pinned numbers) still reads `exact` -> the vendor row with decode_is_family_default 1. Both
    before and after the ruling dates; nothing is ambiguous. Before this correction the AFTER probe read no_match on every key."""
    conn = _probe_conn(seeded_db)
    try:
        want = {}
        for model, arm in BON_KEYS:
            t1 = f"{model}|bon|{arm}|grandfathered_T1@v1"
            vendor = f"{model}|bon|{arm}|vendor_preset@v1"
            vp, _ = idx.rows[vendor]
            vsent = {k: vp[k] for k in DM.PARAMS if vp.get(k) is not None}
            assert "top_p" in vsent and vsent["temperature"] is not None, vendor       # a vendor draw sends more than the temperature
            for when in (BEFORE, AFTER):
                rid = f"t1:{t1}:{when[:10]}"
                _insert(conn, rid, model, BON_EVAL, arm, "per_draw", {"temperature": 1.0}, when, "grandfathered_T1")
                want[rid] = (t1, vendor, 0)
                rid = f"vendor:{vendor}:{when[:10]}"
                _insert(conn, rid, model, BON_EVAL, arm, vp["seed_rule"], vsent, when, "generation_config")
                want[rid] = (vendor, vendor, 1)
        got = {r["run_id"]: dict(r) for r in conn.execute(f"SELECT run_id, policy_id, match, candidates_json, in_scope_json FROM ({DM.match_select_sql()})")}
        for rid, e in DM.tiebreak(conn, f"({DM.match_select_sql()})").items():
            assert not e["policy_id"], (rid, e)                    # nothing here needs the tie-break
        bad = []
        for rid, (policy, family, _) in want.items():
            g = got[rid]
            if (g["match"], g["policy_id"]) != ("exact", policy):
                bad.append((rid, g["match"], g["policy_id"], g["candidates_json"], DM.no_match_reasons(conn, rid)))
                continue
            scope = json.loads(g["in_scope_json"])
            assert policy in scope and family in scope, (rid, scope)
            assert json.loads(g["candidates_json"]) == [policy], (rid, g["candidates_json"])
            conn.execute("UPDATE run SET decode_policy_id = ? WHERE run_id = ?", (policy, rid))
        assert not bad, bad
        fam = {r["run_id"]: (r["decode_family_default_policy_id"], r["decode_is_family_default"])
               for r in conn.execute(f"SELECT r.run_id, {S.decode_family_default_sql('r')} FROM run r")}
        for rid, (policy, family, is_default) in want.items():
            assert fam[rid] == (family, is_default), (rid, fam[rid], (family, is_default))
    finally:
        conn.close()


def test_resolve_the_export_and_decode_precedence_are_unchanged(pf, seeded_db):
    """`resolve()` returns the vendor row for every key (the default flag, which the ruling kept); the grandfathered row is readable by its
    bare ref and now LIVE (`find` lists it); the committed export is the fresh registry's export; `decode_precedence` reads a T1 row as `t1`
    and a vendor row as `vendor` from the matched policy id, as before."""
    for model, arm in BON_KEYS:
        p = pf.resolve(model, eval_id=BON_EVAL, thinking_mode=arm)
        assert p.policy_id == f"{model}|bon|{arm}|vendor_preset@v1" and p.is_default == 1 and p.superseded_by_policy_id is None, model
        g = pf.by_ref(f"{model}|bon|{arm}|grandfathered_T1")
        assert g.is_default == 0 and g.superseded_by_policy_id is None and g.request_params() == {"temperature": 1.0}, model
        live = {x.policy_id for x in pf.find(model, eval_id=BON_EVAL, thinking_mode=arm)}
        assert live == {p.policy_id, g.policy_id}, (model, live)
        assert PREC.decode_kind("grandfathered_T1", g.policy_id) == "t1" and PREC.decode_kind("grandfathered_T1", p.policy_id) == "vendor"
        assert PREC.decode_kind("grandfathered_T1", None) == "t1" and PREC.decode_kind("generation_config", None) == "vendor"
    assert pf.model_independent(eval_id=BON_EVAL, thinking_mode="off", mode="grandfathered_T1").request_params() == {"temperature": 1.0}
    with DB.reading(seeded_db) as conn:
        assert DP.check_all(conn, DP.export_paths()) == []


# --------------------------------------------------------------------------------------------
# (B) multi-turn: the vendor-preset route is its own seed (per_goal_turn); the paper models' ids do not move
# --------------------------------------------------------------------------------------------


def test_the_three_multi_turn_routes_are_split_by_seed_rule_and_identical_otherwise():
    seeds = _instrument_seeds()
    for ev in MT_EVALS:
        paper, vendor = seeds[f"multiturn_{ev}_cap4096_generation_config"], seeds[f"multiturn_{ev}_cap4096_vendor_preset"]
        assert tuple(paper["decode_policy_refs"]) == PAPER_REFS and paper["decode_seed_rule"] == "none", ev
        assert tuple(vendor["decode_policy_refs"]) == VENDOR_REFS and vendor["decode_seed_rule"] == "per_goal_turn", ev
        assert vendor["paper_status"] == paper["paper_status"] == "canonical" and "its own canonical seed" in vendor["paper_status_ruling_id"], ev
        assert "per_goal_turn" in vendor["decode_seed_formula"] or "(seed_base, goal, turn)" in vendor["decode_seed_formula"], ev
        diff = {k for k in set(paper) | set(vendor) if k not in ROUTE_MAY_DIFFER and paper.get(k) != vendor.get(k)}
        assert diff == set(), (ev, diff)
    for ev in ("crescendo", "actorattack"):
        ntpl = seeds[f"multiturn_{ev}_cap4096_generation_config_non_template"]
        assert ntpl["decode_seed_rule"] == "per_goal_turn" and ntpl["variant_of"] == f"multiturn_{ev}_cap4096_vendor_preset", ev
        assert tuple(ntpl["decode_policy_refs"]) == ("qwen35_35b_a3b|multi_turn|off|vendor_preset",) and ntpl["paper_status"] == "sensitivity", ev
    # the three vendor rows do require per_goal_turn (seed_rule on the row): a `none` instrument could never match them
    rows = _seed_rows()
    for ref in VENDOR_REFS:
        assert rows[ref]["seed_rule"] == "per_goal_turn" and "seed" in rows[ref]["params_sent"], ref


def test_the_expanded_records_follow_the_split_and_the_paper_models_ids_do_not_move(bundle):
    recs = [r["payload"] for r in bundle.records["instrument"]]
    by_seed = {}
    for p in recs:
        by_seed.setdefault(p["extra"]["seed_instrument_id"], []).append(p)
    for ev in MT_EVALS:
        paper = by_seed[f"multiturn_{ev}_cap4096_generation_config"]
        assert len(paper) == 2 and all(p["decode_seed_rule"] == "none" for p in paper), ev
        shared = next(p for p in paper if len(p["extra"]["model_ids"]) > 1)
        qwen = next(p for p in paper if p["extra"]["model_ids"] == ["qwen35_4b_it"])
        assert sorted(shared["extra"]["model_ids"]) == SHARED_MODELS and shared["decode_source"] == "generation_config", ev
        assert shared["instrument_id"] == PAPER_MT_IDS[(ev, "shared")] and qwen["instrument_id"] == PAPER_MT_IDS[(ev, "qwen35_4b_it")], ev
        for p in paper:
            assert not (set(p["extra"]["model_ids"]) & set(VENDOR_ROUTE)), (ev, p["extra"]["model_ids"])
        vendor = by_seed[f"multiturn_{ev}_cap4096_vendor_preset"]
        assert sorted(p["extra"]["model_id"] for p in vendor) == sorted(VENDOR_ROUTE), ev
        for p in vendor:
            pol, _ = bundle.policies.rows[p["extra"]["decode_policy_id"]]
            assert p["decode_seed_rule"] == "per_goal_turn" and p["extra"]["model_ids"] == [p["extra"]["model_id"]], (ev, p["extra"]["model_id"])
            assert p["extra"]["decode_policy_ref"] == f"{p['extra']['model_id']}|multi_turn|off|vendor_preset", ev
            assert p["decode_source"] == pol["decode_source"] and p["extra"]["seed_decode_source"] == "vendor_preset_per_model", ev
            # the loader copies the row's SENT numbers onto the instrument: all six for a six-key vendor row
            assert {k: p[f"decode_{k}"] for k in DM.PARAMS} == bundle.policies.numbers(pol["policy_id"]), ev
            assert p["instrument_id"] not in PAPER_MT_IDS.values()
        ids = {p["instrument_id"] for p in paper} | {p["instrument_id"] for p in vendor}
        assert len(ids) == 5, ev                                                   # five distinct identities per attack
    # the non_template variants: the 35B's vendor row with the mode swapped and nothing else (per_goal_turn included)
    for ev in ("crescendo", "actorattack"):
        variant = by_seed[f"multiturn_{ev}_cap4096_generation_config_non_template"]
        base = next(p for p in by_seed[f"multiturn_{ev}_cap4096_vendor_preset"] if p["extra"]["model_id"] == "qwen35_35b_a3b")
        assert len(variant) == 1 and variant[0]["decode_seed_rule"] == "per_goal_turn" and variant[0]["extra"]["variant_of"] == base["extra"]["seed_instrument_id"]
        swapped = identity.instrument_identity_fields(base)
        swapped["applied_positions_mode"] = "non_template"
        assert identity.instrument_id(swapped) == variant[0]["instrument_id"] != base["instrument_id"], ev


def test_jobplan_resolves_the_vendor_route_for_the_three_models_and_the_paper_route_for_the_served_models(seed_index):
    for ev in MT_EVALS:
        for m in VENDOR_ROUTE:
            payload, extra = JP.canonical_instrument(seed_index, ev, m, "lora_native")
            assert extra["seed_instrument_id"] == f"multiturn_{ev}_cap4096_vendor_preset" and payload["decode_seed_rule"] == "per_goal_turn", (ev, m)
            assert extra["model_ids"] == [m] and extra["decode_policy_ref"] == f"{m}|multi_turn|off|vendor_preset", (ev, m)
        for m in PAPER_SERVED:
            payload, extra = JP.canonical_instrument(seed_index, ev, m, "lora_native")
            assert extra["seed_instrument_id"] == f"multiturn_{ev}_cap4096_generation_config" and payload["decode_seed_rule"] == "none", (ev, m)
            assert m in extra["model_ids"] and not (set(extra["model_ids"]) & set(VENDOR_ROUTE)), (ev, m)
            key = "qwen35_4b_it" if m == "qwen35_4b_it" else "shared"
            assert payload["instrument_id"] == PAPER_MT_IDS[(ev, key)], (ev, m)
    for ev in ("crescendo", "actorattack"):
        variant, extra = JP.canonical_instrument(seed_index, ev, "qwen35_35b_a3b", "lora_native", positions_mode="non_template")
        assert extra["seed_instrument_id"] == f"multiturn_{ev}_cap4096_generation_config_non_template" and variant["decode_seed_rule"] == "per_goal_turn"
        assert extra["variant_of"] == f"multiturn_{ev}_cap4096_vendor_preset"
