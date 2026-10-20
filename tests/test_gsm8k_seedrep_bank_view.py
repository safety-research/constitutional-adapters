"""The Qwen seed-replicate battery (2026-09-18) — `--bank-view <lane view>`: the roster-validated vocabulary.

The GSM8K launcher's ONE argv line (`g8k_common.sh g8k_driver_argv`) hands the driver `--bank-view "$VIEW"` — the matrix row's view,
the bank the row's objects come from (`<key>/cells_view/<view>` is the `--defenses-dir` beside it). The seed-replicate matrix is the
first whose rows sit on a lane view ITSELF (`s1` / `s2` / `s3`; every earlier matrix kept `view` at `main` | `it` and let the lane
view ride the row's `bank_view`), and the driver's parser admitted `main` | `it` alone — so those rows were refused at parse
time. `main` is not an answer: the GSM8K reader holds the rows' stamp to the matrix row's `view` and would refuse the disagreement;
omitting the flag drops the per-row identity. The fix is additive and consistent: `contracts.bank_views_admitted()` = `BANK_VIEWS`
+ every view the full22 roster declares, read at the driver's flag, the per-row identity block, gate I2 and the reader; `BANK_VIEWS`
itself (IFEval's two instrument views) is unchanged; no roster -> today's vocabulary, byte for byte.

An external object's bank is not a roster view — the lane's Arditi rows evaluate on
`s<s>` but their object comes from `cells_view/ar_c0p3` — so the vocabulary ALSO admits every `bank_view` a GSM8K matrix or lane
record declares on its rows (enumerated from `configs/sdfbase/gsm8k/*.json`), and the stamp a row expects is ONE rule
(`contracts.expected_bank_view_stamp`, used by the reader and by `g8k_seedrep_arditi.bank_view_stamp`): an instrument-view row
stamps its view (the panel convention every existing store follows), a lane-view row stamps its SOURCE bank when it declares one,
else the lane view itself. Known-bads: an Arditi row stamped `s1`, a seed row stamped `ar_c0p3` — both refused by the reader.
"""
from __future__ import annotations

import importlib.util
import io
import contextlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC

REPO = Path(__file__).resolve().parents[1]
ROSTER = REPO / "configs/sdfbase_full22_roster.yaml"
MATRIX = REPO / "configs/sdfbase/gsm8k/matrix_qwen_seedrep.json"
ARDITI_RECORD = REPO / "configs/sdfbase/gsm8k/arditi_rows_qwen_seedrep.json"
ALLPOS_MATRIX = REPO / "configs/sdfbase/gsm8k/matrix_arditi_allpos.json"
LLAMA_MATRIX = REPO / "configs/sdfbase/gsm8k/matrix_llama.json"
G8K = REPO / "scripts/sdfbase/gsm8k"
ARDITI_BANK = "ar_c0p3"
TREE = "c" * 40
LANE_VIEWS = ("s1", "s2", "s3")
UNKNOWN_VIEWS = ("s9", "bogus", "cells_view", "S1", "")

if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))


def _roster_union() -> set[str]:
    doc = yaml.safe_load(ROSTER.read_text())
    return {str(v) for t in doc["targets"].values() for v in (t.get("views") or {})}


ELIGIBLE = {"matrix*.json": ("g8k_matrix/1", "g8k_matrix/2"), "arditi_rows_*.json": ("g8k_seedrep_arditi_rows/1",),
            "paired_allmodels_rows.json": ("g8k_paired_allmodels_rows/1",)}     # the all-model appendix record (2026-09-22)


def _declared_banks() -> set[str]:
    """Every `bank_view` string on the rows of every ELIGIBLE GSM8K document (a matrix or a lane record — by NAME and by the
    writers' schemas), computed here from the documents so the contracts' enumeration is checked against an independent
    reading, never against itself. A reading that scanned every `*.json` with a `rows` list would mirror the contracts'
    own over-broad scan and mask it — this one is the eligibility the contracts must implement."""
    import fnmatch
    out: set[str] = set()
    for f in (REPO / "configs/sdfbase/gsm8k").glob("*.json"):
        schemas = next((v for k, v in ELIGIBLE.items() if fnmatch.fnmatchcase(f.name, k)), None)
        if schemas is None:
            continue
        doc = json.loads(f.read_text())
        assert doc.get("schema") in schemas, (f.name, doc.get("schema"))
        for r in doc["rows"]:
            if r.get("bank_view") is not None:
                assert isinstance(r["bank_view"], str) and r["bank_view"], (f.name, r.get("bank_view"))
                out.add(r["bank_view"])
    return out


def test_the_two_instrument_views_are_unchanged_and_the_roster_views_join_the_vocabulary():
    # BANK_VIEWS (and IFEval's merged copy) are exactly what they were: the fix widens the ADMITTED set, not the constant
    assert C.BANK_VIEWS == ("main", "it") == IC.BANK_VIEWS
    assert C.ifeval_tag("qwen35_4b", "main") == "qwen35_4b" and C.ifeval_tag("qwen35_4b", "it") == "qwen35_4b_it"
    # FLIPPED DELIBERATELY: this pin used to say the IFEval instrument keeps its two views and `ifeval_tag("qwen35_4b", "s1")`
    # RAISES. The batch-64 three-seed Qwen family now runs IFEval on the seed lane views s1..s3 / s1p..s3p through the same
    # instrument (native serving, the sibling document matrix_qwen_seedrep.json), so `ifeval_tag` admits contracts.IFEVAL_VIEWS —
    # a NEW constant beside BANK_VIEWS, which stays ("main", "it"). A SOURCE bank (ar_c0p3) is still not an IFEval view.
    assert C.ifeval_tag("qwen35_4b", "s1") == "qwen35_4b_s1" and C.ifeval_tag("qwen35_4b", "s3p") == "qwen35_4b_s3p"
    assert C.IFEVAL_VIEWS == (*C.BANK_VIEWS, *C.IFEVAL_LANE_VIEWS) and set(LANE_VIEWS) <= set(C.IFEVAL_LANE_VIEWS) <= set(C.bank_views_admitted())
    with pytest.raises(ValueError, match="is not one of"):
        C.ifeval_tag("qwen35_4b", ARDITI_BANK)
    admitted = C.bank_views_admitted()
    assert admitted[:2] == C.BANK_VIEWS
    assert set(admitted) == set(C.BANK_VIEWS) | _roster_union() | _declared_banks()   # enumerated from the documents, not typed here
    assert len(admitted) == len(set(admitted))                         # no duplicates: main / it / a declared `it` counted once
    assert set(LANE_VIEWS) <= set(admitted)
    assert C.roster_bank_views() == tuple(sorted(_roster_union()))
    # the SOURCE banks the GSM8K matrices / lane records declare: the seed-replicate Arditi rows' `ar_c0p3`
    # (also the allpos matrix's), the llama matrix's `ar_r0p05` — neither is a roster view, both name a bank objects come from
    assert C.declared_bank_views() == tuple(sorted(_declared_banks()))
    assert {ARDITI_BANK, "ar_r0p05"} <= set(C.declared_bank_views()) and ARDITI_BANK not in _roster_union()
    assert {ARDITI_BANK, "ar_r0p05"} <= set(admitted) and C.check_bank_view_arg(ARDITI_BANK) == ARDITI_BANK
    # the eligibility registry is the writers' own: the matrix schemas g8k_cells declares and the Arditi record's
    sys.path.insert(0, str(G8K))
    import g8k_cells as M
    import g8k_seedrep_arditi as AR
    assert dict(C.GSM8K_BANK_VIEW_DOCUMENTS) == ELIGIBLE
    assert C.bank_view_document_schemas("matrix_qwen_seedrep.json") == (M.MATRIX_SCHEMA, M.MATRIX_SCHEMA_2) == ("g8k_matrix/1", "g8k_matrix/2")
    assert C.bank_view_document_schemas("matrix.json") == (M.MATRIX_SCHEMA, M.MATRIX_SCHEMA_2)
    assert C.bank_view_document_schemas("arditi_rows_qwen_seedrep.json") == (AR.SCHEMA,) == ("g8k_seedrep_arditi_rows/1",)
    import g8k_paired_allmodels as PA
    assert C.bank_view_document_schemas("paired_allmodels_rows.json") == (PA.SCHEMA,) == ("g8k_paired_allmodels_rows/1",)
    # the appendix record's three external rows declare the staged source banks; the roster-served constitution row declares none
    assert {"paired_d1_gemma4e4b_s1", "paired_d1_llama8b_s1", "paired_d1_nemotron3nano4b_s1"} <= set(C.declared_bank_views()) <= set(admitted)
    for foreign in ("store_pins_default.json", "debug_trace.json", "matrix_x.yaml", "rows.json", "arditi_rows.json", "paired_allmodels_rows_v2.json"):
        assert C.bank_view_document_schemas(foreign) is None
    # every document in the repo's gsm8k dir is either eligible (and then read fail-closed above) or foreign by name
    names = sorted(p.name for p in (REPO / "configs/sdfbase/gsm8k").glob("*.json"))
    # foreign by name (DERIVED from the directory, never typed: main keeps adding store-pin files — store_pins_arditi_allpos.json — and
    # a67a624b's object_fit_batches.json, the object GENERATION stamp): every JSON no eligible glob names, and every one of those is a
    # store-pins file or the fit-batches stamp — none a bank-view document, none parsed
    import fnmatch
    foreign = [n for n in names if C.bank_view_document_schemas(n) is None]
    assert foreign == [n for n in names if not any(fnmatch.fnmatch(n, g) for g in ELIGIBLE)]
    assert foreign and all(n.startswith("store_pins_") or n == "object_fit_batches.json" for n in foreign), foreign
    assert {"store_pins_default.json", "store_pins_itdoc.json", "store_pins_llama.json", "object_fit_batches.json"} <= set(foreign)
    assert ROSTER.resolve() in {p.resolve() for p in C.full22_roster_candidates() if p.exists()}


def test_check_bank_view_arg_admits_main_it_and_the_lane_views_and_refuses_an_unknown_view_by_name():
    for view in ("main", "it", *LANE_VIEWS):
        assert C.check_bank_view_arg(view) == view
        assert C.bank_view_problem(view) is None
    for bad in (*UNKNOWN_VIEWS, None, 1):
        with pytest.raises(ValueError) as ei:
            C.check_bank_view_arg(bad)
        msg = str(ei.value)
        assert msg.startswith(f"{C.FLAG_BANK_VIEW} must be one of [") and msg.endswith(f", got {bad!r}")
        assert "'s1'" in msg and "'main'" in msg                       # the message names the whole admitted vocabulary
        assert "no full22 roster found" not in msg                     # the roster WAS found: a plain unknown view
        problem = C.bank_view_problem(bad)
        assert problem is not None and problem.startswith(f"{C.FLAG_BANK_VIEW} {bad!r} is not one of [")


def test_without_a_roster_the_vocabulary_is_todays_and_a_lane_view_is_refused_naming_the_paths_tried(tmp_path):
    absent = tmp_path / "no-such-roster.yaml"
    assert C.roster_bank_views(path=absent) is None
    assert C.bank_views_admitted(path=absent) == C.BANK_VIEWS                        # byte-for-byte today's vocabulary
    assert C.bank_view_problem("main", path=absent) is None and C.bank_view_problem("it", path=absent) is None
    why = C.bank_view_problem("s1", path=absent)
    assert why is not None and "no full22 roster found at" in why and str(absent) in why
    assert "a lane view is admitted only from the roster's views" in why
    assert C.full22_roster_candidates(path=absent) == (absent,)                       # an explicit path is tried ALONE


def test_the_vocabulary_is_the_roster_given_never_a_list_typed_here(tmp_path, monkeypatch):
    fx = tmp_path / "roster.yaml"
    fx.write_text(yaml.safe_dump({"targets": {"m": {"views": {"main": {}, "it": {}, "zz9": {}}}, "n": {"views": {"main": {}, "aa1": {}}}}}))
    assert C.roster_bank_views(path=fx) == ("aa1", "it", "main", "zz9")
    assert C.bank_views_admitted(path=fx) == ("main", "it", "aa1", "zz9")            # BANK_VIEWS first, then the roster's, sorted
    assert C.bank_view_problem("zz9", path=fx) is None
    assert C.bank_view_problem("s1", path=fx) is not None                               # not THIS roster's view -> refused
    # the environment override wins over the repo's roster (a staged tree elsewhere names its roster this way)
    monkeypatch.setenv(C.ENV_FULL22_ROSTER, str(fx))
    assert C.full22_roster_candidates()[0] == fx
    assert C.bank_views_admitted() == ("main", "it", "aa1", "zz9")
    assert C.check_bank_view_arg("zz9") == "zz9"
    with pytest.raises(ValueError, match="must be one of"):
        C.check_bank_view_arg("s1")
    monkeypatch.delenv(C.ENV_FULL22_ROSTER)
    assert "s1" in C.bank_views_admitted()                                              # back on the repo roster
    # an edited roster re-reads (the cache is keyed on the file's mtime, not its path alone)
    fx.write_text(yaml.safe_dump({"targets": {"m": {"views": {"main": {}, "it": {}, "yy8": {}}}}}))
    os.utime(fx, ns=(fx.stat().st_atime_ns + 10 ** 9, fx.stat().st_mtime_ns + 10 ** 9))
    assert C.bank_views_admitted(path=fx) == ("main", "it", "yy8")
    # a document that is not a full22 roster raises rather than reading as "no lane views"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"wave": "x"}))
    with pytest.raises(ValueError, match="not a full22 roster"):
        C.bank_views_admitted(path=bad)
    bad.write_text(yaml.safe_dump({"targets": {"m": {"cells": {}}}}))
    with pytest.raises(ValueError, match="carries no `views:` mapping"):
        C.bank_views_admitted(path=bad)
    # the SOURCE banks come from the ELIGIBLE GSM8K documents beside the roster (`<roster dir>/sdfbase/gsm8k/`: `matrix*.json`
    # under the matrix schemas, `arditi_rows_*.json` under the record's): none there -> none declared; a real-schema matrix
    # declaring `zz_bank` on a row -> admitted; a lane record declaring `yy_bank` -> admitted
    d2 = tmp_path / "d2"; g = d2 / "sdfbase" / "gsm8k"; g.mkdir(parents=True)
    fx2 = d2 / "roster.yaml"; fx2.write_text(yaml.safe_dump({"targets": {"m": {"views": {"main": {}, "it": {}}}}}))
    assert C.declared_bank_views(path=fx2) == () and C.bank_views_admitted(path=fx2) == ("main", "it")
    (g / "matrix_x.json").write_text(json.dumps({"schema": "g8k_matrix/2", "rows": [{"view": "main", "bank_view": "zz_bank"}, {"view": "main", "bank_view": None}]}))
    (g / "arditi_rows_x.json").write_text(json.dumps({"schema": "g8k_seedrep_arditi_rows/1", "rows": [{"view": "s1", "bank_view": "yy_bank"}]}))
    assert C.declared_bank_views(path=fx2) == ("yy_bank", "zz_bank")
    assert C.bank_views_admitted(path=fx2) == ("main", "it", "yy_bank", "zz_bank") and C.bank_view_problem("zz_bank", path=fx2) is None
    # --- a FOREIGN document beside the matrices can neither widen admission nor break it, whatever it holds
    (g / "store_pins_x.json").write_text(json.dumps({"_meta": {}, "k": {"bank_view": "not_a_row"}}))          # no rows: foreign by name
    (g / "debug_trace.json").write_text(json.dumps({"schema": "debug_trace/1", "rows": [{"bank_view": "ar_zz9"}]}))   # a stray `rows` list
    (g / "notes.json").write_text("{not json at all")                                                          # malformed, foreign
    (g / "matrix_x.yaml").write_text("rows: [{bank_view: ar_zz8}]")                                            # not even .json
    assert C.declared_bank_views(path=fx2) == ("yy_bank", "zz_bank")
    for stray in ("not_a_row", "ar_zz9", "ar_zz8"):
        assert C.bank_view_problem(stray, path=fx2) is not None
    assert C.bank_view_problem(ARDITI_BANK, path=fx2) is not None                                             # not declared HERE
    # --- and an ELIGIBLE-named document is read fail-closed: malformed / foreign-schema / bad rows REFUSE by name, never drop or guess
    def refuses(name: str, text: str, match: str) -> None:
        f = g / name; f.write_text(text)
        try:
            with pytest.raises(ValueError, match=match) as ei:
                C.declared_bank_views(path=fx2)
            assert name in str(ei.value)
            with pytest.raises(ValueError, match=match):
                C.bank_views_admitted(path=fx2)                                                                 # admission itself refuses
        finally:
            f.unlink()
    refuses("matrix_broken.json", "{not json", "does not parse as JSON")
    refuses("matrix_foreign.json", json.dumps({"schema": "debug_trace/1", "rows": [{"bank_view": "ar_zz9"}]}), "schema 'debug_trace/1' is not one admitted")
    refuses("matrix_noschema.json", json.dumps({"rows": [{"bank_view": "ar_zz9"}]}), "schema None is not one admitted")
    refuses("arditi_rows_wrong.json", json.dumps({"schema": "g8k_matrix/2", "rows": []}), "schema 'g8k_matrix/2' is not one admitted")
    refuses("matrix_norows.json", json.dumps({"schema": "g8k_matrix/2"}), "`rows` is not a list")
    refuses("matrix_badrow.json", json.dumps({"schema": "g8k_matrix/2", "rows": [7]}), r"rows\[0\] is not a mapping")
    refuses("matrix_badbank.json", json.dumps({"schema": "g8k_matrix/2", "rows": [{"bank_view": 7}]}), r"bank_view must be null or a non-empty string, got 7")
    refuses("matrix_emptybank.json", json.dumps({"schema": "g8k_matrix/2", "rows": [{"bank_view": ""}]}), "got ''")
    refuses("matrix_list.json", json.dumps([{"bank_view": "ar_zz9"}]), "not a JSON mapping")
    assert C.declared_bank_views(path=fx2) == ("yy_bank", "zz_bank")                                          # back to the good state
    fx3 = tmp_path / "d3" / "roster.yaml"; fx3.parent.mkdir()
    fx3.write_text(yaml.safe_dump({"targets": {"m": {"views": {"main": {}}}}}))
    assert C.declared_bank_views(path=fx3) is None                                     # no gsm8k dir beside THIS roster


def test_expected_bank_view_stamp_is_one_rule_for_the_launcher_and_the_reader():
    """An instrument-view row stamps its view (the panel convention: every existing store; the source bank rides `bank_view`,
    unstamped); a lane-view row stamps its SOURCE bank when it declares one, else the lane view itself."""
    assert C.expected_bank_view_stamp(view="main", bank_view=None) == "main"
    assert C.expected_bank_view_stamp(view="main", bank_view="ar_r0p05") == "main"        # llama rows: byte identity
    assert C.expected_bank_view_stamp(view="it", bank_view=ARDITI_BANK) == "it"           # the allpos matrix's rows: byte identity
    assert C.expected_bank_view_stamp(view="s1", bank_view=None) == "s1"                  # a seed-replicate matrix row
    assert C.expected_bank_view_stamp(view="s1", bank_view="") == "s1"
    assert C.expected_bank_view_stamp(view="s2", bank_view=ARDITI_BANK) == ARDITI_BANK    # the seed-replicate Arditi rows
    sys.path.insert(0, str(G8K))
    import g8k_seedrep_arditi as AR
    record = AR.load_record()
    assert [AR.bank_view_stamp(r) for r in record["rows"]] == [ARDITI_BANK] * 3
    assert [(r["view"], r["bank_view"]) for r in record["rows"]] == [("s1", ARDITI_BANK), ("s2", ARDITI_BANK), ("s3", ARDITI_BANK)]
    doc = json.loads(MATRIX.read_text())
    # + the paired views s<s>p: their own bank views, stamped as their own view
    assert {AR.bank_view_stamp(r) for r in doc["rows"]} == {"s1", "s2", "s3", "s1p", "s2p", "s3p"} == {r["view"] for r in doc["rows"]}


def test_row_identity_block_gate_i2_row_provenance_and_driver_argv_admit_a_lane_view_and_refuse_an_unknown_one():
    block = C.row_identity_block(cell="none", view="s1", object_artifact_sha256=None, producer_tree_sha=TREE,
                                 sampling_mode=C.SAMPLING_THINKING_CARD)
    assert block[C.ROW_IDENTITY_KEY_VIEW] == "s1" and block[C.ROW_IDENTITY_KEY_ROUTE_KIND] == C.ROUTE_KIND_BASE
    row = {**block, "thinking": C.THINKING_ON}
    assert C.row_identity_problem(row, expected={C.ROW_IDENTITY_KEY_VIEW: "s1"}) is None
    assert C.row_identity_problem({**row, C.ROW_IDENTITY_KEY_VIEW: "s9"}, expected={}) == "row route_kind / view outside the vocabulary"
    assert C.row_identity_problem({**row, C.ROW_IDENTITY_KEY_VIEW: "main"}, expected={C.ROW_IDENTITY_KEY_VIEW: "s1"}) == "row view='main' is not the cell's 's1'"
    for bad in ("s9", "bogus"):
        with pytest.raises(C.RowIdentityError, match=f"{C.FLAG_BANK_VIEW} '{bad}' is not one of"):
            C.row_identity_block(cell="none", view=bad, object_artifact_sha256=None, producer_tree_sha=TREE,
                                 sampling_mode=C.SAMPLING_THINKING_CARD)
    lens = C.row_identity_block(cell="constv2-mix-full__psr_ce_all", view="s2", object_artifact_sha256="ab" * 32,
                                producer_tree_sha=TREE, sampling_mode=C.SAMPLING_THINKING_CARD)
    assert (lens[C.ROW_IDENTITY_KEY_VIEW], lens[C.ROW_IDENTITY_KEY_ROUTE_KIND]) == ("s2", C.ROUTE_KIND_LENS)
    prov = C.row_provenance(cell="none", view="s3", object_artifact_sha256=None, producer_tree_sha=TREE)
    assert prov["view"] == "s3" and prov["route_kind"] == C.ROUTE_KIND_BASE
    with pytest.raises(ValueError, match="must be one of"):
        C.row_provenance(cell="none", view="s9", object_artifact_sha256=None, producer_tree_sha=TREE)
    kw = dict(python="p", script="s", proxy_url="u", routing_table="r", cell="none", config="c", defenses_dir="d", out_root="o",
              controls=[C.CONTROL_GSM8K], thinking="on", concurrency=31, judge_model="j", samples=1)
    plain = C.driver_argv(**kw)
    assert C.driver_argv(**kw, bank_view="s1") == plain + [C.FLAG_BANK_VIEW, "s1"]
    assert C.driver_argv(**kw, bank_view="main") == plain + [C.FLAG_BANK_VIEW, "main"]      # unchanged
    with pytest.raises(ValueError, match=f"driver_argv: {C.FLAG_BANK_VIEW} 's9' is not one of"):
        C.driver_argv(**kw, bank_view="s9")


def _driver_lines(tmp_path: Path, rows: str) -> list[list[str]]:
    """The dry run's DRIVER: argv lines for the given matrix rows (the launcher's ONE argv line, `g8k_driver_argv`), split
    into argv and stripped of the python + script head — exactly what `served_capability_run.py` would parse."""
    r = subprocess.run([sys.executable, str(G8K / "g8k_dry_run.py"), "--stage", "pre", "--matrix", "qwen_seedrep", "--rows", rows,
                        "--tmp", str(tmp_path / "t"), "--python", sys.executable],
                       capture_output=True, text=True, check=False, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    out = []
    for line in r.stdout.splitlines():
        if line.startswith("DRIVER:"):
            argv = shlex.split(line[len("DRIVER:"):].strip())
            assert argv[1] == "scripts/served_capability_run.py", argv[:2]
            out.append(argv[2:])
    return out


def _parse(argv: list[str]):
    import served_capability_run as D
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            return D.build_parser().parse_args(argv), None
    except SystemExit as e:
        return None, (e.code, err.getvalue())


def _arditi_child_argv(seed_none_argv: list[str], row: dict) -> list[str]:
    """An Arditi GSM8K child's argv, composed from the lane record's row the way the launcher composes it: the seed
    view's `none` argv with the cell, the config, the defenses dir (the record's `bank_view_rel.primary` =
    `…/cells_view/ar_c0p3`), the out root, the SOURCE-bank stamp (`g8k_seedrep_arditi.bank_view_stamp`) and the fitted
    contract (tuple -> override; the pin)."""
    import g8k_seedrep_arditi as AR
    argv = list(seed_none_argv)
    def put(flag, value):
        argv[argv.index(flag) + 1] = value
    put("--cell", row["cell"])
    put("--config", "configs/sdfbase/gsm8k/sdfbase_full22_qwen35_4b_it.gsm8k_arditi_allpos.yaml")     # the launcher's choice; a path to argparse
    shared = argv[argv.index("--defenses-dir") + 1].split("/exp/objects/")[0]
    put("--defenses-dir", f"{shared}/{row['bank_view_rel']['primary']}")
    put("--out-root", f"{shared}/exp/gsm8k_paired200_paper_objects_qwen_seedrep/{row['out_root']}")
    put(C.FLAG_BANK_VIEW, AR.bank_view_stamp(row))
    assert row["tuple_name"] and row["fitted"]
    return argv + ["--allow-out-of-support-thinking", "--expect-object-sha256", row["object_sha256"]]


def test_the_drivers_parser_admits_the_24_plus_3_dry_run_argvs_flag_for_flag_and_refuses_an_unknown_view_at_parse_time(tmp_path):
    """The cut-blocking case verbatim: the launcher generates every GSM8K child's argv through the one argv line and the
    driver refused `--bank-view s1` (`choices` main | it). All 24 matrix rows' DRIVER lines parse (the stamp is the row's
    view, the bank the objects come from), and the 3 Arditi children composed from the lane record parse with the
    SOURCE-bank stamp `ar_c0p3`."""
    sys.path.insert(0, str(G8K))
    import g8k_seedrep_arditi as AR
    # the 36-row matrix (2026-09-20): every LAUNCH row — the 24 seed-view rows + the paired views' none + composed rows (the held
    # companion / LoRA-difference rows 20-21 / 24-25 / 28-29 print no driver line)
    argvs = _driver_lines(tmp_path, "0-19,22-23,26-27,30-35")
    assert len(argvs) == 30
    by_seed_none: dict[int, list[str]] = {}
    for argv in argvs:
        ns, refusal = _parse(argv)
        assert refusal is None, refusal
        # the row's VIEW, as emitted: the seed view `s<s>` or (2026-09-20) its paired view `s<s>p`, both under seed base s
        assert ns.inspect_seed_policy == "replicate" and ns.bank_view in (f"s{ns.inspect_seed_base}", f"s{ns.inspect_seed_base}p")
        # the named bank view IS the bank the objects come from: the provenance claim agrees with `--defenses-dir`
        assert ns.defenses_dir.endswith(f"/qwen35_4b/cells_view/{ns.bank_view}"), ns.defenses_dir
        assert ns.config.startswith(f"configs/sdfbase/gsm8k/sdfbase_full22_qwen35_4b_{ns.bank_view}.gsm8k_")
        if ns.cell == "none" and not ns.bank_view.endswith("p"):
            by_seed_none[int(ns.inspect_seed_base)] = argv
    assert sorted(by_seed_none) == [1, 2, 3]
    paired = [(_parse(a)[0].bank_view, _parse(a)[0].cell) for a in argvs if _parse(a)[0].bank_view.endswith("p")]
    assert paired == [(f"s{s}p", c) for s in (1, 2, 3) for c in ("none", "constv2-mix-full__psr_ce_all_minus_ctl")]   # the 6 paired launch rows
    # rows 0 (none), 1 (a fitted PSR object) of s1; the s1 prompted route now sits after the paired rows
    # (argv index 24 = matrix row 30, since the paired views took rows 18-29 and their held rows print no line)
    for i, cell in ((0, "none"), (1, "constv2-mix-full__psr_ce_all"), (24, "prompted__safety_refusal")):
        ns, _ = _parse(argvs[i]); assert (ns.cell, ns.bank_view) == (cell, "s1")
    # the 3 Arditi children: evaluation view s<s> (`--inspect-seed-base`, config, out root), SOURCE bank ar_c0p3 (`--bank-view`,
    # `--defenses-dir`); the parser admits them and the stamp is the record's rule
    for row in AR.load_record()["rows"]:
        s = int(row["inspect_seed_base"])
        argv = _arditi_child_argv(by_seed_none[s], row)
        ns, refusal = _parse(argv)
        assert refusal is None, refusal
        assert ns.bank_view == ARDITI_BANK == AR.bank_view_stamp(row) and ns.inspect_seed_base == s
        assert ns.defenses_dir.endswith(f"/cells_view/{ARDITI_BANK}") and ns.expect_object_sha256 == row["object_sha256"]
        assert f"/qwen35_4b_s{s}/arditi_allpos/" in ns.out_root and ns.cell == "arditi_allpos"
        # an Arditi child stamped with its EVALUATION view parses (s1 is a view) — the READER refuses it against the record
        # (test below); a bank nobody declares is refused here, at parse time
        i = argv.index(C.FLAG_BANK_VIEW) + 1
        ns2, refusal2 = _parse(argv[:i] + ["ar_zz9"] + argv[i + 1:])
        assert ns2 is None and refusal2[0] == 2 and "got 'ar_zz9'" in refusal2[1]
    none_argv = argvs[0]
    i = none_argv.index(C.FLAG_BANK_VIEW) + 1
    # known-bad: an unknown view is refused AT PARSE TIME, by name, before any standup (as the former `choices` did)
    for bad in ("s9", "bogus"):
        ns, refusal = _parse(none_argv[:i] + [bad] + none_argv[i + 1:])
        assert ns is None and refusal[0] == 2 and f"{C.FLAG_BANK_VIEW} must be one of [" in refusal[1] and f"got '{bad}'" in refusal[1]
    # the two instrument views are still accepted (unchanged behaviour)
    for old in C.BANK_VIEWS:
        ns, refusal = _parse(none_argv[:i] + [old] + none_argv[i + 1:])
        assert refusal is None and ns.bank_view == old
    # and the flag absent still parses to None (a run without it emits byte-identical rows)
    ns, refusal = _parse(none_argv[:i - 1] + none_argv[i + 1:])
    assert refusal is None and ns.bank_view is None


def test_the_gsm8k_reader_holds_the_stamp_to_the_matrix_rows_view_and_admits_a_lane_view():
    """Why `--bank-view main` is NOT the answer for an s-view row: the reader holds the rows' `view` to the matrix row's
    (`s1`) and refuses the disagreement; a lane view is admitted through the same vocabulary the driver stamps with."""
    import tinycap_gsm8k_paired200 as G
    doc = json.loads(MATRIX.read_text())
    s1_none = next(r for r in doc["rows"] if r["job_id"] == "g8k-qwen35_4b-s1-none")
    s1_psr = next(r for r in doc["rows"] if r["job_id"] == "g8k-qwen35_4b-s1-mix-psr")
    assert (s1_none["view"], s1_none["route_kind"], s1_none["fitted"]) == ("s1", "base", False)
    manifest = {C.MANIFEST_KEY_PRODUCER: {"tree_sha": TREE}}

    def rows(view: str, *, sha=None, kind="base", n=3):
        return [{"item_id": str(k), "_extra": {"object_artifact_sha256": sha, "route_kind": kind, "view": view, "producer_tree_sha": TREE}}
                for k in range(n)]

    rec, problem = G.row_provenance_record(rows("s1"), row=s1_none, manifest=manifest)
    assert problem is None and rec == {"present": True, "object_artifact_sha256": None, "route_kind": "base", "view": "s1", "producer_tree_sha": TREE}
    rec, problem = G.row_provenance_record(rows("s1", sha=s1_psr["object_sha256"], kind="lens"), row=s1_psr, manifest=manifest)
    assert problem is None and rec["view"] == "s1" and rec["object_artifact_sha256"] == s1_psr["object_sha256"]
    # known-bad A — the `main` alternative: the stamp disagrees with the matrix row, refused by name
    rec, problem = G.row_provenance_record(rows("main"), row=s1_none, manifest=manifest)
    assert rec["present"] is False and problem is not None and "view 'main' != 's1'" in problem
    # known-bad B — a view outside the vocabulary (a matrix row that names it too, so the equality passes): refused as outside
    rec, problem = G.row_provenance_record(rows("s9"), row={**s1_none, "view": "s9"}, manifest=manifest)
    assert rec["present"] is False and problem is not None and "outside" in problem and "'s9'" in problem
    # a run without the flag: no provenance on any row -> present False, legal (unchanged)
    bare = [{"item_id": str(k), "_extra": {}} for k in range(3)]
    rec, problem = G.row_provenance_record(bare, row=s1_none, manifest=manifest)
    assert problem is None and rec["present"] is False
    # --- the Arditi rows evaluate on s<s> but their object comes from cells_view/ar_c0p3 — the stamp is the
    # SOURCE bank (`row["bank_view"]`), held separately from the evaluation view (`row["view"]`, the leg's TAG / out root / seed base)
    ard = json.loads(ARDITI_RECORD.read_text())["rows"]
    for r in ard:
        assert (r["view"], r["bank_view"], r["fitted"], r["route_kind"]) == (f"s{r['inspect_seed_base']}", ARDITI_BANK, True, "lens")
        rec, problem = G.row_provenance_record(rows(ARDITI_BANK, sha=r["object_sha256"], kind="lens"), row=r, manifest=manifest)
        assert problem is None and rec == {"present": True, "object_artifact_sha256": r["object_sha256"], "route_kind": "lens",
                                           "view": ARDITI_BANK, "producer_tree_sha": TREE}
        # known-bad C — an Arditi row stamped with its EVALUATION view: refused by name
        rec, problem = G.row_provenance_record(rows(r["view"], sha=r["object_sha256"], kind="lens"), row=r, manifest=manifest)
        assert rec["present"] is False and problem is not None and f"view '{r['view']}' != '{ARDITI_BANK}'" in problem
    # known-bad D — a seed row stamped with the Arditi SOURCE bank: refused by name
    rec, problem = G.row_provenance_record(rows(ARDITI_BANK), row=s1_none, manifest=manifest)
    assert rec["present"] is False and problem is not None and f"view '{ARDITI_BANK}' != 's1'" in problem
    # byte identity for the campaign matrices (instrument-view rows carrying a source bank): the panel convention stands —
    # the allpos matrix's row (view it, bank_view ar_c0p3) is held to `it`, the llama row (view main, bank_view it) to `main`
    allpos = json.loads(ALLPOS_MATRIX.read_text())["rows"][0]
    assert (allpos["view"], allpos["bank_view"]) == ("it", ARDITI_BANK)
    rec, problem = G.row_provenance_record(rows("it", sha=allpos["object_sha256"], kind="lens"), row=allpos, manifest=manifest)
    assert problem is None and rec["view"] == "it"
    rec, problem = G.row_provenance_record(rows(ARDITI_BANK, sha=allpos["object_sha256"], kind="lens"), row=allpos, manifest=manifest)
    assert rec["present"] is False and f"view '{ARDITI_BANK}' != 'it'" in problem              # unchanged from before this fix
    llama = next(r for r in json.loads(LLAMA_MATRIX.read_text())["rows"] if r.get("bank_view") and r["view"] == "main")
    rec, problem = G.row_provenance_record(rows("main", sha=llama["object_sha256"], kind=llama["route_kind"]), row=llama, manifest=manifest)
    assert problem is None and rec["view"] == "main"


def test_the_launchers_one_argv_line_hands_the_driver_the_rows_view_and_the_reader_the_matrix_rows_view():
    """The contract the launcher rides, pinned at its two ends: `g8k_driver_argv` emits `--bank-view "$VIEW"` (the row's view,
    set from `row['view']` by `g8k_cells.row_sh`), and the reader's vocabulary check reads `contracts.bank_view_problem` (never
    the bare `BANK_VIEWS` membership that would refuse the lane views)."""
    sh = (G8K / "g8k_common.sh").read_text()
    assert 'DRV+=(--attempt-id "$ATTEMPT_ID" --bank-view "$VIEW")' in sh
    cells = (G8K / "g8k_cells.py").read_text()
    assert "f\"VIEW={shlex.quote(row['view'])}\"" in cells
    reader = (REPO / "scripts/tinycap_gsm8k_paired200.py").read_text()
    assert 'C.bank_view_problem(got["view"]) is not None' in reader and 'got["view"] not in C.BANK_VIEWS' not in reader
    assert 'C.expected_bank_view_stamp(view=row["view"], bank_view=row.get("bank_view"))' in reader and '"view": str(row["view"])' not in reader
    assert 'return C.expected_bank_view_stamp(view=row["view"], bank_view=row.get("bank_view"))' in (G8K / "g8k_seedrep_arditi.py").read_text()
    driver = (REPO / "scripts/served_capability_run.py").read_text()
    assert "type=_bank_view_arg" in driver and "choices=list(tc.BANK_VIEWS)" not in driver
