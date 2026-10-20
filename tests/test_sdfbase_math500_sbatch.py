"""math500-thinking-on-full22: the matrix, the overlays, the sbatch
shape (the cluster's hard rules), the generated argv (through the sbatch's OWN dry-run path, never re-typed), the
overlay -> route -> _pick_cell chain through the ACTUAL generated standup argv, the ledger sequencing, the
two-stage dry run with candidate vs certified render facts, and the stale-table hazard. Every count is COMPUTED
from ``cells.M5T_CELLS`` (``cells.counts``), never typed."""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "math500", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cells as M  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer import site as SITE  # noqa: E402
import cluster_site as _cs  # noqa: E402

# Every printed plan line takes its cluster's QoS and log directory from the site config: one synthetic two-cluster site for the
# module (site labels of its own), and the values the printers must read from it.
import tempfile  # noqa: E402
_SITE_ROOT = Path(tempfile.mkdtemp(prefix="m5t-site-"))
_SITE_CFG = _cs.write_site(_SITE_ROOT)
SITE_A, SITE_B = _cs.expect(_SITE_ROOT, "a"), _cs.expect(_SITE_ROOT, "b")


@pytest.fixture(autouse=True)
def _cluster_site(monkeypatch):
    monkeypatch.setenv("SISTEER_CONFIG", str(_SITE_CFG))

M5T = REPO / "scripts/sdfbase/math500"
F22 = REPO / "scripts/sdfbase/full22"
#: the stdlib-only store-following pin rule (scripts/sdfbase/full22/steer_store_pin.py) — THE module the launcher reads,
#: taken off cells.py itself rather than re-imported, so a second copy can never be what a test asserts against
SP_MOD = M.SP
GPU_SBATCHES = ["math500_thinking.sbatch", "smoke_thinking.sbatch", "nemo_backend_probe.sbatch"]
SBATCHES = GPU_SBATCHES + ["dry_run_prod.sbatch"]     # + the CPU-only G-DRY-prod launcher (same prologue, no GPU)
BASE_SHA = "0ee16f82b58244990837f358c16fd0b5ac0d6c22"          # the merged tip the lanes were cut from
COUNTS = M.counts()
#: the PUBLISHED tree (every MATH-500 store's producer, 24 legs COMPLETE) and the 24-row `cells.py list --json` of it
PUBLISHED_TREE = "55e3cbdd0bfb5d53f8d62c50a099a3c1c105100b"
#: the published pull on the local workstation (outside the repo; the real-store checks skip when it is absent)
GRADE_PUBLISHED = SITE.expand_path("$GRADE/tinycap/math500/pull_55e3cbd/controls")
FIXTURE = REPO / "tests/fixtures/math500_matrix_55e3cbdd.json"
#: the constitution-SI prompt maxima MEASURED offline on the pinned tokenizers (500 MATH-500 renders each, ON render,
#: scripts/tinycap_oos_prompt_budget.py --task math500 --system-prompt-cell prompted__si_constitution; decisions-impl-laneA
#: A10) — the numbers behind the long-SI entry's admission: every one + the 16,384 cap fits L 24,064
CONSTITUTION_PROMPT_MAX = {"gemma4e4b": 6981, "qwen35_4b": 6900, "nemotron3nano4b": 6887}
CONSTITUTION_SI_SHA256 = "21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f"


#: the campaign originals may differ from BASE_SHA's in EXACTLY one ruled change -- the multi-turn
#: target cap (attacks.target_max_new_tokens 512 -> the roster's 4,096, the canonical instrument of
#: 2026-09-11) and its comment lines; any other drift still refuses. The overlays' own byte-identity
#: to the generator (--check above) is what keeps them derived from the CURRENT originals.
ALLOWED_ORIGINAL_DRIFT = {("attacks", "target_max_new_tokens"): (512, 4096)}
#: keys ADDED by the ruled changes (absent at BASE_SHA, present now): the canonical best-of-N's climb
#: cap + its sampler rule (`climb_decode: as_drawn`, since 2026-09-17 -- the climb re-draws at the
#: sampler its 512 store records; it was the wave-wide `climb_temperature: 1.0` from 2026-09-11 until then),
#: beside the UNCHANGED base cap bon.max_new_tokens 512
ALLOWED_ORIGINAL_ADDITIONS = {("bon", "climb_cap"): 4096, ("bon", "climb_decode"): "as_drawn",
                              # the over-refusal controls' own cap key (2026-09-11): the ladder root, so the
                              # ruled target cap above moves without moving the controls
                              ("attacks", "controls_max_new_tokens"): 512}


#: list items REMOVED by a ruled change (present at BASE_SHA, absent now): the obsolete eval harnesses
#: were dropped on 2026-09-12, so every campaign config's
#: `controls` lost wmdp and alpaca_eval
ALLOWED_ORIGINAL_REMOVALS = {("controls",): ("wmdp", "alpaca_eval")}
#: list items REMOVED from the campaign MANIFESTS by a ruled change (present at BASE_SHA, absent now): commit 004e74cb4
#: ("Petri: drop the abliterated-Qwen auditor leg, so no GPU is stood up for it") -- across every audit on disk, 231 cells
#: and 21,399 rows carrying an auditor_chain record, the third leg was reached ZERO times, and keeping it cost a GPU on
#: every run -- removed the leg from every campaign manifest's `petri.auditor_fallback_models`.
ALLOWED_MANIFEST_REMOVALS = {("petri", "auditor_fallback_models"): ("openai/huihui-ai/Huihui-Qwen3.6-27B-abliterated",)}


def _admit_roster_declared_prompted_cells(rel: str, base: dict, now: dict) -> None:
    """Since 2026-09-12: a prompted baseline the roster declares on a view rides into that view's campaign
    config (`explicit_cells`) and manifest (`cells`, the harness id) —
    `prompted__si_constitution` / `prompted__constitution` on gemma / qwen / nemotron `main`. Strip from
    ``now`` every prompted cell absent at BASE_SHA so the remaining comparison is the ruled one; a
    non-prompted addition stays and fails by name. These overlays never serve such a cell (the test below)."""
    if rel.startswith("configs/sdfbase_full22_"):
        holder, field = now, "explicit_cells"
        base_cells = list(base.get("explicit_cells") or [])
    elif rel.startswith("configs/capeval/manifests/petri_full22_"):
        holder, field = now["petri"], "cells"
        base_cells = list((base.get("petri") or {}).get("cells") or [])
    else:
        return
    cells = list(holder.get(field) or [])
    added = [c for c in cells if c not in base_cells]
    assert all(c.startswith("prompted__") for c in added), (rel, "a non-prompted cell was added since BASE_SHA", added)
    holder[field] = [c for c in cells if c not in added]


#: the tree the public release rewrote: a campaign original may differ from its bytes there only by the release's declared
#: text rules, which configs/SCRUB_MAP.tsv certifies per file (a change that leaves the parsed document equal needs none)
PRE_RELEASE_SHA = "6801151ed3ba5a62e9fb2ad5d6d01adc8ef82eac"


def _pre_release_text(rel: str, now_text: str) -> str:
    """The campaign original as the public release found it, once its rewrite since is certified: equal bytes, an equal
    parsed document, or a configs/SCRUB_MAP.tsv row mapping the pre-release sha256 to today's."""
    pre = subprocess.run(["git", "show", f"{PRE_RELEASE_SHA}:{rel}"], cwd=REPO, capture_output=True, text=True, check=False)
    if pre.returncode != 0:
        pytest.skip("the pre-release sha is not in this clone")
    if pre.stdout == now_text or yaml.safe_load(pre.stdout) == yaml.safe_load(now_text):
        return pre.stdout
    import hashlib
    from sisteer import scrub_map
    got = scrub_map.check(hashlib.sha256(pre.stdout.encode()).hexdigest(), hashlib.sha256(now_text.encode()).hexdigest(),
                          path=rel, map_path=REPO / "configs" / "SCRUB_MAP.tsv")
    assert got == scrub_map.VIA_MAP, f"{rel} changed since the pre-release tree and no scrub-map row certifies the rewrite"
    return pre.stdout


def _assert_original_unchanged_but_for_the_ruled_cap(rel: str, base_text: str, now_text: str) -> None:
    if base_text == now_text:
        return
    base, now = yaml.safe_load(base_text), yaml.safe_load(now_text)
    _admit_roster_declared_prompted_cells(rel, base, now)
    if not rel.startswith("configs/sdfbase_full22_"):
        for (sect, key), gone in ALLOWED_MANIFEST_REMOVALS.items():
            for item in gone:
                assert item in base[sect][key] and item not in now[sect][key], (rel, sect, key, item, "004e74cb4 dropped it from every campaign manifest")
                base[sect][key].remove(item)
        assert base == now, (f"{rel} drifted from BASE_SHA (no drift is allowed on a manifest beyond a roster-declared prompted cell "
                             "and the ruled auditor-leg drop of 004e74cb4)")
        return
    for (sect, key), (was, is_now) in ALLOWED_ORIGINAL_DRIFT.items():
        assert base[sect].pop(key) == was and now[sect].pop(key) == is_now, (rel, sect, key)
    for (sect, key), is_now in ALLOWED_ORIGINAL_ADDITIONS.items():
        assert key not in base.get(sect, {}), (rel, sect, key, "was present at BASE_SHA")
        assert now[sect].pop(key) == is_now, (rel, sect, key)
    for (sect,), gone in ALLOWED_ORIGINAL_REMOVALS.items():
        for item in gone:
            assert item in base[sect] and item not in now[sect], (rel, sect, item, "PR #46 dropped it from every campaign config")
            base[sect].remove(item)
    assert base["bon"]["max_new_tokens"] == now["bon"]["max_new_tokens"] == 512   # the climb's BASE cap
    assert base == now, (f"{rel} drifted from BASE_SHA beyond the ruled attacks.target_max_new_tokens "
                         "change, the additive bon.climb_* / attacks.controls_max_new_tokens keys and PR #46's controls drop")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# the matrix                                                                   #
# --------------------------------------------------------------------------- #


def _cluster_preference_masked(rows):
    """`rows` with each `cluster_preference` replaced by one placeholder: the site labels were renamed after the published tree."""
    return [dict(r, cluster_preference="<cluster>") if "cluster_preference" in r else r for r in rows]


def _sampling_source_masked(rows):
    """`rows` with each sampling row's `source` replaced by one placeholder: that provenance prose was reworded after the
    published tree, while every number of the sampling row is compared."""
    return [dict(r, sampling=dict(r["sampling"], source="<source>")) if "source" in (r.get("sampling") or {}) else r for r in rows]


def test_base_rows_regenerate_byte_identically_on_the_published_keys_and_gain_exactly_the_extension_keys():
    """math-eight-arms-extension P2 / A5: rows 0-23 == the published tree's `cells.py list --json` on EVERY published
    key (index, job_id, out_root, pin, tuple, time, ...) and carry EXACTLY the extension keys beyond them; the CLI's
    `list --json --published-keys` prints the fixture itself; the fixture regenerates from the published tree's own
    cells.py when that tree is in the clone."""
    fixture = json.loads(FIXTURE.read_text())
    assert len(fixture) == 24 == M.N_BASE == len(M.BASE_ROWS) and set(fixture[0]) == M.PUBLISHED_KEYS
    for row, want in zip(M.BASE_ROWS, fixture):
        assert {k: row[k] for k in want} == want, row["job_id"]
        assert set(row) - set(want) == M.EXTENSION_KEYS, row["job_id"]
        assert row["wave"] == M.WAVE_PUBLISHED and row["status"] == M.STATUS_LAUNCH and row["extension"] is False
        assert row["log_requests"] is False and (row["geometry_eval"], row["geometry_L"]) == (M.GEOMETRY_EVAL, M.GEOMETRY_L)
        assert row["view"] == M.VIEW_MAIN and row["thinking_mode"] == "on" and row["si_sha256"] == M.SI_SHA256_BY_CELL.get(row["cell"])
        assert row["staged_id"] == (M.OBJECT_STAGED_ID if row["fitted"] else None) and row["budget_class"] == M.budget_class(row["cell"])
    r = subprocess.run([sys.executable, str(M5T / "cells.py"), "list", "--json", "--published-keys"], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout) == fixture and r.stdout.strip() == json.dumps(fixture, indent=1).strip()
    base = subprocess.run(["git", "show", f"{PUBLISHED_TREE}:scripts/sdfbase/math500/cells.py"], cwd=REPO, capture_output=True, text=True)
    if base.returncode == 0:      # the fixture's PROVENANCE: the published tree's own generator over this tree's pins
        prog = ("import sys; src = sys.stdin.read(); g = {'__file__': sys.argv[1], '__name__': '__main__'}; "
                "sys.argv = ['cells.py', 'list', '--json']; exec(compile(src, sys.argv[0], 'exec'), g)")
        r2 = subprocess.run([sys.executable, "-c", prog, str(M5T / "cells.py")], input=base.stdout, capture_output=True, text=True, cwd=REPO)
        assert r2.returncode == 0, r2.stderr[-800:]
        # the published tree spells the cluster preference with the site label it had then and the sampling rows' provenance prose
        # as it was worded then; every other key is compared exactly
        assert _sampling_source_masked(_cluster_preference_masked(json.loads(r2.stdout))) == \
            _sampling_source_masked(_cluster_preference_masked(fixture))


def test_extension_rows_are_the_nine_w1_rows_in_the_eight_arm_order_with_their_own_geometry_tuples_and_roots():
    """P2 / §4.1 / A4-A6: rows 24-32 = gemma / qwen / nemotron x (constitution SI, LoRA ctl2, LoRA mix) under the PRIMARY
    sampling row, wave ext1, status launch, the `_lora2` tuple + the base pin on the six LoRA rows, the long-SI entry on
    the three constitution rows, one --time class, WAVE-ISOLATED out roots, the read-back opt-in on; the counts report
    base and extension apart so the published acceptance numbers keep their meaning."""
    ext = M.EXTENSION_ROWS
    assert [r["index"] for r in ext] == list(range(24, 33)) and M.M5T_CELLS[24:33] == ext and M.M5T_CELLS[:24] == M.BASE_ROWS
    assert M.M5T_CELLS[33:41] == M.STEER_ROWS and M.M5T_CELLS[41:47] == M.LLAMA_ROWS      # the steering rung, then the llama rows (their own tests below)
    assert M.M5T_CELLS[47:53] == M.ITDOC_ROWS and len(M.ITDOC_ROWS) == 6                    # E3: the it-doc rows (their own test below)
    # arditi-qwen-all-positions: APPENDED LAST, so every slice above is where it was (its own test below)
    assert M.M5T_CELLS[53:] == M.ALLPOS_ROWS and len(M.ALLPOS_ROWS) == 1
    assert [(r["model_key"], r["cell"]) for r in ext] == [(k, c) for k in M.EXT_MODELS for c in M.EXT_CELLS]
    assert M.EXT_CELLS == (M.CELL_PROMPTED_CONST, M.CELL_LORA_CTL2, M.CELL_LORA_MIX)      # the eight-arm order
    assert M.EXT_MODELS == C.M5T_EXT_MODELS == ("gemma4e4b", "qwen35_4b", "nemotron3nano4b")
    assert M.ALL_CELLS == M.CELLS + M.EXT_CELLS and M.FITTED_CELLS == (M.CELL_OBJECT, M.CELL_LORA_MIX, M.CELL_LORA_CTL2)
    # the seed wave is a SIBLING LIST beside the matrix — WAVES (over LAUNCHER_ROWS)
    # names it last; M5T_CELLS itself is untouched (its own test below)
    assert M.WAVES == (M.WAVE_PUBLISHED, M.WAVE_EXT1, M.WAVE_STEER, M.WAVE_LLAMA, M.WAVE_ITDOC, M.WAVE_ALLPOS, M.WAVE_SEEDREP) and M.COMPLETED_WAVES == (M.WAVE_PUBLISHED,)
    assert len(M.M5T_CELLS) == 54 and len(M.SEEDREP_ROWS) == 33 and M.LAUNCHER_ROWS == M.M5T_CELLS + M.SEEDREP_ROWS and len(M.LAUNCHER_ROWS) == 87   # 33: rows 84-86 included
    pins = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())
    for r in ext:
        assert r["wave"] == M.WAVE_EXT1 and r["status"] == M.STATUS_LAUNCH and r["extension"] is True and r["log_requests"] is True
        assert r["sampling_mode"] == M.SAMPLING_THINKING_CARD and r["thinking_mode"] == "on" and r["view"] == M.VIEW_MAIN
        assert r["sampling"] == M.SAMPLING_ROWS[(r["model_key"], M.SAMPLING_THINKING_CARD)]
        assert r["time_limit"] == M.TIME_EXT == "0-06:00:00" and r["cluster_preference"] == "cluster_a"
        assert r["out_root"] == f"controls_ext1/{r['model_key']}/{M.cell_slug(r['cell'])}/thinking_card"
        assert not r["out_root"].startswith("controls/") and r["job_id"] == f"m5t-{r['model_key']}-{M.cell_short(r['cell'])}"
        assert r["cluster_constraint"] == (M.CLUSTER_CONSTRAINT_PRIMARY if r["model_key"] == "nemotron3nano4b" else M.CLUSTER_CONSTRAINT_ANY)
        assert r["harness_cell"] == M.HARNESS_BY_CELL[r["cell"]] and r["route_kind"] == M.route_kind(r["cell"])
        assert r["reasoning_parser"] == ("gemma4" if r["model_key"] == "gemma4e4b" else None)
        if r["cell"] == M.CELL_PROMPTED_CONST:
            assert r["fitted"] is False and r["route_kind"] == "prompted" and r["tuple_name"] is None
            assert r["object_sha256"] is None and r["staged_id"] is None and r["served_positions_expected"] is None
            assert (r["geometry_eval"], r["geometry_L"]) == ("capability:gen_long_math_longsi", 24064)
            assert r["budget_class"] == M.CELL_PROMPTED_CONST and r["si_sha256"] == CONSTITUTION_SI_SHA256
            assert r["harness_cell"] == "prompted__constitution" and r["expected_stamps"] == ["cell-stable", "not-consulted", "thinking_on"]
        else:
            assert r["fitted"] is True and r["route_kind"] == "lens"
            assert r["tuple_name"] == f"math500_thinking_on_full22_{r['model_key']}_lora2" == C.m5t_ext_tuple_name(r["model_key"])
            assert r["staged_id"] == M.STAGED_ID_BY_CELL[r["cell"]] == f"base__{r['cell']}"
            assert r["object_sha256"] == pins[r["model_key"]][r["staged_id"]]["object_sha256"] and len(r["object_sha256"]) == 64
            assert (r["geometry_eval"], r["geometry_L"]) == (M.GEOMETRY_EVAL, M.GEOMETRY_L) and r["budget_class"] == M.CELL_NONE
            assert r["served_positions_expected"] == "all" and r["expected_stamps"] == ["cell-stable", "always", "thinking_on"]
            assert r["twin_expected_stamps"] == ["cell-stable", "always", "thinking_on"] and r["si_sha256"] is None
    # the three extension pins are distinct from each other and from the minus_ctl pins (+ the rung's eight externals + the
    # llama wave's three main-view objects)
    n_llama_fitted = sum(r["fitted"] for r in M.LLAMA_ROWS)
    n_itdoc, n_allpos = len(M.ITDOC_ROWS), len(M.ALLPOS_ROWS)
    itdoc_pins = {r["object_sha256"] for r in M.ITDOC_ROWS if r["object_sha256"]}       # E3: a held it-doc row may carry no pin yet (null, never a placeholder)
    assert len({r["object_sha256"] for r in M.M5T_CELLS if r["fitted"]} - {None}) == 4 + 6 + len(M.STEER_ROWS) + n_llama_fitted + len(itdoc_pins) + n_allpos
    # the counts: the whole matrix, then base and extension apart
    # the rung's and the llama wave's statuses are DERIVED (a row is held iff its family's render fact is not certified on this
    # tree — see test_the_steer_rows_launchability_follows_the_familys_certified_render_fact), never a count typed here
    n_steer, n_llama = len(M.STEER_ROWS), len(M.LLAMA_ROWS)
    n_held = sum(r["status"] == M.STATUS_HELD for r in M.STEER_ROWS + M.LLAMA_ROWS + M.ITDOC_ROWS + M.ALLPOS_ROWS)
    n_all = 33 + n_steer + n_llama + n_itdoc + n_allpos
    assert COUNTS["rows"] == n_all and COUNTS["fitted"] == 14 + n_steer + n_llama_fitted + n_itdoc + n_allpos
    assert COUNTS["log_requests"] == 9 + n_steer + n_llama + n_itdoc + n_allpos
    assert COUNTS["waves"] == {"published": 24, "ext1": 9, "steer": n_steer, "llama": n_llama, "itdoc": n_itdoc, "allpos": n_allpos}
    assert COUNTS["statuses"] == ({"launch": n_all} if not n_held else {"launch": n_all - n_held, "held": n_held})
    n_llama_const = sum(r["cell"] == M.CELL_PROMPTED_CONST for r in M.LLAMA_ROWS)
    assert COUNTS["geometry"] == {"capability:gen_long_math": 30 + n_steer + n_llama - n_llama_const + n_itdoc + n_allpos,
                                  "capability:gen_long_math_longsi": 3 + n_llama_const}
    assert COUNTS["budget_classes"] == {M.CELL_NONE: 22 + n_steer + n_llama_fitted + 1 + n_itdoc + n_allpos,
                                        M.CELL_PROMPTED: 8 + 1, M.CELL_PROMPTED_CONST: 3 + n_llama_const}
    # the all-positions arm's tuple is its OWN (never the rung's `_steer`), so it is a FIFTEENTH name here
    assert COUNTS["distinct_out_roots"] == n_all == len({r["job_id"] for r in M.M5T_CELLS}) and len(COUNTS["tuple_names"]) == 14 + n_allpos
    assert COUNTS["sampling"] == {"thinking_card": 12 + 9 + 6 + n_itdoc + n_allpos, "general": 3, "battery": 9, "card": 2 + n_llama}     # the no-switch rows ride `card`
    b, e = COUNTS["base"], COUNTS["extension"]
    assert b["rows"] == 24 and b["fitted"] == 8 and b["sampling"] == {"thinking_card": 12, "general": 3, "battery": 9}
    assert b["cells"] == {M.CELL_NONE: 8, M.CELL_PROMPTED: 8, M.CELL_OBJECT: 8} and b["reasoning_parser_gemma4"] == 3
    assert b["distinct_out_roots"] == 24 and len(b["tuple_names"]) == 8 and b["waves"] == {"published": 24}
    assert e["rows"] == 9 and e["fitted"] == 6 and e["sampling"] == {"thinking_card": 9, "general": 0, "battery": 0}
    assert e["cells"] == {M.CELL_PROMPTED_CONST: 3, M.CELL_LORA_CTL2: 3, M.CELL_LORA_MIX: 3} and e["reasoning_parser_gemma4"] == 3
    assert e["tuple_names"] == sorted(C.m5t_ext_tuple_name(m) for m in M.EXT_MODELS) and e["distinct_out_roots"] == 9
    assert e["geometry"] == {"capability:gen_long_math": 6, "capability:gen_long_math_longsi": 3} and e["log_requests"] == 9
    assert e["budget_classes"] == {M.CELL_PROMPTED_CONST: 3, M.CELL_NONE: 6} and e["waves"] == {"ext1": 9}
    assert set(e["pins"]) == {f"{k}/{sid}" for k in M.EXT_MODELS for sid in (M.STAGED_ID_BY_CELL[M.CELL_LORA_MIX], M.STAGED_ID_BY_CELL[M.CELL_LORA_CTL2])}
    assert M.counts(M.BASE_ROWS) == {k: v for k, v in b.items()} and "base" not in M.counts(M.BASE_ROWS)


def test_matrix_base_rows_are_the_published_24_and_every_count_is_derived():
    rows = M.BASE_ROWS
    assert [r["index"] for r in rows] == list(range(24))
    COUNTS = M.counts(rows)          # noqa: N806 — the published acceptance counts, over the base rows
    assert COUNTS["rows"] == 24 and COUNTS["fitted"] == 8
    assert COUNTS["sampling"] == {"thinking_card": 12, "general": 3, "battery": 9}
    assert COUNTS["cells"] == {M.CELL_NONE: 8, M.CELL_PROMPTED: 8, M.CELL_OBJECT: 8}
    assert COUNTS["reasoning_parser_gemma4"] == 3 and COUNTS["distinct_out_roots"] == 24
    assert len(COUNTS["tuple_names"]) == 8 and len(set(r["job_id"] for r in rows)) == 24
    # gemma has no battery row; qwen has three modes; glm / nemotron two
    assert {r["sampling_mode"] for r in rows if r["model_key"] == "gemma4e4b"} == {"thinking_card"}
    assert {r["sampling_mode"] for r in rows if r["model_key"] == "qwen35_4b"} == {"thinking_card", "general", "battery"}
    for key in ("glm46vflash", "nemotron3nano4b"):
        assert {r["sampling_mode"] for r in rows if r["model_key"] == key} == {"thinking_card", "battery"}
    # fitted rows: the object cell under every mode, each with the pin from configs/sdfbase_full22_pins.json
    pins = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())
    for r in rows:
        assert r["fitted"] is (r["cell"] == M.CELL_OBJECT) and r["route_kind"] == M.route_kind(r["cell"])
        if r["fitted"]:
            assert r["object_sha256"] == pins[r["model_key"]][M.OBJECT_STAGED_ID]["object_sha256"] == M.load_pins()[(r["model_key"], M.OBJECT_STAGED_ID)]
            assert r["tuple_name"].startswith(f"math500_thinking_on_full22_{r['model_key']}")
            assert r["expected_stamps"] == ["cell-stable", "always", "thinking_on"] and r["served_positions_expected"] == "all"
        else:
            assert r["object_sha256"] is None and r["tuple_name"] is None
            assert r["expected_stamps"] == ["cell-stable", "not-consulted", "thinking_on"]
        assert r["out_root"] == f"controls/{r['model_key']}/{M.cell_slug(r['cell'])}/{r['sampling_mode']}"
        assert r["time_limit"] == (M.TIME_OBJECT_GLM if r["fitted"] and r["model_key"] == "glm46vflash"
                                   else M.TIME_OBJECT if r["fitted"] else M.TIME_BASE)
        # a row carries EXACTLY the keys its source names (plus the provenance)
        assert set(r["sampling"]) - {"source"} <= set(M.SAMPLING_PARAM_KEYS) and r["sampling"]["source"]
    assert {r["sampling_mode"] for r in rows} == set(M.SAMPLING_MODES)
    # the three battery rows are the vendor presets, cited
    for key, want in (("qwen35_4b", {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}),
                      ("nemotron3nano4b", {"temperature": 1.0, "top_p": 1.0}),
                      ("glm46vflash", {"temperature": 0.8, "top_p": 0.6, "top_k": 2, "repetition_penalty": 1.1})):
        row = M.SAMPLING_ROWS[(key, "battery")]
        assert {k: v for k, v in row.items() if k != "source"} == want and "5a50e47b" in row["source"]
    with pytest.raises(SystemExit, match="not a matrix row"):
        M.row_for("gemma4e4b", M.CELL_NONE, "battery")


def test_the_extension_mirror_is_checked_against_contracts_the_registry_and_the_geometry_table():
    """The extension's import-light mirror (harness ids, the prompted SIs and their digests, the prompt classes and
    their geometry entries, the tuple names, the recorder's caps) is CHECKED against contracts, the registry adapter
    and the geometry table — never trusted."""
    # loaded by PATH under a unique name: scripts/arditi/ carries a module of the same basename, and under a shard layout that imported
    # it first `import record_resolved_sampler` hands back the arditi copy (no PROBE_EXPECT_CHOICES) — the gate flake of 2026-10-08
    rrs = _load("sdfbase_record_resolved_sampler", "scripts/sdfbase/record_resolved_sampler.py")
    from capctrl_serving_up import resolve_standup_geometry
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.serving import geometry as sg
    assert set(M.HARNESS_BY_CELL) == set(M.ALL_CELLS) | set(M.STEER_CELLS) | set(M.SEEDREP_ONLY_CELLS)     # + the seed wave's PSR cells
    # arditi-qwen-all-positions: the mirror is pinned to the adapter for every cell the SERVING vocabulary knows, and to a
    # DECLARED gap for the ones it does not — `PENDING_SERVING_CELLS`. The set is pinned EQUAL to the cells the vocabulary
    # actually refuses; it carried `arditi_allpos` while that cell's Class B registration was in flight in another
    # author's change, this test failed by name the day it landed, and the entry left the pending list and joined the
    # compared mirror below. EMPTY today: the vocabulary refuses none of the mirror's cells, all-positions included.
    def _vocabulary_refuses(cell: str) -> bool:
        try:
            C.parse_launcher_cell(cell)
        except C.CellNameError:
            return True
        return False
    assert {c for c in M.HARNESS_BY_CELL if _vocabulary_refuses(c)} == set(M.PENDING_SERVING_CELLS) == set()
    assert not _vocabulary_refuses(M.CELL_ARDITI_ALLPOS) and M.CELL_ARDITI_ALLPOS in M.HARNESS_BY_CELL
    for cell in M.PENDING_SERVING_CELLS:
        with pytest.raises(ra.RegistryAdapterError, match="Class B vocabulary"):
            ra.harness_cell_id(cell)
        # the mirror still spells it the way a bare Class B id is spelled — the harness id IS the TrainTest id
        assert M.HARNESS_BY_CELL[cell] == cell and M.route_kind(cell) == "lens" and M.prompted_si(cell) is None
    for cell, hid in M.HARNESS_BY_CELL.items():
        if cell in M.PENDING_SERVING_CELLS:
            continue
        assert hid == ra.harness_cell_id(cell) == C.parse_launcher_cell(cell).harness_id, cell
        parsed = C.parse_launcher_cell(cell)          # a lens cell's `si` is the OBJECT's SI; the prompted SI is a prompted cell's
        assert (parsed.si if parsed.kind == "prompted" else None) == M.prompted_si(cell)
        assert {"base": "base", "prompted": "prompted", "fitted": "lens"}[parsed.kind] == M.route_kind(cell)   # the grammar's kinds -> the route kinds
    for cell, si in M.PROMPTED_SI_BY_CELL.items():
        assert ra.HARNESS_TO_TRAINTEST_SI[M.HARNESS_BY_CELL[cell][len("prompted__"):]] == si
        assert hashlib.sha256(ra._traintest_si_text_bytes(si)).hexdigest() == M.SI_SHA256_BY_CELL[cell], cell
    assert M.SI_SHA256_BY_CELL[M.CELL_PROMPTED_CONST] == CONSTITUTION_SI_SHA256          # asserted, never copied
    # the prompt classes and their measured entries, per cell
    for cell in M.ALL_CELLS + M.STEER_CELLS:
        cls = C.math500_prompt_class_of_si(M.prompted_si(cell))
        assert M.prompt_class(cell) == cls and M.geometry_for(cell) == (C.math500_geometry_key(cls), int(sg.geometry_for(C.math500_geometry_key(cls)).max_model_len))
        assert M.budget_class(cell) == (cell if cell.startswith("prompted__") else M.CELL_NONE)
    assert M.PROMPT_CLASS_OF_SI == {si: C.MATH500_PROMPT_CLASS_OF_SI[si] for si in M.PROMPT_CLASS_OF_SI}
    assert M.GEOMETRY_BY_CLASS == {c: (C.MATH500_GEOMETRY_KEY_OF_CLASS[c], int(sg.geometry_for(C.MATH500_GEOMETRY_KEY_OF_CLASS[c]).max_model_len))
                                   for c in (M.PROMPT_CLASS_SHORT, M.PROMPT_CLASS_LONG_SI)}
    assert M.GEOMETRY_BY_CLASS[M.PROMPT_CLASS_LONG_SI] == ("capability:gen_long_math_longsi", 24064)
    # S at the long-SI window, per model, through the standup's own resolver (A4: 24 / 30 / 32 — qwen at the 2026-09-13 row
    # 741,888 // 24,064; it was 29 at the swept 720,896)
    for key, want in {"gemma4e4b": 24, "qwen35_4b": 30, "nemotron3nano4b": 32}.items():
        geom = resolve_standup_geometry("capability:gen_long_math_longsi", None, None, model_key=key)
        assert (int(geom.max_model_len), int(geom.max_num_seqs)) == (24064, want), key
        # the MEASURED constitution prompt maxima fit that window with the cap (A10; the entry's pooled basis max is 7,622)
        assert CONSTITUTION_PROMPT_MAX[key] + M.CAP <= 24064 and CONSTITUTION_PROMPT_MAX[key] <= int(sg.geometry_for("capability:gen_long_math_longsi").basis.max_prompt)
    # the tuple names, per fitted cell
    for key in M.EXT_MODELS:
        assert M.tuple_name(key, M.SAMPLING_THINKING_CARD, M.CELL_LORA_MIX) == M.tuple_name(key, M.SAMPLING_THINKING_CARD, M.CELL_LORA_CTL2) == C.m5t_ext_tuple_name(key)
        assert M.tuple_name(key, M.SAMPLING_THINKING_CARD) == f"math500_thinking_on_full22_{key}" and C.m5t_ext_tuple_name(key) in C.M5T_EXPERIMENT_NAMES
        with pytest.raises(SystemExit, match="thinking_card"):
            M.tuple_name(key, M.SAMPLING_BATTERY, M.CELL_LORA_MIX)
    with pytest.raises(SystemExit, match="not a fitted cell"):
        M.tuple_name("gemma4e4b", M.SAMPLING_THINKING_CARD, M.CELL_PROMPTED_CONST)
    assert set(M.EXT_CELLS) - {M.CELL_PROMPTED_CONST} == set(C.M5T_EXT_CELLS) and M.EXT_MODELS == C.M5T_EXT_MODELS
    # the recorder's contract: the probe class is partitioned by the driver's KIND
    # tag read back from the request evidence beside the store; the launcher passes NO --probe-cap — the recorder's default
    # under `kinds` is every cap its kind table pins, mirrored here for the one check the launcher owns: the client cap
    # collides with no kind cap (the recorder's refusal condition); the numeric projection is exactly the recorder's keys
    assert M.RECORDER_PROBE_EXPECT == "kinds" and "kinds" in rrs.PROBE_EXPECT_CHOICES
    assert M.RECORDER_PROBE_TAGS_FILENAME == M.REQUEST_EVIDENCE_FILENAME == C.REQUEST_EVIDENCE_FILENAME
    # (the load_check kind fires at the run's per-leg cap — no pinned cap — so the projection is the PINNED caps; recorder-load-check-kinds)
    assert M.RECORDER_PROBE_CAPS == (1, 48, 64, 256, 512, 2048) == tuple(sorted(c for k in rrs.PROBE_KINDS for c in (rrs._pinned_caps(k) or ())))
    assert M.CAP not in M.RECORDER_PROBE_CAPS
    assert 48 == rrs.PROBE_MAX_TOKENS if hasattr(rrs, "PROBE_MAX_TOKENS") else True
    assert tuple(rrs.SAMPLER_KEYS) == M.SAMPLING_PARAM_KEYS and (REPO / M.RESOLVED_SAMPLER_SCRIPT_REL).is_file()
    assert M.sampling_params({"temperature": 0.6, "top_p": 0.95, "source": "prose", "top_k": 20}) == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    # the rung's two cells are bare Class B ids the serving grammar admits, so their harness id IS the cell id,
    # and the `_steer` tuple is the contracts' (llama, a no-switch family, gets None — its row is in support)
    # the rung's two cells are the contracts' `_steer` cells; the steer vocabulary additionally carries the
    # all-positions arm, which is its own wave and its own tuple and must NOT widen the rung's (its own test below)
    assert M.WAVE_STEER_CELLS == C.M5T_STEER_CELLS == ("alphasteer", "arditi")
    assert M.STEER_CELLS == M.WAVE_STEER_CELLS + M.ALLPOS_CELLS == ("alphasteer", "arditi", "arditi_allpos")
    assert set(M.WAVE_STEER_CELLS) <= set(M.STEER_CELLS) <= set(C.launchable_class_b_ids())   # the all-positions arm too, now
    assert set(M.PENDING_SERVING_CELLS).isdisjoint(C.launchable_class_b_ids())      # the declared gap, from the other side
    for key in C.M5T_STEER_MODELS:
        for cell in M.WAVE_STEER_CELLS:
            assert M.tuple_name(key, M.SAMPLING_THINKING_CARD, cell) == C.m5t_steer_tuple_name(key) in C.M5T_EXPERIMENT_NAMES
        with pytest.raises(SystemExit, match="thinking_card"):
            M.tuple_name(key, M.SAMPLING_BATTERY, M.CELL_ALPHASTEER)
    assert M.tuple_name("llama8b", M.SAMPLING_THINKING_CARD, M.CELL_ARDITI) is None      # no switch -> in support, no tuple
    assert M.FLAG_ENABLE_LOG_REQUESTS == "--enable-log-requests" and f'"{M.FLAG_ENABLE_LOG_REQUESTS}"' in (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()
    # the stamp carries the client lock the leg RAN UNDER; the launcher's lock path is the contracts'
    assert M.ROW_IDENTITY_KEYS == ("object_artifact_sha256", "route_kind", "view", "producer_tree_sha", "thinking", "sampling_mode",
                                   "sampling_ref", "pin", "client_lock_sha256")
    assert M.CLIENT_LOCK_REL == C.CLIENT_LOCK_TXT and M.MANIFEST_FRAMEWORKS_KEY == "frameworks"


def test_prompted_si_constitution_full_is_refused_by_name_at_every_gate(monkeypatch):
    """The _full variant is not a legal cell OF THIS EXPERIMENT — the matrix helpers and the overlay
    generator refuse it BY NAME; an unknown prompted id likewise; and the overlay generator refuses a fitted extension
    cell the roster does not stage on the `main` view. The refusal is this experiment's gate, not the vocabulary's: the
    harness translates the id (another lane's prompted cell, `prompted__constitution_full`, since the Llama wave landed),
    so the launcher cell parser and the registry adapter parse it — and the matrix still refuses it."""
    from sisteer.capeval.petri import registry_adapter as ra
    bad = M.CELL_PROMPTED_CONST_FULL_REFUSED
    assert bad == "prompted__si_constitution_full" and bad not in M.ALL_CELLS and bad not in M.HARNESS_BY_CELL
    for fn in (M.prompted_si, M.route_kind, M.prompt_class, M.geometry_for, M.budget_class):
        with pytest.raises(SystemExit, match="not a legal cell of this experiment"):
            fn(bad)
    parsed = C.parse_launcher_cell(bad)                                     # the vocabulary knows it ...
    assert parsed.kind == "prompted" and parsed.harness_id == ra.harness_cell_id(bad) == "prompted__constitution_full"
    assert parsed.harness_id != M.HARNESS_BY_CELL[M.CELL_PROMPTED_CONST]    # ... as a DIFFERENT cell from this experiment's
    with pytest.raises(SystemExit, match="no TrainTest SI"):
        M.prompted_si("prompted__nope")
    gen = _load("_m5t_overlays_refusal", "scripts/sdfbase/math500/make_math500_overlays.py")
    assert gen.explicit_cells("gemma4e4b") == list(M.ALL_CELLS) and gen.explicit_cells("glm46vflash") == list(M.CELLS)
    monkeypatch.setattr(gen, "M5T_CELLS", list(gen.M5T_CELLS) + [dict(gen.M5T_CELLS[24], cell=bad)])
    with pytest.raises(SystemExit, match="not a legal cell of this experiment"):
        gen.explicit_cells("gemma4e4b")
    monkeypatch.setattr(gen, "M5T_CELLS", list(M.M5T_CELLS))
    gen.check_fitted_cells_staged("gemma4e4b")                           # the roster stages both LoRAs on `main`
    monkeypatch.setattr(gen, "roster_main_cells", lambda key: {M.CELL_OBJECT: "x"})
    with pytest.raises(SystemExit, match="not staged"):
        gen.check_fitted_cells_staged("gemma4e4b")


def test_the_literals_mirror_contracts_and_the_geometry_table():
    """The mirror in cells.py (import-light on purpose) is CHECKED against contracts, never trusted."""
    assert M.FLAG_THINKING == C.FLAG_THINKING and M.FLAG_SAMPLING == C.FLAG_SAMPLING and M.FLAG_SAMPLES == C.FLAG_SAMPLES
    assert M.FLAG_LEDGER_URL == C.FLAG_LEDGER_URL and M.FLAG_OOS_PROMPT_BUDGET == C.FLAG_OOS_PROMPT_BUDGET
    assert M.FLAG_ALLOW_OOS_THINKING == C.FLAG_ALLOW_OOS_THINKING and M.FLAG_EXPECT_OBJECT_SHA256 == C.FLAG_EXPECT_OBJECT_SHA256
    assert M.FLAG_MATH500_RETRY_MAX_TOKENS == C.FLAG_MATH500_RETRY_MAX_TOKENS and M.FLAG_UPSTREAM_TIMEOUT_S == C.FLAG_UPSTREAM_TIMEOUT_S
    # the liveness-gate flag and the value the steer rows pass (the logprob mode)
    assert M.FLAG_LENS_LIVENESS == C.FLAG_LENS_LIVENESS == "--lens-liveness" and M.LENS_LIVENESS_STEER == C.LENS_LIVENESS_LOGPROB == "logprob"
    assert C.LENS_LIVENESS_LOGPROB in C.LENS_LIVENESS_MODES
    # TWO statements, both kept at the merge of the serving registration. FIRST the registry's CONTENT (the registration
    # lane's own tripwire, moved by them when the tuple widened): exactly the two original kinds plus the three CONSTANT
    # residual ones, `psr` still out.
    assert C.LOGPROB_LIVENESS_KINDS == ("alphasteer", "static_r", "vector_all", "vector", "diffmean")
    # the gate's KINDS: MEMBERSHIP, never equality. The tuple legitimately grows — the serving registration that admits
    # this lane's all-positions cell widens it too (vector_all, and the other residual-slot kinds) — so pinning its value
    # would make an EXPECTED widening read as a defect. What this lane depends on is that every kind ITS steer cells map
    # to is scoped for the logprob statistic, and that the only ones missing are the kinds of cells the serving
    # vocabulary does not admit yet — the SAME pending landing, declared in one place. Both sides of that equality read
    # {'vector_all'} today and both go empty the moment it lands: it can neither rot nor hide the gap.
    lane_kinds = {SP_MOD.steer_cell_kind(c) for c in M.STEER_CELLS}
    assert lane_kinds - set(C.LOGPROB_LIVENESS_KINDS) == {SP_MOD.steer_cell_kind(c) for c in M.PENDING_SERVING_CELLS}
    assert {"alphasteer", "static_r"} <= lane_kinds & set(C.LOGPROB_LIVENESS_KINDS)      # the rung's two, scoped today
    assert M.UPSTREAM_TIMEOUT_S == C.PROXY_UPSTREAM_TIMEOUT_S == 7200
    assert M.CONTROL == C.CONTROL_MATH500 and M.THINKING_MODE == C.THINKING_ON and M.INSPECT_TASK == "math"
    assert M.LEDGER_FILENAME == C.LEDGER_FILENAME and M.LEDGER_ENDPOINT_FILENAME == C.LEDGER_ENDPOINT_FILENAME
    assert M.OOS_PROMPT_BUDGET_FILENAME == C.OOS_PROMPT_BUDGET_FILENAME
    assert M.INSPECT_BASE_URL_ENV == C.INSPECT_BASE_URL_ENV and M.INSPECT_API_KEY_ENV == C.INSPECT_API_KEY_ENV
    assert C.inspect_task(M.INSPECT_TASK) == "inspect/math" and C.check_thinking_arg(M.THINKING_MODE) == "on"
    # the contracts' flags and modes (on the tree since the integration commit; pinned outright)
    assert M.FLAG_MAX_TOKENS_BY_TASK == C.FLAG_MAX_TOKENS_BY_TASK
    assert M.FLAG_TINYCAP_IDENTITY_VERSION == C.FLAG_TINYCAP_IDENTITY_VERSION
    assert set(M.SAMPLING_MODES) <= set(C.SAMPLING_MODES)
    # the wave root the shell derives before any row is read is the matrix's constant (the mirror is checked)
    assert f'M5T_WAVE_ROOT_REL="{M.WAVE_ROOT_REL}"' in (M5T / "m5t_common.sh").read_text()
    # the cap and the geometry: one pass at 16,384 = the contracts' default; MATH-500's OWN entry
    # capability:gen_long_math (prompt max 1,041 + 16,384 -> 17,664), resolved by the standup for every model
    # with S from the model's budget — 33 / 20 / 44 / 44 (vs 54 / 33 / 73 / 73 at the 10,752 it used to resolve to)
    assert M.CAP == C.MATH500_MAX_TOKENS_DEFAULT == 16384 and M.MAX_TOKENS_BY_TASK == "math=16384"
    assert M.GEOMETRY_EVAL == "capability:gen_long_math" == C.geometry_subentry(
        C.GEOMETRY_CLASS_OF_CONTROL[C.CONTROL_MATH500], prompted=False, thinking=True)
    from capctrl_serving_up import resolve_standup_geometry
    expected_s = {"gemma4e4b": 33, "qwen35_4b": 42, "glm46vflash": 44, "nemotron3nano4b": 44}   # qwen35_4b: 741,888 = 42 x 17,664, MEASURED at this window 2026-09-13
    for key in M.MODEL_KEYS:
        geom = resolve_standup_geometry(M.GEOMETRY_EVAL, None, None, model_key=key)
        assert int(geom.max_model_len) == M.GEOMETRY_L == 17664 and int(geom.max_num_seqs) == expected_s[key], key
        assert 1041 + M.CAP <= int(geom.max_model_len)


def test_sampling_rows_yaml_mirrors_cells_and_when_present_contracts():
    doc = yaml.safe_load((REPO / "configs/sdfbase/math500/sampling_rows.yaml").read_text())
    # the MATH-500 modes first, then the ONE `card` row the GSM8K Llama rows run (llama-paper-rows P6; not a MATH-500 mode)
    assert doc["modes"] == list(M.SAMPLING_MODES) + [C.SAMPLING_CARD] and C.SAMPLING_CARD not in M.SAMPLING_MODES
    seen = set()
    for key, modes in doc["rows"].items():
        for mode, row in modes.items():
            if mode == C.SAMPLING_CARD:
                crow = C.SAMPLING_BY_MODEL_KEY[key]
                assert (key, {k: v for k, v in row.items() if k != "source"}) == ("llama8b", {"temperature": crow.temperature, "top_p": crow.top_p})
                continue
            assert row == M.SAMPLING_ROWS[(key, mode)], (key, mode)
            seen.add((key, mode))
    # the STEER-ONLY model's row is the contracts' shipped card row, mirrored in cells.py and pinned EQUAL here.
    # It is deliberately not in this yaml: the file is the four MATH-500 wave models' sampling registry, and llama8b
    # is a steer-only model of this matrix (no published row, no sensitivity row, no wave of its own here).
    # the no-switch model's row is filed under its shipped `card` MODE (the driver's --sampling card; a `thinking_card` row for
    # llama8b is REFUSED by contracts.sampling_for), and the yaml's ONE `card` row IS that row — mirrored three ways
    steer_only = {(k, m) for (k, m) in M.SAMPLING_ROWS if k in M.STEER_ONLY_MODELS}
    assert steer_only == {("llama8b", M.SAMPLING_CARD)} and seen == set(M.SAMPLING_ROWS) - steer_only and M.SAMPLING_CARD == C.SAMPLING_CARD
    crow = C.SAMPLING_BY_MODEL_KEY["llama8b"]
    assert M.sampling_params(M.SAMPLING_ROWS[("llama8b", M.SAMPLING_CARD)]) == {"temperature": crow.temperature, "top_p": crow.top_p}
    assert M.sampling_params(doc["rows"]["llama8b"][C.SAMPLING_CARD]) == M.sampling_params(M.SAMPLING_ROWS[("llama8b", M.SAMPLING_CARD)])
    assert M.sampling_mode_for("llama8b") == M.SAMPLING_CARD and all(M.sampling_mode_for(k) == M.SAMPLING_THINKING_CARD for k in M.MODEL_KEYS)
    with pytest.raises(C.SamplingUnverified):
        C.sampling_for("llama8b", M.SAMPLING_THINKING_CARD)
    assert C.sampling_for("llama8b", M.SAMPLING_CARD) is crow
    tables = {"thinking_card": C.THINKING_SAMPLING_BY_MODEL_KEY, "general": C.GENERAL_SAMPLING_BY_MODEL_KEY,
              "battery": C.BATTERY_SAMPLING_BY_MODEL_KEY}
    for (key, mode), row in M.SAMPLING_ROWS.items():
        # the steer-only model has no THINKING table row (no thinking mode at all): its row is the SHIPPED card row,
        # pinned against contracts.SAMPLING_BY_MODEL_KEY above
        crow = C.SAMPLING_BY_MODEL_KEY[key] if key in M.STEER_ONLY_MODELS else tables[mode][key]
        for field in M.SAMPLING_PARAM_KEYS:
            assert getattr(crow, field, None) == row.get(field), (key, mode, field)


# --------------------------------------------------------------------------- #
# the overlays                                                                 #
# --------------------------------------------------------------------------- #


def test_overlays_are_generated_and_the_campaign_originals_are_untouched():
    rc = subprocess.run([sys.executable, str(M5T / "make_math500_overlays.py"), "--check"], capture_output=True, text=True, cwd=REPO)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    for key in M.MODEL_KEYS:
        cfg = yaml.safe_load((REPO / f"configs/sdfbase/math500/sdfbase_full22_{key}.math500.yaml").read_text())
        # P2: explicit_cells = THIS model's matrix cells in matrix order — the published three, then (on the three
        # extension models) constitution SI, LoRA ctl2, LoRA mix; glm keeps the published three
        # the steering rows serve their OWN overlay pair (their bank view holds that one cell), so the base overlay
        # lists exactly this model's base-wave and extension cells — never a steering one
        want_cells = list(M.ALL_CELLS) if key in M.EXT_MODELS else list(M.CELLS)
        assert cfg["explicit_cells"] == want_cells == list(dict.fromkeys(
            r["cell"] for r in M.M5T_CELLS if r["model_key"] == key and r.get("object_source") != "external"))
        campaign = yaml.safe_load((REPO / f"configs/sdfbase_full22_{key}.yaml").read_text())
        assert {k: v for k, v in cfg.items() if k != "explicit_cells"} == {k: v for k, v in campaign.items() if k != "explicit_cells"}
        man = yaml.safe_load((REPO / f"configs/capeval/manifests/petri_full22_{key}.math500.yaml").read_text())["petri"]
        cman = yaml.safe_load((REPO / f"configs/capeval/manifests/petri_full22_{key}.yaml").read_text())["petri"]
        assert man["cells"] == [M.HARNESS_BY_CELL[c] for c in want_cells]
        assert man["cells"][:3] == ["none", "prompted__safety", M.OBJECT_HARNESS_ID]
        if key in M.EXT_MODELS:
            assert man["cells"][3:] == ["prompted__constitution", "si_constv2ctl2_full__lora_r4_ce", "si_constv2_mix_full__lora_r4_ce"]
        assert {k: v for k, v in man.items() if k != "cells"} == {k: v for k, v in cman.items() if k != "cells"}
        # F2 / R14: the manifest hashes the standup verifies are UNCHANGED (the constitution's hex16 was already there)
        assert man["prompted_prefix_hashes"]["constitution"] == CONSTITUTION_SI_SHA256[:16] == cman["prompted_prefix_hashes"]["constitution"]
        assert man["chat_template_hash"] == cman["chat_template_hash"]
        # the campaign originals equal the merged tip the lanes were cut from, but for the one ruled change
        for rel in (f"configs/sdfbase_full22_{key}.yaml", f"configs/capeval/manifests/petri_full22_{key}.yaml"):
            base = subprocess.run(["git", "show", f"{BASE_SHA}:{rel}"], cwd=REPO, capture_output=True, text=True)
            if base.returncode != 0:
                pytest.skip("the base sha is not in this clone")
            _assert_original_unchanged_but_for_the_ruled_cap(rel, base.stdout, _pre_release_text(rel, (REPO / rel).read_text()))


@pytest.mark.parametrize("key", ["nemotron3nano4b", "glm46vflash"])
def test_a_roster_declared_prompted_cell_in_the_campaign_original_is_held_out_of_the_overlay(tmp_path, monkeypatch, key):
    """The overlays serve EXACTLY the matrix's cells for the model whatever else the campaign original lists — a
    prompted baseline the roster declares (`prompted__si_constitution`; the manifest spells it `prompted__constitution`)
    rides in the original's explicit_cells / cells and reaches an overlay ONLY where the matrix schedules it (the
    eight-arm extension's rows 24-32 on the three extension models; glm keeps the published three, so there the hold is
    total). Appended to the originals' cell blocks it leaves NO trace either way: the rendered overlay pair is
    byte-identical to the committed one."""
    gen = _load("_m5t_overlays_under_test", "scripts/sdfbase/math500/make_math500_overlays.py")
    want_cells = list(M.ALL_CELLS) if key in M.EXT_MODELS else list(M.CELLS)
    cfg = (REPO / f"configs/sdfbase_full22_{key}.yaml").read_text()
    cfg2 = re.sub(r"^(explicit_cells:\n(?:- .*\n)+)", lambda m: m.group(1) + "- prompted__si_constitution\n", cfg, count=1, flags=re.M)
    man = (REPO / f"configs/capeval/manifests/petri_full22_{key}.yaml").read_text()
    man2 = re.sub(r"^(  cells:\n(?:    - .*\n)+)", lambda m: m.group(1) + "    - prompted__constitution\n", man, count=1, flags=re.M)
    assert cfg2 != cfg and man2 != man
    assert "prompted__si_constitution" in yaml.safe_load(cfg2)["explicit_cells"] and "prompted__constitution" in yaml.safe_load(man2)["petri"]["cells"]
    (tmp_path / f"sdfbase_full22_{key}.yaml").write_text(cfg2)
    (tmp_path / f"petri_full22_{key}.yaml").write_text(man2)
    monkeypatch.setattr(gen, "campaign_config", lambda k: tmp_path / f"sdfbase_full22_{k}.yaml")
    monkeypatch.setattr(gen, "campaign_manifest", lambda k: tmp_path / f"petri_full22_{k}.yaml")
    text = gen.render_config(key)
    assert yaml.safe_load(text)["explicit_cells"] == want_cells
    assert want_cells[:3] == [M.CELL_NONE, M.CELL_PROMPTED, M.CELL_OBJECT]
    assert (M.CELL_PROMPTED_CONST in want_cells) == (key in M.EXT_MODELS)
    assert text == (REPO / f"configs/sdfbase/math500/sdfbase_full22_{key}.math500.yaml").read_text()
    mtext = gen.render_manifest(key)
    assert yaml.safe_load(mtext)["petri"]["cells"] == [M.HARNESS_BY_CELL[c] for c in want_cells]
    assert yaml.safe_load(mtext)["petri"]["cells"][:3] == ["none", "prompted__safety", M.OBJECT_HARNESS_ID]
    assert mtext == (REPO / f"configs/capeval/manifests/petri_full22_{key}.math500.yaml").read_text()
    # the drift helper admits exactly that addition on the originals (against BASE_SHA's text, as the overlay test reads them),
    # and refuses a non-prompted addition by name
    cfg_rel, man_rel = f"configs/sdfbase_full22_{key}.yaml", f"configs/capeval/manifests/petri_full22_{key}.yaml"
    base = subprocess.run(["git", "show", f"{BASE_SHA}:{cfg_rel}"], cwd=REPO, capture_output=True, text=True)
    mbase = subprocess.run(["git", "show", f"{BASE_SHA}:{man_rel}"], cwd=REPO, capture_output=True, text=True)
    if base.returncode != 0 or mbase.returncode != 0:
        pytest.skip("the base sha is not in this clone")
    # the originals as the public release found them (their rewrite since is certified by configs/SCRUB_MAP.tsv), with the same additions
    cfg_pre = _pre_release_text(cfg_rel, cfg)
    cfg2_pre = re.sub(r"^(explicit_cells:\n(?:- .*\n)+)", lambda m: m.group(1) + "- prompted__si_constitution\n", cfg_pre, count=1, flags=re.M)
    _assert_original_unchanged_but_for_the_ruled_cap(cfg_rel, base.stdout, cfg2_pre)
    _assert_original_unchanged_but_for_the_ruled_cap(man_rel, mbase.stdout, man2)
    with pytest.raises(AssertionError, match="non-prompted"):
        _assert_original_unchanged_but_for_the_ruled_cap(cfg_rel, base.stdout, cfg_pre.replace("explicit_cells:\n", "explicit_cells:\n- some__object\n", 1))


# --------------------------------------------------------------------------- #
# the sbatch shape: the cluster's hard rules                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sh", ["bash", "zsh"])
@pytest.mark.parametrize("name", SBATCHES + ["m5t_common.sh"])
def test_shell_files_parse_under_bash_and_zsh(name, sh):
    if shutil.which(sh) is None:
        # bash is the shell the jobs run under and must be present; zsh is the workstation's interactive shell and is
        # absent from the CPU-suite container — its half skips there, the bash half still runs (integration I5)
        assert sh == "zsh", f"{sh} is not installed, and the sbatch runs under it"
        pytest.skip(f"{sh} is not installed here (the bash -n parse of {name} runs as its own node)")
    r = subprocess.run([sh, "-n", str(M5T / name)], capture_output=True, text=True)
    assert r.returncode == 0, (sh, r.stderr)


@pytest.mark.parametrize("name", SBATCHES)
def test_sbatch_headers_carry_the_hard_rules(name):
    src = (M5T / name).read_text()
    common = (F22 / "common.sh").read_text() + (M5T / "m5t_common.sh").read_text()
    for line in src.splitlines():
        if "--export=" in line:
            assert "never" in line.lower(), line       # only the rule's own statement names the flag
    assert "--mem=0" not in src
    gpu = name in GPU_SBATCHES
    for m in ("#SBATCH --qos=high", "#SBATCH --partition=general",
              "#SBATCH --open-mode=append", "#SBATCH --no-requeue", "#SBATCH --signal=B:TERM@120", "#SBATCH --time=",
              "#SBATCH --output=/workspace-vast/%u/exp/logs/") + (("#SBATCH --gres=gpu:1", "#SBATCH --mem=200G") if gpu else ()):
        assert m in src, (name, m)
    mem = [l for l in src.splitlines() if l.startswith("#SBATCH --mem=")][0]
    assert int(mem.split("=")[1].rstrip("G")) <= (200 if gpu else 64)
    if not gpu:
        assert "--gres" not in src and "--stage prod" in src and "f22_serving_venv" in src    # CPU-only; the standup's interpreter
    assert "positional only" in src and "never --export=" in src
    assert "trap cleanup EXIT SIGTERM SIGINT SIGQUIT" in src and "f22_stop_requested" in src
    assert "NCCL_SOCKET_IFNAME=" not in src and "NCCL_SOCKET_IFNAME=" not in common   # the site's nccl.conf decides
    for m in ("hf_env.sh", "NCCL_NVLS_ENABLE=0", "PYTORCH_CUDA_ALLOC_CONF", "mirror-figs.sh",
              "export SISTEER_LENS_STACKED_REDUCTIONS=1", "--reduction-mode stacked --require-attested-mode stacked"):
        assert m in src or m in common, (name, m)
    assert "f22_result" in src                         # the RESULT line rides out through mirror-figs
    if gpu:
        assert "STANDUP_STOP" in src                   # the STOP sentinel before any kill
    assert "F22_REQUIRE_NEMO_BACKEND" in common        # every math500 job demands the Nemotron backend attestation


def test_math500_sbatch_grammar_venvs_bank_check_ledger_order():
    src = (M5T / "math500_thinking.sbatch").read_text()
    # FITTED is required and 0|1; ARRAY resolves the row from SLURM_ARRAY_TASK_ID
    assert 'FITTED_ARG="${5:?fitted 0|1}"' in src and 'case "$FITTED_ARG" in 0|1) ;;' in src
    assert 'if [ "${1:-}" = ARRAY ]' in src and 'm5t_row "${ROW_ARGS[@]}"' in src
    assert 'the matrix says $FITTED' in src            # a FITTED that contradicts the matrix refuses
    # the Nemotron variant venv; the client venv; the CHECK-ONLY bank view with the pin iff fitted
    assert 'if [ "$KEY" = nemotron3nano4b ]; then f22_serving_venv nemotron; else f22_serving_venv; fi' in src
    assert "f22_client_venv" in src
    # the bank check is DISPATCHED on the row's object source (an external lane's view has no roster entry)
    assert "\nm5t_bank_view\n" in src
    common = (M5T / "m5t_common.sh").read_text()
    assert 'f22_bank_view_check "$CPY" "$CELL" "$PIN"' in common and 'st_external_bank_view_check "$rel" "$M5T_BANK_CELL_DIR" "$PIN"' in common
    assert "f22_bank_view " not in src.replace("f22_bank_view_check", "")   # never the campaign's build-under-lock path
    # the order: budget artifact -> standup -> endpoint checks -> ledger -> driver -> the two records (the resolved sampler
    # while the standup's log is complete and the standup still the job's; the row-identity stamp) -> teardown
    idx = [src.index(s) for s in ("m5t_budget_artifact", "f22_standup attacks", "m5t_check_endpoint", "m5t_ledger_up || exit 1",
                                  "m5t_driver_argv", '"${DRV[@]}"', "\nm5t_record_resolved_sampler\nm5t_row_identity\n",
                                  "m5t_ledger_down\nf22_standup_stop")]
    assert idx == sorted(idx), idx
    # the RESULT record names the row's wave, thinking mode, geometry entry and the EFFECTIVE read-back opt-in (P6 / A11)
    assert "wave=${M5T_WAVE:-?} thinking=${M5T_THINKING_MODE:-?} geometry=${M5T_GEOMETRY_EVAL:-?} logreq=${M5T_LOG_REQUESTS_EFFECTIVE:-?}" in src
    assert "M5T_LOG_REQUESTS_OVERRIDE" in src and "--export=" not in src.split("set -uo pipefail", 1)[1]
    assert 'f22_standup attacks "$SERVE_DIR/math500" "" "$M5T_GEOMETRY_EVAL" "" "" ""' in src
    assert 'CUDA_VISIBLE_DEVICES=""' in src and '"$M5T_INSPECT_BASE_URL_ENV=$LEDGER_URL"' in src
    assert "@thinking" not in src and "petri_thinking_on" not in src
    # one durable out root per model x cell x sampling row; the RESULT line lands inside it
    common = (M5T / "m5t_common.sh").read_text()
    assert 'OUT_ROOT="$WAVE_ROOT/$M5T_OUT_REL"' in common and '>> "$OUT_ROOT/RESULT"' in src


def test_smoke_and_nemo_probe_sbatch_are_diagnostic_and_prerequisite_shaped():
    smoke = (M5T / "smoke_thinking.sbatch").read_text()
    assert "#SBATCH --time=0-00:48:00" in smoke and "--cell none" in smoke and "diagnostic" in smoke   # 48 = the formula, pinned below
    assert 'SMOKE_ROOT="$WAVE_ROOT/$M5T_SMOKE_OUT_REL/' in smoke and "controls/" not in smoke.split("set -uo pipefail")[1]
    assert "smoke_thinking_probe.py" in smoke and "--n-on" in smoke and "--cap" in smoke and "m5t_ledger_up" in smoke
    assert "--expect-object-sha256" not in smoke     # base route only: no object, no digest
    nemo = (M5T / "nemo_backend_probe.sbatch").read_text()
    assert "export F22_REQUIRE_NEMO_BACKEND=0" in nemo         # the probe WRITES the attestation it must not require
    assert "f22_serving_venv nemotron" in nemo and "nemo_backend_probe.py" in nemo
    for step in (" packages ", " dispatch ", " duty ", " attest "):
        assert step in nemo, step
    assert '--requirements "$BASE_REQS" "$OVERLAY_REQS"' in nemo and "nvidia-smi dmon" in (M5T / "nemo_backend_probe.py").read_text()
    # the same-GPU peer reference — the base venv, the peer row at Nemotron's served concurrency, the identical
    # burst, BEFORE Nemotron; the attest step admits against it at F22_NEMO_DUTY_RATIO (no absolute floor anywhere)
    assert 'f22_serving_venv ""' in nemo and 'REF_KEY="${2:-qwen35_4b}"' in nemo and 'NEMO_S="$TARGET_S"' in nemo
    # Nemotron's standup FIRST (TARGET_S is the standup's attested served concurrency — no row sets it), then the peer at it
    assert nemo.index('f22_standup attacks "$SERVE_DIR/math500"') < nemo.index('NEMO_S="$TARGET_S"') < nemo.index('f22_serving_venv ""')
    assert nemo.index('f22_serving_venv ""') < nemo.index('f22_standup attacks "$REF_DIR/serving/math500"')
    assert '"$REF_DIR/serving/math500" "" "$M5T_GEOMETRY_EVAL" "" "$NEMO_S" ""' in nemo
    # the peer is served at Nemotron's width under the launcher's explicit over-budget flag — on the REFERENCE standup only
    body = nemo.split("set -uo pipefail", 1)[1]
    code = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
    assert code.count("F22_STANDUP_EXTRA+=(--allow-over-budget)") == 1
    assert code.index('f22_standup attacks "$SERVE_DIR/math500"') < code.index("F22_STANDUP_EXTRA+=(--allow-over-budget)") < code.index('f22_standup attacks "$REF_DIR/serving/math500"')
    assert "--allow-over-budget" not in (M5T / "math500_thinking.sbatch").read_text() and "--allow-over-budget" not in (M5T / "smoke_thinking.sbatch").read_text()
    assert '--duty-reference "$REF_DIR/duty.json" --duty-reference-key "$REF_KEY" --duty-ratio "$F22_NEMO_DUTY_RATIO"' in nemo
    assert "DUTY_FLOOR" not in nemo and "DUTY_FLOOR" not in (M5T / "nemo_backend_probe.py").read_text()
    assert 'ATTESTATION="$PROBE_ROOT/mamba_backend_attestation.json"' in nemo
    assert all(s["fitted"] is False for s in M.M5T_CELLS if s["cell"] == "none")


def test_plan_over_the_published_universe_prints_three_arrays_partitioning_the_rows_never_a_loop():
    """The published wave's planner shape, over the BASE rows as the universe (the python entry point the partition
    tests use; the CLI is wave-scoped since the extension and refuses the published wave — below)."""
    sha = "ab" * 20
    base = M.BASE_ROWS
    lines = M.plan_lines(sha, "cluster_a", max_concurrent=12, rows=base)
    assert len(lines) == 3 and all(l.startswith("sbatch -J m5t-") and "--array=" in l and re.search(r"%\d+ ", l) for l in lines)
    assert sum(_widths(lines)) == 12      # the AGGREGATE allowance
    assert all(l.endswith(f"math500_thinking.sbatch ARRAY {sha}") and f"--qos={SITE_A['qos']} " in l for l in lines)
    covered = _covered(lines)
    assert sorted(covered) == list(range(24)) and len(covered) == 24
    by_time = M.time_classes(base)
    assert set(by_time) == {M.TIME_BASE, M.TIME_OBJECT, M.TIME_OBJECT_GLM}
    assert len(by_time[M.TIME_BASE]) == 16 and len(by_time[M.TIME_OBJECT]) == 6 and len(by_time[M.TIME_OBJECT_GLM]) == 2
    # the whole matrix has FOUR classes: the extension's one class holds all nine rows
    # the whole matrix's fourth class holds the extension's nine rows AND the steering rung (one --time class for both)
    # ... and a FIFTH: the llama wave's own class (1.5 h) holds its six rows
    # ... and a SIXTH: the seed wave's own 4 h class over its 30 rows (time_classes reads LAUNCHER_ROWS)
    assert M.time_classes() == {**by_time, M.TIME_EXT: (list(range(24, 33 + len(M.STEER_ROWS)))
                                                        + [r["index"] for r in M.ITDOC_ROWS + M.ALLPOS_ROWS]),
                                M.TIME_LLAMA: [r["index"] for r in M.LLAMA_ROWS],
                                M.TIME_SEEDREP: [r["index"] for r in M.SEEDREP_ROWS]} and M.CLASS_TAG[M.TIME_EXT] == "ext" and M.CLASS_TAG[M.TIME_LLAMA] == "llama"
    assert M.CLASS_TAG[M.TIME_SEEDREP] == "seedrep" and M.TIME_SEEDREP == "0-04:00:00" and M.TIME_SEEDREP not in by_time
    eur = M.plan_lines(sha, "cluster_b", max_concurrent=4, rows=[r for r in base if r["cluster_constraint"] == M.CLUSTER_CONSTRAINT_ANY])[0]
    assert f"--qos={SITE_B['qos']}" in eur and f"{SITE_B['logs']}/" in eur
    with pytest.raises(SystemExit):
        M.plan_lines(sha, "nope", max_concurrent=1)
    # the CLI REQUIRES --wave (a matrix-wide plan would print completed rows) and refuses the published wave by name (A7 / A12)
    r = subprocess.run([sys.executable, str(M5T / "cells.py"), "plan", "--sha", sha, "--cluster", "cluster_a"], capture_output=True, text=True)
    assert r.returncode == 2 and "--wave" in r.stderr
    r = subprocess.run([sys.executable, str(M5T / "cells.py"), "plan", "--sha", sha, "--cluster", "cluster_a", "--wave", "published"], capture_output=True, text=True)
    assert r.returncode != 0 and "COMPLETE" in r.stderr and "never resubmitted" in r.stderr


def test_plan_is_wave_scoped_asserts_eligibility_and_prints_the_w1_submission_line(tmp_path):
    """`plan --wave ext1` selects that wave's `launch` rows and nothing else; every printed row is asserted
    eligible (a published, held or deferred row refuses BY NAME); `--rows` is a subset of the wave and
    `--cluster-share` complements WITHIN it; `--single-array` prints the ONE line at the class's --time; the record
    line rides the summary; the nemotron rows are primary-cluster-only; the script path is f22_init's layout."""
    sha = "ab" * 20
    script = f"{M.CHECKOUT_ROOT}/{sha}/scripts/sdfbase/math500/math500_thinking.sbatch"
    pl = M.plan(sha, "cluster_a", max_concurrent=9, wave="ext1", single_array=True, assert_eligible_record=True)
    assert pl.rows == list(range(24, 33)) and pl.widths == {M.TIME_EXT: 9} and pl.sizes == {M.TIME_EXT: 9}
    assert pl.lines == [(f"sbatch -J m5t-ext1 --qos={SITE_A['qos']} --time=0-06:00:00 --output={SITE_A['logs']}/%x_%A_%a.out "
                         f"--array=24-32%9 {script} ARRAY {sha}")]
    n_steer, n_held = len(M.STEER_ROWS), sum(r["status"] == M.STATUS_HELD for r in M.STEER_ROWS)
    n_llama, n_llama_held = len(M.LLAMA_ROWS), sum(r["status"] == M.STATUS_HELD for r in M.LLAMA_ROWS)
    n_itdoc, n_itdoc_held = len(M.ITDOC_ROWS), sum(r["status"] == M.STATUS_HELD for r in M.ITDOC_ROWS)
    n_ap, n_ap_held = len(M.ALLPOS_ROWS), sum(r["status"] == M.STATUS_HELD for r in M.ALLPOS_ROWS)
    other_waves_matrix = {"steer/other-wave": n_steer - n_held, **({"steer/held": n_held} if n_held else {}),
                   "llama/other-wave": n_llama - n_llama_held, **({"llama/held": n_llama_held} if n_llama_held else {}),
                   **({"itdoc/other-wave": n_itdoc - n_itdoc_held} if n_itdoc - n_itdoc_held else {}), **({"itdoc/held": n_itdoc_held} if n_itdoc_held else {}),
                   **({"allpos/other-wave": n_ap - n_ap_held} if n_ap - n_ap_held else {}), **({"allpos/held": n_ap_held} if n_ap_held else {})}
    # over the LAUNCHER's universe the seed wave (the sibling list) is named by its wave like every other; `uni` below is the MATRIX alone
    other_waves = {**other_waves_matrix, "qwen_seedrep/other-wave": len(M.SEEDREP_ROWS)}
    assert pl.eligibility == {"wave": "ext1", "rows": list(range(24, 33)), "excluded": {"published/complete": 24, **other_waves}}
    assert any(l.startswith("# assert-eligible: wave=ext1 rows=24-32 all status=launch; excluded published/complete: 24") for l in pl.summary())
    assert M.sbatch_script_path("cluster_a", sha) == script == M.sbatch_script_path("cluster_b", sha)
    # the per-class default form is the same one line (one class present), under the class's job name
    assert M.plan(sha, "cluster_a", max_concurrent=9, wave="ext1").lines == [pl.lines[0].replace("-J m5t-ext1 ", "-J m5t-ext ")]
    assert M.plan(sha, "cluster_a", max_concurrent=9, wave="ext1").eligibility is None      # the assertion RAN; the record is opt-in
    # the eligible set and its guards
    assert [r["index"] for r in M.eligible_rows("ext1")] == list(range(24, 33))
    assert M.plan(sha, "cluster_a", max_concurrent=3, wave="ext1", rows="30-32").rows == [30, 31, 32]
    for spec in ("0-2", "23-25", "0-32"):
        with pytest.raises(SystemExit, match="outside wave 'ext1'"):
            M.plan(sha, "cluster_a", max_concurrent=3, wave="ext1", rows=spec)
    with pytest.raises(SystemExit, match="COMPLETE"):
        M.plan(sha, "cluster_a", max_concurrent=9, wave="published")
    with pytest.raises(SystemExit, match="not a wave"):
        M.plan(sha, "cluster_a", max_concurrent=9, wave="nope")
    # §5.1's overflow: EXPLICIT disjoint row lists inside the wave — gemma + qwen on the overflow cluster, nemotron here
    c = M.plan(sha, "cluster_b", max_concurrent=6, wave="ext1", rows="24-26,27-29")
    assert c.rows == list(range(24, 30)) and all(f"--qos={SITE_B['qos']} " in l and f"{M.CHECKOUT_ROOT}/{sha}/" in l
                                                  and f"--output={SITE_B['logs']}/" in l for l in c.lines)
    b = M.plan(sha, "cluster_a", max_concurrent=3, wave="ext1", rows="30-32")
    assert set(b.rows).isdisjoint(c.rows) and sorted(b.rows + c.rows) == list(range(24, 33))
    with pytest.raises(SystemExit, match="cluster_constraint"):        # nemotron rows never leave the primary cluster
        M.plan(sha, "cluster_b", max_concurrent=3, wave="ext1", rows="30-32")
    # --cluster-share complements WITHIN the wave (never against the matrix: the published rows are never selected)
    assert M.plan(sha, "cluster_a", max_concurrent=3, wave="ext1", cluster_share="30-32").rows == [30, 31, 32]
    assert M.plan(sha, "cluster_b", max_concurrent=6, wave="ext1", cluster_share="30-32").rows == list(range(24, 30))
    with pytest.raises(SystemExit, match="outside wave"):
        M.plan(sha, "cluster_a", max_concurrent=3, wave="ext1", cluster_share="0-14")
    with pytest.raises(SystemExit, match="cluster_constraint"):        # the complement would put nemotron on the overflow cluster
        M.plan(sha, "cluster_b", max_concurrent=3, wave="ext1", cluster_share="24-29")
    # a held / deferred row is never printed (a synthetic universe), and assert_eligible refuses it by name
    uni = [dict(r) for r in M.M5T_CELLS]
    uni[25]["status"] = M.STATUS_HELD
    assert M.plan(sha, "cluster_a", max_concurrent=9, wave="ext1", universe=uni).rows == [24] + list(range(26, 33))
    with pytest.raises(SystemExit, match="outside wave"):
        M.plan(sha, "cluster_a", max_concurrent=9, wave="ext1", rows="24-26", universe=uni)
    with pytest.raises(SystemExit, match="assert-eligible REFUSED.*status=held"):
        M.assert_eligible([uni[25]], "ext1", uni)
    with pytest.raises(SystemExit, match="assert-eligible REFUSED.*wave=published"):
        M.assert_eligible([uni[0]], "ext1", uni)
    rec = M.assert_eligible([uni[24]], "ext1", uni)
    assert rec == {"wave": "ext1", "rows": [24],
                   "excluded": {"published/complete": 24, "ext1/held": 1, "ext1/other-wave": 7, **other_waves_matrix}}
    uni2 = [dict(r) for r in M.M5T_CELLS]
    for r in uni2[24:]:
        r["status"] = M.STATUS_DEFERRED
    with pytest.raises(SystemExit, match="no launchable row"):
        M.plan(sha, "cluster_a", max_concurrent=9, wave="ext1", universe=uni2)
    # the CLI prints §5.1's line and the record; arrays, never a loop
    r = subprocess.run([sys.executable, str(M5T / "cells.py"), "plan", "--sha", sha, "--cluster", "cluster_a", "--wave", "ext1",
                        "--assert-eligible", "--single-array", "--max-concurrent", "9"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert [l for l in r.stdout.splitlines() if l.startswith("sbatch ")] == pl.lines and "for " not in r.stdout
    assert "# assert-eligible: wave=ext1 rows=24-32 all status=launch; excluded published/complete: 24" in r.stdout
    assert f"wave=ext1 rows=9/{len(M.LAUNCHER_ROWS)} (24-32)" in r.stdout and "--single-array" in r.stdout
    r = subprocess.run([sys.executable, str(M5T / "cells.py"), "plan", "--sha", sha, "--cluster", "cluster_a", "--wave", "ext1",
                        "--rows", "0-8", "--max-concurrent", "9"], capture_output=True, text=True)
    assert r.returncode != 0 and "outside wave 'ext1'" in r.stderr and "sbatch " not in r.stdout
    # fix A5: `--cluster` names the cluster's ROLE (primary / overflow) — the same plan as the cluster's own name, which stays
    # accepted (the frozen `cluster_preference` and the published submission lines carry it); an unknown spelling refuses
    # naming the roles; the summary prints the role beside the name
    assert M.resolve_cluster("primary") == M.PRIMARY_CLUSTER == M.resolve_cluster(M.PRIMARY_CLUSTER)
    assert M.resolve_cluster("overflow") == M.OVERFLOW_CLUSTER == M.resolve_cluster(M.OVERFLOW_CLUSTER)
    assert M.CLUSTER_BY_ROLE == {"primary": M.PRIMARY_CLUSTER, "overflow": M.OVERFLOW_CLUSTER} and set(M.CLUSTER_BY_ROLE.values()) == set(M.CLUSTERS)
    with pytest.raises(SystemExit, match="cluster must be one of \\['overflow', 'primary'\\]"):
        M.resolve_cluster("nope")
    with pytest.raises(SystemExit, match="cluster must be one of"):
        M.plan(sha, "nope", max_concurrent=9, wave="ext1")
    by_role = M.plan(sha, "primary", max_concurrent=9, wave="ext1", single_array=True, assert_eligible_record=True)
    assert by_role.lines == pl.lines and by_role.cluster == M.PRIMARY_CLUSTER and by_role.summary() == pl.summary()
    assert f"# plan: cluster={M.PRIMARY_CLUSTER} (primary) qos={SITE_A['qos']}" in by_role.summary()[0]
    assert M.plan(sha, "overflow", max_concurrent=6, wave="ext1", rows="24-26,27-29").lines == c.lines
    assert M.plan(sha, "overflow", max_concurrent=6, wave="ext1", cluster_share="30-32").rows == list(range(24, 30))
    assert M.sbatch_script_path("overflow", sha) == M.sbatch_script_path(M.OVERFLOW_CLUSTER, sha)
    r = subprocess.run([sys.executable, str(M5T / "cells.py"), "plan", "--sha", sha, "--cluster", "overflow", "--wave", "ext1",
                        "--rows", "24-26,27-29", "--max-concurrent", "6"], capture_output=True, text=True)
    assert r.returncode == 0 and [l for l in r.stdout.splitlines() if l.startswith("sbatch ")] == c.lines and "(overflow)" in r.stdout
    # the prose names roles, never a cluster codename: the README and the matrix's CLI grammar
    readme = (M5T / "README.md").read_text()
    for name in M.CLUSTERS:
        assert name not in M.__doc__, name
        assert readme.count(name) == 0, name         # the codenames live at ONE definition site (cells.py) — not in the README


# --------------------------------------------------------------------------- #
# the generated argv, through the sbatch's OWN dry run; counts from the matrix #
# --------------------------------------------------------------------------- #



def _boundary_run(args: str, env_extra: dict) -> "subprocess.CompletedProcess[str]":
    """The REAL `m5t_row` + `m5t_assert_launchable` (one slice of m5t_common.sh) over the tree's cells.py, the way the
    sbatch's production path runs them: ARRAY from SLURM_ARRAY_TASK_ID, or the positional row."""
    import shlex
    funcs = _m5t_block("m5t_row(){", "m5t_resolve_config(){")
    script = ('set -u\nf22_log(){ echo LOG "$@"; }\nf22_fatal(){ echo FATAL "$@"; exit 1; }\n'
              f"m5t_py(){{ echo {shlex.quote(sys.executable)}; }}\nREPO_DIR={shlex.quote(str(REPO))}\nM5T_DIR=scripts/sdfbase/math500\n"
              + funcs + f"\nm5t_row {args}\nm5t_assert_launchable\necho REACHED_THE_OUT_ROOT\n")
    env = {k: v for k, v in os.environ.items() if k != "SLURM_ARRAY_TASK_ID"}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**env, **env_extra})


def test_the_batch_boundary_reasserts_eligibility_for_array_and_positional_rows():
    """Reproduced before the fix: `m5t_row ARRAY` at index 0 and the positional `gemma4e4b none thinking_card`
    both resolved the PUBLISHED row, kept status=launch, and the job went on to `m5t_out_root` under the published
    output root — published-ness is the WAVE (COMPLETED_WAVES), which only `cells.py plan` checked. The job now
    re-asserts the row's eligibility at the batch boundary with cells.py's OWN predicate (`assert_launchable` =
    `assert_eligible` over `eligible_rows`, the one `plan --assert-eligible` runs): a completed-wave row, a held row and
    a deferred row refuse by name BEFORE the out root exists; an extension row is admitted — through the array and the
    positional form alike. The dry run and the diagnostic launchers (published `none` rows into diagnostic roots) stay
    unrestricted."""
    # the predicate itself, and its record
    with pytest.raises(SystemExit, match="'published': this wave is COMPLETE"):
        M.assert_launchable(M.M5T_CELLS[0])
    n_steer, n_held = len(M.STEER_ROWS), sum(r["status"] == M.STATUS_HELD for r in M.STEER_ROWS)
    n_llama, n_llama_held = len(M.LLAMA_ROWS), sum(r["status"] == M.STATUS_HELD for r in M.LLAMA_ROWS)
    n_itdoc, n_itdoc_held = len(M.ITDOC_ROWS), sum(r["status"] == M.STATUS_HELD for r in M.ITDOC_ROWS)
    n_ap, n_ap_held = len(M.ALLPOS_ROWS), sum(r["status"] == M.STATUS_HELD for r in M.ALLPOS_ROWS)
    rec = M.assert_launchable(M.M5T_CELLS[24])
    steer_excluded = {"steer/other-wave": n_steer - n_held, **({"steer/held": n_held} if n_held else {}),
                      "llama/other-wave": n_llama - n_llama_held, **({"llama/held": n_llama_held} if n_llama_held else {}),
                      # E3: the it-doc rows (every one HELD today) are named by their status, never silently absent
                      **({"itdoc/other-wave": n_itdoc - n_itdoc_held} if n_itdoc - n_itdoc_held else {}), **({"itdoc/held": n_itdoc_held} if n_itdoc_held else {}),
                      # arditi-qwen-all-positions: its own wave, named here for the same reason
                      **({"allpos/other-wave": n_ap - n_ap_held} if n_ap - n_ap_held else {}), **({"allpos/held": n_ap_held} if n_ap_held else {}),
                      # the seed wave's 30 rows, named by their wave
                      "qwen_seedrep/other-wave": len(M.SEEDREP_ROWS)}
    assert rec == {"wave": "ext1", "rows": [24],
                   "excluded": {"published/complete": 24, "ext1/other-wave": 8, **steer_excluded}}
    assert M.launchable_line(M.M5T_CELLS[24], rec) == (
        "launchable: row 24 (m5t-gemma4e4b-consi) wave=ext1 status=launch; excluded published/complete: 24, "
        "ext1/other-wave: 8, " + ", ".join(f"{k}: {v}" for k, v in steer_excluded.items()))
    uni = [dict(r) for r in M.M5T_CELLS]
    uni[25]["status"], uni[26]["status"] = M.STATUS_HELD, M.STATUS_DEFERRED
    with pytest.raises(SystemExit, match="assert-eligible REFUSED: row 25 .*status=held"):
        M.assert_launchable(uni[25], uni)
    with pytest.raises(SystemExit, match="assert-eligible REFUSED: row 26 .*status=deferred"):
        M.assert_launchable(uni[26], uni)
    assert M.assert_launchable(uni[24], uni)["rows"] == [24]
    # the CLI the shell calls: exit 1 + the reason on stderr for the published row (by index AND by the positional triple),
    # exit 0 + the record for an extension row
    cli = [sys.executable, str(M5T / "cells.py"), "row"]
    for sel in (["--index", "0"], ["--key", "gemma4e4b", "--cell", "none", "--sampling", "thinking_card"], ["--index", "11"]):
        r = subprocess.run([*cli, *sel, "--assert-launchable"], capture_output=True, text=True)
        assert r.returncode == 1 and "this wave is COMPLETE" in r.stderr and r.stdout == "", (sel, r.stdout, r.stderr)
    r = subprocess.run([*cli, "--index", "24", "--assert-launchable"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("launchable: row 24 (m5t-gemma4e4b-consi) wave=ext1 status=launch"), r.stdout
    # the shell boundary over the REAL functions: ARRAY index 0 and the positional published triple refuse by name and never
    # reach the out root; ARRAY index 24 and a positional extension row are admitted and logged
    for args, env in (("ARRAY", {"SLURM_ARRAY_TASK_ID": "0"}), ("gemma4e4b none thinking_card", {}),
                      ("ARRAY", {"SLURM_ARRAY_TASK_ID": "23"}), (f"qwen35_4b {M.CELL_PROMPTED} general", {})):
        r = _boundary_run(args, env)
        assert r.returncode == 1, (args, r.stdout, r.stderr)
        assert "FATAL matrix row" in r.stdout and "is NOT launchable" in r.stdout and "this wave is COMPLETE" in r.stdout, r.stdout
        assert "wave=published status=launch" in r.stdout and "REACHED_THE_OUT_ROOT" not in r.stdout, r.stdout
    ext = M.M5T_CELLS[24]
    for args, env in (("ARRAY", {"SLURM_ARRAY_TASK_ID": "24"}), (f"{ext['model_key']} {ext['cell']} {ext['sampling_mode']}", {}),
                      ("ARRAY", {"SLURM_ARRAY_TASK_ID": "32"})):
        r = _boundary_run(args, env)
        assert r.returncode == 0, (args, r.stdout, r.stderr)
        assert "LOG launchable: row" in r.stdout and "wave=ext1 status=launch" in r.stdout and "REACHED_THE_OUT_ROOT" in r.stdout, r.stdout
    r = _boundary_run("ARRAY", {})                                     # no index: the row lookup itself refuses, before the predicate
    assert r.returncode == 1 and "SLURM_ARRAY_TASK_ID is unset" in r.stdout and "REACHED_THE_OUT_ROOT" not in r.stdout, r.stdout
    # the sbatch pins the call at the BOUNDARY: once, in the production path, between the row and the out root — never in the
    # dry-run block; the diagnostic launchers (published `none` rows into diagnostic roots) do not call it
    src = (M5T / "math500_thinking.sbatch").read_text()
    assert src.count("\nm5t_assert_launchable\n") == 1
    # math500-steer: m5t_init takes the row too (its serving path is selected inside, before f22_init fixes the path); the
    # row resolves on the very next line
    init_line = 'm5t_init "$WAVE_SHA" "${ROW_ARGS[@]}"'
    assert init_line in src
    boundary = 'm5t_row "${ROW_ARGS[@]}"\n'
    assert src[src.index(init_line):].split("\n", 1)[1].startswith(boundary)
    after_row = src[src.index(init_line):]
    after_row = after_row[after_row.index(boundary) + len(boundary):]
    assert after_row.index("m5t_assert_launchable") < after_row.index("m5t_resolve_config; m5t_out_root") < after_row.index("f22_serving_venv")
    assert src.index("trap cleanup EXIT") < src.index("\nm5t_assert_launchable\n")                  # the trap is armed before it
    dry = src[src.index('if [ "${M5T_DRY_RUN:-0}" = 1 ]; then\n  # stubs'):src.index("  exit 0\nfi\n")]
    assert 'm5t_row "${ROW_ARGS[@]}"' in dry and "m5t_assert_launchable" not in dry
    for name in ("smoke_thinking.sbatch", "nemo_backend_probe.sbatch", "dry_run_prod.sbatch"):
        assert "m5t_assert_launchable" not in (M5T / name).read_text(), name

#: arditi-qwen-all-positions: the rows whose CELL the serving vocabulary does not admit on this tree
#: (`cells.PENDING_SERVING_CELLS`). The kind-aware serving-path selector refuses their launch cells BY NAME — an
#: EXTERNAL gap: the Class B registration lands in another author's change, and the refusal is pinned VERBATIM by
#: `test_the_all_positions_row_is_gated_only_by_the_pending_serving_registration` below. So that everything ELSE about
#: such a row is still exercised here, its dry run is handed the serving path the kind-aware default would itself
#: select (`lora_native` — the arm's declared route), which makes `f22_select_serving_path` take its EXPLICIT branch
#: instead of calling the selector. Nothing else about the row is stubbed, and the moment the cell is registered this
#: bypass is dead: the same test asserts the set is exactly the cells the vocabulary refuses.
PENDING_SERVING_ROWS = {r["index"] for r in M.M5T_CELLS if r["cell"] in M.PENDING_SERVING_CELLS}


def _dry_lines(tmp_path, rows=None):
    """Every row's dry-run text through the sbatch itself, S resolved PER (model, geometry entry) — the constitution
    rows at the long-SI window (A4)."""
    from capctrl_serving_up import resolve_standup_geometry
    rows = M.M5T_CELLS if rows is None else rows
    s_by = {(k, ev): int(resolve_standup_geometry(ev, None, None, model_key=k).max_num_seqs)
            for k, ev in {(r["model_key"], r["geometry_eval"]) for r in rows}}
    out = []
    for row in rows:
        env = {**os.environ, "M5T_DRY_RUN": "1", "M5T_DRY_S": str(s_by[row["model_key"], row["geometry_eval"]]),
               "M5T_DRY_SHARED": str(tmp_path / "shared"), "M5T_DRY_CPY": "/venv/bin/python", "USER": "tester"}
        env.pop("M5T_LOG_REQUESTS_OVERRIDE", None)
        env.pop("F22_SERVING_PATH", None)
        env.pop("M5T_STORE_PINS", None)
        # no row needs an explicit serving path any more: the vocabulary admits every matrix cell, so the kind-aware
        # default selects the route for each row on its own (it did not, while the all-positions cell was unregistered —
        # PENDING_SERVING_CELLS, now empty). A bypass kept past its gap would hide the very selection this asserts.
        if row["cell"] == M.CELL_ARDITI_ALLPOS and _m5t_allpos_engine_entry() is None:
            # allpos-store-pins (2026-09-17): the checkout's store-pins record ADMITS this leg at its launch tree but its engine entry is
            # the stamp's to derive from the pulled store, and until it lands the launcher refuses the leg by name (never `no-store`).
            # The walk hands this ONE row the tests' pins seam over a native-stamped copy of the REAL record (the shape the stamp
            # writes) so every row's argv is still exercised through the launcher's own selection. SELF-CLEARING: the condition is
            # False the moment the stamp lands; the served-row test and the store-pins module's strict-xfail gate keep the pending
            # state visible meanwhile — this is not a hand-named gap.
            env["M5T_STORE_PINS"] = str(_m5t_allpos_stamped_copy(tmp_path / "allpos_stamped_pins.json"))
        r = subprocess.run(["bash", str(M5T / "math500_thinking.sbatch"), row["model_key"], row["cell"], row["sampling_mode"],
                            "0" * 40, "1" if row["fitted"] else "0"] + ([M.key_view(row)] if M.key_view(row) != M.VIEW_MAIN else []),
                           capture_output=True, text=True, env=env)   # the optional view positional names an it-view row — keyed the way
                                                                      # row_for keys (key_view): a steer row's `view` names the EXTERNAL lane's view, it keys as main
        assert r.returncode == 0, (row["job_id"], r.stdout[-500:], r.stderr[-500:])
        out.append((row, r.stdout))
    return out, s_by


def test_dry_run_argv_counts_are_the_matrix_s(tmp_path):
    """The dry-run greps — every expected count computed from cells.counts(), the lines from the sbatch itself."""
    runs, s_by = _dry_lines(tmp_path)
    text = "\n".join(t for _, t in runs)
    standup = [l for l in text.splitlines() if l.startswith("STANDUP:")]
    driver = [l for l in text.splitlines() if l.startswith("DRIVER:")]
    # E3: a HELD row (every it-doc row today) prints its HELD line and NO argv — the launchable rows are the argv count
    n = COUNTS["rows"] - COUNTS["statuses"].get(M.STATUS_HELD, 0)
    assert len(standup) == len(driver) == n
    assert sum(l.startswith("HELD:") for l in text.splitlines()) == COUNTS["statuses"].get(M.STATUS_HELD, 0)
    # the thinking ARM is the ROW's (`not_available` on a no-switch family's steering row), so it is counted per arm
    launchable = [r for r in M.M5T_CELLS if r["status"] != M.STATUS_HELD]
    for arm, k in {a: sum(r["thinking_mode"] == a for r in launchable) for a in {r["thinking_mode"] for r in launchable}}.items():
        assert sum(f"{M.FLAG_THINKING} {arm} " in l for l in driver) == k, arm
    for frag in (f"--controls {M.CONTROL} ", f"{M.FLAG_SAMPLES} {M.SAMPLES} ",
                 f"{M.FLAG_MATH500_RETRY_MAX_TOKENS} {M.RETRY_MAX_TOKENS} ", f"{M.FLAG_MAX_TOKENS_BY_TASK} {M.MAX_TOKENS_BY_TASK} ",
                 f"{M.FLAG_TINYCAP_IDENTITY_VERSION} {M.IDENTITY_VERSION} ", f"{M.FLAG_LEDGER_URL} ", f"{M.FLAG_OOS_PROMPT_BUDGET} "):
        assert sum(frag in l for l in driver) == n, frag
    assert sum(f"{M.FLAG_UPSTREAM_TIMEOUT_S} {M.UPSTREAM_TIMEOUT_S} " in l for l in standup) == n
    assert sum(f"{M.FLAG_LENS_SALT_POLICY} attested" in l for l in standup) == n
    assert text.count("@thinking") == 0 and text.count("petri_thinking_on") == 0
    # the fitted-only flags: exactly the fitted rows, and NEVER on an unfitted row. The driver-argv contract:
    # the driver-side PIN on every fitted row; the OOS override only under a tuple (every fitted row of that
    # extension has one); NO --bank-view on this tree (the driver has no such argument — the view rides the
    # row-identity stamp). The extension's per-row facts: --thinking from the row, the ROW's geometry x its S,
    # --enable-log-requests on the standup iff the row opted in, the recorder argv with --expect from the NUMERIC
    # projection only, the stamp.
    fitted_rows = {r["index"] for r in M.M5T_CELLS if r["fitted"]}
    for (row, t) in runs:
        if row["status"] == M.STATUS_HELD:            # E3: a held row printed HELD and no argv (asserted above)
            assert "HELD:" in t and "DRIVER:" not in t
            continue
        d = [l for l in t.splitlines() if l.startswith("DRIVER:")][0]
        s = [l for l in t.splitlines() if l.startswith("STANDUP:")][0]
        s_row = s_by[row["model_key"], row["geometry_eval"]]
        # the override rides a TUPLE, never fittedness: a no-switch family's fitted row is IN SUPPORT (the steering wave's llama rows)
        assert (f"{M.FLAG_ALLOW_OOS_THINKING}" in d) is bool(row["tuple_name"])
        assert (row["index"] in fitted_rows) is row["fitted"]
        assert (f"{M.FLAG_EXPECT_OBJECT_SHA256} {row['object_sha256']}" in d) is row["fitted"]
        assert (f"{M.FLAG_EXPECT_OBJECT_SHA256} {row['cell']}={row['object_sha256']}" in s) is row["fitted"]
        # the LOGPROB liveness gate rides EXACTLY the fitted rows serving an EXTERNAL lane's
        # object (the steer rows); every other row's driver argv is byte-identical (no flag)
        external = bool(row["fitted"]) and row.get("object_source") == M.OBJECT_SOURCE_EXTERNAL
        assert (f"{M.FLAG_LENS_LIVENESS} {M.LENS_LIVENESS_STEER}" in d) is external, (row["index"], d)
        assert d.count(M.FLAG_LENS_LIVENESS) == int(external)
        assert "--bank-view" not in d and "--bank-view" not in s
        # E2 / P7: under the NO-SWITCH arm the bounded setter is REQUIRED and IS the cap (the statement's value); a switch
        # family's row never names it (its ON table resolves the same cap; the setter only RAISES)
        assert (f"--on-cap-by-task {M.MAX_TOKENS_BY_TASK} " in d + " ") is (row["thinking_mode"] == M.THINKING_NOT_AVAILABLE)
        assert f"--cell {row['cell']} " in s + " " or s.endswith(f"--cell {row['cell']}")        # EVERY job names its own cell
        assert f"--cell {row['cell']} " in d + " "
        assert (f"{M.FLAG_REASONING_PARSER} gemma4" in s) is (row["model_key"] == "gemma4e4b")
        assert (f"{M.FLAG_ENABLE_LOG_REQUESTS} " in s) is bool(row["log_requests"]) is (row["wave"] != M.WAVE_PUBLISHED)
        assert f"{M.FLAG_SAMPLING} {row['sampling_mode']} " in d and f"{M.FLAG_THINKING} {row['thinking_mode']} " in d
        assert f"--concurrency {s_row} " in d
        assert f"GEOMETRY: {row['geometry_eval']} {row['geometry_L']} x {s_row} cap {M.CAP} retry {M.RETRY_MAX_TOKENS} thinking {row['thinking_mode']}" in t
        assert ("OUT OF SUPPORT: tuple=" + str(row["tuple_name"]) in t) is (row["fitted"] and bool(row["tuple_name"]))
        assert f"SAMPLING: mode={row['sampling_mode']} params=" in t and "SAMPLING_SOURCE: " in t
        out_root = f"{tmp_path / 'shared'}/exp/math500_thinking_on_full22/{row['out_root']}"
        assert f"--out-root {out_root} " in d and ("/controls_ext1/" in out_root) is (row["wave"] == M.WAVE_EXT1)
        assert f"WAVE: wave={row['wave']} status={row['status']} view={row['view']} budget_class={row['budget_class']} " in t
        assert f"log_requests={int(bool(row['log_requests']))}" in t
        # fix A4: the launch CONFIGURATION f22_standup composes from — mode attacks, no --thinking-on (server default OFF),
        # the request-level mode from the row, the row's memory utilisation — printed for the compatibility preflight
        assert (f"STANDUP_POLICY: mode=attacks thinking_on_flag=0 server_default_thinking=off request_thinking={row['thinking_mode']} "
                f"gpu_mem_util={row['gpu_mem_util']} max_logprobs=none") in t
        assert f"BUDGET: {tmp_path / 'shared'}/exp/math500_thinking_on_full22/budgets/{row['model_key']}/{row['budget_class']}/{M.OOS_PROMPT_BUDGET_FILENAME} " in t
        assert f"ROW_IDENTITY: {out_root}/{M.ROW_IDENTITY_FILENAME} (cells.py row --index {row['index']} --identity --sha {'0' * 40} --out-root {out_root})" in t
        rs = [l for l in t.splitlines() if l.startswith("RESOLVED_SAMPLER:")]
        assert len(rs) == 1
        if row["log_requests"]:
            words = rs[0].split()[1:]
            assert words[:2] == ["/venv/bin/python", str(REPO / M.RESOLVED_SAMPLER_SCRIPT_REL)]
            assert words[words.index("--out") + 1] == f"{out_root}/{M.RESOLVED_SAMPLER_FILENAME}"
            assert words[words.index("--client-cap") + 1] == str(M.CAP) and words.count("--client-cap") == 1
            # the per-kind mode: the probe class by the tag, no --probe-cap (the kind table is the recorder's
            # default). TWO tag sources, one per EMITTER: the DRIVER's evidence beside the store and the STANDUP's
            # load-check sidecar in the serving dir — the load check probes before the driver exists, so without
            # the second the whole render kind reads UNCLASSIFIED at 512 (34 measured there: 32 + 2). The ledger
            # is still never a source: its rows would double the SAME requests.
            assert words[words.index("--probe-expect") + 1] == M.RECORDER_PROBE_EXPECT == "kinds" and words.count("--probe-expect") == 1
            serve_dir = f"{tmp_path / 'shared'}/exp/math500_thinking_on_full22/serving/m5t_{row['model_key']}_{M.cell_slug(row['cell'])}_{row['sampling_mode']}_dry/math500"
            assert [words[i + 1] for i, w in enumerate(words) if w == "--probe-tags"] == [
                f"{out_root}/{M.REQUEST_EVIDENCE_FILENAME}", f"{serve_dir}/{M.RECORDER_LOAD_CHECK_TAGS_FILENAME}"]
            assert M.RECORDER_LOAD_CHECK_TAGS_FILENAME == "load_check.request_evidence.jsonl"
            assert "--probe-cap" not in words and M.LEDGER_FILENAME not in " ".join(words)
            expects = dict(words[i + 1].split("=", 1) for i, w in enumerate(words) if w == "--expect")
            assert expects == {k: str(v) for k, v in M.sampling_params(row["sampling"]).items()} and "source" not in expects
        else:
            assert rs[0] == "RESOLVED_SAMPLER: not recorded (log_requests=0)"
    # E3: the matrix-wide COUNTS include the six HELD it-doc rows, which print no argv — every argv count is the LAUNCHABLE rows'
    held = [r for r in M.M5T_CELLS if r["status"] == M.STATUS_HELD]
    assert sum(f"{M.FLAG_ALLOW_OOS_THINKING}" in l for l in driver) == sum(r["fitted"] and bool(r["tuple_name"]) for r in launchable)
    assert (sum(M.FLAG_EXPECT_OBJECT_SHA256 in l for l in standup) == sum(M.FLAG_EXPECT_OBJECT_SHA256 in l for l in driver)
            == sum(r["fitted"] for r in launchable) == COUNTS["fitted"] - sum(r["fitted"] for r in held))
    assert sum(f"{M.FLAG_REASONING_PARSER} gemma4" in l for l in standup) == COUNTS["reasoning_parser_gemma4"] - sum(r["model_key"] == "gemma4e4b" for r in held) == 8
    assert sum(f"{M.FLAG_ENABLE_LOG_REQUESTS} " in l for l in standup) == sum(bool(r["log_requests"]) for r in launchable) == 9 + len(M.STEER_ROWS) + len(M.LLAMA_ROWS) + len(M.ALLPOS_ROWS)
    for mode in COUNTS["sampling"]:
        assert sum(f"{M.FLAG_SAMPLING} {mode} " in l for l in driver) == sum(r["sampling_mode"] == mode for r in launchable)
    for cell, k in COUNTS["cells"].items():                       # the held it-view rows of a cell print no STANDUP line
        assert sum(f"--cell {cell} " in l + " " for l in standup) == k - sum(r["cell"] == cell for r in held), cell
    assert text.count("OUT OF SUPPORT: ") == COUNTS["fitted"] - COUNTS["fitted_in_support"] - sum(r["fitted"] for r in held)   # held rows print none
    pins = M.load_pins()
    for (key, sid), pin in pins.items():
        assert sum(pin in l for l in driver) == COUNTS["pins"].get(f"{key}/{sid}", 0), (key, sid)
    roots = {re.search(r"--out-root (\S+) ", l).group(1) for l in driver}
    assert len(roots) + len(held) == COUNTS["distinct_out_roots"] == 33 + len(M.STEER_ROWS) + len(M.LLAMA_ROWS) + len(M.ALLPOS_ROWS) + len(held)   # the held it rows print no argv
    assert all(("/controls_llama/" in r) is any(r.endswith(x["out_root"]) for x in M.LLAMA_ROWS) for r in roots)
    tuples = re.findall(r"OUT OF SUPPORT: tuple=(\S+)", text)
    # every fitted row prints its tuple EXCEPT the in-support ones (the no-switch steering rows: pin, no tuple)
    assert sorted(set(tuples)) == COUNTS["tuple_names"] and len(set(tuples)) == 14 + len(M.ALLPOS_ROWS)
    assert len(tuples) == COUNTS["fitted"] - COUNTS["fitted_in_support"] - sum(r["fitted"] for r in held)   # the held it rows print none
    # the no-switch family's fitted rows — the two steering rows and the llama wave's three LoRA rows — are IN SUPPORT
    assert text.count("IN SUPPORT: route=lens ") == COUNTS["fitted_in_support"] == 2 + sum(r["fitted"] for r in M.LLAMA_ROWS) == 5
    # the DRIVER env: Inspect pointed at the LEDGER, never the proxy
    assert sum(f"DRIVER_ENV: CUDA_VISIBLE_DEVICES= {M.INSPECT_BASE_URL_ENV}=" in l for l in text.splitlines()) == n
    # the opt-out (A11): M5T_LOG_REQUESTS_OVERRIDE=0 in the submitter's environment turns the flag and the recorder off
    row = M.EXTENSION_ROWS[0]
    env = {**os.environ, "M5T_DRY_RUN": "1", "M5T_DRY_S": "24", "M5T_DRY_SHARED": str(tmp_path / "shared"),
           "M5T_DRY_CPY": "/venv/bin/python", "USER": "tester", "M5T_LOG_REQUESTS_OVERRIDE": "0"}
    r = subprocess.run(["bash", str(M5T / "math500_thinking.sbatch"), row["model_key"], row["cell"], row["sampling_mode"], "0" * 40, "0"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and M.FLAG_ENABLE_LOG_REQUESTS not in r.stdout and "RESOLVED_SAMPLER: not recorded (log_requests=0)" in r.stdout
    assert "log_requests=0 (override; the row says 1)" in r.stdout
    env["M5T_LOG_REQUESTS_OVERRIDE"] = "maybe"
    r = subprocess.run(["bash", str(M5T / "math500_thinking.sbatch"), row["model_key"], row["cell"], row["sampling_mode"], "0" * 40, "0"],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "M5T_LOG_REQUESTS_OVERRIDE must be 0 or 1" in r.stdout + r.stderr


# --------------------------------------------------------------------------- #
# the overlay -> route -> _pick_cell chain through the ACTUAL standup argv      #
# --------------------------------------------------------------------------- #


def _stage_bank(tmp_path, key):
    """The object cells staged as the `main` view stages them under <bank_root>/<key>/cells_view/main: the DERIVED
    minus_ctl object (derived_meta.json) and the two BASE LoRA objects of the extension (a budget sidecar)."""
    from test_sdfbase_full22_wave import _stage, _stage_derived
    bank = tmp_path / key / "cells_view" / "main"
    _stage_derived(bank, M.CELL_OBJECT, M.CELL_OBJECT)
    for lora in (M.CELL_LORA_MIX, M.CELL_LORA_CTL2):
        _stage(bank, lora, lora)
    return bank


@pytest.mark.parametrize("key", M.MODEL_KEYS)
def test_overlay_route_and_pick_cell_through_the_generated_standup_argv(tmp_path, key):
    """For EVERY row of this model the standup's cell selection is run on the --cell value the
    sbatch's dry run ACTUALLY generated, the real enumeration + export_routing_table over a view-shaped bank yields
    the row's OWN route (the prompted route with the manifest's system-prefix hash), `_served_route_for` and
    `_pick_cell` resolve it, and the digest argument appears iff FITTED."""
    import dataclasses

    import cell_controls_run as ccr
    import unified_sweep_run as usr
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.petri.registry_adapter import traintest_cell_source
    from sisteer.capeval.serving.proxy import export_routing_table, load_routing_table
    from sisteer.capeval.tiers.petri_runner import resolve_prompt_files
    mt = _load("_m5t_route_mt", "scripts/sdfbase/mt_serving_up.py")
    runs, _ = _dry_lines(tmp_path / "dry", rows=[r for r in M.M5T_CELLS if r["model_key"] == key])    # this model's rows only
    bank = _stage_bank(tmp_path / "bank", key)
    cfg_path = REPO / f"configs/sdfbase/math500/sdfbase_full22_{key}.math500.yaml"
    man_path = REPO / f"configs/capeval/manifests/petri_full22_{key}.math500.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    manifest = yaml.safe_load(man_path.read_text())["petri"]
    spec0 = cm.load_petri_spec(man_path)
    for row, text in runs:
        if row["model_key"] != key:
            continue
        if row["status"] == M.STATUS_HELD:
            # E3: a HELD it-doc row prints its HELD line and NO argv — there is no standup to check until its record binds it
            assert "HELD:" in text and "STANDUP:" not in text
            continue
        if row.get("object_source") == "external":
            # the steering rows serve ANOTHER LANE's bank through their OWN overlay pair — the same chain over a
            # different config and a different bank; its end-to-end test is
            # test_the_w4_steering_rows_serve_their_own_overlay_and_external_bank_end_to_end below
            continue
        standup = [l for l in text.splitlines() if l.startswith("STANDUP:")][0].split()[1:]
        cells_arg = [standup[i + 1] for i, tok in enumerate(standup) if tok == "--cell"]
        assert cells_arg == [row["cell"]]
        digest_args = [standup[i + 1] for i, tok in enumerate(standup) if tok == "--expect-object-sha256"]
        assert (digest_args == [f"{row['cell']}={row['object_sha256']}"]) if row["fitted"] else (digest_args == [])
        tt_cells = mt.select_cells(cfg["explicit_cells"], cells_arg)
        harness = [ra.harness_cell_id(c) for c in tt_cells]
        spec = dataclasses.replace(spec0, serving_launch_flags=[f for f in spec0.serving_launch_flags if f != "--enable-lora"], cells=harness)
        src = traintest_cell_source(spec, repo_root=REPO, config_path=cfg_path, defenses_dir=bank)
        bf = {k: (REPO / "configs/capeval/petri" / k / "BEHAVIOR.md").read_bytes() for k in cm.petri_expected_behavior_keys(spec)}
        cells = dc.enumerate_cells(spec, src, behavior_files=bf, prompt_files=resolve_prompt_files(spec, override_root=REPO / "configs/capeval/petri/prompts"))
        assert not cells.absent, [(a.cell_id, a.reason) for a in cells.absent]
        table_path = tmp_path / f"{row['index']}_routing_table.json"
        export_routing_table(cells, src, table_path, base_model=cfg["model"], serving_mode="interleaved", base_revision=cfg["model_revision"])
        table = load_routing_table(table_path)
        routes = table["routes"]
        by_cell = {r.cell_id: r for r in routes}
        assert row["harness_cell"] in by_cell, (row["cell"], sorted(by_cell))
        route = usr._served_route_for(row["cell"], routes)        # the driver's own route selection, plain spelling
        assert route.cell_id == row["harness_cell"]
        if row["route_kind"] == "prompted":          # the prompted route's system prefix == the manifest's hash for ITS SI
            prefixes = {s.virtual_model: s for s in table["prompted_prefixes"]}
            harness_si = row["harness_cell"][len("prompted__"):]
            assert prefixes[f"petri__{row['harness_cell']}"].sha16 == manifest["prompted_prefix_hashes"][harness_si] == M.SI_SHA256_BY_CELL[row["cell"]][:16]
        if row["fitted"]:
            assert table["lens_served_positions"] == {row["harness_cell"]: ["all"]}   # Evals' stamp, fresh
        # the driver's cell pick over the OVERLAY config: the cell is listed, so _pick_cell accepts it
        args = types.SimpleNamespace(cell=row["cell"], defenses_dir=str(bank))
        cfg_for_pick = dict(cfg, defenses_dir=str(bank))
        picked = ccr._pick_cell(cfg_for_pick, args)
        assert picked.id == row["cell"]
    # a cell the overlay does not list is refused by name (the reason the overlays exist)
    with pytest.raises(SystemExit, match="unknown --cell"):
        ccr._pick_cell(dict(cfg, defenses_dir=str(bank)), types.SimpleNamespace(cell="constv2ctl2-full__psr_ce_all", defenses_dir=str(bank)))


# --------------------------------------------------------------------------- #
# the ledger sequencing                                                        #
# --------------------------------------------------------------------------- #


def _m5t_funcs():
    common = (F22 / "common.sh").read_text()
    m5t = (M5T / "m5t_common.sh").read_text()
    return ('set -u\nf22_log(){ echo LOG "$@"; }\nf22_fatal(){ echo FATAL "$@"; exit 1; }\n'
            + m5t[m5t.index("m5t_ledger_up(){"):m5t.index("m5t_driver_argv(){")])


def test_a_ledger_that_never_writes_its_endpoint_fails_the_job_and_the_driver_never_starts(tmp_path):
    """The ledger is started BEFORE the driver, waited on, and a ledger that never writes
    its endpoint file is `LEDGER_FAILED` — the driver is never started (the sbatch's `m5t_ledger_up || exit 1`
    precedes `m5t_driver_argv`). Run against the real m5t_common.sh functions with a stub client python."""
    stub = tmp_path / "python"
    stub.write_text('''#!/bin/bash
# the client venv python, stubbed: a free port, a ledger process, the endpoint reader
case "$*" in
  *"socket"*) echo 45678 ;;
  *"usage_ledger"*) if [ "${LEDGER_WRITES:-0}" = 1 ]; then for a in "$@"; do case "$prev" in --endpoint-json) ep="$a";; esac; prev="$a"; done
                     sleep 1; echo '{"ledger_url": "http://127.0.0.1:45678/v1", "upstream": "'"$UPSTREAM"'"}' > "$ep"; fi; sleep 30 ;;
  *"ledger_url"*) python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['ledger_url'])" "${@: -1}" ;;
  *) exit 0 ;;
esac
''')
    stub.chmod(0o755)
    serve = tmp_path / "serving/math500"; serve.mkdir(parents=True)
    (tmp_path / "out").mkdir()
    base = (_m5t_funcs() + f"\nCPY={stub}; REPO_DIR={REPO}; PROXY_URL=http://127.0.0.1:1/v1; UPSTREAM=$PROXY_URL; OUT_ROOT={tmp_path}/out; "
            f"SERVE_DIR={tmp_path}/serving; TABLE={serve}/routing_table.json; M5T_LEDGER_WAIT_S=3; M5T_LEDGER_FILENAME=usage_ledger.jsonl; "
            "M5T_LEDGER_ENDPOINT_FILENAME=ledger_endpoint.json; M5T_RESULT=INCOMPLETE; LEDGER_PID=''\n")
    # the ledger never writes its endpoint: LEDGER_FAILED, non-zero, nothing else runs after `|| exit 1`
    r = subprocess.run(["bash", "-c", base + 'm5t_ledger_up || { echo "RESULT: $M5T_RESULT"; m5t_ledger_down; exit 1; }; echo DRIVER_WOULD_START'],
                       capture_output=True, text=True, env={**os.environ, "LEDGER_WRITES": "0"}, timeout=60)
    assert r.returncode == 1 and "RESULT: LEDGER_FAILED" in r.stdout and "DRIVER_WOULD_START" not in r.stdout, (r.stdout, r.stderr)
    # the ledger writes it: LEDGER_URL is read back from the endpoint record beside the routing table
    r = subprocess.run(["bash", "-c", base + 'm5t_ledger_up || exit 1; echo "URL=$LEDGER_URL"; m5t_ledger_down; echo DRIVER_WOULD_START'],
                       capture_output=True, text=True, env={**os.environ, "LEDGER_WRITES": "1"}, timeout=60)
    assert r.returncode == 0 and "URL=http://127.0.0.1:45678/v1" in r.stdout and "DRIVER_WOULD_START" in r.stdout, (r.stdout, r.stderr)
    assert (serve / "ledger_endpoint.json").is_file()   # beside routing_table.json: where the driver binds it
    sb = (M5T / "math500_thinking.sbatch").read_text()
    assert sb.index("m5t_ledger_up || exit 1") < sb.index("m5t_driver_argv\n")


def test_standup_extra_names_every_jobs_own_cell_and_only_the_digest_follows_fitted():
    """Every job names its own cell; ONLY the digest follows FITTED; --enable-log-requests rides the array
    iff the EFFECTIVE opt-in is 1 (the row's value, or the override) — never hardcoded, never on a published row."""
    m5t = (M5T / "m5t_common.sh").read_text()
    # + common.sh's f22_standup_extra_reset (2026-09-16), which m5t_standup_extra calls — production sources
    # full22/common.sh before m5t_common.sh, so the harness carries the helper the same way
    common = (F22 / "common.sh").read_text()
    helper = common[common.index("f22_standup_extra_reset(){"):common.index("\n}\n", common.index("f22_standup_extra_reset(){")) + 3]
    funcs = "set -u\n" + helper + m5t[m5t.index("m5t_standup_extra(){"):m5t.index("m5t_check_endpoint(){")]
    for row in M.M5T_CELLS:
        for effective in (None, "0", "1"):
            script = (funcs + f"\nM5T_UPSTREAM_TIMEOUT_S={M.UPSTREAM_TIMEOUT_S}; M5T_LENS_SALT_POLICY=attested; CELL={row['cell']}; FITTED={int(row['fitted'])}; "
                      f"PIN='{row['object_sha256'] or ''}'; M5T_REASONING_PARSER='{row['reasoning_parser'] or ''}'\n"
                      + (f"M5T_LOG_REQUESTS_EFFECTIVE={effective}\n" if effective is not None else "")
                      + 'm5t_standup_extra; printf "%s\\n" "${F22_STANDUP_EXTRA[@]}"')
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            argv = r.stdout.splitlines()
            want = ["--upstream-timeout-s", "7200", "--lens-salt-policy", "attested"]
            if effective == "1":
                want += [M.FLAG_ENABLE_LOG_REQUESTS]
            want += ["--cell", row["cell"]]
            if row["fitted"]:
                # a HELD it-doc row without a verified pin composes an empty digest (the shell's PIN='') and never reaches the standup
                want += ["--expect-object-sha256", f"{row['cell']}={row['object_sha256'] or ''}"]
            if row["reasoning_parser"]:
                want += ["--reasoning-parser", row["reasoning_parser"]]
            assert argv == want, (row["job_id"], effective, argv)
    assert M.FLAG_ENABLE_LOG_REQUESTS not in (F22 / "common.sh").read_text().split("f22_standup(){", 1)[1].split("--gpu-mem-util", 1)[1].split("\n", 3)[0]
    # fix A4: the lane's SERVER-DEFAULT thinking policy is the standup's default (OFF; the request carries ON). f22_standup
    # composes --thinking-on iff F22_THINKING_ON is set and a batch process inherits the submitter's exported environment,
    # so the variable is REFUSED by name before any standup flag is composed — never read into the launch
    common = (F22 / "common.sh").read_text()
    assert "${F22_THINKING_ON:+--thinking-on}" in common and 'local want_thinking="off"; [ -n "${F22_THINKING_ON:-}" ] && want_thinking="on"' in common
    assert "F22_THINKING_ON" not in (M5T / "math500_thinking.sbatch").read_text()
    row = M.EXTENSION_ROWS[0]
    script = (funcs + 'f22_fatal(){ echo FATAL "$@"; exit 1; }\n'
              + f"M5T_UPSTREAM_TIMEOUT_S={M.UPSTREAM_TIMEOUT_S}; M5T_LENS_SALT_POLICY=attested; CELL={row['cell']}; FITTED=0; PIN=''; "
              "M5T_REASONING_PARSER=''; M5T_LOG_REQUESTS_EFFECTIVE=1\nF22_THINKING_ON=1\nm5t_standup_extra; echo REACHED")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 1 and "FATAL F22_THINKING_ON is set" in r.stdout and "REACHED" not in r.stdout, r.stdout
    # the standup MODE the sbatch passes is m5t_common.sh's one literal (the dry run's STANDUP_POLICY prints it)
    assert "M5T_STANDUP_MODE=attacks" in m5t and 'f22_standup attacks "$SERVE_DIR/math500"' in (M5T / "math500_thinking.sbatch").read_text()


# --------------------------------------------------------------------------- #
# the two-stage dry run: candidate vs certified facts; the stale-table hazard  #
# --------------------------------------------------------------------------- #


def _facts(status):
    return {fam: types.SimpleNamespace(split_mode=M.SPLIT_MODE_CANDIDATE[fam], evidence_status=status, off_ids=(1, 2), on_ids=(1, 3))
            for fam in set(M.FAMILY_BY_MODEL.values())}


def test_two_stage_dry_run_sequencing_with_candidate_vs_certified_facts(tmp_path, monkeypatch, capsys):
    """--stage pre PASSES on candidate facts and asserts the production preflight REFUSES them; --stage
    prod REFUSES candidate facts and passes certified ones (with a fresh table carrying the stamp)."""
    dry = _load("_m5t_dry_run_under_test", "scripts/sdfbase/math500/m5t_dry_run.py")
    only = ["--only", "qwen35_4b", M.CELL_OBJECT, "thinking_card", "--tmp", str(tmp_path / "t")]

    def refusing(family):
        raise ValueError(f"render fact for {family} is candidate, not certified")
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", _facts("candidate"), raising=False)
    monkeypatch.setattr(C, "check_render_fact_certified", refusing, raising=False)
    assert dry.main(["--stage", "pre", *only]) == 0
    out = capsys.readouterr().out
    assert "evidence_status=candidate" in out and "refuses=True" in out and "VERDICT: PASS stage=pre" in out
    assert out.count("G-LENS: ") == 1 and "refinement merged at" in out
    # prod against candidate facts: REFUSED by name
    assert dry.main(["--stage", "prod", *only]) == 2
    out = capsys.readouterr().out
    assert "REFUSED: render fact not certified for family qwen3_5" in out and "VERDICT: REFUSED stage=prod" in out
    # pre against candidate facts whose preflight ADMITS them: REFUSED (the gate would be circular)
    monkeypatch.setattr(C, "check_render_fact_certified", lambda family: None, raising=False)
    assert dry.main(["--stage", "pre", *only]) == 2
    assert "production preflight ADMITS" in capsys.readouterr().out
    # certified facts + a fresh table over a view-shaped bank: prod PASSES and reads the stamp
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", _facts("certified"), raising=False)
    bank_root = tmp_path / "bank"
    _stage_bank(bank_root, "qwen35_4b")
    assert dry.main(["--stage", "prod", *only, "--bank-root", str(bank_root)]) == 0
    out = capsys.readouterr().out
    assert f'SERVED_POSITIONS: row=5 lens_served_positions[{M.OBJECT_HARNESS_ID}] == "all"' in out and "VERDICT: PASS stage=prod" in out
    # prod WITHOUT a bank cannot check the fresh table: refused (never assumed)
    assert dry.main(["--stage", "prod", *only]) == 2
    assert "requires the fresh-table check" in capsys.readouterr().out


def test_the_stale_table_hazard_is_refused_at_stage_prod(tmp_path, monkeypatch, capsys):
    """A table written WITHOUT the served-positions stamp (a pre-refinement exporter) is refused by name at
    --stage prod — never read as 'the refinement misbehaves'."""
    dry = _load("_m5t_dry_run_stale", "scripts/sdfbase/math500/m5t_dry_run.py")
    from sisteer.capeval.serving import proxy as px
    real = px.export_routing_table

    def pre_refinement_export(*a, **kw):
        real(*a, **kw)
        path = Path(a[2])
        doc = json.loads(path.read_text())
        doc.pop("lens_served_positions", None)
        path.write_text(json.dumps(doc))
    monkeypatch.setattr(px, "export_routing_table", pre_refinement_export)
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", _facts("certified"), raising=False)
    bank_root = tmp_path / "bank"
    _stage_bank(bank_root, "nemotron3nano4b")
    only = ["--only", "nemotron3nano4b", M.CELL_OBJECT, "battery", "--tmp", str(tmp_path / "t"), "--bank-root", str(bank_root)]
    assert dry.main(["--stage", "prod", *only]) == 2
    out = capsys.readouterr().out
    assert "REFUSED: served-positions stamp absent — stale table or refinement not merged" in out
    # the same row on the real exporter passes at prod
    monkeypatch.setattr(px, "export_routing_table", real)
    assert dry.main(["--stage", "prod", *only]) == 0


def test_the_prod_dry_run_exports_an_external_rows_fresh_table_over_its_own_view_and_refuses_without_the_shared_root(tmp_path, monkeypatch, capsys):
    """A STEERING row serves its OWN overlay pair over ANOTHER lane's bank view, so the
    prod dry run's fresh-table export must read that pair and `<shared root>/<bank_view_rel[role]>` — the base overlay's six
    cells refused `--cell alphasteer` by name and the first cluster dry run of the steer wave died on it (job 495588). Over a
    synthetic external view at `--shared-root`: exported, the row's own route present, the served-positions stamp `all`,
    PASS; without `--shared-root` the row is NOT exported and prod REFUSES by name (never the base wave's shape in its
    place); a row whose view is staged on the other cluster only refuses by name too. The prod sbatch passes the shared root
    and derives the cluster role from the launcher's one overflow name."""
    dry = _load("_m5t_dry_run_external", "scripts/sdfbase/math500/m5t_dry_run.py")
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", _facts("certified"), raising=False)
    row = M.row_for("gemma4e4b", M.CELL_ALPHASTEER, M.SAMPLING_THINKING_CARD)
    assert row["object_source"] == M.OBJECT_SOURCE_EXTERNAL and row["index"] == 33
    view, obj = _stage_external_bank(tmp_path / "shared", row)
    bank_root = tmp_path / "shared" / "exp" / "objects" / "sdfbase" / "full22"      # the full22 root: unused by an external row, required by prod
    bank_root.mkdir(parents=True)
    only = ["--only", "gemma4e4b", M.CELL_ALPHASTEER, "thinking_card", "--tmp", str(tmp_path / "t"), "--bank-root", str(bank_root)]
    rc = dry.main(["--stage", "prod", *only, "--shared-root", str(tmp_path / "shared")])
    out = capsys.readouterr().out
    assert rc == 0 and "VERDICT: PASS stage=prod" in out, out[-1500:]
    fresh = [l for l in out.splitlines() if l.startswith("FRESH_TABLE: row=33")][0]
    assert "own_route=True" in fresh and f"config=configs/sdfbase/math500/sdfbase_full22_gemma4e4b.math500_{M.CELL_ALPHASTEER}.yaml" in fresh
    assert f"bank_view={view}" in fresh and "routes=['alphasteer', 'none', 'none__lens0']" in fresh      # + the lens-0 twin the exporter mints
    assert 'lens_served_positions={"alphasteer": ["all"]}' in fresh
    assert 'SERVED_POSITIONS: row=33 lens_served_positions[alphasteer] == "all"' in out
    # the same export through the function: the external branch, never the full22 main view
    for d in ("t2", "t3", "t4"):
        (tmp_path / d).mkdir()
    st = dry.fresh_table_stamp(row, bank_root, tmp_path / "t2", shared_root=tmp_path / "shared")
    assert st["exported"] is True and Path(st["bank_view"]) == view and st["stamp_for_cell"] == ["all"]
    # no --shared-root: not exported, prod refuses BY NAME (the base wave's overlay is never run in its place)
    rc = dry.main(["--stage", "prod", *only])
    out = capsys.readouterr().out
    assert rc == 2 and "VERDICT: REFUSED stage=prod" in out
    assert "REFUSED: --stage prod requires the fresh-table check (external row (alphasteer" in out and "needs --shared-root" in out
    # a view staged on the OTHER cluster only: refused by name for this role
    st = dry.fresh_table_stamp(dict(row, bank_view_rel={"primary": None, "overflow": row["bank_view_rel"]["overflow"]}), bank_root,
                               tmp_path / "t3", shared_root=tmp_path / "shared", cluster_role="primary")
    assert st["exported"] is False and "no bank view on the primary cluster" in st["why"]
    # the base-wave rows keep the full22 branch (the same call, no external key): the registry reads
    # <bank_root>/<key>/cells_view/main and refuses by name when nothing is staged there (the pre-existing behaviour)
    base = M.row_for("qwen35_4b", M.CELL_OBJECT, M.SAMPLING_THINKING_CARD)
    with pytest.raises(ValueError, match="explicit_cells"):
        dry.fresh_table_stamp(base, tmp_path / "nobank", tmp_path / "t4", shared_root=tmp_path / "shared")
    # the prod sbatch hands the tool the shared root and this cluster's role, derived from the launcher's one overflow name
    sb = (M5T / "dry_run_prod.sbatch").read_text()
    assert '--shared-root "$SHARED" --cluster-role "$CLUSTER_ROLE"' in sb
    assert 'CLUSTER_ROLE=primary; [ "${CLUSTER:-}" = "$M5T_OVERFLOW_CLUSTER" ] && CLUSTER_ROLE=overflow' in sb
    assert 'M5T_OVERFLOW_CLUSTER=' in (M5T / "m5t_common.sh").read_text()      # the one place the launcher names the overflow cluster


# ----------------------------------------------------------------------------- #
# the smoke probe's pure helpers                                                  #
# ----------------------------------------------------------------------------- #


def test_smoke_probe_helpers_seed_rule_expected_header_and_the_d11_shapes(monkeypatch):
    P = _load("_m5t_smoke_probe", "scripts/sdfbase/math500/smoke_thinking_probe.py")
    h = hashlib.sha256(b"math_00000001\x000").digest()
    assert P.item_seed("math_00000001", 0) == int.from_bytes(h[:4], "big") % (2 ** 31 - 1)
    assert P.DIAGNOSTIC is True
    # expected header: the override wins; else lane A's table; else None (the probe then FAILS, never passes vacuously)
    assert P.expected_header("qwen3_5", "on", [1, 2]) == [1, 2]
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", {"qwen3_5": types.SimpleNamespace(off_ids=(7,) * 7, on_ids=(5,) * 5)}, raising=False)
    assert P.expected_header("qwen3_5", "off", None) == [7] * 7 and P.expected_header("qwen3_5", "on", None) == [5] * 5
    assert P.expected_header("nemotron_h", "on", None) is None
    sc = P.shape_check
    # implicit_open (Qwen, Nemotron): a finished completion must CLOSE; a cap-bound one may not; a stray open tag is wrong
    assert sc("reasoning </think> 42", None, finish_reason="stop", mode="implicit_open", arm="on")["ok"]
    assert sc("only reasoning, truncated", None, finish_reason="length", mode="implicit_open", arm="on")["ok"]          # the truncated Nemotron row: legal shape, UNSCOREABLE
    assert not sc("only reasoning, finished", None, finish_reason="stop", mode="implicit_open", arm="on")["ok"]
    assert not sc("<think>stray</think> 42", None, finish_reason="stop", mode="implicit_open", arm="on")["ok"]
    # explicit_tags (GLM): tagged-and-closed, tag-free (= the answer) both legal; opened-not-closed on a finished row is not
    assert sc("<think>r</think> 42", None, finish_reason="stop", mode="explicit_tags", arm="on")["ok"]
    assert sc("42", None, finish_reason="stop", mode="explicit_tags", arm="on")["ok"]
    assert sc("<think>r", None, finish_reason="length", mode="explicit_tags", arm="on")["ok"]
    assert not sc("<think>r", None, finish_reason="stop", mode="explicit_tags", arm="on")["ok"]
    # record_only (gemma + parser): the parser owns the split — a LEAKED tag in content is the parser failing; a finished
    # row with neither tag nor reasoning is the model declining to think on that draw (counted, never a failure)
    assert sc("42", "thoughts", finish_reason="stop", mode="record_only", arm="on")["ok"]
    r = sc("42", None, finish_reason="stop", mode="record_only", arm="on")
    assert r["ok"] and r["no_thinking_emitted"] is True
    r = sc("42", "thoughts", finish_reason="stop", mode="record_only", arm="on")
    assert r["no_thinking_emitted"] is False
    assert sc("", "thoughts", finish_reason="length", mode="record_only", arm="on")["ok"]
    assert not sc("<think>x</think> 42", None, finish_reason="stop", mode="record_only", arm="on")["ok"]
    assert not sc("x</think> 42", "thoughts", finish_reason="stop", mode="record_only", arm="on")["ok"]
    assert P.MIN_ON_REASONING_SHARE == 0.5
    # OFF: no tag, no reasoning channel
    assert sc("42", None, finish_reason="stop", mode="implicit_open", arm="off")["ok"]
    assert not sc("<think>x</think> 42", None, finish_reason="stop", mode="implicit_open", arm="off")["ok"]
    assert not sc("42", "thoughts", finish_reason="stop", mode="record_only", arm="off")["ok"]


def test_nemo_probe_attest_admits_only_fused_and_at_least_half_the_peer_and_keys_the_venv(tmp_path):
    """Admission against a SAME-GPU peer's identical burst (duty and tok/s each >= the ratio x the peer's); an
    incomparable or missing reference never admits; the venv is keyed; a live RPC stays authoritative over the map."""
    P = _load("_m5t_nemo_probe", "scripts/sdfbase/math500/nemo_backend_probe.py")
    assert P.DUTY_RATIO == 0.5 and not hasattr(P, "DUTY_FLOOR")
    out = tmp_path / "probe"; out.mkdir(); ref_dir = tmp_path / "reference_qwen35_4b"; ref_dir.mkdir()
    reqs = [REPO / "configs/capeval/vllm-requirements.txt", REPO / "configs/capeval/vllm-requirements-nemotron.txt"]
    sha = hashlib.sha256(b"".join(p.read_bytes() for p in reqs)).hexdigest()
    (out / "packages.json").write_text(json.dumps({"ok": True, "mamba_ssm_version": "2.3.2.post1"}))
    (out / "dispatch.json").write_text(json.dumps({"static": {"ok": True, "symbols": []}, "live_worker": {"asked": True, "reports": None}}))
    burst = {"n_seqs": 64, "max_tokens": 512, "gpu_index": "5"}
    def duty(path, mean, tok, **over):
        path.write_text(json.dumps({"sm_duty_mean": mean, "sm_duty_min": 0.0, "tok_s": tok, **burst, **over}))
    duty(ref_dir / "duty.json", 0.30, 1800.0)                 # the peer, eager, launch-bound: the first probe's regime
    duty(out / "duty.json", 0.28, 1400.0)                     # Nemotron at 0.93x / 0.78x the peer
    att = tmp_path / "mamba_backend_attestation.json"
    ns = types.SimpleNamespace(out=str(out), attestation=str(att), requirements=[str(p) for p in reqs],
                               duty_reference=str(ref_dir / "duty.json"), duty_reference_key="qwen35_4b", duty_ratio=P.DUTY_RATIO)
    assert P.cmd_attest(ns) == 0
    doc = json.loads(att.read_text())
    assert doc["admitted"] is True and doc["fused_backend_attested"] is True and doc["serving_requirements_sha256"] == sha
    assert doc["duty_comparable"] is True and doc["duty_floor"] == 0.15 and doc["tok_s_floor"] == 900.0
    assert doc["duty_reference"]["model_key"] == "qwen35_4b" and doc["duty_reference"]["sm_duty_mean"] == 0.30
    assert "same-GPU peer" in doc["admission_rule"] and "static dispatch map" in doc["attestation_basis"]
    # below half the peer's duty, or below half its tok/s, refuses — each alone
    duty(out / "duty.json", 0.10, 1400.0); assert P.cmd_attest(ns) == 1 and json.loads(att.read_text())["duty_ok"] is False
    duty(out / "duty.json", 0.28, 800.0);  assert P.cmd_attest(ns) == 1 and json.loads(att.read_text())["tok_s_ok"] is False
    # an INCOMPARABLE reference (another burst shape, another GPU, missing) never admits, however good the numbers
    duty(out / "duty.json", 0.95, 9000.0)
    duty(ref_dir / "duty.json", 0.30, 1800.0, n_seqs=32);      assert P.cmd_attest(ns) == 1 and json.loads(att.read_text())["duty_comparable"] is False
    duty(ref_dir / "duty.json", 0.30, 1800.0, gpu_index="3");  assert P.cmd_attest(ns) == 1
    # a peer served at another width is another burst shape: not comparable (both records carry the served width)
    duty(ref_dir / "duty.json", 0.30, 1800.0, served={"max_num_seqs": 20}); duty(out / "duty.json", 0.95, 9000.0, served={"max_num_seqs": 44})
    assert P.cmd_attest(ns) == 1 and json.loads(att.read_text())["duty_comparable"] is False
    duty(ref_dir / "duty.json", 0.30, 1800.0, served={"max_num_seqs": 44}); assert P.cmd_attest(ns) == 0
    (ref_dir / "duty.json").unlink();                          assert P.cmd_attest(ns) == 1 and json.loads(att.read_text())["duty_floor"] is None
    duty(ref_dir / "duty.json", 0.30, 1800.0); duty(out / "duty.json", 0.28, 1400.0)
    # a non-fused symbol refuses
    (out / "dispatch.json").write_text(json.dumps({"static": {"ok": False, "symbols": []}, "live_worker": {"asked": False}}))
    assert P.cmd_attest(ns) == 1
    # a live RPC answer is authoritative over the static map
    (out / "dispatch.json").write_text(json.dumps({"static": {"ok": False}, "live_worker": {"asked": True, "reports": [{"fused_backend": True}]}}))
    assert P.cmd_attest(ns) == 0 and "live worker RPC" in json.loads(att.read_text())["attestation_basis"]


# ----------------------------------------------------------------------------- #
# the NON-dry startup: a fresh batch process, set -u, no REPO_DIR                 #
# ----------------------------------------------------------------------------- #

STARTUP_SHA = "c" * 40
STARTUP_USER = "m5t-startup-nobody"      # a user with NO shared root on this machine: the shared-root lookup refuses by name
STARTUP_FORMS = [
    ("math500_thinking.sbatch", ["gemma4e4b", "none", "thinking_card", STARTUP_SHA, "0"], {}),
    ("math500_thinking.sbatch", ["ARRAY", STARTUP_SHA], {"SLURM_ARRAY_TASK_ID": "3"}),
    ("smoke_thinking.sbatch", ["gemma4e4b", STARTUP_SHA], {}),
    ("nemo_backend_probe.sbatch", [STARTUP_SHA], {}),
    ("dry_run_prod.sbatch", [STARTUP_SHA], {}),
]


def _fake_checkout(tmp_path):
    """A checkout-shaped tree holding the REAL helpers (what the prologue must find and source) and the src/sisteer
    marker f22_init looks for — nothing else: the startup must refuse by name before it needs anything more."""
    co = tmp_path / "checkout"
    for rel in ("scripts/sdfbase/full22/common.sh", "scripts/sdfbase/math500/m5t_common.sh", "scripts/sdfbase/math500/cells.py"):
        (co / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, co / rel)
    (co / "src/sisteer").mkdir(parents=True)
    return co


def _startup_env(tmp_path, name, extra, **overrides):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "USER": STARTUP_USER, "SLURM_JOB_ID": "424242",
           "SLURM_JOB_NAME": f"m5t-startup-{name}", "CUDA_VISIBLE_DEVICES": "0", **extra, **overrides}
    assert "REPO_DIR" not in env          # the defect's precondition: a fresh Slurm batch process exports no REPO_DIR
    return env


def _run_startup(tmp_path, name, args, env):
    r = subprocess.run(["bash", "-u", str(M5T / name), *args], env=env, capture_output=True, text=True, timeout=60, cwd=tmp_path)
    return r, r.stdout + r.stderr


@pytest.mark.parametrize("name,args,extra", STARTUP_FORMS)
def test_non_dry_startup_under_set_u_with_no_repo_dir_reaches_the_first_step_and_refuses_by_name(tmp_path, name, args, extra):
    """Reproduced: exit 127 `REPO_DIR: unbound variable` at the second `source`. The launcher's NON-dry
    body, under `bash -u`, started the way Slurm starts it — no REPO_DIR, a user with no shared root, the checkout
    named through M5T_REPO_DIR (the shared-root derivation is the same block; its own refusal is pinned below). It
    must get PAST both sources into the first real step, m5t_init, whose shared-root refusal names the paths it
    tried — never `unbound variable` — and the cleanup trap must already be armed there: its RESULT record fires."""
    env = _startup_env(tmp_path, name, extra, M5T_REPO_DIR=str(_fake_checkout(tmp_path)))
    r, out = _run_startup(tmp_path, name, args, env)
    assert "unbound variable" not in out, out
    assert r.returncode == 1, (r.returncode, out)
    assert "FATAL: set SISTEER_SHARED_ROOT, the shared root of the cluster" in out, out   # the job's site value; no storage path is probed
    assert "RESULT" in out and "rc=1" in out and "result=INCOMPLETE" in out, out       # the trap ran: armed before m5t_init


@pytest.mark.parametrize("name,args,extra", STARTUP_FORMS)
def test_startup_refuses_a_missing_shared_root_or_checkout_by_name_before_sourcing_anything(tmp_path, name, args, extra):
    # (a) no override and no checkout root: the derivation refuses by the variable's name; nothing sourced, so no trap record
    r, out = _run_startup(tmp_path, name, args, _startup_env(tmp_path, name, extra))
    assert r.returncode == 1 and "unbound variable" not in out, out
    assert "set SISTEER_EVAL_REPO_ROOT" in r.stderr and "nothing to source" in r.stderr, out
    assert "RESULT" not in out
    # (b) a checkout without the helpers: refused with the path, what it lacks, and the remedy (README's worktree line)
    empty = tmp_path / "empty"
    empty.mkdir()
    r, out = _run_startup(tmp_path, name, args, _startup_env(tmp_path, name, extra, M5T_REPO_DIR=str(empty)))
    assert r.returncode == 1 and "unbound variable" not in out, out
    assert f"FATAL: no checkout at {empty}" in r.stderr and "worktree add --detach" in r.stderr and STARTUP_SHA in r.stderr, out
    assert "RESULT" not in out


def test_the_checkout_prologue_is_identical_across_launchers_and_the_trap_precedes_every_failing_step():
    """The launchers cannot share the prologue through a sourced file (it is what locates the files), so the
    copies are pinned byte-identical; the trap is armed right after the sources, before m5t_init — the first step
    that can fail — with nothing that can fail in between except the workstation dry-run branch (which exits 0)."""
    blocks = {}
    for name in SBATCHES:
        src = (M5T / name).read_text()
        m = re.search(r"(# --- the checkout, located BEFORE either helper is sourced.*?\n"
                      r"source \"\$REPO_DIR/scripts/sdfbase/full22/common\.sh\"; source \"\$REPO_DIR/scripts/sdfbase/math500/m5t_common\.sh\"\n)",
                      src, re.DOTALL)
        assert m, name
        blocks[name] = m.group(1)
        assert 'source "/workspace' not in src            # no root-guessing source chain remains (it hid the unbound REPO_DIR)
        for line in src.splitlines():
            if "$0" in line:
                assert line.lstrip().startswith("#"), line    # $0 is Slurm's spool copy: named in the comment, never used
            if "BASH_SOURCE" in line and not line.lstrip().startswith("#"):
                assert "M5T_DRY_RUN" in line, line        # the workstation dry run alone may locate itself
        i_src, i_trap, i_init = m.end(), src.index("trap cleanup EXIT SIGTERM SIGINT SIGQUIT"), src.index('m5t_init "$WAVE_SHA"')
        assert i_src < i_trap < i_init, (name, i_src, i_trap, i_init)
        between = re.sub(r'if \[ "\$\{M5T_DRY_RUN:-0\}" = 1 \]; then.*?\nfi\n', "", src[i_src:i_trap], flags=re.DOTALL)
        between = re.sub(r"cleanup\(\)\{.*?\n\}\n", "", between, flags=re.DOTALL)                  # a definition, not a call
        between = "\n".join(l for l in between.splitlines() if not l.lstrip().startswith("#"))
        assert not re.search(r"\b(m5t|f22)_[a-z_]+", between), (name, between)
        # the trap can fire at ANY step: every row fact the record reads has a default, every path it touches is guarded
        cleanup = src[src.index("cleanup(){"):i_trap]
        assert not re.search(r"\$(M5T_INDEX|M5T_JOB_ID|KEY|CELL|SAMPLING_MODE|FITTED|M5T_FAMILY)\b", cleanup), name
        for var in ("FIGS_DIR", "OUT_ROOT", "SMOKE_ROOT", "ATTESTATION"):
            if var in cleanup:
                assert "${" + var + ":-}" in cleanup, (name, var)
        assert "${FIGS_DIR:-}" in cleanup and "f22_result" in cleanup and "RESULT (before the wave root resolved" in cleanup, name
        # a deliberate STOP records itself as such — the armed trap must not file it as an INCOMPLETE failure
        assert re.search(r"f22_stop_requested && \{[^}]*M5T_RESULT=STOP_REQUESTED[^}]*exit 0; \}", src), name
    assert len(set(blocks.values())) == 1, blocks


# --------------------------------------------------------------------------- #
# the plan: ONE aggregate allowance across the time classes; disjoint shares    #
# --------------------------------------------------------------------------- #


def _covered(lines):
    covered = []
    for l in lines:
        spec = re.search(r"--array=([0-9,\-]+)%", l).group(1)
        for part in spec.split(","):
            a, _, b = part.partition("-")
            covered += list(range(int(a), int(b or a) + 1))
    return covered


def _widths(lines):
    return [int(re.search(r"%(\d+) ", l).group(1)) for l in lines]


def test_max_concurrent_is_an_aggregate_allowance_partitioned_across_the_time_classes():
    """`%W` on EACH of three independent arrays (16/6/2 rows) let --max-concurrent 12 run 20 at once
    and 4 run 10. The allowance is AGGREGATE now: the per-array widths sum to min(W, rows) <= W, every class present
    keeps >= 1, no width exceeds its class's row count, and the partition minimises the wave's tail under Slurm's
    STATIC per-array throttles (a slot never migrates between arrays) — checked against every partition by an
    independent brute force; the longest class first on ties."""
    sha = "ab" * 20
    base = M.BASE_ROWS                # the published wave's shape: the partition pins below are ITS (the python universe entry)
    sizes = {tl: len(ix) for tl, ix in M.time_classes(base).items()}
    hours = {tl: M.time_limit_hours(tl) for tl in sizes}
    # re-cost: the 10,752-window classes 3 / 6 / 8 h x (17,664 / 10,752) = 1.643 -> 4.93 / 9.86 / 13.1, to the hour
    assert hours == {M.TIME_BASE: 5.0, M.TIME_OBJECT: 10.0, M.TIME_OBJECT_GLM: 13.0}
    assert [round(h * M.GEOMETRY_L / 10752) for h in (3, 6, 8)] == [5, 10, 13] and M.GEOMETRY_L == 17664
    assert M.time_limit_hours(M.TIME_EXT) == 6.0      # the extension's ONE class: ~2.6x the longest estimate (2.3 h)

    def tail(ws):
        return max(-(-sizes[tl] // ws[tl]) * hours[tl] for tl in sizes)
    for W in (3, 4, 5, 8, 11, 12, 14, 16, 24, 40):
        pl = M.plan(sha, "cluster_a", max_concurrent=W, universe=base)
        assert sum(_widths(pl.lines)) == sum(pl.widths.values()) == min(W, 24) <= W, (W, pl.lines)
        assert all(1 <= pl.widths[tl] <= sizes[tl] for tl in sizes), (W, pl.widths)
        best = min(tail(dict(zip(sizes, ws))) for ws in itertools.product(*(range(1, sizes[tl] + 1) for tl in sizes)) if sum(ws) <= W)
        assert pl.est_tail_h == tail(pl.widths) == best, (W, pl.widths)
        for l in pl.lines:       # each line's width is ITS class's width
            tl = re.search(r"--time=(\S+)", l).group(1)
            assert f"%{pl.widths[tl]} " in l, (W, l)
    # the planned wave (<= 12 free): GLM object 2, object 6, base 4 = 12 in flight, never 20; 4 -> 2/1/1, never 10
    assert M.plan(sha, "cluster_a", max_concurrent=12, universe=base).widths == {M.TIME_OBJECT_GLM: 2, M.TIME_OBJECT: 6, M.TIME_BASE: 4}
    assert M.plan(sha, "cluster_a", max_concurrent=4, universe=base).widths == {M.TIME_OBJECT_GLM: 1, M.TIME_OBJECT: 1, M.TIME_BASE: 2}
    assert list(M.plan(sha, "cluster_a", max_concurrent=12, universe=base).widths) == [M.TIME_OBJECT_GLM, M.TIME_OBJECT, M.TIME_BASE]   # longest first
    # below the class count: refused by name (never a %0 array); a non-positive allowance likewise
    with pytest.raises(SystemExit, match="time classes"):
        M.plan(sha, "cluster_a", max_concurrent=2, universe=base)
    with pytest.raises(SystemExit, match="positive"):
        M.plan(sha, "cluster_a", max_concurrent=0, universe=base)
    with pytest.raises(SystemExit, match="positive"):
        M.plan(sha, "cluster_a", max_concurrent=0, wave="ext1", single_array=True)
    # the printer names every width and their sum, and the shared-budget rule (peers' jobs count)
    summary = "\n".join(M.plan(sha, "cluster_a", max_concurrent=12, universe=base).summary())
    assert "object-glm 2/2" in summary and "object 6/6" in summary and "base 4/16" in summary and "sum 12" in summary, summary
    assert "allowance=12" in summary and "peers" in summary and "16" in summary and "wave=ALL" in summary, summary
    # the whole matrix under the per-class form: FOUR classes, the extension's rows in theirs, still the optimum
    sizes4 = {tl: len(ix) for tl, ix in M.time_classes(M.M5T_CELLS).items()}
    assert sizes4 == {**sizes, M.TIME_EXT: 9 + len(M.STEER_ROWS) + len(M.ITDOC_ROWS) + len(M.ALLPOS_ROWS),
                      M.TIME_LLAMA: len(M.LLAMA_ROWS)}   # the extension's rows, the steering rung, the it-doc rows and the all-positions row share one class; the llama wave has its own
    pl4 = M.plan(sha, "cluster_a", max_concurrent=16, universe=M.M5T_CELLS)
    assert sum(pl4.widths.values()) == 16 and set(pl4.widths) == set(sizes4)
    # a matrix-wide plan (no --wave) carries every LAUNCH row: the held it-doc rows (E3) are never emitted
    assert sorted(_covered(pl4.lines)) == [r["index"] for r in M.M5T_CELLS if r["status"] == M.STATUS_LAUNCH]
    assert "for " not in "\n".join(pl4.lines + pl4.summary())          # arrays, never a loop
    # the LAUNCHER's universe (the default) carries the seed wave, whose rows are OVERFLOW-constrained —
    # a wave-less plan on the primary cluster refuses them BY NAME; the same universe on the overflow cluster plans them in their own class
    with pytest.raises(SystemExit, match="cluster_constraint overflow"):
        M.plan(sha, "cluster_a", max_concurrent=16)
    assert set(M.time_classes()) == set(sizes4) | {M.TIME_SEEDREP} and len(M.time_classes()[M.TIME_SEEDREP]) == len(M.SEEDREP_ROWS)


def test_rows_and_cluster_share_give_disjoint_subsets_so_no_row_runs_on_both_clusters():
    """ONE --cluster-share string on both invocations = the primary cluster's rows and the
    overflow cluster's complement — disjoint and covering the matrix BY CONSTRUCTION; --rows is the explicit form. Specs are Slurm's
    index grammar; a duplicate, an out-of-range index, an empty share, or both flags at once refuse by name."""
    sha = "ab" * 20
    base = M.BASE_ROWS                   # the published wave's shape (the python universe entry point)
    # the first wave re-read under the placement constraint: the nemotron rows 21-23 are primary-only, so
    # the primary cluster's share is 0-14 + 21-23 (18 rows) and the overflow cluster's complement the 6 rows 15-20 (qwen battery + glm battery)
    share = "0-14,21-23"
    b = M.plan(sha, "cluster_a", max_concurrent=12, cluster_share=share, universe=base)
    c = M.plan(sha, "cluster_b", max_concurrent=6, cluster_share=share, universe=base)
    assert set(b.rows).isdisjoint(c.rows) and sorted(set(b.rows) | set(c.rows)) == list(range(24))
    assert sorted(_covered(b.lines)) == b.rows == list(range(15)) + [21, 22, 23] and sorted(_covered(c.lines)) == c.rows == list(range(15, 21))
    assert all(f"--qos={SITE_B['qos']} " in l and f"--output={SITE_B['logs']}/" in l and f"{M.CHECKOUT_ROOT}/{sha}/" in l for l in c.lines)
    assert all(f"--qos={SITE_A['qos']} " in l and f"--output={SITE_A['logs']}/" in l for l in b.lines)
    # the widths respect each cluster's OWN allowance and the class sizes WITHIN its share
    assert sum(b.widths.values()) == 12 and sum(c.widths.values()) == 6
    c_sizes = {tl: len(ix) for tl, ix in M.time_classes(M.select_rows(cluster_share=share, cluster="cluster_b", universe=base)).items()}
    assert c_sizes == {M.TIME_BASE: 4, M.TIME_OBJECT: 1, M.TIME_OBJECT_GLM: 1} and c.widths == c_sizes    # 6 rows, 6 slots: all at once
    # a nemotron row on the overflow cluster is refused BY NAME (the fused-kernel overlay is attested on the primary cluster)
    with pytest.raises(SystemExit, match="cluster_constraint primary"):
        M.plan(sha, "cluster_b", max_concurrent=9, cluster_share="0-14", universe=base)
    # --rows: the explicit form (a class absent from the selection prints no array)
    e = M.plan(sha, "cluster_b", max_concurrent=3, rows="15-17", universe=base)
    assert e.rows == [15, 16, 17] and len(e.lines) == 2 and e.widths == {M.TIME_OBJECT: 1, M.TIME_BASE: 2}
    with pytest.raises(SystemExit, match="twice"):
        M.parse_index_spec("0-3,2")
    with pytest.raises(SystemExit, match="outside"):
        M.parse_index_spec(str(len(M.LAUNCHER_ROWS)))            # the launcher's universe (the matrix + the seed wave)
    assert M.parse_index_spec(str(len(M.M5T_CELLS))) == [len(M.M5T_CELLS)]      # ... so the first seed row's index IS an index
    with pytest.raises(SystemExit, match="outside"):
        M.parse_index_spec("24", n=24)
    with pytest.raises(SystemExit, match="not an index"):
        M.parse_index_spec("0-x")
    with pytest.raises(SystemExit, match="exclusive"):
        M.plan(sha, "cluster_a", max_concurrent=4, rows="0-3", cluster_share="0-3", universe=base)
    with pytest.raises(SystemExit, match="no rows"):
        M.plan(sha, "cluster_b", max_concurrent=4, cluster_share="0-23", universe=base)
    # the summary: the selection, the widths and their sum are named beside the lines
    summary = "\n".join(c.summary())
    assert "rows=6/24" in summary and "15-20" in summary and "sum 6" in summary and SITE_B["qos"] in summary, summary


# --------------------------------------------------------------------------- #
# the smoke probe's requests: bounded concurrency, ONE deterministic dump #
# --------------------------------------------------------------------------- #


class _FakeLedger:
    """A ledger-shaped ``POST /v1/chat/completions`` that RECORDS: every request as it arrived (body + headers), how
    many were in flight together (the overlap the concurrency claim rests on), and the order it answered in. Each
    answer is a well-formed implicit_open completion under the arm's stamp, after a delay that FALLS with the item's
    index — so with several in flight the first wave answers in the REVERSE of the plan order, and a dump that came
    out sorted did so by construction, not by luck."""

    def __init__(self, step_s=0.03):
        self.lock = threading.Lock()
        self.inflight = self.max_inflight = 0
        self.seen, self.answered = [], []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                item = self.headers.get(C.HEADER_SAMPLE) or ""
                on = body[C.CHAT_TEMPLATE_KWARGS_KEY][C.ENABLE_THINKING_KEY]
                with outer.lock:
                    outer.inflight += 1
                    outer.max_inflight = max(outer.max_inflight, outer.inflight)
                    outer.seen.append({"path": self.path, "item": item, "task": self.headers.get(C.HEADER_TASK),
                                       "auth": self.headers.get("Authorization"), "body": body})
                time.sleep(step_s * (8 - int(item.rsplit("_", 1)[1]) % 8))      # item 0 answers LAST of its wave
                resp = {"id": "fake", "choices": [{"index": 0, "finish_reason": "stop",
                                                   "message": {"role": "assistant", "reasoning_content": None,
                                                               "content": "reasoning </think> 42" if on else "42"}}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                        "petri": {"render_regime": "thinking_on" if on else "thinking_off",
                                  "cache_salt_regime": "cell-stable", "resumed_prefill_policy": "not-consulted"}}
                with outer.lock:
                    outer.inflight -= 1
                    outer.answered.append((item, "on" if on else "off"))
                out = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def _smoke_plan(n_on, n_off):
    """The smoke's plan shape: ON over the first n_on ids in order, then OFF over the first n_off — the OFF ids REPEAT
    the first ON ids, which is why (item id) alone is not a total order and the dump is keyed (item id, arm)."""
    ids = [f"math_{i:08d}" for i in range(max(n_on, n_off))]
    msgs = [[{"role": "user", "content": f"Problem {i}."}] for i in range(len(ids))]
    return [(ids[i], msgs[i], True) for i in range(n_on)] + [(ids[i], msgs[i], False) for i in range(n_off)]


def _run_probe(P, srv, plan, concurrency, tmp_path, tag):
    params = {"temperature": 0.6, "top_p": 0.95}
    rows = P.run_completions(srv.url, plan, vmodel="petri__base", cap=1536, params=params, token="tok", mode="implicit_open",
                             model_key="qwen35_4b", family="qwen3_5", sampling_mode="thinking_card", concurrency=concurrency,
                             timeout_s=30)
    path = tmp_path / f"smoke_completions_{tag}.jsonl"
    P.dump_completions(path, rows)
    return rows, [json.loads(l) for l in path.read_text().splitlines()], params


def test_smoke_probe_sends_completions_with_bounded_concurrency_and_dumps_in_one_deterministic_order(tmp_path):
    """Serially, 28 cap-bound draws under a 7,200 s per-request timeout did not reliably fit the smoke's wall.
    The completions now go out with at most --concurrency in flight (a fake ledger COUNTS the overlap), every row
    and every request keep the serial probe's fields, and the dump is in (item id, arm) order whatever order the
    server answered in — here the REVERSE of the plan order within a wave, by the fake's construction."""
    P = _load("_m5t_smoke_probe_conc", "scripts/sdfbase/math500/smoke_thinking_probe.py")
    plan = _smoke_plan(8, 4)             # 12 requests; the 4 OFF ids repeat the first 4 ON ids, as the smoke's do
    srv = _FakeLedger()
    try:
        rows, dumped, params = _run_probe(P, srv, plan, 4, tmp_path, "c4")
        assert 2 <= srv.max_inflight <= 4, srv.max_inflight        # concurrent, and bounded by the knob
        assert len(srv.seen) == len(rows) == len(dumped) == 12
        key = [(r["item_id"], r["arm"]) for r in dumped]
        assert srv.answered != key                                 # the server's order is not the dump's
        assert key == sorted(key) and len(set(key)) == 12 and rows == dumped
        assert key == sorted((item, "on" if on else "off") for item, _, on in plan)
        # the requests, as SENT: the arm as an explicit boolean, the cap, the seed rule, the sampling row, the headers
        by_item = {(s["item"], s["body"][C.CHAT_TEMPLATE_KWARGS_KEY][C.ENABLE_THINKING_KEY]): s for s in srv.seen}
        for item, msgs, on in plan:
            s = by_item[(item, on)]
            assert s["path"] == "/v1/chat/completions" and s["task"] == "smoke/math" and s["auth"] == "Bearer tok"
            b = s["body"]
            assert b["model"] == "petri__base" and b["messages"] == msgs and b["max_tokens"] == 1536
            assert b["seed"] == P.item_seed(item, 0) and b["temperature"] == 0.6 and b["top_p"] == 0.95
            assert b[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: on}
            assert isinstance(b[C.CHAT_TEMPLATE_KWARGS_KEY][C.ENABLE_THINKING_KEY], bool)
        # the rows, as RECORDED: the serial probe's fields, each carrying the value the serial probe put there
        serial_fields = {"diagnostic", "item_id", "arm", "model_key", "family", "split_mode_candidate", "cap", "seed",
                         "sampling_mode", "sampling_params", "finish_reason", "usage", "stamps", "content",
                         "reasoning_content", "latency_s", "error", "shape"}
        for r in dumped:
            assert serial_fields <= set(r) and {"t_start_s", "t_end_s"} <= set(r), sorted(r)
            assert r["diagnostic"] is True and r["model_key"] == "qwen35_4b" and r["family"] == "qwen3_5"
            assert r["split_mode_candidate"] == "implicit_open" and r["cap"] == 1536 and r["seed"] == P.item_seed(r["item_id"], 0)
            assert r["sampling_mode"] == "thinking_card" and r["sampling_params"] == params and r["error"] is None
            assert r["finish_reason"] == "stop" and r["usage"]["completion_tokens"] == 7
            assert r["stamps"]["render_regime"] == ("thinking_on" if r["arm"] == "on" else "thinking_off")
            assert r["stamps"]["cache_salt_regime"] == "cell-stable"
            assert r["content"] == ("reasoning </think> 42" if r["arm"] == "on" else "42") and r["reasoning_content"] is None
            assert r["shape"]["ok"] is True and r["shape"]["arm"] == r["arm"]
            assert 0 <= r["t_start_s"] <= r["t_end_s"] and abs((r["t_end_s"] - r["t_start_s"]) - r["latency_s"]) < 0.05
        # the overlap is legible from the dump itself: some row started before another had ended
        iv = sorted((r["t_start_s"], r["t_end_s"]) for r in dumped)
        assert any(iv[i + 1][0] < iv[i][1] for i in range(len(iv) - 1)), iv
        # the same plan under another width: the same dump order (only the server's answer order varies)
        _, dumped2, _ = _run_probe(P, srv, plan, 3, tmp_path, "c3")
        assert [(r["item_id"], r["arm"]) for r in dumped2] == key
    finally:
        srv.close()
    # concurrency 1 IS the serial probe: never two in flight
    srv1 = _FakeLedger()
    try:
        _run_probe(P, srv1, _smoke_plan(3, 1), 1, tmp_path, "c1")
        assert srv1.max_inflight == 1 and len(srv1.seen) == 4
    finally:
        srv1.close()
    with pytest.raises(ValueError, match="concurrency"):
        P.run_completions("http://127.0.0.1:1/v1", plan, vmodel="v", cap=1, params={}, token=None, mode="implicit_open",
                          model_key="k", family="qwen3_5", sampling_mode="m", concurrency=0, timeout_s=1)


def test_smoke_sbatch_time_is_the_header_s_worst_case_formula_at_x2_and_covers_the_standup_ceiling():
    """The wall is sized from the REALISTIC per-request time at the diagnostic cap under the probe's
    concurrency, x2 — never from the 7,200 s timeout (which no wall covers) and never x5. Every input is read from
    the launcher's own line: the counts from its positional defaults, the concurrency from the probe's default."""
    src = (M5T / "smoke_thinking.sbatch").read_text()
    P = _load("_m5t_smoke_probe_time", "scripts/sdfbase/math500/smoke_thinking_probe.py")
    m = re.search(r"--time = (\d+) x \(standup (\d+) min \+ (\d+) tokenize probes x (\d+) min \+ "
                  r"ceil\((\d+) completions / (\d+) in flight\) x (\d+) min \+ teardown (\d+) min\) = (\d+) x (\d+) min = (\d+) min", src)
    assert m, "the --time formula line is missing from the smoke sbatch header"
    pad, standup, n_probe, t_probe, n_req, conc, t_req, teardown, pad2, subtotal, total = map(int, m.groups())
    assert pad == pad2 == 2                                                    # x2, not x5
    assert subtotal == standup + n_probe * t_probe + math.ceil(n_req / conc) * t_req + teardown
    assert total == pad * subtotal
    assert f"#SBATCH --time=0-{total // 60:02d}:{total % 60:02d}:00" in src and "#SBATCH --time=0-00:30:00" not in src
    n_on = int(re.search(r'N_ON="\$\{3:-(\d+)\}"', src).group(1))
    n_off = int(re.search(r'N_OFF="\$\{4:-(\d+)\}"', src).group(1))
    conc_default = int(re.search(r'SMOKE_CONCURRENCY="\$\{6:-(\d+)\}"', src).group(1))
    assert n_req == n_on + n_off == 28 and conc == conc_default == P.DEFAULT_CONCURRENCY == 8
    assert n_probe == 2 * 2            # two arms x (add_generation_prompt true, false) through /v1/tokenize
    assert '--concurrency "$SMOKE_CONCURRENCY"' in src and "[concurrency=8]" in src
    # f22_standup's own endpoint ceiling fits inside the wall: a standup that never comes up is STANDUP_FAILED, not TIMEOUT
    common = (F22 / "common.sh").read_text()
    w = re.search(r"for i in \$\(seq 1 (\d+)\); do\n\s*\[ -f \"\$out/endpoint\.json\" \] && break\n.*?\n\s*sleep (\d+)\n", common, re.DOTALL)
    assert w, "f22_standup's endpoint wait loop moved"
    assert total * 60 > int(w.group(1)) * int(w.group(2)) == 2400


# --------------------------------------------------------------------------- #
# the eight-arm extension's launcher seams (math-eight-arms-extension §4.1, P6; decisions-impl-laneA A2, A3, A11):     #
# the budget fit re-check at launch, the sampler recorder end to end, the row-identity stamp                           #
# --------------------------------------------------------------------------- #


def _m5t_block(start: str, end: str) -> str:
    m5t = (M5T / "m5t_common.sh").read_text()
    return m5t[m5t.index(start):m5t.index(end)]


def _fit_sidecar(row, *, sha, **over):
    side = {"task": "math", "control": "math500", "cap": M.CAP, "window": row["geometry_L"], "geometry_eval": row["geometry_eval"],
            "fits": True, "why_not": None, "prompt_max": 6981 if row["cell"] == M.CELL_PROMPTED_CONST else 1041, "total": 0,
            "margin": 100, "p99": 0, "n": 500, "arm": row["thinking_mode"], "model": "x", "revision": "y",
            "chat_template_hash": "ab" * 8, "tree_sha": sha,
            "system_prompt_cell": row["cell"] if row["route_kind"] == "prompted" else None,
            "system_prompt_sha16": (row["si_sha256"] or "")[:16] or None}
    side.update(over)
    return side


def _run_budget_check(tmp_path, row, side, sha):
    """m5t_budget_artifact with the core's copy stubbed: the artifact + its sidecar under a wave root, the row's shell facts."""
    import shlex
    root = tmp_path / "wave"
    src = root / "budgets" / row["model_key"] / row["budget_class"] / M.OOS_PROMPT_BUDGET_FILENAME
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(json.dumps({"chat_template_hash": "ab" * 8}))
    if side is not None:
        Path(str(src) + ".fit.json").write_text(json.dumps(side))
    out_root = tmp_path / "out"
    out_root.mkdir(exist_ok=True)
    funcs = _m5t_block("m5t_budget_artifact(){", "m5t_standup_extra(){")
    script = ('set -u\nf22_log(){ echo LOG "$@"; }\nf22_fatal(){ echo FATAL "$@"; exit 1; }\n'
              'm5t_core(){ :; }\nst_budget_artifact(){ cp "$1" "$2"; BUDGET_TEMPLATE_HASH=x; }\n'
              f"m5t_py(){{ echo {shlex.quote(sys.executable)}; }}\n" + funcs
              + f"\nBUDGET_SRC={shlex.quote(str(src))}; OUT_ROOT={shlex.quote(str(out_root))}; M5T_BUDGET_FILENAME={M.OOS_PROMPT_BUDGET_FILENAME}\n"
              f"KEY={row['model_key']}; CELL={row['cell']}; M5T_BUDGET_CLASS={row['budget_class']}; M5T_CAP={M.CAP}; "
              f"M5T_GEOMETRY_EVAL={row['geometry_eval']}; M5T_GEOMETRY_L={row['geometry_L']}; WAVE_SHA={sha}; "
              f"M5T_THINKING_MODE={row['thinking_mode']}; M5T_ROUTE_KIND={row['route_kind']}; M5T_SI_SHA256={shlex.quote(row['si_sha256'] or '')}\n"
              "m5t_budget_artifact\n")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_the_budget_fit_sidecar_is_re_checked_at_launch_against_the_rows_facts(tmp_path):
    """§4.1 (the port of the GSM8K launcher's check, closing the gap plan F4 names): the artifact's .fit.json must say
    fits, at the row's cap, the row's entry AND window, the row's arm, the WAVE sha (a budget measured at another tree
    refuses — re-stamp it at the wave sha), and for a prompted row the row's SI cell + digest; a missing sidecar refuses
    by name; an object row's budget is the bare class's, so a sidecar naming an SI refuses."""
    sha = "ef" * 20
    consi = M.row_for("gemma4e4b", M.CELL_PROMPTED_CONST, M.SAMPLING_THINKING_CARD)
    lora = M.row_for("qwen35_4b", M.CELL_LORA_MIX, M.SAMPLING_THINKING_CARD)
    for row in (consi, lora, M.BASE_ROWS[0], M.row_for("nemotron3nano4b", M.CELL_PROMPTED, M.SAMPLING_THINKING_CARD)):
        r = _run_budget_check(tmp_path / row["job_id"], row, _fit_sidecar(row, sha=sha), sha)
        assert r.returncode == 0 and "fit OK" in r.stdout, (row["job_id"], r.stdout, r.stderr)
        assert (tmp_path / row["job_id"] / "out" / M.OOS_PROMPT_BUDGET_FILENAME).is_file()
    for over, why in ((dict(fits=False, why_not="too long"), "fits=False"), (dict(tree_sha="0" * 40), "tree_sha"),
                      (dict(geometry_eval=M.GEOMETRY_EVAL), "geometry_eval"), (dict(window=17664), "window"),
                      (dict(cap=8192), "cap"), (dict(arm="off"), "arm"), (dict(system_prompt_cell=M.CELL_PROMPTED), "system_prompt_cell"),
                      (dict(system_prompt_sha16="ac3a58f61d53cf24"), "system_prompt_sha16")):
        r = _run_budget_check(tmp_path / f"bad-{why}", consi, _fit_sidecar(consi, sha=sha, **over), sha)
        assert r.returncode != 0 and why in r.stdout and "does not admit this row" in r.stdout, (why, r.stdout)
    r = _run_budget_check(tmp_path / "nosidecar", consi, None, sha)
    assert r.returncode != 0 and "no fit sidecar" in r.stdout
    r = _run_budget_check(tmp_path / "lora-si", lora, _fit_sidecar(lora, sha=sha, system_prompt_cell=M.CELL_PROMPTED_CONST), sha)
    assert r.returncode != 0 and "system_prompt_cell" in r.stdout
    # the dry run names the re-check on every row (the BUDGET: line), with the row's entry / window / arm / tree / SI
    runs, _ = _dry_lines(tmp_path / "dry", rows=[consi, lora])
    for row, t in runs:
        si = row["cell"] if row["route_kind"] == "prompted" else "none"
        assert f"(fit sidecar re-checked at launch: fits, cap {M.CAP}, {row['geometry_eval']} / {row['geometry_L']}, arm on, tree {'0' * 40}, si {si})" in t


def test_the_recorder_argv_the_launcher_prints_agrees_on_a_log_at_the_rows_values_and_diverges_on_a_shifted_one(tmp_path):
    """Under the per-kind rule: the RESOLVED_SAMPLER argv the dry run prints, run FOR
    REAL over a synthetic vLLM log + the driver's request evidence beside the store — client draws at the row's values, ONE
    render probe (512) decoding under the row's card, ten greedy bites (48: the cell + its twin) and two greedy lens-ON bites
    (256), every probe attempt tagged `probe/<kind>` with its cap — reads AGREES, per kind: render AGREES against the card
    row (--expect), bites and lens_on_bite AGREES against greedy, nothing UNCLASSIFIED; a shifted client top_p is DIVERGED;
    the tag file ABSENT leaves every probe UNCLASSIFIED — UNCLASSIFIED_REQUESTS, a finding, never a shape-inferred AGREES;
    the same argv without --expect is REFUSED (rc 2): under `kinds` the render kind's reference is the pinned card row the
    launcher passes, never the driver's declaration; `--expect source=...` is REFUSED (rc 2) before any record."""
    from test_record_resolved_sampler import _ATTEST_INFO, _GREEDY_REF, _SP, _STARTED, _tag_row
    row = M.row_for("qwen35_4b", M.CELL_LORA_MIX, M.SAMPLING_THINKING_CARD)
    assert "source" in row["sampling"] and "source" not in M.sampling_params(row["sampling"])
    runs, _ = _dry_lines(tmp_path / "dry", rows=[row])
    words = [l for l in runs[0][1].splitlines() if l.startswith("RESOLVED_SAMPLER:")][0].split()[1:]
    words[0] = sys.executable
    params = M.sampling_params(row["sampling"])
    card = dict(t=params["temperature"], tp=params["top_p"], tk=params["top_k"], pp=params["presence_penalty"])
    greedy = dict(t=0.0, tp=1.0, tk=0, pp=0.0)
    # the leg's probe shape (the published ledgers, A3): ONE card-row render probe at 512, ten greedy bites at 48 (five on
    # the cell + five on its none__lens0 twin), two greedy lens-ON bites at 256 (a thinking-ON object cell)
    probes = [("render", 512, card)] + [("bites", 48, greedy)] * 10 + [("lens_on_bite", 256, greedy)] * 2

    def log(**over):
        p = {**card, **over}
        lines = [_STARTED, _ATTEST_INFO]
        for i in range(5):                                        # client draws at the row's cap
            lines.append(_SP.format(i=i, cap=M.CAP, seed=100 + i, **p))
        for j, (_kind, cap, tup) in enumerate(probes):            # the scaffolding, each at its kind's cap and reference
            lines.append(_SP.format(i=100 + j, cap=cap, seed=0, **tup))
        return "\n".join(lines) + "\n"

    def tags():                                                   # the driver's request evidence: one probe/<kind> row per attempt
        return "".join(json.dumps(_tag_row(kind, cap, vmodel=(M.CELL_LORA_MIX if j % 2 == 0 else "none__lens0"), n=j)) + "\n"
                       for j, (kind, cap, _) in enumerate(probes))

    def run(text, argv, write_tags=True):
        lg = Path(argv[argv.index("--vllm-log") + 1])
        lg.parent.mkdir(parents=True, exist_ok=True)
        lg.write_text(text)
        out = Path(argv[argv.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        # TWO tag sources, one per emitter: the driver's evidence beside the store and the standup's
        # load-check sidecar. This probe set is the DRIVER's, so its tags go in the first; the second is named and
        # absent, which the recorder records as such rather than failing.
        tag_files = [Path(argv[i + 1]) for i, w in enumerate(argv) if w == "--probe-tags"]
        assert len(tag_files) == argv.count("--probe-tags") == 2
        tf, load_check = tag_files
        assert tf.parent == out.parent and tf.name == M.REQUEST_EVIDENCE_FILENAME
        assert load_check.name == M.RECORDER_LOAD_CHECK_TAGS_FILENAME and load_check.parent != out.parent
        tf.unlink(missing_ok=True)
        if write_tags:
            tf.write_text(tags())
        r = subprocess.run(argv, capture_output=True, text=True, cwd=REPO,
                           env={**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"})
        return r, (json.loads(out.read_text()) if out.exists() else None)

    r, rec = run(log(), words)
    assert r.returncode == 0 and rec["verdict"] == "AGREES", (r.stdout[-800:], r.stderr[-800:], rec)
    assert rec["classes"]["client"]["verdict"] == "AGREES" and rec["classes"]["client"]["n"] == 5
    p = rec["classes"]["probe"]
    assert p["reference_kind"] == "kinds" and p["n"] == len(probes) == 13 and p["n_unclassified"] == 0 and p["n_unclassified_tags"] == 0
    assert p["n_by_kind"] == {"render": 1, "bites": 10, "lens_on_bite": 2, "logprob_bites": 0, "load_check": 0, "positive_control": 0, "standup_controls": 0}
    assert {k: v["verdict"] for k, v in p["kinds"].items()} == {"render": "AGREES", "bites": "AGREES", "lens_on_bite": "AGREES",
                                                                 "logprob_bites": "EMPTY", "load_check": "EMPTY", "positive_control": "EMPTY", "standup_controls": "EMPTY"}
    assert p["kinds"]["bites"]["reference"] == p["kinds"]["lens_on_bite"]["reference"] == _GREEDY_REF
    assert all(float(v) == p["kinds"]["render"]["reference"][k] for k, v in params.items())      # the card row = --expect
    assert p["kinds"]["bites"]["tag_check"]["agrees"] is True and p["kinds"]["bites"]["tag_check"]["tagged"] == {"48": 10}
    # the STANDUP's load-check source is named and absent in this fixture (these probes are the DRIVER's): the
    # recorder records it as missing rather than failing, which is the behaviour a real leg relies on when the load
    # check wrote no sidecar — and the load_check kind is EMPTY above, not UNCLASSIFIED
    missing = rec["probe_tags"]["missing"]
    assert [Path(m).name for m in missing] == [M.RECORDER_LOAD_CHECK_TAGS_FILENAME]
    assert rec["probe_caps"] == [str(c) for c in M.RECORDER_PROBE_CAPS]
    r, rec = run(log(tp=0.8), words)                              # a shifted CLIENT top_p; the probes keep their kinds' references
    assert r.returncode == 0 and rec["verdict"] == "DIVERGED", rec["why"]
    r, rec = run(log(), words, write_tags=False)                  # no tag file at all: a driver that stamps no tag — the untagged,
    # undeclared remainder is NOT_INSTRUMENTED (descriptive, compared against nothing, exit 0; the client class unaffected),
    # never classified by shape (recorder-load-check-kinds; the orchestrator's zero-tag ruling)
    assert r.returncode == 0 and rec["verdict"] == "PROBE_NOT_INSTRUMENTED", rec["why"]
    assert rec["classes"]["probe"]["verdict"] == "NOT_INSTRUMENTED" and rec["classes"]["probe"]["remainder"]["n"] == 13
    assert rec["classes"]["probe"]["n_unclassified"] == 0 and rec["classes"]["client"]["verdict"] == "AGREES"
    no_expect = [w for i, w in enumerate(words) if w != "--expect" and (i == 0 or words[i - 1] != "--expect")]
    r, rec = run(log(), no_expect)
    assert r.returncode == 2 and rec is None and "needs at least one --expect" in r.stderr
    r, rec = run(log(), words + ["--expect", "source=prose"])
    assert r.returncode == 2 and rec is None


def _producer_sampling_block(row: dict) -> dict:
    """`manifest["sampling"]` as the DRIVER writes it (served_capability_run._sampling_block): the mode, the model key,
    the hashed identity of the CONTRACTS row, its source prose — never the matrix dict."""
    crow = C.sampling_for(row["model_key"], row["sampling_mode"])
    return {"mode": row["sampling_mode"], "model_key": row["model_key"],
            **C.sampling_identity(crow, {C.inspect_task(M.INSPECT_TASK): M.SAMPLES}), "source": crow.source}


SYNTHETIC_LOCK_SHA = "5e" * 32          # a synthetic client-lock digest the synthetic driver stores record (never a real lock's)


def _producer_store(root: Path, row: dict, sha: str, *, block, gs_ref, item_refs, manifest_over=None, gen_stats=True, outcomes=True) -> Path:
    """A driver store in the PRODUCER's shape (verified against the published pull, decisions-fix-launcher FA1): the
    manifest carries the `sampling` BLOCK and NO `sampling_ref` field; the digest rides gen_stats (per task) and every
    outcome item."""
    d = root / "cells" / "x__controls__t0"
    d.mkdir(parents=True, exist_ok=True)
    # the driver's frameworks block carries the digest of the client lock it ran under: a synthetic
    # 64-hex here — the producer tree is synthetic too, so the stamp's tree cross-check reads `agree: None` (unreadable
    # tree), recorded and not a problem; the real-tree cases are tested below
    man = {"cell_id": row["cell"], "route": {"object_artifact_sha256": row["object_sha256"]}, "thinking": {"mode": row["thinking_mode"]},
           "producer": {"tree_sha": sha}, "frameworks": {"client_lock_sha256": SYNTHETIC_LOCK_SHA}}
    if block is not None:
        man["sampling"] = block
    man.update(manifest_over or {})
    (d / "manifest.json").write_text(json.dumps(man))
    if gen_stats:
        (d / "gen_stats.json").write_text(json.dumps({"inspect/math": {"n": len(item_refs), "sampling_ref": gs_ref}}))
    if outcomes:
        (d / "outcomes").mkdir(exist_ok=True)
        items = [{"item_id": f"math_{i:08x}", "sampling_ref": r, "scored": True} for i, r in enumerate(item_refs)]
        (d / "outcomes" / "inspect__math.json").write_text(json.dumps({"task": "inspect/math", "items": items, "n_attempted_items": len(items)}))
    return d


def test_the_row_identity_stamp_is_the_canonical_superset_with_the_pin_and_derives_sampling_ref_from_the_drivers_store(tmp_path):
    """The launcher's stamp is IFEval's row-identity superset + the per-row `pin` (= the producer
    tree), written by cells.py from the matrix row with `sampling_ref` DERIVED from the driver's own manifest sampling
    block through contracts.sampling_ref and held to gen_stats + every outcome item (the driver writes NO manifest-level
    `sampling_ref`: reading one gave `null` on every published leg). No manifest, no sampling evidence, or a
    disagreement = `complete: false`, `sampling_ref: null`, the reasons named — never a success-shaped stamp. The shell
    writer runs the same CLI, writes the file atomically and LOGS the verdict."""
    import shlex
    sha = "cd" * 20
    row = M.row_for("gemma4e4b", M.CELL_LORA_CTL2, M.SAMPLING_THINKING_CARD)
    ident = M.row_identity(row, producer_tree_sha=sha, sampling_ref="abc123def456", client_lock_sha256=SYNTHETIC_LOCK_SHA)
    assert tuple(ident) == M.ROW_IDENTITY_KEYS
    assert ident == {"object_artifact_sha256": row["object_sha256"], "route_kind": "lens", "view": "main", "producer_tree_sha": sha,
                     "thinking": "on", "sampling_mode": "thinking_card", "sampling_ref": "abc123def456", "pin": sha,
                     "client_lock_sha256": SYNTHETIC_LOCK_SHA}
    assert M.row_identity(row, producer_tree_sha=sha)["client_lock_sha256"] is None       # never the checkout's digest alone
    with pytest.raises(SystemExit, match="40-hex"):
        M.row_identity(row, producer_tree_sha=sha[:7])
    # no manifest: incomplete, null ref, null lock, the reason
    rec = M.row_identity_record(row, producer_tree_sha=sha, out_root=str(tmp_path / "empty"))
    assert rec["schema"] == M.ROW_IDENTITY_SCHEMA == "m5t_row_identity/3" and rec["complete"] is False and rec["identity"]["sampling_ref"] is None
    assert rec["identity"]["client_lock_sha256"] is None
    assert rec["checks"]["manifest"] is None and "no cells/*/manifest.json" in rec["checks"]["why"] and rec["problems"] == [rec["checks"]["why"]]
    assert rec["row"]["index"] == row["index"] and rec["row"]["tuple_name"] == row["tuple_name"] and rec["row"]["out_root"] == row["out_root"]
    # the PRODUCER's store: the ref derives from the block and agrees with gen_stats + the items -> complete
    block = _producer_sampling_block(row)
    ref = C.sampling_ref(block)
    assert len(ref) == 12 and "source" in block and "sampling_ref" not in block
    out_root = tmp_path / "out"
    _producer_store(out_root, row, sha, block=block, gs_ref=ref, item_refs=(ref, ref, ref))
    rec = M.row_identity_record(row, producer_tree_sha=sha, out_root=str(out_root))
    assert rec["complete"] is True and rec["problems"] == [] and rec["identity"]["sampling_ref"] == ref
    assert rec["identity"]["pin"] == sha == rec["identity"]["producer_tree_sha"]
    # the lock the leg RAN UNDER rides the stamp from the driver's manifest; the producer tree is synthetic
    # here, so the tree cross-check is `agree: None` with the reason (recorded, not a problem), the checkout's own lock
    # digest beside it for the reader
    lock_ev = rec["checks"]["client_lock_evidence"]
    assert rec["identity"]["client_lock_sha256"] == SYNTHETIC_LOCK_SHA == lock_ev["manifest_frameworks_client_lock_sha256"]
    assert lock_ev["agree"] is None and lock_ev["tree_lock_sha256"] is None and "git show" in lock_ev["why"] and lock_ev["tree"] == sha
    assert lock_ev["checkout_lock_sha256"] == hashlib.sha256((REPO / C.CLIENT_LOCK_TXT).read_bytes()).hexdigest()
    ev = rec["checks"]["sampling_ref_evidence"]
    assert ev["agree"] is True and ev["derived_from_manifest_block"] == ref and ev["gen_stats"] == {"inspect/math": ref}
    assert ev["outcomes"] == {"inspect__math.json": {"n_items": 3, "refs": [ref]}} and "contracts.sampling_ref" in ev["how"]
    assert {k: v for k, v in rec["checks"].items() if k.endswith("_agrees")} == {
        "object_artifact_sha256_agrees": True, "thinking_mode_agrees": True, "producer_tree_agrees": True, "cell_agrees": True}
    assert rec["checks"]["manifest"] == "cells/x__controls__t0/manifest.json" and rec["checks"]["n_manifests"] == 1
    # the ref is never READ off a manifest field: an invented `sampling_ref` field is ignored in favour of the derivation
    _producer_store(tmp_path / "field", row, sha, block=block, gs_ref=ref, item_refs=(ref,), manifest_over={"sampling_ref": "0123456789ab"})
    assert M.row_identity_record(row, producer_tree_sha=sha, out_root=str(tmp_path / "field"))["identity"]["sampling_ref"] == ref
    # no success-shaped stamp without agreeing evidence — each gap named
    cases = {
        "noblock": (dict(block=None), "carries no `sampling` block"),
        "nogs": (dict(gen_stats=False), "no gen_stats.json"),
        "nooc": (dict(outcomes=False), "no outcomes/*.json"),
        "gsref": (dict(gs_ref="000000000000"), f"gen_stats.json[inspect/math].sampling_ref 000000000000 != the derived {ref}"),
        "itemref": (dict(item_refs=(ref, "000000000000")), "item refs ['000000000000', '" + ref + "'] != the derived"),
        "cell": (dict(manifest_over={"cell_id": "none"}), "cell_agrees: the driver's manifest disagrees with the row"),
        "tree": (dict(manifest_over={"producer": {"tree_sha": "0" * 40}}), "producer_tree_agrees: the driver's manifest disagrees"),
        # a driver that recorded no lock digest is named (every steering-wave driver writes one)
        "nolock": (dict(manifest_over={"frameworks": {"inspect_ai": "0.3.260"}}), "client_lock_sha256: the driver's manifest carries no frameworks.client_lock_sha256"),
    }
    for name, (kw, needle) in cases.items():
        kw = {"block": block, "gs_ref": ref, "item_refs": (ref,), **kw}
        _producer_store(tmp_path / name, row, sha, **kw)
        bad = M.row_identity_record(row, producer_tree_sha=sha, out_root=str(tmp_path / name))
        assert bad["complete"] is False and bad["identity"]["sampling_ref"] is None and bad["identity"]["client_lock_sha256"] is None, name
        assert any(needle in p for p in bad["problems"]), (name, bad["problems"])
    # the chain of custody on a READABLE tree: with THIS checkout's HEAD as the producer tree, a manifest recording the
    # tree's own lock digest is complete; one recording another digest is named (the leg did not run under its tree's lock)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    head_lock = hashlib.sha256(subprocess.run(["git", "-C", str(REPO), "show", f"{head}:{C.CLIENT_LOCK_TXT}"], capture_output=True, check=True).stdout).hexdigest()
    _producer_store(tmp_path / "headlock", row, head, block=block, gs_ref=ref, item_refs=(ref,), manifest_over={"frameworks": {"client_lock_sha256": head_lock}})
    good = M.row_identity_record(row, producer_tree_sha=head, out_root=str(tmp_path / "headlock"))
    assert good["complete"] is True and good["identity"]["client_lock_sha256"] == head_lock and good["checks"]["client_lock_evidence"]["agree"] is True
    assert good["checks"]["client_lock_evidence"]["tree_lock_sha256"] == head_lock and good["checks"]["client_lock_evidence"]["why"] is None
    _producer_store(tmp_path / "otherlock", row, head, block=block, gs_ref=ref, item_refs=(ref,), manifest_over={"frameworks": {"client_lock_sha256": "ab" * 32}})
    bad = M.row_identity_record(row, producer_tree_sha=head, out_root=str(tmp_path / "otherlock"))
    assert bad["complete"] is False and bad["identity"]["client_lock_sha256"] is None and bad["checks"]["client_lock_evidence"]["agree"] is False
    assert any(p.startswith("client_lock_sha256: the driver's manifest records " + "ab" * 32) and "did not run under its tree's lock" in p for p in bad["problems"]), bad["problems"]
    # the CLI prints the same record; --sha is required
    r = subprocess.run([sys.executable, str(M5T / "cells.py"), "row", "--index", str(row["index"]), "--identity", "--sha", sha,
                        "--out-root", str(out_root)], capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout) == rec
    r = subprocess.run([sys.executable, str(M5T / "cells.py"), "row", "--index", "24", "--identity"], capture_output=True, text=True)
    assert r.returncode != 0 and "--sha" in r.stderr
    p = M.row_identity(M.row_for("qwen35_4b", M.CELL_PROMPTED_CONST, M.SAMPLING_THINKING_CARD), producer_tree_sha=sha)
    assert p["object_artifact_sha256"] is None and p["route_kind"] == "prompted" and p["sampling_ref"] is None and p["view"] == "main"
    # the shell writer (m5t_row_identity): the same CLI, the file written atomically beside the store, the verdict logged
    funcs = _m5t_block("m5t_row_identity(){", "m5t_print_plan(){")
    (tmp_path / "serve" / "math500").mkdir(parents=True)

    def writer(out: Path) -> subprocess.CompletedProcess:
        script = ('set -u\nf22_log(){ echo LOG "$@"; }\n' + f"m5t_py(){{ echo {shlex.quote(sys.executable)}; }}\n" + funcs
                  + f"\nREPO_DIR={shlex.quote(str(REPO))}; M5T_DIR=scripts/sdfbase/math500; OUT_ROOT={shlex.quote(str(out))}; "
                  f"M5T_ROW_IDENTITY_FILENAME={M.ROW_IDENTITY_FILENAME}; M5T_INDEX={row['index']}; WAVE_SHA={sha}; "
                  f"SERVE_DIR={shlex.quote(str(tmp_path / 'serve'))}\nm5t_row_identity\n")
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    r = writer(out_root)
    assert r.returncode == 0 and "row identity:" in r.stdout and f"complete=true sampling_ref={ref}" in r.stdout, (r.stdout, r.stderr)
    assert json.loads((out_root / M.ROW_IDENTITY_FILENAME).read_text()) == rec and not (out_root / f"{M.ROW_IDENTITY_FILENAME}.tmp").exists()
    r = writer(tmp_path / "noblock")
    assert r.returncode == 0 and "complete=false sampling_ref=None problems=" in r.stdout and "no `sampling` block" in r.stdout, r.stdout
    assert json.loads((tmp_path / "noblock" / M.ROW_IDENTITY_FILENAME).read_text())["complete"] is False


def test_the_row_identity_stamp_reads_the_object_sha_off_the_standups_per_cell_map_when_the_route_scalar_is_none(tmp_path):
    """A NATIVE LoRA standup serving the whole bank (the seed-replicate wave) records
    `route.object_artifact_sha256 = None` and every registered cell's digest in `route.object_artifact_sha256_by_cell`
    under its harness id, so comparing the scalar alone read a CORRECT store as `object_artifact_sha256_agrees: false`
    (`complete: false` on every completed fitted seed leg). The stamp resolves the manifest's digest through ONE rule —
    the scalar when it is a non-empty string, else the map under the manifest's own `harness_cell_id`, else the
    `proxy_identity` map the GSM8K pairing reader accepts — records the resolved value + its source in `checks`, and
    compares THAT with the row's. The spellings are the contracts' (pinned equal here; cells.py stays stdlib-only)."""
    assert M.ROUTE_KEY_OBJECT_ARTIFACT_SHA256 == C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256 == "object_artifact_sha256"
    assert M.ROUTE_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL == C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL == "object_artifact_sha256_by_cell"
    assert M.MANIFEST_KEY_ROUTE == C.MANIFEST_KEY_ROUTE == "route" and M.MANIFEST_KEY_HARNESS_CELL_ID == "harness_cell_id"
    sha = "cd" * 20
    row = M.row_for("qwen35_4b", M.CELL_PSR_MIX, M.SAMPLING_THINKING_CARD, view="s1")        # a seed-replicate fitted row: native serving from the whole s1 bank
    assert row["wave"] == "qwen_seedrep" and row["object_sha256"] and row["harness_cell"] == "si_constv2_mix_full__psr_ce_all"
    companion = row["standup_companion_harness_cells"][0]
    companion_sha = row["standup_companion_pins"][row["standup_companion_cells"][0]]
    assert companion != row["harness_cell"] and companion_sha != row["object_sha256"]
    block = _producer_sampling_block(row)
    ref = C.sampling_ref(block)

    def store(name: str, route: dict, **over) -> dict:
        _producer_store(tmp_path / name, row, sha, block=block, gs_ref=ref, item_refs=(ref, ref),
                        manifest_over={"route": route, "harness_cell_id": row["harness_cell"], **over})
        return M.row_identity_record(row, producer_tree_sha=sha, out_root=str(tmp_path / name))

    # the resolver alone, on the real store's shape (the brief's verified manifest: scalar None, the bank's cells in the map)
    bank = {companion: companion_sha, row["harness_cell"]: row["object_sha256"]}
    assert M.manifest_object_artifact_sha256({"harness_cell_id": row["harness_cell"], "route": {"object_artifact_sha256": None, "object_artifact_sha256_by_cell": bank}}) \
        == (row["object_sha256"], M.OBJECT_SHA_SOURCE_ROUTE_BY_CELL)
    # (1) the scalar None + the map under the harness id -> agrees, complete, the value and its source recorded
    rec = store("bank", {"object_artifact_sha256": None, "object_artifact_sha256_by_cell": bank})
    assert rec["complete"] is True and rec["problems"] == [], rec["problems"]
    assert rec["checks"]["object_artifact_sha256_agrees"] is True
    assert rec["checks"]["manifest_object_artifact_sha256"] == row["object_sha256"] == rec["identity"]["object_artifact_sha256"]
    assert rec["checks"]["object_artifact_sha256_source"] == M.OBJECT_SHA_SOURCE_ROUTE_BY_CELL == "route.by_cell"
    assert rec["checks"]["manifest_harness_cell_id"] == row["harness_cell"]
    # (2) a map whose entry under the harness id is another object's digest -> disagrees, the source + both digests named
    bad = store("wrong", {"object_artifact_sha256": None, "object_artifact_sha256_by_cell": {row["harness_cell"]: companion_sha}})
    assert bad["complete"] is False and bad["checks"]["object_artifact_sha256_agrees"] is False
    assert bad["checks"]["manifest_object_artifact_sha256"] == companion_sha and bad["checks"]["object_artifact_sha256_source"] == "route.by_cell"
    assert bad["problems"] == [f"object_artifact_sha256_agrees: the driver's manifest disagrees with the row (manifest {companion_sha} via route.by_cell, row {row['object_sha256']})"]
    # a map that carries only the OTHER cells (the leg's own id absent) resolves None on a fitted row -> disagrees, no source
    absent = store("absent", {"object_artifact_sha256": None, "object_artifact_sha256_by_cell": {companion: companion_sha}})
    assert absent["complete"] is False and absent["checks"]["manifest_object_artifact_sha256"] is None and absent["checks"]["object_artifact_sha256_source"] is None
    assert any(p.startswith("object_artifact_sha256_agrees:") and "no route.object_artifact_sha256, no by-cell entry" in p for p in absent["problems"]), absent["problems"]
    # (3) the scalar takes precedence when present: a lens standup's manifest names the served digest there, whatever the map says
    lens = store("lens", {"object_artifact_sha256": row["object_sha256"], "object_artifact_sha256_by_cell": {row["harness_cell"]: companion_sha}})
    assert lens["complete"] is True and lens["checks"]["object_artifact_sha256_source"] == M.OBJECT_SHA_SOURCE_ROUTE == "route"
    assert lens["checks"]["manifest_object_artifact_sha256"] == row["object_sha256"]
    lens_bad = store("lens_bad", {"object_artifact_sha256": companion_sha, "object_artifact_sha256_by_cell": bank})     # the scalar wins even when the map agrees
    assert lens_bad["complete"] is False and lens_bad["checks"]["object_artifact_sha256_source"] == "route" and lens_bad["checks"]["manifest_object_artifact_sha256"] == companion_sha
    # the empty-string scalar is not a digest: the map is read
    assert store("empty", {"object_artifact_sha256": "", "object_artifact_sha256_by_cell": bank})["checks"]["object_artifact_sha256_source"] == "route.by_cell"
    # (4) the proxy_identity fallback the GSM8K pairing reader accepts (sisteer.eval.anchor_pairing.stamp_from_served_target)
    prox = store("proxy", {"object_artifact_sha256": None}, proxy_identity={"object_artifact_sha256_by_cell": bank})
    assert prox["complete"] is True and prox["checks"]["object_artifact_sha256_source"] == M.OBJECT_SHA_SOURCE_PROXY_BY_CELL == "proxy_identity.by_cell"
    # (5) a map but NO harness id on the manifest: never a guess from the row -> None, incomplete by name (fail-closed)
    _producer_store(tmp_path / "noharness", row, sha, block=block, gs_ref=ref, item_refs=(ref,),
                    manifest_over={"route": {"object_artifact_sha256": None, "object_artifact_sha256_by_cell": bank}})
    nh = M.row_identity_record(row, producer_tree_sha=sha, out_root=str(tmp_path / "noharness"))
    assert nh["complete"] is False and nh["checks"]["manifest_object_artifact_sha256"] is None and nh["checks"]["object_artifact_sha256_source"] is None
    assert nh["checks"]["manifest_harness_cell_id"] is None
    # (6) a `none` row on the same standup: its harness id is in no map, the scalar is None -> None == None agrees, as before
    none_row = M.row_for("qwen35_4b", M.CELL_NONE, M.SAMPLING_THINKING_CARD, view="s1")
    assert none_row["object_sha256"] is None
    none_block = _producer_sampling_block(none_row); none_ref = C.sampling_ref(none_block)
    _producer_store(tmp_path / "none", none_row, sha, block=none_block, gs_ref=none_ref, item_refs=(none_ref,),
                    manifest_over={"route": {"object_artifact_sha256": None, "object_artifact_sha256_by_cell": bank}, "harness_cell_id": none_row["harness_cell"]})
    nr = M.row_identity_record(none_row, producer_tree_sha=sha, out_root=str(tmp_path / "none"))
    assert nr["complete"] is True and nr["checks"]["object_artifact_sha256_agrees"] is True
    assert nr["checks"]["manifest_object_artifact_sha256"] is None and nr["checks"]["object_artifact_sha256_source"] is None
    # the earlier shape (a lens store with the scalar only, as _producer_store writes by default) is byte-for-byte the old verdict
    _producer_store(tmp_path / "old", row, sha, block=block, gs_ref=ref, item_refs=(ref,))
    old = M.row_identity_record(row, producer_tree_sha=sha, out_root=str(tmp_path / "old"))
    assert old["complete"] is True and old["checks"]["object_artifact_sha256_source"] == "route" and old["schema"] == "m5t_row_identity/3"


@pytest.mark.skipif(not GRADE_PUBLISHED.is_dir(), reason="the published pull is not on this machine")
def test_the_row_identity_stamp_reproduces_the_published_legs_sampling_ref_from_their_real_stores():
    """Fix A3 on the REAL published pull: the stamp derives, for each extension model's published `none` leg at the
    published tree, exactly the ref its gen_stats and outcome items carry (the digest the old field read gave `null`)."""
    want = {"gemma4e4b": "83215617d18b", "qwen35_4b": "89a93d3c284f", "nemotron3nano4b": "4fc6d968b926"}
    for key, ref in want.items():
        row = M.row_for(key, M.CELL_NONE, M.SAMPLING_THINKING_CARD)
        rec = M.row_identity_record(row, producer_tree_sha=PUBLISHED_TREE, out_root=str(GRADE_PUBLISHED / key / "none" / M.SAMPLING_THINKING_CARD))
        assert rec["complete"] is True and rec["problems"] == [] and rec["identity"]["sampling_ref"] == ref, (key, rec["problems"])
        ev = rec["checks"]["sampling_ref_evidence"]
        assert ev["gen_stats"] == {"inspect/math": ref} and ev["outcomes"]["inspect__math.json"]["refs"] == [ref]


def test_pull_wave_pulls_one_wave_subtree_and_never_the_published_controls(tmp_path):
    """`pull_wave.sh --subtree controls_ext1 <sha7>` rsyncs that wave's subtree + budgets + nemo_backend
    + the launch-attestation pairs under serving/ (the merge seam) and NOTHING under controls/ (the
    published legs); no --delete; the default shape (controls + diagnostic + dry_run_prod) is the published pull's plus the
    same serving/ pairs; a path-shaped or foreign subtree and a missing source refuse. The rsync argv is asserted through
    --print-argv, never run (the filter chain's EFFECT is proven by the real-rsync test below)."""
    script = M5T / "pull_wave.sh"
    env = {**os.environ, "M5T_PULL_SRC": "TESTHOST:/wave/root", "M5T_PULL_DEST_ROOT": str(tmp_path)}

    def argv(*args):
        r = subprocess.run(["bash", str(script), "--print-argv", *args], capture_output=True, text=True, env=env)
        return r.returncode, r.stdout.splitlines(), r.stderr
    rc, words, _ = argv("--subtree", M.OUT_ROOT_PREFIX_BY_WAVE[M.WAVE_EXT1], "abc1234")
    assert rc == 0 and words[0] == "rsync" and "--delete" not in words and "-az" in words
    srcs = [w for w in words if w.startswith("TESTHOST:/wave/root/")]
    assert srcs == ["TESTHOST:/wave/root/controls_ext1", "TESTHOST:/wave/root/budgets", "TESTHOST:/wave/root/nemo_backend",
                    "TESTHOST:/wave/root/serving"]
    assert "TESTHOST:/wave/root/controls" not in words and words[-1] == f"{tmp_path}/pull_abc1234/"
    assert {"--exclude=*-superseded", "--exclude=*-failed-probe", "--exclude=*-cancelled", "--exclude=*-censored",
            "--exclude=inspect_evals_cache", "--exclude=serving"} <= set(words)
    # the serving/ filter chain, in rsync's first-match order: the retired suffixes BEFORE the include chain (a retired
    # standup dir is never pulled), the two files' includes BEFORE `/serving/**` (everything else under serving/ stays)
    filt = [w for w in words if w.startswith(("--include=", "--exclude="))]
    chain = ["--include=/serving/", "--include=/serving/*/", "--include=/serving/*/math500/",
             "--include=/serving/*/math500/steering_attestation.json", "--include=/serving/*/math500/vllm_server.log",
             # the standup's serving-config provenance rides beside the attestation pair
             "--include=/serving/*/math500/serving_config.json",
             # launch-attestation-native: the NATIVE engine's launch record (a lora-native standup writes it instead of the lens file)
             "--include=/serving/*/math500/serving_attestation.json",
             # native-positions-resolve (the GSM8K pull's shape since #164): the NATIVE cell's standup record — the routing
             # table's engine-neutral served-positions map and the standup jobs record (the launch tree the pins derive from)
             "--include=/serving/*/math500/routing_table.json", "--include=/serving/*/math500/standup_jobs.jsonl",
             "--exclude=/serving/**", "--exclude=serving"]
    assert filt[-len(chain):] == chain and filt.index("--exclude=*-superseded") < filt.index("--include=/serving/")
    rc, words, _ = argv("abc1234")                       # the published wave's pull: its old shape + the serving/ pairs
    assert rc == 0 and [w for w in words if w.startswith("TESTHOST:")] == [
        "TESTHOST:/wave/root/controls", "TESTHOST:/wave/root/budgets", "TESTHOST:/wave/root/nemo_backend",
        "TESTHOST:/wave/root/serving", "TESTHOST:/wave/root/diagnostic", "TESTHOST:/wave/root/dry_run_prod"]
    for bad in ("controls/gemma4e4b", "serving", "budgets", ""):
        rc, _, err = argv("--subtree", bad, "abc1234")
        assert rc == 2 and "--subtree" in err, bad
    r = subprocess.run(["bash", str(script), "--print-argv", "abc1234"], capture_output=True, text=True,
                       env={k: v for k, v in env.items() if k != "M5T_PULL_SRC"})
    assert r.returncode == 2 and "no source" in r.stderr
    rc, words, _ = argv("--src", "OTHER:/r", "--dest", str(tmp_path / "d"), "--subtree", "controls_ext1", "abc1234")
    assert rc == 0 and words[-1] == f"{tmp_path}/d/" and "OTHER:/r/controls_ext1" in words
    # the script carries no host name, no home path, no cluster codename (the machine-local wrapper carries the host)
    src = script.read_text()
    assert "M5T_PULL_SRC" in src and "node-" not in src and "/home/" not in src and "workspace-vast" not in src
    # the out-root prefixes the matrix names are exactly what the subtree grammar admits
    assert set(M.OUT_ROOT_PREFIX_BY_WAVE.values()) == {"controls", "controls_ext1", "controls_steer", "controls_llama", "controls_itdoc",
                                                       "controls_arditi_allpos", "controls_qwen_seedrep"} and all(
        p == "controls" or p.startswith("controls_") for p in M.OUT_ROOT_PREFIX_BY_WAVE.values())
    assert all(r["out_root"].split("/", 1)[0] == M.OUT_ROOT_PREFIX_BY_WAVE[r["wave"]] for r in M.LAUNCHER_ROWS)
    rc, words, _ = argv("--subtree", M.OUT_ROOT_PREFIX_BY_WAVE[M.WAVE_SEEDREP], "abc1234")      # the seed wave's subtree pulls like any other
    assert rc == 0 and "TESTHOST:/wave/root/controls_qwen_seedrep" in words and "TESTHOST:/wave/root/controls" not in words


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is the tool under test")
def test_pull_wave_carries_exactly_the_two_launch_attestation_files_per_standup_dir(tmp_path):
    """Code review C2 / the merge seam (FB5 / FB8): the analysis' `launch_attestation` reads
    `<gen root>/serving/<jobdir>/math500/{steering_attestation.json | serving_attestation.json,vllm_server.log}` (+ the
    serving-config record) — the API server's startup echo is the ONE surviving record of the composed launch flags; the
    ENGINE's record binds it to the job — and `pull_wave.sh` used to exclude serving/ wholesale, so a real eight-arm run
    refused every cross-tree contrast (and, until launch-attestation-native, carried no native record at all). Run the REAL
    rsync over a synthetic wave root (local paths; `-e ssh` is inert for a local transfer) and read the destination: exactly
    the analysis' record files per standup dir land (the lens pair for a lens job, the native attestation + log + config for a
    native job), in the layout
    the sbatch writes and the analysis locates (proven by calling its locator over the pull); a retired (`-superseded`)
    standup dir, another instrument's task subdir, a pair-less dir, a loose file and a `serving` directory NESTED under a
    leg are all out; the published controls/ leg is not pulled under --subtree controls_ext1; a file already at the
    destination survives (never --delete) and a second pull is a no-op, which is how a pull made BEFORE this rode along
    gains its serving/. The task subdir and the two file names are read from the sbatch and the analysis, not retyped."""
    A = importlib.import_module("tinycap_math500_thinking")
    sbatch = (M5T / "math500_thinking.sbatch").read_text()
    assert 'mkdir -p "$SERVE_DIR/math500"' in sbatch and A.LAUNCH_STANDUP_TASK_SUBDIR == "math500"
    assert A.LAUNCH_STANDUP_DIR_PARENT == "serving" and 'serving/m5t_${KEY}_${CELLSLUG}_${SAMPLING_MODE}_${SLURM_JOB_ID' in sbatch
    task, att, log = A.LAUNCH_STANDUP_TASK_SUBDIR, A.LAUNCH_ATTESTATION_FILE, A.LAUNCH_LOG_FILE
    natt, cfg = A.LAUNCH_NATIVE_ATTESTATION_FILE, A.LAUNCH_SERVING_CONFIG_FILE
    assert natt == "serving_attestation.json" and att == "steering_attestation.json" and A.LAUNCH_RECORD_FILE_BY_ENGINE == {"lens": att, "lora_native": natt}
    script_src = (M5T / "pull_wave.sh").read_text()
    assert f"SERVING_TASK={task}" in script_src and f'/{att}"' in script_src and f'/{log}"' in script_src and f'/{natt}"' in script_src and f'/{cfg}"' in script_src
    root, dest = tmp_path / "root", tmp_path / "dest"

    def put(rel: str, text: str | None = None) -> Path:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text if text is not None else rel)
        return f
    wave_leg = "controls_ext1/gemma4e4b/prompted__si_constitution/thinking_card"
    put("controls/gemma4e4b/none/thinking_card/rows.jsonl")                       # the published leg: never under --subtree controls_ext1
    put(f"{wave_leg}/rows.jsonl", '{"id": 1}\n')
    put(f"{wave_leg}/serving/nested.txt")                                          # a `serving` dir nested under a leg: excluded
    put("controls_ext1/gemma4e4b/prompted__si_constitution/thinking_card_1234abcd-superseded/rows.jsonl")
    put("budgets/gemma4e4b/prompt_budget.json"); put("nemo_backend/attestation.json")
    wave_job, pub_job = "m5t_gemma4e4b_consi_thinking_card_500001", "m5t_gemma4e4b_none_thinking_card_483692"
    echo = "INFO api_utils.py:273 non-default args: {'port': 41001, 'dtype': 'bfloat16'}\n"
    for extra in ("chat_template.jinja", "driver.log", "endpoint.json", "cells_served.json"):
        put(f"serving/{wave_job}/{task}/{extra}")
    # native-positions-resolve: the standup RECORD pair rides for every engine (the routing table's served-positions map, the
    # standup jobs record the store pins derive from) — the GSM8K pull's shape since #164
    put(f"serving/{wave_job}/{task}/routing_table.json"); put(f"serving/{wave_job}/{task}/standup_jobs.jsonl")
    put(f"serving/{wave_job}/{task}/{att}", '{"slurm_job_id": "500001", "port": 41001}')
    put(f"serving/{wave_job}/{task}/{log}", echo)
    put(f"serving/{pub_job}/{task}/{att}", '{"slurm_job_id": "483692", "port": 41002}')      # the published job's pair rides too
    put(f"serving/{pub_job}/{task}/{log}", echo)
    # launch-attestation-native: a NATIVE (lora_native) standup dir — the serving attestation INSTEAD of the lens file, the same
    # log, the serving-config record — lands as those three plus the standup record pair; its endpoint record, cells_served.json and standup log stay
    nat_job = "m5t_llama8b_consmix-psr_primary_500002"
    nat_echo = ("INFO api_utils.py:273 non-default args: {'port': 41003, 'lora_modules': [LoRAModulePath(name='none__lora0', path='/a', "
                "base_model_name=None, is_3d_lora_weight=False)], 'enable_lora': True}\n")
    put(f"serving/{nat_job}/{task}/{natt}", '{"kind": "lora_native", "slurm_job_id": "500002", "port": 41003}')
    put(f"serving/{nat_job}/{task}/{log}", nat_echo)
    put(f"serving/{nat_job}/{task}/{cfg}", '{"serving_config_sha256": "' + "0" * 64 + '"}')
    for extra in ("endpoint.recorded.json", "cells_served.json", "standup.log", "serving_path_selection.json"):
        put(f"serving/{nat_job}/{task}/{extra}")
    put(f"serving/{nat_job}/{task}/routing_table.json"); put(f"serving/{nat_job}/{task}/standup_jobs.jsonl")
    put(f"serving/m5t_gemma4e4b_none_thinking_card_483000-superseded/{task}/{att}")          # retired standup dir: out
    put(f"serving/m5t_gemma4e4b_none_thinking_card_483000-superseded/{task}/{log}")
    put(f"serving/g8k_job_777/gsm8k/{att}"); put(f"serving/g8k_job_777/gsm8k/{log}")         # another instrument's task subdir: out
    put(f"serving/m5t_gemma4e4b_x_thinking_card_dry/{task}/driver.log")                      # no pair: pruned
    put("serving/loose.txt")
    (dest / "controls_ext1").mkdir(parents=True)
    (dest / "controls_ext1" / "stale_marker.txt").write_text("from an earlier pull")
    cmd = ["bash", str(M5T / "pull_wave.sh"), "--src", str(root), "--dest", str(dest), "--subtree", "controls_ext1", "abc1234"]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ})
    assert r.returncode == 0, r.stderr
    assert f"launch attestations (serving/<jobdir>/{task}/): 3" in r.stdout
    assert "serving-config records: 1" in r.stdout and "native serving attestations: 1" in r.stdout
    assert "routing tables: 2" in r.stdout and "standup job records: 2" in r.stdout

    def listing() -> list[str]:
        return sorted(str(f.relative_to(dest)) for f in dest.rglob("*") if f.is_file())
    assert listing() == sorted([
        "budgets/gemma4e4b/prompt_budget.json", "nemo_backend/attestation.json", "controls_ext1/stale_marker.txt",
        f"{wave_leg}/rows.jsonl",
        f"serving/{wave_job}/{task}/{att}", f"serving/{wave_job}/{task}/{log}",
        f"serving/{wave_job}/{task}/routing_table.json", f"serving/{wave_job}/{task}/standup_jobs.jsonl",
        f"serving/{pub_job}/{task}/{att}", f"serving/{pub_job}/{task}/{log}",
        f"serving/{nat_job}/{task}/{natt}", f"serving/{nat_job}/{task}/{log}", f"serving/{nat_job}/{task}/{cfg}",
        f"serving/{nat_job}/{task}/routing_table.json", f"serving/{nat_job}/{task}/standup_jobs.jsonl"])
    assert (dest / "serving" / wave_job / task / log).read_text() == echo
    assert (dest / "serving" / nat_job / task / log).read_text() == nat_echo and not (dest / "serving" / nat_job / task / att).exists()
    # the pulled native record is what the analysis' engine-keyed selection reads: the native manifest resolves it, a lens one
    # names it as the other engine's record (never read as the lens file)
    nat_manifest = {"routing_table": f"/cluster/wave/root/serving/{nat_job}/{task}/routing_table.json",
                    "route": {"serving_engine": "lora_native"}, "tinycap": {"served": {}}}
    nd, njobdir, nproblem = A.standup_dir_of(nat_manifest, gen_root=dest)
    assert nproblem is None and njobdir == nat_job and (nd / natt).is_file() and (nd / cfg).is_file() and not (nd / att).exists()
    assert A.launch_engine_of(nat_manifest) == ("lora_native", None)
    lens_read = A.launch_attestation({**nat_manifest, "route": {}}, gen_root=dest)
    assert lens_read["status"] == A.LAUNCH_ABSENT and any("no workers' attestation" in p for p in lens_read["problems"]) and any("torn / mixed" in p for p in lens_read["problems"])
    assert not (dest / "serving" / "g8k_job_777").exists() and not (dest / "serving" / "m5t_gemma4e4b_x_thinking_card_dry").exists()
    # the analysis' own locator resolves the pulled pair under the pull as the leg's gen root, from a manifest that names
    # the CLUSTER path the way the driver stamps it (only the trailing serving/<jobdir>/<task>/routing_table.json is read)
    manifest = {"routing_table": f"/cluster/wave/root/serving/{wave_job}/{task}/routing_table.json"}
    d, jobdir, problem = A.standup_dir_of(manifest, gen_root=dest)
    assert problem is None and jobdir == wave_job and d == dest / "serving" / wave_job / task
    assert (d / att).is_file() and (d / log).is_file()
    before = listing()
    r2 = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ})
    assert r2.returncode == 0 and listing() == before


# ----------------------------------------------------------------------------- #
# the AlphaSteer / Arditi rung                                                    #
# ----------------------------------------------------------------------------- #


STEER_LABEL = ("highest rung within the 4 pp over-refusal budget (gate statistic, prefix rule); per-rung defence not "
               "measured on this ladder, so the defence DELIVERED at this rung is unquantified.")


def test_the_steer_rows_carry_the_admission_records_pins_labels_and_flags():
    """One row per (model, external arm) of the ADMISSION RECORD, keyed by CELL ID and lane root;
    the object pinned by the record's 64-hex sha256 (never the full22 pin file, which stages neither bank); the
    selection label VERBATIM on every AlphaSteer row with its rung + lambda; `grid_limited` false on every active row, the
    rule naming a point alone; the Arditi rows' alpha, served layer, producer repo sha and the frozen serving
    implementation; llama's Arditi point fixed at alpha 0.05 on 2026-09-14 (the inclusive-boundary pick of 0.1,
    `not_robust`, kept under `superseded`)."""
    admission = M.load_steer_admission()
    rows = M.STEER_ROWS
    # the RUNG enumerates its own two cells — never the steer VOCABULARY, which also carries the all-positions arm
    # (its own wave, appended after every row here so none of these indices move)
    assert [(r["model_key"], r["cell"]) for r in rows] == [(k, c) for k in M.STEER_MODELS for c in M.WAVE_STEER_CELLS]
    assert M.STEER_MODELS == ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b") and M.STEER_ONLY_MODELS == ("llama8b",)
    assert [r["index"] for r in rows] == list(range(33, 33 + len(rows))) and M.M5T_CELLS[33:33 + len(rows)] == rows
    pin_file = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())
    for r in rows:
        key, cell = r["model_key"], r["cell"]
        entry = admission[(key, cell)]
        assert r["object_source"] == "external" and r["object_sha256"] == entry["object_sha256"] and len(r["object_sha256"]) == 64
        # the pin is the ADMISSION RECORD's: the full22 file stages no such cell for this model, by construction
        assert cell not in (pin_file.get(key) or {}) and r["staged_id"] == entry.get("staged_id")
        assert r["wave"] == M.WAVE_STEER and r["out_root"].startswith(f"controls_steer/{key}/{cell}/")
        assert r["fitted"] is True and r["route_kind"] == "lens" and r["served_positions_expected"] == "all"
        assert r["view"] == entry["view"] and r["bank_cell_dir"] == entry["bank_cell_dir"]
        assert r["lane_root"] == entry["lane_root"] and r["bank_view_rel"]["primary"].startswith(entry["lane_root"])
        assert r["selection_rule"] == entry["selection_rule"] and r["selection_label"] == entry["selection_label"]
        assert r["log_requests"] is True and r["budget_class"] == M.CELL_NONE and r["geometry_eval"] == M.GEOMETRY_EVAL
        assert r["harness_cell"] == cell and r["cell_slug"] == cell            # a bare Class B id: one spelling
        # the row's MODE is the model's — the no-switch family's shipped `card` row (`--sampling thinking_card` on llama8b is
        # refused by the driver's sampling table: the first-landed steering llama rows could not have run), a switch family's `thinking_card`
        assert r["sampling_mode"] == M.sampling_mode_for(key) == (M.SAMPLING_CARD if key in M.STEER_ONLY_MODELS else M.SAMPLING_THINKING_CARD)
        assert r["out_root"] == f"controls_steer/{key}/{cell}/{r['sampling_mode']}"
        if cell == M.CELL_ALPHASTEER:
            # the label is VERBATIM on every AlphaSteer row — never "defence bar unverified" (which would assert a
            # bar and a failed check), and never the retired minimising rule
            assert r["selection_label"] == STEER_LABEL
            assert isinstance(r["rung"], int) and isinstance(r["lambda"], float) and r["alpha"] is None
            assert r["ceiling_20pp"]["rung"] != r["rung"] and "never run" in r["ceiling_20pp"]["role"]
        else:
            assert r["alpha"] in (0.05, 0.1) and isinstance(r["served_layer"], int) and r["rung"] is None
            assert r["producer"]["serving_impl_frozen_sha"] == "94e294c445dc5eec6ebe0ce3a3e25553004cd4b0"
        assert "smallest" not in r["selection_rule"] and "defence bar" not in r["selection_label"]
        assert r["producer"]["serving_config_sha256_note"].startswith(("the PRODUCER's", "today's value"))
    by = {(r["model_key"], r["cell"]): r for r in rows}
    # the grid-limited flag travelled with nemotron's A4-fallback point (rung 1) — the 2026-09-14 re-pin moved nemotron to
    # rung 2 (and qwen to rung 1; gemma to rung 2 later the same day), and a ruled point is never grid-limited, so no
    # ACTIVE row carries the flag now; the superseded point keeps it under the entry's `superseded` block
    assert [k for k, r in by.items() if r["grid_limited"]] == []
    assert admission[("nemotron3nano4b", M.CELL_ALPHASTEER)]["superseded"]["grid_limited"] is True
    assert {k: r["rung"] for k, r in by.items() if k[1] == M.CELL_ALPHASTEER} == {("gemma4e4b", M.CELL_ALPHASTEER): 2, ("qwen35_4b", M.CELL_ALPHASTEER): 1,
                                                                                  ("nemotron3nano4b", M.CELL_ALPHASTEER): 2, ("llama8b", M.CELL_ALPHASTEER): 1}
    # qwen's Arditi cell came from an EARLIER pinned tree of that lane — recorded per cell, not per model
    assert by[("qwen35_4b", M.CELL_ARDITI)]["producer"]["repo_sha"] == "ef1ae2671c8799a3b8e5658d60abfadc4c446e0f"
    assert {by[(k, M.CELL_ARDITI)]["producer"]["repo_sha"] for k in ("gemma4e4b", "nemotron3nano4b", "llama8b")} == {
        "731677e600271f0647372edadf06da546042daff"}
    # since 2026-09-14 llama's Arditi point has moved off the inclusive-boundary
    # pick (alpha 0.1, `not_robust`, 0.50 of an item inside the 10-item budget) to alpha 0.05 — the object that was the NOTED alternative is
    # now THE pin (re-hashed on disk), `boundary_pick: ruled`, the ruled word on the row; the 0.1 point keeps its sha under `superseded`
    llama_ar = by[("llama8b", M.CELL_ARDITI)]
    assert llama_ar["boundary_pick"] == "ruled" and llama_ar["alpha"] == 0.05 and llama_ar["alternative_not_pinned"] is None
    assert llama_ar["object_sha256"] == "1755a5627a18f931b8f125bcebe52f399da3d7507128de60ff6f55dba717a0b3" and llama_ar["bank_view_rel"]["primary"].endswith("cells_view/ar_r0p05")
    # the MATH-500 row copies no `selection_outcome` (STEER_KEYS); the ruled word is read off the admission ENTRY here
    assert admission[("llama8b", M.CELL_ARDITI)]["selection_outcome"] == "ruled_2026-09-14" and llama_ar["dose_evidence"]["must_report_with"] == ["over_refusal_pp"]
    sup = admission[("llama8b", M.CELL_ARDITI)]["superseded"]
    assert sup["alpha"] == 0.1 and sup["view"] == "ar_r0p1" and sup["object_sha256"] == "0f654e88f5bd7d896226b6e326a83d7431d038e6bce0b9cfb69d0e05adda877f"
    assert sup["object_sha256"] not in {r["object_sha256"] for r in M.M5T_CELLS} and sup["boundary_pick"]["status"] == "not_robust"
    # the pin DISCREPANCY: the llama AlphaSteer pin is the string that EXISTS on disk (hashed on the cluster)
    assert by[("llama8b", M.CELL_ALPHASTEER)]["object_sha256"] == (
        "4cf8e1641a596d2dce36fd151cec4a72493fa6401d7b6da5cb7592720e0c5396")
    assert "4cf8e1641a596d2dcc0e3a37af9df63ed15f6b7ec81e9e30caff0cff8f5d2a48" not in json.dumps(M.M5T_CELLS)
    # nemotron's SUPERSEDED (rung-1, A4-fallback) dose was measured under ANOTHER decode — that transfer sentence rides the
    # superseded block now; the ACTIVE (ruled rung-2) evidence is the oracle's: defence delta, worst degeneration, the over-refusal
    # as an UPPER BOUND that must be reported with the point (decisions-e2b LB10)
    sup = admission[("nemotron3nano4b", M.CELL_ALPHASTEER)]["superseded"]["dose_evidence"]
    assert "top_p 1.0" in sup["decode_measured"] and sup["over_refusal_at_top_p_0p95"] is None
    assert "the transfer is unvalidated" in sup["decode_transfer"]
    nemo = by[("nemotron3nano4b", M.CELL_ALPHASTEER)]["dose_evidence"]
    assert nemo["from_operating_points"]["over_refusal_is_upper_bound"] is True and nemo["must_report_with"] == ["over_refusal_pp"]
    assert "superseded.dose_evidence" in nemo["note"]
    # every pin is DISTINCT: eight objects, eight digests
    assert len({r["object_sha256"] for r in rows}) == len(rows)


def test_the_steer_rows_launchability_follows_the_familys_certified_render_fact():
    """The hold these rows shipped with was a TREE gate, not a design deferral: the no-switch family's admission record was
    complete throughout (object, dose, selection, pin — verified on disk) and only its RENDER was unknown here. The
    Llama lane's certified fact landed 2026-09-13, so those rows are `launch` now — no repin, no
    re-selection, exactly as `steer_hold_reason` promised. What the test pins is the MECHANISM, not today's answer:
    the launcher's mirror equals the contracts' own predicate family by family, the hold reason is derived from it,
    and a family whose fact is withdrawn holds its rows again by name."""
    for family, certified in M.RENDER_FACT_CERTIFIED_BY_FAMILY.items():
        assert certified == C.render_fact_certified(C.render_fact_for(family)), family
    assert set(M.RENDER_FACT_CERTIFIED_BY_FAMILY) == set(M.FAMILY_BY_MODEL.values())
    assert M.RENDER_FACT_CERTIFIED_BY_FAMILY["llama"] is True and C.render_fact_for("llama") is not None
    assert M.SPLIT_MODE_CANDIDATE["llama"] == C.FAMILY_RENDER_FACTS["llama"].split_mode == "none"
    # every steering row is launchable today (E2-16: the class-B record hold on llama8b AlphaSteer was lifted when the rebake
    # lane re-staged the record; the class-D mode defect is fixed here — both llama rows run the card row), and the no-switch
    # family's rows are IN SUPPORT (one render, nothing to authorise)
    assert M.STEER_ROW_HOLDS == {} and [r["status"] for r in M.STEER_ROWS] == [M.STATUS_LAUNCH] * len(M.STEER_ROWS)
    assert [r["index"] for r in M.eligible_rows(M.WAVE_STEER)] == [r["index"] for r in M.STEER_ROWS] == list(range(33, 41))
    for r in M.STEER_ROWS:
        assert r["held_reason"] is None and r["object_sha256"] and r["selection_label"]
        switch = r["model_key"] not in M.STEER_ONLY_MODELS
        assert (r["thinking_mode"] == "on") is switch and bool(r["tuple_name"]) is switch
        assert M.assert_launchable(r)["rows"] == [r["index"]]
        assert r["sampling_mode"] == (M.SAMPLING_THINKING_CARD if switch else M.SAMPLING_CARD)
    pl_steer = M.plan("0" * 40, "primary", max_concurrent=8, wave=M.WAVE_STEER, single_array=True)
    assert pl_steer.rows == list(range(33, 41)) and "--array=33-40%8" in "\n".join(pl_steer.lines)
    # THE ROW-HOLD MECHANISM (E2-12): a (model, cell) whose staged record the driver's provenance mirror refuses is a row hold
    # by name — held with that reason, pinned, refused by the batch boundary, skipped by the plan — and lifts with the entry
    saved_holds = dict(M.STEER_ROW_HOLDS)
    try:
        M.STEER_ROW_HOLDS[("llama8b", M.CELL_ALPHASTEER)] = "held (synthetic): the staged record describes other bytes"
        assert M.steer_row_hold_reason("llama8b", M.CELL_ALPHASTEER) == "held (synthetic): the staged record describes other bytes"
        assert M.steer_row_hold_reason("llama8b", M.CELL_ARDITI) is None and M.steer_row_hold_reason("gemma4e4b", M.CELL_ALPHASTEER) is None
        rows = M._build_steer_rows(33)
        held = [r for r in rows if r["status"] == M.STATUS_HELD]
        assert [(r["model_key"], r["cell"], r["held_reason"]) for r in held] == [("llama8b", M.CELL_ALPHASTEER, "held (synthetic): the staged record describes other bytes")]
        assert held[0]["object_sha256"] and held[0]["sampling_mode"] == M.SAMPLING_CARD
        universe = [dict(r) for r in M.M5T_CELLS[:33]] + rows
        with pytest.raises(SystemExit, match="status=held"):
            M.assert_launchable(held[0], universe)
        assert [r["index"] for r in M.eligible_rows(M.WAVE_STEER, universe)] == [33, 34, 35, 36, 37, 38, 40]
    finally:
        M.STEER_ROW_HOLDS.clear()
        M.STEER_ROW_HOLDS.update(saved_holds)
    # the MECHANISM: withdraw the family's certification and its rows hold again, by name, with no other change
    reason = M.steer_hold_reason("llama8b")
    assert reason is None
    saved = dict(M.RENDER_FACT_CERTIFIED_BY_FAMILY)
    try:
        M.RENDER_FACT_CERTIFIED_BY_FAMILY["llama"] = False
        reason = M.steer_hold_reason("llama8b")
        assert "no CERTIFIED render fact" in reason and "TREE gate, not a design deferral" in reason
        assert M.steer_hold_reason("gemma4e4b") is None
        rows = M._build_steer_rows(33)
        held = [r for r in rows if r["status"] == M.STATUS_HELD]
        assert [r["model_key"] for r in held] == ["llama8b", "llama8b"]
        assert all(r["object_sha256"] and r["held_reason"] == reason for r in held)   # pinned, and still not launchable; the tree gate names itself first
        with pytest.raises(SystemExit, match="status=held"):
            M.assert_launchable(held[0], [dict(r) for r in M.M5T_CELLS[:33]] + rows)
    finally:
        M.RENDER_FACT_CERTIFIED_BY_FAMILY.clear()
        M.RENDER_FACT_CERTIFIED_BY_FAMILY.update(saved)
    # llama's placement is the PRIMARY cluster: the row's constraint says so and the overflow plan refuses it by that constraint.
    # Since the 2026-09-14 re-pin the record binds the overflow ROLE too (the rebake's `ladder_canon/lam1` view is staged on both
    # clusters) — the same rel on both roles; the constraint, not an empty rel, is what keeps the row primary
    llama_as = M.row_for("llama8b", M.CELL_ALPHASTEER, M.SAMPLING_CARD)
    assert llama_as["cluster_constraint"] == M.CLUSTER_CONSTRAINT_PRIMARY
    assert llama_as["bank_view_rel"]["overflow"] == llama_as["bank_view_rel"]["primary"] and llama_as["bank_view_rel"]["primary"].endswith("/ladder_canon/lam1/cells_view/it")


def test_a_malformed_admission_entry_refuses_by_name_rather_than_minting_a_row(tmp_path):
    """P8: a steering row is never launched on an incomplete record. Each required field's absence, a pin that is not
    the OBJECT's 64-hex digest (never a directory digest), and an entry whose own keys disagree with where it is
    filed (keyed by model key AND cell id) refuse BY NAME at load, before any row exists."""
    doc = json.loads(M.STEER_ADMISSION_PATH.read_text())

    def _write(mutate):
        d = json.loads(json.dumps(doc))
        mutate(d["entries"]["gemma4e4b"]["alphasteer"])
        p = tmp_path / f"a{len(list(tmp_path.iterdir()))}.json"
        p.write_text(json.dumps(d))
        return p

    for field in M.STEER_ADMISSION_REQUIRED:
        with pytest.raises(SystemExit, match=f"lacks .*{field}"):
            M.load_steer_admission(_write(lambda e, f=field: e.update({f: None})))
    with pytest.raises(SystemExit, match="not 64-hex"):
        M.load_steer_admission(_write(lambda e: e.update(object_sha256="deadbeef")))
    with pytest.raises(SystemExit, match="keyed by model key AND cell id"):
        M.load_steer_admission(_write(lambda e: e.update(cell="arditi")))
    with pytest.raises(SystemExit, match="keyed by model key AND cell id"):
        M.load_steer_admission(_write(lambda e: e.update(model_key="qwen35_4b")))


def _stage_external_bank(tmp_path, row):
    """A view-shaped EXTERNAL bank for one steering row: `<view dir>/<cell dir>/object.pt` of the arm's real kind with
    the lane's identity sidecar, plus the VIEW_MANIFEST the check requires — the shape the AlphaSteer rebake's
    `ladder_canon/lam<k>/cells_view/<view>` and the Arditi lane's `strength/cells_view/<view>` carry on the cluster."""
    import torch
    from sisteer.capeval.petri import registry_adapter as ra
    view = tmp_path / row["bank_view_rel"]["primary"]
    d = view / row["bank_cell_dir"]
    d.mkdir(parents=True, exist_ok=True)
    kind = ra.ARM_OBJECT_KINDS[row["cell"]]
    # `positions` in the blob is the artifact's own FIT leg and must agree with the sidecar's positions_at_fit (the
    # explicit-cell registry's verdict) — both lanes fit PREFILL-only, which is what the real objects record
    # the blobs' own deployment-shape keys, as the REAL objects record them (read off the primary cluster 2026-09-13:
    # arditi `steer_positions: prefill_all, site: pre`; alphasteer `steer_positions: prefill, site: pre`; NEITHER carries
    # a `positions` key — the served-mode stamp derives from steer_positions, defense_cells.SERVED_MODE_BY_STEER_POSITIONS)
    blob = {"kind": kind, "base_scale": 0.05, "rs": {8: torch.full((4,), 0.1)}, "steer_positions": "prefill_all",
            "site": "pre"} if kind == "static_r" else {
        "kind": kind, "layer": 8, "base_scale": 1.0, "w": torch.zeros(4, 4), "steer_positions": "prefill", "site": "pre"}
    torch.save(blob, d / "object.pt")
    (d / "fit_provenance.json").write_text(json.dumps({
        "resolved_model_id": "synthetic", "resolved_revision": "0" * 40, "chat_template_hash": "0" * 16,
        "positions_at_fit": "prefill", "writer": "synthetic@" + "0" * 40}))
    (view / "VIEW_MANIFEST.json").write_text(json.dumps({"target": row["model_key"], "view": row["view"],
                                                         "entries": [{"cell_id": row["cell"]}]}))
    return view, d / "object.pt"


@pytest.mark.parametrize("cell", ["alphasteer", "arditi"])
def test_the_w4_steering_rows_serve_their_own_overlay_and_external_bank_end_to_end(tmp_path, cell):
    """The chain the six launchable rows will run, on a synthetic bank: the row's OWN overlay pair lists exactly
    `[none, <cell>]`, the standup's REAL `select_cells` admits the dry run's `--cell`, the REAL registry resolves the
    object from the external bank (the cell parses as a Class B id, its arm maps to the blob's stored kind), and the
    launcher's external bank check ACCEPTS the pin and REFUSES a byte that moved."""
    import dataclasses

    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.petri.registry_adapter import traintest_cell_source
    mt = _load("_m5t_steer_mt", "scripts/sdfbase/mt_serving_up.py")
    row = M.row_for("gemma4e4b", cell, M.SAMPLING_THINKING_CARD)
    runs, _ = _dry_lines(tmp_path / "dry", rows=[row])
    text = runs[0][1]
    standup = [l for l in text.splitlines() if l.startswith("STANDUP:")][0].split()[1:]
    assert [standup[i + 1] for i, t in enumerate(standup) if t == "--cell"] == [cell]
    assert [standup[i + 1] for i, t in enumerate(standup) if t == "--expect-object-sha256"] == [f"{cell}={row['object_sha256']}"]
    # the row's OWN overlay pair — never the base wave's six-cell one (the external bank stages this cell alone).
    # Read off the launcher's OWN resolver (m5t_resolve_config), the function the job runs before the standup.
    cfg_path = REPO / f"configs/sdfbase/math500/sdfbase_full22_gemma4e4b.math500_{cell}.yaml"
    man_path = REPO / f"configs/capeval/manifests/petri_full22_gemma4e4b.math500_{cell}.yaml"
    resolve = subprocess.run(["bash", "-c",
                              f'set -u; cd {shlex.quote(str(REPO))}; f22_fatal(){{ echo "FATAL: $*" >&2; exit 9; }}; '
                              f'f22_log(){{ :; }}; source {shlex.quote(str(M5T / "m5t_common.sh"))}; '
                              f'eval "$({shlex.quote(sys.executable)} {shlex.quote(str(M5T / "cells.py"))} row --index {row["index"]} --sh)"; '
                              'm5t_resolve_config; echo "CFG=$CFG"; echo "MANIFEST=$MANIFEST"'],
                             capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert resolve.returncode == 0, (resolve.stdout, resolve.stderr)
    assert f"CFG={cfg_path.relative_to(REPO)}" in resolve.stdout and f"MANIFEST={man_path.relative_to(REPO)}" in resolve.stdout
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["explicit_cells"] == ["none", cell]
    spec0 = cm.load_petri_spec(man_path)
    assert list(spec0.cells) == ["none", ra.harness_cell_id(cell)]
    assert mt.assert_manifest_matches_config(spec0, cfg) == {"none": "none", cell: cell}
    # the standup's own cell selection admits it; the base overlay would REFUSE it (the reason the pair exists)
    assert mt.select_cells(cfg["explicit_cells"], [cell]) == ["none", cell]
    base_cfg = yaml.safe_load((REPO / "configs/sdfbase/math500/sdfbase_full22_gemma4e4b.math500.yaml").read_text())
    with pytest.raises(SystemExit, match="never enumerated"):
        mt.select_cells(base_cfg["explicit_cells"], [cell])
    # the REAL registry over the external bank: the cell resolves, its kind is the blob's, the object path is the bank's
    view, obj = _stage_external_bank(tmp_path / "bank", row)
    spec = dataclasses.replace(spec0, serving_launch_flags=[], cells=[ra.harness_cell_id(cell)])
    src = traintest_cell_source(spec, repo_root=REPO, config_path=cfg_path, defenses_dir=view)
    entry = src.get(cell)
    assert entry is not None and entry.kind == ra.ARM_OBJECT_KINDS[cell] and entry.serving_path == "lens"
    assert Path(entry.object_path) == obj and entry.si_id is None and entry.resolved_spec["arm"] == cell
    # the launcher's EXTERNAL bank check: it accepts the staged bytes under their pin and refuses a drifted object
    import hashlib
    sha = hashlib.sha256(obj.read_bytes()).hexdigest()
    core, common = (F22 / "served_thinking.sh"), (M5T / "m5t_common.sh")
    def _check(cell_dir, pin, rel=row["bank_view_rel"]["primary"]):
        return subprocess.run(["bash", "-c",
                               f'set -u; SHARED={shlex.quote(str(tmp_path / "bank"))}; '
                               'f22_fatal(){ echo "FATAL: $*" >&2; exit 9; }; f22_log(){ echo "$*"; }; '
                               f'source {shlex.quote(str(core))}; st_external_bank_view_check '
                               f'{shlex.quote(rel)} {shlex.quote(cell_dir)} {shlex.quote(pin)}'],
                              capture_output=True, text=True)
    ok = _check(row["bank_cell_dir"], sha)
    assert ok.returncode == 0 and "external bank view OK" in ok.stdout, (ok.stdout, ok.stderr)
    bad = _check(row["bank_cell_dir"], "f" * 64)
    assert bad.returncode == 9 and "!= the admission record's pin" in bad.stderr, (bad.stdout, bad.stderr)
    missing = _check("nosuchcell", sha)
    assert missing.returncode == 9 and "lacks nosuchcell/object.pt" in missing.stderr, missing.stderr
    absent_view = _check(row["bank_cell_dir"], sha, rel="exp/objects/nowhere/cells_view/x")
    assert absent_view.returncode == 9 and "does not exist on this cluster" in absent_view.stderr
    # the shell file the launcher dispatches through is the ONE caller of each check (no second copy)
    assert common.read_text().count("st_external_bank_view_check") == 1


# --------------------------------------------------------------------------- #
# E3 — the INSTRUCT-DIRECT ("it-doc") batch-64 re-fit rows on the `it` view (math-eight-arms-extension; decisions-e3)  #
# --------------------------------------------------------------------------- #

def test_e3_the_itdoc_rows_are_the_it_view_family_generated_from_the_record_held_today_and_bound_at_the_bank_door(tmp_path):
    """A SCOPE ADDITION — this matrix carried no `it`-view row: every LoRA row served the base lane's object transferred onto the
    instruct target. Rows 47-52 are the three LoRA arms on gemma (47-49) and qwen (50-52), `view: it`, wave `itdoc` under
    `controls_itdoc/`, the primary sampling row, the short-class geometry, the ext class, generated from the it-doc ADMISSION RECORD
    keyed by (lane, target, view, cell) — the pin is the record's (null while unverified, never a placeholder), the staged id the
    RULED one (`itdoc64__` / `derived_itdoc64__`), the tuples the published / `_lora2` ones by name, the qualification + the
    cross-model pattern verbatim; every row is HELD today with the record's reason — its entry's own, or the record's MATH-500
    INSTRUMENT hold once the entry's bank is bound (qwen's control + difference since 2026-09-13, gemma's since 2026-09-14) — so
    `plan --wave itdoc` refuses, the batch boundary refuses, the dry stub prints HELD and no argv; each row has its OWN overlay pair
    cut from the `it` campaign config; the bank
    door dispatches an it-doc row to `st_itdoc_bank_check` (the record binding, then the byte check) and the sbatch runs the driver's
    provenance mirror right after it."""
    import itdoc_bind as B
    adm = M.load_itdoc_admission()
    rows = M.ITDOC_ROWS
    assert [r["index"] for r in rows] == list(range(47, 53)) and M.ITDOC_MODELS == ("gemma4e4b", "qwen35_4b") and M.ITDOC_CELLS == (M.CELL_LORA_CTL2, M.CELL_LORA_MIX, M.CELL_OBJECT)
    assert [(r["model_key"], r["cell"]) for r in rows] == [(k, c) for k in M.ITDOC_MODELS for c in M.ITDOC_CELLS]
    assert M.OUT_ROOT_PREFIX_BY_WAVE[M.WAVE_ITDOC] == "controls_itdoc" and M.OBJECT_SOURCE_ITDOC == "itdoc" == B.OBJECT_SOURCE_ITDOC
    pins = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())
    for r in rows:
        e = adm["entries"][(adm["lane"], r["model_key"], M.VIEW_IT, r["cell"])]
        assert set(r) - M.PUBLISHED_KEYS == M.EXTENSION_KEYS | M.ITDOC_KEYS
        assert r["wave"] == M.WAVE_ITDOC and r["view"] == M.VIEW_IT and r["object_source"] == M.OBJECT_SOURCE_ITDOC and r["lane"] == adm["lane"] == "itdoc_eb64"
        # EVERY MATH-500 it-view row is HELD — by its entry's reason, or (the two qwen objects the GSM8K wave serves) by the record's
        # INSTRUMENT hold: the E3 MATH-500 legs wait for their own go; the GSM8K wave alone is authorized
        assert r["status"] == M.STATUS_HELD and r["held_reason"] == (e["hold_reason"] or adm["doc"]["instrument_holds"]["math500"]) and r["held_reason"]
        assert r["object_generation"] == "batch64"
        assert r["fitted"] and r["sampling_mode"] == M.SAMPLING_THINKING_CARD and r["thinking_mode"] == "on" and r["extension"] is True and r["log_requests"] is True
        assert r["object_sha256"] == e["object_sha256"] and (r["object_sha256"] is None or len(r["object_sha256"]) == 64)
        assert r["staged_id"] == e["staged_id"] == B.STAGED_PREFIX_BY_KIND[r["object_kind"]] + r["cell"] and r["bank_sidecar"] == B.SIDECAR_BY_KIND[r["object_kind"]]
        assert r["object_kind"] == ("derived" if r["cell"] == M.CELL_OBJECT else "fitted")
        # the row's bank binding is the RECORD's: both models' control + difference (qwen's since the 4e200b68 roster + pins;
        # gemma's since the roster and the rebuilt `it` views of 2026-09-14 — until then this pinned the two qwen entries alone),
        # never the batch-64 constitution (the view serves the batch-8 object)
        bound = B.bank_complete(e["bank"])
        assert bound is ((r["cell"] != M.CELL_LORA_MIX)), (r["model_key"], r["cell"])
        assert (r["bank_view_rel"] is not None) is bound and (r["bank_cell_dir"] == r["cell"]) is bound
        if bound:
            assert r["bank_view_rel"] == {"primary": f"exp/objects/sdfbase/full22/{r['model_key']}/cells_view/it", "overflow": f"exp/objects/sdfbase/full22/{r['model_key']}/cells_view/it"}
        assert r["job_id"] == f"m5t-{r['model_key']}-it-{M.cell_short(r['cell'])}" and r["out_root"] == f"controls_itdoc/{r['model_key']}/{r['cell_slug']}/thinking_card"
        assert (r["geometry_eval"], r["geometry_L"]) == M.GEOMETRY_BY_CLASS[M.PROMPT_CLASS_SHORT] and r["time_limit"] == M.TIME_EXT and r["budget_class"] == M.CELL_NONE
        assert r["tuple_name"] == (M.tuple_name(r["model_key"], M.SAMPLING_THINKING_CARD, r["cell"])) and r["tuple_name"] in C.AUTHORISED_OOS_EXPERIMENTS
        assert r["cell"] in C.AUTHORISED_OOS_EXPERIMENTS[r["tuple_name"]].cells       # by-name authorisation is view-blind; the bytes are bound per view by the pin
        assert r["qualification_pattern"] == adm["doc"]["qualification_pattern"] and (r["qualification"] or r["object_sha256"] is None)
        assert r["stale_objects"] and all(B.is_sha256(k) for k in r["stale_objects"]) and pins[r["model_key"]]["itdoc__" + M.CELL_LORA_MIX]["object_sha256"] in r["stale_objects"]
        if r["object_kind"] == "derived":
            assert r["parents_expected"] is None or set(r["parents_expected"]) == {M.CELL_LORA_MIX, M.CELL_LORA_CTL2}
        # the same (model, cell, mode) sits on the main view: row_for disambiguates by view; the main row is what it was
        main = M.row_for(r["model_key"], r["cell"], M.SAMPLING_THINKING_CARD)
        assert main["view"] == M.VIEW_MAIN and main["index"] < 47 and M.row_for(r["model_key"], r["cell"], M.SAMPLING_THINKING_CARD, view=M.VIEW_IT) is r
    q = M.row_for("qwen35_4b", M.CELL_LORA_MIX, M.SAMPLING_THINKING_CARD, view=M.VIEW_IT)
    assert q["object_sha256"].startswith("80b75aae") and q["superseded"]["object_sha256"].startswith("0018b32b") and "BUDGET-LIMITED" in q["qualification"]
    assert M.row_for("qwen35_4b", M.CELL_OBJECT, M.SAMPLING_THINKING_CARD, view=M.VIEW_IT)["object_sha256"].startswith("5e9b0009")
    assert M.row_for("gemma4e4b", M.CELL_OBJECT, M.SAMPLING_THINKING_CARD, view=M.VIEW_IT)["object_sha256"].startswith("8b7d1030")   # built + staged 20:08Z; bank bound 2026-09-14; held here by the instrument hold
    # a steering row's `view` names the object BANK's view (`it`, `ar_r0p05`): keyed `main`, so the launcher's positional form still resolves it
    assert M.row_for("gemma4e4b", M.CELL_ALPHASTEER, M.SAMPLING_THINKING_CARD)["index"] == 33 and M.key_view(M.STEER_ROWS[0]) == M.VIEW_MAIN
    # the counts carry the it-doc pins by the FULL tuple
    c = M.counts()
    assert set(c["itdoc_pins"]) == {f"itdoc_eb64/{r['model_key']}/it/{r['cell']}" for r in rows} and c["itdoc"]["rows"] == 6 and c["itdoc"]["statuses"] == {"held": 6}
    # HELD everywhere: the plan, the batch boundary, the row printer, the dry stub (HELD line, no argv), the dry run's HELD_ITDOC line
    with pytest.raises(SystemExit, match="no launchable row"):
        M.plan(SHA0 if "SHA0" in globals() else "0" * 40, "cluster_a", max_concurrent=6, wave=M.WAVE_ITDOC)
    with pytest.raises(SystemExit, match="no launchable row|assert-eligible REFUSED"):      # the whole wave is held: the wave-level refusal fires first
        M.assert_launchable(q)
    with pytest.raises(SystemExit, match="HELD / deferred row"):                              # a no-wave selector naming a held index refuses by name
        M.plan("0" * 40, "primary", max_concurrent=6, rows="51")
    sh = M.row_sh(q)
    assert "M5T_STATUS=held" in sh and "M5T_VIEW=it" in sh and "M5T_OBJECT_SOURCE=itdoc" in sh and "M5T_LANE=itdoc_eb64" in sh and "M5T_OBJECT_KIND=fitted" in sh
    assert f"M5T_STAGED_ID=itdoc64__{M.CELL_LORA_MIX}" in sh and "M5T_ITDOC_ADMISSION=configs/sdfbase/itdoc_admission.json" in sh and "BUDGET-LIMITED" in sh
    assert "M5T_HELD_REASON='the it view serves the BATCH-8 constitution object (itdoc__constv2-mix-full__lora_r4_ce, 0018b32b" in sh
    runs, _ = _dry_lines(tmp_path, rows=[q])
    text = runs[0][1]
    held = [l for l in text.splitlines() if l.startswith("HELD: ")]              # the launcher logs its `matrix row` line first, then HELD
    assert held and held[0].startswith("HELD: index=51 job_id=m5t-qwen35_4b-it-mixlora wave=itdoc reason=the it view serves the BATCH-8 constitution object"), text[-600:]
    assert "DRIVER:" not in text and "STANDUP:" not in text
    DRY = _load("_m5t_dry_e3", "scripts/sdfbase/math500/m5t_dry_run.py")
    r = subprocess.run([sys.executable, str(M5T / "m5t_dry_run.py"), "--stage", "pre", "--only", "qwen35_4b", M.CELL_LORA_MIX, M.SAMPLING_THINKING_CARD,
                        "--only-view", "it", "--tmp", str(tmp_path / "t"), "--python", "/venv/bin/python"], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}"})
    assert r.returncode == 0 and "VERDICT: PASS stage=pre" in r.stdout, r.stdout[-1200:] + r.stderr[-800:]
    assert "HELD_ITDOC: row=51 (lane itdoc_eb64, target qwen35_4b, view it, cell constv2-mix-full__lora_r4_ce) staged_id=itdoc64__constv2-mix-full__lora_r4_ce pin=80b75aae2cef" in r.stdout
    assert "DRIVER:" not in r.stdout and "FRESH_TABLE" not in r.stdout
    # the sbatch positional grammar: the optional trailing view names the it-view row; without it the MAIN row resolves (unchanged)
    src = (M5T / "math500_thinking.sbatch").read_text()
    assert 'VIEW_ARG="${6:-main}"' in src and 'ROW_ARGS=("$KEY_ARG" "$CELL_ARG" "$SAMPLING_ARG" "$VIEW_ARG")' in src and 'FITTED_ARG="${5:?fitted 0|1}"' in src
    assert 'if [ "$M5T_STATUS" = held ]; then printf' in src and "\nm5t_bank_view\nm5t_external_mirror_check\n" in src
    common = (M5T / "m5t_common.sh").read_text()
    assert '--view "${4:-main}"' in common and common.count("st_itdoc_bank_check ") == 1 and common.count("st_external_bank_view_check ") == 1
    assert "m5t_external_mirror_check(){" in common and "resolve_training_mirror" in common
    # the it-doc rows' OWN overlay pairs, cut from the `it` campaign config: explicit_cells [none, <cell>]; the base overlays unchanged
    for r in rows:
        cfg = yaml.safe_load((REPO / f"configs/sdfbase/math500/sdfbase_full22_{r['model_key']}_it.math500_{r['cell_slug']}.yaml").read_text())
        campaign = yaml.safe_load((REPO / f"configs/sdfbase_full22_{r['model_key']}_it.yaml").read_text())
        assert cfg["explicit_cells"] == [M.CELL_NONE, r["cell"]] and {k: v for k, v in cfg.items() if k != "explicit_cells"} == {k: v for k, v in campaign.items() if k != "explicit_cells"}
        man = yaml.safe_load((REPO / f"configs/capeval/manifests/petri_full22_{r['model_key']}_it.math500_{r['cell_slug']}.yaml").read_text())["petri"]
        assert man["cells"] == [M.CELL_NONE, M.HARNESS_BY_CELL[r["cell"]]]
    resolve = subprocess.run(["bash", "-c", f'set -u; cd {shlex.quote(str(REPO))}; f22_fatal(){{ echo "FATAL: $*" >&2; exit 9; }}; f22_log(){{ :; }}; '
                              f'source {shlex.quote(str(M5T / "m5t_common.sh"))}; eval "$({shlex.quote(sys.executable)} {shlex.quote(str(M5T / "cells.py"))} row --index 51 --sh)"; '
                              'm5t_resolve_config; echo "CFG=$CFG"; echo "TAG=$TAG"'], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert resolve.returncode == 0 and "CFG=configs/sdfbase/math500/sdfbase_full22_qwen35_4b_it.math500_constv2-mix-full--lora_r4_ce.yaml" in resolve.stdout and "TAG=qwen35_4b_it" in resolve.stdout
    # the bank door on a SYNTHETIC bank: the it-doc branch binds through the record (refuses the batch-8 entry by lane name) and passes a consistent one
    import hashlib
    shared = tmp_path / "shared"
    view = shared / "exp/objects/sdfbase/full22/qwen35_4b/cells_view/it"
    cells = shared / "exp/objects/sdfbase/full22/qwen35_4b/cells"
    (cells / ("itdoc__" + M.CELL_LORA_MIX)).mkdir(parents=True)
    old = b"the superseded object"
    (cells / ("itdoc__" + M.CELL_LORA_MIX) / "object.pt").write_bytes(old)
    view.mkdir(parents=True)
    os.symlink(Path("../../cells") / ("itdoc__" + M.CELL_LORA_MIX), view / M.CELL_LORA_MIX)
    (view / "VIEW_MANIFEST.json").write_text(json.dumps({"target": "qwen35_4b", "view": "it", "entries": [
        {"cell_id": M.CELL_LORA_MIX, "staged_id": "itdoc__" + M.CELL_LORA_MIX, "object_sha256": hashlib.sha256(old).hexdigest()}]}))
    door = subprocess.run(["bash", "-c", f'set -u; cd {shlex.quote(str(REPO))}; SHARED={shlex.quote(str(shared))}; REPO_DIR={shlex.quote(str(REPO))}; CPY={shlex.quote(sys.executable)}; CLUSTER=cluster_a; '
                           'f22_fatal(){ echo "FATAL: $*" >&2; exit 9; }; f22_log(){ echo "LOG $*"; }; '
                           f'source {shlex.quote(str(F22 / "served_thinking.sh"))}; source {shlex.quote(str(M5T / "m5t_common.sh"))}; '
                           f'eval "$({shlex.quote(sys.executable)} {shlex.quote(str(M5T / "cells.py"))} row --index 51 --sh)"; '
                           f'M5T_BANK_VIEW_REL_PRIMARY=exp/objects/sdfbase/full22/qwen35_4b/cells_view/it; M5T_BANK_CELL_DIR={M.CELL_LORA_MIX}; m5t_bank_view'],
                          capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert door.returncode == 9 and "it-doc bank binding REFUSED for (lane itdoc_eb64, target qwen35_4b, view it, cell constv2-mix-full__lora_r4_ce)" in door.stderr
    # the refusal names the LANE of the staged id the bank entry carries (itdoc_bind.OTHER_LANE_BY_PREFIX); the record's stale-object LABEL
    # ("superseded direct-fit object") rides only when the bytes hash to the real 0018b32b… parent, which a synthetic blob never does
    assert "the batch-8 direct-fit lane" in door.stderr and "the superseded parent" in door.stderr and "external bank view OK" not in door.stdout


# --------------------------------------------------------------------------- #
# arditi-qwen-all-positions — the ALL-POSITIONS Arditi arm (row 53, wave `allpos`)  #
# --------------------------------------------------------------------------- #

#: the ONE outcome word this arm's operating point may carry, VERBATIM. Nothing anywhere may imply the selection rule
#: chose c = 0.3: it returns c = 0.2 at +5.04 pp, and 0.3 at +10.26 pp is over the 10 pp budget.
ALLPOS_OUTCOME = "user_override_over_budget_2026-09-16"
ALLPOS_SHA = "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2"
ALLPOS_LEG_KEY = "qwen35_4b/arditi_allpos/thinking_card@it"
ALLPOS_BANK_VIEW_REL = "exp/objects/arditi_allpos/qwen35_4b/ladder/cells_view/ar_c0p3"
#: the engine the leg's LAUNCH RECORD names (standup_jobs.jsonl `engine` of job 544616): what the stamp must derive from the pulled store
ALLPOS_LAUNCH_ENGINE = "lora_native"


def _m5t_pins_record(path: Path, engines: dict | None = None, *, legs: dict | None = None) -> Path:
    """A SYNTHETIC store-pins record (`_meta` + optional leg -> tree entries + an engine block naming `engines`, leg key -> lens | lora_native,
    the way stamp_store_pins_engines.py writes it); `engines` None = a record that knows no engine. Placeholder trees, never a real record."""
    doc = {"_meta": {"schema": "test_store_pins/2", "note": "synthetic"}, **(legs or {})}
    if engines is not None:
        doc["_meta"]["serving_engine_by_leg"] = {k: {"engine": e, "producer_tree_sha": (legs or {}).get(k, "1" * 40), "pulled_store": "pull_x/" + k, "readings": [], "stamped": "2026-01-01"}
                                                 for k, e in engines.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))
    return path


def _m5t_allpos_engine_entry() -> dict | None:
    """The checkout record's `_meta.serving_engine_by_leg` entry for the all-positions leg — None while the record ADMITS the leg without an
    engine (the stamp has not run on this tree: the launcher then refuses the leg by name), the entry once it has."""
    doc = json.loads(Path(M.STORE_PINS_PATHS[0]).read_text())
    assert ALLPOS_LEG_KEY in doc, "the record no longer admits the all-positions leg"
    return ((doc.get("_meta") or {}).get("serving_engine_by_leg") or {}).get(ALLPOS_LEG_KEY)


def _m5t_allpos_stamped_copy(path: Path, engine: str = ALLPOS_LAUNCH_ENGINE) -> Path:
    """A SYNTHETIC stamped twin of the REAL record: its leg -> tree entries verbatim, plus the engine entry the stamp script will write for
    the all-positions leg once its store is pulled (placeholder provenance) — the tests' seam for the launcher's post-stamp mechanics."""
    doc = json.loads(Path(M.STORE_PINS_PATHS[0]).read_text())
    block = dict(doc["_meta"].get("serving_engine_by_leg") or {})
    block[ALLPOS_LEG_KEY] = {"engine": engine, "producer_tree_sha": doc[ALLPOS_LEG_KEY], "pulled_store": "pull_x/" + ALLPOS_LEG_KEY, "readings": [], "stamped": "2026-01-01"}
    doc["_meta"]["serving_engine_by_leg"] = block
    path.parent.mkdir(parents=True, exist_ok=True)                 # the walker's dry root may not exist yet (the sbatch makes its own stub root)
    path.write_text(json.dumps(doc, indent=1))
    return path


def _assert_allpos_m5t_plan(plan: list, raw: str, row: dict, reason: str) -> None:
    """The plan the sbatch prints for the row when its pin leaves the kind-aware default to select the native slot — `reason` is why the row
    is unpinned (`unpinned:no-store`: no store on record; `store:native(pins)`: the record names a native store)."""
    def line(tag):
        found = [l for l in plan if l.startswith(tag + ":")]
        assert len(found) == 1, (tag, plan)
        return found[0]
    assert f"serving path: not pinned by the row ({reason};" in raw
    assert "[serving-path] AUTO-selected lora_native: 1 of 1 steered cells natively servable; skipped by name: none" in raw
    assert line("SERVING_PATH") == f"SERVING_PATH: path=lora_native pinned_by_row=none reason={reason} selection=auto"
    assert line("LAUNCH_CELLS") == "LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none arditi_allpos"
    assert line("WAVE").startswith("WAVE: wave=allpos status=launch view=it budget_class=none staged_id=ar_c0p3__arditi_allpos")
    standup, driver = line("STANDUP"), line("DRIVER")
    assert [standup.split()[i + 1] for i, t in enumerate(standup.split()) if t == "--cell"] == ["arditi_allpos"]
    assert f"--expect-object-sha256 arditi_allpos={ALLPOS_SHA}" in standup          # the standup door's pin
    assert f"--expect-object-sha256 {ALLPOS_SHA}" in driver                          # the driver door's, a separate boundary
    assert f"{M.FLAG_LENS_LIVENESS} {M.LENS_LIVENESS_STEER}" in driver               # the steer liveness gate rides it
    assert f"{M.FLAG_ALLOW_OOS_THINKING}" in driver and "--thinking on " in driver
    assert "configs/sdfbase/math500/sdfbase_full22_qwen35_4b_it.math500_arditi_allpos.yaml" in driver
    assert "controls_arditi_allpos/qwen35_4b/arditi_allpos/thinking_card" in driver
    assert f"OUT OF SUPPORT: tuple={row['tuple_name']} pin={ALLPOS_SHA} served_positions_expected=all" in "\n".join(plan)
    assert line("GEOMETRY").startswith("GEOMETRY: capability:gen_long_math 17664 x 42 cap 16384")


def _allpos_grammar_landed() -> bool:
    """Whether THIS tree's serving vocabulary admits the all-positions cell id (the Class B registration; landed 2026-09-17)."""
    try:
        C.parse_launcher_cell(M.CELL_ARDITI_ALLPOS)
    except C.CellNameError:
        return False
    return True


def test_the_all_positions_row_is_its_own_wave_appended_and_moves_no_existing_index(tmp_path, monkeypatch):
    """arditi-qwen-all-positions, the structural half: row 53 is ONE row in a wave of its own, APPENDED after every
    existing row, generated from the SAME P8 admission record as the steer rung. Everything above it is where it was —
    every index, every wave slice, every wave's row count, every out root and every job id — which is the whole reason
    the arm is not a third cell of `STEER_ROWS` (that would have shifted every later Slurm array id, out root and pin).
    The rung enumerates `WAVE_STEER_CELLS`; `STEER_CELLS` is the steer VOCABULARY and carries the new cell too."""
    ap = M.ALLPOS_ROWS
    assert len(ap) == 1 and [r["index"] for r in ap] == [53] and M.M5T_CELLS[53:] == ap and len(M.M5T_CELLS) == 54
    # nothing above moved
    assert M.M5T_CELLS[:24] == M.BASE_ROWS and M.M5T_CELLS[24:33] == M.EXTENSION_ROWS
    assert M.M5T_CELLS[33:41] == M.STEER_ROWS and M.M5T_CELLS[41:47] == M.LLAMA_ROWS and M.M5T_CELLS[47:53] == M.ITDOC_ROWS
    assert [r["index"] for r in M.STEER_ROWS] == list(range(33, 41)) and [r["index"] for r in M.ITDOC_ROWS] == list(range(47, 53))
    assert [(r["model_key"], r["cell"]) for r in M.STEER_ROWS] == [(k, c) for k in M.STEER_MODELS for c in M.WAVE_STEER_CELLS]
    assert M.WAVE_STEER_CELLS == ("alphasteer", "arditi") and M.STEER_CELLS == M.WAVE_STEER_CELLS + M.ALLPOS_CELLS
    assert M.ALLPOS_CELLS == (M.CELL_ARDITI_ALLPOS,) == ("arditi_allpos",) and M.ALLPOS_MODELS == ("qwen35_4b",)
    # the rows 0-23 fixture is untouched by the new wave (the published keys AND their values)
    fixture = json.loads(FIXTURE.read_text())
    assert [{k: r[k] for k in fixture[0]} for r in M.BASE_ROWS] == fixture
    # the wave's own vocabulary: its name, its WAVE-ISOLATED out root prefix (a pull of one subtree carries no other wave)
    assert M.WAVE_ALLPOS == "allpos" and M.OUT_ROOT_PREFIX_BY_WAVE[M.WAVE_ALLPOS] == "controls_arditi_allpos"
    # the last wave OF THE MATRIX; WAVES (over the launcher's universe) names the seed wave after it (a sibling list, never a matrix row)
    assert tuple(dict.fromkeys(r["wave"] for r in M.M5T_CELLS))[-1] == M.WAVE_ALLPOS and M.WAVES[-2:] == (M.WAVE_ALLPOS, M.WAVE_SEEDREP)
    assert M.WAVE_ALLPOS not in M.COMPLETED_WAVES
    assert all(r["out_root"].split("/", 1)[0] == M.OUT_ROOT_PREFIX_BY_WAVE[r["wave"]] for r in M.M5T_CELLS)
    assert len({r["out_root"] for r in M.M5T_CELLS}) == len({r["job_id"] for r in M.M5T_CELLS}) == 54
    # the counts report the wave apart, as they do every other wave
    c = M.counts()
    assert c["allpos"] == M.counts(M.ALLPOS_ROWS) and c["allpos"]["rows"] == 1 and c["allpos"]["waves"] == {"allpos": 1}
    assert c["allpos"]["external_pins"] == {"qwen35_4b/arditi_allpos": ALLPOS_SHA}
    assert c["base"]["rows"] == 24 and c["steer"]["rows"] == 8 and c["itdoc"]["rows"] == 6      # unmoved
    # the planner treats it as a wave like any other, and it is eligible alone. `plan` reads the row's pin off the store-pins
    # record (a lens prefix only when every selected row pins lens), so it is planned over the REAL record's entries under a
    # native engine block (the shape the stamp writes; the checkout's record refuses by leg name until the stamp lands — the
    # served-row test and the store-pins module's gate hold that state)
    assert [r["index"] for r in M.eligible_rows(M.WAVE_ALLPOS)] == [53]
    monkeypatch.setattr(M, "STORE_PINS_PATHS", (_m5t_allpos_stamped_copy(tmp_path / "stamped.json"),))
    kw = dict(max_concurrent=1, wave=M.WAVE_ALLPOS, single_array=True, assert_eligible_record=True)
    pl = M.plan("0" * 40, "primary", **kw)
    assert pl.rows == [53] and "--array=53%1" in "\n".join(pl.lines) and f"--qos={SITE_A['qos']} " in pl.lines[0]
    assert "F22_SERVING_PATH=" not in pl.lines[0]      # unpinned (a native store on record): the kind-aware default selects it in the job
    assert pl.eligibility["excluded"]["published/complete"] == 24 and "allpos" not in json.dumps(pl.eligibility["excluded"])
    # the cluster constraint the record declares (a launch PLACEMENT: the object is staged on both clusters)
    assert ap[0]["cluster_constraint"] == M.CLUSTER_CONSTRAINT_PRIMARY
    with pytest.raises(SystemExit, match="cluster_constraint primary"):
        M.plan("0" * 40, "overflow", max_concurrent=1, wave=M.WAVE_ALLPOS)
    monkeypatch.undo()
    monkeypatch.setenv("SISTEER_CONFIG", str(_SITE_CFG))   # undo() dropped the module's synthetic site too
    if _m5t_allpos_engine_entry() is not None:           # the checkout's record, once stamped: the same line
        assert M.plan("0" * 40, "primary", **kw).lines == pl.lines


def test_the_all_positions_row_carries_the_records_pin_dose_selection_and_fit_provenance_and_never_an_alpha():
    """The row's FACTS, all from the admission entry and none re-typed: the 64-hex pin, the `it` PANEL view with the
    LANE's bank view (`ar_c0p3`) kept apart in `bank_view`, the staged id, the lane root and both cluster roles' bank
    paths, the dose as a RAW COEFFICIENT, and the `selection` + `fit_provenance` blocks VERBATIM — both readings, the
    budget, the ladder cap that identifies the reading, the single-trial flag, the informational `alpha_equiv` and the
    replicate note. The row carries NO alpha-shaped field (`alpha`, `rung`, `lambda`, `ceiling_20pp`): this arm's dose
    is a different unit on a different intervention and may never share an axis with the rung's."""
    row = M.ALLPOS_ROWS[0]
    entry = M.load_steer_admission()[("qwen35_4b", M.CELL_ARDITI_ALLPOS)]
    assert row["model_key"] == "qwen35_4b" and row["cell"] == row["harness_cell"] == row["cell_slug"] == "arditi_allpos"
    assert row["wave"] == M.WAVE_ALLPOS and row["status"] == M.STATUS_LAUNCH and row["held_reason"] is None
    assert row["object_source"] == M.OBJECT_SOURCE_EXTERNAL == "external" and row["object_sha256"] == entry["object_sha256"] == ALLPOS_SHA
    assert row["fitted"] is True and row["route_kind"] == "lens" and row["served_positions_expected"] == "all"
    assert row["sampling_mode"] == M.SAMPLING_THINKING_CARD == M.sampling_mode_for("qwen35_4b") and row["thinking_mode"] == "on"
    assert row["sampling"] == M.SAMPLING_ROWS[("qwen35_4b", M.SAMPLING_THINKING_CARD)]      # the model's thinking_card row
    assert row["expected_stamps"] == ["cell-stable", "always", "thinking_on"] and row["log_requests"] is True
    assert (row["geometry_eval"], row["geometry_L"]) == M.GEOMETRY_BY_CLASS[M.PROMPT_CLASS_SHORT] and row["budget_class"] == M.CELL_NONE
    assert row["si_sha256"] is None and row["time_limit"] == M.TIME_EXT
    assert row["out_root"] == "controls_arditi_allpos/qwen35_4b/arditi_allpos/thinking_card"
    assert row["job_id"] == "m5t-qwen35_4b-it-arditi_allpos"
    # THE PANEL VIEW and the LANE's bank view are kept APART (the GSM8K launcher's spelling)
    assert row["view"] == M.VIEW_IT == M.ALLPOS_PANEL_VIEW == "it" and row["bank_view"] == entry["view"] == "ar_c0p3"
    assert row["staged_id"] == entry["staged_id"] == "ar_c0p3__arditi_allpos" and row["bank_cell_dir"] == entry["bank_cell_dir"] == "arditi_allpos"
    assert row["lane_root"] == entry["lane_root"] == "exp/objects/arditi_allpos"
    assert row["bank_view_rel"] == dict(entry["bank_view_rel"]) == {"primary": ALLPOS_BANK_VIEW_REL, "overflow": ALLPOS_BANK_VIEW_REL}
    assert all(rel.startswith(row["lane_root"]) and rel.endswith("cells_view/" + row["bank_view"]) for rel in row["bank_view_rel"].values())
    # THE DOSE: a raw coefficient, never an alpha — and no alpha-shaped key anywhere on the row
    assert row["coefficient"] == 0.3 and row["dose_unit"] == "raw_coefficient"
    assert row["object_kind"] == "vector_all" and row["phase"] == "both" and row["positions"] == "all" and row["skip_bos"] is False
    assert row["served_layer"] == 12 and row["serving_path"] == "lora_native"
    assert set(row) - M.PUBLISHED_KEYS == M.EXTENSION_KEYS | M.ALLPOS_KEYS
    for alpha_shaped in ("alpha", "rung", "lambda", "ceiling_20pp", "boundary_pick", "alternative_not_pinned"):
        assert alpha_shaped not in row, alpha_shaped
    assert M.ALLPOS_KEYS.isdisjoint({"alpha", "rung", "lambda", "ceiling_20pp"})
    # THE SELECTION, verbatim and whole: both readings, the budget, the cap, the trial count, the dose, alpha_equiv, the replicate note
    sel = row["selection"]
    assert sel == dict(entry["selection"]) and sel["outcome"] == ALLPOS_OUTCOME
    assert (sel["rule_reading_pp"], sel["override_reading_pp"], sel["budget_pp"]) == (5.04, 10.26, 10)
    assert sel["rule_returns"] == "ar_c0p2" != row["bank_view"]        # the rule returned a DIFFERENT point
    assert sel["override_reading_pp"] > sel["budget_pp"] >= sel["rule_reading_pp"]      # over budget, and the rule's is not
    assert sel["ladder_cap"] == 2048 and sel["single_trial"] is True
    assert sel["dose"] == {"value": 0.3, "unit": "raw_coefficient"} and sel["dose"]["value"] == row["coefficient"]
    assert sel["alpha_equiv"]["value"] == 0.10540806493282855 and sel["alpha_equiv"]["informational"] is True
    assert "never the dose" in sel["alpha_equiv"]["note"] and "3.46" in sel["replicate_note"]
    # the labels say OVERRIDE and never that the rule picked this dose; the ladder is single-trial, so no significance word
    assert row["selection_rule"] == entry["selection_rule"] and row["selection_label"] == entry["selection_label"]
    assert "DECLARED OVERRIDE" in row["selection_label"] and "OVER BUDGET" in row["selection_label"]
    assert "ar_c0p2" in row["selection_rule"] and "DECLARED OVERRIDE" in row["selection_rule"]
    label = json.dumps({k: row[k] for k in ("selection", "selection_rule", "selection_label", "dose_evidence")})
    for forbidden in ("p-value", "p <", "significant", "confidence interval"):
        assert forbidden not in label.lower().replace("p-value", "p-value"), forbidden
    assert row["grid_limited"] is False
    # THE FIT PROVENANCE, whole: fitted on the SERVED checkpoint, the realised counts, the shas, and served = fit - 1
    fp = row["fit_provenance"]
    assert fp == dict(entry["fit_provenance"]) and fp["fitted_on_model"] == "Qwen/Qwen3.5-4B"
    assert fp["fit_layer"] == 13 and fp["served_layer"] == 12 == row["served_layer"] == fp["fit_layer"] - 1
    assert fp["realised_counts"] == {"harmful_train": 123, "harmful_val": 31, "harmless_train": 125, "harmless_val": 31,
                                     "note": fp["realised_counts"]["note"]} and "never the nominal" in fp["realised_counts"]["note"]
    assert all(len(fp[k]) == 64 for k in ("direction_sha256", "direction_pt_sha256", "fit_record_sha256"))
    # the dose evidence must be reported WITH the reading that qualifies it
    assert row["dose_evidence"]["over_refusal_pp"] == 10.26 and row["dose_evidence"]["ladder_cap"] == 2048
    assert row["dose_evidence"]["must_report_with"] == ["over_refusal_pp", "selection.outcome", "selection.rule_returns", "selection.ladder_cap"]
    # it is NOT the prefill-only arm, and nothing about it was copied from that arm's row
    arditi = M.row_for("qwen35_4b", M.CELL_ARDITI, M.SAMPLING_THINKING_CARD)
    assert arditi["cell"] == "arditi" and arditi["alpha"] == 0.05 and arditi["view"] == "ar_r0p05"
    assert arditi["object_sha256"] != row["object_sha256"] and arditi["out_root"] != row["out_root"]
    assert "bank_view" not in arditi and arditi["wave"] == M.WAVE_STEER


def test_the_all_positions_row_keys_on_the_it_panel_view_end_to_end():
    """The row KEYS on `it` — its panel view — while the steer rung keeps keying `main` (its `view` names the object bank,
    and keying it apart would empty the panel slot). That one decision has to hold at four places at once or the row
    goes to the wrong config, the wrong store or the wrong leg: `key_view`, the positional `<key> <cell> <mode> [<view>]`
    form, the LEG KEY the store-pins record and the analysis use, and the overlay pair the launcher resolves."""
    row = M.ALLPOS_ROWS[0]
    assert M.key_view(row) == M.VIEW_IT == "it" and M.row_for("qwen35_4b", "arditi_allpos", "thinking_card", view="it") is row
    with pytest.raises(SystemExit, match="not a matrix row"):                       # never reachable as a main-view row
        M.row_for("qwen35_4b", "arditi_allpos", "thinking_card")
    # the steer rung is untouched: a steer row's `view` names its BANK (gemma's AlphaSteer view is literally `it`) and keys main
    for r in M.STEER_ROWS:
        assert M.key_view(r) == M.VIEW_MAIN and "bank_view" not in r
    assert M.row_for("gemma4e4b", M.CELL_ALPHASTEER, M.SAMPLING_THINKING_CARD)["view"] == "it"     # the bank's view, keyed main
    # THE LEG KEY, pinned across the lane: the launcher's mirror and the ANALYSIS' own derivation agree, and the
    # store-pins record admits the leg under EXACTLY this spelling (allpos-store-pins, 2026-09-17), at the ONE 40-hex launch tree the
    # launch record names (job 544616; `_meta` names the record) — never under the bare `…/thinking_card` (an entry nobody reads:
    # KNOWN-BAD), never aliasing the `arditi` row 36's tree, and never pooled with it. The engine entry is the STAMP's to derive from
    # the pulled store (absent until stamp_store_pins_engines.py --launcher math500 runs on this tree; the launch record's engine after)
    assert M.leg_keys(row) == [ALLPOS_LEG_KEY] == ["qwen35_4b/arditi_allpos/thinking_card@it"]
    A = importlib.import_module("tinycap_math500_thinking")
    assert A.row_leg_key(row) == ALLPOS_LEG_KEY and A.leg_key_view(ALLPOS_LEG_KEY) == "it"
    # a SECOND record, the seed wave's — created empty with the code (read_pins_engines refuses a non-file), so the
    # `@s<s>` Arditi legs' engine is read from it once pin_m5t_seedrep.py stamps them; the default record is [0] and untouched
    assert len(M.STORE_PINS_PATHS) == 2 and M.STORE_PINS_PATHS[1].name == "store_pins_qwen_seedrep.json"
    seed_doc = json.loads(Path(M.STORE_PINS_PATHS[1]).read_text())
    assert set(seed_doc) == {"_meta"} and seed_doc["_meta"]["schema"] == "m5t_store_pins/2" and seed_doc["_meta"]["serving_engine_by_leg"] == {}
    doc = json.loads(Path(M.STORE_PINS_PATHS[0]).read_text())
    assert [k for k in doc if "arditi_allpos" in k] == [ALLPOS_LEG_KEY] and ALLPOS_LEG_KEY.rsplit("@", 1)[0] not in doc
    assert re.fullmatch(r"[0-9a-f]{40}", doc[ALLPOS_LEG_KEY]) and doc[ALLPOS_LEG_KEY] != doc["qwen35_4b/arditi/thinking_card"]
    assert doc["_meta"]["trees"]["allpos"] == doc[ALLPOS_LEG_KEY] and doc["_meta"]["source_by_leg"][ALLPOS_LEG_KEY] == "allpos"
    assert "544616" in doc["_meta"]["sources_by_tree"]["allpos"] and "row 53" in doc["_meta"]["label"]
    assert A.load_store_pins(M.STORE_PINS_PATHS[0])[ALLPOS_LEG_KEY] == doc[ALLPOS_LEG_KEY]       # the analysis' loader reads it (`_meta` skipped)
    entry = _m5t_allpos_engine_entry()
    if entry is not None:                                                                        # stamped: the launch record's engine, at the pin
        assert entry["engine"] == ALLPOS_LAUNCH_ENGINE and entry["producer_tree_sha"] == doc[ALLPOS_LEG_KEY], entry
    # the overlay pair the LAUNCHER resolves is the `it` TAG's, read off m5t_resolve_config itself
    out = subprocess.run(["bash", "-c",
                          f'set -u; cd {shlex.quote(str(REPO))}; f22_fatal(){{ echo "FATAL: $*" >&2; exit 9; }}; f22_log(){{ :; }}; '
                          f'source {shlex.quote(str(M5T / "m5t_common.sh"))}; '
                          f'eval "$({shlex.quote(sys.executable)} {shlex.quote(str(M5T / "cells.py"))} row --index 53 --sh)"; '
                          'm5t_resolve_config; echo "TAG=$TAG"; echo "CFG=$CFG"; echo "MANIFEST=$MANIFEST"'],
                         capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert out.returncode == 0, (out.stdout, out.stderr)
    assert "TAG=qwen35_4b_it" in out.stdout
    assert "CFG=configs/sdfbase/math500/sdfbase_full22_qwen35_4b_it.math500_arditi_allpos.yaml" in out.stdout
    assert "MANIFEST=configs/capeval/manifests/petri_full22_qwen35_4b_it.math500_arditi_allpos.yaml" in out.stdout
    # the same dispatch leaves every other row's pair where it was — a steer row on a bank view called `it` included
    for index, want_tag in ((33, "gemma4e4b"), (34, "gemma4e4b"), (51, "qwen35_4b_it"), (5, "qwen35_4b")):
        r2 = subprocess.run(["bash", "-c",
                             f'set -u; cd {shlex.quote(str(REPO))}; f22_fatal(){{ echo "FATAL: $*" >&2; exit 9; }}; f22_log(){{ :; }}; '
                             f'source {shlex.quote(str(M5T / "m5t_common.sh"))}; '
                             f'eval "$({shlex.quote(sys.executable)} {shlex.quote(str(M5T / "cells.py"))} row --index {index} --sh)"; '
                             'm5t_resolve_config; echo "TAG=$TAG"'],
                            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
        assert r2.returncode == 0 and f"TAG={want_tag}" in r2.stdout, (index, r2.stdout, r2.stderr)
    # the row's shell facts carry the key view and the bank view apart
    sh = M.row_sh(row)
    assert "M5T_VIEW=it" in sh and "M5T_KEY_VIEW=it" in sh and "M5T_BANK_VIEW=ar_c0p3" in sh
    assert f"M5T_STAGED_ID=ar_c0p3__arditi_allpos" in sh and "M5T_OBJECT_SOURCE=external" in sh and "M5T_STATUS=launch" in sh
    assert f"M5T_BANK_VIEW_REL_PRIMARY={ALLPOS_BANK_VIEW_REL}" in sh and f"M5T_BANK_VIEW_REL_OVERFLOW={ALLPOS_BANK_VIEW_REL}" in sh
    assert "M5T_KEY_VIEW=main" in M.row_sh(M.STEER_ROWS[0]) and "M5T_BANK_VIEW=''" in M.row_sh(M.STEER_ROWS[0])


def test_the_all_positions_overlay_pair_is_a_single_object_standup_regenerating_byte_identically():
    """Its own overlay pair, cut from the `it` TAG's campaign config, `explicit_cells` EXACTLY `[none, arditi_allpos]`:
    a SINGLE-OBJECT standup. Two objects in one standup would serve a second intervention beside this one under this
    arm's name, and the bank view stages exactly one cell. Generated, never hand-edited (byte-identity), and the
    campaign original it is cut from is untouched but for that one block."""
    gen = _load("_m5t_overlays_allpos", "scripts/sdfbase/math500/make_math500_overlays.py")
    row = M.ALLPOS_ROWS[0]
    cfg_path = REPO / "configs/sdfbase/math500/sdfbase_full22_qwen35_4b_it.math500_arditi_allpos.yaml"
    man_path = REPO / "configs/capeval/manifests/petri_full22_qwen35_4b_it.math500_arditi_allpos.yaml"
    assert gen.allpos_overlay_config("qwen35_4b", "it", "arditi_allpos") == cfg_path
    assert gen.allpos_overlay_manifest("qwen35_4b", "it", "arditi_allpos") == man_path
    assert cfg_path.read_text() == gen.render_allpos_config(row) and man_path.read_text() == gen.render_allpos_manifest(row)
    rc = subprocess.run([sys.executable, str(M5T / "make_math500_overlays.py"), "--check"], capture_output=True, text=True, cwd=REPO)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["explicit_cells"] == [M.CELL_NONE, "arditi_allpos"] and len(cfg["explicit_cells"]) == 2
    man = yaml.safe_load(man_path.read_text())["petri"]
    assert man["cells"] == ["none", "arditi_allpos"] == [M.HARNESS_BY_CELL[c] for c in (M.CELL_NONE, row["cell"])]
    # cut from the `it` TAG's campaign pair, byte for byte but for the cell list (the `main` body is another config)
    campaign = yaml.safe_load((REPO / "configs/sdfbase_full22_qwen35_4b_it.yaml").read_text())
    cman = yaml.safe_load((REPO / "configs/capeval/manifests/petri_full22_qwen35_4b_it.yaml").read_text())["petri"]
    assert {k: v for k, v in cfg.items() if k != "explicit_cells"} == {k: v for k, v in campaign.items() if k != "explicit_cells"}
    assert {k: v for k, v in man.items() if k != "cells"} == {k: v for k, v in cman.items() if k != "cells"}
    assert cfg["defenses_dir"].endswith("cells_view/it") and cfg["out_dir"].endswith("_it")
    main_cfg = yaml.safe_load((REPO / "configs/sdfbase_full22_qwen35_4b.yaml").read_text())
    assert {k: v for k, v in campaign.items() if k not in ("explicit_cells", "out_dir", "private_run", "defenses_dir")} == {
        k: v for k, v in main_cfg.items() if k not in ("explicit_cells", "out_dir", "private_run", "defenses_dir")}
    # the base overlay and the steer pairs never list this cell (its bank stages one object, and it is not a panel LoRA); the ONLY other pairs
    # naming it are the seed wave's three Arditi rows' own pairs on the seed views (`<key>_s<s>.math500_arditi_allpos`, single-object too)
    seed_arditi = {REPO / f"configs/sdfbase/math500/sdfbase_full22_qwen35_4b_{v}.math500_arditi_allpos.yaml" for v in M.SEEDREP_VIEWS}
    for other in sorted((REPO / "configs/sdfbase/math500").glob("sdfbase_full22_*.yaml")):
        if other == cfg_path:
            continue
        if other in seed_arditi:
            assert yaml.safe_load(other.read_text())["explicit_cells"] == [M.CELL_NONE, "arditi_allpos"], other.name
            continue
        assert "arditi_allpos" not in yaml.safe_load(other.read_text())["explicit_cells"], other.name
    # the standup's OWN cell selection admits the pair's cell and refuses a second object in this standup
    mt = _load("_m5t_allpos_mt", "scripts/sdfbase/mt_serving_up.py")
    assert mt.select_cells(cfg["explicit_cells"], ["arditi_allpos"]) == ["none", "arditi_allpos"]
    with pytest.raises(SystemExit, match="never enumerated"):
        mt.select_cells(cfg["explicit_cells"], ["arditi"])                 # the prefill-only arm is NOT in this standup
    with pytest.raises(SystemExit, match="never enumerated"):
        mt.select_cells(yaml.safe_load((REPO / "configs/sdfbase/math500/sdfbase_full22_qwen35_4b.math500.yaml").read_text())["explicit_cells"],
                        ["arditi_allpos"])                                  # ...and it is in no other standup either


def test_the_all_positions_arm_has_its_own_oos_tuple_never_the_rungs_steer_one():
    """The tuple is an AUTHORISATION, not a label, so this arm gets its OWN: `_steer` declares "prefill-only
    all-position objects … the highest rung within the 4 pp over-refusal budget", which is false of this intervention
    three times over (BOTH phases; a raw-coefficient dose; a DECLARED OVER-BUDGET override, not the rule's
    return) — and `_steer`'s cells are shared by three MATH-500 tuples and six GSM8K ones that FINISHED legs record by
    name, so widening it would move a recorded experiment under stores that are done (the reason `_lora2` exists).
    The tuple's registry entry was the SECOND declared external landing, now landed: see the served-row test below."""
    row = M.ALLPOS_ROWS[0]
    assert row["tuple_name"] == "math500_thinking_on_full22_qwen35_4b_steer_allpos"
    assert row["tuple_name"] != C.m5t_steer_tuple_name("qwen35_4b") and not row["tuple_name"].endswith("_steer")
    assert M.tuple_name("qwen35_4b", M.SAMPLING_THINKING_CARD, M.CELL_ARDITI_ALLPOS) == row["tuple_name"]
    # the rung's tuple is UNCHANGED and still names exactly its two cells — the new cell is not in it
    steer = C.AUTHORISED_OOS_EXPERIMENTS[C.m5t_steer_tuple_name("qwen35_4b")]
    assert steer.cells == C.M5T_STEER_CELLS == ("alphasteer", "arditi") and M.CELL_ARDITI_ALLPOS not in steer.cells
    # no EXISTING tuple was widened to take it: the tuples naming the cell are EXACTLY the arm's own three (this one and
    # the GSM8K arm's two caps), and every other tuple is as it was
    own = {row["tuple_name"], C.g8k_allpos_tuple_name("qwen35_4b", 8192), C.g8k_allpos_tuple_name("qwen35_4b", 16384)}
    naming = {name for name, exp in C.AUTHORISED_OOS_EXPERIMENTS.items() if M.CELL_ARDITI_ALLPOS in exp.cells}
    assert naming == own, naming ^ own
    for name, exp in C.AUTHORISED_OOS_EXPERIMENTS.items():
        if name not in own:
            assert M.CELL_ARDITI_ALLPOS not in exp.cells, name
    # no sensitivity rows, and no other target: the refusals name the reason
    for mode in (M.SAMPLING_GENERAL, M.SAMPLING_BATTERY, M.SAMPLING_CARD):
        with pytest.raises(SystemExit, match="thinking_card"):
            M.tuple_name("qwen35_4b", mode, M.CELL_ARDITI_ALLPOS)
    for key in ("gemma4e4b", "nemotron3nano4b", "glm46vflash"):
        with pytest.raises(SystemExit, match=r"runs on \['qwen35_4b'\] only"):
            M.tuple_name(key, M.SAMPLING_THINKING_CARD, M.CELL_ARDITI_ALLPOS)
    # the rung's rows keep their own tuple, whatever this arm does
    for r in M.STEER_ROWS:
        want = None if r["model_key"] in M.STEER_ONLY_MODELS else C.m5t_steer_tuple_name(r["model_key"])
        assert r["tuple_name"] == want, r["job_id"]


def test_the_all_positions_row_is_served_now_that_the_registration_and_its_tuple_have_landed(tmp_path, monkeypatch):
    """THE TWO EXTERNAL LANDINGS, both DECLARED, have LANDED — and the row goes green with no repin and no re-selection.

    (1) The Class B SERVING registration: `arditi_allpos` is in the harness vocabulary on this tree, so the kind-aware
    serving-path selector ADMITS the row's launch cells and selects the native slot ITSELF — the sbatch's own dry run
    SUCCEEDS over the row with nothing handed in, and prints the plan the refusal used to stand in for. (2) The OOS
    tuple's REGISTRY entry, which could not be added before (1): registered, with exactly this arm's two cells, spelled
    by the registry's own function, the launcher's mirror of the name pinned equal to it. The MEASURED reason (2) had
    to wait for (1) is kept at the end and skips by name now that the vocabulary has landed.

    The launcher's DECLARED gap (`cells.PENDING_SERVING_CELLS`) is checked after every positive fact: it is the one
    hand-named declaration in the handshake, so it is the one line that does not clear itself."""
    row = M.ALLPOS_ROWS[0]
    # (1) the vocabulary admits the cell — a bare Class B id, its own harness id, a fitted (lens-route) cell
    parsed = C.parse_launcher_cell(M.CELL_ARDITI_ALLPOS)
    assert parsed.harness_id == M.HARNESS_BY_CELL[M.CELL_ARDITI_ALLPOS] == "arditi_allpos" and parsed.kind == "fitted"
    assert M.CELL_ARDITI_ALLPOS in C.launchable_class_b_ids()
    # the row's pin follows the STORE, now ON RECORD in the store-pins record at its launch tree. Over a record that knows no engine
    # (the tests' seam) the disk alone speaks — an empty wave root is `unpinned:no-store`, no wave root `store-unread`; over the REAL
    # record's entries under a native engine block (the shape the stamp writes) it reads `store:native(pins)` — either way unpinned, so
    # the kind-aware default selects the path
    blind, stamped = _m5t_pins_record(tmp_path / "blind.json"), _m5t_allpos_stamped_copy(tmp_path / "stamped.json")
    assert M.serving_path_pin(row, wave_root=tmp_path, store_pins=[blind]) == (None, "unpinned:no-store")
    assert M.serving_path_for(row, wave_root=tmp_path, store_pins=[blind]) is None and M.serving_path_pin(row, store_pins=[blind])[1] == "unpinned:store-unread"
    assert M.serving_path_pin(row, wave_root=tmp_path, store_pins=[stamped]) == (None, "store:native(pins)") == M.serving_path_pin(row, store_pins=[stamped])
    assert M.launch_cells_for(row) == ["none", "arditi_allpos"]
    # ...and the selector ADMITS those cells and picks the native slot itself. Verbatim, from the launcher's own path,
    # with NO F22_SERVING_PATH handed in: nothing bypassed.
    def dry(rec):
        env = {**os.environ, "M5T_DRY_RUN": "1", "M5T_DRY_S": "42", "M5T_DRY_SHARED": str(tmp_path / "shared"),
               "M5T_DRY_CPY": "/venv/bin/python", "USER": "tester"}
        env.pop("F22_SERVING_PATH", None); env.pop("M5T_STORE_PINS", None)
        if rec is not None:
            env["M5T_STORE_PINS"] = str(rec)                                       # the tests' seam (a local dry run alone)
        r = subprocess.run(["bash", str(M5T / "math500_thinking.sbatch"), "qwen35_4b", "arditi_allpos", "thinking_card",
                            "0" * 40, "1", "it"], capture_output=True, text=True, env=env)
        return r, r.stdout + r.stderr, [l for l in r.stdout.splitlines() if not l.startswith("[")]
    for rec, reason in ((blind, "unpinned:no-store"), (stamped, "store:native(pins)")):
        r, raw, plan = dry(rec)
        assert r.returncode == 0, raw[-1500:]
        assert "REFUSED" not in raw and "DefenseCellError" not in raw
        # the row got that far on its OWN pin, correctly: unpinned, so the default was asked — and the default selects
        _assert_allpos_m5t_plan(plan, raw, row, reason)
    # (2) the tuple's registry entry: REGISTERED, exactly this arm's two cells, the registry's own spelling — and the
    # launcher's mirror of the name pinned equal to it (checked, never trusted)
    assert row["tuple_name"] in C.AUTHORISED_OOS_EXPERIMENTS
    assert C.AUTHORISED_OOS_EXPERIMENTS[row["tuple_name"]].cells == ("none", "arditi_allpos") == C.ALLPOS_CELLS == (M.CELL_NONE,) + M.ALLPOS_CELLS
    assert row["tuple_name"] == C.m5t_allpos_tuple_name("qwen35_4b") == M.tuple_name("qwen35_4b", M.SAMPLING_THINKING_CARD, M.CELL_ARDITI_ALLPOS)
    assert C.ALLPOS_MODELS == M.ALLPOS_MODELS == ("qwen35_4b",) and C.ALLPOS_VIEW == M.key_view(row) == "it"
    # the whole-matrix dry-run walk prints the same plan for the row (the walker hands this row the seam only while the checkout's
    # record is unstamped, and reads the checkout's record itself after — the same `store:native(pins)` line either way)
    runs, _ = _dry_lines(tmp_path / "dry", rows=[row])
    text = runs[0][1]
    assert "SERVING_PATH: path=lora_native pinned_by_row=none reason=store:native(pins)" in text
    assert "LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none arditi_allpos" in text
    assert "WAVE: wave=allpos status=launch view=it budget_class=none staged_id=ar_c0p3__arditi_allpos" in text
    assert f"OUT OF SUPPORT: tuple={row['tuple_name']} pin={ALLPOS_SHA} served_positions_expected=all" in text
    assert "GEOMETRY: capability:gen_long_math 17664 x 42 cap 16384" in text
    # the launcher's DECLARED gap must be EMPTY now: nothing the vocabulary admits may sit in it. It is hand-named in
    # cells.py (named, not derived), so it is the one line of the handshake that does not clear itself — and while it
    # carries a cell, `_dry_lines` still hands that cell's rows an explicit serving path the selector no longer needs.
    assert M.PENDING_SERVING_CELLS == () and PENDING_SERVING_ROWS == set(), (
        f"cells.PENDING_SERVING_CELLS={M.PENDING_SERVING_CELLS!r} still declares a cell the serving vocabulary admits — "
        "the registration landed, so the declaration must go (and with it the F22_SERVING_PATH bypass in _dry_lines)")
    # the CHECKOUT's record, as a real launch reads it: refusing by leg name until the engine stamp lands (never `no-store` — a known
    # store is not relaunched on a guessed engine), the `store:native(pins)` plan after; the run shows the pending stamp (AC-13)
    r, raw, plan = dry(None)
    if _m5t_allpos_engine_entry() is None:
        assert r.returncode == 1 and "refused:record-engine-unknown" in raw and ALLPOS_LEG_KEY in raw and "SERVING_PATH:" not in r.stdout, raw[-1500:]
        with pytest.raises(SystemExit, match="refused:record-engine-unknown"):
            M.serving_path_pin(row, wave_root=tmp_path)
        pytest.xfail("store_pins_default.json admits the all-positions leg without an engine entry — the engine stamp (stamp_store_pins_engines.py "
                     "--launcher math500 over the pulled store) has not run on this tree; the checkout read refuses by leg name")
    assert r.returncode == 0, raw[-1500:]
    _assert_allpos_m5t_plan(plan, raw, row, "store:native(pins)")
    assert M.serving_path_pin(row, wave_root=tmp_path) == (None, "store:native(pins)") == M.serving_path_pin(row)
    # the MEASURED reason (2) could not land before (1): a registry tuple naming a cell the grammar refuses raises
    # `CellNameError` out of the authorisation door for EVERY other qwen35_4b row — the finished steer `arditi` row and a
    # PUBLISHED row both stop resolving. The vocabulary has landed, so this skips by name; the measurement stays as the
    # record of why the two landed together.
    if _allpos_grammar_landed():
        pytest.skip("the vocabulary has landed — the ordering constraint this measures no longer binds")
    with_pending = dict(C.AUTHORISED_OOS_EXPERIMENTS)
    with_pending[row["tuple_name"]] = C.AUTHORISED_OOS_EXPERIMENTS[C.m5t_steer_tuple_name("qwen35_4b")]._replace(
        cells=M.ALLPOS_CELLS)
    door = dict(controls=["math500"], mode="on", sampling_mode=M.SAMPLING_THINKING_CARD, samples=1,
                max_tokens_by_task={"math": M.CAP})
    for cell in (M.CELL_ARDITI, M.CELL_OBJECT):
        assert C.check_out_of_support_thinking_authorised(model="qwen35_4b", cell=cell, **door)
    with mock.patch.object(C, "AUTHORISED_OOS_EXPERIMENTS", with_pending):
        for cell in (M.CELL_ARDITI, M.CELL_OBJECT):
            with pytest.raises(C.CellNameError, match="arditi_allpos"):
                C.check_out_of_support_thinking_authorised(model="qwen35_4b", cell=cell, **door)


def test_a_lens_serving_path_on_the_all_positions_row_is_refused_by_name():
    """The arm is served on the graph-captured NATIVE residual slot in EVERY leg (the record's `serving_path`), and a
    lens leg would measure something else under its name. The row pins no path (its store decides, and there is none),
    so the kind-aware default selects native — and a submitter who exports `F22_SERVING_PATH=lens` over a row that DOES
    pin is refused by name. The rule lives in ONE place for both math launchers, so this test states what this row
    declares and proves the conflict refusal on the mechanism that carries it."""
    row = M.ALLPOS_ROWS[0]
    assert row["serving_path"] == "lora_native" == SP_MOD.SERVING_PATH_NATIVE
    assert row["attestation_expect"]["serving_path"] == "lora_native" and row["attestation_expect"]["op"] == "add_vector"
    assert row["attestation_expect"]["coeff"] == 1.0 and row["attestation_expect"]["positions"] == "all"
    assert row["attestation_expect"]["phase"] == "both" and "lens_reduction_mode" not in row["attestation_expect"]
    # the kind-aware default's input: the cell's kind is natively servable, and nothing pins the lens for it
    assert SP_MOD.steer_cell_kind(row["cell"]) == "vector_all" in SP_MOD.NATIVELY_SERVABLE_KINDS
    assert row["cell"] not in [c for c, k in SP_MOD.STEER_CELL_KINDS.items() if k in SP_MOD.LENS_ONLY_KINDS]
    # the launcher's conflict refusal, on the shell function that carries it: a row that PINS a path refuses a
    # submitter who exports another (the same guard that will hold this row the day its store makes it pin)
    funcs = _m5t_block("m5t_select_serving_path(){", "m5t_init(){")
    alpha = M.row_for("gemma4e4b", M.CELL_ALPHASTEER, M.SAMPLING_THINKING_CARD)       # pins the lens BY KIND
    assert M.serving_path_for(alpha) == "lens"
    script = ('set -u\nf22_log(){ echo LOG "$@"; }\nf22_fatal(){ echo FATAL "$@"; exit 1; }\n'
              f"m5t_py(){{ echo {shlex.quote(sys.executable)}; }}\nREPO_DIR={shlex.quote(str(REPO))}\nM5T_DIR=scripts/sdfbase/math500\n"
              f"F22_WAVE_ROOT=/tmp/m5t-allpos-none\nexport F22_SERVING_PATH=lora_native\n"
              + funcs + f"\nm5t_select_serving_path gemma4e4b {M.CELL_ALPHASTEER} thinking_card\n")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 1 and "FATAL the row pins serving path 'lens'" in r.stdout and "refused by name" in r.stdout


def test_a_malformed_all_positions_admission_entry_refuses_by_name_rather_than_minting_the_row(tmp_path):
    """KNOWN-BAD TWINS, each run against a MUTATED copy of the record through the launcher's own loader
    (`cells.load_steer_admission`, which every row of this wave is generated from): the row is never minted on an
    entry that says something else. A wrong outcome word; an ALPHA-shaped dose (the prefill-only arm's shape, the one
    substitution that would silently put two different units on one axis); a `selection` block without its LADDER CAP
    (at another cap the rule returns a different point, so a reading without it is a reading of nothing); a
    `rule_returns` naming the selected point itself (an override supersedes a DIFFERENT dose); an override reading
    that is not actually over budget; a view that is not the coefficient spelled; a `lens` serving path; an
    incomplete `fit_provenance`; and the record's own required fields. Each refusal NAMES the field."""
    doc = json.loads(M.STEER_ADMISSION_PATH.read_text())
    assert doc["entries"]["qwen35_4b"]["arditi_allpos"]["selection_outcome"] == ALLPOS_OUTCOME

    def _write(mutate) -> Path:
        d = json.loads(json.dumps(doc))
        mutate(d["entries"]["qwen35_4b"]["arditi_allpos"])
        p = tmp_path / f"a{len(list(tmp_path.iterdir()))}.json"
        p.write_text(json.dumps(d))
        return p
    # the loader admits the record as committed (the baseline: these twins fail for their own reason, not a broken harness)
    assert ("qwen35_4b", M.CELL_ARDITI_ALLPOS) in M.load_steer_admission(_write(lambda e: None))
    cases = [
        (lambda e: e.update(selection_outcome="chosen"), "selection_outcome"),                      # a LADDER word
        (lambda e: e.update(selection_outcome="ruled_2026-09-14"), "selection_outcome"),            # the prefill-only arm's word
        (lambda e: e.update(alpha=0.1), "alpha"),                                                   # an alpha-shaped entry
        (lambda e: e.update(dose_unit="residual_norm_fraction"), "dose_unit"),                      # the other arm's unit
        (lambda e: e.update(kind="static_r"), "kind"),
        (lambda e: e.update(phase="prefill"), "phase"),
        (lambda e: e["selection"].pop("ladder_cap"), "ladder_cap"),                                 # a reading with no cap
        (lambda e: e["selection"].pop("single_trial"), "single_trial"),
        (lambda e: e["selection"].update(rule_returns=e["view"]), "rule_returns"),                  # supersedes nothing
        (lambda e: e["selection"].update(override_reading_pp=9.0), "override_reading_pp"),          # not over budget
        (lambda e: e["selection"].update(rule_reading_pp=11.0), "rule_reading_pp"),                 # the rule's is over budget
        (lambda e: e.update(view="ar_r0p3"), "view"),                                               # the ALPHA spelling
        (lambda e: e.update(serving_path="lens"), "serving_path"),                                  # not this arm's route
        (lambda e: e["fit_provenance"].pop("direction_sha256"), "fit_provenance.direction_sha256"),
        (lambda e: e["fit_provenance"].update(served_layer=13), "served_layer"),                    # not fit_layer - 1
        (lambda e: e["attestation_expect"].update(coeff=0.3), "attestation_expect.coeff"),          # the dose is in the norm
    ]
    for mutate, needle in cases:
        with pytest.raises(SystemExit) as exc:
            M.load_steer_admission(_write(mutate))
        assert needle in str(exc.value), (needle, str(exc.value)[:400])
    # a coefficient that disagrees with the block's dose, and the record's own required fields
    with pytest.raises(SystemExit, match="coefficient"):
        M.load_steer_admission(_write(lambda e: e.update(coefficient=0.2)))
    for field in M.STEER_ADMISSION_REQUIRED:
        with pytest.raises(SystemExit, match=f"lacks .*{field}"):
            M.load_steer_admission(_write(lambda e, f=field: e.update({f: None})))
    with pytest.raises(SystemExit, match="not 64-hex"):
        M.load_steer_admission(_write(lambda e: e.update(object_sha256="deadbeef")))
    with pytest.raises(SystemExit, match="keyed by model key AND cell id"):
        M.load_steer_admission(_write(lambda e: e.update(cell=M.CELL_ARDITI)))
    # and the wave is never minted from a record that does not admit the arm at all: refused BY NAME, never a silent skip
    dropped = json.loads(json.dumps(doc))
    dropped["entries"]["qwen35_4b"].pop("arditi_allpos")
    p = tmp_path / "dropped.json"
    p.write_text(json.dumps(dropped))
    with pytest.raises(SystemExit, match="names no entry for .*arditi_allpos.*never exists without it"):
        M._build_allpos_rows(53, M.load_pins(), M.load_steer_admission(p))
