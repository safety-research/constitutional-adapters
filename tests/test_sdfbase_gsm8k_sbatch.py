"""The GSM8K paired-200 paper objects: the frozen matrix through the ONE
view-selection interface (its counts, the holds, TAG for all three view kinds), the overlays, the sbatch shape (the
cluster's hard rules; the checkout prologue byte-identical to the MATH-500 family's modulo the helper filename), the
generated argv (through the sbatch's OWN dry-run path, never re-typed: the explicit cap as statement + setter, the
declared spelling, the rung's subset, the overlap leg), the HELD semantics on every production consumer, model-based
placement, the plan's partition, the ledger sequencing, the attempt seam, the two-stage dry run against the expected
served positions, the dose-view production path over a SYNTHETIC peer tree, and the ladder planner's sizing and
idempotence with the ``classify_store`` interface stubbed. Every count is COMPUTED from ``g8k_cells.counts()``, never
typed. Everything here is SYNTHETIC (``TESTHOST``, placeholder pins, scratch trees); nothing is served.

Integration checks: the declared spelling crosses the standup boundary ONCE (the produced argv through the standup's
REAL digest parser and the fresh table's REAL positions lookup); the rung task namespace is the producer's ROW task,
fed by the REAL ``inspect_rung.classify_store`` over a store built through the producer; the stamp triples are the
contract's, the matrix read by the analysis' REAL reader; no expected refusals; the harness's ``item_subset_sha16``;
no dead ``G8K_*`` env contract; the symlinked offline datasets layout."""
from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "gsm8k",
          REPO / "scripts" / "sdfbase" / "full22", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import g8k_cells as M
import g8k_views as V
from sisteer.capeval.tinycap import contracts as C
from front_door_testlib import MATCHERS, SCAN, token_hits  # noqa: E402
import cluster_site as _cs  # noqa: E402

# Every printed plan line takes its cluster's QoS and log directory from the site config: one synthetic two-cluster site for the
# module (site labels of its own), and the values the printers must read from it.
import tempfile  # noqa: E402
_SITE_ROOT = Path(tempfile.mkdtemp(prefix="g8k-site-"))
_SITE_CFG = _cs.write_site(_SITE_ROOT)
SITE_A, SITE_B = _cs.expect(_SITE_ROOT, "a"), _cs.expect(_SITE_ROOT, "b")


@pytest.fixture(autouse=True)
def _cluster_site(monkeypatch):
    monkeypatch.setenv("SISTEER_CONFIG", str(_SITE_CFG))

#: one tier of the publication scan's token table, matched as the scan matches it (the tokens are never spelled here)
SITE_STORAGE_TOKENS = SCAN.Matcher([r for r in MATCHERS["rows"] if r.tier in ("site-label", "storage")])

G8K = REPO / "scripts/sdfbase/gsm8k"
F22 = REPO / "scripts/sdfbase/full22"
M5T = REPO / "scripts/sdfbase/math500"
GPU_SBATCHES = ["gsm8k_thinking.sbatch", "overlap_off.sbatch", "smoke_not_available.sbatch"]
CPU_SBATCHES = ["dry_run_prod.sbatch", "ladder_plan.sbatch"]
SBATCHES = GPU_SBATCHES + CPU_SBATCHES
BASE_SHA = "004e74cb46c7f3b811edcf6b689dbdd3755927c0"          # the campaign originals' last ruled change (2026-09-19: the abliterated-Qwen
#: auditor-leg drop; before it the native-serving default and the gemma4e4b/it control cells, 2026-09-13). Advanced from
#: 0f0e65039 (the work-branch tip the lanes were cut from, 2026-09-10) on 2026-10-02: every campaign manifest had drifted from that
#: pin by those ruled changes, so the "originals untouched" half of the overlays test was red for weeks; the assertions themselves
#: are unchanged -- a frozen sha the originals must still match beyond the ruled, named allowances.
#: the tree the public release rewrote: a campaign original may differ from its bytes there only by the release's declared
#: text rules (site paths to placeholders, neutral free text, site labels), which configs/SCRUB_MAP.tsv certifies per file;
#: a change that leaves the parsed document equal (comment text) needs no certificate
PRE_RELEASE_SHA = "6801151ed3ba5a62e9fb2ad5d6d01adc8ef82eac"
COUNTS = M.published_counts()                                  # the PUBLISHED wave's acceptance counts (E2: the matrix grew appended waves)


#: the campaign originals may differ from BASE_SHA's in EXACTLY one ruled change -- the multi-turn
#: target cap (attacks.target_max_new_tokens 512 -> the roster's 4,096, the canonical instrument of
#: 2026-09-11) and its comment lines; any other drift still refuses. The overlays' own byte-identity
#: to the generator (--check above) is what keeps them derived from the CURRENT originals.
ALLOWED_ORIGINAL_DRIFT = {("attacks", "target_max_new_tokens"): (512, 4096)}
#: keys ADDED since BASE_SHA (absent then, present now): the canonical best-of-N's climb
#: cap + its sampler rule (`climb_decode: as_drawn`, since 2026-09-17 -- the climb re-draws at the
#: sampler its 512 store records; it was the wave-wide `climb_temperature: 1.0` from 2026-09-11 until then),
#: beside the UNCHANGED base cap bon.max_new_tokens 512
ALLOWED_ORIGINAL_ADDITIONS = {("bon", "climb_cap"): 4096, ("bon", "climb_decode"): "as_drawn",
                              # the over-refusal controls' own cap key (2026-09-11): the ladder root, so the
                              # ruled target cap above moves without moving the controls
                              ("attacks", "controls_max_new_tokens"): 512}


#: list items REMOVED since BASE_SHA (present then, absent now): the obsolete eval harnesses were
#: dropped on 2026-09-12, so every campaign config's `controls` lost wmdp and alpaca_eval
ALLOWED_ORIGINAL_REMOVALS = {("controls",): ("wmdp", "alpaca_eval")}


#: since 2026-09-11 the roster LISTS the matched prompted safety baseline on the `it` views, so the campaign
#: `it` configs carry it (explicit_cells, after every base cell) and their manifests spell it `prompted__safety`
#: (cells). Since 2026-09-12 any OTHER prompted baseline the roster declares on a view rides into that view's
#: campaign config / manifest the same way (`prompted__si_constitution` / `prompted__constitution` on gemma /
#: qwen it + main and nemotron main). A prompted cell added since BASE_SHA is the ONE admitted drift on a
#: manifest and the one admitted `explicit_cells` drift on a config; anything else added fails by name.
IT_PROMPTED_CONFIG_CELL = "prompted__safety_refusal"
IT_PROMPTED_MANIFEST_CELL = "prompted__safety"


def _admit_roster_declared_prompted_cells(rel: str, base: dict, now: dict) -> None:
    """Strip from ``now`` every prompted cell absent at BASE_SHA (a roster-declared prompted baseline: the safety
    cell on the `it` views, an admitted one on any view) so the remaining comparison is the admitted one; a
    non-prompted addition stays and fails by name. On an `it` file the additions FOLLOW every base cell (listed
    after the objects)."""
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
    # the batch-64 instruct-direct lane's two cells landed on the `it` campaign configs with their roster entries
    # — matrix itdoc's rows, admitted here by name on `_it` files only; any other non-prompted addition still fails by name
    batch64 = {"constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "si_constv2ctl2_full__lora_r4_ce", "si_constv2_mix_full__lora_r4_ce_minus_ctl"}
    lane_added = [c for c in added if c in batch64 and "_it." in rel]
    added = [c for c in added if c not in lane_added]
    assert all(c.startswith("prompted__") for c in added), (rel, "a non-prompted cell was added since BASE_SHA", added)
    if rel.endswith("_it.yaml") and added:
        cells_wo_lane = [c for c in cells if c not in lane_added]
        assert cells_wo_lane[:len(base_cells)] == base_cells, (rel, "the it view's prompted cells follow every base cell", cells)
    holder[field] = [c for c in cells if c not in added and c not in lane_added]


def _pre_release_text(rel: str, now_text: str) -> str:
    """The campaign original as the public release found it, once its rewrite since is certified: equal bytes, an equal
    parsed document (a comment-only rewrite), or a configs/SCRUB_MAP.tsv row mapping the pre-release sha256 to today's."""
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
        assert base == now, f"{rel} drifted from BASE_SHA (no drift is allowed on a manifest beyond a roster-declared prompted cell)"
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
#: E2 (math-eight-arms-extension, plan P2 / P8): the default matrix is the frozen 42 PUBLISHED rows (wave `published`, no `wave`
#: key — byte for byte the 8f5f1495 document) + the appended waves `ext` (the eight-arm prompted baselines) and `steer` (the
#: AlphaSteer / Arditi rung). The published SHAPE every acceptance assertion below pins is the published wave's; the appended
#: waves have their own tests (the E2 section at the end)
ALL_ROWS = M.matrix_rows()
ROWS = M.rows_of_wave(M.WAVE_PUBLISHED, ALL_ROWS)              # the frozen 42
EXT_ROWS = M.rows_of_wave(M.WAVE_EXT, ALL_ROWS)
STEER_ROWS = M.rows_of_wave(M.WAVE_STEER, ALL_ROWS)
LAUNCHED = M.launch_rows(ROWS)                                 # the published wave's 38 launched rows
#: the trailing held rows a roster-declared, unscheduled prompted cell adds (none over today's roster; five on a tree
#: that declares `prompted__si_constitution`) — every shape below is 42 scheduled rows + these
UNSCHED = M.unscheduled_held_rows(ROWS)
#: since 2026-09-13 the LIVE roster's `it` views carry the BATCH-64 instruct-direct lane's cells — `constv2ctl2-full__lora_r4_ce` <-
#: `itdoc64__…`, `constv2-mix-full__lora_r4_ce_minus_ctl` <- `derived_itdoc64__…`. They are matrix `itdoc`'s rows
#: (configs/sdfbase/gsm8k/views_itdoc.yaml), never the published matrix's: the live-vs-frozen comparisons below SET THEM ASIDE (and re-index the
#: remainder), so the published matrix's invariant — the frozen 42 lead, the prompted holds trail — is read on the rows it is about.
BATCH64_LANE_PREFIXES = ("itdoc64__", "derived_itdoc64__")


def _without_batch64_lane(rows):
    kept = [dict(r) for r in rows if not str(r.get("staged_id") or "").startswith(BATCH64_LANE_PREFIXES)]
    for i, r in enumerate(kept):
        r["index"] = i
    return kept
PY = "/venv/bin/python"                                        # the client-venv python the argv names (a placeholder)
SHA0 = "0" * 40


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DRY = _load("_g8k_dry_run_under_test", "scripts/sdfbase/gsm8k/g8k_dry_run.py")
LP = _load("_g8k_ladder_plan_under_test", "scripts/sdfbase/gsm8k/ladder_plan.py")
#: the analysis' matrix reader (scripts/tinycap_gsm8k_validators.py) — imported READ-ONLY as the matrix's real downstream
VAL = _load("_g8k_validators_read_only", "scripts/tinycap_gsm8k_validators.py")


@pytest.fixture(scope="module")
def s_by():
    return {(k, r): DRY.resolved_s(k, r) for k in M.MODEL_KEYS for r in M.RUNGS}


def _dry(row, *, rung=M.CAP, s, shared, cluster="cluster_a", repo_dir=REPO, fitted=None, declared=None):
    env = {**os.environ, "G8K_DRY_RUN": "1", "G8K_DRY_S": str(s), "G8K_DRY_SHARED": str(shared), "G8K_DRY_CPY": PY,
           "G8K_DRY_CLUSTER": cluster, "G8K_REPO_DIR": str(repo_dir), "USER": "tester"}
    argv = ["bash", str(repo_dir / "scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch"), row["model_key"], row["view"], row["cell"], str(rung),
            SHA0, str(int(row["fitted"]) if fitted is None else fitted), str(int(row["declared"]) if declared is None else declared)]
    r = subprocess.run(argv, capture_output=True, text=True, env=env, check=False)
    return "\n".join(line for line in r.stdout.splitlines() if not line.startswith("[")), r.returncode, r.stdout + r.stderr


def _line(text, tag):
    found = [line for line in text.splitlines() if line.startswith(tag + ":")]
    assert len(found) == 1, (tag, text)
    return found[0]


#: every launcher that takes a matrix by NAME — the row launcher, the production dry run, the ladder planner and the no-switch
#: smoke. `overlap_off.sbatch` is absent on purpose: the overlap legs belong to the published matrix alone and it reads no
#: matrix positional.
MATRIX_TAKING_LAUNCHERS = ("gsm8k_thinking.sbatch", "dry_run_prod.sbatch", "ladder_plan.sbatch", "smoke_not_available.sbatch")
#: the launchers whose matrix allow-list deliberately LAGS `g8k_cells.MATRICES`, and by which names — EMPTY, and the mechanism is
#: kept rather than the exception. A registered matrix must be accepted by EVERY launcher above, because a matrix accepted by
#: only some of them is a half-registration that reads as working: arditi-allpos-cap first registered its matrix in the two row
#: launchers alone, which would have left a document DECLARING `trigger_threshold: 0.01` whose 16,384 rung could never be
#: PLANNED (`ladder_plan.sbatch … <matrix>` refusing the name) — a leg stuck at the cap with an unrunnable escalation and nothing
#: saying so at launch time. If a future matrix genuinely must not reach one launcher, name it here WITH the reason.
ALLPOS_LAGGING_LAUNCHERS: dict = {}


def _admitted_matrices(launcher: str) -> str:
    """The matrix names a launcher's own `case` admits, in its own spelling (`default|llama|itdoc`) — READ from the file, so a
    refusal message is never re-typed here and a launcher that gains a matrix cannot leave this test asserting the old list."""
    m = re.search(r'case "\$G8K_MATRIX" in ([a-z_|]+)\) ;;', (G8K / launcher).read_text())
    assert m, launcher
    return m.group(1)


# --------------------------------------------------------------------------- #
# the matrix, through the one selector                                          #
# --------------------------------------------------------------------------- #


def test_matrix_regenerates_byte_identically_and_has_the_amended_specs_shape():
    py = sys.executable
    for argv in ([py, str(G8K / "make_gsm8k_matrix.py"), "--check"], [py, str(G8K / "make_gsm8k_overlays.py"), "--check"]):
        r = subprocess.run(argv, capture_output=True, text=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, check=False)
        assert r.returncode == 0, r.stdout + r.stderr
    n_u = len(UNSCHED)
    assert [r["index"] for r in ROWS] == list(range(42 + n_u))
    # the 42 SCHEDULED rows (today's frozen shape) come first; the admitted prompted cells, if the roster declares any, trail
    assert {k: COUNTS[k] for k in ("rows", "held", "launched", "fitted", "fitted_launched", "in_support", "declared",
                                    "expected_refusals", "servable", "contrasts_designed", "contrasts_computable", "n_tags")} == {
        "rows": 42 + n_u, "held": 4 + n_u, "launched": 38, "fitted": 32, "fitted_launched": 28, "in_support": 10, "declared": 4,
        "expected_refusals": 0, "servable": 38, "contrasts_designed": 32, "contrasts_computable": 32, "n_tags": 6}   # 2 / 36 / 30 before
    assert COUNTS["tags"] == ["gemma4e4b_it", "gemma4e4b", "qwen35_4b_it", "qwen35_4b", "nemotron3nano4b", "glm46vflash"]
    scheduled_by_tag = {"gemma4e4b_it": 4, "gemma4e4b": 9, "qwen35_4b_it": 4, "qwen35_4b": 9, "nemotron3nano4b": 8, "glm46vflash": 8}
    assert {t: v["rows"] for t, v in COUNTS["by_tag"].items()} == {t: n + sum(r["tag"] == t for r in UNSCHED) for t, n in scheduled_by_tag.items()}
    assert {t: v["launched"] for t, v in COUNTS["by_tag"].items()} == {t: n - (2 if t.endswith("_it") else 0) for t, n in scheduled_by_tag.items()}
    # the four views.yaml holds are EXACTLY the it view's constitution objects on gemma / qwen, with the reason
    d8 = [r for r in M.held_rows() if r not in UNSCHED]
    assert [(r["tag"], r["cell"]) for r in d8] == [("gemma4e4b_it", "constv2-mix-full__psr_ce_all"), ("gemma4e4b_it", "constv2-mix-full__lora_r4_ce"),
                                                   ("qwen35_4b_it", "constv2-mix-full__psr_ce_all"), ("qwen35_4b_it", "constv2-mix-full__lora_r4_ce")]
    assert all("re-fitted" in r["held_reason"] and "batch 64" in r["held_reason"] for r in d8)
    # every OTHER held row is a roster-declared prompted cell this matrix does not schedule, TRAILING (index >= 42)
    assert M.held_rows() == d8 + UNSCHED and ROWS[:42] == [r for r in ROWS if r not in UNSCHED]
    for r in UNSCHED:
        assert r["index"] >= 42 and r["status"] == "held" and r["held_reason"] == M.unscheduled_held_reason(r["cell"])
        assert M.is_unscheduled_prompted(r["cell"]) and r["route_kind"] == "prompted" and not r["fitted"] and r["cell"] not in M.SCHEDULED_PROMPTED_CELLS
    assert all(r["held_reason"] is None for r in LAUNCHED)
    pins = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())
    from sisteer.capeval.petri import registry_adapter as ra
    for r in ROWS:
        assert r["tag"] == M.tag_for(r["model_key"], r["view"]) and r["job_id"] == f"g8k-{r['model_key']}-{r['view']}-{M.cell_short(r['cell'])}"
        assert r["fitted"] is (r["staged_id"] is not None) and r["route_kind"] == M.route_kind(r["cell"])
        assert r["declared"] is (r["cell"] in M.HEADER_LOCATED_CELLS) and (r["declared"] <= r["fitted"])
        assert r["harness_cell"] == ra.harness_cell_id(r["cell"]) == C.cell_spellings(r["cell"], declared=False).harness
        assert r["cell_slug"] == r["cell"].replace("__", "--")
        # B-P1.1: the ONE standup spelling, derived by the contract's door table — `<cell>@thinking` iff declared
        sp = C.cell_spellings(r["cell"], declared=r["declared"])
        assert (r["standup_cell"], r["standup_harness_cell"]) == (sp.traintest, sp.harness)
        if r["declared"]:
            assert r["standup_cell"] == r["cell"] + C.OOS_STANDUP_CELL_SUFFIX and r["standup_harness_cell"] == r["harness_cell"] + C.OOS_STANDUP_CELL_SUFFIX
            assert r["standup_cell"] == sp.expect_key            # the digest key IS the --cell spelling (variant included)
        else:
            assert (r["standup_cell"], r["standup_harness_cell"]) == (r["cell"], r["harness_cell"])
        # B-P1.3: the stamp triple is the CONTRACT's derivation for the row's class, never a launcher literal
        assert r["expected_stamps"] == list(C.g8k_expected_stamps(fitted=r["fitted"], declared=r["declared"]))
        assert r["sampling_mode"] == "thinking_card" and r["sampling"] == M.SAMPLING_ROWS[r["model_key"]]
        assert r["budget_class"] == M.budget_class(r["cell"]) == (r["cell"] if r["cell"].startswith("prompted__") else "none")   # a prompted cell is its own class
        assert r["cluster_constraint"] == ("primary" if r["model_key"] == "nemotron3nano4b" else "any")
        assert r["time_limit"] == M.time_limit_for(r["model_key"], r["fitted"])
        assert r["out_root"] == f"controls/{r['tag']}/{r['cell_slug']}/thinking_card" and r["rung_out_root"] == r["out_root"] + "__rung16384"
        if r["fitted"]:
            assert r["object_sha256"] == pins[r["model_key"]][r["staged_id"]]["object_sha256"]
            assert r["tuple_names"] == {"8192": M.tuple_name(r["model_key"], declared=r["declared"], rung=8192),
                                        "16384": M.tuple_name(r["model_key"], declared=r["declared"], rung=16384)}
            assert r["served_positions_expected"] == ("response" if r["declared"] else "all")
            if r["declared"]:
                assert tuple(r["expected_stamps"]) == C.expected_stamp_triple(lens=True, arm=C.THINKING_ON, header_located=True,
                                                                            attested_policy=None, declared=True)
                assert r["expected_stamps"][1] == C.DECLARED_SALT_POLICY_LABEL and r["expected_stamps"] != ["declared-request-unique", "always", "thinking_on"]
        else:
            assert r["object_sha256"] is None and r["tuple_names"] is None and r["served_positions_expected"] is None
            assert tuple(r["expected_stamps"]) == C.expected_stamp_triple(lens=False, arm=C.THINKING_ON, header_located=False, attested_policy=None)
        # NO row is an expected refusal — gemma's two header-located `main` cells are servable
        # thinking-ON on the declared path; the field stays a boolean for the analysis' reader
        assert r["expected_refusal"] is False and r["expected_refusal_reason"] is None
        assert r["tuple_name"] == ((r["tuple_names"] or {}).get("8192")) and (r["tuple_name"] or "_8192").endswith("_8192")   # lane A: the launch row's name
    assert len({r["out_root"] for r in ROWS}) == COUNTS["distinct_out_roots"] == len(ROWS) and len({r["job_id"] for r in ROWS}) == len(ROWS)
    # over the MATRIX's models (the registry's fifth key, llama8b, runs the no-switch arm under its own matrix — P16)
    assert COUNTS["models"] == ["gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash"] and COUNTS["families"] == ["gemma4", "glm4v", "nemotron_h", "qwen3_5"]
    assert sorted(COUNTS["tuple_names"]) == COUNTS["tuple_names_launch"] == sorted({M.tuple_name(k, declared=d, rung=8192) for k in COUNTS["models"] for d in (False, True)
                                                                                    if not (d and k in ("nemotron3nano4b", "glm46vflash"))})
    with pytest.raises(SystemExit, match="not a matrix row"):
        M.row_for("gemma4e4b", "main", "prompted__safety_refusal", rows=ROWS)     # the PUBLISHED gemma main has no prompted cell (the roster's); E2 appends one (row 42)
    assert M.row_for("gemma4e4b", "main", "prompted__safety_refusal")["wave"] == M.WAVE_EXT


def test_views_select_rows_is_the_one_selector_with_the_cell_rule_and_tag_for_every_view_kind(monkeypatch):
    roster = V.load_roster(V.roster_path(REPO))
    pins = V.load_pins(V.pins_path(REPO))
    # the SCHEDULED cells of every view kind are the frozen shape; whatever else the roster declares on a view is a prompted
    # cell this matrix does not schedule (listed by cells_of, routed by select_rows to the trailing held pass)
    def scheduled(key, view):
        # the batch-64 lane's cells are matrix itdoc's, set aside like _without_batch64_lane does
        return [c for c, s in V.cells_of(roster, key, view) if not M.is_unscheduled_prompted(c) and not str(s or "").startswith(BATCH64_LANE_PREFIXES)]
    for key, view in (("gemma4e4b", "it"), ("gemma4e4b", "main"), ("nemotron3nano4b", "main")):
        assert all(c.startswith("prompted__") and s is None for c, s in V.cells_of(roster, key, view) if M.is_unscheduled_prompted(c))
    assert scheduled("gemma4e4b", "it") == ["none", "prompted__safety_refusal", "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce"]
    assert scheduled("gemma4e4b", "main")[:2] == ["none", "constv2-mix-full__psr_ce_all"] and len(scheduled("gemma4e4b", "main")) == 9
    nemo = scheduled("nemotron3nano4b", "main")
    assert nemo.count("prompted__safety_refusal") == 1 and nemo[0] == "none" and len(nemo) == 8     # the roster's prompted cell, once, in roster order
    # A DOSE view is a legitimate view kind here now that the strength ladder is in the shared roster, and the
    # selector resolves it generically — it enumerates whatever the roster names rather than a fixed kind list,
    # and `tag_for` below already spells a dose TAG. It returns the dose view's OWN cells: its same-pass `none`
    # anchor plus the LoRA at that dose. This assertion used to be a refusal, which was only ever true because
    # the roster had no dose views; it is not a statement that GSM8K should refuse them. What keeps the ladder
    # out of GSM8K launches is the MATRIX, which still carries 6 TAGs and no dose row, not this selector.
    # since the strength frontier's re-mint landed a dose view carries BOTH LoRA arms at that dose — the
    # constitution LoRA and the LoRA subtraction — as batch-64 copies (`d<tok>it64__…`), beside its same-pass `none`
    assert [c for c, _ in V.cells_of(roster, "gemma4e4b", "it_d0p1")] == ["none", "constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"]
    assert [s for _, s in V.cells_of(roster, "gemma4e4b", "it_d0p1")] == [None, "d0p1it64__constv2-mix-full__lora_r4_ce", "d0p1it64__constv2-mix-full__lora_r4_ce_minus_ctl"]
    with pytest.raises(SystemExit, match="roster names no view"):
        V.cells_of(roster, "gemma4e4b", "it_d9p9")        # a view the roster genuinely does not name still refuses
    # llama8b is a shared-roster target since its block landed (2026-09-12): the selector serves it like
    # any other target (its rows live in the Llama matrix); an UNREGISTERED key still refuses by name
    llama = [c for c, _ in V.cells_of(roster, "llama8b", "main")]
    assert llama[0] == "none" and "constv2-mix-full__lora_r4_ce_minus_ctl" in llama
    with pytest.raises(SystemExit, match="no target"):
        V.cells_of(roster, "mistral7b", "main")
    # TAG: the house rule for all three view kinds
    assert M.tag_for("gemma4e4b", "main") == "gemma4e4b" and M.tag_for("gemma4e4b", "it") == "gemma4e4b_it" and M.tag_for("gemma4e4b", "it_d0p1") == "gemma4e4b_it_d0p1"
    # the spec grammar
    assert V.parse_views_spec("gemma4e4b=it,main; qwen35_4b=main") == {"gemma4e4b": ["it", "main"], "qwen35_4b": ["main"]}
    for bad, why in (("gemma4e4b", "not key=view"), ("gemma4e4b=it;gemma4e4b=main", "named twice"), ("gemma4e4b=", "names no view"),
                     ("gemma4e4b=it,it", "twice"), ("", "names no")):
        with pytest.raises(SystemExit, match=why):
            V.parse_views_spec(bad)
    # the selector over the LIVE inputs == the frozen matrix on its scheduled rows (rows and every field); the live roster has
    # moved past the pin (it declares the constitution prompted cell on published views, which the frozen roster does not),
    # so whatever it adds appears only as TRAILING held rows behind the 42 — the frozen document carries none
    spec = V.load_views_yaml(REPO / V.VIEWS_REL)
    assert spec.views == {"gemma4e4b": ["it", "main"], "qwen35_4b": ["it", "main"], "nemotron3nano4b": ["main"], "glm46vflash": ["main"]}
    live = _without_batch64_lane(V.select_rows(roster, pins, spec.views, holds=spec.holds))       # the batch-64 lane's cells are matrix itdoc's
    assert live[:len(ROWS)] == ROWS and live[len(ROWS):] == M.unscheduled_held_rows(live)
    assert all(M.is_unscheduled_prompted(r["cell"]) and r["status"] == "held" for r in live[len(ROWS):])
    # the generator calls the ONE selector exactly once; g8k_cells.py never re-derives (no roster / yaml import)
    gen = _load("_g8k_make_matrix_spy", "scripts/sdfbase/gsm8k/make_gsm8k_matrix.py")
    calls = []
    real = V.select_rows
    monkeypatch.setattr(gen.V, "select_rows", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    text, doc = gen.build(configs_root=REPO, views_yaml=None, views_spec=None)
    assert calls == [1] and doc["rows"][:len(ROWS)] == ROWS and doc["rows"] == ALL_ROWS and text == (REPO / "configs/sdfbase/gsm8k/matrix.json").read_text()
    src = (G8K / "g8k_cells.py").read_text()
    assert "f22_roster" not in src and "import yaml" not in src and "served_cells" not in src
    assert "make_gsm8k_matrix.py" in (G8K / "g8k_dry_run.py").read_text()     # the dry run generates through the generator, never its own rule
    # holds validation
    bad = yaml.safe_load((REPO / V.VIEWS_REL).read_text())
    bad["holds"][0]["view"] = "halfdose"
    p = Path(os.environ.get("TMPDIR", "/tmp")) / "g8k_bad_views.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(SystemExit, match="does not express"):
        V.load_views_yaml(p)
    bad = yaml.safe_load((REPO / V.VIEWS_REL).read_text()); bad["holds"][0]["reason"] = " "
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(SystemExit, match="no reason"):
        V.load_views_yaml(p)
    # a fitted row whose staged id has no pin refuses BY NAME (a dose object not staged where the row runs)
    pins2 = json.loads(json.dumps(pins)); pins2["gemma4e4b"].pop("itdoc__constv2-mix-full__psr_ce_all")
    with pytest.raises(SystemExit, match="no 64-hex object_sha256 for gemma4e4b/itdoc__constv2-mix-full__psr_ce_all"):
        V.select_rows(roster, pins2, {"gemma4e4b": ["it"]})


def test_unholding_is_one_views_yaml_edit_with_no_other_diff():
    roster = V.load_roster(V.roster_path(REPO)); pins = V.load_pins(V.pins_path(REPO))
    spec = V.load_views_yaml(REPO / V.VIEWS_REL)
    holds = [h for h in spec.holds if h["key"] != "gemma4e4b"]
    live = _without_batch64_lane(V.select_rows(roster, pins, spec.views, holds=holds))           # the batch-64 lane's cells are matrix itdoc's
    rows = [r for r in live if r not in M.unscheduled_held_rows(live)]      # the scheduled rows; the trailing holds ride behind
    assert len(rows) == len(ROWS)
    d8_gemma = [r["index"] for r in ROWS if r["tag"] == "gemma4e4b_it" and r["status"] == "held" and r not in UNSCHED]
    d8_qwen = [r["index"] for r in ROWS if r["tag"] == "qwen35_4b_it" and r["status"] == "held" and r not in UNSCHED]
    assert d8_gemma == [2, 3] and d8_qwen == [15, 16]
    for new, old in zip(rows, ROWS):
        if old["index"] in d8_gemma:
            assert new["status"] == "launch" and new["held_reason"] is None
            assert {k: v for k, v in new.items() if k not in ("status", "held_reason")} == {k: v for k, v in old.items() if k not in ("status", "held_reason")}
        else:
            assert new == old                    # a trailing held row is untouched by a views.yaml edit (it is not a views.yaml hold)
    c = M.counts(rows)
    assert c["launched"] == 40 and c["held"] == 2 + len(UNSCHED) and c["fitted_launched"] == 30 and c["servable"] == 40
    # every consumer picks the change up from the matrix alone: plan covers 2 and 3 once un-held
    pl = M.plan(SHA0, "cluster_a", max_concurrent=12, universe=rows)
    assert set(d8_gemma) <= set(pl.rows) and not set(d8_qwen) & set(pl.rows) and not {r["index"] for r in UNSCHED} & set(pl.rows)


# --------------------------------------------------------------------------- #
# HELD on every production consumer; placement                                 #
# --------------------------------------------------------------------------- #


def test_held_rows_are_excluded_by_every_production_consumer(tmp_path, s_by):
    held_idx = [r["index"] for r in M.held_rows()]
    pl = M.plan(SHA0, "cluster_a", max_concurrent=12, universe=ROWS)             # the published universe (the python entry point)
    assert not set(held_idx) & set(pl.rows) and not set(held_idx) & set(_covered(pl.lines)) and pl.held == held_idx
    assert sorted(pl.rows) == [r["index"] for r in LAUNCHED]
    with pytest.raises(SystemExit, match="HELD row"):
        M.plan(SHA0, "cluster_a", max_concurrent=4, rows="0-3")
    with pytest.raises(SystemExit, match="HELD row"):
        M.plan(SHA0, "cluster_a", max_concurrent=4, cluster_share="0-16")
    r = subprocess.run([sys.executable, str(G8K / "g8k_cells.py"), "row", "--index", "2", "--sh"], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "STATUS=held" in r.stdout and "G8K_HELD_REASON='2026-09-10: the instruct-direct" in r.stdout
    # the sbatch: HELD line, exit 0, NO argv (no standup, no ledger, no driver)
    for idx in held_idx:
        row = ROWS[idx]
        text, rc, raw = _dry(row, s=s_by[(row["model_key"], M.CAP)], shared=tmp_path / "shared")
        assert rc == 0 and text.startswith(f"HELD: index={idx} job_id={row['job_id']} ")
        assert ("reason=2026-09-10: the instruct-direct" in text) == (row not in UNSCHED) and (f"reason={M.UNSCHEDULED_HELD_MARK}" in text) == (row in UNSCHED)   # the held mark vs the unscheduled rows
        assert "DRIVER:" not in raw and "STANDUP:" not in raw and "LEDGER:" not in raw and row["job_id"] not in _argv_lines(raw)
    # the dry-run driver: HELD: for the held rows, DRIVER: only for the launched ones, and a HELD row's job id in no argv
    out = tmp_path / "dry.txt"
    r = subprocess.run([sys.executable, str(G8K / "g8k_dry_run.py"), "--stage", "pre", "--rows", "0-3,13-16", "--tmp", str(tmp_path / "t"),
                        "--python", PY], capture_output=True, text=True, check=False, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    out.write_text(r.stdout)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    assert sum(line.startswith("HELD: ") for line in r.stdout.splitlines()) == 4
    assert sum(line.startswith("DRIVER:") for line in r.stdout.splitlines()) == 4
    argv = "\n".join(_argv_lines(r.stdout))
    assert all(ROWS[i]["job_id"] not in argv and ROWS[i]["cell_slug"] not in argv for i in held_idx)
    assert "VERDICT: PASS stage=pre" in r.stdout


def _argv_lines(text):
    return [line for line in text.splitlines() if line.startswith(("DRIVER:", "STANDUP:", "LEDGER:"))]


def test_model_based_placement_refuses_a_primary_row_off_the_primary_cluster_in_plan_and_at_job_start(tmp_path, s_by):
    nemo = [r["index"] for r in LAUNCHED if r["model_key"] == "nemotron3nano4b"]
    assert nemo == list(range(26, 34)) and all(ROWS[i]["cluster_constraint"] == "primary" for i in nemo)
    with pytest.raises(SystemExit, match="cluster_constraint primary"):
        M.plan(SHA0, "cluster_b", max_concurrent=4, rows="26")
    with pytest.raises(SystemExit, match="cluster_constraint primary"):
        M.plan(SHA0, "cluster_b", max_concurrent=16, cluster_share="0-1")           # the complement would carry Nemotron
    with pytest.raises(SystemExit, match="cluster_constraint primary"):
        M.plan_overlap(SHA0, "cluster_b")
    assert M.plan_overlap(SHA0, "cluster_b", keys=("gemma4e4b", "qwen35_4b", "glm46vflash"))[0].count("--array=0-1,3%3") == 1
    assert "--array=0-3%4" in M.plan_overlap(SHA0, "cluster_a")[0] and "overlap_off.sbatch ARRAY" in M.plan_overlap(SHA0, "cluster_a")[0]
    # at job start: the same row under the sbatch's own g8k_row, dry-run on the overflow cluster, dies by name with no argv
    row = ROWS[26]
    text, rc, raw = _dry(row, s=s_by[("nemotron3nano4b", M.CAP)], shared=tmp_path / "shared", cluster="cluster_b")
    assert rc == 1 and "PLACEMENT" in raw and "cluster_constraint primary" in raw and "DRIVER:" not in raw
    text, rc, raw = _dry(row, s=s_by[("nemotron3nano4b", M.CAP)], shared=tmp_path / "shared", cluster="cluster_a")
    assert rc == 0 and "DRIVER:" in text
    # the overlap sbatch's leg on the overflow cluster likewise
    env = {**os.environ, "G8K_DRY_RUN": "1", "G8K_DRY_S": "170", "G8K_DRY_SHARED": str(tmp_path / "shared"), "G8K_DRY_CPY": PY,
           "G8K_DRY_CLUSTER": "cluster_b", "USER": "tester"}
    r = subprocess.run(["bash", str(G8K / "overlap_off.sbatch"), "nemotron3nano4b", SHA0], capture_output=True, text=True, check=False, env=env)
    assert r.returncode == 1 and "PLACEMENT" in r.stdout + r.stderr


# --------------------------------------------------------------------------- #
# the overlays                                                                 #
# --------------------------------------------------------------------------- #


def test_overlays_carry_each_tags_rows_and_the_campaign_originals_are_untouched():
    from sisteer.capeval.petri import registry_adapter as ra
    by_tag = {}
    # a TAG's overlay lists its CAMPAIGN-BANK rows — the published ones and the appended prompted rows (`ext`); a steering row
    # serves its OWN pair (its object is another lane's; see the `steer` section), never the TAG's
    for r in ALL_ROWS:
        if r.get("object_source") != M.OBJECT_SOURCE_EXTERNAL:
            by_tag.setdefault(r["tag"], []).append(r)
    for tag, rows in by_tag.items():
        cfg = yaml.safe_load((REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_{tag}.gsm8k.yaml").read_text())
        campaign = yaml.safe_load((REPO / f"configs/sdfbase_full22_{tag}.yaml").read_text())
        assert cfg["explicit_cells"] == [r["cell"] for r in rows]
        assert {k: v for k, v in cfg.items() if k != "explicit_cells"} == {k: v for k, v in campaign.items() if k != "explicit_cells"}
        man = yaml.safe_load((REPO / f"configs/capeval/manifests/petri_full22_{tag}.gsm8k.yaml").read_text())["petri"]
        cman = yaml.safe_load((REPO / f"configs/capeval/manifests/petri_full22_{tag}.yaml").read_text())["petri"]
        assert man["cells"] == [ra.harness_cell_id(r["cell"]) for r in rows]
        assert {k: v for k, v in man.items() if k != "cells"} == {k: v for k, v in cman.items() if k != "cells"}
        assert man["prompted_prefix_hashes"]["safety"] == "ac3a58f61d53cf24"          # every view can serve the prompted safety cell
        if tag.endswith("_it"):
            # the `it` views serve the matched prompted safety baseline: the overlay always carries it,
            # and since the sampled re-run's lane F (2026-09-09) the roster LISTS it on the `it` views, so
            # the campaign `it` configs carry it too (the earlier claim "not in the campaign" described the
            # pre-lane-F roster; g8k_views.cells_of keeps it SECOND either way)
            assert "prompted__safety_refusal" in cfg["explicit_cells"]
            assert "prompted__safety" in man["cells"]
        # the campaign originals are untouched BUT FOR the ruled changes (the 4,096 cap, the bon.climb_*
        # keys) and the it views' listed prompted cell
        for rel in (f"configs/sdfbase_full22_{tag}.yaml", f"configs/capeval/manifests/petri_full22_{tag}.yaml"):
            base = subprocess.run(["git", "show", f"{BASE_SHA}:{rel}"], cwd=REPO, capture_output=True, text=True, check=False)
            if base.returncode != 0:
                pytest.skip("the base sha is not in this clone")
            _assert_original_unchanged_but_for_the_ruled_cap(rel, base.stdout, _pre_release_text(rel, (REPO / rel).read_text()))


# --------------------------------------------------------------------------- #
# the sbatch shape: the cluster's hard rules; the prologue                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sh", ["bash", "zsh"])
@pytest.mark.parametrize("name", SBATCHES + ["g8k_common.sh", "../full22/served_thinking.sh"])
def test_shell_files_parse_under_bash_and_zsh(name, sh):
    if shutil.which(sh) is None:
        assert sh == "zsh", f"{sh} is not installed, and the sbatch runs under it"
        pytest.skip(f"{sh} is not installed here")
    r = subprocess.run([sh, "-n", str(G8K / name)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, (sh, r.stderr)


@pytest.mark.parametrize("name", SBATCHES)
def test_sbatch_headers_carry_the_hard_rules(name):
    src = (G8K / name).read_text()
    common = (F22 / "common.sh").read_text() + (F22 / "served_thinking.sh").read_text() + (G8K / "g8k_common.sh").read_text()
    for line in src.splitlines():
        if "--export=" in line:
            assert "never" in line.lower(), line
    assert "--mem=0" not in src
    gpu = name in GPU_SBATCHES
    for m in ("#SBATCH --qos=high", "#SBATCH --partition=general", "#SBATCH --open-mode=append", "#SBATCH --no-requeue",
              "#SBATCH --signal=B:TERM@120", "#SBATCH --time=", "#SBATCH --output=/workspace-vast/%u/exp/logs/") + (
                  ("#SBATCH --gres=gpu:1", "#SBATCH --mem=200G") if gpu else ()):
        assert m in src, (name, m)
    mem = [line for line in src.splitlines() if line.startswith("#SBATCH --mem=")][0]
    assert int(mem.split("=")[1].rstrip("G")) <= (200 if gpu else 64)
    if not gpu:
        assert "--gres" not in src
    assert "positional only" in src and "never --export=" in src
    assert "trap cleanup EXIT SIGTERM SIGINT SIGQUIT" in src and "f22_stop_requested" in src
    assert "NCCL_SOCKET_IFNAME=" not in src and "NCCL_SOCKET_IFNAME=" not in common   # the site's nccl.conf decides
    for m in ("hf_env.sh", "NCCL_NVLS_ENABLE=0", "PYTORCH_CUDA_ALLOC_CONF", "mirror-figs.sh",
              "export SISTEER_LENS_STACKED_REDUCTIONS=1", "--reduction-mode stacked --require-attested-mode stacked"):
        assert m in src or m in common, (name, m)
    assert "f22_result" in src
    if gpu:
        assert "STANDUP_STOP" in src and "g8k_attempt_end" in src
    assert "F22_REQUIRE_NEMO_BACKEND=1" in common
    # the datasets guard names BOTH datasets AND their offline config dirs through the ONE emitted literal (g8k_cells.py
    # literals --sh, evaluated by g8k_init) — no dataset name is typed in the shell
    common_sh = (G8K / "g8k_common.sh").read_text()
    assert M.DATASETS_REQUIRED == ("openai___gsm8k", "madrylab___gsm8k-platinum") and set(M.DATASETS_CONFIG_DIRS) == set(M.DATASETS_REQUIRED)
    lits = dict(l.split("=", 1) for l in M.literals_sh().splitlines())
    assert lits["G8K_DATASETS_REQUIRED"] == "'openai___gsm8k madrylab___gsm8k-platinum'"
    assert lits["G8K_DATASETS_CONFIG_DIRS"] == "'openai___gsm8k/default-data_dir=main madrylab___gsm8k-platinum/main'"
    assert "g8k_cells.py\" literals --sh" in common_sh and 'for d in ${G8K_DATASETS_REQUIRED:?' in common_sh and 'for d in ${G8K_DATASETS_CONFIG_DIRS:?' in common_sh
    assert "for d in openai___gsm8k" not in common_sh and common_sh.index("g8k_datasets_guard\n") > common_sh.index("literals --sh")
    # the AUTHORITATIVE check: the wrapper's own plan OFFLINE from the client venv, before any standup, in BOTH GPU launchers
    common_g8k = (G8K / "g8k_common.sh").read_text()
    assert "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1" in common_g8k and "isv.plan_task(task" in common_g8k and 'f22_fatal "datasets probe FAILED' in common_g8k
    if gpu:
        assert src.index("f22_client_venv") < src.index("g8k_datasets_probe") < src.index("f22_standup attacks")


def test_the_prologue_is_identical_across_the_gsm8k_launchers_and_equals_the_math500_one_modulo_the_helper_filename():
    pat = (r"(# --- the checkout, located BEFORE either helper is sourced.*?\n"
           r"source \"\$REPO_DIR/scripts/sdfbase/full22/common\.sh\"; source \"\$REPO_DIR/scripts/sdfbase/{helper}\"\n)")
    blocks = {}
    for name in SBATCHES:
        src = (G8K / name).read_text()
        m = re.search(pat.format(helper=r"gsm8k/g8k_common\.sh"), src, re.DOTALL)
        assert m, name
        blocks[name] = m.group(1)
        assert 'source "/workspace' not in src
        for line in src.splitlines():
            if "$0" in line:
                assert line.lstrip().startswith("#"), line
            if "BASH_SOURCE" in line and not line.lstrip().startswith("#"):
                assert "G8K_DRY_RUN" in line, line
        i_src, i_trap, i_init = m.end(), src.index("trap cleanup EXIT SIGTERM SIGINT SIGQUIT"), src.index('g8k_init "$WAVE_SHA"')
        assert i_src < i_trap < i_init, (name, i_src, i_trap, i_init)
        between = re.sub(r'if \[ "\$\{G8K_DRY_RUN:-0\}" = 1 \]; then.*?\nfi\n', "", src[i_src:i_trap], flags=re.DOTALL)
        between = re.sub(r"cleanup\(\)\{.*?\n\}\n", "", between, flags=re.DOTALL)
        between = "\n".join(line for line in between.splitlines() if not line.lstrip().startswith("#"))
        assert not re.search(r"\b(g8k|f22|st)_[a-z_]+", between), (name, between)
        cleanup = src[src.index("cleanup(){"):i_trap]
        assert not re.search(r"\$(G8K_INDEX|G8K_JOB_ID|KEY|VIEW|CELL|FITTED|DECLARED|RUNG|G8K_FAMILY)\b", cleanup), name
        for var in ("FIGS_DIR", "OUT_ROOT", "DRY_OUT", "PLAN_OUT"):
            if var in cleanup:
                assert "${" + var + ":-}" in cleanup, (name, var)
        assert "${FIGS_DIR:-}" in cleanup and "f22_result" in cleanup and "RESULT (before the wave root resolved" in cleanup, name
        assert re.search(r"f22_stop_requested && \{[^}]*G8K_RESULT=STOP_REQUESTED[^}]*exit 0; \}", src), name
    assert len(set(blocks.values())) == 1, blocks
    # == the MATH-500 launchers' block modulo the helper filename (the override name, the init name, the helper path)
    m5t_src = (M5T / "math500_thinking.sbatch").read_text()
    m5t_block = re.search(pat.format(helper=r"math500/m5t_common\.sh"), m5t_src, re.DOTALL).group(1)
    translated = (m5t_block.replace("M5T_REPO_DIR", "G8K_REPO_DIR").replace("m5t_init", "g8k_init")
                  .replace("scripts/sdfbase/math500/m5t_common.sh", "scripts/sdfbase/gsm8k/g8k_common.sh"))
    assert translated == blocks["gsm8k_thinking.sbatch"]


def test_gsm8k_sbatch_grammar_venvs_bank_check_ledger_order():
    src = (G8K / "gsm8k_thinking.sbatch").read_text()
    assert 'FITTED_ARG="${6:?fitted 0|1}"; DECLARED_ARG="${7:?declared 0|1}"' in src and 'G8K_RUNG_ARG="${4:?rung 8192|16384}"' in src
    assert 'case "$G8K_RUNG_ARG" in 8192|16384) ;;' in src and 'if [ "${1:-}" = ARRAY ]' in src and 'G8K_RUNG_ARG="${3:-8192}"' in src
    assert 'the matrix says $FITTED' in src and 'the matrix says $DECLARED' in src
    assert 'if [ "$KEY" = nemotron3nano4b ]; then f22_serving_venv nemotron; else f22_serving_venv; fi' in src
    # E2 / P8: the bank check is DISPATCHED (g8k_bank_view: the campaign's roster-resolved view, or a steering row's external lane
    # view) — the sbatch names the dispatcher once and never a check directly; the two checks have exactly one caller each
    assert "\ng8k_bank_view\n" in src and "f22_bank_view_check" not in src and "st_external_bank_view_check" not in src
    assert "f22_bank_view " not in src
    common = (G8K / "g8k_common.sh").read_text()
    body = common[common.index("g8k_bank_view(){"):common.index("g8k_resolve_config(){")]
    calls = lambda text, fn: sum(1 for ln in text.splitlines() if ln.strip().startswith(fn + " "))   # call sites, not comment mentions
    assert calls(body, "st_external_bank_view_check") == 1 and calls(body, "f22_bank_view_check") == 2 and calls(common, "st_external_bank_view_check") == 1
    assert 'if [ "$FITTED" = 1 ]; then\n    f22_bank_view_check "$CPY" "$CELL" "$PIN"' in body.replace("elif", "if") and 'f22_bank_view_check "$CPY"\n' in body
    idx = [src.index(s) for s in ("g8k_assert_launchable", "g8k_rung_check", "f22_serving_venv", "f22_client_venv", "g8k_datasets_probe", "\ng8k_bank_view\n", "g8k_budget_artifact",
                                  "g8k_attempt_begin", "f22_standup attacks", "g8k_check_endpoint", "g8k_ledger_up || exit 1",
                                  "g8k_driver_argv", '"${DRV[@]}"', "g8k_ledger_down\nf22_standup_stop")]
    assert idx == sorted(idx), idx
    assert 'f22_standup attacks "$SERVE_DIR/$G8K_SERVE_SUBDIR" "" "$G8K_GEOMETRY_EVAL" "" "" ""' in src
    assert 'CUDA_VISIBLE_DEVICES=""' in src and '"$G8K_INSPECT_BASE_URL_ENV=$LEDGER_URL"' in src
    assert 'OUT_ROOT="$WAVE_ROOT/$G8K_OUT_REL"' in common and '>> "$OUT_ROOT/RESULT"' in src
    assert f'G8K_WAVE_ROOT_REL="{M.WAVE_ROOT_REL}"' in common
    # the core is delegated to, never re-typed: the ledger, the endpoint checks, the driver head / tail, the attempt seam
    for fn in ("st_ledger_up", "st_ledger_down", "st_proxy_token", "st_budget_artifact", "st_check_endpoint", "st_driver_argv_head",
               "st_driver_argv_tail", "st_attempt_begin", "st_attempt_end", "st_print_argv"):
        assert fn in common, fn
    assert "usage_ledger" not in common.replace("# ", "")        # the ledger process is started by the core alone


# --------------------------------------------------------------------------- #
# the generated argv, through the sbatch's OWN dry run; counts from the matrix #
# --------------------------------------------------------------------------- #


def test_dry_run_argv_per_launched_row_carries_the_plan_flags(tmp_path, s_by):
    runs = [(row, *_dry(row, s=s_by[(row["model_key"], M.CAP)], shared=tmp_path / "shared")) for row in LAUNCHED]
    assert all(rc == 0 for _, _, rc, _ in runs), [(r["job_id"], raw[-300:]) for r, _, rc, raw in runs if rc]
    text = "\n".join(t for _, t, _, _ in runs)
    standup = [line for line in text.splitlines() if line.startswith("STANDUP:")]
    driver = [line for line in text.splitlines() if line.startswith("DRIVER:")]
    ledger = [line for line in text.splitlines() if line.startswith("LEDGER:")]
    n = COUNTS["launched"]
    assert len(standup) == len(driver) == len(ledger) == n == 38
    for frag in (f"--controls {M.CONTROL} ", f"{M.FLAG_THINKING} on ", f"{M.FLAG_SAMPLING} thinking_card ", f"{M.FLAG_SAMPLES} 1 ",
                 f"{M.FLAG_MAX_TOKENS_BY_TASK} gsm8k_paired200=8192 ", f"{M.FLAG_ON_CAP_BY_TASK} gsm8k_paired200=8192 ",
                 f"{M.FLAG_TINYCAP_IDENTITY_VERSION} 2 ", f"{M.FLAG_LEDGER_URL} ", f"{M.FLAG_OOS_PROMPT_BUDGET} ", f"{M.FLAG_ATTEMPT_ID} a1 ",
                 f"{M.FLAG_BANK_VIEW} "):
        assert sum(frag in line for line in driver) == n, frag
    # the row's VIEW to the driver (llama-paper-rows: the four provenance fields on every emitted row; contracts.BANK_VIEWS)
    assert M.FLAG_BANK_VIEW == C.FLAG_BANK_VIEW and all(f"{M.FLAG_BANK_VIEW} {row['view']} " in _line(t, "DRIVER") + " " for row, t, _, _ in runs)
    assert all(row["view"] in C.BANK_VIEWS for row in LAUNCHED)
    assert sum(f"{M.FLAG_UPSTREAM_TIMEOUT_S} 7200 " in line for line in standup) == n
    assert sum(f"{M.FLAG_LENS_SALT_POLICY} attested" in line for line in standup) == n
    assert sum("--attempt-id a1" in line for line in ledger) == n
    assert "--math500-retry-max-tokens" not in text and M.FLAG_ITEM_SUBSET not in text
    # E2-B: the LOGPROB liveness gate is the EXTERNAL (steer) rows' alone — every published row's driver argv, the 28 fitted ones
    # included, is byte-identical to before (no flag), and no published row prints a SELECTION line
    assert M.FLAG_LENS_LIVENESS not in text and "SELECTION:" not in text
    fitted = {r["index"] for r in LAUNCHED if r["fitted"]}
    declared = {r["index"] for r in LAUNCHED if r["declared"]}
    for row, t, _, _ in runs:
        d, s = _line(t, "DRIVER"), _line(t, "STANDUP")
        assert (M.FLAG_ALLOW_OOS_THINKING in d) is (row["index"] in fitted)
        assert (f"{M.FLAG_EXPECT_OBJECT_SHA256} {row['object_sha256']}" in d) is row["fitted"]
        # B-P1.1: the digest is keyed on the SAME spelling the standup is asked to serve (its parser wants the exact --cell)
        assert (f"{M.FLAG_EXPECT_OBJECT_SHA256} {row['standup_cell']}={row['object_sha256']}" in s) is row["fitted"]
        assert (f"{M.FLAG_EXPECT_OBJECT_SHA256} {row['cell']}=" in s) is (row["fitted"] and not row["declared"])
        want_cell = row["standup_cell"]
        assert want_cell == row["cell"] + ("@thinking" if row["declared"] else "")
        assert (f"--cell {want_cell} " in s + " ") and ("@thinking" in s) is (row["index"] in declared)   # DECLARED iff header-located
        assert f"--cell {row['cell']} " in d + " " and "@thinking" not in d                              # the driver's plain spelling
        assert (f"{M.FLAG_REASONING_PARSER} gemma4" in s) is (row["model_key"] == "gemma4e4b")
        assert f"--config configs/sdfbase/gsm8k/sdfbase_full22_{row['tag']}.gsm8k.yaml " in d
        assert f"--defenses-dir {tmp_path / 'shared'}/exp/objects/sdfbase/full22/{row['model_key']}/cells_view/{row['view']} " in d
        assert f"--out-root {tmp_path / 'shared'}/{M.WAVE_ROOT_REL}/{row['out_root']} " in d
        assert f"--concurrency {s_by[(row['model_key'], M.CAP)]} " in d
        assert f"GEOMETRY: {M.GEOMETRY_EVAL_BY_RUNG[M.CAP]} {M.GEOMETRY_L_BY_RUNG[M.CAP]} x {s_by[(row['model_key'], M.CAP)]} cap 8192 thinking on" in t
        assert ("OUT OF SUPPORT: tuple=" + str((row["tuple_names"] or {}).get("8192")) in t) is row["fitted"]
        assert ("EXPECTED_REFUSAL:" in t) is row["expected_refusal"]
        assert "SAMPLING: mode=thinking_card params=" in t and "SAMPLING_SOURCE: " in t
    assert sum(M.FLAG_ALLOW_OOS_THINKING in line for line in driver) == COUNTS["fitted_launched"] == 28
    assert sum("@thinking" in line for line in standup) == COUNTS["declared"] == 4
    assert sum(f"{M.FLAG_REASONING_PARSER} gemma4" in line for line in standup) == COUNTS["reasoning_parser_gemma4"] == 11
    assert text.count("EXPECTED_REFUSAL:") == COUNTS["expected_refusals"] == 0
    assert len({re.search(r"--out-root (\S+) ", line).group(1) for line in driver}) == n
    tuples = re.findall(r"OUT OF SUPPORT: tuple=(\S+)", text)
    assert sorted(set(tuples)) == COUNTS["tuple_names"] and len(set(tuples)) == 6 and all(t.endswith("_8192") for t in tuples)
    assert sum(f"DRIVER_ENV: CUDA_VISIBLE_DEVICES= {M.INSPECT_BASE_URL_ENV}=" in line for line in text.splitlines()) == n
    # a FITTED / DECLARED positional that contradicts the matrix refuses by name
    row = ROWS[4]
    _, rc, raw = _dry(row, s=54, shared=tmp_path / "shared", fitted="1")
    assert rc == 1 and "the matrix says 0" in raw
    _, rc, raw = _dry(ROWS[22], s=33, shared=tmp_path / "shared", declared="0")
    assert rc == 1 and "the matrix says 1" in raw


def test_rung_rows_and_overlap_legs_carry_their_own_flags(tmp_path, s_by):
    shared = tmp_path / "shared"
    for row in (ROWS[1], ROWS[22], ROWS[35]):                      # in-support, declared, GLM fitted
        t, rc, _ = _dry(row, rung=M.RUNG_CAP, s=s_by[(row["model_key"], M.RUNG_CAP)], shared=shared)
        assert rc == 0
        d = _line(t, "DRIVER")
        assert f"{M.FLAG_MAX_TOKENS_BY_TASK} gsm8k_paired200=16384 " in d and f"{M.FLAG_ON_CAP_BY_TASK} gsm8k_paired200=16384 " in d
        assert d.endswith(f"{M.FLAG_ITEM_SUBSET} {shared}/{M.WAVE_ROOT_REL}/rung16384/{row['tag']}/{row['cell_slug']}/rung_subset.json")
        assert f"--out-root {shared}/{M.WAVE_ROOT_REL}/{row['rung_out_root']} " in d
        assert f"GEOMETRY: {M.GEOMETRY_EVAL_BY_RUNG[M.RUNG_CAP]} 18944 x {s_by[(row['model_key'], M.RUNG_CAP)]} cap 16384 thinking on" in t
        assert f"RUNG_SUBSET: {shared}/{M.WAVE_ROOT_REL}/rung16384/{row['tag']}/{row['cell_slug']}/rung_subset.json (source {shared}/{M.WAVE_ROOT_REL}/{row['out_root']})" in t
        if row["fitted"]:
            assert f"OUT OF SUPPORT: tuple={row['tuple_names']['16384']} " in t
    with pytest.raises(SystemExit, match="rung must be one of"):
        M.row_sh(ROWS[0], 4096)
    # the overlap legs: thinking OFF, greedy, the OFF table's 512 as a STATEMENT ONLY (no setter), the OFF class's entry
    from capctrl_serving_up import resolve_standup_geometry
    for leg in M.overlap_legs():
        s = int(resolve_standup_geometry(M.OVERLAP_GEOMETRY_EVAL, None, None, model_key=leg["model_key"]).max_num_seqs)
        env = {**os.environ, "G8K_DRY_RUN": "1", "G8K_DRY_S": str(s), "G8K_DRY_SHARED": str(shared), "G8K_DRY_CPY": PY, "USER": "tester"}
        r = subprocess.run(["bash", str(G8K / "overlap_off.sbatch"), leg["model_key"], SHA0], capture_output=True, text=True, check=False, env=env)
        assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
        t = "\n".join(line for line in r.stdout.splitlines() if not line.startswith("["))
        d, st = _line(t, "DRIVER"), _line(t, "STANDUP")
        assert f"{M.FLAG_THINKING} off " in d and f"{M.FLAG_SAMPLING} greedy " in d and f"{M.FLAG_MAX_TOKENS_BY_TASK} gsm8k_paired200=512 " in d
        assert M.FLAG_ON_CAP_BY_TASK not in d and M.FLAG_ALLOW_OOS_THINKING not in d and M.FLAG_ITEM_SUBSET not in d and "--attempt-id a1" in d
        assert f"--out-root {shared}/{M.WAVE_ROOT_REL}/overlap/{leg['model_key']}/none/greedy_off512 " in d and "--cell none " in d
        assert st.endswith("--cell none") and M.FLAG_REASONING_PARSER not in st and M.FLAG_EXPECT_OBJECT_SHA256 not in st
        assert f"GEOMETRY: capability:gen_long 4608 x {s} cap 512 thinking off" in t and "IN SUPPORT: route=base stamps=(cell-stable,not-consulted,thinking_off)" in t


# --------------------------------------------------------------------------- #
# the literals mirror contracts and the geometry table                          #
# --------------------------------------------------------------------------- #


def test_the_literals_mirror_contracts_and_the_geometry_table(s_by):
    assert M.FLAG_THINKING == C.FLAG_THINKING and M.FLAG_SAMPLING == C.FLAG_SAMPLING and M.FLAG_SAMPLES == C.FLAG_SAMPLES
    assert M.FLAG_LEDGER_URL == C.FLAG_LEDGER_URL and M.FLAG_OOS_PROMPT_BUDGET == C.FLAG_OOS_PROMPT_BUDGET
    assert M.FLAG_ALLOW_OOS_THINKING == C.FLAG_ALLOW_OOS_THINKING and M.FLAG_EXPECT_OBJECT_SHA256 == C.FLAG_EXPECT_OBJECT_SHA256
    assert M.FLAG_MAX_TOKENS_BY_TASK == C.FLAG_MAX_TOKENS_BY_TASK and M.FLAG_TINYCAP_IDENTITY_VERSION == C.FLAG_TINYCAP_IDENTITY_VERSION
    assert M.FLAG_UPSTREAM_TIMEOUT_S == C.FLAG_UPSTREAM_TIMEOUT_S and M.FLAG_REASONING_PARSER == C.FLAG_REASONING_PARSER
    # the three flags by NAME: pinned equal once contracts carries them
    for mine, theirs in (("FLAG_ON_CAP_BY_TASK", "--on-cap-by-task"), ("FLAG_ITEM_SUBSET", "--item-subset"), ("FLAG_ATTEMPT_ID", "--attempt-id")):
        assert getattr(M, mine) == theirs == getattr(C, mine, theirs)
    assert M.UPSTREAM_TIMEOUT_S == C.PROXY_UPSTREAM_TIMEOUT_S == 7200 and M.IDENTITY_VERSION == 2
    assert M.CONTROL == C.CONTROL_GSM8K == "gsm8k" and M.INSPECT_TASK == C.GSM8K_PAIRED_TASK_NAME == "gsm8k_paired200"
    assert M.ROW_TASK == C.GSM8K_STANDARD_TASK == C.inspect_task(M.INSPECT_TASK) == "inspect/gsm8k_paired200"   # B-P1.2: the ROW task
    assert LP.ROW_TASK is C.GSM8K_STANDARD_TASK and LP.RUNG_SUBSET_SCHEMA is C.ITEM_SUBSET_SCHEMA and LP.ITEM_SUBSET_FIELDS is C.ITEM_SUBSET_FIELDS
    assert C.TASKS_OF_CONTROL[M.CONTROL] == (C.inspect_task(M.INSPECT_TASK),) and C.THINKING_BY_TASK[C.inspect_task(M.INSPECT_TASK)] == C.THINKING_ON
    assert M.THINKING_MODE == C.THINKING_ON == C.check_thinking_arg("on") and M.OVERLAP_THINKING == C.THINKING_OFF
    assert M.OVERLAP_SAMPLING == C.SAMPLING_GREEDY and M.SAMPLING_THINKING_CARD in C.SAMPLING_MODES
    assert M.N_ITEMS == C.GSM8K_PAIRED_N == 200 and M.HEADER_INVARIANT_THINKING_FAMILIES == C.HEADER_INVARIANT_THINKING_FAMILIES
    assert M.LEDGER_FILENAME == C.LEDGER_FILENAME and M.LEDGER_ENDPOINT_FILENAME == C.LEDGER_ENDPOINT_FILENAME
    assert M.OOS_PROMPT_BUDGET_FILENAME == C.OOS_PROMPT_BUDGET_FILENAME and M.INSPECT_BASE_URL_ENV == C.INSPECT_BASE_URL_ENV
    assert M.INSPECT_API_KEY_ENV == C.INSPECT_API_KEY_ENV and M.LENS_SALT_POLICY == "attested"
    from sisteer.capeval.tiers import cap_ladder as cl
    assert M.RUNG_CAP == cl.CAP_CEILING == 16384 and M.CAP_LADDER_FILENAME == cl.RECORD_NAME and M.CAP == 8192
    # llama-paper-rows (P8): the no-switch diagnostic smoke's literals — the committed dir is the contracts', the probe's own
    # defaults are the launcher's (ONE source: g8k_cells), the arm / split / class it certifies are the contracts' literals
    assert M.SMOKE_COMMITTED_DIR_REL == C.NO_SWITCH_SMOKE_COMPLETIONS_DIR == "data/tinycap/no_switch_smoke"
    assert (M.SMOKE_N, M.SMOKE_CAP, M.SMOKE_CONCURRENCY, M.SMOKE_DIR_REL) == (24, 1536, 8, "diagnostic")
    smoke_probe = _load("_g8k_smoke_probe_lits", M.SMOKE_PROBE_SCRIPT_REL)
    assert smoke_probe.DEFAULT_CONCURRENCY == M.SMOKE_CONCURRENCY and smoke_probe.IDENTITY_VERSION == M.IDENTITY_VERSION
    assert smoke_probe.ARM == C.THINKING_NOT_AVAILABLE == M.THINKING_NOT_AVAILABLE and smoke_probe.SPLIT_MODE == C.THINK_SPLIT_NONE
    assert (REPO / M.SMOKE_PROBE_SCRIPT_REL).is_file() and M.SMOKE_CAP not in C.ON_CAP_BY_TASK_ALLOWED[C.GSM8K_PAIRED_TASK_NAME]
    for key, row in M.SAMPLING_ROWS.items():
        # each model's row is the CONTRACT's row under the model's sampling MODE (thinking_card x4; llama8b's `card`, P6)
        crow = C.SAMPLING_TABLE_BY_MODE[M.SAMPLING_MODE_BY_MODEL[key]][key]
        assert (M.SAMPLING_MODE_BY_MODEL[key] == M.SAMPLING_CARD) is (M.THINKING_MODE_BY_FAMILY[M.FAMILY_BY_MODEL[key]] == M.THINKING_NOT_AVAILABLE)
        for field in M.SAMPLING_PARAM_KEYS:
            assert getattr(crow, field, None) == row.get(field), (key, field)
        assert set(row) - {"source"} <= set(M.SAMPLING_PARAM_KEYS) and row["source"]
    # the geometry: the 8,192 pass on the entry the class map resolves; the rung on its own; the OFF leg on gen_long
    assert M.GEOMETRY_EVAL_BY_RUNG[8192] == C.geometry_subentry(C.GEOMETRY_CLASS_OF_CONTROL[C.CONTROL_GSM8K], prompted=False, thinking=True) == "capability:gen_long_thinking"
    assert M.OVERLAP_GEOMETRY_EVAL == C.geometry_subentry(C.GEOMETRY_CLASS_OF_CONTROL[C.CONTROL_GSM8K], prompted=False, thinking=False) == "capability:gen_long"
    assert M.GEOMETRY_EVAL_BY_RUNG[16384] == "capability:gen_long_gsm8k_rung"
    from sisteer.capeval.serving import geometry as sg
    assert {r: sg.TABLE[M.GEOMETRY_EVAL_BY_RUNG[r]].max_model_len for r in M.RUNGS} == M.GEOMETRY_L_BY_RUNG == {8192: 10752, 16384: 18944}
    assert sg.TABLE[M.OVERLAP_GEOMETRY_EVAL].max_model_len == M.OVERLAP_GEOMETRY_L == 4608
    # the measured maxima (G-LENGTHS, 2026-09-10) fit each entry at its cap
    assert 2462 + 8192 <= 10752 and 2462 + 16384 <= 18944 and 2375 + 512 <= 4608
    # llama8b: the MEASURED 475,136-token budget (job 476440; geometry._STANDUP_MEASURED_BUDGETS) -> 44 at 10,752, 25 at 18,944 (P20);
    # qwen35_4b: the swept 720,896 since 2026-09-11 (its halved ceiling lifted)
    assert {k: s_by[(k, 8192)] for k in M.MODEL_KEYS} == {"gemma4e4b": 54, "qwen35_4b": 69, "nemotron3nano4b": 73, "glm46vflash": 73, "llama8b": 44}   # qwen35_4b: 741,888 // 10752 (the 2026-09-13 row)
    assert {k: s_by[(k, 16384)] for k in M.MODEL_KEYS} == {"gemma4e4b": 31, "qwen35_4b": 39, "nemotron3nano4b": 41, "glm46vflash": 41, "llama8b": 25}   # qwen35_4b: 741,888 // 18944
    # the time classes (x2 the decode-bound estimate) and the overlap class
    assert {M.time_limit_hours(t) for t in (M.TIME_IN_SUPPORT, M.TIME_FITTED, M.TIME_FITTED_GLM, M.TIME_OVERLAP)} == {1.0, 1.5, 2.0, 0.75}
    assert {t: len(ix) for t, ix in M.time_classes(LAUNCHED).items()} == {M.TIME_IN_SUPPORT: 10, M.TIME_FITTED: 22, M.TIME_FITTED_GLM: 6}
    assert M.WAVE_ROOT_REL == "exp/gsm8k_paired200_paper_objects" and M.WAVE_ROOT_REL not in (M5T / "m5t_common.sh").read_text()


# --------------------------------------------------------------------------- #
# the plan: ONE aggregate allowance; disjoint shares; the overflow complement  #
# --------------------------------------------------------------------------- #


def _covered(lines):
    covered = []
    for line in lines:
        spec = re.search(r"--array=([0-9,\-]+)%", line).group(1)
        for part in spec.split(","):
            a, _, b = part.partition("-")
            covered += list(range(int(a), int(b or a) + 1))
    return covered


def _widths(lines):
    return [int(re.search(r"%(\d+) ", line).group(1)) for line in lines]


def test_plan_partitions_w_across_the_classes_and_cluster_share_yields_disjoint_covering_sets():
    sizes = {tl: len(ix) for tl, ix in M.time_classes(LAUNCHED).items()}
    hours = {tl: M.time_limit_hours(tl) for tl in sizes}

    def tail(ws):
        return max(-(-sizes[tl] // ws[tl]) * hours[tl] for tl in sizes)
    for W in (3, 4, 8, 12, 16, 24, 38, 40):
        pl = M.plan(SHA0, "cluster_a", max_concurrent=W, universe=ROWS)      # the published wave's shape (the python universe entry)
        assert sum(_widths(pl.lines)) == sum(pl.widths.values()) == min(W, 38) <= W
        assert all(1 <= pl.widths[tl] <= sizes[tl] for tl in sizes)
        best = min(tail(dict(zip(sizes, ws))) for ws in itertools.product(*(range(1, sizes[tl] + 1) for tl in sizes)) if sum(ws) <= W)
        assert pl.est_tail_h == tail(pl.widths) == best, (W, pl.widths)
        assert sorted(_covered(pl.lines)) == [r["index"] for r in LAUNCHED]
        assert all(line.startswith("sbatch -J g8k-") and line.endswith(f"gsm8k_thinking.sbatch ARRAY {SHA0}") and f"--qos={SITE_A['qos']} " in line for line in pl.lines)
    assert list(M.plan(SHA0, "cluster_a", max_concurrent=12, universe=ROWS).widths) == [M.TIME_FITTED_GLM, M.TIME_FITTED, M.TIME_IN_SUPPORT]
    with pytest.raises(SystemExit, match="time classes"):
        M.plan(SHA0, "cluster_a", max_concurrent=2, universe=ROWS)
    with pytest.raises(SystemExit, match="positive"):
        M.plan(SHA0, "cluster_a", max_concurrent=0, universe=ROWS)
    # the launch shape: the 8 Nemotron rows on the primary, the 30 others on the overflow — ONE share string
    share = "26-33"
    b = M.plan(SHA0, "cluster_a", max_concurrent=8, cluster_share=share, universe=ROWS)
    c = M.plan(SHA0, "cluster_b", max_concurrent=16, cluster_share=share, universe=ROWS)
    assert b.rows == list(range(26, 34)) and len(c.rows) == 30 and set(b.rows).isdisjoint(c.rows)
    assert sorted(set(b.rows) | set(c.rows)) == [r["index"] for r in LAUNCHED]
    assert all(f"--qos={SITE_B['qos']} " in line and f"--output={SITE_B['logs']}/" in line and f"{M.CHECKOUT_ROOT}/{SHA0}/" in line for line in c.lines)
    assert all(f"--qos={SITE_A['qos']} " in line and f"--output={SITE_A['logs']}/" in line for line in b.lines)
    assert sum(c.widths.values()) == 16 and sum(b.widths.values()) == 8 and len(b.lines) == 2 and len(c.lines) == 3
    e = M.plan(SHA0, "cluster_b", max_concurrent=3, rows="34-36", universe=ROWS)
    assert e.rows == [34, 35, 36] and e.widths == {M.TIME_FITTED_GLM: 2, M.TIME_IN_SUPPORT: 1}
    for spec, why in (("0-3,2", "twice"), ("42", "outside"), ("0-x", "not an index")):
        with pytest.raises(SystemExit, match=why):
            M.parse_index_spec(spec, 42)
    with pytest.raises(SystemExit, match="exclusive"):
        M.plan(SHA0, "cluster_a", max_concurrent=4, rows="4-5", cluster_share="4-5", universe=ROWS)
    with pytest.raises(SystemExit, match="no rows"):
        M.plan(SHA0, "cluster_b", max_concurrent=4, cluster_share="0-1,4-14,17-41", universe=ROWS)
    # the CLI is WAVE-SCOPED since E2: the published wave is COMPLETE (its 38 legs reused byte-for-byte), so the published launch
    # shape above lives at the python entry point alone — the CLI refuses a matrix-wide plan (no --wave) and the published wave by name
    r = subprocess.run([sys.executable, str(G8K / "g8k_cells.py"), "plan", "--sha", SHA0, "--cluster", "cluster_b", "--cluster-share", share,
                        "--max-concurrent", "16"], capture_output=True, text=True, check=False)
    assert r.returncode == 2 and "--wave" in r.stderr and "sbatch " not in r.stdout
    r = subprocess.run([sys.executable, str(G8K / "g8k_cells.py"), "plan", "--sha", SHA0, "--cluster", "cluster_b", "--cluster-share", share,
                        "--max-concurrent", "16", "--wave", "published"], capture_output=True, text=True, check=False)
    assert r.returncode != 0 and "COMPLETE" in r.stderr and "never resubmitted" in r.stderr and "sbatch " not in r.stdout
    held_spec = M._array_spec([row["index"] for row in M.held_rows()])          # 2-3,15-16 (+ the trailing held rows when the roster declares any)
    assert held_spec.startswith("2-3,15-16")
    assert f"held={held_spec}" in "\n".join(c.summary()) and "for " not in "\n".join(c.lines + c.summary())
    assert f"rows=30/{len(ROWS)}" in "\n".join(c.summary()) and "peers" in "\n".join(c.summary()) and SITE_B["qos"] in "\n".join(c.lines)


# --------------------------------------------------------------------------- #
# the ledger sequencing, the standup extras, the attempt seam                  #
# --------------------------------------------------------------------------- #


def _f22_standup_extra_reset():
    """common.sh's f22_standup_extra_reset (2026-09-16), which g8k_standup_extra calls: production sources
    full22/common.sh before g8k_common.sh (overlap_off.sbatch), so a harness that runs the lane function alone
    carries the helper the same way."""
    sh = (F22 / "common.sh").read_text()
    i = sh.index("f22_standup_extra_reset(){")
    return sh[i:sh.index("\n}\n", i) + 3]


def _g8k_funcs():
    return ('set -u\nf22_log(){ echo LOG "$@"; }\nf22_fatal(){ echo FATAL "$@"; exit 1; }\n'
            + _f22_standup_extra_reset() + f'source {G8K / "g8k_common.sh"}\n')


def test_a_ledger_that_never_writes_its_endpoint_fails_the_job_and_the_driver_never_starts(tmp_path):
    stub = tmp_path / "python"
    stub.write_text('''#!/bin/bash
case "$*" in
  *"socket"*) echo 45678 ;;
  *"usage_ledger"*) echo "$*" > "$ARGV_LOG"; if [ "${LEDGER_WRITES:-0}" = 1 ]; then for a in "$@"; do case "$prev" in --endpoint-json) ep="$a";; esac; prev="$a"; done
                     sleep 1; echo '{"ledger_url": "http://127.0.0.1:45678/v1", "upstream": "'"$UPSTREAM"'"}' > "$ep"; fi; sleep 30 ;;
  *"ledger_url"*) python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['ledger_url'])" "${@: -1}" ;;
  *) exit 0 ;;
esac
''')
    stub.chmod(0o755)
    serve = tmp_path / "serving/gsm8k"; serve.mkdir(parents=True)
    (tmp_path / "out").mkdir()
    base = (_g8k_funcs() + f"\nCPY={stub}; REPO_DIR={REPO}; PROXY_URL=http://127.0.0.1:1/v1; UPSTREAM=$PROXY_URL; OUT_ROOT={tmp_path}/out; "
            f"SERVE_DIR={tmp_path}/serving; TABLE={serve}/routing_table.json; G8K_LEDGER_WAIT_S=3; G8K_LEDGER_FILENAME=usage_ledger.jsonl; "
            "G8K_LEDGER_ENDPOINT_FILENAME=ledger_endpoint.json; G8K_SERVE_SUBDIR=gsm8k; ATTEMPT_ID=a7; G8K_RESULT=INCOMPLETE; LEDGER_PID=''\n")
    env = {**os.environ, "ARGV_LOG": str(tmp_path / "argv.txt")}
    r = subprocess.run(["bash", "-c", base + 'g8k_ledger_up || { echo "RESULT: $G8K_RESULT"; g8k_ledger_down; exit 1; }; echo DRIVER_WOULD_START'],
                       capture_output=True, text=True, env={**env, "LEDGER_WRITES": "0"}, timeout=60, check=False)
    assert r.returncode == 1 and "RESULT: LEDGER_FAILED" in r.stdout and "DRIVER_WOULD_START" not in r.stdout, (r.stdout, r.stderr)
    r = subprocess.run(["bash", "-c", base + 'g8k_ledger_up || exit 1; echo "URL=$LEDGER_URL"; g8k_ledger_down; echo DRIVER_WOULD_START'],
                       capture_output=True, text=True, env={**env, "LEDGER_WRITES": "1"}, timeout=60, check=False)
    assert r.returncode == 0 and "URL=http://127.0.0.1:45678/v1" in r.stdout and "DRIVER_WOULD_START" in r.stdout, (r.stdout, r.stderr)
    assert (serve / "ledger_endpoint.json").is_file()
    assert "--attempt-id a7" in (tmp_path / "argv.txt").read_text()        # every ledger row will carry the attempt
    sb = (G8K / "gsm8k_thinking.sbatch").read_text()
    assert sb.index("g8k_ledger_up || exit 1") < sb.index("g8k_driver_argv\n")


def test_standup_extra_declares_only_header_located_cells_and_the_digest_follows_fitted():
    for row in LAUNCHED + M.held_rows():
        script = (_g8k_funcs() + f"\nG8K_UPSTREAM_TIMEOUT_S=7200; G8K_LENS_SALT_POLICY=attested; CELL={row['cell']}; FITTED={int(row['fitted'])}; "
                  f"G8K_STANDUP_CELL={row['standup_cell']}; "
                  f"DECLARED={int(row['declared'])}; PIN='{row['object_sha256'] or ''}'; G8K_REASONING_PARSER='{row['reasoning_parser'] or ''}'; G8K_THINKING=on\n"
                  'g8k_standup_extra; printf "%s\\n" "${F22_STANDUP_EXTRA[@]}"')
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
        assert r.returncode == 0, r.stderr
        want = ["--upstream-timeout-s", "7200", "--lens-salt-policy", "attested", "--enable-log-requests", "--cell", row["standup_cell"]]
        if row["fitted"]:
            want += ["--expect-object-sha256", f"{row['standup_cell']}={row['object_sha256']}"]      # B-P1.1: the SAME spelling
        if row["reasoning_parser"]:
            want += ["--reasoning-parser", row["reasoning_parser"]]
        assert r.stdout.splitlines() == want, (row["job_id"], r.stdout)


def test_the_attempt_seam_mints_attempts_files_their_ends_and_reads_the_accepted_one_from_the_manifest(tmp_path):
    state = tmp_path / "out" / "ladder_state.json"
    (tmp_path / "out").mkdir()
    core = f'set -u\nf22_log(){{ echo LOG "$@"; }}\nf22_fatal(){{ echo FATAL "$@"; exit 1; }}\nsource {F22 / "served_thinking.sh"}\nCPY={sys.executable}\n'

    def run(body):
        r = subprocess.run(["bash", "-c", core + body], capture_output=True, text=True, check=False)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout.strip()
    # a1: a generation attempt that dies (DRIVER_FAILED): ended / rc / result filed, nothing accepted
    aid = run(f'st_attempt_begin {state} \'{{"kind": "generation", "job_id": "1", "rung": 8192}}\'; echo $ATTEMPT_ID; st_attempt_end {state} 2 DRIVER_FAILED {tmp_path}/out')
    doc = json.loads(state.read_text())
    assert aid == "a1" and doc["schema"] == "g8k_ladder_state/1" and doc["accepted_generation_attempt"] is None
    assert doc["attempts"][0]["attempt_id"] == "a1" and doc["attempts"][0]["rc"] == 2 and doc["attempts"][0]["result"] == "DRIVER_FAILED"
    assert doc["attempts"][0]["ended"] and doc["attempts"][0]["job_id"] == "1" and doc["attempts"][0]["rung"] == 8192
    # a2: a clean attempt with no manifest naming an earlier one -> accepted a2, source the launcher
    aid = run(f'st_attempt_begin {state} \'{{"kind": "generation", "job_id": "2"}}\'; echo $ATTEMPT_ID; st_attempt_end {state} 0 COMPLETE {tmp_path}/out')
    doc = json.loads(state.read_text())
    assert aid == "a2" and doc["accepted_generation_attempt"] == "a2" and doc["accepted_source"].startswith("launcher") and len(doc["attempts"]) == 2
    # a3: a FINALISATION — the driver's manifest names the accepted generation attempt (a2); the launcher records it, not itself
    man = tmp_path / "out" / "cells" / "cellkey" / "manifest.json"
    man.parent.mkdir(parents=True)
    man.write_text(json.dumps({"accepted_generation_attempt": "a2", "resume_attempts": [{"attempt_id": "a3", "kind": "finalisation"}]}))
    aid = run(f'st_attempt_begin {state} \'{{"kind": "generation", "job_id": "3"}}\'; echo $ATTEMPT_ID; st_attempt_end {state} 0 COMPLETE {tmp_path}/out')
    doc = json.loads(state.read_text())
    assert aid == "a3" and doc["accepted_generation_attempt"] == "a2" and "manifest" in doc["accepted_source"]
    assert [a["attempt_id"] for a in doc["attempts"]] == ["a1", "a2", "a3"]
    # st_attempt_end before any attempt is a no-op (the cleanup trap calls it at any step)
    run(f'ATTEMPT_ID=""; st_attempt_end {state} 1 INCOMPLETE {tmp_path}/out; echo ok')
    assert [a["attempt_id"] for a in json.loads(state.read_text())["attempts"]] == ["a1", "a2", "a3"]


# --------------------------------------------------------------------------- #
# the NON-dry startup: a fresh batch process, set -u, no REPO_DIR              #
# --------------------------------------------------------------------------- #

STARTUP_SHA = "c" * 40
STARTUP_USER = "g8k-startup-nobody"
STARTUP_FORMS = [
    ("gsm8k_thinking.sbatch", ["gemma4e4b", "main", "none", "8192", STARTUP_SHA, "0", "0"], {}),
    ("gsm8k_thinking.sbatch", ["ARRAY", STARTUP_SHA], {"SLURM_ARRAY_TASK_ID": "4"}),
    ("gsm8k_thinking.sbatch", ["ARRAY", STARTUP_SHA, "16384"], {"SLURM_ARRAY_TASK_ID": "22"}),
    ("overlap_off.sbatch", ["qwen35_4b", STARTUP_SHA], {}),
    ("overlap_off.sbatch", ["ARRAY", STARTUP_SHA], {"SLURM_ARRAY_TASK_ID": "1"}),
    ("smoke_not_available.sbatch", ["llama8b", "main", STARTUP_SHA], {}),
    ("dry_run_prod.sbatch", [STARTUP_SHA], {}),
    ("ladder_plan.sbatch", [STARTUP_SHA], {}),
]


def _fake_checkout(tmp_path, *, with_core=True):
    co = tmp_path / "checkout"
    rels = ["scripts/sdfbase/full22/common.sh", "scripts/sdfbase/gsm8k/g8k_common.sh", "scripts/sdfbase/gsm8k/g8k_cells.py",
            "configs/sdfbase/gsm8k/matrix.json"] + (["scripts/sdfbase/full22/served_thinking.sh"] if with_core else [])
    for rel in rels:
        (co / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, co / rel)
    (co / "src/sisteer").mkdir(parents=True)
    return co


def _startup_env(tmp_path, name, extra, **overrides):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "USER": STARTUP_USER, "SLURM_JOB_ID": "424242",
           "SLURM_JOB_NAME": f"g8k-startup-{name}", "CUDA_VISIBLE_DEVICES": "0", **extra, **overrides}
    assert "REPO_DIR" not in env
    return env


def _run_startup(tmp_path, name, args, env):
    r = subprocess.run(["bash", "-u", str(G8K / name), *args], env=env, capture_output=True, text=True, timeout=60, cwd=tmp_path, check=False)
    return r, r.stdout + r.stderr


@pytest.mark.parametrize("name,args,extra", STARTUP_FORMS)
def test_non_dry_startup_under_set_u_with_no_repo_dir_reaches_g8k_init_and_refuses_by_name(tmp_path, name, args, extra):
    env = _startup_env(tmp_path, name, extra, G8K_REPO_DIR=str(_fake_checkout(tmp_path)))
    r, out = _run_startup(tmp_path, name, args, env)
    assert "unbound variable" not in out, out
    assert r.returncode == 1, (r.returncode, out)
    assert "FATAL: set SISTEER_SHARED_ROOT, the shared root of the cluster" in out, out   # the job's site value; no storage path is probed
    assert "RESULT" in out and "rc=1" in out and "result=INCOMPLETE" in out, out
    # the same startup on a checkout WITHOUT the shared core still files its RESULT (the trap's teardown is a no-op there)
    r, out = _run_startup(tmp_path, name, args, _startup_env(tmp_path, name, extra, G8K_REPO_DIR=str(_fake_checkout(tmp_path / "nocore", with_core=False))))
    assert r.returncode == 1 and "RESULT" in out and "result=INCOMPLETE" in out and "command not found" not in out, out


@pytest.mark.parametrize("name,args,extra", STARTUP_FORMS)
def test_startup_refuses_a_missing_shared_root_or_checkout_by_name_before_sourcing_anything(tmp_path, name, args, extra):
    r, out = _run_startup(tmp_path, name, args, _startup_env(tmp_path, name, extra))
    assert r.returncode == 1 and "unbound variable" not in out, out
    assert "set SISTEER_EVAL_REPO_ROOT" in r.stderr and "nothing to source" in r.stderr, out
    assert "RESULT" not in out
    empty = tmp_path / "empty"
    empty.mkdir()
    r, out = _run_startup(tmp_path, name, args, _startup_env(tmp_path, name, extra, G8K_REPO_DIR=str(empty)))
    assert r.returncode == 1 and "unbound variable" not in out, out
    assert f"FATAL: no checkout at {empty}" in r.stderr and "worktree add --detach" in r.stderr and STARTUP_SHA in r.stderr, out
    assert "RESULT" not in out


def test_a_bad_rung_or_fitted_positional_is_refused_before_anything_is_sourced(tmp_path):
    for args, why in ((["gemma4e4b", "main", "none", "4096", SHA0, "0", "0"], "rung must be 8192 or 16384"),
                      (["gemma4e4b", "main", "none", "8192", SHA0, "2", "0"], "fitted must be 0 or 1"),
                      (["gemma4e4b", "main", "none", "8192", SHA0, "0", "x"], "declared must be 0 or 1")):
        r = subprocess.run(["bash", "-u", str(G8K / "gsm8k_thinking.sbatch"), *args], capture_output=True, text=True, check=False, env=_startup_env(tmp_path, "x", {}))
        assert r.returncode == 1 and why in r.stderr and "RESULT" not in r.stdout + r.stderr


# --------------------------------------------------------------------------- #
# the two-stage dry run: expected positions; the dose-view production path      #
# --------------------------------------------------------------------------- #


def _facts(status):
    return {fam: types.SimpleNamespace(split_mode=M.SPLIT_MODE_CANDIDATE[fam], evidence_status=status, off_ids=(1, 2), on_ids=(1, 3))
            for fam in set(M.FAMILY_BY_MODEL.values())}


def _stage_bank(tmp_path, key, view="main"):
    """EVERY fitted cell of the (key, view) TAG staged as a view-shaped bank (the overlay lists the whole view and the
    registry validates every object cell against the bank; the m5t overlay had one object, ours has the view's)."""
    from test_sdfbase_full22_wave import _stage_derived
    bank = tmp_path / key / "cells_view" / view
    bank.mkdir(parents=True, exist_ok=True)
    for r in ROWS:
        if r["model_key"] == key and r["view"] == view and r["fitted"]:
            _stage_derived(bank, r["cell"], r["cell"])
    return bank


def test_two_stage_dry_run_checks_the_fresh_stamp_against_the_rows_expected_positions(tmp_path, monkeypatch, capsys):
    dry = _load("_g8k_dry_run_stages", "scripts/sdfbase/gsm8k/g8k_dry_run.py")
    cell = "constv2-mix-full__lora_r4_ce_minus_ctl"                       # an all-position derived object of qwen main
    only = ["--only", "qwen35_4b", "main", cell, "--tmp", str(tmp_path / "t"), "--python", PY]

    def refusing(family):
        raise ValueError(f"render fact for {family} is candidate, not certified")
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", _facts("candidate"), raising=False)
    monkeypatch.setattr(C, "check_render_fact_certified", refusing, raising=False)
    assert dry.main(["--stage", "pre", *only]) == 0
    out = capsys.readouterr().out
    assert "evidence_status=candidate" in out and "refuses=True" in out and "VERDICT: PASS stage=pre" in out and out.count("G-LENS: ") == 1
    assert dry.main(["--stage", "prod", *only]) == 2
    assert "REFUSED: render fact not certified for family qwen3_5" in capsys.readouterr().out
    monkeypatch.setattr(C, "check_render_fact_certified", lambda family: None, raising=False)
    assert dry.main(["--stage", "pre", *only]) == 2
    assert "production preflight ADMITS" in capsys.readouterr().out
    # certified facts + a fresh table over a view-shaped bank: prod PASSES, the stamp equals the row's expected `all`
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", _facts("certified"), raising=False)
    bank_root = tmp_path / "bank"
    _stage_bank(bank_root, "qwen35_4b")
    assert dry.main(["--stage", "prod", *only, "--bank-root", str(bank_root)]) == 0
    out = capsys.readouterr().out
    row = M.row_for("qwen35_4b", "main", cell)
    assert f'SERVED_POSITIONS: row={row["index"]} lens_served_positions[{row["harness_cell"]}] == "all" (expected)' in out and "VERDICT: PASS stage=prod" in out
    assert f"bank={bank_root}/qwen35_4b/cells_view/main" in out
    assert dry.main(["--stage", "prod", *only]) == 2
    assert "requires the fresh-table check" in capsys.readouterr().out
    # a stamp DISAGREEING with the row's expectation is refused at prod by name (a `response` stamp on an all-position row)
    from sisteer.capeval.serving import proxy as px
    real = px.export_routing_table

    def response_stamp(*a, **kw):
        real(*a, **kw)
        path = Path(a[2]); doc = json.loads(path.read_text())
        doc["lens_served_positions"] = {k: ["response"] for k in doc["lens_served_positions"]}
        # one standup writes two CONSISTENT records (native-positions-resolve): the engine-neutral map beside the lens map
        # carries the same stamp — a tear between the two is the loader's own refusal, before the dry run's check
        if px.SERVED_POSITIONS_KEY in doc:
            doc[px.SERVED_POSITIONS_KEY] = {k: ["response"] for k in doc[px.SERVED_POSITIONS_KEY]}
        path.write_text(json.dumps(doc))
    monkeypatch.setattr(px, "export_routing_table", response_stamp)
    assert dry.main(["--stage", "prod", *only, "--bank-root", str(bank_root)]) == 2
    out = capsys.readouterr().out
    assert f"REFUSED: lens_served_positions[{row['harness_cell']}] = ['response'], not ['all'] (row {row['index']}: served_positions_expected=all)" in out

    def stale(*a, **kw):
        real(*a, **kw)
        path = Path(a[2]); doc = json.loads(path.read_text()); doc.pop("lens_served_positions", None); path.write_text(json.dumps(doc))
    monkeypatch.setattr(px, "export_routing_table", stale)
    assert dry.main(["--stage", "prod", *only, "--bank-root", str(bank_root)]) == 2
    assert "REFUSED: served-positions stamp absent — stale table or refinement not merged" in capsys.readouterr().out
    monkeypatch.setattr(px, "export_routing_table", real)
    assert dry.main(["--stage", "prod", *only, "--bank-root", str(bank_root)]) == 0
    # the it view's bank path is the VIEW's: a prompted it row exports over cells_view/it (no object needed)
    it_row = M.row_for("gemma4e4b", "it", "prompted__safety_refusal")
    _stage_bank(bank_root, "gemma4e4b", view="it")
    assert dry.main(["--stage", "prod", "--only", "gemma4e4b", "it", "prompted__safety_refusal", "--tmp", str(tmp_path / "t2"), "--python", PY,
                     "--bank-root", str(bank_root)]) == 0
    out = capsys.readouterr().out
    assert f"bank={bank_root}/gemma4e4b/cells_view/it" in out and f"FRESH_TABLE: row={it_row['index']}" in out and "own_route=True" in out


def _synthetic_peer_tree(root: Path, *, pin: str | None = "ab" * 32):
    """A peer's checkout, SYNTHETIC: the real roster with an `it_d0p1` dose view added on gemma4e4b (the shape of the strength
    wave's), placeholder pins (TESTHOST), the campaign config / manifest of the view cut from the real `_it` pair."""
    roster = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())
    roster["targets"]["gemma4e4b"]["views"]["it_d0p1"] = {
        "dose": 0.1, "config": "configs/sdfbase_full22_gemma4e4b_it_d0p1.yaml",
        "manifest": "configs/capeval/manifests/petri_full22_gemma4e4b_it_d0p1.yaml",
        "cells": {"constv2-mix-full__lora_r4_ce": "d0p1it__constv2-mix-full__lora_r4_ce"}}
    (root / "configs/capeval/manifests").mkdir(parents=True)
    (root / "configs/sdfbase_full22_roster.yaml").write_text(yaml.safe_dump(roster, sort_keys=False))
    pins = {"gemma4e4b": {}}
    if pin is not None:
        pins["gemma4e4b"]["d0p1it__constv2-mix-full__lora_r4_ce"] = {"object_sha256": pin, "source": "TESTHOST:/synthetic", "verified_cluster_a": True}
    (root / "configs/sdfbase_full22_pins.json").write_text(json.dumps(pins, indent=1))
    cfg = (REPO / "configs/sdfbase_full22_gemma4e4b_it.yaml").read_text()
    cfg = re.sub(r"^explicit_cells:\n(?:- .*\n)+", "explicit_cells:\n- none\n- constv2-mix-full__lora_r4_ce\n", cfg, flags=re.MULTILINE)
    (root / "configs/sdfbase_full22_gemma4e4b_it_d0p1.yaml").write_text(cfg)
    man = (REPO / "configs/capeval/manifests/petri_full22_gemma4e4b_it.yaml").read_text()
    man = re.sub(r"^  cells:\n(?:    - .*\n)+", "  cells:\n    - none\n    - si_constv2_mix_full__lora_r4_ce\n", man, flags=re.MULTILINE)
    (root / "configs/capeval/manifests/petri_full22_gemma4e4b_it_d0p1.yaml").write_text(man)
    return root


def _tree_listing(root: Path):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_the_dose_view_production_path_over_a_synthetic_peer_tree(tmp_path, capsys):
    """P1-7 / spec: `it_d0p1` at a peer's tree becomes a PRODUCTION row through the ONE selector — its matrix, its overlay pair
    and the sbatch's ACTUAL argv construction (cap 8192, thinking on, the view's bank path and pin), stopping before the
    standup / ledger / driver; nothing is written under the peer's tree; a missing pin refuses by name."""
    peer = _synthetic_peer_tree(tmp_path / "peer")
    before = _tree_listing(peer)
    dry = _load("_g8k_dry_run_dose", "scripts/sdfbase/gsm8k/g8k_dry_run.py")
    rc = dry.main(["--stage", "pre", "--views", "gemma4e4b=it_d0p1", "--configs-root", str(peer), "--tmp", str(tmp_path / "t"), "--python", PY])
    out = capsys.readouterr().out
    assert rc == 0, out[-1500:]
    assert _tree_listing(peer) == before                                       # the peer's tree is never written
    tree = tmp_path / "t" / "tree"
    assert f"SCRATCH_TREE: {tree}" in out
    matrix = json.loads((tree / "configs/sdfbase/gsm8k/matrix.json").read_text())
    assert [(r["tag"], r["cell"], r["status"]) for r in matrix["rows"]] == [("gemma4e4b_it_d0p1", "none", "launch"), ("gemma4e4b_it_d0p1", "constv2-mix-full__lora_r4_ce", "launch")]
    assert matrix["rows"][1]["object_sha256"] == "ab" * 32 and matrix["rows"][1]["staged_id"] == "d0p1it__constv2-mix-full__lora_r4_ce" and matrix["views_spec"] == "gemma4e4b=it_d0p1"
    assert yaml.safe_load((tree / "configs/sdfbase/gsm8k/sdfbase_full22_gemma4e4b_it_d0p1.gsm8k.yaml").read_text())["explicit_cells"] == ["none", "constv2-mix-full__lora_r4_ce"]
    assert yaml.safe_load((tree / "configs/capeval/manifests/petri_full22_gemma4e4b_it_d0p1.gsm8k.yaml").read_text())["petri"]["cells"] == ["none", "si_constv2_mix_full__lora_r4_ce"]
    drivers = [line for line in out.splitlines() if line.startswith("DRIVER:")]
    standups = [line for line in out.splitlines() if line.startswith("STANDUP:")]
    assert len(drivers) == len(standups) == 2
    d = drivers[1]
    assert "--thinking on " in d and "--max-tokens-by-task gsm8k_paired200=8192 " in d and "--on-cap-by-task gsm8k_paired200=8192 " in d
    assert f"--defenses-dir {tmp_path / 't' / 'shared'}/exp/objects/sdfbase/full22/gemma4e4b/cells_view/it_d0p1 " in d
    assert f"--expect-object-sha256 {'ab' * 32}" in d and "--allow-out-of-support-thinking" in d
    assert "--config configs/sdfbase/gsm8k/sdfbase_full22_gemma4e4b_it_d0p1.gsm8k.yaml " in d and "--cell constv2-mix-full__lora_r4_ce " in d
    assert f"--expect-object-sha256 constv2-mix-full__lora_r4_ce={'ab' * 32}" in standups[1] and "@thinking" not in standups[1]
    assert f"cells_view/it_d0p1/constv2-mix-full__lora_r4_ce pin={'ab' * 32}" in out
    assert "GEOMETRY: capability:gen_long_thinking 10752 x 54 cap 8192 thinking on" in out and "VERDICT: PASS stage=pre" in out
    assert "TESTHOST" not in d
    # a missing pin refuses by name before any row is emitted
    peer2 = _synthetic_peer_tree(tmp_path / "peer2", pin=None)
    with pytest.raises(SystemExit, match="d0p1it__constv2-mix-full__lora_r4_ce"):
        dry.main(["--stage", "pre", "--views", "gemma4e4b=it_d0p1", "--configs-root", str(peer2), "--tmp", str(tmp_path / "t2"), "--python", PY])
    # a view the peer's roster lacks refuses by name
    with pytest.raises(SystemExit, match="it_d0p5"):
        dry.main(["--stage", "pre", "--views", "gemma4e4b=it_d0p5", "--configs-root", str(peer), "--tmp", str(tmp_path / "t3"), "--python", PY])


def _peer_tree_with_a_declared_prompted_cell(root: Path, cell: str) -> Path:
    """A peer's checkout, SYNTHETIC: the roster in its SCHEDULED shape (every unscheduled prompted cell this tree's roster may
    already declare dropped first — a live roster may list the very cell — and the batch-64 lane's cells,
    which are matrix itdoc's) with ``cell`` declared (mapped to
    null — a prompted baseline, no object) on gemma4e4b's `it` and `main` views and nemotron3nano4b's `main` (the shape the
    live roster has taken since 2026-09-12), the real pins, and those three TAGs' campaign config / manifest pairs copied
    unchanged."""
    roster = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())
    for target in roster["targets"].values():
        for view_doc in (target.get("views") or {}).values():
            for c in [c for c in (view_doc.get("cells") or {}) if M.is_unscheduled_prompted(c)]:
                del view_doc["cells"][c]
            # the batch-64 lane's cells are matrix itdoc's: a live roster carries them, the
            # published shape never did — dropped like _without_batch64_lane sets their rows aside
            for c in [c for c, s in (view_doc.get("cells") or {}).items() if str(s or "").startswith(BATCH64_LANE_PREFIXES)]:
                del view_doc["cells"][c]
    for key, view in (("gemma4e4b", "it"), ("gemma4e4b", "main"), ("nemotron3nano4b", "main")):
        assert not any(M.is_unscheduled_prompted(c) for c, _ in V.cells_of(roster, key, view)), (key, view)
        roster["targets"][key]["views"][view].setdefault("cells", {})[cell] = None
    (root / "configs/capeval/manifests").mkdir(parents=True)
    (root / "configs/sdfbase_full22_roster.yaml").write_text(yaml.safe_dump(roster, sort_keys=False))
    shutil.copy(REPO / "configs/sdfbase_full22_pins.json", root / "configs/sdfbase_full22_pins.json")
    for tag in ("gemma4e4b_it", "gemma4e4b", "nemotron3nano4b"):
        for rel in (f"configs/sdfbase_full22_{tag}.yaml", f"configs/capeval/manifests/petri_full22_{tag}.yaml"):
            shutil.copy(REPO / rel, root / rel)
    return root


def test_a_roster_declared_prompted_cell_without_a_schedule_is_a_trailing_held_row_never_launched(tmp_path, capsys):
    """A `prompted__<si>` cell the roster declares on an expressed view that this matrix does
    NOT schedule (g8k_cells.SCHEDULED_PROMPTED_CELLS) is ADMITTED — a row, so the matrix agrees with the roster — and HELD: it
    trails every scheduled row (their indices and bytes never move), carries the unscheduled reason, is never a launch row,
    never planned, refused by --rows, printed STATUS=held, exits HELD from the sbatch with no argv; each TAG's overlay lists it
    LAST; a views.yaml hold naming it is moot. Two prompted cells on one view never share a job_id."""
    cell = "prompted__si_constitution"
    assert M.is_unscheduled_prompted(cell) and not M.is_unscheduled_prompted(M.CELL_PROMPTED) and not M.is_unscheduled_prompted("none")
    assert M.is_unscheduled_prompted("prompted__anything_else") and not M.is_unscheduled_prompted("constv2-mix-full__lora_r4_ce")
    # the job-id token is the cells table's (`prompted-const`, the six-arm Llama matrix's spelling — its launched rows carry it),
    # distinct from the scheduled baseline's `prompted`; spelled from the table, never typed
    short = M.cell_short(cell)
    assert short == M.cell_short(M.CELL_PROMPTED_CONST) != "prompted" and M.cell_short(M.CELL_PROMPTED) == "prompted" and M.budget_class(cell) == cell
    peer = _peer_tree_with_a_declared_prompted_cell(tmp_path / "peer", cell)
    roster = V.load_roster(V.roster_path(peer)); pins = V.load_pins(V.pins_path(peer))
    spec = V.load_views_yaml(REPO / V.VIEWS_REL)
    # cells_of lists it (the roster serves it), in roster order; select_rows routes it to the trailing held pass
    assert [c for c, s in V.cells_of(roster, "gemma4e4b", "it") if not str(s or "").startswith(BATCH64_LANE_PREFIXES)] == ["none", M.CELL_PROMPTED, "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", cell]   # the batch-64 lane's two cells are matrix itdoc's
    sched = [r for r in ROWS if r not in UNSCHED]                      # the 42 scheduled rows, whatever this tree's roster declares beyond them
    assert [r["index"] for r in sched] == list(range(len(sched))) and len(sched) == 42
    rows = _without_batch64_lane(V.select_rows(roster, pins, spec.views, holds=spec.holds))       # the batch-64 lane's cells are matrix itdoc's
    assert rows[:len(sched)] == sched                                   # the scheduled rows: today's, field for field, indices included
    extra = rows[len(sched):]
    assert [(r["index"], r["tag"], r["cell"], r["status"]) for r in extra] == [
        (len(sched), "gemma4e4b_it", cell, "held"), (len(sched) + 1, "gemma4e4b", cell, "held"), (len(sched) + 2, "nemotron3nano4b", cell, "held")]
    for r in extra:
        assert r["held_reason"] == M.unscheduled_held_reason(cell) and r["held_reason"].startswith(M.UNSCHEDULED_HELD_MARK) and "D28" in r["held_reason"]
        assert not r["fitted"] and r["route_kind"] == "prompted" and r["budget_class"] == cell and r["harness_cell"] == "prompted__constitution"
        assert r["job_id"] == f"g8k-{r['model_key']}-{r['view']}-{short}" and r["object_sha256"] is None and r["tuple_name"] is None
        # a long-SI prompted cell takes its model's FITTED class (the ~8k-token constitution prefill x 200), held or not
        assert r["out_root"] == f"controls/{r['tag']}/prompted--si_constitution/thinking_card" and r["time_limit"] == M.TIME_FITTED == M.time_limit_for(r["model_key"], False, cell)
        assert r["expected_stamps"] == list(C.g8k_expected_stamps(fitted=False, declared=False)) and (r["standup_cell"], r["standup_harness_cell"]) == (cell, "prompted__constitution")
    assert M.unscheduled_held_rows(rows) == extra and M.held_rows(rows) == M.held_rows(sched) + extra and M.unscheduled_held_rows(ROWS) == UNSCHED
    assert len({r["job_id"] for r in rows}) == len(rows) == len(sched) + 3 and len({r["out_root"] for r in rows}) == len(rows)
    # never launched: the launched set, the counts, the plan, --rows, the printed row
    assert M.launch_rows(rows) == LAUNCHED
    base, c = M.counts(sched), M.counts(rows)
    assert c["rows"] == len(sched) + 3 and c["held"] == base["held"] + 3 == 7 and c["by_tag"]["gemma4e4b_it"] == {"rows": base["by_tag"]["gemma4e4b_it"]["rows"] + 1, "held": base["by_tag"]["gemma4e4b_it"]["held"] + 1, "launched": 2}
    fixed = ("launched", "fitted_launched", "in_support", "declared", "servable", "contrasts_designed", "contrasts_computable", "tags", "tuple_names")
    assert {k: c[k] for k in fixed} == {k: base[k] for k in fixed} == {k: COUNTS[k] for k in fixed}
    assert [h["index"] for h in c["held_rows"]] == [h["index"] for h in base["held_rows"]] + [r["index"] for r in extra]
    pl = M.plan(SHA0, M.PRIMARY_CLUSTER, max_concurrent=12, universe=rows)
    idx = {r["index"] for r in extra}
    assert not idx & set(pl.rows) and idx <= set(pl.held) and not idx & set(_covered(pl.lines)) and pl.total_rows == len(rows)
    with pytest.raises(SystemExit, match="HELD"):
        M.plan(SHA0, M.PRIMARY_CLUSTER, max_concurrent=1, rows=str(extra[0]["index"]), universe=rows)
    with pytest.raises(SystemExit, match="HELD"):
        M.plan(SHA0, M.PRIMARY_CLUSTER, max_concurrent=1, cluster_share=str(extra[1]["index"]), universe=rows)
    sh = M.row_sh(extra[0])
    assert "STATUS=held" in sh and M.UNSCHEDULED_HELD_MARK in sh and f"CELL={cell}" in sh
    # a views.yaml hold naming the same cell is moot: the structural reason wins and nothing else moves
    moot = spec.holds + [{"key": "gemma4e4b", "view": "it", "cells": [cell], "reason": "a moot hold"}]
    assert _without_batch64_lane(V.select_rows(roster, pins, spec.views, holds=moot)) == rows
    # the production path over the peer tree through the generators and the sbatch's own dry-run path: HELD: lines, no argv
    dry = _load("_g8k_dry_run_d28", "scripts/sdfbase/gsm8k/g8k_dry_run.py")
    before = _tree_listing(peer)
    rc = dry.main(["--stage", "pre", "--views", "gemma4e4b=it,main; nemotron3nano4b=main", "--configs-root", str(peer), "--tmp", str(tmp_path / "t"), "--python", PY])
    out = capsys.readouterr().out
    assert rc == 0, out[-1500:]
    assert _tree_listing(peer) == before                                        # the peer's tree is never written
    tree = tmp_path / "t" / "tree"
    matrix = M.load_matrix(tree / "configs/sdfbase/gsm8k/matrix.json")
    n_sched = 4 + 9 + 8
    assert [(r["index"], r["tag"], r["status"]) for r in matrix["rows"] if r["cell"] == cell] == [(n_sched, "gemma4e4b_it", "held"), (n_sched + 1, "gemma4e4b", "held"), (n_sched + 2, "nemotron3nano4b", "held")]
    # (a `--views` spec carries no views.yaml holds, so those rows launch here — the peer path's standing behaviour; this hold is structural)
    assert matrix["counts"]["launched"] == n_sched and matrix["counts"]["held"] == 3 and matrix["counts"]["rows"] == n_sched + 3
    held_lines = [line for line in out.splitlines() if line.startswith("HELD:")]
    assert len(held_lines) == 3 and all(f"cell={cell}" in line and M.UNSCHEDULED_HELD_MARK in line for line in held_lines)
    assert sum(line.startswith("DRIVER:") for line in out.splitlines()) == n_sched and not any(f"--cell {cell} " in line for line in out.splitlines())
    assert "VERDICT: PASS stage=pre" in out and "REFUSED" not in out
    # each TAG's overlay lists the admitted cell LAST (a held row's overlay is in place for its later wave) and its manifest the harness id
    cfg = yaml.safe_load((tree / "configs/sdfbase/gsm8k/sdfbase_full22_gemma4e4b_it.gsm8k.yaml").read_text())
    assert cfg["explicit_cells"] == ["none", M.CELL_PROMPTED, "constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", cell]
    assert yaml.safe_load((tree / "configs/capeval/manifests/petri_full22_gemma4e4b_it.gsm8k.yaml").read_text())["petri"]["cells"][-1] == "prompted__constitution"
    assert "(1 held row(s) included" in (tree / "configs/sdfbase/gsm8k/sdfbase_full22_gemma4e4b_it.gsm8k.yaml").read_text()
    assert yaml.safe_load((tree / "configs/sdfbase/gsm8k/sdfbase_full22_nemotron3nano4b.gsm8k.yaml").read_text())["explicit_cells"][-1] == cell
    # the generator's own byte-identity gate holds over the peer tree too
    r = subprocess.run([sys.executable, str(G8K / "make_gsm8k_matrix.py"), "--views", "gemma4e4b=it,main; nemotron3nano4b=main", "--configs-root", str(tree),
                        "--out", str(tree / "configs/sdfbase/gsm8k/matrix.json"), "--check"], capture_output=True, text=True, cwd=REPO,
                       env={**os.environ, "PYTHONPATH": str(REPO / "src")}, check=False)
    assert r.returncode == 0 and f"{n_sched + 3} rows / 3 held / {n_sched} launched" in r.stdout, r.stdout + r.stderr


# --------------------------------------------------------------------------- #
# the ladder planner: sizing, the subset, idempotence — classify_store stubbed #
# --------------------------------------------------------------------------- #


def test_rung_sizing_formula_and_the_all_eligible_tail_case_pin_10_hours_and_a_20_hour_watcher():
    tail = LP.predict(200, 18, 12.0)
    assert tail["waves"] == 12 and tail["time_limit"] == "0-10:00:00" and tail["t_alloc_h"] == 10.0 and tail["watcher_max_hours"] == 20.0
    assert 4.8 < tail["t_pred_h"] < 4.95
    # the expectation (~20 items): ONE wave at S 41 (glm / nemotron) -> 16 + 22.8 + 4 = 42.8 min -> 1.5 h, watcher 3 h; the
    # same 20 at qwen's S 19 are TWO waves -> 65.5 min -> 2.5 h — the allocation follows the formula, never a typed class
    one = LP.predict(20, 41, 12.0)
    assert one["waves"] == 1 and one["time_limit"] == "0-01:30:00" and one["watcher_max_hours"] == 3.0
    two = LP.predict(20, 19, 12.0)
    assert two["waves"] == 2 and two["time_limit"] == "0-02:30:00" and two["watcher_max_hours"] == 5.0
    assert LP.predict(1, 41, 30.0)["time_limit"] == "0-01:00:00" and LP.predict(1, 41, 25.0)["time_limit"] == "0-01:30:00"   # 58 vs 62 min
    assert LP.slurm_time(36000) == "0-10:00:00" and LP.slurm_time(90000) == "1-01:00:00" and LP.round_up(1) == 1800
    for bad in ((0, 18, 12.0), (5, 0, 12.0), (5, 18, 0.0)):
        with pytest.raises(ValueError):
            LP.predict(*bad)
    # the observed rate: the median of completion / latency over forwarded rows; too few rows refuse
    rows = [{"rejected": False, "usage_completion_tokens": 1200, "latency_s": 100.0}] * 9 + [{"rejected": True, "usage_completion_tokens": None, "latency_s": 1.0}]
    with pytest.raises(ValueError, match="only 9 ledger rows"):
        LP.decode_rate_tok_s(rows)
    rows += [{"rejected": False, "usage_completion_tokens": 3000, "latency_s": 100.0}] * 3
    assert LP.decode_rate_tok_s(rows) == 12.0
    # the two-half contingency: disjoint halves, one draw per id; a duplicated id refuses
    halves = LP.split_subset([f"g{i}" for i in range(7)])
    assert sorted(halves[0] + halves[1]) == [f"g{i}" for i in range(7)] and not set(halves[0]) & set(halves[1])
    with pytest.raises(ValueError, match="duplicated"):
        LP.split_subset(["g1", "g2", "g1"])
    assert LP.GSM8K_TRIGGER_THRESHOLD == 0.01 and LP.GSM8K_REPORT_LINE == 0.01 and "0.01" in LP.GSM8K_TRIGGER_RULE and "16,384" in LP.GSM8K_TRIGGER_RULE
    from sisteer.capeval.tiers import inspect_rung as IR
    assert LP.GSM8K_TRIGGER_RULE is IR.GSM8K_TRIGGER_RULE and LP.GSM8K_TRIGGER_RULE.isascii()       # the producer's spelling (ASCII hyphens), by import


def _complete_store(wave_root: Path, row: dict, *, n_rows=200, rate=12.0):
    out = wave_root / row["out_root"]
    (out / "cells" / "k" / "outcomes").mkdir(parents=True)
    (out / "cells" / "k" / "outcomes" / "inspect__gsm8k_paired200.json").write_text("{}")
    (out / "RESULT").write_text("job=1 g8k rc=0 result=COMPLETE\n")
    with (out / "usage_ledger.jsonl").open("w") as f:
        for i in range(n_rows):
            f.write(json.dumps({"rejected": False, "usage_completion_tokens": int(rate * 50), "latency_s": 50.0, "attempt_id": "a1"}) + "\n")
    return out


def test_ladder_planner_writes_the_record_and_subset_sizes_the_rung_and_is_idempotent(tmp_path):
    wave_root = tmp_path / "wave"
    rows = [M.row_for("qwen35_4b", "main", "none"), M.row_for("qwen35_4b", "main", "constv2-mix-full__lora_r4_ce"),
            M.row_for("nemotron3nano4b", "main", "none"), M.held_rows()[0], M.row_for("glm46vflash", "main", "none")]
    for r in rows[:3]:
        _complete_store(wave_root, r)
    ids = [f"gsm8k-{i:03d}" for i in range(7)]
    # lane H's shapes: the record's `source` block carries identity_hash; the subset is the validated rung_subset/1 body
    def rec(out_root, **kw):
        return {"schema": "cap_ladder/1", "rule": LP.GSM8K_TRIGGER_RULE, "report_line": 0.01, "n": 200, "n_boundary": 0,
                "source": {"out_root": str(wave_root / out_root), "identity_hash": kw.pop("hash"), "task": LP.ROW_TASK, "is_rung": False}, **kw}

    def body(out_root, hash_, ids_, **kw):
        return {"schema": "rung_subset/1", "task": LP.ROW_TASK, "source_out_root": str(wave_root / out_root), "source_identity_hash": hash_,
                "source_cap": 8192, "cap": 16384, "rule": LP.GSM8K_TRIGGER_RULE, "threshold": LP.GSM8K_TRIGGER_THRESHOLD, "ids": list(ids_),
                "n_at_cap": kw.get("n_at_cap", 9), "n_non_terminating": kw.get("n_non_terminating", 2), "n_boundary": 0,
                "transport_check": {"ok": True, "binding_hop": "driver", "binding_s": 7200.0}}
    verdicts = {rows[0]["out_root"]: (rec(rows[0]["out_root"], hash="0000aaaa0000aaaa", exit="below_threshold", n_at_cap=4, n_non_terminating=0, share_at_cap=0.02), None),
                rows[1]["out_root"]: (rec(rows[1]["out_root"], hash="1111222233334444", exit="escalate", next_cap=16384, n_at_cap=9, n_non_terminating=2,
                                          transport={"ok": True, "binding_hop": "driver", "binding_s": 7200.0}), body(rows[1]["out_root"], "1111222233334444", ids)),
                rows[2]["out_root"]: (rec(rows[2]["out_root"], hash="5555666677778888", exit="escalate", next_cap=16384, n_at_cap=200, n_non_terminating=0,
                                          transport={"ok": True}), [f"gsm8k-{i:03d}" for i in range(200)])}
    calls = []

    def classify(out_root):
        calls.append(str(out_root))
        rel = str(Path(out_root).relative_to(wave_root))
        return verdicts[rel]
    s_of = {"qwen35_4b": 39, "nemotron3nano4b": 41, "glm46vflash": 41}
    planned, lines = LP.plan_all(rows, wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=classify, s_rung_of=lambda k, c: s_of[k])
    # the held row is never classified; the pending store is not; the ONE writer wrote cap_ladder.json with this instrument's rule
    assert len(planned) == 4 and len(calls) == 3 and all(rows[3]["out_root"] not in c for c in calls)
    assert [p["status"] for p in planned] == ["complete", "complete", "complete", "pending"] and planned[3]["exit"] is None
    written = json.loads((wave_root / rows[0]["out_root"] / "cap_ladder.json").read_text())
    assert written == verdicts[rows[0]["out_root"]][0]                        # the adapter's record VERBATIM (nothing added)
    assert written["exit"] == "below_threshold" and written["rule"] == LP.GSM8K_TRIGGER_RULE and written["report_line"] == 0.01
    assert planned[0]["sizing"] is None and "no rung" in planned[0]["note"]
    # ONE WRITER, walked over the wave root: exactly the classified stores carry cap_ladder.json, nothing else does
    assert sorted(p.parent for p in wave_root.rglob("cap_ladder.json")) == sorted(wave_root / r["out_root"] for r in rows[:3])
    # the escalating cell: the subset bound to the source, the sizing from the actual subset and the observed rate, the rung state
    sub_path = wave_root / "rung16384" / rows[1]["tag"] / rows[1]["cell_slug"] / "rung_subset.json"
    sub = json.loads(sub_path.read_text())
    assert sub == verdicts[rows[1]["out_root"]][1]                              # the adapter's rung_subset/1 body VERBATIM
    assert set(sub) == set(LP.ITEM_SUBSET_FIELDS) and sub["schema"] == "rung_subset/1" == getattr(C, "ITEM_SUBSET_SCHEMA", "rung_subset/1")
    assert LP.ITEM_SUBSET_FIELDS == getattr(C, "ITEM_SUBSET_FIELDS", LP.ITEM_SUBSET_FIELDS)
    assert sub["ids"] == ids and sub["cap"] == 16384 and sub["source_cap"] == 8192 and sub["source_identity_hash"] == "1111222233334444"
    # a bare id sequence from the adapter: the same body, EXACTLY the contract's fields, the hash from the record's source block
    nsub = json.loads((wave_root / "rung16384" / rows[2]["tag"] / rows[2]["cell_slug"] / "rung_subset.json").read_text())
    assert set(nsub) == set(LP.ITEM_SUBSET_FIELDS) and nsub["source_identity_hash"] == "5555666677778888" and len(nsub["ids"]) == 200 and nsub["n_at_cap"] == 200
    assert nsub["task"] == sub["task"] == C.GSM8K_STANDARD_TASK and C.validate_item_subset(nsub) == nsub     # B-P1.2: the ROW task, the contract's validator
    sz = planned[1]["sizing"]
    assert sz["s_rung"] == 39 and sz["rate_tok_s"] == 12.0 and sz["waves"] == 1 and sz["time_limit"] == "0-01:30:00" and sz["watcher_max_hours"] == 3.0
    state = json.loads((wave_root / rows[1]["rung_out_root"] / "ladder_state.json").read_text())
    assert state["rung"]["state"] == "planned" and state["rung"]["subset_sha16"] == LP.subset_sha16(ids) and state["rung"]["n_ids"] == 7 and state["attempts"] == []
    assert LP.subset_sha16 is C.item_subset_sha16 and state["rung"]["subset_sha16"] == C.item_subset_sha16(ids)   # the harness's digest, by import
    # the planner records the pin the rung runs at AND the tree it classified from (equal by default), with the rule / threshold it used
    assert state["rung"]["planned_by"] == {"sha": SHA0, "script": "scripts/sdfbase/gsm8k/ladder_plan.py", "planner_tree_sha": SHA0,
                                            "rule": LP.GSM8K_TRIGGER_RULE, "threshold": LP.GSM8K_TRIGGER_THRESHOLD}
    other_tree = "d" * 40
    _p, lines_other = LP.plan_all(rows, wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=classify, s_rung_of=lambda k, c: s_of[k],
                                  planner_tree_sha=other_tree)
    state_other = json.loads((wave_root / rows[1]["rung_out_root"] / "ladder_state.json").read_text())
    assert state_other["rung"]["planned_by"]["planner_tree_sha"] == other_tree and state_other["rung"]["planned_by"]["sha"] == SHA0
    assert lines_other == lines and all(other_tree not in line for line in lines_other)        # the lines name the PIN, never the planner tree
    # the all-200 Nemotron rung: 5 waves at S 41 -> its own allocation; primary-only placement in the lines
    nz = planned[2]["sizing"]                                                  # 5 waves x 22.8 + 20 min = 133.8 -> x2 -> 4.5 h
    assert nz["waves"] == 5 and nz["n_eligible"] == 200 and nz["time_limit"] == "0-04:30:00" and nz["watcher_max_hours"] == 9.0
    assert lines[0].startswith("# FIRST RUNG") and rows[1]["job_id"] in lines[0] and "watcher --max-hours 3" in lines[0]
    assert lines[1] == (f"sbatch -J g8k-rung-first --qos={SITE_A['qos']} --time=0-01:30:00 --output={SITE_A['logs']}/%x_%A_%a.out "
                        f"--array={rows[1]['index']}%1 {M.CHECKOUT_ROOT}/{SHA0}/scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch ARRAY {SHA0} 16384")
    assert any(f"--array={rows[2]['index']}%1 " in line and "--time=0-04:30:00" in line and line.endswith(f"ARRAY {SHA0} 16384") for line in lines)
    # on the overflow cluster the Nemotron rung is NOT printed as a line; the qwen rung is, at that cluster's QoS
    _, eur = LP.plan_all(rows, wave_root=wave_root, sha=SHA0, cluster="cluster_b", classify=classify, s_rung_of=lambda k, c: s_of[k])
    assert all(f"--array={rows[2]['index']}%1" not in line for line in eur) and any("cluster_constraint primary" in line for line in eur)
    assert any(f"--qos={SITE_B['qos']}" in line and f"--array={rows[1]['index']}%1" in line for line in eur)
    # IDEMPOTENCE: a second invocation re-prints the same plan, writes no second subset file, changes nothing
    listing = sorted(str(p) for p in wave_root.rglob("*") if p.is_file())
    _planned2, lines2 = LP.plan_all(rows, wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=classify, s_rung_of=lambda k, c: s_of[k])
    assert lines2 == lines and sorted(str(p) for p in wave_root.rglob("*") if p.is_file()) == listing
    # a RUNNING rung is never re-planned (the wrapper's attempt in its ladder_state.json)
    st_path = wave_root / rows[1]["rung_out_root"] / "ladder_state.json"
    st = json.loads(st_path.read_text()); st["attempts"] = [{"attempt_id": "a1", "started": "x"}]; st["rung"]["state"] = "running"; st_path.write_text(json.dumps(st))
    planned3, lines3 = LP.plan_all(rows, wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=classify, s_rung_of=lambda k, c: s_of[k])
    assert planned3[1]["sizing"] is None and "not re-planned" in planned3[1]["note"] and all(f"--array={rows[1]['index']}%1" not in line for line in lines3)
    # a DIFFERING subset on a planned rung refuses by name
    st["attempts"] = []; st["rung"]["state"] = "planned"; st_path.write_text(json.dumps(st))
    verdicts[rows[1]["out_root"]] = (verdicts[rows[1]["out_root"]][0], body(rows[1]["out_root"], "1111222233334444", ids[:-1]))
    with pytest.raises(SystemExit, match="already planned over subset"):
        LP.plan_all(rows, wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=classify, s_rung_of=lambda k, c: s_of[k])
    verdicts[rows[1]["out_root"]] = (verdicts[rows[1]["out_root"]][0], body(rows[1]["out_root"], "1111222233334444", ids))
    # a subset body bound to ANOTHER store refuses; a body missing a contract field refuses; another rule in the record refuses
    shutil.rmtree(wave_root / rows[2]["rung_out_root"], ignore_errors=True); shutil.rmtree(wave_root / "rung16384" / rows[2]["tag"], ignore_errors=True)
    verdicts[rows[2]["out_root"]] = (verdicts[rows[2]["out_root"]][0], body(rows[1]["out_root"], "5555666677778888", ids))
    with pytest.raises(SystemExit, match="binds to"):
        LP.plan_all(rows[2:3], wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=classify, s_rung_of=lambda k, c: s_of[k])
    short = body(rows[2]["out_root"], "5555666677778888", ids); short.pop("transport_check")
    verdicts[rows[2]["out_root"]] = (verdicts[rows[2]["out_root"]][0], short)
    with pytest.raises(SystemExit, match="missing \\['transport_check'\\]"):
        LP.plan_all(rows[2:3], wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=classify, s_rung_of=lambda k, c: s_of[k])
    other_rule = dict(verdicts[rows[2]["out_root"]][0]); other_rule["rule"] = "trigger iff share_truncated > 0.01"
    verdicts[rows[2]["out_root"]] = (other_rule, [f"gsm8k-{i:03d}" for i in range(200)])
    with pytest.raises(SystemExit, match="another rule"):
        LP.plan_all(rows[2:3], wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=classify, s_rung_of=lambda k, c: s_of[k])
    # an unbound rung (no source identity hash anywhere) refuses; an inconsistent adapter answer refuses; --max-alloc-h refuses by name
    verdicts[rows[2]["out_root"]] = ({"exit": "escalate", "next_cap": 16384}, [f"gsm8k-{i:03d}" for i in range(200)])
    shutil.rmtree(wave_root / rows[2]["rung_out_root"], ignore_errors=True); shutil.rmtree(wave_root / "rung16384" / rows[2]["tag"], ignore_errors=True)
    with pytest.raises(SystemExit, match="source identity"):
        LP.plan_all(rows[2:3], wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=classify, s_rung_of=lambda k, c: s_of[k])
    verdicts[rows[2]["out_root"]] = ({"exit": "below_threshold"}, ["gsm8k-000"])
    with pytest.raises(SystemExit, match="AND a subset"):
        LP.plan_all(rows[2:3], wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=classify, s_rung_of=lambda k, c: s_of[k])
    verdicts[rows[2]["out_root"]] = ({"exit": "escalate", "source": {"identity_hash": "5555666677778888"}, "n_at_cap": 200, "n_non_terminating": 0,
                                      "transport": {"ok": True}}, [f"gsm8k-{i:03d}" for i in range(200)])
    with pytest.raises(SystemExit, match="exceeds --max-alloc-h"):
        LP.plan_all(rows[2:3], wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=classify, s_rung_of=lambda k, c: 18, max_alloc_h=6.0)
    # a store whose two records disagree refuses; the CLI refuses without lane H's adapter module by name
    (wave_root / rows[0]["out_root"] / "RESULT").write_text("result=DRIVER_FAILED\n")
    with pytest.raises(SystemExit, match="two records disagree"):
        LP.store_status(wave_root / rows[0]["out_root"])
    # one writer: no other CODE file under the launcher names cap_ladder.json (comments, the README and g8k_cells.py's constant aside)
    # — with ONE DECLARED READER: degeneracy_union_column.py
    # pairs the campaign's degeneracy union classifier against the FILED per-item classification (a NON-deciding column —
    # "pair the union against what the ladder filed"), so it reads cap_ladder.json by name and its prose says so honestly. It is never a
    # writer, and the excuse is CONDITIONAL on a positive check of that: around every spelling of the record's name (the literal or
    # cap_ladder's RECORD_NAME constant) no write call may appear on that line or the two lines either side. Every other file keeps the
    # rule as it was; a tmp twin of the reader with ONE added json.dump(..., open(<record>, "w")) line is refused by the same check.
    DECLARED_READER = "degeneracy_union_column.py"
    RECORD_SPELLINGS = ("cap_ladder.json", "RECORD_NAME", "CAP_LADDER_FILENAME")
    WRITE_CALLS = ("write_text(", "write_bytes(", "json.dump(", "atomic_write_json(", ".rename(", ".replace(", "shutil.copy", "shutil.move")
    WRITE_OPEN = re.compile(r"""open\([^)]*["'][wax]""")       # open(..., "w" | "a" | "x"): a read-mode open is not a write

    def record_write_lines(code: str) -> list[int]:
        """The 1-based lines of ``code`` that spell the record's name and have a write call within two lines either side."""
        lines = code.splitlines()
        hits: list[int] = []
        for i, line in enumerate(lines):
            if any(s in line for s in RECORD_SPELLINGS):
                window = lines[max(0, i - 2): i + 3]
                if any(any(w in x for w in WRITE_CALLS) or WRITE_OPEN.search(x) for x in window):
                    hits.append(i + 1)
        return hits

    def guard_code(f: Path) -> str:
        return "\n".join(line for line in f.read_text().splitlines() if not line.lstrip().startswith("#") and "CAP_LADDER_FILENAME = " not in line)

    for f in G8K.glob("*"):
        if f.name != "ladder_plan.py" and f.is_file() and f.suffix in (".py", ".sh", ".sbatch"):
            code = "\n".join(line for line in f.read_text().splitlines() if not line.lstrip().startswith("#") and "CAP_LADDER_FILENAME = " not in line)
            if f.name == DECLARED_READER:
                # the excuse is for a READER by name — the file must actually read the record, and never write it
                assert "cap_ladder.json" in code and "RECORD_NAME" in code and "read_json(p)" in code, f"{f.name}: the declared reader no longer reads the record by name"
                assert record_write_lines(code) == [], f"{f.name}: a write call within two lines of the record's name at {record_write_lines(code)} — the declared reader must never write it"
                continue
            assert "cap_ladder.json" not in code and "CAP_LADDER_FILENAME" not in code.replace("G8K_CAP_LADDER", ""), f.name
    # the KNOWN-BAD twin: the declared reader with ONE added write of the record (json.dump into an open(..., "w") of RECORD_NAME) is
    # refused by the same check; the honest tool passes it
    reader_src = (G8K / DECLARED_READER).read_text()
    assert "    rec = read_json(p)\n" in reader_src
    twin = reader_src.replace("    rec = read_json(p)\n", "    rec = read_json(p)\n    json.dump(rec, open(out_root / CL.RECORD_NAME, \"w\"))\n", 1)
    (tmp_path / "twin_writer.py").write_text(twin)
    assert record_write_lines(guard_code(G8K / DECLARED_READER)) == []
    assert record_write_lines(guard_code(tmp_path / "twin_writer.py")) != []
    # and a twin whose write_text sits ONE line below the spelling (no spelling on its own line) is refused too: the window, not just the
    # same line. The check is a proximity rule on the spelling, not a data-flow analysis — a write through a variable bound far from
    # the spelling is outside its reach; the reader's only writes today are its own output file and the markdown summary.
    assert "    p = out_root / CL.RECORD_NAME\n    if not p.is_file():\n" in reader_src
    twin2 = reader_src.replace("    p = out_root / CL.RECORD_NAME\n    if not p.is_file():\n",
                               "    p = out_root / CL.RECORD_NAME\n    p.write_text(\"{}\")\n    if not p.is_file():\n", 1)
    (tmp_path / "twin_writer2.py").write_text(twin2)
    assert record_write_lines(guard_code(tmp_path / "twin_writer2.py")) != []


# --------------------------------------------------------------------------- #
# the integration fix round                                                   #
# --------------------------------------------------------------------------- #


def _argv_values(line, flag):
    """Every value of a repeatable flag in a printed `LABEL: %q %q …` argv line, through shlex (the %q quoting is shell)."""
    import shlex
    argv = shlex.split(line.split(":", 1)[1])
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == flag]


def _check_endpoint(serve: Path, table: Path, hcell: str, expected_positions: str):
    """g8k_check_endpoint (-> the core's st_check_endpoint) over a REAL exported table, the stamp looked up under `hcell`."""
    script = (_g8k_funcs() + f"\nCPY={sys.executable}; REPO_DIR={REPO}; SERVE_DIR={serve.parent}; G8K_SERVE_SUBDIR=gsm8k; TABLE={table}; "
              f"G8K_UPSTREAM_TIMEOUT_S=7200; BUDGET_TEMPLATE_HASH=feedfacefeedface; G8K_STANDUP_HARNESS_CELL={hcell}; "
              f"G8K_SERVED_POSITIONS_EXPECTED={expected_positions}\ng8k_check_endpoint && echo ENDPOINT_OK")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)


def _stage_view_bank(root: Path, key: str, view: str = "main") -> Path:
    """EVERY fitted cell of the (key, view) TAG staged as a view-shaped bank under its REAL lineage, with the two scale-12
    safety objects at `positions: response` — what makes them HEADER-LOCATED (declared) in the real bank."""
    import torch
    from test_sdfbase_full22_wave import _stage_any, arm_of, object_artifact_name, ra, sha256_file
    bank = root / key / "cells_view" / view
    bank.mkdir(parents=True, exist_ok=True)
    for r in ROWS:
        if r["model_key"] != key or r["view"] != view or not r["fitted"]:
            continue
        if r["cell"] in M.HEADER_LOCATED_CELLS:
            arm = arm_of(r["cell"])
            d = bank / r["cell"]
            d.mkdir(parents=True, exist_ok=True)
            blob = d / object_artifact_name(arm)
            torch.save({"kind": ra.ARM_OBJECT_KINDS[arm], "positions": "response", "response_header_len": 7, "layers": [0],
                        "w": torch.zeros(4)}, blob)
            (d / "train_budget_meta.json").write_text(json.dumps({"cell_id": r["cell"], "completion_tokens": 512, "object_sha256": sha256_file(blob)}))
            (d / "staged_from.json").write_text(json.dumps({"source": "synthetic"}))
        else:
            _stage_any(bank, r["cell"], r["cell"], target=key, legacy_sidecar=(arm_of(r["cell"]) == "lora_r4"))
    return bank


def test_declared_rows_argv_passes_the_standups_real_digest_parser_and_the_fresh_tables_real_positions_lookup(tmp_path, s_by, monkeypatch, capsys):
    """B-P1.1: the declared spelling crosses the standup boundary ONCE. The produced STANDUP argv is fed to the
    standup's REAL cell selection + digest parser (mt_serving_up.select_cells, capctrl_serving_up.parse_object_digest_args /
    expected_artifact_digests_for_cells) and the row's stamp is read from a REALLY exported routing table (the standup's own
    export_routing_table over a view-shaped bank) under the key the standup exports — the variant's harness id. The plain
    spellings are refused / absent at both doors: exactly the shape the bug had (the digest keyed on the plain cell, the
    positions looked up under the plain harness cell, the dry run exporting the plain cell and masking it)."""
    import capctrl_serving_up as cu
    from sisteer.capeval.petri import registry_adapter as ra
    mt = DRY._mt_serving_up()
    declared = [r for r in LAUNCHED if r["declared"]]
    assert len(declared) == 4 and {r["model_key"] for r in declared} == {"gemma4e4b", "qwen35_4b"}
    bank_root = tmp_path / "bank"
    for key in ("gemma4e4b", "qwen35_4b"):
        _stage_view_bank(bank_root, key)
    (tmp_path / "t").mkdir()                                                            # the dry run's --tmp (its main creates it)
    for row in declared:
        text, rc, _ = _dry(row, s=s_by[(row["model_key"], M.CAP)], shared=tmp_path / "shared")
        assert rc == 0
        standup = _line(text, "STANDUP")
        cells = _argv_values(standup, "--cell")
        digests = _argv_values(standup, "--expect-object-sha256")
        assert cells == [row["standup_cell"]] == [row["cell"] + C.OOS_STANDUP_CELL_SUFFIX] and len(digests) == 1
        cfg = yaml.safe_load((REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_{row['tag']}.gsm8k.yaml").read_text())
        tt_cells = mt.select_cells(cfg["explicit_cells"], cells)                       # the standup's own selection
        assert tt_cells == ["none", row["standup_cell"]]
        parsed = cu.parse_object_digest_args(digests, tt_cells)                        # the standup's REAL digest parser
        assert parsed == {row["standup_cell"]: row["object_sha256"]}
        by_harness = cu.expected_artifact_digests_for_cells(tt_cells, [ra.harness_cell_id(c) for c in tt_cells], parsed, required=False)
        assert by_harness == {row["standup_harness_cell"]: row["object_sha256"]}
        with pytest.raises(ValueError, match="not one of this standup's --cell values"):      # the bug: the PLAIN key
            cu.parse_object_digest_args([f"{row['cell']}={row['object_sha256']}"], tt_cells)
        # the fresh table, exported by the standup's own code under the standup spelling: the key is the variant's harness id
        st = DRY.fresh_table_stamp(row, bank_root, tmp_path / "t", REPO)
        assert st["exported"], st
        assert st["stamp_key"] == row["standup_harness_cell"] and st["own_route"] and st["stamp_for_cell"] == ["response"]
        assert row["standup_harness_cell"] in st["lens_served_positions"] and row["harness_cell"] not in st["lens_served_positions"]
        assert row["harness_cell"] not in st["routes"]                                  # the plain cell is not in that table
        # the wrapper's post-standup check (g8k_check_endpoint -> st_check_endpoint's stamp read) over that REAL table
        table = tmp_path / "t" / f"row{row['index']}_routing_table.json"
        assert table.is_file()
        serve = tmp_path / f"serve{row['index']}" / "gsm8k"
        serve.mkdir(parents=True)
        (serve / "endpoint.json").write_text(json.dumps({"proxy_upstream_timeout_s": 7200.0, "chat_template_hash": "feedfacefeedface",
                                                          "lens_served_positions_passthrough": "passed"}))

        ok = _check_endpoint(serve, table, row["standup_harness_cell"], row["served_positions_expected"])
        assert ok.returncode == 0 and "ENDPOINT_OK" in ok.stdout and f"'{row['standup_harness_cell']}': ['response']" in ok.stdout, ok.stdout + ok.stderr
        bad = _check_endpoint(serve, table, row["harness_cell"], row["served_positions_expected"])   # the bug's lookup, now refused by construction
        assert bad.returncode == 1 and "served-positions stamp absent or wrong" in bad.stdout and "ENDPOINT_OK" not in bad.stdout
    # the two-stage dry run at --stage prod over the same bank: the gemma declared row PASSES on the declared spelling (the
    # live acceptance check, `safety_refusal__psr_kl_all@thinking`, in its grading-host form) and is no expected refusal
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", _facts("certified"), raising=False)
    monkeypatch.setattr(C, "check_render_fact_certified", lambda family: None, raising=False)
    dry = _load("_g8k_dry_run_declared", "scripts/sdfbase/gsm8k/g8k_dry_run.py")
    row = M.row_for("gemma4e4b", "main", "safety_refusal__psr_kl_all")
    rc = dry.main(["--stage", "prod", "--only", "gemma4e4b", "main", "safety_refusal__psr_kl_all", "--tmp", str(tmp_path / "t2"), "--python", PY,
                   "--bank-root", str(bank_root)])
    out = capsys.readouterr().out
    assert rc == 0, out[-1500:]
    assert f'SERVED_POSITIONS: row={row["index"]} lens_served_positions[{row["standup_harness_cell"]}] == "response" (expected)' in out
    assert "EXPECTED_REFUSAL:" not in out and "VERDICT: PASS stage=prod" in out
    assert f"--cell {row['standup_cell']} " in _line(out, "STANDUP") + " " and f"--expect-object-sha256 {row['standup_cell']}={row['object_sha256']}" in out


def _producer_store(wave_root: Path, row: dict, *, at_cap: dict, identity_hash: str = "0123456789abcdef", upstream=7200.0, n_ledger=200) -> Path:
    """A store built THROUGH THE PRODUCER's own writers and field names (contracts.*, completion.write_task_outcomes): the one
    cell dir with the version-2 manifest facts the adapter reads, the all-attempted outcome record over the paired-200 ids,
    the wrapper's RESULT line and the 8,192 usage ledger the planner sizes from. Synthetic ids and texts."""
    from sisteer.capeval.tiers import inspect_rung as IR
    from sisteer.capeval.tinycap import completion
    out = wave_root / row["out_root"]
    cell = out / "cells" / "cell__controls__t0__a0__synthetic"
    cell.mkdir(parents=True)
    task = C.GSM8K_STANDARD_TASK
    (cell / "manifest.json").write_text(json.dumps({
        "harness_config_hash": identity_hash,
        "resolved_config": {C.MANIFEST_KEY_TINYCAP: {C.IDENTITY_KEY_IDENTITY_VERSION: 2, C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: {task: M.CAP}}},
        C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S: upstream}))
    ids = [f"gsm8k_{i:08x}" for i in range(M.N_ITEMS)]
    items = [{"item_id": iid, "sample_idx": 0, "attempt": 1, "scored": True, "value": 1.0, "unparseable": None,
              "stop_reason": (IR.STOP_REASON_AT_CAP if iid in at_cap else "stop"), "at_cap": iid in at_cap,
              "think_block_present": True, "think_block_unterminated": False, "stray_open_tag": False,
              "content_empty_with_reasoning": False, "prompt_tokens": 2000, "completion_tokens": int(at_cap.get(iid, 100 + i)), "seed": i}
             for i, iid in enumerate(ids)]
    n = len(ids)
    completion.write_task_outcomes(cell, task, {
        "task": task, "item_set": C.GSM8K_ITEM_SET_PAIRED, "thinking": C.THINKING_ON, "think_split_mode": "implicit_open",
        "unterminated_policy": "unscored", "max_tokens": M.CAP, "samples_per_item": 1, "n_attempted_items": n, "n_attempted_samples": n,
        "n_scored": n, "n_unscored": 0, "n_scored_samples": n, "n_unscored_samples": 0, "n_stop_reason": n, "n_at_cap": len(at_cap),
        "n_think_block_present": n, "n_think_block_unterminated": 0, "n_unterminated_not_at_cap": 0, "n_completion_tokens": n,
        "producer_tree_sha": None, "items": items})
    (out / "RESULT").write_text("job=1 g8k rc=0 result=COMPLETE\n")
    with (out / M.LEDGER_FILENAME).open("w") as f:
        for _ in range(n_ledger):
            f.write(json.dumps({"rejected": False, "usage_completion_tokens": 600, "latency_s": 50.0, "attempt_id": "a1"}) + "\n")
    return out


def test_the_planner_and_the_wrappers_rung_check_consume_the_real_classifiers_subset_in_the_row_task_namespace(tmp_path):
    """B-P1.2: the classifier's subset carries the ROW task `inspect/gsm8k_paired200` (contracts.GSM8K_STANDARD_TASK);
    the planner and the wrapper compared against the unprefixed CLI spelling and would have aborted / refused every rung. Here
    the REAL `inspect_rung.classify_store` runs over a store built through the producer, its subset goes through the planner
    (written verbatim, sized, the harness's digest in the rung state) and the written file through the wrapper's own
    `g8k_rung_check`; a subset in the unprefixed namespace is refused by both."""
    from sisteer.capeval.tiers import inspect_rung as IR
    wave_root = tmp_path / "wave"
    row = M.row_for("qwen35_4b", "main", "constv2-mix-full__lora_r4_ce")
    ids = [f"gsm8k_{i:08x}" for i in range(M.N_ITEMS)]
    at_cap = {iid: M.CAP for iid in ids[:9]}                                            # 4.5 % > the trigger
    loops = set(ids[:2])
    trunc = " ".join(f"step{i} compute value{i} then carry{i}" for i in range(300))
    texts = {(iid, 0): (("the answer is " * 400) if iid in loops else trunc) for iid in ids[:9]}
    out = _producer_store(wave_root, row, at_cap=at_cap)
    record, subset = IR.classify_store(out, texts=texts)                                # the REAL producer, no stub
    assert record["exit"] == "escalate" and subset is not None
    assert subset["task"] == C.GSM8K_STANDARD_TASK == "inspect/gsm8k_paired200" and subset["task"] != M.INSPECT_TASK
    assert sorted(subset["ids"]) == sorted(ids[2:9]) and set(subset) == set(C.ITEM_SUBSET_FIELDS) and subset["rule"] == LP.GSM8K_TRIGGER_RULE
    planned, lines = LP.plan_all([row], wave_root=wave_root, sha=SHA0, cluster="cluster_a",
                                 classify=lambda p: IR.classify_store(p, texts=texts), s_rung_of=lambda k, c: 19)
    assert planned[0]["exit"] == "escalate" and planned[0]["n_ids"] == 7 and planned[0]["sizing"]["s_rung"] == 19, planned
    sub_path = wave_root / M.RUNG_DIR_REL / row["tag"] / row["cell_slug"] / M.RUNG_SUBSET_FILENAME
    written = json.loads(sub_path.read_text())
    assert written == subset and written["task"] == C.GSM8K_STANDARD_TASK             # VERBATIM, in the producer's namespace
    assert C.read_item_subset(sub_path) == written                                      # the driver's own reader admits it
    state = json.loads((wave_root / row["rung_out_root"] / M.LADDER_STATE_FILENAME).read_text())
    assert state["rung"]["subset_sha16"] == C.item_subset_sha16(subset["ids"]) and len(state["rung"]["subset_sha16"]) == 16
    assert json.loads((out / M.CAP_LADDER_FILENAME).read_text()) == record               # the adapter's record verbatim
    assert any(f"--array={row['index']}%1 " in line and line.endswith(f"ARRAY {SHA0} 16384") for line in lines)
    # the wrapper's own check over the written file, with the row's literals from g8k_cells.py (G8K_ROW_TASK), passes …
    def rung_check(subset_file):
        script = (_g8k_funcs() + f"\nCPY={sys.executable}; REPO_DIR={REPO}\n" + M.literals_sh()
                  + f"\nRUNG={M.RUNG_CAP}; G8K_RUNG_CAP={M.RUNG_CAP}; G8K_CAP={M.CAP}; RUNG_SUBSET={subset_file}; SOURCE_OUT_ROOT={out}\n"
                  "g8k_rung_check && echo RUNG_CHECK_OK")
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    ok = rung_check(sub_path)
    assert ok.returncode == 0 and "RUNG_CHECK_OK" in ok.stdout and "rung subset OK: 7 ids" in ok.stdout, ok.stdout + ok.stderr
    # … and the UNPREFIXED namespace (what the wrapper and the planner compared against before the fix) is refused by name
    bad = dict(written); bad["task"] = M.INSPECT_TASK
    bad_path = tmp_path / "bad_subset.json"; bad_path.write_text(json.dumps(bad))
    r = rung_check(bad_path)
    assert r.returncode == 1 and "RUNG_CHECK_OK" not in r.stdout and "task 'gsm8k_paired200' != 'inspect/gsm8k_paired200'" in r.stdout, r.stdout
    shutil.rmtree(wave_root / row["rung_out_root"]); sub_path.unlink()
    with pytest.raises(SystemExit, match="binds to .* task 'gsm8k_paired200'"):
        LP.plan_all([row], wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=lambda p: (record, bad), s_rung_of=lambda k, c: 19)
    # the planner never re-types the producer's constants: by import, identity-checked
    assert LP.GSM8K_TRIGGER_RULE is IR.GSM8K_TRIGGER_RULE and LP.GSM8K_TRIGGER_THRESHOLD is IR.GSM8K_TRIGGER_THRESHOLD
    assert LP.subset_sha16 is C.item_subset_sha16 and LP.ITEM_SUBSET_FIELDS is C.ITEM_SUBSET_FIELDS
    src = (G8K / "ladder_plan.py").read_text()
    assert "hashlib" not in src and '"trigger iff' not in src and '"rung_subset/1"' not in src


def test_the_frozen_matrix_passes_the_analysis_lanes_real_reader_and_carries_no_expected_refusal(tmp_path):
    """The frozen matrix's stamp triples are the CONTRACT's — read back through the analysis'
    REAL reader (`tinycap_gsm8k_validators.load_matrix`, which raised at row 9 on the frozen literal that was not a row of the
    contract) — and no row is an expected refusal: gemma's two header-located `main` cells are servable."""
    mx = VAL.load_matrix(REPO / "configs/sdfbase/gsm8k/matrix.json")                    # raises AnalysisError on a bad row
    # the analysis' reader admits the WHOLE document — the frozen 42 and the appended waves (their external-row rules are pinned
    # in the `ext` / `steer` section); the published wave's numbers are 38 / 4 / 0 / 4 + the trailing held rows
    assert mx["counts"]["launched"] == len(M.launch_rows(ALL_ROWS)) == 38 + len(EXT_ROWS) + len(STEER_ROWS) and mx["counts"]["declared"] == 4
    assert mx["counts"]["expected_refusals"] == 0 and mx["counts"]["held"] == 4 + len(UNSCHED)
    assert VAL.load_matrix([dict(r) for r in ROWS])["counts"]["launched"] == 38
    assert [r["index"] for r in mx["rows"]] == list(range(len(ALL_ROWS)))
    for got, mine in zip(mx["rows"], ALL_ROWS):
        assert got["expected_stamps"] == mine["expected_stamps"] == list(VAL.expected_stamps_for(fitted=mine["fitted"], declared=mine["declared"]))
        assert list(C.g8k_expected_stamps(fitted=mine["fitted"], declared=mine["declared"])) == mine["expected_stamps"]
    assert (VAL.STAMPS_BASE, VAL.STAMPS_FITTED, VAL.STAMPS_DECLARED) == (C.G8K_STAMPS_BASE, C.G8K_STAMPS_FITTED, C.G8K_STAMPS_DECLARED)
    assert C.G8K_STAMPS_DECLARED == ("request-unique", C.DECLARED_SALT_POLICY_LABEL, "thinking_on")
    # the regression: the OLD launcher literal on a declared row is refused by the reader, by name
    rows = json.loads(json.dumps(ROWS))
    rows[9]["expected_stamps"] = ["declared-request-unique", "always", "thinking_on"]
    with pytest.raises(VAL.AnalysisError, match="matrix row 9: expected_stamps"):
        VAL.load_matrix(rows)
    src = (G8K / "g8k_cells.py").read_text() + (G8K / "g8k_views.py").read_text()
    assert "declared-request-unique" not in src and "STAMPS_DECLARED" not in src and "g8k_expected_stamps" in src
    # the launcher files no expected refusal; the contract's refusal is withdrawn (always None); the family tuple still pins
    assert M.expected_refusal("gemma4e4b", True) is None and M.expected_refusal("qwen35_4b", True) is None
    assert C.header_invariant_declaration_refusal("gemma4", header_located=True, arm=C.THINKING_ON) is None
    assert M.HEADER_INVARIANT_THINKING_FAMILIES == C.HEADER_INVARIANT_THINKING_FAMILIES == ("gemma4",)
    with pytest.raises(SystemExit, match="not one of"):
        M.expected_refusal("mistral7b", True)                 # an unregistered key (llama8b is the registry's fifth, P4)
    assert M.expected_refusal("llama8b", False) is None
    gemma = [r for r in LAUNCHED if r["model_key"] == "gemma4e4b" and r["declared"]]
    assert [r["cell"] for r in gemma] == list(M.HEADER_LOCATED_CELLS) and all(r["status"] == "launch" and not r["expected_refusal"] for r in gemma)
    assert COUNTS["expected_refusal_rows"] == [] and COUNTS["servable"] == COUNTS["launched"] == 38 and COUNTS["contrasts_computable"] == COUNTS["contrasts_designed"] == 32
    assert "EXPECTED refusals" not in (G8K / "dry_run_prod.sbatch").read_text()


def test_every_emitted_g8k_literal_is_read_by_the_shell_no_dead_env_contract():
    """The simplify pass: eight `G8K_*` literals rode `RUN_LITERALS_SH` and nothing read them. Now EVERY name the row,
    the overlap leg and the run literals emit is read by g8k_common.sh or a launcher (`$NAME` / `${NAME`), and the datasets
    guard reads the names it once hard-coded."""
    text = "".join((G8K / n).read_text() for n in SBATCHES + ["g8k_common.sh"]) + (F22 / "served_thinking.sh").read_text() + (F22 / "common.sh").read_text()
    emitted = set()
    for out in (M.row_sh(ROWS[9]), M.row_sh(ROWS[9], M.RUNG_CAP), M.row_sh(ROWS[0]), M.overlap_leg_sh(M.overlap_legs()[0]), M.literals_sh()):
        emitted |= {line.split("=", 1)[0] for line in out.splitlines()}
    dead = sorted(n for n in emitted if not re.search(r"\$\{?" + re.escape(n) + r"\b", text))
    assert dead == [], dead
    for gone in ("G8K_REQUEST_EVIDENCE_FILENAME", "G8K_OVERLAP_THINKING", "G8K_OVERLAP_SAMPLING", "G8K_OVERLAP_CAP", "G8K_OVERLAP_GEOMETRY_EVAL",
                 "G8K_OVERLAP_GEOMETRY_L", "G8K_TIME_OVERLAP", "G8K_HARNESS_CELL"):
        assert gone not in emitted and gone not in text, gone
    assert {"G8K_DATASETS_REQUIRED", "G8K_DATASETS_CONFIG_DIRS", "G8K_ROW_TASK", "G8K_STANDUP_CELL", "G8K_STANDUP_HARNESS_CELL"} <= emitted
    # the CLI subcommand g8k_init evals is the same source
    r = subprocess.run([sys.executable, str(G8K / "g8k_cells.py"), "literals", "--sh"], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and r.stdout == M.literals_sh()
    assert dict(M.RUN_LITERALS_SH)["G8K_ROW_TASK"] == C.GSM8K_STANDARD_TASK


def _guard(hf_cache: Path):
    script = (_g8k_funcs() + f"\nREPO_DIR={REPO}\n" + M.literals_sh() + f"\nHF_DATASETS_CACHE={hf_cache}\ng8k_datasets_guard && echo GUARD_OK")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)


def test_the_datasets_guard_accepts_d18s_symlinked_offline_layout_and_refuses_a_missing_config_dir_with_the_recipe(tmp_path):
    """`datasets` 5.0.1 looks OFFLINE for `openai___gsm8k/default-data_dir=main` while the online prepare wrote a hash-named
    dir, so the prepared per-user caches on both clusters carry that name as a SYMLINK to the hash-named dir (platinum's `main`
    is named `main`). The guard follows the link; a dangling or absent config dir refuses by name with the recipe; the probe's
    refusal spells the same recipe."""
    hf = tmp_path / "hf_datasets"
    (hf / "openai___gsm8k" / M.DATASETS_ONLINE_CONFIG_DIR_EXAMPLE / "0.0.0").mkdir(parents=True)
    (hf / "madrylab___gsm8k-platinum" / "main").mkdir(parents=True)
    alias = hf / "openai___gsm8k" / "default-data_dir=main"
    alias.symlink_to(M.DATASETS_ONLINE_CONFIG_DIR_EXAMPLE)
    r = _guard(hf)
    assert r.returncode == 0 and "GUARD_OK" in r.stdout and f"symlink -> {M.DATASETS_ONLINE_CONFIG_DIR_EXAMPLE}" in r.stdout, r.stdout + r.stderr
    assert "madrylab___gsm8k-platinum/main present" in r.stdout
    # the alias dangling: refused by name, the recipe in the message
    alias.unlink(); alias.symlink_to("gone")
    r = _guard(hf)
    assert r.returncode == 1 and "GUARD_OK" not in r.stdout
    assert "carries no offline config dir openai___gsm8k/default-data_dir=main" in r.stdout
    for frag in ("RECIPE (D18)", "HF_HUB_OFFLINE=0", "ln -s", "default-data_dir=main", "g8k_datasets_probe OFFLINE"):
        assert frag in r.stdout, frag
    # the whole dataset dir absent: the first loop refuses
    shutil.rmtree(hf / "madrylab___gsm8k-platinum")
    r = _guard(hf)
    assert r.returncode == 1 and "carries no prepared madrylab___gsm8k-platinum" in r.stdout
    # the probe's refusal describes the same recipe (the wrapper's own plan is the authority; the guard is necessary, not sufficient)
    common_sh = (G8K / "g8k_common.sh").read_text()
    probe_msg = common_sh[common_sh.index('f22_fatal "datasets probe FAILED'):]
    probe_msg = probe_msg[:probe_msg.index("\n")]
    for frag in ("RECIPE (D18", "HF_HUB_OFFLINE=0", "default-8cbc416ed05c3005", "ln -s default-8cbc416ed05c3005", "default-data_dir=main", "OFFLINE on a compute node"):
        assert frag in probe_msg, frag
    assert "`" not in probe_msg
    # the sbatch bodies: the guard runs from g8k_init, before any venv (the CPU launchers too)
    for name in SBATCHES:
        src = (G8K / name).read_text()
        assert 'g8k_init "$WAVE_SHA"' in src
    assert common_sh.index("g8k_init(){") < common_sh.index("  g8k_datasets_guard\n") < common_sh.index("g8k_datasets_guard(){")


# --------------------------------------------------------------------------- #
# llama-paper-rows (P7 / P11 / P15 / P16): the SECOND matrix, its own root,   #
# the no-switch arm through the launcher; the published matrix frozen         #
# --------------------------------------------------------------------------- #

LLAMA_MATRIX = M.MATRICES[M.MATRIX_LLAMA_NAME]
LLAMA_ALL_ROWS = M.matrix_rows(LLAMA_MATRIX)
LLAMA_ROWS = M.rows_of_wave(M.WAVE_PUBLISHED, LLAMA_ALL_ROWS)          # the Llama matrix's twelve (COMPLETE at its pin; the steer rows append behind)
LLAMA_STEER_ROWS = M.rows_of_wave(M.WAVE_STEER, LLAMA_ALL_ROWS)
LLAMA_DOC = M.matrix_doc(LLAMA_MATRIX)
LLAMA_VIEWS = M.VIEWS_BY_MATRIX[M.MATRIX_LLAMA_NAME]
FROZEN = REPO / "configs/sdfbase/gsm8k/frozen/8f5f1495"


def _sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_the_published_matrix_regenerates_from_its_frozen_inputs_and_the_live_registry_has_moved_past_them():
    """The published 42-row document is regenerated from the roster + pins COMMITTED AT THE PIN (immutable copies under
    configs/sdfbase/gsm8k/frozen/8f5f1495/, their sha256 declared in views.yaml and asserted by the generator), byte for byte;
    the LIVE registry now carries a fifth target (llama8b) and the constitution prompted cell on published views, so the
    same views over the live inputs yield the same 42 scheduled rows (plus the trailing held rows) under a DIFFERENT
    roster / pins hash — the drift the frozen inputs exist to keep out of the frozen document."""
    spec = V.load_views_yaml(REPO / V.VIEWS_REL)
    assert spec.schema == M.MATRIX_SCHEMA and spec.wave_root_rel is None and spec.exclude == [] and spec.frozen_inputs is not None
    frozen = V.frozen_input_paths(spec, REPO)
    assert frozen == (FROZEN / "sdfbase_full22_roster.yaml", FROZEN / "sdfbase_full22_pins.json")
    doc = json.loads((REPO / "configs/sdfbase/gsm8k/matrix.json").read_text())
    assert doc["schema"] == M.MATRIX_SCHEMA and "wave_root_rel" not in doc and "exclude" not in doc and "thinking_mode" not in doc["rows"][0]
    # the recorded hashes ARE the frozen copies' (= the pin's tree), and the recorded paths are the published registry's
    assert doc["source"]["roster_sha256"] == _sha256(frozen[0]) == spec.frozen_inputs["roster_sha256"]
    assert doc["source"]["pins_sha256"] == _sha256(frozen[1]) == spec.frozen_inputs["pins_sha256"]
    assert doc["source"]["roster"] == "configs/sdfbase_full22_roster.yaml" and doc["source"]["pins"] == "configs/sdfbase_full22_pins.json"
    # the live registry moved: five targets, a llama8b pins block — the same rows, another hash
    live_roster, live_pins = V.load_roster(V.roster_path(REPO)), V.load_pins(V.pins_path(REPO))
    # the paper's five are IN the live roster (a subset, not equality), and today the ONE other block is the worked example's front-door
    # target qwen35_4b_demo (05 --step roster; demo-import c7e73cd04 — its views are battery / d0p75 / d0p75_relu, none a published view, so
    # it adds no row below): a further front-door target joins the roster only through a deliberate edit of this line, never a silent pass
    assert set(M.MODEL_KEYS) <= set(live_roster["targets"]) and set(live_roster["targets"]) - set(M.MODEL_KEYS) == {"qwen35_4b_demo"} and "llama8b" in live_pins
    assert _sha256(V.roster_path(REPO)) != doc["source"]["roster_sha256"] and _sha256(V.pins_path(REPO)) != doc["source"]["pins_sha256"]
    frozen_roster = V.load_roster(frozen[0])
    assert set(frozen_roster["targets"]) == set(M.MODEL_KEYS) - {"llama8b"}
    assert V.select_rows(frozen_roster, V.load_pins(frozen[1]), spec.views, holds=spec.holds) == ROWS
    # the live roster's other addition since the pin — the constitution prompted cell main declares on the published views —
    # yields only TRAILING held rows behind the 42, never a scheduled row; the frozen document carries none of them
    live_rows = _without_batch64_lane(V.select_rows(live_roster, live_pins, spec.views, holds=spec.holds))   # matrix itdoc's cells set aside
    assert live_rows[:len(ROWS)] == ROWS and live_rows[len(ROWS):] == M.unscheduled_held_rows(live_rows)
    assert {r["cell"] for r in live_rows[len(ROWS):]} <= {M.CELL_PROMPTED_CONST}
    # the four other targets' pins blocks: every entry of the pin is in the live registry byte for byte, and
    # the live registry has only ADDED to them since (the strength-dose lane's ladder pins `d<dose>[it]__…` — the drift the frozen
    # inputs keep out of the frozen document; never a removed or a rewritten entry)
    frozen_pins = V.load_pins(frozen[1])
    for key, block in frozen_pins.items():
        live_block = live_pins[key]
        assert set(block) <= set(live_block) and all(live_block[sid] == block[sid] for sid in block), key
        # the strength-dose lane's bakes (`d<dose>[it]__`) AND, since 2026-09-13, the batch-64 instruct-direct lane's staged ids
        # (`itdoc64__` / `derived_itdoc64__`, matrix itdoc's) — every other addition to a pinned target's block still fails by name
        # ... and native-psr-serving's certification-only `si12__` pin (the constitution response-fit psr, the lane the
        # frozen block's own si12__ pins came from) ... and the qwen-base-seed-replicates lane's staged objects (`base64s<s>__`
        # parents and the admitted `derived_base64s<s>__` LoRA differences, pinned 2026-09-17 from the clusters' staging manifest;
        # the distilled PSR difference failed admission for every seed and has no pin) — every other addition to
        # a pinned target's block still fails by name ... and (2026-09-20) the composed paired-write
        # difference_relu objects `paired_eb64s<s>__` staged for the paired views s<s>p (admitted with the floor alarm by
        # derived_admission.py's psr_paired_write branch)
        # ... and the Arditi refusal-steer object's served-only Class-B identity pin `ar_c0p3__arditi_allpos` (the
        # `ar_<dose>__` family, deliberately admitted here rather than renamed — the OR-Bench and GSM8K records already carry it)
        # ... and (2026-09-23) the all-models one-seed lane's composed paired-write objects `paired_eb8__` /
        # `paired_eb64__` (gemma4e4b / nemotron3nano4b / llama8b: the number names the PARENTS' effective batch, no seed suffix — one seed)
        # ... and (2026-10-04) the dose ladder's bake on a qwen-base-seed-replicate eb64 PARENT, `d<dose>_eb64s<s>__`
        # (`d0p5_eb64s1__constv2-mix-full__lora_r4_ce_minus_ctl`: the registry-backed addition that landing pinned beside the `d<dose>__` bakes)
        assert all(re.match(r"^(d[0-9pm]+(it)?(64)?__|d[0-9pm]+_eb64s[0-9]+__|itdoc64__|derived_itdoc64__|si12__|base64s[0-9]+__|derived_base64s[0-9]+__|paired_eb64s[0-9]+__|paired_eb(8|64)__|ar_c[0-9p]+__)", sid) for sid in set(live_block) - set(block)), (key, sorted(set(live_block) - set(block)))
    # an edited frozen copy refuses BY NAME (never a silently moved document)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "g8k_frozen_edit"
    shutil.rmtree(tmp, ignore_errors=True)
    for rel in ("configs/sdfbase/gsm8k/frozen/8f5f1495/sdfbase_full22_roster.yaml", "configs/sdfbase/gsm8k/frozen/8f5f1495/sdfbase_full22_pins.json"):
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp / rel)
    (tmp / "configs/sdfbase/gsm8k/frozen/8f5f1495/sdfbase_full22_pins.json").write_text("{}\n")
    with pytest.raises(SystemExit, match="frozen pins .* hashes to .* not the declared .* IMMUTABLE"):
        V.frozen_input_paths(spec, tmp)
    # a schema-1 views file may not carry the schema-2 FIELDS
    for bad_doc, why in (({**yaml.safe_load((REPO / V.VIEWS_REL).read_text()), "wave_root_rel": "exp/x"}, "is a g8k_matrix/2 field"),
                         ({**yaml.safe_load((REPO / V.VIEWS_REL).read_text()), "trigger_threshold": 0.05}, "is a g8k_matrix/2 field")):
        p = tmp / "views_bad.yaml"
        p.write_text(yaml.safe_dump(bad_doc))
        with pytest.raises(SystemExit, match=why):
            V.load_views_yaml(p)
    # ... but a schema-1 file's `exclude:` block is ANOTHER lane's, not this loader's: the IFEval views.yaml carries one in its own
    # grammar ({arms_matching | cells, reason}, read by if_views.load_ifeval_policy) and reuses only this neutral enumeration (P31) —
    # the block stays unread (never a schema-2 exclusion, never a refusal of a file the loader does not own), whichever grammar it uses
    for foreign in ([{"arms_matching": "psr", "reason": "the IFEval lane's grammar"}],
                    [{"key": "gemma4e4b", "view": "it", "cells": ["none"], "reason": "the GSM8K schema-2 grammar on a schema-1 file"}]):
        p = tmp / "views_foreign_exclude.yaml"
        p.write_text(yaml.safe_dump({**yaml.safe_load((REPO / V.VIEWS_REL).read_text()), "exclude": foreign}))
        assert V.load_views_yaml(p).exclude == [] and V.load_views_yaml(p).schema == M.MATRIX_SCHEMA


def test_the_llama_matrix_is_schema_2_under_its_own_root_with_the_no_switch_arm_on_every_row():
    py = sys.executable
    for argv in ([py, str(G8K / "make_gsm8k_matrix.py"), "--matrix-name", "llama", "--check"],
                 [py, str(G8K / "make_gsm8k_overlays.py"), "--matrix", "llama", "--check"]):
        r = subprocess.run(argv, capture_output=True, text=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, check=False)
        assert r.returncode == 0, r.stdout + r.stderr
    spec = V.load_views_yaml(LLAMA_VIEWS)
    assert spec.schema == M.MATRIX_SCHEMA_2 and spec.wave_root_rel == "exp/gsm8k_paired200_paper_objects_llama" != M.WAVE_ROOT_REL
    assert spec.views == {"llama8b": ["main", "it"]} and spec.holds == [] and spec.exclude == []
    # IMMUTABLE INPUTS since 2026-09-14 (the third re-drift; main PRs #102 / #107 / #141 each moved the live registry for another
    # lane and drifted this document's header with its rows unchanged): the Llama matrix regenerates from the roster + pins frozen
    # at the commit `frozen_inputs:` names (2f42b0c3, the merge that last regenerated it against the live registry) — never the live
    # registry, exactly like the published matrix (P16). The recorded hashes are the frozen copies' and the recorded paths the live
    # registry's. The live registry may still equal the copies today (it has not moved since 2f42b0c3); the guard is that a later move
    # leaves this document where it is, so this test pins hashes to the COPIES, not to the live files' inequality.
    frozen = V.frozen_input_paths(spec, REPO)
    assert spec.frozen_inputs is not None and frozen is not None and frozen[0].parent == frozen[1].parent
    assert frozen == (frozen[0].parent / "sdfbase_full22_roster.yaml", frozen[0].parent / "sdfbase_full22_pins.json")
    assert LLAMA_DOC["source"]["roster_sha256"] == _sha256(frozen[0]) == spec.frozen_inputs["roster_sha256"]
    assert LLAMA_DOC["source"]["pins_sha256"] == _sha256(frozen[1]) == spec.frozen_inputs["pins_sha256"]
    assert LLAMA_DOC["source"]["roster"] == "configs/sdfbase_full22_roster.yaml" and LLAMA_DOC["source"]["pins"] == "configs/sdfbase_full22_pins.json"
    assert "llama8b" in V.load_roster(frozen[0])["targets"] and "llama8b" in V.load_pins(frozen[1])
    # the same views over the FROZEN inputs reproduce every published row (the selector is deterministic in its inputs)
    assert V.select_rows(V.load_roster(frozen[0]), V.load_pins(frozen[1]), spec.views, holds=spec.holds, schema=spec.schema,
                         trigger_threshold=spec.trigger_threshold)[:12] == LLAMA_ROWS
    # since 2026-09-12 the ladder trigger is 1 % RAW at-cap for everything — the schema-2 views file DECLARES it, the
    # document and every row carry it, the launcher reads the row's; the published matrix declares none and reads the adapter's 3 %
    from sisteer.capeval.tiers import inspect_rung as IR
    assert spec.trigger_threshold == LLAMA_DOC["trigger_threshold"] == M.TRIGGER_THRESHOLD == IR.GSM8K_TRIGGER_THRESHOLD == 0.01
    assert all(r["trigger_threshold"] == 0.01 == M.row_trigger_threshold(r) for r in LLAMA_ROWS)
    assert M.TRIGGER_THRESHOLD_3PCT_RETIRED == IR.GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED == 0.03 and all("trigger_threshold" not in r and M.row_trigger_threshold(r) == 0.01 for r in ROWS)
    assert "trigger_threshold" not in json.loads((REPO / "configs/sdfbase/gsm8k/matrix.json").read_text())
    assert LLAMA_DOC["schema"] == M.MATRIX_SCHEMA_2 and LLAMA_DOC["wave_root_rel"] == spec.wave_root_rel == M.wave_root_rel_of(LLAMA_DOC)
    assert LLAMA_DOC["exclude"] == [] and M.wave_root_rel_of(json.loads((REPO / "configs/sdfbase/gsm8k/matrix.json").read_text())) == M.WAVE_ROOT_REL
    c = M.counts(LLAMA_ROWS)                     # the twelve published rows' shape (E2's two steer rows have their own test)
    assert M.published_counts(LLAMA_ALL_ROWS) == c and len(LLAMA_ALL_ROWS) == 12 + len(LLAMA_STEER_ROWS) and LLAMA_ALL_ROWS[:12] == LLAMA_ROWS
    # TWELVE rows = the SIX-ARM design x 2 views (since 2026-09-12 PSR is OUT BY DEFAULT — not run, not
    # plotted): per view none, prompted__safety_refusal, prompted__si_constitution, the two LoRA fits, the LoRA subtraction
    assert {k: c[k] for k in ("rows", "held", "launched", "fitted", "fitted_launched", "in_support", "declared", "expected_refusals", "servable",
                              "contrasts_designed", "contrasts_computable", "n_tags")} == {
        "rows": 12, "held": 0, "launched": 12, "fitted": 6, "fitted_launched": 6, "in_support": 6, "declared": 0, "expected_refusals": 0,
        "servable": 12, "contrasts_designed": 10, "contrasts_computable": 10, "n_tags": 2}
    assert c["tags"] == ["llama8b", "llama8b_it"] and c["models"] == ["llama8b"] and c["families"] == ["llama"] and c["pins"] == {"llama8b": 6}
    assert c["tuple_names"] == c["tuple_names_launch"] == [] and c["placement"] == {"primary": 0, "any": 12} and c["reasoning_parser_gemma4"] == 0
    # roster order — the llama8b block since it landed (2026-09-12): mix, then ctl2, then the subtraction. The
    # production wave's pin (9ed8e7e1) listed ctl2 first, so rows 1/2 and 8/9 swapped INDEX at the landing (same cells, staged ids,
    # shas, out_roots and job ids as a set); array indices are resolved on the tree a launch runs from, never reused across pins
    OBJECTS3 = ("constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl")
    PROMPTED2 = ("prompted__safety_refusal", "prompted__si_constitution")
    # the cell rule (g8k_views.cells_of): `none` first; on an `it` view the matched prompted SAFETY baseline SECOND (the fixed
    # contract since the sampled re-run's lane F); every other cell in roster order — the three LoRA objects, then the two prompted
    # baselines on `main`; the constitution baseline rides in roster order like an object cell (LAST on both views)
    assert [(r["view"], r["cell"]) for r in LLAMA_ROWS] == ([("main", "none")] + [("main", c) for c in OBJECTS3] + [("main", p) for p in PROMPTED2]
                                                             + [("it", "none"), ("it", "prompted__safety_refusal")] + [("it", c) for c in OBJECTS3]
                                                             + [("it", "prompted__si_constitution")])
    assert [r["job_id"] for r in LLAMA_ROWS if r["view"] == "main"] == ["g8k-llama8b-main-none", "g8k-llama8b-main-mix-lora", "g8k-llama8b-main-ctl-lora",
                                                                        "g8k-llama8b-main-mix-dlora", "g8k-llama8b-main-prompted", "g8k-llama8b-main-prompted-const"]
    assert [r["job_id"] for r in LLAMA_ROWS if r["view"] == "it"][:2] == ["g8k-llama8b-it-none", "g8k-llama8b-it-prompted"]
    assert not any("psr" in r["cell"] for r in LLAMA_ROWS) and not any("distilled" in r["cell"] for r in LLAMA_ROWS)   # PSR out
    # the derived object's pins are the lane's (verified against the pins at de78ab46 through the loader, I21): the LoRA subtraction alone
    DERIVED_SHA = {("main", "constv2-mix-full__lora_r4_ce_minus_ctl"): "4375b67a0fb0621e9ee296041d0842cb24294112e18424a803e8e54e9538a740",
                   ("it", "constv2-mix-full__lora_r4_ce_minus_ctl"): "e85220c9e5d2b9aa4377fea7a9a590048c693118975be9f36ae579b0eeff6f13"}
    assert {(r["view"], r["cell"]): r["object_sha256"] for r in LLAMA_ROWS if "minus_ctl" in r["cell"]} == DERIVED_SHA
    assert not any("si12" in (r["staged_id"] or "") for r in LLAMA_ROWS)          # the SI-distilled comparators are OUT
    # the prompted CONSTITUTION baseline: a prompted route (no object, no pin), its OWN prompt class -> its own geometry entries
    # (16,384 at the 8,192 pass, 24,576 at the rung: measured 2026-09-12, prompt max 8,007) and budget class, the fitted time class
    for r in (x for x in LLAMA_ROWS if x["cell"] == M.CELL_PROMPTED_CONST):
        assert r["route_kind"] == "prompted" and not r["fitted"] and r["object_sha256"] is None and r["harness_cell"] == "prompted__constitution"
        assert r["budget_class"] == M.CELL_PROMPTED_CONST and r["time_limit"] == M.TIME_FITTED_LLAMA and r["job_id"].endswith("-prompted-const")
        # E2 / P3: the long-SI class is SCOPED BY FAMILY — the Llama family resolves PR-A's Llama-scoped entries, exactly as before
        assert M.prompt_class(r["cell"]) == M.PROMPT_CLASS_LONG_SI and M.geometry_for_cell(r["cell"], M.CAP, model_key="llama8b") == ("capability:gen_long_gsm8k_longsi", 16384)
        assert M.geometry_for_cell(r["cell"], M.RUNG_CAP, model_key="llama8b") == ("capability:gen_long_gsm8k_rung_longsi", 24576)
        with pytest.raises(SystemExit, match="scoped by family"):
            M.geometry_for_cell(r["cell"], M.CAP)                                   # the scope is part of the selection, never a default
    for r in (x for x in LLAMA_ROWS if x["cell"] != M.CELL_PROMPTED_CONST):
        assert M.prompt_class(r["cell"]) == M.PROMPT_CLASS_SHORT and M.geometry_for_cell(r["cell"], M.CAP) == (M.GEOMETRY_EVAL_BY_RUNG[M.CAP], 10752)
        assert r["budget_class"] == (M.CELL_PROMPTED if r["cell"] == M.CELL_PROMPTED else M.CELL_NONE)
    assert M.cell_short(M.CELL_PROMPTED) == "prompted" and M.cell_short(M.CELL_PROMPTED_CONST) == "prompted-const"
    assert all(M.geometry_for_cell(r["cell"], M.CAP) == (M.GEOMETRY_EVAL_BY_RUNG[M.CAP], 10752) for r in ROWS)              # the published rows: untouched
    pins = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())
    for r in LLAMA_ROWS:
        assert r["model_key"] == "llama8b" and r["family"] == "llama" and r["thinking_mode"] == M.THINKING_NOT_AVAILABLE == C.THINKING_NOT_AVAILABLE
        assert M.row_thinking(r) == C.THINKING_NOT_AVAILABLE and M.row_sampling_mode(r) == M.SAMPLING_CARD == C.SAMPLING_CARD
        assert r["sampling_mode"] == "card" and r["sampling"] == M.SAMPLING_ROWS["llama8b"] and r["sampling"]["temperature"] == 0.6 and r["sampling"]["top_p"] == 0.9
        assert r["out_root"] == f"controls/{r['tag']}/{r['cell_slug']}/card" and r["rung_out_root"] == r["out_root"] + "__rung16384"
        assert r["declared"] is False and r["tuple_name"] is None and r["tuple_names"] is None       # IN SUPPORT even when fitted (P7)
        assert r["expected_stamps"] == list(C.g8k_expected_stamps(fitted=r["fitted"], declared=False, arm=C.THINKING_NOT_AVAILABLE))
        assert r["expected_stamps"][2] == C.RENDER_REGIME_UNSPECIFIED and r["split_mode_candidate"] == "none" == C.THINK_SPLIT_NONE
        assert r["reasoning_parser"] is None and r["cluster_constraint"] == "any" and r["status"] == "launch" and r["expected_refusal"] is False
        assert r["time_limit"] == (M.TIME_FITTED_LLAMA if (r["fitted"] or r["cell"] == M.CELL_PROMPTED_CONST) else M.TIME_IN_SUPPORT_LLAMA)   # the constitution cell: the 8k prefill
        if r["fitted"]:
            prefix = ("derived__" if r["view"] == "main" else "derived_itdoc__") if "minus_ctl" in r["cell"] else ("base__" if r["view"] == "main" else "itdoc__")
            assert r["staged_id"] == prefix + r["cell"] and r["object_sha256"] == pins["llama8b"][r["staged_id"]]["object_sha256"]
            assert r["served_positions_expected"] == "all" and r["expected_stamps"] == ["cell-stable", "always", "unspecified"] and r["route_kind"] == "lens"
        else:
            assert r["object_sha256"] is None and r["expected_stamps"] == list(C.G8K_STAMPS_BASE_NOT_AVAILABLE)
    assert {M.time_limit_hours(t) for t in (M.TIME_IN_SUPPORT_LLAMA, M.TIME_FITTED_LLAMA)} == {0.75, 1.0}
    assert {t: len(ix) for t, ix in M.time_classes(LLAMA_ROWS).items()} == {M.TIME_IN_SUPPORT_LLAMA: 4, M.TIME_FITTED_LLAMA: 8}   # 2 none + 2 safety-prompted / 6 LoRA + 2 constitution
    # the arm is a FAMILY fact (contracts.NO_THINKING_CHANNEL_FAMILIES), the render fact a candidate with NO switch
    assert M.THINKING_MODE_BY_FAMILY == {"gemma4": "on", "qwen3_5": "on", "glm4v": "on", "nemotron_h": "on", "llama": "not_available"}
    assert {f for f, a in M.THINKING_MODE_BY_FAMILY.items() if a == M.THINKING_NOT_AVAILABLE} == set(C.NO_THINKING_CHANNEL_FAMILIES)
    assert C.FAMILY_RENDER_FACTS["llama"].thinking_switch is False and C.FAMILY_RENDER_FACTS["llama"].split_mode == "none"
    # a no-switch row can never enter a schema-1 matrix; a header-located cell can never be a Llama row
    roster, live_pins = V.load_roster(V.roster_path(REPO)), V.load_pins(V.pins_path(REPO))
    with pytest.raises(SystemExit, match="only a g8k_matrix/2 matrix"):
        V.select_rows(roster, live_pins, {"llama8b": ["main"]})
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "g8k_schema_mix"
    tmp.mkdir(parents=True, exist_ok=True)
    doc1 = json.loads((REPO / "configs/sdfbase/gsm8k/matrix.json").read_text()); doc1["rows"][0]["thinking_mode"] = "not_available"
    (tmp / "m1.json").write_text(json.dumps(doc1))
    with pytest.raises(SystemExit, match="schema-2 field"):
        M.load_matrix(tmp / "m1.json")
    doc2 = json.loads(LLAMA_MATRIX.read_text()); doc2["rows"][0].pop("thinking_mode")
    (tmp / "m2.json").write_text(json.dumps(doc2))
    with pytest.raises(SystemExit, match="schema 2 requires"):
        M.load_matrix(tmp / "m2.json")
    doc3 = json.loads(LLAMA_MATRIX.read_text()); doc3.pop("wave_root_rel")
    (tmp / "m3.json").write_text(json.dumps(doc3))
    with pytest.raises(SystemExit, match="names its wave root"):
        M.load_matrix(tmp / "m3.json")
    # the overlays state the arm; the TAGs' campaign files are the registry's (five targets, P4)
    cfg = (REPO / "configs/sdfbase/gsm8k/sdfbase_full22_llama8b.gsm8k.yaml").read_text()
    assert "thinking: NOT AVAILABLE (the no-switch arm" in cfg and yaml.safe_load(cfg)["explicit_cells"] == [r["cell"] for r in LLAMA_ROWS if r["view"] == "main"]
    cfg_it = yaml.safe_load((REPO / "configs/sdfbase/gsm8k/sdfbase_full22_llama8b_it.gsm8k.yaml").read_text())
    assert cfg_it["explicit_cells"] == [r["cell"] for r in LLAMA_ROWS if r["view"] == "it"] and cfg_it["model"] == "NousResearch/Meta-Llama-3.1-8B-Instruct"
    gemma_cfg = (REPO / "configs/sdfbase/gsm8k/sdfbase_full22_gemma4e4b.gsm8k.yaml").read_text()
    assert "thinking ON declared on every request." in gemma_cfg and "NOT AVAILABLE" not in gemma_cfg
    # the analysis' real reader admits the schema-2 document (the twelve + the two steer rows, every row under the no-switch arm)
    mx = VAL.load_matrix(LLAMA_MATRIX)
    assert mx["counts"]["launched"] == 12 + len(M.launch_rows(LLAMA_STEER_ROWS)) and all(r.get("thinking_mode") == "not_available" for r in mx["rows"])


def test_the_row_literals_carry_the_arm_the_sampling_mode_and_the_matrix_root_and_the_overlap_legs_skip_the_no_switch_family():
    fitted = next(r for r in LLAMA_ROWS if r["fitted"] and r["view"] == "main")
    sh = dict(l.split("=", 1) for l in M.row_sh(fitted, wave_root_rel=M.wave_root_rel_of(LLAMA_DOC)).splitlines())
    assert sh["G8K_THINKING"] == "not_available" and sh["G8K_SAMPLING_MODE"] == "card" and sh["G8K_TUPLE"] == "''" and sh["FITTED"] == "1"
    assert sh["G8K_WAVE_ROOT_REL"] == "exp/gsm8k_paired200_paper_objects_llama" and sh["PIN"] == fitted["object_sha256"]
    assert sh["G8K_EXPECTED_STAMPS"] == "cell-stable,always,unspecified" and sh["G8K_REASONING_PARSER"] == "''" and sh["G8K_TIME"] == M.TIME_FITTED_LLAMA
    assert sh["G8K_GEOMETRY_EVAL"] == M.GEOMETRY_EVAL_BY_RUNG[M.CAP] and sh["G8K_GEOMETRY_L"] == "10752"
    rung = dict(l.split("=", 1) for l in M.row_sh(fitted, M.RUNG_CAP, wave_root_rel=M.wave_root_rel_of(LLAMA_DOC)).splitlines())
    assert rung["G8K_TUPLE"] == "''" and rung["RUNG"] == "16384" and rung["G8K_OUT_REL"].endswith("/card__rung16384")
    # the published rows: `on` / thinking_card / the published root — the SAME emitted names (one launcher, one vocabulary)
    pub = dict(l.split("=", 1) for l in M.row_sh(ROWS[9]).splitlines())
    assert pub["G8K_THINKING"] == "on" and pub["G8K_SAMPLING_MODE"] == "thinking_card" and pub["G8K_WAVE_ROOT_REL"] == M.WAVE_ROOT_REL
    assert set(pub) == set(sh) and pub["G8K_TUPLE"] == ROWS[9]["tuple_names"]["8192"]
    assert "G8K_THINKING" not in dict(M.RUN_LITERALS_SH) and "G8K_SAMPLING_MODE" not in dict(M.RUN_LITERALS_SH)      # ROW facts now (P7)
    # the literals name the MATRIX's root — evaluated by g8k_init BEFORE F22_WAVE_ROOT; the CLI is the same source
    assert M.literals_sh().splitlines()[-1] == f"G8K_WAVE_ROOT_REL={M.WAVE_ROOT_REL}"
    assert M.literals_sh(LLAMA_MATRIX).splitlines()[-1] == "G8K_WAVE_ROOT_REL=exp/gsm8k_paired200_paper_objects_llama"
    r = subprocess.run([sys.executable, str(G8K / "g8k_cells.py"), "literals", "--sh", "--matrix", "llama"], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and r.stdout == M.literals_sh(LLAMA_MATRIX)
    common = (G8K / "g8k_common.sh").read_text()
    assert 'g8k_literals(){' in common and 'literals --sh --matrix "$G8K_MATRIX"' in common
    init = common[common.index("g8k_init(){"):common.index("g8k_datasets_guard(){")]
    assert init.index("g8k_literals") < init.index('export F22_WAVE_ROOT="${F22_WAVE_ROOT:-$shared/$G8K_WAVE_ROOT_REL}"') < init.index('f22_init "$sha"')
    assert 'G8K_MATRIX="${G8K_MATRIX:-default}"' in common
    # the overlap legs: switch-arm models only; a no-switch model's overlap is not_applicable (P8), never a leg
    assert M.OVERLAP_MODEL_KEYS == ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash") and len(M.overlap_legs()) == 4
    with pytest.raises(SystemExit, match="no overlap leg for \\['llama8b'\\]"):
        M.overlap_legs(("gemma4e4b", "llama8b"))
    with pytest.raises(SystemExit, match="no overlap leg for"):
        M.overlap_leg_for("llama8b")
    with pytest.raises(SystemExit, match="not_applicable"):
        M.plan_overlap(SHA0, "cluster_a", keys=("llama8b",))
    leg = dict(l.split("=", 1) for l in M.overlap_leg_sh(M.overlap_legs()[0]).splitlines())
    assert leg["G8K_THINKING"] == "off" and leg["G8K_SAMPLING_MODE"] == "greedy" and leg["G8K_WAVE_ROOT_REL"] == M.WAVE_ROOT_REL
    # the geometry: llama8b's measured budget resolves the shared entries (P20; no per-row override)
    assert M.FLAG_BANK_VIEW == C.FLAG_BANK_VIEW == "--bank-view"


def test_the_plan_names_a_non_default_matrix_by_its_trailing_positionals_and_the_default_line_is_unchanged():
    pl = M.plan(SHA0, "cluster_a", max_concurrent=16, matrix=LLAMA_MATRIX, universe=LLAMA_ROWS)     # the lane's twelve (the python universe entry)
    assert pl.matrix == "llama" and pl.wave_root_rel == "exp/gsm8k_paired200_paper_objects_llama" and pl.rows == list(range(12)) and pl.held == []
    assert len(pl.lines) == 2 and all(line.endswith(f"gsm8k_thinking.sbatch ARRAY {SHA0} 8192 llama") and f"--qos={SITE_A['qos']} " in line for line in pl.lines)
    assert [re.search(r"-J (g8k-\S+)", line).group(1) for line in pl.lines] == ["g8k-fitted-llama", "g8k-in-support-llama"]
    assert sorted(_covered(pl.lines)) == list(range(12)) and sum(_widths(pl.lines)) == 12
    assert any(line.startswith("# matrix: llama -> wave root <shared root>/exp/gsm8k_paired200_paper_objects_llama") for line in pl.summary())
    pub = M.plan(SHA0, "cluster_a", max_concurrent=12, universe=ROWS)
    assert pub.matrix == "default" and all(line.endswith(f"gsm8k_thinking.sbatch ARRAY {SHA0}") for line in pub.lines) and not any("# matrix:" in s for s in pub.summary())
    assert M.array_tail(SHA0) == f"ARRAY {SHA0}" and M.array_tail(SHA0, rung=M.RUNG_CAP) == f"ARRAY {SHA0} 16384"
    assert M.array_tail(SHA0, rung=M.RUNG_CAP, matrix=LLAMA_MATRIX) == f"ARRAY {SHA0} 16384 llama"
    with pytest.raises(SystemExit, match="by NAME"):
        M.array_tail(SHA0, matrix=Path("/tmp/not-a-registered-matrix.json"))
    assert M.resolve_matrix(None) == M.resolve_matrix("default") == M.MATRIX_PATH and M.resolve_matrix("llama") == LLAMA_MATRIX
    assert M.matrix_name_of(M.MATRIX_PATH) == "default" and M.matrix_name_of(LLAMA_MATRIX) == "llama"
    with pytest.raises(SystemExit, match="neither a matrix name"):
        M.resolve_matrix("bogus")
    # the CLI: the same plan through --matrix — WAVE-SCOPED since E2 (the lane's twelve are COMPLETE: `--wave published` refuses by
    # name, and the matrix positional rides the wave's line exactly as it rode the lane's)
    steer = M.plan(SHA0, "cluster_a", max_concurrent=16, matrix=LLAMA_MATRIX, wave=M.WAVE_STEER)
    assert steer.rows == [12, 13] and all(line.endswith(f"gsm8k_thinking.sbatch ARRAY {SHA0} 8192 llama") for line in steer.lines)
    r = subprocess.run([sys.executable, str(G8K / "g8k_cells.py"), "plan", "--sha", SHA0, "--cluster", "cluster_a", "--max-concurrent", "16", "--matrix", "llama",
                        "--wave", M.WAVE_STEER], capture_output=True, text=True, check=False)
    # math500-steer: an all-external (steer) submission's line carries the lens route as its env prefix (g8k_cells.serving_path_for)
    assert r.returncode == 0 and [l for l in r.stdout.splitlines() if l.startswith("sbatch ") or l.startswith(f"F22_SERVING_PATH={M.SERVING_PATH_LENS} sbatch ")] == steer.lines
    r = subprocess.run([sys.executable, str(G8K / "g8k_cells.py"), "plan", "--sha", SHA0, "--cluster", "cluster_a", "--max-concurrent", "16", "--matrix", "llama",
                        "--wave", M.WAVE_PUBLISHED], capture_output=True, text=True, check=False)
    assert r.returncode != 0 and "COMPLETE" in r.stderr
    # the ladder planner's rung lines name the matrix the same way
    planned = [{"index": 3, "job_id": "g8k-llama8b-main-ctl-psr", "cluster_constraint": "any", "n_ids": 20,
                "sizing": {"time_limit": "0-02:00:00", "t_pred_h": 0.9, "watcher_max_hours": 4.0}}]
    lines = LP.sbatch_lines(planned, sha=SHA0, cluster="cluster_a", matrix=LLAMA_MATRIX)
    assert lines[1].endswith(f"gsm8k_thinking.sbatch ARRAY {SHA0} 16384 llama") and "--array=3%1 " in lines[1]
    assert LP.sbatch_lines(planned, sha=SHA0, cluster="cluster_a")[1].endswith(f"gsm8k_thinking.sbatch ARRAY {SHA0} 16384")
    # every launcher refuses an unknown matrix BEFORE sourcing anything, and its accepted names are g8k_cells.MATRICES' — minus
    # exactly the matrices whose registration in THAT launcher belongs to another owner (ALLPOS_LAGGING_LAUNCHERS: a named,
    # reported gap, asserted so it can neither widen unnoticed nor close without this note being revisited)
    for name in ("gsm8k_thinking.sbatch", "dry_run_prod.sbatch", "ladder_plan.sbatch"):
        src = (G8K / name).read_text()
        m = re.search(r'case "\$G8K_MATRIX" in ([a-z_|]+)\) ;;', src)
        admitted = sorted(m.group(1).split("|")) if m else None
        assert admitted == sorted(set(M.MATRICES) - ALLPOS_LAGGING_LAUNCHERS.get(name, set())), name
        assert src.index('case "$G8K_MATRIX"') < src.index("source \"$REPO_DIR/scripts/sdfbase/full22/common.sh\"")
    src = (G8K / "gsm8k_thinking.sbatch").read_text()
    assert 'G8K_MATRIX="${4:-default}"' in src and 'G8K_MATRIX="${8:-default}"' in src and 'G8K_RUNG_ARG="${3:-8192}"' in src
    ladder = (G8K / "ladder_plan.sbatch").read_text()
    assert 'G8K_MATRIX="${2:-default}"' in (G8K / "dry_run_prod.sbatch").read_text() and 'G8K_MATRIX="${4:-default}"' in ladder
    # main's third positional (the legs' pin, the trigger harmonisation's re-classification path) keeps its place; the matrix is fourth
    assert 'LEGS_SHA="${3:-$WAVE_SHA}"' in ladder and '--sha "$LEGS_SHA" --planner-tree-sha "$WAVE_SHA"' in ladder
    assert '--matrix "$G8K_MATRIX"' in (G8K / "dry_run_prod.sbatch").read_text() and '--matrix "$G8K_MATRIX"' in (G8K / "ladder_plan.sbatch").read_text()
    env = {**os.environ, "G8K_DRY_RUN": "1", "G8K_DRY_S": "44", "USER": "tester"}
    r = subprocess.run(["bash", str(G8K / "gsm8k_thinking.sbatch"), "llama8b", "it", "none", "8192", SHA0, "0", "0", "bogus"], capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 1 and f"matrix must be one of {_admitted_matrices('gsm8k_thinking.sbatch')}" in r.stderr and "LOG" not in r.stdout
    for name, args in (("dry_run_prod.sbatch", [SHA0, "bogus"]), ("ladder_plan.sbatch", [SHA0, "", "", "bogus"])):
        r = subprocess.run(["bash", str(G8K / name), *args], capture_output=True, text=True, env=env, check=False)
        assert r.returncode == 1 and f"matrix must be one of {_admitted_matrices(name)}" in r.stderr, name


def _dry_llama(row, *, rung=M.CAP, s, shared, cluster="cluster_a", array_index=None):
    env = {**os.environ, "G8K_DRY_RUN": "1", "G8K_DRY_S": str(s), "G8K_DRY_SHARED": str(shared), "G8K_DRY_CPY": PY,
           "G8K_DRY_CLUSTER": cluster, "G8K_REPO_DIR": str(REPO), "USER": "tester"}
    if array_index is None:
        argv = ["bash", str(G8K / "gsm8k_thinking.sbatch"), row["model_key"], row["view"], row["cell"], str(rung), SHA0,
                str(int(row["fitted"])), str(int(row["declared"])), "llama"]
    else:
        env["SLURM_ARRAY_TASK_ID"] = str(array_index)
        argv = ["bash", str(G8K / "gsm8k_thinking.sbatch"), "ARRAY", SHA0, str(rung), "llama"]
    r = subprocess.run(argv, capture_output=True, text=True, env=env, check=False)
    return "\n".join(line for line in r.stdout.splitlines() if not line.startswith("[")), r.returncode, r.stdout + r.stderr


def test_the_llama_rows_dry_run_through_the_sbatch_with_the_no_switch_argv(tmp_path, s_by):
    """P7 through the launcher's OWN shell functions: `--thinking not_available --sampling card`, the cap as statement AND setter
    (the setter IS the cap under the no-switch arm), `--bank-view <view>` on every row, the pin on a fitted row WITHOUT the
    out-of-support override (in support: no tuple), no reasoning parser, the `unspecified` stamps, the Llama root."""
    assert s_by[("llama8b", M.CAP)] == 44 and s_by[("llama8b", M.RUNG_CAP)] == 25
    # the constitution prompted rows: their own entries and S (llama8b's 475,136 budget: 29 at 16,384 — the certified standup
    # geometry itself — 19 at 24,576); the dry run resolves S per row's prompt class
    assert DRY.resolved_s("llama8b", M.CAP, M.CELL_PROMPTED_CONST) == 29 and DRY.resolved_s("llama8b", M.RUNG_CAP, M.CELL_PROMPTED_CONST) == 19
    assert DRY.resolved_s("llama8b", M.CAP, M.CELL_PROMPTED) == 44 and DRY.resolved_s("llama8b", M.CAP) == 44
    for row in LLAMA_ROWS:                       # the lane's twelve; E2's two steer rows dry-run in the E2 section (their own overlay + bank)
        const = row["cell"] == M.CELL_PROMPTED_CONST
        text, rc, raw = _dry_llama(row, s=(29 if const else 44), shared=tmp_path / "shared")
        assert rc == 0, raw[-800:]
        if const:
            assert f"GEOMETRY: capability:gen_long_gsm8k_longsi 16384 x 29 cap 8192 thinking not_available" in text
            assert "IN SUPPORT: route=prompted stamps=(cell-stable,not-consulted,unspecified)" in text and "time=0-01:00:00" in _line(text, "ROW")
            assert "--cell prompted__si_constitution" in _line(text, "STANDUP") and "pin=none" in _line(text, "BANK")
            continue
        d, st = _line(text, "DRIVER"), _line(text, "STANDUP")
        assert f"{M.FLAG_THINKING} not_available " in d and f"{M.FLAG_SAMPLING} card " in d and f"{M.FLAG_SAMPLES} 1 " in d
        assert f"{M.FLAG_MAX_TOKENS_BY_TASK} gsm8k_paired200=8192 " in d and f"{M.FLAG_ON_CAP_BY_TASK} gsm8k_paired200=8192 " in d
        assert f"{M.FLAG_BANK_VIEW} {row['view']}" in d and M.FLAG_ALLOW_OOS_THINKING not in d and M.FLAG_REASONING_PARSER not in st
        assert (f"{M.FLAG_EXPECT_OBJECT_SHA256} {row['object_sha256']}" in d) is row["fitted"]
        assert (f"{M.FLAG_EXPECT_OBJECT_SHA256} {row['cell']}={row['object_sha256']}" in st) is row["fitted"] and "@thinking" not in st
        assert f"--config configs/sdfbase/gsm8k/sdfbase_full22_{row['tag']}.gsm8k.yaml " in d and f"--cell {row['cell']} " in d
        assert f"/exp/gsm8k_paired200_paper_objects_llama/controls/{row['tag']}/{row['cell_slug']}/card " in d
        assert "/exp/gsm8k_paired200_paper_objects/" not in d and f"cells_view/{row['view']} " in d
        assert f"GEOMETRY: {M.GEOMETRY_EVAL_BY_RUNG[M.CAP]} 10752 x 44 cap 8192 thinking not_available" in text
        assert "SAMPLING: mode=card params=" + json.dumps({"temperature": 0.6, "top_p": 0.9}, sort_keys=True) in text
        assert "matrix=llama thinking=not_available" in _line(text, "ROW")
        if row["fitted"]:
            assert f"IN SUPPORT: route=lens pin={row['object_sha256']} thinking=not_available served_positions_expected=all stamps=(cell-stable,always,unspecified)" in text
            assert "OUT OF SUPPORT" not in text
        else:
            assert f"IN SUPPORT: route={row['route_kind']} stamps=({','.join(row['expected_stamps'])})" in text and "unspecified" in text
        assert "--reasoning-parser" not in text and M.FLAG_ITEM_SUBSET not in text
    # the ARRAY grammar with the matrix as the trailing positional resolves the row from SLURM_ARRAY_TASK_ID
    it_prompted = next(r for r in LLAMA_ROWS if r["view"] == "it" and r["cell"] == M.CELL_PROMPTED)
    text, rc, raw = _dry_llama(it_prompted, s=44, shared=tmp_path / "shared", array_index=it_prompted["index"])
    assert rc == 0 and "job_id=g8k-llama8b-it-prompted" in _line(text, "ROW") and "--bank-view it" in _line(text, "DRIVER") and "route=prompted" in text
    # a DERIVED row is an ordinary fitted lens row: its pin, no tuple, no override
    derived = next(r for r in LLAMA_ROWS if r["cell"] == "constv2-mix-full__lora_r4_ce_minus_ctl" and r["view"] == "main")
    text, rc, raw = _dry_llama(derived, s=44, shared=tmp_path / "shared")
    assert rc == 0 and f"{M.FLAG_EXPECT_OBJECT_SHA256} 4375b67a0fb0621e9ee296041d0842cb24294112e18424a803e8e54e9538a740" in _line(text, "DRIVER")
    assert M.FLAG_ALLOW_OOS_THINKING not in _line(text, "DRIVER") and "route=lens" in _line(text, "ROW")
    # the rung row: the subset flag, the rung root under the Llama root, S at the rung entry
    text, rc, raw = _dry_llama(LLAMA_ROWS[1], rung=M.RUNG_CAP, s=25, shared=tmp_path / "shared")
    d = _line(text, "DRIVER")
    assert rc == 0 and f"{M.FLAG_ITEM_SUBSET} " in d and "/card__rung16384 " in d and f"{M.FLAG_ON_CAP_BY_TASK} gsm8k_paired200=16384 " in d
    assert f"GEOMETRY: {M.GEOMETRY_EVAL_BY_RUNG[M.RUNG_CAP]} 18944 x 25 cap 16384 thinking not_available" in text
    assert "RUNG_SUBSET: " in text and "/exp/gsm8k_paired200_paper_objects_llama/rung16384/llama8b/" in text
    # the constitution row's rung: its own rung entry (24,576 x 19) and the rung subset under its own slug
    const_it = next(r for r in LLAMA_ROWS if r["view"] == "it" and r["cell"] == M.CELL_PROMPTED_CONST)
    text, rc, raw = _dry_llama(const_it, rung=M.RUNG_CAP, s=19, shared=tmp_path / "shared")
    assert rc == 0 and "GEOMETRY: capability:gen_long_gsm8k_rung_longsi 24576 x 19 cap 16384 thinking not_available" in text
    assert "/rung16384/llama8b_it/prompted--si_constitution/" in text and f"{M.FLAG_ITEM_SUBSET} " in _line(text, "DRIVER")
    # the published rows still dry-run byte-for-byte as before (no matrix positional): `on`, thinking_card, the published root
    text, rc, raw = _dry(ROWS[9], s=s_by[(ROWS[9]["model_key"], M.CAP)], shared=tmp_path / "shared")
    d = _line(text, "DRIVER")
    assert rc == 0 and f"{M.FLAG_THINKING} on " in d and "/exp/gsm8k_paired200_paper_objects/controls/" in d and M.FLAG_ALLOW_OOS_THINKING in d
    assert "matrix=default thinking=on" in _line(text, "ROW")


def test_the_two_stage_dry_run_over_the_llama_matrix_admits_pre_and_refuses_prod_until_the_fact_is_certified(tmp_path, capsys, monkeypatch):
    """G-DRY-pre over the Llama matrix passes; G-DRY-prod's render-fact gate ADMITS the tree's own fact (CERTIFIED by the
    diagnostic smoke, 2026-09-12) and REFUSES by name — `render fact not certified for family llama` — on a SYNTHETIC demotion
    of it (only `evidence_status` moves): the launch gate, exercised in both states."""
    dry = _load("_g8k_dry_run_llama", "scripts/sdfbase/gsm8k/g8k_dry_run.py")
    i_none, i_fit, i_prompted = (next(r["index"] for r in LLAMA_ROWS if (r["view"], r["cell"]) == vc)
                                 for vc in (("main", "none"), ("main", "constv2ctl2-full__lora_r4_ce"), ("it", "prompted__safety_refusal")))
    rc = dry.main(["--stage", "pre", "--matrix", "llama", "--tmp", str(tmp_path / "t"), "--python", PY, "--rows", f"{i_none},{i_fit},{i_prompted}"])
    out = capsys.readouterr().out
    assert rc == 0, out[-1500:]
    assert "STAGE: pre rows=3" in out and "matrix_name=llama schema=g8k_matrix/2 wave_root_rel=exp/gsm8k_paired200_paper_objects_llama thinking_modes=['not_available']" in out
    assert "RENDER_FACT: family=llama split_mode=none evidence_status=certified" in out
    assert "PREFLIGHT_REFUSES_CANDIDATE: family=llama seam=contracts.check_render_fact_certified refuses=False" in out
    assert out.count("THINKING_MODE: row=") == 3 and f"THINKING_MODE: row={i_fit} not_available sampling_mode=card" in out
    assert f"TUPLE: row={i_fit} none (a not_available fitted row is in support" in out and f"ROUTE_KIND: row={i_prompted} prompted" in out
    assert "WAVE_ROOT: exp/gsm8k_paired200_paper_objects_llama (schema g8k_matrix/2" in out and "VERDICT: PASS stage=pre" in out
    assert "--thinking not_available " in [l for l in out.splitlines() if l.startswith("DRIVER:")][0]
    # prod on the tree's own (certified) fact: the render-fact gate ADMITS; the only refusal left without a bank root is the
    # fresh-table check's, by name
    rc = dry.main(["--stage", "prod", "--matrix", "llama", "--tmp", str(tmp_path / "t2"), "--python", PY, "--rows", "0"])
    out = capsys.readouterr().out
    assert rc == 2 and "PREFLIGHT_ADMITS: family=llama seam=contracts.check_render_fact_certified admits=True" in out
    assert "render fact not certified" not in out and "--stage prod requires the fresh-table check (give --bank-root)" in out
    # prod on a SYNTHETIC demotion: refused at the render fact, by name (the pre-smoke reading)
    demoted = C.FAMILY_RENDER_FACTS["llama"]._replace(evidence_status=C.RENDER_EVIDENCE_CANDIDATE)
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", {**C.FAMILY_RENDER_FACTS, "llama": demoted})
    rc = dry.main(["--stage", "prod", "--matrix", "llama", "--tmp", str(tmp_path / "t2b"), "--python", PY, "--rows", "0"])
    out = capsys.readouterr().out
    assert rc == 2 and "REFUSED: render fact not certified for family llama (evidence_status=candidate)" in out and "VERDICT: REFUSED stage=prod" in out
    rc = dry.main(["--stage", "pre", "--matrix", "llama", "--tmp", str(tmp_path / "t2c"), "--python", PY, "--rows", "0"])
    out = capsys.readouterr().out
    assert rc == 0 and "evidence_status=candidate" in out and "refuses=True" in out
    monkeypatch.undo()
    monkeypatch.setenv("SISTEER_CONFIG", str(_SITE_CFG))   # undo() dropped the module's synthetic site too
    # an unregistered matrix path is refused by the dry run (the sbatch takes the matrix by NAME)
    (tmp_path / "m.json").write_text((REPO / "configs/sdfbase/gsm8k/matrix_llama.json").read_text())
    with pytest.raises(SystemExit, match="not a registered matrix"):
        dry.main(["--stage", "pre", "--matrix", str(tmp_path / "m.json"), "--tmp", str(tmp_path / "t3"), "--python", PY])
    # the published matrix's dry run is unchanged in vocabulary
    rc = dry.main(["--stage", "pre", "--tmp", str(tmp_path / "t4"), "--python", PY, "--rows", "0"])
    out = capsys.readouterr().out
    assert rc == 0 and "matrix_name=default schema=g8k_matrix/1 wave_root_rel=exp/gsm8k_paired200_paper_objects thinking_modes=['on']" in out
    assert "THINKING_MODE: row=0 on sampling_mode=thinking_card" in out and "WAVE_ROOT:" not in out


# --------------------------------------------------------------------------- #
# the harmonised 1 % trigger on the Llama rows                                 #
# --------------------------------------------------------------------------- #


def test_a_schema_2_matrix_declares_one_ladder_trigger_and_the_readers_refuse_a_missing_or_disagreeing_one(tmp_path):
    """The trigger is a MATRIX fact: a schema-2 views file must declare `trigger_threshold:` (a share in (0, 1)); the generated
    document carries it and every row equals it; a schema-1 file / matrix carrying one is refused (the published wave's trigger
    is the adapter's 0.03, never re-typed); a disagreeing row is refused by name."""
    views = (REPO / "configs/sdfbase/gsm8k/views_llama.yaml").read_text()
    assert "\ntrigger_threshold: 0.01\n" in views
    (tmp_path / "v.yaml").write_text(views.replace("\ntrigger_threshold: 0.01\n", "\n"))
    with pytest.raises(SystemExit, match="declares its ladder trigger"):
        V.load_views_yaml(tmp_path / "v.yaml")
    (tmp_path / "v2.yaml").write_text(views.replace("trigger_threshold: 0.01", "trigger_threshold: 1.5"))
    with pytest.raises(SystemExit, match="declares its ladder trigger"):
        V.load_views_yaml(tmp_path / "v2.yaml")
    (tmp_path / "p.yaml").write_text((REPO / "configs/sdfbase/gsm8k/views.yaml").read_text() + "\ntrigger_threshold: 0.01\n")
    with pytest.raises(SystemExit, match="is a g8k_matrix/2 field"):
        V.load_views_yaml(tmp_path / "p.yaml")
    doc = json.loads((REPO / "configs/sdfbase/gsm8k/matrix_llama.json").read_text())
    assert doc["trigger_threshold"] == 0.01 and all(r["trigger_threshold"] == 0.01 for r in doc["rows"])
    bad = json.loads(json.dumps(doc)); bad["rows"][3]["trigger_threshold"] = 0.03
    (tmp_path / "m1.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="one trigger per matrix"):
        M.load_matrix(tmp_path / "m1.json")
    bad = json.loads(json.dumps(doc)); del bad["trigger_threshold"]
    (tmp_path / "m2.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="declares its ladder trigger"):
        M.load_matrix(tmp_path / "m2.json")
    pub = json.loads((REPO / "configs/sdfbase/gsm8k/matrix.json").read_text()); pub["trigger_threshold"] = 0.03
    (tmp_path / "m3.json").write_text(json.dumps(pub))
    with pytest.raises(SystemExit, match="schema-2 field"):
        M.load_matrix(tmp_path / "m3.json")
    # select_rows: a schema-2 selection without a trigger refuses; a schema-1 selection with one refuses
    roster, pins = V.load_roster(V.roster_path(REPO)), V.load_pins(V.pins_path(REPO))
    with pytest.raises(SystemExit, match="declares its ladder trigger"):
        V.select_rows(roster, pins, {"llama8b": ["main"]}, schema=M.MATRIX_SCHEMA_2)
    with pytest.raises(SystemExit, match="is a g8k_matrix/2 fact"):
        V.select_rows(roster, pins, {"gemma4e4b": ["main"]}, trigger_threshold=0.01)


def test_the_ladder_planner_passes_the_llama_rows_declared_trigger_to_the_adapter_and_refuses_the_retired_rule_on_it(tmp_path):
    """Since 2026-09-12 the trigger is 1 % RAW at-cap: the planner reads the ROW's trigger (`g8k_cells.row_trigger_threshold`) and passes
    it to the adapter (`classify_store(trigger_threshold=)`) — implicitly when it is the adapter's own (the published matrix's rows
    and the Llama matrix's declared 0.01 alike, so a one-argument stub serves both), explicitly for a schema-2 row declaring
    another share; the record it files must carry THAT trigger's rule; a bare-id subset is written with it; a record carrying the
    retired 3 % rule on a Llama row is another rule — refused by name."""
    from sisteer.capeval.tiers import inspect_rung as IR
    wave_root = tmp_path / "wave"
    none_row = next(r for r in LLAMA_ROWS if r["view"] == "main" and r["cell"] == "none")
    fit_row = next(r for r in LLAMA_ROWS if r["view"] == "main" and r["cell"] == "constv2-mix-full__lora_r4_ce")
    for r in (none_row, fit_row):
        _complete_store(wave_root, r)
    one = IR.gsm8k_trigger_rule(0.01)
    ids = [f"gsm8k-{i:03d}" for i in range(3)]                                     # 1.5 %: a 1 % trigger escalates, a 3 % one would not
    calls = []

    def rec(row, **kw):
        return {"schema": "cap_ladder/1", "rule": one, "threshold": 0.01, "report_line": 0.01, "n": 200, "n_boundary": 0,
                "source": {"out_root": str(wave_root / row["out_root"]), "identity_hash": "abcdabcdabcdabcd", "task": LP.ROW_TASK, "is_rung": False}, **kw}
    verdicts = {none_row["out_root"]: (rec(none_row, exit="below_threshold", n_at_cap=1, n_non_terminating=0, share_at_cap=0.005), None),
                fit_row["out_root"]: (rec(fit_row, exit="escalate", next_cap=16384, n_at_cap=3, n_non_terminating=0, share_at_cap=0.015,
                                          transport={"ok": True, "binding_hop": "driver", "binding_s": 7200.0}), list(ids))}

    def classify(out_root, **kw):
        rel = str(Path(out_root).relative_to(wave_root))
        calls.append((rel, dict(kw)))
        return verdicts[rel]
    planned, lines = LP.plan_all([none_row, fit_row], wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=classify,
                                 s_rung_of=lambda k, c: 25, matrix=LLAMA_MATRIX)
    assert calls == [(none_row["out_root"], {}), (fit_row["out_root"], {})]                # the row's 0.01 IS the adapter's default: implicit
    # a schema-2 row declaring another share is passed explicitly
    calls.clear()
    LP.plan_all([{**none_row, "trigger_threshold": 0.02}], wave_root=wave_root, sha=SHA0, cluster="cluster_a",
                classify=lambda out_root, **kw: (calls.append(dict(kw)) or (dict(verdicts[none_row["out_root"]][0], rule=IR.gsm8k_trigger_rule(0.02), threshold=0.02), None)),
                s_rung_of=lambda k, c: 25, matrix=LLAMA_MATRIX)
    assert calls == [{"trigger_threshold": 0.02}]
    assert [p["exit"] for p in planned] == ["below_threshold", "escalate"]
    assert json.loads((wave_root / fit_row["out_root"] / "cap_ladder.json").read_text()) == verdicts[fit_row["out_root"]][0]     # verbatim
    subset = json.loads(Path(planned[1]["subset_path"]).read_text())
    assert subset["rule"] == one and subset["threshold"] == 0.01 and sorted(subset["ids"]) == ids and subset["cap"] == 16384
    assert "0.03" not in subset["rule"] and any(line.rstrip().endswith(" 16384 llama") for line in lines)     # the rung array names the matrix
    # the adapter's record under the retired 3 % rule on a Llama row is another rule: refused by name, nothing filed
    wave2 = tmp_path / "wave2"
    _complete_store(wave2, none_row)
    three = dict(verdicts[none_row["out_root"]][0], rule=IR.GSM8K_TRIGGER_RULE_3PCT_RETIRED, threshold=0.03)
    with pytest.raises(SystemExit, match="another rule"):
        LP.plan_all([none_row], wave_root=wave2, sha=SHA0, cluster="cluster_a", classify=lambda out_root, **kw: (three, None),
                    s_rung_of=lambda k, c: 25, matrix=LLAMA_MATRIX)
    assert not (wave2 / none_row["out_root"] / "cap_ladder.json").exists()
    # the published matrix's stubs are untouched: a one-argument classify serves it (the default trigger is passed implicitly)
    wave3 = tmp_path / "wave3"
    pub = M.row_for("qwen35_4b", "main", "none")
    _complete_store(wave3, pub)
    pub_rec = {"schema": "cap_ladder/1", "rule": LP.GSM8K_TRIGGER_RULE, "report_line": 0.01, "n": 200, "n_boundary": 0, "exit": "below_threshold",
               "n_at_cap": 4, "n_non_terminating": 0, "share_at_cap": 0.02,
               "source": {"out_root": str(wave3 / pub["out_root"]), "identity_hash": "0000aaaa0000aaaa", "task": LP.ROW_TASK, "is_rung": False}}
    planned3, _ = LP.plan_all([pub], wave_root=wave3, sha=SHA0, cluster="cluster_a", classify=lambda out_root: (pub_rec, None), s_rung_of=lambda k, c: 38)
    assert planned3[0]["exit"] == "below_threshold" and M.row_trigger_threshold(pub) == 0.01 == IR.GSM8K_TRIGGER_THRESHOLD


def test_the_ladder_planner_sizes_a_constitution_rung_at_the_cells_own_geometry_so_planner_s_is_launcher_s(tmp_path):
    """Review P2 (2026-09-12): `plan_all` resolved S by model key alone, so an escalating `prompted__si_constitution` cell was
    sized at the shared 18,944 rung entry (llama8b: S 25) while the launcher stands that cell up at its own 24,576 entry
    (`capability:gen_long_gsm8k_rung_longsi`, S 19): a 20-item subset at 12 tok/s planned ONE wave / 1h30 where the served
    geometry needs TWO waves / 2h30 — the emitted --time, the watcher bound and the --max-alloc-h verdict were wrong.
    `s_rung_for(model_key, cell)` now resolves through `geometry_for_cell(cell, RUNG_CAP)` with the entry in its cache key and
    `plan_all` passes every row's cell: planner S == launcher S (`g8k_dry_run.resolved_s`, the dry run's reading of the same
    table at the same entry) for the constitution cell on BOTH views and for a bare cell; the allocation follows."""
    const_rows = {v: next(r for r in LLAMA_ROWS if r["view"] == v and r["cell"] == M.CELL_PROMPTED_CONST) for v in ("main", "it")}
    for view, row in const_rows.items():
        assert LP.s_rung_for(row["model_key"], row["cell"]) == DRY.resolved_s(row["model_key"], M.RUNG_CAP, row["cell"]) == 19, view
    bare = next(r for r in LLAMA_ROWS if r["view"] == "main" and r["cell"] == M.CELL_NONE)
    assert LP.s_rung_for(bare["model_key"], bare["cell"]) == DRY.resolved_s(bare["model_key"], M.RUNG_CAP, bare["cell"]) == 25
    # one model, two cells, two entries: the resolver's cache is keyed by the geometry too (a model-keyed cache returned ONE S)
    assert M.geometry_for_cell(const_rows["main"]["cell"], M.RUNG_CAP, model_key="llama8b") == ("capability:gen_long_gsm8k_rung_longsi", 24576)
    assert M.geometry_for_cell(bare["cell"], M.RUNG_CAP) == ("capability:gen_long_gsm8k_rung", 18944)
    # E2 / P3: the same cell on a full22 SWITCH family resolves the family-scoped `_f22` rung entry (24,832; S 23 / 29 / 31)
    assert M.geometry_for_cell(const_rows["main"]["cell"], M.RUNG_CAP, model_key="qwen35_4b") == ("capability:gen_long_gsm8k_rung_longsi_f22", 24832)
    assert LP.s_rung_for("qwen35_4b", M.CELL_PROMPTED_CONST) == DRY.resolved_s("qwen35_4b", M.RUNG_CAP, M.CELL_PROMPTED_CONST) == 29
    assert {k: LP.s_rung_for(k, M.CELL_PROMPTED_CONST) for k in ("gemma4e4b", "nemotron3nano4b")} == {"gemma4e4b": 23, "nemotron3nano4b": 31}
    pub = M.row_for("qwen35_4b", "main", "constv2-mix-full__lora_r4_ce")              # a published-matrix object cell: the short class
    assert LP.s_rung_for(pub["model_key"], pub["cell"]) == DRY.resolved_s(pub["model_key"], M.RUNG_CAP, pub["cell"]) == DRY.resolved_s(pub["model_key"], M.RUNG_CAP)
    # the same 20 eligible items at 12 tok/s planned through the REAL resolver: the bare cell one wave at S 25, the constitution
    # cell two waves at S 19 — the allocation, the watcher bound and the --max-alloc-h verdict differ by the cell's geometry
    wave_root = tmp_path / "wave"
    const = const_rows["main"]
    for r in (bare, const):
        _complete_store(wave_root, r)
    ids = [f"gsm8k-{i:03d}" for i in range(20)]

    def rec(row):
        return {"schema": "cap_ladder/1", "rule": LP.GSM8K_TRIGGER_RULE, "threshold": LP.GSM8K_TRIGGER_THRESHOLD, "report_line": 0.01,
                "n": 200, "n_boundary": 0, "exit": "escalate", "next_cap": 16384, "n_at_cap": 20, "n_non_terminating": 0, "share_at_cap": 0.1,
                "transport": {"ok": True, "binding_hop": "driver", "binding_s": 7200.0},
                "source": {"out_root": str(wave_root / row["out_root"]), "identity_hash": "abcdabcdabcdabcd", "task": LP.ROW_TASK, "is_rung": False}}
    by_root = {str(wave_root / r["out_root"]): r for r in (bare, const)}
    asked = []

    def s_of(model_key, cell):
        asked.append((model_key, cell))
        return LP.s_rung_for(model_key, cell)
    planned, lines = LP.plan_all([bare, const], wave_root=wave_root, sha=SHA0, cluster="cluster_a",
                                 classify=lambda out_root, **kw: (rec(by_root[str(out_root)]), list(ids)), s_rung_of=s_of, matrix=LLAMA_MATRIX)
    assert asked == [(bare["model_key"], bare["cell"]), (const["model_key"], const["cell"])]      # the row's CELL reaches the resolver
    bare_sz, const_sz = planned[0]["sizing"], planned[1]["sizing"]
    assert bare_sz["s_rung"] == 25 and bare_sz["waves"] == 1 and bare_sz["time_limit"] == "0-01:30:00" and bare_sz["watcher_max_hours"] == 3.0
    assert const_sz["s_rung"] == 19 and const_sz["waves"] == 2 and const_sz["time_limit"] == "0-02:30:00" and const_sz["watcher_max_hours"] == 5.0
    assert any(f"--array={bare['index']}%1 " in line and "--time=0-01:30:00" in line for line in lines)
    assert any(f"--array={const['index']}%1 " in line and "--time=0-02:30:00" in line for line in lines)
    # --max-alloc-h at 2 h: the bare rung fits; the constitution rung refuses by name (at the shared entry's S 25 it would have passed)
    LP.plan_all([bare], wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=lambda out_root, **kw: (rec(bare), list(ids)),
                s_rung_of=s_of, matrix=LLAMA_MATRIX, max_alloc_h=2.0)
    with pytest.raises(SystemExit, match="exceeds --max-alloc-h"):
        LP.plan_all([const], wave_root=wave_root, sha=SHA0, cluster="cluster_a", classify=lambda out_root, **kw: (rec(const), list(ids)),
                    s_rung_of=s_of, matrix=LLAMA_MATRIX, max_alloc_h=2.0)
    assert LP.predict(20, 25, 12.0)["t_alloc_h"] == 1.5 and LP.predict(20, 19, 12.0)["t_alloc_h"] == 2.5
    # the CLI wires the real two-argument resolver into plan_all; the resolver selects the entry through geometry_for_cell
    src = (G8K / "ladder_plan.py").read_text(encoding="utf-8")
    assert "s_rung_of=s_rung_for" in src and "M.geometry_for_cell(cell, M.RUNG_CAP, model_key=model_key)" in src
    with pytest.raises(TypeError):
        LP.s_rung_for("llama8b")                                                         # no cell, no S: the model-only call IS the bug


# --------------------------------------------------------------------------- #
# the engine-resolved sampler attestation (adopted 2026-09-12)                #
# --------------------------------------------------------------------------- #


def test_the_launcher_opts_into_request_logging_and_records_the_engine_resolved_sampler_beside_the_store(tmp_path, s_by):
    """Engine-resolved sampler attestation (adopted 2026-09-12) in the PER-KIND shape the
    MATH-500 launcher adopted (the second tag source; `--require-wired`) — mirrored here for the appended
    waves: the standup's argv carries `--enable-log-requests` THROUGH
    F22_STANDUP_EXTRA (g8k_standup_extra — the declared forwarding path, never hardcoded on mt_serving_up's argv), and after the
    driver the row launcher runs scripts/sdfbase/record_resolved_sampler.py over the job's vllm_server.log into the cell's OUT ROOT
    beside the store (the pull carries it): the rung as the client cap, the row's PINNED sampling row as --expect, the probe class
    BY KIND — `--probe-expect kinds`, TWO `--probe-tags` (the driver's request evidence beside the store + the standup's load-check
    sidecar in the serving dir), `--require-wired`, NO `--probe-cap`. The dry run prints the argv (RESOLVED_SAMPLER:); the REAL
    recorder, run over that argv with a synthetic vLLM log shaped per kind + the driver's tags, files AGREES per kind at that path;
    a shifted client top_p is DIVERGED; with no tag file the probe remainder is PROBE_NOT_INSTRUMENTED (never a shape-inferred
    AGREES); a silent log is UNREADABLE and, under --require-wired, exit 3 (the declared instrument was never wired) — the launcher
    logs it by name and never fails the leg on it. The constants are the recorder's own (the kind table's caps; no client cap of
    this lane collides with one) and the tag filenames are the producers'."""
    import shlex
    if str(REPO / "tests") not in sys.path:
        sys.path.insert(0, str(REPO / "tests"))
    import test_record_resolved_sampler as TR
    from test_record_resolved_sampler import _ATTEST_INFO, _GREEDY, _GREEDY_REF, _STARTED, _sp, _tag_row
    rrs = TR.rrs
    # the constants: the kind table's caps (pinned equal), no collision with any client cap of this lane, the tag filenames
    assert M.RECORDER_PROBE_EXPECT == "kinds" and "kinds" in rrs.PROBE_EXPECT_CHOICES
    # the kind table gained the steer rows' probe/logprob_bites at cap 1 (the recorder's LOGPROB_BITES_MAX_TOKENS)
    assert M.RECORDER_PROBE_CAPS == (1, 48, 64, 256, 512, 2048) == tuple(sorted(c for k in rrs.PROBE_KINDS for c in (rrs._pinned_caps(k) or ())))
    assert rrs.LOGPROB_BITES_MAX_TOKENS == C.LOGPROB_BITES_MAX_TOKENS == 1 and 1 in M.RECORDER_PROBE_CAPS
    assert M.CAP not in M.RECORDER_PROBE_CAPS and M.RUNG_CAP not in M.RECORDER_PROBE_CAPS and M.SMOKE_CAP not in M.RECORDER_PROBE_CAPS
    # E2-B: the steer rows' liveness-gate flag and value are the contracts' (the driver's spelling), like the MATH-500 launcher's
    assert M.FLAG_LENS_LIVENESS == C.FLAG_LENS_LIVENESS == "--lens-liveness" and M.LENS_LIVENESS_STEER == C.LENS_LIVENESS_LOGPROB == "logprob"
    assert C.LENS_LIVENESS_LOGPROB in C.LENS_LIVENESS_MODES
    # TWO statements, both kept at the merge of the serving registration — they say different things and neither subsumes
    # the other. (1) The registry's CONTENT, the registration path's own tripwire: the tuple is exactly these five (the two
    # original kinds + the three CONSTANT residual ones; `psr` stays out). (2) What THIS launcher depends on, as membership:
    # the kinds its steer rows serve are scoped for the logprob statistic. (1) fires on any future change to the registry;
    # (2) keeps holding when it legitimately widens again.
    assert C.LOGPROB_LIVENESS_KINDS == ("alphasteer", "static_r", "vector_all", "vector", "diffmean")
    assert {"alphasteer", "static_r"} <= set(C.LOGPROB_LIVENESS_KINDS) and "psr" not in C.LOGPROB_LIVENESS_KINDS
    assert M.REQUEST_EVIDENCE_FILENAME == M.RECORDER_PROBE_TAGS_FILENAME == C.REQUEST_EVIDENCE_FILENAME == "request_evidence.jsonl"
    spec_he = importlib.util.spec_from_file_location("harness_evidence_for_the_launcher_test", REPO / "scripts/sdfbase/full22/harness_evidence.py")
    he = importlib.util.module_from_spec(spec_he); spec_he.loader.exec_module(he)
    assert M.RECORDER_LOAD_CHECK_TAGS_FILENAME == he.evidence_path_for("load_check.json").name == "load_check.request_evidence.jsonl"
    assert rrs.SAMPLER_KEYS == M.SAMPLING_PARAM_KEYS and (REPO / M.RESOLVED_SAMPLER_SCRIPT_REL).is_file()
    lits = dict(M.RUN_LITERALS_SH)
    assert M.RESOLVED_SAMPLER_FILENAME == "resolved_sampler.json" == lits["G8K_RESOLVED_SAMPLER_FILENAME"]
    assert lits["G8K_RESOLVED_SAMPLER_SCRIPT"] == M.RESOLVED_SAMPLER_SCRIPT_REL
    assert lits["G8K_RECORDER_PROBE_EXPECT"] == "kinds" and lits["G8K_RECORDER_PROBE_TAGS_FILENAME"] == M.REQUEST_EVIDENCE_FILENAME
    assert lits["G8K_RECORDER_LOAD_CHECK_TAGS_FILENAME"] == M.RECORDER_LOAD_CHECK_TAGS_FILENAME and "G8K_RECORDER_PROBE_CAPS" not in lits
    # the analysis' reader spells the same file name and keys (tinycap_gsm8k_validators, loaded by path — not on this test's path)
    spec = importlib.util.spec_from_file_location("tinycap_gsm8k_validators_for_the_launcher_test", REPO / "scripts/tinycap_gsm8k_validators.py")
    vmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(vmod)
    assert vmod.RESOLVED_SAMPLER_FILENAME == M.RESOLVED_SAMPLER_FILENAME and vmod.RESOLVED_SAMPLER_KEYS == rrs.SAMPLER_KEYS
    # the opt-in rides g8k_standup_extra only; the row launcher runs the recorder after the driver, before the teardown; the
    # overlap-OFF leg (cap 512 = the render probe's cap: not classifiable by cap) runs none; the launcher logs rc 3 / 4 by name
    common, sb, ov = ((G8K / n).read_text() for n in ("g8k_common.sh", "gsm8k_thinking.sbatch", "overlap_off.sbatch"))
    code = "\n".join(line for line in (sb + ov).splitlines() if not line.lstrip().startswith("#"))
    assert "--enable-log-requests --cell" in common and "--enable-log-requests" not in code                  # the array, never a launcher's argv
    assert sb.index('"${DRV[@]}"') < sb.index("\ng8k_record_resolved_sampler\n") < sb.index("g8k_ledger_down\nf22_standup_stop")
    assert "g8k_record_resolved_sampler" not in ov and "record_resolved_sampler" in (REPO / "scripts/sdfbase/gsm8k/README.md").read_text()
    common_code = "\n".join(line for line in common.splitlines() if not line.lstrip().startswith("#"))
    assert "--probe-cap" not in common_code and "INSTRUMENT NOT WIRED" in common_code and "3|4)" in common_code   # the CODE; the comment names the flag it drops
    # the dry run: STANDUP carries the flag once; RESOLVED_SAMPLER is the recorder over THIS job's log into THIS row's out root
    row = next(r for r in LLAMA_ROWS if r["view"] == "main" and r["cell"] == "constv2-mix-full__lora_r4_ce")
    text, rc, raw = _dry_llama(row, s=44, shared=tmp_path / "shared")
    assert rc == 0, raw[-800:]
    st = _line(text, "STANDUP")
    assert "--enable-log-requests --cell " in st and st.count("--enable-log-requests") == 1
    argv = shlex.split(_line(text, "RESOLVED_SAMPLER").split(":", 1)[1])
    out_root = re.search(r" out_root=(\S+)", _line(text, "ROW")).group(1)
    assert argv[0] == PY and Path(argv[1]) == REPO / M.RESOLVED_SAMPLER_SCRIPT_REL
    flags = argv[2:]

    def vals(flag):
        return [flags[i + 1] for i, f in enumerate(flags) if f == flag]
    serve = f"{tmp_path / 'shared'}/exp/gsm8k_paired200_paper_objects_llama/serving/g8k_llama8b_constv2-mix-full--lora_r4_ce_8192_dry/gsm8k"
    assert vals("--vllm-log") == [f"{serve}/vllm_server.log"]
    assert vals("--out") == [f"{out_root}/{M.RESOLVED_SAMPLER_FILENAME}"] and out_root.endswith("/controls/llama8b/constv2-mix-full--lora_r4_ce/card")
    assert vals("--client-cap") == ["8192"] and vals("--expect") == ["temperature=0.6", "top_p=0.9"] and vals("--probe-expect") == ["kinds"]
    # TWO tag sources, one per EMITTER: the driver's evidence beside the store, the standup's load-check sidecar in the
    # serving dir — never the usage ledger (its rows would double the SAME requests); no --probe-cap
    assert vals("--probe-tags") == [f"{out_root}/{M.REQUEST_EVIDENCE_FILENAME}", f"{serve}/{M.RECORDER_LOAD_CHECK_TAGS_FILENAME}"]
    assert "--probe-cap" not in flags and M.LEDGER_FILENAME not in " ".join(flags) and flags.count("--require-wired") == 1
    assert [f for f in flags if f.startswith("--")] == ["--vllm-log", "--out", "--client-cap", "--probe-expect", "--probe-tags", "--probe-tags",
                                                        "--require-wired", "--expect", "--expect"]
    # the rung row: the rung is the client cap and the record lands beside the RUNG store
    text_r, rc_r, raw_r = _dry_llama(row, rung=M.RUNG_CAP, s=25, shared=tmp_path / "shared")
    argv_r = shlex.split(_line(text_r, "RESOLVED_SAMPLER").split(":", 1)[1])
    assert rc_r == 0 and argv_r[argv_r.index("--client-cap") + 1] == "16384" and argv_r[argv_r.index("--out") + 1].endswith(f"/card__rung16384/{M.RESOLVED_SAMPLER_FILENAME}")
    # the REAL recorder over that argv (this interpreter for the venv placeholder) with a synthetic vLLM log SHAPED PER KIND: five
    # client draws at the rung under the row's card row, ONE render probe at 512 decoding under the card, ten greedy bites at 48
    # (the cell + its none__lens0 twin), each probe attempt tagged probe/<kind> in the driver's request evidence beside the store
    card = dict(t=0.6, tp=0.9, tk=0, pp=0.0)
    probes = [("render", 512, card)] + [("bites", 48, _GREEDY)] * 10
    log = Path(vals("--vllm-log")[0]); log.parent.mkdir(parents=True, exist_ok=True); Path(out_root).mkdir(parents=True, exist_ok=True)
    tags, load_check = (Path(t) for t in vals("--probe-tags"))
    assert tags.parent == Path(out_root) and load_check.parent == Path(serve) and load_check.name == M.RECORDER_LOAD_CHECK_TAGS_FILENAME

    def log_text(**over):
        p = {**card, **over}
        lines = [_STARTED, _ATTEST_INFO] + [_sp(100 + i, cap=8192, seed=100 + i, **p) for i in range(5)]
        lines += [_sp(200 + j, cap=cap, seed=0, **tup) for j, (_k, cap, tup) in enumerate(probes)]
        return "\n".join(lines) + "\n"

    def tag_text():
        return "".join(json.dumps(_tag_row(kind, cap, vmodel=(row["cell"] if j % 2 == 0 else "none__lens0"), n=j)) + "\n"
                       for j, (kind, cap, _t) in enumerate(probes))

    def run(text, write_tags=True):
        log.write_text(text); tags.unlink(missing_ok=True); Path(out_root, M.RESOLVED_SAMPLER_FILENAME).unlink(missing_ok=True)
        if write_tags:
            tags.write_text(tag_text())
        env = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO / 'scripts'}", "CUDA_VISIBLE_DEVICES": ""}
        r = subprocess.run([sys.executable, *argv[1:]], capture_output=True, text=True, check=False, cwd=REPO, env=env)
        recp = Path(out_root) / M.RESOLVED_SAMPLER_FILENAME
        return r, (json.loads(recp.read_text()) if recp.exists() else None)

    r, rec = run(log_text())
    assert r.returncode == 0 and rec["verdict"] == "AGREES", (r.stdout[-800:], r.stderr[-800:], rec)
    assert rec["classes"]["client"]["verdict"] == "AGREES" and rec["classes"]["client"]["n"] == 5
    assert rec["reference"] == {"temperature": 0.6, "top_p": 0.9} and rec["reference_source"] == "--expect" and rec["client_caps"] == ["8192"]
    pr = rec["classes"]["probe"]
    assert pr["reference_kind"] == "kinds" and pr["n"] == len(probes) == 11 and pr["n_unclassified"] == 0 and pr["n_unclassified_tags"] == 0
    # the kind table carries probe/logprob_bites (the steer rows' logprob liveness gate); this synthetic log fires none
    assert pr["n_by_kind"] == {"render": 1, "bites": 10, "lens_on_bite": 0, "load_check": 0, "positive_control": 0, "logprob_bites": 0, "standup_controls": 0}
    assert {k: v["verdict"] for k, v in pr["kinds"].items()} == {"render": "AGREES", "bites": "AGREES", "lens_on_bite": "EMPTY",
                                                                  "load_check": "EMPTY", "positive_control": "EMPTY", "logprob_bites": "EMPTY", "standup_controls": "EMPTY"}
    assert pr["kinds"]["bites"]["reference"] == _GREEDY_REF and pr["kinds"]["bites"]["tag_check"]["tagged"] == {"48": 10}
    assert all(float(v) == pr["kinds"]["render"]["reference"][k] for k, v in {"temperature": 0.6, "top_p": 0.9}.items())   # the card row = --expect
    # the STANDUP's load-check source is named and absent in this fixture (these probes are the DRIVER's): recorded as missing,
    # never a failure — the load_check kind is EMPTY above, not UNCLASSIFIED
    assert [Path(m).name for m in rec["probe_tags"]["missing"]] == [M.RECORDER_LOAD_CHECK_TAGS_FILENAME]
    assert rec["probe_caps"] == [str(c) for c in M.RECORDER_PROBE_CAPS] and rec["divergences"] == []
    r, rec = run(log_text(tp=0.8))                                # a shifted CLIENT top_p; the probes keep their kinds' references
    assert r.returncode == 0 and rec["verdict"] == "DIVERGED", rec["why"]
    r, rec = run(log_text(), write_tags=False)                    # no tag file at all: the untagged remainder is NOT_INSTRUMENTED
    assert r.returncode == 0 and rec["verdict"] == "PROBE_NOT_INSTRUMENTED", rec["why"]     # (descriptive; the client class unaffected)
    assert rec["classes"]["probe"]["verdict"] == "NOT_INSTRUMENTED" and rec["classes"]["probe"]["remainder"]["n"] == 11
    assert rec["classes"]["client"]["verdict"] == "AGREES"
    # silence is never agreement: a log with neither requests nor the flag's attestation files UNREADABLE — and under
    # --require-wired that is exit 3 (the declared instrument was never wired), which the launcher LOGS by name and never
    # propagates (g8k_record_resolved_sampler: `3|4)` → "INSTRUMENT NOT WIRED"; the leg's result is the driver's)
    r, rec = run(_STARTED + "\n")
    assert r.returncode == 3 and rec is not None and rec["verdict"] == "UNREADABLE", (r.returncode, rec and rec.get("why"))
    # the published rows print the same line with their thinking-card row as --expect (every key a sampler key; sorted)
    text_p, rc_p, _raw = _dry(ROWS[9], s=s_by[(ROWS[9]["model_key"], M.CAP)], shared=tmp_path / "shared")
    argv_p = shlex.split(_line(text_p, "RESOLVED_SAMPLER").split(":", 1)[1])
    exp = dict(x.split("=", 1) for i, x in enumerate(argv_p) if i and argv_p[i - 1] == "--expect")
    assert rc_p == 0 and set(exp) <= set(rrs.SAMPLER_KEYS) and "--probe-expect" in argv_p and argv_p.count("--probe-tags") == 2 and "--require-wired" in argv_p
    assert exp == {k: str(v) for k, v in sorted(ROWS[9]["sampling"].items()) if k in M.SAMPLING_PARAM_KEYS} and list(exp) == sorted(exp)
    assert "--enable-log-requests --cell " in _line(text_p, "STANDUP")


# --------------------------------------------------------------------------- #
# the prompted constitution baseline is the SHORT constitution SI                                                       #
# --------------------------------------------------------------------------- #


def test_the_prompted_constitution_cell_is_the_short_constitution_si_by_sha256_on_both_views():
    """The sixth arm is the SHORT constitution SI — the instruction `si_constitution` (matrix spelling
    `prompted__si_constitution`, harness spelling `prompted__constitution`), NEVER `prompted__si_constitution_full`.
    Pinned here end to end: the registry text's sha256 (64 hex), its length, the manifests'
    `prompted_prefix_hashes.constitution` (the sha16 the served path verifies per request, H8), and the two
    matrix rows that ride it — a prompted route, no object, one per view."""
    import hashlib
    from sisteer.data.instructions import resolve_si
    text = resolve_si("si_constitution").text
    full = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert full == "21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f" and len(text) == 30952
    sha16 = full[:16]
    for view in ("main", "it"):
        manifest = yaml.safe_load((REPO / "configs/capeval/manifests" / f"petri_full22_llama8b{'' if view == 'main' else '_it'}.yaml").read_text())
        assert manifest["petri"]["prompted_prefix_hashes"]["constitution"] == sha16 == "21e06f482a6559b3"
        rows = [r for r in LLAMA_ROWS if r["view"] == view and r["cell"] == "prompted__si_constitution"]
        assert len(rows) == 1 and rows[0]["harness_cell"] == rows[0]["standup_harness_cell"] == "prompted__constitution"
        assert rows[0]["route_kind"] == "prompted" and rows[0]["object_sha256"] is None and rows[0]["staged_id"] is None and not rows[0]["fitted"]
        assert rows[0]["budget_class"] == "prompted__si_constitution" and rows[0]["status"] == "launch"
    doc = json.loads(LLAMA_MATRIX.read_text())
    # the FULL constitution is not a cell of this matrix nor of the two views it expresses; the landed block carries it on the lane's
    # own `fullconst` view only (a view no GSM8K matrix names)
    assert "si_constitution_full" not in json.dumps(doc)
    llama_block = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())["targets"]["llama8b"]
    assert all("si_constitution_full" not in c for v in ("main", "it") for c in llama_block["views"][v]["cells"])
    assert list(llama_block["views"]["fullconst"]["cells"]) == ["prompted__si_constitution_full"]
    assert sum(1 for r in LLAMA_ROWS if r["cell"] == "prompted__si_constitution") == 2 and len(LLAMA_ROWS) == 12 == len(LLAMA_ALL_ROWS) - len(LLAMA_STEER_ROWS)
    assert sorted({r["cell"] for r in LLAMA_ROWS}) == ["constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "constv2ctl2-full__lora_r4_ce",
                                                       "none", "prompted__safety_refusal", "prompted__si_constitution"]
    assert not any("psr" in r["cell"] for r in LLAMA_ROWS)                          # PSR is OUT by the same ruling


# --------------------------------------------------------------------------- #
# the llama paper rows: the NO-SWITCH diagnostic                               #
# smoke — the launcher's shape, its argv through its OWN dry run, the probe's  #
# wire (no switch on any body), the ledger reconciliation, the --time formula  #
# --------------------------------------------------------------------------- #


def _dry_smoke(key, view, *, s, shared, n=None, cap=None, conc=None, matrix=None, sha=SHA0):
    env = {**os.environ, "G8K_DRY_RUN": "1", "G8K_DRY_S": str(s), "G8K_DRY_SHARED": str(shared), "G8K_DRY_CPY": PY,
           "G8K_REPO_DIR": str(REPO), "USER": "tester"}
    argv = ["bash", str(G8K / "smoke_not_available.sbatch"), key, view, sha]
    if any(v is not None for v in (n, cap, conc, matrix)):
        argv += [str(n or ""), str(cap or ""), str(conc or "")] + ([matrix] if matrix else [])
    r = subprocess.run(argv, capture_output=True, text=True, env=env, check=False)
    return "\n".join(line for line in r.stdout.splitlines() if not line.startswith("[")), r.returncode, r.stdout + r.stderr


def test_smoke_not_available_sbatch_is_diagnostic_no_switch_shaped():
    """The MATH-500 smoke's shape for the no-switch class: ONE GPU, the plan's 0-00:30:00, base route (cell none, no digest,
    no object), the diagnostic root (never controls/), the probe through the ledger, the recorder, the smoke's own G-FIT."""
    smoke = (G8K / "smoke_not_available.sbatch").read_text()
    body = smoke.split("set -uo pipefail")[1]
    assert "#SBATCH --time=0-00:30:00" in smoke and "diagnostic" in smoke and "--diagnostic" in (G8K / "g8k_common.sh").read_text()
    common = (G8K / "g8k_common.sh").read_text()
    assert 'SMOKE_ROOT="$WAVE_ROOT/$G8K_SMOKE_DIR_REL/$G8K_FAMILY/smoke_${SLURM_JOB_ID:-manual}"; OUT_ROOT="$SMOKE_ROOT"' in common
    assert "controls/" not in body and "--expect-object-sha256" not in body and "--allow-over-budget" not in smoke
    assert 'g8k_row "$KEY_ARG" "$VIEW_ARG" none' in body and "g8k_smoke_check_row" in body and "g8k_smoke_budget" in body
    assert "g8k_ledger_up || exit 1" in body and "g8k_smoke_probe_argv" in body and '"${SPA[@]}"' in body
    assert "g8k_record_resolved_sampler" in body and "g8k_attempt_begin" in body and "g8k_check_endpoint" in body
    assert 'f22_bank_view_check "$CPY"\n' in body and 'f22_bank_view_check "$CPY" "$CELL"' not in body      # base route: check-only, no cell
    idx = [body.index(x) for x in ("g8k_init", "g8k_smoke_args", "g8k_row", "g8k_smoke_check_row", "g8k_smoke_root", "f22_stop_requested",
                                   "f22_serving_venv", "f22_client_venv", "g8k_datasets_probe", "f22_bank_view_check", "g8k_smoke_budget",
                                   "g8k_attempt_begin", "g8k_standup_extra", "f22_standup attacks", "g8k_check_endpoint", "g8k_ledger_up",
                                   "g8k_smoke_probe_argv", "g8k_record_resolved_sampler", "g8k_ledger_down\nf22_standup_stop")]
    assert idx == sorted(idx), idx
    assert 'G8K_MATRIX="${7:-llama}"' in smoke and "[matrix=llama]" in smoke and 'SMOKE_N_ARG="${4:-}"' in smoke
    # the probe: the family and the arm are CHOICES (a switched family / the ON arm are refused by argparse), --diagnostic REQUIRED
    probe = (REPO / M.SMOKE_PROBE_SCRIPT_REL).read_text()
    assert "choices=sorted(C.NO_THINKING_CHANNEL_FAMILIES)" in probe and "choices=[ARM]" in probe
    assert 'action="store_true", required=True' in probe and "no_switch_render_certified" in probe
    assert "C.chat_template_kwargs(ARM)" in probe and "assert C.CHAT_TEMPLATE_KWARGS_KEY not in body" in probe


def test_the_smoke_dry_run_composes_the_no_switch_probe_argv_with_no_switch_on_the_wire(tmp_path, s_by):
    """Through the launcher's OWN dry-run path: family llama, `--thinking not_available`, `--diagnostic`, the `card` row, n 24 /
    cap 1,536 / concurrency 8 from g8k_cells, the row's stamp triple, the attempt id + identity version 2, the diagnostic root;
    NO `chat_template_kwargs` / `enable_thinking` token anywhere; the standup's `--cell none` + `--enable-log-requests` with no
    parser and no digest; the recorder at the smoke cap; the row's production geometry entry."""
    none_main = next(r for r in LLAMA_ROWS if (r["view"], r["cell"]) == ("main", "none"))
    assert none_main["route_kind"] == "base" and not none_main["fitted"] and M.row_thinking(none_main) == M.THINKING_NOT_AVAILABLE
    text, rc, raw = _dry_smoke("llama8b", "main", s=44, shared=tmp_path / "shared")
    assert rc == 0, raw[-1200:]
    sp = _line(text, "SMOKE_PROBE")
    assert f" {M.SMOKE_PROBE_SCRIPT_REL} " in sp.replace(str(REPO) + "/", " ") or M.SMOKE_PROBE_SCRIPT_REL in sp
    for tok in ("--model-key llama8b ", "--family llama ", f"{M.FLAG_THINKING} not_available ", "--diagnostic ", "--sampling-mode card ",
                f"--n {M.SMOKE_N} ", f"--cap {M.SMOKE_CAP} ", f"--concurrency {M.SMOKE_CONCURRENCY} ", f"{M.FLAG_ATTEMPT_ID} a1 ",
                f"{M.FLAG_TINYCAP_IDENTITY_VERSION} 2 ", f"--task-label {M.SMOKE_TASK_LABEL} ", "--expect-stamps cell-stable\\,not-consulted\\,unspecified "):
        assert tok in sp, (tok, sp)
    import shlex
    argv = shlex.split(sp.split(":", 1)[1])
    assert argv[1].endswith(M.SMOKE_PROBE_SCRIPT_REL) and argv[argv.index("--sampling-params") + 1] == json.dumps({"temperature": 0.6, "top_p": 0.9})
    assert argv[argv.index("--expect-stamps") + 1] == "cell-stable,not-consulted,unspecified" and "--diagnostic" in argv
    assert "/exp/gsm8k_paired200_paper_objects_llama/diagnostic/llama/smoke_manual" in sp and "controls/" not in text
    assert C.CHAT_TEMPLATE_KWARGS_KEY not in text and C.ENABLE_THINKING_KEY not in text and "@thinking" not in text
    st = _line(text, "STANDUP")
    assert "--cell none" in st and "--enable-log-requests" in st and M.FLAG_REASONING_PARSER not in st and M.FLAG_EXPECT_OBJECT_SHA256 not in st
    assert f"GEOMETRY: {M.GEOMETRY_EVAL_BY_RUNG[M.CAP]} 10752 x 44 cap {M.SMOKE_CAP} thinking not_available" in text
    assert "SMOKE: job_id=g8k-llama8b-main-none key=llama8b view=main tag=llama8b cell=none family=llama thinking=not_available sampling=card route=base n=24 cap=1536 concurrency=8 diagnostic=true" in text
    rs = _line(text, "RESOLVED_SAMPLER")
    assert f"--client-cap {M.SMOKE_CAP} " in rs and "--expect temperature=0.6 --expect top_p=0.9" in rs and "--probe-cap" not in rs
    assert "--probe-expect kinds " in rs and rs.count("--probe-tags") == 2 and "--require-wired" in rs      # the ONE recorder function, the per-kind shape
    assert "BUDGET: " in text and f"arm not_available cap {M.SMOKE_CAP} entry {M.GEOMETRY_EVAL_BY_RUNG[M.CAP]} tree {SHA0}" in text
    assert "--attempt-id a1" in _line(text, "LEDGER")
    # the `it` view: the same shape on the it TAG's overlay (config from the view), still cell none, still base
    text_it, rc, raw = _dry_smoke("llama8b", "it", s=44, shared=tmp_path / "shared")
    assert rc == 0 and "job_id=g8k-llama8b-it-none" in text_it and "tag=llama8b_it" in text_it and "--family llama " in _line(text_it, "SMOKE_PROBE")
    # positional overrides ride the probe and the recorder alike
    text2, rc, raw = _dry_smoke("llama8b", "main", s=44, shared=tmp_path / "shared", n=6, cap=512, conc=2)
    assert rc == 0 and "--n 6 --cap 512 --concurrency 2 " in _line(text2, "SMOKE_PROBE") and "--client-cap 512 " in _line(text2, "RESOLVED_SAMPLER")
    # a switched family's row (the published matrix) is refused BY NAME before anything is composed
    text3, rc, raw = _dry_smoke("gemma4e4b", "main", s=20, shared=tmp_path / "shared", n=24, cap=1536, conc=8, matrix="default")
    assert rc == 1 and "certifies a switch-less family" in raw and "SMOKE_PROBE:" not in raw and "smoke_thinking.sbatch" in raw
    # a bad matrix name / a non-numeric override is refused by name
    r = subprocess.run(["bash", str(G8K / "smoke_not_available.sbatch"), "llama8b", "main", SHA0, "24", "1536", "8", "nope"],
                       capture_output=True, text=True, check=False, env=_startup_env(tmp_path, "x", {}))
    assert r.returncode == 1 and f"matrix must be one of {_admitted_matrices('smoke_not_available.sbatch')}" in r.stderr
    text4, rc, raw = _dry_smoke("llama8b", "main", s=44, shared=tmp_path / "shared", n="x", cap=1536, conc=8)
    assert rc == 1 and "positive integers" in raw


class _FakeNoSwitchLedger:
    """A ledger-shaped ``POST /v1/chat/completions`` + ``/v1/tokenize`` that RECORDS every request as it arrived (raw bytes,
    headers), counts the overlap, and answers tag-free content under the `unspecified` stamp — the proxy's literal for an
    absent switch. It also writes a ledger-shaped row per completion (thinking NULL, body_sha256 of the exact bytes,
    request_id, attempt_id) so the probe's read-back reconciles against what a real ledger would file."""

    def __init__(self, ledger_path, *, attempt_id="a1", header=(128006, 78191, 128007, 271), stray_on=(), step_s=0.03, unconditional=False):
        import hashlib
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.lock = threading.Lock()
        self.inflight = self.max_inflight = 0
        self.seen, self.answered = [], []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.loads(raw)
                with outer.lock:
                    outer.inflight += 1
                    outer.max_inflight = max(outer.max_inflight, outer.inflight)
                    outer.seen.append({"path": self.path, "sample": self.headers.get(C.HEADER_SAMPLE), "task": self.headers.get(C.HEADER_TASK),
                                       "request_id": self.headers.get(C.HEADER_REQUEST_ID), "auth": self.headers.get("Authorization"),
                                       "body": body, "sha256": hashlib.sha256(raw).hexdigest()})
                if self.path.endswith("/tokenize"):
                    if unconditional:      # the NousResearch shape: the header after EVERY render, whatever the flag says
                        toks = [1, 2, 3]
                        for m in body["messages"][1:]:
                            toks += (list(header) + [500 + sum(ord(c) * (i + 1) for i, c in enumerate(w)) % 400 for w in m["content"].split()] + [9]) if m["role"] == "assistant" else [4]
                        toks += list(header)
                    else:                  # a gated template: the flag appends the header
                        toks = [1, 2, 3] + (list(header) if body.get("add_generation_prompt") else [])
                    resp = {"tokens": toks, "count": len(toks)}
                else:
                    item = (self.headers.get(C.HEADER_SAMPLE) or "x/0").split("/")[0]
                    time.sleep(step_s * (8 - int(item.rsplit("_", 1)[1], 16) % 8))
                    content = ("<think>hm</think> 42" if item in stray_on else "The answer is 42.")
                    resp = {"id": "fake", "choices": [{"index": 0, "finish_reason": "stop",
                                                       "message": {"role": "assistant", "reasoning_content": None, "content": content}}],
                            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                            "petri": {"virtual_model": body["model"], "cell_id": "none", "render_regime": "unspecified",
                                      "cache_salt_regime": "cell-stable", "resumed_prefill_policy": "not-consulted"}}
                    ctk = body.get(C.CHAT_TEMPLATE_KWARGS_KEY) or {}
                    lrow = {"ts": "t", "task": self.headers.get(C.HEADER_TASK), "sample": self.headers.get(C.HEADER_SAMPLE),
                            "vmodel": body["model"], "thinking": (bool(ctk[C.ENABLE_THINKING_KEY]) if C.ENABLE_THINKING_KEY in ctk else None),
                            "prompt_tokens_pre": 11, "max_tokens_requested": body["max_tokens"], "has_tools": False, "n_tools": 0,
                            "status": 200, "error_code": None, "rejected": False, "usage_prompt_tokens": 11, "usage_completion_tokens": 7,
                            "finish_reason": "stop", "latency_s": 0.1, "tokenize_mismatch": False, "error_class": "none",
                            "request_id": self.headers.get(C.HEADER_REQUEST_ID), "body_sha256": hashlib.sha256(raw).hexdigest(),
                            C.LEDGER_ATTEMPT_ID_FIELD: attempt_id}
                    with outer.lock:
                        with open(ledger_path, "a") as fh:
                            fh.write(json.dumps(lrow) + "\n")
                        outer.answered.append(item)
                with outer.lock:
                    outer.inflight -= 1
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


def test_the_no_switch_smoke_probe_sends_kwarg_free_bodies_and_reconciles_the_ledger_by_request_id(tmp_path):
    """The probe's wire: NO `chat_template_kwargs` on any body (completions AND the two tokenize probes), the `card` row, the seed
    rule, a minted request id per draw, the row's task label; bounded concurrency; ONE deterministic dump (item id); the header
    measured as the driver measures it (agp true minus false); every completion certified through the contracts' seam; the ledger
    read back reconciles by request id — thinking NULL on every row, body_sha256 == the bytes sent; a stray tag FAILS the row."""
    P = _load("_g8k_smoke_probe_wire", M.SMOKE_PROBE_SCRIPT_REL)
    ledger = tmp_path / "usage_ledger.jsonl"
    ids = [f"gsm8k_{i:08x}" for i in range(12)]
    plan = [(i, [{"role": "system", "content": "few-shot"}, {"role": "user", "content": f"Q {i}"}]) for i in ids]
    params = {"temperature": 0.6, "top_p": 0.9}
    srv = _FakeNoSwitchLedger(ledger)
    try:
        header, hfacts = P.header_ids_no_switch(srv.url, "petri__base", token="tok")
        assert header == [128006, 78191, 128007, 271] == list(C.FAMILY_RENDER_FACTS["llama"].off_ids)
        assert hfacts["construction"] == P.HEADER_CONSTRUCTION_GENERATION_PROMPT_DIFF and hfacts["agp_invariant"] is False
        tok_bodies = [s["body"] for s in srv.seen if s["path"].endswith("/tokenize")]
        assert len(tok_bodies) == 2 and [b["add_generation_prompt"] for b in tok_bodies] == [True, False]
        assert all(C.CHAT_TEMPLATE_KWARGS_KEY not in b for b in tok_bodies)
        assert P.certify_completion("llama", header, regime=None, content="", reasoning=None, finish_reason=None)["error"] is None
        rows = P.run_completions(srv.url, plan, vmodel="petri__base", cap=1536, params=params, token="tok", model_key="llama8b",
                                 family="llama", sampling_mode="card", task_label="smoke/gsm8k", concurrency=4, timeout_s=30, header_ids=header)
        assert 2 <= srv.max_inflight <= 4 and len(rows) == 12 and [r["item_id"] for r in rows] == sorted(ids) and srv.answered != sorted(ids)
        comps = [s for s in srv.seen if s["path"].endswith("/chat/completions")]
        by_rid = {s["request_id"]: s for s in comps}
        assert len(by_rid) == 12 and all(s["task"] == "smoke/gsm8k" and s["auth"] == "Bearer tok" for s in comps)
        for r in rows:
            s = by_rid[r["request_id"]]
            b = s["body"]
            assert C.CHAT_TEMPLATE_KWARGS_KEY not in b and C.ENABLE_THINKING_KEY not in b and r["body_has_chat_template_kwargs"] is False
            assert b["max_tokens"] == 1536 and b["seed"] == P.item_seed(r["item_id"], 0) and b["temperature"] == 0.6 and b["top_p"] == 0.9
            assert s["sample"] == f"{r['item_id']}/0" and s["sha256"] == r["body_sha256"]
            assert r["arm"] == "not_available" and r["split_mode"] == "none" and r["cert_class"] == C.CERT_CLASS_NO_SWITCH
            assert r["diagnostic"] is True and r["error"] is None and r["stamps"]["render_regime"] == "unspecified"
            assert r["shape"]["ok"] is True and r["shape"]["answer_is_content"] is True and r["shape"]["present"] is False
            assert r["certification"]["render_certified"] is True and r["certification"]["error"] is None
            assert r["certification"]["measured_header_ids_by_arm"] == {"not_available": header}
            assert r["content"] == "The answer is 42." and r["reasoning_content"] is None
        path = tmp_path / "smoke_completions.jsonl"
        P.dump_completions(path, rows)
        assert [json.loads(l) for l in path.read_text().splitlines()] == rows
        rec = P.ledger_reconciliation(P.read_ledger(ledger), rows, attempt_id="a1")
        assert rec["ok"] is True and rec["thinking_values"] == ["None"] and rec["n_body_sha256_mismatch"] == 0 and rec["n_rows"] == 12
        # a wrong attempt id, a missing row, a foreign body hash: each refuses by name
        assert P.ledger_reconciliation(P.read_ledger(ledger), rows, attempt_id="a2")["attempt_id_on_every_row"] is False
        assert P.ledger_reconciliation(P.read_ledger(ledger)[1:], rows, attempt_id="a1")["n_missing_request_ids"] == 1
        tampered = [{**r, "body_sha256": "0" * 64} for r in rows[:1]] + rows[1:]
        assert P.ledger_reconciliation(P.read_ledger(ledger), tampered, attempt_id="a1")["n_body_sha256_mismatch"] == 1
    finally:
        srv.close()
    # the NousResearch shape (smoke 490518): the flag is never consulted, so the diff is EMPTY — the probe reads the unconditional
    # trailing header by content substitution (two sentinel renders, no kwargs) and lands on the SAME 4-id tuple
    srv_u = _FakeNoSwitchLedger(tmp_path / "usage_ledger_u.jsonl", unconditional=True)
    try:
        header_u, hf = P.header_ids_no_switch(srv_u.url, "petri__base", token="tok")
        assert header_u == [128006, 78191, 128007, 271] and hf["construction"] == P.HEADER_CONSTRUCTION_UNCONDITIONAL_TRAILING
        assert hf["agp_invariant"] is True and hf["render_len_by_agp"] == {"true": 7, "false": 7} and hf["content_start"] == 7
        assert hf["closing_scaffold"] == [9, 128006, 78191, 128007, 271]
        tb = [s["body"] for s in srv_u.seen if s["path"].endswith("/tokenize")]
        assert len(tb) == 4 and all(C.CHAT_TEMPLATE_KWARGS_KEY not in b for b in tb) and [b["add_generation_prompt"] for b in tb] == [True, False, False, False]
        assert [len(b["messages"]) for b in tb] == [1, 1, 2, 2] and tb[2]["messages"][1]["role"] == "assistant" and tb[2]["messages"][1]["content"] != tb[3]["messages"][1]["content"]
    finally:
        srv_u.close()
    # a stray tag / a reasoning channel / a wrong stamp: the shape and the seam both refuse the row by name
    bad = P.shape_check("<think>x</think> 42", None, finish_reason="stop")
    assert bad["ok"] is False and "stray" in bad["why"] and bad["stray_open_tag"] is True and bad["stray_close_tag"] is True
    assert P.shape_check("42", "some reasoning", finish_reason="stop")["ok"] is False
    assert P.shape_check("42", None, finish_reason="stop")["ok"] is True and P.shape_check("4", None, finish_reason="length")["ok"] is True
    assert P.shape_check("42", None, finish_reason="None")["ok"] is False
    c = P.certify_completion("llama", header, regime="thinking_off", content="42", reasoning=None, finish_reason="stop")
    assert c["error"] and "render_regime" in c["error"] and c["render_certified"] is False
    c = P.certify_completion("llama", header, regime="unspecified", content="<think>a</think>b", reasoning=None, finish_reason="stop")
    assert c["error"] and "think-tag literal" in c["error"]
    c = P.certify_completion("llama", [1, 2], regime="unspecified", content="42", reasoning=None, finish_reason="stop")
    assert c["error"] and "tuple equality" in c["error"]
    ledger2 = tmp_path / "usage_ledger2.jsonl"
    srv2 = _FakeNoSwitchLedger(ledger2, stray_on={ids[3]})
    try:
        rows2 = P.run_completions(srv2.url, plan[:5], vmodel="petri__base", cap=1536, params=params, token=None, model_key="llama8b",
                                  family="llama", sampling_mode="card", task_label="smoke/gsm8k", concurrency=2, timeout_s=30,
                                  header_ids=[128006, 78191, 128007, 271])
        strays = [r for r in rows2 if not r["shape"]["ok"]]
        assert [r["item_id"] for r in strays] == [ids[3]] and strays[0]["certification"]["error"] and "think-tag literal" in strays[0]["certification"]["error"]
    finally:
        srv2.close()
    # argparse: the family / the arm are CHOICES, --diagnostic is REQUIRED — a switched family, the ON arm, or a silent run refuse
    base = ["--ledger-url", "u", "--proxy-url", "p", "--routing-table", "t", "--endpoint-json", "e", "--ledger-path", "l",
            "--model-key", "llama8b", "--out", str(tmp_path / "o")]
    for extra in (["--family", "nemotron_h", "--thinking", "not_available", "--diagnostic"],
                  ["--family", "llama", "--thinking", "on", "--diagnostic"],
                  ["--family", "llama", "--thinking", "not_available"],
                  ["--family", "llama", "--thinking", "not_available", "--diagnostic", "--sampling-mode", "thinking_card"],
                  ["--family", "llama", "--thinking", "not_available", "--diagnostic", "--tinycap-identity-version", "1"]):
        with pytest.raises(SystemExit) as e:
            P.main(base + extra)
        assert e.value.code == 2
    with pytest.raises(ValueError, match="concurrency"):
        P.run_completions("http://127.0.0.1:1/v1", plan, vmodel="v", cap=1, params={}, token=None, model_key="k", family="llama",
                          sampling_mode="card", task_label="x", concurrency=0, timeout_s=1)


def test_smoke_not_available_sbatch_time_is_the_header_s_worst_case_formula_at_x2():
    """The MATH-500 smoke's rule on the no-switch smoke: the wall from the realistic per-request time at the diagnostic cap under
    the probe's concurrency, x2 — never from the 7,200 s timeout, never x5; the counts are g8k_cells' (ONE source); the plan's own
    0-00:30:00 (§3.7 step 3), which the header says is BELOW f22_standup's 40-min endpoint ceiling (a hung standup files INCOMPLETE)."""
    import math
    src = (G8K / "smoke_not_available.sbatch").read_text()
    m = re.search(r"--time = (\d+) x \(standup (\d+) min \+ (\d+) tokenize probes x (\d+) min \+ "
                  r"ceil\((\d+) completions / (\d+) in flight\) x (\d+) min \+ teardown (\d+) min\) = (\d+) x (\d+) min = (\d+) min", src)
    assert m, "the --time formula line is missing from the smoke sbatch header"
    pad, standup, n_probe, t_probe, n_req, conc, t_req, teardown, pad2, subtotal, total = map(int, m.groups())
    assert pad == pad2 == 2
    assert subtotal == standup + n_probe * t_probe + math.ceil(n_req / conc) * t_req + teardown and total == pad * subtotal
    assert f"#SBATCH --time=0-{total // 60:02d}:{total % 60:02d}:00" in src and total == 30
    assert n_req == M.SMOKE_N == 24 and conc == M.SMOKE_CONCURRENCY == 8
    assert n_probe == 2                # ONE render x (add_generation_prompt true, false) through /v1/tokenize — no arms
    common = (F22 / "common.sh").read_text()
    w = re.search(r"for i in \$\(seq 1 (\d+)\); do\n\s*\[ -f \"\$out/endpoint\.json\" \] && break\n.*?\n\s*sleep (\d+)\n", common, re.DOTALL)
    assert w and int(w.group(1)) * int(w.group(2)) == 2400 and total * 60 < 2400 and "40 min" in src and "INCOMPLETE" in src


# --------------------------------------------------------------------------- #
# E2 — the appended waves (math-eight-arms-extension plan P2 / P3 / P8; decisions-e2)   #
# --------------------------------------------------------------------------- #

FROZEN_FIXTURE = REPO / "tests/fixtures/eight_arms/gsm8k_matrix_8f5f1495.json"
#: the steer arms of the PANEL view — the rung's two, which is what the published and Llama matrices' steer waves carry.
#: DERIVED from the per-arm view table, never re-typed: an arm registered for another view (the all-positions Arditi one, `it`)
#: is another matrix's row, and enumerating `M.STEER_CELLS` here would demand it of documents it is not in.
PANEL_STEER_CELLS = tuple(c for c in M.STEER_CELLS if M.external_row_view_for(c) == M.VIEW_MAIN)
ADMISSION = M.load_steer_admission()
EXT_CELLS_BY_KEY = {"gemma4e4b": [M.CELL_PROMPTED, M.CELL_PROMPTED_CONST], "qwen35_4b": [M.CELL_PROMPTED, M.CELL_PROMPTED_CONST],
                    "nemotron3nano4b": [M.CELL_PROMPTED_CONST]}
#: the four full22 families' long-SI S at the `_f22` entries (the dry run's resolved S: ModelBudget.seqs_at over the pinned budgets)
F22_LONGSI_S = {"gemma4e4b": (35, 23), "qwen35_4b": (44, 29), "nemotron3nano4b": (47, 31)}   # qwen35_4b: 741,888 // 16,640 = 44 (the 2026-09-13 row; 43 under the swept 720,896), // 24,832 = 29


def _stage_external_bank_g8k(root, row):
    """A view-shaped EXTERNAL bank for one steering row at its PRIMARY-role view rel: `<root>/<bank_view_rel.primary>/<cell dir>/
    object.pt` of the arm's real kind IN THE REAL BLOBS' SHAPE (read off the staged objects on the primary cluster, 2026-09-13,
    decisions-e2 E2-7): the AlphaSteer rebake's `kind / layer / layers / steer_positions: prefill / site: pre / base_scale` and the
    Arditi strength lane's `kind: static_r / rs / steer_positions: prefill_all / site / per_layer_r_norm` — NEITHER carries a
    `positions` key (the served mode the routing table stamps from the resolved spec), so the fresh table records no
    served-positions fact for the cell; plus the lane's identity sidecar (`positions_at_fit`, the fit regime) and the VIEW_MANIFEST
    the launcher's check requires. The MATH-500 launcher's `_stage_external_bank`, on the GSM8K row's fields."""
    import torch
    from sisteer.capeval.petri import registry_adapter as ra
    view = root / row["bank_view_rel"]["primary"]
    d = view / row["bank_cell_dir"]
    d.mkdir(parents=True, exist_ok=True)
    kind = ra.ARM_OBJECT_KINDS[row["cell"]]
    blob = ({"kind": kind, "base_scale": 0.05, "rs": {8: torch.full((4,), 0.1)}, "steer_positions": "prefill_all", "site": "pre",
             "per_layer_r_norm": {8: 1.0}} if kind == "static_r" else
            {"kind": kind, "layer": 8, "layers": [8], "base_scale": 1.0, "w": torch.zeros(4, 4), "steer_positions": "prefill", "site": "pre"})
    torch.save(blob, d / "object.pt")
    (d / "fit_provenance.json").write_text(json.dumps({"resolved_model_id": "synthetic", "resolved_revision": "0" * 40, "chat_template_hash": "0" * 16,
                                                       "positions_at_fit": "prefill" if kind != "static_r" else "all", "writer": "synthetic@" + "0" * 40}))
    (view / "VIEW_MANIFEST.json").write_text(json.dumps({"target": row["model_key"], "view": row["bank_view"], "entries": [{"cell_id": row["cell"]}]}))
    return view, d / "object.pt"


def test_e2_the_default_matrix_appends_the_ext_and_steer_waves_behind_the_frozen_42_and_the_llama_matrix_its_steer_rows(tmp_path):
    """E2 (plan P2 / P8): ONE matrix per `--matrix` name — the published rows first and byte-identical to the
    8f5f1495 document (no `wave` key: the analysis' reader and every published consumer see the rows they saw), then the `ext`
    wave (the eight-arm prompted baselines the panel view lacked: gemma / qwen safety + constitution, nemotron constitution — prompted
    routes, no pin, no roster edit) and the `steer` wave (the AlphaSteer / Arditi rung, one row per admission-record entry on the
    PANEL view `main`, the arm's view as `bank_view`, the object pinned by the record's sha, the selection label VERBATIM, the rung's own
    `_steer_<cap>` tuple on a switch model); the Llama matrix appends its two steer rows (no-switch: no tuple, `card`, `primary` for
    the AlphaSteer object staged on one cluster). The document names the admission record it was generated from, by sha."""
    doc = M.matrix_doc()
    frozen = json.loads(FROZEN_FIXTURE.read_text())["rows"]
    assert len(ALL_ROWS) == 53 and ALL_ROWS[:42] == frozen and all("wave" not in r for r in ALL_ROWS[:42])
    assert [r["index"] for r in ALL_ROWS] == list(range(53)) and [r["index"] for r in EXT_ROWS] == list(range(42, 47))
    assert [r["index"] for r in STEER_ROWS] == list(range(47, 53)) and ROWS == ALL_ROWS[:42] and LAUNCHED == M.launch_rows(ALL_ROWS[:42])
    assert [(r["model_key"], r["view"], r["cell"]) for r in EXT_ROWS] == [(k, "main", c) for k, cells in EXT_CELLS_BY_KEY.items() for c in cells]
    # the published matrix's steer wave is the PANEL view's arms (M.EXTERNAL_ROW_VIEW_BY_CELL declares the view per arm; an arm
    # whose view is not `main` — the all-positions Arditi one — is another matrix's row and never appears here)
    assert PANEL_STEER_CELLS == (M.CELL_ALPHASTEER, M.CELL_ARDITI) and set(PANEL_STEER_CELLS) < set(M.STEER_CELLS)
    assert [(r["model_key"], r["cell"]) for r in STEER_ROWS] == [(k, c) for k in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b") for c in PANEL_STEER_CELLS]
    # the document: the appended waves' blocks, the admission record by path + sha, the counts per wave
    assert [(e["key"], e["view"], e["cells"]) for e in doc["extensions"]] == [(k, "main", c) for k, c in EXT_CELLS_BY_KEY.items()]
    assert [(e["key"], e["arms"]) for e in doc["steering"]] == [(k, list(PANEL_STEER_CELLS)) for k in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b")]
    assert doc["source"]["steer_admission"] == str(M.STEER_ADMISSION_REL) == "configs/sdfbase/steer_admission.json"
    assert doc["source"]["steer_admission_sha256"] == _sha256(M.STEER_ADMISSION_PATH)
    c = M.counts(ALL_ROWS)
    assert c["waves"] == {"published": 42, "ext": 5, "steer": 6} and c["launched_by_wave"] == {"published": 38, "ext": 5, "steer": 6}
    assert c["statuses"] == {"launch": 49, "held": 4} and c["deferred"] == 0 and c["rows"] == 53 and c["launched"] == 49 and c["held"] == 4
    assert sorted(c["external_pins"]) == [f"{k}/{cell}" for k in ("gemma4e4b", "nemotron3nano4b", "qwen35_4b") for cell in PANEL_STEER_CELLS]
    assert set(c["external_pins"].values()).isdisjoint(set(c["pins"].values())) and c["external_pins"]["gemma4e4b/alphasteer"] == ADMISSION[("gemma4e4b", "alphasteer")]["object_sha256"]
    assert M.published_counts() == M.counts(ROWS) == COUNTS and COUNTS["launched"] == 38 and COUNTS["rows"] == 42
    # the ext rows: prompted, unpinned, on the panel view, under controls_ext/, at the class's --time (the constitution prefill: fitted class)
    for r in EXT_ROWS:
        const = r["cell"] == M.CELL_PROMPTED_CONST
        assert r["wave"] == M.WAVE_EXT and r["extension"] is True and not r["fitted"] and not r["declared"] and r["route_kind"] == "prompted"
        assert r["object_sha256"] is None and r["tuple_name"] is None and r["tuple_names"] is None and r["staged_id"] is None
        assert r["view"] == "main" and r["tag"] == r["model_key"] and r["status"] == M.STATUS_LAUNCH and r["held_reason"] is None
        assert r["out_root"] == f"controls_ext/{r['tag']}/{r['cell_slug']}/thinking_card" and r["rung_out_root"] == r["out_root"] + "__rung16384"
        assert r["time_limit"] == (M.TIME_FITTED if const else M.TIME_IN_SUPPORT) == M.time_limit_for(r["model_key"], False, r["cell"])
        assert r["job_id"] == f"g8k-{r['model_key']}-main-" + ("prompted-const" if const else "prompted")
        assert r["expected_stamps"] == list(C.g8k_expected_stamps(fitted=False, declared=False)) == ["cell-stable", "not-consulted", "thinking_on"]
        assert r["cluster_constraint"] == ("primary" if r["model_key"] == "nemotron3nano4b" else "any") and r["harness_cell"] == ("prompted__constitution" if const else "prompted__safety")
        assert r["budget_class"] == r["cell"] and "prompted__" in r["standup_cell"]
    # the steer rows: the admission record's facts on the row, the panel view, the lane view beside it, the pin, the tuple rule
    for r in STEER_ROWS:
        key, cell = r["model_key"], r["cell"]
        e = ADMISSION[(key, cell)]
        assert r["wave"] == M.WAVE_STEER and r["object_source"] == M.OBJECT_SOURCE_EXTERNAL and r["extension"] is True
        assert r["fitted"] and not r["declared"] and r["route_kind"] == "lens" and r["served_positions_expected"] == "all"
        assert r["view"] == "main" and r["tag"] == key and r["bank_view"] == e["view"] and r["bank_view_rel"] == e["bank_view_rel"]
        assert r["bank_cell_dir"] == e["bank_cell_dir"] and r["lane_root"] == e["lane_root"] and r["bank_view_rel"]["primary"].startswith(e["lane_root"])
        assert r["object_sha256"] == e["object_sha256"] and len(r["object_sha256"]) == 64 and r["staged_id"] == e["staged_id"]
        assert r["selection_label"] == e["selection_label"] and r["selection_rule"] == e["selection_rule"]
        if cell == M.CELL_ALPHASTEER:
            assert r["selection_label"] == json.loads(M.STEER_ADMISSION_PATH.read_text())["selection_label_alphasteer"] and r["rung"] is not None and r["lambda"] is not None
        assert r["tuple_name"] == M.steer_tuple_name(key, rung=M.CAP) == C.g8k_steer_tuple_name(key, M.CAP) == f"gsm8k_paired200_{key}_steer_8192"
        assert r["tuple_names"] == {"8192": f"gsm8k_paired200_{key}_steer_8192", "16384": f"gsm8k_paired200_{key}_steer_16384"}
        assert r["out_root"] == f"controls_steer/{key}/{cell}/thinking_card" and r["rung_out_root"] == r["out_root"] + "__rung16384"
        assert r["time_limit"] == M.TIME_FITTED and r["job_id"] == f"g8k-{key}-main-{cell}" and r["expected_stamps"] == ["cell-stable", "always", "thinking_on"]
        # the rule of 2026-09-14 moved nemotron off its A4-fallback point, so no launch row is grid-limited any more
        # (the superseded point's flag lives under the entry's `superseded` block)
        assert r["cluster_constraint"] == ("primary" if key == "nemotron3nano4b" else "any") and r["grid_limited"] is False
        assert r["status"] == M.STATUS_LAUNCH and r["harness_cell"] == r["standup_cell"] == cell and r["thinking_mode" if "thinking_mode" in r else "cell"] in (cell, "on")
        assert M.external_row_problem(r) is None
        # E2-B: the selection OUTCOME rides every AlphaSteer row beside the selection label — the ORACLE's word
        # (scripts/alphasteer_f22/operating_points.py): `ruled_2026-09-14` on qwen lam1 and nemotron lam2,
        # `ruled_2026-09-14_second` on gemma lam2 (the second ruling of that day moved gemma lam3 -> lam2; the oracle
        # spells it `_second` so a reader can tell which ruling a pin was cut against; the record re-stamped from the
        # interim `candidate_2026-09-14` on 2026-09-15), `chosen_confirmed_2026-09-14` on llama (lam1 confirmed); the
        # superseded points are kept under each entry's `superseded` block with their shas; an Arditi row carries None
        assert r["selection_outcome"] == e.get("selection_outcome")
        if cell == M.CELL_ALPHASTEER:
            expected_outcome = {"gemma4e4b": M.SELECTION_OUTCOME_RULED_SECOND, "llama8b": M.SELECTION_OUTCOME_CHOSEN_CONFIRMED}.get(key, M.SELECTION_OUTCOME_RULED)
            assert r["selection_outcome"] == expected_outcome
            # the oracle-derived source names operating_points.py on every ruled / confirmed entry; the ruling stamp is on every entry
            assert "operating_points.py" in e["selection_outcome_source"]
            assert e["ruling"]["ruled_on"] == "2026-09-14"
            assert (e.get("superseded") is not None) is (key != "llama8b")
            if key != "llama8b":
                assert e["superseded"]["object_sha256"] != e["object_sha256"] and len(e["superseded"]["object_sha256"]) == 64
                # the point each ruling displaced: nemotron's A4 fallback, qwen's ladder `chosen`, and on gemma the FIRST ruling's lam3
                # (the ladder's own `chosen`, lam1, sits under `superseded_earlier` — gemma's point moved twice on 2026-09-14)
                assert e["superseded"]["selection_outcome"] == {"nemotron3nano4b": M.SELECTION_OUTCOME_A4_FALLBACK,
                                                               "gemma4e4b": M.SELECTION_OUTCOME_RULED}.get(key, M.SELECTION_OUTCOME_CHOSEN)
            if key == "gemma4e4b":
                # the interim candidate record's `ruled_disqualified_pending_user` / `ruling_note` are gone: the disqualification became a
                # ruling, so lam3 is `superseded` (on 2026-09-14, by the second ruling, with the oracle's grounds), and lam1 keeps its
                # earlier supersession; the ruling block names both what it supersedes (the ladder's lam1) and what it displaced (lam3)
                assert "ruled_disqualified_pending_user" not in e and "ruling_note" not in e and not e["selection_outcome"].startswith("candidate_")
                sup = e["superseded"]
                assert sup["rung"] == 3 and sup["object_sha256"].startswith("748e067e") and sup["superseded_on"] == "2026-09-14"
                assert "SECOND ruling" in sup["superseded_by"] and "lam3 -> lam2" in sup["superseded_by"] and "147/200" in sup["superseded_by"]
                assert sup["why_superseded"]["applies_to_rung_i"] == 3 and sup["first_ruling"]["ruled_on"] == "2026-09-14"
                assert [h["rung"] for h in e["superseded_earlier"]] == [1] and e["superseded_earlier"][0]["selection_outcome"] == M.SELECTION_OUTCOME_CHOSEN
                assert e["superseded_earlier"][0]["object_sha256"] not in (e["object_sha256"], sup["object_sha256"])
                assert e["ruling"]["second_ruling"]["moved"] == "lam3 -> lam2" and e["ruling"]["displaces"]["rung_i"] == 3 and e["ruling"]["supersedes"]["rung_i"] == 1
                assert e["dose_evidence"]["from_operating_points"]["over_refusal_is_upper_bound"] is True and "over_refusal_orbench" in e["dose_evidence"]["must_report_with"]
            assert r["rung"] == e["rung"] == {"gemma4e4b": 2, "qwen35_4b": 1, "nemotron3nano4b": 2, "llama8b": 1}[key]
            for role in ("primary", "overflow"):
                assert r["bank_view_rel"][role].endswith(f"ladder_canon/lam{r['rung']}/cells_view/{r['bank_view']}")
        elif key == "llama8b":
            # the ruling of 2026-09-14 moved llama8b's Arditi point off the battery's inclusive-boundary
            # pick (alpha 0.1 -> 0.05): the ONE non-null word an Arditi entry may carry, with `ruling` + `superseded` bound by the gate
            # (steer_admission_ruling.arditi_entry_problems); the superseded 0.1 point keeps its sha under `superseded`
            assert r["selection_outcome"] == M.SELECTION_OUTCOME_RULED and e["selection_outcome_source"].startswith("RULING 2026-09-14")
            assert r["alpha"] == e["ruling"]["alpha"] == 0.05 and r["bank_view"] == e["ruling"]["view"] == "ar_r0p05" and r["object_sha256"] == e["ruling"]["object_sha256"]
            assert e["superseded"]["alpha"] == 0.1 and e["superseded"]["view"] == "ar_r0p1" and e["superseded"]["object_sha256"] == e["ruling"]["supersedes"]["object_sha256"] != r["object_sha256"]
            assert e["superseded"]["superseded_on"] == e["ruling"]["ruled_on"] == "2026-09-14" and r["alternative_not_pinned"] is None
        else:
            assert r["selection_outcome"] is None and e["selection_outcome_source"].startswith("not applicable")
        for bad, why in ((None, "selection_outcome None on an alphasteer"), ("picked", "selection_outcome 'picked'")):
            if cell == M.CELL_ALPHASTEER:
                assert why in M.selection_outcome_problem(cell, bad)
        # an Arditi entry carries null (the battery's pick) or exactly the ruled word (a user ruling over it) — never a ladder word
        assert M.selection_outcome_problem(M.CELL_ARDITI, M.SELECTION_OUTCOME_CHOSEN) and M.selection_outcome_problem(M.CELL_ARDITI, None) is None
        assert M.selection_outcome_problem(M.CELL_ARDITI, M.SELECTION_OUTCOME_RULED) is None and M.ARDITI_SELECTION_OUTCOMES == (None, M.SELECTION_OUTCOME_RULED)
        for ladder_word in (M.SELECTION_OUTCOME_A4_FALLBACK, M.SELECTION_OUTCOME_CHOSEN_CONFIRMED, M.SELECTION_OUTCOME_RULED_SECOND,
                            M.SELECTION_OUTCOME_CANDIDATE):
            assert "ladder" in M.selection_outcome_problem(M.CELL_ARDITI, ladder_word)
        assert M.external_row_problem({**r, "selection_outcome": "picked"}) and "selection_outcome" in M.external_row_problem({k: v for k, v in r.items() if k != "selection_outcome"})
    # the Llama matrix: the twelve published rows byte-for-byte, then its two steer rows (no-switch: card, no tuple, in support)
    assert len(LLAMA_ALL_ROWS) == 14 and [r["index"] for r in LLAMA_STEER_ROWS] == [12, 13] and LLAMA_ROWS == LLAMA_ALL_ROWS[:12]
    for r in LLAMA_STEER_ROWS:
        e = ADMISSION[("llama8b", r["cell"])]
        assert r["thinking_mode"] == M.THINKING_NOT_AVAILABLE and r["sampling_mode"] == "card" and r["tuple_name"] is None and r["tuple_names"] is None
        assert r["object_sha256"] == e["object_sha256"] and r["bank_view"] == e["view"] and r["view"] == "main" and r["bank_view_rel"] == e["bank_view_rel"]
        assert r["out_root"] == f"controls_steer/llama8b/{r['cell']}/card" and r["time_limit"] == M.TIME_FITTED_LLAMA and r["wave"] == M.WAVE_STEER
        assert r["expected_stamps"] == ["cell-stable", "always", "unspecified"] and r["object_source"] == "external"
    # llama AlphaSteer stays PRIMARY by its constraint; since the 2026-09-14 re-pin (decisions-e2b LB10 (i)) the record binds the overflow
    # ROLE too — the same `ladder_canon/lam1` rel on both roles — so the constraint, not an empty rel, is what keeps the row off the overflow
    assert LLAMA_STEER_ROWS[0]["cluster_constraint"] == "primary"
    assert LLAMA_STEER_ROWS[0]["bank_view_rel"]["overflow"] == LLAMA_STEER_ROWS[0]["bank_view_rel"]["primary"] and LLAMA_STEER_ROWS[0]["bank_view_rel"]["primary"].endswith("/ladder_canon/lam1/cells_view/it")
    # the first open question is resolved by the RULING of 2026-09-14, not by a re-run of the battery: the inclusive-boundary pick (0.1, `not_robust`) is
    # superseded; the active row is alpha 0.05 with `boundary_pick: ruled` and the ruled word, the 0.1 pick's flag kept under `superseded`
    assert LLAMA_STEER_ROWS[1]["cluster_constraint"] == "any" and LLAMA_STEER_ROWS[1]["boundary_pick"] == "ruled" and LLAMA_STEER_ROWS[1]["alpha"] == 0.05
    assert LLAMA_STEER_ROWS[1]["selection_outcome"] == M.SELECTION_OUTCOME_RULED and LLAMA_STEER_ROWS[1]["bank_view"] == "ar_r0p05"
    assert ADMISSION[("llama8b", "arditi")]["superseded"]["boundary_pick"]["status"] == "not_robust"
    # E2-12 / E2-16: both llama steer rows launch today (the class-B record hold on the AlphaSteer object was lifted when the
    # rebake lane re-staged its record); a `holds:` entry naming (llama8b, main, <steer cell>) would hold the row — the
    # steering pass honours the block — pinned and never launched (the mechanism, pinned below on a synthetic reason)
    alpha, ard = LLAMA_STEER_ROWS
    assert alpha["status"] == ard["status"] == M.STATUS_LAUNCH and alpha["held_reason"] is None and M.held_rows(LLAMA_ALL_ROWS) == []
    ldoc = M.matrix_doc(LLAMA_MATRIX)
    assert "extensions" not in ldoc and [(e["key"], e["arms"]) for e in ldoc["steering"]] == [("llama8b", list(PANEL_STEER_CELLS))]
    assert ldoc["source"]["steer_admission_sha256"] == doc["source"]["steer_admission_sha256"] and ldoc["holds"] == []
    assert M.counts(LLAMA_ALL_ROWS)["launched_by_wave"] == {"published": 12, "steer": 2} and M.counts(LLAMA_ALL_ROWS)["held"] == 0
    held_row = V._external_row(12, key="llama8b", cell="alphasteer", entry=ADMISSION[("llama8b", "alphasteer")], where="synthetic", family="llama",
                               arm=M.THINKING_NOT_AVAILABLE, sampling_mode="card", schema2=True, trigger_threshold=0.01,
                               held_reason="held (synthetic): the staged record describes other bytes")
    assert held_row["status"] == M.STATUS_HELD and held_row["held_reason"].startswith("held (synthetic)") and held_row["object_sha256"] == alpha["object_sha256"]
    assert {k: v for k, v in held_row.items() if k not in ("status", "held_reason")} == {k: v for k, v in alpha.items() if k not in ("status", "held_reason")}
    with pytest.raises(SystemExit, match="status=held"):
        M.assert_launchable(held_row, [*LLAMA_ALL_ROWS[:12], held_row, LLAMA_ALL_ROWS[13]])
    assert M.eligible_rows(M.WAVE_STEER, [*LLAMA_ALL_ROWS[:12], held_row, LLAMA_ALL_ROWS[13]]) == [LLAMA_ALL_ROWS[13]]
    # the generators agree with the committed documents (both matrices, the overlays)
    py = sys.executable
    for argv in ([py, str(G8K / "make_gsm8k_matrix.py"), "--check"], [py, str(G8K / "make_gsm8k_overlays.py"), "--check"]):
        r = subprocess.run(argv, capture_output=True, text=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, check=False)
        assert r.returncode == 0, r.stdout + r.stderr
    # the loader holds every external row to its shape: a pin that is not 64-hex, a wave-less external row, an out root under
    # another wave's prefix and an unknown status each refuse by name (the matrix is never half-trusted)
    def loads(mutate):
        d = json.loads(json.dumps(doc))
        mutate(d["rows"])
        p = tmp_path / "m.json"
        p.write_text(json.dumps(d))
        return M.load_matrix(p)
    with pytest.raises(SystemExit, match="object_sha256"):
        loads(lambda rows: rows[47].__setitem__("object_sha256", "abc"))
    with pytest.raises(SystemExit, match="wave"):
        loads(lambda rows: rows[47].pop("wave"))
    with pytest.raises(SystemExit, match="controls_steer"):
        loads(lambda rows: rows[47].__setitem__("out_root", rows[47]["out_root"].replace("controls_steer/", "controls/")))
    with pytest.raises(SystemExit, match="status"):
        loads(lambda rows: rows[42].__setitem__("status", "queued"))
    # E2-B: a steer row whose selection_outcome is not the ladder's vocabulary, or an Arditi row carrying one, refuses at load by name
    with pytest.raises(SystemExit, match="selection_outcome"):
        loads(lambda rows: rows[47].__setitem__("selection_outcome", "picked"))
    with pytest.raises(SystemExit, match="selection_outcome"):
        loads(lambda rows: rows[48].__setitem__("selection_outcome", M.SELECTION_OUTCOME_CHOSEN))
    assert loads(lambda rows: None)["counts"]["rows"] == 53


def test_e2_the_plan_is_wave_scoped_and_the_batch_boundary_refuses_complete_held_and_deferred_rows(tmp_path):
    """E2 (plan P2 / F11e; ported from the MATH-500 launcher's fix G1): a submission names ONE wave and every printed row is in
    that wave's eligible set — the published wave is COMPLETE (never resubmitted, even by index), a held or deferred row is never
    a job, an index outside the wave refuses by name, the overflow complement is taken WITHIN the wave; the batch boundary
    (`assert_launchable`, the sbatch's `g8k_assert_launchable`) re-asserts the ONE resolved row by the same predicate. The -J tag
    of a class is the rows' model scope, never the bare literal (0-01:00:00 is both the full22 in-support class and the Llama
    fitted class)."""
    assert M.eligible_rows(M.WAVE_EXT) == EXT_ROWS and M.eligible_rows(M.WAVE_STEER) == STEER_ROWS and M.COMPLETED_WAVES == ("published",)
    with pytest.raises(SystemExit, match="COMPLETE"):
        M.eligible_rows(M.WAVE_PUBLISHED)
    with pytest.raises(SystemExit, match="not a wave of this matrix"):
        M.eligible_rows("ext1")
    with pytest.raises(SystemExit, match="no launchable row"):
        M.eligible_rows(M.WAVE_EXT, [dict(r, status=M.STATUS_HELD, held_reason="x") for r in EXT_ROWS])
    # the ext submission: two classes (the constitution prefill at the fitted class), full22 tags, the default matrix's tail
    pl = M.plan(SHA0, "cluster_a", max_concurrent=12, wave=M.WAVE_EXT, assert_eligible_record=True)
    assert pl.rows == [42, 43, 44, 45, 46] and pl.sizes == {M.TIME_FITTED: 3, M.TIME_IN_SUPPORT: 2} and pl.wave == M.WAVE_EXT
    assert [re.search(r"-J (g8k-\S+)", l).group(1) for l in pl.lines] == ["g8k-fitted", "g8k-in-support"] and pl.tags == {M.TIME_FITTED: "fitted", M.TIME_IN_SUPPORT: "in-support"}
    assert all(l.endswith(f"gsm8k_thinking.sbatch ARRAY {SHA0}") and f"--qos={SITE_A['qos']}" in l for l in pl.lines)
    assert "--array=43,45-46%" in pl.lines[0] and "--array=42,44%" in pl.lines[1]
    assert pl.eligibility == {"wave": "ext", "rows": [42, 43, 44, 45, 46], "excluded": {"published/complete": 42, "steer/other-wave": 6}}
    assert any(l.startswith("# assert-eligible: wave=ext rows=42-46 all status=launch; excluded published/complete: 42, steer/other-wave: 6") for l in pl.summary())
    assert any("wave=ext" in l for l in pl.summary()) and not any("fitted-llama" in l for l in pl.lines + pl.summary())
    # the steer submission: one fitted class; nemotron is primary-only, so the overflow cluster takes the complement WITHIN the wave
    pl = M.plan(SHA0, "cluster_a", max_concurrent=12, wave=M.WAVE_STEER, assert_eligible_record=True)
    assert pl.rows == [47, 48, 49, 50, 51, 52] and len(pl.lines) == 1 and "--array=47-52%6" in pl.lines[0] and "-J g8k-fitted " in pl.lines[0]
    assert pl.eligibility["excluded"] == {"published/complete": 42, "ext/other-wave": 5}
    with pytest.raises(SystemExit, match="cluster_constraint primary"):
        M.plan(SHA0, "cluster_b", max_concurrent=12, wave=M.WAVE_STEER)
    assert M.plan(SHA0, "cluster_b", max_concurrent=12, wave=M.WAVE_STEER, cluster_share="51-52").rows == [47, 48, 49, 50]
    assert M.plan(SHA0, "cluster_a", max_concurrent=12, wave=M.WAVE_STEER, cluster_share="51-52").rows == [51, 52]
    assert f"--qos={SITE_B['qos']}" in M.plan(SHA0, "cluster_b", max_concurrent=12, wave=M.WAVE_STEER, rows="47-50").lines[0]
    # an index outside the wave — a published (complete) row above all — refuses by name on either selector
    with pytest.raises(SystemExit, match=r"outside wave 'ext''s eligible set: 4 \(g8k-gemma4e4b-main-none: wave published"):
        M.plan(SHA0, "cluster_a", max_concurrent=12, wave=M.WAVE_EXT, rows="4")
    with pytest.raises(SystemExit, match="outside wave 'steer'"):
        M.plan(SHA0, "cluster_a", max_concurrent=12, wave=M.WAVE_STEER, cluster_share="42")
    with pytest.raises(SystemExit, match="held"):
        M.plan(SHA0, "cluster_a", max_concurrent=12, wave=M.WAVE_EXT, rows="2")
    # a plan with NO wave over the whole matrix is the python entry point the partition tests use; the CLI requires --wave
    assert M.plan(SHA0, "cluster_a", max_concurrent=12, universe=ROWS).rows == [r["index"] for r in LAUNCHED]
    # the batch boundary: the published rows (launched OR held) are complete; a deferred / held row of a launchable wave refuses
    for r in (ROWS[4], ROWS[2], ROWS[9]):
        with pytest.raises(SystemExit, match="COMPLETE"):
            M.assert_launchable(r)
    rec = M.assert_launchable(EXT_ROWS[0])
    assert rec == {"wave": "ext", "rows": [42], "excluded": {"published/complete": 42, "ext/other-wave": 4, "steer/other-wave": 6}}
    assert M.launchable_line(EXT_ROWS[0], rec) == ("launchable: row 42 (g8k-gemma4e4b-main-prompted) wave=ext status=launch; excluded "
                                                     "published/complete: 42, ext/other-wave: 4, steer/other-wave: 6")
    assert M.assert_launchable(STEER_ROWS[-1])["rows"] == [52]
    deferred = dict(EXT_ROWS[0], status=M.STATUS_DEFERRED)
    with pytest.raises(SystemExit, match=r"assert-eligible REFUSED: row 42 \(g8k-gemma4e4b-main-prompted\) wave=ext status=deferred"):
        M.assert_launchable(deferred, [*ALL_ROWS[:42], deferred, *ALL_ROWS[43:]])
    held = dict(STEER_ROWS[0], status=M.STATUS_HELD, held_reason="synthetic")
    with pytest.raises(SystemExit, match="status=held"):
        M.assert_launchable(held, [*ALL_ROWS[:47], held, *ALL_ROWS[48:]])
    # the Llama matrix: its published twelve are complete, its two steer rows launchable (the AlphaSteer one primary-only: its
    # object is staged on the primary cluster alone); the trailing positionals name the matrix
    assert M.eligible_rows(M.WAVE_STEER, LLAMA_ALL_ROWS) == LLAMA_STEER_ROWS
    with pytest.raises(SystemExit, match="COMPLETE"):
        M.assert_launchable(LLAMA_ROWS[0], LLAMA_ALL_ROWS)
    pl = M.plan(SHA0, "cluster_a", max_concurrent=12, wave=M.WAVE_STEER, matrix=LLAMA_MATRIX)
    assert pl.rows == [12, 13] and pl.lines[0].endswith(f"ARRAY {SHA0} 8192 llama") and "-J g8k-fitted-llama " in pl.lines[0] and "--array=12-13%2" in pl.lines[0]
    with pytest.raises(SystemExit, match="cluster_constraint primary"):
        M.plan(SHA0, "cluster_b", max_concurrent=12, wave=M.WAVE_STEER, matrix=LLAMA_MATRIX)
    assert M.plan(SHA0, "cluster_b", max_concurrent=12, wave=M.WAVE_STEER, matrix=LLAMA_MATRIX, rows="13").rows == [13]
    with pytest.raises(SystemExit, match="Llama lane and of the full22"):
        M.class_tag_for(M.TIME_IN_SUPPORT, [ROWS[4], LLAMA_ROWS[0]])
    assert M.class_tag_for(M.TIME_IN_SUPPORT, [ROWS[4]]) == "in-support" and M.class_tag_for(M.TIME_FITTED_LLAMA, [LLAMA_ROWS[1]]) == "fitted-llama"
    # the CLI: `plan` requires --wave; the published wave refuses; `row --assert-launchable` is the shell boundary's call
    def cli(*args):
        r = subprocess.run([sys.executable, str(G8K / "g8k_cells.py"), *args], capture_output=True, text=True, cwd=REPO)
        return r.returncode, r.stdout, r.stderr
    rc, _, err = cli("plan", "--sha", SHA0, "--cluster", "cluster_a")
    assert rc == 2 and "--wave" in err
    rc, _, err = cli("plan", "--sha", SHA0, "--cluster", "cluster_a", "--wave", "published")
    assert rc == 1 and "COMPLETE" in err
    rc, out, _ = cli("plan", "--sha", SHA0, "--cluster", "cluster_a", "--wave", "ext", "--assert-eligible")
    assert rc == 0 and "# assert-eligible: wave=ext rows=42-46" in out and out.count("sbatch -J") == 2
    rc, out, _ = cli("plan", "--sha", SHA0, "--cluster", "cluster_a", "--wave", "steer", "--matrix", "llama")
    assert rc == 0 and out.count("sbatch -J") == 1 and f"ARRAY {SHA0} 8192 llama" in out
    rc, _, err = cli("row", "--index", "4", "--assert-launchable")
    assert rc == 1 and "COMPLETE" in err
    rc, out, _ = cli("row", "--index", "42", "--assert-launchable")
    assert rc == 0 and out.startswith("launchable: row 42 (g8k-gemma4e4b-main-prompted) wave=ext status=launch")
    rc, _, err = cli("row", "--matrix", "llama", "--index", "0", "--assert-launchable")
    assert rc == 1 and "COMPLETE" in err
    rc, out, _ = cli("row", "--matrix", "llama", "--key", "llama8b", "--view", "main", "--cell", "arditi", "--assert-launchable")
    assert rc == 0 and out.startswith("launchable: row 13 (g8k-llama8b-main-arditi) wave=steer status=launch")
    rc, out, _ = cli("list")
    assert rc == 0 and "wave=ext" in out and "wave=steer object_source=external" in out and out.count("wave=published") == 42


def test_e2_the_shell_batch_boundary_runs_g8k_cells_own_predicate_after_the_row_resolves(tmp_path):
    """The sbatch's production path calls `g8k_assert_launchable` right after `g8k_row` and before the config resolves, the out
    root exists or a venv builds; the dry-run path never calls it (a complete row dry-runs for its argv). The function runs
    g8k_cells.py's OWN predicate over the job's matrix: a published row resolved by index or by triple is FATAL by name, a
    launchable row logs the eligibility record, a held row exited HELD before the boundary."""
    src = (G8K / "gsm8k_thinking.sbatch").read_text()
    # math500-steer: g8k_init takes the row too (its serving path is selected inside, before f22_init fixes the path)
    i_row, i_assert, i_cfg = src.index('g8k_init "$WAVE_SHA" "${ROW_ARGS[@]}"'), src.index("\ng8k_assert_launchable\n"), src.index("g8k_resolve_config; g8k_out_root\nf22_stop_requested")
    assert i_row < i_assert < i_cfg and src.count("\ng8k_assert_launchable\n") == 1
    dry_block = src[src.index('if [ "${G8K_DRY_RUN:-0}" = 1 ]; then\n  # stubs'):src.index("g8k_print_plan\n  exit 0\nfi")]
    assert "g8k_assert_launchable" not in dry_block and 'g8k_row "${ROW_ARGS[@]}"' in dry_block

    def run(matrix, *row_args, array_index=None):
        env = {**os.environ, "CPY": sys.executable}
        if array_index is not None:
            env["SLURM_ARRAY_TASK_ID"] = str(array_index)
        script = (_g8k_funcs() + f"REPO_DIR={shlex.quote(str(REPO))}; G8K_MATRIX={matrix}; CLUSTER=cluster_a; cd \"$REPO_DIR\"\n"
                  f"g8k_row {' '.join(row_args)}\ng8k_assert_launchable\necho PASSED_THE_BOUNDARY\n")
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
        return r.returncode, r.stdout + r.stderr
    rc, out = run("default", "gemma4e4b", "main", "none")
    assert rc == 1 and "FATAL matrix row 4 (g8k-gemma4e4b-main-none: wave=published status=launch) is NOT launchable" in out and "COMPLETE" in out
    assert "PASSED_THE_BOUNDARY" not in out and "matrix row 4:" in out
    rc, out = run("default", "ARRAY", array_index=9)                                  # a declared published row, by index
    assert rc == 1 and f"row 9 ({ROWS[9]['job_id']}: wave=published status=launch) is NOT launchable" in out and ROWS[9]["declared"]
    rc, out = run("default", "ARRAY", array_index=42)
    assert rc == 0 and "PASSED_THE_BOUNDARY" in out and "wave=ext" in out
    assert "LOG launchable: row 42 (g8k-gemma4e4b-main-prompted) wave=ext status=launch; excluded published/complete: 42, ext/other-wave: 4, steer/other-wave: 6" in out
    rc, out = run("default", "gemma4e4b", "main", "alphasteer")
    assert rc == 0 and "PASSED_THE_BOUNDARY" in out and "object_source=external bank_view=it" in out and "launchable: row 47" in out
    rc, out = run("default", "ARRAY", array_index=2)                                  # HELD: exits 0 from g8k_row, never reaches the boundary
    assert rc == 0 and f"HELD: {ROWS[2]['job_id']}" in out and ROWS[2]["status"] == "held" and "launchable" not in out and "PASSED_THE_BOUNDARY" not in out
    rc, out = run("llama", "llama8b", "main", "none")
    assert rc == 1 and "matrix row 0 (g8k-llama8b-main-none: wave=published status=launch) is NOT launchable" in out
    rc, out = run("llama", "ARRAY", array_index=13)
    assert rc == 0 and "launchable: row 13 (g8k-llama8b-main-arditi) wave=steer status=launch; excluded published/complete: 12, steer/other-wave: 1" in out
    assert "object_source=external bank_view=ar_r0p05" in out          # the ruled alpha-0.05 view (2026-09-14), not the superseded ar_r0p1
    rc, out = run("llama", "ARRAY", array_index=12)
    assert rc == 0 and "launchable: row 12 (g8k-llama8b-main-alphasteer) wave=steer status=launch" in out and "bank_view=it" in out


def test_e2_the_ext_rows_dry_run_at_the_full22_long_si_geometry_on_the_panel_view(tmp_path):
    """E2 / plan P3: the constitution prompted rows of the four full22 families run at the class's OWN measured entries
    (`capability:gen_long_gsm8k_longsi_f22` 16,640 / `..._rung_longsi_f22` 24,832; gemma 8,338 / qwen 8,332 / nemotron 8,288
    prompt max, pooled n = 600), never PR-A's Llama-scoped ones; the safety prompted rows at the shared short-class entries;
    every ext row on the PANEL view `main` (`--bank-view main`, the campaign's own bank), the TAG overlay, the `controls_ext/`
    root, the fitted-class --time on the constitution prefill; a long-SI cell with no model key refuses (the scope is part of
    the selection)."""
    with pytest.raises(SystemExit, match="scoped by family — pass model_key"):
        M.geometry_for_cell(M.CELL_PROMPTED_CONST, M.CAP)
    assert M.geometry_for_cell(M.CELL_PROMPTED_CONST, M.CAP, model_key="gemma4e4b") == ("capability:gen_long_gsm8k_longsi_f22", 16640)
    assert M.geometry_for_cell(M.CELL_PROMPTED_CONST, M.RUNG_CAP, model_key="qwen35_4b") == ("capability:gen_long_gsm8k_rung_longsi_f22", 24832)
    assert M.geometry_for_cell(M.CELL_PROMPTED_CONST, M.CAP, model_key="llama8b") == ("capability:gen_long_gsm8k_longsi", 16384)
    assert M.geometry_for_cell(M.CELL_PROMPTED, M.CAP, model_key="gemma4e4b") == M.geometry_for_cell(M.CELL_NONE, M.CAP) == ("capability:gen_long_thinking", 10752)
    assert {M.longsi_scope(k) for k in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash")} == {M.LONGSI_SCOPE_F22} and M.longsi_scope("llama8b") == M.LONGSI_SCOPE_LLAMA
    from sisteer.capeval.serving import geometry as G
    ent = {t: G.geometry_for("capability", t) for t in ("gen_long_gsm8k_longsi_f22", "gen_long_gsm8k_rung_longsi_f22")}
    assert {k: v.max_model_len for k, v in ent.items()} == {"gen_long_gsm8k_longsi_f22": 16640, "gen_long_gsm8k_rung_longsi_f22": 24832}
    assert all(set(v.basis.families_measured) == {"gemma4e4b", "qwen35_4b", "nemotron3nano4b"} and v.basis.max_prompt == 8338 and v.basis.n_calls == 600 for v in ent.values())
    for r in EXT_ROWS:
        key, const = r["model_key"], r["cell"] == M.CELL_PROMPTED_CONST
        s_cap, s_rung = F22_LONGSI_S[key] if const else (DRY.resolved_s(key, M.CAP), DRY.resolved_s(key, M.RUNG_CAP))
        assert (DRY.resolved_s(key, M.CAP, r["cell"]), DRY.resolved_s(key, M.RUNG_CAP, r["cell"])) == (s_cap, s_rung)
        text, rc, raw = _dry(r, s=s_cap, shared=tmp_path / "shared")
        assert rc == 0, raw[-800:]
        d, st = _line(text, "DRIVER"), _line(text, "STANDUP")
        assert _line(text, "WAVE") == (f"WAVE: wave=ext status=launch object_source=none bank_view=none bank_view_rel_primary=none "
                                       f"bank_view_rel_overflow=none bank_cell_dir=none config=configs/sdfbase/gsm8k/sdfbase_full22_{key}.gsm8k.yaml")
        assert f"{M.FLAG_BANK_VIEW} main" in d and f"{M.FLAG_THINKING} on " in d and f"--cell {r['cell']} " in d and f"--cell {r['cell']}" in st
        assert f"--config configs/sdfbase/gsm8k/sdfbase_full22_{key}.gsm8k.yaml " in d and f"/exp/gsm8k_paired200_paper_objects/controls_ext/{key}/{r['cell_slug']}/thinking_card " in d
        assert f"--defenses-dir {tmp_path / 'shared'}/exp/objects/sdfbase/full22/{key}/cells_view/main " in d and M.FLAG_EXPECT_OBJECT_SHA256 not in d
        # E2-B: a prompted (non-external) row passes NO liveness-gate flag and prints no SELECTION line — its argv is byte-identical to before
        assert M.FLAG_LENS_LIVENESS not in d and not [l for l in text.splitlines() if l.startswith("SELECTION:")]
        assert "IN SUPPORT: route=prompted stamps=(cell-stable,not-consulted,thinking_on)" in text and "OUT OF SUPPORT" not in text
        assert f"time={'0-01:30:00' if const else '0-01:00:00'} placement={r['cluster_constraint']}" in _line(text, "ROW") and "matrix=default thinking=on" in _line(text, "ROW")
        geom = "capability:gen_long_gsm8k_longsi_f22 16640" if const else "capability:gen_long_thinking 10752"
        assert f"GEOMETRY: {geom} x {s_cap} cap 8192 thinking on" in text and "BANK: " in text and "pin=none" in _line(text, "BANK")
        # the rung: the class's rung entry, the setter at 16,384, the subset under controls_ext's rung root
        text, rc, raw = _dry(r, rung=M.RUNG_CAP, s=s_rung, shared=tmp_path / "shared")
        assert rc == 0, raw[-800:]
        d = _line(text, "DRIVER")
        geom = "capability:gen_long_gsm8k_rung_longsi_f22 24832" if const else "capability:gen_long_gsm8k_rung 18944"
        assert f"GEOMETRY: {geom} x {s_rung} cap 16384 thinking on" in text and f"{M.FLAG_ON_CAP_BY_TASK} gsm8k_paired200=16384 " in d
        assert f"{M.FLAG_ITEM_SUBSET} " in d and f"/controls_ext/{key}/{r['cell_slug']}/thinking_card__rung16384 " in d and "RUNG_SUBSET: " in text
        assert f"/exp/gsm8k_paired200_paper_objects/rung16384/{key}/{r['cell_slug']}/" in text
    # the primary-only ext row (nemotron) refuses off the primary cluster at job start
    nemo = next(r for r in EXT_ROWS if r["model_key"] == "nemotron3nano4b")
    text, rc, raw = _dry(nemo, s=47, shared=tmp_path / "shared", cluster="cluster_b")
    assert rc != 0 and "PLACEMENT" in raw and "DRIVER:" not in text
    # the ladder planner sizes a full22 constitution rung at the cell's OWN f22 entry (planner S == launcher S)
    assert {k: LP.s_rung_for(k, M.CELL_PROMPTED_CONST) for k in F22_LONGSI_S} == {k: v[1] for k, v in F22_LONGSI_S.items()}
    assert LP.s_rung_for("gemma4e4b", M.CELL_NONE) == DRY.resolved_s("gemma4e4b", M.RUNG_CAP) == 31 and LP.s_rung_for("llama8b", M.CELL_PROMPTED_CONST) == 19


def test_e2_the_steer_rows_dry_run_through_their_own_overlay_and_the_external_lane_bank(tmp_path, s_by):
    """E2 / P8 through the launcher's own shell: a steering row reads its OWN overlay pair (never the TAG's), binds the EXTERNAL
    lane's view for the cluster role it runs on (the primary-role rel on the primary cluster, the overflow-role rel on the
    overflow cluster), holds the object to the admission pin at the standup (`--expect-object-sha256 <cell>=<pin>`) and the
    driver, runs on the PANEL view (`--bank-view main`) under the rung's `_steer_<cap>` tuple (a switch model: out of support,
    the override), under `controls_steer/`; a primary-only row refuses off the primary cluster; the Llama rows are no-switch
    (`not_available`, `card`, in support, no tuple), the AlphaSteer one primary-only."""
    for r in STEER_ROWS:
        key, cell, pin = r["model_key"], r["cell"], r["object_sha256"]
        rel_p, rel_o = r["bank_view_rel"]["primary"], r["bank_view_rel"]["overflow"]
        text, rc, raw = _dry(r, s=s_by[(key, M.CAP)], shared=tmp_path / "shared")
        assert rc == 0, raw[-800:]
        d, st = _line(text, "DRIVER"), _line(text, "STANDUP")
        assert _line(text, "WAVE") == (f"WAVE: wave=steer status=launch object_source=external bank_view={r['bank_view']} bank_view_rel_primary={rel_p} "
                                       f"bank_view_rel_overflow={rel_o} bank_cell_dir={r['bank_cell_dir']} config=configs/sdfbase/gsm8k/sdfbase_full22_{key}.gsm8k_{cell}.yaml")
        assert _line(text, "BANK") == f"BANK: {tmp_path / 'shared'}/{rel_p}/{r['bank_cell_dir']} pin={pin}"
        assert f"--cell {cell} {M.FLAG_EXPECT_OBJECT_SHA256} {cell}={pin}" in st and "@thinking" not in st
        assert f"--config configs/sdfbase/gsm8k/sdfbase_full22_{key}.gsm8k_{cell}.yaml " in d and f"sdfbase_full22_{key}.gsm8k.yaml" not in d
        assert f"--defenses-dir {tmp_path / 'shared'}/{rel_p} " in d and f"--cell {cell} " in d and f"{M.FLAG_EXPECT_OBJECT_SHA256} {pin}" in d
        assert f"/exp/gsm8k_paired200_paper_objects/controls_steer/{key}/{cell}/thinking_card " in d and f"{M.FLAG_BANK_VIEW} main" in d
        assert f"{M.FLAG_THINKING} on " in d and f"{M.FLAG_SAMPLING} thinking_card " in d and M.FLAG_ALLOW_OOS_THINKING in d
        assert f"OUT OF SUPPORT: tuple=gsm8k_paired200_{key}_steer_8192 pin={pin} served_positions_expected=all declared=0 stamps=(cell-stable,always,thinking_on)" in text
        assert f"GEOMETRY: capability:gen_long_thinking 10752 x {s_by[(key, M.CAP)]} cap 8192 thinking on" in text
        assert f"job_id=g8k-{key}-main-{cell} key={key} view=main tag={key} cell={cell} rung=8192 fitted=1 declared=0 route=lens" in _line(text, "ROW")
        assert "time=0-01:30:00" in _line(text, "ROW")
        # E2-B (decisions-e2b LB1 G4): the LOGPROB liveness gate rides EXACTLY the fitted rows serving an EXTERNAL lane's object —
        # once, the contracts' spelling — so a small correct edit is never refused as "inert" by greedy-text divergence
        assert f" {M.FLAG_LENS_LIVENESS} {M.LENS_LIVENESS_STEER} " in d + " " and d.count(M.FLAG_LENS_LIVENESS) == 1
        # E2-B: the ladder's selection outcome + the grid-limited flag beside the verbatim selection label, on the row and in the plan
        outcome = r["selection_outcome"]
        assert (outcome in M.SELECTION_OUTCOMES) is (cell == M.CELL_ALPHASTEER) and (outcome is None) is (cell == M.CELL_ARDITI)
        assert _line(text, "SELECTION") == f"SELECTION: outcome={outcome or 'none'} grid_limited={int(r['grid_limited'])} label={r['selection_label']}"
        # the ACTIVE points since the 2026-09-14 rulings (decisions-e2b LB10 / LB20): the oracle's `ruled_` word on qwen and
        # nemotron, its `ruled_…_second` word on gemma (lam2 by the SECOND ruling of that day, which moved off the first ruling's
        # lam3); a ruled point is never grid-limited (it does not move on a judge flip), so every AlphaSteer row prints grid_limited=0
        # now — the superseded `a4_fallback` (nemotron lam1, grid-limited) lives under the entry's `superseded` block, not on the row
        if cell == M.CELL_ALPHASTEER:
            assert outcome == {"gemma4e4b": M.SELECTION_OUTCOME_RULED_SECOND}.get(key, M.SELECTION_OUTCOME_RULED) and "grid_limited=0" in _line(text, "SELECTION")
    # the overflow cluster binds the OVERFLOW-role rel; since the 2026-09-14 re-pin (decisions-e2b LB10 (i)) both roles name the rebake's
    # `ladder_canon/lam<rung>/cells_view/<view>` — the ruled / candidate rung is staged under `ladder_canon/` on BOTH clusters (the overflow's
    # `ladder/` tree is a recorded-superseded reclaim candidate, never a bank path) — so the two rels are equal and carry the row's rung
    gem = next(r for r in STEER_ROWS if r["model_key"] == "gemma4e4b" and r["cell"] == "alphasteer")
    text, rc, raw = _dry(gem, s=s_by[("gemma4e4b", M.CAP)], shared=tmp_path / "shared", cluster="cluster_b")
    assert rc == 0 and f"--defenses-dir {tmp_path / 'shared'}/{gem['bank_view_rel']['overflow']} " in _line(text, "DRIVER")
    assert gem["bank_view_rel"]["overflow"] == gem["bank_view_rel"]["primary"] and f"/ladder_canon/lam{gem['rung']}/cells_view/{gem['bank_view']}" in gem["bank_view_rel"]["overflow"]
    assert gem["rung"] == 2 and "/ladder/" not in gem["bank_view_rel"]["overflow"]
    for r in STEER_ROWS:
        if r["model_key"] == "nemotron3nano4b":
            text, rc, raw = _dry(r, s=s_by[("nemotron3nano4b", M.CAP)], shared=tmp_path / "shared", cluster="cluster_b")
            assert rc != 0 and "PLACEMENT" in raw and "DRIVER:" not in text
    # the rung of a steer row: the `_steer_16384` tuple, the rung root, the subset
    text, rc, raw = _dry(gem, rung=M.RUNG_CAP, s=s_by[("gemma4e4b", M.RUNG_CAP)], shared=tmp_path / "shared")
    assert rc == 0 and "OUT OF SUPPORT: tuple=gsm8k_paired200_gemma4e4b_steer_16384 " in text and "/controls_steer/gemma4e4b/alphasteer/thinking_card__rung16384 " in _line(text, "DRIVER")
    assert "RUNG_SUBSET: " in text and f"{M.FLAG_ON_CAP_BY_TASK} gsm8k_paired200=16384 " in _line(text, "DRIVER")
    # the Llama steer rows (the llama matrix): no-switch, in support, their own overlay pair, the Llama root
    for r in LLAMA_STEER_ROWS:
        cell, pin = r["cell"], r["object_sha256"]
        text, rc, raw = _dry_llama(r, s=44, shared=tmp_path / "shared")
        assert rc == 0, raw[-800:]
        d = _line(text, "DRIVER")
        assert f"{M.FLAG_THINKING} not_available " in d and f"{M.FLAG_SAMPLING} card " in d and M.FLAG_ALLOW_OOS_THINKING not in d
        assert f"--config configs/sdfbase/gsm8k/sdfbase_full22_llama8b.gsm8k_{cell}.yaml " in d and f"{M.FLAG_EXPECT_OBJECT_SHA256} {pin}" in d
        assert f"/exp/gsm8k_paired200_paper_objects_llama/controls_steer/llama8b/{cell}/card " in d and f"--defenses-dir {tmp_path / 'shared'}/{r['bank_view_rel']['primary']} " in d
        assert f"IN SUPPORT: route=lens pin={pin} thinking=not_available served_positions_expected=all stamps=(cell-stable,always,unspecified)" in text
        assert "OUT OF SUPPORT" not in text and "matrix=llama thinking=not_available" in _line(text, "ROW") and "time=0-01:00:00" in _line(text, "ROW")
        assert f"GEOMETRY: capability:gen_long_thinking 10752 x 44 cap 8192 thinking not_available" in text and "SAMPLING: mode=card params=" in text
        # E2-B: the no-switch steer rows take the same LOGPROB liveness gate (once) and print the same SELECTION line (llama AlphaSteer:
        # lam1 CONFIRMED by the 2026-09-14 ruling — the oracle's `chosen_confirmed_` word)
        assert f" {M.FLAG_LENS_LIVENESS} {M.LENS_LIVENESS_STEER} " in d + " " and d.count(M.FLAG_LENS_LIVENESS) == 1
        assert _line(text, "SELECTION") == f"SELECTION: outcome={r['selection_outcome'] or 'none'} grid_limited=0 label={r['selection_label']}"
        # llama Arditi: the rule of 2026-09-14 moved the point (alpha 0.1 -> 0.05), so the row prints the ruled word too — the one
        # non-null outcome an Arditi row may carry (g8k_cells.ARDITI_SELECTION_OUTCOMES); the plan's SELECTION line reads it off the row
        assert r["selection_outcome"] == (M.SELECTION_OUTCOME_CHOSEN_CONFIRMED if cell == M.CELL_ALPHASTEER else M.SELECTION_OUTCOME_RULED)
        if cell == M.CELL_ARDITI:
            assert "outcome=ruled_2026-09-14" in _line(text, "SELECTION") and pin == "1755a5627a18f931b8f125bcebe52f399da3d7507128de60ff6f55dba717a0b3"
    alpha, ard = LLAMA_STEER_ROWS
    text, rc, raw = _dry_llama(alpha, s=44, shared=tmp_path / "shared", cluster="cluster_b")
    assert rc != 0 and "PLACEMENT" in raw                          # primary by cluster_constraint (the overflow rel is bound but the constraint refuses first)
    text, rc, raw = _dry_llama(ard, s=44, shared=tmp_path / "shared", cluster="cluster_b", array_index=13)
    assert rc == 0 and f"--defenses-dir {tmp_path / 'shared'}/{ard['bank_view_rel']['overflow']} " in _line(text, "DRIVER") and "job_id=g8k-llama8b-main-arditi" in _line(text, "ROW")


@pytest.mark.parametrize("cell", ["alphasteer", "arditi"])
def test_e2_a_steer_row_serves_its_own_overlay_pair_and_the_external_bank_end_to_end(tmp_path, monkeypatch, capsys, cell):
    """The chain a GSM8K steering leg runs, on a synthetic external bank (the MATH-500 launcher's test, on this lane's seams):
    the row's OWN overlay pair lists exactly `[none, <cell>]` and the launcher's resolver (`g8k_resolve_config`) picks it, the
    standup's REAL `select_cells` admits the cell (the TAG overlay refuses it), the REAL registry resolves the object from the
    external view, `g8k_bank_view` DISPATCHES on the object source — the external check (pin held, view + role + manifest
    refusals) for a steer row, the campaign check (three args fitted, one in support) for every other row — and the prod-stage
    dry run exports the fresh table over the external view through `--shared-root` (or the root derived from `--bank-root`),
    refusing prod without it."""
    import dataclasses

    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.petri.registry_adapter import traintest_cell_source
    mt = _load("_g8k_steer_mt", "scripts/sdfbase/mt_serving_up.py")
    row = M.row_for("gemma4e4b", "main", cell, ALL_ROWS)
    cfg_path = REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_gemma4e4b.gsm8k_{cell}.yaml"
    man_path = REPO / f"configs/capeval/manifests/petri_full22_gemma4e4b.gsm8k_{cell}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["explicit_cells"] == ["none", cell] and cfg["model"] == yaml.safe_load((REPO / "configs/sdfbase/gsm8k/sdfbase_full22_gemma4e4b.gsm8k.yaml").read_text())["model"]
    spec0 = cm.load_petri_spec(man_path)
    assert list(spec0.cells) == ["none", ra.harness_cell_id(cell)] and mt.assert_manifest_matches_config(spec0, cfg) == {"none": "none", cell: cell}
    assert mt.select_cells(cfg["explicit_cells"], [cell]) == ["none", cell]
    with pytest.raises(SystemExit, match="never enumerated"):
        mt.select_cells(yaml.safe_load((REPO / "configs/sdfbase/gsm8k/sdfbase_full22_gemma4e4b.gsm8k.yaml").read_text())["explicit_cells"], [cell])
    # every steer row of both matrices has its pair, generated (make_gsm8k_overlays --check passes above); no TAG overlay lists a steer cell
    for r in STEER_ROWS + LLAMA_STEER_ROWS:
        assert (REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_{r['model_key']}.gsm8k_{r['cell']}.yaml").is_file()
        assert (REPO / f"configs/capeval/manifests/petri_full22_{r['model_key']}.gsm8k_{r['cell']}.yaml").is_file()
    for tag in COUNTS["tags"] + ["llama8b", "llama8b_it"]:
        assert not set(M.STEER_CELLS) & set(yaml.safe_load((REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_{tag}.gsm8k.yaml").read_text())["explicit_cells"])

    def sh(body, **env):
        r = subprocess.run(["bash", "-c", _g8k_funcs() + f"REPO_DIR={shlex.quote(str(REPO))}; cd \"$REPO_DIR\"; CPY=/venv/bin/python\n" + body],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src"), **env})
        return r.returncode, r.stdout + r.stderr
    row_sh = M.row_sh(row)
    rc, out = sh(row_sh + "g8k_resolve_config; echo CFG=$CFG; echo MANIFEST=$MANIFEST")
    assert rc == 0 and f"CFG={cfg_path.relative_to(REPO)}" in out and f"MANIFEST={man_path.relative_to(REPO)}" in out, out
    rc, out = sh(M.row_sh(ROWS[5]) + "g8k_resolve_config; echo CFG=$CFG")
    assert rc == 0 and "CFG=configs/sdfbase/gsm8k/sdfbase_full22_gemma4e4b.gsm8k.yaml" in out
    # the REAL registry over the external view: the cell resolves, its kind is the blob's, its path is the staged object
    bank_root = tmp_path / "shared"
    view, obj = _stage_external_bank_g8k(bank_root, row)
    spec = dataclasses.replace(spec0, serving_launch_flags=[], cells=[ra.harness_cell_id(cell)])
    entry = traintest_cell_source(spec, repo_root=REPO, config_path=cfg_path, defenses_dir=view).get(cell)
    assert entry is not None and entry.kind == ra.ARM_OBJECT_KINDS[cell] and entry.serving_path == "lens" and Path(entry.object_path) == obj
    sha = _sha256(obj)
    # g8k_bank_view: the dispatch, through the REAL check functions (f22's stubbed to echo its argv — its own tests hold it)
    stub = 'f22_bank_view_check(){ echo F22CHECK "$#" "$@"; }\n'
    rc, out = sh(row_sh + f"PIN={sha}\nSHARED={shlex.quote(str(bank_root))}; CLUSTER=cluster_a\n" + stub + "g8k_bank_view; echo DIR=$TT_TWO_SI_DEFENSES_DIR")
    assert rc == 0 and "external bank view OK" in out and f"DIR={view}" in out and "F22CHECK" not in out, out
    assert f"LOG external bank: lane view {row['bank_view']} (the panel view is main); selection: {row['selection_label'][:40]}" in out
    rc, out = sh(row_sh + f"PIN={'f' * 64}\nSHARED={shlex.quote(str(bank_root))}; CLUSTER=cluster_a\n" + stub + "g8k_bank_view")
    assert rc == 1 and "!= the admission record's pin" in out
    rc, out = sh(row_sh + f"PIN={sha}\nSHARED={shlex.quote(str(bank_root))}; CLUSTER=cluster_b\n" + stub + "g8k_bank_view")
    if row["bank_view_rel"]["overflow"] != row["bank_view_rel"]["primary"]:          # the AlphaSteer rebake's `ladder/` tree on the overflow cluster
        assert rc == 1 and "does not exist on this cluster" in out and row["bank_view_rel"]["overflow"] in out
    else:                                                                             # the Arditi lane stages the SAME rel on both clusters
        assert rc == 0 and "external bank view OK" in out
    rc, out = sh(row_sh + f"PIN={sha}\nSHARED={shlex.quote(str(bank_root))}; CLUSTER=cluster_b; G8K_BANK_VIEW_REL_OVERFLOW=\n" + stub + "g8k_bank_view")
    assert rc == 1 and "names no bank view for cluster cluster_b" in out and "staged on the other cluster only" in out
    rc, out = sh(M.row_sh(ROWS[5]) + f"SHARED={shlex.quote(str(bank_root))}; CLUSTER=cluster_a\n" + stub + "g8k_bank_view")
    assert rc == 0 and f"F22CHECK 3 /venv/bin/python {ROWS[5]['cell']} {ROWS[5]['object_sha256']}" in out and "external" not in out   # a fitted campaign row
    rc, out = sh(M.row_sh(ROWS[4]) + f"SHARED={shlex.quote(str(bank_root))}; CLUSTER=cluster_a\n" + stub + "g8k_bank_view")
    assert rc == 0 and "F22CHECK 1 /venv/bin/python\n" in out                                                                        # an in-support row
    assert (G8K / "g8k_common.sh").read_text().count("st_external_bank_view_check ") == 1
    # g8k_external_mirror_check (E2-12): the DRIVER's own provenance mirror over the external object, before the standup — the
    # synthetic sidecar's writer is unknown to the mirror (the Arditi lane's real shape), a `fit_class_b.py` writer
    # passes, a record bound to OTHER bytes refuses (llama's AlphaSteer view); a campaign row is a no-op
    sbatch_src = (G8K / "gsm8k_thinking.sbatch").read_text()
    assert "\ng8k_bank_view\ng8k_external_mirror_check\n" in sbatch_src and sbatch_src.count("g8k_external_mirror_check") == 2
    from sisteer.eval import sweep as SW
    sidecar = obj.parent / "fit_provenance.json"
    base_rec = json.loads(sidecar.read_text())
    mirror = f"PIN={sha}\nSHARED={shlex.quote(str(bank_root))}; CLUSTER=cluster_a; CPY={shlex.quote(sys.executable)}\n" + stub + "g8k_bank_view; g8k_external_mirror_check"
    rc, out = sh(row_sh + mirror)
    assert rc == 1 and "the driver's provenance mirror REFUSES it" in out and "UnrecordedMirrorError" in out, out
    assert "is not in sweep.SI_PAIR_FIT_WRITERS or sweep.CLASS_B_FIT_WRITERS and the record carries no budget schema" in out
    sidecar.write_text(json.dumps({**base_rec, "writer": f"{SW.CLASS_B_FIT_WRITERS[0]}@" + "1" * 40}))
    rc, out = sh(row_sh + mirror)
    assert rc == 0 and f"LOG provenance mirror OK: status=verified fit_kind={SW.FIT_KIND_CLASS_B}" in out and f"object_sha={sha[:12]}" in out, out
    sidecar.write_text(json.dumps({**base_rec, "writer": f"{SW.CLASS_B_FIT_WRITERS[0]}@" + "1" * 40, "object_sha256": "6" * 64}))
    rc, out = sh(row_sh + mirror)
    assert rc == 1 and "records object_sha256 666666666666" in out and "a stale object/sidecar pair" in out, out
    sidecar.write_text(json.dumps(base_rec))
    rc, out = sh(M.row_sh(ROWS[5]) + f"SHARED={shlex.quote(str(bank_root))}; CLUSTER=cluster_a\n" + stub + "g8k_bank_view; g8k_external_mirror_check; echo NOOP_OK")
    assert rc == 0 and "NOOP_OK" in out and "provenance mirror" not in out
    # the two-stage dry run over the external view: prod exports the fresh table through the row's OWN pair over the lane's view
    # and holds the stamp to the row's `served_positions_expected` — and REFUSES on the real blobs' shape (E2-7): the resolved spec
    # of a `positions`-less blob states no served mode, the table stamps no fact for the cell, and the thinking-ON proxy reads an
    # absent fact as header-located (fail closed: `thinking_on_undeclared` on every undeclared steered request). The gate is the
    # pre-launch catch for the switch-model steer rows; the row's expectation (`all`, the adapter's prefill-only broadcast) is
    # right and the STAMP is what has to land before these rows are launchable
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", _facts("certified"), raising=False)
    monkeypatch.setattr(C, "check_render_fact_certified", lambda family: None, raising=False)
    # THE EXPECTATION IS THE LIBRARY'S (E2-11): what `defense_cells.served_positions_of` reads off the real registry's resolved
    # spec for these bytes — None while the blobs state no `positions` (the table stamped no fact and the prod dry run
    # REFUSED), `all` since the served mode is derived from `steer_positions` (the adapters' prefill-only broadcast) — so the
    # verdict below follows the library, never a retyped answer
    from sisteer.capeval.petri import defense_cells as dc
    stamped = dc.served_positions_of(entry.resolved_spec)
    assert stamped in (None, "all"), stamped
    only = ["--only", "gemma4e4b", "main", cell, "--tmp", str(tmp_path / "t"), "--python", PY]
    want_rc = 0 if stamped == "all" else 2
    assert DRY.main(["--stage", "prod", *only, "--shared-root", str(bank_root)]) == want_rc
    out = capsys.readouterr().out
    if stamped is None:
        assert f"FRESH_TABLE: row={row['index']} bank={view} routes={sorted([cell, 'none', 'none__lens0'])} own_route=True lens_served_positions={{}} " in out
        assert f"REFUSED: lens_served_positions[{cell}] = None, not ['all'] (row {row['index']}: served_positions_expected=all)" in out and "VERDICT: PASS" not in out
    else:
        assert f'SERVED_POSITIONS: row={row["index"]} lens_served_positions[{cell}] == "all" (expected)' in out and "VERDICT: PASS stage=prod" in out
    assert f"WAVE_STATUS: row={row['index']} wave=steer status=launch object_source=external" in out
    assert f"TUPLE: row={row['index']} gsm8k_paired200_gemma4e4b_steer_8192 rung=gsm8k_paired200_gemma4e4b_steer_16384" in out
    assert 'WAVES: {"steer": 1} complete=[\'published\']' in out and '"launched_by_wave": {"ext": 5, "published": 38, "steer": 6}' in out   # the selection's waves; the matrix's counts
    # the shared root derived from a --bank-root under it (the campaign banks' fixed layout): the same export, the same verdict
    assert DRY.main(["--stage", "prod", *only, "--bank-root", str(bank_root / "exp/objects/sdfbase/full22")]) == want_rc
    assert f"FRESH_TABLE: row={row['index']} bank={view}" in capsys.readouterr().out
    # prod without a shared root refuses by name; pre prints the miss and passes (the stamp is a prod-stage fact)
    assert DRY.main(["--stage", "prod", *only]) == 2
    out = capsys.readouterr().out
    assert f"FRESH_TABLE: row={row['index']} not exported (no --shared-root" in out and "REFUSED: --stage prod requires the fresh-table check" in out
    assert DRY.main(["--stage", "pre", *only, "--shared-root", str(bank_root)]) == 0
    out = capsys.readouterr().out
    assert "VERDICT: PASS stage=pre" in out and f"FRESH_TABLE: row={row['index']} bank={view}" in out
    assert ("lens_served_positions={} " in out) is (stamped is None) and (f'lens_served_positions[{cell}] == "all"' in out) is (stamped == "all")
    # a blob that DOES state the served mode (the fix's shape, whichever layer stamps it) passes prod with the expected `all`
    import torch
    blob = torch.load(obj, map_location="cpu", weights_only=True)
    torch.save({**blob, "positions": "all"}, obj)
    (obj.parent / "fit_provenance.json").write_text(json.dumps({**json.loads((obj.parent / "fit_provenance.json").read_text()), "positions_at_fit": "all"}))
    assert DRY.main(["--stage", "prod", *only, "--shared-root", str(bank_root)]) == 0
    out = capsys.readouterr().out
    assert f'SERVED_POSITIONS: row={row["index"]} lens_served_positions[{cell}] == "all" (expected)' in out and "VERDICT: PASS stage=prod" in out
    sha = _sha256(obj)
    # a primary-only steer row on the overflow cluster: refused by name, no export attempted
    assert DRY.main(["--stage", "pre", "--only", "nemotron3nano4b", "main", cell, "--cluster", "cluster_b", "--tmp", str(tmp_path / "t2"), "--python", PY]) == 0
    out = capsys.readouterr().out
    assert "PLACEMENT: row=" in out and "refused on cluster_b by name (primary-only) — as designed" in out and "FRESH_TABLE" not in out
    # a drifted object: the fresh table still exports (the registry hashes bytes, it holds no pin) but the launcher's check refuses first
    obj.write_bytes(obj.read_bytes() + b"\0")
    rc, out = sh(row_sh + f"PIN={sha}\nSHARED={shlex.quote(str(bank_root))}; CLUSTER=cluster_a\n" + stub + "g8k_bank_view")
    assert rc == 1 and "!= the admission record's pin" in out and "a drifted or re-staged object" in out


def test_e2_the_pull_carries_the_three_attestation_files_from_every_cluster_root_and_never_deletes(tmp_path):
    """The GSM8K lane's pull names ONE wave subtree (`controls_ext` / `controls_steer`; the
    published `controls` + overlap by default), rides the shared per-model inputs along, and carries EXACTLY three files per
    standup dir — `serving/<jobdir>/gsm8k/{steering_attestation.json,vllm_server.log,serving_config.json}` — from the primary
    root AND from every `--serving-src` root (the published pull is cross-cluster: 6 of its 42 manifests name the overflow
    root), one rsync per host with the same anchored filter chain; `--dry-run` itemizes and writes nothing; never --delete;
    the script carries no host name. Asserted through --print-argv (the chain's EFFECT is the real-rsync test below).
    Since native-positions-resolve the chain ALSO names the standup records both served-positions resolutions read (the declared
    cell's STANDUP_RECORD_FILES, the native cell's NATIVE_STANDUP_RECORD_FILES), each by its own explicit include."""
    script = G8K / "pull_wave.sh"
    env = {**os.environ, "G8K_PULL_SRC": "TESTHOST:/wave/root", "G8K_PULL_DEST_ROOT": str(tmp_path)}

    def argv(*args, **kw):
        r = subprocess.run(["bash", str(script), "--print-argv", *args], capture_output=True, text=True, env={**env, **kw})
        transfers = [t.splitlines() for t in r.stdout.split("---\n")] if r.returncode == 0 else []
        return r.returncode, transfers, r.stderr
    rc, tr, _ = argv("--subtree", "controls_steer", "--serving-src", "OTHER:/wave/root", "abc1234")
    assert rc == 0 and len(tr) == 2
    for words in tr:
        assert words[0] == "rsync" and "--delete" not in words and "-az" in words and "--ignore-missing-args" in words and "--prune-empty-dirs" in words
        assert words[-1] == f"{tmp_path}/pull_abc1234/" and "-n" not in words and "-i" not in words
        filt = [w for w in words if w.startswith(("--include=", "--exclude="))]
        chain = ["--include=/serving/", "--include=/serving/*/", "--include=/serving/*/gsm8k/",
                 "--include=/serving/*/gsm8k/steering_attestation.json", "--include=/serving/*/gsm8k/vllm_server.log",
                 "--include=/serving/*/gsm8k/serving_config.json",
                 # native-positions-resolve: the standup records BOTH served-positions resolutions read — the declared cell's
                 # STANDUP_RECORD_FILES and the native cell's NATIVE_STANDUP_RECORD_FILES — by name; the lens registry's
                 # directory is admitted so that its ONE file can be
                 "--include=/serving/*/gsm8k/routing_table.json", "--include=/serving/*/gsm8k/standup_jobs.jsonl",
                 "--include=/serving/*/gsm8k/serving_attestation.json",
                 "--include=/serving/*/gsm8k/lens_registry/", "--include=/serving/*/gsm8k/lens_registry/registry.json",
                 "--exclude=/serving/**", "--exclude=serving"]
        assert filt[-len(chain):] == chain and filt.index("--exclude=*-superseded") < filt.index("--include=/serving/")
        # every record file either resolution path reads has its OWN explicit include, pinned from the validators' constants
        # (a subdirectory file needs its directory admitted first); no include is a wildcard over files
        includes = {w for w in filt if w.startswith("--include=")}
        record_files = set(VAL.STANDUP_RECORD_FILES) | set(VAL.NATIVE_STANDUP_RECORD_FILES) | {"vllm_server.log", "serving_config.json"}
        assert record_files == {"routing_table.json", "standup_jobs.jsonl", "steering_attestation.json", "lens_registry/registry.json",
                                "serving_attestation.json", "vllm_server.log", "serving_config.json"}
        for f in record_files:
            assert f"--include=/serving/*/gsm8k/{f}" in includes, f
            if "/" in f:
                assert f"--include=/serving/*/gsm8k/{f.rsplit('/', 1)[0]}/" in includes, f
        assert not any(w.endswith(("/*", "/**", "*.json", "*.jsonl", "*.log")) for w in includes)
        # the KNOWN-BAD: the pre-change three-file chain lacks every standup-record file, so this pin would have failed on it
        old_chain = chain[:6] + chain[-2:]
        missing = [f for f in set(VAL.STANDUP_RECORD_FILES) | set(VAL.NATIVE_STANDUP_RECORD_FILES) if f"--include=/serving/*/gsm8k/{f}" not in old_chain]
        assert sorted(missing) == ["lens_registry/registry.json", "routing_table.json", "serving_attestation.json", "standup_jobs.jsonl"]
        assert {"--exclude=*-superseded", "--exclude=*-failed-probe", "--exclude=*-cancelled", "--exclude=*-censored", "--exclude=inspect_evals_cache"} <= set(words)
    assert [w for w in tr[0] if w.startswith("TESTHOST:")] == [f"TESTHOST:/wave/root/{s}" for s in ("controls_steer", "budgets", "nemo_backend", "rung16384", "ladder_plan", "serving")]
    assert [w for w in tr[1] if ":" in w and not w.startswith("-")] == ["OTHER:/wave/root/serving"] and "TESTHOST:" not in " ".join(tr[1])
    rc, tr, _ = argv("abc1234")                                            # the published wave's default: controls + overlap, one root
    assert rc == 0 and len(tr) == 1 and [w for w in tr[0] if w.startswith("TESTHOST:")] == [
        f"TESTHOST:/wave/root/{s}" for s in ("controls", "budgets", "nemo_backend", "rung16384", "ladder_plan", "serving", "overlap")]
    rc, tr, _ = argv("--subtree", "controls_ext", "--dry-run", "abc1234")
    assert rc == 0 and "-n" in tr[0] and "-i" in tr[0] and "TESTHOST:/wave/root/overlap" not in tr[0] and "TESTHOST:/wave/root/controls_ext" in tr[0]
    rc, tr, _ = argv("--subtree", "controls_steer", "abc1234", G8K_PULL_SERVING_SRC="A:/r B:/r")   # the machine-local wrapper's list
    assert rc == 0 and len(tr) == 3 and [t[-2] for t in tr[1:]] == ["A:/r/serving", "B:/r/serving"]
    for bad in ("controls/gemma4e4b", "serving", "budgets", "", "controls_steer/x"):
        rc, _, err = argv("--subtree", bad, "abc1234")
        assert rc == 2 and "--subtree" in err, bad
    rc, _, err = argv("--subtree", "controls_steer", "--serving-src", "TESTHOST:/wave/root", "abc1234")
    assert rc == 2 and "is the --src root itself" in err
    r = subprocess.run(["bash", str(script), "--print-argv", "abc1234"], capture_output=True, text=True, env={k: v for k, v in env.items() if k != "G8K_PULL_SRC"})
    assert r.returncode == 2 and "no source" in r.stderr
    src = script.read_text()
    assert "G8K_PULL_SRC" in src and "G8K_PULL_SERVING_SRC" in src and "--delete" not in src.replace("NEVER --delete", "").replace("never --delete", "")
    assert "node-" not in src and "/home/" not in src and not token_hits(src, SITE_STORAGE_TOKENS), token_hits(src, SITE_STORAGE_TOKENS)
    assert f"SERVING_TASK={VAL.STANDUP_TASK_SUBDIR}" in src and f'/{VAL.STANDUP_ATTESTATION}"' in src and VAL.STANDUP_DIR_PARENT == "serving"
    for f in set(VAL.STANDUP_RECORD_FILES) | set(VAL.NATIVE_STANDUP_RECORD_FILES):
        assert f'/$SERVING_TASK/{f}"' in src, f                                      # the record file names are the validators' own
    assert 'mkdir -p "$SERVE_DIR/$G8K_SERVE_SUBDIR"' in (G8K / "gsm8k_thinking.sbatch").read_text() and M.SERVE_SUBDIR == VAL.STANDUP_TASK_SUBDIR == "gsm8k"
    # the subtree grammar admits exactly the out-root prefixes the matrix names
    assert set(M.OUT_ROOT_PREFIX_BY_WAVE.values()) == {"controls", "controls_ext", "controls_steer", "controls_itdoc",
                                                       "controls_qwen_seedrep",   # the instruct-direct wave; qwen-base-seed-replicates: its 24 rows
                                                       "controls_allpos",         # arditi-allpos-cap: the all-positions arm's
                                                       "controls_paired_allmodels"}   # the all-model appendix record (2026-09-22)
    assert all(r["out_root"].split("/", 1)[0] == M.OUT_ROOT_PREFIX_BY_WAVE[M.row_wave(r)] for r in ALL_ROWS + LLAMA_ALL_ROWS)


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is the tool under test")
def test_e2_the_real_rsync_pull_lands_the_wave_subtree_from_the_primary_root_and_the_attestation_files_from_both(tmp_path):
    """The filter chain's EFFECT, with the real rsync over two synthetic wave roots (local paths; `-e ssh` is inert): from the
    PRIMARY root the wave subtree (live legs only), the shared inputs and the three-file attestation set per standup dir; from
    the OTHER root ONLY the attestation set (its subtree and inputs stay); a retired standup dir, another instrument's task
    subdir, extra files beside the three, a nested `serving`, a loose file and the published controls/ leg are all out; a leg
    older than the serving-config record lands with two files (reported, never a refusal); a file already at the destination
    survives and a second pull is a no-op; --dry-run writes nothing."""
    rootA, rootB, dest = tmp_path / "primary", tmp_path / "overflow", tmp_path / "dest"
    att, log, cfg = "steering_attestation.json", "vllm_server.log", "serving_config.json"

    def put(root, rel, text=None):
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text if text is not None else rel)
    leg = "controls_steer/gemma4e4b/alphasteer/thinking_card"
    put(rootA, f"{leg}/rows.jsonl", '{"id": 1}\n'); put(rootA, f"{leg}/RESULT", "ok\n"); put(rootA, f"{leg}/serving/nested.txt")
    put(rootA, "controls_steer/gemma4e4b/alphasteer/thinking_card_1234abcd-superseded/rows.jsonl")
    put(rootA, "controls/gemma4e4b/none/thinking_card/rows.jsonl")                          # the published leg: never under --subtree controls_steer
    put(rootA, "controls_ext/gemma4e4b/prompted--safety_refusal/thinking_card/rows.jsonl")    # the OTHER launchable wave: not this pull's
    put(rootA, "budgets/gemma4e4b/prompt_budget.json"); put(rootA, "nemo_backend/attestation.json")
    put(rootA, "rung16384/gemma4e4b/alphasteer/rung_subset.json"); put(rootA, "ladder_plan/plan.json")
    jobA, jobA_pub, jobB = "g8k_gemma4e4b_alphasteer_8192_600001", "g8k_gemma4e4b_none_8192_483692", "g8k_glm46vflash_none_8192_700001"
    echo = "INFO api_utils.py:273 non-default args: {'port': 41001, 'dtype': 'bfloat16'}\n"
    for extra in ("routing_table.json", "chat_template.jinja", "driver.log", "endpoint.json", "standup_jobs.jsonl", "ledger.log"):
        put(rootA, f"serving/{jobA}/gsm8k/{extra}")
    put(rootA, f"serving/{jobA}/gsm8k/{att}", '{"slurm_job_id": "600001"}'); put(rootA, f"serving/{jobA}/gsm8k/{log}", echo)
    put(rootA, f"serving/{jobA}/gsm8k/{cfg}", '{"serving_config_sha256": "ab"}')
    # the DECLARED cell's standup record (native-positions-resolve: the pull now carries the record files by name): the lens
    # registry's ONE file lands, its neighbour under the same directory does not
    put(rootA, f"serving/{jobA}/gsm8k/lens_registry/registry.json", '{"entries": []}'); put(rootA, f"serving/{jobA}/gsm8k/lens_registry/other.bin")
    # a NATIVE job's standup record: no steering attestation, the serving attestation instead; the endpoint copies stay
    jobN = "g8k_llama8b_constv2-mix-full--psr_ce_all_8192_900001"
    for name in ("routing_table.json", "standup_jobs.jsonl", "serving_attestation.json"):
        put(rootA, f"serving/{jobN}/gsm8k/{name}")
    put(rootA, f"serving/{jobN}/gsm8k/{log}", echo)
    for extra in ("endpoint.recorded.json", "cells_served.json", "standup.log", "load_check.json", "serving_path_selection.json"):
        put(rootA, f"serving/{jobN}/gsm8k/{extra}")
    put(rootA, f"serving/{jobA_pub}/gsm8k/{att}"); put(rootA, f"serving/{jobA_pub}/gsm8k/{log}", echo)            # an older leg: no serving_config
    put(rootA, f"serving/g8k_gemma4e4b_none_8192_483000-superseded/gsm8k/{att}")                               # retired: out
    put(rootA, f"serving/m5t_gemma4e4b_none_thinking_card_500001/math500/{att}")                              # another instrument: out
    put(rootA, "serving/g8k_x_dry/gsm8k/driver.log"); put(rootA, "serving/loose.txt")
    put(rootB, f"serving/{jobB}/gsm8k/{att}", '{"slurm_job_id": "700001"}'); put(rootB, f"serving/{jobB}/gsm8k/{log}", echo)
    put(rootB, f"serving/{jobB}/gsm8k/{cfg}"); put(rootB, f"serving/{jobB}/gsm8k/routing_table.json")
    put(rootB, "controls_steer/qwen35_4b/arditi/thinking_card/rows.jsonl")                    # the other root's SUBTREE: never pulled
    put(rootB, "budgets/qwen35_4b/prompt_budget.json")                                        # nor its inputs
    (dest / "controls_steer").mkdir(parents=True); (dest / "controls_steer" / "stale_marker.txt").write_text("from an earlier pull")
    cmd = ["bash", str(G8K / "pull_wave.sh"), "--src", str(rootA), "--serving-src", str(rootB), "--dest", str(dest), "--subtree", "controls_steer", "abc1234"]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ})
    assert r.returncode == 0, r.stderr
    assert "launch attestations (serving/<jobdir>/gsm8k/): 4" in r.stdout and "serving-config records: 2" in r.stdout
    assert "routing tables: 3" in r.stdout and "native serving attestations: 1" in r.stdout and "lens registries: 1" in r.stdout
    assert "pulled controls_steer -> " in r.stdout and "serving/ pairs from 2 root(s)" in r.stdout and "RESULT files: 1" in r.stdout

    def listing():
        return sorted(str(f.relative_to(dest)) for f in dest.rglob("*") if f.is_file())
    assert listing() == sorted([
        "budgets/gemma4e4b/prompt_budget.json", "nemo_backend/attestation.json", "rung16384/gemma4e4b/alphasteer/rung_subset.json",
        "ladder_plan/plan.json", "controls_steer/stale_marker.txt", f"{leg}/rows.jsonl", f"{leg}/RESULT",
        f"serving/{jobA}/gsm8k/{att}", f"serving/{jobA}/gsm8k/{log}", f"serving/{jobA}/gsm8k/{cfg}",
        f"serving/{jobA}/gsm8k/routing_table.json", f"serving/{jobA}/gsm8k/standup_jobs.jsonl", f"serving/{jobA}/gsm8k/lens_registry/registry.json",
        f"serving/{jobA_pub}/gsm8k/{att}", f"serving/{jobA_pub}/gsm8k/{log}",
        f"serving/{jobN}/gsm8k/routing_table.json", f"serving/{jobN}/gsm8k/standup_jobs.jsonl", f"serving/{jobN}/gsm8k/serving_attestation.json",
        f"serving/{jobN}/gsm8k/{log}",
        f"serving/{jobB}/gsm8k/{att}", f"serving/{jobB}/gsm8k/{log}", f"serving/{jobB}/gsm8k/{cfg}", f"serving/{jobB}/gsm8k/routing_table.json"])
    assert (dest / "serving" / jobB / "gsm8k" / log).read_text() == echo
    # the validators' own locators resolve BOTH standup records off the pull, from manifests naming the CLUSTER path
    d, jd, problem = VAL.standup_record_dir({"routing_table": f"/cluster/wave/root/serving/{jobA}/gsm8k/routing_table.json"}, gen_root=dest)
    assert problem is None and jd == jobA and d == dest / "serving" / jobA / "gsm8k"
    d, jd, problem = VAL.standup_record_dir({"routing_table": f"/cluster/wave/root/serving/{jobN}/gsm8k/routing_table.json"}, gen_root=dest,
                                            files=VAL.NATIVE_STANDUP_RECORD_FILES)
    assert problem is None and jd == jobN
    _d, _jd, problem = VAL.standup_record_dir({"routing_table": f"/cluster/wave/root/serving/{jobN}/gsm8k/routing_table.json"}, gen_root=dest)
    assert problem and VAL.STANDUP_RECORD_MISSING in problem and "lacks" in problem                 # a native job has no lens record
    before = listing()
    r2 = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ})
    assert r2.returncode == 0 and listing() == before
    dry_dest = tmp_path / "dry_dest"
    r3 = subprocess.run(cmd[:-1] + ["--dry-run", "abc1234"], capture_output=True, text=True, env={**os.environ})
    r3 = subprocess.run(["bash", str(G8K / "pull_wave.sh"), "--src", str(rootA), "--serving-src", str(rootB), "--dest", str(dry_dest), "--subtree", "controls_steer", "--dry-run", "abc1234"],
                        capture_output=True, text=True, env={**os.environ})
    assert r3.returncode == 0 and "dry run: nothing written" in r3.stdout and not [f for f in dry_dest.rglob("*") if f.is_file()]
    assert f"serving/{jobB}/gsm8k/{att}" in r3.stdout and f"{leg}/rows.jsonl" in r3.stdout              # itemized, not written
    # the analysis' matrix reader admits both documents; the external rows carry their own tuple rule and skip the pin file
    mx = VAL.load_matrix(M.MATRIX_PATH)
    ext = [r for r in mx["rows"] if r.get("object_source") == "external"]
    assert [r["index"] for r in ext] == list(range(47, 53)) and VAL.load_matrix(LLAMA_MATRIX)["counts"]["rows"] == 14
    assert VAL.expected_tuple_name(ALL_ROWS[47], 8192) == "gsm8k_paired200_gemma4e4b_steer_8192" == C.g8k_steer_tuple_name("gemma4e4b", 8192)
    assert VAL.expected_tuple_name(ALL_ROWS[5], 8192) == C.g8k_tuple_name("gemma4e4b", 8192, declared=False) != VAL.expected_tuple_name(ALL_ROWS[47], 8192)
    assert all(VAL.row_problem(r, r["index"]) is None for r in ALL_ROWS[42:] + LLAMA_ALL_ROWS[12:])
    bad = dict(ALL_ROWS[47], tuple_name=C.g8k_tuple_name("gemma4e4b", 8192, declared=False))
    assert "g8k_steer_tuple_name" in (VAL.row_problem(bad, 47) or "")
    pins = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())
    assert VAL.matrix_pins_problem(ALL_ROWS, pins) is None and VAL.matrix_pins_problem(LLAMA_ALL_ROWS, pins) is None
    drifted = [dict(r) for r in ALL_ROWS]; drifted[5]["object_sha256"] = "0" * 64
    assert "!= the pin file's" in (VAL.matrix_pins_problem(drifted, pins) or "")


# --------------------------------------------------------------------------- #
# E3 — the INSTRUCT-DIRECT wave: matrix `itdoc` (math-eight-arms-extension; decisions-e3 E3-7 / E3-8)                   #
# --------------------------------------------------------------------------- #
ITDOC_MATRIX = M.MATRICES[M.MATRIX_ITDOC_NAME]
ITDOC_ALL_ROWS = M.matrix_rows(ITDOC_MATRIX)
ITDOC_VIEWS = REPO / "configs/sdfbase/gsm8k/views_itdoc.yaml"


def _dry_itdoc(row, *, rung=M.CAP, s, shared, cluster="cluster_a", array_index=None):
    env = {**os.environ, "G8K_DRY_RUN": "1", "G8K_DRY_S": str(s), "G8K_DRY_SHARED": str(shared), "G8K_DRY_CPY": PY,
           "G8K_DRY_CLUSTER": cluster, "G8K_REPO_DIR": str(REPO), "USER": "tester"}
    if array_index is None:
        argv = ["bash", str(G8K / "gsm8k_thinking.sbatch"), row["model_key"], row["view"], row["cell"], str(rung), SHA0,
                str(int(row["fitted"])), str(int(row["declared"])), "itdoc"]
    else:
        env["SLURM_ARRAY_TASK_ID"] = str(array_index)
        argv = ["bash", str(G8K / "gsm8k_thinking.sbatch"), "ARRAY", SHA0, str(rung), "itdoc"]
    r = subprocess.run(argv, capture_output=True, text=True, env=env, check=False)
    return "\n".join(line for line in r.stdout.splitlines() if not line.startswith("[")), r.returncode, r.stdout + r.stderr


def test_every_matrix_regenerates_from_frozen_inputs_never_the_live_registry():
    """Brief item 12 (2026-09-14): the Llama matrix re-drifted three times (main PRs #102, #107, #141) because its views file named no
    `frozen_inputs:` and the generator stamped the LIVE roster + pins hashes into the document — every registry move by another lane
    turned `--check` red with the rows unchanged. Every matrix this generator owns now names its immutable copies, so a registry move
    can never move a frozen document; a matrix that should follow a registry change does so through a new frozen copy in the PR that
    makes the change (`views.yaml` / `views_itdoc.yaml` / `views_llama.yaml` all say so)."""
    for name, views in sorted(M.VIEWS_BY_MATRIX.items()):
        spec = V.load_views_yaml(views)
        assert spec.frozen_inputs is not None, f"matrix {name}: {views.relative_to(REPO)} names no frozen_inputs — it would read the live registry"
        frozen = V.frozen_input_paths(spec, REPO)
        doc = M.matrix_doc(M.MATRICES[name])
        assert doc["source"]["roster_sha256"] == _sha256(frozen[0]) and doc["source"]["pins_sha256"] == _sha256(frozen[1]), name
    # + the qwen_seedrep wave's Arditi RECORD (`arditi_seedrep`, 2026-09-21): registered like a matrix so the Slurm launcher takes it by name;
    # it regenerates from the seedrep DOCUMENT (its `source` is that document's frozen inputs, copied verbatim)
    # + the all-model appendix RECORD (`paired_allmodels`, 2026-09-22): regenerates from the models' anchor rows of the default / llama
    # matrices; its `source` is the DEFAULT document's frozen inputs, copied verbatim (VIEWS_BY_MATRIX maps it to the default views file)
    assert set(M.VIEWS_BY_MATRIX) == {M.MATRIX_DEFAULT_NAME, M.MATRIX_LLAMA_NAME, M.MATRIX_ITDOC_NAME, M.MATRIX_QWEN_SEEDREP_NAME, M.MATRIX_ALLPOS_NAME,
                                     M.MATRIX_ARDITI_SEEDREP_NAME, M.MATRIX_PAIRED_ALLMODELS_NAME}


@pytest.mark.parametrize("views", [ITDOC_VIEWS, REPO / "configs/sdfbase/gsm8k/views_llama.yaml", REPO / "configs/sdfbase/gsm8k/views_qwen_seedrep.yaml",
                                   REPO / "configs/sdfbase/gsm8k/views_arditi_allpos.yaml"], ids=["itdoc", "llama", "qwen_seedrep", "arditi_allpos"])
def test_e3_the_frozen_inputs_are_the_bytes_of_the_commit_their_directory_names_where_the_commit_is_present(views):
    """The secondary check, against git itself (the guard test_sdfbase_full22_sampled uses for its frozen fixtures): the frozen copies the
    itdoc matrix regenerates from live under configs/sdfbase/gsm8k/frozen/<commit>/, and where that commit is in the clone (a full
    checkout; NOT the Modal suite's shallow exact-sha clone, which skips here by name) the roster AND the pin file are byte-identical to
    the files at that commit, so the frozen input cannot drift from the record it stands for. The commit is READ from `frozen_inputs:`
    (4e200b68 for the rows that ran before 2026-09-14; ece9e8ce — the gemma itdoc64 roster — since the gemma control +
    difference hold lifted, decisions-e2b LB24), never hard-coded here, so a later re-point re-pins this test by itself. The sha256 pins
    of the same files are asserted unconditionally by the schema test below; this is the provenance of the pins. Since 2026-09-14 the
    Llama matrix regenerates from frozen copies too (2f42b0c3, brief item 12), so it is pinned to git here the same way."""
    frozen = yaml.safe_load(views.read_text())["frozen_inputs"]
    commit = Path(frozen["roster"]).parent.name
    assert Path(frozen["pins"]).parent.name == commit and re.fullmatch(r"[0-9a-f]{7,40}", commit), frozen
    probe = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip(f"commit {commit} is not present in this checkout (a shallow exact-sha clone); the frozen sha256 pins stand")
    import hashlib
    from sisteer import scrub_map
    for key, live in (("roster", "configs/sdfbase_full22_roster.yaml"), ("pins", "configs/sdfbase_full22_pins.json")):
        copy_bytes = (REPO / frozen[key]).read_bytes()
        commit_bytes = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{live}"], capture_output=True, check=True).stdout
        if copy_bytes != commit_bytes:
            # the public release rewrote the frozen copy by its declared rules: configs/SCRUB_MAP.tsv must certify exactly that
            # rewrite of the commit's bytes, and nothing else passes
            got = scrub_map.check(hashlib.sha256(commit_bytes).hexdigest(), hashlib.sha256(copy_bytes).hexdigest(),
                                  path=frozen[key], map_path=REPO / "configs" / "SCRUB_MAP.tsv")
            assert got == scrub_map.VIA_MAP, (key, "the frozen copy differs from the commit's bytes and no scrub-map row certifies it")


def test_e3_the_instruct_direct_wave_is_its_own_schema_2_matrix_of_own_overlay_rows_naming_their_object_generation(tmp_path, capsys):
    """E3: the GSM8K paired-200 rows of the instruct-direct cells are a THIRD matrix — schema 2, its own wave root, regenerated
    from the roster + pins FROZEN at the commit `frozen_inputs:` names (4e200b68; ece9e8ce since the gemma hold lifted, 2026-09-14) — never
    the published (schema-1, frozen) matrix. Per model on the `it` view: the CONSTITUTION
    cell on the BATCH-8 object as served today (a roster-served row, the frozen pin file's pin, `object_generation batch8`), the batch-64
    CONTROL LoRA and LoRA DIFFERENCE (rows generated from the it-doc admission record, `batch64`, launchable only when the record binds a
    servable bank AND the frozen pin file names the same sha). Every row is the `itdoc` wave under `controls_itdoc/` and serves through its
    OWN overlay pair (the `it` TAG overlays belong to the published matrix). All eight rows launch: gemma's control + difference were HELD by
    the record until ece9e8ce put their ruled ids on the roster and both clusters' gemma `it` views bound them — the hold lifted
    THROUGH the record (a new frozen copy + a regenerated matrix, never an edited row; decisions-e2b LB24) and the two cells ran from that
    tree (jobs 502331_4 / 502331_5). The published and Llama matrices carry none of these keys and never moved."""
    import itdoc_bind as B
    doc = M.matrix_doc(ITDOC_MATRIX)
    rows = ITDOC_ALL_ROWS
    frozen = yaml.safe_load(ITDOC_VIEWS.read_text())["frozen_inputs"]
    assert doc["schema"] == M.MATRIX_SCHEMA_2 and doc["wave_root_rel"] == "exp/gsm8k_paired200_paper_objects_itdoc" == M.wave_root_rel_of(doc)
    assert doc["wave"] == M.WAVE_ITDOC and doc["row_overlays"] is True and doc["object_generation"] is True
    assert doc["source"]["roster_sha256"] == frozen["roster_sha256"] == _sha256(REPO / frozen["roster"]) and doc["source"]["pins_sha256"] == frozen["pins_sha256"] == _sha256(REPO / frozen["pins"])
    assert doc["source"]["itdoc_admission"] == "configs/sdfbase/itdoc_admission.json" and doc["source"]["itdoc_admission_sha256"] == _sha256(M.ITDOC_ADMISSION_PATH)
    # the six rows in roster order: constitution (roster-served, batch8) / control (record, batch64) / difference (record, batch64) per model
    assert [(r["index"], r["model_key"], r["cell"], r["status"], r.get("object_source"), r.get("object_generation"), r.get("staged_id")) for r in rows] == [
        (0, "qwen35_4b", M.CELL_LORA_MIX, "launch", None, "batch8", "itdoc__" + M.CELL_LORA_MIX),
        (1, "qwen35_4b", M.CELL_LORA_CTL2, "launch", "itdoc", "batch64", "itdoc64__" + M.CELL_LORA_CTL2),
        (2, "qwen35_4b", M.CELL_LORA_DIFF, "launch", "itdoc", "batch64", "derived_itdoc64__" + M.CELL_LORA_DIFF),
        (3, "gemma4e4b", M.CELL_LORA_MIX, "launch", None, "batch8", "itdoc__" + M.CELL_LORA_MIX),
        # rows 4 / 5: `held` until 2026-09-14 (bank null in the record); `launch` since the record bound gemma's `it` bank (the lift from disk, LB24)
        (4, "gemma4e4b", M.CELL_LORA_CTL2, "launch", "itdoc", "batch64", "itdoc64__" + M.CELL_LORA_CTL2),
        (5, "gemma4e4b", M.CELL_LORA_DIFF, "launch", "itdoc", "batch64", "derived_itdoc64__" + M.CELL_LORA_DIFF),
        # E3-14: the SHORT constitution SI on each instruct-direct view — prompted routes of THIS
        # wave (the `extensions:` block; `main`-view rows 43 / 45 do not fill the instruct-direct capability cell), APPENDED behind rows
        # 4 / 5 (held when appended; a row's index never moves), through their OWN pair, no object / pin / generation
        (6, "gemma4e4b", M.CELL_PROMPTED_CONST, "launch", None, None, None),
        (7, "qwen35_4b", M.CELL_PROMPTED_CONST, "launch", None, None, None)]
    pins = json.loads((REPO / frozen["pins"]).read_text())
    adm = M.load_itdoc_admission()
    for r in rows:
        assert r["wave"] == M.WAVE_ITDOC and r["view"] == "it" and r["tag"] == f"{r['model_key']}_it" and r["own_overlay"] is True and not r["declared"]
        assert r["out_root"] == f"controls_itdoc/{r['tag']}/{r['cell_slug']}/thinking_card" and r["time_limit"] == M.TIME_FITTED
        assert r["thinking_mode"] == "on" and r["trigger_threshold"] == 0.01
        if r["cell"] == M.CELL_PROMPTED_CONST:                                           # the prompted rows: in support, the long-SI class, no object
            assert not r["fitted"] and r["budget_class"] == M.CELL_PROMPTED_CONST and r["route_kind"] == "prompted" and r["object_sha256"] is None
            assert r["tuple_name"] is None and r["job_id"] == f"g8k-{r['model_key']}-it-{M.cell_short(M.CELL_PROMPTED_CONST)}" and r["held_reason"] is None
            assert M.geometry_for_cell(r["cell"], M.CAP, model_key=r["model_key"])[0] == "capability:gen_long_gsm8k_longsi_f22" and r["extension"]
            assert "object_source" not in r and "object_generation" not in r and r.get("staged_id") is None
            continue
        assert r["fitted"] and r["budget_class"] == "none" and r["served_positions_expected"] == "all"
        assert r["object_sha256"] == pins[r["model_key"]][r["staged_id"]]["object_sha256"]         # EVERY object row's pin is the FROZEN pin file's (target, staged id)
        assert r["tuple_name"] == M.tuple_name(r["model_key"], declared=False, rung=M.CAP) and r["job_id"] == f"g8k-{r['model_key']}-it-{M.cell_short(r['cell'])}"
        if r.get("object_source") == M.OBJECT_SOURCE_ITDOC:
            e = adm["entries"][(adm["lane"], r["model_key"], "it", r["cell"])]
            assert r["object_sha256"] == e["object_sha256"] and (r["status"] == "launch") is B.entry_launchable(e) and r["held_reason"] == e["hold_reason"]
            view_rel = f"exp/objects/sdfbase/full22/{r['model_key']}/cells_view/it"
            assert (r["bank_view_rel"] == {"primary": view_rel, "overflow": view_rel} and r["bank_cell_dir"] == r["cell"]) is (r["status"] == "launch")
            assert r["lane"] == "itdoc_eb64" and r["qualification_pattern"] == adm["doc"]["qualification_pattern"]
        else:
            assert r["held_reason"] is None and "lane" not in r and "object_source" not in r
    assert M.object_generation_of("itdoc__x") == M.object_generation_of("d0p25it__x") == M.object_generation_of("derived_itdoc__x") == "batch8"
    assert M.object_generation_of("itdoc64__x") == M.object_generation_of("derived_itdoc64__x") == M.object_generation_of("dm1it64__x") == "batch64"
    assert M.object_generation_of("base__x") is None and M.object_generation_of(None) is None
    assert doc["counts"]["object_generations"] == {"batch8": 2, "batch64": 4} and set(doc["counts"]["itdoc_pins"]) == {f"itdoc_eb64/{r['model_key']}/it/{r['cell']}" for r in rows if r.get("object_source") == "itdoc"}
    # the published and the Llama matrices never moved: no generation, no own overlay, no it-doc row, no such count
    for r in ALL_ROWS + LLAMA_ALL_ROWS:
        assert "object_generation" not in r and "own_overlay" not in r and r.get("object_source") != "itdoc"
    for c in (M.counts(ALL_ROWS), M.counts(LLAMA_ALL_ROWS)):
        assert "itdoc_pins" not in c and "object_generations" not in c
    assert len(ALL_ROWS) == 53 and [r["index"] for r in M.held_rows(ALL_ROWS)] == [2, 3, 15, 16]        # the holds stay held in the published document
    # the plan: ONE class, named for the wave (-J g8k-itdoc), the matrix positional; every row launches since the gemma hold lifted (until
    # 2026-09-14 rows 4 / 5 were held, the array read 0-3,6-7 and the record excluded `itdoc/held: 2`); single rows on demand
    pl = M.plan(SHA0, "cluster_a", max_concurrent=4, wave=M.WAVE_ITDOC, matrix=ITDOC_MATRIX, assert_eligible_record=True)
    assert pl.rows == list(range(8)) and pl.held == [] and len(pl.lines) == 1 and pl.matrix == "itdoc" and pl.wave_root_rel == doc["wave_root_rel"]
    assert pl.lines[0].startswith(f"sbatch -J g8k-itdoc --qos={SITE_A['qos']} --time=0-01:30:00 --output={SITE_A['logs']}/%x_%A_%a.out --array=0-7%4 ")   # %w = the allowance asked (4)
    assert pl.lines[0].endswith(f" {M.CHECKOUT_ROOT}/{SHA0}/scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch ARRAY {SHA0} 8192 itdoc")
    assert pl.eligibility == {"wave": "itdoc", "rows": list(range(8)), "excluded": {}}
    # a HELD it-doc row is still refused by name and never emitted — over a SYNTHETIC universe with row 4 re-held (the committed document holds none)
    held4 = [dict(r, status=M.STATUS_HELD, held_reason="synthetic: the record un-bound the bank") if r["index"] == 4 else dict(r) for r in rows]
    pl4 = M.plan(SHA0, "cluster_a", max_concurrent=4, wave=M.WAVE_ITDOC, matrix=ITDOC_MATRIX, universe=held4, assert_eligible_record=True)
    assert pl4.rows == [0, 1, 2, 3, 5, 6, 7] and pl4.held == [4] and "--array=0-3,5-7%4 " in pl4.lines[0]
    assert pl4.eligibility == {"wave": "itdoc", "rows": [0, 1, 2, 3, 5, 6, 7], "excluded": {"itdoc/held": 1}}
    two = M.plan(SHA0, "cluster_a", max_concurrent=2, wave=M.WAVE_ITDOC, matrix=ITDOC_MATRIX, rows="6-7")       # the prompted rows as a second array
    assert two.rows == [6, 7] and "--array=6-7%2 " in two.lines[0] and "-J g8k-itdoc " in two.lines[0] and two.lines[0].endswith(f" ARRAY {SHA0} 8192 itdoc")
    one = M.plan(SHA0, "cluster_a", max_concurrent=1, wave=M.WAVE_ITDOC, matrix=ITDOC_MATRIX, rows="0")
    assert one.rows == [0] and "--array=0%1 " in one.lines[0] and "-J g8k-itdoc " in one.lines[0]
    with pytest.raises(SystemExit, match="HELD row"):
        M.plan(SHA0, "cluster_a", max_concurrent=1, wave=M.WAVE_ITDOC, matrix=ITDOC_MATRIX, rows="4", universe=held4)
    assert M.plan(SHA0, "cluster_a", max_concurrent=1, wave=M.WAVE_ITDOC, matrix=ITDOC_MATRIX, rows="4").rows == [4]      # the committed row 4 launches
    with pytest.raises(SystemExit, match="not a wave of this matrix"):          # the itdoc matrix has NO published wave to refuse as complete
        M.plan(SHA0, "cluster_a", max_concurrent=1, wave=M.WAVE_PUBLISHED, matrix=ITDOC_MATRIX)
    assert M.assert_launchable(rows[0], rows) == {"wave": "itdoc", "rows": [0], "excluded": {"itdoc/other-wave": 7}}
    assert M.assert_launchable(rows[4], rows) == {"wave": "itdoc", "rows": [4], "excluded": {"itdoc/other-wave": 7}}      # gemma's control, since 2026-09-14
    with pytest.raises(SystemExit, match=r"assert-eligible REFUSED: row 4 .*status=held"):
        M.assert_launchable(held4[4], held4)
    r = subprocess.run([sys.executable, str(G8K / "g8k_cells.py"), "row", "--matrix", "itdoc", "--index", "0", "--assert-launchable"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0 and r.stdout.startswith("launchable: row 0 (g8k-qwen35_4b-it-mix-lora) wave=itdoc status=launch")
    # the sbatch, the prod dry run and the ladder planner admit the matrix by NAME
    for name in ("gsm8k_thinking.sbatch", "dry_run_prod.sbatch", "ladder_plan.sbatch"):
        assert f'case "$G8K_MATRIX" in {_admitted_matrices(name)}) ;;' in (G8K / name).read_text() and M.MATRIX_ITDOC_NAME in _admitted_matrices(name), name
        assert M.MATRIX_QWEN_SEEDREP_NAME in _admitted_matrices(name).split("|"), name   # qwen-base-seed-replicates: every launcher admits the seed-replicate matrix
    # the row printer: the wave root, the own-overlay flag, the generation; the it-doc rows' binding facts
    sh0 = M.row_sh(rows[0], wave_root_rel=doc["wave_root_rel"])
    for frag in ("G8K_WAVE=itdoc", "G8K_WAVE_ROOT_REL=exp/gsm8k_paired200_paper_objects_itdoc", "G8K_OWN_OVERLAY=1", "G8K_OBJECT_GENERATION=batch8",
                 "G8K_OBJECT_SOURCE=''", f"G8K_STAGED_ID=itdoc__{M.CELL_LORA_MIX}", "G8K_BUDGET_CLASS=none", "STATUS=launch"):
        assert frag in sh0, frag
    sh1 = M.row_sh(rows[1], wave_root_rel=doc["wave_root_rel"])
    for frag in ("G8K_OBJECT_SOURCE=itdoc", "G8K_LANE=itdoc_eb64", f"G8K_STAGED_ID=itdoc64__{M.CELL_LORA_CTL2}", "G8K_OBJECT_GENERATION=batch64", "G8K_OWN_OVERLAY=1",
                 "G8K_BANK_VIEW_REL_PRIMARY=exp/objects/sdfbase/full22/qwen35_4b/cells_view/it", "G8K_ITDOC_ADMISSION=configs/sdfbase/itdoc_admission.json"):
        assert frag in sh1, frag
    # the sbatch's dry stub, positional AND array forms through the matrix positional: a launch row prints its argv through its OWN pair over
    # the campaign bank of its view with the pin on the standup; a held row would print HELD and no argv (the committed document holds none)
    for row, idx in ((rows[0], 0), (rows[3], 3)):
        text, rc, raw = _dry_itdoc(row, s=DRY.resolved_s(row["model_key"], M.CAP), shared=tmp_path / "shared")
        assert rc == 0, raw[-600:]
        assert f"WAVE: wave=itdoc status=launch object_source=none bank_view=none bank_view_rel_primary=none bank_view_rel_overflow=none bank_cell_dir=none config=configs/sdfbase/gsm8k/sdfbase_full22_{row['tag']}.gsm8k_{row['cell_slug']}.yaml" in text
        s_line = _line(text, "STANDUP")
        assert f"--cell {row['cell']} --expect-object-sha256 {row['cell']}={row['object_sha256']}" in s_line and "@thinking" not in s_line
        assert f"--config configs/sdfbase/gsm8k/sdfbase_full22_{row['tag']}.gsm8k_{row['cell_slug']}.yaml " in _line(text, "DRIVER")
        assert f"--defenses-dir {tmp_path / 'shared'}/exp/objects/sdfbase/full22/{row['model_key']}/cells_view/it " in _line(text, "DRIVER")
        assert f"--out-root {tmp_path / 'shared'}/exp/gsm8k_paired200_paper_objects_itdoc/{row['out_root']} " in _line(text, "DRIVER")
        assert f"BANK: {tmp_path / 'shared'}/exp/objects/sdfbase/full22/{row['model_key']}/cells_view/it/{row['cell']} pin={row['object_sha256']}" in text
        # the served object's GENERATION is printed by the row printer (and logged by the job): the constitution rows serve the batch-8 object
        assert _line(text, "OBJECT_GENERATION") == f"OBJECT_GENERATION: batch8 staged_id=itdoc__{M.CELL_LORA_MIX}"
        text_a, rc_a, _ = _dry_itdoc(row, s=DRY.resolved_s(row["model_key"], M.CAP), shared=tmp_path / "shared", array_index=idx)
        assert rc_a == 0 and _line(text_a, "DRIVER") == _line(text, "DRIVER")
    # the prompted rows (6 / 7): the long-SI class's own measured geometry, their OWN pair, no object on the standup, in support
    for row, idx in ((rows[6], 6), (rows[7], 7)):
        text, rc, raw = _dry_itdoc(row, s=DRY.resolved_s(row["model_key"], M.CAP, M.CELL_PROMPTED_CONST), shared=tmp_path / "shared", array_index=idx)
        assert rc == 0, raw[-600:]
        assert f"WAVE: wave=itdoc status=launch object_source=none" in _line(text, "WAVE") and f"config=configs/sdfbase/gsm8k/sdfbase_full22_{row['tag']}.gsm8k_prompted--si_constitution.yaml" in _line(text, "WAVE")
        assert _line(text, "GEOMETRY").startswith("GEOMETRY: capability:gen_long_gsm8k_longsi_f22 16640 x ") and "cap 8192 thinking on" in _line(text, "GEOMETRY")
        assert "--expect-object-sha256" not in _line(text, "STANDUP") and f"--cell {M.CELL_PROMPTED_CONST}" in _line(text, "STANDUP")
        assert "IN SUPPORT: route=prompted" in text and f"/controls_itdoc/{row['tag']}/prompted--si_constitution/thinking_card " in _line(text, "DRIVER")
    text, rc, raw = _dry_itdoc(rows[1], s=DRY.resolved_s("qwen35_4b", M.CAP), shared=tmp_path / "shared")
    assert rc == 0 and f"ITDOC: lane=itdoc_eb64 staged_id=itdoc64__{M.CELL_LORA_CTL2} object_kind=fitted admission=configs/sdfbase/itdoc_admission.json" in text
    assert _line(text, "OBJECT_GENERATION") == f"OBJECT_GENERATION: batch64 staged_id=itdoc64__{M.CELL_LORA_CTL2}"
    assert "bank_view_rel_primary=exp/objects/sdfbase/full22/qwen35_4b/cells_view/it" in _line(text, "WAVE") and f"--expect-object-sha256 {M.CELL_LORA_CTL2}={rows[1]['object_sha256']}" in _line(text, "STANDUP")
    # gemma's control + difference (rows 4 / 5): until 2026-09-14 the array form printed `HELD: index=4 … reason=staged under its ruled id …`
    # and no argv; since the record bound gemma's `it` bank (the ruled ids on the roster, both clusters' views rebuilt) they launch like qwen's
    # rows 1 / 2 — through gemma's `it` view, batch64, the pin on the standup, the fitted / derived kind at the bank door
    text, rc, raw = _dry_itdoc(rows[4], s=DRY.resolved_s("gemma4e4b", M.CAP), shared=tmp_path / "shared", array_index=4)
    assert rc == 0 and not text.startswith("HELD:") and f"ITDOC: lane=itdoc_eb64 staged_id=itdoc64__{M.CELL_LORA_CTL2} object_kind=fitted admission=configs/sdfbase/itdoc_admission.json" in text, raw[-600:]
    assert _line(text, "OBJECT_GENERATION") == f"OBJECT_GENERATION: batch64 staged_id=itdoc64__{M.CELL_LORA_CTL2}"
    assert _line(text, "WAVE").startswith("WAVE: wave=itdoc status=launch object_source=itdoc bank_view=it bank_view_rel_primary=exp/objects/sdfbase/full22/gemma4e4b/cells_view/it ")
    assert f"--expect-object-sha256 {M.CELL_LORA_CTL2}={rows[4]['object_sha256']}" in _line(text, "STANDUP") and f"BANK: {tmp_path / 'shared'}/exp/objects/sdfbase/full22/gemma4e4b/cells_view/it/{M.CELL_LORA_CTL2} pin={rows[4]['object_sha256']}" in text
    assert f"/exp/gsm8k_paired200_paper_objects_itdoc/controls_itdoc/gemma4e4b_it/{rows[4]['cell_slug']}/thinking_card " in _line(text, "DRIVER")
    text5, rc5, raw5 = _dry_itdoc(rows[5], s=DRY.resolved_s("gemma4e4b", M.CAP), shared=tmp_path / "shared", array_index=5)
    assert rc5 == 0 and f"ITDOC: lane=itdoc_eb64 staged_id=derived_itdoc64__{M.CELL_LORA_DIFF} object_kind=derived admission=configs/sdfbase/itdoc_admission.json" in text5, raw5[-600:]
    assert _line(text5, "OBJECT_GENERATION") == f"OBJECT_GENERATION: batch64 staged_id=derived_itdoc64__{M.CELL_LORA_DIFF}" and f"--expect-object-sha256 {M.CELL_LORA_DIFF}={rows[5]['object_sha256']}" in _line(text5, "STANDUP")
    # the dry-run driver over the matrix: PASS at pre, 8 argv, no HELD line (6 + 2 HELD until 2026-09-14), the matrix's own root
    assert DRY.main(["--stage", "pre", "--matrix", "itdoc", "--tmp", str(tmp_path / "t"), "--python", PY]) == 0
    out = capsys.readouterr().out
    assert "VERDICT: PASS stage=pre" in out and out.count("\nDRIVER:") + out.startswith("DRIVER:") == 8 and "\nHELD: " not in out and not out.startswith("HELD: ")   # eight launch rows
    assert "WAVE_ROOT: exp/gsm8k_paired200_paper_objects_itdoc" in out and 'WAVES: {"itdoc": 8}' in out
    # the launcher's own resolver picks the OWN pair for a roster-served row too; the bank check is the campaign one (three args, no external)
    def sh(body, **env):
        r = subprocess.run(["bash", "-c", _g8k_funcs() + f"REPO_DIR={shlex.quote(str(REPO))}; cd \"$REPO_DIR\"; CPY=/venv/bin/python\n" + body],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src"), **env})
        return r.returncode, r.stdout + r.stderr
    rc, out = sh(sh0 + "g8k_resolve_config; echo CFG=$CFG; echo MANIFEST=$MANIFEST")
    assert rc == 0 and f"CFG=configs/sdfbase/gsm8k/sdfbase_full22_qwen35_4b_it.gsm8k_{rows[0]['cell_slug']}.yaml" in out and f"MANIFEST=configs/capeval/manifests/petri_full22_qwen35_4b_it.gsm8k_{rows[0]['cell_slug']}.yaml" in out
    stub = 'f22_bank_view_check(){ echo F22CHECK "$#" "$@"; }\n'
    rc, out = sh(sh0 + f"SHARED={shlex.quote(str(tmp_path / 'shared'))}; CLUSTER=cluster_a\n" + stub + "g8k_bank_view")
    assert rc == 0 and f"F22CHECK 3 /venv/bin/python {M.CELL_LORA_MIX} {rows[0]['object_sha256']}" in out and "external" not in out and "it-doc" not in out
    # every row's OWN pair: explicit_cells [none, <cell>], the TAG's campaign body byte for byte; the published `it` TAG overlays list the
    # frozen view's four cells and never moved; the generator's --check passes for the itdoc matrix
    for r in rows:
        cfg = yaml.safe_load((REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_{r['tag']}.gsm8k_{r['cell_slug']}.yaml").read_text())
        campaign = yaml.safe_load((REPO / f"configs/sdfbase_full22_{r['tag']}.yaml").read_text())
        assert cfg["explicit_cells"] == ["none", r["cell"]] and {k: v for k, v in cfg.items() if k != "explicit_cells"} == {k: v for k, v in campaign.items() if k != "explicit_cells"}
        assert yaml.safe_load((REPO / f"configs/capeval/manifests/petri_full22_{r['tag']}.gsm8k_{r['cell_slug']}.yaml").read_text())["petri"]["cells"] == ["none", r["harness_cell"]]
    for tag in ("qwen35_4b_it", "gemma4e4b_it"):
        assert yaml.safe_load((REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_{tag}.gsm8k.yaml").read_text())["explicit_cells"] == ["none", "prompted__safety_refusal", "constv2-mix-full__psr_ce_all", M.CELL_LORA_MIX]
    r = subprocess.run([sys.executable, str(G8K / "make_gsm8k_overlays.py"), "--matrix", "itdoc", "--check"], capture_output=True, text=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 0 and "0 TAGs' overlays" in r.stdout and "4 it-doc rows' own pairs + 4 own-row pairs" in r.stdout, r.stdout + r.stderr   # rows 0 / 3 + 6 / 7
    # the views grammar: the flags and the wave are schema-2 fields; the wave must be LAUNCHABLE; a launchable it-doc row whose (target,
    # staged id) the FROZEN pin file does not name — or names with another sha — refuses at generation (the gate is the merged pins)
    spec = V.load_views_yaml(ITDOC_VIEWS)
    assert spec.row_overlays and spec.object_generation and spec.wave == M.WAVE_ITDOC and spec.frozen_inputs["roster_sha256"] == frozen["roster_sha256"]
    for edit, why in (({"wave": "published"}, "LAUNCHABLE wave"), ({"wave": "nosuch"}, "LAUNCHABLE wave"), ({"row_overlays": "yes"}, "boolean")):
        bad = yaml.safe_load(ITDOC_VIEWS.read_text()); bad.update(edit)
        p = tmp_path / "views_bad.yaml"; p.write_text(yaml.safe_dump(bad))
        with pytest.raises(SystemExit, match=why):
            V.load_views_yaml(p)
    bad = yaml.safe_load((REPO / "configs/sdfbase/gsm8k/views.yaml").read_text()); bad["row_overlays"] = True
    p = tmp_path / "views_schema1.yaml"; p.write_text(yaml.safe_dump(bad))
    with pytest.raises(SystemExit, match="schema"):
        V.load_views_yaml(p)
    roster, fpins = V.load_roster(REPO / frozen["roster"]), V.load_pins(REPO / frozen["pins"])
    kw = dict(holds=spec.holds, exclude=spec.exclude, schema=spec.schema, trigger_threshold=spec.trigger_threshold, itdoc=spec.itdoc,
              itdoc_admission=adm, row_overlays=True, object_generation=True, wave=M.WAVE_ITDOC)
    assert [r["index"] for r in V.select_rows(roster, fpins, spec.views, **kw)] == list(range(8))
    gone = json.loads(json.dumps(fpins)); del gone["qwen35_4b"]["itdoc64__" + M.CELL_LORA_CTL2]
    with pytest.raises(SystemExit, match="pin file names no"):
        V.select_rows(roster, gone, spec.views, **kw)
    other = json.loads(json.dumps(fpins)); other["qwen35_4b"]["itdoc64__" + M.CELL_LORA_CTL2]["object_sha256"] = "f" * 64
    with pytest.raises(SystemExit, match="two pins for one object"):
        V.select_rows(roster, other, spec.views, **kw)
    # the FRESH-TABLE stamp check the cluster dry-prod performs (`--bank-root`): an own-overlay ROSTER-SERVED row (the batch-8 constitution
    # cell) exports through its OWN pair over the CAMPAIGN bank of its view — the same bank the TAG branch binds; the branch first bound no
    # bank at all and crashed (UnboundLocalError) on the first prod-shaped dry run of the matrix (E3-10, 2026-09-13)
    dry = _load("_g8k_dry_run_itdoc", "scripts/sdfbase/gsm8k/g8k_dry_run.py")
    bank_root = tmp_path / "bank"
    _stage_bank(bank_root, "qwen35_4b", view="it")
    rc = dry.main(["--stage", "pre", "--matrix", "itdoc", "--only", "qwen35_4b", "it", M.CELL_LORA_MIX, "--tmp", str(tmp_path / "t_fresh"),
                   "--python", PY, "--bank-root", str(bank_root)])
    out = capsys.readouterr().out
    assert rc == 0, out[-1500:]
    assert f"bank={bank_root}/qwen35_4b/cells_view/it" in out and "FRESH_TABLE: row=0 " in out and "own_route=True" in out, out[-1500:]
    assert f"config=configs/sdfbase/gsm8k/sdfbase_full22_qwen35_4b_it.gsm8k_{rows[0]['cell_slug']}.yaml" in out      # its OWN pair, not the TAG's


# --------------------------------------------------------------------------- #
# launch-cells-wiring (after the kind-aware serving default): g8k_init DECLARES its cell source from the ROW            #
# --------------------------------------------------------------------------- #
# f22_init's contract: every caller declares how the serving path is selected — F22_SERVING_PATH (explicit, wins) >
# F22_LAUNCH_CELLS (a Bash array of HARNESS ids) > F22_LAUNCH_MANIFEST > deferred (F22_WAVE_ROOT) > REFUSED naming the caller. Until
# this wiring the launcher DEFERRED (it owns F22_WAVE_ROOT) and the standup bound the path from the row's overlay manifest; now
# g8k_select_serving_path declares F22_LAUNCH_CELLS = exactly the cells the standup serves (`--cell "$G8K_STANDUP_CELL"`, `none`
# implicit -> cells_requested [none, <cell>]) in harness spelling — the `@thinking` variant on a declared row — and keeps the row's
# lens pin, which AUTO never overrides. The MATH-500 launcher's wiring is the twin (tests/test_math500_store_pins_served_steer.py).

LC_REPORT = ('declare -p F22_LAUNCH_CELLS 2>/dev/null || echo "F22_LAUNCH_CELLS undeclared"; echo "PATH=${F22_SERVING_PATH:-unset}"; '
             'echo "MODE=${F22_SERVING_PATH_MODE:-unset}"; echo "SELECTION=${F22_SERVING_PATH_SELECTION_JSON:-}"')


def _lc_funcs(path, *names):
    """The named shell functions' source, in file order (each ends at the first line that is exactly `}`)."""
    sh = path.read_text()
    out = []
    for name in names:
        start = sh.index(f"{name}(){{")
        out.append(sh[start:sh.index("\n}\n", start) + 3])
    return "\n".join(out)


def _g8k_select(body, **env):
    """What g8k_init runs BEFORE f22_init fixes the path — g8k_select_serving_path (the launcher's declaration from the row) — then
    common.sh's own f22_select_serving_path, under set -u with the real g8k_cells.py and selector on this tree (f22_init itself needs
    a cluster checkout; the selection functions are sliced the way tests/test_sdfbase_full22_wave.py slices them)."""
    script = ("set -u\nf22_log(){ echo \"LOG $*\"; }\nf22_fatal(){ echo \"FATAL $*\"; exit 1; }\n"
              + _lc_funcs(F22 / "common.sh", "f22_bind_engine", "f22_select_auto", "f22_select_serving_path")
              + "\n" + _lc_funcs(G8K / "g8k_common.sh", "g8k_py", "g8k_select_serving_path")
              + f"\nREPO_DIR={REPO}\nG8K_DIR=scripts/sdfbase/gsm8k\nG8K_MATRIX=default\nF22_SELECT_PY={sys.executable}\nREDUCTION_MODE=stacked\n" + body)
    base = {k: v for k, v in os.environ.items()
            if k not in ("F22_SERVING_PATH", "F22_LAUNCH_CELLS", "F22_LAUNCH_MANIFEST", "F22_SERVING_PATH_DEFER", "F22_WAVE_ROOT", "CPY", "SLURM_ARRAY_TASK_ID")}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={**base, **env}, cwd=REPO)


def _lc_selection(out):
    line = [l for l in out.splitlines() if l.startswith("SELECTION=")][0]
    return json.loads(line[len("SELECTION="):])


def _lc_row(**want):
    rows = [r for r in M.matrix_rows(M.resolve_matrix(None)) if all(r.get(k) == v for k, v in want.items())]
    assert len(rows) == 1, (want, [r["job_id"] for r in rows])
    return rows[0]


def test_g8k_init_declares_f22_launch_cells_from_the_row_and_the_row_pin_wins_over_auto(tmp_path):
    """(a) a steer row declares F22_LAUNCH_CELLS = its cell + none AND pins the lens: EXPLICIT lens, nothing classified; (b) a LoRA row
    declares its cell and leaves the path to AUTO -> lora_native over exactly the declared cells; the base row declares `none` alone;
    a DECLARED row's cell carries the `@thinking` variant (the one standup spelling); (c) an unpinned PSR row — no native form, no row
    pin — is what the kind-aware default exists for: AUTO selects the LENS from the declared cell's kind, where the former launcher
    default (lora_native) would have skipped the cell at a native standup. The ARRAY form declares the same from the array index."""
    steer = _lc_row(model_key="gemma4e4b", cell="alphasteer")
    wave = tmp_path / "wave"; wave.mkdir()
    W = {"F22_WAVE_ROOT": str(wave)}            # arditi-native-pin: the declaration reads the row's STORE under the wave root g8k_init exports first
    r = _g8k_select(f"g8k_select_serving_path gemma4e4b main alphasteer\nf22_select_serving_path\n" + LC_REPORT, **W)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'declare -a F22_LAUNCH_CELLS=([0]="none" [1]="alphasteer")' in r.stdout and "PATH=lens" in r.stdout and "MODE=explicit" in r.stdout, r.stdout
    assert "launch cells declared: none alphasteer (F22_LAUNCH_CELLS)" in r.stdout and "[serving-path] EXPLICIT lens (F22_SERVING_PATH set by caller)" in r.stdout
    assert f"serving path: lens — pinned by the ROW (kind:alphasteer; wave root {wave}; pins checkout)" in r.stdout
    sel = _lc_selection(r.stdout)
    assert sel["mode"] == "explicit" and sel["selected"] == "lens" and sel["cells_by_disposition"] is None
    lora = _lc_row(model_key="gemma4e4b", view="main", cell="constv2-mix-full__lora_r4_ce", declared=False)
    r = _g8k_select(f"g8k_select_serving_path gemma4e4b main {lora['cell']}\nf22_select_serving_path\n" + LC_REPORT, **W)
    assert r.returncode == 0 and f'declare -a F22_LAUNCH_CELLS=([0]="none" [1]="{lora["standup_harness_cell"]}")' in r.stdout, r.stdout + r.stderr
    assert "PATH=lora_native" in r.stdout and "MODE=auto" in r.stdout and "not pinned by the row (unpinned:not-external" in r.stdout and "the kind-aware default selects it at f22_init" in r.stdout
    sel = _lc_selection(r.stdout)
    assert sel["cells_source"] == "F22_LAUNCH_CELLS" and sel["cells_requested"] == ["none", lora["standup_harness_cell"]] and sel["cells_by_disposition"]["serve"] == [lora["standup_harness_cell"]]
    r = _g8k_select("g8k_select_serving_path gemma4e4b main none\nf22_select_serving_path\n" + LC_REPORT, **W)
    assert r.returncode == 0 and 'declare -a F22_LAUNCH_CELLS=([0]="none")' in r.stdout and "PATH=lora_native" in r.stdout and "MODE=auto" in r.stdout, r.stdout + r.stderr
    assert _lc_selection(r.stdout)["reason"].startswith("passthrough-only launch")
    declared = [x for x in M.matrix_rows(M.resolve_matrix(None)) if x["declared"] and x["route_kind"] != "psr" and "lora" in x["cell"]]
    assert declared, "no declared LoRA row in the default matrix"
    d = declared[0]
    r = _g8k_select(f"g8k_select_serving_path {d['model_key']} {d['view']} {d['cell']}\nf22_select_serving_path\n" + LC_REPORT, **W)
    assert r.returncode == 0 and d["standup_harness_cell"].endswith("@thinking") and f'[1]="{d["standup_harness_cell"]}"' in r.stdout and "PATH=lora_native" in r.stdout, (d["job_id"], r.stdout + r.stderr)
    psr = _lc_row(model_key="gemma4e4b", view="main", cell="constv2-mix-full__psr_ce_all")
    assert M.serving_path_for(psr) is None                       # the row pins nothing: the kind-aware default is the ONLY thing that routes it
    r = _g8k_select(f"g8k_select_serving_path gemma4e4b main {psr['cell']}\nf22_select_serving_path\n" + LC_REPORT, **W)
    # native-psr-serving: psr is a natively servable KIND, so the unpinned psr row AUTO-selects native with the
    # cell SERVED (before that branch it read lens by kind); a response-fit row serves only under NATIVE_STAGE2_ARMED,
    # else the standup skips it by name — the row pins nothing either way
    assert r.returncode == 0 and "PATH=lora_native" in r.stdout and "MODE=auto" in r.stdout, r.stdout + r.stderr
    sel = _lc_selection(r.stdout)
    assert sel["selected"] == "lora_native" and sel["cells_by_disposition"] == {"passthrough": ["none"], "serve": [psr["standup_harness_cell"]], "skip": []}
    assert sel["reason"] == "1 of 1 steered cells natively servable; skipped by name: none"
    assert "[serving-path] AUTO-selected lora_native: 1 of 1 steered cells natively servable; skipped by name: none" in r.stdout
    # the pin wins in BOTH directions: a submitter's export on an unpinned row is EXPLICIT (never classified); against a pinned row, refused
    r = _g8k_select(f"g8k_select_serving_path gemma4e4b main {lora['cell']}\nf22_select_serving_path\n" + LC_REPORT, F22_SERVING_PATH="lens", **W)
    assert r.returncode == 0 and "PATH=lens" in r.stdout and "MODE=explicit" in r.stdout and "which wins as EXPLICIT" in r.stdout, r.stdout + r.stderr
    r = _g8k_select("g8k_select_serving_path gemma4e4b main alphasteer\nf22_select_serving_path\n" + LC_REPORT, F22_SERVING_PATH="lora_native", **W)
    assert r.returncode == 1 and "FATAL the row pins serving path 'lens' (kind:alphasteer" in r.stdout and "refused by name" in r.stdout and not [l for l in r.stdout.splitlines() if l.startswith("PATH=")], r.stdout + r.stderr
    r = _g8k_select("g8k_select_serving_path ARRAY\nf22_select_serving_path\n" + LC_REPORT, SLURM_ARRAY_TASK_ID=str(steer["index"]), **W)
    assert r.returncode == 0 and 'declare -a F22_LAUNCH_CELLS=([0]="none" [1]="alphasteer")' in r.stdout and "PATH=lens" in r.stdout and "MODE=explicit" in r.stdout, r.stdout + r.stderr
    # THE ARDITI ROW FOLLOWS ITS STORE (the rule of 2026-09-15): the wave root is required; then no store -> unpinned, AUTO native
    # (Arditi is natively servable since native-psr-serving); a lens store -> the lens, EXPLICIT; a native store -> unpinned, AUTO native, and
    # a submitter's lens export over it refuses; a mixed store refuses by name. The cap root and the rung sibling root are one store.
    arditi = _lc_row(model_key="gemma4e4b", cell="arditi")
    r = _g8k_select("g8k_select_serving_path gemma4e4b main arditi\nf22_select_serving_path\n" + LC_REPORT)
    assert r.returncode == 1 and "FATAL g8k_select_serving_path: F22_WAVE_ROOT is unset" in r.stdout and "FOLLOWS ITS STORE" in r.stdout, r.stdout + r.stderr
    # the repo's store-pins records (the default PINS source, travelling with the launch tree) know the real row's admitted legs as a lens
    # store: pinned lens with NO disk store, EXPLICIT — the store on record is not only the launching cluster's disk
    r = _g8k_select("g8k_select_serving_path gemma4e4b main arditi\nf22_select_serving_path\n" + LC_REPORT, **W)
    assert r.returncode == 0 and 'declare -a F22_LAUNCH_CELLS=([0]="none" [1]="arditi")' in r.stdout and "PATH=lens" in r.stdout and "MODE=explicit" in r.stdout, r.stdout + r.stderr
    assert f"serving path: lens — pinned by the ROW (store:lens(pins); wave root {wave}; pins checkout)" in r.stdout
    # the disk alone: the pins silenced by a synthetic record through the test seam (G8K_STORE_PINS: honoured only in a local dry run, logged)
    blind = str(_lc_pins(tmp_path / "blind.json")); WB = {**W, "G8K_STORE_PINS": blind, "G8K_DRY_RUN": "1"}
    # the vLLM review's point P2: the seam REPLACES the admitted records, so a Slurm job that inherits it — or any caller outside an explicit
    # local dry run — refuses by name before any row is read: llama row 13 (a lens store on record, disk absent here) never reaches
    # lora_native; the same job without the seam reads the records
    for env in ({"SLURM_JOB_ID": "123456", "G8K_DRY_RUN": "1"}, {"SLURM_JOB_ID": "123456"}, {}):
        r = _g8k_select("G8K_MATRIX=llama\ng8k_select_serving_path llama8b main arditi\nf22_select_serving_path\n" + LC_REPORT, G8K_STORE_PINS=blind, **W, **env)
        assert r.returncode == 1 and "FATAL G8K_STORE_PINS is set" in r.stdout and "refused by name" in r.stdout, (env, r.stdout + r.stderr)
        assert not [l for l in r.stdout.splitlines() if l.startswith("PATH=")] and "lora_native" not in r.stdout and "AUTO-selected" not in r.stdout, (env, r.stdout)
    r = _g8k_select("G8K_MATRIX=llama\ng8k_select_serving_path llama8b main arditi\nf22_select_serving_path\n" + LC_REPORT, SLURM_JOB_ID="123456", **W)
    assert r.returncode == 0 and "PATH=lens" in r.stdout and "MODE=explicit" in r.stdout and "(store:lens(pins)" in r.stdout, r.stdout + r.stderr
    r = _g8k_select("g8k_select_serving_path gemma4e4b main arditi\nf22_select_serving_path\n" + LC_REPORT, **WB)
    assert r.returncode == 0 and 'declare -a F22_LAUNCH_CELLS=([0]="none" [1]="arditi")' in r.stdout and "PATH=lora_native" in r.stdout and "MODE=auto" in r.stdout, r.stdout + r.stderr
    assert f"serving path: not pinned by the row (unpinned:no-store; wave root {wave}; pins {blind})" in r.stdout and "[serving-path] AUTO-selected lora_native: 1 of 1 steered cells natively servable; skipped by name: none" in r.stdout
    assert _lc_selection(r.stdout)["cells_by_disposition"] == {"passthrough": ["none"], "serve": ["arditi"], "skip": []}
    root, rung_root = wave / arditi["out_root"], wave / arditi["rung_out_root"]
    _lc_store(root, "lens")
    r = _g8k_select("g8k_select_serving_path gemma4e4b main arditi\nf22_select_serving_path\n" + LC_REPORT, **WB)
    assert r.returncode == 0 and "PATH=lens" in r.stdout and "MODE=explicit" in r.stdout and f"serving path: lens — pinned by the ROW (store:lens(disk); wave root {wave}" in r.stdout, r.stdout + r.stderr
    assert _lc_selection(r.stdout)["cells_by_disposition"] is None
    r = _g8k_select("g8k_select_serving_path gemma4e4b main arditi\nf22_select_serving_path\n" + LC_REPORT, **W)          # disk + the records
    assert r.returncode == 0 and "PATH=lens" in r.stdout and "(store:lens(disk+pins)" in r.stdout, r.stdout + r.stderr
    r = _g8k_select("g8k_select_serving_path gemma4e4b main arditi\nf22_select_serving_path\n" + LC_REPORT, F22_SERVING_PATH="lora_native", **WB)
    assert r.returncode == 1 and "FATAL the row pins serving path 'lens' (store:lens(disk)" in r.stdout and "refused by name" in r.stdout, r.stdout + r.stderr
    r = _g8k_select("g8k_select_serving_path ARRAY\nf22_select_serving_path\n" + LC_REPORT, SLURM_ARRAY_TASK_ID=str(arditi["index"]), **WB)
    assert r.returncode == 0 and "PATH=lens" in r.stdout and "(store:lens(disk)" in r.stdout, r.stdout + r.stderr
    shutil.rmtree(root); _lc_store(rung_root, "lens")                                  # the rung sibling alone is the row's store too
    r = _g8k_select("g8k_select_serving_path gemma4e4b main arditi\nf22_select_serving_path\n" + LC_REPORT, **WB)
    assert r.returncode == 0 and "PATH=lens" in r.stdout and "(store:lens(disk)" in r.stdout, r.stdout + r.stderr
    shutil.rmtree(rung_root); _lc_store(root, "native")
    r = _g8k_select("g8k_select_serving_path gemma4e4b main arditi\nf22_select_serving_path\n" + LC_REPORT, **WB)
    assert r.returncode == 0 and "PATH=lora_native" in r.stdout and "MODE=auto" in r.stdout and f"not pinned by the row (store:native(disk); wave root {wave}" in r.stdout, r.stdout + r.stderr
    r = _g8k_select("g8k_select_serving_path gemma4e4b main arditi\nf22_select_serving_path\n" + LC_REPORT, F22_SERVING_PATH="lens", **WB)
    assert r.returncode == 1 and "FATAL the row's steer store on record is NATIVE-stamped (store:native(disk)" in r.stdout and "refused by name" in r.stdout and not [l for l in r.stdout.splitlines() if l.startswith("PATH=")], r.stdout + r.stderr
    r = _g8k_select("g8k_select_serving_path gemma4e4b main arditi\nf22_select_serving_path\n" + LC_REPORT, **W)          # native disk vs the records' lens
    assert r.returncode == 1 and "BOTH engines' stamps" in r.stdout + r.stderr and "pins: ['lens']" in r.stdout + r.stderr, r.stdout + r.stderr
    _lc_store(rung_root, "lens")                                                       # a native cap root beside a lens rung root: mixed
    r = _g8k_select("g8k_select_serving_path gemma4e4b main arditi\nf22_select_serving_path\n" + LC_REPORT, **WB)
    assert r.returncode == 1 and "BOTH engines' stamps" in r.stdout + r.stderr and "FATAL g8k_cells.py:" in r.stdout and "refused the pin" in r.stdout, r.stdout + r.stderr


def _lc_pins(path: Path, engines: dict | None = None) -> Path:
    """A SYNTHETIC store-pins record: `_meta` (+ an engine block naming `engines`, leg key -> lens | lora_native, when given) and no leg."""
    doc = {"_meta": {"schema": "test_store_pins/2", "note": "synthetic"}}
    if engines is not None:
        doc["_meta"]["serving_engine_by_leg"] = {k: {"engine": e, "producer_tree_sha": "1" * 40, "pulled_store": "pull_x/" + k, "readings": [], "stamped": "2026-01-01"} for k, e in engines.items()}
    path.write_text(json.dumps(doc, indent=1))
    return path


def _lc_store(root: Path, engine: str) -> None:
    """A SYNTHETIC steer store (the drivers' shape, never a real store's bytes): the lens record's identity with no engine stamp; the
    native record's `+lora+` identity stamped `serving_engine: lora_native` on the cell route."""
    ident = "proxy_vllm+lens+interleaved" if engine == "lens" else "proxy_vllm+lora+interleaved"
    cell = root / "cells" / "arditi__controls__t0__a0__11111111__22222222"
    cell.mkdir(parents=True, exist_ok=True)
    route = {"virtual_model": "petri__arditi", "serving_identity": ident, **({"serving_engine": "lora_native"} if engine == "native" else {})}
    (cell / "manifest.json").write_text(json.dumps({"cell_id": "arditi", "route": route}))
    (root / "evidence.json").write_text(json.dumps({"serving_identity": ident, "engine": "vllm"}))


def test_g8k_known_bad_an_init_with_no_declaration_and_no_manifest_is_refused_by_name_and_a_rowless_init_still_defers():
    """The KNOWN-BAD (the serving default's own refusal, through the launcher's slice): no g8k_select_serving_path, no manifest, no wave root ->
    refused naming the caller and every way to declare. The row-less callers (the smoke, the overlap leg, the ladder plan, the prod dry
    run) declare nothing on purpose and DEFER through the wave root they own — pinned in g8k_init's text: the literals and the root come
    first (the declaration reads the row's store under the root, arditi-native-pin), the declaration next, all before f22_init."""
    r = _g8k_select("f22_select_serving_path\n" + LC_REPORT)
    assert r.returncode == 1 and "FATAL serving-path selection: bash declared no cell source" in r.stdout and not [l for l in r.stdout.splitlines() if l.startswith("PATH=")], r.stdout + r.stderr
    for needle in ("F22_LAUNCH_CELLS=(<harness ids>; empty = this job stands up no target)", "F22_LAUNCH_MANIFEST=<the launch's manifest>",
                   "F22_SERVING_PATH_DEFER=1", "F22_WAVE_ROOT", "set F22_SERVING_PATH explicitly"):
        assert needle in r.stdout, needle
    r = _g8k_select("export F22_WAVE_ROOT=/dev/null/owned-root\nf22_select_serving_path\n" + LC_REPORT)
    assert r.returncode == 0 and "MODE=deferred" in r.stdout and "F22_LAUNCH_CELLS undeclared" in r.stdout and "DEFERRED to the standup: this job owns its wave root" in r.stdout, r.stdout + r.stderr
    src = (G8K / "g8k_common.sh").read_text()
    init = src[src.index("g8k_init(){"):src.index("\n}\n", src.index("g8k_init(){"))]
    assert init.index("g8k_literals") < init.index("export F22_WAVE_ROOT=") < init.index('[ $# -ge 1 ] && g8k_select_serving_path "$@"') < init.index('f22_init "$sha"')
    select = src[src.index("g8k_select_serving_path(){"):src.index("g8k_init(){")]
    assert select.index('root="${F22_WAVE_ROOT:-}"') < select.index('--serving-path --wave-root "$root" ${pins_args[@]+"${pins_args[@]}"})') < select.index("--launch-cells)") < select.index("F22_LAUNCH_CELLS=()") < select.index('export F22_SERVING_PATH="$pin"')
    assert select.count("--wave-root") == 2 and "G8K_SERVING_PATH_REASON=" in select and "reason=%s" in src[src.index("g8k_print_plan(){"):]
    for name in ("smoke_not_available.sbatch", "overlap_off.sbatch", "ladder_plan.sbatch", "dry_run_prod.sbatch"):
        text = (G8K / name).read_text()
        assert 'g8k_init "$WAVE_SHA"\n' in text and "F22_LAUNCH_CELLS" not in text, name


def test_g8k_dry_run_declares_the_launch_cells_and_prints_the_selected_path_not_a_retyped_default(tmp_path, monkeypatch):
    """The dry-run branch runs the SAME f22_select_serving_path over the same declaration: SERVING_PATH: is the path f22_init fixes
    (selection=explicit on the row's pin, =auto otherwise; the unpinned PSR row reads NATIVE by KIND since native-psr-serving)
    WITH the pin's reason (arditi-native-pin), and LAUNCH_CELLS: the declared cells. The Arditi row follows the store under the dry run's
    shared root: none -> unpinned:no-store and AUTO native; a lens store -> store:lens, EXPLICIT (the rung leg reads the cap root too)."""
    steer = _lc_row(model_key="gemma4e4b", cell="alphasteer")
    text, rc, raw = _dry(steer, s=20, shared=tmp_path / "shared")
    assert rc == 0, raw[-1500:]
    assert _line(text, "SERVING_PATH") == "SERVING_PATH: path=lens pinned_by_row=lens reason=kind:alphasteer selection=explicit"
    assert _line(text, "LAUNCH_CELLS") == "LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none alphasteer" and "[serving-path] EXPLICIT lens" in raw
    none = _lc_row(model_key="gemma4e4b", view="main", cell="none")
    text, rc, raw = _dry(none, s=20, shared=tmp_path / "shared")
    assert rc == 0 and _line(text, "SERVING_PATH") == "SERVING_PATH: path=lora_native pinned_by_row=none reason=unpinned:not-external selection=auto", raw[-1500:]
    assert _line(text, "LAUNCH_CELLS") == "LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none" and "AUTO-selected lora_native: passthrough-only launch" in raw
    psr = _lc_row(model_key="gemma4e4b", view="main", cell="constv2-mix-full__psr_ce_all")
    text, rc, raw = _dry(psr, s=20, shared=tmp_path / "shared")
    assert rc == 0 and _line(text, "SERVING_PATH") == "SERVING_PATH: path=lora_native pinned_by_row=none reason=unpinned:not-external selection=auto", raw[-1500:]
    assert _line(text, "LAUNCH_CELLS") == f"LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none {psr['standup_harness_cell']}" and "AUTO-selected lora_native: 1 of 1 steered cells natively servable; skipped by name: none" in raw
    d = [x for x in M.matrix_rows(M.resolve_matrix(None)) if x["declared"] and "lora" in x["cell"]][0]
    text, rc, raw = _dry(d, s=20, shared=tmp_path / "shared")
    assert rc == 0 and _line(text, "LAUNCH_CELLS") == f"LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none {d['standup_harness_cell']}" and d["standup_harness_cell"].endswith("@thinking"), raw[-1500:]
    assert _line(text, "SERVING_PATH") == "SERVING_PATH: path=lora_native pinned_by_row=none reason=unpinned:not-external selection=auto"
    arditi = _lc_row(model_key="gemma4e4b", cell="arditi")
    text, rc, raw = _dry(arditi, s=20, shared=tmp_path / "shared")                         # no disk store; the repo's records: lens by the pins
    assert rc == 0 and _line(text, "SERVING_PATH") == "SERVING_PATH: path=lens pinned_by_row=lens reason=store:lens(pins) selection=explicit", raw[-1500:]
    assert _line(text, "LAUNCH_CELLS") == "LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none arditi" and "[serving-path] EXPLICIT lens" in raw
    monkeypatch.setenv("G8K_STORE_PINS", str(_lc_pins(tmp_path / "blind.json")))          # the disk alone (the test seam, logged)
    text, rc, raw = _dry(arditi, s=20, shared=tmp_path / "shared")
    assert rc == 0 and _line(text, "SERVING_PATH") == "SERVING_PATH: path=lora_native pinned_by_row=none reason=unpinned:no-store selection=auto", raw[-1500:]
    assert "AUTO-selected lora_native: 1 of 1 steered cells natively servable; skipped by name: none" in raw and "pins " + str(tmp_path / "blind.json") in raw
    _lc_store(tmp_path / "shared" / M.WAVE_ROOT_REL / arditi["out_root"], "lens")
    for rung in (M.CAP, M.RUNG_CAP):
        text, rc, raw = _dry(arditi, rung=rung, s=20, shared=tmp_path / "shared")
        assert rc == 0 and _line(text, "SERVING_PATH") == "SERVING_PATH: path=lens pinned_by_row=lens reason=store:lens(disk) selection=explicit", (rung, raw[-1500:])
    monkeypatch.delenv("G8K_STORE_PINS")
    text, rc, raw = _dry(arditi, s=20, shared=tmp_path / "shared")                         # disk + the records
    assert rc == 0 and _line(text, "SERVING_PATH") == "SERVING_PATH: path=lens pinned_by_row=lens reason=store:lens(disk+pins) selection=explicit", raw[-1500:]
    # the dry-run branch runs the selector on python3 as f22_init does — never on $CPY, a placeholder path (PY) in these tests; and it
    # sets F22_WAVE_ROOT from its stub root before the declaration reads the store under it
    src = (G8K / "gsm8k_thinking.sbatch").read_text()
    dry = src[src.index('if [ "${G8K_DRY_RUN:-0}" = 1 ]; then'):src.index("  exit 0\nfi\n")]
    assert dry.count("f22_select_serving_path\n") == 1 and "F22_SELECT_PY=" not in dry
    assert dry.index('F22_WAVE_ROOT="$WAVE_ROOT"') < dry.index('g8k_select_serving_path "${ROW_ARGS[@]}"') < dry.index("\n  f22_select_serving_path\n") < dry.index('g8k_row "${ROW_ARGS[@]}"')


# --------------------------------------------------------------------------- #
# arditi-allpos-cap (2026-09-17): the ALL-POSITIONS Arditi row — a FOURTH matrix, ONE row, on the INSTRUCT-DIRECT view #
# --------------------------------------------------------------------------- #
# The arm is `arditi_allpos`: the Arditi refusal direction served at EVERY position in both phases, dosed as a RAW COEFFICIENT
# (c = 0.3) baked into a `vector_all` object, on the graph-captured NATIVE residual slot. It is NOT the `arditi` cell of the
# published matrix's steer wave (a `static_r` object, prefill-only, dosed as a residual-norm fraction): two interventions in two
# units, never pooled, never filed under one another's id. The row's view is `it`, so it is its own matrix under its own wave
# root — the published document, its counts and its row indices do not move, and "a cell that has never been run has no store"
# is structural rather than asserted.
#
# THE SERVING-SIDE REGISTRATION IS ANOTHER AUTHOR'S PR. Until `arditi_allpos` is a cell id this tree's harness grammar admits
# (`defense_cells.parse_cell_id` / `registry_adapter.harness_cell_id` / `contracts.cell_spellings`), MATRIX GENERATION cannot
# run here — it derives the row's harness and standup spellings through that grammar. Everything that reads the COMMITTED
# matrix is stdlib and runs regardless; the generation-dependent assertions skip BY NAME, so the day the registration lands
# they run with no edit here.
ALLPOS_MATRIX = M.MATRICES[M.MATRIX_ALLPOS_NAME]
ALLPOS_VIEWS = M.VIEWS_BY_MATRIX[M.MATRIX_ALLPOS_NAME]
ALLPOS_ALL_ROWS = M.matrix_rows(ALLPOS_MATRIX)
ALLPOS_CELL = M.CELL_ARDITI_ALLPOS
ALLPOS_TAG = "qwen35_4b_it"
#: the row's own store-pins record (allpos-store-pins, 2026-09-17): registered LAST in STORE_PINS_PATHS, both legs at their launch tree
ALLPOS_PINS = M.STORE_PINS_PATHS[-1]
ALLPOS_LEG_KEYS = ("qwen35_4b/it/arditi_allpos@8192", "qwen35_4b/it/arditi_allpos@16384")
#: the engine BOTH legs' LAUNCH RECORDS name (standup_jobs.jsonl `engine` of jobs 544615 / 544730): what the stamp must derive from the pulled stores
ALLPOS_LAUNCH_ENGINE = "lora_native"


def _allpos_engine_entries() -> dict | None:
    """The checkout record's `_meta.serving_engine_by_leg` entries for the row's two legs — None while the record ADMITS them without an
    engine (the stamp has not run on this tree: the launchers then refuse the legs by name), the two entries once it has; a record stamped
    for one leg but not the other is refused here (never read as either state)."""
    block = (json.loads(ALLPOS_PINS.read_text()).get("_meta") or {}).get("serving_engine_by_leg") or {}
    have = {k: block.get(k) for k in ALLPOS_LEG_KEYS}
    if all(v is None for v in have.values()):
        return None
    assert all(v is not None for v in have.values()), f"half-stamped record: {have}"
    return have


def _allpos_stamped_copy(path: Path, engine: str = ALLPOS_LAUNCH_ENGINE) -> Path:
    """A SYNTHETIC stamped twin of the REAL record: its leg -> tree entries verbatim, plus the engine block the stamp script will write once
    the stores are pulled (placeholder provenance) — the tests' seam for the launcher's post-stamp mechanics, never the record's bytes."""
    doc = json.loads(ALLPOS_PINS.read_text())
    doc["_meta"]["serving_engine_by_leg"] = {k: {"engine": engine, "producer_tree_sha": doc[k], "pulled_store": "pull_x/" + k, "readings": [], "stamped": "2026-01-01"}
                                             for k in ALLPOS_LEG_KEYS}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))
    return path


def _allpos_stamp_pending_xfail(raw: str) -> None:
    """The checkout's record admits both legs but carries no engine yet: the pin REFUSES by leg name (never `no-store` — a known store is not
    relaunched on a guessed engine). Asserted, then filed as an expected failure that names the pending stamp: the run shows it (AC-13), and
    the test passes with no edit once stamp_store_pins_engines.py --matrix arditi_allpos has run on this branch."""
    assert "refused:record-engine-unknown" in raw and ALLPOS_LEG_KEYS[0] in raw and str(ALLPOS_PINS) in raw, raw[-1500:]
    pytest.xfail("store_pins_arditi_allpos.json admits both legs without an engine entry — the engine stamp (stamp_store_pins_engines.py "
                 "--launcher gsm8k --matrix arditi_allpos over the pulled stores) has not run on this tree; the checkout read refuses by leg name")


def _assert_allpos_native_plan(text: str, raw: str, r: dict, reason: str) -> None:
    """The plan the sbatch prints for the row when its pin leaves the kind-aware default to select the native slot — `reason` is why the row
    is unpinned (`unpinned:no-store`: no store on record; `store:native(pins)`: the record names a native store)."""
    assert f"serving path: not pinned by the row ({reason};" in raw
    assert "the kind-aware default selects it at f22_init BY KIND over the declared launch cells: none arditi_allpos (F22_LAUNCH_CELLS)" in raw
    assert "serving path DECLARED by the row's record: lora_native" in raw
    assert _line(text, "SERVING_PATH") == f"SERVING_PATH: path=lora_native pinned_by_row=none reason={reason} selection=auto"
    assert _line(text, "LAUNCH_CELLS") == f"LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=none {ALLPOS_CELL}"
    assert _line(text, "WAVE").startswith(f"WAVE: wave=allpos status=launch object_source=external bank_view=ar_c0p3 ")
    assert f"config=configs/sdfbase/gsm8k/sdfbase_full22_{ALLPOS_TAG}.gsm8k_{ALLPOS_CELL}.yaml" in _line(text, "WAVE")
    assert f"--cell {ALLPOS_CELL} --expect-object-sha256 {ALLPOS_CELL}={r['object_sha256']}" in _line(text, "STANDUP")
    assert f"{M.FLAG_BANK_VIEW} it " in _line(text, "DRIVER") and f" {M.FLAG_LENS_LIVENESS} {M.LENS_LIVENESS_STEER} " in _line(text, "DRIVER") + " "
    assert f"/exp/gsm8k_paired200_paper_objects_arditi_allpos/controls_allpos/{ALLPOS_TAG}/{ALLPOS_CELL}/thinking_card " in _line(text, "DRIVER")
    assert f"OUT OF SUPPORT: tuple={r['tuple_name']} pin={r['object_sha256']}" in text and "_steer_allpos_8192" in r["tuple_name"]
    assert _line(text, "SELECTION").startswith(f"SELECTION: outcome={M.SELECTION_OUTCOME_ALLPOS_USER_OVERRIDE} grid_limited=0 label=")


def _allpos_grammar_landed() -> bool:
    """Whether THIS tree's harness grammar admits the all-positions cell id (the other author's serving-side registration)."""
    try:
        C.cell_spellings(ALLPOS_CELL, declared=False)
    except Exception:
        return False
    return True


def _need_allpos_grammar():
    if not _allpos_grammar_landed():
        pytest.skip(f"this tree's harness grammar does not admit {ALLPOS_CELL!r} yet (the serving-side registration — "
                    "defense_cells.SERVED_ONLY_CLASS_B_IDS / registry_adapter.ARM_OBJECT_KINDS — lands in another author's PR); "
                    "the committed matrix and every stdlib reader of it are checked above")


def _dry_allpos(row, *, rung=M.CAP, s, shared, cluster="cluster_a", array_index=None, env_extra=None):
    env = {**os.environ, "G8K_DRY_RUN": "1", "G8K_DRY_S": str(s), "G8K_DRY_SHARED": str(shared), "G8K_DRY_CPY": PY,
           "G8K_DRY_CLUSTER": cluster, "G8K_REPO_DIR": str(REPO), "USER": "tester", **(env_extra or {})}
    if array_index is None:
        argv = ["bash", str(G8K / "gsm8k_thinking.sbatch"), row["model_key"], row["view"], row["cell"], str(rung), SHA0,
                str(int(row["fitted"])), str(int(row["declared"])), M.MATRIX_ALLPOS_NAME]
    else:
        env["SLURM_ARRAY_TASK_ID"] = str(array_index)
        argv = ["bash", str(G8K / "gsm8k_thinking.sbatch"), "ARRAY", SHA0, str(rung), M.MATRIX_ALLPOS_NAME]
    r = subprocess.run(argv, capture_output=True, text=True, env=env, check=False)
    return "\n".join(line for line in r.stdout.splitlines() if not line.startswith("[")), r.returncode, r.stdout + r.stderr


def test_allpos_the_row_is_the_admission_records_object_on_the_instruct_direct_view_with_its_selection_and_fit_provenance():
    """THE ROW (arditi-allpos-cap): one row, `it` view, TAG `qwen35_4b_it`, the P8 admission record's object — its pin, its bank
    view `ar_c0p3` and staged id, its lane root, the rung's `_steer_<cap>` tuple (the analysis' `expected_tuple_name` for
    every external row) — carrying on ITSELF the record's `selection` block (both readings, the budget, the LADDER CAP that
    identifies the reading, the single-trial flag, the dose as a raw coefficient, `alpha_equiv` labelled informational, the
    replicate note) and its `fit_provenance` block, because that is where every table, figure and provenance line reads them."""
    adm = M.load_steer_admission()
    e = adm[("qwen35_4b", ALLPOS_CELL)]
    assert len(ALLPOS_ALL_ROWS) == 1
    r = ALLPOS_ALL_ROWS[0]
    doc = M.matrix_doc(ALLPOS_MATRIX)
    assert doc["schema"] == M.MATRIX_SCHEMA_2 and doc["wave_root_rel"] == "exp/gsm8k_paired200_paper_objects_arditi_allpos"
    assert doc["wave"] == M.WAVE_ALLPOS and doc["row_overlays"] is True and doc["trigger_threshold"] == 0.01
    assert doc["source"]["steer_admission"] == str(M.STEER_ADMISSION_REL) and doc["source"]["steer_admission_sha256"] == _sha256(M.STEER_ADMISSION_PATH)
    # the row's own facts
    assert (r["index"], r["model_key"], r["view"], r["tag"], r["cell"]) == (0, "qwen35_4b", "it", ALLPOS_TAG, ALLPOS_CELL)
    assert r["job_id"] == "g8k-qwen35_4b-it-arditi_allpos" and r["status"] == M.STATUS_LAUNCH and r["held_reason"] is None
    assert r["object_source"] == M.OBJECT_SOURCE_EXTERNAL and r["wave"] == M.WAVE_ALLPOS and r["own_overlay"] is True
    assert r["fitted"] and not r["declared"] and r["served_positions_expected"] == "all" and r["route_kind"] == "lens"
    assert r["object_sha256"] == e["object_sha256"] == "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2"
    assert r["staged_id"] == e["staged_id"] == "ar_c0p3__arditi_allpos" and r["bank_view"] == e["view"] == "ar_c0p3"
    assert r["bank_cell_dir"] == e["bank_cell_dir"] == ALLPOS_CELL and r["lane_root"] == e["lane_root"] == "exp/objects/arditi_allpos"
    assert r["bank_view_rel"] == e["bank_view_rel"] == {role: "exp/objects/arditi_allpos/qwen35_4b/ladder/cells_view/ar_c0p3" for role in ("primary", "overflow")}
    assert all(not str(rel).startswith("/") for rel in r["bank_view_rel"].values())      # relative to the shared root, never a cluster path
    assert r["out_root"] == f"controls_allpos/{ALLPOS_TAG}/{ALLPOS_CELL}/thinking_card" and r["rung_out_root"] == r["out_root"] + "__rung16384"
    assert r["time_limit"] == M.TIME_FITTED and r["cluster_constraint"] == M.PLACEMENT_PRIMARY and r["thinking_mode"] == "on"
    assert r["expected_stamps"] == list(C.g8k_expected_stamps(fitted=True, declared=False)) == ["cell-stable", "always", "thinking_on"]
    # the tuple is this ARM'S OWN, never the rung's — an OOS tuple authorises a DESCRIBED intervention, and the
    # `_steer` one describes a prefill-only, residual-norm-fraction arm inside its budget, which this is not (see
    # test_allpos_the_arm_has_its_own_oos_tuple…). The analysis' reader resolves the same name for the row.
    assert r["tuple_name"] == M.external_tuple_name("qwen35_4b", ALLPOS_CELL, rung=M.CAP) == "gsm8k_paired200_qwen35_4b_steer_allpos_8192"
    assert r["tuple_names"] == {"8192": "gsm8k_paired200_qwen35_4b_steer_allpos_8192", "16384": "gsm8k_paired200_qwen35_4b_steer_allpos_16384"}
    assert r["tuple_name"] == VAL.expected_tuple_name(r, M.CAP) and r["tuple_names"]["16384"] == VAL.expected_tuple_name(r, M.RUNG_CAP)
    assert r["tuple_name"] != M.steer_tuple_name("qwen35_4b", rung=M.CAP) and "_steer_allpos_" in r["tuple_name"]
    # THE LEG KEYS, pinned across the lane: `<model_key>/<view>/<cell>@<cap>` with the ROW's own view
    assert M.leg_keys(r) == ["qwen35_4b/it/arditi_allpos@8192", "qwen35_4b/it/arditi_allpos@16384"]
    # the SELECTION block on the row, verbatim from the record: the outcome word, BOTH readings, the budget, the ladder cap,
    # the single-trial flag, the dose in RAW COEFFICIENTS, alpha_equiv informational only
    sel = r["selection"]
    assert sel == e["selection"] and sel["outcome"] == r["selection_outcome"] == M.SELECTION_OUTCOME_ALLPOS_USER_OVERRIDE
    assert sel["rule_returns"] == "ar_c0p2" and sel["rule_reading_pp"] == 5.04 and sel["override_reading_pp"] == 10.26
    assert sel["budget_pp"] == 10 and sel["override_reading_pp"] > sel["budget_pp"] > sel["rule_reading_pp"]   # the pin is OVER budget
    assert sel["ladder_cap"] == 2048 and sel["single_trial"] is True
    assert sel["dose"] == {"value": 0.3, "unit": M.DOSE_UNIT_RAW_COEFFICIENT} and r["alpha"] is None
    assert sel["alpha_equiv"]["informational"] is True and sel["alpha_equiv"]["value"] != sel["dose"]["value"]
    assert "replicate" in " ".join(sel) or sel.get("replicate_note")
    # nothing on the row may say the RULE chose this dose, and no significance claim may ride a single-trial ladder
    blob = json.dumps(r).lower()
    assert "p-value" not in blob and "p_value" not in blob and "statistically significant" not in blob
    assert "user_override" in blob and "over_budget" in blob
    # the FIT PROVENANCE block, likewise on the row (the paper's provenance line reads it from here)
    fp = r["fit_provenance"]
    assert fp == e["fit_provenance"] and fp["fitted_on_model"] == "Qwen/Qwen3.5-4B" and fp["served_layer"] == 12 == fp["fit_layer"] - 1
    assert fp["realised_counts"]["harmful_train"] == 123 and fp["direction_sha256"] and fp["fit_record_sha256"]
    # served on the NATIVE residual slot in every leg, and the arm's record says so where the launcher reads it
    assert r["attestation_expect"]["serving_path"] == "lora_native" == M.declared_serving_path(r)
    assert r["attestation_expect"]["positions"] == "all" and r["attestation_expect"]["phase"] == "both"
    assert M.external_row_problem(r) is None and M.allpos_row_problem(r) is None
    # the standup is SINGLE-OBJECT by construction: `none` (implicit) + this one cell, in harness spelling
    assert M.launch_cells_for(r) == ["none", ALLPOS_CELL] and r["standup_harness_cell"] == r["harness_cell"] == ALLPOS_CELL
    # the analysis' REAL reader admits the document — it runs the harness grammar over every evidence row's cell, so like
    # generation it waits on the serving-side registration (another author's PR); everything above is stdlib and runs today
    _need_allpos_grammar()
    assert VAL.load_matrix(ALLPOS_MATRIX)["counts"]["launched"] == 1


def test_allpos_the_cell_is_not_the_arditi_cell_and_the_vocabulary_refuses_every_other_arms_word():
    """The arm is APPENDED to the rung's cells (the eight-arm panel order of the two before it does not move), sits on its OWN
    declared view, and its selection-outcome vocabulary is exactly one word — null and every AlphaSteer / Arditi word refused,
    each by name. KNOWN-BAD TWINS: the wrong word, an alpha-shaped dose, and a row of this arm on the panel view."""
    assert M.STEER_CELLS == (M.CELL_ALPHASTEER, M.CELL_ARDITI, ALLPOS_CELL) and ALLPOS_CELL != M.CELL_ARDITI
    assert M.EXTERNAL_ROW_VIEW_BY_CELL == {M.CELL_ALPHASTEER: "main", M.CELL_ARDITI: "main", ALLPOS_CELL: "it"}
    assert M.external_row_view_for(ALLPOS_CELL) == "it" and M.external_row_view_for(M.CELL_ARDITI) == "main"
    with pytest.raises(SystemExit, match="not one of the rung's cells"):
        M.external_row_view_for("arditi_allpositions")
    # the ONE admissible word; null and every other arm's word refused BY NAME
    assert M.selection_outcome_problem(ALLPOS_CELL, M.SELECTION_OUTCOME_ALLPOS_USER_OVERRIDE) is None
    assert M.ARDITI_ALLPOS_SELECTION_OUTCOMES == (M.SELECTION_OUTCOME_ALLPOS_USER_OVERRIDE,)
    for bad in (None, "", "chosen_by_rule", M.SELECTION_OUTCOME_RULED, *M.SELECTION_OUTCOMES):
        problem = M.selection_outcome_problem(ALLPOS_CELL, bad)
        assert problem and repr(bad) in problem and M.SELECTION_OUTCOME_ALLPOS_USER_OVERRIDE in problem, bad
    # …and the word is this arm's alone: it is refused on both of the other arms
    for other in (M.CELL_ALPHASTEER, M.CELL_ARDITI):
        assert M.selection_outcome_problem(other, M.SELECTION_OUTCOME_ALLPOS_USER_OVERRIDE)
    r = ALLPOS_ALL_ROWS[0]
    # KNOWN-BAD: a wrong outcome word on the row itself
    for bad in (None, M.SELECTION_OUTCOME_RULED, M.SELECTION_OUTCOME_CHOSEN):
        assert M.external_row_problem({**r, "selection_outcome": bad, "selection": {**r["selection"], "outcome": bad}})
    # KNOWN-BAD: an ALPHA-shaped dose — the unit swapped, the dose replaced by an alpha, or an alpha beside it
    assert "raw_coefficient" in M.allpos_row_problem({**r, "selection": {**r["selection"], "dose": {"value": 0.3, "unit": M.DOSE_UNIT_RESIDUAL_NORM_FRACTION}}})
    assert M.allpos_row_problem({**r, "selection": {**r["selection"], "dose": {"value": 0.105, "unit": "alpha"}}})
    assert "no `alpha` dose" in M.allpos_row_problem({**r, "alpha": 0.10540806493282855})
    assert "informational" in M.allpos_row_problem({**r, "selection": {**r["selection"], "alpha_equiv": {"value": 0.105}}})
    # KNOWN-BAD: the single-trial flag dropped, the ladder cap dropped, the blocks dropped
    assert "single_trial" in M.allpos_row_problem({**r, "selection": {**r["selection"], "single_trial": False}})
    assert "ladder_cap" in M.allpos_row_problem({**r, "selection": {k: v for k, v in r["selection"].items() if k != "ladder_cap"}})
    assert "`selection` block" in M.allpos_row_problem({k: v for k, v in r.items() if k != "selection"})
    assert "fit_provenance" in M.allpos_row_problem({k: v for k, v in r.items() if k != "fit_provenance"})
    # KNOWN-BAD: this arm's row on the PANEL view (and an existing arm's row on the instruct-direct one)
    assert "sits on the 'it' view" in M.external_row_problem({**r, "view": "main", "tag": "qwen35_4b"})
    main_row = M.row_for("qwen35_4b", "main", M.CELL_ARDITI, ALL_ROWS)
    assert "sits on the 'main' view" in M.external_row_problem({**main_row, "view": "it", "tag": "qwen35_4b_it"})
    assert "tag" in M.external_row_problem({**r, "tag": "qwen35_4b"})      # the TAG follows the view, always


def test_allpos_the_views_file_refuses_a_steering_entry_on_the_wrong_view_and_an_unexpressed_one(tmp_path):
    """The views grammar (KNOWN-BAD twins): a `steering:` entry may name a `view:`, but only the one its ARM declares and only
    one the `views:` block expresses; the default is `main`, so every entry written before this is unchanged."""
    spec = V.load_views_yaml(ALLPOS_VIEWS)
    assert spec.wave == M.WAVE_ALLPOS and spec.row_overlays and spec.views == {"qwen35_4b": ["it"]}
    assert [(e["key"], e.get("view"), e["arms"]) for e in spec.steering] == [("qwen35_4b", "it", [ALLPOS_CELL])]
    # the published views file names no `view:` and still reads `main` — byte-identical behaviour
    published = V.load_views_yaml(M.VIEWS_BY_MATRIX[M.MATRIX_DEFAULT_NAME])
    assert all("view" not in e for e in published.steering) and published.steering == M.matrix_doc()["steering"]
    base = yaml.safe_load(ALLPOS_VIEWS.read_text())

    def loads(mutate):
        doc = copy.deepcopy(base)          # NOT a json round trip: `since:` parses as a date, which json cannot carry
        mutate(doc)
        p = tmp_path / "views_bad.yaml"
        p.write_text(yaml.safe_dump(doc))
        return V.load_views_yaml(p)

    # a view the model does not express (this matrix expresses `it` alone)
    with pytest.raises(SystemExit, match="does not express that view"):
        loads(lambda d: d["steering"][0].__setitem__("view", "main"))
    # …the same refusal reached from the other side: the arm's declared view is `it`, so `main` is also the wrong view for it
    with pytest.raises(SystemExit, match="does not express that view|sits on the 'it' view"):
        loads(lambda d: (d["views"].__setitem__("qwen35_4b", ["it", "main"]), d["steering"][0].__setitem__("view", "main")))
    # a PANEL-view arm dragged onto this matrix's instruct-direct view
    with pytest.raises(SystemExit, match="sits on the 'main' view"):
        loads(lambda d: d["steering"][0].__setitem__("arms", [M.CELL_ARDITI]))
    # an arm that is not a rung cell at all
    with pytest.raises(SystemExit, match="arms must be distinct members"):
        loads(lambda d: d["steering"][0].__setitem__("arms", ["arditi_allpositions"]))


def test_allpos_the_overlay_pair_is_the_it_tags_body_with_exactly_two_cells_and_regenerates_byte_identically(tmp_path):
    """The row serves through its OWN pair, cut from the `it` TAG's campaign config (never the `main` body the steering pairs
    use, and never the published `it` TAG overlay): `explicit_cells: [none, arditi_allpos]`, everything else byte for byte. The
    generator is the only writer (`--check`), and the KNOWN-BAD twin — a second object hand-added to the standup's cells —
    drifts from it by name."""
    cfg_path = REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_{ALLPOS_TAG}.gsm8k_{ALLPOS_CELL}.yaml"
    man_path = REPO / f"configs/capeval/manifests/petri_full22_{ALLPOS_TAG}.gsm8k_{ALLPOS_CELL}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    campaign = yaml.safe_load((REPO / f"configs/sdfbase_full22_{ALLPOS_TAG}.yaml").read_text())
    assert cfg["explicit_cells"] == ["none", ALLPOS_CELL]
    assert {k: v for k, v in cfg.items() if k != "explicit_cells"} == {k: v for k, v in campaign.items() if k != "explicit_cells"}
    man = yaml.safe_load(man_path.read_text())["petri"]
    cman = yaml.safe_load((REPO / f"configs/capeval/manifests/petri_full22_{ALLPOS_TAG}.yaml").read_text())["petri"]
    assert man["cells"] == ["none", ALLPOS_ALL_ROWS[0]["harness_cell"]] == ["none", ALLPOS_CELL]
    assert {k: v for k, v in man.items() if k != "cells"} == {k: v for k, v in cman.items() if k != "cells"}
    # the `it` TAG's body is the one this row's lane serves, so the pair is the TAG's — not the `main` campaign config the
    # published steering pairs are cut from (the two differ in the cell list and the out_dir namespace alone)
    assert cfg["defenses_dir"].endswith("/cells_view/it") and cfg["out_dir"].endswith("_it") and cfg["model"] == campaign["model"]
    # the generator is the only writer, for THIS matrix and for every other (the published pairs cannot move)
    py = sys.executable
    for name in sorted(M.MATRICES):
        r = subprocess.run([py, str(G8K / "make_gsm8k_overlays.py"), "--matrix", name, "--check"], capture_output=True, text=True,
                           cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, check=False)
        assert r.returncode == 0, (name, r.stdout + r.stderr)
        if name == M.MATRIX_ALLPOS_NAME:
            assert "0 TAGs' overlays (+ 0 steering rows' own pairs + 0 it-doc rows' own pairs + 1 own-row pairs)" in r.stdout, r.stdout
    # KNOWN-BAD: a second object in the standup's cells — the pair is regenerated into a scratch configs root with the extra
    # cell hand-added, and the generator refuses it by name (a standup listing two objects would select this arm's by name and
    # skip or serve the other under this row's leg)
    root = tmp_path / "tree"
    for rel in (f"configs/sdfbase_full22_{ALLPOS_TAG}.yaml", f"configs/capeval/manifests/petri_full22_{ALLPOS_TAG}.yaml",
                f"configs/sdfbase/gsm8k/sdfbase_full22_{ALLPOS_TAG}.gsm8k_{ALLPOS_CELL}.yaml",
                f"configs/capeval/manifests/petri_full22_{ALLPOS_TAG}.gsm8k_{ALLPOS_CELL}.yaml"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, root / rel)
    bad = root / f"configs/sdfbase/gsm8k/sdfbase_full22_{ALLPOS_TAG}.gsm8k_{ALLPOS_CELL}.yaml"
    bad.write_text(bad.read_text().replace(f"explicit_cells:\n- none\n- {ALLPOS_CELL}\n",
                                           f"explicit_cells:\n- none\n- {ALLPOS_CELL}\n- {M.CELL_LORA_MIX}\n"))
    r = subprocess.run([py, str(G8K / "make_gsm8k_overlays.py"), "--configs-root", str(root), "--matrix", str(ALLPOS_MATRIX), "--check"],
                       capture_output=True, text=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, check=False)
    assert r.returncode == 1 and "DRIFT" in r.stdout and str(bad) in r.stdout, r.stdout + r.stderr


def test_allpos_the_published_and_the_other_matrices_cannot_move_and_this_rows_pins_live_in_its_own_record():
    """THE REGRESSION: a fourth matrix cannot perturb the three that exist. The published document keeps its 53 rows, their
    indices, their counts and their bytes; the Llama and instruct-direct documents keep theirs; none of them gains an
    all-positions row. And the row's store pins live in a FOURTH record of its own (store_pins_arditi_allpos.json, registered
    last): both legs at the ONE launch tree the launch records name, no other record naming either leg, and no engine entry
    anyone typed (the engine block is the stamp script's to derive from the pulled stores)."""
    frozen = json.loads(FROZEN_FIXTURE.read_text())["rows"]
    assert len(ALL_ROWS) == 53 and ALL_ROWS[:42] == frozen and [r["index"] for r in ALL_ROWS] == list(range(53))
    assert M.counts(ALL_ROWS)["waves"] == {"published": 42, "ext": 5, "steer": 6} and M.published_counts() == COUNTS
    assert [r["index"] for r in M.held_rows(ALL_ROWS)] == [2, 3, 15, 16]
    for rows in (ALL_ROWS, LLAMA_ALL_ROWS, ITDOC_ALL_ROWS):
        assert all(r["cell"] != ALLPOS_CELL for r in rows) and all(r.get("wave") != M.WAVE_ALLPOS for r in rows)
    assert all("selection" not in r and "fit_provenance" not in r for r in ALL_ROWS + LLAMA_ALL_ROWS + ITDOC_ALL_ROWS)
    assert all(not r.get("own_overlay") for r in STEER_ROWS + LLAMA_STEER_ROWS)     # the steering rows keep the model-keyed pair
    # …and they REGENERATE with the same rows and counts: a fourth matrix, a fourth views file and a third steering-entry shape
    # cannot move a row of the three documents that existed. (Their `source.steer_admission_sha256` tracks the SHARED admission
    # record, which grows an entry for this arm — re-stamping that line belongs to whoever edits the record, and is not a row.)
    import make_gsm8k_matrix as MK
    for name in (M.MATRIX_DEFAULT_NAME, M.MATRIX_LLAMA_NAME, M.MATRIX_ITDOC_NAME):
        _text, doc = MK.build(configs_root=REPO, views_yaml=None, views_spec=None, matrix_name=name)
        committed = M.matrix_doc(M.MATRICES[name])
        assert doc["rows"] == committed["rows"] and doc["counts"] == committed["counts"], name
    # the new wave is isolated: its own prefix, its own root, and no other matrix's row under it
    assert M.out_root_prefix_for(M.WAVE_ALLPOS) == "controls_allpos" and M.WAVE_ALLPOS not in M.COMPLETED_WAVES
    assert M.wave_root_rel_of(M.matrix_doc(ALLPOS_MATRIX)) not in {M.WAVE_ROOT_REL, M.wave_root_rel_of(M.matrix_doc(ITDOC_MATRIX)),
                                                                   M.wave_root_rel_of(M.matrix_doc(LLAMA_MATRIX))}
    # THE ROW'S PINS LIVE IN ITS OWN RECORD (allpos-store-pins, 2026-09-17): the fourth record, named after the matrix (the stamp script
    # derives `store_pins_<matrix_name_of>.json`), admits EXACTLY the row's two legs at ONE 40-hex launch tree, its `_meta` naming that tree
    # and both legs; the three other records name neither leg, at either cap, and no record carries an engine entry for them that the stamp
    # script has not derived (test_math500_store_pins_served_steer.py holds the stamp gate)
    r = ALLPOS_ALL_ROWS[0]
    assert [p.stem.removeprefix("store_pins_") for p in M.STORE_PINS_PATHS] == [M.MATRIX_DEFAULT_NAME, M.MATRIX_LLAMA_NAME, M.MATRIX_ITDOC_NAME, M.MATRIX_ALLPOS_NAME]
    assert ALLPOS_PINS.name == f"store_pins_{M.matrix_name_of(ALLPOS_MATRIX)}.json" and M.leg_keys(r) == list(ALLPOS_LEG_KEYS)
    own = json.loads(ALLPOS_PINS.read_text())
    assert own["_meta"]["schema"].startswith("g8k_store_pins/") and {k for k in own if k != "_meta"} == set(ALLPOS_LEG_KEYS)
    tree = own[ALLPOS_LEG_KEYS[0]]
    assert own[ALLPOS_LEG_KEYS[1]] == tree and re.fullmatch(r"[0-9a-f]{40}", tree), tree
    assert own["_meta"]["trees"] == {"allpos": tree} and own["_meta"]["source_by_leg"] == {k: "allpos" for k in ALLPOS_LEG_KEYS}
    assert tree in own["_meta"]["sources_by_tree"]["allpos"] and "standup_jobs.jsonl" in own["_meta"]["sources_by_tree"]["allpos"]
    for path in M.STORE_PINS_PATHS[:-1]:
        doc = json.loads(Path(path).read_text())
        assert not (set(ALLPOS_LEG_KEYS) & set(doc)), path
        assert not (set(ALLPOS_LEG_KEYS) & set((doc.get("_meta") or {}).get("serving_engine_by_leg") or {})), path
    entries = _allpos_engine_entries()
    if entries is not None:                                       # stamped: the launch records' engine, at the pin, for both legs
        assert set(entries) == set(ALLPOS_LEG_KEYS) and all(e["engine"] == ALLPOS_LAUNCH_ENGINE and e["producer_tree_sha"] == tree for e in entries.values()), entries


def test_allpos_the_serving_path_pin_follows_the_record_and_the_row_declares_the_native_slot(tmp_path):
    """THE PIN FOLLOWS THE STORE — and the store is now ON RECORD in the row's own pins record (both legs at their launch tree). The
    launcher's mechanics over a controlled record (the `--store-pins` seam): with a record that knows no engine the disk alone speaks — an
    empty wave root reads `unpinned:no-store` (the kind-aware default then selects the native residual slot at `f22_init`, which is what this
    arm requires), no wave root `unpinned:store-unread`; with the REAL record's entries under a native engine block (the shape the stamp
    script writes) the pin reads `store:native(pins)` with or without a wave root — unpinned, the store's engine named, so the default
    selects native and a lens export is refused downstream; a lens-stamped twin would pin the lens (the reading is the record's, not the
    kind's). The CLI prints each the way the launcher reads it (two lines). The CHECKOUT's record is then read as a real launch reads it:
    refusing by leg name until the engine stamp lands, `store:native(pins)` after. The row's RECORD declares the native slot for every leg
    of the arm either way; every row before it declares none."""
    r = ALLPOS_ALL_ROWS[0]
    root = tmp_path / "wave"
    root.mkdir()
    blind, stamped = _lc_pins(tmp_path / "blind.json"), _allpos_stamped_copy(tmp_path / "stamped.json")
    assert M.serving_path_pin(r, wave_root=root, store_pins=[blind]) == (None, "unpinned:no-store")
    assert M.serving_path_for(r, wave_root=root, store_pins=[blind]) is None and M.serving_path_pin(r, store_pins=[blind]) == (None, "unpinned:store-unread")
    assert M.serving_path_pin(r, wave_root=root, store_pins=[stamped]) == (None, "store:native(pins)") == M.serving_path_pin(r, store_pins=[stamped])
    assert M.serving_path_for(r, wave_root=root, store_pins=[stamped]) is None
    assert M.serving_path_pin(r, wave_root=root, store_pins=[_allpos_stamped_copy(tmp_path / "lens.json", engine="lens")]) == ("lens", "store:lens(pins)")
    cli = [sys.executable, str(G8K / "g8k_cells.py"), "row", "--matrix", M.MATRIX_ALLPOS_NAME, "--index", "0", "--serving-path", "--wave-root", str(root)]
    for rec, want in ((blind, "\nunpinned:no-store\n"), (stamped, "\nstore:native(pins)\n")):
        out = subprocess.run(cli + ["--store-pins", str(rec)], capture_output=True, text=True, cwd=REPO, check=False)
        assert out.returncode == 0 and out.stdout == want, out.stdout + out.stderr
    out = subprocess.run([sys.executable, str(G8K / "g8k_cells.py"), "row", "--matrix", M.MATRIX_ALLPOS_NAME, "--index", "0", "--launch-cells"],
                         capture_output=True, text=True, cwd=REPO, check=False)
    assert out.returncode == 0 and out.stdout == f"none\n{ALLPOS_CELL}\n", out.stdout + out.stderr
    # the row's RECORD declares the native slot for every leg of the arm; every row before this declares none
    out = subprocess.run([sys.executable, str(G8K / "g8k_cells.py"), "row", "--matrix", M.MATRIX_ALLPOS_NAME, "--index", "0", "--declared-serving-path"],
                         capture_output=True, text=True, cwd=REPO, check=False)
    assert out.returncode == 0 and out.stdout == "lora_native\n", out.stdout + out.stderr
    for other in STEER_ROWS + LLAMA_STEER_ROWS:
        assert M.declared_serving_path(other) is None, other["job_id"]
    # the CHECKOUT's records, as a real launch reads them (the CLI without --store-pins reads the same)
    out = subprocess.run(cli, capture_output=True, text=True, cwd=REPO, check=False)
    if _allpos_engine_entries() is None:
        with pytest.raises(SystemExit, match="refused:record-engine-unknown") as exc:
            M.serving_path_pin(r, wave_root=root)
        assert out.returncode == 1 and out.stdout == "" and "refused:record-engine-unknown" in out.stderr, out.stdout + out.stderr
        _allpos_stamp_pending_xfail(str(exc.value) + out.stderr)
    assert M.serving_path_pin(r, wave_root=root) == (None, "store:native(pins)") == M.serving_path_pin(r)
    assert out.returncode == 0 and out.stdout == "\nstore:native(pins)\n", out.stdout + out.stderr


def test_allpos_known_bad_an_explicit_lens_export_on_this_row_is_refused_by_name(tmp_path):
    """KNOWN-BAD: `F22_SERVING_PATH=lens` on this row. The lens WOULD serve a `vector_all` blob, so the leg does not fail — it
    completes and files a different intervention's numbers under this arm's name. The launcher refuses at its FIRST
    row-resolution step, before `f22_init` can fix the export as EXPLICIT. The pin is read first, so the refusal is exercised through
    the pins seam over a record that knows no engine AND over the REAL record's entries under a native engine block (the same refusal:
    the RECORD declares the path), then over the checkout's record (refusing by leg name until the engine stamp lands). A row whose
    record declares no path (every row before this one) is untouched: the same export still wins there."""
    r = ALLPOS_ALL_ROWS[0]
    blind, stamped = _lc_pins(tmp_path / "blind.json"), _allpos_stamped_copy(tmp_path / "stamped.json")
    for rec in (blind, stamped):
        text, rc, raw = _dry_allpos(r, s=33, shared=tmp_path / "shared", env_extra={"F22_SERVING_PATH": "lens", "G8K_STORE_PINS": str(rec)})
        assert rc == 1 and "the row's RECORD declares serving path 'lora_native'" in raw and "F22_SERVING_PATH=lens" in raw, raw[-1500:]
        assert "DIFFERENT intervention" in raw and "SERVING_PATH:" not in text
    # the same export on an existing steer row still resolves EXPLICIT (its record declares no path — nothing moved for it)
    alpha = M.row_for("gemma4e4b", "main", M.CELL_ALPHASTEER, ALL_ROWS)
    text, rc, raw = _dry(alpha, s=20, shared=tmp_path / "shared2")
    assert rc == 0 and _line(text, "SERVING_PATH") == "SERVING_PATH: path=lens pinned_by_row=lens reason=kind:alphasteer selection=explicit"
    # the checkout's records
    text, rc, raw = _dry_allpos(r, s=33, shared=tmp_path / "shared3", env_extra={"F22_SERVING_PATH": "lens"})
    assert rc == 1 and "SERVING_PATH:" not in text
    if _allpos_engine_entries() is None:
        _allpos_stamp_pending_xfail(raw)
    assert "the row's RECORD declares serving path 'lora_native'" in raw and "F22_SERVING_PATH=lens" in raw and "DIFFERENT intervention" in raw


def test_allpos_the_dry_run_declares_the_native_route_and_the_single_object_standup(tmp_path):
    """The sbatch's own dry run over this row, the pins source held by the tests' seam (`G8K_STORE_PINS`, honoured in a local dry run
    alone): over a record that knows no engine the pin reads the empty stub wave root — `unpinned:no-store` — and over the REAL record's
    entries under a native engine block (the shape the stamp writes) it reads `store:native(pins)`; either way the row is unpinned, the
    kind-aware default selects the native slot at `f22_init`, and the standup serves exactly `none` + this one object. THE SERVING-SIDE
    REGISTRATION IS EXTERNAL: until `arditi_allpos` is a cell id this tree's grammar admits, `f22_select_serving_path` refuses at the
    kind-aware selector — the launcher's own declaration (the pin, its reason and the launch cells) is made and logged BEFORE that, so
    everything this slice owns is checked here either way. The overflow cluster refuses the row by PLACEMENT. Then the checkout's own
    record, as a real launch reads it: refusing by leg name until the engine stamp lands, the `store:native(pins)` plan after."""
    r = ALLPOS_ALL_ROWS[0]
    blind, stamped = _lc_pins(tmp_path / "blind.json"), _allpos_stamped_copy(tmp_path / "stamped.json")
    text, rc, raw = _dry_allpos(r, s=33, shared=tmp_path / "shared", env_extra={"G8K_STORE_PINS": str(blind)})
    # the launcher's declaration, made before the selector runs: the pin FOLLOWS THE STORE (the seam's record knows none, the stub root
    # holds none) and the standup serves exactly `none` + this one object
    assert "serving path: not pinned by the row (unpinned:no-store;" in raw
    assert "the kind-aware default selects it at f22_init BY KIND over the declared launch cells: none arditi_allpos (F22_LAUNCH_CELLS)" in raw
    assert f"serving path DECLARED by the row's record: lora_native" in raw
    if not _allpos_grammar_landed():
        assert rc == 1 and "[serving-path] REFUSED: DefenseCellError: malformed cell id 'arditi_allpos'" in raw
        assert "SERVING_PATH:" not in text and "DRIVER:" not in text
        pytest.skip(f"the serving-side registration for {ALLPOS_CELL!r} has not landed on this tree — the refusal above IS the "
                    "expected external state; the plan lines below are asserted the moment it does")
    assert rc == 0, raw[-1500:]
    _assert_allpos_native_plan(text, raw, r, "unpinned:no-store")
    text, rc, raw = _dry_allpos(r, s=33, shared=tmp_path / "shared_stamped", env_extra={"G8K_STORE_PINS": str(stamped)})
    assert rc == 0, raw[-1500:]
    _assert_allpos_native_plan(text, raw, r, "store:native(pins)")
    # the OVERFLOW cluster: this arm's legs are placed on the primary one, so the dry run there refuses by name (through the seam, so
    # the pin cannot mask the placement refusal)
    text, rc, raw = _dry_allpos(r, s=33, shared=tmp_path / "shared", cluster="cluster_b", env_extra={"G8K_STORE_PINS": str(stamped)})
    assert rc != 0 and "PLACEMENT" in raw and "DRIVER:" not in text
    # the CHECKOUT's records, as a real launch reads them
    text, rc, raw = _dry_allpos(r, s=33, shared=tmp_path / "shared_checkout")
    if _allpos_engine_entries() is None:
        assert rc == 1 and "SERVING_PATH:" not in text and "DRIVER:" not in text
        _allpos_stamp_pending_xfail(raw)
    assert rc == 0, raw[-1500:]
    _assert_allpos_native_plan(text, raw, r, "store:native(pins)")


def test_allpos_the_matrix_regenerates_byte_identically_from_its_frozen_inputs():
    """The document is GENERATED, never hand-edited, from its `frozen_inputs:` copies + the admission record (`--check`), and it
    reads no live registry. Skips by name until the serving-side registration lands (generation runs the harness grammar)."""
    spec = V.load_views_yaml(ALLPOS_VIEWS)
    assert spec.frozen_inputs is not None
    roster, pins = V.frozen_input_paths(spec, REPO)
    doc = M.matrix_doc(ALLPOS_MATRIX)
    assert doc["source"]["roster_sha256"] == _sha256(roster) and doc["source"]["pins_sha256"] == _sha256(pins)
    _need_allpos_grammar()
    r = subprocess.run([sys.executable, str(G8K / "make_gsm8k_matrix.py"), "--matrix-name", M.MATRIX_ALLPOS_NAME, "--check"],
                       capture_output=True, text=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, check=False)
    assert r.returncode == 0, r.stdout + r.stderr


def test_allpos_every_matrix_taking_launcher_admits_every_registered_matrix_so_the_rung_can_be_planned(tmp_path, monkeypatch):
    """A REGISTERED matrix is accepted by EVERY launcher that takes one by name — the row launcher, the production dry run, the
    LADDER PLANNER and the no-switch smoke. A half-registration reads as working and is not: this matrix was first added to the
    two row launchers alone, which would have left a document DECLARING `trigger_threshold: 0.01` whose 16,384 rung could never
    be planned — the leg would sit at the cap with an unrunnable escalation and nothing would say so at launch time. So the rung
    path is pinned end to end here too: the planner's line and `plan`'s line both name the matrix, and the row launcher admits
    the name they print. `plan` reads the row's pin off the store-pins records (a lens prefix only when every selected row pins
    lens), so it is planned over the REAL record's entries under a native engine block (the shape the stamp writes: unpinned, no
    prefix) and then over the checkout's record, which refuses by leg name until the engine stamp lands and prints the same line after."""
    for name in MATRIX_TAKING_LAUNCHERS:
        admitted = sorted(_admitted_matrices(name).split("|"))
        assert admitted == sorted(set(M.MATRICES) - ALLPOS_LAGGING_LAUNCHERS.get(name, set())), name
        assert M.MATRIX_ALLPOS_NAME in admitted or M.MATRIX_ALLPOS_NAME in ALLPOS_LAGGING_LAUNCHERS.get(name, set()), name
    assert not ALLPOS_LAGGING_LAUNCHERS, f"a launcher lags the registry: {ALLPOS_LAGGING_LAUNCHERS} — each needs a stated reason"
    # the ladder planner's rung line and the 8,192 submission line both carry the matrix as the trailing positional the row
    # launcher parses, and the row launcher admits it (the escalation the trigger promises is actually submittable)
    assert M.array_tail(SHA0, matrix=ALLPOS_MATRIX) == f"ARRAY {SHA0} 8192 {M.MATRIX_ALLPOS_NAME}"
    assert M.array_tail(SHA0, rung=M.RUNG_CAP, matrix=ALLPOS_MATRIX) == f"ARRAY {SHA0} 16384 {M.MATRIX_ALLPOS_NAME}"
    planned = [{"index": 0, "job_id": ALLPOS_ALL_ROWS[0]["job_id"], "cluster_constraint": "primary", "n_ids": 20,
                "sizing": {"time_limit": "0-02:00:00", "t_pred_h": 0.9, "watcher_max_hours": 4.0}}]
    lines = LP.sbatch_lines(planned, sha=SHA0, cluster="cluster_a", matrix=ALLPOS_MATRIX)
    assert lines[1].endswith(f"gsm8k_thinking.sbatch ARRAY {SHA0} 16384 {M.MATRIX_ALLPOS_NAME}") and "--array=0%1 " in lines[1]
    kw = dict(max_concurrent=1, wave=M.WAVE_ALLPOS, matrix=ALLPOS_MATRIX, assert_eligible_record=True)
    monkeypatch.setattr(M, "STORE_PINS_PATHS", (_allpos_stamped_copy(tmp_path / "stamped.json"),))
    pl = M.plan(SHA0, "cluster_a", **kw)
    assert pl.rows == [0] and pl.held == [] and pl.wave_root_rel == "exp/gsm8k_paired200_paper_objects_arditi_allpos"
    assert pl.lines[0].endswith(f"ARRAY {SHA0} 8192 {M.MATRIX_ALLPOS_NAME}") and f"--qos={SITE_A['qos']} " in pl.lines[0]
    assert not pl.lines[0].startswith("F22_SERVING_PATH=")            # a native store on record: unpinned, so no lens prefix
    assert pl.eligibility == {"wave": M.WAVE_ALLPOS, "rows": [0], "excluded": {}}
    # …and the row is refused on the overflow cluster by name (the record places this arm's legs on the primary one)
    with pytest.raises(SystemExit, match="cluster_constraint primary"):
        M.plan(SHA0, "cluster_b", max_concurrent=1, wave=M.WAVE_ALLPOS, matrix=ALLPOS_MATRIX)
    # a name no launcher knows is still refused by each of them, by name
    for name in MATRIX_TAKING_LAUNCHERS:
        assert "nosuch" not in _admitted_matrices(name), name
    # the checkout's records
    monkeypatch.undo()
    monkeypatch.setenv("SISTEER_CONFIG", str(_SITE_CFG))   # undo() dropped the module's synthetic site too
    assert M.STORE_PINS_PATHS[-1] == ALLPOS_PINS
    if _allpos_engine_entries() is None:
        with pytest.raises(SystemExit, match="refused:record-engine-unknown") as exc:
            M.plan(SHA0, "cluster_a", **kw)
        _allpos_stamp_pending_xfail(str(exc.value))
    assert M.plan(SHA0, "cluster_a", **kw).lines == pl.lines


def test_allpos_the_arm_has_its_own_oos_tuple_never_the_rungs_steer_one():
    """The tuple is an AUTHORISATION, not a label, so this arm gets its OWN `_steer_allpos_<cap>` pair. The rung's
    `_steer_<cap>` tuples declare prefill-only objects dosed as a fraction of the residual norm, picked as the highest rung
    within the over-refusal budget — false of this intervention three times over (BOTH phases; a raw-coefficient dose on a
    `vector_all` object; a DECLARED OVERRIDE sitting OVER a 10 pp budget, not the rule's return) — and their `cells` are
    `M5T_STEER_CELLS`, shared by six GSM8K tuples and three MATH-500 ones whose FINISHED legs record their tuple BY NAME, so
    widening it would move a recorded experiment under stores that are done. The MATH-500 launcher made the same split."""
    r = ALLPOS_ALL_ROWS[0]
    assert r["tuple_name"] == "gsm8k_paired200_qwen35_4b_steer_allpos_8192"
    assert r["tuple_names"]["16384"] == "gsm8k_paired200_qwen35_4b_steer_allpos_16384"
    assert M.ALLPOS_TUPLE_SUFFIX == "_steer_allpos" and M.ALLPOS_MODELS == ("qwen35_4b",)
    # the per-cell rule: the two rung arms keep the rung's name BYTE-IDENTICAL, and only this arm diverges
    for cell in (M.CELL_ALPHASTEER, M.CELL_ARDITI):
        for rung in M.RUNGS:
            assert M.external_tuple_name("qwen35_4b", cell, rung=rung) == M.steer_tuple_name("qwen35_4b", rung=rung)
    assert M.external_tuple_name("qwen35_4b", ALLPOS_CELL, rung=M.CAP) != M.steer_tuple_name("qwen35_4b", rung=M.CAP)
    for row in STEER_ROWS + LLAMA_STEER_ROWS:      # every existing steer row's tuple is untouched
        want = M.steer_tuple_name(row["model_key"], rung=M.CAP) if row["tuple_name"] else None
        assert row["tuple_name"] == want, row["job_id"]
    # no other target: the admission record admits one, and the refusal says so
    for key in ("gemma4e4b", "nemotron3nano4b", "glm46vflash", "llama8b"):
        with pytest.raises(SystemExit, match=r"runs on \['qwen35_4b'\] only"):
            M.steer_allpos_tuple_name(key, rung=M.CAP)
    with pytest.raises(SystemExit, match="not one of the rung's cells"):
        M.external_tuple_name("qwen35_4b", "constv2-mix-full__lora_r4_ce", rung=M.CAP)
    # the analysis' reader resolves the SAME name per cell — the rung arms' unchanged, this arm's its own
    for row in STEER_ROWS:
        assert VAL.expected_tuple_name(row, M.CAP) == C.g8k_steer_tuple_name(row["model_key"], M.CAP)
    for cap in M.RUNGS:
        assert VAL.expected_tuple_name(r, cap) == r["tuple_names"][str(cap)]


def test_allpos_the_registry_names_the_arms_tuples_the_mirror_is_pinned_equal_and_no_existing_tuple_was_widened():
    """THE HANDSHAKE, CLOSED (the MATH-500 lane's shape). The serving-side registration has landed on this tree and the
    arm's tuples with it: (1) the registry NAMES both caps' `_steer_allpos_<cap>` tuples with exactly this arm's two cells,
    and this launcher's stdlib-only MIRROR of the name (`steer_allpos_tuple_name`, `ALLPOS_TUPLE_SUFFIX` — mirrored because
    the node's python3 reads this module before any venv exists) is pinned EQUAL to the registry's own function, per
    admitted target and rung: the mirror is checked, never trusted. (2) is the assertion that matters most and is
    unchanged: NO existing tuple was widened to take the cell — finished legs record their tuple's `cells` by name, so a
    future change folding this arm into the `_steer` tuples would move recorded experiments under stores that are done.
    (3) keeps the MEASUREMENT of why the vocabulary and the tuples had to land in ONE change (a registry tuple naming a
    grammar-refused cell takes the authorisation door down for a FINISHED and a PUBLISHED row of the same model) and skips
    by name now that they have — the skip is the constraint no longer binding, not a gap."""
    r = ALLPOS_ALL_ROWS[0]
    # (1) LANDED: both caps' tuples are registered, each with exactly this arm's two cells, spelled by the registry's own
    # name function — and the mirror this module carries is pinned equal to it over every admitted target and every rung
    assert C.ALLPOS_TUPLE_SUFFIX == M.ALLPOS_TUPLE_SUFFIX == "_steer_allpos"
    assert C.ALLPOS_MODELS == M.ALLPOS_MODELS == ("qwen35_4b",) and C.ALLPOS_CELLS == ("none", ALLPOS_CELL)
    for cap in M.RUNGS:
        name = r["tuple_names"][str(cap)]
        assert name in C.AUTHORISED_OOS_EXPERIMENTS, name
        assert C.AUTHORISED_OOS_EXPERIMENTS[name].cells == ("none", ALLPOS_CELL) == C.ALLPOS_CELLS, name
        assert name == C.g8k_allpos_tuple_name("qwen35_4b", cap) == VAL.expected_tuple_name(r, cap)
        for key in M.ALLPOS_MODELS:
            assert M.steer_allpos_tuple_name(key, rung=cap) == C.g8k_allpos_tuple_name(key, cap) \
                == f"{C.G8K_TUPLE_PREFIX}_{key}{C.ALLPOS_TUPLE_SUFFIX}_{cap}", (key, cap)
    assert r["tuple_name"] == C.g8k_allpos_tuple_name("qwen35_4b", M.CAP)
    # (2) NO existing tuple was widened to take the cell — on either instrument. The tuples naming it are EXACTLY the arm's
    # own three (this instrument's two caps + the MATH-500 one); every other tuple is as it was, and the rung's `_steer`
    # tuple still names exactly its two cells
    own = {r["tuple_names"][str(cap)] for cap in M.RUNGS} | {C.m5t_allpos_tuple_name("qwen35_4b")}
    naming = {name for name, exp in C.AUTHORISED_OOS_EXPERIMENTS.items() if ALLPOS_CELL in exp.cells}
    assert naming == own, naming ^ own
    for name, exp in C.AUTHORISED_OOS_EXPERIMENTS.items():
        if name not in own:
            assert ALLPOS_CELL not in exp.cells, name
    assert C.AUTHORISED_OOS_EXPERIMENTS[C.g8k_steer_tuple_name("qwen35_4b", M.CAP)].cells == C.M5T_STEER_CELLS == ("alphasteer", "arditi")
    # (3) the MEASURED reason the entry could not land before the vocabulary: with it registered and `arditi_allpos` still
    # outside the grammar, the door raises for a FINISHED row and a PUBLISHED row of the same model. The grammar has
    # landed, so this skips by name — the measurement stays as the record of why the two landed together.
    if _allpos_grammar_landed():
        pytest.skip("the vocabulary has landed — the ordering constraint this measures no longer binds")
    from unittest import mock
    door = dict(controls=["gsm8k"], mode=M.THINKING_MODE, sampling_mode=M.SAMPLING_THINKING_CARD, samples=M.SAMPLES,
                max_tokens_by_task={M.INSPECT_TASK: M.CAP})
    for cell in (M.CELL_ARDITI, "constv2-mix-full__lora_r4_ce_minus_ctl"):
        assert C.check_out_of_support_thinking_authorised(model="qwen35_4b", cell=cell, **door)
    pending = dict(C.AUTHORISED_OOS_EXPERIMENTS)
    pending[r["tuple_name"]] = C.AUTHORISED_OOS_EXPERIMENTS[C.g8k_steer_tuple_name("qwen35_4b", M.CAP)]._replace(
        cells=("none", ALLPOS_CELL))
    with mock.patch.object(C, "AUTHORISED_OOS_EXPERIMENTS", pending):
        for cell in (M.CELL_ARDITI, "constv2-mix-full__lora_r4_ce_minus_ctl"):
            with pytest.raises(C.CellNameError, match=ALLPOS_CELL):
                C.check_out_of_support_thinking_authorised(model="qwen35_4b", cell=cell, **door)


def test_check_endpoint_reads_the_neutral_served_positions_map_first_and_the_passthrough_flag_is_engine_aware(tmp_path):
    """The post-standup endpoint check (st_check_endpoint) over the two table shapes the two engines write: a NATIVE standup
    writes `lens_served_positions: {}` and the cell's mode in the engine-neutral `served_positions` map, and its endpoint carries
    `engine: lora_native` and NO lens hand-off flag; a LENS standup writes the lens map and `lens_served_positions_passthrough`.
    Before this fix the shell door read the lens map only and then demanded the lens flag, so every natively served fitted leg
    died after a healthy standup (the arditi_allpos legs, 2026-09-17). Known-bad twins for every acceptance."""
    cell = "arditi_allpos"
    def table(**maps):
        t = tmp_path / f"table_{len(list(tmp_path.iterdir()))}.json"
        t.write_text(json.dumps({"version": 2, "routes": [{"cell_id": cell, "petri_position_mode": "all"}], **maps}))
        return t
    def serve(**ep):
        d = tmp_path / f"serve_{len(list(tmp_path.iterdir()))}" / "gsm8k"; d.mkdir(parents=True)
        (d / "endpoint.json").write_text(json.dumps({"proxy_upstream_timeout_s": 7200.0, "chat_template_hash": "feedfacefeedface", **ep}))
        return d
    native_table = table(lens_served_positions={}, served_positions={cell: ["all"]})
    # (1) native table + native endpoint (no lens flag): accepted through the NEUTRAL map; the flag is logged not applicable
    ok = _check_endpoint(serve(engine="lora_native"), native_table, cell, "all")
    assert ok.returncode == 0 and "ENDPOINT_OK" in ok.stdout, ok.stdout + ok.stderr
    assert "served positions OK (served_positions)" in ok.stdout and "not applicable on the native engine" in (ok.stdout + ok.stderr)
    # (2) the same table under a LENS endpoint without the hand-off flag: the stamp passes, the flag is still demanded
    bad = _check_endpoint(serve(), native_table, cell, "all")
    assert bad.returncode == 1 and "lens_served_positions_passthrough=None" in (bad.stdout + bad.stderr) and "ENDPOINT_OK" not in bad.stdout
    # (3) neither map names the cell: refused as before
    bad = _check_endpoint(serve(engine="lora_native"), table(lens_served_positions={}, served_positions={}), cell, "all")
    assert bad.returncode == 1 and "served-positions stamp absent or wrong" in (bad.stdout + bad.stderr)
    # (4) both maps name the cell and disagree: a torn table, refused by name
    bad = _check_endpoint(serve(engine="lora_native"), table(lens_served_positions={cell: ["response"]}, served_positions={cell: ["all"]}), cell, "all")
    assert bad.returncode == 1 and "torn table" in (bad.stdout + bad.stderr)
    # (5) a neutral stamp that is not the row's expectation: refused naming the neutral source
    bad = _check_endpoint(serve(engine="lora_native"), table(lens_served_positions={}, served_positions={cell: ["response"]}), cell, "all")
    assert bad.returncode == 1 and "served_positions['arditi_allpos'] = ['response'], expected ['all']" in (bad.stdout + bad.stderr)
    # (6) an OLDER lens-only table under a lens endpoint with the flag: accepted through the lens map (the fallback), unchanged behaviour
    ok = _check_endpoint(serve(lens_served_positions_passthrough="passed"), table(lens_served_positions={cell: ["all"]}), cell, "all")
    assert ok.returncode == 0 and "ENDPOINT_OK" in ok.stdout and "served positions OK (lens_served_positions)" in ok.stdout, ok.stdout + ok.stderr


def test_the_allpos_matrix_declares_its_contrasts_expected_refusal_with_the_readers_own_ruling_constant():
    """arditi-allpos-cap: the all-positions row's contrast against the `it` view's lens `none` anchor is O106-inadmissible and no
    native anchor is run. The views spec DECLARES it on the steering entry, the generator stamps
    `expected_contrast_refusal` on the row (appended last) and echoes it in the document's steering block, and the code is the
    launcher's constant pinned equal to the reader's CROSS_ENGINE_RULING — never a new string. Every other matrix carries none."""
    import importlib.util as _iu
    import sys as _sys
    scripts = REPO / "scripts"
    if str(scripts) not in _sys.path:
        _sys.path.insert(0, str(scripts))
    spec = _iu.spec_from_file_location("tinycap_math500_thinking_for_g8k_pin", scripts / "tinycap_math500_thinking.py")
    reader = _iu.module_from_spec(spec)
    _sys.modules["tinycap_math500_thinking_for_g8k_pin"] = reader
    spec.loader.exec_module(reader)
    assert M.EXPECTED_CONTRAST_REFUSAL_CROSS_ENGINE == reader.CROSS_ENGINE_RULING == "O106"
    assert M.EXPECTED_CONTRAST_REFUSAL_CODES == (reader.CROSS_ENGINE_RULING,)
    doc = json.loads((REPO / "configs" / "sdfbase" / "gsm8k" / "matrix_arditi_allpos.json").read_text())
    (row,) = doc["rows"]
    assert row["cell"] == "arditi_allpos" and row["expected_contrast_refusal"] == "O106" and list(row)[-1] == "expected_contrast_refusal"
    assert row["expected_refusal"] is False and row["expected_refusal_reason"] is None          # the LEG-level field is untouched
    assert doc["steering"][0]["expected_contrast_refusal"] == "O106"
    for name in ("matrix.json", "matrix_llama.json", "matrix_itdoc.json"):
        p = REPO / "configs" / "sdfbase" / "gsm8k" / name
        if p.is_file():
            other = json.loads(p.read_text())
            assert not any("expected_contrast_refusal" in r for r in other["rows"]), name
    # the generator refuses a code the reader does not derive
    import yaml as _yaml
    views = _yaml.safe_load((REPO / "configs" / "sdfbase" / "gsm8k" / "views_arditi_allpos.yaml").read_text())
    views["steering"][0]["expected_contrast_refusal"] = "O999"
    with pytest.raises(SystemExit, match=r"expected_contrast_refusal 'O999' is not one of the reader's refusal codes"):
        V._steering_entries(views, {"qwen35_4b": ["it"]}, path=Path("views_arditi_allpos.yaml"))
