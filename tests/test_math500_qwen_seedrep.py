"""The MATH-500 `qwen_seedrep` wave (cells.SEEDREP_ROWS, indices 54-86): the batch-64 three-seed Qwen family on the OVERFLOW
cluster. A SIBLING LIST beside the matrix (the eight-arm reader loads the live M5T_CELLS and refuses a PSR cell), resolved by the
launcher through LAUNCHER_ROWS. Pins: the 30 rows and their order (24 seed + 3 composed + 3 Arditi), every object pin against the
pin file THROUGH the roster's own view -> staged-id mapping, the Arditi rows as the all-positions row re-keyed onto the seed views
(no O106 declaration, a pairing note, the external bank kept), the tuple names / seed rule / companions / bank-view stamps against
the contracts and eval_seed, the six launcher names empty on every matrix row, the overflow-only plan (one array at the cluster's
QoS; the primary cluster refuses by name), the grading host dry run's argv on the canary / first-arm / composed / Arditi rows, the
30 own overlay pairs (byte-identity, the companion enumerated), and the dry run's wave walk + a seed row's fresh table over ITS
view's bank. Pattern: tests/test_gsm8k_arditi_seedrep_slurm.py. Nothing launched."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "math500",
          REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cells as M  # noqa: E402
import eval_seed as ES  # noqa: E402
from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

M5T = REPO / "scripts/sdfbase/math500"
PINS = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())
ROSTER = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())
ADMISSION = json.loads((REPO / "configs/sdfbase/steer_admission.json").read_text())["entries"]["qwen35_4b"]["arditi_allpos"]
KEY = "qwen35_4b"
COMPANION = "constv2-mix-full__lora_r4_ce"
COMPANION_H = "si_constv2_mix_full__lora_r4_ce"
PSR_MIX, PSR_CTL2, PAIRED = "constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all_minus_ctl"
ARDITI = "arditi_allpos"
ARDITI_SHA = "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2"
SEED_VIEWS, PAIRED_VIEWS = ("s1", "s2", "s3"), ("s1p", "s2p", "s3p")
EIGHT = ("none", PSR_MIX, PSR_CTL2, COMPANION, "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl",
         "prompted__safety_refusal", "prompted__si_constitution")
NATIVE, NATIVE_PAIRED, STEER_ALLPOS = ("math500_thinking_on_full22_qwen35_4b_native", "math500_thinking_on_full22_qwen35_4b_native_paired",
                                       "math500_thinking_on_full22_qwen35_4b_steer_allpos")
SHA0 = "0" * 40
SEEDREP = M.SEEDREP_ROWS
SIX = ("M5T_OWN_OVERLAY", "M5T_INSPECT_SEED_POLICY", "M5T_INSPECT_SEED_BASE", "M5T_STANDUP_COMPANION_CELLS", "M5T_STANDUP_COMPANION_PINS",
       "M5T_BANK_VIEW_STAMP")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sh(row):
    return dict(line.split("=", 1) for line in M.row_sh(row).splitlines())


def _roster_staged(view: str, cell: str):
    """The roster's staged id of `cell` on `view` — None on a prompted cell (listed with a null staged id) and on `none` (implicit, unlisted)."""
    return ROSTER["targets"][KEY]["views"][view]["cells"].get(cell)


def _pin(staged: str) -> str:
    e = PINS[KEY][staged]
    return e["object_sha256"] if isinstance(e, dict) else e


def _dry(row, shared: Path, extra_env=None):
    """The sbatch's OWN dry-run printing for one seed row (the positional form with the row's lane view), keyed by line tag."""
    env = {**os.environ, "M5T_DRY_RUN": "1", "M5T_DRY_S": "29", "M5T_DRY_SHARED": str(shared), "M5T_DRY_CPY": "/venv/bin/python", "USER": "tester"}
    for k in ("F22_SERVING_PATH", "M5T_STORE_PINS", "M5T_LOG_REQUESTS_OVERRIDE", "M5T_DRY_WAVE"):
        env.pop(k, None)
    env.update(extra_env or {})
    r = subprocess.run(["bash", str(M5T / "math500_thinking.sbatch"), row["model_key"], row["cell"], row["sampling_mode"], SHA0,
                        "1" if row["fitted"] else "0", M.key_view(row)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, (row["job_id"], r.stdout[-800:], r.stderr[-800:])
    lines = {}
    for line in r.stdout.splitlines():
        if line.startswith("["):
            continue
        tag = line.split(":", 1)[0]
        assert tag not in lines, (tag, r.stdout)
        lines[tag] = line
    return lines


# --------------------------------------------------------------------------- the rows


def test_the_seed_wave_is_a_sibling_list_of_thirty_rows_beside_the_untouched_matrix():
    """24 seed rows (s1 / s2 / s3 x the eight arms, in the eight-arm order) + 3 composed rows (s<s>p, NO own `none`) + 3 Arditi rows
    (seed bases 1..3) + the paired views' own `none` = 33 rows at indices 54-86, a sibling list; M5T_CELLS is the 54-row matrix it was,
    LAUNCHER_ROWS the launcher's universe; every row keys on its lane view, launches, runs the primary thinking_card row on the seed
    wave's one time class, under the OVERFLOW constraint, with its own overlay; wave-isolated out roots carry the TAG segment."""
    assert len(M.M5T_CELLS) == 54 and len(SEEDREP) == 33 and M.LAUNCHER_ROWS == M.M5T_CELLS + SEEDREP and len(M.LAUNCHER_ROWS) == 87
    assert [r["index"] for r in SEEDREP] == list(range(54, 87)) and M.WAVES[-1] == M.WAVE_SEEDREP == "qwen_seedrep"
    assert M.SEEDREP_VIEWS == SEED_VIEWS and M.SEEDREP_PAIRED_VIEWS == PAIRED_VIEWS and M.SEEDREP_MODELS == (KEY,) and M.SEEDREP_CELLS == EIGHT
    assert [(r["view"], r["cell"]) for r in SEEDREP] == ([(v, c) for v in SEED_VIEWS for c in EIGHT] + [(v, PAIRED) for v in PAIRED_VIEWS]
                                                          + [(v, ARDITI) for v in SEED_VIEWS] + [(v, "none") for v in PAIRED_VIEWS])
    # each paired view's OWN `none` anchor, APPENDED LAST (84-86) so nothing above moves (the earlier list had none)
    assert [(r["index"], r["view"]) for r in SEEDREP if r["cell"] == "none" and r["view"] in PAIRED_VIEWS] == [(84, "s1p"), (85, "s2p"), (86, "s3p")]
    for r in SEEDREP:
        s = int(r["view"][1])
        assert r["wave"] == M.WAVE_SEEDREP and r["status"] == M.STATUS_LAUNCH and r["model_key"] == KEY and r["sampling_mode"] == "thinking_card"
        assert r["thinking_mode"] == "on" and r["extension"] is True and r["log_requests"] is True and r.get("held_reason") is None
        assert M.key_view(r) == r["view"] and r["own_overlay"] is True and r["time_limit"] == M.TIME_SEEDREP == "0-04:00:00"
        assert r["cluster_constraint"] == M.CLUSTER_CONSTRAINT_OVERFLOW == "overflow" and r["cluster_preference"] == M.PRIMARY_CLUSTER
        assert r["out_root"] == f"controls_qwen_seedrep/{KEY}_{r['view']}/{M.cell_slug(r['cell'])}/thinking_card"
        assert r["job_id"] == f"m5t-{KEY}-{r['view']}-{M.cell_short(r['cell'])}" and r["harness_cell"] == M.HARNESS_BY_CELL[r["cell"]]
        assert r["inspect_seed_policy"] == "replicate" and r["inspect_seed_base"] == r["seed_base"] == s == ES.seed_of_view(r["view"])
        assert M.leg_keys(r) == [f"{KEY}/{r['cell']}/thinking_card@{r['view']}"] and M.store_roots_rel(r) == [r["out_root"]]
        assert r["sampling"] == M.SAMPLING_ROWS[(KEY, "thinking_card")] and r["sampling"]["temperature"] == 0.6 and r["sampling"]["top_p"] == 0.95
        assert (r["geometry_eval"], r["geometry_L"]) == (("capability:gen_long_math_longsi", 24064) if r["cell"] == "prompted__si_constitution"
                                                          else ("capability:gen_long_math", 17664))
        assert r["route_kind"] == M.route_kind(r["cell"]) and (r["route_kind"] == "lens") is r["fitted"]
        assert M.row_for(KEY, r["cell"], "thinking_card", view=r["view"]) is r
    assert len({r["out_root"] for r in M.LAUNCHER_ROWS}) == len({r["job_id"] for r in M.LAUNCHER_ROWS}) == 87
    assert M.OUT_ROOT_PREFIX_BY_WAVE[M.WAVE_SEEDREP] == "controls_qwen_seedrep" and M.CLASS_TAG[M.TIME_SEEDREP] == "seedrep"
    assert M.SEEDREP_ONLY_CELLS == (PSR_MIX, PSR_CTL2, PAIRED) and M.cell_short(PSR_MIX) == "mixpsr" and M.cell_short(PSR_CTL2) == "ctl2psr" and M.cell_short(PAIRED) == "dpsr"
    for c in M.SEEDREP_ONLY_CELLS:                                   # the harness ids are the adapter's (the mirror is checked, never trusted)
        assert M.HARNESS_BY_CELL[c] == ra.harness_cell_id(c) == C.parse_launcher_cell(c).harness_id and M.route_kind(c) == "lens"
    counts = M.counts()
    c = counts["seedrep"]
    assert c["rows"] == 33 and c["fitted"] == 21 and c["statuses"] == {"launch": 33} and c["waves"] == {"qwen_seedrep": 33}
    assert c["tuple_names"] == sorted([NATIVE, NATIVE_PAIRED, STEER_ALLPOS]) and c["distinct_out_roots"] == 33 and c["fitted_in_support"] == 0
    assert c["budget_classes"] == {"none": 27, "prompted__safety_refusal": 3, "prompted__si_constitution": 3}
    assert c["external_pins"] == {f"{KEY}/{ARDITI}": ARDITI_SHA} and len(c["pins"]) == 18       # the 18 seed objects, each carried by some row
    # the matrix's own counts are untouched by the sibling list
    assert counts["rows"] == 54 and "qwen_seedrep" not in counts["waves"] and M.counts(M.M5T_CELLS)["rows"] == 54


def test_every_seed_rows_object_is_the_views_roster_object_pinned_in_the_pin_file():
    """The pin of every fitted seed row == configs/sdfbase_full22_pins.json under the STAGED id the ROSTER maps the cell to on that view
    (base64s<s>__ parents, derived_base64s<s>__ LoRA difference, paired_eb64s<s>__ composed) — cells.seedrep_staged_id is a rule, the
    roster is the authority, and the two are held equal here; the composed rows' pins are the `paired_eb64s<s>__` entries; a bare /
    prompted row carries none; the companion's pin on every view is the view's `base64s<s>__constv2-mix-full__lora_r4_ce`."""
    seen = set()
    for r in SEEDREP:
        if r["cell"] == ARDITI:
            continue
        staged = M.seedrep_staged_id(r["view"], r["cell"])
        assert staged == _roster_staged(r["view"], r["cell"]), (r["view"], r["cell"])
        if r["fitted"]:
            assert r["staged_id"] == staged and r["object_sha256"] == _pin(staged) and len(r["object_sha256"]) == 64
            assert r["object_generation"] == "batch64" and r["served_positions_expected"] == "all"
            assert r["expected_stamps"] == ["cell-stable", "always", "thinking_on"] and r["twin_expected_stamps"] == ["cell-stable", "always", "thinking_on"]
            seen.add(r["object_sha256"])
        else:
            assert staged is None and r["staged_id"] is None and r["object_sha256"] is None and r["object_generation"] is None
            assert r["expected_stamps"] == ["cell-stable", "not-consulted", "thinking_on"] and r["twin_expected_stamps"] is None
        if r["view"] in PAIRED_VIEWS and r["cell"] != "none":                                 # The paired views' none rows are bare
            assert r["cell"] == PAIRED and r["staged_id"].startswith("paired_eb64s") and r["tuple_name"] == NATIVE_PAIRED
    assert len(seen) == 18                                            # 3 seeds x (4 parents + the LoRA difference) + 3 composed: distinct objects
    seed_pins = M.load_seedrep_pins()
    for view in SEED_VIEWS + PAIRED_VIEWS:
        assert seed_pins[(view, COMPANION)] == _pin(_roster_staged(view, COMPANION)) == _pin(f"base64s{view[1]}__{COMPANION}")
    with pytest.raises(SystemExit, match="not a cell of the seed wave"):
        M.seedrep_staged_id("s1", "alphasteer")
    with pytest.raises(SystemExit, match="not a seed-replicate view"):
        M.seedrep_staged_id("it", COMPANION)
    # the mirror of eval_seed: every seed / paired view's base is the LITERAL training seed
    assert M.SEEDREP_SEED_BASE_OF_VIEW == ES.SEED_BASE_BY_ANY_VIEW == {"s1": 1, "s2": 2, "s3": 3, "s1p": 1, "s2p": 2, "s3p": 3}
    assert M.INSPECT_SEED_POLICY_REPLICATE == C.INSPECT_SEED_POLICY_REPLICATE == "replicate"
    for view in SEED_VIEWS + PAIRED_VIEWS:
        assert C.check_inspect_seed_policy("replicate", M.seedrep_seed_base(view)) == ("replicate", ES.seed_of_view(view))
    # a pin file lacking a seed object refuses at build, never a placeholder
    with pytest.raises(SystemExit, match="no 64-hex object_sha256"):
        M._seedrep_pin({KEY: {}}, KEY, "s1", PSR_MIX)


def test_the_arditi_seed_rows_are_the_allpos_row_rekeyed_onto_the_seed_views():
    """The admission entry's facts verbatim (row 53's shape: the pin ba2212f0…, the dose, the selection block, the
    external bank view ar_c0p3 on BOTH clusters), re-keyed onto s1 / s2 / s3 at seed bases 1 / 2 / 3 — cluster_constraint overflow
    (never the entry's `primary`), NO O106 declaration (the anchor is the native s<s>/none), a pairing note naming the GSM8K C4
    precedent, no companion (a single-object engine), the `_steer_allpos` tuple (view is not a compared field), `--bank-view ar_c0p3`."""
    ard = [r for r in SEEDREP if r["cell"] == ARDITI]
    row53 = M.ALLPOS_ROWS[0]
    assert [r["view"] for r in ard] == list(SEED_VIEWS) and [r["index"] for r in ard] == [81, 82, 83]
    moved = {"index", "job_id", "view", "out_root", "time_limit", "cluster_constraint", "expected_refusal", "wave"}
    for r in ard:
        s = int(r["view"][1])
        assert r["object_source"] == M.OBJECT_SOURCE_EXTERNAL and r["object_sha256"] == ARDITI_SHA == ADMISSION["object_sha256"]
        assert r["bank_view"] == "ar_c0p3" == ADMISSION["view"] and r["staged_id"] == "ar_c0p3__arditi_allpos" and r["bank_cell_dir"] == "arditi_allpos"
        assert r["bank_view_rel"] == ADMISSION["bank_view_rel"] and r["bank_view_rel"]["overflow"] == "exp/objects/arditi_allpos/qwen35_4b/ladder/cells_view/ar_c0p3"
        assert r["cluster_constraint"] == "overflow" and ADMISSION["cluster_constraint"] == "primary"          # Overridden on the seed rows alone
        assert r["expected_refusal"] is None and row53["expected_refusal"] == "O106" and "C4" in r["pairing_note"] and "max_lora_rank 1" in r["pairing_note"]
        assert r["tuple_name"] == STEER_ALLPOS == row53["tuple_name"] and r["fitted"] and r["serving_path"] == "lora_native"
        assert r["inspect_seed_policy"] == "replicate" and r["inspect_seed_base"] == r["seed_base"] == s and r["own_overlay"] is True
        assert M.row_companions(r) == [] and r["standup_companion_cells"] == [] and r["standup_companion_pins"] == {} and r["object_generation"] is None
        assert r["bank_view_stamp"] == "ar_c0p3" == C.expected_bank_view_stamp(view=r["view"], bank_view=r["bank_view"])
        assert M.key_view(r) == r["view"] and M.leg_keys(r) == [f"{KEY}/{ARDITI}/thinking_card@{r['view']}"]
        assert r["job_id"] == f"m5t-{KEY}-{r['view']}-arditi_allpos" and r["out_root"] == f"controls_qwen_seedrep/{KEY}_{r['view']}/arditi_allpos/thinking_card"
        assert M.launch_cells_for(r) == ["none", ARDITI]
        # every OTHER field is row 53's, byte for byte (the same admission entry)
        for k in set(row53) - moved - M.SEEDREP_ARDITI_KEYS:
            assert r[k] == row53[k], (r["view"], k)
        assert set(r) - M.PUBLISHED_KEYS == M.EXTENSION_KEYS | M.ALLPOS_KEYS | M.SEEDREP_ARDITI_KEYS
        # the pin follows the store (no store, no pins entry): unpinned, the kind-aware default selects the native slot; the empty seed
        # record is consulted beside the default one
        assert M.serving_path_pin(r) == (None, "unpinned:store-unread")
        sh = _sh(r)
        assert sh["M5T_OBJECT_SOURCE"] == "external" and sh["M5T_BANK_VIEW"] == "ar_c0p3" and sh["M5T_KEY_VIEW"] == r["view"] and sh["M5T_VIEW"] == r["view"]
        assert sh["M5T_BANK_VIEW_STAMP"] == "ar_c0p3" and sh["M5T_STANDUP_COMPANION_CELLS"] == "''" and sh["M5T_INSPECT_SEED_BASE"] == str(s)
        assert sh["M5T_BANK_VIEW_REL_OVERFLOW"] == r["bank_view_rel"]["overflow"] and sh["PIN"] == ARDITI_SHA
    assert len(M.STORE_PINS_PATHS) == 2 and M.STORE_PINS_PATHS[1].name == "store_pins_qwen_seedrep.json"
    with pytest.raises(SystemExit, match="names no entry for .*arditi_allpos"):
        M._build_seedrep_rows(54, M.load_seedrep_pins(), {})


def test_the_tuples_seed_rule_companions_and_bank_view_stamps_mirror_the_contracts():
    """Against the contracts: the five lane cells -> `_native`, the composed cell -> `_native_paired`, the Arditi rows ->
    `_steer_allpos`, none / prompted -> no tuple; EACH of the 30 rows' identity (the native engine's served reduction mode stated, as the
    driver's doors do) resolves to exactly ONE registry tuple — never OutOfSupportAuthorisationAmbiguous; the companion is the view's
    pinned r=4 constitution LoRA on every campaign row but its own and on no Arditi row; the stamp is contracts.expected_bank_view_stamp's
    and every stamp is in the driver's admitted vocabulary."""
    assert M.SEEDREP_NATIVE_CELLS == C.M5T_NATIVE_CELLS == C.G8K_NATIVE_CELLS and (PAIRED,) == C.M5T_NATIVE_PAIRED_CELLS
    assert M.SEEDREP_TUPLE_NATIVE.format(key=KEY) == C.m5t_native_tuple_name(KEY) == NATIVE
    assert M.SEEDREP_TUPLE_NATIVE_PAIRED.format(key=KEY) == C.m5t_native_paired_tuple_name(KEY) == NATIVE_PAIRED
    assert M.INSTRUMENT_BANK_VIEWS == C.BANK_VIEWS and M.SEEDREP_COMPANION_CELL == COMPANION and M.HARNESS_BY_CELL[COMPANION] == COMPANION_H
    admitted = C.bank_views_admitted()
    for r in SEEDREP:
        want = (NATIVE if r["cell"] in M.SEEDREP_NATIVE_CELLS else NATIVE_PAIRED if r["cell"] == PAIRED else STEER_ALLPOS if r["cell"] == ARDITI else None)
        assert r["tuple_name"] == want, r["job_id"]
        if r["fitted"]:
            got = C.check_out_of_support_thinking_authorised(model=KEY, cell=r["cell"], controls=["math500"], mode="on", sampling_mode="thinking_card",
                                                              samples=1, reduction_mode=C.REDUCTION_MODE_NOT_APPLICABLE, max_tokens_by_task={"math": M.CAP})
            assert got == want, (r["job_id"], got)
        assert r["bank_view_stamp"] == C.expected_bank_view_stamp(view=r["view"], bank_view=r["bank_view"]) == M.seedrep_bank_view_stamp(r["view"], r["bank_view"])
        assert r["bank_view_stamp"] in admitted and C.bank_view_problem(r["bank_view_stamp"]) is None
        assert r["bank_view_stamp"] == ("ar_c0p3" if r["cell"] == ARDITI else r["view"])
        comps = M.row_companions(r)
        if r["cell"] in (COMPANION, ARDITI):
            assert comps == [] and M.launch_cells_for(r) == ["none"] + ([r["harness_cell"]] if r["cell"] != "none" else [])
        else:
            assert comps == [(COMPANION, COMPANION_H, f"base64s{r['view'][1]}__{COMPANION}", _pin(f"base64s{r['view'][1]}__{COMPANION}"))]
            assert M.launch_cells_for(r) == ["none"] + ([r["harness_cell"]] if r["cell"] != "none" else []) + [COMPANION_H]
            assert r["standup_companion_harness_cells"] == [COMPANION_H] and r["standup_companion_staged_ids"] == {COMPANION: comps[0][2]}
    for bad in (dict(model_key="gemma4e4b"), dict(sampling_mode="general"), dict(cell="alphasteer")):
        with pytest.raises(SystemExit):
            M.seedrep_tuple_name(**{**dict(model_key=KEY, sampling_mode="thinking_card", cell=PSR_MIX), **bad})
    assert M.seedrep_bank_view_stamp("it", "ar_c0p3") == "it" and M.seedrep_bank_view_stamp("s2", None) == "s2"


def test_the_six_launcher_names_are_empty_on_every_matrix_row_and_set_on_the_seed_rows():
    """The argv byte-identity contract: the six M5T_* names row_sh gained are EMPTY on rows 0-53 (so m5t_common's `[ -n … ]` branches
    append nothing there — the sbatch test pins the argv itself) and carry the seed facts on rows 54-86."""
    for r in M.M5T_CELLS:
        sh = _sh(r)
        assert [sh[k] for k in SIX] == ["''"] * 6, r["job_id"]
        # `bank_view` is the all-positions row's own key and `object_generation` the it-doc rows' — neither is a seed-only key
        assert not any(k in r for k in M.SEEDREP_KEYS - {"bank_view", "object_generation"}), r["job_id"]
    for r in SEEDREP:
        sh = _sh(r)
        assert sh["M5T_OWN_OVERLAY"] == "1" and sh["M5T_INSPECT_SEED_POLICY"] == "replicate" and sh["M5T_INSPECT_SEED_BASE"] == str(r["seed_base"])
        assert sh["M5T_BANK_VIEW_STAMP"] == r["bank_view_stamp"] and sh["M5T_WAVE"] == "qwen_seedrep" and sh["M5T_TIME"] == "0-04:00:00"
        comps = M.row_companions(r)
        assert sh["M5T_STANDUP_COMPANION_CELLS"] == (COMPANION if comps else "''")
        assert sh["M5T_STANDUP_COMPANION_PINS"] == (f"{COMPANION}={comps[0][3]}" if comps else "''")
        assert sh["M5T_OUT_REL"] == r["out_root"] and sh["M5T_TUPLE"] == (r["tuple_name"] or "''")


# --------------------------------------------------------------------------- the plan


def test_plan_prints_one_array_on_the_overflow_cluster_and_refuses_the_primary_by_name(tmp_path, monkeypatch):
    """`plan --cluster overflow --wave qwen_seedrep` = ONE array line at the cluster's QoS, 0-04:00:00 over 54-86 at the given width
    (one time class; the per-class form names the class, --single-array the wave); `--rows` narrows within the wave (the canary / first-arm
    protocol submits 54 and 55 alone, then 56-83); `--cluster primary` refuses BY NAME; the batch boundary admits a seed row and refuses a
    published one; the eligibility record names every other wave."""
    import cluster_site as _cs
    monkeypatch.setenv("SISTEER_CONFIG", str(_cs.write_site(tmp_path)))     # each cluster's QoS and log directory: a synthetic site
    over = _cs.expect(tmp_path, "b")
    sha = "ab" * 20
    script = f"{M.CHECKOUT_ROOT}/{sha}/scripts/sdfbase/math500/math500_thinking.sbatch"
    pl = M.plan(sha, "overflow", max_concurrent=28, wave=M.WAVE_SEEDREP, assert_eligible_record=True)
    assert pl.rows == list(range(54, 87)) and pl.widths == {M.TIME_SEEDREP: 28} and pl.sizes == {M.TIME_SEEDREP: 33} and pl.cluster == M.OVERFLOW_CLUSTER
    assert pl.lines == [f"sbatch -J m5t-seedrep --qos={over['qos']} --time=0-04:00:00 --output={over['logs']}/%x_%A_%a.out --array=54-86%28 {script} ARRAY {sha}"]
    assert M.plan(sha, "overflow", max_concurrent=28, wave=M.WAVE_SEEDREP, single_array=True).lines == [pl.lines[0].replace("-J m5t-seedrep ", "-J m5t-qwen_seedrep ")]
    assert pl.eligibility == {"wave": "qwen_seedrep", "rows": list(range(54, 87)),
                              "excluded": {"published/complete": 24, "ext1/other-wave": 9, "steer/other-wave": len(M.STEER_ROWS), "llama/other-wave": len(M.LLAMA_ROWS),
                                           **({"itdoc/held": sum(r["status"] == "held" for r in M.ITDOC_ROWS)} if any(r["status"] == "held" for r in M.ITDOC_ROWS) else {}),
                                           **({"itdoc/other-wave": sum(r["status"] == "launch" for r in M.ITDOC_ROWS)} if any(r["status"] == "launch" for r in M.ITDOC_ROWS) else {}),
                                           "allpos/other-wave": 1}}
    assert "F22_SERVING_PATH=" not in pl.lines[0]                    # no lens prefix: nothing pins the lens (the Arditi rows are unpinned)
    assert M.plan(sha, "overflow", max_concurrent=28, wave=M.WAVE_SEEDREP, rows="56-83").rows == list(range(56, 84))
    assert M.plan(sha, "overflow", max_concurrent=1, wave=M.WAVE_SEEDREP, rows="54").lines[0].split("--array=")[1].startswith("54%1 ")
    assert [r["index"] for r in M.eligible_rows(M.WAVE_SEEDREP)] == list(range(54, 87))
    for spec in ("53", "0-83", "53-55"):
        with pytest.raises(SystemExit, match="outside wave 'qwen_seedrep'"):
            M.plan(sha, "overflow", max_concurrent=4, wave=M.WAVE_SEEDREP, rows=spec)
    with pytest.raises(SystemExit, match="cluster_constraint overflow.*D65 / D72"):
        M.plan(sha, "primary", max_concurrent=28, wave=M.WAVE_SEEDREP)
    with pytest.raises(SystemExit, match="cluster_constraint overflow"):
        M.plan(sha, M.PRIMARY_CLUSTER, max_concurrent=1, wave=M.WAVE_SEEDREP, rows="54")
    # the CLI: one sbatch line at the cluster's QoS, the overflow root; the primary cluster refuses on stderr with no line
    r = subprocess.run([sys.executable, str(M5T / "cells.py"), "plan", "--sha", sha, "--cluster", "overflow", "--wave", "qwen_seedrep",
                        "--max-concurrent", "28", "--assert-eligible"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert [l for l in r.stdout.splitlines() if l.startswith("sbatch ")] == pl.lines and "for " not in r.stdout
    assert f"wave=qwen_seedrep rows=33/{len(M.LAUNCHER_ROWS)} (54-86)" in r.stdout and f"qos={over['qos']}" in r.stdout
    r = subprocess.run([sys.executable, str(M5T / "cells.py"), "plan", "--sha", sha, "--cluster", "primary", "--wave", "qwen_seedrep", "--max-concurrent", "4"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "cluster_constraint overflow" in r.stderr and "sbatch " not in r.stdout
    # the batch boundary (m5t_assert_launchable's predicate): a seed row is admitted, by index and by the positional form with its view
    rec = M.assert_launchable(SEEDREP[0])
    assert rec["wave"] == "qwen_seedrep" and rec["rows"] == [54] and rec["excluded"]["qwen_seedrep/other-wave"] == 32 and rec["excluded"]["published/complete"] == 24
    cli = [sys.executable, str(M5T / "cells.py"), "row"]
    for sel in (["--index", "54"], ["--index", "83"], ["--key", KEY, "--cell", PAIRED, "--sampling", "thinking_card", "--view", "s2p"]):
        out = subprocess.run([*cli, *sel, "--assert-launchable"], capture_output=True, text=True)
        assert out.returncode == 0 and out.stdout.startswith("launchable: row ") and "wave=qwen_seedrep status=launch" in out.stdout, (sel, out.stderr)
    out = subprocess.run([*cli, "--index", "87"], capture_output=True, text=True)
    assert out.returncode != 0 and "outside 0..86" in out.stderr
    out = subprocess.run([*cli, "--index", "0", "--assert-launchable"], capture_output=True, text=True)
    assert out.returncode == 1 and "this wave is COMPLETE" in out.stderr
    # `list` names every launcher row, the seed rows with their views
    out = subprocess.run([sys.executable, str(M5T / "cells.py"), "list"], capture_output=True, text=True)
    assert out.returncode == 0 and len(out.stdout.splitlines()) == 87 and "view=s1p" in out.stdout and " m5t-qwen35_4b-s3-arditi_allpos " in out.stdout
    out = subprocess.run([sys.executable, str(M5T / "cells.py"), "list", "--json"], capture_output=True, text=True)
    assert [r["index"] for r in json.loads(out.stdout)] == list(range(87))


# --------------------------------------------------------------------------- the launcher's argv (the sbatch's own dry run)


def test_the_grading_host_dry_run_prints_the_seed_rule_the_companion_and_the_bank_view_stamp(tmp_path):
    """The canary s1/none, the first arm s1/constitution ReLU, the composed s1p row, the Arditi s1 row: through the sbatch
    itself — `--inspect-seed-policy replicate --inspect-seed-base 1`, `--bank-view s1` (`ar_c0p3` on Arditi), the companion `--cell
    constv2-mix-full__lora_r4_ce --expect-object-sha256 …=<the view's pin>` on the STANDUP argv (never the driver's), the own overlay
    pair (TAG qwen35_4b_s1 / s1p), `--allow-out-of-support-thinking` iff a tuple, `--expect-object-sha256 <pin>` iff fitted,
    `--lens-liveness logprob` on the Arditi row only, the native path selected by kind, the seed wave's out root."""
    pin1 = _pin(f"base64s1__{COMPANION}")
    rows = {"none": M.row_for(KEY, "none", "thinking_card", view="s1"), "psr": M.row_for(KEY, PSR_MIX, "thinking_card", view="s1"),
            "paired": M.row_for(KEY, PAIRED, "thinking_card", view="s1p"), "arditi": M.row_for(KEY, ARDITI, "thinking_card", view="s1"),
            "mix": M.row_for(KEY, COMPANION, "thinking_card", view="s1")}
    for name, row in rows.items():
        L = _dry(row, tmp_path / "shared")
        drv = shlex.split(L["DRIVER"][len("DRIVER:"):])
        st = shlex.split(L["STANDUP"][len("STANDUP:"):])
        view = M.key_view(row)
        assert drv[drv.index("--inspect-seed-policy") + 1] == "replicate" and drv[drv.index("--inspect-seed-base") + 1] == "1"
        assert drv[drv.index("--bank-view") + 1] == ("ar_c0p3" if name == "arditi" else view) and drv.count("--bank-view") == 1
        assert drv[drv.index("--config") + 1] == f"configs/sdfbase/math500/sdfbase_full22_{KEY}_{view}.math500_{row['cell_slug']}.yaml"
        assert drv[drv.index("--cell") + 1] == row["cell"] and drv.count("--cell") == 1
        assert drv[drv.index("--out-root") + 1].endswith(f"/exp/math500_thinking_on_full22/{row['out_root']}")
        assert ("--allow-out-of-support-thinking" in drv) is bool(row["tuple_name"])
        assert ([drv[i + 1] for i, t in enumerate(drv) if t == "--expect-object-sha256"] == ([row["object_sha256"]] if row["fitted"] else []))
        assert (drv.count("--lens-liveness") == 1 and drv[drv.index("--lens-liveness") + 1] == "logprob") is (name == "arditi")
        assert "--math500-retry-max-tokens" in drv and drv[drv.index("--max-tokens-by-task") + 1] == "math=16384" and "--on-cap-by-task" not in drv
        cells_on_standup = [st[i + 1] for i, t in enumerate(st) if t == "--cell"]
        digests = [st[i + 1] for i, t in enumerate(st) if t == "--expect-object-sha256"]
        own_digest = [f"{row['cell']}={row['object_sha256']}"] if row["fitted"] else []
        if name in ("arditi", "mix"):
            assert cells_on_standup == [row["cell"]] and digests == own_digest and "COMPANIONS" not in L
        else:
            assert cells_on_standup == [row["cell"], COMPANION] and digests == own_digest + [f"{COMPANION}={pin1}"]
            assert L["COMPANIONS"] == f"COMPANIONS: cells={COMPANION} pins={COMPANION}={pin1}"
        assert "--enable-log-requests" in st and st[:4] == ["--upstream-timeout-s", "7200", "--lens-salt-policy", "attested"]
        assert L["INSPECT_SEED"] == "INSPECT_SEED: policy=replicate base=1"
        assert L["BANK_VIEW_STAMP"].startswith(f"BANK_VIEW_STAMP: {row['bank_view_stamp']} (--bank-view; view={view} bank_view=")
        assert L["SERVING_PATH"].startswith("SERVING_PATH: path=lora_native pinned_by_row=none reason=" + ("unpinned:no-store" if name == "arditi" else "unpinned:not-external"))
        assert L["LAUNCH_CELLS"] == "LAUNCH_CELLS: source=F22_LAUNCH_CELLS cells=" + " ".join(M.launch_cells_for(row))
        assert f"WAVE: wave=qwen_seedrep status=launch view={view} " in L["WAVE"] and f"time=0-04:00:00" in L["ROW"]
        if row["tuple_name"]:
            assert L["OUT OF SUPPORT"].startswith(f"OUT OF SUPPORT: tuple={row['tuple_name']} pin={row['object_sha256']} ")
        else:
            assert L["IN SUPPORT"].startswith(f"IN SUPPORT: route={row['route_kind']} ")
    # the TAG the launcher resolves for a seed row, off m5t_resolve_config itself: `<key>_<view>` and the own pair
    for index, tag in ((54, "qwen35_4b_s1"), (78, "qwen35_4b_s1p"), (83, "qwen35_4b_s3")):
        out = subprocess.run(["bash", "-c",
                              f'set -u; cd {shlex.quote(str(REPO))}; f22_fatal(){{ echo "FATAL: $*" >&2; exit 9; }}; f22_log(){{ :; }}; '
                              f'source {shlex.quote(str(M5T / "m5t_common.sh"))}; '
                              f'eval "$({shlex.quote(sys.executable)} {shlex.quote(str(M5T / "cells.py"))} row --index {index} --sh)"; '
                              'm5t_resolve_config; echo "TAG=$TAG"; echo "CFG=$CFG"; echo "MANIFEST=$MANIFEST"'],
                             capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
        row = M.LAUNCHER_ROWS[index]
        assert out.returncode == 0 and f"TAG={tag}" in out.stdout, (index, out.stdout, out.stderr)
        assert f"CFG=configs/sdfbase/math500/sdfbase_full22_{tag}.math500_{row['cell_slug']}.yaml" in out.stdout
        assert f"MANIFEST=configs/capeval/manifests/petri_full22_{tag}.math500_{row['cell_slug']}.yaml" in out.stdout
    # the shell refuses a companion pin that is not <cell>=<64-hex>, a companion equal to the row's cell, and a base with no policy
    common = (M5T / "m5t_common.sh").read_text()
    funcs = common[common.index("m5t_standup_extra(){"):common.index("m5t_check_endpoint(){")]
    base = ("set -u\nf22_fatal(){ echo FATAL \"$@\"; exit 1; }\nf22_log(){ :; }\nf22_standup_extra_reset(){ F22_STANDUP_EXTRA=(\"$@\"); }\n" + funcs
            + f"\nM5T_UPSTREAM_TIMEOUT_S=7200; M5T_LENS_SALT_POLICY=attested; CELL={PSR_MIX}; FITTED=1; PIN={'ab' * 32}; M5T_REASONING_PARSER=''; M5T_JOB_ID=t\n")
    r = subprocess.run(["bash", "-c", base + f"M5T_STANDUP_COMPANION_CELLS={COMPANION}; M5T_STANDUP_COMPANION_PINS={COMPANION}=abc\nm5t_standup_extra"], capture_output=True, text=True)
    assert r.returncode == 1 and "is not <cell>=<64-hex sha256>" in r.stdout
    r = subprocess.run(["bash", "-c", base + f"M5T_STANDUP_COMPANION_CELLS={PSR_MIX}; M5T_STANDUP_COMPANION_PINS={PSR_MIX}={'cd' * 32}\nm5t_standup_extra"], capture_output=True, text=True)
    assert r.returncode == 1 and "the row's own cell or none" in r.stdout
    r = subprocess.run(["bash", "-c", base + f"M5T_STANDUP_COMPANION_CELLS={COMPANION}; M5T_STANDUP_COMPANION_PINS=''\nm5t_standup_extra"], capture_output=True, text=True)
    assert r.returncode == 1 and "every companion is pinned" in r.stdout
    drv_funcs = common[common.index("m5t_driver_argv(){"):common.index("m5t_recorder_argv(){")]
    r = subprocess.run(["bash", "-c", "set -u\nf22_fatal(){ echo FATAL \"$@\"; exit 1; }\nm5t_core(){ :; }\nst_driver_argv_head(){ DRV=(x); }\nst_driver_argv_tail(){ :; }\n"
                        + drv_funcs + "\nM5T_CONTROL=math500; M5T_THINKING_MODE=on; SAMPLING_MODE=thinking_card; M5T_SAMPLES=1; M5T_RETRY=0; M5T_MAX_TOKENS_BY_TASK=math=16384; "
                        "M5T_IDENTITY_VERSION=2; OUT_ROOT=/o; M5T_BUDGET_FILENAME=b; TARGET_S=1; SERVE_DIR=/s; FITTED=0; M5T_TUPLE=''; PIN=''; M5T_JOB_ID=t\n"
                        "M5T_INSPECT_SEED_BASE=1\nm5t_driver_argv"], capture_output=True, text=True)
    assert r.returncode == 1 and "but no inspect_seed_policy" in r.stdout


# --------------------------------------------------------------------------- the overlays


def test_the_own_overlay_pairs_regenerate_byte_identically_with_the_companion_enumerated():
    """30 pairs `sdfbase_full22_qwen35_4b_<view>.math500_<cellslug>.yaml` + manifests, GENERATED (--check is the gate): explicit_cells =
    [none, <cell>] + companions ([none, companion] on the `none` row and the companion's own; [none, arditi_allpos] on the Arditi rows;
    [none, composed, companion] on s<s>p), the manifests' cells the same in harness spelling; the view's campaign body byte for byte below
    the block; the standup's own select_cells admits the row's cell AND the companion and refuses a cell the pair never enumerated."""
    gen = _load("_m5t_overlays_seedrep", "scripts/sdfbase/math500/make_math500_overlays.py")
    rc = subprocess.run([sys.executable, str(M5T / "make_math500_overlays.py"), "--check"], capture_output=True, text=True, cwd=REPO)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    mt = _load("_m5t_seedrep_mt", "scripts/sdfbase/mt_serving_up.py")
    campaign_by_view = {v: (yaml.safe_load((REPO / f"configs/sdfbase_full22_{KEY}_{v}.yaml").read_text()),
                            yaml.safe_load((REPO / f"configs/capeval/manifests/petri_full22_{KEY}_{v}.yaml").read_text())["petri"])
                        for v in SEED_VIEWS + PAIRED_VIEWS}
    n = 0
    for r in SEEDREP:
        view = M.key_view(r)
        cfg_path = REPO / f"configs/sdfbase/math500/sdfbase_full22_{KEY}_{view}.math500_{r['cell_slug']}.yaml"
        man_path = REPO / f"configs/capeval/manifests/petri_full22_{KEY}_{view}.math500_{r['cell_slug']}.yaml"
        assert gen.seedrep_overlay_config(r) == cfg_path and gen.seedrep_overlay_manifest(r) == man_path and cfg_path.is_file() and man_path.is_file()
        text, mtext = cfg_path.read_text(), man_path.read_text()
        assert text == gen.render_seedrep_config(r) and mtext == gen.render_seedrep_manifest(r)
        want = ["none"] + ([] if r["cell"] == "none" else [r["cell"]]) + ([] if r["cell"] in (COMPANION, ARDITI) else [COMPANION])
        assert gen.seedrep_cells(r) == want
        cfg = yaml.safe_load(text)
        man = yaml.safe_load(mtext)["petri"]
        assert cfg["explicit_cells"] == want and man["cells"] == [M.HARNESS_BY_CELL[c] for c in want]
        assert cfg["defenses_dir"].endswith(f"cells_view/{view}") and cfg["out_dir"].endswith(f"_{view}") and cfg["model_revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        campaign, cman = campaign_by_view[view]
        assert {k: v for k, v in cfg.items() if k != "explicit_cells"} == {k: v for k, v in campaign.items() if k != "explicit_cells"}
        assert {k: v for k, v in man.items() if k != "cells"} == {k: v for k, v in cman.items() if k != "cells"}
        assert man["chat_template_hash"] == cman["chat_template_hash"] and "--limit-mm-per-prompt" in mtext
        assert f"--inspect-seed-base {r['seed_base']}" in text and f"--bank-view {r['bank_view_stamp']}" in text and "GENERATED by" in text
        # the standup's own selection over the pair: the row's cell + the companion admitted; the prefill-only `arditi` never
        assert mt.select_cells(cfg["explicit_cells"], [r["cell"]] + [c for c, *_ in M.row_companions(r)]) == want
        with pytest.raises(SystemExit, match="never enumerated"):
            mt.select_cells(cfg["explicit_cells"], ["arditi"])
        n += 1
    assert n == 33
    assert len(list((REPO / "configs/sdfbase/math500").glob(f"sdfbase_full22_{KEY}_s*.math500_*.yaml"))) == 33
    assert len(list((REPO / "configs/capeval/manifests").glob(f"petri_full22_{KEY}_s*.math500_*.yaml"))) == 33
    # the matrix's pairs are untouched by the generator's new outputs (byte-identity above covers them; the base overlay lists no PSR cell)
    base_cfg = yaml.safe_load((REPO / f"configs/sdfbase/math500/sdfbase_full22_{KEY}.math500.yaml").read_text())
    assert not set(base_cfg["explicit_cells"]) & set(M.SEEDREP_ONLY_CELLS)


# --------------------------------------------------------------------------- the two-stage dry run


def test_the_dry_run_walks_the_seed_wave_by_wave_or_by_environment_and_exports_a_seed_rows_fresh_table_over_its_view(tmp_path, monkeypatch, capsys):
    """`m5t_dry_run.py --stage pre --wave qwen_seedrep` walks the 30 rows (DRIVER lines byte-identical to the sbatch's per row; PASS),
    M5T_DRY_WAVE in the environment selects the same walk, the default walk is the 54-row matrix, `--only … --only-view s1p` reaches a
    seed row, and the fresh-table export of a seed row (the `none` anchor with its companion) runs over ITS view's bank
    `<bank root>/qwen35_4b/cells_view/s1` through the row's OWN pair — the companion's route beside the anchor's, the companion stamped
    `all` — never the `main` view or the base overlay."""
    DRY = _load("_m5t_dry_run_seedrep", "scripts/sdfbase/math500/m5t_dry_run.py")
    from test_sdfbase_full22_wave import _stage
    py = "/venv/bin/python"
    monkeypatch.delenv("M5T_DRY_WAVE", raising=False)
    rc = DRY.main(["--stage", "pre", "--wave", "qwen_seedrep", "--tmp", str(tmp_path / "w"), "--python", py])
    out = capsys.readouterr().out
    assert rc == 0 and "STAGE: pre rows=33 sha=" in out and "wave=qwen_seedrep" in out and "VERDICT: PASS stage=pre" in out, out[-1500:]
    drivers = [l for l in out.splitlines() if l.startswith("DRIVER:")]
    assert len(drivers) == 33 and all("--inspect-seed-policy replicate" in d and "--bank-view" in d for d in drivers)
    assert sum("--allow-out-of-support-thinking" in d for d in drivers) == 21 and sum("--lens-liveness logprob" in d for d in drivers) == 3
    assert sorted(re.findall(r"--bank-view (\S+)", "\n".join(drivers))) == sorted(r["bank_view_stamp"] for r in SEEDREP)
    # the DRIVER lines correspond to SEEDREP row for row (each names ITS row's out root and seed base), and ONE row re-spawned through the
    # sbatch shows the printed line is the sbatch's own byte for byte — re-spawning all thirty doubled this test's wall time for that one fact
    for r, d in zip(SEEDREP, drivers):
        assert f"/{r['out_root']} " in d + " " and f"--inspect-seed-base {r['seed_base']} " in d + " ", r["job_id"]
    assert DRY.shell_plan(SEEDREP[0], s=DRY.resolved_s_for_row(SEEDREP[0]), shared=tmp_path / "w" / "shared", python=py).splitlines().count(drivers[0]) == 1
    assert '"rows": 33' in out.split("COUNTS: ", 1)[1].splitlines()[0]
    monkeypatch.setenv("M5T_DRY_WAVE", "qwen_seedrep")
    rc = DRY.main(["--stage", "pre", "--tmp", str(tmp_path / "e"), "--python", py, "--only", KEY, PAIRED, "thinking_card", "--only-view", "s2p"])
    out = capsys.readouterr().out
    assert rc == 0 and "STAGE: pre rows=1 sha=" in out and "wave=" not in out.splitlines()[0] and "--inspect-seed-base 2" in out
    with pytest.raises(SystemExit, match="exclusive"):
        DRY.main(["--stage", "pre", "--wave", "qwen_seedrep", "--only", KEY, "none", "thinking_card"])
    with pytest.raises(SystemExit, match="names no row"):
        DRY.main(["--stage", "pre", "--wave", "nope"])
    # the fresh table of the s1 anchor over a synthetic bank of ITS view: the own pair, the companion beside `none`
    row = M.row_for(KEY, "none", "thinking_card", view="s1")
    bank_root = tmp_path / "bank"
    _stage(bank_root / KEY / "cells_view" / "s1", COMPANION, COMPANION)
    for d in ("fresh", "fresh2", "fresh3", "fresh4"):
        (tmp_path / d).mkdir()
    st = DRY.fresh_table_stamp(row, bank_root, tmp_path / "fresh")
    assert st["exported"] and st["own_route"] and st["bank_view"] == str(bank_root / KEY / "cells_view" / "s1")
    # the routes: the anchor, the scale-0 twin the export mints beside any fitted cell (`none__lens0`), and the companion's
    assert st["config"] == f"configs/sdfbase/math500/sdfbase_full22_{KEY}_s1.math500_none.yaml" and sorted(st["routes"]) == ["none", "none__lens0", COMPANION_H]
    assert st["lens_served_positions"] == {COMPANION_H: ["all"]} and st["stamp_for_cell"] is None       # the anchor's own cell has no stamp
    assert DRY.fresh_table_stamp(row, None, tmp_path / "fresh2") == {"exported": False, "why": "no --bank-root: the fresh-table check runs where the bank views exist"}
    # a fitted seed row over the same synthetic bank: its cell's stamp AND the companion's
    psr = M.row_for(KEY, "constv2ctl2-full__lora_r4_ce", "thinking_card", view="s1")
    _stage(bank_root / KEY / "cells_view" / "s1", psr["cell"], psr["cell"])
    st = DRY.fresh_table_stamp(psr, bank_root, tmp_path / "fresh3")
    assert st["exported"] and st["own_route"] and st["stamp_for_cell"] == ["all"] and sorted(st["routes"]) == sorted(["none", "none__lens0", COMPANION_H, psr["harness_cell"]])
    assert st["lens_served_positions"] == {COMPANION_H: ["all"], psr["harness_cell"]: ["all"]}
    # an Arditi seed row exports over the EXTERNAL lane's view for the OVERFLOW role (the cluster the wave runs on), not the campaign bank
    ard = M.row_for(KEY, ARDITI, "thinking_card", view="s1")
    st = DRY.fresh_table_stamp(ard, bank_root, tmp_path / "fresh4")
    assert not st["exported"] and "needs --shared-root" in st["why"]
