"""The `qwen_seedrep` GSM8K paired-200 matrix's LANDING — registered in g8k_cells.MATRICES / VIEWS_BY_MATRIX and admitted by every
launcher's case list; ONE natively served `none` ANCHOR row per view (the reader keys a cell's anchor as `<model_key>/<view>/none` and
refuses the lens-era published card against a native cell by name); the STANDUP COMPANION every standup of the wave registers beside the
row's own cell (the reader compares the two legs' echoed launch flags outside the cell-set plumbing, `max_lora_rank` among them, and the
native composer fits that flag to the registered adapters' largest r — a `none`-only standup would pair with the PSR rows and be refused
against the LoRA rows); the PROVISIONAL pre-staging freeze (the committed matrix = three HELD anchor rows, nothing launchable until the
staging PR re-freezes the roster); the PROMPTED-ROUTES freeze (the two prompted SI baselines on every seed view as `extensions:` rows
since 2026-09-17, appended after the 18 base rows, each with the view's seed base and the companion — 24 launch rows); and byte-identity
for every other matrix. Everything synthetic: placeholder pins (`01` x 32 — never a real object), a scratch tree cut from the committed
views file, the launcher's OWN shell functions run under bash, nothing launched."""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
import cluster_site as _cs  # noqa: E402

# The printers take each cluster's QoS and log directory from the site config: one synthetic two-cluster site for the module.
import tempfile  # noqa: E402
_SITE_CFG = _cs.write_site(Path(tempfile.mkdtemp(prefix="g8k-qwen-site-")))


@pytest.fixture(autouse=True)
def _cluster_site(monkeypatch):
    monkeypatch.setenv("SISTEER_CONFIG", str(_SITE_CFG))

for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "gsm8k",
          REPO / "scripts" / "sdfbase" / "full22", REPO / "scripts" / "sdfbase" / "seedrep"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import eval_seed as ES  # noqa: E402
import g8k_cells as M  # noqa: E402
import g8k_views as V  # noqa: E402
import make_gsm8k_matrix as MK  # noqa: E402

G8K = REPO / "scripts/sdfbase/gsm8k"
VIEWS = REPO / "configs/sdfbase/gsm8k/views_qwen_seedrep.yaml"
MATRIX = REPO / "configs/sdfbase/gsm8k/matrix_qwen_seedrep.json"
NAME = "qwen_seedrep"
COMPANION = "constv2-mix-full__lora_r4_ce"
PAIRED_CELL = "constv2-mix-full__psr_ce_all_minus_ctl"   # the composed paired write (difference_relu since 2026-09-19), on the s<s>p views
PARENTS = ["constv2-mix-full__psr_ce_all", COMPANION, "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce"]
DERIVED = ["constv2-mix-full__lora_r4_ce_minus_ctl", "constv2-mix-full__psr_ce_all_minus_ctl_distilled"]
CELLS6 = PARENTS + DERIVED
#: decisions O20 (2026-09-17): the distilled PSR difference failed admission for every seed — the live views carry these FIVE
CELLS5 = PARENTS + [DERIVED[0]]
#: 2026-09-17: the two PROMPTED baselines every seed view lists after its objects (null staged ids)
#: and the matrix schedules as `extensions:` rows — appended after the 18 base rows, the it-view order (safety, then the constitution SI)
PROMPTED = ["prompted__safety_refusal", "prompted__si_constitution"]
PROMPTED_H = ["prompted__safety", "prompted__constitution"]
STAGING_SHA8 = "ca6fcb92"       # the lane's staging commit (history): the roster + pins the 18-row matrix was frozen on
PROMPTED_FREEZE_SHA8 = "cfdcd2c0"        # the commit of the roster with the prompted baselines on s1-s3 (the pins byte-identical to ca6fcb92's)
FREEZE_SHA8 = "840aa63e"        # the roster + pins with the PAIRED views s1p/s2p/s3p (the composed
                                # difference_relu arm, sha-pinned paired_eb64s<s>__…) — the 24 seed-view rows RAN at tree b54f5b9c under cfdcd2c0
                                # (byte-identical rows; the provenance split is written in views_qwen_seedrep.yaml)
PSR = [c for c in CELLS6 if "psr" in c]
LORA = [c for c in CELLS6 if "lora" in c]
SHA0 = "0" * 40
PY = "/venv/bin/python"
ENV = {**os.environ, "PYTHONPATH": str(REPO / "src"), "SISTEER_CONFIG": str(_SITE_CFG)}   # built at import: the site rides it explicitly


def _sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _views_without_frozen(text: str) -> str:
    out = re.sub(r"\nfrozen_inputs:\n(?:  .*\n)+", "\n", text)
    assert "frozen_inputs:" not in out.split("\nschema:", 1)[1]
    return out


def _views_seed_only(text: str) -> str:
    """The committed views file projected onto the SEED views (s1-s3): the paired views s<s>p (2026-09-20) — their seed bases and
    their holds — removed, so the synthetic scenarios below keep exercising the seed-view mechanics they were written for over a
    synthetic roster that stages the six cells on s1-s3 alone. Comments are dropped (YAML round-trip); the generator reads data."""
    doc = yaml.safe_load(text)
    doc["views"] = {k: [v for v in vs if not ES.is_paired_view(v)] for k, vs in doc["views"].items()}
    doc["inspect_seed_base_by_view"] = {v: b for v, b in doc["inspect_seed_base_by_view"].items() if not ES.is_paired_view(v)}
    doc["holds"] = [h for h in doc.get("holds") or [] if not ES.is_paired_view(h["view"])]
    return "# the committed views file projected onto the seed views (comments dropped)\n" + yaml.safe_dump(doc, sort_keys=False)


def _synthetic_tree(tmp_path: Path, *, seeds=(1, 2, 3), staged=True, drop_cells=(), views_text: str | None = None) -> Path:
    """A scratch tree: the lane's three seed views with the six cells STAGED (`staged=True`, the post-staging roster) or still under
    `pending_fit` (the committed pre-staging roster's shape), placeholder pins, the real campaign config + manifest of each view
    (the own-overlay pair is cut from them), the committed views file minus its `frozen_inputs:` (the tree's live roster is read)."""
    root = tmp_path / "tree"
    (root / "configs/sdfbase/gsm8k").mkdir(parents=True)
    (root / "configs/capeval/manifests").mkdir(parents=True)
    views, pins = {}, {}
    for s in seeds:
        cells = {}
        for c in CELLS6:
            if (s, c) in drop_cells:
                continue
            sid = ("derived_base64s" if c in DERIVED else "base64s") + f"{s}__{c}"
            cells[c] = sid
            pins[sid] = {"object_sha256": f"{s:02x}" * 32}                   # SYNTHETIC placeholder, never a real object
        views[f"s{s}"] = {"config": f"configs/sdfbase_full22_qwen35_4b_s{s}.yaml",
                          "manifest": f"configs/capeval/manifests/petri_full22_qwen35_4b_s{s}.yaml",
                          **({"cells": cells} if staged else {"cells": None, "pending_fit": cells})}
        shutil.copy(REPO / f"configs/sdfbase_full22_qwen35_4b_s{s}.yaml", root / "configs")
        shutil.copy(REPO / f"configs/capeval/manifests/petri_full22_qwen35_4b_s{s}.yaml", root / "configs/capeval/manifests")
    roster = {"targets": {"qwen35_4b": {"model": "Qwen/Qwen3.5-4B", "model_revision": "0" * 40, "views": views}}}
    (root / "configs/sdfbase_full22_roster.yaml").write_text(yaml.safe_dump(roster, sort_keys=False))
    (root / "configs/sdfbase_full22_pins.json").write_text(json.dumps({"qwen35_4b": pins}, indent=1))
    (root / "configs/sdfbase/gsm8k/views_qwen_seedrep.yaml").write_text(_views_without_frozen(views_text if views_text is not None else _views_seed_only(VIEWS.read_text())))
    return root


def _build(root: Path) -> tuple[Path, dict, list]:
    text, doc = MK.build(configs_root=root, views_yaml=root / "configs/sdfbase/gsm8k/views_qwen_seedrep.yaml", views_spec=None)
    out = root / "configs/sdfbase/gsm8k/matrix_qwen_seedrep.json"
    out.write_text(text)
    return out, doc, M.load_matrix(out)["rows"]


def _shell(row_sh: str, body: str) -> tuple[int, list[str], str]:
    """A launcher function RUN under bash from a row's shell facts (the launcher's own code, never re-typed)."""
    script = (f'set -u\nf22_log(){{ :; }}\nf22_fatal(){{ echo FATAL "$@" >&2; exit 1; }}\nf22_standup_extra_reset(){{ F22_STANDUP_EXTRA=("$@"); }}\n'
              f'REPO_DIR={shlex.quote(str(REPO))}\nsource {shlex.quote(str(G8K / "g8k_common.sh"))}\n' + M.literals_sh() + row_sh
              + 'CPY=/venv/bin/python; PROXY_URL=http://127.0.0.1:1/v1; TABLE=/serve/routing_table.json; LEDGER_URL=http://127.0.0.1:2/v1; '
                'CFG=configs/x.yaml; TT_TWO_SI_DEFENSES_DIR=/bank/view; OUT_ROOT=/out; TARGET_S=44; SERVE_DIR=/serve; ATTEMPT_ID=a1; RUNG_SUBSET=/out/rung_subset.json\n'
              + body)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    return r.returncode, r.stdout.splitlines(), r.stderr


def _standup_argv(row: dict, doc: dict) -> tuple[int, list[str], str]:
    return _shell(M.row_sh(row, wave_root_rel=M.wave_root_rel_of(doc)), 'g8k_standup_extra\nfor a in "${F22_STANDUP_EXTRA[@]}"; do printf "%s\\n" "$a"; done\n')


def _driver_argv(row: dict, doc: dict) -> list[str]:
    rc, out, err = _shell(M.row_sh(row, wave_root_rel=M.wave_root_rel_of(doc)), 'g8k_driver_argv\nfor a in "${DRV[@]}"; do printf "%s\\n" "$a"; done\n')
    assert rc == 0, err
    return out


# --------------------------------------------------------------------------- the landing: registration, launchers, the provisional freeze
def test_the_matrix_is_registered_and_the_committed_document_regenerates_from_its_staging_freeze():
    assert M.MATRICES[NAME] == MATRIX and M.VIEWS_BY_MATRIX[NAME] == VIEWS and MATRIX.is_file()
    assert set(M.MATRICES) == set(M.VIEWS_BY_MATRIX) >= {"default", "llama", "itdoc", NAME}   # registered in BOTH tables (beside any other lane's matrix)
    spec = V.load_views_yaml(VIEWS)
    assert spec.schema == M.MATRIX_SCHEMA_2 and spec.wave == NAME and spec.row_overlays and spec.exclude == []
    # the PAIRED views s<s>p hold the seed's r=4 companion and its LoRA difference
    # (measured on s<s>; served on s<s>p for the max_lora_rank invariant and the misalign leg) — the only holds of this matrix
    assert [(h["key"], h["view"], h["cells"]) for h in spec.holds] == [("qwen35_4b", f"s{s}p", [COMPANION, DERIVED[0]]) for s in (1, 2, 3)]
    assert [(e["key"], e["view"], e["cells"], e["since"]) for e in spec.extensions] == [("qwen35_4b", f"s{s}", PROMPTED, "2026-09-17") for s in (1, 2, 3)]
    assert spec.steering == [] and spec.itdoc == []
    assert spec.standup_companions == {"cell": COMPANION, "reason": spec.standup_companions["reason"]}
    assert spec.inspect_seed_policy == "replicate" and spec.inspect_seed_base_by_view == ES.SEED_BASE_BY_ANY_VIEW == {**ES.SEED_BASE_BY_VIEW, **ES.PAIRED_SEED_BASE_BY_VIEW}
    # THE STAGING FREEZE (2026-09-17; decisions O20): the roster + pins of the lane's staging commit ca6fcb92, copied under
    # frozen/<sha8>/ and sha-pinned — every seed view lists its FIVE admitted cells (the four parents + the LoRA difference; the
    # distilled PSR difference failed admission for every seed and is on no view), nothing under pending_fit. (The PROVISIONAL freeze
    # before it, f7999f34, was the pre-staging roster: cells empty, six pending_fit, the three none rows HELD by the companion rule.)
    frozen = V.frozen_input_paths(spec, REPO)
    assert frozen is not None and Path(spec.frozen_inputs["roster"]).parent.name == FREEZE_SHA8 == Path(spec.frozen_inputs["pins"]).parent.name
    doc = M.matrix_doc(MATRIX)
    assert doc["source"]["roster_sha256"] == _sha256(frozen[0]) == spec.frozen_inputs["roster_sha256"]
    assert doc["source"]["pins_sha256"] == _sha256(frozen[1]) == spec.frozen_inputs["pins_sha256"]
    # no pin MOVED across the three freezes: the prompted freeze (cfdcd2c0) re-pinned nothing (pins byte-identical to the staging
    # freeze's); the paired-view freeze (840aa63e) ADDED exactly the three composed paired-write pins and changed no existing one
    staging = REPO / f"configs/sdfbase/gsm8k/frozen/{STAGING_SHA8}"
    prompted_freeze = REPO / f"configs/sdfbase/gsm8k/frozen/{PROMPTED_FREEZE_SHA8}"
    assert _sha256(staging / "sdfbase_full22_pins.json") == _sha256(prompted_freeze / "sdfbase_full22_pins.json")
    staging_pins = json.loads((staging / "sdfbase_full22_pins.json").read_text())["qwen35_4b"]
    new_pins = json.loads(frozen[1].read_text())["qwen35_4b"]
    assert {k: v for k, v in new_pins.items() if k in staging_pins} == staging_pins
    assert sorted(set(new_pins) - set(staging_pins)) == [f"paired_eb64s{s}__{PAIRED_CELL}" for s in (1, 2, 3)]
    for s in (1, 2, 3):
        rec = new_pins[f"paired_eb64s{s}__{PAIRED_CELL}"]
        assert rec["verified_cluster_a"] is True and rec["verified_cluster_b"] is True and rec["sidecar_object_sha256_recorded"] == "match"
        assert rec["source"].endswith(f"sdfbase/reseed/derived/psr_paired_write/qwen35_4b_base_eb64_s{s}/{PAIRED_CELL}")
    assert _sha256(staging / "sdfbase_full22_roster.yaml") != spec.frozen_inputs["roster_sha256"]
    frozen_roster = yaml.safe_load(frozen[0].read_text())
    frozen_pins = json.loads(frozen[1].read_text())["qwen35_4b"]
    for s in (1, 2, 3):
        view = frozen_roster["targets"]["qwen35_4b"]["views"][f"s{s}"]
        assert list(view["cells"]) == CELLS5 + PROMPTED and "pending_fit" not in view              # staged: the five admitted + the two prompted (null), nothing pending
        assert all(sid == (f"derived_base64s{s}__{c}" if c.endswith("_minus_ctl") else f"base64s{s}__{c}") for c, sid in view["cells"].items() if c not in PROMPTED)
        assert all(view["cells"][c] is None for c in PROMPTED)
        assert all(sid in frozen_pins for c, sid in view["cells"].items() if c not in PROMPTED)
        assert f"derived_base64s{s}__{DERIVED[1]}" not in frozen_pins and DERIVED[1] not in view["cells"]   # the dropped distilled object: no pin, no cell
    # the committed document: 24 LAUNCH rows — per view the native `none` anchor FIRST (indices 0 / 6 / 12), then the five cells
    # (the 18 staging-freeze rows at their indices), then the six PROMPTED extension rows APPENDED (18-23: s1 safety, s1 constitution, ...)
    rows = doc["rows"]
    # ... then (2026-09-20, the paired views) the 12 s<s>p rows at 18-29 — per view the native `none` anchor FIRST, the composed
    # difference_relu arm, then the HELD companion + LoRA difference — and the six PROMPTED extension rows moved to 30-35 (scheduled
    # rows precede extension rows). The 24 seed-view rows are byte-identical to the 24-row matrix on every field but `index`
    # (the six prompted rows moved), which is the provenance split views_qwen_seedrep.yaml writes down.
    PAIRED_ROWS = [("none", "launch"), (PAIRED_CELL, "launch"), (COMPANION, "held"), (DERIVED[0], "held")]
    assert [(r["index"], r["view"], r["cell"], r["status"]) for r in rows] == \
        [(6 * (s - 1) + i, f"s{s}", c, "launch") for s in (1, 2, 3) for i, c in enumerate(["none", *CELLS5])] + \
        [(18 + 4 * (s - 1) + i, f"s{s}p", c, st) for s in (1, 2, 3) for i, (c, st) in enumerate(PAIRED_ROWS)] + \
        [(30 + 2 * (s - 1) + i, f"s{s}", c, "launch") for s in (1, 2, 3) for i, c in enumerate(PROMPTED)]
    assert [r["index"] for r in M.eligible_rows(NAME, rows)] == [r["index"] for r in rows if r["status"] == "launch"] == \
        [*range(20), 22, 23, 26, 27, *range(30, 36)]
    for r in rows:
        s = ES.seed_of_view(r["view"])          # `s<s>` and its paired view `s<s>p` carry the same replicate seed
        cells = frozen_roster["targets"]["qwen35_4b"]["views"][r["view"]]["cells"]
        pin_c = frozen_pins[f"base64s{s}__{COMPANION}"]["object_sha256"]
        assert (r["held_reason"] is None) == (r["status"] == "launch") and not r["declared"] and r["own_overlay"] is True and r["wave"] == NAME
        if r["view"].endswith("p"):
            assert r["cell"] in ("none", PAIRED_CELL, COMPANION, DERIVED[0]) and r["budget_class"] == "none"
            if r["cell"] == PAIRED_CELL:        # the composed paired write: fitted, pinned to the staged paired_eb64s<s>__ id
                assert r["fitted"] is True and r["staged_id"] == f"paired_eb64s{s}__{PAIRED_CELL}" and r["object_sha256"] == frozen_pins[r["staged_id"]]["object_sha256"]
        assert r["inspect_seed_policy"] == "replicate" and r["inspect_seed_base"] == ES.SEED_BASE_BY_ANY_VIEW[r["view"]]
        if r["cell"] == "none":
            assert r["fitted"] is False and r["route_kind"] == "base" and r["object_sha256"] is None
        elif r["cell"] in PROMPTED:      # a prompted route: unfitted, no pin, no tuple, its own budget class, the constitution row in the fitted time class
            assert r["fitted"] is False and r["route_kind"] == "prompted" and r["object_sha256"] is None and r["staged_id"] is None and r["tuple_name"] is None
            assert r["budget_class"] == r["cell"] and r["extension"] is True and r["harness_cell"] == V.harness_cell_id(r["cell"])
            assert r["time_limit"] == (M.TIME_FITTED if r["cell"] == "prompted__si_constitution" else M.TIME_IN_SUPPORT)
        else:
            assert r["fitted"] is True and r["object_sha256"] == frozen_pins[cells[r["cell"]]]["object_sha256"]
        if r["cell"] == COMPANION:      # the companion's own row registers itself once — no second --cell
            assert M.row_companions(r) == [] and r["standup_companion_cells"] == [] and r["standup_companion_pins"] == {}
            assert M.launch_cells_for(r) == ["none", V.harness_cell_id(COMPANION)]
        else:
            assert [c for c, *_ in M.row_companions(r)] == [COMPANION] and r["standup_companion_cells"] == [COMPANION]
            assert r["standup_companion_staged_ids"] == {COMPANION: f"base64s{s}__{COMPANION}"} and r["standup_companion_pins"] == {COMPANION: pin_c}
            assert M.launch_cells_for(r) == ["none"] + ([] if r["cell"] == "none" else [V.harness_cell_id(r["cell"])]) + [V.harness_cell_id(COMPANION)]
        assert M.companion_row_problem(r, doc) is None and list(r)[-5:] == [*M.STANDUP_COMPANION_FIELDS, M.GEOMETRY_CLASS_KEY]   # the class LAST
    c = doc["counts"]
    assert doc["standup_companions"]["cell"] == COMPANION
    assert (c["rows"], c["held"], c["launched"], c["fitted_launched"], c["in_support"], c["contrasts_designed"]) == (36, 6, 30, 18, 12, 24)
    assert c["by_tag"] == {**{f"qwen35_4b_s{s}": {"rows": 8, "held": 0, "launched": 8} for s in (1, 2, 3)},
                           **{f"qwen35_4b_s{s}p": {"rows": 4, "held": 2, "launched": 2} for s in (1, 2, 3)}} and c["pins"] == {"qwen35_4b": 24}
    assert doc["extensions"] == spec.extensions
    # the generator's --check: matrix + overlays byte-identical (24 own pairs)
    for argv in ([sys.executable, str(G8K / "make_gsm8k_matrix.py"), "--matrix-name", NAME, "--check"],
                 [sys.executable, str(G8K / "make_gsm8k_overlays.py"), "--matrix", NAME, "--check"]):
        r = subprocess.run(argv, capture_output=True, text=True, cwd=REPO, env=ENV)
        assert r.returncode == 0, (argv, r.stdout, r.stderr)
    for s in (1, 2, 3):
        pin_c = frozen_pins[f"base64s{s}__{COMPANION}"]["object_sha256"]
        # the anchor row's own pair: [none, companion] — the companion registered on the standup argv with its pin, never measured
        cfg = (REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_qwen35_4b_s{s}.gsm8k_none.yaml").read_text()
        man = (REPO / f"configs/capeval/manifests/petri_full22_qwen35_4b_s{s}.gsm8k_none.yaml").read_text()
        assert re.search(rf"^explicit_cells:\n- none\n- {re.escape(COMPANION)}\n\n", cfg, re.M)
        assert re.search(rf"^  cells:\n    - none\n    - {re.escape(V.harness_cell_id(COMPANION))}\n  [a-z]", man, re.M)
        assert f"mt_serving_up.py --cell none --cell {COMPANION} --expect-object-sha256 {COMPANION}={pin_c}, at the GSM8K entries" in cfg
        # every steered row's own pair: [none, <cell>, companion] — [none, companion] on the companion's own row
        for cell in CELLS5:
            slug = cell.replace("__", "--")
            ov = (REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_qwen35_4b_s{s}.gsm8k_{slug}.yaml").read_text()
            want = ["none", cell] + ([] if cell == COMPANION else [COMPANION])
            assert re.search(r"^explicit_cells:\n" + "".join(rf"- {re.escape(c)}\n" for c in want) + r"\n", ov, re.M), (s, cell)
        assert not (REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_qwen35_4b_s{s}.gsm8k_{DERIVED[1].replace('__', '--')}.yaml").exists()
        # the prompted rows' own pairs: [none, <prompted>, companion] — the companion registered (pinned) on the standup, the prompted
        # route measured by the driver; the manifest carries the harness ids
        for cell, hcell in zip(PROMPTED, PROMPTED_H):
            slug = cell.replace("__", "--")
            ov = (REPO / f"configs/sdfbase/gsm8k/sdfbase_full22_qwen35_4b_s{s}.gsm8k_{slug}.yaml").read_text()
            man_p = (REPO / f"configs/capeval/manifests/petri_full22_qwen35_4b_s{s}.gsm8k_{slug}.yaml").read_text()
            assert re.search(rf"^explicit_cells:\n- none\n- {re.escape(cell)}\n- {re.escape(COMPANION)}\n\n", ov, re.M), (s, cell)
            assert re.search(rf"^  cells:\n    - none\n    - {re.escape(hcell)}\n    - {re.escape(V.harness_cell_id(COMPANION))}\n  [a-z]", man_p, re.M), (s, cell)
            assert f"--cell {COMPANION} --expect-object-sha256 {COMPANION}={pin_c}" in ov


def test_every_launcher_admits_the_matrix_by_name_and_the_anchor_rows_launch_with_the_companion(tmp_path):
    for name in ("gsm8k_thinking.sbatch", "dry_run_prod.sbatch", "ladder_plan.sbatch", "smoke_not_available.sbatch"):
        src = (G8K / name).read_text()
        m = re.search(r'case "\$G8K_MATRIX" in ([a-z_|]+)\) ;;', src)
        assert m and sorted(m.group(1).split("|")) == sorted(M.MATRICES) and NAME in m.group(1).split("|"), name
        assert f"matrix must be one of {m.group(1)} (g8k_cells.py MATRICES)" in src, name   # the refusal names exactly the list the case admits
    env = {**os.environ, "G8K_DRY_RUN": "1", "G8K_DRY_S": "44", "G8K_DRY_SHARED": str(tmp_path / "shared"), "G8K_DRY_CPY": PY,
           "G8K_DRY_CLUSTER": "cluster_a", "G8K_REPO_DIR": str(REPO), "USER": "tester"}
    env.pop("F22_SERVING_PATH", None)
    frozen = V.frozen_input_paths(V.load_views_yaml(VIEWS), REPO)
    pins = json.loads(frozen[1].read_text())["qwen35_4b"]
    comp_h = V.harness_cell_id(COMPANION)

    def run(argv, **extra):
        r = subprocess.run(["bash", str(G8K / "gsm8k_thinking.sbatch"), *argv], capture_output=True, text=True, env={**env, **extra})
        text = "\n".join(l for l in r.stdout.splitlines() if not l.startswith("["))
        return r, text

    def line(text, key):
        m = re.search(rf"^{key}: (.*)$", text, re.M)
        assert m, (key, text)
        return m.group(1)

    # positional and ARRAY form of the s1 anchor row: since the STAGING freeze (the companion is a staged cell of every view) the row
    # LAUNCHES — no HELD line (the pre-staging document HELD every anchor by the companion rule): the standup registers `none` + the
    # companion (with its pin) so it composes max_lora_rank 8 like every other standup of the view, and the driver measures `none`
    # alone under the view's replicate seed base
    for argv, extra in ((["qwen35_4b", "s1", "none", "8192", SHA0, "0", "0", NAME], {}), (["ARRAY", SHA0, "8192", NAME], {"SLURM_ARRAY_TASK_ID": "0"})):
        r, text = run(argv, **extra)
        assert r.returncode == 0 and "HELD:" not in text, r.stdout + r.stderr
        pin1 = pins[f"base64s1__{COMPANION}"]["object_sha256"]
        assert line(text, "ROW").startswith("index=0 job_id=g8k-qwen35_4b-s1-none key=qwen35_4b view=s1 tag=qwen35_4b_s1 cell=none rung=8192 fitted=0 declared=0 route=base ")
        assert line(text, "LAUNCH_CELLS") == f"source=F22_LAUNCH_CELLS cells=none {comp_h}"
        assert line(text, "STANDUP_COMPANIONS") == f"cells={COMPANION} pins={COMPANION}={pin1}"
        assert line(text, "STANDUP").endswith(f"--cell none --cell {COMPANION} --expect-object-sha256 {COMPANION}={pin1}")
        assert line(text, "INSPECT_SEED") == f"policy=replicate base={ES.seed_base(1)}"
        drv = shlex.split(line(text, "DRIVER"))
        assert [drv[i + 1] for i, a in enumerate(drv) if a == "--cell"] == ["none"] and "--expect-object-sha256" not in drv     # the anchor: unpinned, alone
        assert drv[drv.index("--inspect-seed-policy") + 1] == "replicate" and drv[drv.index("--inspect-seed-base") + 1] == str(ES.seed_base(1))
        assert drv[drv.index("--bank-view") + 1] == "s1"
    # ARRAY index 12 = s3's anchor (the none rows sit at 0 / 6 / 12), under s3's base
    r, text = run(["ARRAY", SHA0, "8192", NAME], SLURM_ARRAY_TASK_ID="12")
    assert r.returncode == 0 and "HELD:" not in text and line(text, "ROW").startswith("index=12 job_id=g8k-qwen35_4b-s3-none key=qwen35_4b view=s3 tag=qwen35_4b_s3 cell=none ")
    assert line(text, "INSPECT_SEED") == f"policy=replicate base={ES.seed_base(3)}" and line(text, "STANDUP_COMPANIONS").endswith(pins[f"base64s3__{COMPANION}"]["object_sha256"])
    # ARRAY index 5 = s1's LoRA difference (the admitted derived object): its own --cell + pin on the standup AND the driver, the
    # companion beside it on the standup only
    r, text = run(["ARRAY", SHA0, "8192", NAME], SLURM_ARRAY_TASK_ID="5")
    cell = DERIVED[0]
    pin_d = pins[f"derived_base64s1__{cell}"]["object_sha256"]
    assert r.returncode == 0 and "HELD:" not in text and line(text, "ROW").startswith(f"index=5 job_id=g8k-qwen35_4b-s1-mix-dlora key=qwen35_4b view=s1 tag=qwen35_4b_s1 cell={cell} rung=8192 fitted=1 declared=0 ")
    assert line(text, "LAUNCH_CELLS") == f"source=F22_LAUNCH_CELLS cells=none {V.harness_cell_id(cell)} {comp_h}"
    assert line(text, "BANK").endswith(f"/cells_view/s1/{cell} pin={pin_d}")
    st = shlex.split(line(text, "STANDUP"))
    assert [st[i + 1] for i, a in enumerate(st) if a == "--cell"] == [cell, COMPANION]
    assert [st[i + 1] for i, a in enumerate(st) if a == "--expect-object-sha256"] == [f"{cell}={pin_d}", f"{COMPANION}={pins[f'base64s1__{COMPANION}']['object_sha256']}"]
    drv = shlex.split(line(text, "DRIVER"))
    assert [drv[i + 1] for i, a in enumerate(drv) if a == "--cell"] == [cell] and drv[drv.index("--expect-object-sha256") + 1] == pin_d
    # a foreign name is refused before anything is sourced, naming every registered matrix
    r = subprocess.run(["bash", str(G8K / "gsm8k_thinking.sbatch"), "qwen35_4b", "s1", "none", "8192", SHA0, "0", "0", "bogus"], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and f"matrix must be one of default|llama|itdoc|{NAME}" in r.stderr
    # ARRAY index 19 = s1's constitution-prompted row: unfitted (no pin on standup or driver), its own --cell + the companion
    # (pinned) on the standup, the replicate flags under s1's base, the LONG-SI time class
    r, text = run(["ARRAY", SHA0, "8192", NAME], SLURM_ARRAY_TASK_ID="31")   # 31 since the paired views took 18-29 (2026-09-20)
    assert r.returncode == 0 and "HELD:" not in text and line(text, "ROW").startswith("index=31 job_id=g8k-qwen35_4b-s1-prompted-const key=qwen35_4b view=s1 tag=qwen35_4b_s1 cell=prompted__si_constitution rung=8192 fitted=0 declared=0 route=prompted ")
    assert line(text, "LAUNCH_CELLS") == f"source=F22_LAUNCH_CELLS cells=none prompted__constitution {comp_h}"
    st = shlex.split(line(text, "STANDUP"))
    assert [st[i + 1] for i, a in enumerate(st) if a == "--cell"] == ["prompted__si_constitution", COMPANION] and st.count("--expect-object-sha256") == 1
    drv = shlex.split(line(text, "DRIVER"))
    assert [drv[i + 1] for i, a in enumerate(drv) if a == "--cell"] == ["prompted__si_constitution"] and "--expect-object-sha256" not in drv
    assert line(text, "INSPECT_SEED") == f"policy=replicate base={ES.seed_base(1)}"
    # the planner PLANS the wave: every one of the 24 rows launched, none held; the arrays split by TIME CLASS — the fitted class
    # (0-01:30:00) holds the 15 objects + the 3 constitution-prompted rows (the long-SI prefill takes the fitted class), the in-support
    # class (0-01:00:00) the 3 anchors + the 3 safety-prompted rows
    r = subprocess.run([sys.executable, str(G8K / "g8k_cells.py"), "plan", "--matrix", NAME, "--wave", NAME, "--sha", SHA0, "--cluster", "cluster_a"],
                       capture_output=True, text=True, cwd=REPO, env=ENV)
    assert r.returncode == 0 and "no launchable row" not in r.stdout + r.stderr, r.stdout + r.stderr
    assert re.search(r"^# plan: .* rows=30/36 launched \(0-19,22-23,26-27,30-35\) held=20-21,24-25,28-29", r.stdout, re.M), r.stdout
    assert re.search(r"^# widths, longest class first: fitted \d+/21 rows .*, in-support \d+/9 rows ", r.stdout, re.M), r.stdout
    arrays = re.findall(r"^sbatch -J (g8k-[a-z-]+) .* --array=([0-9,%-]+) .*gsm8k_thinking\.sbatch ARRAY " + SHA0 + " 8192 " + NAME + "$", r.stdout, re.M)
    assert dict(arrays).keys() == {"g8k-fitted", "g8k-in-support"} and dict(arrays)["g8k-fitted"].startswith("1-5,7-11,13-17,19,23,27,31,33,35%") and dict(arrays)["g8k-in-support"].startswith("0,6,12,18,22,26,30,32,34%"), arrays


# --------------------------------------------------------------------------- the 27-row matrix over a STAGED synthetic roster
def test_after_staging_every_view_carries_its_native_anchor_first_and_every_standup_registers_the_companion(tmp_path):
    root = _synthetic_tree(tmp_path)
    out, doc, rows = _build(root)
    assert doc["standup_companions"] == {"cell": COMPANION, "reason": doc["standup_companions"]["reason"]}
    # 21 base rows (the synthetic six-cell roster) + the 6 prompted extension rows appended
    assert len(rows) == 27 and [(r["view"], r["cell"]) for r in rows] == [(f"s{s}", c) for s in (1, 2, 3) for c in ["none", *CELLS6]] + \
        [(f"s{s}", c) for s in (1, 2, 3) for c in PROMPTED]
    c = doc["counts"]
    assert (c["rows"], c["held"], c["launched"], c["fitted_launched"], c["in_support"], c["contrasts_designed"]) == (27, 0, 27, 18, 9, 24)
    assert [r["index"] for r in M.eligible_rows(NAME, rows)] == list(range(27))
    for r in rows:
        s = int(r["view"][1:])
        pin_c = f"{s:02x}" * 32
        anchor = r["cell"] == "none"
        prompted = r["cell"] in PROMPTED
        # the anchor row: the relu_psr precedent's shape — unfitted, no pin, no tuple, route_kind base, no positions expectation, in support
        if anchor:
            assert r["fitted"] is False and r["object_sha256"] is None and r["staged_id"] is None and r["tuple_name"] is None and r["tuple_names"] is None
            assert r["route_kind"] == "base" and r["served_positions_expected"] is None and r["time_limit"] == M.TIME_IN_SUPPORT and r["budget_class"] == "none"
            assert r["job_id"] == f"g8k-qwen35_4b-s{s}-none" and r["out_root"] == f"controls_qwen_seedrep/qwen35_4b_s{s}/none/thinking_card"
            assert r["standup_cell"] == r["standup_harness_cell"] == r["harness_cell"] == "none"
        elif prompted:               # a prompted route — unfitted, no pin, no tuple; its own budget class; the harness id on the standup
            assert r["fitted"] is False and r["object_sha256"] is None and r["staged_id"] is None and r["tuple_name"] is None
            assert r["route_kind"] == "prompted" and r["served_positions_expected"] is None and r["budget_class"] == r["cell"]
            assert r["standup_cell"] == r["cell"] and r["standup_harness_cell"] == r["harness_cell"] == V.harness_cell_id(r["cell"])
            assert r["out_root"] == f"controls_qwen_seedrep/qwen35_4b_s{s}/{M.cell_slug(r['cell'])}/thinking_card"
        else:
            assert r["fitted"] and r["object_sha256"] == pin_c and r["route_kind"] == "lens" and r["served_positions_expected"] == "all"
        # every row: thinking ON, the Qwen thinking card, the replicate seed rule at ITS view's base, the wave, its own pair
        assert r["thinking_mode"] == "on" and r["sampling_mode"] == "thinking_card" and (r["sampling"]["temperature"], r["sampling"]["top_p"], r["sampling"]["top_k"]) == (0.6, 0.95, 20)
        assert r["inspect_seed_policy"] == "replicate" and r["inspect_seed_base"] == ES.seed_base(s) and r["wave"] == NAME and r["own_overlay"] is True
        assert r["status"] == "launch" and r["held_reason"] is None and not r["declared"]
        # THE COMPANION: on every row but its own — the same cell, its harness id, the view's staged id and the frozen pin — in the row's own view
        comps = M.row_companions(r)
        if r["cell"] == COMPANION:
            assert comps == [] and r["standup_companion_cells"] == [] and r["standup_companion_harness_cells"] == [] and r["standup_companion_pins"] == {}
            assert M.launch_cells_for(r) == ["none", "si_constv2_mix_full__lora_r4_ce"]
        else:
            assert comps == [(COMPANION, "si_constv2_mix_full__lora_r4_ce", f"base64s{s}__{COMPANION}", pin_c)]
            assert r["standup_companion_staged_ids"] == {COMPANION: f"base64s{s}__{COMPANION}"} and r["standup_companion_pins"] == {COMPANION: pin_c}
            want = ["none"] + ([] if anchor else [r["standup_harness_cell"]]) + ["si_constv2_mix_full__lora_r4_ce"]
            assert M.launch_cells_for(r) == want
        assert M.companion_row_problem(r, doc) is None and list(r)[-5:] == [*M.STANDUP_COMPANION_FIELDS, M.GEOMETRY_CLASS_KEY]   # the class LAST
    # a view short of a derived object: 26 rows, contiguous, the anchor, the companion and the prompted rows untouched
    root2 = _synthetic_tree(tmp_path / "t2", drop_cells={(2, DERIVED[1])})
    _, doc2, rows2 = _build(root2)
    assert doc2["counts"]["rows"] == 26 and [r["index"] for r in rows2] == list(range(26)) and sum(r["cell"] == "none" for r in rows2) == 3
    assert sum(r["cell"] in PROMPTED for r in rows2) == 6


def test_the_matrix_loader_refuses_companion_fields_that_disagree_with_the_document(tmp_path):
    root = _synthetic_tree(tmp_path)
    out, doc, rows = _build(root)
    text = out.read_text()

    def refused(mutate, needle):
        bad = json.loads(text)
        mutate(bad)
        p = tmp_path / f"bad_{abs(hash(needle))}.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match=needle):
            M.load_matrix(p)
    def set_companions(row: dict, cell: str) -> None:
        row["standup_companion_cells"], row["standup_companion_harness_cells"] = [cell], ["x"]
        row["standup_companion_staged_ids"], row["standup_companion_pins"] = {cell: "s"}, {cell: "01" * 32}
    refused(lambda d: d["rows"][0].pop("standup_companion_pins"), "not all of")
    refused(lambda d: d["rows"][1].__setitem__("standup_companion_cells", [DERIVED[0]]), "are not one set of cells")     # the lists / dicts disagree
    refused(lambda d: set_companions(d["rows"][1], DERIVED[0]), "a launch row's companions are exactly")                  # a consistent set, not the document's
    refused(lambda d: d["rows"][0].__setitem__("standup_companion_cells", []) or d["rows"][0].__setitem__("standup_companion_harness_cells", [])
            or d["rows"][0].__setitem__("standup_companion_staged_ids", {}) or d["rows"][0].__setitem__("standup_companion_pins", {}), "a launch row's companions are exactly")
    refused(lambda d: (d["rows"][1].__setitem__("status", "held") or d["rows"][1].__setitem__("held_reason", "x") or set_companions(d["rows"][1], DERIVED[0])),
            "are not the document's")                                                                                       # a held row may be empty, never another cell
    refused(lambda d: d["rows"][0]["standup_companion_pins"].__setitem__(COMPANION, "abc"), "not a 64-hex sha256")
    refused(lambda d: d["rows"][2].__setitem__("standup_companion_cells", [COMPANION]) or d["rows"][2].__setitem__("standup_companion_harness_cells", ["x"])
            or d["rows"][2].__setitem__("standup_companion_staged_ids", {COMPANION: "s"}) or d["rows"][2].__setitem__("standup_companion_pins", {COMPANION: "01" * 32}),
            "not a steered cell other than the row's own")
    refused(lambda d: d.pop("standup_companions"), "declares no standup_companions")
    # the views loader: the block is schema-2 + row_overlays, names a steered cell, carries a reason
    base = _views_without_frozen(VIEWS.read_text())

    def views(replacing: str, with_: str) -> Path:
        assert base.count(replacing) == 1, replacing
        p = tmp_path / f"views_{abs(hash((replacing, with_)))}.yaml"
        p.write_text(base.replace(replacing, with_))
        return p
    with pytest.raises(SystemExit, match="never `none`"):
        V.load_views_yaml(views(f"  cell: {COMPANION}\n", "  cell: none\n"))
    with pytest.raises(SystemExit, match="never `none`, never a prompted"):
        V.load_views_yaml(views(f"  cell: {COMPANION}\n", "  cell: prompted__si_constitution\n"))
    with pytest.raises(SystemExit, match="needs `row_overlays: true`"):
        V.load_views_yaml(views("row_overlays: true\n", "row_overlays: false\n"))
    with pytest.raises(SystemExit, match="carries exactly"):
        V.load_views_yaml(views(f"  cell: {COMPANION}\n", f"  cell: {COMPANION}\n  since: today\n"))
    # a companion the view does not stage HOLDS the row by name (the committed pre-staging shape), never a silent none-only standup —
    # the prompted extension rows too (they resolve the companion through the same enumeration)
    held_root = _synthetic_tree(tmp_path / "held", staged=False)
    _, held_doc, held_rows = _build(held_root)
    assert [(r["cell"], r["status"]) for r in held_rows] == [("none", "held")] * 3 + [(c, "held") for _s in (1, 2, 3) for c in PROMPTED]
    assert all(r["held_reason"].startswith(M.COMPANION_HELD_MARK) for r in held_rows) and all(M.row_companions(r) == [] for r in held_rows)


# --------------------------------------------------------------------------- the overlays and the shell
def test_the_own_overlay_pairs_enumerate_the_companion_and_the_anchor_once(tmp_path):
    root = _synthetic_tree(tmp_path)
    out, doc, rows = _build(root)
    r = subprocess.run([sys.executable, str(G8K / "make_gsm8k_overlays.py"), "--configs-root", str(root), "--matrix", str(out)], capture_output=True, text=True, cwd=REPO, env=ENV)
    assert r.returncode == 0, r.stderr
    written = [l.split("wrote ", 1)[1] for l in r.stdout.splitlines() if l.startswith("wrote ")]
    assert len(written) == 54 and all("gsm8k_" in w for w in written)                       # 27 own pairs (21 + the 6 prompted), no TAG overlay
    for s in (1, 2, 3):
        pairs = {"none": (["none", COMPANION], ["none", "si_constv2_mix_full__lora_r4_ce"]),
                 PSR[0]: (["none", PSR[0], COMPANION], ["none", "si_constv2_mix_full__psr_ce_all", "si_constv2_mix_full__lora_r4_ce"]),
                 COMPANION: (["none", COMPANION], ["none", "si_constv2_mix_full__lora_r4_ce"]),
                 DERIVED[0]: (["none", DERIVED[0], COMPANION], ["none", "si_constv2_mix_full__lora_r4_ce_minus_ctl", "si_constv2_mix_full__lora_r4_ce"]),
                 PROMPTED[0]: (["none", PROMPTED[0], COMPANION], ["none", PROMPTED_H[0], "si_constv2_mix_full__lora_r4_ce"]),
                 PROMPTED[1]: (["none", PROMPTED[1], COMPANION], ["none", PROMPTED_H[1], "si_constv2_mix_full__lora_r4_ce"])}
        for cell, (want_cfg, want_man) in pairs.items():
            cfg = (root / f"configs/sdfbase/gsm8k/sdfbase_full22_qwen35_4b_s{s}.gsm8k_{M.cell_slug(cell)}.yaml").read_text()
            man = (root / f"configs/capeval/manifests/petri_full22_qwen35_4b_s{s}.gsm8k_{M.cell_slug(cell)}.yaml").read_text()
            got_cfg = re.search(r"^explicit_cells:\n((?:- .*\n)+)", cfg, re.M).group(1).replace("- ", "").split()
            got_man = re.search(r"^  cells:\n((?:    - .*\n)+)", man, re.M).group(1).replace("- ", "").split()
            assert got_cfg == want_cfg and got_man == want_man, (s, cell, got_cfg, got_man)
            head = cfg.split("\n# sdfbase_full22_qwen35_4b")[0]
            flags = re.search(r"mt_serving_up.py (--cell .*?), at the GSM8K entries", head, re.S).group(1).replace("\n# ", " ")
            pin_c = f"{s:02x}" * 32
            if cell == "none":
                assert flags == f"--cell none --cell {COMPANION} --expect-object-sha256 {COMPANION}={pin_c}" and "STANDUP COMPANIONS" in head
            elif cell == COMPANION:
                assert flags == f"--cell {COMPANION} --expect-object-sha256 {COMPANION}=<the row's object sha256>" and "STANDUP COMPANIONS" not in head
            else:
                assert flags == f"--cell {cell} --expect-object-sha256 {cell}=<the row's object sha256> --cell {COMPANION} --expect-object-sha256 {COMPANION}={pin_c}"
            # the campaign body rides byte for byte below the pair's cells block (the WHOLE block: the committed campaign config's
            # `explicit_cells` lists the view's staged cells since the parents were staged 2026-09-17, not `none` alone)
            src = (root / f"configs/sdfbase_full22_qwen35_4b_s{s}.yaml").read_text()
            src_tail = re.split(r"^explicit_cells:\n(?:- .*\n)+", src, maxsplit=1, flags=re.M)[1]
            assert cfg.endswith(src_tail) and src_tail.strip()


def test_the_launcher_passes_the_companion_to_the_standup_and_the_driver_measures_the_row_alone(tmp_path):
    root = _synthetic_tree(tmp_path)
    out, doc, rows = _build(root)
    anchor, psr, comp, minus = rows[0], rows[1], rows[2], rows[5]
    assert (anchor["cell"], psr["cell"], comp["cell"], minus["cell"]) == ("none", PSR[0], COMPANION, DERIVED[0])
    pin1 = "01" * 32
    for r in (anchor, psr, comp, minus):
        facts = dict(l.split("=", 1) for l in M.row_sh(r, wave_root_rel=doc["wave_root_rel"]).splitlines())
        want_cells = "''" if r is comp else COMPANION
        want_pins = "''" if r is comp else f"{COMPANION}={pin1}"
        assert facts["G8K_STANDUP_COMPANION_CELLS"] == want_cells and facts["G8K_STANDUP_COMPANION_PINS"] == want_pins, r["cell"]
    # the standup argv (g8k_standup_extra RUN): the row's own --cell (+ pin iff fitted), then `--cell <companion> --expect-object-sha256 <companion>=<pin>`
    rc, argv, err = _standup_argv(anchor, doc)
    assert rc == 0 and argv[-6:] == ["--cell", "none", "--cell", COMPANION, "--expect-object-sha256", f"{COMPANION}={pin1}"], (argv, err)
    assert argv.count("--expect-object-sha256") == 1                                      # the anchor carries no pin of its own
    rc, argv, err = _standup_argv(psr, doc)
    assert rc == 0 and argv[-6:] == ["--cell", PSR[0], "--expect-object-sha256", f"{PSR[0]}={pin1}", "--cell", COMPANION, "--expect-object-sha256", f"{COMPANION}={pin1}"][-6:]
    assert argv[argv.index("--cell") + 1] == PSR[0] and argv.count("--cell") == 2
    rc, argv, err = _standup_argv(comp, doc)
    assert rc == 0 and argv.count("--cell") == 1 and argv[-4:] == ["--cell", COMPANION, "--expect-object-sha256", f"{COMPANION}={pin1}"]
    rc, argv, err = _standup_argv(minus, doc)
    assert rc == 0 and argv.count("--cell") == 2 and argv[-2:] == ["--expect-object-sha256", f"{COMPANION}={pin1}"]
    # the driver: the row's cell alone — no pin on the anchor, the row's pin (never the companion's) on a cell; the replicate flags on every row
    for r in (anchor, psr, comp):
        drv = _driver_argv(r, doc)
        assert drv[:3] == [PY, "scripts/served_capability_run.py", "generate"] and drv[drv.index("--bank-view") + 1] == r["view"]
        i = drv.index("--inspect-seed-policy")
        assert drv[i:i + 4] == ["--inspect-seed-policy", "replicate", "--inspect-seed-base", str(ES.seed_base(int(r["view"][1:])))]
        assert drv.count("--expect-object-sha256") == (0 if r is anchor else 1) and "--cell" not in drv[:3]
        if r is not anchor:
            assert drv[drv.index("--expect-object-sha256") + 1] == r["object_sha256"]
    # a companion pin that is not <cell>=<64-hex>, or a companion equal to the row's own cell, refuses by name in the shell
    sh = M.row_sh(psr, wave_root_rel=doc["wave_root_rel"]).replace(f"G8K_STANDUP_COMPANION_PINS={COMPANION}={pin1}", f"G8K_STANDUP_COMPANION_PINS={COMPANION}=abc")
    rc, _, err = _shell(sh, "g8k_standup_extra\n")
    assert rc == 1 and "is not <cell>=<64-hex sha256>" in err
    sh = M.row_sh(psr, wave_root_rel=doc["wave_root_rel"]).replace(f"G8K_STANDUP_COMPANION_CELLS={COMPANION}", f"G8K_STANDUP_COMPANION_CELLS={PSR[0]}")
    rc, _, err = _shell(sh, "g8k_standup_extra\n")
    assert rc == 1 and "the row's own cell or none" in err
    # the dry-run printer names the companions on a row that carries one, and only there
    text_anchor, text_comp = (_shell(M.row_sh(r, wave_root_rel=doc["wave_root_rel"]), "g8k_print_plan\n")[1] for r in (anchor, comp))
    assert any(l == f"STANDUP_COMPANIONS: cells={COMPANION} pins={COMPANION}={pin1}" for l in text_anchor)
    assert not any(l.startswith("STANDUP_COMPANIONS:") for l in text_comp)
    assert any(l.startswith("INSPECT_SEED: policy=replicate base=") for l in text_anchor)


# --------------------------------------------------------------------------- the reader-side facts the design rests on
def test_the_reader_keys_the_anchor_per_view_and_compares_the_fitted_lora_rank_across_legs():
    """The two facts the row shape follows from, pinned against the reader and the composer themselves (never a re-typed table):
    (1) tinycap_gsm8k_paired200.design_contrasts pairs every cell against `<model_key>/<view>/none` — ONE anchor per view, never another
    view's; (2) tinycap_math500_thinking.LAUNCH_EXCLUDED_KEYS leaves `max_lora_rank` / `lora_target_modules` / `enable_lora` COMPARED and
    excludes only the cell-set plumbing, while vllm_server.fitted_max_lora_rank fits the flag to the registered adapters over
    LEGAL_LORA_RANK_CAPS (1, 8, ...) — so a `none`-only standup (r=1 adapters, cap 1) differs from a LoRA r=4 row's (cap 8)."""
    sys.path.insert(0, str(REPO / "scripts"))
    import tinycap_gsm8k_paired200 as G
    import tinycap_math500_thinking as MT
    from sisteer.capeval.serving import vllm_server as VS
    excluded = set(MT.LAUNCH_EXCLUDED_KEYS)
    assert {"lora_modules", "max_loras", "max_cpu_loras"} <= excluded
    assert not ({"max_lora_rank", "lora_target_modules", "enable_lora", "cudagraph_capture_sizes", "enforce_eager"} & excluded)
    assert MT.LAUNCH_RECORD_FILE_BY_ENGINE["lora_native"] == "serving_attestation.json" and "same engine" in MT.CROSS_ENGINE_WHY
    assert VS.LEGAL_LORA_RANK_CAPS[:2] == (1, 8)
    rows = [r for r in M.matrix_rows(MATRIX) if r["cell"] == "none"]          # the committed document's three native anchor rows
    assert [r["view"] for r in rows] == ["s1", "s2", "s3", "s1p", "s2p", "s3p"]
    # the three anchor rows + one s1 PSR cell: the contrast's anchor key is s1's `none` (never s2's or s3's, both present); with every leg
    # a missing store the reader names the anchor leg first — the key it looked up
    matrix = {"launch": [{**r, "declared": False} for r in rows] + [{**rows[0], "cell": PSR[0], "fitted": True}]}
    missing = lambda: {"legs": {"8192": {"missing": "no store (synthetic)"}}, "base_leg": "8192", "rung_leg": None}   # noqa: E731
    cells = {f"qwen35_4b/{v}/none": missing() for v in ("s1", "s2", "s3")}
    cells[f"qwen35_4b/s1/{PSR[0]}"] = missing()
    contrasts = G.design_contrasts(matrix, cells, refusals={}, n_resamples=10, seed=0, ci=0.95)
    assert [(c["cell"], c["view"], c["none_key"], c["cell_key"]) for c in contrasts] == [(PSR[0], "s1", "qwen35_4b/s1/none", f"qwen35_4b/s1/{PSR[0]}")]
    assert contrasts[0]["computable"] is False and contrasts[0]["reason"].startswith("qwen35_4b/s1/none: no store")
    # an s1 cell with NO s1 anchor in the design is refused by name — another view's anchor never stands in
    del cells["qwen35_4b/s1/none"]
    contrasts = G.design_contrasts(matrix, cells, refusals={}, n_resamples=10, seed=0, ci=0.95)
    assert contrasts[0]["reason"] == "qwen35_4b/s1/none: not in the design"


# --------------------------------------------------------------------------- every other matrix: byte-identical behaviour
def test_every_other_matrix_carries_no_companion_field_and_renders_the_same_standup_argv():
    for name in ("default", "llama", "itdoc"):
        doc = M.matrix_doc(M.MATRICES[name])
        assert M.STANDUP_COMPANIONS_KEY not in doc
        rows = doc["rows"]
        assert not any(set(M.STANDUP_COMPANION_FIELDS) & set(r) for r in rows) and all(M.row_companions(r) == [] and M.companion_row_problem(r, doc) is None for r in rows)
        row = next(r for r in rows if r["status"] == "launch" and r["fitted"] and r.get("object_source") is None)
        assert M.launch_cells_for(row) == ["none", row["standup_harness_cell"]]
        sh = M.row_sh(row, wave_root_rel=M.wave_root_rel_of(doc))
        facts = dict(l.split("=", 1) for l in sh.splitlines())
        assert facts["G8K_STANDUP_COMPANION_CELLS"] == "''" and facts["G8K_STANDUP_COMPANION_PINS"] == "''"
        rc, argv, err = _standup_argv(row, doc)
        without = "\n".join(l for l in sh.splitlines() if not l.startswith("G8K_STANDUP_COMPANION_")) + "\n"
        rc2, argv2, _ = _shell(without, 'G8K_STANDUP_COMPANION_CELLS=""; G8K_STANDUP_COMPANION_PINS=""\ng8k_standup_extra\nfor a in "${F22_STANDUP_EXTRA[@]}"; do printf "%s\\n" "$a"; done\n')
        assert rc == rc2 == 0 and argv == argv2 and argv.count("--cell") == 1, (name, err)
        text = _shell(sh, "g8k_print_plan\n")[1]
        assert not any(l.startswith("STANDUP_COMPANIONS:") for l in text)
    # the views loader: a matrix without the block reads None; select_rows without it appends no field
    for name in ("default", "llama", "itdoc"):
        assert V.load_views_yaml(M.VIEWS_BY_MATRIX[name]).standup_companions is None
