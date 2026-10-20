"""The eight-arm extension's FROZEN-ARTIFACT regression.

The two published reports must reproduce through the NEW code paths: MATH-500 ``final/report.json`` (producer tree
55e3cbdd, 24 legs) and GSM8K ``final_1pct/report.json`` (pin 8f5f1495, 38 base + 9 rung legs + the overlap record), each
re-run in the runners' run-F shape (``--no-eval-logs --no-figs --require-report-match``) over the published pulls with the
PUBLISHED-SHAPE matrices pinned as fixtures (B10: the tree's own ``cells.py`` / ``matrix.json`` grow extension rows).
"Byte-identical" is the analysis' own equality (B1): every numeric leaf outside the declared skip list equal, none added,
none missing — the two-venv procedure's criterion. The GSM8K ``final/`` (3 %-era) report is not a target (B15).

The grade directory lives outside the repository; the two live reproductions skip when it is absent (the Modal CPU suite
has none). The third reproduction needs no grade directory: it proves the NEW path — a pinned root
``PATH=PIN``, ``--waves published``, a declared tree set — reproduces the legacy path's numeric leaves over a SYNTHETIC
published-shaped store that carries the two launch-attestation files per leg, which is what the real published pull will
carry once the operator's re-pull of ``serving/`` lands (until then the real pull cannot exercise this path).
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import math500_synthetic_store as S
import pytest
from sisteer import site as SITE

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "eight_arms"
#: the published 24-row MATH-500 matrix: the launcher's fixture (`cells.py list --json --published-keys` regenerates it;
#: `test_sdfbase_math500_sbatch.py` pins it) — the ONE copy in the tree (merge M2)
M5T_FIXTURE = REPO / "tests" / "fixtures" / "math500_matrix_55e3cbdd.json"
GRADE = SITE.expand_path("$GRADE") / "tinycap"
M5T = GRADE / "math500"
G8K = GRADE / "gsm8k"
M5T_PIN, G8K_PIN = "55e3cbd", "8f5f149"

_needs_grade = pytest.mark.skipif(not GRADE.is_dir(), reason="the graded artifacts live outside the repository (absent here)")
#: the synthetic new-path reproduction's declared tree set: the store pin's tree and one further declared tree no leg comes from
T1, T2 = S.FULL_SHA, "a" * 40
PROBE_FILES = ("scripts/served_capability_run.py", "scripts/sdfbase/math500/cells.py")
WHY = "the published legs are reused at their tree; a second tree is declared and carries no leg — the new path's admission shape"


def _analysis():
    spec = importlib.util.spec_from_file_location("tinycap_math500_thinking_frozen", REPO / "scripts" / "tinycap_math500_thinking.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run(script: str, *args: str, out: Path) -> tuple[int, dict]:
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "WANDB_MODE": "disabled", "MPLBACKEND": "Agg"}
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / script), *args, "--no-eval-logs", "--no-figs", "--no-mirror",
                           "--wandb-mode", "disabled", "--out", str(out)], cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
    report = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    return proc.returncode, {"report": report, "stderr": proc.stderr[-2000:], "stdout": proc.stdout[-1000:]}


def _assert_match(rc: int, res: dict, *, published: Path) -> None:
    assert rc == 0, f"rc {rc}; stderr: {res['stderr']}"
    m = res["report"]["report_match"]
    assert m["against"] == str(published)
    assert m["equal"] is True and m["differing"] == [] and m["missing"] == [] and m["added"] == [] and m["type_changed"] == [], m
    assert m["shared_numeric"] > 10_000, m["shared_numeric"]


@_needs_grade
@pytest.mark.skipif(not (M5T / "final" / "report.json").is_file() or not (M5T / f"pull_{M5T_PIN}").is_dir(),
                    reason="the published MATH-500 pull / final report is absent")
def test_math500_final_reproduces_through_the_eight_arm_code_paths(tmp_path):
    """The 24-leg published MATH-500 report (one bare --gen-root, no --waves, the published-shape matrix) is numerically
    equal, leaf for leaf, to a fresh run of this tree — and the report keeps the PUBLISHED shape (no eight-arm blocks)."""
    published = M5T / "final" / "report.json"
    rc, res = _run("tinycap_math500_thinking.py", "--gen-root", str(M5T / f"pull_{M5T_PIN}"), "--sha", M5T_PIN,
                   "--math500-ids", str(M5T / "final" / "math500_ids.json"),
                   "--cells", str(M5T_FIXTURE),
                   "--sampling-rows", str(REPO / "configs" / "sdfbase" / "math500" / "sampling_rows.yaml"),
                   "--require-report-match", str(published), out=tmp_path / "m5t.json")
    _assert_match(rc, res, published=published)
    rep = res["report"]
    assert "design" not in rep and "eightarm" not in rep and "gen_roots" not in rep and "waves" not in rep["provenance"]
    assert len(rep["legs"]) == 24 and rep["verdict"]["word"] is not None
    # the provenance sentences (strings) and the per-leg identity keys ride the published shape without moving a number
    assert rep["provenance"]["trigger"].startswith("MATH-500: one pass at 16,384")
    assert set(rep["provenance"]["sampling"]) == {"gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b"}
    row = next(iter(rep["provenance"]["cells"].values()))
    assert {"object_artifact_sha256", "route_kind", "view", "producer_tree_sha", "thinking", "sampling_ref", "sampling_mode", "pin"} <= set(row)
    assert row["pin"] == M5T_PIN and row["view"] == "main" and row["thinking"] == "on"


@_needs_grade
@pytest.mark.skipif(not (G8K / "final_1pct" / "report.json").is_file() or not (G8K / f"pull_{G8K_PIN}").is_dir(),
                    reason="the published GSM8K pull / 1 % report is absent")
def test_gsm8k_final_1pct_reproduces_through_the_eight_arm_code_paths(tmp_path):
    """The 1 % GSM8K report (38 base + 9 rung legs, the overlap record at the pin) is numerically equal, leaf for leaf, to a
    fresh run of this tree over the published pull with the published 42-row matrix document pinned as a fixture."""
    published = G8K / "final_1pct" / "report.json"
    rc, res = _run("tinycap_gsm8k_paired200.py", "--gen-root", str(G8K / f"pull_{G8K_PIN}"), "--sha", G8K_PIN,
                   "--matrix", str(FIXTURES / "gsm8k_matrix_8f5f1495.json"),
                   "--ids", str(G8K / "final_1pct" / "gsm8k_paired200_ids.json"), "--overlap", str(G8K / "final_1pct" / "overlap.json"),
                   "--require-report-match", str(published), out=tmp_path / "g8k.json")
    _assert_match(rc, res, published=published)
    rep = res["report"]
    assert len(rep["legs"]) == 47 and rep["matrix"]["counts"]["rows"] == 42
    assert rep["truncation"]["trigger_threshold"] == 0.01


def test_the_frozen_matrix_fixtures_are_the_published_documents():
    """The fixtures are the PUBLISHED matrices: 24 MATH-500 rows (the published triple × the four models + sensitivity rows)
    and the 42-row GSM8K document (38 launched + 4 held) — the shapes the two published reports were produced from."""
    m5t = json.loads(M5T_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(m5t, list) and len(m5t) == 24 and [r["index"] for r in m5t] == list(range(24))
    assert {r["cell"] for r in m5t} == {"none", "prompted__safety_refusal", "constv2-mix-full__lora_r4_ce_minus_ctl"}
    assert all(r["out_root"].endswith(f"controls/{r['model_key']}/{r['cell_slug']}/{r['sampling_mode']}") for r in m5t)
    assert not any("status" in r or "wave" in r for r in m5t), "the published matrix carries no design keys"
    g8k = json.loads((FIXTURES / "gsm8k_matrix_8f5f1495.json").read_text(encoding="utf-8"))
    rows = g8k["rows"]
    assert g8k["schema"] == "g8k_matrix/1" and len(rows) == 42 and g8k["counts"]["launched"] == 38 and g8k["counts"]["held"] == 4
    assert sum(1 for r in rows if r["status"] == "launch") == 38 and [r["index"] for r in rows] == list(range(42))


#: the report blocks the NEW path adds (strings, bools, counts of rows) — the only places an added numeric leaf may sit
NEW_PATH_BLOCKS: tuple[str, ...] = ("design", "eightarm", "gen_roots", "tree_set")


def test_the_new_path_reproduces_the_legacy_path_leaf_for_leaf_over_a_published_shaped_store_with_launch_attestations(tmp_path, monkeypatch):
    """Acceptance: the two live reproductions above run the LEGACY path (a bare root, no ``--waves``) and
    assert the eight-arm blocks are absent — legacy-path preservation. This one exercises the NEW path over a SYNTHETIC
    published-shaped store whose every leg carries the two launch-attestation files beside it (the shape the real
    published pull takes after the operator re-pulls ``serving/``): the pinned root ``PATH=PIN``, ``--waves published``
    and a declared tree set. Every numeric leaf the two paths share is equal, none is missing or re-typed, and every
    leaf the new path ADDS sits in an eight-arm block (:data:`NEW_PATH_BLOCKS`) or a per-leg launch / compatibility record
    — the published numbers do not move when the published legs are read through the extension's admission."""
    M = _analysis()
    pub = tmp_path / "pull_pub"
    design = S.design_rows()
    S.build_all(pub, rows=design)                    # the 24 published-shaped legs at T1, the launch record beside each
    pairs = sorted(pub.glob(f"{M.LAUNCH_STANDUP_DIR_PARENT}/*/{M.LAUNCH_STANDUP_TASK_SUBDIR}"))
    assert len(pairs) == 24 and all((d / M.LAUNCH_ATTESTATION_FILE).is_file() and (d / M.LAUNCH_LOG_FILE).is_file() for d in pairs)
    published_keys = set(S.CELLS_MODULE.PUBLISHED_KEYS)
    legacy_rows = [{k: r[k] for k in r if k in published_keys} for r in design]
    assert not any("status" in r or "wave" in r for r in legacy_rows) and len(legacy_rows) == 24    # the published-shaped matrix
    inputs = tmp_path / "in"
    cells_legacy, cells_new = S.write_cells_json(inputs / "legacy.json", legacy_rows), S.write_cells_json(inputs / "design.json", design)
    rows, facts = S.write_sampling_rows(inputs / "rows.json"), S.write_render_facts(inputs / "facts.json")
    ids = inputs / "ids.json"
    ids.write_text(json.dumps(S.IDS), encoding="utf-8")
    common = ["--sampling-rows", str(rows), "--render-facts", str(facts), "--math500-ids", str(ids), "--analysis-sha", T1, "--bootstrap", "400",
              "--wandb-mode", "disabled", "--no-mirror", "--no-eval-logs", "--no-figs"]
    # the LEGACY path: a bare root bound to --sha, the published-shaped matrix, no waves
    out_legacy = tmp_path / "legacy_report.json"
    assert M.main(["--gen-root", str(pub), "--sha", S.SHA7, "--cells", str(cells_legacy), *common, "--out", str(out_legacy)]) == 0
    legacy = json.loads(out_legacy.read_text(encoding="utf-8"))
    assert "design" not in legacy and "eightarm" not in legacy and "gen_roots" not in legacy and len(legacy["legs"]) == 24
    assert M.production_refusals(legacy) == [] and legacy["verdict"]["word"] is not None
    # the NEW path: the root pinned to the store pin, --waves published, a declared tree set (verified on a stubbed git)
    monkeypatch.setattr(M, "_git_output", S.fake_git_for({T1, T2}, {(T1, T2): list(PROBE_FILES)}, diff_args=M.GIT_DIFF_NAMES_ARGS))
    review = inputs / "tree_set_review.md"
    review.write_text("".join(f"- `{g}` — a declared tree no published leg comes from; no shared served-identity field moves\n" for g in PROBE_FILES), encoding="utf-8")
    out_new = tmp_path / "new_report.json"
    assert M.main(["--gen-root", f"{pub}={S.SHA7}", "--sha", S.SHA7, "--also-sha", T2, "--tree-diff-allow-file", str(review), "--tree-set-reason", WHY,
                   "--waves", "published", "--cells", str(cells_new), *common, "--out", str(out_new)]) == 0
    new = json.loads(out_new.read_text(encoding="utf-8"))
    assert new["design"]["waves"] == ["published"] and new["gen_roots"][0]["pin"] == S.SHA7 and len(new["legs"]) == 24
    assert M.production_refusals(new) == [] and new["verdict"]["word"] == legacy["verdict"]["word"]
    assert all(leg["launch"]["status"] == M.LAUNCH_ATTESTED for leg in new["legs"].values())         # the pair beside every leg was read
    # both paths read the pair beside every leg: the settings verdict is True on each (the real pull, without the pair, reads
    # None on the legacy path and refuses on the new one — FB5's designed behaviour until the operator's re-pull)
    for rep in (legacy, new):
        assert all(c["compatibility"]["server_settings_equal"] is True and all(c["compatibility"]["launch_attested"].values())
                   for c in rep["comparisons"] if c["estimands"] is not None)
    # leaf for leaf: the analysis' own equality (B1) over the two reports
    d = M.numeric_leaf_diff(legacy, new)
    assert d["differing"] == [] and d["missing"] == [] and d["type_changed"] == [], {k: d[k][:10] for k in ("differing", "missing", "type_changed")}
    assert d["shared_numeric"] > 10_000, d["shared_numeric"]
    # (paths are tuples): an added leaf sits in an eight-arm block, or is one of the matrix's own EXTENSION keys echoed back under
    # `cells[i]` (the design rows carry `geometry_L`; the published-shaped rows do not) — never a computed number
    extension_keys = set(S.CELLS_MODULE.EXTENSION_KEYS)
    stray = [p for p in d["added"] if not (p and (p[0] in NEW_PATH_BLOCKS or (p[0] == "cells" and len(p) == 3 and p[2] in extension_keys)))]
    assert stray == [], sorted({(p[0], p[-1]) for p in stray})
    assert {p[0] for p in d["added"]} <= set(NEW_PATH_BLOCKS) | {"cells"}

