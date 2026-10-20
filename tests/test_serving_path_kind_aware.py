"""The KIND-AWARE serving-path default (the rule of 2026-09-15): "use native when possible, select the lens
route by itself when a launch's steered cells are all lens-only kinds; make the selection recoverable in the logs."

Pinned here, against known-bad inputs:

- ``registry_adapter.select_serving_path`` — the ONE rule (the standup's own serve / skip / refuse disposition per
  cell, ``partition_cells_for_native``): every steered cell lens-only -> ``lens``; at least one natively servable
  -> ``lora_native`` with the skip list; passthrough-only or an empty list -> ``lora_native`` (the entry-point
  default); a REFUSE-class cell (``positions`` missing, a serve declaration with a spec, ``circuit_breaker``
  ``alpha != r``), a duplicate or a malformed id -> the partition's refusal, propagated; an EXPLICIT path wins over
  every auto verdict and classifies nothing; the log line and the record for each case.
- ``registry_adapter.kind_only_cell_source`` — the KIND by name (what the wave shell can read at ``f22_init``: no
  venv, no bank view, no torch), over every cell class; an unmapped arm refuses.
- With a REGISTRY source (specs read) a position-masked LoRA fit is ``skip`` and an all-masked view selects the
  lens; the SAME cell under the kind-only source reads as its natively servable kind — the init-time limitation,
  stated and pinned (the standup's partition skips it by name afterwards, as before).
- ``scripts/sdfbase/full22/serving_path_select.py`` — the ONE call site the shells reach the rule through: ``--sh``
  assignments a shell can eval, ``--json``, a real manifest, ``--explicit``, refusals exit 2; it imports with torch /
  numpy / transformers blocked (the node's stdlib python + PyYAML is all it has at ``f22_init``).
- No shell predicate on cell names beside the selector (common.sh, petri_drive_full22.sh).

CPU-only; synthetic entries; no model loads."""
from __future__ import annotations

import dataclasses
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402
from sisteer.capeval.serving import serve_declaration as sd  # noqa: E402

SELECTOR = REPO / "scripts/sdfbase/full22/serving_path_select.py"
COMMON = REPO / "scripts/sdfbase/full22/common.sh"
DRIVE = REPO / "scripts/sdfbase/full22/petri_drive_full22.sh"

_ALL = {"positions": "all", "response_header_len": None, "layers": [0, 2], "rank": 4}
_RESPONSE = {"positions": "response", "response_header_len": 3, "layers": [0, 2], "rank": 4}
_NO_POSITIONS = {"layers": [0, 2], "rank": 4}

LORA, LORA2, PSR, PROMPTED = "si_safety__lora_r4", "si_constv2_mix_full__lora_r4_ce", "si_safety__psr_kl_all", "prompted__constitution"
ALPHA, ARDITI, ALPHA_T = "alphasteer", "arditi", "alphasteer@thinking"

OVERRIDE = "; explicit F22_SERVING_PATH would override"


def _kind_only(cells):
    return ra.select_serving_path(cells, ra.kind_only_cell_source(cells), spec_source=ra.SPEC_SOURCE_KIND_ONLY)


def _entry(cell_id, kind, spec, **kw):
    return dc.RegistryEntry(cell_id=cell_id, kind=kind, si_id=kw.pop("si_id", "safety"), object_path=f"/x/{cell_id}.pt",
                            serving_path="lens", resolved_spec=None if spec is None else {"kind": kind, **spec}, **kw)


# --- 1. the rule over the kind-only source (what the shell runs at f22_init) -----------------------------------------

def test_vocabulary():
    assert ra.SERVING_PATH_SELECTION_MODES == ("auto", "explicit") == (ra.SERVING_PATH_SELECTION_AUTO, ra.SERVING_PATH_SELECTION_EXPLICIT)
    assert ra.SERVING_PATH_SELECTION_KEY == "serving_path_selection" and ra.SERVING_PATH_LOG_PREFIX == "[serving-path]"
    assert ra.SERVING_PATH_SELECTION_RULE == "kind-aware-2026-09-15"
    assert (ra.SPEC_SOURCE_REGISTRY, ra.SPEC_SOURCE_KIND_ONLY) == ("registry", "kind_only")
    assert ra.ServingPathSelection._fields == ("mode", "selected", "reason", "cells_by_disposition", "skipped", "spec_source", "cells_source")


def test_every_steered_cell_lens_only_selects_the_lens_with_a_kind_tally():
    """An all-AlphaSteer list (x2 incl. a declared variant) plus the base cell: the lens, by itself; the tally is by
    object KIND in first-seen order; every steered cell recorded as skipped with the disposition's own reason. Since
    native-psr-serving Arditi (kind static_r) and the other residual-slot kinds (psr, vector, vector_all,
    diffmean) are natively servable, so the MATH-500 steer wave's shape (AlphaSteer x2 + Arditi) selects NATIVE with
    AlphaSteer skipped by name — pinned below."""
    sel = _kind_only(["none", ALPHA, ALPHA_T])
    assert (sel.mode, sel.selected, sel.spec_source) == ("auto", "lens", "kind_only")
    assert sel.reason == "every steered cell is lens-only (alphasteer×2)"
    assert sel.log_line == "[serving-path] AUTO-selected lens: every steered cell is lens-only (alphasteer×2)" + OVERRIDE
    assert sel.cells_by_disposition == {"passthrough": ["none"], "serve": [], "skip": [ALPHA, ALPHA_T]}
    assert [(s.cell_id, s.kind) for s in sel.skipped] == [(ALPHA, "alphasteer"), (ALPHA_T, "alphasteer")]
    for s in sel.skipped:
        assert s.reason == dc.native_serving_disposition(s.kind, None)[1] and "not natively exportable" in s.reason
    # AlphaSteer x2 + Arditi: NATIVE (Arditi serves on the residual slot), AlphaSteer skipped by name
    sel = _kind_only(["none", ALPHA, ARDITI, ALPHA_T])
    assert (sel.mode, sel.selected) == ("auto", "lora_native")
    assert sel.reason == f"1 of 3 steered cells natively servable; skipped by name: {ALPHA} (alphasteer), {ALPHA_T} (alphasteer)"
    assert sel.cells_by_disposition == {"passthrough": ["none"], "serve": [ARDITI], "skip": [ALPHA, ALPHA_T]}
    # PSR / vector-only views select NATIVE too (residual-slot kinds) — nothing skipped, said so
    sel = _kind_only([PSR, "si_safety__vector_all", PROMPTED])
    assert sel.selected == "lora_native" and sel.reason == "2 of 2 steered cells natively servable; skipped by name: none"
    assert sel.cells_by_disposition == {"passthrough": [PROMPTED], "serve": [PSR, "si_safety__vector_all"], "skip": []}
    # a Class B kind OUTSIDE NATIVE_LORA_SERVABLE_KINDS (antidote) is what selects the lens by itself
    sel = _kind_only(["antidote", PROMPTED])
    assert sel.selected == "lens" and sel.reason == "every steered cell is lens-only (antidote×1)"
    assert sel.cells_by_disposition["passthrough"] == [PROMPTED]
    # a stacked cell is lens-only by construction
    sel = _kind_only(["none", "stacked__safety__lora_r4"])
    assert sel.selected == "lens" and sel.skipped[0].kind == cm.PETRI_STACKED_KIND


def test_a_mixed_view_selects_native_and_names_the_skipped_cells():
    sel = _kind_only(["none", PSR, LORA, ALPHA, PROMPTED, LORA2])
    assert (sel.mode, sel.selected) == ("auto", "lora_native")
    # the psr cell SERVES (a residual-slot kind, natively servable); AlphaSteer is the one skipped by name
    assert sel.reason == f"3 of 4 steered cells natively servable; skipped by name: {ALPHA} (alphasteer)"
    assert sel.log_line == f"[serving-path] AUTO-selected lora_native: 3 of 4 steered cells natively servable; skipped by name: {ALPHA} (alphasteer)" + OVERRIDE
    assert sel.cells_by_disposition == {"passthrough": ["none", PROMPTED], "serve": [PSR, LORA, LORA2], "skip": [ALPHA]}
    assert [s.cell_id for s in sel.skipped] == [ALPHA]
    # every steered cell servable: nothing skipped, said so
    sel = _kind_only([LORA, LORA2])
    assert sel.selected == "lora_native" and sel.reason == "2 of 2 steered cells natively servable; skipped by name: none"
    assert sel.skipped == () and sel.cells_by_disposition["passthrough"] == []


def test_passthrough_only_and_an_empty_list_select_the_entry_point_default_with_the_reason():
    sel = _kind_only(["none", PROMPTED, "prompted__safety"])
    assert (sel.mode, sel.selected) == ("auto", ra.DEFAULT_STANDUP_SERVING_PATH) == ("auto", "lora_native")
    assert sel.reason == "passthrough-only launch (none / prompted cells only, no steered cell): the entry-point default lora_native"
    assert sel.cells_by_disposition == {"passthrough": ["none", PROMPTED, "prompted__safety"], "serve": [], "skip": []}
    sel = ra.select_serving_path([], ra.kind_only_cell_source([]), cells_source="none (KEY/VIEW unset)", spec_source=ra.SPEC_SOURCE_KIND_ONLY)
    assert sel.selected == "lora_native" and sel.reason == "no cell list (source: none (KEY/VIEW unset)): the entry-point default lora_native"
    assert sel.log_line.startswith("[serving-path] AUTO-selected lora_native: no cell list (source: none (KEY/VIEW unset))")
    assert ra.select_serving_path([], ra.kind_only_cell_source([])).reason == "no cell list: the entry-point default lora_native"


def test_a_refuse_class_cell_refuses_as_today_and_so_do_malformed_and_duplicate_ids():
    """Known-bad: with a registry source, a LoRA record the engine cannot judge (positions MISSING, a serve declaration
    carried in the spec, a circuit_breaker with alpha != r) is the partition's REFUSE, propagated — never a lens
    fallback, never a guess; a duplicate id and a malformed id refuse whatever the source."""
    entries = {"none": dc.RegistryEntry(cell_id="none", kind="none"),
               LORA: _entry(LORA, "lora_all", _NO_POSITIONS), ALPHA: dc.RegistryEntry(cell_id=ALPHA, kind="alphasteer")}
    with pytest.raises(ra.RegistryAdapterError, match="no 'positions'") as exc:
        ra.select_serving_path(["none", LORA, ALPHA], dc.MappingCellSource(entries))
    assert LORA in str(exc.value) and "--serving-path lens" in str(exc.value) and "refusing" in str(exc.value)
    declared = {LORA: _entry(LORA, "lora_all", sd.apply_variant(dict(_ALL), sd.VARIANT_THINKING))}
    with pytest.raises(ra.RegistryAdapterError, match="serve declaration"):
        ra.select_serving_path([LORA], dc.MappingCellSource(declared))
    cb = {"circuit_breaker": dc.RegistryEntry(cell_id="circuit_breaker", kind="circuit_breaker", object_path="/x/cb.pt", serving_path="lens",
                                              resolved_spec={"kind": "circuit_breaker", "rank": 4, "lora_alpha": 8, "positions": "all"})}
    # circuit_breaker is a RETIRED arm — no harness id parses to it, so the partition never reaches the source; the
    # disposition itself is what would refuse (pinned so the table's refuse row is not lost)
    assert dc.native_serving_disposition("circuit_breaker", cb["circuit_breaker"].resolved_spec)[0] == "refuse"
    with pytest.raises(ra.RegistryAdapterError, match="named twice"):
        _kind_only([ALPHA, ALPHA])
    with pytest.raises(dc.DefenseCellError, match="malformed cell id"):
        _kind_only(["bogus__thing"])
    with pytest.raises(ra.RegistryAdapterError, match="spec_source"):
        ra.select_serving_path([], ra.kind_only_cell_source([]), spec_source="guess")


def test_a_registry_source_sees_a_masked_fit_as_lens_only_but_the_kind_only_source_cannot():
    """With the objects' specs (a REGISTRY source — the standup's partition) a position-masked LoRA
    fit is ``skip``, so a view whose only steered cell is such a fit selects the LENS; the kind-only source the shell has
    at f22_init reads the same cell as its natively servable KIND and selects native — the standup then skips the cell
    by name (``cells_skipped``) and, with nothing left to serve, refuses traffic loudly. The limitation is real and
    stated in the record (``spec_source``), never papered over."""
    masked = {"none": dc.RegistryEntry(cell_id="none", kind="none"), LORA: _entry(LORA, "lora", _RESPONSE)}
    sel = ra.select_serving_path(["none", LORA], dc.MappingCellSource(masked))
    assert (sel.selected, sel.spec_source) == ("lens", "registry")
    assert sel.reason == "every steered cell is lens-only (lora×1)" and "position-masked" in sel.skipped[0].reason
    sel = _kind_only(["none", LORA])
    assert (sel.selected, sel.spec_source) == ("lora_native", "kind_only") and sel.cells_by_disposition["serve"] == [LORA]
    # mixed with the specs: the masked fit skipped by name, the all-positions fit served -> native (as the standup does)
    mixed = {**masked, LORA2: _entry(LORA2, "lora_all", _ALL, si_id="constv2_mix_full")}
    sel = ra.select_serving_path(["none", LORA, LORA2], dc.MappingCellSource(mixed))
    assert sel.selected == "lora_native" and sel.cells_by_disposition == {"passthrough": ["none"], "serve": [LORA2], "skip": [LORA]}


def test_an_explicit_path_wins_over_every_auto_verdict_and_classifies_nothing():
    # (AlphaSteer-only is the lens-only shape; Arditi / psr are natively servable kinds since native-psr-serving)
    for cells, auto in ((["none", ALPHA], "lens"), (["none", ALPHA, ARDITI], "lora_native"), (["none", LORA, PSR], "lora_native"),
                        (["none"], "lora_native"), ([], "lora_native")):
        src = ra.kind_only_cell_source(cells)
        assert ra.select_serving_path(cells, src, spec_source=ra.SPEC_SOURCE_KIND_ONLY).selected == auto
        for explicit in ("lens", "lora_native"):
            sel = ra.select_serving_path(cells, src, explicit=explicit, cells_source="F22_LAUNCH_CELLS")
            assert (sel.mode, sel.selected, sel.reason) == ("explicit", explicit, "F22_SERVING_PATH set by caller")
            assert sel.cells_by_disposition is None and sel.skipped == () and sel.spec_source is None
            assert sel.log_line == f"[serving-path] EXPLICIT {explicit} (F22_SERVING_PATH set by caller)"
            assert sel.record() == {"mode": "explicit", "selected": explicit, "reason": "F22_SERVING_PATH set by caller",
                                    "cells_by_disposition": None, "skipped": [], "spec_source": None,
                                    "cells_source": "F22_LAUNCH_CELLS", "rule": "kind-aware-2026-09-15"}
    # explicit even wins over a REFUSING list: nothing is classified (the standup refuses it later, as today)
    bad = {LORA: _entry(LORA, "lora_all", _NO_POSITIONS)}
    assert ra.select_serving_path([LORA], dc.MappingCellSource(bad), explicit="lora_native").selected == "lora_native"
    with pytest.raises(ra.RegistryAdapterError, match="not one of"):
        ra.select_serving_path([], ra.kind_only_cell_source([]), explicit="auto")


def test_the_record_shape_and_json_compactness():
    sel = _kind_only(["none", ALPHA, LORA, PSR])
    rec = sel.record()
    assert set(rec) == {"mode", "selected", "reason", "cells_by_disposition", "skipped", "spec_source", "cells_source", "rule"}
    assert rec["skipped"] == [dict(s._asdict()) for s in sel.skipped] and rec["skipped"][0].keys() == {"cell_id", "kind", "reason"}
    line = json.dumps(rec, separators=(",", ":"), sort_keys=True)
    assert "\n" not in line and json.loads(line) == rec              # ONE line: it rides a standup_jobs.jsonl row


# --- 2. the kind-only source ------------------------------------------------------------------------------------------

def test_kind_only_source_names_every_cell_class_by_kind_and_refuses_an_unmapped_arm(monkeypatch):
    src = ra.kind_only_cell_source(["none", PROMPTED, LORA, LORA2, PSR, ALPHA, ARDITI, "antidote", "caa_static_r", "stacked__safety__lora_r4", ALPHA_T])
    kinds = {c: src.get(c).kind for c in src.entries}
    assert kinds == {"none": "none", PROMPTED: "prompted", LORA: "lora_all", LORA2: "lora_all", PSR: "psr", ALPHA: "alphasteer",
                     ARDITI: "static_r", "antidote": "antidote", "caa_static_r": "caa_static_r",
                     "stacked__safety__lora_r4": cm.PETRI_STACKED_KIND, ALPHA_T: "alphasteer"}
    assert all(e.resolved_spec is None and e.object_path is None for e in src.entries.values())
    assert src.get(PROMPTED).si_id == "constitution" and src.get(LORA).si_id == "safety" and src.get(ALPHA).si_id is None
    assert getattr(src, "serving_path_policy", "lens") == "lens"       # the partition accepts it as a lens-policy read
    # every harness Class A method is mapped today (the unmapped branch is a vocabulary guard): drop one to exercise it
    assert set(dc.CLASS_A_METHODS) <= set(ra.ARM_OBJECT_KINDS)
    monkeypatch.setattr(ra, "ARM_OBJECT_KINDS", {k: v for k, v in ra.ARM_OBJECT_KINDS.items() if k != "lora_r4"})
    with pytest.raises(ra.RegistryAdapterError, match="no object-kind mapping"):
        ra.kind_only_cell_source([LORA])


# --- 3. the selector script — the ONE call site -------------------------------------------------------------------------

def _selector(*argv, check=True):
    r = subprocess.run([sys.executable, str(SELECTOR), *argv], capture_output=True, text=True, cwd=REPO)
    if check:
        assert r.returncode == 0, (r.stdout, r.stderr)
    return r


def _sh_vars(out: str) -> dict:
    vals = {}
    for line in out.splitlines():
        name, _, quoted = line.partition("=")
        vals[name] = shlex.split(quoted)[0] if quoted else ""
    return vals


def test_selector_sh_output_evals_in_bash_and_matches_the_rule():
    r = _selector("--cells", "none", ALPHA, "--sh")                  # AlphaSteer-only: the lens-only shape
    v = _sh_vars(r.stdout)
    assert set(v) == {"F22_SERVING_PATH_SELECTED", "F22_SERVING_PATH_MODE", "F22_SERVING_PATH_LOG", "F22_SERVING_PATH_SELECTION_JSON"}
    assert (v["F22_SERVING_PATH_SELECTED"], v["F22_SERVING_PATH_MODE"]) == ("lens", "auto")
    assert v["F22_SERVING_PATH_LOG"] == "[serving-path] AUTO-selected lens: every steered cell is lens-only (alphasteer×1)" + OVERRIDE
    rec = json.loads(v["F22_SERVING_PATH_SELECTION_JSON"])
    assert rec["cells_requested"] == ["none", ALPHA] and rec["cells_requested_traintest"] == ["none", ALPHA]
    assert rec["cells_source"] == "--cells" and rec["spec_source"] == "kind_only" and rec["rule"] == "kind-aware-2026-09-15"
    # bash evals it (the single quotes inside the AlphaSteer reason survive shlex quoting)
    r2 = _selector("--cells", "none", ALPHA, LORA, "--cells-source", "manifest:x.yaml", "--sh")
    b = subprocess.run(["bash", "-c", 'set -u\n' + r2.stdout + '\necho "$F22_SERVING_PATH_SELECTED|$F22_SERVING_PATH_MODE"; echo "$F22_SERVING_PATH_LOG"; '
                        'python3 -c "import json,sys; r=json.loads(sys.argv[1]); print(r[\'cells_source\'], r[\'skipped\'][0][\'kind\'])" "$F22_SERVING_PATH_SELECTION_JSON"'],
                       capture_output=True, text=True)
    assert b.returncode == 0, (b.stdout, b.stderr)
    assert b.stdout.splitlines() == ["lora_native|auto",
                                     f"[serving-path] AUTO-selected lora_native: 1 of 2 steered cells natively servable; skipped by name: {ALPHA} (alphasteer)" + OVERRIDE,
                                     "manifest:x.yaml alphasteer"]


def test_selector_reads_a_real_manifest_and_agrees_with_the_configs_cells():
    """The (KEY, VIEW) manifest is the cell list in harness spelling — the gemma4e4b main view (LoRA + PSR + prompted)
    selects native with every steered cell served (the four PSR cells are natively servable kinds); the train/test
    spellings in the record are the config's."""
    import yaml
    man = "configs/capeval/manifests/petri_full22_gemma4e4b.yaml"
    r = _selector("--manifest", man, "--json")
    rec = json.loads(r.stdout)
    assert rec["selected"] == "lora_native" and rec["cells_source"] == f"manifest:{man}"
    assert rec["cells_requested"] == list(cm.load_petri_spec(REPO / man).cells)
    assert rec["cells_requested_traintest"] == yaml.safe_load((REPO / "configs/sdfbase_full22_gemma4e4b.yaml").read_text())["explicit_cells"]
    assert rec["log_line"].startswith("[serving-path] AUTO-selected lora_native: 8 of 8 steered cells natively servable; skipped by name: none")
    assert rec["cells_by_disposition"]["skip"] == [] and {c for c in rec["cells_requested"] if "psr" in c} <= set(rec["cells_by_disposition"]["serve"])
    assert rec["cells_by_disposition"]["passthrough"] == ["none", PROMPTED]


def test_selector_explicit_and_refusals():
    v = _sh_vars(_selector("--cells", ALPHA, "--explicit", "lora_native", "--sh").stdout)
    assert (v["F22_SERVING_PATH_SELECTED"], v["F22_SERVING_PATH_MODE"]) == ("lora_native", "explicit")
    assert v["F22_SERVING_PATH_LOG"] == "[serving-path] EXPLICIT lora_native (F22_SERVING_PATH set by caller)"
    assert json.loads(v["F22_SERVING_PATH_SELECTION_JSON"])["cells_by_disposition"] is None
    v = _sh_vars(_selector("--sh").stdout)                                # no cells at all
    assert v["F22_SERVING_PATH_SELECTED"] == "lora_native" and "no cell list (source: none)" in v["F22_SERVING_PATH_LOG"]
    for argv, needle in ((["--cells", ALPHA, ALPHA], "named twice"), (["--cells", "bogus__thing"], "malformed cell id"),
                         (["--manifest", "/nonexistent/petri.yaml"], "REFUSED")):
        r = _selector(*argv, "--sh", check=False)
        assert r.returncode == 2 and r.stdout == "" and "[serving-path] REFUSED:" in r.stderr and needle in r.stderr, (argv, r.stderr)
    r = _selector("--cells", ALPHA, "--explicit", "auto", "--sh", check=False)
    assert r.returncode == 2 and "invalid choice" in r.stderr


def test_selector_runs_without_torch_numpy_or_transformers():
    """What the node has at f22_init: stdlib python + PyYAML. The selector's whole import chain and a selection run
    with the heavy modules blocked — a torch import anywhere on the path fails this."""
    code = ("import sys\n"
            "for m in ('torch','numpy','transformers','safetensors','vllm','pandas','peft'): sys.modules[m] = None\n"
            f"sys.argv = ['x', '--cells', 'none', '{ALPHA}', '{PSR}', '--sh']\n"
            f"import runpy; runpy.run_path({str(SELECTOR)!r}, run_name='__main__')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, (r.stdout, r.stderr)
    # AlphaSteer skipped by kind, the psr cell a natively servable kind: native — classified with no torch imported
    assert _sh_vars(r.stdout)["F22_SERVING_PATH_SELECTED"] == "lora_native"


def test_the_shells_reach_the_rule_only_through_the_selector():
    """ONE Python function, called through ONE script; the shells carry no predicate on cell names or kinds."""
    for surface in (COMMON, DRIVE):
        text = surface.read_text()
        assert "serving_path_select.py" in text, surface.name
        assert "F22_SERVING_PATH_SELECTED" in text and "[serving-path] EXPLICIT $F22_SERVING_PATH (F22_SERVING_PATH set by caller)" in text
        for word in ("alphasteer", "arditi", "static_r", "psr_kl_all"):
            assert not any(word in l for l in text.splitlines() if not l.lstrip().startswith("#") and "case" in l), (surface.name, word)
    sel_src = SELECTOR.read_text()
    assert "ra.select_serving_path(" in sel_src and "ra.kind_only_cell_source(" in sel_src
    callers = [p for p in (REPO / "scripts").rglob("*.py") if "select_serving_path(" in p.read_text()]
    assert callers == [SELECTOR]


# --- 4. the real launcher pairings: the launch's OWN manifest, never the campaign's -----------------

def test_selector_on_the_real_alphasteer_campaign_and_strength_screen_manifests_selects_the_lens():
    """The AlphaSteer campaign's manifests (as22_targets.manifest_rel — the ONE table as22_resolve_config reads) and the
    strength screen's (`alphasteer_f22_<key>_<view>.yaml`) name lens-only steered cells: the rule selects the LENS and the
    record names THAT manifest — where the (KEY, VIEW) campaign manifest for the same key would have selected native."""
    sys.path.insert(0, str(REPO / "scripts/alphasteer_f22"))
    import as22_targets as t
    for key in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b"):
        man = t.manifest_rel(key)
        assert man.startswith("configs/capeval/manifests/alphasteer_f22_") and (REPO / man).is_file(), man
        r = _selector("--manifest", man, "--json")
        rec = json.loads(r.stdout)
        assert rec["selected"] == "lens" and rec["cells_source"] == f"manifest:{man}", (key, rec["reason"])
        assert rec["cells_by_disposition"]["serve"] == [] and "alphasteer" in rec["cells_by_disposition"]["skip"], rec["cells_by_disposition"]
        assert rec["log_line"].startswith("[serving-path] AUTO-selected lens: every steered cell is lens-only (alphasteer×")
        # the SAME key's campaign manifest is a different cell set — LoRA-bearing on the paper targets
        campaign = f"configs/capeval/manifests/petri_full22_{key}{'' if t.view_of(key) == 'main' else '_' + t.view_of(key)}.yaml"
        if (REPO / campaign).is_file():
            assert json.loads(_selector("--manifest", campaign, "--json").stdout)["selected"] == "lora_native", campaign
    screens = sorted(p for p in (REPO / "configs/capeval/manifests").glob("alphasteer_f22_*_as_d*.yaml"))
    assert len(screens) >= 10, len(screens)
    for man in screens:
        rec = json.loads(_selector("--manifest", str(man.relative_to(REPO)), "--json").stdout)
        assert rec["selected"] == "lens", (man.name, rec["reason"])
