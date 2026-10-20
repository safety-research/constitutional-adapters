#!/usr/bin/env python
"""Derive the misalignment-eval serving manifests and train/test configs from the
full22 wave's GENERATED files — never hand-edited, byte-stable, idempotent.

For each target key in :data:`TARGETS` (a key is a (roster target, wave view) pair, :func:`wave_of`; the
wave's file stems already spell the view, so the key doubles as the stem):

  configs/capeval/manifests/petri_full22_<target>.yaml  ->  configs/capeval/manifests/misalign_full22_<target>.yaml
  configs/sdfbase_full22_<target>.yaml                  ->  configs/misalign_full22_<target>.yaml

The manifest keeps EVERY field of the Petri manifest (the same target pins, serving flags, chat-template
hash, prompted prefix hashes, bloom options — the standup validates it as a Petri manifest and refuses a
drifted pin exactly as it does for the wave) with two changes:

  * ``audit: misalign_full22_<target>``;
  * ``cells`` = the Petri manifest's cells (the key's full22 view's objects, plus whatever prompted
    baseline the roster already serves there) + ``prompted__safety`` + ``prompted__constitution`` (the two
    SI arms — deduplicated, appended in that order) + ``alphasteer``
    iff ``configs/alphasteer_f22_pins.json[target]["selected__alphasteer"]["status"] == "served"``. A
    pins status of ``no_admissible_operating_point`` (glm46vflash) EXCLUDES the cell and is printed;
    a target the AlphaSteer campaign's roster never listed (llama8b) is ``not_in_campaign`` — excluded on
    that evidence; any other status, or a missing pin for a listed target, refuses (never a guess).

The TrainTest config is the wave's config with its header rewritten, ``explicit_cells`` = the manifest
cells translated through ``registry_adapter.traintest_cell_id`` (harness ``prompted__safety`` <->
TrainTest ``prompted__safety_refusal``, ``prompted__constitution`` <-> ``prompted__si_constitution``,
Class B ``alphasteer`` the identity), and the three per-run keys (``defenses_dir`` / ``out_dir`` /
``private_run``) renamed into the misalign namespace. Everything else is byte-for-byte the wave's:
``traintest_cell_source`` derives the SI vocabulary from ``explicit_cells`` (an explicit-cells config
declares no ``si_ids``), so listing the two prompted cells is what makes ``safety_refusal`` and
``si_constitution`` enumerable; the Class B arm is enumerable because ``alphasteer`` is in
``methods.ARMS`` and the bank view (misalign_bank_view.sh) stages its object under that id.

The BASELINE keys ``<family>_bl`` (the two published baselines on the agentic instrument, LOWEST
priority) are generated from the family's main-view sources with a FIXED cell list
``[none, arditi, alphasteer]`` — the Arditi refusal direction and the AlphaSteer RE-BAKE rung, pinned by sha256 in
``configs/misalign_baseline_pins.json`` (the campaign's ``alphasteer_f22_pins.json`` names the ORIGINAL selected
object, a different artifact measured under the same cell id in the family's own panel; the two never share a
results root). The pins' chat-template hash and model revision must equal the family manifest's, else refusal.

The EM MODEL-ORGANISMS keys ``<family>_org`` (organisms mode serves the EM adapters as vLLM LoRA modules and
overlays every non-LoRA cell as ``<em-organism>::<cell>``, LENS path, em8q only). WITH a pins entry for the family in
``configs/misalign_org_pins.json`` (qwen35_4b: batch-64 preferred — the ReLU and LoRA constitution /
control arms plus BOTH differences, direct and distilled; the safety-distilled arms dropped; AlphaSteer
out) the object cells are EXACTLY the pinned cells — the effective-batch-64 seed-1 fits, re-hashed against
the pin by misalign_bank_view.sh, NOT the wave's main-view objects (the same cell ids name its older batch-8 fits) —
plus the two prompted arms; the psr_ce_all cells are IN for that key, an exception to the 2026-09-12
psr-out rule, and render_config's wave guard lets it drop ONLY the two safety cells. The pins carry the lane's own
quality verdicts (a missing ADMISSION.json, an ``admitted: false``) and the generated headers repeat them verbatim.
WITHOUT a pins entry (llama8b_org) the key serves the wave's main-view cells + the prompted arms, minus alphasteer.

LANE-SCOPED POLICIES. A target key in :data:`LANE_POLICY`
(the seed-replicate views ``qwen35_4b_s1|s2|s3`` of roster target ``qwen35_4b``) is rendered from its OWN wave pair
(``petri_full22_qwen35_4b_s<s>.yaml`` / ``sdfbase_full22_qwen35_4b_s<s>.yaml``, :func:`source_stem`) under the policy
INSTEAD of the em-misalign defaults above: SERVED cells = ``none`` + the view's ADMITTED steered objects (FIVE since
2026-09-17: the distilled PSR difference failed the lane's admission for every seed and is on no view; PSR cells INCLUDED —
the 2026-09-12 psr-out rule is the em-misalign lane's, and this lane's spec names the ReLU objects) + the TWO PROMPTED SI arms
the view's roster lists (``prompted: True`` — the spec's "none / prompted x3" puts
``prompted__safety`` / ``prompted__constitution`` on every seed view's engine, so every arm of seed s draws under seed base s;
a policy that admits them REFUSES a view whose Petri manifest does not list them — never a guess), NO ``alphasteer``; and
SERVED != MEASURED: the config carries ``misalign_policy.measure_none: false`` with the MEASURED cells (the admitted objects and
the prompted arms), which misalign_proxy.sbatch (through misalign_cells.py) turns into a runner ``cells.json`` WITHOUT ``none`` /
``none__lens0`` / ``none__lora0`` — the ``none`` route stays SERVED for the standup's health checks and steering-bites, and
the pairing anchor is the campaign's EXISTING ``qwen35_4b`` main-view ``none`` misalign store (no new unsteered
measurement anywhere in the lane). A policy cell the view's Petri manifest does not list refuses (never a guess). Every
other target renders byte-identically (``--check``).

Both outputs are validated before writing: the manifest through ``manifest.load_petri_spec`` +
``defense_cells.parse_cell_id`` on every cell, and the manifest/config cell sets are checked to be ONE
enumeration (the same check mt_serving_up.py makes at standup).

THE FRONT DOOR'S PHASE VIEWS. Every model of the models table
``reproduce/models.yaml`` (read through its ONE reader ``reproduce/lib/models.py``; ``SISTEER_MODELS_TABLE`` overrides the path, as the
stage scripts do) whose ``views.<view>`` matches :data:`PHASE_VIEW_RE` (``u0``, ``s<N>[p]``, ``d<a>[p<b>][_relu]``, ``dm<a>[p<b>][_relu]``;
``main`` / ``it`` are roster views, never phase views) is a target key ``<view_legacy_prefix>_<view>`` rendered from its OWN wave pair
``petri_full22_<key>_<view>.yaml`` / ``sdfbase_full22_<key>_<view>.yaml`` — DERIVED at import (:func:`front_door_targets`), never a hand
list, so a future model needs no edit here; an absent table contributes nothing, a table without its reader refuses, a derived key
that collides with a hand-listed one refuses. The FRONT-DOOR POLICY (spec FD 5: the full battery on all ten arms, the ReLU arms
included; FD 22: unsteered first): SERVED == MEASURED == EXACTLY the view's Petri manifest cells — ``none`` + the view's objects + the
prompted SI arms the roster lists on that view; nothing appended (the phase views partition the ten arms: u0 carries the two prompted
SIs, s2 the residual group, s1 the LoRA group, the dose views one difference object each), nothing dropped (the 2026-09-12 psr-out
rule is the em-misalign lane's), no ``alphasteer`` (not one of the ten arms; the campaign never fitted these models), every served
leg measured (``none`` included: the u0 view IS the unsteered measurement and a steered view's ``none`` is its same-standup anchor —
today's default rule, so the config carries NO ``misalign_policy`` block). ``--resolve <target>`` prints ``<roster target> <view>
<kind>`` for any target key (misalign_proxy.sbatch derives its (KEY, VIEW) for a front-door target from it: one table).

Usage (from the repo root, the grading host venv):
  PYTHONPATH=src python scripts/sdfbase/full22/misalign/make_misalign_manifest.py            # write
  PYTHONPATH=src python scripts/sdfbase/full22/misalign/make_misalign_manifest.py --check    # exit 1 on drift
  PYTHONPATH=src python scripts/sdfbase/full22/misalign/make_misalign_manifest.py --resolve qwen35_4b_demo_battery   # -> "qwen35_4b_demo battery front_door"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]
MANIFEST_DIR = REPO / "configs" / "capeval" / "manifests"
ALPHASTEER_PINS = REPO / "configs" / "alphasteer_f22_pins.json"
ALPHASTEER_ROSTER = REPO / "configs" / "alphasteer_f22_roster.yaml"
#: Targets served under a chat template OTHER than the pinned tokenizer's own. The campaign's
#: Llama checkpoint (NousResearch/Meta-Llama-3.1-8B-Instruct @d10aef79) ships a 348-byte MINIMAL template
#: (sha16 b48c47f644389271: the messages only, no default system block); Meta's stock Llama-3.1 template
#: (4,614 bytes, sha16 e10ca381b1ccc5cf, byte-identical in meta-llama/ and unsloth/Llama-3.1-8B-Instruct)
#: always emits the "Cutting Knowledge Date / Today Date" system header. The EM model organisms were trained
#: and first measured (2026-09-11) under the stock template and read ~4x less misaligned under the minimal
#: one (financial 31.2% -> 7.5%), so the organisms transfer target `llama8b` is served under the STOCK
#: template: the manifest pins ITS hash, misalign_proxy.sbatch passes it to the standup (--chat-template),
#: the standup re-hashes and records it. `llama8b_it` (the agentic panel, already measured) keeps the
#: campaign's pinned render.
CHAT_TEMPLATE_OVERRIDE = {"llama8b_org": REPO / "configs" / "capeval" / "chat_templates" / "llama31_instruct_stock.jinja"}


def chat_template_pin(target: str):
    """(sha16, path) of the template a target is served under when it is NOT the pinned tokenizer's; (None, None) otherwise."""
    path = CHAT_TEMPLATE_OVERRIDE.get(target)
    if path is None:
        return None, None
    if not path.is_file():
        raise SystemExit(f"{target}: chat template override {path} is not on this tree")
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16], path

#: qwen-base-seed-replicates: the seed-replicate views of qwen35_4b — roster views `s1|s2|s3`, TAG `qwen35_4b_s<s>` —
#: and their six DECLARED steered objects as HARNESS ids (registry_adapter.harness_cell_id of the roster cell ids: the two parents
#: per corpus and the two derived objects; the campaign naming grammar)
SEEDREP_TARGET = "qwen35_4b"
SEEDREP_VIEWS = ("s1", "s2", "s3")
#: the PAIRED views `s<s>p` — the composed difference_relu arm (the undistilled paired write, admitted 2026-09-20 with
#: the floor alarm by derived_admission.py's psr_paired_write branch) beside the seed's constitution LoRA (the GSM8K
#: standup companion) and the LoRA difference: the paper's difference-vs-difference comparison off one standup. Its own
#: view per seed because a bank view may not carry an entry an earlier roster does not name while jobs pinned to that
#: roster run against it. Target keys `qwen35_4b_s<s>p`; lane `paired`, never `reseed`.
PAIRED_VIEWS = ("s1p", "s2p", "s3p")
def paired_of(view: str) -> str:
    """The seed view a paired view is an alias of — read from the roster's `paired_of` (the twin marker), refusing a paired view
    without it or one whose key does not spell its own name (`s<s>p` over `s<s>`)."""
    import yaml as _yaml
    roster = _yaml.safe_load(ROSTER.read_text(encoding="utf-8"))
    v = roster["targets"][SEEDREP_TARGET]["views"][view]
    seed = v.get("paired_of")
    if not seed or view != f"{seed}p":
        raise SystemExit(f"{SEEDREP_TARGET}/{view}: a paired view declares `paired_of: <seed view>` and is named `<seed view>p` (got {seed!r})")
    return seed


def paired_reserved_cells(view: str) -> list[str]:
    """The TrainTest cell ids a paired view RE-SERVES from its seed view (the roster's `cells` the two views share, under the SAME
    staged ids) — the twin the misalign_policy block records. Refuses a shared cell whose staged id differs (a third object under
    the twin's name)."""
    import yaml as _yaml
    roster = _yaml.safe_load(ROSTER.read_text(encoding="utf-8"))
    views = roster["targets"][SEEDREP_TARGET]["views"]
    seed = paired_of(view)
    own, theirs = views[view].get("cells") or {}, views[seed].get("cells") or {}
    shared = [c for c in own if c in theirs]
    bad = {c: (own[c], theirs[c]) for c in shared if own[c] != theirs[c]}
    if bad:
        raise SystemExit(f"{SEEDREP_TARGET}/{view}: re-served cell(s) differ from {seed}'s staged ids: {bad}")
    if not shared:
        raise SystemExit(f"{SEEDREP_TARGET}/{view}: `paired_of: {seed}` but no cell is re-served from it")
    return shared
PAIRED_HARNESS = ("si_constv2_mix_full__psr_ce_all_minus_ctl", "si_constv2_mix_full__lora_r4_ce",
                  "si_constv2_mix_full__lora_r4_ce_minus_ctl")
SEEDREP_SIX_HARNESS = ("si_constv2_mix_full__psr_ce_all", "si_constv2_mix_full__lora_r4_ce",
                       "si_constv2ctl2_full__psr_ce_all", "si_constv2ctl2_full__lora_r4_ce",
                       "si_constv2_mix_full__lora_r4_ce_minus_ctl",
                       "si_constv2_mix_full__psr_ce_all_minus_ctl_distilled")
#: decisions O20 (2026-09-17): the distilled PSR difference `si_constv2_mix_full__psr_ce_all_minus_ctl_distilled` FAILED the lane's
#: admission predicate for EVERY seed (scripts/sdfbase/seedrep/derived_admission.py: recovery verdict and/or gate liveness min over
#: layers 0.06-0.21 < the published Qwen derived object's floor 0.8135; the records are RS/reseed/admission/qwen35_4b_base_eb64_s<s>__
#: constv2-mix-full__psr_ce_all_minus_ctl_distilled.json) and is DROPPED from every seed view: never staged, never pinned, never
#: served. The policy therefore serves + measures the FIVE ADMITTED objects; the declared six above stay as the lane's grammar.
SEEDREP_DROPPED_HARNESS = ("si_constv2_mix_full__psr_ce_all_minus_ctl_distilled",)
SEEDREP_ADMITTED_HARNESS = tuple(c for c in SEEDREP_SIX_HARNESS if c not in SEEDREP_DROPPED_HARNESS)
#: the LANE POLICY per target key: `cells` = the view's ADMITTED objects (SEEDREP_ADMITTED_HARNESS, five; harness ids; SERVED beside `none`, and the
#: MEASURED set), `psr: include` (the PSR cells stay), `prompted: True` (the view's two prompted SI arms,
#: PROMPTED_CELLS, are SERVED and MEASURED too — every seed view's roster lists them since 2026-09-17) / `alphasteer: False` (never served),
#: `measure_none: False` (`none` served for health only; the runner never measures it), `pairing_anchor` = the store the
#: admitted cells are paired against downstream (the campaign anchor, under $SHARED/exp/). Consulted BEFORE psr_out and the
#: prompted / alphasteer appends; a target absent here renders exactly as before.
LANE_POLICY = {f"{SEEDREP_TARGET}_{v}": {"lane": "reseed", "view": v, "cells": SEEDREP_ADMITTED_HARNESS, "psr": "include",
                                        "prompted": True, "alphasteer": False, "measure_none": False,
                                        "pairing_anchor": f"exp/misalign/{SEEDREP_TARGET}/agentic/none"}
               for v in SEEDREP_VIEWS}
#: the paired views: the three harness cells above, no prompted arms (the roster lists none on these views), `none` served for
#: health only and the SHARED campaign anchor kept: the composed arm pairs against the same `none` as the five s<s> cells
#: (the GSM8K matrix's per-view `none` is that reader's own requirement, not this leg's).
LANE_POLICY.update({f"{SEEDREP_TARGET}_{v}": {"lane": "paired", "view": v, "cells": PAIRED_HARNESS, "psr": "include",
                                              "prompted": False, "alphasteer": False, "measure_none": False,
                                              "pairing_anchor": f"exp/misalign/{SEEDREP_TARGET}/agentic/none"}
                    for v in PAIRED_VIEWS})
#: the FRONT DOOR's phase views, DERIVED from the models table — see the
#: module docstring. The grammar is the seedrep grader twin's (`grade_full22_native.PHASE_VIEW_RE`): the same tags there and here.
#: `s<N>r` = the ReLU-parent subset of phase N (`s2r` = control_relu + constitution_relu at alpha 1; a view outside this grammar
#: renders nothing here while the planner still plans it) — the same grammar in every copy, pinned equal by tests
PHASE_VIEW_RE = re.compile(r"^(u0|s[0-9]+r?p?|s[0-9]+rl|d[0-9]+(p[0-9]+)?(_relul?)?|dm[0-9]+(p[0-9]+)?(_relul?)?)$")
#: a STEERED-WAVE phase view (`s<N>r`) measures no unsteered leg — its Petri
#: manifest lists no `none`, its misalign pair serves EXACTLY its cells and carries `misalign_policy.measure_none: false` so the base route
#: the standup serves implicitly (health checks, attestation) is never measured. Mirror of f22_roster.STEERED_WAVE_VIEW_RE (tests hold them equal).
STEERED_WAVE_VIEW_RE = re.compile(r"^s[0-9]+rl?$")


MODELS_TABLE = Path(os.environ.get("SISTEER_MODELS_TABLE") or (REPO / "reproduce" / "models.yaml"))
MODELS_TABLE_READER = REPO / "reproduce" / "lib" / "models.py"     # the ONE reader of the table (duplicate keys refused; review A9)
FRONT_DOOR_STATUS = "front_door"     # the alphasteer status a front-door key reports: exactly the view's cells, no campaign pin consulted


def _models_reader():
    """``reproduce/lib/models.py`` loaded by path (the ONE reader; duplicate keys refused); absent beside this tree, refused."""
    if not MODELS_TABLE_READER.is_file():
        raise SystemExit(f"models table reader {MODELS_TABLE_READER} is not on this tree — refusing to read the table another way")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_misalign_reproduce_models_table", MODELS_TABLE_READER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_models_table(path: Path) -> dict:
    """``reproduce/models.yaml`` through its ONE reader (``reproduce/lib/models.py`` ``load_models``); a table without its reader
    beside this tree is refused, never read another way."""
    return _models_reader().load_models(str(path))


def front_door_targets(table_path: "Path | str | None" = None) -> dict:
    """``{target key: (model key, view)}`` for every PHASE view of every model in the models table that names a
    ``view_legacy_prefix``: ``models.<key>.views.<view>`` matching PHASE_VIEW_RE -> ``<view_legacy_prefix>_<view>``. An ABSENT table
    contributes nothing (the paper's tree before the front door); the paper's four models carry ``main`` / ``it`` only and contribute
    nothing; two models sharing a prefix + view refuse (one key, one wave pair)."""
    p = Path(table_path) if table_path is not None else MODELS_TABLE
    if not p.is_file():
        return {}
    out: dict = {}
    reader = _models_reader()
    for key, m in reader.load_models(str(p)).items():
        if not isinstance(m, dict):
            continue
        prefix, views = m.get("view_legacy_prefix"), m.get("views")
        if not prefix or not isinstance(views, dict):
            continue
        phase = reader.phase_views_of(m)             # the reader's ONE classification: the grammar's views + the row's declared wave (demo-import M3)
        for view in views:
            view = str(view)
            if view not in phase:
                continue
            tag = f"{prefix}_{view}"
            if tag in out and out[tag] != (str(key), view):
                raise SystemExit(f"{p.name}: models {out[tag][0]!r} and {key!r} both derive the misalign target {tag!r} — one key, one wave pair")
            out[tag] = (str(key), view)
    return out


FRONT_DOOR = front_door_targets()
#: the hand-listed campaign keys (the paper's models, their panels and the seed-replicate enumerations); a derived front-door key may
#: not shadow one (two policies under one key), so the union refuses on a collision instead of picking silently
HAND_LISTED_TARGETS = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash", "llama8b", "llama8b_it",
                       "gemma4e4b_it", "qwen35_4b_it", "llama8b_org", "qwen35_4b_org", "qwen35_4b_psr", "qwen35_4b_allpos",
                       "gemma4e4b_bl", "qwen35_4b_bl", "nemotron3nano4b_bl", "llama8b_bl",
                       *LANE_POLICY)
if set(HAND_LISTED_TARGETS) & set(FRONT_DOOR):
    raise SystemExit(f"{MODELS_TABLE.name}: front-door target(s) {sorted(set(HAND_LISTED_TARGETS) & set(FRONT_DOOR))} collide with hand-listed "
                     "misalign targets — one key, one policy; rename the model's view_legacy_prefix")
TARGETS = (*HAND_LISTED_TARGETS, *FRONT_DOOR)
ROSTER = REPO / "configs" / "sdfbase_full22_roster.yaml"
PINS = REPO / "configs" / "sdfbase_full22_pins.json"


def is_front_door(target: str) -> bool:
    return target in FRONT_DOOR


def _single_phase_keys(table_path: "Path | str | None" = None) -> frozenset:
    """The models of the table on the SINGLE-PHASE recipe (reproduce/lib/models.single_phase): a front-door model
    with no u0 view — its one wave carries its own unsteered legs, so none of its views is a steered-wave view."""
    p = Path(table_path) if table_path is not None else MODELS_TABLE
    if not p.is_file():
        return frozenset()
    import importlib.util
    spec = importlib.util.spec_from_file_location("_misalign_reproduce_models_table", MODELS_TABLE_READER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return frozenset(k for k in mod.front_door_keys(str(p)) if mod.single_phase(k, str(p)))


SINGLE_PHASE_KEYS = _single_phase_keys()


def is_steered_wave_target(target: str) -> bool:
    """A front-door target whose wave view measures no unsteered leg: s<N>r, and every front-door view except u0 of a
    model that runs in phases (a table that declares u0); a single-phase model's views (SINGLE_PHASE_KEYS) never are."""
    if not is_front_door(target):
        return False
    key, view = wave_of(target)
    return bool(STEERED_WAVE_VIEW_RE.match(view)) or (view != "u0" and key not in SINGLE_PHASE_KEYS)


def lane_policy(target: str) -> "dict | None":
    return LANE_POLICY.get(target)
#: the published-baselines keys: `<family>_bl` -> the family's main view, cells [none, arditi, alphasteer]
BASELINE_SUFFIX = "_bl"
BASELINE_FAMILIES = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b")
BASELINE_CELLS = ("arditi", "alphasteer")
BASELINE_PINS = REPO / "configs" / "misalign_baseline_pins.json"
BASELINE_STATUS = "baseline"   # the alphasteer status a `_bl` key reports: the RE-BAKE rung from BASELINE_PINS, not the campaign pin
#: 2026-09-17: the three ReLU / PSR `psr_ce_all` cells RETURN for the qwen leaking vendor panel, served on the
#: NATIVE path (the residual slot) and paired against the panel's existing lens `none` (there is no native anchor)
#: — a `<family>_psr` key: `none` + the wave main view's psr_ce_all cells, no prompted arm, no AlphaSteer
#: (OUT everywhere since 2026-09-17), agentic only. The psr-out rule of 2026-09-12 still governs every OTHER key.
PSR_PANEL_SUFFIX = "_psr"
PSR_PANEL_TOKEN = "psr_ce_all"
PSR_PANEL_STATUS = "psr_panel"


def is_psr_panel(target: str) -> bool:
    return target.endswith(PSR_PANEL_SUFFIX)


#: the Qwen Arditi ALL-POSITIONS arm: a `_bl`-style panel keyed WITHOUT the `_bl` suffix so the
#: baselines machinery is not engaged — cells [none, arditi_allpos] on the `it` view, a SINGLE-OBJECT native standup, the object
#: pinned in configs/arditi_allpos_pins.json (qwen35_4b.ar_c0p3__arditi_allpos; c = 0.3 raw_coefficient is a DECLARED
#: OVERRIDE of the selection rule, whose output is c = 0.2 — nothing here implies the rule chose 0.3).
ALLPOS_TARGET = "qwen35_4b_allpos"
ALLPOS_CELL = "arditi_allpos"
ALLPOS_STATUS = "allpos_panel"


def is_allpos_panel(target: str) -> bool:
    return target == ALLPOS_TARGET


ALLPOS_PINS = REPO / "configs" / "arditi_allpos_pins.json"
ALLPOS_STAGED_ID = "ar_c0p3__arditi_allpos"


def allpos_pin() -> dict:
    """The arm's pin entry (configs/arditi_allpos_pins.json qwen35_4b.ar_c0p3__arditi_allpos); absent = refusal."""
    if not ALLPOS_PINS.is_file():
        raise SystemExit(f"{ALLPOS_TARGET}: {ALLPOS_PINS.name} is not on this tree")
    rec = json.loads(ALLPOS_PINS.read_text(encoding="utf-8")).get("qwen35_4b", {}).get(ALLPOS_STAGED_ID)
    if not rec or not rec.get("object_sha256"):
        raise SystemExit(f"{ALLPOS_TARGET}: {ALLPOS_PINS.name} has no qwen35_4b.{ALLPOS_STAGED_ID} pin with an object_sha256")
    return rec


#: the EM MODEL-ORGANISMS panels (`<family>_org`) — the family's main-view cells + the prompted arms,
#: served in organisms mode with the organism adapters as `<em-tag>::<cell>` overlays (misalign_proxy.sbatch). NO alphasteer
#: cell: AlphaSteer is OUT everywhere since 2026-09-17, whatever the alphasteer pins say for the family.
ORG_PANEL_SUFFIX = "_org"
ORG_STATUS = "organisms_no_alphasteer"


def is_org_panel(target: str) -> bool:
    return target.endswith(ORG_PANEL_SUFFIX)


#: the PINNED organisms panel (batch-64 fits are preferred): an `_org` key whose family has an entry in
#: configs/misalign_org_pins.json serves EXACTLY the pinned objects (the reseed lane's effective-batch-64 seed-1 fits, each
#: re-hashed against its pin by misalign_bank_view.sh) + the two prompted arms — never the wave's main-view objects.
ORG_PINS = REPO / "configs" / "misalign_org_pins.json"
ORG_PINS_SCHEMA = "misalign_org_pins/1"
ORG_PINNED_STATUS = "organisms_pinned"
#: the wave main-view cells the pinned organisms panel DROPS, by name (the safety-distilled arms are out of scope here);
#: every other wave cell must be served, psr_ce_all included — render_config's missing_wave guard
ORG_DROPPED_WAVE_CELLS = ("safety_refusal__lora_r4", "safety_refusal__psr_kl_all")
ORG_IDENTITY_SIDECARS = ("train_budget_meta.json", "fit_provenance.json", "derived_meta.json")


def org_pins(target: str) -> dict | None:
    """The family's pinned organisms-panel record (``configs/misalign_org_pins.json[<roster target>]`` + the file's cell
    order under ``"_cells"``), or None when ``target`` is not an ``_org`` key or the file has no entry for its family
    (llama8b_org: the wave path). A malformed file or entry is a refusal, never a guess."""
    if not is_org_panel(target) or not ORG_PINS.is_file():
        return None
    pins = json.loads(ORG_PINS.read_text(encoding="utf-8"))
    if pins.get("_schema") != ORG_PINS_SCHEMA or not isinstance(pins.get("_cells"), list) or not pins["_cells"]:
        raise SystemExit(f"{ORG_PINS.name}: unexpected schema/_cells")
    fam, _ = wave_of(target)
    rec = pins.get(fam)
    if rec is None:
        return None
    if not isinstance(rec, dict):
        raise SystemExit(f"{ORG_PINS.name}: malformed entry for {fam}")
    for k in ("model", "resolved_revision", "chat_template_hash", "fit_lane", "fit_model_key", "fit_model_id",
              "fit_model_revision", "seed", "effective_batch"):
        if rec.get(k) in (None, ""):
            raise SystemExit(f"{ORG_PINS.name}: {fam} lacks {k}")
    for c in pins["_cells"]:
        o = rec.get(c)
        if not isinstance(o, dict):
            raise SystemExit(f"{ORG_PINS.name}: {fam} has no pin for {c}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(o.get("object_sha256", ""))):
            raise SystemExit(f"{ORG_PINS.name}: {fam}/{c} lacks a 64-hex object_sha256")
        for k in ("bank", "source", "kind", "identity_sidecar", "seed", "effective_batch", "fit_model_key"):
            if o.get(k) in (None, ""):
                raise SystemExit(f"{ORG_PINS.name}: {fam}/{c} lacks {k}")
        if o["identity_sidecar"] not in ORG_IDENTITY_SIDECARS:
            raise SystemExit(f"{ORG_PINS.name}: {fam}/{c} identity_sidecar {o['identity_sidecar']} is not one of {ORG_IDENTITY_SIDECARS}")
        if not str(o["source"]).startswith(f"{o['bank']}/"):
            raise SystemExit(f"{ORG_PINS.name}: {fam}/{c} source {o['source']} is not under its bank {o['bank']}/")
        if (o["seed"], o["effective_batch"], o["fit_model_key"]) != (rec["seed"], rec["effective_batch"], rec["fit_model_key"]):
            raise SystemExit(f"{ORG_PINS.name}: {fam}/{c} (seed, effective_batch, fit_model_key) differ from the family's — "
                             "one replicate per panel")
        if "admission" not in o:
            raise SystemExit(f"{ORG_PINS.name}: {fam}/{c} lacks an `admission` record (the lane's ADMISSION.json verdict, or null)")
        if o["admission"] is None and not o.get("admission_note"):
            raise SystemExit(f"{ORG_PINS.name}: {fam}/{c} has no ADMISSION record and no admission_note saying what the lane's record is")
    return {**rec, "_cells": list(pins["_cells"])}


def _org_quality_line(o: dict) -> str:
    """One phrase on the reseed lane's OWN quality record for a pinned organisms-panel object — the header repeats what the
    lane recorded, never launders it: a fitted cell's VALIDATE.json, a derived object's ADMISSION.json verdict (admitted or
    NOT), or the absence of any admission record."""
    adm = o.get("admission")
    if isinstance(adm, dict):
        if adm.get("admitted") is True:
            return "ADMISSION.json admitted: true"
        return "ADMISSION.json admitted: FALSE — " + "; ".join(adm.get("reasons") or ["no reason recorded"])
    val = (o.get("records") or {}).get("VALIDATE.json")
    if isinstance(val, dict) and val.get("ok") is True:
        return "fitted cell: VALIDATE.json ok: true + COMPLETE"
    gl = o.get("gate_liveness") or {}
    return ("NO ADMISSION.json — gate_liveness.json only"
            + (f" (min {gl['min']:.4f} over {gl.get('n_layers')} layers)" if gl.get("min") is not None else ""))


def is_baseline(target: str) -> bool:
    return target.endswith(BASELINE_SUFFIX)


def baseline_family(target: str) -> str:
    fam = target[: -len(BASELINE_SUFFIX)]
    if fam not in BASELINE_FAMILIES:
        raise SystemExit(f"{target}: not a baseline family ({BASELINE_FAMILIES})")
    return fam


def baseline_pins(target: str) -> dict:
    """The family's two pinned baseline objects; a malformed pins file is a refusal, never a guess."""
    fam = baseline_family(target)
    if not BASELINE_PINS.is_file():
        raise SystemExit(f"{target}: {BASELINE_PINS.name} is not on this tree")
    pins = json.loads(BASELINE_PINS.read_text(encoding="utf-8"))
    if pins.get("_schema") != "misalign_baseline_pins/1" or tuple(pins.get("_cells") or ()) != BASELINE_CELLS:
        raise SystemExit(f"{BASELINE_PINS.name}: unexpected schema/cells")
    rec = pins.get(fam)
    if not isinstance(rec, dict):
        raise SystemExit(f"{BASELINE_PINS.name}: no entry for {fam}")
    for c in BASELINE_CELLS:
        o = rec.get(c) or {}
        if not re.fullmatch(r"[0-9a-f]{64}", str(o.get("object_sha256", ""))):
            raise SystemExit(f"{BASELINE_PINS.name}: {fam}/{c} lacks a 64-hex object_sha256")
        for k in ("source", "identity_sidecar", "chat_template_hash", "resolved_revision", "selection_outcome"):
            if not o.get(k):
                raise SystemExit(f"{BASELINE_PINS.name}: {fam}/{c} lacks {k}")
        if not str(o["source"]).startswith(f"{o.get('bank')}/{fam}/"):
            raise SystemExit(f"{BASELINE_PINS.name}: {fam}/{c} source {o['source']} is not under its bank {o.get('bank')}/{fam}/")
    return rec
#: 2026-09-12: PSR is out — not run, not plotted. A misalign config generated after that date carries NO
#: PSR cell (the wave views still list them for the campaign's other evals); the panels run before it keep theirs as record.
PSR_OUT_TOKEN = "psr"


def psr_out(cells):
    return [c for c in cells if PSR_OUT_TOKEN not in c]


def psr_policy(target: str, cells):
    """The em-misalign ruling (psr_out) for every target but a LANE_POLICY one whose policy says `psr: include`
    (qwen-base-seed-replicates: its spec names the PSR objects; the 2026-09-12 ruling was that lane's)."""
    pol = lane_policy(target)
    if pol is not None and pol["psr"] == "include":
        return list(cells)
    if pol is not None and pol["psr"] != "out":
        raise SystemExit(f"{target}: LANE_POLICY psr={pol['psr']!r} is neither 'include' nor 'out'")
    if is_front_door(target):
        return list(cells)      # spec FD 5 names the ReLU arms; the 2026-09-12 psr-out rule is the em-misalign lane's
    return psr_out(cells)


def source_stem(target: str) -> str:
    """The wave config/manifest a misalign target is generated FROM: `<roster target>` for a main-view key,
    `<roster target>_<view>` for any other view (`_it`; the seed-replicate `_s1|_s2|_s3`) — so `llama8b_org` (main view,
    stock template) reads llama8b's, and `qwen35_4b_s1` reads the s1 view's own generated pair."""
    rt, view = wave_of(target)
    return rt if view == "main" else f"{rt}_{view}"
#: misalign target key -> (roster target, wave view). One served model can carry several object-bearing views
#: (`main` = the base-lane objects transferred onto the instruct model — the organism-transfer test; `it` = the
#: instruct-direct objects — the paper's fourth family); every other key is its own roster target's `main` view.
#: misalign_bank_view.sh and misalign_proxy.sbatch carry the same table.
WAVE_OF = {"llama8b": ("llama8b", "main"), "llama8b_it": ("llama8b", "it"),
           # the instruct-direct views of the two other families with a base model: agentic
           # misalignment on both instruct-direct and base->instruct objects, side by side
           "gemma4e4b_it": ("gemma4e4b", "it"), "qwen35_4b_it": ("qwen35_4b", "it"),
           "qwen35_4b_psr": ("qwen35_4b", "main"),   # the PSR / ReLU panel: the main view's psr_ce_all cells
           "qwen35_4b_allpos": ("qwen35_4b", "it"),   # The Arditi ALL-POSITIONS arm's misalignment panel (paper view `it`)
           # the EM-organisms transfer target: llama8b's main view served under the STOCK chat template; a key
           # of its own so the agentic `llama8b` panel keeps the campaign render its `llama8b_it` twin has
           "llama8b_org": ("llama8b", "main"),
           # the Qwen3.5-4B EM model organisms (LoRA r32 adapters trained on the paper's datasets,
           # $SHARED/exp/models/em_organisms/qwen35_4b/<tag>/adapter) x qwen35_4b's main-view objects, em8q only; the
           # campaign's own pinned render (thinking off) — no template override: the organisms were trained under it
           "qwen35_4b_org": ("qwen35_4b", "main"),
           # the published baselines: every family's MAIN view sources (the served model, geometry and template
           # of the campaign panel); the cells are the two pinned baseline objects, not the wave's
           "gemma4e4b_bl": ("gemma4e4b", "main"), "qwen35_4b_bl": ("qwen35_4b", "main"),
           "nemotron3nano4b_bl": ("nemotron3nano4b", "main"), "llama8b_bl": ("llama8b", "main"),
           # qwen-base-seed-replicates: the seed-replicate views, each its own wave pair (source_stem)
           **{f"{SEEDREP_TARGET}_{v}": (SEEDREP_TARGET, v) for v in SEEDREP_VIEWS},
           **{f"{SEEDREP_TARGET}_{v}": (SEEDREP_TARGET, v) for v in PAIRED_VIEWS},
           # paper-companion-repo: the front door's phase views, (model key, view) from the models table (front_door_targets)
           **FRONT_DOOR}
#: the two prompted SI arms every target serves in this eval, harness ids
PROMPTED_CELLS = ("prompted__safety", "prompted__constitution")
ALPHASTEER_CELL = "alphasteer"
ALPHASTEER_STAGED_ID = "selected__alphasteer"
#: the alphasteer statuses this generator understands; anything else is a refusal
ALPHASTEER_SERVED = "served"
ALPHASTEER_NOT_SERVED = "no_admissible_operating_point"
ALPHASTEER_NOT_IN_CAMPAIGN = "not_in_campaign"   # the campaign's roster never listed the target: no arm exists
ALPHASTEER_LANE_EXCLUDED = "lane_excluded"       # a LANE_POLICY target with alphasteer: False — excluded by policy, never by pin

_CELL_LINE = re.compile(r"^    - (\S+)\s*$")


def wave_of(target: str) -> tuple[str, str]:
    """(roster target, wave view) behind a misalign target key."""
    return WAVE_OF.get(target, (target, "main"))


def _sisteer():
    """Lazy import of the two vocabulary-owning modules (repo src on sys.path); torch-free."""
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri import registry_adapter as ra
    return cm, dc, ra


#: the AlphaSteer campaign's roster + pins live on its own lane (alphasteer-f22-run); a tree without them (main,
#: 2026-09-12: "no AlphaSteer object can be served on main today") serves NO alphasteer cell on any target — a
#: recorded state the generator prints, and the configs regenerate WITH the cell when that lane lands (--check
#: shows the drift then)
ALPHASTEER_NOT_ON_TREE = "not_on_tree"


def alphasteer_status(target: str) -> tuple[str, dict]:
    if is_baseline(target):
        return BASELINE_STATUS, baseline_pins(target)["alphasteer"]
    if is_front_door(target):
        # the front door serves exactly its view's cells (no AlphaSteer arm among the ten, and the campaign never fitted these models)
        return FRONT_DOOR_STATUS, {}
    pol = lane_policy(target)
    if pol is not None and not pol["alphasteer"]:
        # the lane's policy excludes the arm whatever the campaign pins say: a recorded exclusion, not an unpinned object
        return ALPHASTEER_LANE_EXCLUDED, {}
    roster_target, _ = wave_of(target)
    if not ALPHASTEER_PINS.is_file() or not ALPHASTEER_ROSTER.is_file():
        return ALPHASTEER_NOT_ON_TREE, {}
    pins = json.loads(ALPHASTEER_PINS.read_text(encoding="utf-8"))
    rec = (pins.get(roster_target) or {}).get(ALPHASTEER_STAGED_ID)
    if rec is None:
        # a target the AlphaSteer campaign never listed (its roster IS the campaign's definition) has no arm to
        # pin — an absence with evidence, not an unpinned object (llama8b, 2026-09-11)
        roster = yaml.safe_load(ALPHASTEER_ROSTER.read_text(encoding="utf-8"))
        if roster_target not in (roster.get("targets") or {}):
            return ALPHASTEER_NOT_IN_CAMPAIGN, {}
        raise SystemExit(f"{ALPHASTEER_PINS.name}: no {roster_target}/{ALPHASTEER_STAGED_ID} entry although "
                         f"{ALPHASTEER_ROSTER.name} lists the target — the arm's inclusion is decided by the pins, "
                         "never assumed")
    status = rec.get("status")
    if status == ALPHASTEER_SERVED:
        if not rec.get("object_sha256"):
            raise SystemExit(f"{ALPHASTEER_PINS.name}: {target} is 'served' but carries no object_sha256")
        return status, rec
    if status == ALPHASTEER_NOT_SERVED:
        if rec.get("object_sha256") is not None:
            raise SystemExit(f"{ALPHASTEER_PINS.name}: {target} is '{status}' but carries an object_sha256")
        return status, rec
    raise SystemExit(f"{ALPHASTEER_PINS.name}: {target}/{ALPHASTEER_STAGED_ID} status {status!r} is neither "
                     f"{ALPHASTEER_SERVED!r} nor {ALPHASTEER_NOT_SERVED!r} — refusing to guess")


def alphasteer_view(target: str) -> str | None:
    """The alphasteer roster's view for the target (``it`` on gemma/qwen, ``main`` on nemotron/glm) — the
    bank layout is per target, not per view, so this is recorded for the header only."""
    if not ALPHASTEER_ROSTER.is_file():
        return None
    roster = yaml.safe_load(ALPHASTEER_ROSTER.read_text(encoding="utf-8"))
    t = roster["targets"].get(wave_of(target)[0]) or {}
    for view, v in (t.get("views") or {}).items():
        if (v.get("cells") or {}).get(ALPHASTEER_CELL) == ALPHASTEER_STAGED_ID:
            return view
    return None


def misalign_cells(petri_cells: list[str], target: str) -> tuple[list[str], str]:
    """The misalign manifest's harness cell list + the alphasteer status it was decided by."""
    if is_baseline(target):
        # the wave's objects and the prompted arms are NOT served here: the panel is none + the two pinned baselines
        return ["none", *BASELINE_CELLS], BASELINE_STATUS
    pol = lane_policy(target)
    if pol is not None:
        # THE LANE POLICY: served = none + exactly the policy's cells, every one of which the view's Petri
        # manifest must list (a cell outside the view refuses — never a guess); + the two PROMPTED SI arms iff the policy
        # admits them (`prompted: True`) — the view's manifest must list BOTH (the roster
        # puts them on every seed view; a view that lacks one refuses by name, never a silent one-arm panel); no alphasteer
        if "none" not in petri_cells:
            raise SystemExit(f"{target}: the view's Petri manifest lists no `none` cell — the standup's bare route is served from it")
        outside = [c for c in pol["cells"] if c not in petri_cells]
        if outside:
            raise SystemExit(f"{target}: LANE_POLICY names cells the view's Petri manifest does not list {outside} "
                             f"(manifest cells: {petri_cells}) — a policy may only select cells of its view; refusing")
        if pol["alphasteer"]:
            raise SystemExit(f"{target}: LANE_POLICY admits an alphasteer arm — this generator renders the lane's admitted objects "
                             "and prompted arms only (extend the policy handling before admitting it)")
        prompted = list(PROMPTED_CELLS) if pol["prompted"] else []
        missing_prompted = [c for c in prompted if c not in petri_cells]
        if missing_prompted:
            raise SystemExit(f"{target}: LANE_POLICY admits the prompted arms {list(PROMPTED_CELLS)} but the view's Petri manifest does not "
                             f"list {missing_prompted} (manifest cells: {petri_cells}) — regenerate the wave (make_full22_configs.py) from a "
                             "roster whose seed view lists `prompted__safety_refusal: null` and `prompted__si_constitution: null`; refusing")
        cells = ["none", *pol["cells"], *prompted]
        if len(set(cells)) != len(cells):
            raise SystemExit(f"{target}: duplicate cells in the lane policy {cells}")
        return cells, ALPHASTEER_LANE_EXCLUDED
    if is_steered_wave_target(target):
        # THE STEERED-WAVE FRONT-DOOR POLICY: served objects == measured == EXACTLY
        # the view's Petri cells, which list NO `none` — the standup serves the base route implicitly for its health checks and attestation,
        # and the config's `misalign_policy.measure_none: false` keeps it unmeasured; never an alphasteer cell; a manifest listing `none` refuses
        if not petri_cells:
            raise SystemExit(f"{target}: the steered-wave view's Petri manifest lists no cell — nothing to measure")
        if "none" in petri_cells or any(c.startswith("none__") for c in petri_cells):
            raise SystemExit(f"{target}: the steered-wave view's Petri manifest lists an unsteered leg {[c for c in petri_cells if c == 'none' or c.startswith('none__')]} "
                             "— a steered-wave view (s<N>r) measures none (D1152 / D1154); regenerate the wave (make_full22_configs.py)")
        if ALPHASTEER_CELL in petri_cells:
            raise SystemExit(f"{target}: the view's Petri manifest lists `{ALPHASTEER_CELL}` — not one of the front door's arms; refusing")
        if len(set(petri_cells)) != len(petri_cells):
            raise SystemExit(f"{target}: duplicate cells in the view's Petri manifest {petri_cells}")
        return list(petri_cells), FRONT_DOOR_STATUS
    if is_front_door(target):
        # THE FRONT-DOOR POLICY (paper-companion-repo; spec FD 5 / FD 22): served == measured == EXACTLY the view's Petri cells — `none`
        # first, the view's objects (ReLU arms included) and whatever prompted SI arms the roster lists on THIS view; nothing appended
        # (the phase views partition the ten arms), nothing dropped, never an alphasteer cell (a manifest listing one refuses by name)
        if not petri_cells or petri_cells[0] != "none":
            raise SystemExit(f"{target}: the view's Petri manifest must list `none` first (got {petri_cells[:1]}) — the standup's bare route")
        if ALPHASTEER_CELL in petri_cells:
            raise SystemExit(f"{target}: the view's Petri manifest lists `{ALPHASTEER_CELL}` — not one of the front door's arms; refusing")
        if len(set(petri_cells)) != len(petri_cells):
            raise SystemExit(f"{target}: duplicate cells in the view's Petri manifest {petri_cells}")
        return list(petri_cells), FRONT_DOOR_STATUS
    if is_allpos_panel(target):
        return ["none", ALLPOS_CELL], ALLPOS_STATUS   # single-object native standup; no prompted arm, no alphasteer
    if is_psr_panel(target):
        psr = [c for c in petri_cells if PSR_PANEL_TOKEN in c]
        if len(psr) != 3:
            raise SystemExit(f"{target}: expected the three psr_ce_all cells in the Petri manifest, found {psr}")
        return ["none", *psr], PSR_PANEL_STATUS   # no prompted arm, no alphasteer (OUT everywhere, 2026-09-17)
    cells = list(petri_cells)
    for c in PROMPTED_CELLS:
        if c not in cells:
            cells.append(c)
    if is_org_panel(target):
        pins = org_pins(target)
        if pins is None:
            return [c for c in cells if c != ALPHASTEER_CELL], ORG_STATUS
        # the PINNED organisms panel: none + the pinned objects (harness ids, the pins' order) + the two prompted arms — the
        # Petri manifest's cells are NOT consulted (they name the wave's batch-8 objects under the same ids and the
        # safety-distilled arms this panel drops); no alphasteer
        _, _, ra = _sisteer()
        pinned = ["none", *(ra.harness_cell_id(c) for c in pins["_cells"]), *PROMPTED_CELLS]
        if len(set(pinned)) != len(pinned):
            raise SystemExit(f"{target}: duplicate cells in {pinned}")
        return pinned, ORG_PINNED_STATUS
    status, _ = alphasteer_status(target)
    if status == ALPHASTEER_SERVED and ALPHASTEER_CELL not in cells:
        cells.append(ALPHASTEER_CELL)
    if status != ALPHASTEER_SERVED and ALPHASTEER_CELL in cells:
        raise SystemExit(f"{target}: the Petri manifest lists {ALPHASTEER_CELL} but the alphasteer pins say "
                         f"{status} — refusing")
    if len(set(cells)) != len(cells):
        raise SystemExit(f"{target}: duplicate cells in {cells}")
    return cells, status


# --------------------------------------------------------------------------- manifest

def _split_cells_block(text: str) -> tuple[str, list[str], str]:
    """(before, cell ids, after) around the manifest's ``  cells:`` block."""
    lines = text.split("\n")
    try:
        start = next(i for i, ln in enumerate(lines) if ln == "  cells:")
    except StopIteration:
        raise SystemExit("manifest has no `  cells:` block")
    j = start + 1
    cells = []
    while j < len(lines) and _CELL_LINE.match(lines[j]):
        cells.append(_CELL_LINE.match(lines[j]).group(1))
        j += 1
    if not cells:
        raise SystemExit("manifest `cells:` block is empty")
    return "\n".join(lines[:start]), cells, "\n".join(lines[j:])


def _replace_header(text: str, header: str) -> str:
    """Swap the leading comment block (every line up to the first non-comment line) for ``header``."""
    lines = text.split("\n")
    k = 0
    while k < len(lines) and (lines[k].startswith("#") or lines[k] == ""):
        k += 1
    return header.rstrip("\n") + "\n" + "\n".join(lines[k:])


#: the misalign audit's Petri fallback chains are ITS OWN, never the seed views' roster knobs. The source
#: Petri manifest (`petri_full22_qwen35_4b_s<s>.yaml`) now renders the battery's per-view chains
#: (`views.<view>.petri_auditor_fallback_models` / `petri_judge_fallback_models`: opus-4-8 before grok for both roles)
#: and this generator copies that manifest's text byte-for-byte -- so without this rule the knobs would leak in and RE-KEY
#: the misalign cells whose canonical rows ran under the committed manifests. The COMMITTED misalign manifests are
#: authoritative: the wave template's one-leg auditor chain and NO judge chain. Normalized by name here; `--check` holds the
#: committed bytes.
MISALIGN_AUDITOR_CHAIN: list[str] = ["openrouter/x-ai/grok-4.6"]
_CHAIN_BLOCK_RE = r"^  {key}:[^\n]*\n(?:    - [^\n]*\n)*"
#: EVERY other `*_fallback_models` block of the copied Petri manifest -- the sticky judge chain AND the per-request refusal-fallback
#: knobs (`auditor_refusal_fallback_models` / `judge_refusal_fallback_models`), and any such knob added later --
#: is stripped by PATTERN: in the misalign render they are incoherent (a Sonnet fallback for a Grok auditor, whose point is
#: refusal resistance; a judge chain misalign does not have), and the rule above makes the committed
#: misalign manifest authoritative with its OWN chains. The Petri knobs leaked once (the u0 misalign manifest drifted on 2026-10-01
#: when the u0 view gained the refusal-fallback knobs); a pattern cannot leak the next one.
_OTHER_FALLBACK_BLOCK_RE = r"^  (?!auditor_fallback_models:)\w+_fallback_models:[^\n]*\n(?:    - [^\n]*\n)*"


def scope_fallback_chains(text: str) -> str:
    """Rewrite the copied Petri manifest's chain blocks to the misalign audit's own: ``auditor_fallback_models``
    becomes MISALIGN_AUDITOR_CHAIN with no comment; EVERY other ``*_fallback_models`` block -- ``judge_fallback_models``, the
    per-request ``auditor_refusal_fallback_models`` / ``judge_refusal_fallback_models`` knobs, and any such knob added later -- is
    removed by pattern (option (A), 2026-10-01). Refuses a source without exactly one auditor chain block. Idempotent."""
    import re as _re
    n = len(_re.findall(r"^  auditor_fallback_models:", text, flags=_re.M))
    if n != 1:
        raise SystemExit(f"misalign manifest source: expected exactly one `  auditor_fallback_models:` block, found {n}")
    block = "  auditor_fallback_models:\n" + "".join(f"    - {m}\n" for m in MISALIGN_AUDITOR_CHAIN)
    text = _re.sub(_CHAIN_BLOCK_RE.format(key="auditor_fallback_models"), block.replace("\\", "\\\\"), text, count=1, flags=_re.M)
    text = _re.sub(_OTHER_FALLBACK_BLOCK_RE, "", text, flags=_re.M)      # the judge chain + every refusal-fallback knob, by pattern
    return text


def render_manifest(target: str) -> tuple[str, list[str]]:
    src = MANIFEST_DIR / f"petri_full22_{source_stem(target)}.yaml"
    if not src.is_file():
        raise SystemExit(f"missing {src} — regenerate the wave (make_full22_configs.py) first")
    text = scope_fallback_chains(src.read_text(encoding="utf-8"))     # The misalign audit's own chains, never the Petri knobs
    before, petri_cells, after = _split_cells_block(text)
    cells, status = misalign_cells(petri_cells, target)
    # 2026-09-12: no PSR cell elsewhere; the PSR panel keeps its three; the PINNED organisms panel keeps its
    # psr_ce_all cells too (an exception to the psr-out rule, by name); a LANE_POLICY `psr: include`
    # target (the seed-replicate views) keeps its view's — psr_policy is psr_out for every other target
    cells = cells if status in (PSR_PANEL_STATUS, ORG_PINNED_STATUS) else psr_policy(target, cells)
    n_audit = len(re.findall(r"^  audit: petri_full22_\S+$", before, re.M))
    if n_audit != 1:
        raise SystemExit(f"{src.name}: expected one `  audit: petri_full22_...` line, found {n_audit}")
    before = re.sub(r"^  audit: petri_full22_\S+$", f"  audit: misalign_full22_{target}", before, flags=re.M)
    prompted = [c for c in cells if c.startswith("prompted__")]
    after = re.sub(r"^(  prompted_prefix_hashes:\s+# sha256_hex16 over the EXACT SI text bytes \(H8\); ).*$",
                   lambda m: m.group(1) + (f"verified per request (H8) for the prompted cell(s) {', '.join(prompted)}" if prompted
                                           else "no prompted cell on this phase view (the roster lists none here) — carried as a harness-hash input" if status == FRONT_DOOR_STATUS
                                           else "no prompted cell in this panel (the PSR panel) — carried as a harness-hash input" if status == PSR_PANEL_STATUS
                                           else "no prompted cell in this panel (the all-positions panel) — carried as a harness-hash input" if status == ALLPOS_STATUS
                                           else "no prompted cell in this panel (the baselines panel) — carried as a harness-hash input"),
                   after, count=1, flags=re.M)
    added = [c for c in cells if c not in petri_cells]
    view = alphasteer_view(target)
    if status == BASELINE_STATUS:
        rec = baseline_pins(target)
        m_tpl = re.search(r"^  chat_template_hash: ([0-9a-f]{16})", text, re.M)
        m_rev = re.search(r"^  target_revision: (\S+)", text, re.M)
        for c in BASELINE_CELLS:
            o = rec[c]
            if not m_tpl or o["chat_template_hash"] != m_tpl.group(1):
                raise SystemExit(f"{target}: {BASELINE_PINS.name} {c} chat_template_hash {o['chat_template_hash']} != the "
                                 f"family manifest's {m_tpl.group(1) if m_tpl else None} — the object was fitted under another render")
            if not m_rev or o["resolved_revision"] != m_rev.group(1):
                raise SystemExit(f"{target}: {BASELINE_PINS.name} {c} resolved_revision {o['resolved_revision']} != the "
                                 f"family manifest's {m_rev.group(1) if m_rev else None}")
        ar, al = rec["arditi"], rec["alphasteer"]
        as_note = (f"#   * `arditi` — the Arditi refusal direction (arXiv 2406.11717; alpha {ar['alpha']} of the residual norm baked into\n"
                   f"#     the bytes, kind static_r, served layer {ar['served_layer']}, prefill-only), the arditi lane's object\n"
                   f"#     exp/objects/{ar['source']} (sha256 {ar['object_sha256'][:16]}…).\n"
                   f"#   * `alphasteer` — the AlphaSteer RE-BAKE rung {al['rung']} (lambda {al['lambda']:.4g}, selection_outcome\n"
                   f"#     `{al['selection_outcome']}`), the AlphaSteer lane's object exp/objects/{al['source']}\n"
                   f"#     (sha256 {al['object_sha256'][:16]}…) — NOT the campaign's `selected__alphasteer` the family panel serves.\n"
                   "#     Both pinned in configs/misalign_baseline_pins.json; misalign_bank_view.sh re-hashes before linking.")
    elif status == ALLPOS_STATUS:
        ap = allpos_pin()
        as_note = ("#   * NO `alphasteer` cell and NO prompted arm: the Arditi ALL-POSITIONS panel (2026-09-17) is `none`\n"
                   f"#     + `{ALLPOS_CELL}` (kind {ap.get('kind')}, served layer {ap.get('served_layer')}, dose c = {ap.get('c')} raw_coefficient — the\n"
                   "#     DECLARED OVERRIDE of the selection rule, whose output is c = 0.2; both readings live in the arms table),\n"
                   f"#     object exp/objects/{ap.get('source')} (sha256 {ap['object_sha256'][:16]}…), pinned in configs/arditi_allpos_pins.json\n"
                   f"#     qwen35_4b.{ALLPOS_STAGED_ID}; a SINGLE-OBJECT native standup (a second object beside it would be skipped by name).")
    elif status == PSR_PANEL_STATUS:
        as_note = ("#   * NO `alphasteer` cell and NO prompted arm: the PSR / ReLU panel (2026-09-17) is `none` + the\n"
                   "#     wave main view's three psr_ce_all cells (constitution, control, distilled difference — ReLU steering,\n"
                   "#     Heyman & Vandeputte 2026), served on the NATIVE path (the residual slot) and paired against the panel's\n"
                   "#     existing lens `none` (no native anchor). AlphaSteer is OUT everywhere (2026-09-17).")
    elif status == ALPHASTEER_SERVED:
        as_note = (f"#   * `alphasteer` — the AlphaSteer Class B arm (alphasteer_f22 campaign, view `{view}`, staged id\n"
                   f"#     `selected__alphasteer`; pin configs/alphasteer_f22_pins.json, status `served`), served from the\n"
                   "#     misalign bank view's `alphasteer` entry (misalign_bank_view.sh: sha256-checked against the pin).")
    elif status == ORG_PINNED_STATUS:
        rec = org_pins(target)
        m_tpl = re.search(r"^  chat_template_hash: ([0-9a-f]{16})", text, re.M)
        m_rev = re.search(r"^  target_revision: (\S+)", text, re.M)
        m_mod = re.search(r"^  target_model: (\S+)", text, re.M)
        if not m_tpl or rec["chat_template_hash"] != m_tpl.group(1):
            raise SystemExit(f"{target}: {ORG_PINS.name} chat_template_hash {rec['chat_template_hash']} != the family manifest's "
                             f"{m_tpl.group(1) if m_tpl else None} — the objects were fitted under another render")
        if not m_rev or rec["resolved_revision"] != m_rev.group(1):
            raise SystemExit(f"{target}: {ORG_PINS.name} resolved_revision {rec['resolved_revision']} != the family manifest's "
                             f"{m_rev.group(1) if m_rev else None}")
        if not m_mod or rec["model"] != m_mod.group(1):
            raise SystemExit(f"{target}: {ORG_PINS.name} model {rec['model']} != the family manifest's {m_mod.group(1) if m_mod else None}")
        for c in rec["_cells"]:
            th = rec[c].get("template_hash")
            if th is not None and th != m_tpl.group(1):
                raise SystemExit(f"{target}: {ORG_PINS.name} {c} template_hash {th} != the family manifest's {m_tpl.group(1)} — "
                                 "the object was fitted under another render")
        fam = wave_of(target)[0]
        as_note = ("#   * NO `alphasteer` cell (AlphaSteer is OUT everywhere since 2026-09-17) and NO safety-distilled arm\n"
                   "#     (safety_refusal__lora_r4 / safety_refusal__psr_kl_all: dropped by the same rule — never asked for). This is the\n"
                   "#     EM MODEL-ORGANISMS panel with PINNED objects: the rule is `prefer batch-64`, so every object cell\n"
                   f"#     is the RESEED lane's effective-batch-{rec['effective_batch']} seed-{rec['seed']} fit (fit key `{rec['fit_model_key']}`,\n"
                   f"#     {rec['fit_model_id']} @ {rec['fit_model_revision'][:12]}; seeds s2/s3 exist, s1 ONLY serves), pinned by sha256 in\n"
                   f"#     configs/misalign_org_pins.json [{fam}] and re-hashed by misalign_bank_view.sh before a view entry exists — NOT the\n"
                   "#     wave main view's objects (the same cell ids name its older batch-8 fits there). The psr_ce_all cells are IN for\n"
                   "#     this key by the 2026-09-17 rule (an exception to the 2026-09-12 psr-out rule), on the LENS path. The lane's own\n"
                   "#     quality record per object, verbatim:\n"
                   + "".join(f"#       {c:<50} {rec[c]['kind']:<18} sha {rec[c]['object_sha256'][:16]}…  {_org_quality_line(rec[c])}\n"
                             for c in rec["_cells"])
                   + "#     Organisms mode serves every non-LoRA cell also as `<em-organism>::<cell>` (the organism adapter as a vLLM\n"
                   "#     LoRA module under the object), em8q only.")
    elif status == FRONT_DOOR_STATUS:
        as_note = ("#   * NO `alphasteer` cell: the FRONT DOOR (paper-companion-repo; make_misalign_manifest.front_door_targets) serves exactly\n"
                   "#     the view's cells — the ten arms of spec FD 5 carry no AlphaSteer arm and the campaign never fitted this model; no\n"
                   "#     campaign pin is consulted.")
    elif status == ALPHASTEER_LANE_EXCLUDED:
        pol_ = lane_policy(target)
        as_note = (("#   * NO `alphasteer` cell: the LANE POLICY (make_misalign_manifest.LANE_POLICY) serves\n"
                    "#     the view's admitted steered objects + its two prompted SI arms — the campaign pins are not consulted.")
                   if pol_ is not None and pol_["prompted"] else
                   ("#   * NO `alphasteer` cell and NO `prompted__*` arm: the LANE POLICY (make_misalign_manifest.LANE_POLICY)\n"
                    "#     serves the view's admitted steered objects only — the campaign pins are not consulted."))
    elif status == ORG_STATUS:
        as_note = ("#   * NO `alphasteer` cell: AlphaSteer is OUT everywhere (since 2026-09-17), whatever the alphasteer pins record\n"
                   "#     for the family. This is an EM MODEL-ORGANISMS panel: organisms mode serves every non-LoRA cell\n"
                   "#     also as `<em-organism>::<cell>` (the organism adapter as a vLLM LoRA module under the object), em8q only.")
    elif status == ALPHASTEER_NOT_IN_CAMPAIGN:
        as_note = ("#   * NO `alphasteer` cell: configs/alphasteer_f22_roster.yaml lists no such target — the AlphaSteer\n"
                   "#     campaign never fitted this model; an explicit exclusion, never an oversight.")
    elif status == ALPHASTEER_NOT_ON_TREE:
        as_note = ("#   * NO `alphasteer` cell: the AlphaSteer campaign's roster + pins (configs/alphasteer_f22_{roster.yaml,\n"
                   "#     pins.json}) are not on this tree — no AlphaSteer object can be served from it; the cell returns\n"
                   "#     (as --check drift) when that lane lands.")
    else:
        as_note = (f"#   * NO `alphasteer` cell: configs/alphasteer_f22_pins.json records `{status}` for this target\n"
                   "#     (no admissible operating point) — an explicit exclusion, never an oversight.")
    pol = lane_policy(target)
    prompted_para = ((f"#     the view's {len(pol['cells'])} ADMITTED steered objects under the lane policy (decisions O20, 2026-09-17: the distilled PSR\n"
                      f"#     difference failed admission for every seed and is on no view): PSR cells INCLUDED (the 2026-09-12 PSR ruling was the\n"
                      f"#     em-misalign lane's; this lane's spec names the ReLU objects), "
                      + ("+ the view's TWO PROMPTED SI arms (constitution SI / safety SI; the client sends the exact SI bytes as the first\n"
                         "#     system message and the proxy hash-verifies them per request; every arm of\n"
                         "#     seed s on seed s's engine), no alphasteer." if pol["prompted"] else "no prompted arm, no alphasteer.")
                      + " SERVED != MEASURED:\n"
                      f"#     `none` is SERVED for the standup's health checks and steering-bites and NEVER measured —\n"
                      f"#     configs/misalign_full22_{target}.yaml carries `misalign_policy.measure_none: false` and the MEASURED\n"
                      f"#     cells (the admitted objects" + (" + the prompted arms" if pol["prompted"] else "") + "), misalign_proxy.sbatch builds the runner's cells.json from them alone (no `none` dir or row is\n"
                      f"#     produced), and every steered row is paired downstream against the campaign's EXISTING main-view anchor\n"
                      f"#     {pol['pairing_anchor']} through the preflight-checked sampler match (anchor_sampler_preflight.py).\n"
                      "#     The path-matched control leg `none__lens0` / `none__lora0` is DERIVED by defense_cells.enumerate_cells,\n"
                      "#     never listed, and never measured either.")
                     if pol is not None else
                     "#     the two PROMPTED SI arms (constitution SI / safety SI; the client sends the exact SI bytes as the\n"
                     "#     first system message and the proxy hash-verifies them per request, H8) and, where served, the\n"
                     f"#     AlphaSteer arm — the cells of configs/misalign_full22_{target}.yaml's explicit_cells as harness ids\n"
                     "#     (TrainTest prompted__safety_refusal <-> prompted__safety, prompted__si_constitution <->\n"
                     "#     prompted__constitution, alphasteer <-> alphasteer). The path-matched control leg `none__lens0` is\n"
                     "#     DERIVED by defense_cells.enumerate_cells, never listed."
                     if status not in (BASELINE_STATUS, PSR_PANEL_STATUS, ALLPOS_STATUS) else
                     f"#     the one object cell of configs/misalign_full22_{target}.yaml's explicit_cells as a harness id ({ALLPOS_CELL}; no\n"
                     "#     prompted cell, so the prompted_prefix_hashes below are carried as harness-hash inputs only). The path-matched\n"
                     "#     control leg (none__lora0 on the native engine) is DERIVED, never listed."
                     if status == ALLPOS_STATUS else
                     f"#     the three PSR cells of configs/misalign_full22_{target}.yaml's explicit_cells as harness ids (si_<...> <->\n"
                     "#     the TrainTest ids; no prompted cell, so the prompted_prefix_hashes below are carried as harness-hash inputs\n"
                     "#     only). The path-matched control leg (none__lens0 / none__lora0 by engine) is DERIVED, never listed."
                     if status == PSR_PANEL_STATUS else
                     f"#     the cells of configs/misalign_full22_{target}.yaml's explicit_cells as harness ids (both Class B, the\n"
                     "#     identity mapping; no prompted cell, so the prompted_prefix_hashes below are carried as harness-hash inputs\n"
                     "#     only). The path-matched control leg `none__lens0` is DERIVED by defense_cells.enumerate_cells, never listed.")
    cells_note = ((f"#   * `cells` = `none` + the view's {len(pol['cells'])} admitted objects (LANE_POLICY[{target!r}]; PSR included"
                   + (f") + the two prompted SI arms {', '.join(f'`{c}`' for c in PROMPTED_CELLS)}: SERVED for the\n"
                      "#     standup; MEASURED = the admitted objects + the prompted arms (see below). Cells:" if pol["prompted"] else
                      ", nothing appended): SERVED for the\n"
                      "#     standup; MEASURED = the admitted objects only (see below). Cells:"))
                  if pol is not None else
                  f"#   * `cells` = the Petri manifest's cells + {', '.join(f'`{c}`' for c in added) if added else '(nothing added)'}:"
                  if status != BASELINE_STATUS else
                  "#   * `cells` = [none, arditi, alphasteer] — NOT the Petri manifest's: the PUBLISHED BASELINES panel (the\n"
                  "#     2026-09-13 rule) serves the unsteered anchor and the two pinned baseline objects only; the wave's\n"
                  "#     objects and the prompted arms are measured in the family's own panel (misalign_full22_<family>). Cells:")
    if status == FRONT_DOOR_STATUS:
        rt_, view_ = wave_of(target)
        n_obj = sum(1 for c in cells if c != "none" and not c.startswith("prompted__"))
        n_pr = sum(1 for c in cells if c.startswith("prompted__"))
        cells_note = ((f"#   * `cells` = EXACTLY the Petri manifest's cells of the `{view_}` STEERED-WAVE phase view of {rt_} ({n_obj} object cell(s) + {n_pr} prompted\n"
                       "#     SI arm(s), NO `none`: a steered-wave view measures no unsteered leg). The standup\n"
                       "#     serves the base route implicitly for its health checks and attestation; `misalign_policy.measure_none: false` (the config)\n"
                       "#     keeps it and its derived controls twin UNMEASURED. Cells:")
                      if is_steered_wave_target(target) else
                      (f"#   * `cells` = EXACTLY the Petri manifest's cells of the `{view_}` phase view of {rt_} (none + {n_obj} object cell(s) + {n_pr} prompted\n"
                       "#     SI arm(s), nothing appended, nothing dropped): the FRONT-DOOR POLICY (paper-companion-repo, spec FD 5 / FD 22 — the full\n"
                       "#     battery on every arm the view carries, ReLU arms included; every served leg is measured, `none` included). Cells:"))
        prompted_para = ("#     the view's cells of configs/misalign_full22_{0}.yaml's explicit_cells as harness ids (TrainTest `<si>__<arm>` <-> `si_<si>__<arm>`,\n"
                         "#     prompted__safety_refusal <-> prompted__safety, prompted__si_constitution <-> prompted__constitution; a Class B id the\n"
                         "#     identity). A prompted arm, where the view lists one, is the SI text as the first system message, hash-verified per\n"
                         "#     request (H8). The path-matched control leg (`none__lora0` on the native engine) is DERIVED by\n"
                         "#     defense_cells.enumerate_cells, never listed.").format(target)
    if status == ORG_PINNED_STATUS:
        rec = org_pins(target)
        cells_note = (f"#   * `cells` = none + the {len(rec['_cells'])} PINNED objects of configs/misalign_org_pins.json [{wave_of(target)[0]}] + the two\n"
                      "#     prompted arms — NOT the Petri manifest's cells: those name the wave main view's objects (the older batch-8 fits,\n"
                      "#     under the same cell ids) and its safety-distilled arms, which this panel drops (2026-09-17). Cells:")
        prompted_para = ("#     the two PROMPTED SI arms (constitution SI / safety SI; the client sends the exact SI bytes as the first\n"
                         "#     system message and the proxy hash-verifies them per request, H8) and the pinned objects — the cells of\n"
                         f"#     configs/misalign_full22_{target}.yaml's explicit_cells as harness ids (TrainTest `<si>__<arm>` <-> `si_<si>__<arm>`,\n"
                         "#     prompted__safety_refusal <-> prompted__safety, prompted__si_constitution <-> prompted__constitution). The\n"
                         "#     path-matched control leg `none__lens0` is DERIVED by defense_cells.enumerate_cells, never listed.")
    header = f"""# Misalignment evals (agentic-misalignment leaking + murder; Betley 8Q) — em-misalign, target {target}.
# GENERATED by scripts/sdfbase/full22/misalign/make_misalign_manifest.py from
# configs/capeval/manifests/petri_full22_{source_stem(target)}.yaml — never hand-edited (the generator's --check pins
# byte-identity). Every field is the Petri manifest's (same target pins, serving flags, chat-template
# hash, prompted prefix hashes, bloom options: scripts/sdfbase/mt_serving_up.py validates it as a Petri
# manifest and refuses a drifted pin exactly as the wave does) EXCEPT:
#   * `audit: misalign_full22_{target}`;
{cells_note}
{prompted_para}
{as_note}
#
# The Petri audit fields below (behaviors, scenario bundle, auditor / judge models, bloom_options) are
# carried UNCHANGED as harness-hash inputs; this eval runs no Petri audit — the drivers are
# sisteer.capeval.misalign.runner (agentic + em8q families) against the standup's loopback proxy.
"""
    out = _replace_header(before, header) + "\n  cells:\n" + "\n".join(f"    - {c}" for c in cells) + "\n" + after
    tpl_sha, tpl_path = chat_template_pin(target)
    if tpl_sha is not None:
        n = len(re.findall(r"^  chat_template_hash: [0-9a-f]{16}.*$", out, flags=re.M))
        if n != 1:
            raise SystemExit(f"{src.name}: expected one chat_template_hash line, found {n}")
        rel = tpl_path.relative_to(REPO).as_posix()
        out = re.sub(r"^  chat_template_hash: ([0-9a-f]{16}).*$",
                     lambda m: (f"  chat_template_hash: {tpl_sha}   # sha256_hex16 of {rel} — Meta's STOCK Llama-3.1 template, "
                                f"served by --chat-template; NOT the pinned tokenizer's minimal template ({m.group(1)})"),
                     out, count=1, flags=re.M)
    return out, cells


# --------------------------------------------------------------------------- config

def _sub_line(body: str, key_regex: str, new_line: str, *, count: int = 1) -> str:
    n = len(re.findall(key_regex, body, flags=re.M))
    if n != count:
        raise SystemExit(f"expected {count} line(s) matching {key_regex!r}, found {n}")
    return re.sub(key_regex, new_line, body, count=count, flags=re.M)


def render_config(target: str, harness_cells: list[str]) -> tuple[str, list[str]]:
    _, _, ra = _sisteer()
    src = REPO / "configs" / f"sdfbase_full22_{source_stem(target)}.yaml"
    if not src.is_file():
        raise SystemExit(f"missing {src} — regenerate the wave (make_full22_configs.py) first")
    text = src.read_text(encoding="utf-8")
    marker = "# -- the explicit cell enumeration (ordered; the registry is EXACTLY this) ---\nexplicit_cells:\n"
    if marker not in text:
        raise SystemExit(f"{src.name}: no explicit_cells marker block")
    tail = text[text.index(marker) + len(marker):]
    # the wave's explicit_cells list ends at the first blank line. The sdfbase-llama8b configs carry a TRAILER
    # between the list and that blank line (a comment + the target's `xstest_subset` key, roster
    # targets.<t>.xstest_subset): carried verbatim after the new list, so the drivers still read it.
    lines = tail.split("\n")
    k = 0
    wave_cells = []
    while k < len(lines) and lines[k].startswith("- "):
        wave_cells.append(lines[k][2:].strip())
        k += 1
    t0 = k
    while k < len(lines) and lines[k] != "":
        if lines[k].startswith("- "):
            raise SystemExit(f"{src.name}: malformed explicit_cells block (a cell after the trailer)")
        k += 1
    if not wave_cells or k >= len(lines):
        raise SystemExit(f"{src.name}: malformed explicit_cells block")
    trailer = lines[t0:k]
    body = "\n".join(lines[k + 1:])          # from `# -- everything below:` on
    _, view = wave_of(target)
    tt_cells = [ra.traintest_cell_id(c) for c in harness_cells]
    if is_steered_wave_target(target):
        if "none" in tt_cells or any(c.startswith("none__") for c in tt_cells):   # s3r-fix: a steered-wave config lists no unsteered leg
            raise SystemExit(f"{target}: a steered-wave misalign config lists an unsteered leg in {tt_cells} — none is served implicitly, never listed (D1152 / D1154)")
    elif tt_cells[0] != "none":
        raise SystemExit(f"{target}: the first cell must be `none`, got {tt_cells[0]}")
    opins = org_pins(target)
    missing_wave = [c for c in wave_cells if c not in tt_cells and PSR_OUT_TOKEN not in c]   # PSR cells are dropped by ruling
    if is_baseline(target) or is_psr_panel(target) or is_allpos_panel(target):
        missing_wave = []   # the baselines panel serves NONE of the wave's objects by design; the PSR panel serves only its three
    elif opins is not None:
        # the PINNED organisms panel drops EXACTLY the two safety-distilled wave cells and serves every
        # other wave cell — psr_ce_all included, so the PSR exemption above does not apply to it; a safety cell that crept
        # back in refuses too
        missing_wave = [c for c in wave_cells if c not in tt_cells and c not in ORG_DROPPED_WAVE_CELLS]
        undropped = [c for c in ORG_DROPPED_WAVE_CELLS if c in tt_cells]
        if undropped:
            raise SystemExit(f"{target}: the pinned organisms panel lists {undropped} — the safety-distilled arms are OUT "
                             "(the user, 2026-09-17)")
    pol = lane_policy(target)
    if pol is not None:
        # the lane serves the view's OBJECTS (PSR included) and drops its prompted baselines by policy: only a dropped OBJECT
        # is a refusal here
        missing_wave = [c for c in wave_cells if c not in tt_cells and not c.startswith("prompted__")]
    if is_front_door(target) and tt_cells != wave_cells:
        raise SystemExit(f"{target}: the front door serves EXACTLY the wave's {view} view cells; misalign {tt_cells} != wave {wave_cells}")
    if missing_wave:
        raise SystemExit(f"{target}: the misalign cells drop wave cells {missing_wave} — the eval serves a "
                         f"SUPERSET of the wave's {view} view")
    body = _sub_line(body, r"^defenses_dir: .*$", f"defenses_dir: out/misalign/{target}/cells_view")
    body = _sub_line(body, r"^out_dir: .*$", f"out_dir: out/misalign_full22_{target}")
    body = _sub_line(body, r"^private_run: .*$", f"private_run: misalign_full22_{target}")
    added = [c for c in tt_cells if c not in wave_cells]
    status, rec = ((PSR_PANEL_STATUS, {}) if is_psr_panel(target) else (ALLPOS_STATUS, allpos_pin()) if is_allpos_panel(target)
                   else (ORG_PINNED_STATUS, opins) if opins is not None
                   else (ORG_STATUS, {}) if is_org_panel(target) else alphasteer_status(target))
    if status == ALLPOS_STATUS:
        as_line = (f"#       {ALLPOS_CELL:<32} <- exp/objects/{rec.get('source')} (object_sha256 {rec['object_sha256']}, kind {rec.get('kind')},\n"
                   f"#       served layer {rec.get('served_layer')}, c = {rec.get('c')} raw_coefficient — the declared override; the rule's output is 0.2)\n"
                   "#     Pin: configs/arditi_allpos_pins.json. No alphasteer cell (AlphaSteer is out everywhere since 2026-09-17).")
    elif status == PSR_PANEL_STATUS:
        as_line = "#       (no alphasteer cell: AlphaSteer is OUT everywhere since 2026-09-17)"
    elif status == ORG_PINNED_STATUS:
        as_line = ("\n".join(f"#       {c:<50} <- exp/objects/{rec[c]['source']}\n"
                             f"#       {'':<50}    (object_sha256 {rec[c]['object_sha256']}, kind {rec[c]['kind']},\n"
                             f"#       {'':<50}    sidecar {rec[c]['identity_sidecar']}; {_org_quality_line(rec[c])})"
                             for c in rec["_cells"])
                   + f"\n#     Pins: configs/misalign_org_pins.json [{wave_of(target)[0]}] — the reseed lane's fit key `{rec['fit_model_key']}`\n"
                   f"#     (effective_batch {rec['effective_batch']}, seed {rec['seed']}; {rec['fit_model_id']} @ {rec['fit_model_revision'][:12]}), READ-ONLY there; seeds\n"
                   "#     s2/s3 exist, s1 ONLY serves. No alphasteer cell (AlphaSteer is OUT everywhere since 2026-09-17) and no\n"
                   "#     safety-distilled cell (dropped by the same ruling).")
    elif status == ORG_STATUS:
        as_line = "#       (no alphasteer cell: AlphaSteer is OUT everywhere since 2026-09-17; the organisms panel)"
    elif status == BASELINE_STATUS:
        ar = baseline_pins(target)["arditi"]
        as_line = (f"#       arditi                           <- exp/objects/{ar['source']} (object_sha256 {ar['object_sha256']}, "
                   f"alpha {ar['alpha']} residual_norm_fraction, static_r, prefill)\n"
                   f"#       {ALPHASTEER_CELL:<32} <- exp/objects/{rec['source']} (object_sha256 {rec['object_sha256']}, "
                   f"RE-BAKE rung {rec['rung']}, lambda {rec.get('lambda')}, selection_outcome {rec['selection_outcome']})\n"
                   "#     Pins: configs/misalign_baseline_pins.json. The family's own panel serves the campaign's original\n"
                   "#     `selected__alphasteer` under the same cell id — a DIFFERENT object; the two results roots never mix.")
    elif status == ALPHASTEER_SERVED:
        as_line = (f"#       {ALPHASTEER_CELL:<32} <- alphasteer_f22 bank: <target>/cells/{ALPHASTEER_STAGED_ID} "
                   f"(object_sha256 {rec['object_sha256']}, lambda {rec.get('lambda')})")
    elif status == FRONT_DOOR_STATUS:
        as_line = "\n".join(_lane_pin_lines(target, tt_cells)) + "\n#       (no alphasteer cell: the front door serves exactly the view's cells — spec FD 5's arms carry none)"
    elif status == ALPHASTEER_LANE_EXCLUDED:
        as_line = "\n".join(_lane_pin_lines(target, tt_cells)) + ("\n#       (no alphasteer cell: the lane policy serves the admitted objects + the two prompted arms)"
                                                                  if lane_policy(target)["prompted"] else
                                                                  "\n#       (no alphasteer cell, no prompted arm: the lane policy serves the admitted objects only)")
    elif status == ALPHASTEER_NOT_IN_CAMPAIGN:
        as_line = "#       (no alphasteer cell: the AlphaSteer campaign's roster lists no such target)"
    elif status == ALPHASTEER_NOT_ON_TREE:
        as_line = "#       (no alphasteer cell: the AlphaSteer campaign's roster + pins are not on this tree)"
    else:
        as_line = f"#       (no alphasteer cell: alphasteer_f22_pins.json status `{status}`)"
    prompted_cfg = ((f"#     LANE POLICY (qwen-base-seed-replicates): the {len(pol['cells'])} admitted objects"
                     + (" + the two prompted SI arms (the SI text as the system prompt, no object, no view entry; the proxy verifies the\n"
                        "#     exact SI bytes per request against the manifest's prompted_prefix_hashes)" if pol["prompted"] else "")
                     + " are SERVED beside `none` and are the MEASURED set\n"
                     f"#     (`misalign_policy` below: measure_none false — `none` is served for the standup's health checks and steering-\n"
                     f"#     bites only; the runner's cells.json carries them; pairing anchor {pol['pairing_anchor']}). Each object cell's\n"
                     "#     staged id and object sha256 (configs/sdfbase_full22_roster.yaml + configs/sdfbase_full22_pins.json):")
                    if pol is not None else
                    "#     The two prompted arms serve the SI text as the system prompt (no object, no view entry; the proxy\n"
                    "#     verifies the exact SI bytes per request against the manifest's prompted_prefix_hashes, H8);\n"
                    "#     `traintest_cell_source` derives the SI vocabulary from these ids, which is what makes\n"
                    "#     safety_refusal / si_constitution enumerable here. The AlphaSteer arm (Class B, methods.ARMS\n"
                    "#     `alphasteer`) is served from the misalign bank view's `alphasteer` entry:"
                    if not (is_baseline(target) or is_psr_panel(target) or is_allpos_panel(target)) else
                    "#     The one object (the all-positions Arditi direction, kind vector_all) is served from the misalign bank view on the\n"
                    "#     NATIVE path (the residual slot, dose c raw_coefficient); no prompted cell, so no SI vocabulary is enumerated here:"
                    if is_allpos_panel(target) else
                    "#     The three PSR (ReLU steering) objects are served from the misalign bank view on the NATIVE path (the residual\n"
                    "#     slot, dose 1.0, all positions); no prompted cell, so no SI vocabulary is enumerated here:"
                    if is_psr_panel(target) else
                    "#     Both are Class B objects served from the misalign bank view (`arditi`: defense_cells.SERVED_ONLY_CLASS_B_IDS,\n"
                    "#     kind static_r; `alphasteer`: methods.ARMS); no prompted cell, so no SI vocabulary is enumerated here:")
    view_cfg = ((f"#   * `defenses_dir` names the MISALIGN bank view $SHARED/exp/objects/misalign/{target}/cells_view\n"
                 f"#     (misalign_bank_view.sh: the wave's {view} view entries — the admitted objects, PSR included; no alphasteer entry by\n"
                 "#     policy" + ("; a prompted arm has no bank entry" if pol["prompted"] else "") + "); on the cluster it is the TT_TWO_SI_DEFENSES_DIR export the misalign_proxy.sbatch job sets.")
                if pol is not None else
                f"#   * `defenses_dir` names the MISALIGN bank view $SHARED/exp/objects/misalign/{target}/cells_view\n"
                f"#     (misalign_bank_view.sh: the wave's {view} view entries + `alphasteer`, sha256-checked against\n"
                "#     configs/alphasteer_f22_pins.json); on the cluster it is the TT_TWO_SI_DEFENSES_DIR export the\n"
                "#     misalign_proxy.sbatch job sets."
                if not (is_baseline(target) or is_psr_panel(target) or is_allpos_panel(target)) else
                f"#   * `defenses_dir` names the MISALIGN ALL-POSITIONS view $SHARED/exp/objects/misalign/{target}/cells_view (one entry,\n"
                "#     arditi_allpos, a pin-form view-manifest entry pointing at configs/arditi_allpos_pins.json; built by hand for\n"
                "#     Modal). em8q is not run for this panel (agentic only)."
                if is_allpos_panel(target) else
                f"#   * `defenses_dir` names the MISALIGN PSR view $SHARED/exp/objects/misalign/{target}/cells_view (misalign_bank_view.sh\n"
                "#     qwen35_4b_psr: the wave main view's three psr_ce_all entries); on Modal the launcher points at the family's staged\n"
                "#     main view, a superset. em8q is not run for this panel (agentic only)."
                if is_psr_panel(target) else
                f"#   * `defenses_dir` names the MISALIGN BASELINES view $SHARED/exp/objects/misalign/{target}/cells_view\n"
                "#     (misalign_bank_view.sh <family>_bl: exactly the two pinned entries, each sha256-checked against\n"
                "#     configs/misalign_baseline_pins.json — no wave entry); on the cluster it is the TT_TWO_SI_DEFENSES_DIR\n"
                "#     export the misalign_proxy.sbatch job sets. em8q is not run for this panel (agentic only).")
    cells_line = ('[none, arditi, alphasteer] — the PUBLISHED BASELINES panel: no wave object, no prompted arm; the cells:' if is_baseline(target)
                  else '[none + the three psr_ce_all cells] — the PSR / ReLU panel (native path, no prompted arm, no AlphaSteer); the cells:' if is_psr_panel(target)
                  else '[none, arditi_allpos] — the Arditi ALL-POSITIONS panel (native single-object standup, no prompted arm, no AlphaSteer); the cells:' if is_allpos_panel(target)
                  else f"the wave's `{view}` view cells + the cells this eval ADDS:")
    if status == FRONT_DOOR_STATUS:
        cells_line = (f"EXACTLY the wave's `{view}` phase-view cells (the FRONT-DOOR POLICY, paper-companion-repo spec FD 5 / FD 22: nothing appended,\n"
                      "#     nothing dropped, ReLU arms included, no AlphaSteer arm); cells this eval ADDS:")
        prompted_cfg = (("#     SERVED != MEASURED: a steered-wave view lists NO `none`; the standup\n"
                         "#     serves the base route implicitly and `misalign_policy.measure_none: false` below keeps it unmeasured — the measured set is\n"
                         "#     exactly the view's cells, paired against the u0 wave's unsteered store. A prompted arm, where the view lists one,\n")
                        if is_steered_wave_target(target) else
                        ("#     Every served leg is MEASURED, `none` included (the u0 view IS the unsteered measurement; a steered view's `none` is its\n"
                         "#     same-standup anchor) — today's default rule, so no `misalign_policy` block. A prompted arm, where the view lists one,\n"))
        prompted_cfg += (
                        "#     serves the SI text as the system prompt (no object, no view entry; the proxy verifies the exact SI bytes per request\n"
                        "#     against the manifest's prompted_prefix_hashes, H8). Each object cell's staged id and object sha256\n"
                        "#     (configs/sdfbase_full22_roster.yaml + configs/sdfbase_full22_pins.json):")
        view_cfg = (f"#   * `defenses_dir` names the MISALIGN bank view $SHARED/exp/objects/misalign/{target}/cells_view\n"
                    f"#     (misalign_bank_view.sh: the wave's {view} view entries, ReLU objects included; no alphasteer entry; a prompted arm has no\n"
                    "#     bank entry); on the cluster it is the TT_TWO_SI_DEFENSES_DIR export the misalign_proxy.sbatch job sets.")
    if status == ORG_PINNED_STATUS:
        cells_line = (f"[none + the {len(rec['_cells'])} PINNED reseed objects + the two prompted arms] — the EM MODEL-ORGANISMS panel (the\n"
                      f"#     2026-09-17 rule: prefer batch-64). The object cells are configs/misalign_org_pins.json [{wave_of(target)[0]}]'s: the reseed lane's\n"
                      f"#     effective-batch-{rec['effective_batch']} seed-{rec['seed']} fits — the SAME cell ids as the wave's `{view}` view but DIFFERENT objects (the wave's\n"
                      "#     are the older batch-8 fits); the wave's safety-distilled arms are dropped and no AlphaSteer cell exists (since\n"
                      "#     2026-09-17). Cells the wave's view does not list:")
        prompted_cfg = ("#     The two prompted arms serve the SI text as the system prompt (no object, no view entry; the proxy verifies the\n"
                        "#     exact SI bytes per request against the manifest's prompted_prefix_hashes, H8); `traintest_cell_source` derives\n"
                        "#     the SI vocabulary from these ids, which is what makes safety_refusal / si_constitution enumerable here. The\n"
                        f"#     {len(rec['_cells'])} object cells are the reseed lane's PINNED effective-batch-{rec['effective_batch']} seed-{rec['seed']} objects, served on the LENS\n"
                        "#     path from the misalign bank view's pin-verified entries (organisms mode overlays each as `<em-organism>::<cell>`):")
        view_cfg = (f"#   * `defenses_dir` names the MISALIGN ORGANISMS view $SHARED/exp/objects/misalign/{target}/cells_view\n"
                    f"#     (misalign_bank_view.sh {target}: exactly the {len(rec['_cells'])} pinned entries, each sha256-checked against\n"
                    "#     configs/misalign_org_pins.json and linked READ-ONLY into the reseed lane's store — no wave entry, no alphasteer;\n"
                    "#     its VIEW_MANIFEST.json carries pin-form entries); on the cluster it is the TT_TWO_SI_DEFENSES_DIR export the\n"
                    "#     misalign_proxy.sbatch job sets. em8q only (organisms mode skips the agentic family).")
    header = f"""# misalign_full22_{target}.yaml — the misalignment evals ({'agentic-misalignment leaking + murder — the PUBLISHED BASELINES panel' if is_baseline(target) else 'agentic-misalignment leaking + murder; Betley 8Q'})
# over the full22 wave's served {'model' if is_baseline(target) else 'cells'}, target {target}. GENERATED by
# scripts/sdfbase/full22/misalign/make_misalign_manifest.py from configs/sdfbase_full22_{source_stem(target)}.yaml —
# never hand-edited (--check pins byte-identity). Every key below is that file's, byte-for-byte as
# data, EXCEPT —
#
#   * `explicit_cells` = {cells_line}
{chr(10).join(f'#       {c}' for c in added) if added else '#       (none)'}
{prompted_cfg}
{as_line}
{view_cfg}
#   * out_dir / private_run get the `misalign_full22_{target}` namespace.
#
# Serving: scripts/sdfbase/mt_serving_up.py --mode petri (the certified 16384 x budget-S Petri geometry)
# with configs/capeval/manifests/misalign_full22_{target}.yaml; drivers: sisteer.capeval.misalign.runner
# on the node against loopback (scripts/sdfbase/full22/misalign/misalign_proxy.sbatch).
#
# serving target: {_pin(text, 'model')} @ {_pin(text, 'model_revision')}

# -- the explicit cell enumeration (ordered; the registry is EXACTLY this) ---
explicit_cells:
"""
    out = header + "".join(f"- {c}\n" for c in tt_cells) + "".join(ln + "\n" for ln in trailer) + _policy_block(target, tt_cells) + "\n" + body
    return out, tt_cells


def _policy_block(target: str, tt_cells: list[str]) -> str:
    """The `misalign_policy:` block a LANE_POLICY config carries (SERVED != MEASURED; read by misalign_proxy.sbatch through
    misalign_cells.py): the measured cells as TrainTest ids (exactly the served cells minus `none`), measure_none, the
    pairing anchor. Empty for every other target (byte-identical configs)."""
    pol = lane_policy(target)
    if pol is None and is_steered_wave_target(target):
        # SERVED != MEASURED for a steered-wave front-door view — the standup serves `none` implicitly (health checks,
        # attestation; mt_serving_up select_cells) and misalign_cells.py reads this block to leave the base leg and its derived controls twin
        # UNMEASURED: the measured set is exactly the view's cells; the pairing anchor is the u0 wave's own none store
        rt, view = wave_of(target)
        return "\n".join(["", "# -- the STEERED-WAVE policy: SERVED != MEASURED. The standup serves the base",
                          "#    route implicitly (never listed above); misalign_proxy.sbatch builds the runner's cells.json from the served enumeration MINUS",
                          "#    `none` / its derived controls twin; the pairing anchor is the u0 wave's unsteered store, never a same-standup none --",
                          "misalign_policy:", "  lane: front_door_steered_wave", f"  view: {view}", "  measure_none: false", "  measured_cells:"]
                         + [f"  - {c}" for c in tt_cells] + [f"  pairing_anchor: misalign_full22_{rt}_u0/agentic/none"]) + "\n"
    if pol is None:
        return ""
    measured = [c for c in tt_cells if c != "none"]
    lines = ["", "# -- the LANE POLICY (qwen-base-seed-replicates): SERVED != MEASURED. misalign_proxy.sbatch builds the runner's",
             "#    cells.json from the standup's served enumeration MINUS `none` / its derived controls twin when measure_none is false;",
             "#    the pairing anchor is the campaign's EXISTING main-view `none` store (no new unsteered measurement in this lane) --",
             "misalign_policy:", f"  lane: {pol['lane']}", f"  view: {pol['view']}",
             f"  measure_none: {'true' if pol['measure_none'] else 'false'}",
             "  measured_cells:"] + [f"  - {c}" for c in measured] + [
             f"  psr: {pol['psr']}", f"  prompted: {'true' if pol['prompted'] else 'false'}",
             f"  alphasteer: {'true' if pol['alphasteer'] else 'false'}",
             f"  pairing_anchor: {pol['pairing_anchor']}"]
    if pol["lane"] == "paired":
        # the twin marker: a paired view RE-SERVES its seed view's r=4 companion and LoRA difference — the
        # same staged ids, a different standup — so a run of those cells under s<s>p is a STANDUP replicate of the s<s> run, never
        # a seed replicate; the reader keys on object sha and must not pool them. Recorded here from the roster's `paired_of`.
        seed = paired_of(pol["view"])
        shared = paired_reserved_cells(pol["view"])
        reserved = [c for c in measured if c in shared]          # TrainTest ids, as `measured_cells` above
        assert reserved, (target, "the paired view's re-served cells are not among its measured cells")
        lines += ["  # the twin: these measured cells are RE-SERVED from the seed view (same staged ids; a standup replicate, never a seed replicate)",
                  f"  reserves_from: {seed}", "  reserved_cells:"] + [f"  - {c}" for c in reserved]
    return "\n".join(lines) + "\n"


def _lane_pin_lines(target: str, tt_cells: list[str]) -> list[str]:
    """One header line per measured cell: `<cell> <- <staged id> (object_sha256 <sha>)` from the roster view + the pins,
    `(staging pending)` for a view / pin not yet on the tree (the objects are staged later and this is regenerated); a malformed pin refuses."""
    rt, view = wave_of(target)
    staged: dict = {}
    if ROSTER.is_file():
        roster = yaml.safe_load(ROSTER.read_text(encoding="utf-8")) or {}
        staged = dict((((roster.get("targets") or {}).get(rt) or {}).get("views") or {}).get(view, {}).get("cells") or {})
    pins: dict = {}
    if PINS.is_file():
        pins = dict((json.loads(PINS.read_text(encoding="utf-8")) or {}).get(rt) or {})
    out = []
    for c in tt_cells:
        if c == "none":
            continue
        if c.startswith("prompted__"):
            out.append(f"#       {c:<48} <- (prompted baseline: the SI as the system prompt; no object, no pin — H8-verified per request)")
            continue
        sid = staged.get(c)
        if sid is None:
            out.append(f"#       {c:<48} <- (staging pending: the roster's {rt}/{view} view does not list it yet — W5)")
            continue
        rec = pins.get(sid) if isinstance(sid, str) else None
        if rec is None:
            out.append(f"#       {c:<48} <- {sid} (pin pending: not in sdfbase_full22_pins.json yet — W5)")
            continue
        sha = str(rec.get("object_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise SystemExit(f"{target}: {PINS.name} {rt}/{sid} object_sha256 {sha!r} is not 64 hex — refusing")
        out.append(f"#       {c:<48} <- {sid} (object_sha256 {sha})")
    return out


def _pin(text: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}: (.*)$", text, re.M)
    if not m:
        raise SystemExit(f"config has no top-level {key!r}")
    return m.group(1).rstrip()


# --------------------------------------------------------------------------- validation

def validate(target: str, manifest_text: str, config_text: str, tmp_dir: Path) -> None:
    """The standup's own checks, before a byte lands: the manifest loads as a Petri spec, every cell id
    parses, the manifest/config cell sets are one enumeration, and the target pins agree."""
    cm, dc, ra = _sisteer()
    mp = tmp_dir / f"misalign_full22_{target}.yaml"
    mp.write_text(manifest_text, encoding="utf-8")
    spec = cm.load_petri_spec(mp)
    if spec.audit != f"misalign_full22_{target}":
        raise SystemExit(f"{target}: audit is {spec.audit!r}")
    for c in spec.cells:
        dc.parse_cell_id(c)
    dc.validate_petri_spec_for_enumeration(spec)
    cfg = yaml.safe_load(config_text)
    want = list(cfg["explicit_cells"])
    tt_of = {c: ra.traintest_cell_id(c) for c in spec.cells}
    missing = [c for c, tt in tt_of.items() if tt not in want]
    unserved = [c for c in want if c not in tt_of.values()]
    if missing or unserved:
        raise SystemExit(f"{target}: manifest/config mismatch: manifest cells not in the config {missing}; "
                         f"config cells the manifest does not serve {unserved}")
    if (spec.target_model, spec.target_revision) != (cfg["model"], cfg["model_revision"]):
        raise SystemExit(f"{target}: manifest pins {spec.target_model}@{spec.target_revision} but the config "
                         f"serves {cfg['model']}@{cfg['model_revision']}")
    for si in ("safety", "constitution"):
        if si not in spec.prompted_prefix_hashes:
            raise SystemExit(f"{target}: manifest lacks prompted_prefix_hashes[{si!r}]")


def wave_sources_present(target: str) -> bool:
    """Whether the wave's two GENERATED inputs for ``target`` exist on this tree. A target whose wave has
    not landed on this tree is SKIPPED with a note — an absent source is a different state from a drifted
    output, and only the latter is a --check failure."""
    return ((MANIFEST_DIR / f"petri_full22_{source_stem(target)}.yaml").is_file()
            and (REPO / "configs" / f"sdfbase_full22_{source_stem(target)}.yaml").is_file())


def lane_cells_pending(target: str) -> "str | None":
    """A LANE_POLICY target whose landed wave pair lists NONE of the policy's cells yet is a THIRD state beside an absent
    wave and a drifted output: the seed-replicate views land with every object `pending_fit` and a Petri manifest of
    `none` alone until the fits are admitted and staged into the view, so their misalign pair cannot be rendered yet —
    SKIPPED with a note (the pair is generated once the admitted objects are staged), never refused and never a --check failure. A
    view listing SOME of the policy's cells is NOT pending: it renders, and misalign_cells refuses the missing ones (never
    a guess). Returns the note, or None when the target is not pending."""
    pol = lane_policy(target)
    if pol is None:
        return None
    src = MANIFEST_DIR / f"petri_full22_{source_stem(target)}.yaml"
    _before, petri_cells, _after = _split_cells_block(src.read_text(encoding="utf-8"))
    if any(c in petri_cells for c in pol["cells"]):
        return None
    return (f"its wave's Petri manifest lists none of the lane policy's cells yet (cells: {petri_cells}; every object "
            "pending_fit) — the misalign pair is rendered once the admitted objects are staged into the view")


def present_targets() -> tuple[str, ...]:
    return tuple(t for t in TARGETS if wave_sources_present(t) and lane_cells_pending(t) is None)


def outputs() -> list[tuple[Path, str]]:
    import tempfile
    out = []
    with tempfile.TemporaryDirectory() as td:
        for target in TARGETS:
            if not wave_sources_present(target):
                print(f"skip {target}: its wave's configs are not on this tree (the wave has not landed here)")
                continue
            pending = lane_cells_pending(target)
            if pending:
                print(f"skip {target}: {pending}")
                continue
            m_text, cells = render_manifest(target)
            c_text, _ = render_config(target, cells)
            validate(target, m_text, c_text, Path(td))
            out.append((MANIFEST_DIR / f"misalign_full22_{target}.yaml", m_text))
            out.append((REPO / "configs" / f"misalign_full22_{target}.yaml", c_text))
    return out


def target_kind(target: str) -> str:
    """The policy family a target key renders under (what ``--resolve`` prints beside its (roster target, view))."""
    if is_front_door(target):
        return FRONT_DOOR_STATUS
    pol = lane_policy(target)
    if pol is not None:
        return f"lane:{pol['lane']}"
    if is_baseline(target):
        return "baselines_panel"
    if is_org_panel(target):
        return "organisms_panel"
    if is_psr_panel(target):
        return PSR_PANEL_STATUS
    if is_allpos_panel(target):
        return ALLPOS_STATUS
    return "campaign"


def resolve(target: str) -> tuple[str, str, str]:
    """``(roster target, view, kind)`` for a target key of TARGETS; an unknown key refuses by name listing every admissible one (the
    one table misalign_proxy.sbatch reads instead of a second copy of it)."""
    if target not in TARGETS:
        raise SystemExit(f"unknown misalign target {target!r}; known: {', '.join(TARGETS)}")
    rt, view = wave_of(target)
    return rt, view, target_kind(target)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if any committed file drifts; write nothing")
    ap.add_argument("--resolve", metavar="TARGET", default=None,
                    help="print `<roster target> <view> <kind>` for a target key (front-door keys derive from the models table) and exit; "
                         "an unknown key exits 2 naming every admissible one")
    args = ap.parse_args(argv)
    if args.resolve is not None:
        try:
            rt, view, kind = resolve(args.resolve)
        except SystemExit as e:
            print(str(e), file=sys.stderr)
            return 2
        print(f"{rt} {view} {kind}")
        return 0
    rc = 0
    for dst, text in outputs():
        if args.check:
            if not dst.is_file() or dst.read_text(encoding="utf-8") != text:
                print(f"DRIFT: {dst.relative_to(REPO)}"); rc = 1
            else:
                print(f"ok: {dst.relative_to(REPO)}")
        else:
            dst.write_text(text, encoding="utf-8")
            print(f"wrote {dst.relative_to(REPO)}")
    for target in present_targets():
        status, _ = alphasteer_status(target)
        rt, view = wave_of(target)
        print(f"# {target} (roster {rt}/{view}): alphasteer {status}"
              + ("" if status in (ALPHASTEER_SERVED, BASELINE_STATUS) else " -> no alphasteer cell")
              + (f" (LANE POLICY {LANE_POLICY[target]['lane']}: {len(LANE_POLICY[target]['cells'])} admitted objects"
                 + (f" + {len(PROMPTED_CELLS)} prompted arms" if LANE_POLICY[target]["prompted"] else "")
                 + " served + measured, PSR included, none served for health only)"
                 if status == ALPHASTEER_LANE_EXCLUDED else "")
              + (f" (RE-BAKE rung {_[ 'rung']}, {_['selection_outcome']}; + arditi)" if status == BASELINE_STATUS else "")
              + (" (FRONT DOOR: exactly the view's cells served + measured, ReLU arms included, no alphasteer)" if status == FRONT_DOOR_STATUS else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
