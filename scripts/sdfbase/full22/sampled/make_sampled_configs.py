#!/usr/bin/env python
"""full22 SAMPLED re-run: generate the per-(target, trial) configs from the wave configs + ONE preset file.

Inputs (all committed; NONE of them is modified here):
  configs/sdfbase_full22_<TAG>.yaml          the wave config of a (target, view) in the sampled scope: TAG = <key> for the
                                             `main` view, <key>_<view> otherwise (common.sh f22_resolve_config's TAG);
                                             SAMPLED_VIEWS below names the scope — DESIGN.md's four `main` views, the
                                             two instruct-direct views gemma4e4b/it and qwen35_4b/it (TAGs
                                             gemma4e4b_it / qwen35_4b_it), the strength frontier's 24 per-dose views
                                             (`d<dose>` / `it_d<dose>`, TAGs like nemotron3nano4b_d0p25 /
                                             gemma4e4b_it_d0p25), and the frontier's twelve llama8b dose views (BOTH
                                             families on the one served instruct model, TAGs llama8b_d0p25 /
                                             llama8b_it_d0p25; llama8b's `main` / `it` views are NOT here, see
                                             SAMPLED_VIEWS)
  configs/sdfbase_sampled_decode.yaml        key -> vendor preset, source, source_kind (+ seed_base, n_trials, arm); a
                                             view inherits its KEY's preset (the preset is the served model's, not the view's);
                                             `seed_base_by_view` / `n_trials_by_view` (TAG -> value; declared replicate TAGs
                                             only) give the seedrep views their own seed base and trial count (1: the three
                                             seeds are the replication) — every other TAG keeps the campaign's, byte for byte
  configs/sdfbase_full22_roster.yaml         the model / revision pins the preset file must agree with

Outputs (regenerated; tests/test_sdfbase_full22_sampled.py requires byte-identity), per TAG and trial T:
  configs/sdfbase_full22_<TAG>_sampled_t<T>.yaml            the base-cap config of trial T (static 4096 / controls 512)
  configs/sdfbase_full22_<TAG>_sampled_t<T>_cap8192.yaml    the static 8192-rung config of trial T (chained from the base)
  configs/sdfbase_full22_<TAG>_sampled_t<T>_cap16384.yaml   the static 16384-rung config of trial T (chained from the 8192
                                                            rung: the ladder ceiling is 16,384 with a 1% at-cap trigger;
                                                            the rung LOGIC in the sbatch / analysis is implemented
                                                            elsewhere — this file only makes the config exist for it)

Each output is the wave config's TEXT with exactly these edits (DESIGN.md "The one config knob both drivers read"):
  * a header comment naming this generator and the sha256 of the source config's bytes;
  * `out_dir` / `private_run` renamed to `sdfbase_full22_<TAG>_sampled_t<T>[_cap<rung>]`;
  * a NEW top-level `sampled_decode` block appended at the end (arm, trial, n_trials, seed_base, the preset's keys
    ONLY, source, source_kind, applies_to, static_cap == attacks.static_max_new_tokens, controls_cap == the
    config's controls cap: attacks.controls_max_new_tokens -- the wave's own key since 2026-09-11, 512 -- else
    attacks.target_max_new_tokens, sisteer.eval.cell_controls.controls_cap's rule), closed by the `readouts`
    sub-block: `readouts.gsm8k = {cap: 8192, thinking: "on", preset, source}` -- the vendor's THINKING preset where
    one exists (Qwen), the single preset otherwise, `source` saying which -- and `readouts.mmlu = {instrument:
    letter_logprob_argmax, note}` (sampler-moot). The GSM8K readout reads `readouts.gsm8k`; the static families and
    the judged controls ignore the sub-block (their validators admit the key and read the top-level preset only);
  * rung variants additionally: `attacks.static_max_new_tokens: <rung>`, `sampled_decode.static_cap: <rung>`, and
    `sampled_decode.escalated_from: {kind: static, cap: <previous cap>, private_run: <the previous rung's private_run>}`
    (8192 <- the base trial at 4096; 16384 <- the 8192 rung).
Everything else is byte-for-byte the wave config's text, so the two are equal AS DATA except those keys — the
generator loads its own output and refuses to write otherwise. The Petri manifest is NOT duplicated: the sbatch

Usage:
  python scripts/sdfbase/full22/sampled/make_sampled_configs.py            # write every output
  python scripts/sdfbase/full22/sampled/make_sampled_configs.py --check    # exit 1 on drift (a changed wave config
                                                                           # changes the recorded sha -> drift)
  python scripts/sdfbase/full22/sampled/make_sampled_configs.py --is-in-scope <key> <view>
                                                                           # the shell gate's ONE-list query: exit 0 when
                                                                           # (key, view) is in the scope, 3 when it is not;
                                                                           # reads no file for a SAMPLED_VIEWS key (runs before
                                                                           # any venv); a FRONT-DOOR key reads the models table
                                                                           # + the production roster (below)
  python scripts/sdfbase/full22/sampled/make_sampled_configs.py --tags     # the scope's TAGs, one per line

THE FRONT DOOR'S PHASE VIEWS: every model of the models table
``reproduce/models.yaml`` (read through its ONE reader ``reproduce/lib/models.py``; ``SISTEER_MODELS_TABLE`` overrides the path, as the
stage scripts do) whose ``views.<view>`` matches :data:`PHASE_VIEW_RE` (``u0``, ``s<N>[p]``, ``d<a>[p<b>][_relu]``, ``dm<a>[p<b>][_relu]``)
is a scope key with those views — DERIVED at import (:func:`front_door_views`, :data:`FRONT_DOOR_VIEWS`), NEVER written into
SAMPLED_VIEWS, which stays the paper's PRODUCTION list byte for byte (the tests pin it). A front-door (key, view) is IN SCOPE the
moment the production roster names it (05_stage_roster.sh --step roster pastes the block, --step configs renders the wave pair) and
not before: a view the roster does not name yet is neither rendered nor refused (the hand-listed rule "the scope names key/view but
the roster has no such view" would break the generator for everyone while the roster lacks them), and ``main`` prints a ``skip`` line
per pending TAG. The preset is the KEY's row in configs/sdfbase_sampled_decode.yaml (required, like every scope key); the trial count
and seed base are the campaign's (two trials at 20260909 — stage 06 submits ``static_base_t0`` / ``t1`` and ``controls_t0`` / ``t1``).
Under a lane-private roster the private roster's own views are the scope for every key, exactly as before.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "seedrep"))
from f22_roster import target_capability_omitted  # noqa: E402  (the roster's gsm8k omission for llama8b)
import eval_seed as _es  # noqa: E402  (qwen-base-seed-replicates: the replicate seed bases, stdlib only)
PRESETS = REPO / "configs" / "sdfbase_sampled_decode.yaml"
ROSTER = REPO / "configs" / "sdfbase_full22_roster.yaml"


#: THE ROSTER THIS RUN RENDERS AGAINST, and therefore its SCOPE. Default: the production roster, whose scope is
#: SAMPLED_VIEWS below. A LANE-PRIVATE roster -- `--roster PATH`, or the F22_ROSTER the rest of the pipeline already
#: exports (common.sh), which is how the shell gate `--is-in-scope` sees it without a new flag -- renders ITS OWN views
#: instead, exactly as make_full22_configs.py's certification-only path does.
#:
#: WHY NOT just add a lane's views to SAMPLED_VIEWS: that list is the PRODUCTION scope, and a view in it that the
#: production roster does not name makes check_pins refuse -- i.e. it breaks the generator for everyone. A lane must
#: never be able to make its views exist by editing a shared list, and it must never edit the production roster
#: either, whose record set the stager's precheck holds to equality.
def _roster_from_env() -> Path:
    v = os.environ.get("F22_ROSTER")
    if not v:
        return ROSTER
    p = Path(v)
    return (p if p.is_absolute() else REPO / p).resolve()


_ROSTER_PATH = _roster_from_env()


def _private() -> bool:
    """Whether this run renders a lane-private roster rather than the production one."""
    return _ROSTER_PATH.resolve() != ROSTER.resolve()


def scope_views(key: str) -> tuple:
    """The views rendered for `key`: SAMPLED_VIEWS under the production roster — or, for a FRONT-DOOR key (the models table's phase
    views, FRONT_DOOR_VIEWS), those of its views the production roster names (none until 05 --step roster lands the block) — else
    the private roster's own views for that key (empty when it does not name the key at all)."""
    if not _private():
        if key in SAMPLED_VIEWS:
            return tuple(SAMPLED_VIEWS[key])
        if key in FRONT_DOOR_VIEWS:
            named = ((load_roster().get("targets") or {}).get(key) or {}).get("views") or {}
            return tuple(v for v in FRONT_DOOR_VIEWS[key] if v in named)
        return ()
    t = (load_roster().get("targets") or {}).get(key)
    return tuple((t or {}).get("views") or ())


def scope_keys() -> tuple:
    """Every key the scope can name, hand-listed first (SAMPLED_VIEWS, the paper's order), then the front door's (the models table's
    order) — the iteration order of sampled_tags() and split_tag()."""
    return (*SAMPLED_VIEWS, *(k for k in FRONT_DOOR_VIEWS if k not in SAMPLED_VIEWS))


def front_door_pending() -> list:
    """The front-door (key, view) pairs the models table names but the production roster does not yet — in the table's order; empty
    under a lane-private roster (its own views are the scope there)."""
    if _private():
        return []
    targets = load_roster().get("targets") or {}
    return [(k, v) for k, views in FRONT_DOOR_VIEWS.items() for v in views if v not in ((targets.get(k) or {}).get("views") or {})]
GENERATOR = "scripts/sdfbase/full22/sampled/make_sampled_configs.py"

#: the sampler keys a preset may carry (DESIGN.md: "Send exactly the preset's keys (+ seed)"); jailbreak_run
#: refuses a `sampled_decode` block whose keys fall outside this set, so the generator refuses first
ALLOWED_SAMPLER_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
#: where a preset came from: the snapshot's generation_config.json, the model card's stated settings, or a vendor
#: recommendation that DIFFERS from the shipped file (then `shipped_generation_config` in the preset file records
#: the file's values beside it; the block carries them inside `source`, whose key set is DESIGN's)
SOURCE_KINDS = ("generation_config", "model_card", "vendor_recommendation")
#: the block's non-sampler keys, in the order DESIGN.md writes them
BLOCK_HEAD = ("arm", "trial", "n_trials", "seed_base")
BLOCK_TAIL = ("source", "source_kind", "applies_to", "static_cap", "controls_cap")
#: the per-readout sub-block: the LAST key of the block, rendered behind this marker
#: comment so the bytes the live jobs ran (everything before it) stay reproducible from the current file
READOUTS_MARKER = "  # -- per-readout overrides"
READOUT_NAMES = ("gsm8k", "mmlu")
#: bare words YAML 1.1 reads as a bool / null (PyYAML): a string that spells one must be quoted or `thinking: on` is True
_YAML_RESERVED = {"on", "off", "yes", "no", "true", "false", "null", "y", "n", "~"}
STATIC_CAP_BASE = 4096
STATIC_CAP_ESC = 8192
#: the static escalation rungs, in ladder order; each is chained from the one before (8192 from the base 4096,
#: 16384 from 8192). 16384 is the ladder ceiling (1% at-cap trigger); its run logic lives elsewhere
STATIC_RUNGS = (STATIC_CAP_ESC, 16384)
CONTROLS_CAP = 512
#: the STRENGTH FRONTIER's per-dose views (24 in all): one bank view per dose, named `d<dose>` with the dose spelled
#: 0.25 -> 0p25, 1.5 -> 1p5, -1 -> m1, and the `it_` prefix marking the instruct-direct family. `d1` / `it_d1`
#: re-serve the campaign's UNSCALED objects as a same-pass replicate anchor (the `ep16re` precedent) so the frontier's
#: 1.0 point is measured by the same code on the same day against the same `none` anchor as its neighbours. Ordinary
#: views to this generator: the wave config of the view (make_full22_configs.py's, off the roster) is the source, the
#: preset is the KEY's, the ladder is the campaign's.
DOSE_TOKENS = ("m1", "0p1", "0p25", "0p5", "1", "1p5")
DOSE_VIEWS = tuple(f"d{t}" for t in DOSE_TOKENS)          # the base-lane targets: nemotron3nano4b, glm46vflash (+ llama8b, below)
DOSE_VIEWS_IT = tuple(f"it_d{t}" for t in DOSE_TOKENS)    # the instruct-direct family: gemma4e4b, qwen35_4b (+ llama8b, below)
#: the (key, view) pairs in the sampled scope, as views per preset key — THE ONE LIST. DESIGN.md fixed the four `main`
#: views; the two instruct-direct views joined under the identical contract (four cells each: none, the two itdoc
#: objects, the prompted safety baseline); the strength frontier's 24 per-dose views are above. EXPLICIT on purpose and
#: NEVER derived from the roster's view list — the halfdose views are out of scope and must stay out; a view enters the
#: sampled run by being written here, not by existing in the roster.
#: Every other place that needs the scope ASKS this module rather than restating it: common_sampled.sh's view gate runs
#: `--is-in-scope <key> <view>` (in_scope), submit_sampled.py / grade_static_sampled.py / grade_sampled.sh import
#: sampled_tags() — so the shell and the python can never disagree (tests/test_sdfbase_full22_sampled.py drives both).
#: llama8b's TWELVE dose views are in scope — and ONLY those. The roster's llama8b target carries both ladder families
#: on the one served instruct model: `d<dose>` (the base-family LoRA transferred onto the instruct target) and
#: `it_d<dose>` (the instruct-direct LoRA), so llama8b is the one key that takes DOSE_VIEWS AND DOSE_VIEWS_IT. Its
#: `main` and `it` views are DELIBERATELY NOT here: they ran at their own sha and are origin/main's rows (its scope
#: names them; this branch's does not). The frontier's 1.0 point is not read off them either — `d1` / `it_d1` are the
#: ladder's OWN byte-identical re-serve anchors (the `d1` rule above) and ARE in scope. The preset is the served
#: model's row (configs/sdfbase_sampled_decode.yaml `llama8b`, origin/main's byte for byte); the wave configs are
#: make_full22_configs.py's twelve configs/sdfbase_full22_llama8b_{it_,}d*.yaml.
#: The three SEED-REPLICATE views of `qwen35_4b` (`s1` / `s2` / `s3`, TAGs qwen35_4b_s<s>) each draw the static
#: families and the over-refusal ladder under ITS OWN seed base — the preset file's `seed_base_by_view[tag]`
#: (eval_seed.seed_base(s); `expected_block`) — while every other view keeps the campaign base. They are DECLARED here
#: and PROMOTED into SAMPLED_VIEWS (the rendered scope, the shell gate and sampled_tags()) together with their roster
#: views and their 18 generated configs; `check_pins` REFUSES a roster that names a declared replicate view the scope
#: does not render (the promotion is `SAMPLED_VIEWS["qwen35_4b"] += SEEDREP_VIEWS`, forced, never forgotten), and
#: refuses a `seed_base_by_view` TAG that is not one of these views.
SEEDREP_VIEWS = {"qwen35_4b": tuple(_es.view_of(s) for s in _es.SEEDS)}
# gemma / qwen: the instruct-direct family (it_d*) AND the base->instruct family (bare d*: the
# base-fit constitution / control / subtraction objects served on the instruct target)
#: gemma4e4b's base->instruct ladder is EXTENDED to doses 2 and 4 (every dose from -1 to 1.5 read flat there; the
#: static families + XSTest are the instruments). Only gemma, only the bare family: the roster's
#: `d2` / `d4` views (own objects d2__ / d4__, own same-pass `none`), appended after d1p5 so the ladder order is kept.
HIGH_DOSE_VIEWS = {"gemma4e4b": ("d2", "d4")}
#: THE ReLU (PSR) DIFFERENCE LADDER on qwen35_4b base->instruct: six doses of
#: `constv2-mix-full__psr_ce_all_minus_ctl_distilled` served NATIVELY, four static families + the over-refusal controls,
#: nothing else. Its own view namespace (`d<tok>_relu`, roster `object_family: relu`) because the bare `d<tok>` views are
#: the LoRA ladder's and are already measured, and its own dose UNIT (the ReLU direction norm, not the LoRA write-basis
#: multiplier), so the two ladders are never read as one axis.
RELU_DOSE_VIEWS = {"qwen35_4b": tuple(f"d{t}_relu" for t in DOSE_TOKENS)}
SAMPLED_VIEWS = {"gemma4e4b": ("main", "it", *DOSE_VIEWS_IT, *DOSE_VIEWS, *HIGH_DOSE_VIEWS["gemma4e4b"]),
                 # + the ReLU difference ladder (RELU_DOSE_VIEWS, 2026-09-16) after the bare d*, + the three seed-replicate views
                 # s1/s2/s3 LAST (SEEDREP_VIEWS; PROMOTED 2026-09-15 on `reseed` with their roster views)
                 "qwen35_4b": ("main", "it", *DOSE_VIEWS_IT, *DOSE_VIEWS, *RELU_DOSE_VIEWS["qwen35_4b"], *SEEDREP_VIEWS["qwen35_4b"]),
                 "nemotron3nano4b": ("main", *DOSE_VIEWS), "glm46vflash": ("main", *DOSE_VIEWS),
                 # llama8b: the `main` and `it` views + the FULL-constitution prompted view
                 # (appendix; its sampled configs carry the static leg + the escalation rungs) + the SHORT constitution-SI prompted
                 # view (the paper's constitution-SI arm: every leg at its PER-LEG windows, roster serving.windows). The llama8b
                 # key is NOT one of the four preset KEYS; its main-view TAG is the bare key `llama8b`.
                 "llama8b": ("main", "it", "fullconst", "shortconst",
                             # + the strength frontier's twelve dose views: six base-fit, six it-fit
                             *DOSE_VIEWS, *DOSE_VIEWS_IT)}


#: the FRONT DOOR's phase views, DERIVED from the models table — module docstring. The grammar is
#: the seedrep grader twin's and the misalign generator's (`PHASE_VIEW_RE`): one tag vocabulary across the three.
#: `s<N>r` = the ReLU-parent subset of phase N (`s2r` = control_relu + constitution_relu at alpha 1; a view outside
#: this grammar renders nothing here while the planner still plans it) — the same grammar in every copy, pinned equal by tests
PHASE_VIEW_RE = re.compile(r"^(u0|s[0-9]+r?p?|s[0-9]+rl|d[0-9]+(p[0-9]+)?(_relul?)?|dm[0-9]+(p[0-9]+)?(_relul?)?)$")
MODELS_TABLE = Path(os.environ.get("SISTEER_MODELS_TABLE") or (REPO / "reproduce" / "models.yaml"))
MODELS_TABLE_READER = REPO / "reproduce" / "lib" / "models.py"     # the ONE reader of the table (duplicate keys refused; review A9)


def _models_reader():
    """``reproduce/lib/models.py`` loaded by path (the ONE reader; duplicate keys refused); absent beside this tree, refused."""
    if not MODELS_TABLE_READER.is_file():
        raise SystemExit(f"models table reader {MODELS_TABLE_READER} is not on this tree — refusing to read the table another way")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_sampled_reproduce_models_table", MODELS_TABLE_READER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_models_table(path: Path) -> dict:
    """``reproduce/models.yaml`` through its ONE reader; a table without its reader beside this tree is refused, never read another way."""
    return _models_reader().load_models(str(path))


def front_door_views(table_path: "Path | str | None" = None) -> dict:
    """``{model key: (phase views, in the table's order)}`` for every model of the models table that names phase views. The key IS
    the roster target and the preset key (the wave pair is ``sdfbase_full22_<key>_<view>.yaml``), so a model whose
    ``view_legacy_prefix`` differs from its key refuses (two spellings would name two tags for one wave); a key SAMPLED_VIEWS already
    lists refuses (one key, one scope list). An ABSENT table contributes nothing (the paper's tree before the front door); the paper's
    four models carry ``main`` / ``it`` only and contribute nothing."""
    p = Path(table_path) if table_path is not None else MODELS_TABLE
    if not p.is_file():
        return {}
    out: dict = {}
    reader = _models_reader()
    for key, m in reader.load_models(str(p)).items():
        if not isinstance(m, dict) or not isinstance(m.get("views"), dict):
            continue
        views = tuple(reader.phase_views_of(m))      # the reader's ONE classification: the grammar's views + the row's declared wave (demo-import M3)
        if not views:
            continue
        if key in SAMPLED_VIEWS:
            raise SystemExit(f"{p.name}: model {key!r} names phase views {views} but is a SAMPLED_VIEWS key — one key, one scope list")
        prefix = m.get("view_legacy_prefix")
        if prefix and prefix != key:
            raise SystemExit(f"{p.name}: model {key!r} spells its views {prefix!r}_<view> (view_legacy_prefix) but the sampled TAG is "
                             f"{key!r}_<view> (the roster target / preset key) — one spelling per wave; refusing")
        out[str(key)] = views
    return out


FRONT_DOOR_VIEWS = front_door_views()


def tag_of(key: str, view: str) -> str:
    """common.sh f22_resolve_config's TAG: the key for `main`, `<key>_<view>` otherwise."""
    return key if view == "main" else f"{key}_{view}"


def in_scope(key: str, view: str) -> bool:
    """Whether (key, view) is in the rendered scope — the ONE question the shell gate asks (`--is-in-scope`): SAMPLED_VIEWS
    under the production roster, the lane roster's own views under a private one (scope_views)."""
    return view in scope_views(key)


def split_tag(tag: str) -> tuple:
    """A TAG -> (key, view): the exact-match inverse of tag_of over the scope; refuses a TAG outside it."""
    for key in scope_keys():
        for v in scope_views(key):      # lane-private roster: a key it does not name renders nothing
            if tag == tag_of(key, v):
                return key, v
    raise SystemExit(f"{tag!r} is not a (target, view) in the sampled scope {sampled_tags()}")


def key_of(tag: str) -> str:
    """The preset KEY a TAG serves (the view inherits the served model's preset); refuses a TAG outside the scope."""
    return split_tag(tag)[0]


def sampled_tags() -> list:
    """Every TAG in the scope, key order (SAMPLED_VIEWS', then the front door's) then view order (SAMPLED_VIEWS': main first;
    a front-door key: the models table's, once the roster names the view)."""
    return [tag_of(k, v) for k in scope_keys() for v in scope_views(k)]


def seedrep_tags() -> list:
    """The declared seed-replicate TAGs (SEEDREP_VIEWS through tag_of), whether or not they are in the rendered scope yet."""
    return [tag_of(k, v) for k, vs in SEEDREP_VIEWS.items() for v in vs]


#: THE PER-TAG SEED-BASE RULE TABLE: TAG -> the seed base its `seed_base_by_view` entry must equal. The seed-replicate
#: TAGs come from eval_seed (seed_base(s) = s, the LITERAL TRAINING SEED; the older hashed rule is retired). Views that
#: draw at the training seed of the object they serve are added HERE as literal data rows (`"<tag>": <base>`), so
#: `load_presets` never learns a second rule: it holds the data file's `seed_base_by_view` to EXACTLY this table's keys
#: and values, and every base to "never the campaign base".
#: THE SEEDED DOSE VIEWS: the batch-64 seed-replicate CONTROL parents at strength -1, one
#: view per (fit seed k, class) -- `dm1_ctllora64s<k>` (the control LoRA) and `dm1_ctlrelu64s<k>` (the control PSR) --
#: served by the LANE-PRIVATE roster configs/sdfbase_full22_roster_ctl_neg1_eb64.yaml, never by the production scope.
#: Each draws at the TRAINING SEED of the object it serves, exactly the s<k> views' rule (seed_base(k) = k, ONE trial),
#: so its rows join the rule table below as the "views ... added HERE as literal data rows" the table's
#: docstring provides for. The seed is the view name's trailing `s<k>` (seed_of_lane_view), the same one rule the
#: wave generator's family_of_segment strips -- one spelling per seed, no s1 special case.
SEEDED_LANE_VIEWS = {"qwen35_4b": tuple(f"dm1_{cls}s{k}" for k in _es.SEEDS for cls in ("ctllora64", "ctlrelu64"))}
_LANE_SEED_RE = re.compile(r"^[a-z0-9_]+?s(?P<seed>[1-9][0-9]*)$")


def seed_of_lane_view(view: str) -> int:
    """`dm1_ctllora64s2` -> 2: the fit seed a seeded lane view's name carries (its trailing `s<k>`); refuses any other view."""
    m = _LANE_SEED_RE.match(str(view))
    if not m or int(m.group("seed")) not in _es.SEEDS:
        raise ValueError(f"{view!r} is not a seeded lane view (a trailing s<k>, k in {_es.SEEDS})")
    return int(m.group("seed"))


def lane_replicate_tags() -> list:
    """The seeded lane TAGs (SEEDED_LANE_VIEWS through tag_of), whether or not the run's roster renders them."""
    return [tag_of(k, v) for k, vs in SEEDED_LANE_VIEWS.items() for v in vs]


def replicate_tags() -> list:
    """Every TAG that draws at a training seed with its own trial count: the declared seed-replicate TAGs (seedrep_tags)
    + the seeded lane TAGs (lane_replicate_tags) -- the key set `n_trials_by_view` may name."""
    return seedrep_tags() + lane_replicate_tags()


SEED_BASE_RULE_BY_TAG = {**{tag_of(k, v): _es.seed_base(_es.seed_of_view(v)) for k, vs in SEEDREP_VIEWS.items() for v in vs},
                         **{tag_of(k, v): _es.seed_base(seed_of_lane_view(v)) for k, vs in SEEDED_LANE_VIEWS.items() for v in vs}}


def seed_base_of_tag(presets: dict, tag: str) -> int:
    """The seed base a TAG's sampled configs draw under: the preset file's `seed_base_by_view[tag]` when the TAG has one (the
    lane's replicate views), else the campaign `seed_base` — the ONE lookup `expected_block`, the header and the tests share."""
    return int((presets.get("seed_base_by_view") or {}).get(tag, presets["seed_base"]))


def n_trials_of_tag(presets: dict, tag: str) -> int:
    """The number of sampled trials a TAG renders (`t0` .. `t<n-1>`, base + every static rung each): the preset file's
    `n_trials_by_view[tag]` when the TAG has one (qwen-seedrep-native-battery, 1 trial per seed view: three seeds ARE the
    replication), else the campaign `n_trials` — the ONE lookup `expected_block`,
    `outputs`, the header and the tests share. `load_presets` pins the map to the declared replicate TAGs."""
    return int((presets.get("n_trials_by_view") or {}).get(tag, presets["n_trials"]))


def stale_outputs(presets: dict) -> list:
    """Committed sampled configs of a TAG whose `n_trials_by_view` entry no longer renders them: a `_sampled_t<T>` file
    (base or rung) with T >= the TAG's trial count. `--check` refuses them — a config that no generator renders is a
    stale instrument, deleted in the same change that shrinks the count (the six `qwen35_4b_s<s>_sampled_t1*`)."""
    out = []
    for tag in sorted(presets.get("n_trials_by_view") or {}):
        n = n_trials_of_tag(presets, tag)
        for p in sorted((REPO / "configs").glob(f"sdfbase_full22_{tag}_sampled_t*.yaml")):
            m = re.fullmatch(rf"sdfbase_full22_{re.escape(tag)}_sampled_t(\d+)(_cap\d+)?\.yaml", p.name)
            if m and int(m.group(1)) >= n:
                out.append(p)
    return out


def rung_cap(esc) -> int | None:
    """The static cap a variant escalates to: None for the base config; `True` (the original bool API) = 8192;
    an int must be one of STATIC_RUNGS."""
    if esc is False or esc is None:
        return None
    if esc is True:
        return STATIC_CAP_ESC
    cap = int(esc)
    if cap not in STATIC_RUNGS:
        raise SystemExit(f"rung {cap} is not one of the static rungs {STATIC_RUNGS}")
    return cap


def previous_rung(cap: int):
    """What a rung is chained from: (previous cap, previous `esc` value) — the base for 8192, the 8192 rung for 16384."""
    i = STATIC_RUNGS.index(cap)
    return (STATIC_CAP_BASE, False) if i == 0 else (STATIC_RUNGS[i - 1], STATIC_RUNGS[i - 1])


def load_presets(path: Path = PRESETS) -> dict:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key, p in d["presets"].items():
        bad = [k for k in p["preset"] if k not in ALLOWED_SAMPLER_KEYS]
        if bad:
            raise SystemExit(f"{path.name}: preset {key} carries non-sampler keys {bad} (allowed: {ALLOWED_SAMPLER_KEYS})")
        if p["source_kind"] not in SOURCE_KINDS:
            raise SystemExit(f"{path.name}: preset {key} source_kind {p['source_kind']!r} must be one of {SOURCE_KINDS}")
        shipped = p.get("shipped_generation_config")
        if p["source_kind"] == "vendor_recommendation" and not shipped:
            raise SystemExit(f"{path.name}: preset {key} is a vendor_recommendation but records no shipped_generation_config "
                             "(the file's values must sit beside the preset that runs)")
        if shipped and shipped == p["preset"]:
            raise SystemExit(f"{path.name}: preset {key} records shipped_generation_config identical to the preset — "
                             "the field exists to show a DIFFERENCE; drop it or fix the preset")
        if not p.get("source"):
            raise SystemExit(f"{path.name}: preset {key} has no source")
    r = d.get("readouts")
    if not isinstance(r, dict) or set(r) != set(READOUT_NAMES):
        raise SystemExit(f"{path.name}: readouts must name exactly {READOUT_NAMES}, got {sorted(r) if isinstance(r, dict) else r!r}")
    g = r["gsm8k"]
    if isinstance(g.get("cap"), bool) or not isinstance(g.get("cap"), int) or g["cap"] < 1:
        raise SystemExit(f"{path.name}: readouts.gsm8k.cap must be a positive int, got {g.get('cap')!r}")
    if g.get("thinking") != "on":
        raise SystemExit(f"{path.name}: readouts.gsm8k.thinking must be the string \"on\" (quoted: YAML reads a bare on as true), got {g.get('thinking')!r}")
    for key, mode in (g.get("thinking_by_key") or {}).items():
        if key not in d["presets"]:
            raise SystemExit(f"{path.name}: readouts.gsm8k.thinking_by_key names {key}, which has no preset")
        if mode not in ("on", "off"):
            raise SystemExit(f"{path.name}: readouts.gsm8k.thinking_by_key.{key} must be the string \"on\" or \"off\", got {mode!r}")
        if not (g.get("thinking_by_key_source") or {}).get(key):
            raise SystemExit(f"{path.name}: readouts.gsm8k.thinking_by_key.{key} needs thinking_by_key_source.{key} (why this model has no thinking row)")
        if key in (g.get("presets") or {}):
            raise SystemExit(f"{path.name}: readouts.gsm8k.thinking_by_key.{key}: a model with no thinking mode cannot also carry a THINKING preset row")
    if not g.get("reuse_source"):
        raise SystemExit(f"{path.name}: readouts.gsm8k.reuse_source (the no-thinking-row statement) is missing")
    for key, tp in (g.get("presets") or {}).items():
        if key not in d["presets"]:
            raise SystemExit(f"{path.name}: readouts.gsm8k.presets names {key}, which has no preset")
        bad = [k for k in tp.get("preset") or {} if k not in ALLOWED_SAMPLER_KEYS]
        if bad or not tp.get("preset"):
            raise SystemExit(f"{path.name}: readouts.gsm8k.presets.{key} carries non-sampler keys {bad} or no preset")
        if not tp.get("source"):
            raise SystemExit(f"{path.name}: readouts.gsm8k.presets.{key} has no source")
        if tp["preset"] == d["presets"][key]["preset"]:
            raise SystemExit(f"{path.name}: readouts.gsm8k.presets.{key} equals the single preset — drop it (a key absent here reuses it)")
    m = r["mmlu"]
    if m.get("instrument") != "letter_logprob_argmax" or not m.get("note"):
        raise SystemExit(f"{path.name}: readouts.mmlu must be {{instrument: letter_logprob_argmax, note: ...}}, got {m!r}")
    # `seed_base_by_view` names EXACTLY the TAGs of the per-tag rule table
    # (SEED_BASE_RULE_BY_TAG: the declared replicate TAGs at eval_seed's base = the literal training seed; a peer lane's rows as data),
    # each at the table's base — a mistyped base, a stray TAG or a missing TAG refuses here, before any config is rendered; absent = no
    # replicate view yet. A base equal to the campaign's refuses: a replicate view draws under its OWN base.
    sbv = d.get("seed_base_by_view")
    if sbv is not None:
        if not isinstance(sbv, dict) or not sbv:
            raise SystemExit(f"{path.name}: seed_base_by_view must be a non-empty mapping TAG -> seed base, got {sbv!r}")
        if set(sbv) != set(SEED_BASE_RULE_BY_TAG):
            raise SystemExit(f"{path.name}: seed_base_by_view names {sorted(sbv)} but the rule table names "
                             f"{sorted(SEED_BASE_RULE_BY_TAG)} (make_sampled_configs.SEED_BASE_RULE_BY_TAG) — one entry per table row, no strays")
        for tag, base in sbv.items():
            want = SEED_BASE_RULE_BY_TAG[tag]
            if isinstance(base, bool) or not isinstance(base, int) or base != want:
                raise SystemExit(f"{path.name}: seed_base_by_view.{tag} = {base!r} but the rule table says {want} (SEED_BASE_RULE_BY_TAG; the "
                                 "seed-replicate rows are eval_seed.seed_base(s) = s) — the declaration must equal the rule (scripts/sdfbase/seedrep/eval_seed.py)")
            if base == int(d["seed_base"]):
                raise SystemExit(f"{path.name}: seed_base_by_view.{tag} equals the campaign seed_base {d['seed_base']} — a replicate view "
                                 "draws under its OWN base")
    # `n_trials_by_view` names declared replicate TAGs only, each at a positive
    # trial count that DIFFERS from the campaign's `n_trials` (an equal entry is a no-op nobody meant); absent = every TAG
    # renders the campaign's count, byte for byte
    ntv = d.get("n_trials_by_view")
    if ntv is not None:
        if not isinstance(ntv, dict) or not ntv:
            raise SystemExit(f"{path.name}: n_trials_by_view must be a non-empty mapping TAG -> trial count, got {ntv!r}")
        strays = sorted(set(ntv) - set(replicate_tags()))
        if strays:
            raise SystemExit(f"{path.name}: n_trials_by_view names {strays}, not declared replicate TAGs "
                             f"{sorted(replicate_tags())} (make_sampled_configs.SEEDREP_VIEWS + SEEDED_LANE_VIEWS) — no strays")
        for tag, n in ntv.items():
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise SystemExit(f"{path.name}: n_trials_by_view.{tag} = {n!r} must be a positive int")
            if n == int(d["n_trials"]):
                raise SystemExit(f"{path.name}: n_trials_by_view.{tag} equals the campaign n_trials {d['n_trials']} — the map "
                                 "states a DIFFERENCE; drop the entry")
    return d


def load_roster(path: Path | None = None) -> dict:
    path = path or _ROSTER_PATH
    from f22_roster import load_roster as _strict_load     # the ONE roster loader: a duplicate mapping key refuses by name and line
    return _strict_load(Path(path))


def check_pins(presets: dict, roster: dict) -> None:
    """A preset read from one snapshot must not ride a config that serves another; every view in the sampled scope
    must be a view the roster names for that key (a view's config IS the roster's, via make_full22_configs.py)."""
    for key, p in presets["presets"].items():
        t = roster["targets"].get(key)
        if t is None:
            # a row for a target another wave serves (llama8b: the sdfbase-llama8b wave's roster row
            # lands with that wave) — the drivers resolve it by model, never through the roster;
            # the generator renders only SAMPLED_VIEWS, which must be roster targets (below)
            if key in SAMPLED_VIEWS and not _private():
                raise SystemExit(f"preset {key} names a target the roster does not have")
            continue   # lane-private roster: other presets' targets are simply out of this run
        if (p["model"], p["revision"]) != (t["model"], t["model_revision"]):
            raise SystemExit(f"preset {key} was read from {p['model']}@{p['revision']} but the roster serves "
                             f"{t['model']}@{t['model_revision']}")
    for key in scope_keys():
        if key not in presets["presets"]:
            raise SystemExit(f"{'SAMPLED_VIEWS' if key in SAMPLED_VIEWS else 'the models table (FRONT_DOOR_VIEWS)'} names {key} but the "
                             f"preset file has no preset for it")
        views = scope_views(key)
        if not views:
            continue          # lane-private roster: this key is not in ITS scope; or a front-door key the roster does not carry yet
        t = roster["targets"].get(key)
        if t is None:
            raise SystemExit(f"the scope names {key} but the roster has no such target")
        for view in views:
            if view not in (t.get("views") or {}):
                raise SystemExit(f"the scope names {key}/{view} but the roster has no such view")
    # a SEEDED LANE view the run's roster names draws at its training seed with ONE trial: both rows must be in the data
    # file (load_presets already holds seed_base_by_view to the rule table; the trial count is the lane's declaration)
    sbv, ntv = presets.get("seed_base_by_view") or {}, presets.get("n_trials_by_view") or {}
    for key, views in SEEDED_LANE_VIEWS.items():
        for view in views:
            tag = tag_of(key, view)
            if view in ((roster["targets"].get(key) or {}).get("views") or {}) and (tag not in sbv or tag not in ntv):
                raise SystemExit(f"the roster names the seeded lane view {key}/{view} but {PRESETS.name} carries no "
                                 f"seed_base_by_view.{tag} / n_trials_by_view.{tag} — a seeded lane view draws at its training seed, one trial")
    # qwen-base-seed-replicates: a declared replicate view (SEEDREP_VIEWS) needs its seed base in the preset file, and the moment
    # the ROSTER names it, it must be in the rendered scope too — a roster view the sampled scope silently leaves out
    # would be an unrendered instrument, so the generator refuses until the view is promoted into SAMPLED_VIEWS
    sbv = presets.get("seed_base_by_view") or {}
    for key, views in SEEDREP_VIEWS.items():
        for view in views:
            tag = tag_of(key, view)
            if tag not in sbv:
                raise SystemExit(f"replicate view {key}/{view} is declared (SEEDREP_VIEWS) but {PRESETS.name} carries no seed_base_by_view.{tag}")
            in_roster = view in ((roster["targets"].get(key) or {}).get("views") or {})
            if in_roster and not in_scope(key, view):
                raise SystemExit(f"the roster names the replicate view {key}/{view} but SAMPLED_VIEWS does not render it — promote it "
                                 f"(SAMPLED_VIEWS[{key!r}] += SEEDREP_VIEWS[{key!r}]) and generate its sampled configs in the same change")


def wave_config_path(tag: str) -> Path:
    """The wave config of a TAG (`sdfbase_full22_<key>[_<view>].yaml`, make_full22_configs.py's output)."""
    return REPO / "configs" / f"sdfbase_full22_{tag}.yaml"


def sampled_name(tag: str, trial: int, esc=False) -> str:
    cap = rung_cap(esc)
    return f"sdfbase_full22_{tag}_sampled_t{trial}" + (f"_cap{cap}" if cap else "")


def output_path(tag: str, trial: int, esc=False) -> Path:
    return REPO / "configs" / f"{sampled_name(tag, trial, esc)}.yaml"


def expected_block(presets: dict, tag: str, trial: int, esc=False) -> dict:
    """The `sampled_decode` block AS DATA — the contract the tests and the drivers read. `tag` is a TAG (a bare key
    is the `main` view's TAG); the preset is the KEY's."""
    p = presets["presets"][key_of(tag)]
    cap = rung_cap(esc)
    if not (0 <= int(trial) < n_trials_of_tag(presets, tag)):
        raise SystemExit(f"{tag}: trial {trial} is not one of the TAG's {n_trials_of_tag(presets, tag)} trial(s)")
    block = {"arm": presets["arm"], "trial": trial, "n_trials": n_trials_of_tag(presets, tag),   # the TAG's own count on a
             "seed_base": seed_base_of_tag(presets, tag)}       # 1-trial replicate view, its own base; the campaign's elsewhere
    block.update({k: p["preset"][k] for k in ALLOWED_SAMPLER_KEYS if k in p["preset"]})
    block.update({"source": p["source"], "source_kind": p["source_kind"],
                  "applies_to": list(presets["applies_to"]),
                  "static_cap": cap or STATIC_CAP_BASE, "controls_cap": CONTROLS_CAP})
    if cap:
        prev_cap, prev_esc = previous_rung(cap)
        block["escalated_from"] = {"kind": "static", "cap": prev_cap,
                                   "private_run": sampled_name(tag, trial, prev_esc)}
    block["readouts"] = expected_readouts(presets, key_of(tag))
    return block


def expected_readouts(presets: dict, key: str, roster: dict = None) -> dict:
    """The `readouts` sub-block of a KEY: GSM8K thinking-on at the readouts cap under
    the vendor's THINKING preset when the data file names one for the key, else the key's single preset with a `source`
    that says no thinking-specific vendor row exists; MMLU as recorded (sampler-moot). Every view of a key shares it.
    A key whose ROSTER omits gsm8k (targets.<key>.capability_omitted — llama8b, 2026-09-10) records the omission
    INSTEAD: `gsm8k: {omitted: true, reason: ...}` — never a silent absence, never a
    readout block a driver could run."""
    r = presets["readouts"]
    g = r["gsm8k"]
    p = presets["presets"][key]
    omitted = target_capability_omitted(roster if roster is not None else load_roster(), key)
    if "gsm8k" in omitted:
        return {"gsm8k": {"omitted": True, "reason": omitted["gsm8k"]},
                "mmlu": {"instrument": r["mmlu"]["instrument"], "note": r["mmlu"]["note"]}}
    think = (g.get("presets") or {}).get(key)
    mode = (g.get("thinking_by_key") or {}).get(key, g["thinking"])
    if think is not None:
        preset, source = dict(think["preset"]), think["source"]
    else:
        preset = {k: p["preset"][k] for k in ALLOWED_SAMPLER_KEYS if k in p["preset"]}
        reason = (g["thinking_by_key_source"][key] if mode != g["thinking"] else g["reuse_source"])
        source = (f"{p['model']} @ {p['revision'][:8]}: {reason} -- the preset that runs "
                  f"({', '.join(f'{k} {v}' for k, v in preset.items())}; source_kind {p['source_kind']})")
    return {"gsm8k": {"cap": int(g["cap"]), "thinking": mode, "preset": preset, "source": source},
            "mmlu": {"instrument": r["mmlu"]["instrument"], "note": r["mmlu"]["note"]}}


def _scalar(v) -> str:
    """One YAML scalar: floats keep a decimal point (1.0 stays a float when read back), strings that are not
    plain words are JSON-quoted (a JSON string is a valid YAML double-quoted scalar)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    s = str(v)
    if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*", s) and s.lower() not in _YAML_RESERVED:
        return s
    return json.dumps(s, ensure_ascii=True)


def _render_nested(d: dict, indent: int) -> list:
    """Block-style YAML lines of a nested mapping at `indent` spaces (scalars through _scalar)."""
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.append(f"{' ' * indent}{k}:")
            out += _render_nested(v, indent + 2)
        else:
            out.append(f"{' ' * indent}{k}: {_scalar(v)}")
    return out


def render_block(block: dict) -> str:
    lines = ["sampled_decode:"]
    for k, v in block.items():
        if k == "applies_to":
            lines.append(f"  applies_to: [{', '.join(_scalar(x) for x in v)}]")
        elif k == "escalated_from":
            lines.append("  # a RE-DRAW at the larger cap (same seed), never a continuation; only the base run's at-cap")
            lines.append("  # rows are regenerated, every terminated row is copied forward (escalate_static.py)")
            lines.append("  escalated_from:")
            for kk, vv in v.items():
                lines.append(f"    {kk}: {_scalar(vv)}")
        elif k == "readouts":
            lines.append(f"{READOUTS_MARKER} (the coordinator's second amendment, 2026-09-09; lane G1 reads readouts.gsm8k): GSM8K")
            lines.append("  # THINKING-ON at the readouts cap under the vendor's THINKING preset where one exists (Qwen), the single preset")
            lines.append("  # above otherwise (`source` says which); MMLU is a teacher-forced letter-logprob argmax (sampler-moot). The static")
            lines.append("  # families and the judged controls IGNORE this sub-block (their validators admit the key, read the preset above).")
            if (v.get("gsm8k") or {}).get("omitted"):
                lines.append("  # gsm8k is OMITTED for this key (roster capability_omitted; the reason below) — no gsm8k readout runs on it.")
            lines.append("  readouts:")
            lines += _render_nested(v, 4)
        else:
            lines.append(f"  {k}: {_scalar(v)}")
    return "\n".join(lines) + "\n"


def _sub_line(body: str, key_regex: str, new_line: str) -> str:
    n = len(re.findall(key_regex, body, flags=re.M))
    if n != 1:
        raise SystemExit(f"expected exactly 1 line matching {key_regex!r}, found {n}")
    return re.sub(key_regex, new_line, body, count=1, flags=re.M)


def render(presets: dict, tag: str, trial: int, esc=False) -> str:
    """One output: `tag` a TAG (a bare key = its `main` view), `esc` False (base) | True / 8192 | 16384 (a rung)."""
    key = key_of(tag)
    cap = rung_cap(esc)
    src_path = wave_config_path(tag)
    src_bytes = src_path.read_bytes()
    src_sha = hashlib.sha256(src_bytes).hexdigest()
    text = src_bytes.decode("utf-8")
    name = sampled_name(tag, trial, esc)
    text = _sub_line(text, rf"^out_dir: out/sdfbase_full22_{re.escape(tag)}$", f"out_dir: out/{name}")
    text = _sub_line(text, rf"^private_run: sdfbase_full22_{re.escape(tag)}$", f"private_run: {name}")
    if cap:
        text = _sub_line(text, rf"^  static_max_new_tokens: {STATIC_CAP_BASE}$",
                         f"  static_max_new_tokens: {cap}")
    block = expected_block(presets, tag, trial, esc)
    p = presets["presets"][key]
    extra = ""
    if cap:
        prev_cap, prev_esc = previous_rung(cap)
        prev_name = sampled_name(tag, trial, prev_esc)
        prev_what = "the base run" if prev_esc is False else f"the {prev_cap} rung"
        extra = (f"#   * `attacks.static_max_new_tokens` {prev_cap} -> {cap} and `sampled_decode.static_cap` "
                 f"{cap}: the static {cap} RUNG of trial {trial}, a SEPARATE store\n"
                 f"#     (`sampled_decode.escalated_from` names {prev_what} {prev_name}; only its at-cap rows are\n"
                 f"#     re-drawn, DESIGN.md 'The primary comparison is SAME CAP vs SAME CAP').\n")
    header = (f"# {name}.yaml — full22 SAMPLED re-run, trial {trial}" + (f" @ the static {cap} rung" if cap else "") + "\n"
              f"# (scripts/sdfbase/full22/sampled/DESIGN.md). GENERATED by {GENERATOR} from\n"
              f"# configs/{src_path.name} (source config sha256 {src_sha}) and\n"
              f"# configs/sdfbase_sampled_decode.yaml — never hand-edited; `--check` refuses drift and the test pins byte-identity.\n"
              f"# The source config's text follows BYTE-FOR-BYTE (its own header included), EXCEPT:\n"
              f"#   * `out_dir` / `private_run` -> `{name}` (a NEW store; the greedy store is never touched).\n"
              f"{extra}"
              f"#   * the top-level `sampled_decode` block APPENDED at the end: {p['model']} @ {p['revision'][:8]}'s\n"
              f"#     vendor preset ({', '.join(f'{k} {v}' for k, v in p['preset'].items())}; source_kind {p['source_kind']}),\n"
              f"#     seed_base {seed_base_of_tag(presets, tag)}, trial {trial} of {n_trials_of_tag(presets, tag)}. The Petri manifest stays the wave's\n"
              f"#     configs/capeval/manifests/petri_full22_{tag}.yaml (the sbatch pairs them).\n"
              f"#\n")
    tail = ("\n# -- full22 SAMPLED re-run (DESIGN.md): the vendor-recommended sampler, read by jailbreak_run.py generate\n"
            "# (--trial must equal `trial`; static_cap must equal attacks.static_max_new_tokens) and by\n"
            "# served_capability_run.py (controls_gen_config); every row records `decode` + `sampling_seed` --\n"
            + render_block(block))
    out = header + text + tail
    _verify(out, src_bytes.decode("utf-8"), block, name, cap)
    return out


def _verify(out: str, src_text: str, block: dict, name: str, cap) -> None:
    """The output is the wave config AS DATA except the permitted keys, and its block is the intended one."""
    got, src = yaml.safe_load(out), yaml.safe_load(src_text)
    if got["sampled_decode"] != block:
        raise SystemExit(f"{name}: rendered sampled_decode != intended block\n{got['sampled_decode']}\n{block}")
    if got["out_dir"] != f"out/{name}" or got["private_run"] != name:
        raise SystemExit(f"{name}: out_dir / private_run not renamed")
    changed = {k for k in set(got) | set(src) if got.get(k) != src.get(k)}
    want = {"out_dir", "private_run", "sampled_decode"} | ({"attacks"} if cap else set())
    if changed != want:
        raise SystemExit(f"{name}: keys differing from the wave config {sorted(changed)} != permitted {sorted(want)}")
    a, b = dict(got["attacks"]), dict(src["attacks"])
    if cap:
        if a.pop("static_max_new_tokens") != cap or b.pop("static_max_new_tokens") != STATIC_CAP_BASE or a != b:
            raise SystemExit(f"{name}: attacks differs from the wave config beyond static_max_new_tokens")
    if got["sampled_decode"]["static_cap"] != got["attacks"]["static_max_new_tokens"]:
        raise SystemExit(f"{name}: static_cap != attacks.static_max_new_tokens")
    # the controls' cap: attacks.controls_max_new_tokens (the wave's own key since 2026-09-11) else
    # attacks.target_max_new_tokens -- sisteer.eval.cell_controls.controls_cap's rule, repeated so this
    # generator needs no sisteer import
    ccap = got["attacks"].get("controls_max_new_tokens", got["attacks"]["target_max_new_tokens"])
    if got["sampled_decode"]["controls_cap"] != ccap:
        raise SystemExit(f"{name}: controls_cap != the config's controls cap {ccap} "
                         "(attacks.controls_max_new_tokens, else attacks.target_max_new_tokens)")


def outputs(presets: dict) -> list:
    """[(path, tag, trial, esc)] in generation order: key (preset order), view (SAMPLED_VIEWS order: main first),
    trial, then base / cap8192 / cap16384. `esc` is False for the base config and the rung's cap (int) otherwise."""
    out = []
    # the RENDERED scope is SAMPLED_VIEWS: a preset row for a target outside it (llama8b —
    # the sdfbase-llama8b wave's row, rendered by that wave) is a driver-side row, not a config
    for key in presets["presets"]:
        if key not in scope_keys():
            continue
        for view in scope_views(key):    # a front-door key: its roster-named phase views (none until 05 lands the block)
            tag = tag_of(key, view)
            for trial in range(n_trials_of_tag(presets, tag)):
                for esc in (False, *STATIC_RUNGS):
                    out.append((output_path(tag, trial, esc), tag, trial, esc))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if any committed output drifts")
    ap.add_argument("--is-in-scope", nargs=2, metavar=("KEY", "VIEW"),
                    help="the shell gate's query: exit 0 (printing the TAG) when (KEY, VIEW) is in SAMPLED_VIEWS, 3 when it "
                         "is not (the scope on stderr); reads no file, so it runs before any venv exists")
    ap.add_argument("--tags", action="store_true", help="print the scope's TAGs, one per line; reads no file")
    ap.add_argument("--roster", type=Path, default=None,
                    help="render a LANE-PRIVATE roster's own views (default: $F22_ROSTER, else the production roster)")
    args = ap.parse_args(argv)
    if args.roster:
        global _ROSTER_PATH
        _ROSTER_PATH = args.roster.resolve()
    if args.is_in_scope:
        key, view = args.is_in_scope
        if in_scope(key, view):
            print(tag_of(key, view)); return 0
        print(f"{key}/{view} is not in the sampled scope: {sampled_tags()}", file=sys.stderr); return 3
    if args.tags:
        print("\n".join(sampled_tags())); return 0
    presets = load_presets()
    check_pins(presets, load_roster())
    for key, view in front_door_pending():
        print(f"skip {tag_of(key, view)}: the roster does not carry {key}/{view} yet (05_stage_roster.sh --step roster) — rendered once it does")
    rc = 0
    for dst, tag, trial, esc in outputs(presets):
        text = render(presets, tag, trial, esc)
        if args.check:
            if not dst.is_file() or dst.read_text(encoding="utf-8") != text:
                print(f"DRIFT: {dst}"); rc = 1
            else:
                print(f"ok: {dst}")
        else:
            dst.write_text(text, encoding="utf-8"); print(f"wrote {dst}")
    for p in stale_outputs(presets):
        # a committed config no trial renders any more (n_trials_by_view shrank the TAG's count): delete it in the same change
        print(f"{'DRIFT' if args.check else 'STALE'}: {p} — trial beyond the TAG's n_trials_by_view count; delete it"); rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
