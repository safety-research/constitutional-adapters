#!/usr/bin/env python3
"""Generate the PADDED STATIC leg's config + manifest pair from the full22 campaign originals
(docs/static_padding.md). The campaign files are UNTOUCHED (every wave test pinning them stays green); the
outputs are the same files with the named keys changed and a header prepended, ONE config per padding level
because the harness holds ONE padding contract per store (static_padding.assert_static_padding_consistent) and a
store's identity is the config's `private_run`:

  configs/sdfbase/static_padding/sdfbase_full22_<key>.padstatic_p<pad>.yaml     (pad in PADS: 0, 8192, 16384, 32768, 65536 —
                                                                                 the five rungs; THE ADDED RUNGS below)
      the campaign config configs/sdfbase_full22_<key>.yaml with
        * `explicit_cells` replaced by EXACTLY the three prompted-side arms (CELLS: the same-wave bare anchor and the
          two prompted SI baselines — no fitted object, so no bank view entry is needed; steered arms join later
          under their own pair and the wave's bank view);
        * `out_dir` / `private_run` renamed to pad_static_<key>_p<pad> (the store identity: one contract per store);
        * `attacks.static_max_new_tokens` raised from the campaign's 4,096 to STATIC_CAP = 16,384, the static ladder's
          CEILING, at EVERY level (the p0 control included: padded and unpadded are contrasted at ONE cap) — WHY below;
        * for pad > 0, an `attacks.static_padding` block PINNING the contract — tokens, placement, mode, reserve —
          inserted after `attacks.static_max_new_tokens`. The FILLER PATH is deliberately NOT pinned here: it is a
          cluster-local resource the sbatch names on the argv (--static-padding-filler, sha-verified by the lane's
          preflight first), so no field is stated twice (a typed flag would silently win over the block). The pad
          level therefore travels with the CONFIG, never as a driver flag: a driver cannot be run under the wrong
          level by a mistyped --static-padding-tokens, because it passes none.
        The p0 config carries NO block (0 = off: every byte the static path writes is as before); it is the
        control of THIS wave — a real run, never a reused older store — and the config the standup serves from.
  configs/capeval/manifests/petri_full22_<key>.padstatic.yaml
      the campaign manifest with `cells:` replaced by those cells' harness ids (mt_serving_up.py refuses a
      config / manifest cell mismatch); content otherwise identical — the same chat_template_hash,
      serving_launch_flags (text-only pin), tool_parser, prompted_prefix_hashes (which already carry the safety and
      constitution SIs' hex16). ONE manifest for the three configs: padding is request content, not a served
      cell, so the served set is the same at every level.

WHY `reserve: 8192` and not the library default 2,048. The harness's window preflight asserts
tokens + attacks.static_max_new_tokens + reserve <= --target-max-model-len, where the reserve stands in for the
system instruction + the payload + the chat template. The constitution SI (`prompted__si_constitution`) renders to
5,973 tokens on the pinned Qwen3.5-4B tokenizer and the longest static prompt under it to 6,412 (grok420; EVERY
variant of every goal of all four families rendered on cluster B 2026-09-22 — the char-longest proxy said 6,403),
so 2,048 would let the harness ADMIT a run whose every constitution-cell request the server then refuses (or clamps
below the cap). 8,192 >= 6,412 + the 2 % pad tolerance's overshoot (1,311) = 7,723 makes the harness's own preflight
honest for this battery; the lane's padding_preflight.py additionally derives the exact rendered need per cell before
any GPU minute. The wave's window is 98,304 (the measured need at 65,536 is 6,412 + 66,847 + 16,384 = 89,643; the rule
is to raise the window, never shrink the pad), so the harness rule holds with room: 65,536 + 16,384 +
8,192 = 90,112, and 98,304 sits one reserve (8,192) above it at the same derived S (padding_preflight.py, check 5).

WHY THE CAP IS THE LADDER'S CEILING (16,384; ladder_decide.CEILING), not the campaign's 4,096 — recorded here
because a 16,384 cap on a leg whose completions are mostly short LOOKS wasteful and is not.
This leg is PREFILL-bound: Qwen3.5-4B is a hybrid target with no prefix caching, so every request re-prefills the whole
pad, while a completion cap costs only for tokens actually generated. The standing truncation rule would escalate any
(cell, readout) above 1 % at-cap to a larger rung — and on this leg a rung re-pays the FULL pad prefill for every
re-drawn item; worse, the rung driver refuses padded stores outright (escalate_static.assert_base_unpadded: it re-draws
the UNPADDED attack_prompt, so a re-draw would be a different measurement wearing the store's name), so a >1 % at-cap
share could only be met by regenerating the entire level at a larger cap. Running ONCE at the ceiling is what makes the
leg single-pass: no rung can be needed, because there is no rung above. The marginal price is the decode of the items
that would have hit 4,096 (the sbatch header carries the arithmetic); the price of a 4,096 cap is a whole level again.

THE STEERED (OBJECT) FAMILY — `_obj` (2026-09-23): the padding leg extended to STEERED objects, ONE
attack family, grok420, on qwen35_4b. A SECOND config family beside the prompted-only one above, ONE generator so the
two never drift:

  configs/sdfbase/static_padding/sdfbase_full22_<key>.padstatic_obj_p<pad>.yaml   (pad in OBJ_PADS: PADS + the two ADDED
      RUNGS below)
      the campaign config as above, EXCEPT `explicit_cells` = the TEN cells of OBJ_CELLS (`none`, the two prompted SI
      baselines, the five fitted s1 objects the production roster's `s1` view serves, the all-positions Arditi rung
      ar_c0p3 under its harness id `arditi_allpos`, and the paired write `constv2-mix-full__psr_ce_all_minus_ctl`
      — full dose, seed 1, never the half-dose view or a *_distilled object), `defenses_dir` = the wave bank view
      `out/full22/<key>/cells_view/padobj` (OBJ_VIEW; on the cluster TT_TWO_SI_DEFENSES_DIR, exported by common.sh
      f22_bank_view), `out_dir` / `private_run` = pad_static_<key>_obj_p<pad> — its OWN namespace, so nothing collides
      with the prompted-only wave's filed stores. Same cap, same padding block, same rules otherwise.
  configs/capeval/manifests/petri_full22_<key>.padstatic_obj.yaml
      the campaign manifest with `cells:` = the ten harness ids (mt_serving_up.py's config / manifest gate; the
      kind-aware serving-path default reads it and selects lora_native — every one of the ten is natively servable).
  configs/sdfbase/static_padding/sdfbase_full22_<key>.padobj_roster.yaml
      the LANE'S OWN ROSTER (a certification-only roster's shape: configs/sdfbase_full22_roster_psrcert.yaml, the
      pairedcert view) naming ONE view, `padobj`, that the production roster does NOT name (common.sh f22_bank_view
      refuses a non-production roster that names a production view — the bank view is shared state keyed by name).
      The five s1 cells map to the SAME staged ids the production `s1` view maps (read from the production roster,
      never retyped; pinned in configs/sdfbase_full22_pins.json); the Arditi rung and the paired write are named
      through the roster's CROSS-BANK PIN FORM (f22_roster.PIN_FORM_KEYS: `{pins, pin}`) against their owning lanes'
      pin files — configs/arditi_allpos_pins.json `qwen35_4b.ar_c0p3__arditi_allpos` and configs/reseed_paired_pins.json
      `qwen35_4b.constv2-mix-full__psr_ce_all_minus_ctl` — so the view entry is a SYMLINK to the source bank's staged
      copy (exp/objects/arditi_allpos/…/cells/ar_c0p3__arditi_allpos, exp/objects/paired_write/qwen35_4b/cells/
      paired_s1__…) whose bytes make_bank_view.py re-hashes to the pin, and NO bytes are copied into the wave bank
      (the form the wave bank's `pairedcert` / `paired_d1_s1` / `ar_c0p3` views already use — verified on cluster A
      2026-09-23, VIEW_MANIFEST.json `form: pin`). The roster is GENERATED here so the view's cell set and the
      configs' explicit_cells are one spelling; the sbatch sets F22_ROSTER to it before f22_init.
      Served natively: every one of the ten cells' exports is a rank-1 zero o_proj carrier (the residual kinds psr /
      psr_paired_write / vector_all) or an o_proj LoRA (lora_r4 / lora_subtraction), all 4,096 wide on Qwen3.5-4B
      (num_attention_heads 16 x head_dim 256; read off every export under exp/multilora_adapters/qwen35_4b on
      cluster A 2026-09-23: max factor dim 4,096 for every kind), so the standup's kernel-ceiling step derives the SAME
      S at the window as the prompted-only wave's — the sbatch's GEOMETRY PIN re-derives and refuses otherwise.

THE ADDED RUNGS (2026-09-24): two more points on the steered curve. OBJ_PADS_ADDED = (8,192, 32,768) joined the
steered family first, so its levels are OBJ_PADS = (0, 8,192, 16,384, 32,768, 65,536): five rungs at ONE window (98,304) and ONE derived
S (5), so the curve is a single comparable series on one instrument, never two. The window holds with room at 32,768: the
measured floor is 6,412 + 33,424 (the pad at its 2 % worst case: 32,768 + ceil(655.36)) + 16,384 = 56,220, the harness rule
32,768 + 16,384 + 8,192 = 57,344 (padding_preflight.py re-derives both before any GPU minute). The three filed steered
configs (p0 / p16384 / p65536) are BYTE-IDENTICAL to before this change — nothing in their text carries a level count, and
the added rungs' one extra header paragraph (ADDED RUNG) is rendered for OBJ_PADS_ADDED only (tests pin the filed sha256s:
a change to a filed config is a regression against the results registry). The added rungs run SHARDED ACROSS JOBS
(target_static_padded_obj.sbatch's `shards` positional, `<lo>-<hi>/N`); the store layout is the one a single job would
write, so the grader and the degeneracy writer read it unchanged with `--nshards N`.

  THE PROMPTED-ONLY FAMILY HAS THE SAME FIVE RUNGS (2026-09-24, later the same day). The first cut of this paragraph gave the added
  rungs to the steered family ONLY — "the prompted-only family's three stores ran on cluster B and are filed, and gain nothing" —
  and that reasoning is SUPERSEDED by the results registry's reference rule (src/sisteer/registry/adapters/static_padding.py):
  every steered store pad_static_<key>_obj_p<PAD> is filed on the instrument the prompted-only REFERENCE standup
  pad_static_<key>_p<PAD> mints at the SAME pad (its `none` + two prompted cells; the pad and the filler contract make the identity,
  the objects never do), and a steered store with no reference at its pad is refused (reference.state "absent"). So the steered
  rungs at 8,192 / 32,768 — generated, graded and mirrored — filed ZERO rows until a reference ran at each; the seed-2 / seed-3
  replicates at those pads read the same reference (its cells are unfitted, hence seed-independent). The references were run
  rather than the rule changed, so PADS = OBJ_PADS: one level list for every family
  (pads_of returns it for each). The three filed prompted-only configs stay byte-identical; the two new ones carry one extra header
  paragraph (REFERENCE STANDUP OF AN ADDED RUNG) rendered for `family is None and pad in OBJ_PADS_ADDED`. They run through
  target_static_padded.sbatch (on cluster A: --qos=high and the shared-storage log path on the sbatch line), one job per level,
  grok420 only, the filed rungs' three shards — the same window, cap, decode default and padding contract, so the reference
  re-mints the instrument the steered rungs already attest (the batch width S is NOT an identity field: a prompted-only standup
  derives its own, and the GEOMETRY PIN holds it to the steered arms' anyway).

THE SEED REPLICATES — `obj_s2` / `obj_s3` (2026-09-24): seeds 2 and 3 of the six FITTED objects, so the finer
claims of the seed-1 curve — the difference LoRA above the constitution objects, the rung where an instruction crosses zero —
are read against object-seed noise. Two more config families beside `obj`, ONE generator so nothing drifts:

  configs/sdfbase/static_padding/sdfbase_full22_<key>.padstatic_obj_s<N>_p<pad>.yaml   (N in OBJ_SEEDS; pad in OBJ_PADS: the
      SAME five rungs) — the campaign config as above, EXCEPT `explicit_cells` = the SEVEN cells of OBJ_SEED_CELLS: the
      same-wave bare anchor `none` first, then the four fitted base64s<N> objects, the difference LoRA derived_base64s<N>, and
      the FULL-dose paired write paired_eb64s<N> — every object the SAME cell id as the seed-1 family, served from the seed-N
      object; `defenses_dir` = the lane's view `out/full22/<key>/cells_view/padobj_s<N>`; `out_dir` / `private_run` =
      pad_static_<key>_obj_s<N>_p<pad> — its own namespace, so nothing collides with the seed-1 stores. NO prompted baseline,
      NO Arditi rung (neither is seed-replicated). Same cap, same padding block, same window, same S.
      WHY `none` IS LISTED although it has no training seed (the existing code cannot reuse the seed-1 `none`): the NATIVE
      STANDUP derives its zero-adapter leg `none__lora0` from the `none` cell (defense_cells.enumerate_cells: the leg is
      the control OF the base cell), exports and registers the zero adapter from that leg, and sizes the LoRA kernel index
      ceiling over the REGISTERED adapters (native_standup.bring_up_native: the zero-leg check precedes the export, which
      precedes lora_kernel_ceiling_geometry).
      A six-cell set without `none` was submitted first (2026-09-24, 29 jobs): every standup refused at 1-5 min with
      "enumerate_cells derived no none__lora0 leg", and the geometry line it had printed before refusing was the IDENTITY
      S = 7 at 98,304 — the kernel-ceiling lowering to S = 5 had not run yet. So `none` is what puts the zero adapter on the
      engine and, through it, the seed-1 wave's width on the instrument; the grader gets a within-standup anchor too (the
      cross-standup seed-1 anchor is no longer needed). Every seed rung therefore re-measures `none` (~0.35-0.46 M completion
      tokens at seed 1, ~4-6 % of a rung).
  configs/capeval/manifests/petri_full22_<key>.padstatic_obj_s<N>.yaml — the seven harness ids (six natively servable objects
      + the `none` passthrough; the kind-aware default selects lora_native with nothing skipped).
  configs/sdfbase/static_padding/sdfbase_full22_<key>.padobj_s<N>_roster.yaml — the lane's roster of ONE view `padobj_s<N>`
      (the production roster names none), every cell in the STRING form: the five fitted ids READ from the production roster's
      `s<N>` view and the paired write from its `s<N>p` view — the wave bank's OWN staged copies, pinned in
      configs/sdfbase_full22_pins.json (base64s<N>__*, derived_base64s<N>__*, paired_eb64s<N>__*). The seed-1 family named its
      paired write through a CROSS-BANK PIN into exp/objects/paired_write/ because that bank holds the vLLM lane's staged copy
      of the seed-1 object; that bank stages NO seed-2/3 copy (checked on cluster A 2026-09-24: cells/ holds paired_s1 only),
      while the wave bank stages both (the production s2p / s3p views serve them), so the string form is the authority here,
      and the view's entries are same-bank symlinks (../../cells/<staged id>) that make_bank_view.py re-hashes to the pins.
      Every seed-2/3 object was re-hashed on cluster A 2026-09-24 == its pin, positions `all` in every identity sidecar
      (train_budget_meta.json for the four fits, derived_meta.json for the two derived objects, scale_matched true).

  THE DECODE SEED BASE IS HELD AT 20260909 (DECODE_SEED_BASE_HELD), the value the seed-1 wave decoded under — the vendor preset's
  campaign base, which the driver takes when no seed flag is passed (the sbatch passes none: the same argv as the seed-1 wave,
  the config aside). NOT the training seed: the campaign's seedrep contract ties the decode seed to the training seed on the
  seedrep lane's views, but here that would vary the object seed AND the decode seed together and confound them; holding the
  decode fixed isolates object-seed variation, which is the question. Recorded in every seed config's header so a later reader
  does not "fix" it to the seedrep spelling. The anchor and the arms of a seed store then differ in the object seed alone.

  THE WIDTH IS PINNED TO S = 5 BY A LITERAL (target_static_padded_obj.sbatch GEOMETRY_PIN_S): the seed-1 wave's attested width
  at 98,304 (identity 741,888 // 98,304 = 7, lowered by the LoRA kernel index ceiling 524,287 // 98,304 = 5). The sbatch passes
  it to the lane preflight (--expect-served-s: the derivation must land on it before any GPU minute) and to the post-standup
  GEOMETRY PIN, which refuses traffic unless the ATTESTED max_num_seqs, the module's derivation at the served window, AND a
  kernel-ceiling step in the attested chain all say 5 — so a standup that came up at the identity 7 (the width the six-cell
  attempt printed) can never reach a store, whatever the derivation or the geometry table says that day.

THE FRONT DOOR'S KEYS (stage 06 `padding`): every model of the models table
``reproduce/models.yaml`` (read through its ONE reader ``reproduce/lib/models.py``; ``SISTEER_MODELS_TABLE`` overrides the path, as the
stage scripts do) whose ``views.u0.cells`` are EXACTLY the three prompted-side arms (CELLS) is a key of the PROMPTED-ONLY family —
DERIVED at import (:func:`front_door_keys`, :data:`FRONT_DOOR_KEYS`), never a hand list — sourced from its ``u0`` wave pair
``configs/sdfbase_full22_<key>_u0.yaml`` / ``petri_full22_<key>_u0.yaml`` (a front-door model has no ``main`` view: ``u0`` IS its
unsteered wave, and stage 06 submits ``target_static_padded.sbatch <key> …`` for the u0 view, which registers under the ``u0`` view when
the roster names it — the sbatch's own rule). A front-door key joins KEYS the moment the production roster names ``<key>/u0`` (05
--step roster pastes the block, --step configs renders the pair) and not before: until then it is neither rendered nor refused
(``main`` prints a ``skip`` line per pending key) and KEYS stays the hand-listed tuple byte for byte (the tests pin it, and
grade_padded_static.KEYS = this KEYS is what run_identity admits). The STEERED families `_obj` / `_obj_s<N>` stay the hand-listed
keys': they are bound to qwen35_4b's `s1` objects and two cross-bank pin files.

THE FRONT DOOR'S PER-VIEW STEERED FAMILIES: a front-door key pads
its STEERED phase views through ONE config family PER VIEW, NAMED AFTER THE VIEW (`s2`, `s1`, `d0p75`, `d0p75_relu`;
:func:`front_door_view_families`, :data:`FRONT_DOOR_VIEW_FAMILIES` — every non-u0 phase view the models table declares), derived from
the ROSTER's own cells: `explicit_cells` = the view's wave config's (`none` + the roster's cells of that view, prompted baselines where
the roster lists them), `defenses_dir` = the PRODUCTION bank view `out/full22/<key>/cells_view/<view>`, no lane roster (the production
roster names the view: the sbatch's F22_ROSTER is configs/sdfbase_full22_roster.yaml), NO cross-bank pin. Each family is cut from the
VIEW's own wave pair `configs/sdfbase_full22_<key>_<view>.yaml` / `petri_full22_<key>_<view>.yaml` — the same overlay as the
prompted-only family (out_dir / private_run `pad_static_<key>_<view>_p<pad>`, the ceiling cap, the padding block), and its configs
`sdfbase_full22_<key>.padstatic_<view>_p<pad>.yaml` / manifest `petri_full22_<key>.padstatic_<view>.yaml` are what stage 06's
`target_static_padded_obj.sbatch <key> … all <view>` rows (lib/stage.py) and the grading daemon read. A view the production roster
does not name yet is neither rendered nor refused (`main` prints a skip line); a family that is not a phase view of its key, or a key
that is not a front-door key, refuses by name (`--resolve-family <key> <family>` is the sbatch's question). The hand-listed families
are byte-identical to before.

Never hand-edit the outputs: tests/test_static_padding_wave.py requires byte-identity (`--check`).

  python scripts/sdfbase/full22/static_padding/make_padding_configs.py            # write every output
  python scripts/sdfbase/full22/static_padding/make_padding_configs.py --check    # exit 1 on drift
  python scripts/sdfbase/full22/static_padding/make_padding_configs.py --list     # the output paths, one per line
                                                                                  # (the prompted-only five configs + manifest,
                                                                                  #  then the steered family's five configs,
                                                                                  #  manifest, roster,
                                                                                  #  then each seed replicate's five + manifest
                                                                                  #  + roster)
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
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))   # f22_roster: the ONE reader of a view's served cells

#: the ONE hand-listed target of this leg (the cheap prompted-side first slice on Qwen3.5-4B; steered arms and other
#: targets come later under their own pairs) — the front door's keys join KEYS below
HAND_LISTED_KEYS = ("qwen35_4b",)
#: the three prompted-side arms, in the campaign's cell order (`none` first: the same-wave bare anchor)
CELLS = ("none", "prompted__safety_refusal", "prompted__si_constitution")
#: the padding levels (tokens ADDED to every item's user turn); 0 = the unpadded control of this wave — the FIVE rungs of EVERY
#: family since 2026-09-24 (module docstring, THE ADDED RUNGS): the prompted-only family's 0 / 16,384 / 65,536 ran first (cluster B,
#: 2026-09-23) and its 8,192 / 32,768 are the REFERENCE standups the results registry requires under the steered rungs at those pads
PADS = (0, 8192, 16384, 32768, 65536)
#: THE ADDED RUNGS (module docstring; 2026-09-24): the two levels added after the first three, at the same window and the same S.
#: They joined the steered `_obj` family first and the prompted-only family the same day (the registry's reference rule); each
#: family's two new configs render one extra header paragraph, so the six filed configs stay byte-identical (tests pin their sha256s)
OBJ_PADS_ADDED = (8192, 32768)
#: the steered family's levels: the SAME five as the prompted-only family's — one comparable series on one instrument, and the
#: reference the registry files each steered rung on exists at every level
OBJ_PADS = PADS
PLACEMENT = "before"
MODE = "fixed"
#: the window preflight's headroom for system instruction + payload + chat template (module docstring)
RESERVE = 8192
#: the completion cap of EVERY level's config: the static ladder's CEILING (ladder_decide.CEILING / LADDERS["static"][-1],
#: pinned by tests/test_static_padding_wave.py), not the campaign's 4,096 — the leg runs ONCE, never climbs (module docstring)
STATIC_CAP = 16384

#: THE STEERED FAMILY (module docstring): its name segment in every generated path / private_run, its wave-bank VIEW, and
#: its ten cells in config order — `none` first (the same-wave bare anchor), the two prompted SI baselines, the five fitted
#: s1 objects in the production `s1` view's order, the all-positions Arditi rung (harness id `arditi_allpos`: the RUNG is
#: the pin's — ar_c0p3 — never part of the cell id), the paired write last
OBJ_FAMILY = "obj"
OBJ_VIEW = "padobj"
#: the production view whose five fitted cells this family re-serves (their staged ids are READ from the roster's view)
OBJ_SOURCE_VIEW = "s1"
OBJ_S1_CELLS = ("constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all",
                "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl")
OBJ_ARDITI_CELL = "arditi_allpos"
OBJ_PAIRED_CELL = "constv2-mix-full__psr_ce_all_minus_ctl"
OBJ_CELLS = CELLS + OBJ_S1_CELLS + (OBJ_ARDITI_CELL, OBJ_PAIRED_CELL)
#: the two cross-bank PIN references (f22_roster.PIN_FORM_KEYS): {cell id: (pins file relative to the repo, `<key>.<entry>`)}
#: — the Arditi rung is c = 0.3 (the pin's staged id spells it; the cell id never does), the paired write full dose seed 1
OBJ_PIN_REFS = {OBJ_ARDITI_CELL: ("configs/arditi_allpos_pins.json", "ar_c0p3__arditi_allpos"),
                OBJ_PAIRED_CELL: ("configs/reseed_paired_pins.json", OBJ_PAIRED_CELL)}
#: the wave pins' staged id of the SAME paired bytes (the production `s1p` view's staged copy): the roster renderer holds the
#: cross-bank pin's object_sha256 equal to it, so the two spellings of this object can never name different bytes
OBJ_PAIRED_WAVE_STAGED_ID = "paired_eb64s1__constv2-mix-full__psr_ce_all_minus_ctl"

#: THE SEED REPLICATES (module docstring; 2026-09-24): the training seeds replicated, their config families (`obj_s2`, `obj_s3`:
#: the name segment in every generated path / private_run), and their SIX cells — the seed-1 family's fitted objects, in its
#: order, behind the same-wave bare anchor `none` (the zero-adapter leg and the seed-1 width derive from it — module docstring);
#: no prompted baseline, no Arditi rung
OBJ_SEEDS = (2, 3)
OBJ_SEED_FAMILIES = tuple(f"{OBJ_FAMILY}_s{n}" for n in OBJ_SEEDS)
OBJ_SEED_CELLS = ("none",) + OBJ_S1_CELLS + (OBJ_PAIRED_CELL,)
#: the seed-1 wave's attested width at its window — the literal the sbatch's GEOMETRY PIN holds every seed standup to
GEOMETRY_PIN_S = 5
#: every config family this generator renders, in output order: prompted-only (None), the steered seed-1 family, the seed replicates
FAMILIES = (None, OBJ_FAMILY) + OBJ_SEED_FAMILIES
#: the decode seed base every level of every family decodes under — the vendor preset's campaign base, taken by the driver when
#: no seed flag is passed (module docstring): HELD here for the seed replicates, never the training seed
DECODE_SEED_BASE_HELD = 20260909

CONFIG_DIR = REPO / "configs" / "sdfbase" / "static_padding"
MANIFEST_DIR = REPO / "configs" / "capeval" / "manifests"
PRODUCTION_ROSTER = REPO / "configs" / "sdfbase_full22_roster.yaml"
WAVE_PINS = REPO / "configs" / "sdfbase_full22_pins.json"

# --------------------------------------------------------------------------------------- the front door's keys (module docstring)
#: the view a front-door key's prompted-only family is cut from (its unsteered wave; a hand-listed key's is the campaign `main`)
FRONT_DOOR_SOURCE_VIEW = "u0"
MODELS_TABLE = Path(os.environ.get("SISTEER_MODELS_TABLE") or (REPO / "reproduce" / "models.yaml"))
MODELS_TABLE_READER = REPO / "reproduce" / "lib" / "models.py"     # the ONE reader of the table (duplicate keys refused; review A9)


def _models_reader():
    """``reproduce/lib/models.py``, the table's ONE reader, loaded by path; a table without its reader beside this tree is refused."""
    if not MODELS_TABLE_READER.is_file():
        raise SystemExit(f"models table: its reader {MODELS_TABLE_READER} is not on this tree — refusing to read it another way")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_padding_reproduce_models_table", MODELS_TABLE_READER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_models_table(path: Path) -> dict:
    """``reproduce/models.yaml`` through its ONE reader; a table without its reader beside this tree is refused, never read another way."""
    return _models_reader().load_models(str(path))


def front_door_keys(table_path: "Path | str | None" = None) -> tuple:
    """The FRONT-DOOR keys of the models table (its reader's rule: a model naming a phase view), in the table's order. A key that
    declares a ``u0`` view (the PHASED recipe) must list EXACTLY the three prompted-side arms there (CELLS, in that order) — the
    prompted-only padding family this generator cuts from its u0 wave pair; any other u0 refuses by name (the family IS those three
    arms; a different u0 is a different instrument, never silently padded). A key with NO u0 view runs the SINGLE-PHASE recipe
    (reproduce/lib/models.single_phase): it has no prompted-only family — every one of its phase views is a
    per-view steered family whose configs carry the view's own unsteered legs (:func:`front_door_view_families`). A key
    HAND_LISTED_KEYS already names refuses (one key, one source view); an ABSENT table contributes nothing."""
    p = Path(table_path) if table_path is not None else MODELS_TABLE
    if not p.is_file():
        return ()
    reader = _models_reader()
    out = []
    for key in reader.front_door_keys(str(p)):
        m = reader.model(key, str(p))
        u0 = ((m.get("views") or {}) if isinstance(m, dict) else {}).get(FRONT_DOOR_SOURCE_VIEW)
        if u0 is not None:
            cells = tuple(str(c) for c in (u0.get("cells") or ()))
            if cells != CELLS:
                raise SystemExit(f"{p.name}: model {key!r}'s {FRONT_DOOR_SOURCE_VIEW} view lists {list(cells)}, not the prompted-only family's "
                                 f"{list(CELLS)} — the padded leg pads exactly those three arms; refusing")
        if key in HAND_LISTED_KEYS:
            raise SystemExit(f"{p.name}: model {key!r} is a hand-listed key of this leg (cut from its campaign `main` config) and names "
                             "phase views — one key, one source view; refusing")
        out.append(str(key))
    return tuple(out)


def single_phase_keys(table_path: "Path | str | None" = None) -> tuple:
    """The front-door keys on the SINGLE-PHASE recipe (no ``u0`` view): no prompted-only family, per-view families for every phase view."""
    p = Path(table_path) if table_path is not None else MODELS_TABLE
    if not p.is_file():
        return ()
    reader = _models_reader()
    return tuple(k for k in front_door_keys(p) if reader.single_phase(k, str(p)))


FRONT_DOOR_KEYS = front_door_keys()
SINGLE_PHASE_KEYS = single_phase_keys()
#: the front-door keys with a prompted-only family (a declared u0 view): the PHASED recipe's
PROMPTED_ONLY_KEYS = tuple(k for k in FRONT_DOOR_KEYS if k not in SINGLE_PHASE_KEYS)


def _roster_names(key: str, view: str, roster_path: "Path | None" = None) -> bool:
    """Whether the production roster (or ``roster_path``) names ``<key>/<view>``."""
    rp = Path(roster_path) if roster_path is not None else PRODUCTION_ROSTER
    if not rp.is_file():
        return False
    roster = yaml.safe_load(rp.read_text(encoding="utf-8")) or {}
    return view in (((roster.get("targets") or {}).get(key) or {}).get("views") or {})


def source_view(key: str) -> str:
    """The wave view a key's prompted-only family is cut from: the campaign ``main`` for a hand-listed key, ``u0`` for a phased
    front-door one; a single-phase key has no prompted-only family and refuses by name (its families are per-view)."""
    if key in SINGLE_PHASE_KEYS:
        raise ValueError(f"{key}: a single-phase front-door key has no prompted-only padding family (no u0 view) — its padded families are "
                         f"its phase views, each cut from its own wave pair ({list(FRONT_DOOR_VIEW_FAMILIES.get(key, ()))})")
    return FRONT_DOOR_SOURCE_VIEW if key in FRONT_DOOR_KEYS else "main"


def front_door_pending() -> tuple:
    """The PHASED front-door keys the production roster does not carry a ``u0`` view for yet — not rendered, not refused, named by ``main``
    (a single-phase key's pending families are per view: :func:`front_door_view_pending`)."""
    return tuple(k for k in PROMPTED_ONLY_KEYS if not _roster_names(k, FRONT_DOOR_SOURCE_VIEW))


#: the PROMPTED-ONLY leg's targets: the hand-listed key(s) + every PHASED front-door key whose u0 view the production roster names (its
#: wave pair exists once 05 --step configs ran). grade_padded_static.KEYS binds this tuple, so run_identity admits a front-door model's
#: prompted-only stores exactly when its padded configs can exist, and the hand-listed tuple is byte-identical until then. A
#: single-phase key's stores are its per-view families' (`pad_static_<key>_<view>_p<pad>`), never this leg's — it never joins KEYS;
#: the GRADER admits it through GRADER_TARGETS (below families_of): the one spelling of every key the grader serves a family for.
KEYS = (*HAND_LISTED_KEYS, *(k for k in PROMPTED_ONLY_KEYS if _roster_names(k, FRONT_DOOR_SOURCE_VIEW)))

# ------------------------------------------------------- the front door's PER-VIEW steered families (module docstring)
#: the phase-view grammar every front-door generator admits (misalign / sampled / gsm8k and the seedrep grader twin): `u0`, `s<N>[p]`,
#: `d<a>[p<b>][_relu]`, `dm<a>[p<b>][_relu]`
#: `s<N>r` = the ReLU-parent subset of phase N (`s2r` = control_relu + constitution_relu at alpha 1; a view outside this grammar
#: renders nothing here while the planner still plans it) — the same grammar in every copy, pinned equal by tests
PHASE_VIEW_RE = re.compile(r"^(u0|s[0-9]+r?p?|s[0-9]+rl|d[0-9]+(p[0-9]+)?(_relul?)?|dm[0-9]+(p[0-9]+)?(_relul?)?)$")


def front_door_view_families(key: str, table_path: "Path | str | None" = None) -> tuple:
    """The STEERED per-view families of a front-door key: its non-u0 phase views, in the table's order, each family NAMED after its
    view (module docstring) — whether or not the production roster names the view yet (the NAME exists so a config name / private
    run parses; rendering refuses until the roster carries the view). For a single-phase key (no u0) that is EVERY phase view, its
    one wave `s<N>` included (that wave carries the unsteered legs the phased recipe's u0 carried; cells_of lists them from the
    roster). A key the table lacks, or one with no phase views, yields ()."""
    p = Path(table_path) if table_path is not None else MODELS_TABLE
    if not p.is_file():
        return ()
    reader = _models_reader()
    m = reader.load_models(str(p)).get(key)
    if not isinstance(m, dict):
        return ()
    # a view the table EXCLUDES from padding by ruling (`exclude_families: [padding]`) is no padded family —
    # omitted here (so it is never rendered or parsed as one) and refused by name below; explicit and tested, never the grammar's accident.
    # The phase views are the reader's ONE classification (models.phase_views_of: the grammar's plus the row's declared wave — demo-import M3).
    return tuple(v for v in reader.phase_views_of(m) if v != FRONT_DOOR_SOURCE_VIEW
                 and "padding" not in (((m.get("views") or {}).get(v) or {}).get("exclude_families") or []))


def front_door_view_excluded(key: str, view: str, table_path: "Path | str | None" = None) -> "str | None":
    """The reason the models table EXCLUDES the padding family from (key, view) — `views.<view>.exclude_families` carries
    `padding` (the s2r objects never run the ladder); None otherwise."""
    p = Path(table_path) if table_path is not None else MODELS_TABLE
    if not p.is_file():
        return None
    m = _load_models_table(p).get(key)
    vd = ((m.get("views") or {}) if isinstance(m, dict) else {}).get(view) or {}
    if "padding" in (vd.get("exclude_families") or []):
        return str(vd.get("exclude_reason") or "excluded by ruling (no reason recorded)")
    return None


def front_door_view_names(key: str, table_path: "Path | str | None" = None) -> tuple:
    """Every non-u0 phase view NAME of a front-door key (the reader's classification: the grammar's views plus the row's declared wave),
    a view the table EXCLUDES from padding INCLUDED — the NAME is admitted so a config name, a private run or a `--resolve-family` call
    spelling it parses and is refused by the RULING (`_require_front_door_family`), never as an unknown family. Which of these
    render is :func:`front_door_view_families`'s question."""
    p = Path(table_path) if table_path is not None else MODELS_TABLE
    if not p.is_file():
        return ()
    reader = _models_reader()
    m = reader.load_models(str(p)).get(key)
    if not isinstance(m, dict):
        return ()
    return tuple(v for v in reader.phase_views_of(m) if v != FRONT_DOOR_SOURCE_VIEW)


FRONT_DOOR_VIEW_FAMILIES = {k: front_door_view_families(k) for k in FRONT_DOOR_KEYS}
#: the front door's per-view family NAMES, the union over every front-door key of its non-u0 phase views — excluded views included, so their
#: names parse and are refused by the ruling — the table-built set the name-only predicate below reads (demo-import M3: a declared wave such
#: as `battery` is a per-view family by the table, never by the grammar; the grammar's names stay admitted exactly as before)
FRONT_DOOR_FAMILY_NAMES = frozenset(v for k in FRONT_DOOR_KEYS for v in front_door_view_names(k))
#: every family NAME this generator renders or parses: the hand-listed keys' (FAMILIES) + the front door's per-view family names — a name
#: is admitted here; which KEY renders which family is families_of / render_config's question (an excluded view's name is admitted and
#: refused by name downstream)
ALL_FAMILY_NAMES = FAMILIES + tuple(dict.fromkeys([f for fams in FRONT_DOOR_VIEW_FAMILIES.values() for f in fams if f not in FAMILIES]
                                                  + [f for k in FRONT_DOOR_KEYS for f in front_door_view_names(k) if f not in FAMILIES]))


def is_front_door_family(family: "str | None") -> bool:
    """Whether a family name is a front door's PER-VIEW steered family NAME (a phase view of a front-door key: a grammar view or the key's
    declared wave, from the table-built FRONT_DOOR_FAMILY_NAMES — a view the table excludes from padding is still the NAME of one, refused
    by the ruling when rendered or resolved), never `obj` / `obj_s<N>` / the prompted-only None. Name-only — its callers (view_of / _fam)
    hold no key — so it reads the union the table built at import, not the grammar."""
    return family is not None and family not in FAMILIES and family != FRONT_DOOR_SOURCE_VIEW and str(family) in FRONT_DOOR_FAMILY_NAMES


def families_of(key: str) -> tuple:
    """The config families rendered for a key, in output order: the hand-listed keys' (prompted-only, `obj`, the seed replicates); a
    phased front-door key's prompted-only family + one steered family per phase view the PRODUCTION roster names (families it does not
    name yet are pending, never rendered); a single-phase key's per-view families alone. The ONE spelling of which key renders which
    family: outputs() iterates it, and grade_padded_static.run_identity holds a config's (key, family) pair to it after admitting the key
    (pads_of / cells_of / private_run know nothing of the key for the hand-listed families, so a stale `<single-phase key>.padstatic_p0`
    or `..._obj_p0` — a family that key never had — is refused HERE, by name)."""
    if key in HAND_LISTED_KEYS:
        return FAMILIES
    prompted = () if key in SINGLE_PHASE_KEYS else (None,)      # the single-phase recipe has no prompted-only family
    return (*prompted, *(v for v in FRONT_DOOR_VIEW_FAMILIES.get(key, ()) if _roster_names(key, v)))


def output_keys() -> tuple:
    """Every key this generator renders for, in output order: the leg's KEYS (hand-listed + the PHASED front-door keys whose u0 the
    production roster carries) and every SINGLE-PHASE front-door key the roster names at least one view of (its per-view families; it
    has no prompted-only family, so it is never in KEYS). A phased key whose u0 is still PENDING is outside this tuple whatever else the
    roster names of it (outputs() renders nothing for it; it joins KEYS, and so this tuple, the moment the roster names its u0) — and
    GRADER_TARGETS, below, IS this tuple, so the grader refuses exactly the keys the generator renders nothing for."""
    return tuple(dict.fromkeys((*KEYS, *(k for k in SINGLE_PHASE_KEYS if families_of(k)))))


#: the GRADER's targets (grade_padded_static.TARGETS: `--target choices=` and run_identity's admission): EXACTLY output_keys(), the keys
#: this generator renders at least one family for — KEYS, the prompted-only leg's, + every SINGLE-PHASE key (the demo: no prompted-only
#: family, never a KEYS member) the production roster names a phase view of, i.e. whose per-view padded configs exist
#: (`pad_static_<key>_<view>_p<pad>`). KEYS keeps its prompted-only meaning; this tuple == KEYS whenever no single-phase key has a
#: roster-named view (the paper's trees). One spelling, read by the grader — never a key retyped there. Bound to
#: output_keys() and NOT to a truthy families_of over FRONT_DOOR_KEYS: a phased key's families_of is (None, …) whether or not the roster
#: names its u0, so that test admitted a pending phased key the generator renders nothing for — and a stale
#: config of a key the roster had dropped could pass both of the grader's gates and be graded against a view the roster no longer serves.
#: The KEY's admission only: the grader then holds the config's (key, family) PAIR to families_of(key) — an admitted single-phase key is
#: rendered its per-view families alone, so its stale prompted-only / obj config is refused there.
GRADER_TARGETS = output_keys()


def front_door_view_pending() -> tuple:
    """The (key, view) pairs of the front door's steered families the production roster does not carry yet (not rendered, not refused)."""
    return tuple((k, v) for k, fams in FRONT_DOOR_VIEW_FAMILIES.items() for v in fams if not _roster_names(k, v))


def _require_front_door_family(key: str, family: str) -> None:
    """A per-view family renders for its OWN key only, once the production roster names the view — refused by name otherwise."""
    if key not in FRONT_DOOR_VIEW_FAMILIES:
        raise SystemExit(f"{key}: not a front-door key of {MODELS_TABLE.name} ({sorted(FRONT_DOOR_VIEW_FAMILIES)}); the per-view steered family "
                         f"{family!r} is the front door's (module docstring) — a hand-listed key's steered families are {list(FAMILIES[1:])}")
    why = front_door_view_excluded(key, family)
    if why is not None:
        raise SystemExit(f"{key}/{family}: the padding family is EXCLUDED for this view by ruling ({why}) — not rendered, not planned (D889)")
    if family not in FRONT_DOOR_VIEW_FAMILIES[key]:
        raise SystemExit(f"{key}: {family!r} is not a steered phase view of this key in {MODELS_TABLE.name} ({list(FRONT_DOOR_VIEW_FAMILIES[key])})")
    if not _roster_names(key, family):
        raise SystemExit(f"{key}/{family}: the production roster {PRODUCTION_ROSTER.name} does not carry this view yet — 05_stage_roster.sh --step "
                         "roster pastes the block, --step configs renders the wave pair; its padded family renders once it does")

_CELLS_BLOCK = re.compile(r"^explicit_cells:\n(?:- .*\n)+", re.M)
_MANIFEST_CELLS = re.compile(r"^  cells:\n(?:    - .*\n)+", re.M)
_OUT_DIR = re.compile(r"^out_dir: .*\n", re.M)
_DEFENSES_DIR = re.compile(r"^defenses_dir: .*\n", re.M)
_PRIVATE_RUN = re.compile(r"^private_run: .*\n", re.M)
_STATIC_CAP = re.compile(r"^  static_max_new_tokens: (\d+)\n", re.M)


def _source_stem(key: str, family: "str | None" = None) -> str:
    """The wave pair's stem: the bare key for a hand-listed key (its campaign `main`; every hand-listed family is cut from it), `<key>_u0`
    for a front-door key's prompted-only family, `<key>_<view>` for a front door's PER-VIEW steered family (its view's own pair)."""
    if is_front_door_family(family):
        return f"{key}_{family}"
    v = source_view(key)
    return key if v == "main" else f"{key}_{v}"


def campaign_config(key: str, family: "str | None" = None) -> Path:
    return REPO / "configs" / f"sdfbase_full22_{_source_stem(key, family)}.yaml"


def campaign_manifest(key: str, family: "str | None" = None) -> Path:
    # the SOURCE is always the repo's committed campaign manifest; MANIFEST_DIR is where the OUTPUT lands (a test redirects it)
    return REPO / "configs" / "capeval" / "manifests" / f"petri_full22_{_source_stem(key, family)}.yaml"


def _fam(family: str | None) -> str:
    """The name segment a family contributes (`_obj`, `_obj_s2`, …), empty for the prompted-only family (whose names are unchanged)."""
    if family is None:
        return ""
    if family not in ALL_FAMILY_NAMES:
        raise ValueError(f"unknown padded-static config family {family!r} (the families are {list(FAMILIES)}; the front door's per-view "
                         f"families {list(ALL_FAMILY_NAMES[len(FAMILIES):])})")
    return f"_{family}"


def seg_of(family: str | None) -> str:
    """:func:`_fam` for readers OUTSIDE this module. The grading side's shell driver asks the generator for the segment
    (``$PY -c 'import make_padding_configs as gen; print(gen.seg_of(fam))'``) instead of carrying a case arm per family, so a
    family this generator gains needs no edit there; an unknown family raises :func:`_fam`'s ValueError, naming the families."""
    return _fam(family)


def seed_of(family: str | None) -> int | None:
    """The training seed a SEED-REPLICATE family serves (`obj_s2` -> 2); None for the prompted-only and the seed-1 steered family."""
    _fam(family)
    if family in OBJ_SEED_FAMILIES:
        return int(family[len(OBJ_FAMILY) + 2:])
    return None


def view_of(family: str) -> str:
    """The wave-bank VIEW a steered family's configs serve from: `padobj` (seed 1), `padobj_s<N>` (a seed replicate), the PRODUCTION
    view itself for a front door's per-view family (`s2` -> `s2`)."""
    n = seed_of(family)
    if family is None:
        raise ValueError("the prompted-only family serves no bank view")
    if is_front_door_family(family):
        return family
    return OBJ_VIEW if n is None else f"{OBJ_VIEW}_s{n}"


def padded_config(key: str, pad: int, family: str | None = None) -> Path:
    return CONFIG_DIR / f"sdfbase_full22_{key}.padstatic{_fam(family)}_p{int(pad)}.yaml"


def padded_manifest(key: str, family: str | None = None) -> Path:
    return MANIFEST_DIR / f"petri_full22_{key}.padstatic{_fam(family)}.yaml"


def padded_roster(key: str, family: str = OBJ_FAMILY) -> Path:
    """A steered family's roster — the one file F22_ROSTER names for its view: the LANE-OWNED roster of `padobj` / `padobj_s<N>` (module
    docstring), the PRODUCTION roster for a front door's per-view family (its view lives there; no lane roster, none generated)."""
    if is_front_door_family(family):
        return PRODUCTION_ROSTER
    return CONFIG_DIR / f"sdfbase_full22_{key}.{view_of(family)}_roster.yaml"


def private_run(key: str, pad: int, family: str | None = None) -> str:
    return f"pad_static_{key}{_fam(family)}_p{int(pad)}"


def cells_of(family: str | None, key: str | None = None) -> tuple:
    """The explicit_cells a family's configs list (the prompted-only three, the steered ten, a seed replicate's six) — or, for a front
    door's PER-VIEW family, `none` + the PRODUCTION roster's cells of that view of `key` in roster order (f22_roster.served_cells: the
    prompted baselines where the roster lists them, every object cell); the key is REQUIRED there (the cells are the roster's, not the
    name's) and a view the roster does not carry refuses by name."""
    _fam(family)
    if family is None:
        return CELLS
    if is_front_door_family(family):
        if key is None:
            raise ValueError(f"cells_of({family!r}): a front door's per-view family lists its ROSTER's cells — pass the key (cells_of(family, key))")
        _require_front_door_family(key, family)
        from f22_roster import load_roster, view_cell_universe
        return tuple(view_cell_universe(load_roster(PRODUCTION_ROSTER), key, family))   # s3r-fix: the ONE universe (no none for a steered-wave view)
    return OBJ_SEED_CELLS if family in OBJ_SEED_FAMILIES else OBJ_CELLS


def pads_of(family: str | None) -> tuple:
    """The padding levels a family's configs exist at: the same five for every family since 2026-09-24 (PADS == OBJ_PADS — the
    prompted-only family's two added rungs are the REFERENCE standups the registry files the steered rungs on; module docstring).
    The ONE spelling every reader of a level list uses (the grader refuses a level outside it); kept per family so a family whose
    ladder ever differs changes one line here and nowhere else."""
    return OBJ_PADS if _fam(family) else PADS


#: `sdfbase_full22_<key>.padstatic[_<family>]_p<pad>.yaml` — the family segment non-greedy so a per-view family carrying `_p` inside its
#: name (`d0p75_relu`) parses to the family and the level (`_p<pad>` is the LAST such segment); the key ends at `.padstatic`
_GENERATED_NAME = re.compile(r"sdfbase_full22_([a-z0-9_]+)\.padstatic(?:_([a-z][a-z0-9_]*?))?_p(\d+)\.yaml")


def _parse_name(path: Path) -> re.Match:
    m = _GENERATED_NAME.fullmatch(Path(path).name)
    if not m:
        raise ValueError(f"{path} is not a generated padded-static config name "
                         f"(sdfbase_full22_<key>.padstatic[_{OBJ_FAMILY}|_<phase view>]_p<pad>.yaml)")
    if m.group(2) is not None and m.group(2) not in ALL_FAMILY_NAMES:
        raise ValueError(f"{path} names a padded-static config family {m.group(2)!r} this generator does not know ({list(FAMILIES[1:])} or "
                         f"the front door's per-view families {list(ALL_FAMILY_NAMES[len(FAMILIES):])})")
    return m


def pad_of(path: Path) -> int:
    """The padding level a generated config's NAME carries (`..._p<pad>.yaml`); refuses any other name."""
    return int(_parse_name(path).group(3))


def key_of(path: Path) -> str:
    """The geometry-table model key a generated config's NAME carries (`sdfbase_full22_<key>.padstatic[_obj]_p<pad>.yaml`) —
    what the lane's preflight derives the standup's S for; refuses any other name."""
    return _parse_name(path).group(1)


def family_of(path: Path) -> str | None:
    """The config family a generated config's NAME carries: None (the prompted-only three), `obj` (the steered ten), a seed
    replicate's `obj_s<N>` (its six), or a front door's per-view family (a phase view name; its cells are the roster's — cells_of(family, key))."""
    return _parse_name(path).group(2)


def harness_cells(cells=CELLS) -> list[str]:
    from sisteer.capeval.petri import registry_adapter as ra
    return [ra.harness_cell_id(c) for c in cells]


def padding_block(pad: int) -> dict:
    """The `attacks.static_padding` block a padded config pins (jailbreak_run._STATIC_PADDING_CFG_KEYS; the filler
    path is the argv's, the tolerance the library default — neither is restated here)."""
    return {"tokens": int(pad), "placement": PLACEMENT, "mode": MODE, "reserve": RESERVE}


def _one(pattern: re.Pattern, text: str, what: str, path: Path):
    ms = list(pattern.finditer(text))
    if len(ms) != 1:
        raise SystemExit(f"{path}: expected exactly one {what} block, found {len(ms)}")
    return ms[0]


def render_config(key: str, pad: int, family: str | None = None) -> str:
    fd = is_front_door_family(family)
    if family is not None and not fd and key not in HAND_LISTED_KEYS:
        raise SystemExit(f"{key}: the steered family {family!r} is bound to the hand-listed keys' s1 objects and cross-bank pins "
                         f"({list(HAND_LISTED_KEYS)}); a front-door key pads its steered views through its PER-VIEW families "
                         f"({list(FRONT_DOOR_VIEW_FAMILIES.get(key, ()))}; module docstring, D51)")
    if fd:
        _require_front_door_family(key, family)
    src_path = campaign_config(key, family)
    if not src_path.is_file():
        raise SystemExit(f"{key}: no wave config at {src_path} — 05_stage_roster.sh --step configs renders it (make_full22_configs.py) once the "
                         f"roster carries {key}/{family if fd else source_view(key)}")
    src = src_path.read_text(encoding="utf-8")
    cells = cells_of(family, key)
    m_cells = _one(_CELLS_BLOCK, src, "explicit_cells", src_path)
    listed = [l[2:].strip() for l in m_cells.group(0).splitlines()[1:]]
    if family is None:
        missing = [c for c in cells if c != "none" and c not in listed and not c.startswith("prompted__")]
        if missing:
            raise SystemExit(f"{src_path}: the campaign config does not list {missing}")
    elif fd:
        # the per-view family IS the view's cells: the wave config's explicit_cells and the roster's served cells must agree exactly
        # (make_full22_configs renders the former from the latter; a drift means a stale wave pair — regenerate, never pad it)
        if listed != list(cells):
            raise SystemExit(f"{src_path}: explicit_cells {listed} are not the production roster's {key}/{family} cells {list(cells)} — the wave pair "
                             "is stale (05_stage_roster.sh --step configs regenerates it); refusing to pad a view whose cells drifted")
    else:
        # the steered ten (or a seed replicate's six) are the LANE ROSTER's cells (render_roster reads the production views +
        # the pin files and refuses any disagreement); the campaign `main` config lists other objects, so no containment check
        _obj_roster_cells(key, family)
    m_cap = _one(_STATIC_CAP, src, "attacks.static_max_new_tokens", src_path)
    campaign_cap = int(m_cap.group(1))
    run = private_run(key, pad, family)
    cap_note = (
        f"  # PADDED LEG (docs/static_padding.md): the cap is the static ladder's CEILING ({STATIC_CAP:,}; ladder_decide.CEILING), not\n"
        f"  # the campaign's {campaign_cap:,} the comment above describes — at EVERY level, the p0 control included, so padded and unpadded\n"
        f"  # are contrasted at ONE cap. WHY: this leg is PREFILL-bound (a hybrid target with no prefix caching re-prefills the whole\n"
        f"  # pad on every request) and a cap costs only for tokens actually generated, so the ceiling's marginal price is the decode\n"
        f"  # of the items that would have hit {campaign_cap:,} — while an escalation rung would re-pay the FULL pad prefill for every\n"
        f"  # re-drawn item, and the rung driver (escalate_static.assert_base_unpadded) refuses a padded base outright, so a >1 % at-cap\n"
        f"  # share could only be met by regenerating the whole level. The ceiling cap makes the leg single-pass: run once, never climb.\n"
    )
    block_note = (
        f"# STATIC PADDING (docs/static_padding.md): the store's pinned contract — {pad:,} neutral PG-19 filler tokens ADDED to\n"
        f"# every item's user turn, placement `{PLACEMENT}` (between a prompted cell's system instruction and the payload; a long\n"
        f"# prefix for the bare anchor), mode `{MODE}` (one filler text, grown with the target). The filler PATH rides the sbatch's\n"
        f"# argv (--static-padding-filler, sha-verified first) and the tolerance is the library default (2 %); neither is restated\n"
        f"# here. reserve {RESERVE:,}: the window preflight's headroom for SI + payload + template (the constitution SI alone renders\n"
        f"# to 5,973 tokens; the library default 2,048 would admit a run the server then refuses). One contract per store: the\n"
        f"# harness refuses to mix padded / unpadded rows or two blocks in one responses file — hence this config's own private_run.\n"
    ) if pad > 0 else ""
    block_yaml = ("".join(f"  {ln}\n" for ln in yaml.safe_dump({"static_padding": padding_block(pad)}, sort_keys=False).splitlines())
                  if pad > 0 else "")
    if fd:
        cells_sentence = (f"`explicit_cells` = EXACTLY the wave config's — `none` + the production roster's {key}/{family} cells {list(cells[1:])}\n"
                          f"# (the FRONT DOOR's PER-VIEW steered family: the view's own objects, the prompted baselines where\n"
                          f"# the roster lists them; no cross-bank pin, no lane roster); `defenses_dir` = `out/full22/{key}/cells_view/{family}`, the\n"
                          f"# PRODUCTION bank view of that phase view (05_stage_roster.sh --step bank-views builds it; on the cluster the\n"
                          f"# TT_TWO_SI_DEFENSES_DIR export of common.sh f22_bank_view)")
        served_by = (f"# Served by scripts/sdfbase/full22/static_padding/target_static_padded_obj.sbatch with its config-family positional `{family}`\n"
                     f"# (stage 06's padding rows for view `{family}`, lib/stage.py; the sbatch asks this generator `--resolve-family {key} {family}` for the\n"
                     f"# view, the roster and the registry view): ONE JOB PER PADDING LEVEL, ONE family (grok420) at attacks.static_max_new_tokens\n"
                     f"# ({STATIC_CAP:,} — the static ladder's CEILING, raised from the campaign's {campaign_cap:,}: the leg is prefill-bound and the rung\n"
                     f"# driver refuses padded stores, so a ceiling cap is what makes it single-pass; the key's own note below) over the 259-goal\n"
                     f"# confirmatory pool, decoded at the standing vendor-preset default (no sampled_decode block, no --greedy). Every level stands\n"
                     f"# up at the same window (98,304); the served width is the preflight's DERIVATION for this model (no literal pin: the\n"
                     f"# seed-1 wave's S = 5 is qwen35_4b's), held by the post-standup GEOMETRY PIN's derivation checks.\n")
    elif family is None:
        cells_sentence = (f"`explicit_cells` = exactly the three prompted-side arms {list(CELLS)}\n"
                          f"# (the same-wave bare anchor + the two prompted SI baselines; no fitted object)")
        served_by = (f"# Served by scripts/sdfbase/full22/static_padding/target_static_padded.sbatch: ONE standup for every level (the pad is request\n"
                     f"# content, not server state), the four static families at attacks.static_max_new_tokens ({STATIC_CAP:,} — the static ladder's\n"
                     f"# CEILING, raised from the campaign's {campaign_cap:,}: the leg is prefill-bound and the rung driver refuses padded stores, so a\n"
                     f"# ceiling cap is what makes it single-pass; the key's own note below) over the 259-goal confirmatory pool, decoded at the\n"
                     f"# standing vendor-preset default (no sampled_decode block, no --greedy).\n")
    elif family in OBJ_SEED_FAMILIES:
        n = seed_of(family)
        cells_sentence = (f"`explicit_cells` = the SEED-{n} REPLICATE's SEVEN cells {list(OBJ_SEED_CELLS)}\n"
                          f"# (the same-wave bare anchor `none`, then the seed-1 steered family's six FITTED objects at training seed {n}: the four\n"
                          f"# base64s{n} fits, the difference LoRA derived_base64s{n}, the FULL-dose paired write paired_eb64s{n} — the same cell ids,\n"
                          f"# served from the seed-{n} objects; NO prompted baseline, NO Arditi rung: neither has a training seed; `none` has none\n"
                          f"# either but is REQUIRED — see SEED REPLICATE below);\n"
                          f"# `defenses_dir` = `out/full22/{key}/cells_view/{view_of(family)}`, the wave-bank view the lane roster\n"
                          f"# {_rel(padded_roster(key, family))} names (six same-bank staged ids read from the production roster's `s{n}` and\n"
                          f"# `s{n}p` views, pinned in configs/sdfbase_full22_pins.json; on the cluster the TT_TWO_SI_DEFENSES_DIR export of\n"
                          f"# common.sh f22_bank_view)")
        served_by = (f"# Served by scripts/sdfbase/full22/static_padding/target_static_padded_obj.sbatch on cluster A with its config-family\n"
                     f"# positional `{family}`: ONE family (grok420) at attacks.static_max_new_tokens ({STATIC_CAP:,} — the static ladder's CEILING,\n"
                     f"# raised from the campaign's {campaign_cap:,}: the leg is prefill-bound and the rung driver refuses padded stores, so a ceiling\n"
                     f"# cap is what makes it single-pass; the key's own note below) over the 259-goal confirmatory pool, decoded at the standing\n"
                     f"# vendor-preset default (no sampled_decode block, no --greedy). Every job stands up at the seed-1 wave's window (98,304) with\n"
                     f"# the GEOMETRY PIN, so every rung of every seed sits on one engine identity (weights, adapters, S x L) — the INSTRUMENT is\n"
                     f"# the seed-1 wave's, unchanged: window 98,304, cap {STATIC_CAP:,}, vendor preset, trial 0, five rungs, grok420.\n")
    else:
        cells_sentence = (f"`explicit_cells` = the STEERED family's TEN cells {list(OBJ_CELLS)}\n"
                          f"# (the same-wave bare anchor, the two prompted SI baselines, the five fitted s1 objects of the production roster's `{OBJ_SOURCE_VIEW}`\n"
                          f"# view, the all-positions Arditi rung c = 0.3 under its harness id `{OBJ_ARDITI_CELL}`, and the paired write\n"
                          f"# `{OBJ_PAIRED_CELL}` at full dose, seed 1 — never the half-dose view or a *_distilled object);\n"
                          f"# `defenses_dir` = `out/full22/{key}/cells_view/{OBJ_VIEW}`, the wave-bank view the lane roster\n"
                          f"# {_rel(padded_roster(key))} names (five same-bank staged ids + two CROSS-BANK PINS, no bytes copied;\n"
                          f"# on the cluster the TT_TWO_SI_DEFENSES_DIR export of common.sh f22_bank_view)")
        served_by = (f"# Served by scripts/sdfbase/full22/static_padding/target_static_padded_obj.sbatch on cluster A: ONE JOB PER PADDING LEVEL\n"
                     f"# (the 2026-09-23 shape), ONE family (grok420) at attacks.static_max_new_tokens ({STATIC_CAP:,} — the static ladder's\n"
                     f"# CEILING, raised from the campaign's {campaign_cap:,}: the leg is prefill-bound and the rung driver refuses padded stores, so a\n"
                     f"# ceiling cap is what makes it single-pass; the key's own note below) over the 259-goal confirmatory pool, decoded at the\n"
                     f"# standing vendor-preset default (no sampled_decode block, no --greedy). Every job stands up at the same window with the\n"
                     f"# GEOMETRY PIN, so the three levels sit on one engine identity (weights, adapters, S x L) across three instances.\n")
    added_note = (
        f"# ADDED RUNG (2026-09-24): {pad:,} joins the filed 0 / 16,384 / 65,536 (with its sibling "
        f"{[p for p in OBJ_PADS_ADDED if p != pad][0]:,}) at the SAME window, 98,304, and the same derived S = 5, so the five\n"
        f"# levels are ONE comparable series on one instrument — never two windows, never two widths. Same generator, same campaign\n"
        f"# body, same cells, same cap, same padding contract; only the level differs. Its JOB SHAPE differs from the filed three: the\n"
        f"# GOAL SET is split across several jobs (target_static_padded_obj.sbatch's `shards` positional, `<lo>-<hi>/N`), each with\n"
        f"# its own standup on its own GPU, every job serving all ten cells for its disjoint shard range, all writing shard00..shard<N-1>\n"
        f"# of THIS store — the layout one job at N would write, so the grader and the degeneracy writer read it unchanged (--nshards N).\n"
    ) if family == OBJ_FAMILY and pad in OBJ_PADS_ADDED else ""
    ref_note = (
        f"# REFERENCE STANDUP OF AN ADDED RUNG (2026-09-24): {pad:,} joins the prompted-only family's filed 0 / 16,384 / 65,536 (with its\n"
        f"# sibling {[p for p in OBJ_PADS_ADDED if p != pad][0]:,}) because the results registry (src/sisteer/registry/adapters/static_padding.py)\n"
        f"# files every steered store pad_static_{key}_obj_p<PAD> on the instrument the prompted-only REFERENCE standup pad_static_{key}_p<PAD>\n"
        f"# mints at the SAME pad — the pad and the filler contract make the identity, the objects never do — and refuses a steered store with\n"
        f"# no reference at its pad (reference.state 'absent'). The steered rungs at 8,192 / 32,768 (generated, graded, mirrored) filed ZERO\n"
        f"# rows until this standup ran; the seed-2 / seed-3 replicates at this pad read the same reference (its cells are unfitted, hence\n"
        f"# seed-independent). The reference is run rather than the rule changed. THE INSTRUMENT IS THE\n"
        f"# FILED RUNGS': window 98,304, cap {STATIC_CAP:,}, the vendor-preset default, the same filler contract (tolerance 0.02, separator, mode,\n"
        f"# placement); the batch width S is not an identity field, and the GEOMETRY PIN holds it to the steered arms' S = 5 regardless. JOB\n"
        f"# SHAPE: target_static_padded.sbatch on cluster A (--qos=high and the shared-storage log path on the sbatch line), ONE job for this\n"
        f"# level, grok420 only, the filed rungs' three shards (shard00..shard02 — the layout the grader reads with --nshards 3).\n"
    ) if family is None and pad in OBJ_PADS_ADDED else ""
    seed_note = (
        f"# SEED REPLICATE (2026-09-24): training seed {seed_of(family)} of the six fitted objects, beside the seed-1 family's\n"
        f"# stores (pad_static_{key}_obj_p<pad>), on the IDENTICAL instrument. Two deliberate choices, recorded so a later reader does not\n"
        f"# undo them. (1) THE DECODE SEED BASE IS HELD AT {DECODE_SEED_BASE_HELD} — the vendor preset's campaign base the seed-1 wave decoded\n"
        f"# under, taken by the driver when no seed flag is passed (the sbatch passes none) — and is NOT set to the training seed: the\n"
        f"# seedrep contract ties decode seed to training seed on its own lane's views, but here that would vary the object seed AND the\n"
        f"# decode seed together and confound them; holding the decode fixed isolates object-seed variation, which is the question.\n"
        f"# (2) `none` IS LISTED although it has no training seed. The user's question was: reuse the seed-1 wave's `none` if the existing\n"
        f"# code allows, otherwise run it — and it does not allow: the native standup derives its zero-adapter leg none__lora0 FROM the\n"
        f"# `none` cell, registers the zero adapter from that leg, and sizes the LoRA kernel index ceiling over the REGISTERED adapters. A\n"
        f"# six-cell set without `none` refused every standup at 1-5 min (\"enumerate_cells derived no none__lora0 leg\", 29 jobs, 2026-09-24),\n"
        f"# and the geometry it printed before refusing was the IDENTITY S = 7 — the lowering to the seed-1 wave's S = 5 had not run. So\n"
        f"# `none` is what puts the zero adapter, and with it the seed-1 width, on this engine; it is also this store's within-standup\n"
        f"# anchor for the grader. THE WIDTH IS PINNED: the sbatch holds every standup to S = {GEOMETRY_PIN_S} by a literal (GEOMETRY_PIN_S), in the\n"
        f"# preflight and in the post-standup GEOMETRY PIN (attested S, the module's derivation and a kernel-ceiling step in the attested\n"
        f"# chain must all say {GEOMETRY_PIN_S}), so a standup at the identity 7 can never reach a store. JOB SHAPE: every rung runs SHARDED\n"
        f"# across jobs (target_static_padded_obj.sbatch's `<lo>-<hi>/N` positional), each job serving all seven cells for its disjoint\n"
        f"# shard range, so the store is shard00..shard<N-1> of THIS run — the layout one job at N would write (the grader reads --nshards N).\n"
    ) if family in OBJ_SEED_FAMILIES else ""
    fd_note = (
        f"# FRONT DOOR, PER-VIEW STEERED FAMILY `{family}` (paper-companion-repo, 2026-09-29): the test model's steered\n"
        f"# phase views are padded ONE family per view, named after the view, derived from the ROSTER's own cells — never from qwen35_4b's\n"
        f"# `obj` family (bound to ITS s1 objects and two cross-bank pin files). Cut from the view's own wave pair configs/{campaign_config(key, family).name}\n"
        f"# / {campaign_manifest(key, family).name}: the same overlay as every other family (out_dir / private_run, the ceiling cap, the\n"
        f"# padding block), so the store `{run}` sits beside the prompted-only reference `pad_static_{key}_p{pad}` at the same pad — the\n"
        f"# results registry files a steered store on the instrument its reference standup mints (adapters/static_padding.py).\n"
    ) if fd else ""
    header = f"""# {padded_config(key, pad, family).name} — the PADDED STATIC leg's {'CONTROL' if pad == 0 else f'{pad:,}-token'} config for {key}{' (STEERED family)' if family == OBJ_FAMILY else ''}{f' (STEERED family, SEED-{seed_of(family)} REPLICATE)' if family in OBJ_SEED_FAMILIES else ''}{f' (STEERED per-view family `{family}`)' if fd else ''}: an OVERLAY of
# configs/{campaign_config(key, family).name} (docs/static_padding.md). GENERATED by scripts/sdfbase/full22/static_padding/
# make_padding_configs.py — never hand-edited (tests/test_static_padding_wave.py requires byte-identity). The campaign
# config's body follows BYTE FOR BYTE except: {cells_sentence}; `out_dir` / `private_run` = `{run}`
# (the store identity: ONE padding contract per store, so each level is its own run and this control is a REAL run of
# this wave, never a reused older store); `attacks.static_max_new_tokens` = {STATIC_CAP:,} (the ladder's ceiling; its note below){'; and the `attacks.static_padding` block below, which PINS the pad level to the config so no driver flag can run it under another level' if pad > 0 else ' — and NO padding block: 0 = off, every byte the static path writes is as before'}.
{served_by}{added_note}{ref_note}{seed_note}{fd_note}#
"""
    body = src[:m_cells.start()] + "explicit_cells:\n" + "".join(f"- {c}\n" for c in cells) + src[m_cells.end():]
    body, n_out = _OUT_DIR.subn(f"out_dir: out/{run}\n", body, count=1)
    body, n_run = _PRIVATE_RUN.subn(f"private_run: {run}\n", body, count=1)
    if n_out != 1 or n_run != 1:
        raise SystemExit(f"{src_path}: no top-level out_dir / private_run line to rename")
    if family is not None:
        body, n_dd = _DEFENSES_DIR.subn(f"defenses_dir: out/full22/{key}/cells_view/{view_of(family)}\n", body, count=1)
        if n_dd != 1:
            raise SystemExit(f"{src_path}: no top-level defenses_dir line to point at the {view_of(family)} view")
    # the cap: the campaign's line becomes the ceiling, with the note above it (ONE line still matches _STATIC_CAP, so the
    # padding block below lands right after it)
    body, n_cap = _STATIC_CAP.subn(lambda _m: cap_note + f"  static_max_new_tokens: {STATIC_CAP}\n", body, count=1)
    if n_cap != 1:
        raise SystemExit(f"{src_path}: no attacks.static_max_new_tokens line to raise to the ceiling")
    if pad > 0:
        m_cap2 = _one(_STATIC_CAP, body, "attacks.static_max_new_tokens", src_path)
        body = body[:m_cap2.end()] + block_note + block_yaml + body[m_cap2.end():]
    out = header + body
    _verify_config_as_data(src_path, src, out, key, pad, family)
    return out


def _verify_config_as_data(src_path: Path, src: str, out: str, key: str, pad: int, family: str | None = None) -> None:
    """The output IS the campaign config as data except the named keys — the generator loads its own output and
    refuses to write otherwise (make_sampled_configs' rule)."""
    want = yaml.safe_load(src)
    want["explicit_cells"] = list(cells_of(family, key))
    want["out_dir"] = f"out/{private_run(key, pad, family)}"
    want["private_run"] = private_run(key, pad, family)
    if family is not None:
        want["defenses_dir"] = f"out/full22/{key}/cells_view/{view_of(family)}"
    want["attacks"] = dict(want["attacks"])
    want["attacks"]["static_max_new_tokens"] = STATIC_CAP
    if pad > 0:
        want["attacks"]["static_padding"] = padding_block(pad)
    got = yaml.safe_load(out)
    if got != want:
        diff = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
        raise SystemExit(f"{padded_config(key, pad, family)}: the rendered text is not {src_path} as data except the named keys (keys differing: {diff})")


def render_manifest(key: str, family: str | None = None) -> str:
    fd = is_front_door_family(family)
    if family is not None and not fd and key not in HAND_LISTED_KEYS:
        raise SystemExit(f"{key}: the steered family {family!r} is bound to the hand-listed keys ({list(HAND_LISTED_KEYS)}); a front-door key "
                         f"pads its steered views through its PER-VIEW families ({list(FRONT_DOOR_VIEW_FAMILIES.get(key, ()))}; module docstring, D51)")
    if fd:
        _require_front_door_family(key, family)
    src_path = campaign_manifest(key, family)
    if not src_path.is_file():
        raise SystemExit(f"{key}: no wave manifest at {src_path} — 05_stage_roster.sh --step configs renders it once the roster carries "
                         f"{key}/{family if fd else source_view(key)}")
    src = src_path.read_text(encoding="utf-8")
    m = _one(_MANIFEST_CELLS, src, "cells", src_path)
    ids = harness_cells(cells_of(family, key))
    what = ("the three prompted-side arms" if family is None
            else f"the FRONT DOOR's per-view steered family `{family}` (none + the production roster's {key}/{family} cells — the view's own)"
            if fd
            else f"the SEED-{seed_of(family)} REPLICATE's seven cells (none, then the seed-1 family's six fitted objects at training seed {seed_of(family)}; no prompted, no Arditi)"
            if family in OBJ_SEED_FAMILIES
            else f"the STEERED family's ten cells (five fitted s1 objects, the Arditi rung `{OBJ_ARDITI_CELL}`, the paired write, the prompted pair, none)")
    header = (f"# {padded_manifest(key, family).name} — the PADDED STATIC leg's OVERLAY of {campaign_manifest(key, family).name} (docs/static_padding.md): the\n"
              f"# campaign manifest BYTE FOR BYTE except `cells`, which names EXACTLY the harness ids of {what} of\n"
              f"# configs/sdfbase/static_padding/sdfbase_full22_{key}.padstatic{_fam(family)}_p<pad>.yaml's explicit_cells (mt_serving_up.py refuses a\n"
              f"# config / manifest cell mismatch). ONE manifest for every padding level: the pad is request content, not a served cell.\n"
              f"# GENERATED by scripts/sdfbase/full22/static_padding/make_padding_configs.py — never hand-edited (byte-identity is tested).\n")
    out = header + src[:m.start()] + "  cells:\n" + "".join(f"    - {c}\n" for c in ids) + src[m.end():]
    want = yaml.safe_load(src)
    want["petri"] = dict(want["petri"])
    want["petri"]["cells"] = ids
    if yaml.safe_load(out) != want:
        raise SystemExit(f"{padded_manifest(key, family)}: the rendered text is not {src_path} as data except `cells`")
    return out


# --------------------------------------------------------------------------------------- the steered family's roster

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_roster_cells(key: str, family: str, prod: dict, wave_pins: dict) -> dict:
    """``{cell id: staged id}`` for a SEED REPLICATE's ``padobj_s<N>`` view (module docstring), every entry in the STRING form
    and READ from the production roster: the five fitted cells from its ``s<N>`` view, the paired write from its ``s<N>p`` view
    (the wave bank's own staged copies: base64s<N>__*, derived_base64s<N>__*, paired_eb64s<N>__*). Each staged id must carry
    the seed (`s<N>__`), be pinned in the wave pins with an object_sha256 and `verified_cluster_a` (the wave runs there), and the
    paired one must be the FULL-dose object the ``s<N>p`` view serves (never a half-dose ``d0p5`` / ``halfdose`` id, never a
    ``*_distilled`` one). Refuses by name."""
    from f22_roster import staged_cells

    n = seed_of(family)
    fitted = staged_cells(prod, key, f"{OBJ_SOURCE_VIEW[:-1]}{n}")           # the production `s<N>` view (`s1` -> `s<N>`)
    paired = staged_cells(prod, key, f"{OBJ_SOURCE_VIEW[:-1]}{n}p")          # the production `s<N>p` view
    out: dict = {}
    for c in OBJ_SEED_CELLS:
        if c == "none":
            continue                                   # the bare anchor is a config cell, never a roster (bank) cell
        src_view, sid = (f"s{n}p", paired.get(c)) if c == OBJ_PAIRED_CELL else (f"s{n}", fitted.get(c))
        if not isinstance(sid, str):
            raise SystemExit(f"{PRODUCTION_ROSTER.name}: view {key}/{src_view} does not map {c} to a staged id (got {sid!r}) — the seed-{n} "
                             "replicate reads its objects from the production views, never retypes them")
        if f"s{n}__" not in sid:
            raise SystemExit(f"{c}: the production {key}/{src_view} view maps it to {sid!r}, which does not carry seed {n} (`s{n}__`)")
        if any(bad in sid for bad in ("d0p5", "halfdose", "distilled")):
            raise SystemExit(f"{c}: staged id {sid!r} is a half-dose or distilled object — OUT by the user's ruling (full dose only)")
        pin = wave_pins.get(sid) or {}
        if not pin.get("object_sha256"):
            raise SystemExit(f"{WAVE_PINS.name}: no object_sha256 pin for {key}/{sid} ({c}) — refusing to name an unpinned object")
        if pin.get("verified_cluster_a") is not True:
            raise SystemExit(f"{WAVE_PINS.name}: {key}/{sid} is not verified on cluster A, where this wave runs")
        out[c] = sid
    return out


def _obj_roster_cells(key: str, family: str = OBJ_FAMILY) -> dict:
    """``{cell id: staged id | None | {pins, pin}}`` for the ``padobj`` view, in OBJ_CELLS order minus ``none`` — READ from
    the authorities, never retyped: the five s1 staged ids from the production roster's ``s1`` view (each pinned in the
    wave pins), the two prompted baselines as ``null``, the Arditi rung and the paired write as CROSS-BANK PIN references
    whose entries must exist, carry the builder's identity fields, and whose ``kind`` is the arm's (ARM_OBJECT_KINDS).
    The paired pin's bytes are held equal to the wave pins' own staged copy of the object (OBJ_PAIRED_WAVE_STAGED_ID), so
    the lane can never serve a paired object other than the one the production ``s1p`` view serves. Refuses by name.
    For a SEED REPLICATE family the six string-form entries of :func:`_seed_roster_cells`, under the same production-view guard."""
    from f22_roster import PIN_FORM_KEYS, is_prompted, load_roster, staged_cells
    from sisteer.capeval.petri.registry_adapter import ARM_OBJECT_KINDS
    from sisteer.eval.defense_cell import arm_of

    view = view_of(family)
    prod = load_roster(PRODUCTION_ROSTER)
    if view in (prod["targets"][key].get("views") or {}):
        raise SystemExit(f"{PRODUCTION_ROSTER.name} names a view {key}/{view} — the lane's view must be its own (f22_bank_view refuses "
                         "a non-production roster that names a production view)")
    wave_pins = _load_json(WAVE_PINS).get(key, {})
    if family in OBJ_SEED_FAMILIES:
        return _seed_roster_cells(key, family, prod, wave_pins)
    s1 = staged_cells(prod, key, OBJ_SOURCE_VIEW)
    out: dict = {}
    for c in OBJ_CELLS:
        if c == "none":
            continue
        if is_prompted(c):
            out[c] = None
            continue
        if c in OBJ_S1_CELLS:
            sid = s1.get(c)
            if not isinstance(sid, str):
                raise SystemExit(f"{PRODUCTION_ROSTER.name}: view {key}/{OBJ_SOURCE_VIEW} does not map {c} to a staged id (got {sid!r})")
            if not (wave_pins.get(sid) or {}).get("object_sha256"):
                raise SystemExit(f"{WAVE_PINS.name}: no object_sha256 pin for {key}/{sid} ({c}) — refusing to name an unpinned object")
            out[c] = sid
            continue
        pins_rel, entry = OBJ_PIN_REFS[c]
        pins_path = REPO / pins_rel
        if not pins_path.is_file():
            raise SystemExit(f"{c}: no pins file at {pins_path}")
        pin = (_load_json(pins_path).get(key) or {}).get(entry)
        if not isinstance(pin, dict):
            raise SystemExit(f"{c}: {pins_rel} has no {key}.{entry} entry")
        absent = [k for k in ("bank", "source", "staged_id", "object_sha256", "kind", "served_layer", "identity_sidecar") if pin.get(k) in (None, "")]
        if absent:
            raise SystemExit(f"{c}: pin {key}.{entry} of {pins_rel} lacks {absent} (make_bank_view.PIN_ENTRY_KEYS)")
        want_kind = ARM_OBJECT_KINDS.get(arm_of(c) or "")
        if want_kind != pin["kind"]:
            raise SystemExit(f"{c}: ARM_OBJECT_KINDS[{arm_of(c)!r}] is {want_kind!r} but pin {key}.{entry} says {pin['kind']!r}")
        if c == OBJ_PAIRED_CELL:
            wave_sha = (wave_pins.get(OBJ_PAIRED_WAVE_STAGED_ID) or {}).get("object_sha256")
            if wave_sha != pin["object_sha256"]:
                raise SystemExit(f"{c}: the cross-bank pin's object_sha256 {str(pin['object_sha256'])[:12]} != the wave pins' "
                                 f"{OBJ_PAIRED_WAVE_STAGED_ID} {str(wave_sha)[:12]} — two spellings of one object must name one set of bytes")
            if pin.get("positions") != "all" or pin.get("scale_matched") is not True:
                raise SystemExit(f"{c}: pin {key}.{entry} is not the full-dose all-positions object (positions {pin.get('positions')!r}, "
                                 f"scale_matched {pin.get('scale_matched')!r})")
        if c == OBJ_ARDITI_CELL and (pin.get("positions") != "all" or pin.get("c") != 0.3):
            raise SystemExit(f"{c}: pin {key}.{entry} is not the all-positions c = 0.3 rung (positions {pin.get('positions')!r}, c {pin.get('c')!r})")
        out[c] = {PIN_FORM_KEYS[0]: pins_rel, PIN_FORM_KEYS[1]: f"{key}.{entry}"}
    return out


def render_roster(key: str, family: str = OBJ_FAMILY) -> str:
    """A steered family's lane roster (module docstring): the production roster's wave-wide identity keys (wave, bank_root,
    staging_manifest) and the target's model pins AS DATA, `lane_flags: {}` (no lane rides this view), and ONE view,
    `padobj` (or a seed replicate's `padobj_s<N>`), whose cells are :func:`_obj_roster_cells`. Loaded back and checked through
    the same readers the jobs use (f22_roster.served_cells / staged_cells) before it is written."""
    from f22_roster import load_roster, served_cells, staged_cells

    view = view_of(family)
    prod = load_roster(PRODUCTION_ROSTER)
    t = prod["targets"][key]
    cells = _obj_roster_cells(key, family)
    doc = {"wave": prod["wave"], "bank_root": prod["bank_root"], "staging_manifest": prod["staging_manifest"],
           "lane_flags": {},
           "targets": {key: {"model": t["model"], "model_revision": t["model_revision"],
                             "views": {view: {"config": _rel(padded_config(key, 0, family)),
                                              "manifest": _rel(padded_manifest(key, family)),
                                              "cells": cells}}}}}
    if family in OBJ_SEED_FAMILIES:
        n = seed_of(family)
        header = (f"# {padded_roster(key, family).name} — the PADDED STATIC leg's STEERED family, SEED-{n} REPLICATE: the LANE-OWNED roster of ONE\n"
                  f"# wave-bank view, `{key}/{view}` (docs/static_padding.md \"Seed replicates\"). GENERATED by scripts/sdfbase/full22/static_padding/\n"
                  f"# make_padding_configs.py — never hand-edited (tests/test_static_padding_wave.py requires byte-identity). NOT a production\n"
                  f"# roster and never a production view: the production roster names no `{view}` view (common.sh f22_bank_view refuses a\n"
                  f"# non-production roster that names a production view — the bank view $SHARED/{prod['bank_root']}/{key}/cells_view/{view}\n"
                  f"# is shared state keyed by name). The sbatch (target_static_padded_obj.sbatch, config-family positional `{family}`) sets\n"
                  f"# F22_ROSTER to this file before f22_init, so every roster reader of the job reads THIS view. It never touches `padobj`\n"
                  f"# (the seed-1 view), s{n}, s{n}p or any peer view.\n"
                  f"#\n"
                  f"# THE CELLS ({len(cells)} served, all staged objects; the configs list them after the same-wave `none` anchor, which the\n"
                  f"# native standup needs to derive its zero-adapter leg — make_padding_configs.py's docstring; no prompted baseline, no Arditi rung):\n"
                  f"#   * the five fitted cells map to the SAME staged ids the production roster's `s{n}` view maps, the paired write to the\n"
                  f"#     SAME staged id its `s{n}p` view maps (read from those views, never retyped; every id carries `s{n}__` and is pinned in\n"
                  f"#     configs/sdfbase_full22_pins.json with verified_cluster_a) — the STRING form, byte for byte the production mechanism:\n"
                  f"#     each view entry is a same-bank symlink ../../cells/<staged id> whose bytes make_bank_view.py re-hashes to the pin.\n"
                  f"#   * the paired write is the FULL-dose seed-{n} object (paired_eb64s{n}__…, the wave bank's own staged copy, which the\n"
                  f"#     bank's paired_d1_s{n} dose view also serves through the relu_paired bank — same sha); the seed-1 family's CROSS-BANK PIN\n"
                  f"#     into exp/objects/paired_write/ has no seed-{n} counterpart (that bank stages paired_s1 only), so the string form is the\n"
                  f"#     authority here. The half-dose views and every *_distilled object are OUT.\n"
                  f"#   * positions `all` for every object (read off each staged copy's identity sidecar on cluster A 2026-09-24 — a\n"
                  f"#     position-restricted fit would be skipped by name by the native standup and silently drop an arm); served layers are\n"
                  f"#     each object's own (the sidecar's, read by the bank-view builder and the standup — no served_layer is retyped here).\n"
                  f"# Wave-wide keys and the target's model pins are the production roster's AS DATA; `lane_flags` is empty because no lane\n"
                  f"# rides this view.\n")
        out = header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=120)
        back = yaml.safe_load(out)
        if back != doc:
            raise SystemExit(f"{padded_roster(key, family)}: the rendered roster does not load back as its data")
        served, staged = served_cells(back, key, view), staged_cells(back, key, view)
        want_served = [c for c in OBJ_SEED_CELLS if c != "none"]
        if list(served) != want_served or len(staged) != len(want_served) or any(not isinstance(v, str) for v in staged.values()):
            raise SystemExit(f"{padded_roster(key, family)}: the roster's served cells {list(served)} are not the family's {want_served}")
        return out
    n_pins = sum(isinstance(v, dict) for v in cells.values())
    header = (f"# {padded_roster(key).name} — the PADDED STATIC leg's STEERED family: the LANE-OWNED roster of ONE wave-bank view,\n"
              f"# `{key}/{OBJ_VIEW}` (docs/static_padding.md). GENERATED by scripts/sdfbase/full22/static_padding/make_padding_configs.py —\n"
              f"# never hand-edited (tests/test_static_padding_wave.py requires byte-identity). NOT a production roster and never a\n"
              f"# production view: the production roster names no `{OBJ_VIEW}` view (common.sh f22_bank_view refuses a non-production\n"
              f"# roster that names a production view — the bank view $SHARED/{prod['bank_root']}/{key}/cells_view/{OBJ_VIEW} is shared\n"
              f"# state keyed by name). The sbatch (target_static_padded_obj.sbatch) sets F22_ROSTER to this file before f22_init, so\n"
              f"# every roster reader of the job (roster_cell_count.py, make_bank_view.py) reads THIS view.\n"
              f"#\n"
              f"# THE CELLS ({len(cells)} served: {len([v for v in cells.values() if v is not None])} staged objects + 2 prompted baselines; the\n"
              f"# configs list them after the same-wave `none` anchor):\n"
              f"#   * the five fitted s1 objects map to the SAME staged ids the production roster's `{OBJ_SOURCE_VIEW}` view maps (read from it,\n"
              f"#     pinned in configs/sdfbase_full22_pins.json) — the string form, byte for byte the production mechanism;\n"
              f"#   * `{OBJ_ARDITI_CELL}` and `{OBJ_PAIRED_CELL}` are named through the CROSS-BANK PIN FORM ({n_pins} entries;\n"
              f"#     f22_roster.PIN_FORM_KEYS, make_bank_view.plan_pin_entry): the view entry is a symlink to the OWNING lane's staged copy\n"
              f"#     (exp/objects/arditi_allpos/{key}/ladder/cells/ar_c0p3__arditi_allpos, exp/objects/paired_write/{key}/cells/\n"
              f"#     paired_s1__{OBJ_PAIRED_CELL}) whose bytes are re-hashed to the pin; no bytes are copied into the wave bank,\n"
              f"#     and the form has no staged_id / rung field, so no roster edit can substitute a rung. The Arditi RUNG (c = 0.3)\n"
              f"#     is the pin's, never the cell id's: the harness id is the bare Class B `{OBJ_ARDITI_CELL}`. The paired write is the\n"
              f"#     FULL-dose seed-1 fitted object (object_sha256 f98f0349…, the same bytes the production `s1p` view stages as\n"
              f"#     {OBJ_PAIRED_WAVE_STAGED_ID} — held equal by the generator); the half-dose view and every *_distilled\n"
              f"#     object are OUT; served_layer 0 is the pin's own (the first of its 32 served blocks).\n"
              f"#   * the two prompted baselines map to null (the SI text as the system prompt; no object, no view entry).\n"
              f"# Wave-wide keys and the target's model pins are the production roster's AS DATA; `lane_flags` is empty because no lane\n"
              f"# rides this view.\n")
    out = header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=120)
    back = yaml.safe_load(out)
    if back != doc:
        raise SystemExit(f"{padded_roster(key)}: the rendered roster does not load back as its data")
    served, staged = served_cells(back, key, OBJ_VIEW), staged_cells(back, key, OBJ_VIEW)
    if list(served) != [c for c in OBJ_CELLS if c != "none"] or len(staged) != len(OBJ_CELLS) - 3:
        raise SystemExit(f"{padded_roster(key)}: the roster's served cells {list(served)} are not the family's {list(OBJ_CELLS)[1:]}")
    return out


def outputs():
    for key in output_keys():
        if key in HAND_LISTED_KEYS or key in PROMPTED_ONLY_KEYS:
            for pad in PADS:
                yield padded_config(key, pad), render_config(key, pad)
            yield padded_manifest(key), render_manifest(key)
        if key not in HAND_LISTED_KEYS:
            # a front-door key: its PER-VIEW steered families the production roster names (module docstring) — five configs + one
            # manifest each, no lane roster (the production roster names the view); a single-phase key renders ONLY these
            for fam in families_of(key):
                if fam is None:
                    continue
                for pad in pads_of(fam):
                    yield padded_config(key, pad, fam), render_config(key, pad, fam)
                yield padded_manifest(key, fam), render_manifest(key, fam)
            continue
        # the STEERED family (module docstring): its five configs (PADS + the added rungs), its manifest, its lane roster
        for pad in OBJ_PADS:
            yield padded_config(key, pad, OBJ_FAMILY), render_config(key, pad, OBJ_FAMILY)
        yield padded_manifest(key, OBJ_FAMILY), render_manifest(key, OBJ_FAMILY)
        yield padded_roster(key), render_roster(key)
        # the SEED REPLICATES (module docstring): per seed, its five configs, its manifest, its lane roster
        for fam in OBJ_SEED_FAMILIES:
            for pad in OBJ_PADS:
                yield padded_config(key, pad, fam), render_config(key, pad, fam)
            yield padded_manifest(key, fam), render_manifest(key, fam)
            yield padded_roster(key, fam), render_roster(key, fam)


def _rel(path: Path) -> str:
    """The path relative to the repo when it is under it (the committed outputs), else as given (a test's tmp dir)."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if any committed output drifts from the generator")
    ap.add_argument("--list", action="store_true", help="print the output paths (relative to the repo), one per line")
    ap.add_argument("--resolve-family", nargs=2, metavar=("KEY", "FAMILY"), default=None,
                    help="the sbatch's question for its config-family positional: prints `<bank view> <roster> <registry view>` for a "
                         "steered family of KEY (obj | obj_s<N> for a hand-listed key; a phase view for a front-door key), exit 2 by name otherwise")
    args = ap.parse_args(argv)
    if args.resolve_family is not None:
        key, fam = args.resolve_family
        try:
            if fam is None or fam == "":
                raise ValueError("the prompted-only family has no config-family positional (target_static_padded.sbatch serves it)")
            _fam(fam)
            if is_front_door_family(fam):
                _require_front_door_family(key, fam)
                view, roster, reg_view = fam, PRODUCTION_ROSTER, fam
            else:
                if key not in HAND_LISTED_KEYS:
                    raise ValueError(f"{fam!r} is a hand-listed key's steered family ({list(HAND_LISTED_KEYS)}), not {key}'s")
                view, roster = view_of(fam), padded_roster(key, fam)
                reg_view = f"s{seed_of(fam) or 1}"      # the phase view the family's objects belong to (obj -> s1, obj_s<N> -> s<N>)
        except (ValueError, SystemExit) as exc:
            print(f"REFUSED: family {fam!r} for {key}: {exc}", file=sys.stderr)
            return 2
        print(f"{view} {_rel(roster)} {reg_view}")
        return 0
    if args.list:
        for key in output_keys():
            if key in HAND_LISTED_KEYS or key in PROMPTED_ONLY_KEYS:
                for pad in PADS:
                    print(_rel(padded_config(key, pad)))
                print(_rel(padded_manifest(key)))
            if key not in HAND_LISTED_KEYS:
                for fam in families_of(key):
                    if fam is None:
                        continue
                    for pad in pads_of(fam):
                        print(_rel(padded_config(key, pad, fam)))
                    print(_rel(padded_manifest(key, fam)))
                continue
            for pad in OBJ_PADS:
                print(_rel(padded_config(key, pad, OBJ_FAMILY)))
            print(_rel(padded_manifest(key, OBJ_FAMILY)))
            print(_rel(padded_roster(key)))
            for fam in OBJ_SEED_FAMILIES:
                for pad in OBJ_PADS:
                    print(_rel(padded_config(key, pad, fam)))
                print(_rel(padded_manifest(key, fam)))
                print(_rel(padded_roster(key, fam)))
        return 0
    for key in front_door_pending():
        print(f"skip {key}: the roster does not carry {key}/{FRONT_DOOR_SOURCE_VIEW} yet (05_stage_roster.sh --step roster) — its prompted-only "
              "padded configs render once it does")
    for key, view in front_door_view_pending():
        print(f"skip {key}/{view}: the roster does not carry this phase view yet (05_stage_roster.sh --step roster) — its per-view steered "
              "padded configs render once it does (D51)")
    drift = []
    n_outputs = 0
    for path, text in outputs():
        n_outputs += 1
        if args.check:
            if not path.is_file():
                drift.append(f"{_rel(path)}: missing")
            elif path.read_text(encoding="utf-8") != text:
                drift.append(f"{_rel(path)}: differs from the generator")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {_rel(path)}")
    if drift:
        print("\n".join(drift))
        print("regenerate: python scripts/sdfbase/full22/static_padding/make_padding_configs.py")
        return 1
    if args.check:
        print(f"ok: {n_outputs} outputs byte-identical to the generator")
    return 0


if __name__ == "__main__":
    sys.exit(main())
