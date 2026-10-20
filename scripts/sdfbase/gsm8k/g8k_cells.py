#!/usr/bin/env python3
"""The GSM8K paired-200 paper-objects campaign — THE FROZEN MATRIX'S READER and the launcher's vocabulary.

The 42-row matrix is GENERATED — ``make_gsm8k_matrix.py`` -> ``configs/sdfbase/gsm8k/matrix.json`` — through the ONE
view-selection interface (``g8k_views.select_rows``) from ``configs/sdfbase/gsm8k/views.yaml`` + the full22 roster + the pins,
and FROZEN (``--check`` pins byte-identity). This file READS it and carries the launcher's literals: the driver flags
mirrored from ``contracts`` (pinned equal by tests/test_sdfbase_gsm8k_sbatch.py — the mirror is checked, never trusted),
the caps and the rung, the geometry entries, the thinking-card sampling rows with provenance, the time classes, the
clusters and the model-based placement rule. Every count the tests, the dry run and the analysis read is DERIVED from
the matrix (``counts()``) — no count is typed twice.

IMPORT-LIGHT ON PURPOSE (stdlib + json only): the sbatch reads it with whatever python is on the node before the venvs
exist, and the dry run on the grading host reads it without torch. The row grammar has an explicit VIEW: a row is
(model key, view, cell) and TAG is ``f22_resolve_config``'s spelling — the bare key for ``main``, ``<key>_<view>``
otherwise (the spelling every config, manifest, bank view and retired store already uses).

HELD rows: a row whose ``status`` is ``held`` is IN the matrix (its index, TAG and overlay are
stable) and EXCLUDED by every production consumer here — ``plan`` never emits its index, ``--rows`` / ``--cluster-share``
naming it refuses by name, ``row`` prints ``STATUS=held`` (the sbatch's ``g8k_row`` then refuses with ``HELD`` and runs
nothing) — and never counted as refused. Un-holding is one ``views.yaml`` edit + regenerate; no code change.

ADMITTED-AND-HELD prompted cells: a ``prompted__<si>`` cell the roster declares on an expressed view that this matrix
does NOT schedule (``SCHEDULED_PROMPTED_CELLS``) is still a row — TRAILING every scheduled row (the scheduled rows'
indices never move), ``status: held`` with ``unscheduled_held_reason`` — so the matrix agrees with the roster byte for
byte while the launched set is exactly the scheduled one; every consumer above treats it like any held row. Scheduling
the cell (naming it in ``SCHEDULED_PROMPTED_CELLS``, or a later extension mechanism) releases the hold.

WAVES: every row belongs to a WAVE — ``published`` (the frozen 42 rows of this matrix, or the Llama matrix's 12:
COMPLETE at their pins, reused byte-for-byte, NEVER resubmitted), ``ext`` (the eight-arm panel's missing prompted
baselines on the main view, appended as ``extensions:`` rows under ``controls_ext/``) and ``steer`` (the AlphaSteer /
Arditi rung, appended as ``steering:`` rows under ``controls_steer/``, their objects ANOTHER LANE's —
``object_source: external`` — pinned by the ADMISSION RECORD ``configs/sdfbase/steer_admission.json``, never the full22
pin file). A row without a ``wave`` key IS a published row (``row_wave``), so the frozen rows' bytes never move.
``plan --wave <name>`` selects that wave's ``launch`` rows and asserts every printed index eligible
(``assert_eligible``); the sbatch re-asserts the ONE row it resolved at the batch boundary (``row --assert-launchable``
-> g8k_common.sh ``g8k_assert_launchable``): a row of a COMPLETED wave, a held or a deferred row refuses by name before
any venv, bank or standup — the submission is not trusted.

CLI (the sbatch, the dry run, the submitter):
  g8k_cells.py list [--json]                                    every row, status + wave included
  g8k_cells.py row --index N [--rung 8192|16384] --sh           one row as shell assignments (the ARRAY task's lookup)
  g8k_cells.py row --key K --view V --cell C [--rung R] [--sh]  one row (JSON, or --sh); a (key, view, cell) absent from the
                                                            frozen matrix refuses by name
  g8k_cells.py row (--index N | --key K --view V --cell C) --assert-launchable
                                                            the BATCH BOUNDARY's predicate: exit 0 + the record iff the row is
                                                            in ITS wave's eligible set; a published (complete), held or
                                                            deferred row refuses by name on stderr, exit 1
  g8k_cells.py row (--index N | --key K --view V --cell C) --serving-path --wave-root ROOT
                                                            the serving path the row PINS and its reason, one per line: the pin
                                                            FOLLOWS THE STORE at ROOT/<out root> (+ the rung sibling root) —
                                                            kind:alphasteer | store:lens | unpinned:no-store | store:native; a
                                                            mixed-engine store refuses by name; --wave-root is required on an
                                                            external row — what g8k_select_serving_path declares before f22_init
  g8k_cells.py counts                                             the acceptance counts, derived from the matrix
  g8k_cells.py literals --sh                                      the run-level literals (RUN_LITERALS_SH) as shell assignments
                                                            — what g8k_init evals before any row is known (the datasets guard)
  g8k_cells.py plan --sha SHA --cluster cluster_a|cluster_b --wave NAME [--assert-eligible] [--max-concurrent W]
                    [--rows SPEC | --cluster-share SPEC]
                                                            the wave's 8,192 rows' sbatch --array lines (one per --time
                                                            class), PRINTED never run; a matrix-wide plan (no --wave) would
                                                            name COMPLETE rows and is refused; a held index refuses; a
                                                            `primary` row on cluster_b refuses (model-based placement)
  g8k_cells.py plan-overlap --sha SHA --cluster ... [--keys K,K]  the thinking-OFF overlap legs (overlap_off.sbatch), same rules
"""
from __future__ import annotations

import argparse
import itertools
import importlib.util
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_FULL22 = str(Path(__file__).resolve().parent.parent / "full22")
if _FULL22 not in sys.path:
    sys.path.insert(0, _FULL22)
import itdoc_bind as B   # noqa: E402 — stdlib-only, like this file: the it-doc admission record + the bank binding (E3)
import steer_store_pin as SP   # noqa: E402 — stdlib-only: an external row's serving-path pin FOLLOWS ITS STORE

MATRIX_PATH = REPO / "configs" / "sdfbase" / "gsm8k" / "matrix.json"
VIEWS_PATH = REPO / "configs" / "sdfbase" / "gsm8k" / "views.yaml"
MATRIX_SCHEMA = "g8k_matrix/1"
#: llama-paper-rows (P11 / P16): the SECOND matrix — the Llama rows under their OWN wave root, schema 2 (its rows carry
#: `thinking_mode`; its document `wave_root_rel`); the published 42-row matrix stays schema 1, byte for byte. Every reader
#: takes the matrix as an explicit argument (`--matrix <name|path>`; the sbatch's trailing positional), the frozen one the
#: default; a NAME resolves through MATRICES, anything else is a path.
MATRIX_SCHEMA_2 = "g8k_matrix/2"
MATRIX_SCHEMAS = (MATRIX_SCHEMA, MATRIX_SCHEMA_2)


def doc_is_schema2(doc: dict) -> bool:
    """Schema-2 semantics: the g8k_matrix/2 matrices AND the qwen_seedrep wave's Arditi record (g8k_seedrep_arditi_rows/1, registered as
    the `arditi_seedrep` document, 2026-09-21), whose rows carry every schema-2 field."""
    return doc.get("schema") == MATRIX_SCHEMA_2 or doc.get("schema") in RECORD_SCHEMAS
MATRIX_DEFAULT_NAME = "default"
MATRIX_LLAMA_NAME = "llama"
#: the THIRD matrix — the instruct-direct (`it`-view) LoRA rows of the two chat-checkpoint
#: models under their OWN wave root, schema 2, regenerated from the roster + pins FROZEN at the commit views_itdoc.yaml's
#: `frozen_inputs:` names (4e200b68, the merge that served the batch-64 control + difference objects on qwen35_4b/it; ece9e8ce,
#: the gemma roster, since the gemma hold lifted 2026-09-14): per model the constitution cell on the BATCH-8 object as served
#: today, the batch-64 control LoRA and the batch-64 LoRA difference (+ the two prompted-constitution rows) — every row through
#: its own overlay pair, each object row naming its generation. The published matrix (schema 1, its roster pinned at 8f5f1495) never moves.
MATRIX_ITDOC_NAME = "itdoc"
#: qwen-base-seed-replicates (the `llama_psr` matrix is the precedent): the seed-replicate matrix — per view
#: `s1` / `s2` / `s3` of `qwen35_4b` ONE natively served `none` anchor row + the view's FIVE admitted staged objects (18 rows at staging;
#: the sixth declared object, the distilled PSR difference, failed admission for every seed and has no row), schema 2
#: under its OWN wave root (exp/gsm8k_paired200_paper_objects_qwen_seedrep) and its OWN launchable wave (WAVE_QWEN_SEEDREP below),
#: thinking ON, the Qwen thinking-card row, EVERY row under the OPT-IN Inspect replicate seed rule at its view's base
#: (configs/sdfbase/gsm8k/views_qwen_seedrep.yaml `inspect_seed_policy` / `inspect_seed_base_by_view`). THE ANCHOR:
#: a natively served cell may not be contrasted against the LENS-era published `none`
#: card — `tinycap_gsm8k_paired200._comparable` -> `tinycap_math500_thinking.server_settings_difference` refuses the engine pair by
#: name — and `design_contrasts` keys every cell's anchor as `<model_key>/<view>/none`, so the matrix carries a `none` row PER VIEW
#: (the relu_psr precedent's row shape: unfitted, no pin, route_kind base, its own overlay pair). THE STANDUP COMPANION
#: (`standup_companions:` in the views file, STANDUP_COMPANION_FIELDS below): the reader compares every echoed launch flag of the
#: two legs' standups except the cell-set plumbing (lora_modules / max_loras / max_cpu_loras), and the native composer FITS
#: `--max-lora-rank` to the registered adapters' largest r (zero adapter and PSR carrier r=1 -> cap 1; a LoRA r=4 / the rank-8
#: subtraction -> cap 8), so a `none`-only standup (cap 1) pairs with the PSR rows but is REFUSED against the three plain-LoRA rows
#: on `launch.max_lora_rank`; every standup of the wave therefore ALSO registers the view's `constv2-mix-full__lora_r4_ce`
#: (unmeasured; the driver's --cell is the row's own) so all six standups of a view compose cap 8 and `lora_target_modules
#: [o_proj]`. REGISTERED on a PROVISIONAL freeze (frozen/f7999f34: the roster before staging, every seed view's `cells` empty) —
#: the committed matrix is the three `none` rows HELD (their companion is not staged); the staging PR re-freezes the roster + pins
#: (`frozen_inputs:`), regenerates the 18-row matrix + overlays and lifts the holds by construction.
MATRIX_QWEN_SEEDREP_NAME = "qwen_seedrep"
#: arditi-allpos-cap (2026-09-17): the FOURTH matrix — ONE row, the all-positions Arditi refusal direction
#: (`arditi_allpos`) on qwen35_4b's INSTRUCT-DIRECT view, under its OWN wave root. It exists as a matrix of its own
#: because the row's view must be `it` and every steering row of the published matrix is `main` by construction: a
#: fourth document is the only shape that gives this arm the view its lane serves, leaves the published 53-row matrix
#: and its counts untouched (nothing existing is re-exported), and makes "a never-run cell has no store" STRUCTURAL —
#: an isolated wave root has no store for anything, so the serving-path pin reads `unpinned:no-store` from the world
#: rather than from a claim. NOT the `arditi` cell: a different intervention at a different dose unit (see
#: CELL_ARDITI_ALLPOS), never pooled into or relabelled as that slot.
MATRIX_ALLPOS_NAME = "arditi_allpos"
#: b64-capability-agentic-queue: the Arditi refusal steer on GSM8K runs exactly as the others — vendor thinking-card
#: decode, thinking ON, the 8,192 -> 16,384 ladder, seed bases 1..3, cluster A. The FIFTH launchable document is the qwen_seedrep
#: wave's Arditi RECORD (g8k_seedrep_arditi.py, schema g8k_seedrep_arditi_rows/1) — the `arditi_allpos` matrix's one row re-keyed
#: onto the seed views s1/s2/s3 at seed bases 1..3, under the seedrep wave root, NO companion (a single-object engine). Until now
#: only the lane's Modal launcher read it; registering it here lets gsm8k_thinking.sbatch / ladder_plan / the dry run take it BY
#: NAME like a matrix (load_matrix admits the record's schema with schema-2 semantics; external_row_problem admits its seed views).
MATRIX_ARDITI_SEEDREP_NAME = "arditi_seedrep"
RECORD_SCHEMA_ARDITI_SEEDREP = "g8k_seedrep_arditi_rows/1"       # == g8k_seedrep_arditi.SCHEMA (pinned equal by test; stdlib-only here)
ROLE_ARDITI_UNPAIRED = "arditi_unpaired_value"                   # == g8k_seedrep_arditi.ROLE — the record rows' `role`
#: the paper's all-model appendix: the paired-write difference ReLU (ReLU's relu_paired objects, staged read-only
#: into exp/objects/sdfbase/relu_paired/<model>) on gemma4e4b `it`, llama8b `it`, nemotron3nano4b `main` at the appendix column's
#: instrument (the default Inspect seed rule, one trial per model), plus nemotron × constitution_relu (the roster's PSR object, whose
#: lens-era row never completed) — a RECORD document of its own (g8k_paired_allmodels.py), its own wave root, read here BY NAME.
MATRIX_PAIRED_ALLMODELS_NAME = "paired_allmodels"
RECORD_SCHEMA_PAIRED_ALLMODELS = "g8k_paired_allmodels_rows/1"   # == g8k_paired_allmodels.SCHEMA (pinned equal by test)
ROLE_PAIRED_ALLMODELS = "paired_allmodels_value"                 # == g8k_paired_allmodels.ROLE
CELL_PAIRED_DIFFERENCE = "constv2-mix-full__psr_ce_all_minus_ctl"   # the composed psr_paired_write arm (difference_relu)
#: record schema -> the role every row of that record carries (load_matrix reads a record with schema-2 semantics)
RECORD_SCHEMAS = {RECORD_SCHEMA_ARDITI_SEEDREP: ROLE_ARDITI_UNPAIRED, RECORD_SCHEMA_PAIRED_ALLMODELS: ROLE_PAIRED_ALLMODELS}
MATRICES = {MATRIX_DEFAULT_NAME: MATRIX_PATH, MATRIX_LLAMA_NAME: REPO / "configs" / "sdfbase" / "gsm8k" / "matrix_llama.json",
            MATRIX_ITDOC_NAME: REPO / "configs" / "sdfbase" / "gsm8k" / "matrix_itdoc.json",
            MATRIX_QWEN_SEEDREP_NAME: REPO / "configs" / "sdfbase" / "gsm8k" / "matrix_qwen_seedrep.json",
            MATRIX_ALLPOS_NAME: REPO / "configs" / "sdfbase" / "gsm8k" / "matrix_arditi_allpos.json",
            MATRIX_ARDITI_SEEDREP_NAME: REPO / "configs" / "sdfbase" / "gsm8k" / "arditi_rows_qwen_seedrep.json",
            MATRIX_PAIRED_ALLMODELS_NAME: REPO / "configs" / "sdfbase" / "gsm8k" / "paired_allmodels_rows.json"}
VIEWS_BY_MATRIX = {MATRIX_DEFAULT_NAME: VIEWS_PATH, MATRIX_LLAMA_NAME: REPO / "configs" / "sdfbase" / "gsm8k" / "views_llama.yaml",
                   MATRIX_ITDOC_NAME: REPO / "configs" / "sdfbase" / "gsm8k" / "views_itdoc.yaml",
                   MATRIX_QWEN_SEEDREP_NAME: REPO / "configs" / "sdfbase" / "gsm8k" / "views_qwen_seedrep.yaml",
                   MATRIX_ALLPOS_NAME: REPO / "configs" / "sdfbase" / "gsm8k" / "views_arditi_allpos.yaml",
                   # the record is DERIVED from the seedrep views document (its seed policy, bases and geometry class)
                   MATRIX_ARDITI_SEEDREP_NAME: REPO / "configs" / "sdfbase" / "gsm8k" / "views_qwen_seedrep.yaml",
                   # the appendix record is DERIVED from the published (default) views document's frozen inputs
                   MATRIX_PAIRED_ALLMODELS_NAME: VIEWS_PATH}
#: the object GENERATION a staged id names (a wave mixing the served batch-8 constitution with the batch-64 control + difference
#: records the mix on every row): the batch-8 instruct-direct ids and their dose bakes vs the batch-64 ones (`64` in the id)
OBJECT_GENERATION_BATCH8, OBJECT_GENERATION_BATCH64 = "batch8", "batch64"
OBJECT_GENERATIONS = (OBJECT_GENERATION_BATCH8, OBJECT_GENERATION_BATCH64)


def object_generation_of(staged_id: "str | None") -> "str | None":
    """`batch64` for `itdoc64__` / `derived_itdoc64__` / `d<dose>it64__`, `batch8` for `itdoc__` / `derived_itdoc__` / `d<dose>it__`,
    None for any other lane (the base lane's `base__` / `derived__` objects are not instruct-direct fits)."""
    sid = str(staged_id or "")
    if not sid:
        return None
    if sid.startswith(("itdoc64__", "derived_itdoc64__")) or "it64__" in sid:
        return OBJECT_GENERATION_BATCH64
    if sid.startswith(("itdoc__", "derived_itdoc__")) or "it__" in sid:
        return OBJECT_GENERATION_BATCH8
    return None

# --------------------------------------------------------------------------- the vocabulary
MODEL_KEYS = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash", "llama8b")
VIEW_MAIN = "main"
VIEW_IT = "it"
CELL_NONE = "none"
CELL_PROMPTED = "prompted__safety_refusal"
#: the roster's prompted-baseline grammar (`prompted__<si>`, mapped to null: the SI text as the system prompt, no object)
PROMPTED_PREFIX = "prompted__"
#: the SIX-ARM design (PSR out by default): a SECOND prompted baseline, the short
#: constitution SI as the system prompt (TrainTest si_constitution -> harness prompted__constitution; ~5.9k Llama tokens, so
#: its rows carry their OWN prompt class, geometry entries and budget class — never the bare cells' window blindly)
CELL_PROMPTED_CONST = "prompted__si_constitution"
PROMPTED_CELLS = (CELL_PROMPTED, CELL_PROMPTED_CONST)
#: the PROMPT CLASS of a cell (the analogue of contracts.MATH500_PROMPT_CLASS_OF_SI for this instrument): `short` = no system
#: turn (none, every object cell) or the ~88-token safety SI; `long_si` = the constitution SI (measured 2026-09-12 on the Llama
#: tokenizer: bare max 2,135, safety-prompted max 2,223, constitution-prompted max 8,007 over the 200 renders, cap-independent)
PROMPT_CLASS_SHORT = "short"
PROMPT_CLASS_LONG_SI = "long_si"
LONG_SI_PROMPTED_CELLS = (CELL_PROMPTED_CONST,)
#: the rung's two EXTERNAL steering cells, in the eight-arm order — bare Class B
#: ids the SERVING grammar admits (`alphasteer` in defense_cells.CLASS_B_IDS, `arditi` in its SERVED_ONLY_CLASS_B_IDS): one
#: object per model, SI-agnostic, served from ANOTHER LANE's bank view and pinned by the ADMISSION RECORD's object sha256
#: (STEER_ADMISSION_REL), never by the full22 pin file (which stages neither). On this instrument a steering row sits on the
#: PANEL view `main` (the eight-arm slots are the main view's); the object's own lane view rides the row as `bank_view`.
CELL_ALPHASTEER = "alphasteer"
CELL_ARDITI = "arditi"
#: arditi-allpos-cap (2026-09-17): the ALL-POSITIONS Arditi refusal direction, APPENDED to the rung's cells so the
#: eight-arm panel order of the two above never moves. It is NOT the `arditi` cell above and is never pooled with it
#: or filed under its id: `arditi` is a `static_r` object steered at PREFILL positions only, dosed as a fraction of
#: the residual norm (`alpha`, views `ar_r…`); this is a `vector_all` object steered at EVERY position in both phases
#: (prefill and every decode step), dosed as a RAW COEFFICIENT baked into the bytes (`c`, views `ar_c…`). Two doses in
#: two units on two interventions: a shared axis or a shared slot would be a different measurement under one name.
CELL_ARDITI_ALLPOS = "arditi_allpos"
#: the refusal CODE a steering entry may DECLARE its row's CONTRAST to refuse with (`expected_contrast_refusal:` on the entry;
#: stamped on the generated row, appended last) — the reader's O106 cross-engine ruling (`tinycap_math500_thinking.
#: CROSS_ENGINE_RULING`; the two spellings are pinned equal by test). The all-positions arm is served NATIVELY on the `it` view
#: and its contrast is against the `it` view's `none` anchor, a LENS store; the reader's compatibility rule refuses that pairing
#: by name (the native-vs-lens numerics difference was never measured). The ruling in force: NO native
#: `none` anchor is run for GSM8K or MATH-500 — the arm's numbers are reported AS the arm's numbers and the pairing's absence
#: accepted. So the design declares the refusal and the reader's production-result rule files the contrast as refused BY DESIGN
#: iff its refusal derives EXACTLY this code; another refusal, or a contrast that computes, still blocks. Distinct from the
#: LEG-level `expected_refusal` (a standup refusing a cell by name — A0's refused_expected)
EXPECTED_CONTRAST_REFUSAL_CROSS_ENGINE = "O106"
EXPECTED_CONTRAST_REFUSAL_CODES: tuple[str, ...] = (EXPECTED_CONTRAST_REFUSAL_CROSS_ENGINE,)
STEER_CELLS = (CELL_ALPHASTEER, CELL_ARDITI, CELL_ARDITI_ALLPOS)
#: THE VIEW each steer arm's row sits on, DECLARED per arm (never "main unless someone says otherwise"): the rung's
#: two arms are the PANEL view's (the eight-arm slots are `main`'s; the object's own lane view rides `bank_view`), and
#: the all-positions arm's row is the INSTRUCT-DIRECT view's — the view its lane serves and the one its analysis reads.
#: `external_row_problem` refuses a row of either arm on the other's view BY NAME, so the two can never be mixed into
#: one column by a views-file edit.
EXTERNAL_ROW_VIEW_BY_CELL = {CELL_ALPHASTEER: VIEW_MAIN, CELL_ARDITI: VIEW_MAIN, CELL_ARDITI_ALLPOS: VIEW_IT}


def external_row_view_for(cell: str) -> str:
    """The view a steer arm's row sits on (:data:`EXTERNAL_ROW_VIEW_BY_CELL`); an unknown arm refuses by name."""
    try:
        return EXTERNAL_ROW_VIEW_BY_CELL[cell]
    except KeyError:
        raise SystemExit(f"external_row_view_for: {cell!r} is not one of the rung's cells {STEER_CELLS}") from None
#: the row field that says the pin is the ADMISSION RECORD's, not the full22 file's (the analysis' validators read the
#: same spelling: tinycap_gsm8k_validators.OBJECT_SOURCE_EXTERNAL)
OBJECT_SOURCE_EXTERNAL = "external"
#: THE SERVING PATH an external-object row PINS — "THE PIN FOLLOWS THE STORE" (the MATH-500 launcher's
#: `serving_path_pin`, mirrored; the rule itself lives in full22/steer_store_pin.py, ONE stdlib module both launchers read). native-serving-
#: default made F22_SERVING_PATH default to `lora_native`, and a native standup SKIPS BY NAME the object kinds it cannot serve, so a
#: single-cell job whose one object cell is skipped refuses there (the MATH-500 array 506737 died that way, 2026-09-14). AlphaSteer
#: (`alphasteer`) has no native form: its row pins the lens BY KIND (`kind:alphasteer`). Arditi (`static_r`) HAS one since native-psr-serving
#: (the residual slot), so its row pins the lens only where its steer store ON RECORD is a lens store (`store:lens` — every store on
#: record today; a finished store is never re-run), is UNPINNED where no store exists (`unpinned:no-store`: the kind-aware default then
#: selects native at f22_init) or where the store is native-stamped (`store:native`), and a store carrying BOTH engines' stamps refuses by
#: name. "On record" is TWO sources read together: the DISK — the row's OWN out root for the cap AND its rung
#: sibling (`rung_out_root`) under the wave root of the cluster the job runs on — and the PINS — this matrix's store-pins record's engine
#: for the row's admitted legs (STORE_PINS_PATHS, travelling with the launch tree, so a store on the OTHER cluster still counts); the reason
#: names the source (`store:lens(disk+pins)`). Read by the sbatch BEFORE f22_init (g8k_select_serving_path, `row --serving-path
#: --wave-root`); the grading host's `plan` reads kinds and pins, never a cluster disk.
SERVING_PATH_LENS = SP.SERVING_PATH_LENS
#: the PINS source of "on record": the repo's admitted store-pins records (the analysis' --store-pins admission; one per matrix,
#: read TOGETHER — leg keys never collide across them), whose `_meta.serving_engine_by_leg` block carries each admitted steer leg's
#: engine, derived from the pulled store by scripts/sdfbase/full22/stamp_store_pins_engines.py. They travel with the launch tree, so a
#: row whose store lives on the OTHER cluster still follows it here.
#: ONE record per matrix, named after it (`store_pins_<matrix name>.json`; stamp_store_pins_engines.py derives the path from matrix_name_of); the
#: all-positions matrix's record was added 2026-09-17 when its two legs launched (arditi-allpos-cap; appended last so the index reads hold)
STORE_PINS_PATHS: tuple[Path, ...] = tuple(REPO / "configs" / "sdfbase" / "gsm8k" / f"store_pins_{name}.json"
                                           for name in (MATRIX_DEFAULT_NAME, MATRIX_LLAMA_NAME, MATRIX_ITDOC_NAME, MATRIX_ALLPOS_NAME))


def leg_keys(row: dict) -> list[str]:
    """The row's admitted-leg keys in the store-pins records, one per cap (`<model_key>/<view>/<cell>@<cap>` — the analysis' leg key,
    tinycap_gsm8k_validators.leg_key, mirrored; pinned equal by test): the pins are consulted at ANY cap, a rung leg being the same row."""
    return [f"{row['model_key']}/{row['view']}/{row['cell']}@{int(cap)}" for cap in RUNGS]


def store_roots_rel(row: dict) -> list[str]:
    """The out roots (relative to the wave root) that are THIS ROW's store on record: the cap leg's and, on a row with a rung
    sibling, the rung's — a rung leg runs on the base leg's engine, so the pin reads both."""
    return [str(row["out_root"])] + ([str(row["rung_out_root"])] if row.get("rung_out_root") else [])


def serving_path_pin(row: dict, *, wave_root: "str | Path | None" = None,
                     store_pins: "Sequence[str | Path] | None" = None) -> tuple[str | None, str]:
    """The pin of a matrix row and WHY: ``(path, reason)`` — ``lens`` / ``kind:alphasteer`` on an AlphaSteer row; on an Arditi row
    ``lens`` / ``store:lens(<sources>)``, None / ``store:native(<sources>)`` or None / ``unpinned:no-store`` from TWO sources read
    together — the DISK store at ``wave_root/<out roots>`` and the PINS records' engine for the row's admitted legs (``store_pins``:
    the repo's by default) — a disagreement refusing by name; None / ``unpinned:not-external`` on every other row (the kind-aware
    default stands). With no ``wave_root`` (the grading host plan) the disk is unread: an Arditi row the pins do not know reads None /
    ``unpinned:store-unread`` — never "no store"."""
    if row.get("object_source") != OBJECT_SOURCE_EXTERNAL:
        return None, SP.REASON_UNPINNED_NOT_EXTERNAL
    roots = None if wave_root is None else [Path(wave_root) / rel for rel in store_roots_rel(row)]
    pins = SP.read_pins_engines(STORE_PINS_PATHS if store_pins is None else [Path(p) for p in store_pins], leg_keys(row))
    if row.get("role") == ROLE_PAIRED_ALLMODELS:
        return paired_write_serving_pin(roots, pins)
    kind = SP.steer_cell_kind(str(row["cell"]))
    path, reason, _stores = SP.pin_for(kind, roots, pins)
    return path, reason


def paired_write_serving_pin(roots: "list[Path] | None", pins) -> tuple[str | None, str]:
    """The store-following pin of the all-model appendix record's EXTERNAL rows (the composed paired-write difference ReLU, kind
    psr_paired_write): the SAME two sources steer_store_pin.pin_for reads for a natively servable steer kind — the disk store(s) at the
    row's out roots and the pins records' engine for its legs — under the one rule this arm admits: the paired write has NO lens form
    (the route relu_gated_vector_pair exists on the native residual slot only; the Qwen composed rows' finished stores are native), so a
    store on record is either native (unpinned: the kind-aware default re-selects lora_native) or absent (unpinned), and a store or pins
    entry carrying the LENS engine's stamp is a foreign store under this row's root — refused by name, never followed. The steer cells'
    kind table is not widened for it: that table mirrors the steer wave's cells (registry_adapter.ARM_OBJECT_KINDS) and is pinned so."""
    stores = () if roots is None else tuple(s for s in (SP.read_store_engines(r) for r in roots) if s is not None)
    disk_engines = sorted({e for s in stores for e in s.engines})
    pins_engines = sorted({p.engine for p in pins})
    if stores and not disk_engines:
        raise SystemExit("REFUSED (paired-write store pin): the row's store is on record but names NO engine — the pin follows the store and "
                         "cannot: " + " | ".join(s.describe() for s in stores))
    if not stores and not pins:
        return None, (SP.REASON_UNPINNED_STORE_UNREAD if roots is None else SP.REASON_UNPINNED_NO_STORE)
    engines = sorted(set(disk_engines) | set(pins_engines))
    if engines != [SP.ENGINE_NATIVE]:
        where = " | ".join([s.describe() for s in stores] + [p.describe() for p in pins])
        raise SystemExit(f"REFUSED (paired-write store pin): a psr_paired_write row's store on record carries the engine stamps {engines} "
                         f"(disk: {disk_engines or 'none'}; pins: {pins_engines or 'none'}) — the paired write has no lens form, so a lens "
                         "stamp under this row's root is a foreign store, not this row's; set it aside before a leg is written here: " + where)
    sources = ([SP.SOURCE_DISK] if disk_engines else []) + ([SP.SOURCE_PINS] if pins_engines else [])
    return None, SP.store_reason(SP.ENGINE_NATIVE, sources)


def serving_path_for(row: dict, *, wave_root: "str | Path | None" = None, store_pins: "Sequence[str | Path] | None" = None) -> str | None:
    """The serving path a matrix row PINS (``serving_path_pin`` without the reason): `lens` on an AlphaSteer row, `lens` on an Arditi row
    whose store on record (disk or pins) is a lens store, None otherwise (the launcher's kind-aware default stands)."""
    return serving_path_pin(row, wave_root=wave_root, store_pins=store_pins)[0]


#: THE LAUNCH CELLS a single-cell job DECLARES to the wave shell (F22_LAUNCH_CELLS, common.sh f22_select_serving_path — the kind-aware
#: default of 2026-09-15; the MATH-500 launcher's `launch_cells_for`, mirrored): the HARNESS spelling of exactly the cells this
#: job's standup serves — `none` (implicit on every standup: mt_serving_up `--cell` serves ONLY the named TrainTest cell with `none`
#: implicit, so `cells_requested` reads [none, <cell>]) plus the row's STANDUP cell in harness spelling (`standup_harness_cell`: the
#: `@thinking` variant when the row declares it — the one spelling the standup's --cell, its digest key and the table share). A row
#: that pins the lens keeps its pin (an explicit F22_SERVING_PATH always wins, recorded EXPLICIT); every other row's path is selected
#: BY KIND over these cells (a LoRA or PSR cell -> lora_native, the PSR cell on the residual slot since native-psr-serving; none / prompted
#: -> the entry-point default) and logged `[serving-path] AUTO-selected`. Read by the sbatch BEFORE f22_init (g8k_select_serving_path),
#: printed by the dry run (LAUNCH_CELLS:).
def launch_cells_for(row: dict) -> list[str]:
    """The harness cells a matrix row's standup serves: ``none`` and, on every row but the base one, the row's ``standup_harness_cell``
    — plus, on a row of a matrix that declares ``standup_companions`` (qwen-base-seed-replicates O12a), each companion's harness cell
    ONCE (``row_companions``): the standup registers them beside the row's own so every standup of the wave composes one LoRA kernel
    configuration; the driver still measures the row's cell alone."""
    hc = row["standup_harness_cell"]
    out = [CELL_NONE] + ([] if hc == CELL_NONE else [hc])
    for _cell, harness, _sid, _pin in row_companions(row):
        if harness not in out:
            out.append(harness)
    return out


def declared_serving_path(row: dict) -> str | None:
    """The serving path the ROW'S OWN RECORD declares for every leg of its arm, or None where it declares none
    (arditi-allpos-cap, 2026-09-17): the admission entry's ``attestation_expect.serving_path``, carried onto the row.

    It exists because for this arm the path is part of the MEASUREMENT, not a performance choice: the all-positions
    direction is specified on the graph-captured native residual slot, and the lens WOULD serve a ``vector_all`` blob —
    so a leg forced onto the lens does not fail, it completes and files a different intervention's numbers under this
    arm's name. The launcher refuses that conflict at its first row-resolution step (``g8k_select_serving_path``),
    before ``f22_init`` can fix an explicit path from the environment. Every row whose record declares nothing (the
    steering rung's two arms, every campaign row) reads None here and is untouched: the kind-aware default still
    decides."""
    declared = (row.get("attestation_expect") or {}).get("serving_path")
    return str(declared) if declared else None


#: qwen-base-seed-replicates (the fitted --max-lora-rank rule of the GSM8K reader's launch comparison): the STANDUP
#: COMPANION fields a row carries when its matrix declares `standup_companions:` — the TrainTest cells the standup registers BESIDE
#: the row's own (`--cell <c> --expect-object-sha256 <c>=<pin>` on mt_serving_up's argv, g8k_common.sh g8k_standup_extra), their
#: harness ids (the overlay pair enumerates them), the staged ids and the frozen pin file's pins. Parallel lists / dicts keyed by
#: the TrainTest cell id; EMPTY on the companion's own row (it registers itself) and on a HELD row whose companion is not staged;
#: ABSENT on every row of every matrix that declares no companions (byte-identity of the other matrices).
STANDUP_COMPANION_CELLS = "standup_companion_cells"
STANDUP_COMPANION_HARNESS_CELLS = "standup_companion_harness_cells"
STANDUP_COMPANION_STAGED_IDS = "standup_companion_staged_ids"
STANDUP_COMPANION_PINS = "standup_companion_pins"
STANDUP_COMPANION_FIELDS = (STANDUP_COMPANION_CELLS, STANDUP_COMPANION_HARNESS_CELLS, STANDUP_COMPANION_STAGED_IDS, STANDUP_COMPANION_PINS)
#: the document block (`standup_companions: {cell, reason}`) and the hold a row takes while its companion is not a staged cell of its view
STANDUP_COMPANIONS_KEY = "standup_companions"
COMPANION_HELD_MARK = "standup companion not staged"


def row_companions(row: dict) -> list[tuple[str, str, str, str]]:
    """``[(cell, harness_cell, staged_id, pin), ...]`` — the row's standup companions in list order; ``[]`` on a row without the
    fields (every other matrix) or with empty ones (the companion's own row, a held row awaiting staging)."""
    cells = row.get(STANDUP_COMPANION_CELLS) or []
    harness = row.get(STANDUP_COMPANION_HARNESS_CELLS) or []
    sids = row.get(STANDUP_COMPANION_STAGED_IDS) or {}
    pins = row.get(STANDUP_COMPANION_PINS) or {}
    return [(str(c), str(harness[i]), str(sids[c]), str(pins[c])) for i, c in enumerate(cells)]


def companion_row_problem(row: dict, doc: dict) -> str | None:
    """qwen-base-seed-replicates: why a row's standup-companion fields disagree with the document — fields on a matrix that declares
    none; a declaring matrix's row missing one; lists of unequal length; a launch row whose companions are not exactly the document's
    cell (or empty ONLY on that cell's own row); a companion pin that is not a 64-hex sha; a companion naming `none` or the row's own
    cell. None when consistent (every row of a matrix without the block carries no field)."""
    decl = doc.get(STANDUP_COMPANIONS_KEY)
    present = [f for f in STANDUP_COMPANION_FIELDS if f in row]
    if decl is None:
        return f"carries {present} but the document declares no {STANDUP_COMPANIONS_KEY}" if present else None
    if not doc_is_schema2(doc):
        return f"{STANDUP_COMPANIONS_KEY} is a {MATRIX_SCHEMA_2} field"
    if not isinstance(decl, dict) or not isinstance(decl.get("cell"), str) or not decl["cell"] or not isinstance(decl.get("reason"), str) or not decl["reason"].strip():
        return f"the document's {STANDUP_COMPANIONS_KEY} is not {{cell: <TrainTest cell id>, reason: <text>}}: {decl!r}"
    if len(present) != len(STANDUP_COMPANION_FIELDS):
        return f"the document declares {STANDUP_COMPANIONS_KEY} but the row carries {present}, not all of {list(STANDUP_COMPANION_FIELDS)}"
    cells, harness, sids, pins = (row[f] for f in STANDUP_COMPANION_FIELDS)
    if not (isinstance(cells, list) and isinstance(harness, list) and isinstance(sids, dict) and isinstance(pins, dict)):
        return f"{STANDUP_COMPANION_CELLS} / {STANDUP_COMPANION_HARNESS_CELLS} are lists and {STANDUP_COMPANION_STAGED_IDS} / {STANDUP_COMPANION_PINS} dicts"
    if len(cells) != len(harness) or set(cells) != set(sids) or set(cells) != set(pins) or len(set(cells)) != len(cells):
        return f"the companion lists / dicts are not one set of cells: {cells} / {harness} / {sorted(sids)} / {sorted(pins)}"
    for c in cells:
        if c == CELL_NONE or c == row.get("cell") or is_prompted_cell(c):
            return f"companion {c!r} is not a steered cell other than the row's own ({row.get('cell')!r})"
        pin = pins[c]
        if not (isinstance(pin, str) and len(pin) == 64 and all(ch in "0123456789abcdef" for ch in pin)):
            return f"companion {c!r} pin {pin!r} is not a 64-hex sha256"
        if not (isinstance(sids[c], str) and sids[c]):
            return f"companion {c!r} has no staged id"
    want = [] if row.get("cell") == decl["cell"] else [decl["cell"]]
    if row.get("status") == STATUS_LAUNCH and cells != want:
        return f"a launch row's companions are exactly {want} (the document's), got {cells}"
    if cells and cells != want:
        return f"companions {cells} are not the document's {want}"
    return None
#: the P8 ADMISSION RECORD — ONE file for both math instruments (the MATH-500 launcher's cells.py reads the same one);
#: keyed by MODEL KEY then CELL ID, each entry naming its LANE ROOT and per-cluster bank view beside the 64-hex object sha256
STEER_ADMISSION_REL = Path("configs/sdfbase/steer_admission.json")
STEER_ADMISSION_PATH = REPO / STEER_ADMISSION_REL
#: the keys a steering row's admission entry MUST carry (a missing one refuses the whole record BY NAME at load)
STEER_ADMISSION_REQUIRED = ("object_sha256", "view", "bank_cell_dir", "bank_view_rel", "lane_root", "selection_rule",
                            "selection_label", "grid_limited", "cluster_constraint", "staged_id")
#: the AlphaSteer canonical ladder's SELECTION OUTCOME per entry: the ladder
#: record's `chosen` block either picked the operating rung under (a)+(b)+(c) (`chosen`) or, when NO live rung satisfied all
#: three, took the A4 FALLBACK — the smallest lambda satisfying (b)+(c) with the over-refusal ceiling (a) relaxed (`a4_fallback`;
#: nemotron3nano4b rung 1: paired increase upper bound 3.2 pp > the 2 pp bar). The four AlphaSteer points were therefore NOT
#: selected the same way, so the outcome rides EVERY AlphaSteer row beside the verbatim selection label (read from the ladder
#: record on disk at admission, never from a message); an Arditi entry carries `null` (the arditi battery's alpha pick has no
#: such outcome).
SELECTION_OUTCOME_CHOSEN = "chosen"
SELECTION_OUTCOME_A4_FALLBACK = "a4_fallback"
#: the operating point moved on three models on 2026-09-14 (gemma, qwen lam1, nemotron lam2; llama lam1
#: confirmed): the outcome words are the ORACLE's (scripts/alphasteer_f22/operating_points.py — `ruled_<date>`
#: where the point moved, `chosen_confirmed_<date>` where it did not); `chosen` / `a4_fallback` stay in the vocabulary for
#: the superseded points (kept under each entry's `superseded` block and on the matrices that ran them)
SELECTION_OUTCOME_RULED = "ruled_2026-09-14"
SELECTION_OUTCOME_CHOSEN_CONFIRMED = "chosen_confirmed_2026-09-14"
#: gemma was ruled TWICE that day: the first ruling named lam3, the SECOND ruling (later 2026-09-14) moved it lam3 -> lam2
#: after lam3 failed over-refusal on every reading, declined 147/200 GSM8K items and cost -8.0 pp MMLU. The oracle spells the
#: second ruling's word `_second` so a reader can tell which of the day's two rulings a pin was cut against; gemma's entry carries
#: it (the admission record re-stamped 2026-09-15)
SELECTION_OUTCOME_RULED_SECOND = "ruled_2026-09-14_second"
#: a CANDIDATE point — the INTERIM gemma lam2 of 2026-09-14 (commit 47bbca53), active while the first ruling's lam3
#: stood disqualified on the measured readings and before the second ruling landed; row 47's store was produced under it and
#: carries the word verbatim, so it stays in the vocabulary. The load-time gate admits a candidate only when its rung / lambda / sha
#: equal the ladder record's row at that rung AND that rung is not the ruled one (steer_admission_ruling.is_candidate): a
#: candidate at the ruled rung is a contradiction and refuses by name
SELECTION_OUTCOME_CANDIDATE = "candidate_2026-09-14"
SELECTION_OUTCOMES = (SELECTION_OUTCOME_CHOSEN, SELECTION_OUTCOME_A4_FALLBACK, SELECTION_OUTCOME_RULED,
                      SELECTION_OUTCOME_CHOSEN_CONFIRMED, SELECTION_OUTCOME_RULED_SECOND, SELECTION_OUTCOME_CANDIDATE)


#: the ONE non-null outcome an Arditi entry may carry: the rule of 2026-09-14 moved llama8b's Arditi operating point off the
#: battery's own pick (alpha 0.1, the inclusive-boundary pick, -> 0.05); an Arditi entry carrying it must bind its `ruling` + `superseded`
#: blocks, its view / staged id / bank paths to the ruled alpha, and (where the battery record is present) the alpha to the battery's
#: for that view (scripts/sdfbase/steer_admission_ruling.arditi_entry_problems). The ladder words (`chosen`, `a4_fallback`,
#: `chosen_confirmed_*`, `ruled_*_second`, `candidate_*`) stay the AlphaSteer ladder's and are refused on an Arditi entry.
ARDITI_SELECTION_OUTCOMES = (None, SELECTION_OUTCOME_RULED)
#: the ONE outcome an ALL-POSITIONS Arditi entry / row may carry (arditi-allpos-cap, 2026-09-17): the dose c = 0.3 is a
#: DECLARED USER OVERRIDE of the selection rule, which returns c = 0.2 at +5.04 pp over-refusal; 0.3 measures +10.26 pp
#: against a 10 pp budget, so the pinned point is OVER budget and the record says so in its own name. The word is
#: mandatory and exact: `None` is refused (a null would read as "the rule's own pick", which this is not), and every
#: AlphaSteer ladder word and the Arditi ruling word are refused (they belong to other arms' instruments, and one of
#: them appearing here would silently re-attribute the choice to a rule that did not make it).
SELECTION_OUTCOME_ALLPOS_USER_OVERRIDE = "user_override_over_budget_2026-09-16"
ARDITI_ALLPOS_SELECTION_OUTCOMES = (SELECTION_OUTCOME_ALLPOS_USER_OVERRIDE,)
#: the dose UNIT of the all-positions arm — a raw coefficient baked into the object's bytes, NEVER an alpha (a
#: residual-norm fraction is the other Arditi arm's unit on a different intervention). `alpha_equiv` may ride the
#: entry as arithmetic convenience and must say so; it is not a dose and never shares an axis with the alpha arms.
DOSE_UNIT_RAW_COEFFICIENT = "raw_coefficient"
DOSE_UNIT_RESIDUAL_NORM_FRACTION = "residual_norm_fraction"


def selection_outcome_problem(cell: str, outcome) -> str | None:
    """Why `outcome` is not a valid `selection_outcome` for `cell`: an AlphaSteer entry / row must carry one of
    :data:`SELECTION_OUTCOMES`; an ALL-POSITIONS Arditi one exactly :data:`SELECTION_OUTCOME_ALLPOS_USER_OVERRIDE` (the declared
    user override; never null, never another arm's word); an Arditi one carries ``None`` (the battery's own pick has no such
    outcome) or, when a user ruling overrode that pick, exactly :data:`SELECTION_OUTCOME_RULED`
    (:data:`ARDITI_SELECTION_OUTCOMES`) — never a ladder word."""
    if cell == CELL_ALPHASTEER:
        if outcome not in SELECTION_OUTCOMES:
            return f"selection_outcome {outcome!r} on an alphasteer entry — must be one of {SELECTION_OUTCOMES} (the canonical ladder record's `chosen` block; D90 addendum)"
    elif cell == CELL_ARDITI_ALLPOS:
        if outcome not in ARDITI_ALLPOS_SELECTION_OUTCOMES:
            return (f"selection_outcome {outcome!r} on an {cell} entry — the all-positions arm's operating point is a DECLARED USER "
                    f"OVERRIDE of its selection rule and says so in exactly one word, {SELECTION_OUTCOME_ALLPOS_USER_OVERRIDE!r}: null "
                    "would read as the rule's own pick (it is not — the rule returns c = 0.2 and the pinned dose is over budget), and "
                    "the AlphaSteer ladder's words and the Arditi ruling word belong to other arms' instruments")
    elif outcome not in ARDITI_SELECTION_OUTCOMES:
        return (f"selection_outcome {outcome!r} on a {cell} entry — an Arditi entry carries null (the battery's pick) or exactly "
                f"{SELECTION_OUTCOME_RULED!r} (a user ruling that overrode it); the ladder's words are the AlphaSteer entries'")
    return None


#: the INSTRUCT-DIRECT ("it-doc") batch-64 RE-FIT objects on the `it` view —
#: the three LoRA arms in the eight-arm order (LoRA ctl2 · LoRA mix · LoRA mix−ctl), generated from the it-doc ADMISSION RECORD
#: (configs/sdfbase/itdoc_admission.json, ONE file for both math instruments) keyed by (lane, model key, view, cell id) -> object
#: sha256. A row whose entry carries the pin + its ruled staged id + a complete bank binding launches; anything less is HELD with the
#: record's reason (never a placeholder sha). The `constv2-mix-full__lora_r4_ce` row REPLACES the held direct-fit row IN PLACE
#: (index, TAG and job id kept: the slot the campaign held for it; the superseded object recorded on the row); the other two are
#: appended. `object_source: itdoc` tells every reader the pin is the it-doc record's, not the full22 pin file's.
CELL_LORA_CTL2 = "constv2ctl2-full__lora_r4_ce"
CELL_LORA_MIX = "constv2-mix-full__lora_r4_ce"
CELL_LORA_DIFF = "constv2-mix-full__lora_r4_ce_minus_ctl"
ITDOC_CELLS = (CELL_LORA_CTL2, CELL_LORA_MIX, CELL_LORA_DIFF)
OBJECT_SOURCE_ITDOC = B.OBJECT_SOURCE_ITDOC
OBJECT_SOURCES = (OBJECT_SOURCE_EXTERNAL, OBJECT_SOURCE_ITDOC)
ITDOC_ADMISSION_REL = B.ADMISSION_REL
ITDOC_ADMISSION_PATH = REPO / ITDOC_ADMISSION_REL


def load_itdoc_admission(path: Path = ITDOC_ADMISSION_PATH) -> dict:
    """The it-doc admission record (itdoc_bind.load_admission: every entry validated by name at load)."""
    return B.load_admission(path)


def load_steer_admission(path: Path = STEER_ADMISSION_PATH) -> dict[tuple[str, str], dict]:
    """``{(model key, cell): entry}`` from the admission record — every entry validated for the fields a row reads
    (:data:`STEER_ADMISSION_REQUIRED`, a 64-hex object sha256, the cell id agreeing with its key, the AlphaSteer
    `selection_outcome` in :data:`SELECTION_OUTCOMES`). A malformed entry refuses BY NAME here, at generation, rather than at
    the batch boundary of a job that already holds a GPU. Stdlib only (the file is JSON), like everything else this module reads."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"no admission record at {path} — the steering rows are generated from it (plan P8)")
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict] = {}
    for key, by_cell in (doc.get("entries") or {}).items():
        for cell, entry in by_cell.items():
            where = f"{path.name}: entries.{key}.{cell}"
            missing = [f for f in STEER_ADMISSION_REQUIRED if entry.get(f) in (None, "")]
            if missing:
                raise SystemExit(f"{where}: the admission record lacks {missing} — a steering row is never launched on an "
                                 "incomplete record (plan P8)")
            sha = entry["object_sha256"]
            if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
                raise SystemExit(f"{where}: object_sha256 {sha!r} is not 64-hex — the pin is the OBJECT's digest, never a "
                                 "directory digest (D5)")
            if entry.get("cell") != cell or entry.get("model_key") != key:
                raise SystemExit(f"{where}: the entry names ({entry.get('model_key')!r}, {entry.get('cell')!r}) — an entry is "
                                 "keyed by model key AND cell id, and must say so (D45)")
            if cell not in STEER_CELLS:
                raise SystemExit(f"{where}: {cell!r} is not one of the rung's cells {STEER_CELLS}")
            rel = entry["bank_view_rel"]
            if not isinstance(rel, dict) or set(rel) != {"primary", "overflow"}:
                raise SystemExit(f"{where}: bank_view_rel must map the two cluster roles {{primary, overflow}} to a path relative "
                                 f"to the shared root (null where the object is not staged), got {rel!r}")
            problem = selection_outcome_problem(cell, entry.get("selection_outcome"))
            if problem:
                raise SystemExit(f"{where}: {problem}")
            out[(key, cell)] = entry
    # the RULED operating point: every AlphaSteer entry's (rung, lambda, object sha, outcome word) and
    # bank paths must bind to the oracle's record where that record is present on this machine — a stale pin refuses BY NAME
    # here, at generation, rather than hashing clean and serving a superseded dose (steer_admission_ruling; skipped, never
    # guessed, on a machine without the ladder record)
    _ruling_check(out, where=path.name)
    return out


def _ruling_check(entries: dict, *, where: str):
    spec = importlib.util.spec_from_file_location("steer_admission_ruling", REPO / "scripts" / "sdfbase" / "steer_admission_ruling.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.check_entries(entries, where=where)
#: the prompted cells a matrix SCHEDULES — emitted as launch rows inside their (key, view) block — PER
#: SCHEMA. Every OTHER `prompted__<si>` cell the roster declares on an expressed view is ADMITTED as a trailing row with
#: status `held` (`unscheduled_held_reason`), never launched, so the matrix agrees with the roster while the launched set does
#: not move; a scheduling change names the cell here and the hold releases. The PUBLISHED (schema-1) matrix schedules the
#: matched prompted safety baseline only; the Llama (schema-2) matrix schedules BOTH prompted
#: baselines — its six-arm design and the noise-floor pair are LAUNCHED rows,
#: the wave ran them. `SCHEDULED_PROMPTED_CELLS` keeps naming the published matrix's set (its consumers read it).
SCHEDULED_PROMPTED_CELLS_BY_SCHEMA = {MATRIX_SCHEMA: (CELL_PROMPTED,), MATRIX_SCHEMA_2: PROMPTED_CELLS}
SCHEDULED_PROMPTED_CELLS = SCHEDULED_PROMPTED_CELLS_BY_SCHEMA[MATRIX_SCHEMA]
#: the prefix of every reason `unscheduled_held_reason` stamps — how a consumer tells an unscheduled hold from a views.yaml hold
UNSCHEDULED_HELD_MARK = "D28 unscheduled"
#: the two scale-12 safety objects of gemma / qwen `main` are served at `response` (full22 README §2; plan F4) — HEADER-
#: LOCATED, so their thinking-ON render must be DECLARED (`<cell>@thinking` to the standup, `petri_thinking_on` on every
#: request); every other fitted object is `positions: all` (undeclared, the render certified live)
HEADER_LOCATED_CELLS = ("safety_refusal__psr_kl_all", "safety_refusal__lora_r4")
FAMILY_BY_MODEL = {"gemma4e4b": "gemma4", "qwen35_4b": "qwen3_5", "glm46vflash": "glm4v", "nemotron3nano4b": "nemotron_h",
                   "llama8b": "llama"}
#: llama-paper-rows (P7): the whole-arm literal each FAMILY's rows run under — `on` for the four thinking families,
#: `not_available` (contracts.THINKING_NOT_AVAILABLE, mirrored; pinned equal by test) for a family whose chat template
#: carries NO thinking switch. The literal rides the ROW (`thinking_mode`, `G8K_THINKING` per row); the launcher can never
#: emit `off` for a paper row.
THINKING_ON = "on"
THINKING_NOT_AVAILABLE = "not_available"
THINKING_MODE_BY_FAMILY = {"gemma4": THINKING_ON, "qwen3_5": THINKING_ON, "glm4v": THINKING_ON, "nemotron_h": THINKING_ON,
                           "llama": THINKING_NOT_AVAILABLE}
#: contracts.HEADER_INVARIANT_THINKING_FAMILIES (pinned equal by test): a family whose header does NOT move with the
#: thinking switch (gemma4: the ON assistant header EQUALS the OFF header). Since 2026-09-10
#: (`gemma-thinking-on-invariant`) that is the HEADER-INVARIANT CASE, admitted and recorded as the second header at
#: every refusal site — a header-located cell of such a family IS servable thinking-ON on the DECLARED path
#: (`<cell>@thinking`; `contracts.header_invariant_declaration_refusal` is withdrawn and always None). The tuple names
#: no refusal here any more: it is kept as the family fact the dry run prints and the test pins equal to the contract's.
HEADER_INVARIANT_THINKING_FAMILIES = ("gemma4",)
#: the served vLLM reasoning parser per model — gemma4 ONLY (vllm_server refuses every other family's)
REASONING_PARSER_BY_MODEL = {"gemma4e4b": "gemma4"}
#: the campaign's per-model gpu_memory_utilization (common.sh f22_resolve_config: glm needs 0.92 for KV coverage)
GPU_MEM_UTIL_BY_MODEL = {"glm46vflash": "0.92"}
GPU_MEM_UTIL_DEFAULT = "0.9"
SPLIT_MODE_CANDIDATE = {"gemma4": "record_only", "qwen3_5": "implicit_open", "glm4v": "explicit_tags", "nemotron_h": "implicit_open",
                        "llama": "none"}

# --------------------------------------------------------------------------- THE FRONT DOOR's models
#: The front-door models (`reproduce/models.yaml`, read through its ONE reader `reproduce/lib/models.py`; `SISTEER_MODELS_TABLE`
#: overrides the path, as the stage scripts do) are NOT written into MODEL_KEYS / FAMILY_BY_MODEL / SAMPLING_MODE_BY_MODEL / SAMPLING_ROWS
#: — those literals are pinned equal to `contracts` (whose rows for these models are deferred, RELEASE_CHECKLIST 3b) and to the frozen
#: documents' bytes. Their facts are DERIVED from the table, LAZILY (:func:`front_door_facts`): the g8k family from `architecture.family`
#: (:data:`G8K_FAMILY_BY_TABLE_FAMILY`), the arm from `thinking.gsm8k` (held equal to the family's THINKING_MODE_BY_FAMILY), the sampling
#: MODE from the arm (thinking_card / card), the sampling ROW from `serve.generation_config` (the shipped file's keys — the paper's rule,
#: never an invented number), the reasoning parser and the placement from the family's rule (gemma4's parser; the Nemotron fused-kernel
#: overlay's primary-only placement), gpu_mem_util the default. Every accessor below (:func:`family_of`, :func:`sampling_mode_of`,
#: :func:`sampling_row_of`, :func:`reasoning_parser_of`, :func:`gpu_mem_util_of`, :func:`placement_for`, :func:`longsi_scope`,
#: :func:`expected_refusal`) answers from the literals for a paper key — byte for byte — and from the table for a front-door key. Each
#: front-door key has ONE matrix document of its own, `frontdoor_<key>` -> configs/sdfbase/gsm8k/matrix_frontdoor_<key>.json (schema 2,
#: wave `frontdoor_<key>`, wave root `<published root>_frontdoor_<key>`), DERIVED by make_gsm8k_matrix.py from the table's phase views the
#: roster names — never a views file, never a row in the frozen default matrix. The derivation is lazy so that the stdlib-only promise
#: above holds for every paper path: only a `frontdoor_*` name or a front-door key reads the table (PyYAML through the reader; a python
#: without it refuses BY NAME there and nowhere else).
G8K_FAMILY_BY_TABLE_FAMILY = {"qwen3_5_hybrid": "qwen3_5", "gemma4_dense": "gemma4",
                              "nemotron_h_hybrid": "nemotron_h", "llama_dense": "llama"}
#: the arm per models-table `thinking.gsm8k` spelling
G8K_ARM_BY_TABLE_THINKING = {"on": THINKING_ON, "not_available": THINKING_NOT_AVAILABLE}
FRONT_DOOR_MATRIX_PREFIX = "frontdoor_"
#: the phase-view grammar (the misalign generator's, the sampled / padding generators' and the seedrep grader twin's: one tag vocabulary)
#: `s<N>r` = the ReLU-parent subset of phase N (`s2r` = control_relu + constitution_relu at alpha 1; a view outside this grammar
#: renders nothing here while the planner still plans it) — the same grammar in every copy, pinned equal by tests
PHASE_VIEW_RE = re.compile(r"^(u0|s[0-9]+r?p?|s[0-9]+rl|d[0-9]+(p[0-9]+)?(_relul?)?|dm[0-9]+(p[0-9]+)?(_relul?)?)$")
MODELS_TABLE = Path(os.environ.get("SISTEER_MODELS_TABLE") or (REPO / "reproduce" / "models.yaml"))
MODELS_TABLE_READER = REPO / "reproduce" / "lib" / "models.py"
_FRONT_DOOR_FACTS: dict = {}


def _models_reader(path: Path):
    """``reproduce/lib/models.py`` loaded by path (the ONE reader; duplicate keys refused); a table without its reader refuses; a python
    without PyYAML (the node's, before any venv) refuses BY NAME — the paper's paths never come here."""
    if not MODELS_TABLE_READER.is_file():
        raise SystemExit(f"models table {path}: its reader {MODELS_TABLE_READER} is not on this tree — refusing to read it another way")
    spec = importlib.util.spec_from_file_location("_g8k_reproduce_models_table", MODELS_TABLE_READER)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)          # the reader's own dependency (PyYAML) is imported THERE, never here: this module stays stdlib-only
    except ImportError as exc:
        raise SystemExit(f"the front door's models are derived from {path} through {MODELS_TABLE_READER.name}, whose dependency is missing "
                         f"({exc}); run this through the client venv (g8k_common.sh g8k_py) — the node's bare python3 serves the frozen matrices only") from None
    return mod


def _load_models_table(path: Path) -> dict:
    """``reproduce/models.yaml`` through its ONE reader (above)."""
    return _models_reader(path).load_models(str(path))


SHIPPED_GENERATION_CONFIGS = REPO / "configs" / "sdfbase_shipped_generation_config.json"


def _ships_none_row(p: Path, key: str, ck: dict, table: dict) -> dict:
    """The sampling row of a front-door model whose served checkpoint ships no generation_config.json (`serve.generation_config: null`):
    the shipped-file table (configs/sdfbase_shipped_generation_config.json) must record the repo as ``ships_none`` at the table's pinned
    revision, and a MODEL_KEYS model of the same table must serve the SAME checkpoint — its hand-listed row (SAMPLING_ROWS, the paper's
    own row for those bytes) is the front-door model's, with a source naming both facts. Refuses by name otherwise."""
    hf, rev = str(ck.get("hf_repo")), str(ck.get("revision") or "")
    try:
        shipped = (json.loads(SHIPPED_GENERATION_CONFIGS.read_text(encoding="utf-8")).get("models") or {}).get(hf)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{p.name}: model {key!r} serve.generation_config is null and {SHIPPED_GENERATION_CONFIGS.name} is unreadable ({exc})")
    if not isinstance(shipped, dict) or shipped.get("state") != "ships_none" or str(shipped.get("revision") or "") != rev:
        raise SystemExit(f"{p.name}: model {key!r} serve.generation_config is null, but {SHIPPED_GENERATION_CONFIGS.name} does not record "
                         f"{hf} @ {rev[:8]} as `ships_none` (it says {shipped!r}) — the GSM8K rows decode at the shipped generation_config.json "
                         "(the paper's rule); a checkpoint shipping none needs its row stated, never guessed")
    twins = [k for k, mm in table.items() if k in MODEL_KEYS and isinstance(mm, dict)
             and str(((mm.get("serve") or {}).get("checkpoint") or {}).get("hf_repo")) == hf
             and str(((mm.get("serve") or {}).get("checkpoint") or {}).get("revision") or "") == rev]
    if len(twins) != 1 or twins[0] not in SAMPLING_ROWS:
        raise SystemExit(f"{p.name}: model {key!r} serves {hf} @ {rev[:8]}, which ships no generation_config.json, and {len(twins)} paper model "
                         f"of the table serves the same checkpoint ({twins}) — the front door's row is the paper model's hand-listed row "
                         "(SAMPLING_ROWS) for those bytes, or nothing")
    twin = twins[0]
    row = {k: v for k, v in SAMPLING_ROWS[twin].items() if k in SAMPLING_PARAM_KEYS}
    row["source"] = (f"reproduce/models.yaml models.{key}.serve.generation_config is null: {hf} @ {rev[:8]} ships no generation_config.json "
                     f"({SHIPPED_GENERATION_CONFIGS.name}: ships_none), so the row is the paper's for the SAME checkpoint — "
                     f"SAMPLING_ROWS[{twin!r}] ({', '.join(f'{k} {row[k]}' for k in row)}); {SAMPLING_ROWS[twin]['source']}")
    return row


def front_door_facts(table_path: "Path | str | None" = None) -> dict:
    """``{key: {family, arm, sampling_mode, sampling, reasoning_parser, gpu_mem_util, placement, views, hf_repo, revision}}`` for every
    models-table model with phase views — cached per table path. Refusals, by name: a key MODEL_KEYS already names (one key, one table);
    an `architecture.family` outside G8K_FAMILY_BY_TABLE_FAMILY; a `thinking.gsm8k` that is neither `on` nor `not_available`, or one that
    disagrees with the family's arm; a served checkpoint with no `generation_config` (the rows decode at the shipped file — a model
    shipping none needs a stated row, never a guess) or one carrying a non-sampler key. An ABSENT table derives nothing."""
    p = Path(table_path) if table_path is not None else MODELS_TABLE
    if p in _FRONT_DOOR_FACTS:
        return _FRONT_DOOR_FACTS[p]
    out: dict = {}
    if p.is_file():
        reader = _models_reader(p)
        table = reader.load_models(str(p))
        for key, m in table.items():
            if not isinstance(m, dict) or not isinstance(m.get("views"), dict):
                continue
            views = tuple(reader.phase_views_of(m))   # the reader's ONE classification: the grammar's views + the row's declared wave (demo-import M3)
            if not views:
                continue                                  # the paper's four models (main / it) contribute nothing
            if key in MODEL_KEYS:
                raise SystemExit(f"{p.name}: model {key!r} names phase views {views} but is a MODEL_KEYS key — one key, one table")
            tfam = (m.get("architecture") or {}).get("family")
            fam = G8K_FAMILY_BY_TABLE_FAMILY.get(str(tfam))
            if fam is None:
                raise SystemExit(f"{p.name}: model {key!r} architecture.family {tfam!r} maps to no g8k family "
                                 f"({sorted(G8K_FAMILY_BY_TABLE_FAMILY)}); register the family (contracts + this table) before its rows exist")
            think = str((m.get("thinking") or {}).get("gsm8k"))
            arm = G8K_ARM_BY_TABLE_THINKING.get(think)
            if arm is None:
                raise SystemExit(f"{p.name}: model {key!r} thinking.gsm8k {think!r} is neither `on` nor `not_available` — the GSM8K rows run "
                                 "thinking ON (the switch families) or the no-switch arm; an `off` row is not a row of this instrument")
            if THINKING_MODE_BY_FAMILY[fam] != arm:
                raise SystemExit(f"{p.name}: model {key!r} thinking.gsm8k {think!r} disagrees with its family {fam!r}'s arm {THINKING_MODE_BY_FAMILY[fam]!r}")
            serve = m.get("serve") or {}
            gc = serve.get("generation_config")
            ck = serve.get("checkpoint") or {}
            if gc is None:
                # a served checkpoint that SHIPS NO generation_config.json (the table says `generation_config: null`): the row is never
                # guessed — it is the hand-listed row of the MODEL_KEYS model that serves the SAME checkpoint (the paper's own row for
                # those bytes; the worked example qwen35_4b_demo serves the paper's Qwen3.5-4B), admitted only when the shipped-file
                # table records the repo as `ships_none` at that revision; anything else refuses by name
                row = _ships_none_row(p, key, ck, table)
            else:
                if not isinstance(gc, dict) or not gc:
                    raise SystemExit(f"{p.name}: model {key!r} serve.generation_config is {gc!r} — the GSM8K rows decode at the shipped "
                                     "generation_config.json (the paper's rule); a checkpoint shipping none says `null` and shares a paper model's checkpoint")
                bad = [k for k in gc if k not in SAMPLING_PARAM_KEYS]
                if bad:
                    raise SystemExit(f"{p.name}: model {key!r} serve.generation_config carries non-sampler keys {bad} (allowed: {SAMPLING_PARAM_KEYS})")
                row = {k: gc[k] for k in SAMPLING_PARAM_KEYS if k in gc}
                row["source"] = (f"reproduce/models.yaml models.{key}.serve.generation_config — the shipped generation_config.json of "
                                 f"{ck.get('hf_repo')} @ {str(ck.get('revision', ''))[:8]} (the models table's pinned revision; exactly the file's "
                                 f"sampler keys: {', '.join(f'{k} {gc[k]}' for k in row if k != 'source')}); the front door's "
                                 f"{'thinking-ON' if arm == THINKING_ON else 'no-switch'} GSM8K rows")
            out[str(key)] = {"family": fam, "arm": arm,
                             "sampling_mode": SAMPLING_THINKING_CARD if arm == THINKING_ON else SAMPLING_CARD,
                             "sampling": row,
                             "reasoning_parser": "gemma4" if fam == "gemma4" else None,      # REASONING_PARSER_BY_MODEL's rule is the family's
                             "gpu_mem_util": GPU_MEM_UTIL_DEFAULT,
                             "placement": PLACEMENT_PRIMARY if fam == "nemotron_h" else PLACEMENT_ANY,   # PRIMARY_ONLY_MODELS' rule: the fused kernels
                             "views": views, "hf_repo": ck.get("hf_repo"), "revision": ck.get("revision")}
    _FRONT_DOOR_FACTS[p] = out
    return out


def front_door_model_keys() -> tuple:
    return tuple(front_door_facts())


def all_model_keys() -> tuple:
    """MODEL_KEYS + the front door's (the admissible model keys of a matrix row)."""
    return (*MODEL_KEYS, *front_door_model_keys())


def _fd(key: str) -> "dict | None":
    return front_door_facts().get(key) if key not in MODEL_KEYS else None


def family_of(key: str) -> "str | None":
    """The g8k family of a model key: FAMILY_BY_MODEL for a paper key, the table's for a front-door key, None for neither."""
    fam = FAMILY_BY_MODEL.get(key)
    if fam is not None:
        return fam
    f = _fd(key)
    return f["family"] if f else None


def sampling_mode_of(key: str) -> str:
    if key in SAMPLING_MODE_BY_MODEL:
        return SAMPLING_MODE_BY_MODEL[key]
    f = _fd(key)
    if f is None:
        raise SystemExit(f"sampling_mode_of: {key!r} is not one of {all_model_keys()}")
    return f["sampling_mode"]


def sampling_row_of(key: str) -> dict:
    if key in SAMPLING_ROWS:
        return dict(SAMPLING_ROWS[key])
    f = _fd(key)
    if f is None:
        raise SystemExit(f"sampling_row_of: {key!r} is not one of {all_model_keys()}")
    return dict(f["sampling"])


def reasoning_parser_of(key: str) -> "str | None":
    f = _fd(key)
    return f["reasoning_parser"] if f else REASONING_PARSER_BY_MODEL.get(key)


def gpu_mem_util_of(key: str) -> str:
    f = _fd(key)
    return f["gpu_mem_util"] if f else GPU_MEM_UTIL_BY_MODEL.get(key, GPU_MEM_UTIL_DEFAULT)


def front_door_matrix_name(key: str) -> str:
    return f"{FRONT_DOOR_MATRIX_PREFIX}{key}"


def front_door_key_of(name: str) -> str:
    """The models-table key behind a `frontdoor_<key>` matrix name; refuses a name that is not a front-door key's, listing them."""
    if not str(name).startswith(FRONT_DOOR_MATRIX_PREFIX):
        raise SystemExit(f"{name!r} is not a front-door matrix name ({FRONT_DOOR_MATRIX_PREFIX}<models.yaml key>)")
    key = str(name)[len(FRONT_DOOR_MATRIX_PREFIX):]
    if key not in front_door_facts():
        raise SystemExit(f"{name!r}: {key!r} is not a front-door model of {MODELS_TABLE.name} (its models with phase views: "
                         f"{list(front_door_model_keys())})")
    return key


def front_door_matrices() -> dict:
    """``{frontdoor_<key>: configs/sdfbase/gsm8k/matrix_frontdoor_<key>.json}`` — the front door's registry beside MATRICES (never in it:
    the frozen names and their views files are pinned)."""
    return {front_door_matrix_name(k): REPO / "configs" / "sdfbase" / "gsm8k" / f"matrix_{FRONT_DOOR_MATRIX_PREFIX}{k}.json" for k in front_door_model_keys()}


def front_door_wave(key: str) -> str:
    """The launchable wave of a front-door matrix (its out roots: `controls_frontdoor_<key>/`, out_root_prefix_for's default rule)."""
    return f"{FRONT_DOOR_MATRIX_PREFIX}{key}"


def front_door_wave_root_rel(key: str) -> str:
    """A front-door matrix's OWN wave root: `<published root>_frontdoor_<key>` (a schema-2 matrix never shares the published one)."""
    return f"{WAVE_ROOT_REL}_{FRONT_DOOR_MATRIX_PREFIX}{key}"
#: model-based PLACEMENT: the Nemotron fused-kernel serving overlay builds and is attested on the PRIMARY
#: cluster alone (the overflow cluster's nodes carry CUDA 13.0 only), so EVERY Nemotron row — its 8,192 rows,
#: its overlap leg, any of its rungs — carries `primary`; every other row may overflow (`any`)
PLACEMENT_PRIMARY = "primary"
PLACEMENT_ANY = "any"
PRIMARY_ONLY_MODELS = ("nemotron3nano4b",)
PRIMARY_CLUSTER = "cluster_a"

# --------------------------------------------------------------------------- sampling: each model's THINKING-CARD row (spec, answer 1)
SAMPLING_THINKING_CARD = "thinking_card"
#: llama-paper-rows (P6): the SHIPPED `card` row for a model with NO thinking mode — a "thinking card" would be a misnomer
SAMPLING_CARD = "card"
#: the sampling MODE each model's rows run under (`--sampling <mode>`; the out root's last segment): the four thinking
#: families' `thinking_card`, the no-switch family's `card`. Pinned equal to contracts.SAMPLING_TABLE_BY_MODE by test.
SAMPLING_MODE_BY_MODEL = {"gemma4e4b": SAMPLING_THINKING_CARD, "qwen35_4b": SAMPLING_THINKING_CARD, "glm46vflash": SAMPLING_THINKING_CARD,
                          "nemotron3nano4b": SAMPLING_THINKING_CARD, "llama8b": SAMPLING_CARD}
#: THE LITERAL ROWS with provenance — the MATH-500 primary rows at the vendor values, byte-identical here;
#: a row carries EXACTLY the keys its source names (absent = the request carries no such key). Pinned equal to
#: contracts.THINKING_SAMPLING_BY_MODEL_KEY (the four) / contracts.SAMPLING_BY_MODEL_KEY (llama8b's `card` row) by test.
SAMPLING_ROWS: dict[str, dict] = {
    "llama8b": {
        "temperature": 0.6, "top_p": 0.9,
        "source": ("generation_config.json of NousResearch/Meta-Llama-3.1-8B-Instruct @ d10aef79: do_sample true, temperature 0.6, "
                   "top_p 0.9, no top_k, no penalty — re-read 2026-09-11 from the HF hub at the pinned revision and from the primary "
                   "cluster's cohort cache (identical; llama-paper-rows P6); the base checkpoint NousResearch/Meta-Llama-3.1-8B @ "
                   "1f47e50c ships the same numbers (_from_model_config true). The GSM8K paired-200 Llama rows run this `card` row "
                   "under the no-switch arm (thinking: not available)")},
    "gemma4e4b": {
        "temperature": 1.0, "top_p": 0.95, "top_k": 64,
        "source": ("generation_config.json of google/gemma-4-E4B-it @ee0ef602 (do_sample true; vendor default, not "
                   "reasoning-tuned; no penalty term) — re-read 2026-09-09; identical to the sampled battery's preset, so "
                   "gemma has no battery row")},
    "qwen35_4b": {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
        "source": ("Qwen/Qwen3.5-4B @851bf6e8 ships no generation_config.json (.no_exist marker); the Qwen3.5 model "
                   "card's THINKING-mode 'precise' row as reported to the project = the Qwen3 card's thinking-mode "
                   "guidance (T 0.6 / top_p 0.95 / top_k 20 / min_p 0; penalties only against endless repetition at a "
                   "stated cost) — the PRIMARY row: mathematics legitimately repeats tokens")},
    "glm46vflash": {
        "temperature": 0.8, "top_p": 0.6, "top_k": 2,
        "source": ("generation_config.json of zai-org/GLM-4.6V-Flash @411bb4d7 (do_sample true; no penalty term in the "
                   "file) — read 2026-09-09 over ssh from the cohort cache")},
    "nemotron3nano4b": {
        "temperature": 1.0, "top_p": 0.95,
        "source": ("generation_config.json of nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 @dfaf35de (temperature 1.0, top_p "
                   "0.95; no top_k, no do_sample key, no penalty) AND the card's explicit reasoning-task recommendation "
                   "'temperature=1.0 and top_p=0.95 are recommended for reasoning tasks' — read 2026-09-09")},
}
SAMPLING_PARAM_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")

# --------------------------------------------------------------------------- the run's literals (mirrored from contracts; test-pinned)
CONTROL = "gsm8k"                          # contracts.CONTROL_GSM8K — plans the ONE task inspect/gsm8k_paired200
INSPECT_TASK = "gsm8k_paired200"           # contracts.GSM8K_PAIRED_TASK_NAME — the CLI cap flags' and plan_task's spelling
ROW_TASK = "inspect/gsm8k_paired200"       # contracts.GSM8K_STANDARD_TASK — the ROW task the producer files (the ladder
                                           # record's / subset's `task`, the outcomes file's name); the rung subset is
                                           # compared against THIS spelling, never the unprefixed one (review B-P1.2)
THINKING_MODE = "on"                       # DECLARED, never assumed: the literal the authorisation compares
CAP = 8192                                 # the explicit ON cap; the setter carries it
RUNG_CAP = 16384                           # the standing ladder's ceiling (cap_ladder.CAP_CEILING): the ONE rung, no rung above
RUNGS = (CAP, RUNG_CAP)
SAMPLES = 1
IDENTITY_VERSION = 2
UPSTREAM_TIMEOUT_S = 7200                  # contracts.PROXY_UPSTREAM_TIMEOUT_S
LENS_SALT_POLICY = "attested"
N_ITEMS = 200                              # contracts.GSM8K_PAIRED_N
#: the geometry entry per rung (chosen FROM the measured 10-shot prompts; L pinned equal to the
#: table by test). MEASURED 2026-09-10 on the grading host (20 artifacts, scripts/tinycap_oos_prompt_budget.py --task
#: gsm8k_paired200): the longest ON prompt is 2,462 tokens (qwen35_4b, prompted class; bare 2,373;
#: gemma 2,398 / 2,310; nemotron 2,424 / 2,336; glm 2,216 / 2,128 — the safety system turn adds a constant 88-89), so the
#: 8,192 pass FITS the existing capability:gen_long_thinking (2,462 + 8,192 = 10,654 <= 10,752, margin 98 — the
#: reuse rule), which is also the entry the contracts' class map resolves for a thinking-ON GSM8K control; ONLY the rung
#: needs its own entry: 2,462 + 16,384 = 18,846 -> block floor 18,848 -> capability:gen_long_gsm8k_rung = 18,944 (the
#: MAX-PROMPT branch, like MATH-500's). S per model: 54 / 33 / 73 / 73 at 10,752; 31 / 19 / 41 / 41 at 18,944.
GEOMETRY_EVAL_BY_RUNG = {CAP: "capability:gen_long_thinking", RUNG_CAP: "capability:gen_long_gsm8k_rung"}
GEOMETRY_L_BY_RUNG = {CAP: 10752, RUNG_CAP: 18944}
#: the LONG-SI prompt class (prompted__si_constitution; user ruling 2026-09-12): MEASURED 2026-09-12 on the grading host with
#: scripts/tinycap_oos_prompt_budget.py --task gsm8k_paired200 --arm not_available --system-prompt-cell prompted__si_constitution
#: --config configs/sdfbase/gsm8k/sdfbase_full22_llama8b.gsm8k.yaml --model-key llama8b (200 renders under the pinned Llama
#: tokenizer's no-switch render): prompt max 8,007 (p99 7,975; the SI adds a constant 5,872), so NEITHER shared entry holds it
#: (8,007 + 8,192 = 16,199 > 10,752; 8,007 + 16,384 = 24,391 > 18,944 and > MATH's longsi 24,064). Its OWN entries, the
#: MAX-PROMPT branch: 16,199 -> block floor 16,208 -> capability:gen_long_gsm8k_longsi = 16,384 (headroom 176); 24,391 ->
#: block floor 24,400 -> capability:gen_long_gsm8k_rung_longsi = 24,576 (headroom 176). S for llama8b at its 475,136 budget:
#: 29 at 16,384 (= the certified standup geometry itself), 19 at 24,576. L pinned equal to the table by test.
GEOMETRY_EVAL_BY_RUNG_LONGSI = {CAP: "capability:gen_long_gsm8k_longsi", RUNG_CAP: "capability:gen_long_gsm8k_rung_longsi"}
GEOMETRY_L_BY_RUNG_LONGSI = {CAP: 16384, RUNG_CAP: 24576}
#: the LONG-SI class on the FOUR full22 families: SEPARATELY SCOPED entries —
#: the Llama-scoped ones above stay EXACTLY as registered (the Llama matrix, its launcher and tests bind them). MEASURED
#: 2026-09-13 on the grading host (scripts/tinycap_oos_prompt_budget.py --task gsm8k_paired200 --arm on --cap 8192 --config
#: configs/sdfbase_full22_<key>.yaml --model-key <key> --system-prompt-cell prompted__si_constitution --offline; 200 renders
#: each under the pinned tokenizers' thinking-ON render): prompt max gemma4e4b 8,338 / qwen35_4b 8,332 / nemotron3nano4b
#: 8,288 (p99 8,314 / 8,304 / 8,259) — instruction length is a property of the rendered prompt in a template: the
#: Llama entry's 16,384 does NOT hold 8,338 + 8,192 = 16,530. The MAX-PROMPT branch: 16,530 -> block floor 16,544 ->
#: capability:gen_long_gsm8k_longsi_f22 = 16,640 (headroom 96); 8,338 + 16,384 = 24,722 -> 24,736 ->
#: capability:gen_long_gsm8k_rung_longsi_f22 = 24,832 (headroom 96). S at the three models' budgets: 35 / 43 / 47 at 16,640;
#: 23 / 29 / 31 at 24,832. L pinned equal to the table by test.
GEOMETRY_EVAL_BY_RUNG_LONGSI_F22 = {CAP: "capability:gen_long_gsm8k_longsi_f22", RUNG_CAP: "capability:gen_long_gsm8k_rung_longsi_f22"}
GEOMETRY_L_BY_RUNG_LONGSI_F22 = {CAP: 16640, RUNG_CAP: 24832}
#: the long-SI class's FAMILY SCOPE: the no-switch family -> PR-A's Llama-scoped entries; every full22 switch family -> the
#: `_f22` entries. The scope is the FAMILY (contracts.FAMILY_BY_MODEL_KEY, mirrored here as FAMILY_BY_MODEL), so a new model
#: of a measured family resolves without a table edit; a family outside both refuses by name.
LONGSI_SCOPE_LLAMA = "llama"
LONGSI_SCOPE_F22 = "f22"
LONGSI_SCOPE_BY_FAMILY = {"gemma4": LONGSI_SCOPE_F22, "qwen3_5": LONGSI_SCOPE_F22, "glm4v": LONGSI_SCOPE_F22, "nemotron_h": LONGSI_SCOPE_F22,
                          "llama": LONGSI_SCOPE_LLAMA}
GEOMETRY_LONGSI_BY_SCOPE = {LONGSI_SCOPE_LLAMA: (GEOMETRY_EVAL_BY_RUNG_LONGSI, GEOMETRY_L_BY_RUNG_LONGSI),
                            LONGSI_SCOPE_F22: (GEOMETRY_EVAL_BY_RUNG_LONGSI_F22, GEOMETRY_L_BY_RUNG_LONGSI_F22)}


#: THE COMMON GEOMETRY CLASS a MATRIX may declare (S x L is an identity
#: field shared by EVERY arm of a leg): `geometry_class: common_f22` on a schema-2 views file puts EVERY row of that matrix — short cells
#: included — on the common pair capability:gen_long_gsm8k_common_f22 (16,640 at the 8,192 pass) / capability:gen_long_gsm8k_rung_common_f22
#: (24,832 at the rung), copies of the `_f22` long-SI entries' MEASURED values registered under a common name (geometry.py). Kernel S:
#: 524,287 // 16,640 = 31 and 524,287 // 24,832 = 21 under the LoRA kernel index ceiling (D 4096). The per-prompt-class rule stays the
#: default for every matrix that declares no class. The class is FAMILY-SCOPED like the entries it copies (measured on the full22 switch
#: families): a llama8b row under it refuses by name.
GEOMETRY_CLASS_COMMON_F22 = "common_f22"
GEOMETRY_EVAL_BY_RUNG_COMMON_F22 = {CAP: "capability:gen_long_gsm8k_common_f22", RUNG_CAP: "capability:gen_long_gsm8k_rung_common_f22"}
GEOMETRY_L_BY_RUNG_COMMON_F22 = {CAP: 16640, RUNG_CAP: 24832}
GEOMETRY_CLASSES = {GEOMETRY_CLASS_COMMON_F22: (GEOMETRY_EVAL_BY_RUNG_COMMON_F22, GEOMETRY_L_BY_RUNG_COMMON_F22)}
GEOMETRY_CLASS_SCOPE = {GEOMETRY_CLASS_COMMON_F22: LONGSI_SCOPE_F22}
GEOMETRY_CLASS_KEY = "geometry_class"


def prompt_class(cell: str) -> str:
    """The prompt class of a cell: `long_si` for the constitution prompted cell, `short` for every other cell."""
    return PROMPT_CLASS_LONG_SI if cell in LONG_SI_PROMPTED_CELLS else PROMPT_CLASS_SHORT


def longsi_scope(model_key: str) -> str:
    """The long-SI class's family scope of a model (LONGSI_SCOPE_BY_FAMILY over the model's family); an unregistered model
    or an unscoped family refuses by name."""
    fam = family_of(model_key)
    if fam is None:
        raise SystemExit(f"longsi_scope: {model_key!r} is not one of {all_model_keys()}")
    try:
        return LONGSI_SCOPE_BY_FAMILY[fam]
    except KeyError:
        raise SystemExit(f"longsi_scope: family {fam!r} ({model_key}) has no long-SI geometry scope — measure its constitution "
                         "prompts and register its entries (plan P3) before a constitution row of it is a production row") from None


def geometry_for_cell(cell: str, rung: int, model_key: str | None = None, geometry_class: str | None = None) -> tuple[str, int]:
    """(geometry entry, L) for a cell at a rung — selected by the cell's PROMPT CLASS, never by 'prompted or not' (the safety
    system turn adds 88 tokens; the constitution one ~5,900), and, for the LONG-SI class, by the model's FAMILY SCOPE (E2 /
    P3: the Llama entries hold the Llama render; the four full22 families' `_f22` entries hold theirs). A long-SI cell with
    no `model_key` refuses by name: the scope is part of the selection, never a default.

    A MATRIX-DECLARED `geometry_class` (qwen-seedrep-native-battery, `GEOMETRY_CLASSES`) overrides the prompt-class rule for EVERY
    cell of that matrix — the common pair for short and long-SI cells alike, so every arm of a leg shares one S x L; an unknown class,
    or a class whose family scope is not the model's (a llama8b row under the f22-measured common class), refuses by name. `None`
    (every other matrix) is the rule above, byte for byte."""
    if geometry_class is not None:
        if geometry_class not in GEOMETRY_CLASSES:
            raise SystemExit(f"geometry_for_cell({cell!r}, {rung}): geometry_class {geometry_class!r} is not one of {sorted(GEOMETRY_CLASSES)}")
        if model_key is None:
            raise SystemExit(f"geometry_for_cell({cell!r}, {rung}): the geometry class {geometry_class!r} is scoped by family — pass model_key")
        if longsi_scope(model_key) != GEOMETRY_CLASS_SCOPE[geometry_class]:
            raise SystemExit(f"geometry_for_cell({cell!r}, {rung}): geometry_class {geometry_class!r} was measured on the "
                             f"{GEOMETRY_CLASS_SCOPE[geometry_class]!r} family scope; {model_key!r} is {longsi_scope(model_key)!r} — "
                             "measure its prompts and register its own class before a row of it is a production row")
        evals, ls = GEOMETRY_CLASSES[geometry_class]
        return evals[rung], ls[rung]
    if prompt_class(cell) == PROMPT_CLASS_LONG_SI:
        if model_key is None:
            raise SystemExit(f"geometry_for_cell({cell!r}, {rung}): the long-SI class is scoped by family — pass model_key")
        evals, ls = GEOMETRY_LONGSI_BY_SCOPE[longsi_scope(model_key)]
        return evals[rung], ls[rung]
    return GEOMETRY_EVAL_BY_RUNG[rung], GEOMETRY_L_BY_RUNG[rung]
#: the thinking-OFF overlap leg: greedy at the OFF table's 512 — statement == resolution, no
#: setter — on the OFF class's entry: the OFF renders' max is 2,375 (qwen), + 512 = 2,887 <= capability:gen_long's 4,608
#: (margin 1,721; S 128 / 78 / 170 / 170), the entry the class map resolves for a thinking-OFF GSM8K control
OVERLAP_THINKING = "off"
OVERLAP_SAMPLING = "greedy"
OVERLAP_CAP = 512
OVERLAP_GEOMETRY_EVAL = "capability:gen_long"
OVERLAP_GEOMETRY_L = 4608
OVERLAP_OUT_REL_TEMPLATE = "overlap/{key}/none/greedy_off512"
#: driver flag spellings (contracts.FLAG_*; pinned equal by test — the three flags by NAME)
FLAG_THINKING = "--thinking"
FLAG_SAMPLING = "--sampling"
FLAG_SAMPLES = "--samples"
FLAG_LEDGER_URL = "--ledger-url"
FLAG_OOS_PROMPT_BUDGET = "--oos-prompt-budget"
FLAG_ALLOW_OOS_THINKING = "--allow-out-of-support-thinking"
FLAG_EXPECT_OBJECT_SHA256 = "--expect-object-sha256"
FLAG_MAX_TOKENS_BY_TASK = "--max-tokens-by-task"        # the STATEMENT (identity v2: must equal the resolved cap)
FLAG_ON_CAP_BY_TASK = "--on-cap-by-task"                # the bounded SETTER: the same grammar, v2 only
FLAG_ITEM_SUBSET = "--item-subset"                      # the rung's subset file
FLAG_ATTEMPT_ID = "--attempt-id"                        # the attempt seam: the ledger and the driver
FLAG_TINYCAP_IDENTITY_VERSION = "--tinycap-identity-version"
FLAG_UPSTREAM_TIMEOUT_S = "--upstream-timeout-s"
FLAG_LENS_SALT_POLICY = "--lens-salt-policy"
FLAG_REASONING_PARSER = "--reasoning-parser"
FLAG_CELL = "--cell"
FLAG_BANK_VIEW = "--bank-view"                          # the row's VIEW to the driver (llama-paper-rows: row-level provenance —
                                                        # object_artifact_sha256 / route_kind / view / producer_tree_sha on every row)
#: the DECLARED spelling (`<cell>@thinking`) is NOT re-typed here: the matrix row carries `standup_cell` /
#: `standup_harness_cell`, derived ONCE at generation time by the contract's own `contracts.cell_spellings(cell,
#: declared=)` (g8k_views.select_rows) — the standup's `--cell`, its `--expect-object-sha256` key and the fresh table's
#: `lens_served_positions` key all read that one spelling (review B-P1.1)
#: the file names the sbatch, the driver and the planner agree on
LEDGER_FILENAME = "usage_ledger.jsonl"
LEDGER_ENDPOINT_FILENAME = "ledger_endpoint.json"
OOS_PROMPT_BUDGET_FILENAME = "oos_prompt_budget.json"
LADDER_STATE_FILENAME = "ladder_state.json"
RUNG_SUBSET_FILENAME = "rung_subset.json"
CAP_LADDER_FILENAME = "cap_ladder.json"                 # cap_ladder.RECORD_NAME (pinned equal by test)
#: the ENGINE-RESOLVED SAMPLER attestation (the per-kind partition, the second tag source and `--require-wired` follow
#: the MATH-500 launcher's shape exactly): the standup opts
#: into vLLM's `--enable-log-requests` through F22_STANDUP_EXTRA (g8k_common.sh g8k_standup_extra), and after the driver the
#: launcher runs scripts/sdfbase/record_resolved_sampler.py over the server's own log — the sampler the ENGINE resolved (its
#: defaults merged in), compared within request classes — into the cell's out root, beside the store, so it rides the pull; the
#: analysis reads it (tinycap_gsm8k_validators.resolved_sampler_record, pinned equal) as a FINDING, never a gate. The client cap
#: is the row's rung and the reference the row's PINNED sampling row (`--expect k=v` per key of the numeric projection; never the
#: driver's declaration). The PROBE class is partitioned BY KIND (`--probe-expect kinds`): the driver's own tag per probe attempt
#: (`X-Tinycap-Task: probe/<kind>` + the wire body's max_tokens, written to `<out_root>/request_evidence.jsonl` —
#: contracts.request_evidence_record) and the STANDUP's load-check sidecar in the serving dir
#: (`load_check.request_evidence.jsonl`, harness_evidence.evidence_path_for) — TWO tag sources, one per EMITTER (disjoint request
#: sets; never the evidence file AND the usage ledger for the same requests, whose rows would double the counts) — each kind
#: against its own pinned reference (render -> the card row = --expect; bites, lens_on_bite, load_check -> greedy), the shape
#: cross-checked against the recorder's PROBE_KIND_TABLE; an untagged or mis-shaped probe is UNCLASSIFIED, a finding, never a
#: declaration to widen. NO `--probe-cap` is passed: under `kinds` the recorder's default is every cap its kind table pins —
#: RECORDER_PROBE_CAPS mirrors them (pinned equal by test) for the ONE check the launcher owns, that no client cap (8,192 /
#: 16,384; the smoke's 1,536) collides with a kind cap. `--require-wired`: this launcher DECLARES the opt-in on every row and
#: reads AFTER the leg, so a record proving the instrument was never wired (UNREADABLE 3 / FLAG_SET_NO_REQUESTS_YET 4) is logged
#: BY NAME by g8k_record_resolved_sampler; every verdict ABOUT the sampler still exits 0 and never decides the leg.
RESOLVED_SAMPLER_FILENAME = "resolved_sampler.json"
RESOLVED_SAMPLER_SCRIPT_REL = "scripts/sdfbase/record_resolved_sampler.py"
REQUEST_EVIDENCE_FILENAME = "request_evidence.jsonl"          # the driver's tag log beside the store (contracts.REQUEST_EVIDENCE_FILENAME)
RECORDER_PROBE_EXPECT = "kinds"
RECORDER_PROBE_TAGS_FILENAME = REQUEST_EVIDENCE_FILENAME
RECORDER_LOAD_CHECK_TAGS_FILENAME = "load_check.request_evidence.jsonl"   # the standup's load-check sidecar (harness_evidence.EVIDENCE_SUFFIX)
#: the kind caps, from the recorder's PROBE_KIND_TABLE: probe/bites 48, the standup's positive_control 64, probe/lens_on_bite 256,
#: probe/render 512 and the driver's ON-arm render retry 2,048 (the load check has no pinned cap: it fires at the run's cap); and,
#: since the steer rows' logprob liveness gate, the driver's probe/logprob_bites at 1 (contracts.LOGPROB_BITES_MAX_TOKENS) —
#: the MATH-500 mirror (math500/cells.py) moved with it and so does this one; the table is the roster, the mirrors follow it.
RECORDER_PROBE_CAPS = (1, 48, 64, 256, 512, 2048)
#: the MATH-500 behaviour mirrored here for the GSM8K steer rows: the driver's liveness-gate flag and the value every EXTERNAL
#: (steer) row passes — the LOGPROB statistic rules the positive control on an alphasteer / static_r cell
#: (contracts.LOGPROB_LIVENESS_KINDS; the driver scopes it by kind; fidelity.check_logprob_steering_bites over the bites prompts with the
#: scale-0 twin's REPEAT as the measured noise floor), the text bites reported beside it, the record stating that logprob distance is
#: not a dose axis. A published / ext row passes no such flag (its driver argv is byte-identical; pinned by the sbatch test). The text
#: statistic refused the low-dose AlphaSteer points as "inert" on MATH-500 (array 495677: gemma lambda 0.057 and nemotron lambda 0.049
#: at 0/5 while qwen lambda 0.548 passed at 2/5) — a correct small edit need not flip a greedy token.
FLAG_LENS_LIVENESS = "--lens-liveness"                          # contracts.FLAG_LENS_LIVENESS
LENS_LIVENESS_STEER = "logprob"                                 # contracts.LENS_LIVENESS_LOGPROB
#: the Inspect provider's env names (contracts.INSPECT_BASE_URL_ENV / INSPECT_API_KEY_ENV)
INSPECT_BASE_URL_ENV = "TINYCAP_BASE_URL"
INSPECT_API_KEY_ENV = "TINYCAP_API_KEY"
#: the new wave root (never the full22 wave's, never the MATH-500 one)
WAVE_ROOT_REL = "exp/gsm8k_paired200_paper_objects"
SERVE_SUBDIR = "gsm8k"                                  # $SERVE_DIR/gsm8k/{endpoint.json, routing_table.json, ledger.log, driver.log}
RUNG_DIR_REL = "rung16384"                              # $WAVE_ROOT/rung16384/<TAG>/<cell_slug>/rung_subset.json
#: the NO-SWITCH diagnostic smoke — the certification of a switch-less family's
#: render fact (`smoke_not_available.sbatch` + `smoke_not_available_probe.py`, the MATH-500 smoke's shape): ONE GPU, the
#: matrix's `none` row on the BASE route, SMOKE_N paired items at SMOKE_CAP under the row's `card` sampling row, SMOKE_CONCURRENCY
#: in flight through the usage ledger; every record `diagnostic: true`, no tinycap store, out root
#: $WAVE_ROOT/<SMOKE_DIR_REL>/<family>/smoke_<job>/ (never controls/). The PASSING run's completions, report and ledger are
#: COMMITTED under SMOKE_COMMITTED_DIR_REL/<family>/ (contracts.NO_SWITCH_SMOKE_COMPLETIONS_DIR, pinned equal) and re-read by
#: tests/test_no_switch_smoke_real_completions.py before the fact reads `certified`.
SMOKE_DIR_REL = "diagnostic"
SMOKE_PROBE_SCRIPT_REL = "scripts/sdfbase/gsm8k/smoke_not_available_probe.py"
SMOKE_TASK_LABEL = "smoke/gsm8k"                        # the ledger's X-Tinycap-Task on every smoke request
SMOKE_N = 24
SMOKE_CAP = 1536
SMOKE_CONCURRENCY = 8
SMOKE_COMMITTED_DIR_REL = "data/tinycap/no_switch_smoke"
#: the datasets the wrapper loads OFFLINE on the cluster (the first is prepared on both clusters): the guard names both
DATASETS_REQUIRED = ("openai___gsm8k", "madrylab___gsm8k-platinum")
#: the OFFLINE config dir `datasets` 5.0.1 looks for under each dataset dir (verified on both clusters): inspect_evals'
#: `hf_dataset("openai/gsm8k", data_dir="main", revision=cc7b047b…)` is looked up OFFLINE as `default-data_dir=main` while an
#: ONLINE prepare names its arrow cache by a kwargs hash (`default-8cbc416ed05c3005`), so the prepared per-user caches carry
#: `openai___gsm8k/default-data_dir=main` as a SYMLINK to the hash-named dir; platinum's config `main` IS named `main`
#: offline. The guard accepts a symlink (it follows it) and refuses a dangling one by name.
DATASETS_CONFIG_DIRS = {"openai___gsm8k": "default-data_dir=main", "madrylab___gsm8k-platinum": "main"}
DATASETS_ONLINE_CONFIG_DIR_EXAMPLE = "default-8cbc416ed05c3005"   # the hash-named dir the pinned online prepare produced
#: --time classes at 8,192 (re-costed from the MATH-500 first cell — 500 items in ~9-10 min at S = 44
#: with a 16,384 cap): 200 items <= 5 min base / <= 10 min lens; the cap tail 8,192 / 25 tok/s ~ 5.5 min base, / 12 ~ 11 min
#: lens; standup 3-16 min -> in-support 16 + 5 + 5.5 ~ 27 min, fitted 16 + 10 + 11 ~ 37 min, GLM fitted (the slowest
#: per-sequence decoder) ~ 50 min; each x2 -> 1 h / 1.5 h / 2 h. The overlap OFF leg: 16 + 2 + 512 / 25 tok/s ~ 19 min x2 ->
#: 45 min. Rung rows are NOT a class here: ladder_plan.py computes T_alloc = 2 x T_pred per rung from the subset.
TIME_IN_SUPPORT = "0-01:00:00"
TIME_FITTED = "0-01:30:00"
TIME_FITTED_GLM = "0-02:00:00"
TIME_OVERLAP = "0-00:45:00"
#: llama-paper-rows (P14): the Llama rows' classes, right-sized from the lane's measured stacked-lens throughput (~1,000
#: tok/s aggregate at 44 x 10,752, ~23 tok/s per sequence; ~200-300-token completions with no thinking channel; an at-cap
#: loop ~6 min in parallel; 5-7 min job overhead): 0-00:45:00 in support, 0-01:00:00 fitted (~2x the estimate)
TIME_IN_SUPPORT_LLAMA = "0-00:45:00"
TIME_FITTED_LLAMA = "0-01:00:00"
#: the --time class TAGS (the plan's `-J g8k-<tag>`), per model scope: the full22 families' three classes and the Llama rows'
#: two share ONE literal (TIME_IN_SUPPORT == TIME_FITTED_LLAMA == 0-01:00:00), so a tag is never looked up by the literal alone
#: (one merged dict once read the published in-support class as `fitted-llama`) — class_tag_for reads the rows' model scope
CLASS_TAG_F22 = {TIME_IN_SUPPORT: "in-support", TIME_FITTED: "fitted", TIME_FITTED_GLM: "fitted-glm"}
CLASS_TAG_LLAMA = {TIME_IN_SUPPORT_LLAMA: "in-support-llama", TIME_FITTED_LLAMA: "fitted-llama"}
CLASS_TAG = {**CLASS_TAG_F22, **CLASS_TAG_LLAMA}   # every class literal -> SOME tag (diagnostics that hold no rows); the plan uses class_tag_for


def class_tag_for(time_limit: str, rows: list[dict]) -> str:
    """The `-J` tag of one --time class of a submission: the Llama lane's spelling when every row of the class is a Llama row,
    the full22 families' otherwise; a class mixing the two scopes (one literal, two lanes — never one matrix's plan) refuses."""
    llama = {r["model_key"] == "llama8b" for r in rows}
    if llama == {True, False}:
        raise SystemExit(f"class {time_limit}: rows of the Llama lane and of the full22 families share one --time class — plan them "
                         f"from their own matrices ({[r['job_id'] for r in rows]})")
    table = CLASS_TAG_LLAMA if llama == {True} else CLASS_TAG_F22
    if time_limit not in table:
        raise SystemExit(f"class {time_limit}: not a --time class of the {'Llama lane' if llama == {True} else 'full22 families'} "
                         f"({sorted(table)})")
    return table[time_limit]
#: the expected stamp triple per class — (cache_salt_regime, resumed_prefill_policy, render_regime) — is NOT spelled here:
#: the matrix row's `expected_stamps` is derived at generation time by the contract's `contracts.g8k_expected_stamps(fitted=,
#: declared=)` (g8k_views.select_rows), and the analysis' reader validates every row against the same function. A frozen
#: literal once rode here with a triple that was not a row of the contract.
#: the launcher's readiness wait for the usage ledger (tinycap_served.LEDGER_WAIT_S)
LEDGER_WAIT_S = 180
STATUS_LAUNCH = "launch"
STATUS_HELD = "held"
#: a DESIGN-ONLY status (plan P5): a row whose store is not yet owed — the analysis files it, no launcher runs it
STATUS_DEFERRED = "deferred"
STATUSES = (STATUS_LAUNCH, STATUS_HELD, STATUS_DEFERRED)
#: WAVES (plan P2 / P5): a row without a `wave` key is a PUBLISHED row of its matrix — the
#: frozen 42 (COMPLETE at 8f5f1495) or the Llama matrix's 12 (COMPLETE at its own pin) — reused byte-for-byte and
#: never resubmitted; `ext` = the eight-arm prompted baselines (extensions:), `steer` = the AlphaSteer / Arditi rung (steering:)
WAVE_PUBLISHED = "published"
WAVE_EXT = "ext"
WAVE_STEER = "steer"
WAVE_ITDOC = "itdoc"               # E3: the instruct-direct re-fit rows (itdoc:), generated from the it-doc admission record
#: the `qwen_seedrep` matrix's 18 rows (views_qwen_seedrep.yaml `wave: qwen_seedrep`) —
#: LAUNCHABLE (not in COMPLETED_WAVES; the relu_psr precedent: a new matrix is NOT a new wave, and a schema-2 document without
#: `wave:` yields rows `row_wave` reads as the COMPLETE `published` wave), out roots under controls_qwen_seedrep/ of its own root
WAVE_QWEN_SEEDREP = "qwen_seedrep"
#: arditi-allpos-cap (2026-09-17): the all-positions Arditi arm's own wave, under its own matrix's own wave root — so the
#: ONE row's store, when it exists, is the only thing under that tree and no other wave's leg can be mistaken for it
WAVE_ALLPOS = "allpos"
#: the waves whose legs are COMPLETE and reused byte-for-byte — `plan --wave <one of these>` and the batch boundary refuse by name
COMPLETED_WAVES = (WAVE_PUBLISHED,)
#: the WAVE-ISOLATED out-root prefix per wave (F11a: a pull of one subtree never carries another wave's leg; budgets/,
#: nemo_backend/ and rung16384/ stay shared under the one wave root)
OUT_ROOT_PREFIX_PUBLISHED = "controls"
OUT_ROOT_PREFIX_BY_WAVE = {WAVE_PUBLISHED: OUT_ROOT_PREFIX_PUBLISHED, WAVE_EXT: "controls_ext", WAVE_STEER: "controls_steer",
                           WAVE_ITDOC: "controls_itdoc", WAVE_QWEN_SEEDREP: "controls_qwen_seedrep", WAVE_ALLPOS: "controls_allpos",
                           "paired_allmodels": "controls_paired_allmodels"}
#: qwen-base-seed-replicates: the OPT-IN Inspect seed policy a schema-2 matrix may declare on every row (contracts.INSPECT_SEED_POLICY_REPLICATE,
#: spelled as a literal: this file is stdlib-only; pinned equal by tests/test_gsm8k_replicate_seed_rule.py) and the driver flags it rides
INSPECT_SEED_POLICY_REPLICATE = "replicate"
FLAG_INSPECT_SEED_POLICY = "--inspect-seed-policy"
FLAG_INSPECT_SEED_BASE = "--inspect-seed-base"


def row_wave(row: dict) -> str:
    """A row's wave: its `wave` field, else `published` (the frozen rows carry none — their bytes never move)."""
    return str(row.get("wave") or WAVE_PUBLISHED)


def out_root_prefix_for(wave: str) -> str:
    """The out-root prefix of a wave: `controls` for the published wave, `controls_<wave>` otherwise."""
    return OUT_ROOT_PREFIX_BY_WAVE.get(wave, f"{OUT_ROOT_PREFIX_PUBLISHED}_{wave}")

#: the clusters a plan can name (registry labels). Each one's QoS and log directory come from its block in the site config
#: (`cluster_site`, read when a plan is printed): the job scripts' own headers name one site's values
CLUSTERS = ("cluster_a", "cluster_b")
_CLUSTERS_LIB = None


def cluster_site(cluster: str) -> dict:
    """`{"qos", "logs"}` of `cluster` (a registry label) from its block in the site config this process uses
    (reproduce/lib/clusters.py: `SISTEER_CONFIG`, else the site directory's config.env), refused by name when the site config
    does not describe it. Loaded on first use, so the job-side subcommands never need a site config."""
    global _CLUSTERS_LIB
    if cluster not in CLUSTERS:
        raise SystemExit(f"cluster must be one of {sorted(CLUSTERS)}, got {cluster!r}")
    if _CLUSTERS_LIB is None:
        spec = importlib.util.spec_from_file_location("g8k_site_clusters", REPO / "reproduce" / "lib" / "clusters.py")
        _CLUSTERS_LIB = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_CLUSTERS_LIB)
    try:
        b = _CLUSTERS_LIB.block_for(cluster)
    except SystemExit as exc:
        raise SystemExit(f"REFUSING: {str(exc.code).removeprefix('REFUSED: ')}") from None
    return {"qos": b["CLUSTER_QOS"], "logs": b["CLUSTER_LOG_DIR"].rstrip("/")}

#: the checkout an emitted sbatch line names, spelled for the cluster shell that runs the line: the job's own
#: SISTEER_EVAL_REPO_ROOT (f22_init derives its checkout from the same variable), refused by name where it is not exported
CHECKOUT_ROOT = "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}"
#: the per-user `high` budget on EACH cluster (GPUs, running + pending) — shared with every peer job under this user
HIGH_BUDGET_GPUS = 16
SBATCH_REL = "scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch"
OVERLAP_SBATCH_REL = "scripts/sdfbase/gsm8k/overlap_off.sbatch"

#: the run-level literals every job shares, as shell assignments: `row_sh` / `overlap_leg_sh` append them and `g8k_init`
#: evals them (`g8k_cells.py literals --sh`) so the sbatch has ONE source. EVERY name here is READ by g8k_common.sh or a
#: launcher (tests/test_sdfbase_gsm8k_sbatch.py pins it: no dead env contract — the overlap leg's own values ride the
#: leg's row variables, and seven unread literals were dropped in the integration fix round).
#: llama-paper-rows (P7): `G8K_THINKING` and `G8K_SAMPLING_MODE` are ROW facts now (THINKING_MODE_BY_FAMILY /
#: SAMPLING_MODE_BY_MODEL, emitted by `row_sh` / `overlap_leg_sh` per row), no longer run-level literals — a Llama row
#: runs `not_available` / `card` beside a gemma row's `on` / `thinking_card` under ONE launcher.
RUN_LITERALS_SH: tuple = (
    ("G8K_CONTROL", CONTROL),
    ("G8K_INSPECT_TASK", INSPECT_TASK),
    ("G8K_SAMPLES", SAMPLES),
    ("G8K_IDENTITY_VERSION", IDENTITY_VERSION),
    ("G8K_UPSTREAM_TIMEOUT_S", UPSTREAM_TIMEOUT_S),
    ("G8K_LENS_SALT_POLICY", LENS_SALT_POLICY),
    ("G8K_LEDGER_WAIT_S", LEDGER_WAIT_S),
    ("G8K_INSPECT_BASE_URL_ENV", INSPECT_BASE_URL_ENV),
    ("G8K_INSPECT_API_KEY_ENV", INSPECT_API_KEY_ENV),
    ("G8K_LEDGER_FILENAME", LEDGER_FILENAME),
    ("G8K_LEDGER_ENDPOINT_FILENAME", LEDGER_ENDPOINT_FILENAME),
    ("G8K_BUDGET_FILENAME", OOS_PROMPT_BUDGET_FILENAME),
    ("G8K_LADDER_STATE_FILENAME", LADDER_STATE_FILENAME),
    ("G8K_RUNG_SUBSET_FILENAME", RUNG_SUBSET_FILENAME),
    ("G8K_RESOLVED_SAMPLER_FILENAME", RESOLVED_SAMPLER_FILENAME),
    ("G8K_RESOLVED_SAMPLER_SCRIPT", RESOLVED_SAMPLER_SCRIPT_REL),
    ("G8K_RECORDER_PROBE_EXPECT", RECORDER_PROBE_EXPECT),
    ("G8K_RECORDER_PROBE_TAGS_FILENAME", RECORDER_PROBE_TAGS_FILENAME),
    ("G8K_RECORDER_LOAD_CHECK_TAGS_FILENAME", RECORDER_LOAD_CHECK_TAGS_FILENAME),
    ("G8K_SERVE_SUBDIR", SERVE_SUBDIR),
    ("G8K_RUNG_DIR_REL", RUNG_DIR_REL),
    ("G8K_SMOKE_DIR_REL", SMOKE_DIR_REL),
    ("G8K_SMOKE_PROBE_SCRIPT", SMOKE_PROBE_SCRIPT_REL),
    ("G8K_SMOKE_TASK_LABEL", SMOKE_TASK_LABEL),
    ("G8K_SMOKE_N", SMOKE_N),
    ("G8K_SMOKE_CAP", SMOKE_CAP),
    ("G8K_SMOKE_CONCURRENCY", SMOKE_CONCURRENCY),
    ("G8K_DATASETS_REQUIRED", " ".join(DATASETS_REQUIRED)),
    ("G8K_DATASETS_CONFIG_DIRS", " ".join(f"{d}/{DATASETS_CONFIG_DIRS[d]}" for d in DATASETS_REQUIRED)),
    ("G8K_N_ITEMS", N_ITEMS),
    ("G8K_PRIMARY_CLUSTER", PRIMARY_CLUSTER),
    ("G8K_ROW_TASK", ROW_TASK),
)


# --------------------------------------------------------------------------- pure helpers over a row
def tag_for(key: str, view: str) -> str:
    """f22_resolve_config's TAG: the bare key for `main`, `<key>_<view>` otherwise."""
    return key if view == VIEW_MAIN else f"{key}_{view}"


def cell_slug(cell: str) -> str:
    """The out-root path segment: `__` -> `--` (cell_controls.store_key_token's convention, mirrored — stdlib only here)."""
    return cell.replace("__", "--")


_SHORT_SI = {"constv2-mix-full": "mix", "constv2ctl2-full": "ctl", "safety_refusal": "si12"}
_SHORT_ARM = {"psr_ce_all": "psr", "lora_r4_ce": "lora", "psr_kl_all": "psr", "lora_r4": "lora",
              "psr_ce_all_minus_ctl_distilled": "dpsr", "lora_r4_ce_minus_ctl": "dlora"}


def cell_short(cell: str) -> str:
    """The job-id token of a cell: `none` / `prompted` (the matched safety baseline) / `prompted-const` (the constitution
    baseline — the six-arm Llama matrix's launched rows carry it; the published matrix holds the cell), else
    `<si-short>-<arm-short>` from the tables above, else the cell with `__` -> `-` (a peer's cell the tables do not name — and
    any other prompted cell — still gets a distinct, legible token; two prompted cells on one view never share a job_id)."""
    if cell == CELL_NONE:
        return "none"
    if cell == CELL_PROMPTED:
        return "prompted"
    if cell == CELL_PROMPTED_CONST:
        return "prompted-const"
    if cell.startswith("prompted__"):
        return cell.replace("__", "-")
    si, _, arm = cell.partition("__")
    if si in _SHORT_SI and arm in _SHORT_ARM:
        return f"{_SHORT_SI[si]}-{_SHORT_ARM[arm]}"
    return cell.replace("__", "-")


def is_prompted_cell(cell: str) -> bool:
    """The roster's prompted-baseline grammar: `prompted__<si>` (the SI text as the system prompt, no object)."""
    return cell.startswith(PROMPTED_PREFIX)


def scheduled_prompted_cells(schema: str = MATRIX_SCHEMA) -> tuple[str, ...]:
    """The prompted cells the matrix of ``schema`` schedules (SCHEDULED_PROMPTED_CELLS_BY_SCHEMA); an unknown schema refuses."""
    try:
        return SCHEDULED_PROMPTED_CELLS_BY_SCHEMA[schema]
    except KeyError:
        raise SystemExit(f"schema {schema!r} is not one of {MATRIX_SCHEMAS} — no scheduled prompted cells") from None


def is_unscheduled_prompted(cell: str, schema: str = MATRIX_SCHEMA) -> bool:
    """A prompted cell the roster may declare that the matrix of ``schema`` (the published one by default) does not
    schedule — admitted as a trailing held row."""
    return is_prompted_cell(cell) and cell not in scheduled_prompted_cells(schema)


def unscheduled_held_reason(cell: str, schema: str = MATRIX_SCHEMA) -> str:
    """The reason `select_rows` stamps on an admitted-and-held prompted row; starts with UNSCHEDULED_HELD_MARK."""
    return (f"{UNSCHEDULED_HELD_MARK} (2026-09-12): the roster declares the prompted cell {cell} on this view but this matrix "
            f"schedules only {', '.join(scheduled_prompted_cells(schema))} — admitted as a held row so the matrix agrees with the roster "
            "while the launched set is unchanged; scheduling the cell releases the hold.")


def route_kind(cell: str) -> str:
    return "base" if cell == CELL_NONE else "prompted" if is_prompted_cell(cell) else "lens"


def budget_class(cell: str) -> str:
    """The prompt-budget artifact CLASS a cell shares (per (model, cell class) — an object cell renders the same
    messages as `none`; a prompted cell adds ITS OWN system turn, so every prompted cell is its own class)."""
    return cell if is_prompted_cell(cell) else CELL_NONE


#: the FRONT-DOOR models (reproduce/models.yaml) whose FITTED GSM8K rows run under their own `_frontdoor` tuple pair (so they stay
#: comparable with this model's thinking-ON u0 rows and the paper's GSM8K mode) — a MIRROR of contracts.G8K_FRONTDOOR_MODELS / G8K_FRONTDOOR_TUPLE_INFIX (this module is stdlib-only, so the
#: constants are mirrored, never imported, and pinned EQUAL by test): the paper's `gsm8k_paired200_<key>_<cap>` name is the paper's models'
FRONTDOOR_TUPLE_MODELS: tuple[str, ...] = ()     # empty: no onboarded model carries fitted GSM8K rows on this tree
FRONTDOOR_TUPLE_INFIX = "_frontdoor"
#: the s3r composed difference cell names its OWN front-door pair `_frontdoor_diff` — a MIRROR of
#: contracts.G8K_FRONTDOOR_DIFF_CELLS / G8K_FRONTDOOR_DIFF_TUPLE_INFIX (this module is stdlib-only; the two are pinned EQUAL by test)
FRONTDOOR_DIFF_CELLS = ("constv2-mix-full__psr_ce_all_minus_ctl",)
FRONTDOOR_DIFF_TUPLE_INFIX = "_frontdoor_diff"
#: the s1 view's cells name their own pairs: the LoRA parents `_frontdoor_lora`, their subtraction `_frontdoor_lora_diff` —
#: MIRRORS of contracts.G8K_FRONTDOOR_LORA_* (stdlib-only module; pinned EQUAL by test)
FRONTDOOR_LORA_CELLS = ("constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce")
FRONTDOOR_LORA_TUPLE_INFIX = "_frontdoor_lora"
FRONTDOOR_LORA_DIFF_CELLS = ("constv2-mix-full__lora_r4_ce_minus_ctl",)
FRONTDOOR_LORA_DIFF_TUPLE_INFIX = "_frontdoor_lora_diff"


def tuple_name(model_key: str, *, declared: bool, rung: int, cell: str | None = None) -> str:
    """The authorised out-of-support tuple's registry name for a FITTED row at a rung:
    `gsm8k_paired200_<key>[_declared]_<cap>` — twelve names, the cap compared; a FRONT-DOOR model's fitted rows name its own
    `gsm8k_paired200_<key>_frontdoor_<cap>` pair (comparable with this model's thinking-ON u0 rows and the paper's GSM8K mode; the paper's name is the paper's models')."""
    if model_key in FRONTDOOR_TUPLE_MODELS:
        if declared:
            raise SystemExit(f"tuple_name: a front-door model's fitted row is never DECLARED ({model_key}); the front-door pair is undeclared")
        if cell in FRONTDOOR_DIFF_CELLS:
            infix = FRONTDOOR_DIFF_TUPLE_INFIX           # The composed difference's own pair
        elif cell in FRONTDOOR_LORA_DIFF_CELLS:
            infix = FRONTDOOR_LORA_DIFF_TUPLE_INFIX      # The LoRA subtraction's own pair
        elif cell in FRONTDOOR_LORA_CELLS:
            infix = FRONTDOOR_LORA_TUPLE_INFIX           # The LoRA parents' own pair
        else:
            infix = FRONTDOOR_TUPLE_INFIX                # the ReLU parents
        return f"gsm8k_paired200_{model_key}{infix}_{rung}"
    return f"gsm8k_paired200_{model_key}{'_declared' if declared else ''}_{rung}"


def steer_tuple_name(model_key: str, *, rung: int) -> str:
    """The tuple an EXTERNAL steering row of a SWITCH model runs under at a rung (contracts.g8k_steer_tuple_name, mirrored
    and pinned equal by test): `gsm8k_paired200_<key>_steer_<cap>` — the all-position shape over the two external cells,
    a NEW name beside the twelve (the published stores record their tuple BY NAME; math-eight-arms-extension P8)."""
    return f"gsm8k_paired200_{model_key}_steer_{rung}"


#: the targets whose ALL-POSITIONS direction the admission record admits (arditi-allpos-cap, 2026-09-17): qwen35_4b alone
ALLPOS_MODELS = ("qwen35_4b",)
#: the all-positions arm's tuple suffix — a MIRROR of contracts.ALLPOS_TUPLE_SUFFIX, which the registry now carries. It is
#: mirrored rather than imported because this module is stdlib-only (the node's python3 reads it before any venv exists, so
#: it cannot import the library), and the two are pinned EQUAL by test — the mirror is checked, never trusted.
ALLPOS_TUPLE_SUFFIX = "_steer_allpos"


def steer_allpos_tuple_name(model_key: str, *, rung: int) -> str:
    """The ALL-POSITIONS arm's OWN tuple at a rung (contracts.g8k_allpos_tuple_name, which the registry now carries; mirrored
    here for the stdlib-only constraint and pinned equal to it by test): `gsm8k_paired200_<key>_steer_allpos_<cap>`. A target
    the admission record does not admit refuses by name."""
    if model_key not in ALLPOS_MODELS:
        raise SystemExit(f"steer_allpos_tuple_name: the all-positions direction runs on {list(ALLPOS_MODELS)} only (the "
                         f"admission record admits no other target's), not {model_key!r}")
    return f"gsm8k_paired200_{model_key}{ALLPOS_TUPLE_SUFFIX}_{rung}"


def external_tuple_name(model_key: str, cell: str, *, rung: int) -> str:
    """The out-of-support tuple an EXTERNAL row runs under, PER CELL — never one name widened to cover a second arm.

    An OOS tuple is an AUTHORISATION, not a label, and every finished store records its tuple BY NAME. The steering
    rung's `_steer_<cap>` tuples declare prefill-only objects dosed as a fraction of the residual norm and picked as the
    highest rung within the over-refusal budget; the all-positions arm is false of that on all three counts (BOTH
    phases, a RAW COEFFICIENT dose, and a declared user override sitting OVER its budget), so widening those tuples'
    `cells` would both misdescribe this arm and move an experiment that six GSM8K tuples' finished legs already record.
    It gets its own (:func:`steer_allpos_tuple_name`), and the two steering arms keep theirs byte-identical. The
    MATH-500 launcher's `cells.tuple_name` makes the same split for the same reason."""
    if cell == CELL_ARDITI_ALLPOS:
        return steer_allpos_tuple_name(model_key, rung=rung)
    if cell in STEER_CELLS:
        return steer_tuple_name(model_key, rung=rung)
    raise SystemExit(f"external_tuple_name: {cell!r} is not one of the rung's cells {STEER_CELLS}")


def time_limit_for(model_key: str, fitted: bool, cell: str = CELL_NONE) -> str:
    """The --time class of a row: fitted vs in support per model; a LONG-SI prompted cell (its ~8k-token prefill x 200) takes
    the FITTED class of its model (llama8b: 0-01:00:00; the full22 families: 0-01:30:00 — the margin for the
    constitution prefill), never the bare cells' in-support class."""
    long_si = prompt_class(cell) == PROMPT_CLASS_LONG_SI
    if model_key == "llama8b":
        return TIME_FITTED_LLAMA if (fitted or long_si) else TIME_IN_SUPPORT_LLAMA
    return TIME_FITTED_GLM if fitted and model_key == "glm46vflash" else TIME_FITTED if (fitted or long_si) else TIME_IN_SUPPORT


def placement_for(model_key: str) -> str:
    f = _fd(model_key)
    if f is not None:
        return f["placement"]                      # the front door: the family's rule (the fused-kernel overlay is the Nemotron family's)
    return PLACEMENT_PRIMARY if model_key in PRIMARY_ONLY_MODELS else PLACEMENT_ANY


def expected_refusal(model_key: str, declared: bool) -> str | None:
    """The by-name refusal a row is EXPECTED to meet at the standup, or None. NO row carries one: a header-located
    object of a header-invariant family (gemma4's two scale-12 `main` cells) is SERVABLE thinking-ON on the declared
    path — an ON header equal to the OFF header is admitted and recorded as the second header
    (`contracts.header_invariant_declaration_refusal` is withdrawn, always None). The row field stays (a boolean +
    reason lane A's reader honours as a tested capability); the live acceptance check of the rule is a gemma standup
    of `safety_refusal__psr_kl_all@thinking`."""
    fam = family_of(model_key)
    if fam is None:
        raise SystemExit(f"expected_refusal: {model_key!r} is not one of {all_model_keys()}")
    return None


# --------------------------------------------------------------------------- the frozen matrix
def resolve_matrix(name_or_path: "str | Path | None") -> Path:
    """`--matrix <name|path>` / the sbatch's trailing positional -> the matrix file: None / `default` -> the frozen
    published matrix; a name in MATRICES (`llama`) -> its file; anything else a path (refused if absent, by name)."""
    if name_or_path is None:
        return MATRIX_PATH
    if str(name_or_path) in MATRICES:
        return MATRICES[str(name_or_path)]
    if str(name_or_path).startswith(FRONT_DOOR_MATRIX_PREFIX):
        return front_door_matrices()[front_door_matrix_name(front_door_key_of(str(name_or_path)))]   # refuses a non-front-door key by name
    p = Path(name_or_path)
    if not p.is_file():
        raise SystemExit(f"--matrix {name_or_path!r} is neither a matrix name ({sorted(MATRICES)}; or {FRONT_DOOR_MATRIX_PREFIX}<models.yaml key>) nor a file")
    return p


def matrix_name_of(path: Path) -> str:
    """The NAME a matrix file is known by (the sbatch positional the plan prints), or its path when it is neither."""
    p = Path(path).resolve()
    for name, known in MATRICES.items():
        if Path(known).resolve() == p:
            return name
    m = re.fullmatch(rf"matrix_{FRONT_DOOR_MATRIX_PREFIX}([a-z0-9_]+)\.json", p.name)
    if m and p.parent == (REPO / "configs" / "sdfbase" / "gsm8k").resolve():
        return front_door_matrix_name(m.group(1))            # a name from the file's spelling; resolve_matrix() holds it to the table
    return str(path)


def load_matrix(path: Path = MATRIX_PATH) -> dict:
    """The frozen matrix document (make_gsm8k_matrix.py's output); refuses a missing or foreign file by name. Schema 1 (the
    published matrix) or schema 2 (llama-paper-rows: rows carrying `thinking_mode`, a document carrying `wave_root_rel`)."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"no frozen matrix at {path} — generate it: python scripts/sdfbase/gsm8k/make_gsm8k_matrix.py")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (doc.get("schema") not in MATRIX_SCHEMAS and doc.get("schema") not in RECORD_SCHEMAS) or not isinstance(doc.get("rows"), list):
        raise SystemExit(f"{path}: not a {' | '.join(MATRIX_SCHEMAS)} matrix or a record ({' | '.join(RECORD_SCHEMAS)}) (schema {doc.get('schema')!r})")
    # a RECORD document (the Arditi seed-replicate record, the all-model appendix record) is read with schema-2 semantics: its rows carry
    # every schema-2 field (thinking_mode, wave, out roots under its own wave root) and the document names its wave root and trigger
    schema2 = doc_is_schema2(doc)
    if doc["schema"] == RECORD_SCHEMA_ARDITI_SEEDREP:
        bad = [i for i, r in enumerate(doc["rows"]) if r.get("role") != ROLE_ARDITI_UNPAIRED or r.get("object_source") != OBJECT_SOURCE_EXTERNAL]
        if bad:
            raise SystemExit(f"{path}: a {RECORD_SCHEMA_ARDITI_SEEDREP} record's rows are EXTERNAL rows of role {ROLE_ARDITI_UNPAIRED!r}; rows {bad} are not")
    elif doc["schema"] in RECORD_SCHEMAS:
        # the all-model appendix record: every row carries the record's role (its composed rows are external, its constitution row roster-served)
        role = RECORD_SCHEMAS[doc["schema"]]
        bad = [i for i, r in enumerate(doc["rows"]) if r.get("role") != role]
        if bad:
            raise SystemExit(f"{path}: a {doc['schema']} record's rows carry role {role!r}; rows {bad} do not")
    if schema2 and not isinstance(doc.get("wave_root_rel"), str):
        raise SystemExit(f"{path}: a {MATRIX_SCHEMA_2} matrix names its wave root (`wave_root_rel`)")
    if not schema2 and "wave_root_rel" in doc:
        raise SystemExit(f"{path}: a {MATRIX_SCHEMA} matrix carries no `wave_root_rel` (schema-2 field)")
    trig = doc.get("trigger_threshold")
    if schema2 and not (isinstance(trig, (int, float)) and not isinstance(trig, bool) and 0.0 < float(trig) < 1.0):
        raise SystemExit(f"{path}: a {MATRIX_SCHEMA_2} matrix declares its ladder trigger (`trigger_threshold`, a share in (0, 1); "
                         f"the instrument's is {TRIGGER_THRESHOLD}) — got {trig!r}")
    if not schema2 and "trigger_threshold" in doc:
        raise SystemExit(f"{path}: a {MATRIX_SCHEMA} matrix carries no `trigger_threshold` (schema-2 field; a schema-1 row reads the adapter's {TRIGGER_THRESHOLD})")
    for i, r in enumerate(doc["rows"]):
        if r.get("index") != i:
            raise SystemExit(f"{path}: row {i} carries index {r.get('index')!r} — the matrix is not in index order")
        if r.get("status") not in STATUSES:
            raise SystemExit(f"{path}: row {i} has status {r.get('status')!r}, not one of {STATUSES}")
        wave = r.get("wave")
        if wave is not None and (not isinstance(wave, str) or not wave.strip()):
            raise SystemExit(f"{path}: row {i} carries wave {wave!r} — a wave is a non-empty name (absent = {WAVE_PUBLISHED!r})")
        prefix = str(r.get("out_root") or "").split("/", 1)[0]
        if prefix != out_root_prefix_for(row_wave(r)):
            raise SystemExit(f"{path}: row {i} (wave {row_wave(r)!r}) has out_root {r.get('out_root')!r} — a wave's rows live under "
                             f"{out_root_prefix_for(row_wave(r))}/ (F11a: one subtree per wave, a pull never carries another wave's leg)")
        src = r.get("object_source")
        if src not in (None,) + OBJECT_SOURCES:
            raise SystemExit(f"{path}: row {i} carries object_source {src!r} — absent (the full22 pin file), {OBJECT_SOURCE_EXTERNAL!r} (the P8 "
                             f"admission record) or {OBJECT_SOURCE_ITDOC!r} (the it-doc admission record)")
        if src == OBJECT_SOURCE_EXTERNAL:
            problem = external_row_problem(r)
            if problem:
                raise SystemExit(f"{path}: row {i}: {problem}")
        if src == OBJECT_SOURCE_ITDOC:
            problem = itdoc_row_problem(r)
            if problem:
                raise SystemExit(f"{path}: row {i}: {problem}")
        if schema2 and r.get("thinking_mode") not in THINKING_MODE_BY_FAMILY.values():
            raise SystemExit(f"{path}: row {i} carries thinking_mode {r.get('thinking_mode')!r} (schema 2 requires one of "
                             f"{sorted(set(THINKING_MODE_BY_FAMILY.values()))})")
        if not schema2 and "thinking_mode" in r:
            raise SystemExit(f"{path}: row {i} carries `thinking_mode`, a schema-2 field, in a {MATRIX_SCHEMA} matrix")
        if schema2 and r.get("trigger_threshold") != trig:
            raise SystemExit(f"{path}: row {i} carries trigger_threshold {r.get('trigger_threshold')!r} != the document's {trig!r} (one trigger per matrix)")
        if not schema2 and "trigger_threshold" in r:
            raise SystemExit(f"{path}: row {i} carries `trigger_threshold`, a schema-2 field, in a {MATRIX_SCHEMA} matrix")
        problem = inspect_seed_row_problem(r, doc)
        if problem:
            raise SystemExit(f"{path}: row {i}: {problem}")
        problem = companion_row_problem(r, doc)
        if problem:
            raise SystemExit(f"{path}: row {i}: {problem}")
        problem = geometry_class_row_problem(r, doc)
        if problem:
            raise SystemExit(f"{path}: row {i}: {problem}")
    if doc.get("inspect_seed_policy") is not None and not schema2:
        raise SystemExit(f"{path}: `inspect_seed_policy` is a {MATRIX_SCHEMA_2} field")
    if doc.get(GEOMETRY_CLASS_KEY) is not None and not schema2:
        raise SystemExit(f"{path}: `{GEOMETRY_CLASS_KEY}` is a {MATRIX_SCHEMA_2} field")
    if doc.get(STANDUP_COMPANIONS_KEY) is not None and not schema2:
        raise SystemExit(f"{path}: `{STANDUP_COMPANIONS_KEY}` is a {MATRIX_SCHEMA_2} field")
    return doc


def inspect_seed_row_problem(row: dict, doc: dict) -> str | None:
    """qwen-base-seed-replicates: why a row's Inspect seed-policy fields disagree with the document — a policy on a schema-1
    matrix, a policy other than `replicate`, a policy without an int base (or a base without the policy), or a row whose
    policy / base is not the document's for its view. None on every row of a matrix that declares no policy."""
    policy, base = row.get("inspect_seed_policy"), row.get("inspect_seed_base")
    doc_policy = doc.get("inspect_seed_policy")
    if policy is None and base is None:
        if doc_policy is not None:
            return f"the document declares inspect_seed_policy {doc_policy!r} but the row carries none"
        return None
    if not doc_is_schema2(doc):
        return f"inspect_seed_policy is a {MATRIX_SCHEMA_2} field"
    if policy != INSPECT_SEED_POLICY_REPLICATE:
        return f"inspect_seed_policy {policy!r} is not {INSPECT_SEED_POLICY_REPLICATE!r} (the one opt-in rule)"
    if isinstance(base, bool) or not isinstance(base, int) or base < 0:
        return f"inspect_seed_policy {policy} needs a non-negative int inspect_seed_base, got {base!r}"
    if doc_policy != policy:
        return f"the row carries inspect_seed_policy {policy!r} but the document declares {doc_policy!r}"
    want = (doc.get("inspect_seed_base_by_view") or {}).get(row.get("view"))
    if want != base:
        return f"inspect_seed_base {base} != the document's base for view {row.get('view')!r} ({want!r})"
    return None


def geometry_class_row_problem(row: dict, doc: dict) -> str | None:
    """qwen-seedrep-native-battery: why a row's `geometry_class` disagrees with the document — a class on a matrix that declares none,
    a declaring matrix's row without one, a class that is not the document's, an unknown class, a class on a schema-1 matrix, or a
    class whose family scope is not the row's model's. None on every row of a matrix that declares no class."""
    decl, got = doc.get(GEOMETRY_CLASS_KEY), row.get(GEOMETRY_CLASS_KEY)
    if decl is None:
        return f"carries {GEOMETRY_CLASS_KEY} {got!r} but the document declares none" if got is not None else None
    if not doc_is_schema2(doc):
        return f"{GEOMETRY_CLASS_KEY} is a {MATRIX_SCHEMA_2} field"
    if decl not in GEOMETRY_CLASSES:
        return f"the document's {GEOMETRY_CLASS_KEY} {decl!r} is not one of {sorted(GEOMETRY_CLASSES)}"
    if got != decl:
        return f"the document declares {GEOMETRY_CLASS_KEY} {decl!r} but the row carries {got!r}"
    if longsi_scope(str(row.get("model_key"))) != GEOMETRY_CLASS_SCOPE[decl]:
        return f"{GEOMETRY_CLASS_KEY} {decl!r} is scoped to the {GEOMETRY_CLASS_SCOPE[decl]!r} families; the row's model {row.get('model_key')!r} is not one"
    return None


def row_geometry_class(row: dict) -> str | None:
    """The row's matrix-declared geometry class (`common_f22` on the seedrep matrix), None everywhere else."""
    return row.get(GEOMETRY_CLASS_KEY)


def row_inspect_seed(row: dict) -> tuple:
    """``(policy, base)`` of a row: ``("replicate", <base>)`` on a seed-replicate matrix, ``(None, None)`` everywhere else."""
    return (row.get("inspect_seed_policy"), row.get("inspect_seed_base")) if row.get("inspect_seed_policy") else (None, None)


def row_thinking(row: dict) -> str:
    """The row's arm: its `thinking_mode` (schema 2) or `on` (a schema-1 row of the published matrix)."""
    return str(row.get("thinking_mode") or THINKING_ON)


def external_row_problem(row: dict) -> str | None:
    """Why an EXTERNAL-object row (a steering row) is not one the launcher can bind: it must be a fitted, undeclared,
    `all`-position row of a rung cell on the panel view `main`, pinned by a 64-hex object sha256, naming its lane's bank
    view per cluster role and the cell dir inside it, the lane's own view name and the selection label — the facts
    `g8k_bank_view` / the standup / the driver hold the served bytes to."""
    cell = row.get("cell")
    if row.get("role") == ROLE_PAIRED_ALLMODELS:
        # the all-model appendix record: the paired-write difference (the composed arm) served on the model's PANEL view — the
        # view the row declares (`panel_view`: gemma / llama `it`, nemotron `main`) — under the default Inspect seed rule, no companion
        if cell != CELL_PAIRED_DIFFERENCE:
            return f"a {ROLE_PAIRED_ALLMODELS!r} external row is the {CELL_PAIRED_DIFFERENCE!r} arm, got cell {cell!r}"
        if row.get("view") != row.get("panel_view") or row.get("view") not in (VIEW_MAIN, VIEW_IT):
            return f"a {ROLE_PAIRED_ALLMODELS!r} row sits on its declared panel view (`it` or `main`), got view {row.get('view')!r} / panel_view {row.get('panel_view')!r}"
        if row.get("inspect_seed_policy") is not None or row.get("standup_companion_cells"):
            return f"a {ROLE_PAIRED_ALLMODELS!r} row runs the default Inspect seed rule with no companion"
    elif cell not in STEER_CELLS:
        return f"object_source external on cell {cell!r}, not one of the rung's cells {STEER_CELLS}"
    want_view = EXTERNAL_ROW_VIEW_BY_CELL.get(cell)
    if row.get("role") == ROLE_PAIRED_ALLMODELS:
        pass                                                   # the view rule above is this record's
    elif row.get("role") == ROLE_ARDITI_UNPAIRED:
        # the qwen_seedrep wave's Arditi RECORD (g8k_seedrep_arditi.py): the `arditi_allpos` row re-keyed onto a SEED view `s<s>` under the
        # wave's REPLICATE seed policy at base s — the one admitted departure from the arm's declared view, and only with the seed facts
        m = re.fullmatch(r"s([1-9][0-9]*)", str(row.get("view") or ""))
        if cell != CELL_ARDITI_ALLPOS or not m:
            return f"a {ROLE_ARDITI_UNPAIRED!r} row is the {CELL_ARDITI_ALLPOS!r} arm on a seed view `s<s>`, got cell {cell!r} on view {row.get('view')!r}"
        if row.get("inspect_seed_policy") != INSPECT_SEED_POLICY_REPLICATE or row.get("inspect_seed_base") != int(m.group(1)):
            return (f"a {ROLE_ARDITI_UNPAIRED!r} row on view {row.get('view')!r} carries the wave's replicate seed policy at base {m.group(1)} "
                    f"(got {row.get('inspect_seed_policy')!r} / {row.get('inspect_seed_base')!r})")
        if row.get("standup_companion_cells"):
            return f"a {ROLE_ARDITI_UNPAIRED!r} row registers NO companion (a single-object engine), got {row.get('standup_companion_cells')!r}"
    elif row.get("view") != want_view:
        return (f"an external row of {cell!r} sits on the {want_view!r} view (its lane's own view rides `bank_view`; "
                f"EXTERNAL_ROW_VIEW_BY_CELL declares the view PER ARM), got view {row.get('view')!r}")
    if row.get("tag") != tag_for(str(row.get("model_key")), str(row.get("view"))):
        return (f"tag {row.get('tag')!r} is not {tag_for(str(row.get('model_key')), str(row.get('view')))!r} — a row's TAG is "
                "f22_resolve_config's spelling of (model key, view), and every overlay and bank path is cut from it")
    if not row.get("fitted") or row.get("declared") or row.get("served_positions_expected") != "all":
        return "an external row is a fitted, undeclared, all-position row"
    sha = row.get("object_sha256")
    if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
        return f"object_sha256 {sha!r} is not the 64-hex pin the admission record carries (D5)"
    rel = row.get("bank_view_rel")
    if not isinstance(rel, dict) or set(rel) != {"primary", "overflow"} or not rel.get("primary"):
        return f"bank_view_rel must name the primary cluster's bank view (and the overflow cluster's, or null), got {rel!r}"
    for f in ("bank_cell_dir", "bank_view", "selection_label", "lane_root"):
        if not row.get(f):
            return f"an external row carries no {f}"
    # an AlphaSteer row names how its rung was selected — `chosen` under (a)+(b)+(c) or the `a4_fallback`
    # (nemotron) — beside the selection label; an Arditi row carries null. Refused here so a row without it never reaches a GPU.
    if "selection_outcome" not in row:
        return "an external row carries no selection_outcome (the AlphaSteer ladder's `chosen` / `a4_fallback`, null on an Arditi row; D90 addendum)"
    problem = selection_outcome_problem(str(row.get("cell")), row.get("selection_outcome"))
    if problem:
        return problem
    if cell == CELL_ARDITI_ALLPOS:
        return allpos_row_problem(row)
    return None


#: the keys an ALL-POSITIONS row's `selection` block MUST carry. They ride the ROW because every
#: table, figure and provenance line reads them from there (the same reason `selection_label` / `dose_evidence` already do), and
#: because a reader must be able to see, without opening the admission record, that (i) the pinned dose is the override and not
#: the rule's, (ii) BOTH readings are present — what the rule returned and what the pinned dose measures — so the override's size is
#: legible, (iii) the reading is identified by its LADDER CAP (at another cap the rule returns a different point, so a selection
#: block without its cap is not a reading of anything), and (iv) the ladder was SINGLE-TRIAL, so no margin here is a significance
#: claim. `alpha_equiv`, where present, must be labelled informational: it is arithmetic on this model, never the dose.
ALLPOS_SELECTION_KEYS = ("outcome", "rule", "rule_returns", "rule_reading_pp", "override_reading_pp", "budget_pp",
                         "ladder_cap", "single_trial", "dose")


def allpos_row_problem(row: dict) -> str | None:
    """Why an ALL-POSITIONS Arditi row is not one the launcher and the paper export can read: the arm's dose is a RAW
    COEFFICIENT (never an alpha), its operating point a declared user override over the rule's budget, its ladder single-trial —
    so the row carries the record's `selection` block (both readings, the budget, the ladder cap, the single-trial flag, the dose
    with its unit, `alpha_equiv` labelled informational where present) and its `fit_provenance` block, or it is not a row."""
    sel = row.get("selection")
    if not isinstance(sel, dict) or not sel:
        return (f"an {CELL_ARDITI_ALLPOS} row carries the admission record's `selection` block (the analysis and the paper export "
                f"read the dose, both readings and the override from the ROW), got {sel!r}")
    missing = [k for k in ALLPOS_SELECTION_KEYS if sel.get(k) in (None, "")]
    if missing:
        return f"an {CELL_ARDITI_ALLPOS} row's `selection` block lacks {missing} (it carries {sorted(ALLPOS_SELECTION_KEYS)})"
    if sel.get("outcome") != row.get("selection_outcome"):
        return (f"`selection.outcome` {sel.get('outcome')!r} != the row's selection_outcome {row.get('selection_outcome')!r} — "
                "one outcome word, in both places or in neither")
    if sel.get("single_trial") is not True:
        return (f"`selection.single_trial` is {sel.get('single_trial')!r} — this ladder was run ONCE; the row states it so no "
                "margin on it is read as a significance claim")
    dose = sel.get("dose")
    if not isinstance(dose, dict) or dose.get("unit") != DOSE_UNIT_RAW_COEFFICIENT or not isinstance(dose.get("value"), (int, float)):
        return (f"`selection.dose` must be a raw coefficient ({{value, unit: {DOSE_UNIT_RAW_COEFFICIENT!r}}}) — the all-positions arm's "
                f"dose is NEVER an alpha ({DOSE_UNIT_RESIDUAL_NORM_FRACTION!r} is another arm's unit on another intervention); got {dose!r}")
    alpha_equiv = sel.get("alpha_equiv")
    if alpha_equiv is not None and (not isinstance(alpha_equiv, dict) or alpha_equiv.get("informational") is not True):
        return ("`selection.alpha_equiv`, where present, is labelled `informational: true` — it is arithmetic convenience on this "
                f"model, never the dose and never an axis shared with the {DOSE_UNIT_RESIDUAL_NORM_FRACTION} arms; got {alpha_equiv!r}")
    if row.get("alpha") is not None:
        return (f"an {CELL_ARDITI_ALLPOS} row carries no `alpha` dose (got {row.get('alpha')!r}) — its dose is the raw coefficient "
                "in `selection.dose`; an alpha here would put two units on one axis")
    if not isinstance(row.get("fit_provenance"), dict) or not row.get("fit_provenance"):
        return (f"an {CELL_ARDITI_ALLPOS} row carries the admission record's `fit_provenance` block (what the direction was fitted "
                f"on, at which layer, over which items) — got {row.get('fit_provenance')!r}")
    return None


def itdoc_row_problem(row: dict) -> str | None:
    """Why an IT-DOC row (E3) is not one the launcher can bind: a fitted, undeclared, `all`-position LoRA row of one of the three
    it-doc cells on the `it` view, keyed to the record's lane, its RULED staged id (the lane token in the id), the object kind and its
    sidecar of the kind; a `launch` row carries the 64-hex pin, a complete bank binding and the pair's qualification VERBATIM; a
    `held` row carries its reason; a derived row names its two expected parents by sha. The pin is the it-doc record's, never the
    full22 pin file's (decisions-e3 E3-3)."""
    if row.get("cell") not in ITDOC_CELLS:
        return f"object_source itdoc on cell {row.get('cell')!r}, not one of the re-fit's cells {ITDOC_CELLS}"
    if row.get("view") != VIEW_IT:
        return f"an it-doc row sits on the instruct-direct view {VIEW_IT!r}, got view {row.get('view')!r}"
    if not row.get("fitted") or row.get("declared") or row.get("served_positions_expected") != "all":
        return "an it-doc row is a fitted, undeclared, all-position row"
    if not row.get("lane"):
        return "an it-doc row names its lane (the admission record's)"
    kind = row.get("object_kind")
    if kind not in B.OBJECT_KINDS:
        return f"object_kind {kind!r} is not one of {B.OBJECT_KINDS}"
    if row.get("bank_sidecar") != B.SIDECAR_BY_KIND[kind]:
        return f"a {kind} object is proven through {B.SIDECAR_BY_KIND[kind]!r}, the row says {row.get('bank_sidecar')!r}"
    sid = row.get("staged_id")
    if sid != B.STAGED_PREFIX_BY_KIND[kind] + str(row["cell"]):
        return f"staged_id {sid!r} is not this lane's ruled spelling {B.STAGED_PREFIX_BY_KIND[kind] + str(row['cell'])!r}"
    sha = row.get("object_sha256")
    if sha is not None and not B.is_sha256(sha):
        return f"object_sha256 {sha!r} is neither null (not yet verified) nor a 64-hex pin — never a placeholder (D5)"
    if kind == B.OBJECT_KIND_DERIVED and row.get("parents_expected") is not None and not (
            isinstance(row["parents_expected"], dict) and len(row["parents_expected"]) == 2 and all(B.is_sha256(v) for v in row["parents_expected"].values())):
        return f"a derived row names EXACTLY its two parents (cell -> 64-hex sha), got {row.get('parents_expected')!r}"
    if row.get("status") == STATUS_LAUNCH:
        if sha is None:
            return "a launch row carries its pin"
        if not B.bank_complete(row.get("bank") if "bank" in row else {"view": row.get("bank_view"), "bank_view_rel": row.get("bank_view_rel"),
                                                                       "bank_cell_dir": row.get("bank_cell_dir"), "cluster_constraint": row.get("cluster_constraint")}):
            return "a launch row carries a complete bank binding (bank_view, bank_view_rel.primary, bank_cell_dir)"
        if not str(row.get("qualification") or "").strip() or not str(row.get("qualification_pattern") or "").strip():
            return "a launch row carries the pair's qualification AND the cross-model qualification pattern VERBATIM"
        if row.get("held_reason"):
            return "a launch row carries no held_reason"
    elif row.get("status") == STATUS_HELD and not str(row.get("held_reason") or "").strip():
        return "a held row carries its reason"
    return None


#: the ladder TRIGGER (the at-cap share above which the eligible items are redrawn at the 16,384 rung): 1 % RAW at-cap, the
#: battery's standing rule, harmonised on 2026-09-12 — the published wave's first analysis ran the 3 % exception of
#: 2026-09-10 (retired; its rungs planned under that pair are admitted on their evidence, inspect_rung.GSM8K_TRIGGER_RULES_ADMITTED).
#: A schema-2 matrix DECLARES its trigger (`trigger_threshold` on the document and on every row); a schema-1 row reads the
#: adapter's; the ladder planner / the analysis read the row's. Mirrors inspect_rung's two constants (pinned equal by test).
TRIGGER_THRESHOLD = 0.01
TRIGGER_THRESHOLD_3PCT_RETIRED = 0.03


def row_trigger_threshold(row: dict) -> float:
    """The row's ladder trigger: its `trigger_threshold` (schema 2) or the adapter's 1 % (a schema-1 row)."""
    return float(row.get("trigger_threshold", TRIGGER_THRESHOLD))


def row_sampling_mode(row: dict) -> str:
    return str(row.get("sampling_mode") or SAMPLING_THINKING_CARD)


def wave_root_rel_of(doc: dict) -> str:
    """The matrix document's wave root (schema 2) or the published root (schema 1)."""
    return str(doc.get("wave_root_rel") or WAVE_ROOT_REL)


_DOCS: dict[Path, dict] = {}


def matrix_doc(path: Path = MATRIX_PATH) -> dict:
    """The loaded document, cached per path (the published matrix is read once per process, as before)."""
    p = Path(path)
    doc = _DOCS.get(p)
    if doc is None:
        doc = _DOCS[p] = load_matrix(p)
    return doc


def matrix_rows(path: Path = MATRIX_PATH) -> list[dict]:
    """Every row of the matrix at `path` (default: the frozen published matrix), held rows INCLUDED (the production
    consumers filter on `status`)."""
    return matrix_doc(path)["rows"]


def launch_rows(rows: list[dict] | None = None) -> list[dict]:
    rows = matrix_rows() if rows is None else rows
    return [r for r in rows if r["status"] == STATUS_LAUNCH]


def unscheduled_held_rows(rows: list[dict] | None = None) -> list[dict]:
    """The held rows that are unscheduled-prompted admissions (a roster-declared prompted cell this matrix does not
    schedule) — told apart from the views.yaml holds by the reason's prefix; they trail every scheduled row."""
    rows = matrix_rows() if rows is None else rows
    return [r for r in rows if r["status"] == STATUS_HELD and str(r.get("held_reason") or "").startswith(UNSCHEDULED_HELD_MARK)]


def held_rows(rows: list[dict] | None = None) -> list[dict]:
    rows = matrix_rows() if rows is None else rows
    return [r for r in rows if r["status"] == STATUS_HELD]


def rows_of_wave(wave: str, rows: list[dict] | None = None) -> list[dict]:
    rows = matrix_rows() if rows is None else rows
    return [r for r in rows if row_wave(r) == wave]


def eligible_rows(wave: str, universe: list[dict] | None = None) -> list[dict]:
    """The rows a submission of `wave` may carry: that wave's `status: launch` rows (plan P2 / F11e). An unknown wave, a
    COMPLETED wave (the published rows — reused byte-for-byte, never re-run) or a wave with nothing launchable refuses by name."""
    universe = matrix_rows() if universe is None else universe
    known = tuple(dict.fromkeys(row_wave(r) for r in universe))
    if wave not in known:
        raise SystemExit(f"--wave {wave!r}: not a wave of this matrix (waves: {list(known)})")
    if wave in COMPLETED_WAVES:
        raise SystemExit(f"--wave {wave!r}: this wave is COMPLETE (every leg reused byte-for-byte from its published tree) and is "
                         f"never resubmitted; the launchable waves are {[w for w in known if w not in COMPLETED_WAVES]}")
    out = [r for r in universe if row_wave(r) == wave and r["status"] == STATUS_LAUNCH]
    if not out:
        statuses = sorted({r["status"] for r in universe if row_wave(r) == wave})
        raise SystemExit(f"--wave {wave!r}: no launchable row (its rows are {statuses}) — a held or deferred row is never submitted")
    return out


def _refuse_outside_wave(indices: list[int], wave: str, eligible: list[dict], universe: list[dict], flag: str) -> None:
    ok = {r["index"] for r in eligible}
    bad = [i for i in indices if i not in ok]
    if bad:
        why = ", ".join(f"{i} ({universe[i]['job_id']}: wave {row_wave(universe[i])}, status {universe[i]['status']})" for i in bad)
        raise SystemExit(f"{flag}: row(s) outside wave {wave!r}'s eligible set: {why} — a published (complete), held or deferred "
                         f"row is never submitted; eligible: {_array_spec([r['index'] for r in eligible])}")


def assert_eligible(selected: list[dict], wave: str, universe: list[dict] | None = None) -> dict:
    """Every selected row is in `wave`'s eligible set (status launch, that wave); refuses BY NAME otherwise. Returns the
    record the planner prints (`# assert-eligible: …`): the rows, and how many of the matrix were excluded and why."""
    universe = matrix_rows() if universe is None else universe
    eligible = {r["index"] for r in eligible_rows(wave, universe)}
    bad = [r for r in selected if r["index"] not in eligible]
    if bad:
        raise SystemExit("assert-eligible REFUSED: " + "; ".join(
            f"row {r['index']} ({r['job_id']}) wave={row_wave(r)} status={r['status']}" for r in bad))
    chosen = {s["index"] for s in selected}
    excluded = [r for r in universe if r["index"] not in chosen]

    def _why(r: dict) -> str:       # the reason a row is NOT in this submission, as the record names it
        return "complete" if row_wave(r) in COMPLETED_WAVES else r["status"] if r["status"] != STATUS_LAUNCH else "other-wave"
    return {"wave": wave, "rows": [r["index"] for r in selected],
            "excluded": {f"{w}/{s}": sum(row_wave(r) == w and _why(r) == s for r in excluded)
                         for w, s in dict.fromkeys((row_wave(r), _why(r)) for r in excluded)}}


def assert_launchable(row: dict, universe: list[dict] | None = None) -> dict:
    """The BATCH BOUNDARY's eligibility predicate (ported from the MATH-500 launcher): the ONE row a job resolved —
    `g8k_row ARRAY` from SLURM_ARRAY_TASK_ID, or the positional `<key> <view> <cell>` — is in ITS OWN wave's eligible set,
    by the same test `plan --wave` runs over a submission: a row of a COMPLETED wave (the published rows, reused
    byte-for-byte), a held or a deferred row refuses BY NAME (SystemExit). Before it the job trusted the submission: an
    `--array=4` resolved the published gemma `none` row, kept `status: launch` (published-ness is the WAVE, never a
    status) and would have run into the published output root. Returns the one-row record."""
    return assert_eligible([dict(row)], row_wave(row), universe)


def launchable_line(row: dict, record: dict) -> str:
    """The one line the launcher logs when :func:`assert_launchable` admits a row."""
    excluded = ", ".join(f"{k}: {v}" for k, v in record["excluded"].items())
    return f"launchable: row {row['index']} ({row['job_id']}) wave={row_wave(row)} status={row['status']}; excluded {excluded}"


def row_for(model_key: str, view: str, cell: str, rows: list[dict] | None = None) -> dict:
    rows = matrix_rows() if rows is None else rows
    for r in rows:
        if (r["model_key"], r["view"], r["cell"]) == (model_key, view, cell):
            return r
    raise SystemExit(f"not a matrix row: ({model_key!r}, {view!r}, {cell!r}) — the frozen matrix's (key, view, cell) "
                     f"triples are {[(r['model_key'], r['view'], r['cell']) for r in rows]}")


def counts(rows: list[dict] | None = None) -> dict:
    """The acceptance counts, DERIVED from the matrix (the tests compare the dry run's greps to these; the analysis' design
    set is `launched`): rows 42 / held 4 / launched 38 / fitted_launched 28 / in_support 10 / declared 4 / expected_refusals
    0 / servable 38 / contrasts_designed 32 / contrasts_computable 32 / tags 6 for THIS experiment's views.yaml over today's
    roster (the two gemma header-located rows are servable; before that 2 / 36 / 30). A roster that declares an
    unscheduled prompted cell on an expressed view adds ONE trailing held row per (view, cell) — rows and held grow
    together, every launched count stays."""
    rows = matrix_rows() if rows is None else rows
    launched = [r for r in rows if r["status"] == STATUS_LAUNCH]
    held = [r for r in rows if r["status"] == STATUS_HELD]
    fitted_l = [r for r in launched if r["fitted"]]
    refused = [r for r in launched if r["expected_refusal"]]
    contrasts = [r for r in launched if r["cell"] != CELL_NONE]
    tags = list(dict.fromkeys(r["tag"] for r in rows))
    waves = list(dict.fromkeys(row_wave(r) for r in rows))
    return {
        "rows": len(rows), "held": len(held), "launched": len(launched),
        # E2 (plan P2 / P5): the rows per WAVE and per status, and the PUBLISHED wave's own acceptance counts apart (the
        # frozen document's numbers keep their meaning beside the appended waves); a matrix with one wave reports itself
        "waves": {w: sum(row_wave(r) == w for r in rows) for w in waves},
        "statuses": {s: sum(r["status"] == s for r in rows) for s in dict.fromkeys(r["status"] for r in rows)},
        "launched_by_wave": {w: sum(row_wave(r) == w and r["status"] == STATUS_LAUNCH for r in rows) for w in waves},
        "deferred": sum(r["status"] == STATUS_DEFERRED for r in rows),
        # the steering rung's EXTERNAL objects, by (model, cell) — the admission record's pins, never the pin file's
        "external_pins": {f"{r['model_key']}/{r['cell']}": r["object_sha256"] for r in rows if r.get("object_source") == OBJECT_SOURCE_EXTERNAL},
        # E3: the it-doc rows' pins by the FULL tuple (a cell id is not unique across lanes); null = held, not yet verified on disk
        **({"itdoc_pins": {f"{r['lane']}/{r['model_key']}/{r['view']}/{r['cell']}": r["object_sha256"] for r in rows if r.get("object_source") == OBJECT_SOURCE_ITDOC}}
           if any(r.get("object_source") == OBJECT_SOURCE_ITDOC for r in rows) else {}),      # only a matrix WITH it-doc rows (the published documents' bytes never move)
        **({"object_generations": {g: sum(r.get("object_generation") == g for r in rows) for g in OBJECT_GENERATIONS}}
           if any(r.get("object_generation") for r in rows) else {}),
        "fitted": sum(r["fitted"] for r in rows), "fitted_launched": len(fitted_l),
        "in_support": sum(not r["fitted"] for r in launched),
        "declared": sum(r["declared"] for r in launched),
        "expected_refusals": len(refused), "servable": len(launched) - len(refused),
        "contrasts_designed": len(contrasts), "contrasts_computable": len([r for r in contrasts if not r["expected_refusal"]]),
        # a `not_available` fitted row carries NO tuple (in support; llama-paper-rows P7) — the sets skip None
        "tuple_names_launch": sorted({r["tuple_name"] for r in fitted_l if r["tuple_name"]}),
        "tags": tags, "n_tags": len(tags),
        "by_tag": {t: {"rows": sum(r["tag"] == t for r in rows), "held": sum(r["tag"] == t for r in held),
                       "launched": sum(r["tag"] == t for r in launched)} for t in tags},
        "held_rows": [{"index": r["index"], "job_id": r["job_id"], "tag": r["tag"], "cell": r["cell"]} for r in held],
        "expected_refusal_rows": [{"index": r["index"], "job_id": r["job_id"]} for r in refused],
        "placement": {PLACEMENT_PRIMARY: sum(r["cluster_constraint"] == PLACEMENT_PRIMARY for r in launched),
                      PLACEMENT_ANY: sum(r["cluster_constraint"] == PLACEMENT_ANY for r in launched)},
        "reasoning_parser_gemma4": sum(r["reasoning_parser"] == "gemma4" for r in launched),
        "tuple_names": sorted({r["tuple_names"][str(CAP)] for r in fitted_l if r["tuple_names"]}),
        "distinct_out_roots": len({r["out_root"] for r in rows}),
        # per the MATRIX's own models (in MODEL_KEYS order), never the module's whole table: a fifth registry key
        # (llama8b) must not add an entry to the published document's counts (byte-identity, P16). The full22 pins only:
        # an external row's pin is the admission record's and is counted under `external_pins`
        "pins": {k: sum(1 for r in rows if r["object_sha256"] and r["model_key"] == k and r.get("object_source") not in OBJECT_SOURCES)
                 for k in (*MODEL_KEYS, *(k for k in dict.fromkeys(r["model_key"] for r in rows) if k not in MODEL_KEYS)) if any(r["model_key"] == k for r in rows)},
        "families": sorted({family_of(r["model_key"]) for r in rows}),
        "models": list(dict.fromkeys(r["model_key"] for r in rows)),
    }


def published_counts(rows: list[dict] | None = None) -> dict:
    """The PUBLISHED wave's acceptance counts (the frozen document's numbers: 42 / 4 / 38 / … for this experiment's
    views.yaml) — over the rows without a `wave` key, whatever the appended waves carry."""
    rows = matrix_rows() if rows is None else rows
    return counts(rows_of_wave(WAVE_PUBLISHED, rows))


def literals_sh(matrix: Path = MATRIX_PATH) -> str:
    """RUN_LITERALS_SH as shell assignments — the ONE source g8k_init evals (the datasets guard reads G8K_DATASETS_REQUIRED /
    G8K_DATASETS_CONFIG_DIRS before the row is known); row_sh / overlap_leg_sh append the same lines. llama-paper-rows
    (P11): the MATRIX's wave root rides here too (`G8K_WAVE_ROOT_REL`, the document's `wave_root_rel` — the published root
    on the schema-1 matrix), evaluated by g8k_init BEFORE F22_WAVE_ROOT so one launcher serves both roots."""
    return ("\n".join(f"{k}={shlex.quote(str(v))}" for k, v in RUN_LITERALS_SH)
            + f"\nG8K_WAVE_ROOT_REL={shlex.quote(wave_root_rel_of(matrix_doc(matrix)))}\n")


def row_sh(row: dict, rung: int = CAP, *, wave_root_rel: str = WAVE_ROOT_REL) -> str:
    """The row as shell assignments the sbatch evals, for ONE rung (positional facts only; the sampling row rides the
    driver's --sampling MODE — the numbers live in contracts' tables and this file's mirror, never in argv). A held row
    is emitted with STATUS=held and its reason; g8k_row refuses to run it. llama-paper-rows: the row's ARM
    (`G8K_THINKING`, its `thinking_mode` — `on` on a schema-1 row), its sampling MODE (`G8K_SAMPLING_MODE`) and the
    matrix's wave root (`G8K_WAVE_ROOT_REL`, the caller's document) ride here; a `not_available` fitted row carries no
    tuple (`G8K_TUPLE` empty — in support; the pin rides `PIN` as identity). The geometry entries are the module's for
    every matrix (the Llama prompt fits the shared windows; S is the model's budget at the standup)."""
    if rung not in RUNGS:
        raise SystemExit(f"rung must be one of {RUNGS}, got {rung!r}")
    out_rel = row["out_root"] if rung == CAP else row["rung_out_root"]
    geometry_eval, geometry_l = geometry_for_cell(row["cell"], rung, model_key=row["model_key"], geometry_class=row_geometry_class(row))
    tuple_for_rung = (row["tuple_names"] or {}).get(str(rung)) if row["fitted"] else None
    lines = [
        f"G8K_INDEX={row['index']}", f"G8K_JOB_ID={shlex.quote(row['job_id'])}", f"KEY={shlex.quote(row['model_key'])}",
        f"VIEW={shlex.quote(row['view'])}", f"TAG={shlex.quote(row['tag'])}",
        f"CELL={shlex.quote(row['cell'])}", f"CELLSLUG={shlex.quote(row['cell_slug'])}",
        f"FITTED={1 if row['fitted'] else 0}", f"DECLARED={1 if row['declared'] else 0}",
        f"RUNG={rung}", f"G8K_CAP={CAP}", f"G8K_RUNG_CAP={RUNG_CAP}",
        f"G8K_MAX_TOKENS_BY_TASK={shlex.quote(f'{INSPECT_TASK}={rung}')}",
        f"G8K_GEOMETRY_EVAL={shlex.quote(geometry_eval)}", f"G8K_GEOMETRY_L={geometry_l}",
        # qwen-seedrep-native-battery: the matrix-declared geometry class the two entries above were selected by (empty on every other
        # matrix's rows: the prompt-class rule); informational for the job log and the dry run — the entry and L above are what the standup reads
        f"G8K_GEOMETRY_CLASS={shlex.quote(row_geometry_class(row) or '')}",
        f"G8K_TUPLE={shlex.quote(tuple_for_rung or '')}",
        f"G8K_THINKING={shlex.quote(row_thinking(row))}", f"G8K_SAMPLING_MODE={shlex.quote(row_sampling_mode(row))}",
        f"G8K_WAVE_ROOT_REL={shlex.quote(wave_root_rel)}",
        f"G8K_ROUTE_KIND={row['route_kind']}", f"G8K_FAMILY={row['family']}", f"G8K_TIME={row['time_limit']}",
        f"G8K_OUT_REL={shlex.quote(out_rel)}", f"G8K_SOURCE_OUT_REL={shlex.quote(row['out_root'])}",
        f"G8K_REASONING_PARSER={shlex.quote(row['reasoning_parser'] or '')}",
        f"G8K_GPU_MEM_UTIL={row['gpu_mem_util']}", f"PIN={shlex.quote(row['object_sha256'] or '')}",
        f"G8K_STANDUP_CELL={shlex.quote(row['standup_cell'])}", f"G8K_STANDUP_HARNESS_CELL={shlex.quote(row['standup_harness_cell'])}",
        f"G8K_SERVED_POSITIONS_EXPECTED={shlex.quote(row['served_positions_expected'] or '')}",
        f"G8K_EXPECTED_STAMPS={shlex.quote(','.join(row['expected_stamps']))}",
        f"G8K_SAMPLING_SOURCE={shlex.quote(row['sampling']['source'])}",
        "G8K_SAMPLING_PARAMS=" + shlex.quote(json.dumps({k: row['sampling'][k] for k in SAMPLING_PARAM_KEYS if k in row['sampling']}, sort_keys=True)),
        f"G8K_BUDGET_CLASS={shlex.quote(row['budget_class'])}",
        f"G8K_CLUSTER_CONSTRAINT={row['cluster_constraint']}",
        f"STATUS={row['status']}", f"G8K_HELD_REASON={shlex.quote(row['held_reason'] or '')}",
        f"G8K_EXPECTED_REFUSAL={shlex.quote(row['expected_refusal_reason'] or '')}",
        # qwen-base-seed-replicates: the OPT-IN Inspect seed policy + base (empty on every other matrix's rows; g8k_driver_argv
        # appends --inspect-seed-policy / --inspect-seed-base exactly when the policy is set)
        f"G8K_INSPECT_SEED_POLICY={shlex.quote(row_inspect_seed(row)[0] or '')}",
        f"G8K_INSPECT_SEED_BASE={shlex.quote('' if row_inspect_seed(row)[1] is None else str(row_inspect_seed(row)[1]))}",
        # qwen-base-seed-replicates (O12a): the STANDUP COMPANIONS — the standup spelling of each (a plain LoRA cell is never declared,
        # so its standup spelling is the cell id) and `<cell>=<pin>` pairs, space-separated (cell ids and shas carry no whitespace);
        # empty on every other matrix's rows and on a row with none, so g8k_standup_extra appends nothing there
        "G8K_STANDUP_COMPANION_CELLS=" + shlex.quote(" ".join(c for c, _h, _sid, _pin in row_companions(row))),
        "G8K_STANDUP_COMPANION_PINS=" + shlex.quote(" ".join(f"{c}={pin}" for c, _h, _sid, pin in row_companions(row))),
    ]
    lines += external_sh(row)
    lines += [f"{k}={shlex.quote(str(v))}" for k, v in RUN_LITERALS_SH]
    return "\n".join(lines) + "\n"


def external_sh(row: dict) -> list[str]:
    """The E2 row facts as shell assignments (math-eight-arms-extension P2 / P8): the WAVE (the batch boundary's predicate
    and the log read it), the object source and — on an EXTERNAL row — the lane's bank view RELATIVE to the shared root per
    cluster role, the cell dir inside it, the lane's own view name and the selection label the record carries verbatim; on an
    IT-DOC row (E3) the lane, the object kind, the ruled staged id, the admission record and the qualification.
    Empty on every other row (and on the overlap legs), so the shell's `[ -n ... ]` branch is the ONE discriminator."""
    rel = row.get("bank_view_rel") or {}
    return [
        f"G8K_WAVE={shlex.quote(row_wave(row))}",
        f"G8K_OBJECT_SOURCE={shlex.quote(str(row.get('object_source') or ''))}",
        # (no serving-path literal here since 2026-09-15: the pin FOLLOWS THE STORE at the row's out root on the cluster, so the sbatch
        # reads it through `row --serving-path --wave-root` BEFORE f22_init — g8k_select_serving_path — never off the row alone)
        f"G8K_BANK_VIEW={shlex.quote(str(row.get('bank_view') or ''))}",
        f"G8K_BANK_VIEW_REL_PRIMARY={shlex.quote(str(rel.get('primary') or ''))}",
        f"G8K_BANK_VIEW_REL_OVERFLOW={shlex.quote(str(rel.get('overflow') or ''))}",
        f"G8K_BANK_CELL_DIR={shlex.quote(str(row.get('bank_cell_dir') or ''))}",
        f"G8K_SELECTION_LABEL={shlex.quote(str(row.get('selection_label') or ''))}",
        # the AlphaSteer ladder's selection outcome (`chosen` / `a4_fallback`; empty on an Arditi row and on
        # every non-external row) and the grid-limited flag, logged by the job beside the label and printed by the dry run
        f"G8K_SELECTION_OUTCOME={shlex.quote(str(row.get('selection_outcome') or ''))}",
        f"G8K_GRID_LIMITED={1 if row.get('grid_limited') else 0}",
        # E3: the it-doc row's binding facts — the lane, the object kind (which sidecar proves it), the ruled staged id the bank
        # entry must name, the record the bank door re-reads, and the pair's qualification (logged beside every leg)
        f"G8K_LANE={shlex.quote(str(row.get('lane') or ''))}",
        f"G8K_OBJECT_KIND={shlex.quote(str(row.get('object_kind') or ''))}",
        f"G8K_STAGED_ID={shlex.quote(str(row.get('staged_id') or ''))}",
        f"G8K_ITDOC_ADMISSION={shlex.quote(str(ITDOC_ADMISSION_REL) if row.get('object_source') == OBJECT_SOURCE_ITDOC else '')}",
        f"G8K_QUALIFICATION={shlex.quote(str(row.get('qualification') or ''))}",
        # E3: a row of a `row_overlays` matrix serves through its OWN pair (g8k_resolve_config); its object generation rides too
        f"G8K_OWN_OVERLAY={shlex.quote('1' if row.get('own_overlay') else '')}",
        f"G8K_OBJECT_GENERATION={shlex.quote(str(row.get('object_generation') or ''))}",
    ]


# --------------------------------------------------------------------------- the overlap legs
#: the models with an overlap leg: those whose rows run the thinking-ON arm (an OFF leg beside an ON store is the overlap
#: check); a NO-SWITCH family has no OFF arm to compare against — its overlap block is `not_applicable`
#: (gsm8k_overlap_check.py --models llama8b; llama-paper-rows P8), never a leg here
OVERLAP_MODEL_KEYS = tuple(k for k in MODEL_KEYS if THINKING_MODE_BY_FAMILY[FAMILY_BY_MODEL[k]] == THINKING_ON)


def overlap_legs(keys: tuple = OVERLAP_MODEL_KEYS) -> list[dict]:
    """One thinking-OFF greedy leg per SWITCH-ARM model's `none` (its `main` view's out root; the `it` store is the same
    model, its agreement recorded by the analysis, never pooled) — Nemotron's on the primary cluster only; a no-switch
    model (llama8b) refuses by name."""
    bad = [k for k in keys if k not in OVERLAP_MODEL_KEYS]
    if bad:
        raise SystemExit(f"no overlap leg for {bad}: a no-switch family has no thinking-OFF arm to compare against (its "
                         f"overlap block is not_applicable — gsm8k_overlap_check.py); the legs are {OVERLAP_MODEL_KEYS}")
    return [{"index": i, "job_id": f"g8k-overlap-{k}", "model_key": k, "view": VIEW_MAIN, "tag": tag_for(k, VIEW_MAIN),
             "cell": CELL_NONE, "out_root": OVERLAP_OUT_REL_TEMPLATE.format(key=k), "time_limit": TIME_OVERLAP,
             "cluster_constraint": placement_for(k), "family": FAMILY_BY_MODEL[k],
             "reasoning_parser": REASONING_PARSER_BY_MODEL.get(k), "gpu_mem_util": GPU_MEM_UTIL_BY_MODEL.get(k, GPU_MEM_UTIL_DEFAULT)}
            for i, k in enumerate(keys)]


def overlap_leg_for(model_key: str) -> dict:
    for leg in overlap_legs():
        if leg["model_key"] == model_key:
            return leg
    raise SystemExit(f"no overlap leg for {model_key!r} (models: {OVERLAP_MODEL_KEYS})")


def overlap_leg_sh(leg: dict) -> str:
    """An overlap leg as the SAME shell assignments a matrix row gets (g8k_common.sh's functions read one vocabulary):
    thinking OFF, greedy, the OFF table's 512 as the cap (statement == resolution, no setter), the OFF class's entry,
    `none` in the model's `main` view, never fitted / declared / held."""
    lines = [
        f"G8K_INDEX={leg['index']}", f"G8K_JOB_ID={shlex.quote(leg['job_id'])}", f"KEY={shlex.quote(leg['model_key'])}",
        f"VIEW={VIEW_MAIN}", f"TAG={shlex.quote(leg['tag'])}", f"CELL={CELL_NONE}", f"CELLSLUG={CELL_NONE}",
        "FITTED=0", "DECLARED=0", f"RUNG={OVERLAP_CAP}", f"G8K_CAP={OVERLAP_CAP}", f"G8K_RUNG_CAP={RUNG_CAP}",
        f"G8K_MAX_TOKENS_BY_TASK={shlex.quote(f'{INSPECT_TASK}={OVERLAP_CAP}')}",
        f"G8K_GEOMETRY_EVAL={shlex.quote(OVERLAP_GEOMETRY_EVAL)}", f"G8K_GEOMETRY_L={OVERLAP_GEOMETRY_L}",
        "G8K_TUPLE=''", "G8K_ROUTE_KIND=base", f"G8K_FAMILY={leg['family']}", f"G8K_TIME={leg['time_limit']}",
        f"G8K_OUT_REL={shlex.quote(leg['out_root'])}", f"G8K_SOURCE_OUT_REL={shlex.quote(leg['out_root'])}",
        f"G8K_REASONING_PARSER={shlex.quote(leg['reasoning_parser'] or '')}",
        f"G8K_GPU_MEM_UTIL={leg['gpu_mem_util']}", "PIN=''", "G8K_STANDUP_CELL=none", "G8K_STANDUP_HARNESS_CELL=none",
        "G8K_SERVED_POSITIONS_EXPECTED=''",
        f"G8K_EXPECTED_STAMPS={shlex.quote('cell-stable,not-consulted,thinking_off')}",
        "G8K_SAMPLING_SOURCE='greedy: argmax decoding, one sample per item; no card value applies (contracts.GREEDY_SAMPLING)'",
        "G8K_SAMPLING_PARAMS=" + shlex.quote(json.dumps({"temperature": 0.0}, sort_keys=True)),
        f"G8K_BUDGET_CLASS={CELL_NONE}", f"G8K_CLUSTER_CONSTRAINT={leg['cluster_constraint']}",
        f"STATUS={STATUS_LAUNCH}", "G8K_HELD_REASON=''", "G8K_EXPECTED_REFUSAL=''",
    ]
    lines += external_sh({})            # the overlap legs are the published wave's, campaign-bank rows: every E2 fact empty
    literals = dict(RUN_LITERALS_SH)
    # the leg's OWN arm / sampling / root: OFF, greedy, the published wave (the legs belong to the default matrix alone)
    literals.update({"G8K_THINKING": OVERLAP_THINKING, "G8K_SAMPLING_MODE": OVERLAP_SAMPLING, "G8K_WAVE_ROOT_REL": WAVE_ROOT_REL})
    lines += [f"{k}={shlex.quote(str(v))}" for k, v in literals.items()]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- the plan
def time_classes(rows: list[dict]) -> dict[str, list[int]]:
    """--time class -> the array indices in it (one sbatch --array per class)."""
    out: dict[str, list[int]] = {}
    for r in rows:
        out.setdefault(r["time_limit"], []).append(r["index"])
    return out


def _array_spec(indices: list[int]) -> str:
    """Slurm's compact index list: runs collapsed to a-b."""
    parts: list[str] = []
    run = [indices[0]]
    for i in indices[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            parts.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
            run = [i]
    parts.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else str(run[0]))
    return ",".join(parts)


def time_limit_hours(tl: str) -> float:
    """Slurm's `D-HH:MM:SS` (the classes' spelling) -> hours."""
    days, _, hms = tl.partition("-")
    h, m, sec = (int(x) for x in hms.split(":"))
    return int(days) * 24 + h + m / 60 + sec / 3600


def parse_index_spec(spec: str, n: int) -> list[int]:
    """Slurm's array-index grammar (`0-7,12,15-23`) -> sorted distinct row indices in 0..n-1; anything else refuses by
    name (a repeated index, an index outside the matrix, a non-index)."""
    out: list[int] = []
    for part in spec.split(","):
        a, dash, b = part.strip().partition("-")
        if not a.isdigit() or (dash and not b.isdigit()):
            raise SystemExit(f"--rows/--cluster-share: {part!r} is not an index or an a-b range")
        lo, hi = int(a), (int(b) if dash else int(a))
        if lo > hi:
            raise SystemExit(f"--rows/--cluster-share: {part!r} runs backwards")
        for i in range(lo, hi + 1):
            if not 0 <= i < n:
                raise SystemExit(f"--rows/--cluster-share: index {i} is outside 0..{n - 1}")
            if i in out:
                raise SystemExit(f"--rows/--cluster-share: index {i} is named twice")
            out.append(i)
    return sorted(out)


def refuse_held(indices: list[int], universe: list[dict], *, flag: str) -> None:
    held = [universe[i] for i in indices if universe[i]["status"] == STATUS_HELD]
    if held:
        raise SystemExit(f"{flag} names HELD row(s) {[(r['index'], r['job_id']) for r in held]} — held: "
                         f"{held[0]['held_reason']}; a held row is never launched (lift the hold where it lives — configs/sdfbase/gsm8k/"
                         "views.yaml for a D8 hold, the it-doc admission record for an it-doc row — and regenerate the matrix)")


def refuse_misplaced(rows: list[dict], cluster: str, *, flag: str) -> None:
    """Model-based placement: a `primary` row refuses every cluster but the primary one, by name."""
    if cluster == PRIMARY_CLUSTER:
        return
    bad = [r for r in rows if r["cluster_constraint"] == PLACEMENT_PRIMARY]
    if bad:
        raise SystemExit(f"{flag}: row(s) {[(r['index'], r['job_id']) for r in bad]} carry cluster_constraint "
                         f"{PLACEMENT_PRIMARY} (the Nemotron fused-kernel serving overlay builds and is attested on the "
                         f"primary cluster alone) and cannot run on {cluster} — keep them in the {PRIMARY_CLUSTER} share")


def select_rows(*, rows_spec: str | None = None, cluster_share: str | None = None, cluster: str | None = None,
                universe: list[dict] | None = None, wave: str | None = None) -> list[dict]:
    """The LAUNCH rows this submission carries (their matrix indices intact: the ARRAY task resolves its row from
    SLURM_ARRAY_TASK_ID). With `wave` (E2, plan P2 / F11e) the universe of choice is that wave's ELIGIBLE set
    (:func:`eligible_rows`): `--rows SPEC` must name a subset of it, `--cluster-share SPEC` names the PRIMARY cluster's
    subset of it and ONE string passed on both clusters yields the primary's set and the overflow's complement WITHIN the
    wave — never against the whole matrix (a complement over the matrix would select the published, COMPLETE rows). Without
    a wave (the python entry point the partition tests use) the universe is taken as given: every launched row of it. A held
    index refuses by name on either flag; a `primary` row selected for the overflow cluster refuses by name."""
    universe = matrix_rows() if universe is None else universe
    if rows_spec is not None and cluster_share is not None:
        raise SystemExit("--rows and --cluster-share are exclusive: name this cluster's rows, or the primary cluster's share")
    if cluster not in CLUSTERS:
        raise SystemExit(f"cluster must be one of {sorted(CLUSTERS)}, got {cluster!r}")
    eligible = eligible_rows(wave, universe) if wave is not None else [r for r in universe if r["status"] == STATUS_LAUNCH]
    launch_idx = [r["index"] for r in eligible]
    if rows_spec is not None:
        idx = parse_index_spec(rows_spec, len(universe))
        refuse_held(idx, universe, flag="--rows")
        if wave is not None:
            _refuse_outside_wave(idx, wave, eligible, universe, "--rows")
    elif cluster_share is not None:
        share = parse_index_spec(cluster_share, len(universe))
        refuse_held(share, universe, flag="--cluster-share")
        if wave is not None:
            _refuse_outside_wave(share, wave, eligible, universe, "--cluster-share")
        idx = sorted(share) if cluster == PRIMARY_CLUSTER else sorted(set(launch_idx) - set(share))
        if not idx:
            raise SystemExit(f"--cluster-share {cluster_share}: no rows are left over on {cluster}")
    else:
        idx = launch_idx
    selected = [universe[i] for i in idx]
    refuse_misplaced(selected, cluster, flag=f"--cluster {cluster}")
    return selected


def partition_widths(selected: list[dict], max_concurrent: int) -> dict[str, int]:
    """The AGGREGATE allowance W split into per-array `%w`: sum(w) = min(W, rows) <= W, every class present keeps >= 1,
    no w exceeds its class's rows; the split minimises the wave's tail under Slurm's static per-array throttles
    (max over classes of ceil(rows / w) x time) by exhaustive search; ties: the sum of finish times, the wider split,
    the longest class first."""
    if not (isinstance(max_concurrent, int) and max_concurrent >= 1):
        raise SystemExit(f"--max-concurrent must be a positive integer, got {max_concurrent!r}")
    classes = time_classes(selected)
    order = sorted(classes, key=lambda tl: -time_limit_hours(tl))
    if max_concurrent < len(order):
        raise SystemExit(f"--max-concurrent {max_concurrent} is below the {len(order)} time classes in this selection "
                         f"({', '.join(CLASS_TAG.get(tl, tl) for tl in order)}): every array needs one slot — raise it "
                         f"or select fewer rows (--rows / --cluster-share)")
    sizes = [len(classes[tl]) for tl in order]
    hours = [time_limit_hours(tl) for tl in order]

    def objective(ws: tuple[int, ...]):
        finish = [-(-n // w) * h for n, w, h in zip(sizes, ws, hours)]
        return (max(finish), sum(finish), -sum(ws), tuple(-w for w in ws))
    candidates = (ws for ws in itertools.product(*(range(1, n + 1) for n in sizes)) if sum(ws) <= max_concurrent)
    return dict(zip(order, min(candidates, key=objective)))


@dataclass
class Plan:
    """One cluster's submission: the sbatch lines (one per --time class present, longest first) and the facts the
    submitter reads before pasting them."""
    cluster: str
    sha: str
    max_concurrent: int
    rows: list[int]
    total_rows: int
    widths: dict[str, int]
    sizes: dict[str, int]
    lines: list[str]
    held: list[int]
    matrix: str = MATRIX_DEFAULT_NAME
    wave_root_rel: str = WAVE_ROOT_REL
    wave: str | None = None
    eligibility: dict | None = None
    tags: dict[str, str] | None = None      # the -J tag per --time class (class_tag_for over the class's rows)

    def tag(self, tl: str) -> str:
        return (self.tags or {}).get(tl) or CLASS_TAG[tl]

    @property
    def est_tail_h(self) -> float:
        return max(-(-self.sizes[tl] // w) * time_limit_hours(tl) for tl, w in self.widths.items())

    def summary(self) -> list[str]:
        c = cluster_site(self.cluster)
        used = sum(self.widths.values())
        per = ", ".join(f"{self.tag(tl)} {w}/{self.sizes[tl]} rows @{tl}" for tl, w in self.widths.items())
        out = [
            (f"# plan: cluster={self.cluster} qos={c['qos']} wave={self.wave or 'ALL (no --wave: the universe as given)'} "
             f"rows={len(self.rows)}/{self.total_rows} launched "
             f"({_array_spec(self.rows)}) held={_array_spec(self.held) if self.held else 'none'} (never emitted) "
             f"allowance={self.max_concurrent} = the AGGREGATE cap on tasks in flight (the per-array %w sum to it)"),
            (f"# widths, longest class first: {per} -> sum {used} <= {self.max_concurrent}; est. tail {self.est_tail_h:g} h "
             f"(each array throttles itself; a slot one array frees never moves to another)"),
            (f"# budget: choose W = the FREE `{c['qos']}` budget: {HIGH_BUDGET_GPUS} GPUs per user on this cluster MINUS every "
             f"`{c['qos']}` job already running or pending under this user, peers' included (the Slurm user is shared): "
             f"squeue -u $USER -h -q {c['qos']} -t RUNNING,PENDING -o '%b' and sum the gpu counts before choosing W"),
            (f"# placement: {PLACEMENT_PRIMARY} rows (nemotron3nano4b; an external object staged on the primary cluster alone) run "
             f"on {PRIMARY_CLUSTER} only; the plan refused none here, so every line below is placeable on {self.cluster}"),
        ] + ([] if self.matrix == MATRIX_DEFAULT_NAME else [
            (f"# matrix: {self.matrix} -> wave root <shared root>/{self.wave_root_rel} (its own root, never the published "
             f"wave's; the ARRAY line's trailing positionals name the rung and the matrix — g8k_common.sh evaluates the "
             f"matrix's literals before F22_WAVE_ROOT)"),
        ])
        if self.eligibility is not None:
            e = self.eligibility
            out.append(f"# assert-eligible: wave={e['wave']} rows={_array_spec(e['rows'])} all status=launch; excluded "
                       + ", ".join(f"{k}: {v}" for k, v in e["excluded"].items())
                       + " — no published (complete), held or deferred row is named")
        return out


def array_tail(sha: str, *, rung: int = CAP, matrix: Path = MATRIX_PATH) -> str:
    """The sbatch's trailing positionals after the script: `ARRAY <sha>` on the published matrix at 8,192 (byte-identical to
    the launched plan); `ARRAY <sha> <rung>` for a rung of it (ladder_plan.py); `ARRAY <sha> <rung> <matrix>` on any other
    matrix, by NAME (the sbatch refuses an unknown name before sourcing anything)."""
    name = matrix_name_of(matrix)
    if name == MATRIX_DEFAULT_NAME:
        return f"ARRAY {sha}" if rung == CAP else f"ARRAY {sha} {rung}"
    if name not in MATRICES and not name.startswith(FRONT_DOOR_MATRIX_PREFIX):
        raise SystemExit(f"a plan names a matrix by NAME ({sorted(MATRICES)}, or {FRONT_DOOR_MATRIX_PREFIX}<key>); {matrix} is not one — register it in g8k_cells.MATRICES")
    return f"ARRAY {sha} {rung} {name}"


def plan(sha: str, cluster: str, *, max_concurrent: int, rows: str | None = None, cluster_share: str | None = None,
         universe: list[dict] | None = None, matrix: Path = MATRIX_PATH, wave: str | None = None,
         assert_eligible_record: bool = False) -> Plan:
    """The 8,192 submission, PRINTED never run: one `sbatch --array` line per --time class present in the selection, over
    ONE script, at the cluster's QoS, positional args only (ARRAY <sha>: the task resolves its row from
    SLURM_ARRAY_TASK_ID; `ARRAY <sha> 8192 <matrix>` on a non-default matrix); the per-array `%w` partition ONE aggregate
    allowance; held rows never appear. With `wave` (E2) the selection is wave-scoped and EVERY printed row is asserted
    eligible (a completed / held / deferred row refuses by name); `assert_eligible_record` adds the record line."""
    universe = matrix_rows(matrix) if universe is None else universe
    c = cluster_site(cluster)
    selected = select_rows(rows_spec=rows, cluster_share=cluster_share, cluster=cluster, universe=universe, wave=wave)
    eligibility = None
    if wave is not None:
        eligibility = assert_eligible(selected, wave, universe)
        if not assert_eligible_record:
            eligibility = None
    widths = partition_widths(selected, max_concurrent)
    classes = time_classes(selected)
    script = f"{CHECKOUT_ROOT}/{sha}/{SBATCH_REL}"
    tail = array_tail(sha, matrix=matrix)
    tags = {tl: class_tag_for(tl, [universe[i] for i in classes[tl]]) for tl in widths}
    if matrix_name_of(matrix) == MATRIX_ITDOC_NAME:
        tags = {tl: MATRIX_ITDOC_NAME for tl in widths}          # The instruct-direct wave's jobs are named for the wave (`-J g8k-itdoc`)
    lines = [f"sbatch -J g8k-{tags[tl]} --qos={c['qos']} --time={tl} "
             f"--output={c['logs']}/%x_%A_%a.out --array={_array_spec(classes[tl])}%{w} "
             f"{script} {tail}" for tl, w in widths.items()]
    # a submission whose EVERY row pins the lens BY KIND (AlphaSteer rows only) carries the lens route on its line too: the job pins it
    # per row regardless (g8k_select_serving_path), the prefix makes the planned line say what the standup will do. A submission with an
    # Arditi row gets NO prefix: that row's pin follows its STORE on the cluster (`store:lens` / `unpinned:no-store` / `store:native`),
    # which the grading host plan cannot read — a prefix would override an unpinned row as EXPLICIT (2026-09-15)
    if selected and all(serving_path_for(r) == SERVING_PATH_LENS for r in selected):
        lines = [f"F22_SERVING_PATH={SERVING_PATH_LENS} {line}" for line in lines]
    return Plan(cluster=cluster, sha=sha, max_concurrent=max_concurrent, rows=[r["index"] for r in selected],
                total_rows=len(universe), widths=widths, sizes={tl: len(classes[tl]) for tl in widths}, lines=lines,
                held=[r["index"] for r in universe if r["status"] == STATUS_HELD],
                matrix=matrix_name_of(matrix), wave_root_rel=wave_root_rel_of(matrix_doc(matrix)), wave=wave, eligibility=eligibility,
                tags=tags)


def plan_overlap(sha: str, cluster: str, *, keys: tuple = OVERLAP_MODEL_KEYS) -> list[str]:
    """The overlap legs' ONE array line for this cluster (overlap_off.sbatch ARRAY <sha>: the task resolves its model
    from SLURM_ARRAY_TASK_ID over overlap_legs()); Nemotron's leg refuses the overflow cluster by name."""
    c = cluster_site(cluster)
    legs = [leg for leg in overlap_legs() if leg["model_key"] in keys]
    if not legs or len(legs) != len(keys):
        raise SystemExit(f"plan-overlap: no overlap leg for {sorted(set(keys) - {l['model_key'] for l in legs})} (the legs: "
                         f"{OVERLAP_MODEL_KEYS}; a no-switch family's overlap is not_applicable)")
    refuse_misplaced(legs, cluster, flag=f"plan-overlap --cluster {cluster}")
    script = f"{CHECKOUT_ROOT}/{sha}/{OVERLAP_SBATCH_REL}"
    idx = [leg["index"] for leg in legs]
    return [(f"sbatch -J g8k-overlap --qos={c['qos']} --time={TIME_OVERLAP} --output={c['logs']}/%x_%A_%a.out "
             f"--array={_array_spec(idx)}%{len(idx)} {script} ARRAY {sha}")]


def _check_sha(sha: str) -> str:
    if not (len(sha) == 40 and all(ch in "0123456789abcdef" for ch in sha)):
        raise SystemExit(f"--sha must be a full 40-hex commit sha, got {sha!r}")
    return sha


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    matrix_help = f"the matrix: a name ({', '.join(MATRICES)}) or a path (default: the frozen published matrix)"
    p = sub.add_parser("list"); p.add_argument("--json", action="store_true"); p.add_argument("--matrix", default=None, help=matrix_help)
    p = sub.add_parser("row"); p.add_argument("--index", type=int, default=None); p.add_argument("--key"); p.add_argument("--view")
    p.add_argument("--cell"); p.add_argument("--rung", type=int, default=CAP); p.add_argument("--sh", action="store_true")
    p.add_argument("--serving-path", action="store_true",
                   help="print the serving path the row PINS and its reason, one per line (path empty = unpinned: the kind-aware default stands); "
                        "an external row's pin FOLLOWS ITS STORE at --wave-root/<out roots>, so --wave-root is required for it")
    p.add_argument("--wave-root", default=None, help="the matrix's wave root on THIS cluster (F22_WAVE_ROOT), under which the row's store on record is read")
    p.add_argument("--store-pins", action="append", default=None, metavar="JSON",
                   help="the store-pins record(s) whose _meta.serving_engine_by_leg the pin reads for the row's admitted legs (default: the repo's "
                        "configs/sdfbase/gsm8k/store_pins_{default,llama,itdoc,arditi_allpos}.json — the launcher never overrides this; a test seam)")
    p.add_argument("--declared-serving-path", action="store_true",
                   help="print the serving path the ROW'S RECORD declares for every leg of its arm, or nothing at all when it declares none "
                        "(g8k_cells.declared_serving_path) — the launcher refuses a conflicting F22_SERVING_PATH before f22_init fixes one")
    p.add_argument("--launch-cells", action="store_true",
                   help="print the HARNESS cells the row's standup serves, one per line (`none` + the row's standup cell) — the sbatch declares "
                        "them as F22_LAUNCH_CELLS before f22_init (the kind-aware serving-path default reads them)")
    p.add_argument("--assert-launchable", action="store_true",
                   help="the batch boundary's predicate (g8k_assert_launchable): exit 0 with the record iff the row is in its wave's "
                        "eligible set; a published (complete), held or deferred row refuses by name on stderr, exit 1")
    p.add_argument("--matrix", default=None, help=matrix_help)
    p = sub.add_parser("counts"); p.add_argument("--matrix", default=None, help=matrix_help)
    p = sub.add_parser("literals"); p.add_argument("--sh", action="store_true"); p.add_argument("--matrix", default=None, help=matrix_help)
    p = sub.add_parser("plan"); p.add_argument("--sha", required=True); p.add_argument("--cluster", required=True)
    p.add_argument("--wave", required=True, help="the wave whose launchable rows this submission carries (ext for the eight-arm prompted rows, "
                                                 "steer for the AlphaSteer / Arditi rung); a matrix-wide plan would name COMPLETE rows and is refused")
    p.add_argument("--assert-eligible", action="store_true", help="print the eligibility record (the assertion itself always runs)")
    p.add_argument("--max-concurrent", type=int, default=12, help="the AGGREGATE cap on tasks in flight across the arrays (the free high budget)")
    p.add_argument("--rows", default=None, help="this cluster's rows (a subset of the wave's eligible set), Slurm index grammar (42-44,46); a held index refuses")
    p.add_argument("--cluster-share", default=None, help="the PRIMARY cluster's rows of the wave; the same string with --cluster cluster_b selects the complement WITHIN the wave")
    p.add_argument("--matrix", default=None, help=matrix_help)
    p = sub.add_parser("plan-overlap"); p.add_argument("--sha", required=True); p.add_argument("--cluster", required=True)
    p.add_argument("--keys", default=",".join(OVERLAP_MODEL_KEYS), help="the models whose overlap leg this cluster carries (switch-arm models only)")
    p = sub.add_parser("overlap-leg"); p.add_argument("--index", type=int, default=None); p.add_argument("--key")
    p.add_argument("--sh", action="store_true")
    a = ap.parse_args(argv)
    matrix = resolve_matrix(getattr(a, "matrix", None))
    if a.cmd == "list":
        rows = matrix_rows(matrix)
        if a.json:
            print(json.dumps(rows, indent=1))
        else:
            for r in rows:
                print(f"{r['index']:2d} {r['job_id']:<36} {r['tag']:<20} {r['cell']:<50} fitted={int(r['fitted'])} "
                      f"declared={int(r['declared'])} {r['cluster_constraint']:<7} time={r['time_limit']} status={r['status']:<8} "
                      f"wave={row_wave(r)}" + (f" object_source={r['object_source']}" if r.get("object_source") else ""))
        return 0
    if a.cmd == "row":
        rows = matrix_rows(matrix)
        if a.index is not None:
            if not 0 <= a.index < len(rows):
                raise SystemExit(f"--index {a.index} outside 0..{len(rows) - 1}")
            row = rows[a.index]
        else:
            if not (a.key and a.view and a.cell):
                raise SystemExit("row: pass --index N, or --key K --view V --cell C")
            row = row_for(a.key, a.view, a.cell, rows)
        if a.assert_launchable:
            print(launchable_line(row, assert_launchable(row, rows)))
            return 0
        if a.serving_path:
            if row.get("object_source") == OBJECT_SOURCE_EXTERNAL and not a.wave_root:
                raise SystemExit(f"row --serving-path: {row['job_id']} is an external-object row, whose pin FOLLOWS ITS STORE — pass "
                                 "--wave-root <this cluster's wave root> (the launcher passes F22_WAVE_ROOT); a store is never assumed absent")
            path, reason = serving_path_pin(row, wave_root=a.wave_root, store_pins=a.store_pins)
            sys.stdout.write(SP.pin_lines(path, reason))
            return 0
        if a.declared_serving_path:
            declared = declared_serving_path(row)
            sys.stdout.write(f"{declared}\n" if declared else "")
            return 0
        if a.launch_cells:
            sys.stdout.write("\n".join(launch_cells_for(row)) + "\n")
            return 0
        sys.stdout.write(row_sh(row, a.rung, wave_root_rel=wave_root_rel_of(matrix_doc(matrix))) if a.sh else json.dumps(row, indent=1) + "\n")
        return 0
    if a.cmd == "counts":
        print(json.dumps(counts(matrix_rows(matrix)), indent=1))
        return 0
    if a.cmd == "literals":
        sys.stdout.write(literals_sh(matrix) if a.sh
                         else json.dumps({**dict(RUN_LITERALS_SH), "G8K_WAVE_ROOT_REL": wave_root_rel_of(matrix_doc(matrix))}, indent=1) + "\n")
        return 0
    if a.cmd == "plan":
        pl = plan(_check_sha(a.sha), a.cluster, max_concurrent=a.max_concurrent, rows=a.rows, cluster_share=a.cluster_share, matrix=matrix,
                  wave=a.wave, assert_eligible_record=a.assert_eligible)
        for line in pl.summary() + pl.lines:
            print(line)
        return 0
    if a.cmd == "plan-overlap":
        keys = tuple(k for k in a.keys.split(",") if k)
        for line in plan_overlap(_check_sha(a.sha), a.cluster, keys=keys):
            print(line)
        return 0
    if a.cmd == "overlap-leg":
        legs = overlap_legs()
        if a.index is not None:
            if not 0 <= a.index < len(legs):
                raise SystemExit(f"--index {a.index} outside 0..{len(legs) - 1} (the overlap legs)")
            leg = legs[a.index]
        elif a.key:
            leg = overlap_leg_for(a.key)
        else:
            raise SystemExit("overlap-leg: pass --index N or --key K")
        sys.stdout.write(overlap_leg_sh(leg) if a.sh else json.dumps(leg, indent=1) + "\n")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
