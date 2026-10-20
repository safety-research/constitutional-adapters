#!/usr/bin/env python3
"""math500-thinking-on-full22 — THE ONE ENUMERATED MATRIX: ``M5T_CELLS``.

Rows 0-23 are the PUBLISHED matrix, frozen (tree 55e3cbdd; every leg COMPLETE and reused byte-for-byte): 12 PRIMARY
rows (``--sampling thinking_card``: 4 models x {none, prompted__safety_refusal, constv2-mix-full__lora_r4_ce_minus_ctl}),
3 qwen35_4b ``general`` rows, 9 ``battery`` rows. EIGHT of them are FITTED (the object cell under each of its sampling
modes): they need the OOS override, the artifact digest and an authorised tuple; none is declared (every object here is
``positions: all`` — lens rule 3 refuses a ``@thinking`` spelling). Those 24 rows regenerate BYTE-IDENTICALLY on every
published key (tests/fixtures/math500_matrix_55e3cbdd.json) — a row's index is its Slurm array id, its out root, pin
and tuple are what the analysis reads — and additionally carry the EXTENSION KEYS (``EXTENSION_KEYS``: wave, status,
geometry, budget class, ...), which the eight-arm analysis reads on every row.

Rows 33-40 are the AlphaSteer / Arditi steering rung (wave ``steer``, external objects pinned by the admission record);
rows 41-46 are the Llama rows (wave ``llama``: the no-switch family's six non-steering arms under
``thinking_mode not_available`` and its shipped ``card`` sampling row — the MODE a no-switch family runs, since
``--sampling thinking_card`` on llama8b is refused by contracts.sampling_for; the steering llama rows run under it too).

Row 53 is the ALL-POSITIONS ARDITI arm (wave ``allpos``, arditi-qwen-all-positions): ONE row, qwen35_4b x
``arditi_allpos`` — the all-positions refusal direction, a ``vector_all`` object served at every position in BOTH
phases on the graph-captured NATIVE residual slot. It is NOT the ``arditi`` cell of rows 33-40 (prefill-only, dose an
alpha): a different intervention, never pooled into that slot or relabelled into it. APPENDED as its own wave, after
``ITDOC_ROWS``, precisely so every existing index, wave slice and count stays where it is — adding it to ``STEER_ROWS``
would shift every later row's Slurm array id, out root and pin.

Rows 54-86 are the QWEN SEED-REPLICATE wave (wave ``qwen_seedrep``): per seed view ``s1`` / ``s2`` / ``s3`` the eight
arms — none, constitution / control ReLU (the PSR cells), constitution / control LoRA, the LoRA difference, the two
prompted SIs — (24), per paired view ``s<s>p`` the COMPOSED difference ReLU
``constv2-mix-full__psr_ce_all_minus_ctl`` with NO own ``none`` (3; it pairs with ``s<s>/none``), and per seed view the
Arditi all-positions refusal steer at seed base s (3): the batch-64 objects (``base64s<s>__…`` / ``derived_base64s<s>__…`` /
``paired_eb64s<s>__…`` in the full22 pin file) served NATIVELY on the OVERFLOW cluster only, each row under the
opt-in Inspect REPLICATE seed rule at ITS view's base (``--inspect-seed-policy replicate --inspect-seed-base s``), with
the view's r=4 companion registered on every standup and ``--bank-view`` stamped. These rows are a SIBLING LIST beside
the matrix — ``SEEDREP_ROWS``, indices 54-86, resolved by the launcher through
``LAUNCHER_ROWS = M5T_CELLS + SEEDREP_ROWS`` — never rows of ``M5T_CELLS``: the MATH-500 reader (tinycap_math500_thinking
``load_cells``) refuses a cell outside its arm registry (no PSR arm) and a ``controls*/<key>_<view>/…`` out root, and it
loads the LIVE ``M5T_CELLS``, so appending them there would refuse the paper's eight-arm analysis (the GSM8K lane's
``arditi_seedrep`` RECORD is the precedent). Every existing index, slice and count is untouched.

Rows 24-32 are the eight-arm EXTENSION (math-eight-arms-extension plan P2 / §4.1, wave ``ext1``): per
gemma4e4b / qwen35_4b / nemotron3nano4b the prompted constitution cell (``prompted__si_constitution`` — the SHORT
constitution SI, harness ``prompted__constitution``; NEVER ``prompted__si_constitution_full``, refused by name) and the
two extra all-position LoRA cells (``constv2ctl2-full__lora_r4_ce``, ``constv2-mix-full__lora_r4_ce``), each under the
model's PRIMARY sampling row, thinking ON, one pass at 16,384 — under a WAVE-ISOLATED out root (``controls_ext1/...``,
so a pull of the wave's subtree never carries a published leg) and with the sampler read-back opt-in on. The
constitution rows resolve the LONG-SI geometry entry (``capability:gen_long_math_longsi``, L 24,064 — measured for the
class, plan F4 / A10); the LoRA rows run under the NEW ``_lora2`` tuples (contracts P4: minus_ctl's shape, new names).
Llama's rows and the AlphaSteer / Arditi stubs are E2's and are not here.

IMPORT-LIGHT ON PURPOSE (stdlib + json only): the sbatch reads it with whatever python is on the node before the
venvs exist, and the workstation dry run reads it without torch. The literals mirrored from ``contracts`` (flag
spellings, the cap, the timeout, the identity version, the harness ids, the prompt classes and their geometry
entries, the extension cells and models, the SI digests) are pinned EQUAL to contracts' / the registry's by
tests/test_sdfbase_math500_sbatch.py — the mirror is checked, never trusted.

CLI (the sbatch and the dry run):
  cells.py list [--json] [--published-keys]     every row the launcher resolves (LAUNCHER_ROWS: the matrix + the seed
                                               wave; --published-keys: rows 0-23 on the published keys only —
                                               byte-identical to the frozen fixture)
  cells.py row --index N --sh                  one row as shell assignments (the ARRAY task's lookup)
  cells.py row --key K --cell C --sampling S   one row (JSON, or --sh)
  cells.py row (--index N | --key K --cell C --sampling S) --assert-launchable
                                               the BATCH BOUNDARY's predicate (m5t_assert_launchable): exit 0 +
                                               the record iff the row is in ITS wave's eligible set (status launch, the
                                               wave not complete); a published / held / deferred row refuses by name
  cells.py row (--index N | --key K --cell C --sampling S) --serving-path --wave-root ROOT
                                               the serving path the row PINS and its reason, one per line: the pin FOLLOWS THE
                                               STORE at ROOT/<out root> (kind:alphasteer | store:lens | unpinned:no-store |
                                               store:native; a mixed-engine store refuses by name; --wave-root is required on an
                                               external row) — what m5t_select_serving_path declares before f22_init
  cells.py row --index N --identity --sha SHA [--out-root PATH]
                                               the row-identity stamp (IFEval's superset + pin); sampling_ref DERIVED
                                               from the driver's manifest sampling block (contracts.sampling_ref) under
                                               --out-root and held to gen_stats + the outcome items — `complete: false`
                                               with the reasons otherwise (this path imports the contracts lazily)
  cells.py counts                              the acceptance counts, derived from the matrix (base / extension apart)
  cells.py plan --sha SHA --cluster primary|overflow --wave NAME [--assert-eligible] [--single-array]
                [--max-concurrent W] [--rows SPEC | --cluster-share SPEC]
                                               the sbatch --array lines for ONE WAVE's launchable rows, PRINTED. The
                                               eligible set is that wave's `status: launch` rows; --rows names a subset
                                               of it, --cluster-share names the PRIMARY cluster's subset of it (the same string
                                               on the overflow cluster selects the complement WITHIN the wave — never against the
                                               whole matrix); W is the AGGREGATE cap on tasks in flight. --cluster is the
                                               cluster's ROLE (CLUSTER_BY_ROLE); its name is accepted too.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import importlib.util
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PINS_PATH = REPO / "configs" / "sdfbase_full22_pins.json"
_FULL22 = str(REPO / "scripts" / "sdfbase" / "full22")
if _FULL22 not in sys.path:
    sys.path.insert(0, _FULL22)
import itdoc_bind as B   # noqa: E402 — stdlib-only, like this file: the it-doc admission record + the bank binding (E3)
import steer_store_pin as SP   # noqa: E402 — stdlib-only: an external row's serving-path pin FOLLOWS ITS STORE

# --------------------------------------------------------------------------- the vocabulary
MODEL_KEYS = ("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b")
CELL_NONE = "none"
CELL_PROMPTED = "prompted__safety_refusal"
CELL_OBJECT = "constv2-mix-full__lora_r4_ce_minus_ctl"
#: the PUBLISHED matrix's three cells (rows 0-23), in its order
CELLS = (CELL_NONE, CELL_PROMPTED, CELL_OBJECT)
#: the eight-arm EXTENSION's cells (math-eight-arms-extension P3 / P4), in the eight-arm order — none · prompted safety ·
#: prompted constitution · LoRA ctl2 · LoRA mix · LoRA mix−ctl (· AlphaSteer · Arditi, E2's stubs)
CELL_PROMPTED_CONST = "prompted__si_constitution"
CELL_LORA_CTL2 = "constv2ctl2-full__lora_r4_ce"
CELL_LORA_MIX = "constv2-mix-full__lora_r4_ce"
EXT_CELLS = (CELL_PROMPTED_CONST, CELL_LORA_CTL2, CELL_LORA_MIX)
#: every cell the matrix names, in the overlays' explicit_cells order (base + extension)
ALL_CELLS = CELLS + EXT_CELLS
#: the models the extension runs on — contracts.M5T_EXT_MODELS (test-pinned): glm is the appendix model (P9 / Q1), llama E2's
EXT_MODELS = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b")
#: the two EXTERNAL steering cells, in the eight-arm order — bare
#: Class B ids the SERVING grammar admits (`alphasteer` in defense_cells.CLASS_B_IDS, `arditi` in its
#: SERVED_ONLY_CLASS_B_IDS): one object per model, SI-agnostic, served from ANOTHER LANE's bank view and pinned by the
#: ADMISSION RECORD's object sha256 (STEER_ADMISSION_PATH), never by the full22 pin file (which stages neither)
CELL_ALPHASTEER = "alphasteer"
CELL_ARDITI = "arditi"
#: the ALL-POSITIONS Arditi refusal direction — a `vector_all` object from another lane's bank, served at
#: EVERY position in BOTH phases (prefill and every decode step) on the graph-captured NATIVE residual slot. A DIFFERENT
#: INTERVENTION from `arditi` above (prefill-only, `static_r`, dose an alpha), never pooled with it, never relabelled into
#: its slot: its dose is a RAW COEFFICIENT c on the vector's norm, its view spells the coefficient (`ar_c0p3`, never `ar_r…`)
#: and its staged id ends `__arditi_allpos`. APPENDED to the steer vocabulary so the eight-arm panel order above is untouched.
CELL_ARDITI_ALLPOS = "arditi_allpos"
#: the refusal CODE row 53's comparison is DECLARED to refuse with — the reader's O106 cross-engine ruling
#: (``tinycap_math500_thinking.CROSS_ENGINE_RULING``; the two spellings are pinned equal by test). The all-positions arm is
#: served NATIVELY (lora_native) on the `it` panel view and is anchored on the main view's `none` leg, a LENS store; the
#: reader's compatibility rule refuses that pairing by name (O106: the native-vs-lens numerics difference was never
#: measured). The ruling in force: NO native `none` anchor is run for GSM8K or MATH-500 — the arm's
#: numbers are reported AS the arm's numbers and the pairing's absence accepted. So the design DECLARES the refusal here
#: (the GSM8K design's `expected_refusal`, mirrored) and the reader's population gate passes the comparison as refused BY
#: DESIGN iff its refusal derives EXACTLY this code; any other refusal, or a contrast that computes, still fails the gate
EXPECTED_REFUSAL_CROSS_ENGINE = "O106"
STEER_CELLS = (CELL_ALPHASTEER, CELL_ARDITI, CELL_ARDITI_ALLPOS)
#: the steering RUNG's two cells — what `_build_steer_rows` enumerates (rows 33-40). `STEER_CELLS` is the steer VOCABULARY (the
#: cells whose object comes from another lane's bank: the kind table, the fitted set and the tuple rule read it); the
#: all-positions arm is in that vocabulary but is its own WAVE, so the rung's rows and indices do not move.
WAVE_STEER_CELLS = (CELL_ALPHASTEER, CELL_ARDITI)
#: the ALL-POSITIONS wave (row 53): one cell, one model — the admission record admits `arditi_allpos` on qwen35_4b alone
ALLPOS_CELLS = (CELL_ARDITI_ALLPOS,)
ALLPOS_MODELS = ("qwen35_4b",)
#: the three PSR cells the seed wave adds to this launcher's
#: vocabulary — the constitution / control ReLU parents and the COMPOSED paired-write difference (the undistilled u_pos - u_neg,
#: admitted by derived_admission.py's psr_paired_write branch; staged `paired_eb64s<s>__…`, served on the paired views `s<s>p`).
#: Launcher spellings; harness ids in HARNESS_BY_CELL (pinned to registry_adapter.harness_cell_id). NOT cells of the eight-arm
#: reader's arm registry (tinycap_math500_thinking.ARMS) — the reason the seed rows live beside M5T_CELLS.
CELL_PSR_MIX = "constv2-mix-full__psr_ce_all"
CELL_PSR_CTL2 = "constv2ctl2-full__psr_ce_all"
CELL_PAIRED = "constv2-mix-full__psr_ce_all_minus_ctl"
SEEDREP_ONLY_CELLS = (CELL_PSR_MIX, CELL_PSR_CTL2, CELL_PAIRED)
#: the models the steering rung runs on — FOUR (glm is the appendix model and gets no steering row).
#: llama8b is a STEER-ONLY model here: it is deliberately NOT in MODEL_KEYS, so no published row, no published count,
#: no overlay of the base wave and no pin lookup moves for it; its rows carry their own facts below and are HELD
#: until its render fact is certified on this tree (RENDER_FACT_CERTIFIED_BY_FAMILY)
STEER_MODELS = ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b")
#: the STEER-ONLY model keys (in STEER_MODELS, outside MODEL_KEYS): every table below carries their facts explicitly
STEER_ONLY_MODELS = tuple(k for k in STEER_MODELS if k not in MODEL_KEYS)
#: the NO-SWITCH family's six non-steering arms — Llama IS in MATH-500
#: (superseding the 2026-09-11 exclusion) under `thinking_mode: not_available` (the certified
#: render fact of family `llama`, on main), its `main` view's objects from the shared llama8b roster block
#: (the lane-key rule: the view suffix names the object bank), the SHIPPED `card` sampling row. Rows 41-46, wave `llama`,
#: in the EIGHT-ARM order: none · prompted safety · prompted constitution · LoRA ctl2 · LoRA mix · LoRA mix−ctl (the two
#: steering arms are the steering rung's rows 39-40). Reported individually and HELD OUT of every pooled mean.
LLAMA_MODELS = ("llama8b",)
LLAMA_CELLS = (CELL_NONE, CELL_PROMPTED, CELL_PROMPTED_CONST, CELL_LORA_CTL2, CELL_LORA_MIX, CELL_OBJECT)
#: E3 (math-eight-arms-extension; decisions-e3): the INSTRUCT-DIRECT ("it-doc") batch-64 RE-FIT row family — a SCOPE
#: ADDITION: this matrix carried NO `it`-view row (every LoRA row served the base-lane object transferred onto the instruct target,
#: `view: main`), so the paper's MATH-500 for gemma / qwen read as base-to-instruct only. Rows 47-52: the three LoRA arms in the
#: eight-arm order (LoRA ctl2 · LoRA mix · LoRA mix−ctl) on gemma4e4b (47-49) and qwen35_4b (50-52), `view: it`, the primary
#: `thinking_card` row, the pinned short-class geometry, wave `itdoc` under `controls_itdoc/`, generated from the it-doc ADMISSION
#: RECORD (configs/sdfbase/itdoc_admission.json — ONE file for both math instruments) keyed by (lane, model key, view, cell id) ->
#: object sha256: a row launches only when its entry carries the pin + its ruled staged id + a complete bank binding + no hold
#: reason, else it is HELD with the record's reason (never a placeholder sha). Every row is HELD today. The bare anchor of an it-view
#: contrast is the model's `main`-view `none` leg (the roster's two views serve ONE target model).
ITDOC_MODELS = ("gemma4e4b", "qwen35_4b")
ITDOC_CELLS = (CELL_LORA_CTL2, CELL_LORA_MIX, CELL_OBJECT)
VIEW_IT = "it"
#: the PANEL view the all-positions row sits on: `it` — the direction was fit against THIS
#: target's `it` wave view on the served instruct checkpoint (the entry's fit_provenance: fitted_on_revision ==
#: the served revision, no base-to-instruct transfer), and the source lane serves the `it` body. The row therefore KEYS
#: on `it` (`key_view`) and is cut from the `it` TAG's campaign config, while the LANE's bank view (`ar_c0p3`, the dose
#: spelled) rides the separate `bank_view` field — the GSM8K launcher's spelling, which keeps the panel view and the
#: object-bank view apart. The steering rung's rows keep their own shape (one `view` field naming the bank).
ALLPOS_PANEL_VIEW = VIEW_IT
OBJECT_SOURCE_ITDOC = B.OBJECT_SOURCE_ITDOC
ITDOC_ADMISSION_REL = B.ADMISSION_REL
ITDOC_ADMISSION_PATH = REPO / ITDOC_ADMISSION_REL


def load_itdoc_admission(path: Path = ITDOC_ADMISSION_PATH) -> dict:
    """The it-doc admission record (itdoc_bind.load_admission: every entry validated by name at load)."""
    return B.load_admission(path)


#: the id the harness REFUSES (not a legal prompted id; the matrix generator refuses it by name too)
CELL_PROMPTED_CONST_FULL_REFUSED = "prompted__si_constitution_full"
#: each fitted cell's STAGED id in configs/sdfbase_full22_pins.json (the bank view maps the cell id to it)
STAGED_ID_BY_CELL = {
    CELL_OBJECT: "derived__constv2-mix-full__lora_r4_ce_minus_ctl",
    CELL_LORA_MIX: "base__constv2-mix-full__lora_r4_ce",
    CELL_LORA_CTL2: "base__constv2ctl2-full__lora_r4_ce",
}
OBJECT_STAGED_ID = STAGED_ID_BY_CELL[CELL_OBJECT]
#: the fitted cells the FULL22 PIN FILE stages (the base wave's + the seed wave's). The steering cells are fitted too, but their
#: objects are another lane's and are pinned by the ADMISSION RECORD — never looked up here (load_pins would refuse)
FITTED_CELLS = tuple(STAGED_ID_BY_CELL)
#: every fitted cell of the matrix, pin file or admission record
ALL_FITTED_CELLS = FITTED_CELLS + STEER_CELLS
OBJECT_HARNESS_ID = "si_constv2_mix_full__lora_r4_ce_minus_ctl"
#: the harness (manifest) spelling of each TrainTest cell — registry_adapter.harness_cell_id's, mirrored import-light
HARNESS_BY_CELL = {
    CELL_NONE: "none", CELL_PROMPTED: "prompted__safety", CELL_OBJECT: OBJECT_HARNESS_ID,
    CELL_PROMPTED_CONST: "prompted__constitution",
    CELL_LORA_MIX: "si_constv2_mix_full__lora_r4_ce", CELL_LORA_CTL2: "si_constv2ctl2_full__lora_r4_ce",
    # the bare Class B ids: the harness id IS the TrainTest id (registry_adapter.harness_cell_id, test-pinned)
    CELL_ALPHASTEER: CELL_ALPHASTEER, CELL_ARDITI: CELL_ARDITI,
    # arditi-qwen-all-positions: a bare Class B id like its neighbours, so its harness id is the same string. Registered
    # in the serving vocabulary since 2026-09-17, so the mirror is pinned to the adapter for this cell like every other
    # (PENDING_SERVING_CELLS, the declared gap it sat in while the registration was in flight, is now empty).
    CELL_ARDITI_ALLPOS: CELL_ARDITI_ALLPOS,
    # the seed wave's PSR cells (registry_adapter.harness_cell_id's spellings, test-pinned)
    CELL_PSR_MIX: "si_constv2_mix_full__psr_ce_all", CELL_PSR_CTL2: "si_constv2ctl2_full__psr_ce_all",
    CELL_PAIRED: "si_constv2_mix_full__psr_ce_all_minus_ctl",
}
#: THE DECLARED, CHECKED GAP (arditi-qwen-all-positions): the matrix cells the SERVING vocabulary does not admit on this
#: tree. EMPTY today. It carried `arditi_allpos` while that cell's Class B registration was in flight in another change;
#: the registration landed and the entry came out, which is the mechanism working as designed — the set is HAND-NAMED,
#: not derived, precisely so that clearing it is a deliberate act, and a test pins it EQUAL to the cells the vocabulary
#: actually refuses, so the day one is registered the mirror disagrees and says so by name. The row needed no repin and
#: no re-selection when it cleared: everything else about it was already complete.
PENDING_SERVING_CELLS = ()
#: the TrainTest SI a prompted cell carries as its system turn (registry_adapter.HARNESS_TO_TRAINTEST_SI's values)
PROMPTED_SI_BY_CELL = {CELL_PROMPTED: "safety_refusal", CELL_PROMPTED_CONST: "si_constitution"}
#: sha256 of the EXACT SI text bytes (sisteer.data.instructions, resolved as the driver resolves it —
#: eval.jailbreak.resolve_instruction) — recorded per prompted row; hex16 == the campaign manifests'
#: prompted_prefix_hashes.<harness si>. Mirrored, test-pinned.
SI_SHA256_BY_CELL = {
    CELL_PROMPTED: "ac3a58f61d53cf2412c02b8bb56063b9683a9c1154ef1f0978efc1449a836e08",
    CELL_PROMPTED_CONST: "21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f",
}
FAMILY_BY_MODEL = {"gemma4e4b": "gemma4", "qwen35_4b": "qwen3_5", "glm46vflash": "glm4v",
                   "nemotron3nano4b": "nemotron_h", "llama8b": "llama"}
#: whether this tree's contracts carry a CERTIFIED render fact for the family (contracts.render_fact_certified over
#: FAMILY_RENDER_FACTS; mirrored import-light, test-pinned EQUAL). The MATH-500 arm is thinking-ON on every switch
#: family, so a family with no certified fact cannot carry production traffic — its rows are HELD by name, never
#: launched on a guess (contracts.check_render_fact_certified is the driver-side seam that refuses the same run).
#: `llama` FLIPPED to True on 2026-09-13: the certified no-switch fact for the family landed on main (its template has
#: no thinking switch, so there is ONE render and the assistant header is invariant), which is what the steering llama
#: rows were held on. The hold was a TREE gate, not a design deferral — their admission record was complete throughout —
#: so they became launch rows with no repin and no re-selection, exactly as `steer_hold_reason` said they would. The
#: test pins this mirror EQUAL to contracts.render_fact_certified, so the two cannot disagree.
RENDER_FACT_CERTIFIED_BY_FAMILY = {"gemma4": True, "qwen3_5": True, "glm4v": True, "nemotron_h": True, "llama": True}
#: the served vLLM reasoning parser per model — gemma4 ONLY (vllm_server refuses every other family's)
REASONING_PARSER_BY_MODEL = {"gemma4e4b": "gemma4"}
#: the campaign's per-model gpu_memory_utilization (common.sh f22_resolve_config: glm needs 0.92 for KV coverage)
GPU_MEM_UTIL_BY_MODEL = {"glm46vflash": "0.92"}
GPU_MEM_UTIL_DEFAULT = "0.9"
#: the candidate think-split mode per family (certified only by G-PARSER — Lane A's FAMILY_RENDER_FACTS)
SPLIT_MODE_CANDIDATE = {"gemma4": "record_only", "qwen3_5": "implicit_open", "glm4v": "explicit_tags",
                        "nemotron_h": "implicit_open",
                        # llama has NO thinking switch, so there is one render and nothing to split: the family's
                        # certified fact carries the `none` split mode, mirrored here
                        "llama": "none"}
#: the cluster a row may run on: the Nemotron fused-kernel serving overlay builds and is attested on the PRIMARY cluster
#: alone, every other model may overflow
CLUSTER_CONSTRAINT_PRIMARY = "primary"
CLUSTER_CONSTRAINT_ANY = "any"
#: a row that runs on the OVERFLOW cluster ALONE — every row of the qwen_seedrep wave (MATH-500 runs on cluster B, not
#: cluster A); `plan --cluster primary` refuses such a row by name (a code refusal, not a memory). The Arditi admission
#: entry's own `cluster_constraint: primary` (a launch PLACEMENT fixed for the it-view arm, not a staging limit — the
#: object is staged and verified on both clusters) is overridden to this on the seed rows.
CLUSTER_CONSTRAINT_OVERFLOW = "overflow"
CLUSTER_CONSTRAINT_BY_MODEL = {"nemotron3nano4b": CLUSTER_CONSTRAINT_PRIMARY}
#: a (model, cell) whose OBJECT is staged on the PRIMARY cluster alone (the admission record's per-cluster bank
#: view: the llama AlphaSteer rung has no overflow-cluster copy, verified 2026-09-13) — the row inherits the
#: primary-only constraint whatever its model's default is
CLUSTER_CONSTRAINT_BY_MODEL_CELL = {("llama8b", CELL_ALPHASTEER): CLUSTER_CONSTRAINT_PRIMARY}
#: the two clusters `plan --cluster` names — their argv spellings are the CLI's contract (the tests and the submission
#: lines carry them): the PRIMARY cluster (the shared cohort cache, the Nemotron attestation, every row's
#: `cluster_preference`) and the OVERFLOW cluster (with its own QoS names). Named ONCE, here.
PRIMARY_CLUSTER = "cluster_a"
OVERFLOW_CLUSTER = "cluster_b"
#: the ROLE spellings `plan --cluster` accepts beside the two names above (fix A5: prose and command lines name the
#: role, never the codename; the codenames stay accepted — the tests and the published submission lines carry them,
#: and every row's frozen `cluster_preference` is the primary cluster's name)
CLUSTER_ROLE_PRIMARY = "primary"
CLUSTER_ROLE_OVERFLOW = "overflow"
CLUSTER_BY_ROLE = {CLUSTER_ROLE_PRIMARY: PRIMARY_CLUSTER, CLUSTER_ROLE_OVERFLOW: OVERFLOW_CLUSTER}


def resolve_cluster(name: str) -> str:
    """`--cluster` -> the cluster's name in CLUSTERS: a role (`primary` / `overflow`) or the name itself; anything else
    refuses by name, listing the roles first."""
    if name in CLUSTER_BY_ROLE:
        return CLUSTER_BY_ROLE[name]
    if name in CLUSTER_BY_ROLE.values():
        return name
    raise SystemExit(f"cluster must be one of {sorted(CLUSTER_BY_ROLE)} (or the cluster's own name), got {name!r}")

# --------------------------------------------------------------------------- waves and statuses (plan P2 / P5)
WAVE_PUBLISHED = "published"       # rows 0-23: COMPLETE at tree 55e3cbdd, reused byte-for-byte, never re-run
WAVE_EXT1 = "ext1"                 # rows 24-32: this extension
WAVE_STEER = "steer"               # rows 33-40: the AlphaSteer / Arditi rung (external objects)
WAVE_LLAMA = "llama"               # rows 41-46: the no-switch family's six non-steering arms
WAVE_ITDOC = "itdoc"               # rows 47-52: E3, the instruct-direct batch-64 re-fit LoRA arms on the `it` view (held today)
WAVE_ALLPOS = "allpos"             # row 53: arditi-qwen-all-positions, the all-positions Arditi arm on qwen35_4b's `it` panel view
WAVE_SEEDREP = "qwen_seedrep"      # rows 54-86 (SEEDREP_ROWS, beside the matrix): the batch-64 three-seed Qwen family on the overflow cluster
#: a WAVE's own cluster placement (the seed wave runs on the overflow cluster ALONE, its Arditi rows included — the admission
#: entry's `primary` is a placement fixed for the it-view arm, not a staging limit); every other wave's rows resolve theirs
#: from the admission entry / the it-doc bank / the model (CLUSTER_CONSTRAINT_BY_MODEL)
CLUSTER_CONSTRAINT_BY_WAVE = {WAVE_SEEDREP: CLUSTER_CONSTRAINT_OVERFLOW}
#: the THREE statuses: `launch` = a row whose store is REQUIRED evidence (the published rows included — the analysis'
#: required set is the `launch` rows of the analysed waves); `held` = a preflight hold; `deferred` = a steering-rung
#: stub. "Already complete" is a property of the WAVE (COMPLETED_WAVES), never a status: a published row is excluded
#: from a submission because its wave is complete, and collected by the analysis because its status is `launch`.
STATUS_LAUNCH = "launch"
STATUS_HELD = "held"
STATUS_DEFERRED = "deferred"
STATUSES = (STATUS_LAUNCH, STATUS_HELD, STATUS_DEFERRED)
#: the waves whose legs are COMPLETE and reused byte-for-byte — `plan --wave <one of these>` refuses by name
COMPLETED_WAVES = (WAVE_PUBLISHED,)
#: the WAVE-ISOLATED out-root prefix per wave (F11a: a pull of one subtree never carries another wave's leg; budgets/
#: and nemo_backend/ stay shared under the one wave root)
OUT_ROOT_PREFIX_BY_WAVE = {WAVE_PUBLISHED: "controls", WAVE_EXT1: "controls_ext1", WAVE_STEER: "controls_steer", WAVE_LLAMA: "controls_llama",
                           WAVE_ITDOC: "controls_itdoc", WAVE_ALLPOS: "controls_arditi_allpos",
                           # the seed wave's own subtree (the GSM8K seedrep wave's spelling), under the lane's ONE wave root
                           WAVE_SEEDREP: "controls_qwen_seedrep"}
VIEW_MAIN = "main"                 # every base-lane object is served from the `main` bank view

# --------------------------------------------------------------------------- sampling
SAMPLING_THINKING_CARD = "thinking_card"
SAMPLING_GENERAL = "general"
SAMPLING_BATTERY = "battery"
SAMPLING_MODES = (SAMPLING_THINKING_CARD, SAMPLING_GENERAL, SAMPLING_BATTERY)
#: which sampling modes each model runs (gemma: primary only — its battery preset equals its primary)
SAMPLING_MODES_BY_MODEL = {
    "gemma4e4b": (SAMPLING_THINKING_CARD,),
    "qwen35_4b": (SAMPLING_THINKING_CARD, SAMPLING_GENERAL, SAMPLING_BATTERY),
    "glm46vflash": (SAMPLING_THINKING_CARD, SAMPLING_BATTERY),
    "nemotron3nano4b": (SAMPLING_THINKING_CARD, SAMPLING_BATTERY),
}
_BATTERY_SRC = ("the LIVE sampled-battery preset: configs/sdfbase_sampled_decode.yaml on branch sampled-rerun "
                "@5a50e47b (sha256 7a065e58fb86217033e27a020ceb207027b447f43a500a094cf12c5ae7becb36), "
                "scripts/sdfbase/full22/sampled/DESIGN.md 'Verified sampler presets' row ")
#: THE LITERAL ROWS, with provenance (unhashed prose beside the numbers). Optional keys ABSENT mean the request
#: carries no such key (vLLM's defaults apply) — a row carries EXACTLY the keys its source names.
SAMPLING_ROWS: dict[tuple, dict] = {
    ("gemma4e4b", SAMPLING_THINKING_CARD): {
        "temperature": 1.0, "top_p": 0.95, "top_k": 64,
        "source": ("generation_config.json of google/gemma-4-E4B-it @ee0ef602 (do_sample true; vendor default, not "
                   "reasoning-tuned; no penalty term) — re-read 2026-09-09; identical to the sampled battery's preset, so "
                   "gemma has no battery row")},
    ("qwen35_4b", SAMPLING_THINKING_CARD): {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
        "source": ("Qwen/Qwen3.5-4B @851bf6e8 ships no generation_config.json (.no_exist marker); the Qwen3.5 model "
                   "card's THINKING-mode 'precise' row as reported to the project = the Qwen3 card's thinking-mode "
                   "guidance (T 0.6 / top_p 0.95 / top_k 20 / min_p 0; penalties only against endless repetition at a "
                   "stated cost) — the PRIMARY row: mathematics legitimately repeats tokens")},
    ("glm46vflash", SAMPLING_THINKING_CARD): {
        "temperature": 0.8, "top_p": 0.6, "top_k": 2,
        "source": ("generation_config.json of zai-org/GLM-4.6V-Flash @411bb4d7 (do_sample true; no penalty term in the "
                   "file) — read 2026-09-09 over ssh from the cohort cache")},
    ("nemotron3nano4b", SAMPLING_THINKING_CARD): {
        "temperature": 1.0, "top_p": 0.95,
        "source": ("generation_config.json of nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 @dfaf35de (temperature 1.0, top_p "
                   "0.95; no top_k, no do_sample key, no penalty) AND the card's explicit reasoning-task recommendation "
                   "'temperature=1.0 and top_p=0.95 are recommended for reasoning tasks' — read 2026-09-09")},
    ("qwen35_4b", SAMPLING_GENERAL): {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
        "source": ("the Qwen3.5 model card's THINKING-mode 'general' row as reported to the project: T 1.0 / top_p 0.95 / "
                   "top_k 20 / min_p 0 / presence_penalty 1.5 / repetition_penalty 1.0 (the card's stated no-op) — the "
                   "ONLY primary-or-general lane whose row carries a penalty (stated, not smoothed); a SENSITIVITY, "
                   "never pooled with the primary")},
    ("qwen35_4b", SAMPLING_BATTERY): {
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
        "source": (_BATTERY_SRC + "qwen35_4b — the card's INSTRUCT (non-thinking) general row; by its own provenance a "
                   "non-thinking row, run here under thinking ON as the sensitivity that honours 'run at the battery's "
                   "settings'")},
    ("nemotron3nano4b", SAMPLING_BATTERY): {
        "temperature": 1.0, "top_p": 1.0,
        "source": (_BATTERY_SRC + "nemotron3nano4b — the general-chat recommendation top_p 1.0 as reported to the "
                   "project (the shipped file's 0.95 recorded beside it in the battery's yaml as "
                   "shipped_generation_config)")},
    ("glm46vflash", SAMPLING_BATTERY): {
        "temperature": 0.8, "top_p": 0.6, "top_k": 2, "repetition_penalty": 1.1,
        "source": (_BATTERY_SRC + "glm46vflash — the card's 'Evaluation Settings' (vLLM): adds repetition_penalty 1.1 to "
                   "the shipped file's three values; the vendor's 16K generation length is not adopted AS SUCH — "
                   "the cap is the stated bound, 16,384 since a later ruling, which happens to equal it")},
}
#: the NO-SWITCH model's sampling MODE: llama8b has no thinking mode, so it runs its SHIPPED
#: `card` row — the driver's `--sampling card` resolves contracts.SAMPLING_TABLE_BY_MODE[card] = SAMPLING_BY_MODEL_KEY,
#: which carries llama8b; `--sampling thinking_card` on llama8b is REFUSED by contracts.sampling_for (no llama row in
#: THINKING_SAMPLING_BY_MODEL_KEY — a "thinking card" for a model with no thinking mode is a misnomer, and
#: tests/test_tinycap_render_facts.py pins the refusal), so a llama row filed under `thinking_card` — rows 39-40 as
#: first landed — could not have passed the driver door. The row's ARM is `thinking_mode` (`not_available`); its MODE is
#: `card`. Mirrored from contracts.SAMPLING_BY_MODEL_KEY["llama8b"] and test-pinned equal, never re-typed.
SAMPLING_CARD = "card"
SAMPLING_ROWS[("llama8b", SAMPLING_CARD)] = {
    "temperature": 0.6, "top_p": 0.9,
    "source": ("generation_config.json of NousResearch/Meta-Llama-3.1-8B-Instruct @d10aef79 (do_sample true; "
               "temperature 0.6, top_p 0.9; no top_k, no penalty) — contracts.SAMPLING_BY_MODEL_KEY['llama8b'], "
               "mirrored here and test-pinned; the model has NO thinking switch, so this ONE row serves its "
               "`not_available` arm (W4 / E2 / the Llama lane's P6)")}
SAMPLING_MODES_BY_MODEL["llama8b"] = (SAMPLING_CARD,)


def sampling_mode_for(model_key: str) -> str:
    """The sampling MODE a model's primary rows run under: the no-switch family's shipped `card` row; every switch family's
    `thinking_card` row (the vendor's thinking-mode guidance)."""
    return SAMPLING_CARD if THINKING_MODE_BY_FAMILY.get(FAMILY_BY_MODEL[model_key]) == THINKING_NOT_AVAILABLE else SAMPLING_THINKING_CARD

SAMPLING_PARAM_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")


def sampling_params(sampling: dict) -> dict:
    """The NUMERIC projection of a matrix sampling dict: exactly the SAMPLING_PARAM_KEYS it carries, never the
    ``source`` prose (the recorder's --expect, the preflight's shared-field compare and M5T_SAMPLING_PARAMS all read
    THIS)."""
    return {k: sampling[k] for k in SAMPLING_PARAM_KEYS if k in sampling}


# --------------------------------------------------------------------------- the run's literals (mirrored from contracts; test-pinned)
CONTROL = "math500"
THINKING_MODE = "on"                       # the literal the authorisation compares (never per-task); every row's thinking_mode in E1
#: the NO-SWITCH arm (contracts.THINKING_NOT_AVAILABLE; mirrored, test-pinned where present): a family whose chat
#: template has no thinking switch runs ONE render — never OFF, never ON. The steer wave's llama rows carry it
THINKING_NOT_AVAILABLE = "not_available"
#: family -> the thinking arm its rows run (absent = THINKING_MODE, `on`)
THINKING_MODE_BY_FAMILY = {"llama": THINKING_NOT_AVAILABLE}
CAP = 16384                                # one pass at 16,384 (the standing truncation rule's ceiling) == contracts.MATH500_MAX_TOKENS_DEFAULT (test-pinned)
RETRY_MAX_TOKENS = 0                       # the retry pass retired
SAMPLES = 1
IDENTITY_VERSION = 2
UPSTREAM_TIMEOUT_S = 7200                  # contracts.PROXY_UPSTREAM_TIMEOUT_S
#: MATH-500's geometry is PER PROMPT CLASS (contracts.MATH500_PROMPT_CLASS_OF_SI / MATH500_GEOMETRY_KEY_OF_CLASS,
#: mirrored here and test-pinned): no system turn or the ~90-token safety SI -> the short class; the ~6k-token
#: constitution SI -> the long-SI class. Each entry is sized to hold EVERY prompt of its class at the 16,384 cap
#: (the driver's assert_pass_fits_window refuses a clamped cap-bound sample), S from the model's budget.
PROMPT_CLASS_SHORT = "short"
PROMPT_CLASS_LONG_SI = "long_si"
PROMPT_CLASS_OF_SI = {"safety_refusal": PROMPT_CLASS_SHORT, "si_constitution": PROMPT_CLASS_LONG_SI}
GEOMETRY_BY_CLASS = {
    PROMPT_CLASS_SHORT: ("capability:gen_long_math", 17664),          # prompt max 1,041 + 16,384 -> 17,664; S 33 / 42 / 44 / 44 (qwen: MEASURED at this window 2026-09-13; the wave ran at the provisional 20)
    PROMPT_CLASS_LONG_SI: ("capability:gen_long_math_longsi", 24064),  # class max 7,622 + 16,384 -> 24,064; S 24 / 29 / 32 / 32
}
GEOMETRY_EVAL, GEOMETRY_L = GEOMETRY_BY_CLASS[PROMPT_CLASS_SHORT]     # the published run's ONE literal (rows 0-23 resolve here)
LENS_SALT_POLICY = "attested"
INSPECT_TASK = "math"
MAX_TOKENS_BY_TASK = f"{INSPECT_TASK}={CAP}"
#: driver flag spellings (contracts.FLAG_*; pinned equal by test)
FLAG_THINKING = "--thinking"
FLAG_SAMPLING = "--sampling"
FLAG_SAMPLES = "--samples"
FLAG_LEDGER_URL = "--ledger-url"
FLAG_OOS_PROMPT_BUDGET = "--oos-prompt-budget"
FLAG_ALLOW_OOS_THINKING = "--allow-out-of-support-thinking"
FLAG_EXPECT_OBJECT_SHA256 = "--expect-object-sha256"
#: the driver's liveness-gate flag and the value every STEER row passes — the logprob statistic rules the positive
#: control on an alphasteer / static_r cell (contracts.LOGPROB_LIVENESS_KINDS; the driver scopes it by kind), the text
#: bites are reported beside it. A published / extension row passes no such flag (its argv is byte-identical; pinned by
#: the sbatch test).
FLAG_LENS_LIVENESS = "--lens-liveness"                          # contracts.FLAG_LENS_LIVENESS
LENS_LIVENESS_STEER = "logprob"                                 # contracts.LENS_LIVENESS_LOGPROB
FLAG_MATH500_RETRY_MAX_TOKENS = "--math500-retry-max-tokens"
FLAG_MAX_TOKENS_BY_TASK = "--max-tokens-by-task"
FLAG_TINYCAP_IDENTITY_VERSION = "--tinycap-identity-version"
FLAG_UPSTREAM_TIMEOUT_S = "--upstream-timeout-s"
FLAG_LENS_SALT_POLICY = "--lens-salt-policy"
FLAG_REASONING_PARSER = "--reasoning-parser"
FLAG_CELL = "--cell"
FLAG_ENABLE_LOG_REQUESTS = "--enable-log-requests"           # the standup's opt-in (mt_serving_up.launch_cmd_extras)
#: the file names the sbatch and the driver agree on
LEDGER_FILENAME = "usage_ledger.jsonl"
LEDGER_ENDPOINT_FILENAME = "ledger_endpoint.json"
OOS_PROMPT_BUDGET_FILENAME = "oos_prompt_budget.json"
REQUEST_EVIDENCE_FILENAME = "request_evidence.jsonl"
#: the ENGINE-RESOLVED SAMPLER attestation (the per-kind rule): after the driver the launcher runs
#: scripts/sdfbase/record_resolved_sampler.py over the standup's vLLM log into the out
#: root, with the row's PINNED canonical sampler as `--expect k=v` per key of the NUMERIC projection (the render probe's
#: reference and the client class's — never the driver's own declaration), `--client-cap` = the row's CAP, and the PROBE
#: class partitioned BY KIND (`--probe-expect kinds`): the driver's own tag per probe attempt — `X-Tinycap-Task:
#: probe/<kind>` + the wire body's max_tokens, written to `<out_root>/request_evidence.jsonl`
#: (contracts.request_evidence_record) — read back through `--probe-tags` (ONE tag source per leg: the request evidence,
#: never also the usage ledger, whose rows would double the counts), each kind against its own pinned reference (render ->
#: the card row = --expect; bites, lens_on_bite -> greedy), the shape cross-checked against the recorder's
#: PROBE_KIND_TABLE. A probe matching no kind is UNCLASSIFIED — a third outcome that fails the probe class; a finding, never
#: a declaration to widen. NO `--probe-cap` is passed: under `kinds` the recorder's default is every cap its kind table
#: pins — RECORDER_PROBE_CAPS mirrors them (pinned equal by test) for the ONE check the launcher owns, that the row's
#: client cap collides with no kind cap (the recorder refuses a leg whose classes are indistinguishable; 16,384 collides
#: with none). The caps, read from the published ledgers: probe/bites 48 (the cell + its
#: none__lens0 twin), probe/lens_on_bite 256, probe/render 512 and the driver's ON-arm render retry 2,048; plus the
#: standup's positive_control at 64 (smoke_positive_control.py) — the standup's load_check
#: has no pinned cap (it fires at the RUN's cap: on this lane the controls cap 512, which the list already carries)
#: (served_capability_run.RENDER_PROBE_MAX_TOKENS + its retry, LENS_ON_BITE_MAX_TOKENS); and, since the steer rows'
#: logprob liveness gate (FLAG_LENS_LIVENESS), the driver's probe/logprob_bites at 1 (contracts.LOGPROB_BITES_MAX_TOKENS).
RESOLVED_SAMPLER_FILENAME = "resolved_sampler.json"
RESOLVED_SAMPLER_SCRIPT_REL = "scripts/sdfbase/record_resolved_sampler.py"
RECORDER_PROBE_EXPECT = "kinds"
RECORDER_PROBE_TAGS_FILENAME = REQUEST_EVIDENCE_FILENAME
#: the SECOND tag source: the STANDUP's own load-check sidecar in the serving dir
#: (`harness_evidence.evidence_path_for(<out>/load_check.json)`). The load check fires its probes BEFORE the driver
#: exists, so with only the driver's evidence every one of its requests is untagged and the render kind reads
#: UNCLASSIFIED at cap 512. Two emitters, disjoint request sets — not the double-counting the recorder warns about
#: (that is the evidence file AND the ledger for the SAME requests).
RECORDER_LOAD_CHECK_TAGS_FILENAME = "load_check.request_evidence.jsonl"
RECORDER_PROBE_CAPS = (1, 48, 64, 256, 512, 2048)
#: the launcher's ROW-IDENTITY stamp (decisions-impl-laneA A2): written into the out root after the driver
ROW_IDENTITY_FILENAME = "row_identity.json"
#: the Inspect provider's env names (contracts.INSPECT_BASE_URL_ENV / INSPECT_API_KEY_ENV)
INSPECT_BASE_URL_ENV = "TINYCAP_BASE_URL"
INSPECT_API_KEY_ENV = "TINYCAP_API_KEY"
#: the new wave root (never the full22 wave's)
WAVE_ROOT_REL = "exp/math500_thinking_on_full22"
#: --time classes (~2x the upper estimate; qwen's provisional S = 33 re-costed), RE-COSTED
#: for MATH-500's own window: the classes are ~2x a DECODE-BOUND upper estimate, decode wall
#: = N * mean_completion / (per-seq rate x S) and S = budget // L, so wall scales with L — x(17,664 / 10,752) = 1.643 on
#: the old 3 / 6 / 8 h -> 4.93 / 9.86 / 13.1 h, rounded to the hour. The fixed standup (12-16 min) is over-scaled by
#: ~8-10 min per class, which covers the doubled cap-tail (the last cap-bound draw: 16,384 / 12 tok/s = 23 min lens,
#: 11 min base — +11 / +5.5 min over the 8,192 tail). Concurrency: gemma 54 -> 33, qwen 33 -> 20, glm / nemotron 73 -> 44.
TIME_BASE = "0-05:00:00"
TIME_OBJECT = "0-10:00:00"
TIME_OBJECT_GLM = "0-13:00:00"
#: the EXTENSION rows' one class: every extension leg's expected wall is
#: <= 2.3 h (gemma LoRA ~0.55 h, qwen LoRA ~1.7 h at the published S 20 — less at the swept 40 — qwen constitution <= 2.3 h
#: at S 14 — less at 29; nemotron ~0.27 h), so 6 h is ~2.6x the longest and one class serves the whole wave
TIME_EXT = "0-06:00:00"
#: the LLAMA rows' one class: no thinking channel, so completions are ~300-900 tokens — ~5-8 min decode
#: + 5-7 min standup per leg at S 26 / 19; 1.5 h is ~2x the constitution row's upper estimate (its 6,9xx-token prefill x 500)
TIME_LLAMA = "0-01:30:00"
#: the SEED WAVE's one class (the required window: > 30 min, < 5 h per row): the measured native
#: `arditi_allpos` leg = 33 min / 500 items; the lens-era Qwen legs 0.9-1.6 h; a loop-heavy LoRA arm at ~50 % at-cap ~1.5 h at
#: the native S; + <= 20 min venv bootstrap on a fresh node -> 4 h is >= 2x the worst case and inside that window
TIME_SEEDREP = "0-04:00:00"
#: the --time class of a NON-PUBLISHED wave's rows (TIME_EXT, the extension class, for every wave not named here) — a fact of the wave, like
#: its out-root prefix, so a row built by any path carries its wave's class
TIME_BY_WAVE = {WAVE_LLAMA: TIME_LLAMA, WAVE_SEEDREP: TIME_SEEDREP}
#: the expected stamp triples: (cache_salt_regime, resumed_prefill_policy, render_regime) — the thinking-ON arm's
STAMPS_FITTED = ("cell-stable", "always", "thinking_on")
STAMPS_TWIN = ("cell-stable", "always", "thinking_on")        # the twin's attested policy, expected `always`
STAMPS_BASE = ("cell-stable", "not-consulted", "thinking_on")
#: the render regime the proxy stamps per ARM (math500-steer): the thinking-ON arm's `thinking_on`; the NO-SWITCH arm's
#: `unspecified` (no enable_thinking on the wire — the llama rows). A stdlib mirror of `contracts.RENDER_REGIME_BY_ARM`
#: restricted to the two MATH-500 arms, test-pinned equal (this file imports nothing outside the stdlib at import time).
RENDER_REGIME_BY_ARM = {THINKING_MODE: "thinking_on", THINKING_NOT_AVAILABLE: "unspecified"}


def stamps_for(fitted: bool, arm: str) -> tuple[str, str, str]:
    """A row's expected stamp triple by its ARM: the salt / policy pair of its class (fitted: cell-stable / always; base:
    cell-stable / not-consulted) and the arm's render regime — the llama rows (`not_available`) stamp `unspecified`, every
    other row `thinking_on` byte for byte as before. An arm outside the table refuses by name."""
    if arm not in RENDER_REGIME_BY_ARM:
        raise ValueError(f"stamps_for: arm {arm!r} is not one of {sorted(RENDER_REGIME_BY_ARM)}")
    salt, policy, _ = STAMPS_FITTED if fitted else STAMPS_BASE
    return (salt, policy, RENDER_REGIME_BY_ARM[arm])


def twin_stamps_for(arm: str) -> tuple[str, str, str]:
    """The twin's expected triple by the ARM (its attested policy `always`, the arm's render)."""
    if arm not in RENDER_REGIME_BY_ARM:
        raise ValueError(f"twin_stamps_for: arm {arm!r} is not one of {sorted(RENDER_REGIME_BY_ARM)}")
    salt, policy, _ = STAMPS_TWIN
    return (salt, policy, RENDER_REGIME_BY_ARM[arm])


def prompted_si(cell: str) -> str | None:
    """The SI a prompted cell carries (None for the bare and object cells). A `prompted__` id outside the
    served vocabulary refuses BY NAME here — `prompted__si_constitution_full` first."""
    if not cell.startswith("prompted__"):
        return None
    if cell == CELL_PROMPTED_CONST_FULL_REFUSED:
        raise SystemExit(f"{cell!r} is not a legal cell of this experiment: the constitution arm is {CELL_PROMPTED_CONST!r} "
                         "(the SHORT constitution SI; harness prompted__constitution) — the _full variant is another lane's "
                         "prompted cell (harness prompted__constitution_full) and is refused by name here (D4)")
    try:
        return PROMPTED_SI_BY_CELL[cell]
    except KeyError:
        raise SystemExit(f"{cell!r}: no TrainTest SI mirrored for this prompted cell (PROMPTED_SI_BY_CELL) — "
                         "a prompted cell outside the served vocabulary is never launched") from None


def prompt_class(cell: str) -> str:
    """The MATH-500 prompt class of a cell (contracts.math500_prompt_class_of_si, mirrored): no system turn -> short."""
    si = prompted_si(cell)
    return PROMPT_CLASS_SHORT if si is None else PROMPT_CLASS_OF_SI[si]


def geometry_for(cell: str) -> tuple[str, int]:
    """(geometry eval key, L) the standup serves THIS cell at — the class's measured entry (test-pinned equal to
    contracts.math500_geometry_key + geometry.geometry_for(...).max_model_len)."""
    return GEOMETRY_BY_CLASS[prompt_class(cell)]


def budget_class(cell: str) -> str:
    """The prompt-budget artifact CLASS a cell shares (per (model, cell class) — the object cells render the
    same messages as `none`; a prompted cell adds ITS system turn, so each prompted SI is its own class)."""
    return cell if prompted_si(cell) is not None else CELL_NONE


#: the launcher's readiness wait for the usage ledger (tinycap_served.LEDGER_WAIT_S)
LEDGER_WAIT_S = 180
#: the run-level literals every job shares, as shell assignments (row_sh appends them so the sbatch has ONE source).
#: The GEOMETRY is PER ROW since the extension (M5T_GEOMETRY_EVAL / M5T_GEOMETRY_L ride the row, not this list).
RUN_LITERALS_SH: tuple = (
    ("M5T_CONTROL", CONTROL),
    ("M5T_CAP", CAP),
    ("M5T_RETRY", RETRY_MAX_TOKENS),
    ("M5T_SAMPLES", SAMPLES),
    ("M5T_IDENTITY_VERSION", IDENTITY_VERSION),
    ("M5T_UPSTREAM_TIMEOUT_S", UPSTREAM_TIMEOUT_S),
    ("M5T_LENS_SALT_POLICY", LENS_SALT_POLICY),
    ("M5T_MAX_TOKENS_BY_TASK", MAX_TOKENS_BY_TASK),
    ("M5T_LEDGER_WAIT_S", LEDGER_WAIT_S),
    ("M5T_INSPECT_BASE_URL_ENV", INSPECT_BASE_URL_ENV),
    ("M5T_INSPECT_API_KEY_ENV", INSPECT_API_KEY_ENV),
    ("M5T_LEDGER_FILENAME", LEDGER_FILENAME),
    ("M5T_LEDGER_ENDPOINT_FILENAME", LEDGER_ENDPOINT_FILENAME),
    ("M5T_BUDGET_FILENAME", OOS_PROMPT_BUDGET_FILENAME),
    ("M5T_REQUEST_EVIDENCE_FILENAME", REQUEST_EVIDENCE_FILENAME),
    ("M5T_RESOLVED_SAMPLER_FILENAME", RESOLVED_SAMPLER_FILENAME),
    ("M5T_RESOLVED_SAMPLER_SCRIPT", RESOLVED_SAMPLER_SCRIPT_REL),
    ("M5T_RECORDER_PROBE_CAPS", " ".join(str(c) for c in RECORDER_PROBE_CAPS)),
    ("M5T_RECORDER_PROBE_EXPECT", RECORDER_PROBE_EXPECT),
    ("M5T_RECORDER_PROBE_TAGS_FILENAME", RECORDER_PROBE_TAGS_FILENAME),
    ("M5T_RECORDER_LOAD_CHECK_TAGS_FILENAME", RECORDER_LOAD_CHECK_TAGS_FILENAME),
    ("M5T_ROW_IDENTITY_FILENAME", ROW_IDENTITY_FILENAME),
)


def cell_slug(cell: str) -> str:
    """The out-root path segment: `__` -> `--` (the store-key convention of cell_controls.store_key_token, mirrored
    here because this file stays stdlib-only)."""
    return cell.replace("__", "--")


def cell_short(cell: str) -> str:
    return {CELL_NONE: "none", CELL_PROMPTED: "prompted", CELL_OBJECT: "object",
            CELL_PROMPTED_CONST: "consi", CELL_LORA_CTL2: "ctl2lora", CELL_LORA_MIX: "mixlora",
            CELL_ALPHASTEER: "alphasteer", CELL_ARDITI: "arditi",
            # never abbreviated to anything an `arditi` row could wear: the two are different interventions
            CELL_ARDITI_ALLPOS: "arditi_allpos",
            # the seed wave's PSR cells: constitution ReLU, control ReLU, the composed difference ReLU
            CELL_PSR_MIX: "mixpsr", CELL_PSR_CTL2: "ctl2psr", CELL_PAIRED: "dpsr"}[cell]


def route_kind(cell: str) -> str:
    """By the cell GRAMMAR (contracts.parse_launcher_cell's kinds): `none` -> base, `prompted__<si>` -> prompted, else
    lens (a bare Class B id — `alphasteer`, `arditi` — is a lens route like any object cell)."""
    if cell == CELL_NONE:
        return "base"
    if cell.startswith("prompted__"):
        prompted_si(cell)          # refuses the _full spelling by name
        return "prompted"
    return "lens"


#: the STEERING ADMISSION RECORD: the ONE file that says which EXTERNAL object each steering row
#: serves. Keyed by MODEL KEY then CELL ID — never by model name alone (one model name covered three
#: artifacts) — and each entry names its LANE ROOT and its per-cluster bank view beside the 64-hex object sha256.
STEER_ADMISSION_PATH = REPO / "configs" / "sdfbase" / "steer_admission.json"
#: the keys a steering row's admission entry MUST carry for the row to be a launch row (a missing one HOLDS it by name)
STEER_ADMISSION_REQUIRED = ("object_sha256", "view", "bank_cell_dir", "bank_view_rel", "lane_root", "selection_rule",
                            "selection_label", "grid_limited", "cluster_constraint")


def is_object_sha(s: object) -> bool:
    """A 64-hex object.pt digest — the OBJECT's sha256, never a directory digest; :func:`is_full_sha` is the 40-hex tree twin."""
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def load_steer_admission(path: Path = STEER_ADMISSION_PATH) -> dict[tuple[str, str], dict]:
    """``{(model key, cell): entry}`` from the admission record — every entry validated for the fields a row reads
    (:data:`STEER_ADMISSION_REQUIRED`, a 64-hex object sha256, the cell id agreeing with its key). A malformed entry
    refuses BY NAME here, at import, rather than at the batch boundary of a job that already holds a GPU."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict] = {}
    for key, by_cell in (doc.get("entries") or {}).items():
        for cell, entry in by_cell.items():
            where = f"{path.name}: entries.{key}.{cell}"
            missing = [f for f in STEER_ADMISSION_REQUIRED if entry.get(f) in (None, "")]
            if missing:
                raise SystemExit(f"{where}: the admission record lacks {missing} — a steering row is never launched on an "
                                 "incomplete record (plan P8)")
            sha = entry["object_sha256"]
            if not is_object_sha(sha):
                raise SystemExit(f"{where}: object_sha256 {sha!r} is not 64-hex — the pin is the OBJECT's digest, never a "
                                 "directory digest (D5)")
            if entry.get("cell") != cell or entry.get("model_key") != key:
                raise SystemExit(f"{where}: the entry names ({entry.get('model_key')!r}, {entry.get('cell')!r}) — an entry is "
                                 "keyed by model key AND cell id, and must say so (D45)")
            out[(key, cell)] = entry
    # the RULED AlphaSteer operating point: each AlphaSteer entry must bind to the oracle's record
    # (scripts/alphasteer_f22/operating_points.py) where that record is present on this machine — a stale pin refuses by
    # name at import, never hashes clean and serves a superseded dose (scripts/sdfbase/steer_admission_ruling.py; skipped on
    # a machine without the ladder record)
    _ruling_spec = importlib.util.spec_from_file_location("steer_admission_ruling", REPO / "scripts" / "sdfbase" / "steer_admission_ruling.py")
    _ruling = importlib.util.module_from_spec(_ruling_spec); _ruling_spec.loader.exec_module(_ruling)  # type: ignore[union-attr]
    _ruling.check_entries(out, where=Path(path).name)
    return out


def load_pins(path: Path = PINS_PATH) -> dict[tuple[str, str], str]:
    """(model key, staged id) -> the pinned object.pt sha256 (configs/sdfbase_full22_pins.json) for EVERY staged id the
    matrix rows name (STAGED_ID_BY_CELL), on every model."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for key in MODEL_KEYS + LLAMA_MODELS:
        for staged in STAGED_ID_BY_CELL.values():
            entry = doc[key][staged]
            sha = entry["object_sha256"] if isinstance(entry, dict) else entry
            if not isinstance(sha, str) or len(sha) != 64:
                raise SystemExit(f"pins: {key}/{staged} has no 64-hex object_sha256")
            out[(key, staged)] = sha
    return out


def tuple_name(model_key: str, sampling_mode: str, cell: str = CELL_OBJECT) -> str:
    """The authorised out-of-support experiment's name for a FITTED row:
    `math500_thinking_on_full22_<key>` for the published object cell's primary row, `…_<key>_<mode>` for its sensitivity
    rows, `…_<key>_lora2` for the two extra LoRA cells (ONE new tuple per model, both
    cells, the primary row only). A NO-SWITCH family's fitted row is IN SUPPORT (one render, nothing to authorise): None,
    whatever the cell (the GSM8K sibling launcher's rule)."""
    if THINKING_MODE_BY_FAMILY.get(FAMILY_BY_MODEL[model_key]) == THINKING_NOT_AVAILABLE:
        if cell not in ALL_FITTED_CELLS:
            raise SystemExit(f"tuple_name: {cell!r} is not a fitted cell of this matrix ({ALL_FITTED_CELLS})")
        return None
    if cell == CELL_OBJECT:
        return f"math500_thinking_on_full22_{model_key}" + ("" if sampling_mode == SAMPLING_THINKING_CARD else f"_{sampling_mode}")
    if cell in (CELL_LORA_MIX, CELL_LORA_CTL2):
        if sampling_mode != SAMPLING_THINKING_CARD:
            raise SystemExit(f"tuple_name: {cell!r} is authorised under {SAMPLING_THINKING_CARD!r} only (no sensitivity "
                             f"rows in the extension), not {sampling_mode!r}")
        return f"math500_thinking_on_full22_{model_key}_lora2"
    if cell == CELL_ARDITI_ALLPOS:
        # arditi-qwen-all-positions: its OWN tuple, never the steer rung's `_steer`. An OOS tuple is an AUTHORISATION, not a
        # label: `_steer` declares "prefill-only all-position objects … the highest rung within the 4 pp over-refusal
        # budget", which is false of this arm three times over (BOTH phases; a raw-coefficient dose on a vector_all
        # object; a DECLARED OVERRIDE over a 10 pp budget, not the rule's return) — and widening `_steer`'s cells
        # would move an experiment that three finished MATH-500 legs and six GSM8K tuples record BY NAME.
        if model_key not in ALLPOS_MODELS:
            raise SystemExit(f"tuple_name: {cell!r} runs on {list(ALLPOS_MODELS)} only (the admission record admits no other "
                             f"target's all-positions direction), not {model_key!r}")
        if sampling_mode != SAMPLING_THINKING_CARD:
            raise SystemExit(f"tuple_name: {cell!r} is authorised under {SAMPLING_THINKING_CARD!r} only (this arm has no "
                             f"sensitivity rows), not {sampling_mode!r}")
        return f"math500_thinking_on_full22_{model_key}_steer_allpos"
    if cell in STEER_CELLS:
        # the OOS tuple authorises a fitted intervention served under a chat render its fit never saw — which only
        # arises on a family that HAS a thinking switch (a no-switch family's steering row returned None above)
        if sampling_mode != SAMPLING_THINKING_CARD:
            raise SystemExit(f"tuple_name: {cell!r} is authorised under {SAMPLING_THINKING_CARD!r} only (the W4 rung has no "
                             f"sensitivity rows), not {sampling_mode!r}")
        return f"math500_thinking_on_full22_{model_key}_steer"
    raise SystemExit(f"tuple_name: {cell!r} is not a fitted cell of this matrix ({FITTED_CELLS})")


#: the 23 keys the published tree's rows carried (tests/fixtures/math500_matrix_55e3cbdd.json)
PUBLISHED_KEYS = frozenset({
    "index", "job_id", "model_key", "cell", "cell_slug", "harness_cell", "sampling_mode", "sampling", "fitted", "tuple_name",
    "route_kind", "served_positions_expected", "expected_stamps", "twin_expected_stamps", "out_root", "time_limit",
    "cluster_preference", "family", "split_mode_candidate", "reasoning_parser", "gpu_mem_util", "object_sha256",
    "smoke_diagnostic_out"})
#: the keys EVERY row carries beyond the published tree's 23 (decisions-impl-laneA A5): the fixture pins the published
#: keys byte-identical on rows 0-23 and this set as the ONLY additions
EXTENSION_KEYS = frozenset({"wave", "status", "thinking_mode", "view", "geometry_eval", "geometry_L", "budget_class",
                            "staged_id", "si_sha256", "log_requests", "cluster_constraint", "extension"})
#: the keys a STEERING row carries beyond those: where its object comes from and under what selection.
#: `object_source: external` is what tells every reader the pin is the ADMISSION RECORD's, not the full22 file's.
OBJECT_SOURCE_EXTERNAL = "external"          # the one spelling (the P11 preflight's row contract reads it too)
#: THE SERVING PATH an external-object row PINS — "THE PIN FOLLOWS THE STORE" (the GSM8K launcher's `serving_path_pin`, mirrored; the rule
#: itself lives in full22/steer_store_pin.py, ONE stdlib module both launchers read). The launcher's F22_SERVING_PATH defaults to
#: `lora_native`, and a native standup SKIPS BY NAME the object kinds it cannot serve, so a single-cell job whose one object cell is
#: skipped refuses there (array 506737, 2026-09-14). AlphaSteer (`alphasteer`) has no native form: its row pins the lens BY KIND
#: (`kind:alphasteer`). Arditi (`static_r`) HAS one (the residual slot), so its row pins the lens only where its steer store ON RECORD is
#: a lens store (`store:lens` — serving_identity proxy_vllm+lens+interleaved, the served-steer stamp class; every store on record today,
#: and a finished store is never re-run), is UNPINNED where no store exists (`unpinned:no-store`: the kind-aware default then selects
#: native at f22_init) or where the store is native-stamped (`store:native`), and a store carrying BOTH engines' stamps refuses by name.
#: "On record" is TWO sources read together: the DISK — the row's OWN out root under the wave root of the cluster the job runs on — and
#: the PINS — the store-pins record's engine for the row's admitted leg (STORE_PINS_PATHS, travelling with the launch tree, so a store on
#: the OTHER cluster still counts); the reason names the source (`store:lens(disk+pins)`). Read by the sbatch BEFORE f22_init
#: (m5t_select_serving_path, `row --serving-path --wave-root`), printed by the dry run (SERVING_PATH: … reason=…); `plan` prefixes only a
#: submission whose every row pins the lens (by kind or by the record — the grading host reads no cluster disk).
SERVING_PATH_LENS = SP.SERVING_PATH_LENS
#: the PINS source of "on record": the repo's admitted store-pins record (the analysis' --store-pins admission), whose
#: `_meta.serving_engine_by_leg` block carries each admitted steer leg's engine, derived from the pulled store by
#: scripts/sdfbase/full22/stamp_store_pins_engines.py. It travels with the launch tree, so a row whose store lives on the OTHER cluster
#: still follows it here.
STORE_PINS_PATHS: tuple[Path, ...] = (REPO / "configs" / "sdfbase" / "math500" / "store_pins_default.json",
                                      # the seed wave's record (empty until its legs complete; pin_m5t_seedrep.py fills it and stamps the
                                      # engines from the pulled stores) — the `@s<s>` Arditi legs' engine is read from HERE once stamped
                                      REPO / "configs" / "sdfbase" / "math500" / "store_pins_qwen_seedrep.json")
LEG_KEY_VIEW_SEP = "@"                     # == tinycap_math500_thinking.LEG_KEY_VIEW_SEP (pinned equal by test)


def leg_keys(row: dict) -> list[str]:
    """The row's admitted-leg key in the store-pins record (`<model_key>/<cell>/<sampling_mode>[@<view>]` on a non-main KEY view — the
    analysis' row_leg_key, mirrored; pinned equal by test): an external row keys on the main view (its `view` names the object bank)."""
    view = key_view(row)
    key = f"{row['model_key']}/{row['cell']}/{row['sampling_mode']}"
    return [key if view == VIEW_MAIN else f"{key}{LEG_KEY_VIEW_SEP}{view}"]


def store_roots_rel(row: dict) -> list[str]:
    """The out roots (relative to the wave root) that are THIS ROW's store on record: the row's out root (this launcher has no rung sibling)."""
    return [str(row["out_root"])]


def serving_path_pin(row: dict, *, wave_root: "str | Path | None" = None,
                     store_pins: "Sequence[str | Path] | None" = None) -> tuple[str | None, str]:
    """The pin of a matrix row and WHY: ``(path, reason)`` — ``lens`` / ``kind:alphasteer`` on an AlphaSteer row; on an Arditi row
    ``lens`` / ``store:lens(<sources>)``, None / ``store:native(<sources>)`` or None / ``unpinned:no-store`` from TWO sources read
    together — the DISK store at ``wave_root/<out root>`` and the PINS record's engine for the row's admitted leg (``store_pins``: the
    repo's by default) — a disagreement refusing by name; None / ``unpinned:not-external`` on every other row (the kind-aware default
    stands). With no ``wave_root`` (the grading host plan) the disk is unread: an Arditi row the pins do not know reads None /
    ``unpinned:store-unread`` — never "no store"."""
    if row.get("object_source") != OBJECT_SOURCE_EXTERNAL:
        return None, SP.REASON_UNPINNED_NOT_EXTERNAL
    kind = SP.steer_cell_kind(str(row["cell"]))
    roots = None if wave_root is None else [Path(wave_root) / rel for rel in store_roots_rel(row)]
    pins = SP.read_pins_engines(STORE_PINS_PATHS if store_pins is None else [Path(p) for p in store_pins], leg_keys(row))
    path, reason, _stores = SP.pin_for(kind, roots, pins)
    return path, reason


def serving_path_for(row: dict, *, wave_root: "str | Path | None" = None, store_pins: "Sequence[str | Path] | None" = None) -> str | None:
    """The serving path a matrix row PINS (``serving_path_pin`` without the reason): `lens` on an AlphaSteer row, `lens` on an Arditi row
    whose store on record (disk or pins) is a lens store, None otherwise (the launcher's kind-aware default stands)."""
    return serving_path_pin(row, wave_root=wave_root, store_pins=store_pins)[0]


#: THE LAUNCH CELLS a single-cell job DECLARES to the wave shell (F22_LAUNCH_CELLS, common.sh f22_select_serving_path — the kind-aware
#: default of 2026-09-15): the HARNESS spelling of exactly the cells this job's standup serves — `none` (implicit on every
#: standup: mt_serving_up `--cell` serves ONLY the named TrainTest cell with `none` implicit, so `cells_requested` reads [none, <cell>])
#: plus the row's own cell — so the serving-path selection and the standup read the SAME cells. Never the overlay manifest's
#: `explicit_cells` (the base wave's six-cell overlay lists cells this job does not serve). A row that pins the lens keeps its pin (an
#: explicit F22_SERVING_PATH always wins, recorded EXPLICIT); every other row's path is selected BY KIND over these cells (a LoRA cell
#: -> lora_native; none / prompted -> the entry-point default) and logged `[serving-path] AUTO-selected`. Read by the sbatch BEFORE
#: f22_init (m5t_select_serving_path), printed by the dry run (LAUNCH_CELLS:).
def standup_cells(row: dict) -> list[str]:
    """The cells a row's standup registers, in LAUNCHER spelling and in the standup's order: the same-wave bare anchor ``none``, the
    row's own cell (unless it IS ``none``) and, on a seed row that carries standup companions (the GSM8K launcher's rule),
    each companion ONCE — the standup registers them beside the row's own so every standup of the wave composes one LoRA kernel
    configuration; the driver still measures the row's cell alone. ONE rule for the row's own overlay pair (`explicit_cells`,
    make_math500_overlays.seedrep_cells) and for the harness cells the sbatch declares (:func:`launch_cells_for`)."""
    cells = [CELL_NONE] + ([] if row["cell"] == CELL_NONE else [str(row["cell"])])
    for cell, _harness, _sid, _pin in row_companions(row):
        if cell not in cells:
            cells.append(cell)
    return cells


def launch_cells_for(row: dict) -> list[str]:
    """The HARNESS cells a row's standup serves — :func:`standup_cells` in the serving vocabulary (HARNESS_BY_CELL is one-to-one on
    every launcher cell, so the two lists correspond position by position)."""
    return [HARNESS_BY_CELL[c] for c in standup_cells(row)]


def row_companions(row: dict) -> list[tuple[str, str, str, str]]:
    """``[(cell, harness cell, staged id, pin), ...]`` — a seed row's standup companions in list order; ``[]`` on a row without the
    fields (every matrix row) or with empty ones (the companion's own row; the Arditi seed rows)."""
    cells = row.get("standup_companion_cells") or []
    harness = row.get("standup_companion_harness_cells") or []
    sids = row.get("standup_companion_staged_ids") or {}
    pins = row.get("standup_companion_pins") or {}
    return [(str(c), str(harness[i]), str(sids[c]), str(pins[c])) for i, c in enumerate(cells)]
STEER_KEYS = frozenset({"object_source", "lane_root", "bank_view_rel", "bank_cell_dir", "selection_rule", "selection_label",
                        "grid_limited", "rung", "lambda", "alpha", "served_layer", "producer", "ceiling_20pp",
                        "dose_evidence", "boundary_pick", "alternative_not_pinned", "held_reason", "admission_notes"})
#: the keys the ALL-POSITIONS row carries beyond the extension keys (arditi-qwen-all-positions). Deliberately DISJOINT from
#: the steer rung's dose vocabulary: no `alpha`, no `rung`, no `lambda`, no `ceiling_20pp` — this arm's dose is a RAW
#: COEFFICIENT and an alpha-shaped field on this row would put two different units on one axis. `bank_view` holds the
#: LANE's view (`ar_c0p3`) apart from the PANEL view in `view` (`it`) — the GSM8K launcher's spelling; the steer rows keep
#: theirs. `selection` and `fit_provenance` ride VERBATIM off the admission entry: every table, figure and provenance line
#: reads the two readings, the budget, the ladder cap, the single-trial flag, the informational alpha_equiv, the replicate
#: note and the direction's provenance from HERE, never from a second copy.
ALLPOS_KEYS = frozenset({"object_source", "lane_root", "bank_view", "bank_view_rel", "bank_cell_dir", "selection_rule",
                         "selection_label", "grid_limited", "selection", "fit_provenance", "coefficient", "dose_unit",
                         "object_kind", "phase", "positions", "skip_bos", "served_layer", "serving_path", "producer",
                         "dose_evidence", "attestation_expect", "admission_notes", "held_reason",
                         # the DECLARED expected refusal of the row's comparison (EXPECTED_REFUSAL_CROSS_ENGINE) and the admission
                         # record's selection-outcome word (the paper export requires both of this arm)
                         "expected_refusal", "selection_outcome"})
#: the keys an IT-DOC row carries beyond the extension keys: where its object comes from and how the bank door binds it.
#: `object_source: itdoc` tells every reader the pin is the IT-DOC ADMISSION RECORD's, not the full22 file's (nor the steer record's).
ITDOC_KEYS = frozenset({"object_source", "lane", "object_kind", "bank_sidecar", "bank_view", "bank_view_rel", "bank_cell_dir",
                        "parents_expected", "superseded", "stale_objects", "qualification", "qualification_pattern", "fit_record", "held_reason",
                        "admission_notes", "object_generation"})
#: the object GENERATION an it-doc row serves — the batch-64 lane's by construction (the GSM8K instruct-direct wave mixes it with the
#: served batch-8 constitution and records the mix on every row; g8k_cells.object_generation_of)
OBJECT_GENERATION_BATCH64 = "batch64"


def panel_job_id(key: str, panel_view: str, cell: str) -> str:
    """`m5t-<key>-<panel view>-<cell short>` — the job id of a non-published row on a NON-MAIN panel view: the same (model, cell) sits
    on the main view as another row (E3's it-doc rows share theirs with an ext1 row; every seed row)."""
    return f"m5t-{key}-{panel_view}-{cell_short(cell)}"


def out_root_for(wave: str, tag: str, cell: str, mode: str) -> str:
    """`<wave prefix>/<tag>/<cell slug>/<mode>` — the TAG segment is `<key>` on every matrix row (the reader's own layout) and
    `<key>_<view>` on a seed row (the GSM8K seedrep wave's layout: a pull of the `controls_qwen_seedrep/` subtree lays the views apart)."""
    return f"{OUT_ROOT_PREFIX_BY_WAVE[wave]}/{tag}/{cell_slug(cell)}/{mode}"


def _row(index: int, key: str, cell: str, mode: str, *, wave: str, status: str, pins: dict,
         admission: dict | None = None, itdoc: dict | None = None, seedrep: dict | None = None) -> dict:
    steer = cell in STEER_CELLS
    allpos = cell in ALLPOS_CELLS
    fitted = cell in FITTED_CELLS or steer
    staged = STAGED_ID_BY_CELL.get(cell)
    if itdoc is not None:
        # E3: the row's object is the it-doc record's — its ruled staged id and its pin (null while unverified), never the pin file's
        staged = str(itdoc["staged_id"])
    if seedrep is not None:
        # the seed wave's CAMPAIGN rows: the object is the seed view's roster object, staged `base64s<s>__…` / `derived_base64s<s>__…`
        # / `paired_eb64s<s>__…` and pinned in the full22 pin file under THAT id (load_seedrep_pins); the PSR cells are fitted too
        fitted = cell in SEEDREP_FITTED_CELLS
        staged = seedrep["staged_id"] if fitted else None
    geval, g_l = geometry_for(cell)
    published = wave == WAVE_PUBLISHED
    # the row's PANEL view — the view it is KEYED by (:func:`key_view`): an it-doc row's own `it`; the all-positions arm's
    # `it` (its LANE's bank view rides `bank_view` instead); a seed row's lane view (`s<s>` / `s<s>p`); `main` everywhere
    # else, a steer row included, whose `view` field names its object BANK rather than a panel slot
    panel_view = (seedrep["view"] if seedrep is not None else VIEW_IT if itdoc is not None
                  else ALLPOS_PANEL_VIEW if allpos else VIEW_MAIN)
    if published:
        job_id = f"m5t-{key}-{cell_short(cell)}" + ("" if mode == SAMPLING_THINKING_CARD else f"-{mode}")
        time_limit = TIME_OBJECT_GLM if fitted and key == "glm46vflash" else TIME_OBJECT if fitted else TIME_BASE
    else:
        # a row on a non-main PANEL view names it in its job id: the same (model, cell) sits on the main view as another
        # row (the it-doc rows share theirs with an ext1 row). A steer row keys `main` and keeps its old id.
        job_id = f"m5t-{key}-{cell_short(cell)}" if panel_view == VIEW_MAIN else panel_job_id(key, panel_view, cell)
        time_limit = TIME_BY_WAVE.get(wave, TIME_EXT)
    # the row's ARM (the extension key `thinking_mode` below): the stamp triples' render regime follows it (math500-steer)
    arm = THINKING_MODE_BY_FAMILY.get(FAMILY_BY_MODEL[key], THINKING_MODE)
    # the PUBLISHED keys first, in the published tree's insertion order (the fixture is compared key for key)
    row = {
        "index": index, "job_id": job_id, "model_key": key, "cell": cell,
        "cell_slug": cell_slug(cell), "harness_cell": HARNESS_BY_CELL[cell],
        "sampling_mode": mode, "sampling": dict(SAMPLING_ROWS[(key, mode)]),
        "fitted": fitted, "tuple_name": ((seedrep_tuple_name(key, mode, cell) if seedrep is not None else tuple_name(key, mode, cell))
                                        if fitted else None),
        "route_kind": route_kind(cell),
        "served_positions_expected": "all" if fitted else None,
        "expected_stamps": list(stamps_for(fitted, arm)),
        "twin_expected_stamps": list(twin_stamps_for(arm)) if fitted else None,
        # a seed row's out root carries the TAG segment `<key>_<view>`; every other row's `<key>` (out_root_for)
        "out_root": out_root_for(wave, f"{key}_{panel_view}" if seedrep is not None else key, cell, mode),
        "time_limit": time_limit,
        "cluster_preference": PRIMARY_CLUSTER,
        "family": FAMILY_BY_MODEL[key], "split_mode_candidate": SPLIT_MODE_CANDIDATE[FAMILY_BY_MODEL[key]],
        "reasoning_parser": REASONING_PARSER_BY_MODEL.get(key),
        "gpu_mem_util": GPU_MEM_UTIL_BY_MODEL.get(key, GPU_MEM_UTIL_DEFAULT),
        "object_sha256": (admission["object_sha256"] if steer else itdoc["object_sha256"] if itdoc is not None
                          else seedrep["pin"] if seedrep is not None
                          else pins[(key, staged)] if fitted else None),
        "smoke_diagnostic_out": f"diagnostic/{FAMILY_BY_MODEL[key]}",
    }
    # the EXTENSION keys (on every row)
    row.update({
        "wave": wave, "status": status,
        "thinking_mode": arm,
        "view": admission["view"] if steer and not allpos else panel_view,
        "geometry_eval": geval, "geometry_L": g_l, "budget_class": budget_class(cell),
        "staged_id": admission.get("staged_id") if steer else staged, "si_sha256": SI_SHA256_BY_CELL.get(cell),
        "log_requests": not published,
        "cluster_constraint": (CLUSTER_CONSTRAINT_BY_WAVE.get(wave)                   # the wave's own placement first (the seed wave)
                               or (admission["cluster_constraint"] if steer
                                   else ((itdoc.get("bank") or {}).get("cluster_constraint") or CLUSTER_CONSTRAINT_BY_MODEL.get(key, CLUSTER_CONSTRAINT_ANY))
                                   if itdoc is not None else CLUSTER_CONSTRAINT_BY_MODEL.get(key, CLUSTER_CONSTRAINT_ANY))),
        "extension": not published,
    })
    if seedrep is not None:
        # the seed row's OWN facts (SEEDREP_KEYS) — its lane view as the object BANK view too (so key_view keys it by the view and
        # the launcher cuts the `<key>_<view>` overlay pair), the own overlay pair, the opt-in replicate seed rule at the view's
        # base, the standup companion (the view's r=4 parent, pinned, unmeasured — empty on the companion's own row), the
        # `--bank-view` stamp (contracts.expected_bank_view_stamp -> the lane view itself, no source bank declared) and the
        # object generation (batch64 on a fitted cell; None on the bare / prompted routes)
        row.update(_seedrep_fields(seedrep["view"], cell, seedrep["companions"], bank_view=panel_view))
        assert set(row) - PUBLISHED_KEYS == EXTENSION_KEYS | SEEDREP_KEYS, sorted(set(row) - PUBLISHED_KEYS)
        return row
    if itdoc is not None:
        # the it-doc row's OWN facts, straight from the admission record (E3): recorded on the ROW because the bank door, every
        # table, figure and provenance line reads them from here. `status` is the RECORD's (entry_launchable), `held_reason` its.
        bank = itdoc.get("bank") or {}
        row.update({
            "object_source": OBJECT_SOURCE_ITDOC, "lane": str(itdoc["lane"]),
            "object_kind": str(itdoc["object_kind"]), "bank_sidecar": B.SIDECAR_BY_KIND[str(itdoc["object_kind"])],
            "bank_view": bank.get("view"), "bank_view_rel": (dict(bank["bank_view_rel"]) if bank.get("bank_view_rel") else None),
            "bank_cell_dir": bank.get("bank_cell_dir"),
            "parents_expected": (dict(itdoc["parents_expected"]) if itdoc.get("parents_expected") else None),
            "superseded": (dict(itdoc["superseded"]) if itdoc.get("superseded") else None),
            "stale_objects": dict(itdoc.get("stale_objects") or {}),
            "qualification": itdoc.get("qualification"),
            "qualification_pattern": itdoc.get("qualification_pattern"),
            "fit_record": (dict(itdoc["fit_record"]) if itdoc.get("fit_record") else None),
            "held_reason": None if status == STATUS_LAUNCH else (str(itdoc.get("hold_reason") or "").strip() or None),
            "admission_notes": list(itdoc.get("notes") or []),
            "object_generation": OBJECT_GENERATION_BATCH64,
        })
        assert set(row) - PUBLISHED_KEYS == EXTENSION_KEYS | ITDOC_KEYS, sorted(set(row) - PUBLISHED_KEYS)
        return row
    if allpos:
        # arditi-qwen-all-positions: the all-positions row's OWN facts, straight from the admission entry. The SELECTION is
        # carried whole and unparaphrased — both readings (the rule's c = 0.2 at +5.04 pp and this point's +10.26 pp), the
        # budget, the LADDER CAP that identifies the reading (at another cap the rule returns a different point), the
        # single-trial flag (every margin here is a ratio against a crude floor, never a significance claim), the dose with
        # its unit, the informational alpha_equiv and the replicate note — because nothing downstream may re-derive or
        # round any of it, and nothing anywhere may say the RULE chose this dose: it is a declared user override, over budget.
        row.update({
            "object_source": OBJECT_SOURCE_EXTERNAL, "lane_root": admission["lane_root"],
            "bank_view": admission["view"], "bank_view_rel": dict(admission["bank_view_rel"]),
            "bank_cell_dir": admission["bank_cell_dir"],
            "selection_rule": admission["selection_rule"], "selection_label": admission["selection_label"],
            "grid_limited": bool(admission["grid_limited"]),
            "selection": dict(admission["selection"]), "fit_provenance": dict(admission["fit_provenance"]),
            "coefficient": admission["coefficient"], "dose_unit": admission["dose_unit"],
            "object_kind": admission["kind"], "phase": admission["phase"], "positions": admission["positions"],
            "skip_bos": bool(admission["skip_bos"]), "served_layer": admission["served_layer"],
            # the arm's DECLARED serving route, in every leg: the graph-captured native residual slot. The row's
            # serving-path PIN is a separate question the store answers (`serving_path_pin`) — with no store on record
            # the kind-aware default selects this same path; a `lens` leg would measure something else under this name.
            "serving_path": admission["serving_path"],
            "producer": dict(admission["producer"]), "dose_evidence": dict(admission["dose_evidence"]),
            "attestation_expect": dict(admission["attestation_expect"]),
            "admission_notes": list(admission.get("notes") or []),
            "held_reason": held_reason_for(key, cell, status),
        })
        row["tuple_name"] = tuple_name(key, mode, cell)
        # arditi-allpos-cap (2026-09-17), the two words the PAPER export requires of this arm and reads off the ROW: the admission
        # record's `selection_outcome` (the same word the GSM8K generator stamps on its external rows; `selection.outcome` must
        # equal it — "one word, in both places") and the DECLARED expected refusal of the row's comparison
        # (EXPECTED_REFUSAL_CROSS_ENGINE: the native arm against the lens `none` anchor is O106-inadmissible and no native anchor
        # is run — the reader files the comparison as refused by design and reports the arm's numbers as the arm's numbers)
        row["selection_outcome"] = admission.get("selection_outcome")
        row["expected_refusal"] = EXPECTED_REFUSAL_CROSS_ENGINE
        assert set(row) - PUBLISHED_KEYS == EXTENSION_KEYS | ALLPOS_KEYS, sorted(set(row) - PUBLISHED_KEYS)
        return row
    if steer:
        # the steer row's OWN facts, straight from the admission record: the pin's provenance, the selection and its
        # VERBATIM label, the grid-limited flag, and the per-arm dose facts. Recorded on the
        # ROW because every table, figure and provenance line reads them from here.
        row.update({
            "object_source": OBJECT_SOURCE_EXTERNAL, "lane_root": admission["lane_root"],
            "bank_view_rel": dict(admission["bank_view_rel"]), "bank_cell_dir": admission["bank_cell_dir"],
            "selection_rule": admission["selection_rule"], "selection_label": admission["selection_label"],
            "grid_limited": bool(admission["grid_limited"]),
            "rung": admission.get("rung"), "lambda": admission.get("lambda"),
            "alpha": admission.get("alpha"), "served_layer": admission.get("served_layer"),
            "producer": dict(admission["producer"]), "ceiling_20pp": admission.get("ceiling_20pp"),
            "dose_evidence": dict(admission["dose_evidence"]),
            "boundary_pick": (admission.get("boundary_pick") or {}).get("status"),
            "alternative_not_pinned": admission.get("alternative_not_pinned"),
            "admission_notes": list(admission.get("notes") or []),
            "held_reason": held_reason_for(key, cell, status),
        })
        row["tuple_name"] = tuple_name(key, mode, cell)
        assert set(row) - PUBLISHED_KEYS == EXTENSION_KEYS | STEER_KEYS, sorted(set(row) - PUBLISHED_KEYS)
        return row
    assert set(row) - PUBLISHED_KEYS == EXTENSION_KEYS, sorted(set(row) - PUBLISHED_KEYS)
    return row


def steer_hold_reason(model_key: str) -> str | None:
    """Why this model's steering rows — and its llama-wave rows — cannot launch on THIS tree — ``None`` when
    they can. The one condition today: the family's render fact is not certified here (RENDER_FACT_CERTIFIED_BY_FAMILY,
    the mirror of contracts.render_fact_certified), so what the model renders and emits under this arm is unknown and a
    production leg would be a guess. It is a HOLD, not a design deferral: the object, the dose and the pin are admitted
    (the record is complete) — only the tree is not ready, and the row flips to `launch` when the fact lands."""
    family = FAMILY_BY_MODEL[model_key]
    if RENDER_FACT_CERTIFIED_BY_FAMILY.get(family):
        return None
    return (f"held: this tree carries no CERTIFIED render fact for family {family!r} "
            f"(contracts.FAMILY_RENDER_FACTS / cells.RENDER_FACT_CERTIFIED_BY_FAMILY) — the admission record for these "
            "rows is complete (object, dose, selection, pin), so this is a TREE gate, not a design deferral: the rows "
            "flip to `launch` when the family's certified fact lands on main, with no repin and no re-selection")


#: PER-ROW holds on the steer rung: a (model key, cell) whose STAGED RECORD the driver's
#: provenance mirror refuses — the object, dose, selection and pin are admitted, the row is not launchable until the record's
#: owner re-stages it; removing the entry flips the row to `launch` with no repin and no re-selection. EMPTY today: the one
#: entry it carried (llama8b AlphaSteer, launch class B — the `it` view's fit_provenance.json was the PARENT object's,
#: object_sha256 645a9f89… ≠ the pin 4cf8e164…) was lifted 2026-09-13 when the AlphaSteer record was re-staged
#: on the primary cluster (it now binds the pinned object's own sha, the parent under `derived_from`); the row's only remaining
#: defect was class D (`thinking_card` → `card`), fixed here. The MECHANISM stays (test-pinned): a refused record is a
#: row hold by name, never a silent launch. (The Arditi rows' fit-writer allow-list — class A — and the low-dose AlphaSteer
#: points' positive control — class C — are not row holds: the first is a shared eval-code fix, the second an instrument
#: decision; both refuse in the job, never silently.)
STEER_ROW_HOLDS: dict[tuple[str, str], str] = {}


def steer_row_hold_reason(model_key: str, cell: str) -> str | None:
    """Why THIS (model, cell) steering row cannot launch on any tree today, or None: the family's tree gate
    (steer_hold_reason) first, then the row's own staged-record hold (STEER_ROW_HOLDS)."""
    return steer_hold_reason(model_key) or STEER_ROW_HOLDS.get((model_key, cell))


def held_reason_for(model_key: str, cell: str, status: str) -> str | None:
    if status != STATUS_HELD:
        return None
    return steer_row_hold_reason(model_key, cell) if cell in STEER_CELLS else steer_hold_reason(model_key)


def _build_rows(pins: dict | None = None) -> list[dict]:
    """Rows 0-23: the PUBLISHED matrix, frozen (wave `published`; status `launch` — their stores are required
    evidence; the wave being complete is what keeps them out of every submission, A12)."""
    pins = pins if pins is not None else load_pins()
    rows: list[dict] = []
    # ORDER: primary rows first (rows 0-11), then qwen general (12-14), then battery (15-23) — the plan's numbering
    for mode in SAMPLING_MODES:
        for key in MODEL_KEYS:
            if mode not in SAMPLING_MODES_BY_MODEL[key]:
                continue
            for cell in CELLS:
                rows.append(_row(len(rows), key, cell, mode, wave=WAVE_PUBLISHED, status=STATUS_LAUNCH, pins=pins))
    return rows


def _build_extension_rows(start: int, pins: dict | None = None) -> list[dict]:
    """Rows 24-32: the eight-arm extension (wave `ext1`, status `launch`): per EXT_MODELS model, the EXT_CELLS in the
    eight-arm order under the PRIMARY sampling row — gemma 24-26, qwen 27-29, nemotron 30-32."""
    pins = pins if pins is not None else load_pins()
    rows: list[dict] = []
    for key in EXT_MODELS:
        for cell in EXT_CELLS:
            rows.append(_row(start + len(rows), key, cell, SAMPLING_THINKING_CARD, wave=WAVE_EXT1, status=STATUS_LAUNCH,
                             pins=pins))
    return rows


def _build_steer_rows(start: int, pins: dict | None = None, admission: dict | None = None) -> list[dict]:
    """The steering rung's rows (wave `steer`): per STEER_MODELS model, the two EXTERNAL steering cells in the eight-arm
    order, under the PRIMARY sampling row, each pinned to the ADMISSION RECORD's object sha256 and carrying its
    selection facts. A model whose family has no certified render fact on this tree is `held` (steer_hold_reason), and so
    is a (model, cell) whose staged record the driver's mirror refuses (STEER_ROW_HOLDS, E2-12); every other row is
    `launch`. A (model, cell) the record does not admit is simply not a row — the
    matrix never carries a steering row with no admission entry (plan P8)."""
    pins = pins if pins is not None else load_pins()
    admission = admission if admission is not None else load_steer_admission()
    rows: list[dict] = []
    for key in STEER_MODELS:
        # the RUNG's two cells (WAVE_STEER_CELLS), never the whole steer vocabulary: the all-positions arm is its own
        # wave, appended after every existing row, so no index, out root, pin or slice of this rung moves for it
        for cell in WAVE_STEER_CELLS:
            entry = admission.get((key, cell))
            if entry is None:
                continue
            status = STATUS_HELD if steer_row_hold_reason(key, cell) else STATUS_LAUNCH
            # the row's MODE is the model's (E2-4): the no-switch family's shipped `card` row, a switch family's `thinking_card`
            rows.append(_row(start + len(rows), key, cell, sampling_mode_for(key), wave=WAVE_STEER, status=status,
                             pins=pins, admission=entry))
    return rows


def _build_llama_rows(start: int, pins: dict | None = None) -> list[dict]:
    """Rows 41-46 (E2, plan P7): the no-switch family's six non-steering arms in the eight-arm order (LLAMA_CELLS)
    under its shipped `card` row, `thinking_mode not_available`, its `main` view's objects (the shared llama8b roster
    block: `base__…` / `derived__…` staged ids, pinned in the full22 pin file), NO tuple on the fitted rows (in support),
    wave `llama`, out roots under `controls_llama/`. `launch` iff the family's render fact is certified on this tree
    (steer_hold_reason: the same TREE gate the steering wave's llama rows carried), never a design deferral."""
    pins = pins if pins is not None else load_pins()
    rows: list[dict] = []
    for key in LLAMA_MODELS:
        status = STATUS_HELD if steer_hold_reason(key) else STATUS_LAUNCH
        for cell in LLAMA_CELLS:
            rows.append(_row(start + len(rows), key, cell, sampling_mode_for(key), wave=WAVE_LLAMA, status=status, pins=pins))
    return rows


def _build_itdoc_rows(start: int, pins: dict | None = None, admission: dict | None = None) -> list[dict]:
    """Rows 47-52 (E3): per ITDOC_MODELS model, the three LoRA arms of the `it` view in the eight-arm order, under the PRIMARY
    sampling row, each generated from the it-doc ADMISSION RECORD's (lane, key, it, cell) entry — `launch` iff the entry is
    launchable (itdoc_bind.entry_launchable: pin + ruled staged id + complete bank + no hold reason), else `held` with the record's
    reason. A (key, it, cell) the record does not name refuses by name (a re-fit row never exists without its record)."""
    pins = pins if pins is not None else load_pins()
    admission = admission if admission is not None else load_itdoc_admission()
    rows: list[dict] = []
    for key in ITDOC_MODELS:
        for cell in ITDOC_CELLS:
            entry = B.entry_for(admission, admission["lane"], key, VIEW_IT, cell)
            if entry is None:
                raise SystemExit(f"cells.py: the it-doc admission record names no entry for {B.tuple_label(admission['lane'], key, VIEW_IT, cell)} "
                                 "— an it-doc row never exists without its record")
            # the record's INSTRUMENT hold keeps every MATH-500 it-view row HELD whatever its bank binding says (the GSM8K wave alone
            # is authorized); a held-by-instrument row carries that reason
            hold = B.instrument_hold(admission, "math500")
            launch = B.entry_launchable(entry) and hold is None
            if not launch and not str(entry.get("hold_reason") or "").strip():
                entry = {**entry, "hold_reason": hold}
            status = STATUS_LAUNCH if launch else STATUS_HELD
            rows.append(_row(start + len(rows), key, cell, SAMPLING_THINKING_CARD, wave=WAVE_ITDOC, status=status, pins=pins, itdoc=entry))
    return rows


def _allpos_row(index: int, key: str, cell: str, admission: dict, *, wave: str) -> dict:
    """ONE all-positions Arditi row from the P8 admission record (keyed by model key and cell id) on `wave` — row 53's shape; the seed
    wave re-keys it onto its views (_build_seedrep_rows). A (model, cell) the record does not admit REFUSES BY NAME: this arm never
    exists without its entry, and a silently absent row would read as a lane that was never scheduled. `launch` unless the row carries
    a staged-record hold (the steering rung's own predicate, unchanged)."""
    entry = admission.get((key, cell))
    if entry is None:
        raise SystemExit(f"cells.py: the steer admission record ({STEER_ADMISSION_PATH.name}) names no entry for "
                         f"({key!r}, {cell!r}) — the all-positions row is generated from the record and never exists without it")
    status = STATUS_HELD if steer_row_hold_reason(key, cell) else STATUS_LAUNCH
    return _row(index, key, cell, sampling_mode_for(key), wave=wave, status=status, pins={}, admission=entry)


def _build_allpos_rows(start: int, pins: dict | None = None, admission: dict | None = None) -> list[dict]:
    """Row 53 (arditi-qwen-all-positions): the ALL-POSITIONS Arditi arm, one row per (ALLPOS_MODELS, ALLPOS_CELLS) pair,
    on the `it` PANEL view under the model's primary sampling row, generated from the SAME P8 admission record as the steering
    rung (keyed by model key and cell id) — its own wave, appended after every existing row. A (model, cell) the record
    does not admit REFUSES BY NAME: this arm never exists without its entry, and a silently absent row would read as a
    lane that was never scheduled. `launch` unless the family's render fact is uncertified here or the row carries a
    staged-record hold (the steering rung's own predicate, unchanged)."""
    # `pins` is the builders' common parameter; an all-positions row's pin is the admission entry's, so it is not read here
    admission = admission if admission is not None else load_steer_admission()
    rows: list[dict] = []
    for key in ALLPOS_MODELS:
        for cell in ALLPOS_CELLS:
            rows.append(_allpos_row(start + len(rows), key, cell, admission, wave=WAVE_ALLPOS))
    return rows


# --------------------------------------------------------------------------- the QWEN SEED-REPLICATE wave
#: the seed views and their PAIRED views: `s<s>` serves replicate s's eight arms; `s<s>p` serves ONLY the composed
#: difference ReLU (no own `none` — it pairs with `s<s>/none`). One target: the roster's qwen35_4b.
SEEDREP_MODELS = ("qwen35_4b",)
SEEDREP_VIEWS = ("s1", "s2", "s3")
SEEDREP_PAIRED_VIEWS = ("s1p", "s2p", "s3p")
#: the replicate SEED BASE per view — the LITERAL training seed of the view (scripts/sdfbase/seedrep/eval_seed.py seed_base(s) = s;
#: the paired view carries its seed's base): a LOCAL MIRROR of eval_seed.SEED_BASE_BY_ANY_VIEW, pinned equal to it by
#: tests/test_math500_qwen_seedrep.py — mirrored, not imported: this file imports nothing outside its own directory + full22 + the
#: stdlib (the client entry points' import walk resolves only those roots; the GSM8K matrices mirror the same table the same way)
SEEDREP_SEED_BASE_OF_VIEW = {"s1": 1, "s2": 2, "s3": 3, "s1p": 1, "s2p": 2, "s3p": 3}
#: the opt-in Inspect seed rule every seed row runs under (contracts.INSPECT_SEED_POLICY_REPLICATE, mirrored; admitted for the
#: `math` task under `thinking_card` — served_capability_run refuses it only for inspect/ifeval and under `protocol`)
INSPECT_SEED_POLICY_REPLICATE = "replicate"
#: the eight arms per seed view, in the EIGHT-ARM order: none · constitution ReLU · control ReLU · constitution LoRA ·
#: control LoRA · difference LoRA · prompted safety · prompted constitution
SEEDREP_CELLS = (CELL_NONE, CELL_PSR_MIX, CELL_PSR_CTL2, CELL_LORA_MIX, CELL_LORA_CTL2, CELL_OBJECT, CELL_PROMPTED, CELL_PROMPTED_CONST)
#: the FITTED seed cells (an object per view in the pin file): the five lane cells (== contracts.G8K_NATIVE_CELLS, test-pinned) and
#: the composed paired-write cell (== contracts.G8K_NATIVE_PAIRED_CELLS)
SEEDREP_NATIVE_CELLS = (CELL_PSR_MIX, CELL_LORA_MIX, CELL_PSR_CTL2, CELL_LORA_CTL2, CELL_OBJECT)
SEEDREP_FITTED_CELLS = SEEDREP_NATIVE_CELLS + (CELL_PAIRED,)
#: the STANDUP COMPANION every seed standup registers beside the row's own cell (GSM8K O12a): the view's r=4 constitution LoRA,
#: pinned and UNMEASURED, so every standup of the wave composes `--max-lora-rank 8` and the reader's launch-flag compare admits every
#: contrast against the view's `none`. Empty on the companion's own row (it registers itself) and on the single-object Arditi rows.
SEEDREP_COMPANION_CELL = CELL_LORA_MIX
assert SEEDREP_COMPANION_CELL in SEEDREP_NATIVE_CELLS       # a lane cell: pinned on every seed view with the other four (load_seedrep_pins)
#: the two native MATH-500 tuples (mirrored by name, pinned equal to contracts.m5t_native_tuple_name /
#: m5t_native_paired_tuple_name): the five lane cells under `_native`, the composed cell under `_native_paired`, both with the
#: native engine's served identity (reduction_mode not_applicable); the Arditi seed rows ride the existing `_steer_allpos`
SEEDREP_TUPLE_NATIVE = "math500_thinking_on_full22_{key}_native"
SEEDREP_TUPLE_NATIVE_PAIRED = "math500_thinking_on_full22_{key}_native_paired"
#: the `--bank-view` stamp rule (contracts.expected_bank_view_stamp, mirrored): a row on a LANE view stamps its SOURCE bank when it
#: declares one (the Arditi seed rows: `ar_c0p3`), else the lane view itself
INSTRUMENT_BANK_VIEWS = (VIEW_MAIN, VIEW_IT)          # == contracts.BANK_VIEWS (test-pinned)
#: the keys a SEED row carries beyond the extension keys (the campaign rows; the Arditi seed rows carry ALLPOS_KEYS | these | pairing_note)
SEEDREP_KEYS = frozenset({"bank_view", "own_overlay", "inspect_seed_policy", "inspect_seed_base", "seed_base",
                          "standup_companion_cells", "standup_companion_harness_cells", "standup_companion_staged_ids",
                          "standup_companion_pins", "bank_view_stamp", "object_generation"})
SEEDREP_ARDITI_KEYS = SEEDREP_KEYS | {"pairing_note"}
#: how the reader is expected to treat an Arditi seed row's contrast against `s<s>/none`
SEEDREP_ARDITI_PAIRING_NOTE = (
    "the Arditi engine is SINGLE-OBJECT by construction (a second object beside arditi_allpos is skipped by name), so its standup "
    "attests max_lora_rank 1 while every other standup of this view composes 8 (the r=4 companion); the reader's launch-flag "
    "comparability verdict decides the pairing against s<s>/none and the VALUE is filed regardless (GSM8K C4 precedent, "
    "arditi_unpaired_value; D71). No O106 refusal is declared: the anchor is the same native engine.")


def seedrep_seed_base(view: str) -> int:
    """The replicate seed base a seed / paired view carries (`s2` -> 2, `s2p` -> 2); any other view refuses by name."""
    try:
        return SEEDREP_SEED_BASE_OF_VIEW[view]
    except KeyError:
        raise SystemExit(f"{view!r} is not a seed-replicate view ({list(SEEDREP_SEED_BASE_OF_VIEW)})") from None


def seedrep_staged_id(view: str, cell: str) -> str | None:
    """The full22 pin file's STAGED id of `cell` on the seed / paired view (the roster's mapping, mirrored by rule and pinned equal to
    the roster by test): the four parents `base64s<s>__<cell>`, the LoRA difference `derived_base64s<s>__<cell>`, the composed
    paired write `paired_eb64s<s>__<cell>`; None on the bare / prompted routes (no object)."""
    s = seedrep_seed_base(view)
    if cell in (CELL_PSR_MIX, CELL_PSR_CTL2, CELL_LORA_MIX, CELL_LORA_CTL2):
        return f"base64s{s}__{cell}"
    if cell == CELL_OBJECT:
        return f"derived_base64s{s}__{cell}"
    if cell == CELL_PAIRED:
        return f"paired_eb64s{s}__{cell}"
    if cell == CELL_NONE or cell.startswith("prompted__"):
        return None
    raise SystemExit(f"seedrep_staged_id: {cell!r} is not a cell of the seed wave ({SEEDREP_CELLS + (CELL_PAIRED,)})")


def load_seedrep_pins(path: Path = PINS_PATH) -> dict[tuple[str, str], str]:
    """(view, cell) -> the pinned object.pt sha256 for EVERY fitted cell the seed wave serves (the eight-arm cells on `s1..s3`, the
    composed cell on `s1p..s3p`, the companion on every view), read from configs/sdfbase_full22_pins.json under the staged ids
    :func:`seedrep_staged_id` names. `load_pins` (the published rows' pins) is untouched."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[tuple[str, str], str] = {}
    for key in SEEDREP_MODELS:
        for view in SEEDREP_VIEWS:
            for cell in SEEDREP_NATIVE_CELLS:
                out[(view, cell)] = _seedrep_pin(doc, key, view, cell)
        for view in SEEDREP_PAIRED_VIEWS:
            for cell in (CELL_PAIRED, SEEDREP_COMPANION_CELL):
                out[(view, cell)] = _seedrep_pin(doc, key, view, cell)
    return out


def _seedrep_pin(doc: dict, key: str, view: str, cell: str) -> str:
    staged = seedrep_staged_id(view, cell)
    entry = (doc.get(key) or {}).get(staged)
    sha = entry["object_sha256"] if isinstance(entry, dict) else entry
    if not is_object_sha(sha):
        raise SystemExit(f"pins: {key}/{staged} (view {view}, cell {cell}) has no 64-hex object_sha256 in {PINS_PATH.name} — a seed row is never "
                         "minted on an unpinned object")
    return sha


def seedrep_tuple_name(model_key: str, sampling_mode: str, cell: str) -> str | None:
    """The authorised out-of-support tuple of a FITTED seed row: the five lane cells -> `…_<key>_native`, the composed cell ->
    `…_<key>_native_paired` (both the native engine's served identity, beside the lens-era tuples), `arditi_allpos` -> the arm's own
    `_steer_allpos` (view is not a compared field; the GSM8K arditi_seedrep rows rode the same pair). The primary sampling row only."""
    if model_key not in SEEDREP_MODELS:
        raise SystemExit(f"seedrep_tuple_name: the seed wave runs on {list(SEEDREP_MODELS)} only, not {model_key!r}")
    if sampling_mode != SAMPLING_THINKING_CARD:
        raise SystemExit(f"seedrep_tuple_name: the seed wave runs under {SAMPLING_THINKING_CARD!r} only (no sensitivity rows), not {sampling_mode!r}")
    if cell in SEEDREP_NATIVE_CELLS:
        return SEEDREP_TUPLE_NATIVE.format(key=model_key)
    if cell == CELL_PAIRED:
        return SEEDREP_TUPLE_NATIVE_PAIRED.format(key=model_key)
    if cell == CELL_ARDITI_ALLPOS:
        return tuple_name(model_key, sampling_mode, cell)
    raise SystemExit(f"seedrep_tuple_name: {cell!r} is not a fitted cell of the seed wave ({SEEDREP_FITTED_CELLS + ALLPOS_CELLS})")


def seedrep_bank_view_stamp(view: str, bank_view: str | None) -> str:
    """The `--bank-view` value a seed row's driver stamps — contracts.expected_bank_view_stamp's rule, mirrored (pinned equal by test):
    an instrument view (`main` / `it`) stamps itself; a LANE view stamps its declared source bank (`ar_c0p3` on the Arditi seed rows),
    else the lane view itself (the lane's own objects live under `cells_view/<view>`)."""
    if view in INSTRUMENT_BANK_VIEWS:
        return view
    return bank_view if bank_view else view


def seedrep_companions(view: str, cell: str, pins: dict) -> list[tuple[str, str, str, str]]:
    """`[(cell, harness cell, staged id, pin)]` — the standup companions of a seed row: the view's r=4 companion (pinned) on every
    campaign row but the companion's own; NONE on the companion's row and on the single-object Arditi rows."""
    if cell in (SEEDREP_COMPANION_CELL, CELL_ARDITI_ALLPOS):
        return []
    c = SEEDREP_COMPANION_CELL
    return [(c, HARNESS_BY_CELL[c], seedrep_staged_id(view, c), pins[(view, c)])]


def _seedrep_fields(view: str, cell: str, comps: list[tuple[str, str, str, str]], *, bank_view: str | None) -> dict:
    """The SEEDREP_KEYS block of a row on seed view `view` with standup companions `comps` (row_companions' shape): the campaign rows,
    and the Arditi rows re-keyed onto the view with no companion."""
    base = seedrep_seed_base(view)
    return {
        "bank_view": bank_view, "own_overlay": True,
        "inspect_seed_policy": INSPECT_SEED_POLICY_REPLICATE, "inspect_seed_base": base, "seed_base": base,
        "standup_companion_cells": [c for c, _h, _sid, _pin in comps],
        "standup_companion_harness_cells": [h for _c, h, _sid, _pin in comps],
        "standup_companion_staged_ids": {c: sid for c, _h, sid, _pin in comps},
        "standup_companion_pins": {c: pin for c, _h, _sid, pin in comps},
        "bank_view_stamp": seedrep_bank_view_stamp(view, bank_view),
        "object_generation": OBJECT_GENERATION_BATCH64 if cell in SEEDREP_FITTED_CELLS else None,
    }


def _build_seedrep_rows(start: int, pins: dict | None = None, admission: dict | None = None) -> list[dict]:
    """Rows 54-86 (wave `qwen_seedrep`, status `launch`): per seed view the eight arms in SEEDREP_CELLS order (24), per paired view
    the composed difference ReLU alone (3), then per seed view the Arditi all-positions refusal steer at seed base s (3) — the allpos
    row's shape (the SAME admission entry row 53 is generated from) RE-KEYED onto the seed view: `view s<s>`, the seed wave's job id /
    out root / time class / overflow constraint, NO O106 declaration (`expected_refusal: None`) and a `pairing_note`, its bank
    view `ar_c0p3` kept apart (so the row keys on `s<s>` and stamps `--bank-view ar_c0p3`). Every campaign row's pin is the seed
    view's roster object (load_seedrep_pins); an unpinned object refuses at import, never a placeholder."""
    pins = pins if pins is not None else load_seedrep_pins()
    admission = admission if admission is not None else load_steer_admission()
    rows: list[dict] = []
    for key in SEEDREP_MODELS:
        # the CAMPAIGN rows in index order: the eight arms per seed view, then the composed cell per paired view (a bare / prompted cell
        # carries no pin; every fitted one the view's roster object's)
        for view, cell in [(v, c) for v in SEEDREP_VIEWS for c in SEEDREP_CELLS] + [(v, CELL_PAIRED) for v in SEEDREP_PAIRED_VIEWS]:
            rows.append(_row(start + len(rows), key, cell, SAMPLING_THINKING_CARD, wave=WAVE_SEEDREP, status=STATUS_LAUNCH, pins={},
                             seedrep={"view": view, "staged_id": seedrep_staged_id(view, cell),
                                      "pin": pins[(view, cell)] if cell in SEEDREP_FITTED_CELLS else None,
                                      "companions": seedrep_companions(view, cell, pins)}))
        for view in SEEDREP_VIEWS:
            cell = CELL_ARDITI_ALLPOS
            row = _allpos_row(start + len(rows), key, cell, admission, wave=WAVE_SEEDREP)
            # the RE-KEYING (the fields the seed VIEW moves; the wave's time class and placement came from _row's per-wave tables, every
            # other field is the admission entry's, byte for byte)
            row.update({
                "view": view, "job_id": panel_job_id(key, view, cell),
                "out_root": out_root_for(WAVE_SEEDREP, f"{key}_{view}", cell, row["sampling_mode"]),
                "expected_refusal": None, "pairing_note": SEEDREP_ARDITI_PAIRING_NOTE,
            })
            row.update(_seedrep_fields(view, cell, [], bank_view=str(row["bank_view"])))
            assert set(row) - PUBLISHED_KEYS == EXTENSION_KEYS | ALLPOS_KEYS | SEEDREP_ARDITI_KEYS, sorted(set(row) - PUBLISHED_KEYS)
            rows.append(row)
        # each PAIRED view's OWN `none` anchor -- APPENDED LAST (indices 84-86) so no earlier index, out root, pin or
        # slice moves; the bare route with the view's r=4 companion, like s<s>/none
        for view in SEEDREP_PAIRED_VIEWS:
            rows.append(_row(start + len(rows), key, CELL_NONE, SAMPLING_THINKING_CARD, wave=WAVE_SEEDREP, status=STATUS_LAUNCH, pins={},
                             seedrep={"view": view, "staged_id": seedrep_staged_id(view, CELL_NONE), "pin": None,
                                      "companions": seedrep_companions(view, CELL_NONE, pins)}))
    return rows


_PINS = load_pins()
_STEER_ADMISSION = load_steer_admission()
_ITDOC_ADMISSION = load_itdoc_admission()
_SEEDREP_PINS = load_seedrep_pins()
#: every pin a launcher row may carry, keyed (model key, staged id) the way `load_pins` keys the published / extension pins — the seed
#: views' re-keyed from (view, cell) through their staged ids; `_counts` reports only the pins some row of ITS `rows` actually carries
_ALL_PINS = {**_PINS, **{(SEEDREP_MODELS[0], seedrep_staged_id(v, c)): sha for (v, c), sha in _SEEDREP_PINS.items()}}
BASE_ROWS: list[dict] = _build_rows(_PINS)
N_BASE = len(BASE_ROWS)                                    # 24
EXTENSION_ROWS: list[dict] = _build_extension_rows(N_BASE, _PINS)
STEER_ROWS: list[dict] = _build_steer_rows(N_BASE + len(EXTENSION_ROWS), _PINS, _STEER_ADMISSION)
LLAMA_ROWS: list[dict] = _build_llama_rows(N_BASE + len(EXTENSION_ROWS) + len(STEER_ROWS), _PINS)
ITDOC_ROWS: list[dict] = _build_itdoc_rows(N_BASE + len(EXTENSION_ROWS) + len(STEER_ROWS) + len(LLAMA_ROWS), _PINS, _ITDOC_ADMISSION)
#: arditi-qwen-all-positions: APPENDED LAST, on purpose — every index, wave slice, count and out root above is untouched
ALLPOS_ROWS: list[dict] = _build_allpos_rows(
    N_BASE + len(EXTENSION_ROWS) + len(STEER_ROWS) + len(LLAMA_ROWS) + len(ITDOC_ROWS), _PINS, _STEER_ADMISSION)
M5T_CELLS: list[dict] = BASE_ROWS + EXTENSION_ROWS + STEER_ROWS + LLAMA_ROWS + ITDOC_ROWS + ALLPOS_ROWS
#: the seed wave, a SIBLING LIST beside the matrix at indices 54-86 (the eight-arm reader loads the LIVE M5T_CELLS and
#: refuses a PSR cell / a `<key>_<view>` out root, so the rows are never appended to it)
SEEDREP_ROWS: list[dict] = _build_seedrep_rows(len(M5T_CELLS), _SEEDREP_PINS, _STEER_ADMISSION)
#: EVERY row the LAUNCHER resolves — `row --index N`, `row_for`, `plan`, `eligible_rows`, `assert_launchable`, `list` read THIS;
#: a row's index is its Slurm array id whichever list it sits in (the two are disjoint and consecutive)
LAUNCHER_ROWS: list[dict] = M5T_CELLS + SEEDREP_ROWS
assert [r["index"] for r in LAUNCHER_ROWS] == list(range(len(LAUNCHER_ROWS)))
WAVES = tuple(dict.fromkeys(r["wave"] for r in LAUNCHER_ROWS))


def key_view(row: dict) -> str:
    """The view a row is KEYED by (E3): an it-doc row's own (`it` — it shares its (model, cell, mode) with the main view's row of
    that cell and is a SECOND row of the triple); the all-positions arm's `it` likewise; every other row `main`. The
    DISCRIMINATOR is whether the row keeps the PANEL view and the object-BANK view apart: a row that carries a separate
    `bank_view` (the GSM8K launcher's spelling, which the all-positions row follows) puts its panel slot in `view`, so `view`
    keys it. A steering row COLLAPSES the two — its `view` field names the object bank's directory (`it`, `ar_r0p05`, …), a
    per-row identity key on a (model, cell, mode) that exists once — so it keys `main`, and keying it apart would make
    `row_for(key, alphasteer, mode)` (the launcher's positional form) resolve nothing."""
    if row.get("object_source") == OBJECT_SOURCE_ITDOC or row.get("bank_view"):
        return str(row["view"])
    return VIEW_MAIN


def row_for(model_key: str, cell: str, sampling_mode: str, rows: list[dict] = LAUNCHER_ROWS, view: str = VIEW_MAIN) -> dict:
    """The row of (model, cell, sampling mode) keyed on `view` (:func:`key_view`) — `main` by default (every row before E3, the
    steering rows included); an it-doc row (E3) is reached by `view=it`, a seed row by its lane view `s<s>` / `s<s>p` (the
    positional `<key> <cell> <mode> [<view>]`), over every row the launcher resolves (LAUNCHER_ROWS)."""
    for r in rows:
        if (r["model_key"], r["cell"], r["sampling_mode"], key_view(r)) == (model_key, cell, sampling_mode, view):
            return r
    raise SystemExit(f"not a matrix row: ({model_key!r}, {cell!r}, {sampling_mode!r}, view {view!r}) — the {len(rows)} rows are "
                     f"{[(r['model_key'], r['cell'], r['sampling_mode'], key_view(r)) for r in rows]}")


def rows_of_wave(wave: str, rows: list[dict] = LAUNCHER_ROWS) -> list[dict]:
    return [r for r in rows if r["wave"] == wave]


def _counts(rows: list[dict]) -> dict:
    fitted = [r for r in rows if r["fitted"]]
    pins = _ALL_PINS
    modes = SAMPLING_MODES + ((SAMPLING_CARD,) if any(r["sampling_mode"] == SAMPLING_CARD for r in rows) else ())
    return {
        "rows": len(rows), "fitted": len(fitted),
        # the three thinking modes always; the no-switch family's `card` mode only where a row runs it (the published
        # acceptance numbers keep their shape)
        "sampling": {m: sum(r["sampling_mode"] == m for r in rows) for m in modes},
        "cells": {c: sum(r["cell"] == c for r in rows) for c in ALL_CELLS if any(r["cell"] == c for r in rows)},
        "reasoning_parser_gemma4": sum(r["reasoning_parser"] == "gemma4" for r in rows),
        # a fitted row with NO tuple is IN SUPPORT (a no-switch family's steering / llama-wave row: one render, nothing to
        # authorise) — counted apart, never sorted against a name
        "tuple_names": sorted({r["tuple_name"] for r in fitted if r["tuple_name"]}),
        "fitted_in_support": sum(1 for r in fitted if not r["tuple_name"]),
        "distinct_out_roots": len({r["out_root"] for r in rows}),
        "pins": {f"{k}/{sid}": sum(r["object_sha256"] == v for r in rows) for (k, sid), v in pins.items()
                 if any(r["object_sha256"] == v for r in rows)},
        # the EXTERNAL objects, by (model, cell) — the admission record's pins, never the pin file's
        "external_pins": {f"{r['model_key']}/{r['cell']}": r["object_sha256"] for r in rows
                          if r.get("object_source") == "external"},
        # E3: the it-doc rows' pins by the FULL tuple (a cell id is not unique across lanes); null = held, not yet verified on disk
        "itdoc_pins": {f"{r['lane']}/{r['model_key']}/{r['view']}/{r['cell']}": r["object_sha256"] for r in rows
                       if r.get("object_source") == OBJECT_SOURCE_ITDOC},
        "geometry": {ev: sum(r["geometry_eval"] == ev for r in rows) for ev, _ in GEOMETRY_BY_CLASS.values()},
        "budget_classes": {c: sum(r["budget_class"] == c for r in rows) for c in dict.fromkeys(r["budget_class"] for r in rows)},
        "log_requests": sum(bool(r["log_requests"]) for r in rows),
        "waves": {w: sum(r["wave"] == w for r in rows) for w in dict.fromkeys(r["wave"] for r in rows)},
        "statuses": {s: sum(r["status"] == s for r in rows) for s in dict.fromkeys(r["status"] for r in rows)},
        "families": sorted({r["family"] for r in rows}),
        "models": [k for k in MODEL_KEYS + STEER_ONLY_MODELS if any(r["model_key"] == k for r in rows)],
    }


def counts(rows: list[dict] = M5T_CELLS) -> dict:
    """The acceptance counts, DERIVED from the matrix (the test compares the dry run's greps to these). Over the full
    matrix the published (`base`) and `extension` counts are reported APART so the published run's acceptance numbers
    keep their meaning."""
    out = _counts(rows)
    if rows is M5T_CELLS:
        out["base"] = _counts(BASE_ROWS)
        out["extension"] = _counts(EXTENSION_ROWS)
        out["steer"] = _counts(STEER_ROWS)
        out["llama"] = _counts(LLAMA_ROWS)
        out["itdoc"] = _counts(ITDOC_ROWS)
        out["allpos"] = _counts(ALLPOS_ROWS)
        out["seedrep"] = _counts(SEEDREP_ROWS)           # the sibling list, reported apart (never inside the matrix's totals)
    return out


def row_sh(row: dict) -> str:
    """The row as shell assignments the sbatch evals (positional facts only; the sampling row rides the driver's
    --sampling MODE — the numbers live in contracts' tables and this file's mirror, never in argv)."""
    comps = row_companions(row)
    lines = [
        f"M5T_INDEX={row['index']}", f"M5T_JOB_ID={shlex.quote(row['job_id'])}", f"KEY={shlex.quote(row['model_key'])}",
        f"CELL={shlex.quote(row['cell'])}", f"CELLSLUG={shlex.quote(row['cell_slug'])}",
        f"SAMPLING_MODE={shlex.quote(row['sampling_mode'])}", f"FITTED={1 if row['fitted'] else 0}",
        f"M5T_TUPLE={shlex.quote(row['tuple_name'] or '')}", f"M5T_ROUTE_KIND={row['route_kind']}",
        f"M5T_FAMILY={row['family']}", f"M5T_TIME={row['time_limit']}",
        f"M5T_OUT_REL={shlex.quote(row['out_root'])}", f"M5T_REASONING_PARSER={shlex.quote(row['reasoning_parser'] or '')}",
        f"M5T_GPU_MEM_UTIL={row['gpu_mem_util']}", f"PIN={shlex.quote(row['object_sha256'] or '')}",
        f"M5T_HARNESS_CELL={shlex.quote(row['harness_cell'])}",
        f"M5T_SAMPLING_SOURCE={shlex.quote(row['sampling']['source'])}",
        "M5T_SAMPLING_PARAMS=" + shlex.quote(json.dumps(sampling_params(row['sampling']), sort_keys=True)),
        f"M5T_BUDGET_CLASS={shlex.quote(row['budget_class'])}",
        f"M5T_SMOKE_OUT_REL={shlex.quote(row['smoke_diagnostic_out'])}",
        # the extension's per-row facts: the wave, the status, the thinking mode, the view, the geometry the
        # standup is given, the staged id, the SI digest, the sampler read-back opt-in (the placement constraint is the
        # planner's — `cells.py plan` — and rides no job)
        f"M5T_WAVE={shlex.quote(row['wave'])}", f"M5T_STATUS={shlex.quote(row['status'])}", f"M5T_HELD_REASON={shlex.quote(str(row.get('held_reason') or ''))}",
        f"M5T_THINKING_MODE={shlex.quote(row['thinking_mode'])}", f"M5T_VIEW={shlex.quote(row['view'])}",
        # the PANEL view the row is KEYED by (key_view) and, where the two are kept apart, the LANE's bank view. The
        # launcher dispatches its overlay pair on the KEY view (a row on a non-main panel view is cut from that view's
        # campaign config), never on `view`: a steering row's `view` is `ar_r0p05` / `it` and names its object bank, and
        # cutting a `gemma4e4b_it` overlay for it would serve another target's config.
        f"M5T_KEY_VIEW={shlex.quote(key_view(row))}",
        f"M5T_BANK_VIEW={shlex.quote(str(row.get('bank_view') or ''))}",
        f"M5T_GEOMETRY_EVAL={shlex.quote(row['geometry_eval'])}", f"M5T_GEOMETRY_L={row['geometry_L']}",
        f"M5T_STAGED_ID={shlex.quote(row['staged_id'] or '')}", f"M5T_SI_SHA256={shlex.quote(row['si_sha256'] or '')}",
        f"M5T_LOG_REQUESTS={1 if row['log_requests'] else 0}",
        # the steering row's OWN binding (empty on every other row, so the shell's `[ -n ... ]` branch is the ONE
        # discriminator): the object source, the lane's bank view RELATIVE to the shared root for THIS cluster role,
        # the cell dir inside it, and the selection label the record carries verbatim
        f"M5T_OBJECT_SOURCE={shlex.quote(str(row.get('object_source') or ''))}",
        # (no serving-path literal here since 2026-09-15: the pin FOLLOWS THE STORE at the row's out root on the cluster, so the sbatch
        # reads it through `row --serving-path --wave-root` BEFORE f22_init — m5t_select_serving_path — never off the row alone)
        f"M5T_BANK_VIEW_REL_PRIMARY={shlex.quote(str((row.get('bank_view_rel') or {}).get('primary') or ''))}",
        f"M5T_BANK_VIEW_REL_OVERFLOW={shlex.quote(str((row.get('bank_view_rel') or {}).get('overflow') or ''))}",
        f"M5T_BANK_CELL_DIR={shlex.quote(str(row.get('bank_cell_dir') or ''))}",
        f"M5T_SELECTION_LABEL={shlex.quote(str(row.get('selection_label') or ''))}",
        # E3: the it-doc row's binding facts — the lane, the object kind (which sidecar proves it), the admission record the bank
        # door re-reads (its ruled staged id rides M5T_STAGED_ID), the pair's qualification (logged beside every leg)
        f"M5T_LANE={shlex.quote(str(row.get('lane') or ''))}",
        f"M5T_OBJECT_KIND={shlex.quote(str(row.get('object_kind') or ''))}",
        f"M5T_ITDOC_ADMISSION={shlex.quote(str(ITDOC_ADMISSION_REL) if row.get('object_source') == OBJECT_SOURCE_ITDOC else '')}",
        f"M5T_QUALIFICATION={shlex.quote(str(row.get('qualification') or ''))}",
        # the SEED row's facts — the own overlay pair, the opt-in replicate seed rule + base, the standup companions
        # (`<cell>` / `<cell>=<pin>` space-joined) and the `--bank-view` stamp. ALL SIX are EMPTY on every other row
        # (test-pinned), so every other row's argv is byte-identical: the shell's `[ -n … ]` branches are the discriminator
        f"M5T_OWN_OVERLAY={shlex.quote('1' if row.get('own_overlay') else '')}",
        f"M5T_INSPECT_SEED_POLICY={shlex.quote(str(row.get('inspect_seed_policy') or ''))}",
        f"M5T_INSPECT_SEED_BASE={shlex.quote('' if row.get('inspect_seed_base') is None else str(row['inspect_seed_base']))}",
        "M5T_STANDUP_COMPANION_CELLS=" + shlex.quote(" ".join(c for c, _h, _sid, _pin in comps)),
        "M5T_STANDUP_COMPANION_PINS=" + shlex.quote(" ".join(f"{c}={pin}" for c, _h, _sid, pin in comps)),
        f"M5T_BANK_VIEW_STAMP={shlex.quote(str(row.get('bank_view_stamp') or ''))}",
    ]
    lines += [f"{k}={shlex.quote(str(v))}" for k, v in RUN_LITERALS_SH]
    return "\n".join(lines) + "\n"


def is_full_sha(s: object) -> bool:
    """A full 40-hex lowercase commit sha — the only pin spelling the launcher accepts (`plan --sha`, `row --identity --sha`)."""
    return isinstance(s, str) and len(s) == 40 and all(ch in "0123456789abcdef" for ch in s)


# --------------------------------------------------------------------------- the row-identity stamp (A2; fix A3)
#: IFEval's row-identity superset + the per-row `pin`: the keys every instrument's rows carry — plus `client_lock_sha256`:
#: the digest of the client lock the leg RAN UNDER, read from the DRIVER's manifest (`frameworks.client_lock_sha256`, the
#: driver hashes its checkout's lock) and held equal to the digest of the launch checkout's lock (this tree's) — so a leg
#: whose store says one lock while its tree carries another is `complete: false` by name, and the analysis's client-lock
#: disclosure reads the digest off the stamp as well as the manifest. Schema 3.
ROW_IDENTITY_KEYS = ("object_artifact_sha256", "route_kind", "view", "producer_tree_sha", "thinking", "sampling_mode",
                     "sampling_ref", "pin", "client_lock_sha256")
ROW_IDENTITY_SCHEMA = "m5t_row_identity/3"
#: the driver's manifest key + the stamp's top-level key for a cell whose predicate (C) was admitted by the CONCLUSIVE
#: logprob liveness verdict (== contracts.MANIFEST_KEY_LENS_ON_BITE_LIVENESS, pinned equal by test; stdlib-only here);
#: present on the stamp ONLY when the manifest carries the block
MANIFEST_KEY_LENS_ON_BITE_LIVENESS = "lens_on_bite_liveness_evidence"
ROW_IDENTITY_LIVENESS_KEY = "liveness_evidence"
CLIENT_LOCK_REL = "configs/capeval/tinycap-client-requirements.txt"     # == contracts.CLIENT_LOCK_TXT (pinned equal by test; stdlib-only here)
MANIFEST_FRAMEWORKS_KEY = "frameworks"
#: where the driver's manifest binds the leg to an OBJECT: the route block's scalar `object_artifact_sha256`
#: (== contracts.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256; a lens standup serving ONE cell writes the served digest there) and
#: its per-cell map `object_artifact_sha256_by_cell` (== contracts.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL,
#: `{harness cell id: sha256(object.pt)}` for every fitted cell the standup registered — a NATIVE LoRA standup serving the
#: whole bank (the seed-replicate wave) writes the scalar None and every cell in the map, so the leg's own digest sits
#: under `manifest["harness_cell_id"]`), plus the `proxy_identity.object_artifact_sha256_by_cell` fallback the GSM8K
#: pairing reader (sisteer.eval.anchor_pairing) accepts. Pinned equal to the contracts' spellings by test; stdlib-only here.
MANIFEST_KEY_ROUTE = "route"
MANIFEST_KEY_HARNESS_CELL_ID = "harness_cell_id"
MANIFEST_KEY_PROXY_IDENTITY = "proxy_identity"
ROUTE_KEY_OBJECT_ARTIFACT_SHA256 = "object_artifact_sha256"
ROUTE_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL = "object_artifact_sha256_by_cell"
#: the `checks.object_artifact_sha256_source` vocabulary of the stamp: which of the three places named the digest
OBJECT_SHA_SOURCE_ROUTE = "route"
OBJECT_SHA_SOURCE_ROUTE_BY_CELL = "route.by_cell"
OBJECT_SHA_SOURCE_PROXY_BY_CELL = "proxy_identity.by_cell"


def manifest_object_artifact_sha256(m: dict) -> tuple[str | None, str | None]:
    """`(sha, source)`: the object digest the driver's manifest binds the leg to — the ONE rule the identity stamp compares
    the row's `object_sha256` against (mirrors the GSM8K pairing reader's by-cell rule). The scalar
    `route.object_artifact_sha256` wins when it is a non-empty string (source `route`); else the per-cell map
    `route.object_artifact_sha256_by_cell` under the manifest's OWN `harness_cell_id` (source `route.by_cell` — the native
    whole-bank standup's shape, where the scalar is None because the proxy serves many objects); else
    `proxy_identity.object_artifact_sha256_by_cell` under the same id (source `proxy_identity.by_cell`); `(None, None)`
    when none names a digest. A map entry is read ONLY under the manifest's recorded harness id — a manifest that carries
    a map but no `harness_cell_id` resolves None (never a guess from the row), and a `none` / prompted row's manifest,
    whose harness id no map carries, resolves None too (agreeing with the row's None as before)."""
    route = m.get(MANIFEST_KEY_ROUTE) if isinstance(m.get(MANIFEST_KEY_ROUTE), dict) else {}
    scalar = route.get(ROUTE_KEY_OBJECT_ARTIFACT_SHA256)
    if isinstance(scalar, str) and scalar:
        return scalar, OBJECT_SHA_SOURCE_ROUTE
    harness = m.get(MANIFEST_KEY_HARNESS_CELL_ID)
    if harness is None:
        return None, None
    proxy_identity = m.get(MANIFEST_KEY_PROXY_IDENTITY) if isinstance(m.get(MANIFEST_KEY_PROXY_IDENTITY), dict) else {}
    for block, source in ((route, OBJECT_SHA_SOURCE_ROUTE_BY_CELL), (proxy_identity, OBJECT_SHA_SOURCE_PROXY_BY_CELL)):
        by_cell = block.get(ROUTE_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL)
        if isinstance(by_cell, dict):
            sha = by_cell.get(str(harness))
            if isinstance(sha, str) and sha:
                return sha, source
    return None, None


def producer_tree_client_lock_sha256(tree_sha: str, repo: Path = REPO) -> dict:
    """The digest of the client lock AT THE LEG'S PRODUCER TREE (`git show <tree>:<lock>` from this checkout's object
    store — at launch the checkout IS that tree; an analysis-time re-derivation from another checkout still reads the
    right tree), beside the digest of THIS checkout's working lock file (informational: equal to the tree's on a clean
    launch). `tree_lock_sha256` None + `why` when the tree cannot be read here (a fake sha, no git) — recorded, never
    invented."""
    import hashlib
    import subprocess
    rec: dict = {"tree": tree_sha, "tree_lock_sha256": None, "checkout_lock_sha256": None, "why": None,
                 "how": f"sha256(git show <producer tree>:{CLIENT_LOCK_REL}); the checkout's working file beside it"}
    p = Path(repo) / CLIENT_LOCK_REL
    if p.is_file():
        rec["checkout_lock_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        r = subprocess.run(["git", "-C", str(repo), "show", f"{tree_sha}:{CLIENT_LOCK_REL}"], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        rec["why"] = f"git show failed ({type(e).__name__}: {e})"
        return rec
    if r.returncode != 0:
        rec["why"] = f"git show {tree_sha[:12]}:{CLIENT_LOCK_REL} failed ({r.stderr.decode('utf-8', 'replace').strip()[:160]})"
        return rec
    rec["tree_lock_sha256"] = hashlib.sha256(r.stdout).hexdigest()
    return rec
#: the producer's shape (served_capability_run): the digest of `manifest["sampling"]` is NOT a manifest field — the
#: driver computes it from that block and stamps it on gen_stats (per task) and on every outcome item / row. The stamp
#: therefore DERIVES it from the block through the producer's own API and cross-checks the two places it was stamped.
MANIFEST_SAMPLING_KEY = "sampling"
GEN_STATS_FILENAME = "gen_stats.json"
OUTCOMES_DIRNAME = "outcomes"


def row_identity(row: dict, *, producer_tree_sha: str, sampling_ref: str | None = None,
                 client_lock_sha256: str | None = None) -> dict:
    """The row's identity block in the canonical superset shape: `pin` IS the producer tree (the wave sha the leg ran
    at — the analysis admits the leg under it); `sampling_ref` is the digest of the DRIVER's sampling block (derived
    from its manifest through contracts.sampling_ref and cross-checked, see sampling_ref_evidence; None when the
    evidence is absent or disagrees — never a value the evidence does not support); `client_lock_sha256` is the digest
    of the client lock the leg RAN UNDER (the driver's manifest, held to the launch checkout's lock — see
    row_identity_record; None when the evidence is absent or disagrees, never the checkout's digest alone)."""
    if not is_full_sha(producer_tree_sha):
        raise SystemExit(f"row_identity: producer_tree_sha must be a full 40-hex sha, got {producer_tree_sha!r}")
    return {
        "object_artifact_sha256": row["object_sha256"], "route_kind": row["route_kind"], "view": row["view"],
        "producer_tree_sha": producer_tree_sha, "thinking": row["thinking_mode"], "sampling_mode": row["sampling_mode"],
        "sampling_ref": sampling_ref, "pin": producer_tree_sha, "client_lock_sha256": client_lock_sha256,
    }


def contracts_sampling_ref(block: dict) -> str:
    """`contracts.sampling_ref(block)` — the PRODUCER's digest, imported lazily (this module stays stdlib-only at import
    time: the sbatch reads the row before any venv exists; the stamp runs AFTER the driver, from the client venv, whose
    guard imports the contracts). Not importable = a refusal by name, never a re-typed digest."""
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from sisteer.capeval.tinycap import contracts as C
    except Exception as e:  # noqa: BLE001 — named, never a fallback
        raise SystemExit(f"row --identity: the producer's contracts (sisteer.capeval.tinycap.contracts) do not import from "
                         f"{sys.executable} ({type(e).__name__}: {e}) — the stamp derives sampling_ref through "
                         "contracts.sampling_ref and never re-types the digest; run it from the client venv") from None
    return C.sampling_ref(block)


def sampling_ref_evidence(manifest_path: str, man: dict, control_task: str) -> dict:
    """What the driver's store says the leg's `sampling_ref` is, from the THREE places the producer stamps it: derived
    from `manifest["sampling"]` (the block the driver hashed), gen_stats' per-task ref, and the set of refs on the
    outcome items. `agree` is True only when the block exists, gen_stats names the control task's ref, every outcome
    file's items carry exactly one ref, and all of them equal the derived digest; `problems` names each gap."""
    d = os.path.dirname(manifest_path)
    problems: list[str] = []
    block = man.get(MANIFEST_SAMPLING_KEY)
    derived = None
    if isinstance(block, dict) and block:
        derived = contracts_sampling_ref(block)
    else:
        problems.append(f"the driver's manifest carries no `{MANIFEST_SAMPLING_KEY}` block — no digest can be derived")
    gs_path = os.path.join(d, GEN_STATS_FILENAME)
    gen_stats: dict = {}
    if os.path.isfile(gs_path):
        gs = json.loads(Path(gs_path).read_text(encoding="utf-8"))
        gen_stats = {task: (v or {}).get("sampling_ref") for task, v in gs.items() if isinstance(v, dict)}
        if gen_stats.get(control_task) is None:
            problems.append(f"{GEN_STATS_FILENAME} carries no sampling_ref for {control_task!r}")
    else:
        problems.append(f"no {GEN_STATS_FILENAME} beside the manifest")
    outcomes: dict = {}
    oc_dir = os.path.join(d, OUTCOMES_DIRNAME)
    oc_files = sorted(glob.glob(os.path.join(oc_dir, "*.json"))) if os.path.isdir(oc_dir) else []
    if not oc_files:
        problems.append(f"no {OUTCOMES_DIRNAME}/*.json beside the manifest")
    for f in oc_files:
        doc = json.loads(Path(f).read_text(encoding="utf-8"))
        items = doc.get("items") or []
        refs = sorted({str(it.get("sampling_ref")) for it in items})
        outcomes[os.path.basename(f)] = {"n_items": len(items), "refs": refs}
        if not items:
            problems.append(f"{OUTCOMES_DIRNAME}/{os.path.basename(f)}: no items")
        elif refs != [derived]:
            problems.append(f"{OUTCOMES_DIRNAME}/{os.path.basename(f)}: item refs {refs} != the derived {derived}")
    for task, ref in gen_stats.items():
        if ref is not None and ref != derived:
            problems.append(f"{GEN_STATS_FILENAME}[{task}].sampling_ref {ref} != the derived {derived}")
    return {"derived_from_manifest_block": derived, "gen_stats": gen_stats, "outcomes": outcomes,
            "agree": not problems, "problems": problems,
            "how": "contracts.sampling_ref(manifest['sampling']) — the producer's digest; cross-checked against gen_stats and every outcome item"}


def row_identity_record(row: dict, *, producer_tree_sha: str, out_root: str | None = None) -> dict:
    """The stamp the sbatch writes to `<out_root>/row_identity.json`: the identity block + the row's matrix facts + a
    cross-check against the driver's store (the manifest under `<out_root>/cells/*/manifest.json`, its gen_stats and
    outcomes): `sampling_ref` DERIVED from the manifest's sampling block through contracts.sampling_ref and held to
    gen_stats + the outcome items; the manifest's object sha (through manifest_object_artifact_sha256: the route scalar,
    else the per-cell map under the manifest's harness id) / thinking mode / producer tree / cell compared with the
    row's. `complete` is True only when every check agrees — a leg whose driver wrote no manifest, or a manifest
    lacking sampling evidence, or a disagreement, gets `complete: false`, `sampling_ref: null` and the reasons in
    `problems`: never a success-shaped stamp."""
    manifests = sorted(glob.glob(os.path.join(out_root, "cells", "*", "manifest.json"))) if out_root else []
    sampling_ref = None
    client_lock_sha256 = None
    problems: list[str] = []
    checks: dict = {"manifest": None}
    if manifests:
        m = json.loads(Path(manifests[0]).read_text(encoding="utf-8"))
        evidence = sampling_ref_evidence(manifests[0], m, f"inspect/{INSPECT_TASK}")
        # the client lock the leg RAN UNDER: the driver's manifest records the digest of ITS checkout's lock;
        # the lock at the leg's PRODUCER TREE must digest the same (the chain of custody the analysis's client-lock
        # disclosure relies on) — a store that says one lock under a tree that carries another is named, never stamped
        # complete; a tree this checkout cannot read leaves `agree: None` (recorded, not a problem: the driver's own
        # record still says what it ran under); a manifest with NO digest is a problem (every driver writes one)
        recorded_lock = (m.get(MANIFEST_FRAMEWORKS_KEY) or {}).get("client_lock_sha256")
        tree_lock = producer_tree_client_lock_sha256(producer_tree_sha)
        lock_evidence = {"manifest_frameworks_client_lock_sha256": recorded_lock, **tree_lock,
                         "agree": (None if tree_lock["tree_lock_sha256"] is None or not recorded_lock
                                   else recorded_lock == tree_lock["tree_lock_sha256"])}
        # the manifest's object digest through the ONE resolver (the scalar, else the per-cell map under the manifest's
        # harness id — the native whole-bank standup writes the scalar None; comparing that alone read every fitted
        # seed-replicate leg as `object_artifact_sha256_agrees: false` on a correct store); the resolved value and its
        # source ride the checks so a reader sees WHICH place named it
        manifest_sha, sha_source = manifest_object_artifact_sha256(m)
        agrees = {
            "object_artifact_sha256_agrees": (manifest_sha == row["object_sha256"]),
            "thinking_mode_agrees": ((m.get("thinking") or {}).get("mode") == row["thinking_mode"]),
            "producer_tree_agrees": ((m.get("producer") or {}).get("tree_sha") == producer_tree_sha),
            "cell_agrees": (m.get("cell_id") == row["cell"]),
        }
        checks = {"manifest": os.path.relpath(manifests[0], out_root), "n_manifests": len(manifests),
                  "sampling_ref_evidence": evidence, "client_lock_evidence": lock_evidence,
                  "manifest_object_artifact_sha256": manifest_sha, "object_artifact_sha256_source": sha_source,
                  "manifest_harness_cell_id": m.get(MANIFEST_KEY_HARNESS_CELL_ID), **agrees}
        if len(manifests) != 1:
            problems.append(f"{len(manifests)} manifests under {out_root}/cells — one leg has one store")
        problems += [f"sampling_ref: {p}" for p in evidence["problems"]]
        if not recorded_lock:
            problems.append(f"client_lock_sha256: the driver's manifest carries no {MANIFEST_FRAMEWORKS_KEY}.client_lock_sha256 — the lock the leg ran under is unrecorded")
        elif lock_evidence["agree"] is False:
            problems.append(f"client_lock_sha256: the driver's manifest records {recorded_lock} but the producer tree {producer_tree_sha[:12]}'s "
                            f"{CLIENT_LOCK_REL} digests {tree_lock['tree_lock_sha256']} — the leg did not run under its tree's lock")
        problems += [f"{k}: the driver's manifest disagrees with the row"
                     + (f" (manifest {manifest_sha} via {sha_source or 'no route.object_artifact_sha256, no by-cell entry under the manifest harness id'}, "
                        f"row {row['object_sha256']})" if k == "object_artifact_sha256_agrees" else "")
                     for k, ok in agrees.items() if not ok]
        if not problems:
            sampling_ref = evidence["derived_from_manifest_block"]
            client_lock_sha256 = recorded_lock
    else:
        why = ("no cells/*/manifest.json under the out root" + (f" {out_root}" if out_root else " (none given)")
               + " — the driver wrote no manifest (a failed or unfinished leg)")
        checks["why"] = why
        problems.append(why)
    # a cell whose predicate (C) was admitted by the CONCLUSIVE logprob liveness verdict (no token bite observable at
    # its dose) carries that fact on the row — copied from the driver's manifest block, present ONLY when the manifest
    # carries it (a cell with a token bite writes none; the stamp is unchanged)
    liveness = (m.get(MANIFEST_KEY_LENS_ON_BITE_LIVENESS) if manifests else None)
    return {"schema": ROW_IDENTITY_SCHEMA, "complete": not problems, "problems": problems,
            "identity": row_identity(row, producer_tree_sha=producer_tree_sha, sampling_ref=sampling_ref, client_lock_sha256=client_lock_sha256),
            "row": {k: row[k] for k in ("index", "job_id", "model_key", "cell", "harness_cell", "sampling_mode", "wave",
                                        "status", "geometry_eval", "geometry_L", "budget_class", "staged_id", "tuple_name",
                                        "si_sha256", "log_requests", "out_root")},
            "checks": checks,
            **({ROW_IDENTITY_LIVENESS_KEY: dict(liveness)} if isinstance(liveness, dict) and liveness else {})}


def time_classes(rows: list[dict] = LAUNCHER_ROWS) -> dict[str, list[int]]:
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


#: the clusters a plan can name, with the placement constraint each honours. Each one's QoS and log directory come from its block
#: in the site config (`cluster_site`, read when a plan is printed): the job scripts' own headers name one site's values
CLUSTERS = {PRIMARY_CLUSTER: {"constraint": CLUSTER_CONSTRAINT_PRIMARY},
            OVERFLOW_CLUSTER: {"constraint": CLUSTER_CONSTRAINT_ANY}}
_CLUSTERS_LIB = None


def cluster_site(cluster: str) -> dict:
    """`{"qos", "logs", "constraint"}` of `cluster` (a registry label): its constraint from CLUSTERS, its QoS and log directory
    from its block in the site config this process uses (reproduce/lib/clusters.py: `SISTEER_CONFIG`, else the site directory's
    config.env), refused by name when the site config does not describe it. Loaded on first use, so the job-side subcommands
    never need a site config."""
    global _CLUSTERS_LIB
    if cluster not in CLUSTERS:
        raise SystemExit(f"cluster must be one of {sorted(CLUSTERS)}, got {cluster!r}")
    if _CLUSTERS_LIB is None:
        spec = importlib.util.spec_from_file_location("m5t_site_clusters", REPO / "reproduce" / "lib" / "clusters.py")
        _CLUSTERS_LIB = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_CLUSTERS_LIB)
    try:
        b = _CLUSTERS_LIB.block_for(cluster)
    except SystemExit as exc:
        raise SystemExit(f"REFUSING: {str(exc.code).removeprefix('REFUSED: ')}") from None
    return {"qos": b["CLUSTER_QOS"], "logs": b["CLUSTER_LOG_DIR"].rstrip("/"), **CLUSTERS[cluster]}

#: the checkout an emitted sbatch line names, spelled for the cluster shell that runs the line: the job's own
#: SISTEER_EVAL_REPO_ROOT (f22_init derives its checkout from the same variable), refused by name where it is not exported
CHECKOUT_ROOT = "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}"
CLASS_TAG = {TIME_BASE: "base", TIME_OBJECT: "object", TIME_OBJECT_GLM: "object-glm", TIME_EXT: "ext", TIME_LLAMA: "llama",
             TIME_SEEDREP: "seedrep"}
#: the per-user `high` budget on EACH cluster (GPUs, running + pending) — shared with every peer job under this user
HIGH_BUDGET_GPUS = 16


def time_limit_hours(tl: str) -> float:
    """Slurm's `D-HH:MM:SS` (the classes' spelling) -> hours."""
    days, _, hms = tl.partition("-")
    h, m, sec = (int(x) for x in hms.split(":"))
    return int(days) * 24 + h + m / 60 + sec / 3600


def parse_index_spec(spec: str, n: int = len(LAUNCHER_ROWS)) -> list[int]:
    """Slurm's array-index grammar (`0-7,12,15-23`) -> sorted distinct row indices in 0..n-1; anything else refuses
    by name (a repeated index, an index outside the matrix, a non-index)."""
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


def eligible_rows(wave: str, universe: list[dict] = LAUNCHER_ROWS) -> list[dict]:
    """The rows a submission of `wave` may carry: that wave's `status: launch` rows (plan P2 / F11e). An unknown wave,
    a COMPLETED wave (the published run — its legs are reused, never re-run; A12) or a wave with nothing launchable
    refuses by name."""
    known = tuple(dict.fromkeys(r["wave"] for r in universe))
    if wave not in known:
        raise SystemExit(f"--wave {wave!r}: not a wave of this matrix (waves: {list(known)})")
    if wave in COMPLETED_WAVES:
        raise SystemExit(f"--wave {wave!r}: this wave is COMPLETE (every leg reused byte-for-byte from its published "
                         f"tree) and is never resubmitted; the launchable waves are "
                         f"{[w for w in known if w not in COMPLETED_WAVES]}")
    out = [r for r in universe if r["wave"] == wave and r["status"] == STATUS_LAUNCH]
    if not out:
        statuses = sorted({r["status"] for r in universe if r["wave"] == wave})
        raise SystemExit(f"--wave {wave!r}: no launchable row (its rows are {statuses}) — a held or deferred row is "
                         "never submitted")
    return out


def _refuse_outside_wave(indices: list[int], wave: str, eligible: list[dict], universe: list[dict], flag: str) -> None:
    ok = {r["index"] for r in eligible}
    by_index = {r["index"]: r for r in universe}
    bad = [i for i in indices if i not in ok]
    if bad:
        why = ", ".join(f"{i} ({by_index[i]['job_id']}: wave {by_index[i]['wave']}, status {by_index[i]['status']})" for i in bad)
        raise SystemExit(f"{flag}: row(s) outside wave {wave!r}'s eligible set: {why} — a published (complete), appendix, "
                         f"held or deferred row is never submitted; eligible: {_array_spec([r['index'] for r in eligible])}")


def _refuse_not_launch(indices: list[int], universe: list[dict], flag: str) -> None:
    """A selector without a wave names LAUNCH rows only: a held / deferred index refuses by name (E3: the it-doc rows are HELD by
    their record; the GSM8K launcher's `refuse_held`, the same rule)."""
    by_index = {r["index"]: r for r in universe}
    bad = [i for i in indices if by_index[i]["status"] != STATUS_LAUNCH]
    if bad:
        first = by_index[bad[0]]
        raise SystemExit(f"{flag} names HELD / deferred row(s) {[(i, by_index[i]['job_id'], by_index[i]['status']) for i in bad]} — "
                         f"{first.get('held_reason') or first.get('deferred_reason') or first['status']}; a held or deferred row is never submitted")


def _parse_indices_of(spec: str, universe: list[dict], flag: str) -> list[int]:
    """`spec` (Slurm index grammar) resolved against the MATRIX INDICES the universe carries (a row's index is its
    Slurm array id, whatever subset of rows the universe is) — an index no row of the universe carries refuses."""
    by_index = {r["index"]: r for r in universe}
    idx = parse_index_spec(spec, max(by_index) + 1)
    missing = [i for i in idx if i not in by_index]
    if missing:
        raise SystemExit(f"{flag} {spec}: index(es) {_array_spec(missing)} name no row of this selection's universe "
                         f"(its rows: {_array_spec(sorted(by_index))})")
    return idx


def select_rows(*, rows_spec: str | None = None, cluster_share: str | None = None, cluster: str | None = None,
                universe: list[dict] = LAUNCHER_ROWS, wave: str | None = None) -> list[dict]:
    """The rows THIS submission carries (their matrix indices intact: the ARRAY task resolves its row from
    SLURM_ARRAY_TASK_ID). With `wave` the universe of choice is that wave's ELIGIBLE set (eligible_rows): `--rows SPEC`
    must name a subset of it, `--cluster-share SPEC` names the PRIMARY cluster's subset of it and the same string on
    the overflow cluster selects the complement WITHIN it — never against the whole matrix (F11e). Without
    a wave (the python entry point the partition tests use) the universe is taken as given."""
    if rows_spec is not None and cluster_share is not None:
        raise SystemExit("--rows and --cluster-share are exclusive: name this cluster's rows, or the primary cluster's share")
    # without a wave the universe is taken as given — its LAUNCH rows (E3: a held it-doc row is never emitted, whatever the selector)
    eligible = eligible_rows(wave, universe) if wave is not None else [r for r in universe if r["status"] == STATUS_LAUNCH]
    by_index = {r["index"]: r for r in universe}          # a row's index is its Slurm array id, never its position here
    if rows_spec is not None:
        idx = _parse_indices_of(rows_spec, universe, "--rows")
        if wave is not None:
            _refuse_outside_wave(idx, wave, eligible, universe, "--rows")
        else:
            _refuse_not_launch(idx, universe, "--rows")
    elif cluster_share is not None:
        if cluster is None:
            raise SystemExit("--cluster-share needs the cluster (primary / overflow) whose side of the share this is")
        cluster = resolve_cluster(cluster)
        share = _parse_indices_of(cluster_share, universe, "--cluster-share")
        if wave is not None:
            _refuse_outside_wave(share, wave, eligible, universe, "--cluster-share")
        else:
            _refuse_not_launch(share, universe, "--cluster-share")
        pool = [r["index"] for r in eligible]
        idx = sorted(share) if cluster == PRIMARY_CLUSTER else sorted(set(pool) - set(share))
        if not idx:
            raise SystemExit(f"--cluster-share {cluster_share}: no rows are left over on {cluster}")
    else:
        idx = [r["index"] for r in eligible]
    return [by_index[i] for i in idx]


def assert_eligible(selected: list[dict], wave: str, universe: list[dict] = LAUNCHER_ROWS) -> dict:
    """Every selected row is in `wave`'s eligible set (status launch, that wave); refuses BY NAME otherwise. Returns the
    record the planner prints (`# assert-eligible: …`): the rows, and how many of the matrix were excluded and why."""
    # eligible_rows already restricts to `wave`'s `status: launch` rows and refuses a completed wave, and a row's index is
    # unique — so membership by index is the whole test
    eligible = {r["index"] for r in eligible_rows(wave, universe)}
    bad = [r for r in selected if r["index"] not in eligible]
    if bad:
        raise SystemExit("assert-eligible REFUSED: " + "; ".join(
            f"row {r['index']} ({r['job_id']}) wave={r['wave']} status={r['status']}" for r in bad))
    excluded = [r for r in universe if r["index"] not in {s["index"] for s in selected}]

    def _why(r: dict) -> str:       # the reason a row is NOT in this submission, as the record names it
        return "complete" if r["wave"] in COMPLETED_WAVES else r["status"] if r["status"] != STATUS_LAUNCH else "other-wave"
    return {"wave": wave, "rows": [r["index"] for r in selected],
            "excluded": {f"{w}/{s}": sum(r["wave"] == w and _why(r) == s for r in excluded)
                         for w, s in dict.fromkeys((r["wave"], _why(r)) for r in excluded)}}


def assert_launchable(row: dict, universe: list[dict] = LAUNCHER_ROWS) -> dict:
    """The BATCH BOUNDARY's eligibility predicate: the ONE row a job resolved — `m5t_row ARRAY` from
    SLURM_ARRAY_TASK_ID, or the documented positional `<key> <cell> <sampling_mode>` — is in ITS OWN wave's eligible
    set, by the same test `plan --assert-eligible` runs over a submission (:func:`assert_eligible` over
    :func:`eligible_rows`): a row of a COMPLETED wave (the published run, whose legs are reused byte-for-byte and never
    re-run), a held or a deferred row refuses BY NAME (SystemExit). Before it the job trusted the submission: an
    `--array=0` or `… gemma4e4b none thinking_card <sha> 0` resolved the published row, kept `status: launch`
    (published-ness is the WAVE, never a status) and would have run into the published output root — the budget
    re-check does not stop it, since a later wave refreshes the same model's `none` budget. Returns the one-row record."""
    return assert_eligible([dict(row)], str(row["wave"]), universe)


def launchable_line(row: dict, record: dict) -> str:
    """The one line the launcher logs when :func:`assert_launchable` admits a row."""
    excluded = ", ".join(f"{k}: {v}" for k, v in record["excluded"].items())
    return f"launchable: row {row['index']} ({row['job_id']}) wave={row['wave']} status={row['status']}; excluded {excluded}"


def partition_widths(selected: list[dict], max_concurrent: int) -> dict[str, int]:
    """The AGGREGATE allowance W split into per-array `%w` (CLUSTER P2): sum(w) = min(W, rows) <= W, every class
    present keeps >= 1, no w exceeds its class's rows. Slurm throttles each array on its own — a slot an array frees
    never migrates to another array — so the split minimises the wave's tail under exactly that model, max over the
    classes of ceil(rows / w) x time, by exhaustive search (a few classes: a few hundred candidates). Ties: the sum
    of the classes' finish times (rows done early), then the wider split, then the longest class first. At the
    planned W = 12 that is GLM object 2, object 6, base 4 — twelve in flight, never twenty."""
    if not (isinstance(max_concurrent, int) and max_concurrent >= 1):
        raise SystemExit(f"--max-concurrent must be a positive integer, got {max_concurrent!r}")
    classes = time_classes(selected)
    order = sorted(classes, key=lambda tl: -time_limit_hours(tl))          # the longest class first
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


def sbatch_script_path(cluster: str, sha: str) -> str:
    """The launcher at the checkout layout f22_init accepts and NOTHING else (F11f):
    `<shared root>/$USER/<eval repo checkout>/<full sha>/scripts/sdfbase/math500/math500_thinking.sbatch`."""
    resolve_cluster(cluster)                       # an unknown cluster still refuses by name
    return f"{CHECKOUT_ROOT}/{sha}/scripts/sdfbase/math500/math500_thinking.sbatch"


@dataclass
class Plan:
    """One cluster's submission: the sbatch lines (one per --time class present, longest first — or ONE array under
    --single-array) and the facts the submitter reads before pasting them — the rows, the per-array widths and their
    sum, the estimated tail, the wave and its eligibility record."""
    cluster: str
    sha: str
    max_concurrent: int
    rows: list[int]                 # matrix indices this cluster carries
    total_rows: int
    widths: dict[str, int]          # --time class -> %w, longest class first
    sizes: dict[str, int]           # --time class -> rows of it in this selection
    lines: list[str]
    wave: str | None = None
    single_array: bool = False
    eligibility: dict | None = None

    @property
    def est_tail_h(self) -> float:
        """The wave's tail under Slurm's static per-array throttles: max over classes of ceil(rows / w) x time."""
        return max(-(-self.sizes[tl] // w) * time_limit_hours(tl) for tl, w in self.widths.items())

    def summary(self) -> list[str]:
        c = cluster_site(self.cluster)
        used = sum(self.widths.values())
        per = ", ".join(f"{CLASS_TAG[tl]} {w}/{self.sizes[tl]} rows @{tl}" for tl, w in self.widths.items())
        role = next(r for r, n in CLUSTER_BY_ROLE.items() if n == self.cluster)
        out = [
            (f"# plan: cluster={self.cluster} ({role}) qos={c['qos']} wave={self.wave or 'ALL (no --wave: the whole matrix)'} "
             f"rows={len(self.rows)}/{self.total_rows} ({_array_spec(self.rows)}) "
             f"allowance={self.max_concurrent} = the AGGREGATE cap on tasks in flight (the per-array %w sum to it)"),
            (f"# widths, longest class first: {per} -> sum {used} <= {self.max_concurrent}; est. tail {self.est_tail_h:g} h "
             f"(each array throttles itself; a slot one array frees never moves to another)"
             + ("; --single-array: ONE array at the longest class's --time" if self.single_array else "")),
            (f"# budget: choose W = the FREE `{c['qos']}` budget: {HIGH_BUDGET_GPUS} GPUs per user on this cluster MINUS every "
             f"`{c['qos']}` job already running or pending under this user, peers' included (the Slurm user is shared): "
             f"squeue -u $USER -h -q {c['qos']} -t RUNNING,PENDING -o '%b' and sum the gpu counts before choosing W"),
        ]
        if self.eligibility is not None:
            e = self.eligibility
            out.append(f"# assert-eligible: wave={e['wave']} rows={_array_spec(e['rows'])} all status=launch; excluded "
                       + ", ".join(f"{k}: {v}" for k, v in e["excluded"].items())
                       + " — no published (complete), held or deferred row is named")
        return out


def plan(sha: str, cluster: str, *, max_concurrent: int, rows: str | None = None, cluster_share: str | None = None,
         universe: list[dict] = LAUNCHER_ROWS, wave: str | None = None, single_array: bool = False,
         assert_eligible_record: bool = False) -> Plan:
    """The submission, PRINTED never run: one `sbatch --array` line per --time class present in the selection (or ONE
    array under `single_array`, at the longest class's --time), over ONE script at the f22_init checkout layout, at the
    cluster's QoS, positional args only (ARRAY <sha>: the task resolves its row from SLURM_ARRAY_TASK_ID); the per-array
    `%w` partition ONE aggregate allowance (partition_widths). With `wave` the selection is wave-scoped and EVERY
    printed row is asserted eligible (a completed / held / deferred row refuses by name); `assert_eligible_record`
    adds the record line to the summary. A row whose cluster constraint the cluster cannot honour (Nemotron off the
    primary cluster; a seed-wave row off the OVERFLOW cluster) refuses by name. `cluster` is a role
    (`primary` / `overflow`) or the cluster's name."""
    cluster = resolve_cluster(cluster)
    c = cluster_site(cluster)
    selected = select_rows(rows_spec=rows, cluster_share=cluster_share, cluster=cluster, universe=universe, wave=wave)
    eligibility = None
    if wave is not None:
        eligibility = assert_eligible(selected, wave, universe)
        if not assert_eligible_record:
            eligibility = None
    misplaced = [r for r in selected if r["cluster_constraint"] == CLUSTER_CONSTRAINT_PRIMARY and c["constraint"] != CLUSTER_CONSTRAINT_PRIMARY]
    if misplaced:
        raise SystemExit(f"cluster {cluster}: row(s) {_array_spec([r['index'] for r in misplaced])} carry cluster_constraint "
                         f"primary (the Nemotron fused-kernel overlay is attested on the primary cluster alone) — name them "
                         "on the primary cluster's line, not this one")
    # the seed wave runs on the OVERFLOW cluster alone (never cluster A) — a code refusal
    misplaced = [r for r in selected if r["cluster_constraint"] == CLUSTER_CONSTRAINT_OVERFLOW and cluster != OVERFLOW_CLUSTER]
    if misplaced:
        raise SystemExit(f"cluster {cluster}: row(s) {_array_spec([r['index'] for r in misplaced])} "
                         f"({', '.join(r['job_id'] for r in misplaced[:4])}{', …' if len(misplaced) > 4 else ''}) carry cluster_constraint "
                         f"{CLUSTER_CONSTRAINT_OVERFLOW} (the qwen_seedrep wave runs on the overflow cluster alone — D65 / D72) — plan them "
                         f"with --cluster {CLUSTER_ROLE_OVERFLOW}, never here")
    classes = time_classes(selected)
    script = sbatch_script_path(cluster, sha)
    if single_array:
        if not (isinstance(max_concurrent, int) and max_concurrent >= 1):
            raise SystemExit(f"--max-concurrent must be a positive integer, got {max_concurrent!r}")
        longest = max(classes, key=time_limit_hours)
        w = min(max_concurrent, len(selected))
        widths = {longest: w}
        sizes = {longest: len(selected)}
        lines = [(f"sbatch -J m5t-{wave or 'all'} --qos={c['qos']} --time={longest} "
                  f"--output={c['logs']}/%x_%A_%a.out --array={_array_spec([r['index'] for r in selected])}%{w} "
                  f"{script} ARRAY {sha}")]
    else:
        widths = partition_widths(selected, max_concurrent)
        sizes = {tl: len(classes[tl]) for tl in widths}
        lines = [f"sbatch -J m5t-{CLASS_TAG[tl]} --qos={c['qos']} --time={tl} "
                 f"--output={c['logs']}/%x_%A_%a.out --array={_array_spec(classes[tl])}%{w} "
                 f"{script} ARRAY {sha}" for tl, w in widths.items()]
    # a submission whose EVERY row pins the lens BY KIND (AlphaSteer rows only) carries the lens route on its line too: the job pins it
    # per row regardless (m5t_select_serving_path), the prefix makes the planned line say what the standup will do. A submission with an
    # Arditi row gets NO prefix: that row's pin follows its STORE on the cluster (`store:lens` / `unpinned:no-store` / `store:native`),
    # which the grading host plan cannot read — a prefix would override an unpinned row as EXPLICIT (2026-09-15)
    if selected and all(serving_path_for(r) == SERVING_PATH_LENS for r in selected):
        lines = [f"F22_SERVING_PATH={SERVING_PATH_LENS} {line}" for line in lines]
    return Plan(cluster=cluster, sha=sha, max_concurrent=max_concurrent, rows=[r["index"] for r in selected],
                total_rows=len(universe), widths=widths, sizes=sizes, lines=lines, wave=wave, single_array=single_array,
                eligibility=eligibility)


def plan_lines(sha: str, cluster: str, *, max_concurrent: int, rows: list[dict] = LAUNCHER_ROWS, wave: str | None = None) -> list[str]:
    """The sbatch lines alone, over every row of `rows` (the CLI prints `plan(...)`: its summary first)."""
    return plan(sha, cluster, max_concurrent=max_concurrent, universe=rows, wave=wave).lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--json", action="store_true")
    p.add_argument("--published-keys", action="store_true",
                   help="rows 0-23 on the published tree's keys only (byte-identical to tests/fixtures/math500_matrix_55e3cbdd.json)")
    p = sub.add_parser("row"); p.add_argument("--index", type=int, default=None); p.add_argument("--key"); p.add_argument("--cell")
    p.add_argument("--sampling"); p.add_argument("--sh", action="store_true")
    p.add_argument("--serving-path", action="store_true",
                   help="print the serving path the row PINS and its reason, one per line (path empty = unpinned: the kind-aware default stands); "
                        "an external row's pin FOLLOWS ITS STORE at --wave-root/<out root>, so --wave-root is required for it")
    p.add_argument("--wave-root", default=None, help="the experiment's wave root on THIS cluster (F22_WAVE_ROOT), under which the row's store on record is read")
    p.add_argument("--store-pins", action="append", default=None, metavar="JSON",
                   help="the store-pins record(s) whose _meta.serving_engine_by_leg the pin reads for the row's admitted leg (default: the repo's "
                        "configs/sdfbase/math500/store_pins_default.json — the launcher never overrides this; a test seam)")
    p.add_argument("--launch-cells", action="store_true",
                   help="print the HARNESS cells the row's standup serves, one per line (`none` + the row's cell) — the sbatch declares them "
                        "as F22_LAUNCH_CELLS before f22_init (the kind-aware serving-path default reads them)")
    p.add_argument("--view", default=VIEW_MAIN, help=f"the row's view ({VIEW_MAIN} by default; {VIEW_IT} for an E3 it-doc row, whose (key, cell, sampling) the main view shares; "
                                                   f"a seed / paired view {SEEDREP_VIEWS + SEEDREP_PAIRED_VIEWS} for a W1 seed row)")
    p.add_argument("--identity", action="store_true", help="print the row-identity stamp (needs --sha; --out-root reads the driver's manifest)")
    p.add_argument("--assert-launchable", action="store_true",
                   help="the batch boundary's predicate (m5t_assert_launchable): exit 0 with the record iff the row is in its wave's "
                        "eligible set; a published (complete), held or deferred row refuses by name on stderr, exit 1")
    p.add_argument("--sha", default=None, help="the producer tree (wave sha) for --identity")
    p.add_argument("--out-root", default=None, help="the leg's out root, for --identity's manifest cross-check")
    sub.add_parser("counts")
    p = sub.add_parser("plan"); p.add_argument("--sha", required=True)
    p.add_argument("--cluster", required=True, help="the cluster's ROLE, primary | overflow (its own name is accepted too)")
    p.add_argument("--wave", required=True, help="the wave whose launchable rows this submission carries (ext1 for the eight-arm W1; "
                                                 "qwen_seedrep for the seed wave, --cluster overflow only); a matrix-wide plan would name "
                                                 "completed rows and is refused")
    p.add_argument("--assert-eligible", action="store_true", help="print the eligibility record (the assertion itself always runs)")
    p.add_argument("--single-array", action="store_true", help="ONE --array line at the longest class's --time (the per-class form is the default)")
    p.add_argument("--max-concurrent", type=int, default=12, help="the AGGREGATE cap on tasks in flight across the arrays (the free high budget)")
    p.add_argument("--rows", default=None, help="this cluster's rows (a subset of the wave's eligible set), Slurm index grammar (24-26,30)")
    p.add_argument("--cluster-share", default=None, help="the PRIMARY cluster's rows of the wave; the same string on the overflow cluster's invocation selects the complement WITHIN the wave")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        if a.published_keys:
            rows = [{k: r[k] for k in r if k in PUBLISHED_KEYS} for r in BASE_ROWS]
            print(json.dumps(rows, indent=1))
            return 0
        if a.json:
            print(json.dumps(LAUNCHER_ROWS, indent=1))
        else:
            for r in LAUNCHER_ROWS:
                print(f"{r['index']:2d} {r['job_id']:<38} {r['model_key']:<16} {r['cell']:<40} {r['sampling_mode']:<14} "
                      f"fitted={int(r['fitted'])} wave={r['wave']:<9} status={r['status']:<8} time={r['time_limit']} "
                      f"geom={r['geometry_eval']} out={r['out_root']}" + (f" view={r['view']}" if r.get("view") != VIEW_MAIN else ""))
        return 0
    if a.cmd == "row":
        if a.index is not None:
            if not 0 <= a.index < len(LAUNCHER_ROWS):
                raise SystemExit(f"--index {a.index} outside 0..{len(LAUNCHER_ROWS) - 1}")
            row = LAUNCHER_ROWS[a.index]
        else:
            if not (a.key and a.cell and a.sampling):
                raise SystemExit("row: pass --index N, or --key K --cell C --sampling S")
            row = row_for(a.key, a.cell, a.sampling, view=a.view)
        if a.assert_launchable:
            print(launchable_line(row, assert_launchable(row)))
            return 0
        if a.identity:
            if not a.sha:
                raise SystemExit("row --identity: pass --sha <the wave sha the leg ran at>")
            print(json.dumps(row_identity_record(row, producer_tree_sha=a.sha, out_root=a.out_root), indent=1, sort_keys=True))
            return 0
        if a.serving_path:
            if row.get("object_source") == OBJECT_SOURCE_EXTERNAL and not a.wave_root:
                raise SystemExit(f"row --serving-path: {row['job_id']} is an external-object row, whose pin FOLLOWS ITS STORE — pass "
                                 "--wave-root <this cluster's wave root> (the launcher passes F22_WAVE_ROOT); a store is never assumed absent")
            path, reason = serving_path_pin(row, wave_root=a.wave_root, store_pins=a.store_pins)
            sys.stdout.write(SP.pin_lines(path, reason))
            return 0
        if a.launch_cells:
            sys.stdout.write("\n".join(launch_cells_for(row)) + "\n")
            return 0
        sys.stdout.write(row_sh(row) if a.sh else json.dumps(row, indent=1) + "\n")
        return 0
    if a.cmd == "counts":
        print(json.dumps(counts(), indent=1))
        return 0
    if a.cmd == "plan":
        if not is_full_sha(a.sha):
            raise SystemExit(f"--sha must be a full 40-hex commit sha, got {a.sha!r}")
        pl = plan(a.sha, a.cluster, max_concurrent=a.max_concurrent, rows=a.rows, cluster_share=a.cluster_share,
                  wave=a.wave, single_array=a.single_array, assert_eligible_record=a.assert_eligible)
        for line in pl.summary() + pl.lines:
            print(line)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
