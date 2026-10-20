#!/usr/bin/env python
"""Generate the sdf-base-steering campaign FIT configs.

STANDALONE on purpose — scripts/make_fit_configs.py is NEVER invoked or
regenerated here: its qwen35/E2B lanes silently skip when INTERIM sources
are absent, and any byte change to an existing two_si_*_fits.yaml
invalidates the expect_sha + fit fingerprints of finished instruct cells.
This generator writes ONLY paths matching configs/sdfbase_*.yaml and
hard-refuses anything else (guarded_write); tests pin that running it
leaves every configs/two_si_*.yaml byte-identical.

Outputs (deterministic — same inputs, same bytes, so the fits' expect_sha
is stable across regenerations):

  configs/sdfbase_gemma4e4b_base_fits.yaml
  configs/sdfbase_qwen35_4b_base_fits.yaml
  configs/sdfbase_glm46vflash_doc_fits.yaml   (2026-09-02; the CHAT model —
                                               no base variant exists)
  configs/sdfbase_gemma4e4b_it_doc_fits.yaml  (2026-09-03; the
                                               INSTRUCT checkpoint trained
                                               DIRECTLY — mix-full only)
  configs/sdfbase_qwen35_4b_it_doc_fits.yaml  (2026-09-03; same)
  configs/sdfbase_nemotron3nano4b_doc_full_fits.yaml (2026-09-06; the INSTRUCT
                                               checkpoint — no base variant —
                                               at FULL scale: mix-full +
                                               ctl2-full, cap 16 / patience 2,
                                               run on Slurm; the paper's 17
                                               Nemotron objects. Its pool dir
                                               is the retired data-scaling
                                               ladder's, nemotron3nano4b_doc:
                                               RETIRED_POOL_PARENTS)
  configs/sdfbase_glm46vflash_doc_full_fits.yaml (2026-09-06; same, for GLM)
  configs/sdfbase_glm46vflash_doc_full_r2t12_fits.yaml (2026-09-07; the GLM
                                               full lane's gated-arm REFIT:
                                               the `_full` config with ONE
                                               change — psr_ce_all's optimizer
                                               is HPO trial r2_t12's, not the
                                               pooled fold's raw-argmin r2_t10
                                               (clip 89.93, diverged) — as a
                                               SIBLING so the `_full` bytes,
                                               two fingerprinted LoRA cells,
                                               never move)
  configs/sdfbase_llama8b_base_fits.yaml      (2026-09-09; the Llama-3.1-8B
                                               BASE checkpoint trained at FULL
                                               scale directly — five cells:
                                               mix-1k / mix-1m / mix-10m are
                                               WORKING cells (identity gate,
                                               memprobe ladder, search
                                               reference), mix-full + ctl2-full
                                               the production cells; effective
                                               batch 64, per key)
  configs/sdfbase_llama8b_it_doc_fits.yaml    (2026-09-09; the INSTRUCT sibling
                                               trained DIRECTLY, same scope,
                                               same effective batch 64)
  configs/sdfbase_qwen35_4b_base_eb64_s{1,2,3}_fits.yaml (2026-09-15; the
                                               qwen-base-seed-replicates lane:
                                               the qwen35_4b_base checkpoint
                                               re-fit from SCRATCH at effective
                                               batch 64 under seeds 1, 2, 3 —
                                               train_kwargs.default.seed governs
                                               LoRA init, data order and the
                                               per-seed held-out carve; s1 also
                                               carries the mix-1k / 1m / 10m
                                               WORKING cells)
  configs/sdfbase_qwen35_4b_demo_base_fits.yaml (2026-10-08; the WORKED
                                               EXAMPLE's lane: the paper's
                                               qwen35_4b_base lane under the
                                               demo key qwen35_4b_demo_base,
                                               with a `fit_selection` block;
                                               configured, not run)

The `fit_selection` block rides
train_kwargs.<arm> on the lanes that declare "fit_selection" in MODELS (the
worked example's qwen35_4b_demo_base lane) and on ANY lane whose "hpo" fold row carries one of
the selection keys — it is ABSENT everywhere else, so the sixteen earlier
configs' bytes never move. ReLU: {fit_search_id, fit_lr_selected_by};
LoRA: {fit_search_id, lora_rank, lora_modules (sorted), per_expert_rank,
fit_lr_selected_by, architecture_selection_ruling_id, init_scheme}. Every
field comes from the folds store's "hpo" row (hpo_pick's output) with the
paper's defaults where the row is silent (rank 4 / o_proj / zero init /
"search"); the trainer serialises the block to fit_selection.json and the
registry hook reads it. `--init-scheme-override SCHEME --models K...` stamps
`init_scheme` into the named lanes' fold rows (both arms; merged, never
replacing a winner) — the stage script passes the run-overrides file's
OVERRIDE_INIT_SCHEME through it, so the committed bytes stay reproducible
from the tree. `--models-table reproduce/models.yaml` cross-checks every
lane's model + revision against the models table, matching a lane to the
row whose fit.checkpoint.model_id IS the lane key (a missing table is a
note, a lane the table does not list a note, a disagreeing pin a refusal).

Fail-loud sentinels (the derive_model_config BATCH_PLACEHOLDER
convention): batch_size_by_arm carries "MEMPROBE" until the Modal memprobe
(scripts/modal/sdfbase_fits.py --mode memprobe) lands — resolve_batch_size
int()s the value, so a wave launched before the measurement crashes
instead of training at a default. lr / warmup_frac / max_grad_norm carry
HPO sentinels that float()-crash until --fold-hpo fills the winners, and
accum_steps carries a sentinel that int()-crashes until --fold-memprobe
fills the measured width (accum = ceil(EFFECTIVE_BATCH / bs)).

Folds are DECLARATIVE REPO DATA: configs/sdfbase_folds.json (committed,
versioned sections "memprobe" and "hpo") is read by default on every
build, so a bare `build_all()` / bare CLI run on a fresh clone reproduces
the committed folded bytes exactly — the regeneration byte-pin never
depends on remembering which folds produced the committed files.

Folding (updates the folds file, regenerates, commit BOTH, launch with
the new expect_sha):

  --fold-memprobe m.json   {model_key: {arm: batch_size}}     -> "memprobe"
  --fold-hpo w.json        {model_key: {arm: {lr, warmup_frac,
                                              max_grad_norm[, selection_guard,
                                              fit_search_id, lora_rank,
                                              lora_modules, per_expert_rank,
                                              init_scheme, fit_lr_selected_by,
                                              architecture_selection_ruling_id
                                              ]}}} -> "hpo"

Each flag deep-merges its JSON into the folds file's section (per model,
per arm) and rebuilds from the merged file. Keys the folds file does not
cover keep their sentinels (HPO winners stay PENDING and float()-crash
until the "hpo" section fills). Unknown model keys / arms / missing winner
fields refuse loudly — a fold must never silently skip a lane.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]

# The HPO selection guard's alarm-code rendering (scripts/hpo_selection_guard.py,
# pure python, no torch): the fold-time refusal below spells alarm codes
# through the guard's ONE ``alarm_code`` so they read exactly as analyze's
# artefact, the confirmation stamp and the next-round driver spell them.
# ``python scripts/tune_arm.py`` has scripts/ on sys.path; a by-path load
# (the tests, a bare ``python scripts/sdfbase/make_sdfbase_configs.py``)
# does not, so fall back to the file.
try:
    import hpo_selection_guard as _guard
except ImportError:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "hpo_selection_guard", REPO / "scripts" / "hpo_selection_guard.py")
    _guard = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_guard)
alarm_code = _guard.alarm_code

#: The committed declarative fold store (sections "memprobe" and "hpo"):
#: read by default on every build so bare regeneration reproduces the
#: committed folded bytes; written ONLY through guarded_write.
FOLDS_FILENAME = "sdfbase_folds.json"
FOLDS_PATH = REPO / "configs" / FOLDS_FILENAME

#: Only file names of this shape may ever be written (guarded_write):
#: the generated sdfbase_*.yaml configs plus the one folds store.
_ALLOWED_NAME = re.compile(
    r"sdfbase_[A-Za-z0-9_]+\.yaml|" + re.escape(FOLDS_FILENAME))

#: batch_size_by_arm sentinel — resolve_batch_size int()s it (fail-loud).
MEMPROBE_SENTINEL = "MEMPROBE"
#: HPO-winner sentinels — float()/int() crash if a wave launches unfilled.
HPO_SENTINEL = "PENDING_SDFBASE_HPO_WINNER"
#: The accum sentinel names the effective batch it stands in for, so a key
#: on a non-default effective batch (the Llama lane's 64) does not carry a
#: sentinel that says EFF8. ACCUM_SENTINEL is the campaign-default (8)
#: spelling, byte-identical to what every frozen config was rendered with.
ACCUM_SENTINEL_FMT = "PENDING_MEMPROBE_EFF{eff}_ACCUM"

#: The campaign's effective batch at >=100k scales: accumulation
#: resolves accum_steps = ceil(EFFECTIVE_BATCH / measured bs). Tiny cells
#: degrade gracefully via the trainer's epoch-boundary flush. A MODELS
#: entry may declare its OWN "effective_batch" (effective_batch_for); this
#: module constant stays the default every frozen key renders under, so
#: their bytes never move.
EFFECTIVE_BATCH = 8
ACCUM_SENTINEL = ACCUM_SENTINEL_FMT.format(eff=EFFECTIVE_BATCH)

#: The two CE arms — the only arms this campaign trains.
CE_ARMS = ("psr_ce_all", "lora_r4_ce")

#: The optimizer winner an "hpo" fold row carries (all three or none).
HPO_WINNER_KEYS = ("lr", "warmup_frac", "max_grad_norm")
#: The FIT-SELECTION keys an "hpo" fold row MAY carry (hpo_pick's output).
#: Any one of them present makes the lane emit a `fit_selection` block — so
#: a row for one of the sixteen earlier lanes must never carry one unless
#: its bytes are meant to move.
HPO_SELECTION_KEYS = ("fit_search_id", "lora_rank", "lora_modules",
                      "per_expert_rank", "init_scheme", "fit_lr_selected_by",
                      "architecture_selection_ruling_id")
#: Every key an "hpo" fold row may carry (selection_guard = the HPO
#: selection guard's record, written by the fold writers).
HPO_FOLD_KEYS = HPO_WINNER_KEYS + ("selection_guard",) + HPO_SELECTION_KEYS
#: Object init schemes (= sisteer.extraction.lora_modules.INIT_SCHEMES,
#: spelled here because this generator imports nothing from the package;
#: a test pins the two equal). "zero" is the paper's (FD 7); the run
#: override selects "random_lecun_1e-3".
INIT_SCHEMES = ("zero", "random_lecun_1e-3")
INIT_SCHEME_DEFAULT = "zero"
#: fit_lr_selected_by values a fold may state (vocab.py's CHECK admits more;
#: a config only ever records a search or a manual pick).
FIT_LR_SELECTED_BY = ("search", "manual")
#: The paper's LoRA architecture (FD 18): rank 4 on the attention output
#: projection of every eligible block.
PAPER_LORA_RANK = 4
PAPER_LORA_MODULES = ["o_proj"]
#: FD 6's per-expert rank rule, as the sweep records it (hpo design `per_expert_rank_rule`): the shared expert keeps r, every routed
#: expert gets max(1, r // active_experts). A selection's `per_expert_rank: null` means DERIVED by this rule (stamp-fix-3).
PER_EXPERT_RANK_RULE = "max(1,r//active_experts)"
#: The LoRA module vocabulary (= lora_modules.KNOWN_MODULES, pinned by test).
LORA_MODULE_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj")
LORA_MLP_MODULES = ("gate_proj", "up_proj", "down_proj")

#: The ten data cells, riding the SI slot of the <si>__<arm> cell scheme.
#: constv2-mix-{10m,full} are the round-2 token-scaling extension
#: (2026-09-02): longer prefixes of the same mixed draw order —
#: 10M official tokens, and the ENTIRE train side (~42.8M).
SI_IDS = [
    "constv2-mix-1k",
    "constv2-mix-10k",
    "constv2-mix-100k",
    "constv2-mix-1m",
    "constv2-mix-10m",
    "constv2-mix-full",
    "constv2-adv-1k",
    "constv2-adv-100k",
    "constv2ctl-100k",
    # 2026-09-03: the r2 register-matched control at the full budget
    # (a nested prefix of the control train side cut at
    # constv2-mix-full's exact official token count). NEW prefix on
    # purpose — see the CAPS row.
    "constv2ctl2-full",
]

#: Workstream 3 (2026-09-02): the two chat lanes run the MIXED
#: curve + its 100k control only — {1k,10k,100k,1M} + ctl-100k — no advice
#: lane, no 10m/full, no transfer battery (out of scope here; they can be
#: ordered later). Each cell keeps the base configs' CAPS row, so a
#: per-cell comparison across the four models is like-for-like. A
#: MODELS entry names its scope under "si_ids"; absent = the full SI_IDS.
WS3_SI_IDS = [
    "constv2-mix-1k",
    "constv2-mix-10k",
    "constv2-mix-100k",
    "constv2-mix-1m",
    "constv2ctl-100k",
]

#: INSTRUCT-DIRECT arms (2026-09-03): the two instruct checkpoints the
#: transfer battery serves, trained on the constitutional corpus
#: DIRECTLY — full scale only, constitutional only — instead of
#: base-fit + zero-shot transfer. One cell per model.
IT_DIRECT_SI_IDS = [
    "constv2-mix-full",
]

#: The instruct-direct cells' caps row: 16 / 2, the "budget for more
#: epochs" rule (as for the r2 control) — directly comparable to
#: constv2ctl2-full and to the re-continued mix-full cells in budget and
#: stopping rule, NOT to the original cap-4 base fits. A per-MODEL
#: override (MODELS[key]["caps"]) because CAPS is keyed by cell and the
#: base configs' mix-full row must keep its bytes (every wave-A' fit
#: fingerprint hashes them).
IT_DIRECT_CAPS: dict[str, dict[str, int]] = {
    "constv2-mix-full": {"epochs": 16, "early_stop_patience": 2},
}

#: FULL-SCALE lanes for the two chat checkpoints (2026-09-06, run on
#: Slurm rather than Modal): the constitutional corpus at full scale AND
#: its r2 register-matched control at the same budget, one cell each, both
#: at cap 16 / patience 2 — the it_doc / ctl2-full "budget for more epochs"
#: row, NOT the base configs' cap-4 mix-full row. The frozen `_doc`
#: configs (FROZEN_CONFIG_SHA16 in tests/test_sdfbase_configs.py) carry their
#: finished 1k-1m cells' fit identity in their BYTES, so these cells live in
#: SIBLING configs under `_full` keys (the it_doc precedent: a new config per
#: lane), and the caps ride the per-MODEL override.
FULL_SI_IDS = [
    "constv2-mix-full",
    "constv2ctl2-full",
]
FULL_CAPS: dict[str, dict[str, int]] = {
    "constv2-mix-full": {"epochs": 16, "early_stop_patience": 2},
    "constv2ctl2-full": {"epochs": 16, "early_stop_patience": 2},
}

#: The Llama-3.1-8B lane (2026-09-09): a
#: `_base` key whose scope is NOT the 9-cell ladder — five cells per key.
#: Three are WORKING cells, never results: mix-1k (the cross-lane identity
#: gate), mix-1m (the memory probe's worst-case rows), mix-10m (the search
#: reference cell — 64 full batches per epoch at effective batch 64, 512
#: optimizer steps at its cap; the 100k cell holds 43 docs, fewer than one
#: batch of 64, so it cannot form a step). The two full-scale cells are
#: the production fits, at the it_doc / ctl2-full cap-16 / patience-2 row.
#: The three working cells' rows equal the shared CAPS rows (pinned by
#: test); every row rides the per-MODEL override so no shared table moves.
LLAMA8B_SI_IDS = [
    "constv2-mix-1k",
    "constv2-mix-1m",
    "constv2-mix-10m",
    "constv2-mix-full",
    "constv2ctl2-full",
]
LLAMA8B_CAPS: dict[str, dict[str, int]] = {
    "constv2-mix-1k": {"epochs": 200, "early_stop_patience": 20},
    "constv2-mix-1m": {"epochs": 24, "early_stop_patience": 3},
    "constv2-mix-10m": {"epochs": 8, "early_stop_patience": 2},
    "constv2-mix-full": {"epochs": 16, "early_stop_patience": 2},
    "constv2ctl2-full": {"epochs": 16, "early_stop_patience": 2},
}
#: The Llama lane trains at effective batch 64 (2026-09-09: 8 was
#: responsible for very spiky training dynamics). Per KEY — never by moving
#: EFFECTIVE_BATCH, whose value is baked into six frozen configs' bytes.
#: Only accum_steps moves (ceil(64 / measured bs)); the micro-batch stays
#: whatever the memprobe returns, so peak memory is unchanged.
LLAMA8B_EFFECTIVE_BATCH = 64

#: The instruct-direct BATCH-64 lanes (2026-09-11): the two `_it_doc`
#: checkpoints re-fit at effective batch 64 (both twins of each
#: constitution/control pair re-fit at 64 so the pair differs in corpus
#: alone) as SIBLING keys `_eb64` — the frozen `_it_doc` configs are the
#: fit identity of four PUBLISHED cells the artifact cites, so their
#: bytes never move and the new cells land in their own out/<key>_eb64/
#: namespace (the `_full` sibling precedent). Four cells per key: mix-1m
#: (the memprobe's worst-case rows) and mix-10m (the LR search
#: reference — 4,102 docs = ~64 optimizer steps per epoch at effective
#: batch 64, cap 8 = ~513 steps; the 1M tier holds 408 docs = 6.4 steps
#: per epoch, too few to validate a learning rate) are WORKING cells at
#: the shared CAPS rows, never results; mix-full + ctl2-full are the
#: production twins at the it_doc / ctl2-full cap-16 / patience-2 row.
#: "pool_key" shares the parent `_it_doc` lane's staged pools + sidecars
#: (same checkpoint). HPO winners come from THIS lane's own batch-64
#: search (never the parent's batch-8 winners); memprobe widths are
#: re-measured.
IT_EB64_SI_IDS = [
    "constv2-mix-1m",
    "constv2-mix-10m",
    "constv2-mix-full",
    "constv2ctl2-full",
]
IT_EB64_CAPS: dict[str, dict[str, int]] = {
    "constv2-mix-1m": {"epochs": 24, "early_stop_patience": 3},
    "constv2-mix-10m": {"epochs": 8, "early_stop_patience": 2},
    "constv2-mix-full": {"epochs": 16, "early_stop_patience": 2},
    "constv2ctl2-full": {"epochs": 16, "early_stop_patience": 2},
}
IT_EB64_EFFECTIVE_BATCH = 64
#: `_eb64` sibling -> the frozen `_it_doc` parent whose staged pool dir it
#: shares (the entry's "pool_key"; pool_key_for validates same checkpoint).
IT_EB64_PARENT = {
    "gemma4e4b_it_doc_eb64": "gemma4e4b_it_doc",
    "qwen35_4b_it_doc_eb64": "qwen35_4b_it_doc",
}

#: Qwen-BASE batch-64 SEED REPLICATES (2026-09-15; the qwen-base-seed-
#: replicates lane). The frozen qwen35_4b_base checkpoint re-fit
#: from SCRATCH at effective batch 64 under three NEW seeds, one SIBLING
#: key per seed (`_eb64_s<s>`): the frozen `_base` config's bytes are the
#: fit identity of four published batch-8 seed-0 cells (the reference
#: column, never a replicate), so nothing there moves. Per key: mix-full +
#: ctl2-full are the production cells at the cap-16 / patience-2 row for
#: BOTH corpora (scratch fits, not the published cap-4->16 continuations);
#: s1 additionally carries three WORKING cells at the shared CAPS rows,
#: never results — mix-1k (the Modal<->Slurm identity gate), mix-1m (the
#: memprobe's worst-case rows, sdfbase_fits.MEMPROBE_SI) and mix-10m (the
#: LR search reference; one search per arm at seed 1, its winners folded
#: under all three keys). Each key owns /vol/pool/<key>/ (own per-seed
#: held-out carve, pools and sidecars — NO pool_key: the split differs per
#: seed, so the parent's pool dir cannot be shared). "seed" is a NEW
#: MODELS field, emitted as train_kwargs.default.seed ONLY when declared
#: (the LEADING_TOKEN_KEY precedent), so every existing config's bytes are
#: untouched; downstream it seeds the LoRA A-factor init
#: (methods._train_lora_ce -> MultiLayerLoRAObject(seed=)), the trainer's
#: per-epoch shuffle (train_object_ce) and the HPO trial (tune_arm trial
#: --seed, threaded by sdfbase_fits._hpo_impl).
SEEDREP_PARENT = "qwen35_4b_base"
SEEDREP_SEEDS = (1, 2, 3)
SEEDREP_EFFECTIVE_BATCH = 64
SEEDREP_SI_IDS_S1 = [
    "constv2-mix-1k",     # identity-gate WORKING cell
    "constv2-mix-1m",     # the memprobe cell (MEMPROBE_SI)
    "constv2-mix-10m",    # the LR search reference
    "constv2-mix-full",
    "constv2ctl2-full",
]
SEEDREP_SI_IDS = [        # s2, s3: the production twins only
    "constv2-mix-full",
    "constv2ctl2-full",
]
SEEDREP_CAPS: dict[str, dict[str, int]] = {
    "constv2-mix-1k": {"epochs": 200, "early_stop_patience": 20},
    "constv2-mix-1m": {"epochs": 24, "early_stop_patience": 3},
    "constv2-mix-10m": {"epochs": 8, "early_stop_patience": 2},
    # spec: scratch, cap 16 / patience 2, BOTH corpora
    "constv2-mix-full": {"epochs": 16, "early_stop_patience": 2},
    "constv2ctl2-full": {"epochs": 16, "early_stop_patience": 2},
}


def seedrep_key(seed: int) -> str:
    """The MODELS key of seed replicate ``seed`` of the qwen35_4b_base lane."""
    return f"qwen35_4b_base_eb64_s{int(seed)}"


SEEDREP_KEYS = tuple(seedrep_key(s) for s in SEEDREP_SEEDS)

#: Leading-special-token policies a config may declare (top-level key
#: ``leading_special_tokens``; = sisteer.extraction.leading_tokens.POLICIES,
#: spelled here too because this generator imports nothing from the
#: package). The doc-CE path encodes flat text with add_special_tokens=True
#: and lets the tokenizer's post-processor decide what leads a sequence —
#: a per-CHECKPOINT fact: google/gemma-4-E4B prepends <bos> (its
#: TemplateProcessing is "<bos> $A"), google/gemma-4-E4B-it prepends
#: NOTHING (its template supplies <bos> as literal text), Qwen3.5 base and
#: instruct both prepend nothing (ByteLevel, no BOS token). An
#: instruct-direct lane therefore DECLARES its base sibling's convention:
#: "bos" for gemma (the loader installs "<bos> $A" on the -it tokenizer and
#: the fit refuses any example without exactly one leading BOS), "none" for
#: qwen (the fit refuses any leading BOS). Verified on the real tokenizers
#: at the pinned revisions, 2026-09-03. The four frozen configs declare
#: nothing: their fits keep observing their tokenizers' own convention,
#: bytes untouched.
LEADING_TOKEN_KEY = "leading_special_tokens"
LEADING_TOKEN_POLICIES = ("bos", "none")

#: Per-cell epoch caps + early-stop patience: generous
#: caps, early stopping does the real work; the big-cell caps bound
#: worst-case burn if stopping misbehaves.
#:
#: Round-2 changes (2026-09-02):
#:   - constv2-mix-1m raised 12 -> 24 epochs (patience 3): three of the
#:     four 1M fits ran the 12-epoch cap out (summary check 4) — the raise
#:     changes the fit fingerprint, so a respawn RETRAINS the 1m cells
#:     (intended: the cap-limited cells are refit; qwen mix-1m LoRA
#:     converged at 9/12 and retrains too, for uniformity).
#:   - constv2-mix-10m: {8, 2}. Step math (realized draw, verified
#:     2026-09-02): 10M official tokens = 4,102 docs = 513 optimizer
#:     steps/epoch at effective batch 8 -> the 8-epoch cap ≈ 4.1k
#:     optimizer steps (~8x the 1m cell's 51 steps/epoch x round 1's
#:     12-epoch budget).
#:   - constv2-mix-full: {4, 1}. The ENTIRE train side after the committed
#:     held-out split: 42,785,435 official tokens (llama3-limiting) =
#:     17,352 docs = 2,169 steps/epoch (4.28x the 10m cell) -> the 4-epoch
#:     cap ≈ 8.7k optimizer steps.
CAPS: dict[str, dict[str, int]] = {
    "constv2-mix-1k": {"epochs": 200, "early_stop_patience": 20},
    "constv2-mix-10k": {"epochs": 120, "early_stop_patience": 12},
    "constv2-mix-100k": {"epochs": 60, "early_stop_patience": 6},
    "constv2-mix-1m": {"epochs": 24, "early_stop_patience": 3},
    "constv2-mix-10m": {"epochs": 8, "early_stop_patience": 2},
    "constv2-mix-full": {"epochs": 4, "early_stop_patience": 1},
    "constv2-adv-1k": {"epochs": 200, "early_stop_patience": 20},
    "constv2-adv-100k": {"epochs": 60, "early_stop_patience": 6},
    "constv2ctl-100k": {"epochs": 60, "early_stop_patience": 6},
    # constv2ctl2-full (2026-09-03): {16, 2}, NOT mix-full's original
    # {4, 1} — the control is budgeted for MORE epochs than the
    # constitutional full cells got, because all four mix-full fits ran
    # that cap out and are being continued to cap 16 separately. Same
    # pool budget (42,785,435 official tokens = 2,169 steps/epoch at
    # effective batch 8 -> the 16-epoch cap ~34.7k optimizer steps), same
    # HPO winners per model x arm. NOTE: adding this row changes both
    # base configs' bytes, so the fit fingerprint's config_sha (the
    # config FILE sha) moves for EVERY cell — exactly what the round-2
    # 10m/full rows did; fits-spawn's --sis scoping (resolve_spawn_sis)
    # is the standing protection, and a continuation of a cell fitted
    # under the previous bytes must deploy from a tree that still carries
    # them.
    "constv2ctl2-full": {"epochs": 16, "early_stop_patience": 2},
}

#: Model table: base (non-instruct) siblings of the instruct
#: campaign models, pinned. `layer` is the campaign convention
#: (round(0.469 x n_layers)) — unused by the all-layer CE arms, kept for
#: schema parity. qwen3_5 is a hybrid-attention architecture: gradient
#: checkpointing is an architecture requirement on BOTH arms.
MODELS: dict[str, dict] = {
    "gemma4e4b_base": {
        "model": "google/gemma-4-E4B",
        "revision": "411aa17b749aa952df1359d2dcea73917a544d9a",
        "layer": 20,
        "gradient_checkpointing": False,
    },
    "qwen35_4b_base": {
        "model": "Qwen/Qwen3.5-4B-Base",
        "revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "layer": 15,
        "gradient_checkpointing": True,
    },
    # GLM-4.6V-Flash (2026-09-02) — the `_doc` lane, deliberately NOT a
    # `_base` one: no base (pretrained-only) variant of GLM-4.6V exists,
    # so these doc-CE fits target the CHAT model itself. The rendering is
    # unchanged (data_mode: documents — flat text, no chat template ever
    # applied); `chat_model` only changes the generated header's wording
    # so the config does not call a chat checkpoint a base model. Dense
    # 10.3B VLM (~9B text decoder) used text-only, full attention on all
    # 40 layers: NO gradient checkpointing (the gemma precedent — qwen's
    # is a hybrid-stack requirement, which does not apply here).
    # layer = round(0.469 x 40) = 19, schema parity only. Pin verified
    # against the HF API 2026-09-02; MIT, ungated.
    "glm46vflash_doc": {
        "model": "zai-org/GLM-4.6V-Flash",
        "revision": "411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
        "layer": 19,
        "gradient_checkpointing": False,
        "chat_model": True,
        "si_ids": WS3_SI_IDS,
    },
    # INSTRUCT-DIRECT arms (2026-09-03). The two instruct siblings
    # — the SAME repo + revision the transfer battery serves
    # (configs/sdfbase_transfer_{gemma4e4b,qwen35_4b}_r2.yaml
    # model_revision; = the probe_only gemma4e4b_it / qwen35_4b_it entries
    # in scripts/modal/sdfbase_fits.py) — as fit-capable `_it_doc` lanes:
    # the corpus is distilled into the instruct model DIRECTLY. Rendering
    # is the doc lane's, unchanged: data_mode documents, flat text via
    # docs_to_ce_examples, the chat template bypassed in training exactly
    # as the chat lanes (probe render = chat, a sdfbase_fits.py
    # property). Scope: mix-full only (si_ids) at cap 16 / patience 2
    # (caps). gradient_checkpointing follows the ARCHITECTURE (qwen3_5
    # hybrid ON, gemma dense OFF); the HPO winners and memprobe widths are
    # the base lane's, COPIED into the folds store under these keys
    # (same architecture, same objective; a fresh HPO at
    # 100k is the recorded fallback if a fit fails its CE floor gate).
    # header_kind: these DO have a base sibling, so the chat-lane header
    # wording ("no base variant of it exists") would be false here.
    "gemma4e4b_it_doc": {
        "model": "google/gemma-4-E4B-it",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "layer": 20,
        "gradient_checkpointing": False,
        "chat_model": True,
        # the base sibling's convention (gemma-4-E4B: <bos> at position 0)
        "leading_special_tokens": "bos",
        "header_kind": ("INSTRUCT model, trained DIRECTLY (the "
                        "instruct-direct arm; its base sibling is the "
                        "gemma4e4b_base lane)"),
        "si_ids": IT_DIRECT_SI_IDS,
        "caps": IT_DIRECT_CAPS,
    },
    "qwen35_4b_it_doc": {
        "model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "layer": 15,
        "gradient_checkpointing": True,
        "chat_model": True,
        # the base sibling's convention (Qwen3.5-4B-Base: no BOS token)
        "leading_special_tokens": "none",
        "header_kind": ("INSTRUCT model, trained DIRECTLY (the "
                        "instruct-direct arm; its base sibling is the "
                        "qwen35_4b_base lane)"),
        "si_ids": IT_DIRECT_SI_IDS,
        "caps": IT_DIRECT_CAPS,
    },
    # INSTRUCT-DIRECT BATCH-64 siblings (2026-09-11): the SAME
    # model + revision + layer + gradient-checkpointing + leading-token
    # policy as the frozen `_it_doc` entries above — the differences are
    # effective_batch 64 (IT_EB64_EFFECTIVE_BATCH), the cell scope
    # (IT_EB64_SI_IDS), the caps row (IT_EB64_CAPS), the header, and
    # pool_key -> the parent `_it_doc` lane's staged pool dir (same
    # checkpoint, so its sidecars validate). The HPO fold under these keys
    # is this lane's own batch-64 search, never a copy of the parent's.
    "gemma4e4b_it_doc_eb64": {
        "model": "google/gemma-4-E4B-it",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "layer": 20,
        "gradient_checkpointing": False,
        "chat_model": True,
        "leading_special_tokens": "bos",
        "header_kind": ("INSTRUCT model, trained DIRECTLY at EFFECTIVE "
                        "BATCH 64 (the instruct-direct batch-64 "
                        "re-fit; sibling of the frozen gemma4e4b_it_doc "
                        "lane, its base sibling is the gemma4e4b_base lane)"),
        "si_ids": IT_EB64_SI_IDS,
        "caps": IT_EB64_CAPS,
        "pool_key": IT_EB64_PARENT["gemma4e4b_it_doc_eb64"],
        "effective_batch": IT_EB64_EFFECTIVE_BATCH,
    },
    "qwen35_4b_it_doc_eb64": {
        "model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "layer": 15,
        "gradient_checkpointing": True,
        "chat_model": True,
        "leading_special_tokens": "none",
        "header_kind": ("INSTRUCT model, trained DIRECTLY at EFFECTIVE "
                        "BATCH 64 (the instruct-direct batch-64 "
                        "re-fit; sibling of the frozen qwen35_4b_it_doc "
                        "lane, its base sibling is the qwen35_4b_base lane)"),
        "si_ids": IT_EB64_SI_IDS,
        "caps": IT_EB64_CAPS,
        "pool_key": IT_EB64_PARENT["qwen35_4b_it_doc_eb64"],
        "effective_batch": IT_EB64_EFFECTIVE_BATCH,
    },
    # FULL-SCALE lanes of the two chat checkpoints (2026-09-06; Slurm).
    # SAME model + revision + layer + gradient-checkpointing choice as the
    # `_doc` entries they descend from (the GLM one above; the Nemotron
    # data-scaling ladder was retired 2026-10-08 — RETIRED_POOL_PARENTS keeps
    # its pool dir resolvable, since the paper's 17 Nemotron objects were fit
    # under it) — the only differences are the cell scope (FULL_SI_IDS), the
    # caps row (FULL_CAPS) and the header. "pool_key" points the held-out
    # pool path at the PARENT lane's staged pool dir (/vol/pool/<parent>/):
    # pools and tokcache sidecars are per CHECKPOINT
    # (a sidecar pins model_id + template_hash), so the same checkpoint
    # shares one pool dir and one held-out pool file — the held-out
    # CONTENT sha, which is what the fit identity binds, is unchanged by
    # where the file sits. HPO winners + memprobe widths are the parent's,
    # COPIED into the folds store under these keys (the it_doc precedent).
    "nemotron3nano4b_doc_full": {
        "model": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "revision": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        "layer": 20,
        "gradient_checkpointing": True,
        "chat_model": True,
        "header_kind": ("CHAT model; no base variant of it exists — the "
                        "FULL-SCALE lane of the frozen nemotron3nano4b_doc "
                        "config, 2026-09-06"),
        "si_ids": FULL_SI_IDS,
        "caps": FULL_CAPS,
        "pool_key": "nemotron3nano4b_doc",
    },
    "glm46vflash_doc_full": {
        "model": "zai-org/GLM-4.6V-Flash",
        "revision": "411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
        "layer": 19,
        "gradient_checkpointing": False,
        "chat_model": True,
        "header_kind": ("CHAT model; no base variant of it exists — the "
                        "FULL-SCALE lane of the frozen glm46vflash_doc "
                        "config, 2026-09-06"),
        "si_ids": FULL_SI_IDS,
        "caps": FULL_CAPS,
        "pool_key": "glm46vflash_doc",
    },
    # GLM gated-arm REFIT (2026-09-07). The glm46vflash_doc_full lane's
    # psr_ce_all cells diverged at full scale: the pooled r1+2 HPO fold's
    # quadratic was not positive-definite, so the fold fell through to the
    # raw argmin, r2_t10 — lr 5.32e-3 with max_grad_norm 89.93, effectively
    # NO gradient clipping (GLM's typical grad norm is 0.25-0.5) — which won
    # a FLAT objective (held-out ratio 0.9146-0.9210 across three decades of
    # lr) by 0.001-0.007, inside noise. At full scale one CE spike (26.5,
    # grad 1e4-1e7) killed 55-95% of the gates. The correction: trial
    # r2_t12 (lr 7.95e-4, max_grad_norm 0.851, warmup_frac 0.05 — a spike
    # guard that fires on a few percent of steps and leaves Adam's character
    # intact; its own 100k record: ratio 0.9182, best epoch 6/12). Its
    # values are folded into sdfbase_folds.json VERBATIM from that trial's
    # summary.json (never retyped). Everything else is the `_full` entry's,
    # byte for byte — same checkpoint, scope, caps, pool dir, and the SAME
    # LoRA block (the LoRA arm is healthy and is NOT meant to be fit under
    # this config; the block exists because the generator renders both CE
    # arms). A SIBLING key, the `_full` precedent: the `_full` config's
    # bytes are the fit identity of two fingerprinted LoRA cells (one
    # COMPLETE, one in flight) and must not move.
    "glm46vflash_doc_full_r2t12": {
        "model": "zai-org/GLM-4.6V-Flash",
        "revision": "411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
        "layer": 19,
        "gradient_checkpointing": False,
        "chat_model": True,
        "header_kind": ("CHAT model; no base variant of it exists — the "
                        "FULL-SCALE lane of the frozen glm46vflash_doc "
                        "config, REFIT 2026-09-07: psr_ce_all's optimizer is "
                        "HPO trial r2_t12's (a clipped, interior point) in "
                        "place of the pooled fold's raw-argmin r2_t10 (clip "
                        "89.93, diverged at full scale); every other byte is "
                        "the glm46vflash_doc_full config's, and the LoRA arm "
                        "is not meant to be fit here"),
        "si_ids": FULL_SI_IDS,
        "caps": FULL_CAPS,
        "pool_key": "glm46vflash_doc",
    },
    # Llama-3.1-8B (2026-09-09). BOTH keys are the
    # ungated NousResearch community mirrors (the meta-llama originals are
    # gated, and the meta-llama instruct template injects a dated system
    # block); pins verified against the HF API 2026-09-09 (verified-facts
    # §5). Dense LlamaForCausalLM, 32 layers, hidden 4096, vocab 128,256:
    # NO gradient checkpointing (the dense precedent; the memprobe may
    # overrule). layer = round(0.469 x 32) = 15, schema parity only.
    # leading_special_tokens: "bos" on BOTH — MEASURED on the real
    # tokenizers at the pinned revisions (verified-facts §15): both encode
    # "" to [128000] and lead every bare document with BOS, so the policy
    # is a no-op installation that asserts the observed behaviour on every
    # encoded example rather than correcting one side (the gemma pair
    # needed a correction; this pair does not — the base-vs-instruct
    # contrast carries no tokenization confound). No pool_key: each key has
    # its own /vol/pool/<key>/ dir and its own per-checkpoint sidecars (the
    # two tokenizers are near-identical but a sidecar pins model_id).
    # effective_batch 64 (LLAMA8B_EFFECTIVE_BATCH), per key. The `_base`
    # naming follows gemma4e4b_base / qwen35_4b_base (a base checkpoint
    # whose instruct sibling is the served target); a plain "llama8b" would
    # collide with the SI-distillation lane's pool dir on the volume.
    "llama8b_base": {
        "model": "NousResearch/Meta-Llama-3.1-8B",
        "revision": "1f47e50cdbe801ad8a5174156ec3a0655108fb9f",
        "layer": 15,
        "gradient_checkpointing": False,
        "leading_special_tokens": "bos",
        # A BASE model with NO frozen parent (there is no frozen Llama
        # `_doc` lane, so `_doc_full`'s FULL_PARENT_OF wording would be
        # false) and no chat_model flag (it is not a chat checkpoint).
        "header_kind": ("BASE model, trained at FULL scale directly "
                        "(2026-09-09 Llama lane: no frozen 1k-1m parent; "
                        "mix-1k / mix-1m / mix-10m are working cells — "
                        "identity gate, memprobe ladder, search reference "
                        "— never results)"),
        "si_ids": LLAMA8B_SI_IDS,
        "caps": LLAMA8B_CAPS,
        "effective_batch": LLAMA8B_EFFECTIVE_BATCH,
    },
    "llama8b_it_doc": {
        "model": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "revision": "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
        "layer": 15,
        "gradient_checkpointing": False,
        "chat_model": True,
        "leading_special_tokens": "bos",
        "header_kind": ("INSTRUCT model, trained DIRECTLY (instruct-direct "
                        "arm; its base sibling is the llama8b_base lane; "
                        "2026-09-09)"),
        "si_ids": LLAMA8B_SI_IDS,
        "caps": LLAMA8B_CAPS,
        "effective_batch": LLAMA8B_EFFECTIVE_BATCH,
    },
}

#: paper-companion-repo (2026-10-08): the WORKED EXAMPLE's fit lane — the
#: paper's qwen35_4b_base lane (model / revision / layer / gradient
#: checkpointing / the 10-cell scope at the shared CAPS rows) under the DEMO
#: key, so a reader's fits land in their own /vol/pool/qwen35_4b_demo_base/
#: dir and out/ bank, never beside the paper's committed objects
#: (reproduce/models.yaml `qwen35_4b_demo`, whose fit model_id is this key).
#: Declares "fit_selection" so the per-arm block is emitted from the first
#: render — the front door's 03 -> 04 contract — and NO
#: "fit_width": one GPU per fit, as the paper's Qwen3.5-4B objects were fit.
#: The fits run at EFFECTIVE BATCH 64 under SEED 3 — the seedrep eb-64 recipe
#: (SEEDREP_EFFECTIVE_BATCH and the "seed" field the `_eb64_s<s>` keys use;
#: seed 3 since the one-seed DEMO.json, 2026-10-12), the recipe the paper's
#: COMMITTED Qwen3.5-4B objects were fit under (objects/DEMO.json records
#: `effective_batch: 64` and training seed 3 for every fitted object), so a
#: reader's fresh fits match the committed ones in batch and seed, and the
#: config's `seed:` line names the fit's seed outright (the trainer records
#: it top-level in train_budget_meta.json). The accum sentinel therefore spells
#: EFF64 and the memprobe fold resolves accum_steps = ceil(64 / bs); the
#: campaign constant EFFECTIVE_BATCH (8) and every frozen key are untouched.
#: CONFIGURED end to end, NOT run in this work (FD 4). The front door's lane
#: (PIPELINE_TEST_KEYS: the lanes that declare "fit_selection").
PIPELINE_TEST_KEYS = ("qwen35_4b_demo_base",)
MODELS["qwen35_4b_demo_base"] = {
    "model": "Qwen/Qwen3.5-4B-Base",
    "revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
    "layer": 15,
    "gradient_checkpointing": True,
    "header_kind": ("BASE model — the WORKED EXAMPLE's lane: the paper's "
                    "qwen35_4b_base lane under the demo key, CONFIGURED end "
                    "to end, NOT run in this work (run_in_this_work: false); "
                    "served on its instruct sibling Qwen/Qwen3.5-4B"),
    "fit_selection": True,
    # the seedrep eb-64 recipe's two fields (the `_eb64_s<s>` keys below
    # declare exactly these): the committed objects' batch and seed — no new
    # constant, no fit_width
    "effective_batch": SEEDREP_EFFECTIVE_BATCH,
    "seed": 3,
}

# Qwen-BASE batch-64 seed replicates (2026-09-15; SEEDREP_* above): the
# qwen35_4b_base pin verbatim (model / revision / layer / gradient
# checkpointing — a test asserts the tie), a per-seed scope + caps row via
# the per-MODEL override, effective batch 64 per key, and the declared
# seed. Registered by loop so the three keys cannot drift from each other;
# the key names are spelled by seedrep_key.
for _s in SEEDREP_SEEDS:
    _ids = SEEDREP_SI_IDS_S1 if _s == 1 else SEEDREP_SI_IDS
    MODELS[seedrep_key(_s)] = {
        "model": "Qwen/Qwen3.5-4B-Base",
        "revision": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "layer": 15,
        "gradient_checkpointing": True,
        "header_kind": (f"BASE model, batch-64 SEED REPLICATE s{_s} of the "
                        "qwen35_4b_base lane (scratch fits; own held-out "
                        "carve, pools and sidecars under /vol/pool/<key>/)"),
        "si_ids": _ids,
        "caps": {k: SEEDREP_CAPS[k] for k in _ids},
        "effective_batch": SEEDREP_EFFECTIVE_BATCH,
        # NEW field: emitted as train_kwargs.default.seed ONLY when declared
        "seed": _s,
    }
del _s, _ids


def fit_width_for(key: str) -> int | None:
    """The lane's declared data-parallel fit width, or None (the
    historical single-process fit; the key is then not emitted)."""
    w = MODELS[key].get("fit_width")
    if w is None:
        return None
    w = int(w)
    if w < 1:
        raise SystemExit(f"[make-sdfbase-configs] {key}: fit_width must be "
                         f">= 1, got {w!r}")
    return w


def seed_for(key: str) -> int | None:
    """The seed a MODELS entry declares (its "seed" field), validated as a
    non-negative int — or None when undeclared, in which case NO seed key
    is emitted and the trainer's own default (0) governs, exactly as every
    frozen config has always run. A bool or a non-int refuses: a seed is a
    deliberate integer, never a truthiness accident."""
    spec = MODELS[key]
    if "seed" not in spec:
        return None
    raw = spec["seed"]
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise SystemExit(f"[make-sdfbase-configs] {key} declares "
                         f"seed={raw!r}, not an int")
    if raw < 0:
        raise SystemExit(f"[make-sdfbase-configs] {key} declares "
                         f"seed={raw} < 0")
    return int(raw)


def effective_batch_for(key: str) -> int:
    """The effective batch a key's accumulation resolves to: the MODELS
    entry's own "effective_batch" when declared (64 on the Llama keys),
    else the campaign constant EFFECTIVE_BATCH (8) — so every frozen key
    renders exactly as before. Validated >= 1."""
    raw = MODELS[key].get("effective_batch", EFFECTIVE_BATCH)
    try:
        eff = int(raw)
    except (TypeError, ValueError):
        raise SystemExit(f"[make-sdfbase-configs] {key} declares "
                         f"effective_batch={raw!r}, not an int")
    if eff < 1:
        raise SystemExit(f"[make-sdfbase-configs] {key} declares "
                         f"effective_batch={eff} < 1")
    return eff


def si_ids_for(key: str) -> list[str]:
    """The data cells a model's config carries: a MODELS entry's own
    "si_ids" scope (the chat lanes), else the full SI_IDS. Every id
    must be a CAPS row — a scope naming an uncapped cell refuses."""
    ids = list(MODELS[key].get("si_ids") or SI_IDS)
    missing = [si for si in ids if si not in CAPS]
    if missing:
        raise SystemExit(f"[make-sdfbase-configs] {key} scopes cell(s) "
                         f"{missing} with no CAPS row")
    return ids


def _leading_policy(key: str) -> str | None:
    """A MODELS entry's declared leading-special-token policy (validated
    against LEADING_TOKEN_POLICIES), None when undeclared."""
    pol = MODELS[key].get("leading_special_tokens")
    if pol is None:
        return None
    if pol not in LEADING_TOKEN_POLICIES:
        raise SystemExit(f"[make-sdfbase-configs] {key} declares "
                         f"leading_special_tokens={pol!r} (known: "
                         f"{list(LEADING_TOKEN_POLICIES)})")
    return str(pol)


def caps_for(key: str, si: str) -> dict[str, int]:
    """The {epochs, early_stop_patience} row cell ``si`` of model ``key``
    trains under: the MODELS entry's own "caps" override for that cell
    (the instruct-direct lanes' cap-16 mix-full) when it has one,
    else the campaign CAPS row. Always a fresh dict."""
    override = (MODELS[key].get("caps") or {}).get(si)
    return dict(override if override is not None else CAPS[si])


def _validate_caps_override(key: str) -> None:
    """A "caps" override may name only cells in the entry's own scope
    (a row for a cell the config does not carry would sit unused and
    silently mean nothing), and every row must be a complete, sane
    {epochs, early_stop_patience} pair."""
    override = MODELS[key].get("caps") or {}
    stray = sorted(set(override) - set(si_ids_for(key)))
    if stray:
        raise SystemExit(f"[make-sdfbase-configs] {key} overrides caps for "
                         f"cell(s) {stray} outside its si_ids scope")
    for si, row in override.items():
        if set(row) != {"epochs", "early_stop_patience"}:
            raise SystemExit(f"[make-sdfbase-configs] {key}/{si} caps "
                             f"override must carry exactly epochs + "
                             f"early_stop_patience (got {sorted(row)})")
        ep, pat = int(row["epochs"]), int(row["early_stop_patience"])
        if not (1 <= pat < ep):
            raise SystemExit(f"[make-sdfbase-configs] {key}/{si} caps "
                             f"override needs 1 <= patience < epochs "
                             f"(got epochs={ep}, patience={pat})")


def config_filename(key: str) -> str:
    return f"sdfbase_{key}_fits.yaml"


#: Where the committed configs live: a key whose config is committed re-emits the W&B project that file records.
COMMITTED_CONFIGS_DIR = REPO / "configs"


def _wandb_project_for(key: str) -> str:
    """The ``wandb_project`` of a key's config. A key with a committed config re-emits the value that file records, so a
    regeneration never moves a frozen config's bytes (each file's bytes are its fits' --expect-config-sha). A key with no
    committed config (a new model) gets the site's fits project, resolved when the config is generated:
    ``sisteer.wandb_entity.wandb_project("fits")`` (SISTEER_WANDB_PROJECT_FITS, else WANDB_PROJECT, else
    SISTEER_WANDB_PROJECT, else ``sisteer-fits``)."""
    path = COMMITTED_CONFIGS_DIR / config_filename(key)
    if path.is_file():
        recorded = (yaml.safe_load(path.read_text()) or {}).get("wandb_project")
        if recorded:
            return str(recorded)
    from sisteer.wandb_entity import wandb_project

    return wandb_project("fits")


#: Pool dirs of lanes RETIRED from MODELS whose staged pools a live lane still
#: shares: dir -> (model, revision), so pool_key_for can keep validating the
#: same-checkpoint rule without the parent's row. nemotron3nano4b_doc was the
#: Nemotron data-scaling ladder (constv2-mix-1k / 10k / 100k / 1m +
#: ctl-100k; configs/sdfbase_nemotron3nano4b_doc_fits.yaml, 2026-09-02),
#: retired 2026-10-08: every one of the paper's 17 Nemotron objects is a
#: full-scale cell of the nemotron3nano4b_doc_full lane (objects/MANIFEST.json),
#: which was published under the ladder's pool dir and out/ root
#: (fit_cell_identity.FROZEN_OUT_KEYS) — those bytes are the objects' fit
#: identity and never move.
RETIRED_POOL_PARENTS: dict[str, tuple[str, str]] = {
    "nemotron3nano4b_doc": ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                            "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"),
}


def pool_key_for(key: str) -> str:
    """The ``/vol/pool/<key>/`` directory a lane's pools and tokcache
    sidecars live under: the entry's own key, or its "pool_key" — a
    same-checkpoint sibling whose staged pools it shares (the `_full` lanes
    reuse the `_doc` lanes' dirs; a retired parent is spelled in
    RETIRED_POOL_PARENTS). Sharing is sound ONLY across an identical model +
    revision (a sidecar pins model_id + template_hash), and a pool_key never
    chains — both refused here."""
    spec = MODELS[key]
    pk = spec.get("pool_key")
    if pk is None:
        return key
    if pk in RETIRED_POOL_PARENTS:
        if RETIRED_POOL_PARENTS[pk] != (spec["model"], spec["revision"]):
            raise SystemExit(
                f"[make-sdfbase-configs] {key} shares the retired pool dir {pk} "
                f"but pins {spec['model']}@{spec['revision'][:8]} vs that dir's "
                f"{RETIRED_POOL_PARENTS[pk][0]}@{RETIRED_POOL_PARENTS[pk][1][:8]} — "
                "pools and sidecars are per checkpoint")
        return str(pk)
    if pk not in MODELS:
        raise SystemExit(f"[make-sdfbase-configs] {key} names pool_key "
                         f"{pk!r}, not a MODELS key (nor a RETIRED_POOL_PARENTS dir)")
    parent = MODELS[pk]
    if parent.get("pool_key") is not None:
        raise SystemExit(f"[make-sdfbase-configs] {key} -> pool_key {pk} "
                         "-> pool_key again: a pool_key never chains")
    if (parent["model"], parent["revision"]) != (spec["model"],
                                                  spec["revision"]):
        raise SystemExit(
            f"[make-sdfbase-configs] {key} shares pool dir {pk} but pins "
            f"{spec['model']}@{spec['revision'][:8]} vs the parent's "
            f"{parent['model']}@{parent['revision'][:8]} — pools and "
            "sidecars are per checkpoint")
    return str(pk)


def heldout_pool_path(key: str) -> str:
    """The shared held-out eval pool, staged per model on the twosi-fits
    volume: every fit of every cell (and every HPO trial) evaluates
    per-epoch CE on exactly this pool. Under a "pool_key" the path is
    the sibling's dir (pool_key_for)."""
    return f"/vol/pool/{pool_key_for(key)}/pairs_constv2-heldout.jsonl"


def _accum_for(bs, effective_batch: int = EFFECTIVE_BATCH) -> object:
    """accum_steps for a measured micro-batch: ceil(effective_batch / bs).
    ``effective_batch`` is the KEY's (effective_batch_for) — the default
    keeps every frozen key's arithmetic and sentinel spelling byte-identical."""
    eff = int(effective_batch)
    if bs is None:
        return ACCUM_SENTINEL_FMT.format(eff=eff)
    bs = int(bs)
    if bs < 1:
        raise SystemExit(f"[make-sdfbase-configs] measured batch size {bs} "
                         "< 1 — a broken memprobe fold, refusing")
    return math.ceil(eff / bs)  # >= 1: bs >= 1 was just checked


def _winner_of(key: str, arm: str, hpo_win: dict | None) -> dict | None:
    """The optimizer winner of a fold row: the three HPO_WINNER_KEYS when
    the row carries them ALL, None when it carries none (a selection-only
    row — the init-scheme override before the search — keeps the optimizer
    sentinels); a partial row refuses."""
    if not hpo_win:
        return None
    present = [k for k in HPO_WINNER_KEYS if k in hpo_win]
    if not present:
        return None
    missing = [k for k in HPO_WINNER_KEYS if k not in hpo_win]
    if missing:
        raise SystemExit(
            f"[make-sdfbase-configs] HPO winner for {key}/{arm} is "
            f"missing {missing} (got {sorted(hpo_win)})")
    return hpo_win


def emits_fit_selection(key: str, hpo_win: dict | None) -> bool:
    """Whether train_kwargs.<arm> carries a `fit_selection` block: the
    lanes that DECLARE it (MODELS "fit_selection": the worked example's demo
    lanes) and any lane whose fold row carries a selection key. Every
    other lane renders exactly as before the block existed."""
    if MODELS[key].get("fit_selection"):
        return True
    return bool(hpo_win) and any(k in hpo_win for k in HPO_SELECTION_KEYS)


def per_expert_rank_for(key: str, rank: int, modules: list[str]) -> dict | None:
    """The MoE per-expert rank map the fit_selection block records (FD 6,
    "LoRA Without Regret"): {shared_expert: r, routed_expert: max(1,
    r // active)} when the lane declares "moe_active_experts" AND the
    module set reaches the FFN (gate/up/down_proj — the routed experts'
    3-D parameters); None for an attention-only set or a dense model.
    Spelled here (= lora_modules.per_expert_rank_map, pinned by test)."""
    active = MODELS[key].get("moe_active_experts")
    if active is None or not any(m in LORA_MLP_MODULES for m in modules):
        return None
    active = int(active)
    if active < 1:
        raise SystemExit(f"[make-sdfbase-configs] {key} declares "
                         f"moe_active_experts={active} < 1")
    return {"shared_expert": int(rank), "routed_expert": max(1, int(rank) // active)}


def lora_rank_for(key: str, arm: str, hpo_win: dict | None) -> int:
    """The LoRA rank a lane's lora arm trains at: the fold row's
    ``lora_rank`` (hpo_pick's or a manual pick) when present, else the
    paper's 4 (every existing config's byte-pinned value)."""
    raw = (hpo_win or {}).get("lora_rank", PAPER_LORA_RANK)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise SystemExit(f"[make-sdfbase-configs] {key}/{arm} fold carries "
                         f"lora_rank={raw!r}, not a positive int")
    return int(raw)


def fit_selection_block(key: str, arm: str, hpo_win: dict | None,
                        lora_rank: int | None = None) -> dict:
    """The per-arm `fit_selection` mapping: what the trainer
    serialises to fit_selection.json and the registry hook records on the
    object row. ReLU: {fit_search_id, fit_lr_selected_by}. LoRA: + the
    architecture {lora_rank, lora_modules (sorted), per_expert_rank,
    architecture_selection_ruling_id, init_scheme}. Every value is the fold
    row's when it states one, else the paper's default (FD 18) — the
    search id stays the HPO sentinel until the search closes, so a fit
    launched before it has a search id that is visibly not one."""
    row = dict(hpo_win or {})
    sid = row.get("fit_search_id", HPO_SENTINEL)
    if sid is not None and not isinstance(sid, str):
        raise SystemExit(f"[make-sdfbase-configs] {key}/{arm} fold carries "
                         f"fit_search_id={sid!r}, not a string")
    by = row.get("fit_lr_selected_by", "search")
    if by not in FIT_LR_SELECTED_BY:
        raise SystemExit(f"[make-sdfbase-configs] {key}/{arm} fold carries "
                         f"fit_lr_selected_by={by!r} (known: "
                         f"{list(FIT_LR_SELECTED_BY)})")
    scheme = row.get("init_scheme", INIT_SCHEME_DEFAULT)   # both arms carry it (FD 7)
    if scheme not in INIT_SCHEMES:
        raise SystemExit(f"[make-sdfbase-configs] {key}/{arm} fold carries "
                         f"init_scheme={scheme!r} (known: {list(INIT_SCHEMES)})")
    if arm == "psr_ce_all":
        return {"fit_search_id": sid, "fit_lr_selected_by": by, "init_scheme": scheme}
    rank = lora_rank if lora_rank is not None else lora_rank_for(key, arm, hpo_win)
    modules = row.get("lora_modules", PAPER_LORA_MODULES)
    if (not isinstance(modules, list) or not modules
            or not all(isinstance(m, str) for m in modules)):
        raise SystemExit(f"[make-sdfbase-configs] {key}/{arm} fold carries "
                         f"lora_modules={modules!r}, not a list of names")
    unknown = sorted(set(modules) - set(LORA_MODULE_NAMES))
    if unknown:
        raise SystemExit(f"[make-sdfbase-configs] {key}/{arm} fold names "
                         f"unknown lora module(s) {unknown} (known: "
                         f"{list(LORA_MODULE_NAMES)})")
    modules = sorted(set(modules))
    stated = row.get("per_expert_rank")
    if stated is not None and (not isinstance(stated, dict) or set(stated) !=
                               {"shared_expert", "routed_expert"}):
        raise SystemExit(f"[make-sdfbase-configs] {key}/{arm} fold carries "
                         f"per_expert_rank={stated!r}; expected null or "
                         "{shared_expert, routed_expert}")
    # stamp-fix-3: ONE resolver. A row's `per_expert_rank: null`
    # (the sweep's and the probe's spelling) means DERIVED by FD 6's rule — the SAME
    # rule the object construction applies (lora_modules.per_expert_rank_map; the
    # trainer validates the explicit map against the loaded model's active-expert
    # count and refuses at build) — from the lane's declared moe_active_experts, so
    # the rendered selection, the trainer kwarg and the saved artefact carry the same
    # explicit map and the publish parity cannot disagree by convention. A stated
    # map must equal the derived one; any map on a dense / attention-only lane refuses.
    derived = per_expert_rank_for(key, rank, modules)
    if stated is not None and {k: int(v) for k, v in stated.items()} != derived:
        raise SystemExit(f"[make-sdfbase-configs] {key}/{arm} fold states "
                         f"per_expert_rank={stated!r} but FD 6's rule "
                         f"({PER_EXPERT_RANK_RULE}; moe_active_experts="
                         f"{MODELS[key].get('moe_active_experts')}) derives "
                         f"{derived!r} for rank {rank} on {modules} — refusing a "
                         "selection the construction would not build")
    per = derived
    ruling = row.get("architecture_selection_ruling_id")
    if ruling is not None and not isinstance(ruling, str):
        raise SystemExit(f"[make-sdfbase-configs] {key}/{arm} fold carries "
                         f"architecture_selection_ruling_id={ruling!r}, not a string")
    block = {"fit_search_id": sid, "lora_rank": rank, "lora_modules": modules,
             "per_expert_rank": per, "fit_lr_selected_by": by,
             "architecture_selection_ruling_id": ruling, "init_scheme": scheme}
    if per is not None:      # the rule and its divisor ride the sidecar: a reader sees what was trained and why
        block["per_expert_rank_rule"] = PER_EXPERT_RANK_RULE
        block["moe_active_experts"] = int(MODELS[key]["moe_active_experts"])
    return block


def _arm_block(key: str, arm: str, memprobe_bs, hpo_win: dict | None) -> dict:
    """One train_kwargs.<arm> block. HPO placeholders float()-crash if a
    wave launches before --fold-hpo; floor_ratio_max is EXPLICITLY null
    (the SI-teacher probe is meaningless for CE — replaced, never silently
    disabled) with the CE-native bar on ce_floor_ratio_max.
    A fold row may carry the optimizer winner (all three HPO_WINNER_KEYS),
    the selection guard's record, and the fit-selection keys
    (HPO_SELECTION_KEYS); the `fit_selection` block is emitted iff
    emits_fit_selection, so every earlier lane renders byte-identically."""
    if hpo_win is not None:
        unknown = sorted(set(hpo_win) - set(HPO_FOLD_KEYS))
        if unknown:
            raise SystemExit(
                f"[make-sdfbase-configs] HPO fold row for {key}/{arm} carries "
                f"unknown key(s) {unknown} (known: {list(HPO_FOLD_KEYS)})")
    win = _winner_of(key, arm, hpo_win)
    block: dict = {
        "lr": float(win["lr"]) if win else HPO_SENTINEL,
        "lr_schedule": "warmup_cosine",
        "warmup_frac": (float(win["warmup_frac"]) if win
                        else HPO_SENTINEL),
        "max_grad_norm": (float(win["max_grad_norm"]) if win
                          else HPO_SENTINEL),
        "weight_decay": 0.01,
    }
    if arm == "psr_ce_all":
        # The anti-dead-gate protection both PSR objectives carry.
        block["gate_reg"] = 0.1
    if arm == "lora_r4_ce":
        # the fold's lora_rank when the search or a manual pick set one
        # (paper-companion-repo FD 6a), else the paper's 4 — byte-identical
        # for every lane without a fold row
        block["lora_rank"] = lora_rank_for(key, arm, hpo_win)
    sel = (fit_selection_block(key, arm, hpo_win, lora_rank=block.get("lora_rank"))
           if emits_fit_selection(key, hpo_win) else None)
    # stamp-fix-2: the ARCHITECTURE the trainer
    # builds rides the arm block TOP-LEVEL, like lora_rank and the optimizer
    # winner do. The fitter POPS `fit_selection` (driver-consumed: serialised
    # to the cell's sidecar, never a trainer kwarg), so a selection that lived
    # only there was PUBLISHED and never TRAINED — the trainer built its
    # defaults (o_proj / zero init) under a sidecar claiming the pick. Each
    # key is emitted iff the fold row STATES it (a lane without a row renders
    # byte-identically; the trainer's defaults are the paper's), with the
    # validated value the sidecar carries, so the two cannot disagree at the
    # source. init_scheme is promoted for BOTH CE arms.
    row = hpo_win or {}

    def _own(v):      # a COPY, never the sidecar's object: PyYAML would otherwise emit an anchor/alias pair (&id001 / *id001)
        return list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v)
    if arm == "lora_r4_ce":
        for k in ("lora_modules", "per_expert_rank"):
            # per_expert_rank ALSO when the sidecar RESOLVED a map from a row that
            # did not state the key (stamp-fix-3): the trainer receives exactly the
            # map the sidecar claims, never a null it would re-derive on its own
            if k in row or (k == "per_expert_rank" and sel is not None and sel.get(k) is not None):
                block[k] = _own(sel[k] if sel is not None else row[k])
    if "init_scheme" in row:
        block["init_scheme"] = sel["init_scheme"] if sel is not None else row["init_scheme"]
    block["accum_steps"] = _accum_for(memprobe_bs, effective_batch_for(key))
    block["floor_ratio_max"] = None
    block["ce_floor_ratio_max"] = 1.1
    if MODELS[key]["gradient_checkpointing"]:
        block["gradient_checkpointing"] = True
    if sel is not None:
        block["fit_selection"] = sel
    return block


def _validate_fold(fold: dict | None, tag: str) -> None:
    if not fold:
        return
    for key, arms in fold.items():
        if key not in MODELS:
            raise SystemExit(f"[make-sdfbase-configs] {tag} names unknown "
                             f"model key {key!r} (known: {sorted(MODELS)})")
        for arm in arms:
            if arm not in CE_ARMS:
                raise SystemExit(f"[make-sdfbase-configs] {tag} names "
                                 f"unknown arm {arm!r} for {key} "
                                 f"(CE arms: {list(CE_ARMS)})")
            # The HPO selection guard (scripts/hpo_selection_guard.py,
            # written by the fold writers) rides a folded arm as its
            # "selection_guard" sub-dict. _arm_block reads the winner
            # (lr / warmup_frac / max_grad_norm) and the fit-selection keys
            # (HPO_SELECTION_KEYS), never the guard record, so the block is
            # inert for the rendered config — but an ALARMED record nobody acknowledged
            # (the fold writer's --allow-alarmed) must never fold silently:
            # refuse at fold time and on every load of the store. Contract
            # keys: alarmed (bool), acknowledged ({by, at} | None).
            val = arms[arm] if isinstance(arms, dict) else None
            guard = val.get("selection_guard") if isinstance(val, dict) else None
            if isinstance(guard, dict) and guard.get("alarmed") \
                    and not guard.get("acknowledged"):
                codes = sorted({alarm_code(a)
                                for a in guard.get("alarms") or []})
                raise SystemExit(
                    f"[make-sdfbase-configs] {tag}: {key}/{arm} carries an "
                    f"UNACKNOWLEDGED selection-guard alarm ({codes}) — the "
                    "selected point is not interior to its search, or the "
                    "fit gave no interior minimum; re-run the fold writer "
                    "with --allow-alarmed to acknowledge it deliberately, "
                    "or pick an interior winner")


def build_config(key: str, memprobe: dict | None = None,
                 hpo: dict | None = None) -> dict:
    """The pure derivation: model key (+ optional folded measurements) ->
    the config mapping. Key order is load-bearing for the deterministic
    render (yaml sort_keys=False)."""
    if key not in MODELS:
        raise SystemExit(f"[make-sdfbase-configs] unknown model key {key!r} "
                         f"(known: {sorted(MODELS)})")
    spec = MODELS[key]
    _validate_caps_override(key)
    mp = (memprobe or {}).get(key) or {}
    hp = (hpo or {}).get(key) or {}
    cfg: dict = {
        "model": spec["model"],
        "model_revision": spec["revision"],
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "layer": spec["layer"],
        "max_len": 8192,
        "data_mode": "documents",
        # Budget provenance schema — NEVER the legacy 512
        # completion default: the transfer/budget check branches on
        # training_budget_kind, so a document_tokens fit validates against
        # sequence_max_len + the sample manifest.
        "training_budget_kind": "document_tokens",
        "sequence_max_len": 8192,
        "completion_tokens": None,
        # CONFIG-LEVEL (the key the driver actually reads —
        # train_two_si_objects reads cfg["teacher_prefix_cache"], default
        # True; a train_kwargs.<arm> copy would read as set and behave as
        # unset, and tune_arm's CE shape guard refuses it there). False is
        # the only honest value for the CE arms — no teacher forward
        # exists to cache — and it is REQUIRED for qwen, where
        # gradient_checkpointing is set: maybe_enable_gradient_checkpointing
        # refuses checkpointing + a true (defaulted) prefix cache together
        # (HF forces use_cache=False under train-mode checkpointing).
        "teacher_prefix_cache": False,
        # The instruct-direct lanes declare their base sibling's
        # leading-special-token convention (LEADING_TOKEN_POLICIES); the
        # key is EMITTED ONLY when declared, so the frozen configs' bytes
        # never move.
        **({LEADING_TOKEN_KEY: _leading_policy(key)}
           if _leading_policy(key) else {}),
        # the data-parallel fit width (top-level, the key the fit
        # sbatch greps and the trainer's gate reads) — EMITTED ONLY when
        # the lane declares it, same rule as the seed and the leading
        # tokens, so every earlier config renders byte-identically.
        **({"fit_width": fit_width_for(key)}
           if fit_width_for(key) is not None else {}),
        "si_ids": si_ids_for(key),
        "batch_size_by_arm": {
            arm: (int(mp[arm]) if arm in mp else MEMPROBE_SENTINEL)
            for arm in CE_ARMS
        },
        "train_kwargs": {
            "default": {
                "holdout_frac": 0.0,
                "heldout_pool": heldout_pool_path(key),
                "early_stop_min_epochs": 1,
                "early_stop_min_delta": 0.0,
                "positions": "all",
                "probe_n": None,
                # The seed-replicate lanes (2026-09-15) declare their seed;
                # EMITTED ONLY when declared, so the frozen configs' bytes
                # never move and their fits keep the trainer's default 0.
                **({"seed": seed_for(key)} if seed_for(key) is not None
                   else {}),
            },
            **{arm: _arm_block(key, arm, mp.get(arm), hp.get(arm))
               for arm in CE_ARMS},
        },
        "train_kwargs_by_si": {si: caps_for(key, si)
                               for si in si_ids_for(key)},
        "wandb_fits": True,
        "wandb_project": _wandb_project_for(key),
    }
    return cfg


def render_config(key: str, cfg: dict) -> str:
    """Deterministic header + mapping — no timestamps, so the config's
    expect_sha (sha256 of the file bytes) is stable across regenerations
    with the same inputs."""
    filled_bs = all(not isinstance(v, str)
                    for v in cfg["batch_size_by_arm"].values())
    filled_hpo = all(
        not isinstance(cfg["train_kwargs"][arm]["lr"], str)
        for arm in CE_ARMS)
    # The base models' header wording is byte-pinned by the committed
    # files; a chat_model lane (GLM-4.6V-Flash, which has no base sibling)
    # says what it is instead of calling a chat checkpoint a base model;
    # an entry with its own header_kind (the instruct-direct lanes, whose
    # base siblings DO exist) says exactly that instead.
    spec = MODELS[key]
    kind = spec.get("header_kind") or (
        "CHAT model; no base variant of it exists"
        if spec.get("chat_model") else "BASE model")
    eff = effective_batch_for(key)
    lines = [
        f"# {config_filename(key)} — GENERATED, do not hand-edit.",
        "# Regenerate: python scripts/sdfbase/make_sdfbase_configs.py"
        " [--fold-memprobe m.json] [--fold-hpo w.json]",
        "# (this generator writes ONLY"
        " configs/sdfbase_*.yaml)",
        "#",
        f"# model: {cfg['model']} @ {cfg['model_revision']} ({kind} —"
        " no chat template is",
        "#   ever applied: data_mode: documents renders flat text via"
        " docs_to_ce_examples)",
        "# budget schema: training_budget_kind: document_tokens;"
        " completion_tokens: null",
        "#   EXPLICITLY (never the legacy 512 default — false provenance"
        " the budget",
        "#   check would read as the real training cap).",
        "# teacher_prefix_cache: false at CONFIG level (the key the driver"
        " reads; default",
        "#   True otherwise) — no CE teacher exists to cache, and the qwen"
        " lane's",
        "#   gradient_checkpointing refuses to run with the cache on.",
        "# floor: floor_ratio_max: null (the SI-teacher probe is"
        " meaningless for CE) with",
        "#   the CE-native bar ce_floor_ratio_max: 1.1 — a degeneration"
        " guard, not a",
        "#   did-it-help gate (cells that merely fail to help land at"
        " ~1.0 and publish).",
        f"# batch_size_by_arm: {'MEASURED (memprobe folded)' if filled_bs else 'MEMPROBE sentinels — int()-crash until'}",
        ("#   the measured widths are folded via --fold-memprobe."
         if not filled_bs else
         f"#   accum_steps = ceil({eff} / bs) — effective"
         f" batch {eff}."),
        f"# train_kwargs lr/warmup_frac/max_grad_norm: "
        f"{'HPO winners folded' if filled_hpo else 'PENDING sentinels — float()-crash until --fold-hpo'}",
        *([f"# leading_special_tokens: {_leading_policy(key)} — the base"
           " sibling's template-free convention, applied at tokenizer load"
           " and asserted on every encoded example before training"] if _leading_policy(key) else []),
        *([f"# seed: {seed_for(key)} (train_kwargs.default.seed) — a SEED"
           " REPLICATE: seeds the LoRA A-factor init, the trainer's per-epoch"
           " shuffle and the HPO trial; the held-out carve under"
           " /vol/pool/<key>/ is this seed's own (2026-09-15)"]
          if seed_for(key) is not None else []),
        *([f"# fit_width: {cfg['fit_width']} — DATA-PARALLEL fits:"
           f" the fit sbatch requests gpu:{cfg['fit_width']} and runs the"
           " arm under torchrun; the"]
          + ["#   ranks group-shard every optimizer step's micro-batches"
             " (effective batch unchanged, accum_steps/width per rank)"]
          if "fit_width" in cfg else []),
        *(_fit_selection_header(cfg)),
        "",
    ]
    body = yaml.safe_dump(cfg, sort_keys=False, width=79,
                          default_flow_style=False, allow_unicode=True)
    return "\n".join(lines) + body


def _fit_selection_header(cfg: dict) -> list[str]:
    """The header line naming the per-arm fit_selection — only when a
    block is present, so every earlier lane's header is
    byte-identical."""
    tk = cfg["train_kwargs"]
    sel = {arm: tk[arm].get("fit_selection") for arm in CE_ARMS}
    if not any(sel.values()):
        return []
    lora = sel.get("lora_r4_ce") or {}
    schemes = "/".join(str((sel.get(a) or {}).get("init_scheme", "-"))
                       for a in CE_ARMS)
    return [
        "# fit_selection (per arm; the trainer serialises it to"
        " fit_selection.json, the registry hook records it on the object):"
        f" init_scheme psr/lora = {schemes};",
        f"#   lora rank {lora.get('lora_rank', '-')} on"
        f" {','.join(lora.get('lora_modules') or []) or '-'}"
        f" (per_expert_rank {lora.get('per_expert_rank')}), selected by"
        f" {lora.get('fit_lr_selected_by', '-')}; search ids ride the"
        " folds store's hpo rows (sentinel until the search closes)",
    ]


def guarded_write(path: Path, text: str, repo: Path = REPO) -> None:
    """The ONLY write path. Hard-refuses any target that is not
    <repo>/configs/sdfbase_*.yaml or the folds store — the two_si
    instruct configs (and anything else) are untouchable from here by
    construction."""
    cfg_dir = (repo / "configs").resolve()
    rp = Path(path)
    rp = (rp if rp.is_absolute() else (repo / rp)).resolve()
    if rp.parent != cfg_dir or not _ALLOWED_NAME.fullmatch(rp.name):
        raise SystemExit(
            f"[make-sdfbase-configs] REFUSING to write {path} — this "
            "generator writes ONLY configs/sdfbase_*.yaml and "
            f"configs/{FOLDS_FILENAME} (a byte change to any two_si_* "
            "config would invalidate finished instruct cells' expect_sha "
            "and fit fingerprints)")
    rp.write_text(text)


def load_folds(path: Path | None = FOLDS_PATH) -> dict:
    """The declarative fold store. Missing file (or path None) -> empty
    sections, i.e. every fold-fed key keeps its sentinel. Unknown
    sections, model keys, or arms refuse loudly."""
    if path is None or not Path(path).exists():
        return {"memprobe": {}, "hpo": {}}
    folds = json.loads(Path(path).read_text())
    unknown = set(folds) - {"memprobe", "hpo"}
    if unknown:
        raise SystemExit(
            f"[make-sdfbase-configs] {path} carries unknown fold "
            f"section(s) {sorted(unknown)} — known: ['hpo', 'memprobe']")
    folds = {"memprobe": folds.get("memprobe") or {},
             "hpo": folds.get("hpo") or {}}
    _validate_fold(folds["memprobe"], f"{FOLDS_FILENAME}[memprobe]")
    _validate_fold(folds["hpo"], f"{FOLDS_FILENAME}[hpo]")
    return folds


def dump_folds(folds: dict) -> str:
    """Deterministic serialization — same folds, same bytes."""
    return json.dumps(folds, indent=2, sort_keys=True) + "\n"


def update_folds(section: str, data: dict, path: Path = FOLDS_PATH,
                 repo: Path = REPO, *, merge_arm: bool = False) -> dict:
    """Deep-merge ``data`` ({model: {arm: ...}}) into one section of the
    folds file (per model, per arm — a partial fold never drops the other
    lanes' recorded values), write it back through guarded_write, and
    return the merged store. ``merge_arm`` (the init-scheme override)
    merges INTO an existing arm row instead of replacing it, so a stamped
    key never drops a recorded winner."""
    if section not in ("memprobe", "hpo"):
        raise SystemExit(f"[make-sdfbase-configs] unknown fold section "
                         f"{section!r} (known: ['hpo', 'memprobe'])")
    _validate_fold(data, f"--fold-{section}")
    folds = load_folds(path if Path(path).exists() else None)
    for key, arms in (data or {}).items():
        row = folds[section].setdefault(key, {})
        for arm, val in arms.items():
            if merge_arm and isinstance(val, dict) and isinstance(row.get(arm), dict):
                row[arm] = {**row[arm], **val}
            else:
                row[arm] = val
    _validate_fold(folds[section], f"{FOLDS_FILENAME}[{section}]")
    guarded_write(Path(path), dump_folds(folds), repo=repo)
    return folds


def stamp_init_scheme(scheme: str, keys: list[str], path: Path = FOLDS_PATH,
                      repo: Path = REPO) -> dict:
    """``--init-scheme-override``: record ``scheme`` as both CE arms'
    ``init_scheme`` in the named lanes' "hpo" fold rows (merged into any
    recorded winner) so the rendered fit_selection blocks — and the
    committed bytes a bare build reproduces — carry the run override (spec
    FD 7 / FD 18: the default stays the paper's zero; this run's overrides
    file selects the random scheme). Only a lane that DECLARES
    fit_selection may be stamped: a stamp on any other lane would emit a
    block into a byte-pinned config."""
    if scheme not in INIT_SCHEMES:
        raise SystemExit(f"[make-sdfbase-configs] --init-scheme-override "
                         f"{scheme!r} is not one of {list(INIT_SCHEMES)}")
    if not keys:
        raise SystemExit("[make-sdfbase-configs] --init-scheme-override needs "
                         "--models K [K ...] naming the lanes to stamp")
    for key in keys:
        if key not in MODELS:
            raise SystemExit(f"[make-sdfbase-configs] unknown model key {key!r}")
        if not MODELS[key].get("fit_selection"):
            raise SystemExit(
                f"[make-sdfbase-configs] REFUSING to stamp init_scheme onto "
                f"{key}: it declares no fit_selection, so the stamp would put "
                "a block into a byte-pinned config (the lane's fit identity)")
    data = {key: {arm: {"init_scheme": scheme} for arm in CE_ARMS} for key in keys}
    return update_folds("hpo", data, path=path, repo=repo, merge_arm=True)


def check_models_table(path: Path | None) -> list[str]:
    """``--models-table``: cross-check every lane's model + revision against
    reproduce/models.yaml (the ONE place a model is described).
    A lane is matched to the table row whose fit.checkpoint.model_id IS the
    lane key (fit slugs are the table's fit model_ids, `_base` and all); a
    lane the table does not list is a note (the sixteen earlier lanes are
    not front-door models). Returns the notes printed; REFUSES a revision
    that disagrees. Tolerant of an absent table (a note, not a refusal) so
    a bare tree still generates."""
    notes: list[str] = []
    if path is None:
        return notes
    path = Path(path)
    if not path.exists():
        notes.append(f"models table {path} absent — pins not cross-checked")
        return notes
    doc = yaml.safe_load(path.read_text()) or {}
    rows = (doc.get("models") or {}) if isinstance(doc, dict) else {}
    by_id: dict[str, tuple[str, dict]] = {}
    for tkey, row in rows.items():
        ck = ((row or {}).get("fit") or {}).get("checkpoint") or {}
        if ck.get("model_id"):
            by_id[str(ck["model_id"])] = (tkey, ck)
    for key, spec in MODELS.items():
        hit = by_id.get(key)
        if hit is None:
            notes.append(f"{key}: no models.yaml row with fit.checkpoint."
                         f"model_id == {key!r} (not a front-door fit model)")
            continue
        tkey, ck = hit
        if ck.get("hf_repo") != spec["model"] or ck.get("revision") != spec["revision"]:
            raise SystemExit(
                f"[make-sdfbase-configs] {key} pins {spec['model']}@"
                f"{spec['revision']} but models.yaml row {tkey!r} says "
                f"{ck.get('hf_repo')}@{ck.get('revision')} — one pin, one "
                "table; fix whichever is stale")
        notes.append(f"{key}: models.yaml row {tkey!r} agrees "
                     f"({spec['model']}@{spec['revision'][:12]})")
    return notes


def build_all(memprobe: dict | None = None,
              hpo: dict | None = None,
              folds_path: Path | None = FOLDS_PATH) -> dict[str, str]:
    """{filename: rendered text} for every campaign config. ``memprobe`` /
    ``hpo`` default (None) to the committed folds file's sections, so a
    bare call reproduces the committed folded bytes; pass a dict to
    override a section, or ``folds_path=None`` for a fully-unfolded
    (all-sentinel) build."""
    folds = load_folds(folds_path)
    if memprobe is None:
        memprobe = folds["memprobe"]
    if hpo is None:
        hpo = folds["hpo"]
    _validate_fold(memprobe, "--fold-memprobe")
    _validate_fold(hpo, "--fold-hpo")
    return {config_filename(key): render_config(key, build_config(
        key, memprobe=memprobe, hpo=hpo)) for key in MODELS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", choices=sorted(MODELS), default=None,
                    help="the lanes to render / stamp (default: every lane)")
    ap.add_argument("--fold-memprobe", type=Path, default=None,
                    help="JSON {model_key: {arm: batch_size}} of measured "
                         "memprobe widths (sdfbase_fits.py --mode memprobe "
                         "prints it fold-ready) — deep-merged into "
                         f"configs/{FOLDS_FILENAME} before the rebuild")
    ap.add_argument("--fold-hpo", type=Path, default=None,
                    help="JSON {model_key: {arm: {lr, warmup_frac, "
                         "max_grad_norm}}} of HPO winners — deep-merged "
                         f"into configs/{FOLDS_FILENAME} before the rebuild")
    ap.add_argument("--check", action="store_true",
                    help="regenerate and DIFF against the committed files; "
                         "exit 1 on drift, write nothing")
    ap.add_argument("--init-scheme-override", choices=INIT_SCHEMES,
                    default=None, metavar="SCHEME",
                    help="stamp SCHEME as both CE arms' fit_selection."
                         "init_scheme on the lanes named by --models (merged "
                         f"into configs/{FOLDS_FILENAME}[hpo], never replacing "
                         "a winner) — the run-overrides file's "
                         "OVERRIDE_INIT_SCHEME (spec FD 7 / FD 18)")
    ap.add_argument("--models-table", type=Path, default=None,
                    metavar="reproduce/models.yaml",
                    help="cross-check every lane's model + revision against "
                         "the models table (absent table = a note; a "
                         "disagreeing pin refuses)")
    args = ap.parse_args(argv)
    if args.fold_hpo is not None:
        # the pick the fold came from may be HELD (03_hpo.sh --phase pick found its basis changed after the promotion) — a
        # reader-side check, fail closed: nothing is merged while hpo_pick.HOLD.json stands beside the fold
        hold = Path(args.fold_hpo).parent / "hpo_pick.HOLD.json"
        if hold.exists():
            try:
                why = json.loads(hold.read_text()).get("reason")
            except Exception:                                          # pragma: no cover
                why = "(unreadable hold)"
            sys.exit(f"[make-sdfbase-configs] REFUSED: the LoRA architecture pick behind {args.fold_hpo} is HELD ({hold}: {why}) — "
                     "nothing merged; re-run 03_hpo.sh --phase pick to re-derive the promotion from the current evidence")

    if args.check and (args.fold_memprobe or args.fold_hpo
                       or args.init_scheme_override):
        ap.error("--check is read-only; run the fold first, then --check")
    if args.init_scheme_override is not None:
        if args.models is None:   # an EXPLICIT list of every lane is a request, not a default
            ap.error("--init-scheme-override needs --models K [K ...] "
                     "naming the lanes to stamp (never every lane)")
        stamp_init_scheme(args.init_scheme_override, list(args.models))
        print(f"[make-sdfbase-configs] stamped init_scheme="
              f"{args.init_scheme_override} into configs/{FOLDS_FILENAME}"
              f"[hpo] for {list(args.models)}")
    if args.models is None:
        args.models = sorted(MODELS)
    for note in check_models_table(args.models_table):
        print(f"[make-sdfbase-configs] models-table: {note}")
    # Folds are declarative repo data: the flags UPDATE the committed
    # folds file (deep-merge, guarded write), and every build — folded or
    # bare — renders from that one store, so committed bytes are always
    # reproducible from the tree alone.
    if args.fold_memprobe:
        update_folds("memprobe",
                     json.loads(args.fold_memprobe.read_text()))
        print(f"[make-sdfbase-configs] merged --fold-memprobe into "
              f"configs/{FOLDS_FILENAME}")
    if args.fold_hpo:
        update_folds("hpo", json.loads(args.fold_hpo.read_text()))
        print(f"[make-sdfbase-configs] merged --fold-hpo into "
              f"configs/{FOLDS_FILENAME}")
    built = build_all()

    rc = 0
    for key in args.models:
        name = config_filename(key)
        text = built[name]
        path = REPO / "configs" / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                print(f"[make-sdfbase-configs] DRIFT: configs/{name}")
                rc = 1
            else:
                print(f"[make-sdfbase-configs] ok: configs/{name}")
            continue
        guarded_write(path, text)
        print(f"[make-sdfbase-configs] wrote configs/{name} "
              f"(sha16 will pin the wave's expect_sha)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
