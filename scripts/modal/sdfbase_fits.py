"""sdfbase-fits — the sdf-base-steering campaign's STANDALONE Modal app:
document-CE fits of {psr_ce_all, lora_r4_ce} on the constv2
corpus cells, against the BASE models google/gemma-4-E4B and
Qwen/Qwen3.5-4B-Base, everything on H200, one-shot GPU containers, zero
idle (no warm pools).

Standalone ON PURPOSE, following the qwen35_4b_hpo.py per-lane precedent:
the deployed shared `twosi-fits` app can have other in-flight calls on it,
and a redeploy mutates it under them — scripts/modal/twosi_fits.py is
NEVER redeployed or edited by this campaign. The VOLUME is shared
deliberately (`twosi-fits`: the HF snapshots and the out/ bank namespace),
under fresh keys pool/{gemma4e4b_base,qwen35_4b_base}/ and
out/{gemma4e4b_base,qwen35_4b_base}/ — zero collision with instruct cells.

Cells: 9 data cells x 2 arms = 18 per model (cell ids ride the SI slot:
constv2-mix-{1k,10k,100k,1m,10m,full}, constv2-adv-{1k,100k},
constv2ctl-100k — the 10m/full pair is the round-2 scaling extension),
positions=all, cells_all leg, max_len 8192, shared held-out pool
pairs_constv2-heldout.jsonl (the config's train_kwargs.default.
heldout_pool).

Modes (run from the worktree so /repo bakes the intended commit; commit +
push BEFORE every wave launch — the image ships the worktree):

  modal run scripts/modal/sdfbase_fits.py --mode stage
      # both base snapshots -> /vol/hf (CPU+network only)
  # pool staging (the grading host, CPU): build pools + manifests +
  #   sidecars locally under $STAGE/sdfbase/pools/<key>/, then per file:
  #   modal volume put twosi-fits <local> pool/<key>/pairs_<cell>.jsonl
  #   (+ .tokcache.npz, + pairs_constv2-heldout.jsonl + sidecar)
  modal run scripts/modal/sdfbase_fits.py --mode onboard
      # one tiny GPU call per model: onboard_model_smoke.py --model-config
      # (does the base ship a chat template? does "" encode to [BOS]/[]?
      # do the layer walkers resolve?) — BEFORE any pool staging
  modal run scripts/modal/sdfbase_fits.py --mode memprobe [--model <key>]
      # trainer-variant re-probe (NEW suffixed artifact, folded one untouched):
      #   --mode memprobe --loss-impl chunked_ce [--grad-ckpt on|off] --bench-out m.json
      # bounded throughput bench on a staged pool (never publishes a cell):
      #   --mode bench --model <key> --arms psr_ce_all --sis constv2ctl2-full
      #     --batch-size <bs> [--accum N] --loss-impl full|chunked_ce [--grad-ckpt on|off]
      #     --steps 200 [--check-grads] --bench-out b.json
      #   + the gradient-norm reconnaissance's per-run overrides (tune_arm.py
      #     bench --lr/--max-grad-norm/--warmup-frac):
      #     --lr 4.5e-4 --max-grad-norm 1e6 --warmup-frac 0.05 --tag recon_1
      #     (a bench with ANY override is a reconnaissance run: --steps then
      #     defaults to, and must equal, recon_clip.RECON_STEPS = 150 — the
      #     recon reader accepts only exactly-budget records; the
      #     generic bench keeps its 200)
      #     -> /vol/out/bench/<key>/<arm>__<tag>.json with status ok|diverged|failed,
      #        grad_norm_preclip.{steps,values,quantiles_after_warmup}, loss
      # per (model x arm) doubling ladder at WORST-CASE rows (the longest
      # real docs of the 1M pool), positions all, max_len 8192, 0.85 VRAM
      # guard -> /vol/out/<key>/memprobe_<arm>_all.json; prints the
      # fold-ready JSON for make_sdfbase_configs.py --fold-memprobe
  # HPO (after fold-memprobe + commit): emit designs locally
  #   (tune_arm.py emit-trials --config configs/sdfbase_<key>_fits.yaml
  #    --arm <arm> --position all -> $STAGE/sdfbase/hpo/trials/
  #    r1_<key>_<arm>_all.json), volume put to hpo/trials/, then:
  modal deploy scripts/modal/sdfbase_fits.py
  modal run scripts/modal/sdfbase_fits.py --mode hpo-spawn [--model ...]
      # one durable FunctionCall per trial on the DEPLOYED app ->
      # /vol/out/hpo/<key>/<arm>_all/r1_tNN/; manifest
      # $STAGE/sdfbase/hpo_calls_r1.json (or --manifest <path>;
      # every row carries the design identity its summary.json must
      # echo, so fits_watch.py --manifest <that file> validates the
      # wave trial by trial — the same watcher as the fits, trials shape)
  # fits (after fold-hpo + commit + push + deploy):
  modal run scripts/modal/sdfbase_fits.py --mode snapshot --model <key> \
      --cells constv2-mix-1m__psr_ce_all,constv2-mix-1m__lora_r4_ce \
      --reason "round-2 cap-24 refit" [--verify-only]
      # BEFORE any refit spawn: copy each to-be-
      # superseded COMPLETE cell to /vol/out/<key>_r1/cells_all/<cell>/
      # (bank layout + _store_identity.json), verify file list + sha256
      # against its COMPLETE manifest, write superseded_by.json /
      # snapshot_of.json on both sides (scripts/sdfbase/snapshot_cells.py).
      # _fit_impl REFUSES to train over a COMPLETE cell of another
      # fingerprint that carries no supersede record. --verify-only
      # re-runs the verify half on an existing snapshot bank.
  modal run scripts/modal/sdfbase_fits.py --mode fits-spawn --model <key> \
      --sis constv2-mix-1m,constv2-mix-10m,constv2-mix-full
      # SCOPED spawn — --sis is REQUIRED: the round-2 config regeneration
      # moved config_sha for every cell, so an unscoped spawn would
      # retrain all 9 SIs from scratch; --retrain-all acknowledges exactly
      # that intent. expect_sha from the LOCAL config bytes; every
      # manifest row records the cell's EXPECTED fit fingerprint
      # (fingerprint_cpu, in-container, through the driver's own
      # derivation); durable spawns (EVERY requested cell — finished cells
      # self-skip in-container on the driver's fingerprint); manifest
      # $STAGE/sdfbase/fit_calls.json (a fresh --manifest per wave;
      # EVERY ROW records its own spawn pin — app / function / the deployed
      # function_id / config_sha / overwrite / checkpoint_every — so
      # fits_watch.py respawns a cell on the function it ran on, never on
      # the manifest-level values of whichever model was spawned last; the
      # row's function_id is the version pin it checks before any successor)
  modal run scripts/modal/sdfbase_fits.py --mode watch
      # the fits_watch.py contract: a cell counts COMPLETE only when its
      # marker's fit_fingerprint EQUALS the manifest row's expected one
      # (a stale round-1 marker under a refit in flight never reads as
      # done); exit 0 all COMPLETE / exit 2 respawn list / bounded by
      # --max-hours — or run scripts/modal/fits_watch.py --manifest
      # $STAGE/sdfbase/fit_calls.json directly under a persistent
      # Monitor
  # CONTINUE a cap-limited COMPLETE cell from its retained
  # <cell>/final_ckpt/ckpt.pt under a RAISED cap: the three 1M cells
  # that hit cap 24 again resume for up to 10 more epochs with early
  # stopping — never retrained. Two steps, explicit --cells only:
  modal run scripts/modal/sdfbase_fits.py --mode snapshot --model <key> \
      --cells constv2-mix-1m__psr_ce_all,constv2-mix-1m__lora_r4_ce \
      --bank-suffix _r2cap24 --continue-cap 34 \
      --reason "cap 24 -> 34 continuation"
      # the cap-24 publication (object, sidecars, metrics, final_ckpt/,
      # reports/) -> /vol/out/<key>_r2cap24/cells_all/<cell>/; the cell's
      # superseded_by.json names the CONTINUATION fingerprint as successor
      # (derived in-container from the retained checkpoint); a cell with
      # no final_ckpt/ is REFUSED here, before anything is spawned
  modal run scripts/modal/sdfbase_fits.py --mode fits-spawn --model <key> \
      --cells constv2-mix-1m__psr_ce_all,constv2-mix-1m__lora_r4_ce \
      --continue-cap 34 --manifest $STAGE/sdfbase/fit_calls_cont.json
      # --sis/--arms/--retrain-all/--overwrite are refused with
      # --continue-cap. Each container ADOPTS <cell>/final_ckpt/ckpt.pt
      # under the continuation fingerprint (driver --continue-from +
      # --epochs-cap, sisteer.extraction.continuation) and the trainer
      # resumes it: the re-run tail of the checkpointed epoch, then epochs
      # 25..34, the prior fit's early-stopping state restored so patience
      # can fire inside the continuation. Fingerprint = the prior identity
      # + {prior fingerprint, resume step/epoch, cap}: deliberately NOT a
      # scratch cap-34 fit's (warmup_cosine makes them different fits), so
      # the cap-24 marker never skips it and the watcher validates the row
      # against exactly the value the container stamps. lr: the cap-34
      # schedule from the resume step onward — a quantified warm restart
      # (the prior cosine had decayed to ~0); see the continuation module.
  # CHAINED continuation (a cell whose retained checkpoint is a
  # previous continuation's, e.g. gemma 1M lora cap 24 -> 34 -> 60) and the
  # lr multiplier. Same two steps; the checkpoint key is PROVEN from the
  # plain identity through the chain the COMPLETE marker records
  # (sisteer.extraction.continuation.resolve_chain), in fingerprint_cpu
  # and snapshot BEFORE any GPU spawn, and again by the driver in-container;
  # the new publication records the full chain. --continue-lr-mult m (default
  # 1.0) scales the restarted schedule's PEAK (warmup + cosine shape
  # unchanged), rides in the fingerprint (a different m is a different fit)
  # and in continued_from.lr_schedule; refused without --continue-cap.
  # Every continuation runs under stopper rule `replay_exempt` (the
  # replayed remainder of the checkpoint epoch is logged, never a strike nor
  # a new best; patience counts from the first FULL post-restart epoch) and
  # carries it in the marker — a same-cap same-multiplier re-issue of a
  # chained continuation predating that rule is therefore admissible:
  modal run scripts/modal/sdfbase_fits.py --mode snapshot --model <key> \
      --cells constv2-mix-1m__lora_r4_ce --bank-suffix _r2cap34b \
      --continue-cap 60 [--continue-lr-mult 0.25] --reason "..."
  modal run scripts/modal/sdfbase_fits.py --mode fits-spawn --model <key> \
      --cells constv2-mix-1m__lora_r4_ce --continue-cap 60 \
      [--continue-lr-mult 0.25] --manifest $STAGE/sdfbase/<file>.json
      # every row also pins the DEPLOYMENT (app_id / app_version /
      # deploy_commit — fits_watch.deployment_pin): a deployed function's
      # id does not change on redeploy (v10 -> v11 kept both fit ids), so
      # the watcher refuses a successor whose app version moved; rows
      # spawned before this pin existed need `fits_watch.py
      # --infer-legacy-app-version` (each row's spawn_time placed in the
      # deployment history — never one assumed number for a manifest).
  modal run scripts/modal/sdfbase_fits.py --mode probe-score --model <key>
      # one GPU container per model; scripts/sdfbase/probe_score.py
      # iterates all cells inside it (model loads once, objects swap).
      # --scale 2.0 = the round-2 2x-strength pass (object rows only,
      # margins land in probe_<ver>_x2/, gate rows skipped)
  modal run scripts/modal/sdfbase_fits.py --mode mmlu --model <key> \
      --cells none,constv2-mix-1m__psr_ce_all,... --run-label r2 \
      [--defenses-root /vol/out/{key}_r1]
      # plain-text 5-shot MMLU letter-logprob (scripts/sdfbase/
      # base_mmlu.py) at winner cells + 1M cells + bare base. --run-label
      # is REQUIRED: outputs land at /vol/out/<key>/mmlu/<label>/
      # mmlu_<label>_<cell>.{json,per_question.jsonl} carrying a shared
      # run identity (object sha, cais/mmlu revision, seed, ordered-qid
      # hash); --defenses-root points an r1 run at the SNAPSHOT bank
      # ({key} expands per model) so r1 and r2 outputs never share a path.
      # --scale 2.0 (with --run-label r2_x2) = the 2x-dose pass: the
      # object is applied at that strength, the prompt is unchanged, and
      # identity.scale records it (the analysis pairs r2_x2 against r2).
  modal run scripts/modal/sdfbase_fits.py --mode gsm8k --model <key> \
      --cells none,constv2-mix-1m__psr_ce_all,... --run-label r2 [--scale 2.0]
      # plain-text 8-shot chain-of-thought GSM8K, greedy, exact match
      # (scripts/sdfbase/base_gsm8k.py — the base-model sibling of the
      # battery's 0-shot chat control), one container per model; outputs
      # at /vol/out/<key>/gsm8k/<label>/gsm8k_<label>_<cell>.{json,
      # per_question.jsonl} with the same run-identity contract as mmlu
      # (+ exemplars sha, max_new_tokens, stop strings, per-row
      # termination / extraction-rule accounting). r2 = 1x on the live
      # bank, r2_x2 = --scale 2.0: paired within label on object_sha256.
  modal run scripts/modal/sdfbase_fits.py --mode ce --model <key> \
      --cells constv2-mix-1m__psr_ce_all[,...] --scales 1.0,2.0
      # standalone held-out CE (scripts/sdfbase/ce_score.py) for STAGED
      # cells at each scale on the three analysis pools (CE_POOLS): the
      # derived app's ce_command layout byte for byte —
      # /vol/out/ce_score/<key>/<cell>__<pool>__scale<s>.json — so the
      # analysis's 2x CE guard reads a fitted cell exactly as it reads a
      # derived object. One container per model.
  modal run scripts/modal/sdfbase_fits.py --mode battery --model <key> \
      --cells <explicit cells> [--attacks ...] [--nshards N] [--wave _r2|_it]
      #   [--shards 1] [--manifest fresh.json]  — a REPAIR spawn: the failed
      #   rows only (same --nshards slicing), into a fresh manifest; a
      #   battery manifest is never overwritten
      # instruct-transfer battery: two_si_run.py generate per
      # (attack x cell x shard) + cell_controls_run.py per cell (attack
      # id "controls") + the benign over-refusal pass per cell (attack id
      # "xstest"), against configs/sdfbase_transfer_<model>[<wave>].yaml
      # (--wave _r2 = the round-2 controls-in-battery namespace: the SAME
      # object bank, a fresh private_run/out_dir). The two per-cell ids
      # are UNSHARDED_BATTERY_ATTACKS: one container each, --nshards
      # ignored.

Resume: each fit container seeds its cell dir from the volume before
training, so a COMPLETE cell self-skips on the driver's fingerprint and a
killed container's successor resumes from the last committed checkpoint;
HPO trials self-skip on tune_arm's fit fingerprint. Re-running any spawn
mode is idempotent.

App name: the wave-A deployment `sdfbase-fits` is FROZEN — its
in-flight calls must never see a redeploy — so a later wave deploys AND
spawns under its own name, and every launcher path follows ONE resolved
name: --app-name > $SDFBASE_APP_NAME > DEFAULT_APP_NAME (`sdfbase-fits`;
the default path is unchanged byte for byte). The GLM / Nemotron wave:

  export SDFBASE_APP_NAME=sdfbase-fits-ws3
  modal deploy --name sdfbase-fits-ws3 scripts/modal/sdfbase_fits.py
  modal run scripts/modal/sdfbase_fits.py --mode stage --model <key>
  modal run scripts/modal/sdfbase_fits.py --mode hpo-spawn --model <key>
      # (or --app-name sdfbase-fits-ws3 on the run instead of the env;
      #  both set and DIFFERENT is refused — one wave, one name)

EVERY mode calls the DEPLOYED app: each spawn/remote in the entrypoint
goes through ONE helper, ``deployed_fn`` (modal.Function.from_name on
the resolved name), so the deploy above precedes the first `modal run`
of ANY mode — stage, onboard and memprobe included. No locally
decorated function object is ever spawned: those bind to the App
object, which is built at import from $SDFBASE_APP_NAME alone (before
any flag is parsed), so a mode that spawned them ran under the
ephemeral app of that import-time name and `--app-name` changed
nothing but the manifest. (A `modal run` still builds its ephemeral
app under the import-time name; nothing is called into it.) Every
manifest doc AND every row records the app it spawned on, so
fits_watch.py reads strictness and the re-spawn target off the row —
never off a default.
"""
import json
import math
import os
import re
import time
from pathlib import Path

import modal

#: The wave-A app name — the default every path resolves to when neither
#: --app-name nor $SDFBASE_APP_NAME is set.
DEFAULT_APP_NAME = "sdfbase-fits"


def resolve_app_name(app_name: str = "") -> str:
    """The deployed app this launcher targets: an explicit ``--app-name``
    wins, then ``$SDFBASE_APP_NAME``, then DEFAULT_APP_NAME.

    Both set AND different is REFUSED: the App object takes its name from
    the env at import, so a run whose flag disagrees with its env would
    build its ephemeral app under one name and look functions up under
    another — two provenance trails for one wave. Fix the shell or the
    flag; the launcher does not pick."""
    env = os.environ.get("SDFBASE_APP_NAME") or ""
    if app_name and env and app_name != env:
        raise SystemExit(
            f"--app-name {app_name!r} disagrees with $SDFBASE_APP_NAME="
            f"{env!r}: one wave, one app name — unset one of them")
    return app_name or env or DEFAULT_APP_NAME


APP_NAME = resolve_app_name()
app = modal.App(APP_NAME)

try:
    # Local (the grading host worktree): image defs + the entrypoint's config-sha
    # pin read real paths. In the CONTAINER this module re-imports from a
    # path with <3 parents and the lookup IndexErrors — the container never
    # needs the worktree (everything lives at /repo), so fall back.
    WORKTREE = Path(__file__).resolve().parents[2]
except IndexError:
    WORKTREE = Path("/repo")

#: The 9 data cells (the SI slot of the <si>__<arm> cell scheme).
#: constv2-mix-{10m,full} are the round-2 token-scaling extension. NOTE
#: for respawns: the round-2 config regeneration changed BOTH configs'
#: bytes, so the fit fingerprint's config_sha moved for EVERY cell — an
#: all-SIS fits-spawn would retrain all of them. fits-spawn therefore
#: REFUSES without --sis (the ordered wave: constv2-mix-1m,constv2-mix-10m,
#: constv2-mix-full) unless --retrain-all states that a full retrain is
#: the intent.
SIS = [
    "constv2-mix-1k",
    "constv2-mix-10k",
    "constv2-mix-100k",
    "constv2-mix-1m",
    "constv2-mix-10m",
    "constv2-mix-full",
    "constv2-adv-1k",
    "constv2-adv-100k",
    "constv2ctl-100k",
    # The r2 register-matched control at the mix-full budget, under its
    # own prefix. Its row moved every cell's config_sha again; the --sis
    # scoping rule above is unchanged.
    "constv2ctl2-full",
]

#: The two CE arms — every cell runs positions=all (cells_all leg).
CE_ARMS = ["psr_ce_all", "lora_r4_ce"]

#: The HPO reference cell + its memprobe/worst-case pool.
HPO_SI = "constv2-mix-100k"
MEMPROBE_SI = "constv2-mix-1m"
#: The shared held-out pool's cell id — the CE arms' whole verdict
#: (train_kwargs.default.heldout_pool names the same file).
HELDOUT_SI = "constv2-heldout"
#: `_eb64` sibling -> the frozen `_it_doc` parent whose pool dir it shares
#: (MODELS[...]["pool_key"], validated by _pool_key).
EB64_PARENT = {
    "gemma4e4b_it_doc_eb64": "gemma4e4b_it_doc",
    "qwen35_4b_it_doc_eb64": "qwen35_4b_it_doc",
}

#: Trial designs staged at /vol/hpo/trials (emitted locally by
#: tune_arm.py emit-trials from the sdfbase fits configs, so the
#: config_sha the trials stamp matches /repo's config bytes).
HPO_TRIALS_DIR = "/vol/hpo/trials"

MODELS = {
    # The base (non-instruct) siblings, both on H200.
    "gemma4e4b_base": {
        "gpu": "H200",
        "cfg": "/repo/configs/sdfbase_gemma4e4b_base_fits.yaml",
        "repo": "google/gemma-4-E4B",
        "rev": "411aa17b749aa952df1359d2dcea73917a544d9a",
        "gated": True,
    },
    "qwen35_4b_base": {
        "gpu": "H200",
        "cfg": "/repo/configs/sdfbase_qwen35_4b_base_fits.yaml",
        "repo": "Qwen/Qwen3.5-4B-Base",
        "rev": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "gated": True,
    },
    # The INSTRUCT siblings, PROBE-SCORE ONLY (the probe's positive
    # control is the instruct model's in-context-constitution margin
    # shift vs instruct-bare — we KNOW instructs follow context).
    # probe_only=True keeps them out of every other mode's default key set:
    # they have no fits config, no pools, no bank. Revisions = the twosi
    # campaign pins (twosi_fits.py MODELS, verified).
    "gemma4e4b_it": {
        "gpu": "H200",
        "cfg": None,
        "repo": "google/gemma-4-E4B-it",
        "rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "gated": True,
        "probe_only": True,
        # instruct gate rows render through the model's native chat
        # template (probe_score --render chat), matching the campaign's
        # prompted-baseline convention.
        "probe_render": "chat",
    },
    "qwen35_4b_it": {
        "gpu": "H200",
        "cfg": None,
        "repo": "Qwen/Qwen3.5-4B",
        "rev": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "gated": True,
        "probe_only": True,
        "probe_render": "chat",
    },
    # GLM-4.6V-Flash (2026-09-02): the `_doc` lane — no base variant of
    # GLM-4.6V exists, so the doc-CE fits target the CHAT model itself
    # (config generated by make_sdfbase_configs.py; MEMPROBE / HPO
    # sentinels until its folds land). MIT, UNGATED; a dense 10.3B VLM
    # used text-only with full attention on all 40 layers, so it rides
    # image_std (the fused-kernel image is for the hybrid Qwen3.5 line
    # only). Loads as Glm4vForConditionalGeneration via
    # models.auto_model_class (no causal-LM auto-mapping in transformers
    # 5.14.1). probe_render: chat — GLM IS the instruct model, so its
    # probe pass renders every row (the two gate rows AND the fitted-cell
    # rows) through its native template, the instruct convention; no
    # probe_only sibling is needed. explicit_only: kept OUT of the bare
    # default key set while the gemma/qwen rounds are in flight (a bare
    # --mode hpo-spawn / probe-score / memprobe would otherwise pull in a
    # model with no designs, pools or bank yet); every GLM mode names
    # --model glm46vflash_doc. Drop the flag once its folds land.
    "glm46vflash_doc": {
        "gpu": "H200",
        "cfg": "/repo/configs/sdfbase_glm46vflash_doc_fits.yaml",
        "repo": "zai-org/GLM-4.6V-Flash",
        "rev": "411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
        "gated": False,
        "probe_render": "chat",
        "explicit_only": True,
    },
    # Nemotron-3-Nano-4B (2026-09-02; NVIDIA Open Model License — NOT gated,
    # no HF token needed). "_doc" because no base variant exists: the
    # instruct checkpoint IS the documents-lane model. The key is the
    # Nemotron lane's pool / out namespace on the volume (/vol/pool/<key>/,
    # /vol/out/<key>/ — where the paper's full-scale Nemotron cells were
    # staged); its config is the FULL-SCALE lane's (the paper's 17 Nemotron
    # objects; make_sdfbase_configs.py `nemotron3nano4b_doc_full`, whose
    # heldout_pool names this dir) since the data-scaling ladder config
    # `sdfbase_nemotron3nano4b_doc_fits.yaml` was retired on 2026-10-08.
    # Every GPU function for it binds image_nemotron (NEMOTRON_IMAGE_KEYS):
    # the hybrid-stack image plus the mamba-ssm fused kernels, so the 21
    # Mamba2 blocks run cuda_kernels_forward, not the pure-torch SSD scan.
    "nemotron3nano4b_doc": {
        "gpu": "H200",
        "cfg": "/repo/configs/sdfbase_nemotron3nano4b_doc_full_fits.yaml",
        "repo": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "rev": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        "gated": False,
        # Nemotron IS the instruct model (like GLM), so its probe pass
        # renders EVERY row — gate rows and fitted-cell rows — through its
        # native ChatML template; no probe_only sibling is needed.
        # explicit_only: named with --model in every mode, never swept
        # into a peer's bare invocation.
        "probe_render": "chat",
        "explicit_only": True,
    },
    # INSTRUCT-DIRECT arms (2026-09-03): the two instruct
    # checkpoints the transfer battery serves — SAME repo + pin as the
    # probe_only gemma4e4b_it / qwen35_4b_it entries above (= the r2
    # transfer configs' model_revision) — as FIT-CAPABLE `_it_doc` keys:
    # the corpus is distilled into the instruct model DIRECTLY (doc-CE,
    # the chat template bypassed in training exactly as the chat
    # lanes), the head-to-head against base-fit + zero-shot transfer.
    # Own generated config, own pools (/vol/pool/<key>/ — own tokcache
    # sidecars too: a sidecar pins model_id, so the base lane's cannot be
    # reused), own bank out/<key>/. probe_render chat: the instruct model
    # IS the served model, gate rows and object rows in one render.
    # explicit_only: named with --model in every mode, never swept into a
    # bare invocation. Image variant = the architecture's (std / qwen).
    # Battery path: battery_wave pins each to the `_it` wave (own
    # namespace + bare anchor, objects applied to the instruct model
    # directly — no sibling staging; configs/sdfbase_transfer_<slug>_it.yaml
    # on the bank BATTERY_BANK_SUFFIX["_it"] = out/transfer_<slug>_it/cells);
    # any other --wave refuses by name, because the slug's other configs
    # name the BASE-fit banks — and `_it` refuses any other key
    # (check_battery_wave).
    # derived_eligible False: the derived app (sdfbase_derived.py) imports
    # this table and sweeps every non-probe_only key by default; these
    # lanes carry ONLY mix-full (no mix-100k / ctl-100k parents), so no
    # derived operation is valid for them — excluded from its default AND
    # its explicit --model.
    "gemma4e4b_it_doc": {
        "gpu": "H200",
        "cfg": "/repo/configs/sdfbase_gemma4e4b_it_doc_fits.yaml",
        "repo": "google/gemma-4-E4B-it",
        "rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "gated": True,
        "probe_render": "chat",
        "explicit_only": True,
        "battery_wave": "_it",
        "derived_eligible": False,
    },
    "qwen35_4b_it_doc": {
        "gpu": "H200",
        "cfg": "/repo/configs/sdfbase_qwen35_4b_it_doc_fits.yaml",
        "repo": "Qwen/Qwen3.5-4B",
        "rev": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "gated": True,
        "probe_render": "chat",
        "explicit_only": True,
        "battery_wave": "_it",
        "derived_eligible": False,
    },
    # INSTRUCT-DIRECT BATCH-64 siblings (2026-09-11): the two
    # `_it_doc` checkpoints re-fit at effective batch 64 — constitution AND
    # control twins — under their OWN keys, so the four published `_it_doc`
    # cells (cited by the artifact) are never superseded, overwritten or
    # relocated: out/<key>_eb64/ is a fresh namespace and the supersede
    # gate has nothing to fire on. "pool_key": the pools and tokcache
    # sidecars are per CHECKPOINT (a sidecar pins model_id + template_hash),
    # so these keys read /vol/pool/<parent _it_doc>/ (_pool_path); their
    # own additions there (mix-1m, mix-10m, ctl2-full + sidecars) validate
    # for the parent too. Same gating / render / explicit_only / battery
    # namespace as the parents; derived_eligible False keeps the derived
    # app's DEFAULT sweep off them (their derived objects are built by
    # explicit, named runs once the twins are fit). Image variant = the
    # architecture's (std / qwen: QWEN_IMAGE_KEYS).
    "gemma4e4b_it_doc_eb64": {
        "gpu": "H200",
        "cfg": "/repo/configs/sdfbase_gemma4e4b_it_doc_eb64_fits.yaml",
        "repo": "google/gemma-4-E4B-it",
        "rev": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "gated": True,
        "probe_render": "chat",
        "explicit_only": True,
        "battery_wave": "_it",
        "derived_eligible": False,
        "pool_key": EB64_PARENT["gemma4e4b_it_doc_eb64"],
    },
    "qwen35_4b_it_doc_eb64": {
        "gpu": "H200",
        "cfg": "/repo/configs/sdfbase_qwen35_4b_it_doc_eb64_fits.yaml",
        "repo": "Qwen/Qwen3.5-4B",
        "rev": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "gated": True,
        "probe_render": "chat",
        "explicit_only": True,
        "battery_wave": "_it",
        "derived_eligible": False,
        "pool_key": EB64_PARENT["qwen35_4b_it_doc_eb64"],
    },
    # Llama-3.1-8B (2026-09-09): the UNGATED NousResearch mirrors, base
    # and instruct, both dense (image_std — the `_variant` default;
    # neither kernel image). This app runs ONLY the paid pieces of that
    # lane — the onboarding smoke, the memprobe, the gradient-norm
    # recon (bench) and the HPO search on the BASE lane, plus the Modal
    # half of the cross-lane identity gate; every production fit runs on
    # Slurm (scripts/sbatch/sdfbase_fit_cell.sbatch). Under the lane's
    # own app name (--app-name sdfbase-fits-llama8b), never the frozen
    # wave-A app. explicit_only: named with --model in every mode, never
    # swept into a peer's bare invocation. probe_render: raw for the
    # base (the base convention), chat for the instruct (it IS the
    # served model). battery_wave: the instruct-direct key belongs to
    # the `_it` namespace like its gemma/qwen siblings.
    # derived_eligible False on BOTH: the derived pairs for this lane
    # run on Slurm (scripts/sbatch/sdfbase_derived_build.sbatch), never
    # through the Modal derived app, which imports this table and
    # sweeps every eligible key by default.
    "llama8b_base": {
        "gpu": "H200",
        "cfg": "/repo/configs/sdfbase_llama8b_base_fits.yaml",
        "repo": "NousResearch/Meta-Llama-3.1-8B",
        "rev": "1f47e50cdbe801ad8a5174156ec3a0655108fb9f",
        "gated": False,
        "explicit_only": True,
        "derived_eligible": False,
    },
    "llama8b_it_doc": {
        "gpu": "H200",
        "cfg": "/repo/configs/sdfbase_llama8b_it_doc_fits.yaml",
        "repo": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "rev": "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
        "gated": False,
        "probe_render": "chat",
        "explicit_only": True,
        "battery_wave": "_it",
        "derived_eligible": False,
    },
}

#: Qwen-BASE batch-64 SEED REPLICATES (2026-09-15; the qwen-base-seed-
#: replicates lane): the qwen35_4b_base checkpoint — SAME repo + pin + gating
#: as the frozen `qwen35_4b_base` entry, on H200 — re-fit from scratch at
#: effective batch 64 under seeds 1, 2, 3 (make_sdfbase_configs.SEEDREP_*),
#: one key per seed with its OWN generated config, OWN /vol/pool/<key>/ (the
#: held-out carve differs per seed, so NO pool_key: _pool_path resolves to
#: the key's own dir) and OWN out/<key>/ namespace, so the four published
#: batch-8 seed-0 cells are never superseded. This app runs only the paid
#: pieces of that lane (memprobe, recon, the seed-1 LR search whose
#: `_hpo_impl` threads the config's train_kwargs.default.seed through
#: `tune_arm.py trial --seed`, the Modal half of the identity gate); every
#: production fit runs on Slurm. explicit_only: named with --model in every
#: mode, never swept into a peer's bare invocation. derived_eligible False:
#: the derived pairs run on Slurm (sdfbase_derived_build.sbatch), never
#: through the Modal derived app's default sweep. probe_render: the base
#: convention (raw), as the parent. Image variant = the architecture's
#: (qwen: QWEN_IMAGE_KEYS below).
SEEDREP_KEYS = ("qwen35_4b_base_eb64_s1", "qwen35_4b_base_eb64_s2",
                "qwen35_4b_base_eb64_s3")
for _k in SEEDREP_KEYS:
    MODELS[_k] = {
        "gpu": "H200",
        "cfg": f"/repo/configs/sdfbase_{_k}_fits.yaml",
        "repo": "Qwen/Qwen3.5-4B-Base",
        "rev": "1001bb4d826a52d1f399e183466143f4da7b741b",
        "gated": True,
        "explicit_only": True,
        "derived_eligible": False,
    }
del _k

#: Model keys on the qwen hybrid-attention stack (Qwen3.5 line): every GPU
#: function for them must bind image_qwen (the fused kernels — triton
#: 3.7.1 + fla 0.5.2 + causal-conv1d), never image_std.
QWEN_IMAGE_KEYS = ("qwen35_4b_base", "qwen35_4b_it", "qwen35_4b_it_doc",
                   "qwen35_4b_it_doc_eb64",
                   # the qwen-base batch-64 seed replicates (2026-09-15):
                   # the same hybrid checkpoint, so the same D66 kernels
                   "qwen35_4b_base_eb64_s1", "qwen35_4b_base_eb64_s2",
                   "qwen35_4b_base_eb64_s3")

#: Model keys on the NemotronH Mamba2-hybrid stack: every GPU function for
#: them binds image_nemotron — image_qwen's recipe PLUS the mamba-ssm fused
#: kernels. NemotronH's Mamba2 mixers resolve their kernels per op
#: (transformers 5.14.1: kernels-hub -> pip mamba_ssm / causal_conv1d ->
#: the inline pure-torch SSD scan) and need ALL FIVE
#: (selective_state_update, mamba_chunk_scan_combined,
#: mamba_split_conv1d_scan_combined, causal_conv1d_fn/update) for the
#: fast path; image_qwen carries only the causal-conv1d pair, so on it the
#: model trained the torch_forward SSM — correct, and 2.09 / 1.88 s per
#: bs-2 step (psr_kl_all / lora_r1_all, max_len 2048, cluster H200,
#: 2026-09-02). On image_nemotron the SAME smoke rows run 0.23 / 0.13
#: s/step at 11.4 GiB peak instead of 28.1 (Modal H200, 2026-09-02; the
#: kernel-less image reproduced the cluster numbers on that machine to the
#: millisecond, so the 9x / 14x is the kernels). Speed-giving kernels ship
#: by default, pinned, dependencies notwithstanding — hence this image and
#: this key table. The onboarding smoke's "Mamba2 kernel path" row records
#: which path a container actually resolved.
NEMOTRON_IMAGE_KEYS = ("nemotron3nano4b_doc",)

#: sdfbase model key -> the INSTRUCT transfer slug the battery serves
#: (configs/sdfbase_transfer_<slug>.yaml — the explicit-cell
#: registry, named by the instruct sibling). The two `_doc` keys map to the
#: two_si slug of the SAME checkpoint (GLM and Nemotron are their own
#: instruct models); neither transfer config exists yet (the transfer
#: batteries are scoped OUT), and the battery's committed-config check
#: refuses loudly until one is written after the fits land.
#: The instruct-direct `_it_doc` keys map to the slug of the instruct
#: model they ARE; their battery runs only under the `_it` wave
#: (MODELS[key]["battery_wave"], enforced by check_battery_wave in the
#: battery entrypoint: configs/sdfbase_transfer_<slug>_it.yaml) — every
#: other wave's config for that slug names a base-fit bank.
TRANSFER_SLUG = {"gemma4e4b_base": "gemma4e4b",
                 "qwen35_4b_base": "qwen35_4b",
                 "glm46vflash_doc": "glm46vflash",
                 "nemotron3nano4b_doc": "nemotron3nano4b",
                 "gemma4e4b_it_doc": "gemma4e4b",
                 "qwen35_4b_it_doc": "qwen35_4b",
                 # the batch-64 siblings (2026-09-11) ARE the same instruct
                 # checkpoints: same slug, same `_it` namespace
                 "gemma4e4b_it_doc_eb64": "gemma4e4b",
                 "qwen35_4b_it_doc_eb64": "qwen35_4b",
                 # Llama-3.1-8B (2026-09-09): both keys map to the two_si
                 # slug of the instruct checkpoint the wave serves
                 # (configs/two_si_llama*.yaml exist; no sdfbase transfer
                 # config does yet, so the battery's committed-config check
                 # refuses by path until one is written — this lane's
                 # evaluation runs on the full22 wave tree, not here).
                 "llama8b_base": "llama",
                 "llama8b_it_doc": "llama",
                 # the qwen-base batch-64 seed replicates (2026-09-15) ARE
                 # the qwen35_4b_base checkpoint, served on the same
                 # instruct sibling: the same slug
                 "qwen35_4b_base_eb64_s1": "qwen35_4b",
                 "qwen35_4b_base_eb64_s2": "qwen35_4b",
                 "qwen35_4b_base_eb64_s3": "qwen35_4b"}

#: The ce mode's pools — the ANALYSIS names (analysis_r2/r2_paths.py
#: CE_POOLS) -> the staged pool file; a byte-for-byte twin of the derived
#: app's table (sdfbase_derived.CE_POOLS; the derived app imports THIS
#: module, so the table cannot live there and be shared — a test pins the
#: two equal). ``control-heldout`` is the control corpus remainder unused
#: by ctl-100k (scripts/sdfbase/build_ctl_heldout.py).
CE_POOLS = {
    "constv2-heldout": "/vol/pool/{key}/pairs_constv2-heldout.jsonl",
    "control-heldout": "/vol/pool/{key}/pairs_constv2ctl-heldout.jsonl",
    "constv2ctl-100k": "/vol/pool/{key}/pairs_constv2ctl-100k.jsonl",
}

#: ce_score outputs land kind-first, in the analysis shape (the derived
#: app's CE_ROOT): ce_score/<model>/<cell>__<pool>__scale<s>.json.
CE_ROOT = "/vol/out/ce_score"

#: The fits bank root a consumer re-verifies a DERIVED object's parents
#: against; harmless for a staged cell (no parents).
PARENTS_ROOT = "/vol/out"

_IGNORE = [".git", "**/__pycache__", ".venv", "out/**", "wandb/**",
           ".pytest_cache", "worktrees"]

#: The standard fit image — the e4b_wave.py recipe (cu128 torch tree from
#: the lockfile-exported requirements). Copied VERBATIM from
#: scripts/modal/twosi_fits.py (which this app never edits or redeploys).
image_std = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unixodbc")
    .pip_install_from_requirements(
        str(WORKTREE / "scripts/modal/requirements.txt"),
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_dir(str(WORKTREE), remote_path="/repo", ignore=_IGNORE)
)

#: The Qwen hybrid-attention image: nvidia/cuda DEVEL base (nvcc 12.8 for
#: the causal-conv1d source build — no wheel exists at torch 2.11), the
#: same requirements tree, then the hybrid stack pins: triton 3.7.1
#: OVERRIDES the lockfile's 3.6.0 (fla's gated-chunk backward miscompiles
#: on [3.4.0, 3.7.1)), flash-linear-attention/fla-core 0.5.2,
#: causal-conv1d 1.6.2.post1 built for sm90 (H100/H200). Copied VERBATIM
#: from twosi_fits.py.
image_qwen = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("unixodbc", "git", "build-essential")
    .pip_install_from_requirements(
        str(WORKTREE / "scripts/modal/requirements.txt"),
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("triton==3.7.1", "fla-core==0.5.2",
                 "flash-linear-attention==0.5.2")
    .run_commands(
        "CUDA_HOME=/usr/local/cuda CC=gcc CXX=g++ "
        "TORCH_CUDA_ARCH_LIST='9.0;10.0' MAX_JOBS=8 "
        "CAUSAL_CONV1D_FORCE_BUILD=TRUE "
        "pip install --no-build-isolation causal-conv1d==1.6.2.post1",
        # The step above re-resolves torch's own triton==3.6.0 pin and
        # silently DOWNGRADES the 3.7.1 we just installed — fla then
        # refuses every gated-chunk backward at runtime (#640; bit the
        # first memprobe, all Hopper). Re-pin LAST with --no-deps, and
        # make the build itself fail if the stack ever regresses.
        "pip install --no-deps triton==3.7.1",
        "python -c \"import triton; v = triton.__version__; "
        "assert v == '3.7.1', v; import causal_conv1d, fla\"",
    )
    .add_local_dir(str(WORKTREE), remote_path="/repo", ignore=_IGNORE)
)

#: The NemotronH (Mamba2-hybrid) image: image_qwen's recipe step for step
#: (same devel base, requirements tree, pins and causal-conv1d source
#: build — every layer through that build is shared with image_qwen's
#: cache; image_qwen itself is untouched, live qwen cells build from it),
#: PLUS the mamba-ssm fused kernels, so transformers' per-op resolution
#: finds all five and NemotronH runs cuda_kernels_forward instead of the
#: pure-torch SSD scan. Copied VERBATIM from twosi_fits.py.
#:
#: mamba-ssm==2.3.2.post1 — the newest PyPI release >= 7 days old on
#: 2026-09-02 (released 2026-05-09; setup.py pins no torch upper bound).
#: Its GitHub wheel matrix stops at torch 2.10 (+ NGC tags), so at torch
#: 2.11 the package's own bdist hook would 404 and fall through to a source
#: build anyway: MAMBA_FORCE_BUILD=TRUE skips that dead probe and makes the
#: build the explicit, deterministic path. nvcc compiles the selective-scan
#: extension for sm_75/80/87/90/100/120 (setup.py's own list — it does not
#: read TORCH_CUDA_ARCH_LIST); ninja==1.13.0 (2025-08-11) goes in first so
#: the .cu files compile in parallel under MAX_JOBS=8 — without ninja,
#: torch's BuildExtension compiles them serially.
#:
#: --no-deps ON PURPOSE: 2.3.2.post1 added hard deps tilelang==0.1.8,
#: apache-tvm-ffi and quack-kernels (-> nvidia-cutlass-dsl, ~200 MB of an
#: unrelated CUDA DSL toolchain) that serve ONLY its new Mamba3 kernels,
#: which mamba_ssm imports behind try/except ImportError. The Mamba2 fast
#: path NemotronH calls (ops.triton.selective_state_update,
#: ops.triton.ssd_combined + the compiled selective_scan_cuda) needs torch,
#: triton, einops and packaging — all already in the tree. To take the
#: Mamba3 chain anyway, drop --no-deps (the triton re-pin below then
#: matters again, as it does for causal-conv1d).
#:
#: The bake check imports exactly what transformers resolves, so a
#: regressed stack fails the BUILD, not the first fit.
image_nemotron = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("unixodbc", "git", "build-essential")
    .pip_install_from_requirements(
        str(WORKTREE / "scripts/modal/requirements.txt"),
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("triton==3.7.1", "fla-core==0.5.2",
                 "flash-linear-attention==0.5.2")
    .run_commands(
        "CUDA_HOME=/usr/local/cuda CC=gcc CXX=g++ "
        "TORCH_CUDA_ARCH_LIST='9.0;10.0' MAX_JOBS=8 "
        "CAUSAL_CONV1D_FORCE_BUILD=TRUE "
        "pip install --no-build-isolation causal-conv1d==1.6.2.post1",
        "pip install --no-deps triton==3.7.1",
        "python -c \"import triton; v = triton.__version__; "
        "assert v == '3.7.1', v; import causal_conv1d, fla\"",
    )
    .run_commands(
        "pip install ninja==1.13.0",
        "CUDA_HOME=/usr/local/cuda CC=gcc CXX=g++ MAX_JOBS=8 "
        "MAMBA_FORCE_BUILD=TRUE "
        "pip install --no-build-isolation --no-deps mamba-ssm==2.3.2.post1",
        # Re-pin triton LAST and assert, exactly as the causal-conv1d step
        # does: --no-deps leaves it alone, and the assert is the proof.
        "pip install --no-deps triton==3.7.1",
        "python -c \"import triton; v = triton.__version__; "
        "assert v == '3.7.1', v; import causal_conv1d, fla, mamba_ssm; "
        "from mamba_ssm.ops.triton.selective_state_update import "
        "selective_state_update; "
        "from mamba_ssm.ops.triton.ssd_combined import "
        "mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined; "
        "from causal_conv1d import causal_conv1d_fn, causal_conv1d_update\"",
    )
    .add_local_dir(str(WORKTREE), remote_path="/repo", ignore=_IGNORE)
)

#: Autoscale CEILING (max-parallelism directive): high enough that Modal's
#: own H200 availability is the binding constraint, never this number.
MAX_CONTAINERS = 288

#: Fit-function timeout. Production leaves it at Modal's 24 h maximum
#: (continuation past it is the successor-creating watcher's job). The
#: continuation drill deploys the SAME commit under a
#: throwaway app name with a short timeout so the real timeout path fires
#: in minutes:
#:   SDFBASE_FIT_TIMEOUT=600 modal deploy --name <drill-app> \
#:       scripts/modal/sdfbase_fits.py
FIT_TIMEOUT_S = int(os.environ.get("SDFBASE_FIT_TIMEOUT", 86400))

#: The SHARED campaign volume (HF snapshots + banks) — same name as the
#: twosi app's; this campaign only ADDS pool/<base-key>/ + out/<base-key>/.
vol = modal.Volume.from_name("twosi-fits", create_if_missing=True)

#: Volume-backed home for the REPLAYED frozen attack corpora (renellm today).
#: The library default is a grading host ~/.local/state path that no container has,
#: which is why renellm skipped silently — see battery_commands. Containment
#: note: these hold transformed harmful prompts, and the volume already holds
#: every response transcript under /vol/private, so this adds no new exposure.
HELDOUT_CORPUS_DIR = "/vol/heldout_corpora"

#: Battery attack ids that are PER-CELL rather than per-(cell x shard):
#: exactly one container each, ``--nshards`` ignored.
#:
#: - ``controls`` — the capability leg (cell_controls_run.py), which shards
#:   nothing.
#: - ``xstest`` — the benign over-refusal pass. Its writer appends to
#:   ONE ``xstest_responses.jsonl`` per cell dir with no shard component
#:   (the BoN layout), and ``two_si_run.cmd_generate`` does not slice goals
#:   on the ``--xstest`` path at all, so a second shard would be a second
#:   container generating the SAME 250 prompts into the same volume file —
#:   the cross-container last-writer-wins hazard bon has, with the GPU
#:   cost doubled for nothing.
UNSHARDED_BATTERY_ATTACKS = ("controls", "xstest")

# The HF token plus the operator's own W&B secret (``WANDB_SECRET`` below:
# WANDB_API_KEY + WANDB_ENTITY + WANDB_PROJECT). The workspace's shared
# ``wandb`` secret carries another member's API key — every fit run this
# app logged through it landed in a foreign W&B entity — and is never
# referenced here. Belt and braces: every ``wandb.init`` the fit drivers
# reach passes ``entity=$WANDB_ENTITY`` explicitly, and
# :func:`require_wandb_entity` refuses at container start unless the
# mounted key's default entity IS that entity. Takes effect at the next
# deploy.
# The W&B secret's name is the site setting SISTEER_MODAL_WANDB_SECRET, read
# where the apps are defined (a deploy or a run): the environment, else the
# site config (src/sisteer/site.py), so a plain `modal deploy` from a shell
# that never exported it still mounts the secret, and the code names no
# account's secret. Set in neither, the functions mount the HF token alone and
# require_wandb_entity refuses at container start, naming the setting (with
# W&B switched off nothing needs it).
WANDB_SECRET_ENV = "SISTEER_MODAL_WANDB_SECRET"


def _site_setting(name: str) -> str:
    """``name`` from the environment, else from the site config, else ''. A site
    config asked for by name ($SISTEER_CONFIG) that is missing refuses. In the
    container (no site config there) it is ''; the deployed functions keep the
    secrets they were defined with."""
    import sys

    value = os.environ.get(name, "").strip()
    if value:
        return value
    src = str(WORKTREE / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer.site import load_site

    return (load_site().config.get(name) or "").strip()


def _stage_path(rel: str) -> Path:
    """``rel`` under the grading host's staging directory: ``$STAGE`` (sisteer.site), the
    setting SISTEER_STAGE_DIR from the environment, else the site config. Unset in
    both, it refuses by name (the local entrypoints only; a container never asks)."""
    import sys

    src = str(WORKTREE / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer import site as _site

    return _site.expand_path(f"$STAGE/{rel}", key="the local staging directory")


WANDB_SECRET = _site_setting(WANDB_SECRET_ENV)
_secrets = [modal.Secret.from_name("hf-token")] + (
    [modal.Secret.from_name(WANDB_SECRET)] if WANDB_SECRET else [])
# Operator kill-switch (the twosi_fits convention): flip to False to run
# with W&B off entirely — metrics.jsonl + the fitmon per-cell mirror still
# carry the curves.
_HAVE_WANDB = True

_COMMON = dict(volumes={"/vol": vol}, secrets=_secrets)


def require_wandb_entity() -> str:
    """Refuse loudly — before any GPU work — unless ``WANDB_ENTITY`` is set
    AND the mounted W&B key's default entity IS that entity (a stray key
    silently routed a whole wave's runs to another member's account; a
    clear error at container start beats that). Returns the entity.
    Skipped entirely when W&B is switched off.

    The check itself is the SHARED validator ``sisteer.wandb_entity``:
    the direct CLI paths (``psr_residual_distill.py --wandb``) run the
    identical rule before their own ``wandb.init``, so the wrapped and
    the direct route can never disagree; API errors are retried a
    bounded number of times there and then refuse too."""
    import os

    if not _HAVE_WANDB or os.environ.get("WANDB_DISABLED", "").lower() in (
            "1", "true"):
        return ""
    _repo_imports()
    from sisteer.wandb_entity import require_wandb_entity as _shared

    which = f" (here {WANDB_SECRET!r})" if WANDB_SECRET else ""
    return _shared(hint=f"mount the W&B secret the site setting {WANDB_SECRET_ENV} "
                        f"names{which}: WANDB_API_KEY + WANDB_ENTITY + "
                        "WANDB_PROJECT; set it in the site config where the app "
                        "is deployed, then redeploy")


def _env(extra=None):
    import os

    env = dict(os.environ)
    env.update({
        "HF_HOME": "/vol/hf",
        "PYTHONPATH": "/repo/src",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if not _HAVE_WANDB:
        env["WANDB_DISABLED"] = "true"
    if extra:
        env.update(extra)
    return env


def _ensure_model(key: str) -> None:
    import os
    from pathlib import Path

    spec = MODELS[key]
    os.environ["HF_HOME"] = "/vol/hf"
    snap = Path(f"/vol/hf/hub/models--{spec['repo'].replace('/', '--')}"
                f"/snapshots/{spec['rev']}")
    if not (snap / "config.json").exists():
        from huggingface_hub import snapshot_download

        tok = (os.environ.get("HF_TOKEN")
               or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        snapshot_download(spec["repo"], revision=spec["rev"],
                          token=tok if spec["gated"] else None)
        vol.commit()
    os.environ["HF_HUB_OFFLINE"] = "1"


def _pool_key(key: str) -> str:
    """The ``/vol/pool/<dir>/`` a key's pools and tokcache sidecars live
    under: its own key, or its MODELS "pool_key" — a same-checkpoint
    sibling whose staged pools it shares (the `_eb64` lanes read the frozen
    `_it_doc` lanes' dirs; the generator's pool_key_for is the same rule
    and already points the config's heldout_pool there). A pool_key never
    chains and must name a registered key of the SAME repo + revision —
    a sidecar pins model_id + template_hash, so sharing across checkpoints
    would be refused at warm time anyway; refuse it here, before a spawn."""
    spec = MODELS[key]
    pk = spec.get("pool_key")
    if pk is None:
        return key
    parent = MODELS.get(pk)
    if parent is None or parent.get("pool_key") is not None:
        raise SystemExit(f"[pool_key] {key} names pool_key {pk!r}: not a "
                         "registered key, or one that chains")
    if (parent["repo"], parent["rev"]) != (spec["repo"], spec["rev"]):
        raise SystemExit(f"[pool_key] {key} -> {pk}: different checkpoint "
                         f"({spec['repo']}@{spec['rev'][:8]} vs "
                         f"{parent['repo']}@{parent['rev'][:8]}); a tokcache "
                         "sidecar pins model_id, so the pools cannot be shared")
    return pk


def _pool_path(key: str, cell: str) -> str:
    return f"/vol/pool/{_pool_key(key)}/pairs_{cell}.jsonl"


def _cfg_yaml(key: str) -> dict:
    import yaml

    return yaml.safe_load(Path(MODELS[key]["cfg"]).read_text())


#: The store-identity sidecar seeded/published alongside cell dirs.
_STORE_IDENT = "_store_identity.json"


def _start_dmon(metrics_path: Path, done) -> list[int]:
    """GPU-duty spot-check (gpu-util skill), shared by the fit and hpo
    impls: a daemon thread waits for the metrics file to show a live
    optimizer loop (>= 3 lines), settles 30s, then samples one
    ``nvidia-smi dmon`` burst. Returns the rows list the thread appends
    to; ``done.set()`` stops the wait early."""
    import subprocess
    import threading

    rows: list[int] = []

    def sample():
        for _ in range(360):
            if done.is_set():
                return
            done.wait(5)
            try:
                if metrics_path.exists() and \
                        sum(1 for _ in metrics_path.open()) >= 3:
                    break
            except OSError:
                pass
        else:
            return
        if done.is_set():
            return
        done.wait(30)
        p = subprocess.run(
            ["nvidia-smi", "dmon", "-i", "0", "-s", "u", "-c", "25"],
            capture_output=True, text=True, timeout=90)
        for line in p.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    rows.append(int(parts[1]))
                except ValueError:
                    pass

    threading.Thread(target=sample, daemon=True).start()
    return rows


def _mean_sm_pct(dmon_rows: list[int]) -> float | None:
    """Steady-state mean SM% from a dmon burst (first row dropped when the
    burst is long enough to have a warmup sample)."""
    if not dmon_rows:
        return None
    steady = dmon_rows[1:] if len(dmon_rows) > 5 else dmon_rows
    return round(sum(steady) / len(steady), 1)


def _median_sec_per_step(metrics_path: Path) -> float | None:
    """Median train/sec_per_step past step 5 from a metrics.jsonl, or None
    when the file/records are absent."""
    import statistics

    if not metrics_path.exists():
        return None
    spst = []
    for line in metrics_path.open():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (rec.get("step") or 0) >= 5 and "train/sec_per_step" in rec:
            spst.append(float(rec["train/sec_per_step"]))
    return round(statistics.median(spst), 3) if spst else None


def _onboard_impl(key: str) -> dict:
    """The cheap early probe: does the base ship a chat template
    (recorded, bypassed either way), does '' encode to [BOS]/[] as
    expected, do the layer walkers resolve — via the standing onboarding
    smoke, resolved from the sdfbase config's model/model_revision pins
    (the base entries live in derive_model_config.MODEL_SPECS)."""
    import subprocess
    import sys

    _ensure_model(key)
    t0 = time.time()
    p = subprocess.run(
        [sys.executable, "/repo/scripts/onboard_model_smoke.py",
         "--model-config", MODELS[key]["cfg"]],
        cwd="/repo", env=_env(), capture_output=True, text=True)
    out = {"model": key, "rc": int(p.returncode),
           "wall_s": round(time.time() - t0, 1),
           "tail": p.stdout.splitlines()[-25:] if p.stdout else []}
    if p.returncode != 0:
        out["status"] = "failed"
        out["stderr_tail"] = str(p.stderr[-2000:])
    else:
        out["status"] = "ok"
    return out


def memprobe_variant_suffix(loss_impl: str = "", grad_ckpt: str = "") -> str:
    """The artifact-name suffix a memprobe VARIANT writes under. EMPTY only
    when NEITHER variant option was supplied — the historical probe
    (``memprobe_<arm>_all.json``, the folded configs' source, never
    overwritten). ANY explicit value — ``full`` / ``config`` included —
    lands under its own name (``__full``, ``__chunked_ce``, ``__gcon``,
    ``__gcoff``, ``__gcconfig``): an operator re-probing the baseline for
    comparison must never clobber the artifact the committed configs were
    folded from."""
    suffix = ""
    if loss_impl:
        suffix += f"__{loss_impl}"
    if grad_ckpt:
        suffix += f"__gc{grad_ckpt}"
    return suffix


def _memprobe_impl(item: tuple) -> dict:
    """ONE (model, arm[, loss_impl[, grad_ckpt]]) doubling ladder at
    WORST-CASE rows: tune_arm probe-mem takes the LONGEST real docs of the
    staged 1M pool (the take_longest convention), positions all, max_len
    8192; keep the largest bs whose peak RESERVED stays under 0.85 x VRAM.
    Writes /vol/out/<key>/memprobe_<arm>_all<suffix>.json — the bare name
    is folded into batch_size_by_arm by make_sdfbase_configs.py
    --fold-memprobe; ``loss_impl`` ("chunked_ce") and ``grad_ckpt``
    ("on"/"off") re-probe the trainer variant under a suffixed name
    (memprobe_variant_suffix), never clobbering the folded one."""
    import os
    import re
    import subprocess
    import sys

    import torch

    key, arm = item[0], item[1]
    loss_impl = str(item[2]) if len(item) > 2 and item[2] else ""
    grad_ckpt = str(item[3]) if len(item) > 3 and item[3] else ""
    suffix = memprobe_variant_suffix(loss_impl, grad_ckpt)
    _ensure_model(key)
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    guard_gib = 0.85 * vram_gib
    pairs = _pool_path(key, MEMPROBE_SI)
    if not os.path.exists(pairs):
        return {"model": key, "arm": arm, "status": "no-pool",
                "error": f"{pairs} not staged — the memprobe ladders the "
                         "LONGEST REAL docs (never synthetic rows: the "
                         "doc shape is the memory shape)"}
    out = {"model": key, "arm": arm, "position": "all",
           "gpu": torch.cuda.get_device_name(0),
           "vram_gib": round(vram_gib, 1), "guard_gib": round(guard_gib, 1),
           "max_len": 8192, "pool": pairs,
           "loss_impl": loss_impl or "config",
           "gradient_checkpointing": grad_ckpt or "config",
           "artifact": f"/vol/out/{key}/memprobe_{arm}_all{suffix}.json"}
    variant_flags = []
    if loss_impl:
        variant_flags += ["--loss-impl", loss_impl]
    if grad_ckpt:
        variant_flags += ["--gradient-checkpointing", grad_ckpt]
    b, best, ladder = 1, None, {}
    while b <= 64:
        p = subprocess.run(
            [sys.executable, "/repo/scripts/tune_arm.py", "probe-mem",
             "--config", MODELS[key]["cfg"], "--pairs", pairs,
             "--si", MEMPROBE_SI, "--arm", arm,
             "--positions", "all", "--batch-size", str(b),
             "--steps", "4", *variant_flags],
            cwd="/repo", env=_env(), capture_output=True, text=True)
        m = re.search(r"peak_alloc=([0-9.]+)GiB peak_reserved="
                      r"([0-9.]+)GiB", p.stdout)
        alloc, reserved = ((float(m.group(1)), float(m.group(2)))
                           if m else (None, None))
        oom = ("CUDA out of memory" in p.stderr
               or "CUDA out of memory" in p.stdout)
        ladder[b] = {"rc": p.returncode, "peak_alloc_gib": alloc,
                     "peak_reserved_gib": reserved, "oom": oom}
        line = (p.stdout.splitlines()[-1] if p.stdout.splitlines() else "")
        print(f"[{key}/{arm}/all] bs={b} rc={p.returncode} alloc={alloc} "
              f"reserved={reserved} oom={oom} | {line}")
        if p.returncode != 0 and not oom:
            ladder[b]["stderr_tail"] = str(p.stderr[-1500:])
            print(f"[{key}/{arm}/all] bs={b} STDERR tail:\n"
                  f"{p.stderr[-1500:]}")
        if (p.returncode == 0 and reserved is not None
                and reserved <= guard_gib):
            best = b
            b *= 2
        else:
            break
    out.update({"best_bs": best, "ladder": ladder})
    Path(f"/vol/out/{key}").mkdir(parents=True, exist_ok=True)
    Path(f"/vol/out/{key}/memprobe_{arm}_all{suffix}.json").write_text(
        json.dumps(out, indent=1))
    vol.commit()
    return out


#: The throughput bench's staged-pool label. The control pool
#: constv2ctl2-full is what the bench measures on; tune_arm resolves --si
#: through the pseudo-instruction registry and reads max_len through the
#: config, so the bench passes the REGISTERED corpus-cell id whose config
#: row matches the pool's shape (max_len 8192) and records the pool file
#: + sha it actually trained on. A pool cell that IS registered is passed
#: through unchanged.
BENCH_SI_LABEL_FALLBACK = "constv2-mix-full"


#: The bench record's explicit outcomes (tune_arm.py bench).
BENCH_OUTCOMES = ("ok", "diverged", "failed")

#: --mode bench's step count when --steps is omitted (0) on a GENERIC
#: throughput bench (tune_arm.py bench's own parser default, kept for
#: every lane that measures throughput). A bench carrying any of the
#: reconnaissance's per-run overrides is a reconnaissance run and takes the
#: reconnaissance budget instead — bench_steps.
BENCH_DEFAULT_STEPS = 200


def bench_cmd(*, cfg: str, pairs: str, si_label: str, arm: str, bs: int,
              steps: int, out_local: str, loss_impl: str = "",
              grad_ckpt: str = "", accum: int = 0, check_grads: bool = False,
              lr: float | None = None, max_grad_norm: float | None = None,
              warmup_frac: float | None = None) -> list[str]:
    """The tune_arm.py bench argv (after the interpreter) for ONE run —
    the throughput knobs plus the reconnaissance's per-run hyperparameter
    overrides: each of --lr /
    --max-grad-norm / --warmup-frac is passed ONLY when given, so a bench
    without overrides is the byte-identical command it always was and
    tune_arm reads the config block."""
    cmd = ["/repo/scripts/tune_arm.py", "bench",
           "--config", cfg, "--pairs", pairs,
           "--si", si_label, "--arm", arm, "--batch-size", str(int(bs)),
           "--steps", str(int(steps)), "--out", str(out_local)]
    if loss_impl:
        cmd += ["--loss-impl", str(loss_impl)]
    if grad_ckpt:
        cmd += ["--gradient-checkpointing", str(grad_ckpt)]
    if accum:
        cmd += ["--accum", str(int(accum))]
    if check_grads:
        cmd.append("--check-grads")
    for flag, val in (("--lr", lr), ("--max-grad-norm", max_grad_norm),
                      ("--warmup-frac", warmup_frac)):
        if val is not None:
            cmd += [flag, repr(float(val))]
    return cmd


def bench_outcome(rc: int, record: dict | None) -> str:
    """ok | diverged | failed for a finished bench subprocess: the record's
    OWN explicit status when it names a non-ok outcome (tune_arm writes
    status diverged / failed with the partial series, then exits non-zero
    — an outcome, not a crash), else ok on a clean exit, else
    failed (no record, or a record written before the outcome field
    existed, from a process that then died)."""
    st = (record or {}).get("status")
    if st in ("diverged", "failed"):
        return str(st)
    return "ok" if int(rc) == 0 else "failed"


def bench_overrides(lr: str = "", max_grad_norm: str = "",
                    warmup_frac: str = "") -> dict:
    """The --mode bench per-run overrides as floats (unset = "" = absent),
    validated before any spawn: finite, positive lr/clip, warmup in
    [0, 1). The clip must be FINITE — the trainer logs the pre-clip norm
    only under a finite clip, so `inf` would record no norms (1e6 is the
    recon's effectively-unclipped value)."""
    out: dict = {}
    for name, raw in (("lr", lr), ("max_grad_norm", max_grad_norm),
                      ("warmup_frac", warmup_frac)):
        if raw is None or str(raw).strip() == "":
            continue
        try:
            v = float(str(raw))
        except ValueError:
            raise SystemExit(f"--mode bench --{name.replace('_', '-')} "
                             f"{raw!r} is not a number")
        if not math.isfinite(v):
            raise SystemExit(f"--mode bench --{name.replace('_', '-')} must "
                             f"be finite (got {raw!r}); the trainer logs the "
                             "pre-clip grad norm only under a finite clip — "
                             "use 1e6 for an effectively unclipped run")
        if name in ("lr", "max_grad_norm") and v <= 0:
            raise SystemExit(f"--mode bench --{name.replace('_', '-')} must "
                             f"be positive (got {raw!r})")
        if name == "warmup_frac" and not (0.0 <= v < 1.0):
            raise SystemExit(f"--mode bench --warmup-frac must lie in [0, 1) "
                             f"(got {raw!r})")
        out[name] = v
    return out


def bench_override_tag(overrides: dict) -> str:
    """The run-tag suffix naming the overrides (so recon runs at different
    settings never overwrite one record); empty without overrides."""
    parts = []
    for name, short in (("lr", "lr"), ("max_grad_norm", "clip"),
                        ("warmup_frac", "wf")):
        if name in overrides:
            parts.append(f"{short}{overrides[name]:g}")
    return ("__" + "__".join(parts)) if parts else ""


def recon_steps() -> int:
    """The reconnaissance step budget, read from the recon READER itself
    (scripts/sdfbase/recon_clip.py RECON_STEPS — pure python, loaded off
    the worktree's scripts/sdfbase like the other local helpers) so the
    launcher and the reader can never disagree about it; never a second
    literal here."""
    _repo_imports()
    import recon_clip
    return int(recon_clip.RECON_STEPS)


def bench_steps(steps: int, overrides: dict) -> int:
    """The optimizer-step count a --mode bench run requests. A bench with
    ANY per-run override (--lr / --max-grad-norm / --warmup-frac) is a
    gradient-norm RECONNAISSANCE run, and the recon reader
    accepts only a record of exactly RECON_STEPS: its stability windows
    certify steps 6-150 alone, so a longer record carries a tail the
    classifier never examined and a shorter one is incomplete (the bench
    default of 200 read stable with 1000x norms over steps 151-200 and
    set the clip). So a recon run's omitted --steps (0) resolves to the
    budget and any other explicit count is refused BEFORE a spawn — a
    record the reader will refuse is GPU time bought for nothing. A generic
    throughput bench (no overrides) keeps BENCH_DEFAULT_STEPS when omitted
    and takes any explicit count as given."""
    recon = bool(overrides)
    n = int(steps or 0)
    if n <= 0:
        return recon_steps() if recon else BENCH_DEFAULT_STEPS
    if recon and n != recon_steps():
        budget = recon_steps()
        raise SystemExit(
            f"--mode bench --steps {n} with per-run overrides "
            f"({', '.join(sorted(overrides))}) is a reconnaissance run at the "
            f"wrong length: the recon reader (scripts/sdfbase/recon_clip.py) "
            f"sets the clip from records of exactly the reconnaissance budget "
            f"{budget} — its stability windows certify steps up to {budget} "
            f"only, so a longer record's tail would be unexamined and a "
            f"shorter one is incomplete — and this run could never set it. "
            f"Pass --steps {budget} or omit --steps")
    return n


def _bench_impl(item: tuple) -> dict:
    """ONE throughput measurement: tune_arm.py bench — the CE adapter's
    production train path (train_object_ce, length-bucketed,
    pre-tokenized sidecar, no held-out pass) on the STAGED pool for
    ``steps`` optimizer steps at ``bs`` x ``accum`` under ``loss_impl``
    (+ optional gradient-checkpointing override), with the trainer's
    per-step token/time record and the heartbeat's continuous SM%
    sampler. Never publishes a cell (max_steps refuses early_stop /
    checkpoints by construction). Item: (key, arm, pool_cell, bs,
    loss_impl, steps, grad_ckpt, accum, check_grads, tag[, lr,
    max_grad_norm, warmup_frac]) — the optional tail is the gradient-norm
    reconnaissance's per-run overrides, None
    = the config block's value. Writes /vol/out/bench/<key>/<tag>.json (a
    NEW tree) and returns the record, whose ``status`` is the bench's OWN
    explicit outcome (ok | diverged | failed, bench_outcome)."""
    import os
    import shutil
    import subprocess
    import sys

    (key, arm, cell, bs, loss_impl, steps, grad_ckpt, accum, check_grads,
     tag) = item[:10]
    lr_o, clip_o, warm_o = (tuple(item[10:13]) + (None, None, None))[:3]
    _ensure_model(key)
    _repo_imports()
    from sisteer.data.instructions import BY_ID

    pairs = _pool_path(key, cell)
    if not os.path.exists(pairs):
        return {"model": key, "arm": arm, "status": "no-pool",
                "error": f"{pairs} not staged"}
    si_label = cell if cell in BY_ID else BENCH_SI_LABEL_FALLBACK
    out_local = Path(f"/root/bench_{key}_{tag}.json")
    cmd = [sys.executable] + bench_cmd(
        cfg=MODELS[key]["cfg"], pairs=pairs, si_label=si_label, arm=arm,
        bs=int(bs), steps=int(steps), out_local=str(out_local),
        loss_impl=loss_impl, grad_ckpt=grad_ckpt, accum=int(accum or 0),
        check_grads=bool(check_grads), lr=lr_o, max_grad_norm=clip_o,
        warmup_frac=warm_o)
    t0 = time.time()
    p = subprocess.run(cmd, cwd="/repo", env=_env(), capture_output=True,
                       text=True)
    res = {"model": key, "arm": arm, "pool_cell": cell, "si_label": si_label,
           "tag": tag, "rc": int(p.returncode),
           "wall_s": round(time.time() - t0, 1), "cmd": cmd[1:]}
    record = json.loads(out_local.read_text()) if out_local.is_file() else None
    if record is not None:
        res.update(record)
    res["status"] = bench_outcome(p.returncode, record)
    # The chunked_ce cross-check line + heartbeat lines are the evidence
    # trail; keep their tail either way.
    keep = [ln for ln in p.stdout.splitlines()
            if "[hb]" in ln or "cross-check" in ln or "[bench]" in ln
            or "[probe-mem]" in ln or "loss_impl=" in ln]
    res["stdout_tail"] = keep[-40:]
    if p.returncode != 0:
        res["stderr_tail"] = str(p.stderr[-3000:])
    d = Path(f"/vol/out/bench/{key}")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tag}.json").write_text(json.dumps(res, indent=1))
    vol.commit()
    return res


def _repo_imports() -> None:
    """Put the repo's src/ and scripts/ trees on sys.path for the helpers
    below, which reuse the DRIVER's and the sdfbase scripts' own functions
    (in-container /repo; locally the worktree). Subprocesses get the same
    through _env()'s PYTHONPATH; the Modal function process itself does
    not."""
    import sys

    for p in (WORKTREE / "src", WORKTREE / "scripts",
              WORKTREE / "scripts" / "sdfbase"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def expected_continuation(cfg_path, pairs_path, si: str, arm: str, *,
                          continue_cap: int, checkpoint,
                          continue_lr_mult: float = 1.0) -> dict:
    """The CONTINUED fit's identity, PROVEN the way the driver proves it:
    the retained checkpoint's key must be reproduced from this config's
    plain identity through the cell's published chain of continuation
    markers (``sisteer.extraction.continuation.resolve_chain`` — the cell
    dir is ``checkpoint.parent.parent``), and the new marker
    (``continuation_marker``: that key, the resume position, the new cap
    [, ``lr_mult``], the ``stopper_rule``) joins the identity.
    Returns ``{"fingerprint", "marker", "chain", "chain_depth",
    "prior_cap", "prior_lr_mult", "prior_stopper_rule",
    "root_fit_fingerprint", "lr_mult", "stopper_rule",
    "replay_exempt_epoch"}`` — ``prior_cap`` None for a first continuation
    (the config's cap applies). Raises ``ContinuationError``
    for an absent / foreign checkpoint, a key the chain does not reproduce
    or a broken chain — so fingerprint_cpu (the pre-spawn check) and the
    snapshot mode refuse exactly what the container's driver would refuse,
    before any GPU spawn."""
    from types import SimpleNamespace

    _repo_imports()
    import train_two_si_objects as t2s
    from sisteer.extraction.continuation import (
        STOPPER_RULE_REPLAY_EXEMPT, check_cap_admissible, check_lr_mult,
        continuation_marker, load_state, resolve_chain, summarize)

    lr_mult = check_lr_mult(continue_lr_mult)
    # The rule the driver runs every continuation under (not a knob) —
    # derived here so the row's fingerprint IS the container's.
    stopper_rule = STOPPER_RULE_REPLAY_EXEMPT
    ckpt = Path(checkpoint)
    state = load_state(ckpt)
    cfg = t2s._cfg_with_positions(t2s._load_cfg(str(cfg_path)), "all")
    pool_sha, config_sha = t2s._cli_fit_identity(
        SimpleNamespace(pairs=str(pairs_path), config=str(cfg_path)), cfg)

    def _fp_of(marker):
        return t2s.cell_fit_fingerprint(cfg, si, arm, positions="all",
                                        pool_sha256=pool_sha,
                                        config_sha=config_sha,
                                        teacher="online",
                                        continued_from=marker or None)

    chain = resolve_chain(ckpt.parent.parent, state, _fp_of(None), _fp_of)
    summ = summarize(state, ckpt)
    # The driver's cap / multiplier / tail rules, applied HERE too (the
    # same helper), so fingerprint_cpu and snapshot refuse before any GPU
    # spawn exactly what _prepare_continuation would refuse after model
    # load. The SI's configured cap is resolved the
    # way the driver resolves `epochs` (train_kwargs_for_si).
    config_cap = int(t2s._train_kwargs_for_si(cfg, si).get("epochs", 6))
    adm = check_cap_admissible(chain=chain, summary=summ,
                               new_cap=int(continue_cap), lr_mult=lr_mult,
                               config_cap=config_cap, stopper_rule=stopper_rule)
    marker = continuation_marker(summ, int(continue_cap), lr_mult,
                                 stopper_rule)
    return {"fingerprint": _fp_of(marker), "marker": marker,
            "chain": [dict(m) for m in chain["chain"]] + [dict(marker)],
            "chain_depth": int(chain["depth"]) + 1,
            "prior_cap": chain["prior_cap"],
            "prior_lr_mult": float(chain["prior_lr_mult"]),
            "prior_stopper_rule": adm["prior_stopper_rule"],
            "root_fit_fingerprint": chain["root_fit_fingerprint"],
            "lr_mult": lr_mult, "stopper_rule": stopper_rule,
            "replay_exempt_epoch": int(summ["epoch"])}


def expected_fit_fingerprint(cfg_path, pairs_path, si: str, arm: str, *,
                             continue_cap: int | None = None,
                             checkpoint=None,
                             continue_lr_mult: float = 1.0) -> str:
    """The fit fingerprint the driver WILL stamp into this cell's COMPLETE
    marker — derived through the driver's OWN functions (the same
    ``cell_fit_fingerprint`` + ``_cli_fit_identity`` pair ``cmd_train``
    calls, under the exact invocation _fit_impl issues: ``--positions
    all``, no teacher artifact), never re-implemented, so the manifest
    row (fits-spawn, via fingerprint_cpu), the in-container gates
    (supersede_gate, retain_final_checkpoint) and the marker itself can
    never disagree. The config's shared held-out pool is a /vol path, so
    this resolves only where the pools are mounted (a container — or a
    test's tmp config pointing at tmp pools).

    ``continue_cap`` + ``checkpoint``: the CONTINUED fit's fingerprint
    through :func:`expected_continuation` — the checkpoint key PROVEN
    from the plain identity through the cell's published chain, plus the
    new marker (prior fingerprint, resume step/epoch/batch, the new cap,
    ``continue_lr_mult`` when not 1.0), exactly what the driver's
    ``--continue-from`` stamps. Raises ``ContinuationError`` when the
    checkpoint is absent, not a train_object_ce checkpoint, or not
    reproducible from this identity."""
    from types import SimpleNamespace

    if continue_cap:
        return expected_continuation(
            cfg_path, pairs_path, si, arm, continue_cap=int(continue_cap),
            checkpoint=checkpoint,
            continue_lr_mult=continue_lr_mult)["fingerprint"]
    # The plain fit's derivation is SHARED with the Slurm path
    # (scripts/sdfbase/fit_cell_identity.py): one function, whichever
    # path is asking.
    _repo_imports()
    from fit_cell_identity import expected_fit_fingerprint as _plain

    return _plain(cfg_path, pairs_path, si, arm)


def continuation_summary(checkpoint) -> dict:
    """The retained checkpoint's tensor-free description for a manifest
    row's ``continued_from`` (prior fingerprint, resume position, stopper
    state) — the same loader the fingerprint derivation uses."""
    _repo_imports()
    from sisteer.extraction.continuation import load_state, summarize

    summ = summarize(load_state(checkpoint), checkpoint)
    return {"prior_fingerprint": summ["checkpoint_key"],
            "resume_step": summ["step"], "resume_epoch": summ["epoch"],
            "resume_batches_done": summ["batches_done"],
            "n_batches": summ["n_batches"], "stopper": summ["stopper"],
            "replayed_partial_epochs": summ.get("replayed_partial_epochs"),
            "checkpoint": summ["path"]}


def _marker_fingerprint(cell_dir) -> tuple[str, str | None]:
    """(``'absent' | 'unparseable' | 'present'``, the marker's
    fit_fingerprint or None) for a cell dir's COMPLETE marker — the shared
    gate (scripts/sdfbase/fit_cell_identity.py, also the Slurm path's)."""
    _repo_imports()
    from fit_cell_identity import marker_fingerprint

    return marker_fingerprint(cell_dir)


def supersede_gate(cell_dir, expect_fingerprint: str) -> str | None:
    """None when the cell dir may be (re)fit under ``expect_fingerprint``:
    no COMPLETE marker (a fresh or in-flight cell), a marker already
    carrying this fingerprint (the ordinary in-container resume/skip), or
    a marker of a DIFFERENT identity whose dir carries a superseded_by.json
    naming that very fingerprint (scripts/sdfbase/snapshot_cells.py took
    the snapshot — the exact event that licenses the overwrite). Otherwise
    the problem string: a published cell of another identity is about to
    be overwritten with no snapshot, the pre-spawn item a refit
    must never skip. ONE derivation, shared with the Slurm path
    (fit_cell_identity.supersede_gate)."""
    _repo_imports()
    from fit_cell_identity import supersede_gate as _shared

    return _shared(cell_dir, expect_fingerprint)


def retain_final_checkpoint(cell_dir, ckpt_cell, *,
                            expect_fingerprint: str) -> bool:
    """Move a COMPLETE cell's transient checkpoint files INTO the published
    cell dir as ``final_ckpt/``: round 1 deleted checkpoints at publish,
    which made exact resume of the cap-limited 1M cells impossible, and
    retention costs megabytes. Returns True when something was
    retained.

    Fires ONLY when the dir's COMPLETE marker carries ``expect_fingerprint``
    — the fit this container ran or validated-skipped. A missing marker, a
    marker of ANOTHER identity (a stale round-1 publication under a refit
    that failed or is still in flight), or no checkpoint files, moves
    nothing: the transient checkpoint is the successor's resume point and
    belongs to the NEW fit, never to the old cell dir. A pre-existing
    final_ckpt/ is superseded (an --overwrite respawn's new checkpoint
    replaces the stale one, never merges with it). ONE derivation, shared
    with the Slurm path (fit_cell_identity.retain_final_checkpoint)."""
    _repo_imports()
    from fit_cell_identity import retain_final_checkpoint as _shared

    return _shared(cell_dir, ckpt_cell, expect_fingerprint=expect_fingerprint)


def _fit_impl(item: tuple) -> dict:
    """ONE fit cell: (model_key, cell, arm, expect_sha, overwrite
    [, out_root, checkpoint_every[, continue_cap]]) — the
    twosi_fits._fit_impl pattern:
    seed resume from the volume, train through the production driver
    (--positions all --token-cache auto --expect-config-sha), COMPLETE
    self-skip on the driver's fingerprint, checkpoint-committer daemon
    thread (30s mtime poll + vol.commit), nvidia-smi duty sampler, publish
    to the volume, then MOVE the transient checkpoint into the published
    cell dir as final_ckpt/ (retain_final_checkpoint — never delete it).
    The shared held-out pool rides the CONFIG
    (train_kwargs.default.heldout_pool =
    /vol/pool/<key>/pairs_constv2-heldout.jsonl), not a CLI flag."""
    require_wandb_entity()  # R2-D16: never log a fit to a foreign account
    import os
    import shutil
    import subprocess
    import sys
    import threading
    from pathlib import Path

    key, si, arm, expect_sha, overwrite = item[:5]
    out_root = str(item[5]) if len(item) > 5 and item[5] else f"out/{key}"
    ckpt_every = int(item[6]) if len(item) > 6 and item[6] else 0
    # A raised cap to CONTINUE the cell's retained checkpoint under
    # (0 = an ordinary fit), and the continuation's lr multiplier
    # (absent / 1.0 = the plain warm restart).
    cont_cap = int(item[7]) if len(item) > 7 and item[7] else 0
    # (`is not None`, not truthiness: an explicit 0 must reach the driver's
    # check_lr_mult and be refused, never silently become 1.0)
    cont_lr_mult = (float(item[8]) if len(item) > 8 and item[8] is not None
                    else 1.0)
    spec = MODELS[key]
    # A reused warm fit container keeps the volume snapshot it was mounted
    # with: without a reload a successor could miss its predecessor's newer
    # transient checkpoint (committed by another container) and re-adopt
    # final_ckpt/ — resuming from the ORIGINAL step with the resume
    # assertion still satisfied, hours of progress silently lost — or gate
    # on a stale COMPLETE / superseded_by.json. Before ANY /vol read.
    vol.reload()
    _ensure_model(key)

    pairs = _pool_path(key, si)
    if not os.path.exists(pairs):
        return {"model": key, "si": si, "arm": arm, "positions": "all",
                "status": "no-pool", "error": f"{pairs} not staged"}
    heldout = _pool_path(key, HELDOUT_SI)
    if not os.path.exists(heldout):
        return {"model": key, "si": si, "arm": arm, "positions": "all",
                "status": "no-heldout-pool",
                "error": f"{heldout} not staged — the shared held-out "
                         "pool is the CE arms' whole verdict"}

    leg = "all"
    cell_id = f"{si}__{arm}"
    local_out = Path(f"/root/out/{key}/cells_{leg}")
    vol_out = Path(f"/vol/{out_root}/cells_{leg}")
    local_out.mkdir(parents=True, exist_ok=True)
    # Continuation: the retained checkpoint must exist (a cell that
    # never retained one cannot be continued) and --overwrite is refused
    # (the continuation supersedes through the snapshot gate below, never
    # by a blind overwrite of the publication it continues).
    final_ckpt = vol_out / cell_id / "final_ckpt" / "ckpt.pt"
    if cont_cap and overwrite:
        return {"model": key, "si": si, "arm": arm, "positions": "all",
                "status": "refused-continue-with-overwrite",
                "error": "--continue-cap with --overwrite: a continuation "
                         "never blind-overwrites the publication it "
                         "continues"}
    if cont_cap and not final_ckpt.is_file():
        return {"model": key, "si": si, "arm": arm, "positions": "all",
                "status": "refused-no-final-checkpoint",
                "error": f"{final_ckpt} absent — the cell retained no "
                         "checkpoint to continue from"}
    if cont_lr_mult != 1.0 and not cont_cap:
        return {"model": key, "si": si, "arm": arm, "positions": "all",
                "status": "refused-lr-mult-without-continuation",
                "error": f"continue_lr_mult {cont_lr_mult} on an ordinary "
                         "fit: the multiplier scales a continuation's "
                         "restarted schedule only"}
    # This fit's identity, through the driver's own derivation — the value
    # the manifest row expects, the marker will carry, and the two gates
    # below key on. For a continuation: the prior identity + the marker
    # derived from the retained checkpoint (what --continue-from stamps),
    # with the checkpoint key PROVEN through the cell's published chain
    # — a key the chain does not reproduce is a deterministic refusal
    # (a `done` outcome the watcher triages, never a respawn).
    _repo_imports()
    from sisteer.extraction.continuation import ContinuationError

    try:
        expect_fp = expected_fit_fingerprint(
            spec["cfg"], pairs, si, arm, continue_cap=cont_cap or None,
            checkpoint=(final_ckpt if cont_cap else None),
            continue_lr_mult=cont_lr_mult)
    except ContinuationError as e:
        print(f"[fit] REFUSED {cell_id}: {e}", flush=True)
        return {"model": key, "si": si, "arm": arm, "positions": "all",
                "status": "refused-continuation-identity",
                "continue_cap": cont_cap, "error": str(e)[:2000]}
    # Stale-COMPLETE supersede gate: a published cell of ANOTHER identity
    # is overwritten only after snapshot_cells.py preserved it.
    gate = supersede_gate(vol_out / cell_id, expect_fp)
    if gate:
        print(f"[fit] REFUSED {cell_id}: {gate}", flush=True)
        return {"model": key, "si": si, "arm": arm, "positions": "all",
                "status": "refused-unsnapshotted-supersede",
                "expect_fingerprint": expect_fp, "error": gate}
    # Seed the fingerprint-validated resume (COMPLETE marker + sidecars).
    if (vol_out / _STORE_IDENT).exists():
        shutil.copy2(vol_out / _STORE_IDENT, local_out / _STORE_IDENT)
    if (vol_out / cell_id).exists() and not overwrite:
        shutil.copytree(vol_out / cell_id, local_out / cell_id,
                        dirs_exist_ok=True)

    # GPU-duty spot-check once the optimizer loop is live (gpu-util skill).
    done = threading.Event()
    metrics_glob = local_out / cell_id / "metrics.jsonl"
    dmon_rows = _start_dmon(metrics_glob, done)

    # Mid-fit checkpoints: written by the driver straight onto the volume
    # (transient path, never inside a cell dir); this thread commits the
    # volume after each new write so a killed container's successor can
    # resume from the last committed checkpoint.
    ckpt_cell = Path(f"/vol/{out_root}/ckpt/{cell_id}")

    def commit_ckpts():
        last = 0.0
        while not done.is_set():
            done.wait(30)
            try:
                p = ckpt_cell / "ckpt.pt"
                m = p.stat().st_mtime if p.exists() else 0.0
            except OSError:
                continue
            if m > last:
                last = m
                try:
                    vol.commit()
                    print(f"[fitckpt] volume commit after checkpoint "
                          f"write ({cell_id})", flush=True)
                except Exception as e:  # noqa: BLE001 — commit retries next tick
                    print(f"[fitckpt] vol.commit failed ({e}); will retry",
                          flush=True)

    threading.Thread(target=commit_ckpts, daemon=True).start()

    cmd = [
        sys.executable, "/repo/scripts/train_two_si_objects.py", "train",
        "--config", spec["cfg"], "--si", si, "--arms", arm,
        "--pairs", pairs, "--token-cache", "auto",
        "--out", str(local_out),
        "--expect-config-sha", expect_sha,
        "--positions", "all",
        "--checkpoint-dir", f"/vol/{out_root}/ckpt",
    ]
    if ckpt_every:
        cmd += ["--checkpoint-every", str(ckpt_every)]
    if cont_cap:
        # The driver verifies the checkpoint against the PUBLISHED cell on
        # the volume (COMPLETE fingerprint == checkpoint key, sidecar/object
        # shas, resume position inside the recorded fit, best snapshot ==
        # object when applicable), adopts it under the continuation
        # fingerprint into --checkpoint-dir, and refuses to publish unless
        # the trainer reports it resumed.
        cmd += ["--continue-from", str(final_ckpt),
                "--epochs-cap", str(cont_cap),
                "--continue-lr-mult", repr(float(cont_lr_mult))]
    if overwrite:
        cmd.append("--overwrite")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/repo", env=_env(),
                          capture_output=True, text=True)
    wall = time.time() - t0
    done.set()

    res: dict = {"model": key, "si": si, "arm": arm,
                 "positions": "all", "rc": int(proc.returncode),
                 "wall_s": round(float(wall), 1),
                 "expect_fingerprint": expect_fp}
    if cont_cap:
        res["continue_cap"] = cont_cap
        res["continue_lr_mult"] = cont_lr_mult
        res["continuation_lines"] = [
            ln for ln in proc.stdout.splitlines()
            if "[train-two-si] continuation" in ln][:20]
    skipped = "skipping (resume)" in proc.stdout or \
        "validated COMPLETE" in proc.stdout
    res["skipped"] = bool(skipped and proc.returncode == 0)
    ck_lines = [ln for ln in proc.stdout.splitlines() if "[fitckpt]" in ln]
    if ck_lines:
        res["fitckpt_lines"] = ck_lines[:30]
        print("\n".join(ck_lines[:30]), flush=True)
    mean_sm = _mean_sm_pct(dmon_rows)
    if mean_sm is not None:
        res["mean_sm_pct"] = mean_sm
    report = local_out / f"{si}__fit_report.json"
    if report.exists():
        rep = json.loads(report.read_text()).get("arms", {}).get(arm, {})
        res["status"] = rep.get("status", "no-arm-record")
        if rep.get("error"):
            res["error"] = f"{rep.get('error_type')}: {rep['error']}"[:800]
    else:
        res["status"] = "no-report"
        res["stderr_tail"] = str(proc.stderr[-1500:])
    spst_med = _median_sec_per_step(metrics_glob)
    if spst_med is not None:
        res["sec_per_step_median"] = spst_med

    if not res["skipped"]:
        # Publish: the cell dir (object + COMPLETE + sidecars + per-cell
        # metrics) verbatim; report/metrics tees under per-cell names so
        # concurrent same-cell containers cannot clobber each other's copy.
        vol_out.mkdir(parents=True, exist_ok=True)
        if (local_out / cell_id).exists():
            shutil.copytree(local_out / cell_id, vol_out / cell_id,
                            dirs_exist_ok=True)
        if ((local_out / _STORE_IDENT).exists()
                and not (vol_out / _STORE_IDENT).exists()):
            shutil.copy2(local_out / _STORE_IDENT, vol_out / _STORE_IDENT)
        rdir = Path(f"/vol/{out_root}/reports")
        rdir.mkdir(parents=True, exist_ok=True)
        if report.exists():
            shutil.copy2(report, rdir / f"report_{cell_id}_{leg}.json")
        tee = local_out / f"{si}__metrics.jsonl"
        if tee.exists():
            shutil.copy2(tee, rdir / f"metrics_{cell_id}_{leg}.jsonl")
        vol.commit()
    # A COMPLETE cell RETAINS its last checkpoint as <cell>/final_ckpt/
    # (retain_final_checkpoint — the why lives on that function). Safe next
    # to the COMPLETE protocol: the marker's manifest validates ONLY its
    # listed files (validated_complete checks files.items();
    # _required_publication_files never globs), so a post-publish
    # final_ckpt/ can't invalidate the cell.
    #
    # Resume semantics under a RAISED CAP: the retained checkpoint_key is
    # the OLD fit fingerprint, so fitckpt's loader refuses it for any new
    # identity — which is why the continuation path (item[7] > 0) has the
    # DRIVER adopt it under the continuation fingerprint after proving it
    # is the prior fit's under exactly this config and belongs to the
    # published object (--continue-from). The warmup_cosine caveat stands
    # and is now QUANTIFIED rather than avoided: the continuation runs the
    # cap-N schedule from the resume step onward (the trainer's own
    # arithmetic), a warm restart from the decayed ~0 lr, recorded in
    # continued_from.lr_schedule of the sidecar/report/marker. Retention
    # (below) applies to the continued fit too: its own last checkpoint
    # replaces the cap-24 one in final_ckpt/ (the cap-24 copy lives in the
    # snapshot bank the operator took first).
    #
    # Retention fires ONLY after a SUCCESSFUL driver run (rc 0 — a fit or
    # a validated skip) AND against a marker carrying THIS fit's
    # fingerprint (retain_final_checkpoint's own gate): a failed refit
    # leaves a stale round-1 marker under the cell dir, and moving the
    # fresh checkpoint into THAT dir would strand the successor's resume
    # point inside the old publication.
    res["final_ckpt_retained"] = False
    if proc.returncode == 0 and retain_final_checkpoint(
            vol_out / cell_id, ckpt_cell, expect_fingerprint=expect_fp):
        vol.commit()
        res["final_ckpt_retained"] = True
        print(f"[fitckpt] {cell_id} published COMPLETE — last checkpoint "
              "retained at final_ckpt/", flush=True)
    return res


def _config_sha16(cfg_path) -> str:
    """The 16-hex config-bytes sha fits-spawn pins as ``expect_sha`` (the
    driver's ``--expect-config-sha`` contract), from whichever copy of the
    config is at ``cfg_path`` — local worktree or baked /repo."""
    import hashlib

    return hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest()[:16]


def _fingerprint_impl(item: tuple) -> dict:
    """IN-CONTAINER: the expected fit fingerprint of every requested cell,
    (model_key, [[si, arm], ...]) -> {"model", "config_sha", "fingerprints":
    {cell_id: fp}}. Runs where the staged pools are mounted (the fingerprint
    hashes the pairs FILE) through the driver's own derivation
    (expected_fit_fingerprint), so the manifest row fits-spawn writes is
    the value the container will stamp into the COMPLETE marker — never a
    local re-implementation. ``config_sha`` is the BAKED config's sha16:
    fits-spawn refuses when it differs from the local ``expect_sha`` (a
    stale deploy would fingerprint the wrong config bytes)."""
    import os

    key, pairs_list = item[0], item[1]
    # (model_key, pairs, continue_cap, out_root[, continue_lr_mult])
    # derives CONTINUED fingerprints from each cell's retained final_ckpt/
    # on the volume — through the chain-proving derivation, so a
    # checkpoint the container would refuse is refused HERE, before any
    # GPU spawn.
    cont_cap = int(item[2]) if len(item) > 2 and item[2] else 0
    out_root = str(item[3]) if len(item) > 3 and item[3] else f"out/{key}"
    cont_lr_mult = (float(item[4]) if len(item) > 4 and item[4] is not None
                    else 1.0)
    spec = MODELS[key]
    # A warm CPU container keeps the volume snapshot it was mounted with;
    # the retained checkpoints + COMPLETE markers read here were written by
    # OTHER containers since — reload first.
    vol.reload()
    fps: dict[str, str] = {}
    conts: dict[str, dict] = {}
    missing: list[str] = []
    missing_ckpt: list[str] = []
    for si, arm in pairs_list:
        pairs = _pool_path(key, si)
        cid = f"{si}__{arm}"
        if not os.path.exists(pairs):
            missing.append(f"{cid}: {pairs} not staged")
            continue
        if not cont_cap:
            fps[cid] = expected_fit_fingerprint(spec["cfg"], pairs, si, arm)
            continue
        ckpt = Path(f"/vol/{out_root}/cells_all/{cid}/final_ckpt/ckpt.pt")
        try:
            exp = expected_continuation(
                spec["cfg"], pairs, si, arm, continue_cap=cont_cap,
                checkpoint=ckpt, continue_lr_mult=cont_lr_mult)
            fps[cid] = exp["fingerprint"]
            conts[cid] = {**continuation_summary(ckpt),
                          "chain": exp["chain"],
                          "chain_depth": exp["chain_depth"],
                          "root_fit_fingerprint": exp["root_fit_fingerprint"],
                          "prior_cap": exp["prior_cap"],
                          "prior_lr_mult": exp["prior_lr_mult"],
                          "lr_mult": exp["lr_mult"],
                          "stopper_rule": exp["stopper_rule"],
                          "prior_stopper_rule": exp["prior_stopper_rule"],
                          "replay_exempt_epoch": exp["replay_exempt_epoch"]}
        except Exception as e:  # noqa: BLE001 — reported, refused at spawn
            missing_ckpt.append(f"{cid}: {type(e).__name__}: {e}")
    out = {"model": key, "config_sha": _config_sha16(spec["cfg"]),
           "fingerprints": fps, "missing_pools": missing}
    if cont_cap:
        out.update({"continue_cap": cont_cap,
                    "continue_lr_mult": cont_lr_mult,
                    "continuations": conts,
                    "missing_checkpoints": missing_ckpt})
    return out


#: Snapshot bank suffixes: ``_r1`` (the round-1 winners), ``_r2cap24``
#: (the cap-24 refits a continuation supersedes).
_BANK_SUFFIX = re.compile(r"^_[A-Za-z0-9]+$")


def snapshot_bank_root(out_root: str, suffix: str = "_r1") -> str:
    """The snapshot bank for a live bank root: ``out/<key>`` ->
    ``out/<key><suffix>`` (default ``_r1``; ``_r2cap24`` for the cap-24
    publications a continuation supersedes; volume-relative, like
    ``out_root``). ONE derivation for --mode snapshot, the operator
    examples and the mmlu ``--defenses-root`` default of an r1 run."""
    if not _BANK_SUFFIX.match(str(suffix)):
        raise ValueError(f"snapshot bank suffix {suffix!r} must look like "
                         "_r1 / _r2cap24 (an underscore + alphanumerics)")
    root = str(out_root).rstrip("/")
    return f"{root}{suffix}"


def _snapshot_impl(item: tuple) -> dict:
    """IN-CONTAINER: snapshot (or --verify-only re-verify) the named
    COMPLETE cells of ``/vol/<out_root>`` into ``/vol/<out_root>_r1``
    through scripts/sdfbase/snapshot_cells.py — file list + sha256
    verified against each COMPLETE manifest on both sides, supersede
    records written on both sides. Each cell's successor fingerprint (the
    refit this snapshot licenses) is derived in-container and recorded in
    superseded_by.json. (model_key, [cell_id, ...], verify_only, reason
    [, out_root[, bank_suffix[, continue_cap]]]) -> per-cell verdicts;
    ``status`` is "ok" only when every cell verified. ``bank_suffix``
    (default ``_r1``) names the snapshot bank; ``continue_cap`` records
    the CONTINUATION's fingerprint as the successor — derived from the
    cell's retained final_ckpt/, and REFUSING a cell without one (a cell
    that cannot be continued must not be snapshotted as if it were about
    to be)."""
    key, cells, verify_only, reason = item[:4]
    out_root = str(item[4]) if len(item) > 4 and item[4] else f"out/{key}"
    suffix = str(item[5]) if len(item) > 5 and item[5] else "_r1"
    cont_cap = int(item[6]) if len(item) > 6 and item[6] else 0
    cont_lr_mult = (float(item[7]) if len(item) > 7 and item[7] is not None
                    else 1.0)
    spec = MODELS[key]
    _repo_imports()
    from snapshot_cells import SnapshotError, snapshot_cell, verify_snapshot
    from sisteer.extraction.continuation import ContinuationError

    # A reused warm container reads the volume snapshot it was mounted
    # with: the first cont2 --verify-only pass refused all 8 cells as
    # "carries no COMPLETE marker" with the data on the volume. Reload
    # before reading, both branches.
    vol.reload()

    src_bank = Path(f"/vol/{out_root}")
    dst_bank = Path(f"/vol/{snapshot_bank_root(out_root, suffix)}")
    rows: list[dict] = []
    for cell in cells:
        row: dict = {"cell_id": cell, "snapshot_bank": str(dst_bank)}
        try:
            if verify_only:
                rec = verify_snapshot(dst_bank, cell)
                row.update(status="verified",
                           fit_fingerprint=rec.get("fit_fingerprint"))
            else:
                si, arm = cell.split("__", 1)
                successor = {"reason": reason}
                pairs = _pool_path(key, si)
                ckpt = src_bank / "cells_all" / cell / "final_ckpt" / "ckpt.pt"
                if cont_cap:
                    successor["continue_cap"] = cont_cap
                    if cont_lr_mult != 1.0:
                        successor["continue_lr_mult"] = cont_lr_mult
                    # The rule the licensed continuation runs under
                    from sisteer.extraction.continuation import (
                        STOPPER_RULE_REPLAY_EXEMPT)
                    successor["stopper_rule"] = STOPPER_RULE_REPLAY_EXEMPT
                    if not ckpt.is_file():
                        raise SnapshotError(
                            f"{ckpt} absent — the cell retained no "
                            "checkpoint, so no continuation can follow "
                            "this snapshot; refusing (R2-D20)")
                if Path(pairs).exists():
                    # For a continuation this PROVES the checkpoint key
                    # through the cell's published chain: a cell whose
                    # continuation the container would refuse is refused
                    # here, never snapshotted as if about to be continued
                    # (a superseded_by.json naming a fingerprint nothing
                    # can produce).
                    try:
                        successor["expected_refit_fingerprint"] = \
                            expected_fit_fingerprint(
                                spec["cfg"], pairs, si, arm,
                                continue_cap=cont_cap or None,
                                checkpoint=(ckpt if cont_cap else None),
                                continue_lr_mult=cont_lr_mult)
                    except ContinuationError as e:
                        raise SnapshotError(
                            f"continuation identity refused: {e}") from e
                rec = snapshot_cell(src_bank, dst_bank, cell,
                                    superseded_by=successor)
                row.update(status="snapshotted",
                           already_present=rec["already_present"],
                           fit_fingerprint=rec["fit_fingerprint"],
                           n_files=len(rec["files"]),
                           final_ckpt_retained=rec.get(
                               "final_ckpt_retained"),
                           reports=sorted(rec.get("reports") or {}),
                           expected_refit_fingerprint=successor.get(
                               "expected_refit_fingerprint"))
        except (SnapshotError, ValueError) as e:
            row.update(status="refused", error=str(e)[:1200])
        rows.append(row)
        print(f"[snapshot] {key}/{cell}: {row['status']}"
              + (f" — {row['error']}" if row.get("error") else ""),
              flush=True)
    if not verify_only:
        vol.commit()
    bad = [r for r in rows if r["status"] == "refused"]
    return {"model": key, "verify_only": bool(verify_only),
            "status": "ok" if not bad else "failed",
            "n_ok": len(rows) - len(bad), "n_bad": len(bad), "rows": rows}


def resolve_spawn_sis(sis: str, retrain_all: bool) -> list[str]:
    """The data cells a fits-spawn will spawn — REFUSING the unscoped
    default: the round-2 config regeneration moved config_sha for every
    cell, so a bare spawn would retrain all nine SIs from scratch.
    ``--sis a,b`` scopes it; ``--retrain-all`` is the one explicit
    acknowledgment that a full retrain is the intent (and is refused
    together with --sis: the two state different intents)."""
    want = [s for s in (sis or "").split(",") if s]
    if want and retrain_all:
        raise SystemExit("[fits-spawn] --sis and --retrain-all together "
                         "state two different intents — pass exactly one")
    if not want and not retrain_all:
        raise SystemExit(
            "[fits-spawn] REFUSED: no --sis given. The round-2 config "
            "regeneration moved config_sha for EVERY cell, so an unscoped "
            "spawn would retrain all 9 SIs from scratch. Scope the wave "
            "(--sis constv2-mix-1m,constv2-mix-10m,constv2-mix-full) or "
            "state the full-retrain intent explicitly with --retrain-all")
    unknown = [s for s in want if s not in SIS]
    if unknown:
        raise SystemExit(f"[fits-spawn] unknown --sis {unknown}; the data "
                         f"cells are {SIS}")
    return want or list(SIS)


#: The per-row SPAWN PIN: the deployed function a cell was spawned on
#: and the item fields its successor must reproduce. One manifest
#: carries a whole campaign, so the top-level copies describe only the
#: LAST fits-spawn that wrote it; fits_watch.py reads the ROW's.
ROW_PIN_FIELDS = ("app", "function", "function_id", "config_sha", "overwrite",
                  "checkpoint_every", "continue_cap", "continue_lr_mult",
                  "app_id", "app_version", "deploy_commit")


def spawn_pin(fname: str, function_id: str | None, expect_sha: str, *,
              overwrite: bool, checkpoint_every: int,
              continue_cap: int = 0,
              app: str = DEFAULT_APP_NAME,
              continue_lr_mult: float = 1.0,
              deployment: dict | None = None) -> dict:
    """The spawn pin every row of ONE fits-spawn carries (ROW_PIN_FIELDS).
    ``continue_cap`` is 0 for an ordinary fit; a continuation row's
    successor rebuilds the cap into its item (fits_watch);
    ``continue_lr_mult`` likewise, 1.0 for the plain warm restart.
    ``app`` is the DEPLOYED app the row spawns on (resolve_app_name), so
    a wave under an override name is re-spawned on that app, never on
    the default. ``deployment`` (fits_watch.deployment_pin) is what a
    redeploy actually changes: ``app_id`` + the app's production
    ``app_version`` (+ the deploy's ``deploy_commit`` for humans); a
    deployed function's id does NOT change on redeploy (v10 -> v11 kept
    both fit function ids), so the watcher pins the version, and refuses
    a successor for a row that has none unless told the version by
    hand."""
    dep = deployment or {}
    return {"app": app, "function": fname,
            "function_id": function_id, "config_sha": expect_sha,
            "overwrite": bool(overwrite),
            "checkpoint_every": int(checkpoint_every),
            "continue_cap": int(continue_cap or 0),
            "continue_lr_mult": float(1.0 if continue_lr_mult is None
                                      else continue_lr_mult),
            "app_id": dep.get("app_id"),
            "app_version": dep.get("app_version"),
            "deploy_commit": dep.get("deploy_commit")}


def _deployment_pin(app_name: str) -> dict:
    """fits_watch.deployment_pin, imported from the sibling file by path
    (the watcher's module is import-safe without modal; sdfbase_fits never
    imports it at module level so `modal run` keeps its import graph)."""
    import importlib.util

    fw_path = Path(__file__).resolve().parent / "fits_watch.py"
    spec = importlib.util.spec_from_file_location("_fits_watch_pin", fw_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.deployment_pin(app_name)


def continuation_cells(cells: str, *, sis: str = "", arms: str = "",
                       retrain_all: bool = False, overwrite: bool = False,
                       cap: int = 0) -> list[tuple[str, str]]:
    """The explicit (si, arm) cells of a ``fits-spawn --continue-cap``:
    ``--cells <si>__<arm>,...`` is REQUIRED and is the only selector
    — --sis/--arms/--retrain-all (a product or a wave, never a named cell)
    and --overwrite (a continuation supersedes through the snapshot gate,
    never a blind overwrite) are refused, as are a cap < 1, a malformed or
    unknown cell id and a duplicate."""
    tag = "[fits-spawn] --continue-cap"
    if int(cap) < 1:
        raise SystemExit(f"{tag}: the raised cap must be >= 1, got {cap}")
    if sis or arms or retrain_all:
        raise SystemExit(f"{tag} takes --cells only: --sis/--arms/"
                         "--retrain-all select a product or a wave, and a "
                         "continuation is per NAMED cell")
    if overwrite:
        raise SystemExit(f"{tag} with --overwrite refused: a continuation "
                         "supersedes the cap-limited publication through "
                         "the snapshot gate, never by blind overwrite")
    want = [c for c in (cells or "").split(",") if c]
    if not want:
        raise SystemExit(f"{tag} needs --cells <si>__<arm>,... (e.g. "
                         "constv2-mix-1m__psr_ce_all,"
                         "constv2-mix-1m__lora_r4_ce)")
    pairs: list[tuple[str, str]] = []
    for c in want:
        if "__" not in c:
            raise SystemExit(f"{tag}: {c!r} is not a <si>__<arm> cell id")
        si, arm = c.split("__", 1)
        if si not in SIS or arm not in CE_ARMS:
            raise SystemExit(f"{tag}: unknown cell {c!r} (SIs {SIS}, arms "
                             f"{CE_ARMS})")
        if (si, arm) in pairs:
            raise SystemExit(f"{tag}: {c!r} named twice")
        pairs.append((si, arm))
    return pairs


def _check_continue_lr_mult(value, continue_cap, tag: str) -> float:
    """``--continue-lr-mult`` as a finite positive float — refused when
    not 1.0 without ``--continue-cap`` (the multiplier scales a
    continuation's restarted schedule; on an ordinary fit the lr is a
    config value, and moving it here would alias a config change)."""
    import math

    try:
        m = float(value if value is not None else 1.0)
    except (TypeError, ValueError):
        raise SystemExit(f"{tag} --continue-lr-mult must be a number, got "
                         f"{value!r}") from None
    if not math.isfinite(m) or m <= 0:
        raise SystemExit(f"{tag} --continue-lr-mult must be a finite "
                         f"positive multiplier, got {value!r}")
    if m != 1.0 and not int(continue_cap or 0):
        raise SystemExit(f"{tag} --continue-lr-mult {m} without "
                         "--continue-cap: the multiplier scales a "
                         "CONTINUATION's restarted schedule only")
    return m


def spawn_rows(key: str, root: str, want_sis: list[str],
               want_arms: list[str], fingerprints: dict[str, str], *,
               pin: dict | None = None,
               cells: list[tuple[str, str]] | None = None,
               continuations: dict | None = None) -> list[dict]:
    """The manifest rows of ONE fits-spawn, each carrying the cell's
    EXPECTED fit fingerprint so fits_watch.py can validate a COMPLETE
    marker's identity instead of its existence, and its own SPAWN PIN
    (``pin`` — :func:`spawn_pin`) so the watcher respawns it on the
    function it ran on with its own config sha, never the manifest-level
    values of whichever model was spawned last — including the deployed
    ``app`` it spawned on, so the watcher's strictness and re-spawn hint
    follow the row, not a default. A cell with no derived fingerprint
    refuses the whole spawn: a row that cannot be validated would sit
    forever as 'complete' on a stale marker.

    ``cells`` names explicit (si, arm) pairs instead of the product;
    ``continuations`` (cell id -> the retained checkpoint's summary,
    fingerprint_cpu's ``continuations``) lands on the row as
    ``continued_from`` so the manifest says what each row resumes."""
    rows = []
    pairs = (list(cells) if cells is not None
             else [(si, arm) for si in want_sis for arm in want_arms])
    for si, arm in pairs:
        cid = f"{si}__{arm}"
        fp = fingerprints.get(cid)
        if not fp:
            raise SystemExit(
                f"[fits-spawn] no expected fit fingerprint for {cid} "
                "(fingerprint_cpu derived none — is its pool staged?); "
                "refusing to spawn a cell the watcher could not "
                "validate")
        row = {
            "model": key, "si": si, "arm": arm,
            "positions": "all", "leg": "all",
            "cell_id": cid, "out_root": root,
            "expect_fingerprint": fp,
            **{k: (pin or {}).get(k) for k in ROW_PIN_FIELDS},
            "call_id": None, "spawn_time": None,
            "status": "pending-spawn",
        }
        if continuations and cid in continuations:
            row["continued_from"] = dict(continuations[cid])
        rows.append(row)
    return rows


def battery_config_path(key: str, wave: str = "") -> str:
    """``/repo/configs/sdfbase_transfer_<slug>[<wave>].yaml`` — the battery
    config of a model key, in a WAVE namespace (``_r2`` = the round-2
    controls-in-battery namespace; '' = round 1). The slug comes
    from TRANSFER_SLUG (the one model-key -> instruct-slug table, next to
    MODELS); the wave must be a ``_r<N>`` or ``_it`` suffix (_WAVE_SUFFIX)
    so _battery_parts reads it back."""
    if key not in TRANSFER_SLUG:
        raise SystemExit(f"[battery] no transfer slug registered for {key} "
                         f"— it has no transfer config (TRANSFER_SLUG covers "
                         f"{sorted(TRANSFER_SLUG)})")
    if wave and not _WAVE_SUFFIX.fullmatch(wave):
        raise SystemExit(f"[battery] --wave must look like _r2 (an _r<N> "
                         f"round) or _it (got {wave!r})")
    return f"/repo/configs/sdfbase_transfer_{TRANSFER_SLUG[key]}{wave}.yaml"


def battery_cells_in_config(cfg_local, want_cells: list[str]) -> list[str]:
    """Refuse a --cells entry the transfer config's ``explicit_cells`` does
    not list: the battery fans out over the config's registry, and a cell
    outside it would fail in every container after the GPU came up."""
    import yaml

    listed = list(yaml.safe_load(Path(cfg_local).read_text())
                  .get("explicit_cells") or [])
    unknown = [c for c in want_cells if c not in listed]
    if unknown:
        raise SystemExit(f"[battery] --cells {unknown} not in "
                         f"{Path(cfg_local).name}'s explicit_cells "
                         f"{listed}")
    return want_cells


def battery_shard_selection(shards: str, nshards: int) -> list[int] | None:
    """``--shards`` for a REPAIR spawn: the shard indexes (within the SAME
    ``--nshards`` slicing, so every goal lands in the shard it had) to spawn
    for the sharded attacks; '' = every shard (the full wave). An index
    outside ``range(nshards)`` refuses — a wrong slicing would re-generate a
    different goal subset under the failed shard's name. Unsharded attacks
    (UNSHARDED_BATTERY_ATTACKS) have only shard 0, so a selection without 0
    spawns none of them; a selection that spawns NOTHING refuses in the
    entrypoint."""
    if not shards:
        return None
    try:
        idx = sorted({int(x) for x in shards.split(",") if x.strip()})
    except ValueError:
        raise SystemExit(f"[battery] --shards must be a comma list of shard "
                         f"indexes (got {shards!r})")
    bad = [i for i in idx if i < 0 or i >= nshards]
    if bad or not idx:
        raise SystemExit(f"[battery] --shards {shards!r} outside "
                         f"range(--nshards={nshards}): {bad or 'empty'} — a "
                         "repair spawn keeps the failed shard's slicing")
    return idx


def expand_defenses_root(defenses_root: str, key: str) -> str | None:
    """The mmlu ``--defenses-root`` per model: ``{key}`` expands to the
    model key so one flag can point every model at its own snapshot bank
    (``/vol/out/{key}_r1``); '' means the live bank (mmlu_command's
    default)."""
    return defenses_root.format(key=key) if defenses_root else None


def _design_si(design: dict) -> str:
    """The data cell a staged HPO design sweeps: its own ``si`` pin when it
    carries one (the round-2 gate_reg grid rides constv2-mix-1m), HPO_SI
    otherwise. hpo-spawn (the local copy) and _hpo_impl (the STAGED copy)
    both resolve it here, so the manifest row and the container agree by
    construction; tune_arm's run_trial refuses an --si contradicting the
    pin."""
    return str(design.get("si") or HPO_SI)


#: The design-file fields a trial summary must ECHO for fits_watch.py to
#: count it as THIS design's publication (the cells manifests'
#: expect_fingerprint, transposed: a trial's fit fingerprint hashes the
#: pool bytes only the container reads, so the launcher records the
#: design-side identity instead). config_sha rides along for provenance —
#: the summary does not carry it (tune_arm refuses any other --config before
#: training, and its fit fingerprint hashes it).
HPO_EXPECT_DESIGN_KEYS = ("model", "model_revision", "config_sha", "box",
                          "warmup_levels")


def hpo_expect_identity(design: dict, index: int, arm: str) -> dict:
    """What the published summary.json of trial ``index`` must carry to be
    THIS design's trial: the canonical model identity emit-trials stamped,
    the search it belongs to (box, warmup levels, round, position, si,
    arm) and the row's own designed point (every key of the design's trial
    row — extra axes included). Recorded on the manifest row at spawn
    time; fits_watch.py compares it to the summary on the volume."""
    trial = design["trials"][index]
    return {**{k: design.get(k) for k in HPO_EXPECT_DESIGN_KEYS},
            "arm": arm, "position": str(design.get("position") or "all"),
            "round": int(design.get("round") or 0),
            "si": _design_si(design), "trial": dict(trial)}


def config_seed(cfg: dict) -> int | None:
    """The seed a fits config declares (``train_kwargs.default.seed``), or
    None when it declares none — the frozen configs, whose trials keep
    running exactly as before (tune_arm's own default, 0). Only the
    seed-replicate configs (2026-09-15) declare one."""
    raw = ((cfg.get("train_kwargs") or {}).get("default") or {}).get("seed")
    return None if raw is None else int(raw)


def hpo_trial_cmd(*, python: str, cfg: str, trials_file: str, index: int,
                  pairs: str, out_root: str, si: str, arm: str, bs: int,
                  epochs: int, seed: int | None = None, wandb: bool = True,
                  profile: bool = False) -> list[str]:
    """The ``tune_arm.py trial`` argv ONE HPO trial runs (a pure function
    so a test can read it): the historical argv byte for byte, plus
    ``--seed <s>`` ONLY when the config declares a seed (config_seed) —
    the seed-replicate search runs at seed 1, on the seed-1 pools; every
    frozen key's argv is unchanged and its trials keep tune_arm's default
    seed 0. ``--no-wandb`` / ``--profile`` ride behind, exactly where they
    always were."""
    cmd = [
        python, "/repo/scripts/tune_arm.py", "trial",
        "--config", cfg,
        "--trials", trials_file,
        "--index", str(int(index)),
        "--pairs", pairs,
        "--out", str(out_root),
        "--si", si,
        "--arm", arm, "--positions", "all",
        "--batch-size", str(int(bs)),
        "--epochs", str(int(epochs)),
        "--fit-subsample", "0",
    ]
    if seed is not None:
        cmd += ["--seed", str(int(seed))]
    if not wandb:
        cmd.append("--no-wandb")
    if profile:
        cmd.append("--profile")  # one profiler trace per cell
    return cmd


def _hpo_impl(item: tuple) -> dict:
    """ONE CE HPO trial: (model_key, arm, index[, round]). tune_arm.py
    trial on the design's data cell — the staged trials payload's "si"
    when it pins one (the round-2 gate_reg mini-sweep rides
    constv2-mix-1m), HPO_SI (mixed-100k) otherwise — positions=all,
    effective batch 8 (the CE adapters' own CE_EFFECTIVE_BATCH), the
    campaign config's per-cell caps for THAT cell (epochs from
    train_kwargs_by_si.<si>; patience read by the adapter from the same
    table), full pool (the cell IS the budget), early stopping on the
    shared held-out pool inside every trial. Trial batch width = the
    folded memprobe width in the config (int()-crash if the fold is
    missing — the fail-loud seam). Results to
    /vol/out/hpo/<key>/<arm>_all/rN_tNN/."""
    require_wandb_entity()  # R2-D16: never log a trial to a foreign account
    import shutil
    import subprocess
    import sys
    import threading
    from pathlib import Path

    key, arm, index = item[:3]
    rnd = int(item[3]) if len(item) > 3 else 1
    spec = MODELS[key]
    # A container's view of the volume is a snapshot from its start: a WARM
    # container recycled from the previous round (scaledown_window 120 s)
    # does not see a design `volume put` after it started, and the round-2
    # spawn of 2026-09-11 failed 16/16 with FileNotFoundError on a design
    # that was on the volume. Same rule as _fit_impl: reload BEFORE any
    # /vol read.
    vol.reload()
    _ensure_model(key)

    cfg = _cfg_yaml(key)
    bs = int(cfg["batch_size_by_arm"][arm])   # MEMPROBE sentinel crashes

    cell = f"{arm}_all"
    tdir = f"r{rnd}_t{index:02d}"
    trials_file = f"{HPO_TRIALS_DIR}/r{rnd}_{key}_{arm}_all.json"
    design = json.loads(Path(trials_file).read_text())
    si = _design_si(design)
    epochs = int(cfg["train_kwargs_by_si"][si]["epochs"])
    pairs = _pool_path(key, si)

    out_root = Path(f"/root/out/hpo/{key}/{cell}")
    trial_dir = out_root / tdir
    trial_dir.mkdir(parents=True, exist_ok=True)
    seeded = Path(f"/vol/out/hpo/{key}/{cell}/{tdir}/summary.json")
    if seeded.exists():
        shutil.copy2(seeded, trial_dir / "summary.json")

    # GPU-duty spot-check once the optimizer loop is live (gpu-util skill).
    metrics_path = trial_dir / "metrics.jsonl"
    done = threading.Event()
    dmon_rows = _start_dmon(metrics_path, done)

    # The config's declared seed (None on every frozen key) reaches the
    # trial as ``--seed`` — the seed-replicate search runs at seed 1.
    cmd = hpo_trial_cmd(
        python=sys.executable, cfg=spec["cfg"], trials_file=trials_file,
        index=index, pairs=pairs, out_root=str(out_root), si=si, arm=arm,
        bs=bs, epochs=epochs, seed=config_seed(cfg),
        wandb=_HAVE_WANDB, profile=(index == 0))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd="/repo", env=_env(),
                          capture_output=True, text=True)
    wall = time.time() - t0
    done.set()

    skipped = "skipping" in proc.stdout and "MATCHING fit" in proc.stdout
    res: dict = {"model": key, "arm": arm, "index": int(index),
                 "round": rnd, "si": si, "bs": bs, "epochs": epochs,
                 "rc": int(proc.returncode),
                 "wall_s": round(float(wall), 1), "skipped": bool(skipped)}
    mean_sm = _mean_sm_pct(dmon_rows)
    if mean_sm is not None:
        res["mean_sm_pct"] = mean_sm

    spath = trial_dir / "summary.json"
    if spath.exists():
        s = json.loads(spath.read_text())
        tr = s.get("trial", {})
        res.update({
            "status": str(s.get("status")),
            "heldout_objective": s.get("heldout_objective"),
            "y": (round(float(s["best_heldout_loss"]), 6)
                  if s.get("best_heldout_loss") is not None else None),
            "lr": tr.get("lr"), "clip": tr.get("max_grad_norm"),
            "warmup": tr.get("warmup_frac"),
            "accum_steps": s.get("accum_steps"),
            "floor_ok": s.get("floor_ok"),
            "floor_ratio": (round(float(s["floor_ratio"]), 4)
                            if s.get("floor_ratio") is not None else None),
            "peak_gib": (round(float(s["peak_alloc_gib"]), 1)
                         if s.get("peak_alloc_gib") else None),
            # the seed the trial RAN under (tune_arm records it; absent
            # on summaries written before 2026-09-15 — never invented)
            **({"seed": s["seed"]} if "seed" in s else {}),
        })
        if proc.returncode != 0 and s.get("status") not in (
                "ok", "nan_diverged", "gate_dead", "floor_failed",
                "ratio_diverged"):
            res["stderr_tail"] = str(proc.stderr[-1200:])
    else:
        res["status"] = "no-summary"
        res["stderr_tail"] = str(proc.stderr[-1200:])
        print(proc.stdout[-2500:])

    spst_med = _median_sec_per_step(metrics_path)
    if spst_med is not None:
        res["sec_per_step_median"] = spst_med

    if not skipped:
        dest = Path(f"/vol/out/hpo/{key}/{cell}/{tdir}")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(trial_dir, dest, dirs_exist_ok=True)
        vol.commit()
    return res


#: The probe registry: ``--probe <ver>`` -> (frozen file stem under
#: /repo/data/sdfbase/ — ``<stem>.jsonl`` + ``<stem>.manifest.json``, the
#: scorer's sha freeze contract — and the output dir stem under
#: /vol/out/<key>/). v1 / v2 keep their historical layouts byte for byte.
#: v3g: the GENRE-MATCHED probe_v3_genre (662 items, sha e2768bb4…; both
#: continuations share the opening move, so a shift within a variant is a
#: values readout) — a SEPARATE instrument reported per variant (R / C,
#: never pooled), its rows under probe_v3/. An unregistered version
#: REFUSES: the file and the dir are one binding.
PROBES = {
    "v1": ("probe_v1", "probe"),
    "v2": ("probe_v2", "probe_v2"),
    "v3g": ("probe_v3_genre", "probe_v3"),
}


def probe_file_stem(ver: str) -> str:
    """The frozen probe file stem for a registered probe version."""
    if ver not in PROBES:
        raise SystemExit(f"[probe-score] unknown probe {ver!r} — registered: "
                         f"{sorted(PROBES)} (PROBES in sdfbase_fits.py)")
    return PROBES[ver][0]


def probe_out_dirname(ver: str, scale: float = 1.0) -> str:
    """The probe-score output dir name: the historical layouts untouched
    at scale 1.0 (``probe`` for v1, ``probe_<ver>`` after, ``probe_v3``
    for the genre-matched v3g), and a ``_x<scale>`` suffix for any off-1
    scale (round-2 2x-strength pass: scale-2 margins must never overwrite
    the scale-1 files)."""
    if ver not in PROBES:
        raise SystemExit(f"[probe-score] unknown probe {ver!r} — registered: "
                         f"{sorted(PROBES)} (PROBES in sdfbase_fits.py)")
    base = PROBES[ver][1]
    return base if float(scale) == 1.0 else f"{base}_x{float(scale):g}"


#: The identity fields probe_score.py stamps in each ``<out>.meta.json``
#: that a probe-score row echoes back to the caller, so the run's return
#: value / log says WHICH object every file measured — plus
#: ``span_version`` (the decision-span schema the file carries; None on a
#: stamp-only sidecar), so the log also says which scorer version wrote
#: it.
PROBE_STAMP_FIELDS = ("object_sha256", "fit_fingerprint", "identity_kind",
                      "stamp_version", "span_version")


def probe_stamp_fields(meta_path) -> dict:
    """The identity stamp of one probe output, read back from its
    sidecar: :data:`PROBE_STAMP_FIELDS` plus ``continuation_pending`` (the
    cell's superseded_by.json announces a continuation that has not
    published yet — the object scored is the one it will supersede) and
    ``generation`` (the continued_from chain length). A sidecar that is
    missing or pre-stamp (no ``stamp_version``) comes back as
    ``{"stamp": "missing"}`` — never a fabricated sha."""
    p = Path(meta_path)
    if not p.is_file():
        return {"stamp": "missing"}
    meta = json.loads(p.read_text(encoding="utf-8"))
    if "stamp_version" not in meta:
        return {"stamp": "missing"}
    out = {k: meta.get(k) for k in PROBE_STAMP_FIELDS}
    ident = meta.get("identity") or {}
    gen = ident.get("object_generation") or {}
    sup = ident.get("superseded_by") or {}
    out["generation"] = gen.get("generation")
    out["continuation_pending"] = bool(sup.get("continuation_pending"))
    return out


def _probe_score_impl(item: tuple) -> dict:
    """ONE model's probe scoring: (model_key, cells_csv[, probe_version
    [, scale]]). Drives the scripts/sdfbase/probe_score.py `score` CLI
    (one --object/--context per invocation) SEQUENTIALLY inside this one
    container: every fitted cell ("all" = every COMPLETE cell in the
    bank; "gates" = NO cells, the two validity-gate rows only — the gate
    (a)-v2 re-run on the instruct siblings) plus the two validity-gate
    rows (bare base, in-context constitution). Frozen probe set +
    manifest ride in /repo; per-item margins land at
    /vol/out/<key>/probe/ for v1 (the historical layout, untouched) and
    /vol/out/<key>/probe_<ver>/ for any later probe version.

    ``scale != 1.0`` (the round-2 2x-strength pass): every OBJECT row is
    applied at that strength (probe_score --scale — honest for both arm
    kinds, tests/test_scale_semantics.py), margins land in the
    _x<scale>-suffixed dir, and the two object-less gate rows are SKIPPED
    — a "bare model at 2x" is not a measurement, and re-scoring the
    scale-1 gates under a 2x label would fabricate a comparison.

    Identity: probe_score.py stamps every output with the object it
    scored (sidecar object_sha256 + fit_fingerprint + generation chain;
    object_sha256 on every row) and REFUSES a cell whose object does not
    match its published COMPLETE marker; each row returned here echoes
    that stamp (``probe_stamp_fields``) so the run log names the object
    behind every file, and flags a cell whose superseded_by.json
    announces an unpublished continuation (``continuation_pending``)."""
    import os
    import subprocess
    import sys

    key, cells = item[0], item[1]
    ver = str(item[2]) if len(item) > 2 else "v1"
    scale = float(item[3]) if len(item) > 3 else 1.0
    spec = MODELS[key]
    # Modal Volumes are snapshot-based: a WARM container reused within its
    # scaledown window keeps the snapshot it was mounted with, so without a
    # reload it could read an old-but-consistent object/COMPLETE pair and
    # stamp a superseded publication as current.
    # Reload FIRST, before any /vol read.
    vol.reload()
    _ensure_model(key)
    stem = probe_file_stem(ver)
    out_dir = Path(f"/vol/out/{key}/{probe_out_dirname(ver, scale)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = Path(f"/vol/out/{key}/cells_all")

    if cells == "gates":
        want = []
    elif not cells or cells == "all":
        want = [f"{si}__{arm}" for si in SIS for arm in CE_ARMS
                if (bank / f"{si}__{arm}" / "COMPLETE").exists()]
    else:
        want = [c for c in cells.split(",") if c]

    def run_one(tag: str, obj_path: str | None, context: str,
                batch_size: int | None = None, render: str = "raw") -> dict:
        cmd = [
            sys.executable, "/repo/scripts/sdfbase/probe_score.py", "score",
            "--probe", f"/repo/data/sdfbase/{stem}.jsonl",
            "--manifest", f"/repo/data/sdfbase/{stem}.manifest.json",
            "--model", spec["repo"], "--revision", spec["rev"],
            "--context", context, "--render", render,
            "--out", str(out_dir / f"{tag}.jsonl"),
        ]
        if batch_size is not None:
            # probe_score's own GPU note: the ~6-7k-token in-context
            # constitution prefix OOMs at the bare-row default of 8
            # (completion_logprobs casts full-sequence logits to fp32) —
            # run those rows at 1-2 on an H200.
            cmd += ["--batch-size", str(batch_size)]
        if obj_path:
            # The fitted cells carry their cell id so probe_score's
            # shared identity gate validates the artifact BEFORE loading.
            cmd += ["--object", obj_path, "--cell", tag]
            if scale != 1.0:
                cmd += ["--scale", str(scale)]
        t0 = time.time()
        p = subprocess.run(cmd, cwd="/repo", env=_env(),
                           capture_output=True, text=True)
        row = {"tag": tag, "rc": int(p.returncode),
               "wall_s": round(time.time() - t0, 1)}
        if p.returncode != 0:
            row["stderr_tail"] = str(p.stderr[-1500:])
            print(f"[probe-score] {key}/{tag} FAILED rc={p.returncode}:\n"
                  f"{p.stderr[-1500:]}", flush=True)
        else:
            # Echo the identity stamp the scorer wrote, so the run
            # output names the object behind every file (an object-less
            # gate row echoes object_sha256=None).
            row.update(probe_stamp_fields(
                str(out_dir / f"{tag}.jsonl") + ".meta.json"))
            sha = row.get("object_sha256")
            print(f"[probe-score] {key}/{tag} ok ({row['wall_s']}s) "
                  f"object_sha256={sha[:12] + '…' if sha else None} "
                  f"gen={row.get('generation')} span={row.get('span_version')}"
                  + (" CONTINUATION-PENDING" if row.get("continuation_pending")
                     else ""), flush=True)
        return row

    rows = []
    # Rendering is a per-MODEL property (probe_render): raw for the base
    # models, chat for instruct models — and it governs EVERY row of
    # a model's pass, gate rows and fitted-cell rows alike, so a fitted
    # cell's margin shift is read against a bare row in the SAME render
    # (a chat-rendered bare gate next to raw-rendered object rows would
    # confound the object effect with the render). The probe_only
    # instruct siblings have no cells, so before GLM-4.6V-Flash (an
    # instruct model WITH fits) the object rows were only ever raw.
    render = spec.get("probe_render", "raw")
    # Validity-gate rows first. The
    # constitution-context row runs at batch 2: its prefix is ~6-7k
    # tokens and the default 8 (sized for bare rows) OOMs per
    # probe_score.py's own GPU note. An off-1 scale pass skips them: both
    # rows are object-less.
    if scale == 1.0:
        rows.append(run_one("bare", None, "bare", render=render))
        rows.append(run_one("constitution", None, "constitution",
                            batch_size=2, render=render))
    else:
        print(f"[probe-score] {key}: scale={scale:g} pass — gate rows "
              "skipped (object-less)", flush=True)
    for cell in want:
        # artifacts only: PSR cells ship an object.pt.provenance.json SIDECAR
        # beside object.pt, which must never read as a second artifact.
        cand = sorted(q for q in (bank / cell).glob("object.*") if q.suffix != ".json")
        if len(cand) != 1:
            rows.append({"tag": cell, "rc": -1,
                         "error": f"no unambiguous artifact in "
                                  f"{bank / cell} ({[c.name for c in cand]})"})
            continue
        rows.append(run_one(cell, str(cand[0]), "bare", render=render))
        vol.commit()
    vol.commit()
    bad = [r for r in rows if r["rc"] != 0]
    return {"model": key, "cells": [r["tag"] for r in rows],
            "status": "ok" if not bad else "failed",
            "n_ok": len(rows) - len(bad), "n_bad": len(bad), "rows": rows}


def mmlu_command(key: str, cells: str, run_label: str,
                 defenses_root: str | None = None,
                 scale: float = 1.0) -> list[str]:
    """The base_mmlu.py argv for ONE model's run — run-scoped by
    contract: ``--run-label`` (REQUIRED by the script) namespaces every
    output under /vol/out/<key>/mmlu/<label>/ with the label in each
    filename, and ``--defenses-root`` selects the bank the objects come
    from (the live bank by default; the ``<key>_r1`` snapshot bank for
    an r1 run), so snapshot and refit outputs never share a path.
    ``scale`` (2026-09-03, the 2x-dose pass): appended as ``--scale`` only
    when off 1.0, so every scale-1 command stays byte-identical to the
    r1/r2/r3 runs already published. Pure command construction (testable
    off-Modal)."""
    import sys

    if not run_label:
        raise SystemExit("[mmlu] run_label is required (r1 for the snapshot "
                         "bank, r2 for the refit bank)")
    cmd = [
        sys.executable, "/repo/scripts/sdfbase/base_mmlu.py",
        "--config", MODELS[key]["cfg"], "--cells", cells,
        "--defenses-root", defenses_root or f"/vol/out/{key}",
        "--n", "500", "--seed", "0",
        "--out-dir", f"/vol/out/{key}/mmlu", "--run-label", run_label,
    ]
    if float(scale) != 1.0:
        cmd += ["--scale", str(float(scale))]
    return cmd


def _mmlu_impl(item: tuple) -> dict:
    """ONE model's base-capability leg: (model_key, cells_csv, run_label
    [, defenses_root[, scale]]) — plain-text 5-shot MMLU letter-logprob
    (scripts/sdfbase/base_mmlu.py) at winner cells + 1M cells + bare
    base, one container per model, outputs run-scoped (mmlu_command).
    ``scale`` is the object strength (the r2_x2 pass = 2.0)."""
    import subprocess

    key, cells, run_label = item[:3]
    defenses_root = str(item[3]) if len(item) > 3 and item[3] else None
    scale = float(item[4]) if len(item) > 4 else 1.0
    _ensure_model(key)
    cmd = mmlu_command(key, cells, run_label, defenses_root, scale)
    t0 = time.time()
    p = subprocess.run(cmd, cwd="/repo", env=_env(), capture_output=True,
                       text=True)
    out = {"model": key, "cells": cells, "run_label": run_label,
           "defenses_root": defenses_root or f"/vol/out/{key}",
           "scale": scale,
           "rc": int(p.returncode),
           "wall_s": round(time.time() - t0, 1),
           "tail": p.stdout.splitlines()[-15:] if p.stdout else []}
    if p.returncode != 0:
        out["status"] = "failed"
        out["stderr_tail"] = str(p.stderr[-2000:])
    else:
        out["status"] = "ok"
        vol.commit()
    return out


def gsm8k_command(key: str, cells: str, run_label: str,
                  defenses_root: str | None = None,
                  scale: float = 1.0, batch_size: int = 0,
                  decode_parity: str | None = None) -> list[str]:
    """The base_gsm8k.py argv for ONE model's run — mmlu_command's twin:
    ``--run-label`` REQUIRED (dir + filename), ``--defenses-root`` picks
    the bank (live by default), n=500 seed 0, outputs under
    /vol/out/<key>/gsm8k/<label>/. ``--scale`` is ALWAYS passed (a new
    surface; no published byte-identity to preserve) so the run identity's
    scale is the launcher's, never a script default. ``batch_size`` > 0
    passes ``--batch-size`` (0 = the script's default): the Mamba2-hybrid
    Nemotron ran its labels at 128 (its per-step cost was the cache-free
    recompute — fixed, but the batch size is part of the run identity, so
    its labels stay at 128 and BOTH labels of a model must use the same
    value to pair). ``decode_parity`` passes ``--decode-parity`` (a volume
    path): the attestation base_gsm8k needs to add cached cells to a label
    that holds cells published before the decode-path fix. Pure command
    construction (testable off-Modal)."""
    import sys

    if not run_label:
        raise SystemExit("[gsm8k] run_label is required (r2 for the 1x pass "
                         "on the live bank, r2_x2 for the 2x pass)")
    cmd = [
        sys.executable, "/repo/scripts/sdfbase/base_gsm8k.py",
        "--config", MODELS[key]["cfg"], "--cells", cells,
        "--defenses-root", defenses_root or f"/vol/out/{key}",
        "--n", "500", "--seed", "0",
        "--out-dir", f"/vol/out/{key}/gsm8k", "--run-label", run_label,
        "--scale", str(float(scale)),
    ]
    if int(batch_size) > 0:
        cmd += ["--batch-size", str(int(batch_size))]
    if decode_parity:
        cmd += ["--decode-parity", str(decode_parity)]
    return cmd


def _gsm8k_impl(item: tuple) -> dict:
    """ONE model's GSM8K leg: (model_key, cells_csv, run_label
    [, defenses_root[, scale]]) — plain-text 8-shot CoT GSM8K, greedy
    (scripts/sdfbase/base_gsm8k.py), every requested cell + bare base in
    one container (model loads once, objects swap), outputs run-scoped
    (gsm8k_command). The script's summary is written only after the full
    requested set scored, so ONE container per (model, label) — a second
    container writing the same label's summary would race it."""
    import subprocess

    key, cells, run_label = item[:3]
    defenses_root = str(item[3]) if len(item) > 3 and item[3] else None
    scale = float(item[4]) if len(item) > 4 else 1.0
    batch_size = int(item[5]) if len(item) > 5 and item[5] else 0
    decode_parity = str(item[6]) if len(item) > 6 and item[6] else None
    _ensure_model(key)
    cmd = gsm8k_command(key, cells, run_label, defenses_root, scale,
                        batch_size, decode_parity)
    t0 = time.time()
    p = subprocess.run(cmd, cwd="/repo", env=_env(), capture_output=True,
                       text=True)
    out = {"model": key, "cells": cells, "run_label": run_label,
           "defenses_root": defenses_root or f"/vol/out/{key}",
           "scale": scale, "batch_size": batch_size or None,
           "decode_parity": decode_parity,
           "rc": int(p.returncode),
           "wall_s": round(time.time() - t0, 1),
           "tail": p.stdout.splitlines()[-20:] if p.stdout else []}
    if p.returncode != 0:
        out["status"] = "failed"
        out["stderr_tail"] = str(p.stderr[-3000:])
    else:
        out["status"] = "ok"
        vol.commit()
    return out


def ce_command(key: str, obj_path: str | None, tag: str, pool_tag: str,
               pool_path: str, *, scale: float, python: str = "python"
               ) -> tuple[list[str], str]:
    """``ce_score.py`` on one pool for one STAGED cell (or the bare model,
    ``obj_path=None`` / tag ``bare``) at ``scale`` — the derived app's
    ce_command, byte for byte in flags and output path
    (``CE_ROOT/<key>/<tag>__<pool>__scale<s>.json``, ``scale`` rendered
    ``:g`` so 1.0 -> ``scale1``), so the analysis reads a fitted cell's
    1x/2x CE exactly as it reads a derived object's. The object goes
    through ce_score's identity gate with its cell id (a staged cell's
    sidecar must name it) and the bank root as parents_root."""
    spec = MODELS[key]
    out = f"{CE_ROOT}/{key}/{tag}__{pool_tag}__scale{float(scale):g}.json"
    cmd = [python, "/repo/scripts/sdfbase/ce_score.py",
           "--model", spec["repo"], "--revision", spec["rev"],
           "--pairs", pool_path, "--pool", pool_tag,
           "--out", out, "--scale", str(float(scale)),
           "--batch-size", "1", "--max-len", "8192"]
    if obj_path is not None:
        cmd += ["--object", obj_path, "--cell", tag,
                "--parents-root", PARENTS_ROOT]
    return cmd, out


def parse_scales(scales: str) -> list[float]:
    """``"1.0,2.0"`` -> ``[1.0, 2.0]``; refuses an empty list, a
    non-positive scale, and duplicates (each scale is its own output
    file, so a duplicate would score twice and say nothing)."""
    out: list[float] = []
    for s in [x for x in str(scales).split(",") if x.strip()]:
        v = float(s)
        if v <= 0:
            raise SystemExit(f"[ce] --scales entries must be > 0, got {v}")
        if v in out:
            raise SystemExit(f"[ce] --scales lists {v:g} twice")
        out.append(v)
    if not out:
        raise SystemExit("[ce] --scales is empty (e.g. 1.0,2.0)")
    return out


def _ce_impl(item: tuple) -> dict:
    """ONE model's held-out CE leg: (model_key, cells_csv, scales_csv
    [, pools_csv]) — ce_score.py per (cell x scale x pool) SEQUENTIALLY in
    one container, artifacts discovered in the LIVE bank (sidecar-aware:
    object.pt.provenance.json is not an artifact), each scored file
    committed as it lands. ``pools_csv`` (``tag:path,...``) overrides
    CE_POOLS; a pool whose file is not staged is reported and skipped,
    never silently substituted."""
    import subprocess
    import sys

    key, cells_csv, scales_csv = item[:3]
    pools_csv = str(item[3]) if len(item) > 3 and item[3] else ""
    _ensure_model(key)
    scales = parse_scales(scales_csv)
    if pools_csv:
        pools = dict(spec.split(":", 1) for spec in pools_csv.split(","))
    else:
        pools = dict(CE_POOLS)
    pools = {t: p.format(key=_pool_key(key)) for t, p in pools.items()}
    rows = []
    for tag, path in list(pools.items()):
        if not Path(path).is_file():
            rows.append({"tag": f"pool/{tag}", "rc": -1,
                         "error": f"{path} not staged — CE leg unavailable"})
            print(f"[ce] {key}: pool {tag} UNAVAILABLE ({path})", flush=True)
            pools.pop(tag)
    bank = Path(f"/vol/out/{key}/cells_all")
    cells = [c for c in cells_csv.split(",") if c]

    def run_one(tag: str, cmd: list[str], out_path: str) -> dict:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        p = subprocess.run(cmd, cwd="/repo", env=_env(),
                           capture_output=True, text=True)
        row = {"tag": tag, "rc": int(p.returncode), "out": out_path,
               "wall_s": round(time.time() - t0, 1)}
        if p.returncode != 0:
            row["stderr_tail"] = str(p.stderr[-2000:])
            print(f"[ce] {key}/{tag} FAILED rc={p.returncode}:\n"
                  f"{p.stderr[-2000:]}", flush=True)
        else:
            print(f"[ce] {key}/{tag} ok ({row['wall_s']}s) "
                  f"{(p.stdout or '').strip().splitlines()[-1:]}", flush=True)
        return row

    for cell in cells:
        if cell == "bare":
            obj_path = None
        else:
            cand = sorted(q for q in (bank / cell).glob("object.*")
                          if q.suffix != ".json")
            if len(cand) != 1:
                rows.append({"tag": f"ce/{cell}", "rc": -1,
                             "error": f"no unambiguous artifact for {cell} "
                                      f"under {bank / cell} "
                                      f"({[c.name for c in cand]})"})
                continue
            obj_path = str(cand[0])
        for s in scales:
            if obj_path is None and s != 1.0:
                continue  # the bare model has no scale
            for ptag, ppath in pools.items():
                cmd, out_path = ce_command(key, obj_path, cell, ptag, ppath,
                                           scale=s, python=sys.executable)
                rows.append(run_one(f"ce/{cell}/{ptag}/x{s:g}", cmd, out_path))
                vol.commit()
    vol.commit()
    bad = [r for r in rows if r["rc"] != 0]
    return {"model": key, "cells": cells, "scales": scales,
            "pools": sorted(pools),
            "status": "ok" if not bad else "failed",
            "n_ok": len(rows) - len(bad), "n_bad": len(bad), "rows": rows}


#: A transfer config's WAVE suffix: ``_r<N>`` (``_r2`` = the round-2
#: controls-in-battery namespace) or ``_it`` (the
#: instruct-direct wave — the `_it_doc` keys' objects, trained ON the
#: instruct checkpoint, applied to it directly, plus any base-fit object
#: whose battery row needs a same-wave re-run). The slug keys the OBJECT
#: bank, which waves may share (objects are identities, not wave state);
#: slug + wave keys the run namespace, matching the config's own
#: out_dir/private_run. No slug ends in a wave suffix (_battery_parts
#: splits on the LAST match), so the grammar stays unambiguous: add a
#: wave here AND in BATTERY_BANK_SUFFIX, never one alone.
_WAVE_SUFFIX = re.compile(r"(?:_r\d+|_it)$")

#: Wave -> the transfer OBJECT BANK it serves from, as a suffix on
#: ``/vol/out/transfer_<slug>``. Rounds 1 and ``_r2`` share the original
#: bank (``""``): objects are identities, and those two waves served the
#: same staged bytes. ``_r3`` serves the CURRENT 1M base-fit objects,
#: which the shared bank does NOT hold — it still carries the round-1
#: fits of both 1M cells, and a staged object is never overwritten — so
#: ``_r3`` gets a bank of its own
#: (``/vol/out/transfer_<slug>_r3/cells``, staged from the fitting bank's
#: cells_all with sha256 verification). ``_it`` likewise gets its own
#: bank, ``/vol/out/transfer_<slug>_it/cells``: it holds each `_it_doc`
#: key's objects copied from THAT key's fitting bank
#: (``/vol/out/<slug>_it_doc/cells_all`` — the object's fit model IS the
#: served instruct model, so staging is a verified copy, no sibling
#: mapping) and, for gemma, the base-fit 1M LoRA's NEW object (staged
#: base -> instruct as the `_r3` bank was); no other bank holds that set.
#: A wave missing from this table refuses loudly: the launcher never
#: guesses which objects a wave means.
BATTERY_BANK_SUFFIX = {"": "", "_r2": "", "_r3": "_r3", "_it": "_it"}


def check_battery_wave(key: str, wave: str) -> None:
    """Refuse a (model key, --wave) pair the battery must not launch —
    BEFORE the config is resolved, so the message names the lane, not a
    missing file. Two rules, one table (``MODELS[key]["battery_wave"]``):

    * a key that DECLARES a wave (the instruct-direct `_it_doc` keys)
      runs only under it — every other wave's transfer config for that
      slug names a BASE-fit bank, so a manifest labelled with this key
      would have served another lane's objects
    * a wave some key declares is that key's namespace: a key that does
      NOT declare it is refused there. The `_it` bank does carry a
      base-fit object (gemma's re-run 1M LoRA), but it rides in the
      `_it_doc` key's launch as one more cell of the wave's config —
      launching the same wave under the base key would write a second
      manifest (battery_calls_<base key>_it.json) for the same namespace
      and label the instruct-direct objects as the base lane's.

    Keys declaring nothing (the base keys among them) keep every
    ``_r<N>`` wave open, as before."""
    want_wave = MODELS[key].get("battery_wave")
    if want_wave and wave != want_wave:
        raise SystemExit(
            f"[battery] --model {key} runs only under --wave "
            f"{want_wave!r} (got {wave!r}): every other wave's "
            f"transfer config for slug {TRANSFER_SLUG.get(key)!r} "
            "serves the base-fit bank, not this lane's objects")
    owners = sorted(k for k, s in MODELS.items()
                    if s.get("battery_wave") == wave)
    if owners and key not in owners:
        raise SystemExit(
            f"[battery] --wave {wave!r} is the namespace of {owners}: "
            f"launch it with one of those --model keys, not {key!r} — "
            "every cell the wave's config lists (a base-fit object staged "
            "into its bank included) rides in that one launch")


def battery_bank(slug: str, wave: str) -> str:
    """``/vol/out/transfer_<slug><bank suffix>/cells`` — the ONE derivation
    of the object bank a battery wave serves from (BATTERY_BANK_SUFFIX);
    the config's relative ``defenses_dir`` must spell the same dir
    (checked at launch by the battery entrypoint: `battery_bank_matches`)."""
    if wave not in BATTERY_BANK_SUFFIX:
        raise SystemExit(f"[battery] wave {wave!r} has no object bank in "
                         f"BATTERY_BANK_SUFFIX {sorted(BATTERY_BANK_SUFFIX)} "
                         "— register which bank the wave serves from")
    return f"/vol/out/transfer_{slug}{BATTERY_BANK_SUFFIX[wave]}/cells"


def battery_bank_matches(cfg_local, cfg_path: str) -> str:
    """The bank battery_commands will export for ``cfg_path`` — after
    checking the committed config's ``defenses_dir`` names THE SAME dir
    (``out/transfer_<slug><suffix>/cells``, volume-relative). A mismatch
    refuses: the container would serve one bank while the config (and the
    grading host aggregate that resolves the relative default) names another."""
    import yaml

    slug, wave = _battery_parts(cfg_path)
    bank = battery_bank(slug, wave)
    cfg = yaml.safe_load(Path(cfg_local).read_text())
    want = bank[len("/vol/"):]
    got = str(cfg.get("defenses_dir"))
    if got != want:
        raise SystemExit(f"[battery] {cfg_local}: defenses_dir {got!r} != the "
                         f"bank this wave serves from ({want!r}, "
                         "BATTERY_BANK_SUFFIX) — the config and the launcher "
                         "must name one bank")
    return bank


def _battery_parts(cfg_path: str) -> tuple[str, str]:
    """(slug, wave) of a battery config path
    (``.../sdfbase_transfer_<slug>[<wave>].yaml``) — the ONE derivation
    battery_commands keys every volume path on."""
    stem = Path(cfg_path).stem
    prefix = "sdfbase_transfer_"
    if not stem.startswith(prefix):
        raise SystemExit(f"[battery] config {cfg_path} is not a "
                         f"{prefix}<slug>[_rN|_it].yaml transfer config")
    rest = stem[len(prefix):]
    m = _WAVE_SUFFIX.search(rest)
    return (rest[:m.start()], m.group(0)) if m else (rest, "")


def _battery_slug(cfg_path: str) -> str:
    """The transfer slug (bank key) of a battery config path."""
    return _battery_parts(cfg_path)[0]


def battery_commands(cfg_path: str, attack: str, cell: str, shard: int,
                     nshards: int, confirmatory: bool
                     ) -> tuple[list[str], dict[str, str]]:
    """(argv, extra-env) for ONE battery shard — every INPUT and OUTPUT
    data path VOLUME-BACKED, because a battery container's local disk
    (/root, /repo/out) dies with the container:

    - ``TT_TWO_SI_DEFENSES_DIR=/vol/out/transfer_<slug>[<bank>]/cells`` —
      the per-model transfer object bank the configs' relative
      ``defenses_dir`` promises the container will export
      (resolve_defenses_dir order: env beats the relative default); the
      bank is the wave's (battery_bank / BATTERY_BANK_SUFFIX: rounds 1 and
      _r2 share the original bank, _r3 and _it each have their own);
    - two_si_run generate gets ``--private-root /vol/private`` — its
      default private root falls back to a container-local dir on Modal,
      so transcripts would die with the container; the run's own
      namespace under it comes from the config's campaign-scoped
      ``private_run``;
    - cell_controls_run generate gets an explicit ``--defenses-dir``
      (same bank; the flag wins over env by that script's own resolve
      order — belt and braces, one value) and a /vol ``--out-root``
      (its flag default is repo-relative ``out/two_si/controls`` ==
      ``/repo/out/...`` in-container: container-local, lost);
    - the ``xstest`` benign pass rides the SAME volume private root as the
      attack shards, because that is where ``_aggregate_xstest`` looks.

    Pure command construction — no Modal objects, no filesystem — so the
    path discipline is testable off-Modal."""
    import sys

    slug, wave = _battery_parts(cfg_path)
    ddir = battery_bank(slug, wave)
    env_extra = {"TT_TWO_SI_DEFENSES_DIR": ddir,
                 # renellm REPLAYS a frozen corpus that
                 # renellm._corpus_path() resolves under this env var,
                 # defaulting to a local state path that exists only on
                 # the grading host. That path does not exist in a
                 # container, so _corpus() returned None, run_renellm took
                 # its loud_skip, and the shard exited 0 having written
                 # NOTHING — vacuously green, and readable as an ASR of
                 # zero. Volume-backed like every other battery input;
                 # the corpus is staged there before the run and verified
                 # by content hash.
                 "SISTEER_HELDOUT_CORPUS_DIR": HELDOUT_CORPUS_DIR}
    if attack == "controls":
        cmd = [sys.executable, "/repo/scripts/cell_controls_run.py",
               "generate", "--config", cfg_path, "--cell", cell,
               "--controls", "mmlu,gsm8k",
               "--defenses-dir", ddir,
               # the run namespace carries the wave, like the config's
               # own out_dir (out/sdfbase_transfer_<slug><wave>)
               "--out-root",
               f"/vol/out/sdfbase_transfer_{slug}{wave}/controls"]
    elif attack == "xstest":
        # the benign over-refusal pass, which the battery never
        # emitted — so `two_si_run.py aggregate` refused every cell before
        # grading (jailbreak_run._aggregate_xstest raises without an
        # xstest_responses.jsonl). `generate --xstest` replaces the attack
        # sweep with one greedy generation over the 250 SAFE XSTest prompts
        # (the 200 unsafe contrast rows of the 450-row file are never
        # used), appended to <private-root>/<private_run>/<cell>/
        # xstest_responses.jsonl — exactly the path the aggregate reads.
        #
        # No --attack: the pass is attack-INDEPENDENT (cmd_generate returns
        # before any attack dispatch), so naming one would imply a scope it
        # does not have. No --shard/--nshards/--confirmatory either: none of
        # them is read on this path (no goal slicing, no eval-goal load) and
        # a second shard would race one volume file — see
        # UNSHARDED_BATTERY_ATTACKS.
        cmd = [sys.executable, "/repo/scripts/two_si_run.py", "generate",
               "--config", cfg_path, "--cell", cell, "--xstest",
               "--private-root", "/vol/private"]
    else:
        cmd = [sys.executable, "/repo/scripts/two_si_run.py", "generate",
               "--config", cfg_path, "--attack", attack, "--cell", cell,
               "--shard", str(shard), "--nshards", str(nshards),
               "--private-root", "/vol/private"]
        if confirmatory:
            cmd.append("--confirmatory")
    return cmd, env_extra


def _battery_impl(item: tuple) -> dict:
    """ONE instruct-transfer battery shard: (cfg_path, attack, cell,
    shard, nshards, confirmatory). attack == "controls" runs the
    capability leg (cell_controls_run.py: s2parity mmlu_letter + gsm8k)
    and attack == "xstest" the benign over-refusal pass, each instead of
    an attack sweep. Both scripts consume the transfer config
    (configs/sdfbase_transfer_<model>.yaml — the explicit-cell registry,
    agent E's seam); this function just shells out, with every data path
    volume-backed (battery_commands)."""
    import subprocess

    cfg_path, attack, cell, shard, nshards, confirmatory = item
    # The transfer config pins the INSTRUCT sibling; its snapshot is
    # staged by the instruct campaign on the same volume (HF_HOME).
    cmd, env_extra = battery_commands(cfg_path, attack, cell, shard,
                                      nshards, confirmatory)
    t0 = time.time()
    p = subprocess.run(cmd, cwd="/repo", env=_env(env_extra),
                       capture_output=True, text=True)
    out = {"config": cfg_path, "attack": attack, "cell": cell,
           "shard": shard, "rc": int(p.returncode),
           "wall_s": round(time.time() - t0, 1),
           "tail": p.stdout.splitlines()[-10:] if p.stdout else []}
    if p.returncode != 0:
        out["status"] = "failed"
        out["stderr_tail"] = str(p.stderr[-2000:])
    else:
        out["status"] = "ok"
        vol.commit()
    return out


# --- per-(image, gpu) function bindings ------------------------------------
# Modal functions bind image/gpu statically, so each model family gets its
# own thin wrapper around the shared implementations above.

@app.function(image=image_std, cpu=8, memory=32768, timeout=14400, **_COMMON)
def stage_std(key: str) -> dict:
    _ensure_model(key)
    return {"model": key, "staged": True}


# CPU-only bookkeeping against the volume (no GPU, no model load): the
# expected-fingerprint derivation fits-spawn records per manifest row, and
# the pre-refit snapshot of superseded cells. image_std carries the driver
# and its deps (the fingerprint resolves the shared held-out pool's hash
# through the driver's own functions).
@app.function(image=image_std, cpu=4, memory=16384, timeout=3600, **_COMMON)
def fingerprint_cpu(item: tuple) -> dict:
    return _fingerprint_impl(item)


@app.function(image=image_std, cpu=4, memory=16384, timeout=7200, **_COMMON)
def snapshot_cpu(item: tuple) -> dict:
    return _snapshot_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=7200, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def onboard_h200_std(key: str) -> dict:
    return _onboard_impl(key)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=7200, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def onboard_h200_qwen(key: str) -> dict:
    return _onboard_impl(key)


@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=7200, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def onboard_h200_nemotron(key: str) -> dict:
    return _onboard_impl(key)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=10800, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def memprobe_h200_std(item: tuple) -> dict:
    return _memprobe_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=10800, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def memprobe_h200_qwen(item: tuple) -> dict:
    return _memprobe_impl(item)


@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=10800, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def memprobe_h200_nemotron(item: tuple) -> dict:
    return _memprobe_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=10800, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def bench_h200_std(item: tuple) -> dict:
    return _bench_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=10800, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def bench_h200_qwen(item: tuple) -> dict:
    return _bench_impl(item)


@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=10800, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def bench_h200_nemotron(item: tuple) -> dict:
    return _bench_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=FIT_TIMEOUT_S, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def fit_h200_std(item: tuple) -> dict:
    return _fit_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=FIT_TIMEOUT_S, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def fit_h200_qwen(item: tuple) -> dict:
    return _fit_impl(item)


@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=FIT_TIMEOUT_S, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def fit_h200_nemotron(item: tuple) -> dict:
    return _fit_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=86400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def hpo_h200_std(item: tuple) -> dict:
    return _hpo_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=86400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def hpo_h200_qwen(item: tuple) -> dict:
    return _hpo_impl(item)


@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=86400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def hpo_h200_nemotron(item: tuple) -> dict:
    return _hpo_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=14400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def score_h200_std(item: tuple) -> dict:
    return _probe_score_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=14400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def score_h200_qwen(item: tuple) -> dict:
    return _probe_score_impl(item)


@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=14400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def score_h200_nemotron(item: tuple) -> dict:
    return _probe_score_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=14400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def mmlu_h200_std(item: tuple) -> dict:
    return _mmlu_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=14400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def mmlu_h200_qwen(item: tuple) -> dict:
    return _mmlu_impl(item)


@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=14400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def mmlu_h200_nemotron(item: tuple) -> dict:
    return _mmlu_impl(item)


# GSM8K is GENERATION (up to 512 new tokens x 500 questions x ~15 cells per
# container), so its budget is 6 h — the 4 h MMLU letter-logprob budget
# would cut a slow-decoding model (the Mamba2 hybrid, the 10B GLM) mid-run.
@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=21600, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def gsm8k_h200_std(item: tuple) -> dict:
    return _gsm8k_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=21600, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def gsm8k_h200_qwen(item: tuple) -> dict:
    return _gsm8k_impl(item)


# Nemotron's HF Mamba2-hybrid decode is kernel-launch-bound (~0.4 s/step at
# batch 32; a 500-question cell ran ~1 h in the smoke): 12 h, and the
# launcher passes --batch-size 128 for it (gsm8k_command).
@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=43200, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def gsm8k_h200_nemotron(item: tuple) -> dict:
    return _gsm8k_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=14400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def ce_h200_std(item: tuple) -> dict:
    return _ce_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=14400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def ce_h200_qwen(item: tuple) -> dict:
    return _ce_impl(item)


@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=14400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def ce_h200_nemotron(item: tuple) -> dict:
    return _ce_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=86400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def battery_h200_std(item: tuple) -> dict:
    return _battery_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=86400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def battery_h200_qwen(item: tuple) -> dict:
    return _battery_impl(item)


# Bound now, though the battery entrypoint still refuses the nemotron key
# (no transfer config yet): the deployed-function name
# f"battery_h200_{_variant(key)}" must resolve the day one lands.
@app.function(gpu="H200", image=image_nemotron, cpu=8, memory=131072,
              timeout=86400, max_containers=MAX_CONTAINERS,
              scaledown_window=120, **_COMMON)
def battery_h200_nemotron(item: tuple) -> dict:
    return _battery_impl(item)


def _variant(key: str) -> str:
    """Which image variant serves this model key — the ONE spelling the
    deployed-function names carry (deployed_fn appends it to a family stem)."""
    if key in NEMOTRON_IMAGE_KEYS:
        return "nemotron"
    return "qwen" if key in QWEN_IMAGE_KEYS else "std"


def deployed_fn(deployed: str, name: str, key: str = ""):
    """The ONE handle every spawn/remote in the entrypoint goes through:
    function ``name`` on the DEPLOYED app ``deployed`` — resolve_app_name's
    answer — via modal.Function.from_name. Never a locally decorated
    function object: those bind to the App object, which is built at import
    from $SDFBASE_APP_NAME alone, so a mode that spawned them ignored
    --app-name and ran under the ephemeral app of the import-time name
    (stage / onboard / memprobe / probe-score / mmlu did exactly that until
    this helper). With ``key``, ``name`` is a GPU family stem ("fit_h200")
    and the image-variant suffix is routed from the key by _variant
    ("fit_h200_nemotron") — the same spelling the bindings above declare."""
    if key:
        name = f"{name}_{_variant(key)}"
    return modal.Function.from_name(deployed, name)


def _flush_manifest(doc: dict, mpath: Path) -> None:
    from datetime import datetime, timezone

    doc["written_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    tmp = mpath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(mpath)


#: results-registry native emission (Modal): the LOCAL launcher registers each spawned
#: call as a queued run when ``$SDFBASE_REGISTRY_PLANS`` names a JSON file ``{"<key>": <planned run>}``
#: (or ``{"<mode>:<key>": ...}``) — the grading host is the launcher host, so the records land in its own
#: spool and are ingested directly. Unset: nothing happens. NOTHING in the container changes in this
#: PR: the in-container ``running`` state (``RecordWriter(spool_root="/vol/registry")`` at function
#: entry + the ``/vol/registry/pushlog/<call_id>.log`` the grading host pull timer diffs) lands with the
#: Modal install, so no deployed function's behaviour moves here.
REGISTRY_PLANS_ENV = "SDFBASE_REGISTRY_PLANS"


def registry_queued(key: str, call, *, mode: str, app_name: str) -> None:
    plans_path = os.environ.get(REGISTRY_PLANS_ENV)
    if not plans_path:
        return
    try:
        from sisteer.registry import hooks as _rh
        with open(plans_path, "r", encoding="utf-8") as fh:
            plans = json.load(fh)
        plan = plans.get(f"{mode}:{key}") or plans.get(key)
        if not isinstance(plan, dict):
            _rh.notice(f"sdfbase_fits: no planned run for {mode}:{key} in {plans_path}; spawn not registered")
            return
        _rh.queued_modal(plan, modal_app_id=str(app_name), modal_call_id=getattr(call, "object_id", None),
                         script=f"scripts/modal/sdfbase_fits.py {mode}")
    except Exception as e:  # noqa: BLE001 — a lost record never costs a launch
        print(f"[registry] sdfbase_fits: spawn of {key} not registered ({type(e).__name__}: {e})", flush=True)


@app.local_entrypoint()
def main(mode: str = "stage", model: str = "", sis: str = "", arms: str = "",
         cells: str = "", attacks: str = "", nshards: int = 1,
         confirmatory: bool = True, overwrite: bool = False,
         out_root: str = "", checkpoint_every: int = 0, manifest: str = "",
         round_no: int = 1, poll: int = 60, max_hours: float = 48.0,
         probe: str = "v1", scale: float = 1.0, retrain_all: bool = False,
         run_label: str = "", defenses_root: str = "", wave: str = "",
         verify_only: bool = False, reason: str = "",
         respawn: bool = True, max_successors: int = 3,
         continue_cap: int = 0, bank_suffix: str = "",
         app_name: str = "", continue_lr_mult: float = 1.0, scales: str = "",
         ce_pools: str = "", batch_size: int = 0, loss_impl: str = "",
         grad_ckpt: str = "", steps: int = 0, accum: int = 0,
         check_grads: bool = False, tag: str = "", bench_out: str = "",
         shards: str = "", decode_parity: str = "",
         lr: str = "", max_grad_norm: str = "", warmup_frac: str = ""):
    # The default key set NEVER includes the probe_only instruct siblings
    # (no fits config / pools / bank) — they run only when named explicitly
    # with --model, and only probe-score-shaped modes can use them. Nor
    # the explicit_only entries (GLM-4.6V-Flash while its folds are
    # pending): a full campaign member, every mode valid, but named with
    # --model rather than swept into a peer's bare invocation.
    keys = ([m for m in model.split(",") if m]
            or sorted(k for k, s in MODELS.items()
                      if not s.get("probe_only")
                      and not s.get("explicit_only")))
    for k in keys:
        if k not in MODELS:
            raise SystemExit(f"--model must be one of {sorted(MODELS)}")
        if MODELS[k].get("probe_only") and mode not in ("stage", "probe-score"):
            raise SystemExit(
                f"--model {k} is probe-score-only (instruct sibling for the "
                f"D18 gate re-run) — not valid for --mode {mode}")
    # The deployed app every spawn / lookup below targets and every
    # manifest doc + row records: --app-name > $SDFBASE_APP_NAME >
    # DEFAULT_APP_NAME. One binding, so no mode can fall back to a literal
    # — and every spawn/remote below goes through the deployed_fn helper
    # on this binding, never a locally decorated function object (those
    # bind to the import-time App and ignore the flag).
    deployed = resolve_app_name(app_name)

    def _fanout_models(tag, family, arg_of, fail_msg="model(s) failed"):
        """Per-model spawn -> gather -> fail-loud, shared by the one-call-
        per-model modes (onboard / probe-score / mmlu). ``family`` is the
        GPU function stem ("onboard_h200"); deployed_fn routes the image
        variant per key and binds the DEPLOYED app."""
        calls = [(k, deployed_fn(deployed, family, key=k).spawn(arg_of(k)))
                 for k in keys]
        for k, fc in calls:
            registry_queued(k, fc, mode=mode, app_name=deployed)
        bad = 0
        for k, c in calls:
            r = c.get()
            print(f"[{tag}] {k}: {json.dumps(r, indent=1)}")
            bad += r.get("status") != "ok"
        if bad:
            raise SystemExit(f"[{tag}] {bad} {fail_msg}")

    if mode == "stage":
        calls = [(k, deployed_fn(deployed, "stage_std").spawn(k))
                 for k in keys]
        for k, c in calls:
            print(f"[stage] {k}: {json.dumps(c.get())}")
        return

    if mode == "onboard":
        _fanout_models("onboard", "onboard_h200", lambda k: k,
                       "model(s) FAILED the smoke")
        return

    if mode == "memprobe":
        # --loss-impl chunked_ce / --grad-ckpt on|off re-probe the
        # trainer variant; the artifact lands under a suffixed name
        # (memprobe_variant_suffix) so the folded probe is never touched.
        want_arms = [a for a in arms.split(",") if a] or CE_ARMS
        suffix = memprobe_variant_suffix(loss_impl, grad_ckpt)
        calls = [(k, a, deployed_fn(deployed, "memprobe_h200",
                                    key=k).spawn((k, a, loss_impl, grad_ckpt)))
                 for k in keys for a in want_arms]
        table: dict = {}
        for k, a, c in calls:
            r = c.get()
            print(f"[memprobe{suffix}] {k}/{a}: best_bs={r.get('best_bs')} "
                  + json.dumps(r.get("ladder", {})))
            table.setdefault(k, {})[a] = r.get("best_bs")
        if bench_out:
            Path(bench_out).write_text(json.dumps(
                {"suffix": suffix, "loss_impl": loss_impl,
                 "grad_ckpt": grad_ckpt, "best_bs": table,
                 "results": [c.get() for _k, _a, c in calls]}, indent=1))
        print(f"\nFOLD-READY{' (variant ' + suffix + ')' if suffix else ''} "
              "(save + pass to make_sdfbase_configs.py --fold-memprobe"
              + (" --loss-impl " + loss_impl if loss_impl else "")
              + ", then COMMIT the regenerated configs):")
        print(json.dumps(table, indent=1))
        return

    if mode == "bench":
        # Throughput of the production CE train path on a STAGED pool
        # (--sis, one cell; default the r2 control pool) for
        # bench_steps(--steps) optimizer steps, per (model x arm), under --loss-impl /
        # --grad-ckpt / --batch-size / --accum. One container per (model,
        # arm), all spawned together; never publishes a cell.
        want_arms = [a for a in arms.split(",") if a] or ["psr_ce_all"]
        cell = (sis.split(",")[0] if sis else "constv2ctl2-full")
        if not batch_size:
            raise SystemExit("--mode bench needs --batch-size (the micro-"
                             "batch to measure at)")
        # The reconnaissance's per-run overrides (--lr / --max-grad-norm /
        # --warmup-frac) ride the item tail and the auto tag.
        overrides = bench_overrides(lr, max_grad_norm, warmup_frac)
        # A recon run (any override) requests EXACTLY the reconnaissance
        # budget — defaulted when --steps is omitted, refused otherwise
        #; a generic bench keeps its default / explicit count.
        steps = bench_steps(steps, overrides)
        run_tag = tag or (f"{cell}__{loss_impl or 'config'}__bs{batch_size}"
                          f"__acc{accum or 'auto'}"
                          f"__gc{grad_ckpt or 'config'}__s{steps}"
                          + bench_override_tag(overrides))
        calls = []
        for k in keys:
            for a in want_arms:
                item = (k, a, cell, int(batch_size), loss_impl, int(steps),
                        grad_ckpt, int(accum), bool(check_grads),
                        f"{a}__{run_tag}", overrides.get("lr"),
                        overrides.get("max_grad_norm"),
                        overrides.get("warmup_frac"))
                calls.append((k, a, deployed_fn(deployed, "bench_h200",
                                                key=k).spawn(item)))
                print(f"[bench] spawned {k}/{a}: {item}")
        results = []
        bad = 0
        for k, a, c in calls:
            r = c.get()
            results.append(r)
            bad += r.get("status") != "ok"
            keep = {kk: r.get(kk) for kk in (
                "status", "loss_impl", "batch_size", "accum_steps",
                "gradient_checkpointing", "steps_measured", "steps_completed",
                "tok_per_s_loss", "tok_per_s_real", "sec_per_step_median",
                "pad_fraction", "gpu_duty_steady", "peak_alloc_gib",
                "peak_reserved_gib", "chunk_check", "grad_parity",
                "lr", "max_grad_norm", "warmup_frac")}
            keep["grad_norm_preclip_after_warmup"] = (
                (r.get("grad_norm_preclip") or {}).get("quantiles_after_warmup"))
            print(f"[bench] {k}/{a}: {json.dumps(keep)}")
            if r.get("status") != "ok":
                print(f"[bench] {k}/{a} {str(r.get('status')).upper()} "
                      f"rc={r.get('rc')} error={r.get('error')!r}\n"
                      + str(r.get("stderr_tail", ""))[-2000:])
        if bench_out:
            Path(bench_out).write_text(json.dumps(results, indent=1))
            print(f"[bench] wrote {bench_out}")
        if bad:
            raise SystemExit(f"[bench] {bad} run(s) did not finish ok "
                             "(diverged / failed) — their records are on "
                             "the volume; the recon reader (scripts/sdfbase/"
                             "recon_clip.py) classifies them")
        return

    if mode == "hpo-spawn":
        # The DURABLE hpo launch (the fits-spawn pattern, one FunctionCall
        # per TRIAL): deploy first (`modal deploy [--name <app>]
        # scripts/modal/sdfbase_fits.py`, the app `deployed` resolves to);
        # this spawns each (model, arm, index, round) against the DEPLOYED
        # hpo functions and exits — call lifetime is server-side. Trials
        # keep whole-trial resume (fingerprint self-skip), so re-running
        # this mode tops up exactly the unfinished set.
        import getpass
        from datetime import datetime, timezone

        want_arms = [a for a in arms.split(",") if a] or CE_ARMS
        rnd = int(round_no)
        n_trials = {}
        design_si = {}
        designs = {}
        for k in keys:
            for a in want_arms:
                tf = _stage_path(f"sdfbase/hpo/trials/"
                                f"r{rnd}_{k}_{a}_all.json")
                if not tf.exists():
                    raise SystemExit(f"[hpo-spawn] no local design {tf} — "
                                     "emit-trials + volume-put it first")
                design = json.loads(tf.read_text())
                n_trials[(k, a)] = len(design["trials"])
                design_si[(k, a)] = _design_si(design)
                designs[(k, a)] = design
        fns = {v: deployed_fn(deployed, f"hpo_h200_{v}")
               for v in ("std", "qwen", "nemotron")}
        mpath = Path(manifest).expanduser() if manifest else (
            _stage_path(f"sdfbase/hpo_calls_r{rnd}.json"))
        mpath.parent.mkdir(parents=True, exist_ok=True)
        doc = {"app": deployed, "round": rnd,
               "launcher": getpass.getuser(), "written_at": None,
               "trials": []}
        # ALWAYS spawn — no local/volume trial-dir existence check may
        # suppress a spawn: tune_arm's in-container fit-fingerprint gate
        # is the ONE authoritative cheap skip (a marker/dir listing can
        # not tell a finished trial from a stale or wrong-identity one).
        # A published trial's container starts, self-skips in seconds,
        # and exits.
        # Every row records the design-side identity its summary must
        # echo (hpo_expect_identity), so fits_watch.py validates a
        # publication rather than counting a summary.json's existence.
        rows = [{"model": k, "arm": a, "index": i, "round": rnd,
                 "si": design_si[(k, a)], "app": deployed, "call_id": None,
                 "spawn_time": None, "status": "pending-spawn",
                 "expect": hpo_expect_identity(designs[(k, a)], i, a)}
                for k in keys for a in want_arms
                for i in range(n_trials[(k, a)])]
        doc["trials"] = rows
        _flush_manifest(doc, mpath)
        for r in rows:
            fn = fns[_variant(r["model"])]
            call = fn.spawn((r["model"], r["arm"], r["index"], rnd))
            r["call_id"] = call.object_id
            r["spawn_time"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
            r["status"] = "spawned (in-container skip expected)"
            _flush_manifest(doc, mpath)
        print(f"[hpo-spawn] {len(rows)} trial(s) spawned on the deployed "
              f"app {deployed!r} (published trials self-skip in-container "
              f"on the trial fingerprint); manifest: {mpath}")
        return

    if mode == "snapshot":
        # BEFORE any refit spawn, relocate each
        # to-be-superseded COMPLETE cell to the <key>_r1 snapshot bank
        # (sha-verified both sides, supersede records both sides) —
        # scripts/sdfbase/snapshot_cells.py, run in a CPU container
        # against the volume. _fit_impl's supersede_gate refuses to train
        # over an unsnapshotted publication of another identity, so this
        # step cannot be skipped by accident. --verify-only re-runs the
        # verify half on an existing snapshot bank.
        if not cells:
            raise SystemExit("[snapshot] --cells required (e.g. "
                             "constv2-mix-1m__psr_ce_all,"
                             "constv2-mix-1m__lora_r4_ce)")
        want_cells = [c for c in cells.split(",") if c]
        for c in want_cells:
            if "__" not in c or c.split("__", 1)[1] not in CE_ARMS:
                raise SystemExit(f"[snapshot] {c!r} is not a <si>__<arm> "
                                 f"cell id with arm in {CE_ARMS}")
        # --bank-suffix names the snapshot bank (_r1 default; _r2cap24 for
        # the cap-24 publications a continuation supersedes);
        # --continue-cap records the continuation's fingerprint as the
        # successor and refuses a cell without a retained checkpoint.
        snapshot_bank_root("out/x", bank_suffix or "_r1")  # validate early
        lr_mult = _check_continue_lr_mult(continue_lr_mult, continue_cap,
                                          "[snapshot]")
        snap = deployed_fn(deployed, "snapshot_cpu")
        calls = [(k, snap.spawn((k, want_cells, bool(verify_only), reason,
                                 out_root or f"out/{k}", bank_suffix or "_r1",
                                 int(continue_cap or 0), lr_mult)))
                 for k in keys]
        bad = 0
        for k, c in calls:
            r = c.get()
            print(f"[snapshot] {k}: {json.dumps(r, indent=1)}")
            bad += r.get("status") != "ok"
        if bad:
            raise SystemExit(f"[snapshot] {bad} model(s) had refused cells "
                             "— nothing about those cells may be refit "
                             "until the snapshot verifies")
        return

    if mode == "fits-spawn":
        # THE DURABLE LAUNCH PATH (the twosi fit-wave hardening): deploy
        # first, then one server-side FunctionCall per cell, manifest
        # pre-seeded with call_id=null rows and atomically rewritten after
        # EVERY spawn. EVERY requested cell spawns — the driver's
        # fingerprint-validated COMPLETE gate in-container is the ONE
        # authoritative cheap skip (a marker-existence listing here could
        # suppress the respawn of a stale or wrong-identity cell).
        # Per-cell kill switch: modal.FunctionCall.from_id(<id>).cancel().
        import getpass
        from datetime import datetime, timezone

        if len(keys) != 1:
            raise SystemExit("[fits-spawn] one --model per invocation "
                             "(the expect_sha is per-model config bytes)")
        key = keys[0]
        cfg_local = WORKTREE / Path(MODELS[key]["cfg"]).relative_to("/repo")
        expect_sha = _config_sha16(cfg_local)
        root = out_root or f"out/{key}"
        cont_cap = int(continue_cap or 0)
        lr_mult = _check_continue_lr_mult(continue_lr_mult, cont_cap,
                                          "[fits-spawn]")
        want_pairs = None
        if cont_cap:
            # Cap-raise continuation: explicit --cells only; the
            # rows' fingerprints are the CONTINUED fits' (derived in-
            # container from each cell's retained final_ckpt/).
            want_pairs = continuation_cells(cells, sis=sis, arms=arms,
                                            retrain_all=retrain_all,
                                            overwrite=overwrite, cap=cont_cap)
            want_sis = list(dict.fromkeys(s for s, _a in want_pairs))
            want_arms = list(dict.fromkeys(a for _s, a in want_pairs))
        else:
            # SCOPED by contract: --sis, or an explicit --retrain-all —
            # never a silent all-SIS default.
            want_sis = resolve_spawn_sis(sis, retrain_all)
            want_arms = [a for a in arms.split(",") if a] or CE_ARMS
        fname = f"fit_h200_{_variant(key)}"
        fn = deployed_fn(deployed, fname)
        # Every manifest row records the cell's EXPECTED fit fingerprint,
        # derived IN-CONTAINER through the driver's own functions (the
        # pools live on the volume). The deployed config bytes must be the
        # local ones, or the derived fingerprints describe a config the
        # fit would refuse anyway — which is also what catches a spawn
        # aimed at the WRONG app (an older frozen deployment has none of
        # the current configs in its image).
        fp_fn = deployed_fn(deployed, "fingerprint_cpu")
        fp_pairs = (want_pairs if want_pairs is not None
                    else [(s, a) for s in want_sis for a in want_arms])
        # (a non-1.0 lr multiplier rides as a 5th element; the plain and
        # plain-continuation items keep the 4-tuple byte for byte)
        fp_req = (key, [[s, a] for s, a in fp_pairs], cont_cap, root)
        if lr_mult != 1.0:
            fp_req += (lr_mult,)
        fp_doc = fp_fn.remote(fp_req)
        # equal bytes, or a deployed config the release rewrote by declared rules only
        # (a certified scrub-map row: the same file with site paths in placeholder form)
        _repo_imports()
        from sisteer import scrub_map as _sm
        cfg_check = _sm.check(fp_doc.get("config_sha"), expect_sha, path=MODELS[key]["cfg"])
        if cfg_check is None:
            raise SystemExit(
                f"[fits-spawn] the DEPLOYED app {deployed!r}'s "
                f"{MODELS[key]['cfg']} hashes to {fp_doc.get('config_sha')}, "
                f"the local config to {expect_sha} — redeploy (modal deploy "
                f"--name {deployed} scripts/modal/sdfbase_fits.py) before "
                "spawning; the manifest must record fingerprints of the "
                "config the fits will actually run")
        if cfg_check == _sm.VIA_MAP:
            print(f"[fits-spawn] the deployed {MODELS[key]['cfg']} is the scrub map's pre-image of the local "
                  f"config ({_sm.VIA_MAP})")
        if fp_doc.get("missing_pools"):
            raise SystemExit("[fits-spawn] pools not staged for: "
                             + "; ".join(fp_doc["missing_pools"]))
        if cont_cap and fp_doc.get("missing_checkpoints"):
            raise SystemExit(
                "[fits-spawn] --continue-cap: no continuable retained "
                "checkpoint for: " + "; ".join(fp_doc["missing_checkpoints"])
                + " — a cell without <cell>/final_ckpt/ckpt.pt cannot be "
                "continued (snapshot + fits-spawn refuse it; a scratch refit "
                "is the only route)")
        # Version pin for the successor-creating watcher (fits_watch.py
        # same_app_version). The deployed function's id is recorded but
        # does NOT change on redeploy (v10 -> v11 kept fit_h200_std /
        # fit_h200_qwen's ids), so the pin that actually moves is the
        # app's production deployment version (app_id + app_version + the
        # deploy commit, fits_watch.deployment_pin) — a successor is
        # refused if the app moved mid-wave.
        try:
            fn.hydrate()
            function_id = fn.object_id
        except Exception as e:  # noqa: BLE001 — no deployed handle (stub)
            function_id = None
            print(f"[fits-spawn] WARNING: could not pin the deployed "
                  f"function id ({type(e).__name__}: {e}); the watcher "
                  "will refuse successors for this manifest")
        try:
            deployment = _deployment_pin(deployed)
        except Exception as e:  # noqa: BLE001 — no deployment (stub)
            deployment = None
            print(f"[fits-spawn] WARNING: could not pin the deployment "
                  f"version of {deployed!r} ({type(e).__name__}: {e}); the "
                  "watcher will refuse successors for these rows unless "
                  "run with --infer-legacy-app-version")
        mpath = Path(manifest).expanduser() if manifest else (
            _stage_path("sdfbase/fit_calls.json"))
        mpath.parent.mkdir(parents=True, exist_ok=True)
        # Top-level values = THIS fits-spawn's (informational: one manifest
        # carries the whole campaign, so on a merged manifest they describe
        # only the last spawn). Every ROW carries its own pin (spawn_pin);
        # fits_watch.py reads the row's, never these.
        doc = {"app": deployed, "function": fname, "model": key,
               "function_id": function_id,
               "deployment": deployment,
               "out_root": root, "config_sha": expect_sha,
               "checkpoint_every": int(checkpoint_every),
               "overwrite": bool(overwrite),
               "continue_cap": cont_cap,
               "continue_lr_mult": lr_mult,
               "launcher": getpass.getuser(),
               "written_at": None, "cells": []}
        if mpath.exists():
            try:
                old = json.loads(mpath.read_text())
                # Keep rows for cells OUTSIDE this request (the other
                # model / another root), so one manifest carries the whole
                # campaign; rows for the requested cells are replaced.
                # EXACTLY the cells this spawn replaces — in continuation
                # mode the explicit --cells, never their SI x arm product:
                # a sparse selector must not drop unrelated rows the
                # watcher still tracks.
                requested = {(key, root, f"{si}__{a}") for si, a in fp_pairs}
                doc["cells"] = [
                    c for c in old.get("cells", [])
                    if (c.get("model"), c.get("out_root"),
                        c.get("cell_id")) not in requested]
            except ValueError:
                print(f"[fits-spawn] {mpath} unparseable — rewriting it")

        # Per-row spawn pin: the row, not the manifest's top level, is
        # what the watcher respawns from.
        pin = spawn_pin(fname, function_id, expect_sha, overwrite=overwrite,
                        checkpoint_every=checkpoint_every,
                        continue_cap=cont_cap, app=deployed,
                        continue_lr_mult=lr_mult, deployment=deployment)
        rows = spawn_rows(key, root, want_sis, want_arms,
                          fp_doc["fingerprints"], pin=pin, cells=want_pairs,
                          continuations=fp_doc.get("continuations"))
        doc["cells"].extend(rows)
        _flush_manifest(doc, mpath)  # the full expected set, BEFORE spawns
        for c in rows:
            item = (key, c["si"], c["arm"], expect_sha, overwrite, root,
                    int(checkpoint_every))
            if cont_cap:
                # The 8th element makes the container a continuation
                # (fits_watch.successor_item mirrors it); a 9th carries a
                # non-1.0 lr multiplier — plain continuation rows keep the
                # 8-tuple byte for byte.
                item += (cont_cap,)
                if lr_mult != 1.0:
                    item += (lr_mult,)
            call = fn.spawn(item)
            c["call_id"] = call.object_id
            c["spawn_time"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
            c["status"] = "spawned (in-container skip expected)"
            _flush_manifest(doc, mpath)  # after EVERY spawn
            print(f"[fits-spawn] {c['cell_id']} -> {call.object_id}")
        print(f"[fits-spawn] {len(rows)} cell(s) spawned on deployed "
              f"{deployed}/{fname} (COMPLETE cells self-skip in-container on the "
              f"driver's fit fingerprint; each row records its expected "
              f"fingerprint for the watcher); manifest: {mpath}")
        print("[fits-spawn] harvest: python scripts/modal/fits_watch.py "
              f"--manifest {mpath}")
        return

    if mode == "watch":
        # The fits_watch.py contract, delegated to fits_watch.py itself
        # (same volume, same manifest row shape): exit 0 all COMPLETE,
        # exit 2 respawn/triage list, exit 3 --max-hours elapsed.
        import subprocess
        import sys

        mpath = Path(manifest).expanduser() if manifest else (
            _stage_path("sdfbase/fit_calls.json"))
        p = subprocess.run(
            [sys.executable, str(WORKTREE / "scripts/modal/fits_watch.py"),
             "--manifest", str(mpath), "--poll", str(int(poll)),
             "--max-hours", str(float(max_hours)),
             "--respawn" if respawn else "--no-respawn",
             "--max-successors", str(int(max_successors))])
        raise SystemExit(p.returncode)

    if mode == "probe-score":
        want = cells or "all"
        probe_local = WORKTREE / f"data/sdfbase/{probe_file_stem(probe)}.jsonl"
        if not probe_local.exists():
            raise SystemExit(f"[probe-score] {probe_local} missing — freeze "
                             f"probe {probe} before scoring it")
        # --scale 2.0 = the round-2 2x-strength pass: object rows only,
        # margins in the _x2 dir (probe_out_dirname), never the scale-1
        # files. Gate rows are object-less and are skipped off-1.
        if float(scale) != 1.0 and want == "gates":
            raise SystemExit("[probe-score] --scale is an OBJECT knob; the "
                             "gate rows are object-less — a scaled gates "
                             "pass measures nothing")
        _fanout_models("probe-score", "score_h200",
                       lambda k: (k, want, probe, float(scale)))
        return

    if mode == "mmlu":
        if not cells:
            raise SystemExit("[mmlu] --cells required (e.g. none,"
                             "constv2-mix-1m__psr_ce_all,...)")
        # Run-scoped by contract: --run-label is REQUIRED
        # (r1 = the snapshot bank, r2 = the refit bank), --defenses-root
        # picks the bank ({key} expands per model).
        if not run_label:
            raise SystemExit("[mmlu] --run-label required (r1 for a "
                             "snapshot-bank run, r2 for the refit bank): "
                             "every output carries it in dir + filename so "
                             "runs never share a path")
        # --scale rides the item (the 2x-dose pass: --run-label r2_x2
        # --scale 2.0); the default 1.0 leaves the command byte-identical.
        _fanout_models("mmlu", "mmlu_h200",
                       lambda k: (k, cells, run_label,
                                  expand_defenses_root(defenses_root, k),
                                  float(scale)))
        return

    if mode == "gsm8k":
        if not cells:
            raise SystemExit("[gsm8k] --cells required (e.g. none,"
                             "constv2-mix-1m__psr_ce_all,...)")
        # Run-scoped like mmlu: --run-label REQUIRED (r2 = 1x on the live
        # bank, r2_x2 = the 2x pass), --defenses-root picks the bank
        # ({key} expands per model), --scale is the object strength.
        if not run_label:
            raise SystemExit("[gsm8k] --run-label required (r2 for the 1x "
                             "pass, r2_x2 for --scale 2.0): every output "
                             "carries it in dir + filename so runs never "
                             "share a path")
        # --batch-size (0 = script default 32): Nemotron runs at 128, see
        # gsm8k_command; it is part of the run identity, so a model's two
        # labels must use the same value. --decode-parity <volume path>:
        # the attestation base_gsm8k requires before adding cached cells to
        # a label holding cells published before the decode-path fix
        # (see require_decode_parity).
        _fanout_models("gsm8k", "gsm8k_h200",
                       lambda k: (k, cells, run_label,
                                  expand_defenses_root(defenses_root, k),
                                  float(scale), int(batch_size),
                                  decode_parity or None))
        return

    if mode == "ce":
        # Standalone held-out CE for STAGED cells at each of --scales on
        # the analysis pools (CE_POOLS; --ce-pools tag:path,... overrides),
        # the derived app's ce_score layout — one container per model.
        if not cells:
            raise SystemExit("[ce] --cells required (staged cell ids, or "
                             "'bare')")
        want_scales = scales or str(float(scale))
        parse_scales(want_scales)  # refuse before any spawn
        _fanout_models("ce", "ce_h200",
                       lambda k: (k, cells, want_scales, ce_pools))
        return

    if mode == "battery":
        # Instruct-transfer battery: one container per
        # (attack x cell x shard), plus one per cell for each of the
        # UNSHARDED_BATTERY_ATTACKS (the capability leg "controls" and the
        # benign over-refusal pass "xstest"), all durable spawns on the
        # DEPLOYED app,
        # against configs/sdfbase_transfer_<model>.yaml (the explicit-cell
        # registry — agent E's seam; this mode just fans out).
        import getpass
        from datetime import datetime, timezone

        if len(keys) != 1:
            raise SystemExit("[battery] one --model per invocation")
        key = keys[0]
        if not cells:
            raise SystemExit("[battery] --cells required (the transfer "
                             "config's explicit_cells enumeration)")
        # An instruct-direct key is batteried ONLY under its own wave
        # namespace (`_it`: own bare anchor, objects applied to the
        # instruct model directly), and that namespace is launched
        # only under an instruct-direct key (check_battery_wave: one
        # table, MODELS[key]["battery_wave"], both directions). Under any
        # other wave the slug's config names a BASE-fit transfer bank, so
        # a manifest labelled with this key would have served another
        # lane's objects.
        check_battery_wave(key, wave)
        # The battery serves the INSTRUCT sibling — the transfer configs
        # are named by it: model key -> transfer slug through
        # the ONE table (TRANSFER_SLUG, read by battery_config_path; an
        # unregistered key refuses by name), plus the WAVE namespace
        # (--wave _r2 = the round-2 controls-in-battery config with its
        # fresh private_run/out_dir).
        cfg_path = battery_config_path(key, wave)
        cfg_local = WORKTREE / Path(cfg_path).relative_to("/repo")
        if not cfg_local.exists():
            raise SystemExit(f"[battery] {cfg_local} missing — the "
                             "transfer config must be committed first")
        # The bank the containers will serve from is the wave's (battery_bank);
        # the committed config must name the same dir, or refuse here — before
        # any spawn — rather than serve one bank under a config naming another.
        bank = battery_bank_matches(cfg_local, cfg_path)
        # xstest is in the DEFAULT set, not an opt-in extra: the aggregate
        # refuses to grade a cell without its benign pass, so a battery that
        # omits it by default produces an ungradable campaign.
        want_attacks = ([a for a in attacks.split(",") if a]
                        or ["bon", "pasttense", "codeattack", "renellm",
                            "grok420", "controls", "xstest"])
        want_cells = battery_cells_in_config(
            cfg_local, [c for c in cells.split(",") if c])
        fname = f"battery_h200_{_variant(key)}"
        fn = deployed_fn(deployed, fname)
        want_shards = battery_shard_selection(shards, nshards)
        mpath = Path(manifest).expanduser() if manifest else (
            _stage_path(f"sdfbase/battery_calls_{key}{wave}.json"))
        mpath.parent.mkdir(parents=True, exist_ok=True)
        # A battery manifest is NEVER overwritten: its rows' call ids are the
        # only handle on containers already running (the watcher, the
        # private-transcript pull and the output check all read them), and
        # a second spawn of the same rows would write the same volume files
        # concurrently. A re-run or a repair names a FRESH --manifest and
        # selects its rows (--attacks/--cells/--shards). The reservation is
        # ATOMIC (exclusive create): two launches racing for one manifest
        # cannot both pass an exists() check and then both spawn.
        try:
            with mpath.open("x") as fh:
                fh.write("{}\n")     # reserved; the first flush fills it
        except FileExistsError:
            raise SystemExit(f"[battery] manifest {mpath} already exists — "
                             "never overwritten; name a fresh --manifest "
                             "(and select the rows to re-spawn with "
                             "--attacks/--cells/--shards)")
        doc = {"app": deployed, "function": fname, "model": key,
               "config": cfg_path, "wave": wave, "bank": bank,
               "confirmatory": bool(confirmatory),
               "shards_selected": want_shards,
               "launcher": getpass.getuser(), "written_at": None,
               "shards": []}
        rows = []
        for atk in want_attacks:
            for cell in want_cells:
                one_per_cell = atk in UNSHARDED_BATTERY_ATTACKS
                shard_range = [0] if one_per_cell else list(range(nshards))
                if want_shards is not None:
                    shard_range = [sh for sh in shard_range if sh in want_shards]
                for sh in shard_range:
                    rows.append({"attack": atk, "cell": cell, "shard": sh,
                                 "nshards": 1 if one_per_cell else nshards,
                                 "app": deployed,
                                 "call_id": None, "spawn_time": None,
                                 "status": "pending-spawn"})
        if not rows:
            mpath.unlink()           # nothing spawned: release the reservation
            raise SystemExit(f"[battery] --attacks {want_attacks} x --cells "
                             f"{want_cells} x --shards {want_shards} selects "
                             "no (attack, cell, shard) row — nothing to spawn")
        doc["shards"] = rows
        _flush_manifest(doc, mpath)
        for r in rows:
            call = fn.spawn((cfg_path, r["attack"], r["cell"], r["shard"],
                             r["nshards"], bool(confirmatory)))
            r["call_id"] = call.object_id
            r["spawn_time"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
            r["status"] = "spawned"
            _flush_manifest(doc, mpath)
            print(f"[battery] {r['attack']}/{r['cell']}/s{r['shard']} -> "
                  f"{call.object_id}")
        print(f"[battery] {len(rows)} shard(s) spawned on deployed "
              f"{deployed}/{fname}; manifest: {mpath}")
        return

    raise SystemExit(f"unknown --mode {mode!r}")
