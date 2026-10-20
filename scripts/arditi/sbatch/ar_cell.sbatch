#!/bin/bash
#SBATCH --job-name=ar-cell
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-08:00:00
#SBATCH --output=/workspace-vast/%u/exp/jailbreak/_logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# ar_cell.sbatch — ONE Arditi strength cell, END TO END, in ONE SERVER STANDUP.
#
# THE STANDUP IS SINGULAR ON PURPOSE, and it is a correctness property rather than an
# efficiency one. The claim an Arditi baseline makes is a TRADE — this much defence for this
# much capability cost — and that trade only means something if both sides were measured on
# the same served state. Capability measured in a later job is a different standup, with a
# different server lifetime between it and the safety numbers. So MMLU, the four single-shot
# families and XSTest all run against one loaded object, in one server. (It also pays the
# standup cost once instead of three times, which will matter at fan-out; that is the lesser
# reason and it is not why the job is shaped this way.)
#
# ONE WINDOW SERVES ALL THREE LEGS. `families_served:short` = 4,864 x S (the measured static
# prompt p99 762 + the 4,096 cap; qwen35_4b S = 148 from its measured budget). The controls
# legs need far less (capability:short p99 prompt 575 + a 512 cap), so the one window covers
# them with room; the cost is the controls legs run at S=148 instead of the 256 a 2,048 window
# would allow. That is the price of the single standup and it is worth it.
#
# LEG ORDER IS DELIBERATE: MMLU first (minutes, no decode — so a broken prefill path is found
# before four hours of generation), then XSTest (short), then the four families (the long
# pole). Each leg's wall clock is recorded separately into timings.json: they have completely
# different shapes and the fan-out has to be sized from the parts, not the total.
#
# WHAT IS DECODED HOW:
#   * the four single-shot families — SAMPLED at the served model's own vendor row, one seeded
#     draw per item, at the 4,096 cap (`sampled_decode` in the config; the preset is read from
#     contracts by AST, never retyped — see make_arditi_sampled_config.py).
#   * XSTest — SAMPLED at the same row, at the 512 cap, with a per-item meta sidecar recording
#     finish_reason / at_cap / the sampler the body carried.
#   * MMLU — LETTER PREFILL ONLY, no decode: a teacher-forced one-token read of the four letter
#     logprobs at temperature 0. The sampler is MOOT there and must not appear; the lane test
#     pins that the scoring payload carries no sampler key.
#
# THE RESOLVED SAMPLER IS RECORDED, NOT ASSUMED. The server runs with --enable-log-requests, so
# vLLM writes one INFO line per request carrying the SamplingParams it actually resolved (the
# prompt is a separate DEBUG record and is NOT logged at INFO, so nothing harmful travels).
# record_resolved_sampler.py parses those lines, compares them against what the client rows say
# they sent, and REFUSES to read silence as agreement. Rationale: the resolved sampler is a
# property of the serving stack, not of the model, so two lanes can resolve the same omitted key
# differently and diverge with no summary statistic showing it.
#
# THE THREE DECLARED DEVIATIONS ride every artifact this writes (they are in the config header,
# the object sidecars and the roster, and cell_facts.json restates them): PREFILL-ONLY phase
# (Arditi's own hook fires at every decode step; ours does not, so generated tokens see the edit
# only through the steered prompt KV — half his intervention), OUR CHAT-TEMPLATE RENDERING rather
# than his per-family strings, and THE UNIT DIRECTION AT A MEASURED FRACTION OF ||h|| rather than
# the raw difference of means. A reader must not be able to infer we ran his method unmodified.
#
# Usage (positional only — NEVER --export=), and always through submit_arditi.sh:
#   sbatch -J ar-cell-<target>-<view> scripts/arditi/sbatch/ar_cell.sbatch <FULL 40-hex sha> <target> <view> [trial=0] [nshards=4] [legs=all] [driver_cap=96]
#     legs: all | controls (standup + MMLU + the XSTest ladder) | xstest (the ladder only) | static (the four families)
#           | orbench (the OR-Bench-Hard cap ladder ALONE — the operating-point leg; not part of `all`)
#           | mmlu_full (standup + MMLU over the WHOLE 1,531-item pinned pool ALONE, into $CTL_ROOT/mmlu_full with the
#             driver's --mmlu-full — a DIFFERENT store identity from the 200-item `mmlu` leg, never pooled with it;
#             not part of `all`). A partial
#     driver_cap: how many static drivers may be alive at once (default 96, the wave's). The liveness probe every
#     driver fires at start-up measures its noise ceiling as the scale-0 twin's REPEAT divergence, and that
#     ceiling is nonzero exactly when other requests share the engine's batches (NOTES §18): under 96 lanes the
#     bar (2x ceiling) sat ABOVE a live steer's signal at alpha 0.05 on gemma and nemotron and the lanes were
#     refused. driver_cap=1 makes every probe run ALONE (ceiling 0.0, as the control legs measure) at the price
#     of one request in flight — the completion pass for a liveness-starved cell, never the default.
#     run writes its records under cells/<TAG>_t<T>/legs_<legs>_<jobid>/ so the full run's records survive; the
#     serving dir is job-suffixed already and the private_run / controls roots are shared by construction
#     (a controls re-run after a failed controls leg: move the failed cell dirs OUT of controls/<TAG>_t<T>/*/cells/
#     first — the driver would overwrite a stale manifest.json in place).
#   bash scripts/arditi/sbatch/ar_cell.sbatch <sha> precheck [<target> <view> [trial]]   # the preconditions only
set -uo pipefail
SHA="${1:?FULL 40-hex sha}"
case "$SHA" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;;
  *) echo "FATAL: '$SHA' is not a full 40-hex sha; f22_init compares it against git rev-parse HEAD and will refuse" >&2; exit 1;;
esac
# `precheck` may arrive as $2 (the submitter's shape) or after the positional args
if [ "${2:-}" = precheck ]; then PRECHECK=precheck; TARGET="${3:-qwen35_4b}"; VIEW="${4:-ar_r0p1}"; TRIAL="${5:-0}"; NSHARDS=4; LEGS=all
else PRECHECK=""; TARGET="${2:?target}"; VIEW="${3:?view}"; TRIAL="${4:-0}"; NSHARDS="${5:-4}"; LEGS="${6:-all}"; fi
case "$LEGS" in all|controls|static|xstest|orbench|mmlu_full) ;; *) echo "FATAL: legs='$LEGS' is not one of all|controls|static|xstest|orbench|mmlu_full" >&2; exit 1;; esac
# positional precheck: `<sha> <target> <view> <trial> <nshards> <legs> [driver_cap] precheck` prechecks THAT cell
# (the $2 form prechecks the pilot's qwen35_4b ar_r0p1 only). $7 is the driver cap when it is an integer.
DRIVER_CAP=96
case "${7:-}" in
  precheck) PRECHECK=precheck;;
  "") ;;
  *) [[ "$7" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: driver_cap='$7' is not a positive integer (or 'precheck')" >&2; exit 1; }
     DRIVER_CAP="$7";;
esac
[ "${8:-}" = precheck ] && PRECHECK=precheck
LEGS_SUFFIX=""; [ "$LEGS" = all ] || LEGS_SUFFIX="_legs_${LEGS}"
# ONE predicate per leg family, derived here and used everywhere below. Job 487454 (legs=xstest) launched the
# STATIC drivers into a finished private_run because LEG 3 was gated on `!= controls` alone; the drivers resumed
# from progress.jsonl and returned in 59 s with the stores byte-unchanged — luck, not design.
# OR-BENCH IS NOT AN ACROSS-DOSE READING. The ladder measures exactly
# THREE things per rung -- the four static jailbreak families, XSTest, and MMLU -- and everything
# else, OR-Bench included, is measured ONLY at the selected operating point (Phase F). So
# `RUN_ORBENCH` is 1 for its OWN legs value and for nothing else: `all` no longer implies it, and
# the operating-point run asks for it by name. The leg itself (LEG 1b) is kept exactly as built --
# the machinery, the `orb_cap512` out-root the OR-Bench grader discovers, and its tests -- because
# the operating-point run needs all of it; only its membership in the per-rung set changed.
RUN_CONTROLS=0; RUN_MMLU=0; RUN_ORBENCH=0; RUN_STATIC=0; RUN_MMLU_FULL=0
case "$LEGS" in
  all)      RUN_CONTROLS=1; RUN_MMLU=1; RUN_STATIC=1;;
  controls) RUN_CONTROLS=1; RUN_MMLU=1;;
  xstest)   RUN_CONTROLS=1;;
  static)   RUN_STATIC=1;;
  orbench)  RUN_ORBENCH=1;;
  mmlu_full) RUN_MMLU_FULL=1;;   # the 1,531-item MMLU read alone (mmlu-full-1531): its own root, its own identity
esac

{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$SHA/scripts/arditi/sbatch/ar_common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$SHA/scripts/arditi/sbatch/ar_common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
# THE LAUNCH'S CELLS, DECLARED BEFORE ar_init (arditi-qwen-all-positions A3.1). f22_select_serving_path
# — which f22_init calls, and which did not exist when this lane was written — FATALs unless the
# caller declares a cell source, so every job of this lane would die at init on this tree. It then
# picks the serving path from those cells BY KIND: `arditi` (static_r) is a lens-only kind and
# selects the lens, exactly as before; `arditi_allpos` (vector_all) is natively servable on the
# residual slot and selects lora_native. The declaration is therefore also what routes this job to
# the right engine, and it is READ FROM THE DOSE TABLE rather than typed, because the cell is a fact
# about the VIEW's unit (`ar_r...` -> arditi, `ar_c...` -> arditi_allpos), not a constant of this
# lane any more.
mapfile -t F22_LAUNCH_CELLS < <(ar_launch_cells "$VIEW")
read -r AR_UNIT AR_CELL < <(ar_unit_of_view "$VIEW")
# mapfile CANNOT FAIL: a process substitution that dies leaves it a clean exit over empty input, and
# an EMPTY F22_LAUNCH_CELLS is not an error to f22_select_serving_path — it is the declared meaning
# "this job stands up no target", which would pick the entry-point default path and serve the wrong
# engine with nothing said. So the array is checked here, where the failure is still legible.
[ "${#F22_LAUNCH_CELLS[@]}" -eq 2 ] && [ "${F22_LAUNCH_CELLS[0]}" = none ] && [ -n "${F22_LAUNCH_CELLS[1]}" ] \
  || { echo "FATAL: ar_launch_cells '$VIEW' yielded [${F22_LAUNCH_CELLS[*]:-}] — expected exactly 'none <cell>'" >&2; exit 1; }
[ -n "$AR_UNIT" ] && [ "${F22_LAUNCH_CELLS[1]}" = "$AR_CELL" ] \
  || { echo "FATAL: view '$VIEW' resolved unit '${AR_UNIT:-}' cell '${AR_CELL:-}', which disagrees with the declared cells [${F22_LAUNCH_CELLS[*]}]" >&2; exit 1; }
# the serving MODE follows the unit, never a flag: the residual-norm-fraction rungs are the lens
# stack pinned byte-identical to the reference serving commit (`production`), the raw-coefficient
# rungs are the paper's own configuration on the graph-captured native residual slot (`native`,
# which the spec REQUIRES in every leg that serves that object). One job script, two modes, chosen
# by what it serves.
case "$AR_UNIT" in
  residual_norm_fraction) AR_MODE=production;;
  raw_coefficient)        AR_MODE=native;;
  *) echo "FATAL: view '$VIEW' resolves to dose unit '$AR_UNIT', which has no serving mode" >&2; exit 1;;
esac
ar_init "$SHA"
ar_assert_serving_identity "$AR_MODE"          # policy + config, endpoint-free: before ANYTHING is served
f22_stop_requested && { f22_log "lane STOP present — not starting"; exit 0; }

ar_resolve_config "$TARGET" "$VIEW"            # KEY VIEW TAG CFG MANIFEST STRENGTH SERVED_LAYER + AR_RUN_TAG AR_KIND AR_PHASE AR_MANIFEST_SERVING_IMPL_COMMIT
SERVE_CFG="$CFG"
# the sampled config and the private run are named by the UNIT's run tag (arditi_strength.run_tag_in:
# `arditi_strength_<t>_ar_r<tok>` | `arditi_allpos_<t>_ar_c<tok>`), never by the lane's old prefix alone
CFG="configs/${AR_RUN_TAG}_sampled_t${TRIAL}.yaml"
[ -f "$CFG" ] || f22_fatal "missing $CFG (generate: scripts/arditi/make_arditi_sampled_config.py --target $TARGET --view $VIEW --trial $TRIAL)"
PRIVATE_RUN="${AR_RUN_TAG}_sampled_t${TRIAL}"

# THE SAMPLED CONFIG IS THE SERVE CONFIG AS DATA, except the three keys — asserted job-side, so
# a config that drifted from its generator never reaches a GPU.
python3 - "$SERVE_CFG" "$CFG" "$TRIAL" "$PRIVATE_RUN" <<'PY' || f22_fatal "the sampled config is not the serve config + a decode block"
import sys, yaml
serve, sampled, trial, prun = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
w, c = (yaml.safe_load(open(p)) for p in (serve, sampled))
changed = {k for k in set(w) | set(c) if w.get(k) != c.get(k)}
assert changed == {"out_dir", "private_run", "sampled_decode"}, f"{sampled} differs from {serve} in {sorted(changed)}"
sd = c["sampled_decode"]
assert sd["trial"] == trial, f"sampled_decode.trial {sd['trial']} != {trial}"
assert c["private_run"] == prun, f"private_run {c['private_run']} != {prun}"
assert sd["static_cap"] == c["attacks"]["static_max_new_tokens"] == 4096, "static cap"
# THE CONFIG'S CONTROLS CAP IS `attacks.controls_max_new_tokens`, ELSE `attacks.target_max_new_tokens`
# -- the wave's own key since main's 2026-09-11 body rewrite, and the rule the sampled generator, the
# shared sampled-config validator and four sdfbase/full22/sampled readers all spell. This assertion was
# single-key, which is the SAME rule only while no body carries the new key: every prefill-only config
# predates the rewrite (target cap 512, no controls key, so the fallback reads exactly what it always
# read), while a config generated from a CURRENT body carries controls_max_new_tokens 512 with 4,096
# left behind in target_max_new_tokens -- so the old form refused the all-positions rungs by comparing
# the controls cap against the multi-turn one. The measured instrument is unchanged either way: 512.
_ccap = c["attacks"].get("controls_max_new_tokens", c["attacks"].get("target_max_new_tokens"))
assert sd["controls_cap"] == _ccap == 512, f"controls cap: block {sd['controls_cap']} vs config {_ccap}, want 512"
assert set(sd["applies_to"]) >= {"static_families", "controls"}, sd["applies_to"]
print(f"sampled config OK: {sampled} (trial {trial}, arm {sd['arm']}, seed_base {sd['seed_base']}, "
      f"preset " + ", ".join(f"{k}={sd[k]}" for k in
      ("temperature","top_p","top_k","min_p","presence_penalty","repetition_penalty") if k in sd) + ")")
PY

# THE PREMISE OF SENDING THE PRESET PER REQUEST, GENERALIZED ("the snapshot ships no
# generation_config.json" was the special case): the decode must be FULLY DETERMINED BY WHAT WE
# SEND — no sampler key may resolve from a vendor file. Checked against the cache this job serves
# FROM and the sampled_decode block this job sends: every sampler key the shipped
# generation_config.json sets must be set by the block. A file key the block does not set is FATAL,
# naming the key and the file's value, so a future target that ships a key we do not send fails here
# and legibly rather than resolving silently. Qwen (no file) satisfies it trivially.
python3 - "$CFG" <<'PY' || f22_fatal "the per-request decode is not fully determined by the request on this node's HF cache"
import os, sys, yaml, glob, json
cfg = yaml.safe_load(open(sys.argv[1]))
model, rev, sd = cfg["model"], cfg["model_revision"], cfg["sampled_decode"]
SAMPLER_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
sent = {k: sd[k] for k in SAMPLER_KEYS if k in sd}
hub = os.path.join(os.environ.get("HF_HOME", ""), "hub")
d = os.path.join(hub, "models--" + model.replace("/", "--"))
real = glob.glob(os.path.join(d, "snapshots", rev, "generation_config.json"))
noex = os.path.join(d, ".no_exist", rev, "generation_config.json")
if not real:
    print(f"premise holds: no generation_config.json for {model}@{rev} "
          f"(.no_exist marker {'present' if os.path.exists(noex) else 'absent, and no real file either'}); "
          f"the request sets {sorted(sent)}")
    raise SystemExit(0)
shipped = {k: v for k, v in json.load(open(real[0])).items() if k in SAMPLER_KEYS}
missing = {k: v for k, v in shipped.items() if k not in sent}
if missing:
    raise SystemExit(f"{model}@{rev} ships {real[0]} setting " + ", ".join(f"{k}={v!r}" for k, v in missing.items())
                     + f" and the sampled_decode block sends only {sorted(sent)} — "
                     + ("that key" if len(missing) == 1 else "those keys")
                     + " would resolve from the vendor file, not from the request. Refusing.")
print(f"premise holds: {model}@{rev} ships generation_config.json setting {shipped}; the request sets every one "
      f"of those keys explicitly ({ {k: sent[k] for k in shipped} }), so nothing resolves from the file")
PY

f22_driver_venv; f22_serving_venv

# THE LIVENESS STATISTIC: the three-leg first-position LOGPROB distance on the neutral probes,
# cleared against the twin's own repeat noise --
# sisteer.eval.attacks.bites_policy.POLICY_LOGPROB_NEUTRAL. Set in the BODY (never --export=), read
# by both the static drivers and the controls driver. text_neutral is retired for admission (it
# probes benign text where an additive refusal direction is designed not to act) but its statistic
# is still measured and written beside every verdict, so each admission the switch changes is
# countable. NOT harmful_continuous: that probes the defence population and would couple the gate
# to the result it gates.
export SISTEER_BITES_POLICY=logprob_neutral

# THE SERVED-EVIDENCE PRE-FLIGHT, on the real tokenizer this job serves. First contact (487306):
# the driver's loader rendered the template's DEFAULT regime (Qwen3.5 thinking-ON, open <think>)
# while the server renders thinking-OFF, chat_markers refused to guess, and MMLU + XSTest died
# before a single request. The unit tests stub the tokenizer and the precheck never rendered it,
# so the smoke that would have caught it never ran. Now it runs here, in precheck AND the job.
"$MPY" - "$CFG" <<'PYTOK' || f22_fatal "the driver cannot derive the served assistant header for $TARGET (see NOTES_arditi_serving §12)"
import sys, yaml
sys.path.insert(0, "scripts")
import served_capability_run as scr
cfg = yaml.safe_load(open(sys.argv[1]))
ev = scr._tok_evidence(cfg["model"], cfg["model_revision"])
if not ev["serve_header_ids"]:
    raise SystemExit("served-evidence pre-flight: the derived header is EMPTY")
print(f"served-evidence pre-flight OK: H={ev['serve_header_len']} header_ids={ev['serve_header_ids']} "
      f"template_kwargs={ev['template_kwargs']} template_hash={ev['served_template_hash']}")
PYTOK

ar_bank_view "$MPY"                            # builds/verifies the view; exports TT_TWO_SI_DEFENSES_DIR, ALL_CELLS
CELLS=("${ALL_CELLS[@]}"); NCELLS=${#CELLS[@]}
[ "$NCELLS" -eq 2 ] || f22_fatal "expected [none $AR_CELL], got ${CELLS[*]}"

# the fused Mamba kernels of the SERVED path: qwen35_4b is a hybrid, and without them the
# engine silently takes a slow path — which is also the quiet one
"$SPY" - <<'PYK' || f22_fatal "the serving venv lacks vLLM's fused Mamba/hybrid ops on $(hostname)"
import importlib, sys
bad = []
for m in ("vllm.model_executor.layers.mamba.ops.ssd_combined",
          "vllm.model_executor.layers.mamba.ops.causal_conv1d"):
    try: importlib.import_module(m)
    except Exception as e: bad.append(f"{m}: {type(e).__name__}: {str(e)[:120]}")
import vllm; print(f"vllm {vllm.__version__}; fused mamba ops: " + ("ALL PRESENT" if not bad else " | ".join(bad)))
sys.exit(1 if bad else 0)
PYK

if command -v ruff >/dev/null 2>&1; then
  ruff check --select F821 --no-cache scripts/arditi/ scripts/served_capability_run.py scripts/jailbreak_run.py \
    || f22_fatal "undefined name(s) — refusing before the standup"
  f22_log "static check (F821) clean"
else
  f22_log "ruff absent on this node; the F821 pass is skipped (the submitter runs it, and THAT is the gate)"
fi

# THE STANDUP'S ARGV, in f22_standup's own grammar. Two things ride beside the positionals, and they
# travel by DIFFERENT roads:
#   * `--enable-log-requests` is a FLAG and goes through F22_STANDUP_EXTRA, the Bash ARRAY common.sh
#     appends VERBATIM to mt_serving_up.py's argv -- the forwarding path every opted-in lane uses
#     (target_sampled.sbatch, asscreen_controls.sbatch, g8k_/m5t_standup_extra). It is what makes the
#     RESOLVED sampler a recorded fact: without it the engine's own SamplingParams never reaches any
#     file, and at 0.25.1 there is no other readback (the response body carries no sampling fields,
#     and the generation_config override line is emitted only when such a file exists -- this target
#     has none). This lane builds the array HERE and nowhere else (pinned by test), THROUGH common.sh's
#     f22_standup_extra_reset: the helper keeps the LAUNCH-TIME HOOK -- a scalar F22_STANDUP_EXTRA a
#     batch job inherits from the submitter's exported environment -- in front of this lane's flag
#     (a bare `=(` clobbered it on the sampled path; a bare `+=` first touch keeps an exported-EMPTY
#     scalar as an empty argv element the parser rejects), and refuses BY NAME a `--serving-path`
#     element riding the array (f22_init runs the same check before the serving path is recorded; the
#     engine pin travels as F22_SERVING_PATH, never as an argv hook). This lane's flag is not that
#     name, so the check passes on it (run, not reasoned: tests/test_ar_cell_standup_argv.py drives
#     this line through the real helpers under every inherited shape and the real parser).
#   * f22_standup's 7th POSITIONAL is vLLM's --max-logprobs, an INTEGER. An earlier version of this
#     script put the flag there, which rendered `--max-logprobs --enable-log-requests`: argparse
#     refused the int before any model loaded, and every standup of this lane (both modes) died with
#     its allocation wasted. The slot now carries the ONE number the MMLU leg needs on the server AND
#     on the driver (the wave's controls rule, target_static.sbatch): the driver's own MMLU_TOP_LOGPROBS,
#     read by IMPORT under $MPY -- never a literal, the engine refuses a request above its cap
#     ("Requested sample logprobs of 64, which is greater than max allowed: 20", 447405) -- handed to
#     standup 1 (which serves MMLU) and to the MMLU driver's --top-logprobs. Standup 2 serves the XSTest
#     re-draws only; every probe its drivers fire asks LOGPROB_LIVENESS_TOP_LOGPROBS / HARMFUL_PROBE_TOP_LOGPROBS
#     (20, the engine default), so its slot stays EMPTY. An unreadable constant is a refusal, never a default.
# THE RAW UNIT'S OBJECT IS PINNED IN THIS LANE'S OWN PINS FILE, NOT THE WAVE'S — so the standup is
# handed the digest instead of being sent to look it up. Job 531726 died here, 26 s into its first
# standup, and the refusal was correct: `mt_serving_up.view_manifest_digests` derives each served
# cell's expected sha from the view manifest + `configs/sdfbase_full22_pins.json`, and this arm's
# `ar_c1__arditi_allpos` is not in the WAVE's pins (it is in `configs/arditi_allpos_pins.json`,
# this lane's own authority) — "refusing an unpinned object". The certification view never hit it
# because it is a wave-bank tenant whose manifest entry uses the CROSS-BANK pin form, which that
# function reads from the roster's named authority instead; this arm's own bank view uses the
# staged form, so the arm's standup path met the wave-pins lookup for the first time here.
# `--expect-object-sha256 <cell>=<sha>` is the documented seam for precisely this, and it is NOT a
# relaxation: the sha handed over is the one `ar_bank_view` computed by hashing the very
# `object.pt` about to be served, already bound three independent ways to the REQUESTED rung's pin
# (bytes, the provenance's `c` canonicalised through the rung grammar, and the view symlink's
# realpath) and held equal to `AR_PIN_OBJECT_SHA256` before any evidence is stamped — and
# `mt_serving_up` still re-hashes the served bytes against it and refuses a drift. The lane's own
# authority is strictly stronger here than the wave's lookup would have been.
# Production is untouched: the array is empty for the lens unit, so its call renders byte for byte
# as before.
AR_OBJECT_DIGEST=()
[ "$AR_MODE" = native ] && AR_OBJECT_DIGEST=(--expect-object-sha256 "${AR_CELL}=${AR_SERVED_OBJECT_SHA256}")
f22_standup_extra_reset --enable-log-requests ${AR_OBJECT_DIGEST[@]+"${AR_OBJECT_DIGEST[@]}"}
CTL_TOP_LOGPROBS=$("$MPY" -c 'import importlib.util,sys; s=importlib.util.spec_from_file_location("scr", sys.argv[1]); m=importlib.util.module_from_spec(s); sys.modules["scr"]=m; s.loader.exec_module(m); print(int(m.MMLU_TOP_LOGPROBS))' scripts/served_capability_run.py) \
  || f22_fatal "cannot read MMLU_TOP_LOGPROBS from scripts/served_capability_run.py under $MPY (the wave's controls phase reads it the same way) -- the server's --max-logprobs and the MMLU driver's --top-logprobs must be ONE number, so neither is defaulted"
[[ "$CTL_TOP_LOGPROBS" =~ ^[1-9][0-9]*$ ]] || f22_fatal "MMLU_TOP_LOGPROBS read as '$CTL_TOP_LOGPROBS', not a positive integer"
f22_log "standup argv: positional 7 (--max-logprobs) = $CTL_TOP_LOGPROBS on standup 1 (= the MMLU driver's --top-logprobs), empty on standup 2; F22_STANDUP_EXTRA=[${F22_STANDUP_EXTRA[*]}]"

SERVE_DIR="$WAVE_ROOT/serving/cell_${TAG}_t${TRIAL}_${SLURM_JOB_ID:-manual}"
PRIVATE_ROOT="$WAVE_ROOT/transcripts_results"
CTL_ROOT="$WAVE_ROOT/controls/${TAG}_t${TRIAL}"
REC_DIR="$WAVE_ROOT/cells/${TAG}_t${TRIAL}"
[ "$LEGS" = all ] || REC_DIR="$REC_DIR/legs_${LEGS}_${SLURM_JOB_ID:-manual}"
mkdir -p "$SERVE_DIR/drivers" "$PRIVATE_ROOT" "$CTL_ROOT" "$REC_DIR"
chmod 700 "$PRIVATE_ROOT" 2>/dev/null || true

if [ "$PRECHECK" = precheck ]; then
  f22_log "PRECHECK OK for $TAG (legs=$LEGS): identity, config pair, decode-determined-by-request premise, venvs, served-evidence pre-flight, bank view ($TT_TWO_SI_DEFENSES_DIR), cells ${CELLS[*]}, served_layer $SERVED_LAYER, standup argv (--max-logprobs $CTL_TOP_LOGPROBS; extras [${F22_STANDUP_EXTRA[*]}]). Not run: the standup or any leg."
  exit 0
fi

# --- timings: one row per component, written as we go so a killed job still reports the parts ---
TIMINGS="$REC_DIR/timings.json"
python3 -c "import json,sys;open(sys.argv[1],'w').write(json.dumps({'tag':sys.argv[2],'trial':int(sys.argv[3]),'job':sys.argv[4],'legs':[]},indent=2))" \
  "$TIMINGS" "$TAG" "$TRIAL" "${SLURM_JOB_ID:-manual}"
mark(){ # $1 = leg name, $2 = t_start, $3 = rc, $4.. = note
  local leg="$1" t0="$2" rc="$3"; shift 3
  python3 - "$TIMINGS" "$leg" "$t0" "$(date +%s)" "$rc" "$*" <<'PY'
import json, sys
p, leg, t0, t1, rc, note = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
d = json.load(open(p))
d["legs"].append({"leg": leg, "seconds": t1 - t0, "minutes": round((t1 - t0) / 60, 2),
                  "rc": rc, "note": note})
d["total_seconds"] = sum(l["seconds"] for l in d["legs"])
json.dump(d, open(p, "w"), indent=2)
PY
  f22_log "TIMING $leg: $(( $(date +%s) - t0 ))s rc=$rc $*"
}

TARGET_PID=""; STANDUP_STOP=""; declare -a DRIVER_PIDS=(); declare -A DRIVER_TAG=(); RC=0; TOTAL_FAIL=0
# THE TRAP IS SPLIT, and it is load-bearing: a shared EXIT+SIGTERM handler whose body
# group-kills re-enters itself until bash's stack overflows -> SIGSEGV -> sacct `0:11`, and
# because f22_result is the last statement the recursion ate the RESULT backstop every time.
# EXIT records only; the signal slot ignores TERM before the group kill so it cannot recurse.
HANDLED=0
finish(){ [ "$HANDLED" -eq 1 ] && return; HANDLED=1
  f22_result "cell $TAG t$TRIAL rc=${RC:-?} failed_drivers=${TOTAL_FAIL}"; }
cleanup(){ HANDLED=0; RC=143; trap '' SIGTERM
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  kill -TERM -$$ 2>/dev/null; wait; finish; }
trap finish EXIT
trap cleanup SIGTERM SIGINT SIGQUIT
TARGET_GPU="$CUDA_VISIBLE_DEVICES"

# ================================================================= THE ONE STANDUP
T0=$(date +%s)
# `--enable-log-requests` rides F22_STANDUP_EXTRA (declared above, with why); positional 7 is the
# server's --max-logprobs = the MMLU driver's --top-logprobs, ONE number read from the driver by import
f22_standup attacks "$SERVE_DIR/cell" "" families_served:short "" "" "$CTL_TOP_LOGPROBS" \
  || { RC=1; mark standup "$T0" 1 "standup FAILED"; exit 1; }
mark standup "$T0" 0 "geometry ${TARGET_S}x${TARGET_L} proxy=$PROXY_URL"
VLLM_LOG="$SERVE_DIR/cell/vllm_server.log"
S1_S="$TARGET_S"; S1_L="$TARGET_L"; S1_NODE="$(hostname)"   # standup 1's geometry: what the trade was measured on

# WHAT THE ENGINE ACTUALLY DID, after the standup and before any traffic: the mode's evidence
# check (production: the serving-commit assertion already made at init; native: the endpoint's
# serving-path selection, its graphs-on attestation and the residual slot's own records), then the
# hooked-layer assertion in the mode's own SCHEMA — the lens records it under
# steering_attestation.worker_reports, the native engine under the residual slot's boundaries, and
# reading one for the other would refuse a healthy server.
ar_assert_serving_evidence "$AR_MODE" "$SERVE_DIR/cell/endpoint.json" "$SERVED_LAYER" \
  || { RC=1; f22_log "serving-evidence assertion FAILED — refusing to generate"; exit 1; }
# the edit lands at its ONE fitted layer and nowhere else, from the server's OWN attestation
ar_assert_hook_layers_in "$AR_MODE" "$SERVE_DIR/cell/endpoint.json" "$SERVED_LAYER" \
  || { RC=1; f22_log "hook-layer assertion FAILED — refusing to generate"; exit 1; }
# THE STORE'S POINTER TO THIS STANDUP'S EVIDENCE (round 3, item 2): in native mode the record the evidence check above
# persisted (stamped with its identity; re-hashed here) is named in SISTEER_SERVING_EVIDENCE_RECORD, and every
# served_target.json the static drivers write below folds its sha + path in, so the grader follows the STORE to the
# record rather than trusting a path supplied by hand. Production mode UNSETS it: no record exists, nothing is folded,
# every shard is byte-identical. The 8192 re-draw standup serves XSTest only (no served_target.json), so this is the
# one binding site.
ar_bind_serving_evidence_record "$AR_MODE" "$SERVE_DIR/cell/endpoint.json"

# the resolved sampler, read once at standup (before any leg) so a divergence is caught early;
# re-read at the end over the whole log with the client rows to compare against
python3 scripts/arditi/record_resolved_sampler.py --vllm-log "$VLLM_LOG" \
  --out "$REC_DIR/resolved_sampler_standup.json" || true

FAMS=(pasttense codeattack renellm grok420)
XS_LAST=none                                 # LEG 2b keys on this; `none` when the controls legs did not run
f22_log "legs=$LEGS (records -> $REC_DIR)"
# ================================================================= LEG 1F — MMLU over the WHOLE pinned pool (legs=mmlu_full ONLY)
# mmlu-full-1531: the SAME letter-prefill read as LEG 1 — the same standup, the same argv — over
# every one of the 1,531 pinned items instead of the s2parity 200-item head (which is its prefix, item for item). Its
# own out-root $CTL_ROOT/mmlu_full, NEVER $CTL_ROOT/mmlu (the published 200-item stores, which a re-run would overwrite
# in place), and the driver's --mmlu-full — read from the contracts by import, never retyped — which keys the store
# apart (resolved_config.mmlu.item_set) so the two widths can never pool or resume into each other.
MMLU_FULL_RC=0
if [ "$RUN_MMLU_FULL" = 1 ]; then
T0=$(date +%s)
MMLU_FULL_FLAG=$("$MPY" -c 'from sisteer.capeval.tinycap import contracts as tc; print(tc.FLAG_MMLU_FULL)') \
  || f22_fatal "cannot read contracts.FLAG_MMLU_FULL under $MPY — this checkout predates the full-pool opt-in (mmlu-full-1531)"
for cell in "${CELLS[@]}"; do
  f22_stop_requested && break
  ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" scripts/served_capability_run.py generate \
      --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --config "$CFG" \
      --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$CTL_ROOT/mmlu_full" \
      --controls mmlu "$MMLU_FULL_FLAG" --trial "$TRIAL" --concurrency "$TARGET_S" \
      --top-logprobs "$CTL_TOP_LOGPROBS" --vllm-log "$VLLM_LOG" ) \
    > "$SERVE_DIR/drivers/mmlu_full__${cell}.log" 2>&1 || { MMLU_FULL_RC=$?; TOTAL_FAIL=$((TOTAL_FAIL+1));
      f22_log "MMLU-full driver FAILED rc=$MMLU_FULL_RC for $cell"; tail -25 "$SERVE_DIR/drivers/mmlu_full__${cell}.log"; }
done
mark mmlu_full "$T0" "$MMLU_FULL_RC" "letter prefill over the whole 1,531-item pinned pool, no decode, both cells"
fi   # RUN_MMLU_FULL
if [ "$RUN_CONTROLS" = 1 ]; then
# ================================================================= LEG 1 — MMLU (letter prefill, no decode)
# FIRST because it is minutes and it exercises a DIFFERENT path through the steered stack than
# the generation legs: a teacher-forced response prefill whose boundary must be declared on the
# lens route. The driver's own logprob positive control compares the steered letter logprobs at
# the scored position against the scale-0 twin's, against a noise floor measured by a same-leg
# repeat — under the DEFAULT bites policy that control is REQUIRED and an inert scored path
# ABORTS the leg rather than returning an accuracy. That is deliberate: an accuracy that matches
# baseline exactly is the signature of steering not being applied on this path at all, and it
# must fail loudly rather than be reported. SISTEER_BITES_POLICY is NOT exported here (the
# AlphaSteer lane sets harmful_continuous because its method nulls on benign input by
# construction; an additive refusal direction has no such claim).
MMLU_RC=0
if [ "$RUN_MMLU" = 1 ]; then
T0=$(date +%s)
for cell in "${CELLS[@]}"; do
  f22_stop_requested && break
  # CUDA_VISIBLE_DEVICES="" is the hard rule's STRICT-SUBSET case, not a violation of it: the rule forbids
  # WIDENING Slurm's value or hardcoding absolute indices, and empty is the strictest subset of any allocation.
  # It is needed rather than decorative -- served_capability_run.py is a pure HTTP client here (the proxy is stood
  # up by a SEPARATE job; $PROXY_URL is remote) but pulls torch transitively, and CUDA initialises lazily, so a
  # stray .cuda() on some code path would claim a GPU Slurm never allocated. Empty makes that fail loudly instead.
  ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" scripts/served_capability_run.py generate \
      --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --config "$CFG" \
      --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$CTL_ROOT/mmlu" \
      --controls mmlu --trial "$TRIAL" --concurrency "$TARGET_S" \
      --top-logprobs "$CTL_TOP_LOGPROBS" --vllm-log "$VLLM_LOG" ) \
    > "$SERVE_DIR/drivers/mmlu__${cell}.log" 2>&1 || { MMLU_RC=$?; TOTAL_FAIL=$((TOTAL_FAIL+1));
      f22_log "MMLU driver FAILED rc=$MMLU_RC for $cell"; tail -25 "$SERVE_DIR/drivers/mmlu__${cell}.log"; }
done
mark mmlu "$T0" "$MMLU_RC" "letter prefill, no decode, both cells"
fi   # RUN_MMLU


# ================================================================= LEG 2 — XSTest: the sampled over-refusal LADDER (rungs 512 + 2048 here; 8192 after the families)
# The canonical instrument, in full, through the PORTED canonical driver functions
# (sisteer.eval.sampled_controls, verbatim from sampled-canonical-main @ 90284b59; NOTES §11):
# rung 512 for every item; rung 2048 re-drawing ONLY the items 512 left at cap (same seed) plus a
# cell-independent floor sample of 200 below-cap items (the re-draw noise floor); rung 8192 likewise
# from 2048. A cell leaves the ladder when its previous rung's at-cap count is zero. That count is
# read by the producer's OWN reader (at_cap_controls.py: exit 2 is a REFUSAL — e.g. a store with no
# sampled decode record — and refuses the rung rather than counting 0). The config carries its own
# sampled_decode block, so the driver's vendor-preset default is not consulted and --greedy is never
# passed; the driver records decode_resolution in gen_stats.json either way.
#
# STANDUPS. Rungs 512 and 2048 both fit the ONE 4,864 window (2,048 cap + the controls' p99 prompt
# ~575), so they run in the SAME standup as MMLU and the families, here. Rung 8192 needs a 10,240
# window and is therefore a SECOND standup, taken only if at-cap survives 2048 and only AFTER the
# families have run on standup 1 — it re-draws already-measured items, it measures nothing new, and
# the defence/capability trade (MMLU vs the families vs XSTest@512/2048) is all on standup 1. The
# ladder record (xstest_ladder.json) says which rung ran on which standup.
FLOOR_K=200
ROOT_OF(){ echo "$CTL_ROOT/cap$1"; }          # controls rung roots: cap512 / cap2048 / cap8192
LADDER_JSON="$REC_DIR/xstest_ladder.json"; echo '{"control":"xstest","floor_k":'"$FLOOR_K"',"rungs":[]}' > "$LADDER_JSON"
at_cap_of(){   # $1 = rung root, $2 = cell -> the integer; return 2 on a REFUSAL (never a guessed 0)
  local out rc err="$REC_DIR/at_cap_controls.last.err"
  out=$(cd "$REPO_DIR" && "$MPY" scripts/sdfbase/full22/sampled/at_cap_controls.py --out-root "$1" --cell "$2" --control xstest 2>"$err"); rc=$?
  [ "$rc" = 0 ] || { f22_log "at_cap_controls.py REFUSED rc=$rc for $1/$2: $(tr '\n' ' ' < "$err" | head -c 400)"; return 2; }
  # with --control the reader prints the BARE integer (its line 213); without it, `<control> n_at_cap=K ...`.
  # Accept either shape, refuse anything else — a count this ladder cannot read is a refusal, never 0.
  out=$(echo "$out" | { grep -o "n_at_cap=[0-9]*" | head -1 | cut -d= -f2; } 2>/dev/null; echo "$out" | grep -E "^[0-9]+$" | head -1)
  out=$(echo "$out" | grep -E "^[0-9]+$" | head -1)
  [[ "$out" =~ ^[0-9]+$ ]] || { f22_log "at_cap_controls.py printed no readable count for $1/$2"; return 2; }
  echo "$out"
}
ladder_note(){   # k=v ... -> one record appended to xstest_ladder.json (rung / n_at_cap / standup as integers)
  "$MPY" - "$LADDER_JSON" "$@" <<'PYNOTE'
import json, sys
p = sys.argv[1]; d = json.load(open(p)); kv = dict(a.split("=", 1) for a in sys.argv[2:])
for k in ("rung", "n_at_cap", "standup"):
    if k in kv: kv[k] = int(kv[k])
d["rungs"].append(kv); json.dump(d, open(p, "w"), indent=2)
PYNOTE
}
xstest_run(){   # $1 = rung root, $2 = cell, $3 = rung, $4 = prev root ("" = base pass), $5 = vllm log -> driver rc
  # --top-logprobs is the SAME read constant as the MMLU leg's: an XSTest run never SENDS it (the driver's 64-wide
  # letter read and its logprob positive control run only with MMLU rows loaded), but the value is bound into the
  # store's resolved-config hash, so it is ONE number here too rather than a second literal
  local extra=()
  [ -n "$4" ] && extra=(--controls-cap "$3" --escalate-from-root "$4" --floor-sample "$FLOOR_K")
  # CUDA_VISIBLE_DEVICES="" is the hard rule's STRICT-SUBSET case, not a violation of it: the rule forbids
  # WIDENING Slurm's value or hardcoding absolute indices, and empty is the strictest subset of any allocation.
  # It is needed rather than decorative -- served_capability_run.py is a pure HTTP client here (the proxy is stood
  # up by a SEPARATE job; $PROXY_URL is remote) but pulls torch transitively, and CUDA initialises lazily, so a
  # stray .cuda() on some code path would claim a GPU Slurm never allocated. Empty makes that fail loudly instead.
  ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" scripts/served_capability_run.py generate \
      --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$2" --config "$CFG" \
      --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$1" \
      --controls xstest --trial "$TRIAL" --concurrency "$TARGET_S" \
      --top-logprobs "$CTL_TOP_LOGPROBS" --vllm-log "$5" "${extra[@]}" ) > "$SERVE_DIR/drivers/xstest${3}__${2}.log" 2>&1
}
# ---- rung 512: every cell, every item (standup 1)
T0=$(date +%s); XS_RC=0
for cell in "${CELLS[@]}"; do
  f22_stop_requested && break
  xstest_run "$(ROOT_OF 512)" "$cell" 512 "" "$VLLM_LOG" || { XS_RC=$?; TOTAL_FAIL=$((TOTAL_FAIL+1));
    f22_log "XSTest@512 driver FAILED rc=$XS_RC for $cell"; tail -25 "$SERVE_DIR/drivers/xstest512__${cell}.log"; }
done
mark xstest_512 "$T0" "$XS_RC" "sampled @512, both cells, standup 1"
# ---- rung 2048: only the cells whose 512 store left items at cap (standup 1, same window)
declare -A ON_LADDER=(); for cell in "${CELLS[@]}"; do ON_LADDER[$cell]=1; done
xstest_rung(){   # $1 = rung, $2 = prev rung, $3 = vllm log, $4 = standup number -> sets NRUNG; runs the rung for the cells still on the ladder
  local RUNG="$1" PREV="$2" LOG="$3" SU="$4" n cell rc=0
  NRUNG=0
  for cell in "${CELLS[@]}"; do
    [ -n "${ON_LADDER[$cell]:-}" ] || continue
    if n=$(at_cap_of "$(ROOT_OF "$PREV")" "$cell"); then
      ladder_note cell="$cell" rung="$PREV" n_at_cap="$n" root="$(ROOT_OF "$PREV")" standup="$( [ "$PREV" -le 2048 ] && echo 1 || echo 2 )"
      f22_log "at-cap @$PREV xstest/$cell: $n"
      if [ "$n" -gt 0 ]; then NRUNG=$((NRUNG+1)); else ON_LADDER[$cell]=""; fi
    else
      f22_log "REFUSING rung $RUNG for $cell: the at-cap count at $PREV was refused — a rung cannot proceed on a guess"
      TOTAL_FAIL=$((TOTAL_FAIL+1)); ON_LADDER[$cell]=""
    fi
  done
  [ "$NRUNG" -gt 0 ] || { f22_log "XSTest ladder: every cell left at rung $PREV — no $RUNG rung"; return 0; }
  f22_stop_requested && return 0
  local T1; T1=$(date +%s)
  for cell in "${CELLS[@]}"; do
    [ -n "${ON_LADDER[$cell]:-}" ] || continue
    xstest_run "$(ROOT_OF "$RUNG")" "$cell" "$RUNG" "$(ROOT_OF "$PREV")" "$LOG" || { rc=$?; TOTAL_FAIL=$((TOTAL_FAIL+1));
      f22_log "XSTest@$RUNG driver FAILED rc=$rc for $cell"; tail -25 "$SERVE_DIR/drivers/xstest${RUNG}__${cell}.log"; }
  done
  mark "xstest_$RUNG" "$T1" "$rc" "rung $RUNG from $PREV, $NRUNG cell(s), floor k=$FLOOR_K, standup $SU"
}
xstest_rung 2048 512 "$VLLM_LOG" 1
XS_LAST=2048; [ "${NRUNG:-0}" -gt 0 ] || XS_LAST=512
fi   # RUN_CONTROLS

# NOTE ON POSITION: this leg sits OUTSIDE the RUN_CONTROLS guard on purpose. It is no longer part
# of the per-rung set (the ladder measures static + XSTest + MMLU), so `legs=orbench` must be able to
# run it ALONE at the operating point without dragging the XSTest ladder along — which is exactly what
# RUN_CONTROLS would do. It still runs on standup 1, which is what keeps it comparable with whatever
# else that job measures.
# ================================================================= LEG 1b — OR-Bench-Hard: the sampled over-refusal LADDER (rungs 512 + 2048 here; 8192 on a second standup)
# THE SECOND OVER-REFUSAL INSTRUMENT, measured ONLY at the selected operating point (`legs=orbench`;
# the rule of 2026-09-16, see RUN_ORBENCH above). It is the XSTest ladder's instrument with
# or_bench_hard_1k (1,319 items) in XSTest's place, through the ONE sourced body the AlphaSteer and
# prefill-Arditi lanes already run for this control (scripts/arditi/sbatch/controls_ladder.sh, ported
# to this tree from the as-bon-leg lineage): rung 512 for every item; rung 2048 re-drawing ONLY the
# items 512 left at cap (same per-item seed) plus the cell-independent floor sample of FLOOR_K
# below-cap items; rung 8192 likewise from 2048, on a SECOND standup with a 10,240 window. EXIT RULE
# le_1pct — a cell leaves the ladder when its at-cap count is <= floor(n_total/100), i.e. 13 of
# 1,319 — NOT the zero rule the XSTest leg above runs; the ladder record (or_bench_hard_1k_ladder.json)
# names which, so no reader takes the two ladders for one procedure. That is the rule every OR-Bench
# record in this campaign was read under, and the grader (scripts/arditi/grade_arditi_orbench.py)
# reads the DEEPEST rung both arms hold and expects 2048: the first shape of this leg — the 512 base
# pass alone, no 2048 rung — would have filed the base pass as a recorded deviation, censored by the
# ~1 in 5 items a 512 cap truncates on this model, which is exactly what the ladder exists to defeat.
#
# ON STANDUP 1, WITH WHATEVER ELSE THE JOB MEASURES, because that is what makes the reading a TRADE:
# the defence and capability sides are only comparable if they were measured on the same served
# state, and a second standup is a second engine that inherits nothing from the first's verdict.
# The 8192 rung, when the 1% rule demands it, re-draws already-measured items on a second standup
# exactly as the XSTest 8192 rung does (LEG 2b), with the SAME pair of post-standup checks in the
# same mode. Both cells of the pair are always drawn in the same job: the `none` anchor a delta is
# read against is THIS standup's, never another job's.
#
# SAME DRIVER, SAME SHAPE AS MMLU / XSTest: one invocation per cell per rung, its own out-root (so
# the store keys apart), the block's own controls cap through the config's sampled_decode, and the
# SAME ONE --top-logprobs constant (CTL_TOP_LOGPROBS, read from the driver by import above): an
# OR-Bench run never SENDS that value, but served_capability_run binds it into the store's
# resolved-config hash, so the sourced body takes it from this caller rather than carrying a second
# literal. It is a JUDGED control (served_capability_judge.JUDGED_CONTROLS = xstest, or_bench_hard_1k):
# this leg GENERATES; the harness judge files judge_summary.json / judge_items.jsonl later, off-cluster.
#
# THE OUT-ROOT IS `orb_cap<CAP>`, WHICH IS THE GRADER'S LAYOUT, NOT A NAME I CHOSE: the grader
# discovers stores by walking <tag>_t0/orb_cap*/cells/* (`ROOT_PREFIX`, a module constant with no
# CLI override) and reads each rung's CAP out of the directory name. Another prefix is invisible to
# it, and the XSTest prefix (cap<N>) would write two controls into one published measurement.
if [ "$RUN_ORBENCH" = 1 ]; then
  CTL_CONTROL=or_bench_hard_1k; CTL_ROOT_PREFIX=orb_cap; CTL_EXIT_RULE=le_1pct; FLOOR_K=200
  # shellcheck source=/dev/null
  source "$REPO_DIR/scripts/arditi/sbatch/controls_ladder.sh" || { RC=1; f22_log "cannot source controls_ladder.sh"; exit 1; }
  ctl_leg_standup1
  if ctl_needs_8192 && ! f22_stop_requested; then
    T0=$(date +%s)
    f22_standup_stop
    # the campaign's 8192-rung window for this model (common_sampled.sh f22s_rung_window): 10,240; S = budget // L.
    # A SECOND STANDUP IS A SECOND ENGINE and inherits nothing from the first's verdict, so it gets the SAME pair
    # of post-standup checks, in the same mode, as standup 1 and as the XSTest 8192 standup (LEG 2b).
    if f22_standup attacks "$SERVE_DIR/cell_orb_rung8192" "" families_served:short 10240 "" "" \
       && ar_assert_serving_evidence "$AR_MODE" "$SERVE_DIR/cell_orb_rung8192/endpoint.json" "$SERVED_LAYER" \
       && ar_assert_hook_layers_in "$AR_MODE" "$SERVE_DIR/cell_orb_rung8192/endpoint.json" "$SERVED_LAYER"; then
      VLLM_LOG2="$SERVE_DIR/cell_orb_rung8192/vllm_server.log"; S2_S="$TARGET_S"; S2_L="$TARGET_L"
      mark standup2_orb_rung8192 "$T0" 0 "geometry ${TARGET_S}x${TARGET_L} — a SECOND standup for the OR-Bench 8192 re-draws only; the pair was measured on standup 1"
      ctl_rung 8192 2048 "$VLLM_LOG2" 2 && [ "${CTL_NRUNG:-0}" -gt 0 ] && CTL_LAST=8192
    else
      RC=1; mark standup2_orb_rung8192 "$T0" 1 "second standup FAILED — the OR-Bench 8192 rung did not run; 2048 is the last rung"
    fi
  fi
  ctl_final_note
fi   # RUN_ORBENCH

if [ "$RUN_STATIC" = 1 ]; then
# ================================================================= LEG 3 — the four single-shot families (sampled, 4096)
# The long pole. Sharded and queued so ~the attested width is in flight; each driver holds one
# request. DRIVER_CAP defaults to the wave's 96 (a static driver's import baseline is ~0.33 GB RSS,
# so this many processes sit well inside the 200G host cgroup); positional 7 lowers it for a
# completion pass whose probes must run alone (header).
T0=$(date +%s)
MAX_DRIVERS=$(( TARGET_S < DRIVER_CAP ? TARGET_S : DRIVER_CAP ))
export SISTEER_HELDOUT_CORPUS_DIR="${SISTEER_HELDOUT_CORPUS_DIR:-$SHARED/exp/heldout_corpora}"
# renellm replays this cluster's copy when there is one, else the paper's corpus as the checkout ships it
[ -s "$SISTEER_HELDOUT_CORPUS_DIR/renellm.jsonl" ] || [ -s "$REPO_DIR/data/heldout_corpora/renellm.jsonl" ] \
  || f22_log "WARNING: no renellm corpus at $SISTEER_HELDOUT_CORPUS_DIR or in $REPO_DIR/data/heldout_corpora — renellm will SKIP LOUDLY and write nothing"
f22_log "phase STATIC: families=${FAMS[*]} cells=$NCELLS nshards=$NSHARDS cap=4096 width=$TARGET_S max_drivers=$MAX_DRIVERS"
for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do for s in $(seq 0 $((NSHARDS-1))); do
  f22_stop_requested && { f22_log "STOP present — not launching more drivers"; break 3; }
  f22_queue_launch "${fam}__${cell}__s$(printf '%02d' "$s")" "$MPY" scripts/jailbreak_run.py generate \
    --config "$CFG" --attack "$fam" --defense "$cell" --confirmatory --trial "$TRIAL" \
    --shard "$s" --nshards "$NSHARDS" --private-root "$PRIVATE_ROOT" \
    --target-engine served --target-proxy-url "$PROXY_URL" \
    --target-routing-table "$TABLE" --target-max-model-len "$TARGET_L"
  sleep 1
done; done; done
f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
mark static "$T0" "$NFAIL" "4 families x $NCELLS cells x $NSHARDS shards @4096, $NFAIL failed drivers"
fi   # RUN_STATIC

# ================================================================= LEG 2b — XSTest rung 8192 (second standup, only if at-cap survived 2048)
if [ "$XS_LAST" = 2048 ]; then
  NEED8192=0
  for cell in "${CELLS[@]}"; do
    [ -n "${ON_LADDER[$cell]:-}" ] || continue
    n=$(at_cap_of "$(ROOT_OF 2048)" "$cell" 2>/dev/null) && [ "$n" -gt 0 ] && NEED8192=1
  done
  if [ "$NEED8192" = 1 ] && ! f22_stop_requested; then
    T0=$(date +%s)
    f22_standup_stop
    # the campaign's 8192-rung window for this model (common_sampled.sh f22s_rung_window): 10,240; S = budget // L
    # A SECOND STANDUP IS A SECOND ENGINE and inherits nothing from the first's verdict, so it gets
    # the SAME pair of post-standup checks, in the same mode — after EACH successful standup,
    # including the optional 8192-cap second one.
    # positional 7 EMPTY: this standup serves the XSTest re-draws only (no MMLU), and every probe its drivers fire
    # asks at most the engine's default 20 logprobs; `--enable-log-requests` still rides F22_STANDUP_EXTRA (above)
    if f22_standup attacks "$SERVE_DIR/cell_rung8192" "" families_served:short 10240 "" "" \
       && ar_assert_serving_evidence "$AR_MODE" "$SERVE_DIR/cell_rung8192/endpoint.json" "$SERVED_LAYER" \
       && ar_assert_hook_layers_in "$AR_MODE" "$SERVE_DIR/cell_rung8192/endpoint.json" "$SERVED_LAYER"; then
      VLLM_LOG2="$SERVE_DIR/cell_rung8192/vllm_server.log"; S2_S="$TARGET_S"; S2_L="$TARGET_L"
      mark standup2_rung8192 "$T0" 0 "geometry ${TARGET_S}x${TARGET_L} — a SECOND standup for the 8192 re-draws only; the trade was measured on standup 1"
      xstest_rung 8192 2048 "$VLLM_LOG2" 2
      XS_LAST=8192
    else
      RC=1; mark standup2_rung8192 "$T0" 1 "second standup FAILED — the 8192 rung did not run; 2048 is the last rung"
    fi
  fi
fi
# the residual at the last rung that ran (a non-zero count at 8192 is a genuine non-termination residual).
# Only when the ladder ran: with legs=static XS_LAST is `none` and ROOT_OF is not even defined (job
# 488315 printed `ROOT_OF: command not found` twice here — harmless, the && chain fell through, but noise).
[ "$RUN_CONTROLS" = 1 ] && for cell in "${CELLS[@]}"; do
  n=$(at_cap_of "$(ROOT_OF "$XS_LAST")" "$cell" 2>/dev/null) && { ladder_note cell="$cell" rung="$XS_LAST" n_at_cap="$n" root="$(ROOT_OF "$XS_LAST")" final=1; f22_log "at-cap @$XS_LAST (final) xstest/$cell: $n"; }
done
f22_standup_stop

# ================================================================= the record
T0=$(date +%s)
# at-cap per (family, cell) — the truncation column, from the producer's own reader (exit 2 is a
# REFUSAL, e.g. a greedy store, and is recorded as such rather than counted as zero)
[ "$RUN_STATIC" = 1 ] && for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do
  n=$("$MPY" scripts/sdfbase/full22/sampled/at_cap_static.py --private-root "$PRIVATE_ROOT" \
        --private-run "$PRIVATE_RUN" --attack "$fam" --defense "$cell" 2>"$REC_DIR/at_cap.err") \
    && f22_log "at_cap @4096 $fam/$cell: $n" \
    || f22_log "at_cap REFUSED for $fam/$cell: $(tr '\n' ' ' < "$REC_DIR/at_cap.err" | head -c 300)"
done; done

# THE RESOLVED SAMPLER, over the whole log, compared against every client row that recorded one.
# Exit 1 = DIVERGED, 3 = UNREADABLE; neither fails the job (the generations are still good and
# the record says so), but both are reported loudly and ride the RESULT line.
SAMPLER_RC=0
"$MPY" scripts/arditi/record_resolved_sampler.py --vllm-log "$VLLM_LOG" \
  $([ "$RUN_STATIC" = 1 ] && for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do
      for f in "$PRIVATE_ROOT/$PRIVATE_RUN/$fam/$cell"/shard*/responses.jsonl; do
        [ -f "$f" ] && printf -- '--sent-static %s ' "$f"; done; done; done) \
  $([ "$RUN_CONTROLS" = 1 ] && for f in "$CTL_ROOT"/cap*/cells/*/gen_stats.json; do
      [ -f "$f" ] && printf -- '--sent-controls %s ' "$f"; done) \
  $([ "$RUN_ORBENCH" = 1 ] && for f in "$CTL_ROOT"/orb_cap*/cells/*/gen_stats.json; do
      [ -f "$f" ] && printf -- '--sent-controls %s ' "$f"; done) \
  ${VLLM_LOG2:+--vllm-log "$VLLM_LOG2"} \
  --out "$REC_DIR/resolved_sampler.json" || SAMPLER_RC=$?
mark record "$T0" "$SAMPLER_RC" "at-cap counts + resolved-sampler record (rc 1=DIVERGED 3=UNREADABLE)"

# cell_facts.json — the identity a reader needs beside any number this cell produced, INCLUDING
# the three declared deviations, which are what make this an "Arditi baseline" honestly.
# THE RECORD IS KEYED ON THE UNIT: the dose under ITS name (`alpha` for the residual-norm
# fraction -- today's key, byte for byte; `c` for the raw coefficient, never `alpha`), the kind and phase
# off the unit's record, the declared deviations THIS unit makes, and the serving_impl_commit the
# manifest actually records (ar_resolve_config verified it: the pinned reference commit for the lens
# lane, this checkout's for the native arm). A raw-coefficient rung filed as "prefill, static_r, alpha"
# would be a lie in every reader downstream.
"$MPY" - "$REC_DIR/cell_facts.json" "$TAG" "$TRIAL" "$STRENGTH" "$SERVED_LAYER" "$CFG" \
        "$MANIFEST" "$TT_TWO_SI_DEFENSES_DIR" "$PRIVATE_RUN" "$CTL_ROOT" "$SHA" \
        "$AR_MANIFEST_SERVING_IMPL_COMMIT" "$S1_S" "$S1_L" "${S2_S:-}" "${S2_L:-}" "${S1_NODE:-}" "$LEGS" \
        "$REC_DIR/resolved_sampler.json" "$DRIVER_CAP" "$AR_UNIT" <<'PY'
import json, sys, yaml, os
(out, tag, trial, alpha, layer, cfgp, man, view, prun, ctl, sha, impl, S, L, S2, L2, node, legs, rs_path,
 driver_cap, unit) = sys.argv[1:22]
sys.path.insert(0, "scripts/arditi")
import arditi_strength as st
spec = st.unit_spec(unit)
production = unit == st.DOSE_UNIT
# THE RESOLVED SAMPLER IS A RESULT (D281/D282): the engine-side confirmation of the vendor row
# rides in the cell record, per request class, so the battery can cite it beside every number.
rs = json.load(open(rs_path)) if os.path.exists(rs_path) else None
resolved_sampler = (None if rs is None else {
    "record": rs_path, "verdict": rs["verdict"], "why": rs["why"],
    "classes": {k: {"n": v["n"], "verdict": v["verdict"], "reference": v.get("reference"),
                    "distinct_resolved": v.get("distinct_resolved")}
                for k, v in (rs.get("classes") or {}).items()},
    "generation_config_override_present": any(g.get("present") for g in rs.get("generation_config_override", []))})
# WHY an MMLU store may have no gen_stats.json, STATED by the store rather than inferred from the
# absence (sisteer.capeval.unscoreable): per store under each MMLU leg's own root -- $CTL_ROOT/mmlu/cells/
# (the 200-item leg) and $CTL_ROOT/mmlu_full/cells/ (the whole-pool leg), reported separately under
# `legs` -- "complete" / "refused_unscoreable" (N of M scored, the items and the top-K cap named) /
# "undeterminable_items" (topk_undeterminable rows: an attempted run whose aggregate the driver refused) /
# "partial_no_record" / "never_ran". Read from each store's own unscoreable_items.json and rows.jsonl,
# never from timings.json or a driver log. An error here lands IN the file rather than costing cell_facts.json.
try:
    from sisteer.capeval.unscoreable import RECORD_NAME, STATUS_REFUSED, mmlu_stores_facts
    mmlu_stores = mmlu_stores_facts(ctl)
    # ...AND WHETHER THE WRITER HALF EXISTS ON THE TREE THAT WROTE THESE STORES. The reader above
    # tells the four statuses apart from what a store carries; whether a store CAN carry
    # unscoreable_items.json depends on the capability driver that wrote it. When the driver has
    # no write of that record, `refused_unscoreable` is unreachable and an unscoreable item
    # reads `partial_no_record` instead. DERIVED by reading the driver's own source, never a typed
    # boolean, so the declaration follows the tree rather than this comment.
    try:
        _drv_rel = "scripts/served_capability_run.py"
        _writes = "unscoreable" in open(_drv_rel, encoding="utf-8").read()
        mmlu_stores["record_writer"] = {
            "driver": _drv_rel, "record": RECORD_NAME, "writes_record": bool(_writes),
            "consequence": (None if _writes else
                            "this tree's driver writes no " + RECORD_NAME + ", so the status "
                            + repr(STATUS_REFUSED) + " is UNREACHABLE here: an item the server's "
                            "top-K could not score leaves partial_no_record, with the scored count "
                            "read off rows.jsonl and no in-store statement of which item or which "
                            "cap. A DECLARED ABSENCE, not an inference from a missing file")}
    except OSError as e:  # noqa: BLE001 — the declaration says it could not be made, never guesses
        mmlu_stores["record_writer"] = {"error": f"{type(e).__name__}: {e}"}
except Exception as e:  # noqa: BLE001 — visible in the record, never a lost cell_facts.json
    mmlu_stores = {"error": f"{type(e).__name__}: {e}"}


def _sha256_file(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _sampler_departure(cfg):
    """The block's sampler against the canonical presets yaml row for this target: None when they
    agree, else what departs, by how much, and whether the departure is SHARED with the arm this
    baseline is compared against (read from the block's own `source`, which the generator wrote)."""
    import re as _re
    sd = cfg["sampled_decode"]; src = sd.get("source", "")
    m = _re.search(r"DEPARTS from ([^\s]+)'s canonical row (\{[^}]*\})", src)
    if not m:
        return None
    keys = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
    return {"canonical_source": m.group(1), "canonical_row": m.group(2),
            "sent": {k: sd[k] for k in keys if k in sd},
            "shared_with_the_compared_arm": "AlphaSteer" in src and "BOTH run" in src,
            "why": src[src.index("DEPARTS from"):][:700]}


legs_1 = {"all": ["mmlu", "xstest_512", "xstest_2048", "static families"],
          "controls": ["mmlu", "xstest_512", "xstest_2048"],
          "static": ["static families"],
          "xstest": ["xstest_512", "xstest_2048"],
          "orbench": ["or_bench_hard_1k_512", "or_bench_hard_1k_2048"],
          "mmlu_full": ["mmlu_full"]}[legs]
cfg = yaml.safe_load(open(cfgp))
record = {
  "tag": tag, "trial": int(trial), **({"alpha": float(alpha)} if production else {"c": float(alpha)}),
  "dose_unit": spec.unit, "phase": spec.served_phase,
  "served_layer": int(layer), "kind": spec.served_kind,
  "model": cfg["model"], "model_revision": cfg["model_revision"],
  "config": cfgp, "manifest": man, "bank_view": view,
  "private_run": prun, "controls_root": ctl,
  "repo_sha": sha, "serving_impl_commit": impl,
  "legs_run": legs,
  # how many static drivers were alive at once in THIS run: the liveness probe's noise ceiling is a
  # function of concurrent load (NOTES §18), so a verdict is only legible beside this number
  "static_driver_cap": int(driver_cap),
  "resolved_sampler": resolved_sampler,
  # completeness of every MMLU store under controls_root, from the stores themselves (above)
  "mmlu_stores": mmlu_stores,
  "geometry": {"standup_1": {"max_num_seqs": int(S), "max_model_len": int(L), "node": node,
                             "legs": legs_1},
               "standup_2": ({"max_num_seqs": int(S2), "max_model_len": int(L2),
                              "legs": [("or_bench_hard_1k_8192" if legs == "orbench" else "xstest_8192")
                                       + " (re-draws of at-cap items only)"]} if S2 else None),
               "trade_measured_on_one_standup": True,
               "why": "the defence/capability trade (MMLU vs the families vs XSTest@512/2048) is "
                      "only meaningful if both sides were measured on the same served state; the "
                      "8192 rung, when it runs, re-draws already-measured items on a second standup"},
  "sampled_decode": cfg["sampled_decode"],
  # the SAMPLER departure, when this target's decode is not the canonical yaml row (None when it
  # IS canonical — the caveat is eliminated, not labelled)
  "sampler_departure": _sampler_departure(cfg),
  # THE SERVING CONFIG'S OWN BYTES, stated rather than inferred: nemotron's cells sit on a newer
  # serve config than the other three targets' (today's re-bake), and the notebook must carry that
  # as provenance. The re-bake's changed keys are VERIFIED INERT for this path — recorded only as
  # `decode_caps`, with no reader for the climb keys — and that is written down here, not assumed.
  "serving_config_sha256": _sha256_file(cfgp),
  "serving_config_note": ("the sampled config's bytes; its base serve config's sha256 is in this "
                          "file's generated header. Nemotron's serve config is today's re-bake and "
                          "differs from the other targets' in keys VERIFIED INERT for this path "
                          "(recorded only as decode_caps; no reader for the climb keys)"),
}
if production:
  record.update({
  "declared_deviations": {
    "phase": "PREFILL-ONLY. Arditi's addition hook is a plain forward pre-hook, ungated on "
             "sequence length, so his intervention fires at prefill AND at every decode step. "
             "Serving prefill-only means generated tokens see the edit only through the steered "
             "prompt KV cache — half his intervention, chosen to phase-match the AlphaSteer lane "
             "on the shared dose pool.",
    "rendering": "this harness's ONE chat-template renderer, not his hardcoded per-family "
                 "template strings.",
    "unit": "the UNIT direction at a measured fraction of ||h||, not the raw difference of "
            "means whose magnitude is the distance between the class means."},
  "alpha_is_a_median": "alpha is the dose at the MEDIAN row, not a constant: ||h|| varies by "
                       "token and prompt, so the realised dose is alpha * ||h||_ref / ||h||. "
                       "qwen35_4b spreads 1.55x p05->p95, so the spread qualifier is not "
                       "mandatory on this target (it is on glm46vflash 8.78x and "
                       "nemotron3nano4b 2.76x).",
  })
else:
  # the raw unit CORRECTS the phase and unit deviations and keeps the rendering one; its dose is a raw
  # coefficient, so no median-alpha note applies -- the cross-model caveat and the alpha_equiv label do
  record.update({
    "declared_deviations": dict(st.RAW_DEVIATIONS_FROM_THE_PAPER),
    "cross_model_caveat": st.RAW_CROSS_MODEL_CAVEAT,
    "alpha_equiv_note": st.ALPHA_EQUIV_IS_ARITHMETIC,
  })
json.dump(record, open(out, "w"), indent=2)
print(f"cell facts -> {out}")
PY

# THE PHASE E DELIVERABLES, harvested into the rung's record. cell_facts.json
# above carries the cell's IDENTITY; these three are what the SELECTION reads, and none of them was
# ever written: the liveness quadruple (a rung without it is UNSELECTABLE — a dose admissible merely
# because it is INERT must never be selected), a durable copy of EVERY standup's positive controls
# (the successful one, the 8192 re-draw's, and any FAILED attempt's — which is exactly when liveness
# matters most, and the file is written before the failure path tears the server down), and the
# at-cap ITEM IDS (never a share: attribution is by item id ∩ the safe subset, and apportioning by
# a share against the wrong denominator produced a spurious result once already).
# ar_c1 (job 109166) finished before this call existed and was harvested POST HOC with the SAME
# script, so every rung's record has identical structure. Never fatal: a rung whose liveness cannot
# be read records the REASON in cell_facts.liveness.error rather than losing the rest of the record.
"$MPY" scripts/arditi/harvest_rung_facts.py --rec-dir "$REC_DIR" --serve-root "$SERVE_DIR" \
  --controls-root "$CTL_ROOT" --cell "$AR_CELL" --jobid "${SLURM_JOB_ID:-manual}" \
  || f22_log "WARNING: the Phase E harvest (liveness + at-cap item ids) did not complete — the rung's numbers are filed, but cell_facts.json may lack liveness; re-run harvest_rung_facts.py post hoc"

cp -f "$TIMINGS" "$FIGS_DIR/timings_${TAG}_t${TRIAL}${LEGS_SUFFIX}.json" 2>/dev/null || true
cp -f "$REC_DIR/resolved_sampler.json" "$FIGS_DIR/resolved_sampler_${TAG}_t${TRIAL}${LEGS_SUFFIX}.json" 2>/dev/null || true
RC=$(( TOTAL_FAIL > 0 ? 2 : 0 ))
f22_log "cell $TAG t$TRIAL done: $TOTAL_FAIL failed drivers; timings $TIMINGS"
"$MPY" -c "import json,sys;d=json.load(open(sys.argv[1]));print('[timings] '+' | '.join(f\"{l['leg']} {l['minutes']}m\" for l in d['legs'])+f\" | TOTAL {round(d['total_seconds']/60,2)}m\")" "$TIMINGS"
exit $RC
