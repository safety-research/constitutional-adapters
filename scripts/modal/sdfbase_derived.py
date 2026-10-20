"""sdfbase-derived-<sha16> — the round-2 DERIVED-OBJECTS Modal app
— a SEPARATE, versioned app on the shared ``twosi-fits`` volume for
every derived-objects GPU run, so the
wave-A fit app (``sdfbase-fits``) is never redeployed while a long fit is
live or resumable — a respawn under new code would resume a volume
checkpoint across commits and nothing would detect it. This app's runs are
short and never resume across versions; it is redeployable freely, and its
NAME carries the commit (``sdfbase-derived-<first 16 hex of HEAD>``, or
``SDFBASE_DERIVED_SHA16`` when set) so a call's provenance names the code.

Modes (run from the worktree at a pushed commit; ``modal deploy`` first for
durable spawns, or ``modal run`` for an attended one-shot):

THE PAIR. Every mode takes ``--mix-si`` / ``--ctl-si`` (defaults: the 100k
pair constv2-mix-100k / constv2ctl-100k the six published derived objects
came from — that path is BYTE-IDENTICAL to what it always was, pinned by
tests/test_sdfbase_derived_golden.py). The full − full decomposition
(the comparison of record since 2026-09-04) is ``--mix-si
constv2-mix-full --ctl-si constv2ctl2-full --capture-budget-tokens
100000`` — the derived ids follow the producers'
``<mix_si>__<arm>_minus_ctl*`` template (the control rides in the
provenance), the CE pools follow the control (CE_POOLS_BY_CTL), and the
capture set is a seeded 100k-official-token subsample of each 17k-doc pool
(psr_residual_distill.capture_subsample: an all-docs capture would make an
epoch ~470x the measured 100k one). Parents are resolved by CELL ID and
OBJECT SHA off the volume's bank, never through the fits configs (the
ctl-full config bytes stay out of every tree other live apps serve).
``--app-name sdfbase-derived-<sha16>`` is REQUIRED on every run and must
equal the app this tree names (HEAD's first 16 hex), so an invocation
records which code it ran and can never be read as having run another
deployment's (an omitted name bypasses the guard). The pair is
CANONICAL: CANONICAL_CTL binds each corpus cell to the one control it is
decomposed against (mix-100k → ctl-100k, mix-full → ctl2-full); any other
control REFUSES before a spawn, and every builder refuses (never rebuilds)
an existing artifact whose sidecar names other parents — the derived ids
carry the corpus cell only, so a non-canonical pair would otherwise
overwrite a published object under its own id.
Every example below is written without the flag for brevity; add
``--app-name sdfbase-derived-<sha16>`` to each.

  modal run scripts/modal/sdfbase_derived.py --mode residual-distill --model <key>
      # the residual pipeline as a FAN-OUT: per model, a CPU step composes V ONCE
      # (+P_mix − P_ctl, parents identity-gated; kept when already on the volume),
      # then ONE H200 container per lr multiplier of --lr-mults (default 0.3,1,3) runs
      # `psr_residual_distill.py calibrate --lr-mult <m>` — the persisted V
      # reloaded and re-admitted, the stratified capture set, numeric
      # floor, planted C, the real-model
      # teacher identity check (write identity + hook contracts, incl. the
      # persisted V's), then controls (a) R(mix, ∅) + (b) R(mix ⊕ C, mix)
      # at THAT multiplier — into calib_parts/lrmult<m>/; a CPU merge
      # (`merge-calibration`) assembles calibration_controls.json with the
      # SAME selection rule the in-process sweep applies; then one
      # production container (`distill`) gates on it and distills V at the
      # selected multiplier: ED + doc-level bootstrap, recovery verdict.
      # Every fit is RESUMABLE (--checkpoint-every, default 50 optimizer
      # steps, identity-keyed trainer checkpoints on the volume) and every
      # finished multiplier is REUSED on a retry (calibrate reads the
      # existing sweep file and skips identity-matching payloads), so a
      # timeout or a production failure re-issues the SAME command and
      # costs at most one checkpoint interval — never the whole sweep.
      # Outputs: /vol/out/derived/psr_paired_write/<key>/<mix>__psr_ce_all_minus_ctl/
      #          /vol/out/derived/psr_residual/<key>/<mix>__psr_ce_all_minus_ctl_distilled/
      #            {calib_parts/lrmult<m>/..., calibration_controls.json, ckpt/, ...}
      # --distill-epochs-cap 60 --distill-patience 5 = the budget
      # (24 / 2 under-budgeted the planted controls); both ride into every
      # container explicitly (the production run's identity must match the
      # calibration's field for field, budget included).
      # --serial runs the pre-fan-out path instead: ONE container per model
      # running `all` (sequential multipliers; same resume + checkpoints).
  modal run scripts/modal/sdfbase_derived.py --mode calibration --model <key>
      # the fan-out WITHOUT the production step (parts + merge — the
      # calibration preflight); --serial = the one-container `all --no-production`.
  modal run scripts/modal/sdfbase_derived.py --mode residual-distill-prod --model <key>
      # PRODUCTION ONLY: one container per model running `distill`, which
      # REQUIRES a complete, identity-matching calibration sweep already on
      # the volume (the container checks the sweep file before staging the
      # model; the CLI then compares the full identity before training) and
      # distills at its selected multiplier. Never runs a calibration fit.
  modal run scripts/modal/sdfbase_derived.py --mode identity --model <key>
      # the identity-ONLY preflight (minutes, no W&B, no distillation):
      # psr_residual_distill.py identity-check — planted C, the real-model
      # write identity + hook contracts of the composite and the reference,
      # and the hook contract of the PERSISTED V reloaded from the volume;
      # the per-layer report is returned inline and lands at
      # /vol/out/derived/psr_residual/<key>/<id>/identity_check_report.json.
  modal run scripts/modal/sdfbase_derived.py --mode lora-subtract --model <key>
      # CPU: the LoRA task-vector subtraction ΔW(mix) − ΔW(ctl)
      # (lora_subtract.py --keep-valid: an existing publication that passes
      # the shared verdict against the live bank and names THESE parents
      # is kept) -> /vol/out/derived/lora_subtraction/<key>/<mix>__lora_r4_ce_minus_ctl/
  modal run scripts/modal/sdfbase_derived.py --mode score --model <key> \
      [--objects auto|mix|id1,id2] [--scales 1.0,2.0] [--probe v2[,v3g]] [--no-parents]
      # probe (probe_score.py --scale, every registered version listed) +
      # held-out CE (ce_score.py, per-doc records) for the derived objects
      # of <key> (--objects auto = every one under /vol/out/derived/*/<key>/;
      # mix = those of --mix-si only) at the listed scales, on the control's
      # CE pools (CE_POOLS_BY_CTL: the corpus held-out, the control held-out,
      # and for the 100k pair the ctl in-sample training pool); plus the
      # pair's four parents and the bare model at scale 1 (P3 needs
      # gain_mix / gain_ctl / CE_bare on the same docs).
      # --ce-skip-existing keeps a CE file already on the volume (the bare
      # and parent rows another pair published) instead of re-scoring it.
      # Probe margins -> /vol/out/<key>/probe_<ver>[_x2]/<id>.jsonl
      # CE records    -> /vol/out/ce_score/<key>/<id>__<pool>__scale<s>.json
      # Every derived object is scored with --parents-root /vol/out: its
      # parents' shas are re-verified against the LIVE bank first.
  modal run scripts/modal/sdfbase_derived.py --mode mmlu --model <key> \
      --run-label <label> [--objects auto|id1,id2]
      # plain-text 5-shot MMLU (base_mmlu.py) at the bare model + every
      # derived object of <key>, each passed as --object <id>=<path> with
      # --parents-root /vol/out (identity gate + parents re-verified), in
      # this app's OWN namespace: /vol/out/derived/mmlu/<key>/<label>/
      # mmlu_<label>_<id>.{json,per_question.jsonl} — never the fits app's
      # /vol/out/<key>/mmlu/, never a bank. --run-label is REQUIRED.
  modal run scripts/modal/sdfbase_derived.py --mode snapshot-verify \
      --model <key> [--cells id1,id2] [--out-root out/<key>]
      # CPU: re-run the verify half of snapshot_cells on the <key>_r1
      # snapshot bank (every COMPLETE cell by default) PLUS the shared
      # staged-cell identity gate on each copy; the per-cell verdicts are
      # recorded under /vol/out/derived/snapshot_verify/<key>/ — the bank
      # itself is read, never written.

One-shot GPU containers, zero idle, every artifact committed to the volume.
Model specs, HF snapshot staging, env, and the image RECIPES are the fit
app's (imported from scripts/modal/sdfbase_fits.py — read, never edited or
redeployed here); the image chains are re-built here only because the
commit env var must be baked BEFORE the local-dir layer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, Path("/repo/scripts/modal")):
    if (_cand / "sdfbase_fits.py").is_file() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
import sdfbase_fits as base  # noqa: E402

WORKTREE = base.WORKTREE
MODELS = base.MODELS
vol = base.vol



def _fn_pair(key: str, std, qwen):
    """The derived app's own std/qwen function for a model key. Local handles
    are correct HERE: this module is its own Modal app (sdfbase-derived-<sha>),
    never the fits app, so the fits module's deployed_fn routing does not
    apply. Selection reuses the fits module's single spelling of the image
    variant (base._variant; nemotron keys have no derived functions)."""
    v = base._variant(key)
    if v == "nemotron":
        raise SystemExit(f"[derived] no derived-app functions for image variant "
                         f"{v!r} (model key {key!r})")
    return qwen if v == "qwen" else std

def _git_sha16() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha[:16] if out.returncode == 0 and len(sha) >= 16 else None


#: The commit the app is named by: the env pin (baked into the image so the
#: container-side import names the SAME app), else HEAD, else "local".
SHA16 = os.environ.get("SDFBASE_DERIVED_SHA16") or _git_sha16() or "local"
APP_NAME = f"sdfbase-derived-{SHA16}"
app = modal.App(APP_NAME)

_IGNORE = base._IGNORE
_ENV_PIN = {"SDFBASE_DERIVED_SHA16": SHA16}

#: Image recipes copied VERBATIM from sdfbase_fits.py (image_std /
#: image_qwen), with the commit pin baked before the local-dir layer.
image_std = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unixodbc")
    .pip_install_from_requirements(
        str(WORKTREE / "scripts/modal/requirements.txt"),
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env(_ENV_PIN)
    .add_local_dir(str(WORKTREE), remote_path="/repo", ignore=_IGNORE)
)

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
        "pip install --no-deps triton==3.7.1",
        "python -c \"import triton; v = triton.__version__; "
        "assert v == '3.7.1', v; import causal_conv1d, fla\"",
    )
    .env(_ENV_PIN)
    .add_local_dir(str(WORKTREE), remote_path="/repo", ignore=_IGNORE)
)

MAX_CONTAINERS = base.MAX_CONTAINERS

#: The HF token plus the OPERATOR'S OWN W&B secret (WANDB_API_KEY +
#: WANDB_ENTITY + WANDB_PROJECT). The workspace's shared ``wandb``
#: secret carries another member's key — this app's first runs landed
#: in a foreign W&B entity — and is never referenced here. Every
#: ``wandb.init`` this app reaches passes ``entity=$WANDB_ENTITY``
#: explicitly, and :func:`sdfbase_fits.require_wandb_entity` refuses at
#: container start unless the mounted key's default entity IS that entity.
#: The W&B secret's name is the site setting SISTEER_MODAL_WANDB_SECRET, read
#: once where the apps are defined (:data:`sdfbase_fits.WANDB_SECRET`).
_SECRETS = [modal.Secret.from_name("hf-token")] + (
    [modal.Secret.from_name(base.WANDB_SECRET)] if base.WANDB_SECRET else [])
_COMMON = dict(volumes={"/vol": vol}, secrets=_SECRETS)

#: Full stdout/stderr of every in-container command, on the volume (a
#: 3000-char stderr tail lost gemma's layer<17 report lines).
LOG_ROOT = "/vol/out/derived/_logs"
RUN_CWD = "/repo"
STDERR_TAIL_CHARS = 3000

#: Derived-object namespaces on the volume (the lora_subtract.py layout).
DERIVED_ROOT = "/vol/out/derived"
DERIVED_KINDS = ("psr_paired_write", "psr_residual", "lora_subtraction")

#: The residual pipeline's per-multiplier calibration parts live under the
#: distilled object's dir (psr_residual_distill.CALIB_PARTS_DIRNAME).
CALIB_PARTS_DIRNAME = "calib_parts"

#: Resumable trainer checkpoints every N optimizer steps for every control
#: and the production fit (the entrypoint default; ~5.5 s/step on gemma →
#: a checkpoint every ~4.6 min, one per ~5 epochs; the loss on a killed
#: container is at most this interval plus the 5-min volume commit).
DEFAULT_CHECKPOINT_EVERY = 50

#: Function timeouts DERIVED from the measured 60-epoch runtime:
#: gemma's cap-60 control took 2985 s (~50 s/epoch; qwen ~1/3 of that),
#: production 15 min gemma / 6.5 min qwen. One calibration container = two
#: controls at one multiplier ≤ 2 × 3000 s + model staging + floor/identity
#: (~10 min) ≈ 1.9 h worst case → 4 h (≥2× pad). One production container =
#: one ≤60-epoch fit + precheck + ED bootstrap + T_arith + liveness ≈ 1.2 h
#: worst case → 3 h. The serial `all` container keeps the 8 h it always had
#: (six controls + production, worst case ~5.3 h). Checkpoints make every
#: timeout recoverable by re-issuing the same command.
EPOCH_S_WORST = 50
CALIB_TIMEOUT_S = 4 * 3600
PROD_TIMEOUT_S = 3 * 3600
SERIAL_TIMEOUT_S = 28800

#: The score mode's CE pools — the ANALYSIS names (analysis_r2/r2_paths.py
#: CE_POOLS) -> the staged pool file. ``control-heldout`` is the control
#: corpus remainder unused by ctl-100k (scripts/sdfbase/build_ctl_heldout.py).
#: These are the 100k pair's pools (the round-1 control); the pair the
#: published objects were scored under, kept byte for byte.
CE_POOLS = {
    "constv2-heldout": "/vol/pool/{key}/pairs_constv2-heldout.jsonl",
    "control-heldout": "/vol/pool/{key}/pairs_constv2ctl-heldout.jsonl",
    "constv2ctl-100k": "/vol/pool/{key}/pairs_constv2ctl-100k.jsonl",
}

#: The 100k pair the app was built on — the entrypoint's defaults.
DEFAULT_MIX_SI = "constv2-mix-100k"
DEFAULT_CTL_SI = "constv2ctl-100k"

#: The full − full pair's CE pools: the corpus held-out (every cell's
#: early-stop pool) and the r2 control's OWN frozen 128-doc held-out
#: twin (``constv2ctl2-heldout``, P3's pool).
#: The ctl2 TRAINING pool (17,073 docs / 42.8M tokens) is NOT a default
#: leg — one in-sample descriptive number costs ~1 H200-h per object; it
#: stays reachable through --ce-pools (CE_POOL_CTL2_INSAMPLE spells it).
CE_POOLS_CTL2 = {
    "constv2-heldout": "/vol/pool/{key}/pairs_constv2-heldout.jsonl",
    "constv2ctl2-heldout": "/vol/pool/{key}/pairs_constv2ctl2-heldout.jsonl",
}
CE_POOL_CTL2_INSAMPLE = {
    "constv2ctl2-full": "/vol/pool/{key}/pairs_constv2ctl2-full.jsonl",
}
#: control SI -> the NAME of its default CE pool table (resolved at call
#: time, so the table a run reads is the module's current one — the seam
#: the 100k golden replay pins). A control not registered here REFUSES
#: (name the pools with --ce-pools) rather than scoring against another
#: control's register pool.
CE_POOLS_BY_CTL = {DEFAULT_CTL_SI: "CE_POOLS", "constv2ctl2-full": "CE_POOLS_CTL2"}


def ce_pools_for(ctl_si: str) -> dict[str, str]:
    """A COPY of the default CE pools of a control SI (templates with
    ``{key}``)."""
    try:
        return dict(globals()[CE_POOLS_BY_CTL[ctl_si]])
    except KeyError:
        raise SystemExit(f"[derived] no default CE pools for control {ctl_si!r} "
                         f"(registered: {sorted(CE_POOLS_BY_CTL)}); pass "
                         "--ce-pools tag:path,...") from None

#: ce_score outputs land kind-first (like derived/), in the analysis shape:
#: ce_score/<model>/<cell>__<pool>__scale<s>.json (one combined JSON).
CE_ROOT = "/vol/out/ce_score"

#: The fits bank root every derived object's parents are re-verified
#: against (probe/CE/MMLU all pass it as --parents-root; the consumers
#: REFUSE a derived object without it).
PARENTS_ROOT = "/vol/out"

#: This app's OWN output namespaces for the two round-2 modes that could
#: otherwise collide with the wave-A app's: nothing here writes under
#: /vol/out/<key>/ (the fits banks + the fits app's mmlu/) or /vol/out/hpo/.
MMLU_ROOT = "/vol/out/derived/mmlu"
SNAPSHOT_VERIFY_ROOT = "/vol/out/derived/snapshot_verify"

#: gate_reg trial dirs (tune_arm layout: object.pt + summary.json each).
GATEREG_TRIALS = "/vol/out/hpo/{key}/psr_ce_all_all"
GATEREG_ROUND = 3
#: The gate_reg design's data cell — the liveness probe pool.
GATEREG_SI = "constv2-mix-1m"

#: The ONE control each corpus cell is decomposed against. A derived id
#: names the corpus cell only (``<mix>__<arm>_minus_ctl*``), so the pair
#: must be a function of the corpus cell or two controls would share an
#: id and a path; a mix cell absent here has no decomposition (refused,
#: never guessed — the 1M rows' budget-mismatched pairing is not revived).
CANONICAL_CTL = {"constv2-mix-100k": "constv2ctl-100k",
                 "constv2-mix-full": "constv2ctl2-full"}


def require_canonical_pair(mix_si: str, ctl_si: str) -> None:
    """Refuse a (corpus, control) pair that is not the registered one."""
    want = CANONICAL_CTL.get(mix_si)
    if want is None:
        raise SystemExit(f"[derived] no canonical control for corpus cell "
                         f"{mix_si!r} (registered: {CANONICAL_CTL}); a "
                         "derived object of it has no defined pair")
    if ctl_si != want:
        raise SystemExit(f"[derived] {mix_si!r} is decomposed against "
                         f"{want!r} only, not {ctl_si!r}: the derived ids "
                         "name the corpus cell alone, so another control "
                         "would overwrite that pair's published objects")


def sidecar_parent_problems(meta_path, mix_si: str, ctl_si: str, arm: str
                            ) -> list[str]:
    """Problems (empty = same pair) when an existing derived sidecar names
    parents other than ``<mix_si>__<arm>`` / ``<ctl_si>__<arm>`` — the
    builders' defense in depth: such an artifact is REFUSED, never rebuilt
    in place (the golden pins the app's commands, not the volume's files)."""
    meta_path = Path(meta_path)
    if not meta_path.is_file():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError as e:
        return [f"{meta_path} unreadable: {e}"]
    out = []
    for role, cell in (("parent_pos", f"{mix_si}__{arm}"),
                       ("parent_neg", f"{ctl_si}__{arm}")):
        got = (meta.get(role) or {}).get("cell_id")
        if got != cell:
            out.append(f"{meta_path}: {role}.cell_id={got!r} — this run's pair "
                       f"needs {cell!r}; an artifact of another pair under "
                       "this id is never rebuilt in place")
    return out


def parent_cells(mix_si: str, ctl_si: str) -> tuple[str, str, str, str]:
    """The pair's four parents (both arms of both cells) whose CE the P3
    read needs on the same pools — in the order the score mode runs them."""
    return (f"{mix_si}__psr_ce_all", f"{ctl_si}__psr_ce_all",
            f"{mix_si}__lora_r4_ce", f"{ctl_si}__lora_r4_ce")


#: The 100k pair's parents (the defaults' spelling, pinned by the golden).
PARENT_CELLS = parent_cells(DEFAULT_MIX_SI, DEFAULT_CTL_SI)


def parse_probe_versions(spec: str) -> list[str]:
    """``--probe`` as a comma list of REGISTERED versions (order kept,
    de-duplicated; ``v2`` alone is the historical single-probe call)."""
    vers = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if tok and tok not in vers:
            vers.append(tok)
    if not vers:
        raise SystemExit("[derived] --probe must list at least one registered "
                         "probe version (e.g. v2 or v3g,v2)")
    for v in vers:
        base.probe_file_stem(v)   # refuses an unregistered version
    return vers


def _sha256_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ce_reuse_problems(out_path, *, key: str, pool_tag: str, pool_path: str,
                      obj_path: str | None, cell: str | None, scale: float
                      ) -> list[str]:
    """Why an EXISTING combined ce_score file may NOT stand in for the run
    about to happen (empty = it may): it must record this pool tag, the
    pool file's CURRENT sha, the CURRENT object bytes' sha (None for the
    bare row), this cell (None for bare), this scale, this model + revision,
    and one record per pool row. Existence alone proves nothing: a stale
    parent publication or a truncated file under the same path would have
    been kept as a result."""
    out_path = Path(out_path)
    if not out_path.is_file():
        return [f"{out_path} absent"]
    try:
        doc = json.loads(out_path.read_text(encoding="utf-8"))
    except ValueError as e:
        return [f"{out_path} unreadable: {e}"]
    spec = MODELS[key]
    problems = []
    try:
        want = {"pool": pool_tag, "pairs_sha256": _sha256_file(pool_path),
                "object_sha256": _sha256_file(obj_path) if obj_path else None,
                "cell": cell, "scale": float(scale),
                "model": spec["repo"], "revision": spec["rev"]}
    except OSError as e:
        return [f"cannot hash the current inputs: {e}"]
    for k, v in want.items():
        got = doc.get(k)
        if k == "scale" and got is not None:
            got = float(got)
        if got != v:
            problems.append(f"{k}: recorded {got!r} != current {v!r}")
    n_rows = sum(1 for l in Path(pool_path).read_text(encoding="utf-8")
                 .splitlines() if l.strip())
    docs = doc.get("docs")
    if not isinstance(docs, list) or len(docs) != n_rows:
        problems.append(f"docs: recorded {len(docs) if isinstance(docs, list) else None} "
                        f"!= {n_rows} pool rows")
    return problems


def _cap_tail(capture_budget_tokens) -> tuple:
    """The trailing item field the residual stages carry ONLY when a capture
    budget is set — the 100k items keep their historical shape."""
    return ((int(capture_budget_tokens),)
            if capture_budget_tokens else ())


# --------------------------------------------------------------------------- #
# Command builders (pure — pinned by tests/test_sdfbase_derived_app.py)         #
# --------------------------------------------------------------------------- #


def resid_out_dir(key: str, mix_si: str) -> str:
    """The distilled object's dir (psr_residual_distill._resid_dir on the
    volume) — calibration_controls.json, calib_parts/, ckpt/ live here."""
    return f"{DERIVED_ROOT}/psr_residual/{key}/{mix_si}__psr_ce_all_minus_ctl_distilled"


def v_object_path(key: str, mix_si: str) -> str:
    """The persisted teacher V (psr_residual_distill.v_derived_id layout)."""
    return f"{DERIVED_ROOT}/psr_paired_write/{key}/{mix_si}__psr_ce_all_minus_ctl/object.pt"


def mult_key(m) -> str:
    """psr_residual_distill.format_mult's spelling (``1.0`` → ``1``), pinned
    equal by tests/test_sdfbase_derived_app.py."""
    return f"{float(m):g}"


def parse_mults(spec: str) -> list[float]:
    """The entrypoint's ``--lr-mults`` list: positive finite floats,
    de-duplicated, order kept (psr_residual_distill.parse_lr_mults's rule)."""
    vals = []
    for tok in str(spec).replace(";", ",").split(","):
        if not tok.strip():
            continue
        try:
            v = float(tok)
        except ValueError:
            raise SystemExit(f"[derived] --lr-mults {spec!r}: {tok!r} is not a number")
        if not (v > 0 and v == v and v != float("inf")):
            raise SystemExit(f"[derived] --lr-mults {spec!r}: every multiplier "
                             "must be a finite number > 0")
        if v not in vals:
            vals.append(v)
    if not vals:
        raise SystemExit("[derived] --lr-mults must list at least one "
                         "multiplier (default 0.3,1,3)")
    return vals


def _resid_base(key: str, mix_si: str, ctl_si: str, sub: str, *,
                python: str, capture_budget_tokens: int | None = None
                ) -> list[str]:
    spec = MODELS[key]
    cmd = [python, "/repo/scripts/sdfbase/psr_residual_distill.py", sub,
           "--root", "/vol/out", "--out-root", DERIVED_ROOT,
           "--pools", f"/vol/pool/{key}", "--model-key", key,
           "--mix-si", mix_si, "--ctl-si", ctl_si,
           "--model", spec["repo"], "--revision", spec["rev"],
           "--config", spec["cfg"]]
    if capture_budget_tokens is not None:
        cmd += ["--capture-budget-tokens", str(int(capture_budget_tokens))]
    return cmd


def _budget_args(distill_epochs_cap, distill_patience, checkpoint_every
                 ) -> list[str]:
    out: list[str] = []
    if distill_epochs_cap is not None:
        out += ["--distill-epochs-cap", str(int(distill_epochs_cap))]
    if distill_patience is not None:
        out += ["--distill-patience", str(int(distill_patience))]
    if checkpoint_every:
        out += ["--checkpoint-every", str(int(checkpoint_every))]
    return out


def residual_command(key: str, mix_si: str, ctl_si: str, *,
                     no_production: bool, lr_mult, wandb: bool,
                     python: str = "python", distill_epochs_cap: int | None = None,
                     distill_patience: int | None = None,
                     checkpoint_every: int | None = None,
                     capture_budget_tokens: int | None = None) -> list[str]:
    """The SERIAL path (``all`` in one container). ``lr_mult``: a number
    (one multiplier) or the comma-separated sweep string (``"0.3,1,3"``)
    the CLI runs in full; the budget knobs and the checkpoint cadence ride
    along explicitly when given. Finished multipliers are reused by the
    CLI itself (resume is its default)."""
    mult = (str(float(lr_mult)) if isinstance(lr_mult, (int, float))
            else str(lr_mult))
    cmd = _resid_base(key, mix_si, ctl_si, "all", python=python,
                      capture_budget_tokens=capture_budget_tokens)
    cmd += ["--lr-mult", mult]
    cmd += _budget_args(distill_epochs_cap, distill_patience, checkpoint_every)
    if no_production:
        cmd.append("--no-production")
    if wandb:
        cmd.append("--wandb")
    return cmd


def calibrate_command(key: str, mix_si: str, ctl_si: str, *, lr_mult: float,
                      wandb: bool, python: str = "python",
                      distill_epochs_cap: int | None = None,
                      distill_patience: int | None = None,
                      checkpoint_every: int | None = None,
                      capture_budget_tokens: int | None = None
                      ) -> tuple[list[str], str]:
    """ONE multiplier's calibration (the fan-out's unit): ``calibrate
    --lr-mult <m> --out <resid dir>/calib_parts/lrmult<m>/calibration_
    controls.json``. Returns (argv, that part file's path)."""
    out = (f"{resid_out_dir(key, mix_si)}/{CALIB_PARTS_DIRNAME}/"
           f"lrmult{mult_key(lr_mult)}/calibration_controls.json")
    cmd = _resid_base(key, mix_si, ctl_si, "calibrate", python=python,
                      capture_budget_tokens=capture_budget_tokens)
    cmd += ["--lr-mult", str(float(lr_mult)), "--out", out]
    cmd += _budget_args(distill_epochs_cap, distill_patience, checkpoint_every)
    if wandb:
        cmd.append("--wandb")
    return cmd, out


def compose_command(key: str, mix_si: str, ctl_si: str, *,
                    python: str = "python") -> tuple[list[str], str]:
    """The CPU compose step that precedes the fan-out: ``compose-v`` (model-
    free, parents identity-gated) writes V ONCE, so the per-multiplier
    calibration containers all RELOAD the same persisted teacher instead of
    racing to compose it (V's blob carries its build time, so two
    concurrent composers would land two different files under one path).
    Returns (argv, V's object path)."""
    MODELS[key]  # a known model key (KeyError otherwise); the step is model-free
    cmd = [python, "/repo/scripts/sdfbase/psr_residual_distill.py", "compose-v",
           "--root", "/vol/out", "--out-root", DERIVED_ROOT,
           "--pools", f"/vol/pool/{key}", "--model-key", key,
           "--mix-si", mix_si, "--ctl-si", ctl_si]
    return cmd, v_object_path(key, mix_si)


def verify_v_command(key: str, mix_si: str, ctl_si: str, *,
                     python: str = "python") -> list[str]:
    """``verify-v``: the persisted V re-admitted against the LIVE bank (exit
    2 on a torn publish, a parent refit under the same cell id, any
    verdict problem) — what the compose stage asks before letting an
    existing V stand."""
    MODELS[key]  # a known model key (KeyError otherwise); the step is model-free
    return [python, "/repo/scripts/sdfbase/psr_residual_distill.py", "verify-v",
            "--root", "/vol/out", "--out-root", DERIVED_ROOT,
            "--pools", f"/vol/pool/{key}", "--model-key", key,
            "--mix-si", mix_si, "--ctl-si", ctl_si]


def merge_command(key: str, mix_si: str, ctl_si: str, *, lr_mults,
                  python: str = "python") -> tuple[list[str], str]:
    """The CPU aggregation: ``merge-calibration --lr-mult <full sweep>`` over
    the parts calibrate_command wrote, into ``<resid dir>/calibration_
    controls.json`` (the CLI's default paths — one spelling). Returns
    (argv, the merged file's path)."""
    MODELS[key]  # a known model key (KeyError otherwise); the merge is model-free
    out = f"{resid_out_dir(key, mix_si)}/calibration_controls.json"
    cmd = [python, "/repo/scripts/sdfbase/psr_residual_distill.py",
           "merge-calibration",
           "--root", "/vol/out", "--out-root", DERIVED_ROOT,
           "--pools", f"/vol/pool/{key}", "--model-key", key,
           "--mix-si", mix_si, "--ctl-si", ctl_si,
           "--lr-mult", ",".join(mult_key(m) for m in lr_mults),
           "--parts", ",".join(
               f"{resid_out_dir(key, mix_si)}/{CALIB_PARTS_DIRNAME}/"
               f"lrmult{mult_key(m)}/calibration_controls.json"
               for m in lr_mults),
           "--out", out]
    return cmd, out


def distill_command(key: str, mix_si: str, ctl_si: str, *, wandb: bool,
                    python: str = "python", distill_epochs_cap: int | None = None,
                    distill_patience: int | None = None,
                    checkpoint_every: int | None = None,
                    capture_budget_tokens: int | None = None) -> list[str]:
    """PRODUCTION ONLY: ``distill`` at the calibration's SELECTED multiplier
    (no ``--lr-mult``), gated by the sweep file in the out dir — the CLI
    refuses an absent / incomplete / other-identity calibration before any
    training."""
    cmd = _resid_base(key, mix_si, ctl_si, "distill", python=python,
                      capture_budget_tokens=capture_budget_tokens)
    cmd += _budget_args(distill_epochs_cap, distill_patience, checkpoint_every)
    if wandb:
        cmd.append("--wandb")
    return cmd


def calibration_precheck(cal_path, *, epochs_cap: int | None,
                         patience: int | None) -> list[str]:
    """What the production-only container checks on the volume BEFORE it
    stages the model: the sweep file exists and is readable, holds a
    COMPLETE sweep with a selected multiplier whose payload is present, the
    payload records an identity fingerprint, and the sweep's budget is the
    budget this run asks for (the CLI then compares the FULL identity
    before training — this is the fail-fast half). Returns problems."""
    cal_path = Path(cal_path)
    if not cal_path.is_file():
        return [f"{cal_path} missing — run --mode calibration (or "
                "residual-distill) first; production never calibrates"]
    try:
        doc = json.loads(cal_path.read_text(encoding="utf-8"))
    except ValueError as e:
        return [f"{cal_path} unreadable: {e}"]
    sweep = doc.get("sweep") if isinstance(doc, dict) else None
    if not isinstance(sweep, dict):
        return [f"{cal_path} is not an lr-mult sweep file (no `sweep` block)"]
    problems = []
    if not sweep.get("complete", False):
        problems.append(f"sweep INCOMPLETE: finished {list(sweep.get('per_mult') or {})} "
                        f"of {sweep.get('lr_mults')}")
    sel = sweep.get("selected_lr_mult")
    per = sweep.get("per_mult") or {}
    if sel is None or mult_key(sel) not in per:
        problems.append(f"selected_lr_mult {sel!r} has no payload in "
                        f"sweep.per_mult {list(per)}")
    else:
        ident = (per[mult_key(sel)].get("identity") or {})
        if not ident.get("fingerprint"):
            problems.append(f"lr_mult={mult_key(sel)} payload records no "
                            "identity fingerprint")
    want = {"epochs_cap": epochs_cap, "patience": patience}
    got = sweep.get("budget") or {}
    for k, v in want.items():
        if v is not None and got.get(k) != int(v):
            problems.append(f"sweep budget {k}={got.get(k)!r} != this run's "
                            f"{int(v)} — the production identity would not "
                            "match the calibration's")
    return problems


def identity_command(key: str, mix_si: str, ctl_si: str, *,
                     python: str = "python",
                     capture_budget_tokens: int | None = None) -> list[str]:
    """The identity-only preflight (no ``--wandb``: nothing is logged)."""
    return _resid_base(key, mix_si, ctl_si, "identity-check", python=python,
                       capture_budget_tokens=capture_budget_tokens)


def lora_object_path(key: str, mix_si: str) -> str:
    """The LoRA subtraction's artifact (lora_subtract.derived_id_for)."""
    return f"{DERIVED_ROOT}/lora_subtraction/{key}/{mix_si}__lora_r4_ce_minus_ctl/object.pt"


def lora_subtract_command(key: str, mix_si: str, ctl_si: str, *,
                          python: str = "python") -> tuple[list[str], str]:
    """The CPU LoRA task-vector subtraction ΔW(mix) − ΔW(ctl) on the volume's
    bank (parents admitted by cell id + object sha through the shared gate;
    ``--keep-valid`` keeps an existing publication that passes the verdict
    against the LIVE bank and names exactly these parents, so a re-issue is
    idempotent). Returns (argv, the object path)."""
    MODELS[key]  # a known model key (KeyError otherwise); the step is model-free
    cmd = [python, "/repo/scripts/sdfbase/lora_subtract.py",
           "--root", "/vol/out", "--out-root", f"{DERIVED_ROOT}/lora_subtraction",
           "--models", key, "--cells", mix_si, "--ctl-si", ctl_si,
           "--pools", f"/vol/pool/{key}", "--keep-valid"]
    return cmd, lora_object_path(key, mix_si)


def probe_command(key: str, obj_path: str, derived_id: str, *, scale: float,
                  ver: str, python: str = "python") -> tuple[list[str], str]:
    spec = MODELS[key]
    # ONE registry binds the frozen probe file and the output dir: this
    # caller kept its own probe_{ver} spelling and would have read a
    # nonexistent probe_v3g.jsonl while writing to probe_v3/.
    stem = base.probe_file_stem(ver)
    out_dir = f"/vol/out/{key}/{base.probe_out_dirname(ver, scale)}"
    # The derived id IS the cell name (analysis DERIVED_CELLS): the row file
    # sits beside the fitted cells' as <cell>.jsonl, no prefix.
    out = f"{out_dir}/{derived_id}.jsonl"
    cmd = [python, "/repo/scripts/sdfbase/probe_score.py", "score",
           "--probe", f"/repo/data/sdfbase/{stem}.jsonl",
           "--manifest", f"/repo/data/sdfbase/{stem}.manifest.json",
           "--model", spec["repo"], "--revision", spec["rev"],
           "--context", "bare", "--render", "raw",
           "--object", obj_path, "--cell", derived_id,
           "--parents-root", PARENTS_ROOT,
           "--out", out]
    if float(scale) != 1.0:
        cmd += ["--scale", str(float(scale))]
    return cmd, out


def ce_command(key: str, obj_path: str | None, tag: str, pool_tag: str,
               pool_path: str, *, scale: float, source_tag: str | None = None,
               python: str = "python") -> tuple[list[str], str]:
    """``ce_score.py`` on one pool; ``obj_path=None`` scores the bare model
    (tag ``bare``). Every object goes through the identity gate with its
    id + the bank root as parents_root — a derived artifact whose parents
    changed under it refuses before the model loads."""
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
    if source_tag:
        cmd += ["--source-tag", source_tag]
    return cmd, out


def liveness_command(key: str, dirs: list[str], *, pairs_path: str,
                     n_probe: int, allow_bare: bool,
                     python: str = "python") -> list[str]:
    """``gate_liveness.py`` over object dirs: writes gate_liveness.json
    ({layer: live_frac}) + a detail sidecar next to each object."""
    spec = MODELS[key]
    cmd = [python, "/repo/scripts/sdfbase/gate_liveness.py",
           "--model", spec["repo"], "--revision", spec["rev"],
           "--pairs", pairs_path, "--n-probe", str(int(n_probe)),
           "--dirs", ",".join(dirs), "--max-len", "8192"]
    if allow_bare:
        cmd.append("--allow-bare")
    return cmd


def mmlu_command(key: str, objects: dict[str, str], run_label: str, *,
                 python: str = "python") -> tuple[list[str], str]:
    """``base_mmlu.py`` over the DERIVED objects of ``key`` (+ the bare
    ``none`` row the per-cell deltas need), run-scoped under THIS app's
    namespace: outputs land at ``MMLU_ROOT/<key>/<label>/mmlu_<label>_<id>
    .{json,per_question.jsonl}`` — never the fits app's /vol/out/<key>/mmlu/
    and never a bank. Each object enters as an explicit ``--object
    <id>=<path>`` (derived ids live under derived/<kind>/<key>/, not in a
    bank's cells_all/) and ``--parents-root`` re-verifies its parents
    against the live bank (base_mmlu refuses a derived object without it).
    ``run_label`` is REQUIRED (base_mmlu namespaces dir + filename by it).
    Returns (argv, the run's output root)."""
    spec = MODELS[key]
    if not run_label:
        raise SystemExit("[derived/mmlu] --run-label is required: every "
                         "output carries it in dir + filename so two runs "
                         "never share a path")
    if not objects:
        raise SystemExit(f"[derived/mmlu] no derived objects for {key}")
    out_dir = f"{MMLU_ROOT}/{key}"
    cells = ["none", *sorted(objects)]
    cmd = [python, "/repo/scripts/sdfbase/base_mmlu.py",
           "--config", spec["cfg"], "--cells", ",".join(cells),
           "--parents-root", PARENTS_ROOT,
           "--n", "500", "--seed", "0",
           "--out-dir", out_dir, "--run-label", run_label]
    for oid in sorted(objects):
        cmd += ["--object", f"{oid}={objects[oid]}"]
    return cmd, out_dir


def snapshot_verify_record_path(key: str, stamp: str) -> str:
    """Where a snapshot-verify run's per-cell verdicts are recorded — under
    this app's namespace, never inside the snapshot bank it verified."""
    return f"{SNAPSHOT_VERIFY_ROOT}/{key}/verify_{stamp}.json"


def verify_snapshot_bank(dst_bank, cells: list[str] | None) -> dict:
    """Re-verify a ``<key>_r1`` snapshot bank — READ-ONLY (nothing under
    the bank is written): for every named cell (every COMPLETE cell when
    ``cells`` is None) the verify half of snapshot_cells (marker + file
    manifest shas + snapshot record naming the same fingerprint + store
    identity) AND the shared staged-cell identity gate on the copy — the
    snapshot is what the r1 MMLU run and the analysis read, so it must
    validate exactly as a live cell does. Returns per-cell rows; ``status``
    is "ok" only when at least one cell was named and every one verified.
    Pure filesystem — testable on a tmp bank."""
    from snapshot_cells import SnapshotError, verify_snapshot
    from sisteer.eval.train_budget import staged_cell_problems

    dst_bank = Path(dst_bank)
    cells_dir = dst_bank / "cells_all"
    if cells is None:
        cells = (sorted(d.name for d in cells_dir.iterdir()
                        if d.is_dir() and (d / "COMPLETE").is_file())
                 if cells_dir.is_dir() else [])
    rows = []
    for cell in cells:
        row: dict = {"cell_id": cell}
        try:
            rec = verify_snapshot(dst_bank, cell)
            problems = staged_cell_problems(cells_dir / cell / "object.pt", cell)
            if problems:
                row.update(status="refused",
                           error="staged-cell identity gate:\n  - "
                                 + "\n  - ".join(problems))
            else:
                row.update(status="verified",
                           fit_fingerprint=rec.get("fit_fingerprint"),
                           source=rec.get("source"),
                           n_files=len(rec.get("files") or {}))
        except (SnapshotError, ValueError, OSError) as e:
            row.update(status="refused", error=str(e)[:1200])
        rows.append(row)
    bad = [r for r in rows if r["status"] != "verified"]
    return {"snapshot_bank": str(dst_bank), "cells": list(cells),
            "status": "ok" if rows and not bad else "failed",
            "n_ok": len(rows) - len(bad), "n_bad": len(bad),
            "note": (None if rows else
                     "no COMPLETE cells in the snapshot bank"),
            "rows": rows}


def gatereg_trial_dirs(key: str, *, round_no: int = GATEREG_ROUND,
                       root: str = GATEREG_TRIALS) -> list[str]:
    """The gate_reg trial dirs that RETAINED an object (tune_arm keeps
    object.pt + summary.json per trial) — the liveness sub-mode's targets."""
    base_dir = Path(root.format(key=key))
    if not base_dir.is_dir():
        return []
    return sorted(str(d) for d in base_dir.iterdir()
                  if d.is_dir() and d.name.startswith(f"r{round_no}_t")
                  and (d / "object.pt").is_file()
                  and (d / "summary.json").is_file())


def derived_object_dirs(key: str, root: str = DERIVED_ROOT,
                        kinds=DERIVED_KINDS) -> dict[str, str]:
    """{derived_id: object.pt path} for every derived object of ``key`` —
    refusing a derived id that appears under two kinds (an id is one
    object)."""
    out: dict[str, str] = {}
    for kind in kinds:
        base_dir = Path(root) / kind / key
        if not base_dir.is_dir():
            continue
        for d in sorted(base_dir.iterdir()):
            p = d / "object.pt"
            if not p.is_file():
                continue
            if d.name in out:
                raise SystemExit(f"[derived] derived id {d.name!r} appears "
                                 f"under two kinds ({out[d.name]} and {p})")
            out[d.name] = str(p)
    return out


def want_objects(key: str, objects: str, *, mix_si: str = DEFAULT_MIX_SI
                 ) -> dict[str, str]:
    """{derived_id: object.pt} for ``objects`` ("auto" / "" = every derived
    object of ``key``; "mix" = every one whose id names ``mix_si`` — the
    pair's own V / P_resid / LoRA-sub, never another pair's; else a comma
    list, each of which must exist)."""
    found = derived_object_dirs(key)
    if not objects or objects == "auto":
        return dict(found)
    if objects == "mix":
        mine = {k: v for k, v in found.items() if k.startswith(f"{mix_si}__")}
        if not mine:
            raise SystemExit(f"[derived] no derived object of {mix_si!r} under "
                             f"{DERIVED_ROOT}/*/{key}/ (have {sorted(found)})")
        return mine
    want = {}
    for oid in [o for o in objects.split(",") if o]:
        if oid not in found:
            raise SystemExit(f"[derived] no derived object {oid!r} under "
                             f"{DERIVED_ROOT}/*/{key}/ (have {sorted(found)})")
        want[oid] = found[oid]
    return want


def parse_scales(scales: str) -> list[float]:
    vals = [float(s) for s in scales.split(",") if s.strip()]
    if not vals:
        raise SystemExit("[derived] --scales must list at least one scale")
    return vals


# --------------------------------------------------------------------------- #
# Implementations (run inside the containers)                                  #
# --------------------------------------------------------------------------- #


def _run_env() -> dict:
    """The in-container subprocess env: the fit app's (HF_HOME, PYTHONPATH,
    …) — W&B routing rides in from the W&B secret (WANDB_API_KEY /
    WANDB_ENTITY / WANDB_PROJECT), inherited as is."""
    return base._env()


def _run(cmd: list[str], tag: str) -> dict:
    env = _run_env()
    t0 = time.time()
    p = subprocess.run(cmd, cwd=RUN_CWD, env=env, capture_output=True,
                       text=True)
    # the FULL streams on the volume — the printed tail is a courtesy, the
    # file is the record (a truncated tail lost the report once)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    safe = tag.replace("/", "__")
    logs = Path(LOG_ROOT)
    logs.mkdir(parents=True, exist_ok=True)
    out_log = logs / f"{safe}__{stamp}.stdout.log"
    err_log = logs / f"{safe}__{stamp}.stderr.log"
    out_log.write_text(p.stdout or "", encoding="utf-8")
    err_log.write_text(p.stderr or "", encoding="utf-8")
    row = {"tag": tag, "rc": int(p.returncode),
           "wall_s": round(time.time() - t0, 1),
           "stdout_tail": p.stdout.splitlines()[-20:] if p.stdout else [],
           "stdout_log": str(out_log), "stderr_log": str(err_log)}
    if p.returncode != 0:
        row["stderr_tail"] = str(p.stderr[-STDERR_TAIL_CHARS:])
        print(f"[derived] {tag} FAILED rc={p.returncode} (full stderr: "
              f"{err_log}):\n{p.stderr[-STDERR_TAIL_CHARS:]}", flush=True)
    else:
        print(f"[derived] {tag} ok ({row['wall_s']}s)", flush=True)
    return row


def _fresh_volume() -> None:
    """Modal Volumes are snapshot-based: a container that is already mounted
    (a WARM container reused for a later call) keeps the snapshot it was
    mounted with and does not see what other containers committed since,
    unless it reloads. Every stage that CONSUMES persisted state — the
    merge reading the parts, production reading the merged sweep, the
    compose stage checking for V, a calibration or serial container reusing
    finished payloads / checkpoints — reloads FIRST, before opening any
    volume file: otherwise a warm merge container could report missing
    parts, and a warm production container could gate on a stale sweep of
    the same identity."""
    vol.reload()


def _start_committer(period_s: float = 300.0):
    """A daemon thread committing the volume every ``period_s`` so
    checkpoints and finished-multiplier payloads survive a killed
    container; returns the Event that stops it."""
    import threading

    done = threading.Event()

    def committer():
        while not done.wait(period_s):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001 — retried next tick
                print(f"[derived] vol.commit failed ({e}); will retry",
                      flush=True)

    threading.Thread(target=committer, daemon=True).start()
    return done


def calibration_summary(cal_path) -> dict | None:
    """The sweep's verdict-relevant numbers (attainable ED per multiplier,
    stop reasons, the selection, both identity verdicts) off a sweep file,
    so the operator reads them off the run's stdout; None when absent."""
    cal_path = Path(cal_path)
    if not cal_path.is_file():
        return None
    try:
        c = json.loads(cal_path.read_text())
    except ValueError:
        return None
    sw = c.get("sweep") or {}
    tic = c.get("teacher_identity_check") or {}
    return {
        "lr_mults": sw.get("lr_mults"),
        "finished_lr_mults": sw.get("finished_lr_mults"),
        "selected_lr_mult": sw.get("selected_lr_mult"),
        "complete": sw.get("complete"),
        "budget": sw.get("budget"),
        "table": sw.get("table"),
        "attainable_ed_content_selected": c.get("attainable_ed_content"),
        "identity_check_passed": tic.get("passed"),
        "teacher_hook_contract_passed": tic.get("teacher_hook_contract_passed"),
        "identity_report": tic.get("report"),
    }


def _resid_result(key: str, mix_si: str, ctl_si: str, row: dict) -> dict:
    """The residual out dir read back: file list, the distilled object's
    verdict / sha / multiplier when derived_meta.json exists, and the
    calibration summary when the sweep file exists."""
    out_dir = Path(resid_out_dir(key, mix_si))
    files = sorted(p.name for p in out_dir.iterdir()) if out_dir.is_dir() else []
    res = {"model": key, "mix_si": mix_si, "ctl_si": ctl_si, "app": APP_NAME,
           "status": "ok" if row["rc"] == 0 else "failed", "run": row,
           "out_dir": str(out_dir), "files": files}
    meta = out_dir / "derived_meta.json"
    if meta.is_file():
        try:
            m = json.loads(meta.read_text())
            res["recovery_verdict"] = m.get("recovery_verdict")
            res["object_sha256"] = m.get("object_sha256")
            res["lr_mult"] = (m.get("trainer") or {}).get("lr_mult")
        except ValueError:
            pass
    cal = calibration_summary(out_dir / "calibration_controls.json")
    if cal is not None:
        res["calibration"] = cal
    return res


def _residual_impl(item: tuple) -> dict:
    """(key, mix_si, ctl_si, no_production, lr_mult[s], wandb[, epochs_cap,
    patience, checkpoint_every]) — the SERIAL path: one model's whole
    residual pipeline (``all``) in one container, committing the volume
    every few minutes so calibration artifacts (each finished lr
    multiplier's controls, every checkpoint) survive a late crash; a retry
    of the same item reuses the finished multipliers and resumes the fit
    that was running."""
    key, mix_si, ctl_si, no_production, lr_mult, wandb = item[:6]
    epochs_cap = int(item[6]) if len(item) > 6 and item[6] is not None else None
    patience = int(item[7]) if len(item) > 7 and item[7] is not None else None
    ckpt_every = int(item[8]) if len(item) > 8 and item[8] else None
    cap_tokens = int(item[9]) if len(item) > 9 and item[9] else None
    _fresh_volume()   # finished payloads / checkpoints committed by others
    if wandb:
        # refuse at container start rather than log to a foreign account
        base.require_wandb_entity()
    base._ensure_model(key)
    done = _start_committer()
    cmd = residual_command(key, mix_si, ctl_si, no_production=bool(no_production),
                           lr_mult=lr_mult, wandb=bool(wandb),
                           python=sys.executable, distill_epochs_cap=epochs_cap,
                           distill_patience=patience,
                           checkpoint_every=ckpt_every,
                           capture_budget_tokens=cap_tokens)
    row = _run(cmd, f"residual/{key}/{mix_si}")
    done.set()
    vol.commit()
    return _resid_result(key, mix_si, ctl_si, row)


def _calibration_impl(item: tuple) -> dict:
    """(key, mix_si, ctl_si, lr_mult, wandb, epochs_cap, patience,
    checkpoint_every) — ONE multiplier's calibration controls in one
    container (the fan-out's unit): the part file lands under calib_parts/
    lrmult<m>/; a re-issue of the same item is a no-op when the part is
    finished under this identity (the CLI's resume), else it resumes the
    control that was running from its checkpoint."""
    key, mix_si, ctl_si, lr_mult, wandb, epochs_cap, patience, ckpt_every = item[:8]
    cap_tokens = int(item[8]) if len(item) > 8 and item[8] else None
    _fresh_volume()   # V from the compose stage; a finished part / checkpoint
    if wandb:
        base.require_wandb_entity()
    base._ensure_model(key)
    done = _start_committer()
    cmd, part = calibrate_command(
        key, mix_si, ctl_si, lr_mult=float(lr_mult), wandb=bool(wandb),
        python=sys.executable, distill_epochs_cap=epochs_cap,
        distill_patience=patience, checkpoint_every=ckpt_every,
        capture_budget_tokens=cap_tokens)
    row = _run(cmd, f"calibrate/{key}/{mix_si}/lrmult{mult_key(lr_mult)}")
    done.set()
    vol.commit()
    res = {"model": key, "mix_si": mix_si, "ctl_si": ctl_si, "app": APP_NAME,
           "lr_mult": float(lr_mult), "part": part,
           "status": "ok" if row["rc"] == 0 else "failed", "run": row}
    cal = calibration_summary(part)
    if cal is not None:
        res["calibration"] = cal
    return res


def _compose_impl(item: tuple) -> dict:
    """(key, mix_si, ctl_si) — CPU: V composed once, before the fan-out. A V
    already on the volume STANDS only if ``verify-v`` re-admits it against
    the live bank (the shared derived-object verdict, parents re-verified);
    a torn publish (object without sidecar), a parent refit under the same
    cell id, or any other verdict problem triggers a REBUILD here, in the
    serialized compose stage — never a loop of downstream refusals. The
    compose is committed only when it succeeded."""
    key, mix_si, ctl_si = item
    _fresh_volume()
    cmd, v_path = compose_command(key, mix_si, ctl_si, python=sys.executable)
    res = {"model": key, "mix_si": mix_si, "ctl_si": ctl_si, "app": APP_NAME,
           "teacher_object": v_path}
    # a V under this id that names OTHER parents is another pair's object:
    # refused here, never recomposed over (the CLI refuses too)
    other = sidecar_parent_problems(Path(v_path).with_name("derived_meta.json"),
                                    mix_si, ctl_si, "psr_ce_all")
    if other:
        print(f"[derived] V at {v_path} belongs to another pair — REFUSED:\n  - "
              + "\n  - ".join(other), flush=True)
        return {**res, "status": "refused", "composed": False, "problems": other}
    if Path(v_path).is_file():
        check = _run(verify_v_command(key, mix_si, ctl_si, python=sys.executable),
                     f"verify-v/{key}/{mix_si}")
        if check["rc"] == 0:
            print(f"[derived] V already on the volume and valid against the "
                  f"live bank ({v_path}); not recomposed", flush=True)
            return {**res, "status": "ok", "composed": False, "verify": check}
        print(f"[derived] V on the volume FAILS the derived-object verdict "
              f"(rc={check['rc']}) — recomposing {v_path}", flush=True)
        res["verify"] = check
    row = _run(cmd, f"compose-v/{key}/{mix_si}")
    if row["rc"] == 0:
        vol.commit()
    return {**res, "status": "ok" if row["rc"] == 0 else "failed",
            "composed": row["rc"] == 0, "run": row}


def _merge_impl(item: tuple) -> dict:
    """(key, mix_si, ctl_si, lr_mults_csv) — CPU: merge the parts into the
    canonical calibration_controls.json (same selection rule); refuses
    when a part is missing, incomplete, duplicated or of another identity."""
    key, mix_si, ctl_si, mults_csv = item
    _fresh_volume()   # the parts were committed by other containers
    mults = parse_mults(mults_csv)
    cmd, out = merge_command(key, mix_si, ctl_si, lr_mults=mults,
                             python=sys.executable)
    row = _run(cmd, f"merge-calibration/{key}/{mix_si}")
    vol.commit()
    res = {"model": key, "mix_si": mix_si, "ctl_si": ctl_si, "app": APP_NAME,
           "lr_mults": mults, "calibration_controls": out,
           "status": "ok" if row["rc"] == 0 else "failed", "run": row}
    cal = calibration_summary(out)
    if cal is not None:
        res["calibration"] = cal
    return res


def _production_impl(item: tuple) -> dict:
    """(key, mix_si, ctl_si, wandb, epochs_cap, patience, checkpoint_every)
    — PRODUCTION ONLY: refuses (status ``refused``, no model staged) unless
    the volume holds a complete sweep at this budget with a fingerprinted
    selected payload (:func:`calibration_precheck`); then ``distill``,
    which compares the full identity before training and distills at the
    selected multiplier, resumable from its checkpoint."""
    key, mix_si, ctl_si, wandb, epochs_cap, patience, ckpt_every = item[:7]
    cap_tokens = int(item[7]) if len(item) > 7 and item[7] else None
    _fresh_volume()   # the merged sweep was committed by the merge container
    cal_path = f"{resid_out_dir(key, mix_si)}/calibration_controls.json"
    problems = calibration_precheck(cal_path, epochs_cap=epochs_cap,
                                    patience=patience)
    if problems:
        print(f"[derived] production REFUSED for {key}/{mix_si}:\n  - "
              + "\n  - ".join(problems), flush=True)
        return {"model": key, "mix_si": mix_si, "ctl_si": ctl_si,
                "app": APP_NAME, "status": "refused",
                "calibration_controls": cal_path, "problems": problems}
    if wandb:
        base.require_wandb_entity()
    base._ensure_model(key)
    done = _start_committer()
    cmd = distill_command(key, mix_si, ctl_si, wandb=bool(wandb),
                          python=sys.executable, distill_epochs_cap=epochs_cap,
                          distill_patience=patience, checkpoint_every=ckpt_every,
                          capture_budget_tokens=cap_tokens)
    row = _run(cmd, f"production/{key}/{mix_si}")
    done.set()
    vol.commit()
    return _resid_result(key, mix_si, ctl_si, row)


def _identity_impl(item: tuple) -> dict:
    """(key, mix_si, ctl_si) — the identity-only preflight: the write
    identity + hook contracts on the planted composite and reference, and
    the hook contract of the persisted V reloaded from the volume. No
    W&B (nothing to log, so no entity guard), no distillation. The
    per-layer report rides back inline next to its volume path."""
    key, mix_si, ctl_si = item[:3]
    cap_tokens = int(item[3]) if len(item) > 3 and item[3] else None
    base._ensure_model(key)
    cmd = identity_command(key, mix_si, ctl_si, python=sys.executable,
                           capture_budget_tokens=cap_tokens)
    row = _run(cmd, f"identity/{key}/{mix_si}")
    vol.commit()
    out_dir = Path(DERIVED_ROOT) / "psr_residual" / key / \
        f"{mix_si}__psr_ce_all_minus_ctl_distilled"
    rep_path = out_dir / "identity_check_report.json"
    res = {"model": key, "mix_si": mix_si, "ctl_si": ctl_si, "app": APP_NAME,
           "status": "ok" if row["rc"] == 0 else "failed", "run": row,
           "report": str(rep_path)}
    if rep_path.is_file():
        try:
            rep = json.loads(rep_path.read_text())
        except ValueError:
            rep = None
        if rep:
            res["passed"] = rep.get("passed")
            res["problems"] = rep.get("problems")
            res["stream_dtype"] = (rep.get("identity") or {}).get(
                "model", {}).get("dtype")
            tc = rep.get("teacher_hook_contract") or {}
            res["teacher_hook_contract_passed"] = tc.get("passed")
            res["teacher_object"] = tc.get("object")
            res["per_layer"] = {
                l: {"write_disc": r.get("rms_write_discrepancy"),
                    "write_tol": r.get("tolerance"),
                    "composite_max_half_ulps": (r.get("hook_contract_composite")
                                                or {}).get("max_err_half_ulps"),
                    "composite_violations": (r.get("hook_contract_composite")
                                             or {}).get("n_violations"),
                    "reference_max_half_ulps": (r.get("hook_contract_reference")
                                                or {}).get("max_err_half_ulps"),
                    "reference_violations": (r.get("hook_contract_reference")
                                             or {}).get("n_violations"),
                    "v_max_half_ulps": ((tc.get("per_layer") or {}).get(l)
                                        or {}).get("max_err_half_ulps"),
                    "v_violations": ((tc.get("per_layer") or {}).get(l)
                                     or {}).get("n_violations"),
                    "v_write_active_frac": ((tc.get("per_layer") or {}).get(l)
                                            or {}).get("write_active_frac"),
                    "n_elements": (r.get("hook_contract_composite")
                                   or {}).get("n_elements")}
                for l, r in (rep.get("per_layer") or {}).items()}
    return res


def _score_impl(item: tuple) -> dict:
    """(key, objects_csv|"auto"|"mix", scales_csv, probe_vers_csv,
    include_parents, ce_pools_csv|""[, mix_si, ctl_si, ce_skip_existing])
    — probe + CE for the derived objects of one model. The six-field item
    is the 100k pair's (defaults: DEFAULT_MIX_SI / DEFAULT_CTL_SI, no
    skipping) and runs exactly the historical command sequence; the pair
    fields select the parents and the control's CE pools, and
    ``ce_skip_existing`` keeps a CE file already on the volume (the bare
    row and any parent another pair scored) instead of re-scoring it."""
    key, objects, scales_csv, ver, include_parents, ce_pools_csv = item[:6]
    mix_si = item[6] if len(item) > 6 and item[6] else DEFAULT_MIX_SI
    ctl_si = item[7] if len(item) > 7 and item[7] else DEFAULT_CTL_SI
    skip_existing = bool(item[8]) if len(item) > 8 else False
    base._ensure_model(key)
    scales = parse_scales(scales_csv)
    vers = parse_probe_versions(ver)
    want = want_objects(key, objects, mix_si=mix_si)
    if ce_pools_csv:
        pools = {}
        for spec in ce_pools_csv.split(","):
            tag, path = spec.split(":", 1)
            pools[tag] = path
    else:
        pools = ce_pools_for(ctl_si)
    pools = {t: p.format(key=key) for t, p in pools.items()}
    rows = []
    for tag, path in list(pools.items()):
        if not Path(path).is_file():
            rows.append({"tag": f"pool/{tag}", "rc": -1,
                         "error": f"{path} not staged — CE leg unavailable"})
            print(f"[derived] pool {tag} UNAVAILABLE ({path})", flush=True)
            pools.pop(tag)

    def ce_row(cmd, out, tag, *, ptag, ppath, obj_path, cell, scale):
        if skip_existing and Path(out).is_file():
            problems = ce_reuse_problems(out, key=key, pool_tag=ptag,
                                         pool_path=ppath, obj_path=obj_path,
                                         cell=cell, scale=scale)
            if not problems:
                print(f"[derived] {tag}: {out} already on the volume and "
                      "records exactly this run's inputs — kept "
                      "(--ce-skip-existing)", flush=True)
                return {"tag": tag, "rc": 0, "skipped": True, "out": str(out)}
            print(f"[derived] {tag}: {out} exists but is NOT this run's "
                  "result — re-scoring:\n  - " + "\n  - ".join(problems),
                  flush=True)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        return _run(cmd, tag)

    for oid, opath in want.items():
        for s in scales:
            for v in vers:
                cmd, _out = probe_command(key, opath, oid, scale=s, ver=v,
                                          python=sys.executable)
                Path(_out).parent.mkdir(parents=True, exist_ok=True)
                # the single-probe tag is the historical one; a multi-probe
                # call names the version so the log files do not collide
                rows.append(_run(cmd, f"probe/{oid}/x{s:g}" if len(vers) == 1
                                 else f"probe/{oid}/{v}/x{s:g}"))
            for ptag, ppath in pools.items():
                cmd, _out = ce_command(key, opath, oid, ptag, ppath, scale=s,
                                       python=sys.executable)
                rows.append(ce_row(cmd, _out, f"ce/{oid}/{ptag}/x{s:g}",
                                   ptag=ptag, ppath=ppath, obj_path=opath,
                                   cell=oid, scale=s))
        vol.commit()
    if include_parents:
        for cell in parent_cells(mix_si, ctl_si):
            # artifacts only: PSR cells ship an object.pt.provenance.json
            # SIDECAR beside object.pt, which must never read as a second
            # artifact (the round-2 harvest bug, fixed at every discovery
            # site).
            cand = sorted(q for q in (Path(f"/vol/out/{key}/cells_all") / cell)
                          .glob("object.*") if q.suffix != ".json")
            if len(cand) != 1:
                rows.append({"tag": f"ce/{cell}", "rc": -1,
                             "error": f"no unambiguous artifact for {cell}"})
                continue
            for ptag, ppath in pools.items():
                cmd, _out = ce_command(key, str(cand[0]), cell, ptag, ppath,
                                       scale=1.0, python=sys.executable)
                rows.append(ce_row(cmd, _out, f"ce/{cell}/{ptag}/x1", ptag=ptag,
                                   ppath=ppath, obj_path=str(cand[0]),
                                   cell=cell, scale=1.0))
        for ptag, ppath in pools.items():
            cmd, _out = ce_command(key, None, "bare", ptag, ppath, scale=1.0,
                                   python=sys.executable)
            rows.append(ce_row(cmd, _out, f"ce/bare/{ptag}", ptag=ptag,
                               ppath=ppath, obj_path=None, cell=None, scale=1.0))
        vol.commit()
    vol.commit()
    bad = [r for r in rows if r["rc"] != 0]
    return {"model": key, "app": APP_NAME, "objects": sorted(want),
            "scales": scales, "probe": ver, "pools": sorted(pools),
            "status": "ok" if not bad else "failed",
            "n_ok": len(rows) - len(bad), "n_bad": len(bad), "rows": rows}


def _lora_subtract_impl(item: tuple) -> dict:
    """(key, mix_si, ctl_si) — CPU: the LoRA task-vector subtraction on the
    volume's bank (lora_subtract_command; ``--keep-valid`` makes a re-issue
    a no-op when the publication stands against the live bank for THESE
    parents). Committed only when the builder succeeded; the sidecar's
    object sha and parents ride back inline."""
    key, mix_si, ctl_si = item
    _fresh_volume()   # the parents' latest publication
    cmd, obj_path = lora_subtract_command(key, mix_si, ctl_si,
                                          python=sys.executable)
    row = _run(cmd, f"lora-subtract/{key}/{mix_si}")
    if row["rc"] == 0:
        vol.commit()
    res = {"model": key, "mix_si": mix_si, "ctl_si": ctl_si, "app": APP_NAME,
           "object": obj_path, "status": "ok" if row["rc"] == 0 else "failed",
           "run": row}
    meta = Path(obj_path).with_name("derived_meta.json")
    if row["rc"] == 0 and meta.is_file():
        try:
            m = json.loads(meta.read_text())
        except ValueError:
            m = {}
        res["object_sha256"] = m.get("object_sha256")
        res["scale_matched"] = m.get("scale_matched")
        res["parents"] = {r: {k: (m.get(r) or {}).get(k)
                              for k in ("cell_id", "object_sha256",
                                        "fit_fingerprint")}
                          for r in ("parent_pos", "parent_neg")}
    return res


def _liveness_impl(item: tuple) -> dict:
    """(key, targets_csv|"auto", n_probe[, mix_si]) — per-layer gate liveness
    for the gate_reg r3 trial objects (probe pool = the design's data cell;
    trial dirs carry no identity sidecar, so --allow-bare, sha recorded) and
    for the psr-gated derived objects of ``key`` (probe pool = the pair's
    MIX parent pool — the 100k one for the three-field item; identity-
    gated). Writes gate_liveness.json beside each."""
    key, targets, n_probe = item[:3]
    mix_si = item[3] if len(item) > 3 and item[3] else DEFAULT_MIX_SI
    base._ensure_model(key)
    trial_dirs = gatereg_trial_dirs(key)
    derived = derived_object_dirs(key, kinds=("psr_paired_write",
                                              "psr_residual"))
    rows = []
    explicit = bool(targets) and targets != "auto"
    if explicit:
        want = {t for t in targets.split(",") if t}
        trial_dirs = [d for d in trial_dirs if Path(d).name in want]
        derived = {k: v for k, v in derived.items() if k in want}
        missing = sorted(want - {Path(d).name for d in trial_dirs} - set(derived))
        if missing:
            # a requested object that resolves to nothing is a failure, not
            # a shorter run: the two-object liveness could otherwise have
            # reported ok on one — or none
            rows.append({"tag": "liveness/targets", "rc": -1,
                         "error": f"requested targets not found under {GATEREG_TRIALS.format(key=key)} "
                                  f"or {DERIVED_ROOT}/psr_*/{key}/: {missing}"})
    trial_pool = f"/vol/pool/{key}/pairs_{GATEREG_SI}.jsonl"
    if trial_dirs and Path(trial_pool).is_file():
        cmd = liveness_command(key, trial_dirs, pairs_path=trial_pool,
                               n_probe=int(n_probe), allow_bare=True,
                               python=sys.executable)
        rows.append(_run(cmd, f"liveness/gatereg/{key}/{len(trial_dirs)}-trials"))
    elif trial_dirs:
        rows.append({"tag": "liveness/gatereg", "rc": -1,
                     "error": f"{trial_pool} not staged"})
    else:
        print(f"[derived] no r{GATEREG_ROUND} trial objects under "
              f"{GATEREG_TRIALS.format(key=key)} — gate_reg liveness stays "
              "declared unavailable", flush=True)
    derived_pool = f"/vol/pool/{key}/pairs_{mix_si}.jsonl"
    if derived and Path(derived_pool).is_file():
        cmd = liveness_command(key, [str(Path(p).parent) for p in derived.values()],
                               pairs_path=derived_pool, n_probe=int(n_probe),
                               allow_bare=False, python=sys.executable)
        rows.append(_run(cmd, f"liveness/derived/{key}/{len(derived)}-objects"))
    elif derived and explicit:
        rows.append({"tag": "liveness/derived", "rc": -1,
                     "error": f"{derived_pool} not staged — the requested "
                              f"derived objects {sorted(derived)} were not probed"})
    vol.commit()
    bad = [r for r in rows if r["rc"] != 0]
    return {"model": key, "app": APP_NAME, "trial_dirs": trial_dirs,
            "derived": sorted(derived), "status": "ok" if not bad else "failed",
            "n_ok": len(rows) - len(bad), "n_bad": len(bad), "rows": rows}


def _mmlu_impl(item: tuple) -> dict:
    """(key, objects_csv|"auto", run_label) — one model's derived-object MMLU
    leg in one container, under this app's own namespace (mmlu_command)."""
    key, objects, run_label = item
    base._ensure_model(key)
    want = want_objects(key, objects)
    cmd, out_dir = mmlu_command(key, want, run_label, python=sys.executable)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    row = _run(cmd, f"mmlu/{key}/{run_label}")
    vol.commit()
    return {"model": key, "app": APP_NAME, "run_label": run_label,
            "objects": sorted(want), "out_dir": out_dir,
            "status": "ok" if row["rc"] == 0 else "failed", "run": row}


def _snapshot_verify_impl(item: tuple) -> dict:
    """(key, cells_csv|"", out_root|"") — CPU, read-only on the bank: the
    ``<key>_r1`` snapshot bank re-verified (verify_snapshot_bank), the
    verdicts recorded under SNAPSHOT_VERIFY_ROOT/<key>/."""
    key, cells_csv, out_root = item
    base._repo_imports()
    dst_bank = Path(f"/vol/{base.snapshot_bank_root(out_root or f'out/{key}')}")
    cells = [c for c in (cells_csv or "").split(",") if c] or None
    res = verify_snapshot_bank(dst_bank, cells)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    record = Path(snapshot_verify_record_path(key, stamp))
    record.parent.mkdir(parents=True, exist_ok=True)
    out = {**res, "model": key, "app": APP_NAME, "record": str(record)}
    record.write_text(json.dumps(out, indent=1, default=str) + "\n")
    vol.commit()
    for r in res["rows"]:
        print(f"[derived/snapshot-verify] {key}/{r['cell_id']}: {r['status']}"
              + (f" — {r['error']}" if r.get("error") else ""), flush=True)
    return out


# --------------------------------------------------------------------------- #
# Functions (std / qwen image variants)                                        #
# --------------------------------------------------------------------------- #


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=SERIAL_TIMEOUT_S, max_containers=MAX_CONTAINERS, **_COMMON)
def residual_h200_std(item: tuple) -> dict:
    return _residual_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=SERIAL_TIMEOUT_S, max_containers=MAX_CONTAINERS, **_COMMON)
def residual_h200_qwen(item: tuple) -> dict:
    return _residual_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=CALIB_TIMEOUT_S, max_containers=MAX_CONTAINERS, **_COMMON)
def calibrate_h200_std(item: tuple) -> dict:
    return _calibration_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=CALIB_TIMEOUT_S, max_containers=MAX_CONTAINERS, **_COMMON)
def calibrate_h200_qwen(item: tuple) -> dict:
    return _calibration_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=PROD_TIMEOUT_S, max_containers=MAX_CONTAINERS, **_COMMON)
def production_h200_std(item: tuple) -> dict:
    return _production_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=PROD_TIMEOUT_S, max_containers=MAX_CONTAINERS, **_COMMON)
def production_h200_qwen(item: tuple) -> dict:
    return _production_impl(item)


# CPU-only: V composed once before the fan-out, and the sweep merge (json
# in, json out) after it; no GPU, no model load.
@app.function(image=image_std, cpu=4, memory=16384, timeout=1800, **_COMMON)
def compose_cpu(item: tuple) -> dict:
    return _compose_impl(item)


@app.function(image=image_std, cpu=4, memory=16384, timeout=1800, **_COMMON)
def calib_merge_cpu(item: tuple) -> dict:
    return _merge_impl(item)


# CPU-only: the LoRA task-vector subtraction (tensor bookkeeping on the two
# parents' factors; no model load).
@app.function(image=image_std, cpu=4, memory=16384, timeout=1800, **_COMMON)
def lora_subtract_cpu(item: tuple) -> dict:
    return _lora_subtract_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=7200, max_containers=MAX_CONTAINERS, **_COMMON)
def identity_h200_std(item: tuple) -> dict:
    return _identity_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=7200, max_containers=MAX_CONTAINERS, **_COMMON)
def identity_h200_qwen(item: tuple) -> dict:
    return _identity_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=14400, max_containers=MAX_CONTAINERS, **_COMMON)
def score_h200_std(item: tuple) -> dict:
    return _score_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=14400, max_containers=MAX_CONTAINERS, **_COMMON)
def score_h200_qwen(item: tuple) -> dict:
    return _score_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=7200, max_containers=MAX_CONTAINERS, **_COMMON)
def liveness_h200_std(item: tuple) -> dict:
    return _liveness_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=7200, max_containers=MAX_CONTAINERS, **_COMMON)
def liveness_h200_qwen(item: tuple) -> dict:
    return _liveness_impl(item)


@app.function(gpu="H200", image=image_std, cpu=8, memory=65536,
              timeout=14400, max_containers=MAX_CONTAINERS, **_COMMON)
def mmlu_h200_std(item: tuple) -> dict:
    return _mmlu_impl(item)


@app.function(gpu="H200", image=image_qwen, cpu=8, memory=131072,
              timeout=14400, max_containers=MAX_CONTAINERS, **_COMMON)
def mmlu_h200_qwen(item: tuple) -> dict:
    return _mmlu_impl(item)


# CPU-only bookkeeping against the volume (no GPU, no model load).
@app.function(image=image_std, cpu=4, memory=16384, timeout=7200, **_COMMON)
def snapshot_verify_cpu(item: tuple) -> dict:
    return _snapshot_verify_impl(item)


MODES = ("residual-distill", "calibration", "residual-distill-prod", "identity",
         "score", "liveness", "mmlu", "snapshot-verify", "lora-subtract")


def residual_pipeline(key: str, mults: list[float], *, mix_si: str, ctl_si: str,
                      wandb: bool, epochs_cap: int, patience: int,
                      checkpoint_every: int, production: bool,
                      compose_fn, calibrate_fn, merge_fn, production_fn,
                      log=print, capture_budget_tokens: int | None = None
                      ) -> dict:
    """ONE model's fan-out, run to completion: compose V once on CPU, spawn
    a calibration container per multiplier (all in flight at once), wait
    for all of them, merge on CPU, then — when ``production`` — one
    production container gated on the merged sweep. Stops at the first
    failed stage with that stage named. The function objects are
    parameters so the orchestration is testable without Modal. Every model
    runs this in its own thread (see the entrypoint), so one model's
    production never waits on another's calibration."""
    out = {"model": key, "mix_si": mix_si, "ctl_si": ctl_si, "app": APP_NAME,
           "lr_mults": [float(m) for m in mults]}
    if capture_budget_tokens:
        out["capture_budget_tokens"] = int(capture_budget_tokens)
    cap = _cap_tail(capture_budget_tokens)
    composed = compose_fn.spawn((key, mix_si, ctl_si)).get()
    out["compose"] = composed
    if composed.get("status") != "ok":
        return {**out, "status": "failed", "failed_stage": "compose"}
    log(f"[derived] {key}: teacher V {'composed' if composed.get('composed') else 'reused'}"
        f" ({composed.get('teacher_object')})", flush=True)
    parts = [(m, calibrate_fn.spawn((key, mix_si, ctl_si, float(m), bool(wandb),
                                     int(epochs_cap), int(patience),
                                     int(checkpoint_every)) + cap))
             for m in mults]
    log(f"[derived] {key}: {len(parts)} calibration container(s) spawned "
        f"(lr_mult {[mult_key(m) for m in mults]})", flush=True)
    part_rows = []
    for m, call in parts:
        r = call.get()
        part_rows.append(r)
        log(f"[derived] {key}: calibration lr_mult={mult_key(m)} "
            f"{r.get('status')} ({(r.get('run') or {}).get('wall_s')}s)",
            flush=True)
    out["parts"] = part_rows
    if any(r.get("status") != "ok" for r in part_rows):
        return {**out, "status": "failed", "failed_stage": "calibration"}
    merged = merge_fn.spawn((key, mix_si, ctl_si,
                             ",".join(mult_key(m) for m in mults))).get()
    out["merge"] = merged
    if merged.get("status") != "ok":
        return {**out, "status": "failed", "failed_stage": "merge"}
    log(f"[derived] {key}: sweep merged, selected lr_mult "
        f"{(merged.get('calibration') or {}).get('selected_lr_mult')}",
        flush=True)
    if not production:
        return {**out, "status": "ok", "calibration": merged.get("calibration")}
    prod = production_fn.spawn((key, mix_si, ctl_si, bool(wandb),
                                int(epochs_cap), int(patience),
                                int(checkpoint_every)) + cap).get()
    out["production"] = prod
    if prod.get("status") != "ok":
        return {**out, "status": "failed", "failed_stage": "production"}
    return {**out, "status": "ok",
            **{k: prod.get(k) for k in ("recovery_verdict", "object_sha256",
                                         "lr_mult", "out_dir", "calibration")}}


@app.local_entrypoint()
def main(mode: str = "residual-distill", model: str = "",
         mix_si: str = "constv2-mix-100k", ctl_si: str = "constv2ctl-100k",
         objects: str = "auto", scales: str = "1.0,2.0", probe: str = "v2",
         lr_mults: str = "0.3,1,3", distill_epochs_cap: int = 60,
         distill_patience: int = 5, checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
         serial: bool = False, no_parents: bool = False,
         ce_pools: str = "", wandb: bool = True, n_probe: int = 32,
         run_label: str = "", cells: str = "", out_root: str = "",
         app_name: str = "", capture_budget_tokens: int = 0,
         ce_skip_existing: bool = False):
    if mode not in MODES:
        raise SystemExit(f"[derived] unknown --mode {mode!r}; one of {MODES}")
    # The app a run names is the app this tree IS (its name carries the
    # commit): an --app-name that names another deployment refuses, so a
    # run can never be read as having executed code it did not (the
    # 2026-09-04 trap: a run against the wrong deployment silently executed
    # stale code and produced unstamped sidecars).
    if not app_name:
        raise SystemExit(f"[derived] --app-name is required: pass the app this "
                         f"tree names, {APP_NAME!r} (commit pin {SHA16}), so the "
                         "run records which code it executed")
    if app_name != APP_NAME:
        raise SystemExit(f"[derived] --app-name {app_name!r} is not this tree's "
                         f"app {APP_NAME!r} (commit pin {SHA16}); run from the "
                         "worktree whose HEAD names that app")
    require_canonical_pair(mix_si, ctl_si)
    cap_tokens = int(capture_budget_tokens) if capture_budget_tokens else None
    cap = _cap_tail(cap_tokens)
    # Every fit-bearing key EXCEPT those the fits table marks
    # derived_eligible=False: the instruct-direct `_it_doc` lanes carry
    # only mix-full — no mix-100k / ctl-100k parents — so no derived
    # operation is valid for them; a bare sweep must not pull them in, and
    # naming them explicitly refuses by the same list.
    base_keys = sorted(k for k, s in MODELS.items()
                       if not s.get("probe_only")
                       and s.get("derived_eligible", True))
    keys = [m for m in model.split(",") if m] or base_keys
    for k in keys:
        if k not in base_keys:
            raise SystemExit(f"[derived] --model must be a base key: {base_keys}")
    print(f"[derived] app {APP_NAME} (commit pin {SHA16}); pair {mix_si} − "
          f"{ctl_si}" + (f"; capture budget {cap_tokens} official tokens per "
                         "source" if cap_tokens else ""), flush=True)
    if mode in ("residual-distill", "calibration") and not serial:
        mults = parse_mults(lr_mults)
        print(f"[derived] FAN-OUT: one calibration container per lr_mult "
              f"{[mult_key(m) for m in mults]} per model, CPU merge, then "
              f"{'production' if mode == 'residual-distill' else 'no production'}; "
              f"budget cap {int(distill_epochs_cap)} epochs / patience "
              f"{int(distill_patience)}; checkpoint every "
              f"{int(checkpoint_every)} steps", flush=True)
        from concurrent.futures import ThreadPoolExecutor

        def one(k):
            return residual_pipeline(
                k, mults, mix_si=mix_si, ctl_si=ctl_si, wandb=bool(wandb),
                epochs_cap=int(distill_epochs_cap), patience=int(distill_patience),
                checkpoint_every=int(checkpoint_every),
                production=(mode == "residual-distill"),
                capture_budget_tokens=cap_tokens,
                compose_fn=compose_cpu,
                calibrate_fn=_fn_pair(k, calibrate_h200_std,
                                           calibrate_h200_qwen),
                merge_fn=calib_merge_cpu,
                production_fn=_fn_pair(k, production_h200_std,
                                            production_h200_qwen))

        # one thread per model: every model's parts go out at once and a
        # model's merge/production never waits on another's calibration
        with ThreadPoolExecutor(max_workers=max(1, len(keys))) as ex:
            futures = [(k, ex.submit(one, k)) for k in keys]
            results = [(k, f.result()) for k, f in futures]
        bad = 0
        for k, r in results:
            print(f"[derived/{mode}] {k}: {json.dumps(r, indent=1, default=str)}")
            bad += r.get("status") != "ok"
        if bad:
            raise SystemExit(f"[derived/{mode}] {bad} model(s) failed")
        return
    if mode in ("residual-distill", "calibration"):
        no_prod = mode == "calibration"
        if not lr_mults.strip():
            raise SystemExit("[derived] --lr-mults must list at least one "
                             "multiplier (default 0.3,1,3)")
        print(f"[derived] SERIAL: one container per model; lr-mult sweep "
              f"{lr_mults}; budget cap {int(distill_epochs_cap)} epochs / "
              f"patience {int(distill_patience)}; checkpoint every "
              f"{int(checkpoint_every)} steps", flush=True)
        calls = [(k, _fn_pair(k, residual_h200_std, residual_h200_qwen).spawn(
            (k, mix_si, ctl_si, no_prod, lr_mults, bool(wandb),
             int(distill_epochs_cap), int(distill_patience),
             int(checkpoint_every)) + cap))
            for k in keys]
    elif mode == "residual-distill-prod":
        print(f"[derived] PRODUCTION ONLY: one container per model running "
              f"`distill` on the volume's complete sweep; budget cap "
              f"{int(distill_epochs_cap)} epochs / patience "
              f"{int(distill_patience)}; checkpoint every "
              f"{int(checkpoint_every)} steps", flush=True)
        calls = [(k, _fn_pair(k, production_h200_std,
                                   production_h200_qwen).spawn(
            (k, mix_si, ctl_si, bool(wandb), int(distill_epochs_cap),
             int(distill_patience), int(checkpoint_every)) + cap))
            for k in keys]
    elif mode == "identity":
        calls = [(k, _fn_pair(k, identity_h200_std,
                                   identity_h200_qwen).spawn(
            (k, mix_si, ctl_si) + cap)) for k in keys]
    elif mode == "lora-subtract":
        calls = [(k, lora_subtract_cpu.spawn((k, mix_si, ctl_si)))
                 for k in keys]
    elif mode == "liveness":
        calls = [(k, _fn_pair(k, liveness_h200_std,
                                   liveness_h200_qwen).spawn(
            (k, objects, int(n_probe), mix_si))) for k in keys]
    elif mode == "mmlu":
        if not run_label:
            raise SystemExit("[derived/mmlu] --run-label required (e.g. r2d): "
                             "every output carries it in dir + filename so "
                             "runs never share a path")
        calls = [(k, _fn_pair(k, mmlu_h200_std, mmlu_h200_qwen).spawn(
            (k, objects, run_label))) for k in keys]
    elif mode == "snapshot-verify":
        calls = [(k, snapshot_verify_cpu.spawn((k, cells, out_root)))
                 for k in keys]
    else:
        parse_scales(scales)
        parse_probe_versions(probe)
        if not ce_pools:
            ce_pools_for(ctl_si)   # refuse an unregistered control up front
        calls = [(k, _fn_pair(k, score_h200_std, score_h200_qwen).spawn(
            (k, objects, scales, probe, not no_parents, ce_pools,
             mix_si, ctl_si, bool(ce_skip_existing))))
            for k in keys]
    bad = 0
    for k, c in calls:
        r = c.get()
        print(f"[derived/{mode}] {k}: {json.dumps(r, indent=1, default=str)}")
        bad += r.get("status") != "ok"
    if bad:
        raise SystemExit(f"[derived/{mode}] {bad} model(s) failed")
