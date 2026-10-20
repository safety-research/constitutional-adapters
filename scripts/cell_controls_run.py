"""Cell-id-keyed controls entry point.

Runs the three wired controls — mmlu (the capeval s2parity letter probe, run
under the cell's serving: the compiled object on the engine, or the SI as
``system=`` for a prompted cell), and xstest / or_bench_hard_1k (the capeval
judged runners) — for ONE DefenseCell, and feeds the control-slot aggregate
contract through ``controls_summary.json`` (which
``two_si_run.py aggregate`` fills from, with provenance validation).

The paired AlpacaEval control (``alpaca_eval``) and the never-implemented
WMDP sentinel were RETIRED with the rule of 2026-09-11 (one
canonical eval battery); both now refuse as unknown controls. Existing
stores still carry ``judged/alpaca_pair`` rows and the frozen store-identity
list still names ``alpaca_eval`` (see ``RESOLVED_CONTROLS_IDENTITY``).

Explicit-cell configs: a config carrying
``explicit_cells`` derives its registry as EXACTLY that ordered cell list —
``_registry`` routes through ``registry_for_controls`` →
``defense_cell.registry_from_cfg``, whose explicit branch validates every
id (arm in ARMS, SI resolvable at run time, artifacts staged) and never
fabricates a Cartesian ``prompted__<pseudo>`` cell. Every surface here keys
on ``cell.id``, so explicit cells serve/judge/emit unchanged; configs
without the key behave byte-identically to before the mode existed
(pinned by tests/test_sdfbase_transfer_registry.py).

GSM8K is refused too (PW3, user instruction U2 2026-09-03): the bespoke
0-shot ``s2parity/gsm8k_0shot`` probe this driver used to run is RETIRED in
favour of a standard harness — inspect_evals' stock ``gsm8k`` on the served
path (``served_capability_run.py generate --controls gsm8k --thinking off``),
whose ``controls_summary.json`` fills the same ``gsm8k`` slot with a payload
that names its instrument. Existing stores' gsm8k rows stay readable
(``DOMAIN_OF`` keeps the row task); nothing new generates it here.

Branch note: ``generate`` and ``judge`` need ``sisteer.capeval`` — the
runners live on branch ``capeval-runners-evals`` and real runs happen on a
checkout merging both branches (this file's capeval-facing calls are
written against that branch's REAL ``tiers/judged.py`` / ``judging.py`` API
as merged on evals-integration). Every capeval import is lazy with a loud
error naming that branch; ``emit`` and ``cells`` are capeval-free.

Demo (the first real compiled object handed over, 2026-08-19 —
cell ``safety_refusal__vector_all`` at the commit-independent defenses dir;
``fit_report.json`` is rewritten while the fit job runs, so artifact
PRESENCE in the cell dir is the truth, never that report)::

  # GPU node (integration checkout):
  uv run python scripts/cell_controls_run.py generate \
    --cell safety_refusal__vector_all \
    --defenses-dir $SHARED/exp/jailbreak/two_si_defenses/cells
  # grading host:
  uv run python scripts/cell_controls_run.py judge \
    --cell safety_refusal__vector_all \
    --defenses-dir $SHARED/exp/jailbreak/two_si_defenses/cells
  uv run python scripts/cell_controls_run.py emit --cell safety_refusal__vector_all

Subcommands:
  generate  — GPU node. ONE pinned model load; ``HFEngine(preloaded=...)``;
              §4.1 serving; parity probes (mmlu+gsm8k) + the judged
              generation spools (raw generations in the cell store's
              mode-0700 ``private/`` dir). All five controls share ONE
              CellStore under ``<out_root>/cells/<store_key_token(cell_id)>
              __controls__...`` (task-prefixed metrics).
  judge     — the grading host. Prints the coordination post FIRST, judges
              the spooled generations (cache-first, Low-Prio org — or
              OpenRouter with --judge-backend openrouter, the grok fallback
              judge), maps each runner's summary to the §4.3 slot payload,
              and writes the provenance-stamped ``controls_summary.json``.
  emit      — CPU, capeval-free: print this cell's controls block in the
              EXACT aggregate slot format (sentinel where not yet run).
  cells     — the sbatch array decode surface (cell_for_array_index).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

import jailbreak_run  # _load_cfg — the lane's one config loader
from two_si_run import _CONTROL_PENDING  # the shared pending-control sentinel

REPO = Path(__file__).resolve().parents[1]

# The three controls this entry point wires (of the config's four — gsm8k is
# the served standard's, PW3).
WIRED_CONTROLS = ("mmlu", "xstest", "or_bench_hard_1k")
#: ``resolved_config["controls"]`` — FROZEN at the historical
#: ``sorted(WIRED_CONTROLS)`` of 2026-09-02 (gsm8k AND alpaca_eval included)
#: so every existing HF controls store keeps its ``harness_config_hash`` (the
#: golden pinned by test_cell_controls_judge_provenance). The list is store
#: IDENTITY, not the run's requested controls (those are the manifest's
#: ``controls``); dropping a retired name from it would fork every store's
#: key for no new information.
RESOLVED_CONTROLS_IDENTITY = ("alpaca_eval", "gsm8k", "mmlu", "or_bench_hard_1k", "xstest")
#: control -> the capeval judged task id (tiers/judged.py TASK_* values;
#: pinned against the real module by the integration test below).
JUDGED_TASK_OF = {
    "or_bench_hard_1k": "judged/or_bench_hard_1k",
    "xstest": "judged/xstest_v2",
}
#: judged tasks sit outside the v0 domain registry — same mapping as
#: capeval_judged_run.DOMAIN_OF, plus the two s2parity probe domains.
DOMAIN_OF = {
    "judged/or_bench_hard_1k": "overrefusal_judged",
    "judged/xstest_v2": "refusal_calibration_judged",
    # RETIRED row task, kept so existing stores'
    # alpaca_pair rows resolve
    "judged/alpaca_pair": "chat_quality_judged",
    # RETIRED row task, kept so existing stores' gsm8k rows resolve (PW3)
    "s2parity/gsm8k_0shot": "math_s2parity",
    "s2parity/mmlu_letter": "knowledge_s2parity",
}
#: §4.3 slot payload contracts — the keys each control's slot MUST carry.
#: The judged three also carry the abstention accounting (bound + cap —
#: tiers/judged.py): ``n`` stays the FULL corpus and abstentions never
#: shrink it; ``n_graded = n - n_abstained`` is the denominator of every
#: point-estimate frac; the ``*_lo``/``*_hi`` bounds divide by ``n`` (the
#: same n the JUDGE_ABSTENTION_MAX_FRAC ceiling gates n_abstained
#: against, so payload readers can check the identity); and
#: ``abstained_by_reason`` splits judge refusals from max_tokens
#: truncations from transport refusals from content-filter blocks
#: (judging.ABSTENTION_REASONS); ``n_reasked`` counts malformed verdicts
#: re-asked once, so a rubric going flaky is a number, not a silent retry.
#: Passing the counts through is what keeps a subset point-estimate from
#: riding the payload unexplained.
_ABSTAIN_KEYS = ("n_graded", "n_abstained", "abstained_by_reason", "n_reasked")
#: (``gsm8k`` is no longer an HF-path payload: the served standard fills the
#: slot with ``contracts.GSM8K_SLOT_PAYLOAD_FIELDS`` — task/item_set/acc/n/… —
#: through ``tinycap.completion.merge_controls_summary``; PW3.)
PAYLOAD_KEYS = {
    "mmlu": ("acc", "n"),
    "xstest": ("n", *_ABSTAIN_KEYS, "safe", "unsafe", "rubric_sha"),
    "or_bench_hard_1k": ("n", *_ABSTAIN_KEYS,
                         "orb_refused_lo", "orb_refused_hi",
                         "full_compliance_frac", "partial_refusal_frac",
                         "full_refusal_frac", "rubric_sha"),
}
_CPI_LABELS = ("full_compliance", "partial_refusal", "full_refusal")


def parse_controls(spec: str) -> list[str]:
    """Validate the --controls csv. ``gsm8k`` (and the retired leg's other
    spellings) refuses pointing at the served standard (PW3); anything else
    unknown — the retired ``alpaca_eval`` and ``wmdp`` included — refuses
    naming the wired set."""
    controls = [c.strip() for c in spec.split(",") if c.strip()]
    if not controls:
        raise SystemExit("[cell-controls] --controls is empty")
    for c in controls:
        if c in ("gsm8k", "gsm8k_0shot", "s2parity/gsm8k_0shot"):
            raise _retired_gsm8k_refusal(f"--controls {c}", "the HF path's s2parity probe")
        if c not in WIRED_CONTROLS:
            raise SystemExit(
                f"[cell-controls] unknown control {c!r}; wired controls: "
                f"{list(WIRED_CONTROLS)}")
    return controls


def _require(control: str, result, keys) -> None:
    if not isinstance(result, dict):
        raise SystemExit(
            f"[cell-controls] {control}: runner returned "
            f"{type(result).__name__}, expected a dict — integration drift "
            "with the capeval judged runners (plan §3.2/§4.3)")
    missing = [k for k in keys if k not in result]
    if missing:
        raise SystemExit(
            f"[cell-controls] {control}: runner summary is missing {missing} "
            f"(got {sorted(result)}) — integration drift with the capeval "
            "judged runners (plan §3.2/§4.3)")


def slot_payload(control: str, result: dict) -> dict:
    """Map one judged runner's returned summary (tiers/judged.py
    ``judge_or_bench`` / ``judge_xstest`` — the real API
    on the integration merge) to the §4.3 aggregate slot payload. Any
    missing field is a loud refusal naming the drift — never a malformed
    slot in the aggregate."""
    if control == "or_bench_hard_1k":
        _require(control, result, ("n", *_ABSTAIN_KEYS, "orb_refused_lo",
                                   "orb_refused_hi", "fracs", "rubric_sha"))
        _require(control, result["fracs"], _CPI_LABELS)
        # fracs are over n_graded (see the runner); the bounds + counts
        # ride beside them so the denominator is never ambiguous.
        return {"n": result["n"],
                **{k: result[k] for k in _ABSTAIN_KEYS},
                "orb_refused_lo": result["orb_refused_lo"],
                "orb_refused_hi": result["orb_refused_hi"],
                **{f"{lab}_frac": result["fracs"][lab] for lab in _CPI_LABELS},
                "rubric_sha": result["rubric_sha"]}
    if control == "xstest":
        _require(control, result,
                 ("n", *_ABSTAIN_KEYS, "safe", "unsafe", "rubric_sha"))
        out = {"n": result["n"],
               **{k: result[k] for k in _ABSTAIN_KEYS},
               "rubric_sha": result["rubric_sha"]}
        for subset in ("safe", "unsafe"):
            _require(control, result[subset],
                     ("n", "n_graded", "n_abstained", "refused_lo",
                      "refused_hi", *_CPI_LABELS))
            out[subset] = {"n": result[subset]["n"],
                           "n_graded": result[subset]["n_graded"],
                           "n_abstained": result[subset]["n_abstained"],
                           "refused_lo": result[subset]["refused_lo"],
                           "refused_hi": result[subset]["refused_hi"],
                           **{f"{lab}_frac": result[subset][lab]
                              for lab in _CPI_LABELS}}
        return out
    raise SystemExit(f"[cell-controls] slot_payload: {control!r} is not a "
                     "judged control")


def _registry(cfg: dict, args):
    from sisteer.eval.cell_controls import registry_for_controls

    return registry_for_controls(cfg, repo_root=REPO,
                                 defenses_dir=args.defenses_dir or None)


def _pick_cell(cfg: dict, args):
    from sisteer.eval.defense_cell import by_id, stacked_cell

    cells = by_id(_registry(cfg, args))
    if args.cell.startswith("stacked__"):
        base_id = args.cell[len("stacked__"):]
        if base_id not in cells:
            raise SystemExit(
                f"[cell-controls] stacked base cell {base_id!r} unknown; "
                f"known cell ids: {sorted(cells)}")
        return stacked_cell(cells[base_id])
    if args.cell not in cells:
        raise SystemExit(
            f"[cell-controls] unknown --cell {args.cell!r}; known cell ids: "
            f"{sorted(cells)}")
    return cells[args.cell]


def _out_root(args) -> Path:
    root = Path(args.out_root)
    return root if root.is_absolute() else REPO / root


def _check_staged_cell(cell) -> None:
    """The shared artifact-identity gate BEFORE the model loads, through
    the ONE dispatcher (train_budget.assert_object_identity): a FITTED
    cell is held to assert_staged_cell (sidecar names THIS cell, object
    bytes match the sidecar, budget kind valid, COMPLETE manifest coherent
    when present), a DERIVED object to assert_derived_object (bytes <->
    sidecar sha, sidecar <-> the provenance inside the bytes, complete and
    mutually consistent parent identities), and a dir claiming both
    refuses. Serving strictness on the sample manifest — the same bar
    two_si_run's generate applies to the same cells: a null
    sample_manifest_sha is a stated absence, fine to resume over, never
    to serve judged/parity controls from. Baselines (no object) and a
    not-yet-staged object skip (the loader's own error names the path);
    a Class B object is sidecar-exempt when none exists."""
    opath = getattr(cell, "object_path", None)
    if opath is None or not Path(opath).exists():
        return
    from sisteer.eval.train_budget import (assert_object_identity,
                                           object_identity_kind)

    # A stacked cell REUSES its base Class-A object dir, whose sidecar
    # records the BASE id — validate against that, not the stacked
    # spelling.
    cid = (cell.id[len("stacked__"):] if cell.id.startswith("stacked__")
           else cell.id)
    # the strictness knobs belong to the budget-sidecar path alone: a
    # derived object has no budget record, and the HPO-promotion lineage's
    # fit record (fit_provenance.json) carries no sample manifest to demand
    staged_kw = ({} if object_identity_kind(opath) in ("derived", "fit_provenance") else
                 {"require_sidecar": getattr(cell, "class_", None) != "B",
                  "require_sample_manifest_sha": True})
    assert_object_identity(opath, cid, consumer="cell-controls", **staged_kw)


def _capeval(module: str):
    from sisteer.eval.cell_controls import _capeval as lazy

    return lazy(module)


def _retired_gsm8k_refusal(flag_text: str, what: str, suffix: str = "") -> SystemExit:
    """The ONE refusal every retired-GSM8K spelling on the HF path raises
    (PW3-4): ``[cell-controls] <flag> refused: `` + the contracts' pointer at
    the served standard (``contracts.gsm8k_retired_message``, its content
    untouched), plus an optional trailing hint."""
    tc = _capeval("sisteer.capeval.tinycap.contracts")
    return SystemExit(f"[cell-controls] {flag_text} refused: "
                      + tc.gsm8k_retired_message(what) + suffix)


# --------------------------------------------------------------------------- #
# shared cell plumbing (generate + judge)                                      #
# --------------------------------------------------------------------------- #


#: --limit subset selection seed: a limited run takes a SEEDED RANDOM
#: subset of the pinned rows, never a head-slice of the raw order (the
#: xstest limit-20 smoke grabbed 20 safe rows from the head — not random,
#: not stratified). One fixed seed means every condition of a battery
#: selects the SAME items (paired comparison), and the selection is
#: recomputable + recorded in the cell manifest.
SUBSET_SEED = 0


def _seeded_subset(rows: list, limit: int, seed: int = SUBSET_SEED) -> list:
    """Permute the pinned rows with ``seed``, take the first ``limit``,
    then restore pinned order WITHIN the subset (selection is the seeded
    part; processing order stays the runner's own order_seed concern)."""
    if not limit or limit >= len(rows):
        return rows
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[i] for i in sorted(order[:limit])]


#: The s2parity probes this driver runs. ``gsm8k`` left with PW3 (retired;
#: the served standard fills its slot) — a ``gsm8k=N`` width refuses below.
PARITY_PROBES = ("mmlu",)


def _parse_parity_spec(req) -> dict:
    """``--parity-n`` value -> ``{probe: requested width}``: an int (or
    int-looking string) covers EVERY probe at that width; a per-task spec
    ``mmlu=N`` covers only the probes it names (a probe omitted = its
    default). Refuses negatives, unknown probes (``gsm8k`` with the pointer
    at the served standard, PW3), and a probe named twice — a request that
    could mean two widths must never reach a store key."""
    if isinstance(req, int):
        if req < 0:
            raise SystemExit(
                f"[cell-controls] --parity-n {req} is not a width "
                "(0 = the full pinned pools; omit the flag for the "
                "instrument default)")
        return {p: req for p in PARITY_PROBES}
    text = str(req).strip()
    if "=" not in text:
        try:
            return _parse_parity_spec(int(text))
        except ValueError:
            raise SystemExit(
                f"[cell-controls] --parity-n {text!r} is neither a width "
                "nor a per-task spec (mmlu=N)") from None
    out: dict = {}
    for part in text.split(","):
        probe, sep, val = part.strip().partition("=")
        probe = probe.strip()
        if sep and probe == "gsm8k":
            raise _retired_gsm8k_refusal(f"--parity-n {part.strip()!r}",
                                         "the HF path's gsm8k parity width")
        if not sep or probe not in PARITY_PROBES:
            raise SystemExit(
                f"[cell-controls] --parity-n: {part.strip()!r} is not "
                f"task=N with task one of {'/'.join(PARITY_PROBES)}")
        if probe in out:
            raise SystemExit(
                f"[cell-controls] --parity-n names {probe} twice")
        try:
            n = int(val)
        except ValueError:
            raise SystemExit(
                f"[cell-controls] --parity-n: {val!r} is not a width "
                f"for {probe}") from None
        if n < 0:
            raise SystemExit(
                f"[cell-controls] --parity-n: {n} is not a width for "
                f"{probe}")
        out[probe] = n
    return out


def _parity_width(args) -> tuple:
    """``--parity-n`` resolution (controls expansion: the benign-controls
    leg must resolve a 5 pp difference, and ``--limit 0`` cannot widen the
    parity probes because their row sets are narrowed to ``PARITY_N`` = 200
    BEFORE the limit applies; per-task widths 2026-09-01: one probe may
    widen without the other — the lora_r1_all arm ran gsm8k at the full
    pool for precision on the number its finding rests on while mmlu
    stayed at the default, a mixed instrument no scalar request can
    represent. Since PW3 the probe set is ``mmlu`` alone — the gsm8k probe
    is retired — so the per-task form can only name mmlu and always
    collapses to the scalar; the dict form below survives for READING the
    stores it keyed).

    Returns ``(loader_kwargs_by_probe, recorded)``. The first maps each
    probe to its row-loader kwargs (``{}`` = the instrument default;
    ``{"n": None}`` = that probe's full pinned pool; ``{"n": N}`` = N).
    ``recorded`` is what lands in the resolved config, in exactly one
    canonical form per instrument so equivalent spellings can never fork
    sibling stores of the identical measurement:

    * ``None`` — every probe at the s2parity default
      (``parity.PARITY_N`` = 200): the key stays out of the config
      entirely, so every existing default-width store keeps its
      harness_config_hash (the steer_positions precedent). An explicit
      200 — scalar or per-task — normalizes here.
    * an int — ALL probes at one requested width (``0`` = the full
      pinned pool: mmlu validation 1,531).
      A per-task spec naming every probe at one equal width collapses
      to this form, so pre-existing scalar-keyed widened stores keep
      their keys.
    * a dict ``{probe: width}`` — per-probe requested widths, holding
      ONLY the probes at a non-default width (``0`` = that probe's own
      full pool). By the two collapses above this form enters a hash
      ONLY when the widths genuinely differ across probes — exactly the
      instrument the scalar seam could not represent.

    Widths are REQUESTS, not realizations: like the scalar seam (where
    ``--parity-n 1531`` realizes the same rows as ``0`` but keys apart),
    a per-task width equal to a pool size is not folded onto ``0`` —
    the manifest's ``parity_n.realized`` records what actually ran.

    ``--n-gsm8k`` (the deprecated gsm8k-only alias, kept as a flag so an
    old script fails with the reason rather than an argparse error) now
    REFUSES whenever it is set: the gsm8k probe is retired (PW3) and the
    served standard has no HF width to widen. Its 0 default still means
    "unset"."""
    req = getattr(args, "parity_n", None)
    if req is None or req == -1 or req == "-1":
        req = None
    legacy = getattr(args, "n_gsm8k", 0) or 0
    if legacy:
        raise _retired_gsm8k_refusal(f"--n-gsm8k {legacy}", "the HF path's --n-gsm8k width",
                                     suffix=" (--parity-n N widens the mmlu probe)")
    if req is None:
        return {p: {} for p in PARITY_PROBES}, None
    per_task = _parse_parity_spec(req)
    default_n = _capeval("sisteer.capeval.tiers.parity").PARITY_N
    per_task = {p: n for p, n in per_task.items() if n != default_n}
    kwargs = {p: ({"n": None} if per_task[p] == 0 else {"n": per_task[p]})
              if p in per_task else {}
              for p in PARITY_PROBES}
    if not per_task:
        return kwargs, None
    if (len(per_task) == len(PARITY_PROBES)
            and len(set(per_task.values())) == 1):
        return kwargs, next(iter(per_task.values()))
    return kwargs, {p: per_task[p] for p in PARITY_PROBES if p in per_task}


def _utc_now() -> str:
    """Second-resolution UTC ISO-8601 (whitespace-free, so it passes the
    summary's counts-only string rule)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _judge_model(args) -> str:
    """``--judge-model`` resolution: the explicit flag, else the capeval
    judging default. The judge model binds at GENERATE time — it enters the
    resolved config (hence the cell-store key) and every JudgeRequest cache
    key — so ``judge`` refuses a store generated under a different judge
    model rather than silently re-keying half the pipeline."""
    if getattr(args, "judge_model", ""):
        return args.judge_model
    return _capeval("sisteer.capeval.judging").JUDGE_MODEL


#: The CLI's --judge-backend choices; pinned equal to judging.JUDGE_BACKENDS
#: by test (the driver imports capeval lazily, so argparse cannot read the
#: constant at parse time).
JUDGE_BACKENDS = ("anthropic", "openrouter")
DEFAULT_JUDGE_BACKEND = "anthropic"


def _judge_backend(args, judge_model: str) -> str:
    """``--judge-backend`` resolution (default anthropic), VALIDATED against
    the judge model: a grok / vendor-prefixed OpenRouter id requires
    ``openrouter``, a bare claude-* id requires ``anthropic``, and a crossed
    pair refuses naming the flag to pass. Like the judge model it binds at
    GENERATE time — a non-default backend enters the resolved config (hence
    the cell-store key; the default is recorded nowhere so every existing
    store keeps its key) — and ``judge`` cross-checks it against the store
    (a legacy store without the key derives its backend from the recorded
    judge model, which implies it)."""
    backend = getattr(args, "judge_backend", "") or DEFAULT_JUDGE_BACKEND
    if backend not in JUDGE_BACKENDS:
        raise SystemExit(f"[cell-controls] --judge-backend {backend!r} is not "
                         f"one of {list(JUDGE_BACKENDS)}")
    need = _capeval("sisteer.capeval.judging").required_backend(judge_model)
    if need is not None and need != backend:
        raise SystemExit(
            f"[cell-controls] --judge-model {judge_model!r} requires "
            f"--judge-backend {need} (got {backend!r}): a grok / "
            "vendor-prefixed OpenRouter id runs on openrouter, a bare "
            "claude-* id on anthropic — pass the matching backend explicitly")
    return backend


def _recorded_judge_backend(manifest: dict) -> str:
    """The backend a cell store was generated for: the resolved config's
    ``judge_backend`` when recorded (non-default runs since 2026-09-02),
    else the top-level manifest stamp, else IMPLIED by the recorded judge
    model (every pre-stamp grok store was generated under
    ``--judge-model x-ai/grok-4.6`` and is an openrouter store), else the
    default."""
    rc = manifest.get("resolved_config") or {}
    if rc.get("judge_backend"):
        return str(rc["judge_backend"])
    if manifest.get("judge_backend"):
        return str(manifest["judge_backend"])
    implied = _capeval("sisteer.capeval.judging").required_backend(
        rc.get("judge_model") or "")
    return implied or DEFAULT_JUDGE_BACKEND


def _cell_system(cell, cfg: dict):
    """The system text a cell serves, resolved ENGINE-FREE: a prompted
    cell's SI (via cell_serving, which needs no engine when cell.arm is
    None); None for the none baseline and for compiled cells (they serve
    the object, not an SI). The judge phase needs this because the spool
    header binds system_sha (r1 #8) — passing None for a prompted cell
    made ``judge`` refuse the very spool its own generate phase wrote
    (battery run, 2026-08-19)."""
    if getattr(cell, "stacked", False):
        # a stacked cell serves its si_id's system ALONGSIDE the object; the
        # spool header must bind that system_sha (r1 #8) or judge refuses the
        # spool its own generate phase wrote. Resolve engine-free (cell_serving
        # for a stacked cell needs an engine to build the object arm).
        from sisteer.eval.jailbreak import resolve_instruction
        return resolve_instruction(cell.si_id).text
    if cell.arm is not None:
        return None
    from sisteer.eval.cell_controls import cell_serving

    return cell_serving(cell, cfg, engine=None)[1]


#: the config key that selects ONE half of XSTest for the over-refusal control (the
#: sdfbase-llama8b config serves the 250 SAFE prompts). Only "safe" is a legal value —
#: over-refusal is read on the safe prompts, the contrast half is a different measurement.
XSTEST_SUBSET_KEY = "xstest_subset"


def xstest_subset_of(cfg: dict):
    """``cfg["xstest_subset"]`` validated: None (the historical 450-row set) or "safe"."""
    sub = cfg.get(XSTEST_SUBSET_KEY)
    if sub is None:
        return None
    if sub != "safe":
        raise SystemExit(f"[cell-controls] {XSTEST_SUBSET_KEY}: {sub!r} is not a legal selection; "
                         "only 'safe' (the 250 safe prompts) or absent (all 450)")
    return "safe"


def _resolved_config(cell, cfg: dict, system, limit,
                     judge_model: str,
                     parity_n: int | dict | None = None,
                     judge_backend: str = DEFAULT_JUDGE_BACKEND) -> dict:
    """The driver's OWN config identity, hashed into the cell key
    (capeval_judged_run._resolved_config convention; HASHED_CONFIG_FILES
    untouched). Deliberately EXCLUDES the requested-controls subset: all
    five controls share one cell store, so a follow-up run adding a
    control must resume onto the SAME cell, not fork a sibling."""
    from sisteer.eval.cell_controls import controls_gen_config

    manifest = _capeval("sisteer.capeval.manifest")
    resolved = {
        "driver": "cell_controls_run_v1",
        "cell_id": cell.id,
        "model_id": cfg["model"],
        "model_revision": cfg.get("model_revision"),
        "gen": controls_gen_config(cfg),
        "controls": list(RESOLVED_CONTROLS_IDENTITY),
        "order_seed": 0,
        "limit": limit or None,
        "scale": float(cfg.get("scale", 1.0)),
        "system_sha": (manifest.sha256_hex16(system.encode("utf-8"))
                       if system else None),
        "judge_model": judge_model,
        "subset_seed": SUBSET_SEED if limit else None,
    }
    # --steer-positions: recorded ONLY when non-default, so every existing
    # all-positions store keeps its harness_config_hash (a generated-only
    # run keys a DIFFERENT store — it is a different serving).
    steer_positions = _steer_positions(cfg)
    if steer_positions != "all":
        resolved["steer_positions"] = steer_positions
    # --parity-n: recorded ONLY when non-default (0 = the full pinned
    # pools; a scalar for all-probes-equal, a per-probe dict — holding
    # only the non-default probes — when the widths differ), so every
    # existing PARITY_N-width store keeps its harness_config_hash, every
    # scalar-keyed widened store keeps its key, and a widened or mixed
    # run keys a DIFFERENT store — a summary can therefore never mix
    # probe widths while looking identical. Recorded whether or not THIS invocation runs the parity
    # probes: the resolved config deliberately excludes the requested-
    # controls subset (above), so a follow-up run adding a judged control
    # must resume onto the same widened store by passing the same width.
    if parity_n is not None:
        resolved["parity_n"] = parity_n
    # --judge-backend: recorded ONLY when non-default (the steer_positions /
    # parity_n precedent), so every existing store — every anthropic store
    # AND every pre-stamp grok store — keeps its harness_config_hash, while
    # a new openrouter store says so in its own key. The judge model already
    # separates grok from sonnet stores; this makes the backend explicit.
    if judge_backend != DEFAULT_JUDGE_BACKEND:
        resolved["judge_backend"] = judge_backend
    # xstest_subset: recorded ONLY when set (the same precedent), so every historical 450-row
    # store keeps its harness_config_hash and a 250-row SAFE store keys a DIFFERENT store — the
    # two item sets must never resume onto or be mistaken for each other
    xstest_subset = xstest_subset_of(cfg)
    if xstest_subset:
        resolved[XSTEST_SUBSET_KEY] = xstest_subset
    return resolved


def _steer_positions(cfg: dict) -> str:
    return str(cfg.get("steer_positions") or "all")


def _store_for(cell, cfg: dict, out_root: Path, *, ihash: str, chash: str):
    """ONE CellStore per cell for ALL five controls (r1 #7 — task-prefixed
    metrics keep them collision-free); dir token, true cell_id in
    the manifest."""
    from sisteer.eval.cell_controls import store_key_token

    cs = _capeval("sisteer.capeval.store")
    token = store_key_token(cell.id)
    key = cs.cell_key(token, "controls", 0, 0, ihash, chash)
    template = {
        "sweep": "two_si_controls",
        "run_id": f"controls-{token}",
        "model_id": cfg["model"],
        "model_revision": cfg.get("model_revision"),
        "arm_name": token,
        "kind": "cell",
        "coeff": float(cfg.get("scale", 1.0)),
        "intervention_hash": ihash,
        "engine": "hf",
        "trial_id": 0,
        "seed": 0,
        "attempt": 0,
        "harness_config_hash": chash,
        "env_digest": cs.env_digest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    return cs.CellStore(out_root, key, row_constants=template), template


def _judged_specs(cfg: dict, controls: list[str], template: dict,
                  system, limit, judge_model: str) -> dict[str, object]:
    """control -> JudgedSpec for the requested judged controls, rows from
    the pinned loaders (counts asserted there)."""
    from sisteer.eval.cell_controls import controls_gen_config

    judged = _capeval("sisteer.capeval.tiers.judged")
    loaders = {
        "or_bench_hard_1k": judged.or_bench_rows,
        "xstest": judged.xstest_rows,
    }
    xsub = xstest_subset_of(cfg)
    # eval_gen_batch_size (2026-08-28): the judged controls' HF generation
    # width. batch_size rides OUTSIDE controls_gen_config on purpose — it is
    # recorded serving identity, never part of gen_config_sha — so raising
    # it is a throughput knob (a peer measured 5x at width 1 -> 64 on this
    # HF path). Default 8 keeps every config-silent run byte-identical.
    gen = {**controls_gen_config(cfg),
           "batch_size": int(cfg.get("eval_gen_batch_size", 8))}
    specs: dict[str, object] = {}
    for control in controls:
        if control not in JUDGED_TASK_OF:
            continue
        rows = loaders[control]()
        if control == "xstest" and xsub:
            # the SAFE half only when the config says so — selected on the loader's rows BEFORE the seeded
            # subset, the spec, the item hashes and every ladder input downstream
            rows = judged.xstest_subset_rows(rows, xsub)
        rows = _seeded_subset(rows, limit)
        specs[control] = judged.JudgedSpec(
            task=JUDGED_TASK_OF[control], template=dict(template),
            domain_of=dict(DOMAIN_OF), rows=rows, order_seed=0,
            system=system, judge_model=judge_model, gen=dict(gen))
    return specs


def _attach_fluency(payloads: dict, cell_id: str, control: str,
                    texts) -> None:
    """Record the distinct-n-gram fluency guard NEXT TO the judged
    fractions: repetition-collapsed text games LLM-judged metrics —
    degenerate output is neither a refusal nor a successful attack, and
    judges resolve that ambiguity in the defense's favour — so every
    judged control carries the fluency of the exact generations it was
    judged on, and a collapse WARNs loudly at judge time instead of
    surfacing as a good defense number in the aggregate."""
    from sisteer.eval.cell_controls import fluency_payload

    fl = fluency_payload(texts)
    payloads[control]["fluency"] = fl
    if fl["degenerate"]:
        print(f"[cell-controls] WARNING {cell_id}/{control}: DEGENERATE "
              f"generations (distinct3_mean={fl['distinct3_mean']}, "
              f"frac_below_threshold={fl['frac_below_threshold']}, floor "
              f"{fl['threshold']:g}) — repetition-collapsed text; do not "
              "read this control's judged fractions as a defense result")


def _spool_for(judged, store, spec, *, ihash: str, model_revision,
               create: bool):
    """The per-task GenSpool inside the cell store's private dir (the
    header binds it to the cell identity)."""
    header = judged.spool_header(spec, intervention_hash=ihash,
                                 model_revision=model_revision)
    name = spec.task.split("/", 1)[1]  # judged/<name> -> <name>
    path = store.cell_dir / "private" / f"gens_{name}.jsonl"
    return judged.GenSpool(path, header, create=create)


# --------------------------------------------------------------------------- #
# generate — GPU: one model load, §4.1 serving, parity + judged spools         #
# --------------------------------------------------------------------------- #


def cmd_generate(args, cfg: dict) -> None:
    from sisteer.eval.cell_controls import cell_serving, controls_provenance
    from sisteer.models import (configure_fast_sdpa, load_model_and_tokenizer,
                                serving_identity)

    # Set the SDPA backend BEFORE any env_digest is recorded:
    # configure_fast_sdpa runs at model load, but a provenance row read
    # pre-load would record the wrong (pre-flip) flag state now that
    # attention_backend is part of env_digest_fields (folded in when it
    # deviates from store.PINNED_SDPA_STATE). Calling it first here makes
    # the recorded backend match what scoring actually used, independent
    # of the internal load-vs-record order. Idempotent.
    configure_fast_sdpa()
    controls = parse_controls(args.controls)
    cell = _pick_cell(cfg, args)
    _check_staged_cell(cell)  # identity gate BEFORE the model loads
    out_root = _out_root(args)

    # ONE pinned model load (two_si.yaml:51-52), shared by every control.
    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    hf_engine = _capeval("sisteer.capeval.serving.hf_engine")
    engine = hf_engine.HFEngine(cfg["model"],
                                revision=cfg.get("model_revision"),
                                preloaded=(model, tok))
    arm, system = cell_serving(cell, cfg, engine)
    manifest_mod = _capeval("sisteer.capeval.manifest")
    ihash = (arm.intervention_hash if arm is not None
             else manifest_mod.intervention_hash({"kind": "none"}))

    cs = _capeval("sisteer.capeval.store")
    judged = _capeval("sisteer.capeval.tiers.judged")
    judge_model = _judge_model(args)
    judge_backend = _judge_backend(args, judge_model)
    parity_loader_kwargs, parity_recorded = _parity_width(args)
    resolved = _resolved_config(cell, cfg, system, args.limit, judge_model,
                                parity_n=parity_recorded,
                                judge_backend=judge_backend)
    chash = cs.harness_config_hash(resolved)
    store, template = _store_for(cell, cfg, out_root, ihash=ihash,
                                 chash=chash)
    specs = _judged_specs(cfg, controls, template, system, args.limit,
                          judge_model)
    # the standing truncation rule's config-time half: an under-sized cap is
    # refused here, before the store exists (tiers/cap_ladder; --cap-sizing)
    sizing_block = cap_sizing_gate(args, specs)

    parity_rows = None
    if "mmlu" in controls:
        # the mmlu parity probe (gsm8k's is retired — parse_controls refused
        # it above; parity.run skips an empty gsm8k row set); rows are
        # prepared here so a limited run's selection lands in the manifest.
        parity = _capeval("sisteer.capeval.tiers.parity")
        corpora = _capeval("sisteer.capeval.corpora")
        pins = corpora.load_pins()
        parity_rows = {
            "mmlu": _seeded_subset(
                parity.mmlu_parity_rows(
                    pins, **parity_loader_kwargs["mmlu"]),
                args.limit),
        }

    # Train-serve identity, measured from the LIVE tokenizer/model
    # (2026-08-20): chat-template hash + resolved model id/revision.
    # Stamped into the manifest here and into the controls_summary
    # provenance at judge time (judge reads it back from this manifest —
    # the tokenizer is not loaded on the judge box).
    ident = serving_identity(tok, cfg, model=model)
    manifest = {
        "cell_id": cell.id,  # the TRUE id; the store key carries the token
        "intervention_hash": ihash,
        "harness_config_hash": chash,
        "model_revision": cfg.get("model_revision"),
        "engine": "hf",
        "row_constants": template,
        "resolved_config": resolved,
        "serving_identity": ident,
        # who will judge this store — explicit here as well as (model, and
        # a non-default backend) inside resolved_config; judge cross-checks
        "judge_model": judge_model,
        "judge_backend": judge_backend,
        "provenance": controls_provenance(cell, cfg,
                                          intervention_hash=ihash,
                                          serving_identity=ident,
                                          judge_model=judge_model,
                                          judge_backend=judge_backend),
    }
    if parity_recorded is not None and parity_rows is not None:
        # the REALIZED per-probe widths of a non-default --parity-n (the
        # requested value is in resolved_config; 0 = full pools) — so what
        # width actually ran is an artifact lookup, never a pins-file
        # archaeology (each probe caps at its own pool size).
        manifest["parity_n"] = {
            "requested": parity_recorded,
            "realized": {c: len(rows) for c, rows in parity_rows.items()},
        }
    if args.limit:
        # the recorded subset: seed + the exact selected item ids, per
        # control — identical across every condition generated at this
        # limit.
        manifest["subset"] = {
            "seed": SUBSET_SEED,
            "limit": args.limit,
            "item_ids": {
                **({c: [r["item_id"] for r in rows]
                    for c, rows in parity_rows.items()}
                   if parity_rows is not None else {}),
                **{c: [r["item_id"] for r in spec.rows]
                   for c, spec in specs.items()},
            },
        }
    store.validate_resume(manifest)
    store.write_cell_manifest(manifest)
    print(f"[cell-controls] generate cell={cell.id} controls={controls} "
          f"-> {store.cell_dir}")

    if parity_rows is not None:
        spec = parity.ParitySpec(template=dict(template),
                                 domain_of=dict(DOMAIN_OF),
                                 gsm8k_rows=[],  # retired (PW3)
                                 mmlu_rows=parity_rows["mmlu"],
                                 order_seed=0, system=system,
                                 # ParitySpec's own gen default, with the
                                 # eval-side width knob (see _judged_specs)
                                 # — max_new_tokens/do_sample stay pinned.
                                 gen={"max_new_tokens": 512,
                                      "do_sample": False,
                                      "batch_size": int(
                                          cfg.get("eval_gen_batch_size", 8))})
        summary = parity.run(engine, arm, spec, store)
        print(f"[cell-controls] parity done: "
              f"{json.dumps(summary, default=str)}")

    ladder_statuses: dict = {}
    for control, spec in specs.items():
        spool = _spool_for(judged, store, spec, ihash=ihash,
                           model_revision=cfg.get("model_revision"),
                           create=True)
        out = judged.generate(engine, arm, spec, spool)
        # the generated items, told to the store's registry recorder (one null-valued item_result each)
        store.note_generated(spec.task, sorted(spool.entries()), str(spool.path))
        print(f"[cell-controls] {control}: {out['n_new']} new / "
              f"{out['n_total']} spooled generations")


# --------------------------------------------------------------------------- #
# judge — grading host: coordination post, then cache-first judging + summary #
# --------------------------------------------------------------------------- #
# RESTORED 2026-09-12: these four helpers left this file when the f14d0083 merge's conflict here was resolved by
# taking main's deletion of the alpaca / cap-ladder block. main never had them; THIS tree's
# served_capability_run.py (the controls driver) calls cap_sizing_gate / measure_cap_ladder, and this file's own
# gate calls cap_sizing_gate. Verbatim from 9635d8a4 (the sha this tree runs from); tests/test_cap_ladder.py pins them.
# --------------------------------------------------------------------------- #
# the standing truncation rule (tiers/cap_ladder) — shared by both drivers     #
# (ported from fst-pair @ 75000b5c; this tree's rung mechanism is the sampled  #
# run's --controls-cap/--escalate-from-root path, so the carry-over bundle     #
# planner did not travel — only the CHECK, the sizing gate, the record and     #
# the transport guard)                                                         #
# --------------------------------------------------------------------------- #


def _judged_cap(specs: dict) -> int:
    """The judged controls' cap of THIS run — read off the specs that
    ``controls_gen_config`` resolved (never a literal)."""
    caps = {int(s.gen["max_new_tokens"]) for s in specs.values()}
    if len(caps) != 1:
        raise SystemExit(f"[cell-controls] the judged specs disagree on the cap: {sorted(caps)}")
    return caps.pop()


def cap_sizing_gate(args, specs: dict) -> dict | None:
    """``--cap-sizing``: the config-time half of the rule. The sizing file
    (``scripts/cap_ladder.py size``) names the smallest rung whose measured
    at-cap share is at or below the threshold per task; the run's cap must be
    at or above it for every judged control it generates, else the run is
    refused BEFORE any store is touched. None when the flag is not given."""
    path = getattr(args, "cap_sizing", "") or ""
    if not path or not specs:
        return None
    cl = _capeval("sisteer.capeval.tiers.cap_ladder")
    try:
        block = cl.check_cap_against_sizing(_judged_cap(specs), cl.load_sizing(Path(path)),
                                            controls=list(specs))
    except cl.CapLadderError as e:
        raise SystemExit(f"[cap-ladder] --cap-sizing: {e}") from None
    block["path"] = str(path)
    print(f"[cap-ladder] cap sizing OK: cap {block['cap']} >= the measured "
          f"recommendation for {sorted(block['controls'])} ({path})")
    return block


def measure_cap_ladder(spool, spec, *, tok, control: str, cell_id: str,
                       prefix: str, source_caps=None, other_draws=None,
                       regenerated=(), source_texts=None,
                       source_caps_of_regenerated=None, climbed: bool = False,
                       hop_timeouts_s=None):
    """Measure one judged control's COMPLETE spool at the run's cap
    (``spec.gen["max_new_tokens"]``), write the per-item sidecar beside the
    spool and print the one loud line. Returns the ``RungStatus``.
    ``hop_timeouts_s`` is the transport guard's input (``{}`` for the
    in-process HF engine; the served driver passes its two hops; ``None``
    refuses any escalation as undeterminable)."""
    cl = _capeval("sisteer.capeval.tiers.cap_ladder")
    cap = int(spec.gen["max_new_tokens"])
    entries = spool.entries()
    status, records = cl.measure_rung(
        entries, tok, cap, source_caps=source_caps, other_draws=other_draws,
        regenerated=regenerated, source_texts=source_texts,
        source_caps_of_regenerated=source_caps_of_regenerated, climbed=climbed,
        hop_timeouts_s=hop_timeouts_s)
    cl.write_sidecar(cl.sidecar_path(spool.path), records,
                     header={"cell_id": cell_id, "control": control, "task": spec.task,
                             "cap": cap, "items_sha": spool.header.get("items_sha"),
                             "intervention_hash": spool.header.get("intervention_hash")})
    print(cl.status_line(prefix, control, status), flush=True)
    return status


def write_cap_ladder(store, manifest: dict, *, cell_id: str, engine: str, cap: int,
                     statuses: dict, sizing=None, extra=None) -> Path:
    """The per-cell record (``cap_ladder.json``) and its manifest block — an
    OUTCOME block like the served driver's ``quality``, never identity, so
    the manifest re-write only adds it (validate_resume re-checks the
    unchanged identity fields)."""
    cl = _capeval("sisteer.capeval.tiers.cap_ladder")
    rec = cl.cell_record(cell_id=cell_id, engine=engine, cap=cap, statuses=statuses,
                         store_dir=store.cell_dir, extra=extra)
    path = cl.write_cell_record(store.cell_dir, rec)
    manifest[cl.MANIFEST_KEY] = rec
    if sizing is not None:
        manifest[cl.MANIFEST_SIZING_KEY] = sizing
    store.write_cell_manifest(manifest)
    return path

# --------------------------------------------------------------------------- #


def _find_cell_store(out_root: Path, cell):
    from sisteer.eval.cell_controls import store_key_token

    cs = _capeval("sisteer.capeval.store")
    cells_root = out_root / "cells"
    token = store_key_token(cell.id)
    matches = (sorted(cells_root.glob(f"{token}__controls__*"))
               if cells_root.is_dir() else [])
    if not matches:
        raise SystemExit(
            f"[cell-controls] no generated cell store under {cells_root} for "
            f"{cell.id!r} (token {token}) — run `generate` first")
    if len(matches) > 1:
        raise SystemExit(
            f"[cell-controls] multiple cell stores for {cell.id!r} under "
            f"{cells_root}: {[m.name for m in matches]} — stale attempts; "
            "remove the dead ones deliberately")
    store = cs.CellStore(out_root, matches[0].name)
    manifest = json.loads(store.manifest_path.read_text())
    if manifest.get("cell_id") != cell.id:
        raise SystemExit(
            f"[cell-controls] manifest at {matches[0]} names cell "
            f"{manifest.get('cell_id')!r}, expected {cell.id!r}")
    store.row_constants = dict(manifest["row_constants"])
    return store, manifest


def _parity_payloads(store_dir: Path, controls: list[str]) -> dict[str, dict]:
    """The mmlu slot payload straight from the store's per-item rows
    (frozen row schema; the aggregate row is excluded). No judging:
    the probe is locally scored. (gsm8k's payload left with PW3: the served
    standard writes the slot through ``tinycap.completion``.)"""
    rows_path = Path(store_dir) / "rows.jsonl"
    if not rows_path.exists():
        raise SystemExit(
            f"[cell-controls] no rows at {rows_path} — run `generate` for "
            "this cell first")
    acc: dict[str, dict[str, float]] = {}
    failures: dict[str, int] = {}
    for line in rows_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task, item, metric = (row.get("task"), row.get("item_id"),
                              row.get("metric"))
        if item == "__aggregate__":
            continue
        if metric == "acc":
            # one value per item: a conflicting repeat means two writers
            # raced on the store (the judged tier's _completed_items makes
            # the same refusal) — never resolve it by file order
            have = acc.setdefault(task, {})
            if item in have and have[item] != float(row["value"]):
                raise SystemExit(
                    f"[cell-controls] {rows_path}: {task} item {item!r} has "
                    f"two DIFFERENT acc rows ({have[item]!r}, "
                    f"{float(row['value'])!r}) — a raced store; refusing to "
                    "summarize it. Inspect rows.jsonl and repair deliberately")
            have[item] = float(row["value"])
        elif metric == "extraction_failed":
            failures[task] = failures.get(task, 0) + int(float(row["value"]))
    out: dict[str, dict] = {}
    tasks = {"mmlu": "s2parity/mmlu_letter"}
    for control, task in tasks.items():
        if control not in controls:
            continue
        items = acc.get(task)
        if not items:
            raise SystemExit(
                f"[cell-controls] no per-item {task} rows in {rows_path} — "
                f"generate did not run {control} for this cell")
        out[control] = {"acc": sum(items.values()) / len(items), "n": len(items)}
    return out


def cmd_judge(args, cfg: dict) -> None:
    from sisteer.eval.cell_controls import (
        controls_provenance,
        write_controls_summary,
    )

    controls = parse_controls(args.controls)
    cell = _pick_cell(cfg, args)
    out_root = _out_root(args)
    store, manifest = _find_cell_store(out_root, cell)
    ihash = manifest["intervention_hash"]
    judge_model = _judge_model(args)
    recorded = (manifest.get("resolved_config") or {}).get("judge_model")
    if recorded and recorded != judge_model:
        raise SystemExit(
            f"[cell-controls] this cell store was generated under "
            f"judge_model={recorded!r} but judge was invoked with "
            f"{judge_model!r} — the judge model binds at generate time (it "
            "keys the store); re-run generate, or pass --judge-model "
            f"{recorded}")
    judge_backend = _judge_backend(args, judge_model)
    recorded_backend = _recorded_judge_backend(manifest)
    if recorded_backend != judge_backend:
        raise SystemExit(
            f"[cell-controls] this cell store was generated for "
            f"judge_backend={recorded_backend!r} but judge was invoked with "
            f"{judge_backend!r} — the backend binds at generate time with the "
            "judge model (a grok store is an openrouter store); pass "
            f"--judge-backend {recorded_backend}")
    recorded_sp = (manifest.get("resolved_config") or {}).get(
        "steer_positions", "all")
    requested_sp = _steer_positions(cfg)
    if recorded_sp != requested_sp:
        raise SystemExit(
            f"[cell-controls] this cell store was generated under "
            f"steer_positions={recorded_sp!r} but judge was invoked with "
            f"{requested_sp!r} — name the serving you are judging (pass "
            f"--steer-positions {recorded_sp}) so a generated-only store "
            "can never be summarized as all-positions serving")

    payloads: dict[str, dict] = {}
    rubric_shas: dict[str, str] = {}
    if "mmlu" in controls:
        payloads.update(_parity_payloads(store.cell_dir, controls))

    judged_controls = [c for c in controls if c in JUDGED_TASK_OF]
    if judged_controls:
        judged = _capeval("sisteer.capeval.tiers.judged")
        judging = _capeval("sisteer.capeval.judging")
        # The judge phase never re-renders prompts, but the spool header
        # binds the system identity (system_sha, r1 #8) — so the spec must
        # carry the SAME system the generate phase served, resolved
        # engine-free (a prompted cell's SI; None otherwise). Passing None
        # unconditionally made judge refuse the prompted cell's own spool.
        specs = _judged_specs(cfg, judged_controls, store.row_constants,
                              _cell_system(cell, cfg), args.limit,
                              judge_model)
        n_unique = sum(len(specs[c].rows) for c in judged_controls)
        # Coordination post BEFORE any wide judge phase (>5 concurrency) —
        # printed for the operator to post when the site names a coordination
        # channel (SISTEER_COORDINATION_CHANNEL); the driver never posts it.
        from sisteer import coordination
        if coordination.channel():
            print(judging.coordination_post(
                n_unique_requests=n_unique, concurrency=args.concurrency,
                model=judge_model, backend=judge_backend,
                context=f"two-SI cell controls ({cell.id})"))
        judge = judging.ClaudeJudge(concurrency=args.concurrency,
                                    model=judge_model, backend=judge_backend)
        for control in judged_controls:
            spec = specs[control]
            # per-control live/cache split for the payload (below): the
            # judge's counters are cumulative across the cell's controls
            calls_before = int(getattr(judge, "n_api_calls", 0) or 0)
            hits_before = int(getattr(judge, "n_cache_hits", 0) or 0)
            spool = _spool_for(judged, store, spec, ihash=ihash,
                               model_revision=cfg.get("model_revision"),
                               create=False)
            if control == "or_bench_hard_1k":
                result = judged.judge_or_bench(spool, judge, spec, store)
            elif control == "xstest":
                result = judged.judge_xstest(spool, judge, spec, store)
            else:
                raise SystemExit(f"[cell-controls] {control!r} is not a judged control")
            payloads[control] = slot_payload(control, result)
            rubric_shas[control] = payloads[control]["rubric_sha"]
            # WHEN this control's verdicts were filed, and how many came
            # live vs from the cache (a judge's decline set drifts over
            # time — the 2026-09-02 sonnet-5 probe — so the numbers are only
            # interpretable with a time on them; per-verdict times live in
            # the judge cache's created_at, this is the control-level stamp).
            # Output-only: no hash, no store key, no provenance identity.
            payloads[control]["judged_at"] = _utc_now()
            payloads[control]["judge_api_calls"] = (
                int(getattr(judge, "n_api_calls", 0) or 0) - calls_before)
            payloads[control]["judge_cache_hits"] = (
                int(getattr(judge, "n_cache_hits", 0) or 0) - hits_before)
            # Fluency guard over the EXACT generations just judged.
            _attach_fluency(payloads, cell.id, control,
                            list(spool.assert_complete(
                                [r["item_id"] for r in spec.rows]).values()))
        print(f"[cell-controls] judge cache: "
              f"hits={getattr(judge, 'n_cache_hits', None)} "
              f"api_calls={getattr(judge, 'n_api_calls', None)}")

    ident = manifest.get("serving_identity")
    if not ident:
        raise SystemExit(
            "[cell-controls] this cell store's manifest carries no "
            "serving_identity (chat-template hash + resolved model id/"
            "revision) — it was generated by pre-stamp code. Re-run "
            "generate; a summary without the serving stamp is exactly the "
            "un-diagnosable artifact the stamp exists to prevent (the "
            "user, 2026-08-20)")
    provenance = controls_provenance(cell, cfg, rubric_shas=rubric_shas,
                                     intervention_hash=ihash,
                                     serving_identity=ident,
                                     judge_model=judge_model,
                                     judge_backend=judge_backend)
    path = write_controls_summary(out_root, cell, payloads, provenance)
    print(f"[cell-controls] wrote {path} (controls: {sorted(payloads)})")


# --------------------------------------------------------------------------- #
# emit — capeval-free: the cell's controls block in the exact slot format      #
# --------------------------------------------------------------------------- #


def cmd_emit(args, cfg: dict) -> None:
    from sisteer.eval.cell_controls import load_controls_summaries

    cell = _pick_cell(cfg, args)
    summaries = load_controls_summaries(_out_root(args), [cell], cfg)
    filled = summaries.get(cell.id, {})
    block = {c: filled.get(c, _CONTROL_PENDING) for c in cfg["controls"]}
    print(json.dumps({"cell_id": cell.id, "controls": block},
                     indent=1, sort_keys=True))


# --------------------------------------------------------------------------- #
# cells — the sbatch array decode surface                                      #
# --------------------------------------------------------------------------- #


def cmd_cells(args, cfg: dict) -> None:
    registry = _registry(cfg, args)
    if args.task_id is not None:
        from sisteer.eval.defense_cell import cell_for_array_index

        cell, shard = cell_for_array_index(registry, args.task_id,
                                           args.nshards)
        print(f"{cell.id} {shard}")
        return
    from sisteer.eval.defense_cell import matrix_counts

    for i, c in enumerate(registry):
        print(f"{i:3d}  {c.id}  (class {c.class_})")
    counts = matrix_counts(registry)
    print(f"# {counts['total']} cells; --array=0-({counts['total']}*NSHARDS - 1)")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _common(p) -> None:
    p.add_argument("--config", default="configs/two_si.yaml")
    p.add_argument("--out-root", default="out/two_si/controls")
    p.add_argument("--defenses-dir", default="",
                   help="explicit compiled-objects dir (wins over the config "
                        "value, the TT_TWO_SI_DEFENSES_DIR env var, and the "
                        "relative default)")
    p.add_argument("--cap-sizing", dest="cap_sizing", default="",
                   help="the standing truncation rule's config-time gate "
                        "(tiers/cap_ladder): a cap_sizing/1 JSON written by "
                        "`scripts/cap_ladder.py size` from the measured length "
                        "distributions; `generate` refuses, before any store is "
                        "touched, a cap below the recommendation for any judged "
                        "control it generates, and records the check in the "
                        "manifest (cap_sizing). Every generate pass measures its "
                        "spools under the rule regardless (cap_ladder.json).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    judge_model_help = (
        "judge model id (default: the capeval judging default). Binds at "
        "GENERATE time — it enters the resolved config, the cell-store key, "
        "and every judge cache key — so judge refuses a mismatched store.")

    judge_backend_help = (
        "which API the judge model is reached through: 'anthropic' (the "
        "default: the Anthropic Low-Prio org, claude-* models) or "
        "'openrouter' (OpenRouter's Anthropic-compatible skin on the "
        "OpenRouter key from the vault — the grok fallback judge, "
        "x-ai/grok-4.6). Validated against --judge-model: a grok / "
        "vendor-prefixed id requires openrouter, a bare claude-* id requires "
        "anthropic; a crossed pair refuses. Binds at generate time like the "
        "judge model (a non-default backend enters the resolved config and "
        "the cell-store key; the default is recorded nowhere, so existing "
        "stores keep their keys) and judge cross-checks it. OpenRouter is "
        "NOT the shared Anthropic pool — the printed coordination line says "
        "so, and no coordination-channel post is owed.")

    coeff_help = (
        "override the config's pre-registered `scale` (the serving "
        "coefficient / alpha) for THIS run — the dose-response seam. It "
        "reaches engine.arm's coeff, the AblationSpec, the resolved config "
        "(hence the cell-store key), and the store's coeff column. The "
        "equivalence gate reads the same override via SISTEER_EQUIV_SCALE "
        "(the sbatch COEFF seam sets both). Empty = the config value.")

    steer_positions_help = (
        "serving position policy for the compiled object: 'all' (the "
        "default — every position, prefill AND decode) or 'response' (the "
        "POSITION MASK: the assistant-header positions at the prompt tail "
        "plus every generated position are steered, so the prefill forward "
        "that emits token 1 is steered — the same mechanism the JB legs "
        "serve). 'generated' — the retired engine-phase gate, which served "
        "token 1 from an UNSTEERED prefill — is refused with a pointer at "
        "'response' rather than aliased, so the literal can never mean two "
        "different interventions across runs. Binds at generate time "
        "(resolved config -> cell-store key); judge refuses a mismatched "
        "store.")

    parity_n_help = (
        "mmlu sample width (the s2parity PARITY_N=200 instrument default "
        "when omitted; 0 = the FULL pinned pool — mmlu validation 1,531). "
        "--limit cannot widen it: the row set is narrowed to PARITY_N "
        "before the limit applies. The seed-0 selection is a PREFIX, so "
        "every narrower width is a strict subset of every wider one and "
        "default-width numbers stay interpretable as a subsample. A "
        "non-default width enters the resolved config (hence the "
        "cell-store key), so widths can never silently mix in one store "
        "or summary. Also accepts the per-task spelling mmlu=N (an "
        "all-probes-equal spec normalizes to the scalar form, so "
        "equivalent spellings key the same store). gsm8k=N refuses: the "
        "gsm8k probe is retired (PW3) — the standard GSM8K runs on the "
        "served path (served_capability_run.py --controls gsm8k).")

    g = sub.add_parser("generate")
    _common(g)
    g.add_argument("--cell", required=True)
    g.add_argument("--controls", default=",".join(WIRED_CONTROLS))
    g.add_argument("--limit", type=int, default=0,
                   help="per-control item cap (0 = full). A limited run "
                        "takes a SEEDED random subset of the pinned rows "
                        "(same items for every condition; seed + ids "
                        "recorded in the cell manifest), never a head-slice")
    g.add_argument("--parity-n", default=None, help=parity_n_help)
    g.add_argument("--judge-model", default="", help=judge_model_help)
    g.add_argument("--judge-backend", default=DEFAULT_JUDGE_BACKEND,
                   choices=JUDGE_BACKENDS, help=judge_backend_help)
    g.add_argument("--coeff", default="", help=coeff_help)
    g.add_argument("--n-gsm8k", type=int, default=0,
                   help="RETIRED (PW3): the gsm8k probe no longer runs here; "
                        "any non-zero value refuses with the pointer at the "
                        "served standard. Kept as a flag so old scripts fail "
                        "with the reason, not an argparse error")
    g.add_argument("--steer-positions", default="all",
                   choices=("all", "response", "generated"),
                   help=steer_positions_help)

    j = sub.add_parser("judge")
    _common(j)
    j.add_argument("--cell", required=True)
    j.add_argument("--controls", default=",".join(WIRED_CONTROLS))
    j.add_argument("--limit", type=int, default=0,
                   help="must match the generate-phase --limit (smoke runs)")
    j.add_argument("--concurrency", type=int, default=200)
    j.add_argument("--judge-model", default="", help=judge_model_help)
    j.add_argument("--judge-backend", default=DEFAULT_JUDGE_BACKEND,
                   choices=JUDGE_BACKENDS, help=judge_backend_help)
    j.add_argument("--coeff", default="", help=coeff_help)
    j.add_argument("--steer-positions", default="all",
                   choices=("all", "response", "generated"),
                   help=steer_positions_help)

    e = sub.add_parser("emit")
    _common(e)
    e.add_argument("--cell", required=True)

    c = sub.add_parser("cells")
    _common(c)
    c.add_argument("--task-id", type=int, default=None)
    c.add_argument("--nshards", type=int, default=1)

    args = ap.parse_args()
    cfg = jailbreak_run._load_cfg(args.config)
    if getattr(args, "coeff", ""):
        cfg["scale"] = float(args.coeff)
    if getattr(args, "steer_positions", "all") == "generated":
        # Kept in choices ONLY so the refusal can explain itself: the
        # engine-phase gate serves token 1 from an unsteered prefill —
        # for a refusal defense the decisive token — and was retired for
        # defense serving (2026-08-21). Refused, not aliased: the literal
        # has meant the gate in every store keyed before 2026-08-28.
        raise SystemExit(
            "[cell-controls] --steer-positions generated is refused: it "
            "selects the retired engine-phase gate (prefill — and token 1 "
            "— served by the plain base model). Use --steer-positions "
            "response: the position mask steers the assistant header at "
            "the prompt tail plus every generated position, so the forward "
            "that emits token 1 is steered.")
    if getattr(args, "steer_positions", "all") != "all":
        cfg["steer_positions"] = args.steer_positions
    {"generate": cmd_generate, "judge": cmd_judge, "emit": cmd_emit,
     "cells": cmd_cells}[args.cmd](args, cfg)


if __name__ == "__main__":
    # A credential rejection that reaches main -- the shared judge's / grader's typed error -- ends the process with
    # the typed text as its terminal line, never a traceback
    from sisteer import vault  # noqa: E402

    with vault.credential_exit(prefix="[cell-controls] "):
        main()
