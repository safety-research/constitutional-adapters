"""capeval judged-controls driver.

Two phases per task, split across machines (the generate-on-GPU /
judge-on-grading-host split):

  # GPU node (sbatch scripts/sbatch/capeval_judged.sbatch):
  uv run python scripts/capeval_judged_run.py generate \
      --task or_bench|xstest --sweep judged_base --base [--limit 20]
  uv run python scripts/capeval_judged_run.py generate \
      --task xstest --sweep judged_cell \
      --arm-artifact out/.../object.pt --arm-kind sisteer_object \
      --arm-name safety-vector --layer 14 --coeff 8.0 \
      [--system-file path/to/si.txt]

  # The grading host (the Low-Prio key + the shared judge cache live here):
  uv run python scripts/capeval_judged_run.py judge \
      --task or_bench --sweep judged_base [--concurrency 200]

``generate`` loads the pinned model ONCE (configs/capeval/models/
llama31_8b_it.yaml), resolves the arm through ``HFEngine.arm`` (or
``--base`` → arm=None), writes the cell manifest (row constants +
spool/base-cache provenance archived for the judge phase), and streams
generations into the private GenSpool. ``judge`` prints the
coordination post (with measured, retry-adjusted token bounds; when the
site names a coordination channel) BEFORE any judging, refuses an incomplete spool, files
per-item + aggregate rows, writes results.json (the cell-done marker),
prints ``cache_hits/api_calls``, and logs aggregates + figures to W&B
(the project ``wandb.init`` names below) and ``$FIGS_DIR``.

The resolved config is hashed into ``task_version`` and the cell key's
config hash — ``runtime.HASHED_CONFIG_FILES`` is NOT edited: the judged
tier sits outside the v0 verdict pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sisteer import coordination  # noqa: E402
from sisteer.capeval import judging  # noqa: E402
from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval import store as cs  # noqa: E402
from sisteer.capeval.manifest import sha256_hex16  # noqa: E402
from sisteer.capeval.tiers import judged  # noqa: E402

MODEL_CFG = REPO / "configs" / "capeval" / "models" / "llama31_8b_it.yaml"
DEFAULT_OUT_ROOT = REPO / "out" / "capeval_judged"

#: task cli-name -> (task id, cell-key token, loader, rubric)
TASKS: Dict[str, Tuple[str, str, Any, Any]] = {
    "or_bench": (judged.TASK_ORBENCH, "judgedorb", judged.or_bench_rows,
                 judging.ORBENCH_RUBRIC_V1),
    "xstest": (judged.TASK_XSTEST, "judgedxst", judged.xstest_rows,
               judging.XSTEST_RUBRIC_V1),
}
#: judged rows live outside the v0 domain registry; the emitter
#: still requires an explicit mapping — never an unmapped emission.
DOMAIN_OF = {judged.TASK_ORBENCH: "overrefusal_judged",
             judged.TASK_XSTEST: "refusal_calibration_judged"}
#: Per-request wall clock on the judge phase's count-tokens client, ALWAYS
#: set — the judge module's own bound (:data:`judging.REQUEST_TIMEOUT_S`), so
#: there is one number. This was the one client the driver built outside
#: ``capeval.judging`` (``max_retries=1`` and nothing else), so it kept the
#: SDK's 600 s default after the judge clients were bounded. A count-tokens
#: call is cheaper than a judge call (no generation), and the judge call
#: answers in ~1.8 s; MEASURED 2026-09-11, two sockets ESTAB with empty
#: send/recv queues stalled a 5,953-item run behind that default and the
#: run's watchdog read the silence as pool degradation. A timeout here
#: degrades to the offline byte bound (measure_token_bounds catches it)
#: instead of stalling the run before the coordination post is printed.
COUNT_TOKENS_TIMEOUT_S = judging.REQUEST_TIMEOUT_S


def _count_tokens_client(org: str = "lowprio", *, max_retries: int = 1,
                         request_timeout_s: float = COUNT_TOKENS_TIMEOUT_S,
                         base_url: Optional[str] = None):
    """The anthropic client for the judge phase's count-tokens bounds
    (:func:`judging.measure_token_bounds`). Retries stay at 1: the bound is
    a courtesy estimate for the coordination post, not a retry loop. The
    timeout is plumbed the way ``max_retries`` is — never a bare number at
    the call site.

    ``base_url`` resolves exactly as the judge's own client does
    (:func:`judging.judge_base_url`: :data:`judging.JUDGE_BASE_URL_ENV` else
    :data:`judging.DEFAULT_JUDGE_BASE_URL` under op / env; the API host pinned
    under ``SISTEER_SECRETS_MODE=wif``, an override naming another host refused
    by name) and is ALWAYS passed, so the SDK never reads ``ANTHROPIC_BASE_URL``
    from the shell: some environments point that at a local gateway, and a
    client that omits ``base_url`` tunnels there while the judge client one call
    later reaches the API. Two clients the same phase builds must land on the
    same host. An explicit ``base_url=`` is the in-process test seam and wins in
    every mode; the driver passes none.

    The client is the vault's for ``org`` (``sisteer.vault.anthropic_client``:
    the org's own key under op / env, a federated keyless client under
    ``SISTEER_SECRETS_MODE=wif``), exactly as the judge's own is."""
    from sisteer import vault

    return vault.anthropic_client(org,
                                  base_url=base_url or judging.judge_base_url(),
                                  max_retries=max_retries,
                                  timeout=request_timeout_s)


def _die(msg: str) -> "NoReturn":  # noqa: F821
    raise SystemExit(f"[capeval-judged] FATAL: {msg}")


def _model_pin() -> Tuple[str, str]:
    import yaml

    cfg = yaml.safe_load(MODEL_CFG.read_text())
    return cfg["model"], cfg["model_revision"]


def _arm_token(name: str) -> str:
    """cell_key refuses '__' and '/' (store.py) — map them injectively;
    the true name is archived in the manifest."""
    return name.replace("__", "--").replace("/", "-")


def _load_system(args: argparse.Namespace) -> Optional[str]:
    if args.system_file:
        text = Path(args.system_file).read_text(encoding="utf-8").strip()
        if not text:
            _die(f"--system-file {args.system_file} is empty")
        return text
    return None


def _rows_for(args: argparse.Namespace) -> List[dict]:
    _task, _token, loader, _rubric = TASKS[args.task]
    rows = loader()  # full pinned load asserts the counts
    if args.limit is not None:
        if args.limit <= 0:
            _die("--limit must be positive")
        rows = rows[:args.limit]
    return rows


def _spec_for(args: argparse.Namespace, rows: List[dict],
              template: Dict[str, Any], task_version: str,
              system: Optional[str]) -> judged.JudgedSpec:
    task, _token, _loader, _rubric = TASKS[args.task]
    return judged.JudgedSpec(
        task=task, template=template, domain_of=dict(DOMAIN_OF), rows=rows,
        task_version=task_version, order_seed=args.order_seed, system=system,
        judge_model=args.judge_model,
        gen={"max_new_tokens": args.max_new_tokens, "do_sample": False,
             "batch_size": args.batch_size})


def _resolved_config(args: argparse.Namespace, *, model_id: str,
                     model_revision: str, system: Optional[str],
                     rubric) -> Dict[str, Any]:
    """The driver's OWN config identity — hashed into task_version and
    the cell key's config hash (HASHED_CONFIG_FILES untouched)."""
    return {
        "driver": "capeval_judged_run_v1",
        "task": TASKS[args.task][0],
        "model_id": model_id,
        "model_revision": model_revision,
        "gen": {"max_new_tokens": args.max_new_tokens, "do_sample": False,
                "batch_size": args.batch_size},
        "order_seed": args.order_seed,
        "limit": args.limit,
        "system_sha": (sha256_hex16(system.encode("utf-8"))
                       if system else None),
        "rubric_sha": rubric.sha,
        "protocol_sha": judging.protocol_sha(rubric),
        # per-run judge model (default
        # claude-sonnet-5); baked into the config hash + cell key + the
        # canonical JudgeRequest keys, so two judge models can never
        # cross-contaminate cached verdicts or share a cell
        "judge_model": args.judge_model,
    }


def _template(args: argparse.Namespace, *, model_id: str,
              model_revision: str, arm_name: str, kind: str,
              coeff: Optional[float], ihash: str,
              chash: str) -> Dict[str, Any]:
    return {
        "sweep": args.sweep,
        "run_id": f"judged-{args.sweep}-{args.task}",
        "model_id": model_id,
        "model_revision": model_revision,
        "arm_name": _arm_token(arm_name),
        "kind": kind,
        "coeff": coeff,
        "intervention_hash": ihash,
        "engine": "hf",
        "trial_id": 0,
        "seed": args.order_seed,
        "attempt": 0,
        "harness_config_hash": chash,
        "env_digest": cs.env_digest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


# --- generate (GPU) -------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    # Pin the SDPA backend BEFORE any env_digest is recorded (the
    # cell_controls_run.cmd_generate discipline): engine.load() happens to
    # pin first today, but the recorded backend must match what scoring
    # used independent of the internal load-vs-record order. Idempotent.
    from sisteer.models import configure_fast_sdpa
    configure_fast_sdpa()
    task, token, _loader, rubric = TASKS[args.task]
    model_id, model_revision = _model_pin()
    system = _load_system(args)
    rows = _rows_for(args)

    from sisteer.capeval.serving.hf_engine import HFEngine

    engine = HFEngine(model_id, revision=model_revision)
    engine.load()

    if args.base:
        arm, kind, coeff = None, "none", None
        arm_name = "base"
        ihash = cm.intervention_hash({"kind": "none"})
    else:
        for req_arg in ("arm_artifact", "arm_kind", "arm_name"):
            if getattr(args, req_arg) in (None, ""):
                _die(f"--{req_arg.replace('_', '-')} is required without "
                     "--base")
        kind, coeff, arm_name = args.arm_kind, args.coeff, args.arm_name
        ab = cm.AblationSpec(name=_arm_token(arm_name), kind=kind,
                             artifact=args.arm_artifact, layer=args.layer,
                             coeffs=[coeff])
        arm = engine.arm(ab, coeff)  # loud kind/layer/sidecar validation
        ihash = arm.intervention_hash

    resolved_cfg = _resolved_config(args, model_id=model_id,
                                    model_revision=model_revision,
                                    system=system, rubric=rubric)
    chash = cs.harness_config_hash(resolved_cfg)
    task_version = f"{judged.TASK_VERSION}+{chash[:8]}"
    template = _template(args, model_id=model_id,
                         model_revision=model_revision, arm_name=arm_name,
                         kind=kind, coeff=coeff, ihash=ihash, chash=chash)
    spec = _spec_for(args, rows, template, task_version, system)

    out_root = Path(args.out_root) / args.sweep
    key = cs.cell_key(_arm_token(arm_name), token, 0, 0, ihash, chash)
    store = cs.CellStore(out_root, key, row_constants=template)

    header = judged.spool_header(spec, intervention_hash=ihash,
                                 model_revision=model_revision)
    spool_path = store.cell_dir / "private" / "gens.jsonl"

    manifest: Dict[str, Any] = {
        "intervention_hash": ihash,
        "harness_config_hash": chash,
        "model_revision": model_revision,
        "engine": "hf",
        "row_constants": template,
        "task": task,
        "task_version": task_version,
        "true_arm_name": arm_name,
        "resolved_config": resolved_cfg,
        "rubric": {"name": rubric.name, "version": rubric.version,
                   "sha": rubric.sha},
        # paths are RELATIVE to the sweep out_root: generate runs on the
        # cluster, judge on the grading host after an rsync of the sweep dir —
        # an absolute cluster path would never resolve there
        "spool": {"path": str(spool_path.relative_to(out_root)),
                  "header": header},
        "n_items": len(rows),
        "items_sha": judged.rows_content_sha(rows),
    }

    store.validate_resume(manifest)
    store.write_cell_manifest(manifest)
    spool = judged.GenSpool(spool_path, header)
    t0 = time.time()
    out = judged.generate(engine, arm, spec, spool)
    print(f"[capeval-judged] generate {task}: {out['n_new']} new / "
          f"{out['n_total']} total in {time.time() - t0:.0f}s -> "
          f"{spool.path}")

    print(f"[capeval-judged] next (the grading host): uv run python "
          f"scripts/capeval_judged_run.py judge --task {args.task} "
          f"--sweep {args.sweep} --cell {key}"
          + (f" --limit {args.limit}" if args.limit else ""))
    return 0


# --- judge (the grading host) --------------------------------------------------------------


def _find_cell(out_root: Path, token: str,
               explicit: Optional[str]) -> cs.CellStore:
    cells_dir = out_root / "cells"
    if explicit:
        cell_dir = cells_dir / explicit
        if not cell_dir.is_dir():
            _die(f"no cell dir {cell_dir}")
        name = explicit
    else:
        matches = sorted(p.name for p in cells_dir.glob(f"*__{token}__*"))
        if len(matches) != 1:
            _die(f"{len(matches)} cells match task token {token!r} under "
                 f"{cells_dir} ({matches[:4]}...) — pass --cell <key>")
        name = matches[0]
    store = cs.CellStore(out_root, name)
    manifest = json.loads(store.manifest_path.read_text())
    store.row_constants = dict(manifest["row_constants"])
    return store


def _bounds_pairs(args, spec, gens, rubric, judge):
    """The unique effective requests for the token bound + post — an
    UPPER bound (cache hits only shrink real spend)."""
    psha = judging.protocol_sha(rubric, temperature=judge.temperature,
                                max_tokens=judge.max_tokens)
    pairs = []
    for r in spec.rows:
        pairs.append((judging.JudgeRequest(
            prompt=r["prompt"], candidate_a=gens[r["item_id"]],
                candidate_b=None, roles=None, rubric_sha=rubric.sha,
                judge_model=spec.judge_model, protocol_sha=psha), rubric))
    return pairs


def cmd_judge(args: argparse.Namespace) -> int:
    task, token, _loader, rubric = TASKS[args.task]
    out_root = Path(args.out_root) / args.sweep
    store = _find_cell(out_root, token, args.cell)
    manifest = json.loads(store.manifest_path.read_text())
    if manifest["task"] != task:
        _die(f"cell {store.key} ran task {manifest['task']!r}, not {task!r}")

    rc_model = manifest["resolved_config"]["judge_model"]
    if args.judge_model not in (None, rc_model):
        _die(f"this cell was generated for judge model {rc_model!r} "
             f"(archived in the manifest, baked into its keys); judging it "
             f"under {args.judge_model!r} would file rows whose provenance "
             "lies — run generate with the other --judge-model instead")

    rows = _rows_for(args)
    if judged.rows_content_sha(rows) != manifest["items_sha"]:
        _die("the pinned rows loaded on this machine do not match the "
             "items_sha the generate phase archived — pins drifted or "
             "--limit differs; refusing to judge")

    # the judge phase rebuilds the spec from the MANIFEST (order seed,
    # gen config, judge model) — never from its own CLI defaults, so
    # position_order and the spool binding can't drift between phases
    template = dict(manifest["row_constants"])
    rc = manifest["resolved_config"]
    spec = judged.JudgedSpec(
        task=task, template=template, domain_of=dict(DOMAIN_OF), rows=rows,
        task_version=manifest["task_version"], order_seed=rc["order_seed"],
        system=None, judge_model=rc["judge_model"], gen=dict(rc["gen"]))
    spool = judged.GenSpool(out_root / manifest["spool"]["path"],
                            manifest["spool"]["header"], create=False)
    gens = spool.assert_complete([r["item_id"] for r in rows])

    cache = judging.JudgeCache()
    judge = judging.ClaudeJudge(cache=cache, concurrency=args.concurrency,
                                model=spec.judge_model)

    # the coordination post, BEFORE any judging (>5 concurrency ⇒ always
    # required at the default 200)
    pairs = _bounds_pairs(args, spec, gens, rubric, judge)
    count_client = None
    if not args.offline_bounds:
        try:
            count_client = _count_tokens_client()
        except Exception as e:  # noqa: BLE001 — bounds fall back offline
            print(f"[capeval-judged] no count-tokens client "
                  f"({type(e).__name__}) — using the offline byte bound")
    bounds = judging.measure_token_bounds(pairs, client=count_client,
                                          max_tokens=judge.max_tokens)
    n_unique = bounds["n_unique_requests"]
    est_min = max(5, round(n_unique * 5 / max(judge.concurrency, 1) / 60))
    post = judging.coordination_post(
        n_unique_requests=n_unique, concurrency=judge.concurrency,
        model=judge.model, est_duration=f"~{est_min}min",
        context=f"capeval judged controls ({task}, sweep {args.sweep})",
        bounds=bounds)
    if coordination.channel():
        print(f"[capeval-judged] {coordination.channel()} post "
              "(post BEFORE judging):\n" + post + "\n")
    manifest["judge_phase"] = {"bounds": dict(bounds),
                               "coordination_post": post,
                               "concurrency": judge.concurrency,
                               "started_at": time.time()}
    store.write_cell_manifest(manifest)

    t0 = time.time()
    if args.task == "or_bench":
        summary = judged.judge_or_bench(spool, judge, spec, store)
    elif args.task == "xstest":
        summary = judged.judge_xstest(spool, judge, spec, store)
    else:
        _die(f"--task {args.task!r} has no judge")
    summary["wall_clock_s"] = time.time() - t0
    summary["cell_key"] = store.key
    store.write_results(summary)  # the ONLY cell-done marker

    print(f"[capeval-judged] {task}: cache_hits/api_calls = "
          f"{judge.n_cache_hits}/{judge.n_api_calls} "
          f"(orgs {judge.n_calls_by_org}); cache: {cache.path}")
    print(f"[capeval-judged] results: {store.results_path}")

    if args.wandb_mode != "disabled":
        _report(args, task, store, summary)
    return 0


# --- figures + W&B (§8: aggregates + figures only — never raw text) --------------


def _rows_by_metric(store: cs.CellStore, task: str) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for r in store._parsed_rows():
        if r["task"] == task:
            out.setdefault(r["metric"], []).append(r)
    return out


def _figures(task: str, store: cs.CellStore,
             summary: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: List[Tuple[str, Any]] = []
    labels = ("full_compliance", "partial_refusal", "full_refusal")
    if task == judged.TASK_ORBENCH:
        by_cat = summary["by_category"]
        cats = sorted(by_cat)
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(cats)), 4))
        bottom = [0.0] * len(cats)
        for lab in labels:
            vals = [by_cat[c][lab] / by_cat[c]["n"] for c in cats]
            ax.bar(cats, vals, bottom=bottom, label=lab)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_ylabel("fraction")
        ax.set_title(f"OR-Bench-hard C/P/I by category (n={summary['n']})")
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        figs.append(("final_orb_cpi_by_category.png", fig))
    elif task == judged.TASK_XSTEST:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = range(len(labels))
        width = 0.38
        for k, subset in enumerate(("safe", "unsafe")):
            vals = [summary[subset][lab] for lab in labels]
            ax.bar([i + k * width for i in x], vals, width, label=subset)
        ax.set_xticks([i + width / 2 for i in x])
        ax.set_xticklabels(labels)
        ax.set_ylabel("fraction")
        ax.set_title(f"XSTest C/P/I, safe (n={summary['safe']['n']}) vs "
                     f"unsafe (n={summary['unsafe']['n']})")
        ax.legend()
        fig.tight_layout()
        figs.append(("final_xst_cpi_safe_vs_unsafe.png", fig))
    else:
        _die(f"{task!r} has no figures")
    return figs


def _report(args: argparse.Namespace, task: str, store: cs.CellStore,
            summary: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    figs = _figures(task, store, summary)
    figs_dir = Path(args.figs_dir or os.environ.get("FIGS_DIR")
                    or (Path(args.out_root) / args.sweep / "figs"))
    figs_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figs:
        fig.savefig(figs_dir / name, dpi=150)
    print(f"[capeval-judged] figures -> {figs_dir}")

    import wandb

    from sisteer.wandb_entity import wandb_project

    run = wandb.init(project=wandb_project("judged"), mode=args.wandb_mode,
                     name=f"{args.sweep}_{args.task}_judge",
                     config={"task": task, "sweep": args.sweep,
                             "cell": store.key})
    flat = {f"{args.task}/{k}": v for k, v in summary.items()
            if isinstance(v, (int, float))}
    wandb.log(flat)
    wandb.log({f"{args.task}/{name.removesuffix('.png')}":
               wandb.Image(str(figs_dir / name)) for name, _fig in figs})
    for _name, fig in figs:
        plt.close(fig)
    run.finish()


# --- cli --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "judge"):
        p = sub.add_parser(name)
        p.add_argument("--task", required=True, choices=sorted(TASKS))
        p.add_argument("--sweep", required=True)
        p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
        p.add_argument("--limit", type=int, default=None,
                       help="first-N rows AFTER the full pinned load "
                            "(smoke runs)")
        p.add_argument("--order-seed", type=int, default=0)
        p.add_argument("--max-new-tokens", type=int, default=512)
        p.add_argument("--batch-size", type=int, default=8)
    sub.choices["generate"].add_argument(
        "--judge-model", default=judging.JUDGE_MODEL,
        help="per-run judge model (spec default claude-sonnet-5); baked "
             "into the cell identity and every cached verdict key")
    sub.choices["judge"].add_argument(
        "--judge-model", default=None,
        help="optional cross-check: must MATCH the judge model the "
             "generate phase archived (the manifest is authoritative)")
    gen = sub.choices["generate"]
    gen.add_argument("--base", action="store_true",
                     help="arm=None: the plain pinned model")
    gen.add_argument("--arm-artifact", default=None)
    gen.add_argument("--arm-kind", default=None,
                     choices=("steering", "sisteer_object"))
    gen.add_argument("--arm-name", default=None)
    gen.add_argument("--layer", type=int, default=None)
    gen.add_argument("--coeff", type=float, default=1.0)
    gen.add_argument("--system-file", default=None,
                     help="render every prompt under this SI as the "
                          "system message (prompted cells)")
    jd = sub.choices["judge"]
    jd.add_argument("--cell", default=None,
                    help="explicit cell key (else the unique task match)")
    jd.add_argument("--concurrency", type=int,
                    default=judging.DEFAULT_CONCURRENCY)
    jd.add_argument("--offline-bounds", action="store_true",
                    help="skip the count-tokens endpoint for the post's "
                         "input bound (byte fallback)")
    jd.add_argument("--wandb-mode", default="online",
                    choices=("online", "offline", "disabled"))
    jd.add_argument("--figs-dir", default=None)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "generate":
        return cmd_generate(args)
    # the judge's typed credential rejection (ClaudeJudge's latch, or the count-tokens bounds pass meeting it first)
    # ends the run with the typed message as the terminal line -- no traceback; write_results (the only cell-done marker)
    # is never reached
    from sisteer import vault
    with vault.credential_exit("lowprio", prefix="[capeval-judged] "):
        return cmd_judge(args)


if __name__ == "__main__":
    sys.exit(main())
