#!/usr/bin/env python3
"""Stage-3 pool sampler — the fast path for the scale-up.

  plan     CPU-only. Materialise the prompt inventory, print per-family /
           per-source counts and the projected generation cell count. No
           model, no GPU — run it on the grading host before spending anything.
  sample   GPU. Data-parallel across every visible GPU (one vLLM engine per
           GPU, contiguous shard each), resumable per cell. Records a
           `generation_provenance.json` sidecar in the out-dir (F12: engine,
           vLLM version, sampling params, model + revision, prompt render —
           `build_two_si_pool`'s own schema/writer, reused); a resume under
           different provenance refuses before the engine builds.
  merge    Fold the per-shard JSONLs into the labelled store layout the pool
           builder reads (`<store>/<family>.jsonl`). Idempotent: re-merging
           the same shards appends nothing (cell-key dedupe), so a graded
           store keeps its graded rows. Propagates the shards' provenance
           sidecar into the store (the leg `build_two_si_pool.py manifest`
           audits), refusing a store whose recorded provenance differs.
  stats    Count what has been generated so far, per family / SI / side.

The generated rows are UNGRADED by construction: grading is the separate
grading host step (`build_two_si_pool.py grade-pos` / `grade-benign`), which is where
StrongREJECT (harmful, threshold 0.5) and the XSTest C/P/I rubric (benign) run
against the Anthropic API.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DEFAULT_REVISION = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"


def _pool_builder():
    """``scripts/build_two_si_pool.py``, imported LAZILY. It owns the
    generation-provenance schema and writer (F12) that this lane reuses —
    duplicating the schema here is exactly the drift the sidecar exists to
    catch — but it also imports THIS module at its own top level
    (``pool_sample._shard``), so a module-level import here would be circular.
    Lazy also keeps ``plan``/``stats`` free of its import cost."""
    scripts = str(Path(__file__).resolve().parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import build_two_si_pool

    return build_two_si_pool


# -- shared: the inventory ------------------------------------------------------

def _control_texts(args) -> list[str]:
    """OR-Bench-hard-1k + XSTest-safe — subtracted from every corpus up front
    so the sampled set is disjoint from the eval controls before generation
    (the pool builder's _strip_scoring is the authoritative second line)."""
    if args.no_control_subtraction:
        return []
    texts: list[str] = []
    try:
        from sisteer.data.content_sets import load_orbench
        texts += load_orbench("or-bench-hard-1k", None)
    except Exception as exc:                       # noqa: BLE001
        print(f"[plan] WARNING: OR-Bench-hard-1k unavailable ({exc}); "
              "control subtraction is INCOMPLETE", file=sys.stderr)
    try:
        from sisteer.eval.jailbreak_data import load_xstest_safe
        texts += [g.text for g in load_xstest_safe()]
    except Exception as exc:                       # noqa: BLE001
        print(f"[plan] WARNING: XSTest-safe unavailable ({exc}); "
              "control subtraction is INCOMPLETE", file=sys.stderr)
    return texts


def _goals(args):
    """JBB harmful + HarmBench (both official splits) — the goal set the eight
    attack families are crossed with."""
    from sisteer.eval.jailbreak_data import load_harmbench, load_jbb

    goals = list(load_jbb("harmful"))
    # functional="standard" ONLY (not functional=None): the ~100 extra non-standard
    # HarmBench behaviors are all "contextual" (median ~1306 chars of prepended
    # context vs ~81 for standard) — off-design for the short-imperative transforms
    # (codeattack/renellm/pasttense/...), used by no transform family in the store.
    # Widening to functional=None would quietly change what the families mean; do
    # NOT (yield pilot, 2026-08-24).
    for split in args.harmbench_splits.split(","):
        goals += load_harmbench(split.strip(), functional="standard",
                                exclude_copyright=True)
    # YIELD PILOT (2026-08-24): restrict to an explicit goal-id allowlist when
    # TT_PILOT_GOAL_IDS_FILE names a JSON list of goal_ids. pool_sample has no
    # goal filter and --limit is a STRIDE across the whole inventory, not a
    # goal subset — this env is the only seam that samples exactly N goals.
    # Absent env -> behaviour is byte-identical (no-op for every other caller).
    import os as _os
    _gf = _os.environ.get("TT_PILOT_GOAL_IDS_FILE")
    if _gf:
        allow = set(json.loads(Path(_gf).read_text()))
        goals = [g for g in goals if g.goal_id in allow]
        if not goals:
            raise SystemExit(
                f"[pilot] TT_PILOT_GOAL_IDS_FILE {_gf} matched 0 goals")
    return goals


def _inventory(args):
    from sisteer.sampling.sources import build_inventory

    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    benign = [f.strip() for f in args.benign_families.split(",") if f.strip()]
    banks = {}
    if args.subenv_bank:
        for item in args.subenv_bank:
            fam, path = item.split("=", 1)
            banks[fam] = json.loads(Path(path).read_text())
    n_benign = {}
    if args.n_benign:
        for item in args.n_benign:
            fam, n = item.split("=", 1)
            n_benign[fam] = int(n)
    return build_inventory(
        goals=_goals(args),
        attack_families=fams,
        benign_families=benign,
        n_wildjailbreak_harmful=args.n_wildjailbreak_harmful,
        n_benign_per_family=n_benign,
        seed=args.seed,
        bon_variants=args.bon_variants,
        subenv_banks=banks,
        control_texts=_control_texts(args),
        wildjailbreak_path=args.wildjailbreak_path,
        # The SAME local TSV backs both WildJailbreak strata. Without this the
        # benign loader falls through to hf_hub_download and 401s on the gated
        # repo — which is exactly what happens on a pod with no HF token, i.e.
        # the intended configuration. One flag, both sides.
        benign_paths=({"wildjailbreak_benign": args.wildjailbreak_path}
                      if args.wildjailbreak_path else None),
    )


def _specs(args):
    """The SI blocks to sample: the undefended pass plus each requested SI."""
    from sisteer.data.instructions import REGISTRY, SAFETY_SI
    from sisteer.sampling.engine import NONE_SI, GenSpec

    # SAFETY_SI (id "safety_refusal") is deliberately NOT in REGISTRY — it is
    # handed to the teacher/distill path as an Instruction OBJECT — but it IS
    # the labelled store's own prompted SI (_STORE_PROMPTED_SI) and one of the
    # three training SIs, so the sampler must be able to name it.
    by_id = {i.id: i for i in REGISTRY}
    by_id.setdefault(SAFETY_SI.id, SAFETY_SI)
    ks = {}
    for item in (args.candidates or []):
        si, k = item.split("=", 1)
        ks[si] = int(k)

    specs = []
    if not args.no_none:
        specs.append(GenSpec(NONE_SI, None, k=1))
    for si in [s.strip() for s in args.sis.split(",") if s.strip()]:
        if si not in by_id:
            raise SystemExit(f"[sample] unknown SI {si!r}; "
                             f"known: {sorted(by_id)}")
        specs.append(GenSpec(si, by_id[si].text, k=ks.get(si, 1),
                             temperature=args.temperature))
    return specs


# -- plan -----------------------------------------------------------------------

def cmd_plan(args) -> None:
    from sisteer.sampling.sources import inventory_counts

    prompts = _inventory(args)
    counts = inventory_counts(prompts)
    if getattr(args, "save_inventory", None):
        from dataclasses import asdict

        out = Path(args.save_inventory)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "key": _inventory_key(args),
            "prompts": [asdict(pr) for pr in prompts],
        }))
        print(f"[plan] wrote inventory cache -> {out} "
              f"({len(prompts)} prompts); pass it to sample/sample-all as "
              f"--inventory so no GPU worker rebuilds it", flush=True)
    specs = _specs(args)
    cells_per_prompt = sum(s.k for s in specs)
    counts["si_blocks"] = {s.si_id: s.k for s in specs}
    counts["cells_per_prompt"] = cells_per_prompt
    counts["total_cells"] = counts["n_prompts"] * cells_per_prompt
    print(json.dumps(counts, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(counts, indent=2))
        print(f"[plan] wrote {args.out}")


# -- sample ---------------------------------------------------------------------

def resolve_revision(model: str, revision: str | None) -> str | None:
    """The revision for ``model``. A pin belongs to the model it pins.

    DEFAULT_REVISION is the sha of DEFAULT_MODEL. Applying it to any other
    model is not a conservative default, it is a wrong one: the hub returns
    "Revision Not Found" and the run dies AFTER every engine has loaded. An
    explicit --revision is always honoured; a pin against a different model is
    refused rather than passed through.
    """
    if revision is None:
        return DEFAULT_REVISION if model == DEFAULT_MODEL else None
    if revision == DEFAULT_REVISION and model != DEFAULT_MODEL:
        raise SystemExit(
            f"[sample] --revision {revision} is the pinned sha for "
            f"{DEFAULT_MODEL}, but --model is {model}. A revision pin belongs "
            f"to the model it pins; pass that model's own sha, or omit "
            f"--revision to take the hub default.")
    return revision


def _shard(items, idx: int, n: int):
    """Round-robin shard, so every worker gets the same MIX of families.

    Was a contiguous slice, on the reasoning that it keeps each shard's family
    mix stable for reporting. Measured 2026-08-26, that reasoning was backwards:
    the inventory is ORDERED BY FAMILY, so contiguous slices give each shard a
    DIFFERENT mix, and the mixes differ in cost. Harmful families mostly yield
    short refusals; the benign tail (phtest, wildjailbreak_benign) yields long
    helpful answers, and at max_new_tokens 4096 that is several times the
    decode. On an 8-way split of the Llama run, shards 0-3 (harmful) finished
    while shards 4-7 (benign) were at 63-94% -- four H100s idle at ~$3.6/hr
    each, and pod utilisation halved.

    A stride keeps prefix caching intact: cache hits come from prompts SHARING
    an SI prefix, and VllmSampler.run already batches SI-block by SI-block, so
    every prompt in a block shares the prefix regardless of which shard it came
    from. What contiguity actually bought was the reporting convenience, which
    is not worth half the GPUs.
    """
    return items[idx::n]


def _inventory_cached(args):
    """The inventory, materialised ONCE and reused by every worker.

    gpu-util checklist item 2. Without this, each of N per-GPU workers
    independently rebuilds the whole prompt inventory -- loading the
    WildJailbreak TSV, HarmBench and JBB, and running every attack transform
    -- WHILE HOLDING AN IDLE GPU, and does it N times concurrently on one
    node's CPUs and one network filesystem. The GPU cannot start until its
    worker finishes CPU work that is identical across all N.

    With --inventory pointing at a file written by `plan --save-inventory`,
    the workers read it instead. The file is keyed by the inventory-defining
    args, so a stale cache from a different family set or seed is REFUSED
    rather than silently sampled -- a wrong inventory that runs is the whole
    class of bug this campaign keeps hitting.
    """
    path = getattr(args, "inventory", None)
    if not path:
        return _inventory(args)
    from sisteer.sampling.sources import SamplePrompt

    blob = json.loads(Path(path).read_text())
    want = _inventory_key(args)
    if blob.get("key") != want:
        raise SystemExit(
            f"[sample] inventory cache {path} was built for different "
            f"inputs and would sample the wrong prompts.\n"
            f"  cache: {blob.get('key')}\n  run:   {want}\n"
            f"Rebuild it: pool_sample.py plan --save-inventory {path} "
            f"<same args as this run>")
    prompts = [SamplePrompt(**r) for r in blob["prompts"]]
    # a cached inventory skips sources' own check, so the evaluation-only corpus (the ReNeLLM
    # paper corpus, under its no-training terms) is refused here too (counts only)
    from sisteer.eval.attacks import eval_only
    eval_only.refuse_texts((p.text for p in prompts),
                           consumer=f"training-pool sampling (the cached inventory {path})")
    print(f"[sample] inventory: {len(prompts)} prompts from cache {path} "
          f"(not rebuilt -- gpu-util item 2)", flush=True)
    return prompts


def _inventory_key(args) -> dict:
    """Everything that changes WHICH prompts the inventory contains."""
    return {
        "families": sorted(f.strip() for f in args.families.split(",") if f.strip()),
        "benign_families": sorted(
            f.strip() for f in args.benign_families.split(",") if f.strip()),
        "n_wildjailbreak_harmful": args.n_wildjailbreak_harmful,
        "n_benign": sorted(args.n_benign or []),
        "harmbench_splits": args.harmbench_splits,
        "seed": args.seed,
        "bon_variants": args.bon_variants,
        "subenv_bank": sorted(args.subenv_bank or []),
        "control_subtraction": not args.no_control_subtraction,
        "wildjailbreak_path": args.wildjailbreak_path,
    }


def _write_sample_provenance(args, specs) -> None:
    """F12: record what generates this out-dir's rows — engine, vLLM version,
    sampling params, model + revision, and the canonical prompt render — in
    the EXACT schema ``build_two_si_pool``'s manifest validates, via its own
    helpers (imported, never re-spelled, so the schema cannot drift). Written
    BEFORE the engine builds, matching every other generation leg's arm-time
    guard: a resume whose provenance differs (engine, model, sampler, render)
    REFUSES here, before any GPU time is spent. ``merge`` then propagates the
    record into the labelled store the manifest audits.

    The per-SI K lives in ``info`` (outside the strict resume set) — raising
    K is the sanctioned delta workflow, a coverage change, not a distribution
    change. Every shard writes byte-identical content (all derive from the
    same args), so the concurrent sample-all children pass each other's
    strict check by construction."""
    bp = _pool_builder()
    # vLLM-only lane: fail with the ACTIONABLE install/fallback error before
    # recording anything (a sidecar for a run that could never generate would
    # poison later resumes with vllm_version: null).
    vllm_mod = bp._require_vllm("pool-sample")
    from sisteer.models import load_tokenizer

    # The SAME construction VllmSampler now uses internally (F8), so the
    # recorded template kwargs + prefix hashes describe the engine's actual
    # renders.
    tok = load_tokenizer(args.model, revision=args.revision)
    prov = bp._generation_provenance(
        engine="vllm", leg="pool-sample",
        cfg={"model": args.model, "model_revision": args.revision},
        vllm_version=getattr(vllm_mod, "__version__", None),
        sampling_params={
            "temperature": float(args.temperature),
            "top_p": 1.0,
            "max_tokens": int(args.max_new_tokens),
            "base_seed": int(args.seed),
            "seed_rule": "engine seed + 1000 * candidate_idx (greedy: None)",
            "sample_rule": "per-SI GenSpec: k == 1 greedy, k > 1 sampled",
        },
        prompt_render=bp._prompt_render_record(
            tok, {s.si_id: s.system_text for s in specs}),
        info={"k_by_si": {s.si_id: int(s.k) for s in specs}})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bp._check_and_write_provenance(out_dir, prov, tag="pool-sample")


def cmd_sample(args) -> None:
    from sisteer.sampling.engine import VllmSampler, load_done_keys

    prompts = _inventory_cached(args)
    if args.limit and args.limit < len(prompts):
        # STRIDE, not a prefix: the inventory is grouped by family (all bon,
        # then all pasttense, ... then the benign tail), so prompts[:N] would
        # smoke-test exactly one family and report a throughput number for the
        # shortest prompts in the set. A stride keeps every family and both
        # sides represented, and is deterministic.
        step = len(prompts) / args.limit
        prompts = [prompts[int(i * step)] for i in range(args.limit)]
    shard = _shard(prompts, args.shard_index, args.shard_count)
    out = Path(args.out_dir) / f"shard_{args.shard_index:02d}.jsonl"
    print(f"[sample] shard {args.shard_index}/{args.shard_count}: "
          f"{len(shard)} of {len(prompts)} prompts -> {out}", flush=True)

    specs = _specs(args)
    _write_sample_provenance(args, specs)   # F12 — before the engine builds
    sampler = VllmSampler(
        args.model, revision=resolve_revision(args.model, args.revision),
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed, max_new_tokens=args.max_new_tokens,
        enforce_eager=args.enforce_eager,
        # getattr, not attribute access: programmatic callers (tests, the
        # orchestrator) build bare Namespaces that predate these flags — the
        # same convention train_two_si_objects uses for `positions`. The
        # defaults match the argparse defaults, so a CLI run is unchanged.
        think_mode=getattr(args, "think_mode", "off"),
        max_num_seqs=getattr(args, "max_num_seqs", None))
    res = sampler.run(shard, specs, out, done=load_done_keys(out))
    print(json.dumps(res, indent=2))


def resolve_gpus(arg: str | None, visible: list[str]) -> list[str]:
    """Device ids to fan out over. Refuses a GPU COUNT typed into a LIST flag.

    ``--gpus`` has always been a comma-separated list of device ids, and that
    bit twice in one afternoon (2026-08-26):

      --gpus 8  on an 8-GPU node  -> device "8"; ids are 0-7, so it names
                                     nothing that exists
      --gpus 4  in a 4-GPU alloc  -> device "4" ALONE. Job 212789 ran, printed
                                     "global shards [0, 1) of 8", and was on
                                     course to generate one EIGHTH of the
                                     inventory and exit 0.

    The shard-space overflow check downstream cannot catch either, because one
    shard always fits inside the global space. Comparing the requested ids
    against the visible ones is what separates "a deliberate subset of this
    node's GPUs" from "someone typed a count".
    """
    ids = ([g.strip() for g in arg.split(",") if g.strip()] if arg
           else list(visible))
    if not ids:
        raise SystemExit("[sample-all] no GPUs found")
    unknown = [g for g in ids if g not in visible] if visible else []
    if unknown:
        raise SystemExit(
            f"[sample-all] --gpus names device(s) {unknown} that are not "
            f"visible here (visible: {visible}). --gpus takes a "
            f"COMMA-SEPARATED LIST OF DEVICE IDS, NOT A GPU COUNT -- if you "
            f"meant 'use N GPUs', omit --gpus entirely and every visible "
            f"device is used. Refusing rather than silently sampling a "
            f"fraction of the inventory.")
    return ids


def cmd_sample_all(args) -> None:
    """Fan out one `sample` child per GPU, each pinned with CUDA_VISIBLE_DEVICES.

    Separate processes rather than threads: each vLLM engine owns a CUDA
    context and a memory pool, and one process per GPU is the only arrangement
    that keeps `gpu_memory_utilization` meaning what it says.

    MULTI-POD fan-out is the same mechanism one level up. The shard space is
    GLOBAL, not per-pod: `--shard-total` is the total GPU count across every
    pod and `--shard-offset` is this pod's first global shard index, so pod p
    with G GPUs runs global shards [offset, offset+G). Nothing here talks to
    another pod — the engines are independent (no NCCL, no collectives), so a
    32-GPU run is 32 disjoint slices of one deterministic inventory and the
    node boundary costs exactly nothing. Defaults (offset 0, total = local
    GPUs) reproduce the single-pod behaviour."""
    gpus = resolve_gpus(args.gpus, _visible_gpus())
    total = args.shard_total or len(gpus)
    offset = args.shard_offset
    if offset + len(gpus) > total:
        raise SystemExit(
            f"[sample-all] shard space overflow: offset {offset} + "
            f"{len(gpus)} local GPUs > --shard-total {total}. Every pod's "
            "[offset, offset+gpus) must fit inside one global shard space, or "
            "shards silently overlap and rows are generated twice.")
    print(f"[sample-all] global shards [{offset}, {offset + len(gpus)}) "
          f"of {total}", flush=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    procs = []
    # Put the running interpreter's bin/ on PATH for the children. Invoking a
    # venv python by ABSOLUTE PATH does not add its bin/ to PATH, and vLLM
    # shells out to `ninja` when it compiles the graph — which then fails with
    # a bare FileNotFoundError('ninja') even though ninja is installed in that
    # very venv. Same class as the verl-cluster skill's "ninja on PATH" note.
    bindir = str(Path(sys.executable).parent)
    base_env = dict(os.environ)
    base_env["PATH"] = bindir + os.pathsep + base_env.get("PATH", "")
    for i, gpu in enumerate(gpus):
        env = dict(base_env, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [sys.executable, __file__, "sample",
               "--shard-index", str(offset + i), "--shard-count", str(total)]
        cmd += _passthrough(args)
        # GLOBAL index, matching the data file `sample` writes
        # (shard_{shard_index:02d}.jsonl). Using the LOCAL i here meant pod B
        # of a multi-pod run wrote shard_00.log..shard_03.log over pod A's --
        # data was always safe (global index), but the logs of the pod that
        # finished first were destroyed by the one that finished second, which
        # is exactly the diagnostic you need when one pod fails and the other
        # does not.
        log = Path(args.out_dir) / f"shard_{offset + i:02d}.log"
        print(f"[sample-all] GPU {gpu} -> shard {offset + i} (log {log})",
              flush=True)
        if i and args.stagger_secs:
            # gpu-util item 2: stagger concurrent model loads off a shared
            # filesystem. N engines each pulling ~15 GiB of weights from the
            # same NFS mount at t=0 serialises on the network, not the GPU.
            time.sleep(args.stagger_secs)
        procs.append((i, gpu, subprocess.Popen(
            cmd, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT)))
    failed = []
    for i, gpu, p in procs:
        rc = p.wait()
        print(f"[sample-all] shard {i} (GPU {gpu}) exit {rc}", flush=True)
        if rc != 0:
            failed.append((i, gpu, rc))
    if failed:
        raise SystemExit(f"[sample-all] {len(failed)} shard(s) FAILED: {failed}")
    print("[sample-all] all shards complete")


def _visible_gpus() -> list[str]:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        return [g.strip() for g in vis.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:                              # noqa: BLE001
        return []


_PASSTHROUGH = (
    "families", "benign_families", "sis", "model", "revision", "dtype",
    "harmbench_splits", "wildjailbreak_path", "out_dir", "think_mode",
    # children read the prebuilt inventory rather than each rebuilding it
    # (gpu-util item 2). NOT "save_inventory": only `plan` writes it, and a
    # worker rewriting it mid-run would race its siblings.
    "inventory",
)
_PASSTHROUGH_NUM = (
    "seed", "bon_variants", "n_wildjailbreak_harmful", "max_new_tokens",
    "max_num_seqs",
    "max_model_len", "gpu_memory_utilization", "temperature", "limit",
)


def _passthrough(args) -> list[str]:
    out: list[str] = []
    for name in _PASSTHROUGH + _PASSTHROUGH_NUM:
        v = getattr(args, name, None)
        if v is not None:
            out += [f"--{name.replace('_', '-')}", str(v)]
    for name in ("no_none", "no_control_subtraction", "enforce_eager"):
        if getattr(args, name, False):
            out.append(f"--{name.replace('_', '-')}")
    # repeatable flags: emit one --flag VALUE per collected item
    for name in ("candidates", "subenv_bank", "n_benign"):
        flag = f"--{name.replace('_', '-')}"
        for item in (getattr(args, name, None) or []):
            out += [flag, item]
    return out


# -- gen-bank -------------------------------------------------------------------

AMPLEGCG_MODEL = "osunlp/AmpleGCG-plus-llama2-sourced-llama2-7b-chat"


def cmd_gen_bank(args) -> None:
    """Generate a family's prompt bank on GPU, for families whose prompts a
    MODEL must produce (currently only amplegcg_plus).

    Writes ``{goal_id: [full_prompt, ...]}`` JSON, which `sample`/`plan` then
    replay via ``--subenv-bank amplegcg_plus=<path>``. Split out from sampling
    because it runs a DIFFERENT model (the 7B suffix generator, not the target)
    and only over the JBB+HarmBench goals — WildJailbreak rows are already
    adversarial and are never crossed with a family.

    No sub-env: `_amplegcg_worker` uses only AutoModelForCausalLM + generate(),
    so it runs in the main env despite the historical 'transformers<4.56'
    docstring. Resumes by skipping goal_ids already in an existing bank."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if args.family != "amplegcg_plus":
        raise SystemExit(f"[gen-bank] no generator for family {args.family!r}; "
                         "only amplegcg_plus needs one; the rest are CPU transforms)")
    out = Path(args.out)
    bank = json.loads(out.read_text()) if out.exists() else {}
    goals = _goals(args)
    todo = [g for g in goals if g.goal_id not in bank]
    print(f"[gen-bank] {len(todo)} of {len(goals)} goals to generate "
          f"({len(bank)} already banked) -> {out}", flush=True)
    if not todo:
        return

    set_seed(args.seed)
    from sisteer.models import configure_fast_sdpa
    configure_fast_sdpa()  # cuDNN-SDPA footgun — every load site, see models.py
    tok = AutoTokenizer.from_pretrained(args.generator_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.generator_model, torch_dtype="auto").eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    dev = next(model.parameters()).device

    n = args.n_suffixes
    for i, g in enumerate(todo):
        # The released AmpleGCG input format (see _amplegcg_worker.py).
        enc = tok(f"### Query:{g.text} ### Prompt:", return_tensors="pt").to(dev)
        with torch.no_grad():
            gen = model.generate(
                **enc, num_beams=n, num_beam_groups=n, diversity_penalty=1.0,
                num_return_sequences=n, max_new_tokens=64, do_sample=False)
        sfx = [tok.decode(x, skip_special_tokens=True).strip()
               for x in gen[:, enc["input_ids"].shape[1]:]]
        # The attack prompt is goal + adversarial suffix (the paper's replay).
        bank[g.goal_id] = [f"{g.text} {x}".strip() for x in sfx if x.strip()]
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(bank))
            print(f"[gen-bank] {i + 1}/{len(todo)} goals", flush=True)
    out.write_text(json.dumps(bank))
    sizes = [len(v) for v in bank.values()]
    print(f"[gen-bank] wrote {len(bank)} goals, "
          f"{sum(sizes)} prompts (min/max per goal {min(sizes)}/{max(sizes)})")


# -- merge / stats --------------------------------------------------------------

def _merge_cell_key(row: dict) -> "str | None":
    """The generation-cell identity ``engine.cell_key`` stamps, rebuilt from a
    raw row (the same reconstruction ``engine.load_done_keys`` performs, made
    tolerant of legacy rows whose variant key is ``order``-based). None when
    no key can be derived — such a row is appended as-is, never deduped."""
    if row.get("sample_idx") is not None:
        vk = f"{int(row['sample_idx'])}:{int(row.get('trial') or 0)}"
    elif row.get("order") is not None:
        vk = str(int(row["order"]))
    else:
        return None
    si = row.get("si_id") or "none"
    return (f"{row.get('attack')}|{row.get('goal_id')}|{vk}|{si}|"
            f"{int(row.get('candidate_idx') or 0)}")


def _existing_cell_keys(path: Path) -> set:
    """Cell keys already in one store family file (tolerating blank/torn
    lines the way every other store reader does)."""
    keys: set = set()
    if not path.exists():
        return keys
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = _merge_cell_key(row)
            if k is not None:
                keys.add(k)
    return keys


def cmd_merge(args) -> None:
    """Fold shard JSONLs into `<store>/<family>.jsonl` — the layout
    `build_two_si_pool._iter_store_rows` streams.

    IDEMPOTENT (2026-08-20): each row's generation cell (`_merge_cell_key`,
    the `engine.cell_key` identity: attack|goal_id|variant_key|si|
    candidate_idx) is appended only if the store does not already hold it —
    within this merge or from an earlier one. So re-running merge on the same
    shards adds nothing, a re-merge after more sampling appends exactly the
    new cells, and a store `grade-store` has already scored KEEPS its graded
    rows. Dedupe was chosen over a refuse-unless-`--append` flag precisely so
    a re-merge never pushes anyone to clear a graded store; silent duplicates
    would corrupt the rejection-sampling accounting unrecoverably."""
    store = Path(args.store_dir).expanduser()
    store.mkdir(parents=True, exist_ok=True)
    os.chmod(store, 0o700)

    # F12: carry the shards' generation-provenance sidecar into the store —
    # the leg the manifest actually audits (`_gather_generation_provenance`
    # reads the STORE dir; without a sidecar there, a config that declares
    # pool.engine refuses to freeze). Propagated BEFORE any row lands, so a
    # store already holding a DIFFERENT record (another engine/model/sampler)
    # refuses the merge before its rows are mixed — the same arm-time guard
    # every generation leg runs. Shards with no sidecar (generated before
    # provenance recording) keep today's behaviour exactly: nothing written,
    # merge proceeds, and the manifest layer polices the absence.
    bp = _pool_builder()
    prov_src = Path(args.out_dir) / bp.GEN_PROVENANCE_NAME
    if prov_src.exists():
        try:
            prov = json.loads(prov_src.read_text())
        except json.JSONDecodeError:
            raise SystemExit(
                f"[merge] {prov_src} is unreadable — refusing to propagate a "
                "corrupt provenance record into the store. Restore it (or "
                "regenerate the shards) before merging.")
        bp._check_and_write_provenance(store, prov, tag="merge")
        print(f"[merge] propagated {bp.GEN_PROVENANCE_NAME} into {store}")
    else:
        print(f"[merge] WARNING: {prov_src} not found — these shards predate "
              "provenance recording, so the store keeps no engine record and "
              "a manifest under a pool.engine-declaring config will need "
              "--allow-unrecorded-provenance.", file=sys.stderr)

    handles: dict[str, object] = {}
    seen: dict[str, set] = {}
    n = n_dup = 0
    try:
        for shard in sorted(Path(args.out_dir).glob("shard_*.jsonl")):
            with shard.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    fam = row.get("attack")
                    if fam not in handles:
                        p = store / f"{fam}.jsonl"
                        seen[fam] = _existing_cell_keys(p)  # before open("a")
                        handles[fam] = p.open("a")
                        os.chmod(p, 0o600)
                    key = _merge_cell_key(row)
                    if key is not None:
                        if key in seen[fam]:
                            n_dup += 1
                            continue
                        seen[fam].add(key)
                    handles[fam].write(line if line.endswith("\n")
                                       else line + "\n")
                    n += 1
    finally:
        for h in handles.values():
            h.close()
    print(f"[merge] wrote {n} rows into {len(handles)} family file(s) "
          f"under {store} ({n_dup} already-present cell(s) skipped — merge "
          "is idempotent)")


def cmd_stats(args) -> None:
    from collections import Counter

    fam, si, side = Counter(), Counter(), Counter()
    n = 0
    for shard in sorted(Path(args.out_dir).glob("shard_*.jsonl")):
        with shard.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                fam[r.get("attack")] += 1
                si[r.get("si_id") or "none"] += 1
                side[r.get("side")] += 1
    print(json.dumps({"n_rows": n, "by_family": dict(fam),
                      "by_si": dict(si), "by_side": dict(side)}, indent=2))


# -- CLI ------------------------------------------------------------------------

def _common(ap, *, families_required: bool = True) -> None:
    # REQUIRED, deliberately. This default used to be
    # "bon,pasttense,pap,encoding,aim,gcg_replay" -- a frozen copy of the old
    # eight-family set that had silently diverged from the config's
    # pool.static_families (now bon, pasttense, codeattack, renellm, grok420,
    # wildjailbreak_harmful). Sampling with the default would have quietly
    # built a pool of three families we no longer train on while omitting
    # three we do, and nothing downstream compares the two lists: the pool
    # builder reads the CONFIG, so the mismatch surfaces only as a
    # PoolCompositionError about "zero realised pairs" long after the GPU
    # hours are spent. An explicit list cannot go stale.
    ap.add_argument("--families", required=families_required,
                    default=None if families_required else "",
                    help="attack families crossed with JBB+HarmBench goals; "
                         "must match the config's pool.static_families "
                         "(minus corpus families like wildjailbreak_harmful, "
                         "which enter via --n-wildjailbreak-harmful)")
    ap.add_argument("--benign-families", default="phtest,wildjailbreak_benign")
    ap.add_argument("--inventory", default=None,
                    help="read the prompt inventory from this cache "
                         "(written by `plan --save-inventory`) instead of "
                         "rebuilding it in every GPU worker. Refused if it "
                         "was built for different inputs.")
    ap.add_argument("--think-mode", default="off",
                    choices=("off", "isolated", "as_shipped"),
                    help="reasoning-model think block. off = pre-closed, the "
                         "cross-model comparable default. isolated = thinking "
                         "ON with the system turn STILL carrying only our SI "
                         "(the valid single-variable contrast). as_shipped = "
                         "the template default, which for Qwen3.8 ALSO "
                         "prepends its own reasoning directive ahead of our "
                         "SI -- a different question, two variables.")
    ap.add_argument("--save-inventory", default=None,
                    help="`plan` only: write the materialised inventory here")
    ap.add_argument("--sis", default="safety_refusal,si_constitution,si_irrelevant")
    ap.add_argument("--candidates", action="append", metavar="SI=K",
                    help="rejection-sampling K per SI, e.g. si_irrelevant=8")
    ap.add_argument("--harmbench-splits", default="val,test")
    ap.add_argument("--n-wildjailbreak-harmful", type=int, default=None)
    ap.add_argument("--n-benign", action="append", metavar="FAM=N")
    ap.add_argument("--wildjailbreak-path", default=None)
    ap.add_argument("--subenv-bank", action="append", metavar="FAM=PATH")
    ap.add_argument("--bon-variants", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-none", action="store_true",
                    help="skip the undefended pass (already in the store)")
    ap.add_argument("--no-control-subtraction", action="store_true")


def _gpu_args(ap) -> None:
    ap.add_argument("--model", default=DEFAULT_MODEL)
    # SENTINEL, not DEFAULT_REVISION. The pinned revision belongs to the pinned
    # MODEL: --model and --revision defaulted as an unlinked pair, so overriding
    # the model alone silently kept Llama's sha and every other model died with
    # "Revision Not Found" -- after eight engines had loaded (2026-08-26,
    # gemma-4-E4B-it). resolve_revision() below pairs them.
    ap.add_argument("--revision", default=None,
                    help="model revision; defaults to the pinned sha ONLY when "
                         "--model is the pinned model, else to the hub default")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="concurrent decode sequences; omit for vLLM's "
                         "default (1024). MUST be lowered for a hybrid "
                         "Mamba/attention model whose weights leave little "
                         "cache -- each decode sequence needs one Mamba cache "
                         "block and vLLM refuses to start when the default "
                         "exceeds what fits (Qwen3.8-27B on an 80GB card: 310 "
                         "blocks available).")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="/workspace/pool_sample")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan"); _common(p)
    p.add_argument("--out", default=None); p.set_defaults(func=cmd_plan)

    p = sub.add_parser("sample"); _common(p); _gpu_args(p)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("sample-all"); _common(p); _gpu_args(p)
    p.add_argument("--gpus", default=None,
                   help="COMMA-SEPARATED DEVICE IDS (e.g. 0,1,2,3), NOT a count. Omit to use every visible device, which is what a Slurm allocation already scopes for you.")
    p.add_argument("--stagger-secs", type=float, default=20.0,
                   help="delay between per-GPU engine launches, so N "
                        "workers do not pull the model off a shared "
                        "filesystem simultaneously (gpu-util item 2). "
                        "0 disables.")
    p.add_argument("--shard-offset", type=int, default=0,
                   help="this pod's first GLOBAL shard index (multi-pod fan-out)")
    p.add_argument("--shard-total", type=int, default=None,
                   help="total GPUs across ALL pods (default: local GPU count)")
    p.set_defaults(func=cmd_sample_all)

    # gen-bank builds a prompt bank for ONE family (--family), so the
    # plural --families it inherits from _common is meaningless here.
    # Requiring it -- which making it required for the sampling paths
    # accidentally did -- breaks a subcommand that never used it.
    p = sub.add_parser("gen-bank"); _common(p, families_required=False)
    p.add_argument("--family", default="amplegcg_plus")
    p.add_argument("--generator-model", default=AMPLEGCG_MODEL)
    p.add_argument("--n-suffixes", type=int, default=8)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_gen_bank)

    p = sub.add_parser("merge")
    p.add_argument("--out-dir", default="/workspace/pool_sample")
    p.add_argument("--store-dir", required=True); p.set_defaults(func=cmd_merge)

    p = sub.add_parser("stats")
    p.add_argument("--out-dir", default="/workspace/pool_sample")
    p.set_defaults(func=cmd_stats)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
