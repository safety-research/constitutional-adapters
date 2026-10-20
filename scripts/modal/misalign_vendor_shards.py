"""em-misalign: the qwen VENDOR-PRESET agentic shards on Modal H200s (murder; leaking, every qwen stem).

GENERALISED 2026-09-16 (clusters clogged, run on Modal): one app runs a PLAN of (target, shard) items —
  --plan "qwen35_4b:13:0,2-12;qwen35_4b_it:8:0-7;qwen35_4b_bl:4:0-3"   (target:n_shards:shard-list; `;` between targets)
  --root misalign_vendor_leak   (the output root under /vol; default misalign_vendor = the murder trees, never appended to)
  --behaviors leaking
Each target reads its own config/manifest from the staged tree and its own bank view at /vol/misalign/objects/<target>/cells_view
(qwen35_4b_it: the instruct-direct objects; qwen35_4b_bl: the pinned baselines, whose pins ride the rows as on the cluster).
ORIGINAL NOTE FOLLOWS (shards 5-12 of the murder run).

No more than 16 GPUs are left pending on cluster B; pending short work moves to Modal and cluster A.
The vendor re-sample of qwen's murder column is 13 legs of ~1 GPU-hour each; shards 0–4 run on
cluster A (arrays 486968 / 487069), the eight object legs (shards 5–12) run here, ONE CONTAINER PER
LEG, with the SAME 13-way binning as misalign_proxy.sbatch's vendor mode (sorted cell ids,
shard k = ids[k::13]) so the two substrates' cells are disjoint and the merge on the grading host is
a plain copy.

Per container (mirrors misalign_proxy.sbatch's vendor mode step for step):
  1. the in-tree served stack from a STAGED tree (/vol/capctrl/repos/sisteer-<sha>, the cluster's exact
     sha): scripts/sdfbase/mt_serving_up.py --mode petri with configs/misalign_full22_qwen35_4b.yaml +
     its manifest, the misalign bank view staged at /vol/misalign/objects/qwen35_4b/cells_view (the nine
     object dirs, symlinks resolved; sha-verified by the standup against the registry), stacked
     reductions attested, at 32768 x 22 with --allow-over-budget — the SOURCE run's measured budget
     (44 x 16384 for 7 h, 22 x 32768 for the ladder's seven shards), which the merged geometry table's
     provisional ceiling for qwen35_4b does not know about;
  2. cells.json from the standup's own enumeration (the sbatch's block, no organisms);
  3. sisteer.capeval.misalign.runner for this shard's ONE leg: --family agentic --behaviors murder
     --n 10 --max-tokens 16384 --sampler vendor --only-cells <leg> --concurrency 22;
  4. outputs under /vol/misalign_vendor/qwen35_4b/ (agentic/<cell>/, serving_shard<k>/, cells_shard<k>.json,
     endpoint.snapshot_shard<k>.json, RESULT), the volume committed every five minutes while the runner
     runs and at the end, so a container death never loses finished responses.

The image is the proven cu129 vllm-0.25.1 / vllm-lens-1.2.1 / transformers-5.14.1 stack (copied from
petri_agentloop_control.py) plus httpx for the runner. HF_HUB_OFFLINE: Qwen/Qwen3.5-4B @851bf6e8 is on the
volume's hf cache. No API key anywhere (the runner talks to loopback; judging happens on the grading host).

Stage (the grading host):
  git archive <sha> | tar -x -C /tmp/stage/sisteer-<sha>
  modal volume put twosi-fits /tmp/stage/sisteer-<sha> capctrl/repos/sisteer-<sha>
  rsync -aL <login-host>:$SHARED/exp/objects/misalign/qwen35_4b/cells_view/ /tmp/stage/cells_view/   # COMPLETE dirs, whole:
  #   the registry re-hashes every COMPLETE-manifested file (metrics.jsonl included) — an excluded file is a torn publication
  modal volume put twosi-fits /tmp/stage/cells_view misalign/objects/qwen35_4b/cells_view
Run:
  modal run --detach scripts/modal/misalign_vendor_shards.py --sha <sha> --shards 5,6,7,8,9,10,11,12
Pull:
  modal volume get twosi-fits misalign_vendor/qwen35_4b $GRADE/misalign_vendor/  (then merge with the cluster A root)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

APP_NAME = "em-misalign-vendor-qwen35_4b"
app = modal.App(APP_NAME)
VOL = modal.Volume.from_name("twosi-fits", create_if_missing=False)
VOL_PATH = "/vol"

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "https://github.com/vllm-project/vllm/releases/download/v0.25.1/"
        "vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
        "#sha256=9e206f370c934a2d4b6b1f05d3d09708d344e05d80260189ef19f60755709431",
        "torch==2.11.0+cu129", "torchvision==0.26.0+cu129",
        "torchaudio==2.11.0+cu129", "vllm-lens==1.2.1",
        "pandas", "pyarrow", "aiohttp", "pyyaml", "numpy",
        "huggingface_hub", "ninja", "httpx",
        extra_index_url="https://download.pytorch.org/whl/cu129",
    )
    .pip_install("transformers==5.14.1", "tokenizers==0.22.2",
                 extra_options="--no-deps --force-reinstall")
    .run_commands(
        "python -c \"import torch, vllm, transformers, aiohttp, yaml, httpx; "
        "assert torch.__version__ == '2.11.0+cu129', torch.__version__; "
        "assert vllm.__version__.startswith('0.25.1'), vllm.__version__; "
        "print('IMAGE OK', vllm.__version__, torch.__version__)\"",
    )
    .env({"HF_HOME": f"{VOL_PATH}/hf", "HF_HUB_OFFLINE": "1",
          "TOKENIZERS_PARALLELISM": "false"})
)

REPOS = f"{VOL_PATH}/capctrl/repos"
ROOT_DEFAULT = "misalign_vendor"
#: the qwen stems this launcher serves: (model key for the standup, wave view recorded in cells_full, baselines panel?)
TARGETS = {"qwen35_4b": dict(model_key="qwen35_4b", view="main", baselines=False),
           "qwen35_4b_it": dict(model_key="qwen35_4b", view="it", baselines=False),
           "qwen35_4b_bl": dict(model_key="qwen35_4b", view="main", baselines=True),
           # the three ReLU / PSR psr_ce_all cells for the qwen leaking vendor panel, NATIVE path,
           # paired against the panel's existing lens `none` (no native anchor) — so the rows land in the qwen35_4b panel root
           # and the shard files carry a `psr_` tag so they never collide with the lens shards' files
           "qwen35_4b_psr": dict(model_key="qwen35_4b", view="main", baselines=False, out_target="qwen35_4b",
                                 objects_target="qwen35_4b", file_tag="psr_"),
           # Scale's Qwen Arditi ALL-POSITIONS arm — its own `_bl`-style panel [none, arditi_allpos]
           # on the `it` view, a SINGLE-OBJECT native standup; the object's digest is ALSO held explicitly at the standup
           # (--expect-object-sha256, recorded in endpoint.json) beside the view manifest's pin-form entry
           "qwen35_4b_allpos": dict(model_key="qwen35_4b", view="it", baselines=False,
                                    expect_digests={"arditi_allpos": "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2"})}
SERVING_PATHS = ("lens", "lora_native")
#: the native path's content-addressed ADAPTER BANK (exported adapters): the same root the tinycap launcher exports into every
#: container (tinycap_served.ADAPTER_BANK_ROOT = contracts.VOL_TINYCAP_ROOT + "/multilora_adapters") — off-cluster there is no
#: shared root to default it under, so it is BOTH exported (SISTEER_ADAPTER_BANK_ROOT) and passed explicitly (--adapter-bank
#: <root>/<model key>, the deterministic default the core's resolve_adapter_bank would derive)
ADAPTER_BANK_ROOT = "/vol/tinycap/multilora_adapters"
PIN_KEYS = ("rung", "rung_i", "lambda", "alpha", "object_sha256", "selection_outcome", "contamination", "source")


def target_paths(target: str, root: str = ROOT_DEFAULT) -> dict:
    """Every per-target path and fact the shard needs; refuses an unknown stem or a root outside misalign_vendor*."""
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r} (known: {sorted(TARGETS)})")
    if not (root == ROOT_DEFAULT or root.startswith(ROOT_DEFAULT + "_")):
        raise ValueError(f"root must be {ROOT_DEFAULT} or {ROOT_DEFAULT}_<suffix>; got {root!r}")
    t = TARGETS[target]
    return dict(target=target, root=root, model_key=t["model_key"], view=t["view"], baselines=t["baselines"],
                family=t["model_key"], file_tag=t.get("file_tag", ""), expect_digests=dict(t.get("expect_digests") or {}),
                objects=f"{VOL_PATH}/misalign/objects/{t.get('objects_target', target)}/cells_view",
                out_root=f"{VOL_PATH}/{root}/{t.get('out_target', target)}",
                cfg=f"configs/misalign_full22_{target}.yaml",
                manifest=f"configs/capeval/manifests/misalign_full22_{target}.yaml")


def parse_shards(spec: str) -> list:
    """``"0,2-12"`` -> [0, 2, 3, ..., 12]; refuses duplicates and negatives."""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if len(set(out)) != len(out) or any(k < 0 for k in out):
        raise ValueError(f"bad shard list {spec!r}")
    return out


def parse_plan(plan: str) -> list:
    """``"qwen35_4b:13:0,2-12;qwen35_4b_it:8:0-7"`` -> [{"target", "k", "n_shards"}, ...] in plan order."""
    items = []
    for entry in str(plan).split(";"):
        entry = entry.strip()
        if not entry:
            continue
        target, n_shards, shards = entry.split(":", 2)
        target_paths(target)   # refuses an unknown stem before any container starts
        n = int(n_shards)
        for k in parse_shards(shards):
            if k >= n:
                raise ValueError(f"shard {k} >= n_shards {n} for {target}")
            items.append({"target": target, "k": k, "n_shards": n})
    if not items:
        raise ValueError("empty plan")
    return items


def baseline_pins(tree: str, family: str) -> dict:
    """The `_bl` panels' pins as misalign_proxy.sbatch records them (F22_BASELINE_PINS): per cell, the PIN_KEYS present."""
    pins = json.loads(Path(f"{tree}/configs/misalign_baseline_pins.json").read_text())[family]
    return {c: {k: pins[c][k] for k in PIN_KEYS if k in pins[c]} for c in ("arditi", "alphasteer")}
#: the 16k-cap window. Concurrency is a PARAMETER here: the cluster's shards serve 22 x 32768 (the source
#: run's measured budget, --allow-over-budget); on Modal the identical 720,896-token PROFILING forward died with
#: CUBLAS_STATUS_EXECUTION_FAILED inside the Qwen3.5 layers (canary attempt 3, 2026-09-12 01:13Z — the cluster's
#: H200s ran that exact forward in ~9 min), so the Modal default is the geometry table's own identity at this window,
#: 11 x 32768 = 360,448 batched tokens, which needs no override and halves the profiling footprint. What concurrency
#: changes: throughput, the attested record, AND WHICH DRAWS ARE PRODUCED — seeded sampling on this stack is not
#: batch-invariant (same seed + prompt, 97–100% of re-drawn continuations fork after a shared prefix), so a Modal
#: leg is a RE-RANDOMISATION of its scenarios' draws, not a bias: the per-sequence window (32768), the cap and the
#: ladder reading are identical, and the estimand is a rate over scenarios. The served S rides into every row
#: (rows.py `serving_max_num_seqs`) so item-level provenance names the geometry.
WINDOW = 32768
SEQS_DEFAULT = 11
N_SHARDS_DEFAULT = 13
DUMMY_BEARER = "petri-proxy"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] [misalign-modal] {msg}", flush=True)


def _cells_rows(cells_served: dict, endpoint: dict) -> tuple[list, list]:
    """The runner's contract from the standup's own enumeration — misalign_proxy.sbatch's block, verbatim
    minus the organism overlays."""
    served = set(endpoint["served_models"])
    legs = list(cells_served["canonical"]) + list(cells_served.get("controls") or [])
    rows, full = [], []
    for c in legs:
        vm = c["virtual_model_name"]
        if vm not in served:
            raise RuntimeError(f"cell {c['cell_id']} vmodel {vm} is not served ({sorted(served)})")
        si = c["si_id"] if c["cell_class"] == "prompted" else None
        rows.append({"cell_id": c["cell_id"], "virtual_model": vm, "serving_identity": c["serving_identity"],
                     "intervention_hash": c["intervention_hash"], "si_id": si, "kind": c["kind"]})
        full.append({**c, "si_id_for_runner": si})
    return rows, full



REGISTRY_VOL_ROOT = "/vol/registry"     # the volume's provisioned registry (emit.MODAL_VOL_ROOT); spool = records/, pending = pending/


def _registry_register(tree: str, cfg_rel: str, model_key: str, selection: str, behaviors: str, serving_path: str, sha: str,
                       job: str, env: dict) -> str:
    """In-container self-registration through the checkout's ``emit_cli register --from-config``; returns the pending
    file path when it exists afterwards, '' otherwise (no provisioned registry on the volume, a refusal, a failure)."""
    try:
        if not Path(REGISTRY_VOL_ROOT).is_dir():
            return ""
        pending_dir = f"{REGISTRY_VOL_ROOT}/pending"
        pend = f"{pending_dir}/{job}.json"
        evals = ",".join(f"agentic_{b.strip()}" for b in behaviors.split(",") if b.strip())
        key = model_key if selection == "main" else f"{model_key}_{selection}"
        renv = dict(env); renv.setdefault("SISTEER_REGISTRY_SPOOL", f"{REGISTRY_VOL_ROOT}/records")
        with open(f"{REGISTRY_VOL_ROOT}/pending_{job}.log", "a") as lf:
            subprocess.call([sys.executable, "-m", "sisteer.registry.emit_cli", "register", "--from-config", f"{tree}/{cfg_rel}",
                             "--model-key", key, "--roster-selection", selection, "--evals", evals, "--serving-path", serving_path,
                             "--campaign", "sdfbase_full22", "--job-id", job, "--job-name", job, "--pending-dir", pending_dir,
                             "--code-sha", sha, "--substrate", "modal"], env=renv, stdout=lf, stderr=subprocess.STDOUT, cwd=tree, timeout=600)
        return pend if Path(pend).is_file() else ""
    except Exception as e:  # noqa: BLE001 — a lost record never costs a shard
        _log(f"[registry] self-registration failed ({type(e).__name__}: {e}); the shard continues unrecorded")
        return ""


def _registry_state(tree: str, job: str, state: str, env: dict) -> None:
    """The shard's terminal state for every run it registered (stdlib emitter, ``--final`` push)."""
    try:
        renv = dict(env); renv.setdefault("SISTEER_REGISTRY_SPOOL", f"{REGISTRY_VOL_ROOT}/records")
        subprocess.call([sys.executable, "-S", "-m", "sisteer.registry.emit_cli", "state", "--state", state, "--pending-dir",
                         f"{REGISTRY_VOL_ROOT}/pending", "--job", job, "--scheduler-state", "MODAL_" + state.upper(), "--final"],
                        env=renv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=tree, timeout=300)
    except Exception as e:  # noqa: BLE001
        _log(f"[registry] terminal state {state} not written ({type(e).__name__}: {e})")


@app.function(image=image, volumes={VOL_PATH: VOL}, gpu="H200", timeout=3 * 3600,
              retries=0)
def shard(item: dict, sha: str, root: str = ROOT_DEFAULT, behaviors: str = "murder",
          n_samples: int = 10, cap: int = 16384, seqs: int = SEQS_DEFAULT, over_budget: bool = False,
          serving_path: str = "lens") -> dict:
    """One leg of one target. ``item`` = {"target", "k", "n_shards"} is first: ``shard.map(items, kwargs=...)`` feeds the
    mapped item positionally. ``seqs`` x WINDOW is the served geometry; ``over_budget`` passes the standup's
    --allow-over-budget (needed only when seqs x WINDOW exceeds the geometry table's budget, as the cluster's 22 does)."""
    SEQS = int(seqs)
    if serving_path not in SERVING_PATHS:
        raise ValueError(f"serving_path must be one of {SERVING_PATHS}; got {serving_path!r}")
    k, n_shards = int(item["k"]), int(item["n_shards"])
    tp = target_paths(item["target"], root)
    tag = tp["file_tag"]
    TARGET, MODEL_KEY, OBJECTS, OUT_ROOT, CFG, MANIFEST = (tp["target"], tp["model_key"], tp["objects"], tp["out_root"],
                                                          tp["cfg"], tp["manifest"])
    tree = f"{REPOS}/sisteer-{sha}"
    VOL.reload()   # see every commit made before this container started (a patched view manifest, a sibling's shard files)
    rec = {"app": APP_NAME, "sha": sha[:12], "shard": k, "n_shards": n_shards, "target": TARGET, "root": root,
           "serving_path": serving_path, "file_tag": tag,
           "behaviors": behaviors, "n_samples": n_samples, "cap": cap, "window": WINDOW, "seqs": SEQS,
           "over_budget": bool(over_budget), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not Path(f"{tree}/src/sisteer").is_dir():
        rec["status"] = "NO_TREE"; rec["tree"] = tree; return rec
    if not Path(OBJECTS).is_dir():
        rec["status"] = "NO_OBJECTS"; rec["objects"] = OBJECTS; return rec
    for rel in (CFG, MANIFEST):
        if not Path(f"{tree}/{rel}").is_file():
            rec["status"] = "NO_CONFIG"; rec["missing"] = rel; return rec
    out_root = Path(OUT_ROOT); (out_root / "agentic").mkdir(parents=True, exist_ok=True)
    srv = out_root / f"serving_{tag}shard{k}"; srv.mkdir(parents=True, exist_ok=True)
    stop = srv / "STOP"
    if stop.exists():
        stop.unlink()
    for stale in ("endpoint.json",):
        if (srv / stale).exists():
            (srv / stale).unlink()
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}/src:{tree}/scripts"
    env["TT_TWO_SI_DEFENSES_DIR"] = OBJECTS
    if serving_path == "lora_native":
        env["SISTEER_ADAPTER_BANK_ROOT"] = ADAPTER_BANK_ROOT
        rec["adapter_bank"] = f"{ADAPTER_BANK_ROOT}/{MODEL_KEY}"

    # --- 1. the standup (the sbatch's f22_standup call for the ladder / vendor geometry) -------------------
    t0 = time.time()
    standup_log = (srv / "standup.log").open("a")
    standup = subprocess.Popen(
        [sys.executable, f"{tree}/scripts/sdfbase/mt_serving_up.py", "--repo-dir", tree,
         "--config", f"{tree}/{CFG}", "--manifest", f"{tree}/{MANIFEST}", "--defenses-dir", OBJECTS,
         "--out", str(srv), "--stop-file", str(stop), "--mode", "petri", "--model-key", MODEL_KEY,
         "--max-model-len", str(WINDOW), "--max-num-seqs", str(SEQS),
         *(["--allow-over-budget"] if over_budget else []),
         "--gpu-mem-util", "0.9", "--ready-timeout-s", "2700",
         # TODO(native-serving-default §8.6): this launcher is not a capability / tinycap surface and pins the
         # STACKED lens reduction (`--reduction-mode stacked --require-attested-mode stacked`, read back off
         # `lens_reduction_mode` below), so it names the LENS standup EXPLICITLY (mt_serving_up's own default is
         # the native engine since 2026-09-13, under which those two flags refuse by name and an argv without
         # this flag would stand up the other engine). The native reading it lacks: endpoint.json.engine /
         # /v1/models petri_proxy.engine.kind held to the requested path (tinycap_served.py has it).
         # `serving_path` = "lora_native" serves the residual slot natively (the PSR / ReLU cells);
         # the lens pins ride the argv ONLY under lens (mt_serving_up refuses them by name on the native engine).
         *(["--serving-path", "lens", "--reduction-mode", "stacked", "--require-attested-mode", "stacked"]
           if serving_path == "lens" else ["--serving-path", "lora_native", "--adapter-bank", f"{ADAPTER_BANK_ROOT}/{MODEL_KEY}"]),
         # a target that pins its object's digest explicitly: held at the standup and recorded in endpoint.json
         *[a for c, d in sorted(tp["expect_digests"].items()) for a in ("--expect-object-sha256", f"{c}={d}")]],
        env=env, stdout=standup_log, stderr=subprocess.STDOUT, cwd=tree)
    ep = srv / "endpoint.json"
    for _ in range(600):
        if ep.exists() or standup.poll() is not None:
            break
        time.sleep(5)

    def _tail(p: Path, n: int = 4000) -> str:
        return p.read_text(errors="replace")[-n:] if p.exists() else ""

    if not ep.exists():
        rec["status"] = "STANDUP_FAILED"; rec["standup_tail"] = _tail(srv / "standup.log")
        VOL.commit(); return rec
    rec["standup_s"] = round(time.time() - t0, 1)
    endpoint = json.loads(ep.read_text())
    (out_root / f"endpoint.snapshot_{tag}shard{k}.json").write_text(json.dumps(endpoint, indent=1, default=str))
    # the geometry this run is specified at, read back from the attested record — never assumed. The ENGINE is read the
    # way tinycap_served.endpoint_engine reads it: `engine` (a block spells it as `kind`), else the row-facing
    # `serving_engine`, else lens BY ERA (every lens standup before the native landing wrote no engine key).
    att_l, att_s = endpoint.get("attested_max_model_len"), endpoint.get("attested_max_num_seqs")
    eng = endpoint.get("engine")
    eng = eng.get("kind") if isinstance(eng, dict) else eng
    eng = eng or endpoint.get("serving_engine") or "lens"
    rec["attested"] = {"max_model_len": att_l, "max_num_seqs": att_s, "engine": eng,
                       "lens_reduction_mode": endpoint.get("lens_reduction_mode")}
    if int(endpoint["max_model_len"]) != WINDOW:
        rec["status"] = "GEOMETRY_MISMATCH"; stop.touch(); VOL.commit(); return rec
    if serving_path == "lens" and (eng != "lens" or endpoint.get("lens_reduction_mode") != "stacked"):
        rec["status"] = "GEOMETRY_MISMATCH"; stop.touch(); VOL.commit(); return rec
    if serving_path == "lora_native" and eng != "lora_native":
        rec["status"] = "GEOMETRY_MISMATCH"; stop.touch(); VOL.commit(); return rec
    # the runner's concurrency = the ATTESTED batch (the hybrid's full cudagraph capture caps the native batch at 8
    # whatever was requested); the requested S stays on the record
    conc = int(att_s or SEQS); rec["concurrency"] = conc
    base = endpoint["proxy_url"].rstrip("/")
    rec["proxy_url"] = base

    # --- 2. cells.json from the standup's enumeration ----------------------------------------------------
    cells_served = json.loads((srv / "cells_served.json").read_text())
    rows, full = _cells_rows(cells_served, endpoint)
    # the `_bl` panels' PINS ride the rows as on the cluster (misalign_proxy.sbatch F22_BASELINE_PINS:
    # selection_outcome on every row); None outside the baselines panels
    bl_pins = baseline_pins(tree, tp["family"]) if tp["baselines"] else None
    if bl_pins:
        for r in full:
            if r["cell_id"] in bl_pins:
                r["baseline_pin"] = bl_pins[r["cell_id"]]
    cells_file = out_root / f"cells_{tag}shard{k}.json"
    # the object form: the runner records every non-`cells` key in each leg's meta.json (`run`), and rows.py
    # lifts `serving` into every emitted row — the attested geometry travels with the items it produced
    cells_file.write_text(json.dumps({"cells": rows, "serving": {
        "substrate": "modal", "shard": k, "n_shards": n_shards, "baseline_pins": bl_pins, "root": root,
        "serving_path": serving_path, "engine": eng, "file_tag": tag,
        "attested_max_num_seqs": att_s, "attested_max_model_len": att_l,
        "max_num_seqs": endpoint.get("max_num_seqs"), "max_model_len": endpoint.get("max_model_len"),
        "node": endpoint.get("node"), "app": APP_NAME}}, indent=2))
    (out_root / f"cells_full_{tag}shard{k}.json").write_text(json.dumps(
        {"target": TARGET, "model_key": MODEL_KEY, "wave_view": tp["view"], "wave_sha": sha, "substrate": "modal",
         "root": root, "baselines": tp["baselines"], "baseline_pins": bl_pins, "serving_path": serving_path, "engine": eng,
         "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "endpoint": {kk: endpoint.get(kk) for kk in ("model", "model_revision", "model_key", "proxy_url", "max_model_len",
                                                      "max_num_seqs", "attested_max_model_len", "attested_max_num_seqs",
                                                      "lens_reduction_mode", "thinking_mode", "served_models", "cells",
                                                      "harness_cells", "bank_objects", "geometry")},
         "cells_served": {kk: cells_served.get(kk) for kk in ("manifest", "config", "defenses_dir", "reduction_mode")},
         "legs": full}, indent=2, default=str))
    ids = sorted(r["cell_id"] for r in rows)
    mine = ids[k::n_shards]
    rec["legs_served"] = ids; rec["legs_run"] = mine
    if not mine:
        rec["status"] = "NO_LEG_FOR_SHARD"; stop.touch(); VOL.commit(); return rec
    _log(f"shard {k}/{n_shards}: {len(ids)} legs served; running {mine}")
    VOL.commit()

    # --- 3. the runner, one leg, the vendor sampler; the volume committed every five minutes --------------
    stop_commits = threading.Event()

    def _committer():
        while not stop_commits.wait(300):
            try:
                VOL.commit()
            except Exception as e:  # a failed commit is retried next tick, never fatal
                _log(f"volume commit failed (will retry): {e}")
    threading.Thread(target=_committer, daemon=True).start()
    renv = dict(env); renv["CUDA_VISIBLE_DEVICES"] = ""
    # results-registry self-registration (2026-09-17, the same mechanism as the cluster jobs' f22_registry_register): the
    # CONTAINER registers this shard's runs itself — one per (agentic behaviour eval x object of $CFG's explicit_cells)
    # through the checkout's emit_cli, against the volume's spool (/vol/registry, provisioned; absent → nothing) — and
    # hands the pending identity to the runner (its generation record) and to misalign.rows' store later. The
    # execution key is this shard's (`modal:<app>:<tag>shard<k>`); a failure is one log line, never a failed shard.
    reg_job = f"modal-{APP_NAME}-{tag}shard{k}"
    reg = _registry_register(tree, CFG, MODEL_KEY, tp["view"], behaviors, serving_path, sha, reg_job, env)
    if reg:
        renv["SISTEER_REGISTRY_RUN_IDENTITY"] = reg
        rec["registry_pending"] = reg
    runner_log = out_root / "agentic" / f"runner_modal_{tag}shard{k}.log"
    t1 = time.time()
    with runner_log.open("a") as lf:
        rc = subprocess.call(
            [sys.executable, "-m", "sisteer.capeval.misalign.runner", "--proxy-url", base,
             "--cells", str(cells_file), "--family", "agentic", "--out", str(out_root / "agentic"),
             "--concurrency", str(conc), "--n", str(n_samples), "--behaviors", *behaviors.split(","),
             "--max-tokens", str(cap), "--sampler", "vendor", "--only-cells", ",".join(mine),
             "--api-key", DUMMY_BEARER],
            env=renv, stdout=lf, stderr=subprocess.STDOUT, cwd=tree)
    stop_commits.set()
    rec["runner_rc"] = rc; rec["runner_s"] = round(time.time() - t1, 1)
    if reg:
        _registry_state(tree, reg_job, "generated" if rc == 0 else "failed", env)
    n_rows = 0
    for cid in mine:
        p = out_root / "agentic" / cid / "responses.jsonl"
        n_rows += sum(1 for ln in p.read_text().splitlines() if ln.strip()) if p.exists() else 0
    rec["responses"] = n_rows
    rec["runner_tail"] = _tail(runner_log, 2500)

    # --- 4. teardown + record -----------------------------------------------------------------------------
    stop.touch()
    try:
        standup.wait(timeout=300)
    except subprocess.TimeoutExpired:
        standup.kill()
    standup_log.close()
    for name in ("vllm_server.log",):
        p = srv / name
        if p.exists() and p.stat().st_size > 5_000_000:  # keep the tail of a huge server log
            p.write_text(p.read_text(errors="replace")[-5_000_000:])
    rec["status"] = "OK" if rc == 0 and n_rows > 0 else "RUNNER_FAILED"
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (out_root / "RESULT").open("a") as f:
        f.write(f"{rec['finished']} substrate=modal target={TARGET} root={root} serving_path={serving_path} shard={tag}{k}/{n_shards} legs={','.join(mine)} sha={sha[:12]} "
                f"rc={rc} responses={n_rows} standup_s={rec['standup_s']} runner_s={rec['runner_s']}\n")
    (srv / "record.json").write_text(json.dumps(rec, indent=1, default=str))
    VOL.commit()
    return rec


@app.local_entrypoint()
def main(sha: str = "", plan: str = "", root: str = ROOT_DEFAULT, behaviors: str = "murder", n_samples: int = 10,
         cap: int = 16384, seqs: int = SEQS_DEFAULT, over_budget: bool = False, serving_path: str = "lens"):
    if not sha:
        raise SystemExit("pass --sha <tree sha staged at /vol/capctrl/repos/sisteer-<sha>>")
    if not plan:
        raise SystemExit('pass --plan "target:n_shards:shards[;...]", e.g. "qwen35_4b:13:0-12;qwen35_4b_it:8:0-7;qwen35_4b_bl:4:0-3"')
    target_paths(next(iter(TARGETS)), root)   # refuses a root outside misalign_vendor* before any container starts
    if serving_path not in SERVING_PATHS:
        raise SystemExit(f"--serving-path must be one of {SERVING_PATHS}; got {serving_path!r}")
    items = parse_plan(plan)
    print(f"[misalign-modal] {APP_NAME}: sha {sha[:12]} root {root} serving_path {serving_path} behaviours {behaviors} N={n_samples} cap={cap} "
          f"geometry {WINDOW}x{seqs}{' (--allow-over-budget)' if over_budget else ''}; {len(items)} item(s): "
          + " ".join(f"{it['target']}:{it['k']}/{it['n_shards']}" for it in items), flush=True)
    results = list(shard.map(items, kwargs=dict(sha=sha, root=root, behaviors=behaviors, n_samples=n_samples, cap=cap,
                                                seqs=seqs, over_budget=over_budget, serving_path=serving_path)))
    bad = 0
    for r in results:
        print(json.dumps({kk: r.get(kk) for kk in ("target", "shard", "status", "legs_run", "responses", "standup_s",
                                                    "runner_s", "runner_rc", "attested")}, default=str), flush=True)
        if r.get("status") != "OK":
            bad += 1
            print("  ---", r.get("standup_tail") or r.get("runner_tail") or r.get("missing") or "", flush=True)
    if bad:
        raise SystemExit(f"{bad} shard(s) not OK")
