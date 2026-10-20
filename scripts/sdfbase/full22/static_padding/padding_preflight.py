#!/usr/bin/env python
"""The PADDED STATIC leg's preflight — run by target_static_padded.sbatch BEFORE the standup, so a run that cannot
work dies in seconds with the numbers, never after the GPU is taken (docs/static_padding.md).

Five checks, every one loud, none skipped silently:

  1. THE FILLER. --filler must be a file whose sha256 (and size, when given) is the corpus this leg was designed
     on: the PG-19 cache scripts/fetch_stage2_data.py --only pg19-filler builds — sha256 PG19_FILLER_SHA256 below,
     2,881,719 bytes, the same bytes on both clusters (checked 2026-09-18; re-verified on cluster B 2026-09-22).
     It must also parse through the harness's own loader (static_padding.load_filler: a JSON object with a `text`
     key). The recorded filler_sha256 on every padded row is what pins the corpus; this makes sure it is THAT one.
  2. THE MODEL. Every config's `model` @ `model_revision` must be a snapshot in the HF cache this job reads
     (HF_HOME, set by hf_env.sh through f22_init; --hf-home overrides) with the files a standup and a tokenizer-only
     load need — config.json, tokenizer.json, and at least one *.safetensors shard. Cluster B carries no HF token
     on purpose, so an absent snapshot cannot be fetched inside the job.
  3. THE CONTRACT. Each config is a generated padded-static config (make_padding_configs.pad_of names its level),
     its `attacks.static_padding.tokens` equals the level in its name (or the p0 control carries no block), and the
     HARNESS's rule holds at the window: tokens + attacks.static_max_new_tokens + reserve <= --window
     (static_padding.window_problem — the very check jailbreak_run runs per driver, so a driver can never be the
     first to find out). And its `attacks.static_max_new_tokens` is the static ladder's CEILING
     (make_padding_configs.STATIC_CAP = 16,384; check_cap) at EVERY level, the p0 control included: this leg is
     prefill-bound and the rung driver refuses padded stores (escalate_static.assert_base_unpadded), so a cap below
     the ceiling would leave a >1 % at-cap share nowhere to go but a whole-level regeneration — refused here, before
     any GPU minute, naming why, so the cap is never quietly "optimised" back down to the campaign's 4,096.
  4. THE RENDERED NEED. The harness's reserve stands in for system instruction + payload + template; this derives
     the REAL number: for every cell of every config, the pinned tokenizer's render of [system = the cell's SI for a
     prompted__<si> baseline; NONE for the bare anchor and for every compiled cell, which the driver serves object-only
     (si_text_of)] + EVERY variant of EVERY goal of each family over the 259-goal confirmatory pool (the same
     builders and the same render counter the wave's window check uses, prompted_window_check.rendered_tokens), taking
     the max IN TOKENS — never the char-longest proxy, which under-reads by a few tokens (grok420 under the constitution
     SI: proxy 6,403, true max 6,412, measured on cluster B 2026-09-22) — plus the pad at its worst case (tokens x
     (1 + tolerance): PadPlan converges within the tolerance either side), plus the cap, must fit the window. A
     refusal names the worst row and the smallest window (a multiple of --align) that would carry every row. This is
     what catches the constitution SI: it renders to 5,973 tokens on Qwen3.5-4B, so at 65,536 padding tokens + the
     16,384 ceiling cap the worst row needs 6,412 + 66,847 + 16,384 = 89,643 — the 4,096-cap sizing's 81,920 window is
     7,723 short, the harness rule (tokens + cap + reserve 8,192) says 90,112, which is also the smallest 1,024-aligned
     window carrying the render, and the wave stands up at 98,304 (the rule: raise the window,
     never shrink the pad; 8,661 tokens of headroom over the render and one reserve, 8,192, over the harness rule, at
     the SAME derived S as 90,112 — the kernel ceiling holds S = 5 from 90,112 to 104,448 — and at the SAME 491,520
     batched tokens the 4,096-cap sizing's 6 x 81,920 allocated, so the memory sum of check 5 is unchanged).
  5. THE GEOMETRY. The S this window buys, through the launcher's OWN arithmetic (never a literal): identity S = the
     model's measured batched-token budget // L (geometry.ModelBudget.seqs_at, the hybrid cap applied), then the LoRA
     kernels' int32 index ceiling (geometry.kernel_ceiling_adjustment at the model's pinned widest module) — which a
     NATIVE standup applies whether or not a steered cell is served, because it ALWAYS registers the zero adapter
     (native_standup.bring_up_native: export_or_reuse_zero -> all_blocks -> lora_kernel_ceiling_geometry), so the S
     recorded here is the S a STEERED standup at the same window resolves too (qwen35_4b at 98,304: identity
     741,888 // 98,304 = 7, ceiling 524,287 // 98,304 = 5 -> S 5). Then the memory sum at that geometry against the
     card: weights (the snapshot's safetensors index) + full-coverage KV at the bytes ONE context token holds per the
     config's OWN layer roster (`layer_types`: only the attention layers hold a KV that grows with context; Qwen3.5-4B
     is 8 full-attention + 24 gated-deltanet layers -> 2 x 8 x 4 kv heads x 256 head_dim x 2 B = 32 KiB/token, a
     QUARTER of the 128 KiB a 32-layer count gives) + the warmup MLP intermediate (S x L x the per-token intermediate
     width x 2 B — a dense block's intermediate_size; an MoE block's active experts x moe_intermediate + the shared
     expert's; from the verified dims table the serving probe plans from
     (lora_modules.ARCH_DIMS_BY_MODEL_KEY), the config read by architecture behind it, refused by name when neither
     yields it) must fit the utilization's reservation (0.9 x 139.81 GiB). Refuses when no row fits the
     ceiling or the sum does not fit; records every number either way, with the certified per-set module widths.

Exit 0 = every check passed; 3 = refused (the problems are printed and written to --out); 2 = a usage problem.
Read-only: it writes nothing but --out.

  python scripts/sdfbase/full22/static_padding/padding_preflight.py --window 98304 --model-key qwen35_4b \\
      --config configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padstatic_p0.yaml \\
      --config configs/sdfbase/static_padding/sdfbase_full22_qwen35_4b.padstatic_p65536.yaml \\
      --filler $CLUSTER_REPO_MIRROR/data/pg19_filler.json --families pasttense,codeattack,renellm,grok420 \\
      --out preflight.json
  --skip-render skips check 4 (a smoke of the cheap checks only; recorded in the JSON, never silent).
  --model-key defaults to the key the config names carry (sdfbase_full22_<key>.padstatic_p<pad>.yaml).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase" / "full22", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

#: the PG-19 filler corpus this leg pads from (docs/static_padding.md "Where the filler has to be"): the sha256 and
#: byte count the doc records
PG19_FILLER_SHA256 = "7351d518dc71c53bfc0637da825db82e1d7dc4c04e9acd0e8f015ea8d7826b97"
PG19_FILLER_BYTES = 2_881_719
STATIC_FAMILIES = ("pasttense", "codeattack", "renellm", "grok420")
EXIT_REFUSED = 3
EXIT_USAGE = 2
#: the card the standup reserves on, as vLLM reports it (geometry's certified basis note: "on a 139.81 GiB card")
H200_GIB = 139.81
#: the standup's default gpu_memory_utilization (common.sh f22_standup: GPU_MEM_UTIL unset = 0.9)
DEFAULT_GPU_MEM_UTIL = 0.9
#: bytes per KV element and per MLP intermediate element on the served bf16 engine
BF16_BYTES = 2
#: the layer type that holds a fixed-size recurrent state instead of a KV that grows with context
#: (vllm_server.hybrid_recurrent_model's read of `layer_types`)
RECURRENT_LAYER_TYPE = "linear_attention"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def align_up(n: int, align: int) -> int:
    """The smallest multiple of ``align`` that is >= ``n`` (the windows this lineage stands up at are multiples of 1,024)."""
    if align < 1:
        raise ValueError(f"align must be >= 1, got {align}")
    return int(math.ceil(int(n) / int(align))) * int(align)


def pad_worst_case(tokens: int, tolerance: float) -> int:
    """The most padding the plan may actually send: the target plus the tolerance's overshoot (PadPlan.apply accepts
    |actual - target| <= tolerance x target, so the served prompt can be up to this much longer than the payload)."""
    return int(tokens) + int(math.ceil(int(tokens) * float(tolerance)))


def rendered_need(prompt_tokens: int, *, pad_tokens: int, tolerance: float, cap: int) -> int:
    """Tokens a request may occupy at most: the rendered [system + payload] prompt, the worst-case pad, the cap."""
    return int(prompt_tokens) + pad_worst_case(pad_tokens, tolerance) + int(cap)


def contract_of(cfg: dict, path: Path) -> dict:
    """The padding contract a generated config pins, read the way the harness reads it (jailbreak_run's defaults for
    the keys the block leaves unstated): {tokens, placement, mode, tolerance, reserve, cap, level}. Refuses a block
    whose `tokens` disagrees with the level in the file name, and a p0 file that carries a block."""
    from sisteer.eval.attacks import static_padding as sp
    import make_padding_configs as gen

    level = gen.pad_of(path)
    block = (cfg.get("attacks") or {}).get("static_padding")
    if level == 0:
        if block:
            raise ValueError(f"{path.name}: the p0 CONTROL carries an attacks.static_padding block {block} — 0 = off, no block")
        block = {}
    elif not isinstance(block, dict) or block.get("tokens") != level:
        raise ValueError(f"{path.name}: attacks.static_padding.tokens = {(block or {}).get('tokens')!r} but the file name pins "
                         f"level {level} — the config, not a flag, carries the level; regenerate (make_padding_configs.py)")
    return {"level": level,
            "tokens": int(block.get("tokens", 0)),
            "placement": block.get("placement", sp.PLACEMENTS[0]),
            "mode": block.get("mode", sp.MODES[0]),
            "tolerance": float(block.get("tolerance", sp.DEFAULT_TOLERANCE)),
            "reserve": int(block.get("reserve", sp.DEFAULT_RESERVE_TOKENS)),
            "cap": int((cfg.get("attacks") or {})["static_max_new_tokens"])}


def check_filler(path: Path, *, want_sha: str, want_bytes: int | None) -> tuple[dict, list[str]]:
    from sisteer.eval.attacks import static_padding as sp

    rec: dict = {"path": str(path)}
    problems: list[str] = []
    if not path.is_file():
        return rec, [f"filler: {path} is not a file (expected at $SISTEER_PROJECT_ROOT/data/pg19_filler.json; build with "
                     "scripts/fetch_stage2_data.py --only pg19-filler)"]
    rec["bytes"] = path.stat().st_size
    rec["sha256"] = sha256_file(path)
    if rec["sha256"] != want_sha:
        problems.append(f"filler: sha256 {rec['sha256'][:16]}… != the corpus this leg pads from {want_sha[:16]}… — a different "
                        "filler is a different measurement; refusing")
    if want_bytes is not None and rec["bytes"] != want_bytes:
        problems.append(f"filler: {rec['bytes']:,} bytes != the expected {want_bytes:,}")
    try:
        text, sha = sp.load_filler(path)
        rec["chars"] = len(text)
        if sha != rec["sha256"]:
            problems.append("filler: load_filler's sha disagrees with the file hash (a torn read?)")
    except sp.StaticPaddingError as e:
        problems.append(f"filler: the harness's loader refuses it: {e}")
    return rec, problems


def check_model(model: str, revision: str, hf_home: Path) -> tuple[dict, list[str]]:
    """The snapshot a standup (weights) and a tokenizer-only load (the padded drivers, this preflight's render) need.
    ``rec["tokenizer_ok"]`` says whether the render can run at all: a tokenizer-only cache (the grading host) still renders,
    and the missing weights are reported as their own problem."""
    snap = hf_home / "hub" / f"models--{model.replace('/', '--')}" / "snapshots" / revision
    rec = {"model": model, "revision": revision, "snapshot": str(snap), "tokenizer_ok": False}
    problems: list[str] = []
    if not snap.is_dir():
        return rec, [f"model: no snapshot {snap} — the pinned {model}@{revision[:12]} is not in this cluster's HF cache "
                     "(no token on cluster B: pre-fetch it, never inside the job)"]
    present = {p.name for p in snap.iterdir()}
    rec["files"] = sorted(present)
    for need in ("config.json", "tokenizer.json"):
        if need not in present:
            problems.append(f"model: snapshot {snap} lacks {need}")
    rec["tokenizer_ok"] = {"config.json", "tokenizer.json"} <= present
    if not any(n.endswith(".safetensors") for n in present):
        problems.append(f"model: snapshot {snap} holds no *.safetensors shard (a tokenizer-only cache serves nothing)")
    return rec, problems


def check_contract(contract: dict, window: int) -> list[str]:
    """The HARNESS's rule (static_padding.window_problem), so a driver is never the first to find out."""
    from sisteer.eval.attacks import static_padding as sp

    if contract["tokens"] == 0:
        return []
    problem = sp.window_problem(pad_tokens=contract["tokens"], cap=contract["cap"], window=window, reserve=contract["reserve"])
    return [f"contract p{contract['level']}: {problem}"] if problem else []


def check_cap(contract: dict) -> list[str]:
    """The cap is the static ladder's CEILING (make_padding_configs.STATIC_CAP; docs/static_padding.md "The cap is the
    ladder's ceiling") at EVERY level, the p0 control included: this leg is prefill-bound, an escalation rung would re-pay
    the whole pad prefill for every re-drawn item, and the rung driver refuses a padded base outright
    (escalate_static.assert_base_unpadded) — so a padded level below the ceiling has no rung to climb to, and a >1 % at-cap
    share would mean regenerating the whole level. Refused before any GPU minute, naming why, so the cap is never quietly
    "optimised" back down by someone who sees 16,384 as waste."""
    import make_padding_configs as gen

    if int(contract["cap"]) == int(gen.STATIC_CAP):
        return []
    return [f"contract p{contract['level']}: attacks.static_max_new_tokens {int(contract['cap']):,} != the static ladder's "
            f"ceiling {gen.STATIC_CAP:,} (make_padding_configs.STATIC_CAP) — this leg runs ONCE at the ceiling: it is prefill-bound, "
            "an escalation rung would re-pay the whole pad prefill, and the rung driver refuses padded stores, so a smaller cap "
            "leaves a >1 % at-cap share nowhere to go but a whole-level regeneration; regenerate (make_padding_configs.py)"]


def kv_bytes_per_token(model_cfg: dict) -> dict:
    """The KV bytes ONE context token holds, from the config's OWN layer roster — never ``num_hidden_layers`` x a
    per-layer figure. Only attention layers hold a KV that grows with context; a hybrid's ``linear_attention`` layers
    (Qwen3.5's gated deltanet, NemotronH's Mamba2) hold a fixed-size per-slot state (vllm_server.hybrid_recurrent_model
    reads the same key). A config without ``layer_types`` is dense: every layer counts. Sliding-window attention
    (gemma) is KV-backed and counted in full — an upper bound. ``2 (K, V) x kv layers x num_key_value_heads x head_dim
    x 2 B``: Qwen3.5-4B = 2 x 8 x 4 x 256 x 2 = 32 KiB."""
    txt = model_cfg.get("text_config") or model_cfg
    n_layers = int(txt["num_hidden_layers"])
    layer_types = list(txt.get("layer_types") or [])
    if layer_types and len(layer_types) != n_layers:
        raise ValueError(f"layer_types names {len(layer_types)} layers but num_hidden_layers is {n_layers}")
    kv_layers = sum(1 for t in layer_types if t != RECURRENT_LAYER_TYPE) if layer_types else n_layers
    heads = int(txt["num_attention_heads"])
    kv_heads = int(txt.get("num_key_value_heads") or heads)
    head_dim = int(txt.get("head_dim") or int(txt["hidden_size"]) // heads)
    return {"num_hidden_layers": n_layers, "kv_layers": kv_layers,
            "recurrent_layers": n_layers - kv_layers, "layer_types_present": bool(layer_types),
            "num_key_value_heads": kv_heads, "head_dim": head_dim,
            "bytes_per_token": 2 * kv_layers * kv_heads * head_dim * BF16_BYTES}


def mlp_intermediate_width(model_key: str, model_cfg: dict | None) -> dict:
    """The per-token MLP-intermediate width (elements) the warmup term multiplies, with its source — FIRST the verified
    dims table the serving probe plans from (``lora_modules.ARCH_DIMS_BY_MODEL_KEY`` through
    ``standup_manifest.mlp_intermediate_elements``: the numbers the probe's plan records and its fold's per-set D derives from),
    THEN the snapshot's config read BY ARCHITECTURE — an MoE (``num_experts`` / ``num_local_experts`` / ``n_routed_experts`` or
    ``moe_intermediate_size`` present): ``num_experts_per_tok`` x the expert intermediate (``moe_intermediate_size``, else
    Mixtral's ``intermediate_size``) + the shared expert's (``shared_expert_intermediate_size``, else ``n_shared_experts`` x the
    expert intermediate), and the wider of that and ``intermediate_size`` when the config names dense blocks too
    (``mlp_only_layers`` / ``first_k_dense_replace``); a dense model: ``intermediate_size``. Never a KeyError: a config that
    yields no width, or a table entry the config disagrees with (another checkpoint, a stale table), is a ``problem`` named
    here and refused by the caller. The five padding units of pass #3 died at 45 s on the dense key ``intermediate_size``
    indexed unguarded against an MoE checkpoint's config, which carries ``moe_intermediate_size`` and no dense key (2026-09-30).
    ``{elements, source, table, config, problem}``."""
    from sisteer.capeval.serving import standup_manifest as sm
    from sisteer.extraction import lora_modules as lm

    out: dict = {"elements": None, "source": None, "table": None, "config": None, "problem": None}
    dims = lm.arch_dims_for(model_key)
    if dims is not None:
        t = sm.mlp_intermediate_elements(dims)
        out["table"] = {"model_key": model_key, **t}
    txt = (model_cfg.get("text_config") or model_cfg) if isinstance(model_cfg, dict) else None
    cfg: dict = {"elements": None, "kind": None, "terms": {}, "why": None}
    if isinstance(txt, dict):
        n_experts = next((txt.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts") if txt.get(k) is not None), None)
        moe_I = txt.get("moe_intermediate_size")
        dense_I = txt.get("intermediate_size")
        if n_experts is not None or moe_I is not None:
            cfg["kind"] = "moe"
            top_k = txt.get("num_experts_per_tok")
            expert_I = moe_I if moe_I is not None else dense_I               # Mixtral names the expert intermediate `intermediate_size`
            shared_I = txt.get("shared_expert_intermediate_size")
            if shared_I is None and txt.get("n_shared_experts") and expert_I is not None:
                shared_I = int(txt["n_shared_experts"]) * int(expert_I)      # DeepSeek's shared experts are expert-wide
            dense_blocks = bool(txt.get("mlp_only_layers")) or int(txt.get("first_k_dense_replace") or 0) > 0
            cfg["terms"] = {"num_experts": n_experts, "num_experts_per_tok": top_k, "expert_intermediate": expert_I,
                            "shared_intermediate": int(shared_I or 0), "dense_blocks": dense_blocks, "dense_intermediate": dense_I}
            if top_k is None or expert_I is None:
                cfg["why"] = (f"an MoE config ({n_experts} experts) without " + " / ".join(k for k, v in (("num_experts_per_tok", top_k), ("moe_intermediate_size (or intermediate_size)", expert_I)) if v is None))
            else:
                moe_block = int(top_k) * int(expert_I) + int(shared_I or 0)
                widths = [moe_block] + ([int(dense_I)] if dense_blocks and dense_I is not None else [])
                cfg["elements"] = max(widths)
        elif dense_I is not None:
            cfg["kind"] = "dense"
            cfg["terms"] = {"intermediate_size": dense_I}
            cfg["elements"] = int(dense_I)
        else:
            cfg["why"] = ("neither an MoE key (num_experts / moe_intermediate_size) nor intermediate_size is present; keys: "
                          + ", ".join(sorted(k for k in txt if "size" in k or "expert" in k or "intermediate" in k)[:12]))
    else:
        cfg["why"] = "no config to read"
    out["config"] = cfg
    table_w = (out["table"] or {}).get("elements")
    if table_w is not None and cfg["elements"] is not None and int(table_w) != int(cfg["elements"]):
        out["problem"] = (f"the verified dims table for {model_key} gives a per-token MLP intermediate of {table_w:,} elements "
                          f"({out['table']['terms']}) but the snapshot's config gives {cfg['elements']:,} ({cfg['kind']}: {cfg['terms']}) — "
                          "another checkpoint than the table was verified on, or a stale table; refusing until they agree")
        return out
    if table_w is not None:
        out["elements"], out["source"] = int(table_w), f"lora_modules.ARCH_DIMS_BY_MODEL_KEY[{model_key!r}]"
    elif cfg["elements"] is not None:
        out["elements"], out["source"] = int(cfg["elements"]), f"config:{cfg['kind']}"
    else:
        out["problem"] = (f"no per-token MLP intermediate width for {model_key}: no verified dims entry (lora_modules.ARCH_DIMS_BY_MODEL_KEY "
                          f"knows {sorted(lm.ARCH_DIMS_BY_MODEL_KEY)}) and the snapshot's config yields none ({cfg['why']}) — the warmup "
                          "term cannot be sized; add the model's ArchDims to the table or name its architecture's keys here")
    return out


def weights_bytes(snapshot: Path) -> int | None:
    """The served weights' size: the safetensors index's ``metadata.total_size`` when the snapshot carries one, else the
    sum of the shards on disk; ``None`` when neither exists (a tokenizer-only cache)."""
    idx = snapshot / "model.safetensors.index.json"
    if idx.is_file():
        try:
            total = (json.loads(idx.read_text(encoding="utf-8")).get("metadata") or {}).get("total_size")
            if isinstance(total, int) and total > 0:
                return total
        except (OSError, ValueError):
            pass
    shards = [p.stat().st_size for p in snapshot.iterdir() if p.name.endswith(".safetensors")]
    return sum(shards) if shards else None


#: the ONE source of the SELECTED LoRA module set: reproduce/models.yaml's per-model
#: `lora_architecture` — the promoted HPO pick (scripts/sdfbase/hpo_pick.py update_models_yaml), committed in the repo and
#: therefore in EVERY mirrored checkout a padded job runs from (steered AND prompted-only), and the same table the generated
#: fits config and the trainer's fit_selection.json sidecar derive from. `SISTEER_MODELS_TABLE` (reproduce/lib/models.py's
#: existing seam) points at another copy of the same table; never a second artefact.
MODELS_TABLE_ENV = "SISTEER_MODELS_TABLE"
MODELS_TABLE_REL = "reproduce/models.yaml"


def models_table_path() -> Path:
    return Path(os.environ.get(MODELS_TABLE_ENV) or (REPO / MODELS_TABLE_REL))


def selected_module_set(model_key: str, models_path: "Path | str | None" = None) -> "tuple[str | None, dict]":
    """``(module_set, info)`` — the SELECTED LoRA module set of ``model_key`` from the models table's promoted
    ``lora_architecture`` (``set_id_for(modules)``: wo / qkvo / attn_mlp / mlp / down), with ``info`` naming the table,
    the modules, the rank, ``selected_by`` and the ruling id. ``(None, info)`` when the model is not in the table (the
    paper's sweep-only keys: the built-in geometry tables stand). REFUSES (``ValueError``) a model whose pick is
    ``pending_user_pick`` / has no modules — steered padding cannot size before the pick — and modules that are not one of
    the five module sets (the probe certifies sets by id)."""
    sys.path.insert(0, str(REPO / "reproduce" / "lib"))
    import models as _models  # reproduce/lib/models.py: the one reader of the models table
    # the vocabulary's stdlib leaf (lora_modules re-exports it): the post-standup geometry pin runs this selection inside the
    # suite's CPU/stdlib venv, which has no transformers (lora_modules -> ..models)
    from sisteer.extraction.lora_module_sets import MODULE_SETS, set_id_for
    path = Path(models_path) if models_path else models_table_path()
    try:
        table = _models.load_models(str(path))
    except (OSError, ValueError) as e:
        raise ValueError(f"{model_key}: the models table {path} cannot be read ({e})") from None
    info = {"source": str(path)}
    if model_key not in table:
        return None, {**info, "reason": f"{model_key!r} is not a row of {path}; the built-in geometry tables stand"}
    la = (table[model_key] or {}).get("lora_architecture") or {}
    info.update({"selected_by": la.get("selected_by"), "selection_ruling_id": la.get("selection_ruling_id"),
                 "rank": la.get("rank"), "modules": la.get("modules")})
    mods = la.get("modules")
    if la.get("selected_by") == "pending_user_pick" or not mods:
        raise ValueError(f"{model_key}: no promoted LoRA architecture pick in {path} (lora_architecture.selected_by="
                         f"{la.get('selected_by')!r}, modules={mods!r}) — steered padding sizes from the SELECTED module set's "
                         "certified geometry, and there is no selection yet; run reproduce/03_hpo.sh --phase pick first")
    sid = set_id_for(mods)
    if sid not in MODULE_SETS:
        raise ValueError(f"{model_key}: lora_architecture.modules {list(mods)!r} in {path} is not one of the module sets the serving "
                         f"probe certifies ({sorted(MODULE_SETS)}; got {sid!r})")
    return sid, {**info, "module_set": sid}


def unsteered_module_set(model_key: str, models_path: "Path | str | None" = None) -> "tuple[None, dict]":
    """The selection an UNSTEERED padded standup makes: NONE. It registers only the zero adapter ``none__lora0`` (the o_proj class —
    native_standup's zero-class rule), so no promoted pick is needed and the resolver's unsteered default sizes it
    (``geometry.UNSTEERED_MODULE_SET``: the site overlay's ``wo`` row when the model has an overlay; the built-in tables otherwise —
    refused below when provisional). A promoted pick, when one exists, is REPORTED in the info but never drives the size: the
    unsteered standup's class is the zero adapter's, whatever the steered design is. (2026-09-30: the unsteered preflight demanded
    the pick and refused u0 padding on a pending one — job 807664.)"""
    from sisteer.capeval.serving import geometry as g
    info: dict = {"selected_by": "unsteered", "module_set": None, "unsteered_default": g.UNSTEERED_MODULE_SET,
                  "reason": "an unsteered padded standup registers only the zero adapter (o_proj class): no pick needed — the resolver's "
                            f"unsteered default ({g.UNSTEERED_MODULE_SET!r}, the record's row for that class) sizes it"}
    try:
        sid, sel = selected_module_set(model_key, models_path)
        info["promoted_pick"] = {"module_set": sid, "rank": sel.get("rank"), "selected_by": sel.get("selected_by")}
    except ValueError as e:
        info["promoted_pick"] = None
        info["pick_note"] = str(e)[:300]
    return None, info


def geometry_expectation(model_key: str, window: int, *, module_set: "str | None" = None, steered: bool = False) -> dict:
    """The S the launcher's OWN arithmetic derives at ``window``: identity S = the model's measured batched-token
    budget // L (geometry.ModelBudget.seqs_at — the hybrid per-slot cap applied, as resolve_geometry(model_key=...)
    sizes it), then the LoRA kernels' int32 index ceiling (geometry.kernel_ceiling_adjustment at the model's pinned
    widest adapted module, D) — the step a NATIVE standup applies to every launch because it ALWAYS registers the zero
    adapter ``none__lora0`` (native_standup.bring_up_native), so the served S here is the S a STEERED standup at the
    same window resolves too: base and steered arms share one engine identity. Raises ``ValueError`` when the ceiling
    admits no row at this window (the standup refuses the same), or when the model's budget is NOT a measurement — a
    PROVISIONAL ceiling or the Llama-certified default: steered padding sizes from a number somebody measured on THIS
    model (the sweep's paste block, or the FD 24 serving probe's row in the site geometry overlay), never from a
    conservative guess. The overlay is keyed by MODULE SET: the record consulted is the SELECTED set's — ``module_set``
    when given (tests), else :func:`selected_module_set` from the models table (the promoted HPO pick, the one source every
    padded job has) — so the S derived here is the S a native standup of THAT design attests; a set the overlay does not
    certify refuses by name. ``steered``: this standup serves a TRAINED adapter of the selected pair, so the overlay's
    ``adapter_parity`` for that pair (the pick's rank; the set's aggregate when no rank is known) must be ``verified`` —
    a ``mismatch`` / ``unverified`` record refuses by name (geometry.require_verified_adapter_parity); an UNSTEERED standup
    reads the same record whatever it says, and the returned ``adapter_parity`` reports it."""
    from sisteer.capeval.serving import geometry as g

    selection: dict = {}
    if module_set is None:
        # STEERED: the promoted pick names the set (required); UNSTEERED: no pick — the zero adapter's class, the resolver's default
        module_set, selection = selected_module_set(model_key) if steered else unsteered_module_set(model_key)

    # THE BUDGET RESOLVER: geometry.budget_for — the ONE lookup over the sweep's keys AND the
    # paper-companion ONBOARDED ones (rows_for), never BUDGET_BY_MODEL indexed directly (which knows the sweep's eight only). A model's
    # OWN row (measured / provisional / onboarded) sizes through its seqs_at (the hybrid cap and the window ceiling applied); a known
    # key with NO row sizes from the Llama-certified default the resolver hands back (with its warning), capped at the hybrid ceiling
    # when rows_for says the model is hybrid; an unknown key refuses (UnknownModelKey, a KeyError — the caller names it). Both consult
    # the SITE geometry overlay (the serving probe's fold) BEFORE the built-in tables.
    rows = g.rows_for(model_key, module_set)
    rb = g.budget_for(model_key, module_set)
    if not rb.measured:
        # steered padding REFUSES a provisional (or default) budget: the S it derives must be the S the engine was measured to
        # come up at on THIS model, or the geometry pin below the standup is holding the arms to a guess
        named = (f"selected module set {module_set!r}" if module_set is not None
                 else f"selected module set {g.UNSTEERED_MODULE_SET!r} (the unsteered default: the zero adapter's class)")
        raise ValueError(
            f"{model_key}: batched-token budget source is {rb.source!r} ({rb.batched_token_budget:,}), not a measurement — a padded "
            "standup (steered or unsteered) sizes only from a measured row; run the FD 24 serving probe (scripts/sbatch/serving_probe.sbatch), "
            "whose verdict folds the measured budget + widest module dim PER MODULE SET into the site geometry overlay "
            f"({g.SITE_GEOMETRY_DIR_ENV}; {named}"
            f"{'' if not selection.get('reason') else ' — ' + str(selection['reason'])}; looked for it: "
            f"{g._site_overlay_where(model_key, module_set)})")
    # geometry certification and adapter parity are SEPARATE facts — the overlay says both. A STEERED standup requires
    # 'verified' parity for the pair it serves; an unsteered one records what the block says and sizes from the geometry regardless
    parity = None
    if rows.site is not None:
        try:
            parity = (g.require_verified_adapter_parity(rows.site, selection.get("rank"), what="this steered padded standup")
                      if steered else g.adapter_parity_for(rows.site, None))
        except g.SiteGeometryError as e:
            raise ValueError(str(e)) from None
    if rb.per_model:
        identity, seqs_cap = int(rb.entry.seqs_at(int(window))), rb.entry.seqs_cap
    else:
        identity = int(g.identity_seqs(int(window), budget=rb.batched_token_budget))
        seqs_cap = g.HYBRID_SEQS_CAP if rows.hybrid else None
        identity = min(identity, seqs_cap) if seqs_cap else identity
    # the LoRA kernels' int32 index ceiling needs the model's PINNED widest adapted module — the site overlay's probe-measured width
    # first, else geometry.LORA_MAX_MODULE_DIM_BY_MODEL (geometry.lora_max_module_dim_for): a model without one refuses BY NAME here
    # (the pin comes from the serving probe's attestation, never a guess)
    try:
        adj = g.kernel_ceiling_adjustment(model_key, int(window), identity, module_set=module_set)
        lora_max_module_dim = int(g.lora_max_module_dim_for(model_key, module_set))
    except g.UnknownModelKey as e:
        raise KeyError(f"{model_key}: {e}") from None
    served = int(adj["to_S"]) if adj else identity
    return {"model_key": model_key, "window": int(window), "batched_token_budget": int(rb.batched_token_budget),
            # the set the standup's class is: the overlay's row when one sized it; the named set; for an unsteered caller on a model with
            # no overlay, the zero adapter's class (what the standup registers) — sized from the built-in row (module_set_source None)
            "module_set": (rows.site.module_set if rows.site is not None else (module_set if module_set is not None else (g.UNSTEERED_MODULE_SET if not steered else None))),
            "module_set_selection": selection, "module_set_source": rows.site_module_set_source,
            "budget_source": rb.source, "budget_per_model": rb.per_model, "budget_certified": bool(rb.entry.certified),
            "budget_from_site_overlay": rows.site is not None,
            "hybrid_seqs_cap": seqs_cap, "identity_S": identity,
            "lora_max_module_dim": lora_max_module_dim,
            "kernel_adjustment": adj, "served_S": served, "batched_tokens": served * int(window),
            "steered": bool(steered), g.ADAPTER_PARITY_KEY: parity}


def check_geometry(model_key: str, window: int, snapshot: Path | None, *, util: float = DEFAULT_GPU_MEM_UTIL,
                   card_gib: float = H200_GIB, steered: bool = False) -> tuple[dict, list[str]]:
    """Check 5: the derived geometry and its memory sum (weights + full-coverage KV + the warmup MLP intermediate over
    the batched-token budget S x L) against the utilization's reservation. The KV per token comes from the snapshot's
    config.json (kv_bytes_per_token); with no config.json the sum is recorded as not computed."""
    try:
        geo = geometry_expectation(model_key, window, steered=steered)
    except (KeyError, ValueError) as e:
        return {"model_key": model_key, "window": int(window)}, [f"geometry: {model_key} at window {window}: {e}"]
    rec = dict(geo)
    problems: list[str] = []
    # the certified per-set module widths the record carries (the D the kernel step above sized from is one of them) — recorded
    # beside the derivation so the reader sees the probe's numbers, never re-derived from the config
    try:
        from sisteer.capeval.serving import geometry as g
        rec["certified_module_dims"] = g.certified_module_dims(model_key)
    except Exception as e:  # noqa: BLE001 — a revoked / unreadable overlay is a named problem, not a traceback
        rec["certified_module_dims"] = None
        problems.append(f"geometry: the certified module dims of {model_key} could not be read — {e}")
    cfg_path = (snapshot / "config.json") if snapshot is not None else None
    if cfg_path is None or not cfg_path.is_file():
        rec["memory"] = {"note": "the snapshot carries no config.json; the memory sum was not computed (see the model problem)"}
        return rec, problems
    model_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    kv = kv_bytes_per_token(model_cfg)
    toks = geo["batched_tokens"]
    w = weights_bytes(snapshot)
    width = mlp_intermediate_width(model_key, model_cfg)
    mem = {"weights_gib": (w / 2**30) if w else None, "weights_known": w is not None,
           "kv_gib": toks * kv["bytes_per_token"] / 2**30,
           "mlp_intermediate": width,
           "mlp_intermediate_gib": (toks * int(width["elements"]) * BF16_BYTES / 2**30) if width["elements"] is not None else None,
           "gpu_memory_utilization": util, "card_gib": card_gib, "reserved_gib": util * card_gib}
    if width["problem"]:
        # never a KeyError: the width the warmup term needs is refused BY NAME and the sum is recorded as not computed
        mem["total_gib"] = None
        rec.update(kv=kv, memory=mem)
        problems.append(f"geometry: {width['problem']}")
        return rec, problems
    mem["total_gib"] = (mem["weights_gib"] or 0.0) + mem["kv_gib"] + mem["mlp_intermediate_gib"]
    rec.update(kv=kv, memory=mem)
    if mem["total_gib"] > mem["reserved_gib"]:
        problems.append(f"geometry: weights {mem['weights_gib'] or 0:.1f} + full-coverage KV {mem['kv_gib']:.1f} + warmup MLP "
                        f"intermediate {mem['mlp_intermediate_gib']:.1f} = {mem['total_gib']:.1f} GiB > the {util} utilization's "
                        f"{mem['reserved_gib']:.1f} GiB at S {geo['served_S']} x L {window} ({toks:,} tokens, {kv['bytes_per_token']} B/token)")
    return rec, problems


def check_served_s(geo: dict, expect: int | None) -> list[str]:
    """The WIDTH PIN, before any GPU minute (2026-09-24): a lane whose instrument is one attested width passes
    ``--expect-served-s`` (the padded steered wave: 5, the seed-1 wave's S at 98,304), and the derivation of check 5 must land
    on it — through a KERNEL-CEILING lowering when the pin is below the identity S, since that lowering is what a native
    standup applies once the zero adapter is registered. A seven-cell set that lost its `none` anchor came up at the identity
    7 (the zero leg derives from `none`; without it no adapter was registered and nothing lowered S), so a derivation that
    does not reach the pin, or reaches it without the step, refuses here and again at the sbatch's post-standup pin."""
    if expect is None or geo.get("served_S") is None:
        return []
    problems: list[str] = []
    if int(geo["served_S"]) != int(expect):
        problems.append(f"geometry: the derived served S {geo['served_S']} at window {geo['window']} != the lane's pinned width {expect} "
                        f"(identity {geo['identity_S']}, kernel adjustment {geo.get('kernel_adjustment')}) — a standup at this window would "
                        "not be on the pinned instrument; refusing")
    if int(expect) < int(geo["identity_S"]) and not geo.get("kernel_adjustment"):
        problems.append(f"geometry: the pinned width {expect} is below the identity S {geo['identity_S']} but no kernel-ceiling step lowers it "
                        "— the pinned instrument is a ceiling-lowered one; refusing")
    return problems


def si_text_of(cell: str) -> str | None:
    """The system instruction a cell's requests carry, by the DRIVER's own rule (jailbreak.build_defense_cell): a
    `prompted__<si>` baseline sends the SI text as the system message, and so does a `stacked__<si>__<arm>` cell (the SI
    beside the object); the bare anchor `none` and every COMPILED cell — Class A `<si>__<arm>` (the fitted s1 objects,
    the paired write) or a bare Class B id (`arditi_allpos`) — send NO system message: the object edits the residual
    stream and the `<si>` of a Class A id names the corpus the object was fitted on, a document-corpus pseudo-instruction
    that is never a system prompt (defense_cell.registry_from_explicit_cells refuses `prompted__constv2-*` for the same
    reason). Resolving it here would render megabytes of corpus into the window arithmetic and refuse every steered
    config at any window (the steered family, 2026-09-23). Resolution goes through the driver's own resolver."""
    if cell == "none":
        return None
    from sisteer.eval import defense_cell as tt
    from sisteer.eval.jailbreak import resolve_instruction

    if not cell.startswith("prompted__") and not cell.startswith("stacked__"):
        return None                       # a compiled cell: object-only, no system message (the driver's rule)
    si = tt.si_of(cell)
    if si is None:
        raise ValueError(f"{cell!r} names no SI — cannot build its system text")
    return resolve_instruction(si).text


def static_variants(goals, families) -> dict:
    """{family: [(goal_id, variant_name, prompt), ...]} — EVERY variant of EVERY goal through each family's OWN builder
    (the builders prompted_window_check.longest_static_prompts uses), so the render below takes the true max IN
    TOKENS. The char-longest proxy under-reads: on 2026-09-22 it named grok420's longest goal at 6,403 rendered tokens
    under the constitution SI while another goal rendered to 6,412 — 9 tokens that a window sized to the proxy would
    not carry. 2,590 variants per cell over the 259-goal pool (pasttense 2 / codeattack 3 / renellm 3 / grok420 2 per
    goal) render in well under a minute on a login node's CPU."""
    from sisteer.eval.attacks import codeattack, grok420, pasttense, renellm

    builders = {"pasttense": pasttense.pasttense_variants, "codeattack": codeattack.codeattack_variants,
                "renellm": renellm.renellm_variants, "grok420": grok420.grok420_variants}
    out: dict = {}
    for fam in families:
        rows = []
        for g in goals:
            try:
                variants = builders[fam](g)
            except Exception as e:  # noqa: BLE001 — recorded, never swallowed
                raise SystemExit(f"[pad-preflight] {fam}: cannot render goal {g.goal_id}: {type(e).__name__}: {e}")
            rows.extend((g.goal_id, name, prompt) for name, prompt in variants)
        out[fam] = rows
    return out


def token_max(count, system: str | None, variants) -> tuple[int, str | None, str | None]:
    """``(max rendered tokens, goal_id, variant)`` over ``variants`` under ``system`` — ``count(system, prompt)`` is the
    render counter (prompted_window_check.rendered_tokens in production; a test injects its own). Every variant is
    rendered: the max in tokens is not the max in characters."""
    best: tuple[int, str | None, str | None] = (0, None, None)
    for gid, name, prompt in variants:
        n = int(count(system, prompt))
        if n > best[0]:
            best = (n, gid, name)
    return best


def render_rows(*, tok, cfg: dict, contract: dict, window: int, variants_by_family: dict, families, count=None) -> list[dict]:
    """One row per (cell, family): the TRUE longest rendered prompt (the token max over every variant of every goal
    under the cell's SI), the worst-case need, and whether it fits."""
    if count is None:
        from prompted_window_check import rendered_tokens

        def count(system, user):
            return rendered_tokens(tok, system, user)

    rows = []
    for cell in cfg["explicit_cells"]:
        system = si_text_of(cell)
        sys_tokens = count(system, "") if system else 0
        for fam in families:
            n, gid, variant = token_max(count, system, variants_by_family[fam])
            need = rendered_need(n, pad_tokens=contract["tokens"], tolerance=contract["tolerance"], cap=contract["cap"])
            rows.append({"level": contract["level"], "cell": cell, "family": fam, "goal_id": gid, "variant": variant,
                         "n_variants": len(variants_by_family[fam]),
                         "system_tokens_rendered": sys_tokens, "prompt_tokens_rendered": n,
                         "pad_worst_case": pad_worst_case(contract["tokens"], contract["tolerance"]),
                         "cap": contract["cap"], "need": need, "window": window, "fits": need <= window})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--window", type=int, required=True, help="the standup's --max-model-len (every level is served at it)")
    ap.add_argument("--config", action="append", required=True, metavar="YAML",
                    help="a generated padded-static config (repeatable: one per level this job runs)")
    ap.add_argument("--filler", default=None, metavar="PATH", help="the PG-19 filler the padded drivers are pointed at")
    ap.add_argument("--filler-sha256", default=PG19_FILLER_SHA256, help="the corpus's full sha256 (default: the leg's)")
    ap.add_argument("--filler-bytes", type=int, default=PG19_FILLER_BYTES, help="the corpus's size; 0 = do not check")
    ap.add_argument("--hf-home", default=None, help="the HF cache root (default $HF_HOME, hf_env.sh's)")
    ap.add_argument("--families", default=",".join(STATIC_FAMILIES), help="comma list of the static families this job runs")
    ap.add_argument("--align", type=int, default=1024, help="the suggested window's alignment")
    ap.add_argument("--skip-render", action="store_true", help="skip check 4 (the tokenizer render); recorded, never silent")
    ap.add_argument("--limit", type=int, default=None, help="SMOKE: head-cap the goal pool for the render")
    ap.add_argument("--model-key", default=None, help="the geometry-table model key (default: the one the config names carry)")
    ap.add_argument("--gpu-mem-util", type=float, default=DEFAULT_GPU_MEM_UTIL,
                    help="the standup's gpu_memory_utilization for the memory sum (common.sh GPU_MEM_UTIL; default 0.9)")
    ap.add_argument("--expect-served-s", type=int, default=None, metavar="S",
                    help="the WIDTH PIN (check_served_s): refuse unless check 5's derived served S is exactly this (the padded steered "
                         "wave passes 5, the seed-1 wave's attested width at 98,304; below the identity S it must come through the "
                         "kernel-ceiling step)")
    ap.add_argument("--steered", action="store_true",
                    help="this standup serves a TRAINED adapter of the selected pair (target_static_padded_obj.sbatch): require the site "
                         "geometry overlay's adapter_parity to be 'verified' for that pair (D145); an unsteered standup omits it")
    ap.add_argument("--out", default=None, help="the JSON record (default: no file)")
    args = ap.parse_args(argv)

    if args.window < 1:
        print(f"[pad-preflight] --window must be a positive int, got {args.window}", flush=True)
        return EXIT_USAGE
    families = tuple(f for f in args.families.split(",") if f)
    bad = [f for f in families if f not in STATIC_FAMILIES]
    if bad or not families:
        print(f"[pad-preflight] --families {args.families!r}: the padded leg runs {list(STATIC_FAMILIES)} only (bad: {bad})", flush=True)
        return EXIT_USAGE
    hf_home = Path(args.hf_home or os.environ.get("HF_HOME") or "")
    if not str(hf_home):
        print("[pad-preflight] no HF_HOME in the environment and no --hf-home — source hf_env.sh first", flush=True)
        return EXIT_USAGE

    rec: dict = {"schema": "padding_preflight/1", "window": int(args.window), "families": list(families),
                 "configs": [], "problems": [], "render_skipped": bool(args.skip_render),
                 "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    problems: list[str] = []

    # 1. the filler
    need_filler = False
    contracts: list[tuple[Path, dict, dict]] = []
    for c in args.config:
        path = Path(c)
        if not path.is_absolute():
            path = REPO / c
        if not path.is_file():
            print(f"[pad-preflight] --config {c}: no such file", flush=True)
            return EXIT_USAGE
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        try:
            contract = contract_of(cfg, path)
        except (ValueError, KeyError) as e:
            problems.append(f"contract: {e}")
            continue
        contracts.append((path, cfg, contract))
        need_filler = need_filler or contract["tokens"] > 0
    if need_filler:
        if not args.filler:
            problems.append("filler: a padded level is configured but no --filler was given")
        else:
            frec, fprob = check_filler(Path(args.filler), want_sha=args.filler_sha256.lower(),
                                       want_bytes=(args.filler_bytes or None))
            rec["filler"] = frec
            problems.extend(fprob)
    else:
        rec["filler"] = {"note": "no padded level configured; the filler is not consulted"}

    # 2. the model, 3. the contract
    seen_models = set()
    for path, cfg, contract in contracts:
        pins = (cfg["model"], cfg["model_revision"])
        if pins not in seen_models:
            seen_models.add(pins)
            mrec, mprob = check_model(*pins, hf_home)
            rec.setdefault("models", []).append(mrec)
            problems.extend(mprob)
        crec = {"config": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path), **contract,
                "private_run": cfg.get("private_run"), "explicit_cells": list(cfg["explicit_cells"])}
        crec["contract_problems"] = check_contract(contract, args.window) + check_cap(contract)
        problems.extend(crec["contract_problems"])
        rec["configs"].append(crec)
    if len({c["private_run"] for c in rec["configs"]}) != len(rec["configs"]):
        problems.append("contract: two configs share a private_run — one padding contract per store")
    if len({tuple(c["explicit_cells"]) for c in rec["configs"]} | {tuple()}) > 2:
        problems.append("contract: the configs do not list the same explicit_cells — one standup serves one cell set")

    # 5. the geometry (pure arithmetic through the launcher's own module, plus the memory sum off the snapshot's
    # config.json — computed once for the ONE model every config pins)
    if contracts:
        import make_padding_configs as gen

        keys = {gen.key_of(path) for path, _, _ in contracts}
        if len(keys) != 1:
            problems.append(f"geometry: the configs name different model keys {sorted(keys)} — one standup serves one model")
        model_key = args.model_key or next(iter(sorted(keys)))
        if args.model_key and keys != {args.model_key}:
            problems.append(f"geometry: --model-key {args.model_key} but the configs are named for {sorted(keys)}")
        snap = None
        for m in rec.get("models") or []:
            if Path(m["snapshot"]).is_dir():
                snap = Path(m["snapshot"])
        grec, gprob = check_geometry(model_key, args.window, snap, util=args.gpu_mem_util, steered=bool(args.steered))
        rec["geometry"] = grec
        problems.extend(gprob)
        rec["steered"] = bool(args.steered)
        rec["expect_served_s"] = args.expect_served_s
        problems.extend(check_served_s(grec, args.expect_served_s))

    # 4. the rendered need (needs the tokenizer only, so it runs on a tokenizer-only cache too — the missing weights
    # stay their own problem above)
    tokenizer_ok = bool(rec.get("models")) and all(m.get("tokenizer_ok") for m in rec["models"])
    if not args.skip_render and contracts and not tokenizer_ok:
        problems.append("render: not run — the pinned tokenizer is not in the cache (see the model problem above)")
    if not args.skip_render and contracts and tokenizer_ok:
        import two_si_run as tsr
        from sisteer.models import load_tokenizer

        path0, cfg0, _ = contracts[0]
        tok = load_tokenizer(cfg0["model"], revision=cfg0.get("model_revision"))
        goals = tsr._eval_goals(cfg0, confirmatory=True)
        if args.limit:
            goals = goals[: args.limit]
        variants_by_family = static_variants(goals, families)
        rows: list[dict] = []
        for path, cfg, contract in contracts:
            rows.extend(render_rows(tok=tok, cfg=cfg, contract=contract, window=args.window,
                                    variants_by_family=variants_by_family, families=families))
        rec["render"] = {"n_goals": len(goals), "n_variants": {f: len(v) for f, v in variants_by_family.items()},
                         "rule": "token max over every variant of every goal under each cell's SI", "rows": rows}
        misfit = [r for r in rows if not r["fits"]]
        if misfit:
            worst = max(rows, key=lambda r: r["need"])
            suggest = align_up(worst["need"], args.align)
            rec["render"]["worst"] = worst
            rec["render"]["smallest_window"] = suggest
            for r in sorted(misfit, key=lambda r: -r["need"])[:8]:
                problems.append(f"render p{r['level']} {r['cell']}/{r['family']}: {r['prompt_tokens_rendered']} rendered + "
                                f"{r['pad_worst_case']} pad (worst case) + {r['cap']} cap = {r['need']} > window {args.window}")
            problems.append(f"render: the smallest window carrying every row is {suggest} (worst need {worst['need']}: "
                            f"p{worst['level']} {worst['cell']}/{worst['family']}); pass it as the sbatch's window positional")
    elif args.skip_render:
        print("[pad-preflight] render check SKIPPED (--skip-render) — recorded, never assumed", flush=True)

    rec["problems"] = problems
    rec["ok"] = not problems
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    levels = ",".join(f"p{c['level']}" for c in rec["configs"])
    if rec.get("render"):
        worst = max(rec["render"]["rows"], key=lambda r: r["need"])
        summary = (f"worst rendered need {worst['need']} (p{worst['level']} {worst['cell']}/{worst['family']}: "
                   f"{worst['prompt_tokens_rendered']} + {worst['pad_worst_case']} + {worst['cap']}) vs window {args.window}")
    else:
        summary = "render not run"
    geo = rec.get("geometry") or {}
    if geo.get("served_S") is not None:
        mem = geo.get("memory") or {}
        summary += (f"; geometry S {geo['served_S']} x L {geo['window']} (identity {geo['identity_S']}"
                    f"{', kernel ceiling ' + str(geo['kernel_adjustment']['ceiling']) if geo.get('kernel_adjustment') else ''})"
                    + (f", memory {mem['total_gib']:.1f} of {mem['reserved_gib']:.1f} GiB at {geo['kv']['bytes_per_token']} B/token KV"
                       if mem.get("total_gib") is not None else ""))
    print(f"[pad-preflight] {'PASS' if rec['ok'] else 'REFUSE'}: window={args.window} levels={levels} families={','.join(families)} "
          f"filler={(rec.get('filler') or {}).get('sha256', 'n/a')[:12]} {summary}" + (f" -> {args.out}" if args.out else ""), flush=True)
    for p in problems:
        print(f"[pad-preflight] REFUSED {p}", flush=True)
    return 0 if rec["ok"] else EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
