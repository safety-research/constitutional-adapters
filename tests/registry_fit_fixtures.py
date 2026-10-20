"""Shared SYNTHETIC fixtures for the adapter tests (fit_sidecars / objects_bank / hpo_logs / wandb_runs):
invented shas, `TESTHOST`-style placeholders, no real transcript or object bytes (repo privacy rule).
"""
from __future__ import annotations

import hashlib
import json
import os

from sisteer.registry import synth
from sisteer.registry.adapters import fit_sidecars as FS

T = synth.t
NOW = T(5000)
CORPUS_SHA = "c" * 64
POOL_SHA = "d" * 64
POOL_CONTENT_SHA = "e" * 64
PARENT_A = "1" * 64
PARENT_B = "2" * 64
REV = "f" * 40


def seeds() -> FS.Seeds:
    s = FS.Seeds(source="test")
    s.model_id_by_hf_repo = {"synthetic/base-4b": synth.FIT_MODEL_ID, "synthetic/instruct-4b": synth.MODEL_ID}
    s.si_sha_by_id = {"synthetic_si": synth.SI_SHA}
    s.si_kind_by_id = {"synthetic_si": "safety_instruction"}
    s.corpus_ids = {"synth-corpus-full": "synth-corpus-full", "synth-ctl-full": "synth-ctl-full"}
    s.paper_label_by_spelling = {"synth-corpus-full__lora_r4_ce": {"paper_label": "constitution_lora", "object_role": "constitution"},
                                 "<si>__lora_r4_ce_minus_ctl": {"paper_label": "difference_lora", "object_role": "difference"},
                                 "synthetic_si__psr_kl_all": {"paper_label": "safety_relu", "object_role": "safety"}}
    s.object_kind_by_spelling = {"lora_r1_all": {"object_kind": "lora", "fit_rank": 4, "source": "alias"}}
    s.out_of_scope_by_spelling, s.out_of_scope_prefixes = FS.out_of_scope_tables(OUT_OF_SCOPE_TABLE)
    return s


#: the shape of `legacy_spellings.yaml: out_of_scope_object_kinds` (results registry) with INVENTED reasons: a bundle
#: of one of these kinds is parked by name by the objects bank (`fit_sidecars.OutOfScopeObject`), never refused
OUT_OF_SCOPE_TABLE = (
    {"legacy": "lofit", "status": "withdrawn", "object_role": "other", "reason": "synthetic: a learned-feature intervention kind outside the taxonomy", "source": "decisions.md D37"},
    {"legacy": "alphakl_soft | alphakl", "status": "withdrawn", "object_role": "other", "reason": "synthetic: a soft-KL variant outside the taxonomy", "source": "decisions.md D37"},
    {"legacy": "token_faithful | token_faithful_all | token_appended", "status": "withdrawn", "object_role": "other",
     "reason": "synthetic: an appended-token objective outside the taxonomy", "source": "decisions.md D37"},
    {"legacy": "diffmean_random | rand_ | synthetic_si__diffmean_random", "status": "withdrawn", "object_role": "other",
     "reason": "synthetic: Gaussian direction controls of the fitting pipeline, never an object under test", "source": "decisions.md D37"},
    {"legacy": "bidpo | did | cast | circuit_breaker", "status": "withdrawn", "object_role": "other",
     "reason": "synthetic: methods registered or retired before the taxonomy", "source": "decisions.md D37"},
)


def _write(dir_path: str, name: str, obj) -> str:
    os.makedirs(dir_path, exist_ok=True)
    p = os.path.join(dir_path, name)
    with open(p, "w", encoding="utf-8") as fh:
        if isinstance(obj, str):
            fh.write(obj)
        else:
            json.dump(obj, fh, indent=1)
    return p


def _blob(dir_path: str, payload: bytes = b"synthetic weights") -> str:
    os.makedirs(dir_path, exist_ok=True)
    p = os.path.join(dir_path, "object.pt")
    with open(p, "wb") as fh:
        fh.write(payload)
    return p


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def corpus_tbm(sha: str, **over):
    d = {"accum_steps": 4, "arm": "lora_r4_ce", "batch_size": 2, "cell_id": "synth-corpus-full__lora_r4_ce", "chat_template_bypassed": True,
         "completion_tokens": None, "data_mode": "documents", "effective_batch": 8, "resolved_batch_size": 2, "heldout_pool_sha": POOL_SHA, "loss_impl": "full", "lr": 1.26e-3,
         "max_len": 8192, "model_id": "synthetic/base-4b", "model_revision": REV, "object_sha256": sha, "positions": "all", "positions_at_fit": "all",
         "response_header_len": None, "sample_manifest_sha": CORPUS_SHA, "si_id": "synth-corpus-full", "site": "o_proj_out", "target_layers": [3, 7, 11, 15],
         "teacher_forward": "solo", "template_hash": "a4aee8afcf2e0711", "training_budget_kind": "document_tokens", "warmup_frac": 0.15,
         "continued_from": {"new_cap": 16, "lr_schedule": {"lr_schedule": "warmup_cosine"}, "stop_reason": "early_stop"},
         "fit_record": {"best_epoch": 4, "best_heldout_loss": 2.04, "early_stop": {"patience": 1, "heldout_n": 128}, "epochs_run": 5,
                        "heldout_curve": [2.38, 2.07, 2.05, 2.04, 2.06, 2.10], "optimizer_steps_run": 10845, "realized_eff_batch": [8, 8], "stopped_early": True}}
    d.update(over)
    return d


def si_tbm(sha: str, **over):
    d = {"accum_steps": 16, "arm": "psr_kl_all", "batch_size": 2, "cell_id": "synthetic_si__psr_kl_all", "completion_tokens": 512, "effective_batch": 32,
         "lora_rank": None, "lr_schedule": "warmup_cosine", "max_grad_norm": 0.496, "max_len": 16384, "model_id": "synthetic/instruct-4b", "model_revision": REV,
         "positions": "response", "positions_at_fit": "response", "response_header_len": 3, "si_id": "synthetic_si", "site": "resid_post",
         "target_layers": list(range(8)), "teacher_forward": "precomputed_topk", "template_hash": "0a2c8073c878ab1d", "warmup_frac": 0.15, "weight_decay": 0.01,
         "teacher_topk": {"path": "/vol/pool/synthmodel/pairs_synthetic_si.jsonl.teachertopk.safetensors", "pool_content_sha256": POOL_CONTENT_SHA, "top_k": 200},
         "object_sha256": sha,
         "fit_record": {"best_epoch": 9, "best_heldout_loss": 0.0088, "epochs_run": 10, "heldout_curve": [0.04, 0.01, 0.009], "loss": "soft",
                        "optimizer_steps": 2550, "stopped_early": False}}
    d.update(over)
    return d


