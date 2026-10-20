"""Generate ``sampling_literals_before.json`` — the ORACLE for the decode-policy redirect (plan §4.2).

Run ONCE at commit 275b7073 (branch registry-seeds, BEFORE any sampling entry point read the
registry export), against the literal tables the harness carried at that commit:

    PYTHONPATH=src python tests/fixtures/registry/gen_sampling_literals_before.py

It records, for every (mode, model_key) the harness served and for every other sampling entry
point, the values AND None-ness of the row, the ``sampling_identity()`` block the driver hashes,
the ``--override-generation-config`` JSON, the protocol rows, the BoN harness default and the
misalignment runner's sampler rows. ``tests/test_registry_decode_policy_entry_points.py`` asserts
the export-backed tables reproduce every entry byte for byte; the file is never regenerated (a
regenerated oracle would prove nothing).
"""
from __future__ import annotations

import json
import os
import sys

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.tiers import inspect_served
from sisteer.capeval.misalign import runner as MR
from sisteer.eval.attacks import bon_served as BON
from sisteer.eval.attacks import target_decode as TD

TASKS = {"inspect/gsm8k_paired200": 1}


def row_dict(r: C.SamplingRow) -> dict:
    return {k: getattr(r, k) for k in C.SAMPLING_ROW_FIELDS}


def main() -> None:
    out: dict = {"generated_at_commit": "275b7073", "identity_tasks": TASKS, "modes": {}}
    for mode, table in C.SAMPLING_TABLE_BY_MODE.items():
        rows = {}
        for key in sorted(table):
            r = table[key]
            rows[key] = {"row": row_dict(r), "identity": C.sampling_identity(r, TASKS),
                         "identity_for_mode": C.sampling_identity_for_mode(mode, key, list(TASKS), TASKS)}
        refuses = sorted(k for k in sg.MODEL_KEYS if k not in table)
        out["modes"][mode] = {"rows": rows, "refuses": refuses}
    out["greedy"] = {"row": row_dict(C.GREEDY_SAMPLING), "identity": C.sampling_identity(C.GREEDY_SAMPLING, TASKS),
                     "identity_for_mode": C.sampling_identity_for_mode("greedy", "gemma4e4b", list(TASKS), TASKS),
                     "GREEDY_TEMPERATURE": C.GREEDY_TEMPERATURE,
                     "inspect_served_GREEDY_TEMPERATURE": inspect_served.GREEDY_TEMPERATURE}
    out["qwen35_vendor_rows"] = {k: row_dict(v) for k, v in C.QWEN35_VENDOR_ROWS.items()}
    out["server_default"] = {hf: {"row": row_dict(r), "override_generation_config": C.server_default_generation_config(r)}
                             for hf, r in C.SERVER_DEFAULT_SAMPLING_BY_MODEL.items()}
    out["server_defaultable_keys"] = list(C.SERVER_DEFAULTABLE_SAMPLING_KEYS)
    out["protocol"] = {
        "BFCL_PROTOCOL_TEMPERATURE": C.BFCL_PROTOCOL_TEMPERATURE, "PROTOCOL_TOP_P": C.PROTOCOL_TOP_P,
        "PROTOCOL_TOP_K": C.PROTOCOL_TOP_K, "SAMPLING_CLAMP_OVERRIDE": dict(C.SAMPLING_CLAMP_OVERRIDE),
        "SAMPLING_CLAMP_OVERRIDE_ARG": C.SAMPLING_CLAMP_OVERRIDE_ARG, "VLLM_MAX_TEMP_PINNED": C.VLLM_MAX_TEMP_PINNED,
        "tasks": {t: {"executed": row_dict(p.executed), "protocol_temperature": p.protocol_temperature,
                      "executed_reason": p.executed_reason} for t, p in C.PROTOCOL_SAMPLING_BY_TASK.items()},
        "identity_for_mode": C.sampling_identity_for_mode("protocol", None, list(C.PROTOCOL_SAMPLING_BY_TASK),
                                                          {t: 1 for t in C.PROTOCOL_SAMPLING_BY_TASK}),
    }
    out["sampling_unverified_model_keys"] = sorted(C.SAMPLING_UNVERIFIED_MODEL_KEYS)
    out["sampling_modes"] = list(C.SAMPLING_MODES)
    out["card_dicts"] = {"_GEMMA_CARD": dict(C._GEMMA_CARD), "_LLAMA_CARD": dict(C._LLAMA_CARD), "_QWEN35_CARD": dict(C._QWEN35_CARD)}
    out["misalign_runner"] = {
        "VENDOR_SAMPLING": {f"{m}|{t}": dict(v) for (m, t), v in MR.VENDOR_SAMPLING.items()},
        "t1_default": MR.resolve_sampling("t1", 1.0, None),
        "vendor_qwen4b": MR.resolve_sampling("vendor", 1.0, {"base_model": "Qwen/Qwen3.5-4B", "thinking_mode": "off"}),
        "build_payload_default_temperature": MR.build_payload("m", [{"role": "user", "content": "x"}], max_tokens=1)["temperature"],
    }
    out["bon_served"] = {"sent_sampler_default": BON._sent_sampler(None), "preset_label_default": BON._preset_label(None, None),
                         "generate_one_detailed_default_temperature": 1.0}
    out["target_decode"] = {"modes": list(TD.MODES), "decode_mode_None": TD.decode_mode(None), "decode_mode_0": TD.decode_mode(0.0),
                            "decode_mode_0p6": TD.decode_mode(0.6)}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampling_literals_before.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(path, sum(len(m["rows"]) for m in out["modes"].values()), "table rows")


if __name__ == "__main__":
    main()
