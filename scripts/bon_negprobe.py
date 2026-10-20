#!/usr/bin/env python
"""GPU negative+positive direction for the served steering-bites gate.

Mirrors tests/test_bon_served.py::test_steering_bites_catches_inert_arm on
REAL hardware: assert_steering_bites must RAISE FidelityError when the
"steered" and "scale0" legs are served IDENTICALLY (a no-op / scale-0 lens
leg standing in for the steered route), and must PASS when the real
steered leg is contrasted against its scale-0 twin.

A gate demonstrated in only one direction is not a gate. The POSITIVE
control runs in the SAME server process precisely so a false negative
(a dead server returning identical output to everything) cannot be read
as the gate firing: the negative "raise" only counts as a real catch if
the positive control proves the server genuinely steers.

Reuses the smoke's GPU-proven worker registry and the production
vllm_server launch/attestation path (same bring-up as the cache-iso gate).
Writes to a FRESH out dir; touches no existing gate record.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import socket
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
from sisteer import site as _site  # noqa: E402
#: the cluster's shared root (sisteer.site: the job's SISTEER_SHARED_ROOT, else the site config's cluster block). Where neither
#: sets it (an import off the cluster) it keeps its placeholder form, and a run refuses by name before it reads or writes there.
try:
    SHARED = _site.expand_path("$SHARED")
except (_site.PlaceholderError, _site.SiteConfigError):
    SHARED = Path("$SHARED")
REGISTRY = SHARED / "petri/serving/steered_smoke/registry"
OUT = SHARED / "petri/serving/bon_negprobe"
ARM = "static_r_matched"
TWIN = "static_r_matched__scale0"

RECORD = "bon_negprobe_record.json"
FAILED = "bon_negprobe_failed.json"
UNAVAILABLE = "bon_negprobe_unavailable.json"

# harmless prompts; a superset of the cache-iso gate's set so the positive
# control clears n_differing>=1 with margin (static_r bites ~75% of prompts).
PROMPTS = (
    "List the first five prime numbers.",
    "Explain briefly why the sky is blue.",
    "Write a haiku about a quiet library.",
    "Name three uses for a paperclip.",
    "Summarize the water cycle in two sentences.",
    "Give two tips for staying hydrated.",
    "What is the capital of France?",
    "Describe a cat in one sentence.",
)
MAX_TOKENS = 48


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _post(url: str, body: dict, *, token: str, timeout: float = 240.0):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), json.loads(resp.read().decode())


def chat(launch, prompt, *, selector, cache_salt):
    payload = {
        "model": launch.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": MAX_TOKENS, "seed": 1234,
        "vllm_xargs": selector, "cache_salt": cache_salt,
    }
    status, body = _post(f"{launch.base_url}/v1/chat/completions", payload,
                         token=launch.api_key)
    if status != 200:
        raise RuntimeError(f"chat status {status}: {body}")
    return str(body["choices"][0]["message"]["content"])


def uniq(tag: str) -> str:
    return f"petri|{tag}|{os.urandom(8).hex()}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (RECORD, FAILED, UNAVAILABLE):
        p = OUT / name
        if p.exists():
            p.unlink()

    try:
        vs.require_vllm()
        from sisteer.capeval.serving import lens_steer as ls
        ls.require_vllm_lens()
    except Exception as e:  # noqa: BLE001
        atomic_json(OUT / UNAVAILABLE, {
            "status": "unavailable", "reason": f"{type(e).__name__}: {e}",
            "recorded_at": time.time()})
        print(f"negprobe unavailable: {e}", file=sys.stderr)
        return 0
    if not (REGISTRY / "registry.json").is_file():
        atomic_json(OUT / UNAVAILABLE, {
            "status": "unavailable",
            "reason": f"no worker registry at {REGISTRY}",
            "recorded_at": time.time()})
        return 0

    from transformers import AutoTokenizer
    from sisteer.eval.attacks import bon_served as bs
    from sisteer.capeval.serving.fidelity import FidelityError

    spec = cm.load_petri_spec(
        Path("configs/capeval/manifests/petri_twosi.yaml"))
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                        trust_remote_code=False)
    template = tok.chat_template
    if not isinstance(template, str) or not template:
        raise RuntimeError("served tokenizer carries no chat template")
    template_path = OUT / "chat_template.jinja"
    template_path.write_bytes(template.encode())
    template_sha16 = cm.sha256_hex16(template.encode())
    flags = [f for f in spec.serving_launch_flags if f != "--enable-lora"]
    spec = dataclasses.replace(
        spec, target_model=MODEL, target_revision=REVISION,
        serving_mode="interleaved", serving_launch_flags=flags,
        chat_template_hash=template_sha16)

    launch = vs.build_server_launch(
        spec, port=free_port(), lens_registry_dir=REGISTRY,
        chat_template_path=template_path, max_model_len=2048,
        max_num_seqs=8)

    reg = json.loads((REGISTRY / "registry.json").read_text())
    by_id = {e["cell_id"]: e for e in reg["entries"]}
    arm_sel = {"petri_object": ARM, "expect_hash": by_id[ARM]["expect_hash"]}
    twin_sel = {"petri_object": TWIN,
                "expect_hash": by_id[TWIN]["expect_hash"]}

    record = {
        "status": "started",
        "model": MODEL, "revision": REVISION,
        "arm": ARM, "twin": TWIN,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": socket.gethostname(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "registry_sha256": hashlib.sha256(
            (REGISTRY / "registry.json").read_bytes()).hexdigest(),
        "template_sha16": template_sha16,
        "n_prompts": len(PROMPTS),
    }

    proc = None
    try:
        proc = vs.launch_server(launch, log_path=OUT / "server.log",
                                ready_timeout_s=900, install_timeout_s=600)

        def _ids(selector, prompt):
            text = chat(launch, prompt, selector=selector,
                        cache_salt=uniq("neg"))
            return list(text.encode("utf-8"))

        # POSITIVE control: real steered leg vs its scale-0 twin. Must PASS
        # (returns diag). This proves the server genuinely steers, so the
        # negative below cannot be a dead-server artifact.
        def positive_probe(_arm):
            steered = [_ids(arm_sel, p) for p in PROMPTS]
            scale0 = [_ids(twin_sel, p) for p in PROMPTS]
            return steered, scale0

        positive_passed = False
        positive_diag = None
        positive_error = None
        try:
            positive_diag = bs.assert_steering_bites([ARM], positive_probe)
            positive_passed = True
        except FidelityError as e:
            positive_error = str(e)

        # NEGATIVE: the scale-0 (no-op) leg served for BOTH positions — a
        # no-op route standing in for the steered route. Greedy decode makes
        # the two legs token-identical, so assert_steering_bites MUST RAISE.
        def inert_probe(_arm):
            leg_a = [_ids(twin_sel, p) for p in PROMPTS]
            leg_b = [_ids(twin_sel, p) for p in PROMPTS]
            return leg_a, leg_b

        negative_raised = False
        negative_error = None
        try:
            bs.assert_steering_bites([ARM], inert_probe)
        except FidelityError as e:
            negative_raised = True
            negative_error = str(e)

        record.update({
            "positive_passed": positive_passed,
            "positive_diag": positive_diag,
            "positive_error": positive_error,
            "negative_raised": negative_raised,
            "negative_error": negative_error,
        })

        ok = positive_passed and negative_raised
        if not ok:
            atomic_json(OUT / FAILED, dict(record, status="failed",
                        recorded_at=time.time()))
            print(f"NEGPROBE FAILED: positive_passed={positive_passed} "
                  f"negative_raised={negative_raised}", file=sys.stderr)
            return 1
        atomic_json(OUT / RECORD, dict(record, status="completed",
                    recorded_at=time.time()))
        print(f"NEGPROBE OK: positive bites (n_differing="
              f"{positive_diag['arms'][ARM]['n_differing']}/"
              f"{positive_diag['arms'][ARM]['n']}); negative raised "
              f"FidelityError: {negative_error!r}")
        return 0
    finally:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    _site.require_expanded(str(SHARED), key="the cluster's shared root (SISTEER_SHARED_ROOT)")
    raise SystemExit(main())
