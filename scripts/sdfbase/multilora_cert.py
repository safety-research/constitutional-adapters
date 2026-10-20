#!/usr/bin/env python
"""Native multi-LoRA certification — ONE target, ONE GPU, ONE job.

Sequentially on one H200, inside the serving venv::

  prep/    0.  preflight      snapshot cached, object.pt sha == the pin, venv imports
           0b. fp32 oracle    petri_serving_gate.py gate-worker-reference --objects-json
                              (GPU FREE — runs and EXITS before any vLLM starts; keyed by
                              the HARNESS cell id, never the TrainTest id)
  unit/    1.  lora-native standup (E's bring_up_lora) -> five positive controls -> CANARY_OK
               identity.json (the unit's full identity) and run_identity.json
               (RunIdentity(engine="lora_native"), the one native run_id producer)
           1b. §10 gate canary on the native standup (REQUIRED; the fallback evidence)
           2.  probes through the lora proxy: lora_native, lora_native_rep, lora_zero, lora_none
           3.  throughput on the lora proxy at B in {8, 32, S}
           4.  stop; wait for the GPU to drain
           5.  lens STACKED (capctrl_serving_up.bring_up, serving_path="lens" EXPLICITLY —
               the standup default is native since 2026-09-13, and an implicit path would
               make the head-to-head native-vs-native): probes lens_stacked, _rep,
               the batch-composition pair compA / compB; throughput; stop
           6.  lens PER_ROW: probes lens_per_row; throughput at B = S; stop
           7.  admission.json = multilora_admission.admission_verdict(...) — the ONE function
               (the FUNCTIONAL rule: readiness attested + the positive controls (six by name; boundary_refusal not applicable without a restricted-mode cell) + gate canary + not
               slower at S; the greedy-token numerics recorded, never a gate);
               head_to_head.json, controls_summary.json, DONE.json (last), RESULT

TARGETS are rows, not models: ``Target.key`` names the ROW (the out dir
``<key>_<jobid>``, the RESULT's ``target=``, ``--model-key`` on the command
line), ``Target.model_key`` the MODEL (the geometry table, the bank view, the
adapter bank, pins.json, the standup's ``model_key``, the run identity). Two
rows share the llama8b model: ``llama8b`` (the lora_all cell) and
``llama8b_subtraction`` (the subtraction cell).

The unit is INDIVISIBLE (§5): ``--resume-from`` reuses ``prep/oracle`` after a
binding check, and a COMPLETE prior unit whole under its original provenance
(``unit_reused_from``) when its identity matches; a partial prior unit is never
consulted. The verdict is a REPORTED outcome: rc 0 iff every control passed,
the attestation exists and every phase completed (a not-admitted target is a
green job with ``verdict=not_admitted`` in RESULT, never a hidden red one).

Every external seam (E's standup, capctrl's lens standup, the two gate
subprocesses, the HTTP probes and windows, the GPU wait, JBB, versions) goes
through :class:`Deps`, so the CPU tests drive the whole record layout with
fakes. The PRODUCTION binding of the LoRA standup's contract is in ONE place,
:func:`_production_bring_up_lora`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Dict, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple)

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lora_serving_up as lsu  # noqa: E402  (E's module: the standup, the cluster layout, its record names)
import multilora_admission as adm  # noqa: E402  (pure, stdlib-only)
# native-psr-serving §1.7: the closed-loop sampler lives in ONE module (the timing harness and this
# driver run the SAME loop); the names are re-exported here for the tests and the seams that read them
from sisteer.capeval.serving.closed_loop import (FILLER_PROMPT, METRIC_RE,  # noqa: E402,F401
                                                closed_loop_via_proxy, read_metrics)
from sisteer.capeval.serving import serving_attestation as sa  # noqa: E402  (stdlib-only leaf)
from sisteer.capeval.serving.serving_attestation import sha256_bytes, sha256_file  # noqa: E402,F401

# --------------------------------------------------------------------------- #
# constants                                                                     #
# --------------------------------------------------------------------------- #

EVAL_DEFAULT = "families_served:short"
PROMPTED_CELL = "prompted__safety_refusal"
PINS_REL = "configs/sdfbase_full22_pins.json"
REQUIREMENTS_REL = "configs/capeval/vllm-requirements.txt"
GATE_SCRIPT_REL = "scripts/petri_serving_gate.py"
DEFENSES_REL = "exp/objects/sdfbase/full22"      # $SHARED/<this>/<key>/cells_view/<view>
PROBE_MAX_TOKENS = 128
THROUGHPUT_MAX_TOKENS = 200
N_JBB = 32
REDUCTION_ENV_VAR = "SISTEER_LENS_STACKED_REDUCTIONS"   # mt_serving_up.REDUCTION_ENV_VAR
#: THE LENS LEGS ARE LENS BY NAME (native-serving-default §1.4): the standup
#: entry points default to the native LoRA engine since 2026-09-13
#: (``registry_adapter.DEFAULT_STANDUP_SERVING_PATH``), so the head-to-head's
#: two lens legs pass ``serving_path="lens"`` to ``capctrl_serving_up.bring_up``
#: EXPLICITLY — without it the default flip would silently make the comparison
#: native-vs-native. The literal is the ``SERVING_PATH_POLICIES`` spelling.
LENS_SERVING_PATH = "lens"
MAMBA_REFUSAL_RE = re.compile(r"Mamba cache block", re.I)
MAMBA_LOWER_REASON = "mamba_cache_blocks_under_graphs"
#: ENGINE-INIT FAILURE CLASSES (job 488932, 2026-09-12): the native standup's
#: server-log tail is classified by name for the next reader, never left as
#: the generic "exited with 1" — the illegal memory access the LoRA Triton
#: kernels' int32 offsets raise on an over-ceiling warmup batch, the hybrid's
#: Mamba cache-block refusal (retried once at S // 2), a CUDA OOM, else a
#: plain engine exit.
ILLEGAL_ACCESS_RE = re.compile(r"illegal memory access", re.I)
WARMUP_RE = re.compile(r"kernel_warmup|flashinfer_autotune|_dummy_run", re.I)
CUDA_OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError", re.I)
ENGINE_FAILURE_CLASSES = ("controls_failed", "mamba_cache_blocks", "illegal_memory_access_at_warmup",
                          "illegal_memory_access", "cuda_out_of_memory", "engine_exit")
ENGINE_FAILURE_HINTS = {
    "controls_failed": (
        "a positive control FAILED AFTER readiness (job 489125: the engine warmed up, captured "
        "graphs and attested; the standup then refused) — the engine is healthy; read "
        "positive_controls.json under lora_native/ for the control's own verdict and error"),
    "illegal_memory_access_at_warmup": (
        "the pinned vLLM 0.25.1 LoRA Triton kernels compute pointer offsets in int32, and the "
        "warmup's full-budget dummy run overflows them when max_num_seqs x max_model_len x the "
        "widest adapted module exceeds 2^31-1 (gemma canary job 488932: 588,544 x 4096); the "
        f"composer now refuses such a geometry and this driver lowers S to the "
        f"{sa.LORA_KERNEL_CEILING_KEY} before its first attempt — an illegal access here means "
        "the ceiling was bypassed or another kernel overflowed"),
    "illegal_memory_access": "a CUDA illegal memory access outside the warmup path — read the server log",
    "mamba_cache_blocks": "the hybrid's Mamba cache-block count refuses the geometry under graphs; "
                          "the driver retries once at S // 2",
    "cuda_out_of_memory": "the engine ran out of GPU memory at init — the KV or graph reservation "
                          "does not fit beside the model at this geometry",
    "engine_exit": "the engine exited before readiness with no classified cause — read the server log",
}
NATIVE_ENGINE = sa.SERVING_ATTESTATION_KIND
LEGS_NATIVE = ("lora_native", "lora_native_rep", "lora_zero", "lora_none")
#: ONE ATTEMPT of this driver = ONE set of records under --out (code review
#: round 2, F1, 2026-09-12). ``ATTEMPT.json`` (the attempt's nonce, written
#: FIRST) and everything an attempt produces — the manifest, the RESULT, the
#: failure marker, the reuse decision, the prep tier and the unit — are moved
#: aside AS A WHOLE into ``<out>/stale-<utc>-<nonce8>/`` at the start of the
#: next attempt in the same out dir, before ANY fallible phase (build_context,
#: preflight, the oracle). Before this the quarantine ran inside ``run_unit``,
#: after preflight and the oracle, so a rerun that failed in either left the
#: earlier unit's ``admission.json`` / ``DONE.json`` in place and the FAILED
#: attempt's RESULT (and the report) presented that verdict as its own
#: (reproduced: ``state=FAILED exit=2 ... verdict=admitted``). The adapter bank
#: (content-addressed, reusable by design, §1.1) and earlier ``stale-*`` dirs
#: are NOT attempt records and stay.
ATTEMPT_FILE = "ATTEMPT.json"
ATTEMPT_RECORDS: Tuple[str, ...] = (ATTEMPT_FILE, "manifest.json", "RESULT", "FAILED.json",
                                    adm.UNIT_REUSE_FILE, adm.REUSED_UNIT_DIR, "prep", "unit")
#: the fused-Mamba-kernel PREFLIGHT: the fp32 oracle child is an HF load, and
#: ``models.assert_fused_mamba_kernels`` refuses a Mamba-hybrid model whose
#: interpreter lacks ``mamba_ssm`` / ``causal_conv1d`` (the standing Nemotron
#: rule — never the pure-torch SSM path). The base serving venv has neither;
#: ``multilora_cert.sbatch`` runs a Nemotron job from the lanes' ``-nemotron``
#: overlay venv (``common.sh f22_serving_venv nemotron``), and this check — the
#: same import line that recipe runs — fails the job by name before any model
#: loads, instead of the guard killing the oracle child after the GPU minutes
#: were spent. The vLLM legs may run from the same venv (the overlay is inert
#: for vLLM).
FUSED_KERNEL_PREFLIGHT = (
    "import mamba_ssm, causal_conv1d; "
    "from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; "
    "from causal_conv1d import causal_conv1d_fn; "
    "print('fused kernels OK', mamba_ssm.__version__, causal_conv1d.__version__)")


@dataclass(frozen=True)
class Target:
    #: the ROW key: the out dir ``<key>_<jobid>``, ``RESULT target=``, ``--model-key``
    key: str
    config: str
    manifest: str
    view: str
    tt_cell: str
    pin_key: str
    #: a Mamba-hybrid base: the oracle interpreter must carry the fused kernels
    fused_mamba_kernels: bool = False
    #: the OTHER fitted cells this row serves beside
    #: ``tt_cell`` (the LoRA twin of a PSR row), with their pins.json keys in the same
    #: order — every one is a canary of the fp32 oracle and a genuine route of the
    #: standup; ``tt_cell`` stays the PRIMARY cell (the throughput leg's vmodel, the
    #: run identity's ``harness_cell``)
    extra_cells: Tuple[str, ...] = ()
    #: their pin keys, pairwise. A key is either BARE — an entry of the row's own pins
    #: file under the row's model key, every stage-1 row — or QUALIFIED
    #: (:data:`PIN_SOURCE_SEP`): ``<pins file>:<model key>.<cell>``, a cell whose pin
    #: lives in ANOTHER authority (native-psr-serving §1.6: the Arditi object is pinned
    #: by ``configs/misalign_baseline_pins.json``, not by the wave's
    #: ``sdfbase_full22_pins.json``). The qualified file is resolved RELATIVE TO THE
    #: ROW'S PINS FILE'S DIRECTORY, so the two authorities travel together and ``--pins``
    #: relocates both (:func:`resolve_pin_key`)
    extra_pin_keys: Tuple[str, ...] = ()
    #: the numerics MODE (``multilora_admission.NUMERICS_MODES``): ``diagnostic`` = the
    #: probe legs + the lens per_row leg run and are recorded; ``not_measured`` =
    #: NO greedy-token agreement is measured (PSR rows): no probe
    #: leg, no per_row leg, ``output_diagnostics=False`` to the standup's controls and
    #: ``--no-output-diagnostics`` to the gate; the lens STACKED leg + throughput are
    #: KEPT (``not_slower_at_S`` compares same-job numbers)
    numerics: str = "diagnostic"
    #: the MODEL key — the geometry table (``resolve_geometry(model_key=)``),
    #: the bank view (``DEFENSES_REL / <model_key> / cells_view``), the adapter
    #: bank (``default_adapter_bank``), ``pins.json[<model_key>]``, the
    #: standup's ``model_key`` and the run identity. ``None`` at construction
    #: means "the row key IS the model key" and is filled in (native-serving-
    #: default §1.4: two rows on one model, the subtraction unit, must not
    #: leak the pin key into argv)
    model_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.model_key is None:
            object.__setattr__(self, "model_key", self.key)
        if len(self.extra_cells) != len(self.extra_pin_keys):
            raise ValueError(f"Target {self.key}: {len(self.extra_cells)} extra_cells but "
                             f"{len(self.extra_pin_keys)} extra_pin_keys")
        if self.tt_cell in self.extra_cells or len(set(self.extra_cells)) != len(self.extra_cells):
            raise ValueError(f"Target {self.key}: a cell is named twice ({self.tt_cell}, {self.extra_cells})")
        if self.numerics not in adm.NUMERICS_MODES:
            raise ValueError(f"Target {self.key}: numerics {self.numerics!r} not in {list(adm.NUMERICS_MODES)}")

    @property
    def cells(self) -> Tuple[str, ...]:
        """Every fitted cell this row serves, the primary first."""
        return (self.tt_cell, *self.extra_cells)

    @property
    def pin_keys(self) -> Tuple[str, ...]:
        return (self.pin_key, *self.extra_pin_keys)

    @property
    def measures_numerics(self) -> bool:
        return self.numerics != adm.NUMERICS_NOT_MEASURED


#: the separator of a QUALIFIED pin key (native-psr-serving §1.6):
#: ``<pins file>:<model key>.<cell>``. A BARE key is an entry of the row's own pins file
#: under the row's model key and is what every stage-1 row uses.
PIN_SOURCE_SEP = ":"


class PinRef(NamedTuple):
    """One cell's pin, RESOLVED: the digest, and exactly where it was read from."""

    sha256: str
    path: Path
    model_key: str
    key: str
    qualified: bool

    def spelling(self) -> str:
        """How the row named it (what the record and the log print)."""
        return (f"{self.path.name}{PIN_SOURCE_SEP}{self.model_key}.{self.key}"
                if self.qualified else self.key)


def resolve_pin_key(pin_key: str, *, pins_path: Path, model_key: str) -> PinRef:
    """The ``object_sha256`` a row's pin key names, and WHERE it came from.

    Two spellings, one resolution (native-psr-serving §1.6):

    * BARE — ``pins[<the row's model key>][<key>]["object_sha256"]`` of the row's own pins
      file (``--pins``, default ``configs/sdfbase_full22_pins.json``). Every stage-1 row.
    * QUALIFIED — ``<pins file>:<model key>.<cell>``: a cell whose pin belongs to ANOTHER
      authority. Stage 2's ``arditi`` canary is pinned by
      ``configs/misalign_baseline_pins.json``, the authority for the misalignment
      baselines, which the wave's pins file neither knows nor should copy — a second
      copy of a pin is a second thing to keep true.

    The qualified file is resolved RELATIVE TO THE ROW'S PINS FILE'S DIRECTORY, never to
    the repo root: the two authorities sit side by side in ``configs/`` in production, and
    a test (or an operator) that relocates ``--pins`` relocates BOTH, so a run can never
    mix a synthetic wave pin with the committed misalignment one. An absolute path is
    taken as given. Every failure is a :class:`CertRefusal` naming the file and the key —
    the pin is what binds a certification to its object, so a missing one is never a
    silent None."""
    key = str(pin_key)
    path, model, qualified = Path(pins_path), str(model_key), False
    if PIN_SOURCE_SEP in key:
        rel, _, ref = key.partition(PIN_SOURCE_SEP)
        parts = ref.split(".")
        if not rel or len(parts) != 2 or not all(parts):
            raise CertRefusal(
                f"pin key {pin_key!r} is not `<pins file>{PIN_SOURCE_SEP}<model key>.<cell>` "
                "(a qualified pin names another authority's file and entry)")
        path = Path(rel) if Path(rel).is_absolute() else Path(pins_path).parent / rel
        model, key, qualified = parts[0], parts[1], True
    if not path.is_file():
        raise CertRefusal(f"pin key {pin_key!r}: no pins file at {path}")
    try:
        sha = read_json(path)[model][key]["object_sha256"]
    except (KeyError, TypeError) as e:
        raise CertRefusal(f"{path} has no {model}/{key}/object_sha256 ({e})") from e
    if not isinstance(sha, str) or len(sha) != 64:
        raise CertRefusal(f"{path} {model}/{key}/object_sha256 is not a 64-hex digest: {sha!r}")
    return PinRef(str(sha), path, model, key, qualified)


#: stage 1: the roster main-view all-positions PSR cell
#: beside the target's LoRA cell; llama8b on the certification-only ``psrcert`` view;
#: numerics NOT measured. The vector-kind canary is ABSENT on every target: no
#: all-positions vector / vector_all / diffmean object is staged anywhere.
_PSR_CELL = "constv2-mix-full__psr_ce_all"
_PSR_PIN = "base__constv2-mix-full__psr_ce_all"
_LORA_TWIN = ("constv2-mix-full__lora_r4_ce",)
_LORA_TWIN_PIN = ("base__constv2-mix-full__lora_r4_ce",)

#: STAGE 2: the three canaries on the
#: certification-only ``psr2cert`` view of gemma4e4b — the RESPONSE-fit PSR cell (the
#: position class's own canary) as the PRIMARY, the ARDITI ``static_r`` object (the
#: constant-add path on real data, ``prefill_all`` at site ``pre``) and the LoRA twin.
#: The Arditi pin lives in the MISALIGNMENT authority, not the wave's pins file, so its
#: key is QUALIFIED (:data:`PIN_SOURCE_SEP`) — the same object the roster's cross-bank
#: mapping form selects for the view, so the certification and the view are held to ONE
#: pin. The harness id of ``arditi`` is the bare class-B id itself.
_PSR2CERT_CELL = "safety_refusal__psr_kl_all"
_PSR2CERT_PIN = "si12__safety_refusal__psr_kl_all"
_ARDITI_CELL = "arditi"
_ARDITI_PIN = "misalign_baseline_pins.json:gemma4e4b.arditi"
#: the CONSTITUTION response-fit psr — the object the 2026-09-04 lens §7.2 rows were
#: measured on (c4de7681…), hand-staged into the wave's bank on cluster A as lane
#: ``si12``; certified beside the safety cell so the view and the row stay ONE set
_CONSTITUTION_CELL = "si_constitution__psr_kl_all"
_CONSTITUTION_PIN = "si12__si_constitution__psr_kl_all"

#: THE PAIRED WRITE (relu-differenced-native-qwen, 2026-09-17): the UNDISTILLED functional
#: residual `relu(w1.z+b1) d1 - relu(w2.z+b2) d2` on the shared pre-write state (kind
#: ``psr_paired_write``, op ``relu_gated_vector_pair``, the residual slot's second gate term),
#: certified on qwen35_4b's certification-only ``pairedcert`` view. The object is the
#: batch-64 seed-1 fitted paired write (``qwen35_4b_base_eb64_s1``), pinned by THIS route's
#: cross-bank authority ``configs/reseed_paired_pins.json`` — that store is read-only and
#: the wave's pins file neither knows the object nor should copy it — so the PRIMARY cell's
#: key is QUALIFIED (the arrangement of the vector canary's row) and the wave's LoRA twin rides
#: the shared file. The pin file also names llama8b's fitted teacher (3ee2adb9..., the wave
#: pin ``derived__constv2-mix-full__psr_ce_all_minus_ctl``) for the second certification round.
_PAIRED_CELL = "constv2-mix-full__psr_ce_all_minus_ctl"
_PAIRED_PIN = f"reseed_paired_pins.json:qwen35_4b.{_PAIRED_CELL}"
#: the second certification round: llama8b's FITTED TEACHER (3ee2adb9..., the coherent virtual
#: teacher the distilled arm was fit to), which the WAVE's pins already hold under llama8b as
#: ``derived__constv2-mix-full__psr_ce_all_minus_ctl`` and the llama8b wave bank already stages —
#: so this row needs no cross-bank pin: a BARE key, like the twin's, on the certification-only
#: ``pairedcert`` view of llama8b (the psrcert view's sibling; the same LoRA twin beside it).
_PAIRED_LLAMA_PIN = "derived__constv2-mix-full__psr_ce_all_minus_ctl"

#: the pin entries are READ from pins.json at run time under the MODEL key
_LLAMA_CONFIG = "configs/sdfbase_full22_llama8b.yaml"
_LLAMA_MANIFEST = "configs/capeval/manifests/petri_full22_llama8b.yaml"
TARGETS: Dict[str, Target] = {t.key: t for t in (
    Target("gemma4e4b", "configs/sdfbase_full22_gemma4e4b.yaml",
           "configs/capeval/manifests/petri_full22_gemma4e4b.yaml", "main",
           "constv2ctl2-full__lora_r4_ce", "base__constv2ctl2-full__lora_r4_ce"),
    Target("qwen35_4b", "configs/sdfbase_full22_qwen35_4b_it.yaml",
           "configs/capeval/manifests/petri_full22_qwen35_4b_it.yaml", "it",
           "constv2-mix-full__lora_r4_ce", "itdoc__constv2-mix-full__lora_r4_ce"),
    Target("nemotron3nano4b", "configs/sdfbase_full22_nemotron3nano4b.yaml",
           "configs/capeval/manifests/petri_full22_nemotron3nano4b.yaml", "main",
           "constv2-mix-full__lora_r4_ce_minus_ctl", "derived__constv2-mix-full__lora_r4_ce_minus_ctl",
           fused_mamba_kernels=True),
    # Llama-3.1-8B (NousResearch/Meta-Llama-3.1-8B-Instruct @ d10aef79, no kernel overlay):
    # the main view's lora_all cell, and the subtraction cell as a SECOND row on the same model
    Target("llama8b", _LLAMA_CONFIG, _LLAMA_MANIFEST, "main",
           "constv2-mix-full__lora_r4_ce", "base__constv2-mix-full__lora_r4_ce"),
    Target("llama8b_subtraction", _LLAMA_CONFIG, _LLAMA_MANIFEST, "main",
           "constv2-mix-full__lora_r4_ce_minus_ctl", "derived__constv2-mix-full__lora_r4_ce_minus_ctl",
           model_key="llama8b"),
    # --- native-psr-serving stage 1: the four PSR rows (numerics not measured)
    Target("gemma4e4b_psr", "configs/sdfbase_full22_gemma4e4b.yaml",
           "configs/capeval/manifests/petri_full22_gemma4e4b.yaml", "main", _PSR_CELL, _PSR_PIN,
           extra_cells=_LORA_TWIN, extra_pin_keys=_LORA_TWIN_PIN, numerics=adm.NUMERICS_NOT_MEASURED,
           model_key="gemma4e4b"),
    Target("qwen35_4b_psr", "configs/sdfbase_full22_qwen35_4b.yaml",
           "configs/capeval/manifests/petri_full22_qwen35_4b.yaml", "main", _PSR_CELL, _PSR_PIN,
           extra_cells=_LORA_TWIN, extra_pin_keys=_LORA_TWIN_PIN, numerics=adm.NUMERICS_NOT_MEASURED,
           model_key="qwen35_4b"),
    Target("nemotron3nano4b_psr", "configs/sdfbase_full22_nemotron3nano4b.yaml",
           "configs/capeval/manifests/petri_full22_nemotron3nano4b.yaml", "main", _PSR_CELL, _PSR_PIN,
           fused_mamba_kernels=True,
           extra_cells=_LORA_TWIN, extra_pin_keys=_LORA_TWIN_PIN, numerics=adm.NUMERICS_NOT_MEASURED,
           model_key="nemotron3nano4b"),
    # llama8b: the certification-only `psrcert` view — its production views carry no PSR cell
    Target("llama8b_psr", "configs/sdfbase_full22_llama8b_psrcert.yaml",
           "configs/capeval/manifests/petri_full22_llama8b_psrcert.yaml", "psrcert", _PSR_CELL, _PSR_PIN,
           extra_cells=_LORA_TWIN, extra_pin_keys=_LORA_TWIN_PIN, numerics=adm.NUMERICS_NOT_MEASURED,
           model_key="llama8b"),
    # --- native-psr-serving stage 2 (C5): the combined certification-only view
    Target("gemma4e4b_psr2cert", "configs/sdfbase_full22_gemma4e4b_psr2cert.yaml",
           "configs/capeval/manifests/petri_full22_gemma4e4b_psr2cert.yaml", "psr2cert",
           _PSR2CERT_CELL, _PSR2CERT_PIN,
           extra_cells=(_CONSTITUTION_CELL, _ARDITI_CELL, *_LORA_TWIN),
           extra_pin_keys=(_CONSTITUTION_PIN, _ARDITI_PIN, *_LORA_TWIN_PIN),
           numerics=adm.NUMERICS_NOT_MEASURED, model_key="gemma4e4b"),
    # --- THE PAIRED WRITE (relu-differenced-native-qwen): the two-gate residual op on the
    # certification-only `pairedcert` view of qwen35_4b — the batch-64 seed-1 fitted
    # paired-write object (the bytes the campaign serves; its composition from the two parents
    # is proven bit-identical on CPU) beside the qwen LoRA twin; numerics not measured
    Target("qwen35_4b_paired", "configs/sdfbase_full22_qwen35_4b_pairedcert.yaml",
           "configs/capeval/manifests/petri_full22_qwen35_4b_pairedcert.yaml", "pairedcert",
           _PAIRED_CELL, _PAIRED_PIN,
           extra_cells=_LORA_TWIN, extra_pin_keys=_LORA_TWIN_PIN,
           numerics=adm.NUMERICS_NOT_MEASURED, model_key="qwen35_4b"),
    # --- THE PAIRED WRITE, second round: llama8b's fitted teacher on its certification-only
    # `pairedcert` view (the wave bank's own staged object, bare wave pin) beside the llama LoRA twin
    Target("llama8b_paired", "configs/sdfbase_full22_llama8b_pairedcert.yaml",
           "configs/capeval/manifests/petri_full22_llama8b_pairedcert.yaml", "pairedcert",
           _PAIRED_CELL, _PAIRED_LLAMA_PIN,
           extra_cells=_LORA_TWIN, extra_pin_keys=_LORA_TWIN_PIN,
           numerics=adm.NUMERICS_NOT_MEASURED, model_key="llama8b"),
)}
#: THE PHASES ``run_unit`` runs, in order, each with the SERVING PATH its standup takes
#: (``None`` = no standup): the fp32 oracle, the native standup + controls, the §10 gate
#: canary, the probes, the native throughput, the lens STACKED leg (the ``not_slower_at_S``
#: head-to-head's other side — every row, whatever its numerics), the lens PER_ROW leg,
#: admission. A ``not_measured`` row drops :data:`NUMERICS_ONLY_PHASES` and nothing else,
#: so EVERY row stands its cells up on BOTH paths: a cell one path cannot serve is a
#: row that cannot reach admission — caught by :func:`cell_serving_problems` in the CPU
#: preflight, before a GPU is allocated (the paired row's cell had no lens adapter and
#: would have died at phase 5 after the whole native phase ran).
UNIT_PHASES: Tuple[Tuple[str, Optional[str]], ...] = (
    ("oracle", None), ("lora_native", "lora_native"), ("gate_canary", "lora_native"),
    ("probes", "lora_native"), ("throughput_native", "lora_native"),
    ("lens_stacked", LENS_SERVING_PATH), ("lens_per_row", LENS_SERVING_PATH), ("admission", None))
NUMERICS_ONLY_PHASES: Tuple[str, ...] = ("probes", "lens_per_row")


def required_phases(target: "Target") -> Tuple[str, ...]:
    """The phases ``run_unit`` runs for ``target``, in order."""
    skip = set(NUMERICS_ONLY_PHASES) if not target.measures_numerics else set()
    return tuple(name for name, _path in UNIT_PHASES if name not in skip)


def required_serving_paths(target: "Target") -> Tuple[str, ...]:
    """The serving paths the row's REQUIRED phases stand up, in first-use order — every
    row's cells must be servable on each (``lora_native`` AND ``lens``)."""
    out: List[str] = []
    for name, path in UNIT_PHASES:
        if path is not None and name in required_phases(target) and path not in out:
            out.append(path)
    return tuple(out)


def cell_serving_problems(kind: str, resolved_spec: Optional[Mapping[str, Any]], *,
                          paths: Sequence[str] = ("lora_native", LENS_SERVING_PATH)) -> List[str]:
    """Every reason a cell of ``kind`` (with this resolved spec) cannot be stood up on one
    of ``paths`` — the ONE predicate per path: ``defense_cells.native_lora_unservable_
    reason`` for the native engine, ``lens_adapters.get_adapter`` for the lens (a kind
    with no lens adapter, or a documented-unservable one, refuses by name). Empty when
    every path serves it."""
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.serving import lens_adapters as la

    problems: List[str] = []
    for path in paths:
        if path == "lora_native":
            reason = dc.native_lora_unservable_reason(kind, resolved_spec)
            if reason is not None:
                problems.append(f"phase lora_native (the native standup) cannot serve kind {kind!r}: {reason}")
        elif path == LENS_SERVING_PATH:
            try:
                la.get_adapter(kind)
            except la.AdapterError as e:
                problems.append(f"phase lens_stacked (the not_slower_at_S head-to-head's lens side) cannot "
                                f"stand up kind {kind!r}: {e}")
        else:
            raise ValueError(f"unknown serving path {path!r}")
    return problems


#: the numerics-NOT-MEASURED rows, by name — the four stage-1 PSR rows, stage 2's combined
#: ``gemma4e4b_psr2cert`` and the paired write's ``qwen35_4b_paired`` (the sbatch's KEY table
#: mirrors them; pinned equal by test)
PSR_TARGET_KEYS: Tuple[str, ...] = tuple(k for k, t in TARGETS.items() if t.numerics == adm.NUMERICS_NOT_MEASURED)

#: THE CONFIGURATION FINGERPRINT: every
#: ``build_parser`` option is classified ONCE here — IDENTITY options change
#: what is measured or how the controls run and are bound into the unit
#: identity (``config`` + ``config_fingerprint`` in ``pre_identity``), so a
#: change reruns the indivisible unit; PATH options only say where records
#: and inputs live (what they point at is bound by digest elsewhere: the
#: object sha, the tree sha, the adapter digests); OPERATIONAL options are
#: timeouts that decide when to give up, never what is measured. The test
#: ``test_every_argparse_option_is_classified`` fails when an option is added
#: to the parser without a classification, so a new knob cannot slip past
#: the identity. Before this ``pre_identity`` bound the probe and throughput
#: settings only: a resume after lowering ``--gpu-mem-util`` 0.9 -> 0.3 reused
#: the old unit with zero native launches and read ``admitted``.
IDENTITY_OPTIONS: Tuple[str, ...] = (
    "model_key", "eval", "cell", "harness_cell", "max_model_len", "max_num_seqs", "batches",
    "window_s", "warmup_s", "repeats", "probe_max_tokens", "throughput_max_tokens",
    "control_probe_max_tokens", "gpu_mem_util", "max_logprobs", "allow_short_probe_set")
# (--admission-margin / --admission-margin-source are RETIRED with the functional rule,
# native-serving-default §1.4: the numerics gate nothing, so no margin exists to amend)
PATH_OPTIONS: Tuple[str, ...] = ("repo_dir", "out", "resume_from", "defenses_dir", "adapter_bank", "pins")
#: operational: timeouts, and --preflight-cells (a CPU-only mode that resolves the row's canary
#: set and EXITS before any unit — it never changes what a unit measures)
OPERATIONAL_OPTIONS: Tuple[str, ...] = ("ready_timeout_s", "oracle_timeout_s", "gate_timeout_s", "preflight_cells")

#: exit codes (sacct-triage; never 0 for a run that validated nothing)
RC_OK = 0
RC_FAILED = 1
RC_REFUSED = 2          # preflight / oracle / binding refusal before any standup
RC_DEAD_ENGINE = 3
RC_CONTROLS = 4         # a positive control failed (or the standup refused): no traffic


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [mlcert] {msg}", flush=True)


def classify_engine_failure(log_text: str, exc: Optional[BaseException] = None) -> Dict[str, Any]:
    """The native standup's failure, by name (:data:`ENGINE_FAILURE_CLASSES`):
    a ``PositiveControlsFailed`` (the engine came up and a control refused;
    job 489125's record said ``engine_exit`` with the "before readiness"
    hint while FAILED.json said ``controls``) is ``controls_failed`` naming
    the control; every other failure is classified from the server log —
    ``failure_class``, ``control``, the four detector bits and the class's
    hint. Never raises."""
    text = log_text or ""
    mamba = bool(MAMBA_REFUSAL_RE.search(text))
    illegal = bool(ILLEGAL_ACCESS_RE.search(text))
    at_warmup = bool(WARMUP_RE.search(text))
    oom = bool(CUDA_OOM_RE.search(text))
    control: Optional[str] = None
    # E's PositiveControlsFailed, by class OR by shape (name + the attached record): a
    # harness that loads E under a second module name hands over the same exception with
    # a different class object, and the classification must not hinge on module identity
    record = getattr(exc, "record", None)
    if exc is not None and (isinstance(exc, lsu.PositiveControlsFailed)
                            or (type(exc).__name__ == "PositiveControlsFailed"
                                and isinstance(record, Mapping))):
        cls = "controls_failed"
        control = record.get("failed_control") if isinstance(record, Mapping) else None
    elif mamba:
        cls = "mamba_cache_blocks"
    elif illegal and at_warmup:
        cls = "illegal_memory_access_at_warmup"
    elif illegal:
        cls = "illegal_memory_access"
    elif oom:
        cls = "cuda_out_of_memory"
    else:
        cls = "engine_exit"
    return {"failure_class": cls, "control": control, "mamba_refusal": mamba,
            "illegal_memory_access": illegal, "at_warmup": at_warmup, "cuda_out_of_memory": oom,
            "hint": ENGINE_FAILURE_HINTS[cls]}


class CertRefusal(RuntimeError):
    """A refusal BEFORE the unit (preflight, oracle binding, resume) -> rc 2."""


class ControlsFailed(RuntimeError):
    """A positive control failed or the native standup refused -> rc 4."""


# --------------------------------------------------------------------------- #
# atomic record writes                                                          #
# --------------------------------------------------------------------------- #

def write_json_atomic(path: Path, obj: Any) -> Path:
    """Every certification record goes through the ONE writer
    (``serving_attestation.write_json_atomic``), with ``default=str`` so
    the Paths and dataclasses a record carries serialise."""
    return sa.write_json_atomic(path, obj, default=str)


def write_done(dir_: Path, **extra: Any) -> Path:
    return write_json_atomic(Path(dir_) / "DONE.json",
                             {"written_at": time.time(), **extra})


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    """``data`` published at ``path`` whole-or-nothing: a sibling temp name,
    fsync, ``os.replace`` (the probe legs' JSONL and the verbatim copy of a
    reused oracle reference)."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# the probe set                                                                 #
# --------------------------------------------------------------------------- #

def build_probe_set(jbb_goals: Sequence[Tuple[str, str]], gate_prompt_set: Sequence[Sequence[Mapping[str, str]]],
                    probe_prompts: Sequence[str], *, max_tokens: int, jbb_revision: Optional[str]
                    ) -> Dict[str, Any]:
    """The 43-probe set: the first ``N_JBB`` JBB goals BY
    SORTED GOAL ID (the caller sorted; asserted here), the gate's frozen
    prompt set, ``served_target.PROBE_PROMPTS``; greedy at ``max_tokens``.
    ``hash`` = sha256 over the canonical JSON of the messages + the sampling."""
    ids = [g for g, _ in jbb_goals]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise CertRefusal("jbb goals must be sorted by goal id and unique")
    messages: List[List[Dict[str, str]]] = []
    sources: List[str] = []
    for gid, text in jbb_goals:
        messages.append([{"role": "user", "content": str(text)}])
        sources.append(f"jbb:{gid}")
    for i, msgs in enumerate(gate_prompt_set):
        messages.append([dict(m) for m in msgs])
        sources.append(f"gate_prompt_set:{i}")
    for i, p in enumerate(probe_prompts):
        messages.append([{"role": "user", "content": str(p)}])
        sources.append(f"probe_prompts:{i}")
    sampling = {"temperature": 0.0, "seed": 0, "max_tokens": int(max_tokens)}
    h = sha256_bytes(canonical_json({"messages": messages, "sampling": sampling}).encode())
    return {"hash": h, "n": len(messages),
            "sources": {"jbb_first32": ids, "gate_prompt_set": len(gate_prompt_set),
                        "probe_prompts": len(probe_prompts), "jbb_revision": jbb_revision},
            "max_tokens": int(max_tokens), "sampling": "greedy", "sampling_params": sampling,
            "probe_sources": sources, "messages": messages}


# --------------------------------------------------------------------------- #
# HTTP: probes and closed-loop windows through the proxy                        #
# --------------------------------------------------------------------------- #

_TOKEN_ID_RE = re.compile(r"^token_id:(\d+)$")


def _parse_token_id(tok: Any) -> Optional[int]:
    if isinstance(tok, str):
        m = _TOKEN_ID_RE.match(tok)
        if m:
            return int(m.group(1))
    return None


def probe_body(vmodel: str, messages: Sequence[Mapping[str, str]], *, max_tokens: int) -> Dict[str, Any]:
    """One greedy probe request: token ids back (``return_token_ids``), the
    first-token top-5 logprobs with tokens spelled as ids
    (``return_tokens_as_token_ids``; vLLM 0.25.1 per-request fields)."""
    return {"model": vmodel, "messages": [dict(m) for m in messages],
            "temperature": 0.0, "seed": 0, "max_tokens": int(max_tokens),
            "return_token_ids": True, "logprobs": True, "top_logprobs": 5,
            "return_tokens_as_token_ids": True}


def parse_probe_response(resp: Mapping[str, Any], *, probe_index: int, vmodel: str) -> Dict[str, Any]:
    """The §1.6 probe row from a chat completion. REFUSES a response without
    ``choices[0].token_ids`` (text can not be compared; the ids are the
    measurement) — never a silent empty row."""
    choices = resp.get("choices") or []
    if not choices:
        raise RuntimeError(f"probe {probe_index}: no choices in the response")
    ch = choices[0]
    token_ids = ch.get("token_ids")
    if not isinstance(token_ids, list):
        raise RuntimeError(f"probe {probe_index}: no token_ids on the choice (return_token_ids not "
                           f"honoured?) keys={sorted(ch)}")
    content = ((ch.get("logprobs") or {}).get("content")) or []
    first = content[0] if content else None
    top = (first or {}).get("top_logprobs") or []
    first_token = {
        "id": int(token_ids[0]) if token_ids else None,
        "logprob": (first or {}).get("logprob"),
        "token": (first or {}).get("token"),
        "top5": [[_parse_token_id(t.get("token")), t.get("logprob")] for t in top[:5]],
        "top5_tokens": [t.get("token") for t in top[:5]],
    }
    return {"probe_index": int(probe_index), "vmodel": vmodel,
            "token_ids": [int(t) for t in token_ids], "n_tokens": len(token_ids),
            "finish_reason": ch.get("finish_reason"), "first_token": first_token,
            "prompt_token_ids": resp.get("prompt_token_ids"),
            "stamps": resp.get("petri")}


async def _post_json(session: Any, url: str, body: Mapping[str, Any], headers: Mapping[str, str],
                     timeout_s: float) -> Tuple[int, Any]:
    import aiohttp
    async with session.post(url, json=body, headers=dict(headers),
                            timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
        if r.status != 200:
            return r.status, (await r.text())[:2000]
        return r.status, await r.json()


async def _probe_leg_async(proxy_url: str, api_key: str, vmodel: str,
                           probes: Sequence[Sequence[Mapping[str, str]]], *, max_tokens: int,
                           concurrency: int, fillers: int, filler_vmodel: Optional[str],
                           timeout_s: float) -> Tuple[List[Dict[str, Any]], List[str]]:
    import aiohttp
    url = proxy_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    rows: Dict[int, Dict[str, Any]] = {}
    errors: List[str] = []

    async def one(session, idx: int, msgs):
        async with sem:
            try:
                status, out = await _post_json(session, url, probe_body(vmodel, msgs, max_tokens=max_tokens),
                                               headers, timeout_s)
                if status != 200:
                    errors.append(f"probe {idx}: HTTP {status}: {out}")
                    return
                rows[idx] = parse_probe_response(out, probe_index=idx, vmodel=vmodel)
            except Exception as e:  # noqa: BLE001 — collected, never dropped
                errors.append(f"probe {idx}: {type(e).__name__}: {e}"[:500])

    async def filler(session, k: int):
        async with sem:
            try:
                body = {"model": filler_vmodel, "messages": [{"role": "user", "content": FILLER_PROMPT}],
                        "temperature": 0.0, "seed": 0, "max_tokens": int(max_tokens)}
                status, out = await _post_json(session, url, body, headers, timeout_s)
                if status != 200:
                    errors.append(f"filler {k}: HTTP {status}: {out}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"filler {k}: {type(e).__name__}: {e}"[:500])

    connector = aiohttp.TCPConnector(limit=int(concurrency) + 8)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(one(session, i, m)) for i, m in enumerate(probes)]
        tasks += [asyncio.create_task(filler(session, k)) for k in range(int(fillers))]
        await asyncio.gather(*tasks)
    return [rows[i] for i in sorted(rows)], errors


def probe_leg(proxy_url: str, api_key: str, vmodel: str, probes: Sequence[Sequence[Mapping[str, str]]], *,
              max_tokens: int, concurrency: int, fillers: int = 0, filler_vmodel: Optional[str] = None,
              timeout_s: float = 900.0) -> Tuple[List[Dict[str, Any]], List[str]]:
    """One probe leg: every probe submitted concurrently (``concurrency`` <= S,
    plus ``fillers`` filler requests to ``filler_vmodel`` for the
    batch-composition legs), rows sorted by ``probe_index``, every failure
    collected. The caller refuses a leg with any error."""
    return asyncio.run(_probe_leg_async(proxy_url, api_key, vmodel, probes, max_tokens=max_tokens,
                                        concurrency=concurrency, fillers=fillers,
                                        filler_vmodel=filler_vmodel, timeout_s=timeout_s))




def gpu_index() -> Optional[str]:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = v.split(",")[0].strip()
    return first if first.isdigit() else None


def gpu_mem_used_mib() -> Optional[int]:
    idx = gpu_index()
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if idx is not None:
        cmd += ["-i", idx]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=20).strip().splitlines()
        return int(out[0])
    except Exception:  # noqa: BLE001
        return None


def wait_gpu_free(threshold_mib: int = 6000, timeout_s: float = 180) -> Dict[str, Any]:
    """The drain wait between standups: poll our device's used memory
    until it is below ``threshold_mib`` (or the timeout, then WARN — never
    raise; the next standup's own readiness is the real check)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        used = gpu_mem_used_mib()
        if used is None or used < threshold_mib:
            return {"waited_s": time.time() - t0, "used_mib": used, "drained": True}
        time.sleep(3)
    used = gpu_mem_used_mib()
    log(f"WARNING: GPU still holds {used} MiB after {timeout_s:.0f}s")
    return {"waited_s": time.time() - t0, "used_mib": used, "drained": False}


class UtilSampler:
    """1 Hz nvidia-smi SM utilisation on our device, for one window."""

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.path: Optional[Path] = None
        self._fh: Optional[Any] = None

    def start(self, path: Path) -> None:
        idx = gpu_index()
        cmd = ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used",
               "--format=csv,noheader,nounits", "-lms", "1000"]
        if idx is not None:
            cmd += ["-i", idx]
        self.path = path
        # the handle is HELD and closed in stop(): one window per (leg, B, rep)
        # x three legs leaked one fd each before (efficiency review)
        self._fh = path.open("w")
        try:
            self.proc = subprocess.Popen(cmd, stdout=self._fh, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001 — no nvidia-smi: SM duty stays None
            self.proc = None
            self._close()

    def _close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def stop(self) -> Dict[str, Any]:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._close()
        utils: List[float] = []
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[1].replace(".", "", 1).isdigit():
                    utils.append(float(parts[1]))
        return {"sm_util_mean": statistics.fmean(utils) if utils else None, "sm_util_n": len(utils)}


def throughput_leg(proxy_url: str, api_key: str, vmodel: str, *, metrics_url: Optional[str],
                   metrics_headers: Mapping[str, str], batches: Sequence[int], repeats: int,
                   max_tokens: int, warmup_s: float, window_s: float, out_dir: Path,
                   leg: str) -> Dict[str, Any]:
    """The windows for one leg: B in ``batches`` x ``repeats`` back to back,
    a ``UtilSampler`` per window, one ``Point`` dict per window (the benchmark
    fields + ``leg``/``vmodel``/``tok_s``/``tok_s_source``). The exact-length
    flags are tried on ONE request first; a proxy that 400s them drops them
    for every window and the record says so (``exact_length_decode``); the
    metrics source is probed once and recorded (``metrics_source``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exact_ok, exact_note = _exact_length_supported(proxy_url, api_key, vmodel, max_tokens=8)
    status, m = (read_metrics(metrics_url, metrics_headers) if metrics_url else (None, {}))
    metrics_ok = "vllm:generation_tokens_total" in m
    # F2: the probe's outcome is RECORDED, never a switch. A failed first read
    # used to disable metric sampling for every window of the leg, and the
    # windows then passed the validity check on client counts alone (no
    # running_mean to judge). Every window now samples the engine's /metrics
    # itself; a window whose counter or gauge came back empty carries
    # ``tok_s None`` + ``throughput_source == "client_fallback"`` and is
    # INVALID for certification (``multilora_admission.point_invalid_reasons``);
    # its ``client_tok_s`` stays in the record as a diagnostic.
    metrics_probe = ("vllm:generation_tokens_total readable via /metrics (Bearer)" if metrics_ok
                     else f"engine /metrics unreadable at the probe (HTTP {status}); windows still sample it")
    points: List[Dict[str, Any]] = []
    n_fallback = 0
    for B in batches:
        for rep in range(int(repeats)):
            sampler = UtilSampler()
            sampler.start(out_dir / f"util_{leg}_B{B}_r{rep}.csv")
            stats, gauges = asyncio.run(closed_loop_via_proxy(
                proxy_url, api_key, vmodel, metrics_url=metrics_url,
                metrics_headers=metrics_headers, B=int(B), max_tokens=max_tokens,
                warmup_s=warmup_s, window_s=window_s, exact_length=exact_ok))
            util = sampler.stop()
            (out_dir / f"gauges_{leg}_B{B}_r{rep}.json").write_text(json.dumps(gauges))
            server_ok = stats["server_tokens"] is not None
            tok_s = stats["server_tok_s"]          # the ENGINE counter, or None — never the client count
            source = adm.THROUGHPUT_SOURCE_SERVER if server_ok else adm.THROUGHPUT_SOURCE_CLIENT_FALLBACK
            n_fallback += 0 if server_ok else 1
            pt = {"leg": leg, "vmodel": vmodel, "B": int(B), "rep": rep, "max_tokens": int(max_tokens),
                  "tok_s": tok_s, "throughput_source": source,
                  "ms_per_step": (1000.0 * B / tok_s) if tok_s else None,
                  "sm_util_mean": util["sm_util_mean"], "sm_util_n": util["sm_util_n"],
                  "metrics_probe": metrics_probe,
                  **stats}
            points.append(pt)
            log(f"  {leg} B={B:>3} r{rep}: {tok_s if tok_s is None else round(tok_s, 1)} tok/s "
                f"({source}; client {stats['client_tok_s'] and round(stats['client_tok_s'], 1)}), "
                f"{pt['ms_per_step'] and round(pt['ms_per_step'], 1)} ms/step, "
                f"running mean {pt['running_mean']}, SM {pt['sm_util_mean']}, errors {pt['errors']}")
    metrics_source = ("vllm:generation_tokens_total via /metrics (Bearer)" if n_fallback == 0 else
                      f"engine counter unreadable on {n_fallback}/{len(points)} window(s): those carry "
                      "client completion_tokens as a DIAGNOSTIC (throughput_source=client_fallback) and "
                      "are invalid for certification")
    return {"leg": leg, "vmodel": vmodel, "B_list": [int(b) for b in batches], "repeats": int(repeats),
            "max_tokens": int(max_tokens), "warmup_s": warmup_s, "window_s": window_s,
            "points": points, "metrics_source": metrics_source, "metrics_probe": metrics_probe,
            "n_client_fallback_windows": n_fallback,
            "throughput_source": (adm.THROUGHPUT_SOURCE_SERVER if n_fallback == 0
                                  else adm.THROUGHPUT_SOURCE_CLIENT_FALLBACK),
            "exact_length_decode": exact_ok, "exact_length_note": exact_note}


def _exact_length_supported(proxy_url: str, api_key: str, vmodel: str, *, max_tokens: int
                            ) -> Tuple[bool, str]:
    """ONE request with ``min_tokens`` + ``ignore_eos`` through the proxy: a
    400 naming either key -> ``(False, why)`` (natural EOS, ``running_mean``
    attests B); 200 -> ``(True, ...)``; any other failure raises (the windows
    can not run against a proxy that fails a plain request)."""
    url = proxy_url.rstrip("/") + "/chat/completions"
    body = {"model": vmodel, "messages": [{"role": "user", "content": FILLER_PROMPT}],
            "max_tokens": int(max_tokens), "min_tokens": int(max_tokens), "ignore_eos": True,
            "temperature": 0.0, "seed": 0}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
        return True, "min_tokens + ignore_eos accepted by the proxy"
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")[:500]
        if e.code == 400 and ("min_tokens" in text or "ignore_eos" in text):
            return False, f"proxy refused the exact-length keys (HTTP 400: {text}); natural EOS"
        raise RuntimeError(f"exact-length probe failed: HTTP {e.code}: {text}") from e


# --------------------------------------------------------------------------- #
# injectable seams                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class Deps:
    """Every external effect of the driver. ``production()`` binds the real
    ones lazily (nothing heavy is imported until called); the tests hand in
    fakes shaped like the real return values."""

    bring_up_lora: Callable[..., Any]
    run_positive_controls: Callable[[Any], Mapping[str, Any]]
    capctrl_bring_up: Callable[..., Any]
    attested_reduction_mode_at: Callable[[Path], Tuple[Optional[str], Optional[str]]]
    resolve_geometry: Callable[..., Any]
    run_subprocess: Callable[[Sequence[str], Path, float], int]
    probe_leg: Callable[..., Tuple[List[Dict[str, Any]], List[str]]]
    throughput_leg: Callable[..., Dict[str, Any]]
    wait_gpu_free: Callable[[], Dict[str, Any]]
    load_jbb_first: Callable[[int], Tuple[List[Tuple[str, str]], Optional[str]]]
    gate_constants: Callable[[], Tuple[Sequence[Sequence[Mapping[str, str]]], str, int]]
    probe_prompts: Callable[[], Sequence[str]]
    manifest_template_hash: Callable[[Path], str]
    versions: Callable[[], Dict[str, Optional[str]]]
    tree_sha: Callable[[Path], Optional[str]]
    snapshot_cached: Callable[[str, str], bool]
    verify_adapter_dir: Callable[[Path, Mapping[str, Any]], Optional[str]]
    served_models: Callable[[str, str], List[str]]
    hostname: Callable[[], str] = socket.gethostname
    now: Callable[[], float] = time.time
    python: str = sys.executable
    #: ``(D, "<module>")``: the widest o_proj in/out feature of the base model
    #: — the zero adapter covers EVERY o_proj layer, so this is exactly the D
    #: the standup computes from its banked adapters (job 488932)
    lora_max_module_dim: Callable[[str, str], Tuple[int, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.lora_max_module_dim is None:
            self.lora_max_module_dim = _production_lora_max_module_dim

    @classmethod
    def production(cls) -> "Deps":
        return cls(
            bring_up_lora=_production_bring_up_lora,
            run_positive_controls=_production_run_positive_controls,
            capctrl_bring_up=_production_capctrl_bring_up,
            attested_reduction_mode_at=_production_attested_reduction_mode_at,
            resolve_geometry=_production_resolve_geometry,
            run_subprocess=_production_run_subprocess,
            probe_leg=probe_leg,
            throughput_leg=throughput_leg,
            wait_gpu_free=wait_gpu_free,
            load_jbb_first=_production_load_jbb_first,
            gate_constants=_production_gate_constants,
            probe_prompts=_production_probe_prompts,
            manifest_template_hash=_production_manifest_template_hash,
            versions=_production_versions,
            tree_sha=_production_tree_sha,
            snapshot_cached=_production_snapshot_cached,
            verify_adapter_dir=_production_verify_adapter_dir,
            served_models=_production_served_models,
        )


def _production_lora_max_module_dim(model: str, revision: str) -> Tuple[int, str]:
    """The widest ``(in, out)`` over every o_proj module of ``model``
    (``lora_export.oproj_target_modules`` off the cached config — the module
    set the banked zero adapter is exported over, hence the standup's D too)."""
    from sisteer.capeval.serving import lora_export as lx
    try:
        modules = lx.oproj_target_modules(model, revision)
    except Exception as e:  # noqa: BLE001 — a LoraExportError or a missing cached config
        raise CertRefusal(f"cannot size the LoRA kernel index ceiling for {model}@{revision[:12]}: "
                          f"{type(e).__name__}: {e}") from e
    name, (d_in, d_out) = max(modules.items(), key=lambda kv: max(kv[1]))
    return max(int(d_in), int(d_out)), name


def _production_bring_up_lora(**kw: Any) -> Any:
    """THE ONE binding to the serving library's seam
    (``bring_up_lora(**kwargs) -> LoraStandup``, "the capctrl pattern: the driver
    drives standups through the same function the CLI does"). Keyword names
    follow the CLI flags; if the library's signature spells them differently,
    THIS function is the only place to reconcile."""
    return lsu.bring_up_lora(**kw)


def _production_run_positive_controls(standup: Any) -> Mapping[str, Any]:
    return lsu.run_positive_controls(standup)


def _production_capctrl_bring_up(**kw: Any) -> Any:
    """THE ONE binding to capctrl's lens standup for the head-to-head's two
    lens legs. The caller (:func:`bring_up_lens`) names the path
    (``serving_path="lens"``, :data:`LENS_SERVING_PATH`) and this binding
    REFUSES anything else: a call without the kwarg, or with another path,
    would let the standup entry points' native default (2026-09-13) turn the
    lens legs into a second native standup, and the "head-to-head" would
    compare the native engine with itself. If capctrl's ``bring_up`` does not
    accept ``serving_path`` (the native front not landed on this tree) the
    refusal names it — the path is never dropped to make the call go
    through."""
    import inspect

    import capctrl_serving_up as cs
    if kw.get("serving_path") != LENS_SERVING_PATH:
        raise CertRefusal(f"the lens leg must be brought up with serving_path={LENS_SERVING_PATH!r}, got "
                          f"{kw.get('serving_path')!r} — an implicit path would follow the standup's native "
                          "default and make the head-to-head native-vs-native")
    if "serving_path" not in inspect.signature(cs.bring_up).parameters:
        raise CertRefusal("capctrl_serving_up.bring_up accepts no serving_path (the native standup front is not "
                          "on this tree): refusing to bring up a lens leg under an implicit serving path")
    return cs.bring_up(**kw)


def _production_attested_reduction_mode_at(path: Path) -> Tuple[Optional[str], Optional[str]]:
    import capctrl_serving_up as cs
    return cs.attested_reduction_mode_at(path)


def _production_resolve_geometry(eval_name: str, *, model_key: str, max_model_len: Optional[int],
                                 max_num_seqs: Optional[int], strict_identity: bool) -> Any:
    from sisteer.capeval.serving import geometry as sg
    return sg.resolve_geometry(eval_name, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                               strict_identity=strict_identity, model_key=model_key)


def _production_run_subprocess(argv: Sequence[str], log_path: Path, timeout_s: float) -> int:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write(f"\n$ {' '.join(str(a) for a in argv)}\n")
        fh.flush()
        try:
            p = subprocess.run([str(a) for a in argv], stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout_s, cwd=str(REPO))
            return int(p.returncode)
        except subprocess.TimeoutExpired:
            fh.write(f"\n[mlcert] TIMEOUT after {timeout_s:.0f}s\n")
            return 124


def _production_load_jbb_first(n: int) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    from sisteer.eval import jailbreak_data as jd
    goals = sorted(jd.load_jbb("harmful"), key=lambda g: g.goal_id)[:n]
    return [(g.goal_id, g.text) for g in goals], jd.JBB_REVISION


def _production_gate_constants() -> Tuple[Sequence[Sequence[Mapping[str, str]]], str, int]:
    import petri_serving_gate as psg
    return psg.GATE_PROMPT_SET, psg.gate_prompt_set_hash(), int(psg.GATE_MAX_TOKENS)


def _production_probe_prompts() -> Sequence[str]:
    from sisteer.eval.attacks import served_target as stg
    return tuple(stg.PROBE_PROMPTS)


def _production_manifest_template_hash(manifest_path: Path) -> str:
    from sisteer.capeval import manifest as cm
    return str(cm.load_petri_spec(Path(manifest_path)).chat_template_hash)


def _production_versions() -> Dict[str, Optional[str]]:
    from sisteer.capeval.serving.run_identity import serving_versions
    return serving_versions(("vllm", "torch", "transformers", "vllm-lens"))


def _production_tree_sha(repo: Path) -> Optional[str]:
    from sisteer.capeval.serving.run_identity import tree_sha
    return tree_sha(repo)


def _production_snapshot_cached(model: str, revision: str) -> bool:
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(model, "config.json", revision=revision, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def _production_verify_adapter_dir(adapter_dir: Path, expected: Mapping[str, Any]) -> Optional[str]:
    """``None`` when the dir on disk hashes to the recorded digests, else why."""
    from sisteer.capeval.serving import lora_export as lx
    try:
        d = lx.adapter_digests(Path(adapter_dir))
    except Exception as e:  # noqa: BLE001
        return f"{adapter_dir}: {type(e).__name__}: {e}"
    for k in ("config_sha256", "weights_sha256", "r"):
        if expected.get(k) != getattr(d, k):
            return f"{adapter_dir}: {k} {getattr(d, k)!r} != recorded {expected.get(k)!r}"
    return None


def _production_served_models(proxy_url: str, api_key: str) -> List[str]:
    from sisteer.capeval.serving.proxy import ProxyClientEngine
    eng = ProxyClientEngine(proxy_url, identity="proxy_vllm+lora+interleaved", api_key=api_key)
    eng.load()
    return list(eng.served_models)


# --------------------------------------------------------------------------- #
# context                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class Ctx:
    args: argparse.Namespace
    deps: Deps
    target: Target
    out: Path
    prep: Path
    unit: Path
    repo: Path
    config_path: Path
    manifest_path: Path
    defenses_dir: Path
    bank_dir: Path
    pins_path: Path
    model: str
    revision: str
    tt_cell: str
    hcell: str
    object_pt: Path
    object_sha256: str
    eval_name: str
    geom: Any
    L: int
    S: int
    started_at: float
    #: native-psr-serving: EVERY fitted cell of the row (primary first) — TrainTest ids,
    #: harness ids, the pinned sha per TrainTest id and per harness id, the object per id.
    #: ``tt_cell`` / ``hcell`` / ``object_pt`` / ``object_sha256`` above are the PRIMARY's.
    cells: List[str] = field(default_factory=list)
    hcells: List[str] = field(default_factory=list)
    objmap_tt: Dict[str, str] = field(default_factory=dict)
    objmap_h: Dict[str, str] = field(default_factory=dict)
    object_pt_by_tt: Dict[str, Path] = field(default_factory=dict)
    facts: Dict[str, Any] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)
    probe_set: Dict[str, Any] = field(default_factory=dict)
    geometry_lowered: Optional[Dict[str, Any]] = None
    lens_cells: List[str] = field(default_factory=list)
    #: this run's attempt nonce (F1): stamped into every record the attempt writes
    attempt_id: Optional[str] = None

    @property
    def probes(self) -> List[List[Dict[str, str]]]:
        return self.probe_set["messages"]

    @property
    def batches(self) -> List[int]:
        return parse_batches(self.args.batches, self.S)


def parse_batches(spec: str, S: int) -> List[int]:
    """``"8,32,S"`` -> ``[8, 32, S]`` (the literal ``S`` is the served width;
    duplicates folded, anything above S dropped — the client would queue,
    not batch — and the result sorted, S always last when present)."""
    out: List[int] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        b = S if tok.upper() == "S" else int(tok)
        if b <= S and b not in out:
            out.append(b)
    return sorted(out)


def parser_option_dests(parser: Optional[argparse.ArgumentParser] = None) -> List[str]:
    """Every option ``build_parser`` defines, by ``dest`` (``help`` excluded)
    — the surface the classification above must cover exactly."""
    parser = parser or build_parser()
    return sorted(a.dest for a in parser._actions if a.dest != "help")


def config_fingerprint(args: argparse.Namespace) -> Dict[str, Any]:
    """The IDENTITY options' values (:data:`IDENTITY_OPTIONS`, in that order)
    and the sha256 of their canonical JSON. Refuses an unclassified option so
    the fingerprint can never silently omit a knob."""
    dests = set(parser_option_dests())
    classified = set(IDENTITY_OPTIONS) | set(PATH_OPTIONS) | set(OPERATIONAL_OPTIONS)
    unclassified = sorted(dests - classified)
    if unclassified:
        raise CertRefusal(f"argparse option(s) {unclassified} are not classified as IDENTITY / PATH / "
                          "OPERATIONAL (multilora_cert.IDENTITY_OPTIONS etc.) — classify them before "
                          "they can be forgotten by the unit identity")
    options = {k: getattr(args, k, None) for k in IDENTITY_OPTIONS}
    return {"options": options, "sha256": sha256_bytes(canonical_json(options).encode("utf-8"))}


def harness_cell_id(tt_cell: str) -> str:
    from sisteer.capeval.petri import registry_adapter as ra
    return ra.harness_cell_id(tt_cell)


def vmodel_of(cell_id: str) -> str:
    """``petri__base`` for ``none``, ``petri__<cell>`` otherwise — the ONE
    mapping ``defense_cells._vmodel`` writes into every route
    (``virtual_model_name`` for a canonical cell; the DERIVED control leg
    ``none__lora0`` is spelled ``VMODEL_PREFIX + cell_id`` exactly as
    ``enumerate_cells`` spells it — ``virtual_model_name`` refuses a control
    leg because it is never a manifest input, not because its vmodel differs).
    Every vmodel this returns is then held to the proxy's ``/v1/models``."""
    from sisteer.capeval.petri import defense_cells as dc
    if cell_id.endswith("__lora0") or cell_id.endswith("__lens0"):
        return dc.VMODEL_PREFIX + cell_id
    return dc.virtual_model_name(cell_id)


def build_context(args: argparse.Namespace, deps: Deps, *, attempt_id: Optional[str] = None) -> Ctx:
    import yaml

    if args.model_key not in TARGETS:
        raise CertRefusal(f"unknown --model-key {args.model_key!r}; known: {sorted(TARGETS)}")
    target = TARGETS[args.model_key]
    repo = Path(args.repo_dir).resolve()
    if not (repo / "src" / "sisteer").is_dir():
        raise CertRefusal(f"--repo-dir {repo} holds no src/sisteer")
    if repo != REPO:
        log(f"WARNING: --repo-dir {repo} differs from this driver's tree {REPO}; imports come from {REPO}")
    config_path = repo / target.config
    manifest_path = repo / target.manifest
    for p in (config_path, manifest_path):
        if not p.is_file():
            raise CertRefusal(f"missing {p}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model, revision = str(cfg["model"]), str(cfg["model_revision"])
    pins_path = Path(args.pins) if args.pins else repo / PINS_REL
    if not pins_path.is_file():
        raise CertRefusal(f"no pins file at {pins_path}")
    # pins.json is keyed by the MODEL (llama8b holds both llama rows' pins); every fitted
    # cell of the row (the primary + extra_cells) resolves its pin here
    if args.cell and target.extra_cells:
        raise CertRefusal(f"--cell cannot override the primary cell of a multi-cell row ({target.key} serves "
                          f"{list(target.cells)}); certify the row as its table names it")
    pinned_tt: Dict[str, str] = {}
    pin_refs: Dict[str, PinRef] = {}
    for tt, pin_key in zip(target.cells, target.pin_keys):
        # a BARE key reads the row's own pins file under the row's model key; a QUALIFIED
        # one names another authority beside it (the Arditi canary's misalignment pin —
        # native-psr-serving §1.6). ONE resolution, so the driver, the preflight and the
        # sbatch's pre-venv check cannot read a cell's pin from different places.
        pin_refs[tt] = resolve_pin_key(pin_key, pins_path=pins_path, model_key=target.model_key)
        pinned_tt[tt] = pin_refs[tt].sha256
    for tt, ref in pin_refs.items():
        if ref.qualified:
            log(f"cell {tt}: pin {ref.spelling()} read from {ref.path} (a separate pin authority)")
    object_sha256 = pinned_tt[target.tt_cell]
    root = lsu.shared_root()
    if args.defenses_dir:
        defenses_dir = Path(args.defenses_dir)
    elif root is not None:
        defenses_dir = root / DEFENSES_REL / target.model_key / "cells_view" / target.view
    else:
        raise CertRefusal("no --defenses-dir and no shared root ($SHARED: SISTEER_SHARED_ROOT or the site config's cluster block)")
    if args.adapter_bank:
        bank_dir = Path(args.adapter_bank)
    else:
        bank_dir = lsu.default_adapter_bank(target.model_key) or Path(args.out) / "adapter_bank"
    tt_cell = args.cell or target.tt_cell
    hcell = args.harness_cell or harness_cell_id(tt_cell)
    if hcell == tt_cell:
        log(f"note: harness id of {tt_cell} equals its TrainTest spelling")
    if args.cell and args.cell != target.tt_cell:
        pinned_tt = {tt_cell: object_sha256}   # a --cell override keeps the single-cell row shape
    cells = [tt_cell, *target.extra_cells]     # a --cell override on a multi-cell row was refused above
    hcells = [hcell] + [harness_cell_id(c) for c in cells[1:]]
    if len(set(hcells)) != len(hcells):
        raise CertRefusal(f"row {target.key}: two cells translate to one harness id ({hcells})")
    objmap_tt = {tt: pinned_tt[tt] for tt in cells}
    objmap_h = {h: objmap_tt[tt] for tt, h in zip(cells, hcells)}
    eval_name = args.eval or EVAL_DEFAULT
    geom = deps.resolve_geometry(eval_name, model_key=target.model_key, max_model_len=args.max_model_len,
                                 max_num_seqs=args.max_num_seqs, strict_identity=True)
    L, S = int(geom.max_model_len), int(geom.max_num_seqs)
    # THE LORA KERNEL INDEX CEILING, BEFORE THE FIRST ATTEMPT (sa module constants; job
    # 488932): the table's S at this eval's L against (2^31-1) // D, D the base's widest
    # o_proj (= the zero adapter's, = the standup's own D) — a table S over it is lowered to
    # ceiling // L HERE, so the unit identity, the standup (told the reason) and ALL THREE legs
    # run at one S and no GPU minute is spent finding out. The per-eval table is untouched.
    D, widest = deps.lora_max_module_dim(model, revision)
    kernel_ceiling = sa.lora_kernel_index_ceiling(D)
    geometry_lowered: Optional[Dict[str, Any]] = None
    if S * L > kernel_ceiling:
        lowered_S = kernel_ceiling // L
        if lowered_S < 1:
            raise CertRefusal(f"max_model_len={L} alone exceeds the LoRA kernel index ceiling "
                              f"{kernel_ceiling:,} tokens (D={D} at {widest}); nothing can serve it natively")
        geom = deps.resolve_geometry(eval_name, model_key=target.model_key, max_model_len=L,
                                     max_num_seqs=lowered_S, strict_identity=False)
        geometry_lowered = {"from": S, "to": lowered_S, "reason": sa.LORA_KERNEL_CEILING_REASON,
                            "L": L, "D": D, "widest_module": widest, "ceiling": kernel_ceiling,
                            "product_table": S * L, "product": lowered_S * L}
        log(f"geometry LOWERED to the LoRA kernel index ceiling before the first attempt: S {S} -> "
            f"{lowered_S} at L={L} (D={D} at {widest}; ceiling {kernel_ceiling:,} tokens; the table's "
            f"{S} x {L} = {S * L:,} would overflow the int32 kernel offsets)")
        S = lowered_S
    out = Path(args.out)
    ctx = Ctx(args=args, deps=deps, target=target, out=out, prep=out / "prep", unit=out / "unit",
              repo=repo, config_path=config_path, manifest_path=manifest_path,
              defenses_dir=defenses_dir, bank_dir=bank_dir, pins_path=pins_path,
              model=model, revision=revision, tt_cell=tt_cell, hcell=hcell,
              object_pt=defenses_dir / tt_cell / "object.pt", object_sha256=object_sha256,
              eval_name=eval_name, geom=geom, L=L, S=S,
              started_at=deps.now(), attempt_id=(attempt_id or secrets.token_hex(16)),
              geometry_lowered=geometry_lowered,
              cells=cells, hcells=hcells, objmap_tt=objmap_tt, objmap_h=objmap_h,
              object_pt_by_tt={tt: defenses_dir / tt / "object.pt" for tt in cells})
    ctx.facts[sa.LORA_KERNEL_CEILING_KEY] = {"D": D, "widest_module": widest, "ceiling": kernel_ceiling,
                                            "L": L, "S": S, "product": S * L,
                                            "lowered": geometry_lowered is not None}
    # the lens legs serve the prompted baseline ONLY when the config lists it
    # (capctrl.bring_up refuses an absent cell; the admission rule is the
    # lora-native standup's, not the lens standup's)
    ctx.lens_cells = list(cells) + ([PROMPTED_CELL] if PROMPTED_CELL in (cfg.get("explicit_cells") or []) else [])
    return ctx


def _stamp(ctx: Ctx) -> Dict[str, Any]:
    """Where and by WHICH ATTEMPT a record was written: the node, the Slurm
    job, and the attempt nonce (F1) — so the manifest, the unit identity, the
    unit's DONE marker and CANARY_OK all name the attempt that wrote them."""
    return {"node": ctx.deps.hostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            adm.ATTEMPT_ID_KEY: ctx.attempt_id}


def mark_phase(ctx: Ctx, name: str, state: str, **updates: Any) -> None:
    """Merge one phase's state into the manifest's ``phases`` map (beside any
    other ``updates``) and rewrite the manifest."""
    write_manifest(ctx, phases={**ctx.manifest.get("phases", {}), name: state}, **updates)


def write_manifest(ctx: Ctx, **updates: Any) -> None:
    ctx.manifest.update(updates)
    base = {
        "target": ctx.target.key, "model_key": ctx.target.model_key, "model": ctx.model, "revision": ctx.revision,
        "tree_sha": ctx.facts.get("tree_sha"), "cell": ctx.tt_cell, "tt_cell": ctx.tt_cell,
        "harness_cell": ctx.hcell, "config": str(ctx.config_path), "manifest": str(ctx.manifest_path),
        "defenses_dir": str(ctx.defenses_dir), "adapter_bank": str(ctx.bank_dir),
        "object_sha256": ctx.object_sha256, "eval": ctx.eval_name,
        # native-psr-serving: every fitted cell of the row and the numerics mode
        "cells": list(ctx.cells), "harness_cells": list(ctx.hcells),
        "object_artifact_sha256_by_cell": dict(ctx.objmap_h), "numerics": ctx.target.numerics,
        "geometry": {"L": ctx.L, "S": ctx.S, "seqs_source": getattr(ctx.geom, "seqs_source", None),
                     "window_source": getattr(ctx.geom, "window_source", None)},
        "geometry_lowered": ctx.geometry_lowered,
        **_stamp(ctx),
        "started_at": ctx.started_at, "resume_from": ctx.args.resume_from,
        # defaults; whatever ctx.manifest holds overrides them below
        "prep_reused": {"oracle": None}, "unit_reused_from": None,
        "prior_attempt_moved_to": None, "prior_unit_moved_to": None,
        "lens_cells": ctx.lens_cells, "argv": sys.argv, "driver_tree": str(REPO),
        "facts": ctx.facts,
    }
    base.update(ctx.manifest)
    write_json_atomic(ctx.out / "manifest.json", base)


# --------------------------------------------------------------------------- #
# phase 0: preflight                                                            #
# --------------------------------------------------------------------------- #

def phase_preflight(ctx: Ctx) -> None:
    d = ctx.deps
    ctx.facts["tree_sha"] = d.tree_sha(ctx.repo)
    ctx.facts["versions"] = d.versions()
    ctx.facts["python"] = d.python
    reqs = ctx.repo / REQUIREMENTS_REL
    ctx.facts["venv_requirements_hash"] = sha256_file(reqs) if reqs.is_file() else None
    # the MANIFEST's 16-hex chat_template_hash (the oracle reference's key of
    # the same name; the standup's 64-hex chat_template_sha256 is recorded in
    # the unit identity once the endpoint exists)
    ctx.facts["chat_template_hash"] = d.manifest_template_hash(ctx.manifest_path)
    problems: List[str] = []
    on_disk: Dict[str, Optional[str]] = {}
    for tt in ctx.cells:
        obj, want = ctx.object_pt_by_tt[tt], ctx.objmap_tt[tt]
        if not obj.is_file():
            problems.append(f"no object at {obj}")
            on_disk[tt] = None
            continue
        have = sha256_file(obj)
        on_disk[tt] = have
        if have != want:
            problems.append(f"{obj} sha256 {have} != pin {want}")
    ctx.facts["object_sha256_on_disk"] = on_disk[ctx.tt_cell]
    ctx.facts["object_sha256_on_disk_by_cell"] = dict(on_disk)
    if not d.snapshot_cached(ctx.model, ctx.revision):
        problems.append(f"pinned snapshot not in the local HF cache: {ctx.model}@{ctx.revision}")
    if not ctx.facts["versions"].get("vllm"):
        problems.append("vllm is not importable in this venv")
    if ctx.facts["venv_requirements_hash"] is None:
        problems.append(f"no {REQUIREMENTS_REL} in {ctx.repo}")
    if not (ctx.repo / GATE_SCRIPT_REL).is_file():
        problems.append(f"no {GATE_SCRIPT_REL} in {ctx.repo}")
    # the probe set
    goals, jbb_rev = d.load_jbb_first(N_JBB)
    gate_prompts, gate_hash, gate_max = d.gate_constants()
    ctx.facts["gate_prompt_set_hash"] = gate_hash
    ctx.facts["gate_max_tokens"] = gate_max
    ctx.probe_set = build_probe_set(goals, gate_prompts, d.probe_prompts(),
                                    max_tokens=int(ctx.args.probe_max_tokens), jbb_revision=jbb_rev)
    if ctx.probe_set["n"] < 43 and not ctx.args.allow_short_probe_set:
        problems.append(f"probe set has {ctx.probe_set['n']} probes (< 43); --allow-short-probe-set to accept")
    if ctx.probe_set["n"] > ctx.S:
        problems.append(f"probe set {ctx.probe_set['n']} > S {ctx.S}: the probes could not be one batch")
    mark_phase(ctx, "preflight", "running")
    if problems:
        write_json_atomic(ctx.prep / "preflight" / "FAILED.json", {"problems": problems})
        raise CertRefusal("preflight: " + "; ".join(problems))
    write_done(ctx.prep / "preflight", facts=ctx.facts, probe_set_hash=ctx.probe_set["hash"])
    mark_phase(ctx, "preflight", "done")
    log(f"preflight OK: {ctx.target.key} {ctx.model}@{ctx.revision[:12]} cell={ctx.tt_cell} -> {ctx.hcell} "
        f"object={ctx.object_sha256[:12]} geometry {ctx.L}x{ctx.S} ({getattr(ctx.geom, 'seqs_source', '?')}) "
        f"probes={ctx.probe_set['n']} hash={ctx.probe_set['hash'][:12]}")


# --------------------------------------------------------------------------- #
# phase 0b: the fp32 oracle                                                     #
# --------------------------------------------------------------------------- #

def oracle_bindings_problems(ref: Mapping[str, Any], *, model: str, revision: str,
                             chat_template_hash: str, gate_prompt_set_hash: str, gate_max_tokens: int,
                             hcell: str, object_sha256: str, tree_sha: Optional[str] = None,
                             objmap: Optional[Mapping[str, str]] = None) -> List[str]:
    """The §1.6 / §5 binding of a ``reference.json`` to THIS run (empty =
    bound): ``cell_id_space == "harness"``; model, revision, template,
    prompt-set hash, ``gate_max_tokens``; ``objects`` keyed by EXACTLY the
    row's harness cells (``objmap``: {harness cell: pinned sha}; the legacy
    pair ``hcell`` / ``object_sha256`` when no ``objmap`` is given) with each
    one's pinned sha; an ``arms`` entry for each; a non-empty ``base``; no
    ``error``; and, when given, the ``tree_sha`` (resume only).

    The template binding is the MANIFEST's 16-hex ``chat_template_hash``
    (``cm.load_petri_spec(manifest).chat_template_hash``), held to the
    reference's key of the SAME name: ``gate-worker-reference`` records both
    the 64-hex ``chat_template_sha256`` over the shim tokenizer's template
    bytes and its 16-hex prefix as ``chat_template_hash``; the standup's
    ``endpoint.json`` carries both spellings too, and the gate canary binds
    the 64-hex there. Comparing the 16-hex to the 64-hex key would refuse
    every honest reference."""
    p: List[str] = []
    if not isinstance(ref, Mapping):
        return ["reference is not a JSON object"]
    if ref.get("error"):
        p.append(f"reference records an error: {str(ref.get('error'))[:200]}")
    if ref.get("cell_id_space") != "harness":
        p.append(f"cell_id_space {ref.get('cell_id_space')!r} != 'harness'")
    for k, want in (("model", model), ("revision", revision), ("chat_template_hash", chat_template_hash),
                    ("gate_prompt_set_hash", gate_prompt_set_hash), ("gate_max_tokens", gate_max_tokens)):
        if ref.get(k) != want:
            p.append(f"{k} {ref.get(k)!r} != {want!r}")
    want = dict(objmap) if objmap is not None else {hcell: object_sha256}
    objs = ref.get("objects")
    if not isinstance(objs, Mapping) or set(objs) != set(want):
        p.append(f"objects keys {sorted(objs) if isinstance(objs, Mapping) else objs!r} != {sorted(want)}")
    else:
        for h in sorted(want):
            got = objs[h].get("sha256") if isinstance(objs[h], Mapping) else None
            if got != want[h]:
                p.append(f"objects[{h}].sha256 {got!r} != pin {want[h]!r}")
    arms = ref.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(want):
        p.append(f"arms keys {sorted(arms) if isinstance(arms, Mapping) else arms!r} != {sorted(want)}")
    if not isinstance(ref.get("base"), Mapping) or not ref.get("base"):
        p.append("base reference missing or empty")
    if tree_sha is not None and ref.get("tree_sha") != tree_sha:
        p.append(f"tree_sha {ref.get('tree_sha')!r} != {tree_sha!r}")
    return p


def _oracle_expectations(ctx: Ctx) -> Dict[str, Any]:
    return dict(model=ctx.model, revision=ctx.revision,
                chat_template_hash=ctx.facts["chat_template_hash"],
                gate_prompt_set_hash=ctx.facts["gate_prompt_set_hash"],
                gate_max_tokens=int(ctx.facts["gate_max_tokens"]),
                hcell=ctx.hcell, object_sha256=ctx.object_sha256, objmap=dict(ctx.objmap_h))


def phase_oracle(ctx: Ctx) -> None:
    """Reuse ``<resume_from>/prep/oracle`` iff its DONE.json exists and every
    binding (incl. ``tree_sha``) matches; else run the oracle child with the
    GPU free and hold its output to the same bindings (minus tree_sha, which
    it records itself). Refuses to leave without ``prep/oracle/DONE.json``."""
    odir = ctx.prep / "oracle"
    odir.mkdir(parents=True, exist_ok=True)
    ref_path = odir / "reference.json"
    preflight_fused_kernels(ctx, odir)
    reused = None
    if ctx.args.resume_from:
        prior = Path(ctx.args.resume_from) / "prep" / "oracle"
        if (prior / "DONE.json").is_file() and (prior / "reference.json").is_file():
            try:
                pref = read_json(prior / "reference.json")
                probs = oracle_bindings_problems(pref, tree_sha=ctx.facts.get("tree_sha"), **_oracle_expectations(ctx))
            except (ValueError, OSError) as e:
                probs = [f"unreadable: {e}"]
            if not probs:
                # the BYTES, verbatim (a re-serialisation would change the
                # reference's sha256, which the unit identity carries)
                _write_bytes_atomic(ref_path, (prior / "reference.json").read_bytes())
                reused = str(prior)
                log(f"oracle: reusing {prior} (every binding matches)")
            else:
                log(f"oracle: NOT reusing {prior}: {probs}")
                write_json_atomic(odir / "reuse_refused.json", {"prior": str(prior), "problems": probs})
        else:
            log(f"oracle: no complete prior oracle under {prior}")
    if reused is None:
        objects_json = odir / "objects.json"
        write_json_atomic(objects_json, {h: str(ctx.object_pt_by_tt[tt]) for tt, h in zip(ctx.cells, ctx.hcells)})
        argv = [ctx.deps.python, str(ctx.repo / GATE_SCRIPT_REL), "gate-worker-reference",
                "--manifest", str(ctx.manifest_path),
                "--objects-json", str(objects_json)]
        for tt, h in zip(ctx.cells, ctx.hcells):
            argv += ["--expect-object-sha256", f"{h}={ctx.objmap_tt[tt]}"]
        argv += ["--out", str(ref_path)]
        log("oracle: " + " ".join(argv))
        t0 = ctx.deps.now()
        rc = ctx.deps.run_subprocess(argv, odir / "oracle.log", float(ctx.args.oracle_timeout_s))
        elapsed = ctx.deps.now() - t0
        if rc != 0 or not ref_path.is_file():
            raise CertRefusal(f"oracle child rc={rc} (log {odir / 'oracle.log'}); no reference written")
        ref = read_json(ref_path)
        probs = oracle_bindings_problems(ref, **_oracle_expectations(ctx))
        if probs:
            write_json_atomic(odir / "FAILED.json", {"problems": probs})
            raise CertRefusal("oracle reference does not bind to this run: " + "; ".join(probs))
        ctx.facts["oracle_elapsed_s"] = elapsed
    ctx.facts["oracle_reference_sha256"] = sha256_file(ref_path)
    write_done(odir, reused_from=reused, reference_sha256=ctx.facts["oracle_reference_sha256"],
               bindings=_oracle_expectations(ctx))
    mark_phase(ctx, "oracle", "done", prep_reused={"oracle": reused})


def preflight_fused_kernels(ctx: Ctx, odir: Path) -> None:
    """Finding 7: on a Mamba-hybrid target (``Target.fused_mamba_kernels``) the
    interpreter that will run the fp32 oracle child (``ctx.deps.python`` — the
    driver's own) must import the fused kernels BEFORE any model loads,
    exactly the check ``common.sh f22_serving_venv nemotron`` runs
    (:data:`FUSED_KERNEL_PREFLIGHT`). A failure is a :class:`CertRefusal`
    naming the venv to run from; the guard in ``models.py`` is never weakened.
    Recorded under ``prep/oracle/kernel_preflight.json`` either way."""
    if not ctx.target.fused_mamba_kernels:
        return
    argv = [ctx.deps.python, "-c", FUSED_KERNEL_PREFLIGHT]
    log(f"fused-kernel preflight ({ctx.target.key}): {ctx.deps.python} -c '<import mamba_ssm, causal_conv1d>'")
    rc = ctx.deps.run_subprocess(argv, odir / "kernel_preflight.log", 300.0)
    write_json_atomic(odir / "kernel_preflight.json",
                      {"python": ctx.deps.python, "rc": rc, "check": FUSED_KERNEL_PREFLIGHT,
                       "target": ctx.target.key, "checked_at": ctx.deps.now()})
    ctx.facts["fused_kernel_preflight_rc"] = rc
    if rc != 0:
        raise CertRefusal(
            f"{ctx.target.key} is a Mamba-hybrid target but the oracle interpreter {ctx.deps.python} "
            f"cannot import mamba_ssm / causal_conv1d (rc={rc}; log {odir / 'kernel_preflight.log'}) "
            "— HFEngine.load() would be refused by models.assert_fused_mamba_kernels after the GPU "
            "minutes were spent. Run this job from the lanes' -nemotron overlay venv "
            "(multilora_cert.sbatch bootstraps sisteer-petri-vllm-nemotron-<sha256[:8] of "
            "vllm-requirements.txt + vllm-requirements-nemotron.txt>); the guard is not weakened")


def require_oracle_done(ctx: Ctx) -> None:
    """The gate to phase 1: ``prep/oracle/DONE.json`` present and the artifact
    bound (called by ``run_unit`` FIRST; pinned by the ordering test)."""
    odir = ctx.prep / "oracle"
    if not (odir / "DONE.json").is_file() or not (odir / "reference.json").is_file():
        raise CertRefusal(f"refusing to enter the unit: no complete oracle under {odir}")
    probs = oracle_bindings_problems(read_json(odir / "reference.json"), **_oracle_expectations(ctx))
    if probs:
        raise CertRefusal("refusing to enter the unit: oracle unbound: " + "; ".join(probs))


# --------------------------------------------------------------------------- #
# identity + resume                                                             #
# --------------------------------------------------------------------------- #

IDENTITY_UNBOUND_FIELDS = ("node", "slurm_job_id", "started_at", "adapter_digests", "written_at",
                           adm.ATTEMPT_ID_KEY)


def pre_identity(ctx: Ctx) -> Dict[str, Any]:
    """Every identity field known BEFORE the native standup — what a prior
    unit is compared against (§5: all fields but node / job / started_at;
    the adapter digests are re-verified on disk instead, see
    :func:`unit_reuse_decision`)."""
    v = ctx.facts.get("versions", {})
    return {
        "schema": "multilora_cert.identity/1",
        "target": ctx.target.key, "model_key": ctx.target.model_key, "model": ctx.model, "revision": ctx.revision,
        "tree_sha": ctx.facts.get("tree_sha"), "tt_cell": ctx.tt_cell, "harness_cell": ctx.hcell,
        "cells": list(ctx.cells), "harness_cells": list(ctx.hcells), "numerics": ctx.target.numerics,
        "object_artifact_sha256_by_cell": dict(ctx.objmap_h),
        "eval": ctx.eval_name,
        "geometry": {"L": ctx.L, "S": ctx.S},
        "geometry_lowered": ctx.geometry_lowered,
        "probe_set_hash": ctx.probe_set["hash"], "n_probes": ctx.probe_set["n"],
        "sampling": "greedy", "max_tokens": int(ctx.args.probe_max_tokens),
        "throughput": {"B_list": ctx.batches, "max_tokens": int(ctx.args.throughput_max_tokens),
                       "window_s": float(ctx.args.window_s), "warmup_s": float(ctx.args.warmup_s),
                       "repeats": int(ctx.args.repeats)},
        "lens_cells": list(ctx.lens_cells),
        "vllm_version": v.get("vllm"), "torch_version": v.get("torch"),
        "transformers_version": v.get("transformers"), "vllm_lens_version": v.get("vllm-lens"),
        "venv_requirements_hash": ctx.facts.get("venv_requirements_hash"),
        "oracle_reference_sha256": ctx.facts.get("oracle_reference_sha256"),
        "chat_template_hash": ctx.facts.get("chat_template_hash"),
        # A3: every measurement- or control-affecting option, by name and as one digest
        "config": config_fingerprint(ctx.args)["options"],
        "config_fingerprint": config_fingerprint(ctx.args)["sha256"],
    }


def identity_bound_view(identity: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in identity.items() if k not in IDENTITY_UNBOUND_FIELDS}


def snapshot_unit(prior: Path) -> Tuple[Dict[str, bytes], List[str]]:
    """The source unit's WHOLE evidence inventory read ONCE, verbatim
    (``adm.read_unit_snapshot``: ``INVENTORY.json``, ``DONE.json`` and every
    inventoried file), then held to the inventory's binding
    (``adm.unit_inventory_problems_from_bytes``: DONE's sha over the
    inventory, every digest, no unlisted file, no required evidence missing).
    Every check the reuse decision makes and the copy the acceptance keeps
    work on THESE bytes, so what this attempt validated is exactly what it
    presents — a source that changes underneath (a rerun in place) cannot
    slip a different record past the validation, and a summaries-only source
    (A2) is refused by naming what it lacks."""
    snap, present, problems = adm.read_unit_snapshot(Path(prior))
    if problems:
        return snap, problems
    return snap, adm.unit_inventory_problems_from_bytes(snap, present=present)


def unit_reuse_decision(ctx: Ctx) -> Tuple[Optional[Path], List[str], Dict[str, bytes]]:
    """§5: reuse the prior unit WHOLE iff ``<prior>/unit/DONE.json`` exists,
    its ``identity.json``'s bound view equals this run's, its records bind to
    ONE attempt, and every recorded adapter digest still verifies against its
    directory on disk. Returns ``(prior_unit_dir | None, reasons, snapshot)``
    — the snapshot is the source's records read ONCE (:func:`snapshot_unit`),
    on which every check here ran and which :func:`accept_reused_unit` copies
    verbatim; empty when not reused."""
    if not ctx.args.resume_from:
        return None, ["no --resume-from"], {}
    prior = Path(ctx.args.resume_from) / "unit"
    reasons: List[str] = []
    if not (prior / "DONE.json").is_file():
        return None, [f"{prior} has no DONE.json (a partial unit is never reused)"], {}
    if not (prior / "identity.json").is_file():
        return None, [f"{prior} has no identity.json"], {}
    snap, problems = snapshot_unit(prior)
    if problems:
        return None, [f"prior unit inventory: {p}" for p in problems], {}
    try:
        pid = json.loads(snap["identity.json"].decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        return None, [f"{prior}/identity.json unreadable: {e}"], {}
    if not isinstance(pid, Mapping):
        return None, [f"{prior}/identity.json is not a JSON object"], {}
    if pid.get("complete") is not True:
        reasons.append(f"{prior}/identity.json is not marked complete")
    mine = identity_bound_view(pre_identity(ctx))
    theirs = identity_bound_view(pid)
    # every field THIS run can state before a standup must match the prior's
    # (the prior carries the standup-derived extras beside them)
    for k in sorted(mine):
        if mine.get(k) != theirs.get(k):
            reasons.append(f"identity.{k}: prior {theirs.get(k)!r} != this run {mine.get(k)!r}")
    # F1: a reused unit is presented under its own provenance, so its three
    # attempt-bound records must agree on the attempt that produced them and
    # the admission must be the one that attempt completed with — judged on
    # the SNAPSHOT bytes, the ones the acceptance will copy
    reasons += [f"prior unit binding: {p}" for p in adm.unit_binding_problems_from_bytes(snap)]
    digests = pid.get("adapter_digests")
    if not isinstance(digests, Mapping) or not digests:
        reasons.append("prior identity records no adapter digests")
    else:
        for name, dg in digests.items():
            d = (dg or {}).get("dir")
            if not d:
                reasons.append(f"adapter {name}: no dir recorded")
                continue
            why = ctx.deps.verify_adapter_dir(Path(d), dg)
            if why:
                reasons.append(f"adapter {name}: {why}")
    if reasons:
        return None, reasons, {}
    return prior, [], snap


def accept_reused_unit(ctx: Ctx, prior: Path, snap: Mapping[str, bytes]) -> Dict[str, Any]:
    """Make the reuse SELF-CONTAINED (the F1 follow-up): write the snapshot's
    bytes verbatim under ``<out>/reused_unit/`` and record the acceptance in
    ``unit_reuse.json`` — this attempt's id, the source path (provenance
    only), the accepted source ``attempt_id``, and a sha256 per copied file —
    then hold the result to the reader's own rule
    (``adm.reused_unit_binding_problems``). Every later reader (the RESULT,
    the report) resolves the reused unit HERE, never at ``unit_reused_from``,
    which a rerun of the source replaces. Returns the accepted admission."""
    copy = ctx.out / adm.REUSED_UNIT_DIR
    copy.mkdir(parents=True, exist_ok=True)
    for name, data in snap.items():
        (copy / name).parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_atomic(copy / name, data)
    src_done = json.loads(snap["DONE.json"].decode("utf-8"))
    src_ident = json.loads(snap["identity.json"].decode("utf-8"))
    src_adm = json.loads(snap["admission.json"].decode("utf-8"))
    digests = {name: sha256_bytes(data) for name, data in snap.items()}
    record = {
        "unit_reused_from": str(prior), "reasons": [], adm.ATTEMPT_ID_KEY: ctx.attempt_id,
        "accepted_at": ctx.deps.now(),
        "source": {adm.ATTEMPT_ID_KEY: src_done.get(adm.ATTEMPT_ID_KEY),
                   adm.DONE_ADMISSION_SHA_KEY: digests["admission.json"],
                   "sha256": digests, "files": sorted(snap),
                   "verdict": src_adm.get("verdict"),
                   "node": src_ident.get("node"), "slurm_job_id": src_ident.get("slurm_job_id"),
                   "started_at": src_ident.get("started_at"),
                   "unit_dir": str(prior), "copied_to": str(copy), "n_files": len(snap)},
        "note": "the reused unit is read at reused_unit/ (this record's verbatim copy of the WHOLE "
                "inventory), bound to source.attempt_id and source.sha256; unit_reused_from is "
                "provenance only; paths the copied documents name under source.unit_dir resolve "
                "to source.copied_to (adm.resolve_reused_path)",
    }
    write_json_atomic(ctx.out / adm.UNIT_REUSE_FILE, record)
    probs = adm.reused_unit_binding_problems(ctx.out, record_attempt_id=ctx.attempt_id,
                                             unit_reused_from=str(prior))
    if probs:
        raise RuntimeError(f"the accepted reuse of {prior} does not bind at {copy}: {probs}")
    log(f"unit REUSED from {prior} (identity match; source attempt "
        f"{str(record['source'][adm.ATTEMPT_ID_KEY])[:8]}) — copied verbatim to {copy}")
    return src_adm


# --------------------------------------------------------------------------- #
# the unit                                                                      #
# --------------------------------------------------------------------------- #

class _Standups:
    """Whatever is live, so a failure anywhere tears it down."""

    def __init__(self) -> None:
        self.live: List[Any] = []

    def add(self, s: Any) -> Any:
        self.live.append(s)
        return s

    def stop(self, s: Any) -> None:
        try:
            s.tear_down()
        except Exception as e:  # noqa: BLE001
            log(f"tear_down raised {type(e).__name__}: {e}")
        if s in self.live:
            self.live.remove(s)

    def stop_all(self) -> None:
        for s in list(self.live):
            self.stop(s)


def _server_log_tail(out_dir: Path, n: int = 4000) -> str:
    """The last ``n`` bytes of the standup's server log (seeked to, never the
    whole multi-MB log decoded), or "" when there is none."""
    p = Path(out_dir) / lsu.SERVER_LOG_FILE
    if not p.exists():
        return ""
    with open(p, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - n))
        return fh.read().decode("utf-8", errors="replace")


def native_standup_kwargs(ctx: Ctx, *, out: Path, S: int, override_reason: Optional[str]) -> Dict[str, Any]:
    """The §1.5 call, keyword-for-flag (``--cell`` twice: the fitted cell and
    the prompted baseline; ``--expect-object-sha256 TT_CELL=SHA``; the eval
    for the geometry table; the explicit S — the table's, or the lowered one
    with ``geometry_override_reason``)."""
    kw: Dict[str, Any] = dict(
        repo_dir=str(ctx.repo), config=str(ctx.config_path), manifest=str(ctx.manifest_path),
        defenses_dir=str(ctx.defenses_dir), out=str(out), model_key=ctx.target.model_key,
        eval=ctx.eval_name, cell=[*ctx.cells, PROMPTED_CELL],
        expect_object_sha256=[f"{tt}={ctx.objmap_tt[tt]}" for tt in ctx.cells],
        adapter_bank=str(ctx.bank_dir), max_model_len=ctx.L, max_num_seqs=int(S),
        gpu_mem_util=str(ctx.args.gpu_mem_util), max_logprobs=int(ctx.args.max_logprobs),
        ready_timeout_s=int(ctx.args.ready_timeout_s), probe_max_tokens=int(ctx.args.control_probe_max_tokens),
    )
    if override_reason is not None:
        kw["geometry_override_reason"] = override_reason
    if not ctx.target.measures_numerics:
        # native-psr-serving §1.6: the standup's controls run with the output
        # diagnostics OFF (stated here AND decided by the core's central policy)
        kw["output_diagnostics"] = False
    return kw


def bring_up_native(ctx: Ctx, standups: _Standups) -> Tuple[Any, Dict[str, Any]]:
    """Phase 1's launch with the ONE hybrid retry: an engine-init refusal whose
    server-log tail names ``Mamba cache block`` -> retry once at ``S // 2`` with
    the reason recorded (``geometry_lowered``); the table is never touched."""
    out = ctx.unit / "lora_native"
    attempts: List[Dict[str, Any]] = []
    S = ctx.S
    # a pre-unit lowering (the kernel index ceiling) is a NON-identity S: the
    # standup must be told the reason, or its strict guard refuses it
    reason: Optional[str] = (ctx.geometry_lowered or {}).get("reason")
    for attempt in range(2):
        out.mkdir(parents=True, exist_ok=True)
        kw = native_standup_kwargs(ctx, out=out, S=S, override_reason=reason)
        log(f"native standup attempt {attempt + 1}: S={S}" + (f" ({reason})" if reason else ""))
        t0 = ctx.deps.now()
        try:
            standup = ctx.deps.bring_up_lora(**kw)
        except Exception as e:  # noqa: BLE001
            # classify over a wide window (the root cause sits hundreds of lines
            # above the API server's final traceback); record the last 4000 bytes
            text = _server_log_tail(out, n=400_000)
            tail = text[-4000:]
            rec = {"attempt": attempt + 1, "S": S, "error": f"{type(e).__name__}: {e}"[:1000],
                   "elapsed_s": ctx.deps.now() - t0, **classify_engine_failure(text, e)}
            attempts.append(rec)
            write_json_atomic(ctx.unit / "native_standup_attempts.json", attempts)
            if attempt == 0 and rec["mamba_refusal"]:
                S_new = max(1, S // 2)
                ctx.geometry_lowered = {"from": ctx.S, "to": S_new, "reason": MAMBA_LOWER_REASON,
                                        "evidence": tail[-1500:], "after": ctx.geometry_lowered}
                write_manifest(ctx, geometry_lowered=ctx.geometry_lowered)
                log(f"Mamba cache-block refusal at S={S}; retrying ONCE at S={S_new}")
                ctx.deps.wait_gpu_free()
                S, reason = S_new, MAMBA_LOWER_REASON
                # the retried standup gets a fresh out dir under the same name
                stale = out.with_name(f"lora_native.attempt1_S{ctx.S}")
                if out.exists():
                    out.rename(stale)
                continue
            what = rec["failure_class"] + (f": {rec['control']}" if rec.get("control") else "")
            raise ControlsFailed(f"native standup refused ({what}): {rec['error']}\n"
                                 f"{rec['hint']}\n--- server log tail ---\n{tail}") from e
        standups.add(standup)
        attempts.append({"attempt": attempt + 1, "S": S, "ready_s": ctx.deps.now() - t0, "ok": True})
        write_json_atomic(ctx.unit / "native_standup_attempts.json", attempts)
        if S != ctx.S:
            ctx.S = S   # every later leg and B list follows the served width
        return standup, attempts[-1]
    raise ControlsFailed("native standup: retry exhausted")  # pragma: no cover


def _standup_endpoint(standup: Any, out: Path) -> Dict[str, Any]:
    """The standup's ``endpoint.json``: what the standup wrote, else what
    ``standup.endpoint()`` reports (written here so the gate canary's
    bootstrap can read it beside ``cells_served.json``). Never overwrites."""
    p = out / lsu.ENDPOINT_FILE
    if p.is_file():
        return read_json(p)
    ep = standup.endpoint() if callable(getattr(standup, "endpoint", None)) else {}
    ep = dict(ep or {})
    write_json_atomic(p, ep)
    return ep


def run_controls(ctx: Ctx, standup: Any, out: Path) -> Dict[str, Any]:
    """The five positive controls' verdicts. E's ``bring_up_lora`` RUNS them
    itself (``run_controls=True``, its default) and writes
    ``positive_controls.json`` before it returns — a failure there raises
    ``PositiveControlsFailed`` out of the standup, which ``bring_up_native``
    turns into rc 4 — so when the file is on disk it is read, never re-run
    (the standup's record is the one the gate canary's bootstrap and
    ``cells_served.json`` carry). The ``run_positive_controls`` seam is
    called only for a standup that wrote no record (``run_controls=False``,
    or a fake); a raise or a failed verdict is :class:`ControlsFailed`."""
    p = out / lsu.POSITIVE_CONTROLS_FILE
    if p.is_file():
        result = read_json(p)
    else:
        try:
            result = dict(ctx.deps.run_positive_controls(standup) or {})
        except Exception as e:  # noqa: BLE001
            write_json_atomic(out / "positive_controls_error.json", {"error": f"{type(e).__name__}: {e}"[:2000]})
            raise ControlsFailed(f"positive controls raised: {type(e).__name__}: {e}") from e
        if not p.is_file():
            write_json_atomic(p, result)
        else:
            result = read_json(p)
    summary = adm.controls_summary(result)
    if not summary["all_passed"]:
        raise ControlsFailed(f"positive controls: {summary['passed']}/{summary['of']} passed; {summary['problems']}")
    return summary


def write_leg(ctx: Ctx, leg: str, rows: List[Dict[str, Any]], errors: List[str], **meta: Any) -> Path:
    pdir = ctx.unit / "probes"
    pdir.mkdir(parents=True, exist_ok=True)
    if errors:
        write_json_atomic(pdir / f"{leg}.errors.json", {"errors": errors, "n_rows": len(rows)})
        raise RuntimeError(f"leg {leg}: {len(errors)} probe error(s), e.g. {errors[0]}")
    if len(rows) != ctx.probe_set["n"]:
        raise RuntimeError(f"leg {leg}: {len(rows)} rows for {ctx.probe_set['n']} probes")
    path = pdir / f"{leg}.jsonl"
    data = "".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows).encode("utf-8")
    _write_bytes_atomic(path, data)
    n_ids = sum(1 for r in rows if any(e[0] is not None for e in (r.get("first_token") or {}).get("top5", [])))
    write_json_atomic(pdir / f"{leg}.DONE.json",
                      {"n_rows": len(rows), "sha256_of_jsonl": sha256_bytes(data), "written_at": ctx.deps.now(),
                       "leg": leg, "top5_ids_parsed_rows": n_ids, **meta})
    return path


def read_leg(ctx: Ctx, leg: str) -> Optional[List[Dict[str, Any]]]:
    """ONE leg loader for the job and the offline readmit:
    ``adm.read_leg_rows`` — the JSONL only when its DONE.json is beside it."""
    return adm.read_leg_rows(ctx.unit, leg)


def run_probe_legs(ctx: Ctx, standup: Any, legs: Sequence[Tuple[str, str, int, Optional[str]]], *,
                   proxy_api_key: str) -> None:
    """``legs`` = ``[(leg, vmodel, n_fillers, filler_vmodel)]``; each leg is
    every probe submitted together (concurrency = n_probes + n_fillers <= S)."""
    for leg, vmodel, n_fill, filler_vmodel in legs:
        conc = ctx.probe_set["n"] + n_fill
        if conc > ctx.S:
            raise RuntimeError(f"leg {leg}: {conc} concurrent > S {ctx.S}")
        log(f"probes {leg}: {ctx.probe_set['n']} x {vmodel}" + (f" + {n_fill} fillers -> {filler_vmodel}" if n_fill else ""))
        t0 = ctx.deps.now()
        rows, errors = ctx.deps.probe_leg(standup.proxy_url, proxy_api_key, vmodel, ctx.probes,
                                          max_tokens=int(ctx.args.probe_max_tokens), concurrency=conc,
                                          fillers=n_fill, filler_vmodel=filler_vmodel)
        write_leg(ctx, leg, rows, errors, vmodel=vmodel, concurrency=conc, n_fillers=n_fill,
                  filler_vmodel=filler_vmodel, elapsed_s=ctx.deps.now() - t0)


def run_throughput(ctx: Ctx, standup: Any, leg: str, vmodel: str, batches: Sequence[int], *,
                   proxy_api_key: str) -> Dict[str, Any]:
    tdir = ctx.unit / "throughput"
    tdir.mkdir(parents=True, exist_ok=True)
    launch = getattr(standup, "launch", None)
    metrics_url = (f"{launch.base_url}/metrics" if launch is not None and getattr(launch, "base_url", None) else None)
    headers = dict(getattr(launch, "auth_headers", {}) or {}) if launch is not None else {}
    log(f"throughput {leg}: {vmodel} at B={list(batches)} x{ctx.args.repeats}")
    rec = ctx.deps.throughput_leg(standup.proxy_url, proxy_api_key, vmodel, metrics_url=metrics_url,
                                  metrics_headers=headers, batches=list(batches), repeats=int(ctx.args.repeats),
                                  max_tokens=int(ctx.args.throughput_max_tokens),
                                  warmup_s=float(ctx.args.warmup_s), window_s=float(ctx.args.window_s),
                                  out_dir=tdir / "raw", leg=leg)
    write_json_atomic(tdir / f"{leg}.json", rec)
    write_json_atomic(tdir / f"{leg}.DONE.json", {"written_at": ctx.deps.now(), "n_points": len(rec["points"])})
    return rec


def _proxy_api_key(endpoint: Mapping[str, Any]) -> str:
    """THE ONE bearer rule for a standup's proxy (``proxy.proxy_client_token``:
    the endpoint's ``proxy_token`` when set, else ``ProxyClientEngine``'s
    default) — the gate canary's bootstrap follows the same function, so the
    two clients of E's proxy can never disagree on whether it wants a bearer
    (code-review item 10, 2026-09-11)."""
    from sisteer.capeval.serving import proxy as px
    return px.proxy_client_token(endpoint)


def _require_served(ctx: Ctx, standup: Any, api_key: str, want: Sequence[str]) -> List[str]:
    served = ctx.deps.served_models(standup.proxy_url, api_key)
    missing = [v for v in want if v not in served]
    if missing:
        raise RuntimeError(f"proxy at {standup.proxy_url} does not serve {missing}; served={sorted(served)}")
    return served


def mint_run_identity(ctx: Ctx, endpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """``unit/run_identity.json`` — ``RunIdentity(..., engine=endpoint["engine"])``,
    the one native ``run_id`` producer. Refuses an endpoint that does not say
    ``engine == "lora_native"``."""
    from sisteer.capeval.serving.run_identity import RunIdentity
    engine = endpoint.get("engine")
    if engine != NATIVE_ENGINE:
        raise ControlsFailed(f"standup endpoint.json engine={engine!r}, expected {NATIVE_ENGINE!r}")
    v = ctx.facts.get("versions", {})
    ident = RunIdentity(
        model=ctx.model, revision=ctx.revision, tree_sha=ctx.facts.get("tree_sha"), model_key=ctx.target.model_key,
        max_model_len=ctx.L, max_num_seqs=ctx.S, client_concurrency=ctx.probe_set["n"],
        budget=int(getattr(ctx.geom, "budget_tokens", None) or ctx.L * ctx.S),
        budget_source=str(getattr(ctx.geom, "budget_source", None) or "table"),
        serving_versions={k: v.get(k) for k in ("vllm", "transformers", "torch")},
        sampling={"temperature": 0.0, "seed": 0, "max_tokens": int(ctx.args.probe_max_tokens)},
        input_fingerprints={"probe_set_hash": ctx.probe_set["hash"],
                            "object_sha256": ctx.object_sha256, "harness_cell": ctx.hcell},
        engine=engine)
    d = ident.as_dict()
    d["written_at"] = ctx.deps.now()
    write_json_atomic(ctx.unit / "run_identity.json", d)
    return d


def _read_native_attestation(out: Path) -> Dict[str, Any]:
    """The standup's ``serving_attestation.json`` through C's STRICT reader
    (item 11: never a lenient ``json.loads`` of whatever is there — a schema
    drift or a foreign file is a :class:`ControlsFailed` naming it)."""
    try:
        return sa.read_serving_attestation(out / sa.SERVING_ATTESTATION_FILE)
    except sa.ServingAttestationError as e:
        raise ControlsFailed(f"the standup's serving attestation under {out} is refused by the strict "
                             f"reader: {e}") from e


def _load_native_table(out: Path) -> Dict[str, Any]:
    """The standup's ``routing_table.json`` through the proxy's loader — a
    VERSION-2 table with adapter blocks is required (item 11)."""
    from sisteer.capeval.serving import proxy as px
    path = out / lsu.ROUTING_TABLE_FILE
    if not path.is_file():
        raise ControlsFailed(f"the standup wrote no {lsu.ROUTING_TABLE_FILE} under {out}")
    try:
        table = px.load_routing_table(path)
    except Exception as e:  # noqa: BLE001 — a ProxyError, or a malformed doc's KeyError/TypeError
        raise ControlsFailed(f"the standup's routing table {path} does not load: "
                             f"{type(e).__name__}: {e}") from e
    if table.get("version") != 2 or not isinstance(table.get(px.ADAPTERS_KEY), Mapping) \
            or not table[px.ADAPTERS_KEY]:
        raise ControlsFailed(f"the standup's routing table {path} is version {table.get('version')!r} "
                             "without adapter blocks — not a lora-native table")
    return table


def _adapter_digests_from(endpoint: Mapping[str, Any], out: Path) -> Dict[str, Any]:
    """Per adapter (zero included): ``dir``, ``config_sha256``, ``weights_sha256``,
    ``r`` — from the routing table's version-2 ``adapters`` blocks (the blocks
    the proxy binds every lora route to), loaded STRICTLY, cross-checked
    against the serving attestation (same adapter set, same digests) and
    against the endpoint's own ``adapters`` map + ``zero_adapter`` block (the
    standup records ``none__lora0`` under that separate key). Any
    disagreement is a :class:`ControlsFailed`: three records that name one
    adapter set must agree on it (item 11)."""
    from sisteer.capeval.serving import proxy as px
    table = _load_native_table(out)
    att = _read_native_attestation(out)
    blocks: Dict[str, Mapping[str, Any]] = dict(table[px.ADAPTERS_KEY])
    if set(blocks) != set(att["adapters"]):
        raise ControlsFailed(f"the routing table's adapters {sorted(blocks)} != the attestation's "
                             f"{sorted(att['adapters'])}")
    for name, block in blocks.items():
        for key in ("config_sha256", "weights_sha256"):
            if block.get(key) != att["adapters"][name].get(key):
                raise ControlsFailed(f"adapter {name!r}: table {key} {block.get(key)!r} != the "
                                     f"attestation's {att['adapters'][name].get(key)!r}")
    ep_blocks: Dict[str, Any] = dict(endpoint.get("adapters") or {})
    zero = endpoint.get("zero_adapter")
    if isinstance(zero, Mapping) and isinstance(zero.get("name"), str):
        ep_blocks.setdefault(zero["name"], zero)
    if not ep_blocks:
        raise ControlsFailed("the endpoint records no adapters (E's `adapters` + `zero_adapter`)")
    for name, b in ep_blocks.items():
        if name not in blocks:
            raise ControlsFailed(f"the endpoint names adapter {name!r} the routing table does not")
        for key in ("dir", "config_sha256", "weights_sha256", "r"):
            if (b or {}).get(key) != blocks[name].get(key):
                raise ControlsFailed(f"adapter {name!r}: endpoint {key} {(b or {}).get(key)!r} != the "
                                     f"routing table's {blocks[name].get(key)!r}")
    return {n: {k: b.get(k) for k in ("dir", "config_sha256", "weights_sha256", "r")}
            for n, b in blocks.items()}


def run_unit(ctx: Ctx) -> Dict[str, Any]:
    """Phases 1-7 on THIS allocation; ``unit/DONE.json`` last. Returns the
    admission record. Raises :class:`ControlsFailed` (rc 4) when the native
    standup or a control fails (no traffic), :class:`CertRefusal` when the
    oracle gate is not met, anything else as rc 1 — after tearing down."""
    require_oracle_done(ctx)
    # A PRIOR UNIT UNDER THIS OUT DIR IS NEVER BUILT ON (§5: the unit is
    # indivisible and a partial one contributes nothing). The quarantine of
    # EVERY earlier attempt's records — the unit among them — happens in
    # :func:`begin_attempt`, before build_context, the preflight and the
    # oracle can fail; so a populated unit dir here means this attempt's
    # isolation was bypassed, and the unit refuses to start on top of it
    # rather than quietly moving it aside now.
    if ctx.unit.exists() and any(ctx.unit.iterdir()):
        raise RuntimeError(f"{ctx.unit} is already populated inside this attempt "
                           f"({sorted(p.name for p in ctx.unit.iterdir())}) — begin_attempt did not "
                           "run, or something wrote under unit/ before the unit phase")
    ctx.unit.mkdir(parents=True, exist_ok=True)
    # identity.json FIRST (§1.6 / §5): every bound field is known before the
    # standup; the adapter digests (and a lowered S) are filled in by an atomic
    # rewrite once the native server is up — a unit without DONE.json is
    # partial whatever it holds, so the rewrite never changes what a reader
    # may reuse
    unit_stamp = {**_stamp(ctx), "started_at": ctx.started_at}
    write_json_atomic(ctx.unit / "identity.json", {**pre_identity(ctx), **unit_stamp, "adapter_digests": None,
                                                   "complete": False})
    standups = _Standups()
    ready_s: Dict[str, Any] = {}
    try:
        # ---- phase 1: the native standup + controls ---------------------------
        native_out = ctx.unit / "lora_native"
        standup, attempt = bring_up_native(ctx, standups)
        ready_s["lora_native"] = attempt.get("ready_s")
        # the attestation through the STRICT reader (item 11): a file that is
        # not a lora_native attestation is a failure, not a presence check —
        # and the functional rule's READINESS input (native-serving-default
        # §1.4): attested iff this strict read succeeded, what it attested copied
        att = _read_native_attestation(native_out)
        readiness = adm.readiness_record(att, path=str(native_out / sa.SERVING_ATTESTATION_FILE))
        controls = run_controls(ctx, standup, native_out)
        endpoint = _standup_endpoint(standup, native_out)
        for ep_key, att_val in (("attested_enforce_eager", att["engine"]["enforce_eager"]),
                                ("attested_cudagraph_mode", att["engine"]["cudagraph_mode"]),
                                # the FULL capture surface (2026-09-16): the endpoint's copy must be the attestation's
                                ("attested_cudagraph_full_capture_max", att["engine"].get("cudagraph_full_capture_max"))):
            if ep_key in endpoint and endpoint[ep_key] != att_val:
                raise ControlsFailed(f"endpoint {ep_key}={endpoint[ep_key]!r} != the attestation's "
                                     f"{att_val!r}")
        # E's endpoint.json is a LIVENESS record — tear_down() unlinks it (phase
        # 4) so a stale endpoint never points at a dead proxy; the unit keeps
        # its own durable copy of what the standup published
        write_json_atomic(ctx.unit / "lora_native_endpoint.json", dict(endpoint))
        objmap = endpoint.get("object_artifact_sha256_by_cell")
        if objmap != dict(ctx.objmap_h):
            raise ControlsFailed(f"endpoint object_artifact_sha256_by_cell {objmap!r} != "
                                 f"{dict(ctx.objmap_h)!r}")
        if not ctx.target.measures_numerics and endpoint.get("output_diagnostics") is not False:
            raise ControlsFailed(f"row {ctx.target.key} is numerics={ctx.target.numerics!r} but the standup "
                                 f"records output_diagnostics={endpoint.get('output_diagnostics')!r} "
                                 "(the controls must have run with the greedy-token agreement OFF)")
        write_json_atomic(native_out / "CANARY_OK",
                          {"written_at": ctx.deps.now(), **_stamp(ctx), "controls": controls})
        log("CANARY_OK: readiness + the five positive controls passed")
        identity = {**pre_identity(ctx), **unit_stamp,
                    "object_artifact_sha256_by_cell": dict(objmap),
                    "adapter_digests": _adapter_digests_from(endpoint, native_out),
                    "attested_enforce_eager": endpoint.get("attested_enforce_eager"),
                    "attested_cudagraph_mode": endpoint.get("attested_cudagraph_mode"),
                    # the SERVED template's full sha (the standup's), beside the
                    # manifest's 16-hex chat_template_hash from pre_identity
                    "chat_template_sha256": endpoint.get("chat_template_sha256"),
                    "complete": True}
        write_json_atomic(ctx.unit / "identity.json", identity)
        mint_run_identity(ctx, endpoint)
        proxy_key = _proxy_api_key(endpoint)
        from sisteer.capeval.petri.defense_cells import NOOP_LORA_CELL_ID
        v_cell, v_zero, v_base = vmodel_of(ctx.hcell), vmodel_of(NOOP_LORA_CELL_ID), vmodel_of("none")
        _require_served(ctx, standup, proxy_key, [vmodel_of(h) for h in ctx.hcells] + [v_zero, v_base])

        # ---- phase 1b: the §10 gate canary (REQUIRED; the unit continues) ------
        # bound to THIS attempt: ATTEMPT.json (nonce + the unit's
        # identity) before the child, canary_run.json (nonce, rc, the record's
        # sha) after; the reader accepts the record only under that binding
        gdir = ctx.unit / "gate_canary"
        attempt = adm.begin_gate_attempt(
            gdir, unit={"harness_cell": ctx.hcell, "object_sha256": ctx.object_sha256,
                        "node": unit_stamp["node"], "slurm_job_id": unit_stamp["slurm_job_id"],
                        "unit_started_at": ctx.started_at, "tree_sha": ctx.facts.get("tree_sha"),
                        adm.ATTEMPT_ID_KEY: ctx.attempt_id})
        argv = [ctx.deps.python, str(ctx.repo / GATE_SCRIPT_REL), "gate",
                "--manifest", str(ctx.manifest_path), "--out-dir", str(gdir),
                "--lora-native-standup", str(native_out),
                "--reference", str(ctx.prep / "oracle" / "reference.json")]
        if not ctx.target.measures_numerics:
            argv.append("--no-output-diagnostics")   # The gate's selection verdict stays structural-only
        log("gate canary: " + " ".join(argv))
        grc = ctx.deps.run_subprocess(argv, gdir / "gate_canary.log", float(ctx.args.gate_timeout_s))
        run = adm.finish_gate_attempt(gdir, attempt, rc=grc)
        gate = adm.gate_canary_summary(gdir, cell=ctx.hcell)
        write_json_atomic(gdir / "canary_summary.json", {"rc": grc, "elapsed_s": run["elapsed_s"],
                                                         "nonce": attempt["nonce"], "summary": gate})
        log(f"gate canary rc={grc} status={gate['status']} passed={gate['passed']} bound={gate['bound']}")

        # ---- phase 2: probes through the lora proxy ----------------------------
        # (native-psr-serving: a not_measured row runs NO probe leg — nothing to record)
        if ctx.target.measures_numerics:
            run_probe_legs(ctx, standup, [("lora_native", v_cell, 0, None), ("lora_native_rep", v_cell, 0, None),
                                          ("lora_zero", v_zero, 0, None), ("lora_none", v_base, 0, None)],
                           proxy_api_key=proxy_key)
        else:
            log(f"probes SKIPPED by design: row {ctx.target.key} is numerics={ctx.target.numerics!r}")
        # ---- phase 3: throughput on the lora proxy -----------------------------
        run_throughput(ctx, standup, "lora_native", v_cell, ctx.batches, proxy_api_key=proxy_key)
        # ---- phase 4: stop ------------------------------------------------------
        standups.stop(standup)
        write_json_atomic(ctx.unit / "lora_native" / "torn_down.json", ctx.deps.wait_gpu_free())

        # ---- phase 5: lens STACKED ----------------------------------------------
        lens_v_cell, lens_v_base = vmodel_of(ctx.hcell), vmodel_of("none")
        n_fill = max(0, min(ctx.S - ctx.probe_set["n"], ctx.probe_set["n"]))
        st_out = ctx.unit / "lens_stacked"
        st = bring_up_lens(ctx, standups, out=st_out, mode="stacked")
        ready_s["lens_stacked"] = st["ready_s"]
        if ctx.target.measures_numerics:
            run_probe_legs(ctx, st["standup"], [("lens_stacked", lens_v_cell, 0, None),
                                                ("lens_stacked_rep", lens_v_cell, 0, None),
                                                ("lens_stacked_compA", lens_v_cell, n_fill, lens_v_cell),
                                                ("lens_stacked_compB", lens_v_cell, n_fill, lens_v_base)],
                           proxy_api_key=_proxy_api_key({}))
        run_throughput(ctx, st["standup"], "lens_stacked", lens_v_cell, ctx.batches, proxy_api_key=_proxy_api_key({}))
        standups.stop(st["standup"])
        write_json_atomic(st_out / "torn_down.json", ctx.deps.wait_gpu_free())

        # ---- phase 6: lens PER_ROW (a NUMERICS leg: the primary floor's other side) -----
        # native-psr-serving: DROPPED for a not_measured row — nothing of it is recorded
        if ctx.target.measures_numerics:
            pr_out = ctx.unit / "lens_per_row"
            pr = bring_up_lens(ctx, standups, out=pr_out, mode="per_row")
            ready_s["lens_per_row"] = pr["ready_s"]
            run_probe_legs(ctx, pr["standup"], [("lens_per_row", lens_v_cell, 0, None)], proxy_api_key=_proxy_api_key({}))
            run_throughput(ctx, pr["standup"], "lens_per_row", lens_v_cell, [ctx.S], proxy_api_key=_proxy_api_key({}))
            standups.stop(pr["standup"])
            write_json_atomic(pr_out / "torn_down.json", ctx.deps.wait_gpu_free())
        else:
            log(f"lens per_row leg DROPPED by design: row {ctx.target.key} is numerics={ctx.target.numerics!r}")

        # ---- phase 7: admission (the ONE function) --------------------------------
        return finish_unit(ctx, gate=gate, controls=controls, ready_s=ready_s, readiness=readiness)
    finally:
        standups.stop_all()


def bring_up_lens(ctx: Ctx, standups: _Standups, *, out: Path, mode: str) -> Dict[str, Any]:
    """The production lens standup (``capctrl_serving_up.bring_up``) at ONE
    reduction mode, pinned the way ``mt_serving_up`` pins it (the env var,
    inherited by the child through ``launch_server``'s ``dict(os.environ)``),
    and REQUIRED to attest it (``attested_reduction_mode_at`` == (mode, None)).
    The path is named EXPLICITLY (``serving_path=LENS_SERVING_PATH``): the
    standup entry points default to the native engine since 2026-09-13, and
    the head-to-head's lens legs must stay lens or the comparison is
    native-vs-native (native-serving-default §1.4)."""
    # the env pin is SCOPED to this leg: the child inherits it at launch (launch_server's dict(os.environ));
    # leaving it set leaked into every later reader of the same process — the Modal merge-tree suite on
    # 20a6064e failed 45 tinycap / sampling-protocol tests with REDUCTION_MODE_MISMATCH after the in-process
    # certification tests left "1" behind (the launcher reads it as the mode the shell expects)
    prior = os.environ.get(REDUCTION_ENV_VAR)
    os.environ[REDUCTION_ENV_VAR] = "1" if mode == "stacked" else "0"
    try:
        geom = ctx.geom
        if ctx.geometry_lowered is not None:
            geom = ctx.deps.resolve_geometry(ctx.eval_name, model_key=ctx.target.model_key, max_model_len=ctx.L,
                                             max_num_seqs=ctx.S, strict_identity=False)
        log(f"lens standup ({mode}, serving_path={LENS_SERVING_PATH}): cells={ctx.lens_cells} S={ctx.S} "
            f"{REDUCTION_ENV_VAR}={os.environ[REDUCTION_ENV_VAR]}")
        t0 = ctx.deps.now()
        standup = ctx.deps.capctrl_bring_up(
            repo=ctx.repo, config_path=ctx.config_path, cell=list(ctx.lens_cells), defenses_dir=ctx.defenses_dir,
            out=out, geom=geom, manifest=str(ctx.manifest_path),
            expect_object_sha256=[f"{tt}={ctx.objmap_tt[tt]}" for tt in ctx.cells],
            gpu_mem_util=str(ctx.args.gpu_mem_util), max_logprobs=int(ctx.args.max_logprobs),
            ready_timeout_s=int(ctx.args.ready_timeout_s),
            serving_path=LENS_SERVING_PATH)
    finally:
        if prior is None:
            os.environ.pop(REDUCTION_ENV_VAR, None)
        else:
            os.environ[REDUCTION_ENV_VAR] = prior
    standups.add(standup)
    ready = ctx.deps.now() - t0
    got, err = ctx.deps.attested_reduction_mode_at(out / "steering_attestation.json")
    write_json_atomic(out / "reduction_mode_check.json", {"required": mode, "attested": got, "error": err})
    if (got, err) != (mode, None):
        raise RuntimeError(f"lens standup at {out} attests reduction mode {got!r} ({err}); required {mode!r}")
    if not (out / "endpoint.json").is_file() and callable(getattr(standup, "endpoint", None)):
        write_json_atomic(out / "endpoint.json", dict(standup.endpoint(ctx.eval_name) or {}))
    return {"standup": standup, "ready_s": ready, "mode": got}


def finish_unit(ctx: Ctx, *, gate: Mapping[str, Any], controls: Mapping[str, Any],
                ready_s: Mapping[str, Any], readiness: Mapping[str, Any]) -> Dict[str, Any]:
    """Phase 7: ``head_to_head.json``, then ``admission.json`` through the ONE
    function (the functional rule over ``readiness`` — the strict-read
    attestation of phase 1 — the positive controls, the gate canary and the
    head-to-head), ``controls_summary.json`` (the controls, the canary AND the
    readiness record, so an offline re-evaluation can hold its own re-read to
    what the job attested), the consumed oracle reference, the inventory and
    ``DONE.json`` last."""
    tdir = ctx.unit / "throughput"
    points_by_leg: Dict[str, List[Dict[str, Any]]] = {}
    for leg in (adm.THROUGHPUT_LEGS if ctx.target.measures_numerics else adm.THROUGHPUT_LEGS_CORE):
        p = tdir / f"{leg}.json"
        if p.is_file():
            points_by_leg[leg] = list(read_json(p).get("points", []))
    h2h = adm.head_to_head(points_by_leg, headline_B=ctx.S, ready_s=ready_s)
    h2h["geometry"] = {"L": ctx.L, "S": ctx.S, "lowered": ctx.geometry_lowered}
    write_json_atomic(ctx.unit / "head_to_head.json", h2h)
    # the inputs are built through the SAME loader the offline readmit uses
    # (`adm.admission_inputs_from_unit` reads every complete leg off unit/probes/),
    # so a re-evaluation of this record reads what this run read. E's raw
    # noop_certificate record rides along: it is the SOLO zero certificate that
    # decides zero_adapter_exact, recorded with its exact scope
    pc_path = ctx.unit / "lora_native" / lsu.POSITIVE_CONTROLS_FILE
    pc_record = read_json(pc_path) if pc_path.is_file() else None
    noop_cert = ((pc_record.get("controls") or {}).get(adm.ZERO_ADAPTER_CONTROL)
                 if isinstance(pc_record, dict) else None)
    inputs = adm.admission_inputs_from_unit(ctx.unit, gate_canary=gate, controls=controls, head_to_head=h2h,
                                            n_probes=ctx.probe_set["n"], max_tokens=int(ctx.args.probe_max_tokens),
                                            attempt_id=ctx.attempt_id,
                                            geometry={"L": ctx.L, "S": ctx.S, "lowered": ctx.geometry_lowered},
                                            noop_certificate=noop_cert,
                                            control_max_tokens=int(ctx.args.control_probe_max_tokens),
                                            readiness=readiness, numerics_mode=ctx.target.numerics)
    admission = adm.admission_verdict(inputs)
    admission_path = write_json_atomic(ctx.unit / "admission.json", admission)
    write_json_atomic(ctx.unit / "controls_summary.json",
                      {"positive_controls": pc_record,
                       "summary": dict(controls), "gate_canary": dict(gate), "readiness": dict(readiness),
                       "note": ("the five standup controls, the REQUIRED gate canary verdict and the readiness "
                                "record (the strict-read serving attestation) — the functional gates' inputs")})
    # the oracle reference this unit consumed, verbatim, so the unit's evidence
    # is self-contained (its sha256 is already in identity.json)
    _write_bytes_atomic(ctx.unit / "oracle_reference.json",
                        (ctx.prep / "oracle" / "reference.json").read_bytes())
    # THE EVIDENCE INVENTORY (A2): every file under unit/ with its digest and
    # the required-evidence list; a unit missing required evidence does not
    # complete — the RESULT then carries no verdict rather than a verdict
    # nobody can audit
    inventory = adm.build_unit_inventory(ctx.unit, attempt_id=ctx.attempt_id)
    inv_path = write_json_atomic(ctx.unit / adm.UNIT_INVENTORY_FILE, inventory)
    if inventory["missing"]:
        raise RuntimeError(f"the unit is missing required evidence and cannot complete: "
                           f"{inventory['missing']} (inventory {inv_path})")
    # DONE.json LAST, bound to THIS attempt, to the admission bytes it
    # completes and to the inventory (F1 / A2: the report and a whole-unit
    # reuse read the records through adm.unit_binding_problems +
    # adm.unit_inventory_problems)
    write_done(ctx.unit, verdict=admission["verdict"], **_stamp(ctx),
               **{adm.DONE_ADMISSION_SHA_KEY: sha256_file(admission_path),
                  adm.DONE_INVENTORY_SHA_KEY: sha256_file(inv_path), "n_files": inventory["n_files"]})
    prs = admission["numerics"].get("prompt_resampling_sensitivity") or {}
    bounds = prs.get("bounds") or {}
    gates = admission["gates"]
    log(f"unit DONE: verdict={admission['verdict']} rule={admission['rule']} reasons={admission['reasons']} "
        f"gates={{{', '.join(f'{k}={gates[k]}' for k in adm.FUNCTIONAL_GATES)}}} "
        f"zero_exact(solo)={admission['zero_adapter_exact']} ratio_at_S={h2h.get('ratio_at_S')} | "
        f"DIAGNOSTIC numerics (gate=false): outcome={admission['numerics'].get('outcome')} "
        f"floor={admission['floor']['source']} primary={admission['floor'].get('disagreement')} "
        f"delta={prs.get('delta_point')} lower95={bounds.get('lower95_one_sided')} "
        f"upper95={bounds.get('upper95_one_sided')} "
        f"zero_concurrent_n_div={admission['zero_adapter']['concurrent'].get('n_diverging')}")
    return admission


# --------------------------------------------------------------------------- #
# RESULT + main                                                                 #
# --------------------------------------------------------------------------- #

def begin_attempt(out: Path, deps: Deps, *, argv: Sequence[str]) -> Dict[str, Any]:
    """Open ONE attempt under ``out`` (F1): mint the attempt nonce, move
    every earlier attempt's records (:data:`ATTEMPT_RECORDS`) aside AS A
    WHOLE into ``out/stale-<utc>-<nonce8>/`` (a rename, never a deletion;
    ``adm.move_aside``), then write ``ATTEMPT.json`` FIRST — before
    build_context, the preflight or the oracle can fail — so nothing an
    earlier attempt produced is in any reader's path while this one runs, and
    a refusal leaves a RESULT beside NO admission. Returns the attempt record."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    attempt_id = secrets.token_hex(16)
    present = [n for n in ATTEMPT_RECORDS if (out / n).exists()]
    tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + attempt_id[:8]
    stale = adm.move_aside(out, ATTEMPT_RECORDS, tag=tag)
    rec = {adm.ATTEMPT_ID_KEY: attempt_id, "started_at": deps.now(), "node": deps.hostname(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "pid": os.getpid(), "argv": list(argv),
           "prior_attempt_moved_to": (str(stale) if stale is not None else None),
           "prior_records_moved": present}
    write_json_atomic(out / ATTEMPT_FILE, rec)
    if stale is not None:
        log(f"attempt {attempt_id[:8]}: an earlier attempt's records {present} under {out} were moved "
            f"aside to {stale} (never reused in place)")
    return rec


def own_admission(ctx: Ctx) -> Optional[Dict[str, Any]]:
    """``unit/admission.json`` iff the unit's records bind to THIS attempt
    (``adm.unit_binding_problems`` with the attempt's id), else ``None`` —
    the RESULT line never carries a verdict this attempt did not produce."""
    if not (ctx.unit / "admission.json").is_file():
        return None
    probs = adm.unit_binding_problems(ctx.unit, expected_attempt_id=ctx.attempt_id)
    if probs:
        log(f"unit/admission.json is not this attempt's completed admission: {probs}")
        return None
    return read_json(ctx.unit / "admission.json")


def result_line(*, state: str, rc: int, elapsed: float, host: str, target: str,
                admission: Optional[Mapping[str, Any]], unit_reused_from: Optional[str]) -> str:
    """The one-line RESULT (native-serving-default §1.4): the verdict, the RULE
    it was computed under, the FOUR functional gates by name
    (``readiness`` / ``controls`` / ``gate`` / ``not_slower`` + ``ratio_at_S``),
    then the RECORDED numerics as a diagnostic (``numerics=diagnostic
    outcome=<three-way> floor=<source>``). A record without an admission
    reads ``verdict=none rule=none readiness=none … numerics=None``; the
    sbatch's fallback line mirrors these fields. A schema-4 admission (a unit
    reused from an earlier record) still spells its legacy ``numerics=<bool>``."""
    a = admission or {}
    nm = a.get("numerics") or {}
    controls = a.get("controls") or {}
    gc = a.get("gate_canary") or {}
    rd = a.get("readiness") or {}
    h2h = a.get("head_to_head") or {}
    if nm.get("mode") == adm.NUMERICS_NOT_MEASURED:
        numerics: Any = adm.NUMERICS_NOT_MEASURED      # native-psr-serving: stated, never a verdict
    else:
        numerics = "diagnostic" if nm.get("gate") is False else nm.get("admitted")
    readiness = "none" if not rd else ("attested" if rd.get("attested") is True else "unattested")
    gate = "none" if not gc else ("passed" if gc.get("passed") else str(gc.get("status") or "missing"))
    ratio = h2h.get("ratio_at_S")
    ratio_txt = f"{float(ratio):.3f}" if isinstance(ratio, (int, float)) and not isinstance(ratio, bool) else "None"
    return (f"state={state} exit={rc} elapsed={int(elapsed)} host={host} target={target} "
            f"verdict={a.get('verdict', 'none')} rule={a.get('rule', 'none')} readiness={readiness} "
            f"controls={controls.get('passed', 0)}/{controls.get('of', adm.N_CONTROLS)} gate={gate} "
            f"not_slower={h2h.get('not_slower_at_S')} ratio_at_S={ratio_txt} "
            f"numerics={numerics} outcome={nm.get('outcome', 'none')} "
            f"floor={(a.get('floor') or {}).get('source', 'none')} "
            f"unit_reused_from={unit_reused_from or 'none'}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-key", "--target", dest="model_key", required=True, choices=sorted(TARGETS),
                    help="the TARGET ROW (Target.key: the out dir <key>_<jobid> and RESULT's target=); the MODEL "
                         "the row certifies is its Target.model_key (llama8b for both llama rows)")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume-from", default=None, help="a prior record dir: prep/oracle reused after a "
                                                          "binding check; a COMPLETE unit reused whole under its "
                                                          "original provenance (§5)")
    ap.add_argument("--eval", default=EVAL_DEFAULT)
    ap.add_argument("--cell", default=None, help="TrainTest cell (default: the target's table entry)")
    ap.add_argument("--harness-cell", default=None, help="override the harness translation (tests only)")
    ap.add_argument("--defenses-dir", default=None)
    ap.add_argument("--adapter-bank", default=None)
    ap.add_argument("--pins", default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--batches", default="8,32,S")
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--warmup-s", type=float, default=20.0)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--probe-max-tokens", type=int, default=PROBE_MAX_TOKENS)
    ap.add_argument("--throughput-max-tokens", type=int, default=THROUGHPUT_MAX_TOKENS)
    ap.add_argument("--control-probe-max-tokens", type=int, default=48)
    # (no --admission-margin / --admission-margin-source: retired with the functional rule — the
    # greedy-token numerics are recorded, never a gate, so there is no margin to amend)
    ap.add_argument("--gpu-mem-util", default="0.9")
    ap.add_argument("--max-logprobs", type=int, default=64)
    ap.add_argument("--ready-timeout-s", type=int, default=1800)
    ap.add_argument("--oracle-timeout-s", type=float, default=1800.0)
    ap.add_argument("--gate-timeout-s", type=float, default=1800.0)
    ap.add_argument("--allow-short-probe-set", action="store_true")
    ap.add_argument("--preflight-cells", action="store_true",
                    help="CPU-only (native-psr-serving §1.6): resolve the row's canary set — tt_cell + "
                         "extra_cells — through the REAL TrainTestCellSource over the row's defenses_dir "
                         "(kind via ARM_OBJECT_KINDS, object path, sha vs the pins, native position mode, "
                         "disposition == serve), write <out>/preflight_cells.json, print it, and EXIT: "
                         "rc 0 when every cell resolves, 2 otherwise. Run before any sbatch")
    return ap


def preflight_cells(target: Target, *, repo: Path, defenses_dir: Path, pins_path: Path) -> Dict[str, Any]:
    """The CPU preflight of a row's CANARY SET: every cell of
    ``target.cells`` resolved through the REAL ``TrainTestCellSource`` the
    standup builds (a LENS-policy source over the row's config + manifest +
    ``defenses_dir``; ``registry_adapter.partition_cells_for_native`` then
    judges each cell's disposition exactly as the standup will) — kind (the
    registry's ``ARM_OBJECT_KINDS``), object path, the object's sha256 held to
    the pins the view was built from, the native position mode of a residual
    kind (``native_position_mode``), and ``disposition == serve``. Returns
    ``{"ok": bool, "problems": [...], "cells": {tt: {...}}}``; a problem is
    named per cell, and a row with any problem is refused BEFORE a GPU is
    asked for. Known-bads (pinned by test): a rung name (``ar_r0p1``) — not a
    registered arm — refuses; a view missing one of the cells refuses.

    Each cell's pin is resolved through :func:`resolve_pin_key`, so a QUALIFIED
    key (stage 2's ``arditi`` canary, pinned by the misalignment authority) is
    held to ITS file and the record says which one. Since stage 2 the POSITION
    MODE is a recorded fact, not a gate: every mode the lens names is served
    natively, and what judges the cell is the disposition (a skip-class cell is
    still refused here, before any GPU)."""
    import dataclasses as _dc
    import yaml
    import lora_serving_up as lsu
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri import registry_adapter as ra

    problems: List[str] = []
    cells_out: Dict[str, Any] = {}
    config_path, manifest_path = repo / target.config, repo / target.manifest
    for p in (config_path, manifest_path, pins_path):
        if not p.is_file():
            return {"ok": False, "problems": [f"missing {p}"], "cells": {}, "target": target.key}
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    explicit = list(cfg.get("explicit_cells") or [])
    tt_cells = list(target.cells)
    for tt in tt_cells:
        if tt not in explicit:
            problems.append(f"cell {tt!r} is not in {target.config}'s explicit_cells {explicit}")
    try:
        harness_ids = [ra.harness_cell_id(c) for c in tt_cells]
    except ra.RegistryAdapterError as e:
        return {"ok": False, "problems": problems + [f"harness translation: {e}"], "cells": {}, "target": target.key}
    if problems:
        return {"ok": False, "problems": problems, "cells": {}, "target": target.key}
    spec0 = cm.load_petri_spec(manifest_path)
    kept, _dropped = lsu.manifest_flags_for_lora_native(spec0.serving_launch_flags)
    spec_all = _dc.replace(spec0, cells=harness_ids, serving_launch_flags=kept)
    try:
        src = ra.traintest_cell_source(spec_all, repo_root=repo, config_path=config_path,
                                       defenses_dir=Path(defenses_dir), serving_path_policy="lens")
        served, skipped = ra.partition_cells_for_native(harness_ids, src)
    except (ra.RegistryAdapterError, ValueError, dc.DefenseCellError) as e:
        return {"ok": False, "problems": [f"cell source / disposition: {type(e).__name__}: {e}"],
                "cells": {}, "target": target.key}
    skipped_by = {s_.cell_id: s_ for s_ in skipped}
    for tt, h, pin_key in zip(tt_cells, harness_ids, target.pin_keys):
        row: Dict[str, Any] = {"harness_cell": h, "pin_key": pin_key}
        try:
            entry = src.get(h)
        except ra.RegistryAdapterError as e:
            problems.append(f"cell {tt!r}: {e}")
            cells_out[tt] = {**row, "status": "refused"}
            continue
        if entry is None or not entry.object_path:
            problems.append(f"cell {tt!r} ({h}): ABSENT from the view at {defenses_dir} (no registry entry / object)")
            cells_out[tt] = {**row, "status": "absent"}
            continue
        obj = Path(entry.object_path)
        row.update(kind=str(entry.kind), object_path=str(obj))
        # EVERY required phase's serving path must be able to stand this cell up (the native
        # standup AND the lens stacked leg) — a row one phase cannot complete is refused HERE
        row["serving_paths"] = list(required_serving_paths(target))
        for p in cell_serving_problems(str(entry.kind), entry.resolved_spec, paths=required_serving_paths(target)):
            problems.append(f"cell {tt!r}: {p}")
        if not obj.is_file():
            problems.append(f"cell {tt!r}: object {obj} is not a file")
            cells_out[tt] = {**row, "status": "absent"}
            continue
        have = sha256_file(obj)
        # the SAME resolution the driver and the sbatch use (bare or qualified — a cell
        # whose pin belongs to another authority is read from THAT file)
        try:
            ref = resolve_pin_key(pin_key, pins_path=pins_path, model_key=target.model_key)
        except CertRefusal as e:
            row.update(sha256=have, pinned_sha256=None)
            problems.append(f"cell {tt!r}: {e}")
            cells_out[tt] = {**row, "status": "problem"}
            continue
        row.update(sha256=have, pinned_sha256=ref.sha256, pins_file=str(ref.path),
                   pin_model_key=ref.model_key)
        if have != ref.sha256:
            problems.append(f"cell {tt!r}: object sha {have[:12]} != pin {ref.sha256[:12]} "
                            f"({ref.spelling()} in {ref.path.name})")
        if str(entry.kind) in dc.RESIDUAL_SLOT_KINDS:
            try:
                mode, header = dc.native_position_mode(str(entry.kind), entry.resolved_spec or {},
                                                       header_required=dc.NATIVE_STAGE2_ARMED)
                row.update(position_mode=mode, response_header_len=header)
                # stage 2 (native-psr-serving §1.8) serves every mode the lens names, so the
                # mode is RECORDED and the DISPOSITION below is what judges the cell. While
                # the flag is False the only servable mode is `all` and anything else is
                # refused here, before a GPU is asked for.
                if mode != "all" and not dc.NATIVE_STAGE2_ARMED:
                    problems.append(f"cell {tt!r}: native position mode {mode!r} is a stage-2 class "
                                    "and stage 2 is not armed on this tree (stage 1 serves `all`)")
            except dc.DefenseCellError as e:
                problems.append(f"cell {tt!r}: position mode cannot be judged: {e}")
        if h in skipped_by:
            row.update(disposition="skip", reason=skipped_by[h].reason)
            problems.append(f"cell {tt!r} ({entry.kind}) would be SKIPPED by the native standup: {skipped_by[h].reason}")
        elif h in served:
            row.update(disposition="serve")
        else:
            row.update(disposition="unknown")
            problems.append(f"cell {tt!r}: neither served nor skipped by the partition")
        cells_out[tt] = {**row, "status": ("ok" if not any(f"cell {tt!r}" in p for p in problems) else "problem")}
    return {"ok": not problems, "problems": problems, "cells": cells_out, "target": target.key,
            "model_key": target.model_key, "view": target.view, "numerics": target.numerics,
            "defenses_dir": str(defenses_dir), "canary_set": list(harness_ids) if not problems else []}


def run_preflight_cells(args: argparse.Namespace) -> int:
    """``--preflight-cells``: :func:`preflight_cells` for the row, written to
    ``<out>/preflight_cells.json`` and printed; rc 0 iff every cell resolves,
    else :data:`RC_REFUSED`. No attempt is opened, no GPU is touched."""
    import lora_serving_up as lsu
    target = TARGETS[args.model_key]
    repo = Path(args.repo_dir).resolve()
    pins_path = Path(args.pins) if args.pins else repo / PINS_REL
    if args.defenses_dir:
        defenses_dir = Path(args.defenses_dir)
    else:
        root = lsu.shared_root()
        if root is None:
            print(json.dumps({"ok": False, "problems": ["no --defenses-dir and no shared root"], "target": target.key}))
            return RC_REFUSED
        defenses_dir = root / DEFENSES_REL / target.model_key / "cells_view" / target.view
    rec = preflight_cells(target, repo=repo, defenses_dir=defenses_dir, pins_path=pins_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "preflight_cells.json", rec)
    print(json.dumps(rec, indent=2, sort_keys=True, default=str))
    log(f"preflight-cells {target.key}: {'OK' if rec['ok'] else 'REFUSED'} canary_set={rec.get('canary_set')} "
        f"problems={rec['problems']}")
    return RC_OK if rec["ok"] else RC_REFUSED


def run(args: argparse.Namespace, deps: Optional[Deps] = None) -> int:
    if getattr(args, "preflight_cells", False):
        return run_preflight_cells(args)
    deps = deps or Deps.production()
    t0 = deps.now()
    out = Path(args.out)
    # F1: this attempt is isolated from every earlier one BEFORE anything can
    # fail — build_context, the preflight and the oracle all raise CertRefusal
    attempt = begin_attempt(out, deps, argv=sys.argv)
    admission: Optional[Dict[str, Any]] = None
    reused: Optional[str] = None
    rc = RC_FAILED
    state = "FAILED"
    host = deps.hostname()
    ctx: Optional[Ctx] = None
    try:
        ctx = build_context(args, deps, attempt_id=attempt[adm.ATTEMPT_ID_KEY])
        write_manifest(ctx, phases={}, prior_attempt_moved_to=attempt["prior_attempt_moved_to"],
                       prior_unit_moved_to=(attempt["prior_attempt_moved_to"]
                                            if "unit" in attempt["prior_records_moved"] else None))
        phase_preflight(ctx)
        phase_oracle(ctx)
        prior_unit, why, snap = unit_reuse_decision(ctx)
        if prior_unit is not None:
            reused = str(prior_unit)
            # the self-contained copy + unit_reuse.json FIRST (the acceptance),
            # then the manifest says reused; the admission is the COPIED bytes'
            admission = accept_reused_unit(ctx, prior_unit, snap)
            mark_phase(ctx, "unit", "reused", unit_reused_from=reused)
            log("no unit phase runs on this allocation")
            controls_ok = bool((admission.get("controls") or {}).get("all_passed"))
            rc = RC_OK if controls_ok else RC_CONTROLS
        else:
            if args.resume_from:
                write_json_atomic(out / adm.UNIT_REUSE_FILE,
                                  {"unit_reused_from": None, "reasons": why, adm.ATTEMPT_ID_KEY: ctx.attempt_id})
                log(f"unit not reused: {why}")
            admission = run_unit(ctx)
            mark_phase(ctx, "unit", "done")
            rc = RC_OK
        state = "DONE"
    except CertRefusal as e:
        log(f"REFUSED: {e}")
        rc = RC_REFUSED
        write_json_atomic(out / "FAILED.json", {"kind": "refusal", "error": str(e)})
    except ControlsFailed as e:
        log(f"CONTROLS FAILED: {e}")
        rc = RC_CONTROLS
        write_json_atomic(out / "FAILED.json", {"kind": "controls", "error": str(e)})
    except Exception as e:  # noqa: BLE001
        import traceback
        log(f"FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        rc = RC_FAILED
        write_json_atomic(out / "FAILED.json", {"kind": type(e).__name__, "error": str(e),
                                                "traceback": traceback.format_exc()})
    if admission is None and ctx is not None:
        # only an admission THIS attempt completed (F1) — never one an earlier
        # attempt left, which begin_attempt moved aside in any case
        admission = own_admission(ctx)
    line = result_line(state=state, rc=rc, elapsed=deps.now() - t0, host=host, target=args.model_key,
                       admission=admission, unit_reused_from=reused)
    (out / "RESULT").write_text(line + "\n")
    log("RESULT " + line)
    return rc


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
