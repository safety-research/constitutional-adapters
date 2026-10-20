#!/usr/bin/env python
"""sdfbase-lora: stand up ONE base model's LORA-NATIVE served target — a
plugin-free ``vllm serve`` with CUDA graphs ON, every LoRA cell of a view
registered as a vLLM native adapter and selected per request by the
``model`` field, plus the zero adapter, behind the existing RoutingProxy on
loopback — then block until a stop file appears (THE WAVE FRONT over the
shared native core ``sisteer.capeval.serving.native_standup``).

This file is the WAVE FRONT: it normalises a wave's inputs — the transfer
config (``explicit_cells``) + its Petri manifest, ``select_cells_lora`` (mt's
``select_cells`` PLUS one admission: a ``prompted__<si>`` request over a
config-named SI is admitted beyond ``explicit_cells``), the SKIP-BY-NAME
partition (as re-drawn by native-psr-serving: PSR / plain-vector cells SERVE on
the residual slot; AlphaSteer / antidote / the stacked wrapper have no native
form and a position-masked LoRA fit no honest one — those are recorded in
``cells_skipped``, never served, never a second lens standup; a LoRA cell whose
record the engine cannot judge REFUSES naming ``--serving-path lens``), the
per-fitted-cell digests
(``--expect-object-sha256``, REQUIRED — capctrl's E45 H1 boundary; or the
view-manifest+pins digests ``mt_serving_up.py`` derives), the manifest's
eager/lora pins stripped and RECORDED (``manifest_flags_dropped``), the served
template hashed against the manifest — into an :class:`EffectiveServing` and
hands it to :func:`native_standup.bring_up_native`, which does everything from
the served template onward (export-or-reuse into the content-addressed bank,
the VERSION-2 routing table, the LoRA kernel index ceiling, the launch, the
STRICT attestation read, the proxy, the FIVE positive controls, the records).
``tool_calls`` is EXPLICIT: this front passes ``eval.startswith("petri")``.

A view with NO LoRA cell (a ``none``-only cell set, a prompted-only view, a
view whose objects were all skipped) stands up natively with the zero adapter
alone (``steering_bites`` / ``selection_honored`` recorded ``not_applicable``),
so base and steered configurations share ONE engine.

EXIT CODES: 0 stop file; 2 a refusal before the controls ran (geometry, pins,
an unservable cell, a swapped object, a version-1 table, a compose or
readiness refusal — the server, if launched, is torn down); 3 the engine
died while serving; 4 a positive control failed (server torn down, no
``endpoint.json``).

LIBRARY SEAM: :func:`bring_up_lora` performs the whole bring-up (the records
included) and returns a live :class:`LoraStandup`; the certification harness
drives standups through the SAME function the CLI does, and every
existing call composes byte-identically (the new options are keyword-only and
default to today's behaviour). It raises :class:`StandupRefused` for its own
preflight refusals, propagates ``vllm_server.VllmServerError`` from compose /
readiness UNWRAPPED, and raises :class:`PositiveControlsFailed` after tearing
the server down. :func:`run_positive_controls` is public too.

THE VERIFIED OBJECT MAP: ``object_artifact_sha256_by_cell`` =
``{harness_cell_id: sha256(object.pt)}`` for every FITTED cell served —
never ``none``, never ``none__lora0``, never a prompted cell — written
identically into ``cells_served.json`` and ``endpoint.json`` and handed to
the proxy. The standup itself NEVER loads an HF model.
"""
from __future__ import annotations

import argparse
import dataclasses
import json  # noqa: F401  (re-exported for the tests' `lsu.json`)
import os  # noqa: F401
import secrets
import socket  # noqa: F401  (the tests monkeypatch `lsu.socket.gethostname`)
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from capctrl_serving_up import (_start_proxy_thread, free_port,  # noqa: E402,F401
                                expected_artifact_digests_for_cells,
                                legacy_object_sha256, parse_object_digest_args,
                                resolve_standup_geometry)
from mt_serving_up import _utc_now, attested_scheduler_geometry, launch_cmd_extras  # noqa: E402,F401
import hpo_probe_status as hps  # noqa: E402  — the front door's record predicate (stdlib-only), handed to the native core's parity gate
from sisteer.capeval.serving import native_standup as ns  # noqa: E402
from sisteer.capeval.serving import serving_attestation as sa  # noqa: E402,F401  (stdlib-only leaf)
# the core's vocabulary, re-exported under this module's names (the certification harness,
# the §10 gate and every test read them off `lora_serving_up`)
from sisteer.capeval.serving.native_standup import (  # noqa: E402,F401
    CELLS_SERVED_FILE, CHAT_TEMPLATE_FILE, CONTROL_NAMES, DERIVED_CONFIG_FILE, ENDPOINT_ATTEMPT_KEY,
    ENDPOINT_FILE, ENGINE, EXIT_CONTROLS_FAILED, EXIT_DEAD_ENGINE, EXIT_REFUSED, POSITIVE_CONTROLS_FILE,
    ROUTING_TABLE_FILE, SERVER_LOG_FILE, STALE_DIR_PREFIX, STRIPPED_MANIFEST_FLAGS, TAG,
    EffectiveServing, LoraStandup, LoraStandupError, PositiveControlsFailed, SkippedCell, StandupRefused,
    REQUESTED_EXPLICIT_HARNESS_KEY, REQUESTED_EXPLICIT_KEY, requested_cells_explicit,
    default_adapter_bank, effective_launch_flags, lora_kernel_ceiling_geometry, probe_prompt_set,
    probe_set_hash, quarantine_prior_endpoint, resolve_adapter_bank, run_positive_controls, shared_root,
    unpublish_endpoint, wait_until_stop)

#: ``--mode`` -> the geometry table's eval (explicit ``--eval`` wins)
EVAL_BY_MODE = {"attacks": "multiturn_attacks", "petri": "petri",
                "static": "families_served", "capability": "capability"}
#: the config keys whose KEYS name SIs (beside the SIs the explicit cells name)
_CONFIG_SI_KEYED = ("max_len_by_si", "batch_size_by_si")
#: the Nemotron backend contract this front loads by default when a caller hands none
#: (the wave and the certification harness call with the default; a nemotron_h target
#: never serves UNATTESTED on the native engine)
MAMBA_CONTRACT_DEFAULT = "configs/capeval/mamba_backend_contract.json"
#: the keyword-only options the wave front gained (eight, plus
#: ``server_default_policy``), each defaulting to today's behaviour — pinned by test
NATIVE_FRONT_OPTIONS = ("thinking_on", "reasoning_parser", "text_only_mm", "enable_log_requests",
                        "sampling_clamp_override", "mamba_backend_contract", "kv_derived_seqs",
                        "attest_mrope", "server_default_policy")


def config_si_vocabulary(cfg: Mapping[str, Any]) -> Tuple[str, ...]:
    """The SIs a transfer config names (the admission set for a prompted
    baseline): every explicit cell's SI (``defense_cell.si_of``) plus the keys
    of the per-SI tables (``max_len_by_si`` / ``batch_size_by_si``). Order:
    first appearance; duplicates dropped."""
    from sisteer.eval import defense_cell as tt

    seen: List[str] = []
    for c in cfg.get(tt.EXPLICIT_CELLS_KEY) or ():
        si = tt.si_of(c) if isinstance(c, str) else None
        if si is not None and si not in seen:
            seen.append(si)
    for key in _CONFIG_SI_KEYED:
        table = cfg.get(key)
        if isinstance(table, Mapping):
            for si in table:
                if isinstance(si, str) and si not in seen:
                    seen.append(si)
    return tuple(seen)


def select_cells_lora(explicit_cells: Sequence[str], requested: Optional[Sequence[str]], *,
                      si_vocabulary: Sequence[str] = ()) -> Tuple[List[str], List[str]]:
    """The TrainTest cells this standup serves — mt's ``select_cells`` rule
    (``explicit_cells`` when nothing is requested — with ``none`` served first
    even when the config lists none of it, a steered-wave config;
    else ``none`` + every requested cell in order, de-duplicated, each cell's
    BASE one of the config's explicit cells) PLUS one admission: a
    ``prompted__<si>`` request whose ``<si>`` is in ``si_vocabulary`` is
    admitted even when it is not an explicit cell. Returns
    ``(cells, prompted_admitted)``; every other unlisted cell refuses by name."""
    from sisteer.capeval.serving import serve_declaration as sd

    want = list(explicit_cells)
    if not requested:
        # a STEERED-WAVE config lists no `none` (858118 / 858119 refused "enumerate_cells derived no none__lora0 leg") — the base route is
        # still SERVED, implicitly, exactly as under --cell and as mt_serving_up.select_cells serves it
        # on the lens path: the native core derives its none__lora0 control twin from it, the positive controls compare against that twin,
        # load_check.py probes it and the attestation reads vmodels["none"]. Neither none nor the twin is a measured, registered or scored
        # leg: the drivers iterate the CONFIG's cells, the registration is the config's gated pairs, the Petri drive freezes only what --cell
        # declared (cells_requested_explicit stays what was named: nothing, here).
        return (want if "none" in want else ["none", *want]), []
    out = ["none"]
    admitted: List[str] = []
    for raw in requested:
        if not isinstance(raw, str) or not raw.strip():
            raise StandupRefused(f"--cell needs a non-empty TrainTest cell id, got {raw!r}")
        cell = raw.strip()
        if cell == "none" or cell in out:
            continue
        try:
            base, _variant = sd.split_variant(cell)
        except sd.ServeDeclarationError as e:
            raise StandupRefused(f"--cell {cell!r}: {e}") from None
        if base not in want:
            if base.startswith("prompted__") and _variant is None:
                si = base[len("prompted__"):]
                if si in si_vocabulary:
                    admitted.append(cell)
                    out.append(cell)
                    continue
                raise StandupRefused(
                    f"--cell {cell!r}: SI {si!r} is not in the config's SI vocabulary "
                    f"{list(si_vocabulary)} — a prompted baseline is admitted beyond "
                    "explicit_cells only for an SI the config names (D14); refusing")
            raise StandupRefused(
                f"--cell {cell!r}: its base {base!r} is not one of the config's explicit_cells "
                f"{want} — refusing to serve a cell the wave never enumerated (only a "
                "prompted__<si> baseline over a config-named SI is admitted beyond them, D14)")
        out.append(cell)
    return out, admitted


def manifest_flags_for_lora_native(flags: Sequence[str]) -> Tuple[List[str], List[str]]:
    """``(kept, dropped)`` — the core's :func:`effective_launch_flags` (the ONE
    derivation of the effective native flags; kept under this name for the
    existing callers and tests)."""
    return effective_launch_flags(flags)


# --------------------------------------------------------------------------- #
# the bring-up                                                                 #
# --------------------------------------------------------------------------- #


def _resolve_geometry(eval_name: str, max_model_len, max_num_seqs, *, model_key: str,
                      allow_over_budget: bool, override_reason: Optional[str],
                      requested_max_tokens: Optional[int] = None):
    from sisteer.capeval.serving import geometry as sg

    if override_reason is not None:
        if not isinstance(override_reason, str) or not override_reason.strip():
            raise StandupRefused("--geometry-override-reason must be a non-empty reason")
        if max_model_len is None and max_num_seqs is None:
            raise StandupRefused(
                "--geometry-override-reason given but neither --max-model-len nor "
                "--max-num-seqs overrides the table — a reason for nothing is refused")
    try:
        if override_reason is not None:
            # the ONE way to serve a non-identity S: the identity guard is
            # relaxed, the budget guard is not, and the reason is RECORDED by the caller
            return sg.resolve_geometry(eval_name, max_model_len=max_model_len,
                                       max_num_seqs=max_num_seqs,
                                       allow_over_budget=bool(allow_over_budget),
                                       strict_identity=False, model_key=model_key)
        return resolve_standup_geometry(eval_name, max_model_len, max_num_seqs,
                                        allow_over_budget=allow_over_budget,
                                        model_key=model_key,
                                        requested_max_tokens=requested_max_tokens)
    except (sg.GeometryOverBudget, sg.GeometryIdentityBroken, sg.GeometryCapExceedsBasis) as e:
        raise StandupRefused(f"refusing the geometry: {e}") from e
    except sg.UnknownModelKey as e:
        raise StandupRefused(f"refusing --model-key: {e}") from e
    except ValueError as e:
        raise StandupRefused(f"refusing the geometry: {e}") from e


def _behavior_and_prompt_files(spec, repo: Path):
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.tiers.petri_runner import resolve_prompt_files

    behavior_files = {}
    for key in cm.petri_expected_behavior_keys(spec):
        bpath = repo / "configs" / "capeval" / "petri" / key / "BEHAVIOR.md"
        if not bpath.is_file():
            raise StandupRefused(f"missing behavior definition {bpath}")
        behavior_files[key] = bpath.read_bytes()
    prompt_files = resolve_prompt_files(spec, override_root=repo / "configs/capeval/petri/prompts")
    return behavior_files, prompt_files


def partition_or_refuse(harness_ids: Sequence[str], lens_source: Any
                        ) -> Tuple[List[str], List[SkippedCell]]:
    """The wave front's skip seam: :func:`native_standup.partition_cells_for_native`
    with a REFUSAL (exit 2) that names the cell and the escape hatch when a
    LoRA-kind object cannot be served honestly."""
    from sisteer.capeval.petri import registry_adapter as ra
    try:
        return ns.partition_cells_for_native(list(harness_ids), lens_source)
    except (ra.RegistryAdapterError, ValueError) as e:
        raise StandupRefused(f"{e}") from e


def default_mamba_contract(repo: Path) -> Optional[Mapping[str, Any]]:
    """The reviewed Nemotron backend contract at the repo's pinned path, or None
    when the checkout has none (a nemotron_h standup then refuses in the core)."""
    from sisteer.capeval.serving import mamba_backend as mb
    path = Path(repo) / MAMBA_CONTRACT_DEFAULT
    return mb.load_contract(path) if path.is_file() else None


def served_hf_id_by_key() -> dict:
    """canonical model key -> the HF id a standup's --model-key must name: the paper's eight (geometry.MODEL_HF_IDS) AND the
    front door's onboarded models (geometry.ONBOARDED_MODEL_HF_IDS — reproduce/models.yaml's served checkpoints), the same two
    tables geometry.rows_for / budget_for size from. Read at call time (the tables are the documented seam). The rule: a key the
    geometry tables know is admitted here, whichever table holds it — a check against MODEL_HF_IDS alone refused every onboarded
    key as `not a canonical model key` (paper-companion-repo stage E, 2026-09-30), so neither standup path consults it on its own."""
    from sisteer.capeval.serving.geometry import MODEL_HF_IDS, ONBOARDED_MODEL_HF_IDS
    return {**MODEL_HF_IDS, **ONBOARDED_MODEL_HF_IDS}


def parity_record_check(record_dir: Path):
    """The native core's ``parity_record_check``: the SAME predicate the front door reads —
    ``hpo_probe_status.probe_state_record`` — over a probe record reachable from this node (the probe cluster); READY iff the
    record is complete, its menu + verdict are of its current generation AND the parity record they read is current
    (``verify_parity_record``). ``job_active`` is False here: a queued/running probe job reads INCOMPLETE_STALLED rather than
    INCOMPLETE_RUNNING, and both refuse the standup."""
    return hps.probe_state_record(Path(record_dir), job_active=False)


def bring_up_lora(*, repo_dir: Path, config: Path, manifest: Path, defenses_dir: Path,
                  out: Path, model_key: str, eval: str,  # noqa: A002 — the §1.5 flag's name
                  cell: Optional[Sequence[str]] = None,
                  adapter_bank: Optional[Path] = None,
                  expect_object_sha256: Optional[Sequence[str]] = None,
                  max_model_len: Optional[int] = None, max_num_seqs: Optional[int] = None,
                  allow_over_budget: bool = False,
                  geometry_override_reason: Optional[str] = None,
                  gpu_mem_util: str = "0.9", max_logprobs: Optional[int] = None,
                  proxy_port: Optional[int] = None, vllm_port: Optional[int] = None,
                  ready_timeout_s: int = 1800, install_timeout_s: int = 900,
                  upstream_timeout_s: Optional[float] = None,
                  no_petri_agent_loop: bool = False, probe_max_tokens: int = 48,
                  mode: Optional[str] = None, run_controls: bool = True,
                  # --- keyword-only, each defaulting
                  # to today's behaviour so the CLI and the certification's existing calls are unchanged
                  thinking_on: bool = False, reasoning_parser: Optional[str] = None,
                  text_only_mm: str = "off", enable_log_requests: bool = False,
                  sampling_clamp_override: Optional[float] = None,
                  mamba_backend_contract: Optional[Mapping[str, Any]] = None,
                  kv_derived_seqs: bool = False, attest_mrope: bool = False,
                  # the SERVER-DEFAULT SAMPLING POLICY (vllm_server.SERVER_DEFAULT_POLICIES):
                  # contract_override = today's launch byte for byte; engine_defaults = no composed override
                  server_default_policy: str = "contract_override",
                  # --- additive: the mt front's translations
                  tool_calls: Optional[bool] = None,
                  expected_digests_source: str = ns.DIGESTS_SOURCE_CLI,
                  driver_max_tokens: Optional[int] = None,
                  extra_endpoint_facts: Optional[Mapping[str, Any]] = None,
                  # --- the OUTPUT-DIAGNOSTICS POLICY override. None = the
                  # core's central policy (off iff a residual-slot cell is served); False = OFF
                  # regardless (the certification driver's `numerics=not_measured` row); True =
                  # ON, refused by the core when the policy says off (a caller may only tighten)
                  output_diagnostics: Optional[bool] = None) -> LoraStandup:
    """The whole bring-up (module docstring), as a function — the certification
    harness drives standups through exactly this. The keyword names ARE the
    §1.5 CLI flags, so ``main`` forwards ``vars(args)`` and a harness composes
    the same dict; ``cell`` is the repeatable ``--cell`` (a list of TrainTest
    ids; ``none`` implicit). Returns the live :class:`LoraStandup` with
    ``positive_controls.json``, ``cells_served.json`` and ``endpoint.json`` on
    disk (``run_controls=False`` skips the controls AND ``endpoint.json``).

    THE NATIVE-SERVING-DEFAULT OPTIONS (the mt / capctrl fronts forward
    them): ``thinking_on`` + ``reasoning_parser`` (the template's ON header
    certified per template before any GPU minute; nothing declared — LoRA cells
    are all-position), ``text_only_mm`` (``auto`` | ``on`` | ``off``),
    ``enable_log_requests``, ``sampling_clamp_override`` (the BFCL protocol
    clamp, attested), ``mamba_backend_contract`` (None -> the repo's pinned
    contract when the checkout carries it; a nemotron_h target never serves
    unattested), ``kv_derived_seqs`` (ONE relaunch at the KV-derived S, a §1.8
    adjustment), ``attest_mrope`` (recorded ``not_applicable``; GLM refuses).
    ``tool_calls`` None infers the agent modality from ``eval`` (``petri`` /
    ``petri:<task>``); the mt front passes its own ``mode == "petri"`` rule.
    ``expected_digests_source`` records where ``expect_object_sha256`` came from
    (``cli`` | ``view_manifest+pins``); ``extra_endpoint_facts`` are the mt
    front's provenance keys, recorded verbatim beside the core's (a collision
    with a core key refuses).

    Raises :class:`StandupRefused` (exit 2) for every preflight refusal;
    ``vllm_server.VllmServerError`` UNWRAPPED from compose / readiness;
    :class:`PositiveControlsFailed` (exit 4) after tearing the server down.
    ``endpoint.json`` is a READINESS record bound to ONE attempt (F3)."""
    from transformers import AutoTokenizer

    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.models import pin_reasoning_template_off

    repo = Path(repo_dir).resolve()
    eval_name, cells, petri_agent_loop = str(eval), cell, not no_petri_agent_loop
    config_path, manifest_path = config, manifest
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # F3: this attempt's nonce, and NO prior attempt's readiness record in the
    # way — before the first refusal, the launch, or a control can run
    attempt_id = secrets.token_hex(16)
    quarantine_prior_endpoint(out, attempt_id=attempt_id)
    if probe_max_tokens is None or int(probe_max_tokens) < 1:
        raise StandupRefused(f"--probe-max-tokens must be >= 1, got {probe_max_tokens!r}")
    if upstream_timeout_s is not None and not float(upstream_timeout_s) > 0:
        raise StandupRefused(f"--upstream-timeout-s must be > 0, got {upstream_timeout_s}")
    if expected_digests_source not in (ns.DIGESTS_SOURCE_CLI, ns.DIGESTS_SOURCE_VIEW_PINS):
        raise StandupRefused(f"expected_digests_source {expected_digests_source!r} is not one of "
                             f"{(ns.DIGESTS_SOURCE_CLI, ns.DIGESTS_SOURCE_VIEW_PINS)}")

    geom = _resolve_geometry(eval_name, max_model_len, max_num_seqs, model_key=model_key,
                             allow_over_budget=allow_over_budget,
                             override_reason=geometry_override_reason,
                             requested_max_tokens=driver_max_tokens)

    # --- pins: config / manifest / model key (mt's rules) -----------------------
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = repo / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, Mapping) or "explicit_cells" not in cfg:
        raise StandupRefused(f"{cfg_path} is not an explicit_cells config — this standup "
                             "serves a wave's enumerated cells")
    model, rev = cfg["model"], cfg["model_revision"]
    man_path = Path(manifest_path)
    if not man_path.is_absolute():
        man_path = repo / man_path
    spec0 = cm.load_petri_spec(man_path)
    if (spec0.target_model, spec0.target_revision) != (model, rev):
        raise StandupRefused(f"manifest pins {spec0.target_model}@{spec0.target_revision} but "
                             f"the config serves {model}@{rev} — one wave, one target")
    hf_id_by_key = served_hf_id_by_key()      # the paper's eight + the onboarded models (never MODEL_HF_IDS alone)
    want_model = hf_id_by_key.get(model_key)
    if want_model is None:
        raise StandupRefused(f"--model-key {model_key!r} is not a canonical model key "
                             f"({sorted(hf_id_by_key)}) — the per-model budget cannot be sized")
    if want_model != model:
        raise StandupRefused(f"--model-key {model_key} names {want_model} but the config serves "
                             f"{model} — the per-model budget would size the batch from the "
                             "WRONG model's measurement; refusing")
    tt_of = {c: ra.traintest_cell_id(c) for c in spec0.cells}
    explicit = list(cfg["explicit_cells"])
    missing = [c for c, tt in tt_of.items() if tt not in explicit]
    unserved = [c for c in explicit if c not in tt_of.values()]
    if missing or unserved:
        raise StandupRefused(f"manifest/config cell mismatch: manifest cells not in the config "
                             f"{missing}; config cells the manifest does not serve {unserved}")
    ddir = Path(defenses_dir).resolve()
    if not ddir.is_dir():
        raise StandupRefused(f"--defenses-dir {ddir} is not a directory")

    # --- the cells (mt's selection + the prompted admission) -----------------------
    tt_cells, prompted_admitted = select_cells_lora(
        explicit, cells, si_vocabulary=config_si_vocabulary(cfg))
    try:
        harness_ids = [ra.harness_cell_id(c) for c in tt_cells]
    except ra.RegistryAdapterError as e:
        raise StandupRefused(str(e)) from e
    # what the caller NAMED (the repeatable --cell), beside what is served -- the records'
    # cells_requested_explicit / cells_requested_explicit_harness; () / () when nothing was named
    requested_explicit, requested_explicit_h = requested_cells_explicit(cells)
    config_used = cfg_path
    implicit_base = "none" in tt_cells and "none" not in explicit     # s3r-fix-2: a steered-wave config's implicitly served base route
    if prompted_admitted or implicit_base:
        # the cell source's explicit registry is EXACTLY the listed ids, so the admitted prompted baselines — and, s3r-fix-2, the base
        # route a steered-wave config lists no `none` for — join a DERIVED config written beside the records; the wave's config is never
        # edited. The registry then holds the base cell the core derives its none__lora0 control twin from (858118 / 858119 refused
        # "enumerate_cells derived no none__lora0 leg" because the registry had no base cell to serve); the wave config the drivers and the
        # registration read is untouched, so neither none nor the twin is measured, registered or scored.
        derived = dict(cfg)
        derived["explicit_cells"] = (["none"] if implicit_base else []) + explicit + [c for c in prompted_admitted if c not in explicit]
        derived["defenses_dir"] = str(ddir)
        config_used = out / DERIVED_CONFIG_FILE
        config_used.write_text(
            "# DERIVED by lora_serving_up.py from " + str(cfg_path) + ": explicit_cells + the "
            + ("implicitly served base route (a steered-wave config lists no `none`; s3r-fix-2) + the " if implicit_base else "")
            + "prompted baselines admitted from the config's SI vocabulary (D14)\n"
            + yaml.safe_dump(derived, sort_keys=False), encoding="utf-8")

    # --- THE SKIP SEAM: the partition over a LENS-policy source (kind + resolved
    # spec read; nothing exported) — stacked cells and position-masked LoRA fits skipped BY
    # NAME (PSR / vector cells serve on the residual slot, native-psr-serving), a LoRA cell
    # that cannot be served honestly REFUSED naming --serving-path lens
    kept_flags, dropped_flags = manifest_flags_for_lora_native(spec0.serving_launch_flags)
    spec_all = dataclasses.replace(spec0, cells=harness_ids, serving_launch_flags=kept_flags)
    try:
        lens_src = ra.traintest_cell_source(spec_all, repo_root=repo, config_path=config_used,
                                            defenses_dir=ddir, serving_path_policy="lens")
    except (ra.RegistryAdapterError, ValueError) as e:
        raise StandupRefused(f"cell source refused: {type(e).__name__}: {e}") from e
    served_ids, skipped = partition_or_refuse(harness_ids, lens_src)
    skipped_h = {s.cell_id for s in skipped}
    served_tt = [tt for tt, h in zip(tt_cells, harness_ids) if h not in skipped_h]
    for s in skipped:
        print(f"{TAG} SKIPPED by name: {s.cell_id!r} (kind {s.kind!r}) — {s.reason}", flush=True)
    # an EXPLICIT --cell that is skip-class refuses (an operator asked for a cell this engine
    # cannot serve; the default whole-view standup skips it by name instead)
    if cells and skipped:
        asked = [c for c in cells if isinstance(c, str) and ra.harness_cell_id(c.strip()) in skipped_h]
        if asked:
            raise StandupRefused(
                f"--cell {asked}: {[(s.cell_id, s.kind) for s in skipped if s.cell_id in {ra.harness_cell_id(a) for a in asked}]} "
                "cannot be served on the native engine (no native form, or a stage-2 position mode) — pass --serving-path lens "
                "for a lens standup of this cell; a whole-view native standup skips it by name")
    # --- the digests over the SERVED cells (REQUIRED per fitted cell; E45 H1) -----------------
    try:
        expected_digests = expected_artifact_digests_for_cells(
            served_tt, served_ids, parse_object_digest_args(expect_object_sha256, served_tt),
            required=True)
    except ValueError as e:
        raise StandupRefused(str(e)) from e
    print(f"{TAG} eval={eval_name} model={model}@{rev[:12]} cells={served_tt} bank_view={ddir} "
          f"expected_digests={sorted(expected_digests)} ({expected_digests_source}) "
          f"prompted_admitted={prompted_admitted} skipped={[s.cell_id for s in skipped]}",
          flush=True)

    # --- the served template ----------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(model, revision=rev)
    pin_reasoning_template_off(tok)
    template_bytes = tok.chat_template.encode()
    tsha16 = cm.sha256_hex16(template_bytes)
    if tsha16 != spec0.chat_template_hash:
        raise StandupRefused(f"served chat template sha16 {tsha16} != the manifest's "
                             f"chat_template_hash {spec0.chat_template_hash} — a stale/wrong pin")

    # --- the EFFECTIVE spec + the lora-native source over the SERVED cells -------------------
    spec = dataclasses.replace(spec0, cells=served_ids, serving_launch_flags=kept_flags)
    try:
        src = ra.traintest_cell_source(spec, repo_root=repo, config_path=config_used,
                                       defenses_dir=ddir, serving_path_policy="lora_native")
    except (ra.RegistryAdapterError, ValueError) as e:
        raise StandupRefused(f"cell source refused: {type(e).__name__}: {e}") from e
    behavior_files, prompt_files = _behavior_and_prompt_files(spec, repo)
    if mamba_backend_contract is None:
        mamba_backend_contract = default_mamba_contract(repo)
    eff = EffectiveServing(
        spec=spec, cell_source=src, lens_source=lens_src, served_cells=tuple(served_ids),
        cells_skipped=tuple(skipped), tokenizer=tok, chat_template_bytes=template_bytes,
        config_used=config_used, defenses_dir=ddir, model_key=model_key, model=model, revision=rev,
        expected_digests=dict(expected_digests), expected_digests_source=expected_digests_source,
        behavior_files=behavior_files, prompt_files=prompt_files, traintest_cells=tuple(served_tt),
        prompted_cells_admitted=tuple(prompted_admitted), manifest_path=man_path, config_path=cfg_path,
        cells_requested_explicit=requested_explicit, cells_requested_explicit_harness=requested_explicit_h,
        manifest_flags_dropped=tuple(dropped_flags), front="lora_serving_up")
    if tool_calls is None:
        tool_calls = eval_name.split(":", 1)[0] == "petri"
    return ns.bring_up_native(
        eff, out=out, eval=eval_name, geometry=geom, tool_calls=bool(tool_calls),
        adapter_bank=resolve_adapter_bank(adapter_bank, model_key=model_key),
        thinking_on=bool(thinking_on), reasoning_parser=reasoning_parser,
        sampling_clamp_override=sampling_clamp_override, text_only_mm=text_only_mm,
        mamba_backend_contract=mamba_backend_contract, kv_derived_seqs=bool(kv_derived_seqs),
        enable_log_requests=bool(enable_log_requests), probe_max_tokens=probe_max_tokens,
        gpu_mem_util=gpu_mem_util, max_logprobs=max_logprobs, proxy_port=proxy_port,
        vllm_port=vllm_port, ready_timeout_s=ready_timeout_s, install_timeout_s=install_timeout_s,
        upstream_timeout_s=upstream_timeout_s, petri_agent_loop=bool(petri_agent_loop),
        geometry_override_reason=geometry_override_reason, allow_over_budget=allow_over_budget,
        mode=mode, run_controls=run_controls, attest_mrope=bool(attest_mrope), attempt_id=attempt_id,
        server_default_policy=server_default_policy,
        front="lora_serving_up", controls_fn=run_positive_controls,
        extra_endpoint_facts=extra_endpoint_facts,
        parity_record_check=parity_record_check,
        # the wrapper-owned argv extras (mt's rule; spelled here, never in the frozen serving tree)
        launch_extras=launch_cmd_extras(gpu_mem_util, max_logprobs, enable_log_requests),
        output_diagnostics=output_diagnostics)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--config", required=True,
                    help="the wave's transfer config (explicit_cells), relative to --repo-dir or absolute")
    ap.add_argument("--manifest", required=True,
                    help="the wave's Petri manifest (target pins == the config's; cells = harness ids)")
    ap.add_argument("--defenses-dir", required=True,
                    help="the bank VIEW (<tt_cell>/object.pt + identity sidecar per cell)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--model-key", required=True,
                    help="the geometry table's MODEL key (gemma4e4b, qwen35_4b, nemotron3nano4b, ...): "
                         "sizes max_num_seqs from that model's measured budget; must name the "
                         "config's model")
    ap.add_argument("--mode", choices=sorted(EVAL_BY_MODE), default=None,
                    help=f"geometry-table eval by mode ({EVAL_BY_MODE}); --eval wins")
    ap.add_argument("--eval", default=None, help="geometry-table eval (eval[:task]), e.g. "
                                                 "families_served:short")
    ap.add_argument("--cell", action="append", default=None, metavar="TT_CELL",
                    help="serve ONLY this TrainTest cell (repeatable; `none` implicit). Default: every "
                         "explicit_cells entry (PSR / vector cells served on the residual slot; a stacked cell or "
                         "a position-masked LoRA fit skipped BY NAME; an explicit skip-class "
                         "--cell refuses). A prompted__<si> over a config-named SI is admitted beyond "
                         "explicit_cells (D14)")
    ap.add_argument("--adapter-bank", default=None,
                    help="the content-addressed adapter bank (default $SHARED/exp/multilora_adapters/<model-key>, "
                         f"else ${ns.ADAPTER_BANK_ROOT_ENV}/<model-key>)")
    ap.add_argument("--expect-object-sha256", action="append", default=None, metavar="TT_CELL=SHA256",
                    help="REQUIRED per fitted cell: the preflight-verified object.pt sha256, keyed by "
                         "the exact --cell spelling (a bare sha when exactly one fitted cell is served)")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--allow-over-budget", action="store_true")
    ap.add_argument("--geometry-override-reason", default=None,
                    help="the ONE way to serve a NON-IDENTITY geometry (strict_identity=False); the "
                         "reason is recorded in every record (the harness passes "
                         "mamba_cache_blocks_under_graphs on a hybrid's cache refusal)")
    ap.add_argument("--gpu-mem-util", default="0.9")
    ap.add_argument("--max-logprobs", type=int, default=None)
    ap.add_argument("--proxy-port", type=int, default=None)
    ap.add_argument("--vllm-port", type=int, default=None)
    ap.add_argument("--ready-timeout-s", type=int, default=1800)
    ap.add_argument("--upstream-timeout-s", type=float, default=None)
    ap.add_argument("--no-petri-agent-loop", action="store_true")
    ap.add_argument("--probe-max-tokens", type=int, default=48,
                    help="greedy budget of the positive controls' probes")
    # --- the eight options, each OFF / absent by default
    ap.add_argument("--thinking-on", action="store_true",
                    help="serve under the template's THINKING-ON render (certified per template before any "
                         "GPU minute; a native standup declares no cell — LoRA cells are all-position)")
    ap.add_argument("--reasoning-parser", default=None,
                    help="THINKING-ON standups only: the served family's PINNED parser "
                         "(contracts.REASONING_PARSER_BY_FAMILY); the composer refuses any other name")
    ap.add_argument("--text-only-mm", choices=("auto", "on", "off"), default="off",
                    help="compose vLLM's --limit-mm-per-prompt {\"image\":0,\"video\":0} for a multimodal "
                         "target (recorded; unhashed)")
    ap.add_argument("--enable-log-requests", action="store_true",
                    help="pass vLLM's --enable-log-requests (the resolved-sampler readback; recorded)")
    ap.add_argument("--server-default-policy", choices=("contract_override", "engine_defaults"), default="contract_override",
                    help="how the engine resolves a sampling key a request OMITS (vllm_server.SERVER_DEFAULT_POLICIES): "
                         "contract_override (default; the contracts' vendor row composed for a pinned model, byte-identical to "
                         "every launch to date) or engine_defaults (nothing composed; vLLM's neutral defaults where no "
                         "generation_config.json ships). Recorded as server_default_policy on every record")
    ap.add_argument("--sampling-clamp-override", type=float, default=None,
                    help="lift vLLM's _MAX_TEMP clamp to the pinned protocol temperature (BFCL, D117); "
                         "recorded in serving_attestation.json and endpoint.json")
    ap.add_argument("--mamba-backend-contract", default=MAMBA_CONTRACT_DEFAULT,
                    help="the REVIEWED Nemotron backend contract (repo-relative or absolute) a nemotron_h "
                         "standup judges its live per-layer worker report against, over the native "
                         "worker's sisteer_mamba_backend_report RPC — mandatory for Nemotron")
    ap.add_argument("--kv-derived-seqs", choices=("on", "off"), default="off",
                    help="on a readiness KV-coverage shortfall re-launch ONCE at S = floor(the KV-backed "
                         "concurrency), recorded as a kv_shrink geometry adjustment (plan §1.8)")
    ap.add_argument("--mrope-attest", action=argparse.BooleanOptionalAction, default=False,
                    help="recorded not_applicable on the native engine (neox-rotary families; GLM refuses)")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.eval is None and args.mode is None:
        ap.error("one of --mode / --eval is required")
    eval_name = args.eval or EVAL_BY_MODE[args.mode]
    from sisteer.capeval.serving import mamba_backend as mb
    from sisteer.capeval.serving import vllm_server as vs

    stop_file = Path(args.stop_file)
    try:
        kwargs = {k: v for k, v in vars(args).items()
                  if k not in ("stop_file", "eval", "mamba_backend_contract", "kv_derived_seqs",
                               "mrope_attest")}
        repo = Path(args.repo_dir).resolve()
        contract_path = Path(args.mamba_backend_contract)
        if not contract_path.is_absolute():
            contract_path = repo / contract_path
        contract = mb.load_contract(contract_path) if contract_path.is_file() else None
        if contract is None:
            print(f"{TAG} no Nemotron backend contract at {contract_path}: a nemotron_h standup "
                  "REFUSES on the native engine (never unattested)", flush=True)
        kwargs.update(repo_dir=repo, config=Path(args.config),
                      manifest=Path(args.manifest), defenses_dir=Path(args.defenses_dir),
                      out=Path(args.out), eval=eval_name,
                      adapter_bank=(Path(args.adapter_bank) if args.adapter_bank else None),
                      mamba_backend_contract=contract,
                      kv_derived_seqs=(args.kv_derived_seqs == "on"),
                      attest_mrope=bool(args.mrope_attest))
        su = bring_up_lora(**kwargs)
    except PositiveControlsFailed as e:
        print(f"{TAG} FAILED (exit {EXIT_CONTROLS_FAILED}): {e}", flush=True)
        return EXIT_CONTROLS_FAILED
    except StandupRefused as e:
        print(f"{TAG} refused (exit {EXIT_REFUSED}): {e}", flush=True)
        return EXIT_REFUSED
    except vs.VllmServerError as e:
        print(f"{TAG} the lora-native launch could not be composed or did not become ready "
              f"(exit {EXIT_REFUSED}): {e}", flush=True)
        return EXIT_REFUSED
    rc = 0
    try:
        print(f"{TAG} READY: proxy {su.proxy_url} serving {len(su.proxy.virtual_models())} vmodels "
              f"{list(su.proxy.virtual_models())}; waiting for {stop_file}", flush=True)
        rc = wait_until_stop(stop_file, su.server)
    finally:
        su.tear_down()
    print(f"{TAG} down.", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
