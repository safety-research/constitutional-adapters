"""The shared NATIVE serving core.

ONE bring-up behind every native LoRA standup — the wave front
(``scripts/sdfbase/lora_serving_up.bring_up_lora``, reached through
``mt_serving_up.py --serving-path lora_native`` and ``common.sh f22_standup``)
and the single-cell capctrl front (``scripts/capctrl_serving_up.bring_up(
serving_path="lora_native")``, hence the tinycap launcher). Each front owns
its OWN normalisation — which cells, which manifest, which digests, which
template-hash policy — and hands the core an :class:`EffectiveServing`; the
core is today's ``bring_up_lora`` body from "the served template" onward:

  1. the served template written and hashed; the EFFECTIVE spec's cells
     enumerated over the lora-native cell source (``enumerate_cells(
     lens_controls=False, lora_controls=<zero-only>)``: no ``none__lens0`` on
     a plugin-free server; the ``none__lora0`` zero-adapter leg always);
  2. every LoRA cell's ``object.pt`` bytes held to its expected digest FIRST
     (capctrl's E45 H1 boundary), then exported-or-reused into the
     content-addressed adapter bank beside the zero adapter; the VERSION-2
     routing table re-loaded and every block re-hashed on disk;
  3. the LoRA kernel index ceiling applied BEFORE the launch (a lowered S is a
     recorded ``geometry.adjustments`` step — §1.8 — never a silent clamp);
  4. ``build_server_launch(lora_native=True, ...)`` with EXPLICIT ``tool_calls``
     (the wave front passes ``eval.startswith("petri")``, capctrl its
     ``--tool-calls``), ``thinking_on`` + ``reasoning_parser`` (the template's
     ON header CERTIFIED per template before any GPU minute; a native standup
     declares no cell — LoRA cells are all-position), the protocol sampling
     clamp override (RECORDED in the attestation and endpoint.json), the
     text-only multimodal pin, the wrapper-owned argv extras the FRONT composes
     (memory utilization, the logprob cap, the resolved-sampler log opt-in — this
     frozen tree spells none of them); ONE relaunch at the
     KV-derived S on a readiness ``KvCoverageShortfall`` when opted in
     (``kv_derived_seqs``), recorded as a ``kv_shrink`` adjustment composed
     after any kernel lowering;
  5. the STRICT attestation read; for a ``nemotron_h`` target the fused Mamba
     backend attested over the PORTED RPC (``native_worker.
     NativeWorkerExtension.sisteer_mamba_backend_report``) and judged against
     the reviewed contract — a Nemotron standup without a passing attestation
     REFUSES — never the quiet pure-torch fallback;
  6. the RoutingProxy with the engine identity and ``thinking_mode``; the FIVE
     positive controls (``steering_bites`` / ``selection_honored`` recorded
     ``not_applicable`` on a zero-adapter-only standup — a ``none``-only cell
     set, a prompted-only view, a view whose objects were all skipped);
     ``cells_served.json`` + ``endpoint.json`` (atomic), bound to ONE attempt
     (F3).

WHY NOT lens for a view with no LoRA cell: the tinycap base configuration
(``--cell none``) must share the steered configurations' ENGINE, or the
pairing refuses across engines by design, and the lens-less
``launch_server`` branch attests nothing (design §4.4).

THE SKIP SEAM (native-psr-serving §1.1): a stacked cell and every
kind outside ``NATIVE_LORA_SERVABLE_KINDS`` (loreft, alphasteer, antidote, …)
have no native form; a position-masked LoRA fit (``positions`` present and not
``"all"``) has no honest native form either (the engine has no position mask);
and a RESIDUAL-SLOT cell (psr / vector / vector_all / diffmean / static_r)
whose native position mode is not ``all`` waits for stage 2's position class
while ``defense_cells.NATIVE_STAGE2_ARMED`` is False — a view that lists any
of them has them SKIPPED BY NAME, the mode in the reason — logged, recorded in
``cells_served.json`` / ``endpoint.json`` ``cells_skipped`` — never a second
lens standup and never a silent drop. Once stage 2 is armed a RESTRICTED-mode
residual cell serves too, and the proxy holds its route to the request
allowlist, the per-request boundary claim and a request-unique cache salt
(``proxy.RESTRICTED_ROUTE_ALLOWED_KEYS``), the standup takes the
LOGPROB LIVENESS statistic for its ``steering_bites`` and reads the worker's
failure ledger after the controls (:func:`residual_ledger_after_controls`). An
all-positions residual cell SERVES: exported as a rank-1 zero o_proj carrier
plus its payload (``lora_export._export_residual_slot``), written by the
worker's residual slot (``native_residual``; the launch env flag
``SISTEER_NATIVE_RESIDUAL_SLOT=1`` arms it), attested at readiness
(``serving_attestation.residual_slot_problems``, schema 2) and recorded per
cell under ``residual_slot`` in both records. A LoRA cell the view meant to
serve whose record the engine cannot judge (``positions`` missing, a serve
declaration, ``circuit_breaker`` alpha != r) REFUSES, as does a residual
record the mode predicate cannot judge; ``--serving-path lens`` is the escape
hatch for both.

THE OUTPUT-DIAGNOSTICS POLICY (native-psr-serving; greedy-token agreement is
never measured for PSR): :func:`output_diagnostics_
policy` is False iff any served lora cell is a residual-slot kind, decided
HERE from the enumerated cells and handed to EVERY controls function
(``controls_fn(su, output_diagnostics=...)``) — so every front door inherits it
(the wave, the smoke, the timing harness, the certification driver, which may
only tighten it). Under False ``selection_honored`` keeps its structural
verdict (the engine's own routing bookkeeping over the rotated mixed layouts)
and computes NO ``numerics`` block; a LoRA-only standup is unchanged
(``True``). The policy is recorded as ``output_diagnostics`` in
``cells_served.json``, ``endpoint.json`` and ``positive_controls.json``.

REAL limitations that remain refusals under native: a
declared ``@thinking`` / serve-declaration cell (no declaration surface);
``--lens-salt-policy request_unique_all`` and ``--require-attested-mode`` (lens
facts the native engine cannot attest); GLM's M-RoPE patch (rides the lens
plugin — the spec's exclusion).

SEAMS. This module composes against the names in ``defense_cells``
(``native_serving_disposition``, ``NATIVE_DISPOSITIONS``), ``registry_adapter``
(``DEFAULT_STANDUP_SERVING_PATH``, ``SkippedCell``,
``partition_cells_for_native``), ``proxy_engine`` (``served_engine_problem``)
and ``contracts`` (``endpoint_is_native``, ``REDUCTION_MODE_NOT_APPLICABLE``),
re-exported under this module's name so every front reads them off ONE module.
``registry_adapter`` / ``defense_cells`` are imported at module level (they are
this module's import footprint already); ``proxy_engine`` and ``contracts`` are
imported lazily inside thin wrappers, because both pull ``proxy`` /
``vllm_server`` — the footprint is unchanged (pinned by test).
"""
from __future__ import annotations

import dataclasses
import json
import os
import secrets
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sisteer.capeval.petri import defense_cells as _dc   # NATIVE_STAGE2_ARMED, read at call time
from sisteer.capeval.petri.defense_cells import (NATIVE_DISPOSITIONS, NATIVE_POSITION_MODES,
                                                  RESIDUAL_SLOT_KINDS, native_serving_disposition)
from sisteer.capeval.petri.registry_adapter import (DEFAULT_STANDUP_SERVING_PATH, SkippedCell,
                                                     partition_cells_for_native)
from sisteer.capeval.serving import serving_attestation as sa  # stdlib-only leaf

# --------------------------------------------------------------------------- #
# the seams — re-exported; see the module docstring                             #
# --------------------------------------------------------------------------- #

SERVING_PATHS = ("lora_native", "lens")
#: the engine vocabulary the wave shell / drivers assert (``served_engine_problem``)
#: — ``proxy_engine.ENGINE_EXPECTATIONS``, a literal here so this module gains no
#: ``proxy_engine`` import; pinned equal by test
ENGINE_EXPECTATIONS = ("lens:stacked", "lens:per_row", "lora_native")
#: the tinycap driver's explicit "no lens reduction" value on a native endpoint
#: — ``contracts.REDUCTION_MODE_NOT_APPLICABLE``, a literal here so this module
#: gains no ``contracts`` import; pinned equal by test
REDUCTION_MODE_NOT_APPLICABLE = "not_applicable"
#: the reason vocabulary of ``not_applicable`` records
#: (``contracts.NOT_APPLICABLE``; pinned equal by test)
NOT_APPLICABLE = "not_applicable"


def served_engine_problem(ident: Mapping[str, Any], *, expect: str) -> Optional[str]:
    """Why the proxy's ``/v1/models`` ``petri_proxy`` identity block does NOT
    attest the engine ``expect`` names (``lens:stacked`` | ``lens:per_row`` |
    ``lora_native``), or ``None`` — ``proxy_engine.served_engine_problem``,
    imported at call time (``proxy_engine`` pulls ``proxy`` / ``contracts`` /
    ``vllm_server``; this module's footprint stays what it was)."""
    from sisteer.capeval.serving.proxy_engine import served_engine_problem as real
    return real(ident, expect=expect)


def endpoint_is_native(ep: Mapping[str, Any]) -> bool:
    """``endpoint.json`` records a native standup iff ``engine == lora_native``
    — ``contracts.endpoint_is_native``, imported at call time (``contracts``
    pulls ``proxy`` / ``vllm_server``)."""
    from sisteer.capeval.tinycap.contracts import endpoint_is_native as real
    return bool(real(ep))


# --------------------------------------------------------------------------- #
# constants shared with the fronts (re-exported by lora_serving_up)            #
# --------------------------------------------------------------------------- #

#: the engine this standup serves — ``endpoint.json["engine"]`` / ``serving_engine``,
#: the attestation's ``kind``, the proxy's ``petri.engine`` stamp
ENGINE = sa.SERVING_ATTESTATION_KIND
EXIT_REFUSED = 2
EXIT_DEAD_ENGINE = 3
EXIT_CONTROLS_FAILED = 4
POSITIVE_CONTROLS_FILE = "positive_controls.json"
CELLS_SERVED_FILE = "cells_served.json"
ENDPOINT_FILE = "endpoint.json"
#: the key under which ``endpoint.json`` / ``cells_served.json`` /
#: ``positive_controls.json`` carry THIS bring-up attempt's nonce (F3)
ENDPOINT_ATTEMPT_KEY = "standup_attempt_id"
#: where a prior attempt's ``endpoint.json`` goes at the start of a bring-up
STALE_DIR_PREFIX = "stale-"
ROUTING_TABLE_FILE = "routing_table.json"
#: the routing table's ENGINE-NEUTRAL served-positions map (``proxy.SERVED_POSITIONS_KEY``; pinned equal by test —
#: the proxy module is imported lazily here), copied into endpoint.json iff the table carries it
SERVED_POSITIONS_KEY = "served_positions"
#: the position mode a PLAIN LoRA adapter attests from its KIND (``defense_cells.NATIVE_POSITION_MODES[0]``): a PEFT
#: adapter applies at every module call — every position — and carries no residual-slot block to state otherwise
POSITION_MODE_ALL = "all"
assert POSITION_MODE_ALL in NATIVE_POSITION_MODES


def attested_position_mode(dig: Any, *, kind: str) -> Optional[str]:
    """The position mode ONE lora cell's adapter ATTESTS (native-positions-resolve), the value
    ``export_routing_table(position_modes=)`` stamps on its route and into the engine-neutral
    ``served_positions`` map: a residual-slot adapter's from its own ``petri_residual_slot`` block
    (``AdapterDigests.petri_position_mode``, read by ``lora_export.adapter_digests`` — ``all`` /
    ``response`` / ``prompt_last`` / ``prefill_all``); a PLAIN LoRA adapter's — a ``kind`` outside
    :data:`RESIDUAL_SLOT_KINDS` (``lora_all``, ``lora_subtraction``, ``circuit_breaker``, ...), which
    has no residual block by construction — :data:`POSITION_MODE_ALL` from its KIND. Never from the
    engine's name: a RESIDUAL-kind adapter whose block states no mode returns ``None`` (the route
    then carries no ``petri_position_mode`` and the map no entry), and every reader refuses that cell
    by name (``NATIVE_POSITION_MODE_UNATTESTED``) rather than default it."""
    mode = getattr(dig, "petri_position_mode", None)
    if mode is not None:
        return str(mode)
    if str(kind) in RESIDUAL_SLOT_KINDS:
        return None
    return POSITION_MODE_ALL
CHAT_TEMPLATE_FILE = "chat_template.jinja"
SERVER_LOG_FILE = "vllm_server.log"
#: the FIRST launch's engine log, kept beside the served launch's when the
#: KV-derived shrink re-launched (capctrl's spelling)
KV_SHORTFALL_LOG = "vllm_server.kv_shortfall.log"
#: the derived config the cell source reads when a prompted cell was admitted
#: beyond ``explicit_cells`` — the original config is never edited
DERIVED_CONFIG_FILE = "standup_config.yaml"
#: the five positive controls, in the order they run
CONTROL_NAMES = ("noop_certificate", "salt_isolation", "steering_bites",
                 "selection_honored", "adapter_digests", "boundary_refusal")
#: the sixth control's ONE probe body (O49): the proxy admits its marker beside EXACTLY this body
#: (``serving_attestation.BOUNDARY_REFUSAL_CONTROL_MESSAGES`` / ``_KWARGS``) and its own nonce
BOUNDARY_REFUSAL_CONTROL_PROMPT = sa.BOUNDARY_REFUSAL_CONTROL_PROMPT
#: the native residual slot's position-mode vocabulary — the literal twin of
#: ``NATIVE_POSITION_MODES`` is ``defense_cells``'s (imported above and re-exported: the
#: torch-free twin of ``native_residual.POSITION_MODES``, pinned equal by
#: tests/test_restricted_route_requests.py). ``all`` is stage 1; every other mode is a
#: RESTRICTED mode (native-psr-serving §1.8), servable only once
#: ``defense_cells.NATIVE_STAGE2_ARMED`` is True.
NATIVE_POSITION_MODE_ALL = "all"
#: the ``steering_bites`` control's two statistics: the
#: TEXT statistic (greedy ids vs the scale-0 twin, ``fidelity._check_steering_bites``)
#: for a LoRA / all-positions cell whose greedy text a live steer changes; the
#: LOGPROB LIVENESS statistic (``fidelity.check_logprob_steering_bites`` over the
#: common top-K support of steered / scale-0 / scale-0-repeat at the first generated
#: position) for a cell whose greedy text MAY equal base — a constant-kind object
#: (``contracts.LOGPROB_LIVENESS_KINDS``: alphasteer, static_r) or any residual cell
#: at a RESTRICTED position mode (a response-fit PSR writes nothing before the
#: header). A ZERO payload is the known-bad: bit-identical steered / twin logprobs
#: on every probe refuse. The text divergence is still REPORTED beside the logprob
#: verdict (a cell live by logprob and inert by text is a fact about dose).
BITES_STATISTIC_TEXT = "text"
BITES_STATISTIC_LOGPROB = "logprob_liveness"
#: the logprob statistic's knobs — the driver's (``scripts/served_capability_run.py``
#: LOGPROB_LIVENESS_* / LOGPROB_BITES_*), spelled here because that script is not
#: importable from the serving tree; pinned equal by test
LOGPROB_LIVENESS_TOP_LOGPROBS = 20
LOGPROB_LIVENESS_MIN_COMMON_TOKENS = 2
LOGPROB_LIVENESS_MIN_CONCLUSIVE_PROBES = 3
LOGPROB_BITES_MIN_RATIO = 2.0
LOGPROB_BITES_ABS_FLOOR = 1e-3
#: the records' key for the POST-CONTROLS ledger read: after
#: the positive controls (which claim every probe on a restricted-mode route) the
#: worker's per-request failure ledger must hold NO unclaimed entry and no latched
#: fault — a nonzero count means the proxy post-check is not running for some
#: route the worker ledgers, and the standup REFUSES
RESIDUAL_LEDGER_AFTER_CONTROLS_KEY = "residual_ledger_after_controls"
#: the manifest flags a native standup strips before composing (the composer
#: refuses both spellings of the eager pin under lora_native and tolerates a
#: bare --enable-lora only as a documented duplicate)
STRIPPED_MANIFEST_FLAGS = ("--enforce-eager", "--enable-lora")
#: ``EffectiveServing.expected_digests_source`` vocabulary
DIGESTS_SOURCE_CLI = "cli"
DIGESTS_SOURCE_VIEW_PINS = "view_manifest+pins"
#: the geometry ADJUSTMENT kinds (``geometry.ADJUSTMENT_KINDS``)
ADJUSTMENT_KERNEL = sa.LORA_KERNEL_CEILING_REASON      # "lora_kernel_index_ceiling"
ADJUSTMENT_KV = "kv_shrink"
#: the model key the spec EXCLUDES from the native engine (its patched rotary
#: rides the lens plugin)
GLM_MODEL_KEY = "glm46vflash"
GLM_FAMILY = "glm4v"
#: the env var that names the adapter bank ROOT off-cluster: the
#: tinycap launcher exports ``/vol/tinycap/multilora_adapters`` in every container
ADAPTER_BANK_ROOT_ENV = "SISTEER_ADAPTER_BANK_ROOT"
TAG = "[lora-standup]"


class LoraStandupError(RuntimeError):
    """The standup's own refusal classes; ``rc`` is the CLI exit code."""

    rc = EXIT_REFUSED


class ProxyHTTPRefusal(LoraStandupError):
    """An HTTP error reply carrying an OpenAI-style ``error.code`` (the proxy's
    refusals): ``status``, ``code`` and the message, so a control can judge a
    refusal BY ITS NAMED CODE (the boundary-refusal control, O49)."""

    def __init__(self, message: str, *, status: int, code: str, body_text: str,
                 error: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.status, self.code, self.body_text = int(status), str(code), body_text
        #: the error body's structured fields (``worker_error_type`` on a ledgered refusal)
        self.error: Dict[str, Any] = dict(error or {})

    @property
    def worker_error_type(self) -> Optional[str]:
        v = self.error.get("worker_error_type")
        return v if isinstance(v, str) else None


class StandupRefused(LoraStandupError):
    """A refusal BEFORE the positive controls ran (exit 2): geometry, pins,
    an unservable cell, a swapped object, a bank/table/attestation problem —
    or, since native-psr-serving stage 2, the ONE post-controls refusal: the
    worker's per-request failure ledger not empty / latched after the controls
    (:func:`residual_ledger_after_controls`). The server, if it was
    launched, has been torn down; no ``endpoint.json`` is published."""

    rc = EXIT_REFUSED


class PositiveControlsFailed(LoraStandupError):
    """A positive control FAILED (exit 4). The server has been torn down,
    ``positive_controls.json`` records the verdicts and no ``endpoint.json``
    exists. ``record`` is that dict."""

    rc = EXIT_CONTROLS_FAILED

    def __init__(self, message: str, record: Mapping[str, Any]):
        super().__init__(message)
        self.record = dict(record)


# --------------------------------------------------------------------------- #
# the adapter bank                                                              #
# --------------------------------------------------------------------------- #


def shared_root() -> Optional[Path]:
    """The per-user shared root (``$SHARED``) on whichever cluster this is,
    or None when not running on a cluster."""
    # $SHARED through sisteer.site (the job's SISTEER_SHARED_ROOT, else CLUSTER_SHARED_ROOT, else the site config's cluster
    # block), counted only where it is a directory on this host
    from sisteer import site as _site
    try:
        root = _site.expand_path("$SHARED")
    except (_site.PlaceholderError, _site.SiteConfigError):
        return None
    return root if root.is_dir() else None


def default_adapter_bank(model_key: str) -> Optional[Path]:
    """``$SHARED/exp/multilora_adapters/<model_key>``, or None
    when no shared root exists here — the caller then needs ``--adapter-bank``
    or :data:`ADAPTER_BANK_ROOT_ENV` (:func:`resolve_adapter_bank`)."""
    root = shared_root()
    return None if root is None else root / "exp" / "multilora_adapters" / model_key


def resolve_adapter_bank(explicit: Optional[Path], *, model_key: Optional[str]) -> Path:
    """The content-addressed adapter bank this standup exports into:
    the explicit path when given; else the cluster default when a shared root
    exists; else ``$SISTEER_ADAPTER_BANK_ROOT/<model_key>`` when the env names
    one (the tinycap launcher's ``/vol/tinycap/multilora_adapters``); else a
    refusal naming all three sources. The default and the env need a model
    key — a bank is per model."""
    if explicit is not None:
        return Path(explicit).resolve()
    root = shared_root()
    env_root = os.environ.get(ADAPTER_BANK_ROOT_ENV)
    if root is not None or env_root:
        if not model_key:
            raise StandupRefused(
                "no --adapter-bank and no --model-key: the default bank "
                f"({'$SHARED/exp/multilora_adapters/<model_key>' if root is not None else ADAPTER_BANK_ROOT_ENV + '/<model_key>'}) "
                "is per model; pass --adapter-bank or --model-key")
        if root is not None:
            return (root / "exp" / "multilora_adapters" / model_key).resolve()
        return (Path(env_root) / model_key).resolve()
    raise StandupRefused(
        "no adapter bank: no --adapter-bank given, no cluster shared root "
        f"($SHARED: SISTEER_SHARED_ROOT or the site config's cluster block) to default it under, and {ADAPTER_BANK_ROOT_ENV} "
        "is unset (the tinycap launcher exports it inside every container)")


# --------------------------------------------------------------------------- #
# effective launch flags, probes, the pure helpers                              #
# --------------------------------------------------------------------------- #


def effective_launch_flags(flags: Sequence[str]) -> Tuple[List[str], List[str]]:
    """``(kept, dropped)``: the manifest's ``serving_launch_flags`` minus every
    :data:`STRIPPED_MANIFEST_FLAGS` token (recorded as ``manifest_flags_dropped``).
    Only bare tokens are dropped — a flag that takes a value is never one of
    these. The ONE place the effective native flags are derived: the standup
    hashes what a client (``petri_run.py --serving-path lora_native``) hashes."""
    kept = [f for f in flags if f not in STRIPPED_MANIFEST_FLAGS]
    dropped = [f for f in flags if f in STRIPPED_MANIFEST_FLAGS]
    return kept, dropped


def probe_prompt_set() -> Tuple[Tuple[Dict[str, str], ...], ...]:
    """The standup's probe set: ``served_target.PROBE_PROMPTS``
    as single user turns + the §10 gate's frozen ``GATE_PROMPT_SET``."""
    from petri_serving_gate import GATE_PROMPT_SET
    from sisteer.eval.attacks.served_target import PROBE_PROMPTS

    single = tuple(({"role": "user", "content": p},) for p in PROBE_PROMPTS)
    return single + tuple(tuple(dict(m) for m in msgs) for msgs in GATE_PROMPT_SET)


def probe_set_hash() -> str:
    from sisteer.capeval import manifest as cm
    return cm.sha256_hex16(cm.canonical_json(
        [list(p) for p in probe_prompt_set()]).encode("utf-8"))


def _http_json(method: str, url: str, *, timeout_s: float,
               payload: Optional[Mapping[str, Any]] = None,
               headers: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """One JSON round trip to the proxy / the upstream; every HTTP or
    transport failure is a :class:`LoraStandupError` naming the call."""
    data = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
    hdrs = dict(headers or {})
    if payload is not None:
        hdrs = {"Content-Type": "application/json", **hdrs}
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")[:2000]
        try:
            err = json.loads(text)["error"]
            code = err["code"]
        except Exception:  # noqa: BLE001 — not an OpenAI-style error body: the plain refusal below
            err, code = None, None
        if isinstance(code, str) and code:
            raise ProxyHTTPRefusal(f"{method} {url} -> HTTP {e.code} ({code}): {text}",
                                   status=e.code, code=code, body_text=text,
                                   error=err if isinstance(err, Mapping) else None) from e
        raise LoraStandupError(f"{method} {url} -> HTTP {e.code}: {text}") from e
    except (urllib.error.URLError, OSError) as e:
        raise LoraStandupError(f"{method} {url} failed: {type(e).__name__}: {e}") from e


def _post_json(url: str, payload: Mapping[str, Any], *, timeout_s: float,
               headers: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    return _http_json("POST", url, timeout_s=timeout_s, payload=payload, headers=headers)


def _get_json(url: str, *, timeout_s: float,
              headers: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    return _http_json("GET", url, timeout_s=timeout_s, headers=headers)


def _token_ids(resp: Mapping[str, Any]) -> List[int]:
    """``choices[0].token_ids`` — the raw ids the exact comparisons rule on."""
    from petri_serving_gate import GateError, served_token_ids
    try:
        return served_token_ids(resp)
    except GateError as e:
        raise LoraStandupError(str(e)) from e


def _adapter_block(name: str, adapter_dir: Path, digests: Any, *, intervention_hash: str,
                   zero: bool) -> Dict[str, Any]:
    """ONE version-2 adapter block (``proxy.ADAPTER_BLOCK_KEYS``) off the bank's
    ``lora_export.AdapterDigests`` — the genuine cells' and the zero leg's alike."""
    return {"name": name, "dir": str(Path(adapter_dir).resolve()),
            "config_sha256": digests.config_sha256, "weights_sha256": digests.weights_sha256,
            "r": int(digests.r), "petri_intervention_hash": intervention_hash,
            "petri_zero_adapter": zero, "petri_source_kind": digests.petri_source_kind}


#: THE STANDUP'S OWN KIND TAG (2026-09-30): every request the native standup sends through its proxy — the positive
#: controls' solo greedy decodes, the mixed batches, the bites probes — carries ``X-Tinycap-Task: probe/standup_controls`` (the
#: recorder's kind for the standup's controls, ``record_resolved_sampler.PROBE_KIND_TABLE["standup_controls"]``, whose cap is the
#: standup's own --probe-max-tokens) and is written, BEFORE the send, as a request-evidence row (``contracts.REQUEST_EVIDENCE_FIELDS``'s
#: shape, version 2) to ``<out>/positive_controls.request_evidence.jsonl`` — the same mechanism the load check uses
#: (scripts/sdfbase/full22/load_check.py + harness_evidence), so the recorder reads it through --probe-tags (or discovers it beside
#: the server log). Before this the standup stamped no tag and every gsm8k job reported its controls as UNCLASSIFIED_REQUESTS.
#: A row is committed AFTER the response with its outcome (`rejected: false` on a 2xx; `rejected: true` + status / error code when the
#: proxy refused or the transport failed — the recorder skips those), and the sidecar is per ATTEMPT (quarantined at bring-up, each row
#: stamped with the attempt id the recorder filters on) — a tag must bind a request that reached the engine, and an
#: old attempt's tags must never count against a new log.
STANDUP_CONTROLS_KIND = "standup_controls"
STANDUP_REQUEST_EVIDENCE_FILE = "positive_controls.request_evidence.jsonl"
_STANDUP_EVIDENCE_COUNTERS: Dict[str, Dict[str, int]] = {}


def prepare_standup_request_evidence(out: Path, payload: Mapping[str, Any], *, attempt_id: Optional[str] = None,
                                     kind: str = STANDUP_CONTROLS_KIND) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Tag one outbound standup request: returns ``(headers, row)`` — the headers to send (``X-Tinycap-Task``, ``X-Tinycap-Sample:
    <vmodel>/<n>``, ``X-Tinycap-Request-Id``) and the request-evidence row (``task`` + ``max_tokens``, the body's digest, the attempt)
    that :func:`commit_standup_request_evidence` appends ONCE THE OUTCOME IS KNOWN. Nothing is written here."""
    import hashlib
    import uuid
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    vmodel = str(payload.get("model") or "-")
    counters = _STANDUP_EVIDENCE_COUNTERS.setdefault(str(Path(out)), {})
    n = counters.get(vmodel, 0)
    counters[vmodel] = n + 1
    request_id = uuid.uuid4().hex
    task = f"probe/{kind}"
    headers = {"X-Tinycap-Task": task, "X-Tinycap-Sample": f"{vmodel}/{n}", "X-Tinycap-Request-Id": request_id}
    cap = payload.get("max_tokens", payload.get("max_completion_tokens"))
    kwargs = payload.get("chat_template_kwargs") if isinstance(payload.get("chat_template_kwargs"), Mapping) else {}
    row = {"request_evidence_version": 2, "ts": _utc_now(), "kind": "probe", "task": task, "sample": headers["X-Tinycap-Sample"],
           "vmodel": vmodel, "petri_thinking_on_present": "petri_thinking_on" in payload, "petri_thinking_on_value": payload.get("petri_thinking_on"),
           "enable_thinking_present": "enable_thinking" in kwargs, "enable_thinking_value": kwargs.get("enable_thinking"),
           "body_sha256": hashlib.sha256(raw).hexdigest(), "request_id": request_id, "body_bytes": len(raw),
           "max_tokens": (int(cap) if isinstance(cap, int) and not isinstance(cap, bool) else cap), "attempt_id": attempt_id}
    return headers, row


def commit_standup_request_evidence(out: Path, row: Mapping[str, Any], *, status: Optional[int], rejected: bool,
                                    error_code: Optional[str] = None, engine_request_id: Optional[str] = None) -> Dict[str, Any]:
    """Append the row with its OUTCOME: ``rejected: false`` only after a 2xx — the request reached the engine and
    is in the server log — else ``rejected: true`` with the HTTP status / error code (the recorder skips a rejected row by its own
    rule, so a control the proxy refused, or a transport failure before vLLM logged anything, never binds a logged request). One
    ``write`` of one line."""
    out = Path(out)
    # an ACCEPTED row carries the ENGINE's request id — the response's `id` (chatcmpl-<hex>), which the
    # proxy forwards unchanged and vLLM logs as `Received request <id>` — so the recorder joins this tag to ONE logged request by
    # identity, never by cap equality; None when the response carried none (the recorder then refuses the row by name)
    full = {**row, "rejected": bool(rejected), "status": status, "error_code": error_code, "committed_ts": _utc_now()}
    if not rejected:
        full["engine_request_id"] = engine_request_id
    out.mkdir(parents=True, exist_ok=True)
    with open(out / STANDUP_REQUEST_EVIDENCE_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(full, sort_keys=True) + "\n")
    return full


def tagged_standup_chat(post: Any, url: str, payload: Mapping[str, Any], *, out: Path, attempt_id: Optional[str],
                        timeout_s: float) -> Dict[str, Any]:
    """The standup's ONE tagged HTTP boundary: prepare the tag, ``post(url, payload, timeout_s=, headers=)``, then commit the
    evidence row with the outcome — accepted after the response, ``rejected`` (status / error code) when the post raised — and
    re-raise the failure unchanged."""
    headers, row = prepare_standup_request_evidence(out, payload, attempt_id=attempt_id)
    try:
        resp = post(url, payload, timeout_s=timeout_s, headers=headers)
    except ProxyHTTPRefusal as e:
        commit_standup_request_evidence(out, row, status=getattr(e, "status", None), rejected=True, error_code=getattr(e, "code", None))
        raise
    except LoraStandupError as e:
        commit_standup_request_evidence(out, row, status=None, rejected=True, error_code=type(e).__name__)
        raise
    except BaseException:
        commit_standup_request_evidence(out, row, status=None, rejected=True, error_code="transport")
        raise
    rid = resp.get("id") if isinstance(resp, Mapping) else None
    commit_standup_request_evidence(out, row, status=200, rejected=False, engine_request_id=(rid if isinstance(rid, str) and rid else None))
    return resp


def quarantine_prior_standup_evidence(out: Path, *, attempt_id: str, stale_dir: Optional[Path] = None) -> Optional[Path]:
    """A PRIOR attempt's tag sidecar under ``out`` is moved aside (into the endpoint's stale dir when one was just made, else a
    fresh ``stale-<utc>-<nonce8>/``) BEFORE this attempt sends anything: the sidecar is per attempt, like endpoint.json — a restart in
    the same serving directory must not lend the old attempt's tags to the new log. A rename, never a deletion."""
    out = Path(out)
    prior = out / STANDUP_REQUEST_EVIDENCE_FILE
    if not prior.exists():
        return None
    dest = stale_dir
    if dest is None:
        tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + attempt_id[:8]
        dest = out / f"{STALE_DIR_PREFIX}{tag}"
        i = 0
        while dest.exists():
            i += 1
            dest = out / f"{STALE_DIR_PREFIX}{tag}-{i}"
        dest.mkdir(parents=True)
    os.replace(prior, dest / STANDUP_REQUEST_EVIDENCE_FILE)
    _STANDUP_EVIDENCE_COUNTERS.pop(str(out), None)
    print(f"{TAG} a prior attempt's {STANDUP_REQUEST_EVIDENCE_FILE} under {out} was moved aside to {dest} "
          "(the tag sidecar is this attempt's alone)", flush=True)
    return dest


def quarantine_prior_endpoint(out: Path, *, attempt_id: str) -> Optional[Path]:
    """F3: move a PRIOR attempt's ``endpoint.json`` under ``out`` into
    ``out/stale-<utc>-<nonce8>/`` BEFORE this attempt does anything. A rename,
    never a deletion. Returns the stale dir, or ``None``."""
    out = Path(out)
    prior = out / ENDPOINT_FILE
    if not prior.exists():
        return None
    tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + attempt_id[:8]
    dest = out / f"{STALE_DIR_PREFIX}{tag}"
    i = 0
    while dest.exists():
        i += 1
        dest = out / f"{STALE_DIR_PREFIX}{tag}-{i}"
    dest.mkdir(parents=True)
    os.replace(prior, dest / ENDPOINT_FILE)
    print(f"{TAG} a prior attempt's {ENDPOINT_FILE} under {out} was moved aside to {dest} "
          "(a crashed standup's readiness record is never this attempt's)", flush=True)
    return dest


def unpublish_endpoint(out: Path) -> bool:
    """Remove ``out/endpoint.json`` (this attempt's readiness record)."""
    try:
        (Path(out) / ENDPOINT_FILE).unlink()
        return True
    except FileNotFoundError:
        return False


def _stop_stack(proxy_loop: Any, server: Any) -> None:
    """Stop the proxy's event loop and shut the vLLM child down (bounded), each
    best-effort — the ONE teardown behind ``LoraStandup.tear_down`` and the
    bring-up's failure path."""
    from sisteer.capeval.serving import vllm_server as vs
    if proxy_loop is not None:
        try:
            proxy_loop.call_soon_threadsafe(proxy_loop.stop)
        except Exception:  # noqa: BLE001
            pass
    try:
        vs._shutdown_child(server)
    except Exception:  # noqa: BLE001
        pass


def _start_proxy_thread(app: Any, host: str, port: int) -> Any:
    """Serve the proxy's aiohttp app on a daemon thread (capctrl's helper, one
    copy in the core so the fronts import it back)."""
    import asyncio
    import threading

    from aiohttp import web
    ready = threading.Event()
    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, host, port).start())
        ready.set()
        loop.run_forever()
    threading.Thread(target=_run, name="proxy", daemon=True).start()
    if not ready.wait(timeout=30):
        raise RuntimeError("proxy thread not ready")
    return loop


def free_port() -> int:
    """A free loopback port (bind 0, read it back) — the one helper every
    launch-only caller (the onboarding gate, the serving probe) uses too."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def write_chat_template(out: Path, template_bytes: bytes) -> Tuple[Path, str]:
    """Materialise the SERVED chat template as ``<out>/chat_template.jinja``
    and return ``(path, sha16)`` — the bytes vLLM is handed and the hash the
    spec's ``chat_template_hash`` is held to (``manifest.sha256_hex16``)."""
    from sisteer.capeval import manifest as cm   # lazy, as every petri import in this module
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tpath = out / CHAT_TEMPLATE_FILE
    tpath.write_bytes(template_bytes)
    return tpath, cm.sha256_hex16(template_bytes)


# --------------------------------------------------------------------------- #
# the effective serving (what a front hands the core)                          #
# --------------------------------------------------------------------------- #


#: the cells a caller NAMED for this standup -- petri_proxy's positional 5 / ``mt_serving_up --cell`` --
#: recorded beside the SERVED list in the pre-launch record, the final ``cells_served.json`` and ``endpoint.json``. The CORE writes them
#: (a front hands the normalised list in on :class:`EffectiveServing`; they are core keys, so no front's extra fact may carry them).
#: ``cells_requested_explicit`` is the roster / TrainTest spelling as named (variants kept, order kept, de-duplicated) and
#: ``cells_requested_explicit_harness`` the same through ``registry_adapter.harness_cell_id``; both ``[]`` when nothing was named.
#: ``none`` appears only when it was NAMED: ``select_cells_lora`` serves it implicitly and lists it first in ``cells_requested``, which
#: is why a drive restricting its legs to the request reads THIS field (the Petri drive derives its controls twin from ``none`` here,
#: never from the served list).
REQUESTED_EXPLICIT_KEY = "cells_requested_explicit"
REQUESTED_EXPLICIT_HARNESS_KEY = "cells_requested_explicit_harness"


def requested_cells_explicit(requested: Optional[Sequence[Any]]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """``(roster spellings, harness ids)`` of exactly what ``requested`` (the repeatable ``--cell``) NAMED: stripped, order kept,
    de-duplicated, ``none`` kept only when named; ``((), ())`` for None / empty. A blank or non-string entry refuses, as does a
    spelling the harness grammar cannot translate (``registry_adapter.harness_cell_id``'s refusal, re-raised as a standup refusal)."""
    from sisteer.capeval.petri import registry_adapter as ra
    names: List[str] = []
    for raw in (requested or ()):
        if not isinstance(raw, str) or not raw.strip():
            raise StandupRefused(f"--cell needs a non-empty TrainTest cell id, got {raw!r}")
        cell = raw.strip()
        if cell not in names:
            names.append(cell)
    try:
        harness = tuple(ra.harness_cell_id(c) for c in names)
    except ra.RegistryAdapterError as e:
        raise StandupRefused(f"--cell {names}: {e}") from e
    return tuple(names), harness


@dataclasses.dataclass(frozen=True)
class EffectiveServing:
    """What ONE front normalised for the core: the EFFECTIVE spec
    (model / revision / cells / template hash AS SERVED; launch flags already
    through :func:`effective_launch_flags`), the two cell sources, the served
    and skipped cells, the tokenizer + template bytes, the config the source
    read, the bank view, the model identity and the expected digests."""

    spec: Any                          # cm.PetriSpec, EFFECTIVE
    cell_source: Any                   # TrainTestCellSource(serving_path_policy="lora_native") over the served cells
    lens_source: Any                   # the lens-policy source the partition read (kind/spec evidence; never exported)
    served_cells: Tuple[str, ...]      # harness ids incl. none / prompted (== spec.cells)
    cells_skipped: Tuple[SkippedCell, ...]
    tokenizer: Any
    chat_template_bytes: bytes
    config_used: Path
    defenses_dir: Path
    model_key: Optional[str]
    model: str
    revision: str
    expected_digests: Mapping[str, str]          # harness id -> sha256(object.pt), fitted cells only
    expected_digests_source: str                  # DIGESTS_SOURCE_CLI | DIGESTS_SOURCE_VIEW_PINS
    # -- additive (W-B): what enumeration needs and what the records name ----
    behavior_files: Mapping[str, bytes] = dataclasses.field(default_factory=dict)
    prompt_files: Mapping[str, bytes] = dataclasses.field(default_factory=dict)
    traintest_cells: Tuple[str, ...] = ()        # the TrainTest spellings of served_cells (records: `cells`)
    prompted_cells_admitted: Tuple[str, ...] = ()
    #: what the caller NAMED (:func:`requested_cells_explicit`) -- roster spellings and harness ids; () / () when nothing was
    cells_requested_explicit: Tuple[str, ...] = ()
    cells_requested_explicit_harness: Tuple[str, ...] = ()
    manifest_path: Optional[Path] = None
    config_path: Optional[Path] = None
    manifest_flags_dropped: Tuple[str, ...] = ()
    #: the front's own name for the records (``lora_serving_up`` | ``capctrl_serving_up`` | ...)
    front: str = "native_standup"


# --------------------------------------------------------------------------- #
# the live standup                                                             #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class LoraStandup:
    """One LIVE lora-native standup — the plugin-free vLLM server + the
    RoutingProxy in front of it — and every record a client (the drivers,
    the certification harness, the §10 gate's lora-native canary, capctrl's
    ``main``) reads. ONE shape for both fronts."""

    out: Path
    model: str
    revision: str
    model_key: Optional[str]
    eval: Optional[str]
    geometry: Any                 # geometry.ResolvedGeometry
    launch: Any                   # vllm_server.ServerLaunch
    server: Any                   # the vllm serve child (subprocess.Popen)
    proxy: Any                    # proxy.RoutingProxy
    proxy_loop: Any
    proxy_url: str
    proxy_port: int
    vllm_port: int
    table_path: Path
    table: Mapping[str, Any]
    tokenizer: Any
    manifest_path: Optional[Path]
    config_path: Optional[Path]   # the config as given
    config_used: Path             # the config the cell source read (derived when a cell was admitted)
    defenses_dir: Path
    adapter_bank: Path
    cells: Tuple[str, ...]        # the TrainTest cells (none implicit, first)
    harness_cells: Tuple[str, ...]
    lora_cells: Tuple[str, ...]   # harness ids of the GENUINE lora cells (never the zero leg)
    prompted_cells_admitted: Tuple[str, ...]
    vmodels: Dict[str, str]       # cell_id -> virtual model
    adapters: Dict[str, Dict[str, Any]]   # cell_id -> adapter block (genuine cells)
    zero_adapter: Dict[str, Any]          # the none__lora0 block
    adapters_reused: Dict[str, bool]
    object_artifact_sha256_by_cell: Dict[str, str]   # harness ids -> sha256(object.pt)
    attestation_path: Path
    attestation: Mapping[str, Any]
    engine_identity: Mapping[str, Any]
    chat_template_hash: str       # sha16 (the manifest pin)
    chat_template_sha256: str     # full sha256 over the served template bytes
    manifest_flags_dropped: Tuple[str, ...]
    geometry_override_reason: Optional[str]
    tool_calls: bool
    petri_agent_loop: bool
    upstream_timeout_s: Optional[float]
    probe_max_tokens: int
    max_logprobs: Optional[int]
    gpu_mem_util: str
    cell_set: Any                 # defense_cells.PetriCellSet
    thinking_header: Any
    thinking_header_error: Optional[str]
    attempt_id: str                # this bring-up's nonce (F3); every record carries it
    mode: Optional[str] = None
    positive_controls: Optional[Dict[str, Any]] = None
    #: what the caller NAMED, off EffectiveServing -- recorded in both records beside the served list
    cells_requested_explicit: Tuple[str, ...] = ()
    cells_requested_explicit_harness: Tuple[str, ...] = ()
    #: the LoRA kernel index ceiling this standup was sized under and, when the
    #: requested S exceeded it, the lowering (byte-identical to the pre-core records)
    kernel_ceiling: Optional[Dict[str, Any]] = None
    geometry_lowered: Optional[Dict[str, Any]] = None
    #: the adapter-parity gate's verdict (geometry.standup_adapter_parity_gate: status, pairs, the residual carriers
    #: with their certificate, the record predicate) — a CERTIFICATION fact every readiness record carries (the steered canary
    #: 818207 read an endpoint without it: the verdict had lived only in the pre-launch cells_served.json, overwritten at readiness)
    adapter_parity_gate: Optional[Dict[str, Any]] = None
    started: float = dataclasses.field(default_factory=time.time)
    # -- the native-serving-default additions, each defaulted ----------------
    cells_skipped: Tuple[SkippedCell, ...] = ()
    #: the §1.8 record: ``[]`` (identity) | ``[kernel]`` | ``[kv]`` | ``[kernel, kv]``, as applied
    geometry_adjustments: Tuple[Dict[str, Any], ...] = ()
    kv_shrink: Optional[Dict[str, Any]] = None
    thinking_mode: str = "off"
    thinking_certification: Optional[Dict[str, Any]] = None
    reasoning_parser: Optional[str] = None
    sampling_clamp_override: Optional[Dict[str, Any]] = None
    sampling_clamp_override_mechanism: Any = None
    text_only_mm: Optional[Dict[str, Any]] = None
    enable_log_requests: bool = False
    #: the SERVER-DEFAULT SAMPLING POLICY the launch was composed under (vllm_server.SERVER_DEFAULT_POLICIES)
    server_default_policy: str = "contract_override"
    mrope_attestation: Optional[Dict[str, Any]] = None
    mrope_attestation_error: Optional[str] = None
    mamba_backend_attestation: Optional[Dict[str, Any]] = None
    mamba_backend_attestation_error: Optional[str] = None
    expected_digests_source: str = DIGESTS_SOURCE_CLI
    front: str = "native_standup"
    #: a front's OWN provenance facts (mt's serving_config sha, driver cap, budget record, bank
    #: objects ...), recorded VERBATIM in both records beside the core's keys — never over one
    extra_facts: Dict[str, Any] = dataclasses.field(default_factory=dict)
    #: native-psr-serving: whether the controls compute the output-space (greedy
    #: agreement) diagnostics — False iff a served cell is a residual-slot kind
    output_diagnostics: bool = True
    #: native-psr-serving §1.5: ``{"armed": bool, "cells": {cell: {kind, residual_fingerprint,
    #: position_mode}}}`` off the readiness attestation (None before it is read)
    residual_slot: Optional[Dict[str, Any]] = None
    #: native-psr-serving §1.8 (fact 15): the worker's ledger state read AFTER the
    #: positive controls (:func:`residual_ledger_after_controls`) — ``None`` until the
    #: controls ran; ``{"checked": False, "reason"}`` on an unarmed (LoRA-only) standup
    residual_ledger_after_controls: Optional[Dict[str, Any]] = None
    #: results-registry native emission (additive): ``{served cell id: {run_id, campaign, expected: {...}}}``
    #: — the registry runs this standup serves, handed in by the launcher; None → the hook reads
    #: ``<out>/registry_runs.json`` when present, else records nothing (one notice when emission is on)
    registry_runs: Optional[Mapping[str, Mapping[str, Any]]] = None

    # -- paths ------------------------------------------------------------------

    @property
    def positive_controls_path(self) -> Path:
        return self.out / POSITIVE_CONTROLS_FILE

    @property
    def cells_served_path(self) -> Path:
        return self.out / CELLS_SERVED_FILE

    @property
    def endpoint_path(self) -> Path:
        return self.out / ENDPOINT_FILE

    @property
    def server_log(self) -> Path:
        return self.out / SERVER_LOG_FILE

    @property
    def routes(self) -> Tuple[Any, ...]:
        return tuple(self.table["routes"])

    @property
    def routes_by_cell(self) -> Dict[str, Any]:
        return {r.cell_id: r for r in self.routes}

    @property
    def zero_vmodel(self) -> str:
        from sisteer.capeval.petri.defense_cells import NOOP_LORA_CELL_ID
        return self.vmodels[NOOP_LORA_CELL_ID]

    @property
    def base_vmodel(self) -> str:
        return self.vmodels["none"]

    @property
    def zero_adapter_only(self) -> bool:
        """A standup with no GENUINE lora cell: the zero adapter is the one
        registered adapter (a none-only cell set, a prompted-only view, a view
        whose objects were all skipped)."""
        return not self.lora_cells

    def server_rc(self) -> Optional[int]:
        proc = self.server
        if not hasattr(proc, "poll"):
            proc = getattr(proc, "proc", None) or getattr(proc, "process", None)
        return proc.poll() if proc is not None else None

    # -- the client surface -----------------------------------------------------

    def chat(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """One chat completion through THIS standup's proxy (the same path
        every driver uses). Every request the standup itself sends — the
        positive controls' solo greedy decodes, the mixed batches, the bites
        probes — is TAGGED at this boundary the way the drivers tag theirs
        (``X-Tinycap-Task: probe/standup_controls``) and written to
        ``<out>/positive_controls.request_evidence.jsonl`` BEFORE it is sent,
        so ``record_resolved_sampler.py --probe-expect kinds`` classifies the
        standup's own traffic by kind instead of reporting it UNCLASSIFIED
        (lane B2's finding, 2026-09-30)."""
        timeout = float(self.upstream_timeout_s or 600.0) + 30.0
        return tagged_standup_chat(_post_json, f"{self.proxy_url}/chat/completions", payload, out=self.out,
                                   attempt_id=self.attempt_id, timeout_s=timeout)

    def greedy_ids(self, vmodel: str, messages: Sequence[Mapping[str, str]]) -> List[int]:
        """Greedy generation of ``messages`` on ``vmodel``, RAW token ids
        (``return_token_ids``), ``--probe-max-tokens`` long."""
        return self.greedy_ids_with_request_id(vmodel, messages)[0]

    def greedy_ids_with_request_id(self, vmodel: str, messages: Sequence[Mapping[str, str]]
                                   ) -> Tuple[List[int], str]:
        """:meth:`greedy_ids` plus the response's ``id`` — the EXTERNAL request
        id the RoutingProxy forwards unchanged; the selection control (d) joins
        the worker's routing observations to it."""
        resp = self.chat({"model": vmodel, "messages": [dict(m) for m in messages],
                          "max_tokens": int(self.probe_max_tokens), "temperature": 0,
                          "return_token_ids": True})
        rid = resp.get("id")
        if not isinstance(rid, str) or not rid:
            raise LoraStandupError(
                f"the proxied response for {vmodel!r} carries no string 'id' — the engine "
                "request id is what the selection control's routing evidence is keyed by; "
                "refusing")
        return _token_ids(resp), rid

    def upstream_models(self) -> Dict[str, Any]:
        """The SERVER's own ``/v1/models`` (authenticated), for the digest control."""
        from sisteer.capeval.serving.proxy import _canonical_v1_base
        return _get_json(f"{_canonical_v1_base(self.launch.base_url)}/models", timeout_s=60.0,
                         headers={"Authorization": f"Bearer {self.launch.api_key}"})

    # -- the records ------------------------------------------------------------

    def _adapter_summary(self) -> Dict[str, Any]:
        return {"adapters": {k: dict(v) for k, v in self.adapters.items()},
                "zero_adapter": dict(self.zero_adapter),
                "adapters_reused": dict(self.adapters_reused)}

    def _geometry_block(self) -> Dict[str, Any]:
        """``geometry.as_dict()`` + the §1.8 ``adjustments`` list (+ capctrl's
        ``kv_shrink`` key inside the block when the KV step fired, the shape
        ``served_geometry_of_endpoint`` reads today)."""
        block = dict(self.geometry.as_dict())
        block["adjustments"] = [dict(a) for a in self.geometry_adjustments]
        if self.kv_shrink is not None:
            block["kv_shrink"] = dict(self.kv_shrink)
        return block

    def _common_facts(self) -> Dict[str, Any]:
        """The native-serving-default facts BOTH records carry (endpoint.json and
        cells_served.json): the thinking regime, the served engine spelling, the
        skipped cells, the attestations asked at readiness, the launch pins."""
        return {
            "serving_engine": ENGINE,
            "cells_skipped": [dict(s._asdict()) for s in self.cells_skipped],
            "thinking_mode": self.thinking_mode,
            "thinking_declaration_key": "petri_thinking_on",
            "thinking_certification": self.thinking_certification,
            "thinking_declared_cells": [],
            "default_chat_template_kwargs": getattr(self.launch, "default_chat_template_kwargs", None),
            # THE SERVER-SIDE SAMPLING DEFAULT: what build_server_launch composed for a model
            # whose pinned snapshot ships no generation_config.json; None = the model's own file governs
            "override_generation_config": getattr(self.launch, "override_generation_config", None),
            # ... and the POLICY it was composed under: `engine_defaults` = None above BY CHOICE (the engine's own
            # resolution of an omitted key governs), `contract_override` = the pinned row, or None for an unpinned model
            "server_default_policy": self.server_default_policy,
            "reasoning_parser": self.reasoning_parser,
            "upstream_timeout_s": self.upstream_timeout_in_force(),
            "text_only_mm": (self.text_only_mm.get("text_only_mm")
                             if isinstance(self.text_only_mm, dict) else None),
            "text_only_mm_record": self.text_only_mm,
            "enable_log_requests": bool(self.enable_log_requests),
            "mrope_attestation": self.mrope_attestation,
            "mrope_attestation_error": self.mrope_attestation_error,
            "mamba_backend_attestation": self.mamba_backend_attestation,
            "mamba_backend_attestation_error": self.mamba_backend_attestation_error,
            "kv_shrink": self.kv_shrink,
            "expected_digests_source": self.expected_digests_source,
            "zero_adapter_only": bool(self.zero_adapter_only),
            # the lens facts a native engine cannot attest, RECORDED not_applicable (never
            # `lens_reduction_mode`: a reader that finds that key reads a lens standup)
            "reduction_mode": REDUCTION_MODE_NOT_APPLICABLE,
            "lens_salt_policy": NOT_APPLICABLE,
            "require_attested_mode": None,
            "standup_front": self.front,
            # native-psr-serving: the diagnostics policy and the residual cells served
            "output_diagnostics": bool(self.output_diagnostics),
            "residual_slot": self.residual_slot,
            # the venv this standup runs in (variant + requirements hash + the kernels it imports)
            sa.SERVING_VENV_KEY: sa.serving_venv_facts(),
            **({"sampling_clamp_override": dict(self.sampling_clamp_override),
                "sampling_clamp_override_mechanism": self.sampling_clamp_override_mechanism}
               if self.sampling_clamp_override is not None else {}),
        }

    def upstream_timeout_in_force(self) -> float:
        """The proxy -> vLLM hop timeout IN FORCE (the flag, else the proxy's own
        default) — mt's ``upstream_timeout_s`` record; the wave's hop guard reads it."""
        if self.upstream_timeout_s is not None:
            return float(self.upstream_timeout_s)
        from sisteer.capeval.serving.proxy import HOP_TIMEOUT_S
        return float(HOP_TIMEOUT_S)

    def cells_served(self) -> Dict[str, Any]:
        return self._cells_served()

    def _cells_served(self) -> Dict[str, Any]:
        att = self.attestation
        att_geom = attested_scheduler_geometry(att)
        return {
            "engine": ENGINE,
            ENDPOINT_ATTEMPT_KEY: self.attempt_id,
            "mode": self.mode, "eval": self.eval, "model_key": self.model_key,
            "model": self.model, "model_revision": self.revision,
            "manifest": str(self.manifest_path), "config": str(self.config_path),
            "config_used": str(self.config_used), "defenses_dir": str(self.defenses_dir),
            "adapter_bank": str(self.adapter_bank),
            "max_logprobs": self.max_logprobs, "gpu_mem_util": self.gpu_mem_util,
            "written_at": _utc_now(),
            "cells_requested": list(self.cells), "harness_cells": list(self.harness_cells),
            # what was NAMED, beside what is served (none implicit above; here only when named)
            REQUESTED_EXPLICIT_KEY: list(self.cells_requested_explicit),
            REQUESTED_EXPLICIT_HARNESS_KEY: list(self.cells_requested_explicit_harness),
            "lora_cells": list(self.lora_cells),
            "prompted_cells_admitted": list(self.prompted_cells_admitted),
            "manifest_flags_dropped": list(self.manifest_flags_dropped),
            "geometry_override_reason": self.geometry_override_reason,
            "geometry": self._geometry_block(),
            "geometry_lowered": self.geometry_lowered,
            sa.LORA_KERNEL_CEILING_KEY: self.kernel_ceiling,
            "adapter_parity_gate": self.adapter_parity_gate,
            "thinking_header": self.thinking_header,
            "thinking_header_error": self.thinking_header_error,
            "chat_template_hash": self.chat_template_hash,
            "chat_template_sha256": self.chat_template_sha256,
            "probe_max_tokens": self.probe_max_tokens,
            "object_artifact_sha256_by_cell": dict(self.object_artifact_sha256_by_cell),
            "expected_object_sha256_by_cell": dict(self.object_artifact_sha256_by_cell),
            **self._adapter_summary(),
            "routing_table": str(self.table_path),
            "routing_table_version": int(self.table["version"]),
            "serving_attestation": str(self.attestation_path),
            "attestation": {"enforce_eager": att["engine"]["enforce_eager"],
                            "cudagraph_mode": att["engine"]["cudagraph_mode"],
                            "cudagraph_full_capture_max": att["engine"].get("cudagraph_full_capture_max"),
                            "lora_config": dict(att["lora_config"]),
                            "id_name_map": dict(att["id_name_map"]),
                            sa.MODEL_RUNNER_KEY: att.get(sa.MODEL_RUNNER_KEY)},
            "attested_max_num_seqs": att_geom[0] if att_geom else None,
            "attested_max_model_len": att_geom[1] if att_geom else None,
            "positive_controls": self.positive_controls,
            RESIDUAL_LEDGER_AFTER_CONTROLS_KEY: self.residual_ledger_after_controls,
            "canonical": [dataclasses.asdict(c) for c in self.cell_set.canonical],
            "controls": [dataclasses.asdict(c) for c in self.cell_set.controls],
            **self._common_facts(),
            **dict(self.extra_facts),
        }

    def endpoint(self, eval_name: Optional[str] = None) -> Dict[str, Any]:
        """The READINESS record. ``eval_name`` (capctrl's ``main`` passes its
        ``--eval``) is accepted for the shared shape; the standup's own eval wins
        when it has one."""
        att = self.attestation
        att_geom = attested_scheduler_geometry(att)
        objmap = dict(self.object_artifact_sha256_by_cell)
        return {
            # capctrl's / mt's common keys
            "proxy_url": self.proxy_url, "proxy_port": self.proxy_port,
            "vllm_port": self.vllm_port, "routing_table": str(self.table_path),
            "mode": self.mode, "eval": self.eval if self.eval is not None else eval_name,
            "model": self.model,
            "model_key": self.model_key, "model_revision": self.revision,
            "revision": self.revision,
            "cell": self.cells[-1] if self.cells else "none",
            "harness_cell": self.harness_cells[-1] if self.harness_cells else "none",
            "cells": list(self.cells), "harness_cells": list(self.harness_cells),
            "cells_requested": list(self.cells),
            REQUESTED_EXPLICIT_KEY: list(self.cells_requested_explicit),                    # what was NAMED (both records)
            REQUESTED_EXPLICIT_HARNESS_KEY: list(self.cells_requested_explicit_harness),
            "served_models": list(self.proxy.virtual_models()),
            "pid": os.getpid(), "node": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "max_model_len": self.geometry.max_model_len,
            "max_num_seqs": self.geometry.max_num_seqs,
            "seqs_source": getattr(self.geometry, "seqs_source", None),
            "attested_max_num_seqs": att_geom[0] if att_geom else None,
            "attested_max_model_len": att_geom[1] if att_geom else None,
            "geometry": self._geometry_block(),
            "geometry_override_reason": self.geometry_override_reason,
            "geometry_lowered": self.geometry_lowered,
            sa.LORA_KERNEL_CEILING_KEY: self.kernel_ceiling,
            "adapter_parity_gate": self.adapter_parity_gate,
            "tool_calls": bool(self.tool_calls),
            "agent_loop_entry": bool(self.petri_agent_loop),
            "petri_agent_loop_entry": bool(self.petri_agent_loop),
            "proxy_token": None,
            "proxy_upstream_timeout_s": self.upstream_timeout_s,
            "chat_template_hash": self.chat_template_hash,
            "chat_template_sha256": self.chat_template_sha256,
            "object_artifact_sha256": legacy_object_sha256(objmap),
            "object_artifact_sha256_by_cell": objmap,
            "max_logprobs": self.max_logprobs,
            "written_at": _utc_now(), "started": self.started,
            # the lora-native additions
            "engine": ENGINE,
            ENDPOINT_ATTEMPT_KEY: self.attempt_id,
            "serving_attestation": str(self.attestation_path),
            "cells_served": str(self.cells_served_path),
            "positive_controls": str(self.positive_controls_path),
            "adapter_bank": str(self.adapter_bank),
            **self._adapter_summary(),
            "attested_enforce_eager": att["engine"]["enforce_eager"],
            "attested_cudagraph_mode": att["engine"]["cudagraph_mode"],
            # THE FULL CAPTURE SURFACE (2026-09-16): the largest batch size that
            # dispatches to a FULL graph (the dispatcher's FULL keys) and the list
            # the composer handed to --cudagraph-capture-sizes; common.sh holds the
            # max to the attested max_num_seqs beside attested_cudagraph_mode
            # (S=51 on the default list ran 49-51 PIECEWISE, attention eager)
            "attested_cudagraph_full_capture_max": att["engine"].get("cudagraph_full_capture_max"),
            "attested_cudagraph_full_capture_sizes": att["engine"].get("cudagraph_full_capture_sizes"),
            "composed_cudagraph_capture_sizes": list(getattr(self.launch, "cudagraph_capture_sizes", ()) or ()),
            "attested_model_runner": att.get(sa.MODEL_RUNNER_KEY),
            "lora_config": dict(att["lora_config"]),
            "id_name_map": dict(att["id_name_map"]),
            "vmodels": dict(self.vmodels),
            "lora_cells": list(self.lora_cells),
            "prompted_cells_admitted": list(self.prompted_cells_admitted),
            "manifest_flags_dropped": list(self.manifest_flags_dropped),
            "routing_table_version": int(self.table["version"]),
            "probe_max_tokens": self.probe_max_tokens,
            "thinking_header": self.thinking_header,
            "thinking_header_error": self.thinking_header_error,
            # the ENGINE-NEUTRAL served-positions map the routing table stamped (proxy.SERVED_POSITIONS_KEY, pinned
            # equal by test): every lora route's attested position mode, what the served driver reads FIRST
            # (native-positions-resolve); PRESENT iff the table carries it — one rule with the lens standup's record
            **({SERVED_POSITIONS_KEY: self.table.get(SERVED_POSITIONS_KEY)} if self.table.get(SERVED_POSITIONS_KEY) is not None else {}),
            RESIDUAL_LEDGER_AFTER_CONTROLS_KEY: self.residual_ledger_after_controls,
            **self._common_facts(),
            **dict(self.extra_facts),
        }

    def write_cells_served(self) -> Path:
        sa.write_json_atomic(self.cells_served_path, self._cells_served(), default=str)
        return self.cells_served_path

    def write_endpoint(self) -> Path:
        record = self.endpoint()
        sa.write_json_atomic(self.endpoint_path, record, default=str)
        self._registry_serving_facts(record)
        return self.endpoint_path

    def _registry_serving_facts(self, record: Mapping[str, Any]) -> None:
        """Results-registry native emission (additive, never raises): one
        ``run_patch`` per served run — the OBSERVED serving facts under ``fields``
        (attested S, window, engine version, adapter fingerprint, positions mode, prefix
        cache) and the IMMUTABLE re-observations (``serving_path`` from ``engine``, the
        adapter block's intervention hash, ``model_revision``) under ``verify``, which the
        harvester compares to the run and REFUSES on any mismatch. ``endpoint.json`` is
        written first and is byte-identical with emission on or off."""
        try:
            from sisteer.registry import hooks as _registry_hooks   # stdlib-only
            # the record the hook reads = endpoint.json's content + the engine identity (versions) the
            # standup holds but the readiness record does not carry; endpoint.json itself is untouched
            observed = json.loads(json.dumps(dict(record, engine_identity=self.engine_identity), default=str))
            _registry_hooks.emit_serving_patches(observed, self.registry_runs, out_dir=str(self.out))
        except Exception as e:  # noqa: BLE001 — a lost record never costs a standup
            print(f"{TAG} [registry] serving facts not recorded ({type(e).__name__}: {e})", flush=True)

    def tear_down(self) -> None:
        _stop_stack(self.proxy_loop, self.server)
        unpublish_endpoint(self.out)


def _utc_now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


#: every key the core writes into endpoint.json / cells_served.json (the collision check for a
#: front's extra facts; pinned by test against a real record)
_CORE_RECORD_KEYS = frozenset({
    "proxy_url", "proxy_port", "vllm_port", "routing_table", "mode", "eval", "model", "model_key",
    "model_revision", "revision", "cell", "harness_cell", "cells", "harness_cells", "cells_requested",
    "served_models", "pid", "node", "slurm_job_id", "max_model_len", "max_num_seqs", "seqs_source",
    "attested_max_num_seqs", "attested_max_model_len", "geometry", "geometry_override_reason",
    "geometry_lowered", sa.LORA_KERNEL_CEILING_KEY, "adapter_parity_gate", "tool_calls", "agent_loop_entry",
    "petri_agent_loop_entry", "proxy_token", "proxy_upstream_timeout_s", "chat_template_hash",
    "chat_template_sha256", "object_artifact_sha256", "object_artifact_sha256_by_cell",
    "expected_object_sha256_by_cell", "max_logprobs", "written_at", "started", "engine",
    ENDPOINT_ATTEMPT_KEY, "serving_attestation", "cells_served", "positive_controls", "adapter_bank",
    "adapters", "zero_adapter", "adapters_reused", "attested_enforce_eager", "attested_cudagraph_mode",
    "attested_cudagraph_full_capture_max", "attested_cudagraph_full_capture_sizes",
    "composed_cudagraph_capture_sizes",
    "attested_model_runner", "lora_config", "id_name_map", "vmodels", "lora_cells",
    "prompted_cells_admitted", REQUESTED_EXPLICIT_KEY, REQUESTED_EXPLICIT_HARNESS_KEY,
    "manifest_flags_dropped", "routing_table_version", "probe_max_tokens",
    "thinking_header", "thinking_header_error", "served_positions", "manifest", "config", "config_used", "defenses_dir",
    "gpu_mem_util", "attestation", "canonical", "controls", "serving_engine", "cells_skipped",
    "thinking_mode", "thinking_declaration_key", "thinking_certification", "thinking_declared_cells",
    "default_chat_template_kwargs", "override_generation_config", "server_default_policy", "reasoning_parser",
    "upstream_timeout_s", "text_only_mm",
    "text_only_mm_record", "enable_log_requests", "mrope_attestation", "mrope_attestation_error",
    "mamba_backend_attestation", "mamba_backend_attestation_error", "kv_shrink",
    "expected_digests_source", "zero_adapter_only", "reduction_mode", "lens_salt_policy", "require_attested_mode",
    "standup_front", "sampling_clamp_override", "sampling_clamp_override_mechanism",
    "output_diagnostics", "residual_slot", RESIDUAL_LEDGER_AFTER_CONTROLS_KEY, sa.SERVING_VENV_KEY})


def legacy_object_sha256(by_cell: Mapping[str, str]) -> Optional[str]:
    """THE LEGACY SCALAR ``object_artifact_sha256`` (capctrl's rule): the single
    distinct digest of ``by_cell`` when there is exactly one, else None."""
    distinct = sorted(set(by_cell.values()))
    return distinct[0] if len(distinct) == 1 else None


def attested_scheduler_geometry(attestation: Any) -> Optional[Tuple[int, int]]:
    """``(max_num_seqs, max_model_len)`` the WORKERS attested (every
    ``scheduler_geometry.worker_reports[*]`` entry), or None when no report
    carries the fields; disagreeing workers raise ``ValueError`` (mt's reader,
    one copy in the core)."""
    if not isinstance(attestation, dict):
        return None
    reports = ((attestation.get("scheduler_geometry") or {}).get("worker_reports") or [])
    seen = set()
    for rep in reports:
        if isinstance(rep, dict) and rep.get("max_num_seqs") is not None \
                and rep.get("max_model_len") is not None:
            seen.add((int(rep["max_num_seqs"]), int(rep["max_model_len"])))
    if not seen:
        return None
    if len(seen) != 1:
        raise ValueError(f"workers disagree on the scheduler geometry: {sorted(seen)}")
    return seen.pop()


# --------------------------------------------------------------------------- #
# the five positive controls                                                   #
# --------------------------------------------------------------------------- #


def _not_applicable(reason: str) -> Dict[str, Any]:
    """A control's RECORDED ``not_applicable`` verdict (the zero-
    adapter-only standup): not run, not failed, the reason stated. The
    certification's ``controls_summary`` still requires five bool PASSED — a
    zero-only standup is never certified (it always has a genuine adapter)."""
    return {"passed": None, "verdict": NOT_APPLICABLE, "reason": reason}


def output_diagnostics_policy(served_kinds: Iterable[str]) -> bool:
    """THE OUTPUT-DIAGNOSTICS POLICY (module docstring; native-psr-serving):
    ``False`` iff any served lora cell's object kind is a residual-slot kind
    (``defense_cells.RESIDUAL_SLOT_KINDS``: psr, vector, vector_all, diffmean,
    static_r) — a standup serving PSR never measures greedy-token agreement;
    ``True`` for a LoRA-only standup (today's behaviour, its records
    comparable to every certification before)."""
    return not any(str(k) in RESIDUAL_SLOT_KINDS for k in served_kinds)


def resolve_output_diagnostics(su: LoraStandup, requested: Optional[bool]) -> Tuple[bool, bool]:
    """The ``output_diagnostics`` value ONE controls run uses, held to the
    standup's RESOLVED policy (``su.output_diagnostics``, what
    :func:`bring_up_native` decided from the served kinds): returns
    ``(effective, inherited)``. ``None`` (the argument omitted — the certification
    driver's one-argument fallback, a repeated run through the public API)
    INHERITS the policy; an explicit ``False`` tightens; an explicit ``True`` on a
    standup whose policy is False REFUSES (:class:`StandupRefused`) — greedy-token
    agreement is never measured on a residual-bearing standup, and the rule is
    enforced HERE, not only at bring-up (the old ``True`` default silently
    re-enabled the diagnostics wherever the controls ran separately)."""
    policy = bool(su.output_diagnostics)
    if requested is None:
        return policy, True
    if bool(requested) and not policy:
        cells = sorted(c for c, f in ((su.residual_slot or {}).get("cells") or {}).items())
        raise StandupRefused(
            "run_positive_controls(output_diagnostics=True) on a standup whose resolved policy is "
            f"OFF (residual-slot cell(s) served {cells}): greedy-token agreement is never measured "
            "for a residual kind (native-psr-serving D19); a caller may only turn the diagnostics "
            "OFF, and the bring-up's refusal is enforced here too")
    return bool(requested), False


def run_positive_controls(su: LoraStandup, *, output_diagnostics: Optional[bool] = None
                          ) -> Dict[str, Any]:
    """The six positive controls (module docstring step 6; the sixth,
    ``boundary_refusal``, is the end-to-end check of the fail-closed ledger
    join, ``not_applicable`` without a restricted-mode adapter), in order, all
    through the LIVE proxy; written to ``positive_controls.json`` whatever the
    outcome. Raises :class:`PositiveControlsFailed` (the record attached) at
    the FIRST failure. On a zero-adapter-only standup ``steering_bites`` and
    ``selection_honored`` are recorded ``not_applicable`` (no lora cell; a
    single adapter's mixed-batch claim would be vacuous) and the other
    three run. ``output_diagnostics`` (THE OUTPUT-DIAGNOSTICS POLICY, module
    docstring): ``None`` INHERITS the standup's resolved policy
    (``su.output_diagnostics``), ``False`` tightens, ``True`` against an OFF
    policy refuses (:func:`resolve_output_diagnostics`) — it decides whether
    ``selection_honored`` computes its ``numerics`` block, and the record says
    which value was used and whether it was inherited. ``steering_bites`` takes
    the LOGPROB LIVENESS statistic for a restricted-mode / constant-kind cell
    (:data:`BITES_STATISTIC_LOGPROB`; :func:`logprob_liveness_cells`) and the
    text statistic for every other cell. The server is NOT torn down here (the
    bring-up does that)."""
    from sisteer.capeval.serving import fidelity
    from sisteer.capeval.serving import proxy as px
    from sisteer.capeval.serving import proxy_engine as pe
    from sisteer.capeval.serving import vllm_server as vs
    from sisteer.eval.attacks import bon_served as bs
    from sisteer.eval.attacks import served_target as stt

    output_diagnostics, inherited = resolve_output_diagnostics(su, output_diagnostics)
    prompts = probe_prompt_set()
    routes = su.routes
    by_cell = su.routes_by_cell
    genuine_vmodels = [su.vmodels[c] for c in su.lora_cells]
    record: Dict[str, Any] = {
        "engine": ENGINE, ENDPOINT_ATTEMPT_KEY: su.attempt_id, "started_at": _utc_now(),
        "probe_set": {"n": len(prompts), "hash": probe_set_hash(),
                      "max_tokens": int(su.probe_max_tokens), "sampling": "greedy"},
        "base_vmodel": su.base_vmodel, "zero_vmodel": su.zero_vmodel,
        "genuine_vmodels": list(genuine_vmodels),
        "zero_adapter_only": bool(su.zero_adapter_only),
        "output_diagnostics": bool(output_diagnostics),
        "output_diagnostics_inherited": bool(inherited),
        "output_diagnostics_policy": bool(su.output_diagnostics),
        "order": list(CONTROL_NAMES),
        "controls": {name: {"passed": None, "skipped": "an earlier control failed"}
                     for name in CONTROL_NAMES},
        "passed": False, "failed_control": None, "error": None,
    }

    def _fail(name: str, exc: BaseException, **partial: Any) -> None:
        record["controls"][name] = {"passed": False,
                                    "error": f"{type(exc).__name__}: {exc}", **partial}
        record["failed_control"] = name
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["finished_at"] = _utc_now()
        sa.write_json_atomic(su.positive_controls_path, record, default=str)
        raise PositiveControlsFailed(
            f"positive control {name!r} FAILED: {type(exc).__name__}: {exc}", record) from exc

    solo_cache: Dict[Tuple[str, str], List[int]] = {}

    def _greedy(vmodel: str, messages: Sequence[Mapping[str, str]]) -> List[int]:
        """A SOLO greedy decode, memoized per (selection, prompt) — never the
        mixed batch's path (its concurrency is the point of control (d))."""
        key = (vmodel, json.dumps([dict(m) for m in messages], sort_keys=True))
        if key not in solo_cache:
            solo_cache[key] = list(su.greedy_ids(vmodel, messages))
        return list(solo_cache[key])

    # (a) the zero-adapter NO-OP CERTIFICATE: none__lora0 == base, bit-exact, every prompt
    name = "noop_certificate"
    try:
        t0 = time.time()
        base_ids = [_greedy(su.base_vmodel, p) for p in prompts]
        zero_ids = [_greedy(su.zero_vmodel, p) for p in prompts]
        diag: Dict[str, Any] = {}
        fidelity._check_scale0_noop({"scale0_ids": zero_ids, "unsteered_ids": base_ids}, diag)
        record["controls"][name] = {
            "passed": True, "n_prompts": len(prompts), "all_exact": True,
            "n_tokens": [len(x) for x in base_ids], "wall_s": round(time.time() - t0, 2),
            "rule": "D1: the zero adapter is exact (choices[0].token_ids equal on every prompt)"}
    except Exception as e:  # noqa: BLE001 — every control's failure is a recorded verdict
        _fail(name, e)

    # (b) SALT ISOLATION over the FULL table (pure; the proxy's own salt derivation)
    name = "salt_isolation"
    try:
        bs.assert_cell_salt_isolation(routes)
        rep = bs.cell_salt_isolation_report(routes)
        record["controls"][name] = {
            "passed": True, "n_routes": len(routes),
            "per_cell": {vm: {"cell_id": e["cell_id"], "lens": e["lens"], "stable": e["stable"]}
                         for vm, e in rep["per_cell"].items()},
            "stable_collisions": rep["stable_collisions"]}
    except Exception as e:  # noqa: BLE001
        _fail(name, e)

    # (c) STEERING BITES per lora cell vs its path-matched zero twin: the TEXT statistic,
    #     or the LOGPROB LIVENESS statistic for a restricted-mode / constant-kind cell
    #     (BITES_STATISTIC_LOGPROB — a zero payload fails)
    name = "steering_bites"
    if su.zero_adapter_only:
        record["controls"][name] = _not_applicable(
            "no lora cell: a zero-adapter-only standup serves nothing to bite")
    else:
        per_cell: Dict[str, Any] = {}
        try:
            t0 = time.time()
            logprob_cells = logprob_liveness_cells(su)
            for cell in su.lora_cells:
                route = by_cell[cell]
                twin = pe.scale0_twin(route, routes)
                if twin is None:
                    raise LoraStandupError(
                        f"lora cell {cell!r} has no scale-0 twin route ({cell}__lora0 / "
                        f"{px.LORA_ZERO_CELL_SUFFIX}) — a steered serve without a bites probe is "
                        "refused, never run on faith")
                arm = cell.split("__")[-1]
                why = logprob_cells.get(cell)
                if why is None:
                    diag = bs.assert_steering_bites(
                        (arm,), stt.probe_fn(su.chat, route.virtual_model, twin.virtual_model))
                    d = diag["arms"].get(arm, {})
                    per_cell[cell] = {"statistic": BITES_STATISTIC_TEXT,
                                      "twin": twin.cell_id, "twin_vmodel": twin.virtual_model,
                                      "arm": arm, "n_probes": d.get("n"),
                                      "n_differing": d.get("n_differing"),
                                      "first_divergences": d.get("first_divergences")}
                    continue
                kind = (su.adapters.get(cell) or {}).get("petri_source_kind")
                block, err = logprob_liveness_block(su, route=route, twin=twin, kind=kind)
                per_cell[cell] = {"statistic": BITES_STATISTIC_LOGPROB, "why": why,
                                  "twin": twin.cell_id, "twin_vmodel": twin.virtual_model,
                                  "arm": arm, "kind": kind, "logprob_liveness": block,
                                  # the text statistic REPORTED beside the verdict, never gating
                                  "text_divergence": text_divergence_report(
                                      su, route=route, twin=twin, arm=arm)}
                if err is not None:
                    raise LoraStandupError(f"lora cell {cell!r} ({why}): {err}")
            record["controls"][name] = {"passed": True, "cells": per_cell,
                                        "wall_s": round(time.time() - t0, 2)}
        except Exception as e:  # noqa: BLE001 — the per-cell blocks so far stay on the record
            _fail(name, e, cells=per_cell)

    # (d) SELECTION HONORED — STRUCTURALLY attested: the engine's own per-request
    #     LoRA bookkeeping, read by the worker's routing recorder over the dev RPC route;
    #     every output comparison RECORDED under `numerics`, never gating
    name = "selection_honored"
    if su.zero_adapter_only:
        record["controls"][name] = _not_applicable(
            "single adapter: the mixed-batch claim would be vacuous (D88) — the zero adapter "
            "is the one registered adapter")
    else:
        try:
            t0 = time.time()

            def mixed_batch(batch: Sequence[Tuple[str, Sequence[Mapping[str, str]]]]
                            ) -> List[Tuple[List[int], str]]:
                with ThreadPoolExecutor(max_workers=max(1, len(batch))) as pool:
                    futures = [pool.submit(su.greedy_ids_with_request_id, vm, msgs)
                               for vm, msgs in batch]
                    return [f.result() for f in futures]

            selection_adapter: Dict[str, Optional[str]] = {su.base_vmodel: None,
                                                           su.zero_vmodel: su.zero_adapter["name"]}
            for cell in su.lora_cells:
                selection_adapter[su.vmodels[cell]] = su.adapters[cell]["name"]
            diag = fidelity.run_multi_lora_routing_control(
                base=su.base_vmodel, adapters=[*genuine_vmodels, su.zero_vmodel],
                zero_adapter=su.zero_vmodel, prompt_set=prompts, mixed_batch=mixed_batch,
                arm=lambda epoch: vs.arm_lora_routing_recorder(su.launch, epoch),
                observe=lambda epoch: vs.lora_routing_observations(su.launch, epoch),
                disarm=lambda epoch: vs.disarm_lora_routing_recorder(su.launch, epoch),
                selection_adapter=selection_adapter,
                attested_id_name_map=su.attestation["id_name_map"], solo=_greedy,
                output_diagnostics=bool(output_diagnostics))
            record["controls"][name] = {**diag, "wall_s": round(time.time() - t0, 2)}
        except Exception as e:  # noqa: BLE001
            _fail(name, e)

    # (e) the ADAPTER-DIGEST ATTESTATION, per name: table block == disk == attestation
    #     == the server's /v1/models root == the asserted id_name_map
    name = "adapter_digests"
    try:
        att = su.attestation
        models = su.upstream_models()
        cards = {m.get("id"): m for m in models.get("data", []) if isinstance(m, Mapping)}
        blocks = {**su.adapters, su.zero_adapter["name"]: su.zero_adapter}
        if set(blocks) != set(att["adapters"]):
            raise LoraStandupError(
                f"the table's adapter blocks {sorted(blocks)} and the attestation's adapters "
                f"{sorted(att['adapters'])} name different sets")
        checked: Dict[str, Any] = {}
        for aname, block in sorted(blocks.items()):
            px.verify_adapter_block_on_disk(block)
            entry = att["adapters"][aname]
            for key in ("config_sha256", "weights_sha256"):
                if block[key] != entry[key]:
                    raise LoraStandupError(
                        f"adapter {aname!r}: table {key} {block[key]} != attested {entry[key]}")
            if Path(entry["dir"]).resolve() != Path(block["dir"]).resolve():
                raise LoraStandupError(
                    f"adapter {aname!r}: attested dir {entry['dir']} != the block's {block['dir']}")
            card = cards.get(aname)
            if card is None or not isinstance(card.get("root"), str):
                raise LoraStandupError(
                    f"adapter {aname!r}: the server's /v1/models lists no card with a root "
                    f"(cards: {sorted(k for k in cards if k)})")
            if Path(card["root"]).resolve() != Path(block["dir"]).resolve():
                raise LoraStandupError(
                    f"adapter {aname!r}: /v1/models root {card['root']} != the block's dir "
                    f"{block['dir']}")
            if Path(entry["models_root"]).resolve() != Path(card["root"]).resolve():
                raise LoraStandupError(
                    f"adapter {aname!r}: the attested models_root {entry['models_root']} != the "
                    f"live /v1/models root {card['root']}")
            lid = int(entry["lora_int_id"])
            if att["id_name_map"].get(str(lid)) != aname:
                raise LoraStandupError(
                    f"adapter {aname!r}: id_name_map[{lid}] = "
                    f"{att['id_name_map'].get(str(lid))!r}, not this adapter")
            if su.engine_identity["adapters"][aname]["fingerprint_bf16"] != entry["fingerprint_bf16"]:
                raise LoraStandupError(f"adapter {aname!r}: engine identity fingerprint drift")
            checked[aname] = {"lora_int_id": lid, "config_sha256": block["config_sha256"],
                              "weights_sha256": block["weights_sha256"],
                              "fingerprint_bf16": entry["fingerprint_bf16"],
                              "dir": block["dir"], "models_root": card["root"],
                              "petri_zero_adapter": block["petri_zero_adapter"]}
        record["controls"][name] = {"passed": True, "adapters": checked,
                                    "id_name_map": dict(att["id_name_map"])}
    except Exception as e:  # noqa: BLE001
        _fail(name, e)

    # (f) BOUNDARY REFUSAL (the fail-closed ledger join, exercised END TO
    #     END on THIS engine): one deliberately header-less request per restricted-mode route must be
    #     REFUSED with the named code and leave the ledger clean; a 200, another code, or an unclaimed
    #     entry fails the control and the standup. Not applicable without a restricted-mode adapter.
    name = "boundary_refusal"
    restricted = restricted_boundary_cells(su)
    if not restricted:
        record["controls"][name] = {**_not_applicable("no restricted-mode (response-boundary) adapter is served: "
                                                      "an all-positions / LoRA-only standup has no boundary to fail"),
                                    "restricted_cells": []}
    else:
        try:
            per_cell = {cell: boundary_refusal_probe(su, cell=cell, position_mode=mode) for cell, mode in restricted}
            record["controls"][name] = {
                "passed": True, "restricted_cells": [c for c, _ in restricted], "cells": per_cell,
                "rule": ("a header-less render (one user turn, add_generation_prompt: false) on every restricted-mode "
                         f"route is refused {px.RESPONSE_BOUNDARY_UNLOCATED!r} (HTTP 500) and the ledger then reads "
                         "unclaimed 0 / ledger_fault false — the claim consumed the worker's entry")}
        except Exception as e:  # noqa: BLE001
            _fail(name, e, restricted_cells=[c for c, _ in restricted])

    record["passed"] = True
    record["finished_at"] = _utc_now()
    sa.write_json_atomic(su.positive_controls_path, record, default=str)
    return record


class BoundaryRefusalControlFailed(LoraStandupError):
    """The boundary-refusal control's own failure (O49): the header-less
    completion was RETURNED, refused under another code, or left an
    unclaimed / faulted ledger — the fail-closed join is not working."""


def restricted_boundary_cells(su: LoraStandup) -> List[Tuple[str, str]]:
    """``[(harness cell, position mode)]`` of the served residual cells whose
    mode locates a RESPONSE BOUNDARY (``native_residual.BOUNDARY_MODES``) — the
    routes the boundary-refusal control drives; ``prompt_last`` / ``prefill_all``
    need no boundary and are never a failure to render header-less."""
    from sisteer.capeval.serving.native_residual import BOUNDARY_MODES
    cells = ((su.residual_slot or {}).get("cells") or {}) if isinstance(su.residual_slot, Mapping) else {}
    return sorted((str(c), str(e.get("position_mode"))) for c, e in cells.items()
                  if isinstance(e, Mapping) and e.get("position_mode") in BOUNDARY_MODES and c in su.vmodels)


def boundary_refusal_probe(su: LoraStandup, *, cell: str, position_mode: str,
                           ledger_timeout_s: float = 60.0) -> Dict[str, Any]:
    """ONE header-less request on ``cell``'s route through the live proxy (the
    path every driver uses), judged fail-closed: the proxy must refuse it with
    :data:`proxy.RESPONSE_BOUNDARY_UNLOCATED` (HTTP 500), and the worker ledger
    read right after must be EMPTY and unfaulted — the entry the worker wrote
    under its ENGINE id was claimed by the completion id. Anything else raises
    :class:`BoundaryRefusalControlFailed`, named."""
    from sisteer.capeval.serving import proxy as px
    from sisteer.capeval.serving import vllm_server as vs
    vmodel = su.vmodels[cell]
    nonce = getattr(getattr(su, "proxy", None), "boundary_control_nonce", None)
    if not isinstance(nonce, str) or not nonce:
        raise BoundaryRefusalControlFailed(
            f"boundary_refusal on {vmodel!r}: this standup's proxy minted no control nonce — the probe is the "
            "standup's own and cannot be presented without it")
    body = {"model": vmodel, "messages": [dict(m) for m in sa.BOUNDARY_REFUSAL_CONTROL_MESSAGES],
            **sa.BOUNDARY_REFUSAL_CONTROL_KWARGS, px.BOUNDARY_REFUSAL_CONTROL_KEY: nonce}
    where = f"boundary_refusal on {vmodel!r} (cell {cell!r}, position mode {position_mode!r}): "
    try:
        resp = su.chat(body)
    except ProxyHTTPRefusal as e:
        status, code, message, worker_error = e.status, e.code, e.body_text, e.worker_error_type
    except LoraStandupError as e:
        raise BoundaryRefusalControlFailed(where + f"the request failed without a named refusal code: {e}") from e
    else:
        raise BoundaryRefusalControlFailed(
            where + f"the proxy RETURNED a completion (id {resp.get('id')!r}) for a header-less render — the worker "
            "did not ledger the unlocated boundary, or the proxy's claim did not find the entry (the engine-id join): "
            "the fail-closed rule is NOT working on this engine; refusing the standup")
    if code != px.RESPONSE_BOUNDARY_UNLOCATED or status != 500:
        raise BoundaryRefusalControlFailed(
            where + f"refused with code {code!r} (HTTP {status}), not {px.RESPONSE_BOUNDARY_UNLOCATED!r} (HTTP 500): "
            f"{message[:400]}")
    if worker_error != ERROR_BOUNDARY_UNLOCATED_TYPE:
        # the refusal must be FOR the unlocated boundary: an unrelated worker failure ledgered on the probe
        # (RequestFactsMissing, PromptLenDisagreement) is claimed by the same code and would empty the ledger —
        # a pass for the wrong reason
        raise BoundaryRefusalControlFailed(
            where + f"refused for worker error_type {worker_error!r}, not {ERROR_BOUNDARY_UNLOCATED_TYPE!r} — an "
            "unrelated worker failure on the probe is not the fail-closed boundary rule at work")
    reports = vs.residual_slot_reports(su.launch, timeout_s=ledger_timeout_s)
    sizes = [r.get("ledger_size") for r in reports]
    faults = [r.get("ledger_fault") for r in reports]
    if not all(isinstance(s, int) and not isinstance(s, bool) for s in sizes) or max(sizes) != 0:
        raise BoundaryRefusalControlFailed(
            where + f"after the refusal the worker ledger still holds unclaimed entries (ledger_size per worker "
            f"{sizes}) — the refusal did not consume the entry it named")
    if not all(isinstance(f, bool) for f in faults) or any(faults):
        raise BoundaryRefusalControlFailed(where + f"after the refusal the ledger reports ledger_fault {faults}")
    return {"vmodel": vmodel, "position_mode": position_mode, "http_status": status, "code": code,
            "worker_error_type": worker_error, "unclaimed_after": 0, "ledger_fault_after": False,
            "n_workers": len(reports), "refusal": message[:300]}


#: the worker's error_type the boundary-refusal control requires (native_residual.ERROR_BOUNDARY_UNLOCATED)
ERROR_BOUNDARY_UNLOCATED_TYPE = "ResponseBoundaryUnlocated"


def logprob_liveness_cells(su: LoraStandup) -> Dict[str, str]:
    """The served lora cells whose ``steering_bites`` statistic is the LOGPROB
    LIVENESS (:data:`BITES_STATISTIC_LOGPROB`), each with WHY: a constant-kind
    object (``contracts.LOGPROB_LIVENESS_KINDS`` — alphasteer, static_r: a fixed
    direction whose greedy text may equal base at the served dose) or a residual
    cell at a RESTRICTED native position mode (a response-fit object writes
    nothing before the header, so greedy text may equal base). Every other cell
    keeps the text statistic. Read off the standup's own records (the adapter
    block's ``petri_source_kind``; ``residual_slot.cells[cell].position_mode``)."""
    from sisteer.capeval.tinycap import contracts as tc

    out: Dict[str, str] = {}
    slot_cells = (su.residual_slot or {}).get("cells") or {}
    for cell in su.lora_cells:
        kind = (su.adapters.get(cell) or {}).get("petri_source_kind")
        mode = (slot_cells.get(cell) or {}).get("position_mode")
        if kind in tc.LOGPROB_LIVENESS_KINDS:
            out[cell] = f"constant-kind object ({kind}): greedy text may equal base at the served dose"
        elif mode is not None and mode != NATIVE_POSITION_MODE_ALL:
            out[cell] = (f"restricted native position mode {mode!r}: the write covers a subset of "
                         "rows, so greedy text may equal base")
    return out


def logprob_liveness_block(su: LoraStandup, *, route: Any, twin: Any, kind: Optional[str],
                           top_logprobs: int = LOGPROB_LIVENESS_TOP_LOGPROBS,
                           min_ratio: float = LOGPROB_BITES_MIN_RATIO,
                           abs_floor: float = LOGPROB_BITES_ABS_FLOOR,
                           min_common_tokens: int = LOGPROB_LIVENESS_MIN_COMMON_TOKENS,
                           min_conclusive_probes: int = LOGPROB_LIVENESS_MIN_CONCLUSIVE_PROBES
                           ) -> Tuple[Dict[str, Any], Optional[str]]:
    """THE LOGPROB LIVENESS statistic through the LIVE proxy (the driver's
    ``_logprob_liveness_evidence``, on the standup's probe set): for every probe,
    ONE greedy token with ``logprobs`` + ``top_logprobs`` +
    ``return_tokens_as_token_ids`` at the first generated position from the
    steered vmodel, the scale-0 twin, and the twin AGAIN (``contracts.BITES_LEGS``,
    sequential per probe so the two twin draws never share a batch); the three
    legs' top-K read on their COMMON token ids; the verdict
    ``fidelity.check_logprob_steering_bites``'s over the conclusive probes — the
    steered divergence must clear ``max(abs_floor, min_ratio x the measured
    repeat-noise ceiling)``, and BIT-IDENTICAL steered / twin logprobs on every
    probe refuse (the ZERO-payload known-bad). Every request carries only
    :data:`proxy.RESTRICTED_ROUTE_ALLOWED_KEYS` members, so the statistic reads
    on a restricted-mode route. ``top_logprobs`` is capped at the standup's
    ``max_logprobs`` when one was pinned. Returns ``(block, error_or_None)``: the
    block is recorded either way (a failed control is durable AND loud) in the
    shape ``contracts.logprob_liveness_conclusive_problems`` reads."""
    from sisteer.capeval.serving import proxy_engine as pe
    from sisteer.capeval.serving.fidelity import FidelityError, check_logprob_steering_bites
    from sisteer.capeval.tinycap import contracts as tc

    top = int(top_logprobs)
    if su.max_logprobs is not None:
        top = max(1, min(top, int(su.max_logprobs)))
    prompts = probe_prompt_set()
    legs = tuple(zip(tc.BITES_LEGS, (route.virtual_model, twin.virtual_model, twin.virtual_model)))

    def one_token(vmodel: str, messages: Sequence[Mapping[str, str]], leg: str, index: int
                  ) -> Dict[int, float]:
        resp = su.chat({"model": vmodel, "messages": [dict(m) for m in messages],
                        "temperature": 0.0, "max_tokens": int(tc.LOGPROB_BITES_MAX_TOKENS),
                        "seed": 0, "logprobs": True, "top_logprobs": top,
                        "return_tokens_as_token_ids": True})
        return pe.first_position_logprobs_by_id(resp, what=f"logprob_liveness probe {index} leg {leg!r}")

    per_probe: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []
    error: Optional[str] = None
    try:
        for i, messages in enumerate(prompts):
            by_leg = {leg: one_token(vm, messages, leg, i) for leg, vm in legs}
            supports = [set(d) for d in by_leg.values()]
            common = sorted(set.intersection(*supports))
            rec = {"item_id": f"probe{i:02d}", "prompt_index": i, "n_common": len(common),
                   "support_sizes": {leg: len(by_leg[leg]) for leg, _ in legs},
                   "supports_disjoint": not (set(by_leg[tc.BITES_LEGS[0]]) & set(by_leg[tc.BITES_LEGS[1]])),
                   "conclusive": len(common) >= int(min_common_tokens)}
            per_probe.append(rec)
            if rec["conclusive"]:
                items.append({"item_id": rec["item_id"], "common_token_ids": common,
                              **{leg: [by_leg[leg][t] for t in common] for leg, _ in legs}})
        if len(items) < int(min_conclusive_probes):
            error = (f"logprob_liveness: only {len(items)} of {len(per_probe)} probe(s) are conclusive (>= "
                     f"{min_common_tokens} common top-{top} token ids across the three legs); the control "
                     f"needs >= {min_conclusive_probes} — UNEVALUABLE, refused (never a text-only pass); per "
                     f"probe: {[(r['item_id'], r['n_common'], r['supports_disjoint']) for r in per_probe]}")
    except Exception as e:  # noqa: BLE001 — an unreadable instrument is a FAILED control, never a pass
        error = f"logprob_liveness: the statistic could not be read — {type(e).__name__}: {e}"
    diag: Dict[str, Any] = {}
    if error is None:
        try:
            check_logprob_steering_bites({"items": items}, diag, min_ratio=min_ratio, abs_floor=abs_floor)
        except FidelityError as e:
            error = str(e)
    noise, bar, median = diag.get("noise_ceiling"), diag.get("bar"), diag.get("steered_divergence_median")
    block: Dict[str, Any] = {
        "applicable": True, "verified": error is None, "kind": kind, "twin": twin.cell_id,
        "statistic": ("sisteer.capeval.serving.fidelity.check_logprob_steering_bites over the COMMON top-K "
                      "token ids of the three legs at the first generated position, on the standup's probe "
                      "set (the driver's liveness instrument, run by the standup)"),
        "legs": list(tc.BITES_LEGS), "leg_order": "sequential per probe: steered, scale0, scale0_repeat",
        "n_probes": len(prompts), "n_conclusive": len(items),
        "min_conclusive_probes": int(min_conclusive_probes), "min_common_tokens": int(min_common_tokens),
        "top_logprobs": top, "max_tokens": int(tc.LOGPROB_BITES_MAX_TOKENS),
        "per_probe": per_probe, "probe_items": items, "diag": diag,
        "noise_floor": {
            "measured": noise,
            "how": ("the scale-0 twin against ITSELF: the same request to the same twin route a second time, "
                    "in this run, on these probes — max over probes of the max abs logprob difference"),
            "min_ratio": float(min_ratio), "abs_floor": float(abs_floor), "bar": bar,
            "binding": (None if bar is None or noise is None else
                        ("measured" if float(min_ratio) * float(noise) >= float(abs_floor) else "abs_floor")),
        },
        "divergence": {"steered_vs_scale0_median": median,
                       "clears_bar": (None if bar is None or median is None else median >= bar)},
        "not_a_dose_axis": tc.LOGPROB_LIVENESS_NOT_A_DOSE_AXIS,
    }
    if error is not None:
        block["error"] = error
    return block, error


def text_divergence_report(su: LoraStandup, *, route: Any, twin: Any, arm: str) -> Dict[str, Any]:
    """The TEXT statistic (``bon_served.assert_steering_bites`` through the live
    proxy) REPORTED beside a logprob-liveness verdict, never gating: a cell live
    by logprob and inert by text is a fact about dose the record keeps."""
    from sisteer.eval.attacks import bon_served as bs
    from sisteer.eval.attacks import served_target as stt

    try:
        diag = bs.assert_steering_bites(
            (arm,), stt.probe_fn(su.chat, route.virtual_model, twin.virtual_model))
        d = diag["arms"].get(arm, {})
        return {"gating": False, "passed": True, "n_probes": d.get("n"),
                "n_differing": d.get("n_differing"), "first_divergences": d.get("first_divergences")}
    except Exception as e:  # noqa: BLE001 — reported, never the verdict
        return {"gating": False, "passed": False, "error": f"{type(e).__name__}: {e}"}


def ledger_problems(reports: Sequence[Mapping[str, Any]]) -> List[str]:
    """Why the worker reports' ledger state is NOT clean:
    ``ledger_fault`` other than False (the latch, or an unreadable value), or a
    ``ledger_size`` other than 0 — an UNCLAIMED response-boundary failure after
    every probe was claimed means the proxy post-check is not running for a
    route the worker ledgers. ``[]`` means clean. A report without the two keys
    is a problem (an older worker cannot attest an empty ledger)."""
    problems: List[str] = []
    if not reports:
        return ["no worker report to read the ledger from"]
    for i, rep in enumerate(reports):
        if not isinstance(rep, Mapping):
            problems.append(f"worker {i}: non-report {rep!r}")
            continue
        fault, size = rep.get("ledger_fault", "<absent>"), rep.get("ledger_size", "<absent>")
        if fault is not False:
            problems.append(f"worker {i}: residual_slot.ledger_fault = {fault!r} after the controls")
        if size != 0:
            problems.append(f"worker {i}: residual_slot.ledger_size = {size!r} unclaimed request "
                            "failure(s) after the controls — the proxy post-check is not claiming "
                            "them (is a restricted-mode route being served without the claim?)")
        # a BOOKKEEPING fault (a runner-state disagreement, missing facts, an unbound read) recorded
        # during the controls is never clean either — readiness refused any at start-up; the controls'
        # own traffic must not have raised one
        faults = rep.get("n_faults", 0)
        if not isinstance(faults, int) or isinstance(faults, bool) or faults != 0:
            problems.append(f"worker {i}: residual_slot.n_faults = {faults!r} after the controls — a "
                            "bookkeeping fault during the controls' own traffic")
    return problems


def unclaimed_failures_problems(su: LoraStandup, *, timeout_s: float = 60.0) -> List[str]:
    """ONE live read of the worker's ledger state (``sisteer_residual_slot_report``
    through ``vllm_server.residual_slot_reports``) judged by :func:`ledger_problems`
    — the helper the certification driver / the §10 gate call AFTER their probes
    (which the proxy claims one by one); the standup calls it after its own
    positive controls (:func:`residual_ledger_after_controls`). On an UNARMED
    (LoRA-only) standup the worker reports an empty ledger by construction."""
    from sisteer.capeval.serving import vllm_server as vs

    try:
        reports = vs.residual_slot_reports(su.launch, timeout_s=timeout_s)
    except vs.VllmServerError as e:
        return [f"the ledger could not be read: {e}"]
    return ledger_problems(reports)


def residual_ledger_after_controls(su: LoraStandup, *, timeout_s: float = 60.0) -> Dict[str, Any]:
    """The POST-CONTROLS ledger record (:data:`RESIDUAL_LEDGER_AFTER_CONTROLS_KEY`):
    on a residual-armed standup, the live read + verdict (``checked: True``,
    ``unclaimed`` = the max ``ledger_size`` over workers, ``ledger_fault`` = any
    worker's, ``problems``); on an unarmed standup ``checked: False`` with the
    reason (nothing writes a ledger there; the LoRA-only records stay as they
    were). The bring-up REFUSES on any problem."""
    from sisteer.capeval.serving import vllm_server as vs

    if not bool((su.residual_slot or {}).get("armed")):
        return {"checked": False, "reason": "residual slot unarmed: no worker ledger to read",
                "problems": []}
    try:
        reports = vs.residual_slot_reports(su.launch, timeout_s=timeout_s)
    except vs.VllmServerError as e:
        return {"checked": True, "n_workers": 0, "unclaimed": None, "ledger_fault": None,
                "problems": [f"the ledger could not be read: {e}"], "read_at": _utc_now()}
    sizes = [s for s in (r.get("ledger_size") for r in reports)
             if isinstance(s, int) and not isinstance(s, bool)]
    faults = [r.get("ledger_fault") for r in reports]
    return {"checked": True, "n_workers": len(reports),
            "unclaimed": max(sizes) if sizes else None,
            "ledger_fault": (any(f is True for f in faults) if all(isinstance(f, bool) for f in faults)
                             else None),
            "problems": ledger_problems(reports), "read_at": _utc_now()}


def controls_verdict_line(record: Mapping[str, Any]) -> str:
    """``k/N passed[, n not_applicable]`` for the log line (N = the control roster)."""
    controls = record.get("controls") or {}
    n_pass = sum(1 for c in controls.values() if isinstance(c, Mapping) and c.get("passed") is True)
    n_na = sum(1 for c in controls.values()
               if isinstance(c, Mapping) and c.get("verdict") == NOT_APPLICABLE)
    line = f"{n_pass}/{len(CONTROL_NAMES)} passed"
    if n_na:
        line += f", {n_na} not_applicable ({', '.join(n for n, c in controls.items() if isinstance(c, Mapping) and c.get('verdict') == NOT_APPLICABLE)})"
    return line


# --------------------------------------------------------------------------- #
# geometry: the kernel ceiling and the KV-derived shrink as §1.8 adjustments   #
# --------------------------------------------------------------------------- #


def _resolve_geometry(eval_name: Optional[str], max_model_len, max_num_seqs, *, model_key: Optional[str],
                      allow_over_budget: bool, override_reason: Optional[str]):
    """The wave front's resolver (byte-identical to the pre-core rule): a
    reason relaxes the identity guard, never the budget guard; ``resolve_standup_
    geometry`` (capctrl's) otherwise."""
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
            return sg.resolve_geometry(eval_name, max_model_len=max_model_len,
                                       max_num_seqs=max_num_seqs,
                                       allow_over_budget=bool(allow_over_budget),
                                       strict_identity=False, model_key=model_key)
        return _resolve_standup_geometry(eval_name, max_model_len, max_num_seqs,
                                         allow_over_budget=allow_over_budget, model_key=model_key)
    except (sg.GeometryOverBudget, sg.GeometryIdentityBroken) as e:
        raise StandupRefused(f"refusing the geometry: {e}") from e
    except sg.UnknownModelKey as e:
        raise StandupRefused(f"refusing --model-key: {e}") from e
    except ValueError as e:
        raise StandupRefused(f"refusing the geometry: {e}") from e


#: capctrl's pre-table defaults (kept when neither --eval nor the flags are given)
LEGACY_MAX_MODEL_LEN = 10240
LEGACY_MAX_NUM_SEQS = 64


def _resolve_standup_geometry(eval_name, max_model_len, max_num_seqs, *,
                              allow_over_budget=False, model_key=None,
                              requested_max_tokens=None):
    """``capctrl_serving_up.resolve_standup_geometry``, one copy in the core:
    explicit flags win; an unset flag comes from the eval's table entry; with no
    eval, the legacy default (over budget accepted, logged LEGACY)."""
    from sisteer.capeval.serving import geometry as sg
    if eval_name is None:
        return sg.resolve_geometry(
            None, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
            fallback=(LEGACY_MAX_MODEL_LEN, LEGACY_MAX_NUM_SEQS),
            allow_over_budget=True, model_key=model_key)
    return sg.resolve_geometry(
        eval_name, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
        allow_over_budget=bool(allow_over_budget), strict_identity=True,
        model_key=model_key, requested_max_tokens=requested_max_tokens)


#: the ONE opt-out of the live probe-record predicate, by name: a caller whose standup IS the certification (a
#: probe's own launch) passes this sentinel as ``parity_record_check``; ``None`` means the DEFAULT predicate, never a skip
PARITY_GATE_OPT_OUT = "probe_own_launch"
#: the front door's record predicate: scripts/sdfbase/hpo_probe_status.py (stdlib-only, piped over ssh by the front door; imported
#: here by path, never re-derived) — relative to the checkout this core runs from
HPO_PROBE_STATUS_PATH = Path(__file__).resolve().parents[4] / "scripts" / "sdfbase" / "hpo_probe_status.py"
_HPS_MODULE: Any = None
#: the unit's SUBMISSION time (ISO-8601 Z), when the job carries it: a staging stamp older than this is refused on a cluster that
#: cannot reach the record — the front door's submit path exports it; unset = no bound (the stamp's other fields
#: and the REVOKED marker still gate)
UNIT_SUBMITTED_UTC_ENV = "SISTEER_UNIT_SUBMITTED_UTC"


def default_parity_record_check(record_dir: Path) -> Tuple[str, str, Dict[str, Any]]:
    """The live predicate every native front applies to a REACHABLE probe record: the front door's own
    ``hpo_probe_status.probe_state_record`` — READY iff the record is complete, its menu + verdict are of its current generation
    AND the parity-pass record they read is current — loaded from :data:`HPO_PROBE_STATUS_PATH`. A tree without that file
    REFUSES (the live record cannot be read; a skip would serve under withdrawn certification). ``job_active`` is False: a
    queued/running probe reads INCOMPLETE_STALLED, which refuses like INCOMPLETE_RUNNING."""
    global _HPS_MODULE
    if _HPS_MODULE is None:
        import importlib.util
        if not HPO_PROBE_STATUS_PATH.is_file():
            raise StandupRefused(f"the front door's record predicate is not on this tree ({HPO_PROBE_STATUS_PATH}) — the live probe "
                                 "record behind the staged overlay cannot be read; refusing (never a skip)")
        spec = importlib.util.spec_from_file_location("_native_standup_hpo_probe_status", HPO_PROBE_STATUS_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # type: ignore[union-attr]
        _HPS_MODULE = mod
    return _HPS_MODULE.probe_state_record(Path(record_dir), job_active=False)


def genuine_adapter_pairs(adapter_blocks: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{cell: {rank, module_set, zero}}`` for every banked adapter block — the (rank, set) each SERVES, read off its own files
    (``adapter_config.json`` ``r``; the set id of the module class its factor keys imply, ``lora_export.adapter_target_class`` minus
    the ``experts`` wrapper word — ``lora_module_sets.set_id_for``): what the runtime parity gate holds the staged overlay to."""
    from ...extraction.lora_module_sets import LoRAModuleError, set_id_for
    from . import lora_export as lx
    from .worker_reports import FingerprintError, read_adapter_config

    out: Dict[str, Dict[str, Any]] = {}
    for name, block in adapter_blocks.items():
        d = Path(block["dir"])
        try:
            cfg = read_adapter_config(d)
            # the set the adapter SERVES: the module class its FILE carries (what vLLM will wrap), and PEFT's declared
            # `target_modules` (the HF short names; a dotted spelling is read by its last component) — both when both are
            # readable, and they must name the same set; the config alone for a file whose header is not a factor set
            declared = [str(m).rsplit(".", 1)[-1] for m in (cfg.get("target_modules") or [])]
            try:
                cls: Optional[Tuple[str, ...]] = lx.adapter_target_class(d)
            except lx.LoraExportError:
                cls = None
            if cls is None and not declared:
                raise StandupRefused(f"cannot read the (rank, module set) adapter {name!r} serves from {d}: the factor header is not "
                                     "readable and adapter_config.json names no target_modules")
            file_set = set_id_for(lx.modules_of_class(cls)) if cls is not None else None
            cfg_set = set_id_for(declared) if declared else None
            if file_set is not None and cfg_set is not None and file_set != cfg_set:
                raise StandupRefused(f"adapter {name!r} at {d}: its factors adapt module set {file_set!r} but its adapter_config.json "
                                     f"declares {cfg_set!r} (target_modules {declared}) — the two disagree; refusing")
            module_set = file_set if file_set is not None else cfg_set
        except (FingerprintError, LoRAModuleError) as e:
            raise StandupRefused(f"cannot read the (rank, module set) adapter {name!r} serves from {d}: {e}") from e
        # a RESIDUAL-SLOT CARRIER (lora_export._export_residual_slot: a rank-1 ZERO o_proj pair that only carries the payload the
        # residual op applies — psr / vector / vector_all / diffmean / static_r / psr_paired_write) is NOT a LoRA the parity record
        # could have certified: its certificate is the residual op's own (the noop certificate + positive controls after launch)
        kind = block.get("petri_source_kind")
        out[name] = {"rank": int(cfg["r"]), "module_set": module_set, "target_class": (list(cls) if cls is not None else None),
                     "zero": bool(block.get("zero")), "kind": kind, "residual": bool(kind in RESIDUAL_SLOT_KINDS and not block.get("zero"))}
    return out


def lora_kernel_ceiling_geometry(adapter_blocks: Mapping[str, Mapping[str, Any]], geom: Any, *,
                                 eval_name: Optional[str], model_key: Optional[str],
                                 allow_over_budget: bool, override_reason: Optional[str]
                                 ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """The LoRA kernel index ceiling of the banked adapters and, when ``geom``'s
    ``S x L`` exceeds it, the LOWERING ``(ceiling_record, lowered | None)``;
    ``lowered`` carries ``_geom`` (the re-resolved geometry at ``ceiling // L``
    under ``sa.LORA_KERNEL_CEILING_REASON``) beside the record fields — byte-
    identical to the pre-core records. A ceiling below one row at this L
    refuses. A legacy (no-eval) geometry — capctrl's fallback — is re-stated
    at the lowered S in place (there is no table row to re-resolve against)."""
    dims = {}
    for name, block in adapter_blocks.items():
        try:
            dims[name] = sa.adapter_module_dims(Path(block["dir"]))
        except sa.ServingAttestationError as e:
            raise StandupRefused(f"cannot size the LoRA kernel index ceiling: adapter {name!r}: {e}") from e
    max_dim, widest = sa.max_module_dim(dims)
    ceiling = sa.lora_kernel_index_ceiling(max_dim)
    L, S = int(geom.max_model_len), int(geom.max_num_seqs)
    record = {"D": max_dim, "widest_module": widest, "ceiling": ceiling, "L": L, "S": S,
              "product": S * L, "limit": sa.LORA_KERNEL_INDEX_LIMIT}
    if S * L <= ceiling:
        return record, None
    lowered_S = ceiling // L
    if lowered_S < 1:
        raise StandupRefused(
            f"max_model_len={L} alone exceeds the LoRA kernel index ceiling {ceiling:,} tokens "
            f"(D={max_dim} at {widest}): no max_num_seqs can serve this window natively")
    if eval_name is not None:
        # THE WINDOW'S SOURCE SURVIVES THE LOWERING (2026-09-17): only S is being lowered, so the
        # re-resolved record must not claim the WINDOW was overridden too. `geometry.resolve_geometry`
        # stamps `window_source: override` for any non-None `max_model_len`, so passing L back
        # explicitly stamped `override` on a window that IS the table's, and the capability readers'
        # shared leg check (`tinycap_math500_thinking.geometry_contract_problem`) refuses
        # `window_source != table` BEFORE it validates the adjustment — every ceiling-lowered native
        # leg read as unusable (measured on the arditi_allpos GSM8K leg 544615: L 10752 == the table
        # resolution, S 69 -> 48). So a TABLE-sourced window is re-resolved from the table (None), which
        # yields that same window with its true source; any other source (an explicit override — the
        # multi-turn cap-4096 windows, `MULTITURN_CAP4096_WINDOWS`, passed precisely because the table's
        # was measured under a smaller cap — a fallback, a measurement) is passed back explicitly, where
        # `override` is the honest stamp and re-resolving from the table would SILENTLY CHANGE the served
        # window. Either way the resolved window must equal the one in force: a table that moved between
        # the two resolutions refuses rather than serving a geometry nobody asked for.
        from_table = getattr(geom, "window_source", None) == "table"
        new_geom = _resolve_geometry(eval_name, None if from_table else L, lowered_S,
                                     model_key=model_key,
                                     allow_over_budget=allow_over_budget,
                                     override_reason=sa.LORA_KERNEL_CEILING_REASON)
        if int(new_geom.max_model_len) != L:
            raise StandupRefused(
                f"the LoRA-kernel-ceiling re-resolution changed the WINDOW: {L} -> "
                f"{int(new_geom.max_model_len)} (window_source {getattr(geom, 'window_source', None)!r} -> "
                f"{new_geom.window_source!r}). Only max_num_seqs may be lowered here; the table entry for "
                f"{eval_name!r} must have moved between the two resolutions. Refusing rather than serving "
                "a window this standup never resolved")
    else:
        # the legacy no-eval geometry (capctrl's pre-table fallback): no table row to
        # re-resolve against; the lowered S is stated in place, its source `override`
        new_geom = dataclasses.replace(geom, max_num_seqs=lowered_S, seqs_source="override")
    record.update(S=lowered_S, product=lowered_S * L)
    lowered = {"reason": sa.LORA_KERNEL_CEILING_REASON, "requested_S": S, "S": lowered_S, "L": L,
               "D": max_dim, "widest_module": widest, "ceiling": ceiling,
               "product_requested": S * L, "product": lowered_S * L,
               "requested_override_reason": override_reason, "_geom": new_geom}
    return record, lowered


def launch_class_geometry(geom, *, served_lora: Optional[Tuple[int, int]], eval_name: Optional[str], model_key: Optional[str],
                          allow_over_budget: bool = False):
    """``(geom', launch_class adjustment | None)`` — the site record's per-window S re-derived over the launches whose adapter class
    dominates ``served_lora`` (geometry.least_admitted_by_window). Only a budget-DERIVED S (``seqs_source == "model_budget"``) is
    re-sized: an explicit, table or fallback S is the caller's. The window is passed back explicitly (the same L; a resolution that moved
    it refuses), the identity guard stays strict (the S is derived, not stated). A record without launch classes resolves to the same S
    and yields None (the caller's S < 2 refusal then speaks); a record where nothing dominates the served class refuses by name."""
    from sisteer.capeval.serving import geometry as sg
    if model_key is None or served_lora is None or getattr(geom, "seqs_source", None) != "model_budget":
        return geom, None
    try:
        new = sg.resolve_geometry(eval_name, max_model_len=int(geom.max_model_len), max_num_seqs=None,
                                  allow_over_budget=bool(allow_over_budget), strict_identity=True, model_key=model_key, served_lora=served_lora)
    except sg.SiteGeometryError as e:
        raise StandupRefused(f"launch-class geometry (rank cap {served_lora[0]}, {served_lora[1]} adapters): {e}") from e
    except (sg.GeometryOverBudget, sg.GeometryIdentityBroken, sg.GeometryCapExceedsBasis, sg.UnknownModelKey, ValueError) as e:
        raise StandupRefused(f"launch-class geometry: {e}") from e
    if int(new.max_model_len) != int(geom.max_model_len):
        raise StandupRefused(f"the launch-class re-resolution changed the WINDOW: {geom.max_model_len} -> {new.max_model_len}; refusing")
    entry = getattr(getattr(new, "budget", None), "entry", None)
    src = getattr(entry, "window_seqs_source", None)
    if int(new.max_num_seqs) == int(geom.max_num_seqs):
        return geom, None
    chosen = ((getattr(entry, "window_seqs_launches", None) or {}).get(int(geom.max_model_len)) or ())
    rec = sg.launch_class_adjustment(from_S=int(geom.max_num_seqs), to_S=int(new.max_num_seqs), served_lora=served_lora,
                                     window=int(geom.max_model_len), window_seqs_source=src, launches=list(chosen))
    return dataclasses.replace(geom, max_num_seqs=int(new.max_num_seqs), seqs_source="override"), rec


def kernel_adjustment_record(lowered: Mapping[str, Any]) -> Dict[str, Any]:
    """The §1.8 ``adjustments`` step for a kernel lowering — EXACTLY the pure shape
    ``geometry.kernel_ceiling_adjustment`` (W-A; ``KERNEL_ADJUSTMENT_FIELDS``)
    re-derives: ``{kind, from_S, to_S, D, ceiling}`` — from the standup's OWN
    numbers (the banked adapters' widest module), so the validator's re-derivation
    from the model's pinned width must agree with what was actually served. The
    widest module and L stay in the ``geometry_lowered`` record beside it."""
    return {"kind": ADJUSTMENT_KERNEL, "from_S": int(lowered["requested_S"]), "to_S": int(lowered["S"]),
            "D": int(lowered["D"]), "ceiling": int(lowered["ceiling"])}


def kv_adjustment_record(shrink: Mapping[str, Any]) -> Dict[str, Any]:
    """The §1.8 ``adjustments`` step for the KV-derived shrink: ``kind``,
    ``from_S`` (the PREVIOUS step's S — the kernel-lowered one when a kernel
    step precedes it), ``to_S``, and every field of ``geometry.kv_shrink_record``
    inline so ``kv_shrink_problem`` re-derives it —
    ``geometry.kv_shrink_adjustment``, one spelling."""
    from sisteer.capeval.serving.geometry import kv_shrink_adjustment
    return dict(kv_shrink_adjustment(shrink))


def kv_derived_geometry(geom: Any, short: Any, out: Path) -> Tuple[Any, Dict[str, Any]]:
    """The re-launch geometry after a ``vllm_server.KvCoverageShortfall``
    under ``kv_derived_seqs`` — ``geom`` re-stated at ``S = floor(kv_backed_
    concurrency)`` (``geometry.kv_shrunk_geometry``) with the shrink record;
    the first launch's engine log moved aside to :data:`KV_SHORTFALL_LOG`.
    Refuses (a ``VllmServerError``) a shortfall that shrinks to nothing or does
    not shrink at all. capctrl's function, one copy in the core."""
    from sisteer.capeval.serving import geometry as sg
    from sisteer.capeval.serving import vllm_server as vs
    if not dataclasses.is_dataclass(geom):
        raise TypeError(f"a KV-derived geometry needs a ResolvedGeometry, not {type(geom).__name__}")
    first = Path(out) / SERVER_LOG_FILE
    if first.exists():
        first.replace(Path(out) / KV_SHORTFALL_LOG)
    try:
        shrink = sg.kv_shrink_record(configured_max_num_seqs=int(geom.max_num_seqs),
                                     max_model_len=int(geom.max_model_len),
                                     kv_backed_concurrency=short.kv_backed_concurrency,
                                     kv_capacity_tokens=short.kv_capacity_tokens, basis=short.basis,
                                     first_launch_log=KV_SHORTFALL_LOG)
        new = sg.kv_shrunk_geometry(geom, shrink)
    except ValueError as e:
        raise vs.VllmServerError(
            f"KV-derived geometry impossible ({e}) — the container's KV cache backs "
            f"{short.kv_backed_concurrency:.2f} full-length sequences at L={geom.max_model_len} "
            f"({short.kv_capacity_tokens:,} tokens; {short.basis}); the original refusal: {short}") from short
    print(f"{TAG} KV-DERIVED SHRINK (kv_derived_seqs on): the container's KV cache backs "
          f"{short.kv_backed_concurrency:.2f} full-length sequences at L={geom.max_model_len} "
          f"({short.kv_capacity_tokens:,} tokens; {short.basis}) < the configured S={geom.max_num_seqs} — "
          f"re-launching ONCE at S={new.max_num_seqs} (= floor); the first launch's engine log is "
          f"{KV_SHORTFALL_LOG}", flush=True)
    return new, shrink


# --------------------------------------------------------------------------- #
# the thinking regime under native (mt's certification, mirrored)               #
# --------------------------------------------------------------------------- #

#: the per-template certification VERDICTS a thinking-ON standup accepts (mt's
#: ``THINKING_SERVABLE_VERDICTS``): a moved, certified header or an unmoved one
THINKING_SERVABLE_VERDICTS = ("certified", "trivially_in_support")


def certify_thinking(tok: Any, *, model_key: Optional[str]) -> Dict[str, Any]:
    """The served template's thinking-ON certification (``chat_markers.
    thinking_on_certification``) with the model key stamped on, NON-FATAL: a
    derivation error is recorded as ``verdict: "error"``, never raised — an OFF
    standup records it; an ON standup refuses a non-servable verdict."""
    from sisteer.chat_markers import generation_header_ids, thinking_on_certification
    try:
        cert = dict(thinking_on_certification(tok))
    except Exception as e:  # noqa: BLE001 — recorded as an error verdict
        cert = {"switch": "enable_thinking" in (getattr(tok, "chat_template", None) or ""),
                "off_ids": list(generation_header_ids(tok)), "on_ids": None, "moves": None,
                "checks": {}, "verdict": "error", "reason": f"{type(e).__name__}: {e}"}
    cert["model_key"] = model_key
    cert["off_header_len"] = len(cert["off_ids"])
    cert["on_header_len"] = None if cert.get("on_ids") is None else len(cert["on_ids"])
    return cert


def thinking_header_fact(certification: Mapping[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """``(fact, error)`` — the proxy's ``thinking_header`` fact read off a
    certification, and why its ON half is UNAVAILABLE (or None)."""
    switch = bool(certification.get("switch"))
    off_ids = list(certification["off_ids"])
    on_ids = certification.get("on_ids")
    if not switch:
        return {"switch": False, "moves": None, "off_ids": off_ids, "on_ids": None}, None
    if on_ids is None:
        return ({"switch": True, "moves": None, "off_ids": off_ids, "on_ids": None},
                f"{certification.get('verdict')}: {certification.get('reason')}")
    on_ids = [int(t) for t in on_ids]
    return {"switch": True, "moves": on_ids != off_ids, "off_ids": off_ids,
            "on_ids": on_ids}, None


# --------------------------------------------------------------------------- #
# the bring-up                                                                 #
# --------------------------------------------------------------------------- #


def _enumerate_native(spec: Any, src: Any, *, behavior_files: Mapping[str, bytes],
                      prompt_files: Mapping[str, bytes], zero_only: bool) -> Any:
    """``enumerate_cells(lens_controls=False, lora_controls=<zero_only>)`` — the
    zero-adapter leg derived beside the lora cells, or ALONE on a zero-only
    standup."""
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri import registry_adapter as ra

    try:
        return dc.enumerate_cells(spec, src, behavior_files=behavior_files, prompt_files=prompt_files,
                                  lens_controls=False, lora_controls=bool(zero_only))
    except (dc.DefenseCellError, ra.RegistryAdapterError) as e:
        raise StandupRefused(f"enumeration refused: {type(e).__name__}: {e}") from e


def residual_slot_facts(att: Mapping[str, Any], served_kinds: Mapping[str, str]) -> Dict[str, Any]:
    """``{"armed": bool, "cells": {cell: {"kind", "residual_fingerprint", "position_mode"}}}``
    off a readiness attestation (schema 2) for the served lora cells whose adapter
    carries a residual payload; refuses when the attestation's ``residual_slot.
    armed`` disagrees with the cells served, a residual cell's attested kind is
    not its registry kind, a served mode is outside :data:`NATIVE_POSITION_MODES`, or
    a served mode is RESTRICTED (not ``all``) while ``defense_cells.NATIVE_STAGE2_ARMED``
    is False (stage 1: the disposition skips such a cell by name)."""
    adapters = att.get("adapters") or {}
    cells: Dict[str, Dict[str, Any]] = {}
    for cell, kind in served_kinds.items():
        block = adapters.get(cell) or {}
        rfp = block.get("residual_fingerprint")
        if kind in RESIDUAL_SLOT_KINDS and rfp is None:
            raise StandupRefused(
                f"cell {cell!r} (kind {kind!r}) is a residual-slot kind but its attested adapter carries "
                "no residual payload — the export did not bank a residual entry for it")
        if rfp is None:
            continue
        if block.get("petri_source_kind") != kind:
            raise StandupRefused(
                f"cell {cell!r}: the attested adapter's petri_source_kind {block.get('petri_source_kind')!r} "
                f"!= the registry kind {kind!r}")
        mode = block.get("position_mode")
        if mode not in NATIVE_POSITION_MODES:
            raise StandupRefused(
                f"cell {cell!r}: attested position_mode {mode!r} is not a native position mode "
                f"{list(NATIVE_POSITION_MODES)}")
        if mode != NATIVE_POSITION_MODE_ALL and not _dc.NATIVE_STAGE2_ARMED:
            # stage 1: the disposition skips a restricted-mode cell by name, so an attested
            # restricted mode here means the worker loaded a payload nothing else admitted
            raise StandupRefused(
                f"cell {cell!r}: attested position_mode {mode!r} — stage 1 serves "
                "mode 'all' only (defense_cells.NATIVE_STAGE2_ARMED is False); the disposition "
                "should have skipped this cell")
        cells[cell] = {"kind": kind, "residual_fingerprint": rfp, "position_mode": mode}
    armed = bool((att.get(sa.RESIDUAL_SLOT_KEY) or {}).get("armed"))
    if armed != bool(cells):
        raise StandupRefused(
            f"the attestation says residual_slot.armed={armed} but {len(cells)} served cell(s) carry a "
            "residual payload — the worker's slot state and the served cells disagree")
    return {"armed": armed, "cells": cells}


def bring_up_native(eff: EffectiveServing, *, out: Path, eval: Optional[str], geometry: Any,  # noqa: A002
                    tool_calls: bool, adapter_bank: Path,
                    thinking_on: bool = False, reasoning_parser: Optional[str] = None,
                    sampling_clamp_override: Optional[float] = None, text_only_mm: str = "off",
                    mamba_backend_contract: Optional[Mapping[str, Any]] = None,
                    kv_derived_seqs: bool = False, enable_log_requests: bool = False,
                    probe_max_tokens: int = 48, gpu_mem_util: str = "0.9",
                    max_logprobs: Optional[int] = None, proxy_port: Optional[int] = None,
                    vllm_port: Optional[int] = None, ready_timeout_s: int = 1800,
                    install_timeout_s: int = 900, upstream_timeout_s: Optional[float] = None,
                    petri_agent_loop: bool = True, geometry_override_reason: Optional[str] = None,
                    allow_over_budget: bool = False, mode: Optional[str] = None,
                    run_controls: bool = True,
                    attest_mrope: bool = False, attempt_id: Optional[str] = None,
                    server_default_policy: str = "contract_override",
                    front: Optional[str] = None,
                    controls_fn: Optional[Any] = None,
                    extra_endpoint_facts: Optional[Mapping[str, Any]] = None,
                    launch_extras: Sequence[str] = (),
                    output_diagnostics: Optional[bool] = None,
                    parity_record_check: Optional[Any] = None) -> LoraStandup:
    """THE NATIVE CORE (module docstring): the whole bring-up from the served
    template onward for the EFFECTIVE serving ``eff`` a front normalised.
    Returns the live :class:`LoraStandup` with ``positive_controls.json``,
    ``cells_served.json`` and ``endpoint.json`` on disk (``run_controls=False``
    skips the controls AND ``endpoint.json``).

    Raises :class:`StandupRefused` (exit 2) for every preflight refusal;
    ``vllm_server.VllmServerError`` UNWRAPPED from compose / readiness (the
    server, if spawned, is already shut down by the launcher);
    :class:`PositiveControlsFailed` (exit 4) after tearing the server down.
    Any failure after the server spawned shuts it down before re-raising.

    ``endpoint.json`` is a READINESS record bound to ONE attempt (F3): a prior
    attempt's copy under ``out`` is moved aside before anything else runs, every
    record carries this attempt's nonce under :data:`ENDPOINT_ATTEMPT_KEY`, and
    every failed path unpublishes the endpoint.

    ``controls_fn`` (default :func:`run_positive_controls`) is the seam a front
    hands its own controls binding through (the wave front passes its module's
    name so the certification's tests can hook it); ``extra_endpoint_facts`` are
    a front's OWN provenance keys, recorded verbatim in both records — a key the
    core already writes refuses (one writer per fact). ``launch_extras`` are the
    WRAPPER-OWNED argv tokens the front composes and the core appends after the
    text-only pin (``mt_serving_up.launch_cmd_extras`` / capctrl's trailing
    tuple: the memory utilization, the logprob cap, the resolved-sampler log
    opt-in) — spelled by the fronts, never by this frozen serving tree;
    ``gpu_mem_util`` / ``max_logprobs`` / ``enable_log_requests`` are the same
    facts as RECORDED, and the core refuses a mismatch between the two."""
    from transformers import AutoConfig

    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.serving import lora_export as lx
    from sisteer.capeval.serving import mamba_backend as mb
    from sisteer.capeval.serving import standup_thinking as st
    from sisteer.capeval.serving import vllm_server as vs
    from sisteer.capeval.serving.proxy import (ADAPTERS_KEY, SERVER_DEFAULT_POLICY_KEY, THINKING_HEADER_ERROR_KEY,
                                               HTTPBackend, ProxyError, RoutingProxy,
                                               TokenizerAgentLoopRender, export_routing_table,
                                               load_routing_table, make_app,
                                               verify_adapter_block_on_disk)

    eval_name = eval
    front_name = front or eff.front
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # F3: this attempt's nonce (the front may have minted it and quarantined already —
    # then this is a no-op), and NO prior attempt's readiness record in the way
    attempt_id = attempt_id or secrets.token_hex(16)
    stale = quarantine_prior_endpoint(out, attempt_id=attempt_id)
    quarantine_prior_standup_evidence(out, attempt_id=attempt_id, stale_dir=stale)
    if probe_max_tokens is None or int(probe_max_tokens) < 1:
        raise StandupRefused(f"--probe-max-tokens must be >= 1, got {probe_max_tokens!r}")
    if upstream_timeout_s is not None and not float(upstream_timeout_s) > 0:
        raise StandupRefused(f"--upstream-timeout-s must be > 0, got {upstream_timeout_s}")
    if text_only_mm not in vs.TEXT_ONLY_MM_MODES:
        raise StandupRefused(f"--text-only-mm {text_only_mm!r} is not one of {vs.TEXT_ONLY_MM_MODES}")
    if not isinstance(tool_calls, bool):
        raise StandupRefused(f"tool_calls must be an explicit bool, got {tool_calls!r} — the core never "
                             "infers the agent modality")
    extra_facts = dict(extra_endpoint_facts or {})
    if extra_facts:
        clash = sorted(k for k in extra_facts if k in _CORE_RECORD_KEYS)
        if clash:
            raise StandupRefused(f"extra_endpoint_facts {clash} collide with keys the core records — one "
                                 "writer per fact")
    run_controls_fn = controls_fn if controls_fn is not None else run_positive_controls
    geom = geometry
    model, rev, model_key = eff.model, eff.revision, eff.model_key
    spec = eff.spec
    if tuple(spec.cells) != tuple(eff.served_cells):
        raise StandupRefused(f"the effective spec's cells {list(spec.cells)} != the served cells "
                             f"{list(eff.served_cells)} — the front must hand the core ONE enumeration")
    if (spec.target_model, spec.target_revision) != (model, rev):
        raise StandupRefused(f"the effective spec pins {spec.target_model}@{spec.target_revision} but the "
                             f"front names {model}@{rev} — one standup, one target")
    kept_flags, dropped_by_core = effective_launch_flags(spec.serving_launch_flags)
    if dropped_by_core:
        raise StandupRefused(f"the effective spec still carries {dropped_by_core} — a front passes its "
                             "manifest flags through effective_launch_flags BEFORE handing the core the spec")
    print(f"{TAG} geometry: {geom.describe()} (model_key={model_key}, "
          f"seqs_source={getattr(geom, 'seqs_source', 'n/a')}"
          + (f", override_reason={geometry_override_reason!r}" if geometry_override_reason else "")
          + ")", flush=True)
    # --- the model's HF config: the family the modality pins and the Nemotron rule key on ----
    mcfg = AutoConfig.from_pretrained(model, revision=rev)
    family = getattr(mcfg, "model_type", None)
    if model_key == GLM_MODEL_KEY or family == GLM_FAMILY:
        raise StandupRefused(
            f"model_key={model_key!r} family={family!r}: GLM-4.6V-Flash is EXCLUDED from the native "
            "engine (its non-neox M-RoPE patch rides the lens plugin, which a plugin-free launch keeps "
            "out) — pass --serving-path lens")
    if eff.cells_skipped:
        for s in eff.cells_skipped:
            print(f"{TAG} SKIPPED by name: cell {s.cell_id!r} (kind {s.kind!r}) — {s.reason}; "
                  "pass --serving-path lens to serve it on the lens", flush=True)

    # --- the served template ----------------------------------------------------------
    tok = eff.tokenizer
    template_bytes = bytes(eff.chat_template_bytes)
    tpath, tsha16 = write_chat_template(out, template_bytes)
    if tsha16 != spec.chat_template_hash:
        raise StandupRefused(f"served chat template sha16 {tsha16} != the effective spec's "
                             f"chat_template_hash {spec.chat_template_hash} — the front handed the core "
                             "a spec whose template pin is not the served bytes'")
    tsha256 = sa.sha256_bytes(template_bytes)

    # --- THE THINKING REGIME, before any GPU minute (mt's rule, mirrored) -----------------------
    thinking_cert = certify_thinking(tok, model_key=model_key)
    thinking_mode = "on" if thinking_on else "off"
    if thinking_on:
        verdict = thinking_cert.get("verdict")
        if verdict not in THINKING_SERVABLE_VERDICTS:
            raise StandupRefused(
                f"--thinking-on REFUSED for model_key={model_key}: the served template's thinking-ON "
                f"header certification is {verdict!r} — {thinking_cert.get('reason')}. Never serving an "
                f"uncertified thinking-ON standup; checks={json.dumps(thinking_cert.get('checks'), default=str)}")
        # a native standup has no header-dependent cell to declare (LoRA cells are all-position;
        # a declared variant refuses through native_lora_unservable_reason) — declared_cells: []
        thinking_header, thinking_header_error = thinking_header_fact(thinking_cert)
    else:
        thinking_header, thinking_header_error = st.standup_thinking_header(tok, declares_thinking=False)

    # --- the cells: the lora-native enumeration (no lens leg, the zero leg always) -------------------
    src = eff.cell_source
    if getattr(src, "serving_path_policy", "lora_native") != "lora_native":
        raise StandupRefused(f"the core needs a lora_native-policy cell source, got "
                             f"serving_path_policy={getattr(src, 'serving_path_policy', None)!r}")

    def _path(cell) -> str:
        return cm.parse_serving_identity(cell.serving_identity).path

    # a first pass tells whether ANY lora route is live (the zero leg then derives on its own)
    probe_set = _enumerate_native(spec, src, behavior_files=eff.behavior_files,
                                  prompt_files=eff.prompt_files, zero_only=False)
    zero_only = not any(_path(c) == "lora" for c in probe_set.canonical)
    cell_set = (_enumerate_native(spec, src, behavior_files=eff.behavior_files,
                                  prompt_files=eff.prompt_files, zero_only=True)
                if zero_only else probe_set)
    if cell_set.absent:
        raise StandupRefused("ABSENT cells (bank incomplete?): "
                             f"{[(a.cell_id, a.reason) for a in cell_set.absent]}")
    lora_cells = [c for c in cell_set.canonical if _path(c) == "lora"]
    lens_cells = [c.cell_id for c in cell_set.all_served() if _path(c) == "lens"]
    if lens_cells:
        raise StandupRefused(f"lens route(s) {lens_cells} on a lora-native table — a plugin-free "
                             "server has no lens hook")
    zero_leg = next((c for c in cell_set.controls if c.cell_id == dc.NOOP_LORA_CELL_ID), None)
    if zero_leg is None:
        raise StandupRefused(f"enumerate_cells derived no {dc.NOOP_LORA_CELL_ID} leg")
    vmodels = {c.cell_id: c.virtual_model_name for c in cell_set.all_served()}
    if zero_only:
        print(f"{TAG} ZERO-ADAPTER-ONLY standup: no lora cell among {list(eff.served_cells)} — the zero "
              f"adapter {dc.NOOP_LORA_CELL_ID} is the one registered adapter; steering_bites and "
              "selection_honored will be recorded not_applicable (the base and steered configurations "
              "share one engine, finding 8)", flush=True)

    # --- the verified object map: hold every lora cell's bytes to its digest FIRST ----------
    expected_digests = dict(eff.expected_digests)
    objmap: Dict[str, str] = {}
    artifact_bytes: Dict[str, bytes] = {}
    sidecar_bytes: Dict[str, bytes] = {}
    for cell in lora_cells:
        entry = src.get(cell.cell_id)
        if entry is None or not entry.object_path:
            raise StandupRefused(f"lora cell {cell.cell_id!r}: the cell source has no object for it")
        data = Path(entry.object_path).read_bytes()
        sha = sa.sha256_bytes(data)
        want = expected_digests.get(cell.cell_id)
        if want is None:
            raise StandupRefused(f"lora cell {cell.cell_id!r} has no --expect-object-sha256 "
                                 "(REQUIRED per fitted cell; capctrl's E45 H1 boundary)")
        if sha != want:
            raise StandupRefused(
                f"lora cell {cell.cell_id!r}: object.pt at {entry.object_path} hashes to {sha}, "
                f"--expect-object-sha256 says {want} — an object replaced since the preflight; "
                "refusing before any export")
        objmap[cell.cell_id] = sha
        artifact_bytes[cell.cell_id] = data
        sidecar_bytes[cell.cell_id] = (Path(entry.sidecar_path).read_bytes()
                                       if entry.sidecar_path else b"")
    stray = sorted(set(expected_digests) - set(objmap))
    if stray:
        raise StandupRefused(f"--expect-object-sha256 names cell(s) {stray} that serve no lora "
                             "adapter here")

    # --- the adapter bank ------------------------------------------------------------------
    bank = Path(adapter_bank).resolve()
    blocks: Dict[str, Dict[str, Any]] = {}
    position_modes: Dict[str, Optional[str]] = {}   # cell -> native position mode (residual entries)
    reused: Dict[str, bool] = {}
    for cell in lora_cells:
        entry = src.get(cell.cell_id)
        resolved = dict(entry.resolved_spec or {})
        resolved["intervention_hash"] = cell.intervention_hash
        try:
            adir, dig, was_reused = lx.export_or_reuse(
                resolved, artifact_bytes[cell.cell_id], bank, base_model=model,
                base_revision=rev, sidecar_bytes=sidecar_bytes[cell.cell_id],
                name_hint=cell.cell_id)
        except lx.LoraExportError as e:
            raise StandupRefused(
                f"adapter export refused for cell {cell.cell_id!r}: {e}") from e
        blocks[cell.cell_id] = _adapter_block(
            cell.cell_id, adir, dig, intervention_hash=cell.intervention_hash,
            zero=bool(dig.petri_zero_adapter))
        # THE ATTESTED POSITION MODE (native-positions-resolve): a residual-slot adapter's from its own
        # `petri_residual_slot` block (the export's record, read by adapter_digests); a PLAIN LoRA adapter's `all`
        # from its KIND (a PEFT adapter at every module call, no residual slot). Never from the engine's name: a
        # residual-kind adapter whose block states no mode stays None — absent from the routes stamp and the
        # served_positions map, refused by the readers by name (NATIVE_POSITION_MODE_UNATTESTED)
        position_modes[cell.cell_id] = attested_position_mode(dig, kind=str(entry.kind))
        reused[cell.cell_id] = bool(was_reused)
        if dig.petri_intervention_hash != cell.intervention_hash:
            raise StandupRefused(
                f"adapter for {cell.cell_id!r}: the export recorded petri_intervention_hash "
                f"{dig.petri_intervention_hash!r} != the cell's {cell.intervention_hash!r}")
    # THE OUTPUT-DIAGNOSTICS POLICY (module docstring): decided HERE, once, from the served
    # cells' kinds — every front door inherits it through controls_fn. A caller may pass
    # ``output_diagnostics``: None = the policy; False = OFF whatever the policy says (the
    # certification driver's `numerics=not_measured` row states it explicitly); True = ON, which
    # the policy may REFUSE (a residual-slot cell is served: greedy-token agreement is never
    # measured for it, and no front may turn it back on)
    served_kinds = {c.cell_id: str(src.get(c.cell_id).kind) for c in lora_cells}
    output_diagnostics_requested = output_diagnostics
    policy = output_diagnostics_policy(served_kinds.values())
    if output_diagnostics is None:
        output_diagnostics = policy
    elif bool(output_diagnostics) and not policy:
        raise StandupRefused(
            "output_diagnostics=True requested but the central policy is OFF for this standup: "
            f"residual-slot cell(s) served {sorted(c for c, k in served_kinds.items() if k in RESIDUAL_SLOT_KINDS)} "
            "— greedy-token agreement is never measured for a residual kind (native-psr-serving D19); "
            "a caller may only turn the diagnostics OFF")
    else:
        output_diagnostics = bool(output_diagnostics)
    if not output_diagnostics:
        print(f"{TAG} output diagnostics OFF: residual-slot cell(s) served "
              f"{sorted(c for c, k in served_kinds.items() if k in RESIDUAL_SLOT_KINDS)} — "
              "greedy-token agreement is not measured (native-psr-serving D19)", flush=True)
    # THE ZERO ADAPTER IS CLASS-AWARE: the launch
    # restricts vLLM's LoRA wrappers to the ONE module class every registered adapter adapts
    # (vllm_server._adapted_module_class), so the zero leg must adapt exactly that class —
    # an o_proj-only zero beside `mlp` cells would refuse the launch (or, under a laxer
    # rule, certify a mislabelled control). The class is read off the GENUINE adapters just
    # banked (lora_export.adapter_target_class: factor keys + `experts` where the file
    # carries routed-expert pairs); one class, else refuse here by name. The paper's
    # o_proj class keeps today's generator and bank entry byte-for-byte; any other class
    # goes through export_zero_adapter_for_class (all-zero factors for every module of the
    # class, the two obligatory expert pairs included). A zero-only standup has no genuine
    # adapter to read and keeps the o_proj class. The generator is export_zero_adapter_for_class
    # for EVERY class: its o_proj branch IS export_or_reuse_zero (byte-for-byte, pinned by
    # lora_export's tests), so the dispatch lives in one place.
    # A RESIDUAL-SLOT CARRIER IS NOT A LoRA OF THE CLASS (the steered canary, 2026-09-30): its rank-1 zero o_proj pair only
    # carries vLLM's slot and the payload, so it never decides the launch class and rides INSIDE the genuine adapters'
    # (lora_export.launch_target_class — the composer's _adapted_module_class reads the same rule; a carrier outside the class,
    # or genuine adapters that disagree, refuse by name). Before this a psr carrier beside an attn_mlp LoRA refused here.
    try:
        classes = {c: (lx.adapter_target_class(Path(b["dir"])), str(b.get("petri_source_kind")) in RESIDUAL_SLOT_KINDS)
                   for c, b in sorted(blocks.items())}
        try:
            zero_class = lx.launch_target_class(classes)
        except lx.LoraExportError as e:
            raise StandupRefused(f"the genuine adapters of this standup adapt different module classes — {e}") from e
        carriers_in_class = sorted(c for c, (_k, carrier) in classes.items() if carrier)
        zdir, zdig, z_reused = lx.export_zero_adapter_for_class(
            bank, base_model=model, base_revision=rev, rank=1,
            lora_modules=lx.modules_of_class(zero_class))
        lx.verify_zero_adapter(zdir)
        zero_file_class = lx.adapter_target_class(zdir)
        if zero_file_class != zero_class:
            raise StandupRefused(
                f"the zero adapter at {zdir} adapts class {list(zero_file_class)} where the "
                f"genuine adapters adapt {list(zero_class)}; refusing")
    except lx.LoraExportError as e:
        raise StandupRefused(f"adapter export refused: {e}") from e
    print(f"{TAG} zero adapter class {list(zero_class)} (from {len(classes) - len(carriers_in_class)} genuine adapter(s)"
          + (f"; residual-slot carrier(s) {carriers_in_class} ride inside it" if carriers_in_class else "") + ")", flush=True)
    zero_block = _adapter_block(dc.NOOP_LORA_CELL_ID, zdir, zdig,
                                intervention_hash=zero_leg.intervention_hash, zero=True)
    reused[dc.NOOP_LORA_CELL_ID] = bool(z_reused)
    print(f"{TAG} adapters banked under {bank}: "
          f"{[(c, 'reused' if reused[c] else 'exported') for c in reused]}", flush=True)

    # --- the VERSION-2 routing table, re-loaded and re-hashed ------------------------------
    table_path = out / ROUTING_TABLE_FILE
    all_blocks = {**blocks, dc.NOOP_LORA_CELL_ID: zero_block}
    try:
        export_routing_table(cell_set, src, table_path, base_model=model,
                             serving_mode="interleaved", base_revision=rev,
                             thinking_header=thinking_header,
                             thinking_header_error=thinking_header_error,
                             adapter_blocks=all_blocks, position_modes=position_modes)
        table = load_routing_table(table_path)
    except ProxyError as e:
        raise StandupRefused(f"routing table refused: {e}") from e
    if table.get("version") != 2 or not table.get(ADAPTERS_KEY):
        raise StandupRefused(f"routing table at {table_path} is version {table.get('version')!r} "
                             "— a lora-native standup serves from a version-2 table with adapter "
                             "blocks only; refusing")
    try:
        for block in table[ADAPTERS_KEY].values():
            verify_adapter_block_on_disk(block)
    except ProxyError as e:
        raise StandupRefused(f"adapter block does not match the bank on disk: {e}") from e
    served = [c.cell_id for c in cell_set.all_served()]
    print(f"{TAG} version-2 routing table exported: {len(served)} legs {served}; "
          f"blocks {sorted(all_blocks)}", flush=True)

    # --- THE ADAPTER-PARITY GATE, BEFORE THE LAUNCH -------------------------------------------------------
    # every GENUINE adapter's (rank, set) — read off its banked files — must be a pair the staged site overlay certifies with
    # adapter parity `verified`; the zero adapter alone / a prompted-only standup gates nothing; an onboarded model with no
    # overlay refuses; when the overlay's probe record is reachable (the probe cluster) and the front handed
    # `parity_record_check` (hpo_probe_status.probe_state_record — the front door's own predicate), it must read READY
    from . import geometry as _sg
    # the live predicate is MANDATORY for every native front: None = the DEFAULT (the front door's own
    # hpo_probe_status.probe_state_record), a callable = the caller's (lora_serving_up hands the same one in), and the one opt-out is
    # PARITY_GATE_OPT_OUT by name — a caller whose standup IS the certification; the record says which applied
    if parity_record_check == PARITY_GATE_OPT_OUT:
        record_check, record_predicate = None, f"opted_out:{PARITY_GATE_OPT_OUT}"
    elif parity_record_check is None:
        record_check, record_predicate = default_parity_record_check, "default:hpo_probe_status.probe_state_record"
    else:
        record_check, record_predicate = parity_record_check, "caller"
    try:
        adapter_parity_gate = _sg.standup_adapter_parity_gate(
            str(model_key), genuine_adapter_pairs(blocks), record_check=record_check,
            what=f"this lora-native standup ({front or 'native'})",
            stamp_not_before=(os.environ.get(UNIT_SUBMITTED_UTC_ENV) or None),
            serving_venv=sa.serving_venv_facts())   # The pair is served only in the venv its certificate was measured in
    except _sg.SiteGeometryError as e:
        raise StandupRefused(f"adapter parity gate: {e}") from e
    # the label says what ACTUALLY ran: the live predicate only when the record was reachable and read; the
    # staging stamp when the record is on another cluster; never "default" for a predicate that never ran
    rec_pred = ((adapter_parity_gate.get("record") or {}).get("predicate"))
    if rec_pred == "live":
        adapter_parity_gate["record_predicate"] = record_predicate
    elif rec_pred is None:
        adapter_parity_gate["record_predicate"] = "not_applicable:" + adapter_parity_gate["status"]
    else:
        adapter_parity_gate["record_predicate"] = rec_pred
    print(f"{TAG} adapter parity gate: {adapter_parity_gate['status']}"
          + (f" — {[(p['adapter'], p['rank'], p['set'], p['status']) for p in adapter_parity_gate['pairs']]}" if adapter_parity_gate.get("pairs") else "")
          + (f"; probe record {adapter_parity_gate['record']['dir']} reads {adapter_parity_gate['record']['state']}" if adapter_parity_gate.get("record") else ""),
          flush=True)

    # --- THE LAUNCH CLASS, BEFORE THE KERNEL CEILING (2026-10-02; the s2r pair target 831867) --------
    # The unsteered site default's per-window S is the least admitted S over EVERY probed launch, so a rank-64 x 128-adapter launch
    # admitted at S=1 capped a rank-1 x 3-adapter standup at one sequence per step — a width at which the multi-LoRA routing control
    # (selection_honored) cannot pass. The launches that DOMINATE the class this standup serves (rank cap >=, adapters >=), at the
    # tightest dominating footprint, size S instead — as a RECORDED adjustment the wrapper's geometry compare reads (from the
    # record-wide S the caller resolved to the class-aware S). A record with no launch classes keeps today's S; a width below two with
    # two or more adapters is refused HERE, in seconds, never at the control fourteen minutes later.
    adjustments: List[Dict[str, Any]] = []
    served_lora = (int(vs.fitted_max_lora_rank([Path(b["dir"]) for b in all_blocks.values()])), len(all_blocks)) if all_blocks else None
    geom, class_step = launch_class_geometry(geom, served_lora=served_lora, eval_name=eval_name, model_key=model_key,
                                             allow_over_budget=allow_over_budget)
    if class_step is not None:
        adjustments.append(class_step)
        geometry_override_reason = geometry_override_reason or _sg.ADJUSTMENT_LAUNCH_CLASS
        print(f"{TAG} geometry re-sized to the served LAUNCH CLASS (rank cap {class_step['max_lora_rank']}, {class_step['max_loras']} adapters): "
              f"S {class_step['from_S']} -> {class_step['to_S']} at L={class_step['L']} ({class_step['window_seqs_source']}; "
              f"launches {class_step['launches']})", flush=True)
    if int(geom.max_num_seqs) < 2 and len(all_blocks) >= 2:
        how = ((class_step or {}).get("window_seqs_source")
               or "no launch-class re-sizing applied: the record carries no launch classes, or the served class is already the tightest")
        raise StandupRefused(
            f"max_num_seqs={geom.max_num_seqs} with {len(all_blocks)} adapters: the multi-LoRA routing control (selection_honored) needs two "
            f"distinct selections in ONE engine step and cannot pass at one sequence per step — the site record sized window "
            f"{geom.max_model_len} at {geom.max_num_seqs} ({how}); re-fold the site record with launch classes (serving_probe geometry_fold "
            f"at a tip that records them) or re-measure the window")

    # --- THE LORA KERNEL INDEX CEILING, BEFORE THE LAUNCH (the second §1.8 adjustment) --------
    kernel_ceiling, geometry_lowered = lora_kernel_ceiling_geometry(
        all_blocks, geom, eval_name=eval_name, model_key=model_key,
        allow_over_budget=allow_over_budget, override_reason=geometry_override_reason)
    if geometry_lowered is not None:
        geom = geometry_lowered.pop("_geom")
        geometry_override_reason = geometry_override_reason or sa.LORA_KERNEL_CEILING_REASON
        adjustments.append(kernel_adjustment_record(geometry_lowered))
        print(f"{TAG} geometry LOWERED to the LoRA kernel index ceiling: S "
              f"{geometry_lowered['requested_S']} -> {geometry_lowered['S']} at L={geometry_lowered['L']} "
              f"(D={geometry_lowered['D']} at {geometry_lowered['widest_module']}, ceiling "
              f"{geometry_lowered['ceiling']:,} tokens; requested product "
              f"{geometry_lowered['product_requested']:,})", flush=True)

    # --- the launch --------------------------------------------------------------------------
    vllm_port = int(vllm_port) if vllm_port else free_port()

    def compose(max_num_seqs: int):
        # ONE composer for the configured launch and the KV-derived re-launch
        try:
            return vs.build_server_launch(
                spec, port=vllm_port, lora_native=True,
                lora_modules={cid: Path(b["dir"]) for cid, b in all_blocks.items()},
                chat_template_path=tpath, max_model_len=geom.max_model_len,
                max_num_seqs=max_num_seqs, tool_calls=tool_calls,
                thinking_on=bool(thinking_on), reasoning_parser=reasoning_parser,
                sampling_clamp_override=sampling_clamp_override,
                # THE SERVER-DEFAULT SAMPLING POLICY (vllm_server.SERVER_DEFAULT_POLICIES):
                # recorded on every record below and echoed by the proxy's identity block
                server_default_policy=server_default_policy)
        except vs.VllmServerError:
            raise

    launch = compose(geom.max_num_seqs)
    # THE TEXT-ONLY MULTIMODAL PIN (capctrl's rule): decided from the model's config against the
    # composed argv; composed tokens ride the wrapper-owned extras; the decision is recorded
    tom = vs.text_only_mm_decision(text_only_mm, model_id=model, revision=rev,
                                   family=family, cmd=launch.cmd)
    tom_tokens = tuple(tom["tokens"]) if tom.get("source") == "standup" else ()
    extras = tuple(str(a) for a in launch_extras)
    # the recorded facts and the front's extras must agree: the utilization value must ride the argv,
    # the logprob cap iff recorded, the log opt-in iff recorded (the token spellings are the fronts')
    if str(gpu_mem_util) not in extras:
        raise StandupRefused(f"launch_extras {list(extras)} do not carry gpu_mem_util={gpu_mem_util!r} — the front "
                             "composes the wrapper-owned flags and records the same values")
    if (max_logprobs is not None) != (str(int(max_logprobs)) in extras if max_logprobs is not None else False):
        raise StandupRefused(f"launch_extras {list(extras)} disagree with max_logprobs={max_logprobs!r}")
    if bool(enable_log_requests) != any(a.endswith("log-requests") for a in extras):
        raise StandupRefused(f"launch_extras {list(extras)} disagree with enable_log_requests={enable_log_requests!r}")
    trailing = extras + tom_tokens

    def with_trailing(base):
        return dataclasses.replace(base, cmd=tuple(base.cmd) + trailing)

    launch = with_trailing(launch)
    pre_launch = {
        "engine": ENGINE, "stage": "pre_launch", "output_diagnostics": output_diagnostics,
        "output_diagnostics_policy": policy, "output_diagnostics_requested": output_diagnostics_requested,
        ENDPOINT_ATTEMPT_KEY: attempt_id,
        "serving_engine": ENGINE, "standup_front": front_name,
        "mode": mode, "eval": eval_name, "model_key": model_key, "model": model, "model_revision": rev,
        "manifest": str(eff.manifest_path), "config": str(eff.config_path),
        "config_used": str(eff.config_used), "defenses_dir": str(eff.defenses_dir),
        "adapter_bank": str(bank), "written_at": _utc_now(),
        "cells_requested": list(eff.traintest_cells), "harness_cells": list(eff.served_cells),
        REQUESTED_EXPLICIT_KEY: list(eff.cells_requested_explicit),                         # what was NAMED
        REQUESTED_EXPLICIT_HARNESS_KEY: list(eff.cells_requested_explicit_harness),
        "cells_skipped": [dict(s._asdict()) for s in eff.cells_skipped],
        "lora_cells": [c.cell_id for c in lora_cells], "zero_adapter_only": zero_only,
        "prompted_cells_admitted": list(eff.prompted_cells_admitted),
        "manifest_flags_dropped": list(eff.manifest_flags_dropped),
        "geometry_override_reason": geometry_override_reason,
        "geometry": {**geom.as_dict(), "adjustments": [dict(a) for a in adjustments]},
        "geometry_lowered": geometry_lowered, sa.LORA_KERNEL_CEILING_KEY: kernel_ceiling,
        "adapter_parity_gate": adapter_parity_gate,
        "object_artifact_sha256_by_cell": dict(objmap), "expected_digests_source": eff.expected_digests_source,
        "adapters": blocks, "zero_adapter": zero_block, "adapters_reused": reused,
        "routing_table": str(table_path), "routing_table_version": int(table["version"]),
        "chat_template_hash": tsha16, "chat_template_sha256": tsha256,
        "thinking_mode": thinking_mode, "thinking_certification": thinking_cert,
        "reasoning_parser": reasoning_parser, "tool_calls": tool_calls,
        "text_only_mm": tom.get("text_only_mm"), "text_only_mm_record": tom,
        "enable_log_requests": bool(enable_log_requests),
        "override_generation_config": getattr(launch, "override_generation_config", None),
        "server_default_policy": launch.server_default_policy,
        "launch_cmd": list(launch.cmd)}
    sa.write_json_atomic(out / CELLS_SERVED_FILE, pre_launch, default=str)
    print(f"{TAG} launching lora-native vLLM on {launch.base_url} ({geom.max_num_seqs} x "
          f"{geom.max_model_len}; {len(launch.lora_modules)} adapters, rank cap "
          f"{launch.max_lora_rank}; tool_calls={tool_calls} thinking={thinking_mode} "
          f"parser={reasoning_parser!r} clamp={launch.sampling_clamp_override!r} text_only_mm={tom['text_only_mm']}) "
          f"node={socket.gethostname()}", flush=True)

    def start(composed):
        return vs.launch_server(composed, log_path=out / SERVER_LOG_FILE,
                                ready_timeout_s=ready_timeout_s, install_timeout_s=install_timeout_s)

    kv_shrink = None
    try:
        server = start(launch)
    except vs.KvCoverageShortfall as short:
        # §1.8: readiness refused for the ONE reason a KV-derived S may answer. Opted in:
        # ONE re-launch at floor(the reported concurrency), same window, composed AFTER any kernel
        # lowering (the shrink's configured S is the kernel-lowered S); a second shortfall raises.
        if not kv_derived_seqs:
            raise
        geom, kv_shrink = kv_derived_geometry(geom, short, out)
        adjustments.append(kv_adjustment_record(kv_shrink))
        geometry_override_reason = geometry_override_reason or ADJUSTMENT_KV
        launch = with_trailing(compose(geom.max_num_seqs))
        server = start(launch)
    proxy_loop = None
    try:
        att_path = out / vs.SERVING_ATTESTATION_FILE
        try:
            att = sa.read_serving_attestation(att_path)
        except sa.ServingAttestationError as e:
            raise StandupRefused(f"the readiness attestation at {att_path} is not a strict "
                                 f"lora_native record: {e}") from e
        try:
            att_geom = attested_scheduler_geometry(att)
        except ValueError as e:
            raise StandupRefused(f"served attestation is inconsistent: {e}") from e
        want_geom = (int(geom.max_num_seqs), int(geom.max_model_len))
        if att_geom is not None and att_geom != want_geom:
            raise StandupRefused(f"resolved geometry {want_geom[0]} x {want_geom[1]} but the "
                                 f"workers attest {att_geom[0]} x {att_geom[1]} — a silently "
                                 "clamped geometry; refusing")
        engine_id = sa.engine_identity(att)
        # --- the M-RoPE question (capctrl's --mrope-attest): NOT APPLICABLE on the native engine —
        # the four admitted families are neox-rotary and the GLM patch rides the lens plugin
        mrope_att: Optional[Dict[str, Any]] = None
        mrope_err = (f"{NOT_APPLICABLE}: the native engine serves the neox-rotary families only "
                     f"(family={family!r}); GLM's M-RoPE patch rides the lens plugin, which a plugin-free "
                     "launch keeps out (the spec's exclusion)" if attest_mrope
                     else "not asked (attest_mrope=False)")
        # --- THE NEMOTRON FUSED-KERNEL BACKEND over the PORTED RPC: MANDATORY for a
        # nemotron_h target, judged as capctrl judges it; a Nemotron standup without a PASSING
        # attestation REFUSES (the launcher's NEMO_BACKEND_UNATTESTED, at the standup's own boundary)
        mamba_att: Optional[Dict[str, Any]] = None
        mamba_err: Optional[str] = None
        if family == mb.NEMOTRON_FAMILY:
            if mamba_backend_contract is None:
                raise StandupRefused(
                    "nemotron_h target but no Nemotron backend contract was handed in "
                    "(--mamba-backend-contract; configs/capeval/mamba_backend_contract.json) — a Nemotron "
                    "standup never serves UNATTESTED on the native engine (the user's rule: never the quiet "
                    "pure-torch fallback)")
            try:
                import importlib.metadata as _md
                installed_vllm = _md.version("vllm")
            except Exception:  # noqa: BLE001 — the contract check then compares nothing
                installed_vllm = None
            cprobs = mb.contract_problems(mamba_backend_contract, installed_vllm_version=installed_vllm)
            if cprobs:
                raise StandupRefused("the Nemotron backend contract is not usable as a denominator:\n  - "
                                     + "\n  - ".join(cprobs))
            expected_layers = mb.expected_n_mamba_layers(mcfg)
            mamba_att = mb.live_attest(launch, mamba_backend_contract, expected_n_mamba_layers=expected_layers)
            print(f"{TAG} mamba backend attestation (RPC {mb.RPC_METHOD} on the native worker): "
                  f"ok={mamba_att['ok']} expected_n_mamba_layers={expected_layers} "
                  f"n_reports={len(mamba_att['reports'])} contract={mamba_att['contract_sha256'][:12]}", flush=True)
            if not mamba_att["ok"]:
                raise StandupRefused(
                    "Nemotron fused-kernel backend attestation FAILED — a Nemotron cell never runs on the "
                    "pure-torch fallback:\n  - " + "\n  - ".join(mamba_att["problems"]))
        else:
            mamba_err = f"not a {mb.NEMOTRON_FAMILY} target (model_type={family!r}): no Mamba backend to attest"
        # --- the ADDITIVE attestation blocks (§1.2): the clamp the launch was
        # composed under and the Nemotron backend verdict, written beside the launcher's record and
        # re-read strictly — the strict reader checks its required keys and tolerates the additions
        extra_att: Dict[str, Any] = {}
        if launch.sampling_clamp_override is not None:
            extra_att["sampling_clamp_override"] = {
                **dict(launch.sampling_clamp_override),
                "mechanism": getattr(launch, "sampling_clamp_override_mechanism", None)}
        if mamba_att is not None:
            extra_att["mamba_backend"] = {k: mamba_att[k] for k in
                                          ("ok", "problems", "rpc_method", "contract_sha256",
                                           "expected_n_mamba_layers", "asked_at")}
            extra_att["mamba_backend"]["n_reports"] = len(mamba_att["reports"])
        if extra_att:
            sa.write_json_atomic(att_path, {**dict(att), **extra_att}, default=str)
            att = sa.read_serving_attestation(att_path)
        # THE RESIDUAL SLOT as served (native-psr-serving §1.5): per residual cell the attested
        # kind / payload fingerprint / mode; armed iff one is served — held to the served kinds
        residual_slot_summary = residual_slot_facts(att, served_kinds)
        try:
            proxy = RoutingProxy(
                table["routes"],
                HTTPBackend(launch.base_url, api_key=launch.api_key,
                            **st.backend_timeout_kwargs(upstream_timeout_s)),
                base_model=model, serving_mode="interleaved",
                prompted_prefixes=table["prompted_prefixes"],
                exact_replay_strategy=None,
                # the launch env PLUS the server-default sampling policy, which the proxy echoes on
                # /v1/models petri_proxy (proxy.SERVER_DEFAULT_POLICY_KEY) so a client reads it off the proxy itself
                launch_attestation={**dict(launch.env), SERVER_DEFAULT_POLICY_KEY: launch.server_default_policy},
                agent_loop_render=TokenizerAgentLoopRender(tok),
                petri_agent_loop_entry=bool(petri_agent_loop),
                base_revision=rev,
                resumed_prefill_policy_cells=None,
                # a native table has no lens route: the attested EMPTY declarations map is the
                # stated "no declared cell" a thinking-ON proxy stands on (declared_cells: [])
                lens_serve_declarations=({} if thinking_on else None),
                thinking_header=table.get("thinking_header"),
                thinking_header_error=table.get(THINKING_HEADER_ERROR_KEY),
                lens_served_positions=table.get("lens_served_positions"),
                object_artifact_sha256_by_cell=dict(objmap),
                adapter_blocks=table[ADAPTERS_KEY],
                engine_attestation=engine_id,
                thinking_mode=thinking_mode)
        except ProxyError as e:
            raise StandupRefused(f"the routing proxy refused to start: {e}") from e
        proxy_port = int(proxy_port) if proxy_port else free_port()
        proxy_loop = _start_proxy_thread(make_app(proxy), "127.0.0.1", proxy_port)
        su = LoraStandup(
            out=out, model=model, revision=rev, model_key=model_key, eval=eval_name,
            geometry=geom, launch=launch, server=server, proxy=proxy, proxy_loop=proxy_loop,
            proxy_url=f"http://127.0.0.1:{proxy_port}/v1", proxy_port=proxy_port,
            vllm_port=vllm_port, table_path=table_path, table=table, tokenizer=tok,
            manifest_path=eff.manifest_path, config_path=eff.config_path, config_used=eff.config_used,
            defenses_dir=eff.defenses_dir, adapter_bank=bank, cells=tuple(eff.traintest_cells),
            harness_cells=tuple(eff.served_cells), lora_cells=tuple(c.cell_id for c in lora_cells),
            prompted_cells_admitted=tuple(eff.prompted_cells_admitted), vmodels=vmodels,
            cells_requested_explicit=tuple(eff.cells_requested_explicit),
            cells_requested_explicit_harness=tuple(eff.cells_requested_explicit_harness),
            adapters=blocks, zero_adapter=zero_block, adapters_reused=reused,
            object_artifact_sha256_by_cell=dict(objmap), attestation_path=att_path,
            attestation=att, engine_identity=engine_id, chat_template_hash=tsha16,
            chat_template_sha256=tsha256, manifest_flags_dropped=tuple(eff.manifest_flags_dropped),
            geometry_override_reason=geometry_override_reason, tool_calls=tool_calls,
            petri_agent_loop=bool(petri_agent_loop),
            upstream_timeout_s=(float(upstream_timeout_s) if upstream_timeout_s is not None else None),
            probe_max_tokens=int(probe_max_tokens), max_logprobs=max_logprobs,
            gpu_mem_util=str(gpu_mem_util), cell_set=cell_set, thinking_header=thinking_header,
            thinking_header_error=thinking_header_error, attempt_id=attempt_id, mode=mode,
            kernel_ceiling=kernel_ceiling, geometry_lowered=geometry_lowered, adapter_parity_gate=adapter_parity_gate,
            cells_skipped=tuple(eff.cells_skipped), geometry_adjustments=tuple(adjustments),
            kv_shrink=kv_shrink, thinking_mode=thinking_mode, thinking_certification=thinking_cert,
            reasoning_parser=reasoning_parser,
            sampling_clamp_override=(dict(launch.sampling_clamp_override)
                                     if launch.sampling_clamp_override is not None else None),
            sampling_clamp_override_mechanism=getattr(launch, "sampling_clamp_override_mechanism", None),
            text_only_mm=tom, enable_log_requests=bool(enable_log_requests),
            server_default_policy=launch.server_default_policy,
            mrope_attestation=mrope_att, mrope_attestation_error=mrope_err,
            mamba_backend_attestation=mamba_att, mamba_backend_attestation_error=mamba_err,
            expected_digests_source=eff.expected_digests_source, front=front_name,
            extra_facts=extra_facts, output_diagnostics=output_diagnostics,
            residual_slot=residual_slot_summary)
        print(f"{TAG} attested: enforce_eager={att['engine']['enforce_eager']} "
              f"cudagraph_mode={att['engine']['cudagraph_mode']} lora_config={att['lora_config']} "
              f"id_name_map={att['id_name_map']} residual_slot={residual_slot_summary}; "
              f"proxy at {su.proxy_url}", flush=True)
        if run_controls:
            su.positive_controls = run_controls_fn(su, output_diagnostics=output_diagnostics)
            print(f"{TAG} positive controls: {controls_verdict_line(su.positive_controls)}", flush=True)
            # THE POST-CONTROLS LEDGER READ (native-psr-serving §1.8, fact 15): the controls
            # claimed every probe on every restricted-mode route, so the worker's per-request
            # failure ledger must be EMPTY and unlatched now — a leftover entry means some
            # route the worker ledgers is served without the proxy's claim; REFUSE, never publish
            su.residual_ledger_after_controls = residual_ledger_after_controls(su)
            if su.residual_ledger_after_controls.get("problems"):
                su.write_cells_served()          # the refusal's evidence stays on disk
                raise StandupRefused(
                    "the worker's per-request failure ledger is not clean after the positive "
                    "controls — the proxy post-check is not claiming what the worker ledgers, so "
                    "an unlocated response boundary could be served as the cell; refusing to publish:"
                    "\n- " + "\n- ".join(su.residual_ledger_after_controls["problems"]))
            print(f"{TAG} residual ledger after controls: {su.residual_ledger_after_controls}", flush=True)
            su.write_cells_served()
            su.write_endpoint()
        else:
            su.write_cells_served()
    except BaseException:
        _stop_stack(proxy_loop, server)
        unpublish_endpoint(out)
        raise
    return su


def wait_until_stop(stop_file: Path, server: Any, *, poll_s: float = 5.0) -> int:
    """Block until ``stop_file`` appears; return 0 then, or 3 when the vLLM
    child exits first. A seam the CPU tests replace."""
    while not stop_file.exists():
        proc = server if hasattr(server, "poll") else (getattr(server, "proc", None)
                                                      or getattr(server, "process", None))
        rc = proc.poll() if proc is not None else None
        if rc is not None:
            print(f"{TAG} FATAL: vLLM server died while serving (rc={rc}) — see "
                  f"{SERVER_LOG_FILE}", flush=True)
            return EXIT_DEAD_ENGINE
        time.sleep(poll_s)
    print(f"{TAG} stop requested; tearing down", flush=True)
    return 0


__all__ = [
    "ADJUSTMENT_KERNEL", "ADJUSTMENT_KV", "ADAPTER_BANK_ROOT_ENV", "CELLS_SERVED_FILE",
    "CHAT_TEMPLATE_FILE", "CONTROL_NAMES", "DEFAULT_STANDUP_SERVING_PATH", "DERIVED_CONFIG_FILE",
    "DIGESTS_SOURCE_CLI", "DIGESTS_SOURCE_VIEW_PINS", "ENDPOINT_ATTEMPT_KEY", "ENDPOINT_FILE", "ENGINE",
    "ENGINE_EXPECTATIONS", "EXIT_CONTROLS_FAILED", "EXIT_DEAD_ENGINE", "EXIT_REFUSED",
    "EffectiveServing", "KV_SHORTFALL_LOG", "LoraStandup", "LoraStandupError", "NATIVE_DISPOSITIONS",
    "NOT_APPLICABLE", "POSITIVE_CONTROLS_FILE", "PositiveControlsFailed", "REDUCTION_MODE_NOT_APPLICABLE",
    "ROUTING_TABLE_FILE", "SERVER_LOG_FILE", "SERVING_PATHS", "STALE_DIR_PREFIX", "STRIPPED_MANIFEST_FLAGS",
    "SkippedCell", "StandupRefused", "TAG", "THINKING_SERVABLE_VERDICTS",
    "attested_scheduler_geometry", "bring_up_native", "certify_thinking", "default_adapter_bank",
    "effective_launch_flags", "endpoint_is_native", "free_port", "write_chat_template",
    "kernel_adjustment_record",
    "kv_adjustment_record", "kv_derived_geometry", "legacy_object_sha256",
    "lora_kernel_ceiling_geometry", "genuine_adapter_pairs", "default_parity_record_check", "PARITY_GATE_OPT_OUT",
    "native_serving_disposition", "partition_cells_for_native",
    "output_diagnostics_policy",
    "probe_prompt_set", "probe_set_hash", "quarantine_prior_endpoint", "resolve_adapter_bank",
    "residual_slot_facts",
    "run_positive_controls", "served_engine_problem", "shared_root", "thinking_header_fact",
    "unpublish_endpoint", "wait_until_stop",
    # the post-controls ledger read + the bites statistics
    "BITES_STATISTIC_LOGPROB", "BITES_STATISTIC_TEXT", "NATIVE_POSITION_MODES",
    "RESIDUAL_LEDGER_AFTER_CONTROLS_KEY", "ledger_problems", "logprob_liveness_block",
    "logprob_liveness_cells", "residual_ledger_after_controls", "resolve_output_diagnostics",
    "text_divergence_report", "unclaimed_failures_problems",
]
