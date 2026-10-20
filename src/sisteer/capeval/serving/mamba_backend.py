"""The NEMOTRON FUSED-KERNEL BACKEND ATTESTATION — runtime discovery compared against a REVIEWED
reference contract.

THE RULE: a Nemotron cell must NEVER run on transformers'/vLLM's pure-torch
SSM fallback, and the fallback is QUIET — a slow job reads as a healthy job. So the decision is made
from an ATTESTATION of the backend that actually serves, never from a missed walltime, and the
attestation is FAIL-CLOSED: anything it cannot prove is a refusal by name.

THE DENOMINATOR is ``configs/capeval/mamba_backend_contract.json`` — a checked-in, human-reviewed
document authored ONCE from the pinned vLLM wheel's SOURCE by ``scripts/mamba_backend_contract_author.py``
and keyed by that wheel's sha256. It names, per ROLE a Mamba2 mixer's forward path dispatches through
(``prefill_conv`` / ``prefill_scan`` / ``decode_conv`` / ``decode_scan``), the exact entry point the
mixer binds and the implementation modules approved for it. The runtime NEVER reads its required set
from the module it is testing (the cluster probe's ``all(...)`` over PRESENT symbols let an absent
binding drop out of the denominator — P1-D; rev 2's "derive from the installed module" made the
implementation its own denominator). Three discoveries are compared against the contract:

  static   ``static_discovery`` — import the model and mixer modules in the vLLM venv, resolve each
           role's binding by exact ``module.qualname``: a name absent from the mixer namespace is
           ``missing``, a same-named callable from another module is ``substituted``, a resolvable
           implementation outside ``approved_impls`` is ``unrecognised`` — each REFUSES.
  live     ``live_backend_report`` — the WORKER-side walk (``lens_worker.SisteerWorkerExtension.
           sisteer_mamba_backend_report``, MANDATORY, asked at EVERY Nemotron standup by
           ``capctrl_serving_up``): every decoder layer whose mixer is the contract's mixer class,
           each role resolved through THAT mixer's module globals, plus the initialised
           selective-state-update dispatch backend and its kernel. Admission needs every worker to
           answer, ``n_mamba_layers`` == the contract's count rule on the pinned config (21 for
           Nemotron-3-Nano-4B), and every layer x role equal to the contract.
  duty     the SAME-GPU PEER RULE (ported from ``scripts/sdfbase/math500/nemo_backend_probe.py``):
           a 64 x 512 decode burst on Nemotron, then the identical burst on a same-size hybrid peer
           (qwen35_4b) in the SAME container at Nemotron's S — admit iff Nemotron's SM duty AND tok/s
           are each >= :data:`DUTY_RATIO` x the peer's (``F22_NEMO_DUTY_RATIO``). The served path runs
           EAGER (vllm-lens forces --enforce-eager), so an absolute floor calibrates nothing; a
           pure-torch path is 9-60x slower, so half the peer's rate still catches it.

``attestation_document`` folds the three into ``mamba_backend_attestation.json`` with ``admitted``;
``readmission_problems`` RE-DERIVES admission from the recorded numbers (the reader shares the
writer's rule — the cluster's ``f22_model_facts`` discipline), keyed by :func:`attestation_cache_key`
(image specs + wheel sha + contract sha + staged tree sha + model@rev + serving config + L x S +
reduction mode — PER GEOMETRY, P1-E: the ladder's 17,664 x 44 needs its own record).

CPU-tested end to end on synthetic modules and reports; the GPU-only pieces (the RPC transport, the
dmon sampler, the burst) are thin and marked.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

#: the reviewed reference contract, repo-relative
CONTRACT_RELPATH = "configs/capeval/mamba_backend_contract.json"
CONTRACT_SCHEMA = "sisteer.mamba_backend_contract/1"
#: the worker RPC (``lens_worker.SisteerWorkerExtension.sisteer_mamba_backend_report``)
RPC_METHOD = "sisteer_mamba_backend_report"
ROLES = ("prefill_conv", "prefill_scan", "decode_conv", "decode_scan")
#: the role whose entry point is a dispatch wrapper (its kernel is the initialised backend's)
DISPATCH_ROLE = "decode_scan"
#: the deviation kinds a discovery can be refused for, by name
KIND_MISSING, KIND_SUBSTITUTED, KIND_UNRECOGNISED = "missing", "substituted", "unrecognised"
#: admission: Nemotron's duty AND tok/s >= this fraction of the same-GPU peer's
DUTY_RATIO = 0.5
DUTY_RATIO_ENV = "F22_NEMO_DUTY_RATIO"
ATTESTATION_FILENAME = "mamba_backend_attestation.json"
#: the model family the attestation applies to (HF config ``model_type``)
NEMOTRON_FAMILY = "nemotron_h"
#: the Mamba block marker in ``hybrid_override_pattern`` (M = Mamba2, - = MLP, * = attention, E = MoE)
MAMBA_PATTERN_CHAR = "M"
#: the standup's durable copy of the live report (endpoint.json is unlinked at STOP)
LIVE_REPORT_FILENAME = "mamba_backend_live_report.json"


class MambaBackendError(RuntimeError):
    """A refusal of the Nemotron backend attestation (named reasons in ``problems``)."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("Nemotron backend attestation REFUSED:\n  - " + "\n  - ".join(self.problems))


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def duty_ratio() -> float:
    """:data:`DUTY_RATIO`, overridable by ``F22_NEMO_DUTY_RATIO`` (the cluster's knob, same name)."""
    raw = os.environ.get(DUTY_RATIO_ENV, "").strip()
    if not raw:
        return DUTY_RATIO
    try:
        v = float(raw)
    except ValueError as e:
        raise ValueError(f"{DUTY_RATIO_ENV}={raw!r} is not a number") from e
    if not (0.0 < v <= 1.0):
        raise ValueError(f"{DUTY_RATIO_ENV}={v} must be in (0, 1]")
    return v


# --- the contract ---------------------------------------------------------------------------------

def wheel_sha256_of(spec: str) -> str:
    """The ``#sha256=<hex>`` fragment of a wheel spec (the launcher's ``VLLM_WHEEL`` string)."""
    m = re.search(r"#sha256=([0-9a-fA-F]{64})$", spec.strip())
    if not m:
        raise ValueError(f"wheel spec carries no #sha256=<64 hex> fragment: {spec!r}")
    return m.group(1).lower()


def load_contract(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no Nemotron backend contract at {p} — author it from the pinned wheel "
                                "with scripts/mamba_backend_contract_author.py and have it reviewed")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise TypeError(f"{p}: the contract must be a JSON object")
    return doc


def contract_sha256(contract: Mapping[str, Any] | Path | str) -> str:
    """The contract's ONE identity: sha256 over its canonical JSON (sorted keys, compact
    separators) — of the mapping given, or of the mapping LOADED from the path given. Round 6
    (bcb-bfcl-200-widen): the path spelling used to hash the FILE BYTES while the standup
    attested the loaded mapping's canonical JSON, so the launcher's expected sha never equalled
    the attestation and every Nemotron standup read NEMO_BACKEND_UNATTESTED with an ``ok: true``
    attestation. A re-serialised file (whitespace, key order) now hashes the same; the raw bytes
    are :func:`contract_file_sha256`, which nothing compares to an attestation."""
    if isinstance(contract, (str, Path)):
        contract = load_contract(contract)
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def contract_file_sha256(path: Path | str) -> str:
    """sha256 over the contract FILE's raw bytes — a provenance stamp of the file as staged,
    never the identity the attestation is compared against (:func:`contract_sha256`)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract_problems(contract: Mapping[str, Any], *, wheel_sha256: Optional[str] = None,
                      installed_vllm_version: Optional[str] = None) -> List[str]:
    """Every reason the contract may NOT be used as a denominator: an unknown schema, an incomplete
    roles table, a role without an entry point / approved implementations, an unreviewed contract
    (any of ``reviewed_by`` / ``reviewed_on`` / ``review_ref`` empty), a wheel hash other than the
    image's pin, or a vLLM version other than the one installed where the discovery runs."""
    problems: List[str] = []
    if contract.get("schema") != CONTRACT_SCHEMA:
        problems.append(f"contract schema {contract.get('schema')!r} != {CONTRACT_SCHEMA!r}")
    roles = contract.get("roles")
    if not isinstance(roles, Mapping):
        problems.append("contract has no roles table")
        roles = {}
    for role in ROLES:
        rec = roles.get(role)
        if not isinstance(rec, Mapping):
            problems.append(f"contract lacks role {role!r}")
            continue
        ep = rec.get("entry_point")
        if not isinstance(ep, str) or "." not in ep:
            problems.append(f"role {role}: entry_point {ep!r} is not a dotted module.qualname path")
        if not isinstance(rec.get("approved_impls"), list) or not rec["approved_impls"]:
            problems.append(f"role {role}: approved_impls is empty")
        if role == DISPATCH_ROLE and not isinstance(rec.get("dispatch"), Mapping):
            problems.append(f"role {role}: the dispatch role must record its registry (dispatch block)")
    extra = sorted(set(roles) - set(ROLES))
    if extra:
        problems.append(f"contract names roles outside the four: {extra}")
    for f in ("model_module", "mixer_module", "mixer_class", "vllm_wheel_sha256", "vllm_version"):
        if not str(contract.get(f) or "").strip():
            problems.append(f"contract field {f!r} is empty")
    if contract.get("nemotron_h_uses_mixer") is not True:
        problems.append("contract records nemotron_h_uses_mixer != true — the model module does not bind the "
                        "mixer class; the roles table would not describe the served model")
    for f in ("reviewed_by", "reviewed_on", "review_ref"):
        if not str(contract.get(f) or "").strip():
            problems.append(f"contract is UNREVIEWED: {f!r} is empty (a human diffs the draft against the "
                            "wheel's source and signs it; see scripts/mamba_backend_contract_author.py)")
    if wheel_sha256 is not None and str(contract.get("vllm_wheel_sha256", "")).lower() != wheel_sha256.lower():
        problems.append(f"contract is for vLLM wheel sha256 {str(contract.get('vllm_wheel_sha256'))[:12]}… but the "
                        f"image pins {wheel_sha256[:12]}… — re-draft and re-review the contract for this wheel")
    if installed_vllm_version is not None and str(contract.get("vllm_version")) != str(installed_vllm_version):
        problems.append(f"contract is for vLLM {contract.get('vllm_version')!r} but the installed vLLM is "
                        f"{installed_vllm_version!r}")
    return problems


def role_entry_points(contract: Mapping[str, Any]) -> Dict[str, str]:
    return {role: str(contract["roles"][role]["entry_point"]) for role in ROLES}


def role_binding_names(contract: Mapping[str, Any]) -> Dict[str, str]:
    """``role -> the NAME the mixer module binds`` (the entry point's last component)."""
    return {role: ep.rsplit(".", 1)[1] for role, ep in role_entry_points(contract).items()}


def expected_n_mamba_layers(config_or_pattern: Any) -> int:
    """The contract's count rule on the pinned HF config: the number of ``M`` blocks in
    ``hybrid_override_pattern`` (a config object, a mapping, or the pattern string itself)."""
    if isinstance(config_or_pattern, str):
        pattern = config_or_pattern
    elif isinstance(config_or_pattern, Mapping):
        pattern = config_or_pattern.get("hybrid_override_pattern")
    else:
        pattern = getattr(config_or_pattern, "hybrid_override_pattern", None)
    if not isinstance(pattern, str) or not pattern:
        raise TypeError("the config carries no hybrid_override_pattern — not a NemotronH config, or an "
                        "unresolved one; the Mamba layer count cannot be derived")
    return pattern.count(MAMBA_PATTERN_CHAR)


# --- static discovery (the vLLM venv, no engine) -------------------------------------------------

def _describe(obj: Any) -> Dict[str, Any]:
    return {"module": getattr(obj, "__module__", None), "qualname": getattr(obj, "__qualname__", None)}


def classify_binding(role: str, bound: Any, contract: Mapping[str, Any], *,
                     impl: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """The deviation KIND of one resolved binding against the contract, or None when it conforms.

    ``bound`` is whatever the mixer's namespace holds under the role's binding name (None = absent);
    ``impl`` is, for the dispatch role, the initialised backend's kernel ``{module, qualname}``
    (None = the dispatch has not been initialised / was not asked — refused for a live report)."""
    rec = contract["roles"][role]
    entry_module, entry_name = str(rec["entry_point"]).rsplit(".", 1)
    if bound is None:
        return KIND_MISSING
    got = _describe(bound)
    if got["module"] != entry_module or got["qualname"] != entry_name:
        return KIND_SUBSTITUTED
    approved = set(rec.get("approved_impls") or ())
    if role == DISPATCH_ROLE:
        if impl is None or not isinstance(impl, Mapping) or impl.get("error"):
            return KIND_UNRECOGNISED
        if impl.get("module") not in approved:
            return KIND_UNRECOGNISED
        return None
    if got["module"] not in approved:
        return KIND_UNRECOGNISED
    return None


def _deviation(role: str, kind: str, bound: Any, contract: Mapping[str, Any], where: str,
               impl: Optional[Mapping[str, Any]] = None) -> str:
    ep = contract["roles"][role]["entry_point"]
    if kind == KIND_MISSING:
        return f"{where}: role {role}: binding {ep.rsplit('.', 1)[1]!r} is MISSING from the mixer namespace (contract: {ep})"
    got = _describe(bound)
    if kind == KIND_SUBSTITUTED:
        return (f"{where}: role {role}: SUBSTITUTED — the mixer binds {got['module']}.{got['qualname']}, "
                f"the contract names {ep}")
    approved = contract["roles"][role].get("approved_impls")
    if role == DISPATCH_ROLE:
        return (f"{where}: role {role}: UNRECOGNISED dispatch implementation {impl!r} — approved: {approved}")
    return f"{where}: role {role}: UNRECOGNISED implementation module {got['module']!r} — approved: {approved}"


def static_discovery(contract: Mapping[str, Any], *,
                     import_module: Callable[[str], Any] = importlib.import_module) -> Dict[str, Any]:
    """Import the contract's model + mixer modules and resolve every role's binding — a record, never
    a verdict (:func:`static_problems` judges it). ``import_module`` is injectable for tests."""
    rec: Dict[str, Any] = {"model_module": None, "mixer_module": None, "mixer_class": None,
                           "nemotron_h_uses_mixer": None, "bindings": {}, "dispatch_registry": None,
                           "vllm_version": None, "errors": []}
    try:
        import importlib.metadata as md
        rec["vllm_version"] = md.version("vllm")
    except Exception as e:  # noqa: BLE001 — recorded; the comparison refuses a None
        rec["errors"].append(f"vllm version unreadable: {type(e).__name__}: {e}")
    try:
        model_mod = import_module(str(contract["model_module"]))
        rec["model_module"] = getattr(model_mod, "__file__", None)
    except Exception as e:  # noqa: BLE001
        rec["errors"].append(f"cannot import {contract['model_module']}: {type(e).__name__}: {e}")
        model_mod = None
    try:
        mixer_mod = import_module(str(contract["mixer_module"]))
        rec["mixer_module"] = getattr(mixer_mod, "__file__", None)
    except Exception as e:  # noqa: BLE001
        rec["errors"].append(f"cannot import {contract['mixer_module']}: {type(e).__name__}: {e}")
        return rec
    cls_name = str(contract["mixer_class"]).rsplit(".", 1)[1]
    mixer_cls = getattr(mixer_mod, cls_name, None)
    if mixer_cls is not None:
        rec["mixer_class"] = f"{getattr(mixer_cls, '__module__', None)}.{getattr(mixer_cls, '__qualname__', None)}"
    if model_mod is not None:
        # the model module binds the mixer CLASS OBJECT (identity, not a text grep)
        rec["nemotron_h_uses_mixer"] = bool(mixer_cls is not None
                                            and getattr(model_mod, cls_name, None) is mixer_cls)
    ns = vars(mixer_mod)
    for role, name in role_binding_names(contract).items():
        bound = ns.get(name)
        rec["bindings"][role] = ({"present": False} if bound is None
                                 else {"present": True, **_describe(bound)})
    try:
        disp = import_module(str(contract.get("dispatch_module") or
                                 contract["roles"][DISPATCH_ROLE]["entry_point"].rsplit(".", 1)[0]))
        registry = getattr(disp, "_BACKEND_REGISTRY", None)
        if isinstance(registry, Mapping):
            rec["dispatch_registry"] = {
                str(getattr(k, "name", k)): f"{getattr(v, '__module__', None)}.{getattr(v, '__qualname__', None)}"
                for k, v in registry.items()}
    except Exception as e:  # noqa: BLE001
        rec["errors"].append(f"dispatch registry unreadable: {type(e).__name__}: {e}")
    return rec


def static_problems(discovery: Mapping[str, Any], contract: Mapping[str, Any]) -> List[str]:
    """Every deviation of a static discovery from the contract (an empty list = conformant)."""
    problems: List[str] = list(discovery.get("errors") or [])
    if discovery.get("mixer_module") is None and not problems:
        problems.append("static discovery did not import the mixer module")
    if discovery.get("mixer_class") != contract["mixer_class"]:
        problems.append(f"mixer class resolved to {discovery.get('mixer_class')!r}, contract: {contract['mixer_class']!r}")
    if discovery.get("nemotron_h_uses_mixer") is not True:
        problems.append("nemotron_h_uses_mixer is not True: the model module does not bind the contract's "
                        "mixer class object — the served model would not go through the attested mixer")
    if contract.get("vllm_version") and discovery.get("vllm_version") != contract.get("vllm_version"):
        problems.append(f"installed vLLM {discovery.get('vllm_version')!r} != the contract's "
                        f"{contract.get('vllm_version')!r}")
    bindings = discovery.get("bindings") or {}
    for role in ROLES:
        b = bindings.get(role)
        if not isinstance(b, Mapping) or not b.get("present"):
            problems.append(_deviation(role, KIND_MISSING, None, contract, "static"))
            continue
        entry_module, entry_name = str(contract["roles"][role]["entry_point"]).rsplit(".", 1)
        if b.get("module") != entry_module or b.get("qualname") != entry_name:
            problems.append(f"static: role {role}: SUBSTITUTED — the mixer binds {b.get('module')}.{b.get('qualname')}, "
                            f"the contract names {entry_module}.{entry_name}")
            continue
        if role != DISPATCH_ROLE and b.get("module") not in set(contract["roles"][role].get("approved_impls") or ()):
            problems.append(f"static: role {role}: UNRECOGNISED implementation module {b.get('module')!r} — "
                            f"approved: {contract['roles'][role].get('approved_impls')}")
    # the dispatch registry the wheel ships must be the one the contract enumerated (a new backend
    # class is a new kernel nobody reviewed)
    want = contract["roles"][DISPATCH_ROLE].get("dispatch", {}).get("backends") or {}
    got = discovery.get("dispatch_registry")
    if want and isinstance(got, Mapping):
        want_map = {k: v.get("class") for k, v in want.items()}
        if dict(got) != want_map:
            problems.append(f"static: dispatch registry {dict(got)} != the contract's {want_map}")
    return problems


def static_attest(contract: Mapping[str, Any], **kw) -> Dict[str, Any]:
    disc = static_discovery(contract, **kw)
    problems = static_problems(disc, contract)
    return {"ok": not problems, "problems": problems, "discovery": disc,
            "contract_sha256": contract_sha256(contract), "asked_at": _utc_now()}


# --- the live worker report (worker side; pure Python over the model tree) -----------------------

#: the mixer attribute NemotronH's decoder layers wrap their block in (Mamba2 / MLP / attention alike)
MIXER_ATTRS = ("mixer",)


def _decoder_layers(model: Any) -> List[Any]:
    lm = getattr(model, "language_model", model)
    lm = getattr(lm, "model", lm)
    layers = getattr(lm, "layers", None)
    if layers is None:
        raise AttributeError("the model has no decoder layers")
    return list(layers)


def live_backend_report(model: Any, *, role_names: Optional[Mapping[str, str]] = None,
                        mixer_class: Optional[str] = None,
                        modules: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The WORKER's per-layer backend report — facts, never a verdict (the standup judges them
    against the contract with :func:`live_problems`).

    Walks the live model's decoder layers; every layer whose ``mixer`` is an instance of the mixer
    class (``mixer_class``: the contract's dotted path; default = any class named ``MambaMixer2``)
    is reported with each role's binding RESOLVED THROUGH THAT MIXER'S MODULE GLOBALS
    (``sys.modules[type(mixer).__module__]``) as ``{module, qualname}`` — i.e. what the mixer's
    forward will actually call. The initialised selective-state-update dispatch backend and its
    kernel are reported once (``dispatch``). ``role_names`` defaults to the contract's binding names
    (:data:`DEFAULT_ROLE_NAMES`); ``modules`` is injectable (tests) and defaults to ``sys.modules``."""
    modules = sys.modules if modules is None else modules
    names = dict(role_names or DEFAULT_ROLE_NAMES)
    rep: Dict[str, Any] = {"rpc_method": RPC_METHOD, "n_layers": None, "n_mamba_layers": 0,
                           "layers": [], "mixer_module": None, "dispatch": None, "vllm_version": None}
    try:
        import importlib.metadata as md
        rep["vllm_version"] = md.version("vllm")
    except Exception as e:  # noqa: BLE001 — a report; the comparison refuses a None
        rep["vllm_version_error"] = f"{type(e).__name__}: {e}"
    try:
        layers = _decoder_layers(model)
        rep["n_layers"] = len(layers)
        for i, layer in enumerate(layers):
            for attr in MIXER_ATTRS:
                mixer = getattr(layer, attr, None)
                if mixer is None:
                    continue
                cls = type(mixer)
                path = f"{cls.__module__}.{cls.__qualname__}"
                is_mixer = (path == mixer_class) if mixer_class else (cls.__qualname__ == "MambaMixer2")
                if not is_mixer:
                    continue
                mod = modules.get(cls.__module__)
                ns = vars(mod) if mod is not None else {}
                roles = {}
                for role, name in names.items():
                    bound = ns.get(name)
                    roles[role] = ({"present": False} if bound is None
                                   else {"present": True, **_describe(bound)})
                rep["layers"].append({"layer_index": i, "attr": attr, "mixer_class": path, "roles": roles})
                rep["mixer_module"] = rep["mixer_module"] or cls.__module__
                rep["n_mamba_layers"] += 1
        rep["dispatch"] = dispatch_backend_report(modules=modules)
    except Exception as e:  # noqa: BLE001 — a report, never a gate; the standup refuses an error
        rep["error"] = f"{type(e).__name__}: {e}"
    return rep


#: the binding names the worker resolves (== the contract's; pinned equal by test)
DEFAULT_ROLE_NAMES: Mapping[str, str] = {
    "prefill_conv": "causal_conv1d_fn",
    "prefill_scan": "mamba_chunk_scan_combined_varlen",
    "decode_conv": "causal_conv1d_update",
    "decode_scan": "selective_state_update",
}
DISPATCH_MODULE_DEFAULT = "vllm.model_executor.layers.mamba.ops.ssu_dispatch"


def dispatch_backend_report(*, modules: Optional[Mapping[str, Any]] = None,
                            dispatch_module: str = DISPATCH_MODULE_DEFAULT) -> Dict[str, Any]:
    """The INITIALISED selective-state-update backend (``ssu_dispatch.get_mamba_ssu_backend()``)
    and the kernel it forwards to: ``{backend, backend_class, kernel: {module, qualname}}``, or
    ``{error}`` when the dispatch module is not loaded / the backend is not initialised (engine
    init calls ``initialize_mamba_ssu_backend`` from ``initialize_kv_cache``, before readiness)."""
    modules = sys.modules if modules is None else modules
    mod = modules.get(dispatch_module)
    if mod is None:
        return {"error": f"{dispatch_module} is not loaded in this worker"}
    try:
        backend = mod.get_mamba_ssu_backend()
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    kernel = getattr(backend, "_kernel", None)
    return {"backend": getattr(backend, "name", None),
            "backend_class": f"{type(backend).__module__}.{type(backend).__qualname__}",
            "kernel": _describe(kernel) if kernel is not None else {"error": "backend has no _kernel"}}


def live_problems(reports: Sequence[Any], contract: Mapping[str, Any], *,
                  expected_n_mamba_layers: int) -> List[str]:
    """Every reason the workers' live reports do NOT prove the served Mamba backend is the
    contract's: no report, an errored report, a layer count other than the contract's rule on the
    pinned config, a layer whose mixer is not the contract's class, and per layer x role the
    ``missing`` / ``substituted`` / ``unrecognised`` deviations (the dispatch role's kernel must be an
    approved implementation and must be INITIALISED)."""
    problems: List[str] = []
    if not reports:
        return [f"no {RPC_METHOD} report from any worker"]
    for w, rep in enumerate(reports):
        where = f"worker {w}"
        if not isinstance(rep, Mapping):
            problems.append(f"{where}: report is {type(rep).__name__}, not a mapping")
            continue
        if rep.get("error"):
            problems.append(f"{where}: backend report errored: {rep['error']}")
        if contract.get("vllm_version") and rep.get("vllm_version") != contract.get("vllm_version"):
            problems.append(f"{where}: reports vLLM {rep.get('vllm_version')!r} != the contract's "
                            f"{contract.get('vllm_version')!r}")
        n = rep.get("n_mamba_layers")
        if n != expected_n_mamba_layers:
            problems.append(f"{where}: {n!r} Mamba mixer layers reported, the pinned config has "
                            f"{expected_n_mamba_layers} (hybrid_override_pattern count of 'M')")
        layers = rep.get("layers") or []
        if len(layers) != n:
            problems.append(f"{where}: n_mamba_layers={n!r} but {len(layers)} layer records")
        disp = rep.get("dispatch")
        if not isinstance(disp, Mapping) or disp.get("error"):
            why = (disp or {}).get("error") if isinstance(disp, Mapping) else repr(disp)
            problems.append(f"{where}: selective-state-update dispatch backend not attested: {why}")
            impl = None
        else:
            impl = disp.get("kernel")
            want_backends = contract["roles"][DISPATCH_ROLE].get("dispatch", {}).get("backends") or {}
            approved_cls = {v.get("class") for v in want_backends.values() if v.get("approved")}
            if approved_cls and disp.get("backend_class") not in approved_cls:
                problems.append(f"{where}: dispatch backend class {disp.get('backend_class')!r} is not an "
                                f"approved backend {sorted(approved_cls)}")
        for lrec in layers:
            if not isinstance(lrec, Mapping):
                problems.append(f"{where}: layer record is {type(lrec).__name__}, not a mapping")
                continue
            li = lrec.get("layer_index")
            if lrec.get("mixer_class") != contract["mixer_class"]:
                problems.append(f"{where}: layer {li}: mixer class {lrec.get('mixer_class')!r} != "
                                f"{contract['mixer_class']!r}")
            roles = lrec.get("roles") or {}
            for role in ROLES:
                b = roles.get(role)
                if not isinstance(b, Mapping) or not b.get("present"):
                    problems.append(_deviation(role, KIND_MISSING, None, contract, f"{where} layer {li}"))
                    continue
                fake = type("_Bound", (), {"__module__": b.get("module"), "__qualname__": b.get("qualname")})
                kind = classify_binding(role, fake, contract, impl=impl)
                if kind is not None:
                    problems.append(_deviation(role, kind, fake, contract, f"{where} layer {li}", impl=impl))
    return problems


def live_attest(launch: Any, contract: Mapping[str, Any], *, expected_n_mamba_layers: int,
                timeout_s: float = 120.0) -> Dict[str, Any]:
    """Ask every worker for its backend report over the launch's authenticated collective_rpc route
    (the same route as the install / scheduler / mrope reports) and judge it against the contract.
    Returns ``{ok, problems, reports, rpc_method, contract_sha256, expected_n_mamba_layers, asked_at}``;
    a transport failure is a problem, never a pass."""
    from sisteer.capeval.serving import (
        vllm_server as vs,  # lazy: the worker imports this module
    )
    try:
        payload = vs._post_collective_rpc(launch, RPC_METHOD, [], timeout_s=timeout_s,
                                          what="Nemotron backend attestation")
        reports = payload.get("results") if isinstance(payload, dict) else None
        reports = list(reports) if isinstance(reports, list) else []
        problems = live_problems(reports, contract, expected_n_mamba_layers=expected_n_mamba_layers)
    except Exception as e:  # noqa: BLE001 — a failed RPC is a refusal with its reason
        reports, problems = [], [f"{RPC_METHOD} RPC failed: {type(e).__name__}: {e}"]
    return {"ok": not problems, "problems": problems, "reports": reports, "rpc_method": RPC_METHOD,
            "contract_sha256": contract_sha256(contract), "expected_n_mamba_layers": expected_n_mamba_layers,
            "asked_at": _utc_now()}


def endpoint_attestation_problems(endpoint: Mapping[str, Any], *,
                                  contract_sha256_expected: Optional[str] = None) -> List[str]:
    """The LAUNCHER's reading of a standup's ``endpoint.json``: the live attestation must be present,
    clean (``ok`` True, no problems, reports from >= 1 worker) and made against the expected
    contract. Anything else -> the launcher's ``NEMO_BACKEND_UNATTESTED`` refusal, named."""
    att = endpoint.get("mamba_backend_attestation")
    if not isinstance(att, Mapping):
        why = endpoint.get("mamba_backend_attestation_error")
        return ["endpoint.json carries no mamba_backend_attestation"
                + (f" ({why})" if why else "") + " — the standup did not attest the Nemotron backend"]
    problems: List[str] = []
    if att.get("ok") is not True:
        problems.append(f"the standup's backend attestation is not ok: {att.get('problems')}")
    if not att.get("reports"):
        problems.append("the standup's backend attestation carries no worker report")
    if att.get("rpc_method") != RPC_METHOD:
        problems.append(f"attestation rpc_method {att.get('rpc_method')!r} != {RPC_METHOD!r}")
    if contract_sha256_expected and att.get("contract_sha256") != contract_sha256_expected:
        problems.append(f"the standup attested against contract {str(att.get('contract_sha256'))[:12]}…, "
                        f"this launch's contract is {contract_sha256_expected[:12]}…")
    return problems


# --- packages (the rule's letter) -----------------------------------------------------------------

def packages_report(*, import_module: Callable[[str], Any] = importlib.import_module) -> Dict[str, Any]:
    """``mamba_ssm`` + ``causal_conv1d`` import with versions (the pip half of the rule; inert for
    the served path on this vLLM, required by the rule's letter and the HF path). CPU-callable."""
    rec: Dict[str, Any] = {"python": sys.executable, "ok": False}
    try:
        cc = import_module("causal_conv1d")
        ms = import_module("mamba_ssm")
        rec.update(mamba_ssm_version=getattr(ms, "__version__", None),
                   causal_conv1d_version=getattr(cc, "__version__", None),
                   mamba_ssm_file=getattr(ms, "__file__", None),
                   causal_conv1d_file=getattr(cc, "__file__", None))
        ssd = import_module("mamba_ssm.ops.triton.ssd_combined")
        ssu = import_module("mamba_ssm.ops.triton.selective_state_update")
        for mod, name in ((cc, "causal_conv1d_fn"), (cc, "causal_conv1d_update"),
                          (ssd, "mamba_chunk_scan_combined"), (ssu, "selective_state_update")):
            if not callable(getattr(mod, name, None)):
                raise ImportError(f"{getattr(mod, '__name__', mod)} has no callable {name}")
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    try:
        import importlib.metadata as md
        rec["vllm_version"] = md.version("vllm")
    except Exception as e:  # noqa: BLE001
        rec["vllm_error"] = f"{type(e).__name__}: {e}"
    return rec


# --- the same-GPU peer duty rule ------------------------------------------------------------

def _numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def duty_admission(duty: Mapping[str, Any], ref: Mapping[str, Any], *, ratio: float) -> Dict[str, Any]:
    """The peer rule on two duty records (the probe's ``duty.json`` shape): comparable iff both carry
    numeric positive duty + tok/s, the same burst (n_seqs, max_tokens), the same GPU index and the
    same served S when both record it; then ``duty_ok`` / ``tok_s_ok`` against ``ratio`` x the peer.
    Incomparable or missing -> never admitted, recorded by name."""
    d_duty, d_tok = duty.get("sm_duty_mean"), duty.get("tok_s")
    r_duty, r_tok = ref.get("sm_duty_mean"), ref.get("tok_s")
    reasons: List[str] = []
    if not all(_numeric(v) for v in (d_duty, d_tok, r_duty, r_tok)):
        reasons.append(f"non-numeric duty/tok_s: nemotron {d_duty!r}/{d_tok!r}, peer {r_duty!r}/{r_tok!r}")
    elif r_duty <= 0 or r_tok <= 0:
        reasons.append(f"peer numbers not positive: {r_duty} / {r_tok}")
    if duty.get("n_seqs") != ref.get("n_seqs") or duty.get("max_tokens") != ref.get("max_tokens"):
        reasons.append(f"bursts differ: nemotron {duty.get('n_seqs')}x{duty.get('max_tokens')} vs peer "
                       f"{ref.get('n_seqs')}x{ref.get('max_tokens')}")
    if str(duty.get("gpu_index")) != str(ref.get("gpu_index")):
        reasons.append(f"different GPUs: {duty.get('gpu_index')!r} vs {ref.get('gpu_index')!r}")
    s_d = (duty.get("served") or {}).get("max_num_seqs")
    s_r = (ref.get("served") or {}).get("max_num_seqs")
    if s_d is not None and s_r is not None and s_d != s_r:
        reasons.append(f"served S differs: nemotron {s_d} vs peer {s_r}")
    comparable = not reasons
    duty_floor = round(ratio * r_duty, 4) if comparable else None
    tok_floor = round(ratio * r_tok, 1) if comparable else None
    return {"duty_comparable": comparable, "incomparable_reasons": reasons, "duty_ratio": ratio,
            "duty_floor": duty_floor, "tok_s_floor": tok_floor,
            "duty_ok": bool(comparable and d_duty >= duty_floor),
            "tok_s_ok": bool(comparable and d_tok >= tok_floor),
            "sm_duty_mean": d_duty, "tok_s": d_tok,
            "peer_sm_duty_mean": r_duty, "peer_tok_s": r_tok}


# --- the cache key (per image x tree x model x config x geometry x reduction mode) ------------------

#: the engine a certificate may be keyed to. ``None`` / ``"lens"``:
#: the historical lens key (no ``serving_engine`` emitted — every certificate already on the volume keeps its
#: address); ``"lora_native"``: the key gains ``serving_engine`` and its ``reduction_mode`` MUST be the tinycap
#: driver's explicit "no lens reduction" value (``contracts.REDUCTION_MODE_NOT_APPLICABLE``, the literal restated
#: here so this stdlib-only leaf imports nothing of the driver)
SERVING_ENGINES = ("lens", "lora_native")
SERVING_ENGINE_NATIVE = "lora_native"
REDUCTION_MODE_NOT_APPLICABLE = "not_applicable"


def cache_key_inputs(*, image_specs: Sequence[str], wheel_sha256: str, contract_sha256: str, tree_sha: str,
                     model_id: str, model_revision: str, serving_config_sha256: str,
                     max_model_len: int, max_num_seqs: int, reduction_mode: str,
                     serving_engine: Optional[str] = None) -> Dict[str, Any]:
    """The certificate's KEY INPUTS (image x tree x model x config x geometry x reduction mode — and, on the native
    engine, the engine). ``serving_engine`` (keyword-only, default absent):
    ``None`` or ``"lens"`` returns BYTE-IDENTICALLY today's dict — no ``serving_engine`` key, ``reduction_mode``
    validated as before — so every historical lens key and every certificate on the volume keeps its address;
    ``"lora_native"`` adds ``"serving_engine": "lora_native"`` and REQUIRES ``reduction_mode ==
    "not_applicable"``. The pairing is validated both ways: ``lora_native`` with a lens mode, or
    ``not_applicable`` without the engine, raises ``ValueError`` (the same shape as the other input checks) — a
    native duty certificate can never share a lens one's address, nor the reverse."""
    for k, v in (("wheel_sha256", wheel_sha256), ("contract_sha256", contract_sha256), ("tree_sha", tree_sha),
                 ("model_id", model_id), ("model_revision", model_revision),
                 ("serving_config_sha256", serving_config_sha256), ("reduction_mode", reduction_mode)):
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"cache key input {k!r} must be a non-empty string, got {v!r}")
    for k, v in (("max_model_len", max_model_len), ("max_num_seqs", max_num_seqs)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"cache key input {k!r} must be a positive int, got {v!r}")
    if serving_engine is not None and serving_engine not in SERVING_ENGINES:
        raise ValueError(f"cache key input 'serving_engine' must be one of {SERVING_ENGINES} or None, "
                         f"got {serving_engine!r}")
    if serving_engine == SERVING_ENGINE_NATIVE and reduction_mode != REDUCTION_MODE_NOT_APPLICABLE:
        raise ValueError(f"cache key input 'reduction_mode' must be {REDUCTION_MODE_NOT_APPLICABLE!r} on the "
                         f"{SERVING_ENGINE_NATIVE!r} engine (no lens reduction is served there), got {reduction_mode!r}")
    if serving_engine != SERVING_ENGINE_NATIVE and reduction_mode == REDUCTION_MODE_NOT_APPLICABLE:
        raise ValueError(f"cache key input 'reduction_mode' {REDUCTION_MODE_NOT_APPLICABLE!r} names no lens reduction; "
                         f"it is the {SERVING_ENGINE_NATIVE!r} engine's value and needs serving_engine="
                         f"{SERVING_ENGINE_NATIVE!r} (got {serving_engine!r})")
    doc = {"image_specs": sorted(str(s) for s in image_specs), "wheel_sha256": wheel_sha256.lower(),
           "contract_sha256": contract_sha256.lower(), "tree_sha": tree_sha, "model_id": model_id,
           "model_revision": model_revision, "serving_config_sha256": serving_config_sha256.lower(),
           "geometry": {"max_model_len": int(max_model_len), "max_num_seqs": int(max_num_seqs)},
           "reduction_mode": reduction_mode}
    if serving_engine == SERVING_ENGINE_NATIVE:
        doc["serving_engine"] = SERVING_ENGINE_NATIVE
    return doc


def attestation_cache_key(**inputs: Any) -> str:
    """sha256 hex over the canonical JSON of :func:`cache_key_inputs` — the attestation's address
    under ``/vol/tinycap/nemo_backend/<key>/``. A rebuilt image, a re-staged tree, another model /
    config, another L x S (the ladder), another reduction mode — or the native engine — is another key."""
    doc = cache_key_inputs(**inputs)
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def attestation_path(root: Path | str, key: str) -> Path:
    return Path(root) / key / ATTESTATION_FILENAME


# --- the folded document and its re-derivation ---------------------------------------------------

def attestation_document(*, packages: Mapping[str, Any], static: Mapping[str, Any], live: Mapping[str, Any],
                         duty: Mapping[str, Any], peer_duty: Mapping[str, Any], peer_key: str, peer_image: str,
                         ratio: float, cache_key: str, key_inputs: Mapping[str, Any],
                         geometry: str = "wave", standups: Sequence[Mapping[str, Any]] = (),
                         extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Fold the probe's records into ``mamba_backend_attestation.json``. ADMISSION = packages import
    AND static discovery clean AND live report clean AND duty_ok AND tok_s_ok (same-GPU peer, identical
    bursts). Every input is recorded so :func:`readmission_problems` can re-derive the verdict."""
    adm = duty_admission(duty, peer_duty, ratio=ratio)
    contract_clean = bool(static.get("ok")) and bool(live.get("ok"))
    doc: Dict[str, Any] = {
        "schema": "sisteer.mamba_backend_attestation/2",
        "cache_key": cache_key, "key_inputs": dict(key_inputs), "geometry": geometry,
        "packages_ok": bool(packages.get("ok")),
        "contract_clean": contract_clean,
        "fused_backend_attested": bool(packages.get("ok")) and contract_clean,
        **{k: adm[k] for k in ("sm_duty_mean", "tok_s", "duty_ratio", "duty_floor", "tok_s_floor",
                                "duty_comparable", "incomparable_reasons", "duty_ok", "tok_s_ok")},
        "sm_duty_min": duty.get("sm_duty_min"),
        "duty_reference": {"model_key": peer_key, "image": peer_image,
                           **{k: peer_duty.get(k) for k in ("sm_duty_mean", "sm_duty_min", "tok_s", "n_seqs",
                                                             "max_tokens", "gpu_index", "burst_s",
                                                             "completion_tokens_total", "served", "error")}},
        "admission_rule": (f"packages import AND static + live discovery conform to the reviewed contract "
                           f"(missing / substituted / unrecognised each refuse; every Mamba layer reported) AND "
                           f"sm_duty_mean >= {ratio} x the same-GPU peer's ({peer_key}) AND tok_s >= {ratio} x the "
                           "peer's, both bursts identical (n_seqs, max_tokens, GPU, served S) in ONE container; the "
                           "served path runs eager, so an absolute duty floor calibrates nothing (D114)"),
        "packages": dict(packages), "static": dict(static), "live": dict(live),
        "duty_record": {k: v for k, v in duty.items() if k not in ("sm_duty_samples", "idle_samples")},
        "standups": [dict(s) for s in standups],
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "attestation_basis": "live worker RPC (sisteer_mamba_backend_report) + static discovery vs the reviewed contract",
    }
    if extra:
        doc.update(dict(extra))
    doc["admitted"] = bool(doc["fused_backend_attested"] and adm["duty_ok"] and adm["tok_s_ok"])
    return doc


def readmission_problems(doc: Mapping[str, Any], contract: Mapping[str, Any], *, ratio: float,
                         cache_key: Optional[str] = None,
                         expected_n_mamba_layers: Optional[int] = None) -> List[str]:
    """RE-DERIVE admission from the recorded numbers (never trust ``admitted`` alone): the key must be
    this launch's, the packages must have imported, the recorded static discovery and live reports must
    STILL conform to the contract (re-judged here), and the peer rule must hold at ``ratio``."""
    problems: List[str] = []
    if cache_key is not None and doc.get("cache_key") != cache_key:
        problems.append(f"attestation is for cache key {str(doc.get('cache_key'))[:12]}…, this launch needs "
                        f"{cache_key[:12]}… (another image / tree / model / config / geometry / reduction mode)")
    if doc.get("packages_ok") is not True or not (doc.get("packages") or {}).get("ok"):
        problems.append("the kernel packages did not import in the probe (packages_ok false)")
    static = doc.get("static") or {}
    if not isinstance(static.get("discovery"), Mapping):
        problems.append("no static discovery recorded")
    else:
        problems += [f"recorded static discovery: {p}" for p in static_problems(static["discovery"], contract)]
    live = doc.get("live") or {}
    n_exp = expected_n_mamba_layers if expected_n_mamba_layers is not None else live.get("expected_n_mamba_layers")
    if not isinstance(n_exp, int):
        problems.append("no expected Mamba layer count recorded for the live report")
    else:
        problems += [f"recorded live report: {p}"
                     for p in live_problems(live.get("reports") or [], contract, expected_n_mamba_layers=n_exp)]
    adm = duty_admission(doc.get("duty_record") or {}, doc.get("duty_reference") or {}, ratio=ratio)
    if not adm["duty_comparable"]:
        problems.append(f"no comparable same-GPU peer reference: {adm['incomparable_reasons']}")
    else:
        if not adm["duty_ok"]:
            problems.append(f"SM duty {adm['sm_duty_mean']} < {ratio} x the peer's {adm['peer_sm_duty_mean']}")
        if not adm["tok_s_ok"]:
            problems.append(f"tok/s {adm['tok_s']} < {ratio} x the peer's {adm['peer_tok_s']}")
    if doc.get("admitted") is not True:
        problems.append(f"the probe did not admit (admitted={doc.get('admitted')!r})")
    return problems


def read_attestation(path: Path | str, contract: Mapping[str, Any], *, ratio: float, cache_key: str,
                     expected_n_mamba_layers: Optional[int] = None) -> Dict[str, Any]:
    """The BOTH-DOORS reader (preflight + launcher): the file for THIS key must exist and re-derive
    admitted. Returns ``{ok, problems, path, doc}``; a missing file is a problem, never a pass."""
    p = Path(path)
    if not p.is_file():
        return {"ok": False, "path": str(p), "doc": None,
                "problems": [(f"no Nemotron backend attestation at {p} for cache key {cache_key[:12]}… — run the "
                              "backend probe at this geometry first (the fused kernels are a PREREQUISITE, never "
                              "inferred from a healthy-looking job)")]}
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "path": str(p), "doc": None, "problems": [f"unreadable attestation {p}: {e}"]}
    problems = readmission_problems(doc, contract, ratio=ratio, cache_key=cache_key,
                                    expected_n_mamba_layers=expected_n_mamba_layers)
    return {"ok": not problems, "problems": problems, "path": str(p), "doc": doc}


# --- the decode burst under a duty sampler (GPU-side; the folding is pure) ------------------------

def fold_duty_samples(samples: Sequence[float], completion_tokens: int, burst_s: float) -> Dict[str, Any]:
    """The duty record's numbers from raw SM samples (fractions in [0, 1]) and the burst's totals."""
    s = [float(x) for x in samples]
    return {"completion_tokens_total": int(completion_tokens), "burst_s": round(float(burst_s), 2),
            "tok_s": round(completion_tokens / burst_s, 1) if burst_s > 0 else None,
            "sm_duty_mean": round(sum(s) / len(s), 4) if s else None,
            "sm_duty_min": min(s) if s else None, "n_samples": len(s)}


def parse_dmon_line(line: str) -> Optional[float]:
    """``nvidia-smi dmon -s u`` row -> SM fraction, None for headers / blanks (columns: gpu sm mem …)."""
    parts = line.split()
    if not parts or parts[0].startswith("#"):
        return None
    try:
        return float(parts[1]) / 100.0
    except (IndexError, ValueError):
        return None


class DmonSampler:  # pragma: no cover — spawns nvidia-smi
    """``nvidia-smi dmon -s u -d 1`` on one GPU, parsed to SM fractions at 1 Hz."""

    name = "nvidia-smi dmon -s u -d 1"

    def __init__(self, gpu_index: str):
        self.gpu = str(gpu_index)
        self.samples: List[float] = []
        self._proc = None
        self._thread = None

    def __enter__(self):
        import subprocess
        import threading
        self._proc = subprocess.Popen(["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-i", self.gpu],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

        def pump():
            for line in self._proc.stdout:  # type: ignore[union-attr]
                v = parse_dmon_line(line)
                if v is not None:
                    self.samples.append(v)
        self._thread = threading.Thread(target=pump, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self._proc.kill()


class PynvmlSampler:  # pragma: no cover — needs the NVML library
    """``pynvml`` utilization polling at 1 Hz — the fallback when ``nvidia-smi`` is absent from the
    container (Modal's images carry the driver libraries, not always the CLI)."""

    name = "pynvml nvmlDeviceGetUtilizationRates @1Hz"

    def __init__(self, gpu_index: str):
        self.gpu = int(str(gpu_index).split(",")[0] or 0)
        self.samples: List[float] = []
        self.errors = 0
        self._stop = None
        self._thread = None

    def __enter__(self):
        import threading

        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu)
        self._stop = threading.Event()

        def pump():
            while not self._stop.is_set():
                try:
                    self.samples.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu / 100.0)
                except Exception:  # noqa: BLE001 — a dropped sample is counted, never fatal mid-burst
                    self.errors += 1
                self._stop.wait(1.0)
        self._thread = threading.Thread(target=pump, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


def pick_sampler(gpu_index: str):  # pragma: no cover — probes the container
    import shutil
    if shutil.which("nvidia-smi"):
        return DmonSampler(gpu_index)
    try:
        import pynvml  # noqa: F401
    except ImportError as e:
        raise RuntimeError("neither nvidia-smi nor pynvml is available — the duty burst cannot be sampled "
                           "(add nvidia-ml-py to the image)") from e
    return PynvmlSampler(gpu_index)


def duty_burst(endpoint: Mapping[str, Any], *, n_seqs: int = 64, max_tokens: int = 512,
               gpu_index: Optional[str] = None, sampler=None, timeout_s: float = 600.0,
               model: str = "petri__base") -> Dict[str, Any]:  # pragma: no cover — live traffic
    """64 concurrent chat completions x 512 tokens on the base route under the OFF render (a DECODE
    burst; the render is not the point), SM duty sampled at 1 Hz for the burst's duration. The
    record's shape is the cluster probe's ``duty.json`` (``duty_admission`` reads it)."""
    import concurrent.futures as cf
    import urllib.request

    url = str(endpoint["proxy_url"]).rstrip("/")
    token = endpoint.get("proxy_token")
    gpu = str(gpu_index if gpu_index is not None else (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0])
    sampler = sampler if sampler is not None else pick_sampler(gpu)
    prompts = [f"Write a detailed essay about the number {i} and its role in mathematics." for i in range(n_seqs)]

    def one(i: int):
        body = {"model": model, "messages": [{"role": "user", "content": prompts[i]}],
                "max_tokens": max_tokens, "temperature": 1.0, "seed": i,
                "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True}
        req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {token}"} if token else {})},
                                     method="POST")
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            r = json.loads(resp.read().decode())
        return (r.get("usage") or {}).get("completion_tokens", 0), time.time() - t0

    t0 = time.time()
    with sampler as smp:
        time.sleep(2)                                    # a clean idle baseline before the burst
        idle = list(smp.samples)
        with cf.ThreadPoolExecutor(max_workers=n_seqs) as ex:
            results = list(ex.map(one, range(n_seqs)))
        burst_s = time.time() - t0 - 2
        time.sleep(1)
    burst_samples = smp.samples[len(idle):]
    toks = sum(r[0] for r in results)
    rec = {"n_seqs": n_seqs, "max_tokens": max_tokens,
           "served": {k: endpoint.get(k) for k in ("model_key", "model", "max_model_len", "max_num_seqs",
                                                    "geometry", "lens_reduction_mode") if k in endpoint},
           **fold_duty_samples(burst_samples, toks, burst_s),
           "sm_duty_samples": burst_samples, "idle_samples": idle, "gpu_index": gpu,
           "sampler": getattr(smp, "name", type(smp).__name__)}
    return rec
