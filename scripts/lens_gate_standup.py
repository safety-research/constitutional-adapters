#!/usr/bin/env python
"""lens_gate_standup — ONE production standup, served from the tree on
PYTHONPATH, for scripts/lens_vectorize_gate.py.

WHY A SEPARATE PROCESS. The gate compares two IMPLEMENTATIONS of the lens
hook: the base commit's (the reference) and the candidate's. A single
process cannot import two versions of ``sisteer``, and ``vllm_server.
launch_server`` composes the worker's PYTHONPATH from the tree the launching
process imported ``sisteer`` from — so the only way the reference server
runs the base commit's worker extension, adapters and registry is to launch
it from a process whose ``sisteer`` IS the base checkout. The gate therefore
spawns this helper once per standup with ``PYTHONPATH=<tree>/src:<tree>/
scripts`` and reads back ``<out>/endpoint.json`` (proxy URL, server URL,
routing table, the tree's commit, the module files actually imported) plus
``<out>/server_auth.json`` (mode 0600: the per-launch API token the gate
needs for ``/collective_rpc`` and ``/metrics``; the old ``capctrl_serving_up
.main`` deliberately never writes it, which is why this helper exists
instead of driving ``main`` directly). ``--stop-file`` ends the standup.

The bring-up body is ``capctrl_serving_up.main``'s at the base commit,
verbatim in its calls (registry + routing-table export, the lens vLLM launch
with readiness attestation, the RoutingProxy) — written against APIs both
trees carry, so the SAME helper serves both. Prefix caching composes OFF
(``--prefix-caching off``, the default — every fidelity standup) or as the
production per-model default (``--prefix-caching default`` — the gate's
shared-prefill standup, 2026-09-06); the endpoint records the requested
regime next to the composed and attested states. It imports NOTHING from its own
directory (``sys.path[0]`` is dropped: this file lives in the candidate
tree) and REFUSES if the imported ``sisteer`` does not live under
``--repo-dir/src``, if the tree's HEAD is not ``--expect-sha``, or if a
tracked file is modified.

THE SERVED REDUCTION MODE (2026-09-04). ``endpoint.json`` records
``lens_reduction_mode`` — ``per_row`` | ``stacked``, the arithmetic the
workers ATTEST for grouped lens reductions (every ``scheduler_geometry.
worker_reports`` entry of the steering attestation, read strictly by
:func:`reduction_mode_attested` through ``lens_worker.
attested_reduction_mode``) — next to ``lens_reduction_mode_error`` (why it
is None: the base commit's workers attest no mode, or the workers disagree)
and ``stacked_reductions_env``, the raw ``SISTEER_LENS_STACKED_REDUCTIONS``
echo kept for provenance only. The attested mode is also handed to the
RoutingProxy, which echoes it in its ``/v1/models`` identity block; an
unattested mode is passed as nothing, never guessed — a consumer that finds
no mode treats the server as per-row, which is what an unattesting tree
serves.

DECLARED-RENDER STANDUPS (fit-support-transfer, 2026-09-07; the gate's
``declared_thinking`` / ``all_positions_carveout`` / ``all_turns_render_check``
cases). A ``--cell`` may name a serve VARIANT (``<cell>@thinking`` /
``@all_positions`` / ``@all_turns``, ``serve_declaration.SERVE_VARIANTS``);
the helper then mirrors ``capctrl_serving_up.bring_up`` exactly: it derives
the two OPTIONAL registry markers iff a served lens cell needs them
(:func:`registry_markers` — the thinking-ON generation header for a cell
declaring the thinking render, recorded EQUAL to the OFF header on a template
whose switch does not move it (gemma-4; the user's ruling 2026-09-10, withdrawing plan
P9's refusal); the assistant turn-end scaffold for a
``response_all_turns`` cell, refused on a template whose turns have no
locatable end), derives the served template's ``thinking_header`` fact with
``capctrl_serving_up.standup_thinking_header``'s discipline (code review X1 /
S-G3: HARD through ``chat_markers.thinking_header_facts`` when a served cell
DECLARES the thinking render — its ON header must certify — and NON-FATAL
through ``thinking_header_facts_or_error`` otherwise, so a thinking-OFF
standup on a template whose ON header cannot certify — Nemotron-3 — keeps the
PARTIAL fact ``{switch, off_ids, on_ids: null, moves: null}`` with the
``ChatMarkerError`` reason beside it as ``thinking_header_error``; only
``ChatMarkerError`` is ever caught, a defect propagates), records the fact and
the reason in the routing table, binds every fitted lens cell's ARTIFACT
BYTES to the digest of the enumerated object (:func:`artifact_digests`;
``object_artifact_sha256_by_cell`` + the legacy scalar in the endpoint),
reads the workers' attested ``lens_serve_declarations``
(:func:`serve_declarations_attested`) and hands the map, the header fact and
its reason, the per-cell digests and ``--lens-salt-policy`` to the
RoutingProxy. Every one of those is passed ONLY when the SERVED tree's API
takes it (:func:`accepted_kwargs` — the cross-tree guard
``resumed_prefill_proxy_kwargs`` already applies, generalised: the reference
standup runs this script against the base commit's ``sisteer``), and a
declared cell on a tree that cannot serve it REFUSES rather than standing up
undeclared. On a DECLARED-RENDER standup (``--declared-render``, or any
served cell spelled ``<cell>@<variant>``; :func:`is_declared_render_standup`)
the artifact digest binding at export AND its publication by the proxy are
REQUIRED: a served tree that drops ``expected_artifact_sha256`` or
``object_artifact_sha256_by_cell`` REFUSES before any launch, naming the
kwarg (:func:`declared_render_refusal`; code review S-G2 — a standup whose
endpoint carried the helper's LOCAL digests while the served proxy published
none certified an identity nobody could read back). The endpoint publishes
what was actually handed over, never what this helper wished to hand over
(``lens_serve_declarations`` / ``_error``, ``thinking_header`` / ``_error``,
``lens_salt_policy`` / ``_error``, ``object_artifact_sha256_by_cell`` /
``object_artifact_sha256`` / ``_bound`` / ``_error`` — the map, the scalar
and ``_bound: true`` ONLY when the proxy received the map, else ``null`` /
``null`` / ``false`` with the reason —, ``registry_markers``,
``declared_render``).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import dataclasses
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


def _drop_own_dir() -> None:
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p and str(Path(p).resolve()) != here]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                          text=True, timeout=60).stdout.strip()


def verify_tree(repo: Path, expect_sha: str) -> str:
    head = _git(repo, "rev-parse", "HEAD")
    if not head:
        raise SystemExit(f"[standup] {repo} is not a git checkout; refusing")
    if head != expect_sha:
        raise SystemExit(f"[standup] {repo} is at {head}, expected {expect_sha}; refusing")
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=str(repo)).returncode
    if dirty != 0:
        raise SystemExit(f"[standup] {repo} has modified tracked files; the gate serves "
                         "immutable checkouts only; refusing")
    return head


def harness_cells(cells, harness_cell_id_fn) -> list:
    """``["none", <harness id>, ...]`` for the TrainTest cells to serve from one
    registry: the baseline first, then each cell's harness translation
    (loud on an untranslatable id), duplicates dropped, order kept — the
    ``cells`` list ``enumerate_cells`` builds the registry and routing table
    from, so several distinct objects serve at once (the gate's multigroup
    case)."""
    return ["none"] + [h for h in cell_map(cells, harness_cell_id_fn).values()
                       if h != "none"]


def cell_map(cells, harness_cell_id_fn) -> dict:
    """TrainTest cell id -> HARNESS cell id, for every cell served (the
    routing table, the registry and the worker's SelectionRecord are all
    keyed by the harness id — ``safety_refusal__psr_kl_all`` serves as
    ``si_safety__psr_kl_all``, while the constitution / irrelevant ids
    coincide textually). Written into endpoint.json so a client never
    guesses the translation. Order kept, duplicates dropped."""
    out: dict = {}
    for c in cells:
        if c not in out:
            out[c] = "none" if c == "none" else harness_cell_id_fn(c)
    return out


def gate_config(cfg: Mapping[str, Any], cells: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """The two-SI config the standup's REGISTRY is built from: the model's
    own config with the served cells' arms ADDED to its class arm lists —
    ``(config, {"class_a_arms": [added], "class_b_arms": [added]})``.

    WHY. ``registry_adapter.traintest_cell_source`` builds the TrainTest
    registry from the config's ``class_a_arms`` / ``class_b_arms`` (module
    defaults when a key is absent), and ``enumerate_cells`` reports a cell
    whose arm that registry lacks as ABSENT. Each model's config lists the
    arms ITS campaign fitted — psr_kl_all / psr_mse_all / lora_r1_all (the
    legacy spelling of lora_r4, filtered out by ``build_registry`` as
    unregistered) + antidote / alphasteer on gemma-4, GLM and Nemotron;
    psr_kl_all + lora_r4 and NO Class B on Qwen3.5; everything but
    vector_all on llama — while the gate's inventory is the ADAPTERS' full
    servable vocabulary with synthetic objects for the rest. Job array
    382416 (2026-09-04) therefore killed the REFERENCE standup of every
    vector_all case, of lora_r1 / loreft on four families, of antidote /
    alphasteer on Qwen3.5, of lora_r4 and the multigroup on GLM / Nemotron
    with ``absent cells: [...]`` before the candidate ever stood up (the
    same code in both trees; the candidate would have died identically).
    The arm lists are a registry VOCABULARY, not a claim that the campaign
    fitted them; the object bytes under ``--defenses-dir`` are what is
    served, so widening the lists to the cells actually being stood up is
    the harness's call. Class A ids are ``<si>__<arm>``, Class B a bare arm
    (``defense_cell.si_of`` / ``arm_of``); ``none`` adds nothing. Order
    kept, no duplicates, nothing else in the config touched."""
    from sisteer.eval import defense_cell as tt

    out = dict(cfg)
    a_arms = list(cfg.get("class_a_arms", tt.CLASS_A_ARMS))
    b_arms = list(cfg.get("class_b_arms", tt.CLASS_B_ARMS))
    added: Dict[str, List[str]] = {"class_a_arms": [], "class_b_arms": []}
    for cell in cells:
        if cell == "none":
            continue
        arm = tt.arm_of(cell)
        if arm is None:
            raise RuntimeError(f"cell {cell!r} names no arm — the gate serves object cells "
                               "(Class A '<si>__<arm>' or a bare Class B arm) next to 'none'")
        key, arms = (("class_b_arms", b_arms) if tt.si_of(cell) is None
                     else ("class_a_arms", a_arms))
        if arm not in arms:
            arms.append(arm)
            added[key].append(arm)
    out["class_a_arms"], out["class_b_arms"] = a_arms, b_arms
    return out, added


def enumerate_gate_cells(*, repo: Path, cfg: Mapping[str, Any], cells_tt: Sequence[str],
                         defenses_dir: Path, manifest: str, chat_template_hash: str,
                         out: Path):
    """The served cell set of one standup: the petri spec pinned to the
    config's model, the TrainTest cell source over ``--defenses-dir`` built
    from :func:`gate_config` (written to ``<out>/two_si_gate.yaml`` — the
    registry reads a config FILE), and ``enumerate_cells`` over the
    ``["none", <harness ids>...]`` list. Returns ``(spec, src, cells,
    gate_cfg_path, arms_added)``; an ABSENT cell is a loud RuntimeError
    naming every absent id and its reason (never a standup that silently
    serves fewer cells than the gate asked for)."""
    import yaml
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.petri.registry_adapter import traintest_cell_source

    gcfg, added = gate_config(cfg, cells_tt)
    gate_cfg_path = Path(out) / "two_si_gate.yaml"
    gate_cfg_path.write_text(
        "# lens_gate_standup: the model's two-SI config with the served cells' arms\n"
        "# added to its class arm lists (gate_config) — a registry vocabulary for THIS\n"
        "# standup, never a fitting record.\n" + yaml.safe_dump(gcfg, sort_keys=False),
        encoding="utf-8")
    cells_list = harness_cells(cells_tt, ra.harness_cell_id)
    model, rev = cfg["model"], cfg["model_revision"]
    spec0 = cm.load_petri_spec(Path(repo) / manifest)
    spec = dataclasses.replace(
        spec0, target_model=model, target_revision=rev,
        serving_mode="interleaved", cells=cells_list,
        serving_launch_flags=[f for f in spec0.serving_launch_flags if f != "--enable-lora"],
        scenario_bundle_hash="ab" * 8, serving_impl_commit="fedcba9876" * 4,
        chat_template_hash=chat_template_hash)
    src = traintest_cell_source(spec, repo_root=Path(repo), config_path=gate_cfg_path,
                                defenses_dir=Path(defenses_dir))
    bf = {k: f"# {k}\n".encode() for k in cm.petri_expected_behavior_keys(spec)}
    if spec.prompt_provenance.startswith("override@"):
        from sisteer.capeval.tiers.petri_runner import resolve_prompt_files
        prompt_files = resolve_prompt_files(spec, override_root=Path(repo) / "configs/capeval/petri/prompts")
    else:
        prompt_files = {"auditor": b"# a\n", "judge": b"# j\n"}
    cells = dc.enumerate_cells(spec, src, behavior_files=bf, prompt_files=prompt_files)
    if cells.absent:
        raise RuntimeError("absent cells: " + "; ".join(
            f"{a.cell_id}: {a.reason}" for a in cells.absent)
            + f" (registry arms class_a={gcfg['class_a_arms']} class_b={gcfg['class_b_arms']}, "
              f"defenses_dir={Path(defenses_dir).resolve()})")
    return spec, src, cells, gate_cfg_path, added


def prefix_caching_attested(att: Any) -> Tuple[Any, Any]:
    """``(resolved state, error)`` — the launcher's RESOLVED prefix-caching
    state out of the steering attestation (``vllm_server.
    STEERING_ATTESTATION_FILE``), read STRICTLY: ``scheduler_geometry.
    expected.enable_prefix_caching`` must be a boolean AND every worker
    report (at least one) must carry a boolean ``enable_prefix_caching``
    (the shape observed in job 382416's attestations: ``expected`` True on
    llama / GLM / gemma-4, False on Qwen3.5 / Nemotron). The state is True
    if ANY of them says True (a cache that is on anywhere is on), else
    False. Anything missing, non-boolean or malformed is ``(None, <error
    string>)`` — recorded in the endpoint as ``prefix_caching_attest_error``
    and REFUSED by the gate (``prefix_caching_ok``); never a None the gate
    could read as a pass."""
    if not isinstance(att, Mapping):
        return None, f"attestation is not a JSON object ({type(att).__name__})"
    sg = att.get("scheduler_geometry")
    if not isinstance(sg, Mapping):
        return None, "attestation carries no scheduler_geometry mapping"
    exp = sg.get("expected")
    if not isinstance(exp, Mapping):
        return None, "scheduler_geometry.expected is not a mapping"
    e_val = exp.get("enable_prefix_caching")
    if not isinstance(e_val, bool):
        return None, ("scheduler_geometry.expected.enable_prefix_caching is not a boolean "
                      f"(got {e_val!r})")
    reports = sg.get("worker_reports")
    if not isinstance(reports, list) or not reports:
        return None, "scheduler_geometry.worker_reports is empty or not a list"
    vals = [e_val]
    for i, r in enumerate(reports):
        v = r.get("enable_prefix_caching") if isinstance(r, Mapping) else None
        if not isinstance(v, bool):
            return None, (f"worker report {i} carries no boolean enable_prefix_caching "
                          f"(got {v!r})")
        vals.append(v)
    return any(vals), None


#: the endpoint / proxy-identity key for the served reduction mode — the
#: literal twin of ``lens_worker.LENS_REDUCTION_MODE_KEY`` (pinned equal by
#: tests/test_lens_reduction_mode_attestation.py); a literal because the base
#: commit's lens_worker has no such name and this helper runs there too
LENS_REDUCTION_MODE_KEY = "lens_reduction_mode"
#: the RoutingProxy parameter carrying the attested per-cell resumed-prefill
#: policy (proxy.RoutingProxy.__init__). Named here because whether the SERVED
#: tree's proxy has it is decided at run time — see
#: :func:`resumed_prefill_proxy_kwargs`.
RESUMED_PREFILL_POLICY_KWARG = "resumed_prefill_policy_cells"


def reduction_mode_attested(att: Any, lw: Any = None) -> Tuple[Any, Any]:
    """``(mode, error)`` — the REDUCTION MODE the standup's workers serve
    grouped lens reductions in (``per_row`` | ``stacked``), read out of the
    steering attestation by ``lens_worker.attested_reduction_mode`` — the ONE
    strict reader (every ``scheduler_geometry.worker_reports`` entry must
    carry ``lens_reduction_mode`` and they must agree on a known mode; else
    ``(None, why)``). Recorded in the endpoint as ``lens_reduction_mode`` /
    ``lens_reduction_mode_error`` and handed to the proxy's identity block
    when attested. A tree whose lens_worker has no such reader — the base
    commit, whose workers attest no mode — is ``(None, <error>)``: recorded,
    never guessed (a consumer that finds no mode treats the server as
    per-row, which is exactly what that tree serves). ``lw`` is injectable
    for tests; the default is the tree's own ``lens_worker``."""
    if lw is None:
        from sisteer.capeval.serving import lens_worker as lw
    reader = getattr(lw, "attested_reduction_mode", None)
    if reader is None:
        return None, ("lens_worker carries no attested_reduction_mode — a tree that "
                      "predates the reduction-mode attestation (its workers attest "
                      "no lens_reduction_mode)")
    return reader(att)


def resumed_prefill_policy_cells_attested(att: Any, lw: Any = None) -> Tuple[Any, Any]:
    """``(mapping, error)`` — the per-cell RESUMED-PREFILL POLICY (``never`` |
    ``always`` | ``conditional``) the standup's workers attest, read out of
    the steering attestation by
    ``lens_worker.resumed_prefill_policy_cells_attested`` — the ONE strict
    reader (every top-level install ``worker_reports`` entry must carry
    ``resumed_prefill_policy_cells`` and they must agree). Handed to the proxy,
    which mints a CELL-STABLE cache salt for an ``always`` cell's steered legs
    and the request-unique salt for everything else; recorded in the endpoint
    as ``resumed_prefill_policy_cells`` / ``..._error``. A tree whose
    lens_worker has no such reader — anything before the resumed-prefill
    landing — is ``(None, <error>)``: recorded, and the proxy then shares
    nothing, which is exactly what that tree serves. ``lw`` is injectable for
    tests; the default is the tree's own ``lens_worker``."""
    if lw is None:
        from sisteer.capeval.serving import lens_worker as lw
    reader = getattr(lw, "resumed_prefill_policy_cells_attested", None)
    if reader is None:
        return None, ("lens_worker carries no resumed_prefill_policy_cells_attested — "
                      "a tree that predates the resumed-prefill landing (its workers "
                      "attest no per-cell policy)")
    return reader(att)


def resumed_prefill_proxy_kwargs(cells: Any, proxy_cls: Any) -> Tuple[Dict[str, Any], Any]:
    """``(kwargs, error)`` for handing the attested per-cell resumed-prefill
    policy to ``RoutingProxy`` — and the CROSS-TREE guard that makes it safe.

    THE HAZARD, which cost a GPU gate run on 2026-09-07: the vectorize gate
    launches the REFERENCE standup with THIS script (the candidate tree's) but
    the REFERENCE tree's ``sisteer`` on the path (``standup_env``; the split is
    recorded in the endpoint's ``served_from``). So the ``RoutingProxy`` this
    module imports may be OLDER than this module — and an unconditional
    ``resumed_prefill_policy_cells=`` kwarg is a ``TypeError`` before readiness
    on every reference standup, which the gate reports as a per-case FAIL with
    a truncated traceback. ``reduction_mode_attested``'s ``getattr(lw, ...)``
    guard exists for exactly this hazard on the reader side; this is its
    constructor twin.

    So: pass the kwarg only when the imported constructor takes it. A proxy
    that does not is a tree from before the resumed-prefill landing, which
    mints a request-unique salt for every steered leg and has no policy to
    honour — the correct behaviour there, recorded (never silently dropped) so
    the endpoint publishes ``resumed_prefill_policy_cells: null`` with the
    reason rather than a map the served proxy never received."""
    try:
        params = inspect.signature(proxy_cls.__init__).parameters
    except (TypeError, ValueError) as e:  # a C-level or unintrospectable ctor
        return {}, f"cannot introspect {getattr(proxy_cls, '__name__', proxy_cls)}: {type(e).__name__}: {e}"
    if RESUMED_PREFILL_POLICY_KWARG in params:
        return {RESUMED_PREFILL_POLICY_KWARG: cells}, None
    return {}, ("the served sisteer's RoutingProxy takes no "
                f"{RESUMED_PREFILL_POLICY_KWARG} — a tree from before the resumed-prefill "
                "landing; every steered leg there rides a request-unique cache_salt and "
                "no policy is honoured")


# --------------------------------------------------------------------------- #
# Declared-render standups (fit-support-transfer): the optional markers, the   #
# digests, the attested declarations, and the cross-tree kwarg guard           #
# --------------------------------------------------------------------------- #

#: the proxy's lens cache-regime vocabulary (``proxy.LENS_SALT_POLICIES``), a
#: literal here because this helper runs against trees whose proxy has no
#: such name; pinned equal by tests/test_lens_declared_render_gate.py
LENS_SALT_POLICIES = ("attested", "request_unique_all")
#: the RoutingProxy kwargs a declared-render standup hands over when the
#: served tree's constructor takes them (:func:`accepted_kwargs`)
PROXY_DECLARED_KWARGS = ("lens_serve_declarations", "lens_serve_declarations_error",
                         "thinking_header", "thinking_header_error", "lens_salt_policy",
                         "object_artifact_sha256_by_cell")
#: the ``export_worker_registry`` kwargs of the same landing
EXPORT_DECLARED_KWARGS = ("assistant_turn_end_ids", "assistant_header_ids_thinking_on",
                          "expected_artifact_sha256")
#: the ``export_routing_table`` kwargs: the served template's thinking-header
#: fact, and the reason its ON certification is unavailable beside it (the
#: proxy lane's S-P1 contract: the table carries the pair and refuses a
#: PARTIAL fact without its reason) — each handed over only when the served
#: tree's export takes it (:func:`accepted_kwargs`)
TABLE_DECLARED_KWARGS = ("thinking_header", "thinking_header_error")
#: ``serve_declaration.VARIANT_SEP`` — a literal here because the base commit
#: has no such module; pinned equal by tests/test_lens_declared_render_gate.py
VARIANT_SEP = "@"


def accepted_kwargs(fn: Any, wanted: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """``(kwargs, dropped)``: the entries of ``wanted`` whose NAME the
    callable's signature takes, and — per name it does not — the reason. The
    generalisation of :func:`resumed_prefill_proxy_kwargs`'s cross-tree guard
    to every kwarg this landing added: the SERVED tree's ``export_worker_
    registry`` / ``export_routing_table`` / ``RoutingProxy`` may predate it
    (the reference standup), and a ``TypeError`` before readiness would read
    as a per-case FAIL. A name whose value is ``None`` is never passed (the
    served API's own default stands) and never reported dropped. An
    unintrospectable callable drops everything, naming why — the caller
    decides which drops are refusals (a DECLARED cell on a tree that cannot
    serve it is one; a digest binding on the base commit is not)."""
    target = fn.__init__ if inspect.isclass(fn) else fn
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError) as e:  # a C-level or unintrospectable callable
        why = f"cannot introspect {getattr(fn, '__name__', fn)}: {type(e).__name__}: {e}"
        return {}, {k: why for k, v in wanted.items() if v is not None}
    out: Dict[str, Any] = {}
    dropped: Dict[str, str] = {}
    for name, value in wanted.items():
        if value is None:
            continue
        if name in params:
            out[name] = value
        else:
            dropped[name] = (f"the served sisteer's {getattr(fn, '__name__', fn)} takes no "
                             f"{name} — a tree from before the fit-support-transfer landing")
    return out, dropped


def is_declared_render_standup(cells: Sequence[str], *, flag: bool) -> bool:
    """Whether this standup serves one of the gate's DECLARED-RENDER cases
    (module docstring): the gate said so (``--declared-render``), or a served
    cell is spelled as a serve variant (``<cell>@<variant>``) — every such
    case serves one, so the requirement can never be lost to a forgotten
    flag. A plain standup (the gate's inventory cases, the base-commit
    reference) is neither."""
    return bool(flag) or any(VARIANT_SEP in str(c) for c in cells)


def declared_render_refusal(*, declared_render: bool, digests: Mapping[str, str],
                            export_dropped: Mapping[str, str], proxy_dropped: Mapping[str, str]
                            ) -> Optional[str]:
    """Why a DECLARED-RENDER standup REFUSES over its artifact binding, or
    ``None`` (code review S-G2). The cases certify that the REAL bank object
    is bound at export and its identity published by the proxy; a served tree
    whose ``export_worker_registry`` drops ``expected_artifact_sha256`` or
    whose ``RoutingProxy`` drops ``object_artifact_sha256_by_cell``
    (:func:`accepted_kwargs`) cannot give that evidence, so the standup
    refuses NAMING the kwarg instead of standing up and letting the endpoint
    publish digests the proxy never received. A standup that is not
    declared-render keeps today's behaviour: the drop is recorded in the
    endpoint (the base-commit reference stands up as before)."""
    if not declared_render:
        return None
    if not digests:
        return ("a declared-render standup serves no fitted lens cell whose artifact could be "
                "bound — nothing to certify; refusing")
    if "expected_artifact_sha256" in export_dropped:
        return ("a declared-render standup needs the registry export BOUND to the artifact "
                "digests, but the expected_artifact_sha256 kwarg was dropped: "
                f"{export_dropped['expected_artifact_sha256']}; refusing")
    if "object_artifact_sha256_by_cell" in proxy_dropped:
        return ("a declared-render standup needs the proxy to PUBLISH the artifact digests, but "
                "the object_artifact_sha256_by_cell kwarg was dropped: "
                f"{proxy_dropped['object_artifact_sha256_by_cell']}; refusing")
    return None


def lens_resolved_specs(cells: Any, src: Any, parse_serving_identity: Callable[[str], Any]
                        ) -> List[Mapping[str, Any]]:
    """The resolved specs of the enumerated CANONICAL lens cells (the cells
    whose declarations decide which optional markers the registry needs —
    ``capctrl_serving_up.bring_up``'s own read; the ``none__lens0`` twin is
    derived from a donor and adds nothing)."""
    out: List[Mapping[str, Any]] = []
    for c in cells.canonical:
        if parse_serving_identity(c.serving_identity).path != "lens":
            continue
        entry = src.get(c.cell_id)
        spec = getattr(entry, "resolved_spec", None) if entry is not None else None
        if isinstance(spec, Mapping):
            out.append(spec)
    return out


def declared_render_needs(lens_specs: Sequence[Mapping[str, Any]], sd: Any
                          ) -> Dict[str, bool]:
    """``{"thinking": bool, "all_turns": bool}`` — whether some served lens
    cell declares the thinking-ON render (``serve_out_of_support`` carrying
    ``"thinking"``) and whether some serves ``positions="response_all_turns"``
    — the two facts that make the registry need its optional markers
    (``capctrl_serving_up.bring_up``). ``sd`` is the served tree's
    ``serve_declaration`` module, or ``None`` on a tree that has none (the
    base commit): no declaration vocabulary, no need. A malformed declaration
    refuses through ``declarations_of`` — never read as undeclared."""
    if sd is None:
        return {"thinking": False, "all_turns": False}
    thinking = any(sd.OUT_OF_SUPPORT_THINKING
                   in sd.declarations_of(s).get(sd.SERVE_OUT_OF_SUPPORT_KEY, [])
                   for s in lens_specs)
    all_turns = any(s.get(sd.POSITIONS_KEY) == "response_all_turns" for s in lens_specs)
    return {"thinking": bool(thinking), "all_turns": bool(all_turns)}


def registry_markers(tok: Any, needs: Mapping[str, bool], cmk: Any) -> Dict[str, Any]:
    """The registry's marker ids for one standup, derived from the SERVED
    tokenizer exactly as ``capctrl_serving_up.bring_up`` derives them:

    - ``assistant_header_ids`` — the generation-prompt header (always);
    - ``assistant_turn_end_ids`` — iff ``needs["all_turns"]``: the turn-end
      scaffold (``chat_markers.assistant_turn_end_ids``); a template whose
      turns have no locatable end (successor-bounded, or one whose successor
      probe re-tokenizes — Qwen3.5) REFUSES: the all-turns mask cannot be
      served on it;
    - ``assistant_header_ids_thinking_on`` — iff ``needs["thinking"]``: the
      thinking-ON generation header (``generation_header_ids_under(tok,
      enable_thinking=True)``); EQUAL to the OFF header on a template whose
      switch does not move it (gemma-4) is recorded as such — admitted, the
      boundary the same under both renders;
    - ``thinking_header`` / ``thinking_header_error`` — the served template's
      thinking-header fact and the reason its ON certification is
      unavailable, with ``capctrl_serving_up.standup_thinking_header``'s
      discipline (code review X1 / S-G3): iff ``needs["thinking"]`` the whole
      fact derives HARD (``chat_markers.thinking_header_facts``; a template
      that refuses it refuses the standup — the declaration cannot be served
      without a certified ON header); otherwise NON-FATALLY through the
      served tree's ``thinking_header_facts_or_error``, which KEEPS the
      partial fact ``{switch, off_ids, on_ids: None, moves: None}`` and
      returns the ``ChatMarkerError`` reason beside it (the Nemotron-3 OFF
      standup). A tree with ``thinking_header_facts`` but no non-fatal helper
      catches ``ChatMarkerError`` ONLY (no partial fact can be kept there;
      the reason says so); any other exception is a DEFECT and propagates. A
      tree with neither (the base commit) records why, with ``None``.

    ``cmk`` is the served tree's ``chat_markers`` module; a need the tree
    cannot derive (no such function) refuses — a declared cell must never
    stand up on a tree that cannot serve its declaration."""
    header = tuple(int(t) for t in cmk.generation_header_ids(tok))
    out: Dict[str, Any] = {"assistant_header_ids": list(header),
                           "assistant_turn_end_ids": None,
                           "assistant_header_ids_thinking_on": None,
                           "thinking_header": None}
    if needs.get("all_turns"):
        derive = getattr(cmk, "assistant_turn_end_ids", None)
        if derive is None:
            raise RuntimeError("a cell serves positions='response_all_turns' but the served "
                               "tree's chat_markers derives no assistant_turn_end_ids — this "
                               "tree cannot serve the all-turns mask; refusing")
        try:
            scaffold = derive(tok)
        except Exception as e:  # noqa: BLE001 — the template's own refusal, named
            raise RuntimeError("a cell serves positions='response_all_turns' but the served "
                               "template's turn end cannot be located "
                               f"({type(e).__name__}: {e}) — the all-turns mask cannot be "
                               "served on it; refusing") from e
        if not scaffold:
            raise RuntimeError("a cell serves positions='response_all_turns' but the served "
                               "template is successor-bounded (no turn terminator) — the "
                               "all-turns mask cannot be served on it; refusing")
        out["assistant_turn_end_ids"] = [int(t) for t in scaffold]
    if needs.get("thinking"):
        derive = getattr(cmk, "generation_header_ids_under", None)
        if derive is None:
            raise RuntimeError("a cell declares the thinking-ON render (@thinking) but the "
                               "served tree's chat_markers derives no thinking-ON header — "
                               "this tree cannot serve the declaration; refusing")
        on_header = tuple(int(t) for t in derive(tok, enable_thinking=True))
        # equal to the OFF header on a header-invariant template (gemma-4): recorded
        # as such, admitted
        out["assistant_header_ids_thinking_on"] = list(on_header)
    facts = getattr(cmk, "thinking_header_facts", None)
    soft = getattr(cmk, "thinking_header_facts_or_error", None)
    out["thinking_header_error"] = None
    if facts is None:
        out["thinking_header_error"] = ("the served tree's chat_markers has no thinking_header_facts — a "
                                        "tree from before the fit-support-transfer landing")
    elif needs.get("thinking"):
        # a DECLARING standup: the whole fact, HARD — the template's own
        # refusal refuses the standup (capctrl_serving_up.standup_thinking_header)
        try:
            out["thinking_header"] = dict(facts(tok))
        except cmk.ChatMarkerError as e:
            raise RuntimeError("a cell declares the thinking-ON render but the served template's "
                               f"thinking-header fact cannot be derived ({type(e).__name__}: {e}); "
                               "refusing") from e
    elif soft is not None:
        # every other standup: the partial fact is KEPT and the reason recorded
        # beside it; the proxy then refuses every declared thinking-ON request on
        # a lens route thinking_on_unattested (fail closed) and serves OFF
        derived, reason = soft(tok)
        out["thinking_header"] = dict(derived)
        out["thinking_header_error"] = reason
    else:
        # a tree with the fact function but not the non-fatal helper: the same
        # narrowing (ChatMarkerError ONLY — a defect is never "unavailable"),
        # with no partial fact to keep
        try:
            out["thinking_header"] = dict(facts(tok))
        except cmk.ChatMarkerError as e:
            out["thinking_header_error"] = (f"{type(e).__name__}: {e} (the served tree's chat_markers has no "
                                            "thinking_header_facts_or_error, so the partial fact cannot be kept)")
    return out


def artifact_digests(cells: Any, src: Any, parse_serving_identity: Callable[[str], Any]
                     ) -> Dict[str, str]:
    """``{harness cell id: sha256}`` of the ARTIFACT BYTES every canonical lens
    cell's enumerated registry entry points at (``entry.object_path`` — the
    very file ``export_worker_registry`` copies and, when handed this map as
    ``expected_artifact_sha256``, holds its copy to). A variant cell resolves
    its base's path, so it carries the base's digest under its own id. A
    lens cell whose entry names no readable artifact refuses —
    ``enumerate_cells`` should have reported it ABSENT."""
    out: Dict[str, str] = {}
    for c in cells.canonical:
        if parse_serving_identity(c.serving_identity).path != "lens":
            continue
        entry = src.get(c.cell_id)
        path = getattr(entry, "object_path", None) if entry is not None else None
        if not path or not Path(path).is_file():
            raise RuntimeError(f"lens cell {c.cell_id!r} has no readable object artifact "
                               f"({path!r}) — enumerate_cells should have reported it absent")
        out[c.cell_id] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return out


def legacy_object_sha256(by_cell: Mapping[str, str]) -> Optional[str]:
    """The LEGACY SCALAR ``object_artifact_sha256`` (``capctrl_serving_up.
    legacy_object_sha256``): the single distinct digest when
    there is exactly one, else ``None``."""
    distinct = sorted(set(by_cell.values()))
    return distinct[0] if len(distinct) == 1 else None


def serve_declarations_attested(att: Any, lw: Any = None) -> Tuple[Any, Any]:
    """``(mapping, error)`` — the per-cell SERVE DECLARATIONS the standup's
    workers attest (``lens_serve_declarations`` on every top-level install
    ``worker_reports`` entry: harness cell id in variant spelling -> one closed
    declaration shape; the ``none__lens0`` twin under its own id when its
    donor declares), read by ``lens_worker.serve_declarations_attested`` — the
    ONE strict reader. An attested EMPTY map is ``({}, None)`` (a standup
    serving no declared cell). A tree whose lens_worker has no such reader —
    the base commit — is ``(None, <error>)``: recorded, never guessed, and the
    proxy then refuses every thinking-ON declaration (fail closed). ``lw`` is
    injectable for tests."""
    if lw is None:
        from sisteer.capeval.serving import lens_worker as lw
    reader = getattr(lw, "serve_declarations_attested", None)
    if reader is None:
        return None, ("lens_worker carries no serve_declarations_attested — a tree that "
                      "predates the fit-support-transfer landing (its workers attest no "
                      "serve declarations)")
    return reader(att)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _start_proxy_thread(app, host, port):
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--expect-sha", required=True)
    ap.add_argument("--config", required=True, help="two-SI config, relative to --repo-dir")
    ap.add_argument("--cell", required=True, action="append",
                    help="a TrainTest cell id to serve next to the `none` baseline; repeat "
                         "the flag to serve SEVERAL cells from one registry (the gate's "
                         "multigroup case: distinct objects in one batch)")
    ap.add_argument("--defenses-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--max-num-seqs", type=int, required=True)
    ap.add_argument("--gpu-mem-util", default="0.9")
    ap.add_argument("--max-logprobs", type=int, default=64)
    ap.add_argument("--ready-timeout-s", type=int, default=1800)
    ap.add_argument("--manifest", default="configs/capeval/manifests/petri_twosi.yaml")
    ap.add_argument("--prefix-caching", choices=("off", "default"), default="off",
                    help="the standup's prefix-caching regime: 'off' (every fidelity standup — the "
                         "B=1 controls compare cold prefills with cold prefills) or 'default' (the "
                         "production default, RESOLVED per model by vllm_server.build_server_launch: "
                         "ON for dense targets, OFF for hybrid-recurrent ones) — the gate's "
                         "shared-prefill case (resumed prefills for per-position ops) needs ON")
    ap.add_argument("--lens-salt-policy", choices=LENS_SALT_POLICIES, default="attested",
                    help="the proxy's lens cache regime (proxy.LENS_SALT_POLICIES, plan §2.4 S1): "
                         "'attested' (today's behaviour — the attested resumed-prefill map and the "
                         "declared-cells rule decide) or 'request_unique_all' (every lens leg mints "
                         "request-unique, audit label forced-request-unique). Recorded in the "
                         "endpoint as lens_salt_policy; REFUSED when the served tree's proxy cannot "
                         "take it and a non-default regime was asked for")
    ap.add_argument("--declared-render", action="store_true",
                    help="this standup serves one of the gate's DECLARED-RENDER cases (declared_thinking / "
                         "all_positions_carveout / all_turns_render_check): the artifact digest binding at "
                         "export AND its publication by the proxy are REQUIRED, so a served tree whose "
                         "export_worker_registry takes no expected_artifact_sha256 or whose RoutingProxy "
                         "takes no object_artifact_sha256_by_cell REFUSES before any launch, naming the "
                         "kwarg (code review S-G2). A standup serving a variant cell (<cell>@<variant>) is "
                         "declared-render whether or not the flag is given; without either, a dropped "
                         "digest kwarg is recorded in the endpoint (the base-commit reference stands up "
                         "as before)")
    args = ap.parse_args()

    _drop_own_dir()
    repo = Path(args.repo_dir).resolve()
    commit = verify_tree(repo, args.expect_sha)

    import sisteer  # noqa: E402  (from PYTHONPATH — the tree under test)
    sisteer_file = Path(sisteer.__file__).resolve()
    if repo / "src" not in sisteer_file.parents:
        raise SystemExit(f"[standup] imported sisteer from {sisteer_file}, not under "
                         f"{repo / 'src'} — PYTHONPATH does not name the tree; refusing")
    import yaml
    from transformers import AutoConfig, AutoTokenizer
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.serving import geometry as sg
    from sisteer.capeval.serving import vllm_server as vs
    from sisteer.capeval.serving.lens_steer import export_worker_registry
    from sisteer.capeval.serving.proxy import (HTTPBackend, RoutingProxy,
                                               TokenizerAgentLoopRender,
                                               export_routing_table,
                                               load_routing_table, make_app)
    from sisteer import chat_markers as cmk
    from sisteer.models import pin_reasoning_template_off
    # the served tree's declaration vocabulary — absent on the base commit,
    # which serves no variant cell and needs no optional marker
    try:
        from sisteer.capeval.serving import serve_declaration as sd
    except ImportError:
        sd = None

    geom = sg.resolve_geometry(None, max_model_len=args.max_model_len,
                               max_num_seqs=args.max_num_seqs,
                               fallback=(10240, 64), allow_over_budget=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stop_file = Path(args.stop_file)
    cfg_path = repo / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model, rev = cfg["model"], cfg["model_revision"]
    print(f"[standup] tree={repo} @ {commit[:12]} cells={args.cell} model={model}@{rev[:12]} "
          f"sisteer={sisteer_file}", flush=True)

    cells_list = harness_cells(args.cell, ra.harness_cell_id)
    cmap = cell_map(args.cell, ra.harness_cell_id)
    harness_id = cells_list[1] if len(cells_list) > 1 else "none"
    tok = AutoTokenizer.from_pretrained(model, revision=rev)
    pin_reasoning_template_off(tok)
    tpath = out / "chat_template.jinja"
    tpath.write_bytes(tok.chat_template.encode())
    tsha = cm.sha256_hex16(tok.chat_template.encode())
    mcfg = AutoConfig.from_pretrained(model, revision=rev)
    hidden = getattr(mcfg, "hidden_size", None) or getattr(
        getattr(mcfg, "text_config", None), "hidden_size", None)
    if not hidden:
        raise RuntimeError(f"cannot resolve hidden_size for {model}")
    spec, src, cells, gate_cfg_path, arms_added = enumerate_gate_cells(
        repo=repo, cfg=cfg, cells_tt=args.cell, defenses_dir=Path(args.defenses_dir),
        manifest=args.manifest, chat_template_hash=tsha, out=out)
    print(f"[standup] registry config {gate_cfg_path} (arms added to {cfg_path.name}: "
          f"{arms_added}); served cells {[c.cell_id for c in cells.all_served()]}", flush=True)
    reg_dir = out / "lens_registry"
    # DECLARED-RENDER STANDUPS (module docstring): the two optional registry
    # markers iff a served lens cell needs them, the served template's
    # thinking-header fact, and every fitted lens cell's artifact digest —
    # each handed to the served tree's API only when it takes the kwarg
    # (accepted_kwargs); a needed marker the tree cannot take REFUSES
    lens_specs = lens_resolved_specs(cells, src, cm.parse_serving_identity)
    needs = declared_render_needs(lens_specs, sd)
    markers = registry_markers(tok, needs, cmk)
    digests = artifact_digests(cells, src, cm.parse_serving_identity)
    export_kwargs, export_dropped = accepted_kwargs(export_worker_registry, {
        "assistant_turn_end_ids": (tuple(markers["assistant_turn_end_ids"])
                                   if markers["assistant_turn_end_ids"] else None),
        "assistant_header_ids_thinking_on": (tuple(markers["assistant_header_ids_thinking_on"])
                                             if markers["assistant_header_ids_thinking_on"] else None),
        "expected_artifact_sha256": dict(digests) if digests else None})
    needed_dropped = {k: v for k, v in export_dropped.items()
                      if k in ("assistant_turn_end_ids", "assistant_header_ids_thinking_on")}
    if needed_dropped:
        raise SystemExit(f"[standup] a served cell needs a registry marker the served tree's "
                         f"export_worker_registry cannot record: {needed_dropped} — a declared "
                         "cell must never stand up on a tree that cannot serve its declaration; "
                         "refusing")
    digests_bound = "expected_artifact_sha256" in export_kwargs
    # S-G2: on a DECLARED-RENDER standup the digest binding and its publication
    # are REQUIRED — probed against the served tree's APIs here, BEFORE the
    # export and any launch (the proxy probe introspects the same constructor
    # with the same kwarg the proxy is built with below), so the refusal names
    # the kwarg and costs no GPU time
    declared_render = is_declared_render_standup(args.cell, flag=args.declared_render)
    _probe, proxy_probe_dropped = accepted_kwargs(
        RoutingProxy, {"object_artifact_sha256_by_cell": (dict(digests) if digests else None)})
    refusal = declared_render_refusal(declared_render=declared_render, digests=digests,
                                      export_dropped=export_dropped, proxy_dropped=proxy_probe_dropped)
    if refusal:
        raise SystemExit(f"[standup] {refusal}")
    print(f"[standup] registry markers: header {markers['assistant_header_ids']}, turn_end "
          f"{markers['assistant_turn_end_ids']}, thinking_on header "
          f"{markers['assistant_header_ids_thinking_on']}; needs {needs}; declared_render {declared_render}; "
          f"{len(digests)} artifact digest(s) {'BOUND' if digests_bound else 'NOT bound'}"
          + (f" ({export_dropped['expected_artifact_sha256']})" if not digests_bound and digests else ""),
          flush=True)
    export_worker_registry(cells, src, reg_dir,
                           assistant_header_ids=tuple(markers["assistant_header_ids"]),
                           zero_op_hidden_size=int(hidden), **export_kwargs)
    table_path = out / "routing_table.json"
    # the served template's thinking-header fact AND the reason its ON
    # certification is unavailable ride into the routing table together
    # (TABLE_DECLARED_KWARGS; the proxy's contract refuses a PARTIAL fact
    # recorded without its reason), each only when the served tree's export
    # takes it. On the base-commit reference tree both are dropped — safe only
    # because that tree derives no fact at all (its chat_markers has no
    # thinking_header_facts, so registry_markers records None + the reason),
    # never a partial one.
    table_kwargs, _table_dropped = accepted_kwargs(export_routing_table, {
        "thinking_header": markers["thinking_header"],
        "thinking_header_error": markers["thinking_header_error"]})
    export_routing_table(cells, src, table_path, base_model=model,
                         serving_mode="interleaved", base_revision=rev, **table_kwargs)
    thinking_header_recorded = "thinking_header" in table_kwargs
    # the ONE reason this helper hands to the proxy and publishes: its own
    # derivation (registry_markers), or — when the served tree's export records
    # no fact at all — that
    thinking_header_error = markers["thinking_header_error"]
    if thinking_header_error is None and not thinking_header_recorded:
        thinking_header_error = "the served tree's export_routing_table records no thinking_header"
    if thinking_header_error is not None:
        print(f"[standup] thinking_header: ON certification UNAVAILABLE ({thinking_header_error}); "
              "a declared thinking-ON request on a lens route refuses", flush=True)
    # PREFIX CACHING OFF on every gate standup, every family (job array 382416,
    # 2026-09-04): with caching on (the dense families' production default),
    # the B=1 CONTROL comparison was warm-vs-cold -- the passthrough
    # singleton and the passthrough-only oracle resumed prompt 0 from cache
    # (the prompt had just been used by the B>1 waves), while the scale-0
    # `none__lens0` request is a LENS route and carries the proxy's
    # request-unique cache_salt (vllm_server / proxy: steered prefills never
    # resume), so it ran a full prefill. A cached-prefix prefill and a full
    # prefill are different attention computations on the same tokens
    # (different query lengths / kernel splits), so controls_B1 read "scale0
    # differs" at position 0 (logprobs, llama / gemma-4) or token 37 (GLM) on
    # BOTH trees for every case -- and passed on Qwen3.5 and Nemotron, the two
    # hybrids where build_server_launch resolves caching OFF. Not a hook
    # defect (the hook is a bit-exact no-op at scale 0) and not the
    # candidate's: a control that compared two cache states. Every gate
    # request is cold here, so the three roles run the same prefill; the
    # shadow compare and the steered B=1 verdicts never depended on caching
    # (steered requests are cold in production too). Both trees carry the
    # prefix_caching kwarg (vllm_server.py is identical in them).
    #
    # THE ONE EXCEPTION (2026-09-06): ``--prefix-caching default`` composes
    # the PRODUCTION default — ``prefix_caching=None``, resolved per model
    # (ON for dense targets, structurally OFF for hybrid-recurrent ones) —
    # for the gate's SHARED-PREFILL case, which needs a cache to resume
    # from: a resumable cell's warm request must be served as a RESUMED
    # PREFILL and attested, a non-resumable cell's must be refused. The
    # endpoint records BOTH the requested regime and the resolved state, so
    # the gate holds every other standup to OFF and that one to ON.
    prefix_caching = False if args.prefix_caching == "off" else None
    launch = vs.build_server_launch(
        spec, port=free_port(), lens_registry_dir=reg_dir, chat_template_path=tpath,
        max_model_len=geom.max_model_len, max_num_seqs=geom.max_num_seqs,
        prefix_caching=prefix_caching)
    launch = dataclasses.replace(
        launch, cmd=tuple(launch.cmd) + ("--gpu-memory-utilization", str(args.gpu_mem_util),
                                         "--max-logprobs", str(int(args.max_logprobs))))
    server = vs.launch_server(launch, log_path=out / "vllm_server.log",
                              ready_timeout_s=args.ready_timeout_s, install_timeout_s=900)
    proxy_loop = None
    rc = 0
    try:
        table = load_routing_table(table_path)
        # a served tree whose table carries the reason too (the proxy's
        # contract) must carry THIS helper's — one source for the proxy and the
        # endpoint, never two that can disagree
        if "thinking_header_error" in table and table.get("thinking_header_error") != thinking_header_error:
            raise SystemExit(f"[standup] the routing table records thinking_header_error="
                             f"{table.get('thinking_header_error')!r} but this helper derived "
                             f"{thinking_header_error!r}; refusing")
        # The readiness attestation the launcher wrote next to the log
        # (vs.STEERING_ATTESTATION_FILE), read ONCE, before the proxy is
        # built (the proxy echoes the served reduction mode read here).
        # Three facts come out of it, each published in the endpoint:
        # - the worker-side RESOLVED batch-invariance state (lens_worker.
        #   batch_invariant_report, merged into each worker's scheduler
        #   report); the base commit's workers carry no such fields —
        #   recorded as-is, judged by the gate against its intent;
        attested = []
        # - the launcher's RESOLVED cache state and any read/parse error,
        #   both published; the gate refuses anything but (False, None);
        prefix_attested, prefix_attest_error = None, None
        # - the served REDUCTION MODE (lens_reduction_mode: per_row |
        #   stacked) every worker attests, strictly agreed
        #   (reduction_mode_attested): None + the reason on the base commit
        #   (its workers attest none) or on any disagreement — never guessed.
        reduction_mode, reduction_mode_error = None, None
        # - the per-cell RESUMED-PREFILL POLICY every worker attests
        #   (resumed_prefill_policy_cells_attested): the proxy mints a
        #   CELL-STABLE lens salt for an `always` cell and the
        #   request-unique one for everything else, so this read is what
        #   makes a steered prefill resumable at all. None + the reason on
        #   any pre-landing tree or disagreement — never guessed.
        resumed_policy_cells, resumed_policy_error = None, None
        # - the per-cell SERVE DECLARATIONS every worker attests
        #   (serve_declarations_attested): the
        #   proxy validates every thinking-ON declaration against this map and
        #   mints declared cells request-unique; an attested {} is the stated
        #   "no declared cell"; None + the reason on any pre-landing tree or
        #   disagreement — the proxy then refuses every declaration (fail closed)
        serve_decls, serve_decls_error = None, None
        try:
            att = json.loads((out / vs.STEERING_ATTESTATION_FILE).read_text())
            sg_ = att.get("scheduler_geometry")
            prefix_attested, prefix_attest_error = prefix_caching_attested(att)
            reduction_mode, reduction_mode_error = reduction_mode_attested(att)
            resumed_policy_cells, resumed_policy_error = \
                resumed_prefill_policy_cells_attested(att)
            serve_decls, serve_decls_error = serve_declarations_attested(att)
            reports = (sg_.get("worker_reports") if isinstance(sg_, dict) else sg_) \
                or att.get("worker_reports") or []
            for r in reports:
                if isinstance(r, dict):
                    attested.append({k: r.get(k) for k in ("batch_invariant_env",
                                                             "batch_invariant_active",
                                                             "batch_invariant_surface",
                                                             "batch_invariant_error") if k in r})
        except Exception as e:  # noqa: BLE001 — recorded, judged by the gate
            attested = [{"error": f"{type(e).__name__}: {e}"}]
            prefix_attest_error = f"{type(e).__name__}: {e}"
            reduction_mode, reduction_mode_error = None, f"{type(e).__name__}: {e}"
            resumed_policy_cells, resumed_policy_error = None, f"{type(e).__name__}: {e}"
            serve_decls, serve_decls_error = None, f"{type(e).__name__}: {e}"
        # ...and whether the SERVED tree's proxy can be told about it at all
        resumed_policy_kwargs, resumed_policy_kwarg_error = \
            resumed_prefill_proxy_kwargs(resumed_policy_cells, RoutingProxy)
        if resumed_policy_kwarg_error:
            # the served proxy honours no policy, so the endpoint must not
            # publish one: a reader deriving a salt regime from a map the proxy
            # never received would be reading a fiction
            resumed_policy_cells = None
            resumed_policy_error = ((f"{resumed_policy_error}; " if resumed_policy_error else "")
                                    + resumed_policy_kwarg_error)
        print("[standup] resumed_prefill_policy_cells="
              + (f"{len(resumed_policy_cells)} cell(s)" if resumed_policy_cells is not None
                 else f"None ({resumed_policy_error})"), flush=True)
        # the declared-render kwargs (module docstring), each ONLY when the
        # served tree's RoutingProxy takes it; the endpoint publishes what was
        # handed over, never what this helper wished to hand over
        declared_kwargs, declared_dropped = accepted_kwargs(RoutingProxy, {
            "lens_serve_declarations": serve_decls,
            "lens_serve_declarations_error": (serve_decls_error if serve_decls is None else None),
            "thinking_header": (table.get("thinking_header") if thinking_header_recorded else None),
            "thinking_header_error": thinking_header_error,
            "lens_salt_policy": args.lens_salt_policy,
            "object_artifact_sha256_by_cell": (dict(digests) if digests_bound else None)})
        if "lens_serve_declarations" in declared_dropped:
            serve_decls = None
            serve_decls_error = ((f"{serve_decls_error}; " if serve_decls_error else "")
                                 + declared_dropped["lens_serve_declarations"])
        if "lens_salt_policy" in declared_dropped and args.lens_salt_policy != "attested":
            raise SystemExit(f"[standup] --lens-salt-policy {args.lens_salt_policy!r} was asked for "
                             f"but {declared_dropped['lens_salt_policy']}; a cache regime the served "
                             "proxy cannot honour is refused, never silently dropped")
        salt_policy_handed = "lens_salt_policy" in declared_kwargs
        # the digest map the proxy actually RECEIVED decides what the endpoint
        # publishes (S-G2): bound at export AND taken by the served proxy — a
        # declared-render standup refused above otherwise, so here the fact is
        # merely recorded, with the reason when it is not
        digests_handed = "object_artifact_sha256_by_cell" in declared_kwargs
        digests_error = None
        if not digests_handed:
            digests_error = (("no fitted lens cell is served — nothing to bind" if not digests else None)
                             or export_dropped.get("expected_artifact_sha256")
                             or declared_dropped.get("object_artifact_sha256_by_cell")
                             or "the digest map was not handed to the proxy")
        print("[standup] lens_serve_declarations="
              + (f"{serve_decls}" if serve_decls is not None else f"None ({serve_decls_error})")
              + f"; lens_salt_policy={args.lens_salt_policy!r}"
              + ("" if salt_policy_handed else f" (NOT handed over: {declared_dropped.get('lens_salt_policy')})"),
              flush=True)
        proxy = RoutingProxy(
            table["routes"], HTTPBackend(launch.base_url, api_key=launch.api_key),
            base_model=model, serving_mode="interleaved",
            prompted_prefixes=table["prompted_prefixes"], exact_replay_strategy=None,
            # the attested mode rides to the proxy's /v1/models identity block
            # (where a consumer reads it); nothing is passed when unattested,
            # so the block stays silent rather than guessing
            launch_attestation=({LENS_REDUCTION_MODE_KEY: reduction_mode}
                                if reduction_mode is not None else None),
            agent_loop_render=TokenizerAgentLoopRender(tok), petri_agent_loop_entry=True,
            # the attested per-cell resumed-prefill policy: the proxy shares a
            # prefill only for an `always` cell, and only because THIS map says
            # so (nothing attested -> nothing shared). Passed only when the
            # SERVED tree's RoutingProxy takes it — the reference standup runs
            # this script against an older sisteer
            **resumed_policy_kwargs,
            # ...and the declared-render facts, on the same terms
            **declared_kwargs)
        proxy_port = free_port()
        proxy_loop = _start_proxy_thread(make_app(proxy), "127.0.0.1", proxy_port)
        proxy_url = f"http://127.0.0.1:{proxy_port}/v1"
        auth = out / "server_auth.json"
        fd = os.open(str(auth), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"api_key": launch.api_key}, f)
        ep = {"proxy_url": proxy_url, "server_base_url": launch.base_url,
              "server_host": launch.host, "server_port": launch.port,
              "routing_table": str(table_path), "cell": args.cell[0],
              "cells": list(args.cell), "harness_cells": cells_list[1:],
              "cell_map": cmap,
              "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT"),
              "batch_invariant_attested": attested,
              # the COMPOSED cache state (False on every fidelity standup; the
              # per-model default on the shared-prefill standup) and the
              # launcher's resolved one, for the gate's prefix_caching_ok /
              # prefix_caching_on_ok checks; the REQUESTED regime for provenance
              "prefix_caching": launch.prefix_caching,
              "prefix_caching_requested": args.prefix_caching,
              "prefix_caching_attested": prefix_attested,
              "prefix_caching_attest_error": prefix_attest_error,
              "shadow_compare": os.environ.get("SISTEER_LENS_SHADOW_COMPARE"),
              "per_row_apply": os.environ.get("SISTEER_LENS_PER_ROW_APPLY"),
              # the served REDUCTION MODE inside a vectorized group, as the
              # workers ATTEST it (per_row | stacked; None + the reason when a
              # worker attests none — the base commit — or they disagree; never
              # a guess), so a stacked standup can never pass as a plain one...
              "lens_reduction_mode": reduction_mode,
              "lens_reduction_mode_error": reduction_mode_error,
              # the per-cell RESUMED-PREFILL POLICY the workers ATTEST, and the
              # reason when they attest none: the proxy was handed exactly this
              # (null -> it shares no steered prefill at all), so a client can
              # derive the salt regime any steered request of this standup was
              # minted under without asking the proxy
              "resumed_prefill_policy_cells": resumed_policy_cells,
              "resumed_prefill_policy_error": resumed_policy_error,
              # ...and the raw env echo (lens_adapters.STACKED_REDUCTIONS_ENV),
              # provenance only: the ATTESTED value above is the mode
              "stacked_reductions_env": os.environ.get("SISTEER_LENS_STACKED_REDUCTIONS"),
              # DECLARED-RENDER facts (module docstring; fit-support-transfer):
              # the per-cell SERVE DECLARATIONS the workers ATTEST — PRESENT
              # whenever attested AND handed to the proxy ({} = no declared
              # cell), ABSENT otherwise with the reason beside it (null when
              # the map is present); the proxy validates every thinking-ON
              # declaration against exactly this map
              **({"lens_serve_declarations": serve_decls} if serve_decls is not None else {}),
              "lens_serve_declarations_error": (None if serve_decls is not None
                                                else serve_decls_error),
              # the served template's thinking-header fact the proxy decides
              # from (recorded in the routing table when the tree takes it)
              "thinking_header": (table.get("thinking_header") if thinking_header_recorded
                                  else None),
              # ...and the reason its ON certification is UNAVAILABLE (None beside
              # a fully derived fact) — the SAME value handed to the proxy, so
              # endpoint.json and /v1/models agree (S-G3)
              "thinking_header_error": thinking_header_error,
              # the proxy's lens cache regime as CONSTRUCTED (null + the reason
              # when the served proxy takes no such setting — it then behaves as
              # today's `attested`, which is the only value this helper lets
              # through to such a tree)
              "lens_salt_policy": (args.lens_salt_policy if salt_policy_handed else None),
              "lens_salt_policy_error": (None if salt_policy_handed
                                         else declared_dropped.get("lens_salt_policy")),
              # every fitted lens cell's ARTIFACT digest (harness id in variant
              # spelling -> sha256 of the enumerated object's bytes) and the
              # legacy scalar (the single distinct digest, else null) — PRESENT
              # only when export_worker_registry held its copy to these digests
              # AND the served proxy received the map (it then publishes the
              # same map in /v1/models); else null / null / false with the
              # reason, never this helper's local digests (S-G2)
              "object_artifact_sha256_by_cell": (dict(digests) if digests_handed else None),
              "object_artifact_sha256": (legacy_object_sha256(digests) if digests_handed else None),
              "object_artifact_sha256_bound": bool(digests_handed),
              "object_artifact_sha256_error": digests_error,
              "declared_render": bool(declared_render),
              # the registry's marker ids as exported (the gate's span
              # arithmetic reads them here rather than re-deriving)
              "registry_markers": {k: markers[k] for k in ("assistant_header_ids",
                                                           "assistant_turn_end_ids",
                                                           "assistant_header_ids_thinking_on")},
              "declared_render_needs": needs,
              "harness_cell": harness_id, "model": model, "revision": rev,
              "config": str(cfg_path), "gate_config": str(gate_cfg_path),
              "arms_added": arms_added,
              "commit": commit, "tree": str(repo),
              "served_from": {"sisteer": str(sisteer_file),
                              "geometry": str(Path(sg.__file__).resolve()),
                              "vllm_server": str(Path(vs.__file__).resolve())},
              "max_model_len": geom.max_model_len, "max_num_seqs": geom.max_num_seqs,
              "pid": os.getpid(), "node": socket.gethostname()}
        tmp = out / "endpoint.json.tmp"
        tmp.write_text(json.dumps(ep, indent=2))
        tmp.rename(out / "endpoint.json")
        print(f"[standup] proxy up at {proxy_url}; waiting for {stop_file}", flush=True)
        while not stop_file.exists():
            proc = getattr(server, "proc", None) or getattr(server, "process", None) or server
            if hasattr(proc, "poll") and proc.poll() is not None:
                print(f"[standup] FATAL: vLLM server died (rc={proc.poll()})", flush=True)
                rc = 3
                break
            time.sleep(2)
        print("[standup] stop requested; tearing down", flush=True)
    finally:
        if proxy_loop is not None:
            try:
                proxy_loop.call_soon_threadsafe(proxy_loop.stop)
            except Exception:
                pass
        try:
            vs._shutdown_child(server)
        except Exception:
            pass
    print("[standup] down.", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
