#!/usr/bin/env python
"""sdfbase-mt: stand up ONE base model's served target for the multi-turn
attack wave (crescendo / actorattack / PAIR) or its Petri audit — EVERY cell
of the wave on one vLLM+lens server behind one RoutingProxy, on loopback —
then block until a stop file appears.

The multi-cell twin of ``scripts/capctrl_serving_up.py`` (which stands up ONE
cell + ``none``; left untouched for the lanes that launch through it). Same
bring-up, same library code, no generation here:

  1. the wave's transfer config (``explicit_cells``: none + the four
     full-corpus cells) + the wave's Petri manifest (the SAME cells as
     harness ids) -> ``traintest_cell_source`` over the per-wave object
     bank (``--defenses-dir``) -> ``enumerate_cells`` (the canonical cells
     plus the derived scale-0 twins the steering-bites probe needs);
  2. worker registry + routing table exported next to the server; the
     served tokenizer's chat template written and hashed against the
     manifest's ``chat_template_hash`` (a stale pin refuses before a GPU
     minute is spent);
  3. ``build_server_launch`` at the eval's geometry from
     ``sisteer.capeval.serving.geometry`` at the ``--model-key`` model's
     batched-token budget (qwen35_4b 720,896: ``--eval multiturn_attacks`` =
     8704 x 82 for the attack drivers; ``--eval petri`` = 16384 x 44 for the
     audit — never an inherited window), tool calls ON for the Petri agent
     modality only; the reduction mode (``--reduction-mode``) is pinned into
     the worker env BEFORE the lens loads, and the mode the workers ATTEST
     (``lens_reduction_mode`` in ``steering_attestation.json``) is read back,
     recorded in ``endpoint.json`` / ``cells_served.json``, and must equal the
     pin (absent on pre-landing workers, recorded as absent);
  4. ``RoutingProxy`` with the Petri-facing agent-loop entry declared (a
     no-op for the attack drivers, whose chats end at the generation header),
     ``endpoint.json`` written atomically (proxy_url, routing_table, mode,
     eval, geometry, served vmodels, the object shas the bank carries, the
     Slurm job id + node) — the drivers and the grading host's Petri driver
     read it;
  5. wait for ``--stop-file``; a dead vLLM (proc exited) ends the standup
     with rc 3 so a watcher sees a FAILED job, never a live proxy over a dead
     engine.

Object regime: every sdfbase object is a ``positions=all`` fit and the lens
serves it at its recorded fit leg (``_BLOB_POSITION_SCHEMAS``); the attack
driver therefore omits ``--positions`` (``--target-engine served`` refuses
``all`` by design).

THINKING-ON ADDITIONS (this file stays the cluster standup and runs the four
capctrl steps it lacked through the SHARED module
``sisteer.capeval.serving.standup_thinking``; every addition is flag-gated
and the composed launch command is byte-identical without the flags):

  * ``--cell <id>`` (repeatable; default = every ``explicit_cells`` entry):
    restricts the exported registry + routing table to the named cells
    (``none`` always present; the ``<cell>__lens0`` twin derived as today);
    the variant spelling ``<cell>@thinking`` rides through to the cell source
    (``serve_declaration.apply_variant`` — the declaration is hashed into the
    cell's ``intervention_hash``); on an all-position cell the lens's rule 3
    refuses it as VACUOUS and the standup surfaces that refusal verbatim;
  * the served template's THINKING-HEADER FACT is ALWAYS computed
    (``standup_thinking.standup_thinking_header``: hard for a declaring cell,
    the partial fact + ``thinking_header_error`` otherwise — the pinned
    Nemotron-3 template's ON header is underivable and that is RECORDED, never
    replaced by a table's ids) and written VERBATIM into the routing table
    and endpoint.json; a declaring cell exports the registry with the second
    header (``assistant_header_ids_thinking_on``), an undeclared standup
    exports the OFF header alone (rule 5);
  * after launch the proxy is built with the attested per-cell facts —
    ``resumed_prefill_policy_cells``, ``lens_serve_declarations`` (EVEN WHEN
    EMPTY), ``thinking_header`` (+ error), ``object_artifact_sha256_by_cell``,
    ``lens_salt_policy`` — and the table's ``lens_served_positions`` stamp
    (what lets the proxy ACCEPT an undeclared
    thinking-ON render on a lens cell whose served mask is not header-located),
    handed over exactly as capctrl does and RECORDED in endpoint.json with
    ``lens_served_positions_passthrough`` saying whether the constructor took it;
  * ``--upstream-timeout-s`` -> ``HTTPBackend(timeout_s=)`` (this
    experiment passes 7200) recorded as ``proxy_upstream_timeout_s``;
    ``--expect-object-sha256 <cell>=<sha>`` (repeatable) holds the registry
    export to the preflight-verified artifact bytes; ``--reasoning-parser``
    (gemma-4 only — ``vllm_server`` refuses any other family's);
    ``--lens-salt-policy`` (default ``attested``);
  * ``endpoint.json`` gains exactly the keys the tinycap driver's standup-record
    readers consume (``check_serve_attestation``, ``_served_identity``, the
    ledger's ``proxy_token``): additive, every existing key keeps its value.
  * the routing table is REGENERATED on every start (it always was): a table
    written before that stamp and loaded by a proxy after it would fail
    closed as ``thinking_on_undeclared`` (the stale-table hazard).

THE STANDUP-WIDE THINKING MODE (``--thinking-on``; the sampled-rerun lineage,
docs/served_thinking_on.md) sits beside the per-cell path: the vLLM launch pins
the ON render (``vllm_server.build_server_launch(thinking_on=True)``), the served
template's ON header is CERTIFIED per template before any GPU minute
(``chat_markers.thinking_on_certification`` — a ``no_switch`` / ``refused`` /
``error`` verdict refuses the standup), every HEADER-DEPENDENT lens cell is
declared BY NAME in its hashed spec (``declared_cell_spellings``: the ``<cell>@thinking``
spelling ``--cell`` uses; a ``positions=all`` fit declares nothing), the two ON-header derivations must agree where both run,
the workers' attested header + declarations must equal the export
(``thinking_attestation_problems``), and the proxy runs ``thinking_mode: on``.
Every artifact (``endpoint.json`` / ``cells_served.json``) carries
``thinking_record``'s fields — ``thinking_mode`` ALWAYS ("off" by default), so a
reader asserts the regime rather than inferring it from absence — and
``upstream_timeout_s`` (the proxy -> vLLM hop timeout in force).

THE SERVING PATH (since 2026-09-13): ``--serving-path
{lora_native,lens}``, default ``lora_native`` (``registry_adapter.
DEFAULT_STANDUP_SERVING_PATH``, the ENTRY-POINT default). Under native this
standup is a TRANSLATION onto the wave front ``lora_serving_up.bring_up_lora``
(itself a front over the shared core ``sisteer.capeval.serving.native_standup``):
the config / manifest / cells / geometry flags forward unchanged; ``--thinking-on``,
``--reasoning-parser``, ``--enable-log-requests``, ``--text-only-mm``,
``--kv-derived-seqs``, ``--mamba-backend-contract``, ``--adapter-bank`` forward to
the core; the lens pins ``--reduction-mode`` (non-default), ``--require-attested-mode``
and ``--lens-salt-policy request_unique_all`` REFUSE by name, exit 2 (a lens fact
the native engine cannot attest; the defaults are accepted and recorded
``not_applicable``); a residual cell of the view (PSR / vector / vector_all / diffmean /
static_r) SERVES on the native engine's residual slot at every position mode the lens
names (native-psr-serving: stage 1 served ``all``, stage 2's per-token position class serves
``response`` / ``prompt_last`` / ``prefill_all`` — the slot armed by the composer, attested
under schema 2); a stacked cell or a position-masked LoRA fit is SKIPPED BY NAME (recorded
in ``cells_skipped`` with the reason), and an explicit ``--cell`` naming a skip-class cell refuses;
``--expect-object-sha256``
forwards when given, else the digest of every served LoRA cell is derived from the
bank view's ``VIEW_MANIFEST.json`` + ``configs/sdfbase_full22_pins.json`` and held
to the served ``object.pt`` (``expected_digests_source: view_manifest+pins``).
``--serving-path lens`` runs the body of :func:`_main` below byte for byte.
"""
from __future__ import annotations

import argparse
import dataclasses
import typing
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from capctrl_serving_up import (_start_proxy_thread, free_port,  # noqa: E402
                                expected_artifact_digests_for_cells, legacy_object_sha256,
                                parse_object_digest_args, resolve_standup_geometry)

EVAL_BY_MODE = {"attacks": "multiturn_attacks", "petri": "petri"}
#: the producer script a LENS standup's serving facts are filed under (hooks.emit_serving_patches' ``script``; the native core's
#: write_endpoint files under its own, capeval.serving.native_standup.write_endpoint) — full22's post-standup gate
#: (scripts/sdfbase/full22/registry_serving_recorded.py SERVING_WRITERS) admits both
LENS_SERVING_FACTS_SCRIPT = "sdfbase.mt_serving_up.lens_endpoint"
#: the serving env var that selects the steering hook's reduction. "0" = the per-row reference
#: path (bit-identical to the legacy hook); "1" = the stacked reductions (faster, ulp-level logit
#: differences). Unread by pre-landing serving code.
REDUCTION_ENV_VAR = "SISTEER_LENS_STACKED_REDUCTIONS"
#: the SERVED-side field: after the stacked-reductions landing every worker's
#: ``sisteer_install_steering`` report names the reduction it actually installed under this key
#: (``per_row`` | ``stacked``). It is the field that tells a later reader which side of the
#: boundary a shard came from — read from the server's own attestation record
#: (``steering_attestation.json``), never from the env we set ourselves (that would be the
#: standup attesting to itself). Pre-landing workers do not emit it; ABSENT is recorded as such.
LENS_REDUCTION_FIELD = "lens_reduction_mode"
REDUCTION_MODES = ("per_row", "stacked")


# --- the standup-wide thinking mode (`--thinking-on`; the sampled-rerun lineage) ------------
# Beside the per-cell `--cell <id>@thinking` declaration path above: ONE flag pins the vLLM
# launch to the ON render, certifies the served template's ON header per template BEFORE any
# GPU minute (`chat_markers.thinking_on_certification`), declares every HEADER-DEPENDENT lens
# cell of the wave, and runs the proxy in `thinking_mode: on` (every lens-route request must
# declare `petri_thinking_on`). OFF (the default) records the certification and changes
# nothing; a `--cell <id>@thinking` standup keeps its per-request regime. Both spellings
# meet at the exported registry: the attestation check reads the exported entries.

#: the thinking MODE recorded on every artifact of a standup (endpoint.json /
#: cells_served.json ``thinking_mode``; the proxy's identity block) — a closed
#: vocabulary, the literal twin of ``proxy.THINKING_MODES``
THINKING_MODES = ("off", "on")
#: the per-template certification VERDICTS (``chat_markers.thinking_on_certification``)
#: a ``--thinking-on`` standup accepts: a moved, certified header (declared cells
#: get the second header) or an unmoved one (nothing to declare)
THINKING_SERVABLE_VERDICTS = ("certified", "trivially_in_support")
#: the per-cell HEADER DEPENDENCE of a served lens cell's ops: ``response`` (a
#: header-located mask — declared under --thinking-on, the ON header exported),
#: ``all`` (every position — nothing to relocate, served undeclared), or a
#: mixed / other shape recorded verbatim
HEADER_DEPENDENCE_RESPONSE = "response"
HEADER_DEPENDENCE_ALL = "all"


class ThinkingResolution(typing.NamedTuple):
    """What ``--thinking-on`` (or its absence) resolved to for ONE standup,
    before any GPU minute: the mode, the served template's certification and
    header fact, and — for an ON standup on a template whose switch moves the
    header — the ON header the registry exports for the DECLARED cells.

    A NamedTuple, not a dataclass, on purpose: tests load this module by FILE
    PATH (``importlib.util.spec_from_file_location`` + ``exec_module``) without a
    ``sys.modules`` entry, and under ``from __future__ import annotations`` a
    dataclass resolves its string annotations through ``sys.modules`` at class
    creation and dies there (``test_mrope_report_hybrid``)."""

    mode: str                       # "off" | "on"
    certification: dict             # chat_markers.thinking_on_certification (+ model_key)
    header: dict                    # {switch, moves, off_ids, on_ids} — the proxy's fact
    header_error: object            # why on_ids is None on a switched template, else None
    #: the ON header exported into the registry (declared cells only): the
    #: certified ids on an ON standup with a MOVED header and >= 1 header-
    #: dependent lens cell; None otherwise (OFF; trivially_in_support; no such cell)
    on_header_ids: object
    #: the lens cells declared ``serve_out_of_support: ["thinking"]`` (harness ids)
    declared_cells: tuple
    #: every lens cell's header dependence (harness id -> "response" | "all" | ...)
    header_dependence: dict


def thinking_header_fact(certification: dict) -> tuple:
    """``(fact, error)`` — the proxy's ``thinking_header`` fact
    (``{switch, moves, off_ids, on_ids}``, the schema ``proxy._validated_thinking_header``
    admits) read off a certification, and the reason its ON half is
    UNAVAILABLE (``on_ids`` None on a switched template) or None."""
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


def served_hf_id_by_key() -> dict:
    """canonical model key -> the HF id a standup's --model-key must name: the paper's eight (geometry.MODEL_HF_IDS) AND the
    front door's onboarded models (geometry.ONBOARDED_MODEL_HF_IDS — reproduce/models.yaml's served checkpoints), the same two
    tables geometry.rows_for / budget_for size from. Read at call time (the tables are the documented seam). The rule: a key the
    geometry tables know is admitted here, whichever table holds it — a check against MODEL_HF_IDS alone refused every onboarded
    key as `not a canonical model key` (paper-companion-repo stage E, 2026-09-30), so neither standup path consults it on its own."""
    from sisteer.capeval.serving.geometry import MODEL_HF_IDS, ONBOARDED_MODEL_HF_IDS
    return {**MODEL_HF_IDS, **ONBOARDED_MODEL_HF_IDS}


def certify_thinking(tok, *, model_key: str) -> dict:
    """The served template's thinking-ON certification
    (``chat_markers.thinking_on_certification``) with the model key stamped
    on, made NON-FATAL: a derivation error is recorded as ``verdict:
    "error"`` with the exception, never raised — an OFF standup records it
    and serves; only an ON standup turns a non-servable verdict into a
    refusal (:func:`resolve_thinking`)."""
    from sisteer.chat_markers import generation_header_ids, thinking_on_certification
    try:
        cert = dict(thinking_on_certification(tok))
    except Exception as e:  # noqa: BLE001 — recorded as an error verdict; an ON standup refuses on it
        cert = {"switch": "enable_thinking" in (getattr(tok, "chat_template", None) or ""),
                "off_ids": list(generation_header_ids(tok)), "on_ids": None, "moves": None,
                "checks": {}, "verdict": "error", "reason": f"{type(e).__name__}: {e}"}
    cert["model_key"] = model_key
    cert["off_header_len"] = len(cert["off_ids"])
    cert["on_header_len"] = None if cert.get("on_ids") is None else len(cert["on_ids"])
    return cert


def lens_header_dependence(spec, source, *, off_header_ids: tuple) -> dict:
    """harness cell id -> the HEADER DEPENDENCE of the cell's served ops, for
    every lens cell of ``spec``: the cell's ops adapted exactly as the worker
    will (``lens_adapters.get_adapter(kind)`` over the registry entry's bytes
    and resolved spec, under the OFF header) and classified by the position
    modes they serve — ``"response"`` when every op is header-located,
    ``"all"`` when none is, else the sorted mode list joined by ``+`` (recorded
    verbatim; such a cell is declared like a response cell, since its
    header-located ops move). A cell that cannot be adapted REFUSES
    (``SystemExit``): the standup must not guess a mask's shape."""
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.serving import lens_adapters as la
    out = {}
    for cid in spec.cells:
        entry = source.get(cid)
        if entry is None or not entry.object_path or entry.serving_path != "lens":
            continue
        parsed = dc.parse_cell_id(cid)
        rspec = dict(entry.resolved_spec) if entry.resolved_spec is not None \
            else {"kind": entry.kind, "si_id": entry.si_id}
        try:
            artifact = Path(entry.object_path).read_bytes()
            sidecar = Path(entry.sidecar_path).read_bytes() if entry.sidecar_path else b""
            kind = rspec.get("kind", entry.kind)
            ops = la.get_adapter(kind)(rspec, artifact, sidecar, header_ids=tuple(off_header_ids))
        except Exception as e:  # noqa: BLE001 — a refusal with its reason, never a guess
            raise SystemExit(f"[mt-standup] cannot classify the mask of lens cell {cid!r} "
                             f"({parsed.cell_class}): {type(e).__name__}: {e} — refusing to "
                             "decide its thinking declaration by guesswork")
        modes = sorted({op.positions for op in ops})
        if modes == [la.POSITIONS_RESPONSE]:
            out[cid] = HEADER_DEPENDENCE_RESPONSE
        elif la.POSITIONS_RESPONSE not in modes:
            out[cid] = HEADER_DEPENDENCE_ALL
        else:
            out[cid] = "+".join(modes)
    return out


def declared_cell_spellings(spec, declared) -> list:
    """``spec.cells`` with every cell in ``declared`` (harness ids) respelled
    ``<cell>@thinking`` (``serve_declaration.join_variant``), the rest verbatim.
    A declaration is DECLARED BY NAME on this tree: the enumeration refuses a
    spec that declares what its id does not (``defense_cells``' name/spec
    agreement), and the cell source applies the ``thinking`` overlay itself
    when the id carries the variant (``registry_adapter``) — so a
    ``--thinking-on`` standup declares its header-dependent cells with exactly
    the spelling ``--cell <id>@thinking`` uses, the intervention hash covers
    the declaration, and a declared cell is a DISTINCT served identity
    (vmodel ``petri__<cell>@thinking``) from its thinking-OFF base. A cell
    already carrying a variant refuses (``join_variant``'s rule: one variant)."""
    from sisteer.capeval.serving import serve_declaration as sd
    declared = frozenset(declared)
    out = []
    for cid in spec.cells:
        if cid in declared:
            try:
                out.append(sd.join_variant(cid, sd.VARIANT_THINKING))
            except sd.ServeDeclarationError as e:
                raise SystemExit(f"[mt-standup] --thinking-on cannot declare {cid!r}: {e}") from None
        else:
            out.append(cid)
    return out


def resolve_thinking(tok, spec, source, *, thinking_on: bool, model_key: str,
                     off_header_ids: tuple) -> ThinkingResolution:
    """Resolve the standup's thinking MODE before any GPU minute.

    OFF (the default): the certification is recorded non-fatally (a reader
    can see what ON would have derived), nothing is declared, no cell's mask
    is classified (``header_dependence`` is ``{}``), no ON header is
    exported — the registry and launch are byte-identical to before.

    ON: the served template's certification must be SERVABLE
    (:data:`THINKING_SERVABLE_VERDICTS`) — ``no_switch`` / ``refused`` /
    ``error`` REFUSE the standup with the reason (never an uncertified
    steered thinking-ON cell). ``certified`` (the header MOVES): every
    HEADER-DEPENDENT lens cell (:func:`lens_header_dependence`) is declared
    and the certified ON header is exported for them; a wave with no such
    cell (this wave's ``positions=all`` fits) declares nothing and exports no
    second header, recorded as such. ``trivially_in_support`` (gemma-4: the
    header does not move): nothing to declare, no second header."""
    cert = certify_thinking(tok, model_key=model_key)
    fact, fact_error = thinking_header_fact(cert)
    if not thinking_on:
        # OFF: nothing to declare, so no cell's mask is classified — classifying means
        # ADAPTING every lens artifact (the worker's own load), a cost the declaration
        # pays, never a byte-identical OFF standup; recorded as {} (not classified)
        return ThinkingResolution(mode="off", certification=cert, header=fact,
                                  header_error=fact_error, on_header_ids=None,
                                  declared_cells=(), header_dependence={})
    dependence = lens_header_dependence(spec, source, off_header_ids=tuple(off_header_ids))
    verdict = cert.get("verdict")
    if verdict not in THINKING_SERVABLE_VERDICTS:
        raise SystemExit(
            f"[mt-standup] --thinking-on REFUSED for model_key={model_key}: the served "
            f"template's thinking-ON header certification is {verdict!r} — "
            f"{cert.get('reason')}. Never serving an uncertified thinking-ON standup; "
            f"checks={json.dumps(cert.get('checks'), default=str)}")
    declared = tuple(sorted(cid for cid, dep in dependence.items()
                            if dep != HEADER_DEPENDENCE_ALL))
    if verdict == "trivially_in_support":
        # the header does not move: a second header would be VACUOUS (the
        # export refuses one), so nothing is declared and the response masks
        # of any header-dependent cell locate the unchanged header as before
        return ThinkingResolution(mode="on", certification=cert, header=fact,
                                  header_error=None, on_header_ids=None,
                                  declared_cells=(), header_dependence=dependence)
    on_ids = tuple(int(t) for t in cert["on_ids"])
    return ThinkingResolution(mode="on", certification=cert, header=fact, header_error=None,
                              on_header_ids=on_ids if declared else None,
                              declared_cells=declared, header_dependence=dependence)


def thinking_record(res: ThinkingResolution, *, launch=None,
                    attested_header=None, attested_header_error=None,
                    attested_declarations=None, attested_declarations_error=None) -> dict:
    """The thinking-mode block every standup artifact carries (endpoint.json
    and cells_served.json; docs/served_thinking_on.md names each field):
    ``thinking_mode``, ``thinking_declaration_key``, ``default_chat_template_kwargs``
    (the composed launch pin, from the ServerLaunch), ``thinking_header`` (+
    ``thinking_header_error``), ``thinking_certification``,
    ``thinking_declared_cells``, ``thinking_header_dependence``,
    ``assistant_header_ids_thinking_on`` (exported) and, after launch, the
    workers' ``attested_assistant_header_ids_thinking_on`` /
    ``lens_serve_declarations`` (+ their ``_error`` twins)."""
    return {
        "thinking_mode": res.mode,
        "thinking_declaration_key": "petri_thinking_on",
        "default_chat_template_kwargs": (getattr(launch, "default_chat_template_kwargs", None)
                                         if launch is not None else None),
        "thinking_header": (dict(res.header) if res.header is not None else None),
        "thinking_header_error": res.header_error,
        "thinking_certification": json.loads(json.dumps(res.certification, default=str)),
        "thinking_declared_cells": list(res.declared_cells),
        "thinking_header_dependence": dict(res.header_dependence),
        "assistant_header_ids_thinking_on": (None if res.on_header_ids is None
                                             else list(res.on_header_ids)),
        "attested_assistant_header_ids_thinking_on": (None if attested_header is None
                                                      else list(attested_header)),
        "attested_assistant_header_ids_thinking_on_error": attested_header_error,
        "lens_serve_declarations": (None if attested_declarations is None
                                    else {k: dict(v) for k, v in attested_declarations.items()}),
        "lens_serve_declarations_error": attested_declarations_error,
    }


def expected_declarations(entries) -> dict:
    """The per-cell declaration map the workers MUST attest for a registry
    exported as ``entries`` (``export_worker_registry``'s return): every
    entry whose resolved spec declares — the declared cells and, when their
    donor declares, the ``none__lens0`` twin under its own id."""
    from sisteer.capeval.serving import serve_declaration as sd
    out = {}
    for e in entries:
        decl = sd.declarations_of(e.resolved_spec)
        if decl:
            out[e.cell_id] = decl
    return out


def thinking_attestation_problems(attestation, *, mode: str, want_header, expected_decls: dict
                                  ) -> tuple:
    """Every reason the workers' install reports do NOT prove the thinking
    regime the standup exported: the attested ON header must equal the
    exported one (``want_header``; None when none was exported), and the
    attested per-cell declarations must equal ``expected_decls``
    (:func:`expected_declarations` over the exported registry entries — BOTH
    spellings of a declaration, ``--thinking-on`` and ``--cell <id>@thinking``).
    Returns ``(problems, attested_header, header_error, attested_decls,
    decls_error)``; an empty ``problems`` is the pass. A pre-landing worker
    (no key) is a problem on an ON standup and a recorded absence on an OFF one."""
    from sisteer.capeval.serving import lens_worker as lw
    header, header_err = lw.thinking_header_attested(attestation)
    decls, decls_err = lw.serve_declarations_attested(attestation)
    problems = []
    want = None if want_header is None else tuple(int(t) for t in want_header)
    if header_err is not None:
        if mode == "on":
            problems.append(f"thinking-ON header not attested: {header_err}")
    elif header != want:
        problems.append(f"workers attest assistant_header_ids_thinking_on={header} but the "
                        f"standup exported {want}")
    if decls_err is not None:
        if mode == "on":
            problems.append(f"serve declarations not attested: {decls_err}")
    elif decls != expected_decls:
        problems.append(f"workers attest lens_serve_declarations={decls} but the standup "
                        f"exported {expected_decls}")
    return problems, header, header_err, decls, decls_err



def served_reduction_mode(attestation) -> tuple:
    """Resolve the reduction mode the SERVER attested, from a ``steering_attestation.json``
    payload. Returns ``(mode, source)``: ``mode`` is ``"per_row"``/``"stacked"`` or ``None`` when
    no worker names one; ``source`` says where it came from (``"worker_reports"``,
    ``"attestation_top"``, ``"absent"``). Raises ``ValueError`` when the workers disagree among
    themselves or a worker names a mode outside :data:`REDUCTION_MODES` — a server whose
    workers did not all install the same reduction must not serve a wave pinned to one."""
    if not isinstance(attestation, dict):
        raise ValueError(f"attestation payload is {type(attestation).__name__}, not a mapping")
    reports = attestation.get("worker_reports") or []
    seen = {}
    for i, rep in enumerate(reports):
        if isinstance(rep, dict) and LENS_REDUCTION_FIELD in rep:
            seen[i] = rep[LENS_REDUCTION_FIELD]
    if seen:
        modes = set(seen.values())
        if len(modes) != 1:
            raise ValueError(f"workers disagree on {LENS_REDUCTION_FIELD}: "
                             f"{dict(sorted(seen.items()))}")
        if len(seen) != len(reports):
            raise ValueError(f"only {len(seen)}/{len(reports)} worker reports carry "
                             f"{LENS_REDUCTION_FIELD}: {dict(sorted(seen.items()))}")
        mode = modes.pop()
        source = "worker_reports"
    elif LENS_REDUCTION_FIELD in attestation:
        mode, source = attestation[LENS_REDUCTION_FIELD], "attestation_top"
    else:
        return None, "absent"
    if mode not in REDUCTION_MODES:
        raise ValueError(f"{LENS_REDUCTION_FIELD}={mode!r} is not one of {REDUCTION_MODES}")
    return mode, source


def served_reduction_mode_strict(attestation) -> tuple:
    """:func:`served_reduction_mode` hardened for the full22 wave: when the tree carries
    ``lens_worker.attested_reduction_mode`` — the ONE strict reader over BOTH report kinds, the
    install reports (``sisteer_install_steering``, top-level ``worker_reports``) AND the
    scheduler-config reports (``sisteer_scheduler_config``, ``scheduler_geometry.worker_reports``),
    requiring every report present to agree — its verdict wins: a mode -> ``(mode,
    "install+scheduler_reports")``; a refusal (disagreement, unknown value, a None-valued key) ->
    ``ValueError``. Only when that reader finds NO key anywhere does the local reader run, so a
    pre-landing server is still recorded as ``(None, "absent")`` and never guessed."""
    try:
        from sisteer.capeval.serving import lens_worker as lw
        strict = getattr(lw, "attested_reduction_mode", None)
    except Exception:  # noqa: BLE001 — a tree without the module falls to the local reader
        strict = None
    if strict is not None and isinstance(attestation, dict):
        mode, err = strict(attestation)
        if mode is not None:
            return mode, "install+scheduler_reports"
        if err and ("disagree" not in err) and ("not one of" not in err) and ("None" not in err):
            # no report carries the key at all (pre-landing) -> the local reader decides absent
            pass
        elif err:
            raise ValueError(err)
    return served_reduction_mode(attestation)


def assert_manifest_matches_config(spec, cfg: dict) -> dict:
    """The standup's manifest <-> config gate, factored so the CPU admission test of a wave view runs
    the SAME check: every manifest cell translates to one of the
    config's explicit cells and every explicit cell is served — the two enumerations are ONE wave.
    Returns ``{harness cell id: TrainTest cell id}`` in manifest order; raises ``SystemExit`` on a
    mismatch, exactly as the standup always did."""
    from sisteer.capeval.petri import registry_adapter as ra

    tt_of = {c: ra.traintest_cell_id(c) for c in spec.cells}
    want = list(cfg["explicit_cells"])
    missing = [c for c, tt in tt_of.items() if tt not in want]
    unserved = [c for c in want if c not in tt_of.values()]
    if missing or unserved:
        raise SystemExit(f"[mt-standup] manifest/config cell mismatch: manifest "
                         f"cells not in the config {missing}; config cells the "
                         f"manifest does not serve {unserved}")
    return tt_of


def attested_scheduler_geometry(attestation) -> tuple:
    """The scheduler geometry the WORKERS attested — ``(max_num_seqs, max_model_len)`` read from
    every ``scheduler_geometry.worker_reports[*]`` entry (the ``sisteer_scheduler_config`` RPC:
    vLLM's live scheduler config, not the argv) — or ``None`` when no report carries the fields.
    Workers that disagree raise ``ValueError``. The standup compares this with the geometry it
    RESOLVED and refuses a mismatch: a silently clamped ``max_num_seqs`` (vLLM lowering the width
    it was asked for) must be visible in the record, never inferred from throughput."""
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


#: M-RoPE DISPATCH ATTESTATION (the full22-newmodels GLM lane, 2026-09-08). vLLM 0.25.1's fused
#: ``triton_mrope`` kernel has no ``is_neox_style`` and rotates every M-RoPE model neox-style; the
#: GLM-4V line is the one GPT-J-pair (``is_neox_style=False``) M-RoPE family, so an EAGER engine
#: (the served path: vllm-lens forces ``--enforce-eager``) mis-rotates every attention layer unless
#: ``capeval.serving.vllm_patches`` routed the 2-D-position case onto ``forward_native`` in the
#: worker. The attestation logic — the RPC name, the non-neox key list, the report judge and the
#: asker — now lives in ``sisteer.capeval.serving.mrope_attest``, so
#: the tinycap standup ``capctrl_serving_up.py`` asks the SAME question the same way; this launcher
#: imports the four names back and behaves byte-for-byte as before. The constant is re-stated here so the
#: wave test's source pin keeps reading it; ``tests/test_mrope_attest_shared.py`` holds it equal to the shared module's.
from sisteer.capeval.serving.mrope_attest import (  # re-exported: the wave test reads them off this module
    NON_NEOX_MROPE_KEYS, mrope_attest, mrope_attestation_problems)  # noqa: F401
MROPE_RPC_METHOD = "sisteer_mrope_report"


def _utc_now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: the standup's SERVING-CONFIG provenance record, beside the server log in the standup dir: the sha256 of
#: the config + manifest the leg actually serves. Written before launch; pulled beside the attestation pair; read by
#: the analyses' ``launch_attestation``.
SERVING_CONFIG_RECORD_FILE = "serving_config.json"
SERVING_CONFIG_RECORD_SCHEMA = "serving_config_record/1"


def _bank_objects(defenses_dir: Path, tt_cells: list[str]) -> dict:
    """object.pt sha256 + the staged_from record per compiled cell — what the
    endpoint advertises so a driver/grader can pin WHICH bytes served it."""
    out = {}
    for cid in tt_cells:
        if cid == "none" or cid.startswith("prompted__"):   # baselines: no object (a prompted cell is the
            continue                                          # bare model behind an H8-verified system prefix)
        d = defenses_dir / cid
        obj = d / "object.pt"
        rec = {"object_sha256": _sha256(obj) if obj.is_file() else None}
        sf = d / "staged_from.json"
        if sf.is_file():
            rec["staged_from"] = json.loads(sf.read_text())
        tbm = d / "train_budget_meta.json"
        if tbm.is_file():
            meta = json.loads(tbm.read_text())
            rec["identity"] = "staged"
            rec["sidecar_object_sha256"] = meta.get("object_sha256")
            rec["positions_at_fit"] = meta.get("positions_at_fit",
                                               meta.get("positions"))
        fpm = d / "fit_provenance.json"
        if fpm.is_file():
            # the HPO-promotion lineage's fit record (qwen's scale-12 si12
            # cells): the same fit-time facts as a budget sidecar under
            # the stage-3 gate's native spelling. It records no object
            # sha (the bank view's pin binds the bytes; object_sha256
            # above is the on-disk truth), so the endpoint advertises
            # the fit facts it DOES carry, normalised the one way every
            # reader normalises them (sweep.read_stage3_fit_provenance)
            from sisteer.eval.sweep import read_stage3_fit_provenance
            norm = read_stage3_fit_provenance(d, object_path=obj)
            rec["identity"] = ("ambiguous" if tbm.is_file() else
                               "fit_provenance")
            rec["sidecar_object_sha256"] = None
            if norm is not None:
                rec["positions_at_fit"] = norm.get("positions_at_fit")
                rec["fit_record"] = {k: norm.get(k) for k in (
                    "resolved_model_id", "resolved_revision",
                    "chat_template_hash", "source", "status")}
        dmm = d / "derived_meta.json"
        if dmm.is_file():
            # a DERIVED object (sdf-base-steering round 2) carries this
            # sidecar INSTEAD of a budget one — it was never trained. The
            # endpoint advertises the same two facts (which bytes, which
            # fit leg) plus the parent identities, so a grader reading the
            # endpoint can tell WHICH decomposition served a leg without
            # re-opening the bank.
            meta = json.loads(dmm.read_text())
            rec["identity"] = "ambiguous" if tbm.is_file() else "derived"
            rec["sidecar_object_sha256"] = meta.get("object_sha256")
            rec["positions_at_fit"] = meta.get("positions")
            rec["derived"] = {
                "derived_id": meta.get("derived_id"),
                "kind": meta.get("kind"),
                "derivation": meta.get("derivation"),
                "parents": {k: {"cell_id": v.get("cell_id"),
                                "object_sha256": v.get("object_sha256")}
                            for k, v in meta.items()
                            if k.startswith("parent_") and isinstance(v, dict)}}
        out[cid] = rec
    return out


def _serving_config_record(out: Path, *, cfg_path: Path, manifest_path: Path, model_key: str, model: str,
                           rev: str, repo: Path) -> dict:
    """THE SERVING-CONFIG PROVENANCE: the sha256 of the config and the
    manifest THIS standup serves, written BEFORE the engine starts (so a leg that dies at launch still records what it
    would have served) into the standup dir, beside the log and the workers' attestation. Two lanes need it for the
    same reason: an object selected on one tree's serving config and evaluated on another's is a SECOND INSTRUMENT,
    and the fork has to be VISIBLE per leg rather than inferred from a wave sha. The value is recorded, never compared
    across arms — each row may legitimately serve its own overlay (an external-bank row does). ONE writer for both
    serving paths."""
    serving_config_record = {
        "schema": SERVING_CONFIG_RECORD_SCHEMA,
        "config": str(cfg_path), "serving_config_sha256": _sha256(cfg_path),
        "manifest": str(manifest_path), "manifest_sha256": _sha256(manifest_path),
        "model_key": model_key, "model": model, "model_revision": rev,
        "repo_dir": str(repo), "written_at": _utc_now(),
        "how": ("sha256 of the config / manifest FILE BYTES this standup was given (mt_serving_up --config / --manifest); "
                "the leg's launch attestation surfaces serving_config_sha256 so a pre-/post-retirement instrument fork "
                "is visible without re-reading the cluster"),
    }
    (Path(out) / SERVING_CONFIG_RECORD_FILE).write_text(json.dumps(serving_config_record, indent=2, default=str) + "\n")
    return serving_config_record


#: the wave's pins file (repo-relative; a test points it at a synthetic one) — the second reader of the bank view's
#: object digests under native (the first is VIEW_MANIFEST.json), the `view_manifest+pins` source
PINS_FILE = "configs/sdfbase_full22_pins.json"
VIEW_MANIFEST_FILE = "VIEW_MANIFEST.json"
#: the entry-point serving paths
SERVING_PATHS = ("lora_native", "lens")


def _pin_form_sha(entry: dict, *, repo_root: Path, cell: str) -> str:
    """The pinned sha of a CROSS-BANK pin-form view entry (``make_bank_view.plan_pin_entry``): the entry records the
    pins FILE the roster named (``pins``, a path relative to the CHECKOUT ROOT — the roster's own spelling, e.g.
    ``configs/misalign_baseline_pins.json``) and the ``<model>.<cell>`` reference (``pin``). The EXACT recorded path
    is resolved against the checkout root (an absolute path as given) — never reduced to a basename beside the wave
    pins (an entry naming an absent ``archive/misalign_baseline_pins.json`` would otherwise silently read
    ``configs/misalign_baseline_pins.json``) — and that entry's ``object_sha256`` is the pin; the wave's pins file
    never carries the rung (the misalignment instrument's authority is the one file). A malformed record, a missing
    file or a missing entry refuses by name."""
    rel, ref = str(entry.get("pins") or ""), str(entry.get("pin") or "")
    parts = ref.split(".")
    if not rel or len(parts) != 2 or not all(parts):
        raise SystemExit(f"[mt-standup] served cell {cell!r}: pin-form view entry records pins={rel!r} pin={ref!r} — "
                         "not a pins file + `<model>.<cell>`; refusing an unpinned object")
    path = Path(rel) if Path(rel).is_absolute() else Path(repo_root) / rel
    if not path.is_file():
        raise SystemExit(f"[mt-standup] served cell {cell!r}: pin-form view entry names pins file {rel!r}, resolved against "
                         f"the checkout {repo_root} as {path} — no such file; refusing an unpinned object")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"[mt-standup] served cell {cell!r}: pins file {path} unreadable ({e}); refusing an unpinned object")
    pin = (doc.get(parts[0]) or {}).get(parts[1]) if isinstance(doc, dict) else None
    sha = pin.get("object_sha256") if isinstance(pin, dict) else None
    if not isinstance(sha, str) or not sha:
        raise SystemExit(f"[mt-standup] served cell {cell!r}: {path} has no {ref} object_sha256 — refusing an unpinned object")
    return sha


def view_manifest_digests(defenses_dir: Path, *, pins_path: Path, target: str, cells: list, harness_ids: list,
                          repo_root: typing.Optional[Path] = None) -> list:
    """The ``--expect-object-sha256`` list for every served OBJECT cell, derived from the bank view: the
    view's ``VIEW_MANIFEST.json`` entry (``cell_id -> staged_id, object_sha256``), the served ``object.pt`` re-hashed,
    and ``pins.json[target][staged_id].object_sha256`` — all three must agree, a served cell absent from the manifest
    or the pins refuses, a drifted object refuses. A pin-form entry (``form: "pin"``; a cross-bank object such as the
    Arditi rung) takes its pin from the authority it records (:func:`_pin_form_sha`, the recorded path
    resolved against ``repo_root`` — the checkout the wave pins live in, ``pins_path``'s grandparent when not given),
    held to the same three-way agreement. ``none`` and prompted cells serve no object and are skipped."""
    repo_root = Path(repo_root) if repo_root is not None else Path(pins_path).resolve().parents[1]
    manifest_path = Path(defenses_dir) / VIEW_MANIFEST_FILE
    if not manifest_path.is_file():
        raise SystemExit(f"[mt-standup] no --expect-object-sha256 given and no {VIEW_MANIFEST_FILE} under "
                         f"{defenses_dir} to derive the served objects' digests from — pass the digests, or build the "
                         "view with make_bank_view.py (which writes the manifest)")
    if not Path(pins_path).is_file():
        raise SystemExit(f"[mt-standup] no pins file at {pins_path} — the view-manifest digests are held to the "
                         "wave's pins; pass --expect-object-sha256 instead")
    entries = {e.get("cell_id"): e for e in (json.loads(manifest_path.read_text()).get("entries") or [])
               if isinstance(e, dict)}
    pins = json.loads(Path(pins_path).read_text()).get(target) or {}
    out = []
    for cell, hid in zip(cells, harness_ids):
        if cell == "none" or hid == "none" or hid.startswith("prompted__"):
            continue
        entry = entries.get(cell)
        if entry is None:
            raise SystemExit(f"[mt-standup] served cell {cell!r} has no entry in {manifest_path} — refusing to serve an "
                             "object the view manifest does not name (pass --expect-object-sha256 for it)")
        staged_id, recorded = entry.get("staged_id"), entry.get("object_sha256")
        if entry.get("form") == "pin":
            # a CROSS-BANK pin-form entry: its pin lives in the authority the roster
            # named, never in the wave's pins file — read it from there
            pinned = _pin_form_sha(entry, repo_root=repo_root, cell=cell)
        else:
            pinned = (pins.get(staged_id) or {}).get("object_sha256") if staged_id else None
        obj = Path(defenses_dir) / cell / "object.pt"
        if not obj.is_file():
            raise SystemExit(f"[mt-standup] served cell {cell!r}: no object.pt at {obj}")
        on_disk = _sha256(obj)
        if not pinned:
            raise SystemExit(f"[mt-standup] served cell {cell!r} (staged {staged_id!r}) is not pinned in {pins_path} "
                             f"under {target!r} — refusing an unpinned object")
        if not (on_disk == recorded == pinned):
            raise SystemExit(f"[mt-standup] served cell {cell!r}: object.pt sha256 {on_disk[:12]} / view manifest "
                             f"{str(recorded)[:12]} / pin {pinned[:12]} disagree — a drifted object; refusing")
        out.append(f"{cell}={on_disk}")
    return out


def select_cells(explicit_cells, requested) -> list:
    """The TrainTest cells this standup serves: ``explicit_cells`` (the config's
    enumeration, today's behaviour) when ``requested`` is None / empty, else
    ``none`` followed by every requested cell in order (de-duplicated; ``none``
    may be named, it is implicit). A requested cell's BASE (its ``@<variant>``
    split off — ``serve_declaration.split_variant``) must be one of the
    config's explicit cells; anything else refuses by name — a standup asked
    for a cell the wave never enumerated must not quietly serve the rest."""
    from sisteer.capeval.serving import serve_declaration as sd
    want = list(explicit_cells)
    if not requested:
        # a STEERED-WAVE config (s<N>r) lists no `none` — the base route is still SERVED, implicitly, exactly as under
        # --cell: load_check.py probes it and the native attestation reads vmodels["none"]; it is never a measured, registered or scored leg
        # (the drivers iterate the config's cells; misalign_policy.measure_none false; the Petri frozen set is the explicit declaration)
        return want if "none" in want else ["none", *want]
    out = ["none"]
    for raw in requested:
        if not isinstance(raw, str) or not raw.strip():
            raise SystemExit(f"[mt-standup] --cell needs a non-empty TrainTest cell id, got {raw!r}")
        cell = raw.strip()
        if cell == "none" or cell in out:
            continue
        try:
            base, _variant = sd.split_variant(cell)
        except sd.ServeDeclarationError as e:
            raise SystemExit(f"[mt-standup] --cell {cell!r}: {e}") from None
        if base not in want:
            raise SystemExit(f"[mt-standup] --cell {cell!r}: its base {base!r} is not one of the "
                             f"config's explicit_cells {want} — refusing to serve a cell the wave "
                             "never enumerated")
        out.append(cell)
    return out


def wait_until_stop(stop_file: Path, server, *, poll_s: float = 5.0) -> int:
    """Block until ``stop_file`` appears; return 0 then, or 3 when the vLLM
    child exits first (a dead engine ends the standup so a watcher sees a
    FAILED job, never a live proxy over a dead engine). A seam: the CPU tests
    replace it to read the standup's records while they exist."""
    while not stop_file.exists():
        proc = getattr(server, "proc", None) or getattr(server, "process", None)
        if proc is None:
            proc = server if hasattr(server, "poll") else None
        if proc is not None and proc.poll() is not None:
            print("[mt-standup] FATAL: vLLM server died while serving "
                  f"(rc={proc.poll()}) — see vllm_server.log", flush=True)
            return 3
        time.sleep(poll_s)
    print("[mt-standup] stop requested; tearing down", flush=True)
    return 0


def launch_cmd_extras(gpu_mem_util, max_logprobs=None, enable_log_requests=False) -> tuple:
    """The flags the standup splices onto ``build_server_launch``'s command — the memory
    utilization every lane passes, plus ``--max-logprobs`` ONLY when a lane asks for one, plus
    ``--enable-log-requests`` ONLY when a lane opts in.

    The controls driver scores MMLU by the top-``MMLU_TOP_LOGPROBS`` logprobs and the engine
    refuses a request above its ``--max-logprobs`` (default 20); the attack lanes never ask for
    logprobs and MUST keep their launch command byte-identical, so ``None`` adds nothing.

    ``--enable-log-requests`` (OPT-IN, default off) makes vLLM 0.25.1 write ONE INFO line per
    request, ``Received request <id>: params: SamplingParams(...)``, carrying the sampling params
    the ENGINE RESOLVED — after its own defaults were merged in for anything the body omitted.
    That object is the only in-engine readback of the resolved sampler at this version: the
    response body carries no sampling fields, and the startup "overridden by the model's
    generation_config.json" line is emitted ONLY when such a file exists, so for a target that
    ships none there is otherwise no evidence at all of what an omitted key became. At INFO the
    PROMPT IS NOT LOGGED (prompt text is a separate DEBUG record), so this is safe on a lane
    whose prompts are harmful. ``scripts/sdfbase/record_resolved_sampler.py`` reads the lines
    back and refuses to read their absence as agreement. Spliced here, beside the two flags
    above, to keep it OUT of ``src/sisteer/capeval/serving/``, which every production leg
    asserts byte-identical to its pin. ``False`` adds nothing.
    """
    extras = ("--gpu-memory-utilization", str(gpu_mem_util))
    if max_logprobs is not None:
        if int(max_logprobs) < 1:
            raise ValueError(f"--max-logprobs must be >= 1, got {max_logprobs!r}")
        extras += ("--max-logprobs", str(int(max_logprobs)))
    if enable_log_requests:
        extras += ("--enable-log-requests",)
    return extras


def serve_chat_template(tok, override):
    """Decide the SERVED chat template. Default: the pinned tokenizer's own. With ``override`` (a file), its
    bytes replace ``tok.chat_template`` so every downstream reader — the thinking certification, the exported
    chat_template.jinja vLLM renders with, the manifest hash check — sees the served template. Why it exists:
    the campaign's pinned Llama checkpoint (NousResearch mirror) ships a 348-byte MINIMAL
    template with no default system block, while Meta's stock Llama-3.1 template always emits a
    "Cutting Knowledge Date / Today Date" system header; the EM model organisms (trained under the stock
    template) read ~4x less misaligned under the minimal render, so the organisms panel is served under the
    stock template — declared here, hash-pinned by the manifest, recorded in cells_served.json."""
    import hashlib
    sha16 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]   # == capeval.manifest.sha256_hex16
    own = tok.chat_template or ""
    rec = {"override": None, "tokenizer_chat_template_sha16": sha16(own)}
    if override is None:
        return rec
    override = Path(override)
    if not override.is_file():
        raise SystemExit(f"[mt-standup] --chat-template {override}: no such file")
    text = override.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit(f"[mt-standup] --chat-template {override}: empty template")
    tok.chat_template = text
    rec["override"] = str(override)
    rec["override_sha16"] = sha16(text)
    return rec
_ENV_UNSET = object()


def main(argv=None) -> int:
    """The standup. The lens reduction switch (``REDUCTION_ENV_VAR``) is written into THIS process's
    environment for the duration of the run — the in-process lens code reads it at import, and the child
    vLLM server is handed the same value explicitly through ``launch_server(extra_env=...)`` — and RESTORED
    on the way out (unset again, or the caller's prior value put back), so an in-process caller (a test
    driving ``main``) never leaks it into whatever runs after it in the same process: the launcher's tests
    read the same variable as their expectation of the served mode and refuse a stale one by name
    (decisions-integration I4)."""
    prior = os.environ.get(REDUCTION_ENV_VAR, _ENV_UNSET)
    try:
        return _main(argv)
    finally:
        if prior is _ENV_UNSET:
            os.environ.pop(REDUCTION_ENV_VAR, None)
        else:
            os.environ[REDUCTION_ENV_VAR] = prior


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--config", required=True,
                    help="the wave's transfer config (explicit_cells), "
                         "relative to --repo-dir or absolute")
    ap.add_argument("--manifest", required=True,
                    help="the wave's Petri manifest (cells = the same cells "
                         "as harness ids; target pins == the config's)")
    ap.add_argument("--defenses-dir", required=True,
                    help="the per-wave object bank (<cell>/object.pt + "
                         "train_budget_meta.json + COMPLETE)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--mode", choices=sorted(EVAL_BY_MODE), required=True)
    ap.add_argument("--eval", default=None,
                    help="geometry-table eval (default: by --mode)")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--driver-max-tokens", type=int, default=None,
                    help="the completion cap the DRIVERS behind this standup will send (the multi-turn "
                         "target's attacks.target_max_new_tokens, 4,096 since 2026-09-11). With a window "
                         "taken from the geometry TABLE, a cap above the entry's measured "
                         "requested_max_tokens REFUSES (GeometryCapExceedsBasis): the table's "
                         "multiturn_attacks 8,704 was measured under a 2,048 cap and clamped the 4,096 "
                         "instrument -- pass --max-model-len explicitly (geometry.MULTITURN_CAP4096_WINDOWS). "
                         "Recorded in endpoint.json driver_max_tokens")
    ap.add_argument("--allow-over-budget", action="store_true")
    ap.add_argument("--reduction-mode", choices=["per_row", "stacked"], default="per_row",
                    help="which STEERING-HOOK REDUCTION the lens serves with, pinned as "
                         "SISTEER_LENS_STACKED_REDUCTIONS (per_row=0, stacked=1). The mt1 wave ran "
                         "entirely on per_row (its objects were served by the vectorized-but-per-row "
                         "hook, main 4133409); the STACKED reductions landing after it is 2-3.5x "
                         "faster at batch >=32 but NOT bit-identical — 1-16 bf16 ulps, deterministic "
                         "per step. These are GREEDY generations compared BETWEEN CELLS, so a "
                         "token-level flip from a reduction-order change inside one attack family "
                         "would put an implementation difference inside the contrast. Every re-run, "
                         "repair shard, top-up and resumed leg of a wave MUST therefore serve the "
                         "mode the wave started on — which is why this defaults to per_row and is "
                         "recorded in the provenance rather than left to an operator")
    ap.add_argument("--require-attested-mode", choices=["per_row", "stacked"], default=None,
                    help="REFUSE unless the workers ATTEST exactly this lens_reduction_mode in their "
                         "install + scheduler-config reports (a pre-landing worker that names no "
                         "mode is a refusal too, never 'per_row by construction'). The full22 wave "
                         "passes --require-attested-mode stacked: every server of the wave must "
                         "prove its arithmetic from the authoritative stamps before any traffic")
    ap.add_argument("--model-key", required=True,
                    help="the geometry table's MODEL key (qwen35_4b) — MANDATORY since the "
                         "per-model budget landed (main 4133409): an unset max_num_seqs is sized "
                         "from THAT model's measured batched-token budget (qwen35_4b 720,896 -> "
                         "attacks 82 x 8704, petri 44 x 16384) instead of the Llama-sized default "
                         "(45 / 24), and the budget guard is held to the model's budget. An "
                         "unknown key REFUSES rather than falling back")
    ap.add_argument("--gpu-mem-util", default="0.9")
    ap.add_argument("--max-logprobs", type=int, default=None,
                    help="vLLM --max-logprobs (the engine default is 20). The controls leg's MMLU "
                         "scoring asks for the driver's MMLU_TOP_LOGPROBS per position and the engine "
                         "refuses a request above its cap ('Requested sample logprobs of 64, which is "
                         "greater than max allowed: 20' — every driver of f22-ctl-gemma4e4b-main 447405, "
                         "2026-09-07, AFTER the provenance step had passed). Unset (the attack lanes, "
                         "which never ask for logprobs) leaves the launch command byte-identical.")
    ap.add_argument("--enable-log-requests", action="store_true",
                    help="pass vLLM's --enable-log-requests so the server logs the RESOLVED "
                         "SamplingParams of every request (one INFO line each; the prompt is a "
                         "separate DEBUG record and is NOT logged at INFO). The only readback of "
                         "the resolved sampler at 0.25.1 — opt in whenever what resolved must be a "
                         "recorded fact rather than an assumption, then run "
                         "scripts/sdfbase/record_resolved_sampler.py over vllm_server.log. Unset "
                         "leaves the launch command byte-identical. Recorded in endpoint.json + "
                         "cells_served.json as enable_log_requests")
    ap.add_argument("--proxy-port", type=int, default=None,
                    help="loopback proxy port (default: a free port)")
    ap.add_argument("--vllm-port", type=int, default=None)
    ap.add_argument("--ready-timeout-s", type=int, default=1800)
    ap.add_argument("--chat-template", default=None, metavar="FILE",
                    help="serve THIS chat template (file bytes) instead of the pinned tokenizer's own; the manifest's "
                         "chat_template_hash must equal its sha16 (the generator pins it per target — D43: the EM "
                         "organisms' native STOCK Llama-3.1 template for target llama8b); recorded in cells_served.json")
    ap.add_argument("--organism", action="append", default=[], metavar="NAME=DIR",
                    help="load a PEFT LoRA adapter DIR as vLLM LoRA module NAME (an EM model organism) and "
                         "declare it in the routing table: every non-lora leg then also serves as "
                         "<vmodel>@NAME, the steering leg on top of the adapter (em-misalign). Repeatable.")
    ap.add_argument("--no-petri-agent-loop", action="store_true",
                    help="do NOT declare the Petri-facing agent-loop entry "
                         "(default: declared — harmless for the attack drivers)")
    # --- the thinking-ON additions (module docstring) ---
    ap.add_argument("--cell", action="append", default=None, metavar="TT_CELL",
                    help="serve ONLY this TrainTest cell (repeatable; `none` implicit). Default: every "
                         "explicit_cells entry. The variant spelling <cell>@thinking declares the "
                         "thinking-ON render for that cell (refused as vacuous on an all-position cell)")
    ap.add_argument("--expect-object-sha256", action="append", default=None, metavar="TT_CELL=SHA256",
                    help="hold the registry export of this fitted cell to the preflight-verified "
                         "object.pt sha256 (repeatable; a bare sha is accepted when exactly one fitted "
                         "cell is served). Recorded in endpoint.json object_artifact_sha256_by_cell")
    ap.add_argument("--reasoning-parser", default=None,
                    help="vLLM reasoning parser (thinking-ON standups; gemma4 only — vllm_server "
                         "refuses any name outside the family's pin). Default: none composed")
    ap.add_argument("--lens-salt-policy", choices=["attested", "request_unique_all"], default="attested",
                    help="the proxy's lens cache regime (capctrl's default `attested`: cell-stable salts "
                         "only for cells the workers attest `always`)")
    ap.add_argument("--upstream-timeout-s", type=float, default=None,
                    help="the proxy's UPSTREAM (proxy -> vLLM) total timeout; unset = the proxy's own "
                         "default (sisteer.capeval.serving.proxy.HOP_TIMEOUT_S, 7,200 s — derived from the "
                         "sampled ladder's 16,384 rung at ~9 tok/s; the old 600 s default 504'd every "
                         "cap-bound GLM draw at the 8192 rung, 2026-09-10). The math500 jobs pass 7200 "
                         "explicitly. Recorded as proxy_upstream_timeout_s (the flag as given, None when "
                         "unset) and upstream_timeout_s (the value in force)")
    ap.add_argument("--thinking-on", action="store_true",
                    help="serve the wave's cells under the served template's THINKING-ON render "
                         "(docs/served_thinking_on.md): the vLLM launch pins "
                         "--default-chat-template-kwargs '{\"enable_thinking\": true}' instead of the "
                         "OFF pin, the template's thinking-ON assistant header is derived and "
                         "CERTIFIED per template before any GPU minute (refused otherwise), every "
                         "header-dependent lens cell is declared serve_out_of_support: ['thinking'] "
                         "in its hashed spec (a positions=all fit declares nothing), and the proxy "
                         "runs thinking_mode: on — every lens-route request must carry "
                         "petri_thinking_on: true. Every fitted object was fit thinking-OFF: this "
                         "serves them OUT OF SUPPORT, as a test. Default: thinking-OFF, byte for byte")
    # --- the serving path and the options the native core takes ---
    # the vocabulary is vllm_server.SERVER_DEFAULT_POLICIES / SERVER_DEFAULT_POLICY_DEFAULT (pinned equal by
    # tests/test_server_default_policy.py); spelled here so the parser builds without importing the serving tree
    ap.add_argument("--server-default-policy", choices=["contract_override", "engine_defaults"], default="contract_override",
                    help="how the served engine resolves a sampling key a REQUEST OMITS (vllm_server.SERVER_DEFAULT_POLICIES): "
                         "contract_override (default; every launch to date, byte for byte — the contracts' vendor row composed as "
                         "--override-generation-config for a pinned model, Qwen3.5) or engine_defaults (nothing composed: vLLM's "
                         "neutral defaults where no generation_config.json ships — what the historical Qwen Petri / BoN / misalignment "
                         "anchors decoded at). Recorded as server_default_policy in endpoint.json, cells_served.json and the proxy's "
                         "/v1/models identity block, on BOTH serving paths (qwen-base-seed-replicates W6, review #1)")
    ap.add_argument("--serving-path", choices=list(SERVING_PATHS), default=None,
                    help="the ENGINE this standup serves on (native-serving-default, 2026-09-13). Absent = the "
                         "entry-point default, registry_adapter.DEFAULT_STANDUP_SERVING_PATH (lora_native): every LoRA "
                         "cell of the view as a vLLM native adapter with CUDA graphs ON through lora_serving_up's wave "
                         "front over the shared native core; PSR / vector cells SERVED on its residual slot at every "
                         "position mode (native-psr-serving); a stacked cell or a position-masked LoRA fit SKIPPED BY "
                         "NAME (cells_skipped), an explicit --cell naming one refuses. lens: today's lens standup, byte "
                         "for byte (common.sh f22_standup passes it under F22_SERVING_PATH=lens)")
    ap.add_argument("--serving-path-selection", default=None, metavar="FILE",
                    help="the shell's serving-path selection record (common.sh f22_select_serving_path: registry_adapter."
                         "select_serving_path's JSON — mode auto|explicit, selected, reason, cells_by_disposition, skipped), "
                         "recorded VERBATIM under `serving_path_selection` in endpoint.json and cells_served.json on BOTH "
                         "paths (the kind-aware default, 2026-09-15). Its `selected` must equal the --serving-path in force "
                         "(a mismatch refuses: the record describes THIS standup). Absent = the records exactly as before")
    ap.add_argument("--text-only-mm", choices=("auto", "on", "off"), default="off",
                    help="native only: compose vLLM's --limit-mm-per-prompt {\"image\":0,\"video\":0} for a multimodal "
                         "target (recorded, unhashed); the lens standup pins it through the manifest")
    ap.add_argument("--kv-derived-seqs", choices=("on", "off"), default="off",
                    help="native only: on a readiness KV-coverage shortfall re-launch ONCE at S = floor(the KV-backed "
                         "concurrency), recorded as a kv_shrink geometry adjustment (plan §1.8)")
    ap.add_argument("--mamba-backend-contract", default="configs/capeval/mamba_backend_contract.json",
                    help="native only: the REVIEWED Nemotron backend contract a nemotron_h standup judges its live "
                         "per-layer worker report against (the ported sisteer_mamba_backend_report RPC); mandatory "
                         "for Nemotron — a missing file makes a Nemotron native standup refuse")
    ap.add_argument("--adapter-bank", default=None,
                    help="native only: the content-addressed adapter bank (default $SHARED/exp/multilora_adapters/"
                         "<model-key>, else $SISTEER_ADAPTER_BANK_ROOT/<model-key>)")
    return ap


def load_serving_path_selection(path, *, serving_path: str):
    """The selection record ``--serving-path-selection`` names (the kind-aware default, 2026-09-15: what
    ``registry_adapter.select_serving_path`` recorded when the shell chose the path), validated against the path in
    force: a dict whose ``mode`` is one of ``registry_adapter.SERVING_PATH_SELECTION_MODES`` and whose ``selected``
    equals ``serving_path`` — the record describes THIS standup, never another launch's verdict. None when no file
    was given (the records exactly as before). Refuses (``SystemExit``) an unreadable or foreign record."""
    if path is None:
        return None
    from sisteer.capeval.petri import registry_adapter as ra
    try:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"[mt-standup] --serving-path-selection {path}: unreadable ({e})")
    if (not isinstance(rec, dict) or rec.get("mode") not in ra.SERVING_PATH_SELECTION_MODES
            or not isinstance(rec.get("selected"), str) or not rec.get("selected")):
        raise SystemExit(f"[mt-standup] --serving-path-selection {path}: not a selection record (needs mode in "
                         f"{ra.SERVING_PATH_SELECTION_MODES} and selected) — got {str(rec)[:200]}")
    if rec["selected"] != serving_path:
        raise SystemExit(f"[mt-standup] --serving-path-selection {path} selected {rec['selected']!r} but this standup "
                         f"serves --serving-path {serving_path} — the record must describe this standup; refusing")
    return rec


def _lens_registry_serving_facts(ep: typing.Mapping[str, typing.Any], out: Path) -> None:
    """Results-registry serving facts for the LENS writer: the SAME hook the native core's
    ``write_endpoint`` calls (``sisteer.registry.hooks.emit_serving_patches``), over the endpoint record just written, reading
    ``<out>/registry_runs.json`` beside it (the launcher's served runs by served key, ``registry_runs_file.py``) — one ``run_patch`` per
    served run with the ATTESTED window as ``context_window_tokens`` (+ the attested S, the config digest, ``model_revision`` under
    ``verify``). Additive and guarded: never raises into the standup, ``endpoint.json`` is byte-identical with emission on or off, nothing
    is written without the runs file (one ``[registry]`` notice when emission is on). Before this the lens body wrote its record and called
    no hook, so a lens standup recorded no serving facts whatever the launcher put beside it; the native path always did (its core's hook)."""
    try:
        from sisteer.registry import hooks as _registry_hooks   # stdlib-only
        observed = json.loads(json.dumps(dict(ep), default=str))
        _registry_hooks.emit_serving_patches(observed, None, out_dir=str(out), script=LENS_SERVING_FACTS_SCRIPT)
    except Exception as e:  # noqa: BLE001 — a lost record never costs a standup
        print(f"[mt-standup] [registry] serving facts not recorded ({type(e).__name__}: {e})", flush=True)


def _main(argv=None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    serving_path = args.serving_path
    if serving_path is None:
        from sisteer.capeval.serving.native_standup import DEFAULT_STANDUP_SERVING_PATH
        serving_path = DEFAULT_STANDUP_SERVING_PATH
    # THE SELECTION RECORD (kind-aware default): validated against the path in force, recorded on both paths
    selection = load_serving_path_selection(args.serving_path_selection, serving_path=serving_path)
    if selection is not None:
        print(f"[mt-standup] serving path selection: {selection['mode']} {selection['selected']} — "
              f"{selection.get('reason')}", flush=True)
    if serving_path != "lens":
        # THE NATIVE DEFAULT: the translation onto the wave front (module docstring)
        return _main_native(args, selection=selection)
    # ---- the LENS escape hatch (--serving-path lens): the body below is today's standup, byte for byte ----
    if args.upstream_timeout_s is not None and not args.upstream_timeout_s > 0:
        raise SystemExit(f"[mt-standup] --upstream-timeout-s must be > 0, got {args.upstream_timeout_s}")

    repo = Path(args.repo_dir).resolve()
    eval_name = args.eval or EVAL_BY_MODE[args.mode]
    # PIN THE REDUCTION MODE before any lens code loads: the worker reads this at startup, and the
    # child server inherits it through launch_server(extra_env=...) below. On PRE-landing serving
    # code the variable is simply unread (per_row is the only path there), so pinning it is
    # harmless now and load-bearing the moment the stacked reductions land.
    reduction_env = {REDUCTION_ENV_VAR: "1" if args.reduction_mode == "stacked" else "0"}
    os.environ.update(reduction_env)
    print(f"[mt-standup] reduction mode: {args.reduction_mode} "
          f"({REDUCTION_ENV_VAR}={reduction_env[REDUCTION_ENV_VAR]})", flush=True)
    from sisteer.capeval.serving.geometry import (GeometryCapExceedsBasis,
                                                  GeometryIdentityBroken,
                                                  GeometryOverBudget)
    try:
        geom = resolve_standup_geometry(eval_name, args.max_model_len,
                                        args.max_num_seqs,
                                        allow_over_budget=args.allow_over_budget,
                                        model_key=args.model_key,
                                        requested_max_tokens=args.driver_max_tokens)
    except (GeometryOverBudget, GeometryIdentityBroken, GeometryCapExceedsBasis) as e:
        raise SystemExit(f"[mt-standup] refusing the geometry: {e}")
    print(f"[mt-standup] geometry: {geom.describe()} (model_key={args.model_key}, "
          f"seqs_source={getattr(geom, 'seqs_source', 'n/a')})", flush=True)

    from transformers import AutoConfig, AutoTokenizer

    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.petri.registry_adapter import traintest_cell_source
    from sisteer.capeval.serving import standup_thinking as st
    from sisteer.capeval.serving import vllm_server as vs
    from sisteer.capeval.serving.proxy import SERVER_DEFAULT_POLICY_KEY
    from sisteer.capeval.serving.lens_steer import export_worker_registry
    from sisteer.capeval.serving.proxy import (HOP_TIMEOUT_S, HTTPBackend, RoutingProxy,
                                               TokenizerAgentLoopRender,
                                               export_routing_table,
                                               load_routing_table, make_app)
    from sisteer.capeval.tiers.petri_runner import resolve_prompt_files
    from sisteer.chat_markers import generation_header_ids
    from sisteer.models import pin_reasoning_template_off

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stop_file = Path(args.stop_file)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if "explicit_cells" not in cfg:
        raise SystemExit(f"[mt-standup] {cfg_path} is not an explicit_cells "
                         "config — this standup serves a wave's enumerated cells")
    model, rev = cfg["model"], cfg["model_revision"]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path
    spec0 = cm.load_petri_spec(manifest_path)
    if (spec0.target_model, spec0.target_revision) != (model, rev):
        raise SystemExit(
            f"[mt-standup] manifest pins {spec0.target_model}@{spec0.target_revision} "
            f"but the config serves {model}@{rev} — one wave, one target")
    # every manifest cell must be one of the config's explicit cells, and
    # every explicit cell must be served: the two enumerations are ONE wave
    tt_of = assert_manifest_matches_config(spec0, cfg)
    # THE SERVING-CONFIG PROVENANCE: written BEFORE the engine starts, beside the log and
    # the workers' attestation (_serving_config_record: both serving paths write it)
    serving_config_record = _serving_config_record(out, cfg_path=cfg_path, manifest_path=manifest_path,
                                                   model_key=args.model_key, model=model, rev=rev, repo=repo)
    # the config's explicit cells — the pool `select_cells` draws the served subset from (main binds this inline beside the
    # manifest/config check; the origin/main merge 3f585dd2 kept our helper form and lost the binding, so every served
    # standup at 4bd493d9 / ec60216b died with NameError before its first request)
    want = list(cfg["explicit_cells"])
    ddir = Path(args.defenses_dir).resolve()
    _KEY_MODEL = served_hf_id_by_key()      # the paper's eight + the onboarded models (never MODEL_HF_IDS alone)
    want_model = _KEY_MODEL.get(args.model_key)
    if want_model is None:
        raise SystemExit(f"[mt-standup] --model-key {args.model_key!r} is not a canonical model key "
                         f"({sorted(_KEY_MODEL)}) — the per-model budget cannot be sized; refusing")
    if want_model is not None and want_model != model:
        raise SystemExit(f"[mt-standup] --model-key {args.model_key} names {want_model} but the "
                         f"config serves {model} — the per-model budget would size the batch from "
                         "the WRONG model's measurement; refusing")
    # --cell: the served subset (default: every explicit cell — byte-identical to before); each
    # requested cell's harness spelling replaces the manifest's list below, variant included
    tt_cells = select_cells(want, args.cell)
    harness_ids = [ra.harness_cell_id(c) for c in tt_cells]
    # the artifact digests the registry export is held to (--expect-object-sha256; capctrl's
    # grammar: <tt_cell>=<sha>, or a bare sha for a single fitted cell). OPTIONAL here — the
    # campaign's own jobs pass none; when given, a digest on a non-object cell or on a cell not
    # served refuses, and the export refuses bytes that differ from the pin
    try:
        expected_digests = expected_artifact_digests_for_cells(
            tt_cells, harness_ids, parse_object_digest_args(args.expect_object_sha256, tt_cells),
            required=False)
    except ValueError as e:
        raise SystemExit(f"[mt-standup] {e}") from None
    print(f"[mt-standup] mode={args.mode} eval={eval_name} model={model}@{rev[:12]} "
          f"cells={tt_cells} bank={ddir}"
          + (f" expected_digests={sorted(expected_digests)}" if expected_digests else ""), flush=True)

    tok = AutoTokenizer.from_pretrained(model, revision=rev)
    tpl_record = serve_chat_template(tok, Path(args.chat_template) if args.chat_template else None)
    pin_reasoning_template_off(tok)
    tpath = out / "chat_template.jinja"
    tpath.write_bytes(tok.chat_template.encode())
    tsha = cm.sha256_hex16(tok.chat_template.encode())
    if tsha != spec0.chat_template_hash:
        raise SystemExit(f"[mt-standup] served chat template sha16 {tsha} != the "
                         f"manifest's chat_template_hash {spec0.chat_template_hash} "
                         "— a stale/wrong pin; refusing")
    tpl_record["sha16"] = tsha
    print(f"[mt-standup] chat template: sha16 {tsha} "
          + (f"(OVERRIDE {tpl_record['override']}; the pinned tokenizer's own template hashes "
             f"{tpl_record['tokenizer_chat_template_sha16']})" if tpl_record["override"] else "(the pinned tokenizer's own)"),
          flush=True)
    mcfg = AutoConfig.from_pretrained(model, revision=rev)
    hidden = getattr(mcfg, "hidden_size", None) or getattr(
        getattr(mcfg, "text_config", None), "hidden_size", None)
    if not hidden:
        raise SystemExit(f"[mt-standup] cannot resolve hidden_size for {model}")

    # the LENS escape hatch strips a manifest's --enable-lora; the native path is the default
    # (--serving-path lora_native, the branch above) and strips --enforce-eager instead
    spec = dataclasses.replace(
        spec0, serving_launch_flags=[f for f in spec0.serving_launch_flags
                                     if f != "--enable-lora"],
        # --cell: the manifest's list narrowed to the requested cells' harness spellings
        # (variant included); without --cell the manifest's own list, untouched — unless the config lists no `none` (a steered-wave
        # view, s3r-fix): the served set then carries the implicit base route the manifest does not list, so the table enumerates it
        **({"cells": harness_ids} if (args.cell or "none" not in want) else {}))
    src = traintest_cell_source(spec, repo_root=repo, config_path=cfg_path,
                                defenses_dir=ddir)
    # THE STANDUP-WIDE THINKING MODE (--thinking-on; resolve_thinking), before any GPU minute:
    # the served template's ON header certified per template (or the standup refused), the
    # HEADER-DEPENDENT lens cells declared in their hashed specs (a NEW served identity), the
    # certified ON header to export. OFF records the certification and changes nothing.
    off_header_ids = tuple(generation_header_ids(tok))
    thinking = resolve_thinking(tok, spec, src, thinking_on=args.thinking_on,
                                model_key=args.model_key, off_header_ids=off_header_ids)
    print(f"[mt-standup] thinking mode: {thinking.mode}; certification ({args.model_key}): verdict="
          f"{thinking.certification.get('verdict')} off_header_len={len(off_header_ids)} "
          f"on_ids={thinking.certification.get('on_ids')} moves={thinking.certification.get('moves')} "
          f"checks={json.dumps(thinking.certification.get('checks'), default=str)}", flush=True)
    print(f"[mt-standup] lens header dependence: {thinking.header_dependence}; --thinking-on declared "
          f"cells: {list(thinking.declared_cells)}; certified ON header: {thinking.on_header_ids}",
          flush=True)
    if thinking.declared_cells:
        # DECLARED BY NAME: a declaration must be visible in the cell's id (the
        # enumeration refuses a spec that declares what its name does not), so the
        # header-dependent cells take the `<cell>@thinking` spelling — exactly what
        # `--cell <id>@thinking` spells — and the source is rebuilt so the adapter
        # applies the overlay and hashes it into the intervention hash
        spec = dataclasses.replace(spec, cells=declared_cell_spellings(spec, thinking.declared_cells))
        harness_ids = list(spec.cells)
        src = traintest_cell_source(spec, repo_root=repo, config_path=cfg_path,
                                    defenses_dir=ddir)
    behavior_files = {}
    for key in cm.petri_expected_behavior_keys(spec):
        bpath = repo / "configs" / "capeval" / "petri" / key / "BEHAVIOR.md"
        if not bpath.is_file():
            raise SystemExit(f"[mt-standup] missing behavior definition {bpath}")
        behavior_files[key] = bpath.read_bytes()
    prompt_files = resolve_prompt_files(
        spec, override_root=repo / "configs/capeval/petri/prompts")
    cells = dc.enumerate_cells(spec, src, behavior_files=behavior_files,
                               prompt_files=prompt_files)
    if cells.absent:
        raise SystemExit("[mt-standup] ABSENT cells (bank incomplete?): "
                         f"{[(a.cell_id, a.reason) for a in cells.absent]}")
    # THE THINKING-HEADER FACT, always: the OFF header stays the lens's
    # primary header (the fits' render, under pin_reasoning_template_off above); a cell
    # DECLARING the thinking-ON render (@thinking) gets the SECOND header — derived hard; on a
    # template whose switch does not move the header (gemma-4) it EQUALS the first and is
    # recorded so, admitted rather than refused; with no declaring cell the registry carries
    # the OFF header alone (rule 5), and a template whose ON header the derivation cannot
    # produce (the pinned Nemotron-3) records the PARTIAL fact + the reason VERBATIM — never
    # a measurement table's ids in its place
    header_ids = tuple(generation_header_ids(tok))
    specs_by_cell = st.lens_resolved_specs_by_cell(cells, src)
    lens_specs = list(specs_by_cell.values())
    # RULE 3 AT THE STANDUP (F11): a `@thinking` declaration on a cell with no header-located
    # served op (every sdfbase object: positions=all) is VACUOUS — the worker's adapter would refuse
    # it at install, after the launch was paid for; refuse it HERE by name instead
    vacuous = st.vacuous_declarations(specs_by_cell)
    if vacuous:
        raise SystemExit("[mt-standup] thinking-ON declaration refused (lens rule 3): the declaration "
                         f"is VACUOUS on {vacuous} — no served op of the cell is header-located "
                         "(positions=all / last: the mask does not move with the render), so a "
                         "declared out-of-support render would change nothing; serve the cell under "
                         "its plain spelling (the render is certified live on every route)")
    declares_thinking = st.declares_thinking(lens_specs)
    try:
        on_header_ids = st.thinking_on_header_ids(tok, header_ids, declares_thinking=declares_thinking)
        thinking_header, thinking_header_error = st.standup_thinking_header(
            tok, declares_thinking=declares_thinking)
    except Exception as e:  # noqa: BLE001 — a declaring cell whose header cannot certify refuses HERE
        raise SystemExit(f"[mt-standup] thinking-ON declaration refused before any GPU minute: "
                         f"{type(e).__name__}: {e}") from None
    if thinking_header_error is not None:
        print("[mt-standup] thinking_header: ON certification UNAVAILABLE "
              f"({thinking_header_error}); serving thinking-OFF; a declared thinking-ON request on a "
              "lens route refuses thinking_on_unattested", flush=True)
    # --thinking-on with declared cells: the per-cell derivation above (the registry's header, the
    # one every worker cross-checks at preflight) and the standup-wide certification must name the
    # SAME ids — two derivations that disagree are refused, never picked between
    if (args.thinking_on and thinking.on_header_ids is not None and on_header_ids is not None
            and tuple(on_header_ids) != tuple(thinking.on_header_ids)):
        raise SystemExit("[mt-standup] --thinking-on: the certified thinking-ON header "
                         f"{list(thinking.on_header_ids)} (chat_markers.thinking_on_certification) != "
                         f"the registry's derivation {list(on_header_ids)} "
                         "(standup_thinking.thinking_on_header_ids) — refusing to export a header the "
                         "workers would not cross-check to")
    # the fact the routing table and the proxy carry: under --thinking-on the standup-wide
    # certification's (an ON proxy needs on_ids; the certification is what admitted the standup),
    # else the per-cell derivation's, exactly as before
    if args.thinking_on:
        table_header, table_header_error = thinking.header, thinking.header_error
    else:
        table_header, table_header_error = thinking_header, thinking_header_error
    reg_dir = out / "lens_registry"
    try:
        reg_entries = export_worker_registry(
            cells, src, reg_dir,
            assistant_header_ids=header_ids,
            zero_op_hidden_size=int(hidden),
            **({"assistant_header_ids_thinking_on": on_header_ids} if on_header_ids is not None else {}),
            **({"expected_artifact_sha256": expected_digests} if expected_digests else {}))
    except Exception as e:  # noqa: BLE001 — the lens's own refusals (rule 3 on a vacuous @thinking,
        # a digest mismatch, an unservable kind) surface VERBATIM, before any launch is composed
        raise SystemExit(f"[mt-standup] registry export refused: {type(e).__name__}: {e}") from None
    table_path = out / "routing_table.json"
    organisms: dict = {}
    for item in args.organism:
        name, sep, d = item.partition("=")
        if not sep or not name or not d:
            raise SystemExit(f"[mt-standup] --organism wants NAME=DIR, got {item!r}")
        adir = Path(d)
        if not (adir / "adapter_config.json").is_file():
            raise SystemExit(f"[mt-standup] --organism {name}: {adir} has no adapter_config.json")
        if name in organisms or "@" in name:
            raise SystemExit(f"[mt-standup] --organism name {name!r} duplicate or carries '@'")
        organisms[name] = adir
    # REGENERATED on every start (never reused): a stale table lacks the facts a newer proxy
    # reads and would fail closed as thinking_on_undeclared
    # the declarations the workers MUST attest: every exported entry whose spec declares, under
    # either spelling (--thinking-on's header-dependent cells, --cell <id>@thinking) + the declared
    # twin; the resolution is reconciled to what was ACTUALLY exported before it is recorded
    want_decls = expected_declarations(reg_entries)
    thinking = thinking._replace(
        header=table_header, header_error=table_header_error,
        on_header_ids=(tuple(on_header_ids) if on_header_ids is not None else None),
        declared_cells=tuple(sorted(want_decls)))
    export_routing_table(cells, src, table_path, base_model=model,
                         serving_mode="interleaved", base_revision=rev,
                         thinking_header=table_header,
                         thinking_header_error=table_header_error,
                         # EM organisms (--organism): declared in the table so the proxy
                         # serves <vmodel>@<organism> for every non-lora leg
                         organisms=list(organisms))
    if organisms:
        print(f"[mt-standup] organisms (vLLM LoRA modules; every non-lora leg also served as "
              f"<vmodel>@<name>): { {k: str(v) for k, v in organisms.items()} }", flush=True)
    served = [c.cell_id for c in cells.canonical] + [c.cell_id for c in cells.controls]
    print(f"[mt-standup] registry + routing table exported: {len(served)} legs "
          f"{served}; hidden={hidden}; declares_thinking={declares_thinking} "
          f"thinking_header={json.dumps(thinking_header)}", flush=True)
    # the budget's EVIDENCE beside the resolved geometry: acceptance validates the
    # RECORDED budget and its basis (source, first_refused, measured_on), never a
    # literal — a provisional ceiling says so in every record
    from sisteer.capeval.serving.geometry import budget_for as _budget_for
    _rb = _budget_for(args.model_key)
    budget_record = {"budget_basis": dataclasses.asdict(_rb.entry.basis),
                     "budget_provisional": _rb.provisional,
                     "budget_source": _rb.source,
                     "budget_entry_model_key": _rb.entry.model_key}
    (out / "cells_served.json").write_text(json.dumps({
        "mode": args.mode, "eval": eval_name, "manifest": str(manifest_path),
        **budget_record,
        "reduction_mode": args.reduction_mode,
        REDUCTION_ENV_VAR: reduction_env[REDUCTION_ENV_VAR],
        "require_attested_mode": args.require_attested_mode,
        "max_logprobs": args.max_logprobs,
        "enable_log_requests": args.enable_log_requests,
        "written_at": _utc_now(),
        "config": str(cfg_path), "serving_config_sha256": serving_config_record["serving_config_sha256"],
        "manifest_sha256": serving_config_record["manifest_sha256"], "defenses_dir": str(ddir),
        # the --cell spellings this standup was asked for (variant included) and their harness ids
        "cells_requested": list(tt_cells), "harness_cells": list(harness_ids),
        # how the shell chose this path (kind-aware default, 2026-09-15) — only when the caller recorded one
        **({ra.SERVING_PATH_SELECTION_KEY: selection} if selection is not None else {}),
        # the thinking regime (thinking_record): mode, header fact (+ error), certification,
        # declared cells — the attested fields are stamped after launch
        **thinking_record(thinking),
        "expected_object_sha256_by_cell": dict(expected_digests),
        "reasoning_parser": args.reasoning_parser,
        # THE SERVER-DEFAULT SAMPLING POLICY (--server-default-policy): the name, so a store's sampler
        # provenance reads off this record without the server log
        "server_default_policy": args.server_default_policy,
        "upstream_timeout_s": args.upstream_timeout_s,
        "canonical": [dataclasses.asdict(c) for c in cells.canonical],
        "controls": [dataclasses.asdict(c) for c in cells.controls],
        "bank_objects": _bank_objects(ddir, list(tt_of.values())),
        "organisms": {k: str(v) for k, v in organisms.items()},
        # the served chat template: the pinned tokenizer's own, or a declared override
        "chat_template": tpl_record},
        indent=2, default=str))

    vllm_port = args.vllm_port or free_port()
    try:
        launch = vs.build_server_launch(
            spec, port=vllm_port, lens_registry_dir=reg_dir,
            chat_template_path=tpath, max_model_len=geom.max_model_len,
            max_num_seqs=geom.max_num_seqs, tool_calls=(args.mode == "petri"),
            # the THINKING-ON render pin (--thinking-on: explicit even where the template defaults
            # ON); False = the OFF pin every launch to date composed
            thinking_on=args.thinking_on,
            # EM organisms as vLLM LoRA modules (--organism): the overlay legs
            lora_modules=organisms,
            # thinking-ON standups only (gemma4); the composer refuses any other family's parser
            reasoning_parser=args.reasoning_parser,
            # THE SERVER-DEFAULT SAMPLING POLICY (--server-default-policy): recorded below
            server_default_policy=args.server_default_policy)
    except vs.VllmServerError as e:
        raise SystemExit(f"[mt-standup] launch composition refused: {e}") from None
    print(f"[mt-standup] composed render pin: --default-chat-template-kwargs "
          f"{launch.default_chat_template_kwargs!r}", flush=True)
    launch = dataclasses.replace(
        launch, cmd=tuple(launch.cmd) + launch_cmd_extras(args.gpu_mem_util, args.max_logprobs, args.enable_log_requests),
        # into the ATTESTATION too: the proxy records launch.env, so the served identity carries
        # which reduction served it — a later reader can tell which side any shard came from
        env={**dict(launch.env), **reduction_env})
    print(f"[mt-standup] launching vLLM on {launch.base_url} "
          f"({geom.max_num_seqs} x {geom.max_model_len}) node={socket.gethostname()}",
          flush=True)
    server = vs.launch_server(launch, log_path=out / "vllm_server.log",
                              ready_timeout_s=args.ready_timeout_s,
                              install_timeout_s=900, extra_env=reduction_env)
    # READ AND RECORD the SERVED reduction mode from the server's own attestation record — the
    # validated per-worker install reports launch_server just wrote — and REFUSE when it is not
    # the mode this wave is pinned to. Two facts, kept apart on purpose: what we PINNED
    # (reduction_env, also carried into the proxy's launch attestation above) and what the
    # workers ATTESTED (`lens_reduction_mode`). Pre-landing workers name no mode; that ABSENCE is
    # recorded as absent (per_row is the only path such a worker has), never rewritten as a mode.
    att_path = out / vs.STEERING_ATTESTATION_FILE
    try:
        attestation = json.loads(att_path.read_text())
    except (OSError, ValueError) as e:
        vs._shutdown_child(server)
        raise SystemExit(f"[mt-standup] cannot read {att_path}: {e} — a lens server without a "
                         "readable attestation must not receive audit traffic")
    try:
        lens_mode, lens_mode_source = served_reduction_mode_strict(attestation)
    except ValueError as e:
        vs._shutdown_child(server)
        raise SystemExit(f"[mt-standup] served attestation is inconsistent: {e}")
    if args.require_attested_mode is not None and lens_mode != args.require_attested_mode:
        vs._shutdown_child(server)
        raise SystemExit(f"[mt-standup] --require-attested-mode {args.require_attested_mode}: the "
                         f"workers attest {LENS_REDUCTION_FIELD}={lens_mode!r} (source "
                         f"{lens_mode_source}) — refusing to serve without the authoritative stamp "
                         "(a pre-landing worker naming no mode is not proof of anything)")
    if lens_mode is not None and lens_mode != args.reduction_mode:
        vs._shutdown_child(server)
        raise SystemExit(f"[mt-standup] the workers attest {LENS_REDUCTION_FIELD}={lens_mode!r} "
                         f"({lens_mode_source}) but --reduction-mode {args.reduction_mode} is "
                         "what this wave is pinned to — refusing to serve the other side of the "
                         "stacked-reductions boundary")
    # THE RESOLVED GEOMETRY MUST BE THE ATTESTED ONE (the full22 geometry change, 2026-09-06): the
    # workers' scheduler-config reports carry vLLM's LIVE max_num_seqs / max_model_len; a width the
    # engine quietly lowered is a different serving geometry and must never pass as the requested one.
    try:
        att_geom = attested_scheduler_geometry(attestation)
    except ValueError as e:
        vs._shutdown_child(server)
        raise SystemExit(f"[mt-standup] served attestation is inconsistent: {e}")
    want_geom = (int(geom.max_num_seqs), int(geom.max_model_len))
    if att_geom is None and args.require_attested_mode is not None:
        vs._shutdown_child(server)
        raise SystemExit("[mt-standup] the workers attest no scheduler geometry (max_num_seqs / "
                         "max_model_len absent from every sisteer_scheduler_config report) — under "
                         "--require-attested-mode the resolved geometry must be attested, not assumed")
    if att_geom is not None and att_geom != want_geom:
        vs._shutdown_child(server)
        raise SystemExit(f"[mt-standup] resolved geometry {want_geom[0]} x {want_geom[1]} but the workers "
                         f"attest {att_geom[0]} x {att_geom[1]} — a silently clamped geometry; refusing")
    print(f"[mt-standup] attested scheduler geometry: "
          f"{att_geom if att_geom is not None else '(absent)'} (resolved {want_geom})", flush=True)
    print(f"[mt-standup] served {LENS_REDUCTION_FIELD}="
          f"{lens_mode if lens_mode is not None else '(absent: pre-landing worker, per_row by construction)'} "
          f"source={lens_mode_source} pinned={args.reduction_mode}", flush=True)
    # THE ROTARY THE ENGINE ACTUALLY RUNS (mrope_attest above): asked of every worker over the same
    # authenticated RPC route as the install + scheduler reports, recorded next to them, and REFUSED
    # when the reports do not prove a non-neox M-RoPE model (GLM-4V) is off the neox-only kernel.
    mrope_att = mrope_attest(launch, args.model_key)
    print(f"[mt-standup] mrope attestation: ok={mrope_att['ok']} "
          f"reports={json.dumps(mrope_att['reports'], default=str)[:600]}", flush=True)
    if not mrope_att["ok"]:
        vs._shutdown_child(server)
        raise SystemExit("[mt-standup] M-RoPE dispatch attestation FAILED — refusing to serve a rotary "
                         "the workers cannot prove correct:\n  - " + "\n  - ".join(mrope_att["problems"]))
    # THE THINKING REGIME THE WORKERS ATTESTED (thinking_attestation_problems): every worker's install
    # report names the ON header it loaded and cross-checked (assistant_header_ids_thinking_on) and the
    # per-cell declarations bound to its loaded ops (lens_serve_declarations); both must equal what this
    # standup exported, or the standup refuses to serve. An OFF standup records what pre-landing workers
    # leave absent and refuses only a disagreement.
    problems, att_on_header, att_on_header_err, att_decls, att_decls_err = \
        thinking_attestation_problems(attestation, mode=thinking.mode,
                                      want_header=thinking.on_header_ids, expected_decls=want_decls)
    if problems:
        vs._shutdown_child(server)
        raise SystemExit("[mt-standup] thinking-mode attestation FAILED — refusing to serve a regime the "
                         "workers do not attest:\n  - " + "\n  - ".join(problems))
    print(f"[mt-standup] thinking attestation: mode={thinking.mode} attested ON header="
          f"{att_on_header if att_on_header_err is None else f'(absent: {att_on_header_err})'} "
          f"declarations={att_decls if att_decls_err is None else f'(absent: {att_decls_err})'}",
          flush=True)
    thinking_rec = thinking_record(thinking, launch=launch, attested_header=att_on_header,
                                   attested_header_error=att_on_header_err,
                                   attested_declarations=att_decls,
                                   attested_declarations_error=att_decls_err)
    # cells_served.json was written before the server came up; stamp the SERVED mode into it too so
    # a reader of either record sees which side the legs were served from.
    cs_path = out / "cells_served.json"
    cs = json.loads(cs_path.read_text())
    cs[LENS_REDUCTION_FIELD] = lens_mode
    cs["lens_reduction_source"] = lens_mode_source
    cs["attested_max_num_seqs"] = att_geom[0] if att_geom else None
    cs["attested_max_model_len"] = att_geom[1] if att_geom else None
    cs["mrope_attestation"] = mrope_att
    cs.update(thinking_rec)
    cs_tmp = cs_path.with_suffix(".json.tmp")
    cs_tmp.write_text(json.dumps(cs, indent=2, default=str))
    cs_tmp.replace(cs_path)
    proxy_loop = None
    rc = 0
    try:
        table = load_routing_table(table_path)
        # the attested per-cell facts the proxy enforces (standup_thinking, capctrl's order): the
        # resumed-prefill policy map (cell-stable salts only for cells the workers attest
        # `always`), the serve declarations (EVEN WHEN EMPTY — an attested {} is the stated "no
        # declared cell"), the table's thinking-header fact + its error, the salt policy, the
        # per-cell digests, and the table's lens_served_positions stamp when the
        # constructor on this tree takes it — recorded either way in endpoint.json
        proxy_kwargs, att_facts = st.routing_proxy_attestation_kwargs(
            attestation_path=att_path, table=table, thinking_header_error=thinking_header_error,
            lens_salt_policy=args.lens_salt_policy, expected_digests=expected_digests,
            routing_proxy_cls=RoutingProxy)
        print(f"[mt-standup] proxy facts: lens_serve_declarations="
              f"{att_facts.get('lens_serve_declarations', None)!r} "
              f"(error {att_facts['lens_serve_declarations_error']!r}) "
              f"resumed_prefill_policy_cells={att_facts['resumed_prefill_policy_cells']!r} "
              f"lens_served_positions={att_facts[st.LENS_SERVED_POSITIONS_KEY]!r} "
              f"({att_facts[st.LENS_SERVED_POSITIONS_PASSTHROUGH_KEY]})", flush=True)
        proxy = RoutingProxy(
            table["routes"],
            HTTPBackend(launch.base_url, api_key=launch.api_key,
                        # the proxy's upstream timeout ONLY when --upstream-timeout-s named one
                        # (this experiment: 7200); the proxy's default otherwise
                        **st.backend_timeout_kwargs(args.upstream_timeout_s)),
            base_model=model, serving_mode="interleaved",
            prompted_prefixes=table["prompted_prefixes"],
            exact_replay_strategy=None,
            # the env we PINNED plus the mode the workers ATTESTED: the /v1/models `petri_proxy`
            # identity block echoes lens_reduction_mode only when the launch attestation carries
            # that exact key, so a client reads the served arithmetic from the proxy itself
            launch_attestation={**dict(launch.env),
                                **({LENS_REDUCTION_FIELD: lens_mode} if lens_mode is not None else {}),
                                # the server-default sampling policy, echoed on /v1/models petri_proxy (proxy.SERVER_DEFAULT_POLICY_KEY)
                                SERVER_DEFAULT_POLICY_KEY: getattr(launch, "server_default_policy", vs.SERVER_DEFAULT_POLICY_DEFAULT)},
            agent_loop_render=TokenizerAgentLoopRender(tok),
            petri_agent_loop_entry=not args.no_petri_agent_loop,
            base_revision=rev,
            # EM organisms (--organism): the overlay legs <vmodel>@<organism>
            organisms=list(organisms),
            # THE THINKING MODE (proxy.THINKING_MODES): an ON proxy refuses to start without the
            # certified fact and the attested declarations in proxy_kwargs, and requires
            # petri_thinking_on: true on every lens-route request
            thinking_mode=thinking.mode,
            **proxy_kwargs)
        proxy_port = args.proxy_port or free_port()
        proxy_loop = _start_proxy_thread(make_app(proxy), "127.0.0.1", proxy_port)
        proxy_url = f"http://127.0.0.1:{proxy_port}/v1"
        ep = {"proxy_url": proxy_url, "proxy_port": proxy_port,
              "vllm_port": vllm_port, "routing_table": str(table_path),
              "mode": args.mode, "eval": eval_name, "model": model,
              "model_key": args.model_key,
              "reduction_mode": args.reduction_mode,
              REDUCTION_ENV_VAR: reduction_env[REDUCTION_ENV_VAR],
              LENS_REDUCTION_FIELD: lens_mode,
              "lens_reduction_source": lens_mode_source,
              "require_attested_mode": args.require_attested_mode,
              "max_logprobs": args.max_logprobs,
              "enable_log_requests": args.enable_log_requests,
              "written_at": _utc_now(),
              "steering_attestation": str(att_path),
              "seqs_source": getattr(geom, "seqs_source", None),
              "model_revision": rev, "cells": list(tt_of.values()),
              "harness_cells": list(spec0.cells),
              "served_models": list(proxy.virtual_models()),
              "pid": os.getpid(), "node": socket.gethostname(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "max_model_len": geom.max_model_len,
              "max_num_seqs": geom.max_num_seqs,
              # what the workers ATTESTED (vLLM's live scheduler config) next to what was resolved —
              # equal by construction here (a mismatch refused above), recorded so a reader never
              # has to infer the served width from throughput
              "attested_max_num_seqs": att_geom[0] if att_geom else None,
              "attested_max_model_len": att_geom[1] if att_geom else None,
              # the rotary the workers attested at readiness (mrope_attest); routed_native_calls is 0
              # here by construction — mrope_attestation.json records the same report again at STOP,
              # after traffic, so the route can be seen to have been EXERCISED
              "mrope_attestation": mrope_att,
              "geometry": geom.as_dict(),
              # the serving-config provenance: what THIS leg serves, recorded beside the geometry
              "serving_config": str(cfg_path),
              "serving_config_sha256": serving_config_record["serving_config_sha256"],
              "manifest_sha256": serving_config_record["manifest_sha256"],
              **budget_record,
              "petri_agent_loop_entry": not args.no_petri_agent_loop,
              "tool_calls": args.mode == "petri",
              "bank_objects": _bank_objects(ddir, list(tt_of.values())),
              "started": time.time(),
              # --- the keys the tinycap driver's standup-record readers consume:
              # _served_identity's agent_loop_entry / reasoning_parser, the ledger and
              # Inspect provider's proxy_token (this loopback proxy runs unauthenticated: None,
              # the client falls back to its default bearer), check_serve_attestation's
              # attestation facts, the upstream timeout, the served template's digest
              "agent_loop_entry": not args.no_petri_agent_loop,
              "reasoning_parser": args.reasoning_parser,
              "proxy_token": None,
              "proxy_upstream_timeout_s": args.upstream_timeout_s,     # argparse type=float; None when unset
              # the proxy -> vLLM hop timeout IN FORCE (the flag, else proxy.HOP_TIMEOUT_S): the sampled
              # ladder's hop_timeout_guard sizes it against the rung, and f22_standup records it
              "upstream_timeout_s": (float(args.upstream_timeout_s) if args.upstream_timeout_s is not None
                                     else HOP_TIMEOUT_S),
              # the thinking regime (thinking_record): mode, launch pin, header fact + certification,
              # declared cells, and what the workers attested — equal to the export by construction here
              **thinking_rec,
              # the drivers' completion cap the geometry was checked against (None = not stated)
              "driver_max_tokens": args.driver_max_tokens,
              # THE SERVER-SIDE SAMPLING DEFAULT: the --override-generation-config JSON
              # build_server_launch composed for a model whose pinned snapshot ships no
              # generation_config.json (Qwen3.5 -> contracts.SERVER_DEFAULT_SAMPLING_BY_MODEL);
              # None = the model's own shipped file governs an omit-params request. A multi-turn
              # run under target_decode.GENERATION_CONFIG decodes at exactly this, so the record
              # of what "the model's own decode" WAS lives here, beside the geometry.
              "override_generation_config": getattr(launch, "override_generation_config", None),
              # ... and the POLICY it was composed under (vllm_server.SERVER_DEFAULT_POLICIES; --server-default-policy):
              # `engine_defaults` = None above BY CHOICE, `contract_override` = the pinned row or None for an unpinned model
              "server_default_policy": getattr(launch, "server_default_policy", vs.SERVER_DEFAULT_POLICY_DEFAULT),
              "chat_template_hash": tsha,
              "cells_requested": list(tt_cells),
              **({ra.SERVING_PATH_SELECTION_KEY: selection} if selection is not None else {}),
              "lens_reduction_mode_error": (None if lens_mode is not None
                                            else f"no worker names {LENS_REDUCTION_FIELD} ({lens_mode_source})"),
              # THE LEGACY SCALAR (capctrl's object_artifact_sha256, its own rule): the single distinct
              # digest of the per-cell map when there is exactly one, null otherwise
              "object_artifact_sha256": legacy_object_sha256(expected_digests),
              **att_facts}
        tmp = out / "endpoint.json.tmp"
        tmp.write_text(json.dumps(ep, indent=2, default=str))
        tmp.rename(out / "endpoint.json")   # atomic: pollers never see a torso
        _lens_registry_serving_facts(ep, out)   # the registry's serving facts, from the record just written
        print(f"[mt-standup] READY: proxy {proxy_url} serving "
              f"{len(ep['served_models'])} vmodels; waiting for {stop_file}",
              flush=True)
        rc = wait_until_stop(stop_file, server)
    finally:
        if proxy_loop is not None:
            try:
                proxy_loop.call_soon_threadsafe(proxy_loop.stop)
            except Exception:  # noqa: BLE001
                pass
        try:
            at_stop = mrope_attest(launch, args.model_key, timeout_s=60.0)
            (out / "mrope_attestation.json").write_text(json.dumps(
                {"model_key": args.model_key, "at_ready": mrope_att, "at_stop": at_stop,
                 "note": ("at_stop is asked AFTER the legs' traffic: for a non-neox M-RoPE model its "
                          "routed_native_calls > 0 is the proof the patched route was exercised; "
                          "at_ready's is 0 by construction")}, indent=2, default=str))
            print(f"[mt-standup] mrope attestation at STOP: ok={at_stop['ok']} "
                  f"routed_native_calls={[r.get('routed_native_calls') for r in at_stop['reports'] if isinstance(r, dict)]}",
                  flush=True)
        except Exception as e:  # noqa: BLE001 — teardown must never be refused
            print(f"[mt-standup] mrope attestation at STOP unavailable: {type(e).__name__}: {e}", flush=True)
        try:
            vs._shutdown_child(server)
        except Exception:  # noqa: BLE001
            pass
        try:
            (out / "endpoint.json").unlink()
        except FileNotFoundError:
            pass
    print("[mt-standup] down.", flush=True)
    return rc


def _main_native(args, selection=None) -> int:
    """The NATIVE translation (module docstring): every lens-only pin refused by name (exit 2), the serving-config
    provenance written, the served objects' digests forwarded or derived from the view manifest + the pins, then
    ``lora_serving_up.bring_up_lora`` (the wave front over the shared native core) with every native option forwarded,
    the stop wait and the teardown. Records: the core's ``endpoint.json`` / ``cells_served.json`` (``engine ==
    lora_native``, never ``lens_reduction_mode``) plus mt's provenance keys beside them — ``selection`` (the shell's
    serving-path selection record, ``load_serving_path_selection``) among them when the caller recorded one."""
    sdfbase = str(Path(__file__).resolve().parent)
    if sdfbase not in sys.path:
        sys.path.insert(0, sdfbase)
    import lora_serving_up as lsu
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.petri.registry_adapter import traintest_cell_source
    from sisteer.capeval.serving import mamba_backend as mb
    from sisteer.capeval.serving import vllm_server as vs
    _KEY_MODEL_NATIVE = served_hf_id_by_key()      # the paper's eight + the onboarded models (never MODEL_HF_IDS alone)
    from sisteer.capeval.serving.geometry import budget_for as _budget_for_native

    def refuse(msg: str) -> int:
        print(f"[mt-standup] refused (exit 2): {msg}", flush=True)
        return 2

    # the lens facts the native engine cannot attest (a REAL limitation: no lens hook) — REFUSED by name; the defaults
    # (per_row / unset / attested) are accepted and recorded not_applicable by the core
    if args.reduction_mode != "per_row":
        return refuse(f"--reduction-mode {args.reduction_mode} is a LENS reduction the native engine has no hook for; "
                      "pass --serving-path lens (the lens standup pins and attests it) — under lora_native the engine "
                      "attests `engine`, not a reduction mode")
    if args.require_attested_mode is not None:
        return refuse(f"--require-attested-mode {args.require_attested_mode} asks for a LENS attestation the native "
                      "engine cannot give; pass --serving-path lens")
    if args.lens_salt_policy != "attested":
        return refuse(f"--lens-salt-policy {args.lens_salt_policy} is a LENS cache regime with no native counterpart; "
                      "pass --serving-path lens")
    if args.upstream_timeout_s is not None and not args.upstream_timeout_s > 0:
        return refuse(f"--upstream-timeout-s must be > 0, got {args.upstream_timeout_s}")
    repo = Path(args.repo_dir).resolve()
    eval_name = args.eval or EVAL_BY_MODE[args.mode]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stop_file = Path(args.stop_file)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if "explicit_cells" not in cfg:
        return refuse(f"{cfg_path} is not an explicit_cells config — this standup serves a wave's enumerated cells")
    model, rev = cfg["model"], cfg["model_revision"]
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path
    spec0 = cm.load_petri_spec(manifest_path)
    if (spec0.target_model, spec0.target_revision) != (model, rev):
        return refuse(f"manifest pins {spec0.target_model}@{spec0.target_revision} but the config serves {model}@{rev} "
                      "— one wave, one target")
    tt_of = assert_manifest_matches_config(spec0, cfg)
    want_model = _KEY_MODEL_NATIVE.get(args.model_key)
    if want_model is None:
        return refuse(f"--model-key {args.model_key!r} is not a canonical model key ({sorted(_KEY_MODEL_NATIVE)})")
    if want_model != model:
        return refuse(f"--model-key {args.model_key} names {want_model} but the config serves {model}")
    serving_config_record = _serving_config_record(out, cfg_path=cfg_path, manifest_path=manifest_path,
                                                   model_key=args.model_key, model=model, rev=rev, repo=repo)
    ddir = Path(args.defenses_dir).resolve()
    want = list(cfg["explicit_cells"])
    try:
        tt_cells, prompted_admitted = lsu.select_cells_lora(want, args.cell, si_vocabulary=lsu.config_si_vocabulary(cfg))
        harness_ids = [ra.harness_cell_id(c) for c in tt_cells]
    except (lsu.StandupRefused, ra.RegistryAdapterError) as e:
        return refuse(str(e))
    # THE DIGESTS: forwarded when given; else derived per served LoRA cell from the view manifest + the pins
    # (the cells the partition skips — a stacked cell, a position-masked LoRA fit — are never asked for one)
    if args.expect_object_sha256:
        digests, digests_source = list(args.expect_object_sha256), lsu.ns.DIGESTS_SOURCE_CLI
    else:
        kept, _dropped = lsu.effective_launch_flags([f for f in spec0.serving_launch_flags if f != "--enable-lora"])
        spec_all = dataclasses.replace(spec0, cells=harness_ids, serving_launch_flags=kept)
        try:
            lens_src = traintest_cell_source(spec_all, repo_root=repo, config_path=cfg_path, defenses_dir=ddir)
            served_ids, _skipped = lsu.ns.partition_cells_for_native(harness_ids, lens_src)
        except (ra.RegistryAdapterError, ValueError) as e:
            return refuse(str(e))
        served_tt = [tt for tt, h in zip(tt_cells, harness_ids) if h in set(served_ids)]
        pins_path = Path(PINS_FILE)
        if not pins_path.is_absolute():
            pins_path = repo / pins_path
        try:
            digests = view_manifest_digests(ddir, pins_path=pins_path, target=args.model_key, cells=served_tt,
                                            harness_ids=[h for h in harness_ids if h in set(served_ids)], repo_root=repo)
        except SystemExit as e:
            return refuse(str(e))
        digests_source = lsu.ns.DIGESTS_SOURCE_VIEW_PINS
    contract_path = Path(args.mamba_backend_contract)
    if not contract_path.is_absolute():
        contract_path = repo / contract_path
    contract = mb.load_contract(contract_path) if contract_path.is_file() else None
    _rb = _budget_for_native(args.model_key)
    extra = {"serving_config": str(cfg_path),
             "serving_config_sha256": serving_config_record["serving_config_sha256"],
             "manifest_sha256": serving_config_record["manifest_sha256"],
             "budget_basis": dataclasses.asdict(_rb.entry.basis), "budget_provisional": _rb.provisional,
             "budget_source": _rb.source, "budget_entry_model_key": _rb.entry.model_key,
             "driver_max_tokens": args.driver_max_tokens,
             "bank_objects": _bank_objects(ddir, list(tt_of.values())),
             "serving_path": "lora_native", "standup_script": "mt_serving_up.py"}
    if selection is not None:
        extra[ra.SERVING_PATH_SELECTION_KEY] = selection
    print(f"[mt-standup] serving path: lora_native (the wave front lora_serving_up.bring_up_lora over the native core); "
          f"mode={args.mode} eval={eval_name} model={model}@{rev[:12]} cells={tt_cells} bank={ddir} "
          f"digests={digests_source} thinking_on={args.thinking_on} reasoning_parser={args.reasoning_parser!r}", flush=True)
    try:
        su = lsu.bring_up_lora(
            repo_dir=repo, config=cfg_path, manifest=manifest_path, defenses_dir=ddir, out=out,
            model_key=args.model_key, eval=eval_name, cell=args.cell,
            adapter_bank=(Path(args.adapter_bank) if args.adapter_bank else None),
            expect_object_sha256=digests, expected_digests_source=digests_source,
            max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
            allow_over_budget=args.allow_over_budget, gpu_mem_util=args.gpu_mem_util,
            max_logprobs=args.max_logprobs, proxy_port=args.proxy_port, vllm_port=args.vllm_port,
            ready_timeout_s=args.ready_timeout_s, install_timeout_s=900,
            upstream_timeout_s=args.upstream_timeout_s, no_petri_agent_loop=args.no_petri_agent_loop,
            mode=args.mode, tool_calls=(args.mode == "petri"), driver_max_tokens=args.driver_max_tokens,
            thinking_on=bool(args.thinking_on), reasoning_parser=args.reasoning_parser,
            text_only_mm=args.text_only_mm, enable_log_requests=bool(args.enable_log_requests),
            mamba_backend_contract=contract, kv_derived_seqs=(args.kv_derived_seqs == "on"),
            attest_mrope=True, extra_endpoint_facts=extra,
            server_default_policy=args.server_default_policy)
    except lsu.PositiveControlsFailed as e:
        print(f"[mt-standup] FAILED (exit {lsu.EXIT_CONTROLS_FAILED}): {e}", flush=True)
        return lsu.EXIT_CONTROLS_FAILED
    except lsu.StandupRefused as e:
        return refuse(str(e))
    except vs.VllmServerError as e:
        return refuse(f"the lora-native launch could not be composed or did not become ready: {e}")
    rc = 0
    try:
        print(f"[mt-standup] READY: proxy {su.proxy_url} serving {len(su.proxy.virtual_models())} vmodels "
              f"(engine lora_native; skipped {[s.cell_id for s in su.cells_skipped]}); waiting for {stop_file}", flush=True)
        rc = wait_until_stop(stop_file, su.server)
    finally:
        su.tear_down()
    print("[mt-standup] down.", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
