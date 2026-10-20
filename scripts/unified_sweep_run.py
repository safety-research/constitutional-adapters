"""Unified jailbreak-robustness + benign-capability sweep driver.

ONE driver that, per defense CONDITION (a DefenseCell at a coeff + mask
variant), evaluates all three legs in a single accelerated pass:

  static    the in-distribution static attack families (n=100 fixed prompts
            each) — batched target generation, NO attacker.
  adaptive  the held-out adaptive families (PAIR/TAP/Crescendo/ActorAttack,
            n=100 goals each) — driven by the ONE SHARED vLLM attacker pool.
  benign    GSM8K / AlpacaEval / OR-Bench-hard / MMLU (n=100) — the capeval
            controls, reused verbatim through cell_controls_run.

Accelerations (see sisteer.eval.sweep for the full statement):
  1. ONE shared attacker pool for the whole sweep (``--attacker-endpoint``);
     the attacker is never steered, so a plain ``vllm serve`` serves every
     condition's adaptive attacks. Not one attacker GPU per condition.
  2. ONE target model load per condition for the JB legs (static + adaptive
     share the loaded steered defense); the benign leg reuses the proven
     cell_controls_run generate path (its own load — kept UNREFACTORED so the
     nsr benign path can never diverge; one extra ~1-2 min load per condition,
     negligible vs generation, recorded in the cost model).
  3. The steered TARGET runs on HF — the working steered-target path today.
     A steered-vLLM serving path (real lens adapters incl. a psr adapter,
     a launchable server, a bit-exact real-artifact gate) EXISTS on
     origin/integrate-main (verified 2026-08-21) -- NOT missing. It is not usable
     BY US yet for two reasons, neither a ban: (a) PRIMARY -- the end-to-end
     server+proxy+steered-generation chain has never produced a generation on GPU
     with a real object (adapters + compat probe only; _bootstrap_live_stack
     still raises); (b) that branch's manifest resolves the template for ITS
     meta-llama-trained objects, which is the wrong template for OUR
     NousResearch-trained psr artifacts -- a per-object template-resolution
     mismatch (the thing the per-object rule + pairing refusal catch), not a
     defect in that pin. See sisteer.eval.sweep for the full statement and the
     branch/SHA every repo-state claim was verified on. Each object is served on
     the template it was TRAINED under (resolved per-object, keyed on object
     sha256); every store is stamped with serving engine + serving path + served
     AND trained template hash + object sha; ``aggregate`` REFUSES any comparison
     differing on engine, template, OR serving path.

Generation engine (``--gen-engine {hf,served}``, default ``hf``). ``served``
routes the WHOLE fixed-prompt leg — BoN AND the non-BoN static/held-out families
— through the ROUTING PROXY over the vLLM server (concurrent fan-out;
``bon_served.ServedStaticDefense`` for the static families, ``bon_generate_
served`` for BoN). Opting in is explicit and the unopted (HF) path is
byte-identical. ``--bon-engine`` stays as a narrower BoN-only override (passing
``--bon-engine served`` ALONE reproduces the pre-generalisation behaviour: BoN
served, every other family HF). Before any served generation the driver runs the
same gates BoN does: ``assert_cell_salt_isolation`` over the FULL route set
(every served cell), and — for a STEERED cell only — ``assert_steering_bites``
through the SAME serving path (an unhooked none/prompted cell has no arm and
skips it). A served family that errors or writes a short/empty store FAILS
LOUDLY (``assert_static_store_complete``): an empty cell that looks complete is
the worst outcome available here. ``--gen-engine-hf-families`` forces a named
family back to HF, LOUDLY, recording engine=hf in its sidecar (never a silent
mix).

COMPARABILITY (not obvious, so stated here). Wiring the non-BoN families to the
served path removes the WITHIN-WAVE engine blend that a BoN-only served run
created (BoN vLLM beside its own family's HF rows), which is a real improvement
— but it does NOT make a vLLM composite comparable to an all-HF one. The engines
are not output-equivalent (measured on the long-SI prompted cell: accept rate
+25.8pp, median 2,413 vs 139 response chars, refusal markers 8.5% vs 70%), so a
fully served composite is STILL not comparable to the existing all-HF anchors
(``none`` 25.1, ``prompted`` 8.1) or to any previously published cell. ONE
engine per comparison, always: ``aggregate`` enforces it by refusing any
cross-engine pair, and ``evaluate.py`` picks ONE engine per invocation for the
same reason.

Subcommands:
  generate   GPU node. Runs a condition's legs; writes each leg's normal store
             PLUS an ``engine_stamp.json`` sidecar (non-invasive: the reused
             drivers are untouched). ``--task-id`` decodes the sbatch array.
  aggregate  the grading host. Reads every condition's JB aggregates + benign summaries +
             their engine sidecars, pairs each condition against its baseline
             (JB vs ``none``; benign vs the matching ``prompted`` cell),
             REFUSING every cross-engine pair, and writes the unified table.
  plan       CPU, no GPU. Dry-run: prints each condition's legs, present-vs-
             absent families (against the live attacks registry), engine
             stamps, and the array decode — the reviewable sweep plan.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from collections.abc import Mapping
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

import jailbreak_run as jr  # the JB leg leaf helpers + config loader

REPO = Path(__file__).resolve().parents[1]

from sisteer.eval import sweep  # noqa: E402


# --------------------------------------------------------------------------- #
# engine-stamp sidecars (non-invasive: written NEXT TO each leg's store)       #
# --------------------------------------------------------------------------- #

def _sidecar_path(out_root: Path, condition_label: str, leg: str,
                  member: str) -> Path:
    d = Path(out_root) / "engine_stamps" / condition_label / leg
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{member}.json"


def _write_sidecar(out_root: Path, cond: sweep.Condition, leg: str,
                   member: str, stamp: dict, *, serving_identity=None,
                   baseline_family=None) -> None:
    """Record the engine stamp (and, when known, the model/template serving
    identity) for one (condition, leg, member) so ``aggregate`` can pair it.
    Both identities are needed to license a comparison — same model/template
    AND same engines."""
    payload = {
        "condition": cond.label, "cell": cond.cell, "coeff": cond.coeff,
        "mask": cond.mask, "leg": leg, "member": member,
        "engine_stamp": stamp,
        "baseline_family": baseline_family,
        "serving_identity": serving_identity,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _sidecar_path(Path(out_root), cond.label, leg, member).write_text(
        json.dumps(payload, indent=2))


def read_sidecar(out_root: Path, condition_label: str, leg: str,
                 member: str) -> dict:
    p = _sidecar_path(Path(out_root), condition_label, leg, member)
    if not p.exists():
        raise SystemExit(
            f"[sweep] no engine stamp at {p} — a leg without an engine stamp "
            "cannot be paired (it was produced by pre-stamp code or a leg that "
            "did not finish). Re-run generate for that (condition, leg).")
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- #
# family <-> live attacks registry reconciliation                              #
# --------------------------------------------------------------------------- #

def _present_absent(families: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Split requested families into those the LIVE attacks registry
    (JB_ATTACKS) can run vs those absent on this branch. The 8 static families
    are complete only on the attack-generalization / twosi-stage3-run lineage;
    this branch (jailbreak lane's vLLM-attacker tip) carries bon + gcg + the 4
    adaptive families. Absent families are surfaced, never silently dropped
    (authoritative-source rule)."""
    from sisteer.eval.attacks import JB_ATTACKS

    present = [f for f in families if f in JB_ATTACKS]
    absent = [f for f in families if f not in JB_ATTACKS]
    return present, absent


# --------------------------------------------------------------------------- #
# defense-OBJECT class of a cell (for the held-out object-axis breakout)        #
# --------------------------------------------------------------------------- #

def _arm_of(cell_id: str):
    """The arm component of a cell id, via the defense_cell registry parser."""
    from sisteer.eval.defense_cell import arm_of

    return arm_of(cell_id)


#: SI id -> semantic role, the axis the ASR table must not pool across.
#: safety_refusal = a real safety SI (candidate defense); si_irrelevant =
#: a semantic PLACEBO / null control (history of the metre);
#: si_constitution = a third distinct SI. Unknown SIs fall through to
#: their own id, still distinct.
_SI_ROLE = {
    "safety_refusal": "safety_relevant",
    "si_irrelevant": "null_control_placebo",
    "si_constitution": "si_constitution",
}


def _object_class_label(cell_id: str) -> str:
    """The defense-OBJECT class of a cell, READ FROM THE REGISTRY
    (``defense_cell``), never inferred from a raw arm string — reader-friendly
    names for the object taxonomy (Class A = SI-distilled, Class B = SI-agnostic
    baseline, plus baselines). A held-out number must never be pooled across
    these: the objects intervene at DIFFERENT phases of the forward (alphasteer /
    the CAA floor never touch decode; antidote edits a prefill position + decode;
    decode-only fits are response-only), so pooling averages a confound. Finer
    training-lane classes (e.g. decode-only fits) parse through the same registry
    when their cells exist."""
    from sisteer.eval.defense_cell import (CLASS_A_ARMS, CLASS_B_ARMS, arm_of,
                                           si_of)

    arm = arm_of(cell_id)
    if arm is None:
        return "baseline"                # none / prompted / prompted__<si>
    si = si_of(cell_id)
    if si is not None:
        # DISTINCT class PER SI ROLE: the seven "fits" are three different
        # kinds and MUST NOT pool. safety_refusal is a real 104-token safety
        # SI (the ONLY candidate-defense rows); si_irrelevant is a SEMANTIC
        # PLACEBO (a document on the history of the metre) — a null control
        # that shows the METHOD distills, never a defense; si_constitution is
        # a third distinct SI. Pooling the placebo with the safety SI would
        # make "distilled objects cut ASR" true by construction. build_asr_table
        # breaks out by this label, so a per-SI-role label is what stops the pool.
        return f"si_distilled:{_SI_ROLE.get(si, si)}"
    if arm in CLASS_B_ARMS:
        return "content_based_arm"       # Class B: NO teacher floor -> "fit" means
                                         # something different (evaluated on its own terms)
    if arm in CLASS_A_ARMS:
        return "si_distilled:single_si"  # bare compiled arm (single-SI lane)
    return "other"


# --------------------------------------------------------------------------- #
# config overrides from a condition                                            #
# --------------------------------------------------------------------------- #

def _cfg_for(cond: sweep.Condition, config_path: str) -> dict:
    cfg = jr._load_cfg(config_path)
    if cond.coeff:
        cfg["scale"] = float(cond.coeff)
    # NOTE deliberately NO steer_positions plumbing: that flag drives the capeval
    # engine-phase gate (prefill unsteered, token 1 unsteered) — the retired
    # mechanism. The sweep's decode-only serving is the POSITION MASK, set on the
    # object in _build_condition_defense; a mask='generated' condition with
    # benign members is refused at Condition construction (one mechanism per
    # condition), so nothing in the sweep may consume the gate flag.
    return cfg


def _apply_cli_overrides(cfg: dict, *, bon_n_max: int | None = None) -> dict:
    """Run-scoped CLI overrides that must NOT edit the SHARED config file.

    ``bon_n_max`` overrides ``cfg['bon']['n_max']`` for THIS run only, so the
    screening sweep can run BoN at a reduced budget (N=64 for screening —
    screen-then-confirm — with 256 reserved for survivors) while
    ``configs/two_si.yaml`` (shared with the two-SI / confirm lanes) stays at 256.
    This is deliberately the CLI path, not a second config: ``_load_cfg`` is a
    flat ``yaml.safe_load`` with no layering, so a "sweep's own config" would be a
    full duplicate that drifts from the shared one. The budget axis stays sound
    because a common N keeps every condition comparable, and every N<=256 is a
    prefix of a 256 run (BoN's continuous best-so-far StrongREJECT vs budget is
    monotone in N), so a screen at 64 is recoverable by extending survivors to
    256. Mutates and returns ``cfg``."""
    if bon_n_max is not None:
        cfg.setdefault("bon", {})["n_max"] = int(bon_n_max)
    return cfg


def _land_defenses_dir(cfg: dict, defenses_dir: str) -> dict:
    """Land the CLI ``--defenses-dir`` in ``cfg["defenses_dir"]`` as an
    ABSOLUTE path — the ``resolve_defenses_dir`` rule-1 seam — so the defense
    BUILD (``registry_from_cfg`` / ``jr._defense_object_path``, hence
    ``_build_condition_defense``) resolves the SAME campaign dir
    ``_resolve_serving`` reads the object's provenance from.

    Without this the CLI dir reached only ``_resolve_serving`` (and the benign
    leg via ``cc_args``): the BUILD fell through to the repo-relative config
    default (``out/two_si/defenses/cells``) — two resolutions of one object,
    violating the record-and-tensors-from-one-place invariant, and on any host
    without that layout (the 2026-08-26 pod smoke) the build died on a missing
    repo-relative ``object.pt`` while the campaign object sat at the dir that
    was passed. Absolute-ness matters: rule 1 fires only on an absolute value
    (and then outranks a ``TT_TWO_SI_DEFENSES_DIR`` export, as an explicit CLI
    choice must). Mutates and returns ``cfg``; falsy ``defenses_dir`` is a
    no-op (config/env resolution unchanged).
    """
    if defenses_dir:
        ddir = Path(defenses_dir).expanduser()
        if not ddir.is_absolute():
            ddir = Path.cwd() / ddir
        cfg["defenses_dir"] = str(ddir)
    return cfg


def _attach_fit_phase_provenance(defense, positions_at_fit, cond) -> None:
    """Attach the sidecar-recorded fit phase to the hydrated object — the
    ``_prov_positions_at_fit`` seam :func:`_positions_at_fit` reads — so the
    FIT side of the three-way requested/fit/measured phase check is the
    training lane's RECORDED fact, never an inference from the object's own
    attributes. The value arrives from the SAME sidecar read that resolved
    the serving mirror (the object file's own directory — object path, never
    a bare cell id, so it cannot describe a different object than the tensors
    being served; six cell_ids exist in BOTH an all-position and a
    response-position fit, disambiguated only by directory). None (a record
    predating the field, or a baseline) attaches nothing, leaving the phase
    gate's default-deny in force."""
    if positions_at_fit is None:
        return
    obj = getattr(defense, "object", None) or getattr(defense, "obj", None)
    if obj is None:
        return  # baseline: nothing was fit, nothing to annotate
    try:
        obj._prov_positions_at_fit = str(positions_at_fit)
    except Exception as e:  # __slots__ etc.: the phase gate then refuses loudly
        print(f"[sweep] {cond.label}: could not attach positions_at_fit "
              f"provenance to {type(obj).__name__} ({e}); a decode-only serve "
              "of this object will default-deny at the phase gate.")


# --------------------------------------------------------------------------- #
# Served BoN (opt-in): route resolution + generation-time probe                 #
# --------------------------------------------------------------------------- #

# The same five plain, harmless probe prompts the cache-isolation gate uses
# (scripts/petri_cache_isolation_gate.py PROMPTS): the probe proves SERVING
# behaviour, not behaviour — a steered arm shifts logits at every position,
# so greedy decode diverges from the scale-0 leg on generic prompts. Kept in
# sync by tests/test_unified_sweep_served_bon.py.
_SERVED_PROBE_PROMPTS = (
    "List the first five prime numbers.",
    "Explain briefly why the sky is blue.",
    "Write a haiku about a quiet library.",
    "Name three uses for a paperclip.",
    "Summarize the water cycle in two sentences.",
)
_SERVED_PROBE_MAX_TOKENS = 48


def _print_family_coverage(fam: str, summary) -> None:
    """State a family's REAL per-goal coverage in the job log (2026-08-28):
    frozen-corpus families can cover fewer goals than they run on (renellm's
    corpus covers only the original 259 test+JBB goal ids -> ~92/112 held-out
    goals; the others run all 112). Any per-family table must print its own
    n, and a composite must never average family rates over unequal
    denominators as if they matched — this line is where an operator first
    sees the denominator differ."""
    if not isinstance(summary, dict) or "n_goals_with_variants" not in summary:
        return
    zero = summary.get("n_goals_zero_variants", 0)
    line = (f"[sweep] {fam}: goal coverage "
            f"{summary['n_goals_with_variants']}/{summary['n_goals_total']} "
            f"(this run generated for {summary['n_goals_run']} goals)")
    if zero:
        line += (f" — {zero} goal(s) have NO {fam} variants in the frozen "
                 "corpus; report this family's own n, never a shared "
                 "denominator")
    print(line)


def _served_harness_cell_id(tt_cell_id: str) -> str:
    """The petri harness cell id for a TrainTest sweep cell — the adapter's
    translation (loud refusal on anything the petri matrix cannot express),
    extended with the sweep's ``stacked__<cell>`` grammar: the harness's
    stacked ids carry the SI WITHOUT the ``si_`` object prefix
    (defense_cells.parse_cell_id), so ``stacked__safety_refusal__psr_kl_all``
    -> ``stacked__safety__psr_kl_all`` and ``stacked__antidote`` (Class B)
    maps through unchanged."""
    from sisteer.capeval.petri import registry_adapter as ra

    if tt_cell_id.startswith("stacked__"):
        inner = ra.harness_cell_id(tt_cell_id[len("stacked__"):])
        if inner.startswith("si_"):
            inner = inner[len("si_"):]
        return "stacked__" + inner
    return ra.harness_cell_id(tt_cell_id)


#: the two legs of the served steering-bites probe, as ``probe.stamps_by_leg``
#: keys them: the steered vmodel and the path-matched scale-0 twin
PROBE_LEG_STEERED = "steered"
PROBE_LEG_SCALE0 = "scale0"


def _served_probe_fn(chat, vmodel: str, twin_vmodel: str, *, extra_body=None):
    """A :data:`bon_served.SteeringProbe` over the SAME serving path the run
    uses: greedy (temperature 0, fixed seed) requests to the cell's virtual
    model and to the path-matched no-op lens leg, one row per probe prompt,
    texts byte-encoded as the id lists ``fidelity._check_steering_bites``
    compares. ``chat`` is a synchronous payload->response callable (a
    ``ProxyClientEngine.chat_completions``); ``cache_salt`` is NEVER sent —
    the proxy owns it.

    ``extra_body`` (fit-support transfer; contract §2.5.3 "ONE shape;
    there is no separate probe shape") is merged VERBATIM into every probe
    payload on BOTH legs — the tinycap driver passes the arm's
    ``chat_template_kwargs`` plus, on a thinking-ON lens standup, the
    ``petri_thinking_on: true`` declaration (the twin is a declared lens cell on
    that standup, so it is declared exactly like the steered leg). With no
    ``extra_body`` every payload is byte-identical to before: every other caller
    passes none. The returned probe also RECORDS each response's stamp triple
    (``contracts.stamp_regimes``, by leg, one per prompt) on
    ``probe.stamps_by_leg`` so the caller can hold the legs to their expected
    rows (§2.5.11); recording never refuses — the caller does.

    THE TEXT COMPARED follows the render the probe REQUESTED
    (:func:`_probe_channel_policy`, recorded on ``probe.channel_policy``):

    - ``both_channels`` — the payload asks for thinking ON (``chat_template_kwargs.
      enable_thinking: true``; the DECLARED header-located lens cell's probe, the
      declaration riding ``extra_body`` on both legs): the server's reasoning
      parser moves the first ``_SERVED_PROBE_MAX_TOKENS`` tokens into the
      reasoning channel (``reasoning`` on this vLLM, ``reasoning_content`` on
      older servers) and leaves ``content`` ``None`` — the gsm8k-paired200
      rows 9 / 10 crashed on ``None.encode`` here — and steering that changes
      only the reasoning still BITES, so the compared text is the two channels
      re-joined in emission order (reasoning, then content).
    - ``content_only`` — every other probe (thinking OFF, and a legacy probe
      with no ``extra_body``): exactly ``content``, byte-identical to before.
      An OFF probe runs on thinking-ON standups too (an undeclared all-position
      cell's bites probe renders OFF on a standup with gemma's reasoning parser
      installed), and nothing establishes that its reasoning channel is empty,
      so it is never read there.

    Under either policy a ``None`` channel is absent, and a leg's prompt whose
    compared text is EMPTY raises :class:`FidelityError` (leg, prompt index and
    finish reason named, never the text) out of the bites control, which the
    driver records as an unverified bite and aborts on — fail closed, never
    certify on nothing (before: an ``AttributeError`` on ``None``, or the
    pairing's own refusal on an empty id row). Each prompt's channel lengths
    and the policy are recorded per leg on ``probe.channels_by_leg``."""
    from sisteer.capeval.serving.fidelity import FidelityError
    from sisteer.capeval.tinycap import contracts as tc

    extra = dict(extra_body or {})
    policy = _probe_channel_policy(extra)
    stamps_by_leg: dict = {PROBE_LEG_STEERED: [], PROBE_LEG_SCALE0: []}
    # a PRESENT malformed stamp (contracts.stamp_regimes refuses it, on every
    # run) is RECORDED per leg here and judged by the caller — the probe never
    # raises out of the bites control, so the refusal is on disk first
    stamp_errors_by_leg: dict = {PROBE_LEG_STEERED: [], PROBE_LEG_SCALE0: []}
    # what each prompt's completion carried, per leg (counts + finish reason
    # only) and the policy it was compared under: the evidence a channel-less
    # draw is read off later
    channels_by_leg: dict = {PROBE_LEG_STEERED: [], PROBE_LEG_SCALE0: []}

    def _greedy(model_name: str, prompt: str, leg: str) -> list[int]:
        resp = chat({
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": _SERVED_PROBE_MAX_TOKENS,
            "seed": 0,
            **extra,
        })
        try:
            stamps_by_leg[leg].append(tc.stamp_regimes(resp)._asdict())
        except tc.OutOfSupportAttestationError as e:
            stamp_errors_by_leg[leg].append(str(e))
        content, reasoning, finish = _probe_channels(resp)
        index = len(channels_by_leg[leg])
        channels_by_leg[leg].append({"prompt_index": index, "content_chars": len(content or ""),
                                     "reasoning_chars": len(reasoning or ""), "finish_reason": finish,
                                     "policy": policy})
        text = ((reasoning or "") + (content or "")) if policy == PROBE_CHANNELS_BOTH else (content or "")
        if not text:
            what = ("either channel (content and reasoning both absent" if policy == PROBE_CHANNELS_BOTH
                    else "its content channel (the OFF probe compares content only")
            raise FidelityError(
                f"steering_bites probe leg {leg!r} prompt {index}: the completion carries NO text in "
                f"{what}, finish_reason={finish!r}) — nothing to compare, so the bite cannot be ruled on; "
                "refusing to certify on an empty draw (a failure, never a pass)")
        return list(text.encode("utf-8"))

    def probe(_arm: str):
        steered = [_greedy(vmodel, p, PROBE_LEG_STEERED) for p in _SERVED_PROBE_PROMPTS]
        scale0 = [_greedy(twin_vmodel, p, PROBE_LEG_SCALE0) for p in _SERVED_PROBE_PROMPTS]
        return steered, scale0

    probe.stamps_by_leg = stamps_by_leg
    probe.stamp_errors_by_leg = stamp_errors_by_leg
    probe.channels_by_leg = channels_by_leg
    probe.channel_policy = policy
    probe.extra_body = dict(extra)
    return probe


#: the bites probe's channel-comparison policies (``probe.channel_policy`` /
#: ``channels_by_leg[*].policy``): chosen from the render the probe REQUESTED
PROBE_CHANNELS_BOTH = "both_channels"          # thinking ON requested: reasoning + content, in emission order
PROBE_CHANNELS_CONTENT_ONLY = "content_only"   # thinking OFF / legacy: content, byte-identical to the pre-channel probe


def _probe_channel_policy(extra_body: Mapping) -> str:
    """Which channels the probe compares, from the payload fragment it sends on
    both legs: :data:`PROBE_CHANNELS_BOTH` iff it REQUESTS thinking ON —
    ``chat_template_kwargs.enable_thinking`` is the JSON ``true``
    (``contracts.render_fragment(arm=on)``, the same key the proxy validates) —
    else :data:`PROBE_CHANNELS_CONTENT_ONLY`. The ``petri_thinking_on``
    declaration is authorisation, not the render, and does not decide it."""
    kwargs = extra_body.get("chat_template_kwargs") if isinstance(extra_body, Mapping) else None
    on = isinstance(kwargs, Mapping) and kwargs.get("enable_thinking") is True
    return PROBE_CHANNELS_BOTH if on else PROBE_CHANNELS_CONTENT_ONLY


def _probe_channels(resp) -> tuple[str | None, str | None, str | None]:
    """``(content, reasoning, finish_reason)`` of an OpenAI-shaped chat
    completion's first choice — the parser-separated reasoning channel read
    under EITHER spelling (``reasoning_content`` on older OpenAI-compatible
    servers, ``reasoning`` on this vLLM 0.25.x — the driver's ``_probe_reply``
    reads the same two). A missing key or a JSON ``null`` reads ``None``:
    absent, never an error here — the caller decides what an absent channel
    means."""
    choice = (resp.get("choices") or [{}])[0] if isinstance(resp, Mapping) else {}
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning_content")
    if reasoning is None:
        reasoning = msg.get("reasoning")
    return msg.get("content"), reasoning, choice.get("finish_reason")


def _served_route_for(tt_cell_id: str, routes, *, harness_id: "str | None" = None):
    """This cell's route out of the FULL exported route set — loud on a cell
    the serving launch did not export (never a silent fall-back to the
    in-process engine: a run asked to serve must serve or refuse).
    ``harness_id`` names the route's cell id EXPLICITLY when it is not the
    bare translation of ``tt_cell_id`` — the tinycap driver's out-of-support
    thinking-ON leg, where the standup serves the cell under its declared
    ``@thinking`` spelling (``contracts.cell_spellings(..., declared=True)
    .harness``) and exports no bare route for it."""
    from sisteer.eval.attacks.bon_served import BonServedError

    harness_id = harness_id or _served_harness_cell_id(tt_cell_id)
    by_cell = {r.cell_id: r for r in routes}
    route = by_cell.get(harness_id)
    if route is None:
        raise BonServedError(
            f"cell {tt_cell_id!r} (harness {harness_id!r}) has no route in "
            f"the routing table ({len(routes)} routes) — the serving launch "
            "did not export it; re-export the routing table with this cell "
            "or run this condition on --bon-engine hf.")
    return route


def _served_setup(cond, args, defense):
    """Resolve the served seam for THIS condition, SHARED by the BoN leg and
    the non-BoN static/held-out families (both serve the SAME cell through the
    SAME routing proxy): the ProxyClientEngine, the FULL route set (the mutual
    salt-isolation proof needs every cell the run serves, not just this one's),
    this cell's route, the SI system text, and — for a steered (lens) cell —
    the steering-bites arms + probe against the path-matched no-op lens leg.
    Every failure is loud; nothing falls back to the in-process engine (a run
    asked to serve must serve or refuse)."""
    from sisteer.capeval.serving.proxy import (ProxyClientEngine,
                                               load_routing_table)
    from sisteer.eval.attacks import bon_served as bs

    if not getattr(args, "bon_proxy_url", None) or \
            not getattr(args, "bon_routing_table", None):
        raise SystemExit(
            "[sweep] a served engine needs BOTH --bon-proxy-url (the "
            "ROUTING PROXY endpoint — never a bare vLLM URL; the proxy owns "
            "per-cell cache_salt isolation and SI-prefix verification) and "
            "--bon-routing-table (the launch's export_routing_table JSON). "
            "These name the proxy for EVERY served family (bon + non-bon), not "
            "bon alone.")
    table = load_routing_table(Path(args.bon_routing_table))
    routes = table["routes"]
    route = _served_route_for(cond.cell, routes)
    client = ProxyClientEngine(args.bon_proxy_url,
                               identity=route.serving_identity)
    # The SI system text the in-process path would have sent — the proxy
    # verifies its bytes against the registered prefix hash (H8), so a
    # mismatch with the serving launch fails loudly per request.
    system = getattr(defense, "system", None)
    steering_arms: tuple = ()
    probe = None
    if bs.is_steered(route):
        # Path-matched no-op leg for the bites probe (proxy_engine.scale0_twin):
        # this cell's own twin when the launch exported one, else the
        # canonical scale-0 leg of its path — none__lens0 for a lens route
        # (the armed hook at scale 0), none__lora0 for a native-LoRA route
        # (the zero adapter). Absent BOTH, the steered-with-no-probe refusal
        # fires downstream — never an unprobed serve.
        from sisteer.capeval.serving import proxy_engine as pe
        twin = pe.scale0_twin(route, routes)
        if twin is not None:
            arm = cond.cell.split("__")[-1]
            probe = _served_probe_fn(client.chat_completions,
                                     route.virtual_model, twin.virtual_model)
            steering_arms = (arm,)
        else:
            steering_arms = (cond.cell.split("__")[-1],)
    return client, routes, route, steering_arms, probe, system


def _bon_served_setup(cond, args, defense):
    """The BoN view of :func:`_served_setup`: a ServedDefense over the routing
    proxy plus the route set / arms / probe bon_generate_served needs. Behaviour
    unchanged from before the non-bon families shared the seam (the proven BoN
    path, job 215881)."""
    from sisteer.eval.attacks import bon_served as bs

    client, routes, route, steering_arms, probe, system = _served_setup(
        cond, args, defense)
    served = bs.ServedDefense.over_client(
        client, virtual_model=route.virtual_model, cell_id=route.cell_id,
        system=system,
        max_model_len=getattr(args, "served_max_model_len", None))
    return served, routes, route, steering_arms, probe


def _effective_engines(args) -> "tuple[str, str]":
    """Resolve ``(non_bon_engine, bon_engine)`` from the flags.

    ``--gen-engine`` is the general knob: it governs EVERY fixed-prompt family
    (the non-BoN static + held-out leg AND, by default, BoN). ``--bon-engine``
    is kept as a narrower, still-working override for the BoN leg ONLY, so an
    invocation that passed just ``--bon-engine served`` before this change keeps
    its exact behaviour (BoN served, the other families HF). Precedence:

      non-BoN engine = ``--gen-engine`` if given, else ``hf`` (the default;
                       opting in is explicit and the unopted path is
                       byte-identical).
      BoN engine     = ``--bon-engine`` if given, else ``--gen-engine`` if
                       given, else ``hf``.

    So ``--gen-engine served`` serves the WHOLE fixed-prompt leg (a fully served
    sweep), ``--bon-engine served`` alone serves only BoN (the pre-existing
    behaviour), and ``--gen-engine served --bon-engine hf`` is an explicit split
    (allowed, but it reintroduces a within-wave engine blend, so it is logged)."""
    gen = getattr(args, "gen_engine", None)
    bon = getattr(args, "bon_engine", None)
    non_bon = gen or "hf"
    bon_eff = bon or gen or "hf"
    return non_bon, bon_eff


class _ServingOnlyDefense:
    """The weightless target for a FULLY SERVED run: it carries ONLY the SI
    system text the served families send (the proxy verifies its prefix hash)
    and ``batch_size=None``. On such a run every family generates through the
    proxy and the object is served by vLLM, so the HF model + object are dead
    weight — loading them is pure contention with the vLLM server on the same
    card. This stub replaces both. It has no ``.generate`` and no ``.model``: if
    an HF-forward leg ever reached it (it never should — the load is gated on
    hf_forward_needed) it would fail LOUDLY, not silently serve an undefended
    model."""

    def __init__(self, system):
        self.system = system
        self.batch_size = None


def _served_only_defense(cond: sweep.Condition, cfg: dict, tok):
    """Build the weightless served target: resolve ``.system`` from the SAME
    source :func:`build_defense_cell` uses (``resolve_instruction``), so the
    bytes match the serving launch's registered prompted prefix (a mismatch is
    caught loudly by the proxy's per-request SI-prefix hash check). A compiled
    (lens) cell sends NO system turn — the object replaces the prompt — so its
    ``.system`` is None, exactly as ``build_defense`` sets it."""
    from sisteer.eval.jailbreak import resolve_instruction
    cell = cond.cell
    system = None
    if cell == "prompted":                       # single-SI lane default
        system = resolve_instruction("safety_refusal").text
    elif cell.startswith("prompted__"):
        system = resolve_instruction(cell[len("prompted__"):]).text
    elif cell.startswith("stacked__"):
        from sisteer.eval.defense_cell import si_of
        si = si_of(cell[len("stacked__"):])
        system = resolve_instruction(si).text if si else None
    # none / compiled: the object replaces the prompt (or there is no defense) —
    # no system turn, matching build_defense(mode='none'|<arm>).
    return _ServingOnlyDefense(system)


def _build_condition_defense(model, tok, cond: sweep.Condition, cfg: dict,
                             prov_positions_at_fit=None):
    """The steered target for a condition, loaded ONCE and shared by the static
    and adaptive JB legs. A DefenseCell id goes through build_defense_cell (the
    two-SI cell path); a single-SI-lane name through build_defense.

    ``prov_positions_at_fit`` is the fit-phase fact from the object's own
    provenance sidecar (via :func:`_resolve_serving`), attached to the hydrated
    object before the decode-phase check so the check's FIT side is the
    recorded value."""
    from sisteer.eval.jailbreak import DEFENSES, COMPILED, build_defense, build_defense_cell

    scale = float(cfg.get("scale", 1.0))
    # eval_gen_batch_size (2026-08-28): the HF generation width for the
    # non-BoN static/held-out legs (the built defense's own batch_size —
    # run_static chunks at it). batch_size is RECORDED serving identity,
    # never a pairing axis, so this is a throughput knob: a peer measured
    # 5x wall-clock (GPU duty 60-70% -> 98-100%) moving this HF path from
    # width 1 to 64, and BoN already runs 64 (DEFAULT_BON_BATCH_SIZE). The
    # module default 8 stands when the config is silent, keeping every
    # existing invocation byte-identical.
    gen_bs = int(cfg.get("eval_gen_batch_size", 8))
    positions = "response" if cond.mask == "generated" else "all"
    # Decode-only boundary: max_new_tokens <= 1 never enters the ctx (100%
    # baseline regardless of arm) — refuse the config outright, before any build.
    if cond.mask == "generated":
        from sisteer.eval.phase_probe import assert_decode_tokens_reachable
        assert_decode_tokens_reachable(jr._target_max_new_tokens(cfg))
    if cond.cell.startswith("stacked__"):
        # measured stack: the object cell served WITH its si_id's system (2026-08-25)
        from sisteer.eval.defense_cell import by_id, registry_from_cfg, stacked_cell
        base = by_id(registry_from_cfg(cfg, repo_root=REPO))[cond.cell[len("stacked__"):]]
        defense = build_defense_cell(model, tok, stacked_cell(base), scale=scale,
                                     positions="all",  # stacking is all-position by construction
                                     batch_size=gen_bs)
        _attach_fit_phase_provenance(defense, prov_positions_at_fit, cond)
        return defense
    if cond.cell in DEFENSES:
        obj_path = (jr._defense_object_path(cfg, cond.cell)
                    if cond.cell in COMPILED else None)
        defense = build_defense(
            model, tok, mode=cond.cell, object_path=obj_path, scale=scale,
            positions=("all" if cond.cell in ("none", "prompted") else positions),
            batch_size=gen_bs)
    else:
        from sisteer.eval.defense_cell import by_id, registry_from_cfg
        cell = by_id(registry_from_cfg(cfg, repo_root=REPO))[cond.cell]
        defense = build_defense_cell(model, tok, cell, scale=scale,
                                     positions=positions, batch_size=gen_bs)
    # The FIT side of the phase check below comes from the provenance sidecar
    # (recorded fact), attached before the check runs.
    _attach_fit_phase_provenance(defense, prov_positions_at_fit, cond)
    # Decode-only serving is gated by the EMPIRICAL three-way phase check, not a
    # per-arm allowlist: verify requested (mask) == measured (probe) == fit
    # (positions_at_fit provenance), returning the phase_relation label
    # (matched / cross_phase) for the stamp, or refusing (mislabelled cell,
    # non-persisting artifact, or unverifiable). Baselines (arm None) are never
    # steered, so they skip the probe.
    if cond.mask == "generated" and _arm_of(cond.cell) is not None:
        # the stamp says position_mask — assert the BUILT object's LIVE state
        # actually expresses it before anything generates (measured state, not
        # the config field the stamp was previously inferred from).
        mech = _assert_position_mask_live(defense, cond)
        print(f"[sweep] {cond.label}: decode mechanism verified live ({mech}).")
        rel = _verify_decode_phase(model, tok, defense, cond, cfg)  # raises on refuse
        print(f"[sweep] {cond.label}: decode-only phase verified ({rel}).")
    return defense


def _assert_position_mask_live(defense, cond) -> str:
    """The decode_mechanism stamp ('position_mask') must be backed by the built
    object's ACTUAL serving state, not inferred from cond.mask — a declared value
    and a measured one diverge exactly where it matters (the previous stamp said
    engine_phase_gate while the JB legs served position_mask). A maskable object
    expresses the serve mask through one of the two real seams
    (_set_response_positions): the vector deployment attrs
    (``positions``/``response_header_len``) or the PositionMaskedFit fit leg
    (``fit_positions``). Returns 'position_mask' or raises — refusing to generate
    under a stamp the object's state does not support."""
    obj = getattr(defense, "obj", None)
    if obj is None:
        raise ValueError(
            f"condition {cond.label!r}: mask='generated' but the built defense "
            "carries no intervention object — nothing can express the position "
            "mask; refusing (a baseline decode-only cell is a config error).")
    deploy_resp = (getattr(obj, "positions", None) == "response"
                   and getattr(obj, "response_header_len", None))
    fit_resp = (getattr(obj, "fit_positions", None) == "response"
                and getattr(obj, "fit_response_header_len", None))
    if not (deploy_resp or fit_resp):
        raise ValueError(
            f"condition {cond.label!r}: the decode_mechanism stamp says "
            "position_mask but the BUILT object's live state expresses no "
            f"response mask (positions={getattr(obj, 'positions', None)!r}, "
            f"response_header_len={getattr(obj, 'response_header_len', None)!r}, "
            f"fit_positions={getattr(obj, 'fit_positions', None)!r}) — refusing "
            "to generate under a mechanism stamp the serving state does not "
            "back (the declared-vs-measured divergence this check exists for).")
    return "position_mask"


def _positions_at_fit(defense, cfg):
    """The object's fit-time applied phase from its provenance record
    (``positions_at_fit``, recorded by the training sidecar alongside model
    id / revision / template hash), mapped to a phase_probe phase; None
    when absent.

    None -> default-deny: an object with no fit-phase provenance is unverifiable
    and must not be served decode-only (the fail-safe). Read from the hydrated
    object's provenance dict or the ``_prov_positions_at_fit`` attribute that
    :func:`_attach_fit_phase_provenance` sets from the SAME sidecar read that
    resolved the serving mirror."""
    from sisteer.eval.phase_probe import phase_of_mask
    obj = getattr(defense, "object", None) or getattr(defense, "obj", None)
    prov = getattr(obj, "provenance", None)
    pos = prov.get("positions_at_fit") if isinstance(prov, dict) else None
    if pos is None:
        pos = getattr(obj, "_prov_positions_at_fit", None)
    return phase_of_mask(str(pos)) if pos is not None else None


def _decode_probe_forward(model, tok, defense):
    """Serve-time probe harness: returns ``(prompt_len, seq_len, header_len,
    ignore_positions, capture)`` where ``capture(applied)`` runs ONE prefill +
    teacher-forced-decode pass over a fixed synthetic sequence and returns
    per-position hidden states WITH (``applied=True``) / WITHOUT the object's ctx.
    Fed to :func:`phase_probe.probe_object_footprint`. Delegates to the shared,
    GPU-validated forward pair in :mod:`sisteer.eval.phase_probe_gpu` (the one
    model-specific call site), so the sweep and the validation script measure the
    footprint identically. Baselines (no object) never reach here."""
    from sisteer.eval.phase_probe import PhaseVerificationError
    from sisteer.eval.phase_probe_gpu import decode_probe_capture
    obj = getattr(defense, "obj", None)
    if obj is None:
        raise PhaseVerificationError(
            "decode-only probe reached a defense with no intervention object — a "
            "baseline should never be probed; refusing (default-deny).")
    return decode_probe_capture(model, tok, obj, scale=getattr(defense, "scale", 1.0))


def _verify_decode_phase(model, tok, defense, cond, cfg) -> str:
    """Empirical serving-phase gate for a decode-only cell, comparing measured
    position FOOTPRINTS (not coarse labels): the measured footprint must equal the
    REQUESTED one (mislabel test) AND be a subset of the arm's DEFINED footprint
    (relocation test), then matched/cross_phase is read against the FIT footprint
    (:func:`phase_probe.verify_footprint`). Returns the phase_relation label or
    refuses (mislabelled, relocation, non-persisting artifact, unverifiable).

    Default-denies when ``positions_at_fit`` is absent (a record predating the
    field, or a partial/override provenance path) — a loud refusal rather than
    an unmeasured phase. When provenance is present it runs the shared GPU
    forward pair (validated by scripts/validate_phase_probe.py)."""
    from sisteer.eval.phase_probe import (PhaseVerificationError, expand_footprint,
                                          phase_of_mask, probe_object_footprint,
                                          verify_footprint)
    from sisteer.eval.decode_capability import defined_footprint
    requested_phase = phase_of_mask(cond.mask)
    fit_phase = _positions_at_fit(defense, cfg)
    if fit_phase is None:
        # default-deny: no fit-time provenance -> unverifiable, refuse loudly.
        raise PhaseVerificationError(
            f"serving phase unverifiable [{cond.label}]: the object exposes no "
            "positions_at_fit provenance — refusing a decode-only serve "
            "(default-deny). This object's fit-time record predates the "
            "positions_at_fit field (or the object was resolved through a "
            "partial/override provenance path, which carries no fit phase).")
    prompt_len, seq_len, header_len, ignore_pos, capture = _decode_probe_forward(
        model, tok, defense)
    measured = probe_object_footprint(
        capture, prompt_len=prompt_len, seq_len=seq_len, header_len=header_len,
        ignore_positions=ignore_pos)
    arm = _arm_of(cond.cell)
    kw = dict(prompt_len=prompt_len, seq_len=seq_len, header_len=header_len)
    requested_fp = expand_footprint(requested_phase, **kw)
    fit_fp = expand_footprint(fit_phase, **kw)
    defined_fp = expand_footprint(defined_footprint(arm), **kw)
    return verify_footprint(requested=requested_fp, defined=defined_fp,
                            measured=measured, fit=fit_fp, what=cond.label)


# --------------------------------------------------------------------------- #
# generate — GPU: load once, run the JB legs; then the benign leg              #
# --------------------------------------------------------------------------- #

class ServingResolution(NamedTuple):
    """One condition's resolved serving mirror + the provenance facts that must
    travel with every row it generates (see sweep.MirrorProvenance)."""

    model_id: str
    revision: "str | None"
    trained_template_hash: "str | None"
    object_sha: "str | None"
    baseline_family: "str | None"
    provenance_status: "str | None"    # verified | partial | override | document_fit | None
    provenance_source: "str | None"    # which record the identity came from
    positions_at_fit: "str | None"     # the recorded fit phase, None if absent


def _resolve_serving(cond: sweep.Condition, cfg: dict,
                     defenses_dir: str) -> ServingResolution:
    """Resolve the PER-OBJECT serving mirror for a condition, keyed on the
    object sha256. The provenance sidecar is read from the OBJECT FILE's own
    directory (object path, never a bare cell id — six cell_ids exist in both
    an all-position and a response-position fit, disambiguated only by which
    defenses_dir the run points at), so the record and the served tensors can
    never come from different objects. A compiled cell resolves from its own
    training provenance and REFUSES loudly if unrecorded (no default-mirror
    fallback) unless a forensic override is given; a baseline (none/prompted)
    has no compiled object, so it serves on the config/override mirror and
    carries baseline_family = its family (the same cell id), replicated per
    template by the table author (or a mixed-population generator)."""
    from sisteer.eval.defense_cell import object_path_for
    is_baseline = cond.cell in ("none",) or cond.cell.startswith("prompted__")
    if is_baseline:
        return ServingResolution(
            model_id=cfg["model"], revision=cfg.get("model_revision"),
            trained_template_hash=None, object_sha=None,
            baseline_family=cond.cell, provenance_status=None,
            provenance_source=None, positions_at_fit=None)
    # A stacked condition serves the BASE object (e.g. safety_refusal__token_faithful)
    # WITH a system prompt; its object path AND fit-provenance are the base object's
    # (fully verified), so the stack never inherits a weaker provenance than its
    # object. Resolve both against the base cell id, not the "stacked__" alias.
    resolve_cell = (cond.cell[len("stacked__"):]
                    if cond.cell.startswith("stacked__") else cond.cell)
    obj_path = object_path_for(defenses_dir, resolve_cell) if defenses_dir else None
    if obj_path is None:
        obj_path = jr._defense_object_path(cfg, resolve_cell)
    prov = sweep.resolve_training_mirror(resolve_cell, obj_path,
                                         override=cond.mirror_override)
    return ServingResolution(
        model_id=prov.model_id, revision=prov.revision,
        trained_template_hash=prov.trained_template_hash,
        object_sha=prov.object_sha, baseline_family=None,
        provenance_status=prov.status, provenance_source=prov.source,
        positions_at_fit=prov.positions_at_fit)


def cmd_generate(args) -> None:
    conditions = sweep.load_conditions(args.conditions)
    if args.task_id is not None:
        cond = sweep.condition_for_array_index(conditions, args.task_id)
    elif args.condition:
        matches = [c for c in conditions if c.label == args.condition]
        if not matches:
            raise SystemExit(f"[sweep] no condition labelled {args.condition!r}")
        cond = matches[0]
    else:
        raise SystemExit("[sweep] pass --task-id <N> or --condition <label>")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_for(cond, args.config)
    # run-scoped CLI overrides (never a shared-config edit); see _apply_cli_overrides.
    _apply_cli_overrides(cfg, bon_n_max=getattr(args, "bon_n_max", None))
    # BEFORE anything resolves an object path: the build and the provenance
    # read must resolve the ONE dir the CLI names (see _land_defenses_dir).
    _land_defenses_dir(cfg, args.defenses_dir)
    if getattr(args, "bon_n_max", None) is not None:
        print(f"[sweep] bon.n_max overridden to {args.bon_n_max} for this run "
              f"via --bon-n-max (configs/two_si.yaml unchanged at "
              f"{jr._load_cfg(args.config).get('bon', {}).get('n_max')})")

    present_static, absent_static = _present_absent(cond.static)
    present_adaptive, absent_adaptive = _present_absent(cond.adaptive)
    present_held_out, absent_held_out = _present_absent(cond.held_out)
    absent = absent_static + absent_adaptive + absent_held_out
    if absent and not args.allow_missing_families:
        raise SystemExit(
            f"[sweep] condition {cond.label!r} requests families {absent} that "
            "are NOT in the live attacks registry (JB_ATTACKS) on this branch. "
            "The full 8 static families live on the attack-generalization / "
            "twosi-stage3-run lineage; merge them in, or pass "
            "--allow-missing-families to run the present ones and record the "
            "absent set (never a silent drop).")
    if absent:
        print(f"[sweep] NOTE: absent families recorded, not run: {absent}")

    # -- the JB legs: ONE target model load, shared steered defense ---------- #
    print(f"[sweep] condition={cond.label} cell={cond.cell} coeff={cond.coeff or '(default)'} "
          f"mask={cond.mask} static={present_static} adaptive={present_adaptive} "
          f"held_out={present_held_out} benign={list(cond.benign)}")
    from sisteer.models import (chat_template_hash, load_model_and_tokenizer,
                                serving_identity)

    # PER-OBJECT serving mirror (keyed on object sha; refuses if unrecorded).
    res = _resolve_serving(cond, cfg, args.defenses_dir)
    model_id, revision, trained_th, obj_sha, base_family = (
        res.model_id, res.revision, res.trained_template_hash,
        res.object_sha, res.baseline_family)
    cfg = {**cfg, "model": model_id, "model_revision": revision}
    print(f"[sweep] serving mirror: {model_id}@{revision or 'main'} "
          f"trained_template={trained_th or '(baseline/none)'} obj_sha={obj_sha} "
          f"provenance={res.provenance_source or '(baseline)'}"
          f"[{res.provenance_status or '-'}]")

    # Effective engines (pure, from the flags) — resolved BEFORE the load so we
    # can decide whether the HF target is dead weight on this run.
    non_bon_engine, bon_engine = _effective_engines(args)
    hf_optout = {f.strip() for f in
                 (getattr(args, "gen_engine_hf_families", "") or "").split(",")
                 if f.strip()}

    def _fam_is_hf(fam: str) -> bool:
        if fam == "bon":
            return bon_engine == "hf"
        return non_bon_engine != "served" or fam in hf_optout

    # The HF target (weights on GPU) is needed ONLY if some leg does an HF
    # forward: an HF-engine family (bon or non-bon), the adaptive leg (always
    # HF-generates), or the decode-phase probe (mask='generated' steered cell,
    # inside _build_condition_defense). On a FULLY SERVED run none fire, and the
    # ~18 GiB load is pure contention with the vLLM server on the same card
    # (job 227485: the HF load OOM'd against a vLLM server holding 96.5% of the
    # H200 — two models sizing buffers off max_num_seqs*max_model_len have no
    # stable operating point). So skip it: tokenizer only + a weightless served
    # defense carrying just the SI system text. NOTE the benign leg reloads its
    # OWN model (cell_controls_run) and is not covered by this — a served run
    # with a benign leg on one GPU is a separate contention, out of scope here.
    # The decode-phase probe clause is gated on NOT being fully served
    # (2026-08-28). The comment above asserts "on a FULLY SERVED run none
    # fire", but the clause carried no served condition, so a masked FITTED
    # cell fired it and loaded ~18 GiB of HF weights beside a vLLM server
    # already holding 136.96 GiB -- OOM on a 112 MiB allocation, killing every
    # fitted condition of job 232263 (_4, _5) at ~25 minutes while the
    # baselines, which have no arm and so skipped the clause, COMPLETED. That
    # split is the signature.
    #
    # Skipping it on a served run is sound rather than merely convenient: the
    # mask is applied SERVER-side from the routing table, so a client-side
    # decode probe re-derives nothing the server used. The served path has its
    # own equivalents -- the lens worker's install/selection record, and the
    # token-1-steered gate the LoRA smoke exercises. Say so out loud, because
    # a silently skipped probe is exactly how a mask stops being checked.
    _fully_served = non_bon_engine == "served" and bon_engine == "served"
    _decode_probe_needed = (cond.mask == "generated"
                            and _arm_of(cond.cell) is not None)
    if _decode_probe_needed and _fully_served:
        print(f"[sweep] {cond.label}: client-side decode-phase probe SKIPPED "
              "(fully served -- the mask is applied server-side from the "
              "routing table; loading an HF target here only contends with "
              "the vLLM server on this card)")
    hf_forward_needed = (
        any(_fam_is_hf(f) for f in (present_static + present_held_out))
        or bool(present_adaptive)
        or (_decode_probe_needed and not _fully_served))

    t_load0 = time.time()
    if hf_forward_needed:
        model, tok = load_model_and_tokenizer(
            cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
            attn_implementation=cfg.get("attn_implementation", "sdpa"))
        ident = serving_identity(tok, cfg, model=model)
    else:
        from transformers import AutoTokenizer
        model = None
        tok = AutoTokenizer.from_pretrained(cfg["model"],
                                            revision=cfg.get("model_revision"))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        ident = serving_identity(tok, cfg, model=None)
    served_th = chat_template_hash(tok)
    # the served==trained invariant (a compiled object must be served on the
    # template it was trained under; baselines are exempt).
    _st = cond.static_stamp(served_template_hash=served_th,
                            trained_template_hash=trained_th, object_sha=obj_sha,
                            provenance_status=res.provenance_status,
                            provenance_source=res.provenance_source)
    sweep.assert_served_matches_trained(_st, what=f"{cond.label} generate")

    def _stamp_kw():
        # batch_size is RECORDED serving identity (not a pairing axis): batched
        # bf16 generation is not byte-identical across batch compositions, so a
        # per-item divergence between runs must be explicable from the stamps.
        # Read from the built defense (the size run_static actually batches at);
        # None before the defense exists (never the case in the leg loops).
        # Read the CAPTURED width, not `defense` — the benign leg `del`s defense
        # before its stamp, and a closure over the freed name NameErrors (every
        # mask=all condition crashed here; the sbatch's post-`log` exit 0 masked
        # it as COMPLETED). _defense_bs survives the del.
        return dict(served_template_hash=served_th,
                    trained_template_hash=trained_th, object_sha=obj_sha,
                    batch_size=_defense_bs,
                    provenance_status=res.provenance_status,
                    provenance_source=res.provenance_source)
    if hf_forward_needed:
        defense = _build_condition_defense(
            model, tok, cond, cfg, prov_positions_at_fit=res.positions_at_fit)
    else:
        # fully served: the HF model + object are dead weight — a weightless
        # served defense carrying only the SI system text (resolved from the
        # same source build_defense_cell uses; the proxy verifies its hash).
        defense = _served_only_defense(cond, cfg, tok)
    _defense_bs = getattr(defense, "batch_size", None)  # captured for the benign stamp (after del defense)
    _tgt = ("loaded + defense built" if hf_forward_needed
            else "NOT loaded (fully served — HF weights are dead weight)")
    print(f"[sweep] target {_tgt} in {time.time() - t_load0:.1f}s "
          f"(engines: non-bon={non_bon_engine} bon={bon_engine}; "
          f"{len(present_static) + len(present_adaptive) + len(present_held_out)} JB legs)")

    goals = jr._eval_goals(cfg, confirmatory=True)  # the 100-goal confirmatory set
    if getattr(args, "max_goals", None):
        goals = goals[:int(args.max_goals)]
        print(f"[sweep] SMOKE: JB goal set capped to first {len(goals)} "
              "(--max-goals); not a full run.")
    root = jr._private_root(cfg, str(out_root / "jb_private"))

    # -- the served-static seam (lazy; SHARED across every served non-BoN
    #    static + held-out family in this condition) ---------------------- #
    #    non_bon_engine / bon_engine / hf_optout were resolved above (they gate
    #    the weightless target load), so just report and reuse them here.
    print(f"[sweep] engines: non-BoN fixed-prompt families -> {non_bon_engine}; "
          f"BoN -> {bon_engine}"
          + (f"; forced HF (--gen-engine-hf-families): {sorted(hf_optout)}"
             if hf_optout else ""))

    _served: dict = {}  # populated once the first served non-BoN family runs

    def _served_family_engine(fam: str) -> str:
        if non_bon_engine != "served" or fam in hf_optout:
            return "hf"
        return "served"

    def _ensure_served_seam() -> dict:
        """Resolve the served seam ONCE, run the pre-generation gates, and open
        ONE aiohttp session + event loop reused across every served non-BoN
        family. ONE loop is deliberate: an ``asyncio.run`` per family closes the
        loop its aiohttp session is bound to, and every family after the first
        would error (the reference driver hit exactly this)."""
        import asyncio
        import aiohttp
        from sisteer.eval.attacks import bon_served as bs
        if _served:
            return _served
        client, routes, route, steering_arms, probe, system = _served_setup(
            cond, args, defense)
        # loud endpoint / virtual-model verification before any generation
        client.load(require_virtual_models=[route.virtual_model])
        # GATE 1 (every served cell): no two distinct cells the run serves can
        # ever share a cached (possibly steered) prefix — the correctness
        # boundary. Proved over the FULL route set.
        bs.assert_cell_salt_isolation(routes)
        # GATE 2 (STEERED cells only): prove the arm actually changes the output
        # through the SAME serving path (the silently-unhooked failure mode).
        # An UNHOOKED cell (none / prompted) carries NO arm — the gate does not
        # apply and must not run; the proxy's per-request SI-prefix verification
        # is its analogous guard.
        if steering_arms:
            if probe is None:
                raise bs.BonServedError(
                    f"cell {cond.cell!r} is steered (arms "
                    f"{list(steering_arms)}) but no steering probe was resolved "
                    "— refusing to serve a steered run with no generation-time "
                    "proof that steering bites (the silently-unhooked failure "
                    "mode)")
            bs.assert_steering_bites(steering_arms, probe)
            print(f"[sweep] {cond.label}: served steering-bites verified "
                  f"({list(steering_arms)}).")
        else:
            print(f"[sweep] {cond.label}: unhooked served cell — no "
                  "steering-bites gate (SI-prefix verification is the proxy's "
                  "own per-request guard).")
        proxy_url = args.bon_proxy_url.rstrip("/")
        loop = asyncio.new_event_loop()
        sess: dict = {"s": None}

        async def achat(payload):
            # DIRECT aiohttp to the PROXY (never a bare vLLM URL): the proxy
            # owns routing, per-cell cache_salt and SI-prefix verification.
            # True async fan-out (not a thread pool) so a whole family can have
            # max_concurrency requests in flight at once — the throughput the
            # ~5.4-min measurement needs.
            if sess["s"] is None:
                sess["s"] = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=600))
            async with sess["s"].post(
                    proxy_url + "/chat/completions", json=dict(payload),
                    headers={"Authorization": f"Bearer {client.api_key}"}) as r:
                body = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"proxy {r.status}: {body[:400]}")
                return json.loads(body)

        async def atokenize(payload):
            # the proxy's read-only /tokenize pass-through: how a served
            # client resolves vLLM's floor-only context-length rejection to
            # the exact prompt length before its once-only clamp
            # (served_window); a non-200 raises and the client files the
            # typed floor_only_unresolvable error rather than guess
            if sess["s"] is None:
                sess["s"] = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=600))
            async with sess["s"].post(
                    proxy_url + "/tokenize", json=dict(payload),
                    headers={"Authorization": f"Bearer {client.api_key}"}) as r:
                body = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"proxy {r.status}: {body[:400]}")
                return json.loads(body)

        _served.update(
            routes=routes, route=route, system=system, loop=loop, sess=sess,
            achat=achat, atokenize=atokenize,
            max_concurrency=int(getattr(args, "bon_max_concurrency", None)
                                or bs.DEFAULT_MAX_CONCURRENCY),
            # the launched window (--served-max-model-len), threaded into
            # every ServedDefense so the once-only max_tokens clamp knows it
            max_model_len=getattr(args, "served_max_model_len", None))
        return _served

    def _close_served_seam() -> None:
        if not _served:
            return
        loop = _served.get("loop")
        sess = _served.get("sess") or {}
        try:
            if sess.get("s") is not None and loop is not None:
                loop.run_until_complete(sess["s"].close())
        except Exception:
            pass
        try:
            if loop is not None:
                loop.close()
        except Exception:
            pass

    def _run_nonbon_family(fam: str, out_dir):
        """Run one non-BoN static/held-out family on its effective engine.
        Returns ``(serving_identity, was_served)`` for the sidecar. A SERVED
        family FAILS LOUDLY on any request error or a short/empty store
        (assert_static_store_complete) — an empty cell must never look
        complete."""
        from sisteer.eval.attacks import get_attack
        from sisteer.eval.attacks import bon_served as bs
        from sisteer.eval.attacks.static_common import \
            assert_static_store_complete
        if _served_family_engine(fam) == "served":
            seam = _ensure_served_seam()
            # a FRESH ServedDefense per family (its own n_target_forwards, JB7),
            # all wrapping the ONE shared loop + aiohttp session.
            sd = bs.ServedDefense(seam["achat"],
                                  virtual_model=seam["route"].virtual_model,
                                  cell_id=seam["route"].cell_id,
                                  system=seam["system"],
                                  max_model_len=seam["max_model_len"],
                                  atokenize=seam["atokenize"])
            served_defense = bs.ServedStaticDefense(
                sd, loop=seam["loop"], max_concurrency=seam["max_concurrency"])
            summary = get_attack(fam)(
                served_defense, goals, out_dir=out_dir,
                max_new_tokens=jr._target_max_new_tokens(cfg),
                seed=cfg.get("seed", 0))
            assert_static_store_complete(summary, out_dir, family=fam,
                                         n_errors=served_defense.n_errors)
            print(f"[sweep] {fam}: served (engine=vllm_served) — store "
                  f"complete, max_inflight={served_defense.max_inflight}")
            _print_family_coverage(fam, summary)
            return seam["route"].serving_identity, True
        if non_bon_engine == "served" and fam in hf_optout:
            print(f"[sweep] {fam}: FORCED HF (--gen-engine-hf-families) while "
                  "the rest of the wave is served — the sidecar RECORDS "
                  "engine=hf for it (a deliberate, recorded per-family engine, "
                  "never a silent mix).")
        # static_common-backed families take max_new_tokens + seed (NOT
        # target_max_new_tokens, which only gcg/adaptive take); trial defaults
        # to 0 and the sub-env family (amplegcg) does not accept it.
        summary = get_attack(fam)(defense, goals, out_dir=out_dir,
                                  max_new_tokens=jr._target_max_new_tokens(cfg),
                                  seed=cfg.get("seed", 0))
        _print_family_coverage(fam, summary)
        return ident, False

    timings: dict[str, float] = {}
    # STATIC leg: fixed prompt sets, no attacker.
    for fam in present_static:
        t0 = time.time()
        was_served = False
        fam_identity = ident
        if fam == "bon":
            run_dir = jr._bon_defense_dir(cfg, root, cond.cell)
            bcfg = cfg["bon"]
            if bon_engine == "served":
                # Served BoN (opt-in): concurrent fan-out through the ROUTING
                # PROXY over the vLLM server — never a bare vLLM URL (the proxy
                # owns per-cell cache_salt isolation and SI-prefix
                # verification). bon_generate_served enforces the per-cell
                # salt-isolation proof over the FULL route set and, for a
                # steered cell, a generation-time steering-bites probe through
                # the SAME serving path, before anything generates.
                from sisteer.eval.attacks.bon_served import (
                    DEFAULT_MAX_CONCURRENCY, bon_generate_served)
                served, routes, route, steering_arms, probe = \
                    _bon_served_setup(cond, args, defense)
                bon_generate_served(
                    served, goals, out_dir=run_dir, routes=routes,
                    n_max=bcfg["n_max"], sigma=bcfg["sigma"], trial=0,
                    max_new_tokens=bcfg["max_new_tokens"],
                    max_concurrency=int(
                        getattr(args, "bon_max_concurrency", None)
                        or DEFAULT_MAX_CONCURRENCY),
                    steering_arms=steering_arms, steering_probe=probe)
                was_served = True
                fam_identity = route.serving_identity
            else:
                from sisteer.eval.attacks.bon import bon_generate
                bon_bs = getattr(args, "bon_batch_size", None)
                bon_generate(defense, goals, out_dir=run_dir,
                             n_max=bcfg["n_max"], sigma=bcfg["sigma"],
                             trial=0,
                             max_new_tokens=bcfg["max_new_tokens"],
                             batch_size=bon_bs)
        else:
            out_dir = jr._gated_shard_dir(cfg, root, fam, cond.cell, 0)
            fam_identity, was_served = _run_nonbon_family(fam, out_dir)
        timings[f"static/{fam}"] = time.time() - t0
        # The sidecar's serving identity must describe the engine that ACTUALLY
        # generated (never silently mixed). batch_size: recorded for HF (bf16
        # batched generation is not byte-identical across batch compositions),
        # but MEANINGLESS under continuous batching, so None for a served row
        # (every served row carries engine + bon_max_concurrency itself). An
        # absent --bon-batch-size is no longer width 1: bon_generate defaults to
        # DEFAULT_BON_BATCH_SIZE (64).
        stamp_kw = _stamp_kw()
        if was_served:
            stamp_kw = {**stamp_kw, "batch_size": None}
        elif fam == "bon":
            from sisteer.eval.attacks.bon import DEFAULT_BON_BATCH_SIZE
            stamp_kw = {**stamp_kw, "batch_size": int(
                getattr(args, "bon_batch_size", None) or DEFAULT_BON_BATCH_SIZE)}
        _write_sidecar(out_root, cond, "static", fam,
                       cond.static_stamp(**stamp_kw),
                       serving_identity=fam_identity,
                       baseline_family=base_family)
        print(f"[sweep] static/{fam} done in {timings[f'static/{fam}']:.1f}s")

    # ADAPTIVE leg: shared vLLM attacker pool.
    if present_adaptive:
        seam_args = argparse.Namespace(
            attack=present_adaptive[0], attacker_engine=cond.attacker_engine,
            attacker_endpoint=(cond.attacker_endpoint or args.attacker_endpoint))
        scorer_fn, attacker_factory, attacker_fn = jr._build_attacker_seams(cfg, seam_args)
        for fam in present_adaptive:
            t0 = time.time()
            out_dir = jr._gated_shard_dir(cfg, root, fam, cond.cell, 0)
            jr._write_engine_manifest(out_dir, cfg, seam_args, fam)
            jr._run_gated_attack(
                fam, defense, goals, attacker_factory=attacker_factory,
                attacker_fn=attacker_fn, scorer_fn=scorer_fn, out_dir=out_dir,
                forward_cap=jr._forward_cap(cfg),
                target_max_new_tokens=jr._target_max_new_tokens(cfg))
            timings[f"adaptive/{fam}"] = time.time() - t0
            _write_sidecar(out_root, cond, "adaptive", fam,
                           cond.adaptive_stamp(**_stamp_kw()),
                           serving_identity=ident, baseline_family=base_family)
            print(f"[sweep] adaptive/{fam} done in {timings[f'adaptive/{fam}']:.1f}s")

    # HELD-OUT leg: EVAL-ONLY families (never training). No attacker — batched
    # target generation over a frozen/deterministic prompt corpus, exactly like
    # the static leg. Each family is LEAKAGE-GATED against the training pool and
    # gets a per-family MANIFEST (content hash, provenance, leakage result)
    # BEFORE generation, so a leaking family REFUSES rather than being replayed.
    # The manifest also breaks out the GOAL-holdout axis (transform- vs
    # full-generalisation, via the AUTHORITATIVE category split) and the defense
    # OBJECT class of this cell, so no reader pools over either axis.
    if present_held_out:
        from sisteer.eval.attacks import get_attack, get_heldout_spec
        from sisteer.eval.attacks.heldout_common import write_family_manifest
        # AUTHORITATIVE goal split (same source as by_category_group), not
        # re-derived here: {goal_id -> "held_in"|"held_out"}.
        group_of, _hi, _ho, _ch = jr._category_groups(cfg, goals)
        obj_class = _object_class_label(cond.cell)
        obj_arm = _arm_of(cond.cell)
        # The training fingerprint is authoritative: load it from the config's
        # declared pool. If the store is not on this node, the gate cannot run
        # here — record it LOUDLY (skipped, never a fabricated pass) rather than
        # silently trusting the family; run the gate where the store lives.
        try:
            from sisteer.eval.attacks.heldout_leakage import (
                load_stage3_training_fingerprint_from_config)
            fingerprint = load_stage3_training_fingerprint_from_config(cfg)
            print(f"[sweep] held-out leakage gate: {len(fingerprint)} training shas")
        except (FileNotFoundError, ValueError) as e:
            fingerprint = None
            print("[sweep] NOTE: held-out leakage gate could not load the "
                  f"training fingerprint here ({e}); manifests record "
                  "leakage_check=skipped — run the gate where the store lives.")
        for fam in present_held_out:
            t0 = time.time()
            out_dir = jr._gated_shard_dir(cfg, root, fam, cond.cell, 0)
            man_path = out_root / "heldout_manifests" / cond.label / f"{fam}.json"
            # preflight: materialise -> leakage-gate (REFUSE on overlap) ->
            # manifest (with the goal-holdout axis + defense-object class)
            write_family_manifest(get_heldout_spec(fam), goals,
                                  manifest_path=man_path, fingerprint=fingerprint,
                                  category_group_of=group_of,
                                  object_class=obj_class, cell=cond.cell,
                                  arm=obj_arm, seed=cfg.get("seed", 0),
                                  attack_regime=sweep.attack_regime_of_leg("held_out"))
            fam_identity, was_served = _run_nonbon_family(fam, out_dir)
            timings[f"held_out/{fam}"] = time.time() - t0
            ho_stamp_kw = _stamp_kw()
            if was_served:
                ho_stamp_kw = {**ho_stamp_kw, "batch_size": None}
            _write_sidecar(out_root, cond, "held_out", fam,
                           cond.held_out_stamp(**ho_stamp_kw),
                           serving_identity=fam_identity,
                           baseline_family=base_family)
            print(f"[sweep] held_out/{fam} done in {timings[f'held_out/{fam}']:.1f}s")

    # tear down the shared served-static event loop + aiohttp session (no-op if
    # no non-BoN family ran served) before the benign leg reloads the model.
    _close_served_seam()

    # free the JB target before the benign leg reloads (kept separate to reuse
    # the proven cell_controls path unmodified).
    del defense, model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # BENIGN leg: reuse cell_controls_run generate verbatim (its own load).
    if cond.benign:
        t0 = time.time()
        import cell_controls_run as ccr
        # A condition with benign members is all-position BY CONSTRUCTION
        # (mask='generated' + benign refuses in Condition.__post_init__), so the
        # benign leg always serves steer_positions='all' — the engine-phase gate
        # (the retired decode-only mechanism) is unreachable from the sweep.
        assert cond.mask != "generated", "unreachable: refused at construction"
        cc_args = argparse.Namespace(
            config=args.config, out_root=str(out_root / "benign"),
            defenses_dir=args.defenses_dir, cell=cond.cell,
            controls=",".join(cond.benign), limit=args.limit,
            judge_model=args.judge_model, coeff=cond.coeff,
            steer_positions="all")
        cc_cfg = jr._load_cfg(args.config)
        # Fail loud if the config's declared controls and the sweep's
        # BENIGN_CONTROLS have diverged (the 2026-08-24 silent-xstest-drop bug):
        # a declared, runnable control must be honoured, never quietly skipped.
        sweep.assert_benign_controls_match_config(cc_cfg)
        if cond.coeff:
            cc_cfg["scale"] = float(cond.coeff)
        ccr.cmd_generate(cc_args, cc_cfg)
        for member in cond.benign:
            _write_sidecar(out_root, cond, "benign", member,
                           cond.benign_stamp(**_stamp_kw()),
                           serving_identity=ident, baseline_family=base_family)
        timings["benign"] = time.time() - t0
        print(f"[sweep] benign leg done in {timings['benign']:.1f}s")

    # record per-leg wall-clock (the cost-model input) + the absent set.
    (out_root / f"timings_{cond.label}.json").write_text(json.dumps({
        "condition": cond.label, "timings_s": timings,
        "absent_families": absent,
    }, indent=2))
    print(f"[sweep] condition {cond.label} generation complete: {timings}")


# --------------------------------------------------------------------------- #
# plan — CPU dry-run: the reviewable sweep plan (no GPU)                        #
# --------------------------------------------------------------------------- #

def cmd_plan(args) -> None:
    conditions = sweep.load_conditions(args.conditions)
    print(f"[sweep] {len(conditions)} conditions (--array=0-{len(conditions) - 1}); "
          f"ONE shared attacker pool serves every adaptive leg.\n")
    for i, c in enumerate(conditions):
        ps, as_ = _present_absent(c.static)
        pa, aa = _present_absent(c.adaptive)
        ph, ah = _present_absent(c.held_out)
        print(f"[{i:2d}] {c.label}  cell={c.cell} coeff={c.coeff or '(cfg)'} mask={c.mask}")
        print(f"     target_engine={c.target_engine} attacker_engine={c.attacker_engine}")
        print(f"     static:   present={ps} absent={as_}")
        print(f"     adaptive: present={pa} absent={aa}")
        print(f"     held_out: present={ph} absent={ah}")
        print(f"     benign:   {list(c.benign)}")
        print(f"     stamps: static={c.static_stamp()['attacker_engine']} "
              f"adaptive={c.adaptive_stamp()['attacker_engine']} "
              f"held_out={c.held_out_stamp()['attacker_engine']} "
              f"benign={c.benign_stamp()['attacker_engine']}")
    any_absent = any(_present_absent(c.static)[1] or _present_absent(c.adaptive)[1]
                     or _present_absent(c.held_out)[1] for c in conditions)
    if any_absent:
        print("\n[sweep] NOTE: some families are absent on this branch "
              "(bon+gcg+adaptive only). The full 8 static families live on the "
              "attack-generalization / twosi-stage3-run lineage — merge them in "
              "or run with --allow-missing-families.")


# --------------------------------------------------------------------------- #
# aggregate — the grading host: engine-matched pairing across conditions (see sweep.py)  #
# --------------------------------------------------------------------------- #

def _make_baseline_of(label_to_cell: dict):
    """Build ``baseline_of(leg, condition_label) -> baseline_family`` from the
    conditions table. Spec baselines: JB robustness (static/adaptive) is
    contrasted vs the undefended ``none`` cell; benign over-refusal vs the
    MATCHING ``prompted__<si>`` cell, the SI taken from the
    object's own cell id. The returned family string is a CELL ID and equals
    the ``baseline_family`` stamped on the matching baseline replica, so the
    aggregator matches object -> replica on (family, template, serving_path).
    A baseline condition itself (cell none / prompted__*) has no comparator."""
    from sisteer.eval.defense_cell import si_of

    def baseline_of(leg: str, condition_label: str):
        cell = label_to_cell.get(condition_label, condition_label)
        if cell == "none" or cell.startswith("prompted__"):
            return None
        if leg == "benign":
            si = si_of(cell)
            return f"prompted__{si}" if si else None
        return "none"

    return baseline_of


def cmd_aggregate(args) -> None:
    # The metric extraction from the JB aggregates + benign summaries is the
    # existing per-lane read (jailbreak_run aggregate + cell_controls emit);
    # this driver's job is the ENGINE-MATCHED PAIRING across them. It reads the
    # engine-stamp sidecars, builds sweep.MetricRow per (leg, member,
    # condition), and calls sweep.build_comparison_table, which REFUSES any
    # cross-engine pair before differencing.
    rows_in = json.loads(Path(args.rows).read_text()) if args.rows else None
    if rows_in is None:
        raise SystemExit(
            "[sweep] aggregate needs --rows <metrics.json>: a list of "
            "{leg, member, condition, metric} objects extracted from the "
            "per-lane aggregates (jailbreak_run aggregate for JB ASR; "
            "cell_controls summaries for benign). The engine stamp for each is "
            "read from its sidecar under --out-root. This split keeps the "
            "authoritative metric read in each lane's own aggregate and the "
            "pairing enforcement here.")
    out_root = Path(args.out_root)
    _conditions = sweep.load_conditions(args.conditions)
    label_to_cell = {c.label: c.cell for c in _conditions}
    label_to_mask = {c.label: c.mask for c in _conditions}
    baseline_of = _make_baseline_of(label_to_cell)
    metric_rows = []
    for r in rows_in:
        sc = read_sidecar(out_root, r["condition"], r["leg"], r["member"])
        metric_rows.append(sweep.MetricRow(
            leg=r["leg"], member=r["member"], condition=r["condition"],
            metric=float(r["metric"]), stamp=sc["engine_stamp"],
            baseline_family=sc.get("baseline_family")))
    table = sweep.build_comparison_table(metric_rows, baseline_of=baseline_of)
    served = [{
        "leg": c.leg, "member": c.member, "condition": c.condition,
        "baseline": c.baseline, "condition_metric": c.condition_metric,
        "baseline_metric": c.baseline_metric, "delta": c.delta,
        "target_engine": c.target_engine, "attacker_engine": c.attacker_engine,
        "serving_path": c.serving_path, "chat_template": c.chat_template,
        "object_sha": c.object_sha,
        # the attack regime is on every served row (fixed_prompt for this sweep).
        "attack_regime": c.attack_regime,
    } for c in table]
    # ASR breakout beside the per-lane by_category_group goal axis: surface the
    # defense OBJECT class (from the registry) and the serving PHASE as axes a
    # reader must not pool over. build_asr_table only ever emits the per-axis
    # breakouts, never a single spanning headline.
    def _oc(c):
        return _object_class_label(label_to_cell.get(c.condition, c.condition))

    def _ph(c):
        # phase = the condition's serving mask: all-position vs decode-only. A
        # DISTINCT deployment axis (not a generalisation axis) — never pooled.
        return label_to_mask.get(c.condition, "all")

    def _pr(c):
        # phase_relation (matched vs cross_phase), stamped at serve time by the
        # empirical three-way phase check. matched and cross_phase answer
        # DIFFERENT questions and must never be pooled. All-position rows are
        # matched by construction; the sidecar carries the verified label for
        # decode-only rows.
        sc = read_sidecar(out_root, c.condition, c.leg, c.member)
        return sc.get("phase_relation", "matched")

    asr_table = sweep.build_asr_table(
        table, object_class_of=_oc, phase_of=_ph, phase_relation_of=_pr)
    out_path = Path(args.out) if args.out else (out_root / "sweep_comparison.json")
    out_path.write_text(json.dumps(
        {"comparisons": served, "asr_table": asr_table}, indent=2))
    print(f"[sweep] wrote {out_path}: {len(served)} engine-matched comparisons "
          "(every cross-engine pair refused before differencing).")
    print(f"[sweep] ASR by object_class: "
          f"{ {k: v['n'] for k, v in asr_table['by_object_class'].items()} }; "
          f"by phase: "
          f"{ {k: v['n'] for k, v in asr_table.get('by_phase', {}).items()} }; "
          f"by attack_regime: "
          f"{ {k: v['n'] for k, v in asr_table.get('by_attack_regime', {}).items()} } "
          "(pooled ASR is only ever the breakout, never a spanning headline).")
    print(f"[sweep] attack_regime coverage: {asr_table['attack_regime_coverage']}")


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--conditions", required=True, help="conditions TSV")
    g.add_argument("--config", default="configs/two_si.yaml")
    g.add_argument("--out-root", required=True)
    g.add_argument("--defenses-dir", default="")
    g.add_argument("--task-id", type=int, default=None,
                   help="sbatch array index -> condition (0-based)")
    g.add_argument("--condition", default="", help="condition label (alt to --task-id)")
    g.add_argument("--limit", type=int, default=100,
                   help="per-benign-control item cap (spec n=100)")
    g.add_argument("--judge-model", default="claude-haiku-4-5")
    g.add_argument("--attacker-endpoint", default="http://127.0.0.1:8000",
                   help="the ONE shared vLLM attacker pool endpoint")
    g.add_argument("--allow-missing-families", action="store_true",
                   help="run present families and RECORD absent ones instead of "
                        "failing loudly (absent = not in the live registry)")
    g.add_argument("--bon-n-max", type=int, default=None,
                   help="override bon.n_max for THIS run without editing the "
                        "shared configs/two_si.yaml (screening sweep: 64; 256 "
                        "reserved for survivors). None -> use the config value.")
    g.add_argument("--max-goals", type=int, default=None,
                   help="cap the JB (static/adaptive/held_out) goal set to the "
                        "first N — for the SMOKE only; None (default) runs the "
                        "full confirmatory set. Benign uses --limit.")
    g.add_argument("--bon-batch-size", type=int, default=None,
                   help="FIXED GPU batch width for the BoN leg (the measured "
                        "width; the ragged final chunk is padded to it). None -> "
                        "bon_generate's default, DEFAULT_BON_BATCH_SIZE (64); "
                        "pass 1 explicitly for the legacy unbatched path. "
                        "Recorded on every BoN row + the BoN engine stamp so a "
                        "width is never compared across widths unnoticed. HF "
                        "engine only.")
    g.add_argument("--gen-engine", choices=("hf", "served"), default=None,
                   help="generation engine for the WHOLE fixed-prompt leg "
                        "(the non-BoN static + held-out families AND, unless "
                        "--bon-engine overrides it, BoN): 'hf' = in-process "
                        "model.generate; 'served' = concurrent fan-out through "
                        "the ROUTING PROXY over the vLLM server (per-cell "
                        "cache_salt isolation proof + steering-bites probe for "
                        "steered cells enforced before generating; rows stamped "
                        "engine=vllm_served, never mixed with HF rows). Default "
                        "(unset) = hf — opting in is explicit and the unopted "
                        "path is byte-identical. Needs --bon-proxy-url + "
                        "--bon-routing-table (they name the proxy for every "
                        "served family, not bon alone).")
    g.add_argument("--gen-engine-hf-families", default="",
                   help="csv of non-BoN families to force onto HF even when "
                        "--gen-engine served (per-family opt-out for a family "
                        "that cannot be served cleanly). Their sidecar RECORDS "
                        "engine=hf — a deliberate, recorded per-family engine, "
                        "never a silent mix. Empty (default) = serve every "
                        "non-BoN family.")
    g.add_argument("--bon-engine", choices=("hf", "served"), default=None,
                   help="OVERRIDE the BoN leg's engine only (the rest of the "
                        "fixed-prompt leg follows --gen-engine). 'hf' = "
                        "in-process model.generate; 'served' = fan-out through "
                        "the ROUTING PROXY (bon_generate_served). Default "
                        "(unset) defers to --gen-engine, else hf. Passing "
                        "'--bon-engine served' ALONE reproduces the pre-"
                        "generalisation behaviour exactly: BoN served, every "
                        "other family HF.")
    g.add_argument("--bon-proxy-url", default=None,
                   help="the routing-proxy endpoint for --bon-engine served "
                        "(e.g. http://localhost:8814) — the PROXY, never a "
                        "bare vLLM URL: routing, per-cell cache_salt and "
                        "SI-prefix verification are the proxy's to own.")
    g.add_argument("--bon-routing-table", default=None,
                   help="path to the serving launch's export_routing_table "
                        "JSON — the FULL route set (the mutual salt-isolation "
                        "proof needs every cell the run serves, not just this "
                        "one's).")
    g.add_argument("--bon-max-concurrency", type=int, default=None,
                   help="in-flight request ceiling for --bon-engine served; "
                        "None -> bon_served.DEFAULT_MAX_CONCURRENCY, which is "
                        "the 'bon' entry's max_num_seqs in "
                        "sisteer.capeval.serving.geometry (the per-eval "
                        "geometry table). Pass the launched max_num_seqs when "
                        "the server was stood up at a different width.")
    g.add_argument("--served-max-model-len", type=int, default=None,
                   help="the served launch's max_model_len as PROVENANCE + a "
                        "cross-check for the client's once-only max_tokens "
                        "clamp (served_window): the window the server states "
                        "in its rejection governs the clamp; a disagreement "
                        "is logged and recorded as window_mismatch. "
                        "petri_families_served_integration passes its "
                        "WAVE_MAX_MODEL_LEN; omitted = None.")

    p = sub.add_parser("plan")
    p.add_argument("--conditions", required=True)

    a = sub.add_parser("aggregate")
    a.add_argument("--conditions", required=True)
    a.add_argument("--out-root", required=True)
    a.add_argument("--rows", default="",
                   help="metrics.json: [{leg, member, condition, metric}, ...] "
                        "from the per-lane aggregates; engine stamps read from "
                        "sidecars under --out-root")
    a.add_argument("--out", default="")

    args = ap.parse_args()
    {"generate": cmd_generate, "plan": cmd_plan,
     "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()
