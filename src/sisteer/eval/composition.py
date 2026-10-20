"""Composition eval: joint k-compliance, averaged over k! orderings.

The headline head-to-head: Radevski-style compose-separately-then-combine
at inference vs MONOLITHIC JOINT DISTILLATION (one object trained on the
joined "do X and Y (and Z)" instruction) vs prompted-NL. The composed arm
runs on exactly two object kinds: the residual vector (additive sum — the
CAA-style baseline Radevski show collapses) and token_faithful with the
learned <and> token ([E_x, e_b1, e_<and>, e_b2] layout).

This module owns the EVAL side: the behavior library, target sets, the
joint checker, ordering enumeration, and the arm loop. The composition
OPERATORS (compose_additive / compose_tokens / train_and_token) live in
extraction/compose.py and are passed in as `compose_fn` — a callable
(objs_in_order, order) -> InterventionObject. A compose_fn with attribute
`order_invariant = True` (the additive sum) is evaluated on one ordering
and the invariance noted once in the record instead of re-generating k!
identical arms.

Behaviors use repo checkers (language/lowercase/max_words + the
sentence_count checker registered in checkers.py); metric = STRICT joint
compliance. Language behaviors are mutually exclusive, so generated
targets never combine two languages (they would be unsatisfiable).
"""

from __future__ import annotations

import itertools
import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..data.instructions import BEHAVIORS as _BEHAVIOR_SIS
from ..data.instructions import joined_behavior_instruction as _joined_si
from ..models import build_chat_prompt
from . import checkers


@dataclass(frozen=True)
class Behavior:
    name: str
    instruction: str  # single-behavior SI text (per-behavior training + NL clause)
    clause: str  # short clause used inside joined instructions
    checker: str
    checker_kwargs: dict


# Short clauses used inside joined "(1) ...; (2) ..." instructions. Only the
# clause wording is owned here; the SI text, checker, and checker kwargs come
# from data/instructions.BEHAVIORS — the ONE behavior library (the two
# copies had drifted, so the eval could score a different instruction than
# the one the objects were trained on).
_CLAUSES: dict[str, str] = {
    "lang_fr": "respond only in French",
    "lang_es": "respond only in Spanish",
    "lang_de": "respond only in German",
    "lowercase": "use lowercase letters only",
    "max_50_words": "keep the response to at most 50 words",
    "three_sentences": "write exactly three sentences",
}

BEHAVIORS: dict[str, Behavior] = {
    name: Behavior(
        name,
        si.text,
        _CLAUSES[name],
        si.checker,
        dict(si.checker_kwargs),
    )
    for name, si in _BEHAVIOR_SIS.items()
}

if set(_CLAUSES) != set(_BEHAVIOR_SIS):  # pragma: no cover - import-time guard
    raise RuntimeError(
        "composition clauses out of sync with data/instructions.BEHAVIORS: "
        f"{sorted(set(_CLAUSES) ^ set(_BEHAVIOR_SIS))}"
    )

_LANGUAGES = {"lang_fr", "lang_es", "lang_de"}


def compatible(names: tuple[str, ...]) -> bool:
    """A target is satisfiable iff it names at most one language."""
    return sum(n in _LANGUAGES for n in names) <= 1


def default_targets(seed: int = 0) -> dict[str, list[tuple[str, ...]]]:
    """8 seen 2-pairs (the <and> train set), 4 unseen 2-pairs, 4 zero-shot
    triples — deterministic, compatibility-filtered (plan defaults)."""
    names = sorted(BEHAVIORS)
    pairs = [p for p in itertools.combinations(names, 2) if compatible(p)]
    triples = [t for t in itertools.combinations(names, 3) if compatible(t)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    rng.shuffle(triples)
    if len(pairs) < 12:
        raise ValueError("need >= 12 compatible pairs for the 8+4 split")
    return {"seen": pairs[:8], "unseen": pairs[8:12], "triples": triples[:4]}


def canonical_order(names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """THE preregistered canonical clause order: sorted(names).

    Every joined instruction built in this module — and the monolithic
    objects the driver trains via data/instructions.joined_behavior_
    instruction — uses this order for its canonical form, so the
    monolithic arm's single ordering is comparable across runs and the
    k!-averaged arms' canonical-order entry is identifiable (the
    per_ordering item whose order is [0..k-1] under sorted names)."""
    return tuple(sorted(names))


def joined_instruction_text(names: tuple[str, ...] | list[str]) -> str:
    """The monolithic 'follow all of these rules' instruction, clause
    order as given.

    Delegates to data/instructions.joined_behavior_instruction — the SAME
    text builder the compiled monolithic objects distill from:
    rendering a different (shorter) clause wording here made the
    prompted-NL arm answer a different instruction than the compiled arms
    trained on, confounding the head-to-head. Never duplicate the string;
    the one source of truth is data/instructions."""
    return _joined_si(list(names)).text


def check_behavior(response: str, name: str) -> bool:
    b = BEHAVIORS[name]
    return checkers.check(b.checker, response, **b.checker_kwargs)


def joint_compliance(response: str, names: tuple[str, ...] | list[str]) -> bool:
    """Strict AND over all k behaviors — the composition metric."""
    return all(check_behavior(response, n) for n in names)


def orderings(k: int) -> list[tuple[int, ...]]:
    return list(itertools.permutations(range(k)))


def _score_responses(responses: list[str], names) -> dict:
    joint = [joint_compliance(r, names) for r in responses]
    per_behavior = {
        n: sum(check_behavior(r, n) for r in responses) / max(len(responses), 1)
        for n in names
    }
    return {
        "joint_strict": sum(joint) / max(len(joint), 1),
        "per_behavior": per_behavior,
    }


def _done_keys(out_jsonl: Path) -> set[tuple]:
    done = set()
    if out_jsonl.exists():
        for line in out_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line from a SIGKILL mid-append
            done.add((tuple(rec["behaviors"]), rec["arm"]))
    return done


def eval_composition(
    model,
    tok,
    *,
    targets: dict[str, list[tuple[str, ...]]],
    prompts: list[str],
    out_dir: Path,
    per_behavior_objs: dict[str, object] | None = None,
    compose_fn: Callable[[list, tuple[int, ...]], object] | None = None,
    monolithic_objs: dict[tuple[str, ...], object] | None = None,
    arms: tuple[str, ...] = ("prompted", "composed", "monolithic"),
    scale: float = 1.0,
    batch_size: int = 8,
    max_new_tokens: int = 200,
    generate_fn: Callable[[list[str], object | None], list[str]] | None = None,
) -> list[dict]:
    """Score each arm on joint k-compliance, averaged over k! orderings.

    - prompted: joined-NL instruction as system prompt — the SAME
      joined_behavior_instruction text the compiled objects trained on,
      clause order permuted (orderings enumerated).
    - composed: compose_fn([per_behavior_objs[n] for n in permuted], order)
      per ordering; a compose_fn with order_invariant=True is evaluated
      once and marked.
    - monolithic: monolithic_objs[canonical_order(target)] — one object
      trained on the joined instruction in the preregistered canonical
      order (sorted names); keys must be canonical tuples.

    Each record carries `object_kind` (the intervention object's kind, or
    None for prompted) and `ordering` ("canonical" for monolithic,
    "kfact_averaged" for enumerated arms, "order_invariant" for the
    additive compose) so k!-averaged and canonical-order numbers stay
    separable at analysis: rec["joint_strict"] is the k! average, and the
    canonical-order number is the per_ordering entry whose order is
    [0..k-1] (names are canonical/sorted).

    generate_fn(rendered_prompts, obj_or_None) -> responses is the test
    seam; default generates on `model` (token-kind objects injected).
    Appends one record per (target, arm) to out_dir/results.jsonl and
    skips completed ones on restart.
    """
    from ..generate import generate_batch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "results.jsonl"

    def _gen(rendered: list[str], obj) -> list[str]:
        if generate_fn is not None:
            return generate_fn(rendered, obj)
        # Inject for EVERY token kind — token (after-BOS / composed <and>),
        # token_appended (monolithic faithful). `== "token"` missed
        # token_appended, silently evaluating the monolithic faithful arm as
        # unsteered.
        if obj is not None and (getattr(obj, "kind", "") or "").startswith("token"):
            rendered = [obj.inject(r) for r in rendered]
        ctx = obj.apply(model, scale=scale) if obj is not None else nullcontext()
        with ctx:
            return generate_batch(
                model, tok, rendered, batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )

    done = _done_keys(out_jsonl)
    # The composed and monolithic arms always generate on the plain prompts
    # (no system text), so render them once — only the prompted arm's
    # rendering varies with the ordering.
    plain_rendered = [build_chat_prompt(tok, p) for p in prompts]
    records: list[dict] = []
    for kind, target_list in targets.items():
        for names in target_list:
            names = tuple(names)
            if not compatible(names):
                raise ValueError(f"unsatisfiable target {names}: two languages")
            k = len(names)
            for arm in arms:
                if (names, arm) in done:
                    continue
                per_ordering: list[dict] = []
                order_invariant = False
                object_kind = None
                if arm == "prompted":
                    order_list = orderings(k)
                    for order in order_list:
                        permuted = tuple(names[i] for i in order)
                        system = joined_instruction_text(permuted)
                        rendered = [
                            build_chat_prompt(tok, p, system=system) for p in prompts
                        ]
                        scored = _score_responses(_gen(rendered, None), names)
                        scored["order"] = list(order)
                        per_ordering.append(scored)
                elif arm == "composed":
                    if compose_fn is None or per_behavior_objs is None:
                        raise ValueError(
                            "composed arm needs compose_fn and per_behavior_objs"
                        )
                    order_invariant = bool(getattr(compose_fn, "order_invariant", False))
                    order_list = orderings(k)[:1] if order_invariant else orderings(k)
                    for order in order_list:
                        permuted = tuple(names[i] for i in order)
                        objs = [per_behavior_objs[n] for n in permuted]
                        composed = compose_fn(objs, order)
                        object_kind = getattr(composed, "kind", None)
                        scored = _score_responses(
                            _gen(plain_rendered, composed), names)
                        scored["order"] = list(order)
                        per_ordering.append(scored)
                elif arm == "monolithic":
                    if monolithic_objs is None:
                        raise ValueError("monolithic arm needs monolithic_objs")
                    # canonical (sorted) order is THE preregistered ordering
                    # for the monolithic arm; keys must be canonical.
                    canon = canonical_order(names)
                    if canon not in monolithic_objs:
                        raise KeyError(
                            f"monolithic_objs lacks the CANONICAL key {canon} "
                            f"for target {names} — key monolithic objects by "
                            "canonical_order(names) (sorted), the "
                            "preregistered clause order they train on"
                        )
                    obj = monolithic_objs[canon]
                    object_kind = getattr(obj, "kind", None)
                    scored = _score_responses(_gen(plain_rendered, obj), names)
                    # indices of the canonical clause order within `names`
                    # (identity [0..k-1] when names is already canonical)
                    scored["order"] = [names.index(n) for n in canon]
                    per_ordering.append(scored)
                else:
                    raise ValueError(f"unknown arm {arm!r}")

                joint = [s["joint_strict"] for s in per_ordering]
                if arm == "monolithic":
                    ordering = "canonical"
                elif order_invariant:
                    ordering = "order_invariant"
                else:
                    ordering = "kfact_averaged"
                rec = {
                    "behaviors": list(names),
                    "kind": kind,
                    "k": k,
                    "arm": arm,
                    "object_kind": object_kind,
                    "ordering": ordering,
                    "joint_strict": sum(joint) / len(joint),
                    "n_orderings_scored": len(per_ordering),
                    "order_invariant": order_invariant,
                    "per_ordering": per_ordering,
                }
                records.append(rec)
                with out_jsonl.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
    return records
