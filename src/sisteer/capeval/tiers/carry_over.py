"""Cap-raise reruns of the judged over-refusal controls (XSTest / OR-Bench):
carry over the generations a lower cap did NOT truncate, regenerate only
the items that hit it, and run a small parity control on the carried set.

WHY ONLY THE AT-CAP ITEMS MOVE (Optimisation 1). Every judged control
decodes GREEDILY (``controls_gen_config``: ``do_sample: False``; the served
driver posts ``temperature 0.0``), and the cap is a hard stop on the number
of generated tokens (HF ``max_new_tokens`` / vLLM ``max_tokens``), never a
knob the model can see, so only a response still running at the cap has a
tail the cap removed. MEASURED CAVEAT (2026-09-09, memory note
``text-diff-is-a-hypersensitive-readout``): greedy decoding on this
project's HF path is NOT a deterministic replay across jobs — 17.9% of
below-cap items came back with different text between two runs, forks early,
and seven at-cap items came back shorter — so the design does NOT rest on
"a terminated response is byte-identical under a higher cap". It rests on
the weaker fact that a terminated text is a complete draw the cap did not
touch: KEEPING it (rather than redrawing it) leaves the cell a mix of two
exchangeable greedy draws, and the parity control below measures exactly
that mixing effect. No byte-identity is asserted anywhere in this module;
per-item text relationships (identical / prefix / fork) are RECORDED by
``tiers/cap_ladder`` as the noise floor a label-flip analysis reads against.
The spools record no ``finish_reason`` (``{"item_id", "text"}`` only), so
the at-cap set is recovered by RE-TOKENIZING each text with the cell's
tokenizer under the geometry measurement's own convention — completion
tokens as vLLM would count them, ``min(ntok(text) + 1, cap)`` (one EOS on a
stop finish, clamped at the cap) — and calling an item at-cap when that
count reaches the cap (:data:`AT_CAP_RULE`). The rule is deliberately the
measurement's, so the rerun's at-cap set is the very set the measurement
reported; the +1 makes a text that re-tokenizes to ``cap - 1`` at-cap too (a
response that spent its last budgeted token on EOS is indistinguishable
from one cut one token short — regenerating a terminated item is harmless,
missing a truncated one is not).

WHICH AT-CAP ITEMS MOVE, AND WHEN (the standing rule, ``tiers/cap_ladder``).
Not every at-cap item is redrawn, and not every cell climbs: a (cell,
readout) climbs to the next rung only when its TRUNCATED share exceeds 1%;
degenerate at-cap texts (decode loops) never move — they are carried
explicitly as ``non_terminating``; boundary-flagged at-cap items are out of
the trigger but redrawn when the cell climbs; and a readout that does not
climb carries its at-cap truncated texts as the ``residual`` the report
names. :func:`build_task_block` applies the rule; :func:`seed_spool`
re-checks it on the served side (a "non_terminating" text must be
degenerate; a residual is carried only when the block does not climb).

WHY THE AT-CAP ITEMS ARE REGENERATED FROM SCRATCH, NOT WARM-STARTED
(Optimisation 2, refused). Seeding the assistant turn with the truncated
text and continuing it IS expressible on the served path — a continued
final message declared through the proxy's ``petri_response_prefill`` gate —
and the ``positions="response"`` mask DOES cover the seed there
(``lens_adapters._row_selection``: the prefill mask runs from the LAST
header occurrence to the end of the prefill, decode rows are all-ones;
pinned by ``tests/test_warm_start_mask_coverage.py``). Two other things
break the equivalence, for EVERY cell, steered or not:

* the server re-renders the seed through the chat template, and HF's
  ``continue_final_message`` cuts the render at the seed's TRIMMED tail
  (``render_jinja_template``: ``rstrip`` when the template trimmed the
  content) — every served template here (Llama-3.x, gemma-4, Qwen3.5)
  applies ``| trim`` to assistant content, so a truncated text ending in
  whitespace loses that whitespace and the continuation starts from a
  different state than the one the model was in at token 512;
* the seed is ``encode(decode(ids))``, not ``ids``: a cut inside a
  multi-byte character or across a merge boundary re-tokenizes differently,
  so the KV state the continuation conditions on is not the original.

Both change the intervention's INPUT, not its mask. Against that, the
saving is at most the first 512 of up to 2048 decode steps on the at-cap
items only. Not worth an equivalence that cannot be proven; the at-cap
items regenerate from the prompt. (A ``phase="prefill"``-only kind would
additionally see its seed STEERED under warm-start where the original
decode did not — none of the in-scope kinds, psr / lora_all, is prefill-only,
but the hazard is pinned by the same test.)

THE ENGINE MIX IS EXPLICIT, AND MEASURED (Gate 3). The carried texts came
from the HF in-process path; the regenerated ones come from the served
(vLLM + lens) path — the standing rule for steered evaluation. A cell's
final spool therefore holds both, and the manifest says exactly which items
came from where (``carry_over`` block). The PARITY CONTROL measures what
the engine change alone does: a seeded sample of the CARRIED (terminated)
items is regenerated on the served path too, into a separate spool, and
judged against the HF labels of the same items — the per-item label-flip
rate is the size of the engine effect the at-cap regeneration inherits.

The bundle (:data:`BUNDLE_SCHEMA`) is built on the grading host from the HF store
(``scripts/capctrl_cap2048_plan.py``) and consumed by the served driver
(``scripts/served_capability_run.py generate --carry-over``), which binds it
to the live spec — same item set (``items_sha``), same system (``system_sha``),
same cell and model revision — and RE-CHECKS every carried text against the
SERVED tokenizer before seeding it: a text the served tokenizer counts at
the cap is a refusal, never a silent carry.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from sisteer.capeval.manifest import canonical_json, sha256_hex16

from .at_cap import AT_CAP_RULE, completion_tokens, is_at_cap, ntok  # noqa: F401 — re-exported: carry_over was the public home
from .judged import GenSpool, JudgedSpec, rows_content_sha

#: the bundle's schema literal — a reader refuses any other
BUNDLE_SCHEMA = "capctrl_carry_over/1"
#: the parity spool's role marker (its header carries the base header + this)
PARITY_ROLE = "parity_control"


class CarryOverError(RuntimeError):
    """A bundle that does not bind to the live spec, or a carried text the
    served tokenizer counts at the cap — loud, never a silent carry."""


# the at-cap rule (AT_CAP_RULE, ntok, completion_tokens, is_at_cap) is defined
# ONCE in at_cap.py — stdlib-only, so cap_ladder (and the shell's ladder_gate.py
# that imports it per gate call) never pays this module's judged -> torch chain


@dataclass
class Classification:
    """One spool's split under :data:`AT_CAP_RULE`."""

    carried: Dict[str, str]                 # item_id -> text, below the cap
    at_cap: List[str]                       # item ids at the cap (regenerate)
    ntok: Dict[str, int] = field(default_factory=dict)  # item_id -> re-tokenized length


def classify(entries: Mapping[str, str], tok, cap: int) -> Classification:
    """Split ``{item_id: text}`` into carried / at-cap under the rule."""
    carried: Dict[str, str] = {}
    at_cap: List[str] = []
    lengths: Dict[str, int] = {}
    for iid, text in entries.items():
        n = ntok(tok, text or "")
        lengths[iid] = n
        if min(n + 1, int(cap)) >= int(cap):
            at_cap.append(iid)
        else:
            carried[iid] = text
    return Classification(carried=carried, at_cap=sorted(at_cap), ntok=lengths)


def warm_start_hazards(tok, texts: Sequence[str]) -> Dict[str, int]:
    """How many of ``texts`` would NOT round-trip as a warm-start seed:
    a trailing-whitespace tail (``continue_final_message`` rstrips it under
    a trimming template), a U+FFFD replacement character (a cut inside a
    multi-byte character), and a text whose ``decode(encode(text))`` is not
    itself. Diagnostics for the plan; the refusal of warm-start does not
    depend on the counts."""
    trailing_ws = sum(1 for t in texts if t and t != t.rstrip())
    replacement = sum(1 for t in texts if "�" in (t or ""))
    non_idempotent = 0
    for t in texts:
        ids = tok(t or "", add_special_tokens=False)["input_ids"]
        if tok.decode(ids, skip_special_tokens=True) != (t or ""):
            non_idempotent += 1
    return {"n": len(texts), "trailing_whitespace": trailing_ws,
            "replacement_char": replacement,
            "encode_decode_non_idempotent": non_idempotent}


# --------------------------------------------------------------------------- #
# the bundle                                                                   #
# --------------------------------------------------------------------------- #


def parity_sample(carried_ids: Sequence[str], n: int, seed: int) -> List[str]:
    """A seeded sample of the carried ids (sorted input, so the draw depends
    on the SET, not on spool order)."""
    ids = sorted(carried_ids)
    if n <= 0 or not ids:
        return []
    return sorted(random.Random(int(seed)).sample(ids, min(int(n), len(ids))))


def build_task_block(*, control: str, task: str, header: Mapping[str, Any],
                     entries: Mapping[str, str], tok, cap: int,
                     parity_n: int, parity_seed: int,
                     source_caps: Optional[Mapping[str, int]] = None,
                     other_draws: Optional[Mapping[str, Sequence[Tuple[int, str]]]] = None,
                     climbed: bool = False,
                     threshold: Optional[float] = None,
                     ceiling: Optional[int] = None,
                     force: bool = False,
                     hop_timeouts_s: Optional[Mapping[str, Optional[float]]] = None,
                     ) -> Dict[str, Any]:
    """One control's bundle block from the source spool's header + entries,
    partitioned under the standing truncation rule (``tiers/cap_ladder``):

    * ``carried`` — below the cap (at the item's own source cap): text
      carried unchanged;
    * ``regenerate`` — at cap and TRUNCATED (not degenerate), plus the
      boundary-flagged at-cap items: redrawn at the new cap — but ONLY when
      the control's rung verdict is ``escalate`` (truncated share above the
      threshold, below the ceiling) or ``force`` is set;
    * ``non_terminating`` — at cap and DEGENERATE (a decode loop): text
      carried unchanged and reported separately; a larger cap cannot fix it;
    * ``residual`` — at cap and truncated, but the control does NOT climb
      (share at or below the threshold, or at the ceiling): text carried
      unchanged and named in the report as the residual.

    ``source_caps`` is the per-item cap that produced each text (a source
    store that was itself a rung; default ``cap`` for every item);
    ``other_draws`` is the boundary evidence (``{item_id: [(cap, text)]}``);
    ``climbed`` says the source store was itself reached by climbing (so a
    ceiling verdict reads ``ceiling``, not ``ceiling_at_outset``). The rung
    verdict rides in the block as ``rung``."""
    from . import cap_ladder as cl

    kw: Dict[str, Any] = {}
    if threshold is not None:
        kw["threshold"] = float(threshold)
    if ceiling is not None:
        kw["ceiling"] = int(ceiling)
    status, records = cl.measure_rung(entries, tok, cap, source_caps=source_caps,
                                      other_draws=other_draws, climbed=climbed,
                                      hop_timeouts_s=hop_timeouts_s, **kw)
    caps = {iid: int((source_caps or {}).get(iid, cap)) for iid in entries}
    carried = {iid: entries[iid] for iid, r in records.items() if not r.at_cap}
    nonterm = {iid: entries[iid] for iid in status.non_terminating_ids}
    at_cap_open = sorted(status.truncated_ids + status.boundary_ids)
    climb = bool(status.escalate or (force and at_cap_open))
    to_regen = at_cap_open if climb else []
    residual = {} if climb else {iid: entries[iid] for iid in at_cap_open}
    assert set(carried) | set(to_regen) | set(nonterm) | set(residual) == set(entries)
    return {
        "control": control,
        "task": task,
        "items_sha": header["items_sha"],
        "system_sha": header.get("system_sha"),
        "source_gen_cfg": dict(header["gen_cfg"]),
        "n_items": len(entries),
        "n_carried": len(carried),
        "n_regenerate": len(to_regen),
        "n_non_terminating": len(nonterm),
        "n_residual": len(residual),
        "carried": dict(sorted(carried.items())),
        "regenerate": list(to_regen),
        "non_terminating": dict(sorted(nonterm.items())),
        "residual": dict(sorted(residual.items())),
        # the at-cap texts being redrawn, for the redraw record (prefix /
        # fork / boundary landing) — raw generations; the bundle is 0600
        "source_at_cap": {iid: entries[iid] for iid in to_regen},
        "source_caps": dict(sorted(caps.items())),
        "forced": bool(force and climb and not status.escalate),
        "rung": status.as_dict(),
        "parity_sample": parity_sample(carried, parity_n, parity_seed),
    }


def bundle_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_bundle(path: Path, bundle: Mapping[str, Any]) -> str:
    """Write the bundle (canonical JSON, mode 0600 — it carries raw
    generations) and return its sha256."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(canonical_json(dict(bundle)) + "\n", encoding="utf-8")
    p.chmod(0o600)
    return bundle_sha256(p)


def load_bundle(path: Path, *, cell_id: str, model_revision: Optional[str],
                new_cap: int, limit: Optional[int],
                expect_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Read and BIND a bundle to the run consuming it: schema, cell id, model
    revision, the raised cap, the --limit and (when given) the file digest
    must all agree, else :class:`CarryOverError`."""
    p = Path(path)
    if not p.is_file():
        raise CarryOverError(f"carry-over bundle {p} does not exist")
    if expect_sha256 is not None:
        got = bundle_sha256(p)
        if got != expect_sha256:
            raise CarryOverError(
                f"carry-over bundle {p} sha256 {got} != the expected "
                f"{expect_sha256} — not the bundle the plan pinned")
    b = json.loads(p.read_text(encoding="utf-8"))
    if b.get("schema") != BUNDLE_SCHEMA:
        raise CarryOverError(
            f"{p}: schema {b.get('schema')!r} != {BUNDLE_SCHEMA!r}")
    if b.get("cell_id") != cell_id:
        raise CarryOverError(
            f"{p}: bundle is for cell {b.get('cell_id')!r}, this run serves "
            f"{cell_id!r} — refusing to seed one cell's spool with another's "
            "generations")
    if (b.get("model_revision") or None) != (model_revision or None):
        raise CarryOverError(
            f"{p}: bundle model_revision {b.get('model_revision')!r} != the "
            f"served {model_revision!r}")
    if int(b.get("new_cap", -1)) != int(new_cap):
        raise CarryOverError(
            f"{p}: bundle new_cap {b.get('new_cap')!r} != this run's "
            f"max_new_tokens {new_cap} — the bundle was planned for another cap")
    src_limit = (b.get("source") or {}).get("limit")
    if (src_limit or None) != (limit or None):
        raise CarryOverError(
            f"{p}: bundle source limit {src_limit!r} != this run's --limit "
            f"{limit!r} — the item set would differ")
    if b.get("at_cap_rule") != AT_CAP_RULE:
        raise CarryOverError(
            f"{p}: at_cap_rule {b.get('at_cap_rule')!r} != {AT_CAP_RULE!r}")
    if not isinstance(b.get("tasks"), dict) or not b["tasks"]:
        raise CarryOverError(f"{p}: bundle carries no tasks")
    b["_sha256"] = bundle_sha256(p)
    b["_path"] = str(p)
    return b


# --------------------------------------------------------------------------- #
# seeding the served spool                                                     #
# --------------------------------------------------------------------------- #


def _system_sha(spec: JudgedSpec) -> Optional[str]:
    return sha256_hex16(spec.system.encode("utf-8")) if spec.system else None


def seed_spool(spool: GenSpool, spec: JudgedSpec, block: Mapping[str, Any], *,
               tok, cap: int, new_cap: int) -> Dict[str, int]:
    """Append the bundle's carried texts to ``spool`` after binding the block
    to the live spec and re-checking every carried text against the SERVED
    tokenizer. Idempotent (an identical replay is skipped; a differing text
    for a seeded id is GenSpool's own refusal). Returns the counts."""
    if block.get("task") != spec.task:
        raise CarryOverError(
            f"bundle block task {block.get('task')!r} != spec task {spec.task!r}")
    expected = [r["item_id"] for r in spec.rows]
    if block.get("items_sha") != rows_content_sha(spec.rows):
        raise CarryOverError(
            f"{spec.task}: bundle items_sha {block.get('items_sha')!r} != the live "
            f"row set's {rows_content_sha(spec.rows)!r} — a different item set "
            "(limit / corpus pin / order); refusing to seed")
    if (block.get("system_sha") or None) != _system_sha(spec):
        raise CarryOverError(
            f"{spec.task}: bundle system_sha {block.get('system_sha')!r} != the "
            f"live spec's {_system_sha(spec)!r} — the SI served differs")
    if int(spec.gen.get("max_new_tokens", -1)) != int(new_cap):
        raise CarryOverError(
            f"{spec.task}: the live gen cap is {spec.gen.get('max_new_tokens')!r}, "
            f"not the bundle's new_cap {new_cap} — pass --max-new-tokens {new_cap}")
    src_cap = int((block.get("source_gen_cfg") or {}).get("max_new_tokens", -1))
    if src_cap != int(cap):
        raise CarryOverError(
            f"{spec.task}: the source spool was generated at max_new_tokens "
            f"{src_cap}, not the bundle's cap {cap}")
    # THE ONE CHANGED VARIABLE IS THE CAP: both sides greedy (do_sample False),
    # or the carried and the regenerated texts would come from two decoding
    # regimes and the cap effect would be confounded with a sampling change
    if bool((block.get("source_gen_cfg") or {}).get("do_sample", False)):
        raise CarryOverError(
            f"{spec.task}: the source spool was SAMPLED (do_sample true) — the "
            "carry-over rule (a terminated greedy response is unchanged by a "
            "higher cap) holds for greedy decoding only; refusing")
    if bool(spec.gen.get("do_sample", False)):
        raise CarryOverError(
            f"{spec.task}: the live gen config samples (do_sample true) — the rerun "
            "must be greedy like the source, with the cap the only change; refusing")
    from . import cap_ladder as cl

    carried = dict(block.get("carried") or {})
    regen = list(block.get("regenerate") or [])
    nonterm = dict(block.get("non_terminating") or {})
    residual = dict(block.get("residual") or {})
    source_caps = {k: int(v) for k, v in (block.get("source_caps") or {}).items()}
    rung = dict(block.get("rung") or {})
    parts = [set(carried), set(regen), set(nonterm), set(residual)]
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            if a & b:
                raise CarryOverError(
                    f"{spec.task}: the bundle's partition overlaps "
                    f"(first {sorted(a & b)[:3]}) — carried / regenerate / "
                    "non_terminating / residual must be disjoint")
    union = set().union(*parts)
    if union != set(expected):
        missing = sorted(set(expected) - union)
        extra = sorted(union - set(expected))
        raise CarryOverError(
            f"{spec.task}: carried ∪ regenerate ∪ non_terminating ∪ residual != the "
            f"live item set — {len(missing)} missing (first {missing[:3]}), "
            f"{len(extra)} unexpected (first {extra[:3]})")
    bad = [iid for iid, text in carried.items()
           if is_at_cap(tok, text, source_caps.get(iid, cap))]
    if bad:
        raise CarryOverError(
            f"{spec.task}: {len(bad)} carried text(s) re-tokenize AT the cap "
            f"{cap} under the SERVED tokenizer (first {bad[:3]}) — the plan's "
            "classification disagrees with the served tokenizer; refusing to "
            "carry a possibly-truncated text")
    # the two EXPLICIT at-cap carries: a non-terminating text must be
    # degenerate under the rule (tokenizer-free, so re-checked here), and a
    # residual is only carried when the rung did NOT climb
    not_loop = [iid for iid, text in nonterm.items() if not cl.is_degenerate(text)]
    if not_loop:
        raise CarryOverError(
            f"{spec.task}: {len(not_loop)} 'non_terminating' text(s) are NOT degenerate "
            f"under {cl.DEGENERATE_RULE!r} (first {not_loop[:3]}) — the plan's loop "
            "classification does not reproduce; refusing to carry them as loops")
    if residual and (rung.get("escalate") or regen):
        raise CarryOverError(
            f"{spec.task}: the block carries {len(residual)} residual at-cap text(s) "
            "but also climbs (rung.escalate / a regenerate list) — a residual is "
            "carried only when the readout does not climb")
    if rung and rung.get("escalate") and not regen and not block.get("forced"):
        raise CarryOverError(
            f"{spec.task}: the rung verdict is escalate but the block regenerates "
            "nothing — an inconsistent plan")
    existing = spool.entries()
    n_added = n_already = 0
    to_seed = {**carried, **nonterm, **residual}
    for iid in sorted(to_seed):
        text = to_seed[iid]
        if iid in existing:
            if existing[iid] != text:
                raise CarryOverError(
                    f"{spec.task}: spool already holds item {iid!r} with a "
                    "DIFFERENT text than the bundle carries — two writers")
            n_already += 1
            continue
        spool.append(iid, text)
        n_added += 1
    return {"n_carried": len(carried), "n_seeded_now": n_added,
            "n_already_present": n_already, "n_regenerate": len(regen),
            "n_non_terminating": len(nonterm), "n_residual": len(residual)}


# --------------------------------------------------------------------------- #
# the parity control's spool                                                   #
# --------------------------------------------------------------------------- #


def parity_spool_path(store_dir: Path, task: str) -> Path:
    name = task.split("/", 1)[1]
    return Path(store_dir) / "private" / f"parity_gens_{name}.jsonl"


def parity_header(base_header: Mapping[str, Any]) -> Dict[str, Any]:
    """The parity spool's header: the cell's judged header plus the role, so
    it can never be mistaken for (or resumed onto) the cell's own spool."""
    return {**dict(base_header), "role": PARITY_ROLE}


def parity_record(*, task: str, ids: Sequence[str], carried: Mapping[str, str],
                  regenerated: Mapping[str, str], finish: Mapping[str, int],
                  spool_path: Path) -> Dict[str, Any]:
    """What the parity leg measured on the SERVED side (text identity vs the
    carried HF texts); the label agreement is the grading host judge step's."""
    identical = [i for i in ids if i in regenerated and regenerated[i] == carried.get(i)]
    return {"task": task, "role": PARITY_ROLE, "n": len(ids), "ids": list(ids),
            "n_regenerated": sum(1 for i in ids if i in regenerated),
            "n_text_identical_to_carried": len(identical),
            "finish_reasons": dict(finish), "spool": str(spool_path)}
