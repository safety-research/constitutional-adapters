#!/usr/bin/env python
"""Per-SI no-op floors — KL(teacher || student) on completion positions,
NO object applied — sampled STRATIFIED BY FAMILY and reported per family.

Provenance, because a floor without its population is how this went wrong
the first time: the original version of this script took ``pool[:n]``, and
the pools' authoritative order is sorted (family, goal_id, variant_key)
(matched_pool's ``order()``), so its "256 paired prompts" (job 187947)
were 235 aim + 21 amplegcg_plus — one family that is 1% of the pool —
while wildjailbreak_benign/harmful (78% of the pool) contributed nothing.
The floors it printed (0.10516 / 0.07129 / 0.06898) are "KL on aim-family
prompts". It also accepted a ``--seed`` it never used (deleted: selection
is the repo's SEEDLESS sha-sort convention now) and validated nothing
before paying the 8B model load.

What this version does instead (review findings 1, 2, 4 + the stratified
amendment):

1. SELECTION — within-family sha256(prompt) rank (``extraction/
   subsample.stratified_sha_rank_indices``), ``--per-family`` rows from
   EVERY family (a thin family contributes all it has). Because the
   per-SI pools are matched on (family, goal_id, variant_key) and share
   the SAME prompt for the same key, the selection picks the SAME prompt
   identities for all three SIs — asserted, not assumed — so the
   cross-SI comparison is paired, not merely equal-sized. The selected
   keys are recorded in the output.

2. ESTIMATOR — the PER-EXAMPLE token-mean KL (``soft_divergence_per_token``
   grouped by each row's completion-token count), the exact estimator the
   guard probes now use, so floors and probe numbers stay directly
   comparable. Batching is memory/wall-clock only; a mean of per-batch
   batchmeans over ragged token counts (the old estimator) moved with
   ``batch_size``.

3. VALIDATE BEFORE THE LOAD — args, config, pool files, required fields,
   the stratified selection and the cross-SI pairing are all checked
   before the model is touched; an SI that yields no finite KL values
   refuses loudly instead of dying inside ``statistics.mean``.

Reported per SI — three numbers that are NOT interchangeable, each
labelled with its population:

- per-family mean KL with 95% CI and n (a thin family reports its point
  estimate WITH its n and no interval — visible thinness, never folded
  into the pool);
- POOL-WEIGHTED pooled floor: family means weighted by their true pool
  share. The pool-representative number a fit criterion should divide by
  — the replacement for the contaminated prefix floors;
- UNWEIGHTED-over-families pooled floor: the mean of family means, what
  an equal-family design measures.

Also persisted per SI (added after job 30542, which kept only the
aggregates above and so could not support a paired per-example test):
``per_example`` — every selected identity (family, goal_id, variant_key,
prompt_sha8) with ITS per-example KL, in selection order. The order is
identical across SIs (asserted by ``validate_and_select``), so two SIs'
lists join row-by-row for a paired comparison with no re-derivation.
``--sis`` measures a subset of the config's si_ids; selection is per-SI
deterministic, so a subset run selects exactly what the full run selects.

FIXED-COMMON-COMPLETION MODE (``--fixed-completion neg --instructions …``)
— the cross-instruction comparison the own-pos floors above CANNOT carry:
each SI's ``pos`` was selected by that SI's own rejection-sampling
acceptance, so an own-pos floor scores every SI on a completion its own
behaviour chose — confounded by construction, worst exactly where the SIs
differ in what they produce, which is the case the comparison exists to
detect. This mode scores every named instruction (any
``resolve_instruction`` id — the config si_ids AND registry references
like format instructions) teacher-forced on the SAME text per row: the
UNDEFENDED completion ``neg`` (generated with no system instruction;
byte-identical across the SI pools — asserted per selected row at load,
never assumed from the pool manifest). The completion token span is
additionally clamped to a budget COMMON across all scored instructions
per row (``plan_common_completion``), so a long-instruction teacher can
never be measured on a shorter prefix of the fixed text than a
short-instruction teacher. What a number here means is pinned in
``READING`` (persisted + printed): how far the instruction moves the
model off the undefended completion, NOT how good what it moves to is.
Each instruction also records its token length and its 2x2 axis cell
(long/short x safety/irrelevant), because the own-pos floors were
monotonic in instruction length and the length-vs-content question is
what the reference instructions exist to answer. Output is written after
EVERY instruction (``partial``/``pending`` flags), so a walltime kill
loses at most the in-flight instruction. ``--verify-selection`` pins the
selection to a prior run's recorded identities, refusing before the load
on any divergence.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sisteer.extraction.subsample import (  # noqa: E402
    sha_hex,
    stratified_sha_rank_indices,
)

#: The prefix-sampled floors this measurement replaces (job 187947 —
#: 235 aim + 21 amplegcg_plus per SI, NOT pool-representative).
CONTAMINATED_FLOORS = {"safety_refusal": 0.10516,
                       "si_constitution": 0.07129,
                       "si_irrelevant": 0.06898}

#: What a fixed-common-completion number MEANS — persisted and printed with
#: every table so the reading can never detach from the numbers.
READING = (
    "This measures HOW FAR an instruction moves the model off the "
    "undefended completion (per-token KL(teacher||student), teacher-forced "
    "on a COMMON fixed text), NOT how good what it moves to is. An "
    "instruction producing incoherent output would score high. Divergence "
    "is necessary, not sufficient, for a defence.")

#: Short, content-free-for-safety format instructions that calibrate the
#: common-text measure: if a format instruction out-scores a safety SI,
#: common-text divergence is demonstrably content-blind at the top of its
#: range and every reading is relative to that ceiling.
FORMAT_REFERENCES = ("json", "bullets")

#: 2x2 length-axis split. The scored instructions sit at 10-88 tokens
#: (short) vs ~5,900-6,400 (long); 1000 splits the 67x gap, and the raw
#: token count is recorded per instruction so the derived label is always
#: recomputable.
AXIS_LENGTH_THRESHOLD_TOKENS = 1000


def pool_key(row: dict) -> tuple:
    """The matched-pool identity — shared across SIs for the same prompt."""
    return (row.get("family", "?"), row.get("goal_id", ""),
            row.get("variant_key", ""))


def load_pool(path: Path) -> list[dict]:
    """Read one pairs_<si>.jsonl, failing loudly on anything malformed —
    all of it BEFORE any model load."""
    if not path.exists():
        raise SystemExit(f"missing pool file: {path}")
    rows = []
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{ln}: not valid JSON ({e})")
        for field in ("prompt", "pos"):
            if not row.get(field):
                raise SystemExit(
                    f"{path}:{ln}: row lacks required field {field!r}")
        rows.append(row)
    if not rows:
        raise SystemExit(f"empty pool file: {path}")
    return rows


def select_stratified(rows: list[dict], per_family: int) -> list[int]:
    """Within-family sha256(prompt)-rank indices, original order — the
    repo's seedless sha-sort convention, stratified."""
    return stratified_sha_rank_indices(
        [r["prompt"] for r in rows],
        [r.get("family", "?") for r in rows], per_family)


def validate_and_select(pools: dict[str, list[dict]],
                        per_family: int) -> dict[str, list[int]]:
    """Stratified selection per SI + the PAIRING check the cross-SI read
    rests on: identical (family, goal_id, variant_key) keys AND identical
    prompts across every SI, in the same order. SystemExit on violation."""
    sel = {si: select_stratified(rows, per_family)
           for si, rows in pools.items()}
    sis = list(pools)
    ref = sis[0]
    ref_keys = [pool_key(pools[ref][i]) for i in sel[ref]]
    ref_prompts = [pools[ref][i]["prompt"] for i in sel[ref]]
    for si in sis[1:]:
        keys = [pool_key(pools[si][i]) for i in sel[si]]
        if keys != ref_keys:
            raise SystemExit(
                f"selection is NOT paired across SIs: {si} selected "
                f"different (family, goal_id, variant_key) identities than "
                f"{ref} — the pools are not matched; a cross-SI comparison "
                "on these samples would not be paired. Refusing.")
        prompts = [pools[si][i]["prompt"] for i in sel[si]]
        if prompts != ref_prompts:
            raise SystemExit(
                f"selection is NOT paired across SIs: {si} and {ref} share "
                "keys but carry DIFFERENT prompt texts for the same key — "
                "pool corruption. Refusing.")
    return sel


def per_example_records(selected: list[dict],
                        values: list[float | None]) -> list[dict]:
    """One record per SELECTED identity, in selection order: the identity
    key (family, goal_id, variant_key, prompt_sha8) plus that example's
    per-example KL. Selection order is identical across SIs (asserted by
    ``validate_and_select``), so two SIs' lists join row-by-row — the
    paired per-example comparison job 30542 couldn't support needs no
    re-derivation. An example that yielded no finite value carries
    ``kl: null`` at its position, never a silent omission: the length
    ALWAYS equals the selection's."""
    if len(selected) != len(values):
        raise ValueError(
            f"per_example_records: {len(selected)} selected identities but "
            f"{len(values)} values — identity/value alignment lost")
    return [dict(rec, kl=v) for rec, v in zip(selected, values)]


def validate_neg_common(pools: dict[str, list[dict]],
                        sel: dict[str, list[int]]) -> None:
    """The fixed-completion premise, asserted at the rows actually measured
    (never assumed from the pool manifest): the undefended completion
    ``neg`` is non-empty and byte-identical across every SI's pool for the
    same selected identity. SystemExit on violation — a fixed text that
    differs across SIs would re-import the very selection confound this
    mode removes."""
    sis = list(pools)
    ref = sis[0]
    ref_negs = [pools[ref][i].get("neg") for i in sel[ref]]
    for k, n in enumerate(ref_negs):
        if not n:
            raise SystemExit(
                f"fixed completion: selected row {k} of {ref} has an "
                "empty/missing neg — no undefended completion to measure "
                "on. Refusing.")
    for si in sis[1:]:
        negs = [pools[si][i].get("neg") for i in sel[si]]
        if negs != ref_negs:
            bad = next(k for k, (a, b) in enumerate(zip(ref_negs, negs))
                       if a != b)
            raise SystemExit(
                f"fixed completion is NOT common: {si} and {ref} carry "
                f"DIFFERENT neg text for the same identity (first at "
                f"selected row {bad}) — the pools' "
                "identical-except-pos invariant does not hold on these "
                "files, so no completion is selection-free. Refusing.")


def verify_selection_against(selected_records: list[dict],
                             prior_path: str) -> None:
    """Pin this run's selection to a PRIOR run's recorded identities
    (e.g. the established 1037 of job 30542) — refuse, before the model
    load, on any divergence, naming the first differing row."""
    p = Path(prior_path)
    if not p.exists():
        raise SystemExit(f"--verify-selection: missing file {p}")
    prior = json.loads(p.read_text())
    prior_sel = (prior.get("selection") or {}).get("selected")
    if not prior_sel:
        raise SystemExit(
            f"--verify-selection: {p} carries no selection.selected to "
            "verify against")
    if prior_sel != selected_records:
        for k, (a, b) in enumerate(zip(prior_sel, selected_records)):
            if a != b:
                raise SystemExit(
                    f"selection DIVERGES from {p} at row {k}: prior {a} "
                    f"vs this run {b} (prior n={len(prior_sel)}, this "
                    f"run n={len(selected_records)}). Refusing.")
        raise SystemExit(
            f"selection DIVERGES from {p}: prior n={len(prior_sel)}, "
            f"this run n={len(selected_records)} (one is a prefix of "
            "the other). Refusing.")
    print(f"[select] selection verified IDENTICAL to {p} "
          f"({len(prior_sel)} identities, prompt hashes included)")


def axis_cell(category: str, n_tokens: int) -> dict:
    """The 2x2 cell an instruction occupies: length long/short (measured
    tokens vs AXIS_LENGTH_THRESHOLD_TOKENS) x content safety/irrelevant
    (the registry category — anything not a safety instruction is
    task-irrelevant for the safety axis: formats, controls, personas)."""
    return {
        "length": ("long" if n_tokens >= AXIS_LENGTH_THRESHOLD_TOKENS
                   else "short"),
        "content": "safety" if category == "safety" else "irrelevant",
    }


def plan_common_completion(tok, rows: list[dict], instr_texts: list[str],
                           max_len: int, max_excluded_frac: float = 0.05,
                           ) -> tuple[list[list[int] | None], list[int],
                                      int]:
    """Per-row completion token ids clamped to the budget COMMON across
    ALL scored instructions.

    ``_encode_aligned``'s per-call budget (max_len minus THAT call's
    prompt) would, wherever it binds, measure a long-instruction teacher
    on a SHORTER prefix of the fixed text than a short-instruction teacher
    — un-fixing the "same fixed text" premise exactly on the longest
    prompts. So the budget here is max_len minus the LONGEST prompt over
    every scored instruction's teacher render and the student render, per
    row; every instruction then measures the identical token span.

    Returns (comp_ids per row — None where excluded, excluded row indices,
    n rows whose completion the common budget truncated). A row whose
    budget admits no completion token is excluded from ALL instructions
    (pairing preserved); more than ``max_excluded_frac`` of rows excluded
    refuses outright — that is a --max-len problem, not a data quirk."""
    from sisteer.models import build_chat_prompt

    eos = tok.eos_token_id
    comp_full: list[list[int]] = []
    for r in rows:
        c = tok(r["neg"], add_special_tokens=False)["input_ids"]
        if eos is not None:
            c = c + [eos]
        comp_full.append(c)
    max_prompt = [len(tok(build_chat_prompt(tok, r["prompt"], system=None),
                          add_special_tokens=True)["input_ids"])
                  for r in rows]
    for text in instr_texts:
        for k, r in enumerate(rows):
            t_len = len(tok(build_chat_prompt(tok, r["prompt"], system=text),
                            add_special_tokens=True)["input_ids"])
            if t_len > max_prompt[k]:
                max_prompt[k] = t_len

    comp: list[list[int] | None] = []
    excluded: list[int] = []
    n_trunc = 0
    for k, (c, mp) in enumerate(zip(comp_full, max_prompt)):
        avail = max_len - mp
        if avail < 1:
            excluded.append(k)
            comp.append(None)
            continue
        if len(c) > avail:
            n_trunc += 1
        comp.append(c[:avail])
    if len(excluded) > max_excluded_frac * len(rows):
        raise SystemExit(
            f"common completion budget excludes {len(excluded)}/{len(rows)} "
            f"rows (> {max_excluded_frac:.0%}) at --max-len {max_len} — "
            "the longest scored instruction leaves no room for the fixed "
            "text on these prompts. Raise --max-len rather than measuring "
            "a sample this budget reshaped.")
    return comp, excluded, n_trunc


def ratios_to_format_references(results: dict[str, dict],
                                refs: tuple[str, ...] = FORMAT_REFERENCES,
                                ) -> dict:
    """Each instruction's pooled divergence as a RATIO to the mean of the
    short format references — the calibration the absolute numbers lack:
    a safety SI at ratio ~1 moves the model no further off the undefended
    text than 'answer in JSON' does. One block per pooled weighting, each
    named; empty (with a note) until every reference is measured."""
    out: dict = {}
    for w in ("pool_weighted", "unweighted_over_families"):
        ref_means = {r: results[r][w]["mean_kl"] for r in refs
                     if r in results}
        if len(ref_means) != len(refs):
            missing = [r for r in refs if r not in results]
            out[w] = {"note": f"unavailable: reference(s) {missing} not "
                              "measured (yet)"}
            continue
        ref_mean = sum(ref_means.values()) / len(ref_means)
        out[w] = {
            "reference_mean": ref_mean,
            "reference_means": ref_means,
            "ratios": {nm: results[nm][w]["mean_kl"] / ref_mean
                       for nm in results},
        }
    return out


def _ci95(mean: float, sem: float | None) -> tuple[float | None,
                                                   float | None]:
    if sem is None:
        return None, None
    return mean - 1.96 * sem, mean + 1.96 * sem


def summarize(per_example: list[float], fams: list[str],
              pool_family_counts: dict[str, int]) -> dict:
    """Per-family table + the two pooled floors, from per-example values.

    Per-family is the sufficient statistic here — both pooled numbers are
    computed FROM the family means, and pooling is irreversible, so the
    table ships alongside them. A family with n < 2 reports its point
    estimate and n with null CI bounds; it is never collapsed into the
    pooled numbers' uncertainty silently (the pooled CIs go null too, with
    the thin families named)."""
    from sisteer.extraction.distill import _mean_sd_sem

    if len(per_example) != len(fams):
        raise ValueError(f"{len(per_example)} values, {len(fams)} labels")
    if not per_example:
        raise ValueError("summarize needs at least one per-example value")
    total_pool = sum(pool_family_counts.values())
    by_fam: dict[str, list[float]] = collections.defaultdict(list)
    for f, v in zip(fams, per_example):
        by_fam[f].append(v)

    per_family = {}
    for f in sorted(by_fam):
        m, sd, sem = _mean_sd_sem(by_fam[f])
        lo, hi = _ci95(m, sem)
        per_family[f] = {
            "mean_kl": m, "sd": sd, "sem": sem,
            "ci95_lo": lo, "ci95_hi": hi, "n": len(by_fam[f]),
            "pool_share": pool_family_counts.get(f, 0) / total_pool,
        }

    fams_sorted = sorted(per_family)
    thin = [f for f in fams_sorted if per_family[f]["sem"] is None]
    means = {f: per_family[f]["mean_kl"] for f in fams_sorted}
    sems = {f: per_family[f]["sem"] for f in fams_sorted}

    # pool-weighted: weights are the TRUE pool shares of the families
    # present, renormalised over them. CI by the delta method over
    # independent family means; null (with the thin families named) when
    # any weight sits on a family whose spread is unresolvable.
    w = {f: per_family[f]["pool_share"] for f in fams_sorted}
    w_tot = sum(w.values())
    pw_mean = sum(w[f] * means[f] for f in fams_sorted) / w_tot
    pw_sem = (math.sqrt(sum((w[f] / w_tot) ** 2 * sems[f] ** 2
                            for f in fams_sorted))
              if not thin else None)
    pw_lo, pw_hi = _ci95(pw_mean, pw_sem)

    # unweighted over families: every family counts once.
    uf_mean = sum(means.values()) / len(fams_sorted)
    uf_sem = (math.sqrt(sum(s ** 2 for s in sems.values()))
              / len(fams_sorted) if not thin else None)
    uf_lo, uf_hi = _ci95(uf_mean, uf_sem)

    return {
        "per_family": per_family,
        "pool_weighted": {
            "mean_kl": pw_mean, "sem": pw_sem,
            "ci95_lo": pw_lo, "ci95_hi": pw_hi,
            "weights": "true pool share of each family present, "
                       "renormalised — the pool-representative floor",
            "families_without_intervals": thin,
        },
        "unweighted_over_families": {
            "mean_kl": uf_mean, "sem": uf_sem,
            "ci95_lo": uf_lo, "ci95_hi": uf_hi,
            "weights": "each family counts once — the equal-family-design "
                       "floor, NOT pool-representative",
            "families_without_intervals": thin,
        },
        "families_covered": len(fams_sorted),
        "n_examples": len(per_example),
    }


def measure_batches(model, tok, n: int, encode, sel_fams: list[str],
                    batch_size: int, label: str
                    ) -> tuple[list[float], list[str],
                               list[float | None], int]:
    """The shared no-grad measurement loop — one code path for the own-pos
    floors and the fixed-common-completion mode, so the two can never
    drift onto different estimators. ``encode(j)`` returns the (teacher,
    student) EncodedSeq pair for example ``j``. Returns (finite
    per-example KLs, their family labels, per-IDENTITY values with None
    where nothing finite was measured, n_dropped). Batching is memory
    only: the estimate is per-example (soft_divergence_per_token /
    per_example_means) and does not move with batch_size."""
    import torch

    from sisteer.extraction.distill import (_gather_completion_logits,
                                            per_example_means,
                                            soft_divergence_per_token)
    from sisteer.extraction.traindata import collate

    device = next(model.parameters()).device
    per_example: list[float] = []
    ex_fams: list[str] = []
    vals_by_identity: list[float | None] = []
    n_dropped = 0
    bs = batch_size
    for i in range(0, n, bs):
        js = list(range(i, min(i + bs, n)))
        cfams = sel_fams[i:i + bs]
        enc = [encode(j) for j in js]
        tb = collate([a for a, _ in enc], tok.pad_token_id).to(device)
        sb = collate([b for _, b in enc], tok.pad_token_id).to(device)
        with torch.no_grad():
            t_logits = _gather_completion_logits(
                model(input_ids=tb.input_ids,
                      attention_mask=tb.attention_mask).logits, tb)
            s_logits = _gather_completion_logits(
                model(input_ids=sb.input_ids,
                      attention_mask=sb.attention_mask).logits, sb)
            t_logp = torch.log_softmax(t_logits.float(), -1)
            s_logp = torch.log_softmax(s_logits.float(), -1)
            per_tok = soft_divergence_per_token("soft", t_logp, s_logp)
            counts = (sb.labels[:, 1:] != -100).sum(1)
            vals = per_example_means(per_tok, counts)
        # per_example_means SKIPS zero-count rows, so re-expand its
        # output to chunk positions before pairing values with
        # identities — zipping vals directly against the chunk would
        # silently shear the alignment after any skipped row. A
        # zero-count row records None (nothing was measured), exactly
        # like a non-finite value.
        counts_l = [int(c) for c in counts.tolist()]
        row_vals: list[float | None] = [None] * len(js)
        vals_it = iter(vals)
        for j, k in enumerate(counts_l):
            if k > 0:
                row_vals[j] = next(vals_it)
        for fam, v in zip(cfams, row_vals):
            if v is not None and math.isfinite(v):
                per_example.append(v)
                ex_fams.append(fam)
                vals_by_identity.append(v)
            else:
                n_dropped += 1
                vals_by_identity.append(None)
                print(f"[warn] {label}: dropped a per-example KL with no "
                      f"finite value (family {fam})", flush=True)
        if per_example and (i // bs) % 40 == 0:
            print(f"[{label}] {len(per_example)}/{n} "
                  f"running_mean="
                  f"{sum(per_example) / len(per_example):.5f}",
                  flush=True)
    return per_example, ex_fams, vals_by_identity, n_dropped


def run_fixed_completion(args, cfg, cfg_path, pools, ref_sel, ref,
                         sel_fams, fam_counts, selected_records,
                         instrs: dict) -> None:
    """Fixed-COMMON-completion divergence: KL(teacher || student) teacher-
    forced on the UNDEFENDED completion (``neg``) — the SAME text for
    every scored instruction. See the module docstring for why the own-pos
    floors cannot carry a cross-SI comparison; ``validate_neg_common`` has
    already asserted the premise by the time this runs. Output is written
    after EVERY instruction so a walltime kill loses at most the one in
    flight."""
    rows = [pools[ref][i] for i in ref_sel]
    names = list(instrs)

    from sisteer.extraction.distill import EncodedSeq
    from sisteer.models import build_chat_prompt, load_model_and_tokenizer

    # EXACTLY the driver's call — see the own-pos path.
    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model.eval()

    instr_tokens = {nm: len(tok(ins.text, add_special_tokens=False)
                            ["input_ids"]) for nm, ins in instrs.items()}
    for nm in names:
        cell = axis_cell(instrs[nm].category, instr_tokens[nm])
        print(f"[axis] {nm:<18} {instr_tokens[nm]:>6} tok  "
              f"{cell['length']}+{cell['content']}", flush=True)

    comp, excluded, n_trunc = plan_common_completion(
        tok, rows, [instrs[nm].text for nm in names], args.max_len)
    excl_set = set(excluded)
    kept = [k for k in range(len(rows)) if k not in excl_set]
    kept_fams = [sel_fams[k] for k in kept]
    if excluded:
        print(f"[warn] {len(excluded)} rows excluded from ALL scored "
              f"instructions (prompt leaves no room for the fixed text at "
              f"--max-len {args.max_len}); pairing preserved", flush=True)
    if n_trunc:
        print(f"[note] the common budget truncates the fixed text on "
              f"{n_trunc}/{len(kept)} kept rows — the SAME span is "
              "measured for every instruction there", flush=True)

    pool_counts = dict(collections.Counter(
        r.get("family", "?") for r in pools[ref]))
    results: dict[str, dict] = {}

    def _save(pending: list[str]) -> None:
        if not args.out:
            return
        payload = {
            "mode": "fixed_completion_common_text",
            "fixed_completion": {
                "field": "neg",
                "definition": "the UNDEFENDED completion: the pool's neg, "
                              "generated by the base model with NO system "
                              "instruction (neg_mode='none'), byte-"
                              "identical across every SI pool for the "
                              "same identity — asserted on all selected "
                              "rows at load, never assumed from the pool "
                              "manifest",
            },
            "reading": READING,
            "results": results,
            "ratios_to_format_references":
                ratios_to_format_references(results),
            "axis_length_threshold_tokens": AXIS_LENGTH_THRESHOLD_TOKENS,
            "instructions_scored": names,
            "pending": pending,
            "partial": bool(pending),
            "selection": {
                "method": "sha256(prompt) hexdigest rank within each "
                          "family stratum, seedless (the repo sha-sort "
                          "convention); identical prompt identities "
                          "across SIs, asserted",
                "per_family_target": args.per_family,
                "families": fam_counts,
                "selected": selected_records,
                "verified_against": args.verify_selection,
            },
            "excluded_rows": [
                dict(selected_records[k],
                     reason="prompt leaves no room for the fixed text "
                            "under the common completion budget")
                for k in excluded],
            "n_truncated_common_budget": n_trunc,
            "estimator": "per-example token-mean forward KL(teacher||"
                         "student) on completion positions "
                         "(soft_divergence_per_token / per_example_means "
                         "— chunk-invariant; batch_size is memory only); "
                         "completion span clamped to the budget COMMON "
                         "across all scored instructions per row, so "
                         "every instruction measures the identical fixed-"
                         "text tokens",
            "model": cfg["model"],
            "config": str(cfg_path),
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        state = (f"partial, pending: {', '.join(pending)}" if pending
                 else "final")
        print(f"[save] wrote {args.out} ({state})", flush=True)

    for idx, nm in enumerate(names):
        ins = instrs[nm]
        t0 = time.time()
        t_prompts = [build_chat_prompt(tok, rows[k]["prompt"],
                                       system=ins.text) for k in kept]
        s_prompts = [build_chat_prompt(tok, rows[k]["prompt"], system=None)
                     for k in kept]
        comps = [comp[k] for k in kept]

        def encode(j: int) -> tuple:
            t_ids = tok(t_prompts[j], add_special_tokens=True)["input_ids"]
            s_ids = tok(s_prompts[j], add_special_tokens=True)["input_ids"]
            if max(len(t_ids), len(s_ids)) + len(comps[j]) > args.max_len:
                raise ValueError(
                    "common completion budget violated at measure time — "
                    "plan_common_completion and encode disagree on a "
                    "prompt length")
            return (EncodedSeq(ids=t_ids + comps[j], prompt_len=len(t_ids)),
                    EncodedSeq(ids=s_ids + comps[j], prompt_len=len(s_ids)))

        per_example, ex_fams, kept_vals, n_dropped = measure_batches(
            model, tok, len(kept), encode, kept_fams, args.batch_size, nm)
        if not per_example:
            raise SystemExit(
                f"{nm}: NO finite per-example KL values survived "
                f"({n_dropped} dropped) — refusing to report a divergence "
                "from nothing.")

        # Re-expand to the FULL selection: excluded rows carry kl: null at
        # their identity's position, identically for every instruction.
        full_vals: list[float | None] = [None] * len(rows)
        for pos, k in enumerate(kept):
            full_vals[k] = kept_vals[pos]

        results[nm] = summarize(per_example, ex_fams, pool_counts)
        results[nm]["per_example"] = per_example_records(
            selected_records, full_vals)
        results[nm].update({
            "instruction_tokens": instr_tokens[nm],
            "category": ins.category,
            "axis": axis_cell(ins.category, instr_tokens[nm]),
            "n_dropped_nonfinite": n_dropped,
            "n_excluded_common_budget": len(excluded),
            "batch_size_memory_only": args.batch_size,
            "max_len": args.max_len,
            "secs": round(time.time() - t0, 1),
        })
        pw = results[nm]["pool_weighted"]
        uf = results[nm]["unweighted_over_families"]
        cell = results[nm]["axis"]
        print(f"=== {nm} ({instr_tokens[nm]} tok, {cell['length']}+"
              f"{cell['content']}): pool-weighted {pw['mean_kl']:.5f} "
              f"[{pw['ci95_lo']:.5f}, {pw['ci95_hi']:.5f}] | "
              f"unweighted-over-families {uf['mean_kl']:.5f} | "
              f"n={results[nm]['n_examples']} over "
              f"{results[nm]['families_covered']} families ===", flush=True)
        _save(names[idx + 1:])

    print("\n===== COMMON-TEXT DIVERGENCE (fixed text: the UNDEFENDED "
          "completion, neg) =====")
    print(READING)
    print(f"\n{'instruction':<18}{'tok':>7}{'cell':>18}"
          f"{'pool-wtd KL':>13}{'unwtd KL':>11}{'n':>6}")
    by_cell = sorted(names, key=lambda nm: (
        results[nm]["axis"]["length"], results[nm]["axis"]["content"], nm))
    for nm in by_cell:
        r = results[nm]
        cell = f"{r['axis']['length']}+{r['axis']['content']}"
        print(f"{nm:<18}{r['instruction_tokens']:>7}{cell:>18}"
              f"{r['pool_weighted']['mean_kl']:>13.5f}"
              f"{r['unweighted_over_families']['mean_kl']:>11.5f}"
              f"{r['n_examples']:>6}")
    ratios = ratios_to_format_references(results)
    pwr = ratios.get("pool_weighted", {})
    if "ratios" in pwr:
        print(f"\nratios to the format-reference mean "
              f"({' + '.join(FORMAT_REFERENCES)}; pool-weighted):")
        for nm in by_cell:
            print(f"  {nm:<18}{pwr['ratios'][nm]:>8.2f}x")
    _save([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/two_si.yaml")
    ap.add_argument("--pairs-dir", required=True,
                    help="dir holding pairs_<si>.jsonl")
    ap.add_argument("--per-family", type=int, default=96,
                    help="rows per family stratum, within-family "
                         "sha256(prompt) rank (thin families contribute "
                         "everything they have)")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="memory width of one no-grad forward — the "
                         "estimate is per-example and does not move with "
                         "this")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--sis", default=None,
                    help="comma-separated subset of the config's si_ids to "
                         "measure (default: all of them). Selection is "
                         "per-SI deterministic — within-family sha rank of "
                         "that SI's own pool — so a subset run selects "
                         "exactly the identities the full run selects.")
    ap.add_argument("--fixed-completion", choices=["neg"], default=None,
                    help="score every --instructions instruction on the "
                         "SAME text per row: the UNDEFENDED completion "
                         "(neg — generated with no system instruction, "
                         "byte-identical across the SI pools; asserted at "
                         "load). Removes the own-pos selection confound "
                         "from cross-instruction comparison. All of the "
                         "config's si_ids' pools are loaded for the "
                         "cross-SI assertion regardless of which "
                         "instructions are scored.")
    ap.add_argument("--instructions", default=None,
                    help="fixed-completion mode's axis: comma-separated "
                         "instruction ids, each resolvable by "
                         "resolve_instruction (SAFETY_SI or "
                         "instructions.BY_ID) — deliberately NOT limited "
                         "to the config's si_ids, so short format "
                         "references can calibrate the safety SIs.")
    ap.add_argument("--verify-selection", default=None,
                    help="path to a prior run's output json; refuse "
                         "(before the model load) unless this run's "
                         "selection.selected is IDENTICAL — pins the "
                         "measurement to the established identities, "
                         "prompt hashes included.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # ---- validate EVERYTHING before the model load --------
    if args.per_family < 1:
        raise SystemExit(f"--per-family must be >= 1, got {args.per_family}")
    if args.batch_size < 1:
        raise SystemExit(f"--batch-size must be >= 1, got {args.batch_size}")
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"missing config: {cfg_path}")
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text())
    si_ids = list(cfg.get("si_ids") or [])
    if not si_ids:
        raise SystemExit(f"{cfg_path}: config names no si_ids")

    instrs: dict = {}
    if args.instructions is not None and args.fixed_completion is None:
        raise SystemExit(
            "--instructions requires --fixed-completion: the default mode "
            "scores the config's si_ids on their OWN pos completions "
            "(--sis subsets them)")
    if args.fixed_completion is not None:
        if args.sis is not None:
            raise SystemExit(
                "--sis subsets the own-pos mode; fixed-completion mode "
                "names its axis with --instructions (and always loads "
                "every config SI's pool for the common-neg assertion)")
        if not args.instructions:
            raise SystemExit(
                "--fixed-completion requires --instructions naming what "
                "to score")
        names = [s.strip() for s in args.instructions.split(",")
                 if s.strip()]
        if not names:
            raise SystemExit(
                f"--instructions names no instructions: "
                f"{args.instructions!r}")
        if len(set(names)) != len(names):
            raise SystemExit(
                f"--instructions repeats one: {args.instructions!r}")
        from sisteer.eval.jailbreak import resolve_instruction
        for nm in names:
            try:
                instrs[nm] = resolve_instruction(nm)
            except KeyError:
                raise SystemExit(
                    f"--instructions names an unknown instruction id "
                    f"{nm!r} (not SAFETY_SI, not in instructions.BY_ID)")

    if args.sis is not None:
        wanted = [s.strip() for s in args.sis.split(",") if s.strip()]
        if not wanted:
            raise SystemExit(f"--sis names no SIs: {args.sis!r}")
        unknown = [s for s in wanted if s not in si_ids]
        if unknown:
            raise SystemExit(
                f"--sis names SIs absent from {cfg_path}'s si_ids: "
                f"{unknown} (config has {si_ids})")
        if len(set(wanted)) != len(wanted):
            raise SystemExit(f"--sis repeats an SI: {args.sis!r}")
        # Config order, not flag order: the reference SI (the pairing
        # anchor whose rows the recorded identities come from) must not
        # move with how the flag was spelled.
        si_ids = [s for s in si_ids if s in wanted]

    pools = {si: load_pool(Path(args.pairs_dir) / f"pairs_{si}.jsonl")
             for si in si_ids}
    sel = validate_and_select(pools, args.per_family)
    ref = si_ids[0]
    sel_fams = [pools[ref][i].get("family", "?") for i in sel[ref]]
    fam_counts = collections.Counter(sel_fams)
    print(f"[select] {len(sel[ref])} shared prompt identities per SI over "
          f"{len(fam_counts)} families (per-family target "
          f"{args.per_family}, within-family sha256(prompt) rank, "
          "seedless):")
    for f, n in sorted(fam_counts.items()):
        print(f"[select]   {f:<24} {n}")

    selected_records = [
        {"family": pools[ref][i].get("family", "?"),
         "goal_id": pools[ref][i].get("goal_id", ""),
         "variant_key": pools[ref][i].get("variant_key", ""),
         "prompt_sha8": sha_hex(pools[ref][i]["prompt"])[:8]}
        for i in sel[ref]]

    if args.fixed_completion is not None:
        # The fixed text must be selection-free: assert neg is common
        # across every SI's pool at the selected rows, BEFORE the load.
        validate_neg_common(pools, sel)
    if args.verify_selection:
        verify_selection_against(selected_records, args.verify_selection)

    if args.fixed_completion is not None:
        run_fixed_completion(args, cfg, cfg_path, pools, sel[ref], ref,
                             sel_fams, dict(fam_counts), selected_records,
                             instrs)
        return

    # ---- the model, only now ------------------------------------------
    from sisteer.data.pairs import Pair, pairs_to_distill_examples
    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.extraction.distill import _encode_aligned
    from sisteer.models import load_model_and_tokenizer

    # EXACTLY the driver's call (train_two_si_objects.py:1616) — passing
    # the whole cfg dict made transformers treat it as a repo id (187895).
    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    model.eval()

    results: dict[str, dict] = {}
    for si in si_ids:
        t0 = time.time()
        rows = [pools[si][i] for i in sel[si]]
        # VERBATIM the driver's loader semantics: not load_pairs() (the
        # emit-pairs jsonl carries extra keys the dataclass rejects), not
        # hand-rolled (Pair requires instruction_id — killed 187912).
        pairs = [Pair(prompt=r["prompt"], pos=r["pos"], neg=r.get("neg", ""),
                      instruction_id=r.get("instruction_id", ""),
                      requirement_a=r.get("requirement_a"),
                      anti_neg=r.get("anti_neg")) for r in rows]
        examples = pairs_to_distill_examples(tok, pairs,
                                             resolve_instruction(si))

        per_example, ex_fams, vals_by_identity, n_dropped = measure_batches(
            model, tok, len(examples),
            lambda j: _encode_aligned(tok, examples[j], args.max_len),
            sel_fams, args.batch_size, si)

        if not per_example:
            raise SystemExit(
                f"{si}: NO finite per-example KL values survived "
                f"({n_dropped} dropped) — refusing to report a floor from "
                "nothing.")
        pool_counts = collections.Counter(
            r.get("family", "?") for r in pools[si])
        results[si] = summarize(per_example, ex_fams, dict(pool_counts))
        results[si]["per_example"] = per_example_records(
            selected_records, vals_by_identity)
        results[si].update({
            "n_dropped_nonfinite": n_dropped,
            "batch_size_memory_only": args.batch_size,
            "max_len": args.max_len,
            "secs": round(time.time() - t0, 1),
        })
        pw = results[si]["pool_weighted"]
        uf = results[si]["unweighted_over_families"]
        print(f"=== {si}: pool-weighted {pw['mean_kl']:.5f} "
              f"[{pw['ci95_lo']:.5f}, {pw['ci95_hi']:.5f}] | "
              f"unweighted-over-families {uf['mean_kl']:.5f} | "
              f"n={results[si]['n_examples']} over "
              f"{results[si]['families_covered']} families ===", flush=True)

    print("\n============ PER-SI FLOORS, STRATIFIED BY FAMILY ============")
    for si, r in results.items():
        pw, uf = r["pool_weighted"], r["unweighted_over_families"]
        print(f"\n{si}")
        print(f"  pool-weighted (pool-representative): {pw['mean_kl']:.5f}"
              f"  95% CI [{pw['ci95_lo']:.5f}, {pw['ci95_hi']:.5f}]")
        print(f"  unweighted over families:            {uf['mean_kl']:.5f}"
              f"  95% CI [{uf['ci95_lo']:.5f}, {uf['ci95_hi']:.5f}]")
        print(f"  {'family':<24}{'mean KL':>10}{'95% CI':>24}{'n':>6}"
              f"{'pool%':>8}")
        for f, fr in sorted(r["per_family"].items(),
                            key=lambda kv: -kv[1]["pool_share"]):
            ci = (f"[{fr['ci95_lo']:.5f}, {fr['ci95_hi']:.5f}]"
                  if fr["ci95_lo"] is not None else "(n too thin)")
            print(f"  {f:<24}{fr['mean_kl']:>10.5f}   {ci:>21}"
                  f"{fr['n']:>6}{100 * fr['pool_share']:>7.1f}%")
        old = CONTAMINATED_FLOORS.get(si)
        if old is not None:
            print(f"  replaces prefix-sampled (235 aim + 21 amplegcg_plus) "
                  f"floor {old:.5f}: pool-weighted moved "
                  f"{100 * (pw['mean_kl'] - old) / old:+.1f}%")

    if args.out:
        payload = {
            "results": results,
            "selection": {
                "method": "sha256(prompt) hexdigest rank within each "
                          "family stratum, seedless (the repo sha-sort "
                          "convention); identical prompt identities "
                          "across SIs, asserted",
                "per_family_target": args.per_family,
                "families": dict(fam_counts),
                "selected": selected_records,
                "verified_against": args.verify_selection,
            },
            "estimator": "per-example token-mean forward KL(teacher||"
                         "student) on completion positions "
                         "(soft_divergence_per_token / per_example_means "
                         "— chunk-invariant; batch_size is memory only)",
            "model": cfg["model"],
            "config": str(cfg_path),
            "replaces_contaminated_prefix_floors": CONTAMINATED_FLOORS,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
