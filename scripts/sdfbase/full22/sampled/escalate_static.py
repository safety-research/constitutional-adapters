#!/usr/bin/env python
"""escalate_static.py — the static families' cap-escalation rung of the full22 SAMPLED re-run.

    escalate_static.py --config configs/sdfbase_full22_<key>_sampled_t<T>_cap8192.yaml --attack <fam>
        --defense <cell> --private-root $PRIVATE_ROOT --proxy-url $PROXY_URL --routing-table $TABLE
        [--max-model-len L] [--concurrency N]

The ladder (scripts/sdfbase/full22/sampled/DESIGN.md): a sampled base run at 4,096 leaves some items
at ``finish_reason == "length"``; this rung RE-DRAWS exactly those items at the escalation config's
``attacks.static_max_new_tokens`` (8,192) with the SAME ``sampling_seed`` — a re-draw, not a
continuation — and copies every terminated row forward unchanged, so the rung root holds every
item of the base run: each item's text at its lowest terminating cap, or its 8,192 draw. NEVER in
place: the base run (``sampled_decode.escalated_from.private_run``) is a read-only input, and the
rung is its own ``private_run`` under the same ``--private-root``, as ONE shard dir ``shard00``
holding ``responses.jsonl`` (base order), ``progress.jsonl`` (one terminal record per goal, the
run_static shape), ``served_target.json`` (the served identity + decode block) and
``escalation.json`` (the counts below). The rung is therefore a JOIN against the base by
``(goal_id, sample_idx, trial)``, never a re-match.

Per row: ``regenerated`` (false = copied forward), ``escalated_from_cap`` (the base cap),
``source_cap`` (the rung that produced THIS text: PRESERVED through every copy — a row that
terminated at 4,096 still says 4,096 inside the 16,384 store — and this cap for a re-draw), plus
the sampled row schema (``decode`` with ``cap`` = the cap in force for this text, so
``decode.cap == source_cap`` always; ``finish_reason``, ``at_cap``, ...). A rung store therefore
carries as many decode blocks as rungs its texts were drawn at (DESIGN.md's two-decode-blocks
design): the base store of the 16,384 rung is such a MIXED-CAP store, and that is its ordinary
shape, never a defect. Every REGENERATED row additionally records its relationship to the base
text of the same item:
``prefix_extension`` (the base bytes are a prefix of the new bytes), ``identical``,
``first_divergence_char`` / ``first_divergence_token`` (null under prefix_extension; the token index
is under the target tokenizer, ``sisteer.models.load_tokenizer`` at the config's model /
model_revision, no special tokens), ``source_completion_tokens`` / ``source_finish_reason`` /
``source_draw_cap`` (the base draw's: its finish, its length, the cap it was drawn at — the base
row's verified source cap, which is the base store's cap except for an at-cap row the decision tool
had left at a lower cap), and ``boundary_landing`` (the re-draw's completion_tokens <
``source_draw_cap``: the item was never cap-limited, the base draw landed on the boundary). NO
noise-floor sample is drawn for the static families (the controls carry it).

REFUSES, never degrades: a base run that is incomplete (rows vs progress, per shard and per goal),
a base row without ``finish_reason`` (a greedy store), a base row whose ``decode`` block is not the
config's preset for this trial / seed_base at the row's PROVENANCE-VERIFIED source cap
(``row_source_cap``: a base row's is the base store's cap; a rung row says so itself — ``regenerated``
true = drawn at that store's cap, false = copied from a rung below — and its marks must agree with
each other and with its own ``decode.cap``), a seed that does not reproduce from (seed_base, trial,
family, goal_id, sample_idx), a failed request (nothing is written: the root is staged and renamed
into place only when every row is in hand), an existing PARTIAL rung root — one without its final
``served_target.json``, or whose sidecar records another target decode, or whose rows contradict
their own provenance at this rung's cap, included (the sidecar is the rows' identity and travels
with them in the one rename; the rows are read with the same rule the base store is). Idempotent:
an existing COMPLETE rung root exits 0 without touching the server. Positive controls are ON
(``build_served_target``: salt isolation + steering-bites) before the first re-draw.
config's preset for this trial / seed_base at the cap its own provenance names (below), a seed that
does not reproduce from (seed_base, trial, family, goal_id, sample_idx), a failed request (nothing is
written: the root is staged and renamed into place only when every row is in hand), an existing
PARTIAL rung root.

A PADDED BASE IS REFUSED (docs/static_padding.md): a store drawn under ``--static-padding-tokens``
records the UNPADDED transform in ``row["attack_prompt"]`` and pins what was actually sent only by
``padding.padded_prompt_sha256``. This rung re-draws ``attack_prompt`` verbatim, so a padded base
would be re-drawn WITHOUT its padding — a different measurement wearing the store's name. The guard
(:func:`assert_base_unpadded`) runs on the base rows before the target is built, before any request:
regenerate the padded run at the larger cap into its own ``private_run`` instead.

A RUNG-SHAPED SOURCE (the 8192 -> 16384 climb): the source of a rung above the first is itself a
rung root, which holds TWO decode blocks by construction — a copied-forward row is its source row
verbatim (its ``decode`` says ``cap`` = the rung that produced its text, ``regenerated: false``,
``source_cap`` = that rung), a re-drawn row carries the source rung's block (``regenerated: true``,
``source_cap`` = the source cap). So each source row's expected block is keyed on ITS OWN
``regenerated`` flag: a copied row against the config's block at its ``source_cap`` (a rung of the
static ladder below the source cap), a re-drawn row against the block at the source cap; a row that
contradicts its flag refuses. A base run's rows (no ``regenerated`` key) are checked against the
source-cap block as before. A copied row KEEPS its ``source_cap`` through this rung, so the 16384
root can hold three caps of text (4096 / 8192 / 16384), each row saying which produced it.
Idempotent: an existing COMPLETE rung root exits 0 without touching the server. Positive controls
are ON (``build_served_target``: salt isolation + steering-bites) before the first re-draw.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as _dt
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Optional

REPO = Path(__file__).resolve().parents[4]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase" / "full22", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import at_cap_static as acs  # noqa: E402  (sibling: the at-cap row rules)
import ladder_decide as ld  # noqa: E402  (sibling: the at-cap disposition file's one reader)
import low_dose_liveness_policy as _low_dose_liveness  # noqa: E402  (the low-dose numeric-liveness seam; scripts/, never src/)

STAGING_NAME = ".shard00.staging"
SHARD_NAME = "shard00"
#: LADDER_RULES.md: an at-cap base row the decision tool left OFF the escalation
#: set is copied forward — still at cap — with WHY on the row.
EXCLUDED_KEY = "escalation_excluded"
EXCLUSIONS = ld.EXCLUSIONS
#: the copy-forward provenance EVERY rung row carries (main writes all three on every row it
#: publishes); a BASE row (run_static's) carries none of them
PROVENANCE_KEYS = ("escalated_from_cap", "regenerated", "source_cap")


def row_source_cap(row: Mapping[str, Any], *, store_cap: int, where: str) -> int:
    """The rung that produced THIS row's text, read from the copy-forward provenance the store
    records and VERIFIED against the row's own decode block (the closing review, 2026-09-10 — the
    16,384 rung's base is the 8,192 producer's mixed-cap store).

    A BASE row (none of ``PROVENANCE_KEYS``) was drawn at the store's own cap. A RUNG row says so
    itself: ``regenerated`` true = re-drawn at the store's cap (``source_cap == store_cap``); false
    = copied forward from the rung that produced it, where it terminated or was left at cap by the
    decision tool (``source_cap < store_cap``, at or below the rung it was copied by,
    ``escalated_from_cap``). In every case the row's ``decode`` block is the block in force for
    THIS text, so ``decode.cap`` IS the source cap and the two must agree. Anything else — a torn
    subset of the marks, a non-bool / non-int mark, marks that disagree with each other or with
    ``decode.cap`` — REFUSES (exit 2): the store's provenance cannot be established and it is never
    read as "at the store's cap"."""
    ident = f"{where}: row (goal {row.get('goal_id')!r}, sample {row.get('sample_idx')!r})"
    present = [k for k in PROVENANCE_KEYS if k in row]
    if not present:
        return int(store_cap)
    if len(present) != len(PROVENANCE_KEYS):
        refuse(f"{ident} carries {present} but not all of {list(PROVENANCE_KEYS)} — torn copy-forward "
               "provenance: the rung that produced this text cannot be established; refusing")
    regenerated, src, esc_from = row["regenerated"], row["source_cap"], row["escalated_from_cap"]
    if not isinstance(regenerated, bool):
        refuse(f"{ident} records regenerated={regenerated!r}, not a bool; refusing")
    for name, val in (("source_cap", src), ("escalated_from_cap", esc_from)):
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            refuse(f"{ident} records {name}={val!r}, not a positive int; refusing")
    dec_cap = (row.get("decode") or {}).get("cap") if isinstance(row.get("decode"), Mapping) else None
    if dec_cap != src:
        what = ("a re-drawn row (regenerated true) re-drawn at another cap than its source_cap says"
                if regenerated else "a copied row (regenerated false) whose block is another rung's — its source_cap is "
                "not a static-ladder rung below the source cap its text was drawn at")
        refuse(f"{ident} records source_cap={src} but its decode block was drawn at cap={dec_cap!r} — the "
               f"copy-forward provenance disagrees with the text's own decode block ({what}: a row that contradicts "
               "its regenerated flag / source_cap marks); refusing")
    if esc_from >= store_cap:
        refuse(f"{ident} records escalated_from_cap={esc_from}, not a rung below this store's cap {store_cap}; "
               "refusing")
    if regenerated and src != store_cap:
        refuse(f"{ident} is marked regenerated (re-drawn at this store's cap {store_cap}) but records "
               f"source_cap={src} — a row re-drawn at another cap than its marks say; it contradicts its regenerated "
               "flag; refusing")
    if not regenerated and (src >= store_cap or src > esc_from):
        refuse(f"{ident} is marked copied forward (regenerated false) but records source_cap={src}, not a static-ladder "
               f"rung below the source cap {store_cap} at or below the rung it was copied by ({esc_from}) — a row that "
               "contradicts its regenerated flag; refusing")
    return int(src)


def load_items(path: Path, *, cap_from: int, base_run: str, attack: str, defense: str,
               trial: int, at_cap_keys: list) -> dict:
    """``{(goal_id, sample_idx): excluded_as | None}`` over EVERY at-cap base row
    from the ``--items`` disposition file: the header must describe THIS base
    store (kind static, cap_from, private_run, attack, defense, trial), every
    at-cap base row must have a row, every row must name an at-cap base row.
    None = escalate (the truncated set); a string = copied forward with
    ``escalation_excluded`` set to it."""
    try:
        header, rows = ld.read_items_file(Path(path))
    except ld.Refusal as exc:
        refuse(f"--items {path}: {exc}")
    want = {"kind": "static", "cap": int(cap_from), "private_run": base_run, "attack": attack,
            "defense": defense, "trial": trial}
    got = {k: header.get(k) for k in want}
    if got != want:
        refuse(f"--items {path}: the file describes {got}, this rung's base is {want}; refusing "
               "to escalate on another store's decision")
    disp: dict = {}
    for r in rows:
        key = (r.get("goal_id"), r.get("sample_idx"))
        if key in disp:
            refuse(f"--items {path}: item {key!r} filed twice")
        disp[key] = r
    at_cap = set(at_cap_keys)
    missing = sorted(at_cap - set(disp), key=str)
    stray = sorted(set(disp) - at_cap, key=str)
    if missing or stray:
        refuse(f"--items {path} does not describe the base store: {len(missing)} at-cap row(s) "
               f"without a disposition (first: {missing[:3]}), {len(stray)} row(s) naming items "
               f"not at cap in the base (first: {stray[:3]})")
    return {k: (None if r["escalate"] else r["excluded_as"]) for k, r in disp.items()}


def refuse(msg: str) -> NoReturn:
    """Exit 2 with the reason on stderr — the rung's one refusal shape."""
    print(f"[escalate-static] REFUSED: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def _seedrep_rung_redraw():
    """``scripts/sdfbase/seedrep/rung_redraw.py`` by path, loaded ONLY when ``--from-materialised``
    is given — the streaming ladder's assembly seam; nothing here changes without the flag."""
    import importlib.util as _ilu
    import sys as _sys
    mod = _sys.modules.get("seedrep_rung_redraw")
    if mod is None:
        spec = _ilu.spec_from_file_location("seedrep_rung_redraw", REPO / "scripts" / "sdfbase" / "seedrep" / "rung_redraw.py")
        mod = _sys.modules["seedrep_rung_redraw"] = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def acs_sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# The escalation config                                                        #
# --------------------------------------------------------------------------- #

def resolve_escalation(cfg: Mapping[str, Any]) -> dict:
    """Validate the ``_cap8192`` config and return the rung's facts: trial,
    seed_base, the preset (+arm), cap_from / cap_to, the base and rung
    private_run names, and the row-level decode blocks at both caps."""
    import jailbreak_run as jr
    from sisteer.eval.attacks.static_common import decode_block

    sd = cfg.get("sampled_decode")
    if not isinstance(sd, dict):
        refuse("the config has no sampled_decode block — an escalation config is a "
               "sampled config whose sampled_decode.escalated_from names the base run")
    if "trial" not in sd:
        refuse("sampled_decode.trial is missing")
    try:
        validated = jr._sampled_decode(cfg, sd["trial"])
    except SystemExit as exc:
        refuse(f"sampled_decode did not validate: {exc}")
    esc = sd.get("escalated_from")
    if not isinstance(esc, dict):
        refuse("sampled_decode.escalated_from is missing — this config is a BASE config, not a "
               "rung; the rung config names {kind: static, cap: <base cap>, private_run: <base run>}")
    if esc.get("kind") != "static":
        refuse(f"sampled_decode.escalated_from.kind={esc.get('kind')!r} != 'static'")
    cap_from = esc.get("cap")
    if isinstance(cap_from, bool) or not isinstance(cap_from, int) or cap_from < 1:
        refuse(f"sampled_decode.escalated_from.cap={cap_from!r} must be a positive int")
    base_run = esc.get("private_run")
    if not isinstance(base_run, str) or not base_run:
        refuse("sampled_decode.escalated_from.private_run must name the base private_run")
    cap_to = jr._static_max_new_tokens(cfg)
    if cap_to <= cap_from:
        refuse(f"the rung cap attacks.static_max_new_tokens={cap_to} must exceed the base cap "
               f"{cap_from} — nothing to escalate to")
    esc_run = cfg.get("private_run")
    if not isinstance(esc_run, str) or not esc_run:
        refuse("the config names no private_run for the rung")
    if esc_run == base_run:
        refuse(f"the rung's private_run {esc_run!r} is the base run itself — a rung is NEVER "
               "written in place")
    trial, seed_base, decode = validated["trial"], validated["seed_base"], validated["decode"]
    ladder = ld.LADDERS["static"]
    if int(cap_from) not in ladder or int(cap_to) not in ladder:
        refuse(f"the rung {cap_from} -> {cap_to} is not a step of the static ladder {list(ladder)}")
    if ladder.index(int(cap_to)) != ladder.index(int(cap_from)) + 1:
        refuse(f"the rung {cap_from} -> {cap_to} skips a rung of the static ladder {list(ladder)}")
    # the config's block at every rung a source row may have been produced at (a rung-shaped source
    # holds copied-forward rows from the rungs below its own cap)
    blocks_by_cap = {int(c): decode_block(decode, cap=int(c), seed_base=seed_base, trial=trial)
                     for c in ladder if int(c) <= int(cap_from)}
    return {"trial": trial, "seed_base": seed_base, "decode": decode, "arm": decode["arm"],
            "cap_from": int(cap_from), "cap_to": int(cap_to), "base_run": base_run,
            "esc_run": esc_run, "ladder": tuple(int(c) for c in ladder),
            "block_from": blocks_by_cap[int(cap_from)], "blocks_by_cap": blocks_by_cap,
            "block_to": decode_block(decode, cap=cap_to, seed_base=seed_base, trial=trial)}


# --------------------------------------------------------------------------- #
# The base store (read-only input)                                             #
# --------------------------------------------------------------------------- #

def expect_block_at(block: Mapping[str, Any], cap: int) -> dict:
    """This rung's decode block (preset / arm / seed_base / trial) at ``cap`` — what a
    row drawn at that rung of THIS ladder records."""
    out = dict(block)
    out["cap"] = int(cap)
    return out


def assert_row_block(row: Mapping[str, Any], *, block: Mapping[str, Any], where: str,
                     what: str) -> int:
    """The ONE per-row rule both readers apply (the closing review, 2026-09-10): the
    row's provenance-verified source cap (:func:`row_source_cap` against the store's cap
    ``block["cap"]``), then its ``decode`` block must be this ladder's block AT THAT cap —
    the sampler / arm / seed_base / trial as strict as ever, the cap the rung that
    produced the text. Returns the source cap."""
    src_cap = row_source_cap(row, store_cap=int(block["cap"]), where=where)
    expect = expect_block_at(block, src_cap)
    if row.get("decode") != expect:
        refuse(f"{where}: row (goal {row.get('goal_id')!r}, sample {row.get('sample_idx')!r}) was "
               f"decoded under {row.get('decode')!r}, not {what} at its source cap {src_cap} "
               f"{expect!r} (preset / arm / seed_base / trial) — refusing")
    return src_cap


def expected_source_block(row: Mapping[str, Any], *, cap_from: int, block_from: Mapping[str, Any],
                          blocks_by_cap: Mapping[int, Mapping[str, Any]], where: str) -> dict:
    """The decode block a SOURCE row must carry, keyed on its own ``regenerated`` flag (module docstring,
    "a rung-shaped source"): absent (a base run) or true -> the source-cap block, and a ``source_cap``,
    when present, must equal the source cap; false (copied forward) -> the block at its ``source_cap``,
    which must be a ladder rung below the source cap."""
    who = f"{where}: row (goal {row.get('goal_id')!r}, sample {row.get('sample_idx')!r})"
    if "regenerated" not in row:
        if "source_cap" in row and int(row["source_cap"]) != int(cap_from):
            refuse(f"{who} carries source_cap={row['source_cap']!r} without a regenerated flag — neither a base "
                   f"row nor a rung row of the source at cap {cap_from}; refusing")
        return dict(block_from)
    if row["regenerated"] is True:
        if int(row.get("source_cap", -1)) != int(cap_from):
            refuse(f"{who} says regenerated: true with source_cap={row.get('source_cap')!r}, but the source rung's "
                   f"cap is {cap_from} — a rung row re-drawn at another cap; refusing")
        return dict(block_from)
    if row["regenerated"] is False:
        sc = row.get("source_cap")
        if isinstance(sc, bool) or not isinstance(sc, int) or sc not in blocks_by_cap or sc >= int(cap_from):
            refuse(f"{who} says regenerated: false with source_cap={sc!r}, which is not a static-ladder rung below "
                   f"the source cap {cap_from} ({sorted(c for c in blocks_by_cap if c < int(cap_from))}); refusing")
        return dict(blocks_by_cap[sc])
    refuse(f"{who} carries regenerated={row['regenerated']!r} (need true | false); refusing")


def load_base_store(base_cell_root: Path, *, family: str, block_from: Mapping[str, Any],
                    seed_base: int, trial: int, cap_from: Optional[int] = None,
                    blocks_by_cap: Optional[Mapping[int, Mapping[str, Any]]] = None
                    ) -> tuple[list[dict], list[dict], list[int]]:
    """Every base row (shard order, file order), every base progress record, and per row
    the PROVENANCE-VERIFIED source cap (the rung that produced its text: ``block_from``'s
    cap for a base row or a re-drawn rung row, the lower rung a copied row records — see
    :func:`row_source_cap`), after the completeness + provenance checks the docstring
    lists. A base that is itself a rung store (the 16,384 rung's 8,192 base) is a
    MIXED-CAP store by construction; each of its rows is validated at ITS cap
    (:func:`assert_row_block`: the row's marks must agree with each other and with its own
    ``decode.cap``, then its block must be this ladder's block AT THAT cap).

    ``cap_from`` / ``blocks_by_cap`` (the rung's facts from :func:`resolve_escalation`;
    lane G2's reader keyed each row's expected block on its ``regenerated`` flag through
    :func:`expected_source_block`) are accepted and CROSS-CHECKED: ``cap_from`` must be the
    store cap ``block_from["cap"]`` and every ``blocks_by_cap[c]`` must equal this ladder's
    block at ``c`` — the two readers agree by construction (the ladder's blocks differ only
    in cap), and a disagreement refuses rather than letting one rule quietly win."""
    store_cap = int(block_from["cap"])
    if cap_from is not None and int(cap_from) != store_cap:
        refuse(f"{base_cell_root}: cap_from={cap_from} but the base block's cap is {store_cap} — the rung's facts disagree")
    for c, blk in (blocks_by_cap or {}).items():
        if dict(blk) != expect_block_at(block_from, int(c)):
            refuse(f"{base_cell_root}: blocks_by_cap[{c}]={dict(blk)!r} is not this ladder's block at {c} "
                   f"{expect_block_at(block_from, int(c))!r} — the two per-row readers would disagree; refusing")
    from sisteer.eval.attacks.static_common import (StaticStoreIncompleteError,
                                                    assert_static_store_complete,
                                                    static_sampling_seed)

    base_cell_root = Path(base_cell_root)
    shards = acs.shard_dirs(base_cell_root)
    if not shards:
        refuse(f"no shard* dirs under {base_cell_root} — the base run has not written this "
               "(attack, defense); nothing to escalate")
    rows: list[dict] = []
    progress: list[dict] = []
    source_caps: list[int] = []
    seen_goals: set = set()
    seen_keys: set = set()
    for shard in shards:
        try:
            assert_static_store_complete({}, shard, family=family)
        except StaticStoreIncompleteError as exc:
            refuse(f"base shard {shard} is INCOMPLETE: {exc}")
        try:
            recs = acs.read_rows(shard / "progress.jsonl")
            srows = acs.read_rows(shard / "responses.jsonl")
        except acs.AtCapRefusal as exc:
            refuse(str(exc))
        want: dict = {}
        for rec in recs:
            gid = rec.get("goal_id")
            if gid in seen_goals or gid in want:
                refuse(f"base goal {gid!r} has more than one terminal progress record "
                       f"(shard {shard.name}) — an ambiguous base; refusing")
            want[gid] = int(rec.get("n_variants", 0))
        got: dict = {}
        for row in srows:
            gid = row.get("goal_id")
            if gid not in want:
                refuse(f"base row (goal {gid!r}) in {shard.name} has no terminal progress "
                       "record — the base shard did not finish this goal; refusing")
            key = (gid, row.get("sample_idx"))
            if key in seen_keys:
                refuse(f"base item {key!r} appears twice — refusing an ambiguous base")
            seen_keys.add(key)
            got[gid] = got.get(gid, 0) + 1
        for gid, n in want.items():
            if got.get(gid, 0) != n:
                refuse(f"base goal {gid!r} has {got.get(gid, 0)} rows but its progress record "
                       f"says n_variants={n} — the base shard {shard.name} is short; refusing")
        for row in srows:
            where = f"{shard / 'responses.jsonl'}"
            try:
                acs.row_at_cap(row, where=where)
            except acs.AtCapRefusal as exc:
                refuse(str(exc))
            # the block at the row's OWN source cap: a base row or a re-drawn rung row at the base
            # store's cap, a copied-forward rung row at the lower rung its provenance names
            source_caps.append(assert_row_block(
                row, block=block_from, where=where,
                what="this rung's base block (the named base run is not the sampled base of this rung)"))
            expect_seed = static_sampling_seed(seed_base, trial, family, row["goal_id"],
                                               row["sample_idx"])
            if row.get("sampling_seed") != expect_seed:
                refuse(f"{where}: row (goal {row['goal_id']!r}, sample {row['sample_idx']!r}) "
                       f"records sampling_seed={row.get('sampling_seed')!r} but "
                       f"(seed_base, trial, family, goal_id, sample_idx) gives {expect_seed} — "
                       "refusing a base whose seeds do not reproduce")
        seen_goals.update(want)
        rows.extend(srows)
        progress.extend(recs)
    return rows, progress, source_caps


def assert_base_unpadded(rows: list, *, base_cell_root: Path, base_run: str) -> None:
    """Refuse a base whose rows carry a ``padding`` block (a store drawn under
    ``jailbreak_run.py generate --static-padding-tokens``). :func:`regenerate` sends
    ``row["attack_prompt"]``, which in a padded store is deliberately the UNPADDED
    text; a re-draw here would issue every at-cap item without its padding and file
    the result under the padded store's ladder — the rung driver does not support
    padded stores. Runs before the target is built and before any request."""
    padded = [r for r in rows if isinstance(r.get("padding"), Mapping)]
    if not padded:
        return
    tokens = sorted({r["padding"].get("tokens") for r in padded}, key=lambda t: (t is None, t))
    refuse(f"base {base_run} ({base_cell_root}) is a PADDED store: {len(padded)} of {len(rows)} row(s) "
           f"carry a padding block ({', '.join(str(t) for t in tokens)} padding tokens). This rung "
           "re-draws row['attack_prompt'], which in a padded store is the UNPADDED text, so a re-draw "
           "here would issue every item WITHOUT its padding — a different measurement wearing the "
           "store's name. The rung driver does not support padded stores; regenerate the padded run at "
           "the larger cap instead (jailbreak_run.py generate with attacks.static_max_new_tokens raised, "
           "the same --static-padding-* flags, its own private_run)")


# --------------------------------------------------------------------------- #
# Divergence of a re-draw against its base text                                #
# --------------------------------------------------------------------------- #

def first_divergence(a, b) -> int:
    """The first index at which sequences ``a`` and ``b`` differ (``min(len)``
    when one is a proper prefix of the other)."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def divergence_fields(base_text: str, new_text: str,
                      tokenize: Optional[Callable[[str], list]]) -> dict:
    """``prefix_extension`` / ``identical`` / ``first_divergence_char`` /
    ``first_divergence_token`` for a regenerated row against the base text of the
    same item. Byte semantics for the prefix test; the token index needs the
    target tokenizer (``tokenize``), required whenever the texts fork."""
    base_b, new_b = base_text.encode("utf-8"), new_text.encode("utf-8")
    prefix = new_b.startswith(base_b)
    out = {"prefix_extension": bool(prefix), "identical": base_b == new_b,
           "first_divergence_char": None, "first_divergence_token": None}
    if prefix:
        return out
    out["first_divergence_char"] = first_divergence(base_text, new_text)
    if tokenize is None:
        refuse("a regenerated text forks from its base and no tokenizer is loaded to place the "
               "first divergent token — refusing rather than writing null for a fork")
    out["first_divergence_token"] = first_divergence(list(tokenize(base_text)),
                                                     list(tokenize(new_text)))
    return out


def load_tokenizer_for(cfg: Mapping[str, Any]) -> Callable[[str], list]:
    """``text -> token ids`` under the target tokenizer at the config's
    model / model_revision, tokenized WITHOUT special tokens (a seam the tests
    stub)."""
    from sisteer.models import load_tokenizer

    tok = load_tokenizer(cfg["model"], revision=cfg.get("model_revision"))

    def tokenize(text: str) -> list:
        return list(tok(text, add_special_tokens=False)["input_ids"])

    return tokenize


# --------------------------------------------------------------------------- #
# The re-draws (bounded thread pool over the detailed surface)                 #
# --------------------------------------------------------------------------- #

def regenerate(target, items: list[tuple[int, dict]], *, cap_to: int, preset: Mapping[str, Any],
               concurrency: int) -> dict:
    """Re-draw every ``(index, base_row)`` item at ``cap_to`` with its recorded
    ``sampling_seed`` through ``target.generate_detailed`` (one request per job,
    ``concurrency`` in flight). Returns ``{index: detailed record}``; ANY failure
    propagates after the pool is torn down — the caller writes nothing."""
    out: dict = {}
    if not items:
        return out
    # request evidence: the re-draw of a row is labelled with the row's OWN identity — the store's
    # canonical task ``static/<family>`` and its ``(goal_id, sample_idx)`` — exactly as ``run_static`` labels the base
    # draw (an un-labelled ``generate_detailed`` under the hook is refused by name before any byte is sent)
    from sisteer.eval.request_evidence_client import LEG_STATIC, store_task_name

    def one(item):
        idx, row = item
        label = (store_task_name(LEG_STATIC, str(row["family"])), str(row["goal_id"]), int(row["sample_idx"]))
        rec = target.generate_detailed([row["attack_prompt"]], max_new_tokens=cap_to,
                                       decode=preset, seeds=[int(row["sampling_seed"])], labels=[label])[0]
        if rec.get("error") is not None:
            raise RuntimeError(f"re-draw of (goal {row['goal_id']!r}, sample {row['sample_idx']!r}) "
                               f"failed: {rec['error']}")
        if rec.get("finish_reason") is None:
            raise RuntimeError(f"re-draw of (goal {row['goal_id']!r}, sample {row['sample_idx']!r}) "
                               "returned no finish_reason — the ladder keys on it")
        if rec.get("completion_tokens") is None:
            raise RuntimeError(f"re-draw of (goal {row['goal_id']!r}, sample {row['sample_idx']!r}) "
                               "returned no completion_tokens — boundary_landing needs it")
        if rec.get("sampling_seed") != int(row["sampling_seed"]):
            raise RuntimeError(f"re-draw of (goal {row['goal_id']!r}, sample {row['sample_idx']!r}) "
                               f"echoed seed {rec.get('sampling_seed')!r}, not {row['sampling_seed']}")
        return idx, rec

    pool = cf.ThreadPoolExecutor(max_workers=max(1, int(concurrency)))
    try:
        futures = [pool.submit(one, it) for it in items]
        for fut in cf.as_completed(futures):
            idx, rec = fut.result()  # the first failure raises here
            out[idx] = rec
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return out


# --------------------------------------------------------------------------- #
# The rung root                                                                 #
# --------------------------------------------------------------------------- #

def existing_complete(shard_dir: Path, *, n_rows: int,
                      expect_decode: Mapping[str, Any]) -> Optional[dict]:
    """The ``escalation.json`` of an existing COMPLETE rung root (idempotence),
    ``None`` when the dir does not exist; a dir that exists but is not complete
    REFUSES (never silently overwritten).

    Complete means the three data files with the base's row count AND the
    store's identity (P2-β): ``served_target.json`` present,
    ``manifest_stage == "final"``, and its decode block the same target decode
    as ``expect_decode`` (this rung's ``block_to``) under the strict rule of
    ``served_target.assert_target_decode_compatible``. The sidecar is written
    into the staging dir and renamed into place WITH the rows, so a
    complete-looking root without it is a partial copy or a lost file whose
    rows' sampler / seed policy / trial cannot be established — resumed as
    complete, it went straight to ``ladder_decide.py``, which reads rows with
    no identity check of its own.

    AND its rows read under the same per-row rule the base store is read with
    (the closing review, 2026-09-10 — ``assert_row_block`` at ``expect_decode``):
    a rung root is a MIXED-CAP store by construction (re-drawn rows at this
    rung's cap, copied rows at the rung that produced them, each saying which),
    so the rows are never required to carry one block; a row whose marks
    contradict each other or its own decode block is not this rung's store and
    is refused, not resumed. The sidecar stays the identity authority for WHAT
    the target decoded under; the rows' provenance is the other fact."""
    shard_dir = Path(shard_dir)
    if not shard_dir.exists():
        return None
    esc_p = shard_dir / "escalation.json"
    resp_p = shard_dir / "responses.jsonl"
    prog_p = shard_dir / "progress.jsonl"
    if not (esc_p.exists() and resp_p.exists() and prog_p.exists()):
        refuse(f"{shard_dir} exists but is not a complete rung root (escalation.json / "
               "responses.jsonl / progress.jsonl missing) — remove it deliberately before "
               "re-running; refusing to overwrite")
    try:
        esc = json.loads(esc_p.read_text())
        root_rows = acs.read_rows(resp_p)
        have = len(root_rows)
    except (json.JSONDecodeError, acs.AtCapRefusal) as exc:
        refuse(f"{shard_dir} is unreadable ({exc}) — refusing to overwrite")
    if esc.get("n_rows") != have or have != n_rows:
        refuse(f"{shard_dir} holds {have} rows but escalation.json says {esc.get('n_rows')} and "
               f"the base run has {n_rows} — a partial or foreign rung root; refusing to overwrite")
    from sisteer.eval.attacks.served_target import (
        ServedTargetError,
        assert_target_decode_compatible,
    )
    from sisteer.eval.attacks.static_common import load_covered_goals

    manifest_p = shard_dir / "served_target.json"
    if not manifest_p.is_file():
        refuse(f"{shard_dir} holds escalation.json / responses.jsonl / progress.jsonl ({have} rows) "
               "but NO served_target.json — the sidecar is staged and renamed into place WITH the "
               "rows, so its absence means a partial copy or a lost file: the sampler / seed policy / "
               "trial of these rows cannot be established. Refusing to resume it as complete or to "
               "overwrite it; restore the sidecar, or remove the root deliberately and re-run")
    try:
        manifest = json.loads(manifest_p.read_text())
    except json.JSONDecodeError as exc:
        refuse(f"{manifest_p} is unreadable ({exc}) — refusing to resume the rung root or to "
               "overwrite it")
    stage = manifest.get("manifest_stage") if isinstance(manifest, dict) else None
    if stage != "final":
        refuse(f"{manifest_p}: manifest_stage={stage!r}, not 'final' — a rung root is published with "
               f"its final manifest in one rename, so a non-final sidecar beside {have} rows is a "
               "partial copy; refusing to resume it or to overwrite it")
    if not load_covered_goals(prog_p):
        refuse(f"{shard_dir}: progress.jsonl records no successfully finished goal beside {have} "
               "rows — not a complete rung root; refusing to resume it or to overwrite it")
    try:
        assert_target_decode_compatible(shard_dir, expect_decode)
    except ServedTargetError as exc:
        refuse(f"{shard_dir}: the existing rung root is not this rung's store — {exc}")
    for row in root_rows:
        assert_row_block(row, block=expect_decode, where=str(resp_p),
                         what="this rung's block (the existing rung root is not this rung's store)")
    return esc


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", required=True, help="the _cap<N> rung config (sampled_decode.escalated_from)")
    ap.add_argument("--attack", required=True)
    ap.add_argument("--defense", required=True, help="the TrainTest cell id")
    ap.add_argument("--private-root", required=True)
    ap.add_argument("--proxy-url", required=True)
    ap.add_argument("--routing-table", required=True)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="the standup's window (arms the once-only max_tokens clamp; provenance)")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="re-draws in flight (a bounded thread pool over the detailed surface)")
    ap.add_argument("--items", default=None, metavar="FILE",
                    help="LADDER_RULES.md: the at-cap disposition file ladder_decide.py static "
                         "wrote for the BASE store of this (attack, defense). Only the rows it "
                         "marks escalate are re-drawn; every other at-cap row is copied forward "
                         f"with '{EXCLUDED_KEY}' (loop|repetitive|boundary). Without it every "
                         "at-cap row is re-drawn (the running jobs' behaviour)")
    ap.add_argument("--from-materialised", default=None, metavar="LADDER_ROOT",
                    help="seedrep streaming ladder (plan §7.3, WS2-E): the CLOSED queue root .../ladder/<leg>/<view>/<cap>; "
                         "an escalating item with an elected, VERIFIED materialised draw is taken from it, every other "
                         "one is drawn here. Default off: without it this rung is byte-identical to before")
    ap.add_argument("--results", default=None, metavar="SNAPSHOT",
                    help="with --from-materialised: the results Dict snapshot (rung_redraw.py snapshot)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import jailbreak_run as jr
    from sisteer.eval.attacks.served_target import build_served_target, write_target_manifest
    from sisteer.eval.attacks.static_common import ensure_private_dir, sampler_preset_of

    if args.attack not in jr._SAMPLED_STATIC_FAMILIES:
        refuse(f"--attack {args.attack!r} is not one of the sampled static families "
               f"{sorted(jr._SAMPLED_STATIC_FAMILIES)}")
    cfg = jr._load_cfg(args.config)
    if args.defense not in jr._valid_defense_ids(cfg):
        refuse(f"unknown --defense {args.defense!r} for {args.config}")
    rung = resolve_escalation(cfg)
    root = Path(args.private_root)
    base_cell_root = root / rung["base_run"] / args.attack / args.defense
    rows, progress, source_caps = load_base_store(base_cell_root, family=args.attack,
                                                  block_from=rung["block_from"],
                                                  seed_base=rung["seed_base"], trial=rung["trial"],
                                                  cap_from=rung["cap_from"], blocks_by_cap=rung["blocks_by_cap"])
    # a padded base refuses HERE — before the rung root is examined, before the
    # target's positive controls, before a single re-draw (docs/static_padding.md)
    assert_base_unpadded(rows, base_cell_root=base_cell_root, base_run=rung["base_run"])
    esc_cell_root = root / rung["esc_run"] / args.attack / args.defense
    shard_dir = esc_cell_root / SHARD_NAME
    done = existing_complete(shard_dir, n_rows=len(rows), expect_decode=rung["block_to"])
    if done is not None:
        print(f"[escalate-static] {shard_dir} is already a complete rung root — nothing to do",
              file=sys.stderr, flush=True)
        print(json.dumps(done), flush=True)
        return 0
    at_cap_items = [(i, r) for i, r in enumerate(rows) if r["finish_reason"] == "length"]
    # LADDER_RULES.md: with --items only the TRUNCATED at-cap rows re-draw; the
    # loop / repetitive / boundary ones are copied forward, still at cap, stamped
    disposition: dict = {}
    if args.items:
        disposition = load_items(Path(args.items), cap_from=rung["cap_from"], base_run=rung["base_run"],
                                 attack=args.attack, defense=args.defense, trial=rung["trial"],
                                 at_cap_keys=[(r["goal_id"], r["sample_idx"]) for _, r in at_cap_items])
    excluded = {(r["goal_id"], r["sample_idx"]): disposition[(r["goal_id"], r["sample_idx"])]
                for _, r in at_cap_items
                if disposition.get((r["goal_id"], r["sample_idx"])) is not None}
    regen_items = [(i, r) for i, r in at_cap_items if (r["goal_id"], r["sample_idx"]) not in excluded]
    n_excluded_by = {c: sum(1 for v in excluded.values() if v == c) for c in EXCLUSIONS}
    print(f"[escalate-static] {args.attack}/{args.defense}: base {rung['base_run']} has "
          f"{len(rows)} rows, {len(at_cap_items)} at cap {rung['cap_from']}"
          + (f", {len(excluded)} excluded by --items {n_excluded_by}" if args.items else "")
          + f" -> re-drawing {len(regen_items)} at {rung['cap_to']} (seeds unchanged), "
          f"concurrency {args.concurrency}", flush=True)

    tokenize = load_tokenizer_for(cfg) if regen_items else None
    # at low dose the steering_bites text probe inside build_served_target refuses a steer that is live and merely
    # small; when SISTEER_LOW_DOSE_LIVENESS_RECORD names a numeric record for THIS cell and THIS routing table that one
    # refusal is recorded as an observation (served_target.json) and the rung proceeds. Env unset: nothing installed.
    _ld_guard = _low_dose_liveness.SteeringBitesGuard(args.defense, args.routing_table,
                                                      log=lambda m: print(f"[escalate-static] {m}", flush=True))
    with _ld_guard.installed():
        target = build_served_target(proxy_url=args.proxy_url, routing_table=args.routing_table,
                                     cell_id=args.defense, controls=True,
                                     max_model_len=args.max_model_len)
    _ld_guard.fold_into(target.evidence)
    sb = target.evidence.get("steering_bites", {})
    print(f"[escalate-static] target vmodel={target.virtual_model} steering_bites="
          f"{'verified' if sb.get('verified') else sb.get('reason', 'n/a')}", flush=True)
    preset = sampler_preset_of(rung["block_to"])
    if args.from_materialised:  # the streaming ladder's assembly — default off, byte-identical without it
        if not args.results:
            refuse("--from-materialised needs --results <the results Dict snapshot>")
        target = _seedrep_rung_redraw().wrap_static_target(
            target, regen_items, ladder_root=args.from_materialised, results=args.results, family=args.attack,
            defense=args.defense, seed_base=rung["seed_base"], cap_to=rung["cap_to"], preset=preset,
            log=lambda m: print(f"[escalate-static] {m}", flush=True))
    regen = regenerate(target, regen_items, cap_to=rung["cap_to"], preset=preset,
                       concurrency=args.concurrency)
    if len(regen) != len(regen_items):
        refuse(f"{len(regen)} re-draws for {len(regen_items)} escalating items — refusing a short rung")

    out_rows: list[dict] = []
    n_after = n_prefix = n_fork = n_boundary = n_identical = 0
    n_after_regenerated = n_after_excluded = 0
    regen_per_goal: dict = {}
    for i, base in enumerate(rows):
        row = dict(base)
        row["escalated_from_cap"] = rung["cap_from"]
        # the rung that produced the base text (load_base_store verified it against the row's
        # own decode block): the base store's cap for a base row or a re-drawn rung row, the
        # lower rung a copied row records
        src_cap = source_caps[i]
        if i not in regen:
            row["regenerated"] = False
            # PRESERVED, never relabelled to this rung's base cap: a row that terminated at 4,096
            # still says 4,096 inside the 16,384 store (the closing review, 2026-09-10); for a
            # rung-shaped source this IS the source row's own source_cap (row_source_cap read it)
            row["source_cap"] = src_cap
            key = (base["goal_id"], base["sample_idx"])
            if key in excluded:
                row[EXCLUDED_KEY] = excluded[key]
                n_after += 1  # copied forward STILL at cap
                n_after_excluded += 1
            out_rows.append(row)
            continue
        rec = regen[i]
        for stale in ("max_tokens_clamped_from", "prompt_tokens_source", "window_mismatch"):
            row.pop(stale, None)
        row["response"] = rec["text"]
        row["decode"] = dict(rung["block_to"])
        row["finish_reason"] = rec["finish_reason"]
        row["completion_tokens"] = rec["completion_tokens"]
        row["prompt_tokens"] = rec.get("prompt_tokens")
        row["max_tokens"] = int(rec["max_tokens"])
        row["at_cap"] = rec["finish_reason"] == "length"
        if rec.get("max_tokens_clamped_from") is not None:
            row["max_tokens_clamped_from"] = int(rec["max_tokens_clamped_from"])
            row["prompt_tokens_source"] = rec.get("prompt_tokens_source")
        if rec.get("window_mismatch") is not None:
            row["window_mismatch"] = rec["window_mismatch"]
        # the request evidence the record carries, when it does (the streaming ladder's elected materialised draw;
        # a served re-draw whose target records them): the ids the audit binds this ROW through — copied forward
        # only when present, so a record without them leaves the row exactly as it was
        for evidence_key in ("request_id", "completion_id"):
            if rec.get(evidence_key) is not None:
                row[evidence_key] = rec[evidence_key]
        row["regenerated"] = True
        row["source_cap"] = rung["cap_to"]
        row["source_completion_tokens"] = base.get("completion_tokens")
        row["source_finish_reason"] = base.get("finish_reason")
        # the cap the base text was drawn at — the base store's cap, except for an at-cap row the
        # decision tool had left at a lower rung; the boundary test is against THAT cap
        row["source_draw_cap"] = src_cap
        row["boundary_landing"] = int(rec["completion_tokens"]) < src_cap
        row.update(divergence_fields(base["response"], rec["text"], tokenize))
        n_after += int(row["at_cap"])
        n_after_regenerated += int(row["at_cap"])
        n_prefix += int(row["prefix_extension"])
        n_fork += int(not row["prefix_extension"])
        n_identical += int(row["identical"])
        n_boundary += int(row["boundary_landing"])
        regen_per_goal[base["goal_id"]] = regen_per_goal.get(base["goal_id"], 0) + 1
        out_rows.append(row)

    out_progress: list[dict] = []
    for rec in progress:
        p = dict(rec)
        n_re = regen_per_goal.get(rec.get("goal_id"), 0)
        # the REAL target forwards this rung spent on the goal (the base record's
        # value counted the base draws); the copied count is stated beside it
        p["n_target_forwards"] = n_re
        p["n_regenerated"] = n_re
        p["n_copied"] = int(rec.get("n_variants", 0)) - n_re
        out_progress.append(p)

    escalation = {
        "attack": args.attack, "defense": args.defense,
        "cap_from": rung["cap_from"], "cap_to": rung["cap_to"],
        "n_rows": len(out_rows), "n_at_cap_before": len(at_cap_items),
        "n_regenerated": len(regen), "n_at_cap_after": n_after,
        "n_copied": len(out_rows) - len(regen),
        "n_prefix_extension": n_prefix, "n_fork": n_fork, "n_identical": n_identical,
        "n_boundary_landing": n_boundary,
        # LADDER_RULES.md: the decision tool's exclusions (--items). n_at_cap_after
        # counts the WHOLE rung store (re-drawn rows still at cap + excluded rows
        # copied forward at cap); the two parts are stated beside it.
        "item_filter": bool(args.items),
        "items_file": None if not args.items else str(args.items),
        "items_file_sha256": None if not args.items else acs_sha256(Path(args.items)),
        "n_escalation_excluded": len(excluded),
        "n_escalation_excluded_by_class": n_excluded_by,
        "n_at_cap_after_regenerated": n_after_regenerated,
        "n_at_cap_after_excluded": n_after_excluded,
        # the rung root is a MIXED-CAP store: every row at the block of the rung that produced its
        # text (re-drawn rows at cap_to, copied rows at their preserved source cap), counted by cap
        "n_rows_by_source_cap": {str(c): sum(1 for r in out_rows if r["source_cap"] == c)
                                 for c in sorted({r["source_cap"] for r in out_rows})},
        "seed_base": rung["seed_base"], "trial": rung["trial"], "arm": rung["arm"],
        "decode": dict(rung["block_to"]),
        "base_private_run": rung["base_run"], "private_run": rung["esc_run"],
        "n_base_shards": len(acs.shard_dirs(base_cell_root)),
        "concurrency": int(args.concurrency), "max_model_len": args.max_model_len,
        "config": str(args.config), "routing_table": str(args.routing_table),
        "n_target_forwards": target.n_target_forwards,
        "re_draw_not_continuation": True,
        "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }

    # stage, then rename into place: a rung root either exists complete or not at all
    ensure_private_dir(esc_cell_root)
    staging = esc_cell_root / STAGING_NAME
    if staging.exists():
        shutil.rmtree(staging)  # ours by construction, never complete
    ensure_private_dir(staging)
    _write_jsonl(staging / "responses.jsonl", out_rows)
    _write_jsonl(staging / "progress.jsonl", out_progress)
    # the CLIENT hop's timeout (the ProxyClientEngine build_served_target constructs; proxy.CLIENT_TIMEOUT_S
    # by import, never a retyped number), recorded where hop_timeout_guard.py reads it back beside the
    # standup's upstream_timeout_s: a transport-killed draw reads as a SHORT completion, so the rung's
    # manifest says what could have killed one
    from sisteer.capeval.serving.proxy import CLIENT_TIMEOUT_S
    write_target_manifest(staging, target, stage="final", decode=rung["block_to"],
                          extra={"client_timeout_s": float(CLIENT_TIMEOUT_S),
                                 "client_timeout_source": "sisteer.capeval.serving.proxy.CLIENT_TIMEOUT_S"})
    (staging / "escalation.json").write_text(json.dumps(escalation, indent=2, default=str))
    os.replace(staging, shard_dir)
    print(f"[escalate-static] wrote {shard_dir}: {escalation['n_rows']} rows, "
          f"{escalation['n_regenerated']} re-drawn ({escalation['n_at_cap_after']} still at cap "
          f"{rung['cap_to']}; {n_prefix} prefix-extensions, {n_fork} forks, {n_boundary} boundary "
          f"landings), {escalation['n_copied']} copied", file=sys.stderr, flush=True)
    print(json.dumps(escalation), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
