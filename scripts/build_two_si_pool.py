"""Build (and freeze) the multi-SI matched training-pool manifest.

Subcommands, deliberately separable (task piece A vs pieces B/C):

  manifest      PIECE A — CPU-only, no GPU, no network. Reads the already-graded
                rows from the mode-0700 machine-local store (undefended `none`
                + the SAFETY_SI `prompted` rows), the per-SI `pos` rows
                generate-pos has produced (constitution + the stage-3
                irrelevant SI), and the benign PHTest rows generate-benign/
                grade-benign have produced; computes the N-way matched
                intersection, the benign all-comply intersection, the 50/50
                balance, the >=200 hard floor, the composition assertion and
                the identity assertions, and freezes a COUNTS-ONLY manifest to
                `cfg["pool_manifest"]` — the artifact `scripts/two_si_run.py
                aggregate` reads first and refuses to run without. Runs
                standalone against whatever rows exist; when an SI's side is
                missing it says so (coverage gaps). Under the stage-3 gates a
                thin/confounded pool RAISES rather than falling back.

  generate-pos  PIECE B — GPU-side, GENERATION ONLY. Fills the coverage
                gaps A reports for ONE SI (`--si`, default si_constitution):
                runs the SI-prompted model over every `none`-successful train
                prompt that still lacks an accepted `pos` and persists the
                responses UNGRADED (`score: null`, `graded: false`) to that
                SI's private pos run. `--candidates K` draws K rejection-
                sampling candidates per prompt (the irrelevant SI reaches
                parity with the safety SIs by raising K;
                only THWARTING completions are later accepted). It NEVER
                builds a grader client — the Low-Prio key lives in `~/.config`
                on the grading host and cluster `/home` is node-local, so
                grading on this path crashes. `--dry-run` reports the gap
                counts without loading a model (CPU-only). The heavy imports
                (the model, the generator) are lazy so `manifest` never pays
                for them.

  grade-pos     PIECE C — on the grading host. Reads the ungraded rows
                generate-pos wrote (EVERY pos run, or one `--si`), grades each
                with the live Low-Prio StrongREJECT judge, and rewrites the
                rows with their score so a subsequent `manifest` run counts
                them as usable `pos` rows. Grader failure -> the row stays
                UNSCORED and the run fails loudly; never a fabricated refusal
                0 (JB1). Resumable + cache-aware.

  generate-benign  PIECE B' — GPU-side, GENERATION ONLY. The stage-3 benign
                half: loads the PHTest hard-benign
                prompts (content_sets.load_phtest, pinned revision) and
                generates, per prompt, the undefended `none` response (the
                shared `neg`) plus one SI-prompted response per configured SI
                — all persisted UNGRADED to the benign run. Resumable per
                (prompt, condition).

  grade-benign  PIECE C' — grading-host-side. Grades the ungraded benign SI rows
                with the live XSTest C/P/I judge and rewrites them with their
                `xstest_label`; `manifest` then admits only the items EVERY
                SI complied with (rubric C — content_sets.
                select_xstest_complied is the single acceptance helper).

  grade-store   grading-host-side. Grades the labelled STORE itself — the step the
                `pool_sample.py sample-all` -> `merge` lane needs (that lane
                writes every merged row `score: null` by construction, and
                grade-pos/grade-benign only ever read the generate-pos/-benign
                RUN dirs, never the store; verified 2026-08-20). StrongREJECT
                for every harmful-family pool row under EVERY condition (the
                undefended `none` row's score decides requirement_a
                strict-vs-relaxed, so it is graded too), XSTest C/P/I for the
                benign SI rows — plus, when pool.bidpo_benign is enabled,
                the benign `none` rows too:
                bidpo's reversed side needs the undefended response VERIFIED
                as a compliance. Resume-safe; atomic per-file rewrites.

  generate-anti-neg  PIECE B'' — GPU-side, GENERATION ONLY. bidpo's anti-safety
                negatives (bidpo amendment 2026-08-20): for EVERY harmful `none`
                train prompt, sample K candidate completions under
                `safety_refusal.anti_text` (the AUTHORED anti text, reused) and
                persist them UNGRADED to `pool.anti_neg.run`, keyed with the
                `anti_si_id` pseudo-SI. Uses the SAME vLLM sampler engine as the
                main array (`sampling.engine`), resumable per cell
                (`engine.cell_key` / `_anti_neg_done_keys`) — a re-run or a
                raised K generates only the delta. No grader here.
                Requires `pool.anti_neg.enabled`. Data-parallel: `--shard-
                index/--shard-count` slice the prompt inventory (disjoint
                contiguous shards, one output file per shard), and
                `generate-anti-neg-all` fans out one worker per GPU exactly
                like `pool_sample.py sample-all`.

  grade-anti-neg  PIECE C'' — grading-host-side. Grades the anti-neg rows with the
                live Low-Prio StrongREJECT judge and rewrites their score;
                `manifest` then accepts the first candidate scored a genuine
                COMPLIANCE (> threshold) as bidpo's rejected side, in a SEPARATE
                `bidpo_anti_neg` block that never alters matched_intersection's
                `neg` (bidpo-only; every other arm keeps consuming the undefended
                `neg`, identity invariant untouched).

  emit-pairs    PIECE D — turns the FROZEN manifest into the
                `{prompt,pos,neg}` jsonl `scripts/train_two_si_objects.py
                train --pairs` consumes. Reads the frozen manifest
                (tamper-checked against its embedded `manifest_sha256`) plus
                the private stores, resolves every manifest pair id back to
                its exact text BY HASH (the manifest is authoritative — a
                store row whose sha no longer matches fails loudly, never a
                silently different pool), and writes ONE jsonl per SI under a
                mode-0700 dir. Harmful text: the output NEVER lands inside the
                repo (guarded), only under the private root. Regime-aware: in
                `matched_intersection` the per-SI files carry the SAME ordered
                prompts and the SAME `neg`, differing only in `pos` —
                re-asserted ON WRITE via `assert_pools_identical_except_pos`
                (reused, never re-derived); in `native_per_si` each SI's own
                pool is emitted and a `pairs_meta.json` sidecar records the
                regime + pool sizes so a later reader cannot mistake which
                comparison the objects came from (the sidecar is written in
                BOTH regimes). Only GRADED rows qualify as a `pos` (the
                three-state rule); ungraded rows are skipped and their count
                reported.

Engines (ruling 2026-08-25): the generation legs select their engine via
`--engine` / `pool.engine` — `vllm` (the campaign path: the shared
prefix-cached engine, with HF-teacher sampling parity ported from the
vllm-generate-pos lane) or `hf` (the RETAINED contingency, byte-for-byte the
pre-ruling behaviour); generate-anti-neg is vllm-only. Every real generation
records a `generation_provenance.json` sidecar (engine, vLLM version, sampling
params, model + revision) in its run dir; a resume with different provenance
REFUSES loudly, `manifest` embeds the records and refuses an engine-mixed pool
(`--allow-engine-mix` records the sanctioned HF-contingency confound
instead), and emit-pairs copies the block into pairs_meta.json so the emitted
pairs carry their own engine record.

Containment: attack prompts and completions never leave the mode-0700 store or
the manifest boundary. The manifest holds counts, ids and hashes only
(`matched_pool.assert_no_store_text` enforces this before it is frozen); the
emit-pairs output holds text but only ever inside a mode-0700 private dir."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # jailbreak_run

import jailbreak_run  # reuse its config loader / private-root resolution
import pool_sample  # reuse its contiguous _shard + GPU discovery (sample-all)

REPO = Path(__file__).resolve().parents[1]


def _degen(text):
    """`{degenerate, degenerate_reason}` for one completion.

    Computed from the TEXT at emit time rather than read off the store, so it
    cannot race a grader rewriting the store and cannot go stale if a row is
    regenerated. Same detector the store stamper uses.
    """
    from sisteer.eval.degeneracy import classify
    return classify(text)


# The labelled store's `prompted` condition was recorded under SAFETY_SI,
# so a store `prompted` row IS the safety_refusal SI-prompted teacher. Every
# OTHER SI's pos rows must be generated (generate-pos) into its own pos run —
# there is no store row for them. (The pool_sample store path later stamped
# every SI row with an explicit `si_id` — safety_refusal included — so in a
# merged store the bare-`prompted` mapping is the LEGACY encoding and the
# explicit rows are the live one; both are the same source and are treated
# identically by `_store_pos_si` below.)
_STORE_PROMPTED_SI = "safety_refusal"


def _store_pos_si(cfg: dict) -> "str | None":
    """Which SI the labelled STORE still serves harmful `pos` rows for.

    Historically always `_STORE_PROMPTED_SI`: its teacher rows were
    recorded in the store — as bare ``si_condition=="prompted"`` rows in the
    original store, as explicit ``si_id`` rows in the pool_sample merge — and
    generate-pos refused to generate for it, so the store row was the ONLY
    source and its realised K was pinned at 1.

    Declaring ``pool.pos_runs.<store SI>`` flips that SI to a GENERATED pos
    source, giving equal rejection-sampling K=8 for all three SIs — at K=1
    the store teacher's ~44% pasttense compliance gated the all-SIs matched
    intersection there at 33.7%. The generated run then becomes
    AUTHORITATIVE for that SI's harmful pos rows and the store's rows for it
    are EXCLUDED from pool assembly — never merged, never a silent mix of
    the fresh rejection-sampled target with the historical single draw.
    (Admitting both sources would collide at candidate_idx 0: matched_pool's
    `place_cand` raises `conflicting graded verdicts` whenever the two draws
    disagree, and silently interleaves them when they agree.)

    BENIGN rows are deliberately NOT affected: generate-pos is harmful-side
    only, so the benign half's SI rows (XSTest-labelled) have no generated
    replacement and the store remains their source for EVERY SI.

    Returns the store-served SI, or None once a pos run is declared for it.
    """
    if _STORE_PROMPTED_SI in (cfg["pool"].get("pos_runs") or {}):
        return None
    return _STORE_PROMPTED_SI


# --------------------------------------------------------------------------- #
# config helpers — pos runs, benign run, scoring-disjointness inputs            #
# --------------------------------------------------------------------------- #


def _pos_runs(cfg: dict) -> dict[str, str]:
    """`si_id -> private run name` for every SI whose pos rows are GENERATED.

    `pool.pos_runs` declares them; the legacy `pool.constitution_run` key
    still names si_constitution's run, and any undeclared non-store SI
    defaults to `two_si_pos_<si_id>`. The store's own prompted SI
    (`_STORE_PROMPTED_SI`) joins ONLY when `pool.pos_runs` names a run for it
    EXPLICITLY — that declaration is the flip that makes the run authoritative
    and excludes the store's rows for that SI (`_store_pos_si`); with no
    declaration it keeps its historical store-served behaviour and is absent
    here, exactly as before."""
    pool = cfg["pool"]
    declared = dict(pool.get("pos_runs") or {})
    if "si_constitution" not in declared:
        declared["si_constitution"] = pool.get(
            "constitution_run", "two_si_pos_constitution")
    return {si: declared.get(si, f"two_si_pos_{si}")
            for si in cfg["si_ids"]
            if si != _STORE_PROMPTED_SI or si in declared}


# bidpo's anti-safety negative source (bidpo amendment 2026-08-20). The pseudo-SI
# id the anti-neg rows are stamped with is NOT a comparison SI — it is bidpo's
# dispreferred side only, generated under safety_refusal.anti_text.
_ANTI_SI_DEFAULT = "anti_safety_refusal"
# The legacy single-GPU output file. A SHARDED run (generate-anti-neg
# --shard-index/--shard-count, or the generate-anti-neg-all fan-out) writes one
# `anti_neg_shard_NN.jsonl` per shard instead — N appending processes must
# never share a file (O_APPEND interleaving is not safe for multi-line JSONL
# flushes). Readers take BOTH via `_anti_neg_paths`.
_ANTI_NEG_FILE = "anti_neg.jsonl"


def _anti_neg_shard_file(shard_index: int, shard_count: int) -> str:
    """The output filename for one generate-anti-neg worker. shard 0/1 (the
    default) is the legacy shard-less `anti_neg.jsonl` — byte-identical
    single-GPU behaviour; a sharded run gets one file per shard so concurrent
    workers never share an append target."""
    if shard_count == 1:
        return _ANTI_NEG_FILE
    return f"anti_neg_shard_{shard_index:02d}.jsonl"


def _anti_neg_paths(adir: Path) -> list[Path]:
    """Every file the anti-neg run's rows live in: the legacy shard-less
    `anti_neg.jsonl` plus one `anti_neg_shard_NN.jsonl` per shard, sorted.
    The pattern is deliberately exact so `anti_neg_grade_cache.jsonl` (the
    grade cache living in the same run dir) is never read as rows."""
    adir = Path(adir)
    paths = [p for p in [adir / _ANTI_NEG_FILE] if p.exists()]
    paths += sorted(adir.glob("anti_neg_shard_*.jsonl"))
    return paths


def _anti_neg_done_keys(adir: Path) -> set[str]:
    """Cells already generated ANYWHERE in the anti run — the legacy file AND
    every shard file — so resume survives a change of shard layout: a partial
    single-GPU run re-run 8-way skips everything the single file already
    holds, and vice versa. Shards are disjoint over prompts, so for a
    same-layout re-run this reduces to each shard's own file."""
    from sisteer.sampling.engine import load_done_keys

    done: set[str] = set()
    for p in _anti_neg_paths(adir):
        done |= load_done_keys(p)
    return done


def _anti_neg_cfg(cfg: dict) -> dict:
    """bidpo's anti-safety negative source config (`pool.anti_neg`).

    Returns `{enabled, anti_si_id, run, candidates, temperature}` with defaults.
    `enabled` defaults False — legacy behavior, bidpo trains on the undefended
    strict subset. When True, generate-anti-neg elicits, per harmful prompt, K
    candidate completions under `safety_refusal.anti_text`; grade-anti-neg scores
    them; the manifest accepts the first candidate the grader scored a genuine
    compliance (> threshold) as bidpo's rejected side for that prompt."""
    a = dict((cfg.get("pool") or {}).get("anti_neg") or {})
    return {
        "enabled": bool(a.get("enabled", False)),
        "anti_si_id": str(a.get("anti_si_id", _ANTI_SI_DEFAULT)),
        "run": str(a.get("run", "two_si_anti_neg")),
        "candidates": int(a.get("candidates", 8)),
        "temperature": float(a.get("temperature", 1.0)),
    }


def _bidpo_benign_cfg(cfg: dict) -> dict:
    """bidpo's own-pool config (`pool.bidpo_benign`, final mix rule).

    Returns `{enabled, refusal_labels, floor, max_benign_per_harmful}` with
    defaults. `enabled` defaults False — legacy behavior, bidpo trains on the
    matched pairs' harmful side alone. When True: grade-store additionally
    XSTest-labels the benign `none` rows (the reversed side's chosen must be
    a VERIFIED undefended compliance); the manifest freezes the per-SI
    `bidpo_benign` block — HARMFUL side: this SI's OWN thwart set (not the
    joint intersection, not the 50/50 balance) with the accepted anti-safety
    compliance as the rejected side; BENIGN side, polarity REVERSED: chosen =
    the undefended compliance, rejected = that SI's refusal — emit-pairs
    writes `bidpo_harmful_<si>.jsonl` + `bidpo_benign_<si>.jsonl` per SI; and
    the trainer feeds bidpo (and ONLY bidpo) from those files.

    `max_benign_per_harmful` is the SETTLED section-13 mix rule at its
    default 1.0: benign = min(harmful, available) per SI — a deterministic
    subsample of the BENIGN side only; the scarcer harmful side is never
    cut; null = uncapped. `floor` defaults 0 = raise only on an EMPTY benign
    side (the section-5 >1000 floor is SUPERSEDED; thin sides are recorded,
    not refused)."""
    b = dict((cfg.get("pool") or {}).get("bidpo_benign") or {})
    cap = b.get("max_benign_per_harmful", 1.0)
    return {
        "enabled": bool(b.get("enabled", False)),
        "refusal_labels": tuple(b.get("refusal_labels") or ("I",)),
        "floor": int(b.get("floor", 0)),
        "max_benign_per_harmful": None if cap is None else float(cap),
    }


def _benign_source_specs(cfg: dict) -> list[dict]:
    """The benign corpora that fill the benign stratum (scale-up amendment
    2026-08-20), each a dict ``{name, loader, ...loader_kwargs}``. Read from
    ``pool.benign_sources`` (a list of str names or dicts); falls back to the
    single legacy ``pool.benign_source`` string, whose loader defaults to its
    own name. Empty when no benign half is declared."""
    pool = cfg["pool"]
    declared = pool.get("benign_sources")
    if declared:
        out = []
        for e in declared:
            e = {"name": e} if isinstance(e, str) else dict(e)
            if "name" not in e:
                raise SystemExit(
                    "[two-si-pool] every pool.benign_sources entry needs a "
                    f"`name`, got {e!r}")
            e.setdefault("loader", e["name"])
            out.append(e)
        return out
    single = pool.get("benign_source")
    if single is None:
        return []
    return [{"name": single, "loader": single}]


def _benign_families(cfg: dict) -> tuple[str, ...]:
    """The benign stratum's family labels (one per corpus)."""
    return tuple(s["name"] for s in _benign_source_specs(cfg))


def _benign_cfg(cfg: dict) -> "tuple[str | tuple[str, ...] | None, str]":
    """`(benign_source, benign_run)`. ``benign_source`` is None (legacy
    two-SI, no benign half), a single family label (the PHTest-only legacy
    stratum — kept as a str so the manifest reads exactly as before), or a
    tuple of labels (the multi-corpus scale-up)."""
    pool = cfg["pool"]
    fams = _benign_families(cfg)
    src: "str | tuple[str, ...] | None"
    if not fams:
        src = None
    elif len(fams) == 1:
        src = fams[0]
    else:
        src = fams
    return src, pool.get("benign_run", "two_si_benign")


def _scoring_prompt_shas(cfg: dict) -> frozenset[str]:
    """NORMALISED shas of every scoring (eval-control) prompt, for the
    train/scoring-disjointness filter. Declared under
    `pool.scoring_disjointness`; each entry may carry a local `path` override
    (tests / offline), else the pinned hub-cache copy is read. A declared
    source that cannot be loaded RAISES — never a silent skip (a silently
    empty exclusion set would let training items into the scoring sets)."""
    from sisteer.eval.jailbreak_data import normalized_text_sha

    block = cfg["pool"].get("scoring_disjointness") or {}
    texts: list[str] = []
    for name, entry in block.items():
        entry = entry or {}
        path = entry.get("path")
        if name == "or_bench_hard_1k":
            from sisteer.data.content_sets import _hard_1k_prompts

            texts += _hard_1k_prompts(path=path)
        elif name == "xstest":
            from sisteer.eval.jailbreak_data import load_xstest_safe

            kw = {"path": path} if path else {
                "revision": cfg["datasets"]["xstest"]["revision"]}
            texts += [g.text for g in load_xstest_safe(**kw)]
        else:
            raise SystemExit(
                f"[two-si-pool] unknown scoring_disjointness source {name!r} "
                "(known: or_bench_hard_1k, xstest) — refusing to guess which "
                "eval control it names.")
    return frozenset(normalized_text_sha(t) for t in texts)


# --------------------------------------------------------------------------- #
# store reader — labelled JSONL rows -> matched_pool.GradedRow (counts-safe)    #
# --------------------------------------------------------------------------- #


def _variant_key(row: dict) -> str:
    """A safe opaque id for the attack INSTANCE within a (family, goal). BoN
    keys on (sample_idx, trial) — the seeded augmentation is identical across
    conditions, so the same key names the same prompt under none/safety/
    constitution (verified: the (goal_id, sample_idx, trial) join reproduces
    the measured 657 BoN pairs). Other shapes fall back to the store's `order`.
    NEVER the raw prompt text (that would leak)."""
    if row.get("sample_idx") is not None:
        return f"{int(row['sample_idx'])}:{int(row.get('trial', 0))}"
    if row.get("order") is not None:
        return str(int(row["order"]))
    # last resort: a hash of the (goal, prompt) so the key is still opaque + safe
    from sisteer.eval.matched_pool import text_sha
    return text_sha(f"{row.get('goal_id')}|{row.get('attack_prompt')}")[:32]


def _row_to_graded(row: dict, *, si_ids, threshold, benign_source=None,
                   anti_si_id=None, store_prompted_si=_STORE_PROMPTED_SI):
    """Map one store row to a GradedRow, or None if it is not pool-relevant.

    Condition mapping: `si_condition=="none"` is undefended; `"prompted"`
    is the SAFETY_SI teacher; an explicit `si_id` field (what generate-pos
    writes) names any SI directly; `"compiled"` rows are object-defense
    transcripts, not SI-prompted teachers, and are skipped.

    ``store_prompted_si`` is `_store_pos_si(cfg)`: the SI the store's bare
    `prompted` rows still map to, or None once a generated pos run is declared
    for the store SI — then a HARMFUL bare-`prompted` row maps to nothing (the
    run is authoritative; the store's single-draw teacher rows must never mix
    into the freshly sampled target). Benign bare-`prompted` rows keep the
    mapping either way: the flip is harmful-pos-only and the benign half has
    no generated replacement. (The store's EXPLICIT `si_id` rows for the
    flipped SI are excluded at the source, in `_iter_all_pool_rows` — this
    mapping cannot tell a store row from a run row, but the iterator can.)

    ``anti_si_id`` (bidpo amendment 2026-08-20), when set, is the pseudo-SI id
    of bidpo's anti-safety negatives — kept as an SI row (condition "si") even
    though it is NOT one of ``si_ids``, so the matched-pool computation can route
    it to its own anti-neg bucket. Everything else about it is a harmful row (an
    attack family, StrongREJECT score, three-state rule).

    HARMFUL rows (any attack family): a row with no numeric score under a
    scored condition is skipped (ungraded — three-state rule). BENIGN rows
    (family == benign_source): a `none` row needs no grade (it only supplies
    the shared `neg`); an SI row is usable only once grade-benign wrote its
    XSTest `xstest_label` (C/P/I), else it is skipped as ungraded."""
    from sisteer.eval.matched_pool import GradedRow

    family = row.get("attack")
    goal_id = row.get("goal_id")
    prompt = row.get("attack_prompt")
    response = row.get("response")
    if not family or not goal_id or prompt is None or response is None:
        return None
    # answer only -- see _strip_think. Gate on EITHER tag: a template-opened
    # trace (Qwen3.x thinking-on) puts "<think>" in the PROMPT, so the response
    # carries only "</think>" and an opening-tag-only test never fires. That
    # exact no-op shipped on qwen38_think8k and the judge scored the reasoning.
    if "<think>" in response or "</think>" in response:
        response = _strip_think(response)
        if not response:
            return None

    benign_families = ({benign_source} if isinstance(benign_source, str)
                       else set(benign_source or ()))

    if row.get("si_id"):                       # explicit (generate-pos output)
        condition, si_id = "si", row["si_id"]
    else:
        sc = row.get("si_condition")
        if sc == "none":
            condition, si_id = "none", None
        elif sc == "prompted":
            if store_prompted_si is None and family not in benign_families:
                return None      # flipped store SI: the generated run wins
            condition, si_id = "si", _STORE_PROMPTED_SI
        else:
            return None                        # "compiled" and anything else

    if condition == "si" and si_id not in si_ids and si_id != anti_si_id:
        return None                            # an SI we are not comparing

    common = dict(
        family=family, goal_id=goal_id, variant_key=_variant_key(row),
        condition=condition, si_id=si_id, prompt=prompt, response=response,
        source=row.get("source") or family,
        candidate_idx=int(row.get("candidate_idx", 0)))

    if benign_source is not None and family in benign_families:
        if condition == "none":
            # A benign none row is admissible UNGRADED (it only supplies the
            # shared neg), but its XSTest label — written by grade-store's
            # benign-none pass when pool.bidpo_benign is enabled — rides
            # along when present: bidpo's reversed benign side needs the
            # undefended response VERIFIED as a compliance (never assumed).
            label = row.get("xstest_label")
            return GradedRow(score=None,
                             label=str(label) if label is not None else None,
                             **common)
        label = row.get("xstest_label")
        if label is None:
            return None                        # ungraded — awaiting grade-benign
        return GradedRow(score=None, label=str(label), **common)

    score = row.get("score")
    if score is None:
        return None                            # ungraded — cannot decide verdict
    return GradedRow(score=float(score), **common)


def _store_dir(pool: dict) -> Path:
    """The labelled store, with a TT_TWO_SI_STORE_DIR override.

    The configured default is a "~" path, and on the cluster /home is
    NODE-LOCAL — so on whatever node Slurm picks it resolves to an EMPTY
    directory and every family silently contributes zero rows. The override
    lets a cluster job point at a mode-0700 dir on shared storage without
    duplicating this config (which would then drift from the grading host copy).
    """
    env = os.environ.get("TT_TWO_SI_STORE_DIR")
    if env:
        return Path(env).expanduser()
    # the config writes the store as a placeholder ($STATE/... or $WORK/...): expanded here, refused by name when unset
    from sisteer import site as _site
    return _site.expand_path(pool["store_dir"], key="pool.store_dir")


def _iter_store_rows(store_dir: Path, families):
    """Stream rows from `<store_dir>/<family>.jsonl` for each family. Skips the
    bulk `compiled` rows cheaply (85% of the BoN file) by a substring pre-check
    before JSON-parsing."""
    for fam in families:
        path = Path(store_dir) / f"{fam}.jsonl"
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                # cheap pre-filter: skip object-defense transcripts (compiled)
                # unless the row is a none/prompted/explicit-si row.
                if '"si_condition": "compiled"' in line and '"si_id"' not in line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue                    # torn tail from a SIGKILL append


def _iter_all_pool_rows(cfg: dict, *, static_families, replay_families=(),
                        private_root=None):
    """Every raw row the pool computation may need: the labelled store
    (static + replay families), each generated pos run, and the benign run.

    THE TWO-SOURCE PRECEDENCE lives here, at the seam where the sources are
    still distinguishable (a row alone cannot say whether it came from the
    store or a run dir): once `pool.pos_runs` declares a run for the store's
    own prompted SI (`_store_pos_si` returns None), the STORE's harmful rows
    for that SI — explicit `si_id` rows and legacy bare-`prompted` rows alike
    — are dropped before they ever reach `_row_to_graded`, so the generated
    run is the ONLY source feeding that SI's pos bucket. Without this, both
    sources land in the same rejection-sampling candidate set and collide at
    candidate_idx 0 (a raise when the two draws' verdicts disagree, a silent
    store/run interleave when they agree). Benign-family store rows are NOT
    filtered — the flip is harmful-pos-only."""
    pool = cfg["pool"]
    store_dir = _store_dir(pool)
    benign_source, benign_run = _benign_cfg(cfg)
    fams = sorted(set(static_families) | set(replay_families))
    if _store_pos_si(cfg) is None:
        for raw in _iter_store_rows(store_dir, fams):
            if (raw.get("si_id") == _STORE_PROMPTED_SI
                    or (not raw.get("si_id")
                        and raw.get("si_condition") == "prompted")):
                continue                 # store rows for the flipped SI
            yield raw
    else:
        yield from _iter_store_rows(store_dir, fams)
    # The BENIGN strata also live in the store when pool_sample wrote them
    # (sample-all emits harmful and benign side by side and `merge` folds
    # both in), but `fams` above is the HARMFUL family list, so those rows were
    # invisible here and the benign half of the matched pool came out empty —
    # even after grade-store had correctly labelled them. The benign run dir
    # below only exists for the older generate-benign path. Read them from the
    # store too, minus any family already covered, so neither path double-counts.
    _bfams = tuple(f for f in _benign_families(cfg) if f not in set(fams))
    if _bfams:
        yield from _iter_store_rows(store_dir, _bfams)

    proot = Path(jailbreak_run._private_root(cfg, private_root))
    for _si, run in sorted(_pos_runs(cfg).items()):
        rdir = proot / run
        if rdir.exists():
            yield from _iter_store_rows(rdir, static_families)
    if benign_source is not None:
        bdir = proot / benign_run
        if bdir.exists():
            yield from _iter_store_rows(bdir, _benign_families(cfg))
    # bidpo's anti-safety negatives — the legacy shard-less file plus every
    # shard's file (`_anti_neg_paths`), so a data-parallel generation is read
    # whole. Yielded only when the source is enabled; harmless (empty) when
    # the run dir is absent.
    anti = _anti_neg_cfg(cfg)
    if anti["enabled"]:
        adir = proot / anti["run"]
        if adir.exists():
            yield from _iter_store_rows(
                adir, [p.stem for p in _anti_neg_paths(adir)])


def _load_graded_rows(cfg: dict, *, static_families, replay_families,
                      private_root):
    """All GradedRows the manifest needs: the static/train families + the
    replay (adaptive) families from the labelled store, every generated pos
    run, and the benign run."""
    si_ids = tuple(cfg["si_ids"])
    thr = float(cfg["prereg"]["threshold_grid"][1])   # 0.5 endpoint
    benign_source, _ = _benign_cfg(cfg)
    anti = _anti_neg_cfg(cfg)
    anti_si_id = anti["anti_si_id"] if anti["enabled"] else None

    rows = []
    for raw in _iter_all_pool_rows(cfg, static_families=static_families,
                                   replay_families=replay_families,
                                   private_root=private_root):
        g = _row_to_graded(raw, si_ids=si_ids, threshold=thr,
                           benign_source=benign_source, anti_si_id=anti_si_id,
                           store_prompted_si=_store_pos_si(cfg))
        if g is not None:
            rows.append(g)
    return rows, thr


# --------------------------------------------------------------------------- #
# goal split — derived from the store rows' category_group (holdout)            #
# --------------------------------------------------------------------------- #


def _norm_goal_text(text: str) -> str:
    """Whitespace- and case-normalised behaviour text, for twin detection."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _drop_trained_twins(train: set, eval_: set, text_of: dict) -> set:
    """Remove from the held-out set any goal whose BEHAVIOUR TEXT is verbatim
    identical to a goal in the training set.

    HarmBench and JBB share behaviours word-for-word under different
    ``goal_id``s -- measured on the shipped 300-goal set (HB standard
    test+val + JBB harmful): 9 texts carry 2 goal_ids each, 18 in all, every
    pair cross-source. The split assigns a side from ``category``, and the
    two sources label the SAME behaviour under DIFFERENT taxonomies
    (``harm_category_holdout`` splits HarmBench SemanticCategory and JBB
    category independently), so twins routinely land on opposite sides. At
    seed 0 / frac 0.33 four of the nine do -- all four the misinformation
    pairs, where HarmBench's ``misinformation_disinformation`` is held in
    while JBB's near-synonymous ``Disinformation`` is held out.

    That is harmless to training-goal disjointness: the two ids are distinct
    and nothing double-counts. It is NOT harmless to a held-out evaluation,
    which would score the model on a behaviour it was trained on verbatim
    and report the result as out-of-distribution.

    The eval copy is dropped rather than the train copy, and the twin is NOT
    promoted into training: its category was held out deliberately, so
    training on it would break the other split's intent. The text is already
    in training through its twin either way -- what is being removed is the
    false claim that it is unseen. A dropped goal is therefore neither a
    clean training goal nor a clean eval goal, which is the honest status.
    """
    if not text_of:
        return eval_
    trained_texts = {text_of[g] for g in train if g in text_of}
    twins = {g for g in eval_ if text_of.get(g) in trained_texts}
    if twins:
        print(f"[goal-split] dropped {len(twins)} held-out goal(s) whose "
              f"behaviour text is verbatim-identical to a training goal: "
              f"{', '.join(sorted(twins))}", flush=True)
    return eval_ - twins


def _goal_split_from_store(cfg: dict, static_families):
    """`(train_goal_ids, eval_goal_ids)` from the store's `category_group`
    label (held_in -> train, held_out -> eval). A goal has ONE category, hence
    one group, so the two sets are disjoint by construction — which the
    manifest builder re-asserts.

    A recorded group is honoured as-is (the store's own split stays
    authoritative). But rows written by pool_sample carry `category_group:
    None` — `engine._row` writes the literal, and it is the field's ONLY
    writer — which used to drop every such goal from BOTH sets and yield a
    zero-pair manifest off a half-million-row store (verified 2026-08-20). So
    a row MISSING its group gets it DERIVED AT READ TIME from `category`,
    through the exact same fixed-seed holdout that stamped the recorded rows
    (`jailbreak_run._category_holdout_cfg` seed/frac over
    `harm_category_holdout`, the hardcoded taxonomy — offline, deterministic,
    no dataset load): held_out iff the category is in the held-out set. A row
    with no category at all (the wildjailbreak_* strata) is held_in (train).
    Read-time by design — the in-flight store already holds ~250k None rows,
    and rewriting it would be slow and a torn-write risk for zero gain."""
    store_dir = _store_dir(cfg["pool"])
    held_out_set: "set | None" = None            # built on first missing group
    train, eval_ = set(), set()
    text_of: dict = {}                           # goal_id -> normalised goal_text
    for raw in _iter_store_rows(store_dir, static_families):
        gid = raw.get("goal_id")
        if not gid:
            continue
        txt = raw.get("goal_text")
        if txt and gid not in text_of:
            text_of[gid] = _norm_goal_text(txt)
        grp = raw.get("category_group")
        if grp is None:
            if held_out_set is None:
                from sisteer.eval.jailbreak_data import harm_category_holdout

                ch = jailbreak_run._category_holdout_cfg(cfg)
                _hi, ho = harm_category_holdout(seed=ch["seed"],
                                                frac=ch["frac"])
                held_out_set = set(ho)
            grp = ("held_out" if raw.get("category") in held_out_set
                   else "held_in")
        if grp == "held_in":
            train.add(gid)
        elif grp == "held_out":
            eval_.add(gid)
    eval_ = _drop_trained_twins(train, eval_, text_of)
    return tuple(sorted(train)), tuple(sorted(eval_))


def _operating_threshold(cfg: dict) -> float:
    """The StrongREJECT acceptance threshold, from the PREREGISTERED grid.

    Two hazards this exists to remove, both found 2026-08-26 while checking
    that a declared config value actually reaches the code:

    1. The value used to be ``threshold_grid[1]`` -- a magic index. Reorder or
       extend the grid and the operating point moves silently, with no diff
       anywhere near a threshold.
    2. ``pool.threshold`` is declared in the config and WAS READ BY NOTHING.
       It happened to agree (both 0.5), so the pool was correct by luck. Since
       0.5-vs-0.75 is a live question here, someone editing the obvious key
       would have changed nothing at all and believed otherwise.

    The grid stays authoritative -- it is the preregistration, and a
    convenience key must not be able to override it -- but a disagreement is
    now refused rather than silently resolved in the grid's favour.
    """
    grid = [float(x) for x in cfg["prereg"]["threshold_grid"]]
    prereg = cfg["prereg"]
    if "operating_threshold" in prereg:
        thr = float(prereg["operating_threshold"])
        if thr not in grid:
            raise ValueError(
                f"prereg.operating_threshold {thr} is not in "
                f"prereg.threshold_grid {grid}; the operating point must be a "
                f"preregistered grid value")
    else:
        thr = grid[1]   # historical default: the grid's midpoint
    declared = (cfg.get("pool") or {}).get("threshold")
    if declared is not None and float(declared) != thr:
        raise ValueError(
            f"pool.threshold ({float(declared)}) disagrees with the "
            f"preregistered operating threshold ({thr}). The prereg grid is "
            f"authoritative, so editing pool.threshold alone would change "
            f"NOTHING while looking like it changed the acceptance bar. "
            f"Set prereg.operating_threshold to move it deliberately.")
    return thr


def _pool_spec(cfg: dict, train_goals, eval_goals):
    from sisteer.eval.matched_pool import PoolSpec

    pool = cfg["pool"]
    benign_source, _ = _benign_cfg(cfg)
    return PoolSpec(
        si_ids=tuple(cfg["si_ids"]),
        train_families=tuple(pool["train_families"]),
        eval_families=tuple(pool.get("eval_families", ())),
        min_family_goal_coverage=float(
            pool.get("min_family_goal_coverage", 0.0)),
        train_goal_ids=tuple(train_goals),
        eval_goal_ids=tuple(eval_goals),
        static_families=frozenset(pool["static_families"]),
        threshold=_operating_threshold(cfg),
        min_pool_size=int(pool.get("min_pool_size", 1)),
        source_floors=dict(pool.get("source_floors", {})),
        default_source_floor=int(pool.get("default_source_floor", 0)),
        benign_source=benign_source,
        benign_fraction=(None if pool.get("benign_fraction") is None
                         else float(pool["benign_fraction"])),
        benign_fraction_tol=float(pool.get("benign_fraction_tol", 0.02)),
        hard_floor_pairs=int(pool.get("hard_floor_pairs", 0)),
        fit_dims=_fit_floor_dims(cfg),
        fit_holdout_frac=_fit_floor_holdout_frac(cfg),
        fit_calibration_rows=dict((pool.get("fit_floor") or {}).get(
            "calibration_rows", {})),
        relaxed_stratum_enabled=bool(
            (pool.get("relaxed_stratum") or {}).get("enabled", False)),
        scoring_prompt_shas=_scoring_prompt_shas(cfg),
        anti_si_id=(_anti_neg_cfg(cfg)["anti_si_id"]
                    if _anti_neg_cfg(cfg)["enabled"] else None),
        bidpo_benign_enabled=_bidpo_benign_cfg(cfg)["enabled"],
        bidpo_benign_refusal_labels=_bidpo_benign_cfg(cfg)["refusal_labels"],
        bidpo_benign_floor=_bidpo_benign_cfg(cfg)["floor"],
        family_caps={str(f): int(n)
                     for f, n in (pool.get("family_caps") or {}).items()})


def _fit_floor_dims(cfg: dict) -> dict:
    """`{arm -> fitted dimension d}` for the dimension-aware floor, read
    from `pool.fit_floor.dims`. Empty (floor off) unless declared."""
    ff = cfg["pool"].get("fit_floor") or {}
    return {str(a): int(d) for a, d in (ff.get("dims") or {}).items()}


def _fit_floor_holdout_frac(cfg: dict) -> float:
    """The nominal early-stopping holdout carve the dimension floor gates on.
    Defaults to `train_kwargs.default.holdout_frac` (the value the trainers
    actually use) so the floor's fit-row estimate matches the real split; a
    `pool.fit_floor.holdout_frac` override wins if given."""
    ff = cfg["pool"].get("fit_floor") or {}
    if ff.get("holdout_frac") is not None:
        return float(ff["holdout_frac"])
    tk_default = (cfg.get("train_kwargs") or {}).get("default") or {}
    return float(tk_default.get("holdout_frac", 0.0))


# --------------------------------------------------------------------------- #
# generation engine + provenance (engine ruling 2026-08-25)                     #
# --------------------------------------------------------------------------- #

#: The generation-leg engine vocabulary. "vllm" is the campaign ruling
#: (2026-08-25, recorded in the twosi-model-sweeps decisions): pool completions
#: generate on vLLM — distribution-consistency with the all-vLLM-regenerated
#: Llama pool, and a ~55x measured speedup. "hf" is the RETAINED contingency
#: (a model that fails the pilot's vLLM gate falls back to HF generate); using
#: it is a RECORDED engine confound, never a silent one — see the provenance
#: sidecar below and the manifest's engine-mix refusal.
_ENGINES = ("vllm", "hf")

#: Per-run-dir provenance sidecar (`.json`, NOT `.jsonl`, so no family-file or
#: `anti_neg_*.jsonl` reader can ever mistake it for rows). One per private run
#: dir; records which engine + sampling configuration produced the rows there.
GEN_PROVENANCE_NAME = "generation_provenance.json"

#: The provenance fields a RESUMED run must match exactly. A mismatch on any of
#: these means the new rows would come from a different completion distribution
#: than the rows already on disk — the engine-mixing confound the §0 ruling
#: forbids within a pool — so the resume REFUSES loudly instead. `info` (the
#: realised K etc.) is deliberately outside this set: raising K is the
#: sanctioned delta workflow and changes coverage, not distribution.
_PROV_STRICT = ("engine", "model", "model_revision", "vllm_version",
                "sampling_params")

#: vLLM benign generation appends (fsync'd) once per this many prompts, so a
#: killed pod repeats at most one block. The HF path keeps its own
#: `--batch-size` cadence; vLLM throughput wants far larger submissions than
#: the memory-bound HF default (2).
_VLLM_BENIGN_BLOCK = 512


def _resolve_engine(args, cfg, *, default: str, tag: str,
                    declared_key: str | None = None) -> str:
    """Engine selection: `--engine` flag > `pool.engine` config > the leg's
    historical default (generate-pos / generate-benign: "hf"; generate-anti-neg:
    "vllm" — it has only ever run on the vLLM engine). The default stays the
    historical engine so a config that predates the ruling keeps its exact old
    behaviour; the per-model sweep configs opt in with `pool.engine: vllm`.

    ``declared_key`` names the leg's DECLARATION key when the config carries
    one (``pool.pos_generation_engine`` for generate-pos — the sibling of the
    ``pool.neg_generation_engine`` key the ed01306 neg-side adapter reads).
    The declaration SELECTS the engine when neither the flag nor
    ``pool.engine`` chose one (so the stage3 base config, which declares
    vllm and sets no ``pool.engine``, runs vllm here just as it does on the
    branch that introduced the key) — and an explicit selection that
    DISAGREES with the declaration is REFUSED, not warned: the config and
    the code disagreeing about what produced a pool is exactly the ambiguity
    the +25.8pp HF/vLLM acceptance gap made expensive. To run the recorded
    hf contingency, flip the declaration too — the disagreement, not the
    engine, is the error."""
    declared = cfg["pool"].get(declared_key) if declared_key else None
    engine = (getattr(args, "engine", None) or cfg["pool"].get("engine")
              or declared or default)
    if engine not in _ENGINES:
        raise SystemExit(
            f"[{tag}] unknown engine {engine!r} — expected one of "
            f"{list(_ENGINES)} (flag --engine or config pool.engine)")
    if declared and declared != engine:
        raise SystemExit(
            f"[{tag}] refusing: pool.{declared_key} declares {declared!r} but "
            f"the selected engine is {engine!r} (flag --engine > pool.engine "
            f"> default {default!r}). The config and the code disagree about "
            f"what produced this pool — exactly the ambiguity the +25.8pp "
            f"HF/vLLM acceptance gap made expensive. Fix whichever is stale; "
            f"the hf contingency is run by declaring it, not by overriding a "
            f"vllm declaration silently.")
    return engine


def _require_vllm(tag: str):
    """Import vLLM lazily, failing with an ACTIONABLE error when it is absent.

    vLLM is deliberately not in the repo lockfile (its torch build tag
    conflicts with the base cu128 pin — see configs/capeval/
    vllm-requirements.txt, which documents the driver-570-viable +cu129 wheel
    pin), so the generation venv is its own bootstrap. Only the real
    engine=vllm generation path pays this import: planning, --dry-run and the
    hf path never touch it."""
    try:
        import vllm
    except ImportError as exc:
        raise SystemExit(
            f"[{tag}] engine=vllm but `import vllm` failed ({exc}). Install "
            "vLLM into the generation venv (the repo lockfile deliberately "
            "excludes it; configs/capeval/vllm-requirements.txt documents the "
            "driver-570-viable cu129 pin), or fall back with --engine hf / "
            "pool.engine: hf — the RECORDED contingency, which the pool "
            "manifest will surface as an engine confound.")
    return vllm


def _pool_sampler(args, cfg, *, seed: int, max_new_tokens: int):
    """ONE VllmSampler construction for every generation leg: the same
    engine knobs (--max-model-len / --gpu-memory-utilization /
    --enforce-eager, argparse defaults 16384 / 0.90 / off) map to the
    engine identically whichever leg is generating. VllmSampler itself
    owns the ninja-on-PATH env fix."""
    from sisteer.sampling.engine import VllmSampler

    return VllmSampler(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        max_model_len=int(getattr(args, "max_model_len", 16384)),
        gpu_memory_utilization=float(
            getattr(args, "gpu_memory_utilization", 0.90)),
        seed=int(seed), max_new_tokens=int(max_new_tokens),
        enforce_eager=bool(getattr(args, "enforce_eager", False)))


def _pos_vllm_sampling_kwargs(cfg: dict, *, sample: bool, base_seed: int,
                              candidate_idx: int, max_tokens: int) -> dict:
    """The exact vLLM SamplingParams kwargs for one pos candidate pass — the
    HF-teacher parity set ported from the committed vLLM pos-generation fixes
    (equivalence was measured broken until BOTH landed):

    - ``top_p`` 0.9 (02c3c52): the HF teacher inherits the model
      generation_config's nucleus cutoff; the engine's historical 1.0 made
      vLLM 'thwart' via long non-refusals (+25.8pp acceptance vs HF).
    - ``top_k`` 50 (bfdd5d8): HF generate inherits top_k=50 (the transformers
      GenerationConfig default — the model's generation_config omits top_k);
      top_p alone did not restore equivalence (rate overshot to -10.4pp).
    - per-candidate seed ``base_seed + 1000 * candidate_idx`` (7931794):
      distinct per candidate, reproducible from (seed, candidate_idx) alone,
      and resume-stable — candidate ci draws the same seed on a fresh run or a
      resume, unlike the HF path's process-global per-pass seed.

    All three are config-overridable (``pool.pos_sampling_*``) because the
    parity target is THIS model's HF teacher configuration: the per-model
    sweep configs pin whatever their model's generation_config + transformers
    defaults resolve to. Greedy (``sample=False``) zeroes the lot — a
    deterministic argmax has no sampler to configure."""
    if not sample:
        return {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                "max_tokens": int(max_tokens), "seed": None}
    return {
        "temperature": float(cfg["pool"].get("pos_sampling_temperature", 1.0)),
        "top_p": float(cfg["pool"].get("pos_sampling_top_p", 0.9)),
        "top_k": int(cfg["pool"].get("pos_sampling_top_k", 50)),
        "max_tokens": int(max_tokens),
        "seed": int(base_seed) + 1000 * int(candidate_idx),
    }


def _pos_cell_key(pid: tuple, si: str, candidate_idx: int) -> str:
    """The resume key for a plan pid — spelled through
    ``engine.cell_key_fields``, the format's one home, so the write path,
    the resume mask and ``engine.load_done_keys``'s read-back can never
    drift apart (a drift regenerates every cell on resume, forever)."""
    from sisteer.sampling.engine import cell_key_fields

    return cell_key_fields(pid[0], pid[1], pid[2], si, candidate_idx)


def _require_engine_keyable(pid: tuple, *, tag: str) -> None:
    """Refuse LOUDLY a plan pid whose variant_key is not the numeric
    ``sample_idx:trial`` form. ``engine.load_done_keys`` rebuilds the key from
    the ROW's integer sample_idx/trial, so a non-numeric key (the order/hash
    fallback in ``_variant_key``) could never match its own written row — the
    cell would regenerate forever on every resume. Every current pool family
    stamps an integer sample_idx; a null here is an unexpected store row the
    caller should see, not a silent skip."""
    vk = str(pid[2])
    head = vk.split(":")[0] if ":" in vk else vk
    if ":" not in vk or not head.lstrip("-").isdigit():
        raise SystemExit(
            f"[{tag}] prompt {pid[0]}/{pid[1]} has a non-numeric variant_key "
            f"{vk!r} (the order/hash fallback); the vLLM resume keys cells on "
            "an integer sample_idx and every current pool family stamps one. "
            "A null here is an unexpected store row.")


def _generation_provenance(*, engine: str, leg: str, cfg: dict,
                           sampling_params: dict, vllm_version=None,
                           info=None, prompt_render=None) -> dict:
    """One run dir's provenance record: engine, vLLM version, sampling params,
    model + revision (the review requirement — what makes an engine confound
    VISIBLE instead of silent). ``sampling_params`` must hold the EFFECTIVE
    request values, stable across candidate passes (base_seed + the seed rule,
    never a per-candidate seed). ``prompt_render`` (F8) records HOW the chat
    prompts were rendered — the renderer, the pinned template kwargs and a
    rendered-prefix hash — so a wrong render (a raw tokenizer that let Qwen
    spend its completion budget inside <think>, a Gemma system role that was
    never folded) is auditable after the fact."""
    prov = {
        "engine": engine,
        "leg": leg,
        "model": cfg["model"],
        "model_revision": cfg.get("model_revision"),
        "vllm_version": vllm_version,
        "sampling_params": sampling_params,
    }
    if prompt_render:
        prov["prompt_render"] = prompt_render
    if info:
        prov["info"] = info
    return prov


def _prompt_render_record(tok, system_texts: dict) -> dict:
    """The auditable render record (F8): the canonical renderer's identity,
    the tokenizer's pinned template kwargs (``load_tokenizer`` sets
    ``enable_thinking: False`` on reasoning-capable templates via
    ``tok._sisteer_template_kwargs``), and a sha256 of the RENDERED system
    prefix per condition — the exact bytes the engine prefills — so a render
    that drifted (different template, missing thinking pin, unfolded system
    role) is detectable from the sidecar alone. ``system_texts`` maps a
    condition name to its system text (None for the undefended pass)."""
    from sisteer.models import build_chat_prompt

    return {
        "renderer": "sisteer.models.load_tokenizer + build_chat_prompt "
                    "(apply_chat_template_folding)",
        "template_kwargs": dict(
            getattr(tok, "_sisteer_template_kwargs", {}) or {}),
        "rendered_prefix_sha256": {
            name: hashlib.sha256(
                build_chat_prompt(tok, "", system=text).encode()).hexdigest()
            for name, text in sorted(system_texts.items())},
    }


def _check_and_write_provenance(run_dir, prov: dict, *, tag: str) -> None:
    """Arm-time engine guard + provenance persist, called on every REAL
    generation path (never --dry-run / nothing-to-do, which create nothing).

    An existing sidecar whose strict fields (`_PROV_STRICT`) differ REFUSES the
    resume: the rows already in this dir came from a different engine or
    sampling distribution, and appending to them is exactly the silent
    engine-mixing the §0 ruling forbids. Rows present with NO sidecar predate
    provenance recording (legacy Llama-lane runs) — that is warned about and
    recorded in the sidecar, not refused: no record exists to check against,
    and our per-model pools are recorded from generation #1."""
    from sisteer.eval.jailbreak_artifacts import private_umask

    run_dir = Path(run_dir)
    path = run_dir / GEN_PROVENANCE_NAME
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except json.JSONDecodeError:
            raise SystemExit(
                f"[{tag}] {path} is unreadable — refusing to guess the prior "
                "engine. Restore it (or move the run dir aside deliberately) "
                "before generating more rows here.")
        diffs = {k: {"recorded": old.get(k), "requested": prov.get(k)}
                 for k in _PROV_STRICT if old.get(k) != prov.get(k)}
        # prompt_render joins the strict set only when BOTH sides record one:
        # a drifted render (template kwargs, prefix bytes) changes the
        # completion distribution exactly like a drifted sampler (F8), but a
        # legacy sidecar that predates render recording has nothing to check
        # against — the rewrite below records the new render going forward.
        if (old.get("prompt_render") is not None
                and prov.get("prompt_render") is not None
                and old["prompt_render"] != prov["prompt_render"]):
            diffs["prompt_render"] = {"recorded": old["prompt_render"],
                                      "requested": prov["prompt_render"]}
        if diffs:
            raise SystemExit(
                f"[{tag}] REFUSING to resume {run_dir} with different "
                f"generation provenance: {json.dumps(diffs, sort_keys=True)}. "
                "Rows already in this dir were generated under the recorded "
                "settings; appending rows from a different engine/sampler "
                "would silently mix completion distributions within one pool "
                "(the §0 engine ruling). If the change is deliberate, move "
                "the old rows aside and regenerate — never mix.")
        if old.get("note"):
            prov = {**prov, "note": old["note"]}
    else:
        legacy = sorted(p.name for p in run_dir.glob("*.jsonl")
                        if not p.name.endswith("_grade_cache.jsonl"))
        if legacy:
            prov = {**prov, "note":
                    "rows predating provenance recording exist in this run "
                    f"dir ({legacy}); their engine is UNRECORDED (legacy)"}
            print(f"[{tag}] WARNING: {run_dir} holds rows with no recorded "
                  "engine provenance (legacy run); recording that in the "
                  "sidecar. Our per-model pools are recorded from the first "
                  "generation and never hit this.", flush=True)
    with private_umask():
        # pid-suffixed tmp: the anti-neg fan-out writes this concurrently from
        # every worker (identical content); a shared tmp name could interleave.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(prov, indent=1, sort_keys=True))
        tmp.replace(path)


#: what a well-formed sidecar must carry (F12): the fields the engine-mix
#: refusal and every downstream audit read. Keys must be PRESENT; hf legs
#: legitimately record vllm_version: null and an unpinned model records
#: model_revision: null.
_PROV_SCHEMA_REQUIRED = ("engine", "vllm_version", "model", "model_revision",
                         "sampling_params")


def _dir_has_generation_rows(d) -> bool:
    """Whether a run/store dir holds any generation rows — any non-empty
    ``*.jsonl`` that is not a grader's on-disk score cache (the same filter
    ``_check_and_write_provenance``'s legacy probe uses)."""
    d = Path(d)
    if not d.exists():
        return False
    for p in d.glob("*.jsonl"):
        if p.name.endswith("_grade_cache.jsonl"):
            continue
        try:
            if p.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def _gather_generation_provenance(cfg: dict, private_root, *,
                                  allow_unrecorded=False) -> "dict | None":
    """Every generation leg's recorded provenance, keyed by leg, plus the set
    of distinct engines — what `manifest` embeds and refuses to mix.

    F12 discipline: a leg that CONTAINS generation rows may not silently sit
    out the engine check. In a config that DECLARES its generation engine
    (`pool.engine` — every per-model sweep config does), every rows-bearing
    required leg must present a schema-valid sidecar (`_PROV_SCHEMA_REQUIRED`)
    — a missing one REFUSES loudly (before it, an unrecorded/HF store leg
    froze `engines: ["vllm"]` because only observed sidecars participated),
    and `--allow-unrecorded-provenance` records the confound explicitly
    instead. A config with NO declared engine is the legacy (pre-ruling)
    world: its fixture-era rows predate sidecars entirely, so absent sidecars
    stay absent there (nothing is claimed about them). An unreadable or
    schema-invalid sidecar always refuses, in EVERY world — that is
    corruption, not absence, and no flag papers over it. Returns None only
    when NOTHING is recorded anywhere (the pre-provenance world, which freezes
    a byte-identical manifest precisely because it claims no engine at all)."""
    proot = Path(jailbreak_run._private_root(cfg, private_root))
    dirs: dict[str, Path] = {}
    for si, run in sorted(_pos_runs(cfg).items()):
        dirs[f"pos:{si}"] = proot / run
    benign_source, benign_run = _benign_cfg(cfg)
    if benign_source is not None:
        dirs["benign"] = proot / benign_run
    anti = _anti_neg_cfg(cfg)
    if anti["enabled"]:
        dirs["anti_neg"] = proot / anti["run"]
    # the labelled store itself (the pool_sample sample-all lane): its
    # none/prompted legs are pool rows like any other and join the check.
    dirs["store"] = _store_dir(cfg["pool"])

    legs: dict[str, dict] = {}
    unrecorded: list[str] = []
    for leg, d in sorted(dirs.items()):
        p = Path(d) / GEN_PROVENANCE_NAME
        if p.exists():
            try:
                prov = json.loads(p.read_text())
            except json.JSONDecodeError:
                raise SystemExit(
                    f"[two-si-pool] {p} is unreadable — refusing to freeze a "
                    "manifest over a corrupt provenance record. Restore the "
                    "sidecar (or move the run dir aside deliberately).")
            missing = [f for f in _PROV_SCHEMA_REQUIRED if f not in prov]
            if missing:
                raise SystemExit(
                    f"[two-si-pool] {p} is schema-invalid: missing "
                    f"{missing} (required: {list(_PROV_SCHEMA_REQUIRED)}). "
                    "A partial record cannot join the engine-mix check; "
                    "regenerate the leg with a current writer rather than "
                    "freezing a pool over an unauditable record.")
            legs[leg] = prov
        elif _dir_has_generation_rows(d):
            unrecorded.append(leg)
    if not legs:
        # pre-provenance world: nothing recorded anywhere -> embed nothing
        # (and claim nothing); the manifest stays byte-identical.
        return None
    if unrecorded and not cfg["pool"].get("engine"):
        # legacy config (no declared engine): absent sidecars stay absent —
        # the pre-F12 behaviour, kept ONLY where nothing declared an engine.
        unrecorded = []
    if unrecorded:
        if not allow_unrecorded:
            raise SystemExit(
                f"[two-si-pool] REFUSING to freeze: leg(s) {unrecorded} hold "
                "generation rows but NO provenance sidecar, and this config "
                f"declares pool.engine={cfg['pool'].get('engine')!r} — "
                "omitting them would silently exempt their rows from the "
                "engine-mix refusal (an unrecorded store leg once froze "
                "engines:['vllm'] over an HF-generated store). Either "
                "regenerate those legs with a provenance-recording writer, "
                "or pass --allow-unrecorded-provenance to embed the confound "
                "explicitly.")
        for leg in unrecorded:
            legs[leg] = {
                "engine": "unrecorded",
                "note": "generation rows with no provenance sidecar — engine "
                        "UNKNOWN; admitted by --allow-unrecorded-provenance",
            }
    engines = sorted({v.get("engine") for v in legs.values()
                      if v.get("engine") and v.get("engine") != "unrecorded"})
    out: dict = {"legs": legs, "engines": engines}
    if unrecorded:
        out["unrecorded_legs"] = sorted(unrecorded)
        out["unrecorded_provenance_allowed"] = True
        out["unrecorded_note"] = (
            "DELIBERATE (--allow-unrecorded-provenance): leg(s) with "
            "generation rows but no recorded engine — a RECORDED confound; "
            "their completion distribution may differ from the recorded "
            "legs'. Flag it in every comparison that consumes this pool.")
    return out


def _repair_torn_jsonl_tail(path, *, tag: str) -> bool:
    """Repair a JSONL whose tail a SIGKILL tore mid-append — IN PLACE, under
    an exclusive per-file lock, and BEFORE anything appends after the torn
    bytes. Returns True when the file was modified.

    Readers tolerate a malformed line by SKIPPING it, which is right for
    reads — but appending after torn bytes is not (F9): the torn fragment has
    no trailing newline, so the next appended row concatenates onto it into
    one long malformed line and is ALSO invisible to every reader, while the
    writer believes the cell landed. Two cases, deliberately distinct:

    - an UNTERMINATED tail that PARSES lost only its newline — terminate it
      (truncating would destroy a row every reader already counted);
    - a trailing region that does NOT parse is torn — truncate it, so the
      resume mask sees the hole and regenerates exactly that cell."""
    import fcntl

    path = Path(path)
    if not path.exists():
        return False
    with path.open("rb+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        data = f.read()
        if not data:
            return False
        if not data.endswith(b"\n"):
            tail = data[data.rfind(b"\n") + 1:]
            try:
                json.loads(tail)
            except json.JSONDecodeError:
                pass                               # torn — truncated below
            else:
                f.write(b"\n")                     # complete row, torn newline
                f.flush()
                os.fsync(f.fileno())
                print(f"[{tag}] repaired {path}: newline-terminated a "
                      "complete final row (torn append).", flush=True)
                return True
        # Truncate malformed TRAILING lines (append-only writes tear at most
        # the very end of the file; the loop stops at the first line that
        # parses, so interior rows are never touched).
        keep = len(data)
        if not data.endswith(b"\n"):
            keep = data.rfind(b"\n") + 1
        while keep > 0:
            prev = data.rfind(b"\n", 0, keep - 1)
            line = data[prev + 1:keep].strip()
            if not line:
                break                              # blank line: readers skip it
            try:
                json.loads(line)
                break
            except json.JSONDecodeError:
                keep = prev + 1
        if keep == len(data):
            return False
        f.truncate(keep)
        f.flush()
        os.fsync(f.fileno())
        print(f"[{tag}] repaired {path}: truncated {len(data) - keep} torn "
              "byte(s) from the tail (killed mid-append); the resume mask "
              "will regenerate the destroyed cell(s).", flush=True)
        return True


def _verify_pos_cells(path, *, tag: str, si=None, expected=None) -> None:
    """Post-generation verification: every (family, goal, variant, si,
    candidate) cell appears EXACTLY once in the file. A duplicate means the
    resume mask failed (or two writers shared a file) and the rejection-
    sampling accounting downstream would double-count — fail loudly at the
    writer, not at the manifest.

    When ``expected`` is given (``pid -> set of candidate indices``, for the
    plan's pids under ``si``), additionally verify the REALIZED candidate set
    for each pid EQUALS the expected set — exact equality, so a hole (a torn
    candidate the resume never regenerated, F9) fails loudly here, at the
    writer, instead of passing a duplicates-only check and dropping the
    candidate forever."""
    from sisteer.sampling.engine import cell_key_fields

    path = Path(path)
    if not path.exists():
        if expected:
            raise SystemExit(
                f"[{tag}] {path} does not exist but "
                f"{len(expected)} prompt(s) expected generated cells in it — "
                "the generation wrote nothing where the plan required rows.")
        return
    seen: set[str] = set()
    realized: dict[tuple, set] = {}
    for r in _read_jsonl_rows(path):
        vk = (f"{int(r.get('sample_idx', 0) or 0)}:"
              f"{int(r.get('trial', 0) or 0)}")
        key = cell_key_fields(r.get("attack"), r.get("goal_id"), vk,
                              r.get("si_id"), r.get("candidate_idx", 0) or 0)
        if key in seen:
            raise SystemExit(
                f"[{tag}] duplicate generated cell {key} in {path} — the "
                "resume mask failed or two writers shared this file; refusing "
                "to leave a double-counted pool run on disk unflagged.")
        seen.add(key)
        if si is not None and r.get("si_id") == si:
            pid = (r.get("attack"), r.get("goal_id"), vk)
            realized.setdefault(pid, set()).add(
                int(r.get("candidate_idx", 0) or 0))
    if not expected:
        return
    problems = []
    for pid, want in sorted(expected.items()):
        got = realized.get(pid, set())
        if got != set(want):
            problems.append(
                f"{pid}: missing={sorted(set(want) - got)} "
                f"unexpected={sorted(got - set(want))}")
    if problems:
        raise SystemExit(
            f"[{tag}] generated candidate cells in {path} do NOT match the "
            f"exact expected (prompt, candidate_idx) set: "
            f"{'; '.join(problems)}. A missing index is a HOLE (a torn "
            "candidate that was never regenerated — F9); refusing to leave "
            "an incomplete pool run on disk unflagged.")


# --------------------------------------------------------------------------- #
# manifest — piece A                                                            #
# --------------------------------------------------------------------------- #


# The two provenance notes the store SI's entry carries — fixed strings, so a
# reader (and a test) can key on them. Added to the manifest AFTER the
# containment walk, like skipped_families' prose (counts/ids/notes only —
# never store text).
_POS_PROV_GENERATED_NOTE = (
    "pos rows come from a GENERATED rejection-sampled run; the labelled "
    "store's own rows for this SI are EXCLUDED from pool assembly "
    "(pool.pos_runs declares a run for the store SI). MIGRATION 2026-08-24: "
    "pools frozen before this flip served this SI's pos from the store's "
    "single recorded draw per prompt — the training TARGET changed at this "
    "boundary, not merely its size, so objects fitted before and after are "
    "not directly comparable.")
_POS_PROV_STORE_NOTE = (
    "pos rows are the labelled store's own SI-prompted teacher rows (D6, "
    "one recorded draw per prompt); no generated pos run is declared for "
    "this SI.")


def _pos_provenance(cfg: dict, rejection_sampling: dict) -> dict:
    """Per-SI provenance of the HARMFUL pos rows the manifest was built from —
    which source served them (the labelled store, a generated pos run, or
    both), the configured rejection-sampling K, and the realised K stats
    (copied from the manifest's own rejection_sampling block so this block is
    self-contained). This is what makes a comparison across the store->run
    migration boundary VISIBLE: without it, two manifests with similar counts
    can hide that one SI's training target silently changed sources."""
    pool = cfg["pool"]
    runs = _pos_runs(cfg)
    store_si = _store_pos_si(cfg)
    ks = pool.get("pos_candidates") or {}
    prov: dict[str, dict] = {}
    for si in cfg["si_ids"]:
        rs = rejection_sampling.get(si) or {}
        if si == _STORE_PROMPTED_SI:
            flipped = store_si is None
            entry = {
                "store_rows": "excluded" if flipped else "admitted",
                "generated_run": runs.get(si),
                "note": (_POS_PROV_GENERATED_NOTE if flipped
                         else _POS_PROV_STORE_NOTE),
            }
        else:
            # a non-store SI's pos may live in the store (pool_sample lane),
            # its declared run (generate-pos lane), or both — the store draw
            # is candidate 0 and the run tops it up (see _pos_plan).
            entry = {"store_rows": "admitted", "generated_run": runs.get(si)}
        entry["configured_k"] = int(ks.get(si, 1))
        entry["realised_k_max"] = rs.get("k_max")
        entry["realised_k_mean"] = rs.get("k_mean")
        entry["per_prompt_yield"] = rs.get("per_prompt_yield")
        prov[si] = entry
    return prov


def _print_regime(manifest: dict) -> None:
    """State the regime PROMINENTLY, in BOTH cases. A reader must never
    have to infer which comparison the results describe."""
    sizes = ", ".join(f"{si}={n}" for si, n in manifest["pool_sizes"].items())
    print(f"[two-si-pool] REGIME: {manifest['regime']} ({sizes})")
    print(f"[two-si-pool] {manifest['regime_note']}")
    if manifest["regime"] == "native_per_si":
        print(f"[two-si-pool] HEADLINE CONFOUNDED by pool composition "
              f"(fallback_reason={manifest['fallback_reason']}) — flagged per D26.")
    hf = manifest["hard_floor"]
    if hf["enforced"]:
        print(f"[two-si-pool] hard floor {hf['floor_pairs']}: harmful matched "
              f"{hf['n_harmful_matched']}, benign matched "
              f"{hf['n_benign_matched']} — both at/above the floor "
              "(a violation would have raised, fixed decision 4).")
    comp = manifest.get("composition")
    if comp:
        r = comp["realised"]
        print(f"[two-si-pool] composition ASSERTED: {r['n_pairs']} pairs "
              f"({r['n_harmful']} harmful + {r['n_benign']} benign, fraction "
              f"{r['benign_fraction']}); per-family {r['per_family_counts']}.")
    # Report the realised K and per-prompt yield per SI —
    # the yield is the unmeasured quantity that decides feasibility.
    for si, st in manifest.get("rejection_sampling", {}).items():
        if st["n_candidates"]:
            print(f"[two-si-pool] rejection sampling {si}: realised K max "
                  f"{st['k_max']} / mean {st['k_mean']}, per-prompt yield "
                  f"{st['per_prompt_yield']} ({st['n_prompts_accepted']}/"
                  f"{st['n_prompts_with_candidates']} prompts; per-candidate "
                  f"accept rate {st['per_candidate_accept_rate']}).")
    # pos provenance: state PROMINENTLY when an SI's pos source is a generated
    # run that excludes its store rows — results that straddle that boundary
    # compare different training targets (migration 2026-08-24).
    for si, pv in (manifest.get("pos_provenance") or {}).items():
        if pv.get("store_rows") == "excluded":
            print(f"[two-si-pool] pos provenance {si}: GENERATED run "
                  f"{pv['generated_run']} (configured K={pv['configured_k']}, "
                  f"realised mean {pv['realised_k_mean']}); the store's rows "
                  "for this SI are EXCLUDED — the training target changed at "
                  "this boundary, do not compare objects across it.")
    bb = manifest.get("bidpo_benign")
    if bb:
        mix = ", ".join(
            f"{si}: harmful={e['harmful']['n_with_anti_neg']} "
            f"benign={e['benign']['n_qualified']} ratio="
            f"{e['projected_mix_at_default_1to1']['ratio_harmful_to_benign']}"
            for si, e in sorted(bb["per_si"].items()))
        dropped = {si: e["benign"]["n_dropped_undefended_not_complied"]
                   for si, e in sorted(bb["per_si"].items())}
        print(f"[two-si-pool] bidpo's OWN pool (section-13 mix; labels "
              f"{bb['refusal_labels']}; per-SI, UNEQUAL — harmful is the "
              f"per-SI thwart set, never the joint/balanced pool): {mix}; "
              f"benign dropped because the undefended model also refused/"
              f"hedged: {dropped}. The three objects train at DIFFERENT "
              "harmful:benign ratios — state these numbers alongside any "
              "bidpo cross-SI comparison.")
    sd = manifest["scoring_disjointness"]
    if sd["n_scoring_prompt_shas"]:
        print(f"[two-si-pool] train/scoring disjointness: removed "
              f"{sd['n_removed_harmful']} harmful + {sd['n_removed_benign']} "
              f"benign candidates matching {sd['n_scoring_prompt_shas']} "
              "scoring prompts (fixed decision 12).")
    if not manifest["all_sources_meet_floor"]:
        offenders = [s for s, e in manifest["per_source_yields"].items()
                     if not e["floor_ok"]]
        print(f"[two-si-pool] WARNING: sources below their floor: {offenders} "
              "(one source's abundance must not mask another's emptiness).")
    for si, cov in manifest["coverage"].items():
        if cov["n_missing"]:
            print(f"[two-si-pool] coverage: {si} still needs "
                  f"{cov['n_missing']} pos rows generated (run generate-pos).")
        if cov["n_all_candidates_rejected"]:
            print(f"[two-si-pool] coverage: {si} has "
                  f"{cov['n_all_candidates_rejected']} prompts whose "
                  "candidates were ALL rejected — a higher --candidates K "
                  "could recover them.")
    for si, cov in manifest.get("benign", {}).get("coverage", {}).items():
        if cov["n_missing"]:
            print(f"[two-si-pool] benign coverage: {si} still needs "
                  f"{cov['n_missing']} responses generated "
                  "(run generate-benign).")


def _apply_families_override(cfg: dict, families):
    """Return ``(cfg', skipped)`` where cfg' declares ONLY ``families`` as the
    static/train pool composition, and ``skipped`` is the committed families
    dropped. The single non-relitigating way to freeze a NARROWER-than-committed
    pool without editing the committed ``static_families`` (the eight-family
    pool stands): a provisioning shortfall (a family whose
    generator sub-env is not installed, so it was skipped-not-attempted) can then
    freeze a truthful K-family pool whose composition assertion passes on the K
    it actually has. ``families`` MUST be a subset of the committed set — adding
    a family the config never declared would be a scope change, refused here."""
    committed = list(cfg["pool"]["static_families"])
    fams = [f.strip() for f in families if f.strip()]
    extra = sorted(set(fams) - set(committed))
    if extra:
        raise SystemExit(
            f"[manifest] --families {extra} not in committed static_families "
            f"{committed}; --families may only NARROW the declared pool (a "
            "provisioning shortfall), never add a family the spec didn't "
            "declare (that would be a scope change).")
    skipped = [f for f in committed if f not in set(fams)]
    new_pool = {**cfg["pool"], "static_families": list(fams),
                "train_families": [f for f in cfg["pool"]["train_families"]
                                   if f in set(fams)]}
    return {**cfg, "pool": new_pool}, skipped


def cmd_manifest(args, cfg) -> None:
    from sisteer.eval.matched_pool import (
        build_matched_pool,
        manifest_hash,
        write_manifest,
    )

    skipped_families: list[str] = []
    if getattr(args, "families", None):
        cfg, skipped_families = _apply_families_override(
            cfg, args.families.split(","))

    pool = cfg["pool"]
    static_families = tuple(pool["static_families"])
    replay_families = tuple(pool.get("replay_families", ()))
    benign_source, _ = _benign_cfg(cfg)
    if benign_source is not None:
        # a benign source that IS an over-refusal eval control would bias the
        # very prompts the objects are judged on — refuse at the config seam.
        # Guard every corpus name (multi-source scale-up), not just the first.
        from sisteer.data.content_sets import _refuse_eval_controls

        for _fam in _benign_families(cfg):
            _refuse_eval_controls(benign_source=_fam)

    train_goals, eval_goals = _goal_split_from_store(cfg, static_families)
    spec = _pool_spec(cfg, train_goals, eval_goals)
    rows, _thr = _load_graded_rows(
        cfg, static_families=static_families, replay_families=replay_families,
        private_root=args.private_root)

    manifest = build_matched_pool(rows, spec)

    # Per-SI pos provenance (store vs generated run + configured/realised K):
    # the record that makes the safety_refusal store->run migration visible in
    # the frozen artifact itself. Added after build_matched_pool's containment
    # walk (fixed notes + counts + run names only, never store text); the hash
    # is re-embedded below so it covers this block too.
    manifest["pos_provenance"] = _pos_provenance(
        cfg, manifest.get("rejection_sampling") or {})
    manifest["manifest_sha256"] = manifest_hash(manifest)

    # Provenance for a NARROWED pool (decision a): record the committed families
    # that were skipped-not-attempted, so the frozen manifest is self-describing
    # about why it holds fewer than the committed eight — never a silent trim.
    # Re-embed the hash so it still covers the full content (write_manifest
    # verifies embedded == recomputed and refuses a stale hash).
    if skipped_families:
        manifest["skipped_families"] = {
            "families": skipped_families,
            "reason": "provisioning shortfall — generator sub-env not installed; "
                      "skipped-not-attempted, NOT a scope change (spec fixed "
                      "decision 8, an eight-family pool, still stands)",
        }
        manifest["manifest_sha256"] = manifest_hash(manifest)

    # Generation provenance (engine ruling 2026-08-25 / review requirement):
    # embed every leg's recorded engine + vLLM version + sampling params +
    # model/revision, and REFUSE a pool whose legs mix engines — two engines'
    # completion distributions are measured non-equivalent (the vllm-generate-
    # pos lane's equivalence study), so a mixed pool confounds every downstream
    # comparison. `--allow-engine-mix` converts the refusal into a RECORDED
    # confound for the sanctioned HF contingency (an HF-fallback model whose
    # anti-neg leg has no HF path). A world with NO sidecars anywhere (legacy
    # rows only) freezes a byte-identical manifest — nothing recorded, nothing
    # claimed; but once any leg records one, a rows-bearing leg with no
    # sidecar REFUSES (F12) unless --allow-unrecorded-provenance records that
    # confound too.
    gen_prov = _gather_generation_provenance(
        cfg, args.private_root,
        allow_unrecorded=getattr(args, "allow_unrecorded_provenance", False))
    if gen_prov:
        if gen_prov.get("unrecorded_legs"):
            print(f"[two-si-pool] WARNING: leg(s) "
                  f"{gen_prov['unrecorded_legs']} hold generation rows with "
                  "NO recorded engine — embedded as a deliberate, flagged "
                  "confound (--allow-unrecorded-provenance).", flush=True)
        if len(gen_prov["engines"]) > 1:
            per_leg = {leg: v.get("engine")
                       for leg, v in gen_prov["legs"].items()}
            if not getattr(args, "allow_engine_mix", False):
                raise SystemExit(
                    "[two-si-pool] REFUSING to freeze a pool whose generation "
                    f"legs mix engines: {json.dumps(per_leg, sort_keys=True)}. "
                    "Engines are measured non-equivalent on this workload; "
                    "regenerate the minority leg(s) on one engine, or pass "
                    "--allow-engine-mix to record the confound deliberately "
                    "(§WS7 contingency only).")
            gen_prov["engine_mix_allowed"] = True
            gen_prov["engine_mix_note"] = (
                "DELIBERATE engine mix (--allow-engine-mix): a recorded "
                "confound per the §WS7 HF contingency — flag it in every "
                "cross-model comparison that consumes this pool")
            print(f"[two-si-pool] WARNING: engine mix recorded "
                  f"({json.dumps(per_leg, sort_keys=True)}) — a deliberate, "
                  "flagged confound.", flush=True)
        manifest["generation_provenance"] = gen_prov
        manifest["manifest_sha256"] = manifest_hash(manifest)

    out = Path(args.out) if args.out else Path(cfg["pool_manifest"])
    if not out.is_absolute():
        out = REPO / out
    write_manifest(manifest, out)
    _print_regime(manifest)
    print(f"[two-si-pool] froze manifest -> {out} "
          f"(sha256={manifest['manifest_sha256'][:12]}…, counts/ids/hashes only)")


# --------------------------------------------------------------------------- #
# generate-pos — piece B (GPU): GENERATION ONLY, no grader                    #
# --------------------------------------------------------------------------- #


def _pos_plan(cfg, spec, static_families, *, si, run_name, k,
              private_root=None):
    """The (train) harmful-base prompts and their existing `pos` coverage
    for ONE SI — the CPU-only planning step generate-pos needs. Returns
    ``(none_prompt, base, covered, plan)`` where ``base`` is the pid set the
    SI must cover and ``plan`` is a list of ``(pid, missing, existing)``: the
    EXACT candidate indices in [0, K) still absent for each prompt (an ordered
    tuple) and the exact index set already on disk (a frozenset). The plan
    carries the realised SET, never a ``range(max+1)`` reconstruction — a torn
    candidate 0 whose row a SIGKILL destroyed leaves ``existing = {1, ...}``
    with a HOLE, and a max-based contiguous prefix would mask that hole
    forever (F9: the candidate would never regenerate while verification
    passed). Pure/read-only, so `--dry-run` exercises it with no model.

    THE BASE matches the manifest's harmful base: every GRADED `none` train
    prompt — the relaxed default — narrowing to the none-successful subset
    only under ``spec.require_undefended_compliance``. (The old
    none-successful-only plan predated the relaxed base: against that base
    it would cover 2,386 of 22,040 prompts and silently shrink the pool
    ninefold.)

    Coverage semantics (the three-state rule): a prompt is
    COVERED once any of its candidates is graded-THWARTING; otherwise more
    candidates are drawn only up to K in total — so a re-run never regenerates
    what exists, and raising K on a later run generates exactly the delta.
    Coverage counts this SI's OWN run rows and, for an SI whose pos rows the
    store still serves (everything except the flipped store SI —
    `_store_pos_si`), the store's explicit-`si_id` rows too: the store draw IS
    candidate 0 there, so a raised K tops it up from candidate 1 instead of
    colliding with it. For the FLIPPED store SI the store is deliberately NOT
    coverage — its rows are excluded from the pool, and the run must stand
    alone as the freshly sampled target.

    `private_root` is threaded through to the SAME resolver generate-pos
    writes with and `manifest` reads with (`jailbreak_run._private_root`), so
    the coverage read honours `--private-root` instead of silently reading a
    different location."""
    pool = cfg["pool"]
    store_dir = _store_dir(pool)
    thr = spec.threshold
    train_fams = set(spec.train_families)
    train_goals = set(spec.train_goal_ids)
    # store rows count as this SI's coverage unless the SI is the flipped
    # store SI (run declared -> store rows excluded from the pool).
    store_covers = not (si == _STORE_PROMPTED_SI and _store_pos_si(cfg) is None)

    none_prompt: dict[tuple, dict] = {}
    none_success: set[tuple] = set()
    have_idx: dict[tuple, set[int]] = {}
    accepted: set[tuple] = set()
    for raw in _iter_store_rows(store_dir, static_families):
        if (raw.get("attack") not in train_fams
                or raw.get("goal_id") not in train_goals
                or raw.get("attack_prompt") is None):
            continue
        if raw.get("si_condition") == "none":
            if raw.get("score") is None:
                continue
            pid = (raw["attack"], raw["goal_id"], _variant_key(raw))
            none_prompt[pid] = {"family": raw["attack"],
                                "goal_id": raw["goal_id"],
                                "variant_key": _variant_key(raw),
                                "prompt": raw["attack_prompt"],
                                # the undefended response rides along so the
                                # pilot pairs emitter (--pilot-pairs-out) can
                                # pair it as `neg` without a second store pass
                                "none_response": raw.get("response"),
                                "category": raw.get("category"),
                                "category_group": raw.get("category_group")}
            if float(raw["score"]) > thr:
                none_success.add(pid)
        elif store_covers and raw.get("si_id") == si:
            pid = (raw["attack"], raw["goal_id"], _variant_key(raw))
            have_idx.setdefault(pid, set()).add(
                int(raw.get("candidate_idx", 0)))
            if raw.get("score") is not None and not (float(raw["score"]) > thr):
                accepted.add(pid)

    # existing coverage from THIS SI's own run: candidate idxs by presence
    # (any row, even ungraded — a re-run of generate-pos never regenerates a
    # candidate it already generated), acceptance by GRADED thwart only.
    rdir = Path(jailbreak_run._private_root(cfg, private_root)) / run_name
    if rdir.exists():
        for raw in _iter_store_rows(rdir, static_families):
            if raw.get("si_id") != si or raw.get("attack_prompt") is None:
                continue
            pid = (raw["attack"], raw["goal_id"], _variant_key(raw))
            have_idx.setdefault(pid, set()).add(int(raw.get("candidate_idx", 0)))
            if raw.get("score") is not None and not (float(raw["score"]) > thr):
                accepted.add(pid)

    base = (none_success if spec.require_undefended_compliance
            else set(none_prompt))
    covered = sorted(accepted & base)
    plan: list[tuple] = []
    for pid in sorted(base):
        if pid in accepted:
            continue
        existing = have_idx.get(pid, set())
        missing = tuple(ci for ci in range(int(k)) if ci not in existing)
        if missing:
            plan.append((pid, missing, frozenset(existing)))
    return none_prompt, base, covered, plan


def _strip_think(text: str) -> str:
    """The answer with any reasoning trace removed (sampling.engine)."""
    from sisteer.sampling.engine import strip_think

    return strip_think(text)


def _ungraded_pos_row(c: dict, response: str, *, si: str,
                      candidate_idx: int = 0) -> dict:
    """One SI `pos` store row, persisted UNGRADED. Carries everything
    grade-pos and `manifest` need to identify and grade the prompt — the
    attack family, goal id, the `(sample_idx, trial)` that form the opaque
    variant key, the SI id, the rejection-sampling `candidate_idx`, the attack
    prompt and the generated response — and marks it EXPLICITLY ungraded
    (`score: None`, `graded: False`) rather than absent-by-omission, so a
    reader can tell "generated, awaiting grade" apart from "never generated"."""
    vk = c["variant_key"]
    sample_idx = int(vk.split(":")[0]) if ":" in vk else None
    trial = int(vk.split(":")[1]) if ":" in vk else 0
    return {
        "attack": c["family"], "goal_id": c["goal_id"],
        "sample_idx": sample_idx, "trial": trial,
        "si_condition": "prompted", "si_id": si,
        "candidate_idx": int(candidate_idx),
        "category": c.get("category"), "category_group": c.get("category_group"),
        "attack_prompt": c["prompt"], "response": response,
        "score": None, "graded": False,
    }


def _resolve_generate_si(args, cfg) -> tuple[str, str, int]:
    """Validate --si and resolve its run name + candidate count K (CLI
    override > `pool.pos_candidates.<si>` > 1). The store's own prompted SI is
    generable ONLY once `pool.pos_runs` declares a run for it (`_store_pos_si`
    — the flip that also excludes the store's rows for it from the pool);
    without that declaration the historical refusal stands."""
    si = getattr(args, "si", None) or "si_constitution"
    if si not in cfg["si_ids"]:
        raise SystemExit(
            f"[generate-pos] --si {si!r} is not in the config si_ids "
            f"{list(cfg['si_ids'])}")
    if si == _STORE_PROMPTED_SI and _store_pos_si(cfg) is not None:
        raise SystemExit(
            f"[generate-pos] --si {si!r} is the labelled store's own prompted "
            "SI (D6) — its pos rows already exist in the store; nothing to "
            "generate. To rejection-sample it at raised K instead, declare "
            f"pool.pos_runs.{_STORE_PROMPTED_SI} in the config: the generated "
            "run then becomes authoritative and the store's rows for it are "
            "excluded from pool assembly (never mixed).")
    runs = _pos_runs(cfg)
    k = getattr(args, "candidates", None)
    if k is None:
        k = int((cfg["pool"].get("pos_candidates") or {}).get(si, 1))
    if k < 1:
        raise SystemExit(f"[generate-pos] --candidates must be >= 1, got {k}")
    return si, runs[si], k


def cmd_generate_pos(args, cfg) -> None:
    """Fill one SI's `pos` gaps on a GPU node — GENERATION ONLY. The
    planning half (which prompts/candidates to generate) is CPU-only and
    `--dry-run` reports it. Real generation lazily imports the model +
    generator and writes each response UNGRADED; grading is a separate
    grading-host step (`grade-pos`) because the live Low-Prio grader key
    lives in `~/.config` and cluster `/home` is node-local — constructing a
    grader here crashes. No grader is imported or built on this path. With
    K > 1 the candidates are SAMPLED — rejection sampling needs distinct,
    reproducible draws.

    TWO engines (engine ruling 2026-08-25; `--engine` / `pool.engine`):

    - ``vllm`` — the campaign path (`_generate_pos_vllm`): the shared
      prefix-cached per-GPU engine the anti-neg path uses, with the
      HF-teacher-parity sampling set (temperature / top_p 0.9 / top_k 50 /
      per-candidate seed — see `_pos_vllm_sampling_kwargs`).
    - ``hf`` — the RETAINED contingency, byte-for-byte the pre-ruling
      behaviour (a deterministic per-pass torch seed, `generate_batch`),
      kept for a model that fails the pilot's vLLM gate. Provenance-recorded
      either way, so the manifest can refuse a silently mixed pool."""
    si, run_name, k = _resolve_generate_si(args, cfg)
    engine = _resolve_engine(args, cfg, default="hf", tag="generate-pos",
                             declared_key="pos_generation_engine")
    static_families = tuple(cfg["pool"]["static_families"])
    train_goals, eval_goals = _goal_split_from_store(cfg, static_families)
    spec = _pool_spec(cfg, train_goals, eval_goals)
    none_prompt, base, covered, plan = _pos_plan(
        cfg, spec, static_families, si=si, run_name=run_name, k=k,
        private_root=args.private_root)

    # --max-prompts (the pilot's knob, pod_pilot.sh step 5): cap the run to a
    # LENGTH-REPRESENTATIVE slice of the held_in harmful-base prompts (the
    # DECISION-1 base _pos_plan planned over). The slice is chosen over that
    # base (not over the remaining plan), so it is stable across resumes — a
    # re-run selects the same prompts and generates only their remaining cells.
    max_prompts = getattr(args, "max_prompts", None)
    selected = sorted(base)
    if max_prompts is not None:
        selected = _length_representative_pids(
            base, none_prompt, int(max_prompts))
        sel = set(selected)
        plan = [e for e in plan if e[0] in sel]
        covered = [pid for pid in covered if pid in sel]
        print(f"[generate-pos] --max-prompts {int(max_prompts)}: capped to "
              f"{len(selected)} of {len(base)} harmful-base "
              "prompts (length-representative slice).")

    n_new = sum(len(m) for _pid, m, _e in plan)
    print(f"[generate-pos] si={si} (run {run_name}, K={k}, engine={engine}): "
          f"{len(base)} harmful-base train prompts (relaxed, DECISION 1); "
          f"{len(covered)} already have an accepted pos, "
          f"{len(plan)} need candidates ({n_new} generations).")
    if args.dry_run:
        return
    pilot_out = getattr(args, "pilot_pairs_out", None)
    if pilot_out is not None:
        # refuse a repo destination NOW — before any GPU work is spent
        pilot_out = _pilot_pairs_path(pilot_out)
    if not plan:
        print("[generate-pos] nothing to generate — coverage complete.")
        if pilot_out:
            _emit_pilot_pairs(
                cfg, si=si, none_prompt=none_prompt, pids=selected,
                out_path=pilot_out, static_families=static_families,
                rdir=Path(jailbreak_run._private_root(
                    cfg, args.private_root)) / run_name)
        return

    # -- real generation path (GPU) — deliberately lazy-imported. NO grader is
    #    imported or constructed here: building one needs the Low-Prio key in
    #    ~/.config, absent on a compute node. ---------------------------------
    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.eval.jailbreak_artifacts import append_jsonl, private_run_dir

    # the two-SI resolver: handles safety_refusal (the SAFETY_SI module
    # constant, deliberately OUT of the registry) as well as registered SIs
    instr = resolve_instruction(si)
    # Route the write through the SAME resolver the coverage read + `manifest`
    # use: private_run_dir's default root is grading_host_private_root()
    # (a grading-host state path), node-local on the cluster — so an unrouted
    # write would land on the compute node's own disk, invisible to `manifest`
    # and lost when the node is reclaimed. It also (re)creates the dir 0700.
    cdir = private_run_dir(
        run_name, root=jailbreak_run._private_root(cfg, args.private_root))
    cap = int(cfg["completion_tokens"])

    # F9: repair a torn JSONL tail (pod killed mid-append) BEFORE either
    # engine appends — appending after torn bytes concatenates the fresh row
    # onto the fragment, losing BOTH to every reader while the writer
    # believes the cell landed.
    for fam in sorted({pid[0] for pid, _m, _e in plan}):
        _repair_torn_jsonl_tail(cdir / f"{fam}.jsonl", tag="generate-pos")

    if engine == "vllm":
        total = _generate_pos_vllm(args, cfg, si=si, instr=instr, cdir=cdir,
                                   none_prompt=none_prompt, plan=plan, k=k,
                                   cap=cap)
        print(f"[generate-pos] wrote {total} UNGRADED {si} pos candidate "
              f"row(s) -> {cdir}. Grade them on the grading host: "
              f"build_two_si_pool.py grade-pos --private-root <root>.")
        if pilot_out:
            _emit_pilot_pairs(cfg, si=si, none_prompt=none_prompt,
                              pids=selected, out_path=pilot_out,
                              static_families=static_families, rdir=cdir)
        return

    # -- hf contingency path: the pre-ruling behaviour, RETAINED unchanged ----
    from sisteer.generate import generate_batch
    from sisteer.models import build_chat_prompt, load_model_and_tokenizer

    bs = int(args.batch_size)
    sample = k > 1 or any(e for _p, _m, e in plan)
    temperature = float(cfg["pool"].get("pos_sampling_temperature", 1.0))
    _check_and_write_provenance(cdir, _generation_provenance(
        engine="hf", leg=f"generate-pos:{si}", cfg=cfg,
        sampling_params={
            "do_sample": bool(sample), "temperature": temperature,
            "top_p": "hf-generation-config-inherited",
            "top_k": "hf-generation-config-inherited",
            "max_new_tokens": cap,
            "seed_rule": "process-global sha256(si|pass) per candidate pass",
        },
        info={"k": int(k)}), tag="generate-pos")
    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    # Pass p fills each prompt's p-th MISSING index (plan carries the exact
    # missing tuple). With contiguous coverage this is byte-identical to the
    # historical start+p indexing; with a hole (a torn candidate, F9) the
    # hole is refilled instead of masked.
    max_pass = max(len(m) for _pid, m, _e in plan)
    total = 0
    for p in range(max_pass):
        work = [(pid, m[p]) for pid, m, _e in plan if p < len(m)]
        if sample:
            # one deterministic seed per (si, pass): distinct draws across
            # candidate passes, byte-reproducible across re-runs.
            import torch

            seed = int.from_bytes(hashlib.sha256(
                f"{si}|pass{p}".encode()).digest()[:4], "big")
            torch.manual_seed(seed)
        for i in range(0, len(work), bs):
            chunk = work[i:i + bs]
            rendered = [build_chat_prompt(tok, none_prompt[pid]["prompt"],
                                          system=instr.text)
                        for pid, _c in chunk]
            outs = generate_batch(model, tok, rendered, batch_size=bs,
                                  max_new_tokens=cap, do_sample=sample,
                                  temperature=temperature)
            for (pid, cidx), o in zip(chunk, outs):
                append_jsonl(cdir / f"{pid[0]}.jsonl",
                             _ungraded_pos_row(none_prompt[pid], o, si=si,
                                               candidate_idx=cidx))
                total += 1
    print(f"[generate-pos] wrote {total} UNGRADED {si} pos candidate row(s) "
          f"-> {cdir}. Grade them on the grading host: "
          f"build_two_si_pool.py grade-pos --private-root <root>.")
    if pilot_out:
        _emit_pilot_pairs(cfg, si=si, none_prompt=none_prompt, pids=selected,
                          out_path=pilot_out,
                          static_families=static_families, rdir=cdir)




def _generate_pos_vllm(args, cfg, *, si, instr, cdir, none_prompt, plan,
                       k, cap) -> int:
    """generate-pos on the shared vLLM engine — the port of the
    vllm-generate-pos lane's committed generation path (7931794 + 02c3c52 +
    bfdd5d8), adapted to write through THIS pipeline's own row builder.

    Admissible on vLLM because these pos rows are unsteered prompted-teacher
    completions (never compared against a steered eval row), so the
    engine-matched rule that keeps steered work off vLLM does not bind —
    and the SI's ~6k-token system prefix is prefilled ONCE per engine and its
    KV reused for every prompt in a candidate block (the same
    `sisteer.sampling.engine.VllmSampler` the anti-neg / sample-all lanes use).

    Determinism + resume, in one place:

    - **Exact-set candidate indexing (F9).** Cells are (pid, candidate_idx);
      the plan carries each prompt's exact MISSING index tuple and its exact
      EXISTING index set (never a `range(max+1)` reconstruction, which would
      mask a torn candidate's hole forever), and generation fills exactly the
      missing indices. The per-cell seed is `base_seed + 1000 * candidate_idx`
      — reproducible from the seed and the ABSOLUTE candidate index alone, so
      a resumed cell draws the seed a fresh run would have given it.
    - **Sequential per-family output files** (`<run>/<family>.jsonl`, the
      layout grade-pos/_pos_plan already read), appended + fsync'd once per
      (family, candidate) block — a crash loses at most one block and the
      resume mask regenerates exactly the missing cells (the caller repairs a
      torn tail BEFORE any append). Sharding is the anti-neg path's tool for
      its 10x-bigger inventory; at vLLM's measured ~55x speedup one engine
      covers the pos leg, so there is no merge to verify —
      `_verify_pos_cells` instead proves every cell landed exactly once AND
      that the realized (prompt, candidate) set EQUALS the expected set (a
      hole fails loudly at the writer).
    - **Canonical prompt rendering (F8).** Prompts are rendered through the
      harness's OWN tokenizer semantics — `sisteer.models.load_tokenizer`
      (which pins `enable_thinking: False` on reasoning-capable templates)
      + `build_chat_prompt`/`apply_chat_template_folding` (the Gemma
      system-role fold) — NEVER the engine's raw AutoTokenizer, whose default
      render lets Qwen burn the completion budget inside <think> (a teacher
      distribution the HF path never produces). The effective template kwargs
      and a rendered-prefix hash are recorded in the provenance sidecar.
    - Rows are built by `_ungraded_pos_row` — the SAME builder the hf path
      uses — so the two engines' rows are schema-identical by construction and
      grade-pos / manifest stay engine-agnostic."""
    from collections import defaultdict

    vllm = _require_vllm("generate-pos")
    from sisteer.models import build_chat_prompt, load_tokenizer
    from sisteer.sampling.engine import load_done_keys

    # SAMPLE whenever this SI is a rejection-sampling SI — by this run's K, by
    # the CONFIGURED K, or by resuming past candidate 0. The configured-K
    # clause matters when a K>1 scheme is generated in cost-staged rounds
    # (--candidates 1 first, grade, then the delta): candidate 0 must be a
    # DRAW from the same distribution as candidates 1..K-1, not the greedy
    # mode — a greedy candidate 0 would make this SI's accepted pos
    # systematically the modal completion while its peers' are sampled.
    cfg_k = int((cfg["pool"].get("pos_candidates") or {}).get(si, 1))
    sample = max(int(k), cfg_k) > 1 or any(e for _p, _m, e in plan)
    base_seed = int(getattr(args, "seed", 0) or 0)

    # F8: the canonical tokenizer — the same construction every HF-path
    # render and every fit uses, template pin included.
    tok = load_tokenizer(cfg["model"], revision=cfg.get("model_revision"))

    kw0 = _pos_vllm_sampling_kwargs(cfg, sample=sample, base_seed=base_seed,
                                    candidate_idx=0, max_tokens=cap)
    _check_and_write_provenance(cdir, _generation_provenance(
        engine="vllm", leg=f"generate-pos:{si}", cfg=cfg,
        vllm_version=getattr(vllm, "__version__", None),
        sampling_params={
            **{f: v for f, v in kw0.items() if f != "seed"},
            "sample": bool(sample), "base_seed": base_seed,
            "seed_rule": "base_seed + 1000 * candidate_idx (greedy: None)",
        },
        prompt_render=_prompt_render_record(tok, {si: instr.text}),
        info={"k": int(k)}), tag="generate-pos")

    # Group by attack family — grade-pos and _pos_plan read the pos run as
    # per-family <run>/<family>.jsonl files — and mask each pid's EXACT
    # existing index set so a prior candidate is never regenerated and a hole
    # (F9) always is. The disk union is belt-and-suspenders: the plan already
    # folds run-file coverage in.
    by_fam: dict[str, list] = defaultdict(list)
    for pid, missing, existing in plan:
        _require_engine_keyable(pid, tag="generate-pos")
        by_fam[pid[0]].append((pid, tuple(missing), frozenset(existing)))

    sampler = _pool_sampler(args, cfg, seed=base_seed, max_new_tokens=cap)

    total = 0
    rendered_of: dict = {}   # per-pid chat render, built once across passes

    def _rendered(pid):
        if pid not in rendered_of:
            rendered_of[pid] = build_chat_prompt(
                tok, none_prompt[pid]["prompt"], system=instr.text)
        return rendered_of[pid]

    for fam in sorted(by_fam):
        out_path = cdir / f"{fam}.jsonl"
        done = {_pos_cell_key(pid, si, ci)
                for pid, _m, existing in by_fam[fam] for ci in existing}
        done |= load_done_keys(out_path)
        for ci in range(int(k)):
            todo = [pid for pid, m, _e in by_fam[fam]
                    if ci in m and _pos_cell_key(pid, si, ci) not in done]
            if not todo:
                continue
            rendered = [_rendered(pid) for pid in todo]
            sp = vllm.SamplingParams(**_pos_vllm_sampling_kwargs(
                cfg, sample=sample, base_seed=base_seed, candidate_idx=ci,
                max_tokens=cap))
            outs = sampler.llm.generate(rendered, sp)
            # Append after the block so a crash mid-block loses only that
            # block's work; resume regenerates exactly the missing cells.
            with out_path.open("a") as f:
                for pid, o in zip(todo, outs):
                    f.write(json.dumps(_ungraded_pos_row(
                        none_prompt[pid], o.outputs[0].text, si=si,
                        candidate_idx=ci)) + "\n")
                    total += 1
                f.flush()
                os.fsync(f.fileno())
            print(f"[generate-pos] {fam} candidate {ci}: +{len(todo)} rows "
                  f"(total {total})", flush=True)
        # F9: exact-set verification — realized cells must EQUAL what the
        # plan's existing set plus this run's missing set say should now be
        # on disk; a hole or an extra fails loudly here, at the writer.
        _verify_pos_cells(
            out_path, tag="generate-pos", si=si,
            expected={pid: set(existing) | set(m)
                      for pid, m, existing in by_fam[fam]})
    return total


def _length_representative_pids(pids, none_prompt, n: int) -> list:
    """A LENGTH-REPRESENTATIVE slice of ``pids`` (--max-prompts): order by
    prompt length (ties broken by pid, so the slice is deterministic and
    stable across resumes) and take n evenly spaced entries — the pilot's
    small pool must exercise the same length spectrum the full pool would
    (short chat prompts through the long adversarial tail), never the first n
    of an arbitrary ordering. Returned in sorted-pid order (the plan's own
    order), so slicing changes WHICH prompts run, not their sequence."""
    if n < 1:
        raise SystemExit(f"[generate-pos] --max-prompts must be >= 1, got {n}")
    pids = sorted(pids)
    if n >= len(pids):
        return pids
    by_len = sorted(pids,
                    key=lambda pid: (len(none_prompt[pid]["prompt"]), pid))
    if n == 1:
        picks = [by_len[len(by_len) // 2]]
    else:
        step = (len(by_len) - 1) / (n - 1)
        picks = [by_len[round(i * step)] for i in range(n)]
    return sorted(picks)


def _pilot_pairs_path(out_path) -> Path:
    """Resolve + guard the --pilot-pairs-out destination. Called at command
    START (before any GPU work — a refused path must not cost a generation
    pass) and again at write time (belt and suspenders)."""
    out_path = Path(out_path).expanduser().resolve()
    if out_path == REPO or out_path.is_relative_to(REPO):
        raise SystemExit(
            f"[generate-pos] refusing to write pilot pairs under the repo "
            f"({out_path}) — the pairs jsonl carries attack text and must "
            "only ever live in a private dir outside the repo "
            "(--pilot-pairs-out elsewhere).")
    return out_path


def _emit_pilot_pairs(cfg, *, si, rdir, none_prompt, pids, out_path,
                      static_families) -> None:
    """``--pilot-pairs-out``: emit the UNGRADED ``{prompt,pos,neg,
    instruction_id}`` pairs for ``pids`` straight from the run dir + the
    store's undefended responses — the pilot's step-5 artifact
    (``pod_pilot.sh``), consumed by the acceptance probe and the pilot
    training runs without a manifest or a grading pass. ``pos`` is the
    LOWEST-index candidate on disk (deterministic across resumes); ``neg`` is
    the store's undefended response. Every row is stamped ``ungraded`` +
    ``pilot_only`` so nothing downstream can mistake this file for the frozen
    pool. Attack text: written mode-0600 (atomic) under a 0700-created parent
    and REFUSED inside the repo — the emit-pairs containment discipline."""
    from sisteer.eval.jailbreak_artifacts import private_umask

    out_path = _pilot_pairs_path(out_path)

    pos_by_pid: dict[tuple, tuple] = {}
    if Path(rdir).exists():
        for raw in _iter_store_rows(Path(rdir), static_families):
            if raw.get("si_id") != si or raw.get("response") is None:
                continue
            pid = (raw["attack"], raw["goal_id"], _variant_key(raw))
            ci = int(raw.get("candidate_idx", 0) or 0)
            cur = pos_by_pid.get(pid)
            if cur is None or ci < cur[0]:
                pos_by_pid[pid] = (ci, raw["response"])

    rows, holes = [], []
    for pid in sorted(pids):
        info = none_prompt.get(pid)
        pos = pos_by_pid.get(pid)
        if info is None or info.get("none_response") is None or pos is None:
            holes.append(pid)
            continue
        rows.append({
            # the exact fields train_two_si_objects._load_pairs consumes...
            "prompt": info["prompt"], "pos": pos[1],
            "neg": info["none_response"], "instruction_id": si,
            # ...the pair identity, so a reader can trace any pilot row...
            "family": pid[0], "goal_id": pid[1], "variant_key": pid[2],
            "candidate_idx": pos[0],
            # ...and the honest three-state stamps: never the frozen pool.
            "ungraded": True, "pilot_only": True,
        })
    if holes:
        preview = ", ".join(map(str, holes[:3]))
        raise SystemExit(
            f"[generate-pos] pilot pairs: {len(holes)} selected prompt(s) "
            f"have no generated {si} candidate or no undefended response on "
            f"disk ({preview}{', …' if len(holes) > 3 else ''}) — generation "
            "did not cover the slice; refusing to emit a silently smaller "
            "pilot pool.")
    if not rows:
        raise SystemExit(
            "[generate-pos] pilot pairs: the selected slice is EMPTY — "
            "nothing to emit (no none-successful prompts survived the "
            "selection); refusing to write an empty pilot pool.")
    with private_umask():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    _rewrite_private_jsonl(out_path, rows)
    print(f"[generate-pos] pilot pairs: wrote {len(rows)} UNGRADED "
          f"{{prompt,pos,neg}} pair(s) -> {out_path} (pilot-only; pos = the "
          "lowest-index candidate, neg = the undefended store response).")


# --------------------------------------------------------------------------- #
# grade-pos — piece C (the grading host): grade the ungraded rows generate-pos wrote      #
# --------------------------------------------------------------------------- #


def _read_jsonl_rows(path: Path) -> list[dict]:
    """Every JSONL row in file order, tolerating a torn final line from a
    SIGKILL mid-append (same tolerance `_iter_store_rows` gives readers)."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _rewrite_private_jsonl(path: Path, rows: list[dict]) -> None:
    """Atomically rewrite a mode-0600 private JSONL with `rows` (tmp + replace
    under umask 077, so a crash mid-write cannot corrupt the file or loosen its
    mode). Only ever used on this script's OWN run outputs, never the shared
    labelled store — so replacing rows in place (to fold in each row's
    freshly-computed score/label) is safe here."""
    from sisteer.eval.jailbreak_artifacts import private_umask

    path = Path(path)
    with private_umask():
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def cmd_grade_pos(args, cfg) -> None:
    """Grade the ungraded SI `pos` rows generate-pos wrote — a grading-host
    step. Reads every pos run (`<private-root>/<run>/<family>.jsonl`; one SI
    with `--si`), grades every row still lacking a score via the live Low-Prio
    StrongREJECT judge, and rewrites each row with its score so a subsequent
    `manifest` run counts it as a usable `pos` (thwarting candidates only —
    the rejection-sampling acceptance lives in the manifest computation).
    Grader failure -> the row stays UNSCORED and the run raises loudly; never
    a fabricated refusal 0 (JB1). Resumable + cache-aware: an already-scored
    row is skipped, and the live grader's on-disk ScoreCache means a re-run
    after a partial failure re-grades only the rows that were left unscored."""
    static_families = tuple(cfg["pool"]["static_families"])
    runs = _pos_runs(cfg)
    only_si = getattr(args, "si", None)
    if only_si is not None:
        if only_si not in runs:
            raise SystemExit(
                f"[grade-pos] --si {only_si!r} has no generated pos run "
                f"(known: {sorted(runs)})")
        runs = {only_si: runs[only_si]}

    proot = Path(jailbreak_run._private_root(cfg, args.private_root))
    per_file: dict[Path, list[dict]] = {}
    ungraded_items: list[tuple] = []
    rows_by_item: dict[tuple, list[dict]] = {}
    grade_dirs: list[Path] = []
    for si, run in sorted(runs.items()):
        cdir = proot / run
        if not cdir.exists():
            continue
        grade_dirs.append(cdir)
        for fam in static_families:
            path = cdir / f"{fam}.jsonl"
            if not path.exists():
                continue
            rows = _read_jsonl_rows(path)
            per_file[path] = rows
            for row in rows:
                if row.get("score") is not None:
                    continue                    # already graded — skip (resume)
                prompt, response = row.get("attack_prompt"), row.get("response")
                if prompt is None or response is None:
                    continue                    # not a gradable pos row
                # SCORE THE ANSWER, NOT THE REASONING. On a thinking model the
                # response carries a <think> trace; judging it would measure a
                # different object than on every non-thinking arm, so the
                # numbers would not be comparable across models. An
                # UNTERMINATED trace means the token budget ran out before any
                # answer -- strip_think returns "" for that, and an empty
                # answer is skipped as ungradable rather than scored as a
                # refusal, which is how a truncation artefact would otherwise
                # become a confident "thinking mode is safer".
                # A template-opened trace has NO "<think>" in the response -- the tag was
                # emitted as part of the PROMPT (Qwen3.x with thinking on), so gating
                # on "<think>" made this whole strip a silent no-op and the judge
                # scored the reasoning. That shipped on qwen38_think8k: 99.5% of
                # harmful rows and 99.7-99.8% of benign rows carried "</think>" with
                # no opening tag. Gate on EITHER tag; strip_think handles both forms.
                if "<think>" in response or "</think>" in response:
                    response = _strip_think(response)
                    if not response:
                        continue
                # Grade item shape MATCHES the BoN/adaptive live path
                # (dataset, revision, goal, response): the judge scores on goal +
                # response, so the attack family and goal id ride in the first two
                # slots (used only for the cache key + a leak-free error message).
                item = (fam, row.get("goal_id"), prompt, response)
                ungraded_items.append(item)
                rows_by_item.setdefault(item, []).append(row)

    if not grade_dirs:
        print("[grade-pos] no generated pos rows to grade under "
              f"{proot} — run generate-pos on a GPU node first.")
        return

    n_ungraded = sum(len(v) for v in rows_by_item.values())
    print(f"[grade-pos] {n_ungraded} ungraded pos row(s) across "
          f"{len(per_file)} family file(s) in {len(grade_dirs)} run(s).")
    if args.dry_run:
        return
    if not ungraded_items:
        print("[grade-pos] nothing to grade — every row already has a score.")
        return

    # Coordination courtesy line (print-and-proceed), printed when the site
    # names a coordination channel (SISTEER_COORDINATION_CHANNEL); grading
    # NEVER waits on the post going out — the behavioural protection is the
    # grader's same-org backoff, not the post.
    channel = (os.environ.get("SISTEER_COORDINATION_CHANNEL") or "").strip()
    n_unique = len(dict.fromkeys(ungraded_items))
    peak = min(args.concurrency, max(1, n_unique))
    model_id = (cfg.get("grader") or {}).get("model", "unknown-model")
    if channel:
        print(f"[coordination] ready to post to {channel} "
              "(grading proceeds now — print-and-proceed):\n"
              f"  {peak} {model_id} lowprio ~{max(1, n_unique // 5000)}h "
              f"(two-SI grade-pos, {n_unique} StrongREJECT judgments)",
              flush=True)

    grade = _build_grader(args, cfg, grade_dirs[0])  # LiveStrongREJECTGrader.grade
    # grade() returns a score for EVERY item or raises loudly if any stays
    # UNSCORED after retries — so a grader failure never lands a 0, and the
    # rewrite below runs only once every ungraded row has a real score.
    scores = grade(list(dict.fromkeys(ungraded_items)))
    for item, score in scores.items():
        for row in rows_by_item.get(item, []):
            row["score"] = float(score)
            row["graded"] = True

    for path, rows in per_file.items():
        _rewrite_private_jsonl(path, rows)
    print(f"[grade-pos] graded {n_ungraded} row(s). Run `manifest` to fold "
          "them into the matched pool.")


def _build_grader(args, cfg, cache_dir,
                  cache_name="constitution_pos_grade_cache.jsonl"):
    """The StrongREJECT grade fn for grade-pos / grade-anti-neg (live Low-Prio
    path). Returns `fn(items) -> {item: score}` keyed on `(family, goal_id,
    prompt, response)` (the `(dataset, revision, goal, response)` live-grader
    shape). Lazy-imported so generate-pos / generate-anti-neg / `manifest` /
    `--dry-run` never touch the grader — importing it and its `~/.config` key
    is what crashes on a compute node. ``cache_name`` keeps each pass's
    on-disk score cache disjoint."""
    from sisteer.eval.attacks.grading_live import LiveStrongREJECTGrader

    grader = LiveStrongREJECTGrader(
        model=cfg["grader"]["model"], concurrency=args.concurrency,
        cache_path=Path(cache_dir) / cache_name,
        allow_unscored=int(getattr(args, "allow_unscored", 0) or 0))
    return grader.grade


# --------------------------------------------------------------------------- #
# generate-anti-neg / grade-anti-neg — bidpo's anti-safety negative (piece B''/ #
# C''). Mirrors generate-pos/grade-pos in shape, with two differences: it       #
# generates for EVERY harmful prompt (not just none-successful ones), and its    #
# acceptance direction is COMPLIANCE (score > threshold), the opposite of pos.   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _AntiPrompt:
    """One harmful `none`-condition prompt, wrapped in the minimal attribute
    surface the sampler engine reads (``engine._row`` / ``engine.cell_key`` /
    ``engine.load_done_keys``). Used by BOTH anti-neg (`_anti_none_prompts`)
    and generate-pos (`_pos_engine_prompt`) — both feed the same vLLM engine.
    Rebuilt from the store's own `none` rows so the cell key
    (``family|goal_id|variant_key|si|candidate_idx``) joins EXACTLY with the
    pool's harmful pairs (every family stamps an int ``sample_idx``, so
    ``variant_key`` matches ``_variant_key``). ``category_group`` is carried so
    generate-pos's engine rows keep the store's group label the HF path wrote
    (anti-neg leaves it None, deriving the split at read time)."""

    family: str
    goal_id: str
    sample_idx: int
    trial: int
    text: str
    category: object = None
    source: object = None
    side: str = "harmful"
    variant_name: object = None
    category_group: object = None

    @property
    def variant_key(self) -> str:
        return f"{int(self.sample_idx)}:{int(self.trial)}"


def _pos_engine_prompt(c: dict) -> "_AntiPrompt":
    """Wrap one ``_pos_plan`` planning dict in the engine's prompt surface (a
    pos prompt IS a harmful `none`-condition attack prompt). ``sample_idx`` /
    ``trial`` come from the opaque ``variant_key`` (``sample_idx:trial``) — the
    same parse ``_ungraded_pos_row`` does. A variant_key with no integer
    ``sample_idx`` (the order/hash fallback in ``_variant_key``) cannot be
    engine-keyed, so refuse LOUDLY here rather than let the engine crash on
    ``int(None)`` deep in a block: every current pool family stamps an integer
    ``sample_idx`` (verified across the store), so a non-numeric key means an
    unexpected store row the caller should see, not a silent skip."""
    vk = str(c["variant_key"])
    head = vk.split(":")[0] if ":" in vk else vk
    if ":" not in vk or not head.lstrip("-").isdigit():
        raise SystemExit(
            f"[generate-pos] prompt {c.get('family')}/{c.get('goal_id')} has a "
            f"non-numeric variant_key {vk!r} (the order/hash fallback); the "
            "vLLM engine keys cells on an integer sample_idx and every current "
            "pool family stamps one. A null here is an unexpected store row.")
    return _AntiPrompt(
        family=c["family"], goal_id=c["goal_id"], sample_idx=int(head),
        trial=int(vk.split(":")[1]), text=c["prompt"],
        category=c.get("category"), source=None, side="harmful",
        variant_name=None, category_group=c.get("category_group"))


def _anti_none_prompts(cfg, spec, static_families) -> dict:
    """Every harmful `none` TRAIN prompt as an ``_AntiPrompt``, keyed by pid —
    the set the anti-safety negative pass covers (EVERY harmful prompt, not the
    none-successful subset generate-pos uses). Pure/read-only over the store, so
    `--dry-run` needs no model."""
    store_dir = _store_dir(cfg["pool"])
    train_fams = set(spec.train_families)
    train_goals = set(spec.train_goal_ids)
    out: dict[tuple, _AntiPrompt] = {}
    for raw in _iter_store_rows(store_dir, static_families):
        if raw.get("si_condition") != "none":
            continue
        if (raw.get("attack") not in train_fams
                or raw.get("goal_id") not in train_goals):
            continue
        if raw.get("attack_prompt") is None or raw.get("sample_idx") is None:
            continue
        pid = (raw["attack"], raw["goal_id"], _variant_key(raw))
        if pid in out:
            continue
        out[pid] = _AntiPrompt(
            family=raw["attack"], goal_id=raw["goal_id"],
            sample_idx=int(raw["sample_idx"]), trial=int(raw.get("trial", 0)),
            text=raw["attack_prompt"], category=raw.get("category"),
            source=raw.get("source"), side=raw.get("side", "harmful"),
            variant_name=raw.get("variant_name"))
    return out


def cmd_generate_anti_neg(args, cfg) -> None:
    """PIECE B'' (GPU, GENERATION ONLY): elicit bidpo's anti-safety negatives.

    For EVERY harmful `none` train prompt, sample K candidate completions under
    ``safety_refusal.anti_text`` as the system prompt (the AUTHORED anti text —
    reused, never re-authored), persisted UNGRADED to the anti-neg run keyed with
    the ``anti_si_id`` pseudo-SI. NO grader is imported here (the live key
    lives in ~/.config, absent on a compute node). Resume is
    per-cell via ``engine.cell_key`` / ``_anti_neg_done_keys`` — a re-run (or a
    raised K) generates exactly the missing cells, never a re-run of what exists.
    Uses the same vLLM engine as the main array (prefix-cached), one engine per
    process.

    DATA-PARALLEL across GPUs (2026-08-20), the ``pool_sample.py`` pattern
    reused shape-for-shape: ``--shard-index``/``--shard-count`` give this
    worker a contiguous slice of the harmful-prompt inventory
    (``pool_sample._shard`` — the same deterministic split, over the same
    deterministically-ordered store read, so N shards are disjoint and cover
    exactly the set one shard would). Each shard appends to its OWN file
    (``_anti_neg_shard_file``) — concurrent appends to one shared file are the
    bug class the budget pass hit — and every reader takes the union
    (``_anti_neg_paths``). Defaults 0/1 reproduce the single-GPU behaviour
    byte-for-byte, legacy filename included. ``generate-anti-neg-all`` fans
    out one worker per GPU."""
    anti = _anti_neg_cfg(cfg)
    if not anti["enabled"]:
        raise SystemExit(
            "[generate-anti-neg] pool.anti_neg.enabled is false — nothing to "
            "generate. Enable pool.anti_neg (and its anti_si_id/run/candidates) "
            "to build bidpo's anti-safety negatives.")
    # This leg has only ever run on the vLLM engine — there is no HF anti-neg
    # path to retain, and the HF contingency covers generate-pos /
    # generate-benign only. Refuse engine=hf LOUDLY rather than inventing an
    # HF branch whose sampling distribution nothing has validated: a model
    # that fails the pilot's vLLM gate has NO anti-neg source (bidpo's cell
    # becomes a recorded hole), which the record must state, not paper over.
    if _resolve_engine(args, cfg, default="vllm",
                       tag="generate-anti-neg") != "vllm":
        raise SystemExit(
            "[generate-anti-neg] engine=hf is not implemented for the "
            "anti-neg leg (it has always run on the vLLM engine; the §WS7 HF "
            "contingency covers generate-pos/generate-benign only). On an "
            "HF-fallback model, bidpo's anti-neg source is a RECORDED hole — "
            "say so in the run record rather than generating with an "
            "unvalidated engine.")
    from sisteer.data.instructions import SAFETY_SI

    anti_text = SAFETY_SI.anti_text
    if not anti_text:
        raise SystemExit(
            "[generate-anti-neg] safety_refusal.anti_text is empty — the "
            "authored anti-safety instruction is the REQUIRED elicitation prompt "
            "(bidpo amendment); refusing to generate without it.")
    static_families = tuple(cfg["pool"]["static_families"])
    train_goals, eval_goals = _goal_split_from_store(cfg, static_families)
    spec = _pool_spec(cfg, train_goals, eval_goals)
    prompts = list(_anti_none_prompts(cfg, spec, static_families).values())

    k = int(args.candidates) if args.candidates is not None else anti["candidates"]
    if k < 1:
        raise SystemExit(f"[generate-anti-neg] --candidates must be >= 1, got {k}")

    from sisteer.sampling.engine import cell_key

    shard_index = getattr(args, "shard_index", None)
    shard_count = getattr(args, "shard_count", None)
    shard_index = 0 if shard_index is None else int(shard_index)
    shard_count = 1 if shard_count is None else int(shard_count)
    if shard_count < 1 or not (0 <= shard_index < shard_count):
        raise SystemExit(
            f"[generate-anti-neg] bad shard spec {shard_index}/{shard_count}: "
            "need shard-count >= 1 and 0 <= shard-index < shard-count (an "
            "out-of-range shard would silently generate an empty slice).")
    n_total = len(prompts)
    prompts = pool_sample._shard(prompts, shard_index, shard_count)

    # Path only (no creation) so `--dry-run` has no side effects; the real path
    # creates it mode-0700 via private_run_dir below. _anti_neg_done_keys
    # tolerates missing files (empty set), so the coverage count works before
    # any write — and it reads EVERY anti file, so a partial run under one
    # shard layout resumes correctly under another.
    adir = Path(jailbreak_run._private_root(cfg, args.private_root)) / anti["run"]
    out_path = adir / _anti_neg_shard_file(shard_index, shard_count)
    si = anti["anti_si_id"]
    done = _anti_neg_done_keys(adir)
    todo = sum(1 for p in prompts for ci in range(k)
               if cell_key(p, si, ci) not in done)
    print(f"[generate-anti-neg] anti_si={si} (run {anti['run']}, K={k}, "
          f"shard {shard_index}/{shard_count}): {len(prompts)} of {n_total} "
          f"harmful train prompts x K = {len(prompts) * k} cells; "
          f"{len(done)} already generated, {todo} to generate.")
    if args.dry_run:
        return
    if not todo:
        print("[generate-anti-neg] nothing to generate — coverage complete.")
        return

    # real generation (GPU) — lazy-imported; NO grader on this path.
    # (vLLM's own import stays where it always was — inside VllmSampler — so
    # this command's behaviour under a stubbed sampler is unchanged; the
    # version probe for provenance is best-effort.)
    try:
        import vllm as _vllm_mod
        _vllm_version = getattr(_vllm_mod, "__version__", None)
    except ImportError:
        _vllm_version = None
    from sisteer.eval.jailbreak_artifacts import private_run_dir, private_umask
    from sisteer.sampling.engine import GenSpec

    private_run_dir(                            # create mode-0700 (repairs mode)
        anti["run"], root=jailbreak_run._private_root(cfg, args.private_root))
    # top_p 1.0 / top_k disabled are the engine's HISTORICAL anti-neg values —
    # bfdd5d8 deliberately preserved them (the HF-parity set applies to the pos
    # teacher only; anti-neg has no HF reference run to match). Recorded so a
    # resume with a drifted seed/temperature/engine fails loudly.
    # Record the anti-neg render too (F8 completeness): the anti-neg leg now
    # renders through VllmSampler's canonical load_tokenizer path, so hash its
    # rendered system prefix like the pos/benign legs — a drifted anti-neg
    # render then refuses on resume via the strict prompt_render check.
    from sisteer.models import load_tokenizer as _load_tok
    _anti_tok = _load_tok(cfg["model"], revision=cfg.get("model_revision"))
    _check_and_write_provenance(adir, _generation_provenance(
        engine="vllm", leg="generate-anti-neg", cfg=cfg,
        vllm_version=_vllm_version,
        sampling_params={
            "temperature": float(anti["temperature"]), "top_p": 1.0,
            "top_k": -1, "max_tokens": int(cfg["completion_tokens"]),
            "sample": k > 1, "base_seed": int(args.seed),
            "seed_rule": "engine seed + 1000 * candidate_idx (greedy: None)",
        },
        info={"k": int(k)},
        prompt_render=_prompt_render_record(_anti_tok, {si: anti_text})),
        tag="generate-anti-neg")
    gen_spec = GenSpec(si_id=si, system_text=anti_text, k=k,
                       temperature=anti["temperature"])
    sampler = _pool_sampler(args, cfg, seed=int(args.seed),
                            max_new_tokens=int(cfg["completion_tokens"]))
    with private_umask():                       # anti_neg.jsonl stays mode-0600
        res = sampler.run(prompts, [gen_spec], out_path, done=done)
    print(json.dumps(res, indent=2))
    print(f"[generate-anti-neg] wrote UNGRADED anti-safety negatives -> "
          f"{out_path}. Grade them on the grading host: build_two_si_pool.py "
          "grade-anti-neg --private-root <root>.")


def cmd_generate_anti_neg_all(args, cfg) -> None:
    """Fan out one generate-anti-neg worker per GPU — ``pool_sample.py
    sample-all`` reused shape-for-shape (flags, shard space, env handling).

    Separate processes rather than threads: each vLLM engine owns a CUDA
    context and a memory pool, and one process per GPU is the only arrangement
    that keeps `gpu_memory_utilization` meaning what it says. The engines are
    independent (no NCCL, no collectives), so the shard space is GLOBAL like
    sample-all's: `--shard-total` is the total GPU count across every pod and
    `--shard-offset` this pod's first global shard index. Each child's
    CUDA_VISIBLE_DEVICES is one entry of what the environment already exposes
    (``pool_sample._visible_gpus`` reads the existing variable first) — a
    strict subset, never widened. Worker logs land INSIDE the mode-0700 run
    dir (they carry counts and paths only, never completion text — but the
    private dir costs nothing and a vLLM stack trace is not worth auditing
    for leaks)."""
    anti = _anti_neg_cfg(cfg)
    if not anti["enabled"]:
        raise SystemExit(
            "[generate-anti-neg-all] pool.anti_neg.enabled is false — nothing "
            "to generate.")
    # UNION (stage3 -> twosweep merge): both guards, they catch different
    # things.
    # same refusal as the per-shard command, before any worker is spawned:
    # the anti-neg leg is vLLM-only (no HF path exists — see generate-anti-neg)
    if _resolve_engine(args, cfg, default="vllm",
                       tag="generate-anti-neg-all") != "vllm":
        raise SystemExit(
            "[generate-anti-neg-all] engine=hf is not implemented for the "
            "anti-neg leg — see generate-anti-neg's refusal.")
    # ...and the SHARED --gpus resolver, not a second copy. This function is
    # "sample-all reused shape-for-shape", and it had reproduced sample-all's
    # unguarded --gpus parsing too: a GPU COUNT typed into a DEVICE-ID LIST
    # silently ran one GPU over 1/N of the work and exited 0 (job 212789).
    # Fixing pool_sample alone would have left this site to hit the identical
    # bug on the very next pass -- the gpu-util meta-rule, audit every site.
    gpus = pool_sample.resolve_gpus(args.gpus, pool_sample._visible_gpus())
    if not gpus:
        raise SystemExit("[generate-anti-neg-all] no GPUs found")
    total = args.shard_total or len(gpus)
    offset = args.shard_offset
    if offset + len(gpus) > total:
        raise SystemExit(
            f"[generate-anti-neg-all] shard space overflow: offset {offset} + "
            f"{len(gpus)} local GPUs > --shard-total {total}. Every pod's "
            "[offset, offset+gpus) must fit inside one global shard space, or "
            "shards silently overlap and rows are generated twice.")
    print(f"[generate-anti-neg-all] global shards [{offset}, "
          f"{offset + len(gpus)}) of {total}", flush=True)

    from sisteer.eval.jailbreak_artifacts import private_run_dir, private_umask

    adir = private_run_dir(                      # created 0700 before any log
        anti["run"], root=jailbreak_run._private_root(cfg, args.private_root))

    # Put the running interpreter's bin/ on PATH for the children: a venv
    # python invoked by ABSOLUTE PATH does not add its bin/ to PATH, and vLLM
    # shells out to `ninja` — same fix as pool_sample.cmd_sample_all.
    bindir = str(Path(sys.executable).parent)
    base_env = dict(os.environ)
    base_env["PATH"] = bindir + os.pathsep + base_env.get("PATH", "")
    passthrough = ["--config", args.config, "--seed", str(args.seed),
                   "--max-model-len", str(args.max_model_len),
                   "--gpu-memory-utilization",
                   str(args.gpu_memory_utilization)]
    if args.candidates is not None:
        passthrough += ["--candidates", str(args.candidates)]
    if args.private_root is not None:
        passthrough += ["--private-root", str(args.private_root)]
    if args.enforce_eager:
        passthrough.append("--enforce-eager")
    if getattr(args, "engine", None):
        # a CLI --engine override outranks pool.engine; the children re-resolve
        # from the same config, so the override must travel with them
        passthrough += ["--engine", args.engine]

    procs = []
    with private_umask():                        # logs stay 0600 in the 0700 dir
        for i, gpu in enumerate(gpus):
            env = dict(base_env, CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [sys.executable, __file__, "generate-anti-neg",
                   "--shard-index", str(offset + i),
                   "--shard-count", str(total)] + passthrough
            log = Path(adir) / f"anti_neg_shard_{offset + i:02d}.log"
            print(f"[generate-anti-neg-all] GPU {gpu} -> shard {offset + i} "
                  f"(log {log})", flush=True)
            procs.append((offset + i, gpu, subprocess.Popen(
                cmd, env=env, stdout=log.open("w"),
                stderr=subprocess.STDOUT)))
    failed = []
    for idx, gpu, p in procs:
        rc = p.wait()
        print(f"[generate-anti-neg-all] shard {idx} (GPU {gpu}) exit {rc}",
              flush=True)
        if rc != 0:
            failed.append((idx, gpu, rc))
    if failed:
        raise SystemExit(
            f"[generate-anti-neg-all] {len(failed)} shard(s) FAILED: {failed}")
    print("[generate-anti-neg-all] all shards complete")


def cmd_grade_anti_neg(args, cfg) -> None:
    """PIECE C'' (the grading host): grade bidpo's anti-safety negatives with the live
    Low-Prio StrongREJECT judge and rewrite each row with its score, so
    `manifest` accepts the first candidate scored a genuine COMPLIANCE
    (> threshold) as bidpo's rejected side. Reads EVERY file the anti run
    holds — the legacy shard-less anti_neg.jsonl plus each shard's
    anti_neg_shard_NN.jsonl (``_anti_neg_paths``), so a sharded generation is
    graded whole. Grades LAZILY in rounds (the grade-store scheme, opposite
    acceptance direction): candidate c+1 is graded only once candidates 0..c
    are all scored non-compliances, because the manifest freezes the SMALLEST
    complying candidate_idx — the identical result at a fraction of the K=8
    calls; skipped candidates stay honestly ungraded (three-state rule).
    Grader failure -> the row stays UNSCORED and the run raises loudly; never
    a fabricated 0 (JB1). Resumable + cache-aware (its own on-disk score
    cache); every file is rewritten atomically after each round."""
    anti = _anti_neg_cfg(cfg)
    if not anti["enabled"]:
        raise SystemExit(
            "[grade-anti-neg] pool.anti_neg.enabled is false — nothing to grade.")
    adir = Path(jailbreak_run._private_root(cfg, args.private_root)) / anti["run"]
    paths = _anti_neg_paths(adir)
    if not paths:
        print(f"[grade-anti-neg] no anti-neg rows under {adir} — run "
              "generate-anti-neg on a GPU node first.")
        return

    si_ids = tuple(cfg["si_ids"])
    thr = float(cfg["prereg"]["threshold_grid"][1])
    rows_by_path = {p: _read_jsonl_rows(p) for p in paths}
    # Groups may SPAN files (a legacy partial single-GPU file plus a later
    # sharded run), so the rounds run across the whole run dir and every file
    # is rewritten (atomically) after each round.
    all_rows = [r for rows in rows_by_path.values() for r in rows]
    groups = _candidate_groups(all_rows, si_ids, anti["anti_si_id"],
                               sides=("anti",))

    def _is_scored(row):
        return row.get("score") is not None

    qualifies = _sr_qualifies(si_ids, anti["anti_si_id"], thr)
    n_ungraded = sum(1 for g in groups.values() for r in g if not _is_scored(r))
    nxt, never, _unres = _lazy_round0_stats(groups, is_scored=_is_scored,
                                            qualifies=qualifies)
    print(f"[grade-anti-neg] {n_ungraded} ungraded anti-neg row(s) across "
          f"{len(paths)} file(s); lazy round 0 grades {nxt} "
          f"({never} already proven unnecessary by an accepted compliance).")
    if args.dry_run:
        return
    if not nxt:
        print("[grade-anti-neg] nothing to grade — every candidate the "
              "smallest-compliance rule can consult is already scored.")
        return

    grade = _build_grader(args, cfg, adir,
                          cache_name="anti_neg_grade_cache.jsonl")

    def _rewrite_all():
        for p, rows in rows_by_path.items():
            _rewrite_private_jsonl(p, rows)

    def _fold(row, score):
        row["score"] = float(score)
        row["graded"] = True

    n, rounds = _lazy_candidate_rounds(
        groups, is_scored=_is_scored, qualifies=qualifies, grade=grade,
        fold_row=_fold, rewrite=_rewrite_all, tag="grade-anti-neg")
    print(f"[grade-anti-neg] graded {n} row(s) in {rounds} round(s) across "
          f"{len(paths)} file(s). Run `manifest` + `emit-pairs` to fold the "
          "accepted compliances into bidpo's rejected side.")


# --------------------------------------------------------------------------- #
# generate-benign / grade-benign — the stage-3 benign half                    #
# --------------------------------------------------------------------------- #


def _benign_goal_id(benign_source: str, prompt: str) -> str:
    """A safe opaque id for one benign prompt — the sha16 of its exact bytes,
    prefixed with the source. Deterministic, so re-runs and readers key the
    same prompt identically without ever persisting the text as a key."""
    from sisteer.eval.matched_pool import text_sha

    return f"{benign_source}_{text_sha(prompt)[:16]}"


def _load_benign_corpora(cfg: dict, *, phtest_path_override: str | None = None):
    """Load every declared benign corpus (scale-up amendment 2026-08-20),
    each through its pinned loader in `content_sets.BENIGN_LOADERS` — never a
    re-implementation. Returns an ordered ``list[(corpus_name, prompts)]``,
    DEDUPED ACROSS corpora by ``normalized_text_sha`` (a prompt in an earlier
    corpus is not re-added by a later one, so the families stay disjoint and
    a shared prompt is counted once). `pool.benign_n` (null = all) caps each
    corpus; per-corpus loader kwargs come from the `benign_sources` spec;
    `phtest_path_override` (CLI --phtest-path) keeps the PHTest corpus off the
    hub for tests/offline. A legacy single `benign_source` yields one corpus,
    unchanged."""
    from sisteer.data.content_sets import BENIGN_LOADERS
    from sisteer.eval.jailbreak_data import normalized_text_sha

    pool = cfg["pool"]
    seed = int(pool.get("benign_seed", 0))
    seen: set[str] = set()
    out: list[tuple[str, list[str]]] = []
    for spec in _benign_source_specs(cfg):
        name = spec["name"]
        loader_name = spec.get("loader", name)
        loader = BENIGN_LOADERS.get(loader_name)
        if loader is None:
            raise SystemExit(
                f"[generate-benign] unknown benign loader {loader_name!r} for "
                f"corpus {name!r} (known: {sorted(BENIGN_LOADERS)})")
        kwargs = {k: v for k, v in spec.items()
                  if k not in ("name", "loader")}
        kwargs.setdefault("seed", seed)
        if pool.get("benign_n") is not None:
            kwargs.setdefault("n", pool.get("benign_n"))
        if name == "phtest":
            if phtest_path_override or pool.get("benign_path"):
                kwargs["path"] = phtest_path_override or pool.get("benign_path")
            kwargs.setdefault("split", pool.get("benign_split", "all"))
        prompts = loader(**kwargs)
        deduped = []
        for p in prompts:
            h = normalized_text_sha(p)
            if h in seen:
                continue
            seen.add(h)
            deduped.append(p)
        out.append((name, deduped))
    return out


def _benign_plan(cfg, corpora, *, private_root=None):
    """(CPU-only) which (prompt, condition) benign generations are missing.
    Conditions: "none" (undefended, the shared neg) + one per configured SI.
    ``corpora`` is ``_load_benign_corpora``'s output. Returns
    ``(bdir, by_gid, missing)`` where ``by_gid[gid] = (corpus_name, prompt)``
    and ``missing`` maps condition -> ordered list of gids still needing a
    response (across ALL corpora)."""
    benign_source, benign_run = _benign_cfg(cfg)
    if benign_source is None:
        raise SystemExit(
            "[generate-benign] the config declares no pool.benign_source — "
            "nothing to generate (declare benign_source/benign_fraction for "
            "the stage-3 benign half).")
    bdir = Path(jailbreak_run._private_root(cfg, private_root)) / benign_run
    by_gid: dict[str, tuple[str, str]] = {}
    for name, prompts in corpora:
        for p in prompts:
            by_gid[_benign_goal_id(name, p)] = (name, p)

    have: dict[str, set[str]] = {}
    if bdir.exists():
        for raw in _iter_store_rows(bdir, _benign_families(cfg)):
            gid = raw.get("goal_id")
            if not gid or raw.get("response") is None:
                continue
            kind = raw["si_id"] if raw.get("si_id") else (
                "none" if raw.get("si_condition") == "none" else None)
            if kind:
                have.setdefault(kind, set()).add(gid)

    conditions = ["none", *cfg["si_ids"]]
    missing = {
        cond: [g for g in sorted(by_gid) if g not in have.get(cond, set())]
        for cond in conditions}
    return bdir, by_gid, missing


def _benign_row(name: str, gid: str, prompt: str, response: str,
                cond: str) -> dict:
    """One UNGRADED benign store row — the single builder BOTH engines write
    through, so their rows are schema-identical by construction. Each corpus is
    its OWN family/source (labelled stratum), so the manifest's per-family
    counts keep every corpus visible."""
    row = {"attack": name, "source": name, "goal_id": gid,
           "order": 0, "attack_prompt": prompt, "response": response,
           "score": None, "xstest_label": None, "graded": False}
    if cond == "none":
        row["si_condition"] = "none"
    else:
        row["si_condition"] = "prompted"
        row["si_id"] = cond
        row["candidate_idx"] = 0
    return row


def cmd_generate_benign(args, cfg) -> None:
    """PIECE B' (GPU, GENERATION ONLY): the benign half's raw material. For
    every PHTest prompt, generate the undefended `none` response (the shared
    `neg`) and one SI-prompted response per configured SI, persisted UNGRADED
    to the benign run. No grader is imported here; the XSTest labels come
    from grade-benign on the grading host. Resumable per (prompt,
    condition).

    Both generations are GREEDY (the historical `generate_batch` default), so
    the two engines (`--engine` / `pool.engine`; hf is the retained
    contingency, vllm the campaign ruling) are deterministic argmax decodes of
    the same model — no sampler configuration to keep in parity here, unlike
    the pos leg. Provenance-recorded either way."""
    engine = _resolve_engine(args, cfg, default="hf", tag="generate-benign")
    benign_source, benign_run = _benign_cfg(cfg)
    corpora = _load_benign_corpora(cfg, phtest_path_override=args.phtest_path)
    bdir, by_gid, missing = _benign_plan(cfg, corpora,
                                         private_root=args.private_root)
    n_missing = sum(len(v) for v in missing.values())
    per = ", ".join(f"{cond}={len(v)}" for cond, v in missing.items())
    corp = ", ".join(f"{name}={len(ps)}" for name, ps in corpora)
    print(f"[generate-benign] {len(by_gid)} benign prompts ({corp}) x "
          f"{len(missing)} conditions: {n_missing} responses missing "
          f"({per}; engine={engine}).")
    if args.dry_run:
        return
    if not n_missing:
        print("[generate-benign] nothing to generate — coverage complete.")
        return

    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.eval.jailbreak_artifacts import append_jsonl, private_run_dir

    # Load-bearing for the mode-0700 creation; run name resolved above.
    bdir = private_run_dir(
        benign_run, root=jailbreak_run._private_root(cfg, args.private_root))
    cap = int(cfg["completion_tokens"])

    if engine == "vllm":
        _generate_benign_vllm(args, cfg, bdir=bdir, by_gid=by_gid,
                              missing=missing, cap=cap,
                              resolve_instruction=resolve_instruction)
        print(f"[generate-benign] wrote {n_missing} UNGRADED benign row(s) -> "
              f"{bdir}. Label the SI rows on the grading host: "
              "build_two_si_pool.py grade-benign --private-root <root>.")
        return

    # -- hf contingency path: the pre-ruling behaviour, RETAINED unchanged ----
    from sisteer.generate import generate_batch
    from sisteer.models import build_chat_prompt, load_model_and_tokenizer

    _check_and_write_provenance(bdir, _generation_provenance(
        engine="hf", leg="generate-benign", cfg=cfg,
        sampling_params={"do_sample": False, "max_new_tokens": cap,
                         "greedy": True}), tag="generate-benign")
    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))

    bs = int(args.batch_size)
    for cond, gids in missing.items():
        if not gids:
            continue
        system = None if cond == "none" else resolve_instruction(cond).text
        for i in range(0, len(gids), bs):
            chunk = gids[i:i + bs]
            rendered = [build_chat_prompt(tok, by_gid[g][1], system=system)
                        for g in chunk]
            outs = generate_batch(model, tok, rendered, batch_size=bs,
                                  max_new_tokens=cap)
            for g, o in zip(chunk, outs):
                name, prompt = by_gid[g]
                append_jsonl(bdir / f"{name}.jsonl",
                             _benign_row(name, g, prompt, o, cond))
    print(f"[generate-benign] wrote {n_missing} UNGRADED benign row(s) -> "
          f"{bdir}. Label the SI rows on the grading host: "
          "build_two_si_pool.py grade-benign --private-root <root>.")


def _generate_benign_vllm(args, cfg, *, bdir, by_gid, missing, cap,
                          resolve_instruction) -> None:
    """generate-benign on the shared vLLM engine. GREEDY (temperature 0), the
    exact decode discipline the hf path's `generate_batch` default carries —
    a deterministic argmax has no sampler to keep in parity, so the pos leg's
    top_p/top_k port does not apply here. Conditions run as contiguous blocks
    (`none` + one per SI) so each SI's system prefix is prefilled once and
    prefix-cached; appends are fsync'd every `_VLLM_BENIGN_BLOCK` prompts so a
    killed pod repeats at most one block, and `_benign_plan`'s per
    (prompt, condition) resume skips everything already on disk."""
    vllm = _require_vllm("generate-benign")
    from sisteer.eval.jailbreak_artifacts import append_jsonl
    from sisteer.models import build_chat_prompt, load_tokenizer

    # F8: render through the harness's OWN tokenizer + template semantics
    # (enable_thinking pinned off, Gemma system-role fold) — never the
    # engine's raw AutoTokenizer render; see _generate_pos_vllm.
    tok = load_tokenizer(cfg["model"], revision=cfg.get("model_revision"))
    system_of = {cond: (None if cond == "none"
                        else resolve_instruction(cond).text)
                 for cond in missing}
    _check_and_write_provenance(bdir, _generation_provenance(
        engine="vllm", leg="generate-benign", cfg=cfg,
        vllm_version=getattr(vllm, "__version__", None),
        sampling_params={"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                         "max_tokens": cap, "greedy": True, "seed": None},
        prompt_render=_prompt_render_record(tok, system_of)),
        tag="generate-benign")

    sampler = _pool_sampler(args, cfg, seed=int(getattr(args, "seed", 0) or 0),
                            max_new_tokens=cap)
    sp = vllm.SamplingParams(temperature=0.0, max_tokens=cap)

    for cond, gids in missing.items():
        if not gids:
            continue
        system = system_of[cond]
        for i in range(0, len(gids), _VLLM_BENIGN_BLOCK):
            chunk = gids[i:i + _VLLM_BENIGN_BLOCK]
            rendered = [build_chat_prompt(tok, by_gid[g][1], system=system)
                        for g in chunk]
            outs = sampler.llm.generate(rendered, sp)
            for g, o in zip(chunk, outs):
                name, prompt = by_gid[g]
                append_jsonl(bdir / f"{name}.jsonl",
                             _benign_row(name, g, prompt,
                                         o.outputs[0].text, cond))
            print(f"[generate-benign] {cond}: +{len(chunk)} rows", flush=True)


def cmd_grade_benign(args, cfg) -> None:
    """PIECE C' (the grading host): label the benign SI rows with the live XSTest C/P/I
    judge (only `C` is consumed). `none` rows need no label
    (they only supply the shared `neg`). Judge failure -> the row stays
    UNLABELLED and the run raises loudly; never a fabricated label. Resumable:
    already-labelled rows are skipped."""
    benign_source, benign_run = _benign_cfg(cfg)
    if benign_source is None:
        raise SystemExit(
            "[grade-benign] the config declares no pool.benign_source — "
            "nothing to grade.")
    bdir = Path(jailbreak_run._private_root(cfg, args.private_root)) / benign_run
    paths = [bdir / f"{name}.jsonl" for name in _benign_families(cfg)
             if (bdir / f"{name}.jsonl").exists()]
    if not paths:
        print(f"[grade-benign] no benign rows to grade under {bdir} — run "
              "generate-benign on a GPU node first.")
        return

    # collect the unlabelled SI rows across every corpus file, grade ONCE,
    # then write each file back (a corpus with no unlabelled rows is untouched)
    rows_by_path: dict[Path, list[dict]] = {}
    items: list[tuple] = []
    rows_by_item: dict[tuple, list[dict]] = {}
    for path in paths:
        rows = _read_jsonl_rows(path)
        rows_by_path[path] = rows
        for row in rows:
            if not row.get("si_id") or row.get("xstest_label") is not None:
                continue                        # a none row, or already labelled
            prompt, response = row.get("attack_prompt"), row.get("response")
            if prompt is None or response is None:
                continue
            item = (row.get("attack"), row.get("goal_id"), prompt, response)
            items.append(item)
            rows_by_item.setdefault(item, []).append(row)

    n = sum(len(v) for v in rows_by_item.values())
    print(f"[grade-benign] {n} unlabelled benign SI row(s) across "
          f"{len(paths)} corpus file(s).")
    if args.dry_run:
        return
    if not items:
        print("[grade-benign] nothing to grade — every SI row has a label.")
        return

    grade_xs = _build_xstest_grader(args, cfg, bdir)
    labels = grade_xs(list(dict.fromkeys(items)))
    for item, label in labels.items():
        for row in rows_by_item.get(item, []):
            row["xstest_label"] = str(label)
            row["graded"] = True
    for path, rows in rows_by_path.items():
        _rewrite_private_jsonl(path, rows)
    print(f"[grade-benign] labelled {n} row(s) across {len(paths)} file(s). "
          "Run `manifest` to fold them into the benign intersection.")


def _build_xstest_grader(args, cfg, cache_dir):
    """The XSTest C/P/I label fn for grade-benign (live Low-Prio path).
    Returns `fn(items) -> {item: "C"|"P"|"I"}` keyed on `(source, goal_id,
    prompt, response)`. Lazy-imported — same discipline as the
    StrongREJECT grader."""
    from sisteer.eval.attacks.grading_live import LiveStrongREJECTGrader

    grader = LiveStrongREJECTGrader(
        model=cfg["grader"]["model"], concurrency=args.concurrency,
        cache_path=Path(cache_dir) / "benign_xstest_grade_cache.jsonl",
        # Same tolerance as the StrongREJECT path (_build_grader). Wiring it
        # there and NOT here left grade-store half-guarded: the SR side rode
        # out its un-gradeable items and the XSTest side then killed the run
        # anyway, hours in. A per-pass grader is not a per-pass POLICY.
        allow_unscored=int(getattr(args, "allow_unscored", 0) or 0))
    return grader.grade_xstest


# --------------------------------------------------------------------------- #
# grade-store — grade the labelled store ITSELF (pool_sample merge output)      #
# --------------------------------------------------------------------------- #


def _store_row_grade_kind(row: dict, si_ids, anti_si_id,
                          store_pos_si=_STORE_PROMPTED_SI) -> "str | None":
    """Which grade one HARMFUL-family store row needs, mirroring
    ``_row_to_graded``'s condition mapping EXACTLY (same acceptance, no
    re-derivation): an explicit ``si_id`` names any SI directly (what the
    sampler engine / generate-pos write); ``si_condition=="none"`` is the
    undefended row (its score decides requirement_a strict-vs-relaxed, so it
    needs a grade like every other harmful row); ``"prompted"`` without an
    ``si_id`` is the store's own SAFETY_SI teacher. ``"compiled"`` rows are
    object-defense transcripts — never pool rows, never graded here (grading
    them would be ~85% of the BoN file spent on rows every reader skips).

    ``store_pos_si`` is `_store_pos_si(cfg)` when the rows come from the
    STORE's harmful families: None means the store SI has flipped to a
    generated pos run, so its store rows (explicit si_id and legacy bare
    `prompted` alike) are no longer pool rows and need no grade — the same
    exclusion `_iter_all_pool_rows` applies at read time. Callers grading
    BENIGN files or a RUN dir pass the default (the flip never touches those).

    Returns ``"none" | "si" | "anti" | None`` (None = not a pool row)."""
    if row.get("attack_prompt") is None or row.get("response") is None:
        return None
    if row.get("si_id"):
        si = row["si_id"]
        if si in si_ids:
            if store_pos_si is None and si == _STORE_PROMPTED_SI:
                return None                    # flipped: the run is the source
            return "si"
        if anti_si_id is not None and si == anti_si_id:
            return "anti"
        return None                            # an SI we are not comparing
    sc = row.get("si_condition")
    if sc == "none":
        return "none"
    if sc == "prompted":
        if store_pos_si is None:
            return None                        # flipped: the run is the source
        return "si"                            # the store SAFETY_SI
    return None                                # "compiled" and anything else


def _candidate_groups(rows, si_ids, anti_si_id, *, sides=("none", "si", "anti"),
                      store_pos_si=_STORE_PROMPTED_SI):
    """Group gradable pool rows into rejection-sampling candidate sets — one
    group per (family, goal_id, variant_key, condition), rows sorted by
    ``candidate_idx``. The group is the unit the lazy grading rounds resolve:
    the manifest's reduction (``matched_pool._reduce_side``) accepts a prompt
    iff ANY candidate qualifies and freezes the SMALLEST qualifying
    ``candidate_idx``, so within one group candidates are graded strictly in
    index order and grading stops the moment the answer is decided.
    ``store_pos_si`` is threaded to ``_store_row_grade_kind`` — pass
    `_store_pos_si(cfg)` when grouping STORE harmful rows."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        kind = _store_row_grade_kind(row, si_ids, anti_si_id,
                                     store_pos_si=store_pos_si)
        if kind is None or kind not in sides:
            continue
        si = row.get("si_id") or ("__prompted__" if kind == "si" else "__none__")
        key = (row.get("attack"), row.get("goal_id"), _variant_key(row), si)
        groups.setdefault(key, []).append(row)
    for g in groups.values():
        g.sort(key=lambda r: int(r.get("candidate_idx", 0)))
    return groups


def _sr_qualifies(si_ids, anti_si_id, thr, store_pos_si=_STORE_PROMPTED_SI):
    """Does one SCORED harmful row already decide its group? Mirrors the
    manifest's acceptance directions exactly: an SI pos candidate qualifies by
    THWARTING (``not score > thr`` — matched_pool's ``_reduce_side`` lambda),
    an anti candidate by a genuine COMPLIANCE (``score > thr``), and a `none`
    row never short-circuits anything (it is not rejection-sampled; its score
    is needed whatever it is — it decides requirement_a strict-vs-relaxed).
    ``store_pos_si`` threads through to ``_store_row_grade_kind``."""
    def q(row: dict) -> bool:
        s = row.get("score")
        if s is None:
            return False
        kind = _store_row_grade_kind(row, si_ids, anti_si_id,
                                     store_pos_si=store_pos_si)
        if kind == "anti":
            return float(s) > thr
        if kind == "si":
            return not (float(s) > thr)
        return False
    return q


def _next_candidate(cand_rows, is_scored, qualifies):
    """The next row to grade in one group, or None when the group is RESOLVED
    (a scored candidate below the first ungraded index qualifies — grading
    anything further can never change the frozen smallest-qualifying index)
    or EXHAUSTED (every candidate scored, none qualified). Walking in index
    order also repairs odd partial states: a qualifying score at index c
    resolves the group only once every index < c is scored, so the lazy
    result is IDENTICAL to grading all K candidates."""
    for r in cand_rows:                         # sorted by candidate_idx
        if is_scored(r):
            if qualifies(r):
                return None                     # resolved
            continue                            # scored, did not qualify
        return r                                # smallest unscored candidate
    return None                                 # exhausted


def _lazy_round0_stats(groups, *, is_scored, qualifies):
    """What one grading round would send NOW, plus the ungraded rows the lazy
    scheme has already proven unnecessary. Returns ``(next_round_rows,
    never_needed_rows, unresolved_groups)`` — the dry-run numbers that size
    the coordination post with real calls instead of the naive worst case."""
    nxt = never = unresolved = 0
    for cand_rows in groups.values():
        r = _next_candidate(cand_rows, is_scored, qualifies)
        if r is not None:
            nxt += 1
            unresolved += 1
            continue
        if any(is_scored(x) and qualifies(x) for x in cand_rows):
            never += sum(1 for x in cand_rows if not is_scored(x))
    return nxt, never, unresolved


def _lazy_candidate_rounds(groups, *, is_scored, qualifies, grade, fold_row,
                           rewrite, tag):
    """Round-based rejection-sampling grading: grading candidate c+1 can
    never change the answer once candidate c qualified, so ~93% of K=8
    blocks resolve on round 0.

    Each round grades, for every unresolved group, its single next candidate
    (``_next_candidate``); folds the results in; and calls ``rewrite()`` so
    progress is durable per round. Candidates past the qualifying one stay
    honestly UNGRADED (score/label null — the three-state rule: a reader can
    tell "never needed" from "scored low"). Terminates in <= K rounds; a
    grader that returns nothing for a pending item raises loudly rather than
    spinning (the real grader already raises on any unscored item, JB1).
    Returns ``(rows_graded, rounds)``."""
    total = rounds = 0
    while True:
        batch: dict[tuple, list[dict]] = {}
        for cand_rows in groups.values():
            nxt = _next_candidate(cand_rows, is_scored, qualifies)
            if nxt is None:
                continue
            item = (nxt["attack"], nxt.get("goal_id"), nxt["attack_prompt"],
                    nxt["response"])
            batch.setdefault(item, []).append(nxt)
        if not batch:
            return total, rounds
        values = grade(list(batch))
        n = 0
        for item, value in values.items():
            for row in batch.get(item, []):
                fold_row(row, value)
                n += 1
        if n == 0:
            raise RuntimeError(
                f"[{tag}] grader returned no result for {len(batch)} pending "
                "item(s) — refusing to spin (JB1: an unscored item raises, "
                "never silently retries forever).")
        rewrite()
        total += n
        rounds += 1
        print(f"[{tag}] round {rounds}: graded {n} row(s) "
              f"({len(batch)} unique item(s))", flush=True)


def _grade_store_plan(cfg, store_dir: Path):
    """(CPU-only) what grade-store would grade, per family and side — the
    whole of ``--dry-run``, and the pass the real path uses to pick which
    files carry work. Counts only; one family file's rows in memory at a
    time, and unique items counted through an opaque sha — no text retained.

    Reports BOTH the naive total (every ungraded pool row) and the lazy
    numbers (exact round-0 calls; rows already proven unnecessary), so the
    coordination post is sized from real calls, not the worst case.

    Returns ``(report, harmful_pending, benign_pending)`` where the pending
    lists name the family files whose next lazy round has at least one call
    (a file whose ungraded rows all sit in resolved groups has NO work)."""
    pool = cfg["pool"]
    si_ids = tuple(cfg["si_ids"])
    anti = _anti_neg_cfg(cfg)
    anti_si_id = anti["anti_si_id"] if anti["enabled"] else None
    thr = float(cfg["prereg"]["threshold_grid"][1])
    harmful_fams = sorted(set(pool["static_families"])
                          | set(pool.get("replay_families", ())))
    benign_fams = sorted(_benign_families(cfg))
    # bidpo's reversed benign side: its chosen side is the UNDEFENDED
    # benign response, which must be a VERIFIED XSTest
    # compliance — so with the feature enabled the benign `none` rows need
    # labels too (they used to need none: "only supplies the shared neg").
    # Feature off -> byte-identical behavior, none rows stay ungraded.
    grade_none = _bidpo_benign_cfg(cfg)["enabled"]
    benign_sides = ("si", "none") if grade_none else ("si",)
    # HARMFUL store rows honour the store-vs-generated-run precedence: once
    # the store SI flips to a generated run its store rows need no grade
    # (they are not pool rows). BENIGN rows always keep the default — the
    # flip is harmful-pos-only.
    store_pos = _store_pos_si(cfg)
    sr_q = _sr_qualifies(si_ids, anti_si_id, thr, store_pos_si=store_pos)
    sr_scored = lambda r: r.get("score") is not None            # noqa: E731
    xs_scored = lambda r: r.get("xstest_label") is not None     # noqa: E731
    xs_q = lambda r: r.get("xstest_label") == "C"               # noqa: E731

    def _item_sha(row):
        h = hashlib.sha256()
        for part in (str(row.get("attack")), str(row.get("goal_id")),
                     row["attack_prompt"], row["response"]):
            h.update(part.encode())
            h.update(b"\x00")
        return h.digest()

    report = {"store_dir": str(store_dir), "harmful": {}, "benign": {}}
    sr_items: set[bytes] = set()
    xs_items: set[bytes] = set()
    n_sr_rows = n_xs_rows = 0
    sr_round0 = sr_never = xs_round0 = xs_never = 0
    harmful_pending: list[str] = []
    benign_pending: list[str] = []

    for fam in harmful_fams:
        path = store_dir / f"{fam}.jsonl"
        if not path.exists():
            continue
        rows = _read_jsonl_rows(path)
        cnt = {"none": 0, "si": 0, "anti": 0}
        n_skipped = 0
        for row in rows:
            kind = _store_row_grade_kind(row, si_ids, anti_si_id,
                                         store_pos_si=store_pos)
            if kind is None:
                n_skipped += 1
                continue
            if row.get("score") is not None:
                continue                        # already graded — resume
            cnt[kind] += 1
            n_sr_rows += 1
            sr_items.add(_item_sha(row))
        groups = _candidate_groups(rows, si_ids, anti_si_id,
                                   store_pos_si=store_pos)
        nxt, never, _unres = _lazy_round0_stats(
            groups, is_scored=sr_scored, qualifies=sr_q)
        sr_round0 += nxt
        sr_never += never
        report["harmful"][fam] = {
            "rows": len(rows), "ungraded": cnt,
            "ungraded_total": sum(cnt.values()),
            "skipped_non_pool_rows": n_skipped,
            "lazy_round0_calls": nxt,
            "ungraded_rows_never_needed": never}
        if nxt:
            harmful_pending.append(fam)

    for fam in benign_fams:
        path = store_dir / f"{fam}.jsonl"
        if not path.exists():
            continue
        rows = _read_jsonl_rows(path)
        n_unlabelled = n_none = n_none_unlabelled = 0
        for row in rows:
            kind = _store_row_grade_kind(row, si_ids, anti_si_id)
            if kind in (None, "anti"):
                continue
            if kind == "none":
                if not grade_none:
                    n_none += 1                 # supplies the shared neg — no label
                    continue
                if row.get("xstest_label") is not None:
                    continue                    # already labelled — resume
                n_none_unlabelled += 1
                n_xs_rows += 1
                xs_items.add(_item_sha(row))
                continue
            if row.get("xstest_label") is not None:
                continue                        # already labelled — resume
            n_unlabelled += 1
            n_xs_rows += 1
            xs_items.add(_item_sha(row))
        groups = _candidate_groups(rows, si_ids, anti_si_id,
                                   sides=benign_sides)
        nxt, never, _unres = _lazy_round0_stats(
            groups, is_scored=xs_scored, qualifies=xs_q)
        xs_round0 += nxt
        xs_never += never
        entry = {
            "rows": len(rows), "unlabelled_si_rows": n_unlabelled,
            "lazy_round0_calls": nxt,
            "unlabelled_rows_never_needed": never}
        if grade_none:
            # bidpo reversed benign side: the undefended rows need labels too.
            entry["unlabelled_none_rows"] = n_none_unlabelled
        else:
            entry["none_rows_needing_no_label"] = n_none
        report["benign"][fam] = entry
        if nxt:
            benign_pending.append(fam)

    report["totals"] = {
        "strongreject_rows_naive": n_sr_rows,
        "strongreject_unique_items_naive": len(sr_items),
        "strongreject_round0_calls": sr_round0,
        "strongreject_rows_never_needed_now": sr_never,
        "strongreject_worst_case_calls": n_sr_rows - sr_never,
        "xstest_rows_naive": n_xs_rows,
        "xstest_unique_items_naive": len(xs_items),
        "xstest_round0_calls": xs_round0,
        "xstest_rows_never_needed_now": xs_never,
        "xstest_worst_case_calls": n_xs_rows - xs_never,
    }
    return report, harmful_pending, benign_pending


def cmd_grade_store(args, cfg) -> None:
    """Grade the labelled STORE itself — the step the pool_sample path was
    missing (verified 2026-08-20): ``pool_sample.py sample-all`` -> ``merge``
    writes every row ``score: null, graded: false`` by construction (three-
    state rule), ``manifest`` reads the store via ``_iter_all_pool_rows`` and
    ``_row_to_graded`` skips any harmful row without a numeric score — but
    ``grade-pos``/``grade-benign`` only ever read the generate-pos/-benign RUN
    dirs, never the store. So a merged store yields ZERO matched pairs until
    this runs.

    What it grades (the exact set ``_row_to_graded`` will consume):
    - HARMFUL families (static + replay): a StrongREJECT score for EVERY pool
      condition — the undefended ``none`` row (its score decides
      requirement_a), each comparison SI's rows, and the anti pseudo-SI's rows
      when ``pool.anti_neg.enabled``. ``compiled`` rows are skipped.
    - BENIGN families: an XSTest C/P/I ``xstest_label`` for the SI rows;
      ``none`` rows need no grade (they only supply the shared ``neg``) —
      UNLESS ``pool.bidpo_benign.enabled``, in which case the benign
      ``none`` rows are XSTest-labelled too: bidpo's
      reversed benign side uses the undefended response as its CHOSEN side,
      and that must be a VERIFIED compliance (label ``C``), never an assumed
      one. Same judge, same item shape, same lazy machinery (a none row is a
      K=1 group, so round 0 covers it).

    Same graders as grade-pos/grade-benign, so numbers are comparable by
    construction (``LiveStrongREJECTGrader.grade`` / ``.grade_xstest``, item
    shape ``(family, goal_id, attack_prompt, response)``). Resume-safe twice
    over: rows already carrying a score/label are skipped, and the on-disk
    ScoreCache (in the store dir) means a re-run after a partial failure
    re-grades only what is missing. Each family file is rewritten ATOMICALLY
    (tmp + rename, mode 0600) after every grading round, so an interruption
    loses at most one round's un-folded scores — and those are still in the
    cache. Grader failure -> the affected rows stay UNSCORED and the run
    raises loudly; never a fabricated 0 (JB1).

    Rejection-sampled candidate blocks (si_irrelevant at K=8; the anti
    pseudo-SI) are graded LAZILY in rounds: candidate c+1 is graded only
    when candidates 0..c are scored and none qualified, because the
    manifest's reduction accepts a prompt iff ANY candidate qualifies and
    freezes the SMALLEST qualifying candidate_idx — so later candidates
    can never change the answer. The frozen pool is IDENTICAL to grading
    all K; the calls saved are the ones that could not matter. Skipped
    candidates stay honestly ungraded (score null — "never needed" is
    distinguishable from "scored low"). One knock-on, named honestly: the
    manifest's rejection_sampling k_mean / per_candidate_accept_rate now
    describe the graded candidates only (per_prompt_yield, the feasibility
    number, is unchanged)."""
    store_dir = (Path(args.store_dir).expanduser() if args.store_dir
                 else _store_dir(cfg["pool"]))
    if not store_dir.exists():
        raise SystemExit(
            f"[grade-store] store dir {store_dir} does not exist — point "
            "--store-dir (or TT_TWO_SI_STORE_DIR) at the merged store.")

    si_ids = tuple(cfg["si_ids"])
    anti = _anti_neg_cfg(cfg)
    anti_si_id = anti["anti_si_id"] if anti["enabled"] else None
    thr = float(cfg["prereg"]["threshold_grid"][1])

    report, harmful_pending, benign_pending = _grade_store_plan(cfg, store_dir)
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return
    if not harmful_pending and not benign_pending:
        print("[grade-store] nothing to grade — every pool row already has "
              "its score/label.")
        return

    # Graders constructed lazily on the first real call: a store with no
    # remaining work must never touch the key in ~/.config.
    built: dict[str, object] = {}

    def _sr(items):
        if "sr" not in built:
            built["sr"] = _build_grader(args, cfg, store_dir,
                                        cache_name="store_grade_cache.jsonl")
        return built["sr"](items)

    def _xs(items):
        if "xs" not in built:
            built["xs"] = _build_xstest_grader(args, cfg, store_dir)
        return built["xs"](items)

    def _fold_score(row, score):
        row["score"] = float(score)
        row["graded"] = True

    def _fold_label(row, label):
        row["xstest_label"] = str(label)
        row["graded"] = True

    # One file at a time: lazy rounds, atomic rewrite per round, move on —
    # durable progress, memory bounded to one family file. Harmful files
    # honour the store-vs-generated-run precedence (`_store_pos_si`).
    store_pos = _store_pos_si(cfg)
    for fam in harmful_pending:
        path = store_dir / f"{fam}.jsonl"
        rows = _read_jsonl_rows(path)
        n, rounds = _lazy_candidate_rounds(
            _candidate_groups(rows, si_ids, anti_si_id,
                              store_pos_si=store_pos),
            is_scored=lambda r: r.get("score") is not None,
            qualifies=_sr_qualifies(si_ids, anti_si_id, thr,
                                    store_pos_si=store_pos),
            grade=_sr, fold_row=_fold_score,
            rewrite=lambda p=path, rws=rows: _rewrite_private_jsonl(p, rws),
            tag=f"grade-store {fam}")
        print(f"[grade-store] {fam}: graded {n} row(s) in {rounds} round(s) "
              f"-> {path.name}")

    # bidpo reversed benign side: label the undefended `none` rows too (each a
    # K=1 candidate group — the lazy rounds handle both sides in one pass).
    benign_sides = (("si", "none") if _bidpo_benign_cfg(cfg)["enabled"]
                    else ("si",))
    for fam in benign_pending:
        path = store_dir / f"{fam}.jsonl"
        rows = _read_jsonl_rows(path)
        n, rounds = _lazy_candidate_rounds(
            _candidate_groups(rows, si_ids, anti_si_id, sides=benign_sides),
            is_scored=lambda r: r.get("xstest_label") is not None,
            qualifies=lambda r: r.get("xstest_label") == "C",
            grade=_xs, fold_row=_fold_label,
            rewrite=lambda p=path, rws=rows: _rewrite_private_jsonl(p, rws),
            tag=f"grade-store {fam}")
        print(f"[grade-store] {fam}: labelled {n} row(s) in {rounds} "
              f"round(s) -> {path.name}")

    print("[grade-store] done. Run `manifest` to fold the graded store into "
          "the matched pool.")


# --------------------------------------------------------------------------- #
# emit-pairs — piece D: frozen manifest -> per-SI {prompt,pos,neg} jsonl        #
# --------------------------------------------------------------------------- #

# The fixed names emit-pairs writes under its mode-0700 output dir. Named here
# (not inline) so the overnight orchestrator and the tests share one spelling.
PAIRS_META_NAME = "pairs_meta.json"


def pairs_filename(si_id: str) -> str:
    """The per-SI pairs jsonl filename — the file `train_two_si_objects.py
    train --pairs` consumes for that SI."""
    return f"pairs_{si_id}.jsonl"


def bidpo_harmful_filename(si_id: str) -> str:
    """The per-SI bidpo HARMFUL pairs jsonl:
    this SI's OWN thwart set with the accepted anti-safety compliance as the
    rejected side — a SUPERSET of the matched intersection's harmful half
    (per-SI thwart, no 50/50 balance subsample), so it cannot ride in
    `pairs_<si>.jsonl`. pos = this SI's refusal (CHOSEN), neg = the anti
    compliance (REJECTED). Consumed by bidpo and ONLY bidpo."""
    return f"bidpo_harmful_{si_id}.jsonl"


def bidpo_benign_filename(si_id: str) -> str:
    """The per-SI REVERSED-POLARITY benign pairs jsonl — written beside
    `pairs_<si>.jsonl` so the fit driver's default sibling lookup (and the
    overnight rsync of the pairs dir) needs no extra plumbing. Consumed by
    bidpo and ONLY bidpo; its rows carry `requirement_a: "benign_reversed"`
    with pos = the undefended compliance (CHOSEN) and neg = this SI's
    refusal (REJECTED)."""
    return f"bidpo_benign_{si_id}.jsonl"


# The requirement_a stamp on every reversed benign row — the trainer REFUSES
# a benign file whose rows carry anything else, so the reversed-polarity file
# can never be confused with a matched-pool pairs file (whose pos is an SI
# response, the OPPOSITE polarity).
BIDPO_BENIGN_REQ_A = "benign_reversed"


def _is_ungraded_pos_candidate(raw: dict, si_ids, benign_source=None) -> bool:
    """True for a store row that WOULD be an SI `pos` but has no score (or,
    for a benign row, no XSTest label) yet — the generated-awaiting-grade
    state of the three-state rule. These are never usable training pos
    rows; emit-pairs skips them and reports the count so an unattended run's
    log says exactly how much pos material is still waiting on grading."""
    if not raw.get("attack") or not raw.get("goal_id"):
        return False
    if raw.get("attack_prompt") is None or raw.get("response") is None:
        return False
    is_si = bool(raw.get("si_id")) or raw.get("si_condition") == "prompted"
    if not is_si:
        return False
    if raw.get("si_id") and raw["si_id"] not in si_ids:
        return False
    if benign_source is not None and raw.get("attack") == benign_source:
        return raw.get("xstest_label") is None
    return raw.get("score") is None


def _load_rows_counting_ungraded(cfg: dict, *, static_families, private_root):
    """The graded rows emit-pairs resolves text from (same sources as
    `_load_graded_rows`, minus the replay families it never needs), plus a
    count of the SKIPPED ungraded pos candidates (three-state rule)."""
    si_ids = tuple(cfg["si_ids"])
    thr = float(cfg["prereg"]["threshold_grid"][1])   # 0.5 endpoint
    benign_source, _ = _benign_cfg(cfg)
    anti = _anti_neg_cfg(cfg)
    anti_si_id = anti["anti_si_id"] if anti["enabled"] else None

    rows, n_ungraded = [], 0
    for raw in _iter_all_pool_rows(cfg, static_families=static_families,
                                   private_root=private_root):
        g = _row_to_graded(raw, si_ids=si_ids, threshold=thr,
                           benign_source=benign_source, anti_si_id=anti_si_id,
                           store_prompted_si=_store_pos_si(cfg))
        if g is not None:
            rows.append(g)
        elif _is_ungraded_pos_candidate(raw, si_ids, benign_source):
            n_ungraded += 1
    return rows, n_ungraded


def _load_frozen_manifest(path: Path) -> dict:
    """Read the frozen manifest and refuse a tampered/stale one: the embedded
    `manifest_sha256` must match the recomputed content hash (the same guard
    `write_manifest` applies on freeze). The manifest is the AUTHORITATIVE
    record of the training pool — emitting pairs from an edited copy would
    silently train on a different pool than the frozen one."""
    from sisteer.eval.matched_pool import MatchedPoolError, manifest_hash

    if not path.exists():
        raise SystemExit(
            f"[emit-pairs] frozen manifest missing at {path} — run "
            "`build_two_si_pool.py manifest` first (emit-pairs only ever "
            "emits the pool the manifest froze, never a re-derived one).")
    manifest = json.loads(path.read_text())
    embedded = manifest.get("manifest_sha256")
    recomputed = manifest_hash(manifest)
    if embedded != recomputed:
        raise MatchedPoolError(
            f"[emit-pairs] manifest {path} failed its hash check (embedded "
            f"{embedded} != content {recomputed}) — refusing to emit pairs "
            "from a tampered or hand-edited manifest.")
    return manifest


def _resolve_manifest_pairs(manifest: dict, rows):
    """Resolve every manifest pair id back to its exact store text, BY HASH.

    Returns ``(emitted, records)``: ``emitted[si] -> ordered list of
    {prompt,pos,neg,...} dicts`` and ``records[si] -> ordered list of
    PairRecord`` (rebuilt from the resolved text, so the identity re-assert
    checks what is actually being written, not what the manifest claims).

    The manifest is authoritative: for each pair id the store row whose
    prompt/response sha MATCHES the frozen sha is selected — duplicates
    resolve to the frozen choice (a rejection-sampling candidate resolves the
    same way: only the frozen pos_sha's candidate matches), and a store that
    diverged from the manifest (regenerated rows, edited text) fails loudly
    naming the pair, never a silently different pool (authoritative-source
    rule)."""
    from sisteer.eval.matched_pool import (MatchedPoolError, PairRecord,
                                           text_sha)

    regime = manifest["regime"]
    si_ids = list(manifest["si_ids"])

    # bidpo's anti-safety negatives (bidpo amendment 2026-08-20): the frozen
    # per-pair anti_neg_sha, resolved back to text BY HASH like pos/neg. Present
    # only when the source was enabled at manifest time.
    anti_block = manifest.get("bidpo_anti_neg")
    anti_si_id = anti_block["anti_si_id"] if anti_block else None
    anti_map = ({(e["family"], e["goal_id"], e["variant_key"]): e["anti_neg_sha"]
                 for e in anti_block["ordered_pairs"]} if anti_block else {})

    none_cands: dict[tuple, list] = {}
    si_cands: dict[tuple, list] = {}
    for r in rows:
        if r.condition == "none":
            none_cands.setdefault(r.prompt_id, []).append(r)
        elif r.condition == "si":
            si_cands.setdefault((r.si_id, r.prompt_id), []).append(r)

    def entries_for(si: str) -> list[tuple[dict, str]]:
        """(manifest entry, pos_sha-for-this-SI) in frozen order."""
        if regime == "matched_intersection":
            return [(e, e["pos_sha"][si])
                    for e in manifest["matched_intersection"]["ordered_pairs"]]
        return [(e, e["pos_sha"])
                for e in manifest["native_per_si"][si]["ordered_pairs"]]

    emitted: dict[str, list[dict]] = {}
    records: dict[str, list] = {}
    for si in si_ids:
        out_rows, out_recs = [], []
        for entry, pos_sha in entries_for(si):
            pid = (entry["family"], entry["goal_id"], entry["variant_key"])
            none_row = next(
                (r for r in none_cands.get(pid, ())
                 if text_sha(r.prompt) == entry["prompt_sha"]
                 and text_sha(r.response) == entry["neg_sha"]), None)
            if none_row is None:
                raise MatchedPoolError(
                    f"[emit-pairs] store diverged from the frozen manifest: no "
                    f"`none` row for {pid} matches prompt_sha/neg_sha — the "
                    "store changed after the manifest froze; re-run `manifest` "
                    "rather than emitting a pool the manifest never froze.")
            si_row = next(
                (r for r in si_cands.get((si, pid), ())
                 if text_sha(r.response) == pos_sha
                 and text_sha(r.prompt) == entry["prompt_sha"]), None)
            if si_row is None:
                raise MatchedPoolError(
                    f"[emit-pairs] store diverged from the frozen manifest: no "
                    f"graded {si} row for {pid} matches its frozen pos_sha (and "
                    "prompt_sha) — an ungraded/regenerated row is not the "
                    "frozen pos; re-run grading + manifest.")
            row_out = {
                # The exact fields train_two_si_objects._load_pairs consumes...
                "prompt": none_row.prompt,
                "pos": si_row.response,
                "neg": none_row.response,
                "instruction_id": si,
                # ...the requirement-(a) label so the trainer can
                # route bidpo to the STRICT (genuine-harmful-compliance neg)
                # subset. Present on matched_intersection entries;
                # absent (-> None) in the native regime, where bidpo refuses.
                "requirement_a": entry.get("requirement_a"),
                # ...whether either side is an incoherent/repetitious
                # completion (2026-08-27: FLAG, do not filter). A small
                # model loops under greedy decode, and StrongREJECT reads a
                # loop as a successful defense -- so a flagged `pos` may be a
                # "refusal" the model never meant. Filtering would fix one
                # model and break comparison against the sibling pools, which
                # carry no filter; flagging lets the consumer decide.
                **{f"pos_{k}": v for k, v in
                   _degen(si_row.response).items()},
                **{f"neg_{k}": v for k, v in
                   _degen(none_row.response).items()},
                # ...plus the pair identity, so a 3am reader can trace any
                # emitted row back to its manifest entry without re-hashing.
                "family": entry["family"], "goal_id": entry["goal_id"],
                "variant_key": entry["variant_key"],
            }
            # bidpo's anti-safety negative for THIS harmful pair (SI-independent,
            # so the same anti_neg is written on every SI's file — bidpo's neg
            # differs only across arms, never across SIs). Resolved BY HASH so a
            # store that diverged from the frozen manifest fails loudly, never a
            # silently different rejected side. Harmful pairs with no accepted
            # anti_neg (not in the block) get no field -> bidpo drops them.
            if pid in anti_map:
                anti_sha = anti_map[pid]
                anti_row = next(
                    (r for r in si_cands.get((anti_si_id, pid), ())
                     if text_sha(r.response) == anti_sha
                     and text_sha(r.prompt) == entry["prompt_sha"]), None)
                if anti_row is None:
                    raise MatchedPoolError(
                        f"[emit-pairs] store diverged from the frozen manifest: "
                        f"no anti-safety row for {pid} matches its frozen "
                        "anti_neg_sha (and prompt_sha) — re-run grade-anti-neg + "
                        "manifest rather than emitting a different rejected side.")
                row_out["anti_neg"] = anti_row.response
            out_rows.append(row_out)
            out_recs.append(PairRecord(
                family=entry["family"], goal_id=entry["goal_id"],
                variant_key=entry["variant_key"],
                prompt_sha=text_sha(none_row.prompt),
                neg_sha=text_sha(none_row.response),
                pos_sha=text_sha(si_row.response)))
        emitted[si] = out_rows
        records[si] = out_recs
    return emitted, records


def _resolve_bidpo_pool(manifest: dict, rows):
    """Resolve the manifest's ``bidpo_benign`` block back to text BY HASH —
    the anti-neg discipline: the manifest is authoritative, and a store that
    diverged from a frozen sha fails loudly, never a silently different
    pair. Returns ``{si: {"harmful": [...], "benign": [...]}}`` in the
    block's frozen order, or ``{}`` when the block is absent.

    POLARITY, deliberately explicit, in the shape the untouched
    ``pairs_to_pref_examples`` consumes (pos -> chosen, neg -> rejected):

    - HARMFUL rows: pos = this SI's thwarting refusal (chosen_sha), neg = the
      anti-safety compliance (rejected_sha); ``requirement_a`` carries the
      frozen strict/relaxed label for reporting.
    - BENIGN rows, REVERSED: pos = the UNDEFENDED benign compliance
      (chosen_sha), neg = this SI's refusal (rejected_sha); every row is
      stamped ``requirement_a = BIDPO_BENIGN_REQ_A`` and the trainer refuses
      a file without that stamp."""
    from sisteer.eval.matched_pool import MatchedPoolError, text_sha

    block = manifest.get("bidpo_benign")
    if not block:
        return {}
    anti_si_id = block["anti_si_id"]
    none_cands: dict[tuple, list] = {}
    si_cands: dict[tuple, list] = {}
    for r in rows:
        if r.condition == "none":
            none_cands.setdefault(r.prompt_id, []).append(r)
        elif r.condition == "si":
            si_cands.setdefault((r.si_id, r.prompt_id), []).append(r)

    def _find(bucket, key, prompt_sha, response_sha):
        return next(
            (r for r in bucket.get(key, ())
             if text_sha(r.response) == response_sha
             and text_sha(r.prompt) == prompt_sha), None)

    emitted: dict[str, dict[str, list[dict]]] = {}
    for si, entry in block["per_si"].items():
        h_rows = []
        for e in entry["harmful"]["ordered_pairs"]:
            pid = (e["family"], e["goal_id"], e["variant_key"])
            si_row = _find(si_cands, (si, pid), e["prompt_sha"],
                           e["chosen_sha"])
            if si_row is None:
                raise MatchedPoolError(
                    f"[emit-pairs] store diverged from the frozen manifest: "
                    f"no {si} row for {pid} matches bidpo's frozen harmful "
                    "chosen_sha (the SI's thwarting refusal) — re-run "
                    "grading + manifest rather than emitting a different "
                    "chosen side.")
            anti_row = _find(si_cands, (anti_si_id, pid), e["prompt_sha"],
                             e["rejected_sha"])
            if anti_row is None:
                raise MatchedPoolError(
                    f"[emit-pairs] store diverged from the frozen manifest: "
                    f"no anti-safety row for {pid} matches bidpo's frozen "
                    "harmful rejected_sha — re-run grade-anti-neg + manifest "
                    "rather than emitting a different rejected side.")
            h_rows.append({
                "prompt": si_row.prompt,
                "pos": si_row.response,        # CHOSEN: this SI's refusal
                "neg": anti_row.response,      # REJECTED: the anti compliance
                **{f"pos_{k}": v for k, v in _degen(si_row.response).items()},
                **{f"neg_{k}": v for k, v in _degen(anti_row.response).items()},
                "instruction_id": si,
                "requirement_a": e["requirement_a"],
                "family": e["family"], "goal_id": e["goal_id"],
                "variant_key": e["variant_key"],
            })
        b_rows = []
        for e in entry["benign"]["ordered_pairs"]:
            pid = (e["family"], e["goal_id"], e["variant_key"])
            none_row = _find(none_cands, pid, e["prompt_sha"],
                             e["chosen_sha"])
            if none_row is None:
                raise MatchedPoolError(
                    f"[emit-pairs] store diverged from the frozen manifest: "
                    f"no benign `none` row for {pid} matches its frozen "
                    "prompt_sha/chosen_sha — the undefended compliance "
                    "bidpo's reversed side prefers is not the one the "
                    "manifest froze; re-run grading + manifest rather than "
                    "emitting a different chosen side.")
            si_row = _find(si_cands, (si, pid), e["prompt_sha"],
                           e["rejected_sha"])
            if si_row is None:
                raise MatchedPoolError(
                    f"[emit-pairs] store diverged from the frozen manifest: "
                    f"no {si} benign row for {pid} matches its frozen "
                    "rejected_sha (and prompt_sha) — re-run grading + "
                    "manifest rather than emitting a different rejected "
                    "side.")
            b_rows.append({
                # pos = CHOSEN = the undefended compliance; neg = REJECTED =
                # this SI's refusal. The REVERSED polarity is the point.
                "prompt": none_row.prompt,
                "pos": none_row.response,
                "neg": si_row.response,
                **{f"pos_{k}": v for k, v in _degen(none_row.response).items()},
                **{f"neg_{k}": v for k, v in _degen(si_row.response).items()},
                "instruction_id": si,
                "requirement_a": BIDPO_BENIGN_REQ_A,
                "family": e["family"], "goal_id": e["goal_id"],
                "variant_key": e["variant_key"],
            })
        emitted[si] = {"harmful": h_rows, "benign": b_rows}
    return emitted


def cmd_emit_pairs(args, cfg) -> None:
    """PIECE D: turn the frozen manifest into the per-SI `{prompt,pos,neg}`
    jsonl files `train_two_si_objects.py train --pairs` consumes.
    Deterministic (the manifest's frozen order, byte-stable across runs),
    regime-aware, and identity-asserted on write in the matched regime.
    Output only ever lands under a mode-0700 dir outside the repo."""
    from sisteer.eval.jailbreak_artifacts import private_run_dir, private_umask
    from sisteer.eval.matched_pool import assert_pools_identical_except_pos

    pool = cfg["pool"]
    static_families = tuple(pool["static_families"])
    pairs_run = pool.get("pairs_run", "two_si_pairs")

    mpath = Path(args.manifest) if args.manifest else Path(cfg["pool_manifest"])
    if not mpath.is_absolute():
        mpath = REPO / mpath
    manifest = _load_frozen_manifest(mpath)
    regime = manifest["regime"]
    si_ids = list(manifest["si_ids"])

    rows, n_ungraded = _load_rows_counting_ungraded(
        cfg, static_families=static_families, private_root=args.private_root)
    emitted, records = _resolve_manifest_pairs(manifest, rows)
    # bidpo's OWN per-SI pool:
    # resolved BY HASH from its own manifest block; {} when the block is
    # absent. Deliberately NOT identity-asserted — bidpo's per-SI files
    # DIFFER across SIs by design (per-SI thwart harmful side + each SI's
    # own benign refusals, unequal counts and ratios) and are EXPLICITLY
    # OUTSIDE the identical-except-pos guarantee below.
    bidpo_pool = _resolve_bidpo_pool(manifest, rows)

    if regime == "matched_intersection":
        # Re-asserted ON WRITE against the text actually being emitted
        # (reusing the module's five-invariant assert, never a
        # re-derivation): every SI's file carries the SAME ordered prompts and
        # the SAME neg, differing only in pos.
        assert_pools_identical_except_pos(records)

    # State the regime prominently in BOTH cases.
    sizes = ", ".join(f"{si}={len(emitted[si])}" for si in si_ids)
    print(f"[emit-pairs] REGIME: {regime} ({sizes})")
    print(f"[emit-pairs] {manifest['regime_note']}")
    anti_block = manifest.get("bidpo_anti_neg")
    if anti_block:
        rs = anti_block.get("rejection_sampling", {})
        print(f"[emit-pairs] bidpo anti-neg ({anti_block['anti_si_id']}): "
              f"{anti_block['n_with_anti_neg']}/{anti_block['n_harmful_pairs']} "
              "harmful pairs carry an accepted anti-safety compliance "
              f"(per-prompt yield {rs.get('per_prompt_yield')}, K max "
              f"{rs.get('k_max')}); the rest fall through to bidpo's loud check "
              "at train time. Every non-bidpo arm keeps consuming `neg`.")
    bb_block = manifest.get("bidpo_benign")
    if bb_block:
        mixes = bb_block["per_si"]
        bb_counts = "; ".join(
            f"{si}: harmful={mixes[si]['harmful']['n_with_anti_neg']}, "
            f"benign={mixes[si]['benign']['n_qualified']}, projected ratio "
            f"{mixes[si]['projected_mix_at_default_1to1']['ratio_harmful_to_benign']}"
            for si in si_ids)
        print(f"[emit-pairs] bidpo's OWN pool (section-13 mix; labels "
              f"{bb_block['refusal_labels']}; per-SI, UNEQUAL): {bb_counts}. "
              "Harmful = this SI's own thwart set + accepted anti compliance "
              "(NOT the matched intersection, NOT the 50/50 balance); benign "
              "= REVERSED polarity (chosen = the undefended compliance, "
              "rejected = that SI's refusal). NOTE: bidpo's files are the "
              "EXPLICIT exception to the identical-ordered-prompts-across-SIs "
              "guarantee above — they differ per SI by design, and the "
              "per-SI counts AND ratios must be stated alongside any bidpo "
              "cross-SI number.")
    if n_ungraded:
        print(f"[emit-pairs] skipped {n_ungraded} generated-but-UNGRADED pos "
              "row(s) (not usable until grading scores/labels them — the "
              "three-state rule, D47).")
    if args.dry_run:
        print("[emit-pairs] dry run — nothing written.")
        return

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (Path(jailbreak_run._private_root(cfg, args.private_root))
                   / pairs_run).resolve()
    # Harmful text NEVER lands inside the repo (it may be committed/pushed) —
    # checked BEFORE anything is created, so a refused path leaves no empty dir
    # behind in the repo.
    if out_dir == REPO or out_dir.is_relative_to(REPO):
        raise SystemExit(
            f"[emit-pairs] refusing to write pairs under the repo ({out_dir}) "
            "— the pairs jsonl carries attack text and must only ever live in "
            "a mode-0700 private dir (use --out-dir outside the repo).")
    if args.out_dir:
        with private_umask():
            out_dir.mkdir(parents=True, exist_ok=True)
        try:
            out_dir.chmod(0o700)
        except OSError:
            pass
    else:
        # private_run_dir creates it mode 0700 (and repairs a loose mode).
        private_run_dir(pairs_run,
                        root=jailbreak_run._private_root(cfg, args.private_root))

    meta_files = {}
    for si in si_ids:
        path = out_dir / pairs_filename(si)
        _rewrite_private_jsonl(path, emitted[si])
        file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        meta_files[si] = {"file": path.name, "n_pairs": len(emitted[si]),
                          "file_sha256": file_sha}
        print(f"[emit-pairs] wrote {len(emitted[si])} pairs -> {path}")

    # bidpo's own files, two per SI (harmful + benign-reversed), BESIDE the
    # pairs files (the fit driver's default sibling lookup + the overnight
    # pairs-dir rsync pick them up with no extra plumbing). Written only when
    # the manifest froze the block, so a feature-off emit is byte-identical
    # to before.
    bb_files = {}
    for si in si_ids:
        if si not in bidpo_pool:
            continue
        hpath = out_dir / bidpo_harmful_filename(si)
        _rewrite_private_jsonl(hpath, bidpo_pool[si]["harmful"])
        bpath = out_dir / bidpo_benign_filename(si)
        _rewrite_private_jsonl(bpath, bidpo_pool[si]["benign"])
        bb_files[si] = {
            "harmful": {
                "file": hpath.name,
                "n_pairs": len(bidpo_pool[si]["harmful"]),
                "file_sha256":
                    hashlib.sha256(hpath.read_bytes()).hexdigest()},
            "benign": {
                "file": bpath.name,
                "n_pairs": len(bidpo_pool[si]["benign"]),
                "file_sha256":
                    hashlib.sha256(bpath.read_bytes()).hexdigest()},
        }
        print(f"[emit-pairs] wrote bidpo {si}: "
              f"{len(bidpo_pool[si]['harmful'])} harmful -> {hpath}; "
              f"{len(bidpo_pool[si]['benign'])} benign-REVERSED -> {bpath}")

    # The regime sidecar — written in BOTH regimes so a later reader can never
    # mistake which comparison the trained objects came from. Counts/ids/
    # hashes only (no text), deliberately deterministic (no timestamps) so
    # re-running on the same manifest is byte-identical.
    sidecar = {
        "regime": regime,
        "regime_note": manifest["regime_note"],
        "headline_confounded_by_pool_composition":
            manifest["headline_confounded_by_pool_composition"],
        "pool_sizes": {si: len(emitted[si]) for si in si_ids},
        "si_ids": si_ids,
        "manifest_sha256": manifest["manifest_sha256"],
        "n_ungraded_pos_skipped": n_ungraded,
        "files": meta_files,
        "consumer": "scripts/train_two_si_objects.py train --pairs",
    }
    # Generation provenance travels WITH the emitted pairs (review
    # requirement): engine, vLLM version, sampling params, model + revision
    # for every generation leg, copied from the frozen manifest so the pairs
    # a trainer consumes carry their own engine record. The sidecar covers
    # every emitted file (each is listed with its sha under `files`); the
    # pairs jsonls themselves stay pure {prompt,pos,neg} rows — a header row
    # would corrupt the training format. Absent when the manifest predates
    # provenance recording, keeping legacy emits byte-identical.
    if manifest.get("generation_provenance"):
        sidecar["generation_provenance"] = manifest["generation_provenance"]
    # bidpo's rejected-side provenance (counts only), so the sidecar records
    # which negative source bidpo will train on and its realised coverage.
    if anti_block:
        sidecar["bidpo_anti_neg"] = {
            "enabled": True,
            "anti_si_id": anti_block["anti_si_id"],
            "n_harmful_pairs": anti_block["n_harmful_pairs"],
            "n_with_anti_neg": anti_block["n_with_anti_neg"],
            "n_missing_anti_neg": anti_block["n_missing_anti_neg"],
            "per_prompt_yield": anti_block.get(
                "rejection_sampling", {}).get("per_prompt_yield"),
        }
    # bidpo's own pool: counts + the label setting + the per-SI projected
    # mix, so the trainer can stamp every object's meta.json with the
    # PER-SI counts AND ratios (the confound context — the three objects
    # train at different harmful:benign RATIOS, trivially misread as an SI
    # effect) without re-opening the manifest. Counts only, never text.
    if bb_block:
        sidecar["bidpo_benign"] = {
            "enabled": True,
            "anti_si_id": bb_block["anti_si_id"],
            "refusal_labels": list(bb_block["refusal_labels"]),
            "undefended_required_label": bb_block[
                "undefended_required_label"],
            "harmful_selection":
                "per-SI thwart with an accepted anti compliance — NOT the "
                "matched intersection, NOT the 50/50 balance subsample",
            "benign_polarity": "reversed: chosen = undefended compliance, "
                               "rejected = that SI's refusal",
            "mix_rule": "benign = min(max_benign_per_harmful x harmful, "
                        "available) per SI; harmful never subsampled",
            "identity_exception":
                "bidpo's files are PER-SI and deliberately OUTSIDE the "
                "identical-ordered-prompts-across-SIs guarantee",
            "per_si": {
                si: {
                    "n_harmful": bb_block["per_si"][si]["harmful"][
                        "n_with_anti_neg"],
                    "n_benign_qualified": bb_block["per_si"][si]["benign"][
                        "n_qualified"],
                    "n_dropped_undefended_not_complied":
                        bb_block["per_si"][si]["benign"][
                            "n_dropped_undefended_not_complied"],
                    "projected_mix_at_default_1to1":
                        bb_block["per_si"][si][
                            "projected_mix_at_default_1to1"],
                } for si in si_ids},
            "files": bb_files,
            "consumer": "bidpo ONLY (train_two_si_objects.pairs_for_arm)",
        }
    with private_umask():
        tmp = out_dir / (PAIRS_META_NAME + ".tmp")
        tmp.write_text(json.dumps(sidecar, indent=1, sort_keys=True))
        tmp.replace(out_dir / PAIRS_META_NAME)
    print(f"[emit-pairs] sidecar -> {out_dir / PAIRS_META_NAME} "
          f"(regime={regime}, sizes: {sizes}, ungraded skipped: {n_ungraded})")


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="PIECE A: freeze the counts-only manifest")
    m.add_argument("--config", default="configs/two_si.yaml")
    m.add_argument("--private-root", default=None,
                   help="override the private store root (generated pos/benign rows)")
    m.add_argument("--out", default=None,
                   help="manifest path override (default cfg['pool_manifest'])")
    m.add_argument("--families", default=None,
                   help="comma-separated subset of the committed static_families "
                        "to freeze (decision a: narrow the pool to the families "
                        "actually generated when a generator sub-env is missing). "
                        "MUST be a subset of the committed set — the dropped "
                        "families are recorded as skipped-not-attempted in the "
                        "manifest. Default: the committed eight.")
    m.add_argument("--allow-engine-mix", action="store_true",
                   help="record (instead of refuse) a pool whose generation "
                        "legs mix engines — ONLY for the sanctioned §WS7 HF "
                        "contingency; the mix is embedded in the manifest as "
                        "a flagged confound")
    m.add_argument("--allow-unrecorded-provenance", action="store_true",
                   help="record (instead of refuse) leg(s) that hold "
                        "generation rows but no provenance sidecar (e.g. a "
                        "labelled store generated before sidecar recording). "
                        "Their engine is UNKNOWN; the manifest embeds them "
                        "as an explicit unrecorded-provenance confound "
                        "rather than silently exempting their rows from the "
                        "engine-mix check")

    g = sub.add_parser("generate-pos",
                       help="PIECE B (GPU): generate one SI's pos gaps "
                            "UNGRADED (no API key needed; grade on the grading host)")
    g.add_argument("--config", default="configs/two_si.yaml")
    g.add_argument("--si", default="si_constitution",
                   help="which SI's pos rows to generate (any config si_id "
                        "except the store's own prompted SI)")
    g.add_argument("--candidates", type=int, default=None,
                   help="rejection-sampling candidates per prompt (K, fixed "
                        "decision 8). Default: pool.pos_candidates.<si>, "
                        "else 1. Raise it until the SI reaches parity — the "
                        "manifest reports the realised K and per-prompt yield")
    g.add_argument("--private-root", default=None,
                   help="private store root to WRITE the pos rows under — pass a "
                        "shared-storage 0700 dir on the cluster; the default "
                        "~/.local/state is node-local /home there (D47 audit)")
    g.add_argument("--dry-run", action="store_true",
                   help="report gap counts only; load no model (CPU-only)")
    g.add_argument("--batch-size", type=int, default=2,
                   help="generation batch size — hf engine only (small: the "
                        "~6k-token long-SI prefill is memory-heavy, D9/D35); "
                        "the vllm engine batches per candidate block itself")
    g.add_argument("--engine", choices=list(_ENGINES), default=None,
                   help="generation engine (default: pool.engine, else hf — "
                        "the pre-ruling behaviour). The sweep campaign runs "
                        "vllm (engine ruling 2026-08-25); hf is the retained "
                        "per-model contingency and is provenance-recorded as "
                        "an engine confound")
    g.add_argument("--seed", type=int, default=0,
                   help="vllm engine base seed; candidate ci draws at "
                        "seed + 1000*ci (resume-stable)")
    g.add_argument("--max-model-len", type=int, default=16384,
                   help="vllm context: covers the ~6k-token SI prefix + the "
                        "longest attack prompt + the completion cap")
    g.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    g.add_argument("--enforce-eager", action="store_true")
    g.add_argument("--max-prompts", type=int, default=None,
                   help="cap generation to a LENGTH-REPRESENTATIVE slice of "
                        "the held_in harmful-base prompts (evenly spaced "
                        "over the prompt-length ordering; deterministic, so "
                        "a resume selects the same slice) — the pilot's "
                        "small-pool knob (pod_pilot.sh step 5)")
    g.add_argument("--pilot-pairs-out", default=None,
                   help="also emit the UNGRADED {prompt,pos,neg,"
                        "instruction_id} pairs for the selected prompts to "
                        "this jsonl (pilot-only: pos = the lowest-index "
                        "generated candidate, neg = the store's undefended "
                        "response; rows stamped ungraded + pilot_only). "
                        "Written mode-0600 under a 0700 parent; MUST be "
                        "outside the repo")

    gr = sub.add_parser("grade-pos",
                        help="PIECE C (the grading host): grade the ungraded pos rows "
                             "generate-pos wrote (live Low-Prio StrongREJECT)")
    gr.add_argument("--config", default="configs/two_si.yaml")
    gr.add_argument("--si", default=None,
                    help="grade only this SI's pos run (default: every pos run)")
    gr.add_argument("--private-root", default=None,
                    help="private store root the pos rows were written under "
                         "(the same value passed to generate-pos)")
    gr.add_argument("--dry-run", action="store_true",
                    help="report how many rows need grading; call no API")
    gr.add_argument("--concurrency", type=int, default=200,
                    help="peak concurrent live grader requests (Low-Prio). The "
                         "default (200) matches the repo's other live graders "
                         "and the standing wide-execution rule; grade-pos "
                         "PRINTS the ready-to-post coordination line at "
                         "launch when the site names a channel "
                         "(SISTEER_COORDINATION_CHANNEL) and proceeds "
                         "(print-and-proceed — the post is a "
                         "courtesy, never a gate; on 429s the grader "
                         "throttles its width down on the same org)")
    gr.add_argument("--allow-unscored", type=int, default=0,
                    help="tolerate up to N items the judge can NEVER score "
                         "(un-gradeable content surviving max_retries). They "
                         "are DROPPED, never scored 0: their rows keep "
                         "score=None and stay honestly ungraded (the three-"
                         "state rule), so they are excluded from the pool "
                         "rather than entering it with a fabricated refusal. "
                         "Default 0 = the historical loud refusal (JB1); the "
                         "same knob grade-store/grade-anti-neg already expose "
                         "(build_grader reads it), wired here too so a handful "
                         "of un-gradeable pos completions cannot block a whole "
                         "22k-row grade. Loud warning whenever it fires.")

    ga = sub.add_parser(
        "generate-anti-neg",
        help="PIECE B'' (GPU): elicit bidpo's anti-safety negatives for EVERY "
             "harmful prompt under safety_refusal.anti_text, UNGRADED (K "
             "rejection-sampling candidates; grade on the grading host with "
             "grade-anti-neg). Requires pool.anti_neg.enabled.")
    ga.add_argument("--config", default="configs/two_si.yaml")
    ga.add_argument("--candidates", type=int, default=None,
                    help="rejection-sampling candidates per prompt (K). Default: "
                         "pool.anti_neg.candidates. Only the first candidate the "
                         "grader scores a genuine compliance (> threshold) is "
                         "accepted; raise K until the per-prompt yield suffices.")
    ga.add_argument("--private-root", default=None,
                    help="private store root to WRITE the anti-neg rows under — "
                         "a shared-storage 0700 dir on the cluster (the default "
                         "~/.local/state is node-local /home there, D47)")
    ga.add_argument("--dry-run", action="store_true",
                    help="report the cell counts only; load no model (CPU-only)")
    ga.add_argument("--engine", choices=list(_ENGINES), default=None,
                    help="must resolve to vllm — the anti-neg leg has no HF "
                         "path (loud refusal otherwise)")
    ga.add_argument("--seed", type=int, default=0)
    ga.add_argument("--max-model-len", type=int, default=16384,
                    help="vLLM context: covers the short anti prefix + the "
                         "longest attack prompt + completion cap")
    ga.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ga.add_argument("--enforce-eager", action="store_true")
    ga.add_argument("--shard-index", type=int, default=0,
                    help="this worker's contiguous slice of the harmful-"
                         "prompt inventory (the pool_sample sample "
                         "convention). Default 0/1 = the whole inventory, "
                         "exactly the single-GPU behaviour; a sharded worker "
                         "writes its own anti_neg_shard_NN.jsonl")
    ga.add_argument("--shard-count", type=int, default=1)

    gaa = sub.add_parser(
        "generate-anti-neg-all",
        help="Fan out one generate-anti-neg worker per GPU (the pool_sample "
             "sample-all pattern): one vLLM engine per GPU, a disjoint "
             "contiguous prompt shard and its own output file each; resume "
             "per cell across the whole run.")
    gaa.add_argument("--config", default="configs/two_si.yaml")
    gaa.add_argument("--candidates", type=int, default=None,
                     help="rejection-sampling candidates per prompt (K); "
                          "default pool.anti_neg.candidates")
    gaa.add_argument("--private-root", default=None,
                     help="private store root to WRITE the anti-neg rows "
                          "under (passed through to every worker)")
    gaa.add_argument("--engine", choices=list(_ENGINES), default=None,
                     help="must resolve to vllm (passed through to every "
                          "worker); the anti-neg leg has no HF path")
    gaa.add_argument("--seed", type=int, default=0)
    gaa.add_argument("--max-model-len", type=int, default=16384)
    gaa.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    gaa.add_argument("--enforce-eager", action="store_true")
    gaa.add_argument("--gpus", default=None,
                     help="comma-separated GPU ids (default: "
                          "CUDA_VISIBLE_DEVICES if set, else nvidia-smi — "
                          "children always get a strict subset)")
    gaa.add_argument("--shard-offset", type=int, default=0,
                     help="this pod's first GLOBAL shard index "
                          "(multi-pod fan-out)")
    gaa.add_argument("--shard-total", type=int, default=None,
                     help="total GPUs across ALL pods (default: local GPU "
                          "count)")

    gar = sub.add_parser(
        "grade-anti-neg",
        help="PIECE C'' (the grading host): grade bidpo's anti-safety negatives with the "
             "live Low-Prio StrongREJECT judge (accept a candidate as a genuine "
             "compliance iff score > threshold)")
    gar.add_argument("--config", default="configs/two_si.yaml")
    gar.add_argument("--private-root", default=None,
                     help="private store root the anti-neg rows were written "
                          "under (the same value passed to generate-anti-neg)")
    gar.add_argument("--dry-run", action="store_true",
                     help="report how many rows need grading; call no API")
    gar.add_argument("--concurrency", type=int, default=4,
                     help="peak concurrent live grader requests (Low-Prio); "
                          "same coordination rule as grade-pos (>5 needs the "
                          "coordination-channel post)")
    gar.add_argument("--allow-unscored", type=int, default=0,
                     help="tolerate up to N items the judge can NEVER score "
                          "(un-gradeable content surviving max_retries). They "
                          "are DROPPED, never scored 0: their rows keep "
                          "score=None and stay honestly ungraded, so they are "
                          "excluded from the pool rather than entering it with "
                          "a fabricated verdict. Default 0 = the historical "
                          "loud refusal (JB1). Loud warning whenever it fires.")

    gb = sub.add_parser("generate-benign",
                        help="PIECE B' (GPU): generate the benign half's none "
                             "+ per-SI responses UNGRADED (PHTest, decisions "
                             "1-3; label on the grading host with grade-benign)")
    gb.add_argument("--config", default="configs/two_si.yaml")
    gb.add_argument("--phtest-path", default=None,
                    help="local PHTest CSV override (tests/offline; default: "
                         "pool.benign_path, else the pinned hub cache)")
    gb.add_argument("--private-root", default=None,
                    help="private store root to WRITE the benign rows under")
    gb.add_argument("--dry-run", action="store_true",
                    help="report missing (prompt, condition) counts only; "
                         "load no model (CPU-only)")
    gb.add_argument("--batch-size", type=int, default=2,
                    help="generation batch size (hf engine only)")
    gb.add_argument("--engine", choices=list(_ENGINES), default=None,
                    help="generation engine (default: pool.engine, else hf); "
                         "both engines decode GREEDILY here, so there is no "
                         "sampler parity to keep — see generate-pos for the "
                         "sampled leg")
    gb.add_argument("--seed", type=int, default=0,
                    help="vllm engine seed (greedy decode: engine init only)")
    gb.add_argument("--max-model-len", type=int, default=16384)
    gb.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    gb.add_argument("--enforce-eager", action="store_true")

    gs = sub.add_parser(
        "grade-store",
        help="Grade the labelled STORE itself (the pool_sample sample-all -> "
             "merge output): StrongREJECT scores for every harmful-family row "
             "under every pool condition (none / SI / anti pseudo-SI), XSTest "
             "C/P/I labels for the benign SI rows. Live Low-Prio; resume-safe "
             "(rows with a score/label are skipped + on-disk ScoreCache); "
             "atomic per-file rewrites.")
    gs.add_argument("--config", default="configs/two_si.yaml")
    gs.add_argument("--store-dir", default=None,
                    help="store dir override (default: TT_TWO_SI_STORE_DIR, "
                         "else pool.store_dir)")
    gs.add_argument("--dry-run", action="store_true",
                    help="report the ungraded counts per family and side "
                         "(JSON); call no API — use this to size the "
                         "coordination post before spending anything")
    gs.add_argument("--concurrency", type=int, default=200,
                    help="peak concurrent live grader requests (Low-Prio). "
                         "The default (200) is far past the >5-concurrent "
                         "threshold: post to the site's coordination "
                         "channel before a real run.")
    gs.add_argument("--allow-unscored", type=int, default=0,
                    help="tolerate up to N items the judge can NEVER score "
                         "(un-gradeable content surviving max_retries). They "
                         "are DROPPED, never scored 0: their rows keep "
                         "score=None and stay honestly ungraded, so they are "
                         "excluded from the pool rather than entering it with "
                         "a fabricated verdict. Default 0 = the historical "
                         "loud refusal (JB1). Loud warning whenever it fires. "
                         "Sized for a long run: grade-anti-neg hit 5 "
                         "permanently un-gradeable items in 126k rows.")

    gbr = sub.add_parser("grade-benign",
                         help="PIECE C' (the grading host): label the benign SI rows "
                              "with the live XSTest C/P/I judge")
    gbr.add_argument("--config", default="configs/two_si.yaml")
    gbr.add_argument("--private-root", default=None,
                     help="private store root the benign rows were written under")
    gbr.add_argument("--dry-run", action="store_true",
                     help="report how many rows need labels; call no API")
    gbr.add_argument("--concurrency", type=int, default=4,
                     help="peak concurrent live judge requests (Low-Prio); "
                          "same coordination rule as grade-pos")

    e = sub.add_parser("emit-pairs",
                       help="PIECE D: turn the FROZEN manifest into the per-SI "
                            "{prompt,pos,neg} jsonls train_two_si_objects.py "
                            "consumes (written under a mode-0700 dir, never "
                            "the repo)")
    e.add_argument("--config", default="configs/two_si.yaml")
    e.add_argument("--manifest", default=None,
                   help="frozen manifest path (default cfg['pool_manifest']); "
                        "its embedded manifest_sha256 is verified before any "
                        "pair is emitted")
    e.add_argument("--private-root", default=None,
                   help="private store root the generated pos/benign rows live "
                        "under (the same value passed to the generate/grade "
                        "steps)")
    e.add_argument("--out-dir", default=None,
                   help="dir for the per-SI pairs jsonls + pairs_meta.json "
                        "sidecar (created mode 0700; MUST be outside the "
                        "repo). Default: <private-root>/<pool.pairs_run> "
                        "(two_si_pairs)")
    e.add_argument("--dry-run", action="store_true",
                   help="report the regime, per-SI pair counts and the "
                        "ungraded-skip count without writing anything")

    return ap


def main() -> None:
    args = build_parser().parse_args()
    cfg = jailbreak_run._load_cfg(args.config)
    {"manifest": cmd_manifest, "generate-pos": cmd_generate_pos,
     "grade-pos": cmd_grade_pos, "generate-benign": cmd_generate_benign,
     "grade-benign": cmd_grade_benign, "grade-store": cmd_grade_store,
     "generate-anti-neg": cmd_generate_anti_neg,
     "generate-anti-neg-all": cmd_generate_anti_neg_all,
     "grade-anti-neg": cmd_grade_anti_neg,
     "emit-pairs": cmd_emit_pairs}[args.cmd](args, cfg)


if __name__ == "__main__":
    # A credential rejection that reaches main -- the shared judge's / grader's typed
    # error -- ends the process with the typed text as its terminal line, never a traceback
    from sisteer import vault  # noqa: E402

    with vault.credential_exit(prefix="[two-si-pool] "):
        main()
