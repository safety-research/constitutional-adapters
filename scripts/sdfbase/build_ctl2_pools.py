#!/usr/bin/env python
"""Build the r2 register-matched CONTROL pools for sdf-base-steering round 2:
a control HELD-OUT split carved from the r2 control finals with the
constitutional split's own machinery, then two pools, both under the NEW
``ctl2-`` doc-id prefix / ``constv2ctl2-`` cell prefix — never ``ctl-`` /
``constv2ctl-``: the r2 corpus re-mints production's id grammar AND the
earlier control run's, so the old prefix would alias two different
documents under one id.

- ``constv2ctl2-heldout``: the frozen 128-doc eval subset of the control
  held-out set, rendered whole — the ``constv2-heldout`` twin (same salts,
  same 5% whole-group carve, same eval-subset rule, same size class), NOT
  the remainder-of-a-draw the old ``constv2ctl-heldout`` was.
- ``constv2ctl2-full``: ONE seeded natural-ratio draw of the control TRAIN
  side (``sha_shuffle`` under the control draw salt +
  ``greedy_interleave`` at the train side's own advice:stories token
  share), cut as a NESTED PREFIX at EXACTLY the constitutional
  ``constv2-mix-full`` cell's official token count — read from that cell's
  pool manifest, never typed — so ctl2-full vs mix-full is budget-matched.
  The unused tail of the draw is recorded in the manifest
  (``remainder``: prefixed ids in draw order + token totals), so
  prefix ⊕ remainder is the complete train-side order and a later
  "use all 45.8M" decision is a manifest read, not a re-draw.

The release's canonical input bytes are PINNED in code (RELEASE_SHAS) and
asserted before a record is read; other bytes build only under
``--unpinned-*-sha`` and every manifest then says ``release_bytes_pinned:
false``. The output dir must be fresh (refuse_nonempty_out), and the
``verify`` subcommand re-reads each pool against its manifest (sha, row
count, ordered ids) — run it after the preflight, which rewrites manifests.

Everything else is build_pools.py's machinery, CALLED, not copied:
``make_doc`` (renderer byte-parity with the corpus repo, chat-markup and
canary checks), ``sha_shuffle`` / ``greedy_interleave`` / ``cut_cell``
(authoritative set-rule budget, forward-scan exact truncation, the
boundary skip rule) and ``build_cell`` (pool rows + manifest; the
record-not-refuse ratio gate for the control lane). Both input finals are
sha256-verified against the shas the release named BEFORE anything is
read (a mismatch refuses), and a dirty generator tree refuses like every
sdfbase builder (``--allow-dirty`` records dirty=true + the tree-diff sha).

Outputs (``--out``, a FRESH directory — never the shared pools dir, so no
finished cell's manifest is restamped):
  pairs_constv2ctl2-heldout.jsonl + .manifest.json
  pairs_constv2ctl2-full.jsonl    + .manifest.json
  pools_summary.json
plus the split artifact (``--split-out``, default
data/sdfbase/ctl2_heldout_split.json, ids only, + .provenance.json).
Both manifests and the summary carry ``recipe`` (``recipe_identity``): the
semantic identity of what was sampled and how — CTL2_RECIPE_VERSION, the
control-input shas, the id prefix, the salts, the split parameters, the
tokenizer pins — never the checkout's commit; ``ctl2_pools_reuse.py``
refuses to reuse a completed build whose recipe is not today's builder's.

Then, as for every pool: ``build_pools.py preflight --pools <out>``
(max-length under both pinned target tokenizers), the tokcache sidecar
of the full pool per model (train_two_si_objects.py precompute-tokens),
and the volume puts under pool/<key>/.

  python scripts/sdfbase/build_ctl2_pools.py build \\
      --control-advice  .../final-control-scale-20260903-r2-advice.jsonl \\
      --control-stories .../final-control-scale-20260903-r2-stories.jsonl \\
      --budget-manifest $STAGE/sdfbase/pools/pairs_constv2-mix-full.manifest.json \\
      --out $STAGE/sdfbase/pools_ctl2
  python scripts/sdfbase/build_pools.py preflight --pools $STAGE/sdfbase/pools_ctl2
  python scripts/sdfbase/build_ctl2_pools.py verify --pools $STAGE/sdfbase/pools_ctl2
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import build_heldout_split as hsplit  # noqa: E402
import build_pools as bp  # noqa: E402

#: The r2 control lane's doc-id prefix — applied AFTER ranking and
#: selection (which use the original ids, like the earlier ``ctl-``), so
#: rows, manifests, truncation and skip records all carry it.
CTL2_ID_PREFIX = "ctl2-"
FULL_CELL = "constv2ctl2-full"
HELDOUT_CELL = "constv2ctl2-heldout"
#: build_cell lanes: "control" gives the full cell the control lane's
#: semantics (ratio gate applied at >=100k, record-not-refuse);
#: "control_heldout" is the old constv2ctl-heldout's lane string.
LANE_FULL = "control"
LANE_HELDOUT = "control_heldout"
#: The constitutional cell whose official token count is the budget.
BUDGET_CELL = "constv2-mix-full"
CORPUS_RELEASE = "control-scale-20260903-r2"
#: The release's canonical bytes (ConstDataGen release doc
#: the control-scale r2 release notes).
#: ALWAYS enforced: a pool that carries the release's name was cut from
#: these bytes and nothing else. Other inputs (tests, a future release)
#: go through ``expect_shas`` and are labelled ``unpinned`` in every
#: manifest — the label can never claim a release the bytes are not.
RELEASE_SHAS = {
    "control_advice":
        "ba89cf60175f54347022df63d89214983855b5f726a8ea16427ba3797a3abae7",
    "control_stories":
        "915129b2b89eb00bda0a0ca13923708d6bb7cff3b4332d67f69c073555ab6dad",
}
SPLIT_OUT = hsplit.WORKTREE_ROOT / "data/sdfbase/ctl2_heldout_split.json"
#: The ctl2 RECIPE version. Bump it BY HAND whenever the sampling / cut /
#: tokenization ALGORITHM changes — ``sha_shuffle``, ``greedy_interleave``,
#: the nested-prefix cut and ``cut_cell``'s set rule, ``make_doc``'s
#: rendering, how the id prefix is applied, the split carve — i.e. anything
#: that would change the SELECTION or the pool bytes with every input named
#: in ``recipe_identity`` held fixed. A metadata-only change (a docstring,
#: the generator provenance, a new manifest field) does NOT bump it: a
#: completed build must survive a checkout that merely changed metadata,
#: which is why ``ctl2_pools_reuse.py`` compares this identity and never the
#: checkout's commit sha.
CTL2_RECIPE_VERSION = "ctl2-recipe-v1"
#: The reference-tokenizer pins the staged tokenizers are verified against
#: (repo, HF revision, per-file sha256 — the harness's pins file at the repo
#: root; ``corpusgen stage-tokenizers`` refuses a family that does not reproduce
#: them). Part of the recipe: the official token count is the minimum over
#: these three tokenizers, so a different pin is a different cut.
TOKENIZER_PINS = _HERE.parents[1] / "corpora/TOKENIZERS.json"


def load_tokenizer_pins(path=None) -> dict:
    """The tokenizer pins as the recipe records them: ``{"pins_file": <repo-
    relative path>, "pins_sha256": <the file's sha256>, "pins": {name: {"repo",
    "revision", "files": {filename: sha256}}}}``. ``pins`` is the SEMANTIC
    identity (compared by ``ctl2_pools_reuse.py``); ``pins_sha256`` is
    recorded provenance of the file the pins were read from (an edit to the
    file's ``_doc`` alone must not invalidate a completed build). Refuses a
    missing, unreadable or malformed pins file — the recipe is never recorded
    with the tokenizers unknown."""
    p = Path(path) if path else TOKENIZER_PINS
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise bp.SamplerError(
            f"tokenizer pins {p}: unreadable ({e}) — the ctl2 recipe records "
            "the reference-tokenizer pins the staged tokenizers are verified "
            "against, and cannot be recorded without them")
    fams = doc.get("tokenizers") if isinstance(doc, dict) else None
    if not isinstance(fams, dict) or not fams:
        raise bp.SamplerError(f"tokenizer pins {p}: no 'tokenizers' map")
    pins: dict[str, dict] = {}
    for name in sorted(fams):
        t = fams[name]
        files = t.get("files") if isinstance(t, dict) else None
        ok = (isinstance(t, dict) and isinstance(t.get("repo"), str)
              and isinstance(t.get("revision"), str) and isinstance(files, list)
              and all(isinstance(f, dict) and isinstance(f.get("name"), str)
                      and isinstance(f.get("sha256"), str) for f in files))
        if not ok:
            raise bp.SamplerError(
                f"tokenizer pins {p}: family {name!r} is not "
                "{repo, revision, files: [{name, sha256}]}")
        pins[name] = {"repo": t["repo"], "revision": t["revision"],
                      "files": {f["name"]: f["sha256"]
                                for f in sorted(files, key=lambda f: f["name"])}}
    root = _HERE.parents[1]
    rel = (str(p.resolve().relative_to(root)) if p.resolve().is_relative_to(root)
           else _tilde(p))
    return {"pins_file": rel, "pins_sha256": hsplit.sha256_file(p), "pins": pins}


def recipe_identity(*, shas: dict, pinned: bool, release_label: str,
                    salt: str, eval_salt: str, frac: float, eval_n: int,
                    tokenizer_pins: dict) -> dict:
    """The STABLE SEMANTIC identity of a ctl2 build's recipe — everything
    that determines the control SELECTION and the pool bytes besides the
    budget (which the full cell's ``budget_source`` records separately):
    the algorithm version, the control-input bytes (the release label, its
    pinned flag and the two corpus shas), the id prefix, the two cells and
    the budget cell, the three salts (the held-out carve, the eval subset,
    the full cell's draw), the split parameters and the tokenizer pins.
    NEVER the checkout's commit: a metadata-only change must not invalidate
    completed work. Recorded as ``recipe`` on BOTH cell manifests and in
    ``pools_summary.json``; ``ctl2_pools_reuse.py`` compares a recorded
    build's block against this checkout's declared one before any reuse."""
    return {
        "version": CTL2_RECIPE_VERSION,
        "release": release_label,
        "release_bytes_pinned": bool(pinned),
        "corpus_shas": dict(shas),
        "id_prefix": CTL2_ID_PREFIX,
        "cells": [HELDOUT_CELL, FULL_CELL],
        "budget_cell": BUDGET_CELL,
        "salts": {"heldout": salt, "eval": eval_salt, "draw": bp.SALT_CTL},
        "split": {"frac": frac, "eval_n": eval_n},
        "tokenizers": copy.deepcopy(tokenizer_pins),
    }


def _tilde(p) -> str:
    """Paths in manifests: home-relative, never an absolute home path."""
    s = str(p)
    home = str(Path.home())
    return "~" + s[len(home):] if s.startswith(home + "/") else s


def verify_sha(path: Path, expected: str, tag: str) -> str:
    """sha256 of ``path``; REFUSES when it differs from ``expected`` — the
    bytes a pool is cut from are asserted BEFORE a record is read, never
    discovered afterwards. ``expected`` is required (the release pins by
    default; a caller naming other bytes is labelled unpinned)."""
    if not expected:
        raise bp.SamplerError(f"[{tag}] no expected sha256 for {path} — "
                              "every input's bytes must be asserted")
    got = hsplit.sha256_file(Path(path))
    if got != expected.strip().lower():
        raise bp.SamplerError(
            f"[{tag}] {path} hashes to {got} but {expected.strip().lower()} "
            "was asserted — refusing to build from bytes that were not "
            "named")
    return got


def refuse_nonempty_out(out_dir: Path) -> None:
    """The output dir is FRESH by contract: a build into the shared pools
    dir would replace its pools_summary.json with a two-cell table, and a
    rerun into a half-written dir could pair new pool bytes with stale
    manifests. Any existing file refuses (an empty dir is fine)."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise bp.SamplerError(
            f"--out {out_dir} is not empty — the ctl2 build writes a FRESH "
            "directory (never the shared pools dir, never over a previous "
            "build); remove it or name another")


def read_budget(manifest_path: Path) -> dict:
    """The budget the control cell is matched to: the constitutional
    ``constv2-mix-full`` pool manifest's ``official_tokens``.
    Refuses a manifest of any other cell — the budget must be READ from
    the cell it matches, never typed."""
    mp = Path(manifest_path)
    if not mp.is_file():
        raise bp.SamplerError(f"budget manifest {mp} missing — the control "
                              f"cell is budget-matched to {BUDGET_CELL}'s "
                              "pool manifest")
    m = json.loads(mp.read_text(encoding="utf-8"))
    if m.get("cell") != BUDGET_CELL:
        raise bp.SamplerError(
            f"{mp} is the manifest of cell {m.get('cell')!r}, not "
            f"{BUDGET_CELL!r} — the control budget is that cell's official "
            "token count and nothing else")
    budget = m.get("official_tokens")
    if not isinstance(budget, int) or budget <= 0:
        raise bp.SamplerError(f"{mp}: official_tokens={budget!r} is not a "
                              "positive integer")
    return {
        "cell": m["cell"],
        "manifest": _tilde(mp),
        "manifest_sha256": hsplit.sha256_file(mp),
        "pool_sha256": m.get("pool_sha256"),
        "official_tokens": budget,
        "limiting_tokenizer": m.get("limiting_tokenizer"),
        "per_tokenizer_cumulative_totals":
            m.get("per_tokenizer_cumulative_totals"),
        "n_docs": m.get("n_docs"),
    }


def _fam_totals(docs) -> dict[str, int]:
    return bp._token_totals(docs)


def _fam_counts(docs) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in docs:
        out[d.family] = out.get(d.family, 0) + 1
    return out


def _edited(recs: list[dict]) -> int:
    return sum(1 for r in recs if (r.get("lineage") or {}).get("register_edits"))


def verify_control_shas(control_advice, control_stories,
                        expect_shas: dict | None = None) -> tuple[dict, bool, str]:
    """Assert both control finals' bytes BEFORE anything is read. Returns
    ``(shas, pinned, release_label)``: ``pinned`` is whether the bytes are
    the release's (RELEASE_SHAS); the label never claims the release for
    other bytes."""
    expect_shas = dict(RELEASE_SHAS if expect_shas is None else expect_shas)
    shas = {
        "control_advice": verify_sha(Path(control_advice),
                                     expect_shas.get("control_advice", ""),
                                     "ctl2-advice"),
        "control_stories": verify_sha(Path(control_stories),
                                      expect_shas.get("control_stories", ""),
                                      "ctl2-stories"),
    }
    pinned = shas == RELEASE_SHAS
    release_label = (CORPUS_RELEASE if pinned
                     else f"unpinned({CORPUS_RELEASE}-shaped inputs, bytes "
                          "not the release's)")
    return shas, pinned, release_label


def write_ctl2_split(split: dict, split_out, *, shas: dict, pinned: bool,
                     release_label: str, generator: dict) -> str:
    """Write the control split artifact + its ``.provenance.json`` sidecar
    (the constitutional sidecar's shape via ``hsplit.split_provenance``
    plus the release label / pinned flag / corpus shas). Returns the
    artifact's sha256.

    Idempotent on an UNCHANGED artifact: when ``split_out`` already holds
    exactly these bytes and its sidecar exists, neither file is touched —
    the sidecar keeps the provenance of the tree that first produced the
    bytes (a later build merely reproduced them; the seed replicates
    build their splits in one committed step and the pools in another).
    Different bytes overwrite both, as before."""
    split_out = Path(split_out)
    body = json.dumps(split, indent=1, ensure_ascii=False) + "\n"
    prov_path = split_out.with_suffix(".provenance.json")
    if (split_out.is_file() and prov_path.is_file()
            and split_out.read_text(encoding="utf-8") == body):
        return hsplit.sha256_file(split_out)
    split_out.parent.mkdir(parents=True, exist_ok=True)
    split_out.write_text(body, encoding="utf-8")
    split_sha = hsplit.sha256_file(split_out)
    prov = hsplit.split_provenance(split, artifact=split_out,
                                   generator=generator)
    prov.update({"corpus": release_label, "release_bytes_pinned": pinned,
                 "corpus_shas": shas})
    prov_path.write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n",
                         encoding="utf-8")
    return split_sha


def build_ctl2(out_dir, *, control_advice, control_stories, budget_manifest,
               expect_shas: dict | None = None, rig=None,
               generator: dict | None = None, split_out=None,
               frac: float = hsplit.FRAC, eval_n: int = hsplit.EVAL_N,
               salt: str = hsplit.SALT_HELDOUT,
               eval_salt: str = hsplit.SALT_EVAL,
               register_edit_manifests: dict | None = None,
               tokenizers_dir=None) -> dict:
    """The whole build; returns {cell: manifest}. ``expect_shas``
    ({"control_advice": sha, "control_stories": sha}) defaults to the
    RELEASE's pinned bytes and is enforced BEFORE anything is read; a
    caller naming other bytes gets every manifest labelled
    ``release_bytes_pinned: false`` / ``release: unpinned(...)`` — the
    release label is never claimed for bytes the release did not ship.
    ``register_edit_manifests`` names the release's per-corpus edit
    manifests (recorded by sha when present, null when the release has not
    written one yet). ``salt`` / ``eval_salt`` thread into the control
    held-out carve (default: the campaign's constitutional salts; the
    seed replicates pass ``sdfbase-heldout-v1-s<s>`` /
    ``sdfbase-heldout-eval-v1-s<s>``). ``out_dir`` must be fresh
    (refuse_nonempty_out)."""
    out_dir = Path(out_dir)
    control_advice = hsplit.require_final(control_advice, "--control-advice")
    control_stories = hsplit.require_final(control_stories, "--control-stories")
    refuse_nonempty_out(out_dir)
    shas, pinned, release_label = verify_control_shas(
        control_advice, control_stories, expect_shas)
    budget_src = read_budget(budget_manifest)
    budget = budget_src["official_tokens"]

    tokens_mod = hsplit.generator_tokens()
    rig = rig or bp.StagedCountRig(tokenizers_dir)
    generator = generator or hsplit.generator_provenance()
    # The recipe identity (recipe_identity): recorded on BOTH cell manifests
    # and the summary, so a later `02 --step pools` can tell a completed
    # build cut under another salt / pin / release from today's builder's.
    # The staged rig's own provenance (the MANIFEST.json corpusgen stage-tokenizers
    # wrote: source repo, snapshot, per-file sha256 — deterministic) rides
    # along under tokenizers.rig as provenance, beside the pins it was
    # verified against. Computed BEFORE anything is written: a missing pins
    # file refuses with the output dir untouched.
    recipe = recipe_identity(shas=shas, pinned=pinned,
                             release_label=release_label, salt=salt,
                             eval_salt=eval_salt, frac=frac, eval_n=eval_n,
                             tokenizer_pins=load_tokenizer_pins())
    recipe["tokenizers"]["rig"] = copy.deepcopy(getattr(rig, "provenance", None))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. The control held-out split: the constitutional split's own
    #    machinery (whole A3 groups / whole subcategories, sha-ranked under
    #    the given salts — the constitutional ones by default — 5% by the
    #    authoritative set rule, 128-doc eval subset) on the control
    #    finals. Structure checks run inside.
    split = hsplit.build_split(control_advice, control_stories, frac=frac,
                               eval_n=eval_n, salt=salt, eval_salt=eval_salt)
    split_out = Path(split_out) if split_out else SPLIT_OUT
    split_sha = write_ctl2_split(split, split_out, shas=shas, pinned=pinned,
                                 release_label=release_label,
                                 generator=generator)

    # 2. Render every record through the corpus repo's renderer (byte
    #    parity asserted per doc; chat-markup + canary checks).
    adv_recs = hsplit.load_jsonl(control_advice)
    sto_recs = hsplit.load_jsonl(control_stories)
    adv_docs = [bp.make_doc(r, "advice", tokens_mod) for r in adv_recs]
    sto_docs = [bp.make_doc(r, "control_stories", tokens_mod)
                for r in sto_recs]
    by_id = {d.doc_id: d for d in adv_docs + sto_docs}
    if len(by_id) != len(adv_docs) + len(sto_docs):
        raise bp.SamplerError("duplicate doc ids across the control corpus")

    heldout_ids = set(split["advice"]["ids"]) | set(split["stories"]["ids"])
    eval_ids = list(split["eval_subset"]["ids"])
    if not set(eval_ids) <= heldout_ids:
        raise bp.SamplerError("internal: eval subset is not inside the "
                              "held-out set")
    train_adv = [d for d in adv_docs if d.doc_id not in heldout_ids]
    train_sto = [d for d in sto_docs if d.doc_id not in heldout_ids]
    train_ids = {d.doc_id for d in train_adv + train_sto}
    if train_ids & heldout_ids or len(train_ids) + len(heldout_ids) != len(by_id):
        raise bp.SamplerError("internal: held-out / train sides are not a "
                              "partition of the corpus")

    corpus_block = {
        "release": release_label,
        "release_bytes_pinned": pinned,
        "id_prefix": CTL2_ID_PREFIX,
        "files": {"control_advice": _tilde(control_advice),
                  "control_stories": _tilde(control_stories)},
        "corpus_shas": shas,
        "n_records": {"advice": len(adv_recs), "stories": len(sto_recs)},
        "official_tokens": {
            "advice": split["token_totals"]["advice"]["family_official"],
            "stories": split["token_totals"]["stories"]["family_official"],
        },
        "register_edited_records": {"advice": _edited(adv_recs),
                                    "stories": _edited(sto_recs)},
        "v1_lineage_records": {
            "advice": sum(1 for r in adv_recs
                          if (r.get("lineage") or {}).get("v1_sha256")),
            "stories": sum(1 for r in sto_recs
                           if (r.get("lineage") or {}).get("v1_sha256")),
        },
    }
    corpus_block["official_tokens"]["combined"] = sum(
        corpus_block["official_tokens"].values())
    split_block = {
        "artifact": _tilde(split_out),
        "sha256": split_sha,
        "salt": split["salt"],
        "frac": split["frac"],
        "eval_salt": split["eval_subset"]["salt"],
        "eval_n": split["eval_subset"]["n"],
        "eval_family_counts": split["eval_subset"]["family_counts"],
        **{f"{fam}_{k}": split["token_totals"][fam][k]
           for fam in ("advice", "stories")
           for k in ("heldout_records", "heldout_groups", "heldout_official",
                     "train_records", "train_official")},
    }
    rem_manifests = {}
    for fam, p in (register_edit_manifests or {}).items():
        rem_manifests[fam] = (
            {"path": _tilde(p), "sha256": hsplit.sha256_file(Path(p))}
            if p and Path(p).is_file() else None)
    constdatagen = {"release": release_label, "release_bytes_pinned": pinned,
                    "corpus_shas": shas,
                    "register_edit_manifests": rem_manifests}

    manifests: dict[str, dict] = {}

    # 3. The held-out EVAL pool: the eval subset, whole, prefixed.
    ev_docs = [replace(by_id[i], doc_id=CTL2_ID_PREFIX + i) for i in eval_ids]
    m_h = bp.build_cell(
        HELDOUT_CELL, LANE_HELDOUT, None, ev_docs, _fam_totals(ev_docs),
        salts={"draw": split["eval_subset"]["salt"],
               "heldout": split["salt"]},
        natural_ratio=None, out_dir=out_dir, rig=rig, split_sha=split_sha,
        corpus_shas=shas, generator=generator,
        constdatagen_provenance=constdatagen)
    m_h["corpus"] = corpus_block
    m_h["split"] = split_block
    m_h["heldout_set"] = {
        "n_docs": len(heldout_ids),
        "doc_ids": sorted(CTL2_ID_PREFIX + i for i in heldout_ids),
        "eval_subset_is_subset": True,
    }
    manifests[HELDOUT_CELL] = m_h

    # 4. The full cell: ONE natural-ratio draw of the train side, cut as a
    #    nested prefix at the constitutional full cell's exact count.
    t_adv = min(_fam_totals(train_adv).values())
    t_sto = min(_fam_totals(train_sto).values())
    natural = t_adv / (t_adv + t_sto)
    order = bp.greedy_interleave(
        {"advice": bp.sha_shuffle(train_adv, bp.SALT_CTL),
         "stories": bp.sha_shuffle(train_sto, bp.SALT_CTL)},
        {"advice": natural, "stories": 1.0 - natural})
    if len(order) != len(train_ids):
        raise bp.SamplerError("internal: the draw does not cover the train "
                              "side")
    docs, cum, skips = bp.cut_cell(order, budget, rig)
    prefix_raw = [d.doc_id for d in docs]
    n_walked = len(prefix_raw) + len(skips)  # docs consumed from the order
    if order[:len(prefix_raw)] != docs[:len(prefix_raw)] and not skips:
        # Whole-doc prefix except the (possibly truncated) boundary doc:
        # ids must match position by position.
        if [d.doc_id for d in order[:len(prefix_raw)]] != prefix_raw:
            raise bp.SamplerError("internal: the cut is not a prefix of the "
                                  "draw order")
    used = set(prefix_raw)
    remainder = [d for d in order if d.doc_id not in used]
    if len(remainder) + len(used) != len(order):
        raise bp.SamplerError("internal: prefix + remainder != train side")
    skipped_ids = {s["doc_id"] for s in skips}
    if not skipped_ids <= {d.doc_id for d in remainder}:
        raise bp.SamplerError("internal: a boundary-skipped doc left the "
                              "remainder")
    docs = [replace(d, doc_id=CTL2_ID_PREFIX + d.doc_id) for d in docs]
    for s in skips:
        s["doc_id"] = CTL2_ID_PREFIX + s["doc_id"]
    m_f = bp.build_cell(
        FULL_CELL, LANE_FULL, budget, docs, cum,
        salts={"draw": bp.SALT_CTL, "heldout": split["salt"]},
        natural_ratio=natural, out_dir=out_dir, rig=rig,
        split_sha=split_sha, corpus_shas=shas, generator=generator,
        boundary_skips=skips, ratio_gate_mode="record",
        constdatagen_provenance=constdatagen)
    rem_totals = _fam_totals(remainder)
    train_totals = _fam_totals(train_adv + train_sto)
    m_f["corpus"] = corpus_block
    m_f["split"] = split_block
    m_f["budget_source"] = budget_src
    m_f["train_side"] = {
        "n_docs": len(order),
        "per_tokenizer_totals": train_totals,
        "official_tokens": min(train_totals.values()),
        "family_counts": _fam_counts(order),
        "family_official_tokens": {"advice": t_adv, "stories": t_sto},
        "natural_ratio": natural,
    }
    m_f["nesting"] = {
        "rule": "the cell is a PREFIX of one deterministic draw of the "
                "train side (sha_shuffle + greedy_interleave), cut at the "
                "budget by the authoritative set rule; the boundary doc is "
                "truncated and counted as USED (its untrained tail is "
                "dropped, never leaked into the remainder)",
        "draw_salt": bp.SALT_CTL,
        "order_n_docs": len(order),
        "prefix_n_docs": len(prefix_raw),
        "docs_walked": n_walked,
        "truncated_boundary_doc": (m_f["truncation"] or {}).get("doc_id"),
        "boundary_skipped_docs": sorted(CTL2_ID_PREFIX + i
                                        for i in skipped_ids),
    }
    m_f["remainder"] = {
        "note": "the unused tail of the draw, NOT trained (R2-D48 (a): the "
                "control is budget-matched to constv2-mix-full; a decision "
                "to use the whole 45.8M is a longer prefix of this same "
                "order)",
        "n_docs": len(remainder),
        "doc_ids_in_order": [CTL2_ID_PREFIX + d.doc_id for d in remainder],
        "per_tokenizer_totals": rem_totals,
        "official_tokens": min(rem_totals.values()) if remainder else 0,
        "family_counts": _fam_counts(remainder),
    }
    pref_ids = set(m_f["doc_ids_in_order"])
    held_pref = {CTL2_ID_PREFIX + i for i in heldout_ids}
    rem_ids = set(m_f["remainder"]["doc_ids_in_order"])
    ev_pref = set(m_h["doc_ids_in_order"])
    m_f["disjointness"] = {
        "full_x_heldout": len(pref_ids & held_pref),
        "full_x_eval_subset": len(pref_ids & ev_pref),
        "full_x_remainder": len(pref_ids & rem_ids),
        "remainder_x_heldout": len(rem_ids & held_pref),
        "eval_subset_within_heldout": ev_pref <= held_pref,
        "full_plus_remainder_equals_train": (
            pref_ids | rem_ids == {CTL2_ID_PREFIX + i for i in train_ids}
            and len(pref_ids) + len(rem_ids) == len(train_ids)),
        "train_plus_heldout_equals_corpus": (
            len(train_ids) + len(heldout_ids) == len(by_id)),
    }
    if any(m_f["disjointness"][k] for k in ("full_x_heldout",
                                            "full_x_eval_subset",
                                            "full_x_remainder",
                                            "remainder_x_heldout")):
        raise bp.SamplerError(f"internal: disjointness violated: "
                              f"{m_f['disjointness']}")
    if not all(m_f["disjointness"][k] for k in (
            "eval_subset_within_heldout", "full_plus_remainder_equals_train",
            "train_plus_heldout_equals_corpus")):
        raise bp.SamplerError(f"internal: partition violated: "
                              f"{m_f['disjointness']}")
    manifests[FULL_CELL] = m_f

    for cell, m in manifests.items():
        m["recipe"] = copy.deepcopy(recipe)   # both cells: one build, one recipe
        (out_dir / f"pairs_{cell}.manifest.json").write_text(
            json.dumps(m, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")
    summary = {
        "corpus": release_label,
        "release_bytes_pinned": pinned,
        "heldout_split_sha": split_sha,
        "generator": generator,
        "recipe": copy.deepcopy(recipe),
        "cells": {
            cell: {
                "lane": m["lane"], "budget": m["budget"],
                "n_docs": m["n_docs"],
                "official_tokens": m["official_tokens"],
                "limiting_tokenizer": m["limiting_tokenizer"],
                "family_counts": m["family_counts"],
                "realized_ratio": m["realized_ratio"],
                "natural_ratio": m["natural_ratio"],
                "truncated_doc": (m["truncation"] or {}).get("doc_id"),
                "boundary_skips": [s["doc_id"] for s in m["boundary_skips"]],
                "official_loss_tokens": m["official_loss_tokens"],
                "pool_sha256": m["pool_sha256"],
            } for cell, m in manifests.items()},
    }
    (out_dir / "pools_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return manifests


def verify_pools(pools_dir: Path, cells=(HELDOUT_CELL, FULL_CELL)) -> dict:
    """Post-build / post-preflight check that each pool FILE is the one its
    manifest describes: sha256, row count and the ordered doc ids must all
    agree (the preflight rewrites manifests in place, so this runs after
    it too). Returns {cell: {...}}; any disagreement refuses."""
    pools_dir = Path(pools_dir)
    out: dict[str, dict] = {}
    for cell in cells:
        pool = pools_dir / f"pairs_{cell}.jsonl"
        mp = pools_dir / f"pairs_{cell}.manifest.json"
        if not pool.is_file() or not mp.is_file():
            raise bp.SamplerError(f"verify: {pool} or {mp} missing")
        m = json.loads(mp.read_text(encoding="utf-8"))
        sha = hsplit.sha256_file(pool)
        ids = []
        with open(pool, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("instruction_id") != cell:
                        raise bp.SamplerError(
                            f"verify: {pool.name} row {len(ids) + 1} carries "
                            f"instruction_id {row.get('instruction_id')!r}")
                    if not str(row.get("doc_id", "")).startswith(CTL2_ID_PREFIX):
                        raise bp.SamplerError(
                            f"verify: {pool.name} row {len(ids) + 1} doc_id "
                            f"{row.get('doc_id')!r} lacks the {CTL2_ID_PREFIX} "
                            "prefix")
                    ids.append(row["doc_id"])
        problems = []
        if sha != m.get("pool_sha256"):
            problems.append(f"pool sha256 {sha} != manifest {m.get('pool_sha256')}")
        if len(ids) != m.get("n_docs"):
            problems.append(f"rows {len(ids)} != manifest n_docs {m.get('n_docs')}")
        if ids != m.get("doc_ids_in_order"):
            problems.append("ordered doc ids differ from the manifest")
        if problems:
            raise bp.SamplerError(f"verify: {cell}: " + "; ".join(problems))
        out[cell] = {"pool_sha256": sha, "n_docs": len(ids),
                     "official_tokens": m.get("official_tokens")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="split + the two ctl2 pools")
    b.add_argument("--control-advice", type=Path, required=True)
    b.add_argument("--control-stories", type=Path, required=True)
    b.add_argument("--unpinned-advice-sha", default="",
                   help="build from NON-release advice bytes with this "
                        "sha256 (every manifest is labelled unpinned); "
                        "default: the release's pinned sha, enforced")
    b.add_argument("--unpinned-stories-sha", default="",
                   help="as above for the stories final")
    b.add_argument("--budget-manifest", type=Path,
                   default=bp.DEFAULT_OUT / f"pairs_{BUDGET_CELL}.manifest.json",
                   help=f"the {BUDGET_CELL} pool manifest whose "
                        "official_tokens is the budget")
    b.add_argument("--out", type=Path, default=Path("$STAGE/sdfbase/pools_ctl2"))
    b.add_argument("--split-out", type=Path, default=SPLIT_OUT)
    b.add_argument("--salt", default=hsplit.SALT_HELDOUT,
                   help="control held-out carve salt (seed replicates: "
                        f"{hsplit.SALT_HELDOUT}-s<s>); default = the "
                        "campaign's constitutional salt")
    b.add_argument("--eval-salt", default=hsplit.SALT_EVAL,
                   help="control eval-subset salt (seed replicates: "
                        f"{hsplit.SALT_EVAL}-s<s>)")
    b.add_argument("--register-edit-manifest", action="append", default=[],
                   metavar="FAMILY=PATH",
                   help="the release's per-corpus register-edit manifest "
                        "(recorded by sha; repeatable)")
    b.add_argument("--tokenizers-dir", type=Path, default=None,
                   help="the staged reference tokenizers (default "
                        "$CORPUS_GENERATOR_TOKENIZERS_DIR, else <corpus data "
                        "root>/tokenizers)")
    b.add_argument("--code-root", type=Path, default=None,
                   help="corpus_generator code root (default <repo>/corpus_generator, "
                        "or $CORPUS_GENERATOR_ROOT)")
    b.add_argument("--allow-dirty", action="store_true")
    v = sub.add_parser("verify", help="pool bytes/rows/ids vs manifests")
    v.add_argument("--pools", type=Path, default=Path("$STAGE/sdfbase/pools_ctl2"))
    args = ap.parse_args(argv)
    for _k in ("out", "pools", "budget_manifest"):   # the defaults sit under the staging directory, $STAGE
        if getattr(args, _k, None) is not None:
            setattr(args, _k, bp.site_path(getattr(args, _k), "--" + _k.replace("_", "-")))
    try:
        if args.cmd == "verify":
            rep = verify_pools(args.pools)
            for cell, r in rep.items():
                print(f"verify OK {cell}: {r['n_docs']} rows, official "
                      f"{r['official_tokens']:,}, sha {r['pool_sha256'][:16]}")
            return 0
        if args.code_root:
            hsplit.set_code_root(args.code_root)
        generator = hsplit.refuse_dirty_generator_tree(args.allow_dirty,
                                                      tag="build-ctl2-pools")
        rem = {}
        for spec in args.register_edit_manifest:
            fam, _, p = spec.partition("=")
            rem[fam] = p
        expect = None
        if args.unpinned_advice_sha or args.unpinned_stories_sha:
            if not (args.unpinned_advice_sha and args.unpinned_stories_sha):
                raise bp.SamplerError("--unpinned-*-sha: name BOTH inputs' "
                                      "bytes or neither")
            expect = {"control_advice": args.unpinned_advice_sha,
                      "control_stories": args.unpinned_stories_sha}
        manifests = build_ctl2(
            args.out, control_advice=args.control_advice,
            control_stories=args.control_stories,
            budget_manifest=args.budget_manifest,
            expect_shas=expect, generator=generator, split_out=args.split_out,
            salt=args.salt, eval_salt=args.eval_salt,
            register_edit_manifests=rem,
            tokenizers_dir=args.tokenizers_dir)
    except (bp.SamplerError, hsplit.SplitError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"release: {manifests[FULL_CELL]['corpus']['release']} "
          f"(bytes pinned: {manifests[FULL_CELL]['corpus']['release_bytes_pinned']})")
    for cell, m in manifests.items():
        trunc = (m["truncation"] or {}).get("doc_id")
        ratio = (f"{m['realized_ratio']:.4f}"
                 if m["realized_ratio"] is not None else "n/a")
        print(f"{cell}: {m['n_docs']} docs, official {m['official_tokens']:,} "
              f"({m['limiting_tokenizer']}), advice share {ratio}, "
              f"truncated={trunc}, families {m['family_counts']}, "
              f"pool sha {m['pool_sha256'][:16]}")
    mf = manifests[FULL_CELL]
    print(f"budget {mf['budget']:,} from {mf['budget_source']['cell']} "
          f"({mf['budget_source']['manifest']}); remainder "
          f"{mf['remainder']['n_docs']} docs / "
          f"{mf['remainder']['official_tokens']:,} official tokens; "
          f"disjointness {mf['disjointness']}")
    print(f"wrote pools + manifests under {args.out}; split at "
          f"{args.split_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
