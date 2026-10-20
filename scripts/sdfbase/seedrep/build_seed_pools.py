#!/usr/bin/env python
"""Per-seed held-out splits + pools for the Qwen-base seed replicates.

For seed ``s`` (key ``qwen35_4b_base_eb64_s<s>``), in order:

1. constitution split under the seed salts ``sdfbase-heldout-v1-s<s>`` /
   ``sdfbase-heldout-eval-v1-s<s>`` at the LANE fraction **0.085** with the
   contrastive probes' source documents pinned into held-out at group
   granularity (``build_heldout_split`` ``--pin-groups-from``) →
   ``data/sdfbase/heldout_split.s<s>.json`` (+ ``.provenance.json``); the
   control split under the same seed salts at the campaign's 0.05 →
   ``data/sdfbase/ctl2_heldout_split.s<s>.json``;
2. constitution pools from that split (``build_pools`` mixed + heldout
   lanes, draw salt ``sdfbase-draw-v1`` UNCHANGED — the seed's train side
   is what varies): ``constv2-mix-full`` + ``constv2-heldout``; seed 1
   additionally cuts ``constv2-mix-10m`` / ``-1m`` / ``-1k`` as nested
   prefixes (the search reference, the memprobe cell, the identity-gate
   cell); max-length preflight;
3. control pools (``build_ctl2_pools``) budget-matched to THIS seed's
   ``pairs_constv2-mix-full.manifest.json`` official token count (read
   after the preflight, so the recorded manifest sha is the final one);
   preflight + ``verify``;
4. nesting: ``verify-nesting`` on seed 1's ladder only; seeds 2/3 record
   ``n/a — single mixed cell``; then ``check_pool_integrity`` on every seed;
5. ``check_probe_pins`` against every constitution training pool;
6. the volume layout ``<out-root>/vol/pool/<key>/`` (hard links) with the
   cells a fit reads: ``pairs_constv2-mix-full``, ``pairs_constv2ctl2-full``,
   ``pairs_constv2-heldout`` (+ 10m / 1m / 1k for seed 1), and a per-seed
   report under ``<out-root>/reports/``.

Steps 7–8 of §4 (tokcache sidecars per cell, uploads to the Modal volume
and both clusters) need the seed keys' fit configs and are run
separately; ``--manifest`` writes ``<out-root>/pools_manifest.json`` (sha256
+ bytes + row count of every staged file) once the vol dirs exist.

Stages: the dirty-generator-tree gate runs ONCE per invocation, so the
production sequence is ``--stage splits`` for all seeds in one process
(``--seed 1 --seed 2 --seed 3``), commit the split artifacts, then
``--stage pools`` per seed (in parallel) from the clean tree that carries
them — every pool manifest then records a generator commit that CONTAINS
the split it was cut from. ``--stage all`` runs both for one seed.

``--frac`` is a builder parameter; the lane value is 0.085 (O2) and any
other constitution value REFUSES unless ``--frac-reason`` names why (the
reason lands in the split's provenance sidecar). The control fraction is
the campaign's 0.05 and is not a flag.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SDFBASE = _HERE.parent
for _p in (_HERE, _SDFBASE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_ctl2_pools as b2  # noqa: E402
import build_heldout_split as hsplit  # noqa: E402
import build_pools as bp  # noqa: E402
import check_pool_integrity as cpi  # noqa: E402
import check_probe_pins as cpp  # noqa: E402

#: The lane's constitution carve fraction (decisions O2): the 42
#: probe-pinned groups are ≈ 5.07 % / 5.10 % of each family, and 0.085 puts
#: ≈ 40 % of the held-out mass under the seed-salted walk.
LANE_FRAC = 0.085
#: The control corpus carries no probe dependence: the campaign's 5 %.
CTL_FRAC = hsplit.FRAC
SEEDS = (1, 2, 3)
KEY_TEMPLATE = "qwen35_4b_base_eb64_s{s}"
SALT_TEMPLATE = hsplit.SALT_HELDOUT + "-s{s}"
EVAL_SALT_TEMPLATE = hsplit.SALT_EVAL + "-s{s}"
DRAW_SALT = bp.SALT_MIX
BASE_CELLS = ("constv2-mix-full", bp.HELDOUT_CELL)
#: Seed 1 only: the search reference (10m), the memprobe cell (1m), the
#: identity-gate WORKING cell (1k) — nested prefixes of the s1 draw.
S1_EXTRA_CELLS = ("constv2-mix-10m", "constv2-mix-1m", "constv2-mix-1k")
CTL_CELLS_ON_VOLUME = (b2.FULL_CELL,)
DATA_DIR = hsplit.WORKTREE_ROOT / "data/sdfbase"
DEFAULT_OUT_ROOT = Path("$STAGE/sdfbase/seedrep")   # under the staging directory; expanded through sisteer.site after parsing
DEFAULT_PROBES = (DATA_DIR / "probe_v2.jsonl", DATA_DIR / "probe_v3_genre.jsonl")
CONTROL_DIR = hsplit.GENERATOR_ROOT / "data/control"
CONTROL_STEM = "final-control-scale-20260903-r2"
DEFAULT_CONTROL_ADVICE = CONTROL_DIR / f"{CONTROL_STEM}-advice.jsonl"
DEFAULT_CONTROL_STORIES = CONTROL_DIR / f"{CONTROL_STEM}-stories.jsonl"
DEFAULT_REGISTER_EDIT = {
    "advice": CONTROL_DIR / f"{CONTROL_STEM}-advice.register-edit.json",
    "stories": CONTROL_DIR / f"{CONTROL_STEM}-stories.register-edit.json",
}


class SeedPoolsError(RuntimeError):
    """Loud refusal — never a silently different lane."""


def _log(*args) -> None:
    print(*args, flush=True)


def _tilde(p) -> str:
    s = str(p)
    home = str(Path.home())
    return "~" + s[len(home):] if s.startswith(home + "/") else s


def key_of(s: int) -> str:
    return KEY_TEMPLATE.format(s=s)


def salts_of(s: int) -> tuple[str, str]:
    return SALT_TEMPLATE.format(s=s), EVAL_SALT_TEMPLATE.format(s=s)


def cells_of(s: int) -> tuple[str, ...]:
    return BASE_CELLS + (S1_EXTRA_CELLS if s == 1 else ())


def volume_cells_of(s: int) -> tuple[str, ...]:
    return cells_of(s) + CTL_CELLS_ON_VOLUME


def split_paths(s: int, data_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    return (data_dir / f"heldout_split.s{s}.json",
            data_dir / f"ctl2_heldout_split.s{s}.json")


def refuse_frac(frac: float, reason: str | None) -> None:
    """The lane value is 0.085; another constitution fraction runs only
    with a stated reason (recorded in provenance)."""
    if abs(frac - LANE_FRAC) > 1e-12 and not reason:
        raise SeedPoolsError(
            f"--frac {frac} is not the lane value {LANE_FRAC} (decisions O2) "
            "— pass --frac-reason '<why>' to build another constitution "
            "fraction; the reason is recorded in the split's provenance")


def refuse_nonempty(path: Path, what: str) -> None:
    if path.exists() and any(path.iterdir()):
        raise SeedPoolsError(
            f"{what} {path} is not empty — the seed build writes FRESH "
            "directories (a rerun into a half-written dir could pair new "
            "bytes with stale manifests); remove it or name another root")


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise SeedPoolsError(f"{dst} already exists")
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


class _MemoTok:
    """One preflight target-tokenizer KEY over a shared underlying tokenizer:
    several ``TARGET_TOKENIZERS`` slugs pin the same ``(repo, revision)``
    (the ``_full`` / ``_eb64`` siblings), and the campaign preflight
    tokenizes every row once per KEY. Keys that share bytes share this memo
    of (text, add_special_tokens) -> count, so the pass costs one
    tokenization per distinct checkpoint while every key still lands in
    the manifest with the numbers the standard preflight would write."""

    def __init__(self, model_key: str, inner, memo: dict):
        self.model_key = model_key
        self.inner = inner
        self.repo = inner.repo
        self.revision = inner.revision
        self._memo = memo

    def n_tokens(self, text: str, add_special_tokens: bool) -> int:
        k = (text, add_special_tokens)
        n = self._memo.get(k)
        if n is None:
            n = self._memo[k] = self.inner.n_tokens(text, add_special_tokens)
        return n


def preflight_tokenizers() -> dict[str, _MemoTok]:
    """``bp.load_target_tokenizers()`` with the duplicate checkpoints shared
    (same keys, same tokenizers, same counts — fewer tokenizations)."""
    inner: dict[tuple[str, str], object] = {}
    memos: dict[tuple[str, str], dict] = {}
    out: dict[str, _MemoTok] = {}
    for key, (repo, rev) in bp.TARGET_TOKENIZERS.items():
        pin = (repo, rev)
        if pin not in inner:
            inner[pin] = bp.HFTargetTokenizer(key, repo, rev)
            memos[pin] = {}
        out[key] = _MemoTok(key, inner[pin], memos[pin])
    return out


def _clear_memos(toks: dict) -> None:
    for t in toks.values():
        memo = getattr(t, "_memo", None)
        if isinstance(memo, dict):
            memo.clear()


PREFILL_CHUNK = 128


def _prefill_memos(pools_dir: Path, toks: dict, chunk: int = PREFILL_CHUNK
                   ) -> dict:
    """Fill the shared count memos for every row of every pool under
    ``pools_dir`` with the underlying fast tokenizers' BATCH encoder
    (``HFTargetTokenizer._tok(list_of_texts)``, Rust-parallel; measured
    3.6× the per-text path on the grading host with identical ids), once per
    distinct checkpoint. ``bp.run_preflight`` then runs its standard
    per-key, per-row loop entirely from memo hits and writes the numbers
    it always writes. A tokenizer without a batch encoder (tests' fakes)
    is left to the per-text path. Returns {pin: n_texts} for the log."""
    texts_by_flag: dict[bool, list[str]] = {True: [], False: []}
    for pool_path in sorted(Path(pools_dir).glob("pairs_*.jsonl")):
        with open(pool_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    texts_by_flag[True].append(row["prompt"])
                    texts_by_flag[False].append(row["pos"])
    done: dict[tuple[str, str], int] = {}
    for t in toks.values():
        inner = getattr(t, "inner", None)
        memo = getattr(t, "_memo", None)
        hf = getattr(inner, "_tok", None)
        if inner is None or memo is None or hf is None:
            continue
        pin = (inner.repo, inner.revision)
        if pin in done:
            continue
        n = 0
        for flag, texts in texts_by_flag.items():
            todo = sorted({x for x in texts if (x, flag) not in memo})
            for i in range(0, len(todo), chunk):
                batch = todo[i:i + chunk]
                ids = hf(batch, add_special_tokens=flag)["input_ids"]
                if len(ids) != len(batch):
                    raise SeedPoolsError("batch encoder returned "
                                         f"{len(ids)} rows for {len(batch)}")
                for text, row_ids in zip(batch, ids):
                    memo[(text, flag)] = len(row_ids)
                n += len(batch)
        done[pin] = n
    return done


def _count_rows(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# --------------------------------------------------------------------------
# Stage 1 — splits
# --------------------------------------------------------------------------

def stage_splits(s: int, *, frac: float, frac_reason: str | None,
                 generator: dict, advice_final: Path, stories_final: Path,
                 control_advice: Path, control_stories: Path, probes,
                 data_dir: Path = DATA_DIR, expect_shas: dict | None = None,
                 eval_n: int = hsplit.EVAL_N, log=_log) -> dict:
    salt, eval_salt = salts_of(s)
    split_path, ctl2_split_path = split_paths(s, data_dir)
    for p in (split_path, ctl2_split_path):
        if p.exists():
            raise SeedPoolsError(
                f"{p} already exists — a seed split is built once and "
                "committed; remove it deliberately to re-carve")
    extra = {"seed": s, "key": key_of(s), "lane_frac": LANE_FRAC,
             "frac_override_reason": frac_reason}

    t0 = time.time()
    split = hsplit.build_split(advice_final, stories_final, frac=frac,
                               salt=salt, eval_salt=eval_salt, eval_n=eval_n,
                               pin_probe_files=list(probes))
    hsplit.write_split(split_path, split, generator=generator, salt=salt,
                       eval_salt=eval_salt, frac=frac, extra=extra)
    log(f"[s{s}] constitution split -> {_tilde(split_path)} "
        f"({time.time() - t0:.1f}s)")
    hsplit.print_split_summary(split)
    pin_rep = cpp.check(split_path, probes, [], frac=frac)
    if not pin_rep["ok"]:
        raise SeedPoolsError("probe-pin check on the fresh split: "
                             + "; ".join(pin_rep["findings"]))

    t0 = time.time()
    shas, pinned, label = b2.verify_control_shas(control_advice,
                                                 control_stories, expect_shas)
    ctl_split = hsplit.build_split(control_advice, control_stories,
                                   frac=CTL_FRAC, salt=salt,
                                   eval_salt=eval_salt, eval_n=eval_n)
    b2.write_ctl2_split(ctl_split, ctl2_split_path, shas=shas, pinned=pinned,
                        release_label=label, generator=generator)
    log(f"[s{s}] control split -> {_tilde(ctl2_split_path)} "
        f"({time.time() - t0:.1f}s; release bytes pinned: {pinned})")
    hsplit.print_split_summary(ctl_split)

    def _fam(sp, fam):
        acc = sp["token_totals"][fam]
        return {"heldout_groups": acc["heldout_groups"],
                "heldout_records": acc["heldout_records"],
                "heldout_official": acc["heldout_official"],
                "frac_realized": acc["heldout_frac_realized"],
                "train_records": acc["train_records"],
                "train_official": acc["train_official"]}

    return {
        "seed": s, "key": key_of(s), "salt": salt, "eval_salt": eval_salt,
        "constitution": {
            "artifact": _tilde(split_path),
            "sha256": hsplit.sha256_file(split_path),
            "frac_requested": frac,
            "families": {f: _fam(split, f) for f in ("advice", "stories")},
            "pins": {f: {k: split["pins"][f][k] for k in
                         ("n_pinned_groups", "n_seed_drawn_groups",
                          "pinned_records", "seed_drawn_records",
                          "pinned_frac_of_family",
                          "seed_drawn_share_of_heldout")}
                     for f in ("advice", "stories")},
            "n_pinned_source_ids": split["pins"]["n_pinned_source_ids"],
            "heldout_docs": len(split["advice"]["ids"])
            + len(split["stories"]["ids"]),
            "eval_family_counts": split["eval_subset"]["family_counts"],
        },
        "control": {
            "artifact": _tilde(ctl2_split_path),
            "sha256": hsplit.sha256_file(ctl2_split_path),
            "frac_requested": CTL_FRAC,
            "release_bytes_pinned": pinned,
            "families": {f: _fam(ctl_split, f) for f in ("advice", "stories")},
            "heldout_docs": len(ctl_split["advice"]["ids"])
            + len(ctl_split["stories"]["ids"]),
            "eval_family_counts": ctl_split["eval_subset"]["family_counts"],
        },
        "probe_pins_on_split": {k: pin_rep[k] for k in
                                ("n_probe_source_ids",
                                 "n_probe_sources_in_heldout", "frac_ok")},
    }


# --------------------------------------------------------------------------
# Stage 2 — pools, checks, volume layout
# --------------------------------------------------------------------------

def stage_pools(s: int, *, frac: float, generator: dict, out_root: Path,
                advice_final: Path, stories_final: Path, control_advice: Path,
                control_stories: Path, probes, register_edit: dict | None,
                data_dir: Path = DATA_DIR, expect_shas: dict | None = None,
                eval_n: int = hsplit.EVAL_N, rig=None, toks=None,
                log=_log) -> dict:
    salt, eval_salt = salts_of(s)
    split_path, ctl2_split_path = split_paths(s, data_dir)
    for p in (split_path, ctl2_split_path):
        if not p.is_file():
            raise SeedPoolsError(f"{p} missing — run --stage splits (and "
                                 "commit the artifacts) before --stage pools")
    key = key_of(s)
    pools_dir = out_root / f"pools_s{s}"
    ctl2_dir = out_root / f"pools_ctl2_s{s}"
    vol_dir = out_root / "vol/pool" / key
    reports = out_root / "reports"
    for d, what in ((pools_dir, "pools dir"), (ctl2_dir, "ctl2 pools dir"),
                    (vol_dir, "volume dir")):
        refuse_nonempty(d, what)
    reports.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    # 2. constitution pools (draw salt unchanged) + preflight
    t0 = time.time()
    cells = cells_of(s)
    manifests = bp.build_pools(
        pools_dir, split_path, advice_final=advice_final,
        stories_final=stories_final, lanes=("mixed", "heldout"),
        only_cells=set(cells), generator=generator, rig=rig)
    timings["build_pools"] = time.time() - t0
    log(f"[s{s}] pools {sorted(manifests)} -> {_tilde(pools_dir)} "
        f"({timings['build_pools']:.0f}s)")
    if set(manifests) != set(cells):
        raise SeedPoolsError(f"built {sorted(manifests)} != requested "
                             f"{sorted(cells)}")
    t0 = time.time()
    toks = toks if toks is not None else preflight_tokenizers()
    filled = _prefill_memos(pools_dir, toks)
    log(f"[s{s}] preflight memos prefilled (batch encoder) for "
        f"{len(filled)} distinct checkpoint(s) ({time.time() - t0:.0f}s)")
    bp.run_preflight(pools_dir, toks=toks)
    _clear_memos(toks)
    timings["preflight_pools"] = time.time() - t0
    log(f"[s{s}] preflight {_tilde(pools_dir)} passed "
        f"({timings['preflight_pools']:.0f}s)")

    # 3. control pools budget-matched to THIS seed's mix-full (manifest read
    #    after the preflight, so budget_source.manifest_sha256 is final)
    t0 = time.time()
    before = ctl2_split_path.read_bytes()
    budget_manifest = pools_dir / f"pairs_{b2.BUDGET_CELL}.manifest.json"
    ctl_manifests = b2.build_ctl2(
        ctl2_dir, control_advice=control_advice,
        control_stories=control_stories, budget_manifest=budget_manifest,
        expect_shas=expect_shas, rig=rig, generator=generator,
        split_out=ctl2_split_path, frac=CTL_FRAC, eval_n=eval_n, salt=salt,
        eval_salt=eval_salt, register_edit_manifests=register_edit)
    if ctl2_split_path.read_bytes() != before:
        raise SeedPoolsError(
            f"the pools stage re-derived a DIFFERENT control split than "
            f"{_tilde(ctl2_split_path)} — the committed artifact and the "
            "build disagree (corpus bytes or builder changed?)")
    timings["build_ctl2"] = time.time() - t0
    log(f"[s{s}] ctl2 pools {sorted(ctl_manifests)} -> {_tilde(ctl2_dir)} "
        f"(budget {ctl_manifests[b2.FULL_CELL]['budget']:,}; "
        f"{timings['build_ctl2']:.0f}s)")
    t0 = time.time()
    _prefill_memos(ctl2_dir, toks)
    bp.run_preflight(ctl2_dir, toks=toks)
    _clear_memos(toks)
    b2.verify_pools(ctl2_dir)
    timings["preflight_ctl2"] = time.time() - t0
    log(f"[s{s}] preflight + verify {_tilde(ctl2_dir)} passed "
        f"({timings['preflight_ctl2']:.0f}s)")

    report = stage_checks(s, frac=frac, generator=generator, out_root=out_root,
                          advice_final=advice_final, stories_final=stories_final,
                          control_advice=control_advice,
                          control_stories=control_stories, probes=probes,
                          data_dir=data_dir, timings=timings, log=log)
    return report


# --------------------------------------------------------------------------
# Stage 3 — checks on EXISTING pool dirs (steps 4-6), volume layout, report
# --------------------------------------------------------------------------

def stage_checks(s: int, *, frac: float, generator: dict, out_root: Path,
                 advice_final: Path, stories_final: Path, control_advice: Path,
                 control_stories: Path, probes, data_dir: Path = DATA_DIR,
                 timings: dict | None = None, log=None) -> dict:
    """Steps 4-6 over pool dirs that already exist (a completed
    ``--stage pools`` whose checks refused, or a re-verification): nesting,
    ``check_pool_integrity``, ``check_probe_pins``, the volume layout and
    the per-seed report. Reads the PREFLIGHTED manifests from disk."""
    log = log or _log
    timings = dict(timings or {})
    salt, eval_salt = salts_of(s)
    split_path, ctl2_split_path = split_paths(s, data_dir)
    key = key_of(s)
    pools_dir = out_root / f"pools_s{s}"
    ctl2_dir = out_root / f"pools_ctl2_s{s}"
    vol_dir = out_root / "vol/pool" / key
    reports = out_root / "reports"
    for d in (pools_dir, ctl2_dir):
        if not (d / "preflight_report.json").is_file():
            raise SeedPoolsError(f"{d} has no preflight_report.json — the pools "
                                 "stage did not finish its preflight; rebuild")
    refuse_nonempty(vol_dir, "volume dir")
    reports.mkdir(parents=True, exist_ok=True)
    missing = [c for c in cells_of(s)
               if not (pools_dir / f"pairs_{c}.manifest.json").is_file()]
    if missing:
        raise SeedPoolsError(
            f"seed {s} pool dir {_tilde(pools_dir)} lacks {missing} — seed 1 "
            "carries the 1k/1m/10m/full ladder, seeds 2/3 mix-full + heldout")
    manifests = {c: cpi.load_manifest(pools_dir, c) for c in cells_of(s)}
    ctl_manifests = {c: cpi.load_manifest(ctl2_dir, c)
                     for c in (b2.FULL_CELL, b2.HELDOUT_CELL)}

    # 4. nesting (s1's ladder; a single mixed cell reads n/a) + integrity
    nesting, nesting_details = cpi.verify_nesting(pools_dir)
    log(f"[s{s}] nesting: {nesting}")
    t0 = time.time()
    integrity = cpi.run(pools=pools_dir, ctl2=ctl2_dir, split=split_path,
                        ctl2_split=ctl2_split_path, advice_final=advice_final,
                        stories_final=stories_final,
                        control_advice=control_advice,
                        control_stories=control_stories)
    (reports / f"integrity_s{s}.json").write_text(
        json.dumps(integrity, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    timings["integrity"] = time.time() - t0
    if not integrity["ok"]:
        raise SeedPoolsError(
            "pool integrity FAILED: "
            + "; ".join(c["name"] for c in integrity["failures"]))
    if integrity["nesting"] != nesting:
        raise SeedPoolsError(f"nesting verdicts disagree: driver {nesting!r} "
                             f"vs integrity {integrity['nesting']!r}")
    log(f"[s{s}] pool integrity OK ({integrity['n_checks']} checks, "
        f"{timings['integrity']:.0f}s)")

    # 5. probe pins against every constitution training pool
    train_pools = sorted(pools_dir.glob("pairs_constv2-mix-*.jsonl"))
    pin_rep = cpp.check(split_path, probes, train_pools, frac=frac)
    (reports / f"probe_pins_s{s}.json").write_text(
        json.dumps(pin_rep, indent=1) + "\n", encoding="utf-8")
    if not pin_rep["ok"]:
        raise SeedPoolsError("probe-pin check FAILED: "
                             + "; ".join(pin_rep["findings"]))
    log(f"[s{s}] probe pins OK over {len(train_pools)} training pool(s)")

    # 6. the volume layout
    files: dict[str, dict] = {}
    for cell in volume_cells_of(s):
        src_dir = ctl2_dir if cell in CTL_CELLS_ON_VOLUME else pools_dir
        for suffix in (".jsonl", ".manifest.json"):
            name = f"pairs_{cell}{suffix}"
            how = _link_or_copy(src_dir / name, vol_dir / name)
            entry = {"sha256": hsplit.sha256_file(vol_dir / name),
                     "bytes": (vol_dir / name).stat().st_size, "via": how}
            if suffix == ".jsonl":
                entry["n_rows"] = _count_rows(vol_dir / name)
            files[name] = entry
    log(f"[s{s}] volume layout {_tilde(vol_dir)}: {len(files)} files")

    def _cell(m):
        return {"n_docs": m["n_docs"], "official_tokens": m["official_tokens"],
                "budget": m["budget"], "limiting": m["limiting_tokenizer"],
                "family_counts": m["family_counts"],
                "truncated_doc": (m["truncation"] or {}).get("doc_id"),
                "boundary_skips": [x["doc_id"] for x in m["boundary_skips"]],
                "official_loss_tokens": m["official_loss_tokens"],
                "max_encoded_row_len": m.get("max_encoded_row_len"),
                "pool_sha256": m["pool_sha256"]}

    mf = manifests["constv2-mix-full"]
    cf = ctl_manifests[b2.FULL_CELL]
    report = {
        "seed": s, "key": key, "salt": salt, "eval_salt": eval_salt,
        "draw_salt": DRAW_SALT, "frac": frac, "ctl_frac": CTL_FRAC,
        "split": {"artifact": _tilde(split_path),
                  "sha256": hsplit.sha256_file(split_path)},
        "ctl2_split": {"artifact": _tilde(ctl2_split_path),
                       "sha256": hsplit.sha256_file(ctl2_split_path)},
        "pools_dir": _tilde(pools_dir), "ctl2_dir": _tilde(ctl2_dir),
        "cells": {**{c: _cell(m) for c, m in manifests.items()},
                  **{c: _cell(m) for c, m in ctl_manifests.items()}},
        "budget_match": {
            "constitution_mix_full_official_tokens": mf["official_tokens"],
            "ctl2_full_official_tokens": cf["official_tokens"],
            "ctl2_budget": cf["budget"],
            "ctl2_remainder_docs": cf["remainder"]["n_docs"],
            "ctl2_remainder_official_tokens": cf["remainder"]["official_tokens"],
            "reference_seed0_mix_full_official_tokens": 42_785_435,
            "delta_vs_seed0": mf["official_tokens"] - 42_785_435,
            "delta_vs_seed0_frac": mf["official_tokens"] / 42_785_435 - 1.0,
        },
        "nesting": nesting, "nesting_details": nesting_details,
        "integrity": {"ok": integrity["ok"], "n_checks": integrity["n_checks"],
                      "report": _tilde(reports / f"integrity_s{s}.json")},
        "probe_pins": {"ok": pin_rep["ok"],
                       "train_pools": pin_rep["train_pools"],
                       "frac_realized": pin_rep["frac_realized"],
                       "pins": pin_rep["pins"],
                       "report": _tilde(reports / f"probe_pins_s{s}.json")},
        "volume_dir": _tilde(vol_dir), "volume_files": files,
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
        "generator": generator,
    }
    (reports / f"seed_s{s}.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    log(f"[s{s}] report -> {_tilde(reports / f'seed_s{s}.json')}")
    return report


# --------------------------------------------------------------------------
# The staged-files manifest (sha256 + bytes + rows of every vol file)
# --------------------------------------------------------------------------

def write_pools_manifest(out_root: Path) -> Path:
    vol = out_root / "vol/pool"
    keys = sorted(p.name for p in vol.iterdir() if p.is_dir()) if vol.is_dir() else []
    if not keys:
        raise SeedPoolsError(f"no vol/pool/<key>/ under {out_root}")
    entry: dict = {"root": _tilde(out_root), "vol": _tilde(vol),
                   "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "keys": {}}
    for key in keys:
        files = {}
        for f in sorted((vol / key).iterdir()):
            if not f.is_file():
                continue
            rec = {"sha256": hsplit.sha256_file(f), "bytes": f.stat().st_size}
            if f.suffix == ".jsonl":
                rec["n_rows"] = _count_rows(f)
            files[f.name] = rec
        entry["keys"][key] = {"dir": _tilde(vol / key), "n_files": len(files),
                              "files": files}
    entry["locations"] = {
        "local": {"verified": True, "note": "sha256 of every file above"},
        "modal_volume": None, "cluster_a": None, "cluster_b": None,
        "note": "remote locations are filled by §4 step 8 (upload + "
                "round-trip sha256 + file COUNT per location)",
    }
    out = out_root / "pools_manifest.json"
    out.write_text(json.dumps(entry, indent=1) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, action="append", default=[],
                    choices=SEEDS, help="replicate seed (repeatable)")
    ap.add_argument("--frac", type=float, default=LANE_FRAC,
                    help=f"constitution carve fraction (lane value {LANE_FRAC})")
    ap.add_argument("--frac-reason", default=None,
                    help="REQUIRED to build any constitution --frac other "
                         "than the lane value; recorded in provenance")
    ap.add_argument("--stage", choices=("splits", "pools", "checks", "all"),
                    default="all",
                    help="splits: step 1 only; pools: steps 2-6 (needs the "
                         "committed splits); checks: steps 4-6 over EXISTING "
                         "preflighted pool dirs (re-verification, or after a "
                         "refused check); all: 1-6 for one seed")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--advice-final", type=Path, default=hsplit.ADVICE_FINAL)
    ap.add_argument("--stories-final", type=Path, default=hsplit.STORIES_FINAL)
    ap.add_argument("--control-advice", type=Path,
                    default=DEFAULT_CONTROL_ADVICE)
    ap.add_argument("--control-stories", type=Path,
                    default=DEFAULT_CONTROL_STORIES)
    ap.add_argument("--probes", type=Path, nargs="+",
                    default=list(DEFAULT_PROBES))
    ap.add_argument("--register-edit-manifest", action="append", default=None,
                    metavar="FAMILY=PATH",
                    help="default: the release's two manifests when present")
    ap.add_argument("--manifest", action="store_true",
                    help="only (re)write <out-root>/pools_manifest.json "
                         "from the vol/pool dirs present")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    args.out_root = bp.site_path(args.out_root, "--out-root")

    try:
        if args.manifest:
            out = write_pools_manifest(args.out_root)
            print(f"wrote {_tilde(out)}")
            return 0
        if not args.seed:
            raise SeedPoolsError("--seed is required (repeatable)")
        seeds = sorted(set(args.seed))
        refuse_frac(args.frac, args.frac_reason)
        generator = hsplit.refuse_dirty_generator_tree(
            args.allow_dirty, tag="build-seed-pools")
        if args.register_edit_manifest is None:
            rem = {f: p for f, p in DEFAULT_REGISTER_EDIT.items()
                   if Path(p).is_file()}
        else:
            rem = {}
            for spec in args.register_edit_manifest:
                fam, _, p = spec.partition("=")
                rem[fam] = p
        common = dict(advice_final=args.advice_final,
                      stories_final=args.stories_final,
                      control_advice=args.control_advice,
                      control_stories=args.control_stories,
                      probes=list(args.probes))
        for s in seeds:
            if args.stage in ("splits", "all"):
                stage_splits(s, frac=args.frac, frac_reason=args.frac_reason,
                             generator=generator, **common)
            if args.stage in ("pools", "all"):
                stage_pools(s, frac=args.frac, generator=generator,
                            out_root=args.out_root, register_edit=rem,
                            **common)
            if args.stage == "checks":
                stage_checks(s, frac=args.frac, generator=generator,
                             out_root=args.out_root, **common)
    except (SeedPoolsError, bp.SamplerError, bp.PreflightError,
            hsplit.SplitError, cpp.ProbePinError, cpi.IntegrityError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"done: seeds {seeds}, stage {args.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
