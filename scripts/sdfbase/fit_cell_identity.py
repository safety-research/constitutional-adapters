"""fit_cell_identity — the fit-cell identity gates, Modal-free.

The three gates every sdf-base-steering fit cell runs around the trainer —
the EXPECTED fit fingerprint (through the driver's own derivation), the
stale-COMPLETE supersede gate, and final-checkpoint retention — used to live
only inside ``scripts/modal/sdfbase_fits.py``, which imports ``modal`` at the
top. The Slurm lane (``scripts/sdfbase/slurm_fit_cell.py``, 2026-09-06)
runs the SAME trainer code path on a cluster node where ``modal`` is not
installed, and a second copy of a gate is how two lanes learn to disagree.
So the gates live here, importable anywhere the repo's ``scripts/`` and
``src/`` trees are, and ``sdfbase_fits.py`` delegates to them: one
derivation, whether the container or the sbatch job is asking.

Semantics are the Modal app's, unchanged (their docstrings carry the why;
the review findings they answer are R2 finding 1a/1c and the 2026-09-02
retention directive). ``expected_fit_fingerprint`` here is the PLAIN fit's
(no ``--continue-cap``); the continuation derivation stays with the app,
which is the only place that spawns one.

Since 2026-09-10 this module also owns the WRITABLE NAMESPACE of a fit
config (``out_key`` / ``out_root_for`` / ``published_elsewhere_in_bank``, the
section at the bottom): the one derivation of where a config's cells,
checkpoints and reports land, which the sbatch reads through the ``out-key``
CLI and ``slurm_fit_cell.py`` enforces.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

#: The store-identity sidecar the trainer writes beside the cell dirs
#: (train_two_si_objects.assert_store_identity) — {model, model_revision,
#: positions}; one objects root holds ONE (model, revision, position leg).
STORE_IDENT = "_store_identity.json"


def repo_imports() -> None:
    """Put the repo's src/, scripts/ and scripts/sdfbase/ on sys.path so the
    trainer's own functions import (in a container /repo, on a cluster the
    checkout the job runs from — REPO is derived from this file)."""
    for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase"):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def config_sha16(cfg_path) -> str:
    """The 16-hex config-BYTES sha the fits pin as ``expect_sha`` (the
    trainer's ``--expect-config-sha`` contract) — from whichever copy of the
    config is at ``cfg_path``."""
    return hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest()[:16]


def expected_fit_fingerprint(cfg_path, pairs_path, si: str, arm: str) -> str:
    """The fit fingerprint the trainer WILL stamp into this cell's COMPLETE
    marker — derived through the trainer's OWN functions (the same
    ``cell_fit_fingerprint`` + ``_cli_fit_identity`` pair ``cmd_train``
    calls, under the exact invocation the lanes issue: ``--positions all``,
    no teacher artifact), never re-implemented, so the pre-spawn record, the
    gates and the marker can never disagree. The config's shared held-out
    pool is a ``/vol`` path: this resolves where the pools are mounted (a
    container), or where ``SISTEER_VOL_ROOT`` maps it (a cluster node)."""
    from types import SimpleNamespace

    repo_imports()
    import train_two_si_objects as t2s

    cfg = t2s._cfg_with_positions(t2s._load_cfg(str(cfg_path)), "all")
    pool_sha, config_sha = t2s._cli_fit_identity(
        SimpleNamespace(pairs=str(pairs_path), config=str(cfg_path)), cfg)
    return t2s.cell_fit_fingerprint(cfg, si, arm, positions="all",
                                    pool_sha256=pool_sha,
                                    config_sha=config_sha, teacher="online")


def marker_fingerprint(cell_dir) -> tuple[str, str | None]:
    """(``'absent' | 'unparseable' | 'present'``, the marker's
    fit_fingerprint or None) for a cell dir's COMPLETE marker."""
    mp = Path(cell_dir) / "COMPLETE"
    if not mp.is_file():
        return "absent", None
    try:
        marker = json.loads(mp.read_text())
    except ValueError:
        return "unparseable", None
    fp = marker.get("fit_fingerprint") if isinstance(marker, dict) else None
    return "present", (str(fp) if fp else None)


def supersede_gate(cell_dir, expect_fingerprint: str) -> str | None:
    """None when the cell dir may be (re)fit under ``expect_fingerprint``:
    no COMPLETE marker (a fresh or in-flight cell), a marker already
    carrying this fingerprint (the ordinary resume/skip), or a marker of a
    DIFFERENT identity whose dir carries a superseded_by.json naming that
    very fingerprint (scripts/sdfbase/snapshot_cells.py took the snapshot —
    the exact event that licenses the overwrite). Otherwise the problem
    string: a published cell of another identity is about to be overwritten
    with no snapshot, the pre-spawn check a refit must never skip."""
    repo_imports()
    from snapshot_cells import SUPERSEDE_RECORD, supersede_record

    presence, stored = marker_fingerprint(cell_dir)
    if presence == "absent":
        return None
    if presence == "unparseable":
        return (f"{Path(cell_dir) / 'COMPLETE'} does not parse — refusing "
                "to train over a publication whose identity cannot be read")
    if stored == expect_fingerprint:
        return None
    try:
        rec = supersede_record(cell_dir)
    except Exception as e:  # noqa: BLE001 — an unreadable record is a refusal
        return f"{Path(cell_dir) / SUPERSEDE_RECORD} unreadable: {e}"
    if rec is None:
        return (f"{cell_dir} is COMPLETE under fit fingerprint "
                f"{(stored or 'absent')[:12]}…, not this fit's "
                f"{expect_fingerprint[:12]}…, and carries no "
                f"{SUPERSEDE_RECORD}: snapshot it first (modal run "
                "scripts/modal/sdfbase_fits.py --mode snapshot --model "
                f"<key> --cells {Path(cell_dir).name}) — a refit must never "
                "overwrite an unsnapshotted publication")
    if rec.get("fit_fingerprint") != stored:
        return (f"{Path(cell_dir) / SUPERSEDE_RECORD} records fingerprint "
                f"{str(rec.get('fit_fingerprint'))[:12]}… but the marker "
                f"carries {(stored or 'absent')[:12]}… — the CURRENT "
                "publication was never snapshotted (re-run --mode snapshot)")
    return None


def retain_final_checkpoint(cell_dir, ckpt_cell, *,
                            expect_fingerprint: str) -> bool:
    """Move a COMPLETE cell's transient checkpoint files INTO the published
    cell dir as ``final_ckpt/``: deleting checkpoints at publish made exact
    resume of the cap-limited 1M cells impossible, and retention costs
    megabytes. Returns True when something was retained.

    Fires ONLY when the dir's COMPLETE marker carries ``expect_fingerprint``
    — the fit this job ran or validated-skipped. A missing marker, a marker
    of ANOTHER identity (a stale publication under a refit that failed or is
    still in flight), or no checkpoint files, moves nothing: the transient
    checkpoint is the successor's resume point and belongs to the NEW fit,
    never to the old cell dir. A pre-existing final_ckpt/ is superseded (a
    respawn's new checkpoint replaces the stale one, never merges with it)."""
    cell_dir, ckpt_cell = Path(cell_dir), Path(ckpt_cell)
    presence, stored = marker_fingerprint(cell_dir)
    if (presence != "present" or stored is None
            or stored != expect_fingerprint or not ckpt_cell.is_dir()):
        return False
    files = sorted(p for p in ckpt_cell.iterdir() if p.is_file())
    if not files:
        shutil.rmtree(ckpt_cell, ignore_errors=True)
        return False
    dest = cell_dir / "final_ckpt"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.move(str(src), str(dest / src.name))
    shutil.rmtree(ckpt_cell, ignore_errors=True)
    return True


# --------------------------------------------------------------------------- #
# The writable namespace of a fit config (2026-09-10)                          #
# --------------------------------------------------------------------------- #
# A fit cell's outputs — cells_all/<cell_id>/, ckpt/<cell_id>/, reports/ —
# land under ONE root, <bank>/<out_key>/. Until 2026-09-10 the sbatch built
# out_key from the POOL key (the dir the config's held-out pool lives in), so
# every config sharing a pool shared a root: glm46vflash_doc_full and its
# refit sibling glm46vflash_doc_full_r2t12 both resolved to newmodels_full/
# glm46vflash_doc — the same cell dirs, the same reports/, and the same
# ckpt/<cell_id>/ slot, which the trainer's fitckpt rotates with os.replace
# after IGNORING a checkpoint whose key is another fit's ("starting this fit
# fresh"): two saves and the frozen fit's resume state is gone. supersede_gate
# does not cover it — it admits any cell without a COMPLETE marker, an
# interrupted or in-flight frozen fit included. So the namespace is derived
# from the CONFIG's identity — its key, configs/sdfbase_<key>_fits.yaml — and
# the launcher enforces the derivation (slurm_fit_cell refuses an --out-root
# that is not the config's own): no choice of positional args can put two
# configs under one root.
#
# FROZEN_OUT_KEYS pins the layout that already exists on cluster A: the two
# full-scale lanes published under newmodels_full/<POOL key> (2026-09-06/07),
# the parents of the campaign's derived objects — those dirs may not move, so
# the pin keeps them resolving exactly as before. Everything else (every
# other bank, every other key) is <bank>/<config key>. A config whose own key
# IS another config's pinned dir in that bank has no root there (the small
# glm46vflash_doc config in newmodels_full/ would land on the frozen
# glm46vflash_doc_full lane) and is refused.

#: ``(bank, config key) -> dir under <bank>/`` for the roots that predate the
#: config-keyed namespace and are published. Pinned dir for dir; add a row
#: ONLY for a root that already exists — never to alias a new config onto an
#: old root, which is the bug this section exists to end.
FROZEN_OUT_KEYS: dict[tuple[str, str], str] = {
    ("newmodels_full", "glm46vflash_doc_full"): "glm46vflash_doc",
    ("newmodels_full", "nemotron3nano4b_doc_full"): "nemotron3nano4b_doc",
}

#: A fits config is configs/sdfbase_<key>_fits.yaml; the key is the identity
#: the sbatch's positional model_key names and the namespace is keyed on.
CONFIG_NAME_RE = re.compile(r"^sdfbase_(?P<key>[A-Za-z0-9][A-Za-z0-9_]*)_fits\.yaml$")


class LayoutError(ValueError):
    """A fit config has no writable namespace where it was pointed."""


def config_key_of(cfg_path) -> str:
    """The config key (``sdfbase_<key>_fits.yaml`` -> ``<key>``) — read from
    the file's own name, the one place the identity lives; a file named
    otherwise has no key and no namespace (LayoutError)."""
    name = Path(cfg_path).name
    m = CONFIG_NAME_RE.match(name)
    if not m:
        raise LayoutError(
            f"{cfg_path}: a fits config is configs/sdfbase_<key>_fits.yaml — "
            f"the key is the fit's identity and cannot be read from {name!r}")
    return m.group("key")


def out_key(bank: str, config_key: str) -> str:
    """The dir under ``<bank>/`` that ``config_key``'s cells, checkpoints and
    reports live in: the FROZEN_OUT_KEYS pin when one exists, else the config
    key itself. Injective per bank — two configs never share a dir — because
    a config whose own key is another config's pinned dir is refused
    (LayoutError) rather than aliased onto it."""
    pinned = FROZEN_OUT_KEYS.get((bank, config_key))
    if pinned is not None:
        return pinned
    for (b, owner), d in FROZEN_OUT_KEYS.items():
        if b == bank and d == config_key:
            raise LayoutError(
                f"{bank}/{config_key} is the frozen namespace of config "
                f"{owner} (FROZEN_OUT_KEYS); config {config_key} has no "
                f"writable root in bank {bank} — its own key IS that dir")
    return config_key


def out_root_for(bank_root, cfg_path) -> Path:
    """``<bank_root>/<out_key(bank, key)>`` for the config at ``cfg_path``:
    the ONE derivation the sbatch computes (``out-key`` CLI) and
    ``slurm_fit_cell.py`` re-derives and enforces."""
    bank_root = Path(bank_root)
    return bank_root / out_key(bank_root.name, config_key_of(cfg_path))


def published_elsewhere_in_bank(own_root, cell_id: str,
                                expect_fingerprint: str) -> Path | None:
    """The COMPLETE marker of ``cell_id`` already published under EXACTLY
    ``expect_fingerprint`` in ANOTHER root of the same bank, or None.

    Before the config-keyed namespace a sibling config's cells landed in its
    pool-mate's root: glm46vflash_doc_full_r2t12's two psr_ce_all cells sit
    in newmodels_full/glm46vflash_doc/ (Slurm jobs 441097/441098,
    2026-09-07/08) and are the parents of derived objects, while its own
    root is empty — a launch there would refit the same identity from scratch
    for a day and leave two copies. A published fit is published wherever it
    sits; moving one is a human's call, so the launcher refuses instead."""
    own_root = Path(own_root)
    for marker in sorted(own_root.parent.glob(f"*/cells_all/{cell_id}/COMPLETE")):
        root = marker.parents[2]
        if root == own_root:
            continue
        presence, stored = marker_fingerprint(marker.parent)
        if presence == "present" and stored == expect_fingerprint:
            return marker
    return None


def main(argv=None) -> int:
    """``python3 fit_cell_identity.py out-key --bank <bank> --config <path>``
    prints the config's out_key (the sbatch's OUT_KEY) — stdlib only, so the
    batch shell can run it before any venv is touched; a refusal is exit 3
    with the reason on stderr."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ok = sub.add_parser("out-key", help="the dir under <bank>/ this config "
                                        "writes to (fit_cell_identity.out_key)")
    ok.add_argument("--bank", required=True)
    ok.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    try:
        print(out_key(a.bank, config_key_of(a.config)))
    except LayoutError as e:
        print(f"[fit-cell-identity] REFUSED: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
