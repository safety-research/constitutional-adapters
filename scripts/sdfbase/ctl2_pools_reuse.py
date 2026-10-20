#!/usr/bin/env python
"""Make `reproduce/02_prepare_model.sh --step pools` re-runnable past its ctl2 step
(paper-companion ruling D98 (i)).

The pools chain runs the builders in one remote bash string. The earlier builders
overwrite their outputs, but ``build_ctl2_pools.py build`` REFUSES a non-empty
``--out`` by contract (``refuse_nonempty_out``: a rerun into a half-written dir
could pair new pool bytes with stale manifests), so a re-run after a later failure
used to die ~27 minutes in at the ctl2 step. The builder is seeded — a rebuild
reproduces the recorded pool bytes (verified on the live cluster, D105) — so a
finished build is worth keeping, and an unfinished one is worth stepping around,
never over.

Three subcommands, all run ON THE CLUSTER from the pinned checkout by the fit venv's
interpreter; every answer is ONE stdout line the chain (or the stage script) reads with
``read -r``:

``decide --root <canonical pools_ctl2 dir> --budget-manifest <mix-full manifest>``
    prints ``REUSE <dir>`` or ``BUILD <dir>`` and exits 0; exits 3 with
    ``REFUSED: …`` on stderr when the recorded build cannot be trusted. The CURRENT
    build is the one the pointer file ``<root>.CURRENT`` names when that file
    exists, else ``<root>`` itself.

    * current dir missing or empty  -> ``BUILD <root>`` (the first run; the chain
      ``mkdir -p``'s it);
    * current dir COMPLETE          -> the recorded BUDGET must match the current
      one (below), the build must carry a RECIPE record (below; none -> REFUSED),
      then ``verify_pools`` (pool bytes / rows / ordered ids against the
      manifests; disagree -> REFUSED by name: a complete build is never rebuilt
      over and never deleted); then the recorded recipe against today's builder's:
      equal -> ``REUSE <dir>``; different -> ``BUILD <root>.build-<stamp>`` (the
      complete build is left in place, named on stderr with every differing field,
      and NOT reused: its control selection is another builder's);
    * current dir INCOMPLETE        -> ``BUILD <root>.build-<stamp>`` where the stamp
      is ``job<SLURM_JOB_ID>`` under Slurm, else ``<UTC %Y%m%dT%H%M%SZ>-pid<pid>``;
      the partial dir is left in place (named on stderr); a target that already
      exists is REFUSED, and so is a pre-existing expectation sidecar for it (below);
    * a pointer naming a dir that does not exist (or is empty) is dangling -> REFUSED.

    COMPLETE means: for both cells (``constv2ctl2-heldout``, ``constv2ctl2-full``)
    ``pairs_<cell>.jsonl`` and ``pairs_<cell>.manifest.json`` exist and each manifest
    carries the ``preflight`` stamp ``build_pools.run_preflight`` writes;
    ``pools_summary.json`` exists; ``preflight_report.json`` exists and records an
    EMPTY ``violations`` list (the preflight PASSED). Anything else is incomplete.

``commit --root <root> --dir <built dir> --budget-manifest <mix-full manifest>``
    re-checks COMPLETE + ``verify_pools`` on the built dir (REFUSED otherwise), checks
    the build's recorded budget against the current one, that its recorded recipe
    exists and equals today's builder's (the builder that just ran disagrees with
    the module otherwise — impossible unless the manifests were tampered or the
    build ran from another checkout) and, when an expectation sidecar exists for the
    dir, that the rebuild reproduced the recorded bytes (all below; REFUSED by name
    otherwise, and then NOTHING is written), writes the pointer ``<root>.CURRENT``
    atomically (tmp + ``os.replace``; JSON with the dir, the UTC stamp, each cell's
    ``pool_sha256``, the budget and the recipe), renames a checked sidecar to
    ``<dir>.expect.checked.json`` (never deleted) and prints ``CURRENT <dir>``.

``identity --root <root> --budget-manifest <mix-full manifest> [--volume <vol dir>]``
    prints ONE line of ``key=value`` tokens (no whitespace inside a value) for the
    stage script's DONE-marker check (Sol re-check of D98 i, P1: a marker that only
    said "completed" was honoured after a recipe change, so ``decide`` never ran):
    ``declared_recipe=<recipe_digest of current_recipe()>``
    ``budget=<the manifest's pool_sha256>:<its official_tokens>``
    ``current_dir=<the CURRENT dir (the pointer's, else the root) | none>``
    ``recorded_recipe=<recipe_digest of that dir's full-cell manifest recipe | none>``
    ``recorded_budget=<its budget_source pool_sha256:official_tokens | none>``
    ``volume_links=<ok | ok:copy | unchecked | no_current_dir | pairs_<cell>.<jsonl|manifest.json>:<why>>``.
    ``none`` for the middle three when there is no (non-empty) current dir, no readable
    manifest, no recipe block or no usable budget_source — this command READS and never
    judges (``decide`` refuses by name); exit 3 ``REFUSED:`` when the budget manifest is
    missing or unreadable (it is the chain's own output: absent means the constitutional
    pools were never built), the pointer is unreadable, or a dir has whitespace in its
    path. ``volume_links`` (Sol, on 17a20bcb5; ``ok:copy`` Sol, on 708a7db63) says whether
    the volume the fits read — ``--volume``, the chain's ``$SHARED/vol/pool/<fit key>`` —
    actually HOLDS the CURRENT build's bytes. Its grammar (one word, no whitespace, ``:``
    joining the two-part forms)::

        volume_links = "ok" | "ok:copy" | "unchecked" | "no_current_dir" | <file> ":" <why>
        <file>       = "pairs_" <cell> (".jsonl" | ".manifest.json")     (<cell> in CELLS order)
        <why>        = "missing_in_volume" | "missing_in_current_dir" | "not_the_current_builds_file"

    ``ok``: for BOTH ctl2 cells, ``<vol>/pairs_<cell>.jsonl`` and ``<vol>/pairs_<cell>.manifest.json``
    exist and are the SAME FILE (``os.path.samefile``, st_dev + st_ino — the chain's ``ln -f``)
    as the CURRENT dir's. ``ok:copy``: every file exists, at least one is NOT the same file,
    and every such file's sha256 equals the CURRENT dir's — the chain's loop is
    ``ln -f … || cp -f …``, so a site whose volume is on another filesystem than the build
    (EXDEV) holds byte-equal COPIES, which the fits read identically; the sha256 is computed
    for the non-same files only, so the common same-device path stats four inodes and hashes
    nothing. Else the FIRST reason, in cell order, pool before manifest: ``no_current_dir``,
    ``<file>:missing_in_volume``, ``<file>:missing_in_current_dir`` or
    ``<file>:not_the_current_builds_file`` — a different file whose bytes DIFFER: a STALE copy
    (the build was rebuilt after it was taken) or a link left on a superseded build (another
    selection). ``unchecked`` when ``--volume`` is not given (the token is always present,
    always last: the line's shape never changes).
    The stage honours a marker only when ALL hold: ``current_dir`` is not ``none``,
    ``recorded_recipe`` and ``recorded_budget`` are readable, the marker's ``recipe=``
    equals ``declared_recipe`` AND ``recorded_recipe``, its ``budget=`` equals ``budget``
    AND ``recorded_budget``, and ``volume_links`` is ``ok`` or ``ok:copy`` — the recorded
    side is what the shared volume ACTUALLY holds, which another checkout may have rebuilt
    under its own recipe and re-pointed (the marker is local to one checkout; the build is
    shared). After a chain it writes the marker from ``recorded_recipe`` / ``recorded_budget``,
    equal to the declared ones by construction after ``commit``, and only when
    ``volume_links`` is ``ok`` (recorded on the marker as ``links=hard``) or ``ok:copy``
    (``links=copy``): the link loop just ran — on REUSE as on BUILD, so a stale copy is
    refreshed to the CURRENT bytes — and anything else means it failed.

``recipe_digest(recipe)`` (pure) is the sha256 of the canonical JSON (sorted keys,
compact separators) of a recipe block with the ``RECIPE_PROVENANCE_ONLY`` leaves removed,
so a pins-file ``_doc`` edit or a staged rig's own manifest never changes it while any
compared leaf (a salt, a pin's revision, a corpus sha, the version) does.

The RECIPE identity (Sol, P1 follow-up on the budget check). The budget says how MUCH
the full cell holds; the recipe says WHAT was sampled and HOW, and the two are
independent: a checkout can change a salt, a pinned control release or the tokenizer
pins while the mix-full pool bytes and ``official_tokens`` stay equal, and the budget
check alone would then link an old control selection into the volume as if current.
``build_ctl2_pools.py build`` records ``recipe`` (``build_ctl2_pools.recipe_identity``)
on BOTH cell manifests: ``version`` (``CTL2_RECIPE_VERSION``, bumped by hand when the
sampling / cut / tokenization ALGORITHM changes), ``release`` + ``release_bytes_pinned``
+ ``corpus_shas`` (the control-input bytes), ``id_prefix``, ``cells``, ``budget_cell``,
``salts`` (``heldout`` / ``eval`` / ``draw``), ``split`` (``frac`` / ``eval_n``) and
``tokenizers`` (``pins_file``, ``pins_sha256``, ``pins`` = {name: repo / revision /
per-file sha256} from ``corpora/TOKENIZERS.json``, plus ``rig`` = the staged
tokenizers' own MANIFEST provenance). ``current_recipe`` declares today's from the
builder's constants — the chain always builds PINNED (``RELEASE_SHAS`` /
``CORPUS_RELEASE``) under the builder's default salts and split parameters
(``build_heldout_split.SALT_HELDOUT`` / ``SALT_EVAL`` / ``FRAC`` / ``EVAL_N``; the chain
line passes no ``--salt``) — and ``compare_recipe`` names every differing leaf
(``salts.heldout: X (recorded) != Y (current)``). Compared: every leaf of the declared
recipe. NOT compared (recorded provenance only, ``RECIPE_PROVENANCE_ONLY``):
``tokenizers.pins_sha256`` (the pins FILE's sha — an edit to its ``_doc`` alone is
metadata; the pins' repo / revision / file shas are the identity and ARE compared) and
``tokenizers.rig`` (this tool stages no tokenizers, so it has no current value; the
staging itself verifies the rig against the pins). Never the checkout's commit: a
metadata-only change must not invalidate completed work. A complete build with NO
recipe record (built before the recipe was recorded) is REFUSED by name — its
selection cannot be checked, so it is neither reused nor rebuilt over; move it aside
(or remove ``<root>.CURRENT``) to rebuild. A complete build whose recipe DIFFERS is
not a refusal: the chain BUILDs a fresh stamped sibling (the old dir untouched, the
differing fields on stderr) and ``commit`` moves the pointer, so the volume is linked
to a selection today's builder made. The recipe is checked only AFTER the build's
bytes are verified: the record lives in the manifests, and a manifest whose pool
bytes disagree with it is not trusted to say what recipe cut them. The pointer records
the recipe too and ``decide`` checks it against the dir's as a belt (a pointer from
before recipes were recorded carries none and is judged by the dir alone).

The BUDGET identity (Sol stage-E review, P1). The ctl2 FULL cell is budget-matched to
the constitutional ``constv2-mix-full`` pool: ``build_ctl2_pools.py build`` reads that
cell's manifest (``--budget-manifest``) and records what it read as ``budget_source``
on the full cell's manifest (``cell``, ``manifest``, ``manifest_sha256``,
``pool_sha256``, ``official_tokens``, …; the HELDOUT cell's manifest carries none —
that cell is cut by the split, not the budget). A complete ctl2 build is therefore
worth reusing only if it was cut against the budget the chain just rebuilt, and the
chain rebuilds it EVERY run: ``build_pools.py build`` rewrites
``pairs_constv2-mix-full.manifest.json`` (its ``generator`` provenance carries the
checkout's commit sha, so the manifest FILE's sha256 differs between runs from
different shas even when the pool bytes are identical) and ``build_pools.py preflight``
rewrites it again with a ``preflight`` stamp, both BEFORE the decision here. So the
identity compared is SEMANTIC, never the file's: the mix-full manifest's
``pool_sha256`` (the seeded pool bytes the budget was read from) and
``official_tokens`` (the budget number). ``manifest_sha256`` is recorded provenance
and is not compared. A recorded build whose ``budget_source`` disagrees on either, or
carries none (unverifiable: an older builder), is REFUSED by name — neither reused,
rebuilt over nor deleted; move it aside by hand (or remove ``<root>.CURRENT``) and
re-run. The pointer records the budget too, and ``decide`` checks it as a belt.

The EXPECTATION sidecar. When ``decide`` steps around a partial dir it writes a
sibling FILE ``<target>.expect.json`` (tmp + ``os.replace``; the builder refuses a
non-empty target, so nothing goes inside it) recording the partial's manifests'
``pool_sha256`` per cell (``previous_dir``, ``cells``, ``budget`` = the current
budget's ``pool_sha256`` + ``official_tokens``, ``utc``): the seeded rebuild must
reproduce those bytes, and ``commit`` refuses by name when a rebuilt cell's sha differs
from the recorded one — the pointer is NOT written and the chain's link loop never runs,
so the volume is NOT re-pointed. Only cells whose bytes are EXPECTED to reproduce are
recorded. First the partial's RECIPE (Sol re-check of D98 i, P1 (2)): the recipe its
readable cell manifests record (the full cell's ``recipe``; both cells when both are
present) must EQUAL today's declared one (``compare_recipe`` empty) — an interrupted
build followed by a salt / pin / release change makes the rebuilt bytes differ BY DESIGN,
and an expectation recorded anyway made the new, valid build fail ``check_expectation``
at commit so ``.CURRENT`` never moved. When it differs, or when no readable manifest
records a recipe, EVERY cell is skipped and the reason is named on stderr and in the
sidecar's ``skipped`` (``recipe changed (<field>: X (recorded) != Y (current)): the
rebuilt bytes are expected to differ`` / ``no recipe recorded on the partial build``),
with the differing fields under ``recipe_changed`` when there are any. Then, per cell:
the full cell is skipped when the partial's full manifest carries no ``budget_source``
(the builder writes the manifests once before the budget is stamped on and once after,
so a partial may hold the earlier write) or carries one that disagrees with the current
budget (the partial was cut against a different mix-full; its full bytes are not this
build's). A sidecar is written when at least one cell is recorded, and ALSO — with NO
cell expectations — whenever the expectations were skipped for the recipe (a COMPLETE
build stepped around because its recipe differs, or a partial whose recipe differs or is
unrecorded), so ``commit`` records which build the new one superseded (``previous_dir``
on the pointer) and why; the rebuilt bytes are EXPECTED to differ. A partial with no
readable manifest at all (a stray file) gets no sidecar: there is nothing to expect and
no recipe to compare.

stdlib only; ``build_ctl2_pools`` is imported lazily where a build is actually verified
or a recipe declared (that module's sibling imports are heavier than a first run needs).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

#: The two ctl2 cells — spelled here so a first run (nothing to verify) imports
#: nothing heavy; tests/test_ctl2_pools_reuse.py pins them equal to
#: build_ctl2_pools.HELDOUT_CELL / FULL_CELL (and BUDGET_CELL), and ``verify``
#: refuses a drift.
HELDOUT_CELL = "constv2ctl2-heldout"
FULL_CELL = "constv2ctl2-full"
CELLS = (HELDOUT_CELL, FULL_CELL)
#: The constitutional cell the full cell is budget-matched to (build_ctl2_pools.BUDGET_CELL).
BUDGET_CELL = "constv2-mix-full"
#: The two ``budget_source`` fields that identify a budget semantically (module docstring).
BUDGET_KEYS = ("pool_sha256", "official_tokens")
#: Recipe leaves that are recorded provenance, not identity (module docstring): the pins FILE's sha and the staged rig's
#: own manifest. Every other leaf of the declared recipe is compared.
RECIPE_PROVENANCE_ONLY = ("tokenizers.pins_sha256", "tokenizers.rig")
POINTER_SUFFIX = ".CURRENT"
BUILD_INFIX = ".build-"
EXPECT_SUFFIX = ".expect.json"
EXPECT_CHECKED_SUFFIX = ".expect.checked.json"
RC_REFUSED = 3


class Refusal(Exception):
    """A decision that must not be taken silently: printed as ``REFUSED: …``, exit 3."""


# --------------------------------------------------------------------------------------------------------------------
# pure pieces
# --------------------------------------------------------------------------------------------------------------------
def pointer_path(root: Path) -> Path:
    """``<root>.CURRENT`` — a sibling FILE of the canonical dir, never inside it."""
    root = Path(root)
    return root.with_name(root.name + POINTER_SUFFIX)


def expect_path(target: Path) -> Path:
    """``<target>.expect.json`` — the expectation sidecar, a sibling FILE of the build target."""
    target = Path(target)
    return target.with_name(target.name + EXPECT_SUFFIX)


def expect_checked_path(target: Path) -> Path:
    """``<target>.expect.checked.json`` — where ``commit`` renames a sidecar it has checked."""
    target = Path(target)
    return target.with_name(target.name + EXPECT_CHECKED_SUFFIX)


def read_pointer(root: Path) -> dict | None:
    """The pointer's JSON when the file exists (``dir`` required), None when it does not."""
    p = pointer_path(root)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refusal(f"pointer {p} is unreadable ({e}) — fix or remove it by hand; nothing is rebuilt over it")
    if not isinstance(d, dict) or not isinstance(d.get("dir"), str) or not d["dir"]:
        raise Refusal(f"pointer {p} names no dir — fix or remove it by hand")
    return d


def is_empty_dir(d: Path) -> bool:
    d = Path(d)
    return d.is_dir() and not any(d.iterdir())


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json_atomic(p: Path, payload: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def _utc(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))


def completeness(d: Path) -> str | None:
    """None when ``d`` is a COMPLETE ctl2 build, else the first reason it is not."""
    d = Path(d)
    if not d.is_dir():
        return "not a directory"
    for cell in CELLS:
        pool = d / f"pairs_{cell}.jsonl"
        mp = d / f"pairs_{cell}.manifest.json"
        if not pool.is_file():
            return f"{pool.name} missing"
        if not mp.is_file():
            return f"{mp.name} missing"
        try:
            m = _load_json(mp)
        except (OSError, ValueError) as e:
            return f"{mp.name} unreadable ({e})"
        if not isinstance(m, dict) or "preflight" not in m:
            return f"{mp.name} carries no preflight stamp"
    if not (d / "pools_summary.json").is_file():
        return "pools_summary.json missing"
    rp = d / "preflight_report.json"
    if not rp.is_file():
        return "preflight_report.json missing"
    try:
        rep = _load_json(rp)
    except (OSError, ValueError) as e:
        return f"preflight_report.json unreadable ({e})"
    violations = rep.get("violations") if isinstance(rep, dict) else None
    if violations != []:
        return "preflight_report.json does not record an empty violations list (the preflight did not PASS)"
    return None


def _builder():
    """The builder module, imported lazily (build_pools + build_heldout_split ride along) and checked for a drift of the
    names this tool spells itself."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import build_ctl2_pools as b2  # noqa: E402
    if (b2.HELDOUT_CELL, b2.FULL_CELL) != CELLS:
        raise Refusal(f"cell names drifted: build_ctl2_pools says {(b2.HELDOUT_CELL, b2.FULL_CELL)}, this tool {CELLS}")
    if b2.BUDGET_CELL != BUDGET_CELL:
        raise Refusal(f"budget cell drifted: build_ctl2_pools says {b2.BUDGET_CELL!r}, this tool {BUDGET_CELL!r}")
    return b2


def verify(d: Path) -> dict[str, str]:
    """{cell: pool_sha256} from ``build_ctl2_pools.verify_pools`` (pool bytes, row count and ordered doc ids
    against each manifest); any disagreement is a Refusal naming the cell."""
    b2 = _builder()
    try:
        rep = b2.verify_pools(Path(d), cells=CELLS)
    except b2.bp.SamplerError as e:
        raise Refusal(f"{d}: {e} — the recorded ctl2 build disagrees with its manifests; it is neither rebuilt "
                      "over nor deleted: inspect it, move it aside by hand, then re-run")
    return {cell: r["pool_sha256"] for cell, r in rep.items()}


# --------------------------------------------------------------------------------------------------------------------
# the budget identity
# --------------------------------------------------------------------------------------------------------------------
def budget_identity(src, *, where: str) -> dict:
    """The semantic budget ``{pool_sha256, official_tokens}`` of a manifest-like dict (the mix-full manifest itself, a
    recorded ``budget_source``, or a pointer's / sidecar's ``budget``); a Refusal naming ``where`` when either is unusable."""
    if not isinstance(src, dict):
        raise Refusal(f"{where}: not a JSON object")
    sha = src.get("pool_sha256")
    if not isinstance(sha, str) or not sha:
        raise Refusal(f"{where}: pool_sha256={sha!r} is not a non-empty string")
    tokens = src.get("official_tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        raise Refusal(f"{where}: official_tokens={tokens!r} is not a positive integer")
    return {"pool_sha256": sha, "official_tokens": tokens}


def read_budget_manifest(mp: Path) -> dict:
    """The CURRENT budget: ``{path, pool_sha256, official_tokens}`` read from the constv2-mix-full pool manifest the chain
    just rebuilt (``--budget-manifest``); refuses a missing / unreadable manifest or one of any other cell."""
    mp = Path(mp)
    if not mp.is_file():
        raise Refusal(f"budget manifest {mp} missing — the ctl2 full cell is budget-matched to the {BUDGET_CELL} pool "
                      "manifest, which the chain builds before this decision")
    try:
        m = _load_json(mp)
    except (OSError, ValueError) as e:
        raise Refusal(f"budget manifest {mp} is unreadable ({e})")
    if not isinstance(m, dict) or m.get("cell") != BUDGET_CELL:
        got = m.get("cell") if isinstance(m, dict) else None
        raise Refusal(f"{mp} is the manifest of cell {got!r}, not {BUDGET_CELL!r} — the control budget is that cell's "
                      "official token count and nothing else")
    ident = budget_identity(m, where=f"budget manifest {mp}")
    return {"path": str(mp), **ident}


def _describe(ident: dict) -> str:
    return f"pool_sha256 {ident['pool_sha256']} / official_tokens {ident['official_tokens']}"


def recorded_budget(d: Path) -> tuple[Path, dict | None]:
    """``(full-cell manifest path, its budget_source or None)`` for a ctl2 build dir whose full manifest is readable."""
    mp = Path(d) / f"pairs_{FULL_CELL}.manifest.json"
    m = _load_json(mp)
    src = m.get("budget_source") if isinstance(m, dict) else None
    return mp, (src if isinstance(src, dict) else None)


def check_recorded_budget(d: Path, current: dict, *, root: Path, verb: str) -> dict:
    """Refuse unless ``d``'s full-cell manifest records a ``budget_source`` whose semantic identity equals ``current``.
    Returns the recorded identity. ``verb`` is what the caller was about to do ("reused" / "recorded as CURRENT")."""
    mp, src = recorded_budget(d)
    fate = (f"it is neither {verb}, rebuilt over nor deleted: move it aside by hand (or remove {pointer_path(root)}) "
            "and re-run")
    if src is None:
        raise Refusal(f"{d}: {mp.name} records no budget_source — the recorded ctl2 build cannot be checked against the "
                      f"current {BUDGET_CELL} budget ({_describe(current)}, from {current['path']}); unverifiable, so {fate}")
    rec = budget_identity(src, where=f"{mp} budget_source")
    if any(rec[k] != current[k] for k in BUDGET_KEYS):
        raise Refusal(f"{d}: the recorded ctl2 build was cut against a different {BUDGET_CELL} budget — {mp.name} records "
                      f"budget_source {_describe(rec)}, the current budget manifest {current['path']} says "
                      f"{_describe(current)}; {fate}")
    return rec


def check_pointer_budget(ptr: dict, current: dict, *, root: Path) -> None:
    """Belt: a pointer that recorded a ``budget`` must agree with the current one (a pointer written before budgets were
    recorded carries none and is checked through its dir's manifest alone)."""
    if "budget" not in ptr:
        return
    p = pointer_path(root)
    rec = budget_identity(ptr["budget"], where=f"pointer {p} budget")
    if any(rec[k] != current[k] for k in BUDGET_KEYS):
        raise Refusal(f"pointer {p} recorded budget {_describe(rec)}, the current budget manifest {current['path']} says "
                      f"{_describe(current)} — the recorded ctl2 build at {ptr['dir']} was cut against a different "
                      f"{BUDGET_CELL} budget; it is neither reused, rebuilt over nor deleted: move it aside by hand "
                      f"(or remove {p}) and re-run")


# --------------------------------------------------------------------------------------------------------------------
# the recipe identity
# --------------------------------------------------------------------------------------------------------------------
def current_recipe() -> dict:
    """Today's DECLARED ctl2 recipe, from the builder's constants and nothing recorded: the chain always builds PINNED
    (``RELEASE_SHAS`` / ``CORPUS_RELEASE`` — other bytes refuse in the builder) under the builder's default salts and split
    parameters (its ``build_ctl2`` defaults are the ``build_heldout_split`` constants; the chain line passes no ``--salt``),
    with the tokenizer pins read from this checkout's ``corpora/TOKENIZERS.json``. No ``tokenizers.rig``: this tool stages
    nothing. A missing or malformed pins file is a Refusal (today's recipe cannot be declared, so nothing is compared)."""
    b2 = _builder()
    hs = b2.hsplit
    try:
        pins = b2.load_tokenizer_pins()
    except b2.bp.SamplerError as e:
        raise Refusal(f"cannot declare today's ctl2 recipe: {e}")
    return b2.recipe_identity(shas=dict(b2.RELEASE_SHAS), pinned=True, release_label=b2.CORPUS_RELEASE,
                              salt=hs.SALT_HELDOUT, eval_salt=hs.SALT_EVAL, frac=hs.FRAC, eval_n=hs.EVAL_N,
                              tokenizer_pins=pins)


def recorded_recipe(d: Path) -> tuple[Path, dict | None]:
    """``(full-cell manifest path, its recipe block or None)`` for a ctl2 build dir whose full manifest is readable; None
    means the build was made before recipes were recorded (or the block is not an object)."""
    mp = Path(d) / f"pairs_{FULL_CELL}.manifest.json"
    m = _load_json(mp)
    rec = m.get("recipe") if isinstance(m, dict) else None
    return mp, (rec if isinstance(rec, dict) else None)


def _leaves(d, prefix: str = "") -> dict[str, object]:
    """Dotted leaf paths of a nested dict (lists and scalars are leaves; an empty dict is a leaf too)."""
    out: dict[str, object] = {}
    for k in sorted(d):
        path = f"{prefix}{k}"
        v = d[k]
        if isinstance(v, dict) and v:
            out.update(_leaves(v, path + "."))
        else:
            out[path] = v
    return out


def _provenance_only(path: str) -> bool:
    return any(path == p or path.startswith(p + ".") for p in RECIPE_PROVENANCE_ONLY)


_ABSENT = object()


def _identity_leaves(recipe: dict) -> dict[str, object]:
    """The compared leaves of a recipe block: every leaf but the provenance-only ones."""
    return {k: v for k, v in _leaves(recipe).items() if not _provenance_only(k)}


def _fmt(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True)


def compare_recipe(recorded, current: dict) -> list[str]:
    """Every leaf of the DECLARED recipe ``current`` that ``recorded`` does not carry equal, as
    ``<dotted.path>: <recorded> (recorded) != <current> (current)`` (``(not recorded)`` for a missing leaf); the
    provenance-only leaves (``RECIPE_PROVENANCE_ONLY``) are never compared. Empty means equal. Leaves ``recorded`` carries
    beyond the declared ones are ignored: a field a later builder dropped is not a different selection — an algorithm
    change is what the version bump is for."""
    if not isinstance(recorded, dict):
        return [f"recipe: {_fmt(recorded)} (recorded) is not a JSON object"]
    rec, cur = _leaves(recorded), _leaves(current)
    diffs = []
    for path in sorted(cur):
        if _provenance_only(path):
            continue
        if path not in rec:
            diffs.append(f"{path}: (not recorded) != {_fmt(cur[path])} (current)")
        elif rec[path] != cur[path]:
            diffs.append(f"{path}: {_fmt(rec[path])} (recorded) != {_fmt(cur[path])} (current)")
    return diffs


def _fields(diffs: list[str]) -> str:
    return ", ".join(d.split(":", 1)[0] for d in diffs)


def _without_provenance(recipe: dict) -> dict:
    """A deep copy of ``recipe`` with the ``RECIPE_PROVENANCE_ONLY`` leaves removed (each is a fixed dotted path)."""
    out = copy.deepcopy(recipe)
    for path in RECIPE_PROVENANCE_ONLY:
        node = out
        parts = path.split(".")
        for k in parts[:-1]:
            node = node.get(k) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(parts[-1], None)
    return out


def recipe_digest(recipe: dict) -> str:
    """sha256 of the canonical JSON (sorted keys, compact separators, ASCII) of ``recipe`` without its provenance-only
    leaves: the identity a DONE marker certifies (module docstring). Equal recipes (``compare_recipe`` empty, no extra
    compared leaves) digest equal; any compared leaf changes it; the pins FILE's sha and the staged rig never do."""
    if not isinstance(recipe, dict):
        raise Refusal(f"recipe {_fmt(recipe)} is not a JSON object; no digest")
    canon = json.dumps(_without_provenance(recipe), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def require_recorded_recipe(d: Path, *, root: Path, verb: str) -> dict:
    """The recipe a complete build at ``d`` records — a Refusal by name when the full manifest carries none (built before
    recipes were recorded: unverifiable) or when the heldout manifest's block disagrees with the full cell's (one build
    records ONE recipe on both; a disagreement means tampered or mixed manifests). ``verb`` is what the caller was about
    to do ("reused" / "recorded as CURRENT")."""
    d = Path(d)
    mp, rec = recorded_recipe(d)
    if rec is None:
        raise Refusal(f"{d}: {mp.name} records no recipe — the recorded ctl2 build carries no recipe record (built before "
                      "the recipe was recorded); its control selection cannot be checked against today's builder, so it is "
                      f"neither {verb}, rebuilt over nor deleted: move it aside by hand (or remove {pointer_path(root)}) to "
                      "rebuild — never silently reused")
    mh = d / f"pairs_{HELDOUT_CELL}.manifest.json"
    m = _load_json(mh)
    other = m.get("recipe") if isinstance(m, dict) else None
    if not isinstance(other, dict):
        raise Refusal(f"{d}: {mh.name} records no recipe while {mp.name} does — one ctl2 build records ONE recipe on both "
                      f"cell manifests; these were tampered or mixed from two builds; neither {verb}, rebuilt over nor "
                      "deleted: inspect them by hand")
    a, b = _identity_leaves(other), _identity_leaves(rec)
    if a != b:
        fields = sorted(k for k in set(a) | set(b) if a.get(k, _ABSENT) != b.get(k, _ABSENT))
        raise Refusal(f"{d}: the two cell manifests disagree on the recipe ({', '.join(fields)}) — one ctl2 build records "
                      f"ONE recipe on both; these were tampered or mixed from two builds; neither {verb}, rebuilt over nor "
                      "deleted: inspect them by hand")
    return rec


def check_pointer_recipe(ptr: dict, recorded: dict, *, root: Path) -> None:
    """Belt: a pointer that recorded a ``recipe`` must agree with the one its dir records (a pointer written before
    recipes were recorded carries none and is judged through its dir alone)."""
    if "recipe" not in ptr:
        return
    p = pointer_path(root)
    diffs = compare_recipe(ptr["recipe"], recorded)
    if diffs:
        raise Refusal(f"pointer {p} recorded a recipe that differs from the one its dir {ptr['dir']} records — "
                      f"{'; '.join(diffs)} — the pointer or the build changed under it; the recorded ctl2 build is neither "
                      f"reused, rebuilt over nor deleted: inspect both by hand (or remove {p})")


def check_built_recipe(built: Path, current: dict, *, root: Path) -> dict:
    """At commit: the build that just finished must record a recipe equal to today's declared one — the builder that just
    ran and this module read the same checkout, so a difference is impossible unless the manifests were tampered or the
    build ran from another checkout. Returns the recorded recipe."""
    mp, rec = recorded_recipe(built)
    if rec is None:
        raise Refusal(f"{built}: {mp.name} records no recipe — the recorded ctl2 build carries no recipe record, yet the "
                      f"builder that just ran records one ({current['version']}) on every manifest: impossible unless the "
                      "manifests were tampered or the build ran from another checkout; nothing recorded as CURRENT, the "
                      "volume links are NOT re-pointed; inspect the dir by hand")
    rec = require_recorded_recipe(built, root=root, verb="recorded as CURRENT")     # + the two cells agree
    diffs = compare_recipe(rec, current)
    if diffs:
        raise Refusal(f"{built}: the build's recorded recipe differs from this checkout's builder — {'; '.join(diffs)} — "
                      "the builder that just ran disagrees with the module: impossible unless the manifests were tampered "
                      "or the build ran from another checkout; nothing recorded as CURRENT, the volume links are NOT "
                      "re-pointed; inspect the dir by hand")
    return rec


# --------------------------------------------------------------------------------------------------------------------
# the expectation sidecar
# --------------------------------------------------------------------------------------------------------------------
class PartialExpectation(NamedTuple):
    """What ``recorded_cell_shas`` found on a partial: the cells whose bytes must reproduce, the cells skipped (by reason),
    and — when the expectations were skipped for the RECIPE — why (``recipe_why``) and the differing fields
    (``recipe_changed``; None when the partial simply records no recipe)."""
    cells: dict[str, str]
    skipped: dict[str, str]
    recipe_why: str | None
    recipe_changed: list[str] | None


def partial_recipe_verdict(manifests: list[tuple[str, dict]], declared: dict) -> tuple[str | None, list[str] | None]:
    """``(why the expectations are skipped, the differing fields)`` — ``(None, None)`` when every readable manifest that
    records a recipe records one EQUAL to ``declared`` (``compare_recipe`` empty, so the seeded rebuild is expected to
    reproduce the bytes). ``manifests`` are ``(manifest name, its JSON object)`` for the partial's readable cell manifests;
    with none recording a recipe the verdict is ``no recipe recorded on the partial build``; with any recording a
    different one, ``recipe changed (<diffs>): the rebuilt bytes are expected to differ`` (the union of every differing
    field, sorted)."""
    present = [(name, m["recipe"]) for name, m in manifests if isinstance(m.get("recipe"), dict)]
    if not present:
        names = ", ".join(name for name, _ in manifests)
        return (f"no recipe recorded on the partial build ({names} carry no recipe block): its control selection cannot be "
                "told from today's builder's, so the rebuilt bytes are not expected to reproduce it"), None
    diffs = sorted({d for _, rec in present for d in compare_recipe(rec, declared)})
    if not diffs:
        return None, None
    return f"recipe changed ({'; '.join(diffs)}): the rebuilt bytes are expected to differ", diffs


def recorded_cell_shas(partial: Path, current: dict, note, *, declared: dict) -> PartialExpectation:
    """What the seeded rebuild beside a PARTIAL build is expected to reproduce, from the partial's manifests. The
    partial's recorded recipe must equal ``declared`` (today's, ``current_recipe()``) for ANY cell to be recorded
    (``partial_recipe_verdict``; module docstring); then the full cell is recorded only when its manifest's
    ``budget_source`` equals the current budget; a manifest that is missing, unreadable or carries no string
    ``pool_sha256`` is skipped. Every skip is named on ``note``."""
    partial = Path(partial)
    cells: dict[str, str] = {}
    skipped: dict[str, str] = {}
    readable: dict[str, dict] = {}
    for cell in CELLS:
        mp = partial / f"pairs_{cell}.manifest.json"
        if not mp.is_file():
            skipped[cell] = f"{mp.name} missing"
            continue
        try:
            m = _load_json(mp)
        except (OSError, ValueError) as e:
            skipped[cell] = f"{mp.name} unreadable ({e})"
            continue
        if not isinstance(m, dict):
            skipped[cell] = f"{mp.name} is not a JSON object"
            continue
        readable[cell] = m
    recipe_why, recipe_changed = (None, None) if not readable else partial_recipe_verdict(
        [(f"pairs_{cell}.manifest.json", m) for cell, m in readable.items()], declared)
    for cell, m in readable.items():
        mp = partial / f"pairs_{cell}.manifest.json"
        if recipe_why is not None:
            skipped[cell] = recipe_why
            continue
        sha = m.get("pool_sha256")
        if not isinstance(sha, str) or not sha:
            skipped[cell] = f"{mp.name} records no pool_sha256"
            continue
        if cell == FULL_CELL:
            src = m.get("budget_source")
            if not isinstance(src, dict):
                skipped[cell] = (f"{mp.name} records no budget_source (the builder stamps it on its final manifest "
                                 "write), so the budget its bytes were cut against is unknown")
                continue
            try:
                rec = budget_identity(src, where=f"{mp} budget_source")
            except Refusal as e:
                skipped[cell] = f"{e}"
                continue
            if any(rec[k] != current[k] for k in BUDGET_KEYS):
                skipped[cell] = (f"{mp.name} records budget_source {_describe(rec)}, the current budget is "
                                 f"{_describe(current)} — the partial was cut against a different {BUDGET_CELL} budget, "
                                 "so its full-cell bytes are not this build's")
                continue
        cells[cell] = sha
    for cell, why in skipped.items():
        note(f"note: {partial}: no reproduction expectation for {cell} — {why}")
    return PartialExpectation(cells, skipped, recipe_why, recipe_changed)


def write_expectation(target: Path, *, previous_dir: Path, cells: dict[str, str], skipped: dict[str, str],
                      current: dict, now: float | None = None, recipe_changed: list[str] | None = None) -> Path:
    """Write ``<target>.expect.json`` atomically; the caller has checked neither sidecar exists. ``recipe_changed`` (the
    differing recipe fields, when a COMPLETE build is being stepped around for its recipe) is recorded as given."""
    payload = {
        "target": str(target),
        "previous_dir": str(previous_dir),
        "cells": dict(cells),
        "skipped": dict(skipped),
        "budget": {k: current[k] for k in BUDGET_KEYS},
        "budget_manifest": current["path"],
        "utc": _utc(now),
    }
    if recipe_changed is not None:
        payload["recipe_changed"] = list(recipe_changed)
    p = expect_path(target)
    _write_json_atomic(p, payload)
    return p


def read_expectation(built: Path) -> tuple[Path, dict] | None:
    """``(sidecar path, its JSON)`` when ``<built>.expect.json`` exists, else None; an unreadable or malformed sidecar refuses."""
    p = expect_path(built)
    if not p.exists():
        return None
    try:
        d = _load_json(p)
    except (OSError, ValueError) as e:
        raise Refusal(f"expectation sidecar {p} is unreadable ({e}) — nothing recorded as CURRENT; inspect it by hand")
    if (not isinstance(d, dict) or not isinstance(d.get("cells"), dict) or not isinstance(d.get("previous_dir"), str)
            or not isinstance(d.get("budget"), dict)):
        raise Refusal(f"expectation sidecar {p} is malformed (previous_dir / cells / budget) — nothing recorded as CURRENT")
    return p, d


def check_expectation(built: Path, shas: dict[str, str], current: dict) -> tuple[Path, dict] | None:
    """When a sidecar exists for ``built``: its budget must be the current one and every recorded cell sha must equal
    the rebuilt cell's; a mismatch refuses by name. Returns ``(sidecar path, its JSON)`` or None when there is none."""
    found = read_expectation(built)
    if found is None:
        return None
    p, exp = found
    rec = budget_identity(exp["budget"], where=f"expectation sidecar {p} budget")
    if any(rec[k] != current[k] for k in BUDGET_KEYS):
        raise Refusal(f"expectation sidecar {p} was written against budget {_describe(rec)}, the budget manifest "
                      f"{current['path']} now says {_describe(current)} — the {BUDGET_CELL} budget changed between the "
                      "decision and this commit; NOT recorded as CURRENT, the volume links are NOT re-pointed; re-run")
    for cell, want in exp["cells"].items():
        if cell not in shas:
            raise Refusal(f"expectation sidecar {p} names an unknown cell {cell!r} (this tool knows {CELLS}) — nothing "
                          "recorded as CURRENT; inspect it by hand")
        if want != shas[cell]:
            raise Refusal(f"the seeded rebuild of {cell} at {built} produced sha {shas[cell]}, the previous build at "
                          f"{exp['previous_dir']} recorded {want} — the builder is not reproducing the recorded bytes; "
                          f"NOT recorded as CURRENT, the volume links are NOT re-pointed; inspect both dirs by hand "
                          f"(the sidecar {p} is left in place)")
    return p, exp


# --------------------------------------------------------------------------------------------------------------------
# decide / commit
# --------------------------------------------------------------------------------------------------------------------
def build_stamp(env=None, *, now: float | None = None, pid: int | None = None) -> str:
    """``job<SLURM_JOB_ID>`` under Slurm (one stamp per job, so a resubmit never collides with itself), else a UTC
    time + pid stamp."""
    env = os.environ if env is None else env
    job = (env.get("SLURM_JOB_ID") or "").strip()
    if job:
        return f"job{job}"
    t = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
    return f"{t}-pid{os.getpid() if pid is None else pid}"


def fresh_build_dir(root: Path, stamp: str) -> Path:
    root = Path(root)
    return root.with_name(f"{root.name}{BUILD_INFIX}{stamp}")


def decide(root: Path, budget_manifest: Path, *, env=None, note=None, now: float | None = None) -> tuple[str, Path]:
    """(``"REUSE"`` | ``"BUILD"``, dir) — see the module docstring. ``note`` receives human-readable asides
    (the CLI sends them to stderr; stdout carries the decision alone)."""
    root = Path(root)
    note = note or (lambda s: None)
    if any(ch.isspace() for ch in str(root)):
        raise Refusal(f"--root {root!r} contains whitespace; the chain reads the decision as two words")
    current_budget = read_budget_manifest(budget_manifest)
    ptr = read_pointer(root)
    if ptr is None:
        current = root
    else:
        current = Path(ptr["dir"])
        if not current.is_dir():
            raise Refusal(f"pointer {pointer_path(root)} names {current}, which does not exist (dangling) — "
                          "restore the dir or remove the pointer by hand; nothing is rebuilt over it")
        if is_empty_dir(current):
            raise Refusal(f"pointer {pointer_path(root)} names {current}, which is empty (dangling) — "
                          "restore the dir or remove the pointer by hand")
    if not current.exists() or is_empty_dir(current):
        return "BUILD", root
    reason = completeness(current)
    if reason is None:
        # the semantic budget first (cheap, and the more informative refusal), then the pointer's belt, then the recipe
        # RECORD must exist (unverifiable otherwise), then the bytes; only a build whose manifests are trusted has its
        # recorded recipe compared against today's (module docstring)
        check_recorded_budget(current, current_budget, root=root, verb="reused")
        if ptr is not None:
            check_pointer_budget(ptr, current_budget, root=root)
        rec = require_recorded_recipe(current, root=root, verb="reused")
        shas = verify(current)
        recorded = (ptr or {}).get("cells") or {}
        for cell, sha in shas.items():
            if cell in recorded and recorded[cell] != sha:
                raise Refusal(f"{current}: {cell} pool sha256 {sha} != the sha the pointer "
                              f"{pointer_path(root)} recorded ({recorded[cell]}) — the recorded build changed "
                              "under its pointer; inspect it by hand")
        if ptr is not None:
            check_pointer_recipe(ptr, rec, root=root)
        diffs = compare_recipe(rec, current_recipe())
        if not diffs:
            return "REUSE", current
        state = f"a complete ctl2 build cut under a different recipe ({_fields(diffs)})"
        target = _fresh_target(root, env, current, state)
        note(f"note: {current} is {state} — {'; '.join(diffs)} — left in place, never rebuilt over, and NOT reused: "
             f"its control selection is not today's builder's; building into {target}")
        skipped = {cell: f"recipe changed ({_fields(diffs)}): the rebuilt bytes are expected to differ from the previous "
                         "build's, so there is no reproduction expectation" for cell in CELLS}
        side = write_expectation(target, previous_dir=current, cells={}, skipped=skipped, current=current_budget, now=now,
                                 recipe_changed=diffs)
        note(f"note: no reproduction expectation for {', '.join(CELLS)} — the recipe changed; the previous build and the "
             f"differing fields are recorded in {side} for commit")
        return "BUILD", target
    state = f"an incomplete ctl2 build ({reason})"
    target = _fresh_target(root, env, current, state)
    note(f"note: {current} is {state} — left in place, never rebuilt over; building into {target}")
    # the partial's recorded recipe is compared with today's DECLARED one before any byte expectation is recorded (module
    # docstring, finding (2)); the declaration is this checkout's builder constants + pins, read once here
    found = recorded_cell_shas(current, current_budget, note, declared=current_recipe())
    if found.cells:
        side = write_expectation(target, previous_dir=current, cells=found.cells, skipped=found.skipped, current=current_budget,
                                 now=now)
        note(f"note: the seeded rebuild must reproduce the recorded pool sha256 of {', '.join(sorted(found.cells))} from "
             f"{current}; recorded in {side} for commit to check")
    elif found.recipe_why is not None:
        # skipped for the RECIPE: a sidecar with no cell expectations still records which build the new one supersedes and
        # why (commit copies previous_dir / recipe_changed onto the pointer); the rebuilt bytes are expected to differ
        side = write_expectation(target, previous_dir=current, cells={}, skipped=found.skipped, current=current_budget, now=now,
                                 recipe_changed=found.recipe_changed)
        note(f"note: no reproduction expectation for {', '.join(sorted(found.skipped))} — {found.recipe_why}; the previous "
             f"build is recorded in {side} for commit")
    return "BUILD", target


def _fresh_target(root: Path, env, current: Path, state: str) -> Path:
    """The stamped sibling a build steps into when ``current`` (``state`` says why) is not built over; a target or a
    sidecar for it that already exists refuses."""
    target = fresh_build_dir(root, build_stamp(env))
    if target.exists():
        raise Refusal(f"{target} already exists — the fresh build target must not; move it aside by hand "
                      f"({current} is left in place: {state})")
    for side in (expect_path(target), expect_checked_path(target)):
        if side.exists():
            raise Refusal(f"{side} already exists — an expectation sidecar for the fresh build target {target} must not "
                          f"(a previous decision or commit left it); move it aside by hand ({current} is left in place: "
                          f"{state})")
    return target


def commit(root: Path, built: Path, budget_manifest: Path, *, env=None, now: float | None = None) -> Path:
    """Record ``built`` as the CURRENT build of ``root`` (after COMPLETE + verify + the budget, recipe and expectation
    checks); returns the pointer path. A refusal leaves everything as it was."""
    root = Path(root)
    built = Path(os.path.abspath(built))
    reason = completeness(built)
    if reason is not None:
        raise Refusal(f"{built} is not a complete ctl2 build ({reason}) — nothing recorded as CURRENT")
    shas = verify(built)
    current_budget = read_budget_manifest(budget_manifest)
    check_recorded_budget(built, current_budget, root=root, verb="recorded as CURRENT")
    recipe = check_built_recipe(built, current_recipe(), root=root)
    checked = check_expectation(built, shas, current_budget)
    if checked is not None and expect_checked_path(built).exists():
        raise Refusal(f"{expect_checked_path(built)} already exists beside the sidecar {checked[0]} — a previous commit "
                      f"of {built} left it; nothing recorded as CURRENT, nothing deleted; inspect both by hand")
    env = os.environ if env is None else env
    payload = {
        "dir": str(built),
        "root": str(root),
        "utc": _utc(now),
        "slurm_job_id": (env.get("SLURM_JOB_ID") or "").strip() or None,
        "cells": shas,
        "budget": {k: current_budget[k] for k in BUDGET_KEYS},
        "budget_manifest": current_budget["path"],
        "recipe": recipe,
        # the build this one was checked against (a partial's recorded bytes) — None when nothing was reproduced …
        "reproduced_from": checked[1]["previous_dir"] if checked is not None and checked[1]["cells"] else None,
        # … and the build it stepped around, whatever the reason (a partial, or a complete build under another recipe)
        "previous_dir": checked[1]["previous_dir"] if checked is not None else None,
        "recipe_changed": checked[1].get("recipe_changed") if checked is not None else None,
    }
    p = pointer_path(root)
    _write_json_atomic(p, payload)
    if checked is not None:
        os.rename(checked[0], expect_checked_path(built))     # kept, never deleted: the record of what was checked
    return p


# --------------------------------------------------------------------------------------------------------------------
# identity (the DONE marker's certified identities)
# --------------------------------------------------------------------------------------------------------------------
IDENTITY_KEYS = ("declared_recipe", "budget", "current_dir", "recorded_recipe", "recorded_budget", "volume_links")
NONE = "none"
#: ``volume_links`` values that are not a ``<file>:<why>`` reason (module docstring)
LINKS_OK = "ok"
LINKS_OK_COPY = "ok:copy"
LINKS_UNCHECKED = "unchecked"
LINKS_NO_CURRENT_DIR = "no_current_dir"
#: the ``<why>`` half of a ``pairs_<cell>.<suffix>:<why>`` reason
LINKS_MISSING_IN_VOLUME = "missing_in_volume"
LINKS_MISSING_IN_CURRENT = "missing_in_current_dir"
LINKS_NOT_CURRENT = "not_the_current_builds_file"
#: the volume files checked per cell, in the order the first reason is reported
CELL_FILES = tuple(f"pairs_{cell}{suffix}" for cell in CELLS for suffix in (".jsonl", ".manifest.json"))


def _no_whitespace(p: Path, what: str) -> None:
    if any(ch.isspace() for ch in str(p)):
        raise Refusal(f"{what} {str(p)!r} contains whitespace; the stage script reads the identity line as space-separated tokens")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def volume_links(volume: Path, current: Path) -> str:
    """``ok`` when every ``CELL_FILES`` entry exists in ``volume`` and is the SAME FILE (``os.path.samefile``: st_dev + st_ino,
    symlinks followed — the chain hard-links with ``ln -f``) as ``current``'s; ``ok:copy`` when every entry exists, at least
    one is a different file and every different file's sha256 equals ``current``'s (``cp -f``, the loop's cross-device
    fallback: byte-equal copies read the same; hashed only for the non-same files, so a linked volume reads no bytes); else
    the FIRST reason in ``CELL_FILES`` order as ``<file>:<why>`` (no whitespace: the stage reads the line as tokens). A file
    the volume lacks is ``missing_in_volume``; one the CURRENT dir lacks (an incomplete build) ``missing_in_current_dir``; a
    different file with DIFFERENT bytes — a STALE copy, or a link still on a superseded build — ``not_the_current_builds_file``.
    A copy that cannot be READ to compare is a ``Refusal`` (exit 3 by name), never a traceback and never a verdict."""
    volume, current = Path(volume), Path(current)
    copied = False
    for name in CELL_FILES:
        v, c = volume / name, current / name
        if not v.is_file():
            return f"{name}:{LINKS_MISSING_IN_VOLUME}"
        if not c.is_file():
            return f"{name}:{LINKS_MISSING_IN_CURRENT}"
        if os.path.samefile(v, c):
            continue
        try:
            same_bytes = _sha256_file(v) == _sha256_file(c)
        except OSError as e:
            raise Refusal(f"volume_links: cannot read {name} to compare a copy with the CURRENT build's ({e})") from e
        if not same_bytes:
            return f"{name}:{LINKS_NOT_CURRENT}"
        copied = True
    return LINKS_OK_COPY if copied else LINKS_OK


def identity(root: Path, budget_manifest: Path, volume: Path | None = None) -> dict[str, str]:
    """The six ``IDENTITY_KEYS`` (module docstring): today's declared recipe digest and the current budget, then what the
    CURRENT dir (the pointer's, else the root) records — ``none`` where there is nothing to read — and, when ``volume`` is
    given, whether that volume holds the CURRENT dir's files (``volume_links``; ``unchecked`` otherwise). Reads only; the
    one judgement (a refusal) is a missing / unreadable budget manifest, an unreadable pointer or whitespace in a path."""
    root = Path(root)
    _no_whitespace(root, "--root")
    current_budget = read_budget_manifest(budget_manifest)
    out = {"declared_recipe": recipe_digest(current_recipe()),
           "budget": f"{current_budget['pool_sha256']}:{current_budget['official_tokens']}",
           "current_dir": NONE, "recorded_recipe": NONE, "recorded_budget": NONE,
           "volume_links": LINKS_UNCHECKED if volume is None else LINKS_NO_CURRENT_DIR}
    ptr = read_pointer(root)
    current = Path(ptr["dir"]) if ptr is not None else root
    if not current.is_dir() or is_empty_dir(current):
        return out
    _no_whitespace(current, "the CURRENT dir")
    out["current_dir"] = str(current)
    if volume is not None:
        out["volume_links"] = volume_links(volume, current)
    try:
        _, rec = recorded_recipe(current)
    except (OSError, ValueError):
        rec = None
    if rec is not None:
        out["recorded_recipe"] = recipe_digest(rec)
    try:
        _, src = recorded_budget(current)
        b = budget_identity(src, where="recorded budget_source") if src is not None else None
    except (OSError, ValueError, Refusal):
        b = None
    if b is not None:
        out["recorded_budget"] = f"{b['pool_sha256']}:{b['official_tokens']}"
    return out


def identity_line(ident: dict[str, str]) -> str:
    """The one stdout line: ``key=value`` tokens in ``IDENTITY_KEYS`` order."""
    return " ".join(f"{k}={ident[k]}" for k in IDENTITY_KEYS)


# --------------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    budget_help = (f"the {BUDGET_CELL} pool manifest the ctl2 full cell is budget-matched to "
                   f"(the chain's $POOLS/pairs_{BUDGET_CELL}.manifest.json, rebuilt every run)")
    d = sub.add_parser("decide", help="print `REUSE <dir>` or `BUILD <dir>` for the canonical pools_ctl2 dir")
    d.add_argument("--root", type=Path, required=True, help="the canonical pools_ctl2 dir (the chain's $CTL2)")
    d.add_argument("--budget-manifest", type=Path, required=True, help=budget_help)
    c = sub.add_parser("commit", help="record a complete, verified build as CURRENT (<root>.CURRENT)")
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--dir", type=Path, required=True, help="the dir the build + preflight just finished in")
    c.add_argument("--budget-manifest", type=Path, required=True, help=budget_help)
    i = sub.add_parser("identity", help="print one line of key=value tokens: today's declared recipe digest and budget, "
                                        "the CURRENT build's recorded ones (the DONE marker's certified identities) and "
                                        "whether the volume holds the CURRENT build's files")
    i.add_argument("--root", type=Path, required=True, help="the canonical pools_ctl2 dir (the chain's $CTL2)")
    i.add_argument("--budget-manifest", type=Path, required=True, help=budget_help)
    i.add_argument("--volume", type=Path, default=None,
                   help="the volume dir the fits read (the chain's $VOL = $SHARED/vol/pool/<fit key>): volume_links=ok when "
                        "both ctl2 cells' pool + manifest there are the SAME FILE as the CURRENT build's, ok:copy when the "
                        "ones that are not are byte-equal copies (a volume on another filesystem), else the first reason; "
                        "volume_links=unchecked without it")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "decide":
            mode, where = decide(args.root, args.budget_manifest, note=lambda s: print(s, file=sys.stderr))
            print(f"{mode} {where}")
        elif args.cmd == "identity":
            print(identity_line(identity(args.root, args.budget_manifest, args.volume)))
        else:
            commit(args.root, args.dir, args.budget_manifest)
            print(f"CURRENT {args.dir}")
    except Refusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return RC_REFUSED
    return 0


if __name__ == "__main__":
    sys.exit(main())
