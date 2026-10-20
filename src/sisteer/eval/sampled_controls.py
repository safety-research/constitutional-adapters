"""Sampled-controls machinery for the judged over-refusal controls: the
per-item seed, the cap-escalation ladder, its per-item meta sidecar, and the
served judged-control runner (scripts/sdfbase/full22/sampled/DESIGN.md and
LADDER_RULES.md).

This module is the ONE home of these definitions. ``scripts/served_capability_run.py``
imports every one of them by name and defines none itself; a second driver
(a tree pinned to an older serving stack) imports the same module instead of
carrying a copy. Extracted 2026-09-12 from the driver, where each definition
had lived since the sampled re-run landed; every one of those definitions is
byte-identical to the driver's span at the commit the extraction branched
from (``6027389c``).

THIS FILE IS CANONICAL, AND IT HAS A DECLARED VENDOR. Branch ``arditi-cell1``
carries a copy of this module at the same path, ON PURPOSE and indefinitely:
that lane runs from a tree pinned byte-identical to the W8 serving stack, which
cannot merge main (main's ``src/sisteer/capeval/serving/`` has diverged by 16
files, and merging would break the production identity gate every cell of that
lane asserts), so it vendors this module instead of importing main's. Main is
the one place these definitions change; the vendored copy is re-mirrored from
here. ``tests/test_sampled_controls_drift.py`` is the standing contract that
keeps the two identical (every definition, and the whole body from the first
definition on) and is expected to live as long as the lane does. The relationship
describes itself in :data:`VENDOR_CONTRACT` (main's header, below the imports): which
copy is canonical, where the vendor is expected, the contract's scope and what it
does NOT cover, and the sha256 of every definition it is a contract with — so the
direction is read off a field, never inferred from an assertion's argument order,
and the check needs no network.

THE DRIVER SEAM. Four names the bodies reference at CALL time were module
globals of the driver and are deliberately not defined here: the driver's own
request builder ``_chat_payload`` and fan-out ``_fanout`` (each tree keeps its
versions), the sibling script module ``cell_controls_run`` (``ccr``), and the
``DRIVER_VERSION`` string the escalation identity check compares. The driver
binds them ONCE at import through :func:`bind_driver_globals`; until then each
is an :class:`_Unbound` sentinel that RAISES on any use (call, attribute,
comparison, truth test), naming itself and the seam, so an unbound seam is a
loud failure at the first ladder call and never a ``None`` that flows into a
store. A rebind to a DIFFERENT object is refused: two drivers binding two
builders into one module is exactly the quiet divergence this module exists
to prevent.

THE REPO LOCATOR. ``REPO`` is NOT in the seam. :func:`_ladder_decide` loads
``scripts/sdfbase/full22/sampled/ladder_decide.py`` under :data:`REPO`, which
:func:`repo_root` resolves from THIS FILE's location (the first parent holding
``pyproject.toml``), never from a caller or the working directory;
:func:`set_repo_root` is the explicit override for a pinned or relocated tree.

RE-EXPORTS, never copies: :func:`controls_cap` and :data:`CONTROLS_CAP_KEY`
from ``sisteer.eval.cell_controls`` (the judged controls' OWN cap,
``attacks.controls_max_new_tokens``, with the older ``attacks.target_max_new_tokens``
as the fallback for configs that predate the key: the 2026-09-11 multi-turn
ruling moved the latter to 4,096 and the controls must not move with it), and
:data:`FIT_PROVENANCE_TEMPLATE_CHECK_KEY` from ``sisteer.eval.sweep`` where it
lives (``_SOURCE_IDENTITY_KEYS`` reads it through ``sweep_mod``, verbatim).

The only lazy imports inside the bodies are ``importlib.util``
(``_ladder_decide``), ``sisteer.capeval.tiers.judged`` (``_run_judged_served``,
``_load_source_control``) and ``sisteer.capeval.manifest`` (``_read_source_spool``);
the module imports nothing from ``sisteer.capeval.tinycap`` or
``sisteer.capeval.serving``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from sisteer.eval.cell_controls import (SAMPLED_DECODE_SAMPLER_KEYS,
                                        SampledDecodeError,
                                        controls_gen_config)
from sisteer.eval import sweep as sweep_mod
# re-exports (module docstring): the controls' own cap accessor + its key, and
# the fit-provenance identity key, each imported from the module that owns it
from sisteer.eval.cell_controls import CONTROLS_CAP_KEY, controls_cap  # noqa: F401
from sisteer.eval.sweep import FIT_PROVENANCE_TEMPLATE_CHECK_KEY  # noqa: F401

# --------------------------------------------------------------------------- #
# THE VENDOR CONTRACT (main's own; NOT part of the shared body below)          #
# --------------------------------------------------------------------------- #
#: This module has one canonical home and a declared vendor that carries the SAME
#: body; the field says which is which, what is shared, what is NOT in scope, where
#: the vendored copy is expected to be found, and the CONTENT it is a contract with —
#: the sha256 of every definition of the shared body, of the whole body from
#: BODY_MARKER on, and of sampled_decode.py. ``tests/test_sampled_controls_drift.py``
#: asserts this module's own body against these hashes (deterministic, no ref fetch:
#: a shallow exact-sha checkout can run it), so a change to the body that does not
#: also re-mirror the vendor and update the field is red on the PR that makes it.
#: The vendor's port test carries the same table and asserts its copy against it.
#: The field lives in the HEADER, outside the shared body, on purpose: a value that
#: names the mirror's commit cannot live inside the bytes that commit contains, and
#: two sides pinning each other's commit would chase each other forever — so main
#: names the vendor's commit, and both sides pin the module by content.
#: The hash table is a plain "name sha256" text table (no assignment operator between
#: a KEY-named definition and a hex digest, so a secret scanner has nothing to match).
VENDOR_CONTRACT = {
    "canonical": {"ref": "main", "path": "src/sisteer/eval/sampled_controls.py"},
    "vendors": (
        {"ref": "arditi-cell1", "path": "src/sisteer/eval/sampled_controls.py",
         "why": "runs from a tree pinned byte-identical to the W8 serving stack; main's "
                "serving/ has diverged by 16 files and merging would break the production "
                "identity gate every cell asserts",
         "expected_at": {"ref": "arditi-cell1",
                         "sha": "3a85a2791e3cac9ba738361ebad039ce159b1d09"},
         "state": "LANDED on arditi-cell1 2026-09-12: the battery that generated configs from "
                  "that branch closed its stores (its last two cells are terminal), so the "
                  "mirror was merged and this field now names the lane tip rather than the "
                  "preparation branch"},
    ),
    "scope": {
        "covers": ("src/sisteer/eval/sampled_controls.py", "src/sisteer/eval/sampled_decode.py"),
        "excludes": {
            "src/sisteer/eval/cell_controls.py":
                "the vendor keeps its own pin (90284b59) for the cell_controls names it copied; "
                "main's controls_gen_config now routes through controls_cap, which reaches the "
                "vendor when it re-pins, and every vendor config carries target_max_new_tokens: "
                "512 (none carries controls_max_new_tokens), so the cap is unchanged meanwhile. "
                "The absent controls_cap on the vendor is NOT drift.",
        },
    },
    "shared": "every top-level definition, and the whole body from BODY_MARKER on, byte for byte",
    "header": "docstring and imports (and, on main only, this field); a vendor may own these",
    "direction": "a vendor is re-mirrored from the canonical copy; the canonical copy is "
                 "never edited to match a vendor",
    "enforced_by": {"main": "tests/test_sampled_controls_drift.py",
                    "arditi-cell1": "tests/test_sampled_controls_port.py"},
    "definition_sha256": dict(line.split() for line in """
CONTROLS_CAP_LADDER a7e3235035d768d435ba42170356d9580aeeeccff53fbec94c8d4cde9f30792d
FLOOR_SAMPLE_DEFAULT dca8df3b51a0b1460d6dec9c41605697e2be944c267afcdc36256ce6773cfb1b
FLOOR_SAMPLE_RULE 5c35c0b56d283d1835e3fa8beb8e358688b59a4ef2cbb1017d76551534169261
META_SUFFIX ef380bbec1c1eb342670235e4be335492d4030e44938830a544714b31911c3c8
META_ROW_KEYS db3c8c9001320cf34e5dec0684ce544fa89c1e9aa226c86d867f85e009faa324
META_REGEN_KEYS ab64a1b41abb502a86e8449044a7fa53ac63bbe9ebe3ad835312c6a3b42f3c6c
META_COPIED_KEYS c77dd6a10a6e1b5c561d94c7cee00c12856b6b6b406bad4c083f0b808519547c
META_EXCLUDED_KEY 4087860231e91f7a8d2a15fabeb805751aeb7e2b89e385e66586c00ba10ca376
META_EXCLUSIONS d5fc6bcccb4307ea75e846a7a2bb43f59a0c8bdf6f4b802946a12bf50c90258b
sampling_seed 5734d097cdbd33ea4d1de2f0eac80e786a6062fd137ecd2faf36832a8e4db66f
floor_rank_key 806870e14805aca421f605435d77bbe5ce7b562e244848385d47c1a68035d149
floor_sample_ids eda79c794467b23d8d76afa451bd9a033fec82851c2fd723f385efe8743eb14e
_decode_of af2eb4ab9cee6344433feec414c60601eb8a434250721118b2cebd7a4c33f4a9
_decode_block 941feec9ec12c0ca334bb41f4bfb7701061973bfbccfabfabe42aa2344abadb9
_text_sha 75a25546773874da3e54f6680280756b231c11ef749d6d150f513ed3c160bd06
GenMeta 20cf05318511efbd963f611b87d581feed4b9f173f63258b0f763b7b4e61cb1a
_first_divergence 685b6f281a8e76fdb607804d07b42a487100242022bcb8d6ca884894975d038b
_fork_fields 0b4d508d83c593c6c9ea04c7d3441de8b443dcc6c05036ed9f78f07fb2cb93d7
_meta_stats 1ccd0a32791fc446ba5204aa003fd7013aad12ffe2e4938a363414f26314c7a9
Escalation 383f0cd8d925b20a1b9d97580184f8cd42f3757646636abd88d89291abc9112a
SourceControl 65ae2e0b73bde872f62c78b31e93e89a5ac523ee2ef55c2d7927e9af8f7645ff
_ladder_decide 98b2246c8433d72755f9d996d2613f950073f5bf20e69789f6097e3e5a988072
_read_items_files e3c0a7f743033e0706d72a64c5924fe2ba4e8a10dbef73d65435226460a4c6df
_apply_items a7d8dfe9acb5caa42c548a6b6b7d266e725a3b86dd66fc63120b9d59701f2b0c
_locate_source_store a5aae9f47fb6b60b7926d42e59323feeb2ac80115eed25c3c3c076ba2202a4d1
_decoder_of 85123db60f958e238bab6b6ad02c7b911df01728505d2e1dcc107f7e96cebe78
_apply_escalation 15c2319ba2316185d10f0bca2494e295cb407452a5fb122f7c8ea09bb2b0f4d6
_run_judged_served 0db58eda749240e862bef7f1fd3354c01ff5f58152cd09407a27e75ac9d29b72
_SOURCE_IDENTITY_KEYS 1317dbcfb86f278f87cfc12b94cb2677db068d91a367b646dc21a552adfce85c
_validate_source_identity d11de7bc0398e4cdccbcef33afdb5b7e5a555c5bee1bcff868f55463c61b824a
_read_source_spool 5a6ca16fa00bc175da43a9fbbd25e48359a0fd8e439668088f7a844f15cb76fb
_load_source_control 269e32769f722550af3362c5a062478053de1d28ea8249941bb290f64887dc40
DRIVER_GLOBALS 195dcb22fe2b29ad5533964d644788ac1155dfff1a80fe4b2f6bbd4db3ce0b63
_Unbound e42b22f973b58cd9c0dfd34224cfdeb54406614a46503fdcbfaaf5614d783026
_chat_payload b420c7542b03b273a36bf9e824405bd0d68fbcb82f3775a6ccb06f182d568b85
_fanout c11ca91f3b9679315aba7c3c12265bc3759421f6397761d09c5ff8e694d0af29
ccr e1e91ccaeb5dff094ac6b0c2ae77bbb0cf4f073c91b79d4ebe533082d2fdb51d
DRIVER_VERSION 934f236ec69026d707c880b3ae44d62b7588de183e9ba666b96f254081a97e65
bind_driver_globals 0809123462d5c9978851ce855d52954c35329c1b6dc66e039c059472e1aba922
driver_globals_bound 837e9e25b7045c210b7a056df5872f215344f105d5af1fd88ce69b00d7f4a306
repo_root c3b8cc5fb59eaa4c679d63b9b6d01ee5205e208f6dfc1f75d8cfe6cc94a10402
set_repo_root 6db7a0f1473cdd67c8bcfd085c6b040f4ae1bbb2a936060e483a66ffd04fce66
REPO 3932e021657b377d650628c515e813930815ca39b16436df034449c0bafd1766
BODY_MARKER 75ebe41fe5cd1988a798585e15e1ca0cbbfc56db39b81ab803de44772b8e7ba5
""".splitlines() if line.strip()),
    "body_sha256": "dd2da28d77ee3e57809a839c396a115c37fff26ccba58d6fb9bc6121cc3f8170",
    "sampled_decode_sha256": "c476f1c9f9273e6b5516bbe1c7bb816b71ba08f3572a4c1bbc2a8f112976b64c",
}

#: The controls cap ladder (DESIGN.md + LADDER_RULES.md): a rung regenerates
#: ONLY the items that hit the cap at the rung below it (with ``--items``,
#: only the TRUNCATED ones among them), copies every other item's text
#: forward unchanged, and is a complete standard store in its own root.
#: ``--controls-cap`` must name a rung above the base, and the source store
#: under ``--escalate-from-root`` must sit exactly one rung below it. The
#: ceiling is 16,384 (the user, 2026-09-09).
CONTROLS_CAP_LADDER = (512, 2048, 8192, 16384)
#: ``--floor-sample`` default (the coordinator, 2026-09-09): on every
#: escalation run, this many BELOW-cap items of the source rung are re-drawn
#: at the new cap too — the re-draw noise floor the truncated items'
#: label-flip rate is read against. Chosen cell-INDEPENDENTLY
#: (``floor_sample_ids``), so the same items are sampled across cells.
FLOOR_SAMPLE_DEFAULT = 200
FLOOR_SAMPLE_RULE = ("rank the source rung's below-cap item_ids by "
                     "sha256(f'{seed_base}|floor|{control}|{item_id}') "
                     "hexdigest ascending and take the first k (all when "
                     "fewer than k exist)")
#: The per-item sidecar's name beside its spool:
#: ``gens_<name>.jsonl`` -> ``gens_<name>.meta.jsonl``.
META_SUFFIX = ".meta.jsonl"
#: Every meta row's keys (ids and numbers only — never text; the text is
#: bound by its sha256). ``regenerated``: this store drew the text itself
#: after the item hit the cap at the rung below (or as a floor sample);
#: false on a base store's fresh draws and on every copied-forward row.
#: ``source_cap``: the cap of the request that produced the text — a copied
#: row keeps the rung that produced it, a fresh or regenerated row carries
#: this store's cap. ``floor_sample``: a below-cap item re-drawn for the
#: noise floor (regenerated is true too).
META_ROW_KEYS = ("item_id", "sampling_seed", "max_tokens", "finish_reason",
                 "completion_tokens", "prompt_tokens", "at_cap", "regenerated",
                 "floor_sample", "source_cap", "text_sha256")
#: The prefix-vs-fork fields every REGENERATED row adds (the coordinator,
#: 2026-09-09): the rung text against the SAME item's text in the source
#: store. ``prefix_extension``: the source text is a prefix of the new text
#: (``identical_text`` is the degenerate case, counted separately);
#: ``first_divergence_char`` / ``first_divergence_token``: the index of the
#: first differing character / served-tokenizer token, null when the source
#: is a prefix; ``boundary_landing``: the source was at cap but the re-draw
#: finished in fewer tokens than the source's cap — the item was never
#: cap-limited on that draw.
META_REGEN_KEYS = ("source_finish_reason", "source_completion_tokens",
                   "identical_text", "prefix_extension",
                   "first_divergence_char", "first_divergence_token",
                   "boundary_landing")
#: the keys a copied-forward row takes from its source row verbatim
META_COPIED_KEYS = ("item_id", "sampling_seed", "max_tokens", "finish_reason",
                    "completion_tokens", "prompt_tokens", "at_cap",
                    "source_cap", "text_sha256")
#: LADDER_RULES.md: an at-cap source item that the decision tool left OFF the
#: escalation set (``--items``) is copied forward — still at cap — with WHY on
#: its meta row: loop | repetitive | boundary. Only on a copied (regenerated
#: false) at-cap row; never on a fresh or regenerated one.
META_EXCLUDED_KEY = "escalation_excluded"
META_EXCLUSIONS = ("loop", "repetitive", "boundary")

def sampling_seed(seed_base: int, trial: int, readout: str,
                  item_key: str) -> int:
    """DESIGN.md's per-request seed, cell-INDEPENDENT on purpose (common
    random numbers across cells and lanes): the first 4 bytes of
    sha256(f"{seed_base}|{trial}|{readout}|{item_key}"), big-endian.
    ``readout`` is the control name as spelled on ``--controls`` (``xstest``
    / ``or_bench_hard_1k``), ``item_key`` the dataset item_id."""
    digest = hashlib.sha256(
        f"{seed_base}|{trial}|{readout}|{item_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")

def floor_rank_key(seed_base: int, control: str, item_id: str) -> str:
    """The floor sample's ranking key (FLOOR_SAMPLE_RULE): no cell in it,
    so every cell samples the same items."""
    return hashlib.sha256(
        f"{seed_base}|floor|{control}|{item_id}".encode("utf-8")).hexdigest()

def floor_sample_ids(below_cap_ids, *, seed_base: int, control: str,
                     k: int) -> List[str]:
    """The first ``k`` of ``below_cap_ids`` under FLOOR_SAMPLE_RULE (all of
    them when fewer than k exist; none when k == 0)."""
    ranked = sorted(set(below_cap_ids),
                    key=lambda iid: floor_rank_key(seed_base, control, iid))
    return ranked[:max(0, int(k))]

def _decode_of(gen: dict) -> Optional[dict]:
    """The sampler keys a request body carries under ``gen`` — None for a
    greedy gen (the request is then byte-identical to what it always was)."""
    if not gen.get("do_sample"):
        return None
    return {k: gen[k] for k in SAMPLED_DECODE_SAMPLER_KEYS if k in gen}

def _decode_block(gen: dict) -> dict:
    """gen_stats' ``decode`` block — the static rows' shape (DESIGN.md):
    {do_sample, <preset keys>, cap, arm, seed_base, trial}. A GREEDY gen
    (do_sample false) records {do_sample: false, cap, arm: greedy}."""
    if not gen.get("do_sample"):
        return {"do_sample": False, "cap": int(gen["max_new_tokens"]),
                "arm": gen.get("arm", "greedy")}
    return {"do_sample": True, **(_decode_of(gen) or {}),
            "cap": int(gen["max_new_tokens"]), "arm": gen["arm"],
            "seed_base": int(gen["seed_base"]), "trial": int(gen["trial"])}

def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

class GenMeta:
    """The per-item sidecar beside a GenSpool: ``private/gens_<name>.meta.jsonl``,
    one JSON row per item (META_ROW_KEYS, + META_REGEN_KEYS on a regenerated
    row), fsynced per append, mode 0600 in the spool's 0700 private dir.
    Ids and numbers only — never text; ``text_sha256`` binds each row to the
    spooled text it describes.

    Written META-THEN-SPOOL per item, so on resume a row whose item never
    reached the spool (a kill between the two appends) is an orphan: the
    item is re-drawn and a fresh row appended. ``bound`` therefore keys each
    SPOOLED item to the LAST row whose text_sha256 matches its spooled text
    — never a positional guess — and refuses a spooled item with no matching
    row (torn by something other than the write order, e.g. a second writer;
    the spool's own differing-text refusal fires first for a sampled draw).
    Written only when a sampled decode is in force: a greedy store carries no
    sidecar, so a greedy store can never read as "0 at cap" (at_cap_controls
    refuses one)."""

    def __init__(self, spool_path: Path, *, repair_torn: bool = True) -> None:
        spool_path = Path(spool_path)
        if not spool_path.name.endswith(".jsonl"):
            raise RuntimeError(f"{spool_path}: not a .jsonl spool path")
        self.path = spool_path.with_name(
            spool_path.name[:-len(".jsonl")] + META_SUFFIX)
        self.repair_torn = repair_torn

    @classmethod
    def for_spool(cls, spool, **kw) -> "GenMeta":
        return cls(Path(spool.path), **kw)

    def exists(self) -> bool:
        return self.path.exists()

    def append(self, row: dict) -> None:
        """A row is exactly META_ROW_KEYS, plus EITHER META_REGEN_KEYS (a
        regenerated row) OR the one META_EXCLUDED_KEY (a copied-forward at-cap
        row the decision tool excluded from the escalation set —
        LADDER_RULES.md)."""
        keys = set(row)
        base = set(META_ROW_KEYS)
        if keys not in (base, base | set(META_REGEN_KEYS), base | {META_EXCLUDED_KEY}):
            raise RuntimeError(
                f"{self.path}: meta row for {row.get('item_id')!r} has keys "
                f"{sorted(keys)}; a row is exactly META_ROW_KEYS, optionally "
                f"+ META_REGEN_KEYS or + {META_EXCLUDED_KEY!r}")
        if not isinstance(row["at_cap"], bool) or not isinstance(
                row["regenerated"], bool) or not isinstance(row["floor_sample"], bool):
            raise RuntimeError(f"{self.path}: at_cap/regenerated/floor_sample "
                               "must be bools")
        if META_EXCLUDED_KEY in row:
            if row[META_EXCLUDED_KEY] not in META_EXCLUSIONS:
                raise RuntimeError(f"{self.path}: {META_EXCLUDED_KEY}="
                                   f"{row[META_EXCLUDED_KEY]!r} not in {META_EXCLUSIONS}")
            if row["regenerated"] or not row["at_cap"]:
                raise RuntimeError(f"{self.path}: {META_EXCLUDED_KEY} belongs only on a "
                                   "COPIED (regenerated false) AT-CAP row")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if new:
            os.chmod(self.path, 0o600)

    def rows(self) -> List[dict]:
        """Every row in file order. A torn FINAL line (a kill mid-append) is
        truncated away when ``repair_torn`` (this run's own sidecar: the item
        never reached the spool, so nothing is lost) and REFUSED otherwise
        (a source store is read, never written); a corrupt non-final line is
        damage and refuses either way."""
        if not self.path.exists():
            raise RuntimeError(f"{self.path}: no meta sidecar — a greedy "
                               "store, or one written before the sidecar "
                               "existed; nothing per-item can be read")
        text = self.path.read_text(encoding="utf-8")
        lines = text.splitlines()
        out: List[dict] = []
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1 and not text.endswith("\n") and self.repair_torn:
                    keep = "".join(l + "\n" for l in lines[:i])
                    tmp = self.path.with_name(self.path.name + ".tmp")
                    tmp.write_text(keep, encoding="utf-8")
                    os.replace(tmp, self.path)
                    os.chmod(self.path, 0o600)  # the replace took the tmp's mode
                    print(f"[served-cap] {self.path}: truncated a torn final "
                          "meta line (interrupted append); item will rerun")
                    break
                raise RuntimeError(
                    f"{self.path}: corrupt meta row at line {i + 1}; refusing")
            if not isinstance(row, dict) or "item_id" not in row:
                raise RuntimeError(f"{self.path}: line {i + 1} is not a meta "
                                   "row; refusing")
            out.append(row)
        return out

    def bound(self, entries: Dict[str, str]) -> "tuple[Dict[str, dict], int]":
        """``({item_id: row}, n_superseded)`` over the SPOOLED ``entries``:
        per item the last row whose text_sha256 matches the spooled text.
        ``n_superseded`` counts rows that describe no spooled text (orphans
        of interrupted appends). A spooled item with no matching row
        refuses — its at-cap status cannot be read and is never guessed."""
        rows = self.rows()
        by_id: Dict[str, dict] = {}
        n_used = 0
        for row in rows:
            iid = row["item_id"]
            if iid in entries and row.get("text_sha256") == _text_sha(entries[iid]):
                by_id[iid] = row
        n_used = len(by_id)
        missing = [iid for iid in entries if iid not in by_id]
        if missing:
            raise RuntimeError(
                f"{self.path}: {len(missing)} spooled item(s) have no meta row "
                f"matching their spooled text (first: {missing[:3]}) — their "
                "at-cap status cannot be read; refusing rather than guessing")
        return by_id, len(rows) - n_used

def _first_divergence(a, b) -> int:
    """Index of the first position where sequences a and b differ, or the
    shorter length when one is a prefix of the other."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n

def _fork_fields(source_text: str, new_text: str, tok, src_row: dict,
                 new_completion_tokens) -> dict:
    """META_REGEN_KEYS for one regenerated item (module constants)."""
    identical = new_text == source_text
    prefix = new_text.startswith(source_text)
    if prefix:
        char = tok_idx = None
    else:
        char = _first_divergence(source_text, new_text)
        ids = tok([source_text, new_text], add_special_tokens=False)["input_ids"]
        tok_idx = _first_divergence(list(ids[0]), list(ids[1]))
    landing = bool(src_row["at_cap"]
                   and isinstance(new_completion_tokens, int)
                   and new_completion_tokens < int(src_row["max_tokens"]))
    return {"source_finish_reason": src_row["finish_reason"],
            "source_completion_tokens": src_row["completion_tokens"],
            "identical_text": identical, "prefix_extension": prefix,
            "first_divergence_char": char, "first_divergence_token": tok_idx,
            "boundary_landing": landing}

def _meta_stats(bound: Dict[str, dict], *, cap: int, gen: dict,
                n_superseded: int, meta_path: Path) -> dict:
    """gen_stats[control]'s sampled additions, from the COMPLETE bound meta
    (every item, not just this invocation's): the finish_reason counts the
    judge reads its at-cap share from, the at-cap ids, the copy/regenerate
    split, the floor sample and the prefix-vs-fork tallies."""
    rows = list(bound.values())
    regen = [r for r in rows if r["regenerated"]]
    at_cap = sorted(r["item_id"] for r in rows if r["at_cap"])
    return {
        "finish_reasons": dict(sorted(Counter(
            r["finish_reason"] for r in rows).items(), key=lambda kv: str(kv[0]))),
        "cap": int(cap),
        "decode": _decode_block(gen),
        "n_at_cap": len(at_cap),
        "at_cap_item_ids": at_cap,
        "n_regenerated": len(regen),
        "n_copied": sum(1 for r in rows
                        if not r["regenerated"] and int(r["source_cap"]) != int(cap)),
        "n_drawn_here": sum(1 for r in rows if int(r["source_cap"]) == int(cap)),
        "n_floor_sample": sum(1 for r in rows if r["floor_sample"]),
        "n_boundary_landing": sum(1 for r in regen if r.get("boundary_landing")),
        "n_prefix_extension": sum(1 for r in regen if r.get("prefix_extension")),
        "n_identical_text": sum(1 for r in regen if r.get("identical_text")),
        "n_fork": sum(1 for r in regen if not r.get("prefix_extension")),
        "n_meta_superseded": int(n_superseded),
        "meta": str(meta_path),
        # LADDER_RULES.md: at-cap items copied forward because the decision
        # tool excluded them from the escalation set, by why
        "n_escalation_excluded": sum(1 for r in rows if META_EXCLUDED_KEY in r),
        "n_escalation_excluded_by_class": {
            c: sum(1 for r in rows if r.get(META_EXCLUDED_KEY) == c)
            for c in META_EXCLUSIONS},
    }

@dataclass
class Escalation:
    """One rung of the controls ladder, resolved from the CLI before the
    store identity is computed. ``items``: the at-cap disposition rows of
    ``--items`` keyed by (control, item_id) — None when the rung regenerates
    every at-cap item (today's behaviour, the running jobs')."""
    root: Path                 # the source rung's out-root (never written)
    src_cell_dir: Path
    src_manifest: dict
    source_cap: int
    cap: int
    floor_k: int
    items: Optional[Dict[tuple, dict]] = None
    items_files: Optional[List[str]] = None

@dataclass
class SourceControl:
    """One control's source-rung data, loaded and validated before this rung
    writes anything. ``excluded``: at-cap source items the decision tool
    left off the escalation set ({item_id: loop|repetitive|boundary}) —
    copied forward, never re-drawn."""
    control: str
    texts: Dict[str, str]
    meta: Dict[str, dict]
    at_cap_ids: List[str]
    floor_ids: List[str]
    excluded: Dict[str, str] = field(default_factory=dict)

    @property
    def regen_ids(self) -> set:
        return (set(self.at_cap_ids) - set(self.excluded)) | set(self.floor_ids)

def _ladder_decide():
    """The decision tool (scripts/sdfbase/full22/sampled/ladder_decide.py),
    loaded by path: the ONE reader of the at-cap disposition file both
    drivers' ``--items`` consume."""
    import importlib.util

    path = REPO / "scripts" / "sdfbase" / "full22" / "sampled" / "ladder_decide.py"
    spec = importlib.util.spec_from_file_location("_served_cap_ladder_decide", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod

def _read_items_files(paths: List[str], *, cap: int, cell_id: str,
                      controls: List[str]) -> Dict[tuple, dict]:
    """Every ``--items`` file (one per (cell, control) decision) merged into
    ``{(control, item_id): row}``. Each header must be a CONTROLS disposition
    at this rung's SOURCE cap for THIS cell; every row's control must be one
    this run requested (a file for another control is a wiring error, not
    something to skip); a key filed twice refuses."""
    ld = _ladder_decide()
    out: Dict[tuple, dict] = {}
    for p in paths:
        try:
            header, rows = ld.read_items_file(Path(p))
        except ld.Refusal as e:
            raise SystemExit(f"[served-cap] --items {p}: {e}")
        if header.get("kind") != "controls":
            raise SystemExit(f"[served-cap] --items {p}: kind={header.get('kind')!r} is not "
                             "'controls' — a static disposition file belongs to escalate_static.py")
        if int(header.get("cap", -1)) != int(cap):
            raise SystemExit(f"[served-cap] --items {p}: the file describes the store at cap "
                             f"{header.get('cap')!r}, this rung's source is at {cap}; refusing")
        if header.get("cell") != cell_id:
            raise SystemExit(f"[served-cap] --items {p}: cell {header.get('cell')!r} != this "
                             f"run's {cell_id!r}; refusing")
        if header.get("control") not in controls:
            raise SystemExit(f"[served-cap] --items {p}: control {header.get('control')!r} is "
                             f"not among this run's --controls {controls}; refusing")
        for r in rows:
            if r.get("control") != header["control"]:
                raise SystemExit(f"[served-cap] --items {p}: row control {r.get('control')!r} "
                                 f"!= header control {header['control']!r}")
            key = (r["control"], r["item_id"])
            if key in out:
                raise SystemExit(f"[served-cap] --items: {key!r} filed twice; refusing")
            out[key] = r
    return out

def _apply_items(items: Optional[Dict[tuple, dict]], control: str,
                 at_cap_ids: List[str], *, where: str) -> Dict[str, str]:
    """``{item_id: excluded_as}`` for ``control`` from the disposition rows:
    every at-cap source item must have a row (the file describes THIS
    store), every row must name an at-cap source item, and the excluded
    ones are those with ``escalate`` false. None (no --items) => nothing
    excluded: every at-cap item regenerates, as before."""
    if items is None:
        return {}
    rows = {iid: r for (c, iid), r in items.items() if c == control}
    at_cap = set(at_cap_ids)
    missing = sorted(at_cap - set(rows))
    stray = sorted(set(rows) - at_cap)
    if missing or stray:
        raise SystemExit(
            f"[served-cap] --items for {control} does not describe {where}: "
            f"{len(missing)} at-cap source item(s) without a disposition row (first: "
            f"{missing[:3]}), {len(stray)} row(s) naming items not at cap in the source "
            f"(first: {stray[:3]}); refusing to escalate on a mismatched decision")
    return {iid: r["excluded_as"] for iid, r in rows.items() if not r["escalate"]}

def _locate_source_store(root: Path, *, cell_id: str, ihash: str):
    """The SAME cell's store under the source root: same true cell id, same
    intervention hash. Exactly one match or a refusal."""
    cells_dir = Path(root) / "cells"
    if not cells_dir.is_dir():
        raise SystemExit(f"[served-cap] --escalate-from-root {root}: no cells/ "
                         "dir — not a served controls root")
    matches = []
    for d in sorted(cells_dir.iterdir()):
        mp = d / "manifest.json"
        if not mp.is_file():
            continue
        try:
            m = json.loads(mp.read_text())
        except json.JSONDecodeError as e:
            raise SystemExit(f"[served-cap] {mp}: unreadable manifest ({e}); "
                             "refusing to locate the source store past it")
        if m.get("cell_id") == cell_id and m.get("intervention_hash") == ihash:
            matches.append((d, m))
    if len(matches) != 1:
        raise SystemExit(
            f"[served-cap] --escalate-from-root {root}: {len(matches)} stores "
            f"match cell {cell_id!r} @ intervention {ihash} (need exactly 1): "
            f"{[str(d) for d, _ in matches]}")
    return matches[0]

def _decoder_of(gen: dict) -> dict:
    """The rung-invariant part of a gen: everything but the cap, the rung
    record, the floor sample and the HF batch width."""
    return {k: v for k, v in gen.items()
            if k not in ("max_new_tokens", "escalated_from", "floor_sample",
                         "batch_size")}

def _apply_escalation(args, cfg: dict, *, cell_id: str, ihash: str,
                      out_root: Path, controls: List[str]
                      ) -> "tuple[dict, Optional[Escalation]]":
    """Resolve ``--controls-cap N --escalate-from-root ROOT`` (both or
    neither) into an in-memory copy of the config whose ``sampled_decode``
    carries ``controls_cap = N``, ``escalated_from = {cap: <source rung cap>,
    root: str(ROOT)}`` and ``floor_sample = {k, rule}`` — BEFORE the gen and
    the store identity are computed, so the rung store gets its own key.
    Refuses: a greedy config (no ladder), an off-ladder N, a source root
    that is this run's out-root, a source store that is not this cell's,
    not sampled, not exactly one rung below N, not this config's base store
    (for the first rung), or drawn under a different decoder.

    ``--items`` (LADDER_RULES.md): the at-cap disposition file(s) the
    decision tool wrote for the SOURCE rung — read here so a bad file
    refuses before anything exists on disk; applied per control in
    ``_load_source_control``. Without it every at-cap item regenerates
    (today's behaviour, pinned)."""
    esc_flag = getattr(args, "controls_cap", None) is not None
    root_flag = bool(getattr(args, "escalate_from_root", ""))
    if esc_flag != root_flag:
        raise SystemExit("[served-cap] --controls-cap and --escalate-from-root "
                         "go together (both or neither)")
    if not esc_flag:
        if getattr(args, "floor_sample", None) is not None:
            raise SystemExit("[served-cap] --floor-sample rides an escalation "
                             "run (--controls-cap/--escalate-from-root); a "
                             "base store has no source rung to re-draw from")
        if getattr(args, "items", None):
            raise SystemExit("[served-cap] --items rides an escalation run "
                             "(--controls-cap/--escalate-from-root): it names the "
                             "at-cap items of the SOURCE rung to re-draw")
        return cfg, None
    try:
        base_gen = controls_gen_config(cfg)
    except SampledDecodeError as e:
        raise SystemExit(f"[served-cap] {args.config}: {e}")
    if not base_gen.get("do_sample"):
        raise SystemExit("[served-cap] --controls-cap needs a config with a "
                         "sampled_decode block applying to controls; a greedy "
                         "config has no cap ladder (greedy stores are never "
                         "re-generated — DESIGN.md)")
    cap = int(args.controls_cap)
    if cap not in CONTROLS_CAP_LADDER or cap == CONTROLS_CAP_LADDER[0]:
        raise SystemExit(f"[served-cap] --controls-cap {cap} is not a rung above "
                         f"the base on the ladder {CONTROLS_CAP_LADDER}")
    prev = CONTROLS_CAP_LADDER[CONTROLS_CAP_LADDER.index(cap) - 1]
    root = Path(args.escalate_from_root)
    if not root.is_dir():
        raise SystemExit(f"[served-cap] --escalate-from-root {root}: not a directory")
    if root.resolve() == Path(out_root).resolve():
        raise SystemExit(f"[served-cap] --escalate-from-root {root} is this run's "
                         "--out-root — a rung is NEVER written into its source "
                         "root (DESIGN.md: never in place)")
    src_dir, src_manifest = _locate_source_store(root, cell_id=cell_id, ihash=ihash)
    src_rc = src_manifest.get("resolved_config") or {}
    if src_rc.get("driver") != DRIVER_VERSION:
        raise SystemExit(f"[served-cap] {src_dir}: not a {DRIVER_VERSION} store "
                         f"(driver={src_rc.get('driver')!r}); refusing")
    src_gen = src_rc.get("gen") or {}
    if not src_gen.get("do_sample"):
        raise SystemExit(f"[served-cap] {src_dir}: the source store is GREEDY "
                         f"(gen={src_gen}); the ladder escalates sampled stores "
                         "only — greedy stores are never re-generated")
    if src_gen.get("readout") is not None:
        raise SystemExit(f"[served-cap] {src_dir}: the source store is a readout store "
                         f"(gen.readout={src_gen.get('readout')!r}) of the retired "
                         "sampled-rerun GSM8K leg; a judged-controls rung cannot escalate it")
    source_cap = int(src_gen.get("max_new_tokens", -1))
    if source_cap != prev:
        raise SystemExit(f"[served-cap] {src_dir}: source cap {source_cap} is not "
                         f"the rung below {cap} on {CONTROLS_CAP_LADDER} ({prev}); "
                         "the source must be exactly one rung down")
    if "escalated_from" not in src_gen and source_cap != int(base_gen["max_new_tokens"]):
        raise SystemExit(f"[served-cap] {src_dir}: the source is a base store at cap "
                         f"{source_cap} but this config's base cap is "
                         f"{base_gen['max_new_tokens']}; not this config's base store")
    floor = getattr(args, "floor_sample", None)
    k = FLOOR_SAMPLE_DEFAULT if floor is None else int(floor)
    if k < 0:
        raise SystemExit(f"[served-cap] --floor-sample {k} must be >= 0")
    cfg2 = copy.deepcopy(cfg)
    sd = cfg2["sampled_decode"]
    sd["controls_cap"] = cap
    sd["escalated_from"] = {"cap": source_cap, "root": str(root)}
    sd["floor_sample"] = {"k": k, "rule": FLOOR_SAMPLE_RULE}
    try:
        rung_gen = controls_gen_config(cfg2)
    except SampledDecodeError as e:
        raise SystemExit(f"[served-cap] rung config: {e}")
    if _decoder_of(src_gen) != _decoder_of(rung_gen):
        raise SystemExit(
            f"[served-cap] {src_dir}: the source store's decoder "
            f"{_decoder_of(src_gen)} != this config's {_decoder_of(rung_gen)} — "
            "a rung re-draws under the SAME preset, arm, seed_base and trial; "
            "refusing to mix decoders across rungs")
    items = None
    items_files = list(getattr(args, "items", None) or [])
    if items_files:
        items = _read_items_files(items_files, cap=source_cap, cell_id=cell_id,
                                  controls=controls)
    return cfg2, Escalation(root=root, src_cell_dir=src_dir,
                            src_manifest=src_manifest, source_cap=source_cap,
                            cap=cap, floor_k=k, items=items,
                            items_files=items_files or None)

async def _run_judged_served(post, *, route, control, spec, spool,
                             concurrency, meta: Optional[GenMeta] = None,
                             source: Optional[SourceControl] = None,
                             tok=None) -> dict:
    """One judged control's generation leg: raw prompts through the proxy,
    texts into the standard GenSpool (resumable; header already bound to
    the cell identity). assert_complete gates the finish, so a partial
    spool can never look done.

    Greedy (``meta`` None): byte-for-byte the leg it always was. Sampled
    (``meta`` given, iff ``spec.gen`` is a sampled decode): every draw
    carries the preset keys and its per-item ``sampling_seed``, and a meta
    row (META_ROW_KEYS) is appended beside each spooled text — meta first,
    then the spool (GenMeta's resume rule). A draw whose response carries
    no finish_reason or no usage counts is a per-item FAILURE (its at-cap
    status cannot be read), collected and refused like a transport error.

    A rung (``source`` given): items the source rung left at cap, plus the
    floor sample, are re-drawn at this cap with the SAME seed and their row
    gains the prefix-vs-fork fields (META_REGEN_KEYS, against the source
    text under ``tok``); every other item's text is COPIED forward
    unchanged with its source row (regenerated false, source_cap kept)."""
    from sisteer.capeval.tiers import judged

    gen = spec.gen
    decode = _decode_of(gen)
    cap = int(gen["max_new_tokens"])
    if (meta is None) != (decode is None):
        raise RuntimeError(f"{control}: a meta sidecar is written iff a sampled "
                           f"decode is in force (meta={meta is not None}, "
                           f"decode={decode is not None})")
    if source is not None and (decode is None or tok is None):
        raise RuntimeError(f"{control}: an escalation needs a sampled decode and "
                           "the served tokenizer")
    done = spool.entries()
    todo = [r for r in judged._permuted(spec.rows, spec.order_seed)
            if r["item_id"] not in done]
    seeds: Dict[str, int] = {}
    if decode is not None:
        seeds = {r["item_id"]: sampling_seed(int(gen["seed_base"]), int(gen["trial"]),
                                             control, r["item_id"])
                 for r in spec.rows}
    n_copied_now = 0
    draw = todo
    floor_set = set(source.floor_ids) if source is not None else set()
    if source is not None:
        regen = source.regen_ids
        draw = [r for r in todo if r["item_id"] in regen]
        for r in todo:
            iid = r["item_id"]
            if iid in regen:
                continue
            src_row = source.meta[iid]
            row = {k: src_row[k] for k in META_COPIED_KEYS}
            row["regenerated"] = False
            row["floor_sample"] = False
            if iid in source.excluded:  # an at-cap item the decision left at cap
                row[META_EXCLUDED_KEY] = source.excluded[iid]
            meta.append(row)
            spool.append(iid, source.texts[iid])
            n_copied_now += 1
    work = [(r["item_id"],
             _chat_payload(route.virtual_model, r["prompt"],
                           system=spec.system, max_tokens=cap,
                           decode=decode, seed=seeds.get(r["item_id"], 0)))
            for r in draw]
    results, errors, finish, rmeta = await _fanout(post, work,
                                                   concurrency=concurrency)
    n_new = 0
    for r in draw:
        iid = r["item_id"]
        text = results.get(iid)
        if text is None:
            continue  # its error is in `errors`; the item reruns on resume
        if meta is not None:
            m = rmeta[iid]
            unread = [k for k in ("finish_reason", "completion_tokens",
                                  "prompt_tokens") if m.get(k) is None]
            if unread:
                errors.append((iid, f"response carries no {unread} — a sampled "
                                    "draw's at-cap status and length cannot be "
                                    "read; refusing to spool it"))
                continue
            row = {"item_id": iid, "sampling_seed": seeds[iid], "max_tokens": cap,
                   "finish_reason": m["finish_reason"],
                   "completion_tokens": int(m["completion_tokens"]),
                   "prompt_tokens": int(m["prompt_tokens"]),
                   "at_cap": m["finish_reason"] == "length",
                   "regenerated": source is not None,
                   "floor_sample": iid in floor_set,
                   "source_cap": cap, "text_sha256": _text_sha(text)}
            if source is not None:
                row.update(_fork_fields(source.texts[iid], text, tok,
                                        source.meta[iid], int(m["completion_tokens"])))
            meta.append(row)
        spool.append(iid, text)
        n_new += 1
    if errors:
        raise SystemExit(
            f"[served-cap] {control}: {len(errors)} of {len(draw)} requests "
            f"FAILED (first: {errors[:3]}) — the spool holds {n_new} new "
            "items and resumes; re-run generate")
    spool.assert_complete([r["item_id"] for r in spec.rows])
    stats = {"control": control, "n_new": n_new + n_copied_now,
             "n_total": len(spec.rows), "finish_reasons": finish,
             "spool": str(spool.path)}
    if meta is not None:
        try:
            bound, n_superseded = meta.bound(spool.entries())
        except RuntimeError as e:
            raise SystemExit(f"[served-cap] {control}: {e}")
        stats.update(_meta_stats(bound, cap=cap, gen=gen,
                                 n_superseded=n_superseded, meta_path=meta.path))
        stats["n_drawn_now"] = n_new
        stats["n_copied_now"] = n_copied_now
        print(f"[served-cap] {control}: {n_new} drawn + {n_copied_now} copied now / "
              f"{len(spec.rows)} spooled; cap={cap} n_at_cap={stats['n_at_cap']} "
              f"n_regenerated={stats['n_regenerated']} n_copied={stats['n_copied']} "
              f"n_floor_sample={stats['n_floor_sample']} "
              f"finish={stats['finish_reasons']}")
        return stats
    print(f"[served-cap] {control}: {n_new} new / {len(spec.rows)} spooled "
          f"generations finish={finish}")
    return stats


_SOURCE_IDENTITY_KEYS = ("driver", "engine", "cell_id", "model_id",
                         "model_revision", "order_seed", "limit", "scale",
                         "system_sha", "judge_model", "subset_seed",
                         "steer_positions",
                         sweep_mod.FIT_PROVENANCE_TEMPLATE_CHECK_KEY)


def _validate_source_identity(esc: Escalation, resolved: dict,
                              model_revision) -> None:
    src_rc = esc.src_manifest["resolved_config"]
    diff = [k for k in _SOURCE_IDENTITY_KEYS
            if src_rc.get(k) != resolved.get(k)]
    if diff:
        raise SystemExit(
            f"[served-cap] {esc.src_cell_dir}: the source store differs from this "
            f"run on {diff}: source "
            f"{ {k: src_rc.get(k) for k in diff} } vs this run "
            f"{ {k: resolved.get(k) for k in diff} } — a rung is the SAME cell "
            "under the same identity with only the cap raised; refusing")
    if esc.src_manifest.get("model_revision") != model_revision:
        raise SystemExit(f"[served-cap] {esc.src_cell_dir}: source model_revision "
                         f"{esc.src_manifest.get('model_revision')!r} != this "
                         f"run's {model_revision!r}; refusing")


def _read_source_spool(path: Path, header: dict, expected_ids) -> Dict[str, str]:
    """A READ-ONLY twin of GenSpool's reader for the source store: the same
    header binding (canonical_json-normalised equality, r1 #8), the same
    first-write-wins dedup with a differing-duplicate refusal, the same
    completeness gate — but a torn or corrupt line REFUSES instead of being
    repaired in place (GenSpool rewrites a torn tail; a source root is never
    written)."""
    from sisteer.capeval.manifest import canonical_json
    from sisteer.capeval.tiers.judged import _HEADER_KEY

    if not path.exists():
        raise SystemExit(f"[served-cap] {path}: the source store has no spool "
                         "for this control; refusing")
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            raise SystemExit(f"[served-cap] {path}: corrupt or torn line {i + 1} in "
                             "the SOURCE spool — a complete store has none; "
                             "refusing (the source is never repaired in place)")
    if not rows or _HEADER_KEY not in rows[0]:
        raise SystemExit(f"[served-cap] {path}: no spool header; refusing")
    want = json.loads(canonical_json(dict(header)))
    if rows[0][_HEADER_KEY] != want:
        raise SystemExit(
            f"[served-cap] {path}: source spool header mismatch — written under "
            f"{rows[0][_HEADER_KEY]!r}, this run's source identity resolves "
            f"{want!r}; refusing to read it as this cell's source")
    texts: Dict[str, str] = {}
    for row in rows[1:]:
        iid, text = row["item_id"], row["text"]
        if iid in texts:
            if texts[iid] != text:
                raise SystemExit(f"[served-cap] {path}: item {iid!r} appears twice "
                                 "with DIFFERENT text in the source spool; refusing")
            continue
        texts[iid] = text
    expected = list(expected_ids)
    missing = [i for i in expected if i not in texts]
    extra = sorted(set(texts) - set(expected))
    if missing or extra:
        raise SystemExit(
            f"[served-cap] {path}: the source spool is not the complete item set "
            f"— {len(missing)} missing (first: {missing[:3]}), {len(extra)} "
            f"unexpected (first: {extra[:3]}); an incomplete source cannot be "
            "escalated")
    return {i: texts[i] for i in expected}


def _load_source_control(esc: Escalation, control: str, spec, *,
                         gen: dict) -> SourceControl:
    """Load + validate ONE control's source-rung data: the spool (header
    bound to the source's own manifest exactly as served_capability_judge
    rebuilds it, complete), the meta sidecar (present, bound to every
    spooled text, at_cap bools, the SAME per-item seeds this rung derives),
    and the source's gen_stats entry (the completeness stamp its producer
    wrote after assert_complete). Returns the texts, the bound meta, the
    at-cap ids (regenerated at this rung) and the floor sample."""
    from sisteer.capeval.tiers import judged

    src_manifest = esc.src_manifest
    src_rc = src_manifest["resolved_config"]
    if control not in (src_manifest.get("controls") or []):
        raise SystemExit(f"[served-cap] {esc.src_cell_dir}: control {control!r} was "
                         f"not generated in the source store (controls="
                         f"{src_manifest.get('controls')}); refusing")
    gs_path = esc.src_cell_dir / "gen_stats.json"
    if not gs_path.is_file():
        raise SystemExit(f"[served-cap] {esc.src_cell_dir}: no gen_stats.json — the "
                         "source run never completed; refusing to escalate an "
                         "incomplete store")
    gs = json.loads(gs_path.read_text()).get(control)
    if not isinstance(gs, dict) or "decode" not in gs or "n_at_cap" not in gs:
        raise SystemExit(f"[served-cap] {gs_path}: no sampled entry for {control!r} "
                         "(decode/n_at_cap absent) — not a completed sampled "
                         "store; refusing")
    src_spec = judged.JudgedSpec(
        task=spec.task, template=dict(src_manifest["row_constants"]),
        domain_of=dict(ccr.DOMAIN_OF), rows=spec.rows,
        order_seed=int(src_rc.get("order_seed", 0)), system=spec.system,
        judge_model=src_rc["judge_model"], gen=dict(src_rc["gen"]))
    header = judged.spool_header(
        src_spec, intervention_hash=src_manifest["intervention_hash"],
        model_revision=src_manifest["model_revision"])
    name = spec.task.split("/", 1)[1]
    path = esc.src_cell_dir / "private" / f"gens_{name}.jsonl"
    ids = [r["item_id"] for r in spec.rows]
    texts = _read_source_spool(path, header, ids)
    meta = GenMeta(path, repair_torn=False)
    if not meta.exists():
        raise SystemExit(f"[served-cap] {meta.path}: the source store has no meta "
                         "sidecar — per-item at-cap status cannot be read; "
                         "refusing (a greedy or pre-sidecar store never escalates)")
    try:
        bound, _ = meta.bound(texts)
    except RuntimeError as e:
        raise SystemExit(f"[served-cap] {e}")
    if int(gs.get("n_total", -1)) != len(ids):
        raise SystemExit(f"[served-cap] {gs_path}: {control} n_total={gs.get('n_total')} "
                         f"!= {len(ids)} expected items; refusing")
    seed_base, trial = int(gen["seed_base"]), int(gen["trial"])
    for iid in ids:
        row = bound[iid]
        if not isinstance(row.get("at_cap"), bool):
            raise SystemExit(f"[served-cap] {meta.path}: item {iid!r} at_cap="
                             f"{row.get('at_cap')!r} is not a bool; refusing")
        want = sampling_seed(seed_base, trial, control, iid)
        if row.get("sampling_seed") != want:
            raise SystemExit(f"[served-cap] {meta.path}: item {iid!r} was drawn with "
                             f"sampling_seed={row.get('sampling_seed')!r} but this "
                             f"rung derives {want} — a re-draw must reuse the SAME "
                             "seed; refusing")
    at_cap_ids = [i for i in ids if bound[i]["at_cap"]]
    if sorted(at_cap_ids) != sorted(gs.get("at_cap_item_ids") or []):
        raise SystemExit(f"[served-cap] {gs_path}: {control} at_cap_item_ids disagree "
                         "with the meta sidecar; the source store is inconsistent, "
                         "refusing")
    below = [i for i in ids if not bound[i]["at_cap"]]
    floor = floor_sample_ids(below, seed_base=seed_base, control=control,
                             k=esc.floor_k)
    excluded = _apply_items(esc.items, control, at_cap_ids, where=str(esc.src_cell_dir))
    return SourceControl(control=control, texts=texts, meta=bound,
                         at_cap_ids=at_cap_ids, floor_ids=floor, excluded=excluded)

# --------------------------------------------------------------------------- #
# THE DRIVER SEAM + THE REPO LOCATOR (this lane's additions, agreed with the   #
# module's upstream owner 2026-09-12; NOT copied definitions, and NOT in the    #
# byte-equality test's name list)                                              #
# --------------------------------------------------------------------------- #
#: The FOUR names the copied bodies reference at CALL time that were module
#: globals of the source driver and are deliberately not copied: the driver's
#: own request builder and fan-out (each tree keeps its versions), the sibling
#: script module ``cell_controls_run``, and the driver version string the
#: escalation identity check compares. The driver binds them ONCE at import
#: through :func:`bind_driver_globals`. Until then each is an :class:`_Unbound`
#: sentinel that RAISES on any use — call, attribute, comparison — naming
#: itself and the seam; a seam that silently yielded ``None`` is the defect
#: class this campaign has paid for repeatedly. A rebind to a DIFFERENT object
#: is refused: two drivers binding two builders into one module is exactly the
#: quiet divergence this port exists to prevent.
#:
#: ``REPO`` is NOT in the seam: it comes from :func:`repo_root`, a locator on
#: this module's own file (the first parent holding ``pyproject.toml``), with
#: :func:`set_repo_root` as the override for a pinned or relocated tree. That
#: is the seam that lets the same module serve a tree checked out at an older
#: pin and main alike.
DRIVER_GLOBALS = ("_chat_payload", "_fanout", "ccr", "DRIVER_VERSION")


class _Unbound:
    """A placeholder for a driver global that has not been bound. Every use
    raises the same message, so an unbound seam is a loud, named failure at
    the first ladder call — never a ``None`` that flows into a store."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def _refuse(self, *_a, **_k):
        raise RuntimeError(
            f"sisteer.eval.sampled_controls.{self._name} is UNBOUND — the driver must call "
            f"bind_driver_globals(...) once at import (names: {DRIVER_GLOBALS})")

    __call__ = __getattr__ = __eq__ = __ne__ = __truediv__ = __str__ = _refuse
    __bool__ = _refuse

    def __repr__(self) -> str:   # repr must not raise (tracebacks, logging)
        return f"<unbound sampled_controls.{object.__getattribute__(self, '_name')}>"


_chat_payload = _Unbound("_chat_payload")
_fanout = _Unbound("_fanout")
ccr = _Unbound("ccr")
DRIVER_VERSION = _Unbound("DRIVER_VERSION")


def bind_driver_globals(*, _chat_payload, _fanout, ccr, DRIVER_VERSION) -> None:
    """Bind the four call-time names (:data:`DRIVER_GLOBALS`) ATOMICALLY: the module's
    visible state goes from fully unbound to fully bound with nothing in between, and a
    refused call changes nothing. In order:

    1. the REBIND check, before anything else: a name already bound to a DIFFERENT
       object refuses, every such name named, so a second driver fails before the
       module is touched; the same objects again (or an equal ``DRIVER_VERSION``
       string) is idempotent;
    2. every incoming value is validated and EVERY offender is reported in ONE error:
       an :class:`_Unbound` sentinel handed back, a ``_chat_payload`` or ``_fanout``
       that is not callable, a ``ccr`` without the ``DOMAIN_OF`` the bodies read, a
       ``DRIVER_VERSION`` that is not a non-empty ``str``;
    3. only then are the four assigned, from the staging dict, in a single pass.

    (The first form validated and assigned NAME BY NAME, so a refusal at a later name
    left the earlier ones bound: a latent half-bound state. Fixed 2026-09-12 on the
    module owner's ruling; ``tests/test_sampled_controls_module.py`` asserts a refused
    bind leaves EVERY name a sentinel.)"""
    g = globals()
    staged = {"_chat_payload": _chat_payload, "_fanout": _fanout, "ccr": ccr,
              "DRIVER_VERSION": DRIVER_VERSION}
    # 1. refuse a rebind FIRST. A sentinel is never compared (it compares by raising);
    #    an offered sentinel is an offender for step 2, not a clash.
    clashes = [name for name, value in staged.items()
               if not isinstance(g[name], _Unbound) and not isinstance(value, _Unbound)
               and g[name] is not value and g[name] != value]
    if clashes:
        raise RuntimeError(
            "bind_driver_globals: refusing to rebind "
            + ", ".join(f"sampled_controls.{n} (already bound to {g[n]!r}; offered "
                        f"{staged[n]!r})" for n in clashes)
            + " — one driver per process binds this module; nothing was changed")
    # 2. validate every name; report every offender at once
    offenders = []
    for name, value in staged.items():
        if isinstance(value, _Unbound):
            offenders.append(f"{name}: an unbound sentinel")
        elif name in ("_chat_payload", "_fanout") and not callable(value):
            offenders.append(f"{name}: not callable ({type(value).__name__})")
        elif name == "ccr" and not hasattr(value, "DOMAIN_OF"):
            offered = getattr(value, "__name__", None) or type(value).__name__
            offenders.append(f"ccr: {offered} has no DOMAIN_OF (the bodies read ccr.DOMAIN_OF)")
        elif name == "DRIVER_VERSION" and (not isinstance(value, str) or not value):
            offenders.append(f"DRIVER_VERSION: not a non-empty str ({value!r})")
    if offenders:
        raise RuntimeError("bind_driver_globals: refused, nothing was bound — "
                           + "; ".join(offenders))
    # 3. one pass from the staging dict: fully unbound -> fully bound, no state between
    for name, value in staged.items():
        g[name] = value


def driver_globals_bound() -> list:
    """The names of :data:`DRIVER_GLOBALS` still unbound (empty = ready)."""
    return [n for n in DRIVER_GLOBALS if isinstance(globals()[n], _Unbound)]


def repo_root(start: "Path | None" = None) -> Path:
    """The repository root: the first parent of ``start`` (default: this
    file) that holds ``pyproject.toml``. Refuses loudly when none does — a
    module installed outside a checkout has no ``scripts/`` tree to locate
    ``ladder_decide.py`` in, and must say so rather than resolve a wrong root."""
    here = Path(start or __file__).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").is_file():
            return p
    raise RuntimeError(
        f"sampled_controls.repo_root: no pyproject.toml above {here} — this module is not "
        "inside a repository checkout; call set_repo_root(<checkout>) for a relocated tree")


def set_repo_root(path) -> Path:
    """The override for a pinned or relocated tree: point ``REPO`` at
    ``path`` (which must hold ``pyproject.toml``)."""
    global REPO
    p = Path(path).resolve()
    if not (p / "pyproject.toml").is_file():
        raise RuntimeError(f"set_repo_root: {p} holds no pyproject.toml")
    REPO = p
    return REPO


#: resolved at import from this file's location (never from a driver global);
#: :func:`set_repo_root` overrides it
REPO = repo_root()


#: the first line of the shared body (the byte-identity boundary the contract names)
BODY_MARKER = "#: The controls cap ladder (DESIGN.md + LADDER_RULES.md)"
