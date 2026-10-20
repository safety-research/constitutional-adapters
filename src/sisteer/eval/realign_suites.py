"""Realign-LOCAL suite registry + the shared per-record hashing contract.

Mirrors ``eval/suites.py`` (a thin ``{name: (module, attr)}`` table with a
lazy ``get_runner``) but lives entirely inside the campaign so the shared
dispatcher is never touched (F19). The realign driver consumes THIS registry;
nothing here is registered in ``eval/suites.py::SUITES``.

Also the single home for the F13 record-hash helpers, so every suite runner
(agentic / mask / xstest / collateral) stamps its records identically:

- ``suite_revision`` — the pinned dataset sha or committed-artifact tag; the
  revision the item content was drawn from. Each loader supplies its own.
- ``item_sha`` — sha256 over the ARM-INDEPENDENT item content. Stable across
  arms, so analysis pairs on it (a prompting arm and a steered arm answering
  the same item share one ``item_sha``).
- ``prompt_sha`` — sha256 over the COMPLETE rendered prompt actually fed to
  the model. Proves the intervention rendered (a system prompt / injected
  sentinel changes it) and legitimately differs across arms; used for
  resume/intervention verification, never for pairing.
"""

from __future__ import annotations

import hashlib
import json
from importlib import import_module
from pathlib import Path

# suite name -> (module under sisteer.eval, entry-point attribute). Only the
# three judged held-out / benign suites; the collateral panels
# (MMLU/GSM8K/wikitext-ppl) are not judged suites and are driven directly.
REALIGN_SUITES: dict[str, tuple[str, str]] = {
    "agentic": ("sisteer.eval.agentic_misalignment.runner", "run_agentic_arm"),
    "mask": ("sisteer.eval.mask", "run_mask_arm"),
    "xstest": ("sisteer.eval.xstest", "run_xstest_arm"),
}


def get_runner(suite: str):
    """Resolve a realign suite name to its runner (import lazily, like the
    shared ``eval.suites.get_runner``)."""
    if suite not in REALIGN_SUITES:
        raise KeyError(
            f"unknown realign suite {suite!r}; known: {sorted(REALIGN_SUITES)}"
        )
    module_name, attr = REALIGN_SUITES[suite]
    return getattr(import_module(module_name), attr)


# -- F13 record-hash contract --------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def item_sha(content: dict | str) -> str:
    """Stable content hash of the ARM-INDEPENDENT item (what pairing keys on).

    Dicts are hashed canonically (sorted keys) so field order never changes
    the digest; strings are hashed as-is."""
    if isinstance(content, str):
        payload = content
    else:
        payload = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return _sha256(payload)


def prompt_sha(rendered: str) -> str:
    """Hash of the complete rendered prompt string actually sent to the model
    (arm composition included)."""
    return _sha256(rendered)


# -- arm composition (shared by every free-form arm runner) --------------------


def compose_system(safety_si: str | None, scenario_system: str | None) -> str | None:
    """Compose the arm's safety SI with a suite's own system prompt, safety SI
    FIRST (the pre-registered composition order), then the scenario/pressure
    scaffold. Either side may be ``None``."""
    parts = [p for p in (safety_si, scenario_system) if p]
    return "\n\n".join(parts) if parts else None


def arm_name(system: str | None, obj) -> str:
    """The arm label stamped on records: an object present ⇒ steered; else a
    system prompt present ⇒ prompted; else baseline."""
    if obj is not None:
        return "steered"
    return "prompted" if system else "baseline"


# -- item-level resume IO (shared by every free-form arm runner) ---------------


def read_done_ids(out_jsonl: str | Path, id_field: str) -> set[str]:
    """Ids already written to ``out_jsonl`` under ``id_field``, tolerant of a
    torn final line (a crash mid-append leaves a partial JSON row, which we
    skip rather than choke on)."""
    path = Path(out_jsonl)
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn last line — resume overwrites nothing, appends.
            if id_field in row:
                done.add(str(row[id_field]))
    return done


def append_records(out_jsonl: str | Path, records: list[dict]) -> None:
    """Append records as JSONL, flushing so a crash can't lose a completed
    batch (item-level resume then skips them on the next run)."""
    if not records:
        return
    path = Path(out_jsonl)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
