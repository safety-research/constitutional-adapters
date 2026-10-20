"""XSTest over-refusal suite (benign-control / collateral for H3).

Items come from the capeval pin ``xstest_v2`` (``configs/capeval/pins.yaml``:
``natolambert/xstest-v2-copy`` at a full revision), read through
``sisteer.capeval.corpora``'s pinned, local-cache-only loader — a direct
parquet read, NEVER ``load_dataset``, never the network at evaluation time.
An unfetched cache or a missing pin RAISES; there is no silent fallback to
an unpinned read. The capeval judged tier (``capeval.tiers.judged.xstest_rows``)
reads the same pin, so every XSTest reader on this tree shares one revision.

Schema at the pinned revision: ``data/prompts`` has 450 rows with ``id, type,
prompt, annotation_1, annotation_2, agreement, final_label, completion``. Safe
vs unsafe is the ``type`` prefix: the 200 ``contrast_*`` types are the unsafe
prompts, the other 250 are the safe prompts. The 250 + 200 split is a load
gate — a different count means the pin moved, and the loader refuses rather
than silently re-pinning.

Runner contract mirrors the other realign suites: free-form generation, one
record per prompt, NEVER scored here. Scoring is judge-side with the official
XSTest taxonomy (rubric ``xstest.v1``: full_compliance / partial_refusal /
full_refusal, :data:`XSTEST_LABELS`); the H3 collateral criterion is the
full-compliance rate on the safe prompts.
"""

from __future__ import annotations

import random

from ..generate import generate_with_object
from ..models import build_chat_prompt
from .realign_suites import (
    append_records,
    arm_name,
    compose_system,
    item_sha,
    prompt_sha,
    read_done_ids,
)

#: the pins.yaml entry the items are read from (``sisteer.capeval.corpora``).
XSTEST_PIN = "xstest_v2"

#: the pinned split; a loaded frame with any other count is refused.
N_SAFE = 250
N_UNSAFE = 200

#: the ``type`` prefix that marks the unsafe contrast prompts.
CONTRAST_PREFIX = "contrast_"

SUBSETS = ("all", "safe", "unsafe")

#: the arm-independent item fields hashed into ``item_sha`` (what pairing
#: keys on); analysis recomputes the same hash over exactly these fields.
ITEM_SHA_FIELDS: tuple[str, ...] = ("item_id", "prompt", "xstest_type")

#: the judge-side label vocabulary of rubric ``xstest.v1`` (never applied here).
XSTEST_LABELS: tuple[str, ...] = ("full_compliance", "partial_refusal", "full_refusal")


def xstest_pin(pins: dict | None = None) -> dict:
    """The validated ``xstest_v2`` pin entry (``repo_id``, ``revision``, …)."""
    from ..capeval.corpora import load_pins, pin_entry

    return pin_entry(pins if pins is not None else load_pins(), XSTEST_PIN)


def xstest_revision(pins: dict | None = None) -> str:
    """The pinned dataset revision the items are drawn from (``suite_revision``)."""
    return str(xstest_pin(pins)["revision"])


def _xstest_frame(entry: dict):
    """The frame of one validated pin entry: a local-cache-only read, never the network."""
    from ..capeval.corpora import _fetch_pinned_frames

    return _fetch_pinned_frames(entry, XSTEST_PIN)


def items_from_frame(df, *, revision: str | None = None) -> list[dict]:
    """Every XSTest item of a loaded frame, in frame order (no subset, no
    shuffle, no cap).

    Each item: ``{item_id, prompt, xstest_type, safe, suite_revision}`` where
    ``safe`` is ``True`` for the 250 safe prompts and ``False`` for the 200
    ``contrast_*`` unsafe prompts. The 250 + 200 gate runs over the FULL
    frame before any selection, and ids must be unique.
    """
    for col in ("id", "prompt", "type"):
        if col not in df.columns:
            raise ValueError(
                f"xstest: column {col!r} not in {list(df.columns)} — refusing "
                "to guess a schema"
            )
    items: list[dict] = []
    for row in df.to_dict("records"):
        xstest_type = str(row["type"])
        items.append(
            {
                "item_id": str(row["id"]),
                "prompt": str(row["prompt"]),
                "xstest_type": xstest_type,
                "safe": not xstest_type.startswith(CONTRAST_PREFIX),
                "suite_revision": revision,
            }
        )
    n_safe = sum(1 for it in items if it["safe"])
    n_unsafe = len(items) - n_safe
    if (n_safe, n_unsafe) != (N_SAFE, N_UNSAFE):
        raise ValueError(
            f"xstest: {n_safe} safe + {n_unsafe} unsafe != the pinned "
            f"{N_SAFE} + {N_UNSAFE} — the pin moved; never silently re-pin"
        )
    if len({it["item_id"] for it in items}) != len(items):
        raise ValueError("xstest: duplicate item ids in the pinned frame")
    return items


def load_xstest(
    subset: str = "all",
    n: int | None = None,
    seed: int = 0,
    *,
    pins: dict | None = None,
    frame=None,
    revision: str | None = None,
) -> list[dict]:
    """Item dicts from XSTest at the pinned revision.

    ``subset`` ∈ {all, safe, unsafe}; items are shuffled deterministically by
    ``seed`` and capped by ``n`` (after the subset). By default the frame is
    read through the ``xstest_v2`` pin (``pins`` overrides the pins table;
    the revision is the pin's). ``frame`` injects an already-loaded frame —
    for tests, with its ``revision`` given explicitly — and skips the read.
    A missing cache or pin raises; never a silent unpinned fallback.
    """
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; use one of {SUBSETS}")
    if frame is None:
        # ONE validated pin entry serves both the read and the stamp, so the items and the
        # ``suite_revision`` they carry cannot come from two reads of pins.yaml
        entry = xstest_pin(pins)
        frame = _xstest_frame(entry)
        revision = str(entry["revision"])
    items = items_from_frame(frame, revision=revision)
    if subset == "safe":
        items = [it for it in items if it["safe"]]
    elif subset == "unsafe":
        items = [it for it in items if not it["safe"]]
    random.Random(f"realign-xstest:{subset}:{seed}").shuffle(items)
    if n is not None:
        items = items[:n]
    return items


def run_xstest_arm(
    model,
    tok,
    items: list[dict],
    *,
    system: str | None = None,
    obj=None,
    scale: float = 1.0,
    out_jsonl,
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> list[dict]:
    """Generate + record the arm's response per XSTest prompt. Never scores.

    The arm's safety SI is the system prompt (XSTest items have no scenario
    system of their own). Item-level resume on ``item_id``.
    """
    arm = arm_name(system, obj)
    done = read_done_ids(out_jsonl, "item_id")
    pending = [it for it in items if it["item_id"] not in done]

    written: list[dict] = []
    for i in range(0, len(pending), batch_size):
        chunk = pending[i : i + batch_size]
        rendered = [
            build_chat_prompt(tok, it["prompt"], system=compose_system(system, None))
            for it in chunk
        ]
        responses = generate_with_object(
            model, tok, rendered, obj=obj, scale=scale,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
        )
        records = []
        for it, prompt, response in zip(chunk, rendered, responses):
            records.append(
                {
                    "item_id": it["item_id"],
                    "prompt": it["prompt"],
                    "xstest_type": it["xstest_type"],
                    "safe": it["safe"],
                    "response": response,
                    "arm": arm,
                    "suite_revision": it.get("suite_revision"),
                    "item_sha": item_sha({f: it[f] for f in ITEM_SHA_FIELDS}),
                    "prompt_sha": prompt_sha(prompt),
                }
            )
        append_records(out_jsonl, records)
        written.extend(records)
    return written
