"""Roster-declared capability OMISSIONS applied to a served run's readouts — a lane-side overlay, never part of
``sampled_decode.py``'s shared body.

The wave roster (``configs/sdfbase_full22_roster.yaml``) is the ONE place a target's capability omissions live
(``targets.<key>.capability_omitted``, each with a reason — the sdfbase-llama8b target omits gsm8k:
inspect/gsm8k_paired200 is the paper's one GSM8K instrument, 2026-09-10). The sampled configs the generator
writes carry that omission in their ``readouts`` (``make_sampled_configs.expected_readouts`` via
``f22_roster.target_capability_omitted``). A run whose config carries NO ``sampled_decode`` block gets the vendor
default injected by ``sampled_decode.resolve_default_decode`` — main's shared builder, which knows nothing of the
roster — so the drivers apply THIS overlay right after that resolution, and a default block for an omitting target
renders the same omission record the generator writes, never a readout a driver could run.

Why here and not in ``sampled_decode.py``: that module's whole body is under ``sampled_controls.VENDOR_CONTRACT``
(main is canonical, a vendor branch re-mirrors it byte for byte, and any change to it ships with the re-mirrored
vendor). A roster-driven behaviour of one wave does not belong in that shared body; it lives beside it
(2026-09-12).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

REPO = Path(__file__).resolve().parents[3]
#: the wave roster: targets.<key>.capability_omitted.<control> -> the reason it is omitted
ROSTER_PATH = REPO / "configs" / "sdfbase_full22_roster.yaml"


def _load_roster(roster: Any) -> Mapping[str, Any]:
    if roster is None:
        roster = ROSTER_PATH
    if isinstance(roster, (str, Path)):
        path = Path(roster)
        if not path.is_file():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return roster or {}


def gsm8k_omission(key: str, roster: Any = None) -> Optional[str]:
    """The roster's reason for OMITTING the gsm8k readout on target ``key``, or None: no roster file (a checkout
    without the wave roster), no target row for the key (a preset row the roster does not name), or no omission.
    ``roster`` = a loaded roster mapping, a path, or None for :data:`ROSTER_PATH`."""
    target = ((_load_roster(roster)).get("targets") or {}).get(key) or {}
    omitted = target.get("capability_omitted") or {}
    reason = omitted.get("gsm8k") if isinstance(omitted, Mapping) else None
    return str(reason) if reason else None


def readouts_with_omissions(readouts: Mapping[str, Any], key: str, roster: Any = None) -> Dict[str, Any]:
    """``readouts`` (a ``sampled_decode.readouts`` sub-block, the shared builder's rendering) with the roster's
    omissions applied for target ``key``: an omitted gsm8k becomes ``{"omitted": True, "reason": <the roster's>}`` —
    exactly the record ``make_sampled_configs.expected_readouts`` writes — and every other readout passes through
    unchanged. No omission -> an equal copy."""
    out: Dict[str, Any] = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in (readouts or {}).items()}
    reason = gsm8k_omission(key, roster)
    if reason is not None and "gsm8k" in out:
        out["gsm8k"] = {"omitted": True, "reason": reason}
    return out


def apply_to_config(cfg: Mapping[str, Any], key: Optional[str], roster: Any = None) -> Dict[str, Any]:
    """A copy of ``cfg`` whose ``sampled_decode.readouts`` carry the roster's omissions for target ``key`` (the
    drivers call this right after ``sampled_decode.resolve_default_decode``). Without a key, a block or readouts,
    the config comes back as an equal copy."""
    out = dict(cfg)
    block = out.get("sampled_decode")
    if not key or not isinstance(block, Mapping) or not isinstance(block.get("readouts"), Mapping):
        return out
    new_block = dict(block)
    new_block["readouts"] = readouts_with_omissions(block["readouts"], key, roster)
    out["sampled_decode"] = new_block
    return out
