#!/usr/bin/env python3
"""Verify the pinned shipped generation_config table against the REAL snapshots.

``configs/sdfbase_shipped_generation_config.json`` pins, per model the sampled-decode
data file states a preset for, what the model's snapshot ships as
``generation_config.json`` (tri-state: present + its sampler values, ships_none,
unknown). ``tests/test_preset_vs_shipped_generation_config.py`` asserts the yaml
presets against that pin, so the check runs where no model files exist (the Modal
CPU suite, the grading host's partial cache). THIS script is the other instrument: it
asserts the pin against a real source, so the pin cannot rot silently.

Two sources, both tri-state, never collapsing "unavailable" into "ships none":

* ``--cache-root`` (default: ``$HF_HOME`` / ``~/.cache/huggingface``): the HF cache's
  ``snapshots/<revision>/generation_config.json`` is PRESENT; its
  ``.no_exist/<revision>/generation_config.json`` marker (huggingface_hub's record of
  the hub's 404 for that file at that revision) is SHIPS_NONE; neither -- no snapshot,
  or a partial snapshot that never asked for the file -- is UNKNOWN. Cluster A's shared
  model cache holds every campaign snapshot; the grading host holds partial ones, so
  most rows read UNKNOWN there, and the script says so by name.
* ``--hub``: the hub's file listing at the revision
  (``/api/models/<id>/revision/<rev>``) then the file itself (``/resolve/``). Listed
  and 200 is PRESENT; not listed, or 404, is SHIPS_NONE; 401/403 (a gated repo without
  a token), a network failure or any other status is UNKNOWN. A token is read from
  ``$HF_TOKEN`` when set and never printed; every campaign model is public today.

Verdict per row: VERIFIED (the source agrees with the pin on state, on the exact
sampler-key set and values, and on the file's sha256 / etag where the source exposes
them), CONTRADICTION (the pin is wrong or stale: fix the pin, then the yaml's
declarations), UNVERIFIED (the source could not tell -- run where the snapshots are).
Exit 0 = every row VERIFIED; 1 = any CONTRADICTION; 2 = no contradiction but some row
UNVERIFIED. Standard library only, so it runs on a bare cluster python3::

    python3 scripts/sdfbase/full22/sampled/verify_shipped_generation_config.py \
        --cache-root "$HF_HOME"          # cluster A, the cohort cache
    python3 scripts/sdfbase/full22/sampled/verify_shipped_generation_config.py --hub
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[4]
PIN_PATH = REPO / "configs" / "sdfbase_shipped_generation_config.json"
FILE = "generation_config.json"
STATES = ("present", "ships_none", "unknown")
VERIFIED, CONTRADICTION, UNVERIFIED = "VERIFIED", "CONTRADICTION", "UNVERIFIED"
EXIT = {VERIFIED: 0, CONTRADICTION: 1, UNVERIFIED: 2}
HUB = "https://huggingface.co"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class PinMalformed(RuntimeError):
    """The pin table does not satisfy its own schema."""


@dataclass
class Observation:
    """What ONE source says about ONE (model, revision)."""
    state: str                      # present | ships_none | unknown
    where: str                      # what was read, for the message
    sampler_keys: Optional[Dict[str, Any]] = None
    do_sample: Any = None           # True / False / None = the file states none
    sha256: Optional[str] = None
    etag: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    problems: List[str] = field(default_factory=list)   # a present file that is not readable JSON, etc.


# ------------------------------------------------------------------ the pin
def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def load_pin(path: Path = PIN_PATH) -> Dict[str, Any]:
    """The pin table read from ``path`` and schema-checked (:func:`validate_pin`)."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PinMalformed(f"{path}: unreadable ({e})") from e
    return validate_pin(d, Path(path).name)


def validate_pin(d: Any, origin: str = "pin") -> Dict[str, Any]:
    """The pin table's schema. Refuses rather than reading a malformed row as
    anything: a state outside the tri-state, a sampler key outside the considered
    set, a non-numeric value, a revision that is not 40 hex, a present row without
    its sha256 or provenance, a ships_none row without evidence (or with file data)."""
    if not isinstance(d, dict) or not isinstance(d.get("models"), dict) or not d["models"]:
        raise PinMalformed(f"{origin}: no non-empty `models` mapping")
    considered = d.get("sampler_keys_considered")
    if (not isinstance(considered, list) or not considered
            or any(not isinstance(k, str) for k in considered) or len(set(considered)) != len(considered)):
        raise PinMalformed(f"{origin}: `sampler_keys_considered` must be a non-empty list of distinct key names")
    for model, e in d["models"].items():
        what = f"{origin} models[{model!r}]"
        if not isinstance(e, dict):
            raise PinMalformed(f"{what} is not a mapping")
        if not isinstance(e.get("revision"), str) or not _HEX40.match(e["revision"]):
            raise PinMalformed(f"{what}: `revision` must be the 40-hex commit sha, got {e.get('revision')!r}")
        if e.get("state") not in STATES:
            raise PinMalformed(f"{what}: `state` must be one of {STATES}, got {e.get('state')!r}")
        if e["state"] == "present":
            sk = e.get("sampler_keys")
            if not isinstance(sk, dict):
                raise PinMalformed(f"{what}: a present row needs a `sampler_keys` mapping (possibly empty)")
            bad = [k for k in sk if k not in considered]
            if bad:
                raise PinMalformed(f"{what}: sampler_keys {bad} are outside sampler_keys_considered {considered}")
            nonnum = [k for k, v in sk.items() if not _is_number(v)]
            if nonnum:
                raise PinMalformed(f"{what}: sampler_keys {nonnum} are not numbers")
            if not isinstance(e.get("sha256"), str) or not _HEX64.match(e["sha256"]):
                raise PinMalformed(f"{what}: a present row needs the file's `sha256` (64 hex)")
            if e.get("etag") is not None and not isinstance(e["etag"], str):
                raise PinMalformed(f"{what}: `etag` must be a string when given")
            rf = e.get("read_from")
            if not isinstance(rf, list) or not rf or any(not isinstance(s, str) or not s for s in rf):
                raise PinMalformed(f"{what}: a present row needs a non-empty `read_from` list (where the file was read)")
        elif e["state"] == "ships_none":
            ev = e.get("evidence")
            if not isinstance(ev, list) or not ev or any(not isinstance(s, str) or not s for s in ev):
                raise PinMalformed(f"{what}: a ships_none row needs a non-empty `evidence` list (what said the file does not exist)")
            for k in ("sampler_keys", "sha256", "etag"):
                if e.get(k) is not None:
                    raise PinMalformed(f"{what}: a ships_none row carries `{k}`; there is no file to describe")
    return d


# ------------------------------------------------------------------ reading a file
def sampler_values(raw: Dict[str, Any], considered: Sequence[str]) -> Tuple[Dict[str, Any], List[str]]:
    """Exactly the considered keys the file states, plus the problems (a non-numeric value)."""
    out: Dict[str, Any] = {}
    problems: List[str] = []
    for k in considered:
        if k in raw:
            if _is_number(raw[k]):
                out[k] = raw[k]
            else:
                problems.append(f"{k} is not a number in the file ({raw[k]!r})")
    return out, problems


def observe_bytes(data: bytes, where: str, considered: Sequence[str], etag: Optional[str] = None) -> Observation:
    sha = hashlib.sha256(data).hexdigest()
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        return Observation("present", where, sha256=sha, etag=etag,
                           problems=[f"the file is not JSON ({e})"])
    if not isinstance(raw, dict):
        return Observation("present", where, sha256=sha, etag=etag, problems=["the file is not a JSON object"])
    sk, problems = sampler_values(raw, considered)
    return Observation("present", where, sampler_keys=sk, do_sample=raw.get("do_sample"),
                       sha256=sha, etag=etag, raw=raw, problems=problems)


# ------------------------------------------------------------------ source: an HF cache
def hub_dir_of(cache_root: Path) -> Path:
    """``<root>/hub`` when the root is an HF_HOME, else the root itself (already a hub dir)."""
    cache_root = Path(cache_root).expanduser()
    return cache_root / "hub" if (cache_root / "hub").is_dir() else cache_root


def default_cache_root() -> Path:
    return Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else Path.home() / ".cache" / "huggingface"


def observe_cache(model: str, revision: str, hub_dir: Path, considered: Sequence[str]) -> Observation:
    """Tri-state read of one HF cache: the snapshot file, the no-exist marker, or unknown."""
    repo = Path(hub_dir) / f"models--{model.replace('/', '--')}"
    snap = repo / "snapshots" / revision / FILE
    marker = repo / ".no_exist" / revision / FILE
    if snap.is_file():
        real = snap.resolve()
        etag = real.name if real.parent.name == "blobs" else None
        return observe_bytes(snap.read_bytes(), str(snap), considered, etag=etag)
    if marker.exists():
        return Observation("ships_none", f"{marker} (the hub's 404 for the file at this revision, recorded by huggingface_hub)")
    if (repo / "snapshots" / revision).is_dir():
        n = len(list((repo / "snapshots" / revision).iterdir()))
        return Observation("unknown", f"{repo / 'snapshots' / revision} is a PARTIAL snapshot ({n} files) with no {FILE} and no "
                                      f".no_exist marker for it: the file was never asked for here -- absent-because-unavailable, "
                                      f"NOT ships_none")
    if repo.is_dir():
        return Observation("unknown", f"{repo} holds no snapshot of revision {revision[:8]}: nothing to read here")
    return Observation("unknown", f"{repo} does not exist in this cache: nothing to read here")


# ------------------------------------------------------------------ source: the hub
def _hub_get(url: str, token: Optional[str], timeout: float) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "sisteer-verify-shipped-generation-config"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 -- fixed https host
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def observe_hub(model: str, revision: str, considered: Sequence[str], token: Optional[str] = None,
                timeout: float = 30.0) -> Observation:
    """Tri-state read of the hub: the revision's file listing, then the file. A token
    (``$HF_TOKEN``) is sent when given and never echoed; an auth failure or a network
    failure is UNKNOWN, never ships_none."""
    api = f"{HUB}/api/models/{model}/revision/{revision}"
    try:
        code, body = _hub_get(api, token, timeout)
    except (urllib.error.URLError, OSError, ValueError) as e:
        return Observation("unknown", f"{api}: network failure ({e}); the hub could not be asked")
    if code in (401, 403):
        return Observation("unknown", f"{api} -> {code}: not readable {'with' if token else 'without'} a token "
                                      f"(a gated repo): the hub would not say")
    if code != 200:
        return Observation("unknown", f"{api} -> {code}: the hub did not answer the listing")
    try:
        listing = json.loads(body.decode("utf-8"))
        names = [s["rfilename"] for s in listing.get("siblings", [])]
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as e:
        return Observation("unknown", f"{api}: unparseable listing ({e})")
    if FILE not in names:
        return Observation("ships_none", f"{api} lists {len(names)} files at {revision[:8]} and {FILE} is not among them")
    url = f"{HUB}/{model}/resolve/{revision}/{FILE}"
    try:
        code, body = _hub_get(url, token, timeout)
    except (urllib.error.URLError, OSError, ValueError) as e:
        return Observation("unknown", f"{url}: network failure ({e}) after the listing named the file")
    if code == 200:
        return observe_bytes(body, f"{url} -> 200 (listed at {revision[:8]})", considered)
    if code == 404:
        return Observation("ships_none", f"{url} -> 404")
    if code in (401, 403):
        return Observation("unknown", f"{url} -> {code}: listed but not readable {'with' if token else 'without'} a token")
    return Observation("unknown", f"{url} -> {code}: the hub did not serve the file")


# ------------------------------------------------------------------ the comparison
def _num_eq(a: Any, b: Any) -> bool:
    return _is_number(a) and _is_number(b) and float(a) == float(b)


def compare(model: str, entry: Dict[str, Any], obs: Observation, considered: Sequence[str]) -> Tuple[str, str]:
    """(verdict, message) for one pinned row against one observation."""
    rev = entry["revision"]
    who = f"{model} @ {rev[:8]}"
    if obs.state == "unknown":
        return UNVERIFIED, f"{who}: pin says {entry['state']}; this source cannot tell -- {obs.where}"
    if obs.problems:
        return CONTRADICTION, f"{who}: the source's file is not readable as a generation config ({'; '.join(obs.problems)}) -- {obs.where}"
    if obs.state != entry["state"]:
        detail = ""
        if obs.state == "present":
            detail = f" (the file states sampler keys {obs.sampler_keys}, sha256 {obs.sha256})"
        return CONTRADICTION, (f"{who}: the pin says {entry['state']} but the source says {obs.state}{detail} -- {obs.where}. "
                               f"Re-read the snapshot and re-pin; then re-check the yaml's declarations for this model")
    if obs.state == "ships_none":
        return VERIFIED, f"{who}: ships no {FILE} at this revision, as pinned -- {obs.where}"
    pinned = dict(entry["sampler_keys"])
    seen = dict(obs.sampler_keys or {})
    diff = sorted(k for k in set(pinned) | set(seen)
                  if (k in pinned) != (k in seen) or not _num_eq(pinned.get(k), seen.get(k)))
    if diff:
        return CONTRADICTION, (f"{who}: the pinned shipped sampler values {pinned} differ from the file's {seen} on {diff} "
                               f"-- {obs.where}. The pin is stale or wrong: re-pin from this file, then re-check the yaml's "
                               f"presets and deliberate_departure declarations against it")
    if obs.sha256 != entry["sha256"]:
        return CONTRADICTION, (f"{who}: sampler values agree but the file's sha256 {obs.sha256} is not the pinned "
                               f"{entry['sha256']} -- {obs.where}. The file changed (or the pin was copied from another "
                               f"file); re-pin the sha256 so the provenance describes this file")
    if obs.etag and entry.get("etag") and obs.etag != entry["etag"]:
        return CONTRADICTION, f"{who}: the cache blob's etag {obs.etag} is not the pinned {entry['etag']} -- {obs.where}"
    return VERIFIED, f"{who}: ships {seen} (do_sample {obs.do_sample!r}), sha256 {obs.sha256[:12]}..., as pinned -- {obs.where}"


def verify(pin: Dict[str, Any], *, hub_dir: Optional[Path] = None, hub: bool = False, token: Optional[str] = None,
           models: Optional[Sequence[str]] = None, timeout: float = 30.0) -> List[Tuple[str, str, str]]:
    """[(model, verdict, message)] for every pinned model (or the ``models`` given), from one source."""
    considered = pin["sampler_keys_considered"]
    out: List[Tuple[str, str, str]] = []
    for model, entry in pin["models"].items():
        if models and model not in models:
            continue
        if hub:
            obs = observe_hub(model, entry["revision"], considered, token=token, timeout=timeout)
        else:
            assert hub_dir is not None
            obs = observe_cache(model, entry["revision"], hub_dir, considered)
        verdict, msg = compare(model, entry, obs, considered)
        out.append((model, verdict, msg))
    return out


def overall(results: Sequence[Tuple[str, str, str]]) -> str:
    verdicts = {v for _, v, _ in results}
    if CONTRADICTION in verdicts:
        return CONTRADICTION
    if UNVERIFIED in verdicts or not results:
        return UNVERIFIED
    return VERIFIED


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pin", type=Path, default=PIN_PATH, help=f"the pin table (default {PIN_PATH.relative_to(REPO)})")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--cache-root", type=Path, default=None,
                     help="an HF_HOME (or its hub dir) to read snapshots from; default $HF_HOME or ~/.cache/huggingface")
    src.add_argument("--hub", action="store_true", help="ask the hub instead of a cache ($HF_TOKEN is sent when set)")
    ap.add_argument("--model", action="append", default=None, help="check only this pinned model id (repeatable)")
    ap.add_argument("--timeout", type=float, default=30.0)
    a = ap.parse_args(argv)
    try:
        pin = load_pin(a.pin)
    except PinMalformed as e:
        print(f"PIN MALFORMED: {e}", file=sys.stderr)
        return EXIT[CONTRADICTION]
    if a.model:
        unknown = [m for m in a.model if m not in pin["models"]]
        if unknown:
            print(f"not pinned: {unknown}; pinned models are {sorted(pin['models'])}", file=sys.stderr)
            return EXIT[CONTRADICTION]
    if a.hub:
        source = "the hub"
        results = verify(pin, hub=True, token=os.environ.get("HF_TOKEN") or None, models=a.model, timeout=a.timeout)
    else:
        hub_dir = hub_dir_of(a.cache_root if a.cache_root is not None else default_cache_root())
        source = f"HF cache {hub_dir}"
        results = verify(pin, hub_dir=hub_dir, models=a.model)
    for _, verdict, msg in results:
        print(f"{verdict}: {msg}")
    final = overall(results)
    counts = {v: sum(1 for _, x, _ in results if x == v) for v in (VERIFIED, CONTRADICTION, UNVERIFIED)}
    print(f"-- {final}: {counts[VERIFIED]} verified, {counts[CONTRADICTION]} contradicted, {counts[UNVERIFIED]} unverified "
          f"of {len(results)} pinned rows, against {source}")
    if final == UNVERIFIED:
        print("   an UNVERIFIED row is absent-because-unavailable here, NOT ships_none; run where the snapshots are: "
              "on a cluster, `python3 scripts/sdfbase/full22/sampled/verify_shipped_generation_config.py "
              "--cache-root $HF_HOME` (its Hugging Face cache), or anywhere with network, `--hub`")
    return EXIT[final]


if __name__ == "__main__":
    sys.exit(main())
