"""The SMOKE goal-shard knob as the consumers DOWNSTREAM of the planner read it: the job-side closer and the two production graders.

The planner (reproduce/lib/stage.py, `smoke_goals` / `eval_nshards` / `shard_goal_ids`) reads `OVERRIDE_SMOKE_GOALS=<lo>-<hi>/N` from the
model's overrides document and narrows ONLY the multi-turn targets and the per-view padding levels to shards lo..hi of N, freezing
`coverage.expected = goals[s::N]` over those shards in the inventory. Every registry row the smoke produces carries the SAME document
through the central run-overrides stamp (`sisteer.registry.emit.RecordWriter`: `extra.run_overrides.{id, sha256, keys}` on every record,
read from the job env's `$SISTEER_RUN_OVERRIDES` at registration), so the knob's value rides the run row as `run.extra_json
.run_overrides.keys.OVERRIDE_SMOKE_GOALS`. That stamp is the ONE record a consumer reads the subset from: the closer (inside the job, from
the same env through the same reader the registration used), the multi-turn grader and the padded-static grader (from the run row). The
same grammar and the same dealing rule (`goals[s::N]`) as the planner's, so what a consumer expects is what the inventory froze, by
construction; no store certifies itself — the registration's producer key (`sh<shards>of<N>` for a multi-turn target,
`p<pad>-sh<lo>_<hi>_<N>` for a padding level) must spell the same shards, and the shards on disk must be exactly lo..hi.

No stamp, or a stamp without the key: every consumer behaves exactly as before (the paper's rows). A stamp that names the key under an
OVERRIDE_ID without `smoke`, or a malformed spec, refuses by name — a goal subset is a smoke's, never a run of record's. The static,
controls, BoN, agentic, capability and Petri families never take the knob, stamp or no stamp (`NARROWED_FAMILIES`): their rows under the
smoke document are stamped like every other row and run whole.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

#: the key in the overrides document, and in the stamp's `keys`
KEY = "OVERRIDE_SMOKE_GOALS"
ID_KEY = "OVERRIDE_ID"
#: the document's OVERRIDE_ID must say it is a smoke (case-insensitive substring; the planner's rule, stage.SMOKE_OVERRIDE_ID_MARK)
ID_MARK = "smoke"
#: N <= 99: the padding store names its shards shard00..shard99 (target_static_padded_obj.sbatch; the planner's stage.SMOKE_MAX_NSHARDS)
MAX_NSHARDS = 99
SPEC_RE = re.compile(r"^(?P<lo>[0-9]+)-(?P<hi>[0-9]+)/(?P<n>[0-9]+)$")
#: the families the planner narrows under the knob (stage.eval_nshards); every other family runs whole under the same document
NARROWED_FAMILIES = ("multiturn", "padding")
#: the producer key's shard segment: a multi-turn target's `…-sh<shards>of<N>` (common.sh f22_registry_producer_key; target_attacks.sbatch
#: `sh${SHARDS//,/_}of${NSHARDS}`), a padding level's `…-p<pad>-sh<lo>_<hi>_<N>` (target_static_padded_obj.sbatch `p${PAD}-sh${SHARDS_SPEC//[-\/]/_}`;
#: the unsharded single-job shape spells `p<pad>-sh<N>`)
MULTITURN_KEY_RE = re.compile(r"-sh(?P<shards>[0-9]+(?:_[0-9]+)*)of(?P<n>[0-9]+)$")
PADDING_KEY_RE = re.compile(r"-p(?P<pad>[0-9]+)-sh(?P<lo>[0-9]+)_(?P<hi>[0-9]+)_(?P<n>[0-9]+)$")
PADDING_WHOLE_KEY_RE = re.compile(r"-p(?P<pad>[0-9]+)-sh(?P<n>[0-9]+)$")
STAMP_SOURCE = "registry run_overrides stamp (extra.run_overrides.keys.OVERRIDE_SMOKE_GOALS)"


class SmokeGoalsRefused(ValueError):
    """A stamp that names the knob but cannot be honoured, by name: the id, the spec, a key or a layout that disagrees."""


@dataclass(frozen=True)
class SmokeGoals:
    """The knob as read: shards lo..hi (inclusive, 0-based) of N, and the document it came from."""

    spec: str
    lo: int
    hi: int
    nshards: int
    override_id: str
    overrides_sha256: Optional[str] = None
    overrides_path: Optional[str] = None

    @property
    def indices(self) -> range:
        return range(self.lo, self.hi + 1)

    @property
    def shards(self) -> List[str]:
        """The shard directory names the store holds under the knob: shard<lo>..shard<hi>."""
        return [f"shard{s:02d}" for s in self.indices]

    def deal(self, goal_ids: Sequence[str]) -> Dict[str, List[str]]:
        """The drivers' slicing, byte for byte (jailbreak_run.py generate; the planner's stage.shard_goal_ids): shard s holds goals[s::N]."""
        ids = list(goal_ids)
        return {f"shard{s:02d}": ids[s::self.nshards] for s in self.indices}

    def goal_ids(self, goal_ids: Sequence[str]) -> List[str]:
        """The subset of the eval set this knob keeps, shard by shard in shard order (the order a per-pair goalset file would list)."""
        dealt = self.deal(goal_ids)
        return [g for sh in self.shards for g in dealt[sh]]

    def n_goals(self, total: int) -> int:
        """How many of `total` goals land in shards lo..hi of N (the subset's size, with no ids in hand)."""
        return sum(len(range(s, int(total), self.nshards)) for s in self.indices)

    def record(self) -> dict:
        """The provenance a consumer writes beside what it narrowed: the spec, the range, N, the shard names, the document's identity."""
        return {"source": STAMP_SOURCE, "key": KEY, "spec": self.spec, "lo": self.lo, "hi": self.hi, "nshards": self.nshards,
                "shards": self.shards, "override_id": self.override_id, "overrides_sha256": self.overrides_sha256,
                "overrides_path": self.overrides_path}


def parse_spec(raw: str, where: str = KEY) -> Tuple[int, int, int]:
    """(lo, hi, N) of `<lo>-<hi>/N` — the planner's grammar (stage.parse_smoke_goals), held here too: a malformed spec, N < 1,
    N > MAX_NSHARDS, lo > hi or hi >= N refuses by name. The bare `N` form (every shard) is not a smoke and is refused too."""
    spec = str(raw or "").strip()
    m = SPEC_RE.match(spec)
    if not m:
        raise SmokeGoalsRefused(f"{where}={spec!r} is not `<lo>-<hi>/N` (shards lo..hi, inclusive and 0-based, of N; e.g. 0-0/99) — the smoke "
                                f"goal-shard knob is a shard RANGE of a shard total, nothing else")
    lo, hi, n = int(m.group("lo")), int(m.group("hi")), int(m.group("n"))
    if n < 1:
        raise SmokeGoalsRefused(f"{where}={spec!r}: N must be >= 1")
    if n > MAX_NSHARDS:
        raise SmokeGoalsRefused(f"{where}={spec!r}: N={n} > {MAX_NSHARDS} — the padding store names shards shard00..shard{MAX_NSHARDS} "
                                f"(target_static_padded_obj.sbatch), so one value shared by multi-turn and padding is at most /{MAX_NSHARDS}")
    if lo > hi:
        raise SmokeGoalsRefused(f"{where}={spec!r}: lo {lo} > hi {hi} (inverted range)")
    if hi >= n:
        raise SmokeGoalsRefused(f"{where}={spec!r}: hi {hi} is out of range for N={n} (the shards are 0..{n - 1})")
    return lo, hi, n


def from_stamp(stamp: Optional[Mapping[str, object]], *, where: str = "run_overrides") -> Optional[SmokeGoals]:
    """The knob a run-overrides STAMP sets (`{"id", "sha256", "keys": {...}, "path"}` as emit.run_overrides_stamp builds it and the run
    row's extra_json carries it), or None: no stamp, or a stamp whose keys do not name OVERRIDE_SMOKE_GOALS (the paper's rows). Refuses
    by name when the key is set under an id that does not say `smoke`, or when the spec is malformed."""
    if not stamp or not isinstance(stamp, Mapping):
        return None
    keys = stamp.get("keys")
    if not isinstance(keys, Mapping):
        return None
    raw = str(keys.get(KEY) or "").strip()
    if not raw:
        return None
    ident = str(stamp.get("id") or keys.get(ID_KEY) or "").strip()
    if ID_MARK not in ident.lower():
        raise SmokeGoalsRefused(f"{where}: {KEY}={raw!r} narrows the goal pool but the overrides id {ident or '(unset)'!r} names no smoke (the id must "
                                f"contain `{ID_MARK}`) — a goal subset is a smoke's, never a run of record's; refusing")
    lo, hi, n = parse_spec(raw, f"{where}: {KEY}")
    sha = stamp.get("sha256")
    path = stamp.get("path")
    return SmokeGoals(spec=f"{lo}-{hi}/{n}", lo=lo, hi=hi, nshards=n, override_id=ident,
                      overrides_sha256=str(sha) if sha else None, overrides_path=str(path) if path else None)


def from_env(path: Optional[str] = None) -> Optional[SmokeGoals]:
    """The knob the job's own overrides document sets (`$SISTEER_RUN_OVERRIDES`, or `path`), read through the SAME reader the registration
    stamped the run with (sisteer.registry.emit.run_overrides_stamp) — what the job-side closer reads, inside the job, where the registry
    database is not at hand but the document is (the job side reads it from the wave checkout). None when the variable is unset."""
    from sisteer.registry import emit

    try:
        stamp = emit.run_overrides_stamp(path)
    except OSError as exc:
        raise SmokeGoalsRefused(f"the run-overrides document cannot be read ({exc}) — its smoke goal-shard knob is unknowable; refusing") from None
    return from_stamp(stamp, where=f"overrides document {str((stamp or {}).get('path') or path or '')}".rstrip())


def from_run_row(conn, run_id: str) -> Optional[SmokeGoals]:
    """The knob stamped on ONE run row (`run.extra_json.run_overrides`, read-only), or None when the row carries no stamp or no key. A run
    the registry does not hold is None too (the caller's worklist already located it; nothing is inferred here)."""
    hit = conn.execute("SELECT extra_json FROM run WHERE run_id = ?", (str(run_id),)).fetchone()
    if hit is None or not hit[0]:
        return None
    try:
        extra = json.loads(hit[0]) or {}
    except (TypeError, ValueError):
        return None
    return from_stamp(extra.get("run_overrides") if isinstance(extra, Mapping) else None, where=f"run {str(run_id)[:40]}… run_overrides")


def key_shards(producer_key: Optional[str], family: str) -> Optional[Tuple[List[int], int]]:
    """(shard indices, N) the producer key SPELLS for `family` (`multiturn`: `…-sh<shards>of<N>`; `padding`: `…-p<pad>-sh<lo>_<hi>_<N>`, or the
    unsharded `…-p<pad>-sh<N>` = every shard of N); None when the key carries no shard segment this reader knows."""
    key = str(producer_key or "")
    if family == "multiturn":
        m = MULTITURN_KEY_RE.search(key)
        if not m:
            return None
        return sorted(int(x) for x in m.group("shards").split("_")), int(m.group("n"))
    if family == "padding":
        m = PADDING_KEY_RE.search(key)
        if m:
            lo, hi, n = int(m.group("lo")), int(m.group("hi")), int(m.group("n"))
            return list(range(lo, hi + 1)), n
        m = PADDING_WHOLE_KEY_RE.search(key)
        if m:
            n = int(m.group("n"))
            return list(range(n)), n
        return None
    return None


def key_disagreement(producer_key: Optional[str], smoke: SmokeGoals, family: str) -> Optional[str]:
    """The refusal, or None, when the registration's producer key does not spell the stamped knob's shards and N — the job ran other
    positionals than the document the stamp names (a hand-edited submission, a document changed after the plan)."""
    parsed = key_shards(producer_key, family)
    if parsed is None:
        return (f"the registration's producer key {str(producer_key or '')!r} carries no shard segment for a {family} run, but its run-overrides stamp "
                f"sets {KEY}={smoke.spec} — a smoke run's key spells its shards (sh<shards>of<N> / p<pad>-sh<lo>_<hi>_<N>); refusing")
    shards, n = parsed
    if n != smoke.nshards or shards != list(smoke.indices):
        return (f"the registration's producer key {str(producer_key or '')!r} spells shards {shards} of {n} but its run-overrides stamp sets "
                f"{KEY}={smoke.spec} (shards {list(smoke.indices)} of {smoke.nshards}) — the job did not run the document's plan; refusing")
    return None


def nshards_disagreement(nshards: Optional[int], smoke: SmokeGoals) -> Optional[str]:
    """The refusal, or None, when a caller's shard total (the sbatch's NSHARDS the closer is handed) is not the stamped knob's N."""
    if nshards is None:
        return None
    if int(nshards) != smoke.nshards:
        return (f"the job's shard plan says N={int(nshards)} but its run-overrides stamp sets {KEY}={smoke.spec} (N={smoke.nshards}) — one store, "
                f"one layout; refusing")
    return None


def layout_disagreement(present: Sequence[str], smoke: SmokeGoals) -> Optional[str]:
    """The refusal, or None, when the shard directories on disk are not EXACTLY shard<lo>..shard<hi>: a shard outside the range is a store
    laid out under another plan (a different measurement — the producer key bakes the shards), a missing one is an incomplete smoke."""
    have = sorted(str(s) for s in present)
    want = smoke.shards
    if have != want:
        missing = [s for s in want if s not in have]
        extra = [s for s in have if s not in want]
        return (f"the store's shards {have} are not the stamped smoke's {want} ({KEY}={smoke.spec}): "
                f"{len(missing)} missing {missing[:4]}, {len(extra)} outside the range {extra[:4]} — refusing")
    return None
