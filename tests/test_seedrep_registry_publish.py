"""Qwen seed-replicate battery: records reach the registry only
from ACCEPTED attempts, only against a COMMITTED admission, under collision-free per-attempt names.

A two-snapshot fake volume: two concurrent shards (each its own attempt root and attempt-unique spool, the SAME run ids and
the SAME local sequence numbers, so their filenames collide by construction) plus a superseded re-execution of shard 0 and
an attempt nobody accepted. The published set must equal the accepted union EXACTLY — every file of both accepted
attempts, nothing from the superseded or unaccepted one — with no filename collision, re-sequenced per attempt; the REAL
inbox merge then lands them beside the grading host's own files without a conflict.

ATTEMPT.json here is the writer's shape (``seedrep_attempt/1``: the six axes in ``control_key``, the run sha prefix then
``wave_uuid/leg/engine/stage/shard``, with ``launch_plan_row`` the standup id STRING); the accepted map's keys are those
control keys (the accepted Dict's keys). Admissions are built by the producer that writes them
(``logical_run_complete.admission_block``), never re-spelled here, except for the deliberately malformed known-bads.

P1-1: publication is BOUND to the admission (run, wave, the complete accepted-attempt set with each attempt's six axes, the
logical sha of every source file): a base-only admission refuses a ``rung`` attempt and a same-run-id attempt of another
wave; a source file changed or added after the verdict, an admitted shard missing from the tree, a pending completion —
each refuses the run whole. P1-2: names are ``<block·1e6 + seq>-<type>`` with ``block`` a pure function of the attempt id,
so a base shard and an 8,192-rung shard of one run publish together without collision, a retry derives the same names
with nothing persisted, and a same-W relaunch (new wave, new attempt ids, the same run ids) never reuses a name.
"""
from __future__ import annotations

import json
import re as _re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts" / "sdfbase" / "seedrep",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import logical_run_complete as LRC  # noqa: E402
import registry_publish as PUB  # noqa: E402
from sisteer.registry import emit, identity, synth  # noqa: E402
from sisteer.registry import harvest as H  # noqa: E402
from sisteer.registry import records as R  # noqa: E402

WSHA = "a" * 40
W7, WAVE, LEG, VIEW = WSHA[:7], "11111111-2222-3333-4444-555555555555", "crescendo", "s2"
WAVE2 = "22222222-3333-4444-5555-666666666666"
CAMPAIGN = "synthbattery"
RUN_A = identity.run_id(CAMPAIGN, f"seedrep_battery:{W7}:{LEG}:{VIEW}", "none", "crescendo", synth.INSTRUMENT_ID)
RUN_B = identity.run_id(CAMPAIGN, f"seedrep_battery:{W7}:{LEG}:{VIEW}", "objA@a1b2c3d4", "crescendo", synth.INSTRUMENT_ID)
RUNS = (RUN_A, RUN_B)
FILE_KINDS = (("run_state", False), ("item_result", True), ("run_patch", False))     # what _fill_spool writes per run, in seq order


def _re_escape(x: str) -> str:
    return _re.escape(x)


def _key(shard: int, view: str = VIEW, leg: str = LEG, wave: str = WAVE, stage: str = "base", w7: str = W7) -> str:
    """The control key — the accepted Dict's key and ATTEMPT.json's ``control_key``."""
    return f"{w7}/{wave}/{leg}/{view}/{stage}/{shard}"


def _attempt(root: Path, shard: int, aid: str, *, superseded_by=None, complete=True, with_shard=True, stage="base", wave=WAVE,
             w=WSHA) -> PUB.AttemptRoot:
    a = root / LEG / VIEW / str(shard) / "attempts" / aid
    a.mkdir(parents=True)
    meta = {"schema": "seedrep_attempt/1", "attempt_id": aid, "call_id": f"fc-{aid}", "task_id": f"ta-{aid}", "W": w, "wave_uuid": wave,
            "launch_plan_row": f"{LEG}__{VIEW}__{stage}__shard{shard:02d}", "prior_attempts_seen": [], "superseded_by": superseded_by}
    if with_shard:
        meta["control_key"] = _key(shard, wave=wave, stage=stage, w7=w[:7])
    else:
        meta.update({"leg": LEG, "view": VIEW, "stage": stage})              # the spelled-out shape, shard omitted
    (a / PUB.ATTEMPT_FILE).write_text(json.dumps(meta))
    if complete:
        (a / PUB.COMPLETE_FILE).write_text("{}")
    return PUB.read_attempt(a)


def _fill_spool(a: PUB.AttemptRoot, run_ids, n_items: int, tag: str):
    """A shard's records for each run: running + a gz item file + a patch — the same local seq numbers on every shard."""
    written = []
    for rid in run_ids:
        w = emit.RecordWriter(str(a.spool), CAMPAIGN, rid, synth.producer(substrate="modal"), disabled=False)
        written.append(w.emit_state("running", "completion_artifact", f"{tag}/manifest.json"))
        written.append(w.emit_items([{"item_id": f"{tag}-g{i:03d}", "metric_id": "asr", "trial_index": 0, "sample_index": 0, "turn_index": -1,
                                      "judgment_id": "", "value": 0.0, "label": None, "api_pool": None, "created_utc": synth.t(i), "judged_utc": None,
                                      "withdrawn_by_revision_id": None} for i in range(n_items)]))
        written.append(w.emit_patch(fields={"node_hostname": synth.TESTHOST}))
    return written


def _admission(rid: str, attempts) -> dict:
    """The producer's own admission block for ``rid`` over ``attempts`` (each attempt's six axes + its run dir's file shas)."""
    rd_name = identity.run_dir_name(rid)
    shards = [LRC.ShardItems(a, set(), "spool", [], PUB.run_dir_files(a.spool / CAMPAIGN / rd_name)) for a in attempts]
    return LRC.admission_block({"run_id": rid, "campaign": CAMPAIGN, "code_sha": WSHA}, LEG, VIEW, shards)


def _admit(root: Path, run_ids, attempts, *, state=PUB.COMPLETION_COMMITTED, name="union", mutate=None) -> Path:
    """What ``logical_run_complete.py`` leaves behind for an ADMITTED run; ``mutate(adm)`` edits the block for a known-bad."""
    d = root / name
    d.mkdir(exist_ok=True)
    for rid in run_ids:
        adm = _admission(rid, attempts)
        if mutate:
            mutate(adm)
        (d / f"{identity.run_dir_name(rid)}.json").write_text(json.dumps({
            "run_id": rid, "verdict": "ADMITTED", "reasons": [], "admission": adm,
            "completion": {"state": state, "admission_utc": synth.t(0), "record_ids": {}}}))
    return d


def _refuse(root: Path, run_ids, name="union") -> Path:
    d = root / name
    d.mkdir(exist_ok=True)
    for rid in run_ids:
        (d / f"{identity.run_dir_name(rid)}.json").write_text(json.dumps({"run_id": rid, "verdict": "REFUSED", "reasons": ["synthetic"]}))
    return d


def _accepted(root: Path, mapping: dict, name="accepted.json") -> Path:
    p = root / name
    p.write_text(json.dumps(mapping))
    return p


def _logical_shas(paths):
    return {R.file_logical_sha(str(p)) for p in paths}


def _names(rep, aid, rid):
    return sorted(Path(p["dest"]).name for p in rep.published if p["attempt_id"] == aid and p["run_dir"] == identity.run_dir_name(rid))


def _expected_names(aid):
    """The names the block rule assigns to ``_fill_spool``'s three files of one run."""
    b = PUB.attempt_block(aid)
    return sorted(R.record_file_name(PUB.block_seq(b, i), t, gz=gz) for i, (t, gz) in enumerate(FILE_KINDS, 1))


@pytest.fixture
def vol(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    return tmp_path


def test_two_shards_plus_a_superseded_and_an_unaccepted_attempt_publish_exactly_the_accepted_union(vol):
    roots = vol / "stores"
    a0 = _attempt(roots, 0, "att-0")
    a1 = _attempt(roots, 1, "att-1")
    late = _attempt(roots, 0, "att-0-late", superseded_by="att-0")     # a re-executed shard 0 that lost the acceptance
    lone = _attempt(roots, 1, "att-1-nobody")                            # complete, unmarked, but nobody accepted it
    f0 = _fill_spool(a0, RUNS, 3, "s0")
    f1 = _fill_spool(a1, RUNS, 4, "s1")
    _fill_spool(late, RUNS, 3, "s0late")
    _fill_spool(lone, RUNS, 2, "nobody")
    # the shards collide on every local filename by construction
    assert sorted(p.name for p in (a0.spool / CAMPAIGN / identity.run_dir_name(RUN_A)).iterdir()) == \
        sorted(p.name for p in (a1.spool / CAMPAIGN / identity.run_dir_name(RUN_A)).iterdir())
    attempts = PUB.discover_attempts([roots])
    assert [a.attempt_id for a in attempts] == ["att-0", "att-0-late", "att-1", "att-1-nobody"]
    assert attempts[0].axes == {"w7": W7, "wave_uuid": WAVE, "container": LEG, "view": VIEW, "stage": "base", "shard": 0}    # read off control_key
    assert attempts[0].leg == attempts[0].container == LEG and attempts[0].meta["launch_plan_row"] == f"{LEG}__{VIEW}__base__shard00"
    accepted = PUB.load_accepted(_accepted(vol, {_key(0): "att-0", _key(1): "att-1"}))
    inbox = vol / "registry" / "inbox"
    union = _admit(vol, RUNS, [a0, a1])
    rep = PUB.publish(attempts, accepted, inbox=inbox, host="seedrep-modal", union_dir=union)
    # published == the accepted union, exactly: every file of att-0 and att-1, by bytes
    assert rep.published_attempt_ids() == {"att-0", "att-1"}
    assert len(rep.published) == len(f0) + len(f1) == 12 and rep.duplicates == []
    assert _logical_shas(p["dest"] for p in rep.published) == _logical_shas(f0 + f1)
    assert {(s["attempt_id"], s["reason"].split(":")[0]) for s in rep.skipped_attempts} == {("att-0-late", "superseded_by"), ("att-1-nobody", "not_accepted")}
    # nothing from the superseded attempt: its bytes are not in the inbox
    inbox_shas = _logical_shas(p for p in inbox.rglob("*.jsonl*") if p.is_file())
    assert not (inbox_shas & _logical_shas(R.iter_record_files(str(late.spool)))) and not (inbox_shas & _logical_shas(R.iter_record_files(str(lone.spool))))
    # no filename collision: ONE block per attempt (a pure function of its id), above any grading-host-local sequence, the gz suffix kept
    dests = [Path(p["dest"]) for p in rep.published]
    assert len(set(dests)) == len(dests)
    assert rep.blocks == {aid: PUB.attempt_block(aid) for aid in ("att-0", "att-1")} and rep.blocks["att-0"] != rep.blocks["att-1"]
    for aid in ("att-0", "att-1"):
        assert _names(rep, aid, RUN_A) == _expected_names(aid) == _names(rep, aid, RUN_B)
    assert all(Path(p["dest"]).parent == inbox / "seedrep-modal" / CAMPAIGN / p["run_dir"] for p in rep.published)
    assert all({p[ax] for p in rep.published if p["attempt_id"] == "att-1"} == {v} for ax, v in (("shard", 1), ("stage", "base"), ("wave_uuid", WAVE)))
    ledger = inbox / "seedrep-modal" / PUB.LEDGER_DIR / "published.jsonl"
    assert sum(1 for _ in open(ledger)) == 12
    # a re-publication (a retry) is all duplicates under the SAME names and writes nothing new — with nothing persisted to get it
    rep2 = PUB.publish(attempts, accepted, inbox=inbox, host="seedrep-modal", union_dir=union)
    assert rep2.published == [] and len(rep2.duplicates) == 12 and rep2.blocks == rep.blocks
    assert sorted(Path(d["dest"]).name for d in rep2.duplicates) == sorted(Path(p["dest"]).name for p in rep.published)
    # the REAL inbox merge: the grading host already holds its own 000001-run.jsonl for RUN_A (queued_modal); nothing conflicts
    reg = vol / "registry"
    H.ensure_layout(str(reg))
    own = emit.RecordWriter(str(reg / "records"), CAMPAIGN, RUN_A, synth.producer(), disabled=False)
    own_path = own.emit_run(synth.run_payload("k", i=1, run_id=RUN_A))
    merged = H.scan_inbox(str(reg))
    assert {s for _d, s in merged} == {"merged"} and len(merged) == 12
    landed = sorted(p.name for p in (reg / "records" / CAMPAIGN / identity.run_dir_name(RUN_A)).iterdir())
    assert landed == sorted(["000001-run.jsonl"] + _expected_names("att-0") + _expected_names("att-1"))
    assert Path(own_path).is_file() and not list((inbox / "seedrep-modal" / CAMPAIGN).rglob("*.jsonl*"))
    # the harvester classifies every published file by its name (RECORD_FILE_RE): the block form keeps the <digits>-<type> grammar
    for fn in landed:
        assert R.RECORD_FILE_RE.match(fn).group("type") in R.RECORD_TYPES
    # every writer allocates its number AT WRITE TIME (registry follow-ups-4 item 13): a grading host writer alive before the merge and a
    # fresh one both list the directory and scan past the published blocks -- neither collides with a published name, and the alive
    # writer no longer re-issues a small number from a census taken before the merge (the cached-census bug the rule fixed)
    nxt = own.emit_patch(fields={"notes": "after publication"})
    fresh = emit.RecordWriter(str(reg / "records"), CAMPAIGN, RUN_A, synth.producer(), disabled=False).emit_patch(fields={"notes": "fresh writer"})
    top = max(int(n.split("-", 1)[0]) for n in landed)
    assert Path(nxt).name == R.record_file_name(top + 1, "run_patch")
    assert Path(fresh).name == R.record_file_name(top + 2, "run_patch")
    assert Path(nxt).name not in landed and Path(fresh).name not in landed


def test_admission_binds_the_attempt_set_a_base_only_admission_refuses_a_rung_and_another_waves_attempt(vol):
    """The accepted map names every attempt below, the admission names ONLY the base shard."""
    roots = vol / "stores"
    base = _attempt(roots, 0, "att-0")
    rung = _attempt(roots, 0, "att-0-rung", stage="8192")                      # an accepted 8,192-rung attempt of the same run
    other = _attempt(roots, 0, "att-0-w2", wave=WAVE2)                          # an accepted same-run-id attempt of ANOTHER wave
    for a, tag in ((base, "b"), (rung, "r"), (other, "w2")):
        _fill_spool(a, RUNS, 2, tag)
    attempts = PUB.discover_attempts([roots])
    acc = PUB.load_accepted(_accepted(vol, {_key(0): "att-0", _key(0, stage="8192"): "att-0-rung", _key(0, wave=WAVE2): "att-0-w2"}))
    assert all(acc.is_accepted(a)[0] for a in attempts)                          # all three ARE accepted; admission is the gate
    union = _admit(vol, RUNS, [base])
    inbox = vol / "inbox"
    rep = PUB.publish(attempts, acc, inbox=inbox, host="h", union_dir=union)
    assert rep.published_attempt_ids() == {"att-0"} and len(rep.published) == 6 and set(rep.blocks) == {"att-0"}
    skipped = {(s["attempt_id"], s["run_dir"]) for s in rep.skipped_run_dirs}
    assert skipped == {(aid, identity.run_dir_name(r)) for aid in ("att-0-rung", "att-0-w2") for r in RUNS}
    for s in rep.skipped_run_dirs:
        assert s["reason"].startswith("attempt_not_admitted:") and "['att-0']" in s["reason"]
        assert ("stage '8192'" in s["reason"]) == (s["attempt_id"] == "att-0-rung") and (WAVE2 in s["reason"]) == (s["attempt_id"] == "att-0-w2")
    inbox_shas = _logical_shas(p for p in inbox.rglob("*.jsonl*") if p.is_file())
    for a in (rung, other):
        assert not (inbox_shas & _logical_shas(R.iter_record_files(str(a.spool))))
    # KNOWN-BAD 1: a source file rewritten after the verdict (bytes changed) refuses the run outright — nothing of it is written
    victim = next(iter(R.iter_record_files(str(base.spool / CAMPAIGN / identity.run_dir_name(RUN_A)))))
    orig = Path(victim).read_bytes()
    Path(victim).write_bytes(orig + (json.dumps({"record_type": "run_patch", "payload": {"run_id": RUN_A}}) + "\n").encode())
    with pytest.raises(PUB.PublishRefused, match="differ from the admitted set.*bytes changed"):
        PUB.publish(attempts, acc, inbox=vol / "inbox2", host="h", union_dir=union)
    assert not (vol / "inbox2").exists()
    Path(victim).write_bytes(orig)
    # KNOWN-BAD 2: a file ADDED to the run dir after the verdict is outside the admission
    extra = Path(victim).with_name("000009-run_patch.jsonl")
    extra.write_bytes((json.dumps({"record_type": "run_patch", "payload": {"run_id": RUN_A}}) + "\n").encode())
    with pytest.raises(PUB.PublishRefused, match=r"outside the admission: \['000009-run_patch.jsonl'\]"):
        PUB.publish(attempts, acc, inbox=vol / "inbox3", host="h", union_dir=union)
    extra.unlink()
    # KNOWN-BAD 3: the admission names a second shard that is not in the pulled tree -> the run publishes whole or not at all
    ghost = PUB.AttemptRoot(vol / "ghost", "att-ghost", None, {"w7": W7, "wave_uuid": WAVE, "container": LEG, "view": VIEW, "stage": "base", "shard": 1})
    (ghost.spool / CAMPAIGN / identity.run_dir_name(RUN_A)).mkdir(parents=True)
    union3 = _admit(vol, (RUN_A,), [base, ghost], name="union3")
    rep3 = PUB.publish(attempts, acc, inbox=vol / "inbox4", host="h", union_dir=union3)
    assert rep3.published == [] and any(s["reason"].startswith("admitted_shard_missing:['att-ghost']") for s in rep3.skipped_run_dirs)
    # KNOWN-BAD 4: an ADMITTED verdict whose completion is still PENDING (a crash before the completion records) publishes nothing
    union4 = _admit(vol, RUNS, [base], state=PUB.COMPLETION_PENDING, name="union4")
    rep4 = PUB.publish(attempts, acc, inbox=vol / "inbox5", host="h", union_dir=union4)
    assert rep4.published == [] and all(s["reason"].startswith("union_verdict:ADMITTED but completion pending") for s in rep4.skipped_run_dirs
                                        if s["attempt_id"] == "att-0")
    # KNOWN-BAD 5: an admission whose entry names another wave than the attempt's ATTEMPT.json is a contradiction, refused loudly
    def other_wave(adm):
        adm["wave_uuid"] = WAVE2
        for e in adm["attempts"].values():
            e["wave_uuid"] = WAVE2
    union5 = _admit(vol, RUNS, [base], name="union5", mutate=other_wave)
    with pytest.raises(PUB.PublishRefused, match="admitted under wave_uuid"):
        PUB.publish(attempts, acc, inbox=vol / "inbox6", host="h", union_dir=union5)
    # KNOWN-BAD 6: a pre-admission verdict file (ADMITTED, no admission block) is not a licence
    union6 = vol / "union6"
    union6.mkdir()
    for rid in RUNS:
        (union6 / f"{identity.run_dir_name(rid)}.json").write_text(json.dumps({"run_id": rid, "verdict": "ADMITTED", "reasons": []}))
    rep6 = PUB.publish(attempts, acc, inbox=vol / "inbox7", host="h", union_dir=union6)
    assert rep6.published == [] and all("no admission block" in s["reason"] for s in rep6.skipped_run_dirs if s["attempt_id"] == "att-0")


def test_names_are_one_block_per_attempt_across_stages_and_relaunches(vol):
    """Base shard 0 + 8,192-rung shard 0 publish together without collision; a same-W relaunch never reuses names."""
    roots = vol / "stores"
    base = _attempt(roots, 0, "att-base")
    rung = _attempt(roots, 0, "att-rung", stage="8192")
    _fill_spool(base, (RUN_A,), 3, "base")
    _fill_spool(rung, (RUN_A,), 3, "rung")           # the SAME local names (000001-…) on the same shard index, another stage
    attempts = PUB.discover_attempts([roots])
    acc = PUB.load_accepted(_accepted(vol, {_key(0): "att-base", _key(0, stage="8192"): "att-rung"}))
    union = _admit(vol, (RUN_A,), [base, rung])                                 # an admission naming both (each entry its own stage)
    inbox = vol / "inbox"
    rep = PUB.publish(attempts, acc, inbox=inbox, host="h", union_dir=union)
    assert rep.published_attempt_ids() == {"att-base", "att-rung"} and rep.blocks["att-base"] != rep.blocks["att-rung"]
    assert _names(rep, "att-base", RUN_A) == _expected_names("att-base") and _names(rep, "att-rung", RUN_A) == _expected_names("att-rung")
    assert len({p["dest"] for p in rep.published}) == 6
    # a same-W RELAUNCH: a new wave_uuid, new attempt ids, the SAME run id and shard -> other blocks, none of the earlier names
    roots2 = vol / "stores2"
    b2 = _attempt(roots2, 0, "att-base-2", wave=WAVE2)
    _fill_spool(b2, (RUN_A,), 3, "base2")
    acc2 = PUB.load_accepted(_accepted(vol, {_key(0, wave=WAVE2): "att-base-2"}, name="accepted2.json"))
    union2 = _admit(vol, (RUN_A,), [b2], name="union2")
    rep2 = PUB.publish(PUB.discover_attempts([roots2]), acc2, inbox=inbox, host="h", union_dir=union2)
    assert _names(rep2, "att-base-2", RUN_A) == _expected_names("att-base-2")
    assert not ({Path(p["dest"]).name for p in rep2.published} & {Path(p["dest"]).name for p in rep.published})
    # the first wave's retry derives ITS names again (a pure function of the attempt id): all duplicates, nothing new
    rep3 = PUB.publish(attempts, acc, inbox=inbox, host="h", union_dir=union)
    assert rep3.published == [] and len(rep3.duplicates) == 6 and rep3.blocks == rep.blocks
    # a dry run copies nothing and needs no state either
    roots3 = vol / "stores3"
    b3 = _attempt(roots3, 1, "att-dry", wave=WAVE2)
    _fill_spool(b3, (RUN_A,), 1, "dry")
    acc3 = PUB.load_accepted(_accepted(vol, {_key(1, wave=WAVE2): "att-dry"}, name="accepted3.json"))
    rep4 = PUB.publish(PUB.discover_attempts([roots3]), acc3, inbox=vol / "inbox_dry", host="h", union_dir=_admit(vol, (RUN_A,), [b3], name="union3"),
                       dry_run=True)
    assert rep4.blocks == {"att-dry": PUB.attempt_block("att-dry")} and len(rep4.published) == 3 and not (vol / "inbox_dry").exists()
    # the block rule: 1 <= block, distinct across every attempt id this file minted, stable
    ids = ("att-0", "att-1", "att-base", "att-rung", "att-base-2", "att-dry", "att-0-rung", "att-0-w2")
    blocks = {i: PUB.attempt_block(i) for i in ids}
    assert len(set(blocks.values())) == len(ids) and all(b >= 1 for b in blocks.values()) and PUB.attempt_block("att-0") == blocks["att-0"]
    with pytest.raises(PUB.PublishRefused):
        PUB.block_seq(0, 1)
    with pytest.raises(PUB.PublishRefused):
        PUB.block_seq(1, PUB.ATTEMPT_BLOCK)
    assert PUB.published_name("000007-item_result.jsonl.gz", 5) == "5000007-item_result.jsonl.gz"


def test_known_bad_inputs_are_refused_or_skipped_with_a_named_reason(vol):
    roots = vol / "stores"
    a0 = _attempt(roots, 0, "att-0")
    a1 = _attempt(roots, 1, "att-1")
    noshard = _attempt(roots, 2, "att-noshard", with_shard=False)
    _fill_spool(a0, RUNS, 2, "s0")
    _fill_spool(a1, RUNS, 2, "s1")
    _fill_spool(noshard, (RUN_A,), 2, "s2")
    attempts = PUB.discover_attempts([roots])
    assert noshard.axes == {"w7": W7, "wave_uuid": WAVE, "container": LEG, "view": VIEW, "stage": "base", "shard": None}   # the spelled-out shape
    # an absent axis is a MISMATCH, never a wildcard — the legacy no-shard attempt is not accepted whatever key names its id
    full = PUB.parse_axes(_key(2))
    assert PUB.axis_mismatch(full, noshard.axes) == ("shard", 2, None) and PUB.axis_mismatch(full, dict(full)) is None
    assert PUB.axis_mismatch(full, {**full, "stage": None}) == ("stage", "base", None) and PUB.axis_mismatch({**full, "view": None}, full) == ("view", None, VIEW)
    assert PUB.absent_or_invalid_axes(noshard.axes) == ["shard"] and PUB.absent_or_invalid_axes(full) == []
    assert PUB.absent_or_invalid_axes({**full, "shard": "2"}) == ["shard"] and PUB.absent_or_invalid_axes({**full, "shard": -1}) == ["shard"]
    assert PUB.absent_or_invalid_axes({**full, "w7": "zzzzzzz"}) == ["w7"] and PUB.absent_or_invalid_axes({**full, "stage": None, "shard": None}) == ["stage", "shard"]
    inbox = vol / "inbox"
    # (a) an admission for RUN_A only: RUN_B's dirs are skipped with the reason, RUN_A publishes; a REFUSED verdict names itself
    union = _admit(vol, (RUN_A,), [a0, a1])
    acc = PUB.load_accepted(_accepted(vol, {_key(0): "att-0", _key(1): "att-1", _key(2): "att-noshard"}))
    rep = PUB.publish(attempts, acc, inbox=inbox, host="h", union_dir=union)
    assert {p["run_dir"] for p in rep.published} == {identity.run_dir_name(RUN_A)}
    assert {(s["run_dir"], s["reason"]) for s in rep.skipped_run_dirs} == {(identity.run_dir_name(RUN_B), "union_verdict:absent")}
    assert [(s["attempt_id"], s["reason"].split(";")[0]) for s in rep.skipped_attempts] == [("att-noshard", "ATTEMPT.json states no ['shard']")]
    assert acc.is_accepted(noshard) == (False, "ATTEMPT.json states no ['shard']; an attempt is accepted only on an exact six-axis match with its key")
    _refuse(vol, (RUN_B,))
    rep = PUB.publish(attempts, acc, inbox=vol / "inbox2", host="h", union_dir=union)
    assert {(s["run_dir"], s["reason"]) for s in rep.skipped_run_dirs} == {(identity.run_dir_name(RUN_B), "union_verdict:REFUSED")}
    # (b) the union verdict is REQUIRED on the production path
    with pytest.raises(PUB.PublishRefused, match="union-dir is required"):
        PUB.publish(attempts, acc, inbox=vol / "inbox3", host="h")
    # (c) a same-named destination with DIFFERENT bytes is refused, never overwritten
    dest = next(Path(p["dest"]) for p in PUB.publish(attempts, acc, inbox=vol / "inbox4", host="h", union_dir=union).published)
    dest.write_text('{"record_type":"tampered"}\n')
    with pytest.raises(PUB.PublishRefused, match="different bytes"):
        PUB.publish(attempts, acc, inbox=vol / "inbox4", host="h", union_dir=union)
    # (d) an attempt accepted under two keys; under another axis's key (each of the six); a host the merge would skip
    with pytest.raises(PUB.PublishRefused, match="two keys"):
        PUB.load_accepted(_accepted(vol / "x1", {_key(0): "att-0", _key(1): "att-0"}) if (vol / "x1").mkdir() is None else None)
    for kw, ax in (({"view": "s3"}, "view"), ({"stage": "8192"}, "stage"), ({"wave": WAVE2}, "wave_uuid"), ({"leg": "controls"}, "container"),
                   ({"w7": "fffffff"}, "w7"), ({"shard": 5}, "shard")):
        acc_other = PUB.load_accepted(_accepted(vol, {_key(**{**{"shard": 0}, **kw}): "att-0", _key(1): "att-1"}))
        rep = PUB.publish(attempts, acc_other, inbox=vol / f"inbox5{ax}", host="h", union_dir=union)
        assert rep.published_attempt_ids() == set()          # att-1 alone cannot publish: the admission binds BOTH shards
        assert [s for s in rep.skipped_attempts if s["attempt_id"] == "att-0"][0]["reason"].startswith(f"accepted_under_other_{ax}:")
        assert any(s["reason"].startswith("admitted_shard_missing:['att-0']") for s in rep.skipped_run_dirs)
    with pytest.raises(PUB.PublishRefused, match="host"):
        PUB.publish(attempts, acc, inbox=vol / "inbox6", host="_probe", union_dir=union)
    # (e) the three key spellings parse to ONE axis set; a key of ANY other shape is REFUSED at load — an attempt is never accepted by
    # its id alone; a malformed control_key in ATTEMPT.json refuses the root, it is not read past to the spelled-out
    # fields even when they are present; an incomplete attempt never publishes
    axes = PUB.parse_axes(_key(0))
    assert axes == PUB.parse_axes(_key(0).replace("/", "|")) == PUB.parse_axes(json.dumps(_key(0).split("/"))) and axes["shard"] == 0
    for n, (bad_key, why) in enumerate((("opaque-key-0", "no axis separator"), (f"{W7}/{WAVE}/{LEG}/{VIEW}/base", "5 parts"),
                                        (f"{W7}/{WAVE}/{LEG}/{VIEW}/base/0/extra", "7 parts"), (f"{W7}/{WAVE}/{LEG}/{VIEW}/base/x", "not a non-negative integer"),
                                        (f"{W7}/{WAVE}/{LEG}/{VIEW}/base/-1", "not a non-negative integer"), (f"{W7}/{WAVE}//{VIEW}/base/0", "empty container"),
                                        (f"{W7}/{WAVE}/{LEG}/{VIEW}/ /0", "empty stage"), (f"zz/{WAVE}/{LEG}/{VIEW}/base/0", "hex"),
                                        (json.dumps([W7, WAVE, LEG]), "3 parts"), ("", "no axis separator"))):
        with pytest.raises(PUB.PublishRefused, match=why):
            PUB.parse_axes(bad_key)
        with pytest.raises(PUB.PublishRefused, match="never accepted by its id alone"):
            PUB.load_accepted(_accepted(vol, {bad_key: "att-0", _key(1): "att-1"}, name=f"acc_bad_{n}.json"))
    assert PUB.parse_axes([W7, WAVE, LEG, VIEW, "base", 3])["shard"] == 3 == PUB.parse_axes([W7, WAVE, LEG, VIEW, "base", "3"])["shard"]   # a list key
    # a shard is read WITHOUT coercion — canonical decimal only, never stripped / padded / rounded / a boolean
    for bad_shard in (" 3 ", "03", "+3", "1.5", "true", "-0", "٣", "1\n"):
        with pytest.raises(PUB.PublishRefused, match="not a non-negative integer in canonical decimal|carries surrounding whitespace"):
            PUB.parse_axes([W7, WAVE, LEG, VIEW, "base", bad_shard])
    assert [PUB.parse_shard(v) for v in (0, 7, "0", "12", True, False, 1.5, -1, -0.5, "01", " 1", "+1", "1.0", "", None, [1])] == \
           [0, 7, 0, 12, None, None, None, None, None, None, None, None, None, None, None, None]
    assert PUB.parse_axes(f"{W7}/{WAVE}/{LEG}/{VIEW}/base/0")["shard"] == 0
    # `$` matches before a final newline, so "1\n" used to be shard 1 and "abcdefg\n" a valid sha prefix; and a key was .strip()ped
    # before splitting, so a delimited key ending in "1 " normalised to shard 1 — now a TRUE full match, and no trimming anywhere
    assert PUB.parse_shard("1\n") is None and PUB.parse_shard("1 ") is None and PUB.parse_shard("\n1") is None
    assert PUB.absent_or_invalid_axes({**full, "w7": "abcdef0\n"}) == ["w7"] and PUB.absent_or_invalid_axes({**full, "w7": "abcdef0"}) == []
    for bad_key, why in ((f"{_key(0)} ", "surrounding whitespace"), (f" {_key(0)}", "surrounding whitespace"), (f"{_key(0)}\n", "surrounding whitespace"),
                         (f"{W7}/{WAVE}/{LEG}/{VIEW}/base /0", "stage 'base ' carries surrounding whitespace"),
                         (f"{W7}/{WAVE}/{LEG}/{VIEW}/base/1\n", "surrounding whitespace"), (f"{W7}\n/{WAVE}/{LEG}/{VIEW}/base/1", "W7 .* carries surrounding whitespace")):
        with pytest.raises(PUB.PublishRefused, match=why):
            PUB.parse_axes(bad_key)
        with pytest.raises(PUB.PublishRefused, match="never accepted by its id alone"):
            PUB.load_accepted(_accepted(vol, {bad_key: "att-0"}, name=f"acc_ws_{abs(hash(bad_key))}.json"))
    for bad_list in ([W7, WAVE, LEG, VIEW, "base", "1\n"], [W7, WAVE, LEG, VIEW, "base", "1 "], [W7 + "\n", WAVE, LEG, VIEW, "base", 1], [W7, WAVE, LEG, VIEW, " base", 1]):
        with pytest.raises(PUB.PublishRefused, match="surrounding whitespace|not a non-negative integer in canonical decimal"):   # the shard by its own rule
            PUB.parse_axes(bad_list)
    # list members in their STATED types — a numeric W is not its hex rendering; the key itself must be str or list
    for bad_list, why in (([1234567, WAVE, LEG, VIEW, "base", 1], "W7 is 1234567 \\(int\\), not a string"), ([W7, 11111111, LEG, VIEW, "base", 1], "wave_uuid is 11111111 \\(int\\)"),
                          ([W7, WAVE, [LEG], VIEW, "base", 1], "container is \\['crescendo'\\] \\(list\\)"), ([W7, WAVE, LEG, 1.5, "base", 1], "view is 1.5 \\(float\\)"),
                          ([W7, WAVE, LEG, VIEW, True, 1], "stage is True \\(bool\\)"), ([W7, WAVE, LEG, VIEW, None, 1], "stage is None \\(NoneType\\)")):
        with pytest.raises(PUB.PublishRefused, match=why):
            PUB.parse_axes(bad_list)
        with pytest.raises(PUB.PublishRefused, match=why):
            PUB.parse_axes(json.dumps(bad_list))                                    # the same list as a JSON string: parsed members keep their types
    for bad_key in (1234567, 1.5, True, {"w7": W7}, b"x"):
        with pytest.raises(PUB.PublishRefused, match="a key is a string or a JSON list, never another type"):
            PUB.parse_axes(bad_key)
    with pytest.raises(PUB.PublishRefused, match="names no attempt id"):
        PUB.load_accepted(_accepted(vol, {_key(0): 7}, name="acc_int_aid.json"))
    with pytest.raises(PUB.PublishRefused, match="names no attempt id"):
        PUB.load_accepted(_accepted(vol, {_key(0): "att-0 "}, name="acc_ws_aid.json"))
    assert PUB.axis_mismatch(full, {**full, "shard": "2"}) == ("shard", 2, "2")   # typed: the string "2" is not the shard 2
    assert PUB.record_run_id({"record_type": 7, "payload": {"run_id": RUN_A}}) is None and PUB.record_run_id({"record_type": "run_patch", "payload": {"run_id": 7}}) is None
    bad_root = vol / "stores_badkey" / LEG / VIEW / "0" / "attempts" / "att-badkey"
    bad_root.mkdir(parents=True)
    (bad_root / PUB.ATTEMPT_FILE).write_text(json.dumps({"attempt_id": "att-badkey", "control_key": "opaque", "leg": LEG, "view": VIEW,
                                                         "stage": "base", "shard": 0, "W": WSHA, "wave_uuid": WAVE}))
    with pytest.raises(PUB.PublishRefused, match="malformed control_key is refused, not fallen back from"):
        PUB.read_attempt(bad_root)
    with pytest.raises(PUB.PublishRefused, match="malformed control_key"):
        PUB.discover_attempts([vol / "stores_badkey"])
    (bad_root / PUB.ATTEMPT_FILE).write_text(json.dumps({"attempt_id": "att-badkey", "control_key": None, "leg": LEG, "view": VIEW, "stage": "base",
                                                         "W": WSHA, "wave_uuid": WAVE}))
    assert PUB.read_attempt(bad_root).axes == noshard.axes          # a null key is an ABSENT key: the spelled-out shape still reads
    # (f) EVERY record of every admitted file must name the admission's run: a file whose FIRST row is RUN_A and a
    # later row RUN_B, and a file with a record lacking payload.run_id, each refuse the run — nothing is copied. The admission is built
    # by the test helper (no run check) so the refusal is publication's own, and the old first-record check would have passed both.
    roots9 = vol / "stores9"
    am = _attempt(roots9, 0, "att-m")
    _fill_spool(am, (RUN_A,), 2, "m")
    rd = am.spool / CAMPAIGN / identity.run_dir_name(RUN_A)
    prod = synth.producer(substrate="modal")
    mixed = rd / R.record_file_name(4, "run_patch")
    R.write_records(str(mixed), [R.make_record("run_patch", {"run_id": RUN_A, "fields": {"node_hostname": "a"}}, prod, "11111111-1111-4111-8111-111111111111", synth.t(1)),
                                 R.make_record("run_patch", {"run_id": RUN_B, "fields": {"node_hostname": "b"}}, prod, "22222222-2222-4222-8222-222222222222", synth.t(2))])
    union9 = _admit(vol, (RUN_A,), [am], name="union9")
    acc9 = PUB.load_accepted(_accepted(vol, {_key(0): "att-m"}, name="acc9.json"))
    with pytest.raises(PUB.PublishRefused, match="record 2 \\(run_patch\\) names run .*never published"):
        PUB.publish(PUB.discover_attempts([roots9]), acc9, inbox=vol / "inbox9", host="h", union_dir=union9)
    assert not (vol / "inbox9").exists()
    with pytest.raises(PUB.PublishRefused, match="record 2 \\(run_patch\\) names run"):
        PUB.run_dir_files(rd, RUN_A)
    assert mixed.name in PUB.run_dir_files(rd)                                  # without a run the helper still lists the files (the admission's input)
    # an ARTIFACT binds through extra.run_id (the store recorder's completion / rows artifacts): bound to RUN_A it publishes with the run,
    # bound to RUN_B or unbound it refuses; a record type with no run binding at all (object) never belongs in a run dir
    def art(run_id, aid):
        extra = {"run_id": run_id} if run_id is not None else {}
        return R.make_record("artifact", {"artifact_id": "f" * 64, "kind": "completion", "uri_primary": f"/x/{aid}", "bytes": 1, "extra": extra},
                             prod, aid, synth.t(4))
    assert PUB.record_run_id(art(RUN_A, "a")) == RUN_A and PUB.record_run_id(art(None, "a")) is None
    assert PUB.record_run_id(R.make_record("object", {"object_id": "o" * 64}, prod, "44444444-4444-4444-8444-444444444444", synth.t(5))) is None
    mixed.unlink()
    art_file = rd / R.record_file_name(6, "artifact")
    for bad_art, why in ((art(RUN_B, "55555555-5555-4555-8555-555555555555"), f"record 1 \\(artifact\\) names run {_re_escape(repr(RUN_B))}"),
                         (art(None, "66666666-6666-4666-8666-666666666666"), "record 1 \\(artifact\\) names run None")):
        R.write_records(str(art_file), [bad_art])
        with pytest.raises(PUB.PublishRefused, match=why):
            PUB.run_dir_files(rd, RUN_A)
        art_file.unlink()
    R.write_records(str(rd / R.record_file_name(7, "object")), [R.make_record("object", {"object_id": "o" * 64}, prod, "77777777-7777-4777-8777-777777777777", synth.t(6))])
    with pytest.raises(PUB.PublishRefused, match="record 1 \\(object\\) names run None"):
        PUB.run_dir_files(rd, RUN_A)
    (rd / R.record_file_name(7, "object")).unlink()
    R.write_records(str(art_file), [art(RUN_A, "88888888-8888-4888-8888-888888888888")])
    assert art_file.name in PUB.run_dir_files(rd, RUN_A)                     # the run's own artifact rides with it
    (rd / R.record_file_name(5, "run_patch")).write_text(json.dumps(R.make_record("run_patch", {"fields": {"node_hostname": "c"}}, prod,
                                                                                    "33333333-3333-4333-8333-333333333333", synth.t(3))) + "\n")
    union9b = _admit(vol, (RUN_A,), [am], name="union9b")
    with pytest.raises(PUB.PublishRefused, match="record 1 \\(run_patch\\) names run None"):
        PUB.publish(PUB.discover_attempts([roots9]), acc9, inbox=vol / "inbox9", host="h", union_dir=union9b)
    assert not (vol / "inbox9").exists()
    (rd / R.record_file_name(5, "run_patch")).unlink()
    union9c = _admit(vol, (RUN_A,), [am], name="union9c")                      # the clean dir (+ the run's own artifact) publishes
    rep9 = PUB.publish(PUB.discover_attempts([roots9]), acc9, inbox=vol / "inbox9", host="h", union_dir=union9c)
    assert rep9.published_attempt_ids() == {"att-m"} and len(rep9.published) == 4
    dead = _attempt(roots, 3, "att-dead", complete=False)
    _fill_spool(dead, (RUN_A,), 1, "dead")
    rep = PUB.publish(PUB.discover_attempts([roots]), PUB.load_accepted(_accepted(vol, {_key(3): "att-dead"})), inbox=vol / "inbox8", host="h", union_dir=union)
    assert rep.published == [] and rep.skipped_attempts[-1]["reason"].startswith("incomplete")


def test_parsers_apply_no_coercion_to_a_stated_value_source_guard():
    """Every function that READS a value an attempt, an accepted map, a launch plan, a sidecar or a record STATES is
    scanned (by AST, not text — comments and docstrings do not count) for `str(` / `int(` / `float(` / `.strip(` / `.lower(` / `.match(`
    calls. Every call found must be in the allowlist below with its reason, and every allowlisted call must still exist — so a future
    edit can neither reintroduce a coercion silently nor leave a stale excuse behind. The detector is itself checked on a known-bad."""
    import ast
    import inspect
    import textwrap
    from collections import Counter
    NAMES, ATTRS = {"str", "int", "float"}, {"strip", "lower", "match"}

    def coercions(fn) -> Counter:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        found: Counter = Counter()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if (isinstance(f, ast.Name) and f.id in NAMES) or (isinstance(f, ast.Attribute) and f.attr in ATTRS):
                    found[ast.unparse(node)] += 1
        return found
    # the detector on a known-bad
    def _bad(v):
        return str(v).strip().lower()
    assert set(coercions(_bad)) == {"str(v).strip().lower()", "str(v).strip()", "str(v)"}
    DETECT = "whitespace / emptiness DETECTION: the compared value stays the original"
    OWN_PATH = "a Path of our own, rendered for the os API — not a stated value"
    OWN_PLAN = "the plan identity: WS4-H's own generated file, validated at load by hooks.missing_identity_fields — not an attempt's statement"
    allow = {
        PUB.parse_shard: {"int(v)": "after SHARD_RE.fullmatch: a canonical ASCII decimal -> int is lossless"},
        PUB._spelled: {"v.strip()": DETECT},
        PUB.parse_axes: {"key.strip()": DETECT, "part.strip()": DETECT},
        PUB.axis_mismatch: {},
        PUB.absent_or_invalid_axes: {"v.strip()": DETECT},
        PUB.read_attempt: {"aid.strip()": DETECT, "sup.strip()": DETECT},
        PUB.load_accepted: {"aid.strip()": DETECT},
        PUB.record_run_id: {},
        PUB.run_dir_files: {"str(rd)": OWN_PATH},
        PUB.check_admitted_source: {},
        PUB.admission_for: {},
        LRC.load_launch_plan: {},
        LRC.launch_plan_join: {"str(Path(*pf.path.resolve().parts[-2:]))": OWN_PATH},
        LRC.shards_of: {},
        LRC.scan_run_dir: {"str(rd)": OWN_PATH},
        LRC.shard_items: {},
        LRC.evaluate_run: {"str(ident['run_id'])": OWN_PLAN, "str(ident['campaign'])": OWN_PLAN, "int(n_planned)": OWN_PLAN,
                           "int(expected_shards)": "a count WE derived (the CLI flag or the launch plan's base rows)",
                           "str(exc)": "exception rendering into the verdict's reasons", "str(sh.attempt.root)": OWN_PATH},
    }
    for fn, allowed in allow.items():
        found = coercions(fn)
        stray = sorted(set(found) - set(allowed))
        stale = sorted(set(allowed) - set(found))
        assert not stray, f"{fn.__module__}.{fn.__name__}: coercion-shaped call(s) on a value that may be STATED, not allowlisted: {stray}"
        assert not stale, f"{fn.__module__}.{fn.__name__}: allowlisted call(s) no longer present (update the list): {stale}"


def test_cli_dry_run_writes_nothing_and_reports(vol, capsys):
    roots = vol / "stores"
    a0 = _attempt(roots, 0, "att-0")
    _fill_spool(a0, (RUN_A,), 1, "s0")
    union = _admit(vol, (RUN_A,), [a0])
    acc = _accepted(vol, {_key(0): "att-0"})
    rc = PUB.main(["--accepted", str(acc), "--attempts", str(roots), "--union-dir", str(union), "--inbox", str(vol / "inbox"), "--host", "h", "--dry-run"])
    assert rc == 0
    rep = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rep["n_published"] == 3 and rep["blocks"] == {"att-0": PUB.attempt_block("att-0")} and not (vol / "inbox").exists()
    with pytest.raises(SystemExit):        # --union-dir is required by the CLI itself
        PUB.main(["--accepted", str(acc), "--attempts", str(roots), "--inbox", str(vol / "inbox"), "--host", "h"])
    rc = PUB.main(["--accepted", str(acc), "--attempts", str(roots), "--union-dir", str(union), "--inbox", str(vol / "inbox"), "--host", "h"])
    assert rc == 0 and (vol / "inbox" / "h" / PUB.LEDGER_DIR / "published.jsonl").is_file()
