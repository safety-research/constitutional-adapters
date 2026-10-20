"""The Petri drive -> proxy completion contract (the TOCTOU).

The proxy accepts a Petri drive ONLY through attempt-bound evidence; the drive used to publish a counts-only STOP, so every
SUCCESSFUL drive landed as `stopped_unbound`, rc 1. The drive now re-reads the tunnel record at publish time and publishes ONE
atomic bundle (the verdict record + the STOP content carrying `attempt=<id>`) under the ATTEMPT-NAMESPACED key
`<K>/<attempt_id>/completion.json` while the record is still this drive's attempt -- and NOTHING when the record is missing (the
proxy's wait ended; it pops the record), unreadable or another attempt's (stale), or when the drive HOLDS (that wait ends on ANY
non-waiting evidence, so a verdict would end the proxy the hold keeps up). The binding read is not atomic with the write: a
replacement attempt B may install its record under the same six-axis key in between. The attempt id in the KEY is
what makes that harmless -- B derives and watches only ITS key. The generic `<K>/drive_verdict.json` + `<K>/STOP` pair (readable
by whoever holds the key) is written only under `--legacy-generic-completion`, and the race tests below show why it is opt-in.

`ws2d_parse_stop_sentinel` / `ws2d_petri_completion_evidence` are the proxy's, copied VERBATIM from `scripts/modal/seedrep_served.py`
@ `seedrep-ws2d` 0e51107b (:1041-1102; their module imports `modal` and builds images at load, so it is replicated rather than
imported). `ws2d_attempt_completion_evidence` is the reader the proxy derives: ITS verdict logic applied at the
current attempt's key. The proxy's own tests run against THIS publisher (`publish_completion`, `completion_key`, `completion_bundle`,
`drive_verdict_record`, `stop_sentinel_extra`, `sentinel_content`).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts/modal/seedrep_petri_drive.py"
sys.path.insert(0, str(REPO / "scripts"))

from test_seedrep_petri_drive import WAVE, _load  # noqa: E402
from test_seedrep_petri_drive_review_fixes import ATTEMPT, ATTEMPT_ROOT, MINIMUM_TUNNEL, TUNNEL_KEY, W, W7  # noqa: E402


@pytest.fixture(scope="module")
def d():
    return _load()


# --------------------------------------------------------------- the proxy's consumer, VERBATIM (seedrep_served.py @ 0e51107b :1041-1102)
PETRI_VERDICT_NAME = "drive_verdict.json"
STOP_SENTINEL_RE = re.compile(r"^STOP\s+(?P<utc>\S+)\s+drive=(?P<drive>\S+)(?P<extra>.*)$", re.S)


def ws2d_parse_stop_sentinel(content):
    text = content.decode(errors="replace") if isinstance(content, (bytes, bytearray)) else str(content or "")
    m = STOP_SENTINEL_RE.match(text.strip())
    if not m:
        return None
    out = {"utc": m.group("utc"), "drive_id": m.group("drive")}
    for tok in m.group("extra").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = int(v) if v.isdigit() else v
    return out


def ws2d_petri_completion_evidence(sentinels, key, attempt_id):
    verdict_raw = sentinels.get(f"{key}/{PETRI_VERDICT_NAME}")
    if verdict_raw is not None:
        try:
            v = json.loads(verdict_raw) if isinstance(verdict_raw, (str, bytes, bytearray)) else dict(verdict_raw)
        except (ValueError, TypeError):
            return {"state": "stopped_unbound", "why": "drive_verdict.json is not JSON", "evidence": None}
        counts = (v.get("verdict") or {}).get("counts") or {}
        if v.get("attempt_id") != attempt_id:
            return {"state": "stopped_unbound", "why": f"drive_verdict names attempt {v.get('attempt_id')!r}, not {attempt_id}", "evidence": v}
        if (v.get("verdict") or {}).get("release") is True and int(counts.get("failed", 1)) == 0:
            return {"state": "complete", "why": "drive verdict bound to this attempt, release with zero failed", "evidence": v}
        return {"state": "incomplete", "why": f"drive verdict counts {counts}", "evidence": v}
    stop_raw = sentinels.get(f"{key}/STOP")
    if stop_raw is None:
        return {"state": "waiting", "why": "no STOP / verdict sentinel yet", "evidence": None}
    st = ws2d_parse_stop_sentinel(stop_raw)
    if st is None:
        return {"state": "stopped_unbound", "why": f"STOP content unparseable: {str(stop_raw)[:120]!r}", "evidence": None}
    if st.get("attempt") != attempt_id:
        return {"state": "stopped_unbound", "why": f"STOP carries attempt={st.get('attempt')!r}, not this attempt's id", "evidence": st}
    if int(st.get("failed", 1)) == 0:
        return {"state": "complete", "why": "STOP bound to this attempt with failed=0", "evidence": st}
    return {"state": "incomplete", "why": f"STOP bound to this attempt with failed={st.get('failed')}", "evidence": st}


def ws2d_attempt_completion_evidence(sentinels, key, attempt_id):
    """The reader the proxy derives: read ONLY `<K>/<attempt_id>/completion.json` for the CURRENT attempt, and
    judge it with the verdict branch of the verbatim reader above -- so the bundle must satisfy their EXISTING verdict logic byte for byte."""
    raw = sentinels.get(f"{key}/{attempt_id}/completion.json")
    if raw is None:
        return {"state": "waiting", "why": "no completion bundle under this attempt's key", "evidence": None}
    return ws2d_petri_completion_evidence({f"{key}/{PETRI_VERDICT_NAME}": raw}, key, attempt_id)


# --------------------------------------------------------------------------- the drive's side
KEY = TUNNEL_KEY                                                                 # the drive's key_for("") for this proxy, == control_key
BOUND = {"attempt_id": ATTEMPT, "attempt_root_on_volume": ATTEMPT_ROOT, "control_key": TUNNEL_KEY}   # what main binds at start
CK = f"{KEY}/{ATTEMPT}/completion.json"                                          # THE attempt-namespaced completion key
IDENTITY = {"W": W, "W7": W7, "wave_uuid": WAVE, "view": "s1", "stage": "base", "shard": 0, "petri_pass": ""}
RELEASE = {"counts": {"complete": 8, "incomplete": 0, "failed": 0}, "nruns": 8, "release": True,
           "detail": [(f"leg_{i}_s1", f"rk{i}", "complete") for i in range(8)]}
UTC = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
#: the REPLACEMENT proxy attempt B of the same row (same six-axis key), as a re-execution would publish it
ATT_B2 = "ta-01HREPL-bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"
ROOT_B2 = ATTEMPT_ROOT.rsplit("/", 1)[0] + "/" + ATT_B2
RECORD_B = {**MINIMUM_TUNNEL, "attempt_id": ATT_B2, "attempt_root_on_volume": ROOT_B2, "function_call_id": "fc-2",
            "serving_dir_on_volume": f"{ROOT_B2}/exp/seedrep_battery_modal/petri/s1/serve"}
TK = f"{KEY}/modal_tunnel.json"


class Recording(dict):
    """A Dict double that records the ORDER of writes."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.writes = []

    def __setitem__(self, k, v):
        self.writes.append(k)
        super().__setitem__(k, v)


class Racing(Recording):
    """A Dict double in which replacement attempt B installs ITS tunnel record at a chosen point of A's write sequence: B's record lands
    immediately BEFORE A's write number ``install_before`` (0 = after A's binding read, before A's first write; n = between A's n-th and
    (n+1)-th write) -- the TOCTOU window, reproduced."""

    def __init__(self, *a, install_before, record_b=RECORD_B, **k):
        super().__init__(*a, **k)
        self.install_before, self.record_b, self.installed_at = install_before, record_b, None

    def __setitem__(self, k, v):
        if self.installed_at is None and len(self.writes) == self.install_before:
            dict.__setitem__(self, TK, json.dumps(self.record_b))                 # B clears the key's sentinels and installs its record
            self.installed_at = len(self.writes)
        super().__setitem__(k, v)


def _publish(d, sentinels, *, current, verdict, write_stop, stop_why, drive_id="D1", legacy=False):
    logs = []
    out = d.publish_completion(sentinels, current_tunnel=current, key=KEY, bound=BOUND, drive_id=drive_id, verdict=verdict, identity=IDENTITY,
                               write_stop=write_stop, stop_why=stop_why, legacy_generic=legacy, log=logs.append)
    return out, logs


def test_a_released_drive_publishes_one_atomic_bundle_under_its_attempt_key_that_ws2d_accepts(d):
    sentinels = Recording({TK: json.dumps(MINIMUM_TUNNEL)})
    out, logs = _publish(d, sentinels, current=sentinels[TK], verdict=RELEASE, write_stop=True, stop_why="release")
    assert out["binding"] == "bound" and out["binding_why"] == f"the record under the key names this drive's attempt {ATTEMPT}"
    assert out["completion_key"] == CK == d.completion_key(KEY, ATTEMPT) and out["published"] == [CK] == sentinels.writes       # ONE write
    assert logs == [] and out["write_stop"] is True and out["stop_why"] == "release" and out["legacy_generic"] is False
    assert set(sentinels) == {TK, CK}                                              # NO generic key by default
    # the bundle: the proxy's required keys + the drive's provenance + the STOP, exact
    bundle = json.loads(sentinels[CK])
    assert bundle == out["bundle"]
    assert re.fullmatch(UTC, bundle.pop("written_at"))
    stop = bundle["stop"]["content"]
    assert out["stop_content"] == stop
    assert re.fullmatch(f"STOP {UTC} drive=D1 complete=8 incomplete=0 failed=0 attempt={re.escape(ATTEMPT)}\n", stop)
    assert bundle == {"schema": "seedrep_petri_drive_completion/1", "attempt_id": ATTEMPT, "attempt_root_on_volume": ATTEMPT_ROOT, "control_key": KEY,
                      "drive_id": "D1", "W": W, "W7": W7, "wave_uuid": WAVE, "view": "s1", "stage": "base", "shard": 0, "petri_pass": "",
                      "verdict": {"counts": {"complete": 8, "incomplete": 0, "failed": 0}, "nruns": 8, "release": True,
                                  "detail": [[f"leg_{i}_s1", f"rk{i}", "complete"] for i in range(8)]},
                      "stop_why": "release", "stop": {"content": stop, "why": "release"}}
    # the verdict-record view (what the legacy mirror writes) is the bundle minus `stop`, under the verdict schema
    vr = dict(out["verdict_record"])
    assert re.fullmatch(UTC, vr.pop("written_at"))
    assert vr == {**{k: v for k, v in bundle.items() if k != "stop"}, "schema": "seedrep_petri_drive_verdict/1"}
    # the proxy's verdict logic at THIS attempt's key: complete; at another attempt's key: waiting; its generic reader: nothing to read
    ev = ws2d_attempt_completion_evidence(sentinels, KEY, ATTEMPT)
    assert ev["state"] == "complete" and ev["why"] == "drive verdict bound to this attempt, release with zero failed" and ev["evidence"]["drive_id"] == "D1"
    assert ws2d_attempt_completion_evidence(sentinels, KEY, ATT_B2)["state"] == "waiting"
    assert ws2d_petri_completion_evidence(sentinels, KEY, ATTEMPT)["state"] == "waiting"
    assert d.drive_rc(RELEASE, out) == 0


def test_the_legacy_generic_mirror_is_opt_in_bundle_first_and_satisfies_ws2d_s_verbatim_generic_reader(d):
    sentinels = Recording({TK: json.dumps(MINIMUM_TUNNEL)})
    out, logs = _publish(d, sentinels, current=sentinels[TK], verdict=RELEASE, write_stop=True, stop_why="release", legacy=True)
    assert out["published"] == [CK, f"{KEY}/drive_verdict.json", f"{KEY}/STOP"] == sentinels.writes and out["legacy_generic"] is True and logs == []
    rec = json.loads(sentinels[f"{KEY}/drive_verdict.json"])
    assert rec == out["verdict_record"] and rec["schema"] == "seedrep_petri_drive_verdict/1" and "stop" not in rec and rec["attempt_id"] == ATTEMPT
    assert sentinels[f"{KEY}/STOP"] == out["stop_content"] == out["bundle"]["stop"]["content"]
    # the proxy's verbatim generic reader: complete via the verdict, then via the STOP alone
    ev = ws2d_petri_completion_evidence(sentinels, KEY, ATTEMPT)
    assert ev["state"] == "complete" and ev["why"] == "drive verdict bound to this attempt, release with zero failed"
    sentinels.pop(f"{KEY}/drive_verdict.json")
    ev = ws2d_petri_completion_evidence(sentinels, KEY, ATTEMPT)
    assert ev["state"] == "complete" and ev["why"] == "STOP bound to this attempt with failed=0"
    assert ev["evidence"]["attempt"] == ATTEMPT and ev["evidence"]["failed"] == 0 and ev["evidence"]["drive_id"] == "D1"
    # KNOWN-BAD (the original repro): the OLD counts-only STOP of a successful drive is stopped_unbound to the consumer
    old = {f"{KEY}/STOP": d.sentinel_content("STOP", "D1", "complete=8 incomplete=0 failed=0").decode()}
    ev_old = ws2d_petri_completion_evidence(old, KEY, ATTEMPT)
    assert ev_old["state"] == "stopped_unbound" and ev_old["why"] == "STOP carries attempt=None, not this attempt's id"
    assert d.drive_rc(RELEASE, out) == 0


def test_the_toctou_a_replacement_attempt_installing_its_record_after_a_s_binding_read_cannot_misread_a_s_completion(d):
    # (a) THE FIX: B installs its record after A validated record A and before A's (single) write -- A's bundle lands under A's key,
    # B (watching ITS key) sees nothing of A's, and the generic reader has nothing to misread
    s = Racing({TK: json.dumps(MINIMUM_TUNNEL)}, install_before=0)
    out, _ = _publish(d, s, current=json.dumps(MINIMUM_TUNNEL), verdict=RELEASE, write_stop=True, stop_why="release")
    assert s.installed_at == 0 and json.loads(s[TK])["attempt_id"] == ATT_B2        # B was the live attempt when A's write landed
    assert out["published"] == [CK] and CK in s and f"{KEY}/{ATT_B2}/completion.json" not in s and set(s) == {TK, CK}
    assert ws2d_attempt_completion_evidence(s, KEY, ATT_B2)["state"] == "waiting"
    assert ws2d_attempt_completion_evidence(s, KEY, ATTEMPT)["state"] == "complete"
    assert ws2d_petri_completion_evidence(s, KEY, ATT_B2) == {"state": "waiting", "why": "no STOP / verdict sentinel yet", "evidence": None}
    # (b) THE REPRO (re-validated): with the legacy generic mirror ON, B installs between A's writes -> the generic reader for B
    # classifies A's attempt-bound evidence as stopped_unbound and B is torn down; the namespaced reader for B is unaffected. This is why the
    # generic pair is opt-in.
    for install_before in (1, 2):          # 1: between the bundle and the generic verdict; 2: between the generic verdict and the generic STOP
        s = Racing({TK: json.dumps(MINIMUM_TUNNEL)}, install_before=install_before)
        out, _ = _publish(d, s, current=json.dumps(MINIMUM_TUNNEL), verdict=RELEASE, write_stop=True, stop_why="release", legacy=True)
        assert s.installed_at == install_before and out["published"] == [CK, f"{KEY}/drive_verdict.json", f"{KEY}/STOP"]
        ev = ws2d_petri_completion_evidence(s, KEY, ATT_B2)
        assert ev["state"] == "stopped_unbound" and ev["why"] == f"drive_verdict names attempt {ATTEMPT!r}, not {ATT_B2}"
        assert ws2d_attempt_completion_evidence(s, KEY, ATT_B2)["state"] == "waiting"
        assert ws2d_attempt_completion_evidence(s, KEY, ATTEMPT)["state"] == "complete"
    # (c) ... and when B's clear had popped the generic verdict so only A's generic STOP is seen, the generic reader is stopped_unbound on the STOP
    s.pop(f"{KEY}/drive_verdict.json")
    ev = ws2d_petri_completion_evidence(s, KEY, ATT_B2)
    assert ev["state"] == "stopped_unbound" and ev["why"] == f"STOP carries attempt={ATTEMPT!r}, not this attempt's id"
    # (d) B's own completion later lands under B's key and disturbs nothing of A's
    b_bound = {"attempt_id": ATT_B2, "attempt_root_on_volume": ROOT_B2, "control_key": KEY}
    outb = d.publish_completion(s, current_tunnel=json.dumps(RECORD_B), key=KEY, bound=b_bound, drive_id="D2", verdict=RELEASE, identity=IDENTITY,
                                write_stop=True, stop_why="release", log=lambda m: None)
    assert outb["published"] == [f"{KEY}/{ATT_B2}/completion.json"] == [d.completion_key(KEY, ATT_B2)]
    assert ws2d_attempt_completion_evidence(s, KEY, ATT_B2)["state"] == "complete" and ws2d_attempt_completion_evidence(s, KEY, ATTEMPT)["state"] == "complete"
    assert json.loads(s[f"{KEY}/{ATT_B2}/completion.json"])["drive_id"] == "D2" and json.loads(s[CK])["drive_id"] == "D1"
    # (e) the stale case at read time is still refused outright: A reads B's record -> nothing published (rc 2)
    out_stale, _ = _publish(d, Recording({TK: json.dumps(RECORD_B)}), current=json.dumps(RECORD_B), verdict=RELEASE, write_stop=True, stop_why="release")
    assert out_stale["binding"] == "stale" and out_stale["published"] == [] and d.drive_rc(RELEASE, out_stale) == 2


@pytest.mark.parametrize("current, state, why", [
    (None, "missing", f"no modal_tunnel.json under the key -- proxy attempt {ATTEMPT} ended (WS2-D pops the record when its wait ends) "
                      "or its record was never published"),
    (b"{not json", "unreadable", "modal_tunnel.json under the key is not JSON (Expecting property name enclosed in double quotes: line 1 column 2 (char 1))"),
    ("[]", "unreadable", "modal_tunnel.json under the key is not a JSON object (list)"),
])
def test_a_missing_or_unreadable_tunnel_record_refuses_to_publish_anything(d, current, state, why):
    sentinels = Recording()
    out, logs = _publish(d, sentinels, current=current, verdict=RELEASE, write_stop=True, stop_why="release")
    assert (out["binding"], out["binding_why"]) == (state, why)
    assert out["published"] == [] and out["completion_key"] is None and out["bundle"] is None and out["verdict_record"] is None and out["stop_content"] is None
    assert sentinels.writes == [] and dict(sentinels) == {}
    assert logs == [f"COMPLETION_UNPUBLISHED binding={state}: {why} -- no completion.json written under this drive's attempt key; the Petri stores of "
                    f"attempt {ATTEMPT} stand locally; a drive bound to the live attempt continues them with --resume-from"]
    assert ws2d_attempt_completion_evidence(sentinels, KEY, ATTEMPT)["state"] == "waiting"
    assert d.drive_rc(RELEASE, out) == 2                                        # released, NOT publishable -- never 0


def test_a_stale_tunnel_record_of_another_attempt_refuses_to_publish_and_names_every_differing_field(d):
    sentinels = Recording({TK: json.dumps(RECORD_B)})
    out, logs = _publish(d, sentinels, current=json.dumps(RECORD_B), verdict=RELEASE, write_stop=True, stop_why="release")
    assert out["binding"] == "stale"
    assert out["binding_why"] == ("another attempt's record now holds the key -- "
                                  f"attempt_id: record {ATT_B2!r} != bound {ATTEMPT!r}; "
                                  f"attempt_root_on_volume: record {ROOT_B2!r} != bound {ATTEMPT_ROOT!r}")
    assert out["published"] == [] and sentinels.writes == [] and set(sentinels) == {TK}
    assert len(logs) == 1 and logs[0].startswith("COMPLETION_UNPUBLISHED binding=stale: another attempt's record now holds the key")
    assert d.drive_rc(RELEASE, out) == 2
    assert ws2d_attempt_completion_evidence(sentinels, KEY, ATT_B2)["state"] == "waiting"       # the live attempt keeps waiting (not ours)
    # the legacy STOP this drive WOULD have written IS stopped_unbound to B (the hazard the refusal and the namespace both close)
    would_be_stop = {f"{KEY}/STOP": d.sentinel_content("STOP", "D1", d.stop_sentinel_extra(RELEASE["counts"], ATTEMPT, "release")).decode()}
    assert ws2d_petri_completion_evidence(would_be_stop, KEY, ATT_B2)["why"] == f"STOP carries attempt={ATTEMPT!r}, not this attempt's id"
    # a record differing ONLY in control_key (a mis-keyed record of another stage) is stale too, named alone
    other_key = {**MINIMUM_TUNNEL, "control_key": f"{W7}/{WAVE}/petri/s1/petri-rerun/0"}
    out2, _ = _publish(d, Recording(), current=json.dumps(other_key), verdict=RELEASE, write_stop=True, stop_why="release")
    assert out2["binding"] == "stale"
    assert out2["binding_why"] == f"another attempt's record now holds the key -- control_key: record {other_key['control_key']!r} != bound {KEY!r}"
    other_root = {**MINIMUM_TUNNEL, "attempt_root_on_volume": ATTEMPT_ROOT + "x"}
    out3, _ = _publish(d, Recording(), current=other_root, verdict=RELEASE, write_stop=True, stop_why="release")   # a dict, as a Dict may hand back
    assert out3["binding"] == "stale" and out3["binding_why"].endswith(f"attempt_root_on_volume: record {ATTEMPT_ROOT + 'x'!r} != bound {ATTEMPT_ROOT!r}")
    # bytes of the SAME record bind; the producer's extra fields (host/port/relay_port/standup_id/written_at) do not unbind
    rich = {**MINIMUM_TUNNEL, "host": "x.modal.host", "port": 443, "relay_port": 18984, "standup_id": "su-1", "written_at": "2026-09-18T00:00:00Z"}
    out4, _ = _publish(d, Recording(), current=json.dumps(rich).encode(), verdict=RELEASE, write_stop=True, stop_why="release")
    assert out4["binding"] == "bound" and out4["published"] == [CK]


def test_a_held_drive_publishes_nothing_so_ws2d_keeps_waiting_and_the_proxy_stays_up(d):
    held = {"counts": {"complete": 6, "incomplete": 0, "failed": 2}, "nruns": 8, "release": False, "detail": []}
    write_stop, why = d.release_decision(held)
    assert (write_stop, why) == (False, "held")
    sentinels = Recording({TK: json.dumps(MINIMUM_TUNNEL)})
    out, logs = _publish(d, sentinels, current=json.dumps(MINIMUM_TUNNEL), verdict=held, write_stop=write_stop, stop_why=why)
    assert out["binding"] == "bound" and out["published"] == [] and out["bundle"] is None and out["verdict_record"] is None and out["stop_content"] is None
    assert sentinels.writes == [] and set(sentinels) == {TK}
    assert logs == ["COMPLETION_HELD: no completion.json published (binding=bound) -- the proxy stays up for the re-drive"]
    assert ws2d_attempt_completion_evidence(sentinels, KEY, ATTEMPT)["state"] == "waiting"
    assert d.drive_rc(held, out) == 1
    # KNOWN-BAD: had the bundle gone out, the consumer's wait would have ENDED on it (incomplete, rc 1) -- the proxy the hold keeps up would be gone
    leaked = {CK: json.dumps(d.completion_bundle(bound=BOUND, drive_id="D1", verdict=held, identity=IDENTITY, stop_why="held"))}
    assert ws2d_attempt_completion_evidence(leaked, KEY, ATTEMPT)["state"] == "incomplete"
    # a held drive over a MISSING record publishes nothing as well, and says which it is
    out2, logs2 = _publish(d, Recording(), current=None, verdict=held, write_stop=False, stop_why="held")
    assert out2["binding"] == "missing" and out2["published"] == [] and logs2 == ["COMPLETION_HELD: no completion.json published (binding=missing) -- "
                                                                                 "the proxy stays up for the re-drive"]


def test_teardown_after_failure_publishes_a_bound_bundle_that_ws2d_reads_as_incomplete_never_complete(d):
    failed = {"counts": {"complete": 5, "incomplete": 1, "failed": 2}, "nruns": 8, "release": False, "detail": [("leg_3_s1", "rk3", "failed_rc_1")]}
    write_stop, why = d.release_decision(failed, teardown_after_failure=True)
    assert (write_stop, why) == (True, "teardown-after-failure")
    sentinels = Recording()
    out, logs = _publish(d, sentinels, current=json.dumps(MINIMUM_TUNNEL), verdict=failed, write_stop=write_stop, stop_why=why)
    assert out["published"] == [CK] == sentinels.writes and logs == []
    b = json.loads(sentinels[CK])
    assert b["attempt_id"] == ATTEMPT and b["verdict"]["release"] is False and b["verdict"]["counts"] == {"complete": 5, "incomplete": 1, "failed": 2}
    assert b["verdict"]["detail"] == [["leg_3_s1", "rk3", "failed_rc_1"]] and b["stop_why"] == b["stop"]["why"] == "teardown-after-failure"
    assert re.fullmatch(f"STOP {UTC} drive=D1 complete=5 incomplete=1 failed=2 attempt={re.escape(ATTEMPT)} reason=teardown-after-failure\n",
                        b["stop"]["content"])
    ev = ws2d_attempt_completion_evidence(sentinels, KEY, ATTEMPT)
    assert ev["state"] == "incomplete" and ev["why"] == "drive verdict counts {'complete': 5, 'incomplete': 1, 'failed': 2}"
    # the legacy mirror of the same: incomplete via the generic verdict, then via the generic STOP alone
    s2 = Recording()
    _publish(d, s2, current=json.dumps(MINIMUM_TUNNEL), verdict=failed, write_stop=write_stop, stop_why=why, legacy=True)
    assert ws2d_petri_completion_evidence(s2, KEY, ATTEMPT)["state"] == "incomplete"
    s2.pop(f"{KEY}/drive_verdict.json")
    ev2 = ws2d_petri_completion_evidence(s2, KEY, ATTEMPT)
    assert ev2["state"] == "incomplete" and ev2["why"] == "STOP bound to this attempt with failed=2" and ev2["evidence"]["reason"] == "teardown-after-failure"
    assert d.drive_rc(failed, out) == 1


def test_the_publisher_pieces_refuse_bad_inputs_and_main_wires_them_in_order(d):
    # completion_key: the exact spelling; a bad attempt id / a bad six-axis key refused
    assert d.completion_key(KEY, ATTEMPT) == f"{KEY}/{ATTEMPT}/completion.json" == CK
    for bad in ("", "a/b", "../x", ".hidden", None, 7):
        with pytest.raises(SystemExit, match="not a path-safe attempt id"):
            d.completion_key(KEY, bad)
    for badkey in ("", KEY + "/", None):
        with pytest.raises(ValueError, match="needs the six-axis key"):
            d.completion_key(badkey, ATTEMPT)
    # publish_completion refuses a key that is not the bound record's control_key, before any read or write
    s = Recording({TK: json.dumps(MINIMUM_TUNNEL)})
    with pytest.raises(ValueError, match="is not the bound record's control_key"):
        d.publish_completion(s, current_tunnel=s[TK], key=KEY + "x", bound=BOUND, drive_id="D1", verdict=RELEASE, identity=IDENTITY, write_stop=True,
                             stop_why="release", log=lambda m: None)
    assert s.writes == []
    # completion_bundle = the verdict record + `stop`, under the completion schema
    b = d.completion_bundle(bound=BOUND, drive_id="D1", verdict=RELEASE, identity=IDENTITY, stop_why="release")
    vr = d.drive_verdict_record(bound=BOUND, drive_id="D1", verdict=RELEASE, identity=IDENTITY, stop_why="release")
    assert set(b) - set(vr) == {"stop"} and b["schema"] == "seedrep_petri_drive_completion/1" and vr["schema"] == "seedrep_petri_drive_verdict/1"
    assert {k: v for k, v in b.items() if k not in ("stop", "schema", "written_at")} == {k: v for k, v in vr.items() if k not in ("schema", "written_at")}
    assert b["stop"]["why"] == "release" and re.fullmatch(f"STOP {UTC} drive=D1 complete=8 incomplete=0 failed=0 attempt={re.escape(ATTEMPT)}\n", b["stop"]["content"])
    # stop_sentinel_extra: the exact grammar; a bad attempt id and a spaced / empty reason refused
    assert d.stop_sentinel_extra({"complete": 8, "incomplete": 0, "failed": 0}, ATTEMPT, "release") == f"complete=8 incomplete=0 failed=0 attempt={ATTEMPT}"
    assert d.stop_sentinel_extra({"complete": 1, "incomplete": 2, "failed": 3}, ATTEMPT, "teardown-after-failure") \
        == f"complete=1 incomplete=2 failed=3 attempt={ATTEMPT} reason=teardown-after-failure"
    for bad in ("", "a b", "../x", ".hidden", None, 7):
        with pytest.raises(SystemExit, match="not a path-safe attempt id"):
            d.stop_sentinel_extra({"complete": 8, "incomplete": 0, "failed": 0}, bad, "release")
    for bad_why in ("", "tear down", "a\tb"):
        with pytest.raises(ValueError, match="one whitespace-free token"):
            d.stop_sentinel_extra({"complete": 8, "incomplete": 0, "failed": 0}, ATTEMPT, bad_why)
    with pytest.raises(SystemExit, match="not a path-safe attempt id"):
        d.drive_verdict_record(bound={**BOUND, "attempt_id": "a/b"}, drive_id="D1", verdict=RELEASE, identity=IDENTITY, stop_why="release")
    # attempt_binding alone
    assert d.attempt_binding(BOUND, json.dumps(MINIMUM_TUNNEL)) == ("bound", f"the record under the key names this drive's attempt {ATTEMPT}")
    assert d.attempt_binding(BOUND, None)[0] == "missing" and d.attempt_binding(BOUND, "null")[0] == "unreadable"
    assert d.attempt_binding(BOUND, {**MINIMUM_TUNNEL, "attempt_id": None})[0] == "stale"
    # read_sentinel_now: absent -> None; a RAISING Dict is retried, then re-raised -- never read as missing
    assert d.read_sentinel_now({"k": "v"}, "k") == "v" and d.read_sentinel_now({}, "k") is None

    class Flaky:
        def __init__(self, fails):
            self.fails, self.calls = fails, 0

        def get(self, k):
            self.calls += 1
            if self.calls <= self.fails:
                raise ConnectionError("blip")
            return "v"
    slept, logs = [], []
    assert d.read_sentinel_now(Flaky(2), "k", sleep=slept.append, log=logs.append) == "v"
    assert slept == [10.0, 10.0] and logs == ["sentinel read k failed (ConnectionError: blip) -- try 1/3", "sentinel read k failed (ConnectionError: blip) -- try 2/3"]
    with pytest.raises(ConnectionError, match="blip"):
        d.read_sentinel_now(Flaky(3), "k", sleep=slept.append, log=logs.append)
    assert slept == [10.0] * 4 and len(logs) == 5
    # drive_rc: released + bound + published -> 0; released + anything else -> 2; not released -> 1 whatever the binding
    pub = {"binding": "bound", "published": [CK]}
    assert d.drive_rc(RELEASE, pub) == 0
    for c in ({"binding": "missing", "published": []}, {"binding": "stale", "published": []}, {"binding": "unreadable", "published": []},
              {"binding": "bound", "published": []}):
        assert d.drive_rc(RELEASE, c) == 2
    held = {**RELEASE, "release": False, "counts": {"complete": 7, "incomplete": 0, "failed": 1}}
    assert d.drive_rc(held, pub) == 1 and d.drive_rc(held, {"binding": "missing", "published": []}) == 1
    # the constants agree with the consumer's; the binding fields are fields the drive REQUIRES of the record at start
    assert d.COMPLETION_NAME == "completion.json" and d.COMPLETION_SCHEMA == "seedrep_petri_drive_completion/1"
    assert d.VERDICT_NAME == PETRI_VERDICT_NAME == "drive_verdict.json" and d.VERDICT_SCHEMA == "seedrep_petri_drive_verdict/1"
    assert d.STOP_NAME == "STOP" and d.TUNNEL_FILE == "modal_tunnel.json"
    assert d.BINDING_FIELDS == ("attempt_id", "attempt_root_on_volume", "control_key") and all(k in d.TUNNEL_REQUIRED_KEYS for k in d.BINDING_FIELDS)
    # main(): the generic STOP write is gone; release decision -> bind from the START record -> re-read NOW -> publish under the attempt key
    # (legacy mirror by flag only) -> local copy of the bundle -> the record carries the completion, stop_written means WRITTEN -> drive_rc
    src = SRC.read_text()
    main = src.split("def main(")[1]
    assert 'ap.add_argument("--legacy-generic-completion", action="store_true"' in src      # the parser is built outside main()
    assert 'sentinels[stop_key] = sentinel_content("STOP"' not in main and 'return 0 if verdict["release"] else 1' not in main
    assert "verdict_key=key_for(VERDICT_NAME)" not in main and "(astate / VERDICT_NAME)" not in main
    i = main.index
    assert i("write_stop, stop_why = release_decision(verdict, teardown_after_failure=args.teardown_after_failure)") \
        < i("bound = {k: tunnel[k] for k in BINDING_FIELDS}") \
        < i('completion = publish_completion(sentinels, current_tunnel=read_sentinel_now(sentinels, tkey, log=log), key=key_for(""), bound=bound,') \
        < i("write_stop=write_stop, stop_why=stop_why, legacy_generic=args.legacy_generic_completion, log=log)") \
        < i('(astate / COMPLETION_NAME).write_text(json.dumps(completion["bundle"], indent=1))') \
        < i('stop_written=bool(completion["stop_content"])') < i("completion=completion") \
        < i("write_drive_rec()\n    return drive_rc(verdict, completion)")
    assert i('attempt_id = check_attempt_id(tunnel["attempt_id"])') < i("bound = {k: tunnel[k] for k in BINDING_FIELDS}")
    # the publisher body: the binding check, then the ONE bundle write, then the legacy pair only under the flag
    pub = src.split("def publish_completion(")[1].split("\ndef ")[0]
    assert pub.index("attempt_binding(bound, current_tunnel)") < pub.index("sentinels[ck] = json.dumps(bundle)") < pub.index("if legacy_generic:") \
        < pub.index('sentinels[f"{key}/{VERDICT_NAME}"]') < pub.index('sentinels[f"{key}/{STOP_NAME}"]')
