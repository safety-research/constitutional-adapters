"""qwen-seedrep-native-battery — F-S, from the S3 BoN smoke (`bon__constv2-mix-full__lora_r4_ce__s1`).

The BoN child (`scripts/sdfbase/full22/bon_served_run.py`) died at its FIRST send — not a draw but the steering-bites
PROBE: `bon_generate_served` -> `assert_steering_bites` -> `served_target.probe_fn._greedy` -> the attached client's
`chat_completions` -> `RequestEvidenceClient.headers_for`, which refused "an unlabelled target request under request
evidence … (a probe call site declares unlabelled_kind instead)". `ServedDefense.over_client` attached the child's one
client WITHOUT `unlabelled_kind`, while `build_served_target` (the static / multi-turn wiring) attaches its client with
`probe/bites` — so the same probe that passes there was refused here. And the draws behind the probe were unlabelled
too: the child hard-coded `request_headers_fn = None` under a hook that was ON, so the first draw would have been
refused next.

The fix, proved here on the REAL chain (no stub of the client): the child attaches its client ONCE with
`unlabelled_kind=PROBE_KIND_BITES` before `over_client` (whose own attach is then the idempotent no-op) and plugs
`rev.maybe_request_headers_fn(LEG_BON)` — the labels are the audit's own (`strongreject_bon`, `<goal_id>/<sample_idx>`)
with a fresh id per attempt — into `bon_generate_served`; `ServedDefense._send` refuses an unlabelled draw by name (so a
draw can never be recorded as a probe once the client carries the probe kind); and the manifest's `units.n_done` (the
key `read_bon_store` compares to the stored rows) is written, without which the audit REFUSED every real BoN
store as "the run did not finish". Then the real `request_evidence_audit.audit_root` over the produced records +
the produced store -> AGREES with every draw bound. The original wiring is reproduced verbatim as the known-bad.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "scripts" / "modal", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
import test_bon_served as tbs  # noqa: E402  (the canonical route set `assert_cell_salt_isolation` runs over)
import test_seedrep_request_evidence as tre  # noqa: E402  (the fake proxy, the real client builder)
import test_seedrep_w17_static_labels as w17  # noqa: E402  (the ledger-row-as-the-proxy-writes builder)
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.eval import request_evidence_client as rev  # noqa: E402
from sisteer.eval.attacks import bon  # noqa: E402
from sisteer.eval.attacks import bon_served as bs  # noqa: E402
from sisteer.eval.attacks import served_target as st  # noqa: E402

S = 1                       # the view's literal seed base (root view `s1`)
ARM = "base"                # the fake proxy serves `petri__base`; the fixture root maps arm -> `petri__<arm>`
VMODEL, TWIN = tre.VMODEL, tre.TWIN
CAP = 64
N_MAX = 2
PRESET = {k: getattr(tre.VENDOR, k) for k in bs.SAMPLER_KEYS if getattr(tre.VENDOR, k) is not None}
TASK = rev.store_task_name(rev.LEG_BON)
PROBE_TASK = C.probe_task_label(C.PROBE_KIND_BITES)
W16_ERROR = ("an unlabelled target request under request evidence: no X-Tinycap-Task / X-Tinycap-Sample on the call and "
             "no `labelled(task, item_id, idx)` context — the driver that knows the item must label the send (a probe call "
             "site declares unlabelled_kind instead)")


@pytest.fixture(autouse=True)
def _hook_off(monkeypatch):
    monkeypatch.delenv(rev.ENV_DIR, raising=False)
    monkeypatch.delenv(rev.ENV_EXTRA, raising=False)
    monkeypatch.delenv(rev.ENV_SEED_BASE, raising=False)
    rev.reset_env_cache()
    yield
    rev.reset_env_cache()


@pytest.fixture()
def proxy():
    p = tre._FakeProxy()
    try:
        yield p
    finally:
        p.close()


def _hook_on(monkeypatch, out_dir: Path, *, attempt_id: str) -> None:
    """The app's three variables for one BoN child (`seedrep_launch_plan.request_evidence_env`)."""
    monkeypatch.setenv(rev.ENV_DIR, str(out_dir))
    monkeypatch.setenv(rev.ENV_SEED_BASE, str(S))
    monkeypatch.setenv(rev.ENV_EXTRA, json.dumps({"leg": "bon", "arm": ARM, "s": S, "attempt_id": attempt_id}))
    rev.reset_env_cache()


def _goals(n):
    return [SimpleNamespace(goal_id=f"g{i}", text=f"goal {i} text", category="cat") for i in range(n)]


def _child_wiring(proxy):
    """`bon_served_run.main`'s client / defense / probe wiring after F-S, on the real `ProxyClientEngine` over the loopback
    proxy: ONE client, attached with the probe kind BEFORE `over_client`, the probe over its `chat_completions`."""
    client = rev.maybe_attach_proxy_client(tre._client(proxy), unlabelled_kind=rev.PROBE_KIND_BITES)
    defense = bs.ServedDefense.over_client(client, virtual_model=VMODEL, cell_id=ARM, serving_engine="lora_native")
    probe = st.probe_fn(client.chat_completions, VMODEL, TWIN)
    return client, defense, probe


# --------------------------------------------------------------------------- #
# The known-bad: the old wiring under the hook, verbatim                        #
# --------------------------------------------------------------------------- #

def test_w16_wiring_refuses_the_bites_probe_verbatim_and_an_unhooked_draw_is_refused_not_recorded_as_a_probe(
        proxy, tmp_path, monkeypatch):
    """At 79185d481 the child's client was attached by `over_client` alone (no `unlabelled_kind`) and its probe ran over
    that client: the FIRST probe send raised the error verbatim, nothing reached the wire. With the probe kind on the
    client (the fix), an unhooked draw (`request_headers_fn=None`, the original literal) is refused by name at the defense —
    never silently recorded as a `probe/bites` attempt."""
    _hook_on(monkeypatch, tmp_path / "ev", attempt_id="ta-fs-w16")
    client = tre._client(proxy)
    bs.ServedDefense.over_client(client, virtual_model=VMODEL, cell_id=ARM, serving_engine="lora_native")   # the known-bad: attaches, no kind
    probe = st.probe_fn(client.chat_completions, VMODEL, TWIN)
    with pytest.raises(rev.RequestEvidenceError) as ei:
        bs.assert_steering_bites(("r4",), probe)
    assert str(ei.value) == W16_ERROR
    assert proxy.chat_posts() == []
    # the fix's client: the probe passes as probe/bites; an UNLABELLED draw is refused before the wire (F-S guard)
    client2, defense, probe2 = _child_wiring(proxy)
    diag = bs.assert_steering_bites(("r4",), probe2)
    assert "r4" in diag["arms"]
    n_probe_posts = 2 * len(st.PROBE_PROMPTS)
    posts = proxy.chat_posts()
    assert len(posts) == n_probe_posts and all(tre._hdr(h, C.HEADER_TASK) == PROBE_TASK for _r, h in posts)
    with pytest.raises(rev.RequestEvidenceError, match="no label"):
        asyncio.run(defense.generate_one_detailed("variant", sampling_seed=bon.bon_decode_seed_v1("g0", 0, 0, S),
                                                  max_new_tokens=CAP, decode=PRESET))
    assert len(proxy.chat_posts()) == n_probe_posts                                        # nothing more on the wire
    records = w17._jsonl(tmp_path / "ev" / C.REQUEST_EVIDENCE_FILENAME)
    assert len(records) == n_probe_posts and all(r["kind"] == "probe" and r["task"] == PROBE_TASK for r in records)


# --------------------------------------------------------------------------- #
# F-S: the child's wiring labels the probe AND every draw; the real audit agrees #
# --------------------------------------------------------------------------- #

def test_bon_child_wiring_labels_probe_and_draws_and_the_real_audit_agrees(proxy, tmp_path, monkeypatch):
    """The full chain, request evidence ON, with F-S: `bon_generate_served` over the child's real client / defense / probe.
    The probe's sends are `probe/bites` + `<vmodel>/<n>`; every draw carries `strongreject_bon` + `<goal_id>/<sample_idx>`
    and its own fresh id; every row carries its attempt's `request_id` / `completion_id`; the manifest's `units.n_done`
    is the stored row count; and the audit AGREES with every draw bound."""
    root = fx.make_root(tmp_path / "stores", leg="bon", view=f"s{S}", arms=(ARM,))
    _hook_on(monkeypatch, root.path, attempt_id=root.attempt_id)      # the evidence dir IS the attempt root (§R2.2)
    client, defense, probe = _child_wiring(proxy)
    routes = tbs._routes(tmp_path)
    # the store lands where a BoN child writes it: <attempt root>/exp/<private_run>/<arm>/bon_responses.jsonl
    out_dir = root.path / "exp" / f"{ac.MODEL_KEY}_s{S}_sampled_t0" / ARM
    goals = _goals(2)

    jsonl = bs.bon_generate_served(
        defense, goals, out_dir=out_dir, routes=routes, n_max=N_MAX, max_new_tokens=CAP, max_concurrency=4,
        steering_arms=("r4",), steering_probe=probe, decode=PRESET,
        decode_seed_policy="replicate_v1", decode_seed_base=S,
        request_headers_fn=rev.maybe_request_headers_fn(rev.LEG_BON))

    rows = w17._jsonl(jsonl)
    n_draws = len(goals) * N_MAX
    n_probe = 2 * len(st.PROBE_PROMPTS)
    assert len(rows) == n_draws
    posts = proxy.chat_posts()
    assert len(posts) == n_probe + n_draws
    probe_posts, draw_posts = posts[:n_probe], posts[n_probe:]
    # the probe: probe/bites, <vmodel>/<n> per route, n from 0
    assert all(tre._hdr(h, C.HEADER_TASK) == PROBE_TASK for _r, h in probe_posts)
    assert sorted(tre._hdr(h, C.HEADER_SAMPLE) for _r, h in probe_posts) == sorted(
        f"{vm}/{n}" for vm in (VMODEL, TWIN) for n in range(len(st.PROBE_PROMPTS)))
    # every draw on the wire carries the store's canonical task and the row's own item / index, one fresh id each
    assert all(tre._hdr(h, C.HEADER_TASK) == TASK == "strongreject_bon" for _r, h in draw_posts)
    assert sorted(tre._hdr(h, C.HEADER_SAMPLE) for _r, h in draw_posts) == sorted(f"{r['goal_id']}/{r['sample_idx']}" for r in rows)
    assert len({tre._hdr(h, C.HEADER_REQUEST_ID) for _r, h in posts}) == len(posts)
    # the recorder's records name the same (task, item, idx) and the seed the row records; the row binds to its attempt
    records = w17._jsonl(root.path / C.REQUEST_EVIDENCE_FILENAME)
    responses = w17._jsonl(root.path / rev.RESPONSE_EVIDENCE_FILENAME)
    assert len(records) == len(responses) == n_probe + n_draws
    assert sum(1 for r in records if r["kind"] == "task") == n_draws
    assert sum(1 for r in records if r["kind"] == "probe") == n_probe
    by_rid = {r["request_id"]: r for r in records}
    cid_by_rid = {x["request_id"]: x["completion_id"] for x in responses}
    for row in rows:
        rec = by_rid[row["request_id"]]
        assert (rec["kind"], rec["task"], rec["item_id"], rec["sample_idx"]) == ("task", TASK, row["goal_id"], row["sample_idx"])
        assert rec["seed"] == row["sampling_seed"] == bon.bon_decode_seed_v1(row["goal_id"], row["sample_idx"], 0, S)
        assert rec["vmodel"] == VMODEL and rec["extra"]["attempt_id"] == root.attempt_id and rec["extra"]["s"] == S
        assert row["completion_id"] == cid_by_rid[row["request_id"]] and row["completion_id"].startswith("cmpl-")
        assert row["decode_seed_policy"] == "replicate_v1" and row["decode_seed_base"] == S
    assert len({r["request_id"] for r in rows}) == n_draws == len({r["completion_id"] for r in rows})
    # the manifest states the planned workload and that every planned slot is covered (the audit's completeness keys)
    manifest = json.loads((out_dir / bs.BON_SERVED_MANIFEST).read_text())
    u = manifest["units"]
    assert (u["n_planned"], u["n_done"], u["n_uncovered"], u["n_work"], u["n_written"]) == (n_draws, n_draws, 0, n_draws, n_draws)
    assert manifest["virtual_model"] == VMODEL

    # THE VERDICT: the real audit over the real records + the real store (+ the ledger rows the proxy would have
    # written for those exact bytes — the probes' too, their own partition — and one `Received request` per completion)
    root.records, root.responses = records, responses
    root.ledger = [w17._ledger_row_as_the_proxy_writes(raw, h, completion_id=cid_by_rid[tre._hdr(h, C.HEADER_REQUEST_ID)])
                   for raw, h in posts]
    root.engine_lines = [f"INFO [request_logger.py:64] Received request {cid}-{uuid.uuid4().hex[:8]}: params: SamplingParams(n=1)"
                         for cid in cid_by_rid.values()]
    root.flush_pipeline()
    verdicts = rea.audit_root(ac.read_attempt_root(root.path), out_dir=tmp_path / "audits", legs=["bon"], policy_file=None,
                              thinking_expect=dict(ac.THINKING_EXPECT_DEFAULT), petri_behaviours=[], model_key=ac.MODEL_KEY,
                              client_status="absent", siblings=[])
    [v] = verdicts
    assert v["verdict"] == "AGREES", v["problem"]
    assert v["checks"]["reconciliation"] is None and v["checks"]["row_binding"]["n_problems"] == 0
    assert v["n_filed"] == len(goals) and v["n_draws"] == n_draws == v["n_draws_bound"] and v["n_records_task"] == n_draws
    assert v["tasks"] == [TASK] and v["target_vmodel"] == VMODEL and v["s"] == S and v["store_complete"] is True
    assert "bon_decode_seed_v1" in v["seed_rule"]

    # a resume over the finished store sends nothing and keeps n_done at the stored count
    bs.bon_generate_served(
        defense, goals, out_dir=out_dir, routes=routes, n_max=N_MAX, max_new_tokens=CAP, max_concurrency=4,
        steering_arms=("r4",), steering_probe=probe, decode=PRESET, decode_seed_policy="replicate_v1", decode_seed_base=S,
        request_headers_fn=rev.maybe_request_headers_fn(rev.LEG_BON))
    assert len(proxy.chat_posts()) == 2 * n_probe + n_draws       # the second probe only; no draw re-sent
    u2 = json.loads((out_dir / bs.BON_SERVED_MANIFEST).read_text())["units"]
    assert (u2["n_planned"], u2["n_done"], u2["n_uncovered"], u2["n_work"], u2["n_written"]) == (n_draws, n_draws, 0, 0, 0)
    assert ac.read_bon_store(out_dir, arm=ARM).complete is True


def test_hook_off_path_is_byte_identical_except_the_additive_n_done(proxy, tmp_path):
    """With the hook OFF the child's wiring is inert: `maybe_attach_proxy_client` is the identity, `maybe_request_headers_fn`
    is None, no tinycap header rides any request, no row carries a binding key — and the manifest gains `units.n_done`."""
    assert rev.maybe_request_headers_fn(rev.LEG_BON) is None
    client, defense, probe = _child_wiring(proxy)
    assert client.chat_completions.__name__ != "chat_completions" or not hasattr(client, "_sisteer_request_evidence")
    out_dir = tmp_path / "off" / ARM
    jsonl = bs.bon_generate_served(
        defense, _goals(1), out_dir=out_dir, routes=tbs._routes(tmp_path), n_max=N_MAX, max_new_tokens=CAP, max_concurrency=2,
        steering_arms=("r4",), steering_probe=probe, decode=PRESET, decode_seed_policy="replicate_v1", decode_seed_base=S,
        request_headers_fn=rev.maybe_request_headers_fn(rev.LEG_BON))
    rows = w17._jsonl(jsonl)
    assert len(rows) == N_MAX and all("request_id" not in r and "completion_id" not in r for r in rows)
    assert all(tre._hdr(h, C.HEADER_TASK) is None and tre._hdr(h, C.HEADER_SAMPLE) is None for _r, h in proxy.chat_posts())
    u = json.loads((out_dir / bs.BON_SERVED_MANIFEST).read_text())["units"]
    assert (u["n_planned"], u["n_done"], u["n_uncovered"]) == (N_MAX, N_MAX, 0)


# --------------------------------------------------------------------------- #
# completeness means every PLANNED slot is filled                              #
# --------------------------------------------------------------------------- #

def _units(out_dir: Path) -> dict:
    return json.loads((out_dir / bs.BON_SERVED_MANIFEST).read_text())["units"]


def test_an_overflowed_slot_reads_incomplete_and_a_successful_resume_completes_the_store(tmp_path, monkeypatch):
    """`PromptOverWindow` is caught inside the generation loop: the slot writes NO row (a typed error row in the errors
    ledger instead) and the function returns normally. At W19a `units.n_done` was the surviving-row count, so
    `n_done == len(rows)` held by construction and the reader called the store COMPLETE. Now the manifest states the
    planned workload: 2 planned, 1 ok + 1 over-window -> `n_planned 2, n_done 1, n_uncovered 1` and the reader says
    INCOMPLETE; a resume at a wider window fills the slot -> `n_done 2 == n_planned` and the store is complete."""
    import test_served_window_clamp as twc
    monkeypatch.setattr(bs, "augment", lambda text, sigma, rng: text)                 # keep the LEN= marker
    monkeypatch.setattr(bs, "assert_cell_salt_isolation", lambda routes: None)
    goals = [SimpleNamespace(goal_id="g_fit", text="LEN=100 fits", category="c"),
             SimpleNamespace(goal_id="g_over", text="LEN=10230 over", category="c")]
    out_dir = tmp_path / "out" / ARM
    kw = dict(out_dir=out_dir, routes=[], n_max=1, sigma=0.0, max_new_tokens=512, max_concurrency=4, decode=PRESET,
              decode_seed_policy="replicate_v1", decode_seed_base=S)

    sd = bs.ServedDefense(twc._WindowedAchat(window=10240), virtual_model=VMODEL, cell_id=ARM, serving_engine="lora_native",
                          max_model_len=10240)
    jsonl = bs.bon_generate_served(sd, goals, **kw)
    rows = w17._jsonl(jsonl)
    assert [r["goal_id"] for r in rows] == ["g_fit"]
    u = _units(out_dir)
    assert (u["n_planned"], u["n_done"], u["n_uncovered"], u["n_work"], u["n_written"], u["n_prompt_over_window"]) == (2, 1, 1, 2, 1, 1)
    assert u["n_done"] == len(rows)                     # the W19a trap: the surviving-row identity holds and proves nothing
    store = ac.read_bon_store(out_dir, arm=ARM)
    assert store.complete is False and store.n_rows == 1
    assert "1 of 2 planned (goal_id, sample_idx) slot(s) uncovered" in store.completeness_note and "1 prompt_over_window" in store.completeness_note
    (erow,) = w17._jsonl(out_dir / bs.BON_ERRORS_FILE)
    assert (erow["goal_id"], erow["error"]) == ("g_over", "prompt_over_window")

    # the successful resume: a wider window, the uncovered slot is the only work, the store becomes complete
    sd2 = bs.ServedDefense(twc._WindowedAchat(window=20480), virtual_model=VMODEL, cell_id=ARM, serving_engine="lora_native",
                           max_model_len=20480)
    bs.bon_generate_served(sd2, goals, **kw)
    rows2 = w17._jsonl(jsonl)
    assert sorted(r["goal_id"] for r in rows2) == ["g_fit", "g_over"]
    u2 = _units(out_dir)
    assert (u2["n_planned"], u2["n_done"], u2["n_uncovered"], u2["n_work"], u2["n_written"], u2["n_prompt_over_window"]) == (2, 2, 0, 1, 1, 0)
    store2 = ac.read_bon_store(out_dir, arm=ARM)
    assert store2.complete is True and store2.n_rows == 2
    assert len(w17._jsonl(out_dir / bs.BON_ERRORS_FILE)) == 1            # the first run's typed error stays on the ledger (history)

    # the reader refuses the explicit unfinished / overflow states by hand too: a manifest whose rows equal n_done but
    # fall short of the plan; the older shape (n_done only) / a run that raised (no units keys); a rows-vs-n_done disagreement
    mp = out_dir / bs.BON_SERVED_MANIFEST
    good = json.loads(mp.read_text())
    for units, needle in (({**good["units"], "n_planned": 3}, "1 of 3 planned"),
                          ({"n_done": 2, "n_prompt_over_window": 0}, "did not finish"),
                          ({}, "did not finish"),
                          ({**good["units"], "n_done": 3, "n_planned": 3}, "2 stored BoN row(s) != units.n_done 3")):
        mp.write_text(json.dumps({**good, "units": units}))
        st_ = ac.read_bon_store(out_dir, arm=ARM)
        assert st_.complete is False and needle in st_.completeness_note, (units, st_.completeness_note)
    mp.write_text(json.dumps(good))
    assert ac.read_bon_store(out_dir, arm=ARM).complete is True


# --------------------------------------------------------------------------- #
# The child scripts spell the wiring the chain test proved                       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("script, n_attach", [("bon_served_run.py", 1), ("bon_climb_run.py", 2)])
def test_child_scripts_attach_the_probe_kind_before_over_client_and_plug_the_bon_headers_seam(script, n_attach):
    src = (REPO / "scripts" / "sdfbase" / "full22" / script).read_text(encoding="utf-8")
    attach = re.compile(r"maybe_attach_proxy_client\(client, unlabelled_kind=_?rev\.PROBE_KIND_BITES\)")
    assert len(attach.findall(src)) == n_attach
    # every attach precedes the over_client that would otherwise attach without the kind
    for m_over in re.finditer(r"bs\.ServedDefense\.over_client\(client", src):
        assert any(m.end() < m_over.start() for m in attach.finditer(src))
    assert "request_headers_fn = None" not in src
    assert re.search(r"request_headers_fn=_?rev\.maybe_request_headers_fn\(_?rev\.LEG_BON\)|request_headers_fn = rev\.maybe_request_headers_fn\(rev\.LEG_BON\)", src)
