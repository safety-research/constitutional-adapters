"""qwen-seedrep-native-battery — F-P (blocking) and F-Q, from the smoke run (`DRIVERS_FAILED` 5/5).

F-P: every static family sends through ONE path, ``static_common.run_static`` -> ``defense.generate_detailed(...)``.
Originally that call carried no ``labels=``; the app runs every child with request evidence ON, so the served target refused the
very first request in ``request_evidence_client.require_label`` ("this task request carries no label") and every static
child died within 16 s. The fix labels each request with the store's OWN identity — the canonical row task
``static/<family>`` (``store_task_name(LEG_STATIC, family)``, the string the audit's ``filed_pairs_static`` /
``read_static_store`` key by) and the row's ``(goal_id, sample_idx)`` — and stamps the attempt binding
(``request_id`` / ``completion_id``) on the row the audit's ``bind_rows`` reads. Proved here on the REAL chain, no stub
of the client: ``run_static`` -> the real ``ServedChatTarget`` -> the real ``ProxyClientEngine`` under the real
``RequestEvidenceClient`` -> a loopback fake proxy; then the real ``request_evidence_audit.audit_root`` over the
produced records + the produced store -> AGREES with every draw bound. The same contract on the ladder's two other
detailed surfaces (``escalate_static.regenerate`` through ``rung_redraw._MaterialisedStaticTarget``) and on the BoN
adapter's ``ServedStaticDefense.generate_detailed``.

F-Q: ``seedrep_served.child_log_name`` omitted the goal shard, so the 5 ``--shard k`` children of one static standup
appended to ONE log and overwrote ONE ``.argv.json``; a goal-sharded child now gets ``__shard<k>``, an unsharded child's
stem is unchanged (every existing reader's path holds), and the spawn loop spells the exit event with the encoder too.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "scripts" / "modal", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import audit_common as ac  # noqa: E402
import judge_run_identity as jri  # noqa: E402
import request_evidence_audit as rea  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
import seedrep_launch_plan as lp  # noqa: E402
import test_seedrep_request_evidence as tre  # noqa: E402  (the fake proxy, the real client builder, the ledger-row builder)
from sisteer.capeval.serving import usage_ledger as ul  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402
from sisteer.eval import request_evidence_client as rev  # noqa: E402
from sisteer.eval.attacks import bon_served as bs  # noqa: E402
from sisteer.eval.attacks import served_target as st  # noqa: E402
from sisteer.eval.attacks import static_common as sc  # noqa: E402
from sisteer.eval.attacks.static_common import static_sampling_seed  # noqa: E402

FAMILY = "pasttense"
S = 1                       # the view's literal seed base (root view `s1`)
ARM = "base"                # the fake proxy serves `petri__base`; the fixture root maps arm -> `petri__<arm>`
VMODEL = tre.VMODEL
CAP = 64
PRESET = {k: getattr(tre.VENDOR, k) for k in bs.SAMPLER_KEYS if getattr(tre.VENDOR, k) is not None}
DECODE = {"arm": "vendor_recommended", **PRESET}
TASK = rev.store_task_name(rev.LEG_STATIC, FAMILY)


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
    """The app's three variables for one static child (`seedrep_launch_plan.request_evidence_env`)."""
    monkeypatch.setenv(rev.ENV_DIR, str(out_dir))
    monkeypatch.setenv(rev.ENV_SEED_BASE, str(S))
    monkeypatch.setenv(rev.ENV_EXTRA, json.dumps({"leg": FAMILY, "arm": ARM, "s": S, "attempt_id": attempt_id}))
    rev.reset_env_cache()


def _target(proxy) -> st.ServedChatTarget:
    """The REAL served target over the REAL proxy client with the REAL recorder attached — `build_served_target`'s wiring."""
    client = rev.maybe_attach_proxy_client(tre._client(proxy), unlabelled_kind=rev.PROBE_KIND_BITES)
    return st.ServedChatTarget(client.chat_completions, virtual_model=VMODEL, cell_id=ARM, serving_engine="lora_native")


def _variants(goal, _seed):
    return [("v0", f"{goal.goal_id} v0"), ("v1", f"{goal.goal_id} v1")]


def _goals(n):
    return [SimpleNamespace(goal_id=f"g{i}", category="cat") for i in range(n)]


def _jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _ledger_row_as_the_proxy_writes(raw: bytes, headers, *, completion_id: str) -> dict:
    """One usage-ledger row for one attempt the fake proxy served: the labels + id consumed off the wire, the digest of the
    bytes received, the additive completion_id — the same fields `seedrep_audit_fixtures.Root.send` writes."""
    row = tre._ledger_row(raw, headers)
    body = json.loads(raw.decode("utf-8"))
    row.update(ts="2026-09-18T00:00:00+00:00", prompt_tokens_pre=7, max_tokens_requested=body.get("max_tokens"), has_tools=False,
               n_tools=0, error_code=None, usage_prompt_tokens=7, usage_completion_tokens=2, finish_reason="stop", latency_s=0.5,
               tokenize_mismatch=False, error_class="none")
    row["completion_id"] = completion_id
    return row


# --------------------------------------------------------------------------- #
# F-P: run_static labels every static draw; the real audit accepts the store    #
# --------------------------------------------------------------------------- #

def test_run_static_labels_every_static_draw_and_the_real_audit_agrees(proxy, tmp_path, monkeypatch):
    """The full chain, request evidence ON: `run_static` over the real served target + real client. At ede8eb329 the first
    request died in `require_label`; now every request carries `static/pasttense` + `<goal_id>/<sample_idx>`, every row
    carries its attempt's `request_id` / `completion_id`, and the audit AGREES with every draw bound."""
    root = fx.make_root(tmp_path / "stores", leg=FAMILY, view=f"s{S}", arms=(ARM,))
    _hook_on(monkeypatch, root.path, attempt_id=root.attempt_id)      # the evidence dir IS the attempt root (§R2.2)
    target = _target(proxy)
    # the store lands where a static child writes it: <attempt root>/exp/<private_run>/<family>/<arm>/shard00
    out_dir = root.path / "exp" / f"{ac.MODEL_KEY}_s{S}_sampled_t0" / FAMILY / ARM / "shard00"
    goals = _goals(2)

    res = sc.run_static(target, goals, out_dir=out_dir, family=FAMILY, variants_fn=_variants, trial=0,
                        max_new_tokens=CAP, decode=DECODE, seed_base=S)

    rows = _jsonl(out_dir / "responses.jsonl")
    assert res["n_goals_run"] == 2 and len(rows) == 4 and res["n_goals_errored"] == 0
    posts = proxy.chat_posts()
    assert len(posts) == 4
    # every request on the wire carries the store's canonical task and the row's own item / index
    assert all(tre._hdr(h, C.HEADER_TASK) == TASK == f"static/{FAMILY}" for _r, h in posts)
    assert sorted(tre._hdr(h, C.HEADER_SAMPLE) for _r, h in posts) == sorted(f"{r['goal_id']}/{r['sample_idx']}" for r in rows)
    # the recorder's records name the same (task, item, idx) and the seed the row records; the row binds to its attempt
    records = _jsonl(root.path / C.REQUEST_EVIDENCE_FILENAME)
    responses = _jsonl(root.path / rev.RESPONSE_EVIDENCE_FILENAME)
    assert len(records) == len(responses) == 4 and all(r["kind"] == "task" for r in records)
    by_rid = {r["request_id"]: r for r in records}
    cid_by_rid = {x["request_id"]: x["completion_id"] for x in responses}
    for row in rows:
        rec = by_rid[row["request_id"]]
        assert (rec["task"], rec["item_id"], rec["sample_idx"]) == (TASK, row["goal_id"], row["sample_idx"])
        assert rec["seed"] == row["sampling_seed"] == static_sampling_seed(S, 0, FAMILY, row["goal_id"], row["sample_idx"])
        assert rec["vmodel"] == VMODEL and rec["extra"]["attempt_id"] == root.attempt_id and rec["extra"]["s"] == S
        assert row["completion_id"] == cid_by_rid[row["request_id"]] and row["completion_id"].startswith("cmpl-")
    assert len({r["request_id"] for r in rows}) == 4 == len({r["completion_id"] for r in rows})

    # THE VERDICT: the real audit over the real records + the real store (+ the ledger rows the proxy would have
    # written for those exact bytes, and an engine log with one `Received request` line per completion)
    root.records, root.responses = records, responses
    root.ledger = [_ledger_row_as_the_proxy_writes(raw, h, completion_id=cid_by_rid[tre._hdr(h, C.HEADER_REQUEST_ID)])
                   for raw, h in posts]
    root.engine_lines = [f"INFO [request_logger.py:64] Received request {cid}-{uuid.uuid4().hex[:8]}: params: SamplingParams(n=1)"
                         for cid in cid_by_rid.values()]
    root.flush_pipeline()
    verdicts = rea.audit_root(ac.read_attempt_root(root.path), out_dir=tmp_path / "audits", legs=[FAMILY], policy_file=None,
                              thinking_expect=dict(ac.THINKING_EXPECT_DEFAULT), petri_behaviours=[], model_key=ac.MODEL_KEY,
                              client_status="absent", siblings=[])
    [v] = verdicts
    assert v["verdict"] == "AGREES", v["problem"]
    assert v["checks"]["reconciliation"] is None and v["checks"]["row_binding"]["n_problems"] == 0
    assert v["n_filed"] == 2 and v["n_draws"] == 4 == v["n_draws_bound"] and v["n_records_task"] == 4
    assert v["tasks"] == [TASK] and v["target_vmodel"] == VMODEL and v["s"] == S
    assert "static_sampling_seed" in v["seed_rule"]


def test_run_static_labels_are_the_audits_static_spelling_and_default_off_adds_no_row_key(proxy, monkeypatch, tmp_path):
    """The label's three parts are the audit's: `store_task_name(LEG_STATIC, family)` == `audit_common.static_task(family)`
    == the Draw the store reader builds; `goal_id` as str; `sample_idx` as the row's int. With the hook OFF the labels
    are inert: the request bytes carry no tinycap header and the row's key set is the pre-change one."""
    for fam in rev.STATIC_FAMILIES:
        assert rev.store_task_name(rev.LEG_STATIC, fam) == ac.static_task(fam) == f"static/{fam}"
    target = _target(proxy)     # hook off: maybe_attach_proxy_client is the identity
    out_dir = tmp_path / "off" / FAMILY / ARM / "shard00"
    sc.run_static(target, _goals(1), out_dir=out_dir, family=FAMILY, variants_fn=_variants, trial=0,
                  max_new_tokens=CAP, decode=DECODE, seed_base=S)
    [row0, row1] = _jsonl(out_dir / "responses.jsonl")
    assert "request_id" not in row0 and "completion_id" not in row1
    assert all(tre._hdr(h, C.HEADER_TASK) is None and tre._hdr(h, C.HEADER_SAMPLE) is None for _r, h in proxy.chat_posts())
    # a detailed surface that cannot take labels is not the surface run_static drives (aligned or refused)
    with pytest.raises(st.ServedTargetError, match="labels"):
        target.generate_detailed(["p", "q"], max_new_tokens=8, decode=PRESET, seeds=[1, 2], labels=[(TASK, "g0", 0)])


# --------------------------------------------------------------------------- #
# F-P on the ladder: escalate_static.regenerate through the materialised stand-in #
# --------------------------------------------------------------------------- #

def _load_escalate_static():
    name = "w17_escalate_static"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "sdfbase" / "full22" / "sampled" / "escalate_static.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_ladder_redraws_are_labelled_with_the_rows_identity_and_materialised_hits_send_nothing(proxy, tmp_path, monkeypatch):
    """`escalate_static.regenerate` -> `rung_redraw._MaterialisedStaticTarget.generate_detailed` -> the real target: the
    re-draw of a row is labelled `(static/<family>, goal_id, sample_idx)` from the row itself (the same omission was on this
    path too — `rung_redraw.py:891` delegated with no labels); an item answered from an elected record sends nothing."""
    es = _load_escalate_static()
    RR = es._seedrep_rung_redraw()
    _hook_on(monkeypatch, tmp_path / "ev", attempt_id="ta-w17-ladder")
    target = _target(proxy)
    cap_to = 128
    seed = lambda g, i: static_sampling_seed(S, 0, FAMILY, g, i)  # noqa: E731
    rows = [{"goal_id": "g0", "sample_idx": 1, "family": FAMILY, "attack_prompt": "g0 v1", "sampling_seed": seed("g0", 1)},
            {"goal_id": "g1", "sample_idx": 0, "family": FAMILY, "attack_prompt": "g1 v0", "sampling_seed": seed("g1", 0)}]
    hit = {"text": "materialised", "finish_reason": "stop", "completion_tokens": 3, "prompt_tokens": 1, "max_tokens": cap_to,
           "sampling_seed": seed("g0", 1), "request_id": "rq-elected", "completion_id": "cmpl-elected"}
    summary = {"n_materialised": 0, "n_drawn_by_assembly": 0}
    wrapped = RR._MaterialisedStaticTarget(target, {("g0 v1", seed("g0", 1)): hit}, cap=cap_to, preset=PRESET, summary=summary)

    out = es.regenerate(wrapped, [(0, rows[0]), (1, rows[1])], cap_to=cap_to, preset=PRESET, concurrency=2)

    assert out[0]["text"] == "materialised" and out[0]["request_id"] == "rq-elected"
    assert summary == {"n_materialised": 1, "n_drawn_by_assembly": 1}
    [(raw, h)] = proxy.chat_posts()                                   # ONE send: the un-materialised item, labelled as its row
    assert (tre._hdr(h, C.HEADER_TASK), tre._hdr(h, C.HEADER_SAMPLE)) == (TASK, "g1/0")
    assert json.loads(raw)["seed"] == seed("g1", 0)
    [rec] = _jsonl(tmp_path / "ev" / C.REQUEST_EVIDENCE_FILENAME)
    assert (rec["task"], rec["item_id"], rec["sample_idx"]) == (TASK, "g1", 0)
    assert out[1]["request_id"] == rec["request_id"] and out[1]["completion_id"] == "cmpl-1"
    # a misaligned labels list refuses at the stand-in too
    with pytest.raises(RuntimeError, match="one label per prompt"):
        wrapped.generate_detailed(["a", "b"], max_new_tokens=cap_to, decode=PRESET, seeds=[1, 2], labels=[(TASK, "g", 0)])


# --------------------------------------------------------------------------- #
# F-P on the BoN adapter's static surface                                       #
# --------------------------------------------------------------------------- #

def test_served_static_defense_labels_each_fanned_out_request(proxy, tmp_path, monkeypatch):
    """`ServedStaticDefense.generate_detailed(labels=)` — the other detailed surface `run_static` can drive — sends each
    request under its label (the ContextVar rides into the `to_thread` transport worker) and stamps the binding."""
    _hook_on(monkeypatch, tmp_path / "ev", attempt_id="ta-w17-bon")
    client = rev.maybe_attach_proxy_client(tre._client(proxy))
    loop = __import__("asyncio").new_event_loop()
    try:
        sd = bs.ServedDefense.over_client(client, virtual_model=VMODEL, cell_id=ARM, serving_engine="lora_native")
        d = bs.ServedStaticDefense(sd, loop=loop, max_concurrency=4)
        seeds = [static_sampling_seed(S, 0, FAMILY, "g0", 0), static_sampling_seed(S, 0, FAMILY, "g1", 0)]
        with pytest.raises(bs.BonServedError, match="labels"):
            d.generate_detailed(["a", "b"], max_new_tokens=8, decode=PRESET, seeds=seeds, labels=[(TASK, "g0", 0)])
        recs = d.generate_detailed(["a", "b"], max_new_tokens=8, decode=PRESET, seeds=seeds,
                                   labels=[(TASK, "g0", 0), (TASK, "g1", 0)])
    finally:
        loop.close()
    posts = proxy.chat_posts()
    assert sorted(tre._hdr(h, C.HEADER_SAMPLE) for _r, h in posts) == ["g0/0", "g1/0"]
    assert all(tre._hdr(h, C.HEADER_TASK) == TASK for _r, h in posts)
    records = {r["request_id"]: r for r in _jsonl(tmp_path / "ev" / C.REQUEST_EVIDENCE_FILENAME)}
    for rec, item in zip(recs, ("g0", "g1")):
        assert rec["request_id"] in records and records[rec["request_id"]]["item_id"] == item
        assert rec["completion_id"] and rec["completion_id"].startswith("cmpl-")


# --------------------------------------------------------------------------- #
# F-Q: the child log / argv stem carries the goal shard                          #
# --------------------------------------------------------------------------- #

class _Chain:
    def __getattr__(self, _name):
        return lambda *a, **k: self

    def __call__(self, *a, **k):
        return self


class _App:
    def __init__(self, name=None, **k):
        self.description = None

    def function(self, **kw):
        return lambda f: f

    def local_entrypoint(self, **kw):
        return lambda f: f


def _load_served():
    """`scripts/modal/seedrep_served.py` by path under a stub `modal` (the review-fixes tests' loader, restated)."""
    name = "w17_seedrep_served"
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType("modal")
    m.App = _App
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: _Chain())
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(), debian_slim=lambda *a, **k: _Chain())
    m.Dict = types.SimpleNamespace(from_name=lambda *a, **k: _Chain())
    m.is_local = lambda: False
    m.current_function_call_id = lambda: "fc-test"
    m.forward = None
    m.FunctionCall = None
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "modal" / "seedrep_served.py")
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal")
    sys.modules["modal"] = m
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    sys.modules[name] = mod
    return mod


def test_child_log_name_carries_the_goal_shard_and_every_reader_resolves_each_child(tmp_path):
    srv = _load_served()
    # the static children's shape (`_static_children`: `goal_shard` k of `goal_nshards` n) — 5 shard children of one arm-run
    src = inspect.getsource(lp._static_children)
    assert '"goal_shard": shard' in src and '"goal_nshards": nsh' in src and '"--shard", str(shard), "--nshards", str(nsh)' in src
    kids = [{"eval_leg": FAMILY, "arm": "none", "s": 1, "argv": ["python", "jailbreak_run.py", "generate", "--shard", str(k), "--nshards", "6"],
             "goal_shard": k, "goal_nshards": 6} for k in range(5)]
    names = [srv.child_log_name(k) for k in kids]
    assert names == [f"{FAMILY}__none__s1__shard{k}" for k in range(5)] and len(set(names)) == 5
    # unsharded children keep the original stem: one shard, no shard axis (controls / BoN / misalign / multi-turn), the rung consumer
    assert srv.child_log_name({"eval_leg": FAMILY, "arm": "none", "s": 1, "goal_shard": 0, "goal_nshards": 1}) == f"{FAMILY}__none__s1"
    assert srv.child_log_name({"eval_leg": "controls", "arm": "none", "s": 1}) == "controls__none__s1"
    assert srv.child_log_name({"eval_leg": "x", "arm": "a/b", "s": 1, "goal_shard": 2, "goal_nshards": 3}) == "x__a_b__s1__shard2"
    consumer = {"eval_leg": FAMILY, "arm": srv.RUNG_CONSUMER_ARM, "s": 0}
    assert srv.child_log_name(consumer) == f"{FAMILY}__consumer__s0"
    # the files the spawn writes under `children/` — one log + one argv record PER SHARD CHILD, and every reader resolves each
    root = tmp_path / "attempt"
    (root / "children").mkdir(parents=True)
    for k, name in zip(kids, names):
        (root / "children" / f"{name}.log").write_text(f"shard {k['goal_shard']} log\n")
        (root / "children" / f"{name}.argv.json").write_text(json.dumps({"argv": k["argv"], "env_registry": {lp.RUN_IDENTITY_ENV: None, lp.SPOOL_ENV: None}}))
    assert sorted(p.name for p in (root / "children").glob("*.log")) == sorted(f"{n}.log" for n in names)
    recs = jri.child_records(root)
    assert len(recs) == 5 and sorted(str(r["argv"][-3]) for _p, r in recs) == [str(k) for k in range(5)]
    # the rung tally lookup reads `children/<child_log_name(consumer)>.log` — unchanged stem, still resolves
    (root / "children" / f"{srv.child_log_name(consumer)}.log").write_text("[rung-redraw] x\n" + json.dumps({"drawn": 1, "enqueued_next": 1}) + "\n")
    assert srv.consumer_tally_from_log(root / "children" / f"{srv.child_log_name(consumer)}.log")["enqueued_next"] == 1
    # the spawn loop names the child through the ONE encoder everywhere (the exit event used to restate it inline)
    run_src = inspect.getsource(srv._run_standup_row)
    assert "f\"{k['eval_leg']}__{k['arm']}__s{k['s']}\"" not in run_src
    assert run_src.count("child_log_name(k)") >= 4 and 'f"{name}.log"' in run_src and 'f"{name}.argv.json"' in run_src
