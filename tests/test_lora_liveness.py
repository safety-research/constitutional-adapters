"""scripts/sdfbase/full22/sampled/lora_liveness.py — the LoRA lane's LOW-DOSE NUMERIC LIVENESS record writer.

CPU only, NO server: a fake chat transport stands in for the proxy, answering the writer's two request shapes (greedy
text with the served intervention-hash stamp; a one-token request with first-position top-K logprobs keyed
`token_id:<int>`). The rule under test is the INSTRUMENT'S OWN (`fidelity.check_logprob_steering_bites`, imported from
src, never re-implemented), so what these tests pin is the writer's reading of it:

- `live_numeric` when the intervention hash is distinct AND the steered first-token logprobs clear max(1e-3, 2 x the
  measured repeat noise) — even with the greedy TEXT identical on every probe (the very case the numeric check exists
  for), which is RECORDED as an observation (`would_have_refused: True`) and gates nothing.
- `not_live` on a bit-identical steer, a steer inside the noise, or a served hash that is not the route's.
- `unevaluable` (never a pass) when the legs share no top-K support, when the server ignores
  `return_tokens_as_token_ids`, when the cell is not lens-served or has no scale-0 twin.
- the record binds to THIS standup: `cell` is the launcher's cell id and `routing_table` the resolved path of the
  table the drivers pass, so `low_dose_liveness_policy.consult` admits it for that (cell, table) and nothing else.
- exit codes 0 / 2 / 3 and the launcher's one-line `--summary`.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
WRITER = REPO / "scripts" / "sdfbase" / "full22" / "sampled" / "lora_liveness.py"
POLICY = REPO / "scripts" / "sdfbase" / "full22" / "low_dose_liveness_policy.py"

from sisteer.capeval.serving.fidelity import FidelityError, check_logprob_steering_bites  # noqa: E402
from sisteer.capeval.serving.proxy import RouteEntry  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ll = _load(WRITER, "lora_liveness")
pol = _load(POLICY, "low_dose_liveness_policy")

LENS = "proxy_vllm+lens+interleaved"
PASS = "proxy_vllm+passthrough+interleaved"
#: the sampled ladder's derived cell and its harness route id (registry_adapter.harness_cell_id)
CELL = "constv2-mix-full__lora_r4_ce_minus_ctl"
HID = "si_constv2_mix_full__lora_r4_ce_minus_ctl"
VM, VM_TWIN, VM_ANCHOR = f"petri__{HID}", "petri__none__lens0", "petri__base"
H_ROUTE, H_TWIN, H_ANCHOR = "cc" * 8, "bb" * 8, "aa" * 8


def _routes():
    return [
        RouteEntry(virtual_model=VM_ANCHOR, cell_id="none", cell_class="base", serving_identity=PASS,
                   intervention_hash=H_ANCHOR),
        RouteEntry(virtual_model=VM_TWIN, cell_id="none__lens0", cell_class="control", serving_identity=LENS,
                   intervention_hash=H_TWIN, control_of="none"),
        RouteEntry(virtual_model=VM, cell_id=HID, cell_class="class_a", serving_identity=LENS, intervention_hash=H_ROUTE),
    ]


def _write_table(path: Path, routes=None):
    path.write_text(json.dumps({
        "version": 1, "base_model": "org/model", "serving_mode": "interleaved", "exact_replay_strategy": None,
        "base_revision": "r", "routes": [dataclasses.asdict(r) for r in (routes or _routes())], "prompted_prefixes": []}))
    return path


# --------------------------------------------------------------------------- #
# the fake transport                                                           #
# --------------------------------------------------------------------------- #

BASE_LP = {11: -0.20, 12: -2.10, 13: -3.40, 14: -5.00}      # the twin's first-token top-K (id -> logprob)


def _shift(lp: dict, delta: float, ids=None) -> dict:
    return {i: (v + delta if (ids is None or i in ids) else v) for i, v in lp.items()}


class FakeChat:
    """Per-vmodel greedy text + served hash stamp; per-vmodel first-token logprob tables. `logprobs[vm]` may be a
    callable (called per request) so a leg can carry repeat noise."""

    def __init__(self, texts, logprobs, hashes, *, token_form="token_id:{}"):
        self.texts, self.logprobs, self.hashes, self.token_form = texts, logprobs, hashes, token_form
        self.payloads = []

    def __call__(self, body):
        self.payloads.append(dict(body))
        vm = body["model"]
        out = {"petri_intervention_hash": self.hashes.get(vm)}
        if body.get("logprobs"):
            lps = self.logprobs[vm]
            lps = lps() if callable(lps) else lps
            entries = [{"token": self.token_form.format(i), "logprob": v} for i, v in lps.items()]
            top = max(entries, key=lambda e: e["logprob"])
            out["choices"] = [{"message": {"content": "x"}, "finish_reason": "length",
                               "logprobs": {"content": [{"token": top["token"], "logprob": top["logprob"],
                                                         "top_logprobs": entries}]}}]
        else:
            out["choices"] = [{"message": {"content": self.texts[vm]}, "finish_reason": "stop"}]
        return out


def _ns_routes():
    r = {x.cell_id: SimpleNamespace(**dataclasses.asdict(x)) for x in _routes()}
    return r[HID], r["none"], r["none__lens0"]


def _measure(chat, **over):
    route, anchor, twin = _ns_routes()
    kw = dict(route=route, anchor=anchor, twin=twin, prompts=ll_prompts(), top_k=20, max_tokens=48,
              check_effect=check_logprob_steering_bites, fidelity_error=FidelityError)
    kw.update(over)
    return ll.measure(chat, **kw)


def ll_prompts():
    from sisteer.eval.attacks.served_target import PROBE_PROMPTS
    return PROBE_PROMPTS


def _same_text_chat(steered_lp, twin_lp=None, twin_repeat_lp=None, *, route_hash=H_ROUTE, anchor_hash=H_ANCHOR, **kw):
    """Every leg answers the SAME greedy text (0/5 differing: the low-dose case) — the numeric facts decide."""
    twin_lp = BASE_LP if twin_lp is None else twin_lp
    calls = {"n": 0}

    def twin_side():
        # the first call per probe is `scale0`, the second `scale0_repeat`
        calls["n"] += 1
        return twin_lp if (twin_repeat_lp is None or calls["n"] % 2 == 1) else twin_repeat_lp
    return FakeChat(texts={VM: "same", VM_TWIN: "same", VM_ANCHOR: "same"},
                    logprobs={VM: steered_lp, VM_TWIN: twin_side, VM_ANCHOR: BASE_LP},
                    hashes={VM: route_hash, VM_TWIN: H_TWIN, VM_ANCHOR: anchor_hash}, **kw)


# --------------------------------------------------------------------------- #
# the verdict logic (measure + decide), against the instrument's own rule      #
# --------------------------------------------------------------------------- #

def test_live_numeric_when_the_effect_clears_the_bar_though_every_greedy_text_is_identical():
    chat = _same_text_chat(_shift(BASE_LP, 0.05))
    rec = _measure(chat)
    assert rec["intervention_hash_distinct"] is True
    d = rec["continuous_effect"]["diag"]
    assert d["steered_divergence_median"] == pytest.approx(0.05) and d["noise_ceiling"] == 0.0
    assert d["bar"] == pytest.approx(1e-3)                      # max(abs_floor, 2 x 0 noise)
    assert rec["continuous_effect"]["error"] is None and rec["continuous_effect"]["n_probes"] == 5
    tx = rec["text_divergence_observation"]
    assert tx["n_probes"] == 5 and tx["n_differing"] == 0 and tx["would_have_refused"] is True and tx["error"] is None
    assert len(tx["rows"]) == 5 and all(r["differs"] is False for r in tx["rows"])
    assert rec["facts"] == {"intervention_hash_distinct": True, "serving_time_effect_above_noise": True}
    assert ll.decide(rec) == "live_numeric"
    # the SAME five prompts as the wave's text probe, on every leg of every request
    prompts = {p["messages"][0]["content"] for p in chat.payloads}
    assert prompts == set(ll_prompts())
    # the text legs are the wave probe's request shape; the logprob legs ask for token ids
    text_reqs = [p for p in chat.payloads if not p.get("logprobs")]
    assert all(p["temperature"] == 0.0 and p["max_tokens"] == 48 and p["seed"] == 0 for p in text_reqs)
    lp_reqs = [p for p in chat.payloads if p.get("logprobs")]
    assert all(p["max_tokens"] == 1 and p["top_logprobs"] == 20 and p["return_tokens_as_token_ids"] is True for p in lp_reqs)
    assert len(lp_reqs) == 15                                    # 5 probes x (steered, scale0, scale0_repeat)


def test_not_live_when_the_steer_is_bit_identical_to_the_twin():
    rec = _measure(_same_text_chat(dict(BASE_LP)))
    assert rec["intervention_hash_distinct"] is True
    assert rec["facts"]["serving_time_effect_above_noise"] is False
    assert "BIT-IDENTICAL" in rec["continuous_effect"]["error"]
    assert rec["continuous_effect"]["diag"]["steered_divergence_median"] == 0.0
    assert ll.decide(rec) == "not_live"


def test_not_live_when_the_effect_is_inside_the_engines_own_repeat_noise():
    # repeat noise 0.01 on the twin -> bar = max(1e-3, 2 x 0.01) = 0.02; a 0.015 steer does not clear it
    rec = _measure(_same_text_chat(_shift(BASE_LP, 0.015), twin_repeat_lp=_shift(BASE_LP, 0.01)))
    d = rec["continuous_effect"]["diag"]
    assert d["noise_ceiling"] == pytest.approx(0.01) and d["bar"] == pytest.approx(0.02)
    assert d["steered_divergence_median"] == pytest.approx(0.015)
    assert rec["facts"]["serving_time_effect_above_noise"] is False and "does not clear the bar" in rec["continuous_effect"]["error"]
    assert ll.decide(rec) == "not_live"
    # the same steer clears a quieter engine
    rec2 = _measure(_same_text_chat(_shift(BASE_LP, 0.015), twin_repeat_lp=_shift(BASE_LP, 0.001)))
    assert rec2["continuous_effect"]["diag"]["bar"] == pytest.approx(0.002) and ll.decide(rec2) == "live_numeric"


def test_not_live_when_the_served_hash_is_not_the_routes_or_equals_the_anchors():
    # the proxy stamps a hash that is not the route's spec: a silently unchanged intervention
    rec = _measure(_same_text_chat(_shift(BASE_LP, 0.05), route_hash=H_ANCHOR))
    assert rec["intervention_hash_distinct"] is False
    assert rec["intervention_hash_detail"]["spec_distinct"] is True and rec["intervention_hash_detail"]["served_matches_spec"] is False
    assert rec["facts"]["serving_time_effect_above_noise"] is True         # the effect alone never makes it live
    assert ll.decide(rec) == "not_live"
    # the route's spec hash equal to the anchor's: not distinct by spec
    route, anchor, twin = _ns_routes()
    anchor2 = SimpleNamespace(**{**vars(anchor), "intervention_hash": H_ROUTE})
    rec2 = _measure(_same_text_chat(_shift(BASE_LP, 0.05), anchor_hash=H_ROUTE), anchor=anchor2)
    assert rec2["intervention_hash_detail"]["spec_distinct"] is False and ll.decide(rec2) == "not_live"


def test_text_divergence_is_an_observation_only_and_the_count_is_recorded():
    chat = _same_text_chat(_shift(BASE_LP, 0.05))
    chat.texts[VM] = "different"                                 # 5/5 differing, still the numeric facts decide
    rec = _measure(chat)
    tx = rec["text_divergence_observation"]
    assert tx["n_differing"] == 5 and tx["would_have_refused"] is False and tx["n_steered_ne_anchor"] == 5
    assert ll.decide(rec) == "live_numeric"
    # a bit-identical steer with divergent text (the hypersensitive-text case) is still not live here
    chat2 = _same_text_chat(dict(BASE_LP))
    chat2.texts[VM] = "different"
    assert ll.decide(_measure(chat2)) == "not_live"


def test_unevaluable_never_a_pass_when_the_legs_share_no_support_or_ids_are_not_honoured():
    with pytest.raises(ll.Unevaluable, match="common"):
        _measure(_same_text_chat({91: -0.1, 92: -1.0}))          # disjoint from the twin's ids
    with pytest.raises(ll.Unevaluable, match="return_tokens_as_token_ids"):
        _measure(_same_text_chat(_shift(BASE_LP, 0.05), token_form="tok{}"))
    # the text legs failing takes fact (A)'s served hashes with them: unevaluable, recorded, never a verdict
    chat = _same_text_chat(_shift(BASE_LP, 0.05))
    del chat.texts[VM_ANCHOR]
    with pytest.raises(ll.Unevaluable, match="intervention_hash_distinct"):
        _measure(chat)


def test_decide_is_unevaluable_without_both_facts_and_basis_names_the_failed_check():
    assert ll.decide({}) == "unevaluable" and ll.decide({"facts": {"intervention_hash_distinct": True}}) == "unevaluable"
    rec = {"facts": {"intervention_hash_distinct": True, "serving_time_effect_above_noise": False},
           "realised_dose": {"why": "no artifact"}}
    b = ll.basis_of(rec, "not_live")
    assert "serving_time_effect_above_noise" in b and "intervention_hash_distinct" not in b.split("FAILED:")[1].split("(")[0]
    assert "PROXY" in b and "no artifact" in b
    assert "live, whatever the greedy-text probe says" in ll.basis_of(rec, "live_numeric")
    assert "could not be read" in ll.basis_of(rec, "unevaluable")


@pytest.mark.parametrize("view,want", [("d0p1", 0.1), ("it_d0p1", 0.1), ("d0p25", 0.25), ("it_d1p5", 1.5), ("dm1", -1.0),
                                       ("d1", 1.0), ("it_d1", 1.0), ("main", None), ("it", None), (None, None), ("d", None)])
def test_nominal_dose_is_read_off_the_view_name(view, want):
    assert ll.nominal_dose_of_view(view) == want


# --------------------------------------------------------------------------- #
# main(): the record, its binding to the standup, the exit codes, --summary    #
# --------------------------------------------------------------------------- #

class FakeClient:
    def __init__(self, chat):
        self._chat = chat
        self.loaded_with = None
        self.served_models = ()
        self.proxy_identity = {"impl": "fake"}

    def load(self, *, require_virtual_models=()):
        self.loaded_with = tuple(require_virtual_models)

    def chat_completions(self, body, headers=None):
        self.headers = getattr(self, "headers", []); self.headers.append(dict(headers or {}))   # the kind labels ride the headers
        return self._chat(body)


def _run(tmp_path, chat, *extra, table, cell=CELL, out=None):
    out = out or (tmp_path / f"liveness_low_dose_{cell}.json")
    made = {}

    def factory(url, identity):
        made.update(url=url, identity=identity)
        made["client"] = FakeClient(chat)
        return made["client"]
    rc = ll.main(["--proxy-url", "http://127.0.0.1:9/v1", "--routing-table", str(table), "--cell", cell, "--tag",
                  "llama8b_it_d0p1", "--view", "it_d0p1", "--out", str(out), *extra], client_factory=factory)
    return rc, (json.loads(out.read_text()) if out.exists() else None), made


@pytest.fixture
def standup(tmp_path):
    (tmp_path / "standup").mkdir()
    return _write_table(tmp_path / "standup" / "routing_table.json")


def test_main_writes_a_live_numeric_record_bound_to_this_cell_and_this_standup(tmp_path, standup, capsys):
    rc, rec, made = _run(tmp_path, _same_text_chat(_shift(BASE_LP, 0.05)), table=standup)
    assert rc == 0 and rec["verdict"] == "live_numeric"
    # the binding low_dose_liveness_policy.consult checks: the launcher's cell id + the resolved routing-table path
    assert rec["cell"] == CELL and rec["harness_cell_id"] == HID
    assert rec["routing_table"] == str(standup.resolve())
    out = tmp_path / f"liveness_low_dose_{CELL}.json"
    got, why = pol.consult(CELL, standup, env={pol.POLICY_ENV: str(out)})
    assert got is not None and "ADMITS" in why
    # the SAME record admits nothing for another cell or another standup's table
    assert pol.consult("constv2-mix-full__lora_r4_ce", standup, env={pol.POLICY_ENV: str(out)})[0] is None
    (tmp_path / "other_standup").mkdir()
    other = _write_table(tmp_path / "other_standup" / "routing_table.json")
    assert pol.consult(CELL, other, env={pol.POLICY_ENV: str(out)})[0] is None
    # the record's shape: every field the policy's summary and a later reader need
    for k in ("verdict", "cell", "routing_table", "nominal_dose", "continuous_effect", "intervention_hash_distinct",
              "realised_dose", "written_at", "basis", "text_divergence_observation", "facts", "checks", "schema", "rule",
              "tag", "view", "probe_prompts", "thresholds", "top_logprobs"):
        assert k in rec, k
    assert rec["nominal_dose"] == 0.1 and rec["tag"] == "llama8b_it_d0p1" and rec["view"] == "it_d0p1"
    d = rec["continuous_effect"]["diag"]
    assert set(d) >= {"steered_divergence_median", "noise_ceiling", "bar", "n_items", "per_item"}
    assert rec["intervention_hash_distinct"] is True
    assert rec["realised_dose"]["available"] is False and "OFFLINE" in rec["realised_dose"]["why"]
    assert "LoRA" in rec["realised_dose"]["why"] and rec["realised_dose"]["effect_size_proxy"] == pytest.approx(0.05)
    assert "PROXY" in rec["basis"] and rec["facts"]["fact_b_basis"].startswith("the serving-time effect")
    assert rec["thresholds"] == {"min_ratio": 2.0, "abs_floor": 1e-3,
                                 "rule_source": "sisteer.capeval.serving.fidelity.check_logprob_steering_bites (the instrument's own)"}
    assert rec["probe_prompts"] == list(ll_prompts()) and rec["top_logprobs"] == 20 and "engine default" in rec["top_logprobs_source"]
    assert rec["twin"] == "none__lens0" and rec["virtual_model"] == VM and rec["thinking_on"] is False
    # the policy's summary reads the record it was written for
    s = pol.summary(rec)
    assert s["record_verdict"] == "live_numeric" and s["steered_divergence_median"] == pytest.approx(0.05)
    assert s["text_divergence_observation"] == {"n_probes": 5, "n_differing": 0} and s["realised_dose"] is False
    assert s["nominal_dose"] == 0.1 and s["tag"] == "llama8b_it_d0p1"
    # the client was built for the route's identity and asked to serve all three legs
    assert made["identity"] == LENS and set(made["client"].loaded_with) == {VM, VM_TWIN, VM_ANCHOR}
    assert "VERDICT live_numeric" in capsys.readouterr().out


def test_main_exit_codes_not_live_and_unevaluable(tmp_path, standup):
    rc, rec, _ = _run(tmp_path, _same_text_chat(dict(BASE_LP)), table=standup)
    assert rc == 2 and rec["verdict"] == "not_live" and "FAILED" in rec["basis"]
    assert pol.consult(CELL, standup, env={pol.POLICY_ENV: str(tmp_path / f"liveness_low_dose_{CELL}.json")})[0] is None
    # no common support -> unevaluable, exit 3, the record says which check
    rc, rec, _ = _run(tmp_path, _same_text_chat({91: -0.1, 92: -1.0}), table=standup)
    assert rc == 3 and rec["verdict"] == "unevaluable"
    assert rec["checks"][-1]["check"] == "measure" and "common" in rec["checks"][-1]["error"]
    # a non-lens cell has no twin to measure against: unevaluable, never a pass
    rc, rec, _ = _run(tmp_path, _same_text_chat(_shift(BASE_LP, 0.05)), table=standup, cell="none",
                      out=tmp_path / "none.json")
    assert rc == 3 and rec["verdict"] == "unevaluable" and rec["checks"][-1]["check"] == "routes"
    assert "not lens-served" in rec["checks"][-1]["error"]
    # a lens cell whose table carries no scale-0 twin
    no_twin = _write_table(tmp_path / "no_twin.json", [r for r in _routes() if r.cell_id != "none__lens0"])
    rc, rec, _ = _run(tmp_path, _same_text_chat(_shift(BASE_LP, 0.05)), table=no_twin, out=tmp_path / "nt.json")
    assert rc == 3 and "no scale-0 twin" in rec["checks"][-1]["error"]
    # a transport failure is not a verdict
    def boom(body):
        raise ConnectionError("proxy down")
    rc, rec, _ = _run(tmp_path, boom, table=standup, out=tmp_path / "down.json")
    assert rc == 3 and rec["verdict"] == "unevaluable" and "ConnectionError" in rec["checks"][-1]["error"]


def test_main_reads_the_standups_logprob_cap_from_endpoint_json_and_honours_thinking_on(tmp_path, standup):
    (standup.parent / "endpoint.json").write_text(json.dumps({"max_logprobs": 64, "routing_table": str(standup)}))
    chat = _same_text_chat(_shift(BASE_LP, 0.05))
    rc, rec, _ = _run(tmp_path, chat, "--thinking-on", table=standup)
    assert rc == 0 and rec["top_logprobs"] == 64 and rec["top_logprobs_source"].startswith("endpoint.json")
    assert all(p["top_logprobs"] == 64 for p in chat.payloads if p.get("logprobs"))
    # thinking ON: every LENS-route request declares it; the bare anchor never does (the smoke's rule)
    from sisteer.capeval.serving.proxy import THINKING_ON_KEY
    for p in chat.payloads:
        assert (p.get(THINKING_ON_KEY) is True) == (p["model"] in (VM, VM_TWIN)), p["model"]
    assert rec["thinking_on"] is True
    # an explicit --top-logprobs wins over the stamp
    rc, rec, _ = _run(tmp_path, _same_text_chat(_shift(BASE_LP, 0.05)), "--top-logprobs", "8", table=standup)
    assert rec["top_logprobs"] == 8 and rec["top_logprobs_source"] == "--top-logprobs"


def test_summary_is_the_launchers_one_line(tmp_path, standup, capsys):
    rc, rec, _ = _run(tmp_path, _same_text_chat(_shift(BASE_LP, 0.05)), table=standup)
    out = tmp_path / f"liveness_low_dose_{CELL}.json"
    capsys.readouterr()
    assert ll.main(["--summary", str(out)]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("live_numeric median=0.05 bar=0.001 n_differing=0/5 hash_distinct=True")
    assert "noise=0" in line and "nominal_dose=0.1" in line and "\n" not in line
    rec["verdict"] = "not_live"
    out.write_text(json.dumps(rec))
    ll.main(["--summary", str(out)])
    assert capsys.readouterr().out.startswith("not_live median=0.05 bar=0.001 n_differing=0/5")
    # an unevaluable record still summarises (None fields, `?` for an unread text probe)
    (tmp_path / "u.json").write_text(json.dumps({"verdict": "unevaluable", "text_divergence_observation": {"error": "x"}}))
    ll.main(["--summary", str(tmp_path / "u.json")])
    assert capsys.readouterr().out.strip() == "unevaluable median=None bar=None n_differing=?/? hash_distinct=None noise=None nominal_dose=None"


def test_main_requires_the_binding_arguments():
    with pytest.raises(SystemExit):
        ll.main(["--proxy-url", "http://x/v1", "--cell", CELL])          # no --routing-table / --out
    assert ll.POLICY_ENV == pol.POLICY_ENV == "SISTEER_LOW_DOSE_LIVENESS_RECORD"
    assert ll.EXIT == {"live_numeric": 0, "not_live": 2, "unevaluable": 3}


def test_the_greedy_text_probes_are_labelled_probe_bites_and_recorded_beside_the_record_while_the_numeric_probes_carry_no_kind(tmp_path, standup):
    """The text probes are the bites shape (served_target.probe_fn, greedy at PROBE_MAX_TOKENS) and were the one untagged
    emitter in the 8192 rung's resolved-sampler re-read; now every one carries X-Tinycap-Task: probe/bites (sample numbered per route) and the
    record names the sidecar beside it (<out stem>.request_evidence.jsonl, the file the sampled lane's reader call hands to the reader); the
    1-token logprob probes carry no kind label."""
    from sisteer.eval.attacks.served_target import PROBE_PROMPTS
    chat = _same_text_chat(_shift(BASE_LP, 0.05))
    rc, rec, made = _run(tmp_path, chat, table=standup)
    assert rc == 0 and rec["verdict"] == "live_numeric"
    client = made["client"]
    assert len(client.headers) == len(chat.payloads) > 0
    text = [(p, h) for p, h in zip(chat.payloads, client.headers) if not p.get("logprobs")]
    numeric = [(p, h) for p, h in zip(chat.payloads, client.headers) if p.get("logprobs")]
    assert text and numeric
    assert all(h["X-Tinycap-Task"] == "probe/bites" and h["X-Tinycap-Sample"].startswith(f"{p['model']}/") and h["X-Tinycap-Request-Id"] for p, h in text)
    assert all(p["temperature"] == 0.0 and p["max_tokens"] == text[0][0]["max_tokens"] for p, _ in text)
    assert sorted(int(h["X-Tinycap-Sample"].rsplit("/", 1)[1]) for p, h in text if p["model"] == text[0][0]["model"]) == list(range(len(PROBE_PROMPTS)))
    assert all(h == {} for _, h in numeric)
    out = tmp_path / f"liveness_low_dose_{CELL}.json"
    assert rec["request_evidence"] == str(out.with_name(f"liveness_low_dose_{CELL}.request_evidence.jsonl")) and rec["kind_tag"] == "probe/bites"
    src = (REPO / "scripts/sdfbase/full22/sampled/lora_liveness.py").read_text()
    assert "on_request=evidence.on_request" in src and 'evidence.labelled(chat_h, "bites")' in src and src.count("greedy_text(chat_text or chat, ") == 3
