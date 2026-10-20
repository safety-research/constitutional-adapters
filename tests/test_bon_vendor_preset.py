"""The BoN `vendor` decode preset for the Qwen seed-replicate native battery.

The FINAL BoN rule (2026-09-17): best-of-N draws under the Qwen vendor instruct row — a NEW instrument,
never pooled with the T = 1.0 rows. The driver therefore gains `--bon-decode-preset vendor`, which is THE REGISTRY ROW
`qwen35_4b_it|bon|off|vendor_preset@v1` of `--model-key` (REQUIRED), resolved by `PolicyFile.by_ref` on the EXACT ref
with `superseded_by_policy_id is None` asserted and all six sampling keys SENT (`target_decode.registry_vendor_row`) —
never `resolve(mode=)` (ignores eval_id; returns the IFEval row), never `find()`, never a default lookup, never the
sampled-decode yaml. Pinned here, against a FIXTURE policy file (the three rows verbatim from e8fd4582 spliced into a copy
of the committed export — the rows are not on this branch's export yet; the gate checks the real registry):

  1. the exact ref string, the six values (presence_penalty 1.5 SENT), the policy record;
  2. KNOWN-BAD rows refuse and nothing is invented: no --model-key, an absent row (NoPolicy), a SUPERSEDED row, a row
     that leaves presence_penalty to the server, a row in another mode; a preset draw with no seed refuses;
  3. the known-WRONG lookup (`resolve(model, eval strongreject_bon, mode vendor_preset)`) returns the IFEval row, and the
     driver's source consults neither it nor `find()` nor the yaml;
  4. the WIRE BODIES: every draw's request carries EXACTLY {model, messages, temperature, max_tokens, seed, top_p, top_k,
     min_p, presence_penalty, repetition_penalty} with the row's values, `max_tokens 4096`, and the seed
     `bon_decode_seed_v1(goal, sample, trial, base=s)` (the literal s); the AUGMENTATION is the frozen bon_seed's
     (identical variant bytes to a bon_seed_v1 run); rows and the manifest record the six keys + the row's identity;
     three bases -> three seed sets, one item set;
  5. the driver's argv reaches the fan-out with the row, `n_max 75`, cap 4096, no shard flags (BoN is UNSHARDED);
  6. the request-evidence seam: default off (rows byte-identical), and when a request_headers_fn is given the
     headers reach the transport and every row carries request_id + completion_id; a defense with no header transport
     refuses.
Everything synthetic (recording transports, invented goals, no server); the fixture rows are the registry's own bytes."""
from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving.proxy_engine import ledger_headers
from sisteer.capeval.tinycap import contracts as tc
from sisteer.eval.attacks import bon as bon_mod
from sisteer.eval.attacks import bon_served as bs
from sisteer.eval.attacks import target_decode as td
from sisteer.registry import decode_policy as dp

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "scripts" / "sdfbase" / "full22", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import bon_served_run as br  # noqa: E402
from test_review_p1_4_multiturn_overflow import _ctx400  # noqa: E402  (the production vLLM context-length 400, one spelling)

FIXTURE = REPO / "tests" / "fixtures" / "seedrep" / "decode_policy_vendor_rows.json"
#: the EXACT refs (impl brief "Registry facts"): resolved by by_ref only, version included
BON_REF = "qwen35_4b_it|bon|off|vendor_preset@v1"
MT_REF = "qwen35_4b_it|multi_turn|off|vendor_preset@v1"
PETRI_REF = "qwen35_4b_it|petri|off|vendor_preset@v1"
GRANDFATHERED = "qwen35_4b_it|bon|off|grandfathered_T1@v1"
IFEVAL_ROW = "qwen35_4b_it|ifeval|off|vendor_preset@v1"
#: the Qwen instruct vendor row — the six numbers every TARGET request SENDS (presence 1.5 is not server-defaultable)
VENDOR_SIX = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 1.5}
#: the EXACT key set of a vendor best-of-N request body
BODY_KEYS = frozenset({"model", "messages", "temperature", "max_tokens", "seed",
                       "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"})
#: the harness-default body (today's, unchanged): temperature 1.0 and nothing else beside the seed
HARNESS_BODY_KEYS = frozenset({"model", "messages", "temperature", "max_tokens", "seed"})
S1_CONFIG = "configs/sdfbase_full22_qwen35_4b_s1.yaml"


def policy_file_with_vendor_rows(tmp_path, *, superseded=(), drop=(), edit=None, name="decode_policy.json") -> str:
    """A copy of the COMMITTED export with the fixture's three vendor rows spliced in (verbatim, e8fd4582; skipped for
    a row the export already carries, so this keeps working once they land) and the grandfathered BoN row
    superseded. Known-bad knobs: ``superseded`` = policy_ids to mark superseded, ``drop`` =
    policy_ids to leave out (absent), ``edit`` = {policy_id: {field: value}} (a torn row shape). The body sha is
    re-pinned so the file is a well-formed export. The base is the COMMITTED export (the package copy, else the repo
    copy) — never ``default_export_path()``, which honours ``$SISTEER_DECODE_POLICY`` and would splice into whatever
    fixture the calling test last pointed the env at."""
    import os
    base = dp.PACKAGE_EXPORT_PATH if os.path.isfile(dp.PACKAGE_EXPORT_PATH) else dp.repo_export_path()
    data = json.loads(Path(base).read_text())
    fx = json.loads(FIXTURE.read_text())
    have = {p["policy_id"] for p in data["policies"]}
    for row in fx["policies"]:
        if row["policy_id"] not in have:
            data["policies"].append(json.loads(json.dumps(row)))
    for pid, patch in fx["supersede"].items():
        for p in data["policies"]:
            if p["policy_id"] == pid:
                p.update(patch)
    data["policies"] = [p for p in data["policies"] if p["policy_id"] not in set(drop)]
    for p in data["policies"]:
        if p["policy_id"] in set(superseded):
            p["superseded_by_policy_id"] = "synthetic|successor|off|vendor_preset@v9"
        if edit and p["policy_id"] in edit:
            p.update(edit[p["policy_id"]])
    data["header"]["policy_set_sha256"] = dp.policy_set_sha256(data)
    out = tmp_path / name
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return str(out)


def _goals(n=2):
    import test_bon_served as tb
    return tb._goals(n)


def _routes(d):
    """The served-BoN tests' real route set, built in ``d`` (the world writes its synthetic artifacts there)."""
    import test_bon_served as tb
    Path(d).mkdir(parents=True, exist_ok=True)
    return tb._routes(Path(d))


class _RecordingChat:
    """An async transport that records the EXACT client body (nothing proxy-owned added) and answers with a
    completion whose text and id encode the request."""

    def __init__(self, *, prompt_tokens=1):
        self.payloads = []
        self.headers = []
        self.prompt_tokens = prompt_tokens

    async def __call__(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        user = payload["messages"][-1]["content"]
        return {"id": f"cmpl-{len(self.payloads):04d}", "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": f"reply::{user}"}}],
                "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": 1, "total_tokens": self.prompt_tokens + 1}}

    async def with_headers(self, payload, headers):
        self.headers.append(dict(headers))
        return await self(payload)


def _rows(out):
    return [json.loads(l) for l in (out / "bon_responses.jsonl").read_text().splitlines() if l.strip()]


def _by_unit(rows):
    return {(r["goal_id"], r["sample_idx"], r["trial"]): r for r in rows}


def _bodies_by_unit(chat, rows):
    by_variant = {r["variant"]: (r["goal_id"], r["sample_idx"], r["trial"]) for r in rows}
    return {by_variant[p["messages"][-1]["content"]]: p for p in chat.payloads}


def _run(tmp_path, name, *, decode, decode_label, chat=None, **kw):
    routes = _routes(tmp_path / f"routes_{name}")
    chat = chat or _RecordingChat()
    defense = bs.ServedDefense(chat, virtual_model="petri__base", cell_id="none", serving_engine="lens",
                               achat_with_headers=chat.with_headers)
    out = tmp_path / name
    bs.bon_generate_served(defense, _goals(2), out_dir=out, routes=routes, n_max=3, sigma=0.4, trial=0,
                           max_new_tokens=4096, max_concurrency=4, decode=decode, decode_label=decode_label, **kw)
    rows = _rows(out)
    return out, rows, _bodies_by_unit(chat, rows), json.loads((out / bs.BON_SERVED_MANIFEST).read_text()), chat


# --------------------------------------------------------------------------- #
# 1. the exact ref, the six values, the record                                 #
# --------------------------------------------------------------------------- #


def test_the_exact_ref_resolves_the_six_keys_with_presence_sent(tmp_path):
    path = policy_file_with_vendor_rows(tmp_path, name="good.json")
    assert td.vendor_preset_policy_ref("qwen35_4b_it", "bon") == BON_REF
    assert td.vendor_preset_policy_ref("qwen35_4b_it", "multi_turn") == MT_REF
    assert br.BON_VENDOR_POLICY_SEGMENT == "bon" and td.VENDOR_PRESET_POLICY_VERSION == 1
    row = td.registry_vendor_row("qwen35_4b", "bon", path=path)
    assert row.ref == BON_REF and row.policy.policy_id == BON_REF and row.policy.policy_ref == "qwen35_4b_it|bon|off|vendor_preset"
    assert row.decode == VENDOR_SIX                                    # values AND the exact key set
    assert set(row.decode) == set(dp.PARAM_KEYS) == set(bs.SAMPLER_KEYS) == set(td.VENDOR_ROW_SENT_KEYS)
    assert row.decode["presence_penalty"] == 1.5 and row.policy.superseded_by_policy_id is None
    assert row.policy.mode == "vendor_preset" and row.policy.decode_source == "model_card" and row.policy.seed_rule == "per_draw"
    rec = row.record()
    assert rec["policy_id"] == BON_REF and rec["policy_version"] == 1 and rec["model_id"] == "qwen35_4b_it"
    assert rec["superseded_by_policy_id"] is None and rec["policy_file"] == path
    assert rec["policy_set_sha256"] == json.loads(Path(path).read_text())["header"]["policy_set_sha256"]
    # every spelling of the served model lands on the same row (the export's model_spellings)
    for spelling in ("qwen35_4b", "qwen35_4b_it", "Qwen/Qwen3.5-4B"):
        assert td.registry_vendor_row(spelling, "bon", path=path).ref == BON_REF
    # the driver's resolver: `vendor` -> the row's six keys + its record; the two literal presets carry NO record
    decode, drec = br.resolve_bon_decode("vendor", "qwen35_4b", path=path)
    assert decode == VENDOR_SIX and drec["policy_id"] == BON_REF
    assert bs.sampler_preset_problem(decode) is None and set(bs.sampler_preset(decode)) == set(bs.SAMPLER_KEYS)
    assert bs._sent_sampler(decode) == VENDOR_SIX and bs._preset_label(decode, "vendor") == "vendor"
    assert br.resolve_bon_decode("harness_default", None, path=path) == (None, None)
    assert br.resolve_bon_decode("anchor_match", "qwen35_4b", path=path) == (br.BON_DECODE_PRESETS["anchor_match"], None)
    assert br.BON_DECODE_PRESETS["vendor"] == br.BON_DECODE_FROM_REGISTRY == "registry"     # a marker, never numbers
    # main's two REGISTRY labels (`vendor_preset` the default, `harness_default` the superseded T=1 row by name) lead the
    # table since the 2026-09-17 rule; the battery's exact-ref `vendor` marker is layered beside them (cut-time union)
    # + `canonical` (2026-10-09): the label of the canonical best-of-N instrument the planner binds for the served model, resolved at launch
    assert tuple(br.BON_DECODE_PRESETS) == ("vendor_preset", "harness_default", "anchor_match", "vendor", "canonical")
    # the same fixture carries the multi-turn and Petri rows the sibling workstreams resolve (caps 4096 / 1024)
    mt = td.registry_vendor_row("qwen35_4b", "multi_turn", path=path)
    assert mt.decode == VENDOR_SIX and mt.policy.max_new_tokens_rule == "4096 per reply"
    assert td.cap_in_max_new_tokens_rule(mt.policy.max_new_tokens_rule) == 4096
    assert td.cap_in_max_new_tokens_rule(td.registry_vendor_row("qwen35_4b", "petri", path=path).policy.max_new_tokens_rule) == 1024
    assert td.cap_in_max_new_tokens_rule(row.policy.max_new_tokens_rule) is None      # BoN's rule names the instrument's caps
    assert td.cap_in_max_new_tokens_rule("512 store + 4096 climb") is None and td.cap_in_max_new_tokens_rule(None) is None


# --------------------------------------------------------------------------- #
# 2. known-bad rows refuse; nothing is invented                                #
# --------------------------------------------------------------------------- #


def test_known_bad_rows_refuse_and_nothing_is_invented(tmp_path):
    good = policy_file_with_vendor_rows(tmp_path, name="good.json")
    # vendor without the served model's key: the row is keyed by the model and never guessed
    for key in (None, ""):
        with pytest.raises(SystemExit, match="requires --model-key"):
            br.resolve_bon_decode("vendor", key, path=good)
    with pytest.raises(SystemExit, match="not one of"):
        br.resolve_bon_decode("nope", "qwen35_4b", path=good)
    # ABSENT: NoPolicy -> refused, naming the exact ref
    absent = policy_file_with_vendor_rows(tmp_path, drop=[BON_REF], name="absent.json")
    with pytest.raises(td.VendorPolicyRefused, match=r"has no row 'qwen35_4b_it\|bon\|off\|vendor_preset@v1'"):
        td.registry_vendor_row("qwen35_4b", "bon", path=absent)
    with pytest.raises(SystemExit, match="has no row"):
        br.resolve_bon_decode("vendor", "qwen35_4b", path=absent)
    # SUPERSEDED: by_ref still returns it (the flag rides the Policy) -> the resolver refuses
    sup = policy_file_with_vendor_rows(tmp_path, superseded=[BON_REF], name="superseded.json")
    assert dp.PolicyFile.load(sup).by_ref(BON_REF).superseded_by_policy_id is not None
    with pytest.raises(td.VendorPolicyRefused, match="SUPERSEDED"):
        td.registry_vendor_row("qwen35_4b", "bon", path=sup)
    with pytest.raises(SystemExit, match="SUPERSEDED"):
        br.resolve_bon_decode("vendor", "qwen35_4b", path=sup)
    # a row that leaves presence_penalty to the server is NOT the vendor row (not server-defaultable)
    five = ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed"]
    partial = policy_file_with_vendor_rows(tmp_path, edit={BON_REF: {"params_sent": five}}, name="partial.json")
    with pytest.raises(td.VendorPolicyRefused, match=r"does not SEND \['presence_penalty'\]"):
        td.registry_vendor_row("qwen35_4b", "bon", path=partial)
    nulled = policy_file_with_vendor_rows(tmp_path, edit={BON_REF: {"presence_penalty": None}}, name="nulled.json")
    with pytest.raises(td.VendorPolicyRefused, match=r"does not SEND \['presence_penalty'\]"):
        td.registry_vendor_row("qwen35_4b", "bon", path=nulled)
    wrong_mode = policy_file_with_vendor_rows(tmp_path, edit={BON_REF: {"mode": "server_default"}}, name="mode.json")
    with pytest.raises(td.VendorPolicyRefused, match="is mode 'server_default'"):
        td.registry_vendor_row("qwen35_4b", "bon", path=wrong_mode)
    notnum = policy_file_with_vendor_rows(tmp_path, edit={BON_REF: {"top_k": True}}, name="bool.json")
    with pytest.raises(td.VendorPolicyRefused, match="not a number"):
        td.registry_vendor_row("qwen35_4b", "bon", path=notnum)
    # a ref typed for another segment / model is not this leg's row
    with pytest.raises(td.VendorPolicyRefused, match="is not the served model's bon vendor row"):
        td.registry_vendor_row("qwen35_4b", "bon", path=good, expect_ref=MT_REF)
    # the COMMITTED export as it stands: absent today (refused, never invented) — or, once W lands the rows, the same six keys
    try:
        live = td.registry_vendor_row("qwen35_4b", "bon")
    except td.VendorPolicyRefused as e:
        assert "has no row" in str(e)
    else:
        assert live.decode == VENDOR_SIX and live.ref == BON_REF
    # a preset draw with no seed refuses at the request builder (the BoN-side twin of served_target._payload's rule)
    d = bs.ServedDefense(_RecordingChat(), virtual_model="petri__base", cell_id="none", serving_engine="lens")
    # (main's _request takes the sampler as `decode=` alone — the temperature rides inside the block, 2026-09-17 ruling)
    with pytest.raises(bs.BonServedError, match="per-draw sampling_seed"):
        d._request("v", sampling_seed=None, max_new_tokens=4096, decode=VENDOR_SIX)
    assert d._request("v", sampling_seed=7, max_new_tokens=4096, decode=VENDOR_SIX)["seed"] == 7
    # the seed BASE missing under replicate_v1 refuses before any write
    routes = _routes(tmp_path / "r")
    with pytest.raises(bs.BonServedError, match="requires a non-negative int decode_seed_base"):
        bs.bon_generate_served(d, _goals(1), out_dir=tmp_path / "nobase", routes=routes, n_max=1, max_concurrency=1,
                               decode=VENDOR_SIX, decode_label="vendor", decode_seed_policy="replicate_v1", decode_seed_base=None)
    assert not (tmp_path / "nobase" / "bon_responses.jsonl").exists()


# --------------------------------------------------------------------------- #
# 3. the known-WRONG lookup, never consulted                                   #
# --------------------------------------------------------------------------- #


def test_the_known_wrong_lookup_returns_the_ifeval_row_and_the_driver_never_consults_it(tmp_path):
    pf = dp.PolicyFile.load(policy_file_with_vendor_rows(tmp_path, name="kw.json"))
    wrong = pf.resolve("qwen35_4b", eval_id="strongreject_bon", mode="vendor_preset")
    assert wrong.policy_id == IFEVAL_ROW and wrong.eval_id == "ifeval" and wrong.policy_id != BON_REF   # mode= ignores eval_id
    assert pf.by_ref(BON_REF).policy_id == BON_REF                                                     # the exact ref is the authority
    # the driver reads the registry through ONE function, which reads by_ref ALONE
    run_src = (REPO / "scripts/sdfbase/full22/bon_served_run.py").read_text()
    assert "registry_vendor_row(model_key, BON_VENDOR_POLICY_SEGMENT" in run_src
    assert run_src.count("resolve(") == run_src.count("__file__).resolve(") == 1           # the path bootstrap, nothing else
    for needle in (".find(", "model_independent(", "pf.resolve(", ".policy(", "sampled_decode", "sdfbase_sampled_decode", "resolve_default_decode"):
        assert needle not in run_src, needle
    td_src = (REPO / "src/sisteer/eval/attacks/target_decode.py").read_text()
    body = td_src[td_src.index("def registry_vendor_row("):]
    assert "pf.by_ref(ref)" in body and "pf.resolve(" not in body and ".find(" not in body and "model_independent(" not in body
    # the row is resolved at CALL time (never at import: the module binds no sisteer name at import, and the ref needs the key)
    head = run_src[:run_src.index("def _served_engine_problem")]
    assert "from sisteer" not in head and "import sisteer" not in head


# --------------------------------------------------------------------------- #
# 4. the wire bodies                                                           #
# --------------------------------------------------------------------------- #


def test_every_draw_carries_the_six_keys_the_literal_seed_rule_and_the_frozen_augmentation(tmp_path):
    decode, rec = br.resolve_bon_decode("vendor", "qwen35_4b", path=policy_file_with_vendor_rows(tmp_path, name="wire.json"))
    out1, rows1, bodies1, man1, _ = _run(tmp_path, "vendor_s1", decode=decode, decode_label="vendor",
                                         decode_seed_policy="replicate_v1", decode_seed_base=1, decode_policy=rec)
    assert len(rows1) == 6 and len(bodies1) == 6
    goals = {g.goal_id: g for g in _goals(2)}
    for (gid, idx, trial), body in bodies1.items():
        assert frozenset(body) == BODY_KEYS, sorted(body)                     # EXACTLY the ten keys, nothing proxy-owned
        for k, v in VENDOR_SIX.items():
            assert body[k] == v, (k, body[k])
        assert body["presence_penalty"] == 1.5 and body["max_tokens"] == 4096 and body["model"] == "petri__base"
        # the DECODE seed is the replicate rule over the LITERAL s; the augmentation seed is the frozen bon_seed
        assert body["seed"] == bon_mod.bon_decode_seed_v1(gid, idx, trial, 1) != bon_mod.bon_seed(gid, idx, trial)
        assert body["messages"] == [{"role": "user", "content": bon_mod.augment(goals[gid].text, 0.4, random.Random(bon_mod.bon_seed(gid, idx, trial)))}]
    units = _by_unit(rows1)
    for (gid, idx, trial), r in units.items():
        assert r["decode"] == VENDOR_SIX and r["bon_decode_preset"] == "vendor"
        assert r["sampling_seed"] == bodies1[(gid, idx, trial)]["seed"] and r["seed_rule"] == "bon_seed_v1+decode:replicate_v1"
        assert r["decode_seed_policy"] == "replicate_v1" and r["decode_seed_base"] == 1 and r["augment_seed"] == bon_mod.bon_seed(gid, idx, trial)
        assert "request_id" not in r and "completion_id" not in r          # the id seam is OFF: no extra keys
    assert man1["decode"] == VENDOR_SIX and man1["bon_decode_preset"] == "vendor" and man1["decode_seed_base"] == 1
    assert man1["decode_policy"]["policy_id"] == BON_REF and man1["decode_policy"]["superseded_by_policy_id"] is None
    assert man1["seed_rule"] == "bon_seed_v1+decode:replicate_v1" and man1["units"]["n_written"] == 6
    # THREE bases -> three seed sets, ONE item set (the augmentation bytes never move)
    _, rows2, bodies2, _, _ = _run(tmp_path, "vendor_s2", decode=decode, decode_label="vendor",
                                   decode_seed_policy="replicate_v1", decode_seed_base=2, decode_policy=rec)
    assert {u: r["variant"] for u, r in _by_unit(rows2).items()} == {u: r["variant"] for u, r in units.items()}
    assert all(bodies2[u]["seed"] != bodies1[u]["seed"] for u in units)
    assert all(bodies2[u]["seed"] == bon_mod.bon_decode_seed_v1(*u, 2) for u in units)
    # the harness default is UNCHANGED: temperature 1.0 and nothing else, the frozen seed as the decode seed, same variants
    # main's rule (2026-09-17): best-of-N has no model-independent default label — the grandfathered T=1 row is NAMED
    # (bs.BON_HARNESS_DEFAULT), the idiom test_bon_decode_seed_policy.py adopted at the reseed landing
    _, rows0, bodies0, man0, _ = _run(tmp_path, "harness", decode=None, decode_label=bs.BON_HARNESS_DEFAULT)
    assert {u: r["variant"] for u, r in _by_unit(rows0).items()} == {u: r["variant"] for u, r in units.items()}
    for u, body in bodies0.items():
        assert frozenset(body) == HARNESS_BODY_KEYS and body["temperature"] == 1.0 and body["seed"] == bon_mod.bon_seed(*u)
    assert all(r["decode"] == {"temperature": 1.0} and r["bon_decode_preset"] == "harness_default" for r in rows0)
    assert "decode_policy" not in man0                                     # no record under the literal presets
    # a resume under the same row and base accepts every row; under another base refuses before any write
    out1b, rows1b, _, _, _ = _run(tmp_path, "vendor_s1", decode=decode, decode_label="vendor",
                                  decode_seed_policy="replicate_v1", decode_seed_base=1, decode_policy=rec)
    assert out1b == out1 and len(rows1b) == 6
    with pytest.raises(bs.BonServedError, match="decode_seed_base 1 and this run asks for 3"):
        _run(tmp_path, "vendor_s1", decode=decode, decode_label="vendor",
             decode_seed_policy="replicate_v1", decode_seed_base=3, decode_policy=rec)


# --------------------------------------------------------------------------- #
# 5. the driver's argv -> the fan-out (75 draws, cap 4096, unsharded)          #
# --------------------------------------------------------------------------- #


class _StubClient:
    made = []

    def __init__(self, base_url, *, identity, **kw):
        _StubClient.made.append((base_url, identity))
        self.proxy_identity = None

    def load(self, *, require_virtual_models=()):
        self.loaded = list(require_virtual_models)
        self.proxy_identity = {"engine": {"kind": "lora_native", "adapters": {"none__lora0": {}}}, "thinking_mode": "off"}

    def chat_completions(self, payload, *, headers=None):
        raise AssertionError("the fan-out is stubbed; no request may be issued here")


def _argv(tmp_path, rt, **over):
    base = {"--config": S1_CONFIG, "--defense": "none", "--private-root": str(tmp_path / "state"),
            "--proxy-url": "http://127.0.0.1:1/v1", "--routing-table": str(rt), "--max-model-len": "11520",
            "--max-concurrency": "9", "--require-engine": "lora_native", "--bon-decode-preset": "vendor",
            "--model-key": "qwen35_4b", "--decode-seed-policy": "replicate_v1", "--decode-seed-base": "1"}
    base.update(over)
    argv = ["--confirmatory"]
    for k, v in base.items():
        if v is not None:
            argv += [k, v]
    return argv


def _wire_main(monkeypatch, tmp_path):
    from sisteer.capeval.serving import proxy as px
    import jailbreak_run as jr
    # the s1 config's explicit_cells validate against STAGED artifacts (an eval registry); a worktree has none, so the id
    # set is stubbed the way every served-driver e2e test stubs it — the driver's own loader still reads the real config
    monkeypatch.setattr(jr, "_valid_defense_ids", lambda cfg: {"none", *cfg["explicit_cells"]})
    routes = _routes(tmp_path / "routes")
    rt = tmp_path / "rt.json"
    rt.write_text("{}")
    monkeypatch.setattr(px, "load_routing_table", lambda p: {"routes": routes})
    monkeypatch.setattr(px, "ProxyClientEngine", _StubClient)
    _StubClient.made = []
    import two_si_run as tsr
    monkeypatch.setattr(tsr, "_eval_goals", lambda cfg, *, confirmatory, attack=None: _goals(2))
    got = {}

    def fake_fanout(defense, goals, **kw):
        got.update(kw, n_goals=len(goals), defense=defense)
        p = Path(kw["out_dir"]) / "bon_responses.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
        # f4dc4b8e1 (BoN 512 store completeness): the driver certifies the store from the manifest's `units` slot accounting BEFORE it
        # writes the completion marker (bon_climb.store_completion over store_units + store_slot_classification) and exits 3 on an
        # undecided plan. The real fanout records these in bon_served_manifest.json (write_bon_served_manifest); this stub records a
        # fully covered plan — every (goal, sample) slot decided by a row, no no-fits — so main reaches the marker the test reads.
        n_planned = len(goals) * int(kw["n_max"])
        (Path(kw["out_dir"]) / bs.BON_SERVED_MANIFEST).write_text(json.dumps(
            {"units": {"n_planned": n_planned, "n_done": n_planned, "n_nofit": 0, "n_undecided": 0}}))
        return p
    monkeypatch.setattr(bs, "bon_generate_served", fake_fanout)
    return rt, got


def test_main_argv_reaches_the_fanout_with_the_registry_row_75_draws_cap_4096_unsharded(monkeypatch, tmp_path):
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, name="main.json"))
    rt, got = _wire_main(monkeypatch, tmp_path)
    assert br.main(_argv(tmp_path, rt)) == 0
    assert got["decode"] == VENDOR_SIX and got["decode_label"] == "vendor"
    assert got["decode_seed_policy"] == "replicate_v1" and got["decode_seed_base"] == 1 and got["trial"] == 0
    assert got["n_max"] == 75 and got["max_new_tokens"] == 4096 and got["max_concurrency"] == 9 and got["n_goals"] == 2
    assert got["decode_policy"]["policy_id"] == BON_REF and got["request_headers_fn"] is None
    assert got["steering_arms"] == () and got["steering_probe"] is None                # `none` carries no arm
    assert got["defense"].serving_engine == "lora_native" and got["defense"].max_model_len == 11520
    rec = json.loads((Path(got["out_dir"]) / "bon_served_run.json").read_text())
    assert rec["decode"] == VENDOR_SIX and rec["bon_decode_preset"] == "vendor"
    assert rec["decode_policy"]["policy_id"] == BON_REF and rec["decode_policy"]["policy_file"].endswith("main.json")
    assert rec["decode_seed_policy"] == "replicate_v1" and rec["decode_seed_base"] == 1
    assert rec["seed_rule"] == "bon_seed_v1+decode:replicate_v1" and rec["n_max"] == 75 and rec["smoke"] is False
    assert rec["serving_engine"] == "lora_native" and rec["max_model_len"] == 11520 and rec["n_goals"] == 2
    # UNSHARDED: the driver has no goal partition — the head-cap is a smoke, never a shard
    dests = {a.dest for a in br.build_parser()._actions}
    assert not ({"shard", "nshards", "shards"} & dests) and "limit_goals" in dests
    # the two literal presets write NO decode_policy key (run records byte-identical)
    got.clear()
    assert br.main(_argv(tmp_path / "h", rt, **{"--bon-decode-preset": "harness_default", "--decode-seed-policy": None,
                                                 "--decode-seed-base": None})) == 0
    rec0 = json.loads((Path(got["out_dir"]) / "bon_served_run.json").read_text())
    assert "decode_policy" not in rec0 and got["decode"] is None and got["decode_policy"] is None
    assert rec0["decode"] == {"temperature": 1.0} and rec0["bon_decode_preset"] == "harness_default"


def test_main_refuses_a_bad_vendor_row_before_the_proxy_is_touched(monkeypatch, tmp_path):
    rt, got = _wire_main(monkeypatch, tmp_path)
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, name="ok.json"))
    with pytest.raises(SystemExit, match="requires --model-key"):
        br.main(_argv(tmp_path, rt, **{"--model-key": None}))
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, superseded=[BON_REF], name="sup.json"))
    with pytest.raises(SystemExit, match="SUPERSEDED"):
        br.main(_argv(tmp_path, rt))
    monkeypatch.setenv(dp.ENV_VAR, policy_file_with_vendor_rows(tmp_path, drop=[BON_REF], name="absent.json"))
    with pytest.raises(SystemExit, match="has no row"):
        br.main(_argv(tmp_path, rt))
    assert _StubClient.made == [] and not got                        # refused BEFORE the client existed or a byte landed
    assert not list((tmp_path / "state").rglob("*.json")) if (tmp_path / "state").exists() else True


# --------------------------------------------------------------------------- #
# 6. the request-evidence seam                                                 #
# --------------------------------------------------------------------------- #


def _seam_headers(goal_id, sample_idx, trial, *, minted, per_attempt=False):
    """ONE attempt's headers as a request-evidence hook mints them: the two tinycap labels in their production spelling
    (proxy_engine.ledger_headers -> contracts.sample_header_value, the format the gate parses) + a request id the test can
    recognise (``per_attempt`` suffixes the attempt ordinal, so a retry's id differs); ``minted`` records every id handed out."""
    rid = f"rid-{goal_id}-{sample_idx}-{trial}" + (f"-attempt{len(minted)}" if per_attempt else "")
    minted.append(rid)
    return {**ledger_headers("strongreject_bon", goal_id, sample_idx), tc.HEADER_REQUEST_ID: rid}


def test_the_request_evidence_seam_is_off_by_default_and_binds_rows_when_armed(tmp_path):
    decode, rec = br.resolve_bon_decode("vendor", "qwen35_4b", path=policy_file_with_vendor_rows(tmp_path, name="seam.json"))
    minted = []
    headers_for = lambda g, i, t: _seam_headers(g, i, t, minted=minted)  # noqa: E731

    out, rows, bodies, man, chat = _run(tmp_path, "armed", decode=decode, decode_label="vendor",
                                        decode_seed_policy="replicate_v1", decode_seed_base=1, decode_policy=rec,
                                        request_headers_fn=headers_for)
    assert len(chat.headers) == 6 == len(chat.payloads) and sorted(minted) == sorted(h[tc.HEADER_REQUEST_ID] for h in chat.headers)
    for r in rows:
        assert r["request_id"] == f"rid-{r['goal_id']}-{r['sample_idx']}-{r['trial']}"
        assert r["completion_id"].startswith("cmpl-")
    assert len({r["completion_id"] for r in rows}) == 6                      # one completion per attempt
    # the bodies are the same ten keys (the headers ride BESIDE the body, never in it)
    assert all(frozenset(b) == BODY_KEYS for b in bodies.values())
    # armed but no header-carrying transport: refused, never sent un-labelled
    bare = bs.ServedDefense(_RecordingChat(), virtual_model="petri__base", cell_id="none", serving_engine="lens")
    with pytest.raises(bs.BonServedError, match="header-carrying transport"):
        bs.bon_generate_served(bare, _goals(1), out_dir=tmp_path / "bare", routes=_routes(tmp_path / "rb"), n_max=1,
                               max_concurrency=1, decode=decode, decode_label="vendor", request_headers_fn=headers_for)
    # a hook that mints no request id is a caller bug, refused
    chat2 = _RecordingChat()
    d2 = bs.ServedDefense(chat2, virtual_model="petri__base", cell_id="none", serving_engine="lens", achat_with_headers=chat2.with_headers)
    with pytest.raises(bs.BonServedError, match="returned no X-Tinycap-Request-Id"):
        bs.bon_generate_served(d2, _goals(1), out_dir=tmp_path / "norid", routes=_routes(tmp_path / "rn"), n_max=1,
                               max_concurrency=1, decode=decode, decode_label="vendor",
                               request_headers_fn=lambda g, i, t: {tc.HEADER_TASK: "strongreject_bon"})
    # over_client provides the header transport from the client's own header-carrying call
    class _Client:
        def __init__(self):
            self.calls = []
            self.proxy_identity = None

        def chat_completions(self, payload, *, headers=None):
            self.calls.append((json.loads(json.dumps(payload)), None if headers is None else dict(headers)))
            return {"id": "cmpl-x", "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}], "usage": {}}
    import asyncio
    c = _Client()
    d3 = bs.ServedDefense.over_client(c, virtual_model="petri__base", cell_id="none", serving_engine="lens")
    t = asyncio.run(d3.generate_one_detailed("v", sampling_seed=1, max_new_tokens=8, temperature=0.7, decode=decode,
                                             attempt_headers_fn=lambda: {tc.HEADER_REQUEST_ID: "r1"}))
    t0 = asyncio.run(d3.generate_one_detailed("v", sampling_seed=1, max_new_tokens=8, temperature=0.7, decode=decode))
    assert c.calls[0][1] == {tc.HEADER_REQUEST_ID: "r1"} and c.calls[1][1] is None     # headers only when handed over
    assert t.completion_id == "cmpl-x" == t0.completion_id and c.calls[0][0] == c.calls[1][0]
    assert t.request_id == "r1" and t0.request_id is None                              # the attempt's id rides the turn
    rec_t = bs.turn_record(t, sampling_seed=1)
    assert "completion_id" not in rec_t and "request_id" not in rec_t                 # turn_record's bytes unchanged


class _ClampOnceChat(_RecordingChat):
    """A proxy stand-in for the window clamp: a request whose ``max_tokens`` exceeds the room the window leaves is the
    production context-length 400 (``_ctx400``); the clamped retry (``max_tokens`` <= room) is answered by the parent.
    Records EVERY body (rejected ones too) and every header set, in send order."""

    def __init__(self, *, window: int, prompt: int):
        super().__init__(prompt_tokens=prompt)
        self.window = window

    async def __call__(self, payload):
        requested = int(payload["max_tokens"])
        if requested > self.window - self.prompt_tokens:
            self.payloads.append(json.loads(json.dumps(payload)))            # the rejected body is recorded too
            raise _ctx400(self.window, self.prompt_tokens, requested)
        return await super().__call__(payload)


def test_a_context_window_retry_mints_a_fresh_request_id_per_attempt_and_reconciles(tmp_path):
    """The clamped retry of a context-length 400 re-sent the SAME ``X-Tinycap-Request-Id`` with a different
    body (``max_tokens`` 4096 -> 704), so the evidence held one id on two records and ``request_evidence_reconciliation``
    refused ("appears on two request-evidence records") while the row kept the draw-level id. Now the hook is called
    once per ACTUAL SEND: two attempts, two distinct ids, the row bound to the served attempt's id + completion id, and
    the REAL gate (contracts.request_evidence_reconciliation) passes over the two attempts' evidence and a two-row
    ledger whose first row is the rejection. The one-id-two-attempts world is pinned as the known-bad the gate refuses,
    and a hook that re-mints an id for the retry is refused at the send."""
    decode, rec = br.resolve_bon_decode("vendor", "qwen35_4b", path=policy_file_with_vendor_rows(tmp_path, name="retry.json"))
    minted = []
    headers_for = lambda g, i, t: _seam_headers(g, i, t, minted=minted, per_attempt=True)  # noqa: E731
    chat = _ClampOnceChat(window=1024, prompt=320)                        # room 704 = the reviewer's 4096 -> 704 clamp
    defense = bs.ServedDefense(chat, virtual_model="petri__base", cell_id="none", serving_engine="lens",
                               max_model_len=1024, achat_with_headers=chat.with_headers)
    out = tmp_path / "retry"
    bs.bon_generate_served(defense, _goals(1), out_dir=out, routes=_routes(tmp_path / "routes_retry"), n_max=1, sigma=0.4,
                           trial=0, max_new_tokens=4096, max_concurrency=1, decode=decode, decode_label="vendor",
                           decode_seed_policy="replicate_v1", decode_seed_base=1, decode_policy=rec,
                           request_headers_fn=headers_for)
    rows = _rows(out)
    assert len(rows) == 1 and len(chat.payloads) == 2 == len(chat.headers) == len(minted)
    ids = [h[tc.HEADER_REQUEST_ID] for h in chat.headers]
    assert ids == minted and len(set(ids)) == 2                           # two attempts, two DISTINCT ids
    assert [p["max_tokens"] for p in chat.payloads] == [4096, 704]        # the rejected body and the clamped one
    assert chat.payloads[0]["seed"] == chat.payloads[1]["seed"] == bon_mod.bon_decode_seed_v1(rows[0]["goal_id"], 0, 0, 1)
    row = rows[0]
    assert row["request_id"] == ids[1] != ids[0]                          # the SERVED attempt's id, not the draw's first
    assert row["completion_id"] == "cmpl-0002" and row["max_tokens_clamped_from"] == 4096   # the clamp fired (704 on the wire above)
    assert defense.n_clamped == 1 and defense.n_prompt_over_window == 0
    # the REAL reconciliation over the wire: one evidence record per attempt (the exact bytes + that attempt's headers),
    # one ledger row per attempt (the first rejected, no completion), the store's filed item, the frozen vendor row and
    # the replicate seed rule -> AGREES (None)
    raws = [json.dumps(p, sort_keys=True).encode() for p in chat.payloads]
    records = [tc.request_evidence_record(raw, headers=h) for raw, h in zip(raws, chat.headers)]
    ledger = [{"task": h[tc.HEADER_TASK], "sample": h[tc.HEADER_SAMPLE], "vmodel": "petri__base",
               tc.LEDGER_REQUEST_ID_FIELD: h[tc.HEADER_REQUEST_ID], tc.LEDGER_BODY_SHA256_FIELD: tc.wire_body_sha256(raw),
               "status": (400 if i == 0 else 200), "rejected": (i == 0), "error_class": ("context_length" if i == 0 else "none"),
               "finish_reason": (None if i == 0 else "stop"), "completion_id": (None if i == 0 else row["completion_id"])}
              for i, (raw, h) in enumerate(zip(raws, chat.headers))]
    frozen = tc.SamplingRow(temperature=0.7, top_p=0.8, top_k=20, source="the fixture's multi-row export", min_p=0.0,
                            presence_penalty=1.5, repetition_penalty=1.0)
    filed = {("strongreject_bon", row["goal_id"])}
    verdict = tc.request_evidence_reconciliation(
        records, ledger, filed=filed, expected_enable_thinking=None, expected_sampling=frozen,
        expected_seed=lambda task, item_id, sample_idx: bon_mod.bon_decode_seed_v1(item_id, sample_idx, 0, 1),
        target_vmodel="petri__base")
    assert verdict is None, verdict
    # the stored row binds to EXACTLY ONE ledger attempt -- the served one -- by request_id
    by_id = {l[tc.LEDGER_REQUEST_ID_FIELD]: l for l in ledger}
    assert len(by_id) == 2 and by_id[row["request_id"]]["rejected"] is False
    assert by_id[row["request_id"]]["completion_id"] == row["completion_id"] and by_id[ids[0]]["rejected"] is True
    # KNOWN-BAD (what the seam did before): ONE id on both attempts' evidence -> the real gate refuses by name
    reused = [dict(h, **{tc.HEADER_REQUEST_ID: ids[0]}) for h in chat.headers]
    bad = tc.request_evidence_reconciliation([tc.request_evidence_record(raw, headers=h) for raw, h in zip(raws, reused)],
                                             ledger, filed=filed, expected_enable_thinking=None)
    assert bad is not None and "appears on two request-evidence records" in bad
    # ... and a hook that re-mints the same id for the retry is refused AT THE SEND: one body went out, no row landed
    chat2 = _ClampOnceChat(window=1024, prompt=320)
    d2 = bs.ServedDefense(chat2, virtual_model="petri__base", cell_id="none", serving_engine="lens",
                          max_model_len=1024, achat_with_headers=chat2.with_headers)
    with pytest.raises(bs.BonServedError, match="re-minted request id 'constant' for the retry"):
        bs.bon_generate_served(d2, _goals(1), out_dir=tmp_path / "reuse", routes=_routes(tmp_path / "routes_reuse"), n_max=1,
                               max_concurrency=1, max_new_tokens=4096, decode=decode, decode_label="vendor",
                               decode_seed_policy="replicate_v1", decode_seed_base=1,
                               request_headers_fn=lambda g, i, t: {**ledger_headers("strongreject_bon", g, i),
                                                                  tc.HEADER_REQUEST_ID: "constant"})
    assert len(chat2.payloads) == 1 and len(chat2.headers) == 1 and not _rows(tmp_path / "reuse")
    # the direct surface: without a hook the retry path is byte-identical (no headers, no ids), the clamp still fires
    chat3 = _ClampOnceChat(window=1024, prompt=320)
    d3 = bs.ServedDefense(chat3, virtual_model="petri__base", cell_id="none", serving_engine="lens", max_model_len=1024)
    t = asyncio.run(d3.generate_one_detailed("v", sampling_seed=1, max_new_tokens=4096, temperature=0.7, decode=decode))
    assert t.request_id is None and t.completion_id == "cmpl-0002" and t.max_tokens_clamped_from == 4096 and t.max_tokens == 704
    assert chat3.headers == [] and [p["max_tokens"] for p in chat3.payloads] == [4096, 704]
