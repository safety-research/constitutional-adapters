"""The OPT-IN BoN decode-seed policy.

`bon_seed_v1` (the default, and every invocation that names no policy) is today's behaviour byte for byte: ONE seed,
bon_seed(goal, sample, trial), drives the augmentation AND the decode; rows and request bodies carry exactly what they
carried before the policy existed. `replicate_v1` keeps the augmentation on the frozen bon_seed — identical variant
TEXTS, the campaign's frozen item set — and draws the DECODE at bon_decode_seed_v1(goal, sample, trial, base): the
request bodies differ ONLY in `seed`, the rows ONLY in the decode-seed fields (`sampling_seed`, `seed_rule` and the
three replicate fields). The resume validator accepts a row iff the sampler AND the row's own policy's seeds reproduce;
a resume under another base, another policy, or with a torn replicate row refuses before any write. Everything
synthetic: the recording backend of tests/test_bon_served.py, invented goals, no server."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sisteer.eval.attacks import bon as bon_mod
from sisteer.eval.attacks import bon_served as bs

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "scripts" / "sdfbase" / "full22", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import eval_seed as es  # noqa: E402

BASE_S1, BASE_S2 = es.seed_base(1), es.seed_base(2)
#: the exact key set (and order) of a bon_seed_v1 served row BEFORE the policy existed — the pre-change golden
#: (`decode_policy_id`, the registry row the sampler came from, rides beside the label since 2026-09-17, main #216)
ROW_KEYS_GOLDEN = ("goal_id", "sample_idx", "trial", "variant", "response", "engine", "serving_engine", "cell_id",
                   "bon_max_concurrency", "finish_reason", "decode", "bon_decode_preset", "decode_policy_id", "sampling_seed", "seed_rule")
#: the sampler these byte-identity tests were written against — the grandfathered T=1 row, BY NAME (since 2026-09-17 best-of-N
#: has no model-independent default: a run names its served model and draws at the vendor preset, or names this row)
HARNESS = dict(decode_label=bs.BON_HARNESS_DEFAULT)


def _world(tmp_path):
    import test_bon_served as tb
    backend = tb.RecordingBackend()
    proxy, routes = tb._proxy(tmp_path, backend)
    defense = bs.ServedDefense.over_proxy(proxy, virtual_model="petri__base", cell_id="none")
    return tb, backend, routes, defense


def _rows(out):
    return [json.loads(l) for l in (out / "bon_responses.jsonl").read_text().splitlines() if l.strip()]


def _by_unit(rows):
    return {(r["goal_id"], r["sample_idx"], r["trial"]): r for r in rows}


def _payloads_by_unit(backend, rows):
    """The recorded request bodies keyed by the unit whose variant they carried (order-independent)."""
    by_variant = {r["variant"]: (r["goal_id"], r["sample_idx"], r["trial"]) for r in rows}
    out = {}
    for p in backend.payloads:
        out[by_variant[p["messages"][-1]["content"]]] = p
    return out


def _run(tmp_path, name, **kw):
    tb, backend, routes, defense = _world(tmp_path)
    out = tmp_path / name
    bs.bon_generate_served(defense, tb._goals(2), out_dir=out, routes=routes, n_max=3, max_concurrency=4, **HARNESS, **kw)
    rows = _rows(out)
    return out, rows, _payloads_by_unit(backend, rows), json.loads((out / bs.BON_SERVED_MANIFEST).read_text())


def test_the_derivation_and_the_policy_checks():
    assert bon_mod.DECODE_SEED_POLICIES == ("bon_seed_v1", "replicate_v1")
    assert bon_mod.seed_rule_for_policy(None) == bon_mod.seed_rule_for_policy("bon_seed_v1") == "bon_seed_v1" == bon_mod.BON_SEED_RULE
    assert bon_mod.seed_rule_for_policy("replicate_v1") == "bon_seed_v1+decode:replicate_v1" == es.BON_SEED_RULE_RECORDED
    assert bon_mod.REPLICATE_ROW_KEYS == ("decode_seed_policy", "decode_seed_base", "augment_seed")
    assert bon_mod.check_decode_seed_policy(None, None) == ("bon_seed_v1", None) == bon_mod.check_decode_seed_policy("bon_seed_v1", None)
    assert bon_mod.check_decode_seed_policy("replicate_v1", BASE_S1) == ("replicate_v1", BASE_S1)
    for policy, base, needle in (("bon_seed_v1", BASE_S1, "takes no decode_seed_base"), ("replicate_v1", None, "requires a non-negative int"),
                                 ("replicate_v1", -1, "requires a non-negative int"), ("replicate_v1", True, "requires a non-negative int"),
                                 ("replicate_v2", BASE_S1, "not one of")):
        with pytest.raises(ValueError, match=needle):
            bon_mod.check_decode_seed_policy(policy, base)
    # the decode seed: deterministic, in bon_seed's range, distinct across bases / units, never the augmentation seed for these units
    a = bon_mod.bon_decode_seed_v1("g0", 0, 0, BASE_S1)
    assert a == bon_mod.bon_decode_seed_v1("g0", 0, 0, BASE_S1) and 0 <= a < 2 ** 31
    assert a != bon_mod.bon_decode_seed_v1("g0", 0, 0, BASE_S2) != bon_mod.bon_decode_seed_v1("g0", 1, 0, BASE_S2)
    assert a != bon_mod.bon_seed("g0", 0, 0)
    assert bon_mod.decode_seed_for("g0", 0, 0) == bon_mod.bon_seed("g0", 0, 0) == bon_mod.decode_seed_for("g0", 0, 0, policy="bon_seed_v1")
    assert bon_mod.decode_seed_for("g0", 0, 0, policy="replicate_v1", decode_seed_base=BASE_S1) == a
    # the rule string names the derivation the ledger and the results doc quote
    assert "bon-decode|{decode_seed_base}|{goal_id}|{sample_idx}|{trial}" in bon_mod.BON_DECODE_SEED_RULE_V1 and "% 2**31" in bon_mod.BON_DECODE_SEED_RULE_V1
    with pytest.raises(ValueError):
        bon_mod.bon_decode_seed_v1("g0", 0, 0, -3)


def test_the_default_policy_is_byte_identical_to_the_pre_change_golden(tmp_path):
    """No policy named (every invocation before this change) and `bon_seed_v1` named explicitly: the same rows byte for byte,
    the golden key set and order, seed == bon_seed on the row AND on the wire, the manifest's frozen rule."""
    out_a, rows_a, req_a, man_a = _run(tmp_path, "implicit")
    out_b, rows_b, req_b, man_b = _run(tmp_path, "explicit", decode_seed_policy="bon_seed_v1")
    assert (out_a / "bon_responses.jsonl").read_bytes() == (out_b / "bon_responses.jsonl").read_bytes()
    assert len(rows_a) == 6
    for r in rows_a:
        assert tuple(r) == ROW_KEYS_GOLDEN                        # no replicate field, the pre-change order
        assert r["seed_rule"] == "bon_seed_v1" and r["sampling_seed"] == bon_mod.bon_seed(r["goal_id"], r["sample_idx"], r["trial"])
        assert r["decode"] == {"temperature": 1.0} and r["bon_decode_preset"] == "harness_default"
    for unit, p in req_a.items():
        assert p["seed"] == bon_mod.bon_seed(*unit) and p["temperature"] == 1.0 and "top_p" not in p and "top_k" not in p
    assert req_a == req_b
    for man in (man_a, man_b):
        assert man["seed_rule"] == "bon_seed_v1" and man["decode_seed_policy"] == "bon_seed_v1" and man["decode_seed_base"] is None


def test_replicate_v1_changes_only_the_decode_seed_fields_and_the_variants_are_identical(tmp_path):
    out_d, rows_d, req_d, man_d = _run(tmp_path, "default")
    out_r, rows_r, req_r, man_r = _run(tmp_path, "replicate", decode_seed_policy="replicate_v1", decode_seed_base=BASE_S1)
    d, r = _by_unit(rows_d), _by_unit(rows_r)
    assert set(d) == set(r) and len(r) == 6
    changed = {"sampling_seed", "seed_rule"}
    added = set(bon_mod.REPLICATE_ROW_KEYS)
    for unit in d:
        rd, rr = d[unit], r[unit]
        assert rr["variant"] == rd["variant"]                                       # the frozen item set: identical bytes
        assert rr["augment_seed"] == bon_mod.bon_seed(*unit) == rd["sampling_seed"]  # the augmentation seed IS the frozen rule's
        assert rr["sampling_seed"] == bon_mod.bon_decode_seed_v1(*unit, BASE_S1) != rd["sampling_seed"]
        assert rr["seed_rule"] == "bon_seed_v1+decode:replicate_v1" and rr["decode_seed_policy"] == "replicate_v1" and rr["decode_seed_base"] == BASE_S1
        assert rr["decode"] == rd["decode"] and rr["bon_decode_preset"] == rd["bon_decode_preset"]
        # everything else the row carries is identical (the response text encodes the request, so it is identical too)
        same = {k for k in rd if k not in changed}
        assert {k: rr[k] for k in same} == {k: rd[k] for k in same}
        assert set(rr) == set(rd) | added
        assert tuple(rr)[:len(ROW_KEYS_GOLDEN)] == ROW_KEYS_GOLDEN and tuple(rr)[len(ROW_KEYS_GOLDEN):] == bon_mod.REPLICATE_ROW_KEYS
        # the request bodies differ ONLY in `seed`
        pd, pr = dict(req_d[unit]), dict(req_r[unit])
        assert pd.pop("seed") == rd["sampling_seed"] and pr.pop("seed") == rr["sampling_seed"] and pd == pr
    assert man_r["seed_rule"] == "bon_seed_v1+decode:replicate_v1" and man_r["decode_seed_policy"] == "replicate_v1" and man_r["decode_seed_base"] == BASE_S1
    assert man_d["seed_rule"] == "bon_seed_v1" and man_d["decode_seed_base"] is None
    # two bases: identical variants, different decode seeds
    out_2, rows_2, req_2, _ = _run(tmp_path, "replicate2", decode_seed_policy="replicate_v1", decode_seed_base=BASE_S2)
    r2 = _by_unit(rows_2)
    assert all(r2[u]["variant"] == r[u]["variant"] and r2[u]["sampling_seed"] != r[u]["sampling_seed"] for u in r)
    assert all(req_2[u]["seed"] != req_r[u]["seed"] for u in r)


def test_the_resume_validator_accepts_and_refuses_by_the_rows_own_recorded_policy(tmp_path):
    tb, backend, routes, defense = _world(tmp_path)
    goals = tb._goals(2)
    rep = dict(decode_seed_policy="replicate_v1", decode_seed_base=BASE_S1)
    out = tmp_path / "store"
    bs.bon_generate_served(defense, goals, out_dir=out, routes=routes, n_max=3, max_concurrency=4, **HARNESS, **rep)
    rows = _rows(out)
    n_calls = backend.n_calls
    # (a) the same base resumes idempotently: nothing regenerated, the store byte-identical
    before = (out / "bon_responses.jsonl").read_bytes()
    bs.bon_generate_served(defense, goals, out_dir=out, routes=routes, n_max=3, max_concurrency=4, **HARNESS, **rep)
    assert backend.n_calls == n_calls and (out / "bon_responses.jsonl").read_bytes() == before
    # (b) another replicate's base refuses before any write
    with pytest.raises(bs.BonServedError, match="drawn under decode_seed_base .* and this run asks for") as ei:
        bs.bon_generate_served(defense, goals, out_dir=out, routes=routes, n_max=3, max_concurrency=4, **HARNESS,
                               decode_seed_policy="replicate_v1", decode_seed_base=BASE_S2)
    assert "different replicate" in str(ei.value) and (out / "bon_responses.jsonl").read_bytes() == before
    # (c) the default policy over a replicate store refuses: another sample identity
    with pytest.raises(bs.BonServedError, match="names the seed rule 'bon_seed_v1\\+decode:replicate_v1'"):
        bs.bon_generate_served(defense, goals, out_dir=out, routes=routes, n_max=3, max_concurrency=4, **HARNESS)
    # (d) a replicate run over a bon_seed_v1 store refuses the same way
    out_d = tmp_path / "default"
    bs.bon_generate_served(defense, goals, out_dir=out_d, routes=routes, n_max=3, max_concurrency=4, **HARNESS)
    with pytest.raises(bs.BonServedError, match="names the seed rule 'bon_seed_v1', this harness implements 'bon_seed_v1\\+decode:replicate_v1'"):
        bs.bon_generate_served(defense, goals, out_dir=out_d, routes=routes, n_max=3, max_concurrency=4, **HARNESS, **rep)
    # (e) a replicate row whose decode seed does not reproduce at ITS base is a different sample identity
    bad = [dict(r) for r in rows]; bad[0]["sampling_seed"] = (bad[0]["sampling_seed"] + 1) % 2 ** 31
    (out / "bon_responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bad))
    with pytest.raises(bs.BonServedError, match="records sampling_seed .* but bon_seed_v1\\+decode:replicate_v1 at decode_seed_base"):
        bs.bon_generate_served(defense, goals, out_dir=out, routes=routes, n_max=3, max_concurrency=4, **HARNESS, **rep)
    # (f) a replicate row whose AUGMENT seed is not the frozen rule's is a different VARIANT identity
    bad = [dict(r) for r in rows]; bad[1]["augment_seed"] += 1
    (out / "bon_responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bad))
    with pytest.raises(bs.BonServedError, match="different VARIANT identity"):
        bs.bon_generate_served(defense, goals, out_dir=out, routes=routes, n_max=3, max_concurrency=4, **HARNESS, **rep)
    # (g) a replicate row missing a policy field is UNVERIFIABLE, never accepted
    for k in bon_mod.REPLICATE_ROW_KEYS:
        torn = [dict(r) for r in rows]; del torn[2][k]
        (out / "bon_responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in torn))
        with pytest.raises(bs.BonServedError, match="UNVERIFIABLE"):
            bs.bon_generate_served(defense, goals, out_dir=out, routes=routes, n_max=3, max_concurrency=4, **HARNESS, **rep)
    # (h) a LEGACY (seedless) store is never relabelled as a replicate
    legacy_keys = [k for k in ROW_KEYS_GOLDEN if k not in bs.ROW_IDENTITY_KEYS]      # engine stamps kept: only the identity fields are absent
    (out / "bon_responses.jsonl").write_text("".join(json.dumps({k: r[k] for k in legacy_keys}) + "\n" for r in rows))
    with pytest.raises(bs.BonServedError, match="LEGACY"):
        bs.bon_generate_served(defense, goals, out_dir=out, routes=routes, n_max=3, max_concurrency=4, **HARNESS, **rep)
    # (i) a misconfigured pair refuses BEFORE any gate (no manifest touched)
    fresh = tmp_path / "fresh"
    with pytest.raises(bs.BonServedError, match="takes no decode_seed_base"):
        bs.bon_generate_served(defense, goals, out_dir=fresh, routes=routes, n_max=3, max_concurrency=4, **HARNESS, decode_seed_base=BASE_S1)
    assert not fresh.exists()


def test_the_driver_takes_the_flags_and_the_anchor_match_preset_is_a_sendable_sampler():
    import bon_served_run as br
    assert br.DECODE_SEED_POLICIES == bon_mod.DECODE_SEED_POLICIES
    ap = br.build_parser()
    base = ["--config", "c.yaml", "--defense", "none", "--private-root", "/r", "--proxy-url", "u", "--routing-table", "t.json",
            "--max-model-len", "4864", "--max-concurrency", "21"]
    ns = ap.parse_args(base)
    assert ns.bon_decode_preset == "vendor_preset" == bs.BON_VENDOR_PRESET and ns.decode_seed_policy == "bon_seed_v1" and ns.decode_seed_base is None
    ns = ap.parse_args(base + ["--bon-decode-preset", "anchor_match", "--decode-seed-policy", "replicate_v1", "--decode-seed-base", str(BASE_S1)])
    assert ns.bon_decode_preset == "anchor_match" and ns.decode_seed_policy == "replicate_v1" and ns.decode_seed_base == BASE_S1
    with pytest.raises(SystemExit):
        ap.parse_args(base + ["--decode-seed-policy", "replicate_v2"])
    # `vendor` IS a choice: BoN runs on the vendor preset, a NEW instrument — the old pin (`vendor` refused by the
    # parser) is superseded; its protective intent stands at the resolver instead: a vendor run with no --model-key
    # (the registry row is keyed by the served model) refuses before the proxy is touched
    # (tests/test_bon_vendor_preset.py pins the row itself)
    assert ap.parse_args(base + ["--bon-decode-preset", "vendor", "--model-key", "qwen35_4b"]).bon_decode_preset == "vendor"
    assert br.BON_DECODE_PRESETS["vendor"] == br.BON_DECODE_FROM_REGISTRY == "registry"   # a marker, never the numbers
    with pytest.raises(SystemExit, match="requires --model-key"):
        br.resolve_bon_decode("vendor", None)
    # the anchor_match preset: exactly the anchor's resolved tuple, a sendable sampler (bon_served's own rule), labelled by name
    preset = br.BON_DECODE_PRESETS["anchor_match"]
    assert preset == {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "presence_penalty": 0.0}
    assert bs.sampler_preset_problem(preset) is None and bs.sampler_preset(preset) == preset
    assert bs._sent_sampler(preset) == preset and bs._preset_label(preset, "anchor_match") == "anchor_match"
    assert br.BON_DECODE_PRESETS["harness_default"] is None is br.BON_DECODE_PRESETS["vendor_preset"]   # registry labels, resolved per model
    assert bs._sent_sampler(None, bs.BON_HARNESS_DEFAULT) == {"temperature": 1.0}
    # the run record names the policy, its base and the rule; the default path still reads no preset table
    src = (REPO / "scripts/sdfbase/full22/bon_served_run.py").read_text()
    assert '"decode_seed_policy": seed_policy, "decode_seed_base": seed_base,' in src and '"seed_rule": bon_mod.seed_rule_for_policy(seed_policy)' in src
    assert "decode_seed_policy=seed_policy, decode_seed_base=seed_base" in src
    assert "sampled_decode" not in src and "resolve_default_decode" not in src
