"""The STRUCTURAL selection verdict and the demoted output diagnostics
(gemma certification canaries 489308 and 489533).

``check_multi_lora_routing_attested``: THE JOIN — the engine renames every
request (``InputProcessor.assign_request_id``: the client's id plus ONE
``-<8 hex>`` suffix), so each external id must match exactly one engine id
under ``serving_attestation.is_engine_id_of`` (zero -> missing, two ->
ambiguous, both refuse naming the selection; a bare prefix, a second suffix
or a parallel-sampling child ``<i>_<id>-…`` never match; unmatched engine ids
are FOREIGN, recorded); pass; a misroute (named by selection and adapter,
never by id); a request under two ids; base under a lora id; the ZERO
adapter under base REFUSES (no structural exemption); an id <-> name
disagreement with the attested map; MIXED STEPS computed over THIS control's
joined requests from the epoch's per-step sets (a fully serial control
refuses; a neighbour's traffic cannot supply the mixing); an unknown runner
kind; overflow / steps_overflow; recorder read failures; the all-base
counterexample. The id contract is pinned EQUAL to the lens gate's
(``scripts/lens_vectorize_gate.py``) so it has one meaning in the tree.

``multi_lora_output_diagnostics``: never raises on disagreement; the base
reference submitted twice as ``fixed_size_repeat_floor``;
``n_undistinguishable`` recorded; ``expected_routing`` over every submitted
row; a row without an id or a duplicate id REFUSES.

``run_multi_lora_routing_control``: MINT the epoch (the client, so every
worker adopts one id) -> arm(epoch) -> batches -> observe(epoch) -> check ->
disarm(epoch) in a finally; an arm that answers another epoch (or none), or
an observe for another epoch, refuses; an arm REFUSAL gets a best-effort
disarm with the minted epoch, recorded in the refusal and never masking it;
a disarm failure after a pass refuses; the two-session regression (a control
after a mixed one, fully serial, refuses). Synthetic ids; no server."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import fidelity as fid
from sisteer.capeval.serving import serving_attestation as sa

PROMPTS = (({"role": "user", "content": "p0"},),
           ({"role": "user", "content": "p1"},),
           ({"role": "user", "content": "p2"},))
TABLE = {"base": [1, 1], "zero": [1, 1], "genuine": [2, 2]}
SEL_ADAPTER = {"base": None, "zero": "none__lora0", "genuine": "cell_a"}
ID_NAME = {"1": "none__lora0", "2": "cell_a"}
ID_OF = {None: 0, "none__lora0": 1, "cell_a": 2}
V2 = "vllm.v1.worker.gpu.model_runner.GPUModelRunner"


def _eng(ext, salt=""):
    """The engine's internal id for external id ``ext``: one ``-<8 hex>`` suffix."""
    return f"{ext}-{hashlib.sha256((ext + salt).encode()).hexdigest()[:8]}"


def _expected(n_per=2):
    return {f"chatcmpl-{label}{i}": label for label in ("base", "zero", "genuine") for i in range(n_per)}


def _obs(expected, *, override=None, exact=False):
    """Honest engine bookkeeping for ``expected``, keyed by ENGINE id."""
    out = {}
    for rid, label in expected.items():
        name = SEL_ADAPTER[label]
        out[rid if exact else _eng(rid)] = {"lora_int_id": ID_OF[name], "lora_name": name}
    out.update(override or {})
    return out


def _steps(expected, *, exact=False, serial=False):
    """Per-step scheduled engine-id sets: one mixed step over everything (or
    one request per step when ``serial``), then a decode step each."""
    eids = [rid if exact else _eng(rid) for rid in expected]
    if serial:
        return [[e] for e in eids]
    return [sorted(eids), sorted(eids)]


def _check(expected=None, observed=None, **over):
    expected = _expected() if expected is None else expected
    kw = dict(expected=expected, selection_adapter=SEL_ADAPTER,
              observed=_obs(expected) if observed is None else observed,
              inconsistent={}, steps=_steps(expected), attested_id_name_map=ID_NAME,
              runner_kind="v2", runner_class=V2)
    kw.update(over)
    return fid.check_multi_lora_routing_attested(**kw)


# --------------------------------------------------------------------------- #
# the id contract (one home; equal to the lens gate's)                          #
# --------------------------------------------------------------------------- #


def test_the_engine_id_contract_and_its_single_home():
    assert sa.ENGINE_REQUEST_ID_SUFFIX_RE.pattern == r"-[0-9a-f]{8}\Z"
    x = "chatcmpl-abc"
    assert sa.is_engine_id_of(x, x)                                    # randomisation off
    assert sa.is_engine_id_of(x + "-0123abcd", x)                      # exactly one suffix
    assert not sa.is_engine_id_of(x + "-0123ABCD", x)                  # uppercase: not the runtime's
    assert not sa.is_engine_id_of(x + "-0123abc", x)                   # 7 hex
    assert not sa.is_engine_id_of(x + "-0123abcd-0123abcd", x)         # two suffixes
    assert not sa.is_engine_id_of(x + "_0123abcd", x)                  # another delimiter
    assert not sa.is_engine_id_of(x + "x-0123abcd", x)                 # a merely prefix-sharing id
    assert not sa.is_engine_id_of("0_" + x + "-0123abcd", x)           # a parallel-sampling child
    assert not sa.is_engine_id_of("", x) and not sa.is_engine_id_of(x, "") and not sa.is_engine_id_of(None, x)
    assert sa.external_request_id_candidates(x + "-0123abcd") == (x + "-0123abcd", x)
    assert sa.external_request_id_candidates(x) == (x,)
    assert sa.external_request_id_candidates("-0123abcd") == ("-0123abcd",)   # a bare suffix strips to nothing
    assert sa.external_request_id_candidates(None) == () and sa.external_request_id_candidates("") == ()
    # PINNED EQUAL to the lens gate's contract (scripts/lens_vectorize_gate.py): one meaning
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("_lens_vectorize_gate_for_pin",
                                                  root / "scripts" / "lens_vectorize_gate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.ENGINE_REQUEST_ID_SUFFIX_RE.pattern == sa.ENGINE_REQUEST_ID_SUFFIX_RE.pattern
    cases = [(x, x), (x + "-0123abcd", x), (x + "-0123ABCD", x), (x + "-0123abc", x),
             (x + "-0123abcd-0123abcd", x), (x + "_0123abcd", x), (x + "x-0123abcd", x),
             ("0_" + x + "-0123abcd", x), ("", x), (x, ""), (None, x), (x, None), (x + "-0123abcd", x + "-0123abcd")]
    assert [mod.is_engine_id_of(a, b) for a, b in cases] == [sa.is_engine_id_of(a, b) for a, b in cases]


# --------------------------------------------------------------------------- #
# the structural verdict                                                        #
# --------------------------------------------------------------------------- #


def test_routing_attested_joins_the_randomised_ids_and_reports_per_selection():
    out = _check()
    assert out["verdict"] == fid.SELECTION_VERDICT_ROUTING_ATTESTED == "routing_attested"
    assert out["verdict"] == sa.SELECTION_VERDICT_ROUTING_ATTESTED
    assert out["n_requests"] == 6 == out["n_observed"] and out["n_missing"] == 0
    assert out["n_ambiguous"] == 0 and out["n_foreign"] == 0
    assert out["n_inconsistent"] == 0 and out["n_misrouted"] == 0 and out["exemption"] is None
    assert out["per_selection"] == {
        "base": {"n": 2, "n_observed": 2, "adapter": None, "lora_int_id": 0},
        "zero": {"n": 2, "n_observed": 2, "adapter": "none__lora0", "lora_int_id": 1},
        "genuine": {"n": 2, "n_observed": 2, "adapter": "cell_a", "lora_int_id": 2}}
    assert out["id_join"]["n_suffixed"] == 6 and out["id_join"]["n_exact"] == 0
    assert out["id_join"]["n_engine_ids_recorded"] == 6
    assert out["steps"] == {"n_steps": 2, "n_steps_with_own_requests": 2, "n_steps_mixed": 2,
                            "max_distinct_selections_per_step": 3,
                            "basis": "computed over this control's joined requests only"}
    assert out["runner_kind"] == "v2" and out["runner_class"] == V2 and out["id_name_map"] == ID_NAME
    json.dumps(out)
    # randomisation OFF (exact ids) joins too
    exp = _expected()
    out = _check(exp, _obs(exp, exact=True), steps=_steps(exp, exact=True))
    assert out["id_join"]["n_exact"] == 6 and out["id_join"]["n_suffixed"] == 0


def test_foreign_engine_ids_are_recorded_never_judged():
    exp = _expected()
    obs = _obs(exp)
    obs["chatcmpl-neighbour-0a1b2c3d"] = {"lora_int_id": 0, "lora_name": None}          # another driver's
    obs["0_" + _eng("chatcmpl-base0")] = {"lora_int_id": 2, "lora_name": "cell_a"}        # a child: not ours
    obs["chatcmpl-base0x-0a1b2c3d"] = {"lora_int_id": 2, "lora_name": "cell_a"}          # prefix-sharing: not ours
    steps = _steps(exp)
    steps[0].append("chatcmpl-neighbour-0a1b2c3d")
    out = _check(exp, obs, steps=steps)
    assert out["n_foreign"] == 3 and out["n_observed"] == 6
    assert out["steps"]["max_distinct_selections_per_step"] == 3           # foreign ids add no selection


def test_a_missing_or_ambiguous_join_refuses_naming_the_selection():
    exp = _expected()
    obs = _obs(exp)
    del obs[_eng("chatcmpl-zero0")]
    with pytest.raises(fid.FidelityError, match="never observed by the engine's LoRA bookkeeping") as ei:
        _check(exp, obs)
    assert "'zero': 1/2" in str(ei.value) and "chatcmpl-zero0" not in str(ei.value)
    assert "5 engine ids recorded, 0 foreign" in str(ei.value)
    # the pre-fix shape (job 489533): observations keyed by engine ids, expected by external ids,
    # joined by EXACT match only -> every request missing; the join now finds them
    assert _check(exp, _obs(exp))["n_observed"] == 6
    # two engine ids for one external id (two suffixes) -> ambiguous, refused
    obs = _obs(exp)
    obs[_eng("chatcmpl-zero0", salt="again")] = {"lora_int_id": 1, "lora_name": "none__lora0"}
    with pytest.raises(fid.FidelityError, match="matched by MORE THAN ONE engine request id") as ei:
        _check(exp, obs)
    assert "'zero': 1" in str(ei.value)
    # the exact form AND a suffixed form of the same external id -> ambiguous too
    obs = _obs(exp)
    obs["chatcmpl-zero0"] = {"lora_int_id": 1, "lora_name": "none__lora0"}
    with pytest.raises(fid.FidelityError, match="MORE THAN ONE"):
        _check(exp, obs)
    # missing outranks a misroute elsewhere (evidence first, then verdict)
    obs = _obs(exp, override={_eng("chatcmpl-genuine1"): {"lora_int_id": 0, "lora_name": None}})
    del obs[_eng("chatcmpl-base0")]
    with pytest.raises(fid.FidelityError, match="never observed"):
        _check(exp, obs)


def test_a_misrouted_request_refuses_naming_selection_and_adapters_never_the_id():
    exp = _expected()
    obs = _obs(exp, override={_eng("chatcmpl-genuine1"): {"lora_int_id": 0, "lora_name": None}})
    with pytest.raises(fid.FidelityError, match="served under an adapter other than") as ei:
        _check(exp, obs)
    msg = str(ei.value)
    assert "1/2 request(s) for 'genuine' ran under ['<no LoRA>'] instead of 'cell_a'" in msg
    assert "chatcmpl-genuine1" not in msg and "genuine1-" not in msg          # never a request id
    obs = _obs(exp, override={_eng("chatcmpl-genuine0"): {"lora_int_id": 1, "lora_name": "none__lora0"}})
    with pytest.raises(fid.FidelityError, match=r"ran under \['none__lora0'\] instead of 'cell_a'"):
        _check(exp, obs)


def test_base_under_a_lora_id_and_zero_under_base_both_refuse_no_exemption():
    exp = _expected()
    with pytest.raises(fid.FidelityError, match=r"request\(s\) for 'base' ran under \['none__lora0'\] instead of '<no LoRA>'"):
        _check(exp, _obs(exp, override={_eng("chatcmpl-base0"): {"lora_int_id": 1, "lora_name": "none__lora0"}}))
    with pytest.raises(fid.FidelityError, match=r"request\(s\) for 'zero' ran under \['<no LoRA>'\] instead of 'none__lora0'") as ei:
        _check(exp, _obs(exp, override={_eng("chatcmpl-zero1"): {"lora_int_id": 0, "lora_name": None}}))
    assert "no base <-> zero exemption structurally" in str(ei.value)


def test_astras_all_base_counterexample_refuses_structurally():
    exp = _expected()
    all_base = {_eng(rid): {"lora_int_id": 0, "lora_name": None} for rid in exp}
    with pytest.raises(fid.FidelityError, match="served under an adapter other than") as ei:
        _check(exp, all_base)
    msg = str(ei.value)
    assert "2/2 request(s) for 'genuine' ran under ['<no LoRA>'] instead of 'cell_a'" in msg
    assert "2/2 request(s) for 'zero' ran under ['<no LoRA>'] instead of 'none__lora0'" in msg


def test_a_request_under_two_bindings_refuses():
    exp = _expected()
    with pytest.raises(fid.FidelityError, match="TWO different LoRA bindings across steps") as ei:
        _check(exp, inconsistent={_eng("chatcmpl-genuine0"): [{"lora_int_id": 2, "lora_name": "cell_a"},
                                                              {"lora_int_id": 1, "lora_name": "none__lora0"}]})
    assert "'genuine': 1" in str(ei.value) and "genuine0" not in str(ei.value)
    _check(exp, inconsistent={"foreign-0a1b2c3d": [{"lora_int_id": 1, "lora_name": "x"}]})   # not ours: ignored


def test_id_name_disagreement_with_the_attestation_refuses():
    exp = _expected()
    obs = _obs(exp, override={_eng("chatcmpl-genuine0"): {"lora_int_id": 2, "lora_name": "none__lora0"}})
    with pytest.raises(fid.FidelityError, match="binding disagrees with the readiness attestation") as ei:
        _check(exp, obs)
    assert "binds lora id 2 to 'none__lora0' but the readiness attestation binds id 2 to 'cell_a'" in str(ei.value)
    obs = _obs(exp, override={_eng("chatcmpl-base0"): {"lora_int_id": 0, "lora_name": "cell_a"}})
    with pytest.raises(fid.FidelityError, match="under lora id 0 \\(no LoRA\\) carries the adapter name"):
        _check(exp, obs)
    obs = _obs(exp, override={_eng("chatcmpl-base0"): {"lora_int_id": "1", "lora_name": None}})
    with pytest.raises(fid.FidelityError, match="not an int"):
        _check(exp, obs)
    with pytest.raises(fid.FidelityError, match="id 0 is NO LoRA"):
        _check(exp, attested_id_name_map={"0": "x", "1": "none__lora0", "2": "cell_a"})
    with pytest.raises(fid.FidelityError, match="is not an int"):
        _check(exp, attested_id_name_map={"one": "none__lora0"})
    with pytest.raises(fid.FidelityError, match="share one adapter name"):
        _check(exp, attested_id_name_map={"1": "cell_a", "2": "cell_a"})
    with pytest.raises(fid.FidelityError, match="empty attested id_name_map"):
        _check(exp, attested_id_name_map={})
    with pytest.raises(fid.FidelityError, match="selects adapter 'ghost', which the readiness attestation does not name"):
        _check(exp, selection_adapter={**SEL_ADAPTER, "genuine": "ghost"})
    with pytest.raises(fid.FidelityError, match="selection 'genuine' has no adapter entry"):
        _check(exp, selection_adapter={"base": None, "zero": "none__lora0"})


def test_mixed_steps_are_the_controls_own_and_a_serial_control_refuses():
    """A recorder-side mixed count could be inherited from a neighbour's
    traffic or an earlier control. The count is computed HERE over the
    joined requests: steps scheduling only foreign ids add nothing; a fully
    serial control refuses as vacuous."""
    exp = _expected()
    # the two-session regression's check half: only serial steps of our own
    with pytest.raises(fid.FidelityError, match="no engine step scheduled >= 2 distinct selections of THIS control") as ei:
        _check(exp, steps=_steps(exp, serial=True))
    assert "6 step(s) recorded, 6 with any of this control's requests, max 1 distinct" in str(ei.value)
    # a neighbour's mixed traffic in the same steps does not make OUR steps mixed
    neighbour = [["nb-a-0a1b2c3d", "nb-b-0a1b2c3d"] + [e] for e in [_eng(r) for r in exp]]
    obs = _obs(exp)
    obs["nb-a-0a1b2c3d"] = {"lora_int_id": 0, "lora_name": None}
    obs["nb-b-0a1b2c3d"] = {"lora_int_id": 2, "lora_name": "cell_a"}
    with pytest.raises(fid.FidelityError, match="no engine step scheduled >= 2 distinct selections"):
        _check(exp, obs, steps=neighbour)
    # one mixed step of our own suffices; steps naming no request of ours are counted, not own
    steps = [["nb-x"], [_eng("chatcmpl-base0"), _eng("chatcmpl-zero0")]] + _steps(exp, serial=True)
    out = _check(exp, steps=steps)
    assert out["steps"] == {"n_steps": 8, "n_steps_with_own_requests": 7, "n_steps_mixed": 1,
                            "max_distinct_selections_per_step": 2,
                            "basis": "computed over this control's joined requests only"}
    with pytest.raises(fid.FidelityError, match="steps is str"):
        _check(exp, steps="x")


def test_unknown_runner_overflow_errors_and_vacuous_inputs_refuse():
    exp = _expected()
    with pytest.raises(fid.FidelityError, match="of kind 'unknown', not one of the pinned runners") as ei:
        _check(exp, runner_kind="unknown", runner_class="tests.Fake")
    assert "'tests.Fake'" in str(ei.value) and "unpinned engine" in str(ei.value)
    with pytest.raises(fid.FidelityError, match="not one of the pinned runners"):
        _check(exp, runner_kind=None)
    assert _check(exp, runner_kind="v1", runner_class="vllm.v1.worker.gpu_model_runner.GPUModelRunner")["runner_kind"] == "v1"
    with pytest.raises(fid.FidelityError, match="OVERFLOWED its request-id bound"):
        _check(exp, overflow=True)
    with pytest.raises(fid.FidelityError, match="OVERFLOWED its per-step list"):
        _check(exp, steps_overflow=True)
    with pytest.raises(fid.FidelityError, match="could not read the engine's state on 2 step"):
        _check(exp, recorder_errors=["AttributeError: x", "AttributeError: y"])
    with pytest.raises(fid.FidelityError, match="no expected requests"):
        _check({})


# --------------------------------------------------------------------------- #
# the demoted output diagnostics                                                #
# --------------------------------------------------------------------------- #


def _world(table, *, with_ids=False):
    counter = [0]

    def solo(name, messages):
        return list(table[name])

    def mixed(batch):
        rows = [list(table[name]) for name, _ in batch]
        if not with_ids:
            return rows
        out = []
        for r in rows:
            counter[0] += 1
            out.append((r, f"chatcmpl-{counter[0]}"))
        return out
    return solo, mixed


def _diag(table=TABLE, *, with_ids=False, mixed=None, solo=None, **over):
    s, m = _world(table, with_ids=with_ids)
    kw = dict(base="base", adapters=("genuine", "zero"), prompt_set=PROMPTS,
              mixed_batch=mixed or m, solo=s if solo is None else solo, zero_adapter="zero",
              request_ids=with_ids)
    kw.update(over)
    return fid.multi_lora_output_diagnostics(**kw)


def test_diagnostics_record_an_honest_engine_with_the_repeat_floor_and_expected_routing():
    d = _diag(with_ids=True)
    assert d["policy"] == "native" and d["gating"] is False and d["rule"] == "equal-size"
    assert d["n_batch"] == 9 and d["n_layouts"] == 3 and d["n_mixed_rows"] == 27
    assert d["n_submitted_rows"] == 63 == d["n_requests"] == len(d["expected_routing"])
    assert sorted(set(d["expected_routing"].values())) == ["base", "genuine", "zero"]
    assert list(d["expected_routing"].values()).count("base") == 9 * 3
    assert d["whole_layout"] == {"signatures_distinct": True, "colliding": [],
                                 "genuine_like_base": [],
                                 "zero_vs_base": {"n_rows": 9, "n_divergent_rows": 0,
                                                  "n_prompts_differing": 0, "first_divergences": []}}
    assert d["zero_equals_base"] is True and d["signatures_distinct"] is True
    pd = d["positional_distinguishability"]
    assert pd == {"n_pairs_checked": 9 * 4, "n_undistinguishable": 0, "first": [],
                  "exemption": "base <-> zero adapter"}
    assert d["selection_agreement"]["genuine"] == {"n_rows": 9, "n_agree": 9, "first_divergences": [],
                                                   "fraction_agree": 1.0}
    assert d["selection_first_divergences"] == [] and d["n_mismatched_rows"] == 0
    fl = d["fixed_size_repeat_floor"]
    assert (fl["selection"], fl["n_rows"], fl["n_divergent_rows"], fl["fraction_divergent"]) == ("base", 9, 0, 0.0)
    assert "kernel-path" in fl["reads_as"]
    json.dumps(d)
    d2 = _diag()
    assert "expected_routing" not in d2 and d2["n_submitted_rows"] == 63


def test_diagnostics_never_raise_on_disagreement_they_record_it():
    d = _diag(dict(TABLE, zero=[1, 2]))
    zb = d["whole_layout"]["zero_vs_base"]
    assert zb["n_divergent_rows"] == 9 and zb["n_prompts_differing"] == 3 and d["zero_equals_base"] is False
    d = _diag(dict(TABLE, genuine=[1, 1]))
    assert d["whole_layout"]["genuine_like_base"] == ["genuine"]
    assert d["whole_layout"]["colliding"] == ["base", "genuine", "zero"] and d["signatures_distinct"] is False
    assert d["positional_distinguishability"]["n_undistinguishable"] == 9 * 4

    def misroute(batch):
        outs = [list(TABLE[n]) for n, _ in batch]
        if len({n for n, _ in batch}) > 1:
            k = [i for i, (n, _) in enumerate(batch) if n == "genuine"][1]
            outs[k] = list(TABLE["base"])
        return outs
    d = _diag(mixed=misroute)
    assert d["n_mismatched_rows"] == 3
    assert d["selection_first_divergences"] == [[0, 4, "genuine", 1, 0], [1, 1, "genuine", 1, 0],
                                                [2, 7, "genuine", 1, 0]]
    ag = d["selection_agreement"]["genuine"]
    assert (ag["n_rows"], ag["n_agree"], ag["fraction_agree"]) == (9, 6, 6 / 9)
    calls = [0]

    def noisy_base(batch):
        calls[0] += 1
        names = {n for n, _ in batch}
        rows = [list(TABLE[n]) for n, _ in batch]
        if names == {"base"} and calls[0] == 4:                            # the REPEAT (4th submission)
            rows[2] = [1, 7]
        return rows
    d = _diag(mixed=noisy_base)
    fl = d["fixed_size_repeat_floor"]
    assert fl["n_divergent_rows"] == 1 and fl["first_divergences"] == [[2, 2, 1]] and fl["fraction_divergent"] == 1 / 9


def test_diagnostics_refuse_only_a_broken_instrument():
    with pytest.raises(fid.FidelityError, match="same-size reference batch for 'base' returned 8 outputs for 9 requests"):
        _diag(mixed=lambda b: [[1, 1]] * (len(b) - 1))
    with pytest.raises(fid.FidelityError, match="every row must be a \\(token_ids, request_id\\) pair"):
        _diag(with_ids=True, mixed=lambda b: [[1, 1, 1] for _ in b])
    with pytest.raises(fid.FidelityError, match="carries no engine request id"):
        _diag(with_ids=True, mixed=lambda b: [([1, 1], "") for _ in b])
    with pytest.raises(fid.FidelityError, match="duplicate request id\\(s\\) inside one batch"):
        _diag(with_ids=True, mixed=lambda b: [([1, 1], "same") for _ in b])
    with pytest.raises(fid.FidelityError, match="already issued by an earlier batch"):
        _diag(with_ids=True, mixed=lambda b: [([1, 1], f"r{i}") for i in range(len(b))])
    with pytest.raises(fid.FidelityError, match="zero_adapter 'zero' is not among the adapters"):
        _diag(adapters=("genuine",))
    with pytest.raises(fid.FidelityError, match="empty prompt set"):
        _diag(prompt_set=())
    d = fid.multi_lora_output_diagnostics(
        base="base", adapters=("a", "b"), prompt_set=PROMPTS,
        mixed_batch=lambda batch: [{"base": [1], "a": [2], "b": [3]}[n] for n, _ in batch])
    assert d["policy"] == "legacy" and d["positional_distinguishability"]["exemption"] is None
    assert d["zero_equals_base"] is None and d["whole_layout"]["zero_vs_base"] is None


# --------------------------------------------------------------------------- #
# the control end to end                                                        #
# --------------------------------------------------------------------------- #


class _Recorder:
    """A fake worker recorder reached through arm/observe/disarm — epochs, the
    engine's ``-<8 hex>`` renaming, one step per submitted batch (mixed when
    the batch mixes selections, unless ``serial``), and the fake engine's own
    bookkeeping for every request."""

    def __init__(self, *, steps_serial=False, fail_disarm=False, drop=(), runner_kind="v2",
                 arm_without_epoch=False, answer_epoch=None, arm_answers_epoch=None,
                 fail_arm=None):
        self.log = []
        self.bound = {}                 # engine id -> adapter name | None
        self.misroute = {}
        self.steps = []
        self.steps_serial = steps_serial
        self.fail_disarm = fail_disarm
        self.drop = set(drop)
        self.runner_kind = runner_kind
        self.arm_without_epoch = arm_without_epoch
        self.answer_epoch = answer_epoch          # observe answers this epoch instead
        self.arm_answers_epoch = arm_answers_epoch  # arm answers this epoch instead
        self.fail_arm = fail_arm                  # arm raises this (after wrapping, as a half-armed group)
        self.epoch = None
        self.armed = False
        self.n_arms = 0

    def arm(self, epoch):
        self.log.append(("arm", epoch))
        if self.armed and epoch != self.epoch:
            raise RuntimeError("already_armed")
        self.n_arms += 1
        self.epoch, self.armed = epoch, True                                  # ADOPT the client's epoch
        self.bound, self.misroute, self.steps = {}, {}, []                    # a FRESH epoch
        if self.fail_arm is not None:
            raise self.fail_arm
        out = {"armed": True, "epoch": self.arm_answers_epoch or self.epoch,
               "runner_kind": self.runner_kind, "n_workers": 1}
        if self.arm_without_epoch:
            out.pop("epoch")
        return out

    def observe(self, epoch):
        self.log.append(("observe", epoch))
        if epoch != self.epoch:
            raise RuntimeError("epoch_mismatch")
        obs = {}
        for eid, name in self.bound.items():
            if eid in self.drop:
                continue
            name = self.misroute.get(eid, name)
            obs[eid] = {"lora_int_id": ID_OF[name], "lora_name": name}
        return {"armed": self.armed, "epoch": self.answer_epoch or self.epoch,
                "runner_kind": self.runner_kind, "runner_class": V2,
                "observations": obs, "inconsistent": {}, "steps": [list(s) for s in self.steps],
                "n_steps": len(self.steps), "overflow": False, "steps_overflow": False,
                "errors": [], "n_observed": len(self.bound), "n_inconsistent": 0,
                "snapshot_errors": 0, "n_workers": 1}

    def disarm(self, epoch):
        self.log.append(("disarm", epoch))
        if self.fail_disarm:
            raise RuntimeError("rpc 500")
        if epoch != self.epoch:
            raise RuntimeError("epoch_mismatch")
        self.armed = False
        return {"armed": False, "disarmed": True, "was_armed": True, "restored": True, "epoch": epoch}

    def mixed_batch(self, batch):
        out, eids = [], []
        for name, _p in batch:
            ext = f"chatcmpl-{len(self.bound) + 1:04d}"
            eid = _eng(ext)
            self.bound[eid] = SEL_ADAPTER[name]
            eids.append(eid)
            out.append((list(TABLE[name]), ext))
        if self.steps_serial:
            self.steps.extend([[e] for e in eids])
        else:
            self.steps.append(sorted(eids))
        return out


def _run(rec, **over):
    kw = dict(base="base", adapters=("genuine", "zero"), zero_adapter="zero", prompt_set=PROMPTS,
              mixed_batch=rec.mixed_batch, arm=rec.arm, observe=rec.observe, disarm=rec.disarm,
              selection_adapter=SEL_ADAPTER, attested_id_name_map=ID_NAME,
              solo=lambda n, m: list(TABLE[n]))
    kw.update(over)
    return fid.run_multi_lora_routing_control(**kw)


def test_control_mints_the_epoch_arms_it_submits_observes_checks_and_disarms():
    rec = _Recorder()
    out = _run(rec)
    assert out["passed"] is True and out["verdict"] == "routing_attested"
    ep = rec.epoch
    assert len(ep) == 32 and int(ep, 16) >= 0                              # the CLIENT's uuid4 hex
    assert rec.log == [("arm", ep), ("observe", ep), ("disarm", ep)] and out["epoch"] == ep
    assert out["n_requests"] == 63 == out["n_observed"] and out["id_join"]["n_suffixed"] == 63
    assert out["per_selection"]["base"]["n"] == 27 and out["per_selection"]["zero"]["n"] == 18
    assert out["steps"]["n_steps"] == 7 and out["steps"]["n_steps_mixed"] == 3     # the 3 rotated layouts
    assert out["steps"]["max_distinct_selections_per_step"] == 3 and out["n_foreign"] == 0
    assert out["runner_kind"] == "v2" and out["recorder"]["recorder_health"]["epoch"] == ep
    assert out["recorder"]["disarm"]["disarmed"] is True
    num = out["numerics"]
    assert num["gating"] is False and "expected_routing" not in num
    json.dumps(out)


def test_the_two_session_regression_a_serial_second_control_refuses():
    """A fully serial second control on the same worker PASSED on three
    mixed steps inherited from the standup's control. Now the second arm is
    a fresh epoch and the mixed count is over the control's own joined
    requests: it refuses as vacuous."""
    rec = _Recorder()
    assert _run(rec)["steps"]["n_steps_mixed"] == 3
    first = rec.epoch
    rec.steps_serial = True
    with pytest.raises(fid.FidelityError, match="no engine step scheduled >= 2 distinct selections of THIS control") as ei:
        _run(rec)
    assert "63 step(s) recorded, 63 with any of this control's requests, max 1 distinct" in str(ei.value)
    assert rec.n_arms == 2 and rec.epoch != first                         # a fresh client-minted epoch
    assert rec.log[-1] == ("disarm", rec.epoch)                            # disarmed its own epoch


def test_control_refuses_a_structural_misroute_and_still_disarms():
    rec = _Recorder()
    orig = rec.mixed_batch

    def mb(batch):
        out = orig(batch)
        for (name, _p), (_ids, ext) in zip(batch, out):
            if name == "genuine" and ext == "chatcmpl-0011":
                rec.misroute[_eng(ext)] = "none__lora0"
        return out
    with pytest.raises(fid.FidelityError, match=r"1/18 request\(s\) for 'genuine' ran under \['none__lora0'\]"):
        _run(rec, mixed_batch=mb)
    assert rec.log[-1] == ("disarm", rec.epoch) and rec.armed is False
    rec = _Recorder(drop={_eng("chatcmpl-0003")})
    with pytest.raises(fid.FidelityError, match="never observed"):
        _run(rec)
    assert rec.log[-1][0] == "disarm"
    rec = _Recorder(runner_kind="unknown")
    with pytest.raises(fid.FidelityError, match="not one of the pinned runners"):
        _run(rec)
    assert rec.log[-1][0] == "disarm"


def test_control_refuses_an_epochless_arm_a_foreign_epoch_and_a_failed_disarm():
    # arm answering no epoch / another epoch: refused, and the minted epoch is best-effort
    # disarmed (recorded in the refusal)
    rec = _Recorder(arm_without_epoch=True)
    with pytest.raises(fid.FidelityError, match="armed under another epoch than this control minted") as ei:
        _run(rec)
    assert "best-effort disarm with the minted epoch: ok (disarmed=True" in str(ei.value)
    assert rec.log[-1] == ("disarm", rec.epoch) and rec.armed is False
    rec = _Recorder(arm_answers_epoch="someone-elses")
    with pytest.raises(fid.FidelityError, match="armed under another epoch"):
        _run(rec)
    assert rec.log[-1][0] == "disarm" and rec.armed is False
    rec = _Recorder(answer_epoch="someone-elses")
    with pytest.raises(fid.FidelityError, match="answered for another epoch"):
        _run(rec)
    assert rec.log[-1] == ("disarm", rec.epoch)
    rec = _Recorder(fail_disarm=True)
    with pytest.raises(fid.FidelityError, match="could not be DISARMED") as ei:
        _run(rec)
    assert isinstance(ei.value.__cause__, RuntimeError)
    rec = _Recorder(fail_disarm=True, drop={_eng("chatcmpl-0001")})
    with pytest.raises(fid.FidelityError, match="never observed"):                   # the body's refusal wins
        _run(rec)


def test_control_output_disagreement_is_recorded_not_refused():
    rec = _Recorder()
    orig = rec.mixed_batch

    def flaky_zero(batch):
        out = orig(batch)
        if len({n for n, _ in batch}) == 1 and batch[0][0] == "zero":
            ids, ext = out[4]
            out[4] = ([1, 9], ext)
        return out
    out = _run(rec, mixed_batch=flaky_zero)
    assert out["passed"] is True and out["verdict"] == "routing_attested"
    assert out["numerics"]["whole_layout"]["zero_vs_base"]["n_divergent_rows"] == 1
    assert out["numerics"]["selection_agreement"]["zero"]["n_agree"] == 8


def test_an_arm_refusal_gets_a_best_effort_disarm_recorded_never_masked():
    """An arm that refused part-way (one worker armed, another refusing)
    used to leave the armed workers wrapped forever. The control disarms
    with the epoch it minted and records the outcome in the arm refusal;
    the arm refusal is what propagates (as its cause)."""
    rec = _Recorder(fail_arm=RuntimeError("HTTP 500 — worker 1 already_armed"))
    with pytest.raises(fid.FidelityError, match="arming the recorder failed \\(RuntimeError: HTTP 500") as ei:
        _run(rec)
    assert isinstance(ei.value.__cause__, RuntimeError)
    assert "best-effort disarm with the minted epoch: ok (disarmed=True, was_armed=True" in str(ei.value)
    assert rec.log[0][0] == "arm" and rec.log[-1] == ("disarm", rec.epoch) and rec.armed is False
    # ... and when even the disarm fails, both facts are in the text, the arm refusal still the cause
    rec = _Recorder(fail_arm=RuntimeError("arm boom"), fail_disarm=True)
    with pytest.raises(fid.FidelityError, match="arming the recorder failed \\(RuntimeError: arm boom\\)") as ei:
        _run(rec)
    assert "best-effort disarm with the minted epoch: failed (RuntimeError: rpc 500)" in str(ei.value)
    assert isinstance(ei.value.__cause__, RuntimeError) and str(ei.value.__cause__) == "arm boom"

