"""``fidelity.check_multi_lora_selection`` — the gate's multi-LoRA
selection check moved out of ``scripts/petri_serving_gate.py`` under two
NAMED policies.

LEGACY (``zero_adapter=None``): byte-for-byte the gate's former
``run_multi_lora_selection`` — pass; mixed-batch contamination; fewer than two
adapters; indistinguishable adapters; base among the adapters; no base; all
refusing as the gate did (same messages, so the gate's own tests keep passing
through the thin wrapper).

NATIVE (``zero_adapter=<name>``): since the gemma certification canary job
489308, ``check_multi_lora_selection`` is NO LONGER the native verdict — it
returns ``multi_lora_output_diagnostics``, a RECORD that never raises on an
output disagreement: the zero adapter differing from base, the genuine
adapter equal to base, colliding genuine adapters, a contaminated mixed row,
an undistinguishable position are all recorded numbers now (the tests below
that used to pin those REFUSALS pin the recorded numbers instead and say so).
The argument refusals still raise (no genuine adapter, ``zero_adapter`` not
among the adapters, ``base`` None, non-distinct names). The native VERDICT is
``check_multi_lora_routing_attested`` (tests/test_fidelity_routing_attested.py)
and the gate's native wrapper requires the routing recorder access; the
legacy wrapper re-raises the output verdict as ``GateError`` unchanged.

Synthetic token ids throughout; no server.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace as dataclasses_replace
from pathlib import Path

import pytest

from sisteer.capeval.serving import fidelity as fid

PROMPTS = (({"role": "user", "content": "p0"},),
           ({"role": "system", "content": "s"}, {"role": "user", "content": "p1"}),
           ({"role": "user", "content": "p2"},))


def _world(table):
    def solo(name, messages):
        return list(table[name])

    def mixed(batch):
        return [list(table[name]) for name, _ in batch]

    return solo, mixed


LEGACY_TABLE = {"base": [1, 1], "adapter_a": [2, 2], "adapter_b": [3, 3]}
NATIVE_TABLE = {"base": [1, 1], "zero": [1, 1], "genuine": [2, 2], "genuine_b": [3, 3]}


def _legacy(**over):
    solo, mixed = _world(LEGACY_TABLE)
    kw = dict(adapters=("adapter_a", "adapter_b"), solo=solo, mixed_batch=mixed,
              prompt_set=PROMPTS, base="base")
    kw.update(over)
    return kw


def _native(**over):
    solo, mixed = _world(NATIVE_TABLE)
    kw = dict(adapters=("genuine", "zero"), solo=solo, mixed_batch=mixed,
              prompt_set=PROMPTS, base="base", zero_adapter="zero")
    kw.update(over)
    return kw


# --------------------------------------------------------------------------- #
# the legacy policy (the gate's discipline, unchanged)                         #
# --------------------------------------------------------------------------- #


def test_legacy_passes_and_reports():
    diag = fid.check_multi_lora_selection(**_legacy())
    assert diag["policy"] == fid.MULTI_LORA_POLICY_LEGACY == "legacy"
    assert diag["adapters"] == ["adapter_a", "adapter_b"] == diag["genuine"]
    assert diag["zero_adapter"] is None and diag["base"] == "base"
    assert diag["n_prompts"] == 3 and diag["n_selections"] == 3 and diag["n_batch"] == 9
    assert diag["zero_equals_base"] is None


def test_legacy_contamination_fails_loudly():
    solo, _ = _world(LEGACY_TABLE)

    def contaminated(batch):
        outs = [list(LEGACY_TABLE[name]) for name, _ in batch]
        n = len(PROMPTS)
        outs[0], outs[n] = outs[n], outs[0]           # base p0 <-> adapter_a p0
        return outs

    with pytest.raises(fid.FidelityError, match="contamination"):
        fid.check_multi_lora_selection(**_legacy(mixed_batch=contaminated))
    # a server routing EVERYTHING to base: under the equal-size rule its same-size
    # references are all base's, indistinguishable — refused (never certified), and
    # the refusal names what it cannot rule out
    base_only = lambda batch: [list(LEGACY_TABLE["base"]) for _ in batch]  # noqa: E731
    with pytest.raises(fid.FidelityError, match="cannot distinguish routing") as ei:
        fid.check_multi_lora_selection(**_legacy(mixed_batch=base_only))
    assert "total cross-request adapter contamination" in str(ei.value)


def test_legacy_needs_base_plus_two_distinct_adapters():
    for adapters in (("adapter_a",), ("adapter_a", "adapter_a"), ("base", "adapter_a")):
        with pytest.raises(fid.FidelityError, match="TWO distinct adapters"):
            fid.check_multi_lora_selection(**_legacy(adapters=adapters))
    with pytest.raises(fid.FidelityError, match="explicit base"):
        fid.check_multi_lora_selection(**_legacy(base=""))
    with pytest.raises(fid.FidelityError, match="explicit base"):
        fid.check_multi_lora_selection(**_legacy(base=None))
    with pytest.raises(fid.FidelityError, match="non-empty name"):
        fid.check_multi_lora_selection(**_legacy(adapters=("adapter_a", "")))


def test_legacy_refuses_indistinguishable_routing():
    solo, mixed = _world({"base": [1, 1], "adapter_a": [1, 1], "adapter_b": [3, 3]})
    with pytest.raises(fid.FidelityError, match="cannot distinguish routing"):
        fid.check_multi_lora_selection(**_legacy(solo=solo, mixed_batch=mixed))


def test_legacy_mixed_batch_length_and_empty_prompt_set_refuse():
    with pytest.raises(fid.FidelityError, match="returned 8 outputs for 9 requests"):
        fid.check_multi_lora_selection(**_legacy(mixed_batch=lambda batch: [[1]] * (len(batch) - 1)))
    with pytest.raises(fid.FidelityError, match="empty prompt set"):
        fid.check_multi_lora_selection(**_legacy(prompt_set=()))


# --------------------------------------------------------------------------- #
# the native policy                                                            #
# --------------------------------------------------------------------------- #


def test_native_records_an_honest_engine_and_is_not_the_verdict():
    diag = fid.check_multi_lora_selection(**_native())
    assert diag["policy"] == fid.MULTI_LORA_POLICY_NATIVE == "native"
    assert diag["genuine"] == ["genuine"] and diag["zero_adapter"] == "zero"
    assert diag["adapters"] == ["genuine", "zero"] and diag["n_batch"] == 9
    assert diag["zero_equals_base"] is True and diag["signatures_distinct"] is True
    # a RECORD, not a verdict — gating False, the base repeat floor present
    assert diag["gating"] is False and "verdict" not in diag
    assert diag["fixed_size_repeat_floor"]["n_divergent_rows"] == 0
    assert diag["n_submitted_rows"] == 9 * (3 + 1 + 3)
    # two genuine adapters beside the zero adapter record too
    diag = fid.check_multi_lora_selection(**_native(adapters=("genuine", "genuine_b", "zero")))
    assert diag["genuine"] == ["genuine", "genuine_b"] and diag["n_batch"] == 12


def test_native_records_a_zero_adapter_that_differs_from_base():
    """This formerly REFUSED ("contradicts the no-op certificate"); job 489308
    showed exactly this on ONE prompt of a 33-row concurrent batch while the
    solo no-op certificate held bit for bit — batch-composition noise at
    BI=0, not a routing fact. Recorded under whole_layout.zero_vs_base now;
    the verdict is structural (test_fidelity_routing_attested.py)."""
    table = dict(NATIVE_TABLE, zero=[1, 2])
    solo, mixed = _world(table)
    diag = fid.check_multi_lora_selection(**_native(solo=solo, mixed_batch=mixed))
    zb = diag["whole_layout"]["zero_vs_base"]
    assert zb["n_prompts_differing"] == 3 and zb["n_divergent_rows"] == 9
    assert diag["zero_equals_base"] is False and diag["gating"] is False


def test_native_records_a_genuine_adapter_equal_to_base():
    """This formerly REFUSED as indistinguishable routing (job 489308 changed
    the instrument, not the fact: an inert adapter is now caught by the
    structural check reading its requests under the adapter's lora id — and
    by steering_bites before it). Recorded under whole_layout and
    positional_distinguishability."""
    table = dict(NATIVE_TABLE, genuine=[1, 1])
    solo, mixed = _world(table)
    diag = fid.check_multi_lora_selection(**_native(solo=solo, mixed_batch=mixed))
    assert diag["whole_layout"]["genuine_like_base"] == ["genuine"]
    assert diag["whole_layout"]["colliding"] == ["base", "genuine", "zero"]
    assert diag["signatures_distinct"] is False
    assert diag["positional_distinguishability"]["n_undistinguishable"] == 9 * 4


def test_native_records_colliding_genuine_adapters():
    """This formerly REFUSED; recorded under whole_layout.colliding now (job
    489308 demotion)."""
    table = dict(NATIVE_TABLE, genuine_b=[2, 2])
    solo, mixed = _world(table)
    diag = fid.check_multi_lora_selection(**_native(adapters=("genuine", "genuine_b", "zero"),
                                                    solo=solo, mixed_batch=mixed))
    assert diag["whole_layout"]["colliding"] == ["genuine", "genuine_b"]
    assert diag["whole_layout"]["genuine_like_base"] == []
    assert diag["positional_distinguishability"]["n_undistinguishable"] == 12 * 2   # both ways, 12 positions


def test_native_refuses_no_genuine_adapter():
    with pytest.raises(fid.FidelityError, match="no genuine adapter beside the zero adapter"):
        fid.check_multi_lora_selection(**_native(adapters=("zero",)))


def test_native_refuses_a_zero_adapter_outside_the_selection_or_no_base():
    with pytest.raises(fid.FidelityError, match="zero_adapter 'zero' is not among the adapters"):
        fid.check_multi_lora_selection(**_native(adapters=("genuine",)))
    with pytest.raises(fid.FidelityError, match="explicit base"):
        fid.check_multi_lora_selection(**_native(base=None))
    with pytest.raises(fid.FidelityError, match="zero_adapter must be a non-empty name"):
        fid.check_multi_lora_selection(**_native(zero_adapter=""))
    with pytest.raises(fid.FidelityError, match="distinct names not including the base"):
        fid.check_multi_lora_selection(**_native(adapters=("genuine", "zero", "base")))
    with pytest.raises(fid.FidelityError, match="distinct names not including the base"):
        fid.check_multi_lora_selection(**_native(adapters=("genuine", "zero", "zero")))


def test_native_records_genuine_zero_contamination_in_the_mixed_batch():
    """Formerly the genuine <-> zero and genuine <-> base swaps REFUSED as
    contamination and the all-base server as indistinguishable; since job
    489308 an output disagreement in a concurrent batch is RECORDED here and
    the same servers refuse STRUCTURALLY (a swapped request is observed under
    the other adapter's lora id; an all-base server shows every genuine
    request under id 0 — test_fidelity_routing_attested.py)."""
    solo, _ = _world(NATIVE_TABLE)

    def swap(a, b):
        # the two selections' prompt-0 rows exchanged, wherever the (rotated) layout
        # put them; a homogeneous reference batch holds neither pair, so it is untouched
        def mixed(batch):
            outs = [list(NATIVE_TABLE[name]) for name, _ in batch]
            ia = [k for k, (name, m) in enumerate(batch) if name == a and m is PROMPTS[0]]
            ib = [k for k, (name, m) in enumerate(batch) if name == b and m is PROMPTS[0]]
            if ia and ib:
                outs[ia[0]], outs[ib[0]] = outs[ib[0]], outs[ia[0]]
            return outs
        return mixed

    for a, b in (("genuine", "zero"), ("genuine", "base")):
        diag = fid.check_multi_lora_selection(**_native(mixed_batch=swap(a, b)))
        assert diag["n_mismatched_rows"] == 2 * 3            # both swapped rows, in every layout
        assert diag["selection_agreement"]["genuine"]["n_agree"] == 6
        assert diag["selection_agreement"][b]["n_agree"] == 6
    # zero <-> base: identical outputs, nothing to record
    diag = fid.check_multi_lora_selection(**_native(mixed_batch=swap("zero", "base")))
    assert diag["n_mismatched_rows"] == 0
    # an all-base server: recorded as genuine_like_base + undistinguishable, never raised
    diag = fid.check_multi_lora_selection(**_native(mixed_batch=lambda batch: [[1, 1] for _ in batch]))
    assert diag["whole_layout"]["genuine_like_base"] == ["genuine"]
    assert diag["positional_distinguishability"]["n_undistinguishable"] == 9 * 4


def test_the_two_policies_differ_exactly_where_intended():
    """The native inputs under the LEGACY policy refuse both ways: with the
    zero included they are indistinguishable from base; with it excluded
    there is only one adapter."""
    solo, mixed = _world(NATIVE_TABLE)
    with pytest.raises(fid.FidelityError, match="cannot distinguish routing"):
        fid.check_multi_lora_selection(adapters=("genuine", "zero"), solo=solo, mixed_batch=mixed,
                                       prompt_set=PROMPTS, base="base")
    with pytest.raises(fid.FidelityError, match="TWO distinct adapters"):
        fid.check_multi_lora_selection(adapters=("genuine",), solo=solo, mixed_batch=mixed,
                                       prompt_set=PROMPTS, base="base")
    # and the legacy world under the native policy: naming a genuine adapter as the
    # zero is RECORDED as zero != base (it formerly refused "contradicts the no-op
    # certificate"; job 489308 demoted every output comparison of the native policy)
    solo, mixed = _world(LEGACY_TABLE)
    diag = fid.check_multi_lora_selection(adapters=("adapter_a", "adapter_b"), solo=solo, mixed_batch=mixed,
                                          prompt_set=PROMPTS, base="base", zero_adapter="adapter_b")
    assert diag["zero_equals_base"] is False and diag["whole_layout"]["zero_vs_base"]["n_prompts_differing"] == 3


def test_token_ids_are_compared_as_ints_and_no_generated_ids_leak_into_the_report():
    solo, mixed = _world(NATIVE_TABLE)
    diag = fid.check_multi_lora_selection(**_native(
        solo=lambda name, m: tuple(solo(name, m)),                # tuples vs lists: equal by value
        mixed_batch=lambda batch: [tuple(o) for o in mixed(batch)]))
    assert all(not isinstance(v, (list, tuple)) or all(isinstance(x, str) for x in v)
               for v in diag.values())


# --------------------------------------------------------------------------- #
# the gate's thin wrapper                                                      #
# --------------------------------------------------------------------------- #


def _gate():
    if "_petri_serving_gate" in sys.modules:
        return sys.modules["_petri_serving_gate"]
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("_petri_serving_gate",
                                                  root / "scripts" / "petri_serving_gate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_petri_serving_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_gate_wrapper_defaults_to_legacy_and_re_raises_as_gate_error():
    gate = _gate()
    solo, mixed = _world(LEGACY_TABLE)
    gate.run_multi_lora_selection(base="base", adapters=("adapter_a", "adapter_b"),
                                  mixed_batch=mixed, solo=solo)
    # one adapter: the legacy refusal, as a GateError carrying the fidelity message
    with pytest.raises(gate.GateError, match="TWO distinct adapters") as g:
        gate.run_multi_lora_selection(base="base", adapters=("adapter_a",), mixed_batch=mixed, solo=solo)
    with pytest.raises(fid.FidelityError) as f:
        fid.check_multi_lora_selection(adapters=("adapter_a",), solo=solo, mixed_batch=mixed,
                                       prompt_set=gate.GATE_PROMPT_SET, base="base")
    assert str(g.value) == str(f.value)
    assert isinstance(g.value.__cause__, fid.FidelityError)


def _routing_access(gate, table, *, misroute=None, serial=False):
    """A fake recorder + id-minting mixed batch for the gate's native wrapper:
    epochs, the engine's ``-<8 hex>`` renaming, one step per batch (``serial``
    = one request per step), the engine binding each request to its
    selection's adapter (``misroute``: {label: adapter name} rebinds every
    request of that selection)."""
    import hashlib
    sel_adapter = {"base": None, "zero": "none__lora0", "genuine": "cell_a"}
    id_of = {None: 0, "none__lora0": 1, "cell_a": 2}
    bound, steps, log = {}, [], []
    state = {"epoch": None}

    def mixed_batch(batch):
        out, eids = [], []
        for name, _p in batch:
            ext = f"chatcmpl-{len(bound) + 1:04d}"
            eid = f"{ext}-{hashlib.sha256(ext.encode()).hexdigest()[:8]}"
            bound[eid] = (misroute or {}).get(name, sel_adapter[name])
            eids.append(eid)
            out.append((list(table[name]), ext))
        steps.extend([[e] for e in eids] if serial else [sorted(eids)])
        return out

    def arm(epoch):
        log.append(("arm", epoch))
        state["epoch"] = epoch                                       # ADOPT the control's epoch
        bound.clear()
        del steps[:]
        return {"armed": True, "epoch": epoch, "already_armed": False, "runner_kind": "v2"}

    def observe(epoch):
        log.append(("observe", epoch))
        return {"armed": True, "epoch": epoch, "runner_kind": "v2",
                "runner_class": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
                "observations": {e: {"lora_int_id": id_of[b], "lora_name": b} for e, b in bound.items()},
                "inconsistent": {}, "steps": [list(s) for s in steps], "n_steps": len(steps),
                "overflow": False, "steps_overflow": False, "errors": []}
    return gate.LoraRoutingRecorderAccess(
        arm=arm, observe=observe, disarm=lambda epoch: log.append(("disarm", epoch)) or {"disarmed": True},
        mixed_batch=mixed_batch, selection_adapter=sel_adapter,
        id_name_map={"1": "none__lora0", "2": "cell_a"}), log


def test_gate_wrapper_threads_the_native_policy_through_the_structural_path():
    """The gate's NATIVE wrapper needs the routing recorder access and rules
    on the engine's bookkeeping; its output comparisons are recorded.
    A native probe WITHOUT recorder access refuses (an output comparison alone
    cannot certify selection at concurrency > 1 — job 489308)."""
    gate = _gate()
    solo, mixed = _world(NATIVE_TABLE)
    with pytest.raises(gate.GateError, match="needs the worker's LoRA routing recorder"):
        gate.run_multi_lora_selection(base="base", adapters=("genuine", "zero"), mixed_batch=mixed,
                                      solo=solo, zero_adapter="zero")
    routing, log = _routing_access(gate, NATIVE_TABLE)
    rec = gate.run_multi_lora_selection(base="base", adapters=("genuine", "zero"), mixed_batch=mixed,
                                        solo=solo, zero_adapter="zero", routing=routing)
    P = len(gate.GATE_PROMPT_SET)
    assert rec["passed"] is True and rec["verdict"] == "routing_attested"
    ep = rec["epoch"]
    assert len(ep) == 32 and rec["n_requests"] == 3 * P * 7
    assert log == [("arm", ep), ("observe", ep), ("disarm", ep)]         # the control's own epoch throughout
    assert rec["id_join"]["n_suffixed"] == rec["n_requests"]
    assert rec["numerics"]["gating"] is False and rec["numerics"]["n_batch"] == 3 * P
    # the OUTPUT-level deviation that formerly refused ("contradicts the no-op
    # certificate") is recorded, the structural verdict passes
    bad_solo, bad_mixed = _world(dict(NATIVE_TABLE, zero=[9, 9]))
    routing, _ = _routing_access(gate, dict(NATIVE_TABLE, zero=[9, 9]))
    rec = gate.run_multi_lora_selection(base="base", adapters=("genuine", "zero"), mixed_batch=bad_mixed,
                                        solo=bad_solo, zero_adapter="zero", routing=routing)
    assert rec["passed"] is True and rec["numerics"]["zero_equals_base"] is False
    # a STRUCTURAL misroute (every zero request bound to no LoRA) refuses through the wrapper
    routing, log = _routing_access(gate, NATIVE_TABLE, misroute={"zero": None})
    with pytest.raises(gate.GateError, match=r"request\(s\) for 'zero' ran under \['<no LoRA>'\] instead of 'none__lora0'") as ei:
        gate.run_multi_lora_selection(base="base", adapters=("genuine", "zero"), mixed_batch=mixed,
                                      solo=solo, zero_adapter="zero", routing=routing)
    assert isinstance(ei.value.__cause__, fid.FidelityError) and log[-1][0] == "disarm"
    # a fully serial control: no step of its own mixes two selections — refused as vacuous
    routing, _ = _routing_access(gate, NATIVE_TABLE, serial=True)
    with pytest.raises(gate.GateError, match="no engine step scheduled >= 2 distinct selections of THIS control"):
        gate.run_multi_lora_selection(base="base", adapters=("genuine", "zero"), mixed_batch=mixed,
                                      solo=solo, zero_adapter="zero", routing=routing)
    # a mixed batch of the wrong length still names its size (a broken instrument)
    n = len(gate.GATE_PROMPT_SET) * 3
    routing, _ = _routing_access(gate, NATIVE_TABLE)
    full_mb = routing.mixed_batch
    routing = dataclasses_replace(routing, mixed_batch=lambda batch: full_mb(batch)[:-1])
    with pytest.raises(gate.GateError, match=f"returned {n - 1} outputs for {n} requests"):
        gate.run_multi_lora_selection(base="base", adapters=("genuine", "zero"), mixed_batch=mixed,
                                      solo=solo, zero_adapter="zero", routing=routing)


# --------------------------------------------------------------------------- #
# THE EQUAL-SIZE RULE (job 489125)                                              #
# --------------------------------------------------------------------------- #


def _size_dependent(table):
    """A CPU fake engine that is deterministic WITHIN a batch size but differs
    ACROSS sizes — the shape of interleaved serving at BI=0 (a batch of 1 and
    a batch of N take different GEMM reduction schedules): every output
    carries the batch size as its last token."""
    def solo(name, messages):
        return list(table[name]) + [1]

    def mixed(batch):
        return [list(table[name]) + [len(batch)] for name, _ in batch]

    return solo, mixed


def test_the_equal_size_rule_passes_a_batch_size_dependent_engine_and_records_the_solo_divergence():
    """Job 489125 (gemma certification canary): the control compared each
    mixed row against that request's SOLO output; at BI=0 base's row on
    prompt 9 flipped a token between a batch of 1 and a batch of 15 and the
    control read it as contamination. The module's own check 5 states the
    equal-size rule; the check now follows it — the size-dependent engine
    PASSES (its same-size references agree with its mixed rows) while the
    solo divergence, which the old rule failed on, is RECORDED (check 4) —
    never the verdict's evidence."""
    solo, mixed = _size_dependent(NATIVE_TABLE)
    diag = fid.check_multi_lora_selection(**_native(solo=solo, mixed_batch=mixed))
    assert diag["rule"] == "equal-size" and diag["policy"] == "native"
    assert diag["n_batch"] == 9 and diag["n_selections"] == 3 and diag["selection_first_divergences"] == []
    assert diag["n_layouts"] == 3 and diag["n_mixed_rows"] == 27 and "rotated" in diag["layouts"]
    pd = diag["positional_distinguishability"]
    assert pd["n_undistinguishable"] == 0 and pd["exemption"] == "base <-> zero adapter"
    assert pd["n_pairs_checked"] == 9 * (3 * 2 - 2)          # every position x ordered pair, base<->zero exempt both ways
    assert "same-size single-selection batches" in diag["reference"]
    bi = diag["batch_invariance"]
    assert "recorded only" in bi["note"] and set(bi["solo_vs_same_size"]) == {"base", "genuine", "zero"}
    for name, rec in bi["solo_vs_same_size"].items():
        # every solo differs from its same-size row at the size token (index 2)
        assert (rec["n"], rec["n_divergent"], rec["first_divergences"]) == (3, 3, [[0, 2], [1, 2], [2, 2]]), name
    assert all(rec["n_divergent"] == 0 and rec["n_pairs"] == 6 for rec in bi["within_batch_repeats"].values())
    # the OLD rule — row vs solo — would have failed every row of this engine
    for name in ("base", "genuine", "zero"):
        assert mixed([(name, PROMPTS[0])] * 9)[0] != solo(name, PROMPTS[0])
    # the legacy policy follows the same rule
    lsolo, lmixed = _size_dependent(LEGACY_TABLE)
    ldiag = fid.check_multi_lora_selection(**_legacy(solo=lsolo, mixed_batch=lmixed))
    assert ldiag["rule"] == "equal-size" and ldiag["policy"] == "legacy"
    assert all(rec["n_divergent"] == 3 for rec in ldiag["batch_invariance"]["solo_vs_same_size"].values())
    # solo=None skips the recorded diagnostic and nothing else changes
    nd = fid.check_multi_lora_selection(**_native(solo=None, mixed_batch=mixed))
    assert nd["batch_invariance"]["solo_vs_same_size"] == {} and nd["n_batch"] == 9
    # a deterministic engine records zero divergence everywhere
    dsolo, dmixed = _world(NATIVE_TABLE)
    dd = fid.check_multi_lora_selection(**_native(solo=dsolo, mixed_batch=dmixed))
    assert all(rec["n_divergent"] == 0 for rec in dd["batch_invariance"]["solo_vs_same_size"].values())


def test_a_mixed_row_output_misroute_is_recorded_by_position_under_the_native_policy():
    """This formerly FAILED the native control naming the request, position
    and first divergent token; job 489308 showed a concurrent batch's output
    disagreeing for a non-routing reason, so under the native policy the same
    numbers are RECORDED (`selection_first_divergences`, per-selection
    agreement) and the verdict is the engine's bookkeeping. The LEGACY policy
    keeps the output verdict (test_legacy_* above and the gate's tests)."""
    def misroute(batch):
        outs = [list(NATIVE_TABLE[n]) for n, _ in batch]
        if len({n for n, _ in batch}) > 1:
            k = [i for i, (n, _) in enumerate(batch) if n == "genuine"][1]      # genuine, prompt 1
            outs[k] = list(NATIVE_TABLE["base"])
        return outs
    diag = fid.check_multi_lora_selection(**_native(mixed_batch=misroute))
    # the misroute recurs in every rotated layout (genuine's prompt-1 row moves with it)
    assert diag["selection_first_divergences"] == [[0, 4, "genuine", 1, 0], [1, 1, "genuine", 1, 0],
                                                   [2, 7, "genuine", 1, 0]]
    assert diag["n_mismatched_rows"] == 3 and diag["selection_agreement"]["genuine"]["n_agree"] == 6
    # on a batch-size-dependent engine the same row is recorded the same way
    solo, _ = _size_dependent(NATIVE_TABLE)

    def both(batch):
        outs = [list(NATIVE_TABLE[n]) + [len(batch)] for n, _ in batch]
        if len({n for n, _ in batch}) > 1:
            outs[4] = list(NATIVE_TABLE["base"]) + [len(batch)]
        return outs
    diag = fid.check_multi_lora_selection(**_native(solo=solo, mixed_batch=both))
    # position 4 is genuine's only in layout 0 (zero's in layout 1 reads base == zero; base's in 2)
    assert diag["n_mismatched_rows"] == 1 and diag["selection_first_divergences"] == [[0, 4, "genuine", 1, 0]]
    # a zero row served the genuine adapter's output in the mixed batch only
    def zero_contaminated(batch):
        outs = [list(NATIVE_TABLE[n]) for n, _ in batch]
        if len({n for n, _ in batch}) > 1:
            k = [i for i, (n, _) in enumerate(batch) if n == "zero"][0]
            outs[k] = list(NATIVE_TABLE["genuine"])
        return outs
    diag = fid.check_multi_lora_selection(**_native(mixed_batch=zero_contaminated))
    assert diag["selection_first_divergences"][0] == [0, 6, "zero", 0, 0]
    # a same-size reference batch of the wrong length still names itself (a broken instrument)
    with pytest.raises(fid.FidelityError, match="same-size reference batch for 'base' returned 8 outputs for 9 requests"):
        fid.check_multi_lora_selection(**_native(mixed_batch=lambda b: [[1, 1]] * (len(b) - 1)))


def test_a_selection_must_be_distinguishable_at_every_position_it_is_compared_on():
    """A counterexample: within-batch variation is permitted, so a genuine
    adapter distinct over its WHOLE layout could still equal base exactly
    where the mixed batch ran it — one prompt, three positions,
    ``ref[base] = ref[zero] = [[10],[10],[10]]``, ``ref[genuine] =
    [[20],[10],[10]]``, an all-base mixed batch ``[[10],[10],[10]]``: genuine
    sat at position 1 where its reference equals base's, and every request
    routed to base PASSED through the helper and both wrappers. Now every
    (selection, position) a rotated layout compares must differ from every
    prohibited substitute there, else the check refuses as vacuous naming the
    row; the honest size-dependent fake still passes and a real misroute
    still fails by name."""
    one_prompt = (PROMPTS[0],)

    def all_base_mixed(refs):
        def mixed(batch):
            names = {n for n, _ in batch}
            if len(names) == 1:                                   # a homogeneous same-size reference
                return [list(x) for x in refs[next(iter(names))]]
            return [[10] for _ in batch]                          # every request routed to base
        return mixed

    # the native wrapper's shape: base, genuine, zero — the native policy now RECORDS
    # the undistinguishable positions (it formerly refused as vacuous; job 489308) and the
    # all-base server refuses STRUCTURALLY instead (its genuine requests run under lora id
    # 0 — test_fidelity_routing_attested.py::test_astras_all_base_counterexample_refuses_structurally)
    native_refs = {"base": [[10], [10], [10]], "zero": [[10], [10], [10]], "genuine": [[20], [10], [10]]}
    diag = fid.check_multi_lora_selection(adapters=("genuine", "zero"), solo=None,
                                          mixed_batch=all_base_mixed(native_refs), prompt_set=one_prompt,
                                          base="base", zero_adapter="zero")
    pd = diag["positional_distinguishability"]
    # positions 1 and 2: base<->genuine both ways and genuine<->zero both ways (zero is exempt
    # against base only), 4 ordered pairs x 2 positions
    assert pd["n_undistinguishable"] == 8 and pd["n_pairs_checked"] == 3 * 4
    assert pd["first"] == [[1, "base", "genuine"], [1, "genuine", "base"], [1, "genuine", "zero"], [1, "zero", "genuine"],
                           [2, "base", "genuine"], [2, "genuine", "base"], [2, "genuine", "zero"], [2, "zero", "genuine"]]
    assert diag["selection_agreement"]["genuine"]["n_agree"] == 2                # position 0's row disagreed
    # the legacy wrapper's shape: base + two adapters, each distinguishing only at its own first position
    legacy_refs = {"base": [[10], [10], [10]], "adapter_a": [[20], [10], [10]], "adapter_b": [[30], [10], [10]]}
    with pytest.raises(fid.FidelityError, match="cannot distinguish routing") as ei2:
        fid.check_multi_lora_selection(adapters=("adapter_a", "adapter_b"), solo=None,
                                       mixed_batch=all_base_mixed(legacy_refs), prompt_set=one_prompt, base="base")
    assert "at batch position 1 (gate prompt 0)" in str(ei2.value) and "refuses as vacuous" in str(ei2.value)
    # ... and through the gate's two wrappers (the frozen GATE_PROMPT_SET: 6 prompts, 18 positions)
    gate = _gate()
    P = len(gate.GATE_PROMPT_SET)

    def gate_refs(names):
        # every selection equals base except at its own first block: distinct over the whole
        # layout, identical to base at every position another selection's block occupies
        refs = {n: [[10]] * (3 * P) for n in names}
        for j, n in enumerate(names):
            if n != "base":
                rows = [[10]] * (3 * P)
                rows[j * P] = [20 + j]
                refs[n] = rows
        return refs

    # the native wrapper: the all-base server refuses STRUCTURALLY — every genuine and
    # zero request observed under lora id 0 — and the undistinguishable positions are recorded
    all_base_table = {"base": [10], "genuine": [10], "zero": [10]}
    routing, _log = _routing_access(gate, all_base_table, misroute={"genuine": None, "zero": None})
    with pytest.raises(gate.GateError, match=r"request\(s\) for 'genuine' ran under \['<no LoRA>'\] instead of 'cell_a'"):
        gate.run_multi_lora_selection(base="base", adapters=("genuine", "zero"),
                                      mixed_batch=all_base_mixed(dict(gate_refs(["base", "genuine", "zero"]),
                                                                      zero=[[10]] * (3 * P))),
                                      solo=None, zero_adapter="zero", routing=routing)
    with pytest.raises(gate.GateError, match="cannot distinguish routing"):
        gate.run_multi_lora_selection(base="base", adapters=("adapter_a", "adapter_b"),
                                      mixed_batch=all_base_mixed(gate_refs(["base", "adapter_a", "adapter_b"])),
                                      solo=None)
    # the OLD rule (block-wise layout, whole-layout guard) accepted exactly this: replaying it
    # here pins that the refusal above is what changed
    refs = native_refs
    layout0 = [("base", 0), ("genuine", 0), ("zero", 0)]
    old_pass = all(all_base_mixed(refs)([(n, PROMPTS[0]) for n, _ in layout0])[k] == refs[n][k]
                   for k, (n, _) in enumerate(layout0))
    assert old_pass and tuple(refs["genuine"]) != tuple(refs["base"])
    # the honest deterministic-within-batch-size fake passes the positional rule everywhere
    solo, mixed = _size_dependent(NATIVE_TABLE)
    diag = fid.check_multi_lora_selection(**_native(solo=solo, mixed_batch=mixed))
    assert diag["positional_distinguishability"]["n_undistinguishable"] == 0 and diag["n_layouts"] == 3
    # a selection distinguishable at every position but misrouted (in output) in one row: recorded
    def misroute(batch):
        outs = [list(NATIVE_TABLE[n]) for n, _ in batch]
        if len({n for n, _ in batch}) > 1:
            outs[[k for k, (n, _) in enumerate(batch) if n == "genuine"][0]] = list(NATIVE_TABLE["base"])
        return outs
    diag = fid.check_multi_lora_selection(**_native(mixed_batch=misroute))
    assert diag["n_mismatched_rows"] == 3 and diag["positional_distinguishability"]["n_undistinguishable"] == 0
