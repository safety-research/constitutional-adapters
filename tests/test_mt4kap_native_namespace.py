"""The mt4kap (arditi_allpos multi-turn) grades roots are read under their OWN native serving namespace -- not NS, not NS_RELU.

WHY A NAMESPACE AT ALL. ``scripts/r2_export/cap4096/build_transfer_stats_cap4096.py`` reads every canonical root under
``NAMESPACE_SERVING[ns]``. The lens namespace ``NS`` declares ``engine lens`` / ``lens_reduction_mode "stacked"``, and
``read_cell`` refuses an aggregate row whose ``lens_reduction_mode`` differs. The mt4kap roots are NATIVELY served
(``serving_engine lora_native``, ``lens_reduction_mode null`` on every one of 48 shards), so under ``NS`` every one of
their cell rows refuses and the whole lane build stops. Placing the root
without a declaration is therefore worse than not placing it.

WHY NOT NS_RELU'S ENTRY. Four of its five terms are right for this store; one is wrong. ``merged_store_windows True``
describes relu_native's two-root merged stores (a base root killed at standup + its restart, stood up at different
windows). mt4kap is a single-root store with a uniform window per cell. Under ``True`` the per-shard identity key
records the literal string ``"per shard (serving_windows_by_shard)"`` instead of the real window, and a store whose
shards were served at two windows is silently ADMITTED as one reading (zero clamps); under ``False`` the window rides
the identity tuple and a split REFUSES. ``decode_fallback timing_rows`` is the right value for a different reason than
relu's: every mt4kap sidecar is finalised, so that branch never falls back -- it cross-checks ``last_request_decode``
against the rows' own decode and refuses on disagreement, which is strictly stronger than ``declared_block``.

Each test carries its KNOWN-BAD: the clone with ``True`` admitting a window split, a stacked lens row under the native
namespace, a lens-engine shard, a decode disagreement, a non-mt4kap root that must stay on ``NS``. The builder is loaded
by AST (its module-level import chain reaches numpy and two out-of-repo state trees, none of which this depends on).
Every store here is synthetic: invented windows and decodes under ``tmp_path``, no run-state read.
"""

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
BUILDER = ROOT / "scripts" / "r2_export" / "cap4096" / "build_transfer_stats_cap4096.py"

NATIVE_NS = "arditi_allpos_native"
LENS_NS = "cap4096"
#: the declaration, term by term -- a later edit that "harmonises" it with NS_RELU's fails the first test
EXPECTED_SERVING = {"engine": "lora_native", "lens_reduction_mode": None, "decode_fallback": "timing_rows",
                    "record_engine": True, "merged_store_windows": False}
DECODE = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": None, "note": "synthetic sampler"}


# --------------------------------------------------------------------------- loading the builder by AST

def _tree():
    return ast.parse(BUILDER.read_text())


def _assign_nodes(names):
    """the MODULE-LEVEL assignments to any of `names`, in file order (tuple targets such as `NS, NS_VARIANT = ...` included)"""
    out = []
    for node in _tree().body:
        if not isinstance(node, ast.Assign):
            continue
        targets = set()
        for t in node.targets:
            if isinstance(t, ast.Name):
                targets.add(t.id)
            elif isinstance(t, ast.Tuple):
                targets |= {e.id for e in t.elts if isinstance(e, ast.Name)}
        if targets & set(names):
            out.append(node)
    return out


def _globals():
    """the namespace declarations exec'd out of the builder with only the globals they use"""
    ns = {"WAVE_LENS_MODE": "stacked", "CAP": 4096, "CAP_FILED": 512, "json": json, "Path": Path}
    nodes = _assign_nodes(["NS", "NS_VARIANT", "NS_RELU", "NS_ALLPOS_NATIVE", "NAMESPACE_SERVING", "DECODE_KEYS",
                           "ARM_ROOT_PREFIX", "ARM_ROOT_NAMESPACE"])
    exec(compile(ast.Module(nodes, []), str(BUILDER), "exec"), ns)  # noqa: S102
    return ns


def _with_funcs(ns, *names):
    nodes = [n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in nodes}
    assert not missing, f"not found in the builder: {sorted(missing)}"
    exec(compile(ast.Module(nodes, []), str(BUILDER), "exec"), ns)  # noqa: S102
    return ns


@pytest.fixture(scope="module")
def G():
    return _globals()


# --------------------------------------------------------------------------- 1. the declaration itself

def test_the_native_namespace_is_declared_with_exactly_these_five_terms(G):
    assert "NS_ALLPOS_NATIVE" in G, "no NS_ALLPOS_NATIVE constant: mt4kap roots would be read under the lens namespace and refuse"
    assert G["NS_ALLPOS_NATIVE"] == NATIVE_NS
    assert G["NAMESPACE_SERVING"][NATIVE_NS] == EXPECTED_SERVING


def test_it_differs_from_relu_native_in_exactly_merged_store_windows(G):
    """The one term that makes this its own namespace. A clone of NS_RELU's entry fails here; so does a drift on any other term."""
    mine, relu = G["NAMESPACE_SERVING"][NATIVE_NS], G["NAMESPACE_SERVING"][G["NS_RELU"]]
    assert mine["merged_store_windows"] is False, "a single-root store must keep its real window in the identity tuple"
    assert {k for k in EXPECTED_SERVING if mine[k] != relu[k]} == {"merged_store_windows"}


# --------------------------------------------------------------------------- 2. routing: only mt4kap roots leave NS

@pytest.mark.parametrize("root, expected", [
    ("grades_mt4kap_ch_qwen35_4b_it", NATIVE_NS),
    ("grades_mt4kap_bu_qwen35_4b_it", NATIVE_NS),
    ("grades_mt4kapr2_ch_qwen35_4b_it", NATIVE_NS),          # a retry namespace routes with its arm, by the rule that enumerates it
    ("grades_mt4kar_bu_gemma4e4b_it", LENS_NS),              # the prefill-only Arditi arm: a DIFFERENT arm, lens-served
    ("grades_mt4karr6_ch_llama8b_it", LENS_NS),
    ("grades_mt4kas_bu_qwen35_4b_it", LENS_NS),              # AlphaSteer
    ("grades_mt4kasr4_ch_qwen35_4b_it", LENS_NS),
    ("grades_mt4ksc_ch_qwen35_4b_it", LENS_NS),              # the constitution-SI arm (not an ARM_ROOT_PREFIX token at all)
    ("grades_cap4096_gemma4e4b", LENS_NS),
    ("grades_anchor4096_qwen35_4b_it", LENS_NS),
    ("grades_obj4096_qwen35_4b_it", LENS_NS),
])
def test_only_mt4kap_roots_route_to_the_native_namespace(G, root, expected):
    ns = _with_funcs(dict(G), "namespace_of_root")
    assert ns["namespace_of_root"](root) == expected


def test_the_routing_reaches_its_consumers_not_just_a_table():
    """A declaration nothing calls is inert. The canonical loop and the landing label must both go through namespace_of_root."""
    src = BUILDER.read_text()
    assert re.search(r"^\s*process_root\(r, namespace_of_root\(r\), None\)", src, re.M), "the canonical loop does not route by namespace_of_root"
    assert not re.search(r"^\s*process_root\(r, NS, None\)", src, re.M), "a canonical process_root(r, NS, None) call survives: mt4kap would still refuse"
    assert re.search(r'"namespace": \(NS_VARIANT if v else namespace_of_root\(r\)\)', src), "meta.landing_status would label an mt4kap root with the lens namespace"


def test_the_namespace_is_declared_in_meta_and_in_the_top_level_namespaces_block():
    src = BUILDER.read_text()
    assert re.search(r'NS_ALLPOS_NATIVE: \{"label":', src), "meta.namespaces carries no entry for the native namespace"
    assert re.search(r'NS_ALLPOS_NATIVE: \{"models":', src), "the top-level namespaces block carries no entry for the native namespace"


# --------------------------------------------------------------------------- 3. read_cell's identity gates, per namespace

class _ReachedVerifyCap(Exception):
    """raised by the verify_cap stub: reaching it proves the row passed every identity gate that precedes the disk reads"""


def _read_cell(G):
    ns = dict(G)

    class _T1:
        gated_cell = staticmethod(lambda agg, cell: {"k": 1, "n": 2})

    class _R2:
        _safe = staticmethod(lambda f, agg, cell: f(agg, cell))

    def _verify_cap(cell_dir, completion):
        raise _ReachedVerifyCap()

    ns.update(T1=_T1, R2=_R2, verify_cap=_verify_cap)
    return _with_funcs(ns, "read_cell")["read_cell"]


def _agg(lens_mode, engine="lora_native"):
    return {"serving_engine": engine, "require_engine": engine,
            "cells": {"arditi_allpos": {"lens_reduction_mode": lens_mode, "serving_engine": engine, "decode_cap_tokens": 4096,
                                        "completion": {"gated": {"n_error_outcomes": 0, "n_responses": 3}},
                                        "attack_metrics": {"n_expected": 2, "n_graded": 2, "n_errored": 0}}}}


@pytest.mark.parametrize("namespace, lens_mode, outcome", [
    (NATIVE_NS, "stacked", "refuse"),      # THE KNOWN-BAD: a lens-era row is not admitted into the native namespace
    (NATIVE_NS, None, "admit"),            # the mt4kap rows as graded: lens_reduction_mode null, native stamps
    (LENS_NS, "stacked", "admit"),         # the lens namespace still admits its own rows ...
    (LENS_NS, None, "refuse"),             # ... and still refuses a native row (unchanged)
])
def test_read_cell_admits_a_row_only_into_the_namespace_that_declares_its_lens_mode(G, tmp_path, namespace, lens_mode, outcome):
    read_cell = _read_cell(G)
    agg = _agg(lens_mode, engine=("lora_native" if namespace == NATIVE_NS else "lens"))
    if outcome == "refuse":
        with pytest.raises(SystemExit) as e:
            read_cell(agg, "arditi_allpos", "pair", namespace, tmp_path)
        assert "lens_reduction_mode" in str(e.value)
    else:
        with pytest.raises(_ReachedVerifyCap):
            read_cell(agg, "arditi_allpos", "pair", namespace, tmp_path)


def test_read_cell_under_the_native_namespace_refuses_an_aggregate_stamped_with_another_engine(G, tmp_path):
    """record_engine True binds the aggregate's serving_engine / require_engine to the declared engine."""
    with pytest.raises(SystemExit) as e:
        _read_cell(G)(_agg(None, engine="lens"), "arditi_allpos", "pair", NATIVE_NS, tmp_path)
    assert "serving_engine" in str(e.value)


# --------------------------------------------------------------------------- 4. instrument(): the window term on a single-root store

def _shard(cell_dir, name, window, *, last_request_decode=DECODE, timing_decode=None, lens_mode=None, engine="lora_native"):
    sd = cell_dir / name
    sd.mkdir(parents=True)
    (sd / "served_target.json").write_text(json.dumps({
        "serving_engine": engine, "serving_identity": "proxy_vllm+lora+interleaved", "max_model_len": window, "manifest_stage": "final",
        "last_request_decode": last_request_decode, "proxy_identity": {"lens_reduction_mode": lens_mode}, "target_decode_note": None,
        "window_clamp": {"n_clamped": 0, "n_prompt_over_window": 0, "n_window_mismatch": 0, "n_tokenize_resolved": 0, "n_floor_only_unresolvable": 0},
        "evidence": {"steering_bites": {"applicable": True, "verified": True, "n_differing": 5, "n_probes": 8}}}))
    rows = [{"decode": (timing_decode or DECODE), "max_tokens": 4096, "completion_tokens": 10 + i, "finish_reason": "stop"} for i in range(3)]
    (sd / "target_timing.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _instrument(G):
    return _with_funcs(dict(G), "instrument", "timing_rows_decode")["instrument"]


def test_a_uniform_single_root_store_yields_one_identity_tuple_carrying_its_real_window(G, tmp_path):
    for i in range(2):
        _shard(tmp_path, f"shard0{i}", 49152)
    inst = _instrument(G)(tmp_path, G["NAMESPACE_SERVING"][NATIVE_NS])
    assert inst["serving_window_tokens"] == 49152, "the identity tuple must carry the real window, not a merged-view placeholder"
    assert inst["n_shards"] == 2 and inst["n_window_clamped"] == 0
    assert "serving_windows_by_shard" not in inst and "serving_window_note" not in inst, "merged-store fields assert a merge that did not happen"
    assert "n_shards_decode_from_timing_rows" not in inst, "every sidecar was finalised: the timing_rows branch cross-checked, it never fell back"
    assert inst["serving_engine"] == "lora_native" and inst["lens_reduction_mode"] is None and inst["decode_mode"] == "shipped_sampling"


def test_a_window_split_REFUSES_under_the_declaration_and_is_silently_admitted_by_the_relu_clone(G, tmp_path):
    """THE KNOWN-BAD behind merged_store_windows False. Two shards, two windows, zero clamps: on a single-root store that is two
    instruments. The declaration refuses; NS_RELU's `True` (right for its two-root merged stores) would file it as one reading."""
    _shard(tmp_path, "shard00", 49152)
    _shard(tmp_path, "shard01", 16384)
    serving = G["NAMESPACE_SERVING"][NATIVE_NS]
    with pytest.raises(SystemExit) as e:
        _instrument(G)(tmp_path, serving)
    assert "not uniform across shards" in str(e.value)
    clone = {**serving, "merged_store_windows": True}
    inst = _instrument(G)(tmp_path, clone)
    assert inst["serving_windows_by_shard"] == {"16384": 1, "49152": 1} and inst["serving_window_tokens"] == 49152, \
        "the clone admitted a window split as one reading -- exactly what the False forbids"


@pytest.mark.parametrize("bad, needle", [
    (dict(lens_mode="stacked"), "lens_reduction_mode"),                                   # a lens-era shard in a native store
    (dict(engine="lens"), "serving_engine"),                                              # a shard drawn on another engine
    (dict(timing_decode={**DECODE, "temperature": 0.7}), "disagrees"),                    # the cross-check timing_rows runs on a finalised sidecar
])
def test_instrument_under_the_native_namespace_refuses_a_shard_that_breaks_the_declared_identity(G, tmp_path, bad, needle):
    _shard(tmp_path, "shard00", 49152)
    _shard(tmp_path, "shard01", 49152, **bad)
    with pytest.raises(SystemExit) as e:
        _instrument(G)(tmp_path, G["NAMESPACE_SERVING"][NATIVE_NS])
    assert needle in str(e.value)


def test_an_unfinalised_sidecar_is_the_only_way_timing_rows_falls_back_and_it_is_counted(G, tmp_path):
    """relu_native's case, pinned so the two reasons for `timing_rows` stay distinguishable: a null last_request_decode takes the
    rows' decode and stamps n_shards_decode_from_timing_rows; a finalised sidecar never does (previous tests)."""
    _shard(tmp_path, "shard00", 49152)
    _shard(tmp_path, "shard01", 49152, last_request_decode=None)
    inst = _instrument(G)(tmp_path, G["NAMESPACE_SERVING"][NATIVE_NS])
    assert inst["n_shards_decode_from_timing_rows"] == 1 and inst["decode"]["temperature"] == 1.0
