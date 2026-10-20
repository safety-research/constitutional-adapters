"""The passthrough-anchor pairing allowance (``sisteer.eval.anchor_pairing``) over SYNTHETIC stamps and records only
(invented ids, placeholder shas; nothing here is recorded data).

* ``anchor_pairing_verdict``: a lens-era passthrough ``none`` vs a natively served steered row is ADMITTED and stamped with the
  allowance; a steered LENS row vs a native row is REFUSED; a native ``none`` vs a native steered row pairs with
  ``anchor_engine == lora_native`` and no allowance; a passthrough anchor on another chat template is REFUSED (the template stays
  a pairing axis); a prompted anchor, an anchor carrying an object sha, a torn stamp — refused by name;
* ``engine_only_differences`` splits the vLLM launch / recorded keys that ARE the engine from every other difference;
* the adapters project each producer's record shape (served_target / bon_served_run / capeval manifest / petri cells.json /
  the sweep's engine stamp / a GSM8K leg) onto one ``PairingStamp``; the GSM8K leg reads the launch-resolved engine first.

This is the self-contained subset of the lane's ``tests/test_anchor_pairing.py`` (relu-steering-native-evals @ 823a0400): the
tests that import only ``anchor_pairing`` and modules on main. The sweep opt-in, the ``paired_vs_anchor.py`` CLI legs and the
GSM8K paired-store tests stay on the lane with the scripts and fixtures they exercise.
"""
from __future__ import annotations

import pytest

from sisteer.capeval.tinycap import contracts as C
from sisteer.eval import anchor_pairing as ap
from sisteer.eval import sweep

LENS_PT = "proxy_vllm+passthrough+interleaved"
LENS_ST = "proxy_vllm+lens+interleaved"
NATIVE_ST = "proxy_vllm+lora+interleaved"
NATIVE = C.SERVING_ENGINE_LORA_NATIVE
LENS = C.SERVING_ENGINE_LENS
SHA = "ab" * 32
TMPL = "1111111122222222"
PSR_LIKE = "constv2ctl2-full__lora_r4_ce"      # a fitted main cell stands in for the PSR cell (same route class)


def _st(cell, ident, engine=None, sha=None, tmpl=TMPL, view=None, model=None):
    return ap.PairingStamp(cell=cell, serving_identity=ident, serving_engine=engine, object_sha=sha, chat_template=tmpl, view=view, model=model)


# =========================================================================== #
# the rule                                                                      #
# =========================================================================== #


def test_constants_are_by_reference_to_the_stored_record_contract():
    assert ap.ENGINE_NATIVE == C.SERVING_ENGINE_LORA_NATIVE == "lora_native" and ap.ENGINE_LENS == C.SERVING_ENGINE_LENS
    assert ap.PASSTHROUGH_ANCHOR_ALLOWANCE == "passthrough-anchor-engine-agnostic-2026-09-15"
    assert C.REDUCTION_MODE_FIELD in ap.ENGINE_RECORDED_KEYS
    # every engine launch key is a wrapper-owned option of the vLLM launch composer (the two engines' argv differ on these
    # only because the composer sets them per engine) — pinned against the composer's own table
    vs = pytest.importorskip("sisteer.capeval.serving.vllm_server")
    owned = {o.lstrip("-").replace("-", "_") for o in vs._WRAPPER_OWNED_OPTIONS}
    owned |= {o[len("no_"):] for o in owned if o.startswith("no_")}
    assert set(ap.ENGINE_LAUNCH_KEYS) <= owned, set(ap.ENGINE_LAUNCH_KEYS) - owned


def test_lens_passthrough_none_vs_native_steered_is_admitted_and_stamped():
    v = ap.anchor_pairing_verdict(_st("psr", NATIVE_ST, NATIVE, SHA), _st("none", LENS_PT, None, None, view="it"))
    assert v.admitted and v.allowance == ap.PASSTHROUGH_ANCHOR_ALLOWANCE
    assert (v.anchor_engine, v.steered_engine, v.anchor_view) == (LENS, NATIVE, "it")
    assert "BY DECISION" in v.reason
    # the anchor stamped `lens` explicitly reads the same
    assert ap.anchor_pairing_verdict(_st("psr", NATIVE_ST, NATIVE, SHA), _st("none", LENS_PT, LENS)).allowance == ap.PASSTHROUGH_ANCHOR_ALLOWANCE
    # a native passthrough none vs a LENS-era steered row is NOT admitted either way round: the steered row must be native
    v2 = ap.anchor_pairing_verdict(_st("lora", LENS_ST, None, SHA), _st("none", LENS_PT, NATIVE))
    assert not v2.admitted and "not a 'lora' route" in v2.reason


def test_lens_steered_vs_native_steered_is_refused():
    v = ap.anchor_pairing_verdict(_st("psr", NATIVE_ST, NATIVE, SHA), _st("lora", LENS_ST, None, SHA))
    assert not v.admitted and v.allowance is None
    assert "anchor cell is 'lora'" in v.reason and "serving engines differ" in v.reason


def test_native_none_vs_native_steered_pairs_with_no_allowance():
    v = ap.anchor_pairing_verdict(_st("psr", NATIVE_ST, NATIVE, SHA), _st("none", LENS_PT, NATIVE, None, view="main"))
    assert v.admitted and v.allowance is None and v.anchor_engine == NATIVE and v.steered_engine == NATIVE and v.anchor_view == "main"
    # the lens world, unchanged: a lens steered row vs the lens passthrough none
    v = ap.anchor_pairing_verdict(_st("lora", LENS_ST, None, SHA), _st("none", LENS_PT))
    assert v.admitted and v.allowance is None and v.anchor_engine == LENS


def test_template_stays_a_pairing_axis():
    v = ap.anchor_pairing_verdict(_st("psr", NATIVE_ST, NATIVE, SHA, tmpl="aaaa"), _st("none", LENS_PT, tmpl="bbbb"))
    assert not v.admitted and "chat_template differs" in v.reason
    # one side recorded, one absent: not equality
    v = ap.anchor_pairing_verdict(_st("psr", NATIVE_ST, NATIVE, SHA, tmpl="aaaa"), _st("none", LENS_PT, tmpl=None))
    assert not v.admitted and "one side only" in v.reason
    # neither recorded: not compared here, named in the reason
    v = ap.anchor_pairing_verdict(_st("psr", NATIVE_ST, NATIVE, SHA, tmpl=None), _st("none", LENS_PT, tmpl=None))
    assert v.admitted and "not recorded on either side" in v.reason
    # the served model is an axis when both record it
    v = ap.anchor_pairing_verdict(_st("psr", NATIVE_ST, NATIVE, SHA, model="org/a"), _st("none", LENS_PT, model="org/b"))
    assert not v.admitted and "served model differs" in v.reason


def test_prompted_object_sha_and_torn_anchors_refuse_by_name():
    steered = _st("psr", NATIVE_ST, NATIVE, SHA)
    v = ap.anchor_pairing_verdict(steered, _st("prompted__safety", LENS_PT))
    assert not v.admitted and "anchor cell is 'prompted__safety'" in v.reason
    v = ap.anchor_pairing_verdict(steered, _st("none", LENS_PT, None, SHA))
    assert not v.admitted and "carries an object sha" in v.reason
    # a torn stamp (native engine beside a lens identity) is a refusal, never an exception
    v = ap.anchor_pairing_verdict(steered, _st("none", LENS_ST, NATIVE))
    assert not v.admitted and "torn" in v.reason
    v = ap.anchor_pairing_verdict(_st("psr", NATIVE_ST, LENS, SHA), _st("none", LENS_PT))
    assert not v.admitted and "torn" in v.reason
    # no identity and no engine: no engine can be read
    v = ap.anchor_pairing_verdict(steered, _st("none", None))
    assert not v.admitted and "neither a serving_identity nor a serving_engine" in v.reason
    with pytest.raises(ap.PairingError):
        ap.resolve_engine(_st("none", "proxy_vllm+bogus+interleaved"))


def test_engine_only_differences_splits_the_launch_and_recorded_keys():
    eng, other = ap.engine_only_differences(["launch.enforce_eager", "launch.lora_modules", "launch.dtype", C.REDUCTION_MODE_FIELD, "engine"])
    assert eng == ["launch.enforce_eager", "launch.lora_modules", "reduction_mode"] and other == ["engine", "launch.dtype"]
    assert ap.engine_only_differences({}) == ([], [])


def test_adapters_project_each_producer_record_shape():
    st = ap.stamp_from_served_target({"tt_cell_id": "none", "harness_cell_id": "none", "serving_identity": LENS_PT, "base_model": "org/m",
                                      "proxy_identity": {"object_artifact_sha256_by_cell": {}}}, cell="none", view="it")
    assert (st.cell, st.serving_identity, st.serving_engine, st.object_sha, st.view, st.model) == ("none", LENS_PT, None, None, "it", "org/m")
    st = ap.stamp_from_served_target({"tt_cell_id": "psr", "harness_cell_id": "si_psr", "serving_identity": NATIVE_ST, C.SERVED_ENGINE_FIELD: NATIVE,
                                      "object_artifact_sha256_by_cell": {"si_psr": SHA}}, cell="psr")
    assert (st.serving_engine, st.object_sha) == (NATIVE, SHA) and ap.resolve_engine(st) == NATIVE
    st = ap.stamp_from_bon_run({"cell_id": "none", "harness_cell_id": "none", "serving_identity": LENS_PT,
                                "proxy_identity": {"base_model": "org/m", "object_artifact_sha256_by_cell": {}}, "lens_reduction_mode": "stacked"})
    assert ap.resolve_engine(st) == LENS and st.object_sha is None
    man = {"cell_id": "none", "engine": "vllm", "serving_identity": LENS_PT, "serving_path": "passthrough", "model_id": "org/m",
           C.MANIFEST_KEY_ROUTE: {"serving_identity": LENS_PT, C.REDUCTION_MODE_FIELD: "stacked"},
           "fit_provenance": {"object_sha": None, "served_template_hash": TMPL}}
    st = ap.stamp_from_capeval_manifest(man, view="main")
    assert (st.serving_identity, st.serving_engine, st.chat_template, st.object_sha) == (LENS_PT, None, TMPL, None) and ap.resolve_engine(st) == LENS
    nat = {"cell_id": "psr", "engine": "vllm", "serving_identity": NATIVE_ST, "serving_path": "lora", C.SERVED_ENGINE_FIELD: NATIVE,
           C.MANIFEST_KEY_ROUTE: {"serving_identity": NATIVE_ST, C.SERVED_ENGINE_FIELD: NATIVE, C.REDUCTION_MODE_FIELD: C.REDUCTION_MODE_NOT_APPLICABLE},
           "fit_provenance": {"object_sha": SHA, "served_template_hash": TMPL}}
    st = ap.stamp_from_capeval_manifest(nat)
    assert st.serving_engine == NATIVE and st.object_sha == SHA
    with pytest.raises(ap.PairingError):        # a relabelled manifest (lens stamps beside a lora identity) refuses through the one reader
        ap.stamp_from_capeval_manifest({**nat, C.SERVED_ENGINE_FIELD: LENS, C.MANIFEST_KEY_ROUTE: {**nat[C.MANIFEST_KEY_ROUTE], C.SERVED_ENGINE_FIELD: LENS}})
    st = ap.stamp_from_petri_cell({"cell_id": "none", "serving_identity": LENS_PT, "object_path": None}, view="main")
    assert ap.resolve_engine(st) == LENS
    st = ap.stamp_from_sweep_stamp(sweep.engine_stamp(target_engine="vllm", serving_path="lora", served_template_hash=TMPL, object_sha=SHA), cell="psr")
    assert st.serving_identity == NATIVE_ST and ap.resolve_engine(st) == NATIVE and st.chat_template == TMPL
    st = ap.stamp_from_sweep_stamp(sweep.engine_stamp(target_engine="hf", serving_path="hf", served_template_hash=TMPL), cell="none")
    assert st.serving_identity is None
    leg = {"leg": "llama8b/main/psr@8192", "cell": "psr", "view": "main", "identity": {"serving_identity": NATIVE_ST, "object_artifact_sha256": SHA,
           "chat_template_hash": TMPL, "model_id": "org/m"}, "arm_contract": {"steered_path": "lora"}}
    st = ap.stamp_from_gsm8k_leg(leg)
    assert (st.serving_engine, st.object_sha, st.chat_template, st.model) == (NATIVE, SHA, TMPL, "org/m")



# =========================================================================== #
# the GSM8K leg adapter                                                         #
# =========================================================================== #


def test_gsm8k_leg_stamp_reads_the_launch_resolved_engine_before_the_steered_path_rule():
    """`stamp_from_gsm8k_leg`: the engine is the leg's `launch.engine` (the manifest's `route.serving_engine` as the reader resolves
    it — present iff native), so a native passthrough `none` reads NATIVE; without a filed launch engine the steered-path rule
    stands (lora -> native, lens -> lens, None -> lens by era).

    Carved from the lane's copy: the launch `status` values are the literal strings `tinycap_math500_thinking.LAUNCH_ATTESTED` /
    `LAUNCH_ABSENT` spell ("attested" / "absent") rather than the module attributes, so this file imports no script; the stamp reader
    never reads `status`, only `launch.engine`."""
    base = {"cell": "none", "leg": "llama8b/main/none@8192", "view": "main", "identity": {"serving_identity": LENS_PT, "model_id": "m", "chat_template_hash": TMPL},
            "arm_contract": {"steered_path": None}}
    native_none = ap.stamp_from_gsm8k_leg({**base, "launch": {"engine": NATIVE, "status": "attested"}})
    assert native_none.serving_engine == NATIVE and ap.resolve_engine(native_none) == NATIVE and ap.is_unsteered_passthrough(native_none) == (True, "")
    absent_native = ap.stamp_from_gsm8k_leg({**base, "launch": {"engine": NATIVE, "status": "absent"}})    # the engine is filed whether or not the record was found
    assert ap.resolve_engine(absent_native) == NATIVE
    assert ap.resolve_engine(ap.stamp_from_gsm8k_leg({**base, "launch": {"engine": LENS}})) == LENS
    assert ap.resolve_engine(ap.stamp_from_gsm8k_leg(base)) == LENS                                                 # no launch engine: by identity path, lens by era
    assert ap.resolve_engine(ap.stamp_from_gsm8k_leg({**base, "launch": {"engine": None}})) == LENS                 # a torn manifest's None falls to the path rule
    steered = {**base, "cell": PSR_LIKE, "identity": {**base["identity"], "serving_identity": NATIVE_ST}, "arm_contract": {"steered_path": "lora"}}
    assert ap.resolve_engine(ap.stamp_from_gsm8k_leg(steered)) == NATIVE
    # a same-engine native pair: admitted with NO allowance; the lens none beside the native cell: the allowance's shape
    assert ap.anchor_pairing_verdict(ap.stamp_from_gsm8k_leg(steered), native_none).allowance is None
    assert ap.anchor_pairing_verdict(ap.stamp_from_gsm8k_leg(steered), ap.stamp_from_gsm8k_leg({**base, "launch": {"engine": LENS}})).allowance == ap.PASSTHROUGH_ANCHOR_ALLOWANCE


# =========================================================================== #
# stage-(e) review fixes (2026-09-16): F2 clustered p, F3 copy-forward + cap, F4 --anchor-cell, F5 BoN cap / sampler basis
# =========================================================================== #
