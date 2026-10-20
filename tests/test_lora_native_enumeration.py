"""The ENUMERATION side of the plugin-free vLLM multi-LoRA serving mode.

What is pinned here and why each pin exists:

- ``defense_cells.NOOP_LORA_CELL_ID`` is THE constant for the ``none__lora0``
  leg (it was a bare literal at the derivation site) and equals
  ``lora_export.ZERO_ADAPTER_NAME``, the adapter name the zero export writes
  and the frontend serves it under. ``defense_cells`` is torch-free and
  ``lora_export`` imports torch, so the two modules cannot share the constant
  by import — this test is the pin (``test_native_lora_constants_agree``
  pins it again beside the others).
- ``NATIVE_LORA_SERVABLE_KINDS`` covers every kind the exporter can export.
- ``native_lora_unservable_reason`` — the ONE predicate — admits the kinds the
  three certification cells are (``lora_all``, ``lora_subtraction``;
  ``circuit_breaker`` at unit scaling) and refuses psr / loreft / a masked
  fit / a declared serve variant / a folded circuit_breaker / the stacked
  wrapper, each with a quotable reason.
- ``TrainTestCellSource(serving_path_policy=...)`` over the REAL explicit
  registry on a synthetic bank: ``lens`` is byte-identical to the pre-policy
  adapter; ``lora_native`` marks servable objects ``lora`` and refuses a psr
  cell BY NAME; an unknown policy refuses at construction; the YAML
  constructor threads the policy.
- ``enumerate_cells``: the default (lens) output equals a GOLDEN captured
  from the pre-change tree at 9486b825 — every leg, every hash, the legacy
  lens+lora hybrid table included — so the lens path is byte-identical;
  ``lens_controls=False`` derives ``none__lora0`` and never ``none__lens0``
  and refuses a canonical lens route; ``none__lora0`` derives iff a
  lora route is live; on a lora-native table a lora route whose object is
  not natively servable refuses, while on a lens table the same
  route is the legacy hybrid and enumerates; a stacked cell marked lora
  refuses on every table.

CPU-only; synthetic artifacts throughout; no vLLM, no model loads."""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path

import pytest
import torch
import yaml

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.petri import registry_adapter as ra
from sisteer.capeval.serving import lora_export as lx
from sisteer.capeval.serving import serve_declaration as sd

REV = "0123456789abcdef0123456789abcdef01234567"
IMPL = "fedcba9876" * 4
SAFETY_PREFIX = b"synthetic safety instruction: be safe (golden bytes)"
PROMPT_FILES = {"auditor": b"# golden auditor prompt bytes\n",
                "judge": b"# golden judge prompt bytes\n"}
BLOOM = cm.BloomOptions(max_turns=15, enable_prefill=False, enable_rollback=False,
                        compaction=False, realism_filter=False, epochs=1, seed=0)


def _bf(spec):
    return {k: f"# behavior {k}\ngolden eval bytes\n".encode()
            for k in cm.petri_expected_behavior_keys(spec)}


# ---------------------------------------------------------------------------
# 1. constants and vocabularies
# ---------------------------------------------------------------------------


def test_noop_lora_cell_id_is_the_one_constant_and_equals_the_exporters():
    assert dc.NOOP_LORA_CELL_ID == "none__lora0" == lx.ZERO_ADAPTER_NAME
    assert dc.NOOP_LENS_CELL_ID == "none__lens0"
    assert dc.NOOP_LORA_CELL_ID.rpartition("__")[2] in dc._CONTROL_SUFFIXES
    # a derived control id is never a manifest input
    with pytest.raises(dc.DefenseCellError):
        dc.parse_cell_id(dc.NOOP_LORA_CELL_ID)
    # the derivation site spells the leg through the constant — no bare
    # literal is left in enumerate_cells' body
    body = inspect.getsource(dc.enumerate_cells)
    assert '"none__lora0"' not in body and "NOOP_LORA_CELL_ID" in body


def test_native_servable_kinds_cover_every_exportable_kind():
    # the exporter's dispatch table can never name a kind enumeration
    # refuses by kind (G pins strict equality once A adds lora_subtraction)
    assert set(dc.NATIVE_LORA_SERVABLE_KINDS) >= set(lx.GENUINE_EXPORT_KINDS)
    # native-psr-serving: the five residual-slot kinds joined the set
    assert set(dc.NATIVE_LORA_SERVABLE_KINDS) == {
        "circuit_breaker", "lora", "lora_all", "lora_subtraction",
        "psr", "vector", "vector_all", "diffmean", "static_r",
        "psr_paired_write"}   # the paired write (relu-differenced-native-qwen)
    # the three certification kinds are in it
    assert {"lora_all", "lora_subtraction"} <= set(dc.NATIVE_LORA_SERVABLE_KINDS)


def test_the_per_arm_export_set_is_gone_and_the_interfaces_are_the_plans():
    assert not hasattr(ra, "LORA_EXPORTED_ARMS")
    assert ra.SERVING_PATH_POLICIES == ("lens", "lora_native")
    sig = inspect.signature(dc.enumerate_cells)
    lc = sig.parameters["lens_controls"]
    assert lc.default is True and lc.kind is inspect.Parameter.KEYWORD_ONLY
    # the additive zero-adapter-only kwarg,
    # default False so every existing enumeration is byte-identical
    lo = sig.parameters["lora_controls"]
    assert lo.default is False and lo.kind is inspect.Parameter.KEYWORD_ONLY
    sig = inspect.signature(ra.traintest_cell_source)
    sp = sig.parameters["serving_path_policy"]
    assert sp.default == "lens" and sp.kind is inspect.Parameter.KEYWORD_ONLY
    field = {f.name: f for f in dataclasses.fields(ra.TrainTestCellSource)}
    assert field["serving_path_policy"].default == "lens"


# ---------------------------------------------------------------------------
# 2. the predicate
# ---------------------------------------------------------------------------

_LORA_ALL = {"kind": "lora_all", "layers": [0, 2], "rank": 4,
             "positions": "all", "response_header_len": None}
_LORA_SUB = {"kind": "lora_subtraction", "layers": [0, 2], "rank": 4,
             "positions": "all", "response_header_len": None,
             "provenance": {"kind": "lora_subtraction"}}
_LORA_R1 = {"kind": "lora", "layer": 15, "positions": "all"}
_CB_UNIT = {"kind": "circuit_breaker", "rank": 4, "lora_alpha": 4.0,
            "transform_layers": [0, 1], "target_layers": [2]}


@pytest.mark.parametrize("kind,spec", [
    ("lora_all", _LORA_ALL),
    ("lora_subtraction", _LORA_SUB),
    ("lora", _LORA_R1),
    ("circuit_breaker", _CB_UNIT),
    ("circuit_breaker", {**_CB_UNIT, "rank": 8, "lora_alpha": 8}),  # int alpha
    # native-psr-serving: an all-positions residual-slot record serves ...
    ("psr", {"kind": "psr", "layers": [0, 2], "positions": "all"}),
    ("vector_all", {"kind": "vector_all"}),
    # ... and since stage 2 (NATIVE_STAGE2_ARMED, step 7) so does every mode the lens
    # names: the response-fit psr with its recorded header width, and static_r's two
    # prefill shapes. The per-token position class serves them; the `position_mode_stage2`
    # skip is retired (the stage-1 world is pinned in test_native_position_mode.py).
    ("psr", {"kind": "psr", "layers": [0, 2], "positions": "response", "response_header_len": 3}),
    ("static_r", {"kind": "static_r", "base_scale": 0.05, "steer_positions": "prefill_all",
                  "site": "pre", "layers": [19]}),
])
def test_predicate_admits_the_natively_servable_shapes(kind, spec):
    assert dc.native_lora_unservable_reason(kind, spec) is None


@pytest.mark.parametrize("kind,spec,needles", [
    # a residual record the mode predicate cannot judge — `last` is not a native mode, so
    # it refuses by name rather than being served at every position (the response-fit psr
    # SERVES since stage 2; see the admitted shapes above)
    ("psr", {"kind": "psr", "layers": [0, 2], "positions": "last"},
     ("residual-slot record cannot be judged", "not a native residual-slot mode")),
    ("loreft", {"kind": "loreft", "positions": "all"},
     ("kind 'loreft'", "not natively exportable")),
    ("alphasteer", {"kind": "alphasteer"}, ("not natively exportable",)),
    (cm.PETRI_STACKED_KIND, {"kind": cm.PETRI_STACKED_KIND},
     ("stacked cells serve on the lens only", "D6")),
    ("lora_all", {**_LORA_ALL, "positions": "response", "response_header_len": 3},
     ("positions='response'", "position-masked")),
    ("lora_all", {**_LORA_ALL, "positions": "prompt"},
     ("positions='prompt'", "position-masked")),
    ("lora_subtraction", {**_LORA_SUB, "positions": "response"},
     ("positions='response'", "position-masked")),
    ("lora", {"kind": "lora", "layer": 15}, ("no 'positions'", "fit leg")),
    ("lora_all", {k: v for k, v in _LORA_ALL.items() if k != "positions"},
     ("no 'positions'",)),
    ("circuit_breaker", {**_CB_UNIT, "lora_alpha": 16.0},
     ("non-unit LoRA scaling alpha=16.0 r=4", "folded into lora_b",
      "attestation is undefined")),
    ("circuit_breaker", {**_CB_UNIT, "lora_alpha": 2},
     ("alpha=2.0 r=4", "folded")),
    ("circuit_breaker", {"kind": "circuit_breaker", "rank": 4},
     ("no 'rank'/'lora_alpha'",)),
])
def test_predicate_refuses_with_a_quotable_reason(kind, spec, needles):
    reason = dc.native_lora_unservable_reason(kind, spec)
    assert reason is not None
    for needle in needles:
        assert needle in reason, (needle, reason)


def test_predicate_without_a_resolution_judges_the_kind_only():
    """``resolved_spec=None`` is the adapter's 'artifact not on disk' — the
    cell reports ABSENT downstream, nothing is served, so only the KIND is
    judged: a psr cell is still refused (the refusal holds for an unstaged
    view), a servable kind passes."""
    # native-psr-serving: a psr cell is a servable kind — absent, it serves (ABSENT downstream)
    assert dc.native_lora_unservable_reason("psr", None) is None
    assert "not natively exportable" in dc.native_lora_unservable_reason("loreft", None)
    assert "D6" in dc.native_lora_unservable_reason(cm.PETRI_STACKED_KIND, None)
    assert dc.native_lora_unservable_reason("lora_all", None) is None
    assert dc.native_lora_unservable_reason("circuit_breaker", None) is None


def test_predicate_refuses_a_declared_variant_before_reading_positions():
    """The position variants REPLACE ``positions`` in the resolved spec, so
    a response-fit object served ``@all_positions`` reads positions='all'
    — the exporter refuses it from the BYTES; the predicate must agree
    from the SPEC, which it can only do by refusing the declaration."""
    response_fit = {**_LORA_ALL, "positions": "response", "response_header_len": 3}
    overlaid = sd.apply_variant(response_fit, sd.VARIANT_ALL_POSITIONS)
    assert overlaid["positions"] == "all"  # the hazard, reproduced
    reason = dc.native_lora_unservable_reason("lora_all", overlaid)
    assert reason is not None and "serve declaration" in reason and "lens-only" in reason
    # a render-side declaration (thinking) is lens-only too: the proxy's
    # thinking-ON gate is a lens-route gate
    thinking = sd.apply_variant(dict(_LORA_ALL), sd.VARIANT_THINKING)
    assert "serve declaration" in dc.native_lora_unservable_reason("lora_all", thinking)
    # a malformed declaration is a reason, never a crash
    bad = {**_LORA_ALL, sd.SERVE_OUT_OF_SUPPORT_KEY: "not-a-list"}
    assert "malformed serve declaration" in dc.native_lora_unservable_reason("lora_all", bad)


# ---------------------------------------------------------------------------
# 3. the REAL source over the explicit registry on a synthetic bank
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
GEMMA_IT = "google/gemma-4-E4B-it"
GEMMA_IT_REV = "ee0ef6023621cff504d758262d4e04895a5af4a2"
SI = "constv2-mix-full"            # fit-lane spelling of the corpus SI
HSI = "constv2_mix_full"           # harness spelling
TT_LORA, TT_PSR, TT_SUB = (f"{SI}__lora_r4_ce", f"{SI}__psr_ce_all",
                           f"{SI}__lora_r4_ce_minus_ctl")
H_LORA, H_PSR, H_SUB = (f"si_{HSI}__lora_r4_ce", f"si_{HSI}__psr_ce_all",
                        f"si_{HSI}__lora_r4_ce_minus_ctl")
D_IN, D_OUT, RANK = 10, 6, 3


def _lora_all_blob(positions="all"):
    g = torch.Generator().manual_seed(11)
    layers = [0, 2]
    return {"kind": "lora_all", "layers": layers, "rank": RANK,
            "A": {str(l): torch.randn(D_IN, RANK, generator=g) for l in layers},
            "B": {str(l): torch.randn(D_OUT, RANK, generator=g) for l in layers},
            "positions": positions,
            "response_header_len": 3 if positions == "response" else None}


def _psr_blob(positions="response"):
    """The synthetic psr cell. Its fit leg defaults to `response` — a mode the native
    engine SERVES since stage 2 (step 7), and the shape the scale-12 objects have;
    `last` gives a record the mode predicate cannot judge at all (the refusal case)."""
    g = torch.Generator().manual_seed(12)
    return {"kind": "psr", "layers": [0, 2], "train_direction": "pos",
            "positions": positions,
            "response_header_len": 3 if positions == "response" else None,
            "units": {str(l): torch.randn(D_IN, generator=g) for l in (0, 2)}}


def _sub_blob(positions="all"):
    g = torch.Generator().manual_seed(13)
    blob = {"kind": "lora_subtraction", "layers": [0, 2], "rank": RANK,
            "positions": positions,
            "response_header_len": 3 if positions == "response" else None}
    for side in ("pos", "neg"):
        blob[f"A_{side}"] = {str(l): torch.randn(D_IN, RANK, generator=g) for l in (0, 2)}
        blob[f"B_{side}"] = {str(l): torch.randn(D_OUT, RANK, generator=g) for l in (0, 2)}
    return blob


def _stage_budget_cell(bank: Path, tt_cell: str, blob: dict) -> Path:
    """A staged TRAINED cell in the bank's layout — object + the document-
    schema budget sidecar the explicit registry validates (the same shape
    tests/test_derived_arms_serving.py stages its parents in)."""
    from sisteer.eval import defense_cell as tt

    p = tt.object_path_for(bank, tt_cell)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, p)
    meta = {"cell_id": tt_cell, "model_id": "google/gemma-4-E4B",
            "model_revision": "4" * 40, "object_sha256": tt.sha256_file(p),
            "training_budget_kind": "document_tokens", "sequence_max_len": 256,
            "completion_tokens": None, "sample_manifest_sha": "ab" * 32,
            "positions": blob["positions"],
            "response_header_len": blob.get("response_header_len"),
            "target_layers": [0, 2], "template_hash": "e3b0c44298fc1c14",
            "fit_fingerprint": "d3" * 32}
    (p.parent / "train_budget_meta.json").write_text(json.dumps(meta))
    return p


def _stage_derived_cell(bank: Path, tt_cell: str, blob: dict) -> Path:
    """A staged DERIVED cell — object + derived_meta.json binding the bytes
    to two parent identities, the same provenance embedded in the bytes."""
    from sisteer.eval import defense_cell as tt

    p = tt.object_path_for(bank, tt_cell)
    p.parent.mkdir(parents=True, exist_ok=True)

    def parent(cell, sha, fp):
        return {"cell_id": cell, "model_key": "gemma4e4b_base",
                "object_path": f"/vol/out/gemma4e4b_base/cells_all/{cell}/object.pt",
                "object_sha256": sha, "budget_meta_sha256": "bb" * 32,
                "fit_fingerprint": fp, "model_id": "google/gemma-4-E4B",
                "model_revision": "4" * 40,
                "model_revision_source": "train_budget_meta.json",
                "positions": "all", "response_header_len": None,
                "target_layers": [0, 2], "template_hash": "e3b0c44298fc1c14",
                "training_budget_kind": "document_tokens"}
    prov = {"kind": "lora_subtraction", "derived_id": tt_cell,
            "parent_pos": parent(TT_LORA, "aa" * 32, "d3" * 32),
            "parent_neg": parent("constv2ctl2-full__lora_r4_ce", "cc" * 32, "39" * 32)}
    payload = dict(blob)
    payload["provenance"] = prov
    torch.save(payload, p)
    meta = {**prov, "model_key": "gemma4e4b_base", "model_id": "google/gemma-4-E4B",
            "model_revision": "4" * 40, "positions": payload["positions"],
            "response_header_len": payload.get("response_header_len"),
            "rank": payload["rank"], "layers": payload["layers"],
            "builder": "scripts/sdfbase/lora_subtract.py",
            "object_sha256": tt.sha256_file(p)}
    (p.parent / "derived_meta.json").write_text(json.dumps(meta, indent=1))
    return p


def _bank(tmp_path: Path, *, lora_positions="all", with_psr=True, psr_positions="response") -> Path:
    bank = tmp_path / "bank"
    _stage_budget_cell(bank, TT_LORA, _lora_all_blob(lora_positions))
    if with_psr:
        _stage_budget_cell(bank, TT_PSR, _psr_blob(psr_positions))
    _stage_derived_cell(bank, TT_SUB, _sub_blob())
    return bank


def _explicit_source(bank: Path, cells, **kw) -> ra.TrainTestCellSource:
    """The plan's 'TrainTestCellSource over a synthetic registry': the REAL
    explicit-cell registry built directly over the bank."""
    from sisteer.eval import defense_cell as tt

    registry = tt.registry_from_explicit_cells(
        {"explicit_cells": ["none", *cells], "defenses_dir": str(bank)})
    return ra.TrainTestCellSource(cells=tt.by_id(registry), defenses_dir=bank, **kw)


def _real_spec(cells, **over) -> cm.PetriSpec:
    kwargs = dict(
        audit="lora-native-enumeration-real-source", behaviors=list(cm.PETRI_BEHAVIORS),
        cells=list(cells), scenario_bundle_hash="ab" * 8,
        auditor_model="anthropic/claude-sonnet-5", judge_model="anthropic/claude-opus-5",
        scenarios_model="anthropic/claude-sonnet-5",
        prompt_provenance="package-pinned@petri-bloom==0.2.6",
        target_model=GEMMA_IT, target_revision=GEMMA_IT_REV,
        serving_backend="proxy_vllm", serving_mode="exact_replay",
        serving_packages={"vllm": "0.25.1+cu129", "vllm-lens": "1.2.1"},
        serving_launch_flags=["--enforce-eager"], chat_template_hash="12" * 8,
        tool_parser="llama3_json", serving_impl_commit=IMPL,
        # validate_petri_spec wants a non-empty map; no prompted cell is in
        # play here so nothing is verified against it
        prompted_prefix_hashes={"safety": "ac3a58f61d53cf24"}, bloom_options=BLOOM)
    kwargs.update(over)
    return cm.PetriSpec(**kwargs)


class TestRealSourcePolicy:
    def test_lens_policy_is_the_default_and_byte_identical(self, tmp_path):
        bank = _bank(tmp_path)
        cells = [TT_LORA, TT_PSR, TT_SUB]
        default = _explicit_source(bank, cells)
        explicit = _explicit_source(bank, cells, serving_path_policy="lens")
        assert default.serving_path_policy == "lens"
        for hid in (H_LORA, H_PSR, H_SUB):
            e = default.get(hid)
            assert e == explicit.get(hid)                     # same object, same policy
            assert e.serving_path == "lens"
        assert default.get(H_LORA).kind == "lora_all"
        assert default.get(H_PSR).kind == "psr"
        assert default.get(H_SUB).kind == "lora_subtraction"
        # a lens standup over this bank: every object lens, the lens twin only
        spec = _real_spec(["none", H_LORA, H_PSR, H_SUB])
        cs = dc.enumerate_cells(spec, default, behavior_files=_bf(spec),
                                prompt_files=PROMPT_FILES)
        assert [a.cell_id for a in cs.absent] == []
        assert {c.serving_identity for c in cs.canonical if c.cell_class != "base"} == {
            "proxy_vllm+lens+exact_replay"}
        assert [c.cell_id for c in cs.controls] == [dc.NOOP_LENS_CELL_ID]

    def test_lora_native_marks_the_certification_kinds_lora(self, tmp_path):
        bank = _bank(tmp_path, with_psr=False)
        lens = _explicit_source(bank, [TT_LORA, TT_SUB])
        native = _explicit_source(bank, [TT_LORA, TT_SUB],
                                  serving_path_policy="lora_native")
        for hid, kind in ((H_LORA, "lora_all"), (H_SUB, "lora_subtraction")):
            n, l = native.get(hid), lens.get(hid)
            assert n.serving_path == "lora" and n.kind == kind
            # the policy decides the PATH and nothing else about the entry
            assert dataclasses.replace(n, serving_path="lens") == l
        # base and prompted entries carry no path under either policy
        assert native.get("none") == lens.get("none")
        assert native.get("none").serving_path is None

    def test_lora_native_serves_a_response_fit_psr_cell_since_stage_2(self, tmp_path):
        """native-psr-serving step 7: the synthetic psr cell is fit at 'response', a mode the
        per-token position class now serves — so it is a lora route like any other, and the
        whole view enumerates. (The stage-1 refusal it used to get is pinned, with the flag
        monkeypatched, in tests/test_native_position_mode.py.)"""
        bank = _bank(tmp_path)
        native = _explicit_source(bank, [TT_LORA, TT_PSR, TT_SUB],
                                  serving_path_policy="lora_native")
        assert native.get(H_LORA).serving_path == "lora"
        assert native.get(H_PSR).serving_path == "lora" and native.get(H_PSR).kind == "psr"
        spec = _real_spec(["none", H_LORA, H_PSR])
        cs = dc.enumerate_cells(spec, native, behavior_files=_bf(spec),
                                prompt_files=PROMPT_FILES, lens_controls=False)
        assert H_PSR in {c.cell_id for c in cs.all_served()}

    def test_lora_native_refuses_a_residual_record_it_cannot_judge_by_name(self, tmp_path):
        """What a residual cell CAN still be refused for: a record whose position mode is
        outside the native vocabulary. Never served at every position instead — a refusal
        naming the cell, the kind and the escape hatch."""
        bank = _bank(tmp_path, psr_positions="last")
        native = _explicit_source(bank, [TT_LORA, TT_PSR, TT_SUB],
                                  serving_path_policy="lora_native")
        assert native.get(H_LORA).serving_path == "lora"   # the view is otherwise fine
        with pytest.raises(ra.RegistryAdapterError) as exc:
            native.get(H_PSR)
        msg = str(exc.value)
        assert H_PSR in msg and "kind 'psr'" in msg and "D2" in msg
        assert "not a native residual-slot mode" in msg and "lens standup" in msg
        # and the refusal is the source's, not a downstream 404: enumeration
        # over this source refuses too, naming the cell
        spec = _real_spec(["none", H_LORA, H_PSR])
        with pytest.raises(ra.RegistryAdapterError, match=H_PSR):
            dc.enumerate_cells(spec, native, behavior_files=_bf(spec),
                               prompt_files=PROMPT_FILES, lens_controls=False)

    def test_lora_native_refuses_a_masked_fit_by_name(self, tmp_path):
        bank = _bank(tmp_path, lora_positions="response", with_psr=False)
        lens = _explicit_source(bank, [TT_LORA])
        assert lens.get(H_LORA).serving_path == "lens"      # the lens serves the mask
        native = _explicit_source(bank, [TT_LORA], serving_path_policy="lora_native")
        with pytest.raises(ra.RegistryAdapterError) as exc:
            native.get(H_LORA)
        assert H_LORA in str(exc.value)
        assert "positions='response'" in str(exc.value)
        assert "position-masked" in str(exc.value)

    def test_lora_native_refuses_a_declared_variant_by_name(self, tmp_path):
        bank = _bank(tmp_path, with_psr=False)
        native = _explicit_source(bank, [TT_LORA], serving_path_policy="lora_native")
        assert native.get(H_LORA).serving_path == "lora"
        with pytest.raises(ra.RegistryAdapterError) as exc:
            native.get(f"{H_LORA}@thinking")
        assert f"{H_LORA}@thinking" in str(exc.value)
        assert "serve declaration" in str(exc.value)

    def test_an_unstaged_servable_cell_is_absent_not_refused(self, tmp_path):
        """A Cartesian registry over an EMPTY bank (nothing staged): under
        ``lora_native`` the servable kind is judged by kind only and reports
        ABSENT at enumeration as it always did; the psr cell still refuses by
        kind — the refusal holds for a view whose objects are not even
        staged yet."""
        from sisteer.eval import defense_cell as tt

        bank = tmp_path / "empty"
        bank.mkdir()
        registry = tt.build_registry(bank, si_ids=(SI,),
                                     class_a_arms=("lora_r4_ce", "psr_ce_all"),
                                     class_b_arms=())
        native = ra.TrainTestCellSource(cells=tt.by_id(registry), defenses_dir=bank,
                                        serving_path_policy="lora_native")
        entry = native.get(H_LORA)
        assert entry.serving_path == "lora" and entry.resolved_spec is None
        spec = _real_spec(["none", H_LORA])
        cs = dc.enumerate_cells(spec, native, behavior_files=_bf(spec),
                                prompt_files=PROMPT_FILES, lens_controls=False)
        assert [a.cell_id for a in cs.absent] == [H_LORA]
        assert str(tt.object_path_for(bank, TT_LORA)) in cs.absent[0].reason
        assert cs.controls == ()          # nothing lora is LIVE -> no none__lora0
        # native-psr-serving: an unstaged psr cell is a servable kind judged by kind alone —
        # a lora route with no resolution, ABSENT downstream exactly like the lora cell
        psr_entry = native.get(H_PSR)
        assert psr_entry.serving_path == "lora" and psr_entry.resolved_spec is None

    def test_an_unknown_policy_refuses_at_construction(self, tmp_path):
        bank = _bank(tmp_path, with_psr=False)
        with pytest.raises(ra.RegistryAdapterError, match="serving_path_policy"):
            _explicit_source(bank, [TT_LORA], serving_path_policy="lora")
        with pytest.raises(ra.RegistryAdapterError, match="serving_path_policy"):
            _explicit_source(bank, [TT_LORA], serving_path_policy=None)
        # the YAML constructor validates the policy FIRST — before the config
        # is even read (a missing config would otherwise be the refusal)
        with pytest.raises(ra.RegistryAdapterError, match="serving_path_policy"):
            ra.traintest_cell_source(_real_spec(["none", H_LORA]), repo_root=REPO,
                                     config_path=tmp_path / "does-not-exist.yaml",
                                     defenses_dir=bank, serving_path_policy="native")

    def test_the_yaml_constructor_threads_the_policy(self, tmp_path):
        bank = _bank(tmp_path, with_psr=False)
        cfg = tmp_path / "explicit.yaml"
        cfg.write_text(yaml.safe_dump({
            "explicit_cells": ["none", TT_LORA, TT_SUB], "defenses_dir": str(bank),
            "model": GEMMA_IT, "model_revision": GEMMA_IT_REV}))
        spec = _real_spec(["none", H_LORA, H_SUB])
        lens = ra.traintest_cell_source(spec, repo_root=REPO, config_path=cfg,
                                        defenses_dir=bank)
        native = ra.traintest_cell_source(spec, repo_root=REPO, config_path=cfg,
                                          defenses_dir=bank,
                                          serving_path_policy="lora_native")
        assert lens.serving_path_policy == "lens" and native.serving_path_policy == "lora_native"
        assert lens.get(H_LORA).serving_path == "lens"
        assert native.get(H_LORA).serving_path == "lora"
        assert native.get(H_SUB).serving_path == "lora"

    def test_lora_native_enumeration_derives_the_zero_adapter_leg_only(self, tmp_path):
        """End to end on the real source: the lora-native table carries
        the base, the two adapters and ``none__lora0`` — no ``none__lens0``,
        no lens identity anywhere — and the zero leg's intervention hash is
        ``none``'s (the same intervention, a different serving)."""
        bank = _bank(tmp_path, with_psr=False)
        native = _explicit_source(bank, [TT_LORA, TT_SUB],
                                  serving_path_policy="lora_native")
        spec = _real_spec(["none", H_LORA, H_SUB])
        cs = dc.enumerate_cells(spec, native, behavior_files=_bf(spec),
                                prompt_files=PROMPT_FILES, lens_controls=False)
        assert [a.cell_id for a in cs.absent] == []
        by_id = {c.cell_id: c for c in cs.all_served()}
        assert set(by_id) == {"none", H_LORA, H_SUB, dc.NOOP_LORA_CELL_ID}
        assert dc.NOOP_LENS_CELL_ID not in by_id
        assert not any("+lens+" in c.serving_identity for c in cs.all_served())
        for hid in (H_LORA, H_SUB):
            assert by_id[hid].serving_identity == "proxy_vllm+lora+exact_replay"
        lora0 = by_id[dc.NOOP_LORA_CELL_ID]
        assert lora0.cell_class == "control" and lora0.control_of == "none"
        assert lora0.kind == cm.NONE_KIND and lora0.object_path is None
        assert lora0.virtual_model_name == dc.VMODEL_PREFIX + dc.NOOP_LORA_CELL_ID
        assert lora0.serving_identity == "proxy_vllm+lora+exact_replay"
        assert lora0.intervention_hash == by_id["none"].intervention_hash
        assert cs.virtual_model_map()["petri__none__lora0"] is lora0
        # the SAME source under the default flag would derive the lens twin
        # too — pinning that lens_controls, not the policy, suppresses it
        cs_default = dc.enumerate_cells(spec, native, behavior_files=_bf(spec),
                                        prompt_files=PROMPT_FILES)
        assert [c.cell_id for c in cs_default.controls] == [
            dc.NOOP_LENS_CELL_ID, dc.NOOP_LORA_CELL_ID]


# ---------------------------------------------------------------------------
# 4. enumerate_cells over synthetic sources: the golden and the table rules
# ---------------------------------------------------------------------------

GOLDEN_CELLS = ["none", "prompted__safety", "si_safety__psr_kl_all",
                "si_safety__lora_r4", "alphasteer"]
GOLDEN_BYTES = {"si_safety__psr_kl_all": b"golden-psr-object-bytes",
                "si_safety__lora_r4": b"golden-lora-all-object-bytes",
                "alphasteer": b"golden-alphasteer-object-bytes"}
#: enumerate_cells' output for the fixture below, captured from the
#: PRE-CHANGE tree (9486b825, before lens_controls / the predicate existed):
#: (cell_id, cell_class, kind, si_id, vmodel, serving_identity,
#:  intervention_hash, harness_config_hash, petri_run_key, control_of,
#:  si_prefix_hash) for every served leg, canonical then controls. Every
#: hash is over fixed literal bytes and spec fields, so the capture is
#: deterministic; equality here IS "the lens path is byte-identical",
#: legacy lens+lora hybrid table (a lora route beside lens routes) included.
#: Re-pinned 2026-09-19: the harness_config_hash and petri_run_key columns moved because the
#: DEFAULT auditor chain lost its abliterated-Qwen third leg — a different chain is a
#: different instrument, so every run key moved; cell ids, kinds, identities, intervention hashes and si_prefix
#: hashes are unchanged, which is what this golden guards.
GOLDEN_ROWS = [
    ("none", "base", "none", None, "petri__base",
     "proxy_vllm+passthrough+exact_replay", "2ff9cdff091aef6f",
     "aec47f500db4a930", "689824645a6c5f5e", None, None),
    ("prompted__safety", "prompted", "prompted", "safety", "petri__prompted__safety",
     "proxy_vllm+passthrough+exact_replay", "e4b645fa424fb03c",
     "aec47f500db4a930", "a42cb11a5a3f9632", None, "13fa8bfda46cc4fb"),
    ("si_safety__psr_kl_all", "class_a", "psr", "safety", "petri__si_safety__psr_kl_all",
     "proxy_vllm+lens+exact_replay", "918f53c9493cbc81",
     "ac8537ac98f0e061", "aba17fff0e5bffdb", None, None),
    ("si_safety__lora_r4", "class_a", "lora_all", "safety", "petri__si_safety__lora_r4",
     "proxy_vllm+lora+exact_replay", "71d4664df2d683c5",
     "f899415221aa69d8", "efcacf9bcc5f8f7e", None, None),
    ("alphasteer", "class_b", "alphasteer", None, "petri__alphasteer",
     "proxy_vllm+lens+exact_replay", "4b3acf7345900825",
     "ac8537ac98f0e061", "23b4e680cc9ba702", None, None),
    ("none__lens0", "control", "none", None, "petri__none__lens0",
     "proxy_vllm+lens+exact_replay", "2ff9cdff091aef6f",
     "ac8537ac98f0e061", "139bddd485034753", "none", None),
    ("none__lora0", "control", "none", None, "petri__none__lora0",
     "proxy_vllm+lora+exact_replay", "2ff9cdff091aef6f",
     "f899415221aa69d8", "bb865117f5aab716", "none", None),
]


def _golden_spec(**over) -> cm.PetriSpec:
    kw = dict(
        audit="lora-native-enumeration-golden", behaviors=list(cm.PETRI_BEHAVIORS),
        cells=list(GOLDEN_CELLS), scenario_bundle_hash="ab" * 8,
        auditor_model="anthropic/claude-sonnet-5", judge_model="anthropic/claude-opus-5",
        scenarios_model="anthropic/claude-sonnet-5",
        prompt_provenance="package-pinned@petri-bloom==0.2.6",
        target_model="meta-llama/Llama-3.1-8B-Instruct", target_revision=REV,
        serving_backend="proxy_vllm", serving_mode="exact_replay",
        serving_packages={"vllm": "0.25.1+cu129", "vllm-lens": "1.2.1"},
        serving_launch_flags=["--enable-lora", "--enforce-eager"],
        chat_template_hash="12" * 8, tool_parser="llama3_json",
        serving_impl_commit=IMPL,
        prompted_prefix_hashes={"safety": cm.sha256_hex16(SAFETY_PREFIX)},
        bloom_options=BLOOM)
    kw.update(over)
    return cm.PetriSpec(**kw)


def _golden_entries(root: Path):
    def art(cid):
        p = root / f"{cid}.pt"
        p.write_bytes(GOLDEN_BYTES[cid])
        return str(p)
    return {
        "none": dc.RegistryEntry(cell_id="none", kind="none"),
        "prompted__safety": dc.RegistryEntry(
            cell_id="prompted__safety", kind="prompted", si_id="safety",
            si_prefix=SAFETY_PREFIX),
        "si_safety__psr_kl_all": dc.RegistryEntry(
            cell_id="si_safety__psr_kl_all", kind="psr", si_id="safety",
            object_path=art("si_safety__psr_kl_all"), serving_path="lens",
            resolved_spec={"kind": "psr", "si_id": "safety", "layers": [0, 2],
                           "positions": "response"}),
        "si_safety__lora_r4": dc.RegistryEntry(
            cell_id="si_safety__lora_r4", kind="lora_all", si_id="safety",
            object_path=art("si_safety__lora_r4"), serving_path="lora",
            resolved_spec={"kind": "lora_all", "si_id": "safety", "layers": [0, 2],
                           "rank": 4, "positions": "all", "response_header_len": None}),
        "alphasteer": dc.RegistryEntry(
            cell_id="alphasteer", kind="alphasteer", si_id=None,
            object_path=art("alphasteer"), serving_path="lens",
            resolved_spec={"kind": "alphasteer", "layer": 15}),
    }


def _rows(cs: dc.PetriCellSet):
    return [(c.cell_id, c.cell_class, c.kind, c.si_id, c.virtual_model_name,
             c.serving_identity, c.intervention_hash, c.harness_config_hash,
             c.petri_run_key, c.control_of, c.si_prefix_hash)
            for c in cs.all_served()]


def _enum(spec, entries, **kw):
    return dc.enumerate_cells(spec, dc.MappingCellSource(entries),
                              behavior_files=_bf(spec), prompt_files=PROMPT_FILES, **kw)


def test_the_lens_policy_enumeration_equals_the_pre_change_golden(tmp_path):
    spec = _golden_spec()
    cs = _enum(spec, _golden_entries(tmp_path))
    assert [a.cell_id for a in cs.absent] == []
    assert _rows(cs) == GOLDEN_ROWS
    # the explicit default spells the same table
    assert _rows(_enum(spec, _golden_entries(tmp_path), lens_controls=True)) == GOLDEN_ROWS
    # the control legs are spelled from the constants
    assert [c.cell_id for c in cs.controls] == [dc.NOOP_LENS_CELL_ID, dc.NOOP_LORA_CELL_ID]


def _lora_only_entries(root: Path, *, lora=True):
    """A lora-native table's shape: base, a prompted baseline, and (optionally)
    ONE natively servable object marked lora — no lens entry anywhere."""
    ent = {k: v for k, v in _golden_entries(root).items()
           if k in ("none", "prompted__safety", "si_safety__lora_r4")}
    if not lora:
        ent.pop("si_safety__lora_r4")
    return ent


@pytest.mark.parametrize("lens_controls", [True, False])
@pytest.mark.parametrize("lora_live", [True, False])
def test_none_lora0_derives_iff_a_lora_route_is_live(tmp_path, lens_controls, lora_live):
    cells = ["none", "prompted__safety"] + (["si_safety__lora_r4"] if lora_live else [])
    cs = _enum(_golden_spec(cells=cells), _lora_only_entries(tmp_path, lora=lora_live),
               lens_controls=lens_controls)
    ids = [c.cell_id for c in cs.controls]
    assert (dc.NOOP_LORA_CELL_ID in ids) is lora_live
    assert (dc.NOOP_LENS_CELL_ID in ids) is lens_controls
    expected = ([dc.NOOP_LENS_CELL_ID] if lens_controls else []) + (
        [dc.NOOP_LORA_CELL_ID] if lora_live else [])
    assert ids == expected
    if lora_live:
        by_id = {c.cell_id: c for c in cs.all_served()}
        assert by_id[dc.NOOP_LORA_CELL_ID].intervention_hash == by_id["none"].intervention_hash


def test_lens_controls_false_never_derives_or_admits_a_lens_leg(tmp_path):
    cs = _enum(_golden_spec(cells=["none", "prompted__safety", "si_safety__lora_r4"]),
               _lora_only_entries(tmp_path), lens_controls=False)
    assert [c.cell_id for c in cs.controls] == [dc.NOOP_LORA_CELL_ID]
    assert dc.NOOP_LENS_CELL_ID not in {c.cell_id for c in cs.all_served()}
    assert not any("+lens+" in c.serving_identity for c in cs.all_served())
    # a canonical LENS route on that table refuses by name (a lens
    # request there is base served under a lens identity)
    spec = _golden_spec()  # every golden cell, psr and alphasteer on the lens
    with pytest.raises(dc.DefenseCellError) as exc:
        _enum(spec, _golden_entries(tmp_path), lens_controls=False)
    msg = str(exc.value)
    assert "si_safety__psr_kl_all" in msg and "serving_path='lens'" in msg
    assert "no lens hook" in msg and "D15" in msg


def test_a_lora_native_table_refuses_an_unservable_lora_route(tmp_path):
    """'psr marked lora -> 404 later' becomes a loud enumeration
    refusal on a lora-native table. On a LENS table the same entry is the
    legacy lens+lora hybrid (the shape tests/test_petri_cells.py,
    test_petri_stats.py and test_petri_proxy.py build with synthetic kinds)
    and enumerates unchanged — pinned here so the gating is deliberate."""
    ent = _lora_only_entries(tmp_path)
    psr = _golden_entries(tmp_path)["si_safety__psr_kl_all"]
    ent["si_safety__psr_kl_all"] = dataclasses.replace(psr, serving_path="lora")
    cells = ["none", "prompted__safety", "si_safety__lora_r4", "si_safety__psr_kl_all"]
    with pytest.raises(dc.DefenseCellError) as exc:
        _enum(_golden_spec(cells=cells), ent, lens_controls=False)
    msg = str(exc.value)
    assert "si_safety__psr_kl_all" in msg and "kind 'psr'" in msg
    # native-psr-serving step 7: the golden psr entry is fit at 'response' — a mode the
    # position class SERVES — but its record carries no header width, and the served
    # region's boundary is the fit's own: an incomplete record refuses, never a guess
    assert "no response_header_len" in msg and "cannot be judged" in msg and "D2" in msg
    # the legacy hybrid: a lens table admits the same entry as a lora route
    cs = _enum(_golden_spec(cells=cells), ent)
    by_id = {c.cell_id: c for c in cs.all_served()}
    assert by_id["si_safety__psr_kl_all"].serving_identity == "proxy_vllm+lora+exact_replay"
    assert [c.cell_id for c in cs.controls] == [dc.NOOP_LENS_CELL_ID, dc.NOOP_LORA_CELL_ID]


def test_a_lora_native_table_refuses_a_masked_lora_route(tmp_path):
    ent = _lora_only_entries(tmp_path)
    masked = ent["si_safety__lora_r4"]
    ent["si_safety__lora_r4"] = dataclasses.replace(
        masked, resolved_spec={**masked.resolved_spec, "positions": "response",
                               "response_header_len": 3})
    cells = ["none", "prompted__safety", "si_safety__lora_r4"]
    with pytest.raises(dc.DefenseCellError) as exc:
        _enum(_golden_spec(cells=cells), ent, lens_controls=False)
    assert "si_safety__lora_r4" in str(exc.value)
    assert "positions='response'" in str(exc.value) and "position-masked" in str(exc.value)
    # the lens table admits the legacy shape
    cs = _enum(_golden_spec(cells=cells), ent)
    assert {c.cell_id for c in cs.controls} == {dc.NOOP_LENS_CELL_ID, dc.NOOP_LORA_CELL_ID}


def test_a_lora_native_table_refuses_a_declared_variant_lora_route(tmp_path):
    ent = _lora_only_entries(tmp_path)
    base = ent["si_safety__lora_r4"]
    variant = f"si_safety__lora_r4@{sd.VARIANT_ALL_POSITIONS}"
    ent[variant] = dataclasses.replace(
        base, cell_id=variant,
        resolved_spec=sd.apply_variant(dict(base.resolved_spec), sd.VARIANT_ALL_POSITIONS))
    cells = ["none", "prompted__safety", "si_safety__lora_r4", variant]
    with pytest.raises(dc.DefenseCellError) as exc:
        _enum(_golden_spec(cells=cells), ent, lens_controls=False)
    assert variant in str(exc.value) and "serve declaration" in str(exc.value)


@pytest.mark.parametrize("lens_controls", [True, False])
def test_a_stacked_cell_marked_lora_refuses_on_every_table(tmp_path, lens_controls):
    """No stacked SI-plus-LoRA cell is served natively — the refusal
    does not depend on which table the stack is on."""
    from sisteer.capeval.serving import lens_adapters as la

    stacked = "stacked__safety__lora_r4"
    ent = _lora_only_entries(tmp_path)
    art = tmp_path / "stk.pt"
    art.write_bytes(b"synthetic stacked object bytes")
    ent[stacked] = dc.RegistryEntry(
        cell_id=stacked, kind="lora_all", si_id="safety", object_path=str(art),
        serving_path="lora", si_prefix=SAFETY_PREFIX,
        resolved_spec={"kind": "lora_all", "si_id": "safety", "layers": [0, 2],
                       "rank": 4, "positions": "all", "response_header_len": None})
    cells = ["none", "prompted__safety", "si_safety__lora_r4", stacked]
    with pytest.raises(dc.DefenseCellError, match="D6") as exc:
        _enum(_golden_spec(cells=cells), ent, lens_controls=lens_controls)
    assert stacked in str(exc.value) and "STACKED" in str(exc.value)
    # the same stack registered on the lens enumerates (a lens table) — the
    # ruling is about the PATH, not the cell
    ent[stacked] = dataclasses.replace(ent[stacked], serving_path="lens")
    cs = _enum(_golden_spec(cells=cells), ent)
    assert {c.cell_id for c in cs.canonical} >= {stacked}
    assert la.STACKED_KIND == cm.PETRI_STACKED_KIND  # the wrapper kind the predicate names


# ---------------------------------------------------------------------------
# 5. lora_controls (the zero-adapter-only standup's table) — additive; the
#    default enumeration is the golden
# ---------------------------------------------------------------------------


def test_lora_controls_false_is_the_default_and_equals_the_golden(tmp_path):
    spec = _golden_spec()
    assert _rows(_enum(spec, _golden_entries(tmp_path), lora_controls=False)) == GOLDEN_ROWS
    assert _rows(_enum(spec, _golden_entries(tmp_path))) == GOLDEN_ROWS


@pytest.mark.parametrize("lens_controls", [True, False])
def test_lora_controls_true_derives_none_lora0_with_no_lora_route(tmp_path, lens_controls):
    """A view with NO LoRA cell still stands up natively with the zero adapter
    alone (§1.2: base and steered configurations share an engine), so its table
    carries ``none__lora0`` although no lora route is live — and nothing else
    changes: the leg's identity, hash and vmodel are exactly the ones a live lora
    route would derive."""
    cells = ["none", "prompted__safety"]
    cs = _enum(_golden_spec(cells=cells), _lora_only_entries(tmp_path, lora=False),
               lens_controls=lens_controls, lora_controls=True)
    ids = [c.cell_id for c in cs.controls]
    assert ids == ([dc.NOOP_LENS_CELL_ID] if lens_controls else []) + [dc.NOOP_LORA_CELL_ID]
    by_id = {c.cell_id: c for c in cs.all_served()}
    lora0 = by_id[dc.NOOP_LORA_CELL_ID]
    assert lora0.control_of == "none" and lora0.cell_class == "control"
    assert lora0.serving_identity == "proxy_vllm+lora+exact_replay"
    assert lora0.intervention_hash == by_id["none"].intervention_hash
    assert lora0.virtual_model_name == "petri__none__lora0"
    # no lora route was live: every canonical leg is passthrough
    assert {c.serving_identity for c in cs.canonical} == {"proxy_vllm+passthrough+exact_replay"}
    # the same rows the live-lora table derives for the control leg (the zero
    # leg is a function of the base cell and the spec, not of the lora route)
    live = _enum(_golden_spec(cells=cells + ["si_safety__lora_r4"]), _lora_only_entries(tmp_path),
                 lens_controls=lens_controls)
    live0 = {c.cell_id: c for c in live.controls}[dc.NOOP_LORA_CELL_ID]
    for f in ("serving_identity", "intervention_hash", "virtual_model_name", "control_of", "kind"):
        assert getattr(live0, f) == getattr(lora0, f), f


def test_lora_controls_true_with_a_live_lora_route_derives_the_leg_once(tmp_path):
    cells = ["none", "prompted__safety", "si_safety__lora_r4"]
    cs = _enum(_golden_spec(cells=cells), _lora_only_entries(tmp_path),
               lens_controls=False, lora_controls=True)
    assert [c.cell_id for c in cs.controls] == [dc.NOOP_LORA_CELL_ID]
    cs_default = _enum(_golden_spec(cells=cells), _lora_only_entries(tmp_path), lens_controls=False)
    assert _rows(cs) == _rows(cs_default)


def test_lora_controls_needs_the_base_cell(tmp_path):
    """No base cell, no zero leg: the control is derived FROM ``none`` (its
    intervention hash is none's), so a view without it derives nothing."""
    ent = {k: v for k, v in _lora_only_entries(tmp_path, lora=False).items() if k != "none"}
    cs = _enum(_golden_spec(cells=["prompted__safety"]), ent, lens_controls=False, lora_controls=True)
    assert cs.controls == ()

