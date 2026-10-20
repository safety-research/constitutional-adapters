"""The LoRA-kernel-ceiling lowering moves S and ONLY S (2026-09-17, on the arditi_allpos GSM8K leg 544615).

`native_standup.lora_kernel_ceiling_geometry` re-resolves the geometry at `ceiling // L`. It used to pass the unchanged
window L back explicitly, and `geometry.resolve_geometry` stamps `window_source: override` for ANY non-None window — so
a window that IS the table's was recorded as overridden, and the capability readers' shared leg check refuses
`window_source != table` before it reads the adjustment: every ceiling-lowered native leg read as unusable. Now a
TABLE-sourced window is re-resolved from the table (its true source survives), an OVERRIDDEN window is passed back
explicitly (where `override` is honest, and re-resolving from the table would silently change the served window — the
multi-turn cap-4096 case), and a window that MOVES between the two resolutions refuses."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sisteer.capeval.serving import geometry as sg  # noqa: E402
from sisteer.capeval.serving import native_standup as ns  # noqa: E402

L = 4864
EVAL_4864 = next(k for k in sg.KNOWN_KEYS if sg.TABLE[k].max_model_len == L)
GEMMA_D = sg.LORA_MAX_MODULE_DIM_BY_MODEL["gemma4e4b"]     # the real dim: 121 -> 107 at 4864


def _blocks_with_dim(monkeypatch, D):
    """Fake the banked adapters' module dims so the ceiling binds: `adapter_module_dims` is read per block dir."""
    monkeypatch.setattr(ns.sa, "adapter_module_dims", lambda _dir: {"o_proj": (D, D)})
    monkeypatch.setattr(ns.sa, "max_module_dim", lambda dims: (D, "o_proj"))
    return {"cell": {"dir": "/nonexistent/adapter"}}


def _lowered(monkeypatch, geom, D, **kw):
    blocks = _blocks_with_dim(monkeypatch, D)
    record, lowered = ns.lora_kernel_ceiling_geometry(blocks, geom, eval_name=EVAL_4864, model_key="gemma4e4b",
                                                      allow_over_budget=False, override_reason=None, **kw)
    return record, lowered


def test_a_table_window_keeps_its_table_source_and_only_S_moves(monkeypatch):
    ident = sg.resolve_geometry(EVAL_4864, model_key="gemma4e4b")
    assert ident.window_source == "table" and ident.max_model_len == L
    record, lowered = _lowered(monkeypatch, ident, GEMMA_D)     # the real gemma dim: 121 -> 107
    assert lowered is not None and lowered["S"] < ident.max_num_seqs and lowered["L"] == L
    g = lowered["_geom"]
    assert g.max_model_len == L and g.max_num_seqs == lowered["S"]
    assert g.seqs_source == "override"                     # S was lowered: honest
    assert g.window_source == "table"                      # THE FIX: the window was never overridden


def test_an_overridden_window_is_passed_back_explicitly_and_stays_overridden(monkeypatch):
    """The multi-turn cap-4096 shape: the caller overrode the window on purpose; re-resolving from the table would
    SILENTLY change the served window, so the lowering keeps passing it explicitly and `override` stays true."""
    base = sg.resolve_geometry(EVAL_4864, max_model_len=L, max_num_seqs=121, strict_identity=False, model_key="gemma4e4b")
    assert base.window_source == "override"
    seen = {}
    real = ns._resolve_geometry
    def spy(eval_name, max_model_len, max_num_seqs, **kw):
        seen["max_model_len"] = max_model_len
        return real(eval_name, max_model_len, max_num_seqs, **kw)
    monkeypatch.setattr(ns, "_resolve_geometry", spy)
    _, lowered = _lowered(monkeypatch, base, GEMMA_D)
    assert lowered is not None and seen["max_model_len"] == L      # passed EXPLICITLY, not None
    assert lowered["_geom"].window_source == "override" and lowered["_geom"].max_model_len == L


def test_a_window_that_moves_between_the_two_resolutions_refuses(monkeypatch):
    """Known-bad: the table entry changed under the standup — the lowering must refuse rather than serve a window
    nobody resolved. Simulated by a resolver that hands back a different L."""
    ident = sg.resolve_geometry(EVAL_4864, model_key="gemma4e4b")
    moved = replace(ident, max_model_len=L + 1024, max_num_seqs=100, seqs_source="override")
    monkeypatch.setattr(ns, "_resolve_geometry", lambda *a, **k: moved)
    with pytest.raises(ns.StandupRefused, match="changed the WINDOW"):
        _lowered(monkeypatch, ident, GEMMA_D)
