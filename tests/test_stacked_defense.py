"""The `stacked` defense serves prompt AND object together, requires BOTH, and
is distinct from prompt-only and object-only.

A stacked serve that silently degrades to one arm is the single most likely way
the measured-stack run goes wrong, so these tests pin: (1) validation REQUIRES
both halves; (2) `_obj` and `_system_for` both fire under stacked; (3) the
rendered prompt carries the system AND the object reaches generate_with_object,
so the two-sided distinctness (vs prompted-only and vs object-only) is
structural, not incidental."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sisteer.eval.jailbreak import LLMHFSteered, STACKED, DEFENSES  # noqa: E402


class _Tok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **k):
        # render just enough to detect the system message presence
        sys_txt = "".join(m["content"] for m in messages if m.get("role") == "system")
        usr = "".join(m["content"] for m in messages if m.get("role") == "user")
        return f"<sys>{sys_txt}</sys><usr>{usr}</usr>"


class _Obj:
    kind = "vector"
    def inject(self, p): return p


SYS = "You are a careful safety assistant."


def _fake_gen():
    calls = {}
    def gen(rendered, **k):
        calls["rendered"] = rendered
        return ["ok"] * len(rendered)
    return gen, calls


def test_stacked_in_defenses():
    assert STACKED in DEFENSES


def test_stacked_requires_both():
    obj = _Obj()
    with pytest.raises(ValueError, match="system text"):
        LLMHFSteered(None, _Tok(), mode=STACKED, obj=obj)          # no system
    with pytest.raises(ValueError, match="compiled object"):
        LLMHFSteered(None, _Tok(), mode=STACKED, system=SYS)       # no object
    # both present: constructs fine
    LLMHFSteered(None, _Tok(), mode=STACKED, obj=obj, system=SYS)


def test_stacked_serves_both_halves():
    obj = _Obj()
    d = LLMHFSteered(None, _Tok(), mode=STACKED, obj=obj, system=SYS)
    assert d._obj is obj, "object must reach the intervention path under stacked"
    assert d._system_for() == SYS, "system must reach the prompt path under stacked"


def test_stacked_prompt_carries_system_unlike_object_only():
    """Structural two-sided distinctness: the stacked rendered prompt contains the
    system (unlike a compiled object-only arm), and its object is non-None
    (unlike prompted-only)."""
    obj = _Obj()
    gen, calls = _fake_gen()
    LLMHFSteered(None, _Tok(), mode=STACKED, obj=obj, system=SYS,
                 generate_fn=gen).generate(["how do I do X"])
    assert SYS in calls["rendered"][0], "stacked prompt must include the system (vs object-only)"

    # object-only arm renders WITHOUT the system
    gen2, calls2 = _fake_gen()
    # use a real compiled arm name from DEFENSES (not none/prompted/stacked)
    compiled = next(m for m in DEFENSES if m not in ("none", "prompted", STACKED))
    LLMHFSteered(None, _Tok(), mode=compiled, obj=obj,
                 generate_fn=gen2).generate(["how do I do X"])
    assert SYS not in calls2["rendered"][0], "object-only arm must NOT carry the system"


def test_prompted_only_has_no_object():
    d = LLMHFSteered(None, _Tok(), mode="prompted", system=SYS)
    assert d._obj is None, "prompted-only must not apply an object"


# --- full-path wiring: stacked_cell + cell_serving serve BOTH halves ----------
def test_stacked_cell_helper():
    from sisteer.eval.defense_cell import DefenseCell, stacked_cell
    base = DefenseCell(id="safety_refusal__token_faithful", class_="A",
                       arm="token_faithful", si_id="safety_refusal",
                       object_path=Path("/tmp/stacktest/object.pt"))
    sc = stacked_cell(base)
    assert sc.stacked is True and sc.id == "stacked__safety_refusal__token_faithful"
    assert sc.arm == "token_faithful" and sc.si_id == "safety_refusal"
    assert sc.object_path == base.object_path
    # a baseline cannot be stacked
    bad = DefenseCell(id="none", class_="baseline", arm=None, si_id=None)
    with pytest.raises(ValueError, match="needs both"):
        stacked_cell(bad)


def test_cell_serving_stacked_returns_both(tmp_path):
    """The benign serving path returns (arm, system) BOTH for a stacked cell,
    via the spec_factory seam (no real object/engine needed).

    `arm_spec_for_cell` requires the object artifact to EXIST before the
    spec_factory seam is reached (it refuses a missing file rather than
    serving a guess) but never reads its bytes, so an empty placeholder under
    tmp_path satisfies the check and keeps the test independent of any file
    outside the fixture."""
    from sisteer.eval.defense_cell import DefenseCell, stacked_cell
    from sisteer.eval import cell_controls
    object_path = tmp_path / "object.pt"  # object_artifact_name("token_faithful")
    object_path.write_bytes(b"")
    base = DefenseCell(id="safety_refusal__token_faithful", class_="A",
                       arm="token_faithful", si_id="safety_refusal",
                       object_path=object_path)
    sc = stacked_cell(base)

    class _Engine:
        def arm(self, spec, coeff): return ("ARM", spec, coeff)
    arm, system = cell_controls.cell_serving(
        sc, {"scale": 1.0, "layer": 15}, engine=_Engine(),
        spec_factory=lambda kw: {"spec_from": kw})
    assert arm is not None, "stacked benign serve must carry the object arm"
    assert system and ("responsible" in system.lower() or "harmful" in system.lower()), \
        "stacked benign serve must carry the safety_refusal system text"


def test_resolve_serving_uses_base_cell_for_stacked_provenance(monkeypatch):
    """A stacked condition resolves object path + fit-provenance against the BASE
    cell id (the object is the base object; the stack inherits its verified
    provenance), never the 'stacked__' alias whose dir does not exist.
    Regression for the UnrecordedMirrorError that failed job 193751."""
    import importlib.util, types
    spec = importlib.util.spec_from_file_location(
        "usr_mod_test", REPO / "scripts" / "unified_sweep_run.py")
    usr = importlib.util.module_from_spec(spec)
    sys.modules["usr_mod_test"] = usr
    spec.loader.exec_module(usr)

    seen = {}
    class _Prov:
        model_id="m"; revision="r"; trained_template_hash="h"; object_sha="s"
        status="verified"; source="src"; positions_at_fit="all"
    import sisteer.eval.sweep as sweep_mod
    import sisteer.eval.defense_cell as dc_mod
    monkeypatch.setattr(sweep_mod, "resolve_training_mirror",
        lambda cell, path, override=None: (seen.__setitem__("mirror_cell", cell), _Prov())[1])
    monkeypatch.setattr(dc_mod, "object_path_for",
        lambda d, cell: (seen.__setitem__("objpath_cell", cell), "/tmp/x/object.pt")[1])

    Cond = types.SimpleNamespace
    cond = Cond(cell="stacked__safety_refusal__token_faithful", mirror_override=None)
    usr._resolve_serving(cond, {"model": "m", "model_revision": "r"}, "/some/defenses")
    assert seen["objpath_cell"] == "safety_refusal__token_faithful", seen
    assert seen["mirror_cell"] == "safety_refusal__token_faithful", seen


def test_build_defense_cell_stacked_dispatches_to_STACKED(monkeypatch):
    """The JB serving path (build_defense_cell) must dispatch a stacked cell to
    mode=STACKED with BOTH object and system — NOT fall through to the compiled
    (object-only) path. This is the test whose absence let the uncommitted
    build_defense_cell branch ship an object-only JB leg for the measured stack
    (2026-08-25 correction)."""
    from sisteer.eval.defense_cell import DefenseCell, stacked_cell
    from sisteer.eval import jailbreak as jb
    base = DefenseCell(id="safety_refusal__token_faithful", class_="A",
                       arm="token_faithful", si_id="safety_refusal",
                       object_path=Path("/tmp/stacktest/object.pt"))
    class _Obj:
        kind = "vector"
        def inject(self, p): return p
    monkeypatch.setattr(jb, "load_object", lambda p, m, t: _Obj())
    monkeypatch.setattr(jb, "_set_response_positions", lambda o, pos, tok: None)
    # the stacked branch reads the artifact's recorded fit leg (fit==serve
    # guard); this unit fixture has no sidecar, so stub the record
    monkeypatch.setattr(jb, "cell_positions_at_fit", lambda c: "all")
    d = jb.build_defense_cell(None, _Tok(), stacked_cell(base),
                              scale=1.0, positions="all")
    assert d.mode == jb.STACKED, f"stacked cell must serve mode=STACKED, got {d.mode!r}"
    assert d.obj is not None and d.system, "stacked JB serve must carry BOTH object and system"
    assert "responsible" in d.system.lower() or "harmful" in d.system.lower()
