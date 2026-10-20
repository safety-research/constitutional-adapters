"""The standup half of thinking-ON serving (docs/served_thinking_on.md):
``build_server_launch(thinking_on=...)``'s render pin and its refusal on a
switch-less template, the OFF composition's BYTE-IDENTITY, and
``scripts/sdfbase/mt_serving_up.py``'s resolution pieces — the
certification-to-fact projection, the per-cell header-dependence
classification over real adaptable artifacts, ``resolve_thinking``'s
verdict handling (OFF records; ON refuses ``no_switch`` / ``refused``,
declares header-dependent cells only, declares nothing on an unmoved
header or an all-fit wave), the declaring source wrapper, the expected
attestation and its problem list, and the record's exact field names.
Plus source-text pins on the standup's wiring (the flag, the launch kwarg,
the refusal, the endpoint fields).

All offline, all synthetic.
"""

from __future__ import annotations

import dataclasses
import io
import json
import re
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sisteer import chat_markers as cmk  # noqa: E402
from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.serving import lens_steer as ls  # noqa: E402
from sisteer.capeval.serving import serve_declaration as sd  # noqa: E402
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402

from test_petri_proxy import _behavior_files, _spec  # noqa: E402
from test_thinking_on_chat_markers import TEMPLATES, _template_tok  # noqa: E402

D = 8
RESP_CELL = "si_safety__psr_kl_all"
ALL_CELL = "si_constitution__psr_kl_all"
SRC = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text(encoding="utf-8")


def _standup():
    import mt_serving_up
    return mt_serving_up


# --------------------------------------------------------------------------- #
# build_server_launch: the render pin                                          #
# --------------------------------------------------------------------------- #

THINK_TEMPLATE_BYTES = (b"{# synthetic reasoning template #}"
                        b"{% if enable_thinking %}<think>{% endif %}\n")
PLAIN_TEMPLATE_BYTES = b"{# synthetic plain template #}{{ messages }}\n"


def _tpl(tmp_path, data):
    p = tmp_path / "chat_template.jinja"
    p.write_bytes(data)
    return p


def _launch(tmp_path, data, **kw):
    return vs.build_server_launch(
        _spec(target_model="Qwen/Qwen3.5-4B", chat_template_hash=cm.sha256_hex16(data),
              serving_launch_flags=["--enforce-eager"], serving_mode="interleaved"),
        port=8801, chat_template_path=_tpl(tmp_path, data), **kw)


def test_thinking_on_pins_the_on_render_explicitly(tmp_path):
    launch = _launch(tmp_path, THINK_TEMPLATE_BYTES, thinking_on=True)
    cmd = list(launch.cmd)
    assert cmd[cmd.index("--default-chat-template-kwargs") + 1] == '{"enable_thinking": true}'
    assert launch.default_chat_template_kwargs == vs.THINKING_ON_TEMPLATE_KWARGS
    assert json.loads(vs.THINKING_ON_TEMPLATE_KWARGS) == {"enable_thinking": True}
    assert json.loads(vs.THINKING_OFF_TEMPLATE_KWARGS) == {"enable_thinking": False}
    # no reasoning parser is composed either way (wrapper-owned)
    assert "--reasoning-parser" not in cmd


def test_thinking_off_composition_is_byte_identical(tmp_path):
    default = _launch(tmp_path, THINK_TEMPLATE_BYTES)
    explicit = _launch(tmp_path, THINK_TEMPLATE_BYTES, thinking_on=False)
    assert default.cmd == explicit.cmd and dict(default.env) == dict(explicit.env)
    cmd = list(default.cmd)
    assert cmd[cmd.index("--default-chat-template-kwargs") + 1] == '{"enable_thinking": false}'
    on = _launch(tmp_path, THINK_TEMPLATE_BYTES, thinking_on=True)
    # the ON launch differs from the OFF one in exactly that one value
    off_cmd, on_cmd = list(default.cmd), list(on.cmd)
    assert len(off_cmd) == len(on_cmd)
    assert [(a, b) for a, b in zip(off_cmd, on_cmd) if a != b] == [
        ('{"enable_thinking": false}', '{"enable_thinking": true}')]
    assert dict(default.env) == dict(on.env)


def test_thinking_on_refuses_a_switchless_template(tmp_path):
    plain = _launch(tmp_path, PLAIN_TEMPLATE_BYTES)
    assert "--default-chat-template-kwargs" not in plain.cmd
    with pytest.raises(vs.VllmServerError, match="no thinking switch"):
        _launch(tmp_path, PLAIN_TEMPLATE_BYTES, thinking_on=True)
    with pytest.raises(vs.VllmServerError, match="must be a bool"):
        _launch(tmp_path, THINK_TEMPLATE_BYTES, thinking_on=1)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# mt_serving_up: the resolution pieces                                          #
# --------------------------------------------------------------------------- #


def _psr_blob(positions, hdr_len=None) -> bytes:
    g = torch.Generator().manual_seed(11)
    blob = {"kind": "psr", "layers": [0], "positions": positions,
            "directions": {"0": torch.randn(D, generator=g) * 0.05},
            "gate_w": {"0": torch.randn(D, generator=g) * 0.3},
            "gate_b": {"0": torch.randn(1, generator=g) * 0.5}}
    if hdr_len is not None:
        blob["response_header_len"] = hdr_len
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def _world(tmp_path, off_len, *, cells=(RESP_CELL, ALL_CELL)):
    """none + a RESPONSE-fit psr cell + an ALL-fit psr cell (the wave's shape)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    resp = tmp_path / "resp.pt"
    resp.write_bytes(_psr_blob("response", off_len))
    allp = tmp_path / "all.pt"
    allp.write_bytes(_psr_blob("all"))
    entries = {"none": dc.RegistryEntry(cell_id="none", kind="none")}
    for cid in cells:
        si = dc.parse_cell_id(cid).si_id
        art = resp if cid == RESP_CELL else allp
        pos = "response" if cid == RESP_CELL else "all"
        entries[cid] = dc.RegistryEntry(cell_id=cid, kind="psr", si_id=si, object_path=str(art),
                                        serving_path="lens",
                                        resolved_spec={"kind": "psr", "si_id": si, "positions": pos})
    source = dc.MappingCellSource(entries)
    spec = _spec(serving_mode="interleaved", cells=["none", *cells], prompted_prefix_hashes={})
    return spec, source


def test_certification_projects_to_the_proxy_fact():
    m = _standup()
    tok = _template_tok("qwen35_4b")
    cert = m.certify_thinking(tok, model_key="qwen35_4b")
    assert cert["model_key"] == "qwen35_4b" and cert["off_header_len"] == 7 and cert["on_header_len"] == 5
    fact, err = m.thinking_header_fact(cert)
    assert fact == cmk.thinking_header_facts(tok) and err is None
    gem = m.certify_thinking(_template_tok("gemma4e4b"), model_key="gemma4e4b")
    assert gem["verdict"] == "trivially_in_support" and m.thinking_header_fact(gem)[0]["moves"] is False
    refused = {**cert, "on_ids": None, "moves": None, "verdict": "refused", "reason": "x"}
    fact, err = m.thinking_header_fact(refused)
    assert fact == {"switch": True, "moves": None, "off_ids": cert["off_ids"], "on_ids": None}
    assert err == "refused: x"


def test_lens_header_dependence_classifies_by_adapting(tmp_path):
    m = _standup()
    tok = _template_tok("qwen35_4b")
    off = cmk.generation_header_ids(tok)
    spec, source = _world(tmp_path, len(off))
    dep = m.lens_header_dependence(spec, source, off_header_ids=off)
    assert dep == {RESP_CELL: "response", ALL_CELL: "all"}
    # an artifact the adapter refuses is a refusal, never a guess
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a blob")
    bad = dc.MappingCellSource({**source.entries, RESP_CELL: dc.RegistryEntry(
        cell_id=RESP_CELL, kind="psr", si_id="safety", object_path=str(broken), serving_path="lens",
        resolved_spec={"kind": "psr", "si_id": "safety", "positions": "response"})})
    with pytest.raises(SystemExit, match="cannot classify"):
        m.lens_header_dependence(spec, bad, off_header_ids=off)


def test_resolve_thinking_off_records_and_declares_nothing(tmp_path):
    m = _standup()
    tok = _template_tok("qwen35_4b")
    off = cmk.generation_header_ids(tok)
    spec, source = _world(tmp_path, len(off))
    res = m.resolve_thinking(tok, spec, source, thinking_on=False, model_key="qwen35_4b",
                             off_header_ids=off)
    assert res.mode == "off" and res.declared_cells == () and res.on_header_ids is None
    assert res.certification["verdict"] == "certified" and res.header["moves"] is True
    # OFF classifies no mask (classifying ADAPTS every lens artifact — the declaration's
    # cost, never a byte-identical OFF standup's): recorded as {} (not classified)
    assert res.header_dependence == {}
    rec = m.thinking_record(res)
    assert rec["thinking_mode"] == "off" and rec["thinking_declared_cells"] == []
    assert rec["assistant_header_ids_thinking_on"] is None
    # an OFF standup on a template whose ON header cannot be certified still
    # records the (partial) fact and its reason, and serves
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cmk, "THINKING_ON_REQUIRES_PLAIN_TURN_PREFIX", True)
        nemo = _template_tok("nemotron3nano4b")
        nemo_off = cmk.generation_header_ids(nemo)
        nspec, nsource = _world(tmp_path / "n", len(nemo_off))
        res = m.resolve_thinking(nemo, nspec, nsource, thinking_on=False, model_key="nemotron3nano4b",
                                 off_header_ids=nemo_off)
    assert res.mode == "off" and res.header["on_ids"] is None and res.header_error.startswith("refused:")
    assert res.certification["verdict"] == "refused"


def test_resolve_thinking_on_declares_the_header_dependent_cells_only(tmp_path):
    m = _standup()
    tok = _template_tok("qwen35_4b")
    off = cmk.generation_header_ids(tok)
    spec, source = _world(tmp_path, len(off))
    res = m.resolve_thinking(tok, spec, source, thinking_on=True, model_key="qwen35_4b",
                             off_header_ids=off)
    assert res.mode == "on" and res.declared_cells == (RESP_CELL,)
    assert res.on_header_ids == cmk.generation_header_ids_under(tok, enable_thinking=True)
    assert res.header["on_ids"] == list(res.on_header_ids) and res.header_error is None
    # DECLARED BY NAME (the merge onto main): the declared cell takes the `@thinking`
    # spelling in the spec, the source applies the overlay under that id, the all-fit
    # cell and the base entries are untouched; a spec that declared what its name did
    # not would be refused by the enumeration (defense_cells' name/spec agreement)
    declared_id = sd.join_variant(RESP_CELL, sd.VARIANT_THINKING)
    spec2 = dataclasses.replace(spec, cells=m.declared_cell_spellings(spec, res.declared_cells))
    assert spec2.cells == ["none", declared_id, ALL_CELL]
    base = source.get(RESP_CELL)
    wrapped = dc.MappingCellSource({**source.entries, declared_id: dataclasses.replace(
        base, cell_id=declared_id, resolved_spec=sd.declare_thinking(base.resolved_spec))})
    assert sd.declares_thinking(wrapped.get(declared_id).resolved_spec)
    assert not sd.declares_thinking(wrapped.get(ALL_CELL).resolved_spec)
    assert not sd.declares_thinking(source.get(RESP_CELL).resolved_spec)
    with pytest.raises(SystemExit, match="cannot declare"):
        m.declared_cell_spellings(spec2, [declared_id])           # one variant per cell
    # end to end: enumerate + export with the resolution, then the expected
    # attestation (declared cell + its twin) and a passing problem list
    cells = dc.enumerate_cells(spec2, wrapped, behavior_files=_behavior_files(spec2),
                               prompt_files={"auditor": b"# a\n", "judge": b"# j\n"})
    assert not cells.absent
    entries = ls.export_worker_registry(cells, wrapped, tmp_path / "reg", assistant_header_ids=off,
                                        assistant_header_ids_thinking_on=res.on_header_ids,
                                        zero_op_hidden_size=D)
    want = m.expected_declarations(entries)
    assert want == {declared_id: sd.thinking_declaration(),
                    dc.NOOP_LENS_CELL_ID: sd.thinking_declaration()}
    loaded = ls.WorkerObjectRegistry.load(tmp_path / "reg")
    assert loaded.serve_declarations_by_cell() == want
    rep = {"lens_serve_declarations": want, "assistant_header_ids_thinking_on": list(res.on_header_ids)}
    att = {"worker_reports": [dict(rep), dict(rep)]}
    problems, hdr, hdr_err, decls, decls_err = m.thinking_attestation_problems(
        att, mode=res.mode, want_header=res.on_header_ids, expected_decls=want)
    assert problems == [] and hdr == res.on_header_ids and decls == want
    # a worker attesting a different header, or a pre-landing worker, refuses
    bad = {"worker_reports": [dict(rep), {**rep, "assistant_header_ids_thinking_on": list(off)}]}
    assert m.thinking_attestation_problems(bad, mode=res.mode, want_header=res.on_header_ids,
                                           expected_decls=want)[0]
    old = {"worker_reports": [{"cells": ["x"]}]}
    assert len(m.thinking_attestation_problems(old, mode=res.mode, want_header=res.on_header_ids,
                                               expected_decls=want)[0]) == 2
    # the same pre-landing worker is only RECORDED on an OFF standup
    off_res = m.resolve_thinking(tok, spec, source, thinking_on=False, model_key="qwen35_4b",
                                 off_header_ids=off)
    problems, hdr, hdr_err, decls, decls_err = m.thinking_attestation_problems(
        old, mode=off_res.mode, want_header=off_res.on_header_ids, expected_decls={})
    assert problems == [] and hdr is None and "predates" in hdr_err and decls is None
    rec = m.thinking_record(res, launch=types.SimpleNamespace(
        default_chat_template_kwargs=vs.THINKING_ON_TEMPLATE_KWARGS), attested_header=hdr,
        attested_header_error=None, attested_declarations=want)
    assert rec["thinking_mode"] == "on" and rec["thinking_declared_cells"] == [RESP_CELL]
    assert rec["default_chat_template_kwargs"] == '{"enable_thinking": true}'
    assert rec["assistant_header_ids_thinking_on"] == list(res.on_header_ids)
    assert rec["lens_serve_declarations"] == want and rec["thinking_header_dependence"] == res.header_dependence


def test_resolve_thinking_on_an_all_fit_wave_declares_nothing(tmp_path):
    """This wave's shape: every sdfbase object is a positions=all fit, so a
    thinking-ON standup on a MOVED header declares no cell and exports no
    second header — the regime is carried by the launch pin, the proxy mode
    and the record."""
    m = _standup()
    tok = _template_tok("qwen35_4b")
    off = cmk.generation_header_ids(tok)
    spec, source = _world(tmp_path, len(off), cells=(ALL_CELL,))
    res = m.resolve_thinking(tok, spec, source, thinking_on=True, model_key="qwen35_4b",
                             off_header_ids=off)
    assert res.mode == "on" and res.declared_cells == () and res.on_header_ids is None
    assert res.header["moves"] is True and res.header_dependence == {ALL_CELL: "all"}
    cells = dc.enumerate_cells(spec, source, behavior_files=_behavior_files(spec),
                               prompt_files={"auditor": b"# a\n", "judge": b"# j\n"})
    entries = ls.export_worker_registry(cells, source, tmp_path / "reg", assistant_header_ids=off,
                                        assistant_header_ids_thinking_on=res.on_header_ids,
                                        zero_op_hidden_size=D)
    assert m.expected_declarations(entries) == {}
    # main's index states the second header as null when none was exported (the key is
    # always written on this tree; the sampled lineage omitted it) — no ON header, stated
    assert json.loads((tmp_path / "reg/registry.json").read_text())[ls.REGISTRY_THINKING_HEADER_KEY] is None


def test_resolve_thinking_on_an_unmoved_header_declares_nothing(tmp_path):
    m = _standup()
    tok = _template_tok("gemma4e4b")
    off = cmk.generation_header_ids(tok)
    spec, source = _world(tmp_path, len(off))
    res = m.resolve_thinking(tok, spec, source, thinking_on=True, model_key="gemma4e4b",
                             off_header_ids=off)
    assert res.mode == "on" and res.declared_cells == () and res.on_header_ids is None
    assert res.certification["verdict"] == "trivially_in_support" and res.header["moves"] is False


def test_resolve_thinking_on_refuses_an_uncertified_or_switchless_template(tmp_path):
    m = _standup()
    from test_thinking_on_chat_markers import _NoSwitchTok
    plain = _NoSwitchTok()
    off = cmk.generation_header_ids(plain)
    spec, source = _world(tmp_path, len(off))
    with pytest.raises(SystemExit, match="REFUSED.*no_switch"):
        m.resolve_thinking(plain, spec, source, thinking_on=True, model_key="llama8b",
                           off_header_ids=off)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cmk, "THINKING_ON_REQUIRES_PLAIN_TURN_PREFIX", True)
        nemo = _template_tok("nemotron3nano4b")
        nemo_off = cmk.generation_header_ids(nemo)
        nspec, nsource = _world(tmp_path / "n", len(nemo_off))
        with pytest.raises(SystemExit, match="REFUSED.*refused"):
            m.resolve_thinking(nemo, nspec, nsource, thinking_on=True, model_key="nemotron3nano4b",
                               off_header_ids=nemo_off)
    # the OFF standup on the same templates records and proceeds
    assert m.resolve_thinking(plain, spec, source, thinking_on=False, model_key="llama8b",
                              off_header_ids=off).mode == "off"


@pytest.mark.parametrize("key", sorted(TEMPLATES))
def test_every_wave_template_resolves_under_thinking_on(tmp_path, key):
    m = _standup()
    tok = _template_tok(key)
    off = cmk.generation_header_ids(tok)
    spec, source = _world(tmp_path, len(off))
    res = m.resolve_thinking(tok, spec, source, thinking_on=True, model_key=key, off_header_ids=off)
    assert res.mode == "on"
    if key == "gemma4e4b":
        assert res.declared_cells == () and res.on_header_ids is None
    else:
        assert res.declared_cells == (RESP_CELL,) and res.on_header_ids is not None
        assert tok.decode(list(res.on_header_ids)) == {
            "qwen35_4b": "<|im_start|>assistant\n<think>\n",
            "nemotron3nano4b": "<|im_start|>assistant\n<think>\n",
            "glm46vflash": "<|assistant|>\n"}[key]


def test_record_field_names_are_the_documented_ones():
    m = _standup()
    res = m.ThinkingResolution(mode="off", certification={"verdict": "certified"},
                               header={"switch": True, "moves": True, "off_ids": [1], "on_ids": [2]},
                               header_error=None, on_header_ids=None, declared_cells=(),
                               header_dependence={})
    assert set(m.thinking_record(res)) == {
        "thinking_mode", "thinking_declaration_key", "default_chat_template_kwargs",
        "thinking_header", "thinking_header_error", "thinking_certification",
        "thinking_declared_cells", "thinking_header_dependence",
        "assistant_header_ids_thinking_on", "attested_assistant_header_ids_thinking_on",
        "attested_assistant_header_ids_thinking_on_error", "lens_serve_declarations",
        "lens_serve_declarations_error"}
    assert m.thinking_record(res)["thinking_declaration_key"] == "petri_thinking_on"
    assert m.THINKING_MODES == ("off", "on") and m.THINKING_SERVABLE_VERDICTS == (
        "certified", "trivially_in_support")


# --------------------------------------------------------------------------- #
# source-text pins on the standup's wiring                                      #
# --------------------------------------------------------------------------- #


def test_standup_wiring_source_pins():
    """The merged wiring (the sampled-rerun standup ported onto main's per-cell
    path): the flag, the launch kwarg, the resolution over the (possibly
    --cell-narrowed) spec, the declaration BY NAME, the export of the
    registry's own derivation held equal to the certification, the keyword
    attestation check, the proxy mode beside main's attested kwargs, and the
    record on both artifacts."""
    assert 'ap.add_argument("--thinking-on", action="store_true"' in SRC
    assert "thinking_on=args.thinking_on," in SRC                     # the launch kwarg
    assert "reasoning_parser=args.reasoning_parser," in SRC           # main's parser kwarg, untouched ...
    assert "server_default_policy=args.server_default_policy)" in SRC  # ... closed by the additive kwarg (ae212d50)
    assert "resolve_thinking(tok, spec, src, thinking_on=args.thinking_on" in SRC
    assert "spec = dataclasses.replace(spec, cells=declared_cell_spellings(spec, thinking.declared_cells))" in SRC
    assert "ThinkingDeclaredSource" not in SRC                       # declared by NAME, never by spec alone
    # the registry exports ITS derivation (the header every worker cross-checks); the
    # certification must agree with it, or the standup refuses before any GPU minute
    assert '**({"assistant_header_ids_thinking_on": on_header_ids} if on_header_ids is not None else {})' in SRC
    assert "tuple(on_header_ids) != tuple(thinking.on_header_ids)" in SRC
    assert ("thinking_attestation_problems(attestation, mode=thinking.mode,\n"
            "                                      want_header=thinking.on_header_ids, expected_decls=want_decls)") in SRC
    assert "thinking-mode attestation FAILED" in SRC
    assert "thinking_mode=thinking.mode," in SRC and "**proxy_kwargs)" in SRC
    assert SRC.count("**thinking_record(thinking),") == 1 and SRC.count("**thinking_rec,") == 1
    assert "cs.update(thinking_rec)" in SRC
    # the proxy -> vLLM hop timeout IN FORCE is recorded beside main's flag-as-given
    assert '"upstream_timeout_s": (float(args.upstream_timeout_s) if args.upstream_timeout_s is not None' in SRC
    assert '"proxy_upstream_timeout_s": args.upstream_timeout_s,' in SRC
    # the OFF pin stays the tokenizer's resting state (every render site expects it)
    assert "pin_reasoning_template_off(tok)" in SRC
    # the pre-existing wiring the byte-identity depends on is untouched
    assert 'assistant_header_ids=header_ids,' in SRC
    assert re.search(r"--reduction-mode.*default=\"per_row\"", SRC)
