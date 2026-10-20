"""The Nemotron backend probe's ``gen_long`` geometry — the long-generation
capability and agentic rows' OWN entry.

The worker-side gate (``nemotron_worker_admission``, ifeval-canonical-battery fix lane X) reads the fused-Mamba
certificate at a key over the ROW's config digest, this tree, the reduction mode, the serving ENGINE and the
kernel-adjusted served L x S of THIS configuration; the writer (``--nemotron-probe --geometry <g>``) certified
IFEval entries only, so no shipped path could write the certificate a native BCB/BFCL container wanted (the nemotron
canary died at the gate: 16,384 x 48, ``capability:gen_long``, ``lora_native``). ``gen_long`` plans the lane row's own
pass-1 ``none`` configuration — protocol sampling, thinking off, the lane roster's runnable legs — the SAME plan the
wave launch makes, so the certificate is for the entry the wave's containers serve; refused by name on a non-lane row,
under any sampling but protocol, and on a tree without the lane roster. Every IFEval token is untouched (their pins
live in test_tinycap_launcher_port / test_ifeval_container_layout / test_nemotron_probe_cpu)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import widen_roster as W

import test_tinycap_launcher as TL  # noqa: E402  (the modal stub, the by-path loader)

REPO = Path(__file__).resolve().parents[1]
NEMO_S1, NEMO_S1P = "nemotron3nano4b-f22s1", "nemotron3nano4b-f22s1p"
WAVE_OF = {NEMO_S1: "nemotron3nano4b_s1", NEMO_S1P: "nemotron3nano4b_s1p"}


@pytest.fixture(scope="module")
def L():
    return TL._load_launcher(TL._stub_modal(), app_name=TL.TEST_APP_NAME, name="tinycap_served_gen_long_under_test")


def test_gen_long_is_the_last_probe_geometry_and_names_no_ifeval_cap(L):
    assert L.PROBE_GEOMETRIES[-1] == L.PROBE_GEOMETRY_GEN_LONG == "gen_long"
    assert L.PROBE_GEOMETRIES[:-1] == ("2048", "4096", "8192", "longsi_2048", "longsi_4096", "longsi_8192")   # unchanged
    assert L.PROBE_GEOMETRY_DEFAULT == "2048"
    assert L.probe_geometry_cap("gen_long") == (False, None)
    assert L.probe_geometry_cap("2048") == (False, 2048) and L.probe_geometry_cap("longsi_8192") == (True, 8192)
    with pytest.raises(ValueError, match="--geometry must be one of"):
        L.probe_geometry_cap("16384")
    assert L.probe_entry_label("gen_long").startswith("capability:gen_long (")
    assert L.probe_entry_label("2048") == "capability:ifeval_2048" and L.probe_entry_label("longsi_4096") == "capability:ifeval_longsi_4096"


def test_gen_long_plans_the_lane_rows_own_pass1_none_configuration(L):
    """The probe's configuration IS the wave launch's: `none` x off x the wave's runnable legs under protocol, the pooled
    keys and the pass-1 geometry identical to what `plan_configurations` gives the wrapper's argv — 16,384 x 48 for the
    nemotron rows (the gate's own numbers); the native key sits at the kernel-adjusted S below it."""
    roster = W.load_roster()
    digests = {}
    for slug, wave in WAVE_OF.items():
        row = C.model_row(slug)
        assert C.is_full22_row(row) and row.family == "nemotron_h"
        legs = W.runnable_legs(roster, wave)
        assert "bigcodebench_200" in legs
        assert L.bcbw_probe_legs(row, None) == legs == L.bcbw_probe_legs(row, REPO)
        cfg = L.probe_geometry_cfg(row, "gen_long", reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT,
                                   sampling_mode=C.SAMPLING_PROTOCOL)
        assert (cfg["cell"], cfg["mode"], cfg["model"], list(cfg["controls"])) == ("none", C.THINKING_OFF, slug, legs)
        assert "capability:gen_long" in cfg["geometry_keys"]
        assert ("capability:agent" in cfg["geometry_keys"]) is ("bfcl_200" in legs)
        (wave_cfg,) = L.plan_configurations(cells=["none"], modes=[C.THINKING_OFF], controls=legs, model=slug,
                                            sampling_mode=C.SAMPLING_PROTOCOL)
        assert cfg == wave_cfg
        geom, _ = L.served_geometry(cfg, 1)
        wgeom, _ = L.served_geometry(wave_cfg, 1)
        assert (geom.max_model_len, geom.max_num_seqs) == (wgeom.max_model_len, wgeom.max_num_seqs) == (16384, 48)
        key_geom = L.nemotron_key_geometry(geom, L.ENGINE_LORA_NATIVE)
        assert key_geom.max_model_len == 16384 and 0 < int(key_geom.max_num_seqs) < 48      # the §1.8 kernel step
        assert L.nemotron_key_geometry(geom, L.ENGINE_LENS) is geom                          # the lens key: the requested geometry
        digests[slug] = L.sha256_file(REPO / row.cfg)
    # one certificate per ROW: the key holds the row's config digest, and the two rows' configs differ
    assert digests[NEMO_S1] != digests[NEMO_S1P]


def test_gen_long_refuses_a_non_lane_row_a_non_protocol_draw_and_a_tree_without_the_lane_roster(L, tmp_path):
    ife = C.model_row("nemotron3nano4b-sdfbase-main")
    assert not C.is_full22_row(ife) and ife.family == "nemotron_h"
    with pytest.raises(ValueError, match="certifies a bcb-bfcl-200-widen lane row"):
        L.probe_geometry_cfg(ife, "gen_long", reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT, sampling_mode=C.SAMPLING_PROTOCOL)
    row = C.model_row(NEMO_S1)
    with pytest.raises(ValueError, match="draw under --sampling 'protocol'"):
        L.probe_geometry_cfg(row, "gen_long", reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT, sampling_mode=C.SAMPLING_DEFAULT)
    # the legs come from the TREE's lane roster: no roster, no wave for the row, no runnable leg — each refused by name
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(ValueError, match="no lane roster at"):
        L.bcbw_probe_legs(row, bare)
    doc = yaml.safe_load((REPO / W.ROSTER_REL).read_text(encoding="utf-8"))
    without = copy.deepcopy(doc)
    for w in [n for n, b in without["waves"].items() if b.get("model_row") == NEMO_S1]:
        del without["waves"][w]
    t1 = tmp_path / "t1"
    (t1 / W.ROSTER_REL).parent.mkdir(parents=True)
    (t1 / W.ROSTER_REL).write_text(yaml.safe_dump(without, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="no wave of .* names model_row 'nemotron3nano4b-f22s1'"):
        L.bcbw_probe_legs(row, t1)
    blocked = copy.deepcopy(doc)
    for n, b in blocked["waves"].items():
        if b.get("model_row") == NEMO_S1:
            b["legs"] = {leg: {"status": "blocked", "reason": "test", "unblock": "test"} for leg in b["legs"]}
    t2 = tmp_path / "t2"
    (t2 / W.ROSTER_REL).parent.mkdir(parents=True)
    (t2 / W.ROSTER_REL).write_text(yaml.safe_dump(blocked, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="have no runnable leg"):
        L.bcbw_probe_legs(row, t2)


def test_gen_long_reads_the_live_parser_verdict_off_the_staged_tree_never_off_the_module(L, monkeypatch):
    """The first gen_long attempt in a container (2026-09-23, runs 5514a6f-d2432796 / -a46f451b) returned IFEVAL_ROW_HELD:
    with nemotron's bfcl_200 runnable the pass-1 plan runs the BFCL live-verdict gate, and the plan was made WITHOUT the
    staged tree — inside the image `contracts` sits at /root/sisteer with no configs/ beside it, so the gate refused ("no
    tree to read the live verdict … pass tree="). The container condition here: `source_checkout_root` answers None (the
    module is not imported from a checkout); the plan handed the tree succeeds, the plan handed none refuses by name."""
    row = C.model_row(NEMO_S1)
    legs = W.runnable_legs(W.load_roster(), WAVE_OF[NEMO_S1])
    assert "bfcl_200" in legs, legs                                                   # the gate is exercised only with BFCL planned
    monkeypatch.setattr(C, "source_checkout_root", lambda: None)
    assert "no tree to read the live verdict" in C.tool_parser_live_gate_problem(row.family)
    assert C.tool_parser_live_gate_problem(row.family, tree=REPO) is None
    cfg = L.probe_geometry_cfg(row, "gen_long", reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT,
                               sampling_mode=C.SAMPLING_PROTOCOL, ifeval_matrix_tree=REPO)
    assert list(cfg["controls"]) == legs and cfg["tool_calls"] is True
    with pytest.raises(ValueError, match="no tree to read the live verdict"):
        L.probe_geometry_cfg(row, "gen_long", reduction_mode=C.REDUCTION_MODE_LAUNCHER_DEFAULT,
                             sampling_mode=C.SAMPLING_PROTOCOL, ifeval_matrix_tree=None)
    # the probe body hands its resolved staged tree to the plan (the pinned call text: ifeval_matrix_tree=tree)
    src = TL.LAUNCHER.read_text(encoding="utf-8")
    body = src[src.index("def _nemotron_probe("):src.index("def nemotron_backend_probe(")]
    assert "ifeval_matrix_tree=tree)" in body
    gl = src[src.index("if geometry == PROBE_GEOMETRY_GEN_LONG:"):src.index("longsi, cap = probe_geometry_cap(geometry)")]
    assert "ifeval_matrix_tree=ifeval_matrix_tree)" in gl and "bcbw_probe_legs(row, ifeval_matrix_tree)" in gl


def test_the_dry_run_prints_the_gen_long_probe_line_and_plans_no_configuration_for_a_lane_row(L, capsys):
    """The real probe-only launch (`--nemotron-probe --geometry gen_long`, no pass) plans nothing, so the dry run plans
    nothing either (a lane row's configurations launch only through the wrapper's snapshot) and prints the probe's own
    configuration: the legs, the pooled keys, the pass-1 geometry and the key's kernel-adjusted L x S."""
    legs = W.runnable_legs(W.load_roster(), WAVE_OF[NEMO_S1])
    (cfg,) = L.plan_configurations(cells=["none"], modes=[C.THINKING_OFF], controls=legs, model=NEMO_S1, sampling_mode=C.SAMPLING_PROTOCOL)
    L.main(app_name=L.APP_NAME, sha="f6cf462", model=NEMO_S1, nemotron_probe=True, geometry="gen_long",
           sampling=C.SAMPLING_PROTOCOL, serving_path=L.SERVING_PATH_LORA_NATIVE, dry_run=True)
    out = capsys.readouterr().out
    assert f"the gen_long configuration is {NEMO_S1} x none, controls {legs}, keys {list(cfg['geometry_keys'])}, pass-1 geometry 16384 x 48" in out
    assert "capability:gen_long" in out
    assert "the certificate key's L x S is 16384 x " in out and "(lora_native)" in out
    assert f"nemotron probe (dry run): {NEMO_S1} x none at capability:gen_long (" in out
    assert "dry run: 0 GPU configuration container(s)" in out
    # a non-lane row is refused by name before any plan
    with pytest.raises(SystemExit, match="certifies a bcb-bfcl-200-widen lane row"):
        L.main(app_name=L.APP_NAME, sha="f6cf462", model="nemotron3nano4b-sdfbase-main", nemotron_probe=True, geometry="gen_long",
               sampling=C.SAMPLING_PROTOCOL, serving_path=L.SERVING_PATH_LORA_NATIVE, dry_run=True)
    # ... and a card draw
    with pytest.raises(SystemExit, match="draw under --sampling 'protocol'"):
        L.main(app_name=L.APP_NAME, sha="f6cf462", model=NEMO_S1, nemotron_probe=True, geometry="gen_long",
               serving_path=L.SERVING_PATH_LORA_NATIVE, dry_run=True)
    # the launcher argv the wrapper / operator composes carries the probe flag pair last (as for every IFEval entry)
    argv = L.launch_argv("a" * 40, model=NEMO_S1, nemotron_probe=True, geometry="gen_long", sampling=C.SAMPLING_PROTOCOL)
    assert argv[-3:] == [L.FLAG_NEMOTRON_PROBE, L.FLAG_GEOMETRY, "gen_long"] and "--pass1" not in argv
