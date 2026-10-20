"""The SHORT constitution-SI arm as a prompted-only view.

The paper's constitution-SI arm is the SHORT registry SI ``si_constitution`` (31,222 bytes, sha256
21e06f48…6a9a0f) — NOT the whole document ``si_constitution_full`` (the Llama-only appendix view
``fullconst``). The lane adds it as view ``shortconst`` MATCHING THE SAMPLED LANE'S MECHANISM: cell
``prompted__si_constitution``, harness ``prompted__constitution``, virtual model
``petri__prompted__constitution``; the text resolved only through registry_adapter -> TrainTest
``resolve_instruction`` and pinned by hash (the proxy verifies every request, H8; every standup writes
``si_prefix_record.json`` with the FULL sha256 / byte length / first + last line cross-checked against the
routing table, refusing traffic on a mismatch); every leg at PER-LEG windows max-prompt-sized for that
leg's cap. This test pins each seam: the SI identity through the served resolver, the record writer's
presence at every standup, the window check's per-phase arguments, and the sbatch scripts' per-leg reads.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import registry_adapter as ra
from sisteer.data import instructions as ins

REPO = Path(__file__).resolve().parents[1]
F22 = REPO / "scripts" / "sdfbase" / "full22"
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(F22))
import f22_roster as fr  # noqa: E402
import prompted_window_check as pwc  # noqa: E402

SHA256 = "21e06f482a6559b39309c4f1128f0411230c88e02b3a9eaa7b5d50829e6a9a0f"
ROSTER = fr.load_roster()


def test_the_short_si_resolves_through_the_served_harness_path_to_the_pinned_bytes():
    """The FOURTH independent measurement of the same bytes (three lanes agree): registry_adapter._traintest_si_text_bytes ->
    TrainTest resolve_instruction, never a source file; its hex16 IS the manifest's PREFIX_HASHES.constitution."""
    data = bytes(ra._traintest_si_text_bytes("si_constitution"))
    assert hashlib.sha256(data).hexdigest() == SHA256 and len(data) == 31222
    lines = data.decode("utf-8").splitlines()
    assert len(lines) == 121 and lines[0] == "Claude’s core values"
    assert lines[-1].endswith("to appropriate humans if asked to do so.")
    assert cm.sha256_hex16(data) == SHA256[:16] == "21e06f482a6559b3"
    assert ra.harness_cell_id("prompted__si_constitution") == "prompted__constitution"
    assert ra.traintest_cell_id("prompted__constitution") == "prompted__si_constitution"
    import make_full22_configs as gen
    assert gen.PREFIX_HASHES["constitution"] == SHA256[:16] and gen.harness_cell_id("prompted__si_constitution") == "prompted__constitution"
    # distinct from the full document's arm end to end
    full = ins.BY_ID["si_constitution_full"].text.encode("utf-8")
    assert cm.sha256_hex16(full) == "82f88c5532e4939e" != SHA256[:16] and len(full) != len(data)
    from sisteer.capeval.petri import defense_cells as dc
    assert dc.virtual_model_name("prompted__constitution") == "petri__prompted__constitution"      # the proxy's DISTINCT prompted vmodel (H8)
    assert dc.virtual_model_name("prompted__constitution_full") == "petri__prompted__constitution_full" != dc.virtual_model_name("prompted__constitution")


def test_the_roster_view_and_its_manifest_are_the_sampled_lanes_mechanism():
    v = ROSTER["targets"]["llama8b"]["views"]["shortconst"]
    assert v["prompted_only"] is True and v["cells"] == {"prompted__si_constitution": None} and "legs" not in v
    assert fr.view_legs(ROSTER, "llama8b", "shortconst") == fr.LEGS
    spec = cm.load_petri_spec(REPO / v["manifest"])
    assert spec.cells == ["none", "prompted__constitution"] and spec.prompted_prefix_hashes["constitution"] == SHA256[:16]
    assert "constitution_full" not in spec.prompted_prefix_hashes
    cfg = yaml.safe_load((REPO / v["config"]).read_text())
    assert cfg["explicit_cells"] == ["none", "prompted__si_constitution"] and cfg["bon"]["max_new_tokens"] == 4096
    text = (REPO / v["config"]).read_text()
    assert "si_prefix sha16 21e06f482a6559b3 (verified per request, H8)" in text and "PER-LEG windows" in text
    assert "static + BoN 10496; controls (MMLU letter prefill + xstest / or_bench) 6656; ladder rungs 2048 -> 8064, 8192 -> 14592, 16384 -> 22784" in text
    assert "pair 18688" in text and "Petri at the certified 16,384" in text


def test_every_standup_writes_the_si_prefix_record_and_refuses_traffic_on_a_mismatch():
    common = (F22 / "common.sh").read_text()
    assert "f22_record_si_prefix(){" in common
    i_fn = common.index("f22_record_si_prefix(){")
    body = common[i_fn:i_fn + 6000]
    assert "ra._traintest_si_text_bytes(si_id)" in body and "ra.harness_cell_id(cell)" in body     # the served resolver, the harness translation
    assert '"sha256": full' in body and '"byte_length": len(data)' in body and '"first_line"' in body and '"last_line"' in body
    assert "prompted_prefixes" in body and 'spec["sha16"] != full[:16]' in body and "sys.exit(1)" in body
    assert 'REFUSING TRAFFIC: the SI prefix record disagrees with the routing table' in body
    # f22_standup calls it after the load check, and a refusal touches STOP (no traffic)
    i_up = common.index('f22_log "target up:')
    i_lc = common.index("scripts/sdfbase/full22/load_check.py --endpoint")
    i_rec = common.index('f22_record_si_prefix "$out" || { touch "$out/STOP"; return 1; }')
    assert i_up < i_lc < i_rec < i_fn
    # the sampled sbatch names the sampled lane's function (an alias of the one writer) after each of its standups
    sampled = (F22 / "sampled" / "common_sampled.sh").read_text()
    assert "f22s_record_si_prefix(){ f22_record_si_prefix" in sampled
    src = (F22 / "sampled" / "target_sampled.sbatch").read_text()
    for sub in ("static", "static_esc", "controls_512", "controls_${RUNG}"):
        assert f'f22s_record_si_prefix "$SERVE_DIR/{sub}" || {{ RC=1; exit 1; }}' in src, sub


def test_the_sbatch_scripts_stand_each_phase_up_at_the_views_per_leg_window():
    sampled = (F22 / "sampled" / "common_sampled.sh").read_text()
    # the view gate asks the generator's ONE scope list (main's f22s_view_in_scope), which names shortconst for llama8b
    assert 'f22s_view_in_scope "$key" "$view"' in sampled
    if str(F22 / "sampled") not in sys.path:
        sys.path.insert(0, str(F22 / "sampled"))
    import make_sampled_configs as msc  # noqa: WPS433  (the generator's ONE scope list)
    assert msc.in_scope("llama8b", "shortconst") and msc.in_scope("llama8b", "fullconst")
    # every roster_view.py call names the lane's roster (`--roster "$F22_ROSTER"`, 2026-09-18): the reader defaults to the
    # PRODUCTION roster and never reads F22_ROSTER itself, so a lane-private roster was invisible to the whole sampled leg
    for var, field in (("VIEW_WINDOW_STATIC", "window static"), ("VIEW_WINDOW_CONTROLS", "window controls"),
                       ("VIEW_RUNG_WINDOW_2048", "rung_window 2048"), ("VIEW_RUNG_WINDOW_8192", "rung_window 8192"), ("VIEW_RUNG_WINDOW_16384", "rung_window 16384")):
        # `--roster "$F22_ROSTER"` on every roster_view.py call (ctl-neg1 lane): roster_view.py defaults to the PRODUCTION roster
        # and does not read F22_ROSTER itself, so a lane roster reaches the sampled leg only through this suffix
        assert f'{var}=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" {field} --roster "$F22_ROSTER")' in sampled, var
    # the union with the sampled lane's long-SI mechanism: a roster-declared VIEW window wins, else the long-SI table
    # under F22S_LONG_SI, else the defaults — ONE resolution order in common_sampled.sh
    assert "f22s_base_window(){" in sampled
    assert 'if [ -n "${VIEW_WINDOW_STATIC:-${VIEW_WINDOW:-}}" ]; then echo "${VIEW_WINDOW_STATIC:-$VIEW_WINDOW}"' in sampled
    assert 'elif [ -n "${F22S_LONG_SI:-}" ]; then echo 10752' in sampled and 'else echo ""; fi;;' in sampled
    assert 'if [ -n "${VIEW_WINDOW_CONTROLS:-${VIEW_WINDOW:-}}" ]; then echo "${VIEW_WINDOW_CONTROLS:-$VIEW_WINDOW}"' in sampled
    assert 'elif [ -n "${F22S_LONG_SI:-}" ]; then echo 6656' in sampled and 'else echo 2048; fi;;' in sampled
    assert sampled.index('if [ -n "$vr" ] && [ -z "${VIEW_WINDOW:-}" ]; then echo "$vr"; return 0; fi') < sampled.index('if [ -n "${F22S_LONG_SI:-}" ]; then                                           # a LONG-SI cell on a table view')
    # the per-leg rung window wins for a per-leg view; the one-window view keeps its branch (16384 -> max(L, 51200)); a table view its model table
    for view_window, rung_var, rung, want in (("", "VIEW_RUNG_WINDOW_8192=14592", 8192, "14592"), ("", "VIEW_RUNG_WINDOW_2048=8064", 2048, "8064"),
                                              ("45056", "", 16384, "51200"), ("", "", 8192, "10240")):
        got = subprocess.run(["bash", "-c", f'f22_fatal(){{ echo FATAL "$@" >&2; exit 1; }}; KEY=llama8b; VIEW_WINDOW={view_window}; {rung_var + "; " if rung_var else ""}'
                              f'eval "$(sed -n "/^f22s_rung_window()/,/^}}/p" {F22 / "sampled" / "common_sampled.sh"})"; f22s_rung_window {rung}'],
                             capture_output=True, text=True)
        assert got.stdout.strip() == want, (view_window, rung_var, rung, got.stdout, got.stderr)
    src = (F22 / "sampled" / "target_sampled.sbatch").read_text()
    assert '[ -z "$VIEW_WINDOW" ] || export F22_LOAD_CHECK_CAP="${F22_LOAD_CHECK_CAP:-8192}"' in src   # the 8,192 probe cap is the ONE-WINDOW view's
    assert 'f22_window_check "$SERVE_DIR/static_esc" "$(f22s_rung_window 8192)" static 8192' in src
    assert 'f22_window_check "$SERVE_DIR/static_esc16" "$WINDOW16" static 16384' in src
    assert 'f22_window_check "$SERVE_DIR/controls_512" "$(f22s_base_window controls)" controls' in src
    assert 'f22_window_check "$SERVE_DIR/controls_${RUNG}" "$WINDOW" controls "$RUNG"' in src
    st = (F22 / "target_static.sbatch").read_text()
    assert '[ -z "$VIEW_WINDOW" ] || export F22_LOAD_CHECK_CAP="${F22_LOAD_CHECK_CAP:-8192}"' in st
    assert 'f22_window_check "$SERVE_DIR/static" "${VIEW_WINDOW_STATIC:-4864}" static,bon "$( [ -n "$VIEW_WINDOW" ] || echo base )"' in st
    # the wave's controls standup composes its window as CTL_L: the view's own controls window (this per-leg view's 6,656) first,
    # else the roster's gsm8k window for a gsm8k-bearing table view, else the table's 2,048
    assert 'CTL_L="${VIEW_WINDOW_CONTROLS:-${VIEW_WINDOW_GSM8K:-2048}}"' in st
    assert 'f22_standup attacks "$SERVE_DIR/controls" "" capability:short "$CTL_L" "" "$CTL_TOP_LOGPROBS"' in st
    # the multi-turn and Petri legs of a prompted-only view probe the PROMPTED route with the prefix (the load check)
    for name in ("target_attacks.sbatch", "petri_proxy.sbatch"):
        assert 'if f22_view_prompted_only; then export F22_LOAD_CHECK_SYSTEM_SI="${F22_LOAD_CHECK_SYSTEM_SI:-$(f22_prompted_si)}"; fi' in (F22 / name).read_text(), name
    pp = (F22 / "petri_proxy.sbatch").read_text()
    assert "llama8b_shortconst) PROXY_PORT=8989;;" in pp and "llama8b_shortconst) PPORT=8989;;" in (F22 / "petri_drive_full22.sh").read_text()
    # the window check's per-phase arguments (common.sh f22_window_check: legs + an optional cap, `base` = each leg's base cap only)
    common = (F22 / "common.sh").read_text()
    assert 'local out="$1" L="$2" legs="${3:-}" cap="${4:-}" si view_legs' in common and '${cap:+--cap "$cap"}' in common


def test_the_window_check_reads_the_configs_caps_and_admits_cap_base(tmp_path):
    cfg = yaml.safe_load((REPO / "configs/sdfbase_full22_llama8b_shortconst.yaml").read_text())
    assert pwc.leg_caps(cfg, ("static", "bon", "controls", "capability")) == {"static": (4096, 8192), "bon": (4096,), "controls": (512,), "capability": (1,)}
    assert pwc.leg_caps(cfg, ("static", "bon"), cap_override="base") == {"static": (4096,), "bon": (4096,)}
    assert pwc.leg_caps(cfg, ("controls",), cap_override=2048) == {"controls": (2048,)}
    assert pwc.MMLU_CAP == 1 and pwc.STATIC_ESC_CAP == 8192 and pwc.LEGS == ("static", "bon", "capability", "controls")
    # the view's controls list resolves through the roster from the config's private_run
    assert pwc._view_controls(cfg) == ("mmlu", "xstest", "or_bench_hard_1k")
    assert pwc._view_controls(yaml.safe_load((REPO / "configs/sdfbase_full22_llama8b_fullconst.yaml").read_text())) == ("mmlu",)
    # the check over synthetic renders: a per-leg base standup at cap base passes where the default (base + 8,192) would refuse

    class Tok:
        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
            n = sum(len(m["content"].split()) for m in messages) + 10
            return list(range(n))
    system = " ".join(["w"] * 5000)
    static = {"grok420": (" ".join(["p"] * 400), "g1")}
    caps_base = pwc.leg_caps(cfg, ("static",), cap_override="base")
    rec = pwc.check(tok=Tok(), system=system, max_model_len=9600, legs=("static",), static_prompts=static, bon_goals=[], cap_prompts={}, caps=caps_base)
    assert rec["ok"] and rec["caps"] == {"static": [4096]} and rec["rows"][0]["need"] == 5410 + 4096
    rec2 = pwc.check(tok=Tok(), system=system, max_model_len=9600, legs=("static",), static_prompts=static, bon_goals=[], cap_prompts={},
                     caps=pwc.leg_caps(cfg, ("static",)))
    assert not rec2["ok"] and any("@ cap 8192" in p for p in rec2["problems"])
    rec3 = pwc.check(tok=Tok(), system=system, max_model_len=6656, legs=("controls", "capability"), static_prompts={}, bon_goals=[],
                     cap_prompts={"mmlu": " ".join(["m"] * 700)}, ctl_prompts={"xstest": " ".join(["x"] * 30)}, caps=pwc.leg_caps(cfg, ("controls", "capability")))
    assert rec3["ok"] and {r["leg"] for r in rec3["rows"]} == {"controls", "capability"} and rec3["schema"] == "window_check/2"


def test_the_greedy_printer_prints_the_omissions_and_the_per_leg_windows(capsys, tmp_path, monkeypatch):
    """The launcher's plan check reads the PRINTED lines: fullconst's three omissions (the regression the shortconst WINDOWS line
    introduced and the plan check caught, 2026-09-12) and shortconst's per-leg WINDOWS line, both present in one offline render."""
    import submit_newmodels as sn
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cluster_site import write_site
    monkeypatch.setenv("SISTEER_CONFIG", str(write_site(tmp_path)))      # each cluster's host and QoS come from the site config
    rc = sn.main(["a" * 40, "--targets", "llama8b/fullconst,llama8b/shortconst", "--cluster", "llama8b/fullconst=cluster_b",
                  "--cluster", "llama8b/shortconst=cluster_b", "--petri-cluster", "cluster_b", "--offline", "--grading-host-worktree", "/tmp/x"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "# LEGS llama8b/fullconst: static, bon, capability; window 45056 for every standup" in out
    for leg, why in (("controls", "cheap legs only"), ("attacks", "no multi-turn"), ("petri", "the Petri window is certified")):
        assert f"#   omitted {leg}: {why}" in out, leg
    assert "# LEGS llama8b/shortconst" not in out     # every leg: no LEGS / omission block
    assert ("# WINDOWS llama8b/shortconst PER LEG (roster views.shortconst.serving.windows): static + BoN 10496 x 45; controls 6656 x 71; "
            "ladder rungs 2048 -> 8064 x 58, 8192 -> 14592 x 32, 16384 -> 22784 x 20; Petri 16384 x 29") in out
    assert "#   multi-turn pair: cap 4096 -> window PINNED 18688 x 25" in out and "f22-att-llama8b-shortconst-all" in out
