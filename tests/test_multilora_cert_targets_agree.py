"""native-psr-serving steps 4 and 7: the certification driver's ``Target`` table
and the sbatch's ``case "$KEY"`` rows are ONE table in two spellings — pinned
equal here for EVERY row, extra cells included — and the numerics-not-measured
rows say what the plan says (§1.6):

* the four STAGE-1 PSR rows: the roster main-view all-positions PSR cell beside
  the target's LoRA twin, ``numerics="not_measured"``, llama8b on the
  certification-only ``psrcert`` view.
* the STAGE-2 row ``gemma4e4b_psr2cert``: the combined ``psr2cert`` view's
  three canaries — the response-fit PSR cell, the ARDITI ``static_r`` object and
  the LoRA twin — with the Arditi pin named through a QUALIFIED key, because it
  belongs to the misalignment authority and not to the wave's pins file. Both
  spellings resolve that key the same way (:func:`resolve_pin_key` and the
  sbatch's ``pin_sha``), which is pinned here against the COMMITTED pin files.

Every pin present in its committed file, every config/manifest present on the
tree. Synthetic data only; the sbatch fragment is RUN under bash (never
grepped)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import multilora_admission as adm  # noqa: E402
import multilora_cert as mc  # noqa: E402

from sisteer.capeval import manifest as cm  # noqa: E402

SBATCH = REPO / "scripts/sdfbase/multilora_cert.sbatch"
PSR_CELL = "constv2-mix-full__psr_ce_all"
LORA_TWIN = "constv2-mix-full__lora_r4_ce"
SNAPS = {"gemma4e4b": "models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2",
         "qwen35_4b": "models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
         "nemotron3nano4b": "models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
         "llama8b": "models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"}


def _sbatch_row(key):
    """Run the sbatch's OWN row block (``EXTRA_CELLS=""`` through ``esac``) under bash for
    ``key`` -> ``(rc, [MODEL_KEY, VIEW, CELL, PINKEY, EXTRA_CELLS, EXTRA_PINKEYS, snapshot])``."""
    src = SBATCH.read_text()
    start = src.index('EXTRA_CELLS=""; EXTRA_PINKEYS=""\ncase "$KEY" in')
    end = src.index("\nesac\n", start) + len("\nesac\n")
    frag = src[start:end]
    script = (f'KEY={key}; HF_HOME=/hf; log(){{ echo "LOG: $*"; }}; FINAL_RC=""\n{frag}\n'
              'echo "$MODEL_KEY|$VIEW|$CELL|$PINKEY|$EXTRA_CELLS|$EXTRA_PINKEYS|${SNAP#/hf/hub/}"')
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1].split("|")


def test_every_sbatch_row_equals_its_target_extra_cells_included():
    """One table, two spellings: MODEL_KEY / VIEW / CELL / PINKEY / EXTRA_CELLS /
    EXTRA_PINKEYS and the pinned snapshot, for EVERY row of ``TARGETS`` — a row
    added to one side without the other fails here."""
    for key, t in mc.TARGETS.items():
        rc, got = _sbatch_row(key)
        assert rc == 0, (key, got)
        want = [t.model_key, t.view, t.tt_cell, t.pin_key, " ".join(t.extra_cells), " ".join(t.extra_pin_keys),
                SNAPS[t.model_key]]
        assert got == want, (key, got, want)
    rc, got = _sbatch_row("bogus")
    assert rc == 2
    # the KEY list the sbatch REQUIRES names every row (and nothing else)
    src = SBATCH.read_text()
    listed = src.split('KEY="${1:?REQUIRED: ', 1)[1].split('}"', 1)[0].split(" | ")
    assert sorted(listed) == sorted(mc.TARGETS), (listed, sorted(mc.TARGETS))


def test_the_four_psr_rows_are_the_plans_stage_one_rows():
    assert mc.PSR_TARGET_KEYS == ("gemma4e4b_psr", "qwen35_4b_psr", "nemotron3nano4b_psr",
                                  "llama8b_psr", "gemma4e4b_psr2cert",
                                  "qwen35_4b_paired",   # the paired write (relu-differenced-native-qwen) ...
                                  "llama8b_paired")     # ... and its second round: llama8b's fitted teacher
    pins = json.loads((REPO / mc.PINS_REL).read_text())
    for key in [k for k in mc.PSR_TARGET_KEYS if k.endswith("_psr")]:
        t = mc.TARGETS[key]
        assert t.model_key == key.removesuffix("_psr") and t.tt_cell == PSR_CELL
        assert t.extra_cells == (LORA_TWIN,) and t.cells == (PSR_CELL, LORA_TWIN)
        assert t.pin_key == "base__constv2-mix-full__psr_ce_all" and t.extra_pin_keys == ("base__constv2-mix-full__lora_r4_ce",)
        assert t.numerics == adm.NUMERICS_NOT_MEASURED and not t.measures_numerics
        assert t.view == ("psrcert" if key == "llama8b_psr" else "main")
        assert t.fused_mamba_kernels is (key == "nemotron3nano4b_psr")
        # every cell's pin is in the committed pins.json under the MODEL key (a 64-hex sha)
        for pin_key in t.pin_keys:
            sha = pins[t.model_key][pin_key]["object_sha256"]
            assert len(sha) == 64 and int(sha, 16) >= 0, (key, pin_key)
        # config + manifest exist and list the cells (llama8b: the generated psrcert pair)
        cfg_path, man_path = REPO / t.config, REPO / t.manifest
        assert cfg_path.is_file() and man_path.is_file(), (key, t.config, t.manifest)
        cfg = yaml.safe_load(cfg_path.read_text())
        assert set(t.cells) <= set(cfg["explicit_cells"]), (key, cfg["explicit_cells"])
        if key == "llama8b_psr":
            assert cfg["explicit_cells"] == ["none", PSR_CELL, LORA_TWIN]
            assert "psrcert" in t.config and "psrcert" in t.manifest
    # the LoRA rows are untouched: single cell, numerics measured
    for key in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b", "llama8b_subtraction"):
        t = mc.TARGETS[key]
        assert t.extra_cells == () and t.numerics == adm.NUMERICS_DIAGNOSTIC and t.measures_numerics


def test_the_stage_two_row_is_the_combined_psr2cert_view_with_the_arditi_pin_through_its_own_file():
    """C5: three canaries, one view — and the Arditi one is pinned by the
    MISALIGNMENT authority, named through a qualified pin key so the certification binds
    the same object the roster's cross-bank mapping form selects for the view."""
    t = mc.TARGETS["gemma4e4b_psr2cert"]
    assert t.model_key == "gemma4e4b" and t.view == "psr2cert"
    assert t.tt_cell == "safety_refusal__psr_kl_all" and t.pin_key == "si12__safety_refusal__psr_kl_all"
    # T5b: the constitution response-fit psr — the lens §7.2 rows' own object — certified beside the safety cell
    assert t.extra_cells == ("si_constitution__psr_kl_all", "arditi", LORA_TWIN)
    assert t.cells == ("safety_refusal__psr_kl_all", "si_constitution__psr_kl_all", "arditi", LORA_TWIN)
    assert t.numerics == adm.NUMERICS_NOT_MEASURED and not t.measures_numerics
    assert t.config == "configs/sdfbase_full22_gemma4e4b_psr2cert.yaml"
    assert t.manifest == "configs/capeval/manifests/petri_full22_gemma4e4b_psr2cert.yaml"
    cfg = yaml.safe_load((REPO / t.config).read_text())
    assert cfg["explicit_cells"] == ["none", "safety_refusal__psr_kl_all", "si_constitution__psr_kl_all", "arditi", LORA_TWIN]
    assert cfg["defenses_dir"] == "out/full22/gemma4e4b/cells_view/psr2cert"
    # the harness id of `arditi` is the BARE class-B id
    assert mc.harness_cell_id("arditi") == "arditi"
    spec = cm.load_petri_spec(REPO / t.manifest)
    assert list(spec.cells) == ["none", "si_safety__psr_kl_all", "si_constitution__psr_kl_all", "arditi",
                                "si_constv2_mix_full__lora_r4_ce"]
    # the PIN KEYS: two bare (the wave's own file), one QUALIFIED (the misalignment file,
    # resolved BESIDE the row's pins file so both authorities travel together)
    wave_pins = REPO / mc.PINS_REL
    assert t.extra_pin_keys == ("si12__si_constitution__psr_kl_all", "misalign_baseline_pins.json:gemma4e4b.arditi",
                                "base__constv2-mix-full__lora_r4_ce")
    refs = {c: mc.resolve_pin_key(k, pins_path=wave_pins, model_key=t.model_key)
            for c, k in zip(t.cells, t.pin_keys)}
    assert not refs["safety_refusal__psr_kl_all"].qualified
    assert refs["safety_refusal__psr_kl_all"].path == wave_pins
    arditi = refs["arditi"]
    assert arditi.qualified and arditi.path == REPO / "configs/misalign_baseline_pins.json"
    assert arditi.model_key == "gemma4e4b" and arditi.key == "arditi"
    # the same digest the roster's mapping form selects for the view (ONE pin, two readers)
    mis = json.loads((REPO / "configs/misalign_baseline_pins.json").read_text())
    assert arditi.sha256 == mis["gemma4e4b"]["arditi"]["object_sha256"]
    roster = yaml.safe_load((REPO / "configs/sdfbase_full22_roster_psrcert.yaml").read_text())
    cell = roster["targets"]["gemma4e4b"]["views"]["psr2cert"]["cells"]["arditi"]
    assert cell == {"pins": "configs/misalign_baseline_pins.json", "pin": "gemma4e4b.arditi"}


@pytest.mark.parametrize("bad,needle", [
    ("nope.json:gemma4e4b.arditi", "no pins file at"),
    ("misalign_baseline_pins.json:gemma4e4b", "is not `<pins file>"),
    ("misalign_baseline_pins.json:gemma4e4b.ar_r0p1", "has no gemma4e4b/ar_r0p1"),
    ("no_such_key", "has no gemma4e4b/no_such_key"),
])
def test_a_pin_key_that_does_not_resolve_refuses_by_name(bad, needle):
    with pytest.raises(mc.CertRefusal, match=needle):
        mc.resolve_pin_key(bad, pins_path=REPO / mc.PINS_REL, model_key="gemma4e4b")


def test_the_sbatchs_pin_resolution_agrees_with_the_drivers_on_every_row():
    """ONE rule in two spellings: the sbatch's ``pin_sha`` (which runs BEFORE the venv, so
    a drifted object is caught in seconds) resolves a bare and a qualified key exactly as
    ``resolve_pin_key`` does — run here as bash, against the committed pin files."""
    src = SBATCH.read_text()
    start = src.index("pin_sha(){")
    frag = src[start:src.index('"$1"; }', start) + len('"$1"; }')]
    for key, t in mc.TARGETS.items():
        for cell, pin_key in zip(t.cells, t.pin_keys):
            want = mc.resolve_pin_key(pin_key, pins_path=REPO / mc.PINS_REL,
                                      model_key=t.model_key).sha256
            script = (f'PINS={REPO / mc.PINS_REL}; MODEL_KEY={t.model_key}\n{frag}\n'
                      f'pin_sha "{pin_key}"')
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            assert r.returncode == 0, (key, cell, r.stderr)
            assert r.stdout.strip() == want, (key, cell)


def test_target_refuses_a_malformed_row():
    with pytest.raises(ValueError, match="extra_pin_keys"):
        mc.Target("x", "c", "m", "v", "cell", "pin", extra_cells=("a",))
    with pytest.raises(ValueError, match="named twice"):
        mc.Target("x", "c", "m", "v", "cell", "pin", extra_cells=("cell",), extra_pin_keys=("p",))
    with pytest.raises(ValueError, match="numerics"):
        mc.Target("x", "c", "m", "v", "cell", "pin", numerics="sometimes")
    t = mc.Target("x", "c", "m", "v", "cell", "pin", extra_cells=("b",), extra_pin_keys=("pb",),
                  numerics=adm.NUMERICS_NOT_MEASURED)
    assert t.cells == ("cell", "b") and t.pin_keys == ("pin", "pb") and not t.measures_numerics


def test_preflight_cells_is_an_operational_option_and_the_parser_accepts_it():
    """--preflight-cells is classified (the fingerprint test refuses an unclassified
    option) as OPERATIONAL: it runs before any unit and changes nothing a unit measures."""
    assert "preflight_cells" in mc.OPERATIONAL_OPTIONS and "preflight_cells" not in mc.IDENTITY_OPTIONS
    args = mc.build_parser().parse_args(["--model-key", "gemma4e4b_psr", "--repo-dir", "/r", "--out", "/o",
                                         "--preflight-cells"])
    assert args.preflight_cells is True and args.model_key == "gemma4e4b_psr"
