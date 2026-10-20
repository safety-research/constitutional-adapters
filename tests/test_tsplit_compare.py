"""_tsplit_compare (P4 proof verdict): twin-run env-parity mode + the kept
stored-reference mode, both enforcing by exit code (fail-fast).

The twin mode is the criterion revision after job 112390: same node, same
job, same code, same staged inputs, only the venv/lock differs — so BOTH
metrics and per-record generations must be equal between the twins, any
difference is an env effect, and differing records are reported verbatim
(top-severity evidence). The stored campaign cell demotes to a
diagnostic-only reference. All offline: pure-JSON fixtures under tmp_path.
"""

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import _tsplit_compare as tc  # noqa: E402


def _write_cell(root: Path, results: dict) -> Path:
    cell = root / tc.CELL.parent
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "results.json").write_text(json.dumps(results))
    return root


def _results(loose: float = 0.17, response: str = "steered says hi",
             records: bool = True) -> dict:
    res = {
        "instruction": tc.CELL.parts[0],
        "method": "vector",
        "arms": [
            {"arm": "baseline", "scale": 0.0,
             "compliance_strict": 0.02, "compliance_loose": 0.03},
            {"arm": "steered", "scale": 2.0,
             "compliance_strict": 0.17, "compliance_loose": loose},
        ],
        "capability_ext": {"0.0": {"gsm8k": 0.89, "mmlu": 0.62}},
    }
    if records:
        res["records"] = [
            [{"prompt": "base prompt", "response": "base reply",
              "strict": "False", "loose": "False"}],
            [{"prompt": "steer prompt", "response": response,
              "strict": "True", "loose": "True"}],
        ]
    return res


def _util_log(tmp_path: Path, text: str = "0 %\n37 %\n") -> Path:
    p = tmp_path / "util.log"
    p.write_text(text)
    return p


def _main(argv: list[str], monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["_tsplit_compare.py", *argv])
    tc.main()


# ---------------------------------------------------------------- twin mode


def test_twin_equal_passes(tmp_path, monkeypatch, capsys):
    split = _write_cell(tmp_path / "split", _results())
    prod = _write_cell(tmp_path / "prod", _results())
    _main(["--twin-split", str(split), "--twin-prod", str(prod),
           "--util-log", str(_util_log(tmp_path))], monkeypatch)
    out = capsys.readouterr().out
    assert "twin metric equality: OK" in out
    assert "twin record equality: OK" in out
    assert "PASS: twins equal" in out


def test_twin_metric_mismatch_is_env_effect_and_fatal(
        tmp_path, monkeypatch, capsys):
    split = _write_cell(tmp_path / "split", _results(loose=0.18))
    prod = _write_cell(tmp_path / "prod", _results(loose=0.17))
    with pytest.raises(SystemExit) as exc:
        _main(["--twin-split", str(split), "--twin-prod", str(prod),
               "--util-log", str(_util_log(tmp_path))], monkeypatch)
    assert "ENV EFFECT" in str(exc.value.code)
    out = capsys.readouterr().out
    assert "TWIN METRIC MISMATCH" in out
    assert "split=0.18 != prod=0.17" in out


def test_twin_record_mismatch_reports_verbatim_and_dumps_json(
        tmp_path, monkeypatch, capsys):
    # Metrics identical; ONE generation differs textually — the top-severity
    # outcome: fatal, both responses printed in full, diff JSON written.
    split = _write_cell(tmp_path / "split",
                        _results(response="version A of the reply"))
    prod = _write_cell(tmp_path / "prod",
                       _results(response="version B of the reply"))
    with pytest.raises(SystemExit) as exc:
        _main(["--twin-split", str(split), "--twin-prod", str(prod),
               "--util-log", str(_util_log(tmp_path))], monkeypatch)
    assert "ENV EFFECT" in str(exc.value.code)
    assert "per-record generations differ" in str(exc.value.code)
    out = capsys.readouterr().out
    assert "TWIN RECORD MISMATCH" in out
    assert "arm steered@2.0, record 0" in out
    assert "version A of the reply" in out       # verbatim, both sides
    assert "version B of the reply" in out
    dump = split / tc.CELL.parent / tc.RECORD_DIFF_NAME
    entries = json.loads(dump.read_text())
    assert len(entries) == 1
    assert entries[0]["arm"] == "steered@2.0"
    assert entries[0]["split"]["response"] == "version A of the reply"
    assert entries[0]["prod"]["response"] == "version B of the reply"


def test_twin_missing_records_refuses_to_pass(tmp_path, monkeypatch):
    # Absent evidence is a failure, never a silent pass (authoritative-
    # source rule): the cell always writes records.
    split = _write_cell(tmp_path / "split", _results(records=False))
    prod = _write_cell(tmp_path / "prod", _results())
    with pytest.raises(SystemExit) as exc:
        _main(["--twin-split", str(split), "--twin-prod", str(prod),
               "--util-log", str(_util_log(tmp_path))], monkeypatch)
    assert "ENV EFFECT" in str(exc.value.code)


def test_twin_campaign_is_diagnostic_only(tmp_path, monkeypatch, capsys):
    # Twins agree with each other but differ from the stored campaign cell
    # (the 112390 cross-node situation): the verdict is PASS; the campaign
    # delta is printed as context, never enforced.
    split = _write_cell(tmp_path / "split", _results(loose=0.18))
    prod = _write_cell(tmp_path / "prod", _results(loose=0.18))
    campaign = _write_cell(tmp_path / "campaign", _results(loose=0.17))
    _main(["--twin-split", str(split), "--twin-prod", str(prod),
           "--campaign", str(campaign),
           "--util-log", str(_util_log(tmp_path))], monkeypatch)
    out = capsys.readouterr().out
    assert "PASS: twins equal" in out
    assert "DIAGNOSTIC (not enforced)" in out
    assert "split-twin=0.18 != campaign=0.17" in out


def test_twin_mode_needs_both_twin_flags(tmp_path, monkeypatch, capsys):
    split = _write_cell(tmp_path / "split", _results())
    with pytest.raises(SystemExit):
        _main(["--twin-split", str(split),
               "--util-log", str(_util_log(tmp_path))], monkeypatch)
    assert "BOTH --twin-split and --twin-prod" in capsys.readouterr().err


# ------------------------------------------------------------ A/A mode


def test_aa_equal_passes(tmp_path, monkeypatch, capsys):
    a = _write_cell(tmp_path / "splita", _results())
    b = _write_cell(tmp_path / "splitb", _results())
    _main(["--aa-first", str(a), "--aa-second", str(b),
           "--util-log", str(_util_log(tmp_path))], monkeypatch)
    out = capsys.readouterr().out
    assert "A/A metric equality: OK" in out
    assert "A/A record equality: OK" in out
    assert "PASS: A/A twins equal" in out


def test_aa_metric_mismatch_is_nondeterminism_not_env_effect(
        tmp_path, monkeypatch, capsys):
    # Same env on both sides: a mismatch means the determinism knobs failed,
    # and the verdict must say so — never "ENV EFFECT".
    a = _write_cell(tmp_path / "splita", _results(loose=0.18))
    b = _write_cell(tmp_path / "splitb", _results(loose=0.17))
    with pytest.raises(SystemExit) as exc:
        _main(["--aa-first", str(a), "--aa-second", str(b),
               "--util-log", str(_util_log(tmp_path))], monkeypatch)
    assert "NONDETERMINISM" in str(exc.value.code)
    assert "ENV EFFECT" not in str(exc.value.code)
    assert "PROPOSE to the coordinator" in str(exc.value.code)
    out = capsys.readouterr().out
    assert "A/A METRIC MISMATCH" in out
    assert "splitA=0.18 != splitB=0.17" in out


def test_aa_record_mismatch_reports_verbatim_and_dumps_json(
        tmp_path, monkeypatch, capsys):
    a = _write_cell(tmp_path / "splita", _results(response="take one"))
    b = _write_cell(tmp_path / "splitb", _results(response="take two"))
    with pytest.raises(SystemExit) as exc:
        _main(["--aa-first", str(a), "--aa-second", str(b),
               "--util-log", str(_util_log(tmp_path))], monkeypatch)
    assert "NONDETERMINISM" in str(exc.value.code)
    out = capsys.readouterr().out
    assert "A/A RECORD MISMATCH" in out
    assert "take one" in out and "take two" in out   # verbatim, both sides
    dump = a / tc.CELL.parent / tc.AA_RECORD_DIFF_NAME
    entries = json.loads(dump.read_text())
    assert entries[0]["splitA"]["response"] == "take one"
    assert entries[0]["splitB"]["response"] == "take two"


def test_aa_mode_needs_both_flags(tmp_path, monkeypatch, capsys):
    a = _write_cell(tmp_path / "splita", _results())
    with pytest.raises(SystemExit):
        _main(["--aa-first", str(a),
               "--util-log", str(_util_log(tmp_path))], monkeypatch)
    assert "BOTH --aa-first and --aa-second" in capsys.readouterr().err


def test_aa_mode_cannot_combine_with_twin_or_stored(tmp_path, monkeypatch,
                                                    capsys):
    root = _write_cell(tmp_path / "x", _results())
    with pytest.raises(SystemExit):
        _main(["--aa-first", str(root), "--aa-second", str(root),
               "--twin-split", str(root), "--twin-prod", str(root),
               "--util-log", str(_util_log(tmp_path))], monkeypatch)
    assert "cannot combine" in capsys.readouterr().err


# ---------------------------------------------------- stored-reference mode


def test_stored_reference_equal_passes(tmp_path, monkeypatch, capsys):
    scratch = _write_cell(tmp_path / "scratch", _results())
    campaign = _write_cell(tmp_path / "campaign", _results())
    _main(["--scratch", str(scratch), "--campaign", str(campaign),
           "--util-log", str(_util_log(tmp_path))], monkeypatch)
    out = capsys.readouterr().out
    assert "metric equality: OK" in out
    assert "PASS: metrics equal" in out


def test_stored_reference_mismatch_fatal(tmp_path, monkeypatch, capsys):
    scratch = _write_cell(tmp_path / "scratch", _results(loose=0.18))
    campaign = _write_cell(tmp_path / "campaign", _results(loose=0.17))
    with pytest.raises(SystemExit) as exc:
        _main(["--scratch", str(scratch), "--campaign", str(campaign),
               "--util-log", str(_util_log(tmp_path))], monkeypatch)
    assert "scratch cell metrics != stored campaign metrics" \
        in str(exc.value.code)
    assert "METRIC MISMATCH" in capsys.readouterr().out


def test_stored_reference_records_are_diagnostic(tmp_path, monkeypatch,
                                                 capsys):
    # Identical metrics, differing records: stored-reference mode PASSES
    # (cross-time records are informative, not enforced) but says so.
    scratch = _write_cell(tmp_path / "scratch", _results(response="A"))
    campaign = _write_cell(tmp_path / "campaign", _results(response="B"))
    _main(["--scratch", str(scratch), "--campaign", str(campaign),
           "--util-log", str(_util_log(tmp_path))], monkeypatch)
    out = capsys.readouterr().out
    assert "raw records identical: False" in out
    assert "PASS: metrics equal" in out


# ------------------------------------------------------------ shared checks


def test_zero_gpu_util_fatal_in_twin_mode(tmp_path, monkeypatch):
    split = _write_cell(tmp_path / "split", _results())
    prod = _write_cell(tmp_path / "prod", _results())
    with pytest.raises(SystemExit) as exc:
        _main(["--twin-split", str(split), "--twin-prod", str(prod),
               "--util-log", str(_util_log(tmp_path, "0 %\n0 %\n"))],
              monkeypatch)
    assert "61380" in str(exc.value.code)


def test_mode_flags_are_exclusive(tmp_path, monkeypatch, capsys):
    root = _write_cell(tmp_path / "x", _results())
    with pytest.raises(SystemExit):
        _main(["--scratch", str(root), "--twin-split", str(root),
               "--twin-prod", str(root),
               "--util-log", str(_util_log(tmp_path))], monkeypatch)
    assert "cannot combine" in capsys.readouterr().err


# ------------------------------------------------------------ the proof's cell


def test_the_proof_runs_and_stages_the_cell_this_verdict_reads():
    """The GPU proof, its staged bundle and this verdict name ONE cell, and the
    cell's instruction is registered with a programmatic checker, so a
    submission cannot load the model and only then fail on an unknown id."""
    from sisteer.data.instructions import resolve_si

    si, arm = tc.CELL.parts[0], tc.CELL.parts[1]
    assert resolve_si(si).checker != "external"
    lines = (REPO / "scripts" / "sbatch" / "torch_split_proof.sbatch").read_text().splitlines()
    code = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    assert set(re.findall(r"--si (\S+)", code)) == {si}
    assert code.count(f"--suite ifeval --si {si} --arm {arm}") == 3   # splitA, splitB, prod
    for staged in ("pairs.jsonl", f"{arm}/vector.safetensors", f"{arm}/vector.json",
                   f"{arm}/manifest.json"):
        assert f'"$OUT_PROOF/{si}/{staged}"' in code, staged
    # no stored campaign cell exists for this instruction, and a missing
    # --campaign cell is FATAL at the very end of the job
    assert "--campaign" not in code
