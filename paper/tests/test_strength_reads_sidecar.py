"""The committed strength-reads projection (`paper/notebooks/strength_reads_sidecar.py`, `paper/notebooks/data/strength_reads/`).

Pins:
  * the committed files load through the notebook's own loader (`judge_rules.load_strength_reads`), vouched by their manifests, with the
    paper policy's XSTest judge (`paper_judges.json`) as the MiniMax label's judge, and carry no machine-local path or host identity;
  * on a synthetic full sidecar: `write` keeps the record's per_item reads and the declared columns only, makes the battery records
    repo-relative, and loads to the same per-run reads as the full table; `check` is clean on what `write` wrote and names a tampered
    table or manifest; `write` refuses a sidecar whose manifest does not vouch for its table and a projection that would carry a path;
  * when the machine-local full sidecars are present (the grading host, under the directory STRENGTH_READS_FULL_ROOT names), `check`
    against them is clean (skipped elsewhere).
"""
import hashlib
import json
import os
import pathlib
import sys

import pandas as pd
import pyarrow.parquet as pq
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "paper" / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
import judge_rules as JR  # noqa: E402
import strength_reads_sidecar as SRS  # noqa: E402

COMMITTED = sorted((NOTEBOOKS / "data" / "strength_reads").glob("*/strength_reads.parquet"))
#: the full sidecars the committed projection was written from, relative to the directory STRENGTH_READS_FULL_ROOT names on the grading
#: host; they exist there only, so the `check` round trip skips when the variable is unset or a sidecar is absent
FULL_ROOT_ENV = "STRENGTH_READS_FULL_ROOT"
FULL = {"supplement_cpi": "supplement/cpi/sidecar/strength_reads.parquet",
        "supplement2_xstest": "supplement2/xstest/sidecar/strength_reads.parquet"}
SONNET = "3b" * 32
MINIMAX = "5e" * 32


def _paper_xstest_judge():
    with open(NOTEBOOKS / "data" / "paper_judges.json", encoding="utf-8") as fh:
        return next(j["judge_id"] for j in json.load(fh)["judges"] if j["rubric_name"] == "xstest_cpi")


def test_the_committed_projection_loads_vouched_with_the_paper_judge_and_carries_no_path():
    assert [p.parent.name for p in COMMITTED] == sorted(FULL)
    out = JR.load_strength_reads([str(p) for p in COMMITTED])
    assert _paper_xstest_judge() in out and len(out) == 2
    runs = set()
    for p in COMMITTED:
        man = json.loads((p.parent / "manifest.json").read_text())
        assert man["table_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest() and man["filed_check"]["judge_id"] == _paper_xstest_judge()
        assert man["columns"] == list(SRS.COLUMNS) and all(k.startswith("strength/") for k in man["battery_records"])
        t = pd.read_parquet(p)
        assert list(t.columns) == list(SRS.COLUMNS) and set(t["read"]) == {"per_item"} and len(t) == man["rows"]
        assert man["runs"] == t["run_id"].nunique()
        runs |= set(t["run_id"])
        assert SRS.leaks(pq.read_table(p), man) == []
        text = (p.parent / "manifest.json").read_text()
        assert "/home/" not in text and "~/" not in text and "/tmp/" not in text
    assert set(out[_paper_xstest_judge()]) == runs


def _full_sidecar(d: pathlib.Path, records_root: str = "/grading/strength") -> pathlib.Path:
    """A synthetic full sidecar: two runs, a per_item and a shared_rung read per item, machine-local columns the projection drops."""
    rows = []
    for run in ("sdfbase:xstest:aaaa:1111", "sdfbase:xstest:bbbb:2222"):
        for k in range(3):
            for read in ("per_item", "shared_rung"):
                rows.append(dict(run_id=run, arm_kind="steered", item_id=f"xst{k:04d}", trial=0, read=read, rung=512 if read == "per_item" else 2048,
                                 top_rung=2048, read_is_top_rung=read != "per_item", at_cap=bool(k == 2), store=f"/grading/store/{run}",
                                 sonnet_label="full_compliance", sonnet_status="judged", minimax_outcome="verdict",
                                 minimax_label="full_refusal" if k == 1 else "full_compliance", ledger="/grading/ledger.jsonl"))
    d.mkdir(parents=True, exist_ok=True)
    p = d / "strength_reads.parquet"
    pd.DataFrame(rows).to_parquet(p)
    man = {"table": str(p), "table_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "rows": len(rows), "runs": 2, "preview": False,
           "paused_run_dirs": {}, "unfinished_run_dirs": {},
           "filed_check": {"judge_id": MINIMAX, "counts": {"agree": 6}, "disagreements": []},
           "reads_scope": {"path": "/grading/scope", "report": {"sonnet_judge": SONNET, "battery_record": {"path": "scripts/x.py", "sha256": "c" * 64},
                                                                 "records": {f"{records_root}/lora/battery_curves.json": "d" * 64}}}}
    (d / "manifest.json").write_text(json.dumps(man))
    return p


def test_write_projects_loads_the_same_and_check_names_a_tamper(tmp_path):
    full = _full_sidecar(tmp_path / "full")
    out = tmp_path / "out"
    assert SRS.main(["write", "--source", f"syn={full}", "--out", str(out)]) == 0
    t = pd.read_parquet(out / "syn" / "strength_reads.parquet")
    assert list(t.columns) == list(SRS.COLUMNS) and set(t["read"]) == {"per_item"} and len(t) == 6
    man = json.loads((out / "syn" / "manifest.json").read_text())
    assert man["battery_records"] == {"strength/lora/battery_curves.json": "d" * 64} and man["source"]["rows"] == 12
    a = JR.load_strength_reads([str(full)])
    b = JR.load_strength_reads([str(out / "syn" / "strength_reads.parquet")])
    assert set(a) == set(b) == {MINIMAX, SONNET}
    for j in a:
        for r in a[j]:
            assert a[j][r].sort_values("item_id").reset_index(drop=True).equals(b[j][r].sort_values("item_id").reset_index(drop=True))
    assert SRS.main(["check", "--source", f"syn={full}", "--out", str(out)]) == 0
    # a second write is byte-identical (the committed file does not depend on when or where it was written)
    before = (out / "syn" / "strength_reads.parquet").read_bytes()
    assert SRS.main(["write", "--source", f"syn={full}", "--out", str(out)]) == 0
    assert (out / "syn" / "strength_reads.parquet").read_bytes() == before
    # a tampered table, then a tampered manifest, are named
    t.loc[0, "minimax_label"] = "full_refusal"
    t.to_parquet(out / "syn" / "strength_reads.parquet", index=False)
    assert SRS.main(["check", "--source", f"syn={full}", "--out", str(out)]) == 1
    assert SRS.main(["write", "--source", f"syn={full}", "--out", str(out)]) == 0
    man["preview"] = True
    (out / "syn" / "manifest.json").write_text(json.dumps(man))
    assert SRS.main(["check", "--source", f"syn={full}", "--out", str(out)]) == 1


def test_write_refuses_an_unvouched_table_and_a_projection_that_would_carry_a_path(tmp_path, capsys):
    full = _full_sidecar(tmp_path / "full")
    man = json.loads((full.parent / "manifest.json").read_text())
    man["table_sha256"] = "0" * 64
    (full.parent / "manifest.json").write_text(json.dumps(man))
    assert SRS.main(["write", "--source", f"syn={full}", "--out", str(tmp_path / "out")]) == 2
    assert "is not its manifest's table_sha256" in capsys.readouterr().err
    bad = _full_sidecar(tmp_path / "bad")
    t = pd.read_parquet(bad)
    t["item_id"] = t["item_id"].where(t.index != 0, os.path.expanduser("~/leaked/item"))
    t.to_parquet(bad)
    man = json.loads((bad.parent / "manifest.json").read_text())
    man["table_sha256"] = hashlib.sha256(bad.read_bytes()).hexdigest()
    (bad.parent / "manifest.json").write_text(json.dumps(man))
    assert SRS.main(["write", "--source", f"syn={bad}", "--out", str(tmp_path / "out2")]) == 2
    assert "machine-local paths" in capsys.readouterr().err and not (tmp_path / "out2" / "syn" / "strength_reads.parquet").exists()
    assert SRS.main(["write", "--source", f"Bad-Name={full}", "--out", str(tmp_path / "out3")]) == 2


def test_check_against_the_full_sidecars_on_the_grading_host():
    root = os.environ.get(FULL_ROOT_ENV, "")
    if not root:
        pytest.skip(f"{FULL_ROOT_ENV} is unset: the full reads sidecars are machine-local to the grading host")
    paths = {k: os.path.join(os.path.expanduser(root), v) for k, v in FULL.items()}
    if not all(os.path.exists(p) for p in paths.values()):
        pytest.skip(f"the full reads sidecars are not under {FULL_ROOT_ENV}: they are machine-local to the grading host")
    assert SRS.main(["check", *[x for k, p in sorted(paths.items()) for x in ("--source", f"{k}={p}")]]) == 0
