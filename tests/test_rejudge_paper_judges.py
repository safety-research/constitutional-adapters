"""The judge rows the paper reads (paper/notebooks/data/paper_judges.json), rebuilt from scripts/rejudge/ and compared with the committed
paper frame's judge table, column by column: the judge id (a hash of the judge model, the rubric hash, the protocol hash and the
threshold rule) and every recorded field.

- StrongREJECT v1 / v2 and the OR-Bench / XSTest C/P/I rows (MiniMax-M3, batch_or_live) are rebuilt from the code's defaults alone: the
  batch protocol plus the live route the live engines send by default.
- The agentic row (MiniMax-M3), while the policy lists it, is rebuilt with the live route recorded in its own row: that run's routing
  was narrowed by hand to one provider, so the defaults name another judge id. The paper reports no agentic eval, and a policy without
  the row skips that test.
- The Petri row (GLM-5.3) is rebuilt from the six answer tools the Opus judge was sent (tests/fixtures/rejudge/petri_answer_tools.json,
  whose scored dimensions are the committed scenario bundle's dimension files) and the judge template of the pinned inspect-petri. The
  rubric hash reads that template, so where inspect-petri is not installed the row is rebuilt with the frame's rubric hash and only the
  rubric test is skipped.

The frame's text redacts one attribution in the XSTest threshold rule; the hashed text is the original (``terra.file_records``), so the id
is rebuilt from it and the comparison applies the frame's redaction. Offline: no transcripts, no registry, no API."""
from __future__ import annotations

import json
import pathlib
import re
import sys

import yaml

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rejudge.common import filing as F  # noqa: E402
from rejudge.petri import file_records as PFR  # noqa: E402
from rejudge.petri import judges as PJ  # noqa: E402
from rejudge.strongreject import file_records as SFR  # noqa: E402
from rejudge.strongreject import live as SRL  # noqa: E402
from rejudge.terra import agentic as A  # noqa: E402
from rejudge.terra import file_records as TFR  # noqa: E402
from rejudge.terra import live as TL  # noqa: E402
from sisteer.registry import snapshot as SN  # noqa: E402

PAPER_JUDGES = json.loads((ROOT / "paper" / "notebooks" / "data" / "paper_judges.json").read_text(encoding="utf-8"))
BY_KEY = {(j["rubric_name"], j["rubric_version"]): j for j in PAPER_JUDGES["judges"]}
#: the six answer tools of the Petri judge (tool sha256 -> tool), and the scenario bundle their scored dimensions come from
PETRI_TOOLS = json.loads((ROOT / "tests" / "fixtures" / "rejudge" / "petri_answer_tools.json").read_text(encoding="utf-8"))
PETRI_BUNDLE = ROOT / "data" / "petri" / "scenarios" / "8812ba083bb5473c"
#: the Petri judge's rubric hash: the inspect-petri 3.0.11 judge template and highlights text plus the six answer tools
PETRI_RUBRIC_SHA256 = "16ad19ee79e0c24975ff9c5fd8f2159944c124d1c219cbe8d1b1afb223150fa2"
#: (family, rubric name, judge model) of the rows the paper's figures read; the policy may also list the agentic row, which no figure
#: reads (the paper reports no agentic eval)
PAPER_ROWS = {("strongreject", "strongreject", "minimax-m3"), ("over_refusal", "or_bench_cpi", "minimax-m3"),
              ("over_refusal", "xstest_cpi", "minimax-m3"), ("petri", "petri_bloom_judge", "glm-5.3")}


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    sid, _ = SN.paper_frame_current(str(ROOT / "paper" / "snapshot"))
    return pd.read_parquet(ROOT / "paper" / "snapshot" / "snapshots" / str(sid) / "judge.parquet")


def _live(providers, routing):
    """The ``live`` argument of the filers' judge_spec: the route a live engine sends and the providers it may be served by."""
    return {"routing": routing, "providers": {k: {"slug": v["slug"], "quantization": v["quantization"]} for k, v in sorted(providers.items())},
            "with_batch": True}


def _redact(text: str) -> str:
    """The frame's redaction of a dated attribution and a decision-log reference in a judge row's text."""
    return re.sub(r" \(D\d+\)", "", re.sub(r"\(the user \d{4}-\d{2}-\d{2}\)", "(the authors' rule)", text))


def _row(frame, judge_id):
    rows = frame[frame.judge_id == judge_id]
    assert len(rows) == 1, f"judge {judge_id[:12]} is not in the paper frame"
    return rows.iloc[0]


def _same_row(payload, row):
    """Every field of the rebuilt payload equals the frame's: the columns as text (the frame's redaction applied to the code's), and
    ``extra`` key by key against ``extra_json``."""
    for k, v in payload.items():
        if k == "extra":
            assert v == json.loads(row["extra_json"]), "extra differs"
            continue
        assert k in row.index, f"the frame has no column {k}"
        got = v if not isinstance(v, (dict, list)) else json.dumps(v, sort_keys=True)
        assert (_redact(got) if isinstance(got, str) else got) == row[k], k


def _agentic_entry():
    rows = [j for j in PAPER_JUDGES["judges"] if j["family"] == "agentic"]
    if not rows:
        pytest.skip("the paper policy lists no agentic row")
    (entry,) = rows
    return entry


def test_paper_judges_lists_the_figures_rows_and_the_filers_pin_the_minimax_ones():
    rows = [j for j in PAPER_JUDGES["judges"] if j["family"] != "agentic"]
    assert sorted((j["family"], j["rubric_name"], j["judge_model"]) for j in rows) == sorted(
        [("strongreject", "strongreject", "minimax-m3")] * 2 + sorted(PAPER_ROWS - {("strongreject", "strongreject", "minimax-m3")}))
    assert {j["rubric_version"] for j in rows if j["rubric_name"] == "strongreject"} == {"v1", "v2"}
    assert {(j["rubric_name"], j["judge_model"]) for j in PAPER_JUDGES["judges"] if j["family"] == "agentic"} <= {
        (A.RUBRIC_NAME, "minimax-m3")}
    pinned = F.MINIMAX_M3_JUDGE_IDS
    assert pinned["strongreject_minimax_m3"] == {"sr": BY_KEY[("strongreject", "v1")]["judge_id"],
                                                 "sr2": BY_KEY[("strongreject", "v2")]["judge_id"]}
    assert pinned["cpi_minimaxm3"] == {"or_bench_cpi": BY_KEY[("or_bench_cpi", "v1")]["judge_id"],
                                       "xstest_cpi": BY_KEY[("xstest_cpi", "v1")]["judge_id"]}
    assert SFR.PINNED_JUDGE_IDS == pinned["strongreject_minimax_m3"] and TFR.PINNED_JUDGE_IDS == pinned["cpi_minimaxm3"]
    assert PAPER_JUDGES["petri_alias"] == {"glm-5.3": next(j["judge_id"] for j in PAPER_JUDGES["judges"] if j["family"] == "petri")}


@pytest.mark.parametrize("kind,version", [("sr", "v1"), ("sr2", "v2")])
def test_strongreject_rows_rebuild_from_the_defaults(frame, kind, version):
    payload = SFR.judge_spec(kind, judge="minimax-m3", live=_live(SRL.PROVIDERS, SRL.routing()))
    assert payload["judge_id"] == BY_KEY[("strongreject", version)]["judge_id"]
    _same_row(payload, _row(frame, payload["judge_id"]))


@pytest.mark.parametrize("rubric", ["or_bench_cpi", "xstest_cpi"])
def test_cpi_rows_rebuild_from_the_defaults(frame, rubric):
    providers, routing, _ = TL.routing_for(TL.expected_parameters({"judge": "minimax-m3", "family": "cpi"}))
    payload = TFR.judge_spec(rubric, judge="minimax-m3", live=_live(providers, routing))
    assert payload["judge_id"] == BY_KEY[(rubric, "v1")]["judge_id"]
    _same_row(payload, _row(frame, payload["judge_id"]))


def test_the_agentic_row_rebuilds_from_its_recorded_route(frame):
    entry = _agentic_entry()
    route = json.loads(_row(frame, entry["judge_id"])["extra_json"])["protocol"]["live_route"]
    payload = TFR.judge_spec(A.RUBRIC_NAME, judge="minimax-m3", live=_live(route["providers"], route["provider_routing"]))
    assert payload["judge_id"] == entry["judge_id"]
    _same_row(payload, _row(frame, payload["judge_id"]))


def _petri_entry_and_row(frame):
    entry = next(j for j in PAPER_JUDGES["judges"] if j["family"] == "petri")
    return entry, _row(frame, entry["judge_id"])


def _dimension_text(path: pathlib.Path) -> str:
    """A scored dimension's text as inspect-petri 3.0.11 builds it from a dimension file (``_parse_dimension_file`` and
    ``_alignment_answer_type``): ``- <front matter description>``, then a blank line and the body when the body is not empty."""
    text = path.read_text(encoding="utf-8")
    end = text.index("\n---\n", 4)
    meta, body = yaml.safe_load(text[4:end]), text[end + 5:].strip()
    return f"- {meta['description']}" + (f"\n\n{body.rstrip()}" if body else "")


def test_the_petri_answer_tools_are_the_frames_and_their_dimensions_the_bundles(frame):
    """The committed tools are the ones the frame's Petri row lists (by hash), and every scored dimension of each tool is the committed
    scenario bundle's dimension file for that behaviour, in the order inspect-petri reads them (sorted by file name)."""
    _, row = _petri_entry_and_row(frame)
    tools = PETRI_TOOLS["tools"]
    assert all(PFR.R.tool_sha(t) == sha for sha, t in tools.items())
    assert sorted(tools) == json.loads(row["extra_json"])["answer_tool_sha256"]
    assert sorted(PETRI_TOOLS["behaviours"].values()) == sorted(tools)
    for behaviour, sha in PETRI_TOOLS["behaviours"].items():
        files = sorted((PETRI_BUNDLE / behaviour / "scenarios" / "dimensions").glob("*.md"))
        props = tools[sha]["input_schema"]["properties"]
        assert list(props) == ["highlights", "summary", *(f.stem for f in files), "justification"], behaviour
        for f in files:
            assert props[f.stem] == {"type": "integer", "description": _dimension_text(f), "minimum": 1, "maximum": 10}, (behaviour, f.stem)


def test_the_petri_row_rebuilds_from_the_glm_profile(frame, monkeypatch):
    """Every field but the rubric hash, which reads the inspect-petri template: that hash is taken from the frame here and recomputed
    by the next test."""
    entry, row = _petri_entry_and_row(frame)
    monkeypatch.setattr(PFR, "rubric_sha256", lambda _tools: row["rubric_sha256"])
    payload = PFR.arm_judge_spec(PETRI_TOOLS["tools"], PJ.profile("glm-5.3"), [16000])
    assert payload["judge_id"] == entry["judge_id"]
    _same_row(payload, row)
    assert F.canonical_sha256(PFR.protocol(judge="glm-5.3")) == "b6ee14ea0a88884f17c50ff4ea0847a15a6cc48c9c1b3ed88a01321c367dd65c"
    assert PFR.judge_spec({}, judge="glm-5.3")["protocol_sha256"] == row["protocol_sha256"]


def test_the_petri_rubric_hash_and_judge_id_are_recomputed_from_the_answer_tools(frame):
    """The rubric hash from the committed tools and the installed inspect-petri's template (``file_records.rubric_material`` refuses
    any version but 3.0.11), and the judge id from it with nothing taken from the frame."""
    pytest.importorskip("inspect_petri")
    entry, row = _petri_entry_and_row(frame)
    assert PFR.rubric_sha256(PETRI_TOOLS["tools"]) == PETRI_RUBRIC_SHA256 == row["rubric_sha256"]
    payload = PFR.arm_judge_spec(PETRI_TOOLS["tools"], PJ.profile("glm-5.3"), [16000])
    assert payload["judge_id"] == entry["judge_id"]
    _same_row(payload, row)
