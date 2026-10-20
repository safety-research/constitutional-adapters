"""misalign_ladder_merge: first-rung rows + rerun rows for the selected (truncated, non-loop) pairs."""
import importlib.util
import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "misalign_ladder_merge", HERE.parent / "scripts/sdfbase/full22/misalign/misalign_ladder_merge.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

PROSE = " ".join(f"w{i * 7919 % 1013}" for i in range(300))
LOOP = "ab" * 300


def _row(cell, item, s, resp, fr, harmful, beh="leaking"):
    return {"family": "agentic", "cell_id": cell, "virtual_model": f"petri__{cell}", "item_id": item,
            "sample_idx": s, "behavior": beh, "prompt_sha": "f" * 64, "response": resp, "finish_reason": fr,
            "usage": {"completion_tokens": 5}, "model_reported": f"petri__{cell}", "harmful": harmful,
            "classifier_harmful": harmful, "structural_gate": harmful, "judge_tag": "am_leak.v1"}


def _tree(root, cell, rows, cap, extra_meta=None):
    d = root / "agentic" / cell
    d.mkdir(parents=True)
    M.write_jsonl(d / "judged.jsonl", rows)
    M.write_jsonl(d / "responses.jsonl", rows)
    (d / "meta.json").write_text(json.dumps({"cell_id": cell, "params": {"max_tokens": cap}, **(extra_meta or {})}))


def _src(tmp_path):
    src = tmp_path / "src"
    rows = [_row("obj", "leak_a", 0, "done", "stop", 1),
            _row("obj", "leak_a", 1, PROSE, "length", 0),       # truncated deliberation -> selected
            _row("obj", "leak_b", 0, LOOP, "length", 0),        # truncated loop -> degenerate, stays
            _row("obj", "murd_a", 0, PROSE, "length", 0, beh="murder")]
    _tree(src, "obj", rows, 3000)
    _tree(src, "none", [_row("none", "leak_a", 0, "x", "stop", 0)], 3000)
    (src / "cells.json").write_text("[]")
    (src / "em8q" / "obj").mkdir(parents=True)
    (src / "em8q" / "obj" / "judged.jsonl").write_text('{"item_id": "q", "sample_idx": 0}\n')
    return src


def test_merge_takes_rerun_rows_for_selected_pairs_and_keeps_loops(tmp_path):
    src = _src(tmp_path)
    rerun = tmp_path / "rerun"
    _tree(rerun, "obj", [_row("obj", "leak_a", 1, PROSE + " acted", "stop", 1),
                         _row("obj", "murd_a", 0, PROSE, "length", 0, beh="murder")], 16384)
    out = tmp_path / "out"
    report = M.merge_root(src, rerun, out, min_zlib=M.RERUN_MIN_ZLIB_RATIO, allow_missing=False)
    merged = {M.key(r): r for r in M.read_jsonl(out / "agentic" / "obj" / "judged.jsonl")}
    assert len(merged) == 4
    assert merged[("leak_a", 1)]["ladder_rung"] == 16384 and merged[("leak_a", 1)]["harmful"] == 1
    assert merged[("leak_b", 0)]["ladder_rung"] == 3000 and merged[("leak_b", 0)]["response"] == LOOP
    assert merged[("leak_a", 0)]["ladder_rung"] == 3000
    assert merged[("murd_a", 0)]["ladder_rung"] == 16384 and merged[("murd_a", 0)]["finish_reason"] == "length"
    c = report["agentic/obj"]
    assert c["leaking"] == {"n": 3, "at_cap_src": 2, "degenerate": 1, "selected": 1, "rerun_present": 1,
                            "at_cap_after_all": 1}
    assert c["murder"] == {"n": 1, "at_cap_src": 1, "selected": 1, "rerun_present": 1, "at_cap_after": 1,
                           "at_cap_after_all": 1}
    # untouched cell and the non-laddered family are copied verbatim; root files too
    assert M.read_jsonl(out / "agentic" / "none" / "judged.jsonl")[0]["ladder_rung"] == 3000
    assert (out / "em8q" / "obj" / "judged.jsonl").read_text() == '{"item_id": "q", "sample_idx": 0}\n'
    assert (out / "cells.json").exists()
    meta = json.loads((out / "agentic" / "obj" / "meta.json").read_text())
    assert meta["ladder"]["selected"] == 2 and meta["ladder"]["degenerate"] == 1 and meta["ladder"]["rerun_cap"] == 16384


def test_missing_rerun_row_refuses_unless_allowed(tmp_path):
    src = _src(tmp_path)
    rerun = tmp_path / "rerun"
    _tree(rerun, "obj", [_row("obj", "leak_a", 1, "acted", "stop", 1)], 16384)  # murd_a missing
    with pytest.raises(SystemExit, match="no judged rerun row"):
        M.merge_root(src, rerun, tmp_path / "out", min_zlib=M.RERUN_MIN_ZLIB_RATIO, allow_missing=False)
    report = M.merge_root(src, rerun, tmp_path / "out2", min_zlib=M.RERUN_MIN_ZLIB_RATIO, allow_missing=True)
    assert report["agentic/obj"]["murder"]["unrerun"] == 1
    merged = {M.key(r): r for r in M.read_jsonl(tmp_path / "out2" / "agentic" / "obj" / "judged.jsonl")}
    assert merged[("murd_a", 0)]["ladder_rung"] == 3000


def test_foreign_rerun_row_and_same_root_refuse(tmp_path):
    src = _src(tmp_path)
    rerun = tmp_path / "rerun"
    _tree(rerun, "obj", [_row("obj", "leak_a", 0, "again", "stop", 1)], 16384)  # leak_a s0 completed at src
    with pytest.raises(SystemExit, match="not a selected"):
        M.merge_root(src, rerun, tmp_path / "out", min_zlib=M.RERUN_MIN_ZLIB_RATIO, allow_missing=False)
    with pytest.raises(SystemExit, match="NEW root"):
        M.merge_root(src, rerun, src, min_zlib=M.RERUN_MIN_ZLIB_RATIO, allow_missing=False)


# --------------------------------------------------------------------------- refusals before anything is written (2026-10-19)
def _rerun(tmp_path):
    rerun = tmp_path / "rerun"
    _tree(rerun, "obj", [_row("obj", "leak_a", 1, PROSE + " acted", "stop", 1),
                         _row("obj", "murd_a", 0, PROSE, "length", 0, beh="murder")], 16384)
    return rerun


def test_a_missing_or_empty_root_refuses_by_name_and_writes_nothing(tmp_path):
    src, rerun, out = _src(tmp_path), _rerun(tmp_path), tmp_path / "out"
    kw = dict(min_zlib=M.RERUN_MIN_ZLIB_RATIO, allow_missing=False)
    with pytest.raises(SystemExit, match=r"--src-root .*/absent: not a directory \(the first rung's judged root\)"):
        M.merge_root(tmp_path / "absent", rerun, out, **kw)
    with pytest.raises(SystemExit, match=r"--rerun-root .*/absent: not a directory \(the ladder rung's judged root\)"):
        M.merge_root(src, tmp_path / "absent", out, **kw)
    # a src root that exists but holds no laddered cell (empty, or only a non-laddered family) wrote `{}` as the report before
    empty = tmp_path / "empty"; empty.mkdir()
    with pytest.raises(SystemExit, match=r"--src-root .*/empty: no laddered family cell \(agentic/<cell>/\) to merge"):
        M.merge_root(empty, rerun, out, **kw)
    (empty / "em8q" / "obj").mkdir(parents=True)
    with pytest.raises(SystemExit, match="no laddered family cell"):
        M.merge_root(empty, rerun, out, **kw)
    # an empty rung refuses unless --allow-missing, which lets every src row stand
    rung = tmp_path / "rung"; rung.mkdir()
    with pytest.raises(SystemExit, match=r"--rerun-root .*/rung: no laddered family cell .*the rung is empty.*--allow-missing"):
        M.merge_root(src, rung, out, **kw)
    assert not out.exists()                                     # every refusal above came before the output tree
    report = M.merge_root(src, rung, out, min_zlib=M.RERUN_MIN_ZLIB_RATIO, allow_missing=True)
    assert report["agentic/obj"]["leaking"]["unrerun"] == 1 and report["agentic/obj"]["murder"]["unrerun"] == 1


def test_the_cli_refuses_without_a_traceback_and_the_usage_names_no_machine(tmp_path, capsys):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as e:
        M.main(["--src-root", str(tmp_path / "absent"), "--rerun-root", str(tmp_path), "--out-root", str(out)])
    assert isinstance(e.value, M.LadderMergeError) and "not a directory" in str(e.value.code) and not out.exists()
    usage = M.__doc__[M.__doc__.index("Usage"):]
    assert "python scripts/sdfbase/full22/misalign/misalign_ladder_merge.py" in usage and "$GRADE/misalign/" in usage
    assert "~" not in usage and ".venv" not in usage


def test_the_merge_rule_is_the_one_the_registry_names(tmp_path):
    """The registry names the merged roots' instrument by the tool's commit (em_misalign.LADDER_MERGE_TOOL); the tool's later edit added
    refusals before any write and a usage text, not a change to the merge. The tool at that commit and this one write byte-identical trees
    and reports from the same inputs. Skipped where the clone does not carry the commit (a squashed export)."""
    import hashlib
    import subprocess
    from sisteer.registry.adapters import em_misalign as EM         # importable: loading the tool put the checkout's src/ on sys.path
    tool = EM.LADDER_MERGE_TOOL
    r = subprocess.run(["git", "-C", str(HERE.parent), "cat-file", "blob", f"{tool['commit']}:{tool['path']}"], capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"commit {tool['commit'][:12]} is not in this clone (a squashed export): the identity's bytes cannot be read")
    old_bytes = r.stdout
    assert hashlib.sha256(old_bytes).hexdigest() == tool["sha256"]
    assert hashlib.sha1(b"blob %d\0" % len(old_bytes) + old_bytes).hexdigest() == tool["blob_sha1"]
    old_path = tmp_path / "old" / "scripts" / "sdfbase" / "full22" / "misalign" / "misalign_ladder_merge.py"
    old_path.parent.mkdir(parents=True)
    old_path.write_bytes(old_bytes)
    old_spec = importlib.util.spec_from_file_location("misalign_ladder_merge_identity", old_path)
    OLD = importlib.util.module_from_spec(old_spec)
    old_spec.loader.exec_module(OLD)
    for allow_missing, rerun_rows in ((False, None), (True, [_row("obj", "leak_a", 1, "acted", "stop", 1)])):
        case = tmp_path / f"case_{allow_missing}"
        src = _src(case)
        rerun = case / "rerun"
        _tree(rerun, "obj", rerun_rows or [_row("obj", "leak_a", 1, PROSE + " acted", "stop", 1),
                                          _row("obj", "murd_a", 0, PROSE, "length", 0, beh="murder")], 16384)
        reports = [mod.merge_root(src, rerun, case / name, min_zlib=M.RERUN_MIN_ZLIB_RATIO, allow_missing=allow_missing)
                   for mod, name in ((OLD, "out_old"), (M, "out_new"))]
        assert reports[0] == reports[1]
        files = [{p.relative_to(case / name): p.read_bytes() for p in sorted((case / name).rglob("*")) if p.is_file()}
                 for name in ("out_old", "out_new")]
        assert files[0] == files[1] and files[0]
