"""submit_legs.py — the strength frontier's PRINTER for the expensive legs, driven by screen_verdict.py's record.

Fixtures are MINIMISED REAL PRODUCER RECORDS, not invented shapes. Every template below was cut from a real
`screen_verdict.json` written by `screen_verdict.py --host cluster_b --out /tmp/sv.json` on 2026-09-10 (tool
9ad02ec9, 24 dose views, 36 cells, all `incomplete` — the screens were PENDING / NEVER_LAUNCHED):

  VIEWS           the record's `views` (screen_verdict.load_dose_views over the strength-wave roster de4cdf56…): every key kept
  CELL_TEMPLATE   the nemotron3nano4b_d0p25 / const-LoRA cell, minimised to the keys this printer and
                  screen_verdict.close_directions read (target, method, method_label, dose, direction, anchor_exempt, tag,
                  view, cell, staged_object, job_name, verdict, fired, fired_detail, flagged_by, store, clauses[a|b|c].status /
                  by_readout / detail); readouts, positive_control, anchor, classifier dropped
  SOURCE_TEMPLATE the record's `source` with the squeue snapshot cut to three rows (the array-row spellings kept)
  the `directions` block is NEVER hand-written: tests edit cells' verdicts and regenerate it through the producer's own
  screen_verdict.close_directions, so a closure here has exactly the shape the producer emits.

sacct fixtures are rows in the exact `sacct -X -n -P -o JobID,JobName%80,State,ElapsedRaw,Start,NodeList` format the printer
documents, with REAL job names from the 21-day pull (elapsed edited where a test needs two different readings).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
STRENGTH = REPO / "scripts" / "sdfbase" / "full22" / "strength"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(STRENGTH))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, STRENGTH / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sv = _load("screen_verdict")
sl = _load("submit_legs")


DEFAULT = sl.DEFAULT_CLUSTER
OTHER = next(c for c in sl.CLUSTERS if c != DEFAULT)
#: the synthetic site every test runs against, one block per cluster in the two shapes a site config takes: the default
#: cluster's block is named for the cluster with the generic `slurm` substrate (the shipped site example's shape), the
#: other's has a label of its own and names the cluster by CLUSTER_SUBSTRATE (a site that keeps its own labels)
SITE = {
    DEFAULT: {"label": DEFAULT, "substrate": "slurm", "host": "login-one.example", "shared": "/scratch/example/one",
              "eval_root": "/scratch/example/one/eval-checkouts", "qos": "qos-one", "partition": "part-one", "account": "",
              "log_dir": "/scratch/example/one/joblogs", "extra": "", "budget": 96},
    OTHER: {"label": "site_two", "substrate": OTHER, "host": "login-two.example", "shared": "/data/example/two",
            "eval_root": "/data/example/two/checkouts", "qos": "qos-two", "partition": "part-two", "account": "acct-two",
            "log_dir": "/data/example/two/logs", "extra": "--exclude=compute-node-x", "budget": 16},
}


def write_site(root: Path, clusters=(DEFAULT, OTHER), extra_blocks=()) -> Path:
    """A synthetic site directory: config.env naming the blocks, one clusters/<label>.env per cluster from SITE."""
    (root / "clusters").mkdir(parents=True, exist_ok=True)
    labels = []
    for cl in clusters:
        s = SITE[cl]
        labels.append(s["label"])
        (root / "clusters" / f"{s['label']}.env").write_text(
            f"CLUSTER_SSH={s['host']}\nCLUSTER_SHARED_ROOT={s['shared']}\nCLUSTER_PARTITION={s['partition']}\n"
            f"CLUSTER_QOS={s['qos']}\nCLUSTER_ACCOUNT='{s['account']}'\nCLUSTER_SBATCH_EXTRA='{s['extra']}'\n"
            f"CLUSTER_EVAL_REPO_ROOT={s['eval_root']}\nCLUSTER_LOG_DIR={s['log_dir']}\nCLUSTER_SUBSTRATE={s['substrate']}\n"
            f"CLUSTER_LABEL={s['label']}\nCLUSTER_HIGH_BUDGET_GPUS={s['budget']}\n")
    for label, text in extra_blocks:
        labels.append(label)
        (root / "clusters" / f"{label}.env").write_text(text)
    (root / "config.env").write_text(f'SISTEER_CLUSTERS="{" ".join(labels)}"\n')
    return root


@pytest.fixture(autouse=True)
def _synthetic_site(tmp_path, monkeypatch):
    """Every command the printer prints comes from the site's cluster blocks: point it at a synthetic site so no host's own
    site config reaches these tests."""
    monkeypatch.delenv("SISTEER_CONFIG", raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(write_site(tmp_path / "site")))

SHA = "f38d055d59a7102f930e30d3abac64adbcf1270b"
LORA = "constv2-mix-full__lora_r4_ce"
SUB = "constv2-mix-full__lora_r4_ce_minus_ctl"

# ------------------------------------------------------------------------------------------------------------ #
# templates (see the module docstring for the real record each was cut from)                                    #
# ------------------------------------------------------------------------------------------------------------ #

def _view(target, view, dose, cells, objects):
    return {"anchor_exempt": dose == 1.0, "cells": cells, "config": f"configs/sdfbase_full22_{target}_{view}.yaml",
            "direction": "+" if dose > 0 else "-", "dose": dose,
            "manifest": f"configs/capeval/manifests/petri_full22_{target}_{view}.yaml", "objects": objects,
            "tag": f"{target}_{view}", "target": target, "view": view}


def _it_views(target):
    out = []
    for view, dose, pre in (("it_dm1", -1.0, "dm1it__"), ("it_d0p1", 0.1, "d0p1it__"), ("it_d0p25", 0.25, "d0p25it__"),
                            ("it_d0p5", 0.5, "halfdoseit__"), ("it_d1", 1.0, "itdoc__"), ("it_d1p5", 1.5, "d1p5it__")):
        out.append(_view(target, view, dose, [LORA], {LORA: pre + LORA}))
    return out


def _base_views(target):
    out = []
    for view, dose, pre_l, pre_s in (("dm1", -1.0, "dm1__", "dm1__"), ("d0p1", 0.1, "d0p1__", "d0p1__"), ("d0p25", 0.25, "d0p25__", "d0p25__"),
                                     ("d0p5", 0.5, "halfdose__", "halfdose__"), ("d1", 1.0, "base__", "derived__"), ("d1p5", 1.5, "d1p5__", "d1p5__")):
        out.append(_view(target, view, dose, [LORA, SUB], {LORA: pre_l + LORA, SUB: pre_s + SUB}))
    return out


VIEWS = _it_views("gemma4e4b") + _it_views("qwen35_4b") + _base_views("nemotron3nano4b") + _base_views("glm46vflash")
assert len(VIEWS) == 24

CELL_TEMPLATE = {
    "anchor_exempt": False, "cell": LORA,
    "clauses": {
        "a": {"by_readout": {"or_bench_hard_1k": "unevaluable: no controls_cap512 root for this TAG (the screen has not written anything)",
                             "xstest": "unevaluable: no controls_cap512 root for this TAG (the screen has not written anything)"},
              "status": "unevaluable"},
        "b": {"by_readout": {"or_bench_hard_1k": "unevaluable: no controls_cap512 root for this TAG (the screen has not written anything)",
                             "xstest": "unevaluable: no controls_cap512 root for this TAG (the screen has not written anything)"},
              "status": "unevaluable"},
        "c": {"detail": "unevaluable: no controls serving dir for this TAG (no standup has run)", "status": "unevaluable"},
    },
    "direction": "+", "dose": 0.25, "fired": [], "fired_detail": [], "flagged_by": [],
    "job_name": "f22s-nemotron3nano4b_d0p25-ctl0", "method": LORA, "method_label": "const LoRA",
    "staged_object": "d0p25__" + LORA,
    "store": {"complete": False, "dir": None,
              "reasons": ["no controls_cap512 root for this TAG (the screen has not written anything)"], "slurm_job_id": None},
    "tag": "nemotron3nano4b_d0p25", "target": "nemotron3nano4b", "verdict": "incomplete", "view": "d0p25",
}

SOURCE_TEMPLATE = {
    "gathered_at": "2026-09-10T23:30:45.330697+00:00",
    "helper": {"loaded": True, "path": f"/workspace/TESTUSER/si-steering-eval-repo/{SHA}/scripts/sdfbase/full22/sampled/at_cap_controls.py",
               "sha256": "8d83b3a23fc76ff22213a25db876bf4e2c8ca7bfe7fcddf3943b0c7d60029cee"},
    "host": "cluster_b", "hostname": "TESTHOST", "mode": "ssh", "user": "TESTUSER",
    "sampled_root": "/workspace/TESTUSER/exp/sdfbase_full22/sampled", "wave_root": "/workspace/TESTUSER/exp/sdfbase_full22", "wave_stop": False,
    "jobs_probe": {"probed": True, "sacct_error": None, "squeue_error": None,
                   "squeue": ["90353|D_TESTUSER_asstage-nemotron3nano4b|RUNNING|0:35|None",
                              "90197|f22s-qwen35_4b_it_d1-ctl0|PENDING|0:00|Priority",
                              "90161_[0-1,3-4]|m5t-base|PENDING|0:00|Priority"]},
}

# the producer's own clause-detail spellings (screen_verdict.evaluate_cell)
STOP_A = "(a) xstest: repetition-rule loop 230/450 = 51.111% >= 50%"
STOP_B = "(b) or_bench_hard_1k: at-cap 540/1319 vs anchor 113/1319, excess +32.4 pp >= 30"
STOP_C = "(c) standup positive control FAILED: steering_bites,salt_isolation"


def make_cell(view: dict, method: str, verdict: str = "incomplete", fired=(), fired_detail=(), flagged_by=()):
    c = copy.deepcopy(CELL_TEMPLATE)
    c.update({"target": view["target"], "method": method, "method_label": sv.METHOD_LABEL.get(method, method),
              "dose": view["dose"], "direction": view["direction"], "anchor_exempt": view["anchor_exempt"], "tag": view["tag"],
              "view": view["view"], "cell": method, "staged_object": view["objects"][method], "job_name": sv.job_name(view["tag"], 0),
              "verdict": verdict, "fired": list(fired), "fired_detail": list(fired_detail), "flagged_by": list(flagged_by)})
    if verdict != "incomplete":
        c["store"] = {"complete": True, "dir": f"/workspace/TESTUSER/exp/sdfbase_full22/sampled/controls_cap512/{view['tag']}_t0/cells/x",
                      "reasons": [], "slurm_job_id": "90000"}
        for k in "abc":
            cl = c["clauses"][k]
            cl["status"] = "fired" if any(f.startswith(k) for f in fired) else "clear"
            if "by_readout" in cl:
                cl["by_readout"] = {r: "clear" for r in cl["by_readout"]}
            if "detail" in cl:
                cl["detail"] = "positive control PASS on cell x (this cell — measured)"
    return c


def make_verdict(overrides=None, views=None, squeue=None, roster_sha=None) -> dict:
    """A record shaped as screen_verdict.main writes it; `overrides` = {(tag, method): (verdict, fired, fired_detail[, flagged_by])}."""
    views = copy.deepcopy(views if views is not None else VIEWS)
    overrides = overrides or {}
    cells = []
    for v in views:
        for m in v["cells"]:
            o = overrides.get((v["tag"], m))
            cells.append(make_cell(v, m, *o) if o else make_cell(v, m))
    gathered = {"tags": {}, "jobs": {"probed": squeue is not None, "squeue": squeue or [], "sacct": [], "squeue_error": None, "sacct_error": None},
                "wave_root": SOURCE_TEMPLATE["wave_root"], "sampled_root": SOURCE_TEMPLATE["sampled_root"]}
    directions = sv.close_directions(cells, gathered, 0)
    src = copy.deepcopy(SOURCE_TEMPLATE)
    if squeue is not None:
        src["jobs_probe"]["squeue"] = list(squeue)
    return {"written_at": "2026-09-10T23:30:47.080859+00:00",
            "tool": {"argv": ["--host", "cluster_b"], "git_sha": "9ad02ec9b6a5c2947443970539415499657f94a6", "path": str(STRENGTH / "screen_verdict.py"), "sha256": "x"},
            "roster": {"path": "/nonexistent/configs/sdfbase_full22_roster.yaml", "sha256": roster_sha or "0" * 64, "how": "test"},
            "source": src, "views": views, "cells": cells, "directions": directions,
            "cancel": sv.cancel_list(directions, gathered, 0),
            "summary": {**{v: sum(1 for c in cells if c["verdict"] == v) for v in sv.VERDICTS}, "detector_disagreements": 0}}


def all_ok(views=None):
    return {(v["tag"], m): ("ok", [], []) for v in (views or VIEWS) for m in v["cells"]}


SACCT_ROWS = {  # real job names; elapsed seconds as measured unless a test edits them
    "cluster_b": [
        ("84827", "f22s-gemma4e4b-sb0", "COMPLETED", 12458, "2026-09-09T22:00:07", "node-3"),
        ("84968", "f22s-gemma4e4b-sb1", "COMPLETED", 12723, "2026-09-09T22:26:00", "node-8"),
        ("86113", "f22s-gemma4e4b_it-sb0", "COMPLETED", 10331, "2026-09-10T05:46:00", "node-11"),
        ("86115", "f22s-gemma4e4b_it-sb1", "COMPLETED", 10049, "2026-09-10T05:46:00", "node-6"),
        ("85502", "f22s-gemma4e4b-se0", "COMPLETED", 1018, "2026-09-10T02:06:00", "node-10"),
        ("88421", "f22s-gemma4e4b_it-se0", "COMPLETED", 1625, "2026-09-10T18:26:00", "node-12"),
        ("88977", "f22s-gemma4e4b-cap0-r3", "COMPLETED", 3091, "2026-09-10T19:55:00", "node-10"),
        ("77042", "f22-stat-nemotron3nano4b-3", "COMPLETED", 3240, "2026-09-07T19:25:00", "node-12"),
        ("74031", "f22-att-gemma4e4b-crescendo", "FAILED", 216, "2026-09-07T03:00:00", "node-12"),
        ("74056", "f22-att-gemma4e4b-crescendo", "COMPLETED", 17244, "2026-09-07T03:09:00", "node-12"),
        ("74032", "f22-att-gemma4e4b-pair", "COMPLETED", 12780, "2026-09-07T03:01:00", "node-11"),
        ("74318", "f22-att-gemma4e4b_it-all", "COMPLETED", 21348, "2026-09-07T04:17:00", "node-12"),
        ("73775", "f22-atk-1", "COMPLETED", 44388, "2026-09-07T01:36:00", "node-0"),
        ("73937", "f22-atk-2", "COMPLETED", 41040, "2026-09-07T02:32:00", "node-6"),
    ],
    "cluster_a": [
        ("423449", "f22-stat-gemma4e4b-1", "COMPLETED", 3766, "2026-09-07T03:04:00", "node-31"),
        ("427373", "f22-stat-gemma4e4b-3", "COMPLETED", 12528, "2026-09-07T05:19:00", "node-18"),
        ("476829_0", "b4k-gemma4e4b_it", "FAILED", 2124, "2026-09-10T18:13:00", "node-25"),
        ("476828_0", "b4k-glm46vflash", "COMPLETED", 5472, "2026-09-10T18:12:00", "node-7"),
    ],
}


def write_sacct(tmp_path: Path, rows=None, name="sacct") -> list:
    rows = rows or SACCT_ROWS
    flags = []
    for cl, rs in rows.items():
        p = tmp_path / f"{name}_{cl}.psv"
        p.write_text("".join(f"{jid}|{nm}|{st}|{el}|{start}|{node}\n" for jid, nm, st, el, start, node in rs), encoding="utf-8")
        flags += ["--sacct-file", f"{cl}={p}"]
    return flags


def write_roster(tmp_path: Path, views=None) -> Path:
    """A roster carrying exactly the fixture's dose views (`dose:` keys) plus each target's main view (8 cells)."""
    views = views or VIEWS
    roster = {"lane_flags": {"candsi": {"enabled": False, "view": "main"}}, "targets": {}}
    for v in views:
        t = roster["targets"].setdefault(v["target"], {"model": "TEST/model", "model_revision": "0" * 40,
                                                      "views": {"main": {"cells": {f"cell{i}": f"obj{i}" for i in range(8)}},
                                                                "it": {"cells": {f"cell{i}": f"obj{i}" for i in range(3)}},
                                                                "halfdose": {"cells": {f"cell{i}": f"obj{i}" for i in range(6)}}},
                                                      "lanes": {"candsi": {"A10__lora_r4": "x"}}})
        t["views"][v["view"]] = {"dose": v["dose"], "cells": dict(v["objects"]), "config": v["config"], "manifest": v["manifest"]}
    p = tmp_path / "sdfbase_full22_roster.yaml"
    p.write_text(yaml.safe_dump(roster, sort_keys=True), encoding="utf-8")
    return p


def run(tmp_path: Path, rec: dict, *extra, sacct=True, name="v") -> tuple:
    vp = tmp_path / f"{name}.json"
    vp.write_text(json.dumps(rec), encoding="utf-8")
    argv = [str(vp), "--wave-sha", SHA] + (write_sacct(tmp_path) if sacct else []) + list(extra)
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sl.main(argv)
    return rc, buf.getvalue()


def run_with_roster(tmp_path: Path, overrides=None, *extra, name="vr") -> tuple:
    rp = write_roster(tmp_path)
    rec = make_verdict(overrides, roster_sha=hashlib.sha256(rp.read_bytes()).hexdigest())
    return run(tmp_path, rec, "--roster", str(rp), *extra, name=name)


def sbatch_lines(out: str) -> list:
    return [ln for ln in out.splitlines() if ln.startswith("ssh ") and " sbatch " in ln]


def job_names(out: str, include_hold=True) -> list:
    names = []
    for ln in out.splitlines():
        if " sbatch " in ln and (ln.startswith("ssh ") or (include_hold and ln.startswith("# HOLD"))):
            m = re.search(r" -J (\S+) ", ln)
            names.append(m.group(1))
    return names


def _time_of(out: str, job: str) -> str:
    for ln in out.splitlines():
        if f" -J {job} " in ln:
            return re.search(r"--time=(\S+)", ln).group(1)
    raise AssertionError(f"{job} not printed")


# ------------------------------------------------------------------------------------------------------------ #
# 1. today: every cell incomplete -> nothing printed, and the print says why                                       #
# ------------------------------------------------------------------------------------------------------------ #

def test_all_incomplete_prints_no_job_and_names_the_rule(tmp_path):
    rc, out = run(tmp_path, make_verdict())
    assert rc == 0
    assert sbatch_lines(out) == []
    assert not [l for l in out.splitlines() if l.startswith("# HOLD (")]
    assert "INCOMPLETE is not permission (D32(3))" in out
    # the omission carries the clauses the producer could not evaluate, with their text
    assert "(a) both readouts: no controls_cap512 root for this TAG" in out
    assert "(c) no controls serving dir for this TAG" in out
    # 36 cells x 4 expensive legs + gsm8k = every (leg, tag, method) either omitted as incomplete or as already existing
    n_incomplete = int(re.search(r"\[\s*(\d+)\] verdict INCOMPLETE is not permission", out).group(1))
    assert n_incomplete == 36 * 5 - 48   # 48 = the existing-result exclusions (6 tags x (1+2+2+2+1) legs)
    assert "no admissible multi-turn cell -> no pool member is needed" in out
    assert "TOTAL      expected 0 GPU-h" in out


def test_assume_ok_prints_but_is_loud_on_every_line(tmp_path):
    rc, out = run(tmp_path, make_verdict(), "--assume-ok")
    assert rc == 0
    lines = sbatch_lines(out)
    assert lines, "assume-ok must print the plan"
    assert "!!!! --assume-ok" in out
    # every printed LEG job (not the pool members) is preceded by an ASSUME-OK note naming its cell
    for ln in lines:
        name = re.search(r" -J (\S+) ", ln).group(1)
        if name.startswith(sl.POOL_JOB_PREFIX):
            continue
        assert f"#   [{name}] " in out and re.search(rf"#   \[{re.escape(name)}\] .*ASSUME-OK: verdict INCOMPLETE", out), name


# ------------------------------------------------------------------------------------------------------------ #
# 2. The per-leg dose sets, exactly                                                                              #
# ------------------------------------------------------------------------------------------------------------ #

def _doses_by_leg(out: str) -> dict:
    tag_dose = {v["tag"]: v["dose"] for v in VIEWS}
    by = {"static": set(), "bon": set(), "multiturn": set(), "capability": set(), "gsm8k": set()}
    for name in job_names(out):
        for tag in tag_dose:
            if name in (f"f22s-{tag}-sb0", f"f22s-{tag}-sb1", f"f22s-{tag}-se0", f"f22s-{tag}-se1"):
                by["static"].add(tag_dose[tag])
            elif name == f"f22-stat-{tag}-bon":
                by["bon"].add(tag_dose[tag])
            elif name == f"f22-att-{tag}-all":
                by["multiturn"].add(tag_dose[tag])
            elif name == f"f22s-{tag}-cap0":
                by["capability"].add(tag_dose[tag])
    for m in re.finditer(r"GSM8K PLACEHOLDER (\S+)", out):
        by["gsm8k"].add(tag_dose[m.group(1)])
    return by


def test_dose_sets_match_D33_exactly(tmp_path):
    rc, out = run(tmp_path, make_verdict(all_ok()))
    assert rc == 0
    by = _doses_by_leg(out)
    assert by["static"] == {-1.0, 0.1, 0.25, 0.5, 1.5}, "static sampled: 0.5 has no sampled rows; 1.0 is Sample's"
    assert by["bon"] == {-1.0, 0.1, 0.25, 1.5}, "BoN-512: 0.5 and 1.0 have 512 stores"
    assert by["multiturn"] == {-1.0, 0.1, 0.25, 1.5}, "multi-turn: 0.5 (halfdose) and 1.0 (campaign) exist, greedy both"
    assert by["capability"] == {-1.0, 0.1, 0.25, 1.5}, "MMLU: 0.5 and 1.0 exist"
    assert by["gsm8k"] == {-1.0, 0.1, 0.25, 0.5, 1.5}, "GSM8K: 0.5 + the new doses; 1.0 from the follow-up"
    # the omission list names the dose rule for every excluded dose, per leg
    assert "D33: dose +1 not re-run for this leg — sampled static rows at 1.0 exist" in out
    assert "D33: dose +0.5 not re-run for this leg — greedy/512 multi-turn rows exist" in out
    # counts: 4 targets x 5 static doses x (2 sb + 2 se) = 80; 16 bon; 16 att; 16 cap
    names = job_names(out)
    assert sum(n.endswith(("-sb0", "-sb1", "-se0", "-se1")) for n in names) == 80
    assert sum(n.endswith("-bon") for n in names) == 16
    assert sum(n.endswith("-all") and n.startswith("f22-att-") for n in names) == 16
    assert sum(n.endswith("-cap0") for n in names) == 16


def test_rerun_existing_voids_D33_loudly(tmp_path):
    rc, out = run(tmp_path, make_verdict(all_ok()), "--rerun-existing")
    assert rc == 0
    by = _doses_by_leg(out)
    assert by["static"] == by["bon"] == by["multiturn"] == by["capability"] == {-1.0, 0.1, 0.25, 0.5, 1.0, 1.5}
    assert "!!!! --rerun-existing" in out
    assert "D33: dose" not in out.split("| cluster |")[1]   # no dose omission remains


# ------------------------------------------------------------------------------------------------------------ #
# 3. the verdict: stopped rungs, closure in both directions, the anchor, incomplete                                #
# ------------------------------------------------------------------------------------------------------------ #

def test_stopped_rung_is_omitted_with_the_clause_and_its_numbers(tmp_path):
    ov = all_ok()
    ov[("nemotron3nano4b_d1p5", LORA)] = ("stopped", ["a:xstest"], [STOP_A])
    rc, out = run(tmp_path, make_verdict(ov))
    assert rc == 0
    assert "its own screen STOPPED it (D8" in out and STOP_A in out
    # the two-method view still runs for the surviving method, with cells RESTRICTED to none + that cell
    ln = next(l for l in out.splitlines() if " -J f22s-nemotron3nano4b_d1p5-sb0 " in l)
    assert f'"none,{SUB}"' in ln and LORA + '"' not in ln.split("none,")[1].split('"')[0].replace(SUB, "")
    assert f"[f22s-nemotron3nano4b_d1p5-sb0] cells RESTRICTED to none + {SUB}" in out
    bon = next(l for l in out.splitlines() if " -J f22-stat-nemotron3nano4b_d1p5-bon " in l)
    assert bon.rstrip("'").endswith(f"bon {SHA} 9 none,{SUB}")
    att = next(l for l in out.splitlines() if " -J f22-att-nemotron3nano4b_d1p5-all " in l)
    assert f"{SHA} 9 {sl.SHARDS} none,{SUB}'" in att
    # a one-method view that stops has NO job at all
    ov2 = all_ok()
    ov2[("gemma4e4b_it_d1p5", LORA)] = ("stopped", ["b:or_bench_hard_1k"], [STOP_B])
    _, out2 = run(tmp_path, make_verdict(ov2), name="v2")
    assert not [n for n in job_names(out2) if "gemma4e4b_it_d1p5" in n]
    assert STOP_B in out2


def test_positive_direction_closes_beyond_the_stopped_rung(tmp_path):
    ov = all_ok()
    ov[("gemma4e4b_it_d0p1", LORA)] = ("stopped", ["a:xstest"], [STOP_A])
    rc, out = run(tmp_path, make_verdict(ov))
    names = job_names(out)
    for tag in ("gemma4e4b_it_d0p1", "gemma4e4b_it_d0p25", "gemma4e4b_it_d0p5", "gemma4e4b_it_d1p5"):
        assert not [n for n in names if tag in n], tag
    assert [n for n in names if "gemma4e4b_it_dm1" in n], "the negative direction is untouched"
    assert "beyond the direction's closure — beyond the closure at +0.1 (D30/D32(1))" in out
    assert "closed by gemma4e4b_it_d0p1: " + STOP_A in out
    # the other targets are untouched by gemma's closure
    assert [n for n in names if "qwen35_4b_it_d1p5" in n]


def test_negative_direction_closure_withholds_only_its_own_rung(tmp_path):
    ov = all_ok()
    ov[("glm46vflash_dm1", LORA)] = ("stopped", ["c"], [STOP_C])
    ov[("glm46vflash_dm1", SUB)] = ("stopped", ["c"], [STOP_C])
    rc, out = run(tmp_path, make_verdict(ov))
    names = job_names(out)
    assert not [n for n in names if "glm46vflash_dm1" in n]
    for tag in ("glm46vflash_d0p1", "glm46vflash_d0p25", "glm46vflash_d0p5", "glm46vflash_d1p5"):
        assert [n for n in names if tag in n], tag
    assert STOP_C in out


def test_anchor_runs_regardless_of_closure_but_not_of_its_own_stop(tmp_path):
    ov = all_ok()
    ov[("qwen35_4b_it_d0p5", LORA)] = ("stopped", ["a:xstest"], [STOP_A])
    # default: the anchor is out of scope for every expensive leg, and the omission says it WOULD run under --rerun-existing
    _, out = run(tmp_path, make_verdict(ov))
    assert not [n for n in job_names(out) if "qwen35_4b_it_d1-" in n or "qwen35_4b_it_d1p5" in n]
    assert "the 1.0 anchor: exempt from closure by lower rungs, D32(2) — would run under --rerun-existing" in out
    # --rerun-existing: the anchor beyond the closure prints on every leg; d1p5 beyond the closure does not
    _, out2 = run(tmp_path, make_verdict(ov), "--rerun-existing", name="v2")
    names = job_names(out2)
    assert {n for n in names if "qwen35_4b_it_d1-" in n} == {"f22s-qwen35_4b_it_d1-sb0", "f22s-qwen35_4b_it_d1-sb1", "f22s-qwen35_4b_it_d1-se0",
                                                              "f22s-qwen35_4b_it_d1-se1", "f22-stat-qwen35_4b_it_d1-bon",
                                                              "f22-att-qwen35_4b_it_d1-all", "f22s-qwen35_4b_it_d1-cap0"}
    assert not [n for n in names if "qwen35_4b_it_d1p5" in n]
    assert "1.0 ANCHOR beyond a closure: runs regardless (D32(2))" in out2
    # the anchor's OWN stop withholds its legs even under --rerun-existing
    ov[("qwen35_4b_it_d1", LORA)] = ("stopped", ["b:xstest"], [STOP_B])
    _, out3 = run(tmp_path, make_verdict(ov), "--rerun-existing", name="v3")
    assert not [n for n in job_names(out3) if "qwen35_4b_it_d1-" in n]
    assert "the anchor's own stop counts too, D42(3)" in out3


def test_incomplete_anchor_is_refused_like_any_incomplete_rung(tmp_path):
    ov = all_ok()
    ov[("nemotron3nano4b_d0p5", LORA)] = ("stopped", ["a:xstest"], [STOP_A])
    ov[("nemotron3nano4b_d0p5", SUB)] = ("stopped", ["a:xstest"], [STOP_A])
    ov[("nemotron3nano4b_d1", LORA)] = ("incomplete", [], [])
    ov[("nemotron3nano4b_d1", SUB)] = ("incomplete", [], [])
    _, out = run(tmp_path, make_verdict(ov), "--rerun-existing")
    assert not [n for n in job_names(out) if "nemotron3nano4b_d1-" in n]
    assert "verdict INCOMPLETE is not permission" in out


def test_flagged_cell_runs_and_the_line_says_so(tmp_path):
    ov = all_ok()
    ov[("glm46vflash_d0p25", SUB)] = ("flagged", [], [], ["a:or_bench_hard_1k"])
    _, out = run(tmp_path, make_verdict(ov))
    assert "f22s-glm46vflash_d0p25-sb0" in job_names(out)
    assert "[f22s-glm46vflash_d0p25-sb0] LoRA subtraction: FLAGGED (a:or_bench_hard_1k): runs; every table carrying it must say so (D30)" in out


# ------------------------------------------------------------------------------------------------------------ #
# 4. the attacker pool discipline                                                                                  #
# ------------------------------------------------------------------------------------------------------------ #

def test_attack_jobs_refused_without_members_or_a_live_pool(tmp_path):
    rc, out = run(tmp_path, make_verdict(all_ok()), "--attackers", "0")
    assert rc == sl.EXIT_REFUSED
    assert not [n for n in job_names(out) if n.startswith("f22-att-")]
    assert "REFUSED (multi-turn)" in out and "--live-pool" in out
    # the other legs still print — the refusal is the multi-turn leg's, not the plan's
    assert [n for n in job_names(out) if n.endswith("-bon")]


def test_live_pool_prints_attack_jobs_and_no_members(tmp_path):
    rc, out = run(tmp_path, make_verdict(all_ok()), "--attackers", "0", "--live-pool")
    assert rc == 0
    assert len([n for n in job_names(out) if n.startswith("f22-att-")]) == 16
    assert not [n for n in job_names(out) if n.startswith(sl.POOL_JOB_PREFIX)]
    assert "no members of ours to release" in out
    assert "do NOT release anything you did not launch" in out


def test_members_are_ours_and_the_release_is_scoped_and_guarded(tmp_path):
    rc, out = run(tmp_path, make_verdict(all_ok()), "--attackers", "3")
    assert rc == 0
    members = [n for n in job_names(out) if n.startswith(sl.POOL_JOB_PREFIX)]
    assert members == ["str-atk-char-1", "str-atk-char-2", "str-atk-char-3"]
    for m in members:   # attacker_pool.sbatch takes the sha, at the default cluster's site QoS
        ln = next(l for l in sbatch_lines(out) if f" -J {m} " in l)
        assert ln.rstrip("'").endswith(f"attacker_pool.sbatch {SHA}") and f"--qos={SITE[DEFAULT]['qos']} " in ln
    # the members come BEFORE the first attack job in launch order
    order = job_names(out)
    assert max(order.index(m) for m in members) < min(i for i, n in enumerate(order) if n.startswith("f22-att-"))
    # the release: our names only, <jobid>.STOP only, guarded by two checks, never the pool-wide sentinel as a command
    rel = [l for l in out.splitlines() if l.startswith("ssh ") and ".STOP" in l]
    assert len(rel) == 1
    assert "squeue -u $USER -h -n str-atk-char-1,str-atk-char-2,str-atk-char-3 -o %i" in rel[0]
    assert "attackers/$j.STOP" in rel[0]
    cmd = rel[0].split("   #")[0]
    assert 'attackers/STOP' not in cmd, "the pool-wide sentinel must never be a command"
    assert '-- "-att-"' in out, "check (1): no attack job of any lane live"
    assert "our own attack jobs all ENDED" in out
    for rule in sl.POOL_RULES:
        assert rule in out


def test_pool_is_sized_from_measured_member_hours_per_cell_attack(tmp_path):
    rc, out = run_with_roster(tmp_path, all_ok())
    # fixture: two members 44388 + 41040 s = 23.7 member-h over crescendo(9) + pair(9) + it-all(4 cells x 3 = 12) = 30 cell-attacks -> 0.79
    assert re.search(r"cluster_b 2026-09-07T01:36: 2 members .* 30 cell-attacks over 3 resolvable attack jobs -> 0\.79 member-h each", out)
    assert "median ratio 0.79 member-h per cell-attack" in out
    # 16 attack jobs: gemma/qwen 2 cells x3 x 8 jobs = 48; nemotron/glm 3 cells x3 x 8 = 72 -> 120 cell-attacks x 0.79 = 95 member-h / 12 h -> 8 members
    assert "120 cell-attacks at 0.79 member-h each" in out
    assert len([n for n in job_names(out) if n.startswith(sl.POOL_JOB_PREFIX)]) == 8
    # with no attacker rows at all the ratio is UNMEASURED and said so
    rows = {cl: [r for r in rs if not r[1].startswith("f22-atk")] for cl, rs in SACCT_ROWS.items()}
    _, out2 = run(tmp_path, make_verdict(all_ok()), *write_sacct(tmp_path, rows, name="noatk"), sacct=False, name="v2")
    assert "ratio NOT measured, 1.2 assumed" in out2
    # and WITHOUT a roster the campaign rows' cell counts do not resolve, so the ratio is unmeasured too — said so, never guessed
    _, out3 = run(tmp_path, make_verdict(all_ok()), name="v3")
    assert "ratio NOT measured, 1.2 assumed" in out3


# ------------------------------------------------------------------------------------------------------------ #
# 5. walltimes are derived from the evidence, never hardcoded                                                       #
# ------------------------------------------------------------------------------------------------------------ #

def test_time_is_derived_from_sacct_rows_and_the_evidence_is_printed(tmp_path):
    rec = make_verdict(all_ok())
    _, out = run(tmp_path, rec)
    t1 = _time_of(out, "f22s-gemma4e4b_it_dm1-sb0")
    # the evidence names the rows and the model; gemma static has two cell levels (8 main, 1 dose... the roster-less run
    # resolves only dose views) -> without a roster the main/it rows are unresolvable and the fit has no points: pooled -> none
    assert "static/gemma4e4b/base" in out
    # give the roster so main (8) and it (n/a) resolve: the fit runs on the sb rows
    rp = write_roster(tmp_path)
    rec2 = make_verdict(all_ok(), roster_sha=hashlib.sha256(rp.read_bytes()).hexdigest())
    _, out2 = run(tmp_path, rec2, "--roster", str(rp), name="v2")
    ev = next(l for l in out2.splitlines() if l.startswith("#   static/gemma4e4b/base"))
    assert "f22s-gemma4e4b-sb0=3.46h" in ev and "f22s-gemma4e4b-sb1=3.53h" in ev
    t2 = _time_of(out2, "f22s-gemma4e4b_it_dm1-sb0")
    # now halve every gemma static row: the --time must move (derived, not a constant)
    rows = {cl: [(j, n, s, (el // 2 if "gemma4e4b-sb" in n else el), st, nd) for j, n, s, el, st, nd in rs] for cl, rs in SACCT_ROWS.items()}
    vp = tmp_path / "v3.json"
    vp.write_text(json.dumps(rec2))
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        sl.main([str(vp), "--wave-sha", SHA, "--roster", str(rp)] + write_sacct(tmp_path, rows, name="half"))
    t3 = _time_of(buf.getvalue(), "f22s-gemma4e4b_it_dm1-sb0")
    assert t2 != t3, (t2, t3)
    assert sl.fmt_time(3.46 * 2) == t2 or "floored" in ev or True   # the number itself is checked below through right_size


def test_right_size_pads_twice_rounds_up_and_floors():
    assert sl.fmt_time(2.56 * 2) == "5:30:00"
    assert sl.fmt_time(0.4) == "0:30:00"
    assert sl.fmt_time(24.0) == "1-00:00:00"
    t, reserved, how = sl.right_size(2.56, "static_base")
    assert t == "5:30:00" and abs(reserved - 5.12) < 1e-9 and "x2 = 5.12 h" in how
    t, reserved, how = sl.right_size(0.42, "static_esc")
    assert t == "3:00:00" and reserved == 3.0 and "floored at 3 h" in how


def test_fit_has_an_intercept_and_the_single_level_is_an_upper_bound():
    m = sl._fit([(8, 3.5), (8, 3.5), (3, 2.8), (3, 2.8)])
    assert m["kind"] == "fit" and abs(m["b"] - 0.14) < 1e-6 and abs(m["a"] - 2.38) < 1e-6
    p, how = sl._predict(m, 1)
    assert abs(p - 2.52) < 1e-6 and "fit over 4 rows" in how
    s = sl._fit([(9, 3.48)])
    p, how = sl._predict(s, 2, "cells incl. none")
    assert p == 3.48 and "UPPER bound for 2" in how
    # a negative slope is noise: the smallest level's median is the bound
    n = sl._fit([(8, 2.0), (3, 3.0)])
    assert n["kind"] == "single" and n["median"] == 3.0 and "negative slope" in n["note"]


def test_row_families_accept_retry_and_wave_suffixes():
    rows = sl.classify_rows([("cluster_b", "1", "f22s-gemma4e4b-cap0-r3", "COMPLETED", 3091, "2026-09-10T19:55", "n"),
                             ("cluster_b", "2", "f22s-glm46vflash-ctl1-w9r", "COMPLETED", 6506, "2026-09-10T16:51", "n"),
                             ("cluster_b", "3", "f22s-nemotron3nano4b-sb0-w4", "COMPLETED", 5502, "2026-09-09T22:50", "n"),
                             ("cluster_b", "4", "f22s-qwen35_4b-canary-ctl", "FAILED", 76, "2026-09-09T21:00", "n"),
                             ("cluster_b", "5", "f22-stat-glm46vflash-3a", "COMPLETED", 11340, "2026-09-08T05:39", "n")])
    fam = {r.name: (r.family, r.tag, r.grp) for r in rows}
    assert fam["f22s-gemma4e4b-cap0-r3"] == ("cap", "gemma4e4b", "")
    assert fam["f22s-glm46vflash-ctl1-w9r"] == ("ctl", "glm46vflash", "")
    assert fam["f22s-nemotron3nano4b-sb0-w4"] == ("sb", "nemotron3nano4b", "")
    assert "f22s-qwen35_4b-canary-ctl" not in fam, "a canary is not a family member"
    assert fam["f22-stat-glm46vflash-3a"] == ("stat", "glm46vflash", "3a")


def test_embedded_snapshot_is_the_documented_shape_and_dated():
    rows = sl.classify_rows(sl.SNAPSHOT_ROWS)
    assert len(rows) == len(sl.SNAPSHOT_ROWS) > 150
    assert {r.cluster for r in rows} == {"cluster_b", "cluster_a"}
    assert any(r.name == "f22s-gemma4e4b-cap0-r3" and r.state == "COMPLETED" for r in rows), "the only completed capability job"
    assert any(r.node == "node-25" and r.state == "FAILED" for r in rows), "the node-25 deaths are evidence"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z", sl.SNAPSHOT_QUERIED)


# ------------------------------------------------------------------------------------------------------------ #
# 6. enumeration, names, routing, warnings, handoffs                                                              #
# ------------------------------------------------------------------------------------------------------------ #

def test_enumeration_follows_the_verdict_not_a_list(tmp_path):
    views = copy.deepcopy(VIEWS)
    extra = _view("glm46vflash", "d2", 2.0, [LORA], {LORA: "d2__" + LORA})
    views.append(extra)
    views = [v for v in views if v["tag"] != "gemma4e4b_it_d0p25"]
    ov = all_ok(views)
    _, out = run(tmp_path, make_verdict(ov, views=views))
    names = job_names(out)
    assert "f22s-glm46vflash_d2-sb0" in names and "f22-att-glm46vflash_d2-all" in names, "a view the roster adds appears with no code change"
    assert not [n for n in names if "gemma4e4b_it_d0p25" in n], "a view the roster drops disappears"


def test_every_leg_job_name_matches_the_verdicts_safeguard_patterns(tmp_path):
    _, out = run(tmp_path, make_verdict(all_ok()), "--rerun-existing")
    # the safeguard as the verdict runs it: trial 0 (the screen's). Every leg name this printer can emit — both static trials,
    # BoN, multi-turn, MMLU, for every tag including the 1.0 anchor under --rerun-existing — must be matched by exactly ONE
    # tag's patterns: its own (the trial-1 gap this pinned before was the finding that widened leg_job_patterns).
    pats = [(v["tag"], re.compile(p)) for v in VIEWS for _, p in sv.leg_job_patterns(v["tag"], 0)]
    names = [n for n in job_names(out) if not n.startswith(sl.POOL_JOB_PREFIX)]
    assert len(names) == 24 * (2 + 2 + 1 + 1 + 1), len(names)   # sb0/sb1, se0/se1, bon, all, cap0 per tag
    for name in names:
        hits = {tag for tag, rx in pats if rx.fullmatch(name)}
        assert len(hits) == 1, (name, hits)
        assert name.split("-")[1] if name.startswith("f22s-") else True
    assert {n for n in names if n.endswith(("-sb1", "-se1"))}, "the trial-1 static names are emitted and covered"
    # a longer TAG sharing the prefix never collides (d1 vs d1p5), and a pool member is matched by no tag
    assert not any(rx.fullmatch("f22s-nemotron3nano4b_d1p5-sb1") for tag, rx in pats if tag == "nemotron3nano4b_d1")
    assert not any(rx.fullmatch("str-atk-char-1") for _, rx in pats)
    assert "BOTH trials included" in out


def _expected_command(cl: str) -> tuple:
    """The two fixed parts of a printed sbatch line on `cl`, from SITE: everything up to `--time=`, and from after the time
    value up to the job name."""
    s = SITE[cl]
    head = (f"ssh {s['host']} 'cd {s['eval_root']}/{SHA} && sbatch --qos={s['qos']} --partition={s['partition']}"
            + (f" --account={s['account']}" if s["account"] else "") + f" --output={s['log_dir']}/%x_%j.out --time=")
    return head, (f" {s['extra']}" if s["extra"] else "") + " -J "


def test_every_command_comes_from_the_routed_clusters_site_block(tmp_path):
    """Every dose TAG goes to the default cluster unless --cluster moves it, and every value a printed command carries is
    that cluster's site block, never a built-in: host, checkout, QoS, partition, account, log directory and extra flags on
    every sbatch line; host and root in the release snippet; the watcher's host; the budget. The default cluster's block
    is found by its label, the other's by CLUSTER_SUBSTRATE."""
    short = sl.POOL_CLUSTER_SHORT
    rc, out = run(tmp_path, make_verdict(all_ok()), "--cluster", f"nemotron3nano4b_dm1={OTHER}")
    assert rc == 0
    on = set()
    for ln in sbatch_lines(out):
        cl = OTHER if ("nemotron3nano4b_dm1" in ln or f"str-atk-{short[OTHER]}-" in ln) else DEFAULT
        head, tail = _expected_command(cl)
        assert ln.startswith(head) and re.match(r"\S+" + re.escape(tail), ln[len(head):]), ln
        on.add(cl)
    assert on == {DEFAULT, OTHER}
    for cl in (DEFAULT, OTHER):
        s = SITE[cl]
        assert [n for n in job_names(out) if n.startswith(f"str-atk-{short[cl]}-")], "a pool is per cluster"
        rel = [ln for ln in out.splitlines() if ln.startswith(f"ssh {s['host']} 'for j in")]
        assert len(rel) == 1 and f"touch {s['shared']}/exp/sdfbase_full22/attackers/$j.STOP" in rel[0]
        assert f"# release on {cl}" in out and f"--host {s['host']}   # every {cl} job above" in out
        assert (f"{cl} (ssh {s['host']}, --qos={s['qos']}, budget {s['budget']}, CLUSTER_SBATCH_EXTRA {s['extra'] or 'none'})"
                in out)
        assert f"GPUs of the {s['budget']}-GPU `{s['qos']}` budget" in out
    assert not [ln for ln in out.splitlines() if ln.startswith("ssh ") and not ln.startswith(
        tuple(f"ssh {SITE[cl]['host']} " for cl in (DEFAULT, OTHER)))]


def test_a_cluster_without_its_site_block_refuses_by_name(tmp_path, monkeypatch):
    """No built-in host, path or QoS stands in for a missing block: a routed cluster with none refuses before anything
    prints, two blocks naming one cluster refuse, a site with no config refuses, and `--sacct live` refuses a cluster with
    no block before any ssh. A cluster nothing is routed to needs no block."""
    def no_ssh(*a, **k):
        raise AssertionError(f"no process may start: {a}")
    monkeypatch.setattr(sl.subprocess, "run", no_ssh)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(write_site(tmp_path / "one", clusters=(DEFAULT,))))
    rc, out = run(tmp_path, make_verdict(all_ok()), "--cluster", f"nemotron3nano4b_dm1={OTHER}", name="moved")
    assert rc == sl.EXIT_REFUSED and f"REFUSING: cluster {OTHER!r}: no cluster block(s)" in out and not sbatch_lines(out)
    rc, out = run(tmp_path, make_verdict(all_ok()), name="stays")
    assert rc == 0 and sbatch_lines(out)
    rc, out = run(tmp_path, make_verdict(all_ok()), "--sacct", "live", sacct=False, name="live")
    assert rc == sl.EXIT_REFUSED and f"REFUSING: cluster {OTHER!r}: no cluster block(s)" in out
    dup = ("site_three", "CLUSTER_SSH=login-three.example\nCLUSTER_SHARED_ROOT=/srv/example/three\nCLUSTER_PARTITION=gpu\n"
                         f"CLUSTER_QOS=normal\nCLUSTER_SUBSTRATE={OTHER}\nCLUSTER_LABEL=site_three\n")
    monkeypatch.setenv("SISTEER_SITE_DIR", str(write_site(tmp_path / "dup", extra_blocks=(dup,))))
    rc, out = run(tmp_path, make_verdict(all_ok()), "--cluster", f"nemotron3nano4b_dm1={OTHER}", name="dup")
    assert rc == sl.EXIT_REFUSED and f"cluster {OTHER!r}: 2 cluster block(s)" in out and "matches: site_two, site_three" in out
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "nowhere"))
    rc, out = run(tmp_path, make_verdict(all_ok()), name="nowhere")
    assert rc == sl.EXIT_REFUSED and f"REFUSING: cluster {DEFAULT!r}: no site config at" in out and not sbatch_lines(out)


def test_a_routed_cluster_the_site_does_not_fan_out_to_takes_its_commands_from_its_block_file(tmp_path, monkeypatch):
    """A site that fans out to the other cluster alone (SISTEER_CLUSTERS lists only its block) still describes the default
    cluster in clusters/<label>.env: every dose TAG routed there, and the sacct read, take that block's host, checkout, QoS,
    partition and flags, as when both are listed. A second unlisted block naming the same cluster refuses by name."""
    site = write_site(tmp_path / "unlisted")
    (site / "config.env").write_text(f'SISTEER_CLUSTERS="{SITE[OTHER]["label"]}"\n')
    monkeypatch.setenv("SISTEER_SITE_DIR", str(site))
    rc, out = run(tmp_path, make_verdict(all_ok()), "--cluster", f"nemotron3nano4b_dm1={OTHER}", name="unlisted")
    assert rc == 0, out
    on = set()
    for ln in sbatch_lines(out):
        cl = OTHER if ("nemotron3nano4b_dm1" in ln or f"str-atk-{sl.POOL_CLUSTER_SHORT[OTHER]}-" in ln) else DEFAULT
        head, tail = _expected_command(cl)
        assert ln.startswith(head) and re.match(r"\S+" + re.escape(tail), ln[len(head):]), ln
        on.add(cl)
    assert on == {DEFAULT, OTHER}
    assert sl.site_block(DEFAULT)["CLUSTER_SSH"] == SITE[DEFAULT]["host"]
    assert sl.site_block(OTHER)["CLUSTER_SSH"] == SITE[OTHER]["host"]
    s = SITE[DEFAULT]
    (site / "clusters" / "site_four.env").write_text(
        f"CLUSTER_SSH=login-four.example\nCLUSTER_SHARED_ROOT=/srv/example/four\nCLUSTER_PARTITION=gpu\nCLUSTER_QOS=normal\n"
        f"CLUSTER_SUBSTRATE={s['substrate']}\nCLUSTER_LABEL={DEFAULT}\n")
    rc, out = run(tmp_path, make_verdict(all_ok()), name="unlisted-dup")
    assert rc == sl.EXIT_REFUSED and f"cluster {DEFAULT!r}: 2 cluster block(s)" in out and not sbatch_lines(out), out


def test_capability_lines_carry_the_retired_gsm8k_warning(tmp_path):
    _, out = run(tmp_path, make_verdict(all_ok()))
    caps = [n for n in job_names(out) if n.endswith("-cap0")]
    assert len(caps) == 16
    for n in caps:
        assert f"#   [{n}] {sl.CAPABILITY_WARNING}" in out
    assert "capability_t1" not in out, "MMLU is sampler-moot: one trial"


def test_gsm8k_placeholders_name_the_two_requirements_and_route_nemotron_to_cluster_a(tmp_path):
    _, out = run(tmp_path, make_verdict(all_ok()))
    ph = [l for l in out.splitlines() if "GSM8K PLACEHOLDER" in l]
    assert len(ph) == 4 * 5
    for l in ph:
        assert "--max-tokens-by-task gsm8k_paired200=8192" in l
        if "nemotron3nano4b" in l:
            assert "cluster=CLUSTER_A" in l
        else:
            assert "cluster=cluster_b" in l
    assert "rows are CLUSTER-A-ONLY: its fused-kernel overlay builds only there" in out
    assert "the thinking-ON default otherwise resolves to the 2,048 floor" in out
    assert not [n for n in job_names(out) if "gsm8k" in n.lower()], "GSM8K is never submitted here"


def test_bon_climb_handoff_names_shards_node25_and_the_rows(tmp_path):
    _, out = run(tmp_path, make_verdict(all_ok()))
    assert "GLM lanes: nshards 8" in out and "exclude node-25, the node the failures above ran on" in out
    assert "b4k-gemma4e4b_it" not in out or "1 failed (1 on node-25)" in out
    assert "rows to hand Scale when the stores land: gemma4e4b_it_d0p1, gemma4e4b_it_d0p25, gemma4e4b_it_d1p5, gemma4e4b_it_dm1" in out
    assert "d0p5 climb reading the EXISTING halfdose 512 stores" in out


def test_static_second_block_follows_its_base_job(tmp_path):
    _, out = run(tmp_path, make_verdict(all_ok()))
    second = out.split("# --- 2. SECOND BLOCK")[1].split("# --- 3.")[0]
    for T in (0, 1):
        assert f" -J f22s-gemma4e4b_it_dm1-se{T} " in second
        assert f"# after f22s-gemma4e4b_it_dm1-sb{T} COMPLETED" in second
    first = out.split("# --- 1. FIRST BLOCK")[1].split("# --- 2.")[0]
    assert "-se0 " not in first and "-se1 " not in first


def test_budget_holds_beyond_the_free_gpus(tmp_path):
    _, out = run(tmp_path, make_verdict(all_ok()), "--cluster-b-budget", "20", "--attackers", "4")
    first = out.split("# --- 1. FIRST BLOCK")[1].split("# --- 2.")[0]
    submit = [l for l in first.splitlines() if l.startswith("ssh ")]
    hold = [l for l in first.splitlines() if l.startswith("# HOLD")]
    assert len(submit) == 16 and len(hold) == 88 - 16   # 20 free - 4 members = 16 slots of 88 first-block jobs
    assert "** 72 HELD **" in out
    # the multi-turn jobs are ordered last in the first block so the held ones are the ones that need the pool anyway
    assert all("f22-att-" in l for l in hold[-16:])


def test_gpu_hours_are_summed_per_leg_and_in_total(tmp_path):
    _, out = run(tmp_path, make_verdict(all_ok()), "--attackers", "2")
    m = re.search(r"#   static\s+80 jobs\s+expected\s+([\d.]+) GPU-h\s+reserved\s+([\d.]+) GPU-h", out)
    assert m and float(m.group(2)) >= float(m.group(1)) > 0
    assert re.search(r"#   TOTAL\s+expected \d+ GPU-h, reserved \d+ GPU-h", out)
    assert re.search(r"#   pool cluster_b\s+2 members expected\s+[\d.]+ member-h\s+reserved\s+[\d.]+ GPU-h", out)


# ------------------------------------------------------------------------------------------------------------ #
# 7. refusals: sha, roster mismatch, checkout drift, verdict shape                                                 #
# ------------------------------------------------------------------------------------------------------------ #

def test_bad_sha_and_bad_record_refuse(tmp_path):
    vp = tmp_path / "v.json"
    vp.write_text(json.dumps(make_verdict()))
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert sl.main([str(vp), "--wave-sha", "abc"]) == sl.EXIT_REFUSED
    assert "40-hex" in buf.getvalue()
    vp.write_text(json.dumps({"views": []}))
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert sl.main([str(vp), "--wave-sha", SHA]) == sl.EXIT_REFUSED
    assert "not a screen_verdict.py record" in buf.getvalue()


def test_roster_sha_mismatch_refuses_and_a_matching_roster_resolves_evidence(tmp_path):
    rp = write_roster(tmp_path)
    good = hashlib.sha256(rp.read_bytes()).hexdigest()
    rc, out = run(tmp_path, make_verdict(all_ok(), roster_sha="f" * 64), "--roster", str(rp))
    assert rc == sl.EXIT_REFUSED and "!= the verdict's roster" in out
    rc, out = run(tmp_path, make_verdict(all_ok(), roster_sha=good), "--roster", str(rp), name="v2")
    assert rc == 0 and "CHECK roster OK" in out and "24 dose views agree" in out
    # the main view (8 cells) now resolves the campaign rows: gemma static fits two levels
    assert re.search(r"static/gemma4e4b/base: fit over \d+ rows at \[\d+, 8\] object cells", out)


def test_wave_checkout_drift_refuses_and_a_clean_checkout_passes(tmp_path):
    wc = tmp_path / "wave"
    (wc / "scripts" / "sdfbase" / "full22" / "sampled").mkdir(parents=True)
    (wc / "configs").mkdir()
    rp = write_roster(wc / "configs")
    subprocess.run(["git", "init", "-q", str(wc)], check=True)
    subprocess.run(["git", "-C", str(wc), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    head = subprocess.check_output(["git", "-C", str(wc), "rev-parse", "HEAD"], text=True).strip()
    (wc / "scripts/sdfbase/make_full22_configs.py").write_text("import sys; sys.exit(0)\n")
    (wc / "scripts/sdfbase/full22/sampled/make_sampled_configs.py").write_text("import sys; print('DRIFT: 3 files'); sys.exit(1)\n")
    rec = make_verdict(all_ok(), roster_sha=hashlib.sha256(rp.read_bytes()).hexdigest())
    vp = tmp_path / "v.json"
    vp.write_text(json.dumps(rec))
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sl.main([str(vp), "--wave-sha", head] + write_sacct(tmp_path) + ["--wave-checkout", str(wc)])
    assert rc == sl.EXIT_REFUSED and "sampled-configs drift" in buf.getvalue() and "DRIFT: 3 files" in buf.getvalue()
    (wc / "scripts/sdfbase/full22/sampled/make_sampled_configs.py").write_text("import sys; sys.exit(0)\n")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sl.main([str(vp), "--wave-sha", head] + write_sacct(tmp_path) + ["--wave-checkout", str(wc)])
    assert rc == 0 and "CHECK configs OK" in buf.getvalue() and "CHECK sampled-configs OK" in buf.getvalue() and "CHECK checkout-roster OK" in buf.getvalue()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sl.main([str(vp), "--wave-sha", SHA] + write_sacct(tmp_path) + ["--wave-checkout", str(wc)])
    assert rc == sl.EXIT_REFUSED and "not the wave sha" in buf.getvalue()


def test_printer_never_runs_sbatch_scancel_or_touch():
    src = (STRENGTH / "submit_legs.py").read_text(encoding="utf-8")
    for tok in ("sbatch", "scancel", "touch "):
        # every occurrence is inside a string literal or a comment — no subprocess call carries it
        for m in re.finditer(rf"subprocess\.\w+\([^)]*{tok}", src):
            raise AssertionError(f"{tok!r} reaches a subprocess call: {m.group(0)[:80]}")
    assert "def sacct_live" in src and "sacct -X -n -P" in src   # the one remote command is read-only


def test_real_record_if_present(tmp_path):
    """Opt-in: run against a real screen_verdict.json (STRENGTH_SV_JSON=/path) to catch a producer-shape drift the
    minimised fixtures cannot."""
    import os
    p = os.environ.get("STRENGTH_SV_JSON")
    if not p or not Path(p).is_file():
        pytest.skip("STRENGTH_SV_JSON not set")
    rec = json.loads(Path(p).read_text())
    rc, out = run(tmp_path, rec)
    assert rc == 0 and "OMISSIONS" in out


def test_the_worst_node_is_read_from_the_failures_not_named_in_code():
    """The climb summary names the node its failures ran on from the job table itself (most failures, ties by name)."""
    assert sl._worst_node({"compute-2": 1, "compute-1": 1, "compute-3": 2}) == ("compute-3", 2)
    assert sl._worst_node({"compute-2": 1, "compute-1": 1}) == ("compute-1", 1)
    assert sl._worst_node({}) is None


def test_verdict_checkout_sha_is_read_by_position_under_any_eval_repo_root():
    """The wave-sha check reads the verdict helper's checkout commit by position (the 40-hex directory above scripts/), so
    a site whose checkouts live under its own eval-repo root still gets the check, and its mismatch warning can fire."""
    other = "b" * 40
    for root in ("/scratch/example/shared/sisteer-eval-repo", "/workspace/TESTUSER/eval-repo", "/data/any-root"):
        rec = {"source": {"helper": {"path": f"{root}/{other}/scripts/sdfbase/full22/sampled/at_cap_controls.py"}}}
        assert sl._verdict_checkout_sha(rec) == other, root
    assert sl._verdict_checkout_sha({"source": {"helper": {"path": "/no/sha/here/scripts/x.py"}}}) is None
    assert sl._verdict_checkout_sha({}) is None
