"""LIFBench proper (arXiv:2411.07037, ACL 2025): adapter + runner.

The benchmark: 3 scenarios (List / MultiDoc / OneDoc), 11 tasks, 2766
instructions auto-scaled across 6 length intervals (4k...128k), scored by
its own programmatic LIFEval rubric — NO LLM judge.

The scorer and data are NOT re-implemented here: they come from a pinned
checkout of github.com/SheldonWu0327/LIFBench-2024 (MIT license, verified
2026-08-13) at <repo>/vendor/lifbench/ (fetched by scripts/
fetch_stage2_data.py; a pinned commit SHA is recorded there). This module
imports the upstream judge functions by file path and drives them
per-record:

- data:   vendor/lifbench/data/prompts/<task>.json — records with
          {prompt, label, param, length, ins_id, param_id}.
- scorer: vendor/lifbench/evaluation/EvaluateFunc.py judge_* functions,
          exactly the mapping in evaluation/Evaluator.py func_map. Each
          judge takes {**record, "output": generation} and returns a
          score_dict whose 'total_score' is the rubric-normalized ARS
          contribution. Judges MUTATE record["label"], so records are
          deep-copied per call.
- ARS aggregation weights: evaluation/ARS_Scores.py ScoreTools (their
  per-task rubric totals, Eq. 3 of the paper).

Stability across length is the paper's IFS (Eq. 6, length perspective):
IFS_t = sigma({ARS_t per length interval}) / mean(...), lower = more
stable. The repo ships no stability function, so Eq. 6 is computed here
(ddof=0 for sigma; the paper does not specify).

Arms: prompted (their protocol as published) + deployable compiled
interventions. Instruction-vs-context splitting for compiled arms is
task-template-dependent and belongs to the DRIVER (data/meta/tasks.json);
this runner accepts a prompt_builder(record) -> user_text seam and
defaults to the record's prompt verbatim.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

# Mirror of evaluation/Evaluator.py func_map (task -> judge function name);
# load_judges() asserts every name exists in the vendored module, so drift
# in a re-pinned checkout fails loudly.
TASK_JUDGE_FUNCS: dict[str, str] = {
    "onedoc-repeat": "judge_onedoc_repeat",
    "onedoc-qa": "judge_onedoc_qa",
    "onedoc-extract": "judge_onedoc_extract",
    "list-single_query_id": "judge_label_equal_output",
    "list-multi_query_id": "judge_labels_equal_outputs",
    "list-offset_query_id": "judge_label_equal_output",
    "list-offset_query_element": "judge_label_equal_output",
    "list-blur_offset_query_id": "judge_list_input_blur_offset_query",
    "list-blur_offset_query_element": "judge_list_input_blur_offset_query",
    "multidoc-batch_label": "judge_multidoc_batch_label",
    "multidoc-find_dup_text": "judge_multidoc_find_dup_text",
}
TASKS = list(TASK_JUDGE_FUNCS)


def default_vendor_dir() -> Path:
    # src/sisteer/eval/lifbench.py -> repo root / vendor / lifbench
    return Path(__file__).resolve().parents[3] / "vendor" / "lifbench"


def vendor_present(vendor_dir: Path | None = None) -> bool:
    root = Path(vendor_dir) if vendor_dir else default_vendor_dir()
    return (root / "evaluation" / "EvaluateFunc.py").is_file()


def _import_by_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_judges(vendor_dir: Path | None = None) -> dict[str, Callable]:
    """task -> vendored judge function. Loud error when the checkout is
    missing — never a silent no-score."""
    root = Path(vendor_dir) if vendor_dir else default_vendor_dir()
    func_path = root / "evaluation" / "EvaluateFunc.py"
    if not func_path.is_file():
        raise FileNotFoundError(
            f"LIFBench vendored checkout not found at {root} (expected "
            "evaluation/EvaluateFunc.py). Run scripts/fetch_stage2_data.py "
            "to clone github.com/SheldonWu0327/LIFBench-2024 (MIT) there."
        )
    module = _import_by_path(func_path, "sisteer_lifbench_evaluatefunc")
    judges = {}
    for task, name in TASK_JUDGE_FUNCS.items():
        if not hasattr(module, name):
            raise AttributeError(
                f"vendored EvaluateFunc.py is missing {name} (task {task}); "
                "the pinned LIFBench commit changed — re-verify the mapping"
            )
        judges[task] = getattr(module, name)
    return judges


def load_score_tools(vendor_dir: Path | None = None):
    """Their ARS_Scores.ScoreTools (per-task rubric weights, Eq. 3)."""
    root = Path(vendor_dir) if vendor_dir else default_vendor_dir()
    path = root / "evaluation" / "ARS_Scores.py"
    if not path.is_file():
        raise FileNotFoundError(f"ARS_Scores.py not found under {root}")
    return _import_by_path(path, "sisteer_lifbench_ars").ScoreTools()


def load_task_records(task: str, vendor_dir: Path | None = None) -> list[dict]:
    root = Path(vendor_dir) if vendor_dir else default_vendor_dir()
    path = root / "data" / "prompts" / f"{task}.json"
    if not path.is_file():
        raise FileNotFoundError(f"LIFBench prompts not found: {path}")
    with path.open() as f:
        return json.load(f)


def score_output(task: str, record: dict, output: str, judges: dict[str, Callable]) -> dict:
    """Run the vendored judge on one generation. Deep-copies the record:
    upstream judges mutate label in place (verified in
    judge_onedoc_repeat), which would corrupt reuse across arms."""
    payload = copy.deepcopy(record)
    payload["output"] = output
    return judges[task](payload)


# The benchmark's on-disk `length` field is a coded INTERVAL LABEL, not a
# token count (verified against data/prompts/*.json at the pinned commit).
LENGTH_CODES = (3, 6, 13, 28, 60, 120)


# -- instance-level train/held-out split --------------------------------------------
#
# The compiled arms train per-task objects on shortest-interval instances;
# scoring the same benchmark is IN-SAMPLE at 4k and CONTAMINATED at every
# longer interval, because the benchmark auto-scales the SAME base
# instances across intervals. The split below excludes the trained base
# instances (all their scaled variants) from scoring at EVERY interval.
#
# Base-instance identity: (ins_id, param_id). Verified against the pinned
# commit adb554da (2026-08-13, all 11 tasks): every task carries an
# IDENTICAL (ins_id, param_id) set at all 6 length intervals, with no
# duplicates within an interval — the label/context are rescaled, the
# identity is stable. check_instance_stability re-verifies at run time so
# a re-pinned checkout that breaks this fails loudly instead of silently
# splitting on a no-longer-valid key.


def instance_key(record: dict) -> tuple:
    """Base-instance identity of a record: stable across length intervals."""
    return (record["ins_id"], record["param_id"])


def check_instance_stability(records: list[dict], task: str) -> list[tuple]:
    """Assert the (ins_id, param_id) set is identical at every interval;
    return the sorted instance keys. Raises on drift (a split keyed on an
    unstable identity would silently leak trained instances)."""
    by_len: dict[int, set[tuple]] = {}
    counts: dict[int, int] = {}
    for r in records:
        by_len.setdefault(r["length"], set()).add(instance_key(r))
        counts[r["length"]] = counts.get(r["length"], 0) + 1
    if not by_len:
        raise ValueError(f"no records for task {task!r}")
    key_sets = list(by_len.values())
    if any(s != key_sets[0] for s in key_sets[1:]) or any(
        len(by_len[length]) != counts[length] for length in by_len
    ):
        raise ValueError(
            f"LIFBench task {task!r}: (ins_id, param_id) is not a stable "
            "base-instance identity across length intervals in this "
            "checkout (differing sets or within-interval duplicates) — the "
            "held-out split cannot be keyed on it; re-verify the pinned "
            "commit before running compiled arms"
        )
    return sorted(key_sets[0])


def train_heldout_split(
    records: list[dict], n_train: int, seed: int, *, task: str
) -> tuple[list[tuple], list[tuple]]:
    """Deterministic instance-level split: (train_keys, heldout_keys).

    train_keys are the base instances compiled objects may train on (at the
    shortest interval); their scaled variants are EXCLUDED from scoring at
    every interval. The prompted baseline must be scored with the SAME
    (n_train, seed) so all arms share one held-out set. Seeded per task via
    a string seed (stable across processes, unlike hash())."""
    keys = check_instance_stability(records, task)
    if not 0 < n_train < len(keys):
        raise ValueError(
            f"n_train={n_train} leaves no held-out instances for task "
            f"{task!r} ({len(keys)} base instances per interval)"
        )
    rng = random.Random(f"lifbench-split:{task}:{seed}")
    train = set(rng.sample(keys, n_train))
    return sorted(train), [k for k in keys if k not in train]


def training_records(
    records: list[dict], n_train: int, seed: int, *, task: str, length: int
) -> list[dict]:
    """DRIVER SEAM: the records a compiled object may be trained on — the
    train-split instances at `length` (an on-disk interval CODE, normally
    the shortest). The driver must build its distillation examples from
    THIS, never from records[:n_train], and pass the same (n_train,
    split_seed) to run_lifbench for every arm including prompted."""
    if length not in {r["length"] for r in records}:
        raise ValueError(
            f"length={length} matches no {task!r} records; on-disk lengths "
            f"are coded interval labels {sorted({r['length'] for r in records})}"
        )
    train, _ = train_heldout_split(records, n_train, seed, task=task)
    train_set = set(train)
    return [r for r in records
            if r["length"] == length and instance_key(r) in train_set]


def load_results(out_jsonl: Path) -> list[dict]:
    """Every row on disk (tolerant of a torn final line). Aggregations
    must run on THIS, not on run_lifbench's return value — that returns
    only newly-generated rows, so after a resume it under-counts (empty
    on a no-op resume) and the ARS/IFS aggregate would silently drop the
    already-finished work."""
    rows: list[dict] = []
    out_jsonl = Path(out_jsonl)
    if out_jsonl.exists():
        for line in out_jsonl.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn final line from a SIGKILL mid-append
    return rows


def _done_items(out_jsonl: Path) -> set[tuple]:
    # scale discriminates the key so one arm JSONL can never be silently
    # reused across a different scale.
    return {
        (r["task"], r["ins_id"], r["param_id"], r["length"], r.get("scale"))
        for r in load_results(out_jsonl)
    }


def run_lifbench(
    model,
    tok,
    arm: str,
    out_dir: Path,
    *,
    obj=None,
    scale: float = 1.0,
    vendor_dir: Path | None = None,
    tasks: list[str] | None = None,
    lengths: list[int] | None = None,
    max_per_task: int | None = None,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    prompt_builder: Callable[[dict], str] | None = None,
    generate_fn: Callable[[str], str] | None = None,
    n_train: int | None = None,
    split_seed: int | None = None,
) -> list[dict]:
    """Generate + LIFEval-score one arm; per-item JSONL resume.

    prompt_builder(record) -> user text (default: record['prompt']
    verbatim — the published prompted protocol). Compiled arms pass the
    driver's instruction-stripped builder plus their object.

    n_train + split_seed (both or neither) enable the instance-level
    held-out split: the n_train base instances per task that compiled
    objects train on (see training_records) are EXCLUDED from scoring
    at EVERY interval — all their scaled variants. DRIVER CONTRACT:
    pass the SAME (n_train, split_seed) for every arm, prompted
    included, so all arms score the one held-out set; train compiled
    objects only on training_records(...). Scored records gain
    `split`/`split_seed` tags, and a results file already holding a
    training-split instance's record (an earlier contaminated run, or a
    different seed) fails LOUDLY instead of resuming over it.
    """
    from ..generate import generate_batch
    from ..models import build_chat_prompt

    if (n_train is None) != (split_seed is None):
        raise ValueError(
            "pass BOTH n_train and split_seed (the held-out split) or "
            "neither — one without the other would silently score the "
            "full, contaminated benchmark"
        )
    judges = load_judges(vendor_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{arm}.jsonl"
    done = _done_items(out_jsonl)

    if generate_fn is not None and obj is not None:
        raise ValueError("pass obj or generate_fn, not both (generate_fn "
                         "replaces generation, so obj would be silently ignored)")

    def _gen(rendered: str) -> str:
        if generate_fn is not None:
            return generate_fn(rendered)
        # startswith covers BOTH token kinds ("token", "token_appended") —
        # the same guard as every other suite.
        if obj is not None and (getattr(obj, "kind", "") or "").startswith("token"):
            rendered = obj.inject(rendered)
        ctx = obj.apply(model, scale=scale) if obj is not None else nullcontext()
        with ctx:
            return generate_batch(
                model, tok, [rendered], batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )[0]

    rows: list[dict] = []
    for task in tasks or TASKS:
        records = load_task_records(task, vendor_dir)
        train_set: set[tuple] = set()
        if n_train is not None:
            # Split on the FULL record set (before any length filter) so
            # the exclusion covers every interval's scaled variants.
            train_keys, _held = train_heldout_split(
                records, n_train, split_seed, task=task
            )
            train_set = set(train_keys)
            # Scoring-path enforcement: a results file that already holds a
            # record for a training-split instance means an earlier run was
            # contaminated (or ran a different seed) — resuming/aggregating
            # over it would silently include in-sample rows.
            poisoned = [
                r for r in load_results(out_jsonl)
                if r.get("task") == task
                and (r.get("ins_id"), r.get("param_id")) in train_set
            ]
            if poisoned:
                raise ValueError(
                    f"{out_jsonl} holds {len(poisoned)} record(s) for "
                    f"TRAINING-split instances of task {task!r} (e.g. "
                    f"{(poisoned[0].get('ins_id'), poisoned[0].get('param_id'))}) "
                    "— contaminated by an earlier run or a different "
                    "split seed; refusing to resume over in-sample rows"
                )
            records = [r for r in records if instance_key(r) not in train_set]
        if lengths:
            on_disk = sorted({r["length"] for r in records})
            filtered = [r for r in records if r["length"] in lengths]
            if not filtered:
                # A token-interval filter ([4096, ...]) matches NO records:
                # succeeding with an empty benchmark would look like a
                # finished arm with zero items — fail loudly.
                raise ValueError(
                    f"lengths={sorted(lengths)} matched no {task} records; "
                    f"the on-disk `length` field holds coded interval labels "
                    f"{on_disk} (expected values {list(LENGTH_CODES)}), not "
                    "token counts"
                )
            records = filtered
        if max_per_task:
            records = records[:max_per_task]
        for record in records:
            key = (task, record["ins_id"], record["param_id"], record["length"],
                   scale)
            if key in done:
                continue
            user_text = prompt_builder(record) if prompt_builder else record["prompt"]
            output = _gen(build_chat_prompt(tok, user_text))
            score_dict = score_output(task, record, output, judges)
            row = {
                "task": task,
                "arm": arm,
                "ins_id": record["ins_id"],
                "param_id": record["param_id"],
                "length": record["length"],
                "scale": scale,
                "output": output,
                "score_dict": score_dict,
            }
            if n_train is not None:
                # instance id (ins_id, param_id) is already in the row;
                # the split tag makes decontamination auditable per record.
                row["split"] = "heldout"
                row["split_seed"] = split_seed
            rows.append(row)
            with out_jsonl.open("a") as f:
                f.write(json.dumps(row) + "\n")
    return rows


# -- aggregation -------------------------------------------------------------------


def ars_by_length(rows: list[dict]):
    """Mean rubric total_score per (arm, task, length) — the
    adherence-vs-length curve LIFBench publishes."""
    import pandas as pd

    flat = [
        {
            "arm": r["arm"], "task": r["task"], "length": r["length"],
            "total_score": r["score_dict"]["total_score"],
        }
        for r in rows
    ]
    df = pd.DataFrame(flat)
    return df.groupby(["arm", "task", "length"])["total_score"].mean().reset_index()


def ars_overall(rows: list[dict], vendor_dir: Path | None = None):
    """Eq. 3: task ARS weighted by the task's rubric total (ScoreTools)."""
    df = ars_by_length(rows)
    tools = load_score_tools(vendor_dir)
    per_task = df.groupby(["arm", "task"])["total_score"].mean().reset_index()
    out = {}
    for arm, group in per_task.groupby("arm"):
        tasks = list(group["task"])
        weights = tools.get_total_scores_weights(tasks)
        out[arm] = float((group["total_score"].to_numpy() * weights).sum())
    return out


def stability_ifs(rows: list[dict]) -> dict:
    """Paper Eq. 6, length perspective: IFS_t = std/mean of the per-length
    ARS means (ddof=0); lower = more stable. Returned per (arm, task) and
    per arm (mean over tasks).

    A task observed at fewer than 2 length intervals has no stability
    signal and scores None (never a fake 0.0 = "perfectly stable"). A
    failed-everywhere task (mean 0) stays inf and PROPAGATES into the
    per-arm mean — dropping it would let catastrophic failure improve
    the stability aggregate."""
    df = ars_by_length(rows)
    out: dict = {"per_task": {}, "per_arm": {}}
    for (arm, task), group in df.groupby(["arm", "task"]):
        ys = group["total_score"].to_numpy()
        if len(ys) < 2:
            ifs = None
        else:
            mean = ys.mean()
            ifs = float(ys.std(ddof=0) / mean) if mean > 0 else float("inf")
        out["per_task"][(arm, task)] = ifs
    for arm in {a for a, _ in out["per_task"]}:
        vals = [v for (a, _), v in out["per_task"].items()
                if a == arm and v is not None]
        out["per_arm"][arm] = float(sum(vals) / len(vals)) if vals else None
    return out
