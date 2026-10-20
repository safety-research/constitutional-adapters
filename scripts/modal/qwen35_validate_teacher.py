"""Pre-spawn validation of the qwen35 transfer-lane teacher artifacts —
CPU-only, one container per (model, SI), the EXACT identity check the fit
driver runs before consuming an artifact (TeacherTopK.load + .validate
against the fit config's model/revision/dtype, the pool's content sha, the
SI's resolved max_len, DEFAULT_TOP_K), through the production code itself
(train_two_si_objects._load_pairs + pool_content_sha256, imported — never
reimplemented). A fit whose artifact fails this check would NOT crash: it
falls back to the ONLINE teacher, silently changing the regime — hence the
gate before fits-spawn.

  ~/venvs/modal/bin/modal run scripts/modal/qwen35_validate_teacher.py
"""
import json
from pathlib import Path

import modal

app = modal.App("qwen35-validate-teacher")

try:
    WORKTREE = Path(__file__).resolve().parents[2]
except IndexError:
    WORKTREE = Path("/repo")

_IGNORE = [".git", "**/__pycache__", ".venv", "out/**", "wandb/**",
           ".pytest_cache", "worktrees"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("unixodbc")
    .pip_install_from_requirements(
        str(WORKTREE / "scripts/modal/requirements.txt"),
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_dir(str(WORKTREE), remote_path="/repo", ignore=_IGNORE)
)

vol = modal.Volume.from_name("twosi-fits")

CELLS = [(m, si) for m in ("qwen35_9b", "qwen35_4b")
         for si in ("si_constitution", "si_irrelevant")]


@app.function(image=image, cpu=8, memory=65536, timeout=3600,
              max_containers=8, volumes={"/vol": vol})
def validate(item: tuple) -> dict:
    import importlib.util
    import sys

    import yaml

    model_key, si = item
    sys.path.insert(0, "/repo/src")
    spec = importlib.util.spec_from_file_location(
        "train_two_si_objects", "/repo/scripts/train_two_si_objects.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["train_two_si_objects"] = m
    spec.loader.exec_module(m)

    from sisteer.eval.train_budget import resolve_max_len
    from sisteer.extraction.teacher_topk import DEFAULT_TOP_K, TeacherTopK

    cfg = yaml.safe_load(
        Path(f"/repo/configs/two_si_{model_key}_fits.yaml").read_text())
    pairs_path = f"/vol/pool/{model_key}/pairs_{si}.jsonl"
    art = pairs_path + ".teachertopk.safetensors"
    res: dict = {"model": model_key, "si": si, "path": art}
    if not Path(art).is_file() or not Path(art + ".json").is_file():
        res["status"] = "MISSING"
        return res
    try:
        store = TeacherTopK.load(art)  # includes internal consistency
        pairs = m._load_pairs(pairs_path)
        problems = store.validate(
            model=cfg.get("model"),
            model_revision=cfg.get("model_revision"),
            si_id=si,
            pool_content_sha256=m.pool_content_sha256(pairs),
            max_len=resolve_max_len(cfg, si),
            top_k=DEFAULT_TOP_K,
            dtype=cfg.get("dtype"))
    except Exception as e:  # noqa: BLE001 — report, caller decides
        res["status"] = "LOAD-FAILED"
        res["error"] = f"{type(e).__name__}: {e}"
        return res
    res.update({
        "status": "VALID" if not problems else "MISMATCH",
        "problems": problems,
        "top_k": store.top_k,
        "n_examples": len(store),
        "n_positions": int(store.idx.shape[0]),
        "meta_max_len": store.meta.get("max_len"),
        "meta_model": store.meta.get("model"),
        "meta_revision": store.meta.get("model_revision"),
    })
    return res


@app.local_entrypoint()
def main():
    results = list(validate.map(CELLS))
    print(json.dumps(results, indent=1))
    bad = [r for r in results if r.get("status") != "VALID"]
    print(f"\nTEACHER VALIDATION: {len(results) - len(bad)}/{len(results)} "
          "VALID")
    for r in bad:
        print("NOT-VALID:", json.dumps(r))
    if bad:
        raise SystemExit(1)
