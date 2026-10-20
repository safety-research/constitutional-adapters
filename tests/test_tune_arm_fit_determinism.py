"""tune_arm.py's response-surface fit is BIT-REPRODUCIBLE: the same rows give
the same artefact bytes in every fresh interpreter.

The incident: analyze's fit went through ``torch.linalg.lstsq`` (MKL LAPACK on
CPU), which is not run-to-run reproducible for identical float64 inputs — eight
fresh processes on the same 14 rows gave three or four distinct argmins at a
relative spread of ~5e-15, and neither single-threading nor
``MKL_CBWR=COMPATIBLE`` removed it. The artefact's bytes are an identity (the
confirmation design pins ``fit_sha256`` and the argmin), so a launcher replay
that re-ran analyze failed a confirmation on a rewritten file. The fit now runs
in plain CPython floats (Householder QR, Jacobi eigen/singular values,
partial-pivot elimination) — this module pins the byte identity ACROSS FRESH
INTERPRETERS (the failure mode; an in-process repeat would not have caught it),
the agreement with an independent solver, and the rank-deficiency detection the
one-sided Jacobi route must keep.

CPU-only, no model. Fourteen synthetic rows in the LR-only shape (a constant
clip, two warmup levels, two pooled rounds).
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import tune_arm  # noqa: E402

CLIP = 0.15
LEVELS = (0.05, 0.15)
ARM = "lora_r4_ce"
BOX = {"lr": [3e-5, 6e-3], "max_grad_norm": [CLIP, CLIP]}
_L0 = math.log(1.1e-3)
# fixed pseudo-noise in log y (literal, so the fixture is the same everywhere)
_JITTER = [0.0031, -0.0022, 0.0007, -0.0040, 0.0018, 0.0005, -0.0011,
           -0.0027, 0.0036, 0.0002, -0.0016, 0.0024, -0.0009, 0.0013]


def fourteen_rows() -> list[dict]:
    """Two rounds x seven trials on a log-spaced lr ladder; y = exp(planted
    quadratic in log lr + a warmup main effect + fixed jitter)."""
    rows = []
    lrs = [3e-5 * (6e-3 / 3e-5) ** (k / 13) for k in range(14)]
    for k, lr in enumerate(lrs):
        rnd, idx = (1, k) if k < 7 else (2, k - 7)
        w = LEVELS[k % 2]
        log_y = 0.62 + 0.05 * (math.log(lr) - _L0) ** 2 - 0.004 * (w == 0.15) + _JITTER[k]
        rows.append({"index": idx, "round": rnd, "lr": lr, "warmup_frac": w,
                     "max_grad_norm": CLIP, "best_heldout_loss": math.exp(log_y)})
    return rows


def _fit(rows):
    return tune_arm.fit_quadratic(rows, levels=LEVELS, fixed_axes={"max_grad_norm": CLIP})


def _child_env(seed: str, threads: str) -> dict:
    env = dict(os.environ)
    env.update({"PYTHONHASHSEED": seed, "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads, "PYTHONPATH": str(REPO / "src")})
    return env


_CHILD = (
    "import json, sys\n"
    f"sys.path.insert(0, {str(REPO / 'scripts')!r})\n"
    "import tune_arm\n"
    "rows = json.loads(sys.stdin.read())\n"
    "fit = tune_arm.fit_quadratic(rows, levels=(0.05, 0.15), fixed_axes={'max_grad_norm': 0.15})\n"
    "sys.stdout.write(json.dumps(fit, sort_keys=True))\n"
)


def _fit_in_fresh_interpreter(rows, seed: str, threads: str) -> str:
    out = subprocess.run([sys.executable, "-c", _CHILD], input=json.dumps(rows),
                         capture_output=True, text=True, env=_child_env(seed, threads),
                         check=True, timeout=600)
    return out.stdout


def test_fixture_fits_to_an_interior_positive_definite_argmin():
    fit = _fit(fourteen_rows())
    assert fit["n_rows"] == 14 and fit["design_rank"] == 5 and not fit.get("rank_deficient")
    assert fit["positive_definite"] and fit["fixed_axes"] == {"max_grad_norm": CLIP}
    assert BOX["lr"][0] < fit["argmin"]["lr"] < BOX["lr"][1]
    assert math.isclose(math.log(fit["argmin"]["lr"]), _L0, abs_tol=0.15)
    assert fit["argmin"]["warmup_frac"] == 0.15          # the planted main effect


def test_fit_is_byte_identical_across_fresh_interpreters():
    """THE incident's shape: two interpreters, different hash seeds and
    thread counts, same rows -> identical JSON bytes, and identical to the
    in-process result. Under torch.linalg.lstsq this failed on the real 14
    rows in 8 of 8 attempts to get 8 equal hashes."""
    rows = fourteen_rows()
    a = _fit_in_fresh_interpreter(rows, "1", "1")
    b = _fit_in_fresh_interpreter(rows, "20260910", "4")
    assert a == b
    assert a == json.dumps(_fit(rows), sort_keys=True)
    sel = json.loads(a)
    assert sel["positive_definite"] and sel["argmin"]["max_grad_norm"] == CLIP


def test_fit_repeats_in_process_and_does_not_depend_on_row_object_identity():
    rows = fourteen_rows()
    first = json.dumps(_fit(rows), sort_keys=True)
    again = json.dumps(_fit(json.loads(json.dumps(rows))), sort_keys=True)
    assert first == again


def test_fit_agrees_with_an_independent_solver():
    np = pytest.importorskip("numpy")
    rows = fourteen_rows()
    fit = _fit(rows)
    A = np.array([[1.0, math.log(r["lr"]), math.log(r["lr"]) ** 2,
                   float(r["warmup_frac"] == 0.15),
                   float(r["warmup_frac"] == 0.15) * math.log(r["lr"])] for r in rows])
    y = np.array([math.log(r["best_heldout_loss"]) for r in rows])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    assert fit["names"] == ["1", "lr", "lr*lr", "warmup", "warmup*lr"]
    for ours, ref in zip(fit["coef"], coef):
        assert math.isclose(ours, ref, rel_tol=1e-9, abs_tol=1e-12)
    An = A / np.linalg.norm(A, axis=0)
    s = np.linalg.svd(An, compute_uv=False)
    assert math.isclose(fit["design_cond"], s[0] / s[-1], rel_tol=1e-9)
    # the stationary point of the fitted parabola at the selected level
    c = dict(zip(fit["names"], coef))
    w = 1.0
    L = -(c["lr"] + w * c["warmup*lr"]) / (2 * c["lr*lr"])
    assert math.isclose(fit["argmin"]["lr"], math.exp(L), rel_tol=1e-9)


def test_dense_helpers_agree_with_numpy_on_random_inputs():
    np = pytest.importorskip("numpy")
    rng = random.Random(20260910)
    for K in (1, 2, 3, 4):
        M = [[rng.uniform(-1, 1) for _ in range(K)] for _ in range(K)]
        S = [[M[i][j] + M[j][i] + (3.0 if i == j else 0.0) for j in range(K)] for i in range(K)]
        ours = tune_arm._eigvalsh_jacobi(S)
        ref = np.linalg.eigvalsh(np.array(S))
        assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12) for a, b in zip(ours, ref))
        rhs = [rng.uniform(-1, 1) for _ in range(K)]
        x = tune_arm._solve_dense(S, rhs)
        assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
                   for a, b in zip(x, np.linalg.solve(np.array(S), np.array(rhs))))
    A = [[rng.uniform(-2, 2) for _ in range(5)] for _ in range(14)]
    s_ours = tune_arm._svdvals_jacobi(A)
    s_ref = np.linalg.svd(np.array(A), compute_uv=False)
    assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12) for a, b in zip(s_ours, s_ref))
    y = [rng.uniform(-1, 1) for _ in range(14)]
    x_ours = tune_arm._lstsq_householder(A, y)
    x_ref = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)[0]
    assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(x_ours, x_ref))


def test_rank_deficiency_is_still_detected():
    """The one-sided Jacobi route must keep the reviewed refusal: a pool at
    ONE warmup level zeroes the warmup and warmup*lr columns (rank 3 of 5),
    and an exactly collinear column reads as rank p-1 — never as a tiny but
    'nonzero' singular value that the 1e-10 threshold would pass."""
    rows = fourteen_rows()
    for r in rows:
        r["warmup_frac"] = 0.05
    fit = _fit(rows)
    assert fit.get("rank_deficient") is True and fit["design_rank"] == 3
    A = [[1.0, math.log(r["lr"]), 2.0 * math.log(r["lr"]), math.log(r["lr"]) ** 2] for r in rows]
    rank, cond = tune_arm.check_design(A)
    assert rank == 3                       # the refusal fires on rank, not cond
    assert cond == float("inf") or cond > tune_arm.MAX_DESIGN_COND


def _write_world(out: Path, rows: list[dict]) -> None:
    for r in rows:
        s = {"trial": {k: r[k] for k in ("index", "lr", "warmup_frac", "max_grad_norm")},
             "round": r["round"], "status": "ok", "arm": ARM,
             "model": "NousResearch/Meta-Llama-3.1-8B", "position": "all", "si": "constv2-mix-10m",
             "box": BOX, "warmup_levels": list(LEVELS),
             "best_heldout_loss": r["best_heldout_loss"], "floor_ok": True,
             "heldout_curve": [2.36, r["best_heldout_loss"]]}
        d = out / f"r{r['round']}_t{r['index']:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(s))


def test_analyze_artefact_is_byte_identical_across_fresh_interpreters(tmp_path):
    """End to end through the CLI the launcher drives: `analyze --rounds 1,2`
    twice in fresh interpreters on the same 14 trial directories writes the
    same r1+2_fit.json bytes (so the sha the confirmation design pins is the
    sha a replay would recompute)."""
    _write_world(tmp_path, fourteen_rows())
    art = tmp_path / "r1+2_fit.json"
    shas = []
    for seed, threads in (("7", "1"), ("70000", "4")):
        if art.exists():
            art.unlink()
        subprocess.run([sys.executable, str(REPO / "scripts/tune_arm.py"), "analyze",
                        "--out", str(tmp_path), "--rounds", "1,2"],
                       capture_output=True, text=True, check=True, timeout=600,
                       cwd=str(REPO), env=_child_env(seed, threads))
        shas.append(art.read_bytes())
    assert shas[0] == shas[1]
    a = json.loads(shas[0])
    assert a["selection"]["selected_from"] == "fitted_argmin" and not a["selection"]["on_edge"]
    assert a["fit"]["n_rows"] == 14 and a["fixed_axes"] == {"max_grad_norm": CLIP}
