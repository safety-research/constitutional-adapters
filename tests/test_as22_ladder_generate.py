"""The lambda ladder's CLUSTER half (alphasteer-full22 Phase C):
``scripts/alphasteer_f22/lambda_ladder.py generate``, ``sbatch/as22_common.sh``,
``sbatch/ladder.sbatch``.

No GPU and no network. The pure pieces — geometry, the §C.3a integrity assertions, the rho
statistics, the grid read-back, the resume slicing — are tested directly; the HTTP layer is a stub;
the standup is exercised through ``--dry-run``, which resolves and prints every path, the geometry,
the rung list and the standup argv without launching anything; and the two shell gates are run for
real against THIS checkout, which is exactly the "unpatched ladder" case one of them must refuse.

The load-bearing tests, and what each would have caught:

* ``test_the_wave_shaped_config_is_refused_by_the_gate_standup`` — the reason
  ``_write_standup_config`` exists at all. ``lens_gate_standup.gate_config`` adds the served arms
  to ``class_*_arms``, which ``registry_from_explicit_cells`` refuses next to ``explicit_cells``.
  Passing our serving config straight through would have died at the first standup, on the GPU,
  after the venvs built.
* ``test_integrity_failure_uncertifies_before_any_statistic`` — an instrument that is wrong must
  stop the job, not produce a number.
* ``test_condition_c_is_the_p05_of_rho_last_and_the_row_min_is_flagged_not_thresholded`` — the row
  min is dominated by the attention-sink token and answers a different question from 3(c)'s.
* ``test_unsteered_batches_must_leave_no_entry`` — the `none` and `none__lens0` legs contributing
  entries would silently inflate the denominator of every quantile.
* ``test_ladder_mode_refuses_an_unpatched_tree`` — on W_A the rho record does not exist, so
  condition (c) could not be measured and the run would fall back, silently, to no criterion.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SBATCH = REPO / "scripts" / "alphasteer_f22" / "sbatch"
COMMON_SH = SBATCH / "as22_common.sh"
LADDER_SBATCH = SBATCH / "ladder.sbatch"


def _code(path: Path) -> str:
    """The EXECUTABLE lines of a shell file — comments dropped, ``#SBATCH`` directives kept.

    Every "this script must never do X" assertion below is about what the script DOES. Grepping the
    raw text instead makes the prose that explains a rule trip the test enforcing it (this file's
    first run: four failures, all of them a comment saying "never write $WAVE_ROOT/STOP" or "NOT
    world-readable exp/logs"), which would push the next author to delete the explanation."""
    keep = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("#SBATCH") or not s.startswith("#"):
            keep.append(line.split("  #")[0] if "  #" in line else line)
    return "\n".join(keep)


def _load_ladder():
    spec = importlib.util.spec_from_file_location(
        "_as22_lambda_ladder", REPO / "scripts" / "alphasteer_f22" / "lambda_ladder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ll = _load_ladder()


# --------------------------------------------------------------------------------------------- #
# geometry                                                                                        #
# --------------------------------------------------------------------------------------------- #

class TestGeometry:
    # want_s = BUDGET_BY_MODEL[key].seqs_at(4864), written out so a silent budget change is caught here:
    #   gemma4e4b        589,824 // 4864 = 121  (swept 2026-09-04)
    #   qwen35_4b        720,896 // 4864 = 148  (swept; main's PROVISIONAL halving to 360,448 was REMOVED
    #                                            from main at 06214c6c on MEASURED evidence: six
    #                                            standups at 148 x 4864 (job 90773, a compute node, tree 1b7099bf),
    #                                            engine-attested KV 993,378 tokens / 204.23x concurrency, and
    #                                            no OOM or allocator refusal on any of the six legs)
    #   glm46vflash      786,432 // 4864 = 161  (swept)
    #   nemotron3nano4b  786,432 // 4864 = 161  (swept; hybrid cap 256 and the 4608 ceiling do not bind at 4864)
    @pytest.mark.parametrize("key,want_s,hybrid", [("gemma4e4b", 121, False),
                                                   ("qwen35_4b", 152, True),
                                                   ("glm46vflash", 161, False),
                                                   ("nemotron3nano4b", 161, True)])
    def test_s_is_computed_from_the_model_budget_never_hardcoded(self, key, want_s, hybrid):
        from sisteer.capeval.serving import geometry as gm
        g = ll.resolve_geometry(key)
        assert g["max_model_len"] == 4864 and g["eval"] == "families_served:short"
        assert g["max_num_seqs"] == want_s == gm.BUDGET_BY_MODEL[key].seqs_at(4864)
        assert g["hybrid"] is hybrid
        assert g["batched_tokens"] == want_s * 4864
        assert f"seqs_at(4864)" in g["seqs_source"]

    def test_qwen_is_hybrid_capped_below_the_plain_quotient(self):
        """The hybrid branch is real and qwen35_4b exercises it: for a hybrid-recurrent family budget // L
        is only an UPPER bound (per-slot recurrent state), so seqs_at caps S at HYBRID_SEQS_CAP wherever the
        plain quotient exceeds it. The window is DERIVED from the budget in force rather than pinned: under
        the swept 720,896 the cap bound at 1536 (469 -> 256); under main's provisional 360,448 (31b869f7)
        the quotient at 1536 is 234 and the cap does not bind there, so the largest 256-multiple whose
        quotient clears the cap is the window (1024: 352 -> 256). At the ladder's own 4864 window the cap
        does not bind under either budget (148, now 74) — asserted too, so a cap that started binding
        the ladder's geometry would be noticed."""
        from sisteer.capeval.serving import geometry as g
        b = g.BUDGET_BY_MODEL["qwen35_4b"]
        assert b.hybrid and b.seqs_cap == g.HYBRID_SEQS_CAP
        L = max(l for l in range(256, 8192, 256) if b.batched_token_budget // l > g.HYBRID_SEQS_CAP)
        assert b.window_ceiling(L) is None, "pick a window without a measured per-window ceiling"
        assert b.seqs_at(L) == g.HYBRID_SEQS_CAP < b.batched_token_budget // L
        assert b.seqs_at(4864) == b.batched_token_budget // 4864 < g.HYBRID_SEQS_CAP

    def test_a_hand_picked_width_is_refused(self):
        """lens_gate_standup takes S explicitly and does NOT enforce the no-chunking identity, so
        this is the only place it is held."""
        with pytest.raises(ll.LadderError, match="no-chunking identity"):
            ll.resolve_geometry("gemma4e4b", max_num_seqs=64)

    def test_an_unmeasured_model_key_refuses_rather_than_defaulting(self):
        with pytest.raises(ll.LadderError, match="no measured batched-token budget"):
            ll.resolve_geometry("not_a_model")


# --------------------------------------------------------------------------------------------- #
# the §C.3a integrity assertions                                                                  #
# --------------------------------------------------------------------------------------------- #

def _entry(req, hook_layer, **over):
    e = {"req_id": req, "layer": hook_layer, "requested": "alphasteer",
         "applied_cell": "alphasteer", "applied_effective_scale": 1.0,
         "path": "per_slot", "phase": "prefill", "n_rows": 12,
         "dtype": "torch.bfloat16", "slice": "harmful",
         "rho_last": 5e-3, "rho_min": 1e-3, "rho_median": 4e-3, "rho_max": 9e-3,
         "add_norm_last": 0.9, "h_norm_last": 190.0}
    e.update(over)
    return e


def _snap(**over):
    s = {"enabled": True, "env": "SISTEER_LENS_RHO_EMIT", "n_entries": 4,
         "n_entries_dropped": 0, "max_entries": 65536, "n_record_errors": 0,
         "first_record_error": None, "note": "...", "slice": "harmful", "steered": True}
    s.update(over)
    return s


def _clean(n_req=3, hook_layers=(6, 7)):
    entries = [_entry(f"r{k}", l) for k in range(n_req) for l in hook_layers]
    return entries, [_snap(n_entries=len(entries))]


class TestRhoIntegrity:
    def test_a_complete_record_passes_and_says_what_it_checked(self):
        entries, snaps = _clean()
        out = ll.check_rho_integrity(entries, snaps, n_steered_requests=3, fit_layers=[7, 8])
        assert out["pass"] is True and out["problems"] == []
        assert out["checks"]["distinct_req_layer_pairs"] == 6
        assert out["checks"]["expected_req_layer_pairs"] == 6
        assert out["checks"]["fit_layers_seen"] == [7, 8]
        assert out["hook_to_fit_layer"] == "+1 (site=pre)"

    def test_a_disabled_record_refuses_and_names_the_missing_flag(self):
        entries, _ = _clean()
        out = ll.check_rho_integrity(entries, [_snap(enabled=False)],
                                     n_steered_requests=3, fit_layers=[7, 8])
        assert out["pass"] is False
        assert any("SISTEER_LENS_RHO_EMIT" in p for p in out["problems"])

    @pytest.mark.parametrize("field,bad", [("applied_cell", "none__lens0"),
                                           ("applied_effective_scale", 0.0),
                                           ("path", "grouped"),
                                           ("phase", "decode"),
                                           ("dtype", "torch.float32")])
    def test_every_per_entry_invariant_refuses(self, field, bad):
        entries, snaps = _clean()
        entries[0] = _entry("r0", 6, **{field: bad})
        out = ll.check_rho_integrity(entries, snaps, n_steered_requests=3, fit_layers=[7, 8])
        assert out["pass"] is False
        assert any(field in p for p in out["problems"])

    def test_dropped_entries_and_record_errors_refuse(self):
        entries, _ = _clean()
        out = ll.check_rho_integrity(entries, [_snap(n_entries_dropped=2, n_record_errors=1,
                                                     first_record_error="RuntimeError: x")],
                                     n_steered_requests=3, fit_layers=[7, 8])
        assert out["pass"] is False
        assert any("DROPPED" in p for p in out["problems"])
        assert any("RuntimeError: x" in p for p in out["problems"])

    def test_an_incomplete_record_refuses_on_the_pair_count(self):
        entries, snaps = _clean()
        out = ll.check_rho_integrity(entries[:-1], snaps, n_steered_requests=3,
                                     fit_layers=[7, 8])
        assert out["pass"] is False
        assert any("does not cover the traffic" in p for p in out["problems"])

    def test_the_fit_layer_set_must_equal_the_objects(self):
        """fit_layer = hook_layer + 1 (site='pre'); a record whose layers are not the object's
        means the wrong blocks were hooked."""
        entries, snaps = _clean(hook_layers=(6, 9))
        out = ll.check_rho_integrity(entries, snaps, n_steered_requests=3, fit_layers=[7, 8])
        assert out["pass"] is False
        assert any("site='pre'" in p for p in out["problems"])

    def test_unsteered_batches_must_leave_no_entry(self):
        """The `none` (unhooked) and `none__lens0` (scale 0) legs contribute nothing. A snapshot
        of an unsteered batch that returned entries means the twin is not path-matched or the
        anchor is being hooked, and every quantile below would be computed over the wrong rows."""
        entries, snaps = _clean()
        snaps.append(_snap(slice="benign", steered=False, n_entries=7))
        out = ll.check_rho_integrity(entries, snaps, n_steered_requests=3, fit_layers=[7, 8])
        assert out["pass"] is False
        assert any("must contribute none" in p for p in out["problems"])

    def test_the_expected_dtype_is_derived_from_the_config_not_hardcoded(self):
        entries, snaps = _clean()
        for e in entries:
            e["dtype"] = "torch.float32"
        out = ll.check_rho_integrity(entries, snaps, n_steered_requests=3, fit_layers=[7, 8],
                                     expect_dtype="torch.float32")
        assert out["pass"] is True


# --------------------------------------------------------------------------------------------- #
# the statistics and condition (c)                                                                #
# --------------------------------------------------------------------------------------------- #

RHO_FIT = {7: {"rho_fp32": {"p05": 2.0e-3, "p50": 4e-3, "p95": 6e-3},
               "rho_model_dtype": {"p05": 2.0e-3, "p50": 4e-3, "p95": 6e-3}},
           8: {"rho_fp32": {"p05": 3.0e-3, "p50": 5e-3, "p95": 7e-3},
               "rho_model_dtype": {"p05": 3.0e-3, "p50": 5e-3, "p95": 7e-3}}}


def _rows(hook_layer, slice_name, rho_lasts, rho_mins=None):
    rho_mins = rho_mins if rho_mins is not None else [v / 5 for v in rho_lasts]
    return [_entry(f"{slice_name}{k}", hook_layer, slice=slice_name,
                   rho_last=v, rho_min=m, rho_median=v * 0.8, rho_max=v * 1.4)
            for k, (v, m) in enumerate(zip(rho_lasts, rho_mins))]


class TestRhoStatistics:
    def test_condition_c_is_the_p05_of_rho_last_min_over_layers(self):
        entries = (_rows(6, "harmful", [4e-3] * 10 + [9e-3] * 10)
                   + _rows(7, "harmful", [6e-3] * 20)
                   + _rows(6, "benign", [1e-5] * 20))
        out = ll.rho_statistics(entries, fit_layers=[7, 8], rho_fit=RHO_FIT, lam=1.0)
        assert out["cond_c_pass"] is True
        assert out["served_rho_argmin_layer"] == 7
        assert out["served_rho_min_p05_over_layers"] == pytest.approx(4e-3, rel=1e-6)
        assert out["served_rho_by_layer"]["7"]["hook_layer"] == 6
        assert out["served_rho_by_layer"]["7"]["benign"]["rho_last"]["p50"] == pytest.approx(1e-5)

    def test_one_weak_layer_fails_c_even_when_every_other_layer_passes(self):
        entries = (_rows(6, "harmful", [9e-3] * 20)
                   + _rows(7, "harmful", [1.4e-3] * 20))
        out = ll.rho_statistics(entries, fit_layers=[7, 8], rho_fit=RHO_FIT, lam=1.0)
        assert out["cond_c_pass"] is False
        assert out["served_rho_argmin_layer"] == 8

    def test_condition_c_is_the_p05_not_the_median(self):
        """A layer at p50 6e-3 with a fifth of its traffic near zero is inert on that fifth; p50
        would pass it."""
        entries = (_rows(6, "harmful", [1e-5] * 4 + [6e-3] * 16)
                   + _rows(7, "harmful", [6e-3] * 20))
        out = ll.rho_statistics(entries, fit_layers=[7, 8], rho_fit=RHO_FIT, lam=1.0)
        p = out["served_rho_by_layer"]["7"]["harmful"]["rho_last"]
        assert p["p50"] == pytest.approx(6e-3) and p["p05"] < ll.COND_C_FLOOR
        assert out["cond_c_pass"] is False

    def test_the_row_min_is_flagged_never_thresholded(self):
        """The within-request row min is dominated by the highest-norm row (the attention
        sink) and answers a different question, so a layer passing on rho_last with a low row min
        PASSES and is FLAGGED."""
        entries = (_rows(6, "harmful", [5e-3] * 20, rho_mins=[1e-4] * 20)
                   + _rows(7, "harmful", [5e-3] * 20, rho_mins=[5e-3] * 20))
        out = ll.rho_statistics(entries, fit_layers=[7, 8], rho_fit=RHO_FIT, lam=1.0)
        assert out["cond_c_pass"] is True
        assert out["served_rho_by_layer"]["7"]["row_min_below_floor_flag"] is True
        assert out["served_rho_by_layer"]["8"]["row_min_below_floor_flag"] is False

    def test_the_fit_side_reading_scales_linearly_in_lambda_and_never_decides(self):
        """rho is linear in lambda, so rho(lambda) = lambda x rho(1). It rides beside the served
        value so the composition effect is a measured ratio (§9.6) — it is not an input to (c)."""
        entries = _rows(6, "harmful", [4e-3] * 20) + _rows(7, "harmful", [6e-3] * 20)
        out = ll.rho_statistics(entries, fit_layers=[7, 8], rho_fit=RHO_FIT, lam=2.0)
        b7 = out["served_rho_by_layer"]["7"]
        assert b7["rho_fit_model_dtype"]["p05"] == pytest.approx(4.0e-3)   # 2.0 x 2.0e-3
        assert b7["served_over_fit_p05"] == pytest.approx(1.0)
        # a fit side that passes everywhere cannot rescue a served layer that does not
        weak = _rows(6, "harmful", [1e-3] * 20) + _rows(7, "harmful", [6e-3] * 20)
        assert ll.rho_statistics(weak, fit_layers=[7, 8], rho_fit=RHO_FIT,
                                 lam=2.0)["cond_c_pass"] is False

    def test_a_layer_with_no_harmful_reading_refuses(self):
        """(c) is a claim about EVERY steered layer, so a missing one cannot be a pass."""
        entries = _rows(6, "harmful", [5e-3] * 20)
        with pytest.raises(ll.LadderError, match="condition \\(c\\) is a claim about EVERY"):
            ll.rho_statistics(entries, fit_layers=[7, 8], rho_fit=RHO_FIT, lam=1.0)

    def test_null_ratios_are_counted_not_silently_dropped(self):
        entries = _rows(6, "harmful", [5e-3] * 18) + _rows(7, "harmful", [5e-3] * 18)
        entries[0]["rho_last"] = None      # a zero-norm row reads null in the probe's record
        out = ll.rho_statistics(entries, fit_layers=[7, 8], rho_fit=RHO_FIT, lam=1.0)
        q = out["served_rho_by_layer"]["7"]["harmful"]["rho_last"]
        assert q["n"] == 17 and q["n_null"] == 1

    def test_quantile_is_the_linear_interpolation_convention(self):
        assert ll.quantile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
        assert ll.quantile([], 0.5) is None
        assert ll.quantiles([])["p05"] is None, "an empty read is null, never a measured 0.0"


# --------------------------------------------------------------------------------------------- #
# the rungs: the grid, the objects                                                                #
# --------------------------------------------------------------------------------------------- #

def _make_object(path: Path, base_scale: float, layers=(7, 8), d=4, jitter=0.0):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    Ms = {str(l): torch.randn(d, d, generator=g) + jitter for l in layers}
    torch.save({"kind": "alphasteer", "Ms": Ms, "layers": list(layers),
                "base_scale": float(base_scale), "site": "pre",
                "steer_positions": "prefill"}, path)


def _ladder_tree(tmp: Path, target="gemma4e4b", view="it", lams=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0)):
    for i, lam in enumerate(lams, start=1):
        _make_object(tmp / target / "ladder" / f"lam{i}" / "cells_view" / view
                     / "alphasteer" / "object.pt", lam, jitter=0.0001 * i)
    return tmp


class TestRungObjects:
    def test_lambdas_are_read_back_from_the_baked_objects(self, tmp_path):
        _ladder_tree(tmp_path)
        got = ll.read_rung_lambdas(tmp_path, "gemma4e4b", "it", [1, 2, 3, 4, 5, 6])
        assert [got[i]["lambda"] for i in range(1, 7)] == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        assert len({got[i]["object_sha256"] for i in got}) == 6

    def test_a_non_geometric_grid_refuses(self):
        """Every fit-side reading scales linearly in lambda, so a grid that is not the recorded one
        misreports every rung, silently."""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        _ladder_tree(tmp, lams=(0.25, 0.5, 1.0, 2.0, 4.0, 7.0))
        with pytest.raises(ll.LadderError, match="geometric grid"):
            ll.read_rung_lambdas(tmp, "gemma4e4b", "it", [1, 2, 3, 4, 5, 6])

    def test_identical_rung_objects_refuse(self, tmp_path):
        """Six rungs must be six distinct objects, hence six distinct intervention hashes — that
        they are served side by side and compared IS the ladder."""
        for i in (1, 2):
            _make_object(tmp_path / "gemma4e4b" / "ladder" / f"lam{i}" / "cells_view" / "it"
                         / "alphasteer" / "object.pt", 0.25)
        with pytest.raises(ll.LadderError, match="same sha256"):
            ll.read_rung_lambdas(tmp_path, "gemma4e4b", "it", [1, 2])

    def test_a_missing_rung_object_refuses_by_name(self, tmp_path):
        with pytest.raises(ll.LadderError, match="no baked object"):
            ll.read_rung_lambdas(tmp_path, "gemma4e4b", "it", [1])


# --------------------------------------------------------------------------------------------- #
# the standup config — the shape finding                                                          #
# --------------------------------------------------------------------------------------------- #

def _some_serving_config():
    """Ours if the config-generating half has landed, else the wave's — the registry-shape question
    is identical for both, and both are ``explicit_cells``-shaped."""
    for rel in ("configs/alphasteer_f22_serve_gemma4e4b.yaml",
                "configs/alphasteer_f22_gemma4e4b_it.yaml",
                "configs/alphasteer_f22_gemma4e4b.yaml",
                "configs/sdfbase_full22_gemma4e4b.yaml"):
        if (REPO / rel).is_file():
            return REPO / rel
    pytest.skip("no explicit_cells serving config on this tree")


def _gate_standup():
    spec = importlib.util.spec_from_file_location(
        "_as22_lens_gate_standup", REPO / "scripts" / "lens_gate_standup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestStandupConfig:
    def test_the_wave_shaped_config_is_refused_by_the_gate_standup(self, tmp_path):
        """WHY ``_write_standup_config`` EXISTS. ``gate_config`` adds the served cells' arms to
        ``class_a_arms`` / ``class_b_arms``; ``registry_from_explicit_cells`` refuses those keys
        next to ``explicit_cells``. Handing our serving config straight to the standup would have
        died at the first rung, on the GPU, after the venvs built."""
        import yaml
        cfg = yaml.safe_load(_some_serving_config().read_text())
        assert "explicit_cells" in cfg, "the fixture must be the wave-shaped config"
        ddir = tmp_path / "cells_view" / "it"
        _make_object(ddir / "alphasteer" / "object.pt", 0.25, d=2560)
        with pytest.raises(ValueError, match="mutually exclusive"):
            _gate_standup().enumerate_gate_cells(
                repo=REPO, cfg=cfg, cells_tt=["alphasteer"], defenses_dir=ddir,
                manifest="configs/capeval/manifests/petri_full22_gemma4e4b.yaml",
                chat_template_hash="ab" * 8, out=tmp_path)

    def test_the_derived_twin_enumerates_exactly_the_trio(self, tmp_path):
        """none + alphasteer + the DERIVED path-matched none__lens0, no absent cells, and the
        manifest's launch flags carried through (the text-only mm pin rides here for qwen/glm)."""
        import yaml
        cfg = yaml.safe_load(_some_serving_config().read_text())
        ddir = tmp_path / "cells_view" / "it"
        _make_object(ddir / "alphasteer" / "object.pt", 0.25, d=2560)
        derived = ll._write_standup_config(cfg, dest=tmp_path / "standup.yaml",
                                           defenses_dir=ddir)
        dcfg = yaml.safe_load(derived.read_text())
        assert "explicit_cells" not in dcfg
        assert dcfg["class_b_arms"] == ["alphasteer"] and dcfg["class_a_arms"] == []
        assert dcfg["defenses_dir"] == str(ddir)
        # everything else is the serving config's data, unchanged
        for k in ("model", "model_revision", "dtype"):
            assert dcfg[k] == cfg[k]
        spec, _src, cells, _p, _added = _gate_standup().enumerate_gate_cells(
            repo=REPO, cfg=dcfg, cells_tt=["alphasteer"], defenses_dir=ddir,
            manifest="configs/capeval/manifests/petri_full22_gemma4e4b.yaml",
            chat_template_hash="ab" * 8, out=tmp_path)
        served = {c.cell_id: c.virtual_model_name for c in cells.all_served()}
        assert served == {"none": "petri__base", "alphasteer": "petri__alphasteer",
                          "none__lens0": "petri__none__lens0"}
        assert cells.absent == ()
        assert spec.target_model == cfg["model"]
        assert "--enforce-eager" in spec.serving_launch_flags

    def test_the_standup_config_is_written_into_a_serve_dir_that_does_not_exist_yet(self, tmp_path):
        """Job 84562 (2026-09-09) died on the first rung's first write: rung_dir existed (the bake made
        cells_view/ under it) but serve/ did not — Standup.__enter__ creates it, and the config is
        written BEFORE Standup runs. The writer now creates its parent, and run_rung creates serve/."""
        import yaml
        cfg = yaml.safe_load(_some_serving_config().read_text())
        rung_dir = tmp_path / "gemma4e4b" / "ladder" / "lam1"
        (rung_dir / "cells_view" / "it" / "alphasteer").mkdir(parents=True)   # exactly what the bake leaves
        assert not (rung_dir / "serve").exists()
        dest = ll._write_standup_config(cfg, dest=rung_dir / "serve" / "ladder_standup_config.yaml",
                                        defenses_dir=rung_dir / "cells_view" / "it")
        assert dest.is_file() and yaml.safe_load(dest.read_text())["class_b_arms"] == ["alphasteer"]
        src = (REPO / "scripts" / "alphasteer_f22" / "lambda_ladder.py").read_text()
        head = src[src.index("def run_rung("):src.index("standup_cfg = _write_standup_config(")]
        assert "serve.mkdir(parents=True, exist_ok=True)" in head, "run_rung must create serve/ before the config write"

    def test_alphasteer_is_a_bare_class_b_id_and_a_per_rung_name_is_refused(self):
        """Mechanised: six rungs cannot be six cell ids. ``alphasteer__lam1`` falls
        through to the SI/method grammar and refuses, which is why a rung is the CANONICAL cell id
        served from a DIFFERENT bank."""
        from sisteer.capeval.petri import defense_cells as dc
        from sisteer.capeval.petri import registry_adapter as ra
        assert "alphasteer" in dc.CLASS_B_IDS
        assert ra.harness_cell_id("alphasteer") == "alphasteer"
        with pytest.raises(Exception):
            dc.parse_cell_id("alphasteer__lam1")
        src = (REPO / "scripts" / "alphasteer_f22" / "lambda_ladder.py").read_text()
        assert "__lam" not in src.replace('f"lam{i}"', "").replace("lam{i}", "")


# --------------------------------------------------------------------------------------------- #
# the standup argv and the endpoint assertions                                                    #
# --------------------------------------------------------------------------------------------- #

class TestStandupArgv:
    def test_the_argv_names_the_gate_standup_the_cell_and_the_geometry(self, tmp_path):
        geom = ll.resolve_geometry("gemma4e4b")
        argv = ll.standup_argv(
            serving_python="/home/u/venvs/vllm/bin/python", repo=Path("/repo"),
            ladder_sha="a" * 40, config=tmp_path / "standup.yaml",
            defenses_dir=tmp_path / "cells_view/it", out=tmp_path / "serve", geom=geom,
            manifest="configs/capeval/manifests/alphasteer_f22_gemma4e4b_it.yaml",
            gpu_mem_util="0.9", max_logprobs=64, ready_timeout_s=1800)
        assert argv[1].endswith("scripts/lens_gate_standup.py")
        pairs = dict(zip(argv[2::2], argv[3::2]))
        assert pairs["--cell"] == "alphasteer"
        assert pairs["--expect-sha"] == "a" * 40
        assert pairs["--max-model-len"] == "4864" and pairs["--max-num-seqs"] == "121"
        # 20 top-logprobs are read per liveness probe, so the server must allow at least that
        assert int(pairs["--max-logprobs"]) >= 20
        assert pairs["--stop-file"].endswith("/serve/STOP")

    def test_the_argv_flags_are_the_ones_the_standup_accepts(self, tmp_path):
        """Anti-drift against ``lens_gate_standup``'s own argparse."""
        text = (REPO / "scripts" / "lens_gate_standup.py").read_text()
        geom = ll.resolve_geometry("gemma4e4b")
        argv = ll.standup_argv(
            serving_python="py", repo=Path("/repo"), ladder_sha="a" * 40,
            config=tmp_path / "c.yaml", defenses_dir=tmp_path, out=tmp_path, geom=geom,
            manifest="m.yaml", gpu_mem_util="0.9", max_logprobs=64, ready_timeout_s=1800)
        for flag in [a for a in argv if a.startswith("--")]:
            assert f'ap.add_argument("{flag}"' in text, f"{flag} is not a standup flag"


def _endpoint(**over):
    ep = {"lens_reduction_mode": "stacked", "max_model_len": 4864, "max_num_seqs": 121,
          "prefix_caching": False, "cells": ["alphasteer"], "proxy_url": "http://x/v1",
          "server_base_url": "http://x", "routing_table": "/t.json"}
    ep.update(over)
    return ep


def _attestation(serve: Path, *, s=121, L=4864, mamba="auto", n_workers=1):
    reports = [{"max_num_seqs": s, "max_num_batched_tokens": s * L,
                "mamba_ssm_cache_dtype": mamba} for _ in range(n_workers)]
    (serve / "steering_attestation.json").write_text(json.dumps(
        {"scheduler_geometry": {"expected": {"max_num_seqs": s, "max_model_len": L},
                                "worker_reports": reports}}))


class TestEndpointAssertions:
    def test_a_correct_standup_passes(self, tmp_path):
        _attestation(tmp_path)
        ll._assert_endpoint(_endpoint(), geom=ll.resolve_geometry("gemma4e4b"), serve=tmp_path)

    def test_a_per_row_standup_is_refused_by_the_driver(self, tmp_path):
        """lens_gate_standup RECORDS the attested reduction mode but does not refuse on it, where
        the wave's f22_standup passes --require-attested-mode stacked. The driver holds that
        line instead."""
        _attestation(tmp_path)
        with pytest.raises(ll.LadderError, match="lens_reduction_mode"):
            ll._assert_endpoint(_endpoint(lens_reduction_mode="per_row"),
                                geom=ll.resolve_geometry("gemma4e4b"), serve=tmp_path)

    def test_prefix_caching_on_is_refused(self, tmp_path):
        _attestation(tmp_path)
        with pytest.raises(ll.LadderError, match="PREFILL observation"):
            ll._assert_endpoint(_endpoint(prefix_caching=True),
                                geom=ll.resolve_geometry("gemma4e4b"), serve=tmp_path)

    def test_a_clamped_width_is_caught_by_the_attestation_not_by_our_request(self, tmp_path):
        _attestation(tmp_path, s=64)
        with pytest.raises(ll.LadderError, match="max_num_seqs"):
            ll._assert_endpoint(_endpoint(), geom=ll.resolve_geometry("gemma4e4b"),
                                serve=tmp_path)

    def test_a_missing_attestation_refuses(self, tmp_path):
        with pytest.raises(ll.LadderError, match="unattested"):
            ll._assert_endpoint(_endpoint(), geom=ll.resolve_geometry("gemma4e4b"),
                                serve=tmp_path)

    def test_a_hybrid_target_without_the_mamba_cache_surface_refuses(self, tmp_path):
        _attestation(tmp_path, s=152, mamba=None)
        with pytest.raises(ll.LadderError, match="mamba_ssm_cache_dtype"):
            ll._assert_endpoint(_endpoint(max_num_seqs=152),
                                geom=ll.resolve_geometry("qwen35_4b"), serve=tmp_path)


# --------------------------------------------------------------------------------------------- #
# the rho RPC envelope                                                                            #
# --------------------------------------------------------------------------------------------- #

class TestRhoRpc:
    def test_the_results_envelope_is_unwrapped(self, monkeypatch):
        rec = {"enabled": True, "entries": [_entry("r0", 6)], "n_entries": 1}
        monkeypatch.setattr(ll, "http_json", lambda *a, **k: {"results": [rec]})
        assert ll.rho_snapshot("http://x", "tok")["n_entries"] == 1

    def test_two_workers_refuse_rather_than_being_merged(self, monkeypatch):
        """The entries carry no worker index of their own, so concatenating two workers' records
        would make the (req_id, layer) completeness count meaningless."""
        rec = {"enabled": True, "entries": []}
        monkeypatch.setattr(ll, "http_json", lambda *a, **k: {"results": [rec, rec]})
        with pytest.raises(ll.LadderError, match="exactly one worker"):
            ll.rho_snapshot("http://x", "tok")

    def test_the_reset_flag_and_the_bearer_token_are_sent(self, monkeypatch):
        seen = {}

        def fake(url, payload=None, *, headers=None, timeout=0, what=""):
            seen.update({"url": url, "payload": payload, "headers": headers})
            return {"results": [{"enabled": True, "entries": []}]}

        monkeypatch.setattr(ll, "http_json", fake)
        ll.rho_snapshot("http://host:9/", "sekret", reset=True)
        assert seen["url"] == "http://host:9/collective_rpc"
        assert seen["payload"] == {"method": "sisteer_rho_record", "args": [True]}
        assert seen["headers"]["Authorization"].startswith("Bearer ")


# --------------------------------------------------------------------------------------------- #
# batching, tagging, resume                                                                       #
# --------------------------------------------------------------------------------------------- #

def _items(n, side="b"):
    return [{"prompt_id": ll.prompt_id(side, k, f"p{k}"), "list_index": k, "held_pos": k,
             "prompt": f"p{k}"} for k in range(n)]


class TestTrafficAndResume:
    def test_batches_are_slice_homogeneous_and_sized_to_the_server_width(self):
        got = list(ll._pending_batches(_items(10), set(), leg="alphasteer",
                                       slice_name="benign", width=4))
        assert [len(b) for b in got] == [4, 4, 2]

    def test_resume_skips_only_what_is_already_on_disk_by_prompt_id(self):
        items = _items(5)
        done = {("alphasteer", "benign", items[1]["prompt_id"]),
                ("alphasteer", "benign", items[3]["prompt_id"]),
                # the SAME prompt on another leg is NOT done for this leg
                ("none", "benign", items[0]["prompt_id"])}
        got = list(ll._pending_batches(items, done, leg="alphasteer", slice_name="benign",
                                       width=10))
        assert [i["held_pos"] for i in got[0]] == [0, 2, 4]

    def test_a_batch_fires_concurrently_and_returns_one_row_per_item(self):
        """The wave's measured lesson: a driver that trickles leaves the server at 13% KV.

        A BARRIER, not a peak-counter: with an instant stub, a serial driver and a concurrent one
        both show a peak of 1, so the timing assertion would pass either way. Eight requests must
        be in flight together or the barrier times out and the batch raises."""
        import threading
        barrier = threading.Barrier(8, timeout=10)

        def chat(payload):
            barrier.wait()
            return {"id": f"chatcmpl-{payload['messages'][-1]['content']}",
                    "petri_intervention_hash": "h",
                    "choices": [{"message": {"content": "out"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 3}}

        rows = ll.run_batch(chat, items=_items(16), leg="alphasteer", vmodel="petri__alphasteer",
                            slice_name="benign", max_tokens=512, batch_index=2, rung_i=1,
                            lam=0.25, concurrency=8)
        assert len(rows) == 16
        assert {r["batch_index"] for r in rows} == {2}
        assert rows[0]["response_id"].startswith("chatcmpl-")
        assert rows[0]["intervention_hash"] == "h"
        assert rows[0]["response"] == "out" and rows[0]["n_completion_tokens"] == 3

    def test_the_generation_payload_is_greedy_and_pinned(self):
        p = ll._gen_payload("petri__alphasteer", "x", ll.BENIGN_MAX_TOKENS)
        assert p["temperature"] == 0.0 and p["seed"] == 0 and p["max_tokens"] == 512
        assert ll._gen_payload("v", "x", ll.HARMFUL_MAX_TOKENS)["max_tokens"] == 1

    def test_entries_are_tagged_by_the_batchs_slice_and_the_reset_is_per_batch(
            self, tmp_path, monkeypatch):
        """ATTRIBUTION IS BY CONSTRUCTION. Batches are slice-homogeneous and the record is
        snapshotted with reset=True after every batch, so a batch's entries all belong to one
        slice even where a req_id cannot be matched to a response id — which is why the match rate
        is recorded BESIDE the tag and never used as the tag."""
        calls = []

        def fake(url, payload=None, **k):
            calls.append(payload)
            return {"results": [{"enabled": True, "n_entries": 2, "n_entries_dropped": 0,
                                 "n_record_errors": 0, "max_entries": 65536,
                                 "first_record_error": None, "env": "SISTEER_LENS_RHO_EMIT",
                                 "note": "n",
                                 "entries": [_entry("unmatchable", 6), _entry("known", 7)]}]}

        monkeypatch.setattr(ll, "http_json", fake)
        ep = {"server_base_url": "http://x"}
        snap = ll._snapshot_and_record(
            serve=tmp_path, endpoint=ep, api_key="t", slice_name="harmful", batch_index=3,
            rung_i=2, lam=0.5, steered=True, response_ids=["known"],
            entries_path=tmp_path / "rho_entries.jsonl",
            snapshots_path=tmp_path / "rho_snapshots.jsonl")
        assert calls[0]["args"] == [True], "reset=True after EVERY batch"
        rows = ll.read_jsonl(tmp_path / "rho_entries.jsonl")
        assert {r["slice"] for r in rows} == {"harmful"}
        assert {r["batch_index"] for r in rows} == {3}
        assert {r["rung_i"] for r in rows} == {2}
        assert [r["fit_layer"] for r in rows] == [7, 8]      # hook + 1
        assert {r["worker_index"] for r in rows} == {0}
        # the cross-check: one of two matched, recorded, decides nothing
        assert snap["n_req_id_matched"] == 1 and snap["n_requests"] == 1
        assert [r["req_id_matched_a_response"] for r in rows] == [False, True]
        assert ll.read_jsonl(tmp_path / "rho_snapshots.jsonl")[0]["slice"] == "harmful"

    def test_the_none_transcript_sha_is_order_independent(self):
        a = [{"prompt_id": "b0000-aa", "response": "x"},
             {"prompt_id": "b0001-bb", "response": "y"}]
        assert ll.none_transcript_sha256(a) == ll.none_transcript_sha256(list(reversed(a)))
        b = [dict(a[0]), {"prompt_id": "b0001-bb", "response": "z"}]
        assert ll.none_transcript_sha256(a) != ll.none_transcript_sha256(b)

    def test_prompt_id_is_the_contracts_shape(self):
        pid = ll.prompt_id("b", 7, "hello")
        assert pid.startswith("b0007-") and len(pid.split("-")[1]) == 8


# --------------------------------------------------------------------------------------------- #
# input resolution                                                                                #
# --------------------------------------------------------------------------------------------- #

class TestInputResolution:
    def test_shared_is_derived_from_the_object_root_at_the_right_depth(self):
        """A REGRESSION. This was ``root.parent.parent``, one level short, which yields a real
        directory (``$SHARED/exp``) rather than an error — so the cluster-A->cluster B re-rooting
        below would simply have failed to find a list it should have found, on the cluster only."""
        assert ll.shared_root_of(Path("/workspace/exampleuser/exp/objects/alphasteer_f22")) \
            == Path("/workspace/exampleuser")
        assert ll.shared_root_of(Path("/shared/exampleuser/exp/objects/alphasteer_f22")) \
            == Path("/shared/exampleuser")

    def test_an_unexpected_root_layout_refuses_instead_of_guessing(self):
        with pytest.raises(ll.LadderError, match="not laid out as"):
            ll.shared_root_of(Path("/tmp/some/other/place"))

    def test_a_shared_list_expands_to_this_clusters_root_and_is_identity_checked(self, tmp_path):
        shared = tmp_path / "cluster" / "shared"
        (shared / "exp" / "jailbreak").mkdir(parents=True)
        real = shared / "exp" / "jailbreak" / "l.json"
        real.write_text('["a","b"]')
        sha = ll.sha256_file(real)
        for declared in ("$SHARED/exp/jailbreak/l.json", "${SHARED}/exp/jailbreak/l.json"):
            got = ll._resolve_list_path(declared, shared=shared, want_sha=sha, what="benign list")
            assert got == real

    def test_an_expanded_list_with_the_wrong_content_refuses(self, tmp_path):
        shared = tmp_path / "cluster" / "shared"
        (shared / "exp").mkdir(parents=True)
        (shared / "exp" / "l.json").write_text('["different"]')
        with pytest.raises(ll.LadderError, match="DIFFERENT list"):
            ll._resolve_list_path("$SHARED/exp/l.json", shared=shared,
                                  want_sha="0" * 64, what="benign list")

    def test_an_unknown_placeholder_refuses_by_name(self, tmp_path):
        with pytest.raises(ll.LadderError, match=r"not a site placeholder"):
            ll._resolve_list_path("$SHARED_ROOT/exp/l.json", shared=tmp_path,
                                  want_sha=None, what="benign list")

    def test_an_absent_expanded_list_refuses_naming_both_spellings(self, tmp_path):
        with pytest.raises(ll.LadderError) as e:
            ll._resolve_list_path("$SHARED/exp/none.json", shared=tmp_path, want_sha=None,
                                  what="benign list")
        assert str(tmp_path / "exp" / "none.json") in str(e.value) and "$SHARED/exp/none.json" in str(e.value)

    def test_the_held_out_counts_come_from_split_v1_never_a_literal(self, tmp_path):
        root, target = tmp_path / "obj", "gemma4e4b"
        cfg = _fixture_tree(root, target, n_benign_held=11, n_harmful_held=3)
        got = ll.load_inputs(root=root, target=target, cfg=cfg, shared=tmp_path)
        assert len(got["benign"]) == 11 and len(got["harmful"]) == 3
        assert got["layers"] == [7, 8] and got["n_layers"] == 2
        assert got["benign"][0]["prompt_id"].startswith("b0000-")
        assert got["harmful"][0]["list_index"] == 1

    def test_a_split_disagreeing_with_the_screen_refuses(self, tmp_path):
        root, target = tmp_path / "obj", "gemma4e4b"
        cfg = _fixture_tree(root, target)
        screen = json.loads((root / target / "screen_f9.json").read_text())
        screen["chosen"]["layers"] = [7, 9]
        (root / target / "screen_f9.json").write_text(json.dumps(screen))
        with pytest.raises(ll.LadderError, match="disagree about which layers"):
            ll.load_inputs(root=root, target=target, cfg=cfg, shared=tmp_path)


def _fixture_tree(root: Path, target: str, *, view="it", n_benign_held=11, n_harmful_held=3,
                  n_rows=40):
    """A synthetic Phase B + bake output: the lists, the split, the screen, the fit meta, the
    parent object and six rung objects. All invented text; no real prompt is copied here."""
    tdir = root / target
    (tdir / "cells" / "alphasteer").mkdir(parents=True, exist_ok=True)
    lists = root / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    benign = [f"benign prompt {k}" for k in range(n_rows)]
    harmful = [f"harmful prompt {k}" for k in range(n_rows)]
    (lists / "benign.json").write_text(json.dumps(benign))
    (lists / "harmful.json").write_text(json.dumps(harmful))
    b_held = list(range(0, n_benign_held))
    h_held = list(range(1, 1 + n_harmful_held))
    (tdir / "split_v1.json").write_text(json.dumps({
        "split_sha256": "s" * 64, "n_heldout_benign": len(b_held),
        "n_heldout_harmful": len(h_held),
        "sides": {"benign": {"heldout_idx": b_held,
                             "fit_idx": [i for i in range(n_rows) if i not in b_held]},
                  "harmful": {"heldout_idx": h_held,
                              "fit_idx": [i for i in range(n_rows) if i not in h_held]}}}))
    (tdir / "screen_f9.json").write_text(json.dumps({
        "sha256": "f" * 64, "hidden_dim": 4, "site": "pre",
        "chosen": {"layers": [7, 8], "n_layers": 2}}))
    (tdir / "cells" / "alphasteer" / "fit_meta.json").write_text(json.dumps({
        "sha256": "m" * 64, "layers": [7, 8], "base_scale": 1.0,
        "per_layer": {"7": {"rho_fp32": {"p05": 2e-3, "p50": 4e-3, "p95": 6e-3},
                            "rho_model_dtype": {"p05": 2e-3, "p50": 4e-3, "p95": 6e-3}},
                      "8": {"rho_fp32": {"p05": 3e-3, "p50": 5e-3, "p95": 7e-3},
                            "rho_model_dtype": {"p05": 3e-3, "p50": 5e-3, "p95": 7e-3}}}}))
    _make_object(tdir / "cells" / "alphasteer" / "object.pt", 1.0)
    _ladder_tree(root, target=target, view=view)
    return {"model": "google/gemma-4-E4B-it", "model_revision": "r" * 40, "dtype": "bfloat16",
            "defenses_dir": "out/x",
            "class_b": {"benign_prompts_path": str(lists / "benign.json"),
                        "harmful_prompts_path": str(lists / "harmful.json"),
                        "benign_sha256": ll.sha256_file(lists / "benign.json"),
                        "harmful_sha256": ll.sha256_file(lists / "harmful.json")}}


# --------------------------------------------------------------------------------------------- #
# the dry run                                                                                     #
# --------------------------------------------------------------------------------------------- #

class TestDryRun:
    def test_dry_run_resolves_everything_and_launches_nothing(self, tmp_path, capsys,
                                                              monkeypatch):
        import yaml
        root = tmp_path / "exp" / "objects" / "alphasteer_f22"
        cfg = _fixture_tree(root, "gemma4e4b")
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        monkeypatch.setenv("SISTEER_LENS_RHO_EMIT", "1")
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True).stdout.strip()

        # spy rather than block: the driver legitimately shells out to `git rev-parse` to check
        # that the checkout IS the named ladder sha. What must not happen is a STANDUP.
        spawned = []
        real_popen = subprocess.Popen

        def spy(argv, *a, **k):
            spawned.append(list(argv) if isinstance(argv, (list, tuple)) else [argv])
            return real_popen(argv, *a, **k)

        monkeypatch.setattr(subprocess, "Popen", spy)
        rc = ll.main(["generate", "--repo-dir", str(REPO), "--ladder-sha", head,
                      "--target", "gemma4e4b", "--view", "it", "--root", str(root),
                      "--config", str(cfg_path), "--manifest", "configs/m.yaml",
                      "--serving-python", "/venv/bin/python", "--dry-run"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "DRY RUN" in out
        assert "121 x 4864 = 588544 batched tokens" in out
        assert "11 benign x 512 tokens, 3 harmful x 1 token" in out
        # the argv is printed one flag per line, so assert on its tokens
        assert "lens_gate_standup.py" in out
        assert "--cell" in out and "alphasteer" in out
        assert "--max-num-seqs" in out and "--expect-sha" in out
        for i in range(1, 7):
            assert f"rung {i}: lambda=" in out
        assert "STANDUP_DONE" in out
        assert not [c for c in spawned if any("lens_gate_standup" in str(a) for a in c)], (
            "a dry run must resolve and print, never stand anything up")

    def test_generate_refuses_when_the_rho_flag_is_not_set(self, tmp_path, monkeypatch):
        """The driver-side belt of §C.2a: a standup launched from a process without the flag
        records nothing, and condition (c) would be undecidable."""
        import yaml
        root = tmp_path / "exp" / "objects" / "alphasteer_f22"
        cfg = _fixture_tree(root, "gemma4e4b")
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        monkeypatch.delenv("SISTEER_LENS_RHO_EMIT", raising=False)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True).stdout.strip()
        assert ll.main(["generate", "--repo-dir", str(REPO), "--ladder-sha", head,
                        "--target", "gemma4e4b", "--view", "it", "--root", str(root),
                        "--config", str(cfg_path), "--manifest", "m.yaml",
                        "--serving-python", "p", "--dry-run"]) == 3

    def test_generate_refuses_a_checkout_that_is_not_the_named_sha(self, tmp_path, monkeypatch):
        import yaml
        root = tmp_path / "exp" / "objects" / "alphasteer_f22"
        cfg = _fixture_tree(root, "gemma4e4b")
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        monkeypatch.setenv("SISTEER_LENS_RHO_EMIT", "1")
        assert ll.main(["generate", "--repo-dir", str(REPO), "--ladder-sha", "b" * 40,
                        "--target", "gemma4e4b", "--view", "it", "--root", str(root),
                        "--config", str(cfg_path), "--manifest", "m.yaml",
                        "--serving-python", "p", "--dry-run"]) == 3

    def test_judge_and_select_is_not_an_entry_point_here(self):
        """No API key ever reaches a cluster node, so this half cannot judge."""
        with pytest.raises(SystemExit):
            ll.build_parser().parse_args(["judge-and-select"])


# --------------------------------------------------------------------------------------------- #
# the shell gates, run for real against THIS checkout                                             #
# --------------------------------------------------------------------------------------------- #

def _sh(script: str, env=None, cwd: Path = REPO):
    """Run ``script`` after sourcing THIS tree's as22_common.sh (its literals), with git commands
    resolving in ``cwd`` — the checkout under test, or a worktree built for the purpose."""
    e = dict(os.environ)
    e.pop("SISTEER_LENS_RHO_EMIT", None)
    e.pop("AS22_LADDER_SHA", None)
    e.update(env or {})
    return subprocess.run(["bash", "-c", f'cd {cwd}; source {COMMON_SH}; {script}'],
                          capture_output=True, text=True, env=e)


#: The wave's serving baseline (as22_common.sh AS22_W8_SHA; pinned literally by
#: test_the_literals_are_the_frozen_ones_and_none_is_self_referential).
W8_SHA = "935dfb43a743ec00bdf41971ecd21302aacd5342"
_W8_PATHS = ("src/sisteer/capeval/serving/", "src/sisteer/capeval/tiers/petri_runner.py")


def _git(*args, cwd: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _w8_object_present() -> bool:
    """A shallow CI checkout does not carry the pinned serving commit (``fatal: bad object``); the
    gate's diff against it cannot run there and the tests below SKIP, saying so, rather than pass
    on nothing."""
    return _git("cat-file", "-e", f"{W8_SHA}^{{commit}}").returncode == 0


def _head_serving_is_w8() -> bool:
    return _git("diff", "--quiet", W8_SHA, "HEAD", "--", *_W8_PATHS).returncode == 0


@pytest.fixture
def w8_tree(tmp_path):
    """A detached checkout of the pinned serving commit (``W8_SHA``): the ONE tree whose
    serving stack the production gate accepts by definition. HEAD is not that tree — W_L
    patches lens_worker.py, and main has since moved serving/ and petri_runner.py past it
    (both deliberately) — so the gate's PASS path, and the ladder gate's UNPATCHED refusal,
    are exercised on a worktree at that commit, built for the test and removed after it.
    Skips where that object is absent (a shallow checkout); never a vacuous pass."""
    if not _w8_object_present():
        pytest.skip(f"this clone does not carry W8 {W8_SHA[:8]} (a published repository without its development history, or a shallow "
                    "checkout): the production gate's pass path needs a W8 tree to run on, and the serving-identity claim is not verified here")
    wt = tmp_path / "w8"
    r = _git("worktree", "add", "--detach", str(wt), W8_SHA)
    assert r.returncode == 0, r.stderr
    try:
        yield wt
    finally:
        _git("worktree", "remove", "--force", str(wt))


class TestServingIdentityGate:
    def test_production_passes_on_a_w8_tree(self, w8_tree):
        r = _sh("as22_assert_serving_identity production", cwd=w8_tree)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "byte-identical to W8" in r.stdout

    def test_production_refuses_this_tree_unless_its_serving_stack_is_w8(self):
        """The half that matters on a landed tree: wherever HEAD's serving stack differs from the
        pinned serving commit the production gate must REFUSE, naming that commit and the files — a
        production leg launched from such a checkout would not serve the wave's bytes while every
        recorded serving_impl_commit says it does. On a checkout identical to the pin there is
        nothing to refuse and the pass path above covers it."""
        if not _w8_object_present():
            pytest.skip(f"this clone does not carry W8 {W8_SHA[:8]} (a published repository without its development history, or a shallow "
                        "checkout); the serving-identity claim is not verified here")
        if _head_serving_is_w8():
            pytest.skip("HEAD's serving stack IS W8's — the pass path is the case here")
        r = _sh("as22_assert_serving_identity production")
        assert r.returncode != 0
        assert "differs from W8" in r.stdout and W8_SHA in r.stdout
        assert "byte-identical to W8" not in r.stdout

    def test_production_refuses_a_stray_rho_flag(self):
        """On an unpatched tree the flag is dead, so this is a belt — but a stray export is exactly
        the drift that should refuse loudly rather than do nothing."""
        r = _sh("as22_assert_serving_identity production",
                env={"SISTEER_LENS_RHO_EMIT": "1"})
        assert r.returncode != 0 and "SISTEER_LENS_RHO_EMIT is SET" in r.stdout

    def test_production_refuses_a_stray_ladder_sha(self):
        r = _sh("as22_assert_serving_identity production", env={"AS22_LADDER_SHA": "a" * 40})
        assert r.returncode != 0 and "AS22_LADDER_SHA is set" in r.stdout

    def test_ladder_refuses_when_the_sha_is_unset_so_it_cannot_be_skipped_by_omission(self):
        r = _sh("as22_assert_serving_identity ladder", env={"SISTEER_LENS_RHO_EMIT": "1"})
        assert r.returncode != 0
        assert "AS22_LADDER_SHA is unset or empty" in r.stdout

    def test_ladder_refuses_an_unpatched_tree(self, w8_tree):
        """THE case that matters most: on W_A the rho record does not exist, so condition (c)
        could not be measured and the run would otherwise fall back, silently, to having no
        criterion at all. The message must say that, not 'mismatch'. Run on the worktree at the
        pinned serving commit — unpatched by construction, named as the ladder sha so clause (i)
        passes and the refusal is the lens_worker clause — because HEAD here IS patched (W_L) and
        would refuse for the opposite reason on a merged tree (a lens_worker.py that is neither
        the pin's nor the frozen one)."""
        r = _sh("as22_assert_serving_identity ladder", cwd=w8_tree,
                env={"SISTEER_LENS_RHO_EMIT": "1", "AS22_LADDER_SHA": W8_SHA})
        assert r.returncode != 0
        assert "UNPATCHED ladder is the dangerous case" in r.stdout
        assert "condition (c) could not be measured" in r.stdout

    def test_ladder_refuses_a_sha_that_is_not_head(self):
        r = _sh("as22_assert_serving_identity ladder",
                env={"SISTEER_LENS_RHO_EMIT": "1", "AS22_LADDER_SHA": "c" * 40})
        assert r.returncode != 0 and "!= the named ladder sha" in r.stdout

    def test_ladder_refuses_with_the_flag_off(self):
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                              capture_output=True, text=True).stdout.strip()
        r = _sh("as22_assert_serving_identity ladder", env={"AS22_LADDER_SHA": head})
        assert r.returncode != 0 and "must be 1" in r.stdout

    def test_a_non_literal_mode_refuses(self):
        for mode in ("", "Production", "ladder ", "both"):
            r = _sh(f"as22_assert_serving_identity '{mode}'")
            assert r.returncode != 0 and "must be the LITERAL" in r.stdout

    def test_the_literals_are_the_frozen_ones_and_none_is_self_referential(self):
        text = COMMON_SH.read_text()
        assert ("AS22_LADDER_LENS_WORKER_SHA256=47da382aac38ba6f3e2494ead2b67dc199029943b"
                "6eadc580bb5616a9b4263be") in text
        assert ("AS22_LADDER_PATCH_SHA256=cc84104a6d879db5389d1ea386eb38ebb3dbab6624759"
                "62f3394f1484ac40edb") in text
        assert "AS22_W8_SHA=935dfb43a743ec00bdf41971ecd21302aacd5342" in text
        # the abbreviation gotcha: the frozen diff carries 8-char blob abbreviations
        assert "core.abbrev" in text and "AS22_ABBREV=8" in text
        # the ladder sha is NEVER a literal here (it would change the sha it records)
        assert "AS22_LADDER_SHA=" not in text


class TestSbatchDiscipline:
    def test_ladder_sbatch_is_the_only_file_exporting_the_rho_flag(self):
        offenders = [p.name for p in SBATCH.glob("*")
                     if p.is_file() and "SISTEER_LENS_RHO_EMIT=1" in p.read_text()
                     and p.name != "as22_common.sh"]
        assert offenders == ["ladder.sbatch"]

    def test_ladder_sbatch_calls_the_ladder_mode_literally_and_no_other(self):
        code = _code(LADDER_SBATCH)
        calls = [l.strip() for l in code.splitlines()
                 if l.strip().startswith("as22_assert_serving_identity")]
        assert calls == ["as22_assert_serving_identity ladder"]

    def test_no_export_flag_anywhere(self):
        """Hard rule 1: --export= crashes slurmd and takes nodes down."""
        for p in (LADDER_SBATCH, COMMON_SH):
            assert "--export=" not in _code(p), p

    def test_the_headers_are_the_one_gpu_private_log_shape(self):
        text = LADDER_SBATCH.read_text()
        for line in ("#SBATCH --qos=high", "#SBATCH --gres=gpu:1",
                     "#SBATCH --mem=200G", "#SBATCH --cpus-per-task=32",
                     "#SBATCH --time=0-06:00:00", "#SBATCH --open-mode=append",
                     "#SBATCH --no-requeue", "#SBATCH --signal=B:TERM@120"):
            assert line in text, line
        assert "--mem=0" not in text
        # harmful prompt text can reach a traceback: the PRIVATE 0700 log dir, both streams
        assert "exp/jailbreak/_logs/%x_%j.out" in text
        assert "exp/jailbreak/_logs/%x_%j.err" in text
        assert "exp/logs/" not in _code(LADDER_SBATCH)

    def test_it_never_writes_a_wave_wide_sentinel(self):
        """$WAVE_ROOT/STOP and attackers/STOP are shared with a peer's live legs."""
        code = _code(LADDER_SBATCH) + "\n" + _code(COMMON_SH)
        assert "WAVE_ROOT/STOP" not in code
        assert "attackers/STOP" not in code
        assert "as22_stop_standups" in code     # our own per-standup stop files instead

    def test_it_refuses_without_the_scrub_attestation_and_bakes_before_the_first_standup(self):
        code = _code(LADDER_SBATCH)      # the ORDER of what it DOES, not of what it mentions
        assert "as22_require_scrub" in code
        # Index on the EXECUTABLE token, never the bare script name. `_code` already drops comments
        # — verified by mutation: deleting the real calls while KEEPING the f22_fatal message that
        # names the script still fails this test, and a comments-only variant fails it too — so
        # this is a belt, not a fix. It removes the class where a mere MENTION could satisfy an
        # ordering assertion, which is the source-text-matching failure this campaign has recorded
        # once already.
        i_bake = code.index("bake_rungs.py --check")
        i_drive = code.index("lambda_ladder.py generate")
        assert code.index("as22_require_scrub") < i_bake < i_drive
        assert "--check" in code[i_bake - 200:i_bake + 400], "verify-then-build, as f22_bank_view"

    def test_a_missing_object_is_fatal_here_even_though_the_bake_calls_it_a_skip(self):
        """The exit-3 seam, pinned. `bake_rungs.py` exits 3 for "no fitted object", which is a
        clean per-target SKIP for a multi-target orchestrator and must be FATAL in this
        SINGLE-target job — otherwise a ladder stands six rungs up against whatever objects happen
        to be on disk. Live right now, not hypothetical: nemotron3nano4b has no object."""
        code = _code(LADDER_SBATCH)
        assert "bake_rc" in code, "the bake's exit code must be captured, not merely tested nonzero"
        window = code[code.index("bake_rungs.py --check"):][:1400]
        assert '"$bake_rc" -eq 3' in window, "exit 3 must be told apart from a bake failure"
        assert window.count("f22_fatal") >= 3, (
            "all three paths must be fatal: exit 3 from --check, exit 3 from the bake, any other "
            "nonzero")
        assert "NOT READY" in window.upper(), (
            "the exit-3 message must say the target is not ready, so a reader is not sent hunting "
            "for a bake bug that does not exist")

    def test_it_sets_the_env_in_the_body_and_never_special_cases_a_target(self):
        code = _code(LADDER_SBATCH)
        for line in ("export SISTEER_LENS_RHO_EMIT=1",
                     "export SISTEER_BITES_POLICY=harmful_continuous",
                     "export SISTEER_LENS_STACKED_REDUCTIONS=1"):
            assert line in code, line
        for key in ("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b"):
            assert key not in code, (
                f"{key} is named in ladder.sbatch's code — the job must be generic over the "
                "target key so nemotron runs it unchanged when its object lands")
        # ...and the same for the driver, asserted STRUCTURALLY rather than by grep: a model key
        # may appear in a docstring (the usage example, and the explanation of why qwen and
        # nemotron take the hybrid branch) but never as a string literal in executable code, so
        # there is no per-target branch to keep in sync.
        import ast
        src = (REPO / "scripts" / "alphasteer_f22" / "lambda_ladder.py").read_text()
        tree = ast.parse(src)
        docs = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) and ast.get_docstring(node):
                docs.add(id(node.body[0].value))
        keys = ("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b")
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docs):
                for key in keys:
                    assert key not in node.value, (
                        f"lambda_ladder.py carries the literal {key!r} in executable code "
                        f"(line {node.lineno}) — the driver must be generic over the target")

    def test_both_shell_files_parse(self):
        for p in (COMMON_SH, LADDER_SBATCH):
            assert subprocess.run(["bash", "-n", str(p)]).returncode == 0, p


# --------------------------------------------------------------------------------------------- #
# serving_impl_commit: UNFROZEN in the tree, the sha that ran in the record                       #
# --------------------------------------------------------------------------------------------- #


class TestServingImplCommit:
    """The ladder's manifests record W_L's serving_impl_commit and a tree cannot record
    its own sha — W_L's one commit touches lens_worker.py, so by definition serving_impl_commit == W_L.
    The checked-in value is therefore the loader's UNFROZEN literal; the standup never reads it, and
    the driver files the sha lens_gate_standup verified before serving."""

    OUR_MANIFEST = "configs/capeval/manifests/alphasteer_f22_gemma4e4b.yaml"

    def test_the_checked_in_manifest_is_unfrozen_and_the_standup_never_reads_it(self, tmp_path):
        import yaml
        from sisteer.capeval import manifest as cm
        spec0 = cm.load_petri_spec(REPO / self.OUR_MANIFEST)
        assert spec0.serving_impl_commit == cm.PETRI_UNFROZEN
        cfg = yaml.safe_load((REPO / "configs/alphasteer_f22_serve_gemma4e4b.yaml").read_text())
        ddir = tmp_path / "cells_view" / "it"
        _make_object(ddir / "alphasteer" / "object.pt", 0.25, d=2560)
        derived = ll._write_standup_config(cfg, dest=tmp_path / "standup.yaml", defenses_dir=ddir)
        # the REAL ladder manifest, UNFROZEN and all: enumeration (which hashes the spec per cell and
        # refuses the literal) still succeeds, because the standup replaces the field first
        spec, _src, cells, _p, _added = _gate_standup().enumerate_gate_cells(
            repo=REPO, cfg=yaml.safe_load(derived.read_text()), cells_tt=["alphasteer"],
            defenses_dir=ddir, manifest=self.OUR_MANIFEST, chat_template_hash="ab" * 8, out=tmp_path)
        assert {c.cell_id for c in cells.all_served()} == {"none", "alphasteer", "none__lens0"}
        assert cells.absent == ()
        assert spec.serving_impl_commit != cm.PETRI_UNFROZEN
        assert cm._is_full_sha(spec.serving_impl_commit)      # the standup's own placeholder, not ours
        with pytest.raises(cm.ManifestError, match="serving_impl_commit is UNFROZEN"):
            cm.petri_harness_config_hash(
                spec0, behavior_files={k: b"# b\n" for k in cm.petri_expected_behavior_keys(spec0)},
                prompt_files={"auditor": b"# a\n", "judge": b"# j\n"},
                serving_identity="proxy_vllm+lens+interleaved")   # ...so no OTHER consumer can serve on it as-is

    def test_the_record_files_the_standups_verified_head_and_refuses_otherwise(self):
        wl = "c" * 40
        assert ll.serving_impl_commit_of({"commit": wl}, ladder_sha=wl, rung_i=1) == wl
        with pytest.raises(ll.LadderError, match="carries no `commit`"):
            ll.serving_impl_commit_of({}, ladder_sha=wl, rung_i=2)
        with pytest.raises(ll.LadderError, match="carries no `commit`"):
            ll.serving_impl_commit_of({"commit": ""}, ladder_sha=wl, rung_i=2)
        with pytest.raises(ll.LadderError, match="did not serve"):
            ll.serving_impl_commit_of({"commit": "d" * 40}, ladder_sha=wl, rung_i=3)

    def test_run_rung_files_that_value_and_nothing_else(self):
        """The record's field is the function's return, not a second read of endpoint.json."""
        src = (REPO / "scripts" / "alphasteer_f22" / "lambda_ladder.py").read_text()
        assert '"serving_impl_commit": ep_commit,' in src
        assert 'ep_commit = serving_impl_commit_of(endpoint, ladder_sha=ladder_sha, rung_i=i)' in src
        assert '"serving_impl_commit": endpoint.get("commit")' not in src


# ------------------------------------------------------------------------------------------------ #
# A statistics refusal leaves a durable record, with the per-layer non-finite diagnosis            #
# ------------------------------------------------------------------------------------------------ #


class TestStatisticsRefusalRecord:
    def test_nonfinite_harmful_readings_are_counted_per_fit_layer(self):
        """glm46vflash rung 5: fit layer 26 had 200/200 harmful rho_last null — the composed edit
        overflowed bf16 with depth. The diagnosis is per FIT layer, harmful slice only, three fields."""
        ents = []
        for k in range(4):
            ents.append(_entry(f"h{k}", 8, slice="harmful", rho_last=0.01, h_norm_last=100.0, add_norm_last=1.0))
            ents.append(_entry(f"h{k}", 25, slice="harmful", rho_last=None, h_norm_last=(None if k < 3 else 6e18),
                               add_norm_last=None))
            ents.append(_entry(f"b{k}", 25, slice="benign", rho_last=None, h_norm_last=None, add_norm_last=None))
        ents.append(_entry("h9", 24, slice="harmful", rho_last=float("inf"), h_norm_last=1e18, add_norm_last=float("nan")))
        d = ll.nonfinite_harmful_by_layer(ents)
        assert set(d) == {"9", "26", "25"}                        # fit layer = hook + 1; benign never counted
        assert d["9"] == {"n": 4, "n_rho_last_nonfinite": 0, "n_h_norm_last_nonfinite": 0, "n_add_norm_last_nonfinite": 0}
        assert d["26"] == {"n": 4, "n_rho_last_nonfinite": 4, "n_h_norm_last_nonfinite": 3, "n_add_norm_last_nonfinite": 4}
        assert d["25"] == {"n": 1, "n_rho_last_nonfinite": 1, "n_h_norm_last_nonfinite": 0, "n_add_norm_last_nonfinite": 1}

    def test_a_layer_with_no_finite_reading_still_refuses_and_the_record_is_written_first(self):
        """rho_statistics keeps refusing (a missing layer cannot be a pass); run_rung now writes the
        uncertified record BEFORE re-raising, as the integrity path always did (evidence durable)."""
        src = (REPO / "scripts" / "alphasteer_f22" / "lambda_ladder.py").read_text()
        body = src[src.index("    try:\n        stats = rho_statistics("):src.index("    record.update(stats)")]
        for needle in ('record["uncertified"] = True', 'record["cond_c_pass"] = None',
                       'record["terminal_refusal"] = "served_rho_unformable"',
                       'record["served_rho_nonfinite_by_fit_layer"] = nonfinite_harmful_by_layer(entries)',
                       'write_json_atomic(rung_dir / "rung_record.json", record)', "raise LadderError("):
            assert needle in body, needle
        assert body.index('write_json_atomic(rung_dir / "rung_record.json", record)') < body.index("raise LadderError(")
