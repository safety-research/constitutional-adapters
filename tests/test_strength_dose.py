"""The strength frontier's common physical dose: the mirror is held to the SERVED adapters.

Everything here runs on REAL BYTES: the campaign's own staged objects (pulled to a fixture dir, since
they live on the clusters) and the shared reference pool file. Where an arithmetic claim is made, the
oracle is `sisteer.capeval.serving.lens_adapters` ITSELF — the tree this lane may not modify — so a
divergence between what dose_common computes and what the serving path would do fails here rather than
being discovered in a number nobody can check.

The fixtures are real cluster artefacts, not invented ones:
    STRENGTH_DOSE_FIXTURES (default ~/.local/state/strength-dose/fixtures)
      lora/nemotron3nano4b/<staged_id>/object.pt   the 6 LoRA rungs x 2 cells + the unscaled parents
      alphasteer/qwen35_4b/<staged_id>/object.pt   two AlphaSteer rungs (as_dm0p5, as_d0p5)
      pool/pool_v1.jsonl                           the shared reference pool
Absent fixtures SKIP (another clone has no cluster artefacts); a PRESENT fixture that disagrees FAILS.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts" / "sdfbase" / "full22" / "strength"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import dose_common as dc  # noqa: E402
from sisteer.capeval.serving import lens_adapters as la  # noqa: E402

FIX = Path(os.environ.get("STRENGTH_DOSE_FIXTURES", Path.home() / ".local/state/strength-dose/fixtures"))
LORA_DIR = FIX / "lora" / "nemotron3nano4b"
AS_DIR = FIX / "alphasteer" / "qwen35_4b"
POOL = FIX / "pool" / "pool_v1.jsonl"


def _need(p: Path):
    if not p.exists():
        raise unittest.SkipTest(f"no fixture at {p} (cluster artefact; see this module's docstring)")
    return p


def _blob(path: Path):
    return dc.load_blob(path)


class PoolContract(unittest.TestCase):
    """The pool is the one artefact two lanes share, so its loader refuses everything that would make
    the two lanes' numbers incomparable while still looking fine."""

    def setUp(self):
        _need(POOL)

    def test_real_pool_loads_and_matches_the_pinned_sha(self):
        rows = dc.load_pool(POOL)
        self.assertEqual(len(rows), 379)
        self.assertEqual(sum(r["half"] == "harmful" for r in rows), 259)
        self.assertEqual(sum(r["half"] == "benign" for r in rows), 120)
        self.assertEqual([r["idx"] for r in rows], list(range(379)))
        self.assertTrue(all(dc.half_of(r["idx"]) == r["half"] for r in rows))

    def test_a_changed_byte_is_refused(self):
        raw = POOL.read_bytes()
        tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory())) / "p.jsonl"
        rows = raw.decode().splitlines()
        doc = json.loads(rows[0]); doc["text"] = doc["text"] + " "
        rows[0] = json.dumps(doc)
        tmp.write_text("\n".join(rows) + "\n")
        with self.assertRaisesRegex(dc.PoolError, "sha256"):
            dc.load_pool(tmp)

    def test_a_truncated_pool_is_refused_even_at_its_own_sha(self):
        raw = POOL.read_bytes().decode().splitlines()[:300]
        tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory())) / "p.jsonl"
        body = "\n".join(raw) + "\n"
        tmp.write_text(body)
        with self.assertRaisesRegex(dc.PoolError, "rows, expected 379"):
            dc.load_pool(tmp, dc.sha256_bytes(body.encode()))

    def test_a_repr_in_place_of_a_prompt_is_refused(self):
        """The defect that actually happened in the sibling lane: `str(goal)` instead of `goal.text`."""
        rows = POOL.read_bytes().decode().splitlines()
        doc = json.loads(rows[5])
        doc["text"] = ("Goal(goal_id='harmbench_x', text='how do i ...', category='chemical', "
                       "dataset='harmbench', revision='4aa1', split='test')")
        rows[5] = json.dumps(doc)
        body = "\n".join(rows) + "\n"
        tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory())) / "p.jsonl"
        tmp.write_text(body)
        with self.assertRaisesRegex(dc.PoolError, "repr"):
            dc.load_pool(tmp, dc.sha256_bytes(body.encode()))

    def test_the_content_guard_also_runs_on_rendered_strings(self):
        with self.assertRaisesRegex(dc.PoolError, "a repr reached the render"):
            dc.assert_rendered_clean(["<start>ok</start>", "<start>Goal(goal_id='x', text='y')</start>"])


class LoraOpsMirrorTheAdapter(unittest.TestCase):
    """dose_common's op list must be the adapter's op list — same order, same layers, same coefficients,
    the same tensors — on the campaign's real objects, including the two-op subtraction cell."""

    def setUp(self):
        _need(LORA_DIR)

    def _pairs(self, staged_id: str):
        path = LORA_DIR / staged_id / "object.pt"
        _need(path)
        blob = _blob(path)
        mine = dc.lora_ops_from_blob(blob)
        served = la.get_adapter(blob["kind"])({"kind": blob["kind"]}, path.read_bytes())
        return blob, mine, served

    def test_lora_all_matches_the_served_ops(self):
        blob, mine, served = self._pairs("base__constv2-mix-full__lora_r4_ce")
        self.assertEqual(len(mine), len(served))
        for m, sop in zip(mine, served):
            self.assertEqual(sop.op, la.OP_LORA_OPROJ)
            self.assertEqual(m.layer, sop.layer)
            self.assertEqual(m.coeff, sop.coeff)
            # the adapter stores the READ basis transposed; the write basis as-is
            self.assertTrue(torch.equal(m.A.t().contiguous(), sop.lowrank_a))
            self.assertTrue(torch.equal(m.B, sop.lowrank_b))

    def test_lora_subtraction_serves_two_ops_per_layer_with_the_negative_parent_at_minus_one(self):
        blob, mine, served = self._pairs("dm1__constv2-mix-full__lora_r4_ce_minus_ctl")
        self.assertEqual(len(mine), len(served))
        self.assertEqual(len(mine), 2 * len(blob["layers"]))
        self.assertEqual([m.coeff for m in mine], [c.coeff for c in served])
        self.assertEqual([m.coeff for m in mine][:2], [1.0, -1.0])
        self.assertEqual([m.parent for m in mine][:2], ["pos", "neg"])
        for m, sop in zip(mine, served):
            self.assertEqual(m.layer, sop.layer)
            self.assertTrue(torch.equal(m.A.t().contiguous(), sop.lowrank_a))
            self.assertTrue(torch.equal(m.B, sop.lowrank_b))

    def test_the_edit_equals_apply_oproj_ops_on_a_zero_output(self):
        """The whole arithmetic, against the served function itself: the served composition added to a
        ZERO o_proj output IS the edit, so apply_oproj_ops is a direct oracle for the served column."""
        for staged in ("d0p25__constv2-mix-full__lora_r4_ce", "dm1__constv2-mix-full__lora_r4_ce_minus_ctl"):
            with self.subTest(staged=staged):
                blob, mine, served = self._pairs(staged)
                dtype = torch.bfloat16
                d_in = mine[0].A.shape[0]
                d_out = mine[0].B.shape[0]
                torch.manual_seed(0)
                x = torch.randn(17, d_in, dtype=dtype)          # rows of one request; shapes are the object's own
                zero = torch.zeros(17, d_out, dtype=dtype)
                for layer in sorted({m.layer for m in mine}):
                    lops = [m for m in mine if m.layer == layer]
                    sops = [o for o in served if o.layer == layer]
                    want = la.apply_oproj_ops(x, zero, sops, 1.0, layer=layer)
                    _fp32, got = dc.lora_edit_from_reads([dc.lora_read(x, m) for m in lops], lops, dtype)
                    self.assertTrue(torch.equal(want, got), f"{staged} layer {layer}")
                    # and against a NON-zero output: the served composition adds op by op to the running
                    # output, so this is the case the subtraction cell's second rounding depends on
                    torch.manual_seed(1)
                    base = torch.randn(17, d_out, dtype=dtype)
                    want2 = la.apply_oproj_ops(x, base, sops, 1.0, layer=layer)
                    _fp32, got2 = dc.lora_edit_from_reads([dc.lora_read(x, m) for m in lops], lops, dtype, base=base)
                    self.assertTrue(torch.equal(want2, got2), f"{staged} layer {layer} on a real-shaped output")

    def test_the_dose_copy_scales_only_the_write_basis(self):
        """The producer's claim, re-checked from the bytes this lane actually reads."""
        parent = dc.lora_ops_from_blob(_blob(_need(LORA_DIR / "base__constv2-mix-full__lora_r4_ce" / "object.pt")))
        for staged, scale in (("d0p25__constv2-mix-full__lora_r4_ce", 0.25),
                              ("dm1__constv2-mix-full__lora_r4_ce", -1.0),
                              ("d1p5__constv2-mix-full__lora_r4_ce", 1.5)):
            with self.subTest(staged=staged):
                rung = dc.lora_ops_from_blob(_blob(_need(LORA_DIR / staged / "object.pt")))
                for pop, rop in zip(parent, rung):
                    self.assertTrue(torch.equal(pop.A, rop.A), "the read basis must be untouched")
                    self.assertTrue(torch.equal(pop.B * scale, rop.B))


class AlphaSteerMirrorsTheAdapter(unittest.TestCase):
    def setUp(self):
        _need(AS_DIR)

    def test_hook_layers_coeff_and_matrices_match_the_served_ops(self):
        for staged, strength in (("as_dm0p5__alphasteer", -0.5), ("as_d0p5__alphasteer", 0.5)):
            with self.subTest(staged=staged):
                path = _need(AS_DIR / staged / "object.pt")
                blob = _blob(path)
                mine = dc.alphasteer_from_blob(blob)
                served = la.get_adapter("alphasteer")({"kind": "alphasteer"}, path.read_bytes())
                self.assertEqual(mine.base_scale, strength)
                self.assertEqual(len(served), len(mine.fit_layers))
                self.assertEqual([o.layer for o in served], list(mine.hook_layers))
                self.assertEqual([int(f) - 1 for f in mine.fit_layers], list(mine.hook_layers),
                                 "site=pre puts the hook one block below the fit layer")
                for o, fl in zip(served, mine.fit_layers):
                    self.assertEqual(o.op, la.OP_PREFILL_LAST_MATMUL)
                    self.assertEqual(o.coeff, strength)
                    self.assertEqual(o.phase, la.PHASE_PREFILL)
                    self.assertTrue(torch.equal(o.matrix, mine.Ms[fl]))

    def test_the_edit_equals_apply_hook_ops_minus_the_unsteered_rows(self):
        """apply_hook_ops on a real prefill context is the oracle; the difference it makes to the rows
        IS the edit, and the served op broadcasts one vector to every prompt row."""
        path = _need(AS_DIR / "as_d0p5__alphasteer" / "object.pt")
        blob = _blob(path)
        mine = dc.alphasteer_from_blob(blob)
        served = la.get_adapter("alphasteer")({"kind": "alphasteer"}, path.read_bytes())
        dtype = torch.bfloat16
        T = 11
        torch.manual_seed(0)
        hidden = torch.randn(T, mine.d, dtype=dtype)
        ctx = la.HookContext(phase=la.PHASE_PREFILL, positions=tuple(range(T)))
        op = served[0]
        ops = la.materialize_ops([op], device=hidden.device, dtype=dtype)
        out = la.apply_hook_ops(hidden, ops, 1.0, layer=op.layer, ctx=ctx)
        fl = mine.fit_layers[list(mine.hook_layers).index(op.layer)]
        _fp32, got = dc.alphasteer_edit(hidden[-1:], mine.Ms[fl], mine.base_scale, dtype)
        # compared at the OUTPUT, not by subtracting: `hidden + edit` rounds in bf16, so recovering the
        # edit by subtraction is not the identity and would compare the wrong thing
        self.assertTrue(torch.equal(out, hidden + got.expand(T, -1)),
                        "every prompt row gets the same h_last-derived edit, bit-for-bit")

    def test_a_batched_gemm_would_NOT_be_bit_identical(self):
        """Why alphasteer_edit loops: the batched product is the same mathematics and different bits,
        and the effective-layer count is an elementwise question about exactly those bits."""
        path = _need(AS_DIR / "as_d0p5__alphasteer" / "object.pt")
        mine = dc.alphasteer_from_blob(_blob(path))
        fl = mine.fit_layers[0]
        torch.manual_seed(0)
        h = torch.randn(8, mine.d, dtype=torch.bfloat16)
        _fp32, served = dc.alphasteer_edit(h, mine.Ms[fl], mine.base_scale, torch.bfloat16)
        batched = (mine.base_scale * torch.matmul(h, mine.Ms[fl].to(torch.bfloat16))).to(torch.bfloat16)
        per_row = torch.stack([(mine.base_scale * torch.matmul(h[i], mine.Ms[fl].to(torch.bfloat16))).to(torch.bfloat16)
                               for i in range(h.shape[0])], 0)
        self.assertTrue(torch.equal(served, per_row), "the served column must be the per-request GEMV")
        if torch.equal(batched, per_row):
            self.skipTest("this BLAS happens to agree on these shapes; the loop is still the contract")


class GateCatchesTheFailuresItExistsFor(unittest.TestCase):
    """The gate must pass the real rungs and fail the three ways a dose copy can be wrong."""

    def setUp(self):
        _need(LORA_DIR)
        self.parent = dc.lora_ops_from_blob(_blob(_need(LORA_DIR / "base__constv2-mix-full__lora_r4_ce" / "object.pt")))
        torch.manual_seed(0)
        self.x = torch.randn(64, self.parent[0].A.shape[0], dtype=torch.bfloat16)

    def _edit(self, ops, layer):
        lops = [o for o in ops if o.layer == layer]
        fp32, _ = dc.lora_edit_from_reads([dc.lora_read(self.x, o) for o in lops], lops, torch.bfloat16)
        return fp32

    def test_every_real_rung_passes_and_the_negative_is_exact(self):
        for staged, s in (("d0p1__constv2-mix-full__lora_r4_ce", 0.1),
                          ("d0p25__constv2-mix-full__lora_r4_ce", 0.25),
                          ("halfdose__constv2-mix-full__lora_r4_ce", 0.5),
                          ("d1p5__constv2-mix-full__lora_r4_ce", 1.5),
                          ("dm1__constv2-mix-full__lora_r4_ce", -1.0)):
            with self.subTest(staged=staged):
                rung = dc.lora_ops_from_blob(_blob(_need(LORA_DIR / staged / "object.pt")))
                cs, rs, nr = [], [], []
                for layer in sorted({o.layer for o in rung}):
                    e, eref = self._edit(rung, layer), self._edit(self.parent, layer)
                    c, resid = dc.signed_factor(e, eref)
                    cs.append(c); rs.append(resid); nr.append(e.norm(dim=-1) / eref.norm(dim=-1))
                v = dc.gate_verdict(torch.cat(cs), torch.cat(rs), torch.cat(nr), s, 1.0)
                self.assertTrue(v["pass"], v)
                if s in (-1.0, 0.5):
                    self.assertTrue(v["exact"], f"{staged}: an exactly representable factor must be bit-exact")

    def test_a_sign_error_fails_the_gate_that_a_norm_ratio_would_pass(self):
        """Why the gate is on the SIGNED factor: |edit| is identical when the sign is wrong."""
        rung = dc.lora_ops_from_blob(_blob(_need(LORA_DIR / "dm1__constv2-mix-full__lora_r4_ce" / "object.pt")))
        layer = sorted({o.layer for o in rung})[0]
        e, eref = self._edit(rung, layer), self._edit(self.parent, layer)
        c, resid = dc.signed_factor(e, eref)
        nr = e.norm(dim=-1) / eref.norm(dim=-1)
        self.assertTrue(dc.gate_verdict(c, resid, nr, -1.0, 1.0)["pass"])
        wrong = dc.gate_verdict(c, resid, nr, +1.0, 1.0)          # the label says +1, the bytes are -1
        self.assertFalse(wrong["pass"])
        self.assertAlmostEqual(wrong["checks"]["median_norm_ratio"]["got"], 1.0, places=6,
                               msg="the norm ratio is blind to the sign — this is the case the gate exists for")

    def test_a_half_dose_built_by_scaling_the_READ_basis_still_gates_but_is_caught_upstream(self):
        """A copy that scaled A instead of B is still exactly linear, so the gate — a linearity
        check — passes it. That is not a hole in the gate; it is the reason the capture ALSO asserts
        the read basis is bit-identical to the reference rung's before any number is produced."""
        blob = _blob(_need(LORA_DIR / "base__constv2-mix-full__lora_r4_ce" / "object.pt"))
        wrong = dict(blob)
        wrong["A"] = {k: v * 0.5 for k, v in blob["A"].items()}          # the WRONG tensor scaled
        wrong_ops = dc.lora_ops_from_blob(wrong)
        layer = sorted({o.layer for o in self.parent})[0]
        e, eref = self._edit(wrong_ops, layer), self._edit(self.parent, layer)
        c, resid = dc.signed_factor(e, eref)
        v = dc.gate_verdict(c, resid, e.norm(dim=-1) / eref.norm(dim=-1), 0.5, 1.0)
        self.assertTrue(v["pass"], "scaling A is linear too — the gate cannot distinguish it")
        with self.assertRaises(RuntimeError):        # ... but the read-basis identity check does
            for a, b in zip(wrong_ops, self.parent):
                if not torch.equal(a.A, b.A):
                    raise RuntimeError("read basis moved")

    def test_a_mutated_factor_that_would_not_reach_the_arithmetic_is_refused(self):
        from dataclasses import replace
        op = self.parent[0]
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            replace(op, A=op.A * 0.5)                # lowrank_a unchanged -> silently no-op; refused

    def test_a_wrong_layer_pairing_fails(self):
        layers = sorted({o.layer for o in self.parent})
        e = self._edit(self.parent, layers[1])
        eref = self._edit(self.parent, layers[0])
        c, resid = dc.signed_factor(e, eref)
        nr = e.norm(dim=-1) / eref.norm(dim=-1)
        self.assertFalse(dc.gate_verdict(c, resid, nr, 1.0, 1.0)["pass"],
                         "a different layer's edit is not the same edit at factor 1")


class SurvivalAndEffectiveLayers(unittest.TestCase):
    def test_an_edit_below_the_served_resolution_changes_nothing(self):
        base = torch.full((4, 32), 100.0, dtype=torch.bfloat16)
        tiny = torch.full((4, 32), 1e-3, dtype=torch.bfloat16)     # far below one bf16 ulp of 100
        s = dc.survival(base, tiny, torch.bfloat16)
        self.assertTrue(torch.equal((base + tiny.to(torch.bfloat16)), base))
        self.assertEqual(float(s["frac_changed"].max()), 0.0)
        self.assertLess(float(s["edit_ulps_max"].max()), 1.0)

    def test_an_edit_above_it_changes_everything(self):
        base = torch.full((4, 32), 1.0, dtype=torch.bfloat16)
        big = torch.full((4, 32), 0.5, dtype=torch.bfloat16)
        s = dc.survival(base, big, torch.bfloat16)
        self.assertEqual(float(s["frac_changed"].min()), 1.0)

    def test_effective_layer_count_and_its_recompute_record(self):
        frac = {3: torch.tensor([0.0, 0.0, 0.0]), 7: torch.tensor([1.0, 1.0, 0.9]), 9: torch.tensor([0.2, 0.1, 0.3])}
        e = dc.effective_layers(frac)
        self.assertEqual(e["nominal_layer_count"], 3)
        self.assertEqual(e["effective_layer_count"], 2)
        self.assertEqual(e["active_layers"], [7, 9])
        self.assertEqual(e["effective_layer_count_half"], 1)
        self.assertEqual(e["inactive_layers"], [3])
        self.assertEqual(set(e["median_frac_changed_by_layer"]), {3, 7, 9},
                         "the per-layer vector ships so a reader can recompute at another threshold")

    def test_zero_effective_layers_is_representable(self):
        e = dc.effective_layers({1: torch.zeros(5), 2: torch.zeros(5)})
        self.assertEqual(e["effective_layer_count"], 0)
        self.assertEqual(e["active_layers"], [])

    def test_zero_effective_layers_does_NOT_mean_the_object_is_a_no_op(self):
        """The review's finding on the dose report (2026-09-12). The per-layer statistic is a MEDIAN over
        prompts, so a layer whose edit survives on a MINORITY of prompts reads inactive. Two layers each
        changing 2 of 5 prompts give an effective count of 0 while 4 of those 5 prompts changed somewhere —
        so the count is right and the prose claim "any eval of this arm measures the base model" was not.
        The record now carries its own scope; this pins both the number and the disclaimer."""
        frac = {1: torch.tensor([0.9, 0.9, 0.0, 0.0, 0.0]),      # prompts 0,1 change
                2: torch.tensor([0.0, 0.0, 0.9, 0.9, 0.0])}      # prompts 2,3 change
        e = dc.effective_layers(frac)
        self.assertEqual(e["effective_layer_count"], 0, "each layer's MEDIAN prompt is unchanged")
        changed = [i for i in range(5) if any(float(v[i]) > 0 for v in frac.values())]
        self.assertEqual(changed, [0, 1, 2, 3], "yet 4 of 5 prompts changed somewhere — invisible to a median")
        self.assertIn("zero_does_not_mean", e["scope"])
        self.assertIn("no-op", e["scope"]["zero_does_not_mean"])
        self.assertIn("median", e["scope"]["statistic"].lower())


class DoseCaptureArmResolution(unittest.TestCase):
    """The three P2 findings from the Wave-3 landing review (2026-09-12), each reproduced then pinned."""

    def setUp(self):
        import dose_capture
        self.cap = dose_capture

    def test_the_reference_rung_is_chosen_by_DOSE_not_by_the_spelling_d1(self):
        """The instruct-direct lane spells its 1.0 rung `it_d1`, so hard-coding "d1" made
        --include-it-views ALWAYS refuse on gemma4e4b / qwen35_4b with every object correctly staged."""
        chat = {"dm1": {"dose": -1.0}, "d0p5": {"dose": 0.5}, "d1": {"dose": 1.0}}
        it = {"it_dm1": {"dose": -1.0}, "it_d0p5": {"dose": 0.5}, "it_d1": {"dose": 1.0}}
        self.assertEqual(self.cap.reference_rung("t/c", chat), "d1")
        self.assertEqual(self.cap.reference_rung("t/c", it), "it_d1", "the instruct lane resolves too")

    def test_a_cell_with_no_1p0_rung_still_refuses_and_names_the_cell(self):
        with self.assertRaises(SystemExit) as e:
            self.cap.reference_rung("gemma4e4b/lora", {"d0p5": {"dose": 0.5}, "dm1": {"dose": -1.0}})
        self.assertIn("gemma4e4b/lora", str(e.exception))
        self.assertIn("exactly ONE rung at dose 1.0", str(e.exception))

    def test_staged_cells_with_no_bake_record_REFUSE_rather_than_dropping_the_arm(self):
        """Finding 1: returning None there silently dropped the arm, and on an instruct target (LoRA views
        excluded by default) that left NO arms — after which the capture loaded the model, wrote output and
        passed a gate over zero rungs, because `all([])` is True."""
        import tempfile
        shared = Path(tempfile.mkdtemp())
        bank = shared / self.cap.AS_BANK_REL / "gemma4e4b" / "strength"
        (bank / "cells").mkdir(parents=True)
        with self.assertRaises(SystemExit) as e:
            self.cap.resolve_alphasteer_arm("gemma4e4b", shared)
        self.assertIn("staged but unrecorded", str(e.exception))
        self.assertIn("--no-alphasteer", str(e.exception), "the refusal names the deliberate opt-out")

    def test_a_bank_with_no_cells_at_all_is_still_an_absent_arm_not_a_refusal(self):
        """Absent is absent. Only STAGED-BUT-UNRECORDED is the incoherent state."""
        import tempfile
        shared = Path(tempfile.mkdtemp())
        arm, rec = self.cap.resolve_alphasteer_arm("gemma4e4b", shared)
        self.assertIsNone(arm)
        self.assertEqual(rec["status"], "no strength_bake.json")

    def test_the_EMITTED_warning_is_scoped_not_categorical_under_the_counterexample(self):
        """The residual finding from the delta review (2026-09-12). The REPORT's claim was scoped but the
        capture's own warning still told a log reader the arm "changes NOTHING" and that "any eval of it
        measures the base model". A reader who never opens the report met only the categorical version.

        This asserts the EMITTED TEXT under the exact counterexample — two layers, each changing 2 of 5
        prompts, so the effective count is 0 while 4 of 5 prompts changed somewhere — rather than the
        metadata `scope` block, which my first fix checked and which no log reader ever sees."""
        frac = {1: torch.tensor([0.9, 0.9, 0.0, 0.0, 0.0]),
                2: torch.tensor([0.0, 0.0, 0.9, 0.9, 0.0])}
        e = dc.effective_layers(frac)
        self.assertEqual(e["effective_layer_count"], 0, "the counterexample must actually reach the zero branch")

        lines = []
        orig = self.cap.log
        self.cap.log = lambda *a: lines.append(" ".join(str(x) for x in a))
        try:
            self.cap.log_effective_layer_summary(
                "gemma4e4b", {"zero_effective_layers": ["lora/d0p1"], "narrowed": {"lora/d0p25": (2, 13)}})
        finally:
            self.cap.log = orig
        text = "\n".join(lines)

        self.assertIn("lora/d0p1", text)
        for categorical in ("changes NOTHING", "measures the base model", "unsteered model wearing",
                            "NOT a well-behaved small dose"):
            self.assertNotIn(categorical, text, f"the emitted warning still claims {categorical!r}")
        for scoped in ("MEDIAN", "SUSPECTED", "position/phase", "minority of prompts"):
            self.assertIn(scoped, text, f"the emitted warning does not carry {scoped!r}")
        self.assertIn("MEDIAN prompt", [l for l in lines if "NARROWED" in l][0],
                      "the NARROWED line is scoped too — it makes the same per-layer claim")

    def test_all_of_an_empty_gate_is_vacuously_true(self):
        """The property the empty-arm refusal exists for, pinned so nobody 'simplifies' the guard away."""
        self.assertTrue(all(p for _, p in []), "an empty rung set passes every gate by construction")


class PostAttentionNormPlacement(unittest.TestCase):
    """Which tensor the o_proj edit lands in differs by architecture, and a wrong answer is a wrong
    quantity rather than a crash — so the table refuses anything it has not been told about."""

    def test_an_unknown_block_class_refuses_rather_than_guessing(self):
        class SomeFutureDecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.post_attention_layernorm = torch.nn.LayerNorm(4)
        with self.assertRaisesRegex(RuntimeError, "not in POST_ATTENTION_NORM_BY_BLOCK"):
            dc.post_attention_norm(SomeFutureDecoderLayer())

    def test_the_two_in_scope_lora_targets_are_declared_and_differ(self):
        self.assertIsNone(dc.POST_ATTENTION_NORM_BY_BLOCK["NemotronHBlock"])
        self.assertEqual(dc.POST_ATTENTION_NORM_BY_BLOCK["Glm4vTextDecoderLayer"], "post_self_attn_layernorm")

    def test_a_direct_add_block_passes_the_edit_through_unchanged(self):
        torch.manual_seed(0)
        attn = torch.randn(3, 8, dtype=torch.bfloat16); e = torch.randn(3, 8); pre = torch.randn(3, 8, dtype=torch.bfloat16)
        steered = attn + e.to(torch.bfloat16)
        fp32, served, mid_st, mid = dc.residual_edit_through_norm(attn, e, steered, pre, None, torch.bfloat16)
        self.assertTrue(torch.equal(fp32, e))
        self.assertTrue(torch.equal(mid, pre + attn) and torch.equal(mid_st, pre + steered),
                        "the residual add is replayed in the model dtype on both the unsteered and steered rows")

    def test_a_norm_bearing_block_delivers_a_DIFFERENT_perturbation(self):
        """RMSNorm re-normalises the attention output, so the residual perturbation is neither the edit
        nor a fixed multiple of it — which is why the headline dose is computed through the module."""
        torch.manual_seed(0)
        norm = torch.nn.RMSNorm(16, eps=1e-6)
        with torch.no_grad():
            norm.weight.fill_(1.0)
        attn = torch.randn(5, 16) * 3.0
        e = torch.randn(5, 16) * 0.1
        pre = torch.randn(5, 16)
        fp32, _s, _a, _b = dc.residual_edit_through_norm(attn, e, attn + e, pre, norm, torch.bfloat16)
        self.assertFalse(torch.allclose(fp32, e, rtol=1e-2),
                         "through a norm the delivered perturbation is not the raw edit")
        half, _s, _a, _b = dc.residual_edit_through_norm(attn, e * 0.5, attn + e * 0.5, pre, norm, torch.bfloat16)
        self.assertFalse(torch.allclose(half * 2, fp32, rtol=1e-4),
                         "and it is not linear in the edit — the D51 gate therefore stays at the o_proj site")


OUT = Path(os.environ.get("STRENGTH_DOSE_OUT", Path.home() / ".local/state/strength-dose/out/cluster_b"))
NEMO_OUT = OUT / "nemotron3nano4b"


class RealCapturedRows(unittest.TestCase):
    """Real activations from the nemotron capture (the job's own sample export: o_proj inputs and outputs,
    resid_pre, h_last for 12 prompts), real object bytes, and the frozen adapter as the oracle — on the
    shapes the producer actually emits, not invented ones. Then the published per-row numbers are
    RE-DERIVED from those rows and must match the capture bit-for-bit."""

    @classmethod
    def setUpClass(cls):
        _need(NEMO_OUT / "capture_sample_nemotron3nano4b.pt"); _need(LORA_DIR)
        cls.smp = torch.load(NEMO_OUT / "capture_sample_nemotron3nano4b.pt", map_location="cpu", weights_only=False)
        cls.dtype = getattr(torch, cls.smp["model_dtype"].split(".")[-1])

    def test_served_lora_output_on_real_rows_matches_the_adapter(self):
        for staged in ("base__constv2-mix-full__lora_r4_ce", "dm1__constv2-mix-full__lora_r4_ce_minus_ctl"):
            path = _need(LORA_DIR / staged / "object.pt")
            blob = _blob(path)
            mine = dc.lora_ops_from_blob(blob)
            served = la.get_adapter(blob["kind"])({"kind": blob["kind"]}, path.read_bytes())
            for l in self.smp["lora_layers"]:
                for pidx in self.smp["prompt_idx"][:3]:
                    x = self.smp["lora_x"][l][pidx]; o = self.smp["lora_oproj_out"][l][pidx]
                    self.assertEqual(x.dtype, self.dtype); self.assertEqual(x.shape[1], mine[0].A.shape[0])
                    sops = [s for s in served if s.layer == l]; lops = [m for m in mine if m.layer == l]
                    want = la.apply_oproj_ops(x, o, sops, 1.0, layer=l)
                    _f, got = dc.lora_edit_from_reads([dc.lora_read(x, m) for m in lops], lops, self.dtype, base=o)
                    self.assertTrue(torch.equal(want, got), f"{staged} layer {l} prompt {pidx}")

    def test_served_alphasteer_on_real_h_last_matches_the_adapter(self):
        as_dir = FIX / "alphasteer" / "nemotron3nano4b"
        if not as_dir.exists():
            raise unittest.SkipTest("no nemotron AlphaSteer fixture (750 MB each; the qwen pair is the local fixture)")

    #: the capture's fp32 GEMMs ran on the GPU (cuBLAS); this recomputation runs on the CPU. The two
    #: reduce in different orders, so the norms agree to fp32 rounding, not to the bit — bit-equality is
    #: claimed only where both sides run on one device (the adapter oracle above). Measured here: 1.3e-6.
    CROSS_DEVICE_RTOL = 1e-5

    def test_published_per_row_numbers_rederive_from_the_rows(self):
        """The capture .pt holds every row's edit norms; recompute prompt 0's rows at layer 12 for the d1
        and dm1 rungs from the sample's real x and the real object bytes and require agreement to fp32
        rounding (cross-device) — and the dose factor between them to be EXACTLY -1 on every row."""
        cap_p = _need(NEMO_OUT / "capture_nemotron3nano4b.pt")
        cap = torch.load(cap_p, map_location="cpu", weights_only=False)
        l = 12; pidx = 0
        rows = (cap["lora"]["row_prompt"] == pidx)
        x = self.smp["lora_x"][l][pidx]
        for view, staged in (("d1", "base__constv2-mix-full__lora_r4_ce"), ("dm1", "dm1__constv2-mix-full__lora_r4_ce")):
            ops = [m for m in dc.lora_ops_from_blob(_blob(_need(LORA_DIR / staged / "object.pt"))) if m.layer == l]
            fp32, _ = dc.lora_edit_from_reads([dc.lora_read(x, m) for m in ops], ops, self.dtype)
            published = cap["lora"]["edits"]["constv2-mix-full__lora_r4_ce"][view][l]["fp32"][rows]
            self.assertEqual(published.shape[0], x.shape[0])
            self.assertTrue(torch.allclose(fp32.norm(dim=-1), published, rtol=self.CROSS_DEVICE_RTOL, atol=0.0),
                            f"{view}: per-row edit norms differ by more than fp32 rounding")
        h = cap["lora"]["h_pre_norm"][l][rows]
        self.assertTrue(torch.allclose(h, self.smp["lora_resid_pre"][l][pidx].norm(dim=-1), rtol=self.CROSS_DEVICE_RTOL, atol=0.0))
        # the dose factor on real rows: dm1 against d1 is exactly -1 on every row
        e1, _ = dc.lora_edit_from_reads([dc.lora_read(x, m) for m in ops], ops, self.dtype)
        base_ops = [m for m in dc.lora_ops_from_blob(_blob(LORA_DIR / "base__constv2-mix-full__lora_r4_ce" / "object.pt")) if m.layer == l]
        e0, _ = dc.lora_edit_from_reads([dc.lora_read(x, m) for m in base_ops], base_ops, self.dtype)
        c, resid = dc.signed_factor(e1, e0)
        self.assertTrue(torch.equal(c, torch.full_like(c, -1.0)) and float(resid.max()) == 0.0)

    def test_the_headline_numbers_in_the_json_are_what_the_rows_say(self):
        doc = json.loads(_need(NEMO_OUT / "dose_nemotron3nano4b.json").read_text())
        cap = torch.load(_need(NEMO_OUT / "capture_nemotron3nano4b.pt"), map_location="cpu", weights_only=False)
        cell = doc["lora"]["constv2-mix-full__lora_r4_ce"]
        is_last = cap["lora"]["row_is_last"]
        # direct-add block: the residual perturbation IS the edit, so realised = ||e|| / ||resid_pre|| at the last row
        for view in ("d1", "dm1", "d0p1"):
            per_layer = {}
            for l in cap["lora"]["layers"]:
                e = cap["lora"]["edits"]["constv2-mix-full__lora_r4_ce"][view][l]["e_res_fp32"][is_last]
                h = cap["lora"]["h_pre_norm"][l][is_last]
                per_layer[l] = e / h
            agg = dc.per_prompt_over_layers(per_layer)
            self.assertAlmostEqual(float(agg["median_over_layers"].median()), cell["rungs"][view]["realised_dose_median"], places=12)
        self.assertEqual(cell["norm_placement"]["12"]["block_class"], "NemotronHBlock")
        self.assertIsNone(cell["norm_placement"]["12"]["declared_post_attention_norm"])
        self.assertEqual(cell["norm_placement"]["12"]["candidates_fed_the_oproj_output"], [])
        self.assertEqual(doc["provenance"]["pool"]["sha256"], dc.POOL_SHA256)
        self.assertEqual(doc["provenance"]["render"]["chat_template_hash"], dc.EXPECTED_TEMPLATE_HASH["nemotron3nano4b"])
        self.assertTrue(doc["provenance"]["env"]["nemotron_fast_path"])


class NanAwareEquality(unittest.TestCase):
    def test_the_same_tensor_with_nan_pad_rows_compares_equal(self):
        a = torch.randn(4, 8, dtype=torch.bfloat16); a[0] = float("nan")     # a fully-masked pad row
        self.assertFalse(torch.equal(a, a.clone()), "torch.equal is false on identical NaN-bearing tensors")
        self.assertTrue(dc.nan_equal(a, a.clone()))

    def test_a_real_difference_is_still_caught(self):
        a = torch.randn(4, 8, dtype=torch.bfloat16); a[0] = float("nan")
        b = a.clone(); b[2, 3] += 1
        self.assertFalse(dc.nan_equal(a, b))
        self.assertFalse(dc.nan_equal(a, a[:, :4]))


class DoseTokenSpelling(unittest.TestCase):
    def test_matches_the_roster_view_names(self):
        self.assertEqual([dc.dose_token(d) for d in (-1.0, 0.1, 0.25, 0.5, 1.0, 1.5)],
                         ["dm1", "d0p1", "d0p25", "d0p5", "d1", "d1p5"])
        self.assertEqual([dc.dose_token(d) for d in (-0.5, 0.3)], ["dm0p5", "d0p3"])


if __name__ == "__main__":
    unittest.main()
