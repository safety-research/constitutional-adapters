"""The rho-emit probe (``SISTEER_LENS_RHO_EMIT``, lens_worker.RhoRecord):
an OBSERVATION-ONLY record of the served per-layer edit ratio
``rho = ||add|| / ||h||`` written at the two block-apply sites AFTER the
production write. CPU-only, over the worker's injectable view seam.

The load-bearing test is TENSOR-LEVEL BIT-IDENTITY: with a fixed batch
and shape, the steered hidden state a steered layer returns must be
byte-identical with the flag OFF and with the flag ON — across a chain of
steered layers where each hook's returned hidden is the next hook's input
(the composed stream). Random fp32 fixtures, not the integer ones the
vectorize tests use: an op-order or dtype change in the production path
would show here as a rounding difference, and the assertion is on bytes.
The comparison against the UNPATCHED tree lives in
scripts/probe_rho/rho_bitident_dump.py (a second process, a second
PYTHONPATH) — a test file cannot import two versions of one module.

The other tests pin what the record SAYS: that ``h_norm`` at a later layer
is the norm of the COMPOSED stream (the earlier layer's edit included),
not the unsteered capture — the quantity the fit side cannot see; that
``add_norm`` is the delta the production path actually wrote; that scale-0
twins and unhooked requests leave no entry; that the grouped decode path
records one row per member; and that the flag is off by default."""

import json
import os

import pytest
import torch

from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_worker as lw
from test_petri_lens_worker import (FakeLayer, FakeView, _forward, _obj,
                                    _ok_header, _registry, _sel)

D = 32  # hidden size of these fixtures (the fakes carry no d of their own)


def _alpha_ops(layers, *, d=D, seed=0, coeff=0.7):
    """AlphaSteer-shaped multi-layer ops: a random fp32 [d, d] per layer,
    small enough that the edit is a few percent of the stream."""
    g = torch.Generator().manual_seed(seed)
    return [la.HookOp(op=la.OP_PREFILL_LAST_MATMUL, layer=l,
                      matrix=torch.randn(d, d, generator=g) * 0.05,
                      coeff=float(coeff), phase=la.PHASE_PREFILL)
            for l in layers]


def _steering(registry, *, n_layers, rho_emit):
    view = FakeView(layers=[FakeLayer() for _ in range(n_layers)])
    kw = {} if rho_emit is None else {"rho_emit": rho_emit}
    st = lw.LensWorkerSteering(view=view, registry=registry, **kw)
    st.preflight(object(), header_verifier=_ok_header)
    st.install()
    return st, view


def _set_prefill_batch(view, spec):
    qsl, seq_lens, nat, pos = _forward([shape for _rid, shape, _s in spec])
    tokens = {rid: list(range(1000 * i, 1000 * i + c + q))
              for i, (rid, (q, c), _s) in enumerate(spec)}
    prompt_lens = {rid: (q if c == 0 else c) for rid, (q, c), _s in spec}
    view.set_batch([rid for rid, _shape, _s in spec],
                   {rid: sel for rid, _shape, sel in spec if sel is not None},
                   tokens, (qsl, seq_lens, nat), prompt_lens=prompt_lens)
    return qsl, nat, pos


def _chain(st, view, spec, layers, h, r):
    """Drive the steered layers IN ORDER, feeding each hook's returned
    hidden into the next hook as its layer output — the composed stream
    (a real forward's blocks sit between; identity here, which changes
    nothing about whether the bytes agree). Returns the per-layer
    returned hidden tensors."""
    _qsl, _nat, pos = _set_prefill_batch(view, spec)
    outs = []
    cur = h
    for l in layers:
        out = st.hook_body(l, (pos, cur, r), (cur, r))
        if out is None:
            new_h = cur
        else:
            new_h, out_r = out
            assert out_r is r
        outs.append(new_h)
        cur = new_h
    return outs


def _bytes(t):
    return t.detach().contiguous().numpy().tobytes()


LAYERS = (1, 2, 3, 5)
SPEC = [("req-a", (7, 0), "alpha"),
        ("req-u", (4, 0), None),
        ("req-b", (5, 0), "alpha"),
        ("req-z", (6, 0), "alpha0")]


def _fixture(seed=11):
    alpha = _obj("alpha-arm", _alpha_ops(LAYERS))
    alpha0 = _obj("alpha-arm__scale0", _alpha_ops(LAYERS), scale=0.0)
    objs = {"alpha": alpha, "alpha0": alpha0}
    spec = [(rid, shape, (None if key is None else
                          _sel(objs[key], scale=(0.0 if key == "alpha0" else None))))
            for rid, shape, key in SPEC]
    nat = sum(q for _rid, (q, _c), _s in spec)
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(nat, D, generator=g)
    r = torch.randn(nat, D, generator=g)
    return _registry(alpha, alpha0), spec, h, r


class TestBitIdentity:
    def test_flag_off_and_on_return_byte_identical_hidden_at_every_layer(self):
        reg, spec, h, r = _fixture()
        st_off, v_off = _steering(reg, n_layers=8, rho_emit=False)
        st_on, v_on = _steering(reg, n_layers=8, rho_emit=True)
        outs_off = _chain(st_off, v_off, spec, LAYERS, h.clone(), r.clone())
        outs_on = _chain(st_on, v_on, spec, LAYERS, h.clone(), r.clone())
        for l, a, b in zip(LAYERS, outs_off, outs_on):
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(a, b), f"layer {l}: values differ"
            assert _bytes(a) == _bytes(b), f"layer {l}: bytes differ"
        # the steering actually happened (this is not two no-ops agreeing)
        assert not torch.equal(outs_off[0], h)
        # the flag-off record is empty by construction; the flag-on one is
        # populated: 2 steered requests x 4 layers (scale-0 and unhooked
        # requests write nothing, so they are not observed)
        off = st_off.rho_record.as_dict()
        on = st_on.rho_record.as_dict()
        assert off["enabled"] is False and off["n_entries"] == 0 and off["entries"] == []
        assert on["enabled"] is True and on["n_entries"] == 2 * len(LAYERS)
        assert on["n_record_errors"] == 0

    def test_default_construction_matches_flag_off_bytes(self):
        """No kwarg and no env var: the unpatched behaviour, byte for byte."""
        reg, spec, h, r = _fixture()
        assert os.environ.get(lw.RHO_EMIT_ENV) is None
        st_def, v_def = _steering(reg, n_layers=8, rho_emit=None)
        st_off, v_off = _steering(reg, n_layers=8, rho_emit=False)
        assert st_def._rho_emit is False
        a = _chain(st_def, v_def, spec, LAYERS, h.clone(), r.clone())
        b = _chain(st_off, v_off, spec, LAYERS, h.clone(), r.clone())
        for x, y in zip(a, b):
            assert _bytes(x) == _bytes(y)


class TestWhatTheRecordSays:
    def _run_on(self):
        reg, spec, h, r = _fixture()
        st, view = _steering(reg, n_layers=8, rho_emit=True)
        outs = _chain(st, view, spec, LAYERS, h.clone(), r.clone())
        rec = st.rho_record.as_dict()
        return reg, spec, h, r, outs, rec

    @staticmethod
    def _rows(spec):
        qsl = [0]
        for _rid, (q, _c), _s in spec:
            qsl.append(qsl[-1] + q)
        return {rid: slice(a, b) for (rid, _s, _sel), a, b
                in zip(spec, qsl, qsl[1:])}

    def test_h_norm_is_the_composed_stream_not_the_unsteered_capture(self):
        reg, spec, h, r, outs, rec = self._run_on()
        rows = self._rows(spec)
        by = {(e["req_id"], e["layer"]): e for e in rec["entries"]}
        # layer 2's input is layer 1's RETURNED hidden (+ residual): the
        # composed stream. The record must carry ITS last-row norm ...
        last_a = rows["req-a"].stop - 1
        composed = (outs[0] + r)[last_a].norm()
        unsteered = (h + r)[last_a].norm()
        e = by[("req-a", 2)]
        assert e["h_norm_last"] == pytest.approx(float(composed), rel=1e-6)
        # ... and that is provably NOT the unsteered capture's norm — the
        # quantity the fit side cannot see.
        assert abs(float(composed) - float(unsteered)) > 1e-4
        assert e["h_norm_last"] != pytest.approx(float(unsteered), rel=1e-6)
        # the first steered layer reads the unedited stream (fit-side rho is
        # exact there, per plan B.7)
        e1 = by[("req-a", 1)]
        assert e1["h_norm_last"] == pytest.approx(float(unsteered), rel=1e-6)

    def test_add_norm_is_the_delta_the_production_path_wrote(self):
        reg, spec, h, r, outs, rec = self._run_on()
        rows = self._rows(spec)
        by = {(e["req_id"], e["layer"]): e for e in rec["entries"]}
        prev = h
        for l, out in zip(LAYERS, outs):
            for rid in ("req-a", "req-b"):
                sl = rows[rid]
                written = (out - prev)[sl]           # the delta that landed
                e = by[(rid, l)]
                assert e["path"] == lw.SELECTION_PATH_PER_SLOT
                assert e["phase"] == la.PHASE_PREFILL
                assert e["n_rows"] == sl.stop - sl.start
                assert e["requested"] == "alpha-arm" and e["applied_cell"] == "alpha-arm"
                assert e["applied_effective_scale"] == 1.0
                assert e["dtype"] == str(torch.float32)
                # alphasteer broadcasts ONE vector to every prompt row, so
                # every row's add norm is the last row's
                n_last = float(written[-1].norm())
                assert e["add_norm_last"] == pytest.approx(n_last, rel=1e-6)
                assert e["add_norm_min"] == pytest.approx(n_last, rel=1e-5)
                assert e["add_norm_max"] == pytest.approx(n_last, rel=1e-5)
                stream = (prev + r)[sl]
                hn = stream.norm(dim=-1)
                assert e["h_norm_min"] == pytest.approx(float(hn.min()), rel=1e-6)
                assert e["h_norm_max"] == pytest.approx(float(hn.max()), rel=1e-6)
                assert e["rho_last"] == pytest.approx(
                    n_last / float(hn[-1]), rel=1e-6)
                assert e["rho_min"] == pytest.approx(
                    float((written.norm(dim=-1) / hn).min()), rel=1e-5)
                assert e["rho_last"] > 0
            prev = out

    def test_scale0_twin_and_unhooked_request_leave_no_entry(self):
        _reg, _spec, _h, _r, _outs, rec = self._run_on()
        rids = {e["req_id"] for e in rec["entries"]}
        assert rids == {"req-a", "req-b"}
        assert {e["layer"] for e in rec["entries"]} == set(LAYERS)

    def test_snapshot_is_json_able_and_reset_clears(self):
        reg, spec, h, r = _fixture()
        st, view = _steering(reg, n_layers=8, rho_emit=True)
        _chain(st, view, spec, LAYERS, h, r)
        snap = st.rho_record.as_dict()
        json.dumps(snap)  # plain scalars only
        assert snap["env"] == lw.RHO_EMIT_ENV == "SISTEER_LENS_RHO_EMIT"
        assert snap["n_entries"] == len(snap["entries"]) == 8
        st.rho_record.reset()
        again = st.rho_record.as_dict()
        assert again["n_entries"] == 0 and again["entries"] == []
        assert again["enabled"] is True  # the flag is not an observation


class TestGroupedPath:
    """AlphaSteer never takes this path (prefill-only ops route per slot),
    but the patch instruments both sites; pin the grouped one too."""

    def _decode_batch(self, view, spec):
        qsl, seq_lens, nat, pos = _forward([shape for _rid, shape, _s in spec])
        tokens = {rid: list(range(1000 * i, 1000 * i + c + q))
                  for i, (rid, (q, c), _s) in enumerate(spec)}
        prompt_lens = {rid: c for rid, (q, c), _s in spec}
        view.set_batch([rid for rid, _shape, _s in spec],
                       {rid: sel for rid, _shape, sel in spec if sel is not None},
                       tokens, (qsl, seq_lens, nat), prompt_lens=prompt_lens)
        return pos

    def _fixture(self):
        g = torch.Generator().manual_seed(3)
        vec = torch.randn(D, generator=g) * 0.1
        op = la.HookOp(op=la.OP_ADD_VECTOR, layer=1, vector=vec)  # row-local
        arm = _obj("vec-arm", [op])
        spec = [("d1", (1, 9), _sel(arm)), ("d2", (1, 13), _sel(arm)),
                ("du", (1, 5), None), ("d3", (1, 7), _sel(arm))]
        h = torch.randn(4, D, generator=g)
        r = torch.randn(4, D, generator=g)
        return _registry(arm), spec, h, r, vec

    def test_group_bit_identity_and_one_entry_per_member(self):
        reg, spec, h, r, vec = self._fixture()
        outs = {}
        recs = {}
        for flag in (False, True):
            st, view = _steering(reg, n_layers=4, rho_emit=flag)
            pos = self._decode_batch(view, spec)
            out = st.hook_body(1, (pos, h.clone(), r.clone()), (h.clone(), r.clone()))
            assert out is not None
            outs[flag] = out[0]
            recs[flag] = st.rho_record.as_dict()
            sel = st.selection_record.as_dict()
            assert sel["n_modifying_grouped"] == 3  # it WAS the grouped path
        assert torch.equal(outs[False], outs[True])
        assert _bytes(outs[False]) == _bytes(outs[True])
        assert recs[False]["n_entries"] == 0
        on = recs[True]
        assert on["n_entries"] == 3
        assert {e["req_id"] for e in on["entries"]} == {"d1", "d2", "d3"}
        for e in on["entries"]:
            assert e["path"] == lw.SELECTION_PATH_GROUPED
            # a ROW-LOCAL group carries no DecodeRows context (the forward
            # skipped the plan lattice: every active op is position-
            # independent), so its phase is unrecorded — None, not "decode"
            assert e["phase"] is None
            assert e["n_rows"] == 1
            assert e["add_norm_last"] == pytest.approx(float(vec.norm()), rel=1e-6)
            assert e["add_norm_last"] == e["add_norm_min"] == e["add_norm_max"]
            assert e["h_norm_last"] == e["h_norm_median"]
            assert e["rho_last"] == pytest.approx(
                e["add_norm_last"] / e["h_norm_last"], rel=1e-6)


class TestFlag:
    def test_default_off_and_env_on(self, monkeypatch):
        reg, _spec, _h, _r = _fixture()
        monkeypatch.delenv(lw.RHO_EMIT_ENV, raising=False)
        st, _v = _steering(reg, n_layers=8, rho_emit=None)
        assert st._rho_emit is False and st.rho_record.enabled is False
        monkeypatch.setenv(lw.RHO_EMIT_ENV, "1")
        st, _v = _steering(reg, n_layers=8, rho_emit=None)
        assert st._rho_emit is True and st.rho_record.enabled is True
        monkeypatch.setenv(lw.RHO_EMIT_ENV, "0")
        st, _v = _steering(reg, n_layers=8, rho_emit=None)
        assert st._rho_emit is False

    def test_apply_path_string_is_untouched_by_the_flag(self):
        """The gate keys comparisons on apply_path; a diagnostic flag must
        not make two standups read as different paths."""
        reg, _spec, _h, _r = _fixture()
        a, _ = _steering(reg, n_layers=8, rho_emit=False)
        b, _ = _steering(reg, n_layers=8, rho_emit=True)
        assert a.apply_path == b.apply_path
