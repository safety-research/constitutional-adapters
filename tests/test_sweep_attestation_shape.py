"""Canary 370305 (2026-09-03): the budget probe read the attestation's
``scheduler_geometry`` (a dict) as the list of worker reports, iterated its KEYS,
and filed the resulting AttributeError as the first refused rung. These pin the
real attestation shape and the wall/abort distinction."""
import pytest

import served_decode_sweep as sw


ATT = {  # the shape launch_server writes (synthetic values, TESTHOST)
    "hostname": "TESTHOST", "n_workers": 1,
    "scheduler_geometry": {
        "rpc_method": "collective_rpc", "n_workers": 1, "note": "resolved engine config",
        "expected": {"max_num_seqs": 48, "max_model_len": 8192, "max_num_batched_tokens": 393216},
        "worker_reports": [{
            "max_num_seqs": 48, "max_model_len": 8192, "max_num_batched_tokens": 393216,
            "kv_num_cache_groups": 1, "kv_num_gpu_blocks": 116655, "kv_block_size": 16,
            "kv_max_concurrency": 227.84,
        }],
    },
    "worker_reports": [{"kv_num_cache_groups": 1, "kv_num_gpu_blocks": 116655, "kv_block_size": 16}],
}


def test_worker_reports_are_read_from_the_nested_dict():
    reps = sw.worker_reports_of(ATT)
    assert len(reps) == 1 and reps[0]["kv_num_gpu_blocks"] == 116655
    facts = sw.kv_facts_from_reports(reps, 8192)
    assert facts["kv_tokens"] == 116655 * 16
    assert facts["kv_kind"] == "uniform_blocks_x_block_size"


def test_worker_reports_fallbacks():
    assert sw.worker_reports_of({"scheduler_geometry": [{"kv_max_concurrency": 2.0}]})[0]["kv_max_concurrency"] == 2.0
    assert sw.worker_reports_of({"worker_reports": [{"a": 1}]}) == [{"a": 1}]
    assert sw.worker_reports_of({}) == []
    # a dict's keys are never reports
    assert sw.kv_facts_from_reports(sw.worker_reports_of({"scheduler_geometry": {"note": "x"}}), 8192)["kv_kind"] == "unreported"


def test_probe_code_errors_are_not_walls():
    probes = [
        {"product": 393216, "max_num_seqs": 48, "status": "launch_error", "error": "AttributeError"},
    ]
    assert sw.next_budget_probe(probes, [393216, 524288], 8192) == 393216  # rung not consumed
    b = sw.budget_bracket(probes)
    assert b["largest_passing"] is None and b["first_refused"] is None
    assert b["wall_reached"] is False and b["n_aborted"] == 1 and b["aborted_statuses"] == ["launch_error"]


def test_real_refusals_still_define_the_wall():
    probes = [
        {"product": 393216, "max_num_seqs": 48, "status": "pass"},
        {"product": 524288, "max_num_seqs": 64, "status": "kv_under_coverage", "error": "KV cache holds"},
    ]
    b = sw.budget_bracket(probes)
    assert b["largest_passing"]["product"] == 393216 and b["first_refused"] == 524288 and b["wall_reached"]
    # the one bisection between 48 and 64 -> 56 x 8192
    assert sw.next_budget_probe(probes, [393216, 524288], 8192) == 56 * 8192


def test_oom_is_a_wall_but_not_ready_is_not():
    assert sw.is_wall({"product": 1, "status": "oom"})
    assert not sw.is_wall({"product": 1, "status": "not_ready"})
    assert not sw.is_wall({"product": 1, "status": "launch_error"})


def test_standup_dir_is_unique_per_job(tmp_path):
    """Canary 371407: reusing standups/<tag> across jobs re-exports the read-only
    lens registry over the previous job's files -> EACCES. The dir carries the job id."""
    class Ctx:  # the two attributes standup_dir reads
        results_dir = tmp_path
        job_id = "111"
    a = sw.standup_dir(Ctx(), "budget_L8192xS48")
    Ctx.job_id = "222"
    b = sw.standup_dir(Ctx(), "budget_L8192xS48")
    assert a != b and a.parent == b.parent == tmp_path / "standups"
    assert a.name == "budget_L8192xS48__job111" and b.name == "budget_L8192xS48__job222"
    Ctx.job_id = None
    c = sw.standup_dir(Ctx(), "x")
    assert c.name.startswith("x__job")  # SLURM_JOB_ID or pid fallback, never bare


def test_vmodel_by_cell_reads_dataclass_routes_and_dict_routes():
    """Canary 372693: load_routing_table returns RouteEntry dataclasses, not dicts."""
    from sisteer.capeval.serving.proxy import RouteEntry
    entry = RouteEntry(virtual_model="petri__base", cell_id="none", cell_class="passthrough",
                       serving_identity="s", intervention_hash="h")
    assert sw.vmodel_by_cell_of({"routes": [entry]}) == {"none": "petri__base"}
    assert sw.vmodel_by_cell_of({"routes": [{"cell_id": "c", "virtual_model": "v"}]}) == {"c": "v"}

    class T:  # object-shaped table
        routes = [entry]
    assert sw.vmodel_by_cell_of(T()) == {"none": "petri__base"}
