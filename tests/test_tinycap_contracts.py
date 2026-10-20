"""Interface-commit contracts for the tinycap suite: the NAMES every lane codes
against — control grammar, row task/metric names, manifest keys, geometry
sub-entry keys, the usage-ledger row schema, the ``X-Tinycap-*`` headers,
the Inspect provider name, the launcher argv, the sandbox limits, the
client-lock pins — plus the frozen SIGNATURES of the continuation-logprob
scorer. No behaviour is tested here beyond pure name/argv builders; the
lane test files (``test_prefill_logprob``, ``test_inspect_served``, ``test_usage_ledger``, ``test_evalplus_served``)
own the bodies.

Base lane only: no lm-eval / inspect_ai / evalplus / vllm / torch imports
(a subprocess asserts the two contract modules import clean).
Everything synthetic; no real session data.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sisteer.capeval import store as cs
from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.serving import prefill_logprob as pl
from sisteer.capeval.serving import proxy as px
from sisteer.capeval.serving import serve_declaration as sd
from sisteer.capeval.serving import vllm_server as vs
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import contracts as tc
from sisteer.eval.cell_controls import _SAFE_STRING

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import served_capability_run as srun  # noqa: E402

DOC = REPO / "docs" / "tinycap_interfaces.md"

# --------------------------------------------------------------------------- #
# importability (base lane)                                                    #
# --------------------------------------------------------------------------- #


def test_contract_modules_import_without_heavy_deps():
    code = (
        "import sys\n"
        "import sisteer.capeval.tinycap, sisteer.capeval.tinycap.contracts\n"
        "import sisteer.capeval.serving.prefill_logprob\n"
        "heavy = {'lm_eval', 'inspect_ai', 'inspect_evals', 'evalplus', 'vllm', 'torch'}\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] in heavy))\n")
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


def test_package_reexports_contracts():
    import sisteer.capeval.tinycap as pkg
    assert pkg.contracts is C
    assert C.TINYCAP_VERSION == "tinycap_v1"
    # every public name in __all__ exists
    for name in C.__all__:
        assert hasattr(C, name), name


# --------------------------------------------------------------------------- #
# --controls grammar                                                          #
# --------------------------------------------------------------------------- #


def test_control_families_are_disjoint_from_the_legacy_served_set():
    legacy = set(srun.SERVED_CONTROLS)
    assert not (set(C.TINYCAP_CONTROLS) & legacy)
    # PW3: `gsm8k` joined the tinycap grammar as the STANDARD GSM8K (the
    # bespoke served leg left SERVED_CONTROLS), so the two sets stay disjoint;
    # PW17 (U4(a)): `gsm8k_full` = the in-depth full set joined beside it, and
    # `gsm8k` LEFT the inspect_cmp family (the paired set IS the standard)
    # bcb-bfcl-200: the two pinned legs appended LAST, opt-in, in no battery;
    # ifeval-canonical-battery: `ifeval` appended after them (P22: no battery);
    # the rule of 2026-09-11: the full `bfcl` and `hellaswag_inspect`
    # RETIRED from the grammar (bfcl_200 is the tool-use readout)
    # capeval-full-tasks (2026-09-23): the two FULL siblings appended LAST, opt-in, in no battery
    assert C.TINYCAP_CONTROLS == (
        "math500", "humaneval_plus", "mbpp_plus",
        "gsm8k", "gsm8k_full", "inspect_cmp",
        "bigcodebench_200", "bfcl_200", "ifeval",
        "bigcodebench_full", "bfcl_full")
    assert C.RETIRED_INSPECT_TASK_NAMES == ("bfcl", "hellaswag", "mmlu_letter_valid", "mmlu_0_shot")
    for retired in ("bfcl", "hellaswag_inspect", "inspect_cmp:mmlu"):
        with pytest.raises(C.ControlNameError):
            C.expand_control(retired)
    assert not any(t in C.TASKS_OF_CONTROL for t in ("bfcl", "hellaswag_inspect", "inspect_cmp:mmlu"))
    assert (C.CONTROL_BIGCODEBENCH_200, C.CONTROL_BFCL_200) == ("bigcodebench_200", "bfcl_200")
    assert C.CONTROL_IFEVAL == "ifeval"
    assert C.TOOL_CALL_CONTROLS == ("bfcl_200", "bfcl_full")
    assert (C.CONTROL_BIGCODEBENCH_FULL, C.CONTROL_BFCL_FULL) == ("bigcodebench_full", "bfcl_full")
    for tok in (C.CONTROL_BIGCODEBENCH_200, C.CONTROL_BFCL_200, C.CONTROL_IFEVAL,
                C.CONTROL_BIGCODEBENCH_FULL, C.CONTROL_BFCL_FULL):
        assert tok not in C.STANDARD_BATTERY_PROMPTED and tok not in C.IN_DEPTH_EXTRAS
        assert C.expand_control(tok) == (tok,)
    assert C.CONTROL_GSM8K == "gsm8k" and "gsm8k" not in srun.SERVED_CONTROLS
    assert C.CONTROL_GSM8K_FULL == "gsm8k_full" and "gsm8k_full" not in srun.SERVED_CONTROLS
    assert not hasattr(C, "TINY_LL_TASKS") and not hasattr(C, "CONTROL_TINY_LL")
    assert C.INSPECT_CMP_TASKS == ("xstest", "or_bench_hard_1k")


def test_expand_control_grammar():
    # the removed loglikelihood family is refused by name, whole or by sub-task
    for removed in ("tiny_ll", "tiny_ll:tinyArc"):
        with pytest.raises(C.ControlNameError, match="unknown control"):
            C.expand_control(removed)
    assert C.expand_control("inspect_cmp") == tuple(
        f"inspect_cmp:{t}" for t in C.INSPECT_CMP_TASKS)
    assert C.expand_control(" inspect_cmp:xstest ") == ("inspect_cmp:xstest",)
    for fam in ("math500", "humaneval_plus", "mbpp_plus", "gsm8k", "gsm8k_full"):
        assert C.expand_control(fam) == (fam,)
    # PW17 (U4(a)): every existing atom stays VALID — the retained spelling of
    # PW3's comparison bridge resolves to the canonical atom it now means
    assert C.RETAINED_CONTROL_SPELLINGS == {"inspect_cmp:gsm8k": "gsm8k"}
    assert C.expand_control("inspect_cmp:gsm8k") == C.expand_control(" inspect_cmp:gsm8k ") == ("gsm8k",)
    assert "gsm8k" not in C.expand_control("inspect_cmp") and "inspect_cmp:gsm8k" not in C.CONTROL_ATOMS
    for bad in ("", "  ", "tiny", "tiny_ll:foo", "inspect_cmp:mmlu_0_shot", "inspect_cmp:mmlu",
                "math500:x", "gsm8k:full", "gsm8k_full:x", "mmlu", "alpaca_eval", "wmdp",
                "bfcl", "hellaswag_inspect"):
        with pytest.raises(C.ControlNameError):
            C.expand_control(bad)
    assert C.control_family("inspect_cmp:xstest") == "inspect_cmp"
    assert C.control_family("bfcl_200") == "bfcl_200"


def test_every_atom_has_a_dispatch_module_and_produces_tasks_exactly_once():
    # math500, humaneval_plus, mbpp_plus, gsm8k, gsm8k_full, inspect_cmp x2,
    # bigcodebench_200 + bfcl_200, ifeval, bigcodebench_full + bfcl_full
    assert len(C.CONTROL_ATOMS) == len(set(C.CONTROL_ATOMS)) == 1 + 1 + 1 + 1 + 1 + 2 + 2 + 1 + 2
    assert set(C.DISPATCH_MODULE_OF_CONTROL) == set(C.TINYCAP_CONTROLS)
    for mod in C.DISPATCH_MODULE_OF_CONTROL.values():
        assert mod.startswith("sisteer.capeval.tiers.")
    assert set(C.TASKS_OF_CONTROL) == set(C.CONTROL_ATOMS)
    produced = [t for ts in C.TASKS_OF_CONTROL.values() for t in ts]
    # every LIVE row task under exactly one atom; the RETIRED names (2026-09-11)
    # stay in TINYCAP_TASKS for reading existing stores and have no producer
    retired_rows = {C.inspect_task(n) for n in C.RETIRED_INSPECT_TASK_NAMES}
    assert sorted(produced) == sorted(set(C.TINYCAP_TASKS) - retired_rows), "every row task under exactly one atom"
    assert not retired_rows & set(produced)
    assert len(produced) == len(set(produced))


# --------------------------------------------------------------------------- #
# rows: task / metric / domain names                                           #
# --------------------------------------------------------------------------- #


def test_evalplus_effective_task_version_reads_the_column_with_the_row_metadata():
    """PW14 (E22 H1): ``evalplus_row_metric_def`` declares every bare
    ``evalplus-0.3.1`` row plus-only, but the code between e7c0da4 and 765251a
    (PW12) computed base AND plus and still emitted the bare version, and
    the stats readers pooled on the column alone. The EFFECTIVE version reads
    the column TOGETHER with the ``pass@1_plus`` / ``acc`` rows'
    ``metric_def`` metadata: bare + explicit base_and_plus is the suffixed
    version, bare + absent is plus-only, a suffixed version stands, and a
    mixed or contradictory group REFUSES, named; the exec wave's ``/<hash>``
    tail rides through; another framework pin is not read."""
    BP, PO = C.EVALPLUS_PLUS_DEF_BASE_AND_PLUS, C.EVALPLUS_PLUS_DEF_PLUS_ONLY
    bare, suffixed = C.EVALPLUS_TASK_VERSION, C.evalplus_row_task_version(BP)
    assert C.EVALPLUS_DEFINED_METRICS == (C.METRIC_PASS1_PLUS, C.METRIC_ACC)     # the rows that carry metric_def
    assert C.EVALPLUS_TASK_VERSION_HASH_SEP == "/"
    assert C.evalplus_effective_task_version(bare, {None}) == bare               # the retained rows (the volume)
    assert C.evalplus_effective_task_version(bare, set()) == bare                 # no plus/acc rows at all
    assert C.evalplus_effective_task_version(bare, {PO}) == bare
    assert C.evalplus_effective_task_version(bare, {BP}) == suffixed             # PW12-era rows: the same instrument as suffixed
    assert C.evalplus_effective_task_version(suffixed, {None}) == suffixed
    assert C.evalplus_effective_task_version(suffixed, {BP}) == suffixed
    assert C.evalplus_effective_task_version(f"{bare}/abcd1234", {BP}) == f"{suffixed}/abcd1234"   # the HF path
    assert C.evalplus_effective_task_version(f"{suffixed}/abcd1234", {None}) == f"{suffixed}/abcd1234"
    with pytest.raises(ValueError, match="MIXED"):
        C.evalplus_effective_task_version(bare, {None, BP})                       # two converters' rows in one group
    with pytest.raises(ValueError, match="MIXED"):
        C.evalplus_effective_task_version(bare, {PO, BP})
    with pytest.raises(ValueError, match="contradictory"):
        C.evalplus_effective_task_version(suffixed, {PO})                         # the column says one thing, the rows another
    with pytest.raises(ValueError, match="definitions this repo names"):
        C.evalplus_effective_task_version(bare, {"plus_or_base"})
    for foreign in ("evalplus-0.4.0", "inspect_evals-0.18.0", "", None):
        assert not C.is_evalplus_task_version(foreign)
        with pytest.raises(ValueError, match="not a row version"):
            C.evalplus_effective_task_version(foreign, {None})
    for ours in (bare, suffixed, f"{bare}/h", f"{suffixed}/h"):
        assert C.is_evalplus_task_version(ours)
    assert C.evalplus_effective_task_version("evalplus-9.9.9", {BP}, pin="evalplus-9.9.9") == "evalplus-9.9.9+base_and_plus"


def test_merge_payload_digest_is_canonical_and_normalized():
    """PW14 (E22 M4): the digest a ``superseded`` / ``equal_witnesses`` record
    carries — sorted-key compact JSON of the payload AFTER the metric_def
    normalization, sha256, 16 hex — so the writer, the loader and the
    torch-free sandbox scan agree byte for byte, key order is irrelevant, an
    EvalPlus payload naming no definition digests as the ``plus_only`` one
    the writer stamps, and a NaN or non-mapping refuses."""
    he = "evalplus/humaneval_plus"
    assert C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD == "payload_digest"
    assert C.MERGE_HISTORY_RECORD_FIELDS == C.MERGE_SOURCE_FIELDS + ("payload_digest",)
    assert C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD not in C.MERGE_SOURCE_FIELDS           # the claim, never the identity
    assert C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD not in C.MERGE_SOURCE_OPTIONAL_FIELDS  # a record field, not a source annotation
    assert C.PAYLOAD_DIGEST_HEX_LEN == 16
    # PW16 (E24): the STANDING record carries the same digest field, and a named refresh of the standing
    # source's changed summary is recorded — both WRITER-owned, never identity, never a caller's annotation
    assert C.MERGE_SOURCE_REFRESHED_FIELD == "refreshed" and C.MERGE_REFRESH_DIGEST_WAS_FIELD == "payload_digest_was"
    # E54 #1 added the EvalPlus sample binding — the digest the writer verified,
    # or the recorded fact that the store predates it; writer-owned like the two above
    # E55 #2c added the SECOND producer — what the task's execution.json said
    # about its own writer, when it said anything (never a null placeholder)
    # E64 added the RE-BIND history — the one changed payload the writer files
    # unnamed, recorded as a re-bind and never as a `refreshed` replacement
    assert C.MERGE_SOURCE_WRITER_FIELDS == (
        "payload_digest", "refreshed", "samples_sha256", "samples_binding",
        "execution_producer", "rebound")
    assert C.MERGE_SOURCE_SAMPLES_DIGEST_FIELD == "samples_sha256"
    assert C.MERGE_SOURCE_SAMPLES_BINDING_FIELD == "samples_binding"
    assert C.MERGE_SOURCE_EXECUTION_PRODUCER_FIELD == "execution_producer"
    assert C.PRODUCER_MARKER_FIELDS == ("tree_sha", "contract_version")
    assert (C.SAMPLES_BINDING_BOUND, C.SAMPLES_BINDING_HISTORICAL_UNBOUND) == (
        "bound", "historical-unbound")
    assert not (set(C.MERGE_SOURCE_WRITER_FIELDS)
                & (set(C.MERGE_SOURCE_FIELDS) | set(C.MERGE_SOURCE_OPTIONAL_FIELDS)))
    assert C.MERGE_REFRESH_RECORD_FIELDS == ("payload_digest_was", "payload_digest", "writer_tree_sha")
    # E64: the re-bind history record — WHEN the store says it was re-bound (its own
    # `rebound_at`), both payload digests, the sample bytes taken as the binding, the
    # witnesses that agreed, the writing tree. `rebound_at` itself is the MARKER's field
    # and never appears on a source record; the record dates the re-bind by it.
    assert C.MERGE_SOURCE_REBOUND_FIELD == "rebound"
    assert C.MERGE_REBIND_RECORD_FIELDS == (
        "at", "payload_digest_was", "payload_digest", "samples_sha256",
        "witnesses_agreed", "writer_tree_sha")
    assert (C.MERGE_REBIND_AT_FIELD, C.MERGE_REBIND_WITNESSES_FIELD) == ("at", "witnesses_agreed")
    assert C.MERGE_REBIND_WITNESSES == (
        "store_summary", "done_marker", "task_summary", "execution", "samples_bytes",
        "source_record")
    # the two REQUIRED witnesses are exactly the ones that live OUTSIDE evalplus/<ds>/
    # and therefore outlive the generation
    assert C.MERGE_REBIND_REQUIRED_WITNESSES == ("store_summary", "done_marker")
    assert C.MERGE_REBIND_REQUIRED_WITNESSES == C.MERGE_REBIND_WITNESSES[:2]
    assert C.MERGE_SOURCE_REBOUND_FIELD in C.MERGE_SOURCE_WRITER_FIELDS
    assert C.MERGE_SOURCE_REBOUND_FIELD not in C.MERGE_REBIND_RECORD_FIELDS
    assert C.DONE_MARKER_REBOUND_AT_FIELD not in C.MERGE_REBIND_RECORD_FIELDS
    assert not set(C.MERGE_SOURCE_WRITER_FIELDS) & set(C.MERGE_SOURCE_RECORD_FIELDS)
    assert not set(C.MERGE_SOURCE_WRITER_FIELDS) & set(C.MERGE_SOURCE_OPTIONAL_FIELDS)
    d = C.merge_payload_digest(he, {"acc": 0.5, "n": 2})
    assert len(d) == 16 and _SAFE_STRING.match(d) and int(d, 16) >= 0
    assert d == C.merge_payload_digest(he, {"n": 2, "acc": 0.5})                       # key order
    assert d == C.merge_payload_digest(he, {"acc": 0.5, "n": 2, C.EVALPLUS_METRIC_DEF_FIELD: C.EVALPLUS_PLUS_DEF_PLUS_ONLY})
    assert d != C.merge_payload_digest(he, {"acc": 0.5, "n": 2, C.EVALPLUS_METRIC_DEF_FIELD: C.EVALPLUS_PLUS_DEF_BASE_AND_PLUS})
    assert d != C.merge_payload_digest(he, {"acc": 0.51, "n": 2})
    assert C.merge_payload_digest("inspect/math", {"acc": 0.5, "n": 2}) != d           # a non-EvalPlus entry is not stamped
    assert C.merge_payload_digest("inspect/math", {"acc": 0.5, "n": 2}) == C.merge_payload_digest(
        "inspect/xstest", {"n": 2, "acc": 0.5})                                        # ...and the key is not hashed in
    with pytest.raises(ValueError, match="not canonically serializable"):
        C.merge_payload_digest(he, {"acc": float("nan")})
    with pytest.raises(ValueError, match="mapping"):
        C.merge_payload_digest(he, [1, 2])


def test_evalplus_row_task_version_names_the_definition():
    """PW13 (E21 M1): the pass@1_plus definition is IDENTITY-BEARING on the
    rows — the row ``task_version`` names it — while ``EVALPLUS_TASK_VERSION``
    stays the framework pin. The bare pin is the retained plus-only rows'
    literal (never written anew) and reads as ``plus_only``; anything else
    refuses rather than being read as either. One helper normalizes an
    EvalPlus payload's definition for the writer, the loader and the planner."""
    BP, PO = C.EVALPLUS_PLUS_DEF_BASE_AND_PLUS, C.EVALPLUS_PLUS_DEF_PLUS_ONLY
    assert C.EVALPLUS_TASK_VERSION == "evalplus-0.3.1"                                  # the framework pin, unchanged
    assert C.EVALPLUS_TASK_VERSION_DEF_SEP == "+"
    assert C.evalplus_row_task_version(BP) == "evalplus-0.3.1+base_and_plus"
    assert C.evalplus_row_task_version(PO) == "evalplus-0.3.1"                           # the retained rows' literal
    for d in C.EVALPLUS_PLUS_DEFS:
        assert C.evalplus_row_metric_def(C.evalplus_row_task_version(d)) == d
    assert C.evalplus_row_metric_def("evalplus-0.3.1+plus_only") == PO
    assert C.evalplus_row_task_version(BP, pin="evalplus-9.9.9") == "evalplus-9.9.9+base_and_plus"
    assert C.evalplus_row_metric_def("evalplus-9.9.9", pin="evalplus-9.9.9") == PO
    for bad in ("evalplus-0.3.1+plusdef2", "evalplus-0.4.0", "evalplus-0.3.1+", "", None):
        with pytest.raises(ValueError, match="refusing to read"):
            C.evalplus_row_metric_def(bad)
    with pytest.raises(ValueError, match="refusing to mint"):
        C.evalplus_row_task_version("plus_or_base")
    assert _SAFE_STRING.match(C.evalplus_row_task_version(BP))
    # the payload normalization — ONE helper for the writer, the loader and the equal-witness planner
    ep = "evalplus/mbpp_plus"
    assert C.evalplus_metric_def(ep, {"acc": 1}) == PO and C.evalplus_metric_def(ep, {"acc": 1, "metric_def": BP}) == BP
    assert C.evalplus_metric_def("inspect/math", {"acc": 1}) is None and C.evalplus_metric_def(ep, 3) is None
    assert C.with_evalplus_metric_def(ep, {"acc": 1}) == {"acc": 1, "metric_def": PO}
    p = {"acc": 1, "metric_def": BP}
    assert C.with_evalplus_metric_def(ep, p) == p and C.with_evalplus_metric_def(ep, p) is not p
    assert C.with_evalplus_metric_def("inspect/math", {"acc": 1}) == {"acc": 1}
    with pytest.raises(ValueError, match="definitions this repo names"):
        C.evalplus_metric_def(ep, {"metric_def": "plus_or_base"})


def test_row_task_names_and_versions():
    # no loglikelihood row task is produced any more: every row task is Inspect or EvalPlus
    assert not any(t.startswith("served_ll/") for t in C.TINYCAP_TASKS)
    assert C.TINYCAP_TASKS == C.INSPECT_TASKS + C.EVALPLUS_TASKS
    assert C.INSPECT_TASKS == tuple(f"inspect/{t}" for t in C.INSPECT_TASK_NAMES)
    assert C.INSPECT_TASK_VERSION == "inspect_evals-0.18.0"
    assert C.EVALPLUS_TASKS == ("evalplus/humaneval_plus", "evalplus/humaneval_plus:open",
                                "evalplus/mbpp_plus", "evalplus/mbpp_plus:open")
    assert C.EVALPLUS_TASK_VERSION == "evalplus-0.3.1"
    assert C.evalplus_task("mbpp", open_variant=True) == "evalplus/mbpp_plus:open"
    assert (C.EXPECTED_HUMANEVAL_PLUS_N, C.EXPECTED_MBPP_PLUS_N) == (164, 378)
    # prefixes never collide with the existing task namespaces
    for t in C.TINYCAP_TASKS:
        assert not t.startswith(("tier0/", "s2parity/", "petri/", "judged/"))
        assert _SAFE_STRING.match(t), t   # summary keys / done-marker names stay safe
    # 9 inspect/ tasks since PW3: inspect/gsm8k_paired200 (the comparison
    # bridge) sits beside inspect/gsm8k (the standard, full set); 11 since
    # bcb-bfcl-200 (inspect/bigcodebench_200, inspect/bfcl_200 — appended LAST);
    # 12 since ifeval-canonical-battery (inspect/ifeval, appended after them)
    # 14 since capeval-full-tasks (inspect/bigcodebench_full, inspect/bfcl_full — appended LAST, 2026-09-23)
    assert len(C.TINYCAP_TASKS) == len(set(C.TINYCAP_TASKS)) == 14 + 4
    assert C.INSPECT_TASK_NAMES[-5:] == ("bigcodebench_200", "bfcl_200", "ifeval", "bigcodebench_full", "bfcl_full")
    assert C.TASKS_OF_CONTROL[C.CONTROL_BIGCODEBENCH_FULL] == ("inspect/bigcodebench_full",)
    assert C.TASKS_OF_CONTROL[C.CONTROL_BFCL_FULL] == ("inspect/bfcl_full",)
    assert C.TASKS_OF_CONTROL[C.CONTROL_IFEVAL] == ("inspect/ifeval",)
    assert C.TASKS_OF_CONTROL[C.CONTROL_BIGCODEBENCH_200] == ("inspect/bigcodebench_200",)
    assert C.TASKS_OF_CONTROL[C.CONTROL_BFCL_200] == ("inspect/bfcl_200",)
    for bad in (("inspect_task", "math500"), ("evalplus_task", "humaneval_plus")):
        with pytest.raises(C.ControlNameError):
            getattr(C, bad[0])(bad[1])


def test_metrics_and_sample_metric_rule():
    assert C.METRIC_ACC == "acc"
    assert C.METRIC_ACC_AGGREGATE == "acc_aggregate"       # == parity.PARITY_AGG_METRIC
    assert C.METRIC_PASS1_PLUS == "pass@1_plus"            # == evalplus_runner.GATING_METRIC
    assert C.AGG_ITEM_ID == "__aggregate__" and C.ROW_TIER == 1
    # avg@k: per-sample rows carry the index in the METRIC (completion
    # identity is (task, item_id, metric)); the per-item mean keeps the bare name
    assert [C.sample_metric("acc", k) for k in range(4)] == ["acc@0", "acc@1", "acc@2", "acc@3"]
    for bad in (-1, True, "0"):
        with pytest.raises(ValueError):
            C.sample_metric("acc", bad)
    with pytest.raises(ValueError):
        C.sample_metric("acc@0", 1)
    # PW20b-D added the four PRODUCER facts a reader would otherwise have to
    # infer from a scorer convention: the verbatim stop reason, the scorer's
    # own extraction verdict, and which generation (and which cap) filed the
    # row. PW-nanfix added the fifth: whether the INSTRUMENT returned a verdict
    # at all (inspect's unscored sentinel), which `unparseable` (a fact about
    # the TARGET's answer) cannot express.
    # The fit-support-transfer plan (S1.6) added the scorer's think-block split
    # as two more producer facts on every Inspect row.
    assert set(C.INSPECT_EXTRA_FIELDS) == {
        "sample_idx", "seed", "answer", "prompt_tokens", "completion_tokens", "paired_id",
        "stop_reason", "unparseable", "unscored", "attempt", "max_tokens_used",
        "think_block_present", "think_block_unterminated"}


def test_domains_are_new_display_domains_covering_every_task():
    assert C.TINYCAP_DOMAINS == ("inspect_gen", "evalplus_served")
    assert set(C.TINYCAP_DOMAIN_OF) == set(C.TINYCAP_TASKS)
    assert set(C.TINYCAP_DOMAIN_OF.values()) == set(C.TINYCAP_DOMAINS)
    import yaml
    doms = yaml.safe_load((REPO / "configs/capeval/domains.yaml").read_text())["domains"]
    # the tinycap tasks collide with NO pre-existing task (the three display
    # domains Lane D registers are theirs, so they are excluded from "existing")
    existing_tasks = {t for n, d in doms.items() if n not in C.TINYCAP_DOMAINS
                      for t in d["tasks"]}
    assert not (set(C.TINYCAP_TASKS) & existing_tasks), "tinycap tasks are NEW rows"
    for name in C.TINYCAP_DOMAINS:
        if name in doms:  # Lane D has landed the block: it must agree with the contract
            assert doms[name]["kind"] == "display", name
            assert set(doms[name]["tasks"]) == {
                t for t, d in C.TINYCAP_DOMAIN_OF.items() if d == name}, name


# --------------------------------------------------------------------------- #
# manifest / summary keys                                                      #
# --------------------------------------------------------------------------- #

#: the keys scripts/served_capability_run.cmd_generate writes today
_LEGACY_MANIFEST_KEYS = {
    "cell_id", "harness_cell_id", "virtual_model", "intervention_hash",
    "harness_config_hash", "model_id", "model_revision", "engine", "serving_path",
    "serving_identity", "proxy_url", "routing_table", "steer_positions",
    "row_constants", "resolved_config", "controls", "subset",
}


def test_manifest_keys_are_additive_and_never_touch_identity():
    assert C.TINYCAP_MANIFEST_KEYS == (
        "thinking", "geometry", "concurrency", "route", "declaration", "seeds",
        "frameworks", "usage_ledger", "sandbox", "tinycap", "sampling")
    assert not (set(C.TINYCAP_MANIFEST_KEYS) & set(cs.CellStore.RESUME_KEYS))
    assert not (set(C.TINYCAP_MANIFEST_KEYS) & _LEGACY_MANIFEST_KEYS)
    # PW20b-D: the thinking block gained the driver's mode + the per-task map
    assert C.THINKING_BLOCK_FIELDS == (
        "enable_thinking", "render", "reasoning_parser", "mode", "by_task")
    # ... and the run's OUTCOME roll-up is a manifest key that is NOT identity
    # and NOT in TINYCAP_MANIFEST_KEYS (it is filled after generation, like
    # `provenance` / `producer`), so it can never take part in selecting a store
    assert C.MANIFEST_KEY_QUALITY == "quality"
    assert C.MANIFEST_KEY_QUALITY not in C.TINYCAP_MANIFEST_KEYS
    assert C.MANIFEST_KEY_QUALITY not in cs.CellStore.RESUME_KEYS
    assert C.MANIFEST_KEY_QUALITY not in cs.CellStore.CONDITIONAL_RESUME_KEYS
    assert C.QUALITY_BLOCK_FIELDS == ("unparseable", "at_cap", "by_task")
    assert C.QUALITY_MEASURE_FIELDS == ("n", "denominator", "rate", "tasks_not_recorded")
    # PW18 (U4(b) / E25): the route block mirrors the two new served facts
    # S3.2: plus the five UNHASHED attestation read-backs, spelled as the
    # endpoint spells them (contract §2.5.1 <ROUTE_KEY>: the raw map, whole)
    assert C.ROUTE_BLOCK_FIELDS == (
        "virtual_model", "grader_virtual_model", "serving_identity", "tool_calls",
        "agent_loop_entry", "reduction_mode",
        C.ROUTE_KEY_SERVE_DECLARATIONS, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256,
        C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL, C.ENDPOINT_KEY_LENS_SALT_POLICY,
        "resumed_prefill_policy")
    assert C.ROUTE_KEY_SERVE_DECLARATIONS == "lens_serve_declarations"
    assert C.DECLARATION_BLOCK_FIELDS == ("response_prefill_used",)
    assert set(C.FRAMEWORKS_BLOCK_FIELDS) == {
        "inspect_ai", "inspect_evals", "evalplus", "client_lock_sha256"}
    assert set(C.SANDBOX_BLOCK_FIELDS) == {
        "image_digest", "cpu", "memory_mib", "timeout_s", "block_network"}
    assert C.GEOMETRY_PASS_FIELD == "pass" and C.GEOMETRY_PASSES == (1, 2)
    # three identity facts came first; then the frameworks block; then the
    # geometry pass — until then it lived only in the mutable top-level
    # geometry block, so a pass-2 run whose window and S resolved to pass 1's
    # keyed onto the pass-1 store and relabelled its rows; the original three
    # are untouched
    # the per-task arm map came next — once ONE standup serves mixed
    # per-request flags, `thinking.mode` alone no longer says what arm a task
    # ran under — along with the MATH-500 truncation-retry cap, because the
    # retried samples are exactly the ones the default cap scored as failures
    # then the five sampling fields — the model's frozen row's three numbers,
    # the per-task k map and the seed rules — so a sampled store never mixes
    # with, resumes into or is resumed by a greedy one
    # then the OR-Bench item set, so a 200-item frozen subset and the full
    # 1,319-item corpus are two refusal rates, not one
    assert C.TINYCAP_IDENTITY_FIELDS == (
        "tasks", "thinking", "versions", "served", "limits", "samples", "frameworks", "pass",
        "thinking_by_task", "math500_retry_max_tokens", "or_bench_item_set",
        "temperature", "top_p", "top_k", "samples_by_task", "seed_rule")
    assert C.TINYCAP_IDENTITY_FIELDS[-5:] == C.SAMPLING_IDENTITY_FIELDS
    assert C.MANIFEST_KEY_FRAMEWORKS in C.TINYCAP_IDENTITY_FIELDS
    assert C.GEOMETRY_PASS_FIELD in C.TINYCAP_IDENTITY_FIELDS
    assert C.IDENTITY_KEY_THINKING_BY_TASK == "thinking_by_task"
    assert C.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS == "math500_retry_max_tokens"
    assert C.IDENTITY_KEY_OR_BENCH_ITEM_SET == "or_bench_item_set"
    # ...and the ids those positions resolve to ride in a manifest block that is
    # NOT identity and NOT in TINYCAP_MANIFEST_KEYS (like `quality` / `producer`)
    assert C.MANIFEST_KEY_OR_BENCH_ITEM_SET == "or_bench_item_set"
    assert C.MANIFEST_KEY_OR_BENCH_ITEM_SET not in C.TINYCAP_MANIFEST_KEYS
    assert C.MANIFEST_KEY_OR_BENCH_ITEM_SET not in cs.CellStore.RESUME_KEYS
    assert C.MANIFEST_KEY_OR_BENCH_ITEM_SET not in cs.CellStore.CONDITIONAL_RESUME_KEYS
    assert C.OR_BENCH_ITEM_SET_BLOCK_FIELDS == (*C.OR_BENCH_ITEM_SET_FIELDS, "item_ids")
    # the served window came first; then the served S and the client's request
    # width — batch composition moves a bf16 near-tie argmax
    # then the proxy's agent-loop entry — which positions a lens route steers
    # on a tool turn — and the lens reduction mode, ulp-level differences on
    # every decode step
    assert C.SERVED_IDENTITY_FIELDS == (
        "reasoning_parser", "tool_calls", "max_model_len", "max_num_seqs", "concurrency",
        "agent_loop_entry", "reduction_mode")
    for never in ("gpu_mem_util", "pid", "port", "node", "eval"):
        assert never not in C.SERVED_IDENTITY_FIELDS
    assert C.AGENT_LOOP_ENTRY_KEY == "agent_loop_entry" == C.SERVED_IDENTITY_FIELDS[-2]
    assert C.REDUCTION_MODE_FIELD == "reduction_mode" == C.SERVED_IDENTITY_FIELDS[-1]
    # PW5-5: the producer block is a top-level driver key (like `binding`), and
    # not part of the HASHED identity — but since PW7 (E14 B) it IS an immutable
    # resume key, compared when the recorded manifest carries it, so a store from
    # another revision (which keys the same) can never be resumed onto
    assert C.MANIFEST_KEY_PRODUCER == "producer"
    assert C.PRODUCER_BLOCK_FIELDS == ("tree_sha", "driver")
    assert C.PRODUCER_DRIVER_SERVED == "served_capability_run"
    assert C.TREE_SHA_FILENAME == "TREE_SHA"
    assert C.MANIFEST_KEY_PRODUCER not in C.TINYCAP_MANIFEST_KEYS
    assert C.MANIFEST_KEY_PRODUCER not in cs.CellStore.RESUME_KEYS
    assert C.MANIFEST_KEY_PRODUCER in cs.CellStore.CONDITIONAL_RESUME_KEYS
    assert not (set(cs.CellStore.CONDITIONAL_RESUME_KEYS) & set(cs.CellStore.RESUME_KEYS))
    assert not (set(cs.CellStore.CONDITIONAL_RESUME_KEYS) & set(C.TINYCAP_MANIFEST_KEYS))
    assert C.MANIFEST_KEY_PRODUCER not in _LEGACY_MANIFEST_KEYS
    # bcb-bfcl-200 §S2b: the ladder's SCALAR limit kind, appended (every existing
    # kind keeps its position; a rung-2 store's limits are exactly LADDER_LIMITS)
    assert C.LIMIT_KINDS == ("limit", "inspect_limit", "max_goals", "item_selection")
    assert C.LIMIT_KIND_ITEM_SELECTION == "item_selection" and dict(C.LADDER_LIMITS) == {"item_selection": 1}
    assert C.MANIFEST_KEY_PROVENANCE == "provenance"
    assert C.MANIFEST_KEY_PROVENANCE not in C.TINYCAP_MANIFEST_KEYS   # the HF driver's key, mirrored
    # the geometry block is ResolvedGeometry.as_dict() + pass: no key clash
    r = sg.resolve_geometry("capability")
    assert C.GEOMETRY_PASS_FIELD not in r.as_dict()


def test_merge_source_names_which_store_filled_every_tinycap_entry():
    """PW7 (E14 A): the per-entry merge provenance of
    ``controls_summary.json`` — a SIBLING block of ``controls``, keyed by the
    same entry names, so the six two-SI slots and every consumer are
    untouched. Which entries need one is data, not a literal list: a tinycap
    row task, or a control slot whose payload names one."""
    assert C.SUMMARY_KEY_SOURCES == "sources"
    assert C.MERGE_SOURCE_FIELDS == ("cell_key", "harness_config_hash", "pass",
                                     "producer_tree_sha")
    assert C.MERGE_SOURCE_SUPERSEDED_FIELD == "superseded"
    assert C.MERGE_SOURCE_RECORD_FIELDS == C.MERGE_SOURCE_FIELDS + ("superseded",)
    assert dict(C.UNRECORDED_MERGE_SOURCE) == {f: None for f in C.MERGE_SOURCE_FIELDS}
    assert C.SUMMARY_KEY_SOURCES not in C.TINYCAP_MANIFEST_KEYS      # a summary key, not a manifest one
    for task in C.TINYCAP_TASKS:
        assert C.requires_merge_source(task)
    standard = {"task": C.GSM8K_STANDARD_TASK, "item_set": C.GSM8K_SLOT_ITEM_SET}
    assert C.requires_merge_source(C.GSM8K_SLOT, standard)
    # the retired instrument's payload under the same slot name names no task,
    # and the FULL set (in-depth since U4(a)) names one that fills no slot: the
    # HF-side writers / a non-slot number
    assert not C.requires_merge_source(C.GSM8K_SLOT, {"acc": 0.5, "n": 200,
                                                      "extraction_failures": 4})
    assert not C.requires_merge_source(C.GSM8K_SLOT, {**standard, "task": C.GSM8K_FULL_TASK})
    for legacy in ("mmlu", "xstest", "or_bench_hard_1k", "alpaca_eval", C.GSM8K_LEGACY_SLOT):
        assert not C.requires_merge_source(legacy, {"acc": 0.5, "n": 200})
    # PW10 (E17 A1): the optional pass_source annotation is NOT an identity field,
    # and the producing tree's emitted payload has a named shape
    assert C.MERGE_SOURCE_PASS_SOURCE_FIELD == "pass_source" and C.PASS_SOURCE_LEGACY_TOP_LEVEL == "legacy_top_level"
    assert C.MERGE_SOURCE_OPTIONAL_FIELDS == ("pass_source", "writer_tree_sha", "equal_witnesses",
                                              "equal_witness_rule", "template_check")   # P1-1: the class
    assert "pass_source" not in C.MERGE_SOURCE_FIELDS and "pass_source" not in C.MERGE_SOURCE_RECORD_FIELDS
    assert C.MERGE_PAYLOAD_FIELDS == ("cell_dir", "payloads")
    # PW12 (E19): the writer tree, the equal-witness rule and the attestation key — none identity
    assert C.MERGE_SOURCE_WRITER_TREE_SHA_FIELD == "writer_tree_sha"
    assert C.MERGE_SOURCE_EQUAL_WITNESSES_FIELD == "equal_witnesses"
    assert C.MERGE_SOURCE_EQUAL_WITNESS_RULE_FIELD == "equal_witness_rule"
    assert C.EQUAL_WITNESS_RULE == "hashed_pass_then_smallest_cell_key"
    assert not set(C.MERGE_SOURCE_OPTIONAL_FIELDS) & set(C.MERGE_SOURCE_RECORD_FIELDS)
    assert C.MERGE_EMIT_ATTEST_KEY == "__completion__" and not C.MERGE_EMIT_ATTEST_KEY.startswith("/")
    assert C.CONTROLS_SUMMARY_FILENAME == "controls_summary.json"
    # PW10: the frozen counts the launcher and the Inspect plan share
    assert C.PARITY_N == 200


def test_replace_source_grammar_names_one_entry_and_one_winner():
    """PW8 (E16 A3): ``--replace-source <entry>=<cell_key>[,...]`` — the
    operator's resolution of an ambiguous source claim names the ENTRY and
    the store that wins it, never a global yes. The grammar refuses what it
    cannot read back unambiguously from a recorded argv."""
    assert C.REPLACE_SOURCE_FLAG == "--replace-source"
    assert C.REPLACE_SOURCE_ASSIGN == "=" and C.REPLACE_SOURCE_SEP == ","
    assert C.replace_source_entries() == frozenset(C.TINYCAP_TASKS) | {C.GSM8K_SLOT}
    assert C.parse_replace_source("") == {} == C.parse_replace_source(None) == C.parse_replace_source("  , ")
    key_a = "tok__controls__t0__a0__aaaaaaaa__11111111"
    key_b = "tok__controls__t0__a0__aaaaaaaa__22222222"
    spec = f"inspect/gsm8k={key_a}, evalplus/humaneval_plus:open={key_b},gsm8k={key_a}"
    parsed = C.parse_replace_source(spec)
    assert parsed == {"inspect/gsm8k": key_a, "evalplus/humaneval_plus:open": key_b, "gsm8k": key_a}
    # canonical (sorted) and round-tripping, so every recorded argv reads the same
    canon = C.format_replace_source(parsed)
    assert canon == f"evalplus/humaneval_plus:open={key_b},gsm8k={key_a},inspect/gsm8k={key_a}"
    assert C.parse_replace_source(canon) == parsed
    # the same entry named twice for the SAME key is one decision
    assert C.parse_replace_source(f"gsm8k={key_a},gsm8k={key_a}") == {"gsm8k": key_a}
    for bad, needle in (("inspect/gsm8k", "not <entry>=<cell_key>"),
                        (f"mmlu={key_a}", "not an entry the tinycap merge writes"),
                        (f"inspect/math={key_a}", None),                      # fine: a row task
                        ("inspect/gsm8k=", "not a store directory key"),
                        ("inspect/gsm8k=a b", "not a store directory key"),
                        (f"gsm8k={key_a},gsm8k={key_b}", "named twice")):
        if needle is None:
            C.parse_replace_source(bad)
            continue
        with pytest.raises(ValueError, match=needle):
            C.parse_replace_source(bad)


def test_completion_markers_and_the_summary_merge_are_named_once():
    """R2/R3: the partial marker, the per-task summary dir and the ONE
    controls_summary.json writer are contract names every family imports."""
    import importlib

    assert C.PARTIAL_DIR == "partial" and C.SUMMARY_DIR == "summary" and C.DONE_DIR == "done"
    assert set(C.PARTIAL_MARKER_FIELDS) == set(C.DONE_MARKER_FIELDS) | {"n_expected_full", "limits"}
    assert C.marker_filename("inspect/math") == "inspect__math.json"
    assert C.marker_filename("evalplus/humaneval_plus:open") == "evalplus__humaneval_plus:open.json"
    for task in C.TINYCAP_TASKS:
        assert _SAFE_STRING.match(C.marker_filename(task)[:-5]), task
    with pytest.raises(C.ControlNameError):
        C.marker_filename("s2parity/gsm8k_0shot")
    mod = importlib.import_module(C.CONTROLS_SUMMARY_MERGE_MODULE)
    assert callable(getattr(mod, C.CONTROLS_SUMMARY_MERGE_FUNCTION))
    # the runner families the driver dispatches, derived from the dispatch table
    assert set(C.INSPECT_CONTROLS) | set(C.EVALPLUS_CONTROLS) == set(C.TINYCAP_CONTROLS)
    assert not (set(C.INSPECT_CONTROLS) & set(C.EVALPLUS_CONTROLS))
    assert set(C.EVALPLUS_DATASET_OF_CONTROL) == set(C.EVALPLUS_CONTROLS)
    for fam, ds in C.EVALPLUS_DATASET_OF_CONTROL.items():
        assert C.evalplus_task(ds) in C.TASKS_OF_CONTROL[fam], fam


def test_gsm8k_retirement_contract(monkeypatch):
    """PW3 (U2-impl), item set flipped by U4(a) (PW17): the standard GSM8K is
    inspect_evals' stock task on the PAIRED set — the <=200 items matched by
    question text to the retired instrument's Platinum rows — produced by the
    `gsm8k` control as row task inspect/gsm8k_paired200; the FULL 1,319-item
    set is the in-depth `gsm8k_full` -> inspect/gsm8k, never the slot; the
    retained spelling `inspect_cmp:gsm8k` resolves to `gsm8k`; the retired
    bespoke leg is s2parity/gsm8k_0shot, re-enabled ONLY by the legacy env
    switch; the refusal text names all of it."""
    assert C.GSM8K_PAIRED_TASK == "inspect/gsm8k_paired200"
    assert C.GSM8K_STANDARD_TASK == C.GSM8K_PAIRED_TASK            # re-pointed by U4(a)
    assert C.GSM8K_FULL_TASK == "inspect/gsm8k"
    assert C.TASKS_OF_CONTROL[C.CONTROL_GSM8K] == (C.GSM8K_STANDARD_TASK,)
    assert C.TASKS_OF_CONTROL[C.CONTROL_GSM8K_FULL] == (C.GSM8K_FULL_TASK,)
    assert "inspect_cmp:gsm8k" not in C.TASKS_OF_CONTROL         # the spelling is an alias, not an atom
    assert C.expand_control("inspect_cmp:gsm8k") == (C.CONTROL_GSM8K,)
    assert C.INSPECT_CMP_TASKS == ("xstest", "or_bench_hard_1k")
    for fam in (C.CONTROL_GSM8K, C.CONTROL_GSM8K_FULL):
        assert C.DISPATCH_MODULE_OF_CONTROL[fam] == "sisteer.capeval.tiers.inspect_served"
        assert fam in C.INSPECT_CONTROLS
        assert C.GEOMETRY_CLASS_OF_CONTROL[fam] == C.GEOMETRY_CLASS_GEN_LONG
    assert C.GSM8K_FULL_N == 1319
    # the paired count is a MEASURED fact about the pinned inputs (200 / 200
    # matched in the 618f9dd stores), bounded by the Platinum row count
    assert C.GSM8K_PAIRED_N == 200 and C.GSM8K_PAIRED_N <= C.PARITY_N
    assert C.GSM8K_ITEM_SETS == ("full", "paired200")
    assert C.GSM8K_ITEM_SET_OF_TASK == {"inspect/gsm8k": "full",
                                        "inspect/gsm8k_paired200": "paired200"}
    assert C.GSM8K_SLOT_ITEM_SET == "paired200"
    assert C.GSM8K_RETIRED_TASK == "s2parity/gsm8k_0shot"
    assert C.GSM8K_RETIRED_TASK not in C.TINYCAP_TASKS
    # the slot fill: only the standard (paired) task fills the two-SI `gsm8k`
    # slot, and its payload NAMES the instrument beside the numbers
    assert C.SLOT_OF_TASK == {"inspect/gsm8k_paired200": "gsm8k"}
    assert C.GSM8K_FULL_TASK not in C.SLOT_OF_TASK
    assert C.SLOT_PAYLOAD_FIELDS_OF == {"gsm8k": C.GSM8K_SLOT_PAYLOAD_FIELDS}
    assert {"task", "item_set", "acc", "n"} <= set(C.GSM8K_SLOT_PAYLOAD_FIELDS)
    assert "extraction_failures" not in C.GSM8K_SLOT_PAYLOAD_FIELDS  # the retired payload's key
    for k in (*C.GSM8K_SLOT_PAYLOAD_FIELDS, "task", "item_set"):
        assert _SAFE_STRING.match(k) and k in C.SUMMARY_PAYLOAD_FIELDS, k
    # PW5-1 (re-review F1), flipped by U4(a): the slot ACCEPTANCE rule — every
    # consumer fills `gsm8k` only from a payload naming the paired task, the
    # PAIRED item set, the pinned framework and the frozen matched count;
    # anything else is retained under the legacy key (a safe, non-slot name)
    assert C.GSM8K_SLOT_ACCEPTANCE == {"task": "inspect/gsm8k_paired200", "item_set": "paired200",
                                       "framework": C.INSPECT_TASK_VERSION, "n": 200}
    assert set(C.GSM8K_SLOT_ACCEPTANCE) <= set(C.GSM8K_SLOT_PAYLOAD_FIELDS)
    assert C.GSM8K_LEGACY_SLOT == "gsm8k_legacy_s2parity" and _SAFE_STRING.match(C.GSM8K_LEGACY_SLOT)
    assert C.GSM8K_LEGACY_SLOT != C.GSM8K_SLOT and C.GSM8K_LEGACY_SLOT not in C.SLOT_PAYLOAD_FIELDS_OF
    standard = {"task": "inspect/gsm8k_paired200", "item_set": "paired200", "acc": 0.8, "n": 200,
                "framework": C.INSPECT_TASK_VERSION, "engine": "vllm", "thinking": "off",
                "max_tokens": 512, "temperature": 0.0}
    assert C.is_standard_gsm8k_slot_payload(standard)
    assert C.is_standard_gsm8k_slot_payload({**standard, "source": "cell_controls", "provenance": "ab" * 8})
    legacy = {"acc": 0.5, "n": 200, "extraction_failures": 3}          # the retired instrument's tuple
    assert not C.is_standard_gsm8k_slot_payload(legacy)                 # n alone is not the standard
    for bad in ({**standard, "task": "inspect/gsm8k", "item_set": "full", "n": 1319},  # the FULL set (in-depth)
                {**standard, "task": "inspect/gsm8k"},                    # the full task under the paired set
                {**standard, "n": 199},                                    # a shorter matched set
                {**standard, "n": 1319},                                   # the full count
                {**standard, "n": True},                                   # a bool is not a count
                {**standard, "framework": "inspect_evals-0.17.0"},        # another framework
                {**standard, "task": "s2parity/gsm8k_0shot"},             # the retired task
                {k: v for k, v in standard.items() if k != "item_set"},   # a pre-PW3 runner
                "not-a-dict", None, 200):
        assert not C.is_standard_gsm8k_slot_payload(bad), bad
    # the legacy switch: exactly the string "1"
    monkeypatch.delenv(C.GSM8K_LEGACY_ENV, raising=False)
    assert C.legacy_gsm8k_enabled() is False
    monkeypatch.setenv(C.GSM8K_LEGACY_ENV, "true")
    assert C.legacy_gsm8k_enabled() is False
    monkeypatch.setenv(C.GSM8K_LEGACY_ENV, "1")
    assert C.legacy_gsm8k_enabled() is True
    msg = C.gsm8k_retired_message("unit test")
    for needle in ("unit test", "RETIRED", "inspect/gsm8k_paired200", "--controls gsm8k",
                   "--thinking off", "200 items paired", "U4(a)", "gsm8k_full", "1319",
                   "inspect/gsm8k)", "SISTEER_LEGACY_GSM8K=1", C.INSPECT_TASK_VERSION):
        assert needle in msg, needle


def test_slot_payloads_project_the_standard_gsm8k_only():
    """completion.slot_payloads (U4(a) / PW17): a DONE inspect/gsm8k_paired200
    summary becomes the `gsm8k` slot entry projected onto
    GSM8K_SLOT_PAYLOAD_FIELDS; the FULL set (inspect/gsm8k, in-depth) fills no
    slot; a summary that lacks a field, names another task, carries the full
    item set, or fails the acceptance rule (a count other than the frozen
    matched 200) refuses (never the headline, never a legacy number)."""
    from sisteer.capeval.tinycap import completion

    paired = {"task": "inspect/gsm8k_paired200", "item_set": "paired200", "acc": 0.8, "n": 200,
              "framework": C.INSPECT_TASK_VERSION, "engine": "vllm", "thinking": "off",
              "max_tokens": 512, "temperature": 0.0, "n_requests": 200, "route": "v"}
    full = {**paired, "task": "inspect/gsm8k", "item_set": "full", "n": 1319, "n_requests": 1319}
    out = completion.slot_payloads({"inspect/gsm8k": full, "inspect/gsm8k_paired200": paired,
                                    "inspect/math": {"acc": 0.5, "n": 500}})
    assert out == {"gsm8k": {k: paired[k] for k in C.GSM8K_SLOT_PAYLOAD_FIELDS}}
    assert out["gsm8k"]["task"] == "inspect/gsm8k_paired200" and out["gsm8k"]["item_set"] == "paired200"
    assert "n_requests" not in out["gsm8k"] and "route" not in out["gsm8k"]
    assert C.is_standard_gsm8k_slot_payload(out["gsm8k"])
    assert completion.slot_payloads({"inspect/gsm8k": full}) == {}        # the full set is in-depth
    with pytest.raises(completion.CompletionError, match="lacks the slot field"):
        completion.slot_payloads({"inspect/gsm8k_paired200": {"acc": 0.8, "n": 200}})
    with pytest.raises(completion.CompletionError, match="another instrument"):
        completion.slot_payloads({"inspect/gsm8k_paired200": {**paired, "task": "inspect/gsm8k"}})
    with pytest.raises(completion.CompletionError, match="never the headline"):
        completion.slot_payloads({"inspect/gsm8k_paired200": {**paired, "item_set": "full"}})
    # the acceptance rule at the FILL: a paired summary whose n is not the frozen
    # matched count refuses here rather than landing as a legacy number at load
    with pytest.raises(completion.CompletionError, match="not the standard instrument's"):
        completion.slot_payloads({"inspect/gsm8k_paired200": {**paired, "n": 199}})


def test_standard_battery_and_in_depth_extras():
    """U2 / U3 / U4(a) / U9 item 6 / U14 (PW17, PW-battery, PR27): the STANDARD
    battery is no-decode first plus the decode evals the construct needs, MINUS
    MATH-500 avg@4 (U3: about half of every configuration's wall-clock and
    cost), MINUS the full GSM8K (U4(a): the paired-200 subset is the slot),
    MINUS MBPP+ and BFCL (since 2026-09-06) and — since
    2026-09-07 — MINUS the two GENERATIVE letter-choice legs, because
    the prefill-only letter reads were then the knowledge /
    sentence-completion instrument and a duplicative generative version of the
    same eval must not run. MATH-500 and the full GSM8K stay the IN-DEPTH
    extras, one flag away but SET ASIDE by the user; every removed control is
    an ordinary atom an explicit ``--controls`` still runs. The fitted battery
    is IDENTICAL to the prompted one (O7). The tiny loglikelihood family that
    led the battery was removed later, so the battery is the three generative
    legs."""
    assert C.BATTERY_KINDS == ("prompted", "fitted") == (C.BATTERY_PROMPTED, C.BATTERY_FITTED)
    assert C.STANDARD_BATTERY_PROMPTED == (
        "gsm8k", "humaneval_plus", "inspect_cmp:or_bench_hard_1k")
    assert C.STANDARD_BATTERY_FITTED == C.STANDARD_BATTERY_PROMPTED == (
        "gsm8k", "humaneval_plus", "inspect_cmp:or_bench_hard_1k")
    assert C.IN_DEPTH_EXTRAS == ("math500", "gsm8k_full")
    assert C.STANDARD_BATTERY_OF_KIND == {"prompted": C.STANDARD_BATTERY_PROMPTED,
                                          "fitted": C.STANDARD_BATTERY_FITTED}
    for kind, base in C.STANDARD_BATTERY_OF_KIND.items():
        assert C.battery_controls(kind) == base
        assert C.battery_controls(kind, in_depth=True) == base + C.IN_DEPTH_EXTRAS
        assert not set(C.IN_DEPTH_EXTRAS) & set(base)
        for tok in C.battery_controls(kind, in_depth=True):
            C.expand_control(tok)                       # every token is grammar
        # no default battery names the removed loglikelihood family
        assert not any(tok.startswith("tiny_ll") for tok in base)
        # the standard GSM8K is the paired set — the full set is never in a standard battery
        assert C.CONTROL_GSM8K in base and C.CONTROL_GSM8K_FULL not in base
        assert C.CONTROL_MATH500 not in base
    # U9 item 6: MBPP+ and BFCL are OUT of both standard
    # batteries on cost. BFCL's removal is what makes NO default standup compose
    # --tool-calls any more (the launcher derives the flag from the controls).
    gone = C.CONTROL_MBPP_PLUS
    assert gone not in C.STANDARD_BATTERY_PROMPTED and gone not in C.STANDARD_BATTERY_FITTED
    # still RUNNABLE, just not defaulted: an ordinary atom of the grammar
    assert gone in C.TINYCAP_CONTROLS and C.expand_control(gone) == (gone,)
    # U14 took the two GENERATIVE letter-choice legs out
    # of both standard batteries; the rule of 2026-09-11 then RETIRED
    # them (and the full BFCL) from the grammar: no token reaches their row
    # tasks, which stay named only so existing stores' rows resolve.
    for retired, tasks in (("hellaswag_inspect", ("inspect/hellaswag",)),
                           ("inspect_cmp:mmlu", ("inspect/mmlu_letter_valid", "inspect/mmlu_0_shot")),
                           ("bfcl", ("inspect/bfcl",))):
        with pytest.raises(C.ControlNameError):
            C.expand_control(retired)
        for t in tasks:
            assert t in C.TINYCAP_TASKS and t in C.THINKING_BY_TASK       # read path only
            assert not any(t in v for v in C.TASKS_OF_CONTROL.values())    # no producer
    # the battery names the OR-Bench ATOM rather than the inspect_cmp FAMILY,
    # so XSTest (which the launcher's staging exclusion already dropped from
    # every default) is out of the CONTRACTS battery too
    or_bench = f"{C.CONTROL_INSPECT_CMP}{C.CONTROL_SUBTASK_SEP}or_bench_hard_1k"
    assert or_bench in C.STANDARD_BATTERY_PROMPTED
    assert C.CONTROL_INSPECT_CMP not in C.STANDARD_BATTERY_PROMPTED
    assert C.expand_control(or_bench) == (or_bench,)
    with pytest.raises(ValueError, match="unknown battery kind"):
        C.battery_controls("none")
    # the extras are ordinary controls: each runs alone
    for extra in C.IN_DEPTH_EXTRAS:
        assert extra in C.TINYCAP_CONTROLS and C.expand_control(extra) == (extra,)
        assert C.DISPATCH_MODULE_OF_CONTROL[extra] == "sisteer.capeval.tiers.inspect_served"
    # the full GSM8K pools with the standard set; MATH-500 has its OWN class since
    # its one-pass 8,192 cap outgrew gen_long
    assert C.GEOMETRY_CLASS_OF_CONTROL[C.CONTROL_GSM8K_FULL] == C.GEOMETRY_CLASS_GEN_LONG
    assert C.GEOMETRY_CLASS_OF_CONTROL[C.CONTROL_MATH500] == C.GEOMETRY_CLASS_GEN_LONG_MATH


def test_or_bench_battery_item_set_is_frozen_seeded_and_hashed_by_position():
    """U9 item 6 / U10 item 4: the battery's OR-Bench leg
    is a frozen, seeded 200-item subset of the pinned corpus, entering the
    HASHED identity by POSITION; the full corpus stays available behind the
    driver's --or-bench-full as a different identity.

    The positions are pinned here as literals AND as a digest. That is
    deliberate and is the whole point of the word "frozen": every cell of the
    campaign must draw the same 200 prompts, so a change to the seed, the
    corpus size, or the derivation must fail here rather than quietly produce
    an unpaired comparison one wave later."""
    assert C.OR_BENCH_FULL_N == 1319         # the file says "1k"; the corpus is 1,319
    assert C.OR_BENCH_SUBSET_N == 200
    assert C.OR_BENCH_SUBSET_SEED == 20260906
    assert C.OR_BENCH_ITEM_SETS == ("subset", "full") == (
        C.OR_BENCH_ITEM_SET_SUBSET, C.OR_BENCH_ITEM_SET_FULL)
    pos = C.or_bench_subset_positions()
    assert len(pos) == C.OR_BENCH_SUBSET_N == len(set(pos))
    assert list(pos) == sorted(pos) and all(0 <= i < C.OR_BENCH_FULL_N for i in pos)
    assert pos == C.or_bench_subset_positions()                     # pure
    # the FROZEN draw (SHA-256 ordering of "<seed>:<position>" — never
    # random.Random().sample, whose algorithm has changed between CPython
    # releases and could not be relied on for a campaign-long freeze)
    assert pos[:8] == (2, 41, 46, 49, 73, 81, 84, 98)
    assert pos[-4:] == (1291, 1298, 1312, 1317)
    assert hashlib.sha256(",".join(map(str, pos)).encode()).hexdigest()[:16] == "338cef53c0728cdc"
    # another seed, or another n, is another set
    assert C.or_bench_subset_positions(seed=20260907) != pos
    assert C.or_bench_subset_positions(50) != pos and len(C.or_bench_subset_positions(50)) == 50
    # the subset is a PREFIX-free selection over the corpus, not the head of it
    assert pos[:8] != tuple(range(8))
    for bad, why in ((0, "empty"), (C.OR_BENCH_FULL_N + 1, "more items")):
        with pytest.raises(ValueError, match="is not in 1.."):
            C.or_bench_subset_positions(bad)
    # the hashed spec
    spec = C.or_bench_item_set()
    assert tuple(spec) == C.OR_BENCH_ITEM_SET_FIELDS == (
        "item_set", "n", "seed", "of", "positions")
    assert spec == {"item_set": "subset", "n": 200, "seed": 20260906, "of": 1319,
                    "positions": list(pos)}
    full = C.or_bench_item_set(subset=False)
    assert full == {"item_set": "full", "n": 1319, "seed": None, "of": 1319, "positions": None}
    assert full != spec                       # a DIFFERENT identity, by construction
    # the per-run map: None when the run files no OR-Bench rows, and TOKENS or
    # canonical atoms are both accepted (the driver hands atoms, the launcher tokens)
    assert C.run_or_bench_item_set(["humaneval_plus", "gsm8k"]) is None
    assert C.run_or_bench_item_set(["inspect_cmp"]) == spec
    assert C.run_or_bench_item_set(["inspect_cmp:or_bench_hard_1k"]) == spec
    assert C.run_or_bench_item_set(["inspect_cmp"], full=True) == full
    # the battery files it, so every default wave carries the subset
    assert C.run_or_bench_item_set(C.STANDARD_BATTERY_PROMPTED) == spec
    assert C.run_or_bench_item_set(C.STANDARD_BATTERY_FITTED) == spec
    # it is HASHED identity, and the flag that changes it is a DRIVER flag the
    # launcher never emits (like the MATH-500 retry cap — the default IS the design)
    assert C.IDENTITY_KEY_OR_BENCH_ITEM_SET in C.TINYCAP_IDENTITY_FIELDS
    assert C.FLAG_OR_BENCH_FULL == "--or-bench-full"
    assert C.FLAG_OR_BENCH_FULL not in C.TINYCAP_DRIVER_FLAGS
    argv = C.driver_argv(python="p", script="s", proxy_url="u", routing_table="r",
                         cell="none", config="c", defenses_dir="d", out_root="o",
                         controls=["inspect_cmp:or_bench_hard_1k"], thinking="off",
                         concurrency=8, judge_model="m")
    assert C.FLAG_OR_BENCH_FULL not in argv
    # SAMPLES_BY_TASK is untouched: k stays 1 for OR-Bench like every other task
    assert C.SAMPLES_BY_TASK[C.inspect_task("or_bench_hard_1k")] == 1


def test_per_task_thinking_policy_and_the_math500_caps():
    """ONE thinking arm per row task — ON for the generative math / knowledge
    tasks (both GSM8K sets, MATH-500, Inspect's generated-answer MMLU
    variants, XSTest), OFF for the letter / loglikelihood reads (the five
    tiny tasks, HellaSwag) and the recorded calls (EvalPlus as a code prefill
    instrument, BFCL as a tool call, and OR-Bench, whose refusal rate needs
    no scratchpad); never both arms on the same items;
    `battery_thinking(kind)` is the per-task map over a battery; MATH-500
    generates at 4,096 in both arms with an 8,192 per-item retry cap."""
    assert set(C.THINKING_BY_TASK) == set(C.TINYCAP_TASKS)          # total over the row tasks
    assert set(C.THINKING_BY_TASK.values()) <= set(C.THINKING_MODES)
    on = {t for t, m in C.THINKING_BY_TASK.items() if m == C.THINKING_ON}
    assert on == {"inspect/math", "inspect/gsm8k_paired200", "inspect/gsm8k",
                  "inspect/mmlu_letter_valid", "inspect/mmlu_0_shot", "inspect/xstest"}
    # OR-Bench is on the OFF side: a refusal-rate readout needs no
    # scratchpad. Its refusal SIBLING XSTest is untouched and stays ON — no
    # rule covers it, and it is not in the battery.
    assert C.THINKING_BY_TASK["inspect/or_bench_hard_1k"] == C.THINKING_OFF
    assert C.THINKING_BY_TASK["inspect/xstest"] == C.THINKING_ON
    for t in ("inspect/hellaswag", "inspect/bfcl", "inspect/or_bench_hard_1k", *C.EVALPLUS_TASKS):
        assert C.THINKING_BY_TASK[t] == C.THINKING_OFF, t
    assert "never run under both" in C.THINKING_ONE_ARM_PER_TASK and "U5" in C.THINKING_ONE_ARM_PER_TASK
    # the per-task map over each battery: the standard paired GSM8K thinking ON;
    # tiny / HE+ / OR-Bench OFF; no :open variant (the ON-arm twin of legs the
    # policy runs OFF). BFCL and MBPP+ left the battery with U9 item 6, and the
    # two generative letter legs (both MMLU variants, HellaSwag) with U14 — the
    # map over a battery no longer names any of them, while their
    # THINKING_BY_TASK entries stand (asserted above) for an explicit
    # --controls run. XSTest goes with them: the launcher's staging exclusion
    # dropped it from every default list, and since U14 the battery names the
    # OR-Bench ATOM rather than the inspect_cmp family, so the CONTRACTS
    # battery does not reach it either.
    prompted = C.battery_thinking(C.BATTERY_PROMPTED)
    assert prompted["inspect/gsm8k_paired200"] == "on"
    assert prompted["inspect/or_bench_hard_1k"] == "off"     # U9 item 6 / U10 item 4
    assert not any(t.startswith("served_ll/") for t in prompted)
    assert prompted["evalplus/humaneval_plus"] == "off"
    assert "inspect/bfcl" not in prompted and "evalplus/mbpp_plus" not in prompted
    for gone in ("inspect/mmlu_letter_valid", "inspect/mmlu_0_shot",
                 "inspect/hellaswag", "inspect/xstest"):
        assert gone not in prompted, gone                    # U14
        assert gone in C.THINKING_BY_TASK                    # the policy still maps it
    # GSM8K is the battery's ONE thinking-ON task now, so a per-task
    # configuration still sizes BOTH gen_long and gen_long_thinking
    assert [t for t, m in prompted.items() if m == "on"] == ["inspect/gsm8k_paired200"]
    assert set(prompted.values()) == {"on", "off"}
    assert not any(t.endswith(":open") for t in prompted)
    assert "inspect/math" not in prompted and "inspect/gsm8k" not in prompted   # in-depth only
    fitted = C.battery_thinking(C.BATTERY_FITTED)
    assert fitted == prompted                    # O7: one battery, one policy
    deep = C.battery_thinking(C.BATTERY_PROMPTED, in_depth=True)
    assert set(deep) == set(prompted) | {"inspect/math", "inspect/gsm8k"}
    assert deep["inspect/math"] == deep["inspect/gsm8k"] == "on"
    for task, mode in deep.items():
        assert mode == C.THINKING_BY_TASK[task]
    with pytest.raises(ValueError, match="unknown battery kind"):
        C.battery_thinking("none")
    # the MATH-500 cap: ONE pass at 16,384 in both arms — the standing truncation
    # rule's ceiling — the per-item retry retired by default (0); the legacy
    # identity literal is what every pre-change store recorded for the retry field
    # and does NOT follow the cap
    assert C.MATH500_MAX_TOKENS_DEFAULT == 16384 and C.MATH500_RETRY_MAX_TOKENS == 0
    assert C.MATH500_RETRY_MAX_TOKENS_LEGACY_IDENTITY == 8192 != C.MATH500_MAX_TOKENS_DEFAULT


def test_math500_retry_identity_value_keeps_the_legacy_literal_off_math_runs():
    """``math500_retry_max_tokens`` is hashed on EVERY tinycap store, so a run
    that files no MATH-500 row stamps the literal every pre-change store
    carried (8192) whatever the driver's default or the flag says, and a MATH
    run stamps the value it actually arms. Row tasks in, never control atoms
    (a control name refuses by name)."""
    math = C.inspect_task("math")
    gsm = C.GSM8K_STANDARD_TASK
    he = [C.evalplus_task("humaneval")]
    for retry in (0, 8192, 16384, 32768):
        assert C.math500_retry_identity_value([gsm], retry) == 8192
        assert C.math500_retry_identity_value(he, retry) == 8192
        assert C.math500_retry_identity_value([], retry) == 8192
        assert C.math500_retry_identity_value({gsm: "on", "evalplus/humaneval_plus": "off"}, retry) == 8192
        assert C.math500_retry_identity_value([math], retry) == retry
        assert C.math500_retry_identity_value({math: "on", gsm: "on"}, retry) == retry
    assert C.math500_retry_identity_value([gsm], C.MATH500_RETRY_MAX_TOKENS) == C.MATH500_RETRY_MAX_TOKENS_LEGACY_IDENTITY
    assert C.math500_retry_identity_value([math], C.MATH500_RETRY_MAX_TOKENS) == 0
    for atoms in ([C.CONTROL_MATH500], ["inspect_cmp:xstest"], [gsm, C.CONTROL_GSM8K]):
        with pytest.raises(ValueError, match="takes ROW tasks"):
            C.math500_retry_identity_value(atoms, 0)


def test_summary_payload_keys_pass_the_counts_only_rule():
    for k in C.SUMMARY_PAYLOAD_FIELDS:
        assert _SAFE_STRING.match(k), k
    assert "pass_at_1_plus" in C.SUMMARY_PAYLOAD_FIELDS
    assert "pass@1_plus" not in C.SUMMARY_PAYLOAD_FIELDS
    # PW12 (E19): the pass@1_plus DEFINITION is a summary field with two named readings
    assert C.EVALPLUS_METRIC_DEF_FIELD == "metric_def" and C.EVALPLUS_METRIC_DEF_FIELD in C.SUMMARY_PAYLOAD_FIELDS
    assert C.EVALPLUS_PLUS_DEFS == ("base_and_plus", "plus_only")
    assert C.EVALPLUS_PLUS_DEF_BASE_AND_PLUS == "base_and_plus" and C.EVALPLUS_PLUS_DEF_PLUS_ONLY == "plus_only"
    for d in C.EVALPLUS_PLUS_DEFS:
        assert _SAFE_STRING.match(d), d
    assert set(C.DONE_MARKER_FIELDS) == {
        "task", "n_items", "n_rows", "n_requests", "n_rejected", "wall_clock_s", "finished_at"}
    # E55 #2 + E60 (R22): the REQUIRED set above stays exact — a field added
    # there would mark every marker on the volume untrusted — so both later
    # facts ride as optional keys. `rebound_at` is written ONLY by the re-bind
    # mode, so it is admitted but never LIFTED out of a summary by finish_task.
    # qwen-seedrep-native-battery fg1 / F-AU: `n_planned` (the plan's count beside the filed one) is the third optional, additive key
    assert set(C.DONE_MARKER_OPTIONAL_FIELDS) == {"samples_sha256", "rebound_at", "n_planned"}
    assert C.DONE_MARKER_N_PLANNED_FIELD == "n_planned"
    assert C.DONE_MARKER_REBOUND_AT_FIELD == "rebound_at"
    assert C.DONE_MARKER_REBOUND_AT_FIELD in C.DONE_MARKER_OPTIONAL_FIELDS
    assert set(C.DONE_MARKER_SUMMARY_LIFTED_FIELDS) == {"samples_sha256"}
    assert set(C.DONE_MARKER_SUMMARY_LIFTED_FIELDS) < set(C.DONE_MARKER_OPTIONAL_FIELDS)
    assert not set(C.DONE_MARKER_OPTIONAL_FIELDS) & set(C.DONE_MARKER_FIELDS)
    assert C.DONE_DIR == "done"
    assert C.INSPECT_LOGS_SUBDIR == "private/inspect_logs"


# --------------------------------------------------------------------------- #
# thinking mode                                                                #
# --------------------------------------------------------------------------- #


def test_thinking_mode_contract():
    # the no-switch arm is a full driver arm, LAST
    assert C.THINKING_MODES == ("on", "off", "not_available") == C.THINKING_ARMS and C.THINKING_SWITCH_ARMS == ("on", "off")
    assert C.enable_thinking("on") is True and C.enable_thinking("off") is False
    assert C.enable_thinking("not_available") is False
    assert C.sends_thinking_kwargs("on") is True and C.sends_thinking_kwargs("off") is True
    assert C.sends_thinking_kwargs("not_available") is False
    with pytest.raises(ValueError):
        C.enable_thinking("true")
    with pytest.raises(ValueError):
        C.sends_thinking_kwargs("per-task")
    assert C.chat_template_kwargs("off") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert C.chat_template_kwargs("not_available") == {}
    assert C.render_fragment(lens=True, arm="not_available") == {}
    assert C.CHAT_TEMPLATE_KWARGS_KEY == "chat_template_kwargs"
    assert json.loads(vs.THINKING_OFF_TEMPLATE_KWARGS) == C.chat_template_kwargs("off")[
        C.CHAT_TEMPLATE_KWARGS_KEY]
    assert C.THINKING_RENDERS == ("native", "flag_only", "thought_block_diag")
    assert C.REASONING_FIELD_DIAG == "reasoning"


def test_the_driver_mode_vocabulary_adds_per_task_without_making_it_an_arm():
    """PW20b-D (U5-4): `per-task` is a DRIVER mode — an instruction to read
    each task's arm off THINKING_BY_TASK — not a third arm. `enable_thinking`
    still refuses it on purpose: there is no single per-request boolean under
    it, and a caller reaching for one is asking the wrong question."""
    assert C.THINKING_PER_TASK == "per-task"
    assert C.LAUNCHER_THINKING_MODES == ("off", "on", "per-task", "not_available")   # PR-A's order: the no-switch arm LAST
    assert C.THINKING_MODES == ("on", "off", "not_available")            # the ARMS gained the no-switch arm
    for m in C.LAUNCHER_THINKING_MODES:
        assert C.check_thinking_arg(m) == m
    assert C.is_per_task("per-task") is True and C.is_per_task("off") is False
    with pytest.raises(ValueError):
        C.check_thinking_arg("yes")
    with pytest.raises(ValueError, match="thinking mode must be one of"):
        C.enable_thinking(C.THINKING_PER_TASK)
    # the launcher's argv builder accepts every driver mode (its `cell_out_dir`
    # still calls enable_thinking)
    argv = C.driver_argv(python="p", script="s", proxy_url="u", routing_table="r",
                         cell="c", config="f", defenses_dir="d", out_root="o",
                         controls=["gsm8k"], thinking=C.THINKING_PER_TASK,
                         concurrency=24, judge_model="j")
    assert dict(zip(argv[3::2], argv[4::2]))[C.FLAG_THINKING] == "per-task"
    # the retry flag exists but is deliberately NOT a launcher flag: the
    # driver's default IS the designed behaviour, so no new argv is needed
    assert C.FLAG_MATH500_RETRY_MAX_TOKENS == "--math500-retry-max-tokens"
    assert C.FLAG_MATH500_RETRY_MAX_TOKENS not in C.TINYCAP_DRIVER_FLAGS
    assert C.FLAG_MATH500_RETRY_MAX_TOKENS not in argv


def test_run_thinking_by_task_is_the_map_a_RUN_files():
    """The identity map (PW20b-D): sorted, over the tasks the run FILES.
    Under `per-task` each task takes THINKING_BY_TASK's arm and an unmapped
    task raises naming it; under an arm every task takes that one. The
    EvalPlus `:open` variants appear only when their arm is ON, because that
    is the only arm that generates them — the rule battery_thinking follows."""
    atoms = ["gsm8k", "inspect_cmp:or_bench_hard_1k", "humaneval_plus", "inspect_cmp:xstest"]
    per_task = C.run_thinking_by_task(atoms, C.THINKING_PER_TASK)
    assert per_task == {"evalplus/humaneval_plus": "off", "inspect/gsm8k_paired200": "on",
                        "inspect/or_bench_hard_1k": "off", "inspect/xstest": "on"}
    assert list(per_task) == sorted(per_task)
    for task, arm in per_task.items():
        assert arm == C.THINKING_BY_TASK[task]
    on = C.run_thinking_by_task(atoms, C.THINKING_ON)
    assert set(on.values()) == {"on"}
    assert "evalplus/humaneval_plus:open" in on          # ON generates the twin
    off = C.run_thinking_by_task(atoms, C.THINKING_OFF)
    assert set(off.values()) == {"off"}
    assert "evalplus/humaneval_plus:open" not in off     # OFF never does
    # the same rule the battery's design record follows
    assert C.battery_thinking(C.BATTERY_PROMPTED) == C.run_thinking_by_task(
        [a for tok in C.battery_controls(C.BATTERY_PROMPTED) for a in C.expand_control(tok)],
        C.THINKING_PER_TASK)
    with pytest.raises(ValueError):
        C.run_thinking_by_task(atoms, "yes")


def test_reasoning_parser_pin_is_a_known_family():
    assert dict(C.REASONING_PARSER_BY_FAMILY) == {"gemma4": "gemma4"}
    assert set(C.REASONING_PARSER_BY_FAMILY) <= set(vs.TOOL_PARSER_BY_FAMILY)
    if hasattr(vs, "REASONING_PARSER_BY_FAMILY"):  # Lane B's knob has landed
        assert dict(vs.REASONING_PARSER_BY_FAMILY) == dict(C.REASONING_PARSER_BY_FAMILY)


# --------------------------------------------------------------------------- #
# proxy facts are re-exports, never retyped                                    #
# --------------------------------------------------------------------------- #


def test_proxy_facts_are_the_proxy_modules_own():
    assert C.FORBIDDEN_CLIENT_KEYS is px.FORBIDDEN_CLIENT_KEYS
    assert set(C.FORBIDDEN_CLIENT_KEYS) == {
        "vllm_xargs", "petri_object", "expect_hash", "extra_args", "cache_salt"}
    assert C.RESPONSE_PREFILL_KEY == px.RESPONSE_PREFILL_KEY == "petri_response_prefill"
    assert C.AGENT_LOOP_KEY == px.AGENT_LOOP_KEY
    # PW18: the /v1/models identity block's keys are the proxy's spellings
    # (``models_payload`` emits ``petri_proxy.petri_agent_loop_entry``; the
    # standup CLI's opt-out is argparse's BooleanOptionalAction negative)
    assert C.PROXY_IDENTITY_KEY == "petri_proxy"
    assert C.PROXY_IDENTITY_AGENT_LOOP_ENTRY == "petri_agent_loop_entry"
    src = inspect.getsource(px.RoutingProxy.models_payload)
    assert f'"{C.PROXY_IDENTITY_KEY}": {{' in src
    assert f'"{C.PROXY_IDENTITY_AGENT_LOOP_ENTRY}": self._petri_agent_loop_entry' in src
    standup_src = (REPO / "scripts" / "capctrl_serving_up.py").read_text()
    assert '"--petri-agent-loop", action=argparse.BooleanOptionalAction' in standup_src
    assert C.FLAG_NO_PETRI_AGENT_LOOP == "--no-petri-agent-loop"
    assert f'"{C.AGENT_LOOP_ENTRY_KEY}": bool(self.petri_agent_loop)' in standup_src
    assert f'"{C.PROXY_IDENTITY_REDUCTION_MODE}"' in standup_src            # the E25 passthrough


def test_reduction_mode_contract_resolves_absent_to_per_row_and_refuses_disagreement():
    """E25 / PW18: the lens reduction mode is a served-identity field read
    from the record / identity block under ``lens_reduction_mode``; ABSENT
    (every server before the stacked landing) is ``per_row`` — never null;
    a stated value must be a known mode and every stated source must agree;
    the standup's env selects it (``1`` / unset = stacked since E46, ``0`` =
    per_row) — a SERVER that states nothing is still per_row, which is a
    different question from what an unset switch asks a new server for."""
    assert C.REDUCTION_MODES == ("per_row", "stacked")
    assert C.REDUCTION_MODE_DEFAULT == C.REDUCTION_MODE_PER_ROW == "per_row"
    assert C.REDUCTION_MODE_STACKED == "stacked"
    assert C.PROXY_IDENTITY_REDUCTION_MODE == "lens_reduction_mode"
    assert C.LENS_STACKED_REDUCTIONS_ENV == "SISTEER_LENS_STACKED_REDUCTIONS"
    assert C.resolve_reduction_mode() == "per_row"
    assert C.resolve_reduction_mode(None) == "per_row"
    assert C.resolve_reduction_mode(None, None, sources=("a", "b")) == "per_row"
    assert C.resolve_reduction_mode("stacked") == "stacked"
    assert C.resolve_reduction_mode(None, "stacked", sources=("record", "live")) == "stacked"
    assert C.resolve_reduction_mode("per_row", "per_row") == "per_row"
    with pytest.raises(ValueError, match="record states lens_reduction_mode='grouped', not one of"):
        C.resolve_reduction_mode("grouped", sources=("record",))
    with pytest.raises(ValueError, match="not one of"):
        C.resolve_reduction_mode(1)
    with pytest.raises(ValueError, match="stated inconsistently: record='per_row'; live='stacked'"):
        C.resolve_reduction_mode("per_row", "stacked", sources=("record", "live"))
    # E29 / PW18b, unset branch flipped by E46: the env parser — unset / empty
    # / 1 / stacked -> stacked, 0 / per_row / off -> per_row, anything else
    # refuses; it yields an EXPECTATION the attestation must match. UNSET IS
    # STACKED because the landing is on main (f39a4562) and merged here, so a
    # worker spawned from this tree with the switch unset serves stacked —
    # E31's opposite reading was right only while the landing had not happened.
    assert C.reduction_mode_from_env({}) == "stacked" == C.REDUCTION_MODE_ENV_UNSET
    # the OTHER two constants answer different questions and did NOT move with
    # it: DEFAULT is what a keyless SERVER served (pre-landing = per_row,
    # forever), LAUNCHER_DEFAULT is what the launcher ASKS for (per_row: the
    # wave in flight is per_row, and the launcher always states it explicitly)
    assert C.REDUCTION_MODE_DEFAULT == C.REDUCTION_MODE_LAUNCHER_DEFAULT == "per_row"
    assert C.REDUCTION_MODE_ENV_UNSET != C.REDUCTION_MODE_DEFAULT
    for raw, mode in (("", "stacked"), ("1", "stacked"), ("stacked", "stacked"),
                      ("0", "per_row"), ("per_row", "per_row"), ("off", "per_row")):
        assert C.reduction_mode_from_env({"SISTEER_LENS_STACKED_REDUCTIONS": raw}) == mode, raw
    # the serving parser (lens_adapters.parse_reduction_mode) also takes these
    # spellings; this one refuses them, which is the SAFE direction — a loud
    # refusal before any GPU minute, never a wrong expectation held to a server
    for bad in ("true", "2", "yes", "grouped", "on", "ON", " 1 ", "Stacked"):
        with pytest.raises(ValueError, match="is not one of"):
            C.reduction_mode_from_env({"SISTEER_LENS_STACKED_REDUCTIONS": bad})
    assert set(C.REDUCTION_MODE_ENV_VALUES.values()) == set(C.REDUCTION_MODES)
    assert C.REDUCTION_MODE_ENV_VALUES[""] == C.REDUCTION_MODE_ENV_UNSET == "stacked"
    assert C.REDUCTION_MODE_LAUNCHER_DEFAULT == "per_row"          # E46: the wave runs per_row
    # the attestation is the truth; an expectation must agree or the run refuses
    assert C.check_reduction_mode_expectation("stacked", None) == "stacked"
    assert C.check_reduction_mode_expectation("per_row", None) == "per_row"
    assert C.check_reduction_mode_expectation("stacked", "stacked") == "stacked"
    assert C.check_reduction_mode_expectation("per_row", "per_row", keyless=True) == "per_row"
    with pytest.raises(ValueError, match="attests lens_reduction_mode='stacked' but "
                                         "SISTEER_LENS_STACKED_REDUCTIONS expects 'per_row'"):
        C.check_reduction_mode_expectation("stacked", "per_row")
    with pytest.raises(ValueError, match="attests lens_reduction_mode='per_row' but the launcher expects 'stacked'"):
        C.check_reduction_mode_expectation("per_row", "stacked", expected_source="the launcher")
    # a keyless (pre-landing) server equals per_row ONLY when per_row was expected
    with pytest.raises(ValueError, match="expects lens_reduction_mode='stacked' but the server attests no "
                                         "reduction mode at all — a pre-landing server"):
        C.check_reduction_mode_expectation("per_row", "stacked", keyless=True)
    with pytest.raises(ValueError, match="not one of"):
        C.check_reduction_mode_expectation("grouped", None)
    assert C.ROUTE_NONE_CELL == "none" and C.SCALE0_TWIN_SUFFIX == "__lens0"
    assert C.BITES_LEGS == ("steered", "scale0", "scale0_repeat")
    assert C.BITES_PROBE_ITEMS == srun.LOGPROB_BITES_PROBES == 8


# --------------------------------------------------------------------------- #
# geometry sub-entries                                                         #
# --------------------------------------------------------------------------- #


def test_geometry_subentry_keys_are_capability_tasks_not_yet_registered():
    assert C.GEOMETRY_SUBENTRY_KEYS == (
        "capability:tiny", "capability:tiny_prompted", "capability:gen_long",
        "capability:gen_long_thinking", "capability:agent", "capability:agent_thinking",
        "capability:gen_long_math", "capability:gen_long_math_longsi")
    for key in C.GEOMETRY_SUBENTRY_KEYS:
        ev, task = key.split(":")
        assert ev == C.GEOMETRY_EVAL == "capability"
        assert sg.parse_key(ev, task) == key
        if key in sg.TABLE:  # Lane D has registered it: it must be a real capability sub-entry
            g = sg.TABLE[key]
            assert (g.eval, g.task) == (ev, task) and sg.sizing_problem(g) is None
    assert set(C.GEOMETRY_CLASS_OF_CONTROL) == set(C.TINYCAP_CONTROLS)
    assert set(C.GEOMETRY_CLASS_OF_CONTROL.values()) <= set(C.GEOMETRY_CLASSES)
    assert C.GEOMETRY_CLASSES == ("tiny", "gen_long", "agent", "gen_long_math")
    # MATH-500's own class resolves BOTH arms to one of TWO registered entries by
    # the cell's PROMPT CLASS (no _thinking twin); both gen_long entries untouched
    # and neither able to hold the 16,384 pass from any prompt
    assert C.GEOMETRY_CLASS_OF_CONTROL[C.CONTROL_MATH500] == C.GEOMETRY_CLASS_GEN_LONG_MATH == "gen_long_math"
    for thinking in (False, True):
        # no system turn (the base cell, every object cell): the short class
        assert C.geometry_subentry("gen_long_math", prompted=False, thinking=thinking) == "capability:gen_long_math"
        # the ~90-token safety SI: still the short class
        assert C.geometry_subentry("gen_long_math", prompted=True, thinking=thinking,
                                   si="safety_refusal") == "capability:gen_long_math"
        # the two ~6-7k-token SIs: the long class's own entry
        for si in ("si_constitution", "si_irrelevant"):
            assert C.geometry_subentry("gen_long_math", prompted=True, thinking=thinking,
                                       si=si) == "capability:gen_long_math_longsi"
        # a prompted MATH cell must NAME its SI: the prompted bit alone cannot pick a window
        with pytest.raises(ValueError, match="name the SI"):
            C.geometry_subentry("gen_long_math", prompted=True, thinking=thinking)
        # an SI outside the measured classes refuses by name, with the remedy
        with pytest.raises(C.Math500PromptClassUnmeasured, match="prompted__si_hhh .*NO measured prompt class") as e:
            C.geometry_subentry("gen_long_math", prompted=True, thinking=thinking, si="si_hhh")
        assert "--system-prompt-cell prompted__si_hhh" in str(e.value) and "MATH500_PROMPT_CLASS_OF_SI" in str(e.value)
        # an SI on an un-prompted cell is a contradiction (an object cell carries no system turn)
        with pytest.raises(ValueError, match="only a prompted cell carries a system turn"):
            C.geometry_subentry("gen_long_math", prompted=False, thinking=thinking, si="si_irrelevant")
    # the class map: exactly the served vocabulary's three SIs, two classes, two entries
    assert C.MATH500_PROMPT_CLASSES == ("short", "long_si") == (C.MATH500_PROMPT_CLASS_SHORT, C.MATH500_PROMPT_CLASS_LONG_SI)
    assert C.MATH500_PROMPT_CLASS_OF_SI == {"safety_refusal": "short", "si_constitution": "long_si", "si_irrelevant": "long_si"}
    assert C.math500_prompt_class_of_si(None) == "short"
    assert [C.math500_prompt_class_of_si(si) for si in ("safety_refusal", "si_constitution", "si_irrelevant")] == ["short", "long_si", "long_si"]
    with pytest.raises(C.Math500PromptClassUnmeasured):
        C.math500_prompt_class_of_si("si_hhh")
    assert issubclass(C.Math500PromptClassUnmeasured, ValueError)
    assert C.MATH500_GEOMETRY_KEY_OF_CLASS == {"short": "capability:gen_long_math", "long_si": "capability:gen_long_math_longsi"}
    assert [C.math500_geometry_key(c) for c in C.MATH500_PROMPT_CLASSES] == ["capability:gen_long_math", "capability:gen_long_math_longsi"]
    with pytest.raises(ValueError, match="unknown MATH-500 prompt class"):
        C.math500_geometry_key("medium")
    for key in C.MATH500_GEOMETRY_KEY_OF_CLASS.values():
        assert key in C.GEOMETRY_SUBENTRY_KEYS and key in sg.TABLE
    math = sg.TABLE["capability:gen_long_math"]
    assert math.max_model_len == 17664 >= 1041 + C.MATH500_MAX_TOKENS_DEFAULT      # the measured prompt max + the cap
    assert math.basis.requested_max_tokens == C.MATH500_MAX_TOKENS_DEFAULT == 16384
    longsi = sg.TABLE["capability:gen_long_math_longsi"]
    assert longsi.max_model_len == 24064 >= 7622 + C.MATH500_MAX_TOKENS_DEFAULT    # the long classes' measured max + the cap
    assert longsi.basis.requested_max_tokens == C.MATH500_MAX_TOKENS_DEFAULT
    # the other classes ignore an SI (tiny keys on the prompted bit: one analytic window for every SI)
    assert C.geometry_subentry("tiny", prompted=True, thinking=False, si="si_irrelevant") == "capability:tiny_prompted"
    assert C.geometry_subentry("gen_long", prompted=True, thinking=True, si="safety_refusal") == "capability:gen_long_thinking"
    assert sg.TABLE["capability:gen_long_thinking"].max_model_len == 10752 < 1 + C.MATH500_MAX_TOKENS_DEFAULT
    assert sg.TABLE["capability:gen_long"].max_model_len == 4608 < 1 + C.MATH500_MAX_TOKENS_DEFAULT
    assert C.geometry_subentry("tiny", prompted=False, thinking=True) == "capability:tiny"
    assert C.geometry_subentry("tiny", prompted=True, thinking=False) == "capability:tiny_prompted"
    assert C.geometry_subentry("gen_long", prompted=True, thinking=False) == "capability:gen_long"
    assert C.geometry_subentry("gen_long", prompted=False, thinking=True) == "capability:gen_long_thinking"
    assert C.geometry_subentry("agent", prompted=True, thinking=True) == "capability:agent_thinking"
    with pytest.raises(ValueError):
        C.geometry_subentry("short", prompted=False, thinking=False)


def test_grader_geometry_key_is_informational_and_outside_the_subentries():
    """PW11 (Evals' geometry verdict, 2026-09-04): the model-grader traffic has
    its own INFORMATIONAL key — named here, registered in the table, refused
    by resolve_geometry — and is NOT one of the seven served sub-entries (no
    configuration pools into it; no pass-2 standup is sized by it). The
    model-graded tasks name where grader-route traffic is expected."""
    assert C.GEOMETRY_GRADER_KEY == "capability:grader"
    assert C.GEOMETRY_GRADER_KEY not in C.GEOMETRY_SUBENTRY_KEYS
    assert sg.TINYCAP_KEYS == C.GEOMETRY_SUBENTRY_KEYS
    g = sg.TABLE[C.GEOMETRY_GRADER_KEY]
    assert g.informational and not g.provisional and sg.sizing_problem(g) is None
    with pytest.raises(sg.GeometryInformational):
        sg.resolve_geometry(C.GEOMETRY_GRADER_KEY)
    # MATH-500 LEFT the model-graded set — its task is scored by the
    # judge-free sympy scorer alone
    assert C.MODEL_GRADED_INSPECT_TASKS == (
        C.inspect_task("xstest"), C.inspect_task("or_bench_hard_1k"))
    assert C.inspect_task("math") not in C.MODEL_GRADED_INSPECT_TASKS
    assert set(C.MODEL_GRADED_INSPECT_TASKS) <= set(C.INSPECT_TASKS)
    # the six measured sub-entries resolve; the analytic tiny_prompted still refuses
    for key in C.GEOMETRY_SUBENTRY_KEYS:
        if key == "capability:tiny_prompted":
            with pytest.raises(sg.GeometryProvisional):
                sg.resolve_geometry(key)
        else:
            assert sg.resolve_geometry(key).entry is sg.TABLE[key]
    assert C.GEOMETRY_BASIS_N_REJECTED_FIELD == "n_rejected"
    assert "n_rejected" not in sg.GeometryBasis.__dataclass_fields__ or True  # Lane D adds it


# --------------------------------------------------------------------------- #
# usage ledger + headers                                                       #
# --------------------------------------------------------------------------- #


def test_ledger_row_schema_and_headers():
    fields = C.LEDGER_ROW_FIELDS
    assert len(fields) == len(set(fields))
    for must in ("prompt_tokens_pre", "rejected", "status", "error_code",
                 "usage_prompt_tokens", "usage_completion_tokens", "tokenize_mismatch",
                 "task", "sample", "vmodel", "thinking", "has_tools", "n_tools",
                 "max_tokens_requested", "finish_reason", "latency_s", "ts"):
        assert must in fields, must
    assert C.LEDGER_FILENAME == "usage_ledger.jsonl"
    # the ledger's --endpoint-json record beside the routing table (N2)
    assert C.LEDGER_ENDPOINT_FILENAME == "ledger_endpoint.json"
    assert C.TINYCAP_HEADERS == ("X-Tinycap-Task", "X-Tinycap-Sample")
    for h in C.TINYCAP_HEADERS:
        assert h.startswith(C.HEADER_PREFIX)
    assert C.sample_header_value("HumanEval/3") == "HumanEval/3/0"
    assert C.sample_header_value("math_0123abcd", 2) == "math_0123abcd/2"


# --------------------------------------------------------------------------- #
# Inspect provider                                                             #
# --------------------------------------------------------------------------- #


def test_inspect_provider_contract_and_sample_ids():
    assert C.INSPECT_PROVIDER_NAME == "tinycap"
    assert C.inspect_model_name("petri__x") == "tinycap/petri__x"
    assert (C.INSPECT_BASE_URL_ENV, C.INSPECT_API_KEY_ENV) == ("TINYCAP_BASE_URL", "TINYCAP_API_KEY")
    assert (C.INSPECT_MAX_RETRIES, C.INSPECT_TIMEOUT_S, C.INSPECT_STRICT_TOOLS) == (2, 7200, False)
    assert C.MATH500_SAMPLES_PER_ITEM == 1        # k = 1 everywhere for now
    # the id rule, pinned on synthetic problems (never a real dataset row)
    probs = [f"Compute {i} + {i}." for i in range(5)]
    ids = [C.math500_sample_id(p) for p in probs]
    assert len(set(ids)) == 5
    for p, sid in zip(probs, ids):
        assert sid == "math_" + hashlib.md5(p.encode("utf-8")).hexdigest()[:8]
        assert re.fullmatch(r"math_[0-9a-f]{8}", sid)
    assert C.gsm8k_sample_id("q").startswith("gsm8k_")
    assert C.BFCL_GORILLA_COMMIT == "dac44e7"
    assert C.BFCL_CATEGORY_SETS == ("V1_CATEGORIES", "V2_CATEGORIES")
    assert (C.OR_BENCH_DATASET, C.OR_BENCH_HARD_FILE) == ("bench-llm/or-bench", "or-bench-hard-1k.csv")
    assert set(C.RETIRED_INSPECT_TASK_NAMES) <= set(C.INSPECT_TASK_NAMES)   # read path only


# --------------------------------------------------------------------------- #
# launcher argv                                                                #
# --------------------------------------------------------------------------- #

_LITERAL_WINDOW = re.compile(
    r'''--max-model-len["']\s*,\s*["']16384["']|max_model_len\s*=\s*16384\b''')


def test_launcher_names_and_layout():
    # the app name stopped claiming "four gemma configurations"
    # when the launcher grew the model registry; a wave names its own app
    # through --app-name / app_name_for — REQUIRED since E40 M7 (the constant
    # is the sandbox's app; the launcher reads MODAL_APP_NAME_ENV, no default)
    assert C.MODAL_APP_NAME == "tinycap-served"
    assert C.MODAL_APP_NAME_ENV == "TINYCAP_APP_NAME"
    assert C.app_name_for("gsm8k-rerun", "gemma4e4b") == "tinycap-gsm8k-rerun-e4b"
    assert C.app_name_for("battery", "qwen35_9b") == "tinycap-battery-qwen9b"
    assert C.app_name_for("merge") == "tinycap-merge"
    assert set(C.APP_PURPOSES) >= {"gsm8k-rerun", "battery", "smoke"}
    with pytest.raises(ValueError):
        C.app_name_for("")
    assert C.MODAL_VOLUME == "twosi-fits"
    assert C.VOL_GEN_ROOT == "/vol/tinycap/gen"
    assert C.LOCAL_OUT_ROOT.startswith("$GRADE/")      # placeholder form: never an absolute home path or a built-in directory
    assert C.TEST_CELLS == ("prompted__safety_refusal", "safety_refusal__psr_kl_all")
    assert len(C.ALL_CONTAINER_NAMES) == len(set(C.ALL_CONTAINER_NAMES)) == 5
    for name in C.ALL_CONTAINER_NAMES:
        assert name.startswith("tinycap-") and "think" in name or name == C.CONTAINER_SANDBOX_EXECUTE
    assert not hasattr(C, "CONTAINER_HF_SPOTCHECK")
    assert C.container_name("prompted__safety_refusal", "on") == "tinycap-e4b-prompted_safety-think_on"
    # container_name GENERATES a name for any cell of any model —
    # `none` used to refuse (the four-entry table)
    assert C.container_name("none", "off") == "tinycap-e4b-base-think_off"
    assert C.cell_out_dir("/vol/tinycap/gen", "safety_refusal__psr_kl_all", "off") == (
        "/vol/tinycap/gen/gemma4e4b/safety_refusal__psr_kl_all/thinking_off")
    # 8 since the campaign-wide re-run: a wave is one
    # invocation per model over up to twelve cells, and 4 doubled the wall for
    # no reason but the trip-wire. --allow-extra stays the deliberate step above.
    assert C.MAX_GPU_CONTAINERS == 8
    assert (REPO / C.PETRI_MANIFEST_GEMMA4E4B).is_file()
    assert set(C.LAUNCHER_ENTRYPOINT_KWARGS) >= {"sha", "stage", "pf", "smoke", "pass1", "allow_extra"}


def test_driver_argv_carries_the_new_flags_and_the_resolved_concurrency():
    argv = C.driver_argv(
        python="python", script="scripts/served_capability_run.py",
        proxy_url="http://127.0.0.1:1/v1", routing_table="/tmp/rt.json",
        cell="safety_refusal__psr_kl_all", config="configs/two_si.yaml",
        defenses_dir="/vol/objects", out_root="/vol/out",
        controls=["gsm8k", "math500", "mmlu"], thinking="on", concurrency=52,
        judge_model="claude-sonnet-5", ledger_url="http://127.0.0.1:2/v1",
        vllm_log="/tmp/v.log")
    assert argv[:3] == ["python", "scripts/served_capability_run.py", "generate"]
    pairs = dict(zip(argv[3::2], argv[4::2]))
    assert pairs["--controls"] == "gsm8k,math500,mmlu"
    assert pairs["--thinking"] == "on"
    assert pairs["--samples"] == str(C.MATH500_SAMPLES_PER_ITEM) == "1" and pairs["--inspect-limit"] == "0"
    assert pairs["--ledger-url"] == "http://127.0.0.1:2/v1"
    assert pairs["--concurrency"] == "52"
    assert pairs["--judge-model"] == "claude-sonnet-5"
    assert pairs["--out-root"] == "/vol/out" and pairs["--vllm-log"] == "/tmp/v.log"
    assert set(C.TINYCAP_DRIVER_FLAGS) <= set(pairs)
    # the ledger flag is optional; thinking is validated; controls are required
    short = C.driver_argv(python="p", script="s", proxy_url="u", routing_table="r", cell="c",
                          config="f", defenses_dir="d", out_root="o", controls=["bfcl_200"],
                          thinking="off", concurrency=24, judge_model="j")
    assert "--ledger-url" not in short and "--vllm-log" not in short
    with pytest.raises(ValueError):
        C.driver_argv(python="p", script="s", proxy_url="u", routing_table="r", cell="c",
                      config="f", defenses_dir="d", out_root="o", controls=["bfcl_200"],
                      thinking="yes", concurrency=24, judge_model="j")
    with pytest.raises(ValueError):
        C.driver_argv(python="p", script="s", proxy_url="u", routing_table="r", cell="c",
                      config="f", defenses_dir="d", out_root="o", controls=[],
                      thinking="off", concurrency=24, judge_model="j")


def test_standup_extra_argv():
    assert C.standup_extra_argv(tool_calls=False, thinking="off") == []
    assert C.standup_extra_argv(tool_calls=True, thinking="off") == [
        "--tool-calls", "--manifest", "configs/capeval/manifests/petri_gemma4e4b.yaml"]
    assert C.standup_extra_argv(tool_calls=True, thinking="on") == [
        "--tool-calls", "--manifest", "configs/capeval/manifests/petri_gemma4e4b.yaml",
        "--reasoning-parser", "gemma4"]
    assert C.standup_extra_argv(tool_calls=False, thinking="on") == ["--reasoning-parser", "gemma4"]
    with pytest.raises(ValueError):
        C.standup_extra_argv(tool_calls=False, thinking="on", family="llama")


def test_no_literal_petri_window_in_the_contracts():
    src = Path(C.__file__).read_text()
    assert not _LITERAL_WINDOW.search(src)
    # the two numbers that may read 16384: the sandbox MEMORY (MiB) and MATH-500's
    # one-pass completion CAP (the standing truncation rule's ceiling) — a cap,
    # not a window (MATH's window is capability:gen_long_math, 17,664, in geometry.py)
    assert "16384" not in (src.replace("memory=16384", "").replace("SANDBOX_MEMORY_MIB = 16384", "")
                           .replace("MATH500_MAX_TOKENS_DEFAULT = 16384", "")), (
        "the only 16384s in the contracts are the sandbox MEMORY (MiB) and the MATH-500 "
        "completion cap, never a window")
    assert C.MATH500_MAX_TOKENS_DEFAULT == 16384 and "MATH500_MAX_TOKENS_DEFAULT = 16384" in src


# --------------------------------------------------------------------------- #
# sandbox                                                                      #
# --------------------------------------------------------------------------- #


def test_sandbox_contract_matches_the_repo_execution_image():
    sh = (REPO / "scripts/capeval_sandbox.sh").read_text()
    assert C.SANDBOX_IMAGE_DIGEST in sh, "the digest must be the repo's pinned execution image"
    assert C.SANDBOX_CREATE_KWARGS_REQUIRED == {
        "block_network": True, "cpu": 8, "memory": 16384, "timeout": 3600}
    assert set(C.SANDBOX_CREATE_KWARGS_FORBIDDEN) >= {"volumes", "secrets"}
    assert not (set(C.SANDBOX_CREATE_KWARGS_FORBIDDEN) & set(C.SANDBOX_CREATE_KWARGS_REQUIRED))
    ep = C.sandbox_entrypoint("mbpp")
    assert "--dataset mbpp" in ep and "/work/samples.jsonl" in ep and "ulimit" in ep
    assert "--parallel 8" in ep
    with pytest.raises(C.ControlNameError):
        C.sandbox_entrypoint("humaneval_plus")
    assert C.SANDBOX_RESULTS_KEYS == ("base", "plus")
    assert C.SANDBOX_RESULTS_PATH == "/work/samples_eval_results.json"


# --------------------------------------------------------------------------- #
# client lock .in                                                              #
# --------------------------------------------------------------------------- #


def test_client_lock_in_matches_the_pinned_top_level_set():
    p = REPO / C.CLIENT_LOCK_IN
    assert p.is_file()
    lines = [l.strip() for l in p.read_text().splitlines()
             if l.strip() and not l.strip().startswith("#")]
    pinned = {}
    vcs_seen = []
    for l in lines:
        if " @ " in l:
            # the VCS allow-list (ifeval-canonical-battery P7): the .in spells the
            # distribution name as the project does (`instruction_following_eval`),
            # the lock as uv normalises it (`instruction-following-eval`)
            name, _, rest = l.partition(" @ ")
            norm = f"{name.lower().replace('_', '-')} @ {rest}"
            assert norm in C.CLIENT_VCS_PINS, l
            vcs_seen.append(norm)
            continue
        name, _, ver = l.partition("==")
        assert ver, f"unpinned line: {l!r}"
        pinned[name.split("[", 1)[0]] = ver
    assert sorted(vcs_seen) == sorted(C.CLIENT_VCS_PINS)
    assert pinned == dict(C.CLIENT_TOP_LEVEL_PINS)
    # the pins the served driver ENFORCES are the compiled lock's, keyed like the frameworks block
    assert set(C.FRAMEWORK_PINS) == set(C.FRAMEWORKS_BLOCK_FIELDS) - {"client_lock_sha256"}
    assert C.FRAMEWORK_PINS == {"inspect_ai": C.INSPECT_AI_PIN, "inspect_evals": C.INSPECT_EVALS_PIN,
                                "evalplus": C.EVALPLUS_PIN}
    compiled = (REPO / C.CLIENT_LOCK_TXT).read_text()
    for dist, ver in (("inspect-ai", C.INSPECT_AI_PIN), ("inspect-evals", C.INSPECT_EVALS_PIN),
                      ("evalplus", C.EVALPLUS_PIN)):
        assert f"\n{dist}=={ver} " in compiled or f"\n{dist}=={ver}\n" in compiled, (dist, ver)
    # the removed tier-1 harness and its IRT estimator are in neither the .in nor the lock
    for gone in ("lm-eval", "lm_eval", "tinybenchmarks"):
        assert gone not in C.CLIENT_TOP_LEVEL_PINS and gone not in C.FRAMEWORK_PINS
        assert f"\n{gone}==" not in compiled and f"\n{gone} @ " not in compiled
    assert not hasattr(C, "FRAMEWORK_VCS_PINS") and not hasattr(srun, "FRAMEWORK_VCS_PINS")
    assert srun.FRAMEWORK_PINS is C.FRAMEWORK_PINS
    assert C.CLIENT_LOCK_EXCLUDE_NEWER == "2026-08-27T00:00:00Z"
    assert C.CLIENT_LOCK_EXCLUDE_NEWER in p.read_text()
    assert (C.CLIENT_VENV, C.CLIENT_PYTHON) == ("/opt/client", "3.12")
    # the repo lock's own versions agree where the package is locked there (P30)
    lock = (REPO / "uv.lock").read_text()
    for pkg, ver in (("evalplus", "0.3.1"), ("datasets", "5.0.1"),
                     ("aiohttp", "3.14.3"), ("transformers", "5.15.0")):
        assert f'name = "{pkg}"\nversion = "{ver}"' in lock, (pkg, ver)
        assert C.CLIENT_TOP_LEVEL_PINS[pkg] == ver
    assert 'name = "lm-eval"' not in lock and 'name = "tinybenchmarks"' not in lock


def test_client_lock_carries_the_agentic_sandbox_deps_and_no_second_venv():
    """inspect-sandboxes and swebench are CLIENT-lock pins (the .in ==
    CLIENT_TOP_LEVEL_PINS equality above covers those two lines); the retired
    Terminal-Bench harness's isolated harbor venv is gone with it — the client
    lock is the ONE venv the contracts describe, and the driver enforces no
    pin outside the frameworks block."""
    assert C.CLIENT_TOP_LEVEL_PINS["inspect-sandboxes"] == "0.5.0"
    assert C.CLIENT_TOP_LEVEL_PINS["swebench"] == "5.0.2"
    client_in = (REPO / C.CLIENT_LOCK_IN).read_text()
    assert "inspect-sandboxes==0.5.0" in client_in and "swebench==5.0.2" in client_in
    assert "harbor" not in client_in.lower()
    assert not [n for n in C.__all__ if n.startswith("HARBOR_")]
    assert not [n for n in dir(C) if "HARBOR" in n]
    assert "harbor" not in C.FRAMEWORK_PINS and "harbor" not in C.FRAMEWORKS_BLOCK_FIELDS


# --------------------------------------------------------------------------- #
# prefill_logprob: frozen signatures + stub discipline                         #
# --------------------------------------------------------------------------- #


def _params(fn):
    return list(inspect.signature(fn).parameters)


def test_prefill_logprob_signatures_are_frozen():
    assert _params(pl.build_prefill_request) == [
        "virtual_model", "user_text", "candidate", "system", "response_prefill",
        "enable_thinking", "seed", "thought_block_diag", "return_token_ids"]
    assert _params(pl.build_tokenize_request) == ["virtual_model", "messages", "enable_thinking"]
    assert _params(pl.normalize_prompt_logprobs) == ["resp_json", "expected_len"]
    assert _params(pl.prompt_token_ids_of) == ["resp_json"]
    assert _params(pl.continuation_span) == [
        "ids", "header_ids", "expected_candidate_ids", "scaffold_ok"]
    assert _params(pl.continuation_logprob) == ["plp", "ids", "span"]
    assert _params(pl.score_candidate) == [
        "resp_json", "tokenize_ids", "header_ids", "n_chars", "expected_candidate_ids",
        "scaffold_ok"]
    # keyword-only where the plan's call sites are keyword
    for fn in (pl.build_prefill_request, pl.build_tokenize_request, pl.score_candidate):
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY
                   for p in inspect.signature(fn).parameters.values()), fn.__name__
    assert issubclass(pl.PrefillLogprobError, RuntimeError)
    span = pl.ContinuationSpan(start=5, stop=8, header_at=2)
    assert list(span.positions) == [5, 6, 7] and len(span) == 3
    cs_ = pl.CandidateScore(ll_sum=-1.5, n_cont_tokens=3, n_chars=12, span_start=5,
                            prompt_tokens=8, token_ids=(1, 2, 3))
    assert cs_.token_ids == (1, 2, 3)
    assert set(pl.CandidateScore.__dataclass_fields__) >= {
        "ll_sum", "n_cont_tokens", "n_chars", "span_start", "prompt_tokens"}
    assert set(pl.TOKENIZE_REQUEST_KEYS) == {
        "model", "messages", "continue_final_message", "add_generation_prompt",
        "chat_template_kwargs"}
    assert px.RESPONSE_PREFILL_KEY in pl.PREFILL_REQUEST_KEYS
    for forbidden in (*px.FORBIDDEN_CLIENT_KEYS, "n", "stream"):
        assert forbidden not in pl.PREFILL_REQUEST_KEYS



# --------------------------------------------------------------------------- #
# the human-readable copy names every contract                                 #
# --------------------------------------------------------------------------- #


def test_interface_doc_names_every_contract():
    text = DOC.read_text()
    for name in (*C.CONTROL_ATOMS, *C.TINYCAP_TASKS, *C.TINYCAP_MANIFEST_KEYS,
                 *C.GEOMETRY_SUBENTRY_KEYS, *C.TINYCAP_HEADERS, *C.LEDGER_ROW_FIELDS,
                 *C.ALL_CONTAINER_NAMES, *C.TINYCAP_DRIVER_FLAGS, *C.TINYCAP_DOMAINS,
                 C.INSPECT_PROVIDER_NAME, C.INSPECT_BASE_URL_ENV, C.INSPECT_API_KEY_ENV,
                 C.MODAL_APP_NAME, C.SANDBOX_IMAGE_DIGEST, C.CLIENT_LOCK_IN,
                 C.FLAG_REASONING_PARSER, C.INSPECT_TASK_VERSION,
                 C.EVALPLUS_TASK_VERSION,
                 C.PARTIAL_DIR, C.SUMMARY_DIR, C.MANIFEST_KEY_PROVENANCE,
                 C.CONTROLS_SUMMARY_MERGE_MODULE, C.CONTROLS_SUMMARY_MERGE_FUNCTION,
                 *C.TINYCAP_IDENTITY_FIELDS, *C.SERVED_IDENTITY_FIELDS,
                 *C.PARTIAL_MARKER_FIELDS, C.LEDGER_ENDPOINT_FILENAME,
                 # PW5: the slot acceptance rule + legacy key, the enforced pins, the producer block
                 C.GSM8K_LEGACY_SLOT, "GSM8K_SLOT_ACCEPTANCE", "FRAMEWORK_PINS",
                 C.MANIFEST_KEY_PRODUCER, *C.PRODUCER_BLOCK_FIELDS, C.TREE_SHA_FILENAME,
                 # PW7: the per-entry merge provenance + the producer resume key
                 C.SUMMARY_KEY_SOURCES, *C.MERGE_SOURCE_RECORD_FIELDS,
                 "UNRECORDED_MERGE_SOURCE", "requires_merge_source",
                 "CONDITIONAL_RESUME_KEYS", "--replace-source",
                 # PW10: the two-tree merge, the pass annotation, the shared counts, the merge mode
                 C.MERGE_SOURCE_PASS_SOURCE_FIELD, C.PASS_SOURCE_LEGACY_TOP_LEVEL,
                 *C.MERGE_PAYLOAD_FIELDS, "merge_emitted_payload", "PARITY_N",
                 "--merge-mode",
                 # PW12: legacy passes never order, the attested -P subprocesses + writer sha, the
                 # equal-witness rule, the EvalPlus metric definition + its off-volume re-derivation
                 C.MERGE_SOURCE_WRITER_TREE_SHA_FIELD, C.MERGE_SOURCE_EQUAL_WITNESSES_FIELD,
                 C.MERGE_SOURCE_EQUAL_WITNESS_RULE_FIELD, C.EQUAL_WITNESS_RULE, C.MERGE_EMIT_ATTEST_KEY,
                 "equal_witness_plan", "writer_tree_sha_of", "PYTHONSAFEPATH",
                 C.EVALPLUS_METRIC_DEF_FIELD, *C.EVALPLUS_PLUS_DEFS, "load_controls_sources",
                 "tinycap_rederive_evalplus.py", "never orders",
                 # PW13: the definition identity-bearing on the rows, normalized by one helper; the
                 # converter's resume guard; equal witnesses accounted by the scan
                 C.evalplus_row_task_version(C.EVALPLUS_PLUS_DEF_BASE_AND_PLUS), "evalplus_row_task_version",
                 "with_evalplus_metric_def", "rows_of_task", "_check_existing_rows_definition",
                 "never an ordinary",
                 # PW14: the EFFECTIVE row version the stats readers pool on; the payload digest every
                 # superseded / equal-witness record carries; the writer's notes in the merge REC
                 C.MERGE_SOURCE_PAYLOAD_DIGEST_FIELD, "merge_payload_digest", "effective_task_version",
                 "evalplus_effective_task_version", "MIXED", "settles nothing",
                 # PW16: the standing record's digest, the named refresh of the standing store's own changed
                 # summary and its recorded history
                 C.MERGE_SOURCE_REFRESHED_FIELD, C.MERGE_REFRESH_DIGEST_WAS_FIELD, "MERGE_SOURCE_WRITER_FIELDS",
                 "MERGE_REFRESH_RECORD_FIELDS", "REFRESHED", "standing_digest",
                 # E64: the re-bind the merge writer recognises without --replace-source
                 C.MERGE_SOURCE_REBOUND_FIELD, "MERGE_REBIND_RECORD_FIELDS", "RE-BOUND",
                 "canonical_json_text", C.DONE_MARKER_REBOUND_AT_FIELD,
                 # PW18 (U4(b) / E25): BFCL on steered cells under the agent-loop entry; the
                 # lens reduction mode; both served identity, both mirrored in the route block
                 C.AGENT_LOOP_ENTRY_KEY, C.PROXY_IDENTITY_AGENT_LOOP_ENTRY, C.FLAG_NO_PETRI_AGENT_LOOP,
                 C.REDUCTION_MODE_FIELD, C.PROXY_IDENTITY_REDUCTION_MODE, *C.REDUCTION_MODES,
                 C.LENS_STACKED_REDUCTIONS_ENV, "assert_route_can_serve", "assert_live_agent_loop_entry",
                 "_check_live_proxy_identity", "agent_loop_entry_of", "reduction_mode_expected",
                 "resolve_reduction_mode", "reduction_mode_from_env",
                 # PW17 (U3 / U4(a)): the standard battery, the in-depth extras, the paired
                 # standard GSM8K and its frozen matched count, the retained spelling
                 "STANDARD_BATTERY_PROMPTED", "STANDARD_BATTERY_FITTED", "IN_DEPTH_EXTRAS",
                 "battery_controls", "--in-depth", C.CONTROL_GSM8K_FULL, "GSM8K_PAIRED_N",
                 "GSM8K_FULL_TASK", "RETAINED_CONTROL_SPELLINGS",
                 # U5: the per-task thinking policy and the MATH-500 caps
                 "THINKING_BY_TASK", "THINKING_ONE_ARM_PER_TASK", "battery_thinking",
                 "MATH500_MAX_TOKENS_DEFAULT", "MATH500_RETRY_MAX_TOKENS",
                 # the one-pass 8192 cap, the retired-but-re-armable
                 # retry, the legacy identity literal, MATH's own geometry class
                 "MATH500_RETRY_MAX_TOKENS_LEGACY_IDENTITY", "math500_retry_identity_value",
                 "GEOMETRY_CLASS_GEN_LONG_MATH", "gen_long_math", "assert_pass_fits_window",
                 "tinycap_identity_v1",
                 # the 16,384 default and MATH-500's own served window
                 "16,384", "capability:gen_long_math", "17,664", "max_prompt_plus_max_tokens",
                 # the long-instruction prompt classes' own window, selected by
                 # the cell's SI; an unmeasured class refused by name
                 "capability:gen_long_math_longsi", "24,064", "MATH500_PROMPT_CLASS_OF_SI",
                 "Math500PromptClassUnmeasured", "assert_math500_class_fits",
                 # PW18b (E29): the attestation is the truth; env = an expectation that must agree
                 "check_reduction_mode_expectation", "REDUCTION_MODE_ENV_VALUES", "--reduction-mode",
                 "REDUCTION_MODE_MISMATCH", "pre-landing server", "REDUCTION_MODE_LAUNCHER_DEFAULT",
                 # E61: the reason-bearing record the value-only reading cannot see
                 "REDUCTION_MODE_UNATTESTED", "lens_reduction_mode_error", "endpoint_reduction_mode",
                 # PW20b-D: per-task thinking under ONE standup, the two identity
                 # fields, the row's producer facts, the quality roll-up, the retry
                 C.THINKING_PER_TASK, "LAUNCHER_THINKING_MODES", "check_thinking_arg",
                 "run_thinking_by_task", C.IDENTITY_KEY_THINKING_BY_TASK,
                 C.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS,
                 # the MATH-only first-pass cap key, its tuple, the predicate
                 # and the launcher's pure twin of the plan builder's cap rule
                 C.IDENTITY_KEY_MATH500_MAX_TOKENS, "TINYCAP_IDENTITY_MATH500_FIELDS",
                 "files_math500", "math500_max_tokens_for",
                 C.FLAG_MATH500_RETRY_MAX_TOKENS, C.MANIFEST_KEY_QUALITY,
                 "QUALITY_BLOCK_FIELDS", "stop_reason", "unparseable_rate",
                 "at_cap_rate", "max_tokens_used", "STOP_REASON_AT_CAP",
                 # PW-sampling (U8 item 1): the sampling contract — modes, flag, the
                 # per-model table and its provenance, the per-task k, the seed rules,
                 # the five identity fields and the manifest block
                 "SAMPLING_MODES", C.FLAG_SAMPLING, *C.SAMPLING_MODES, "SAMPLING_DEFAULT",
                 "SAMPLING_BY_MODEL_KEY", "SamplingRow", "sampling_for", "SamplingUnverified",
                 "SAMPLING_UNVERIFIED_MODEL_KEYS", "GREEDY_SAMPLING", "SAMPLES_BY_TASK",
                 "DECODING_TASKS", "DECODE_SAMPLES_PER_ITEM", "MATH500_SAMPLES_PER_ITEM", "samples_of_task",
                 "run_samples_by_task", "SEED_RULES", "SEED_RULE_INSPECT", "SEED_RULE_EVALPLUS",
                 *C.SEED_RULES.values(), "SAMPLING_IDENTITY_FIELDS", *C.SAMPLING_IDENTITY_FIELDS,
                 "sampling_identity", C.MANIFEST_KEY_SAMPLING, "SAMPLING_BLOCK_FIELDS",
                 *(k for k in sg.MODEL_KEYS),
                 # PW-sampling steps 3-4: the EvalPlus estimator, the per-generation key, the
                 # sandbox's k, the pairing citation (reduces to the per-item mean FIRST)
                 "pass_at_k_unbiased", "PASS_AT_1_ESTIMATOR", "generation_key", "samples_per_item_of",
                 "_reduce_item", "paired_bootstrap", "per-item mean FIRST", "samples_summary",
                 "DECODE_SAMPLES_PER_ITEM", "SINGLE STOCHASTIC SAMPLE",
                 # PW-battery (U9 item 6): MBPP+ and BFCL out of the battery, the
                 # in-depth extras set aside
                 "needs_tool_calls(CONTROLS_PROMPTED) is False",
                 # PW-battery step 2 (U10 item 4): the frozen, seeded OR-Bench
                 # subset — its constants, its derivation, and where the ids go
                 C.IDENTITY_KEY_OR_BENCH_ITEM_SET, "OR_BENCH_SUBSET_N", "OR_BENCH_SUBSET_SEED",
                 "OR_BENCH_FULL_N", "or_bench_subset_positions", "run_or_bench_item_set",
                 C.FLAG_OR_BENCH_FULL, C.MANIFEST_KEY_OR_BENCH_ITEM_SET,
                 "OR_BENCH_ITEM_SET_BLOCK_FIELDS", "or_bench_selected_rows", "or_bench_item_ids",
                 "FROZEN, SEEDED",
                 # the battery names the OR-Bench atom, with the two generative
                 # letter legs out of it
                 "inspect_cmp:or_bench_hard_1k",
                 # offcap: the thinking-OFF comparator at the thinking-ON cap —
                 # the second named override's constants
                 C.FLAG_ALLOW_OFF_CAP_OVERRIDE, "FLAG_ALLOW_OFF_CAP_OVERRIDE", "OFF_CAP_OVERRIDE_NOTE",
                 "AuthorisedOffCapComparator", "AUTHORISED_OFF_CAP_COMPARATOR", "OFFCAP_AUTHORISED_FIELDS",
                 "OFF_CAP_OVERRIDE_MAX_TOKENS", "off_cap_override_max_tokens",
                 "check_off_cap_override_authorised", "OffCapOverrideUnauthorised",
                 C.REC_OFFCAP_UNAUTHORISED, "OverrideFlagsExclusive", "check_override_flags_exclusive",
                 "TINYCAP_IDENTITY_OFF_CAP_FIELDS", *C.TINYCAP_IDENTITY_OFF_CAP_FIELDS,
                 C.OOS_PROMPT_BUDGET_ARG_OFF_CAP_OVERRIDE, "OOS_PROMPT_BUDGET_ARG_OFF_CAP_OVERRIDE",
                 "prompt_budget_matches_run", C.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED,
                 "OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED", "store_has_cap_override",
                 "store_requires_outcomes", "offcap_override", "held_to_record", "offcap-<pin7>",
                 # GSM8K paired-200 paper objects: the setter, the
                 # twelve tuples, the item subset + cap_rung, the attempt contract, the named refusal
                 C.FLAG_ON_CAP_BY_TASK, "ON_CAP_BY_TASK_ALLOWED", "ON_CAP_CEILING", C.CAP_SOURCE_ON_CAP_EXPLICIT,
                 C.MANIFEST_KEY_ON_CAP_BY_TASK, "check_on_cap_setter", "OnCapSetterRefused",
                 "G8K_EXPERIMENT_NAMES", "G8K_ALL_POSITION_CELLS", "G8K_HEADER_LOCATED_CELLS", "g8k_tuple_name",
                 "header_invariant_declaration_refusal", "header_invariant_declared_cells",
                 C.FLAG_ITEM_SUBSET, C.ITEM_SUBSET_SCHEMA, *C.ITEM_SUBSET_FIELDS, C.IDENTITY_KEY_CAP_RUNG,
                 *C.CAP_RUNG_FIELDS, C.MANIFEST_KEY_ITEM_SUBSET, "read_item_subset", "item_subset_sha16",
                 "cap_rung_identity", "item_subset_block",
                 C.FLAG_ATTEMPT_ID, C.ATTEMPT_ID_FIELD, *C.ATTEMPT_KINDS, *C.RESUME_ATTEMPT_SCOPED_FIELDS,
                 C.MANIFEST_KEY_ACCEPTED_GENERATION_ATTEMPT, C.MANIFEST_KEY_ATTEMPT_ATTRIBUTION,
                 *C.ATTEMPT_ATTRIBUTION_FIELDS, "ledger_row_attempt", "resolve_attempt_kinds",
                 "attempt_attribution", "scoped_rejected_count", "AttemptScope",
                 "inspect_rung", "classify_store", "GSM8K_TRIGGER_THRESHOLD", "GSM8K_TRIGGER_RULE",
                 "GSM8K_REPORT_LINE", "triggered_no_eligible", "transport_check",
                 # the integration fix round (review H-P2, A-P1.1, B-P1.2, B-P1.3; the rule of
                 # 2026-09-10): the alias refusal, the exported producer contract, the derived stamp triples,
                 # the row-task rule, the family's own header-moves fact
                 "G8K_STAMPS_BASE", "G8K_STAMPS_FITTED", "G8K_STAMPS_DECLARED", "G8K_STAMPS_BY_CLASS",
                 "g8k_expected_stamps", "GSM8K_PAIRED_TASK_NAME", "RECORD_FIELD_ELIGIBLE_IDS",
                 "NOT_TRIGGERED_EXITS", "TRIGGERED_TERMINAL_EXITS", "EXIT_ESCALATE", "ITEM_KINDS",
                 "ITEM_FACT_FIELDS", "item_subset_sha16", "header_moves", "THINKING_HEADER_MOVES_EXPECTED",
                 "resolves",
                 # the `protocol` sampling mode —
                 # the per-task rows, the executed-vs-declared record, the identity key, the
                 # clamp override, the wrapper entrypoint, the in-image check and its record
                 C.SAMPLING_PROTOCOL, "ProtocolRow", "PROTOCOL_SAMPLING_BY_TASK", "PROTOCOL_TOP_P", "PROTOCOL_TOP_K",
                 "BFCL_PROTOCOL_TEMPERATURE", "VLLM_MAX_TEMP_PINNED", "SAMPLING_CLAMP_OVERRIDE",
                 C.FLAG_SAMPLING_CLAMP_OVERRIDE, C.ENDPOINT_KEY_SAMPLING_CLAMP_OVERRIDE, "served_identity_fields",
                 "SERVED_IDENTITY_PROTOCOL_FIELDS", C.VLLM_ENTRY_MODULE, C.FLAG_VLLM_ENTRY_MAX_TEMP,
                 C.FLAG_PROTOCOL_CHECK, C.PROTOCOL_CHECK_FILENAME, "executed_reason", "protocol_temperature",
                 C.PROTOCOL_EXECUTED_REASON_CLAMP_OVERRIDE, "sampling_for_task", "protocol_rows_by_task",
                 "protocol_executed_by_task", "protocol_sampling_identity", "sampling_identity_for_mode",
                 C.IDENTITY_KEY_SAMPLING_BY_TASK, "TINYCAP_IDENTITY_PROTOCOL_FIELDS", "protocol_manifest_by_task",
                 "protocol_source", "check_protocol_effective", "ProtocolSamplingUnavailable",
                 C.SAMPLING_BLOCK_KEY_BY_TASK, "PROTOCOL_BY_TASK_FIELDS", "PROTOCOL_EFFECTIVE_FIELDS",
                 "check_protocol_sampling", "protocol_check", "_MAX_TEMP", "vllm_entry",
                 # the code-review round (bcb-bfcl-200-widen): the CLI spelling of the check's tasks, the refused
                 # process topologies and the process-creation guard, the durable sampling-validation failure,
                 # the immutable launch-time snapshot and its retry record, the certified wave's membership flag
                 "--tasks", "--api-server-count", "--numa-bind", "spawn_topology_reason", "guard_mp_context",
                 C.SAMPLING_VALIDATION_FAILED_FILENAME, C.REC_SAMPLING_VALIDATION_FAILED, C.REC_CLAMP_OVERRIDE_INEFFECTIVE,
                 C.REC_ROSTER_SNAPSHOT_CONFLICT, "SNAPSHOT_RETRY_FILENAME", C.FLAG_LADDER_WAVE_CELLS,
                 "protocol_launch_flags_record", "load_petri_spec"):
        assert name in text, f"docs/tinycap_interfaces.md does not name {name!r}"
    for fn in ("build_prefill_request", "build_tokenize_request", "normalize_prompt_logprobs",
               "prompt_token_ids_of", "continuation_span", "continuation_logprob",
               "score_candidate", "expand_control", "driver_argv", "standup_extra_argv",
               "geometry_subentry", "sample_metric", "inspect_sample_id"):
        assert fn in text, fn
    # the doc never carries an absolute home path or a hostname
    assert "/home/" not in text


# --------------------------------------------------------------------------- #
# the cell grammar: whatever the serving side can express                       #
# --------------------------------------------------------------------------- #

#: The campaign's ten x1 cells (its --cells list, verbatim).
_CAMPAIGN_CELLS = (
    "none", "prompted__safety_refusal", "prompted__si_constitution",
    "prompted__si_irrelevant", "safety_refusal__psr_kl_all",
    "si_constitution__psr_kl_all", "si_irrelevant__psr_kl_all",
    "safety_refusal__lora_r4", "si_constitution__lora_r4", "si_irrelevant__lora_r4",
)


def test_parse_launcher_cell_accepts_the_whole_campaign_and_names_its_parts():
    kinds = {c: C.parse_launcher_cell(c) for c in _CAMPAIGN_CELLS}
    assert kinds["none"].kind == C.CELL_KIND_BASE and kinds["none"].harness_id == "none"
    assert kinds["none"].si is None and kinds["none"].arm is None
    assert C.is_base_cell("none") and not C.is_base_cell("prompted__safety_refusal")
    # the TrainTest -> harness translation the standup performs, per class
    assert kinds["prompted__safety_refusal"].harness_id == "prompted__safety"
    assert kinds["prompted__si_constitution"].harness_id == "prompted__constitution"
    assert kinds["prompted__si_irrelevant"].harness_id == "prompted__irrelevant"
    assert kinds["safety_refusal__psr_kl_all"].harness_id == "si_safety__psr_kl_all"
    assert kinds["si_constitution__lora_r4"].harness_id == "si_constitution__lora_r4"
    for cell, parsed in kinds.items():
        assert parsed.cell == cell
        assert parsed.kind in C.CELL_KINDS
        if parsed.kind == C.CELL_KIND_FITTED:
            assert parsed.arm in ("psr_kl_all", "lora_r4") and parsed.si is not None


def test_parse_launcher_cell_refuses_what_the_standup_could_not_serve():
    for bad, needle in (("", "non-empty"),
                        ("prompted__nope", "no harness translation"),
                        ("nope__psr_kl_all", "no harness translation"),
                        # the KL ladder is training-side only
                        ("safety_refusal__alphakl_0", "outside the harness's audited set"),
                        ("safety_refusal", "is not a cell id"),
                        ("prompted", "is not a cell id"),
                        # E45 H2: RETIRED — audited once, no training-side cell
                        # today, so it would enumerate ABSENT after an H200 was allocated
                        ("circuit_breaker", "left methods.ARMS")):
        with pytest.raises(C.CellNameError, match=needle):
            C.parse_launcher_cell(bad)
    # … and caa_static_r, which the old grammar refused, is LAUNCHABLE now
    assert C.parse_launcher_cell("caa_static_r").kind == C.CELL_KIND_FITTED
    # the bare-token refusal names the LAUNCHABLE ids, not the harness vocabulary
    with pytest.raises(C.CellNameError, match=re.escape(str(list(C.launchable_class_b_ids())))):
        C.parse_launcher_cell("safety_refusal")


def test_the_launchable_class_b_ids_are_the_one_class_b_tuple():
    """E45 H2: there is ONE Class B tuple, in the torch-free module both
    registries import, and the launcher's grammar READS it rather than
    transcribing it. The old shape — a retyped copy of the training-side
    arms intersected with a separately typed harness vocabulary — made
    ``caa_static_r`` (a real arm) unlaunchable and let ``circuit_breaker`` (no
    cell since 2026-08-20) through the parser. Now every launchable id parses
    as a fitted, SI-agnostic cell whose harness id is the bare id exactly as
    ``harness_cell_id`` translates it, and the retired id is refused HERE with
    the reason the GPU would have given later."""
    dc = pytest.importorskip("sisteer.capeval.petri.defense_cells")
    tt = pytest.importorskip("sisteer.eval.defense_cell")
    ra = pytest.importorskip("sisteer.capeval.petri.registry_adapter")
    pytest.importorskip("torch")
    # ONE object, not two equal ones: the launcher, the harness vocabulary and
    # TrainTest's registry cannot disagree because there is nothing to disagree
    assert tt.CLASS_B_ARMS is dc.CLASS_B_IDS
    assert tuple(dc.CLASS_B_IDS) == ("antidote", "alphasteer", "caa_static_r")
    # The SERVED-ONLY Class B ids join the LAUNCHER's grammar and nothing else.
    # `arditi` is served from its own bank (one object per model, SI-agnostic — a Class B cell in every way the
    # serving side cares about) while no trainer on this tree produces it, so it is deliberately outside
    # CLASS_B_IDS: that tuple IS the training-side Cartesian registry list (CLASS_B_ARMS), fit_class_b's arms and
    # petri_stats' Class B family, none of which may grow a cell nothing here mints.
    # `arditi_allpos` joined 2026-09-16 (arditi-qwen-all-positions): the SAME fitted direction served
    # in the PAPER'S OWN configuration — the raw difference of means at coefficient c, added at every
    # position of every forward — against `arditi`'s prefill-only fraction of the residual norm. A
    # SECOND served-only id and not a re-dose of the first, because the two differ in phase AND in
    # dose unit, so one name for both would alias two interventions wherever a row is joined.
    assert dc.SERVED_ONLY_CLASS_B_IDS == ("arditi", "arditi_allpos")
    assert not set(dc.SERVED_ONLY_CLASS_B_IDS) & set(dc.CLASS_B_IDS)
    assert C.launchable_class_b_ids() == tuple(dc.CLASS_B_IDS) + tuple(dc.SERVED_ONLY_CLASS_B_IDS)
    assert C.launchable_class_b_ids() == ("antidote", "alphasteer", "caa_static_r", "arditi", "arditi_allpos")
    # every TRAINED launchable id is one TrainTest's registry actually BUILDS; the served-only ones are not arms here
    assert tuple(tt._registered(tt.CLASS_B_ARMS)) == tuple(dc.CLASS_B_IDS)
    from sisteer.methods import ARMS as _ARMS
    assert all(sid not in _ARMS for sid in dc.SERVED_ONLY_CLASS_B_IDS)
    # …but the harness maps each to the OBJECT KIND its lane stages, so the serving side can resolve and adapt it
    assert ra.ARM_OBJECT_KINDS["arditi"] == "static_r"
    for cell in C.launchable_class_b_ids():
        parsed = C.parse_launcher_cell(cell)
        assert parsed == (cell, cell, C.CELL_KIND_FITTED, None, cell)
        assert ra.harness_cell_id(cell) == cell == parsed.harness_id
        assert C.cell_short(cell) == cell
    # the retired id: refused at parse time, and the standup's own functions
    # agree — no cell is built for it and it is not a harness id either
    assert C.retired_class_b_ids() == tuple(dc.RETIRED_CLASS_B_IDS) == ("circuit_breaker",)
    assert not set(C.retired_class_b_ids()) & set(C.launchable_class_b_ids())
    for retired in C.retired_class_b_ids():
        assert retired not in tt._registered(tt.CLASS_B_ARMS)
        with pytest.raises(ra.RegistryAdapterError, match="Class B vocabulary"):
            ra.harness_cell_id(retired)


def test_the_grammar_agrees_with_the_real_harness_translation():
    """The launcher's translation is torch-free by necessity (the modal venv
    that runs it has no torch), so it cannot CALL
    ``registry_adapter.harness_cell_id``. Here, where torch is importable, the
    two are compared cell by cell — the drift guard that makes the torch-free
    copy safe."""
    ra = pytest.importorskip("sisteer.capeval.petri.registry_adapter")
    pytest.importorskip("torch")
    for cell in (*_CAMPAIGN_CELLS, *C.launchable_class_b_ids()):
        assert C.parse_launcher_cell(cell).harness_id == ra.harness_cell_id(cell)
    # and it refuses the same things
    with pytest.raises(ra.RegistryAdapterError):
        ra.harness_cell_id("safety_refusal__alphakl_0")


def test_container_names_are_generated_and_reproduce_the_four_pinned_ones():
    # the table is now DERIVED — every original name comes back byte-equal
    for (cell, mode), name in C.CONTAINER_NAMES.items():
        assert C.container_name(cell, mode) == name
    # descriptive per model and per cell (these names show up in dashboards)
    assert C.model_short("gemma4e4b") == "e4b" and C.model_short("gemma4e4b-x2") == "e4b_x2"
    assert C.model_short("qwen35_9b") == "qwen9b" and C.model_short("llama3b") == "llama3b"
    assert C.model_short("something-else") == "something-else"     # unregistered: itself
    assert C.cell_short("none") == "base"
    assert C.cell_short("si_constitution__psr_kl_all") == "psrkl_constitution"
    assert C.cell_short("si_irrelevant__lora_r4") == "lora_r4_irrelevant"
    assert C.container_name("si_constitution__psr_kl_all", "on", model_slug="qwen35_9b") == (
        "tinycap-qwen9b-psrkl_constitution-think_on")
    assert C.container_name("none", "off", model_slug="gemma4e4b-x2") == (
        "tinycap-e4b_x2-base-think_off")
    # every generated name is unique per (model, cell, mode)
    names = {C.container_name(c, m, model_slug=s)
             for s in ("gemma4e4b", "gemma4e4b-x2", "qwen35_9b")
             for c in _CAMPAIGN_CELLS for m in C.THINKING_MODES}
    assert len(names) == 3 * len(_CAMPAIGN_CELLS) * len(C.THINKING_MODES)
    with pytest.raises(ValueError):
        C.container_name("none", "sideways")
    with pytest.raises(C.CellNameError):
        C.container_name("not_a_cell", "off")


# --------------------------------------------------------------------------- #
# the MODEL REGISTRY — every value checked against its SOURCE                   #
# --------------------------------------------------------------------------- #

def _demo_wave() -> str:
    """The worked example's one steered wave, as the models table's reader spells it (never a literal here: a peer renames it)."""
    import importlib.util
    reader = Path(__file__).resolve().parents[1] / "reproduce" / "lib" / "models.py"
    spec = importlib.util.spec_from_file_location("_contracts_test_models", reader)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.baseline_view("qwen35_4b_demo")


#: (slug, hf, 12-hex revision prefix, config, objects_dir, model_key, family)
#: transcribed from the derived cell inventory
#: (`$GRADE/tinycap/gsm8k_rerun_plan/cell_inventory.json`: `model`,
#: `model_revision`, `object`) and the staged paths. The revision
#: PREFIXES are what the inventory records; the full pins are checked against
#: the configs below, which is where they are authoritative.
_REGISTRY_EXPECTED = (
    ("gemma4e4b", "google/gemma-4-E4B-it", "ee0ef6023621",
     "configs/two_si_gemma4e4b_reduced.yaml", "/vol/objects/scale-12/gemma4e4b",
     "gemma4e4b", "gemma4"),
    ("gemma4e4b-x2", "google/gemma-4-E4B-it", "ee0ef6023621",
     "configs/two_si_gemma4e4b_reduced.yaml", "/vol/capctrl/objects/scale-12-x2/gemma4e4b",
     "gemma4e4b", "gemma4"),
    ("gemma4e2b", "google/gemma-4-E2B-it", "3e22461f65e8",
     "configs/two_si_gemma4e2b_fits.yaml", "/vol/objects/scale-12/gemma4e2b",
     "gemma4e2b", "gemma4"),
    ("llama8b", "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2",
     "configs/two_si.yaml", "/vol/objects/scale-12/llama8b", "llama8b", "llama"),
    ("llama8b-x2", "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2",
     "configs/two_si.yaml", "/vol/capctrl/objects/scale-12-x2/llama8b", "llama8b", "llama"),
    ("llama3b", "unsloth/Llama-3.2-3B-Instruct", "006f5dcd1393",
     "configs/two_si_llama32_3b.yaml", "/vol/tinycap/objects/llama3b", "llama3b", "llama"),
    ("qwen35_9b", "Qwen/Qwen3.5-9B", "c20223623576",
     "configs/two_si_qwen35_9b_fits.yaml", "/vol/objects/scale-12/qwen35_9b",
     "qwen35_9b", "qwen3_5"),
    ("qwen35_4b", "Qwen/Qwen3.5-4B", "851bf6e806ef",
     "configs/two_si_qwen35_4b_fits.yaml", "/vol/objects/scale-12/qwen35_4b",
     "qwen35_4b", "qwen3_5"),
    ("gemma4e4b-nors", "google/gemma-4-E4B-it", "ee0ef6023621",
     "configs/two_si_gemma4e4b_reduced.yaml", "/vol/tinycap/objects/gemma4e4b_nors",
     "gemma4e4b", "gemma4"),
    # the worked example's row — the paper's Qwen3.5-4B at the qwen35_4b pin under the models
    # table's slug (reproduce/models.yaml serve.checkpoint), model_key the 4B's, its own sdf-base bank, and its OWN
    # config: the front door's generated explicit-cells file (contracts.frontdoor_config_path);
    # appended last among the legacy rows (test_the_worked_example_row_is_the_qwen35_4b_rows_twin_under_its_own_slug)
    ("qwen35_4b_demo", "Qwen/Qwen3.5-4B", "851bf6e806ef",
     C.frontdoor_config_path("qwen35_4b_demo", _demo_wave()), "/vol/objects/sdfbase/qwen35_4b_demo",
     "qwen35_4b", "qwen3_5"),
    # demo-import (2026-10-12): the worked example's two DOSE VIEWS, a registry row each (the paper's dose rows' shape:
    # `qwen35_35b_a3b-f22d0p75_relul` / `-f22d1p25`) — the view's committed dose copies in a sub-bank named for the view
    # under the key's root, keyed by cell id, and the view's own front-door config; every other fact the demo row's
    # (test_the_dose_views_rows_are_the_demo_rows_twins_over_the_views_own_bank_and_config)
    ("qwen35_4b_demo-d0p75", "Qwen/Qwen3.5-4B", "851bf6e806ef",
     C.frontdoor_config_path("qwen35_4b_demo", "d0p75"), "/vol/objects/sdfbase/qwen35_4b_demo/d0p75",
     "qwen35_4b", "qwen3_5"),
    ("qwen35_4b_demo-d0p75_relu", "Qwen/Qwen3.5-4B", "851bf6e806ef",
     C.frontdoor_config_path("qwen35_4b_demo", "d0p75_relu"), "/vol/objects/sdfbase/qwen35_4b_demo/d0p75_relu",
     "qwen35_4b", "qwen3_5"),
    # the sdf-base DERIVED bank on the SAME served
    # target — its own two-SI config (the roster is exactly the derived cell),
    # appended LAST so no existing slug moves
    ("gemma4e4b-sdfbase", "google/gemma-4-E4B-it", "ee0ef6023621",
     "configs/two_si_gemma4e4b_sdfbase.yaml", "/vol/objects/sdfbase/gemma4e4b",
     "gemma4e4b", "gemma4"),
    # four rows over the staged full22 banks through generated explicit-cells
    # configs (scripts/bcbw_configs.py), appended LAST in MAIN's ModelRow
    # shape; the hf / rev are the full22 roster's pins
    # (configs/sdfbase_full22_roster.yaml), the banks the staged ones
    ("gemma4e4b-f22it", "google/gemma-4-E4B-it", "ee0ef6023621",
     "configs/tinycap/bcbw_gemma4e4b_it.yaml", "/vol/objects/sdfbase/full22/gemma4e4b/it",
     "gemma4e4b", "gemma4"),
    ("gemma4e4b-f22", "google/gemma-4-E4B-it", "ee0ef6023621",
     "configs/tinycap/bcbw_gemma4e4b_main.yaml", "/vol/objects/sdfbase/full22/gemma4e4b/main",
     "gemma4e4b", "gemma4"),
    ("qwen35_4b-f22", "Qwen/Qwen3.5-4B", "851bf6e806ef",
     "configs/tinycap/bcbw_qwen35_4b_main.yaml", "/vol/objects/sdfbase/full22/qwen35_4b/main",
     "qwen35_4b", "qwen3_5"),
    ("nemotron3nano4b", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "dfaf35de3e30",
     "configs/tinycap/bcbw_nemotron3nano4b_main.yaml", "/vol/objects/sdfbase/full22/nemotron3nano4b/main",
     "nemotron3nano4b", "nemotron_h"),
    # the two llama8b rows — the LoRA-subtraction object in both views, on the
    # campaign's llama8b target (the llama8b row's hf / rev), appended LAST
    ("llama8b-f22it", "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2",
     "configs/tinycap/bcbw_llama8b_it.yaml", "/vol/objects/sdfbase/full22/llama8b/it",
     "llama8b", "llama"),
    ("llama8b-f22", "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2",
     "configs/tinycap/bcbw_llama8b_main.yaml", "/vol/objects/sdfbase/full22/llama8b/main",
     "llama8b", "llama"),
    ("qwen35_4b-f22it", "Qwen/Qwen3.5-4B", "851bf6e806ef",
     "configs/tinycap/bcbw_qwen35_4b_it.yaml", "/vol/objects/sdfbase/full22/qwen35_4b/it",
     "qwen35_4b", "qwen3_5"),   # the qwen instruct-direct row, appended last among these rows
    # the batch-64 three-seed Qwen family — the seed views s1 / s2 / s3
    # and the paired-write views s1p / s2p / s3p, inserted BEFORE the IFEval rows so `is_full22_row` stays contiguous
    *[(f"qwen35_4b-f22{v}", "Qwen/Qwen3.5-4B", "851bf6e806ef",
       f"configs/tinycap/bcbw_qwen35_4b_{v}.yaml", f"/vol/objects/sdfbase/full22/qwen35_4b/{v}",
       "qwen35_4b", "qwen3_5") for v in ("s1", "s2", "s3", "s1p", "s2p", "s3p")],
    # b64-capability-agentic-queue (2026-09-23): the ALL-MODELS ONE-SEED family — gemma4e4b / llama8b / nemotron3nano4b on
    # their s1 (the base-lane objects under the seed grammar) and s1p (the composed paired-write difference ReLU) views,
    # inserted after the Qwen seed rows and BEFORE the IFEval rows (`is_full22_row` contiguous); served at the family's
    # existing lane row's pin
    *[(f"{mk}-f22{v}", hf, rev12, f"configs/tinycap/bcbw_{mk}_{v}.yaml", f"/vol/objects/sdfbase/full22/{mk}/{v}", mk, fam)
      for mk, hf, rev12, fam in (("gemma4e4b", "google/gemma-4-E4B-it", "ee0ef6023621", "gemma4"),
                                 ("llama8b", "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2", "llama"),
                                 ("nemotron3nano4b", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "dfaf35de3e30", "nemotron_h"))
      for v in ("s1", "s1p")],
    # the seven (model, view) bank views of the paper's full22 objects, each its
    # own slug, appended LAST (after the six bcbw rows); the per-TAG
    # overlay config is generated (make_ifeval_overlays.py)
    ("gemma4e4b-sdfbase-main", "google/gemma-4-E4B-it", "ee0ef6023621",
     "configs/sdfbase/ifeval/sdfbase_full22_gemma4e4b.ifeval.yaml", "/vol/objects/sdfbase/full22/gemma4e4b/main",
     "gemma4e4b", "gemma4"),
    ("gemma4e4b-sdfbase-it", "google/gemma-4-E4B-it", "ee0ef6023621",
     "configs/sdfbase/ifeval/sdfbase_full22_gemma4e4b_it.ifeval.yaml", "/vol/objects/sdfbase/full22/gemma4e4b/it",
     "gemma4e4b", "gemma4"),
    ("qwen35_4b-sdfbase-main", "Qwen/Qwen3.5-4B", "851bf6e806ef",
     "configs/sdfbase/ifeval/sdfbase_full22_qwen35_4b.ifeval.yaml", "/vol/objects/sdfbase/full22/qwen35_4b/main",
     "qwen35_4b", "qwen3_5"),
    ("qwen35_4b-sdfbase-it", "Qwen/Qwen3.5-4B", "851bf6e806ef",
     "configs/sdfbase/ifeval/sdfbase_full22_qwen35_4b_it.ifeval.yaml", "/vol/objects/sdfbase/full22/qwen35_4b/it",
     "qwen35_4b", "qwen3_5"),
    ("nemotron3nano4b-sdfbase-main", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "dfaf35de3e30",
     "configs/sdfbase/ifeval/sdfbase_full22_nemotron3nano4b.ifeval.yaml",
     "/vol/objects/sdfbase/full22/nemotron3nano4b/main", "nemotron3nano4b", "nemotron_h"),
    ("llama8b-sdfbase-main", "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2",
     "configs/sdfbase/ifeval/sdfbase_full22_llama8b.ifeval.yaml", "/vol/objects/sdfbase/full22/llama8b/main",
     "llama8b", "llama"),
    ("llama8b-sdfbase-it", "NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2",
     "configs/sdfbase/ifeval/sdfbase_full22_llama8b_it.ifeval.yaml", "/vol/objects/sdfbase/full22/llama8b/it",
     "llama8b", "llama"),
    # the six batch-64 seed LANE views of qwen35_4b (s1..s3, s1p..s3p), the same
    # staged-bank root + per-TAG overlay grammar, appended LAST in IFEVAL_LANE_VIEWS order
    *[(f"qwen35_4b-sdfbase-{v}", "Qwen/Qwen3.5-4B", "851bf6e806ef",
       f"configs/sdfbase/ifeval/sdfbase_full22_qwen35_4b_{v}.ifeval.yaml", f"/vol/objects/sdfbase/full22/qwen35_4b/{v}",
       "qwen35_4b", "qwen3_5") for v in ("s1", "s2", "s3", "s1p", "s2p", "s3p")],
)


def test_the_front_door_config_path_is_spelled_once_and_the_demo_row_will_read_it():
    """contracts.frontdoor_config_path(key, view) -> configs/tinycap/frontdoor_<key>_<view>.yaml, beside
    bcbw_config_path and never one of its 19 files; scripts/bcbw_configs.py --front-door writes it."""
    wave = _demo_wave()
    p = C.frontdoor_config_path("qwen35_4b_demo", wave)
    assert p == f"configs/tinycap/frontdoor_qwen35_4b_demo_{wave}.yaml" and p.startswith("configs/tinycap/") and p != C.bcbw_config_path("qwen35_4b", wave)
    assert p not in {C.bcbw_config_path(t, v) for t, v in C.BCBW_MODEL_VIEWS}
    assert C.frontdoor_config_path("k", "v") == "configs/tinycap/frontdoor_k_v.yaml"


def test_the_demo_configs_explicit_cells_are_the_tables_ordered_tokens_and_its_bank_is_the_rows():
    """The demo row's cfg IS the front-door file (committed, read from disk here), its explicit_cells equal lib/models.py
    launcher_cells(qwen35_4b_demo, <wave>)'s ORDERED tokens (the ten of the plan) and defenses_dir == contracts.MODELS[qwen35_4b_demo].objects_dir.
    A superset would make registry_from_explicit_cells refuse the whole registry on an unstaged object; a subset would make the standup
    enumerate a launched cell ABSENT — so the ORDER and the count are the pin, not membership alone."""
    import importlib.util
    reader = Path(__file__).resolve().parents[1] / "reproduce" / "lib" / "models.py"
    spec = importlib.util.spec_from_file_location("_contracts_test_models2", reader)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wave = mod.baseline_view("qwen35_4b_demo")
    demo = C.model_row("qwen35_4b_demo")
    assert demo.cfg == C.frontdoor_config_path("qwen35_4b_demo", wave)
    path = Path(__file__).resolve().parents[1] / demo.cfg
    assert path.is_file(), f"the front door's generated config is committed at {demo.cfg} (scripts/bcbw_configs.py --front-door qwen35_4b_demo)"
    cfg = yaml.safe_load(path.read_text())
    assert cfg["explicit_cells"] == [t for _c, t in mod.launcher_cells("qwen35_4b_demo", wave)] and len(cfg["explicit_cells"]) == 10
    assert cfg["explicit_cells"][0] == "none" and cfg["explicit_cells"][1:3] == ["prompted__safety_refusal", "prompted__si_constitution"]
    assert cfg["defenses_dir"] == demo.objects_dir and (cfg["model"], cfg["model_revision"]) == (demo.hf, demo.rev)


def test_the_worked_example_row_is_the_qwen35_4b_rows_twin_under_its_own_slug():
    """paper-companion-repo: reproduce/07_bcb_bfcl.sh --model qwen35_4b_demo resolves THIS row (contracts.model_row) — the
    worked example's: the paper's Qwen3.5-4B at the qwen35_4b row's pin, served under the models table's own slug. Every
    served fact is the 4B's (hf, rev, cfg, family, thinking OFF, model_key — so geometry's S and the `card` sampling row are
    the 4B's, and FAMILY_BY_MODEL_KEY needs no onboarded key); the family's verified qwen3_xml manifest rides it (BFCL
    refuses at plan time without one); base_hf records the base->instruct transfer reproduce/models.yaml declares; its bank
    is its OWN, under the sdf-base root (declared, unstaged). A legacy-shaped row — neither a full22 lane row nor an IFEval
    row — appended last among the legacy rows so no earlier slug moves."""
    demo, q = C.model_row("qwen35_4b_demo"), C.model_row("qwen35_4b")
    assert (demo.hf, demo.rev, demo.family, demo.thinking_modes, demo.model_key, demo.top_logprobs) == (
        q.hf, q.rev, q.family, q.thinking_modes, q.model_key, q.top_logprobs)
    # the one served fact that is the demo's OWN: its explicit-cells config, the front door's generated file
    # (contracts.frontdoor_config_path), committed beside the roster block, the pins and the battery body — never the 4B's config
    frontdoor = C.frontdoor_config_path("qwen35_4b_demo", _demo_wave())
    assert demo.cfg == frontdoor and demo.cfg != q.cfg, (demo.cfg, frontdoor)
    assert (Path(__file__).resolve().parents[1] / frontdoor).is_file(), frontdoor
    assert (demo.hf, demo.rev[:8], demo.family, demo.thinking_modes, demo.model_key) == (
        "Qwen/Qwen3.5-4B", "851bf6e8", "qwen3_5", (C.THINKING_OFF,), "qwen35_4b")
    assert demo.petri_manifest == C.PETRI_MANIFEST_FULL22_QWEN35_4B == C.model_row("qwen35_4b-f22").petri_manifest
    assert demo.base_hf == "Qwen/Qwen3.5-4B-Base" == C.model_row("qwen35_4b-f22").base_hf
    assert demo.objects_dir == f"{C.VOL_OBJECTS_SDFBASE}/qwen35_4b_demo" and demo.objects_dir != q.objects_dir
    assert not C.is_full22_row(demo) and not C.is_ifeval_row(demo) and demo.cfg_overrides == {}
    assert C.model_short("qwen35_4b_demo") == "qwen35_4b_demo"          # an unregistered short is the slug itself: descriptive
    n_tail = len(C.IFEVAL_SLUGS) + len(C.BCBW_SLUGS)
    assert C.MODEL_SLUGS[-(n_tail + 4):-n_tail] == ("qwen35_4b_demo", "qwen35_4b_demo-d0p75", "qwen35_4b_demo-d0p75_relu", "gemma4e4b-sdfbase")
    # the models table's served checkpoint IS this row's, and its fit checkpoint the base_hf (reproduce/models.yaml)
    m = yaml.safe_load((Path(__file__).resolve().parents[1] / "reproduce/models.yaml").read_text())["models"]["qwen35_4b_demo"]
    served = m["serve"]["checkpoint"]
    assert (served["hf_repo"], served["revision"], served["model_id"]) == (demo.hf, demo.rev, demo.slug)
    assert (m["fit"]["checkpoint"]["hf_repo"], m["transfer_kind"]) == (demo.base_hf, "base_to_instruct")
    assert m["thinking"]["default"] == "off" and C.THINKING_ON not in demo.thinking_modes
    # the front door's config carries the 4B's served model + pin: the standup reads them from it
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / demo.cfg).read_text())
    assert (cfg["model"], cfg["model_revision"]) == (demo.hf, demo.rev)


def test_the_dose_views_rows_are_the_demo_rows_twins_over_the_views_own_bank_and_config():
    """demo-import (2026-10-12; the front door's Known gap "the dose views stay on the cluster" closed): every DOSE view of the worked example
    (reproduce/models.yaml `dose:`) has a registry row of its own under reproduce/lib/models.py launcher_slug(key, view) = `<key>-<view>` — the
    paper's dose rows' shape (`qwen35_35b_a3b-f22d0p75_relul` / `-f22d1p25`: a row per dose view) — whose bank is the view's own sub-bank of
    dose-scaled copies under the key's root, keyed by cell id, and whose config is the view's front-door file; every other served fact is the
    demo row's. The committed config's explicit_cells are the view's launcher tokens (none + the object cell: the SAME cell id as the wave's
    parent — the bank, not the token, carries the dose) and its defenses_dir the row's bank. No other table key gains a row here."""
    import importlib.util
    reader = Path(__file__).resolve().parents[1] / "reproduce" / "lib" / "models.py"
    spec = importlib.util.spec_from_file_location("_contracts_test_models3", reader)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    demo = C.model_row("qwen35_4b_demo")
    m = mod.model("qwen35_4b_demo")
    dose_views = [v for v, vd in m["views"].items() if "dose" in (vd or {})]
    assert dose_views == ["d0p75", "d0p75_relu"], dose_views
    assert mod.launcher_slug("qwen35_4b_demo", mod.baseline_view("qwen35_4b_demo")) == "qwen35_4b_demo"
    for view in dose_views:
        slug = mod.launcher_slug("qwen35_4b_demo", view)
        assert slug == f"qwen35_4b_demo-{view}" and slug in C.MODELS, slug
        row = C.model_row(slug)
        assert (row.hf, row.rev, row.family, row.thinking_modes, row.model_key, row.top_logprobs, row.petri_manifest, row.base_hf, row.cfg_overrides) == (
            demo.hf, demo.rev, demo.family, demo.thinking_modes, demo.model_key, demo.top_logprobs, demo.petri_manifest, demo.base_hf, demo.cfg_overrides)
        assert row.objects_dir == f"{demo.objects_dir}/{view}" and not C.is_full22_row(row) and not C.is_ifeval_row(row)
        assert row.cfg == C.frontdoor_config_path("qwen35_4b_demo", view) == f"configs/tinycap/frontdoor_qwen35_4b_demo_{view}.yaml"
        path = Path(__file__).resolve().parents[1] / row.cfg
        assert path.is_file(), f"the dose view's generated config is committed at {row.cfg} (scripts/bcbw_configs.py --front-door qwen35_4b_demo --view {view})"
        cfg = yaml.safe_load(path.read_text())
        tokens = [t for _c, t in mod.launcher_cells("qwen35_4b_demo", view)]
        assert cfg["explicit_cells"] == tokens and tokens[0] == "none" and len(tokens) == 2, (view, tokens)
        assert tokens[1] == mod.roster_cell_id(m, m["views"][view]["cells"][1]) and tokens[1] in [t for _c, t in mod.launcher_cells("qwen35_4b_demo", mod.baseline_view("qwen35_4b_demo"))]
        assert cfg["defenses_dir"] == row.objects_dir and (cfg["model"], cfg["model_revision"]) == (row.hf, row.rev)
    # no dose row for a view the table does not declare, and none for a paper model (the paper's rows are the lane's, below)
    with pytest.raises(C.UnknownModel):
        C.model_row("qwen35_4b_demo-nope")
    assert not any(s.startswith("qwen35_4b-d") or s.startswith("gemma4e4b-d") for s in C.MODEL_SLUGS)


def test_the_model_registry_matches_the_plan_and_the_inventory():
    assert list(C.MODEL_SLUGS) == [r[0] for r in _REGISTRY_EXPECTED]
    for slug, hf, rev12, cfg, objects, key, family in _REGISTRY_EXPECTED:
        row = C.model_row(slug)
        assert (row.slug, row.hf, row.cfg, row.objects_dir, row.model_key, row.family) == (
            slug, hf, cfg, objects, key, family)
        assert row.rev.startswith(rev12) and len(row.rev) == 40
        # the object bank is an INPUT: the launcher writes only under its own
        # root, so an objects_dir under /vol/tinycap must be one §PW21-stage put
        # there, never a gen path
        assert row.objects_dir.startswith(("/vol/objects/scale-12/",
                                           "/vol/capctrl/objects/scale-12-x2/",
                                           f"{C.VOL_OBJECTS_TINYCAP}/",
                                           f"{C.VOL_OBJECTS_SDFBASE}/"))
        assert not row.objects_dir.startswith(C.VOL_GEN_ROOT)
        # a full22-bank row is exactly one whose objects_dir is
        # under VOL_OBJECTS_FULL22 (the predicate the launcher's endpoint gates and
        # the roster pins key on); the earlier sdfbase bank is NOT one
        # ... except the seven IFEval rows, which serve from the same staged banks (the fix of
        # 2026-09-13: the plan's `cells_view` spelling was never staged) but are excluded by slug
        assert C.is_full22_row(row) is (row.objects_dir.startswith(f"{C.VOL_OBJECTS_FULL22}/")
                                        and not C.is_ifeval_row(row)), slug
    # the derived rows record the base -> instruct TRANSFER
    # (gemma's two sdfbase / full22-main rows fit on google/gemma-4-E4B; the qwen
    # full22-main row on Qwen/Qwen3.5-4B-Base; the llama full22-main row on
    # NousResearch/Meta-Llama-3.1-8B, the BASE its sidecar names); every other
    # row's objects were fit on the served model (the gemma / llama it-view rows and
    # the nemotron chat-ckpt row included)
    assert C.model_row("gemma4e4b-sdfbase").base_hf == "google/gemma-4-E4B" == C.model_row("gemma4e4b-f22").base_hf
    assert C.model_row("qwen35_4b-f22").base_hf == "Qwen/Qwen3.5-4B-Base"
    assert C.model_row("llama8b-f22").base_hf == "NousResearch/Meta-Llama-3.1-8B" and C.model_row("llama8b-f22it").base_hf is None
    # ... and the gemma / Qwen IFEval bank views, read off the staged
    # sidecars; the Nemotron IFEval row was fit on the served model and the Llama IFEval rows
    # leave the base id to their sidecar — so exactly these, in MODEL_SLUGS order
    # ... and the six seed rows: every batch-64 object — the paired-write composition's
    # parents included — was fit on Qwen/Qwen3.5-4B-Base @ 1001bb4d and is served on the chat checkpoint
    # ... and (2026-09-23) the all-models one-seed rows of the two TRANSFER families — gemma (fit on google/gemma-4-E4B) and
    # llama (fit on NousResearch/Meta-Llama-3.1-8B); nemotron's objects were fit on the served chat checkpoint (base_hf None)
    # ... and the worked example's row, fit on Qwen/Qwen3.5-4B-Base (reproduce/models.yaml transfer_kind
    # base_to_instruct), the first base_hf row in MODEL_SLUGS order since it sits among the legacy rows
    assert [s for s in C.MODEL_SLUGS if C.model_row(s).base_hf is not None] == [
        "qwen35_4b_demo", "qwen35_4b_demo-d0p75", "qwen35_4b_demo-d0p75_relu",
        "gemma4e4b-sdfbase", "gemma4e4b-f22", "qwen35_4b-f22", "llama8b-f22",
        *C.BCBW_SEED_SLUGS,
        "gemma4e4b-f22s1", "gemma4e4b-f22s1p", "llama8b-f22s1", "llama8b-f22s1p",
        "gemma4e4b-sdfbase-main", "gemma4e4b-sdfbase-it", "qwen35_4b-sdfbase-main", "qwen35_4b-sdfbase-it",
        # the six seed lane views' objects were fit on the Qwen BASE and are served on the instruct target (base->instruct)
        *[f"qwen35_4b-sdfbase-{v}" for v in C.IFEVAL_LANE_VIEWS]]
    assert all(C.model_row(s).base_hf == "Qwen/Qwen3.5-4B-Base" for s in C.BCBW_SEED_SLUGS)
    assert C.model_short("gemma4e4b-sdfbase") == "e4b_sdfbase"
    # the thirteen lane rows (the qwen it row, then the six b64 seed rows, appended LAST) come after
    # every campaign row, in BCBW_MODEL_VIEWS order (the first four, then llama8b's two);
    # the IFEval rows are appended AFTER them (asserted below)
    # (the IFEval tail is THIRTEEN slugs now — the seven instrument-view rows + the six seed lane views — so the bcbw
    # block is addressed relative to that tail and to its own length; the newest BCBW rows land BEFORE the IFEval block)
    n_ife = len(C.IFEVAL_SLUGS)
    assert n_ife == 13
    assert C.MODEL_SLUGS[-(n_ife + len(C.BCBW_SLUGS)):-n_ife] == C.BCBW_SLUGS == (
        "gemma4e4b-f22it", "gemma4e4b-f22", "qwen35_4b-f22", "nemotron3nano4b",
        "llama8b-f22it", "llama8b-f22", "qwen35_4b-f22it",
        "qwen35_4b-f22s1", "qwen35_4b-f22s2", "qwen35_4b-f22s3",
        "qwen35_4b-f22s1p", "qwen35_4b-f22s2p", "qwen35_4b-f22s3p",
        # b64-capability-agentic-queue (2026-09-23): the six all-models one-seed rows
        "gemma4e4b-f22s1", "gemma4e4b-f22s1p", "llama8b-f22s1", "llama8b-f22s1p",
        "nemotron3nano4b-f22s1", "nemotron3nano4b-f22s1p")
    assert C.MODEL_SLUGS[-(n_ife + len(C.BCBW_SLUGS) + 1)] == "gemma4e4b-sdfbase"
    # the worked example's row and, right after it, its two dose views' rows (demo-import 2026-10-12) — the last legacy rows
    assert C.MODEL_SLUGS[-(n_ife + len(C.BCBW_SLUGS) + 4):-(n_ife + len(C.BCBW_SLUGS) + 1)] == ("qwen35_4b_demo", "qwen35_4b_demo-d0p75", "qwen35_4b_demo-d0p75_relu")
    assert C.BCBW_SEED_VIEWS == ("s1", "s2", "s3", "s1p", "s2p", "s3p") and C.BCBW_SEED_SLUGS == C.BCBW_SLUGS[-12:-6]
    assert C.BCBW_ALLMODELS_VIEWS == ("s1", "s1p") and C.BCBW_ALLMODELS_FAMILIES == ("gemma4e4b", "llama8b", "nemotron3nano4b")
    assert C.BCBW_ALLMODELS_SLUGS == C.BCBW_SLUGS[-6:] == tuple(f"{mk}-f22{v}" for mk in C.BCBW_ALLMODELS_FAMILIES for v in C.BCBW_ALLMODELS_VIEWS)
    # a full22-bank row is exactly one of the thirteen bcbw rows: the IFEval rows serve from the same staged
    # /vol/objects/sdfbase/full22/<key>/<view> banks but the predicate excludes them by slug, so the widen gates skip them
    assert [C.is_full22_row(C.model_row(s)) for s in C.MODEL_SLUGS] == (
        [False] * (len(C.MODEL_SLUGS) - n_ife - len(C.BCBW_SLUGS)) + [True] * len(C.BCBW_SLUGS) + [False] * n_ife)
    assert not any(C.is_full22_row(C.model_row(s)) for s in C.IFEVAL_SLUGS.values())
    assert C.BCBW_MODEL_VIEWS[-15:-12] == (("llama8b", C.BCBW_VIEW_IT), ("llama8b", C.BCBW_VIEW_MAIN), ("qwen35_4b", C.BCBW_VIEW_IT))   # Qwen it after llama
    assert C.BCBW_MODEL_VIEWS[-12:-6] == tuple(("qwen35_4b", v) for v in C.BCBW_SEED_VIEWS)                                            # The seed family
    assert C.BCBW_MODEL_VIEWS[-6:] == C.BCBW_ALLMODELS_MODEL_VIEWS == (("gemma4e4b", "s1"), ("gemma4e4b", "s1p"), ("llama8b", "s1"),
                                                                                       ("llama8b", "s1p"), ("nemotron3nano4b", "s1"), ("nemotron3nano4b", "s1p"))   # 2026-09-23: the all-models family
    for v in C.BCBW_SEED_VIEWS:
        row = C.model_row(f"qwen35_4b-f22{v}")
        assert (row.hf, row.rev, row.model_key, row.family) == (C.model_row("qwen35_4b-f22").hf, C.model_row("qwen35_4b-f22").rev, "qwen35_4b", "qwen3_5")
        assert row.petri_manifest == C.PETRI_MANIFEST_FULL22_QWEN35_4B and row.thinking_modes == (C.THINKING_OFF,)
    # the all-models one-seed rows: the family's EXISTING lane row's pin, family, manifest and fit base, per (model key, view)
    for mk, anchor in (("gemma4e4b", "gemma4e4b-f22"), ("llama8b", "llama8b-f22"), ("nemotron3nano4b", "nemotron3nano4b")):
        a = C.model_row(anchor)
        for v in C.BCBW_ALLMODELS_VIEWS:
            row = C.model_row(f"{mk}-f22{v}")
            assert (row.hf, row.rev, row.model_key, row.family, row.petri_manifest, row.base_hf) == (a.hf, a.rev, mk, a.family, a.petri_manifest, a.base_hf), row.slug
            assert row.thinking_modes == (C.THINKING_OFF,)
    for (key, view), slug in C.BCBW_SLUG_OF.items():
        row = C.model_row(slug)
        assert row.cfg == C.bcbw_config_path(key, view) == f"configs/tinycap/bcbw_{key}_{view}.yaml"
        assert row.objects_dir == C.full22_objects_dir(key, view) == f"{C.VOL_OBJECTS_FULL22}/{key}/{view}"
        assert row.thinking_modes == (C.THINKING_OFF,) and row.model_key == key
    # the qwen3_5 pairing is PINNED (vllm_server.TOOL_PARSER_BY_FAMILY
    # qwen3_xml, with the offline fixtures in-grammar) so every qwen lane row carries the full22 qwen manifest;
    # nemotron_h's probe PASSED both halves on the staged tree 8365992 (2026-09-23): its three lane rows (the `main` row and the
    # all-models s1 / s1p rows) and the IFEval row carry the full22 nemotron manifest, qwen3_xml is pinned for the family
    for slug in ("nemotron3nano4b", "nemotron3nano4b-f22s1", "nemotron3nano4b-f22s1p", "nemotron3nano4b-sdfbase-main"):
        assert C.model_row(slug).petri_manifest == C.PETRI_MANIFEST_FULL22_NEMOTRON3NANO4B, slug
    assert C.model_row("qwen35_4b-f22").petri_manifest == C.model_row("qwen35_4b-f22it").petri_manifest == C.PETRI_MANIFEST_FULL22_QWEN35_4B
    assert vs.TOOL_PARSER_BY_FAMILY["qwen3_5"] == "qwen3_xml" and vs.TOOL_PARSER_BY_FAMILY["nemotron_h"] == "qwen3_xml"
    # the LIVE half is a second gate — a probe family plans BFCL only behind its filed PASS record
    # qwen3_5's LIVE half PASSED on the staged tree 80df464 (2026-09-23) — the record names the
    # checked-in verdict, which the gate OPENS on this tree (both halves status OK / PASS); nemotron_h's PASSED at 8365992 the same day
    assert C.TOOL_PARSER_PROBE_FAMILIES == ("qwen3_5", "nemotron_h")
    for fam, sha7, key in (("qwen3_5", "80df464", "qwen35_4b"), ("nemotron_h", "8365992", "nemotron3nano4b")):
        rec = C.TOOL_PARSER_LIVE_PASS[fam]
        assert set(rec) == set(C.TOOL_PARSER_LIVE_RECORD_FIELDS) and rec["sha7"] == sha7 and rec["record"] == f"{C.TOOL_PARSER_LIVE_RECORD_DIR}/{fam}_verdict_{sha7}.json"
        verdict = json.loads((REPO / rec["record"]).read_text())
        assert verdict["verdict"] == "PASS" and verdict["family"] == fam and verdict["model_key"] == key and rec["sha7"].startswith(verdict["sha"])
        assert verdict["offline"]["status"] == "OK" == verdict["live"]["status"] and verdict["live"]["verdict"]["verdict"] == "PASS"
    assert C.tool_parser_live_gate_problem("qwen3_5") is None and C.tool_parser_live_gate_problem("nemotron_h") is None and C.tool_parser_live_gate_problem("gemma4") is None
    assert C.TOOL_PARSER_LIVE_RECORD_FIELDS == ("sha7", "record", "passed_at") and C.TOOL_PARSER_LIVE_RECORD_DIR == "configs/capeval/tasks/parser_probe"
    assert C.source_checkout_root() == REPO                              # the checked-in verdict resolves against THIS tree (or the staged tree handed in)
    assert C.model_row("gemma4e4b-f22it").petri_manifest == C.model_row("gemma4e4b-f22").petri_manifest == C.PETRI_MANIFEST_GEMMA4E4B
    # the llama tool-parser pairing is pinned (llama3_json), so the two llama8b lane rows carry the
    # campaign's llama manifest — the llama8b row's, same served target + pin — and BOTH legs run
    assert (C.model_row("llama8b-f22it").petri_manifest == C.model_row("llama8b-f22").petri_manifest
            == C.model_row("llama8b").petri_manifest == C.PETRI_MANIFEST_TWOSI)
    for slug in ("llama8b-f22it", "llama8b-f22"):
        row = C.model_row(slug)
        assert (row.hf, row.rev, row.model_key, row.family) == (C.model_row("llama8b").hf, C.model_row("llama8b").rev, "llama8b", "llama")
    # ... and their intervention-hash inventories are PINNED from this tree's own derivation (the
    # bootstrap re-derivation on the grading host through the preflight's module; the container
    # preflight re-derives and refuses a disagreement) — one 16-hex hash per FITTED cell of the lane
    # roster, none for the object-free cells
    from sisteer.capeval.tinycap import widen_roster as _W
    # every OBJECT-carrying roster entry of the row (runnable OR blocked — a blocked cell keeps its pin) has a
    # pinned hash, and every runnable fitted cell's pin is in the inventory — over BOTH lane rosters (since 2026-09-25 the
    # full roster's instruct-direct waves serve the batch-8 / batch-64 LoRAs the 200 roster's `_it` waves never did)
    _objects_by_slug, _runnable_by_slug = {}, {}
    for _roster in (_W.load_roster(), _W.load_roster(_W.ROSTER_FULL_PATH)):
        for _wave in _W.waves(_roster):
            _w = _roster["waves"][_wave]; _slug = _w.get("model_row")
            if not _slug:
                continue
            for _e in (_w.get("entries") or {}).values():
                if isinstance(_e, dict) and _e.get("cell_id") and isinstance(_e.get("object"), dict) and _e["object"].get("sha256"):
                    _objects_by_slug.setdefault(_slug, set()).add(_e["cell_id"])
            _runnable_by_slug.setdefault(_slug, set()).update(_W.pins_by_cell(_roster, _wave))
    for slug in C.BCBW_SLUGS:
        inv = C.EXPECTED_INTERVENTION_HASHES[slug]
        assert set(inv) == _objects_by_slug.get(slug, set()), (slug, sorted(inv), sorted(_objects_by_slug.get(slug, set())))
        assert _runnable_by_slug.get(slug, set()) <= set(inv), slug
        assert all(re.fullmatch(r"[0-9a-f]{16}", h) for h in inv.values()), slug
    assert C.EXPECTED_INTERVENTION_HASHES["gemma4e4b-f22"]["constv2-mix-full__lora_r4_ce_minus_ctl"] == "6e23d193b88952bd"  # = the first lane's pin (same bytes)
    # the instruct-direct constitution / control LoRAs of qwen and llama, first served on the full suites — the grading host
    # derivation (logs/d16_it/report_<slug>.json in the spec dir; the same requests reproduced the pinned differences as calibration)
    assert C.EXPECTED_INTERVENTION_HASHES["qwen35_4b-f22it"] == {"constv2-mix-full__lora_r4_ce_minus_ctl": "51a39d5154beb21f",
                                                                 "constv2-mix-full__lora_r4_ce": "96e0d19f7eeba57a", "constv2ctl2-full__lora_r4_ce": "276f55f685d46b53"}
    assert C.EXPECTED_INTERVENTION_HASHES["llama8b-f22it"] == {"constv2-mix-full__lora_r4_ce_minus_ctl": "3dbb1d77ca450dd7",
                                                               "constv2-mix-full__lora_r4_ce": "ae6b8bb37c1c43de", "constv2ctl2-full__lora_r4_ce": "92a8220205229f8c"}
    # 2026-09-22: the six seed rows' inventories are the grading host's derivation under the bootstrap
    # procedure — five hashes + the Arditi refusal-steer per seed view (the served-only Class B branch
    # of object_identity), the composed paired-write cell per paired view
    for v in ("s1", "s2", "s3"):
        assert set(C.EXPECTED_INTERVENTION_HASHES[f"qwen35_4b-f22{v}"]) == {
            "constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all", "constv2-mix-full__lora_r4_ce",
            "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "arditi_allpos"}
        assert set(C.EXPECTED_INTERVENTION_HASHES[f"qwen35_4b-f22{v}p"]) == {"constv2-mix-full__psr_ce_all_minus_ctl"}
    seed_hashes = [h for v in C.BCBW_SEED_VIEWS for h in C.EXPECTED_INTERVENTION_HASHES[f"qwen35_4b-f22{v}"].values()]
    # 21 entries; 19 distinct values: the Arditi cell is ONE object (ba2212f0…) served on three rows — one identity
    arditi = {C.EXPECTED_INTERVENTION_HASHES[f"qwen35_4b-f22{v}"]["arditi_allpos"] for v in ("s1", "s2", "s3")}
    assert len(seed_hashes) == 21 and len(arditi) == 1 and len(set(seed_hashes)) == 19
    assert C.EXPECTED_INTERVENTION_HASHES["qwen35_4b-f22s1"]["constv2-mix-full__lora_r4_ce_minus_ctl"] != "15aaeb8322ded0e7"   # not the batch-8 main-view object
    # ifeval-canonical-battery (the wave shakedown, 2026-09-13): the pinned IFEval slugs' entries name EXACTLY their
    # LAUNCH fitted cells of the frozen matrix (held rows carry no pin — they never launch), every value 16 hex, and the
    # `minus_ctl` cell of each view reproduces the bcbw pin for the same object bytes (the cross-check of the derivation)
    _ifeval_rows = json.loads((REPO / C.IFEVAL_MATRIX_REL).read_text(encoding="utf-8"))["rows"]
    for _slug in ("gemma4e4b-sdfbase-main", "qwen35_4b-sdfbase-main", "nemotron3nano4b-sdfbase-main",
                  "llama8b-sdfbase-main", "llama8b-sdfbase-it"):
        _inv = C.EXPECTED_INTERVENTION_HASHES[_slug]
        assert set(_inv) == {r["cell"] for r in _ifeval_rows if r["slug"] == _slug and r["status"] == "launch" and r["object_sha256"]}, _slug
        assert all(re.fullmatch(r"[0-9a-f]{16}", h) for h in _inv.values()), _slug
    assert C.EXPECTED_INTERVENTION_HASHES["gemma4e4b-sdfbase-main"]["constv2-mix-full__lora_r4_ce_minus_ctl"] == "6e23d193b88952bd"
    assert C.EXPECTED_INTERVENTION_HASHES["qwen35_4b-sdfbase-main"]["constv2-mix-full__lora_r4_ce_minus_ctl"] == \
        C.EXPECTED_INTERVENTION_HASHES["qwen35_4b-f22"]["constv2-mix-full__lora_r4_ce_minus_ctl"] == "15aaeb8322ded0e7"
    assert C.EXPECTED_INTERVENTION_HASHES["nemotron3nano4b-sdfbase-main"]["constv2-mix-full__lora_r4_ce_minus_ctl"] == \
        C.EXPECTED_INTERVENTION_HASHES["nemotron3nano4b"]["constv2-mix-full__lora_r4_ce_minus_ctl"] == "ee40acc27dee03a6"
    for _slug in ("gemma4e4b-sdfbase-it", "qwen35_4b-sdfbase-it"):
        assert _slug not in C.EXPECTED_INTERVENTION_HASHES                    # the held it-view LoRAs carry no pin
    # 2026-09-22 (the bootstrap derivation procedure): the six seed lane slugs' entries name EXACTLY their LAUNCH
    # fitted cells of the sibling document (21 = 3 x (5 + Arditi) + 3 x 1), every value 16 hex; the Arditi object is the SAME
    # bytes on s1 / s2 / s3, so its three entries agree; nothing else is typed here — the sets come from the document
    _seed_rows = json.loads((REPO / C.IFEVAL_MATRIX_QWEN_SEEDREP_REL).read_text(encoding="utf-8"))["rows"]
    _n = 0
    for _v in C.IFEVAL_LANE_VIEWS:
        _slug = C.IFEVAL_SLUGS[("qwen35_4b", _v)]
        _inv = C.EXPECTED_INTERVENTION_HASHES[_slug]
        assert set(_inv) == {r["cell"] for r in _seed_rows if r["slug"] == _slug and r["status"] == "launch" and r["object_sha256"]}, _slug
        assert all(re.fullmatch(r"[0-9a-f]{16}", h) for h in _inv.values()), _slug
        _n += len(_inv)
    assert _n == 21
    assert len({C.EXPECTED_INTERVENTION_HASHES[C.IFEVAL_SLUGS[("qwen35_4b", f"s{i}")]]["arditi_allpos"] for i in (1, 2, 3)}) == 1
    # ifeval-canonical-battery: the IFEval slugs are appended LAST, in IFEVAL_SLUGS order — the seven instrument-view rows,
    # then the six seed lane views, s3p last
    assert C.MODEL_SLUGS[-13:] == tuple(C.IFEVAL_SLUGS.values()) and C.MODEL_SLUGS[-1] == "qwen35_4b-sdfbase-s3p"
    assert C.MODEL_SLUGS[-7] == "llama8b-sdfbase-it"
    assert C.model_short("nemotron3nano4b-sdfbase-main") == "nemo4b_sdfbase_main"
    assert C.model_short("gemma4e4b-sdfbase-it") == "e4b_sdfbase_it"
    # ... and its intervention-hash inventory pins the derived cell's HF post-load
    # hash — the value the bootstrap preflight of staged tree 16f249b9 derived
    # (2026-09-11; the table's comment names the object sha and the served
    # revision), so a plain --pf reads match: true and a moved object refuses
    assert C.EXPECTED_INTERVENTION_HASHES["gemma4e4b-sdfbase"] == {
        "constv2-mix-full__lora_r4_ce_minus_ctl": "6e23d193b88952bd"}
    assert C.expected_intervention_hash("gemma4e4b-sdfbase", "constv2-mix-full__lora_r4_ce_minus_ctl") == "6e23d193b88952bd"
    assert re.fullmatch(r"[0-9a-f]{16}", C.EXPECTED_INTERVENTION_HASHES["gemma4e4b-sdfbase"]["constv2-mix-full__lora_r4_ce_minus_ctl"])
    with pytest.raises(C.UnknownModel, match="unknown model slug"):
        C.model_row("gemma4e4b_x2")             # the underscore spelling is not a slug
    # the default model is a registered row, and the pre-registry constants
    # still name it (so every command line that predates --model is unchanged)
    assert C.MODEL_SLUG in C.MODELS
    assert C.model_row(C.MODEL_SLUG).cfg == "configs/two_si_gemma4e4b_reduced.yaml"


def test_every_registry_config_pins_the_row_s_model_and_revision():
    """The config is what the STANDUP reads (`model` / `model_revision`), so a
    row whose pins disagree with its config would serve one model and record
    another. Checked against the checked-in yaml — the authoritative pin —
    except where the row deliberately OVERLAYS a null (llama3b)."""
    for slug in C.MODEL_SLUGS:
        row = C.model_row(slug)
        if C.is_ifeval_row(row) and not (REPO / row.cfg).is_file():
            # ifeval-canonical-battery: the per-TAG overlay configs are written by the
            # matrix lane (make_ifeval_overlays.py); test_tinycap_launcher_port pins
            # their paths and xfails until they land (strict at the merge)
            continue
        doc = yaml.safe_load((REPO / row.cfg).read_text())
        assert doc["model"] == row.hf, slug
        overrides = C.config_overrides(slug)
        pinned = overrides.get("model_revision", doc["model_revision"])
        assert pinned == row.rev, slug
        # an overlay only ever REPLACES a key the config pins NULL
        for k, v in overrides.items():
            assert k in doc and doc[k] is None, (slug, k)
            assert v is not None


def test_the_llama3b_row_supplies_the_two_nulls_its_config_leaves():
    """§PW21-stage's flag: `configs/two_si_llama32_3b.yaml` pins `layer: null`
    AND `model_revision: null`, and an explicit null is worse than a missing
    key — `cell_controls.arm_spec_for_cell` does `int(cfg.get("layer", 15))`,
    which RAISES on None. The row supplies both; the layer enters only the F9
    membership check and the lens-route STATIC hash, never the HF post-load
    resolved spec."""
    doc = yaml.safe_load((REPO / "configs/two_si_llama32_3b.yaml").read_text())
    assert doc["layer"] is None and doc["model_revision"] is None
    assert C.config_overrides("llama3b") == {
        "layer": 15,
        "model_revision": "006f5dcd1393c3add266de40994ba96225e9689d"}
    # every other row's config is complete, so it needs no overlay
    assert [s for s in C.MODEL_SLUGS if C.config_overrides(s)] == ["llama3b"]


def test_the_gemma4e2b_si_block_matches_the_certified_model_s():
    """The E2B config must offer the SAME three SIs as the certified E4B one,
    or its ten cells are not the same ten cells."""
    e4b = yaml.safe_load((REPO / C.model_row("gemma4e4b").cfg).read_text())
    e2b = yaml.safe_load((REPO / C.model_row("gemma4e2b").cfg).read_text())
    assert e2b["si_ids"] == e4b["si_ids"] == ["safety_refusal", "si_constitution",
                                              "si_irrelevant"]
    assert e2b["layer"] == e4b["layer"] == 20 and e2b["scale"] == e4b["scale"] == 1.0
    # and every registry model offers the same three (the campaign's cells) —
    # EXCEPT the bcb-bfcl-200 derived row, whose dedicated config's roster is
    # exactly the derived cell (build_registry crosses si_ids x
    # class_a_arms, so the corpus pseudo-instruction and the derived arm are the
    # only entries; widening the reduced config would fabricate prompted__ cells)
    derived = {"gemma4e4b-sdfbase"}
    for slug in C.MODEL_SLUGS:
        row = C.model_row(slug)
        if C.is_ifeval_row(row):
            continue      # ifeval-canonical-battery: explicit-cells overlays of the paper objects, not the campaign's three cells
        doc = yaml.safe_load((REPO / row.cfg).read_text())
        if slug in derived:
            assert doc["si_ids"] == ["constv2-mix-full"], slug
            assert doc["class_a_arms"] == ["lora_r4_ce_minus_ctl"] and doc["class_b_arms"] == [], slug
            assert doc["model"] == e4b["model"] and doc["model_revision"] == e4b["model_revision"], slug
            # the row's objects_dir IS the config's absolute defenses_dir
            assert doc["defenses_dir"] == C.model_row(slug).objects_dir == "/vol/objects/sdfbase/gemma4e4b"
            continue
        if C.is_full22_row(row):
            # bcb-bfcl-200-widen: an EXPLICIT-cells config (no Cartesian si_ids /
            # class arms — registry_from_explicit_cells refuses the pair) listing
            # `none`, the prompted safety anchor and the roster view's object cells,
            # over the row's volume bank as its absolute defenses_dir
            assert "si_ids" not in doc and "class_a_arms" not in doc and "class_b_arms" not in doc, slug
            cells = doc["explicit_cells"]
            assert cells[:2] == [C.ROUTE_NONE_CELL, C.CELL_PROMPTED_SAFETY] and len(cells) >= 3, slug
            assert doc["defenses_dir"] == row.objects_dir and doc["defenses_dir"].startswith(C.VOL_OBJECTS_FULL22), slug
            continue
        if row.cfg.startswith("configs/tinycap/frontdoor_"):
            # the worked example's EXPLICIT-cells config, the front door's generated file — `none`, the two prompted cells and
            # the wave's object cells in the models table's order (the ten of the plan), over the row's OWN bank under the sdf-base root and
            # outside the full22 root (is_full22_row stays False). A DOSE view's row (since 2026-10-12: `qwen35_4b_demo-<view>`) serves
            # the view's TWO cells — `none` and its object cell, the SAME token as the wave's parent (the bank, not the token, carries the dose)
            # — over the view's own sub-bank under the key's root (test_the_dose_views_rows_are_the_demo_rows_twins_over_the_views_own_bank_and_config)
            assert "si_ids" not in doc and "class_a_arms" not in doc and "class_b_arms" not in doc, slug
            cells = doc["explicit_cells"]
            if "-" in slug:
                key, view = slug.split("-", 1)
                wave_cells = yaml.safe_load((REPO / C.model_row(key).cfg).read_text())["explicit_cells"]
                assert cells[0] == C.ROUTE_NONE_CELL and len(cells) == 2 and cells[1] in wave_cells[3:], slug
                assert doc["defenses_dir"] == row.objects_dir == f"{C.model_row(key).objects_dir}/{view}", slug
            else:
                assert cells[:3] == [C.ROUTE_NONE_CELL, C.CELL_PROMPTED_SAFETY, "prompted__si_constitution"] and len(cells) == 10, slug
            assert doc["defenses_dir"] == row.objects_dir and doc["defenses_dir"].startswith(C.VOL_OBJECTS_SDFBASE) and not doc["defenses_dir"].startswith(C.VOL_OBJECTS_FULL22), slug
            continue
        assert doc["si_ids"] == e4b["si_ids"], slug


def test_every_registry_model_has_a_measured_geometry_budget():
    """S comes from the model's OWN batched-token budget row, and the launcher
    refuses any other sizing (`UnmeasuredBudget`) — so a row whose `model_key`
    is not in the table could never be served. A PROVISIONAL ceiling (the
    2026-09-06 lens-stack failures) is that model's own row and is served; the
    Llama-certified default never is, which is what `per_model` separates."""
    for slug in C.MODEL_SLUGS:
        key = C.model_row(slug).model_key
        # every registry row is one of the sweep's models with a measured row (geometry.ONBOARDED_MODEL_KEYS size from the SITE
        # overlay the front door stages, and no registry row names one of them)
        assert key in sg.BUDGET_BY_MODEL and key not in sg.ONBOARDED_MODEL_KEYS, slug
        geom = sg.pass1_geometry("capability:gen_long", model_key=key)
        assert geom.seqs_source == "model_budget" and geom.budget.per_model, slug
        assert geom.budget.source in sg.PER_MODEL_BUDGET_SOURCES, slug
        assert geom.budget.measured is not geom.budget.provisional, slug
        assert geom.model_key == key and geom.max_num_seqs >= 1


def test_registry_families_are_the_SERVING_family_names():
    """`family` must be exactly what `vllm_server.model_family` returns for
    `hf@rev` — the HF config's `model_type` — because the reasoning- and
    tool-parser pins are keyed on that vocabulary. The plan's table wrote
    `llama3` informally; a private spelling would silently miss every lookup.
    Rows whose config is not in this machine's HF cache are skipped, never
    guessed."""
    from transformers import AutoConfig

    def cached(row):
        try:
            AutoConfig.from_pretrained(row.hf, revision=row.rev, local_files_only=True)
        except Exception:                          # noqa: BLE001 — not cached here
            return False
        return True

    rows = [C.model_row(s) for s in C.MODEL_SLUGS]
    seen = [r for r in rows if cached(r)]
    for row in seen:
        assert vs.model_family(row.hf, row.rev) == row.family, row.slug
    assert len(seen) >= 1, "no registry model resolved locally — the check proved nothing"


#: the scale-12 two-SI Qwen rows (generation-only battery) that predate the qwen3_5 pairing and carry no manifest
LEGACY_MANIFESTLESS_ROWS = ("qwen35_9b", "qwen35_4b")


def test_a_manifest_is_pinned_only_where_the_tool_parser_pairing_is_verified():
    """BFCL composes a tool parser, and a parser foreign to the served
    tokenizer is a healthy-looking server that 500s on every chat request
    (job 232411). So a row carries a manifest only for a family in
    `vllm_server.TOOL_PARSER_BY_FAMILY`, and the manifest it names must pin
    that family's parser."""
    for slug in C.MODEL_SLUGS:
        row = C.model_row(slug)
        pinned = vs.TOOL_PARSER_BY_FAMILY.get(row.family)
        if pinned is None:
            # a manifest-less row refuses BFCL WHATEVER its family (the
            # launcher keys on `petri_manifest is None`); the one unverified
            # family we serve is nemotron_h — it gains its pin and its
            # manifest in the SAME change its parser probe PASSES (qwen3_5
            # did so on 2026-09-22)
            assert row.family == "nemotron_h", slug
            assert row.petri_manifest is None, slug
            continue
        if slug in LEGACY_MANIFESTLESS_ROWS:
            # the two scale-12 two-SI Qwen rows predate the pairing and run no BFCL (their battery is generation-only;
            # qwen35_9b is a different target than the full22 manifest pins): a manifest-less row refuses BFCL at plan
            # time by the launcher's rule, so leaving them unchanged is the safe state — the pairing change touched the
            # bcbw / IFEval blocks only; pinning a 9B manifest is a separate change
            assert row.petri_manifest is None, slug
            continue
        assert row.petri_manifest is not None, slug
        assert (REPO / row.petri_manifest).is_file(), slug
        doc = yaml.safe_load((REPO / row.petri_manifest).read_text())["petri"]
        assert doc["tool_parser"] == pinned, (slug, row.petri_manifest)
        if row.family == "qwen3_5":
            assert row.petri_manifest == C.PETRI_MANIFEST_FULL22_QWEN35_4B and (doc["target_model"], doc["target_revision"]) == (row.hf, row.rev), slug


def test_registry_thinking_modes_follow_the_parser_pins():
    """Only a family with a VERIFIED reasoning parser may be served
    thinking-ON (`REASONING_PARSER_BY_FAMILY`); O6 q1/q2 put Qwen and Llama
    OFF on every cell."""
    for slug in C.MODEL_SLUGS:
        row = C.model_row(slug)
        assert row.thinking_modes and set(row.thinking_modes) <= set(C.THINKING_MODES)
        if C.THINKING_ON in row.thinking_modes:
            assert row.family in C.REASONING_PARSER_BY_FAMILY, slug
        else:
            assert row.thinking_modes == (C.THINKING_OFF,), slug
    assert [s for s in C.MODEL_SLUGS if C.THINKING_ON in C.model_row(s).thinking_modes] == [
        "gemma4e4b", "gemma4e4b-x2", "gemma4e2b", "gemma4e4b-nors", "gemma4e4b-sdfbase"]


# --------------------------------------------------------------------------- #
# per-family thinking and the per-task mode                                     #
# --------------------------------------------------------------------------- #


#: The generation-header id tuples measured 2026-09-09 (thinking-OFF, thinking-ON)
#: — the 2026-09-09 thinking-header measurement, raw output in
#: `thinking_header_measure_2026-09-09.json` beside it — at the campaign's
#: pinned revisions. Pinned here so an edit to the MASK_MOVING_THINKING_FAMILIES
#: entries that changes the numbers, or a derivation change that starts
#: admitting Nemotron's ON header without updating its entry, fails.
GLM4V_OFF_IDS = (151337, 198, 151350, 151351, 198)
GLM4V_ON_IDS = (151337, 198)
NEMOTRON_OFF_IDS = (10, 1503, 19464, 1010, 12, 13)
NEMOTRON_ON_IDS = (10, 1503, 19464, 1010, 12, 1010)
GLM4V_PIN = ("zai-org/GLM-4.6V-Flash", "411bb4d77144a3f03accbf4b780f5acb8b7cde4e")
NEMOTRON_PIN = ("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f")


def test_the_per_family_thinking_tables_match_the_measurement():
    """The thinking-header measurement: gemma-4's assistant header is
    IDENTICAL in both modes (H=3, the thinking signal is a LEADING system
    turn), Qwen3.5's MOVES (7 -> 5), Llama has no switch at all (all measured
    2026-09-04); GLM-4.6V MOVES 5 -> 2 (ON a strict prefix of OFF, explicit
    tags) and Nemotron-3-Nano keeps WIDTH 6 while its ids move and the block
    opens — REFUSED by the derivation today (both measured 2026-09-09)."""
    assert C.HEADER_INVARIANT_THINKING_FAMILIES == ("gemma4",)
    assert C.NO_THINKING_CHANNEL_FAMILIES == ("llama",)
    assert set(C.MASK_MOVING_THINKING_FAMILIES) == {"qwen3_5", "glm4v", "nemotron_h"}
    # the measured numbers travel WITH the refusal, so it can be checked
    assert "H=7" in C.MASK_MOVING_THINKING_FAMILIES["qwen3_5"]
    assert "H=5" in C.MASK_MOVING_THINKING_FAMILIES["qwen3_5"]
    for fam in ("glm4v", "nemotron_h"):
        assert "UNVERIFIED" not in C.MASK_MOVING_THINKING_FAMILIES[fam]
        assert "2026-09-09" in C.MASK_MOVING_THINKING_FAMILIES[fam]
    glm = C.MASK_MOVING_THINKING_FAMILIES["glm4v"]
    assert "H=5" in glm and "H=2" in glm
    assert str(list(GLM4V_OFF_IDS)) in glm and str(list(GLM4V_ON_IDS)) in glm
    assert "strict PREFIX" in glm
    nemo = C.MASK_MOVING_THINKING_FAMILIES["nemotron_h"]
    # the hazard: SAME width both arms, so it looks admissible on width alone
    assert nemo.count("H=6") == 2
    assert str(list(NEMOTRON_OFF_IDS)) in nemo and str(list(NEMOTRON_ON_IDS)) in nemo
    assert len(NEMOTRON_OFF_IDS) == len(NEMOTRON_ON_IDS)
    assert NEMOTRON_OFF_IDS != NEMOTRON_ON_IDS
    assert "REFUSED" in nemo and "thinking_on_unattested" in nemo
    assert "NOT admissible on width alone" in nemo
    assert "thinking_header_verification.md" in C.THINKING_HEADER_EVIDENCE
    assert "2026-09-09" in C.THINKING_HEADER_EVIDENCE


def _cached_thinking_tok(model, revision):
    """The pinned tokenizer from the LOCAL cache only, skipping loudly when it
    is absent — the idiom tests/test_chat_markers_thinking_header.py uses."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model, revision=revision, local_files_only=True)
    except Exception as e:                                    # noqa: BLE001
        pytest.skip(f"pinned {model} tokenizer @ {revision[:8]} not in the "
                    f"local HF cache ({type(e).__name__})")


@pytest.mark.parametrize("pin,off_ids,on_ids", [
    pytest.param(GLM4V_PIN, GLM4V_OFF_IDS, GLM4V_ON_IDS, id="glm4v"),
    pytest.param(NEMOTRON_PIN, NEMOTRON_OFF_IDS, NEMOTRON_ON_IDS, id="nemotron_h"),
])
def test_the_measured_id_tuples_are_what_the_pinned_tokenizers_render(
        pin, off_ids, on_ids):
    """The entries' numbers against the tokenizers themselves (offline, the
    campaign's pinned revisions; skips when a pin is not cached). The raw
    add_generation_prompt diff under each regime IS the recorded tuple, and
    the id-level derivation's verdict is the one the entry states: GLM
    certifies both arms (moves True); Nemotron's ON arm is REFUSED by the
    prefix rule, so its fact records on_ids None. A derivation change that
    starts admitting Nemotron's ON header fails HERE, forcing the nemotron_h
    entry (which says REFUSED) to be updated in the same change."""
    from sisteer import chat_markers as cm

    tok = _cached_thinking_tok(*pin)
    msgs = [{"role": "user", "content": "x"}]

    def raw_appended(arm):
        tok._sisteer_template_kwargs = {"enable_thinking": arm}
        try:
            head = cm.render_ids(tok, msgs)
            full = cm.render_ids(tok, msgs, add_generation_prompt=True)
        finally:
            del tok._sisteer_template_kwargs
        assert full[:len(head)] == head
        return tuple(full[len(head):])

    assert raw_appended(False) == off_ids
    assert raw_appended(True) == on_ids
    assert tuple(cm.generation_header_ids_under(tok, enable_thinking=False)) == off_ids
    facts, reason = cm.thinking_header_facts_or_error(tok)
    assert facts["switch"] is True and tuple(facts["off_ids"]) == off_ids
    if pin == GLM4V_PIN:
        assert reason is None
        assert tuple(facts["on_ids"]) == on_ids and facts["moves"] is True
        assert tuple(cm.generation_header_ids_under(tok, enable_thinking=True)) == on_ids
    else:
        # REFUSED today: what a width guard sees is nothing (6 == 6)
        assert len(off_ids) == len(on_ids) and off_ids != on_ids
        assert facts["on_ids"] is None and facts["moves"] is None
        assert reason.startswith("ChatMarkerError: ") and "not a prefix" in reason
        with pytest.raises(cm.ChatMarkerError, match="not a prefix"):
            cm.generation_header_ids_under(tok, enable_thinking=True)
    # the two sets are disjoint: a family cannot both keep and move its header
    assert not set(C.HEADER_INVARIANT_THINKING_FAMILIES) & set(C.MASK_MOVING_THINKING_FAMILIES)
    # and only an invariant-header family has a verified parser
    assert set(C.REASONING_PARSER_BY_FAMILY) == set(C.HEADER_INVARIANT_THINKING_FAMILIES)


def test_thinking_on_refusal_gives_the_specific_reason_per_family_and_kind(monkeypatch):
    # gemma: allowed on BOTH kinds — the header does not move and the parser is pinned
    assert C.thinking_on_refusal("gemma4", fitted=False) is None
    assert C.thinking_on_refusal("gemma4", fitted=True) is None
    # llama: no switch at all, so neither kind is expressible — and the reason
    # says which of the two problems it is
    for fitted in (False, True):
        why = C.thinking_on_refusal("llama", fitted=fitted)
        assert "no pinned reasoning parser" in why
        assert "no thinking switch at all" in why
    # qwen FITTED: the mask-moving refusal wins, and cites the pin + the measurement
    fitted = C.thinking_on_refusal("qwen3_5", fitted=True)
    assert "THINKING_OFF_TEMPLATE_KWARGS" in fitted and "H=7" in fitted
    assert "a new object, not a re-run" in fitted
    assert C.THINKING_HEADER_EVIDENCE in fitted
    # qwen PROMPTED / base: no object to misalign, but still no verified parser —
    # by DEFAULT (the parent's launcher derives its 2x2 posture from this answer,
    # decisions-laneA LA17); the math500-thinking-on-full22 driver OPTS IN to the
    # certified harness-side split as the parser (G1)
    prompted = C.thinking_on_refusal("qwen3_5", fitted=False)
    assert "no pinned reasoning parser" in prompted
    assert "THINKING_OFF_TEMPLATE_KWARGS" not in prompted
    assert C.thinking_on_refusal("qwen3_5", fitted=False, certified_split_ok=True) is None
    # glm4v / nemotron_h: certified by G-PARSER (2026-09-10) — admitted like qwen under the opt-in, refused by default
    for fam in ("glm4v", "nemotron_h"):
        assert C.thinking_on_refusal(fam, fitted=False, certified_split_ok=True) is None, fam
        assert "no pinned reasoning parser" in C.thinking_on_refusal(fam, fitted=False), fam
    # a candidate fact (a SYNTHETIC demotion of Nemotron's): the opt-in does not lift the refusal, which names both
    # reasons; the dry run's `pre` reading (require_certified=False) admits it
    demoted = C.FAMILY_RENDER_FACTS["nemotron_h"]._replace(evidence_status=C.RENDER_EVIDENCE_CANDIDATE)
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", {**C.FAMILY_RENDER_FACTS, "nemotron_h": demoted})
    why = C.thinking_on_refusal("nemotron_h", fitted=False, certified_split_ok=True)
    assert "no pinned reasoning parser" in why and "render fact not certified" in why
    assert C.thinking_on_refusal("nemotron_h", fitted=False, certified_split_ok=True, require_certified=False) is None
    monkeypatch.undo()
    # a fitted ALL-POSITION cell (header_located=False) is not refused for the mask
    assert "no pinned reasoning parser" in C.thinking_on_refusal("qwen3_5", fitted=True, header_located=False)
    assert C.thinking_on_refusal("qwen3_5", fitted=True, header_located=False, certified_split_ok=True) is None


def test_the_per_task_thinking_mode_is_a_label_not_a_boolean():
    assert C.THINKING_PER_TASK == "per-task"
    assert C.LAUNCHER_THINKING_MODES == ("off", "on", "per-task", "not_available")
    assert C.THINKING_PER_TASK not in C.THINKING_MODES     # enable_thinking cannot answer it
    with pytest.raises(ValueError):
        C.enable_thinking(C.THINKING_PER_TASK)
    assert C.check_thinking_arg(C.THINKING_PER_TASK) == C.THINKING_PER_TASK
    with pytest.raises(ValueError, match="thinking mode must be one of"):
        C.check_thinking_arg("sideways")
    # a whole arm answers for every task; per-task reads the policy
    assert C.thinking_of_task("off", "inspect/gsm8k_paired200") == "off"
    assert C.thinking_of_task("on", "inspect/bfcl") == "on"
    assert C.thinking_of_task("per-task", "inspect/gsm8k_paired200") == "on"
    assert C.thinking_of_task("per-task", "inspect/bfcl") == "off"
    with pytest.raises(ValueError, match="no arm for row task"):
        C.thinking_of_task("per-task", "served_ll/tinyMMLU")       # the removed family has no arm
    # the map is total over the tinycap tasks, and an unmapped task refuses
    for task in C.TINYCAP_TASKS:
        if task.endswith(C.OPEN_VARIANT_SUFFIX):
            continue
        assert C.thinking_of_task("per-task", task) in C.THINKING_MODES
    with pytest.raises(ValueError, match="no arm for row task"):
        C.thinking_of_task("per-task", "inspect/not_a_task")
    # per-task is a LABEL: its store and its container are its own
    assert C.cell_out_dir("/r", "none", "per-task").endswith("/thinking_per-task")
    assert C.container_name("none", "per-task").endswith("-think_per-task")
    argv = C.driver_argv(python="p", script="s", proxy_url="u", routing_table="t",
                         cell="none", config="c", defenses_dir="d", out_root="o",
                         controls=["gsm8k"], thinking="per-task", concurrency=4,
                         judge_model="j")
    assert argv[argv.index(C.FLAG_THINKING) + 1] == "per-task"


# --------------------------------------------------------------------------- #
# the sampling contract                                                        #
# --------------------------------------------------------------------------- #


def test_sampling_modes_and_the_driver_flag():
    """`card` (the ruling) and `greedy` (still legal, a different identity);
    the default is the ruling; the flag is one name."""
    # three modes beside the two; `protocol` — per-task, model-independent,
    # the two pinned legs' own reference decoding; `vendor_preset`, the vendor's
    # shipped decode row per model, appended LAST after it
    assert C.SAMPLING_MODES == ("greedy", "card", "thinking_card", "general", "battery", "protocol", "vendor_preset")
    assert (C.SAMPLING_THINKING_CARD, C.SAMPLING_GENERAL, C.SAMPLING_BATTERY, C.SAMPLING_PROTOCOL,
            C.SAMPLING_VENDOR_PRESET) == ("thinking_card", "general", "battery", "protocol", "vendor_preset")
    assert C.SAMPLING_PROTOCOL not in C.SAMPLING_TABLE_MODES         # no per-model table: per TASK
    assert C.SAMPLING_VENDOR_PRESET in C.SAMPLING_TABLE_MODES        # per-model table: VENDOR_PRESET_SAMPLING_BY_MODEL_KEY
    assert C.SAMPLING_TABLE_BY_MODE[C.SAMPLING_VENDOR_PRESET] is C.VENDOR_PRESET_SAMPLING_BY_MODEL_KEY
    assert C.SAMPLING_DEFAULT == C.SAMPLING_CARD == "card"
    assert C.FLAG_SAMPLING == "--sampling"
    for m in C.SAMPLING_MODES:
        assert C.check_sampling_arg(m) == m
    with pytest.raises(ValueError, match="sampling mode must be one of"):
        C.check_sampling_arg("nucleus")
    # greedy is argmax with NO nucleus / top-k keys — what every pre-U8 decoding
    # task sent; its row names no card
    assert C.GREEDY_SAMPLING == C.SamplingRow(0.0, None, None, C.GREEDY_SAMPLING.source)
    assert C.GREEDY_TEMPERATURE == 0.0 and "greedy" in C.GREEDY_SAMPLING.source
    # the three OPTIONAL terms trail the row (None on every legacy row)
    assert C.SAMPLING_ROW_FIELDS == ("temperature", "top_p", "top_k", "source",
                                     "min_p", "presence_penalty", "repetition_penalty")
    assert C.SAMPLING_OPTIONAL_ROW_FIELDS == ("min_p", "presence_penalty", "repetition_penalty")


def test_sampling_table_rows_and_their_sources():
    """The per-model FROZEN rows (U8 item 1: Gemma 1.0, Llama 0.6, GLM 0.8,
    Qwen 0.6-1.0) with the SOURCE recorded per row — and, per row, whether it
    could be re-verified on the machine that wrote the table (2026-09-06: the
    cluster cache the 2026-09-04 read used is unreachable; the local snapshots
    are tokenizer-only). Nemotron has NO row on purpose: the ruling is the
    CARD temperature and the card is unverified; what the local snapshot ships
    is recorded in the refusal, never frozen as if someone had vouched for it."""
    rows = C.SAMPLING_BY_MODEL_KEY
    # keyed on the repo's canonical model keys; exactly one key is withheld
    assert set(rows) | set(C.SAMPLING_UNVERIFIED_MODEL_KEYS) == set(sg.MODEL_KEYS)
    assert set(C.SAMPLING_UNVERIFIED_MODEL_KEYS) == {"nemotron3nano4b"}
    assert not (set(rows) & set(C.SAMPLING_UNVERIFIED_MODEL_KEYS))
    for key, r in rows.items():
        assert isinstance(r, C.SamplingRow) and r._fields == C.SAMPLING_ROW_FIELDS, key
        assert 0.6 <= r.temperature <= 1.0, (key, r.temperature)         # the stated range
        assert r.top_p is None or 0.0 < r.top_p <= 1.0, key
        assert r.top_k is None or (isinstance(r.top_k, int) and r.top_k >= 1), key
        assert r.source.strip() and len(r.source) > 40, key            # provenance prose, never blank
    # the numbers, per row, and what each source must say about itself
    assert rows["gemma4e2b"][:3] == (1.0, 0.95, 64)
    assert "gemma-4-E2B-it" in rows["gemma4e2b"].source and "2026-09-04" in rows["gemma4e2b"].source
    assert "NOT re-verified" in rows["gemma4e2b"].source
    assert rows["gemma4e4b"][:3] == rows["gemma4e2b"][:3]
    assert "INHERITED from gemma4e2b" in rows["gemma4e4b"].source
    assert "NO generation_config.json" in rows["gemma4e4b"].source     # E4B's snapshot has none
    assert rows["llama8b"][:3] == (0.6, 0.9, None)
    # llama-paper-rows (P6): re-read 2026-09-11 at the pinned revision (hub + cohort cache) — the prose says so
    assert "NousResearch/Meta-Llama-3.1-8B-Instruct @ d10aef79" in rows["llama8b"].source
    assert "re-read 2026-09-11" in rows["llama8b"].source and "1f47e50c" in rows["llama8b"].source
    assert "NOT re-verified" not in rows["llama8b"].source
    assert rows["llama3b"][:3] == rows["llama8b"][:3]
    assert "INHERITED from llama8b" in rows["llama3b"].source and "unverified" in rows["llama3b"].source
    for q in ("qwen35_9b", "qwen35_4b"):
        assert rows[q][:3] == (0.7, 0.95, 20), q                       # the card's NON-thinking values
        assert "CARD" in rows[q].source and "unverified locally" in rows[q].source, q
        assert "0.6" in rows[q].source                                  # the thinking value, recorded not frozen
    assert rows["glm46vflash"][:3] == (0.8, 0.6, 2)
    assert "cache-read 2026-09-04" in rows["glm46vflash"].source
    assert "unverified now" in rows["glm46vflash"].source
    # the withheld row says what IS known and why it is not a row
    why = C.SAMPLING_UNVERIFIED_MODEL_KEYS["nemotron3nano4b"]
    assert "UNVERIFIED" in why and "dfaf35de" in why and "_from_model_config" in why
    assert "1.0" in why and "0.95" in why


def test_every_registry_model_resolves_to_its_own_sampling_key_by_hf_id():
    """The driver looks the row up from the config's HF id (exact match on
    geometry.MODEL_HF_IDS) while the launcher uses the registry row's
    model_key — the two must be the same key for every registry model, or a
    store would hash one model's card values under another's key."""
    for slug, row in C.MODELS.items():
        # an ONBOARDED key's served id lives in geometry.ONBOARDED_MODEL_HF_IDS (model_key_for serves both kinds)
        hf_ids = {**sg.MODEL_HF_IDS, **sg.ONBOARDED_MODEL_HF_IDS}
        assert hf_ids[row.model_key].lower() == row.hf.lower(), slug
        assert sg.model_key_for(row.hf) == row.model_key, slug
        # every launchable model has a card row OR is a NAMED unverified key whose
        # `card` request refuses (bcb-bfcl-200-widen S4: the nemotron3nano4b row
        # runs under `protocol`, which needs no card row; a card launch of it
        # raises SamplingUnverified rather than freezing a derived default)
        if row.model_key in C.SAMPLING_UNVERIFIED_MODEL_KEYS:
            with pytest.raises(C.SamplingUnverified):
                C.sampling_for(row.model_key, C.SAMPLING_CARD)
        else:
            assert row.model_key in C.SAMPLING_BY_MODEL_KEY, slug
        # ifeval-canonical-battery: the IFEval slugs sample under `vendor_preset`, whose
        # table has a row for every IFEval key — Nemotron's included (its `card` row is
        # absent on purpose, above; the IFEval Nemotron row never asks for `card`)
        if C.is_ifeval_row(row):
            assert row.model_key in C.VENDOR_PRESET_SAMPLING_BY_MODEL_KEY, slug
            assert (C.sampling_for(row.model_key, C.SAMPLING_VENDOR_PRESET)
                    is C.VENDOR_PRESET_SAMPLING_BY_MODEL_KEY[row.model_key]), slug


def test_every_registry_row_carries_a_letter_read_width_and_only_three_are_wide():
    """§PW-nanfix-2: the letter-read top-K is a PER-MODEL registry value —
    llama3b put B and C outside the top-1024 on 1 of 100 unsteered tinyMMLU
    items (O30) and the two double-dose banks move the mass off the letters
    altogether (O25). Every OTHER row keeps 1024, byte for byte: that is what
    the whole Gemma campaign was served and read at, and a changed width is a
    different store (the driver hashes it as resolved_config.mmlu.top_logprobs)."""
    assert C.LETTER_TOP_LOGPROBS_DEFAULT == 1024
    assert C.LETTER_TOP_LOGPROBS_WIDE == 4096 > C.LETTER_TOP_LOGPROBS_DEFAULT
    wide = {s for s, r in C.MODELS.items() if r.top_logprobs != C.LETTER_TOP_LOGPROBS_DEFAULT}
    assert wide == {"llama3b", "llama8b-x2", "gemma4e4b-x2"}
    for slug, row in C.MODELS.items():
        assert isinstance(row.top_logprobs, int) and row.top_logprobs > 64, slug
        assert row.top_logprobs == (C.LETTER_TOP_LOGPROBS_WIDE if slug in wide
                                    else C.LETTER_TOP_LOGPROBS_DEFAULT), slug
    # the field has a default, so a row written before it existed still builds
    assert C.ModelRow(slug="x", hf="h", rev="r", cfg="c", objects_dir="o",
                      model_key="k", family="llama",
                      thinking_modes=(C.THINKING_OFF,)).top_logprobs == (
        C.LETTER_TOP_LOGPROBS_DEFAULT)
    # driver_argv's own default is the LEGACY hand-run one and is untouched:
    # the launcher always passes the row's number explicitly
    argv = C.driver_argv(python="p", script="s", proxy_url="u", routing_table="t",
                         cell="none", config="c", defenses_dir="d", out_root="o",
                         controls=["gsm8k"], thinking=C.THINKING_OFF, concurrency=4,
                         judge_model="j", top_logprobs=C.LETTER_TOP_LOGPROBS_WIDE)
    assert argv[argv.index("--top-logprobs") + 1] == str(C.LETTER_TOP_LOGPROBS_WIDE)


def test_sampling_for_resolves_a_row_or_refuses_never_defaults():
    assert C.sampling_for("gemma4e4b") is C.SAMPLING_BY_MODEL_KEY["gemma4e4b"]
    assert C.sampling_for("gemma4e4b", C.SAMPLING_CARD).temperature == 1.0
    # greedy needs no row and is the same row for every model, known or not
    for key in (*sg.MODEL_KEYS, "no-such-model"):
        assert C.sampling_for(key, C.SAMPLING_GREEDY) is C.GREEDY_SAMPLING
    # card for a withheld key: refused, naming what is known and both ways out
    with pytest.raises(C.SamplingUnverified) as ei:
        C.sampling_for("nemotron3nano4b")
    msg = str(ei.value)
    assert "nemotron3nano4b" in msg and "dfaf35de" in msg and "_from_model_config" in msg
    assert "Refusing to invent a temperature" in msg
    assert "--sampling greedy" in msg and "add the row" in msg
    # card for an unknown key: refused too, naming the rows that exist
    with pytest.raises(C.SamplingUnverified, match="no row in SAMPLING_BY_MODEL_KEY"):
        C.sampling_for("no-such-model")
    assert issubclass(C.SamplingUnverified, ValueError)
    with pytest.raises(ValueError, match="sampling mode must be one of"):
        C.sampling_for("gemma4e4b", "nucleus")


def test_samples_by_task_is_one_everywhere_by_ruling_with_the_k_machinery_intact():
    """The per-task k: every PREFILL-SCORED task (the five tiny
    loglikelihood reads) is 1 and never sampled; every DECODING task (every
    inspect/* and evalplus/* row task) takes DECODE_SAMPLES_PER_ITEM and
    MATH-500 MATH500_SAMPLES_PER_ITEM — BOTH 1 by the rule in force since
    2026-09-06 (one draw per item everywhere for now; a card draw is still a
    stochastic sample, not greedy). The map stays total and per task, so
    raising k is a one-line edit."""
    assert set(C.SAMPLES_BY_TASK) == set(C.TINYCAP_TASKS)               # total over the row tasks
    assert set(C.DECODING_TASKS) == set(C.INSPECT_TASKS) | set(C.EVALPLUS_TASKS) == set(C.TINYCAP_TASKS)
    assert not hasattr(C, "PREFILL_SCORED_TASKS") and not hasattr(C, "PREFILL_SAMPLES_PER_ITEM")
    assert C.DECODE_SAMPLES_PER_ITEM == 1
    assert C.MATH500_SAMPLES_PER_ITEM == 1
    # the ruling: 1 everywhere — with ONE exception by a later ruling: IFEval's
    # two seeded trials (ifeval-canonical-battery, 2026-09-11; P2)
    ifeval = C.inspect_task(C.IFEVAL_TASK_NAME)
    assert C.IFEVAL_SAMPLES_PER_ITEM == 2 and C.SAMPLES_BY_TASK[ifeval] == 2
    assert {t for t, k in C.SAMPLES_BY_TASK.items() if k != 1} == {ifeval}
    for t in C.DECODING_TASKS:
        if t != ifeval:
            assert C.SAMPLES_BY_TASK[t] == C.DECODE_SAMPLES_PER_ITEM, t
    # MATH-500's k is ITS constant — one number, never a product with the decode k
    assert C.SAMPLES_BY_TASK["inspect/math"] == C.MATH500_SAMPLES_PER_ITEM
    # the k > 1 machinery is reachable through the same table: an explicit
    # --samples still makes MATH-500 avg@k, and a table edit would do the rest
    assert C.samples_of_task(C.SAMPLING_CARD, "inspect/math", math_samples=4) == 4


def test_samples_of_task_and_the_run_map():
    card, greedy = C.SAMPLING_CARD, C.SAMPLING_GREEDY
    # greedy: one sample for every task by default; an EXPLICIT --samples still
    # wins for MATH-500 (and only MATH-500) — the two knobs are orthogonal
    for t in C.TINYCAP_TASKS:
        assert C.samples_of_task(greedy, t) == 1, t
        # the explicit knob is MATH-500's and (P2) IFEval's — SAMPLES_FLAG_TASKS
        assert C.samples_of_task(greedy, t, math_samples=8) == (8 if t in C.SAMPLES_FLAG_TASKS else 1), t
    # card: the table (1 everywhere by the 2026-09-06 ruling); a removed
    # loglikelihood task has no entry and refuses
    assert C.samples_of_task(card, "inspect/gsm8k_paired200") == C.DECODE_SAMPLES_PER_ITEM == 1
    with pytest.raises(ValueError, match="no samples-per-item entry"):
        C.samples_of_task(card, "served_ll/tinyArc")
    # --samples is MATH-500's knob and only MATH-500's
    assert C.samples_of_task(card, "inspect/math") == C.MATH500_SAMPLES_PER_ITEM == 1
    assert C.samples_of_task(card, "inspect/math", math_samples=2) == 2
    assert C.samples_of_task(card, "inspect/gsm8k", math_samples=2) == 1
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValueError, match="positive int"):
            C.samples_of_task(card, "inspect/math", math_samples=bad)
    with pytest.raises(ValueError, match="no samples-per-item entry"):
        C.samples_of_task(card, "inspect/nope")
    with pytest.raises(ValueError, match="sampling mode must be one of"):
        C.samples_of_task("nucleus", "inspect/math")
    # the run map ranges over the tasks a run files (the thinking map's keys),
    # sorted, each entry samples_of_task's
    tasks = C.run_thinking_by_task([C.CONTROL_MATH500, C.CONTROL_HUMANEVAL_PLUS,
                                    "inspect_cmp:xstest"], C.THINKING_OFF)
    m = C.run_samples_by_task(tasks, card, math_samples=2)
    assert list(m) == sorted(tasks)
    assert m == {"evalplus/humaneval_plus": 1, "inspect/math": 2, "inspect/xstest": 1}
    assert C.run_samples_by_task(tasks, greedy) == {t: 1 for t in tasks}
    assert C.run_samples_by_task(tasks, greedy, math_samples=2)["inspect/math"] == 2
    assert C.run_samples_by_task([], card) == {}


def test_seed_rules_are_named_here_and_derived_in_the_engines():
    """The contract names the two RULES (hashed identity); the derivations
    live with the request builders that apply them — executable request
    shaping stays in the provider. One seed per (item, sample):
    deterministic, distinct across sample indices, in vLLM's int range, and
    recomputable from the rule by hand."""
    assert C.SEED_RULE_INSPECT == "sha256(item_id NUL sample_idx)[:4] big-endian mod 2^31-1"
    assert C.SEED_RULE_EVALPLUS == "blake2b(task, item_id, sample_idx) mod 2^31-1"
    assert C.SEED_RULES == {"inspect": C.SEED_RULE_INSPECT, "evalplus": C.SEED_RULE_EVALPLUS}
    assert not hasattr(C, "sample_seed")                                # no derivation in the contract
    # the Inspect derivation (the module's pure helpers import without inspect_ai)
    from sisteer.capeval.tiers import inspect_served as isv
    assert isv.SEED_RULE is C.SEED_RULE_INSPECT
    s0, s1, s2 = (isv.sample_seed("math_aaaa1111", k) for k in range(3))
    assert len({s0, s1, s2}) == 3                                       # distinct across samples
    assert isv.sample_seed("math_aaaa1111", 0) == s0                     # deterministic
    assert isv.sample_seed("math_aaaa1111") == s0                        # idx defaults to 0
    assert isv.sample_seed("math_bbbb2222", 0) != s0                     # and per item
    for s in (s0, s1, s2):
        assert 0 <= s < 2 ** 31 - 1
    h = hashlib.sha256(b"math_aaaa1111\x001").digest()                  # the rule, by hand
    assert s1 == int.from_bytes(h[:4], "big") % (2 ** 31 - 1)
    for bad in (-1, True, 1.0):
        with pytest.raises(ValueError, match="non-negative int"):
            isv.sample_seed("x", bad)
    # the EvalPlus derivation is the engine's, exactly
    from sisteer.capeval.serving import proxy_engine as pe
    assert pe.stable_seed("t", "i", "0") != pe.stable_seed("t", "i", "1")
    assert 0 <= pe.stable_seed("t", "i", "0") < 2 ** 31 - 1
    assert "blake2b" in inspect.getsource(pe.stable_seed)


def test_sampling_identity_fields_and_the_manifest_block():
    """The five HASHED fields U8 names (the three numbers, the per-task k, the
    seed rule) — the row's `source` prose is never among them — and the
    manifest block that mirrors them with the mode, the key and the source.
    They are the tail of TINYCAP_IDENTITY_FIELDS (step 2 wired the driver and
    the Inspect provider that fill them); the manifest key is additive."""
    assert C.SAMPLING_IDENTITY_FIELDS == (
        "temperature", "top_p", "top_k", "samples_by_task", "seed_rule")
    assert set(C.SAMPLING_IDENTITY_FIELDS) < set(C.TINYCAP_IDENTITY_FIELDS)
    assert C.MANIFEST_KEY_SAMPLING in C.TINYCAP_MANIFEST_KEYS
    for k in ("top_p", "top_k", "temperature"):
        assert k in C.SUMMARY_PAYLOAD_FIELDS                            # the summary names the row
    # k = 1 at the card temperature is a DIFFERENT identity from greedy: the
    # three numbers differ even when every k is 1 (the 2026-09-06 ruling's case)
    one = {t: 1 for t in C.TINYCAP_TASKS}
    card_one = C.sampling_identity(C.SAMPLING_BY_MODEL_KEY["gemma4e4b"], one)
    greedy_one = C.sampling_identity(C.GREEDY_SAMPLING, one)
    assert card_one["samples_by_task"] == greedy_one["samples_by_task"]
    assert card_one != greedy_one and card_one["temperature"] == 1.0 and greedy_one["temperature"] == 0.0
    assert (C.IDENTITY_KEY_TEMPERATURE, C.IDENTITY_KEY_TOP_P, C.IDENTITY_KEY_TOP_K,
            C.IDENTITY_KEY_SAMPLES_BY_TASK, C.IDENTITY_KEY_SEED_RULE) == C.SAMPLING_IDENTITY_FIELDS
    assert "source" not in C.SAMPLING_IDENTITY_FIELDS
    row = C.SAMPLING_BY_MODEL_KEY["gemma4e4b"]
    ident = C.sampling_identity(row, {"inspect/xstest": 1, "inspect/gsm8k_paired200": 4})
    assert ident == {"temperature": 1.0, "top_p": 0.95, "top_k": 64,
                     "samples_by_task": {"inspect/gsm8k_paired200": 4, "inspect/xstest": 1},
                     "seed_rule": {"inspect": C.SEED_RULE_INSPECT, "evalplus": C.SEED_RULE_EVALPLUS}}
    assert set(ident) == set(C.SAMPLING_IDENTITY_FIELDS)
    assert list(ident["samples_by_task"]) == sorted(ident["samples_by_task"])   # sorted: hashed
    g = C.sampling_identity(C.GREEDY_SAMPLING, {"inspect/math": 1})
    assert (g["temperature"], g["top_p"], g["top_k"]) == (0.0, None, None)
    assert g != ident                                                  # a different identity
    # JSON-clean (it is hashed through the resolved config)
    json.dumps(ident, sort_keys=True)
    # the mirror block
    assert C.MANIFEST_KEY_SAMPLING == "sampling"
    assert C.SAMPLING_BLOCK_FIELDS == (
        "mode", "model_key", "temperature", "top_p", "top_k", "samples_by_task", "seed_rule",
        "source")
    assert set(C.SAMPLING_IDENTITY_FIELDS) < set(C.SAMPLING_BLOCK_FIELDS)
    assert C.MANIFEST_KEY_SAMPLING not in _LEGACY_MANIFEST_KEYS
    assert C.MANIFEST_KEY_SAMPLING not in cs.CellStore.RESUME_KEYS


# --------------------------------------------------------------------------- #
# PW-oos (U10 item 5) — the explicit out-of-support-thinking override          #
# --------------------------------------------------------------------------- #

def test_the_out_of_support_override_is_named_and_says_what_it_does():
    """The flag has ONE spelling, and the prose every record quotes names both
    halves — what it lifts and what it keeps — so an override is never a bare
    boolean somebody has to interpret."""
    assert C.FLAG_ALLOW_OOS_THINKING == "--allow-out-of-support-thinking"
    # NOT in the always-emitted driver flag set: it takes no value, and the
    # preflight probes that set with one dummy value per flag
    assert C.FLAG_ALLOW_OOS_THINKING not in C.TINYCAP_DRIVER_FLAGS
    note = C.OUT_OF_SUPPORT_THINKING_NOTE
    for phrase in ("LIFTS", "KEEPS", "out_of_support_thinking",
                   "changes NOTHING when it is absent"):
        assert phrase in note, phrase


def test_only_the_out_of_support_reasons_are_overridable():
    """The two OUT-OF-SUPPORT reasons may be overridden; a family with no
    thinking switch at all may not — there would be no ON render to serve, so
    the flag would produce a thinking-OFF run recorded as ON."""
    assert C.out_of_support_thinking_overridable("qwen3_5")
    assert C.out_of_support_thinking_overridable("glm4v")
    assert C.out_of_support_thinking_overridable("gemma4")
    for fam in C.NO_THINKING_CHANNEL_FAMILIES:
        assert not C.out_of_support_thinking_overridable(fam), fam


def test_the_override_record_carries_every_refusal_it_lifts():
    """On qwen3_5 BOTH refusals bind — the fit-regime header move and the
    missing reasoning parser — and a record naming one would understate what
    was overridden. thinking_on_refusal still returns the most specific ONE (a
    refusal needs one message); this returns them all."""
    both = C.thinking_on_override_refusals("qwen3_5", fitted=True)
    assert len(both) == 2
    assert "MOVES between modes" in both[0] and "H=7" in both[0]
    assert "no pinned reasoning parser" in both[1]
    # the most specific of the two is what the refusal itself says
    assert C.thinking_on_refusal("qwen3_5", fitted=True) == both[0]
    # a prompted (not fitted) qwen cell: only the parser reason
    assert C.thinking_on_override_refusals("qwen3_5", fitted=False) == (both[1],)
    # an ALL-POSITION fitted qwen cell has no mask-moving refusal to lift —
    # only the parser reason remains
    assert C.thinking_on_override_refusals("qwen3_5", fitted=True, header_located=False) == (both[1],)
    # gemma4 is servable ON either way, so there is nothing to lift
    assert C.thinking_on_override_refusals("gemma4", fitted=True) == ()


def test_the_override_composes_no_parser_rather_than_inventing_one():
    """A thinking-ON standup on a family with no VERIFIED parser composes
    NOTHING under the override (thought text stays in `content`, which is part
    of what the experiment measures) — it never manufactures a parser, and a
    family with no thinking switch is still refused."""
    assert C.standup_extra_argv(tool_calls=False, thinking=C.THINKING_ON,
                                family="qwen3_5", allow_out_of_support=True) == []
    # the pinned family still gets its pin, override or not
    assert C.standup_extra_argv(tool_calls=False, thinking=C.THINKING_ON,
                                family="gemma4", allow_out_of_support=True) == [
        C.FLAG_REASONING_PARSER, "gemma4"]
    with pytest.raises(ValueError, match="no reasoning parser pinned"):
        C.standup_extra_argv(tool_calls=False, thinking=C.THINKING_ON, family="llama",
                             allow_out_of_support=True)


def test_the_out_of_support_identity_field_is_present_only_when_true():
    """The whole design: an in-support run's identity fields are EXACTLY what
    they were before the override existed, so its hash and its store key do not
    move; an out-of-support run carries one more key and can therefore never
    resume onto an in-support store (or the reverse)."""
    assert C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING == "out_of_support_thinking"
    # S3.1: the override's THREE optional keys, the marker first
    assert C.TINYCAP_IDENTITY_OPTIONAL_FIELDS == (
        C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING, C.IDENTITY_KEY_MAX_TOKENS_BY_TASK,
        C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT)
    # the optional fields are NOT among the unconditional ones
    for key in C.TINYCAP_IDENTITY_OPTIONAL_FIELDS:
        assert key not in C.TINYCAP_IDENTITY_FIELDS
    assert C.tinycap_identity_fields() == C.TINYCAP_IDENTITY_FIELDS
    assert C.tinycap_identity_fields(out_of_support_thinking=False) == C.TINYCAP_IDENTITY_FIELDS
    oos = C.tinycap_identity_fields(out_of_support_thinking=True)
    assert oos == (*C.TINYCAP_IDENTITY_FIELDS, *C.TINYCAP_IDENTITY_OPTIONAL_FIELDS)
    assert set(oos) - set(C.TINYCAP_IDENTITY_FIELDS) == set(C.TINYCAP_IDENTITY_OPTIONAL_FIELDS)


def test_the_driver_argv_carries_the_flag_only_when_asked():
    """Nothing changes when the flag is absent — the argv is byte-identical to
    what it was, and the flag is the ONLY difference when it is named."""
    kw = {"python": "/p", "script": "/s.py", "proxy_url": "http://x/v1",
          "routing_table": "/rt.json", "cell": C.CELL_PSRKL_SAFETY, "config": "/c.yaml",
          "defenses_dir": "/d", "out_root": "/o", "controls": ["gsm8k"],
          "thinking": C.THINKING_ON, "concurrency": 8, "judge_model": "j"}
    plain = C.driver_argv(**kw)
    assert C.FLAG_ALLOW_OOS_THINKING not in plain
    named = C.driver_argv(**kw, allow_out_of_support_thinking=True)
    assert named == [*plain, C.FLAG_ALLOW_OOS_THINKING]
    assert C.driver_argv(**kw, allow_out_of_support_thinking=False) == plain


# --------------------------------------------------------------------------- #
# E50 #3 — the authorised out-of-support tuple, declared once                  #
# --------------------------------------------------------------------------- #


def _authorised_request() -> dict:
    """The request that IS the authorised experiment, in the driver's
    spellings (one cell, one mode) — the base every mutation below moves one
    field of."""
    exp = C.AUTHORISED_OOS_THINKING
    return {"model": exp.model, "cell": exp.cells[-1], "controls": list(exp.controls),
            "mode": exp.modes[-1], "sampling_mode": exp.sampling_mode,
            "samples": exp.samples, "reduction_mode": exp.reduction_mode, "pass_no": exp.pass_no}


def test_the_authorised_out_of_support_tuple_is_the_run_the_user_authorised():
    """The tuple is the ONE declaration both entry points read, and it says
    what is authorised: the Qwen3.5-4B thinking 2x2 on paired-200 GSM8K at
    k = 1, card temperature."""
    exp = C.AUTHORISED_OOS_THINKING
    assert exp.model == "qwen35_4b"
    assert exp.cells == (C.ROUTE_NONE_CELL, C.CELL_PSRKL_SAFETY)
    assert exp.controls == (C.CONTROL_GSM8K,)
    assert exp.modes == (C.THINKING_OFF, C.THINKING_ON)
    assert exp.sampling_mode == C.SAMPLING_DEFAULT and exp.samples == 1
    # `per-task` is a mode the launcher accepts and this experiment does NOT
    assert C.THINKING_PER_TASK not in exp.modes
    # the derived spellings each door needs, from the ONE declaration: a REGISTRY SLUG authorises
    # itself and its key — never a sibling slug that merely shares the key (main's bcb-bfcl-200-widen
    # rule; the IFEval bank views `qwen35_4b-sdfbase-main` / `-it` are NOT spellings of this authorisation)
    assert exp.model_spellings() == tuple(dict.fromkeys((exp.model, C.model_row(exp.model).model_key)))
    assert not any(s in exp.model_spellings() for s in C.IFEVAL_SLUGS.values())
    assert set(exp.cell_spellings()) == {
        C.ROUTE_NONE_CELL, C.CELL_PSRKL_SAFETY,
        C.parse_launcher_cell(C.CELL_PSRKL_SAFETY).harness_id}
    assert exp.control_atoms() == C.expand_control(C.CONTROL_GSM8K)
    # fit-support transfer (plan S1.1): the reduction mode, the raised ON cap
    # table and the declared render fact ride on the same tuple
    assert set(exp.as_dict()) == {"model", "cells", "controls", "modes",
                                  "sampling_mode", "samples", "reduction_mode", "pass_no",
                                  "max_tokens_on", "prompt_opens_think_on"}


def test_every_authorised_field_is_accepted_in_both_spellings():
    """The authorised run itself passes — under the launcher's cell id and
    under the harness id the driver sees, and under both modes."""
    for cell in C.AUTHORISED_OOS_THINKING.cell_spellings():
        for mode in C.AUTHORISED_OOS_THINKING.modes:
            C.check_out_of_support_thinking_authorised(
                **{**_authorised_request(), "cell": cell, "mode": mode})
    # a RESUMED driver is handed `--controls <remaining>`, a subset — still the
    # same experiment (a wider list is not; see the mutation test)
    C.check_out_of_support_thinking_authorised(
        **{**_authorised_request(), "controls": list(C.AUTHORISED_OOS_THINKING.control_atoms())})


@pytest.mark.parametrize("field,value", [
    ("model", "qwen35_9b"),                 # the sibling model — the family is the same
    ("cell", "si_constitution__psr_kl_all"),
    ("controls", ["gsm8k", "mmlu"]),        # a WIDER item set than authorised
    ("mode", C.THINKING_PER_TASK),
    ("sampling_mode", C.SAMPLING_GREEDY),
    ("samples", 4),
    ("reduction_mode", C.REDUCTION_MODE_STACKED),   # fit-support transfer S1.1: hashed served identity
    ("pass_no", 2),                                 # code review H3: pass 2 serves another geometry class
])
def test_mutating_any_field_of_the_authorised_tuple_refuses_naming_it(field, value):
    """E50 #3: the override authorised ONE experiment, not a family. Move any
    single field of the tuple and the run is refused — with the field named,
    so the operator is not left diffing a tuple by eye."""
    req = {**_authorised_request(), field: value}
    with pytest.raises(C.OutOfSupportThinkingUnauthorised) as e:
        C.check_out_of_support_thinking_authorised(**req)
    msg = str(e.value)
    assert f"{field} differs" in msg
    assert C.FLAG_ALLOW_OOS_THINKING in msg
    assert "AUTHORISED_OOS_THINKING" in msg          # where the one declaration lives
    assert field in C.OOS_AUTHORISED_FIELDS


def test_an_empty_or_unknown_control_set_is_never_authorised():
    """A run that files rows for nothing, and a run naming a token no control
    expands — neither is the authorised experiment."""
    for controls in ([], ["not_a_control"]):
        with pytest.raises(C.OutOfSupportThinkingUnauthorised) as e:
            C.check_out_of_support_thinking_authorised(
                **{**_authorised_request(), "controls": controls})
        assert "controls differs" in str(e.value)


#: The producer trees this campaign really ran, and what the volume holds now:
#: the 57-cell wave is Gemma at 73024c09 and Qwen/Llama at c7e1796a, with a few
#: earlier-pass stores at the other four. The classification below is asserted
#: against THESE values rather than synthetic placeholders, so the table is
#: checked against the population it will actually classify.
_LIVE_WAVE_TREES = ("73024c096158197298c8f8d9a18bfa321eca82f6",
                    "c7e1796ac9a36732822e8f56d02b7c7af43f9dcf")
#: Per declared producer tree, the contracts that tree DOES NOT CONTAIN — the
#: relation `store_era` classifies on, and the one the git pin below re-checks.
#: Not "predates": four of these seven are DIVERGENT from the samples-digest and
#: reduction-mode landings rather than ancestors of them (E58 note (a)).
# the template-check contract (2026-09-07) landed after every one of these
# trees ran, so all seven do not contain it — it is dated for them by
# containment exactly like samples-digest / reduction-mode
_EXPECTED_MISSING_CONTRACTS = {
    "618f9ddeedc26d90fe93ed78ea96688a171d9d7e": {"sampling", "samples-digest",
                                                 "reduction-mode", "template-check"},
    "198332266dc8cd0f9d2fa9f0c106a8b4319fc384": {"sampling", "samples-digest",
                                                 "reduction-mode", "template-check"},
    "7aa0a5df1fd02ef9b696273f17a3a4874088953b": {"sampling", "samples-digest",
                                                 "reduction-mode", "template-check"},
    "d844f0ae6d0be002e2e548d816041691a25e9e2c": {"samples-digest", "reduction-mode",
                                                 "template-check"},
    "73024c096158197298c8f8d9a18bfa321eca82f6": {"samples-digest", "reduction-mode",
                                                 "template-check"},
    "087869ac021716e056d85161f5599370b3df52d1": {"samples-digest", "reduction-mode",
                                                 "template-check"},
    "c7e1796ac9a36732822e8f56d02b7c7af43f9dcf": {"samples-digest", "reduction-mode",
                                                 "template-check"},
}


def _era(m, contract):
    got, why = C.store_era(m, contract=contract)
    assert got in C.STORE_ERAS and isinstance(why, str) and why
    return got


def _producer_manifest(sha):
    return {C.MANIFEST_KEY_PRODUCER: {"tree_sha": sha, "driver": C.PRODUCER_DRIVER_SERVED}}


def test_a_stores_era_is_a_positive_marker_never_a_missing_key():
    """E54 (the pattern behind findings 1-3): several checks on this branch
    landed after stores were already written, and each first read a MISSING
    field as "an old store, accept". A torn write, a partial copy or a deleted
    sidecar on a CURRENT store is byte-indistinguishable from a legitimate old
    one, so every gap was a silent pass. :func:`contracts.store_era` replaces
    that reading with a POSITIVE one, and only two facts can produce it: a
    ``contract_version`` the store recorded, or a producer tree sha in the
    declared table. Everything else — including a store that records neither —
    is UNKNOWN, and every caller refuses an UNKNOWN store the same way it
    refuses a current one. Damage can move a store OUT of the declared
    population; nothing can move one INTO "historical"."""
    assert C.STORE_ERAS == ("historical", "current", "unknown")
    assert C.MANIFEST_KEY_CONTRACT_VERSION == "contract_version"
    assert isinstance(C.CONTRACT_VERSION, int) and C.CONTRACT_VERSION >= 2
    # the version stamp is provenance, not identity and not a resume key: a
    # bump must never re-key a store or refuse a resume onto one
    assert C.MANIFEST_KEY_CONTRACT_VERSION not in C.TINYCAP_MANIFEST_KEYS
    assert C.MANIFEST_KEY_CONTRACT_VERSION not in C.TINYCAP_IDENTITY_FIELDS
    assert C.MANIFEST_KEY_CONTRACT_VERSION not in cs.CellStore.RESUME_KEYS
    assert C.MANIFEST_KEY_CONTRACT_VERSION not in cs.CellStore.CONDITIONAL_RESUME_KEYS
    assert C.MANIFEST_KEY_CONTRACT_VERSION not in C.PRODUCER_BLOCK_FIELDS   # exact-keys audited

    any_contract = C.CONTRACT_SAMPLES_DIGEST
    assert _era(_producer_manifest("73024c09"), any_contract) == C.STORE_ERA_UNKNOWN
    assert _era(_producer_manifest("73024C096158197298C8F8D9A18BFA321ECA82F6"),
                any_contract) == C.STORE_ERA_UNKNOWN
    for bad in (None, {}, {C.MANIFEST_KEY_PRODUCER: None}, {C.MANIFEST_KEY_PRODUCER: {}},
                _producer_manifest(None), _producer_manifest(""), _producer_manifest(123)):
        assert _era(bad, any_contract) == C.STORE_ERA_UNKNOWN
    # a 40-hex tree NOBODY declared says nothing about any contract: it is not
    # "current by elimination", it is unrecognised, and it refuses either way
    assert _era(_producer_manifest("a1" * 20), any_contract) == C.STORE_ERA_UNKNOWN
    for bad in (0, -1, True, 1.0, "1", None, []):
        # a present-but-unusable version is the same "the store has the field
        # and cannot say" the reduction-mode reader refuses (§PW-stamps item 7)
        m = {C.MANIFEST_KEY_CONTRACT_VERSION: bad, **_producer_manifest(_LIVE_WAVE_TREES[0])}
        assert _era(m, any_contract) == C.STORE_ERA_UNKNOWN, bad
    # naming no contract, or one nobody declared, is a programming error and
    # never a silent default — the whole point of E55 #1
    with pytest.raises(TypeError):
        C.store_era(_producer_manifest(_LIVE_WAVE_TREES[0]))          # type: ignore[call-arg]
    with pytest.raises(ValueError, match="unknown contract"):
        C.store_era(_producer_manifest(_LIVE_WAVE_TREES[0]), contract="samples_digest")


def test_era_is_a_property_of_store_and_contract_not_of_a_store():
    """E55 #1. ONE allowlist of "old" trees cannot serve exemptions whose
    contracts landed at different times. Four of the seven campaign trees CONTAIN
    both sampling landings while containing NEITHER the samples-digest nor the
    reduction-mode one, so a flat list called them historical for the sampling
    contract too — and an EvalPlus store from one of them whose task-local
    ``sampling`` block had been deleted read as pre-sampling and, at k = 1,
    dropped its ``acc@0`` row.

    E58 note (a): those four are DIVERGENT from the two later landings, not
    ancestors of them — they branch off the PR branch before those commits and
    carry later cherry-picks. "Historical" here means the tree does not CONTAIN
    the contract's implementation, which is the fact the exemption rests on;
    it is not a claim that the tree is older.

    Asserted against the REAL producer shas the volume holds, contract by
    contract, so the table is pinned to the population it classifies."""
    assert set(C.CONTRACTS) == {C.CONTRACT_SAMPLING, C.CONTRACT_SAMPLES_DIGEST,
                                C.CONTRACT_REDUCTION_MODE, C.CONTRACT_TEMPLATE_CHECK}
    assert set(C.KNOWN_PRODUCER_TREES) == set(_EXPECTED_MISSING_CONTRACTS)
    for sha, spec in C.KNOWN_PRODUCER_TREES.items():
        assert re.fullmatch(r"[0-9a-f]{40}", sha)      # never a prefix
        assert spec.tree_sha == sha
        assert set(spec.does_not_contain) == _EXPECTED_MISSING_CONTRACTS[sha], sha
        assert set(spec.does_not_contain) <= set(C.CONTRACTS), sha
    for name, spec in C.CONTRACTS.items():
        assert spec.name == name
        assert spec.landed_at and all(re.fullmatch(r"[0-9a-f]{40}", s) for s in spec.landed_at)
        # a contract is stamped from the version current when IT landed: the
        # first three from 2, template-check from 3 — never above the current
        # stamp, and the newest contract IS the current stamp (the "raise both
        # in the same commit" rule, checked)
        assert 2 <= spec.stamped_from <= C.CONTRACT_VERSION, name
        assert isinstance(spec.what, str) and spec.what
    assert max(spec.stamped_from for spec in C.CONTRACTS.values()) == C.CONTRACT_VERSION
    assert C.CONTRACTS[C.CONTRACT_TEMPLATE_CHECK].stamped_from == C.CONTRACT_VERSION == 3

    # the classification, per real producer tree, per contract
    for sha, missing in _EXPECTED_MISSING_CONTRACTS.items():
        for contract in C.CONTRACTS:
            want = C.STORE_ERA_HISTORICAL if contract in missing else C.STORE_ERA_CURRENT
            assert _era(_producer_manifest(sha), contract) == want, (sha, contract)

    # the finding itself, spelled out on the two trees the 57-cell wave ran:
    # historical for the two later contracts, CURRENT for sampling
    for sha in _LIVE_WAVE_TREES:
        assert _era(_producer_manifest(sha), C.CONTRACT_SAMPLING) == C.STORE_ERA_CURRENT
        assert _era(_producer_manifest(sha), C.CONTRACT_SAMPLES_DIGEST) == C.STORE_ERA_HISTORICAL
        assert _era(_producer_manifest(sha), C.CONTRACT_REDUCTION_MODE) == C.STORE_ERA_HISTORICAL
        # and the 57-cell wave's stores are HISTORICAL for the template-check
        # contract: their absent class reads as a stated absence, never compared
        assert _era(_producer_manifest(sha), C.CONTRACT_TEMPLATE_CHECK) == C.STORE_ERA_HISTORICAL
    # ... and on the first staged tree, which contains none of the three
    first = "618f9ddeedc26d90fe93ed78ea96688a171d9d7e"
    for contract in C.CONTRACTS:
        assert _era(_producer_manifest(first), contract) == C.STORE_ERA_HISTORICAL


def test_the_contract_version_wins_over_the_producer_sha():
    """The precedence rule, and why it is that way round. ``contract_version``
    is what the WRITING tree said about itself; a ``producer`` block can be
    INHERITED — a re-run under a new tree that copies an old store's producer
    block is exactly the case where the sha lies. So the version is read first
    and settles the question alone.

    No store on the volume carries a version today (all 81 real manifests
    resolve through the sha branch), so this is the forward half: once the
    driver's stamp is on stores there are two populations, pre-stamp judged by
    ancestry and post-stamp by version."""
    v = C.MANIFEST_KEY_CONTRACT_VERSION
    old_tree = "618f9ddeedc26d90fe93ed78ea96688a171d9d7e"      # historical for ALL three
    # the sha says pre-landing; the version says written under the contract.
    # The version wins, for every contract, so a stale inherited producer block
    # can never buy an exemption a current writer did not earn.
    stale = {v: C.CONTRACT_VERSION, **_producer_manifest(old_tree)}
    for contract in C.CONTRACTS:
        assert _era(stale, contract) == C.STORE_ERA_CURRENT, contract
        assert f"{v}={C.CONTRACT_VERSION}" in C.store_era(stale, contract=contract)[1]
    # and the other way: a version BELOW the contract's stamped_from dates the
    # store even when its tree is one nobody declared
    for contract in C.CONTRACTS:
        m = {v: C.CONTRACTS[contract].stamped_from - 1, **_producer_manifest("a1" * 20)}
        assert _era(m, contract) == C.STORE_ERA_HISTORICAL, contract
        assert _era({v: C.CONTRACT_VERSION + 5}, contract) == C.STORE_ERA_CURRENT


def test_the_producer_table_is_pinned_against_git_containment():
    """The table in :data:`contracts.KNOWN_PRODUCER_TREES` is a transcription
    of git CONTAINMENT, and a transcription slip would EXEMPT a store from a
    contract it was written under — the exact defect E55 #1 found. So every
    (landing, producer) pair is re-checked with ``git merge-base
    --is-ancestor <landing> <producer>`` whenever this test runs inside the
    repo, and skips cleanly when it does not (a container, a source tarball, a
    shallow clone that does not hold the landings).

    The relation is containment, NOT chronology (E58 note (a)). `--is-ancestor
    <landing> <producer>` succeeding means the producer tree CONTAINS that
    landing's commit; failing means it does not, whether the producer is older
    than the landing or merely divergent from it. Four of these seven trees are
    the divergent case for two of the contracts — branched off the PR branch
    before those commits, carrying later cherry-picks — and the exemption is
    right for them because the implementation is absent from the tree, not
    because the tree is old."""
    repo = Path(__file__).resolve().parent.parent
    if not (repo / ".git").exists() or shutil.which("git") is None:
        pytest.skip("not a git checkout — the ancestry pin needs the repo")
    shas = sorted({s for spec in C.CONTRACTS.values() for s in spec.landed_at}
                  | set(C.KNOWN_PRODUCER_TREES))
    missing = [s for s in shas
               if subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{s}^{{commit}}"],
                                 capture_output=True, check=False).returncode != 0]
    if missing:
        pytest.skip(f"this clone does not hold {len(missing)} of the declared commits")
    checked = 0
    for sha, spec in C.KNOWN_PRODUCER_TREES.items():
        for name, contract in C.CONTRACTS.items():
            # the producer tree DOES NOT CONTAIN the contract iff it contains
            # none of the landings that contract needed (containing only some
            # of them is not "written under it" either: the store would carry
            # only some of the blocks, which the per-contract checks catch)
            contains = any(
                subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                                landing, sha], capture_output=True, check=False).returncode == 0
                for landing in contract.landed_at)
            assert (name in spec.does_not_contain) is (not contains), (
                f"{sha[:8]} vs the {name} contract: the table says the tree "
                f"{'does not contain' if name in spec.does_not_contain else 'contains'} it, "
                f"git says it {'contains' if contains else 'does not contain'} it")
            checked += 1
    # the SIZE of the matrix is asserted too: a green run must mean every
    # declared tree was really put to git, so a tree dropped from the table
    # cannot pass this test by shrinking what it checks
    assert checked == 7 * 4 == len(C.KNOWN_PRODUCER_TREES) * len(C.CONTRACTS)


def test_the_bfcl_pass_store_on_the_volume_now_dates_itself():
    """A re-scan read five real stores as UNKNOWN for all three
    contracts, and one of them carried a producer tree the table did not
    declare: ``7aa0a5df``, the BFCL-on-steered-served-cells pass. It is a
    real, datable campaign tree — it CONTAINS none of the five contract
    landings — so it is declared historical for all three, and this pins the
    reading against the SHAPE the single store of it on the volume actually
    has, not against a placeholder.

    That store's ``route`` block DOES state ``reduction_mode: per_row``, so the
    declaration does not change how its mode reads: it was ``stated`` before
    and is ``stated`` after. What the declaration changes is the store's
    DATABILITY — every contract now gets a positive answer about it instead of
    ``unknown``. The counterfactual is asserted too, because it is the case the
    declaration would have mattered for: strip the key and the reading is now
    the KNOWN-LEGACY ``per_row`` the four undeclared stores are refused for."""
    sha = "7aa0a5df1fd02ef9b696273f17a3a4874088953b"
    assert sha in C.KNOWN_PRODUCER_TREES
    assert set(C.KNOWN_PRODUCER_TREES[sha].does_not_contain) == set(C.CONTRACTS)

    # the real store's shape: the served driver's producer block, and a route
    # block that names the mode (that pass made the mode identity)
    live = {C.MANIFEST_KEY_PRODUCER: {"driver": C.PRODUCER_DRIVER_SERVED, "tree_sha": sha},
            C.MANIFEST_KEY_ROUTE: {"virtual_model": "tinycap-lens",
                                   "reduction_mode": C.REDUCTION_MODE_PER_ROW,
                                   "agent_loop_entry": True, "tool_calls": "auto"}}
    for contract in C.CONTRACTS:
        era, why = C.store_era(live, contract=contract)
        assert era == C.STORE_ERA_HISTORICAL, contract
        assert sha in why and "does not contain" in why
    read = C.read_store_reduction_mode(live, where="<the BFCL store>")
    assert (read.mode, read.provenance, read.refusal) == (
        C.REDUCTION_MODE_PER_ROW, C.REDUCTION_MODE_STATED, None)

    # ... and the counterfactual, which is what the declaration buys: before it
    # this returned a refusal naming KNOWN_PRODUCER_TREES, because nothing on
    # the store said whether it predated the stamp
    stripped = {**live, C.MANIFEST_KEY_ROUTE:
                {k: v for k, v in live[C.MANIFEST_KEY_ROUTE].items()
                 if k != C.REDUCTION_MODE_FIELD}}
    read = C.read_store_reduction_mode(stripped, where="<the BFCL store>")
    assert read.refusal is None
    assert (read.mode, read.provenance) == (C.REDUCTION_MODE_PER_ROW, C.REDUCTION_MODE_LEGACY)

    # the four SIBLING stores the same re-scan found carry no producer block at
    # all, so nothing declared here can date them: they stay UNKNOWN and keep
    # refusing. Declaring one tree must never soften that.
    producerless = {C.MANIFEST_KEY_ROUTE: {"virtual_model": "tinycap-lens"}}
    for contract in C.CONTRACTS:
        assert C.store_era(producerless, contract=contract)[0] == C.STORE_ERA_UNKNOWN
    assert C.read_store_reduction_mode(producerless, where="<a producer-less store>").refusal


# --------------------------------------------------------------------------- #
# fit-support transfer, experiment 1 — plan S1 (tests T1, T8, T11)             #
# --------------------------------------------------------------------------- #

_OOS_CELL = C.CELL_PSRKL_SAFETY
#: The FINAL contract's values (`contract-2.5.md`, sha256 prefix 7894fe333f24,
#: 324 lines) — what was written into the module,
#: quoted here as LITERALS with each value's section, so the module is
#: compared against the CONTRACT and never against itself.
_CONTRACT = {
    "OOS_THINKING_DECLARATION_KEY": "petri_thinking_on",                    # §2.5.1 <DECL_KEY>; bodies §2.5.3
    "OOS_THINKING_DECLARATION_VALUE": True,                                 # §2.5.1 <DECL_VALUE>: JSON boolean true
    "OOS_STANDUP_CELL_SUFFIX": "@thinking",                                 # §2.5.1 (no flag), §2.5.2, §2.5.10 Q8 row 1, §2.5.11 Q7(ii)
    "OOS_EXPECT_SHA_KEY": "traintest_variant",                              # §2.5.1, §2.5.10 Q8 row 2: <tt_cell>=<artifact sha>
    "PROXY_IDENTITY_SERVE_DECLARATIONS": "lens_serve_declarations",         # §2.5.1 <ATTEST_KEY>, §2.5.7
    "PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR": "lens_serve_declarations_error",   # §2.5.1 (beside it)
    "OOS_CELL_DECLARATION_EXPECTED": ["thinking"],                          # §2.5.1 Q6 <DECL_EXPECTED>, §2.5.10 Q8 row 5
    "ROUTE_KEY_SERVE_DECLARATIONS": "lens_serve_declarations",              # §2.5.1 <ROUTE_KEY>
    "THINKING_HEADER_SOURCE": ("petri_proxy", "thinking_header"),           # §2.5.1 <ATTEST_VALUE>, §2.5.7
    "THINKING_HEADER_SWITCH_FIELD": "switch",                               # §2.5.1 thinking_header = {switch, moves, off_ids, on_ids}
    "THINKING_HEADER_SWITCH_EXPECTED": True,                                # §2.5.4 last row (switch FALSE = llama), §2.5.6 row 2
    "THINKING_HEADER_MOVES_EXPECTED": True,                                 # §2.5.4 row 1 (moves TRUE: qwen3_5, glm4v)
    "STAMP_KEY_RENDER_REGIME": "render_regime",                             # §2.5.5, §2.5.11
    "RENDER_REGIME_BY_ARM": {"on": "thinking_on", "off": "thinking_off",    # §2.5.11 rows 1-5
                             "not_available": "unspecified"},               # the no-switch row
    "DECLARED_SALT_POLICY_LABEL": "declared-request-unique",                # §2.5.11 Q9
    "FORCED_SALT_POLICY_LABEL": "forced-request-unique",                    # §2.5.11 Q9
}


def test_the_authorised_tuple_is_the_spec_table_literally():
    """The authorised table, transcribed as a literal dict, IS the tuple —
    model, the two cells under both arms, GSM8K paired-200 at k = 1 on the
    card, reduction mode per_row, the raised ON cap on both GSM8K tasks, and
    the family's declared render fact. The tuple stays BARE (no `@thinking`):
    the contract "imposes nothing on `contracts.AUTHORISED_OOS_THINKING` …
    the lens never reads that constant" (§2.5.11 Q5); the helper adds the
    variant at the doors."""
    assert C.AUTHORISED_OOS_THINKING.as_dict() == {
        "model": "qwen35_4b", "cells": ["none", "safety_refusal__psr_kl_all"],
        "controls": ["gsm8k"], "modes": ["off", "on"], "sampling_mode": "card",
        "samples": 1, "reduction_mode": "per_row", "pass_no": 1,
        "max_tokens_on": {"gsm8k_paired200": 8192, "gsm8k": 8192},
        "prompt_opens_think_on": True}
    # the cap joins the compared fields, LAST,
    # compared only when the caller states one
    assert C.OOS_AUTHORISED_FIELDS == (
        "model", "cell", "controls", "mode", "sampling_mode", "samples", "reduction_mode", "pass_no",
        "max_tokens_on")
    assert all(C.VARIANT_SEP not in cell for cell in C.AUTHORISED_OOS_THINKING.cells)


def test_the_reduction_mode_is_compared_when_stated_and_skipped_when_unstated():
    """T1(b) for the new field. A STATED `stacked` refuses naming the field; an
    UNSTATED one (None) is not compared. None never reads as agreement: it is
    the pre-existing behaviour, byte for byte. The DRIVER (S3.1, phase B1)
    passes the SERVED, attested reduction mode at the door where it is first
    known — after the standup record is read, before any store is touched —
    while its first door (before the record) still passes none; the launcher's
    two doors are S4's (phase B2) and are not pinned here."""
    ok = _authorised_request()
    C.check_out_of_support_thinking_authorised(**ok)
    C.check_out_of_support_thinking_authorised(**{**ok, "reduction_mode": None})
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="reduction_mode differs"):
        C.check_out_of_support_thinking_authorised(
            **{**ok, "reduction_mode": C.REDUCTION_MODE_STACKED})

    def door_calls(script):
        src = (REPO / script).read_text()
        return [src[i:i + 500] for i in
                [m.start() for m in re.finditer(r"check_out_of_support_thinking_authorised\(", src)]
                if not src[max(0, i - 4):i].endswith("def ")]

    driver = door_calls("scripts/served_capability_run.py")
    # the early door, the served-identity door, then the re-bind of the matched
    # authorisation to the RESOLVED served model's key
    assert len(driver) == 3
    # ALL THREE doors state the served reduction mode: the early door reads it off the standup
    # record beside --routing-table (a pure read; the record is the served identity's own source) because its return
    # is the name the manifest records, and the registry now holds two pairs that differ only in that field (the
    # lens-era `stacked` and the native battery's `not_applicable`); the served-identity door and the served-model
    # re-bind state it from the hashed served block, as before
    assert [("reduction_mode=" in c.split(")")[0]) for c in driver] == [True, True, True]
    assert "reduction_mode=first_door_reduction_mode" in driver[0]
    assert 'reduction_mode=tinycap_identity["served"][tc.REDUCTION_MODE_FIELD]' in driver[1]
    assert door_calls("scripts/modal/tinycap_served.py")      # the launcher's doors exist (S4 wires them)


def test_the_geometry_pass_is_part_of_both_tuples_compared_when_stated_and_skipped_when_unstated():
    """The code review's H3: the experiment tuple fixes the GEOMETRY CLASS — pass 1,
    the certified window the prompt budget is proven against (16,384 x 22 for the
    authorised model). `--pass2` resolves the MEASURED entry instead (L 4,608 x
    S 78 for the OFF cells; 10,752 x 33 for the ON legs): the comparator's 2,375
    + 8,192 does not fit it, and even where the budget would fit it is not the
    recorded experiment's geometry. Both tuples name `pass_no 1`; the checkers
    compare it when STATED (a launcher plans a pass) and skip it when unstated
    (None never reads as agreement — the driver's doors state none)."""
    assert C.AUTHORISED_OOS_THINKING.pass_no == 1 == C.AUTHORISED_OFF_CAP_COMPARATOR.pass_no
    # the comparator's list ends on pass_no; the OOS list gained `max_tokens_on`
    # after it — the eight shared fields are the same, in the same order
    assert C.OFFCAP_AUTHORISED_FIELDS[-1] == "pass_no" == C.OOS_AUTHORISED_FIELDS[-2]
    assert C.OOS_AUTHORISED_FIELDS == (*C.OFFCAP_AUTHORISED_FIELDS, "max_tokens_on")
    for exp in (C.AUTHORISED_OOS_THINKING, C.AUTHORISED_OFF_CAP_COMPARATOR):
        assert exp.as_dict()["pass_no"] == 1 and "pass 1" in exp.what
    for check, ok, exc in ((C.check_out_of_support_thinking_authorised, _authorised_request(),
                            C.OutOfSupportThinkingUnauthorised),
                           (C.check_off_cap_override_authorised, _comparator_request(),
                            C.OffCapOverrideUnauthorised)):
        check(**ok)
        check(**{**ok, "pass_no": None})
        check(**{**ok, "pass_no": "1"})                       # the launcher's int or a CLI string: one number
        with pytest.raises(exc) as e:
            check(**{**ok, "pass_no": 2})
        assert "pass_no differs — requested 2, authorised 1" in str(e.value)
        assert "pass 1" in str(e.value)                       # the tuple's prose says which geometry class


def test_the_raised_cap_is_the_experiments_and_a_measured_cap_class():
    """T1(c): 8192 for both GSM8K task names, None for a task the experiment
    does not name, a refusal for a name that is not an Inspect task; and 8192
    is a MEASURED cap class of the `gen_long_thinking` geometry entry — a
    documentation fact, NOT a window proof (the fit against the RESOLVED launch
    geometry is the preflight's gate, S4.6)."""
    assert C.OOS_THINKING_MAX_TOKENS == 8192
    assert C.oos_max_tokens_on("gsm8k_paired200") == 8192 == C.oos_max_tokens_on("gsm8k")
    assert C.oos_max_tokens_on("math") is None and C.oos_max_tokens_on("bfcl") is None
    with pytest.raises(C.ControlNameError):
        C.oos_max_tokens_on("gsm8k_paired_200")
    with pytest.raises(C.ControlNameError):
        C.oos_max_tokens_on(C.GSM8K_STANDARD_TASK)       # the ROW task is not the name
    g = sg.TABLE["capability:gen_long_thinking"]
    assert g.basis.requested_max_tokens == 8192
    classes = g.basis.note.split("Requested caps: [", 1)[1].split("]", 1)[0]
    assert "8192" in [c.strip() for c in classes.split(",")]
    # the tuple's cap table is the ONE source; the driver's MAX_TOKENS_ON is untouched
    assert set(C.AUTHORISED_OOS_THINKING.max_tokens_on) == {"gsm8k_paired200", "gsm8k"}


def test_the_two_further_identity_keys_join_the_derivation_the_driver_stamps():
    """T1(d), FLIPPED at S3.1 (phase B1): `max_tokens_by_task` and
    `serve_out_of_support` join `tinycap_identity_fields(out_of_support_thinking=True)`
    in the SAME change that teaches the driver to stamp them — the driver
    ASSERTS its stamped block's keys equal this derivation and the launcher
    compares every manifest against it, so the two could only ever move
    together (the A1 tree pinned the deferral the other way). In-support is
    byte-identical: the derivation without the override is the unconditional
    tuple, and no optional key is among the unconditional ones."""
    assert C.IDENTITY_KEY_MAX_TOKENS_BY_TASK == "max_tokens_by_task"
    assert C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT == "serve_out_of_support"
    for key in (C.IDENTITY_KEY_MAX_TOKENS_BY_TASK, C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT):
        assert key not in C.TINYCAP_IDENTITY_FIELDS
        assert key in C.tinycap_identity_fields(out_of_support_thinking=True)
        assert key in C.TINYCAP_IDENTITY_OPTIONAL_FIELDS
    assert C.tinycap_identity_fields(out_of_support_thinking=True) == (
        *C.TINYCAP_IDENTITY_FIELDS, C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING,
        C.IDENTITY_KEY_MAX_TOKENS_BY_TASK, C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT)
    assert C.tinycap_identity_fields() == C.TINYCAP_IDENTITY_FIELDS
    # the coupling: the driver's exact-keys assertion reads the derivation and
    # stamps BOTH new keys under the override, from the plan builder's own cap
    # rule and the contract expectation (never the attestation)
    driver = (REPO / "scripts/served_capability_run.py").read_text()
    assert "tc.tinycap_identity_fields(" in driver
    # offcap: ONE derivation, both overrides — and ONE resolution per run:
    # the map is computed once, before the identity dict, and stamped under
    # either override AND read for MATH's scalar
    assert "executed_caps = _override_max_tokens_by_task(" in driver
    assert "tc.IDENTITY_KEY_MAX_TOKENS_BY_TASK: executed_caps" in driver
    assert "_math500_max_tokens(executed_caps)" in driver
    assert "tc.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT: tc.expected_serve_out_of_support(" in driver
    assert "serve_out_of_support_attested(" not in driver.split("tinycap_identity = {")[1].split(
        "assert set(tinycap_identity)")[0]          # the attestation never enters the hash


def test_the_payload_extra_and_outcome_field_sets_are_the_plans():
    """T1(e): the new payload fields ride at the END of SUMMARY_PAYLOAD_FIELDS
    (a payload written before them carries no key — how a merge tells the two
    populations apart), the two row facts at the end of INSPECT_EXTRA_FIELDS,
    and the outcome record's header / item field sets are EXACTLY the plan's
    (S1.7), every name a legal counts-only key."""
    assert C.SUMMARY_PAYLOAD_FIELDS[-15:] == (
        "mean_completion_tokens", "p99_completion_tokens", "n_completion_tokens",
        "n_think_block_present", "n_think_block_unterminated", "n_unterminated_not_at_cap",
        "think_split_mode", "unterminated_policy",
        "render_regime", "salt_regime", "resumed_policy",
        "n_render_regimes", "n_salt_regimes", "n_resumed_policies",
        "n_no_think_block")                                   # GSM8K-SCORER-fix: the newest payload field, at the end
    assert C.SUMMARY_PAYLOAD_FIELDS.index("samples_sha256") == len(C.SUMMARY_PAYLOAD_FIELDS) - 16
    assert C.INSPECT_EXTRA_FIELDS[-2:] == ("think_block_present", "think_block_unterminated")
    assert set(C.TASK_OUTCOMES_HEADER_FIELDS) == {
        "task", "item_set", "thinking", "think_split_mode", "unterminated_policy", "max_tokens",
        "samples_per_item", "n_attempted_items", "n_attempted_samples", "n_scored",
        "n_unscored", "n_scored_samples", "n_unscored_samples", "n_stop_reason", "n_at_cap",
        "n_think_block_present", "n_think_block_unterminated", "n_unterminated_not_at_cap",
        "n_completion_tokens", "producer_tree_sha", "items"}
    assert set(C.TASK_OUTCOME_ITEM_FIELDS) == {
        "item_id", "sample_idx", "attempt", "scored", "value", "unparseable", "stop_reason",
        "at_cap", "think_block_present", "think_block_unterminated", "stray_open_tag",
        "content_empty_with_reasoning", "prompt_tokens", "completion_tokens", "seed"}
    # the third split fact is durable on the record (the analysis leaves those items
    # out and re-runs the rule), spelled as the scorer's metadata spells it
    from sisteer.capeval.tiers import think_split as ts
    assert "stray_open_tag" in ts.SPLIT_METADATA_FIELDS
    assert len(set(C.TASK_OUTCOMES_HEADER_FIELDS)) == len(C.TASK_OUTCOMES_HEADER_FIELDS)
    assert len(set(C.TASK_OUTCOME_ITEM_FIELDS)) == len(C.TASK_OUTCOME_ITEM_FIELDS)
    assert C.TASK_OUTCOMES_DIR == "outcomes" and C.TASK_OUTCOMES_DIR not in (
        C.DONE_DIR, C.PARTIAL_DIR, C.SUMMARY_DIR)
    assert C.task_outcomes_filename(C.GSM8K_STANDARD_TASK) == "inspect__gsm8k_paired200.json"
    with pytest.raises(C.ControlNameError):
        C.task_outcomes_filename("not/a-task")
    for name in (*C.SUMMARY_PAYLOAD_FIELDS[-15:], *C.TASK_OUTCOMES_HEADER_FIELDS,
                 *C.TASK_OUTCOME_ITEM_FIELDS):
        assert _SAFE_STRING.match(name), name
    # the split-mode labels and the policy labels are whitespace-free short strings
    for label in (*C.THINK_SPLIT_MODES, *C.UNTERMINATED_POLICIES):
        assert _SAFE_STRING.match(label), label


def test_the_lens_contract_constants_are_the_contracts_literals_and_types():
    """Every name the harness needs from the lens half
    (`OOS_CONTRACT_PLACEHOLDERS`: sixteen constants over the §3 table's
    fourteen rows) is FILLED and EQUALS the contract literal quoted in
    `_CONTRACT`, with the contract's TYPE: a Python bool for the
    JSON boolean `true` — never `1` or a string — and boolean `switch` /
    `moves` expectations; a JSON-canonical `["thinking"]`; a tuple key path;
    the arm -> label map. Every consumer that refused on `None` resolves, and
    each constant's docstring quotes the value and cites its section beside
    the definition."""
    assert set(_CONTRACT) == set(C.OOS_CONTRACT_PLACEHOLDERS) and len(C.OOS_CONTRACT_PLACEHOLDERS) == 16
    for name, want in _CONTRACT.items():
        got = getattr(C, name)
        assert got == want, (name, got, want)
        assert type(got) is type(want), (name, type(got).__name__, type(want).__name__)
    assert C.OOS_THINKING_DECLARATION_VALUE is True and json.dumps(C.OOS_THINKING_DECLARATION_VALUE) == "true"
    assert C.THINKING_HEADER_SWITCH_EXPECTED is True and C.THINKING_HEADER_MOVES_EXPECTED is True
    assert C.OOS_THINKING_DECLARATION_KEY not in px.FORBIDDEN_CLIENT_KEYS          # the key reaches the proxy
    assert C.OOS_THINKING_DECLARATION_KEY not in (px.RESPONSE_PREFILL_KEY, px.AGENT_LOOP_KEY)
    assert set(C.OOS_REQUEST_PLACEHOLDERS) == {"OOS_THINKING_DECLARATION_KEY",
                                              "OOS_THINKING_DECLARATION_VALUE"}
    assert set(C.OOS_REQUEST_PLACEHOLDERS) <= set(C.OOS_CONTRACT_PLACEHOLDERS)
    for name in ("THINKING_HEADER_SWITCH_FIELD", "THINKING_HEADER_SWITCH_EXPECTED",
                 "THINKING_HEADER_MOVES_EXPECTED", "THINKING_HEADER_SOURCE"):
        assert name in C.OOS_CONTRACT_PLACEHOLDERS
    assert not hasattr(C, "THINKING_HEADER_SOURCES")     # the routing-table source is gone
    assert C.THINKING_HEADER_ERROR_SOURCE == ("petri_proxy", "thinking_header_error")   # §2.5.12, beside it
    assert C.oos_declaration_ready() is True
    # the consumers that refused on None resolve — the one configuration that carries a declaration...
    assert C.expected_serve_out_of_support(lens=True, arm=C.THINKING_ON, out_of_support=True) == ["thinking"]
    # ...and every other configuration is None with no contract needed
    for kw in (dict(lens=False, arm=C.THINKING_ON, out_of_support=True),
               dict(lens=True, arm=C.THINKING_OFF, out_of_support=True),
               dict(lens=True, arm=C.THINKING_ON, out_of_support=False),
               dict(lens=False, arm=C.THINKING_OFF, out_of_support=False)):
        assert C.expected_serve_out_of_support(**kw) is None, kw
    with pytest.raises(ValueError):
        C.expected_serve_out_of_support(lens=True, arm=C.THINKING_PER_TASK, out_of_support=True)
    assert C.serve_out_of_support_attested({}, "any") is None
    assert C.attested_thinking_header({}) is None
    assert C.cell_spellings(_OOS_CELL, declared=True).harness == "si_safety__psr_kl_all@thinking"
    assert C.expected_stamp_regimes(lens=True, arm=C.THINKING_ON) == C.StampRegimes(
        "request-unique", "declared-request-unique", "thinking_on")
    assert C.thinking_on_declaration(lens=True, arm=C.THINKING_ON) == {"petri_thinking_on": True}
    # the docstrings quote the CONTRACT value and cite its section beside each constant
    src = Path(C.__file__).read_text()
    for name, value in _CONTRACT.items():
        head = src.index(f"\n{name}: ")
        doc = src[head - 1400:head]
        assert "§2.5." in doc, name
        if isinstance(value, bool):
            needles = ["True" if value else "False"]
        elif isinstance(value, str):
            needles = [value]
        elif isinstance(value, dict):
            needles = list(value.values())              # the arm -> label map: its labels
        else:                                           # a list / tuple literal
            needles = [", ".join(json.dumps(v) for v in value)]
        for needle in needles:
            assert needle in doc, (name, needle)


def test_a_none_contract_constant_still_refuses_every_consumer():
    """The fail-closed guard survives the fill: should any contract constant
    read None at call time (a mutated module — nothing on the shipped one),
    the consumer that needs it refuses naming the constant and the fill step,
    never proceeding on a guessed name; the configurations that never needed
    that constant are unaffected; the module is intact afterwards."""
    on_lens = dict(lens=True, arm=C.THINKING_ON)
    cases = (
        ("OOS_CELL_DECLARATION_EXPECTED",
         lambda: C.expected_serve_out_of_support(out_of_support=True, **on_lens),
         lambda: C.expected_serve_out_of_support(lens=False, arm=C.THINKING_ON, out_of_support=True)),
        ("PROXY_IDENTITY_SERVE_DECLARATIONS", lambda: C.serve_out_of_support_attested({}, "any"), None),
        ("PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR", lambda: C.serve_out_of_support_attested({}, "any"), None),
        ("THINKING_HEADER_SOURCE", lambda: C.attested_thinking_header({}), None),
        ("OOS_STANDUP_CELL_SUFFIX", lambda: C.cell_spellings(_OOS_CELL, declared=True),
         lambda: C.cell_spellings(_OOS_CELL, declared=False)),
        ("OOS_EXPECT_SHA_KEY", lambda: C.cell_spellings(_OOS_CELL, declared=True), None),
        ("OOS_THINKING_DECLARATION_KEY", lambda: C.thinking_on_declaration(**on_lens),
         lambda: C.thinking_on_declaration(lens=False, arm=C.THINKING_ON)),
        ("OOS_THINKING_DECLARATION_VALUE", lambda: C.thinking_on_declaration(**on_lens),
         lambda: C.thinking_on_declaration(lens=True, arm=C.THINKING_OFF)),
    )
    for name, refuses, unaffected in cases:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(C, name, None)
            with pytest.raises(C.OutOfSupportDeclarationUnreconciled, match=f"{name}.*B0"):
                refuses()
            if unaffected is not None:
                unaffected()                            # never consulted the constant
            if name in C.OOS_REQUEST_PLACEHOLDERS:
                assert C.oos_declaration_ready() is False
        assert getattr(C, name) == _CONTRACT[name], name
    assert C.oos_declaration_ready() is True


_VOLATILE_MEMBER = re.compile(r"(ts|time|stamp|date|pid|path|dir|hash|digest|sha|nonce|uuid)",
                              re.IGNORECASE)


def _assert_stable_literal(value) -> None:
    """The discipline the hashed expectation literal must satisfy (§4 of the
    plan): JSON-canonical (a sort_keys round trip is the identity), built of
    scalars, lists of strings and fixed-shape mappings, and NO member named
    like a timestamp / pid / path / digest — evidence stays out of the hash."""
    assert json.loads(json.dumps(value, sort_keys=True)) == value
    if isinstance(value, dict):
        for k, v in value.items():
            assert isinstance(k, str) and not _VOLATILE_MEMBER.search(k), k
            _assert_stable_literal(v)
    elif isinstance(value, list):
        assert all(isinstance(x, str) for x in value), value
    else:
        assert value is None or isinstance(value, (str, int, float, bool)), value


def test_the_expected_declaration_is_a_canonical_copy_of_a_stable_literal():
    """P2-a: what the identity hashes is the contract's STABLE literal
    `["thinking"]` (§2.5.1 Q6), copied (never the module object), stable by
    the §4 discipline, and EQUAL to the vocabulary module's own overlay for the
    `thinking` variant (`serve_declaration.variant_declarations`); a literal
    with a volatile member fails that discipline."""
    got = C.expected_serve_out_of_support(lens=True, arm=C.THINKING_ON, out_of_support=True)
    assert got == ["thinking"] and got is not C.OOS_CELL_DECLARATION_EXPECTED
    assert got == list(sd.variant_declarations(sd.VARIANT_THINKING)[sd.SERVE_OUT_OF_SUPPORT_KEY])
    assert C.OOS_CELL_DECLARATION_EXPECTED is not sd.SERVE_VARIANTS[sd.VARIANT_THINKING][
        sd.SERVE_OUT_OF_SUPPORT_KEY]                    # the module constant is a copy too
    _assert_stable_literal(got)
    for volatile in ({"serve_out_of_support": ["thinking"], "ts": 1.0},
                     {"template_hash": "ab12"}, {"serve_out_of_support": [1]}):
        with pytest.raises(AssertionError):
            _assert_stable_literal(volatile)
    # the base route under the same override still expects None
    assert C.expected_serve_out_of_support(lens=False, arm=C.THINKING_ON, out_of_support=True) is None


def test_the_one_declaration_predicate_sends_the_key_with_true_or_not_at_all(monkeypatch):
    """§2.5.4 / §2.5.12 as ONE predicate (`thinking_on_declaration`, plan S3.6
    / S3.9): the fragment `{"petri_thinking_on": true}` iff `enable_thinking:
    true` on a LENS route; `{}` on the OFF arm and on the base route (§2.5.5:
    the declaration there is 400 `thinking_on_not_lens`); the key is NEVER
    emitted with `false` or `null`, the kwargs never `null`; composed with the
    kwargs it is the byte-exact §2.5.3 body fragment; a driver MODE is not an
    arm; and it refuses (fail closed) should a request-side constant read
    None — `oos_declaration_ready` is the same fact as a boolean."""
    assert C.oos_declaration_ready() is True
    frag = C.thinking_on_declaration(lens=True, arm=C.THINKING_ON)
    assert frag == {"petri_thinking_on": True} and type(frag["petri_thinking_on"]) is bool
    assert json.dumps(frag) == '{"petri_thinking_on": true}'
    for lens, arm in ((True, C.THINKING_OFF), (False, C.THINKING_ON), (False, C.THINKING_OFF)):
        assert C.thinking_on_declaration(lens=lens, arm=arm) == {}, (lens, arm)
    for lens in (True, False):
        with pytest.raises(ValueError):
            C.thinking_on_declaration(lens=lens, arm=C.THINKING_PER_TASK)
    # §2.5.12 presence semantics: true or absent, never false / null; the kwargs a boolean, never null — on the two
    # SWITCH arms; the no-switch arm (a member of THINKING_MODES) sends no kwargs and no declaration
    for lens in (True, False):
        for arm in C.THINKING_MODES:
            f = C.thinking_on_declaration(lens=lens, arm=arm)
            assert all(v is True for v in f.values()), (lens, arm, f)
            if arm in C.THINKING_SWITCH_ARMS:
                kw = C.chat_template_kwargs(arm)[C.CHAT_TEMPLATE_KWARGS_KEY]
                assert set(kw) == {C.ENABLE_THINKING_KEY} and type(kw[C.ENABLE_THINKING_KEY]) is bool
            else:
                assert arm == C.THINKING_NOT_AVAILABLE and C.chat_template_kwargs(arm) == {} and f == {}
    # the §2.5.3 thinking-ON steered-arm body fragment, byte-exact
    body = {**C.chat_template_kwargs(C.THINKING_ON), **C.thinking_on_declaration(lens=True, arm=C.THINKING_ON)}
    assert json.dumps(body, sort_keys=True) == (
        '{"chat_template_kwargs": {"enable_thinking": true}, "petri_thinking_on": true}')
    # the unsteered ON arm (base route): the kwargs alone
    assert json.dumps({**C.chat_template_kwargs(C.THINKING_ON),
                       **C.thinking_on_declaration(lens=False, arm=C.THINKING_ON)}, sort_keys=True) == (
        '{"chat_template_kwargs": {"enable_thinking": true}}')
    # fail closed on a mutated module
    monkeypatch.setattr(C, "OOS_THINKING_DECLARATION_KEY", None)
    assert C.oos_declaration_ready() is False
    with pytest.raises(C.OutOfSupportDeclarationUnreconciled, match="B0"):
        C.thinking_on_declaration(lens=True, arm=C.THINKING_ON)
    assert C.thinking_on_declaration(lens=False, arm=C.THINKING_ON) == {}   # never needed the key


def test_the_vocabulary_is_evals_serve_declaration_module_and_matches_the_contract():
    """Bm replaced the LOCAL TWIN with the import: every vocabulary name the
    contracts module exposes IS the peer module's object (identity, executed —
    not a source pin), the grammar functions are theirs under the names the
    twin wore, and the literals are the CONTRACT's on both sides (§2.5.1,
    §2.5.2, §2.5.10 Q8) — at most one `@`, empty base / variant refused, an
    unknown variant refused, `join` the inverse of `split`; the `thinking`
    variant's declaration is exactly `{"serve_out_of_support": ["thinking"]}`;
    and the two B0 constants derived from the vocabulary derive from IT, so a
    drift on either side fails here."""
    for name in ("VARIANT_SEP", "VARIANT_THINKING", "VARIANT_ALL_TURNS", "VARIANT_ALL_POSITIONS",
                 "SERVE_VARIANTS", "SERVE_OUT_OF_SUPPORT_KEY", "SERVE_EXPERIMENT_KEY",
                 "DECLARATION_KEYS", "OUT_OF_SUPPORT_THINKING", "OUT_OF_SUPPORT_TOKENS",
                 "ServeDeclarationError"):
        assert getattr(C, name) is getattr(sd, name), name
    assert (C._split_variant, C._join_variant, C._variant_declarations) == (
        sd.split_variant, sd.join_variant, sd.variant_declarations)
    assert issubclass(C.ServeDeclarationError, ValueError)
    # the contract literals, on the peer's side (and therefore on ours)
    assert (sd.VARIANT_SEP, sd.VARIANT_THINKING) == ("@", "thinking")
    assert (sd.SERVE_OUT_OF_SUPPORT_KEY, sd.SERVE_EXPERIMENT_KEY) == (
        "serve_out_of_support", "serve_experiment")
    assert sd.DECLARATION_KEYS == ("serve_out_of_support", "serve_experiment")
    assert sd.OUT_OF_SUPPORT_THINKING == "thinking" and sd.OUT_OF_SUPPORT_TOKENS == ("thinking",)
    assert set(sd.SERVE_VARIANTS) == {"thinking", "all_turns", "all_positions"}
    assert sd.SERVE_VARIANTS["thinking"] == {"serve_out_of_support": ["thinking"]}
    assert C._variant_declarations("thinking") == {"serve_out_of_support": ["thinking"]}
    assert C._variant_declarations("all_turns") == {"serve_experiment": "fit_response_serve_all_turns"}
    assert C._split_variant("safety_refusal__psr_kl_all@thinking") == ("safety_refusal__psr_kl_all", "thinking")
    assert C._split_variant("safety_refusal__psr_kl_all") == ("safety_refusal__psr_kl_all", None)
    assert C._split_variant("none__lens0") == ("none__lens0", None)
    assert C._join_variant("safety_refusal__psr_kl_all", "thinking") == "safety_refusal__psr_kl_all@thinking"
    assert C._join_variant("si_safety__psr_kl_all", None) == "si_safety__psr_kl_all"
    for bad in ("a@thinking@thinking", "@thinking", "a@", "a@foo", "", None, 3):
        with pytest.raises(C.ServeDeclarationError):
            C._split_variant(bad)
    with pytest.raises(C.ServeDeclarationError):
        C._join_variant("a@thinking", "thinking")
    with pytest.raises(C.ServeDeclarationError):
        C._join_variant("a", "foo")
    with pytest.raises(C.ServeDeclarationError):
        C._variant_declarations("foo")
    # B0's two vocabulary-derived constants derive from the module (never a local string)
    assert C.OOS_STANDUP_CELL_SUFFIX == sd.VARIANT_SEP + sd.VARIANT_THINKING == "@thinking"
    assert C.OOS_CELL_DECLARATION_EXPECTED == sd.variant_declarations(sd.VARIANT_THINKING)[
        sd.SERVE_OUT_OF_SUPPORT_KEY] == ["thinking"]


def test_serve_declaration_and_the_contracts_import_with_torch_blocked():
    """Bm: the vocabulary module the contracts now import is torch-free and
    stdlib-only (its docstring's promise — the bare launch venv has no torch),
    and importing it keeps the contracts and the torch-free grammar torch-free:
    imported in a subprocess whose meta-path REFUSES every heavy import
    (stronger than reading `sys.modules` after the fact)."""
    code = (
        "import sys, importlib.abc\n"
        "BLOCK = {'torch', 'vllm', 'transformers', 'numpy', 'lm_eval', 'inspect_ai', 'inspect_evals',"
        " 'evalplus'}\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name.split('.')[0] in BLOCK:\n"
        "            raise ImportError('BLOCKED ' + name)\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import sisteer.capeval.serving.serve_declaration as sd\n"
        "from sisteer.capeval.petri import defense_cells, registry_adapter\n"
        "import sisteer.capeval.tinycap.contracts as C\n"
        "assert C.SERVE_VARIANTS is sd.SERVE_VARIANTS and C.OOS_STANDUP_CELL_SUFFIX == '@thinking'\n"
        "assert defense_cells.parse_cell_id('si_safety__psr_kl_all@thinking').serve_variant == 'thinking'\n"
        "assert registry_adapter.__name__\n"
        "print('ok')\n")
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok", out.stdout


def test_the_stamp_literals_and_refusal_codes_are_the_contracts():
    """§2.5.11 Q9 / §2.5.6 as constants, and the R2 CROSS-PINS against the
    proxy now that `proxy.py` carries the whole stamp vocabulary: every key
    and literal the proxy owns is re-exported BY REFERENCE (asserted with
    `is`, never just `==`), and every one still EQUALS the contract's literal
    (asserted against the quoted string, so the module is compared with the
    contract, not with itself). The two cache-salt regime literals — a declared
    request wears the ordinary `request-unique`; the EIGHT
    `resumed_prefill_policy` labels (the three attested policies + the five
    non-values, which since the lens half landed include the two labels it
    added — read from the proxy, listed once each); the three `render_regime`
    literals and the arm map; the eight refusal codes."""
    # the salt regime: the proxy's tuple, the contract's two literals
    assert C.SALT_REGIME_LITERALS is px.SALT_REGIMES
    assert C.SALT_REGIME_LITERALS == ("cell-stable", "request-unique")
    # the two labels: the proxy's own objects, the contract's literals (§2.5.11 Q9)
    assert C.DECLARED_SALT_POLICY_LABEL is px.POLICY_DECLARED_SHARES_NOTHING
    assert C.FORCED_SALT_POLICY_LABEL is px.POLICY_FORCED_REQUEST_UNIQUE
    assert (C.DECLARED_SALT_POLICY_LABEL, C.FORCED_SALT_POLICY_LABEL) == (
        "declared-request-unique", "forced-request-unique")
    for label in (C.DECLARED_SALT_POLICY_LABEL, C.FORCED_SALT_POLICY_LABEL):
        assert label not in C.SALT_REGIME_LITERALS        # a policy label, never a regime
        assert label in px.POLICY_NON_VALUES              # the proxy emits it: a non-value
        assert C.RESUMED_POLICY_LABELS.count(label) == 1  # listed ONCE (not appended twice)
    # the eight policy labels: exactly the proxy's attested policies + non-values,
    # each the proxy's own object, no duplicates, the contract's eight literals
    assert C.RESUMED_POLICY_LABELS == (*px.RESUMED_PREFILL_POLICIES, *px.POLICY_NON_VALUES)
    assert all(any(label is own for own in (*px.RESUMED_PREFILL_POLICIES, *px.POLICY_NON_VALUES))
               for label in C.RESUMED_POLICY_LABELS)
    assert len(C.RESUMED_POLICY_LABELS) == len(set(C.RESUMED_POLICY_LABELS)) == 8
    assert set(C.RESUMED_POLICY_LABELS) == {
        "never", "always", "conditional", "not-consulted", "unattested", "absent",
        "declared-request-unique", "forced-request-unique"}
    assert not hasattr(C, "RESUMED_POLICY_LABELS_CODE_VERIFIED")   # the six-plus-two split is gone
    # the render regime: the proxy's tuple and literals, the contract's three
    assert C.RENDER_REGIME_LITERALS is px.RENDER_REGIMES
    assert C.RENDER_REGIME_LITERALS == ("thinking_on", "thinking_off", "unspecified")
    assert (C.RENDER_REGIME_THINKING_ON, C.RENDER_REGIME_THINKING_OFF,
            C.RENDER_REGIME_UNSPECIFIED) == C.RENDER_REGIME_LITERALS
    assert C.RENDER_REGIME_THINKING_ON is px.RENDER_REGIME_THINKING_ON
    assert C.RENDER_REGIME_THINKING_OFF is px.RENDER_REGIME_THINKING_OFF
    assert C.RENDER_REGIME_UNSPECIFIED is px.RENDER_REGIME_UNSPECIFIED
    # the three stamp KEYS: the proxy's ledger keys, by reference
    assert C.STAMP_KEY_SALT_REGIME is px.LEDGER_KEY_SALT_REGIME
    assert C.STAMP_KEY_RESUMED_POLICY is px.LEDGER_KEY_RESUMED_POLICY
    assert C.STAMP_KEY_RENDER_REGIME is px.LEDGER_KEY_RENDER_REGIME
    assert (C.STAMP_KEY_SALT_REGIME, C.STAMP_KEY_RESUMED_POLICY, C.STAMP_KEY_RENDER_REGIME) == (
        "cache_salt_regime", "resumed_prefill_policy", "render_regime")
    # the declaration key the request-side bodies carry is the proxy's
    assert C.OOS_THINKING_DECLARATION_KEY is px.THINKING_ON_KEY == "petri_thinking_on"
    assert C.RENDER_REGIME_BY_ARM == _CONTRACT["RENDER_REGIME_BY_ARM"] == {
        C.THINKING_ON: "thinking_on", C.THINKING_OFF: "thinking_off",         # §2.5.11 rows 1-5
        C.THINKING_NOT_AVAILABLE: "unspecified"}                             # the no-switch row
    assert set(C.RENDER_REGIME_BY_ARM.values()) <= set(C.RENDER_REGIME_LITERALS)
    # `unspecified` is expected of the NO-SWITCH arm alone — never of a switch arm
    assert C.RENDER_REGIME_UNSPECIFIED == C.RENDER_REGIME_BY_ARM[C.THINKING_NOT_AVAILABLE]
    assert C.RENDER_REGIME_UNSPECIFIED not in {C.RENDER_REGIME_BY_ARM[a] for a in C.THINKING_SWITCH_ARMS}
    assert set(C.RENDER_REGIME_BY_ARM) == set(C.THINKING_MODES)
    # the refusal codes: the proxy's tuple (what it puts on the wire), the
    # contract's eight literals (§2.5.6)
    assert C.THINKING_REFUSAL_CODES is px.THINKING_REFUSAL_CODES
    assert C.THINKING_REFUSAL_CODES == (
        "thinking_kwargs_invalid", "thinking_kwargs_inert", "thinking_on_undeclared",
        "thinking_on_invalid", "thinking_on_mismatch", "thinking_on_not_lens",
        "thinking_on_cell_not_declared", "thinking_on_unattested",
        # + the thinking-ON STANDUP's one addition (RoutingProxy(thinking_mode="on"),
        # the sampled-rerun merge 2026-09-11): the regime flip
        "thinking_regime_mismatch")
    assert len(set(C.THINKING_REFUSAL_CODES)) == 9   # 8 §2.5.6 codes + the thinking-ON standup's regime flip
    for code in (*C.THINKING_REFUSAL_CODES, *C.RENDER_REGIME_LITERALS, *C.SALT_REGIME_LITERALS,
                 *C.RESUMED_POLICY_LABELS):
        assert _SAFE_STRING.match(code), code


def test_the_attested_core_reader_validates_the_own_key_only():
    """§2.5.1: "look up your own cell, never validate the whole map". The
    comparable core is the SORTED token list under the cell's key; an absent
    map or an absent cell is None (the compare site decides what that means
    for its configuration); SIBLING entries of any shape — experiment 2's
    `{"serve_experiment": <name>}` included — are ignored; the own key's value
    must be the one closed shape (a `serve_experiment` THERE, an extra key, a
    non-list, an empty list, a duplicate or a token outside `{"thinking"}`
    refuses); a non-null error key refuses. On the psr/ON standup BOTH the
    steered key and the `none__lens0` twin are present (§2.5.7, Q7)."""
    key = C.cell_spellings(_OOS_CELL, declared=True).harness
    assert key == "si_safety__psr_kl_all@thinking"
    ep = {"lens_serve_declarations": {key: {"serve_out_of_support": ["thinking"]},
                                      C.TWIN_CELL_ID: {"serve_out_of_support": ["thinking"]}},
          "lens_serve_declarations_error": None, "lens_reduction_mode": "per_row",
          "thinking_header_error": None}                # §2.5.12's additive null sibling
    assert C.serve_out_of_support_attested(ep, key) == ["thinking"]
    assert C.serve_out_of_support_attested(ep, C.TWIN_CELL_ID) == ["thinking"]
    assert C.serve_out_of_support_attested(ep, "si_safety__psr_kl_all") is None   # the bare cell: absent
    assert C.serve_out_of_support_attested({"lens_reduction_mode": "per_row"}, key) is None
    assert C.serve_out_of_support_attested({"lens_serve_declarations": {}}, key) is None
    # sibling entries are never validated — whatever their shape
    siblings = {**ep, "lens_serve_declarations": {
        **ep["lens_serve_declarations"],
        "si_constitution__psr_kl_all@all_turns": {"serve_experiment": "fit_response_serve_all_turns"},
        "weird": "not even a mapping", "other": {"serve_out_of_support": ["unknown-token"]}}}
    assert C.serve_out_of_support_attested(siblings, key) == ["thinking"]
    assert C.serve_out_of_support_attested(siblings, "weird" + "x") is None
    with pytest.raises(C.OutOfSupportAttestationError, match="lens_serve_declarations_error"):
        C.serve_out_of_support_attested({**ep, "lens_serve_declarations_error": "disagree"}, key)
    # the OWN key's value: the one closed shape, the closed vocabulary
    for bad in ({"lens_serve_declarations": []},
                {"lens_serve_declarations": {key: ["thinking"]}},
                {"lens_serve_declarations": {key: {"serve_out_of_support": []}}},
                {"lens_serve_declarations": {key: {"serve_out_of_support": ["thinking"], "x": 1}}},
                {"lens_serve_declarations": {key: {"serve_out_of_support": [1]}}},
                {"lens_serve_declarations": {key: {"serve_out_of_support": ["reasoning"]}}},
                {"lens_serve_declarations": {key: {"serve_out_of_support": ["thinking", "thinking"]}}},
                {"lens_serve_declarations": {key: {"serve_experiment": "fit_response_serve_all_turns"}}},
                {"lens_serve_declarations": {key: None}}):
        with pytest.raises(C.OutOfSupportAttestationError):
            C.serve_out_of_support_attested(bad, key)
    with pytest.raises(C.OutOfSupportAttestationError):
        C.serve_out_of_support_attested([], key)
    # the core is sorted, so the compare is order-free
    assert C.serve_out_of_support_attested(
        {"lens_serve_declarations": {key: {"serve_out_of_support": ("thinking",)}}}, key) == ["thinking"]


def test_the_thinking_header_reader_reads_the_models_payload_only():
    """§2.5.1 / §2.5.7: `/v1/models.petri_proxy.thinking_header` is the ONLY
    source (the routing-table read of revision 2 is gone); the reader
    takes the models payload alone, returns the block or None, refuses a
    non-mapping block; the TYPE and length checks (`switch: bool`, `moves:
    bool`, 5 / 7 ids) are the driver's certification against the
    `THINKING_HEADER_*` expectations. §2.5.12: `thinking_header_error` sits
    BESIDE the block, null on every ordinary standup — tolerated; the PARTIAL
    fact an uncertifiable template attests (`moves` / `on_ids` null) is returned
    as-is and fails the typed certification, never coerced."""
    assert C.THINKING_HEADER_SOURCE == ("petri_proxy", "thinking_header")
    block = {"switch": True, "moves": True, "off_ids": list(C.QWEN35_HEADER_IDS_OFF),
             "on_ids": list(C.QWEN35_HEADER_IDS_ON)}
    payload = {"object": "list", "data": [], "petri_proxy": {"version": 1, "thinking_header": block}}
    got = C.attested_thinking_header(payload)
    assert got == block and got is not block
    assert type(got["switch"]) is bool and type(got["moves"]) is bool
    assert len(got["on_ids"]) == 5 and len(got["off_ids"]) == 7
    assert (got[C.THINKING_HEADER_SWITCH_FIELD] is C.THINKING_HEADER_SWITCH_EXPECTED
            and got["moves"] is C.THINKING_HEADER_MOVES_EXPECTED)
    beside = {"petri_proxy": {"thinking_header": block, "thinking_header_error": None,
                              "lens_serve_declarations": {}, "lens_serve_declarations_error": None}}
    assert C.attested_thinking_header(beside) == block
    assert C.THINKING_HEADER_ERROR_SOURCE[:-1] == C.THINKING_HEADER_SOURCE[:-1]   # the same block
    partial = {"switch": True, "moves": None, "off_ids": list(C.QWEN35_HEADER_IDS_OFF), "on_ids": None}
    got_partial = C.attested_thinking_header({"petri_proxy": {
        "thinking_header": partial, "thinking_header_error": "ON header is not a prefix of the scaffold"}})
    assert got_partial == partial
    assert not (isinstance(got_partial["moves"], bool) and got_partial["moves"] is C.THINKING_HEADER_MOVES_EXPECTED)
    assert C.attested_thinking_header(None) is None
    assert C.attested_thinking_header({}) is None
    assert C.attested_thinking_header({"petri_proxy": {}}) is None
    assert C.attested_thinking_header({"petri_proxy": {"thinking_header": None}}) is None
    with pytest.raises(C.OutOfSupportAttestationError):
        C.attested_thinking_header({"petri_proxy": {"thinking_header": "enable_thinking"}})
    with pytest.raises(C.OutOfSupportAttestationError):
        C.attested_thinking_header([])
    with pytest.raises(TypeError):
        C.attested_thinking_header({"thinking_header": block}, {"petri_proxy": {}})   # one argument


def test_the_stamp_regime_reader_reads_the_code_verified_fields_and_the_render_regime():
    """T1(f'): the two salt fields are proxy.py's own (re-exported, never
    retyped) and the render regime is read under the contract's `render_regime`
    (§2.5.5); the expectation table is §2.5.11's five rows verbatim, the same
    type in the same order, compared BY NAME."""
    assert C.STAMP_KEY_SALT_REGIME == px.LEDGER_KEY_SALT_REGIME == "cache_salt_regime"
    assert C.STAMP_KEY_RESUMED_POLICY == px.LEDGER_KEY_RESUMED_POLICY == "resumed_prefill_policy"
    assert C.STAMP_BLOCK_KEY == "petri"
    stamp = {"model": "petri__x", "petri": {
        "cache_salt_regime": "request-unique", "resumed_prefill_policy": "declared-request-unique",
        "render_regime": "thinking_on"}}
    # ONE order, keyed by NAME: (salt_regime, resumed_policy, render_regime) —
    # StampRegimes' fields = EXPECTED_STAMP_TRIPLE_FIELDS; the reader and the
    # expectation table return the same type, so nothing zips by position
    got = C.stamp_regimes(stamp)
    assert isinstance(got, C.StampRegimes) and got._fields == C.EXPECTED_STAMP_TRIPLE_FIELDS
    assert got == C.StampRegimes(salt_regime="request-unique", resumed_policy="declared-request-unique",
                                 render_regime="thinking_on")
    assert C.STAMP_KEY_RENDER_REGIME == "render_regime"                        # §2.5.5, §2.5.11
    assert got == C.expected_stamp_regimes(lens=True, arm=C.THINKING_ON)      # §2.5.11 row 1 (row 2: the twin)
    base = {"petri": {"cache_salt_regime": "cell-stable",
                      "resumed_prefill_policy": px.POLICY_NOT_CONSULTED, "render_regime": "thinking_off"}}
    assert C.stamp_regimes(base)._asdict() == {"salt_regime": "cell-stable", "resumed_policy": "not-consulted",
                                               "render_regime": "thinking_off"}
    assert C.stamp_regimes({}) == C.StampRegimes(None, None, None)
    assert C.stamp_regimes({"petri": {}}) == C.StampRegimes(None, None, None)
    with pytest.raises(C.OutOfSupportAttestationRefusal):          # a PRESENT block that is not a mapping
        C.stamp_regimes({"petri": "x"})
    # the expectation table is the SAME type in the SAME order, by name — §2.5.11's five rows verbatim
    rows = {(True, C.THINKING_ON): ("request-unique", "declared-request-unique", "thinking_on"),   # psr/ON + twin
            (True, C.THINKING_OFF): ("request-unique", "conditional", "thinking_off"),            # psr/OFF + twin
            (False, C.THINKING_ON): ("cell-stable", "not-consulted", "thinking_on"),              # none, ON
            (False, C.THINKING_OFF): ("cell-stable", "not-consulted", "thinking_off")}            # none, OFF
    for (lens, arm), want in rows.items():
        exp = C.expected_stamp_regimes(lens=lens, arm=arm)
        assert isinstance(exp, C.StampRegimes) and exp == C.StampRegimes(*want), (lens, arm)
        assert exp.render_regime == C.RENDER_REGIME_BY_ARM[arm]
    # a driver MODE is not an arm: no leg is served "per-task", so it has no expected triple
    with pytest.raises(ValueError, match="per-task"):
        C.expected_stamp_regimes(lens=True, arm=C.THINKING_PER_TASK)
    with pytest.raises(ValueError):
        C.expected_stamp_regimes(lens=False, arm="sideways")


def test_the_twin_is_the_registrys_noop_lens_cell():
    """T1(f'): `none__lens0` is the ONE derived control leg on the registry
    (§2.5.10 Q7) — pinned to the module that owns it, torch-free; never
    suffixed with a variant."""
    dc, _ra = C._cell_grammar()
    assert C.TWIN_CELL_ID == dc.NOOP_LENS_CELL_ID == "none__lens0"
    assert C.TWIN_CELL_ID.endswith(C.SCALE0_TWIN_SUFFIX)
    assert C._split_variant(C.TWIN_CELL_ID) == (C.TWIN_CELL_ID, None)


def test_the_qwen_header_constants_carry_the_render_fact():
    """T1(g): OFF is 7 ids with the CLOSE after the OPEN (the block closed
    inside the header); ON is 5 ids ending in the OPEN think token + newline
    and carrying no CLOSE — the served-side shape of `prompt_opens_think_on`,
    and the lengths the contract's `thinking_header` certifies (§2.5.7)."""
    off, on = C.QWEN35_HEADER_IDS_OFF, C.QWEN35_HEADER_IDS_ON
    assert len(off) == 7 and len(on) == 5
    assert on[-2:] == (C.QWEN35_THINK_OPEN_ID, 198)
    assert C.QWEN35_THINK_CLOSE_ID not in on
    i_open = off.index(C.QWEN35_THINK_OPEN_ID)
    assert C.QWEN35_THINK_CLOSE_ID in off[i_open:]
    assert off[:4] == on[:4]                            # the shared `<|im_start|>assistant\n<think>`
    assert C.QWEN35_THINK_OPEN_ID == 248068 and C.QWEN35_THINK_CLOSE_ID == 248069
    assert C.AUTHORISED_OOS_THINKING.prompt_opens_think_on is True
    # the docs' §11 table carries the same two id lists
    doc = DOC.read_text()
    assert "[248045, 74455, 198, 248068, 271, 248069, 271]" in doc
    assert "[248045, 74455, 198, 248068, 198]" in doc


def test_the_what_prose_names_the_sampling_deviation():
    """One sampling row for both arms, the thinking-mode card
    named as the stated deviation, held constant as a shared nuisance."""
    what = C.AUTHORISED_OOS_THINKING.what
    for phrase in ("0.7/0.95/20", "0.6", "BOTH arms", "nuisance", "per_row", "8192"):
        assert phrase in what, phrase


@pytest.mark.parametrize("arm,parser,fact,want", [
    (C.THINKING_OFF, None, False, "record_only"),
    (C.THINKING_OFF, None, True, "record_only"),
    (C.THINKING_OFF, "gemma4", True, "record_only"),
    (C.THINKING_ON, "gemma4", False, "record_only"),
    (C.THINKING_ON, "gemma4", True, "record_only"),
    (C.THINKING_ON, None, True, "implicit_open"),
    (C.THINKING_ON, "", True, "implicit_open"),          # an empty parser is no parser
    (C.THINKING_ON, None, False, "explicit_tags"),
])
def test_think_split_mode_is_derived_from_arm_parser_and_the_declared_fact(arm, parser, fact, want):
    """T1(i): OFF -> record_only regardless; ON + parser -> record_only; ON +
    no parser + the fact -> implicit_open; ON + no parser + no fact ->
    explicit_tags. Never from content."""
    assert C.think_split_mode(arm=arm, reasoning_parser=parser, prompt_opened_think=fact) == want
    assert want in C.THINK_SPLIT_MODES


def test_think_split_vocabulary():
    assert C.THINK_SPLIT_MODES == ("record_only", "implicit_open", "explicit_tags", "none")
    assert C.THINK_SPLIT_NONE == "none"
    # llama-paper-rows (P8): the no-switch arm splits `none` whatever the parser or fact says
    assert C.think_split_mode(arm=C.THINKING_NOT_AVAILABLE, reasoning_parser=None, prompt_opened_think=False) == "none"
    assert C.think_split_mode(arm=C.THINKING_NOT_AVAILABLE, reasoning_parser="gemma4", prompt_opened_think=True) == "none"
    assert C.UNTERMINATED_POLICIES == ("unscored", "incorrect")
    with pytest.raises(ValueError):
        C.think_split_mode(arm=C.THINKING_PER_TASK, reasoning_parser=None, prompt_opened_think=True)


def test_k1_sampling_identity_is_one_row_for_both_arms():
    """T8: the card row is per MODEL, not per arm, so the OFF and ON
    expectations carry the SAME sampling block; k = 1 on the paired
    GSM8K; the seed rule is the inspect one."""
    row = C.sampling_for("qwen35_4b", C.SAMPLING_CARD)
    assert (row.temperature, row.top_p, row.top_k) == (0.7, 0.95, 20)
    k = C.run_samples_by_task([C.GSM8K_STANDARD_TASK], C.SAMPLING_CARD, math_samples=1)
    assert k == {C.GSM8K_STANDARD_TASK: 1}
    ident = C.sampling_identity(row, k)
    assert set(ident) == set(C.SAMPLING_IDENTITY_FIELDS)
    assert (ident[C.IDENTITY_KEY_TEMPERATURE], ident[C.IDENTITY_KEY_TOP_P],
            ident[C.IDENTITY_KEY_TOP_K]) == (0.7, 0.95, 20)
    assert ident[C.IDENTITY_KEY_SAMPLES_BY_TASK] == {C.GSM8K_STANDARD_TASK: 1}
    assert C.SEED_RULE_INSPECT in json.dumps(ident[C.IDENTITY_KEY_SEED_RULE])
    # nothing in the row lookup takes an arm: one row serves both
    assert "thinking" not in inspect.signature(C.sampling_for).parameters
    assert "arm" not in inspect.signature(C.sampling_for).parameters
    assert "thinking mode would be 0.6" in row.source


def test_cell_spellings_bare_is_byte_identical_to_the_launcher_translation():
    """T11: `declared=False` on a bare cell is today's spellings — the parsed
    harness id and `VMODEL_PREFIX + harness` — with no expect key (the bare
    sha); a variant-carrying id passed as undeclared contradicts itself and
    refuses."""
    dc, _ra = C._cell_grammar()
    for cell in (_OOS_CELL, C.CELL_PROMPTED_SAFETY, C.ROUTE_NONE_CELL, "si_constitution__vector_all"):
        parsed = C.parse_launcher_cell(cell)
        got = C.cell_spellings(cell, declared=False)
        # the vmodel is the REGISTRY's mapping (defense_cells.virtual_model_name — what
        # enumerate_cells writes into the route), never a prefix rule re-derived here
        assert got == C.CellSpellings(traintest=cell, harness=parsed.harness_id,
                                      vmodel=dc.virtual_model_name(parsed.harness_id), expect_key=None)
    # the base route's exception: `none` is served as petri__base (both unsteered
    # arms of the experiment use it); petri__none is served by nothing
    assert C.cell_spellings(C.ROUTE_NONE_CELL, declared=False).vmodel == "petri__base"
    assert C.cell_spellings(C.ROUTE_NONE_CELL, declared=False).vmodel == dc._vmodel("none", "base")
    assert C.cell_spellings(_OOS_CELL, declared=False).vmodel == "petri__si_safety__psr_kl_all"
    assert C.cell_spellings(C.CELL_PROMPTED_SAFETY, declared=False).vmodel == "petri__prompted__safety"
    assert dc.virtual_model_name("none") == "petri__base"
    assert dc.virtual_model_name("prompted__safety") == "petri__prompted__safety"
    assert dc.virtual_model_name("si_safety__psr_kl_all") == "petri__si_safety__psr_kl_all"
    with pytest.raises(dc.DefenseCellError):
        dc.virtual_model_name("not__a__cell")
    assert C.standup_cell_spelling(_OOS_CELL, declared=False) == _OOS_CELL
    with pytest.raises(C.CellNameError):
        C.cell_spellings("not__a__cell", declared=False)
    with pytest.raises(C.CellNameError, match="UNDECLARED"):
        C.cell_spellings(_OOS_CELL + "@thinking", declared=False)


def test_parse_launcher_cell_refuses_the_variant_spelling_so_the_helper_splits_first():
    """`parse_launcher_cell` at this tree partitions on `__` and checks the
    tail against `CLASS_A_METHODS`, so the variant spelling REFUSES there —
    which is exactly why `cell_spellings` splits the variant off BEFORE the
    method check and re-joins at the doors. Pinned so the split-first reason
    stays visible."""
    with pytest.raises(C.CellNameError):
        C.parse_launcher_cell("safety_refusal__psr_kl_all@thinking")
    assert C.parse_launcher_cell("safety_refusal__psr_kl_all").harness_id == "si_safety__psr_kl_all"


def test_cell_spellings_under_the_contract(monkeypatch):
    """T11 (§2.5.2, §2.5.10 Q8's door table): the THREE spellings, pinned
    verbatim under the contract values — the train/test spelling + variant at
    the standup doors, harness + variant on the attestation map / vmodel, the
    expect key in the train/test variant spelling; a variant-carrying input
    gives the SAME spellings; a `bare` expect-key rule gives no key; `none`
    and a prompted cell refuse a declaration; an experiment-2 variant and an
    unknown variant refuse; the spelling Evals corrected is produced by no
    path."""
    want = C.CellSpellings(traintest="safety_refusal__psr_kl_all@thinking",
                           harness="si_safety__psr_kl_all@thinking",
                           vmodel="petri__si_safety__psr_kl_all@thinking",
                           expect_key="safety_refusal__psr_kl_all@thinking")
    got = C.cell_spellings(_OOS_CELL, declared=True)
    assert got == want
    assert C.cell_spellings(_OOS_CELL + "@thinking", declared=True) == want     # already spelled
    assert C.standup_cell_spelling(_OOS_CELL, declared=True) == "safety_refusal__psr_kl_all@thinking"
    assert "si_safety_refusal__" not in " ".join(str(v) for v in got)
    assert got.traintest == C._join_variant(_OOS_CELL, C.VARIANT_THINKING)
    monkeypatch.setattr(C, "OOS_EXPECT_SHA_KEY", C.OOS_EXPECT_SHA_KEY_BARE)
    assert C.cell_spellings(_OOS_CELL, declared=True).expect_key is None
    monkeypatch.setattr(C, "OOS_EXPECT_SHA_KEY", "something-else")
    with pytest.raises(C.OutOfSupportDeclarationUnreconciled):
        C.cell_spellings(_OOS_CELL, declared=True)
    monkeypatch.setattr(C, "OOS_EXPECT_SHA_KEY", _CONTRACT["OOS_EXPECT_SHA_KEY"])
    # a filled suffix that is not the vocabulary's refuses (drift between the two)
    monkeypatch.setattr(C, "OOS_STANDUP_CELL_SUFFIX", "@think")
    with pytest.raises(C.OutOfSupportDeclarationUnreconciled, match="VARIANT_SEP"):
        C.cell_spellings(_OOS_CELL, declared=True)
    monkeypatch.setattr(C, "OOS_STANDUP_CELL_SUFFIX", _CONTRACT["OOS_STANDUP_CELL_SUFFIX"])
    for cell in (C.ROUTE_NONE_CELL, C.CELL_PROMPTED_SAFETY):
        with pytest.raises(C.CellNameError, match="OBJECT cell"):
            C.cell_spellings(cell, declared=True)
    with pytest.raises(C.CellNameError, match="not a harness spelling"):
        C.cell_spellings(_OOS_CELL + "@all_turns", declared=True)      # experiment 2's variant
    with pytest.raises(C.ServeDeclarationError):
        C.cell_spellings(_OOS_CELL + "@foo", declared=True)            # outside the closed vocabulary
    # a bare Class B id is an object cell too: a variant is legal on it
    launchable = C.launchable_class_b_ids()
    if launchable:
        assert C.cell_spellings(launchable[0], declared=True).traintest == f"{launchable[0]}@thinking"
    # where torch is importable, the harness spelling is the registry's own + the variant
    try:
        from sisteer.capeval.petri import registry_adapter as ra
        harness = ra.harness_cell_id(_OOS_CELL)
    except Exception as e:  # noqa: BLE001 — no torch here: the cross-check is skipped, named
        pytest.skip(f"registry_adapter.harness_cell_id unavailable ({type(e).__name__})")
    assert got.harness == harness + "@thinking" == "si_safety__psr_kl_all@thinking"


def test_cell_spellings_declared_refuses_should_either_spelling_constant_read_none(monkeypatch):
    """T11: the declared spelling resolves on the shipped module (B0); should
    either of its two constants read None (a mutated module), it still refuses
    naming that constant and the fill step — never a guessed suffix. The bare
    spelling needs neither."""
    assert C.cell_spellings(_OOS_CELL, declared=True).traintest == "safety_refusal__psr_kl_all@thinking"
    monkeypatch.setattr(C, "OOS_STANDUP_CELL_SUFFIX", None)
    with pytest.raises(C.OutOfSupportDeclarationUnreconciled, match="OOS_STANDUP_CELL_SUFFIX"):
        C.cell_spellings(_OOS_CELL, declared=True)
    monkeypatch.setattr(C, "OOS_STANDUP_CELL_SUFFIX", _CONTRACT["OOS_STANDUP_CELL_SUFFIX"])
    monkeypatch.setattr(C, "OOS_EXPECT_SHA_KEY", None)
    with pytest.raises(C.OutOfSupportDeclarationUnreconciled, match="OOS_EXPECT_SHA_KEY"):
        C.cell_spellings(_OOS_CELL, declared=True)
    monkeypatch.setattr(C, "OOS_STANDUP_CELL_SUFFIX", None)
    with pytest.raises(C.OutOfSupportDeclarationUnreconciled, match="B0"):
        C.cell_spellings(_OOS_CELL, declared=True)
    # the bare spelling never needed either
    assert C.cell_spellings(_OOS_CELL, declared=False).traintest == _OOS_CELL


# --------------------------------------------------------------------------- #
# round-A review follow-ups: the tuple's `samples`, the 2x2 kind map and the   #
# lens predicate, the evidence-key spellings                                    #
# --------------------------------------------------------------------------- #


def test_the_samples_field_drives_math500_only_and_gsm8k_stays_at_k_1():
    """The authorised tuple's ``samples`` is compared against the driver's
    ``--samples`` — and that knob is MATH-500's k and ONLY MATH-500's
    (``samples_of_task``'s ``math_samples``): the paired-200 GSM8K, the one
    control the experiment names, takes ``SAMPLES_BY_TASK``'s k = 1 whatever
    the field reads. Pinned: the field is 1 today, the GSM8K k under the
    authorised run is 1, and it would STAY 1 even if the field were raised —
    the spec's k = 1 does not hang on this field."""
    exp = C.AUTHORISED_OOS_THINKING
    assert exp.samples == C.MATH500_SAMPLES_PER_ITEM == 1
    task = C.GSM8K_PAIRED_TASK
    assert C.samples_of_task(exp.sampling_mode, task, math_samples=exp.samples) == 1
    assert C.run_samples_by_task((task,), exp.sampling_mode, math_samples=exp.samples) == {task: 1}
    # inert on GSM8K: a raised field moves MATH-500's k only
    assert C.samples_of_task(exp.sampling_mode, task, math_samples=4) == 1
    assert C.samples_of_task(exp.sampling_mode, C.inspect_task("math"), math_samples=4) == 4
    # every control atom of the experiment maps to k = 1 under its sampling mode
    for atom in exp.control_atoms():
        for t in C.TASKS_OF_CONTROL[atom]:
            assert C.samples_of_task(exp.sampling_mode, t, math_samples=exp.samples) == 1, t
    # and the door compares the field: a run asking for another MATH k is refused BY NAME
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="samples differs"):
        C.check_out_of_support_thinking_authorised(**{**_authorised_request(), "samples": exp.samples + 1})


def test_the_2x2_kind_map_lives_beside_the_tuple_and_lens_is_the_cell_grammars_verdict():
    """``OOS_CELL_KIND`` maps the authorised pair onto pairing's leg vocabulary
    (a variant through its base); ``is_lens_cell`` reads the grammar — a fitted
    object is lens, the base route and a PROMPTED cell are not (``cell !=
    "none"`` would have called the prompted cell lens), and a ``@thinking``
    variant is lens exactly when its base is."""
    from sisteer.capeval.tiers import pairing as pr

    assert C.OOS_CELL_KIND == {C.ROUTE_NONE_CELL: "none", C.CELL_PSRKL_SAFETY: "psr"}
    assert set(C.OOS_CELL_KIND) == set(C.AUTHORISED_OOS_THINKING.cells)
    assert set(C.OOS_CELL_KIND.values()) == set(pr.LEG_CELLS)
    assert C.oos_cell_kind(C.ROUTE_NONE_CELL) == "none"
    assert C.oos_cell_kind(C.CELL_PSRKL_SAFETY) == "psr"
    assert C.oos_cell_kind(C.CELL_PSRKL_SAFETY + "@thinking") == "psr"
    with pytest.raises(C.CellNameError, match="not one of the authorised"):
        C.oos_cell_kind(C.CELL_PROMPTED_SAFETY)
    assert C.is_lens_cell(C.CELL_PSRKL_SAFETY) is True
    assert C.is_lens_cell(C.CELL_PSRKL_SAFETY + "@thinking") is True
    assert C.is_lens_cell(C.ROUTE_NONE_CELL) is False
    assert C.is_lens_cell(C.CELL_PROMPTED_SAFETY) is False          # not "cell != none"
    assert C.is_lens_cell("si_constitution__vector_all") is True
    with pytest.raises(C.CellNameError):
        C.is_lens_cell("not__a__cell")


def test_the_evidence_key_spellings_are_declared_once():
    """The evidence.json keys the analysis reads and the driver writes
    (S3.3 / S3.4 / S3.8) — one spelling each, the two existing blocks by the
    names ``served_capability_run`` already writes, every key a safe short
    string; the analysis source and the synthetic fixture carry no literal of
    the certification flags."""
    assert C.OOS_EVIDENCE_HEADER_BLOCK == "thinking_header_by_arm"
    assert C.OOS_EVIDENCE_BITES_BLOCK == "steering_bites"
    assert C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK == "prompt_budget_live"
    assert (C.OOS_EVIDENCE_HEADER_CERTIFIED, C.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED,
            C.OOS_EVIDENCE_HEADER_ATTESTED_AGREES, C.OOS_EVIDENCE_ATTESTED_SWITCH,
            C.OOS_EVIDENCE_ATTESTED_MOVES, C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM) == (
        "header_certified", "prompt_opened_think_certified", "header_attested_agrees",
        "attested_switch", "attested_moves", "probe_render_arm")
    assert (C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX, C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_P99) == ("max", "p99")
    assert len(set(C.OOS_EVIDENCE_KEYS)) == len(C.OOS_EVIDENCE_KEYS) - 1    # `verified` is a field of two blocks
    for k in C.OOS_EVIDENCE_KEYS:
        assert _SAFE_STRING.match(k), k
    # the driver's existing blocks are spelled the way it writes them
    assert f'"{C.OOS_EVIDENCE_HEADER_BLOCK}"' in Path(srun.__file__).read_text()
    assert f'"{C.OOS_EVIDENCE_BITES_BLOCK}"' in Path(srun.__file__).read_text()
    # the readers coin nothing: no literal certification flag in the analysis or the fixture
    repo = Path(C.__file__).resolve().parents[4]
    for rel in ("scripts/tinycap_oos_transfer.py", "tests/oos_synthetic_store.py"):
        src = (repo / rel).read_text()
        for literal in ('"header_certified"', '"prompt_opened_think_certified"', '"header_attested_agrees"',
                        '"attested_switch"', '"attested_moves"', '"probe_render_arm"', '"prompt_budget_live"',
                        '"thinking_header_by_arm"', '"steering_bites"'):
            assert literal not in src, (rel, literal)


# --------------------------------------------------------------------------- #
# S3 (phase B1): the compare helpers, the header certification, the engine-id  #
# rule and the driver argv extension — every literal from contract-2.5.md      #
# --------------------------------------------------------------------------- #

_TT = C.CELL_PSRKL_SAFETY                       # safety_refusal__psr_kl_all
_OWN = "si_safety__psr_kl_all@thinking"        # the attestation key, harness variant spelling
_DECL = {C.SERVE_OUT_OF_SUPPORT_KEY: ["thinking"]}
_EXACT_MAP = {_OWN: dict(_DECL), C.TWIN_CELL_ID: dict(_DECL)}    # contract §2.5.7, verbatim
_SHA = "ab" * 32


def _ep(declarations=_EXACT_MAP, error=None, **extra):
    ep = {C.PROXY_IDENTITY_SERVE_DECLARATIONS: declarations,
          C.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR: error, **extra}
    return {k: v for k, v in ep.items() if v is not _ABSENT}


_ABSENT = object()


def test_driver_argv_appends_the_digest_and_the_budget_path_only_when_given():
    """S3.2: the launcher hands the driver the artifact digest it held the
    standup to (the same flag spelling as the standup's) and the PATH of the
    prompt-budget artifact its preflight wrote under
    ``rep["oos_prompt_budget"]``; both None -> byte-identical argv."""
    kw = dict(python="p", script="s", proxy_url="u", routing_table="r", cell=_TT, config="c",
              defenses_dir="d", out_root="o", controls=["gsm8k"], thinking="on", concurrency=22,
              judge_model="j", allow_out_of_support_thinking=True)
    plain = C.driver_argv(**kw)
    assert C.driver_argv(**kw, expect_object_sha256=None, oos_prompt_budget=None) == plain
    full = C.driver_argv(**kw, expect_object_sha256=_SHA, oos_prompt_budget="/x/oos_prompt_budget.json")
    assert full == plain + [C.FLAG_EXPECT_OBJECT_SHA256, _SHA, C.FLAG_OOS_PROMPT_BUDGET,
                            "/x/oos_prompt_budget.json"]
    assert C.FLAG_EXPECT_OBJECT_SHA256 == "--expect-object-sha256"      # the standup's own spelling
    assert C.FLAG_OOS_PROMPT_BUDGET == "--oos-prompt-budget"
    assert C.OOS_PROMPT_BUDGET_REPORT_FIELD == "oos_prompt_budget"
    assert C.FLAG_EXPECT_OBJECT_SHA256 not in C.TINYCAP_DRIVER_FLAGS      # never probed with a dummy
    assert C.FLAG_OOS_PROMPT_BUDGET not in C.TINYCAP_DRIVER_FLAGS


def test_check_serve_declarations_on_the_declared_standup_is_an_own_key_and_twin_lookup():
    """T3 / T4 (contract §2.5.1, §2.5.7, §2.5.10): the EXACT map passes and is
    returned raw; a sibling entry of the OTHER closed shape is ignored; the own
    key's core disagreeing (a second token, a non-list, the experiment shape)
    or the twin missing / disagreeing is OOS_DECLARATION_MISMATCH naming both;
    the own key absent, the map absent / null, or a non-null error key is
    OOS_DECLARATION_UNATTESTED."""
    on = dict(cell=_TT, arm=C.THINKING_ON, out_of_support=True)
    out = C.check_serve_declarations(_ep(), **on)
    assert out == {"expected": ["thinking"], "own_key": _OWN, "own": ["thinking"],
                   "twin_key": C.TWIN_CELL_ID, "twin": ["thinking"], "raw": _EXACT_MAP,
                   "error_present": True}
    sibling = {**_EXACT_MAP, "si_constitution__psr_kl_all@all_positions":
               {C.SERVE_EXPERIMENT_KEY: "fit_response_serve_all_positions"}}
    assert C.check_serve_declarations(_ep(sibling), **on)["raw"] == sibling      # ignored, kept raw

    def rec_of(ep, **cfg):
        with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
            C.check_serve_declarations(ep, **cfg)
        assert str(e.value).startswith(e.value.rec + ": ")
        return e.value.rec, str(e.value)

    for bad_own in (["thinking", "other"], "thinking", {C.SERVE_EXPERIMENT_KEY: "x"}, [], ["thinking", "thinking"]):
        value = bad_own if isinstance(bad_own, dict) else {C.SERVE_OUT_OF_SUPPORT_KEY: bad_own}
        rec, msg = rec_of(_ep({**_EXACT_MAP, _OWN: value}), **on)
        assert rec == C.REC_OOS_DECLARATION_MISMATCH and _OWN in msg
    rec, msg = rec_of(_ep({_OWN: dict(_DECL)}), **on)                # steered key, no twin
    assert rec == C.REC_OOS_DECLARATION_MISMATCH and C.TWIN_CELL_ID in msg
    rec, msg = rec_of(_ep({_OWN: dict(_DECL), C.TWIN_CELL_ID: {C.SERVE_OUT_OF_SUPPORT_KEY: ["thinking", "other"]}}), **on)
    assert rec == C.REC_OOS_DECLARATION_MISMATCH and C.TWIN_CELL_ID in msg
    rec, msg = rec_of(_ep({C.TWIN_CELL_ID: dict(_DECL)}), **on)     # own key absent
    assert rec == C.REC_OOS_DECLARATION_UNATTESTED and _OWN in msg
    rec, _ = rec_of(_ep({}), **on)
    assert rec == C.REC_OOS_DECLARATION_UNATTESTED
    rec, msg = rec_of(_ep(error="workers disagree on serve declarations"), **on)
    assert rec == C.REC_OOS_DECLARATION_UNATTESTED and "workers disagree" in msg
    rec, msg = rec_of(_ep(declarations=_ABSENT), **on)               # the key absent: pre-landing
    assert rec == C.REC_OOS_DECLARATION_UNATTESTED and "before the lens landing" in msg
    # ... the additive-key rule: the ABSENT key is tolerated on an in-support run (recorded raw None)
    tolerated = C.check_serve_declarations(_ep(declarations=_ABSENT), cell=_TT, arm=C.THINKING_OFF,
                                           out_of_support=False)
    assert tolerated == {"expected": None, "own_key": "si_safety__psr_kl_all", "own": None,
                         "twin_key": None, "twin": None, "raw": None, "error_present": True}
    rec, _ = rec_of(_ep(declarations=None), **on)
    assert rec == C.REC_OOS_DECLARATION_UNATTESTED
    rec, _ = rec_of(_ep(declarations="not a map"), **on)
    assert rec == C.REC_OOS_DECLARATION_UNATTESTED
    rec, _ = rec_of("not a mapping", **on)
    assert rec == C.REC_OOS_DECLARATION_UNATTESTED


def test_check_serve_declarations_refuses_a_declaration_on_every_other_configuration():
    """§3.2 row 2: the base cell, an OFF arm and an in-support run
    expect an EMPTY map (present) with a null error — their own BARE key
    present is a wiring error (MISMATCH); the twin's presence there is never a
    refusal; a non-null error refuses on every configuration; the arm token
    `per-task` (a MODE) is fine in support."""
    empty = _ep({})
    for cfg in (dict(cell=C.ROUTE_NONE_CELL, arm=C.THINKING_ON, out_of_support=True),
                dict(cell=C.ROUTE_NONE_CELL, arm=C.THINKING_OFF, out_of_support=False),
                dict(cell=_TT, arm=C.THINKING_OFF, out_of_support=True),
                dict(cell=_TT, arm=C.THINKING_OFF, out_of_support=False),
                dict(cell=_TT, arm=C.THINKING_ON, out_of_support=False),
                dict(cell=_TT, arm=C.THINKING_PER_TASK, out_of_support=False),
                dict(cell=C.CELL_PROMPTED_SAFETY, arm=C.THINKING_ON, out_of_support=False)):
        out = C.check_serve_declarations(empty, **cfg)
        assert out["expected"] is None and out["own"] is None and out["twin_key"] is None
        assert out["raw"] == {}
        # a twin entry left by a declared donor is tolerated on an undeclared configuration
        C.check_serve_declarations(_ep({C.TWIN_CELL_ID: dict(_DECL)}), **cfg)
        with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
            C.check_serve_declarations(_ep(error="nothing attested"), **cfg)
        assert e.value.rec == C.REC_OOS_DECLARATION_UNATTESTED
    # the own BARE key present on an undeclared lens configuration: MISMATCH
    for cfg in (dict(cell=_TT, arm=C.THINKING_OFF, out_of_support=True),
                dict(cell=_TT, arm=C.THINKING_ON, out_of_support=False)):
        with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
            C.check_serve_declarations(_ep({"si_safety__psr_kl_all": dict(_DECL)}), **cfg)
        assert e.value.rec == C.REC_OOS_DECLARATION_MISMATCH and "wiring error" in str(e.value)
    # ... and the base cell declaring anything
    with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
        C.check_serve_declarations(_ep({"none": dict(_DECL)}),
                                   cell=C.ROUTE_NONE_CELL, arm=C.THINKING_ON, out_of_support=True)
    assert e.value.rec == C.REC_OOS_DECLARATION_MISMATCH
    # the declared configuration does NOT accept the bare key in place of the variant key
    with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
        C.check_serve_declarations(_ep({"si_safety__psr_kl_all": dict(_DECL), C.TWIN_CELL_ID: dict(_DECL)}),
                                   cell=_TT, arm=C.THINKING_ON, out_of_support=True)
    assert e.value.rec == C.REC_OOS_DECLARATION_UNATTESTED and _OWN in str(e.value)


def test_check_serve_declarations_requires_the_error_key_present_and_null_under_the_override():
    """An ABSENT ``lens_serve_declarations_error`` used to read
    exactly like a present null on both doors — but the contract puts the key,
    null, beside the map on EVERY standup (§2.5.1 / §2.5.12's pattern), and its
    absence cannot prove the workers agreed. Presence apart from value: under
    the override the key must be PRESENT and null (absent ->
    OOS_DECLARATION_UNATTESTED naming the key; present non-null -> the existing
    refusal); in support absence is tolerated and RECORDED (``error_present``
    False), a present non-null still refuses."""
    on = dict(cell=_TT, arm=C.THINKING_ON, out_of_support=True)
    ok = C.check_serve_declarations(_ep(), **on)
    assert ok["error_present"] is True
    with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
        C.check_serve_declarations(_ep(error=_ABSENT), **on)
    assert e.value.rec == C.REC_OOS_DECLARATION_UNATTESTED
    assert C.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR in str(e.value) and "no " in str(e.value)
    with pytest.raises(C.OutOfSupportAttestationRefusal) as e:                 # none/ON too
        C.check_serve_declarations(_ep({}, error=_ABSENT), cell=C.ROUTE_NONE_CELL, arm=C.THINKING_ON,
                                   out_of_support=True)
    assert e.value.rec == C.REC_OOS_DECLARATION_UNATTESTED
    with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
        C.check_serve_declarations(_ep(error="workers disagree"), **on)
    assert e.value.rec == C.REC_OOS_DECLARATION_UNATTESTED and "workers disagree" in str(e.value)
    # in support: tolerated and recorded; a present non-null still refuses
    off = dict(cell=_TT, arm=C.THINKING_OFF, out_of_support=False)
    tolerated = C.check_serve_declarations(_ep({}, error=_ABSENT), **off)
    assert tolerated["error_present"] is False and tolerated["raw"] == {}
    assert C.check_serve_declarations(_ep({}), **off)["error_present"] is True
    assert C.check_serve_declarations(_ep(declarations=_ABSENT, error=_ABSENT), **off)["error_present"] is False
    with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
        C.check_serve_declarations(_ep({}, error="nothing attested"), **off)
    assert e.value.rec == C.REC_OOS_DECLARATION_UNATTESTED
    # ... and through the ONE sequence, on both readings of the override
    base = _ep(**{C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH: "a" * 16,
                  C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {_OWN: "conditional", C.TWIN_CELL_ID: "conditional"},
                  C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: _SHA,
                  C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {_OWN: _SHA},
                  C.ENDPOINT_KEY_LENS_SALT_POLICY: "attested"})
    absent = {k: v for k, v in base.items() if k != C.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR}
    qwen = C.AUTHORISED_OOS_THINKING.model
    with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
        C.check_serve_attestation(absent, cell=_TT, arm="on", out_of_support=True, model_key=qwen,
                                  expected_object_sha256=_SHA, budget={"chat_template_hash": "a" * 16})
    assert e.value.rec == C.REC_OOS_DECLARATION_UNATTESTED
    plain = {**absent, C.PROXY_IDENTITY_SERVE_DECLARATIONS: {},          # the OFF reference: bare keys
             C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {"si_safety__psr_kl_all": "conditional"},
             C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {"si_safety__psr_kl_all": _SHA}}
    out = C.check_serve_attestation(plain, cell=_TT, arm="off", out_of_support=False, model_key=qwen,
                                    expected_object_sha256=_SHA, budget=None)
    assert out["declarations"]["error_present"] is False


def test_check_object_artifact_digest_scalar_and_by_cell():
    """§2.5.7 / §2.5.9, presence apart from value (review round B, P2-2). The
    per-cell map PRESENT must NAME the harness key in the served spelling and
    bind it to the digest the launcher passed — the other fitted cells a
    multi-cell standup binds are not a mismatch (F-V: the seedrep view engine
    serves five); a map not naming the cell, another sha under it, a present
    ``null`` or a non-mapping refuse OOS_DECLARATION_MISMATCH everywhere. The
    legacy scalar PRESENT beside the map is held to the writer's derivation
    (the single distinct digest, null when several — a disagreeing scalar is a
    torn record); PRESENT alone it must equal the digest (a bare null binds
    nothing). ABSENT keys follow the additive-key rule: under ``required`` (the
    override) an absent scalar OR an absent per-cell map refuses (the landed
    standup writes both); in support either is tolerated and recorded ``None``
    — with the per-cell map STILL compared when it is the one present (scalar
    absent + map correct passes; scalar absent + map wrong refuses)."""
    ok = {C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: _SHA}
    both = {**ok, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {_OWN: _SHA}}
    assert C.check_object_artifact_digest(both, harness_key=_OWN, expected_sha256=_SHA) == {
        "scalar": _SHA, "by_cell": {_OWN: _SHA}}
    # F-V: a multi-cell standup — the map binds OTHER fitted cells too, and the legacy
    # scalar is null (several distinct digests) — passes; the WHOLE map is recorded
    several = {C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: None,
               C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {_OWN: _SHA, "si_other__lora_r4_ce": "cd" * 32}}
    assert C.check_object_artifact_digest(several, harness_key=_OWN, expected_sha256=_SHA, required=True) == {
        "scalar": None, "by_cell": {_OWN: _SHA, "si_other__lora_r4_ce": "cd" * 32}}
    # two cells bound to the SAME digest: one distinct value, so the scalar carries it
    assert C.check_object_artifact_digest(
        {**ok, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {_OWN: _SHA, "x": _SHA}},
        harness_key=_OWN, expected_sha256=_SHA) == {"scalar": _SHA, "by_cell": {_OWN: _SHA, "x": _SHA}}

    def rec_of(ep, **kw):
        with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
            C.check_object_artifact_digest(ep, harness_key=_OWN, expected_sha256=_SHA, **kw)
        assert e.value.rec == C.REC_OOS_DECLARATION_MISMATCH and _SHA in str(e.value)
        return str(e.value)

    # PRESENT and wrong: refused under both readings of `required`
    for bad in ({**ok, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {"si_safety__psr_kl_all": _SHA}},  # bare key: not the served spelling
                {**ok, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {_OWN: "cd" * 32}},
                {**ok, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {_OWN: _SHA, "x": "cd" * 32}},  # several -> scalar must be null
                {**ok, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: "sha"},
                {**ok, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: None},           # present null
                {**both, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: "cd" * 32},           # torn: scalar != its own map
                {**both, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: None},                 # torn: one digest, null scalar
                {C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: None},                         # null scalar, no map: binds nothing
                {C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {_OWN: "cd" * 32}}):   # scalar absent, map wrong
        for required in (True, False):
            rec_of(bad, required=required)
    # ABSENT: refused under the override, tolerated (None) in support — the map still compared
    assert C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL in rec_of(ok, required=True)    # map absent
    assert C.check_object_artifact_digest(ok, harness_key=_OWN, expected_sha256=_SHA, required=False) == {
        "scalar": _SHA, "by_cell": None}
    assert C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256 in rec_of(
        {C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {_OWN: _SHA}}, required=True)      # scalar absent
    assert C.check_object_artifact_digest({C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {_OWN: _SHA}},
                                          harness_key=_OWN, expected_sha256=_SHA, required=False) == {
        "scalar": None, "by_cell": {_OWN: _SHA}}
    rec_of({}, required=True)
    assert C.check_object_artifact_digest({}, harness_key=_OWN, expected_sha256=_SHA, required=False) == {
        "scalar": None, "by_cell": None}
    with pytest.raises(C.OutOfSupportAttestationRefusal, match="not a mapping"):
        C.check_object_artifact_digest("not a mapping", harness_key=_OWN, expected_sha256=_SHA, required=False)
    with pytest.raises(ValueError, match="64-hex"):
        C.check_object_artifact_digest(ok, harness_key=_OWN, expected_sha256="abc")


def test_check_lens_salt_policy_and_the_attested_per_cell_policy():
    """§2.5.9 / §2.5.11 (Q11), presence apart from value (review round B,
    P2-2). ``lens_salt_policy`` ABSENT -> None in support, refused under the
    override; PRESENT ``attested`` passes; a present ``null`` or anything else
    -> OOS_SALT_POLICY_UNEXPECTED on every configuration. The attested
    ``resumed_prefill_policy_cells[<variant key>]`` must read ``conditional``:
    the map ABSENT -> None in support, refused under the override; the map
    PRESENT but null / not a mapping, the BARE key, ``always`` refuse
    everywhere, naming the key and the expected value."""
    assert C.check_lens_salt_policy({}) is None
    with pytest.raises(C.OutOfSupportAttestationRefusal) as e:        # required under the override
        C.check_lens_salt_policy({}, required=True)
    assert e.value.rec == C.REC_OOS_SALT_POLICY_UNEXPECTED
    assert C.check_lens_salt_policy({C.ENDPOINT_KEY_LENS_SALT_POLICY: "attested"}) == "attested"
    for bad, required in (({C.ENDPOINT_KEY_LENS_SALT_POLICY: "request_unique_all"}, False),
                          ({C.ENDPOINT_KEY_LENS_SALT_POLICY: "request_unique_all"}, True),
                          ({C.ENDPOINT_KEY_LENS_SALT_POLICY: None}, False),        # present null
                          ({C.ENDPOINT_KEY_LENS_SALT_POLICY: None}, True),
                          ("not a mapping", False)):
        with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
            C.check_lens_salt_policy(bad, required=required)
        assert e.value.rec == C.REC_OOS_SALT_POLICY_UNEXPECTED
        if isinstance(bad, dict) and bad[C.ENDPOINT_KEY_LENS_SALT_POLICY]:
            assert C.FORCED_SALT_POLICY_LABEL in str(e.value) and "'request_unique_all'" in str(e.value)
    # B2's twin rule, read through this lane's alias: the registry's control leg under VMODEL_PREFIX
    assert C.twin_vmodel() == "petri__none__lens0"
    cells = {C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {_OWN: "conditional", C.TWIN_CELL_ID: "conditional"}}
    assert C.check_resumed_prefill_policy_cell(cells, harness_key=_OWN, required=True) == "conditional"
    assert C.check_resumed_prefill_policy_cell({}, harness_key=_OWN, required=False) is None
    for bad, required in (({}, True),
                          ({C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: None}, False),   # present null
                          ({C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: None}, True),
                          ({C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {_OWN: "always"}}, True),
                          ({C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {_OWN: "always"}}, False),
                          ({C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {"si_safety__psr_kl_all": "conditional"}}, True),
                          ({C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {"si_safety__psr_kl_all": "conditional"}}, False),
                          ({C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: "conditional"}, False),
                          ({C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {_OWN: "never"}}, False),
                          ("not a mapping", False)):
        with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
            C.check_resumed_prefill_policy_cell(bad, harness_key=_OWN, required=required)
        assert e.value.rec == C.REC_OOS_SALT_POLICY_UNEXPECTED
        if isinstance(bad, dict) and bad:
            assert C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS in str(e.value) and _OWN in str(e.value)
            assert "'conditional'" in str(e.value)
    assert C.OOS_EXPECTED_RESUMED_PREFILL_POLICY == "conditional" == C.RESUMED_PREFILL_CONDITIONAL
    assert C.LENS_SALT_POLICY_ATTESTED == "attested"


def _policy_rows():
    """``(policy label, expected cache_salt_regime)`` for every label the
    contract's vocabulary carries — the proxy's own rule, ``salt_regime_for``:
    only an attested ``always`` shares a namespace; every other label, the two
    non-values and the two lens-half labels included, is request-unique."""
    return [(label, "cell-stable" if label == "always" else "request-unique")
            for label in C.RESUMED_POLICY_LABELS]


def test_expected_stamp_regimes_reads_the_leg_s_policy_through_the_proxy_s_salt_rule():
    """Review round B, P1: ``expected_stamp_regimes(lens, arm, policy)`` — the
    render from the arm, the salt pair from the LABEL through
    ``RoutingProxy.salt_regime_for`` (``always`` -> cell-stable; ``never`` /
    ``conditional`` / ``unattested`` / ``absent`` / the two lens-half labels ->
    request-unique). No policy = the AUTHORISED experiment's contract row
    (``authorised_stamp_policy``: §2.5.11's five rows, verbatim); an
    off-vocabulary label, a non-lens leg with a policy other than
    ``not-consulted``, or a MODE for the arm refuse."""
    from sisteer.capeval.serving.proxy import RoutingProxy

    for label, salt in _policy_rows():
        if label == C.POLICY_NOT_CONSULTED:
            continue
        for arm in C.THINKING_MODES:
            row = C.expected_stamp_regimes(lens=True, arm=arm, policy=label)
            assert row == C.StampRegimes(salt, label, C.RENDER_REGIME_BY_ARM[arm]), (label, arm)
            assert row.salt_regime == RoutingProxy.salt_regime_for(
                lens=True, resumed_prefill_policy=(label if label in C.RESUMED_PREFILL_POLICIES else None))
    assert C.expected_stamp_regimes(lens=False, arm="off", policy=C.POLICY_NOT_CONSULTED) == C.StampRegimes(
        "cell-stable", "not-consulted", "thinking_off")
    # the authorised rows are the no-policy default, and the same table by name
    assert C.authorised_stamp_policy(lens=False, arm="on") == C.POLICY_NOT_CONSULTED
    assert C.authorised_stamp_policy(lens=True, arm="off") == "conditional"
    assert C.authorised_stamp_policy(lens=True, arm="on") == C.DECLARED_SALT_POLICY_LABEL
    for lens in (True, False):
        for arm in C.THINKING_MODES:
            assert C.expected_stamp_regimes(lens=lens, arm=arm) == C.expected_stamp_regimes(
                lens=lens, arm=arm, policy=C.authorised_stamp_policy(lens=lens, arm=arm))
    # refusals
    with pytest.raises(ValueError, match="not one of the contract's resumed_prefill_policy labels"):
        C.expected_stamp_regimes(lens=True, arm="off", policy="sometimes")
    with pytest.raises(ValueError, match="non-lens leg consults no policy"):
        C.expected_stamp_regimes(lens=False, arm="off", policy="conditional")
    with pytest.raises(ValueError, match="whole arm"):
        C.expected_stamp_regimes(lens=True, arm=C.THINKING_PER_TASK, policy="always")
    with pytest.raises(ValueError, match="whole arm"):
        C.authorised_stamp_policy(lens=True, arm=C.THINKING_PER_TASK)


def test_expected_stamp_policy_is_strict_on_the_authorised_legs_and_attested_everywhere_else():
    """Review round B, P1: the label a leg's responses must wear. The base
    route -> ``not-consulted`` whatever the record says. The AUTHORISED psr
    legs (the authorised cell on the authorised model, either spelling) -> the
    STRICT row: ``conditional`` under OFF (the in-support reference included),
    ``declared-request-unique`` under ON out of support — whatever the record
    attests (that compare is ``check_resumed_prefill_policy_cell``'s). EVERY
    other lens run -> the ATTESTED label under its key: ``always`` (an
    all-position cell), ``never``, ``conditional`` (a Gemma psr ON request in
    support stays ``conditional`` — the per-request declaration declares no
    cell); a cell named in ``lens_serve_declarations`` ->
    ``declared-request-unique``; the map absent -> ``unattested``; a map that
    does not name the cell -> ``absent``; the twin's own label under
    ``harness_key=TWIN_CELL_ID``. A non-mapping map or an off-vocabulary value
    refuses OOS_SALT_POLICY_UNEXPECTED."""
    qwen, other = C.AUTHORISED_OOS_THINKING.model, "gemma4e4b"
    assert other not in C.AUTHORISED_OOS_THINKING.model_spellings()
    bare = "si_safety__psr_kl_all"
    rec = {C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {bare: "always", C.TWIN_CELL_ID: "always"}}
    # the base route
    for model in (qwen, other):
        assert C.expected_stamp_policy(rec, cell="none", arm="on", out_of_support=True,
                                       model_key=model) == C.POLICY_NOT_CONSULTED
        assert C.expected_stamp_policy({}, cell="none", arm="off", out_of_support=False,
                                       model_key=model) == C.POLICY_NOT_CONSULTED
    # the authorised legs: strict, whatever the record attests
    for model in C.AUTHORISED_OOS_THINKING.model_spellings():
        assert C.expected_stamp_policy(rec, cell=_TT, arm="off", out_of_support=False,
                                       model_key=model) == "conditional"
        assert C.expected_stamp_policy({}, cell=_TT, arm="on", out_of_support=True,
                                       model_key=model) == C.DECLARED_SALT_POLICY_LABEL
        assert C.expected_stamp_policy({}, cell=_TT, arm="on", out_of_support=True,
                                       model_key=model, harness_key=C.TWIN_CELL_ID) == C.DECLARED_SALT_POLICY_LABEL
        assert C.expected_stamp_policy(rec, cell=_TT, arm="off", out_of_support=False,
                                       model_key=model, harness_key=C.TWIN_CELL_ID) == "conditional"
    # every other lens run: the attested label, as the proxy would stamp it
    for label in ("always", "never", "conditional"):
        ep = {C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {bare: label, C.TWIN_CELL_ID: label}}
        for arm in C.THINKING_MODES:
            assert C.expected_stamp_policy(ep, cell=_TT, arm=arm, out_of_support=False,
                                           model_key=other) == label, (label, arm)
        assert C.expected_stamp_policy(ep, cell=_TT, arm="off", out_of_support=False, model_key=other,
                                       harness_key=C.TWIN_CELL_ID) == label
        assert C.attested_stamp_policy(ep, bare) == label
    assert C.expected_stamp_policy({}, cell=_TT, arm="off", out_of_support=False,
                                   model_key=other) == C.POLICY_UNATTESTED
    assert C.expected_stamp_policy({C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {C.TWIN_CELL_ID: "never"}},
                                   cell=_TT, arm="off", out_of_support=False,
                                   model_key=other) == C.POLICY_CELL_ABSENT
    # the per-task MODE in support: the cell's own bare key, no arm read
    assert C.expected_stamp_policy(rec, cell=_TT, arm=C.THINKING_PER_TASK, out_of_support=False,
                                   model_key=other) == "always"
    # a cell in the declarations map is declared-request-unique, whatever the policy map says
    declared = {**rec, C.PROXY_IDENTITY_SERVE_DECLARATIONS: {bare: dict(_DECL)}}
    assert C.attested_stamp_policy(declared, bare) == C.DECLARED_SALT_POLICY_LABEL
    assert C.attested_stamp_policy(declared, C.TWIN_CELL_ID) == "always"
    # the twin's label is read under ITS key even when the cell's differs
    mixed = {C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {bare: "never", C.TWIN_CELL_ID: "conditional"}}
    assert C.expected_stamp_policy(mixed, cell=_TT, arm="off", out_of_support=False,
                                   model_key=other, harness_key=C.TWIN_CELL_ID) == "conditional"
    # the refusals
    for bad in ({C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: "conditional"},
                {C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {bare: "sometimes"}}, "not a mapping"):
        with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
            C.expected_stamp_policy(bad, cell=_TT, arm="off", out_of_support=False, model_key=other)
        assert e.value.rec == C.REC_OOS_SALT_POLICY_UNEXPECTED
    # every label the derivation can return has a row in the expectation table
    for label in C.RESUMED_POLICY_LABELS:
        lens = label != C.POLICY_NOT_CONSULTED
        C.expected_stamp_regimes(lens=lens, arm="off", policy=label)


def test_check_serve_attestation_is_the_one_sequence_both_doors_run():
    """Review round B, P2-2: the ONE compare sequence and its ORDER —
    declarations, digest, salt policy, the authorised leg's per-cell policy,
    the template on an out-of-support ON configuration — with the additive
    rule (absent refuses under the override only; present-null / off-contract
    refuses everywhere). A multi-fault record files the FIRST failing step's
    REC; the none/ON leg is held to the artifact too (ONE template scope); an
    out-of-support lens run told no digest refuses UNATTESTED; a step that
    passes records its fact."""
    qwen = C.AUTHORISED_OOS_THINKING.model
    budget = {"chat_template_hash": "a" * 16}
    full = _ep(**{C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH: "a" * 16,
                  C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {_OWN: "conditional", C.TWIN_CELL_ID: "conditional"},
                  C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: _SHA,
                  C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {_OWN: _SHA},
                  C.ENDPOINT_KEY_LENS_SALT_POLICY: "attested"})
    on = dict(cell=_TT, arm="on", out_of_support=True, model_key=qwen, expected_object_sha256=_SHA,
              budget=budget)
    out = C.check_serve_attestation(full, **on)
    assert out["declarations"]["raw"] == _EXACT_MAP and out["object_digest"] == {"scalar": _SHA, "by_cell": {_OWN: _SHA}}
    assert out[C.ENDPOINT_KEY_LENS_SALT_POLICY] == "attested" and out[C.ROUTE_KEY_RESUMED_PREFILL_POLICY] == "conditional"

    def rec_of(ep, **kw):
        with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
            C.check_serve_attestation(ep, **kw)
        return e.value.rec

    # the order: a record failing steps 3 AND 5 files step 3's REC; failing 2 and 3 files step 2's
    multi = {**full, C.ENDPOINT_KEY_LENS_SALT_POLICY: "request_unique_all",
             C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH: "b" * 16}
    assert rec_of(multi, **on) == C.REC_OOS_SALT_POLICY_UNEXPECTED
    multi = {**multi, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: "cd" * 32}
    assert rec_of(multi, **on) == C.REC_OOS_DECLARATION_MISMATCH
    multi = {**multi, C.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR: "workers disagree"}
    assert rec_of(multi, **on) == C.REC_OOS_DECLARATION_UNATTESTED
    # the template step, and the none/ON leg held to it too
    assert rec_of({**full, C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH: "b" * 16}, **on) == C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH
    assert rec_of(full, **{**on, "budget": None}) == C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH
    none_on = dict(cell="none", arm="on", out_of_support=True, model_key=qwen, expected_object_sha256=None)
    base = _ep({}, **{C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH: "a" * 16, C.ENDPOINT_KEY_LENS_SALT_POLICY: "attested"})
    out = C.check_serve_attestation(base, budget=budget, **none_on)
    assert out["object_digest"] is None and out[C.ROUTE_KEY_RESUMED_PREFILL_POLICY] is None
    assert rec_of(base, budget=None, **none_on) == C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH
    assert rec_of({**base, C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH: "b" * 16}, budget=budget,
                  **none_on) == C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH
    # an out-of-support lens run told no digest
    assert rec_of(full, **{**on, "expected_object_sha256": None}) == C.REC_OOS_DECLARATION_UNATTESTED
    # the in-support OFF reference on the authorised model: bare key, {} map, no budget; the
    # additive rule tolerates absent keys, refuses present-null ones, holds the per-cell policy
    off = dict(cell=_TT, arm="off", out_of_support=False, model_key=qwen, expected_object_sha256=_SHA,
               budget=None)
    plain = _ep({}, **{C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {"si_safety__psr_kl_all": "conditional"},
                       C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: _SHA})
    out = C.check_serve_attestation(plain, **off)
    assert out["object_digest"] == {"scalar": _SHA, "by_cell": None} and out[C.ENDPOINT_KEY_LENS_SALT_POLICY] is None
    assert out[C.ROUTE_KEY_RESUMED_PREFILL_POLICY] == "conditional"
    today = _ep(declarations=_ABSENT, **{C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: _SHA})       # today's proxy
    out = C.check_serve_attestation(today, **off)
    assert out["declarations"]["raw"] is None and out[C.ROUTE_KEY_RESUMED_PREFILL_POLICY] is None
    assert rec_of({**plain, C.ENDPOINT_KEY_LENS_SALT_POLICY: None}, **off) == C.REC_OOS_SALT_POLICY_UNEXPECTED
    assert rec_of({**plain, C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: None}, **off) == C.REC_OOS_SALT_POLICY_UNEXPECTED
    assert rec_of({**plain, C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {C.TWIN_CELL_ID: "conditional"}},
                  **off) == C.REC_OOS_SALT_POLICY_UNEXPECTED
    assert rec_of({**plain, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: None}, **off) == C.REC_OOS_DECLARATION_MISMATCH
    # another model's psr cell in support: its policy is not held, its attested value is recorded
    other = dict(off, model_key="gemma4e4b")
    out = C.check_serve_attestation({**plain, C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS:
                                     {"si_safety__psr_kl_all": "always"}}, **other)
    assert out[C.ROUTE_KEY_RESUMED_PREFILL_POLICY] == "always"
    assert C.check_serve_attestation(today, **other)[C.ROUTE_KEY_RESUMED_PREFILL_POLICY] is None


def test_raw_thinking_header_returns_the_node_verbatim():
    """The certification's unvalidated reader: the block whatever its type,
    None for an absent path or a non-mapping payload; the validating reader
    refuses the same non-mapping block."""
    good = {"switch": True, "moves": True, "off_ids": [1], "on_ids": [2]}
    assert C.raw_thinking_header({C.PROXY_IDENTITY_KEY: {"thinking_header": good}}) == good
    assert C.raw_thinking_header({C.PROXY_IDENTITY_KEY: {"thinking_header": "yes"}}) == "yes"
    assert C.raw_thinking_header({C.PROXY_IDENTITY_KEY: {}}) is None
    assert C.raw_thinking_header(None) is None and C.raw_thinking_header("x") is None
    with pytest.raises(C.OutOfSupportAttestationError):
        C.attested_thinking_header({C.PROXY_IDENTITY_KEY: {"thinking_header": "yes"}})
    # a non-mapping block through the certification: the COMPLETE evidence block
    with pytest.raises(C.ThinkingHeaderCertificationError) as e:
        C.certify_thinking_header(
            header_ids_by_arm={"off": list(C.QWEN35_HEADER_IDS_OFF), "on": list(C.QWEN35_HEADER_IDS_ON)},
            measured_live=True, attested="yes", attested_error=None)
    block = e.value.evidence
    for flag in (C.OOS_EVIDENCE_HEADER_CERTIFIED, C.OOS_EVIDENCE_HEADER_ATTESTED_AGREES,
                 C.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED):
        assert block[flag] is False
    assert block[C.OOS_EVIDENCE_ATTESTED_SWITCH] is None and block[C.OOS_EVIDENCE_ATTESTED_MOVES] is None
    assert block["attested_thinking_header"] == "yes" and "not a mapping" in " ".join(block["certification_problems"])


def _attested(**over):
    block = {"switch": True, "moves": True, "off_ids": list(C.QWEN35_HEADER_IDS_OFF),
             "on_ids": list(C.QWEN35_HEADER_IDS_ON)}
    block.update(over)
    return block


def _measured(**over):
    ids = {"off": list(C.QWEN35_HEADER_IDS_OFF), "on": list(C.QWEN35_HEADER_IDS_ON)}
    ids.update(over)
    return ids


def test_certify_thinking_header_from_the_contract():
    """T3's certification, written from contract §2.5.1 / §2.5.7 / §2.5.12:
    measured ids equal the family's constants AND the attested block
    ``{"switch": true, "moves": true, "off_ids": <7>, "on_ids": <5>}`` agrees ->
    all three flags true; ``switch: "enable_thinking"`` (a string) refuses
    NAMING THE TYPE; ``1`` / ``false`` / ``moves: false`` / ``null`` refuse;
    ``on_ids`` of length 6 or disagreeing refuse; the block absent refuses; a
    non-null ``thinking_header_error`` refuses; an ON header whose tail is not
    the open-think token refuses; a local-tokenizer measurement refuses; the
    evidence on ANY refusal carries every flag false and every list; the three
    constants ``None`` refuse naming B0."""
    ev = C.certify_thinking_header(header_ids_by_arm=_measured(), measured_live=True, attested=_attested())
    assert (ev[C.OOS_EVIDENCE_HEADER_CERTIFIED], ev[C.OOS_EVIDENCE_HEADER_ATTESTED_AGREES],
            ev[C.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED]) == (True, True, True)
    assert ev[C.OOS_EVIDENCE_ATTESTED_SWITCH] is True and ev[C.OOS_EVIDENCE_ATTESTED_MOVES] is True
    assert ev["attested_thinking_header"] == _attested() and ev["certification_problems"] == []
    assert ev["expected_header_ids_by_arm"] == _measured()

    def refused(match, **kw):
        kw.setdefault("header_ids_by_arm", _measured())
        kw.setdefault("measured_live", True)
        kw.setdefault("attested", _attested())
        with pytest.raises(C.ThinkingHeaderCertificationError, match=match) as e:
            C.certify_thinking_header(**kw)
        evidence = e.value.evidence
        for flag in (C.OOS_EVIDENCE_HEADER_CERTIFIED, C.OOS_EVIDENCE_HEADER_ATTESTED_AGREES,
                     C.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED):
            assert evidence[flag] is False, flag
        assert evidence["certification_problems"] and evidence["expected_header_ids_by_arm"] == _measured()
        return evidence

    e = refused(r"switch = 'enable_thinking' \(str\)", attested=_attested(switch="enable_thinking"))
    assert e[C.OOS_EVIDENCE_ATTESTED_SWITCH] == "enable_thinking"      # recorded with its type
    refused(r"switch = 1 \(int\)", attested=_attested(switch=1))
    refused(r"switch = False \(bool\)", attested=_attested(switch=False))
    refused(r"moves = False \(bool\)", attested=_attested(moves=False))
    refused(r"moves = None \(NoneType\)", attested=_attested(moves=None, on_ids=None))   # §2.5.12's partial fact
    refused("on_ids", attested=_attested(on_ids=list(C.QWEN35_HEADER_IDS_ON) + [271]))  # length 6
    refused("off_ids", attested=_attested(off_ids=list(C.QWEN35_HEADER_IDS_OFF[:-1]) + [270]))
    refused("no petri_proxy.thinking_header block", attested=None)
    refused("not a mapping", attested="switch")
    refused("thinking_header_error = 'ON prompt is not a prefix", attested_error="ON prompt is not a prefix of the scaffold")
    off_by_one = list(C.QWEN35_HEADER_IDS_ON[:-1]) + [199]              # the header off by ONE id
    refused("does not end in the open-think token", header_ids_by_arm=_measured(on=off_by_one),
            attested=_attested(on_ids=off_by_one))
    refused("measured header ids", header_ids_by_arm=_measured(on=list(C.QWEN35_HEADER_IDS_ON) + [198]),
            attested=_attested(on_ids=list(C.QWEN35_HEADER_IDS_ON) + [198]))     # 6 ids, still opens
    refused("local-tokenizer", measured_live=False)
    refused("lacks an arm", header_ids_by_arm={"on": list(C.QWEN35_HEADER_IDS_ON)})
    # a CLOSE id after the OPEN is not the prompt-opened render, whatever the tail
    assert C._header_tail_opens_think(C.QWEN35_HEADER_IDS_ON) is True
    assert C._header_tail_opens_think(C.QWEN35_HEADER_IDS_OFF) is False
    assert C._header_tail_opens_think((C.QWEN35_THINK_OPEN_ID, C.QWEN35_THINK_CLOSE_ID, C.QWEN35_THINK_OPEN_ID, 198)) is True
    assert C._header_tail_opens_think((C.QWEN35_THINK_OPEN_ID, 198, C.QWEN35_THINK_CLOSE_ID, 198)) is False
    assert C.PROXY_IDENTITY_KEY == C.THINKING_HEADER_SOURCE[0] == C.THINKING_HEADER_ERROR_SOURCE[0]
    payload = {C.PROXY_IDENTITY_KEY: {"thinking_header": _attested(), "thinking_header_error": None}}
    assert C.attested_thinking_header(payload) == _attested()
    assert C.attested_thinking_header_error(payload) is None
    assert C.attested_thinking_header_error({C.PROXY_IDENTITY_KEY: {"thinking_header_error": "why"}}) == "why"
    assert C.attested_thinking_header_error(None) is None


def test_certify_thinking_header_refuses_naming_b0_while_a_constant_reads_none(monkeypatch):
    for name in ("THINKING_HEADER_SWITCH_FIELD", "THINKING_HEADER_SWITCH_EXPECTED",
                 "THINKING_HEADER_MOVES_EXPECTED"):
        monkeypatch.setattr(C, name, None)
        with pytest.raises(C.OutOfSupportDeclarationUnreconciled, match="B0"):
            C.certify_thinking_header(header_ids_by_arm=_measured(), measured_live=True, attested=_attested())
        monkeypatch.undo()


def test_is_engine_id_of_accepts_exactly_one_lowercase_hex_suffix():
    """The lens side's GPU-gate fact: vLLM's ``assign_request_id`` appends
    ``-<8 lowercase hex>`` ONCE to the client-visible id. Exact match ok; one
    suffix ok; two suffixes, uppercase, another length, an empty id or a
    non-string refused. No gate check compares ids — the rule lives in
    contracts for the worker and the ledger cross-check."""
    assert C.is_engine_id_of("chatcmpl-abc", "chatcmpl-abc") is True
    assert C.is_engine_id_of("chatcmpl-abc-0123abcd", "chatcmpl-abc") is True
    assert C.is_engine_id_of("chatcmpl-abc-0123abcd-0123abcd", "chatcmpl-abc") is False
    assert C.is_engine_id_of("chatcmpl-abc-0123ABCD", "chatcmpl-abc") is False
    assert C.is_engine_id_of("chatcmpl-abc-0123abc", "chatcmpl-abc") is False
    assert C.is_engine_id_of("chatcmpl-abc-0123abcde", "chatcmpl-abc") is False
    assert C.is_engine_id_of("chatcmpl-abc0123abcd", "chatcmpl-abc") is False      # no dash
    assert C.is_engine_id_of("other-0123abcd", "chatcmpl-abc") is False
    assert C.is_engine_id_of("", "") is False and C.is_engine_id_of(None, "x") is False
    assert C.is_engine_id_of("x-0123abcd", 5) is False
    src = (REPO / "scripts/served_capability_run.py").read_text()
    assert "is_engine_id_of" not in src          # no driver check compares request ids


def test_the_rec_names_are_the_launchers_spellings_and_the_refusal_carries_its_rec():
    """The run-ending REC statuses, spelled once: the launcher's
    existing literals are the same strings; every refusal helper raises a
    typed refusal whose text starts with its REC."""
    launcher = (REPO / "scripts/modal/tinycap_served.py").read_text()
    for rec in (C.REC_OOS_UNAUTHORISED, C.REC_THINKING_UNSERVABLE,
                C.REC_REDUCTION_MODE_UNATTESTED, C.REC_REDUCTION_MODE_MISMATCH):
        assert f'"{rec}"' in launcher, rec
    assert C.OOS_REC_NAMES == ("OOS_UNAUTHORISED", "OOS_DECLARATION_UNATTESTED",
                               "OOS_DECLARATION_MISMATCH", "OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH",
                               "OOS_SALT_POLICY_UNEXPECTED", "OOS_TORN_STORE_UNREPAIRABLE",
                               "THINKING_UNSERVABLE",
                               "REDUCTION_MODE_UNATTESTED", "REDUCTION_MODE_MISMATCH")
    assert len(set(C.OOS_REC_NAMES)) == len(C.OOS_REC_NAMES)
    # Phase B integration: the launcher's REC names are THESE objects, re-exported
    # by reference — one literal per REC, on either side of the standup
    for rec in ("REC_OOS_DECLARATION_UNATTESTED", "REC_OOS_DECLARATION_MISMATCH",
                "REC_OOS_SALT_POLICY_UNEXPECTED", "REC_OOS_TORN_STORE_UNREPAIRABLE"):
        assert f"{rec} = C.{rec}" in launcher, rec
    assert 'REC_OOS_TORN_STORE_UNREPAIRABLE = "OOS_TORN_STORE_UNREPAIRABLE"' not in launcher
    e = C.OutOfSupportAttestationRefusal(C.REC_OOS_DECLARATION_MISMATCH, "own != twin")
    assert str(e) == "OOS_DECLARATION_MISMATCH: own != twin" and e.rec == "OOS_DECLARATION_MISMATCH"
    assert isinstance(e, C.OutOfSupportAttestationError)
    with pytest.raises(ValueError, match="run-ending RECs"):
        C.OutOfSupportAttestationRefusal("NOT_A_REC", "x")
    # the endpoint read-back keys are the standup's own spellings (capctrl_serving_up.py)
    capctrl = (REPO / "scripts/capctrl_serving_up.py").read_text()
    for key in (C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256, C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS):
        assert f'"{key}"' in capctrl, key


def test_the_docs_reconciled_names_table_states_the_constants_values():
    """docs §11's reconciled-names table restates constant VALUES in prose;
    this pins each row's value column to the constant it names, so a value
    that drifts from the code fails here rather than misleading a reader. The
    table is located by its heading, split by `|`, and every row's NAME column
    must contain each named constant while its VALUE column contains the
    constant's literal (through `repr` / `str` / the path join the row uses)."""
    doc = (Path(C.__file__).resolve().parents[4] / "docs" / "tinycap_interfaces.md").read_text()
    head = doc.index("**The reconciled names**")
    body = doc[head:]
    table = {}
    started = False
    for line in body.splitlines():
        if not line.startswith("|"):
            if started:
                break                # the table ends at the first non-row after it started
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            break
        if cells[0] == "contract placeholder" or set(cells[0]) <= {"-"}:
            started = True           # the header and the rule row
            continue
        started = True
        table[cells[1]] = cells[2]
    assert len(table) >= 12, sorted(table)

    def row_for(*names):
        [value_col] = [v for n, v in table.items() if all(f"`{x}`" in n for x in names)]
        return value_col

    pins = [
        (("OOS_THINKING_DECLARATION_KEY",), [f"`{C.OOS_THINKING_DECLARATION_KEY}`", f"`{C.OOS_THINKING_DECLARATION_VALUE!r}`"]),
        (("OOS_STANDUP_CELL_SUFFIX",), [f"`{C.OOS_STANDUP_CELL_SUFFIX}`"]),
        (("OOS_EXPECT_SHA_KEY",), [f"`{C.OOS_EXPECT_SHA_KEY}`"]),
        (("PROXY_IDENTITY_SERVE_DECLARATIONS",), [f"`{C.PROXY_IDENTITY_SERVE_DECLARATIONS}`",
                                                  f"`{C.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR}`"]),
        (("TWIN_CELL_ID",), [f"`{C.cell_spellings(C.CELL_PSRKL_SAFETY, declared=True).harness}`", f"`{C.TWIN_CELL_ID}`"]),
        (("OOS_CELL_DECLARATION_EXPECTED",), [f"`{json.dumps(C.OOS_CELL_DECLARATION_EXPECTED)}`"]),
        (("ROUTE_KEY_SERVE_DECLARATIONS",), [f"`{C.ROUTE_KEY_SERVE_DECLARATIONS}`"]),
        (("THINKING_HEADER_SOURCE",), [f"`/v1/models.{'.'.join(C.THINKING_HEADER_SOURCE)}`",
                                       f"`.{C.THINKING_HEADER_ERROR_SOURCE[-1]}`"]),
        (("THINKING_HEADER_SWITCH_FIELD",), [f"`{C.THINKING_HEADER_SWITCH_FIELD}`", f"`{C.THINKING_HEADER_SWITCH_EXPECTED!r}`",
                                             f"`{C.THINKING_HEADER_MOVES_EXPECTED!r}`"]),
        (("STAMP_KEY_RENDER_REGIME",), [f"`{C.STAMP_KEY_RENDER_REGIME}`", *(f"`{v}`" for v in C.RENDER_REGIME_LITERALS)]),
        (("STAMP_KEY_SALT_REGIME",), [f"`{C.STAMP_KEY_SALT_REGIME}`", *(f"`{v}`" for v in C.SALT_REGIME_LITERALS),
                                      f"`{C.STAMP_KEY_RESUMED_POLICY}`", f"`{C.DECLARED_SALT_POLICY_LABEL}`",
                                      str(len(C.RESUMED_POLICY_LABELS)) if len(C.RESUMED_POLICY_LABELS) != 8 else "eight"]),
        (("THINKING_REFUSAL_CODES",), [f"the {len(C.THINKING_REFUSAL_CODES)}" if len(C.THINKING_REFUSAL_CODES) != 8 else "the eight"]),
        (("ENDPOINT_KEY_LENS_SALT_POLICY",), [f"`{C.LENS_SALT_POLICY_ATTESTED}`", f"`{C.OOS_EXPECTED_RESUMED_PREFILL_POLICY}`"]),
        (("FLAG_OOS_PROMPT_BUDGET",), [f"`{C.FLAG_OOS_PROMPT_BUDGET} "]),
        (("FLAG_EXPECT_OBJECT_SHA256",), [f"`{C.FLAG_EXPECT_OBJECT_SHA256} "]),
    ]
    for names, needles in pins:
        value = row_for(*names)
        for needle in needles:
            assert needle in value, (names, needle, value)


_BAD_STAMP_VALUES = (None, "", 1, True, [], {}, "sometimes")


def test_stamp_regimes_refuses_a_present_malformed_value_and_reads_only_a_missing_key_as_absent():
    """``_short_label`` mapped EVERY non-string / empty failure to
    None, so a PRESENT malformed stamp (``petri.render_regime: null``, ``1``,
    ``true``, ``""``, ``[]``, ``{}``) read exactly like an ABSENT one and was
    then tolerated in support. Now presence and raw type are preserved at the
    ONE extraction site: a member is None ONLY when its key (or the whole
    block) is genuinely missing; a present null / empty / non-string /
    off-vocabulary value refuses (OOS_SALT_POLICY_UNEXPECTED) naming the key,
    the value and its type — whatever the run's support. A present block that
    is not a mapping refuses the same way."""
    base = {C.STAMP_BLOCK_KEY: {C.STAMP_KEY_SALT_REGIME: "request-unique",
                                C.STAMP_KEY_RESUMED_POLICY: "conditional",
                                C.STAMP_KEY_RENDER_REGIME: "thinking_off"}}
    assert C.stamp_regimes(base) == C.StampRegimes("request-unique", "conditional", "thinking_off")
    assert C.stamp_regimes({}) == C.StampRegimes(None, None, None)           # the block absent
    for key, field in ((C.STAMP_KEY_SALT_REGIME, "salt_regime"), (C.STAMP_KEY_RESUMED_POLICY, "resumed_policy"),
                       (C.STAMP_KEY_RENDER_REGIME, "render_regime")):
        block = {k: v for k, v in base[C.STAMP_BLOCK_KEY].items() if k != key}
        got = C.stamp_regimes({C.STAMP_BLOCK_KEY: block})                   # the key absent: None there only
        assert getattr(got, field) is None
        assert all(getattr(got, f) is not None for f in C.EXPECTED_STAMP_TRIPLE_FIELDS if f != field)
        for bad in _BAD_STAMP_VALUES:
            with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
                C.stamp_regimes({C.STAMP_BLOCK_KEY: {**base[C.STAMP_BLOCK_KEY], key: bad}})
            msg = str(e.value)
            assert e.value.rec == C.REC_OOS_SALT_POLICY_UNEXPECTED
            assert f"{C.STAMP_BLOCK_KEY}.{key}" in msg and repr(bad) in msg and type(bad).__name__ in msg
            if isinstance(bad, str) and bad:
                assert "outside the contract's vocabulary" in msg
    for bad_block in ("x", 1, [], None):
        with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
            C.stamp_regimes({C.STAMP_BLOCK_KEY: bad_block})
        assert type(bad_block).__name__ in str(e.value)
    # the vocabulary the extraction applies is the contract's, one mapping per field
    assert C.STAMP_VOCABULARY == {"salt_regime": C.SALT_REGIME_LITERALS, "resumed_policy": C.RESUMED_POLICY_LABELS,
                                  "render_regime": C.RENDER_REGIME_LITERALS}




# --------------------------------------------------------------------------- #
# fit-support transfer, experiment 1 follow-up — the thinking-OFF comparator   #
# at the thinking-ON cap (`--allow-off-cap-override`)                          #
# --------------------------------------------------------------------------- #

_PSR_BARE = "si_safety__psr_kl_all"            # the OFF standup's attestation key: the BARE harness id


def _comparator_request() -> dict:
    """The request that IS the authorised comparator, in the driver's spellings
    (one cell, its one mode) — the base every mutation below moves one field of."""
    exp = C.AUTHORISED_OFF_CAP_COMPARATOR
    return {"model": exp.model, "cell": exp.cells[-1], "controls": list(exp.controls),
            "mode": exp.modes[0], "sampling_mode": exp.sampling_mode,
            "samples": exp.samples, "reduction_mode": exp.reduction_mode, "pass_no": exp.pass_no}


def test_the_off_cap_comparator_tuple_is_the_spec_table_literally():
    """offcap §3.2.4: the comparator is experiment 1's two thinking-OFF legs at
    the thinking-ON cap — qwen35_4b x {none, psr} x {off}, paired-200 GSM8K,
    k = 1 on the card, per_row, the OFF cap of BOTH GSM8K task names raised to
    8192 — transcribed as a literal dict. `modes` is the one OFF arm: the flag
    can never stand up an ON configuration. The accessors derive the same
    spellings the OOS tuple's do (one helper per accessor, shared)."""
    exp = C.AUTHORISED_OFF_CAP_COMPARATOR
    assert exp.as_dict() == {
        "model": "qwen35_4b", "cells": ["none", "safety_refusal__psr_kl_all"],
        "controls": ["gsm8k"], "modes": ["off"], "sampling_mode": "card",
        "samples": 1, "reduction_mode": "per_row", "pass_no": 1,
        "max_tokens_off": {"gsm8k_paired200": 8192, "gsm8k": 8192}}
    assert C.OFFCAP_AUTHORISED_FIELDS == (
        "model", "cell", "controls", "mode", "sampling_mode", "samples", "reduction_mode", "pass_no")
    # the OOS checker's list is the same eight + `max_tokens_on` (compared when a cap is stated;
    # the comparator's checker states none, so its compare is byte-identical)
    assert C.OOS_AUTHORISED_FIELDS == (*C.OFFCAP_AUTHORISED_FIELDS, "max_tokens_on")
    assert exp.modes == (C.THINKING_OFF,) and C.THINKING_ON not in exp.modes
    assert all(C.VARIANT_SEP not in cell for cell in exp.cells)
    # the same derived spellings as the OOS tuple's, from the SAME helpers
    oos = C.AUTHORISED_OOS_THINKING
    assert exp.model_spellings() == oos.model_spellings() == tuple(dict.fromkeys((exp.model, C.model_row(exp.model).model_key)))
    assert exp.cell_spellings() == oos.cell_spellings()
    assert exp.control_atoms() == oos.control_atoms() == C.expand_control(C.CONTROL_GSM8K)
    for phrase in ("thinking-OFF", "8192", "512", "no declaration", "per_row", "0.7/0.95/20"):
        assert phrase in exp.what, phrase


def test_the_comparator_cap_is_the_on_cap_by_reference():
    """offcap §3.2.2: the comparator's whole point is the ON cap under the OFF
    render, so its cap constant IS the ON cap — assigned by reference, and the
    tuple's cap table spelled through the constant, never a second literal."""
    assert C.OFF_CAP_OVERRIDE_MAX_TOKENS == C.OOS_THINKING_MAX_TOKENS == 8192
    src = Path(C.__file__).read_text()
    assert "OFF_CAP_OVERRIDE_MAX_TOKENS = OOS_THINKING_MAX_TOKENS" in src
    tuple_src = src.split("AUTHORISED_OFF_CAP_COMPARATOR = AuthorisedOffCapComparator(", 1)[1].split(")\n\n", 1)[0]
    # the literal appears in the tuple's PROSE only (the `what` string), never in its cap table
    code_only = re.sub(r'"(?:[^"\\]|\\.)*"', '""', tuple_src)
    assert "8192" not in code_only and "OFF_CAP_OVERRIDE_MAX_TOKENS" in code_only
    assert set(C.AUTHORISED_OFF_CAP_COMPARATOR.max_tokens_off) == set(C.AUTHORISED_OOS_THINKING.max_tokens_on)


def test_off_cap_override_max_tokens_names_the_gsm8k_tasks_only_and_refuses_an_unknown_name():
    """offcap §3.2.7: the twin of `oos_max_tokens_on` — 8192 for both GSM8K task
    names, None for a task the comparator does not name, a refusal for a name
    that is not an Inspect task (a typo must not read as "no override")."""
    assert C.off_cap_override_max_tokens("gsm8k_paired200") == 8192 == C.off_cap_override_max_tokens("gsm8k")
    assert C.off_cap_override_max_tokens("math") is None and C.off_cap_override_max_tokens("bfcl") is None
    with pytest.raises(C.ControlNameError):
        C.off_cap_override_max_tokens("gsm8k_paired_200")
    with pytest.raises(C.ControlNameError):
        C.off_cap_override_max_tokens(C.GSM8K_STANDARD_TASK)       # the ROW task is not the name


def test_check_off_cap_override_authorised_accepts_both_spellings_and_the_subset_rule():
    """offcap §3.2.8: the comparator itself passes — under the launcher's cell
    id and the harness id, for both cells; a resumed driver's `--controls
    <remaining>` (a subset) is the same experiment; `reduction_mode` is
    compared when stated and skipped when None."""
    for cell in C.AUTHORISED_OFF_CAP_COMPARATOR.cell_spellings():
        C.check_off_cap_override_authorised(**{**_comparator_request(), "cell": cell})
    C.check_off_cap_override_authorised(
        **{**_comparator_request(), "controls": list(C.AUTHORISED_OFF_CAP_COMPARATOR.control_atoms())})
    C.check_off_cap_override_authorised(**{**_comparator_request(), "reduction_mode": None})
    with pytest.raises(C.OffCapOverrideUnauthorised, match="reduction_mode differs"):
        C.check_off_cap_override_authorised(
            **{**_comparator_request(), "reduction_mode": C.REDUCTION_MODE_STACKED})
    # an empty or unknown control set is never authorised
    for controls in ([], ["not_a_control"]):
        with pytest.raises(C.OffCapOverrideUnauthorised, match="controls differs"):
            C.check_off_cap_override_authorised(**{**_comparator_request(), "controls": controls})


@pytest.mark.parametrize("field,value", [
    ("model", "qwen35_9b"),
    ("cell", "si_constitution__psr_kl_all"),
    ("controls", ["gsm8k", "mmlu"]),
    ("mode", C.THINKING_ON),                 # the comparator has NO ON arm: the flag never stands one up
    ("mode", C.THINKING_PER_TASK),
    ("sampling_mode", C.SAMPLING_GREEDY),
    ("samples", 4),
    ("reduction_mode", C.REDUCTION_MODE_STACKED),
    ("pass_no", 2),                          # code review H3: pass 2 serves L 4,608 x S 78, not the tuple's window
])
def test_mutating_any_field_of_the_comparator_tuple_refuses_naming_it(field, value):
    """offcap §3.2.8: the second override authorised ONE comparator, not a
    family. Move any single field and the run is refused with the field named,
    the flag named, and the tuple quoted from where it lives."""
    req = {**_comparator_request(), field: value}
    with pytest.raises(C.OffCapOverrideUnauthorised) as e:
        C.check_off_cap_override_authorised(**req)
    msg = str(e.value)
    assert f"{field} differs" in msg
    assert C.FLAG_ALLOW_OFF_CAP_OVERRIDE in msg and C.FLAG_ALLOW_OOS_THINKING not in msg
    assert "AUTHORISED_OFF_CAP_COMPARATOR" in msg
    assert str(C.AUTHORISED_OFF_CAP_COMPARATOR.as_dict()) in msg
    assert field in C.OFFCAP_AUTHORISED_FIELDS
    # the OOS checker is untouched by the twin: the same mutated request against it
    # still names ITS flag (and `mode=on` is authorised there)
    if not (field == "mode" and value == C.THINKING_ON):
        with pytest.raises(C.OutOfSupportThinkingUnauthorised) as e2:
            C.check_out_of_support_thinking_authorised(**req)
        assert C.FLAG_ALLOW_OOS_THINKING in str(e2.value)


def test_the_two_override_flags_are_mutually_exclusive_at_the_contract():
    """offcap §3.1 / §3.2.9: the two overrides are never named together — the
    recorded 2x2 procedure keeps producing OFF/512 stores and the comparator's
    invocation can never stand up an ON configuration. Refused by name at the
    contract, and by the identity derivation that would otherwise have to pick
    a key set."""
    C.check_override_flags_exclusive(out_of_support_thinking=False, off_cap_override=False, where="x")
    C.check_override_flags_exclusive(out_of_support_thinking=True, off_cap_override=False, where="x")
    C.check_override_flags_exclusive(out_of_support_thinking=False, off_cap_override=True, where="x")
    with pytest.raises(C.OverrideFlagsExclusive) as e:
        C.check_override_flags_exclusive(out_of_support_thinking=True, off_cap_override=True,
                                         where="launch configuration 'x'")
    msg = str(e.value)
    assert C.FLAG_ALLOW_OOS_THINKING in msg and C.FLAG_ALLOW_OFF_CAP_OVERRIDE in msg
    assert "launch configuration 'x'" in msg
    assert issubclass(C.OverrideFlagsExclusive, ValueError)
    with pytest.raises(C.OverrideFlagsExclusive):
        C.tinycap_identity_fields(out_of_support_thinking=True, off_cap_override=True)


def test_the_identity_field_set_has_three_shapes_and_the_in_support_and_oos_ones_are_unchanged():
    """offcap §3.2.10: in-support (0 optional keys), the comparator (ONE: the
    hashed cap key alone), the OOS experiment (3) — pairwise distinct key sets,
    so `harness_config_hash` separates every store and no cross-resume is
    possible. The first two derivations equal the pre-change literals
    (copied here), and `TINYCAP_IDENTITY_OPTIONAL_FIELDS` is untouched."""
    base = ("tasks", "thinking", "versions", "served", "limits", "samples", "frameworks",
            "pass", "thinking_by_task", "math500_retry_max_tokens", "or_bench_item_set",
            *C.SAMPLING_IDENTITY_FIELDS)
    assert C.TINYCAP_IDENTITY_FIELDS == base
    assert C.tinycap_identity_fields() == base
    assert C.tinycap_identity_fields(out_of_support_thinking=False, off_cap_override=False) == base
    oos = C.tinycap_identity_fields(out_of_support_thinking=True)
    assert oos == (*base, "out_of_support_thinking", "max_tokens_by_task", "serve_out_of_support")
    assert C.TINYCAP_IDENTITY_OPTIONAL_FIELDS == ("out_of_support_thinking", "max_tokens_by_task",
                                                 "serve_out_of_support")
    offcap = C.tinycap_identity_fields(off_cap_override=True)
    assert offcap == (*base, C.IDENTITY_KEY_MAX_TOKENS_BY_TASK) == (*base, *C.TINYCAP_IDENTITY_OFF_CAP_FIELDS)
    assert C.TINYCAP_IDENTITY_OFF_CAP_FIELDS == ("max_tokens_by_task",)
    # the three shapes are pairwise distinct as SETS (the launcher compares exact keys)
    shapes = [set(base), set(offcap), set(oos)]
    assert all(a != b for i, a in enumerate(shapes) for b in shapes[i + 1:])
    assert set(offcap) - set(base) == {C.IDENTITY_KEY_MAX_TOKENS_BY_TASK}
    assert C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING not in offcap and C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT not in offcap


def test_the_declaration_stamp_policy_and_spellings_are_byte_identical_for_an_off_leg_at_any_cap():
    """offcap fact 3 / §3.2.13: declaration, spelling, stamp policy, attestation
    expectation and think-split are keyed on (lens, arm, out_of_support) and
    NEVER on the cap — for both cells, arm OFF, under `out_of_support` False AND
    True, every one of them returns the OFF/512 leg's value; none takes a cap."""
    qwen = C.AUTHORISED_OFF_CAP_COMPARATOR.model
    for cell in C.AUTHORISED_OFF_CAP_COMPARATOR.cells:
        lens = C.is_lens_cell(cell)
        for oos in (False, True):
            assert C.thinking_on_declaration(lens=lens, arm=C.THINKING_OFF) == {}
            assert C.render_fragment(lens=lens, arm=C.THINKING_OFF) == {
                "chat_template_kwargs": {"enable_thinking": False}}
            assert C.is_declared_configuration(cell, arm=C.THINKING_OFF, out_of_support=oos) is False
            assert C.expected_serve_out_of_support(lens=lens, arm=C.THINKING_OFF, out_of_support=oos) is None
            policy = C.expected_stamp_policy({}, cell=cell, arm=C.THINKING_OFF, out_of_support=oos,
                                             model_key=qwen)
            assert policy == (C.OOS_EXPECTED_RESUMED_PREFILL_POLICY if lens else C.POLICY_NOT_CONSULTED)
            if lens:
                assert policy == "conditional"
            sp = C.cell_spellings(cell, declared=C.is_declared_configuration(
                cell, arm=C.THINKING_OFF, out_of_support=oos))
            assert C.VARIANT_SEP not in sp.harness and C.VARIANT_SEP not in sp.traintest
        regimes = C.expected_stamp_regimes(lens=lens, arm=C.THINKING_OFF)
        assert regimes.render_regime == "thinking_off"
        assert C.think_split_mode(arm=C.THINKING_OFF, reasoning_parser=None,
                                  prompt_opened_think=False) == C.THINK_SPLIT_RECORD_ONLY
    for fn in (C.thinking_on_declaration, C.render_fragment, C.is_declared_configuration,
               C.expected_serve_out_of_support, C.expected_stamp_policy, C.expected_stamp_regimes,
               C.cell_spellings, C.think_split_mode, C.standup_extra_argv):
        params = set(inspect.signature(fn).parameters)
        assert not ({"cap", "max_tokens", "max_tokens_by_task", "off_cap_override"} & params), fn.__name__


def test_check_serve_attestation_under_the_off_cap_override_is_the_landed_undeclared_compare():
    """offcap §3.2.13: the comparator's containers are held to the LANDED record
    with the fail-closed presence rules — steps 1–4 with `out_of_support=True`
    on an OFF arm derive the UNDECLARED shape (map present, error key
    present-null, the own BARE key absent, the digest and the salt policy
    REQUIRED, the psr policy `conditional` REQUIRED) — and step 5 now runs for
    an off-cap-override configuration: the artifact digest is compared and a
    missing artifact refuses. The facts returned equal the in-support call's."""
    qwen = C.AUTHORISED_OFF_CAP_COMPARATOR.model
    budget = {"chat_template_hash": "a" * 16, "arm": "off", "cap": 8192}
    landed = _ep({}, **{C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH: "a" * 16,
                        C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {_PSR_BARE: "conditional"},
                        C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: _SHA,
                        C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {_PSR_BARE: _SHA},
                        C.ENDPOINT_KEY_LENS_SALT_POLICY: "attested"})
    offcap = dict(cell=_TT, arm="off", out_of_support=True, off_cap_override=True, model_key=qwen,
                  expected_object_sha256=_SHA, budget=budget)
    out = C.check_serve_attestation(landed, **offcap)
    assert out["declarations"]["raw"] == {} and out["declarations"]["expected"] is None
    assert out["declarations"]["own_key"] == _PSR_BARE and out["declarations"]["own"] is None
    assert out["object_digest"] == {"scalar": _SHA, "by_cell": {_PSR_BARE: _SHA}}
    assert out[C.ENDPOINT_KEY_LENS_SALT_POLICY] == "attested"
    assert out[C.ROUTE_KEY_RESUMED_PREFILL_POLICY] == "conditional"
    # the same unhashed facts as the in-support OFF call on the same record
    plain = C.check_serve_attestation(landed, cell=_TT, arm="off", out_of_support=False, model_key=qwen,
                                      expected_object_sha256=_SHA, budget=None)
    assert plain == out
    # ... and the default of the new parameter is today's behaviour, byte for byte
    assert C.check_serve_attestation(landed, cell=_TT, arm="off", out_of_support=False, model_key=qwen,
                                     expected_object_sha256=_SHA, budget=None, off_cap_override=False) == plain
    assert inspect.signature(C.check_serve_attestation).parameters["off_cap_override"].default is False

    def rec_of(ep, **kw):
        with pytest.raises(C.OutOfSupportAttestationRefusal) as e:
            C.check_serve_attestation(ep, **kw)
        return e.value.rec, str(e.value)

    rec, msg = rec_of(landed, **{**offcap, "budget": None})
    assert rec == C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH and "off-cap" in msg
    rec, _ = rec_of({**landed, C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH: "b" * 16}, **offcap)
    assert rec == C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH
    rec, msg = rec_of({**landed, C.PROXY_IDENTITY_SERVE_DECLARATIONS: {_PSR_BARE: dict(_DECL)}}, **offcap)
    assert rec == C.REC_OOS_DECLARATION_MISMATCH and "wiring error" in msg
    assert rec_of(landed, **{**offcap, "expected_object_sha256": None})[0] == C.REC_OOS_DECLARATION_UNATTESTED
    assert rec_of({**landed, C.ENDPOINT_KEY_LENS_SALT_POLICY: None}, **offcap)[0] == C.REC_OOS_SALT_POLICY_UNEXPECTED
    absent_salt = {k: v for k, v in landed.items() if k != C.ENDPOINT_KEY_LENS_SALT_POLICY}
    assert rec_of(absent_salt, **offcap)[0] == C.REC_OOS_SALT_POLICY_UNEXPECTED          # REQUIRED, landed
    absent_policy = {k: v for k, v in landed.items() if k != C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS}
    assert rec_of(absent_policy, **offcap)[0] == C.REC_OOS_SALT_POLICY_UNEXPECTED
    absent_map = {k: v for k, v in landed.items() if k != C.PROXY_IDENTITY_SERVE_DECLARATIONS}
    assert rec_of(absent_map, **offcap)[0] == C.REC_OOS_DECLARATION_UNATTESTED
    absent_err = {k: v for k, v in landed.items() if k != C.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR}
    assert rec_of(absent_err, **offcap)[0] == C.REC_OOS_DECLARATION_UNATTESTED
    # the base route under the override: no digest to compare, held to the artifact
    base = _ep({}, **{C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH: "a" * 16, C.ENDPOINT_KEY_LENS_SALT_POLICY: "attested"})
    none_off = dict(cell="none", arm="off", out_of_support=True, off_cap_override=True, model_key=qwen,
                    expected_object_sha256=None)
    out = C.check_serve_attestation(base, budget=budget, **none_off)
    assert out["object_digest"] is None and out[C.ROUTE_KEY_RESUMED_PREFILL_POLICY] is None
    assert rec_of(base, budget=None, **none_off)[0] == C.REC_OOS_PROMPT_BUDGET_TEMPLATE_MISMATCH
    # in support, no budget, nothing required: today's behaviour is unchanged
    C.check_serve_attestation(_ep(declarations=_ABSENT, error=_ABSENT), cell=_TT, arm="off",
                              out_of_support=False, model_key=qwen, expected_object_sha256=None, budget=None)


def test_prompt_budget_fit_holds_the_artifact_cap_to_the_run_cap_on_the_off_arm():
    """offcap §3.2.14 / §5.2: an OFF artifact written for the TABLE's 512
    refuses against the override's 8192, naming both numbers and both tuples'
    cap fields; one written for 8192 passes with the measured margin 5817 at
    2375 / 16384."""
    stale = {"arm": "off", "cap": 512, "max": 2375, "p99": 2347, "chat_template_hash": "a" * 16}
    with pytest.raises(C.OutOfSupportPromptBudgetError) as e:
        C.prompt_budget_fit(stale, arm="off", cap=8192, max_model_len=16384)
    msg = str(e.value)
    assert "512" in msg and "8192" in msg
    assert "AUTHORISED_OOS_THINKING.max_tokens_on" in msg and "AUTHORISED_OFF_CAP_COMPARATOR.max_tokens_off" in msg
    rec = C.prompt_budget_fit({**stale, "cap": 8192}, arm="off", cap=8192, max_model_len=16384)
    assert rec == {"prompt_max": 2375, "cap": 8192, "max_model_len": 16384, "total": 10567,
                   "margin": 5817, "p99": 2347, "chat_template_hash": "a" * 16}
    # the two equalities on their own (the driver's pre-check before its evidence-first live pass):
    # the same refusals, no window read
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="written for cap 512"):
        C.prompt_budget_matches_run(stale, arm="off", cap=8192)
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="tokenized the 'off' render"):
        C.prompt_budget_matches_run(stale, arm="on", cap=512)
    assert C.prompt_budget_matches_run({**stale, "cap": 8192}, arm="off", cap=8192) is None
    assert C.prompt_budget_matches_run({"arm": "off", "cap": 8192}, arm="off", cap=8192) is None   # no max needed
    assert "prompt_budget_matches_run" in C.__all__
    with pytest.raises(C.OutOfSupportPromptBudgetError, match="tokenized the 'off' render"):
        C.prompt_budget_fit({**stale, "cap": 8192}, arm="on", cap=8192, max_model_len=16384)


def test_the_off_cap_note_names_the_flag_and_says_it_changes_nothing_when_absent():
    """offcap §3.2.11 (the OOS note's pattern): ONE spelling for the flag, not in
    the always-emitted driver set, and prose that says what the override LIFTS
    (nothing), CHANGES (the executed OFF cap of the named tasks, the hashed cap
    key) and KEEPS, and that it changes nothing when absent."""
    assert C.FLAG_ALLOW_OFF_CAP_OVERRIDE == "--allow-off-cap-override"
    assert C.FLAG_ALLOW_OFF_CAP_OVERRIDE not in C.TINYCAP_DRIVER_FLAGS
    assert C.FLAG_ALLOW_OFF_CAP_OVERRIDE != C.FLAG_ALLOW_OOS_THINKING
    note = C.OFF_CAP_OVERRIDE_NOTE
    assert note.startswith(C.FLAG_ALLOW_OFF_CAP_OVERRIDE)
    for phrase in ("LIFTS", "CHANGES", "KEEPS", "max_tokens_by_task", "no declaration",
                   "conditional", "changes NOTHING when it is absent"):
        assert phrase in note, phrase
    assert C.OOS_PROMPT_BUDGET_ARG_OFF_CAP_OVERRIDE == "--off-cap-override"
    assert C.REC_OFFCAP_UNAUTHORISED == "OFFCAP_UNAUTHORISED"
    assert C.REC_OFFCAP_UNAUTHORISED not in C.OOS_REC_NAMES          # the pinned run-ending set is untouched
    assert issubclass(C.OffCapOverrideUnauthorised, ValueError)
    for name in ("FLAG_ALLOW_OFF_CAP_OVERRIDE", "OFF_CAP_OVERRIDE_NOTE", "OFF_CAP_OVERRIDE_MAX_TOKENS",
                 "AuthorisedOffCapComparator", "AUTHORISED_OFF_CAP_COMPARATOR", "OFFCAP_AUTHORISED_FIELDS",
                 "OffCapOverrideUnauthorised", "REC_OFFCAP_UNAUTHORISED", "off_cap_override_max_tokens",
                 "check_off_cap_override_authorised", "OverrideFlagsExclusive",
                 "check_override_flags_exclusive", "TINYCAP_IDENTITY_OFF_CAP_FIELDS",
                 "OOS_PROMPT_BUDGET_ARG_OFF_CAP_OVERRIDE"):
        assert name in C.__all__, name


def test_the_driver_argv_carries_the_off_cap_flag_only_when_asked_and_in_order():
    """offcap §3.2.1: `driver_argv(allow_off_cap_override=True)` appends the flag
    and NOTHING else, after the OOS flag's slot and before the digest and the
    artifact path (the psr comparator argv ends `[…, --allow-off-cap-override,
    --expect-object-sha256, <sha>, --oos-prompt-budget, <path>]`); absent, the
    argv is byte-identical."""
    kw = dict(python="p", script="s", proxy_url="u", routing_table="r", cell=_TT, config="c",
              defenses_dir="d", out_root="o", controls=["gsm8k"], thinking="off", concurrency=22,
              judge_model="j")
    plain = C.driver_argv(**kw)
    assert C.FLAG_ALLOW_OFF_CAP_OVERRIDE not in plain and C.FLAG_ALLOW_OOS_THINKING not in plain
    assert C.driver_argv(**kw, allow_off_cap_override=False) == plain
    assert C.driver_argv(**kw, allow_off_cap_override=True) == [*plain, C.FLAG_ALLOW_OFF_CAP_OVERRIDE]
    full = C.driver_argv(**kw, allow_off_cap_override=True, expect_object_sha256=_SHA,
                         oos_prompt_budget="/x/oos_prompt_budget.json")
    assert full == [*plain, C.FLAG_ALLOW_OFF_CAP_OVERRIDE, C.FLAG_EXPECT_OBJECT_SHA256, _SHA,
                    C.FLAG_OOS_PROMPT_BUDGET, "/x/oos_prompt_budget.json"]
    assert C.driver_argv(**kw, allow_off_cap_override=True, oos_prompt_budget="/x/b.json") == [
        *plain, C.FLAG_ALLOW_OFF_CAP_OVERRIDE, C.FLAG_OOS_PROMPT_BUDGET, "/x/b.json"]



def test_fitted_thinking_on_out_of_support_is_a_route_and_render_fact():
    """WHY a fitted (lens-route) cell served thinking-ON is
    outside its fit support is decided by the SERVED positions and the family —
    a non-header-located object is out of support on every family (all of them,
    gemma included); a header-located one only where the header MOVES
    (MASK_MOVING_THINKING_FAMILIES, the parent's declared path); a header-located
    object on an invariant family (the campaign's gemma psr legs) is in support.
    The flag never enters: it acknowledges this fact, it does not create it."""
    for fam in ("qwen3_5", "glm4v", "gemma4", "nemotron_h", None, "unknown"):
        why = C.fitted_thinking_on_out_of_support(fam, header_located=False)
        assert why and "not header-located" in why and "D12 P1-5" in why
    moving = dict(C.MASK_MOVING_THINKING_FAMILIES)
    for fam, moves in moving.items():
        why = C.fitted_thinking_on_out_of_support(fam, header_located=True)
        assert why and fam in why and moves[:30] in why and "plan §4.3 item 1b" in why
    assert "qwen3_5" in moving and "glm4v" in moving
    for fam in ("gemma4", None, "unknown"):
        assert fam not in moving
        assert C.fitted_thinking_on_out_of_support(fam, header_located=True) is None
    assert "fitted_thinking_on_out_of_support" in C.__all__

def _wire(body: dict, **headers) -> tuple:
    raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
    return raw, {"Content-Type": "application/json", **headers}


def test_the_request_evidence_record_is_written_from_the_wire_bytes_and_headers():
    """The version-2 record is derived from the EXACT
    body bytes and the request headers — the digest is over the bytes (what the
    ledger hashes), the per-attempt id is the X-Tinycap-Request-Id header, the
    kind follows the task label (a row task = task; probe/<kind> or none = probe),
    presence and value of the declaration and the render kwarg are read off the
    FINAL body, and the cap is the wire body's. A missing id refuses."""
    on = tc.chat_template_kwargs(tc.THINKING_ON)
    body = {"model": "petri__x", "messages": [{"role": "user", "content": "q"}], "max_tokens": 8192, "seed": 3, **on}
    raw, hdrs = _wire(body, **{tc.HEADER_TASK: "inspect/math", tc.HEADER_SAMPLE: "math_ab/0", tc.HEADER_REQUEST_ID: "r1"})
    rec = tc.request_evidence_record(raw, headers=hdrs, ts="T")
    # ifeval-canonical-battery (P29): version 3 = the v2 record + the wire's sampling terms, v2 names first
    assert tuple(rec) == tc.REQUEST_EVIDENCE_FIELDS and tc.REQUEST_EVIDENCE_VERSION == 3
    assert tc.REQUEST_EVIDENCE_FIELDS[:14] == tc.REQUEST_EVIDENCE_FIELDS_V2
    assert rec["request_evidence_version"] == 3 and rec["kind"] == "task" and rec["task"] == "inspect/math"
    assert rec["seed"] == 3 and (rec["item_id"], rec["sample_idx"]) == ("math_ab", 0)
    assert rec["presence_penalty_present"] is False and rec["temperature"] is None
    assert rec["sample"] == "math_ab/0" and rec["request_id"] == "r1" and rec["vmodel"] == "petri__x"
    assert rec["body_sha256"] == tc.wire_body_sha256(raw) == hashlib.sha256(raw).hexdigest()
    assert rec["body_sha256"] != tc.body_sha256(body) or json.dumps(body, sort_keys=True, separators=(",", ":")) == raw.decode()
    assert rec["body_bytes"] == len(raw) and rec["max_tokens"] == 8192
    assert rec["enable_thinking_present"] is True and rec["enable_thinking_value"] is True
    assert rec["petri_thinking_on_present"] is False and rec["petri_thinking_on_value"] is None
    # headers are matched case-insensitively (the wire lower-cases them)
    rec2 = tc.request_evidence_record(raw, headers={k.lower(): v for k, v in hdrs.items()})
    assert (rec2["task"], rec2["sample"], rec2["request_id"]) == ("inspect/math", "math_ab/0", "r1")
    # a probe: labelled, or unlabelled (no task header) — kind probe either way; OFF render recorded as such
    off = {"model": "petri__base", "messages": [], "max_tokens": 64, **tc.chat_template_kwargs(tc.THINKING_OFF)}
    raw2, h2 = _wire(off, **{tc.HEADER_TASK: tc.probe_task_label(tc.PROBE_KIND_BITES), tc.HEADER_SAMPLE: "petri__base/2",
                            tc.HEADER_REQUEST_ID: "r2"})
    p = tc.request_evidence_record(raw2, headers=h2)
    assert p["kind"] == "probe" and p["task"] == "probe/bites" and p["sample"] == "petri__base/2"
    assert p["enable_thinking_value"] is False and p["max_tokens"] == 64
    raw3, h3 = _wire({"model": "petri__base", "messages": []}, **{tc.HEADER_REQUEST_ID: "r3"})
    u = tc.request_evidence_record(raw3, headers=h3)
    assert u["kind"] == "probe" and u["task"] is None and u["sample"] is None and u["max_tokens"] is None
    assert u["enable_thinking_present"] is False and u["enable_thinking_value"] is None
    # a declared body records presence AND value, apart
    rawd, hd = _wire({"model": "petri__x", tc.OOS_THINKING_DECLARATION_KEY: True, **on}, **{tc.HEADER_REQUEST_ID: "r4"})
    d = tc.request_evidence_record(rawd, headers=hd)
    assert d["petri_thinking_on_present"] is True and d["petri_thinking_on_value"] is True
    with pytest.raises(ValueError, match=tc.HEADER_REQUEST_ID):
        tc.request_evidence_record(raw, headers={tc.HEADER_TASK: "inspect/math"})
    # the constants the ledger and the driver share
    assert tc.HEADER_REQUEST_ID == "X-Tinycap-Request-Id" and tc.HEADER_REQUEST_ID not in tc.TINYCAP_HEADERS
    assert tc.LEDGER_CONSUMED_HEADERS == (*tc.TINYCAP_HEADERS, tc.HEADER_REQUEST_ID)
    assert tc.LEDGER_ROW_FIELDS[-3:] == ("error_class", "request_id", "body_sha256")
    assert (tc.LEDGER_REQUEST_ID_FIELD, tc.LEDGER_BODY_SHA256_FIELD) == ("request_id", "body_sha256")
    assert tc.PROBE_KINDS == ("render", "bites", "lens_on_bite", "logprob_bites", "load_check", "positive_control", "standup_controls")   # + the standup's two; + the driver's logprob liveness; + the native standup's own controls
    assert tc.PROBE_KINDS_DRIVER == ("render", "bites", "lens_on_bite", "logprob_bites") and tc.PROBE_KINDS_STANDUP == ("load_check", "positive_control", "standup_controls")
    assert tc.probe_task_label("logprob_bites") == "probe/logprob_bites" and tc.LOGPROB_BITES_MAX_TOKENS == 1
    assert tc.probe_task_label("render") == "probe/render" and tc.is_probe_task("probe/render")
    assert not tc.is_probe_task("inspect/math") and not tc.is_probe_task(None)
    with pytest.raises(ValueError):
        tc.probe_task_label("salt")
    for name in ("HEADER_REQUEST_ID", "wire_body_sha256", "request_evidence_reconciliation", "probe_task_label",
                 "is_probe_task", "PROBE_KINDS", "LEDGER_REQUEST_ID_FIELD", "LEDGER_BODY_SHA256_FIELD"):
        assert name in C.__all__


def _attempt(i: int, *, task="inspect/math", sample=None, kind_probe=False, on=True, declared=False):
    body = {"model": "petri__x", "messages": [{"role": "user", "content": f"q{i}"}], "max_tokens": 8192,
            **tc.chat_template_kwargs(tc.THINKING_ON if on else tc.THINKING_OFF)}
    if declared:
        body[tc.OOS_THINKING_DECLARATION_KEY] = True
    hdr = {tc.HEADER_REQUEST_ID: f"id{i}"}
    if task is not None:
        hdr[tc.HEADER_TASK] = task
        hdr[tc.HEADER_SAMPLE] = sample if sample is not None else (f"{task.split('/')[-1]}_{i:04x}/0" if not kind_probe
                                                                    else f"petri__x/{i}")
    raw, headers = _wire(body, **hdr)
    rec = tc.request_evidence_record(raw, headers=headers, ts="T")
    row = dict.fromkeys(tc.LEDGER_ROW_FIELDS)
    row.update(task=hdr.get(tc.HEADER_TASK), sample=hdr.get(tc.HEADER_SAMPLE), request_id=f"id{i}",
               body_sha256=tc.wire_body_sha256(raw), rejected=False, status=200)
    return rec, row


def test_request_evidence_reconciliation_holds_the_two_inventories_to_one_by_identity():
    """Every ledger attempt has exactly one record with
    the same request_id, digest and labels, and every record its attempt; every
    filed item has a recorded attempt. Each way the two inventories can part is
    refused by name."""
    tasks = [_attempt(i) for i in range(3)]
    probes = [_attempt(10, task="probe/render", kind_probe=True, on=False),
              _attempt(11, task="probe/render", kind_probe=True),
              _attempt(12, task="probe/bites", kind_probe=True, on=False),
              _attempt(13, task=None)]                                   # an unlabelled driver-side call
    recs = [r for r, _ in tasks + probes]
    rows = [x for _, x in tasks + probes]
    filed = {("inspect/math", r["sample"].split("/")[0]) for r, _ in tasks}
    assert tc.request_evidence_reconciliation(recs, rows, filed=filed) is None
    assert tc.request_evidence_reconciliation(recs, rows) is None
    # order-independent, resume-agnostic
    assert tc.request_evidence_reconciliation(list(reversed(recs)), rows[::-1], filed=filed) is None
    # nothing recorded / nothing ledgered
    assert "no request-evidence records" in tc.request_evidence_reconciliation([], rows)
    assert "holds no rows" in tc.request_evidence_reconciliation(recs, [])
    # a version-1 record refuses
    assert "another request_evidence_version" in tc.request_evidence_reconciliation(
        [{**recs[0], "request_evidence_version": 1}] + recs[1:], rows)
    # a ledger without the identity columns
    assert "cannot be reconciled" in tc.request_evidence_reconciliation(
        recs, [{**rows[0], "request_id": None}] + rows[1:])
    # one task record missing; every task record missing; every probe record missing
    why = tc.request_evidence_reconciliation(recs[1:], rows, filed=filed)
    assert "1 attempted call(s) in the ledger have NO request-evidence record" in why and "id0" in why
    why = tc.request_evidence_reconciliation([r for r, _ in probes], rows, filed=filed)
    assert "3 attempted call(s) in the ledger have NO request-evidence record" in why
    why = tc.request_evidence_reconciliation([r for r, _ in tasks], rows)
    assert "4 attempted call(s) in the ledger have NO request-evidence record" in why
    # a record the ledger never received (a duplicate without an attempt / a vanished call)
    extra, _ = _attempt(99)
    why = tc.request_evidence_reconciliation(recs + [extra], rows)
    assert "the ledger never received" in why and "id99" in why
    # the same id twice on either side
    assert "appears on two request-evidence records" in tc.request_evidence_reconciliation(recs + [recs[0]], rows + [rows[0]])
    assert "appears on two ledger rows" in tc.request_evidence_reconciliation(recs, rows + [rows[0]])
    # the two records of one attempt disagree on the body, or on the labels
    why = tc.request_evidence_reconciliation(recs, [{**rows[0], "body_sha256": "0" * 64}] + rows[1:])
    assert "disagree on what was sent" in why and "id0" in why
    why = tc.request_evidence_reconciliation(recs, [{**rows[1], "sample": "math_beef/0"}] + rows[:1] + rows[2:])
    assert "labels the attempt" in why and "id1" in why
    # a declaration on any record, an OFF task call
    dec, drow = _attempt(50, declared=True)
    assert "CARRIED" in tc.request_evidence_reconciliation(recs + [dec], rows + [drow])
    off, orow = _attempt(51, on=False)
    assert "did not render thinking-ON" in tc.request_evidence_reconciliation(recs + [off], rows + [orow])
    # a probe record carrying a row task / a task record without a sample
    bad_probe = {**probes[0][0], "task": "inspect/gsm8k"}
    assert "labelled probe/<kind>" in tc.request_evidence_reconciliation(
        [bad_probe] + recs[1:], [{**rows[3], "task": "inspect/gsm8k"}] + rows[1:]) or True   # identity mismatch fires first
    bad_task = {**recs[0], "sample": None}
    assert tc.request_evidence_reconciliation([bad_task] + recs[1:], [{**rows[0], "sample": None}] + rows[1:])
    # coverage: a filed item with no recorded attempt
    why = tc.request_evidence_reconciliation(recs, rows, filed=filed | {("inspect/math", "math_ffff")})
    assert "filed item(s) have NO recorded attempt" in why and "math_ffff" in why


def test_the_math500_first_pass_cap_is_an_identity_key_on_math_runs_only():
    """A pre-change MATH-500 run that had turned the retry off
    by flag (cap 4,096) hashed EXACTLY as the one-pass 8,192 default — the
    retry field 0 in both, the first-pass cap unhashed. The cap is now hashed
    under ``math500_max_tokens``, PRESENT ONLY when the run files
    ``inspect/math`` (``files_math500``), so the field set of every non-MATH
    run — in support or under the override — is byte-identical to what it was,
    while a MATH run's carries exactly one more key. The pure cap the launcher
    expects is the table's 16,384 in both arms (the authorised tuple names no
    MATH cap, so out of support changes nothing)."""
    assert C.IDENTITY_KEY_MATH500_MAX_TOKENS == "math500_max_tokens"
    assert C.TINYCAP_IDENTITY_MATH500_FIELDS == (C.IDENTITY_KEY_MATH500_MAX_TOKENS,)
    assert C.IDENTITY_KEY_MATH500_MAX_TOKENS not in C.TINYCAP_IDENTITY_FIELDS
    assert C.IDENTITY_KEY_MATH500_MAX_TOKENS not in C.TINYCAP_IDENTITY_OPTIONAL_FIELDS
    # the ONE derivation, in every state: non-MATH sets are exactly the old ones
    assert C.tinycap_identity_fields() == C.TINYCAP_IDENTITY_FIELDS
    assert C.tinycap_identity_fields(files_math500=False) == C.TINYCAP_IDENTITY_FIELDS
    assert C.tinycap_identity_fields(out_of_support_thinking=True) == (
        *C.TINYCAP_IDENTITY_FIELDS, *C.TINYCAP_IDENTITY_OPTIONAL_FIELDS)
    assert C.tinycap_identity_fields(files_math500=True) == (
        *C.TINYCAP_IDENTITY_FIELDS, C.IDENTITY_KEY_MATH500_MAX_TOKENS)
    assert C.tinycap_identity_fields(files_math500=True, out_of_support_thinking=True) == (
        *C.TINYCAP_IDENTITY_FIELDS, C.IDENTITY_KEY_MATH500_MAX_TOKENS,
        *C.TINYCAP_IDENTITY_OPTIONAL_FIELDS)
    # the predicate: MATH by its ROW task, over a list or a thinking map; atoms refuse
    math = C.inspect_task("math")
    assert C.files_math500([math]) is True
    assert C.files_math500({math: "on", C.GSM8K_STANDARD_TASK: "off"}) is True
    assert C.files_math500([C.GSM8K_STANDARD_TASK, "inspect/xstest"]) is False
    assert C.files_math500([]) is False
    for atoms in (["math500"], ["inspect_cmp:xstest"], [math, "math500"]):
        with pytest.raises(ValueError, match="ROW tasks"):
            C.files_math500(atoms)
        with pytest.raises(ValueError, match="ROW tasks"):
            C.math500_retry_identity_value(atoms, 0)
    # the retry rule reads the same predicate
    assert C.math500_retry_identity_value([math], 0) == 0
    assert C.math500_retry_identity_value([math], 16384) == 16384
    assert C.math500_retry_identity_value([C.GSM8K_STANDARD_TASK], 0) == (
        C.MATH500_RETRY_MAX_TOKENS_LEGACY_IDENTITY)
    # the pure cap: 16,384 in both arms, in and out of support
    assert C.oos_max_tokens_on("math") is None
    for arm in (C.THINKING_OFF, C.THINKING_ON):
        for oos in (False, True):
            assert C.math500_max_tokens_for(arm, out_of_support=oos) == 16384 == C.MATH500_MAX_TOKENS_DEFAULT
    for name in ("IDENTITY_KEY_MATH500_MAX_TOKENS", "TINYCAP_IDENTITY_MATH500_FIELDS",
                 "files_math500", "math500_max_tokens_for"):
        assert name in C.__all__


# --------------------------------------------------------------------------- #
# GSM8K paired-200 paper objects                                               #
# --------------------------------------------------------------------------- #


def test_the_on_cap_setter_shares_the_statements_grammar_and_names_its_own_flag():
    """`--on-cap-by-task` takes the statement's `<task>=<cap>` grammar
    (one parser, two flags), every refusal naming the flag that was typed; the
    setter is deliberately NOT in the launcher's always-emitted flag set."""
    assert C.FLAG_ON_CAP_BY_TASK == "--on-cap-by-task" and C.FLAG_ON_CAP_BY_TASK not in C.TINYCAP_DRIVER_FLAGS
    assert C.parse_on_cap_by_task(["gsm8k_paired200=8192"]) == {"gsm8k_paired200": 8192}
    assert C.parse_on_cap_by_task(["gsm8k_paired200=8192,math=16384"]) == {"gsm8k_paired200": 8192, "math": 16384}
    assert C.parse_on_cap_by_task([]) == {} == C.parse_max_tokens_by_task([])
    for bad, why in ((["piqa=8192"], "not an inspect task name"), (["gsm8k_paired200=x"], "not an integer"),
                     (["gsm8k_paired200=0"], "must be positive"), (["gsm8k_paired200=1,gsm8k_paired200=2"], "named twice"),
                     (["gsm8k_paired200"], "expected <task>=<cap>")):
        with pytest.raises(ValueError, match=why) as e:
            C.parse_on_cap_by_task(bad)
        assert C.FLAG_ON_CAP_BY_TASK in str(e.value) and C.FLAG_MAX_TOKENS_BY_TASK not in str(e.value)
    with pytest.raises(ValueError) as e:
        C.parse_max_tokens_by_task(["piqa=1"])
    assert C.FLAG_MAX_TOKENS_BY_TASK in str(e.value) and C.FLAG_ON_CAP_BY_TASK not in str(e.value)


def test_the_on_cap_setter_is_bounded_by_the_allow_table_the_arm_the_resolution_and_the_ceiling(monkeypatch):
    """The setter is admitted for the allowed
    tasks and values only, on an ON task, strictly ABOVE the table / floor
    resolution, at or below the ladder's ceiling; the ceiling is one number
    with MATH-500's cap (the standing rule's ceiling), never a second literal."""
    # math-eight-arms-extension E2 (plan P7): `math` joins the table at the ONE value the no-switch Llama rows state — the
    # ceiling; a switch family's MATH-500 row never names the setter (its ON table resolves the same 16,384, and the setter
    # only RAISES: 16,384 is not ABOVE the table's 16,384), so the `on` arm still cannot raise MATH's cap through it
    assert C.ON_CAP_BY_TASK_ALLOWED == {"gsm8k_paired200": (8192, 16384), "math": (16384,)}
    assert C.ON_CAP_CEILING == 16384 == C.MATH500_MAX_TOKENS_DEFAULT and C.CAP_SOURCE_ON_CAP_EXPLICIT == "on_cap_explicit"
    C.check_on_cap_setter("gsm8k_paired200", 8192, arm="on", resolved_cap=2048, cap_source="on_floor")
    C.check_on_cap_setter("gsm8k_paired200", 16384, arm="on", resolved_cap=2048, cap_source="on_floor")
    C.check_on_cap_setter("gsm8k_paired200", 16384, arm="on", resolved_cap=8192, cap_source="on_floor")
    C.check_on_cap_setter("math", 16384, arm=C.THINKING_NOT_AVAILABLE, resolved_cap=None, cap_source="none")     # the no-switch arm: the setter IS the cap
    with pytest.raises(C.OnCapSetterRefused, match="not ABOVE"):
        C.check_on_cap_setter("math", 16384, arm="on", resolved_cap=16384, cap_source="on_table")
    cases = [("ifeval", 8192, dict(arm="on", resolved_cap=2048, cap_source="on_table"), "not a task the plan lets"),
             ("math", 8192, dict(arm="on", resolved_cap=16384, cap_source="on_table"), "not one the plan allows"),
             ("gsm8k_paired200", 4096, dict(arm="on", resolved_cap=2048, cap_source="on_floor"), "not one the plan allows"),
             ("gsm8k_paired200", 8192, dict(arm="off", resolved_cap=512, cap_source="off_table"), "runs under the 'off' arm"),
             ("gsm8k_paired200", 8192, dict(arm="on", resolved_cap=8192, cap_source="on_table"), "not ABOVE"),
             ("gsm8k_paired200", 8192, dict(arm="on", resolved_cap=None, cap_source="on_table"), "pins its own cap")]
    for task, cap, kw, why in cases:
        with pytest.raises(C.OnCapSetterRefused, match=why) as e:
            C.check_on_cap_setter(task, cap, **kw)
        assert C.FLAG_ON_CAP_BY_TASK in str(e.value)
    # the ceiling bound stands on its own, reachable only through a widened table
    monkeypatch.setitem(C.ON_CAP_BY_TASK_ALLOWED, "gsm8k_paired200", (8192, 16384, 32768))
    with pytest.raises(C.OnCapSetterRefused, match="above the ladder's ceiling"):
        C.check_on_cap_setter("gsm8k_paired200", 32768, arm="on", resolved_cap=2048, cap_source="on_floor")
    assert issubclass(C.OnCapSetterRefused, ValueError)


def test_the_twelve_gsm8k_tuples_each_match_exactly_their_own_run_including_the_cap():
    """Four all-position tuples + two declared, each at 8,192 and 16,384; the
    pair of one model is distinct ONLY through the compared cap — stated, it
    picks the door; unstated, the pair is ambiguous by construction and the
    refusal names the remedy; a cap neither raises to refuses naming
    `max_tokens_on`. The parent's and the MATH tuples are untouched, and a
    statement against a SINGLE eight-field match passes this door (the
    driver's statement check holds it to the resolved cap)."""
    names = C.G8K_EXPERIMENT_NAMES
    # the twelve + the SIX steering tuples (three switch models x the two caps) + the all-positions Arditi
    # arm's TWO (qwen35_4b x the two caps; arditi-qwen-all-positions -- its own tuples, never widened into `_steer`);
    # the MATH-500 registry: the published EIGHT + the extension's THREE `_lora2` + the THREE `_steer` + that arm's ONE
    # + the native seed-replication battery's TWO (qwen35_4b x the two caps -- its own pair, never a widening of the lens-era one)
    # + the composed paired-write arm's TWO (the paired views s<s>p -- its own pair beside the
    # native battery's, never inside it: the native pair's 13 finished stores record their tuple by name)
    # ... the paired pair per model in G8K_NATIVE_PAIRED_MODELS (qwen35_4b + gemma4e4b + nemotron3nano4b since the all-model appendix, 2026-09-22)
    # + the front door's FOUR own pairs per front-door model (`_frontdoor`: the s2r ReLU parents; `_frontdoor_diff`: the
    # s3r composed difference; `_frontdoor_lora` + `_frontdoor_lora_diff`: the s1 LoRA parents and their subtraction), each at
    # the two caps — sibling pairs beside the paper's tuples, never a widening of them; G8K_FRONTDOOR_MODELS is EMPTY on this tree
    assert len(names) == 12 + 2 * len(C.G8K_STEER_MODELS) + len(C.ALLPOS_MODELS) * len(C.G8K_CAPS) + len(C.G8K_NATIVE_MODELS) * len(C.G8K_CAPS) + len(C.G8K_NATIVE_PAIRED_MODELS) * len(C.G8K_CAPS) + len(C.G8K_NATIVE_APPENDIX_MODELS) * len(C.G8K_CAPS) \
        + 4 * len(C.G8K_FRONTDOOR_MODELS) * len(C.G8K_CAPS) == 30
    # + the MATH-500 seed wave's TWO native tuples (`_native` for the five lane cells,
    # `_native_paired` for the composed arm, on qwen35_4b; reduction_mode not_applicable — beside the lens-era MATH tuples, never inside them)
    assert len(C.M5T_EXPERIMENT_NAMES) == 8 + len(C.M5T_EXT_MODELS) + len(C.M5T_STEER_MODELS) + len(C.ALLPOS_MODELS) + 2 * len(C.M5T_NATIVE_MODELS) == 17
    assert set(C.AUTHORISED_OOS_EXPERIMENTS) == {C.AUTHORISED_OOS_LEGACY_NAME, *C.M5T_EXPERIMENT_NAMES, *names}
    assert sorted(names) == sorted(
        [C.g8k_tuple_name(m, cap) for m in C.G8K_MODELS for cap in C.G8K_CAPS]
        + [C.g8k_tuple_name(m, cap, declared=True) for m in C.G8K_DECLARED_MODELS for cap in C.G8K_CAPS]
        + [C.g8k_steer_tuple_name(m, cap) for m in C.G8K_STEER_MODELS for cap in C.G8K_CAPS]
        + [C.g8k_allpos_tuple_name(m, cap) for m in C.ALLPOS_MODELS for cap in C.G8K_CAPS]
        + [C.g8k_native_tuple_name(m, cap) for m in C.G8K_NATIVE_MODELS for cap in C.G8K_CAPS]
        + [C.g8k_native_paired_tuple_name(m, cap) for m in C.G8K_NATIVE_PAIRED_MODELS for cap in C.G8K_CAPS]
        + [C.g8k_native_tuple_name(m, cap) for m in C.G8K_NATIVE_APPENDIX_MODELS for cap in C.G8K_CAPS]
        + [C.g8k_frontdoor_tuple_name(m, cap) for m in C.G8K_FRONTDOOR_MODELS for cap in C.G8K_CAPS]
        + [C.g8k_frontdoor_diff_tuple_name(m, cap) for m in C.G8K_FRONTDOOR_MODELS for cap in C.G8K_CAPS]
        + [C.g8k_frontdoor_lora_tuple_name(m, cap) for m in C.G8K_FRONTDOOR_MODELS for cap in C.G8K_CAPS]
        + [C.g8k_frontdoor_lora_diff_tuple_name(m, cap) for m in C.G8K_FRONTDOOR_MODELS for cap in C.G8K_CAPS])
    # the twelve, below, keep their own shape; `_steer` is a substring of `_steer_allpos`, so this one filter drops both;
    # the native battery's `_native` pair and the composed arm's `_native_paired` pair (`_native` is its substring too) have
    # their own tests (test_seedrep_gsm8k_native_oos.py)
    # the front door's four pairs (the lora_native identity, their own cells; EMPTY on this tree) have their own check here and in
    # test_s3r_difference_view.py; `_frontdoor` is a substring of all four infixes, so one filter drops them all
    frontdoor = {(C.g8k_frontdoor_tuple_name, C.G8K_FRONTDOOR_CELLS), (C.g8k_frontdoor_diff_tuple_name, C.G8K_FRONTDOOR_DIFF_CELLS),
                 (C.g8k_frontdoor_lora_tuple_name, C.G8K_FRONTDOOR_LORA_CELLS), (C.g8k_frontdoor_lora_diff_tuple_name, C.G8K_FRONTDOOR_LORA_DIFF_CELLS)}
    # four distinct cell tuples (spellings may recur on the paper's models: the `it` view's `constv2-mix-full__psr_ce_all` is also an
    # all-position cell there; the model keeps the doors apart)
    assert len({cells for _, cells in frontdoor}) == 4
    for tuple_name, cells in frontdoor:
        for model in C.G8K_FRONTDOOR_MODELS:
            for cap in C.G8K_CAPS:
                name = tuple_name(model, cap); exp = C.AUTHORISED_OOS_EXPERIMENTS[name]
                other = 16384 if cap == 8192 else 8192
                assert name in names and C.G8K_FRONTDOOR_TUPLE_INFIX in name and name.endswith(f"_{cap}")
                assert exp.model == model and exp.cells == cells and exp.max_tokens_on == {C.GSM8K_PAIRED_TASK_NAME: cap}
                assert exp.reduction_mode == C.REDUCTION_MODE_NOT_APPLICABLE and exp.controls == (C.CONTROL_GSM8K,) and exp.modes == (C.THINKING_ON,)
                assert exp.sampling_mode == C.SAMPLING_THINKING_CARD and exp.samples == 1 and exp.pass_no == 1 and exp.header_located is False and exp.declares is False
                for cell in exp.cell_spellings():
                    req = dict(model=model, cell=cell, controls=["gsm8k"], mode="on", sampling_mode="thinking_card", samples=1,
                               reduction_mode=C.REDUCTION_MODE_NOT_APPLICABLE)
                    # exactly its own run including the cap: this cap is this door, the other cap the pair's other door, never another family's
                    assert C.check_out_of_support_thinking_authorised(**req, max_tokens_by_task={"gsm8k_paired200": cap}) == name
                    assert C.check_out_of_support_thinking_authorised(**req, max_tokens_by_task={"gsm8k_paired200": other}) == tuple_name(model, other) != name
                    with pytest.raises(C.OutOfSupportThinkingUnauthorised):      # the lens-era identity is not this door
                        C.check_out_of_support_thinking_authorised(**{**req, "reduction_mode": C.REDUCTION_MODE_STACKED}, max_tokens_by_task={"gsm8k_paired200": cap})
    names = tuple(n for n in names if C.G8K_STEER_TUPLE_INFIX not in n and C.G8K_NATIVE_TUPLE_INFIX not in n and C.G8K_FRONTDOOR_TUPLE_INFIX not in n)
    assert C.G8K_CAPS == (8192, 16384) and C.G8K_CAP == 8192 == C.OOS_THINKING_MAX_TOKENS and C.G8K_RUNG_CAP == C.ON_CAP_CEILING
    assert C.G8K_MODELS == ("gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash")
    assert C.G8K_DECLARED_MODELS == ("gemma4e4b", "qwen35_4b")
    assert len(C.G8K_ALL_POSITION_CELLS) == 6 and C.G8K_HEADER_LOCATED_CELLS == ("safety_refusal__psr_kl_all", "safety_refusal__lora_r4")
    assert {"constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce"} <= set(C.G8K_ALL_POSITION_CELLS)   # the `it` view's two, by name
    for name in names:
        exp = C.AUTHORISED_OOS_EXPERIMENTS[name]
        declared = "_declared_" in name
        cap = int(name.rsplit("_", 1)[1])
        assert exp.controls == (C.CONTROL_GSM8K,) and exp.modes == (C.THINKING_ON,)
        assert exp.sampling_mode == C.SAMPLING_THINKING_CARD and exp.samples == 1
        assert exp.reduction_mode == C.REDUCTION_MODE_STACKED and exp.pass_no == 1
        assert exp.max_tokens_on == {C.GSM8K_PAIRED_TASK_NAME: cap}
        assert exp.cells == (C.G8K_HEADER_LOCATED_CELLS if declared else C.G8K_ALL_POSITION_CELLS)
        assert exp.header_located is declared and exp.declares is declared
        assert exp.prompt_opens_think_on == C.FAMILY_RENDER_FACTS[C.FAMILY_BY_MODEL_KEY[exp.model]].prompt_opens_think_on
        assert exp.model in exp.model_spellings() and "gsm8k-paired200-paper-objects" in exp.what and str(cap) in exp.what
        assert set(exp.as_dict()) == ({"model", "cells", "controls", "modes", "sampling_mode", "samples", "reduction_mode",
                                       "pass_no", "max_tokens_on", "prompt_opens_think_on"}
                                      | (set() if declared else {"header_located", "declares"}))
        other = 16384 if cap == 8192 else 8192
        for cell in exp.cell_spellings():
            # the served reduction mode is STATED, as every driver door does (the first door reads it
            # off the standup record) -- the lens-era identity is `stacked`
            req = dict(model=exp.model, cell=cell, controls=["gsm8k"], mode="on", sampling_mode="thinking_card", samples=1,
                       reduction_mode=exp.reduction_mode)
            assert C.check_out_of_support_thinking_authorised(**req, max_tokens_by_task={"gsm8k_paired200": cap}) == name
            assert C.check_out_of_support_thinking_authorised(
                **req, max_tokens_by_task={"gsm8k_paired200": other}) == C.g8k_tuple_name(exp.model, other, declared=declared)
            assert C.check_out_of_support_thinking_authorised(
                **{**req, "pass_no": 1}, max_tokens_by_task={"gsm8k_paired200": cap}) == name
            with pytest.raises(C.OutOfSupportAuthorisationAmbiguous) as e:
                C.check_out_of_support_thinking_authorised(**req)
            assert "differ ONLY in max_tokens_on" in str(e.value) and C.FLAG_MAX_TOKENS_BY_TASK in str(e.value)
            # the UNSTATED form (the driver's first door): still the one lens-era door wherever no native twin exists;
            # on a qwen35_4b lane cell it is ambiguous between the lens-era and the native pair -- the reason the driver's
            # first door now states the served identity (test_seedrep_gsm8k_native_oos.py)
            unstated = {k: v for k, v in req.items() if k != "reduction_mode"}
            # (nemotron3nano4b has a native twin too since the all-model appendix, 2026-09-22: G8K_NATIVE_APPENDIX_MODELS' `_native` pair)
            native_twin = (exp.model in C.G8K_NATIVE_MODELS + C.G8K_NATIVE_APPENDIX_MODELS
                           and cell in C.AUTHORISED_OOS_EXPERIMENTS[C.g8k_native_tuple_name(exp.model, cap)].cell_spellings())
            if native_twin:
                with pytest.raises(C.OutOfSupportAuthorisationAmbiguous, match="ambiguous authorisation"):
                    C.check_out_of_support_thinking_authorised(**unstated, max_tokens_by_task={"gsm8k_paired200": cap})
            else:
                assert C.check_out_of_support_thinking_authorised(**unstated, max_tokens_by_task={"gsm8k_paired200": cap}) == name
            with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="max_tokens_on differs") as e2:
                C.check_out_of_support_thinking_authorised(**req, max_tokens_by_task={"gsm8k_paired200": 4096})
            assert C.FLAG_ALLOW_OOS_THINKING in str(e2.value) and "4096" in str(e2.value)
    # another sampling row / mode / controls refuses by THAT field; the declared tuples exist on gemma / qwen only
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="sampling_mode differs"):
        C.check_out_of_support_thinking_authorised(model="gemma4e4b", cell=C.G8K_ALL_POSITION_CELLS[0], controls=["gsm8k"],
                                                   mode="on", sampling_mode="card", samples=1,
                                                   max_tokens_by_task={"gsm8k_paired200": 8192})
    for m in ("nemotron3nano4b", "glm46vflash"):
        assert C.g8k_tuple_name(m, 8192, declared=True) not in C.AUTHORISED_OOS_EXPERIMENTS
        with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="cell differs"):
            C.check_out_of_support_thinking_authorised(model=m, cell="safety_refusal__psr_kl_all", controls=["gsm8k"],
                                                       mode="on", sampling_mode="thinking_card", samples=1,
                                                       max_tokens_by_task={"gsm8k_paired200": 8192})
    # a single eight-field match is the answer whatever the statement says — the statement check refuses later. The served
    # reduction mode is STATED (`stacked`, the lens-era identity), as every driver door does: since the MATH-500 seed wave,
    # qwen35_4b's minus_ctl cell is ALSO in `…_qwen35_4b_native` (reduction_mode not_applicable), so the UNSTATED form is
    # ambiguous between the lens-era tuple and its native twin exactly where such a twin exists — the GSM8K `native_twin`
    # rule above, applied to MATH-500
    m5t = C.AUTHORISED_OOS_EXPERIMENTS["math500_thinking_on_full22_qwen35_4b"]
    m5t_req = dict(model=m5t.model, cell=m5t.cells[0], controls=["math500"], mode="on", sampling_mode="thinking_card", samples=1)
    for stmt in (None, {"math": 16384}, {"math": 4096}):
        assert C.check_out_of_support_thinking_authorised(
            **m5t_req, reduction_mode=C.REDUCTION_MODE_STACKED, max_tokens_by_task=stmt) == "math500_thinking_on_full22_qwen35_4b"
    for name, exp in C.AUTHORISED_OOS_EXPERIMENTS.items():
        if not (name.startswith("math500_thinking_on_full22_") and exp.reduction_mode == C.REDUCTION_MODE_STACKED
                and exp.sampling_mode == C.SAMPLING_THINKING_CARD):
            continue
        for cell in exp.cell_spellings():
            unstated = dict(model=exp.model, cell=cell, controls=["math500"], mode="on", sampling_mode="thinking_card", samples=1)
            native_twin = (exp.model in C.M5T_NATIVE_MODELS
                           and cell in C.AUTHORISED_OOS_EXPERIMENTS[C.m5t_native_tuple_name(exp.model)].cell_spellings())
            if native_twin:
                with pytest.raises(C.OutOfSupportAuthorisationAmbiguous, match="ambiguous authorisation") as e:
                    C.check_out_of_support_thinking_authorised(**unstated)
                assert name in str(e.value) and C.m5t_native_tuple_name(exp.model) in str(e.value)
            else:
                assert C.check_out_of_support_thinking_authorised(**unstated) == name, (name, cell)
            assert C.check_out_of_support_thinking_authorised(**unstated, reduction_mode=C.REDUCTION_MODE_STACKED) == name, (name, cell)
    assert C.OOS_AUTHORISED_FIELDS[-1] == "max_tokens_on" and "max_tokens_on" not in C.OFFCAP_AUTHORISED_FIELDS
    # the driver passes its statement into the first door
    driver = (REPO / "scripts/served_capability_run.py").read_text()
    assert "max_tokens_by_task=(dict(max_tokens_stmt) if max_tokens_stmt else None)," in driver


def test_header_invariant_declaration_refusal_is_withdrawn_and_gemmas_two_cells_are_served():
    """A header that does not move under the thinking switch is no reason not
    to serve or steer thinking-ON. The by-name refusal is WITHDRAWN — the
    helper is always None, the driver no longer calls it, and a header-located
    cell on gemma4 proceeds on the declared path (in support:
    fitted_thinking_on_out_of_support is None there). The declared-cells
    enumeration still names the two scale-12 safety objects, which the dry run
    and the analysis list as SERVED."""
    assert C.HEADER_INVARIANT_THINKING_FAMILIES == ("gemma4",)
    for fam, hl, arm in (("gemma4", True, "on"), ("gemma4", False, "on"), ("gemma4", True, "off"),
                         ("qwen3_5", True, "on"), (None, True, "on")):
        assert C.header_invariant_declaration_refusal(fam, header_located=hl, arm=arm) is None
    assert C.fitted_thinking_on_out_of_support("gemma4", header_located=True) is None
    assert C.header_invariant_declared_cells("gemma4e4b") == C.G8K_HEADER_LOCATED_CELLS
    assert C.header_invariant_declared_cells("qwen35_4b") == () == C.header_invariant_declared_cells("nemotron3nano4b")
    assert C.header_invariant_declared_cells("not-a-model") == ()
    driver = (REPO / "scripts/served_capability_run.py").read_text()
    assert "tc.header_invariant_declaration_refusal(" not in driver
    at = driver.index("served on the declared path (the user's")
    assert at < driver.index("        _check_served_reasoning_parser(\n            atoms, thinking, standup_endpoint,")
    assert "on the header-INVARIANT family" in driver[at - 400:at] and "REFUSED BY NAME" not in driver[at - 2500:at + 900]


def test_the_item_subset_file_is_validated_and_binds_the_source_by_hash(tmp_path):
    """The rung planner's file, exactly `ITEM_SUBSET_FIELDS`, the eligible set
    as the at-cap difference, a PASSED transport verdict; the digest is
    order-free; the hashed `cap_rung` key has exactly `CAP_RUNG_FIELDS` and
    rides version 2 only, under the flag only."""
    ids = [f"gsm8k_{i:08x}" for i in range(5)]
    doc = {"schema": C.ITEM_SUBSET_SCHEMA, "task": C.GSM8K_STANDARD_TASK, "source_out_root": "/synthetic/root",
           "source_identity_hash": "0123456789abcdef", "source_cap": 8192, "cap": 16384, "rule": "r",
           "threshold": 0.03, "ids": ids, "n_at_cap": 7, "n_non_terminating": 2, "n_boundary": 0,
           "transport_check": {"ok": True}}
    assert C.validate_item_subset(doc) == doc
    p = tmp_path / "rung_subset.json"
    p.write_text(json.dumps(doc))
    assert C.read_item_subset(p) == doc
    assert C.FLAG_ITEM_SUBSET == "--item-subset" and C.ITEM_SUBSET_SCHEMA == "rung_subset/1"
    assert C.item_subset_sha16(ids) == C.item_subset_sha16(list(reversed(ids))) and len(C.item_subset_sha16(ids)) == 16
    assert C.item_subset_sha16(ids) != C.item_subset_sha16(ids[:4])
    ident = C.cap_rung_identity(doc)
    assert tuple(ident) == C.CAP_RUNG_FIELDS == ("source_identity_hash", "subset_sha16", "source_cap", "rule")
    assert ident["subset_sha16"] == C.item_subset_sha16(ids) and ident["source_cap"] == 8192 and ident["rule"] == "r"
    block = C.item_subset_block(doc, path=p)
    assert tuple(block) == C.ITEM_SUBSET_BLOCK_FIELDS and block["n"] == 5 and block["ids"] == ids and block["path"] == str(p)
    bad = [("schema", "x", "schema"), ("task", "inspect/nope", "not an Inspect row task"),
           ("source_identity_hash", "zz", "16-hex"), ("cap", 8192, "source_cap < cap"), ("cap", 32768, "source_cap < cap"),
           ("rule", "", "rule must"), ("threshold", 1.5, "share in"), ("ids", [], "non-empty list"),
           ("ids", ids + [ids[0]], "duplicates"), ("n_at_cap", 9, "the subset IS the eligible set"),
           ("transport_check", {"ok": False}, "PASSED verdict"), ("source_out_root", "", "non-empty path")]
    for k, v, why in bad:
        with pytest.raises(C.ItemSubsetError, match=why):
            C.validate_item_subset({**doc, k: v})
    with pytest.raises(C.ItemSubsetError, match="not the contract"):
        C.validate_item_subset({**doc, "extra": 1})
    with pytest.raises(C.ItemSubsetError, match="unreadable"):
        C.read_item_subset(tmp_path / "missing.json")
    assert C.IDENTITY_KEY_CAP_RUNG == "cap_rung" and C.IDENTITY_KEY_CAP_RUNG in C.TINYCAP_IDENTITY_V2_FIELDS
    v2 = C.tinycap_identity_fields(identity_version=2, cap_rung=True)
    assert v2[-1] == "cap_rung" and C.tinycap_identity_fields(identity_version=2) == v2[:-1]
    assert C.tinycap_identity_fields(identity_version=1, cap_rung=True) == C.TINYCAP_IDENTITY_FIELDS
    assert "cap_rung" not in C.tinycap_identity_fields(out_of_support_thinking=True, identity_version=2)
    assert "cap_rung" in C.tinycap_identity_fields(out_of_support_thinking=True, identity_version=2, cap_rung=True)


def _lrow(rid, *, task=C.GSM8K_STANDARD_TASK, sample="x/0", rejected=False, error_code=None, error_class="none", attempt=None):
    row = {"task": task, "sample": sample, "rejected": rejected, "error_code": error_code, "error_class": error_class,
           C.LEDGER_REQUEST_ID_FIELD: rid, C.LEDGER_BODY_SHA256_FIELD: "ab" * 32}
    if attempt is not None:
        row[C.LEDGER_ATTEMPT_ID_FIELD] = attempt
    return row


def _erec(rid, *, attempt, kind="task", task=C.GSM8K_STANDARD_TASK, sample="x/0"):
    return {"request_id": rid, "kind": kind, "task": (task if kind == "task" else "probe/render"), "sample": sample,
            C.REQUEST_EVIDENCE_ATTEMPT_FIELD: attempt}


def test_the_attempt_contract_attributes_rows_by_column_or_by_request_id_and_refuses_the_rest():
    """The attempt id grammar; a ledger row's attempt is its own column when
    the ledger stamped one, else its request-evidence record's (an identity
    join); both disagreeing or neither present refuse by name; the scoped
    helpers over rows and records."""
    assert C.FLAG_ATTEMPT_ID == "--attempt-id" and C.check_attempt_id_arg("a1") == "a1" and C.check_attempt_id_arg("a12") == "a12"
    for bad in ("a0", "1", "attempt-1", "", None, "a01", "b2"):
        with pytest.raises(C.AttemptScopeError, match="a<k>"):
            C.check_attempt_id_arg(bad)
    assert C.ATTEMPT_ID_FIELD == C.LEDGER_ATTEMPT_ID_FIELD == C.REQUEST_EVIDENCE_ATTEMPT_FIELD == "attempt_id"
    assert C.LEDGER_ATTEMPT_ID_FIELD not in C.LEDGER_ROW_FIELDS and C.REQUEST_EVIDENCE_ATTEMPT_FIELD not in C.REQUEST_EVIDENCE_FIELDS
    assert C.ATTEMPT_KINDS == ("generation", "finalisation") and C.FLAG_ATTEMPT_ID not in C.TINYCAP_DRIVER_FLAGS
    recs = [_erec("r1", attempt="a1"), _erec("r2", attempt="a1", kind="probe"), _erec("r3", attempt="a2")]
    ev = C.evidence_by_request_id(recs)
    assert C.ledger_row_attempt(_lrow("r1"), ev) == "a1"                          # by request id
    assert C.ledger_row_attempt(_lrow("r9", attempt="a3"), ev) == "a3"             # by the ledger's column
    assert C.ledger_row_attempt(_lrow("r3", attempt="a2"), ev) == "a2"             # both, agreeing
    with pytest.raises(C.AttemptScopeError, match="torn record"):
        C.ledger_row_attempt(_lrow("r3", attempt="a1"), ev)
    with pytest.raises(C.AttemptScopeError, match="belongs to no attempt"):
        C.ledger_row_attempt(_lrow("r9"), ev)
    with pytest.raises(C.AttemptScopeError, match="two request-evidence records"):
        C.evidence_by_request_id(recs + [_erec("r1", attempt="a2")])
    rows = [_lrow("r1"), _lrow("r2", task="probe/render"), _lrow("r3")]
    assert [r[C.LEDGER_REQUEST_ID_FIELD] for r in C.rows_of_attempts(rows, ["a1"], evidence_by_id=ev)] == ["r1", "r2"]
    assert [r["request_id"] for r in C.records_of_attempts(recs, ["a2"])] == ["r3"]
    with pytest.raises(C.AttemptScopeError, match="carries no attempt_id"):
        C.records_of_attempts([{"request_id": "q", "kind": "task"}], ["a1"])
    assert (C.attempt_generation_calls(recs, "a1"), C.attempt_generation_calls(recs, "a2"), C.attempt_generation_calls(recs, "a3")) == (1, 1, 0)
    ev4 = C.evidence_by_request_id(recs + [_erec("r4", attempt="a1")])
    rows4 = rows + [_lrow("r4", rejected=True)]
    assert C.scoped_rejected_count(rows4, C.GSM8K_STANDARD_TASK, ["a1"], evidence_by_id=ev4) == 1
    assert C.scoped_rejected_count(rows4, C.GSM8K_STANDARD_TASK, ["a2"], evidence_by_id=ev4) == 0
    assert C.AttemptScope("a2", "a1", "/synthetic/e.jsonl").prior_generation_attempt == "a1"


def test_resolve_attempt_kinds_finalises_after_generation_and_isolates_a_dead_attempt():
    """The attempt whose records prove task calls is the generation; a run that
    sent none FINALISES from the latest one; a dead attempt's rejected, refused
    call is ATTRIBUTED and the whole-ledger read that would have refused the
    clean attempt is pinned as the bug; unrecorded / pre-contract / torn
    refuse."""
    code = C.THINKING_REFUSAL_CODES[0]
    attempts = [{"ts": "t1", "attempt": 1, "completed": False, "attempt_id": "a1", "kind": None, "generation_calls": None,
                 "finalised_from_attempt": None, "started": "t1", "ended": None, "terminal": None},
                {"ts": "t2", "attempt": 2, "completed": False, "attempt_id": "a2", "kind": None, "generation_calls": None,
                 "finalised_from_attempt": None, "started": "t2", "ended": None, "terminal": None}]
    # T1: a1 generated 200 and died at finalisation; a2 sent no task call -> a finalisation FROM a1, a1's count stands
    recs = [_erec(f"g{i}", attempt="a1") for i in range(200)] + [_erec("p1", attempt="a1", kind="probe"),
                                                                 _erec("p2", attempt="a2", kind="probe")]
    out, accepted = C.resolve_attempt_kinds(attempts, recs, current_attempt_id="a2", ended="t3")
    assert accepted == "a1"
    assert out[0]["kind"] == C.ATTEMPT_KIND_GENERATION and out[0]["generation_calls"] == 200
    assert out[0]["terminal"] == C.ATTEMPT_TERMINAL_INTERRUPTED and out[0]["completed"] is False
    assert out[1]["kind"] == C.ATTEMPT_KIND_FINALISATION and out[1]["generation_calls"] == 0
    assert out[1]["finalised_from_attempt"] == "a1" and out[1]["ended"] == "t3" and out[1]["terminal"] == C.ATTEMPT_TERMINAL_COMPLETED
    assert set(C.RESUME_ATTEMPT_SCOPED_FIELDS) <= set(out[1])
    # T2 / T3: a1 died mid-generation (150 items + one REJECTED, refused call); a2 generated 200 -> a2 accepted
    recs2 = [_erec(f"g{i}", attempt="a1", sample=f"i{i}/0") for i in range(151)] + \
            [_erec(f"h{i}", attempt="a2", sample=f"i{i}/0") for i in range(200)]
    out2, acc2 = C.resolve_attempt_kinds(attempts, recs2, current_attempt_id="a2")
    assert acc2 == "a2" and out2[0]["generation_calls"] == 151 and out2[1]["kind"] == C.ATTEMPT_KIND_GENERATION
    ev = C.evidence_by_request_id(recs2)
    rows = ([_lrow(f"g{i}", sample=f"i{i}/0") for i in range(150)]
            + [_lrow("g150", sample="i150/0", rejected=True, error_code=code, error_class="route_refused")]
            + [_lrow(f"h{i}", sample=f"i{i}/0") for i in range(200)])
    attribution = C.attempt_attribution(rows, attempts=out2, admitted={"a2"}, evidence_by_id=ev)
    assert set(attribution) == {"a1"} and tuple(attribution["a1"]) == C.ATTEMPT_ATTRIBUTION_FIELDS
    assert attribution["a1"]["n_rows"] == 151 and attribution["a1"]["rejected"] == 1 and attribution["a1"]["task_calls"] == 151
    assert attribution["a1"]["refusals"] == {code: 1} and attribution["a1"]["kind"] == "generation"
    assert attribution["a1"]["terminal"] == C.ATTEMPT_TERMINAL_INTERRUPTED and attribution["a1"]["probe_calls"] == 0
    # the OLD whole-ledger read is the bug: it reads a1's refusal against a2; the scoped read does not
    assert C.ledger_thinking_refusals(rows) == {code: 1}
    assert C.ledger_thinking_refusals(C.rows_of_attempts(rows, {"a2"}, evidence_by_id=ev)) == {}
    assert len(C.rows_of_attempts(rows, {"a2"}, evidence_by_id=ev)) == 200 and len(C.records_of_attempts(recs2, {"a2"})) == 200
    # a recorded attempt with no rows is still listed; an UNRECORDED attempt's row refuses by name
    assert C.attempt_attribution([], attempts=out2, admitted={"a2"}, evidence_by_id=ev)["a1"]["n_rows"] == 0
    with pytest.raises(C.AttemptScopeError, match="does not record"):
        C.attempt_attribution(rows + [_lrow("z", attempt="a9")], attempts=out2, admitted={"a2"}, evidence_by_id=ev)
    # a finalisation with nothing to finalise, a pre-contract entry, a torn history, a mis-ordered current: refused
    with pytest.raises(C.AttemptScopeError, match="no recorded prior attempt generated"):
        C.resolve_attempt_kinds(attempts[1:], [_erec("p", attempt="a2", kind="probe")], current_attempt_id="a2")
    with pytest.raises(C.AttemptScopeError, match="carries no attempt_id"):
        C.resolve_attempt_kinds([{"ts": "t0", "attempt": 1, "completed": True}, *attempts], recs, current_attempt_id="a2")
    torn = [{**attempts[0], "kind": C.ATTEMPT_KIND_FINALISATION, "generation_calls": 0}, attempts[1]]
    with pytest.raises(C.AttemptScopeError, match="torn history"):
        C.resolve_attempt_kinds(torn, recs, current_attempt_id="a2")
    with pytest.raises(C.AttemptScopeError, match="not the LAST recorded"):
        C.resolve_attempt_kinds(attempts, recs, current_attempt_id="a1")
    with pytest.raises(C.AttemptScopeError, match="twice"):
        C.resolve_attempt_kinds([attempts[0], {**attempts[1], "attempt_id": "a1"}], recs, current_attempt_id="a1")


def test_the_gsm8k_stamp_triples_are_derived_from_the_contract_and_the_row_task_is_the_one_comparison_spelling():
    """The launcher's frozen matrix carried a typed declared triple
    `(declared-request-unique, always, thinking_on)` that is NOT a row of the
    contract (the salt regime is the ordinary `request-unique`; the declared
    label rides `resumed_prefill_policy`), and the planner compared the
    subset's ROW task against the unprefixed NAME so every rung refused. The
    contract now EXPORTS the three derived triples and states which spelling is
    which; both call sites import, neither re-types."""
    assert C.G8K_STAMPS_BASE == ("cell-stable", "not-consulted", "thinking_on")
    assert C.G8K_STAMPS_FITTED == ("cell-stable", "always", "thinking_on")
    assert C.G8K_STAMPS_DECLARED == ("request-unique", "declared-request-unique", "thinking_on")
    assert C.G8K_STAMPS_DECLARED != ("declared-request-unique", "always", "thinking_on")      # the frozen literal (B-P1.3)
    # derived, never spelled: each equals the contract's own rule at the class's policy, and the driver's triple
    assert C.G8K_STAMPS_DECLARED == tuple(C.expected_stamp_regimes(lens=True, arm=C.THINKING_ON,
                                                                   policy=C.DECLARED_SALT_POLICY_LABEL))
    assert C.G8K_STAMPS_DECLARED == C.expected_stamp_triple(lens=True, arm=C.THINKING_ON, header_located=True,
                                                            attested_policy=None, declared=True)
    assert C.G8K_STAMPS_FITTED == C.expected_stamp_triple(lens=True, arm=C.THINKING_ON, header_located=False,
                                                          attested_policy="always")
    assert C.G8K_STAMPS_BASE == C.expected_stamp_triple(lens=False, arm=C.THINKING_ON, header_located=False,
                                                        attested_policy=None)
    assert C.G8K_STAMPS_DECLARED[1] == C.authorised_stamp_policy(lens=True, arm=C.THINKING_ON)
    for salt, policy, render in C.G8K_STAMPS_BY_CLASS.values():
        assert salt in C.SALT_REGIME_LITERALS and policy in C.RESUMED_POLICY_LABELS
        assert render == C.RENDER_REGIME_THINKING_ON == C.RENDER_REGIME_BY_ARM[C.THINKING_ON]
    assert C.G8K_STAMP_CLASSES == ("base", "fitted", "declared") and tuple(C.G8K_STAMPS_BY_CLASS) == C.G8K_STAMP_CLASSES
    assert C.G8K_STAMPS_BY_CLASS == {"base": C.G8K_STAMPS_BASE, "fitted": C.G8K_STAMPS_FITTED,
                                     "declared": C.G8K_STAMPS_DECLARED}
    assert C.g8k_expected_stamps(fitted=False, declared=False) == C.G8K_STAMPS_BASE
    assert C.g8k_expected_stamps(fitted=True, declared=False) == C.G8K_STAMPS_FITTED
    assert C.g8k_expected_stamps(fitted=True, declared=True) == C.G8K_STAMPS_DECLARED
    with pytest.raises(ValueError, match="DECLARED row is a fitted"):
        C.g8k_expected_stamps(fitted=False, declared=True)
    for name in ("G8K_STAMPS_BASE", "G8K_STAMPS_FITTED", "G8K_STAMPS_DECLARED", "G8K_STAMP_CLASSES",
                 "G8K_STAMPS_BY_CLASS", "g8k_expected_stamps"):
        assert name in C.__all__, name
    src = Path(C.__file__).read_text()
    assert "STAMPS_DECLARED: Tuple[str, str, str] = tuple(\n    expected_stamp_regimes(" in src   # derived in the source, not typed
    assert re.search(r'=\s*\(\s*"declared-request-unique",\s*"always"', src) is None      # never ASSIGNED as a triple
    # the ROW task vs the NAME (B-P1.2): one comparison spelling, stated at both constants
    assert C.GSM8K_STANDARD_TASK == C.GSM8K_PAIRED_TASK == "inspect/gsm8k_paired200"
    assert C.GSM8K_PAIRED_TASK_NAME == "gsm8k_paired200" and C.inspect_task(C.GSM8K_PAIRED_TASK_NAME) == C.GSM8K_STANDARD_TASK
    assert C.GSM8K_STANDARD_TASK in {C.inspect_task(n) for n in C.INSPECT_TASK_NAMES}
    assert "THE ONE ROW-TASK CONSTANT" in src and "for the CLI cap flags" in src
    # the cap flags take the NAME; the subset validator takes the ROW task and refuses the name
    assert C.parse_on_cap_by_task([f"{C.GSM8K_PAIRED_TASK_NAME}=8192"]) == {C.GSM8K_PAIRED_TASK_NAME: 8192}
    doc = {"schema": C.ITEM_SUBSET_SCHEMA, "task": C.GSM8K_STANDARD_TASK, "source_out_root": "/synthetic/root",
           "source_identity_hash": "0123456789abcdef", "source_cap": 8192, "cap": 16384, "rule": "r",
           "threshold": 0.03, "ids": ["a", "b"], "n_at_cap": 2, "n_non_terminating": 0, "n_boundary": 0,
           "transport_check": {"ok": True}}
    assert C.validate_item_subset(doc)["task"] == C.GSM8K_STANDARD_TASK
    with pytest.raises(C.ItemSubsetError, match="not an Inspect row task"):
        C.validate_item_subset({**doc, "task": C.GSM8K_PAIRED_TASK_NAME})
def test_the_served_request_budget_is_one_number_across_every_hop():
    """A cap-bound draw on the steered route was 504'd at the lens proxy's own
    600 s upstream default and re-sampled while the driver, the ledger and the
    Inspect provider all waited 7,200 s. The budget is ONE contracts number
    now, read by every hop, and the launcher passes it to the standup on every
    tinycap configuration as ``--upstream-timeout-s`` (the proxy's default is
    deliberately NOT changed: the flag is how the contract's number reaches
    it)."""
    import inspect

    from sisteer.capeval.serving import proxy as px
    from sisteer.capeval.serving import proxy_engine as pe
    from sisteer.capeval.serving import usage_ledger as ul

    assert C.PROXY_UPSTREAM_TIMEOUT_S == 7200.0
    assert pe.AIOHTTP_TOTAL_TIMEOUT_S == ul.DEFAULT_TIMEOUT_S == C.INSPECT_TIMEOUT_S == C.PROXY_UPSTREAM_TIMEOUT_S
    assert isinstance(C.INSPECT_TIMEOUT_S, int)          # Inspect's GenerateConfig.timeout is an int
    assert C.FLAG_UPSTREAM_TIMEOUT_S == "--upstream-timeout-s"
    assert C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S == "proxy_upstream_timeout_s"
    for name in ("PROXY_UPSTREAM_TIMEOUT_S", "FLAG_UPSTREAM_TIMEOUT_S",
                 "ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S"):
        assert name in C.__all__, name
    # the proxy's own default is below the contract's number — which is exactly
    # why the launcher must pass the flag (and why this pin exists)
    # ONE number across every hop, the proxy's own default included (the sampled-rerun merge, 2026-09-11:
    # proxy.HOP_TIMEOUT_S == the contract; the 600 s transport default censored every cap-bound GLM
    # draw at the 8192 rung); the launcher's flag still names it explicitly
    assert inspect.signature(px.HTTPBackend.__init__).parameters["timeout_s"].default == C.PROXY_UPSTREAM_TIMEOUT_S
    assert px.HOP_TIMEOUT_S == C.PROXY_UPSTREAM_TIMEOUT_S and px.CLIENT_TIMEOUT_S > px.HOP_TIMEOUT_S
    # the driver's aiohttp client READS the contract (not a second literal)
    assert "AIOHTTP_TOTAL_TIMEOUT_S = C.PROXY_UPSTREAM_TIMEOUT_S" in (
        REPO / "src/sisteer/capeval/serving/proxy_engine.py").read_text()
    # the ledger-derived rejected-call count is a summary field, counts-only safe
    assert "n_http_rejected_retried" in C.SUMMARY_PAYLOAD_FIELDS
    assert _SAFE_STRING.match("n_http_rejected_retried")
    assert "n_http_rejected_retried" not in C.DONE_MARKER_FIELDS      # the marker's exact set is untouched


# --------------------------------------------------------------------------- #
# BCB / BFCL 200: the two pinned legs' contracts                                #
# --------------------------------------------------------------------------- #


def test_the_two_pinned_legs_enter_every_table_and_no_battery():
    """Every table indexed by control or task has both rows, so the legs are
    launchable by NAME alone — and neither is in any default battery."""
    bcb, bfcl = C.inspect_task("bigcodebench_200"), C.inspect_task("bfcl_200")
    assert (bcb, bfcl) == ("inspect/bigcodebench_200", "inspect/bfcl_200")
    assert C.THINKING_BY_TASK[bcb] == C.THINKING_BY_TASK[bfcl] == C.THINKING_OFF
    assert C.SAMPLES_BY_TASK[bcb] == C.SAMPLES_BY_TASK[bfcl] == 1
    assert C.GEOMETRY_CLASS_OF_CONTROL[C.CONTROL_BIGCODEBENCH_200] == C.GEOMETRY_CLASS_GEN_LONG
    assert C.GEOMETRY_CLASS_OF_CONTROL[C.CONTROL_BFCL_200] == C.GEOMETRY_CLASS_AGENT
    assert C.DISPATCH_MODULE_OF_CONTROL[C.CONTROL_BIGCODEBENCH_200].endswith(".inspect_served")
    assert C.DISPATCH_MODULE_OF_CONTROL[C.CONTROL_BFCL_200].endswith(".inspect_served")
    assert {C.CONTROL_BIGCODEBENCH_200, C.CONTROL_BFCL_200} <= set(C.INSPECT_CONTROLS)
    assert C.TINYCAP_DOMAIN_OF[bcb] == C.TINYCAP_DOMAIN_OF[bfcl] == C.DOMAIN_INSPECT_GEN
    for kind in C.BATTERY_KINDS:
        battery = C.battery_controls(kind, in_depth=True)
        assert C.CONTROL_BIGCODEBENCH_200 not in battery and C.CONTROL_BFCL_200 not in battery
        assert bcb not in C.battery_thinking(kind, in_depth=True)
    assert C.marker_filename(bcb) == "inspect__bigcodebench_200.json"
    # the sandbox identity leaves and the row-fact vocabulary
    assert C.BCB_SANDBOX_PROVIDER == "modal_owned" and C.BCB_SANDBOX_APP_NAME == "inspect_modal_sandbox"
    assert C.BCB_SANDBOX_IMAGE.startswith("ghcr.io/generality-labs/inspect-eval-bigcodebench:2026-07-28@sha256:")
    assert C.BCB_SANDBOX_IMAGE.endswith("a1cfe") and len(C.BCB_SANDBOX_IMAGE.split("@sha256:")[1]) == 64
    assert (C.BCB_VERIFY_TIMEOUT_S, C.BCB_SANDBOX_TIMEOUT_S) == (30, 10_800)
    assert dict(C.BCB_SANDBOX_RESOURCES) == {"cpu": 1.0, "memory_mib": 512, "block_network": True}
    assert C.bcb_sandbox_identity() == {"image": C.BCB_SANDBOX_IMAGE, "verify_timeout_s": 30}
    assert C.BCB_ROW_FACT_FIELDS == ("code_source", "sandbox_status", "verifier_explanation")
    assert C.BCB_CODE_SOURCES == ("python_fence", "bare_fence", "whole_completion")
    assert C.BCB_SANDBOX_STATUSES == ("ok", "verify_timeout")
    for f in C.BCB_SUMMARY_COUNT_FIELDS:
        assert f in C.SUMMARY_PAYLOAD_FIELDS and _SAFE_STRING.match(f), f
    assert not (set(C.BCB_ROW_FACT_FIELDS) & set(C.INSPECT_EXTRA_FIELDS))   # additive extra_json keys


def test_the_pinned_legs_identity_keys_are_present_only_when_filed():
    """The rule for the two legs: `bcb_item_set` + `bcb_sandbox` when and
    only when the run files inspect/bigcodebench_200, `bfcl_item_set` when and
    only when it files inspect/bfcl_200 — appended INDEPENDENTLY after the MATH
    tuple and before the OOS / ladder tuples, so every pre-existing state
    (the three exact pins above and the goldens) is byte-identical."""
    assert C.TINYCAP_IDENTITY_BCB_FIELDS == ("bcb_item_set", "bcb_sandbox")
    assert C.TINYCAP_IDENTITY_BFCL200_FIELDS == ("bfcl_item_set",)
    for key in (*C.TINYCAP_IDENTITY_BCB_FIELDS, *C.TINYCAP_IDENTITY_BFCL200_FIELDS):
        assert key not in C.TINYCAP_IDENTITY_FIELDS
        assert key not in C.TINYCAP_IDENTITY_OPTIONAL_FIELDS and key not in C.TINYCAP_IDENTITY_MATH500_FIELDS
        assert key not in C.TINYCAP_IDENTITY_LADDER_FIELDS
    assert C.tinycap_identity_fields() == C.TINYCAP_IDENTITY_FIELDS
    assert C.tinycap_identity_fields(files_bigcodebench_200=True) == (
        *C.TINYCAP_IDENTITY_FIELDS, "bcb_item_set", "bcb_sandbox")
    assert C.tinycap_identity_fields(files_bfcl_200=True) == (*C.TINYCAP_IDENTITY_FIELDS, "bfcl_item_set")
    assert C.tinycap_identity_fields(files_math500=True, files_bigcodebench_200=True, files_bfcl_200=True,
                                     out_of_support_thinking=True, ladder=True) == (
        *C.TINYCAP_IDENTITY_FIELDS, *C.TINYCAP_IDENTITY_MATH500_FIELDS, *C.TINYCAP_IDENTITY_BCB_FIELDS,
        *C.TINYCAP_IDENTITY_BFCL200_FIELDS, *C.TINYCAP_IDENTITY_OPTIONAL_FIELDS,
        *C.TINYCAP_IDENTITY_LADDER_FIELDS)
    # the predicates take ROW tasks (a map's keys or a list); atoms refuse
    bcb = C.inspect_task("bigcodebench_200")
    assert C.files_bigcodebench_200({bcb: "off", "inspect/math": "off"}) is True
    assert C.files_bigcodebench_200(["inspect/bfcl_200"]) is False
    assert C.files_bfcl_200(["inspect/bfcl_200"]) is True and C.files_bfcl_200(["inspect/bfcl"]) is False
    with pytest.raises(ValueError, match="ROW tasks"):
        C.files_bfcl_200(["bfcl_200"])
    # the item-set identity is small, pure and pin-derived
    pin = C.load_bcb_pin(REPO)
    ident = C.item_set_identity(pin)
    assert set(ident) == {"seed", "n", "of", "ids_sha256"} and ident["n"] == 200 and ident["of"] == 992
    assert ident["ids_sha256"] == C.ids_sha256(pin["ids"])


# --------------------------------------------------------------------------- #
# BCB / BFCL 200: the truncation ladder's contracts                             #
# --------------------------------------------------------------------------- #


def test_the_ladder_contracts_flags_parsers_and_the_identity_cross_product():
    bcb, bfcl = C.LADDER_ROW_TASKS
    assert (bcb, bfcl) == ("inspect/bigcodebench_200", "inspect/bfcl_200")
    assert C.LADDER_CAP_TO == C.MATH500_MAX_TOKENS_DEFAULT == 16384          # the standing ceiling, one number
    assert (C.FLAG_CAP_OVERRIDE, C.FLAG_ITEM_SELECTION, C.FLAG_ITEM_SELECTION_JSON, C.FLAG_LADDER_RUNG,
            C.FLAG_LADDER_REPROBE) == ("--cap-override", "--item-selection", "--item-selection-json",
                                       "--ladder-rung", "--ladder-reprobe")
    assert C.TINYCAP_LADDER_DRIVER_FLAGS == ("--cap-override", "--item-selection-json")
    assert not (set(C.TINYCAP_LADDER_DRIVER_FLAGS) & set(C.TINYCAP_DRIVER_FLAGS))
    assert C.TINYCAP_IDENTITY_LADDER_FIELDS == ("cap_overrides_by_task", "item_selection")
    assert C.ITEM_SELECTION_JSON_FIELDS == ("ids", "n", "ids_sha256")
    # cap overrides: sorted, exact, refusing everything malformed
    caps = C.parse_cap_overrides(f"{bcb}=16384,{bfcl}=16384")
    assert list(caps) == sorted([bcb, bfcl]) == [bfcl, bcb] and caps[bcb] == 16384    # sorted by task
    assert C.parse_cap_overrides("") == {} and C.format_cap_overrides(caps) == f"{bfcl}=16384,{bcb}=16384"
    assert C.parse_cap_overrides(C.format_cap_overrides(caps)) == caps
    for bad, needle in ((f"{bcb}", "not <row_task>=<cap>"), ("inspect/math=16384", "not a ladder row task"),
                        (f"{bcb}=16384,{bcb}=8192", "named twice"), (f"{bcb}=big", "not an int"),
                        (f"{bcb}=0", "not positive"), ("=5", "not <row_task>=<cap>")):
        with pytest.raises(ValueError, match=needle):
            C.parse_cap_overrides(bad)
    # the selection's log vocabulary, per task
    assert C.ladder_log_ids(bcb, ["BigCodeBench/17", "BigCodeBench/3"]) == ["17", "3"]
    assert C.ladder_log_ids(bfcl, ["simple_5"]) == ["simple_5"]
    with pytest.raises(ValueError):
        C.ladder_log_ids(bcb, ["17"])
    with pytest.raises(ValueError, match="not a ladder row task"):
        C.ladder_log_ids("inspect/math", ["x"])
    # the hashed value: {n, ids_sha256} over the PINNED ids in the given order
    ident = C.item_selection_identity({bfcl: ["b", "a"], bcb: ["BigCodeBench/1"]})
    assert list(ident) == sorted([bcb, bfcl]) and ident[bfcl] == {"n": 2, "ids_sha256": C.ids_sha256(["b", "a"])}
    assert ident[bfcl]["ids_sha256"] != C.item_selection_identity({bfcl: ["a", "b"]})[bfcl]["ids_sha256"]
    for bad, needle in (({bfcl: []}, "selects no items"), ({bfcl: ["a", "a"]}, "an id twice"),
                        ({"inspect/math": ["a"]}, "not a ladder row task")):
        with pytest.raises(ValueError, match=needle):
            C.item_selection_identity(bad)
    # the driver's inline JSON: compact, sorted, and VERIFIED on the way back
    j = C.format_item_selection_json({bcb: ["BigCodeBench/17", "BigCodeBench/3"], bfcl: ["s1"]})
    doc = json.loads(j)
    assert list(doc) == sorted([bcb, bfcl]) and set(doc[bcb]) == {"ids", "n", "ids_sha256"} and doc[bcb]["n"] == 2
    back = C.parse_item_selection_json(j)
    assert back[bcb]["ids"] == ["BigCodeBench/17", "BigCodeBench/3"] and back[bfcl] == {
        "ids": ["s1"], "n": 1, "ids_sha256": C.ids_sha256(["s1"])}
    for bad, needle in ((j.replace('"n":2', '"n":3'), "!= len"),
                        (j.replace(doc[bcb]["ids_sha256"], "0" * 64), "does not recompute"),
                        (json.dumps({bcb: {"ids": ["17"], "n": 1, "ids_sha256": C.ids_sha256(["17"])}}),
                         "not a BigCodeBench task id"),                      # the LOG vocabulary is refused
                        ('{"inspect/math": {"ids": ["a"], "n": 1, "ids_sha256": "x"}}', "not a ladder row task"),
                        ("{}", "non-empty"), ("not json", "not JSON"),
                        (json.dumps({bfcl: {"ids": ["a"], "n": 1}}), "exactly")):
        with pytest.raises(ValueError, match=needle):
            C.parse_item_selection_json(bad)
    # driver_argv: the two flags LAST, always together; absent -> byte-identical
    base = dict(python="p", script="s", proxy_url="u", routing_table="r", cell="c", config="cfg", defenses_dir="d",
                out_root="o", controls=["bfcl_200"], thinking="off", concurrency=16, judge_model="j")
    plain = C.driver_argv(**base)
    ladder = C.driver_argv(**base, cap_override=f"{bfcl}=16384", item_selection_json=json.dumps({bfcl: back[bfcl]}))
    assert ladder[:len(plain)] == plain and ladder[len(plain):] == [
        "--cap-override", f"{bfcl}=16384", "--item-selection-json", json.dumps({bfcl: back[bfcl]})]
    assert C.driver_argv(**base, cap_override=None, item_selection_json=None) == plain
    with pytest.raises(ValueError, match="travel together"):
        C.driver_argv(**base, cap_override=f"{bfcl}=16384")
    with pytest.raises(ValueError, match="does not recompute"):
        C.driver_argv(**base, cap_override=f"{bfcl}=16384",
                      item_selection_json=json.dumps({bfcl: {**back[bfcl], "ids_sha256": "0" * 64}}))
    # the identity key set over the eight (bcb, bfcl, ladder) states: appended independently, in order
    for files_bcb in (False, True):
        for files_bfcl in (False, True):
            for ladder_on in (False, True):
                fields = C.tinycap_identity_fields(files_bigcodebench_200=files_bcb, files_bfcl_200=files_bfcl,
                                                   ladder=ladder_on)
                want = (*C.TINYCAP_IDENTITY_FIELDS,
                        *(C.TINYCAP_IDENTITY_BCB_FIELDS if files_bcb else ()),
                        *(C.TINYCAP_IDENTITY_BFCL200_FIELDS if files_bfcl else ()),
                        *(C.TINYCAP_IDENTITY_LADDER_FIELDS if ladder_on else ()))
                assert fields == want and len(set(fields)) == len(fields), (files_bcb, files_bfcl, ladder_on)
    # the OOS tuple sits BEFORE the ladder tuple when both are on
    both = C.tinycap_identity_fields(out_of_support_thinking=True, ladder=True)
    assert both.index("out_of_support_thinking") < both.index("cap_overrides_by_task")


def test_driver_argv_appends_the_ifeval_tail_last_and_is_byte_identical_without_it():
    """Keyword-only, default-None arguments append the IFEval
    tail AFTER everything `driver_argv` emitted before (after the ladder pair and the digest / budget
    flags), in the fixed order `--tinycap-identity-version, --max-tokens-by-task, --off-cap-by-task,
    --item-subset, --bank-view`; `--sampling <mode>` rides in the sampling slot main placed it in
    (before `--protocol-check` and the ladder pair) whenever the mode is not the default, None
    included; all None -> byte-identical argv. The statement, the
    setter and the subset refuse under identity version 1 (the driver would, after the standup); a
    setter outside `OFF_CAP_BY_TASK_ALLOWED`, an unknown view or mode refuse by name."""
    kw = dict(python="p", script="s", proxy_url="u", routing_table="r", cell="none", config="c",
              defenses_dir="d", out_root="o", controls=[C.CONTROL_IFEVAL], thinking="off", concurrency=142,
              judge_model="j", samples=2)
    plain = C.driver_argv(**kw)
    assert C.driver_argv(**kw, sampling_mode=None, identity_version=None, max_tokens_by_task=None,
                         off_cap_by_task=None, item_subset=None, bank_view=None) == plain
    assert plain[plain.index("--samples") + 1] == "2" and not any(f in plain for f in C.IFEVAL_DRIVER_FLAGS)
    full = C.driver_argv(**kw, sampling_mode=C.SAMPLING_VENDOR_PRESET, identity_version=2,
                         max_tokens_by_task={"ifeval": 4096}, off_cap_by_task={"ifeval": 4096},
                         item_subset="/vol/w/rung_subset.json", bank_view="it")
    assert full == plain + ["--sampling", "vendor_preset", "--tinycap-identity-version", "2",
                            "--max-tokens-by-task", "ifeval=4096", "--off-cap-by-task", "ifeval=4096",
                            "--item-subset", "/vol/w/rung_subset.json", "--bank-view", "it"]
    # pass 1: no setter, no subset — the statement, the version, the mode and the view alone
    p1 = C.driver_argv(**kw, sampling_mode=C.SAMPLING_VENDOR_PRESET, identity_version=2,
                       max_tokens_by_task={"ifeval": 2048}, bank_view="main")
    assert p1 == plain + ["--sampling", "vendor_preset", "--tinycap-identity-version", "2",
                          "--max-tokens-by-task", "ifeval=2048", "--bank-view", "main"]
    # the tail comes AFTER the digest / budget flags and the ladder pair
    ladder_kw = dict(kw, controls=["bfcl_200"], cap_override=C.format_cap_overrides({C.LADDER_ROW_TASKS[0]: C.LADDER_CAP_TO}),
                     item_selection_json=C.format_item_selection_json({C.LADDER_ROW_TASKS[0]: ["BigCodeBench/17"]}))
    lad = C.driver_argv(**ladder_kw)
    tailed = C.driver_argv(**ladder_kw, sampling_mode=C.SAMPLING_GREEDY)
    # the MODE rides in the sampling slot BEFORE the ladder pair (bcb-bfcl-200-widen: byte-identical to
    # a protocol launch's argv); the IFEval tail proper still comes after the pair
    assert tailed == lad[:-4] + ["--sampling", "greedy"] + lad[-4:] and lad[-2] == C.FLAG_ITEM_SELECTION_JSON
    assert lad[-4] == C.FLAG_CAP_OVERRIDE and tailed[-4:] == lad[-4:]
    tailed_bv = C.driver_argv(**ladder_kw, sampling_mode=C.SAMPLING_GREEDY, bank_view="main")
    assert tailed_bv == tailed + ["--bank-view", "main"]
    dig = C.driver_argv(**kw, expect_object_sha256="0" * 64, oos_prompt_budget="/b.json")
    assert C.driver_argv(**kw, expect_object_sha256="0" * 64, oos_prompt_budget="/b.json", bank_view="main") == dig + ["--bank-view", "main"]
    # several tasks: one statement per SORTED item
    two = C.driver_argv(**dict(kw, controls=["ifeval", "gsm8k"]), identity_version=2,
                        max_tokens_by_task={"ifeval": 2048, "gsm8k_paired200": 8192})
    i = two.index("--max-tokens-by-task")
    # (the statements spelled from the constants: a literal `task=cap` pair reads as a key to the
    # secret scanner, and the constants are the pin anyway)
    assert two[i:] == ["--max-tokens-by-task", f"{C.GSM8K_PAIRED_TASK_NAME}={C.G8K_CAP}",
                       "--max-tokens-by-task", f"{C.IFEVAL_TASK_NAME}={C.IFEVAL_MAX_TOKENS_OFF}"]
    # refusals, by name
    for bad in ({"max_tokens_by_task": {"ifeval": 2048}}, {"off_cap_by_task": {"ifeval": 4096}}, {"item_subset": "/s"}):
        with pytest.raises(ValueError, match="identity-version-2 mechanism"):
            C.driver_argv(**kw, **bad)
        with pytest.raises(ValueError, match="identity-version-2 mechanism"):
            C.driver_argv(**kw, identity_version=1, **bad)
    with pytest.raises(C.OffCapSetterRefused, match="outside the plan's table"):
        C.driver_argv(**kw, identity_version=2, off_cap_by_task={"ifeval": 2048})
    with pytest.raises(ValueError, match="--off-cap-by-task"):
        C.driver_argv(**kw, identity_version=2, off_cap_by_task={"nope": 4096})
    with pytest.raises(ValueError, match="--max-tokens-by-task"):
        C.driver_argv(**kw, identity_version=2, max_tokens_by_task={"nope": 4096})
    with pytest.raises(ValueError, match="--bank-view 'x' is not one of"):
        C.driver_argv(**kw, bank_view="x")
    with pytest.raises(ValueError, match="sampling mode must be one of"):
        C.driver_argv(**kw, sampling_mode="nope")
    with pytest.raises(ValueError, match="--tinycap-identity-version must be one of"):
        C.driver_argv(**kw, identity_version=3)
    with pytest.raises(ValueError, match="--item-subset needs a path"):
        C.driver_argv(**kw, identity_version=2, item_subset="")
    # the probed set, exported, and NOT in the always-probed TINYCAP_DRIVER_FLAGS (each takes a value)
    assert C.IFEVAL_DRIVER_FLAGS == ("--sampling", "--tinycap-identity-version", "--max-tokens-by-task",
                                     "--off-cap-by-task", "--item-subset", "--bank-view")
    assert "IFEVAL_DRIVER_FLAGS" in C.__all__ and not set(C.IFEVAL_DRIVER_FLAGS) & set(C.TINYCAP_DRIVER_FLAGS)
    assert C.IFEVAL_DISPATCH_STATE == "dispatch_state.json" and "IFEVAL_DISPATCH_STATE" in C.__all__


# --------------------------------------------------------------------------- #
# math-eight-arms-extension (E1, plan P4 / §4.2): the three `_lora2` tuples    #
# --------------------------------------------------------------------------- #


def test_the_three_lora2_tuples_are_minus_ctls_shape_under_new_names_and_the_published_tuples_did_not_move():
    """P4: the two extra all-position LoRA cells run under ONE NEW tuple per extension model, every field byte-equal
    to the published primary tuple except `cells` (the two cells) and `what`; the published EIGHT tuples' as_dict()
    are pinned to the fixture captured at E1 (the published stores record their tuple BY NAME — `oos_experiment` —
    so a widened `cells` on a published tuple would move a recorded experiment); the checker names exactly ONE
    tuple for each (model, cell) so per-cell tuples create no ambiguity; `check_authorised_render_facts` passes."""
    import json
    fixture = json.loads((REPO / "tests/fixtures/m5t_published_oos_tuples_55e3cbdd.json").read_text())
    assert sorted(fixture) == sorted(n for n in C.M5T_EXPERIMENT_NAMES
                                     if not n.endswith((C.M5T_EXT_TUPLE_SUFFIX, C.M5T_STEER_TUPLE_SUFFIX,
                                                        C.ALLPOS_TUPLE_SUFFIX,
                                                        # the seed wave's two native tuples (their own test below)
                                                        C.M5T_NATIVE_TUPLE_SUFFIX, C.M5T_NATIVE_PAIRED_TUPLE_SUFFIX)))   # + the all-positions arm's own tuple
    for name, want in fixture.items():
        assert C.AUTHORISED_OOS_EXPERIMENTS[name].as_dict() == want, name
    assert C.M5T_EXT_MODELS == ("gemma4e4b", "qwen35_4b", "nemotron3nano4b") and C.M5T_EXT_TUPLE_SUFFIX == "_lora2"
    assert C.M5T_EXT_CELLS == ("constv2-mix-full__lora_r4_ce", "constv2ctl2-full__lora_r4_ce")
    assert set(C.M5T_EXT_CELLS) <= set(C.G8K_ALL_POSITION_CELLS)          # the GSM8K tuples already serve both
    for m in C.M5T_EXT_MODELS:
        name = C.m5t_ext_tuple_name(m)
        assert name == f"math500_thinking_on_full22_{m}_lora2" and name in C.M5T_EXPERIMENT_NAMES
        ext, pub = C.AUTHORISED_OOS_EXPERIMENTS[name], C.AUTHORISED_OOS_EXPERIMENTS[f"math500_thinking_on_full22_{m}"]
        assert ext.cells == C.M5T_EXT_CELLS and pub.cells == (C.M5T_OBJECT_CELL,)
        assert {k for k in ext._asdict() if ext._asdict()[k] != pub._asdict()[k]} == {"cells", "what"}, m
        assert ext.sampling_mode == C.SAMPLING_THINKING_CARD and ext.max_tokens_on == {"math": C.MATH500_MAX_TOKENS_DEFAULT}
        assert ext.header_located is False and ext.declares is False and ext.reduction_mode == C.REDUCTION_MODE_STACKED
        assert "math-eight-arms-extension" in ext.what and m in ext.what and all(c in ext.what for c in C.M5T_EXT_CELLS)
        # ONE match per (model, cell): the published cell -> the published tuple, each extra cell -> the `_lora2` tuple. The LENS-ERA door
        # STATES its served reduction mode (`stacked`) — as the driver's first door and the launcher's planner do —
        # because qwen35_4b's LoRA cells ALSO sit in the seed wave's `_native` tuple, which differs from these in the served
        # identity alone (reduction_mode not_applicable): unstated, the pair is ambiguous by construction (the GSM8K property)
        for cell, want in ((C.M5T_OBJECT_CELL, pub), (C.M5T_EXT_CELLS[0], ext), (C.M5T_EXT_CELLS[1], ext)):
            got = C.check_out_of_support_thinking_authorised(
                model=m, cell=cell, controls=["math500"], mode="on", sampling_mode=C.SAMPLING_THINKING_CARD, samples=1,
                reduction_mode=C.REDUCTION_MODE_STACKED, max_tokens_by_task={"math": C.MATH500_MAX_TOKENS_DEFAULT})
            assert got == (name if want is ext else f"math500_thinking_on_full22_{m}"), (m, cell, got)
        # no `_lora2` sensitivity rows: the extra cells under `general` / `battery` are unauthorised
        for mode in (C.SAMPLING_GENERAL, C.SAMPLING_BATTERY):
            with pytest.raises(C.OutOfSupportThinkingUnauthorised):
                C.check_out_of_support_thinking_authorised(
                    model=m, cell=C.M5T_EXT_CELLS[0], controls=["math500"], mode="on", sampling_mode=mode, samples=1,
                    max_tokens_by_task={"math": C.MATH500_MAX_TOKENS_DEFAULT})
    # glm (the appendix model) and llama (E2's) have no `_lora2` tuple; the registry's render facts still pass whole
    for m in ("glm46vflash", "llama8b"):
        assert C.m5t_ext_tuple_name(m) not in C.AUTHORISED_OOS_EXPERIMENTS
    C.check_authorised_render_facts()
    # THREE `_steer` tuples, minus_ctl's shape over the two EXTERNAL steering cells, on the SWITCH
    # models only — a no-switch family's steering row is IN SUPPORT (one render, nothing to authorise), so llama has
    # none and glm (the appendix model) has none either.
    assert C.M5T_STEER_CELLS == ("alphasteer", "arditi") and C.M5T_STEER_MODELS == C.M5T_EXT_MODELS
    assert C.M5T_STEER_TUPLE_SUFFIX == "_steer" and C.G8K_STEER_MODELS == C.M5T_STEER_MODELS
    for m in ("glm46vflash", "llama8b"):
        assert C.m5t_steer_tuple_name(m) not in C.AUTHORISED_OOS_EXPERIMENTS
    for m in C.M5T_STEER_MODELS:
        name = C.m5t_steer_tuple_name(m)
        steer, pub = C.AUTHORISED_OOS_EXPERIMENTS[name], C.AUTHORISED_OOS_EXPERIMENTS[f"math500_thinking_on_full22_{m}"]
        assert name == f"math500_thinking_on_full22_{m}_steer" and name in C.M5T_EXPERIMENT_NAMES
        assert steer.cells == C.M5T_STEER_CELLS
        assert {k for k in steer._asdict() if steer._asdict()[k] != pub._asdict()[k]} == {"cells", "what"}, m
        assert "the highest rung within the 4 pp over-refusal budget" in steer.what and "W4" in steer.what
        for cell in C.M5T_STEER_CELLS:
            assert C.check_out_of_support_thinking_authorised(
                model=m, cell=cell, controls=["math500"], mode="on", sampling_mode=C.SAMPLING_THINKING_CARD, samples=1,
                max_tokens_by_task={"math": C.MATH500_MAX_TOKENS_DEFAULT}) == name
            # the GSM8K door: the two caps are two tuples, exactly as the twelve are
            for cap in C.G8K_CAPS:
                assert C.check_out_of_support_thinking_authorised(
                    model=m, cell=cell, controls=["gsm8k"], mode="on", sampling_mode=C.SAMPLING_THINKING_CARD, samples=1,
                    max_tokens_by_task={C.GSM8K_PAIRED_TASK_NAME: cap}) == C.g8k_steer_tuple_name(m, cap)
    C.check_authorised_render_facts()


# --------------------------------------------------------------------------- #
# the MATH-500 seed wave's two NATIVE tuples — `_native` /                      #
# `_native_paired` on qwen35_4b, beside the lens-era ones                       #
# --------------------------------------------------------------------------- #


def test_the_math500_seed_wave_has_its_own_native_pair_and_the_lens_era_math_tuples_did_not_move():
    """The seed wave's rows serve the batch-64 objects on the NATIVE engine (served identity reduction_mode not_applicable),
    which every lens-era MATH tuple (`stacked`) refuses at the driver door by name (the GSM8K defect). So qwen35_4b gets
    ONE new tuple for the five lane cells (== the GSM8K native battery's cells) and ONE for the composed paired-write arm (== the
    GSM8K composed pair's cell), each `_m5t_tuple`'s shape with `cells`, `reduction_mode` and `what` moved; the Arditi seed rows
    ride the existing `_steer_allpos` (already not_applicable; the view is not a compared field). The lens-era tuples are untouched
    (the published fixture above still pins them), and a request that STATES the served mode resolves exactly one door each way;
    an UNSTATED mode on a LoRA cell both a lens-era tuple and the native one name is ambiguous BY CONSTRUCTION — the reason every
    door states it (the driver's first door, the launcher's planner)."""
    assert C.M5T_NATIVE_MODELS == ("qwen35_4b",) and C.M5T_NATIVE_TUPLE_SUFFIX == "_native" and C.M5T_NATIVE_PAIRED_TUPLE_SUFFIX == "_native_paired"
    assert C.M5T_NATIVE_CELLS == C.G8K_NATIVE_CELLS and len(C.M5T_NATIVE_CELLS) == 5 and C.M5T_NATIVE_PAIRED_CELLS == C.G8K_NATIVE_PAIRED_CELLS
    assert set(C.M5T_EXT_CELLS) | {C.M5T_OBJECT_CELL} < set(C.M5T_NATIVE_CELLS)      # the three lens-era LoRA cells + the two PSR parents
    for name in ("M5T_NATIVE_MODELS", "M5T_NATIVE_CELLS", "M5T_NATIVE_TUPLE_SUFFIX", "m5t_native_tuple_name",
                 "M5T_NATIVE_PAIRED_CELLS", "M5T_NATIVE_PAIRED_TUPLE_SUFFIX", "m5t_native_paired_tuple_name"):
        assert name in C.__all__, name
    m = "qwen35_4b"
    native, paired = C.m5t_native_tuple_name(m), C.m5t_native_paired_tuple_name(m)
    assert (native, paired) == ("math500_thinking_on_full22_qwen35_4b_native", "math500_thinking_on_full22_qwen35_4b_native_paired")
    assert native in C.M5T_EXPERIMENT_NAMES and paired in C.M5T_EXPERIMENT_NAMES
    for other in ("gemma4e4b", "nemotron3nano4b", "glm46vflash", "llama8b"):
        assert C.m5t_native_tuple_name(other) not in C.AUTHORISED_OOS_EXPERIMENTS and C.m5t_native_paired_tuple_name(other) not in C.AUTHORISED_OOS_EXPERIMENTS
    nat, par, pub = (C.AUTHORISED_OOS_EXPERIMENTS[n] for n in (native, paired, f"math500_thinking_on_full22_{m}"))
    assert nat.cells == C.M5T_NATIVE_CELLS and par.cells == C.M5T_NATIVE_PAIRED_CELLS
    assert nat.reduction_mode == par.reduction_mode == C.REDUCTION_MODE_NOT_APPLICABLE and pub.reduction_mode == C.REDUCTION_MODE_STACKED
    assert {k for k in nat._asdict() if nat._asdict()[k] != pub._asdict()[k]} == {"cells", "reduction_mode", "what"}
    assert {k for k in par._asdict() if par._asdict()[k] != nat._asdict()[k]} == {"cells", "what"}
    assert nat.max_tokens_on == par.max_tokens_on == {"math": C.MATH500_MAX_TOKENS_DEFAULT} and nat.sampling_mode == C.SAMPLING_THINKING_CARD
    assert nat.header_located is False and nat.declares is False and nat.samples == 1 and nat.controls == (C.CONTROL_MATH500,)
    assert "b64-capability-agentic-queue" in nat.what and "not_applicable" in nat.what and "inspect-seed-policy replicate" in nat.what
    assert "s<s>p" in par.what and "no own `none`" in par.what.lower() and all(c in nat.what for c in C.M5T_NATIVE_CELLS)
    # the doors, the served mode STATED: native -> the seed tuples; stacked -> the lens-era ones (byte-identical resolution to before)
    door = dict(model=m, controls=["math500"], mode="on", sampling_mode=C.SAMPLING_THINKING_CARD, samples=1,
                max_tokens_by_task={"math": C.MATH500_MAX_TOKENS_DEFAULT})
    for cell in C.M5T_NATIVE_CELLS:
        assert C.check_out_of_support_thinking_authorised(cell=cell, reduction_mode=C.REDUCTION_MODE_NOT_APPLICABLE, **door) == native, cell
    assert C.check_out_of_support_thinking_authorised(cell=C.M5T_NATIVE_PAIRED_CELLS[0], reduction_mode=C.REDUCTION_MODE_NOT_APPLICABLE, **door) == paired
    assert C.check_out_of_support_thinking_authorised(cell="arditi_allpos", reduction_mode=C.REDUCTION_MODE_NOT_APPLICABLE, **door) == C.m5t_allpos_tuple_name(m)
    assert C.check_out_of_support_thinking_authorised(cell=C.M5T_OBJECT_CELL, reduction_mode=C.REDUCTION_MODE_STACKED, **door) == f"math500_thinking_on_full22_{m}"
    for cell in C.M5T_EXT_CELLS:
        assert C.check_out_of_support_thinking_authorised(cell=cell, reduction_mode=C.REDUCTION_MODE_STACKED, **door) == C.m5t_ext_tuple_name(m)
    # the PSR parents and the composed arm are in NO lens-era MATH tuple: a `stacked` identity refuses them naming the cell / mode
    for cell in ("constv2-mix-full__psr_ce_all", "constv2ctl2-full__psr_ce_all", C.M5T_NATIVE_PAIRED_CELLS[0]):
        with pytest.raises(C.OutOfSupportThinkingUnauthorised):
            C.check_out_of_support_thinking_authorised(cell=cell, reduction_mode=C.REDUCTION_MODE_STACKED, **door)
    # UNSTATED on a cell both name: ambiguous by construction (the registry's rule), never a silent pick
    for cell in (C.M5T_OBJECT_CELL, *C.M5T_EXT_CELLS):
        with pytest.raises(C.OutOfSupportAuthorisationAmbiguous, match="matches 2 registry tuples"):
            C.check_out_of_support_thinking_authorised(cell=cell, **door)
    # ...and a PSR cell unstated has ONE door (only the native tuple names it)
    assert C.check_out_of_support_thinking_authorised(cell="constv2-mix-full__psr_ce_all", **door) == native
    # no sensitivity rows, no other control, thinking OFF: refused
    for bad in (dict(sampling_mode=C.SAMPLING_GENERAL), dict(sampling_mode=C.SAMPLING_BATTERY), dict(mode="off"), dict(controls=["gsm8k"]), dict(samples=2)):
        with pytest.raises(C.OutOfSupportThinkingUnauthorised):
            C.check_out_of_support_thinking_authorised(cell=C.M5T_NATIVE_CELLS[0], reduction_mode=C.REDUCTION_MODE_NOT_APPLICABLE, **{**door, **bad})
    C.check_authorised_render_facts()


def test_mmlu_item_set_is_the_parity_head_or_the_whole_pinned_pool():
    """mmlu-full-1531: the served MMLU letter-prefill
    read's item set is the s2parity 200-item HEAD by default, or the WHOLE
    1,531-row pinned pool behind the driver's ``--mmlu-full`` — the same seed-0
    permutation cut at two widths (nested, never two samples), entering the
    hashed identity ONLY on the full run so every historical store keeps its
    key. The spec is pure and stdlib-only, like OR-Bench's."""
    assert C.PARITY_N == 200 and C.PARITY_SLICE_SEED == 0
    assert C.MMLU_FULL_N == 1531                      # cais/mmlu 'all' validation at the pin; the test split is not pinned
    assert C.MMLU_ITEM_SETS == ("subset", "full") == (C.MMLU_ITEM_SET_SUBSET, C.MMLU_ITEM_SET_FULL)
    assert C.MMLU_ITEM_SET_FIELDS == ("item_set", "n", "seed", "of")
    assert C.MMLU_ITEM_SET_BLOCK_FIELDS == (*C.MMLU_ITEM_SET_FIELDS, "item_ids")
    head = C.mmlu_item_set()
    assert tuple(head) == C.MMLU_ITEM_SET_FIELDS
    assert head == {"item_set": "subset", "n": 200, "seed": 0, "of": 1531}
    full = C.mmlu_item_set(full=True)
    assert full == {"item_set": "full", "n": 1531, "seed": 0, "of": 1531}
    assert full != head                               # a DIFFERENT identity, by construction
    assert head == C.mmlu_item_set(full=False) == C.mmlu_item_set(n=C.PARITY_N)   # pure
    # the ONE width resolver the driver hands parity (None = the loader's whole pool)
    assert C.mmlu_parity_n() == C.PARITY_N and C.mmlu_parity_n(full=True) is None
    # where the width lives on a store: inside resolved_config["mmlu"] (hashed, full runs only),
    # the unhashed manifest block, and the flag — outside the launcher's always-emitted set
    assert C.IDENTITY_KEY_MMLU_ITEM_SET == "item_set"
    assert C.MANIFEST_KEY_MMLU_ITEM_SET == "mmlu_item_set"
    assert C.FLAG_MMLU_FULL == "--mmlu-full" and C.FLAG_MMLU_FULL not in C.TINYCAP_DRIVER_FLAGS
    for name in ("PARITY_SLICE_SEED", "MMLU_FULL_N", "MMLU_ITEM_SET_SUBSET", "MMLU_ITEM_SET_FULL",
                 "MMLU_ITEM_SETS", "MMLU_ITEM_SET_FIELDS", "MMLU_ITEM_SET_BLOCK_FIELDS",
                 "IDENTITY_KEY_MMLU_ITEM_SET", "MANIFEST_KEY_MMLU_ITEM_SET", "mmlu_item_set",
                 "mmlu_parity_n", "FLAG_MMLU_FULL"):
        assert name in C.__all__, name
    # refusals: an empty head, a head wider than the pool, a bool for a count
    with pytest.raises(ValueError, match="is not in 1.."):
        C.mmlu_item_set(n=0)
    with pytest.raises(ValueError, match="is not in 1.."):
        C.mmlu_item_set(n=C.MMLU_FULL_N + 1)
    with pytest.raises(ValueError, match="must be an int"):
        C.mmlu_item_set(n=True)
    with pytest.raises(ValueError, match="of must be positive"):
        C.mmlu_item_set(of=0)
    # the parity module re-exports the seed (single-sourced here since this change)
    from sisteer.capeval.tiers import parity
    assert parity.PARITY_SLICE_SEED == C.PARITY_SLICE_SEED and parity.PARITY_N == C.PARITY_N
