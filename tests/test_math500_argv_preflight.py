"""math500-thinking-on-full22 — the INTEGRATION test across the cluster wrapper and the
harness: the driver argv the sbatch GENERATES for every row of the matrix is run through
the driver's REAL argument parser and preflight.

The producer is never re-typed: each row's ``DRIVER:`` line comes from ``m5t_dry_run.shell_plan``,
which runs ``math500_thinking.sbatch`` under ``M5T_DRY_RUN=1`` — the same ``m5t_common.sh``
``m5t_driver_argv`` the job executes. The consumer is ``served_capability_run.main`` on that argv,
from the repo root (the job ``cd``s to ``$REPO_DIR``), against a fake standup record carrying what
the preflight reads — ``lens_served_positions`` copied off the FRESHLY exported routing table (lane
B's own export path over a view-shaped bank), an empty declarations map with its error key null,
the per-cell policy ``always`` under the BARE key, the digests under the bare key, the stacked
reduction mode, ``proxy_upstream_timeout_s`` 7200 — the ledger's record beside it, and a CPU
prompt-budget artifact (``cap_task inspect/math``, 500 lengths) in the out root. The preflight is
run up to the FIRST network call (``_make_client`` raises a sentinel): every compare before it is
the real one — sampling row, ledger requirement, the authorised tuple at the driver door (the
matched name must be the row's), the record binding, served positions, the parser gate on the
family fact, the attestation, the stamp policies, the version-2 cap statement, the served
reduction mode against the tuple, the cell pick and the route.

Also here: the dry-run CLI prints the same ``DRIVER:`` lines as the per-row producer; a gemma
``battery`` row is refused by the sampling table before any record is read; a fitted row whose
served map is header-located (or absent) is refused by name; a ``candidate`` fact (a SYNTHETIC
demotion of one of the tree's — every real fact is certified post G-PARSER) refuses production by name
and ``--diagnostic`` admits it, while the tree's own facts admit; ``--stage prod`` reads certification
through ``contracts.check_render_fact_certified`` (a spy); and the two doors — the driver's shape
and the launcher's (its real planner where its model registry reaches) — name the same tuple for
each fitted row.

Post-m4 (decisions-integration I3, closed): ``contracts.MATH500_MAX_TOKENS_DEFAULT`` IS the
experiment's cap (S1, merged by m4), so the sixteen bare / prompted rows' ``--max-tokens-by-task
math=16384`` statement equals the resolved cap and every row is a REQUIRED pass — before m4 those
rows asserted the driver's by-name refusal and xfailed keyed on that equality; the equality is now
a pin of its own. Everything here is SYNTHETIC (``TESTHOST``, placeholder URLs, a synthetic bank);
nothing is served.
"""
from __future__ import annotations

import importlib.util
import json
import shlex
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "math500",
          REPO / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cells as M
import served_capability_run as srun
from test_sdfbase_full22_wave import _stage, _stage_derived
from test_served_capability import stub_framework_versions

from sisteer.capeval.tinycap import contracts as C


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: lane B's dry run — the python producer that drives the sbatch's shell producer
DRY = _load("_m5t_dry_run_argv_preflight", "scripts/sdfbase/math500/m5t_dry_run.py")
#: the client-venv python the argv names (a placeholder: the argv is PARSED here, never executed)
PY = "/venv/bin/python"
#: the served chat template's 16-hex digest — the standup record's and the budget artifact's, equal
TEMPLATE_HASH = "ab" * 8
#: the rows this file's per-row preflights drive: the base wave's and the extension rows'. The steering rows serve ANOTHER LANE's
#: bank through their own overlay pair, so the campaign-bank world `build_world` stages is not their world — their
#: end-to-end chain (overlay -> select_cells -> registry -> the external bank check) is
#: tests/test_sdfbase_math500_sbatch.py::test_the_w4_steering_rows_serve_their_own_overlay_and_external_bank_end_to_end
CAMPAIGN_BANK_ROWS = [r for r in M.M5T_CELLS if r.get("object_source") != "external"]
FITTED_ROWS = [r for r in CAMPAIGN_BANK_ROWS if r["fitted"]]
EXTERNAL_ROWS = [r for r in M.M5T_CELLS if r.get("object_source") == "external"]
#: E3: a HELD row (every it-doc row today: the re-fit objects are staged under their ruled ids but not yet servable) prints its
#: HELD line and NO argv, so the driver-preflight walk runs over the LAUNCHABLE campaign-bank rows; the held ones have their own check
LAUNCHABLE_CAMPAIGN_ROWS = [r for r in CAMPAIGN_BANK_ROWS if r["status"] != M.STATUS_HELD]
HELD_CAMPAIGN_ROWS = [r for r in CAMPAIGN_BANK_ROWS if r["status"] == M.STATUS_HELD]
#: arditi-qwen-all-positions: the rows whose CELL this tree's harness grammar does not admit — DERIVED from what the
#: vocabulary actually refuses (never a row index, which would go stale), and cross-checked against the launcher's own
#: declaration of the gap (`cells.PENDING_SERVING_CELLS`) so the two cannot drift apart. Such a row is launch-ready in
#: every other respect: the serving-side Class B registration lands in another author's PR, and until it does the
#: kind-aware serving-path selector refuses the row's launch cells BY NAME. The whole-matrix dry-run walk below names
#: it and skips by name; nothing else in this file reaches it (its object is EXTERNAL, so it is outside
#: CAMPAIGN_BANK_ROWS and outside FITTED_ROWS).
def _grammar_admits(cell: str) -> bool:
    try:
        C.parse_launcher_cell(cell)
    except C.CellNameError:
        return False
    return True


PENDING_SERVING_ROWS = [r for r in M.M5T_CELLS if not _grammar_admits(r["cell"])]
assert {r["cell"] for r in PENDING_SERVING_ROWS} <= set(M.PENDING_SERVING_CELLS), (
    "a matrix cell the harness grammar refuses is not declared in cells.PENDING_SERVING_CELLS")
#: the matrix as imported — captured because one test hands the CLI a narrowed row list (the pending rows above are
#: walked separately there) and must still count against the WHOLE matrix
ALL_ROWS = list(M.M5T_CELLS)
CANDIDATE_TREE = {fam for fam, f in C.FAMILY_RENDER_FACTS.items()
                  if f.evidence_status != C.RENDER_EVIDENCE_CERTIFIED}
#: llama-paper-rows: the no-switch family's fact (split `none`) entered as this tree's ONE candidate and was certified by
#: the GSM8K paired-200 Llama wave's own smoke (2026-09-12) — never a MATH-500 family (Llama is excluded from MATH-500 by
#: decision), so every fact on the tree is certified; the candidate refusals below run on SYNTHETIC demotions
CANDIDATE_TREE_EXPECTED: set = set()
#: The third ``check_out_of_support_thinking_authorised`` call the driver makes on a fitted row:
#: the matched tuple re-run against the RESOLVED served model's key
WHERE_SERVED_MODEL = "driver run (served model)"


class FirstNetworkCall(Exception):
    """The sentinel ``_make_client`` raises: everything before it passed."""

    def __init__(self, proxy_url, route):
        super().__init__(f"first network call: {proxy_url} for {route.virtual_model}")
        self.proxy_url = proxy_url
        self.route = route


@pytest.fixture(scope="module")
def s_by_model():
    """S per model at the published (short) geometry entry, AND per (model, geometry entry) for every row — the dry
    run's own resolution (never typed). Since the eight-arm extension the geometry is PER ROW (the constitution-SI
    rows run at capability:gen_long_math_longsi, L 24,064, S 24 / 29 / 32), so a row's S is read under its
    ``(model_key, geometry_eval)`` key."""
    out = {k: DRY.resolved_s(k) for k in M.MODEL_KEYS}
    for r in M.M5T_CELLS:
        out[(r["model_key"], r["geometry_eval"])] = DRY.resolved_s_for_row(r)
    return out


def standup_geometry_block(row: dict, s: int) -> dict:
    """The record's ``geometry`` block as the standup writes it: THE ROW's entry resolved through the standup's OWN
    resolver under the model's budget (a key-less resolution would hold S to the llama-certified budget and refuse
    gemma's 33 and glm's / nemotron's 44 at L 17,664)."""
    from capctrl_serving_up import resolve_standup_geometry
    geom = resolve_standup_geometry(row["geometry_eval"], None, None, model_key=row["model_key"])
    block = geom.as_dict()
    assert (int(block["max_model_len"]), int(block["max_num_seqs"])) == (row["geometry_L"], s), block
    return block


# --------------------------------------------------------------------------- #
# the producers: lane B's shell + python, verbatim                            #
# --------------------------------------------------------------------------- #


def produced_lines(row: dict, *, s: int, shared: Path) -> dict[str, str]:
    """The sbatch's dry-run printing for one row, keyed by line tag (ROW / STANDUP / LEDGER / DRIVER / …)."""
    text = DRY.shell_plan(row, s=s, shared=shared, python=PY)
    lines: dict[str, str] = {}
    for line in text.splitlines():
        tag = line.split(":", 1)[0]
        assert tag not in lines, (tag, text)          # one line per tag per row
        lines[tag] = line
    return lines


def driver_argv(lines: dict[str, str]) -> list[str]:
    """``DRIVER:`` -> the argv ``served_capability_run.main`` receives (after the interpreter and the script)."""
    words = shlex.split(lines["DRIVER"][len("DRIVER:"):])
    assert words[:3] == [PY, "scripts/served_capability_run.py", "generate"], words[:3]
    return words[2:]


def flag_values(argv: list[str]) -> dict[str, list[str]]:
    """``{flag: [values]}`` — a store_true flag maps to ``[]``; repeats accumulate."""
    out: dict[str, list[str]] = {}
    i = 1                                             # argv[0] is the sub-command
    while i < len(argv):
        tok = argv[i]
        assert tok.startswith("--"), (tok, argv)
        vals = out.setdefault(tok, [])
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            vals.append(argv[i + 1])
            i += 2
        else:
            i += 1
    return out


# --------------------------------------------------------------------------- #
# the world the argv is run against                                            #
# --------------------------------------------------------------------------- #


def _budget_doc(cfg: dict, key: str) -> dict:
    """The CPU prompt-budget artifact for this cell class: MATH-500's 500 prompts under the ON render."""
    lengths = [400 + (i * 37) % 900 for i in range(500)]
    doc = {"n": len(lengths), "max": max(lengths), "p99": C.nearest_rank_percentile(lengths, 0.99),
           "p50": C.nearest_rank_percentile(lengths, 0.50), "mean": round(sum(lengths) / len(lengths), 2),
           "arm": C.THINKING_ON, "model": cfg["model"], "model_slug": key, "revision": cfg["model_revision"],
           "chat_template_hash": TEMPLATE_HASH, "cap": M.CAP, "cap_task": C.inspect_task(M.INSPECT_TASK),
           "max_plus_cap": max(lengths) + M.CAP, "lengths": lengths,
           "render": {"add_generation_prompt": True, **C.chat_template_kwargs(C.THINKING_ON)},
           "messages_source": "sisteer.capeval.tiers.inspect_served.math500_messages", "tree_sha": None,
           "produced_by": {"script": C.OOS_PROMPT_BUDGET_SCRIPT, "tokenizer": "synthetic"}}
    return C.validate_oos_prompt_budget(doc)


def build_world(tmp_path: Path, row: dict, *, s: int, served_positions="fresh", drop_keys=()) -> SimpleNamespace:
    """The row's generated argv plus everything on disk the preflight reads, at the paths the argv names.

    ``served_positions``: ``"fresh"`` copies the map the FRESH routing table carries (the standup
    echoes the table's map into endpoint.json); a mapping overrides it (a header-located record)."""
    shared = tmp_path / "shared"
    lines = produced_lines(row, s=s, shared=shared)
    argv = driver_argv(lines)
    fl = flag_values(argv)
    key = row["model_key"]
    table_path = Path(fl["--routing-table"][0])
    bank = Path(fl["--defenses-dir"][0])
    out_root = Path(fl["--out-root"][0])
    proxy_url, ledger_url = fl["--proxy-url"][0], fl[M.FLAG_LEDGER_URL][0]
    budget_path = Path(fl[M.FLAG_OOS_PROMPT_BUDGET][0])
    cfg = yaml.safe_load((REPO / fl["--config"][0]).read_text())
    # the bank view the argv names, staged the way the view stages the object cells: the DERIVED minus_ctl object
    # (derived_meta.json) and, for the eight-arm extension's rows, the two BASE LoRA objects (a budget sidecar)
    _stage_derived(bank, M.CELL_OBJECT, M.CELL_OBJECT)
    for lora in (M.CELL_LORA_MIX, M.CELL_LORA_CTL2):
        _stage(bank, lora, lora)
    bank_root = bank.parents[2]
    assert bank == bank_root / key / "cells_view" / "main", bank
    # the FRESH routing table over it, through lane B's own export path (the standup's cell selection
    # + export_routing_table), landed where the argv's --routing-table points
    fresh = tmp_path / "fresh"
    fresh.mkdir(exist_ok=True)
    st = DRY.fresh_table_stamp(row, bank_root, fresh)
    assert st["exported"] and st["own_route"], st
    table_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fresh / f"row{row['index']}_routing_table.json", table_path)
    table = json.loads(table_path.read_text())
    positions = table.get(C.LENS_SERVED_POSITIONS_KEY) if served_positions == "fresh" else served_positions
    bare = row["harness_cell"]
    ep = {"proxy_url": proxy_url, "routing_table": str(table_path), "model": cfg["model"], "model_key": key,
          "cell": row["cell"], "harness_cell": bare, "pid": 0, "node": "TESTHOST", "eval": row["geometry_eval"],
          "reasoning_parser": row["reasoning_parser"], "tool_calls": False, C.AGENT_LOOP_ENTRY_KEY: True,
          "geometry": standup_geometry_block(row, s), "max_model_len": row["geometry_L"], "max_num_seqs": s,
          srun.STANDUP_TEMPLATE_HASH_KEY: TEMPLATE_HASH,
          C.PROXY_IDENTITY_REDUCTION_MODE: C.REDUCTION_MODE_STACKED,
          C.PROXY_IDENTITY_SERVE_DECLARATIONS: {}, C.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR: None,
          C.ENDPOINT_KEY_LENS_SALT_POLICY: C.LENS_SALT_POLICY_ATTESTED,
          C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: ({bare: "always", C.TWIN_CELL_ID: "always"}
                                                       if row["fitted"] else {}),
          C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S: float(M.UPSTREAM_TIMEOUT_S),
          C.LENS_SERVED_POSITIONS_KEY: positions,
          "lens_served_positions_passthrough": "passed"}
    if row["fitted"]:
        ep[C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256] = row["object_sha256"]
        ep[C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL] = {bare: row["object_sha256"]}
    for k in drop_keys:
        ep.pop(k, None)
    (table_path.parent / srun.STANDUP_ENDPOINT_FILENAME).write_text(json.dumps(ep, indent=1))
    # the usage ledger's record beside the table (m5t_ledger_up's --endpoint-json)
    (table_path.parent / C.LEDGER_ENDPOINT_FILENAME).write_text(json.dumps(
        {"ledger_url": ledger_url, "upstream": proxy_url, "ledger_path": str(out_root / C.LEDGER_FILENAME),
         "pid": 0, "probe": {"upstream": proxy_url, "virtual_models": [], "probe_prompt_tokens": 3}}))
    # the budget artifact for this cell class, copied into the out root as the sbatch does
    out_root.mkdir(parents=True, exist_ok=True)
    assert budget_path == out_root / M.OOS_PROMPT_BUDGET_FILENAME
    budget_path.write_text(json.dumps(_budget_doc(cfg, key)))
    return SimpleNamespace(argv=argv, flags=fl, lines=lines, table_path=table_path, table=table, endpoint=ep,
                           proxy_url=proxy_url, ledger_url=ledger_url, budget_path=budget_path, bank=bank,
                           out_root=out_root, cfg=cfg)


def arm_driver(monkeypatch, *, facts) -> None:
    """The process the job runs the driver in: from the repo root, CPU-only, the stacked switch exported
    (common.sh), the framework pins stubbed, the fit-time provenance mirror stubbed (a registry lookup
    outside this contract), and the FIRST network call turned into the sentinel."""
    monkeypatch.chdir(REPO)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv(C.LENS_STACKED_REDUCTIONS_ENV, "1")
    stub_framework_versions(monkeypatch)
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", facts)
    monkeypatch.setattr(srun, "_mirror_provenance", lambda cell, **kw: (
        None if getattr(cell, "object_path", None) is None else SimpleNamespace(
            model_id=None, revision=None, trained_template_hash=TEMPLATE_HASH, object_sha="ab" * 32,
            status="verified", source="synthetic", positions_at_fit="all")))

    def first_call(proxy_url, route):
        raise FirstNetworkCall(proxy_url, route)
    monkeypatch.setattr(srun, "_make_client", first_call)


def certified_facts() -> dict:
    """Every family fact flipped to ``certified`` — the G-DRY-prod state (only ``evidence_status`` moves)."""
    return {fam: f._replace(evidence_status=C.RENDER_EVIDENCE_CERTIFIED) for fam, f in C.FAMILY_RENDER_FACTS.items()}


def spy_checker(monkeypatch) -> list:
    """Record every call to the ONE registry-aware checker (kwargs, matched name)."""
    calls: list = []
    real = C.check_out_of_support_thinking_authorised

    def spy(**kw):
        name = real(**kw)
        calls.append((dict(kw), name))
        return name
    monkeypatch.setattr(C, "check_out_of_support_thinking_authorised", spy)
    return calls


def spy_sampling(monkeypatch) -> list:
    rows: list = []
    real = C.sampling_for

    def spy(model_key, mode=C.SAMPLING_CARD):
        rows.append((model_key, mode))
        return real(model_key, mode)
    monkeypatch.setattr(C, "sampling_for", spy)
    return rows


# --------------------------------------------------------------------------- #
# every row's argv through the driver's parser + preflight                     #
# --------------------------------------------------------------------------- #


def test_the_in_support_cap_is_the_experiments_cap():
    """m4 merged S1: ``contracts.MATH500_MAX_TOKENS_DEFAULT`` is the experiment's cap, so the
    sixteen bare / prompted rows' ``math=16384`` statement equals the cap the plan builder resolves
    and those rows are required passes below (decisions-integration I3: they xfailed keyed on this
    equality before m4). A tree where this fails makes the driver refuse those rows by name."""
    assert C.MATH500_MAX_TOKENS_DEFAULT == M.CAP == 16384        # The ceiling; the fitted tuples name the same cap
    for name in C.M5T_EXPERIMENT_NAMES:
        assert C.AUTHORISED_OOS_EXPERIMENTS[name].max_tokens_on == {"math": M.CAP}, name


@pytest.mark.parametrize("index", [r["index"] for r in HELD_CAMPAIGN_ROWS], ids=[r["job_id"] for r in HELD_CAMPAIGN_ROWS])
def test_every_held_row_prints_held_and_no_argv_so_there_is_nothing_to_preflight(index, s_by_model):
    """E3: a HELD it-doc row (its record binds no servable bank yet) is a design status — the sbatch's dry stub prints the HELD line
    with the record's reason and exits before any STANDUP / LEDGER / DRIVER argv exists, so the preflight above has nothing to walk."""
    row = M.M5T_CELLS[index]
    assert row["status"] == M.STATUS_HELD and row.get("object_source") == M.OBJECT_SOURCE_ITDOC and row["held_reason"]
    lines = produced_lines(row, s=s_by_model[row["model_key"], row["geometry_eval"]], shared=Path("/tmp/m5t-argv-held") / "shared")
    assert set(lines) == {"HELD"} and lines["HELD"].startswith(f"HELD: index={row['index']} job_id={row['job_id']} wave={M.WAVE_ITDOC} reason=")
    assert row["held_reason"] in lines["HELD"]


@pytest.mark.parametrize("index", [r["index"] for r in LAUNCHABLE_CAMPAIGN_ROWS], ids=[r["job_id"] for r in LAUNCHABLE_CAMPAIGN_ROWS])
def test_every_rows_generated_argv_passes_the_driver_preflight_up_to_the_first_network_call(
        tmp_path, monkeypatch, index, s_by_model):
    row = M.M5T_CELLS[index]
    w = build_world(tmp_path, row, s=s_by_model[row["model_key"], row["geometry_eval"]])
    calls = spy_checker(monkeypatch)
    rows_seen = spy_sampling(monkeypatch)
    arm_driver(monkeypatch, facts=certified_facts())
    # the argv's literals are the matrix's, flag for flag (the spellings are contracts', pinned equal by lane B)
    f = w.flags
    no_switch = row["thinking_mode"] == M.THINKING_NOT_AVAILABLE           # E2: the llama wave (P7)
    assert f["--cell"] == [row["cell"]] and f["--controls"] == [M.CONTROL]
    assert f[M.FLAG_THINKING] == [row["thinking_mode"]] and f[M.FLAG_SAMPLING] == [row["sampling_mode"]]
    assert f[M.FLAG_SAMPLES] == [str(M.SAMPLES)] and f[M.FLAG_MATH500_RETRY_MAX_TOKENS] == [str(M.RETRY_MAX_TOKENS)]
    assert f[M.FLAG_MAX_TOKENS_BY_TASK] == [M.MAX_TOKENS_BY_TASK] == [f"math={M.CAP}"]
    # under the no-switch arm the bounded SETTER is REQUIRED and IS the cap (inspect_served.task_max_tokens; contracts admits `math`
    # at the ceiling); a switch family's row never names it
    assert f.get(C.FLAG_ON_CAP_BY_TASK) == ([M.MAX_TOKENS_BY_TASK] if no_switch else None)
    assert f[M.FLAG_TINYCAP_IDENTITY_VERSION] == [str(M.IDENTITY_VERSION)] == ["2"]
    assert f[M.FLAG_LEDGER_URL] == [w.ledger_url] and f[M.FLAG_OOS_PROMPT_BUDGET] == [str(w.budget_path)]
    assert f["--concurrency"] == [str(s_by_model[row["model_key"], row["geometry_eval"]])]
    # the OOS override rides a TUPLE, never fittedness: a no-switch family's fitted row is IN SUPPORT (pin, no override)
    assert (M.FLAG_ALLOW_OOS_THINKING in f) is bool(row["tuple_name"]) and (row["fitted"] and not row["tuple_name"]) is (no_switch and row["fitted"])
    assert f.get(M.FLAG_EXPECT_OBJECT_SHA256) == ([row["object_sha256"]] if row["fitted"] else None)
    assert "@thinking" not in w.lines["DRIVER"] and "petri_thinking_on" not in w.lines["DRIVER"]
    # the STANDUP line names the same cell, and the same pin under the cell's spelling, iff fitted
    standup = shlex.split(w.lines["STANDUP"][len("STANDUP:"):])
    assert [standup[i + 1] for i, t in enumerate(standup) if t == "--cell"] == [row["cell"]]
    digests = [standup[i + 1] for i, t in enumerate(standup) if t == M.FLAG_EXPECT_OBJECT_SHA256]
    assert digests == ([f"{row['cell']}={row['object_sha256']}"] if row["fitted"] else [])
    # the fresh table carries the served-positions stamp for the fitted cell, and the record echoes it
    if row["fitted"]:
        assert w.table[C.LENS_SERVED_POSITIONS_KEY] == {row["harness_cell"]: ["all"]}
        assert w.endpoint[C.LENS_SERVED_POSITIONS_KEY] == {row["harness_cell"]: ["all"]}
    with pytest.raises(FirstNetworkCall) as e:
        srun.main(w.argv)
    # the route the driver resolved: the row's OWN cell under its PLAIN spelling, on the proxy the argv names
    assert e.value.route.cell_id == row["harness_cell"]
    assert e.value.route.virtual_model == f"petri__{row['harness_cell']}" if row["cell"] != M.CELL_NONE \
        else e.value.route.virtual_model == "petri__base"
    assert "@thinking" not in e.value.route.virtual_model and e.value.proxy_url == w.proxy_url
    # the sampling row resolved for the row's (model, mode) — the table's row, no default temperature
    assert rows_seen and set(rows_seen) == {(row["model_key"], row["sampling_mode"])}
    if row["fitted"] and not row["tuple_name"]:
        assert calls == [] and no_switch                                  # in support: no override named, no tuple consulted
    elif row["fitted"]:
        # the door: the ONE registry-aware checker matched exactly the row's tuple, from the parsed argv PLUS
        # the served reduction mode read off the standup record beside --routing-table (the first door STATES
        # the mode, so a native twin differing only in it never makes a lens-era lane's door ambiguous). This
        # world writes a LENS record stating `stacked` beside the table, so that is what the door states —
        # the pin is the fixture's own value, not a constant; were the record absent the mode would
        # legitimately be unstated here and the later full read would refuse (PB13); then again with the
        # served (stacked) reduction mode off the record — the same tuple; then a third time BOUND to the
        # resolved SERVED model's key (the routing table's base_model pins the row's own key, so the same
        # tuple once more, under its own `where`)
        assert [name for _, name in calls] == [row["tuple_name"]] * 3, calls
        door, served, bound = (kw for kw, _ in calls)
        assert (door["model"], door["cell"], list(door["controls"]), door["mode"]) == (
            row["model_key"], row["cell"], [M.CONTROL], row["thinking_mode"])
        assert (door["sampling_mode"], door["samples"], door["where"]) == (row["sampling_mode"], M.SAMPLES, "driver run")
        assert (door["reduction_mode"] == served["reduction_mode"] == C.REDUCTION_MODE_STACKED
                == w.endpoint[C.PROXY_IDENTITY_REDUCTION_MODE])
        assert (served["model"], served["where"]) == (row["model_key"], "driver run")
        assert (bound["model"], bound["reduction_mode"], bound["where"]) == (
            row["model_key"], C.REDUCTION_MODE_STACKED, WHERE_SERVED_MODEL)
    else:
        assert calls == []                            # no override named, no tuple consulted


def test_the_dry_run_cli_prints_the_same_driver_lines_as_the_per_row_producer(tmp_path, capsys, monkeypatch, s_by_model):
    """``m5t_dry_run.py --stage pre`` end to end: its 24 ``DRIVER:`` lines are byte-identical to
    ``shell_plan``'s per row (one producer, two entry points), and its ``--stage pre`` assertion
    runs through lane A's seam for every family — refusing exactly the tree's candidate facts.

    arditi-qwen-all-positions: a row whose cell this tree's harness grammar does not admit yet is walked SEPARATELY and
    positively — the CLI must refuse it BY NAME — and is held out of the every-launchable-row comparison, so this test
    reads "N rows, N-1 walked, 1 pending on the named external registration" rather than dying or quietly walking N-1."""
    tmp = tmp_path / "t"
    argv = ["--stage", "pre", "--tmp", str(tmp), "--python", PY]
    # EACH pending row, POSITIVELY: `--only` it and require the CLI's own REFUSED, carrying the selector's verbatim
    # message. The kind-aware serving-path selector refuses its launch cells because the Class B registration lands in
    # another author's PR; `m5t_dry_run.shell_plan` turns that into its own SystemExit, which is the deliverable of that
    # module and stays exactly as it is. The set is DERIVED from what the vocabulary refuses, never a row index or a
    # cell literal, so when the registration lands it empties by itself, the row joins the walk below, and nothing here
    # needs editing — the assertion after the loop is what pins that property.
    for p in PENDING_SERVING_ROWS:
        with pytest.raises(SystemExit) as exc:
            DRY.main(["--stage", "pre", "--tmp", str(tmp_path / f"pending{p['index']}"), "--python", PY,
                      "--only", p["model_key"], p["cell"], p["sampling_mode"], "--only-view", M.key_view(p)])
        why = str(exc.value)
        capsys.readouterr()
        assert f"REFUSED: the sbatch dry run failed for row {p['index']}" in why, p["job_id"]
        assert f"[serving-path] REFUSED: DefenseCellError: malformed cell id {p['cell']!r}" in why, p["job_id"]
    walked = [r for r in ALL_ROWS if r["index"] not in {p["index"] for p in PENDING_SERVING_ROWS}]
    assert len(walked) + len(PENDING_SERVING_ROWS) == len(ALL_ROWS)
    assert (walked == ALL_ROWS) is (not PENDING_SERVING_ROWS)      # nothing pending => EVERY row is walked below
    if PENDING_SERVING_ROWS:
        monkeypatch.setattr(M, "M5T_CELLS", walked)                # the rows the CLI walks; its producer is untouched
    rc = DRY.main(argv)
    out = capsys.readouterr().out
    assert rc == 0 and "VERDICT: PASS stage=pre" in out, out[-1500:]
    cli = [line for line in out.splitlines() if line.startswith("DRIVER:")]
    # published + the extension rows + the steering rung + the llama wave + the it-doc rows;
    # a HELD row (every it-doc row today) prints HELD and no argv
    launchable = [r for r in walked if r["status"] != M.STATUS_HELD]
    assert len(ALL_ROWS) == 24 + 9 + len(EXTERNAL_ROWS) + len(M.LLAMA_ROWS) + len(M.ITDOC_ROWS) and len(cli) == len(launchable)
    for row, line in zip(launchable, cli):
        lines = produced_lines(row, s=s_by_model[row["model_key"], row["geometry_eval"]], shared=tmp / "shared")
        assert lines["DRIVER"] == line, row["job_id"]
    for fam in sorted({r["family"] for r in walked}):
        # a family the contracts carry NO fact for refuses too — and by a different clause of the same seam (no
        # entry at all, vs an entry whose evidence is only `candidate`). The steering rung's no-switch family is
        # in that state on this tree, which is exactly why its rows are HELD; the mirror the launcher reads is
        # pinned to the seam.
        fact = C.FAMILY_RENDER_FACTS.get(fam)
        refuses = fact is None or fact.evidence_status != C.RENDER_EVIDENCE_CERTIFIED
        assert refuses is not M.RENDER_FACT_CERTIFIED_BY_FAMILY[fam]
        assert (f"PREFLIGHT_REFUSES_CANDIDATE: family={fam} seam=contracts.check_render_fact_certified "
                f"refuses={refuses}") in out, fam
    assert out.count("G-LENS: ") == 1 and "refinement merged at" in out


# --------------------------------------------------------------------------- #
# the refusals: by name, before any record is read where the rule allows        #
# --------------------------------------------------------------------------- #


def test_a_battery_row_for_gemma_is_refused_by_the_sampling_table_before_any_record_is_read(
        tmp_path, monkeypatch, s_by_model):
    """gemma's battery preset equals its primary: the matrix has no such row, the contracts table
    has no such row (``SamplingUnverified``), and a hand-edited argv asking for it refuses in the
    driver's sampling resolution — before the standup record is even opened."""
    with pytest.raises(SystemExit, match="not a matrix row"):
        M.row_for("gemma4e4b", M.CELL_OBJECT, M.SAMPLING_BATTERY)
    with pytest.raises(C.SamplingUnverified, match="BATTERY_SAMPLING_BY_MODEL_KEY"):
        C.sampling_for("gemma4e4b", C.SAMPLING_BATTERY)
    row = M.row_for("gemma4e4b", M.CELL_OBJECT, M.SAMPLING_THINKING_CARD)
    w = build_world(tmp_path, row, s=s_by_model["gemma4e4b"])
    arm_driver(monkeypatch, facts=certified_facts())
    argv = list(w.argv)
    argv[argv.index(M.FLAG_SAMPLING) + 1] = M.SAMPLING_BATTERY
    (w.table_path.parent / srun.STANDUP_ENDPOINT_FILENAME).unlink()      # a later read would refuse differently
    with pytest.raises(SystemExit) as e:
        srun.main(argv)
    msg = str(e.value)
    assert "has no verified sampling row" in msg and "BATTERY_SAMPLING_BY_MODEL_KEY" in msg, msg
    assert "gemma4e4b" in msg and "no standup record" not in msg


@pytest.mark.parametrize("name,positions,drop", [
    ("header_located", ["response"], ()),
    ("mixed_stacked", ["all", "response"], ()),
    ("map_absent", None, (C.LENS_SERVED_POSITIONS_KEY,)),
])
def test_a_fitted_row_whose_served_map_is_header_located_or_absent_is_refused_by_name(
        tmp_path, monkeypatch, s_by_model, name, positions, drop):
    """On the REAL fitted argv: a record that says the served mask is header-located
    (or cannot say) makes this an undeclared header-located configuration, which the record does
    not attest — refused ``OOS_DECLARATION_UNATTESTED`` naming the declaration the cell would need."""
    row = M.row_for("qwen35_4b", M.CELL_OBJECT, M.SAMPLING_THINKING_CARD)
    served = {row["harness_cell"]: positions} if positions is not None else {}
    w = build_world(tmp_path, row, s=s_by_model["qwen35_4b"], served_positions=served, drop_keys=drop)
    calls = spy_checker(monkeypatch)
    arm_driver(monkeypatch, facts=certified_facts())
    with pytest.raises(SystemExit) as e:
        srun.main(w.argv)
    msg = str(e.value)
    assert C.REC_OOS_DECLARATION_UNATTESTED in msg and "@thinking" in msg, (name, msg[:400])
    assert "attests no serve_out_of_support" in msg
    # the tuple was matched first (the door), the record refused after it — never the other way round
    assert [n for _, n in calls] == [row["tuple_name"]]


def test_a_candidate_fact_refuses_production_by_name_and_the_diagnostic_flag_admits_it(
        tmp_path, monkeypatch, s_by_model):
    """G-DRY-pre on the real fitted argv: a family whose fact is ``candidate`` is refused by the
    production preflight naming the gate; ``--diagnostic`` (identity version 2) admits it — the smoke
    that produces the completions G-PARSER certifies. Every fact on this tree is certified (post
    G-PARSER, both smokes), so the candidate is a SYNTHETIC demotion of the Nemotron fact (only
    ``evidence_status`` moves) — and the tree's OWN facts admit the same production argv to the first
    network call (the G-DRY-prod state, on the driver door)."""
    assert CANDIDATE_TREE == CANDIDATE_TREE_EXPECTED, CANDIDATE_TREE
    assert not (CANDIDATE_TREE & set(M.FAMILY_BY_MODEL.values())), "a MATH-500 family's fact is a candidate"
    fam = "nemotron_h"
    key = next(k for k, f in M.FAMILY_BY_MODEL.items() if f == fam)
    row = M.row_for(key, M.CELL_OBJECT, M.SAMPLING_THINKING_CARD)
    w = build_world(tmp_path, row, s=s_by_model[key])
    real = dict(C.FAMILY_RENDER_FACTS)
    demoted = {**real, fam: real[fam]._replace(evidence_status=C.RENDER_EVIDENCE_CANDIDATE)}
    arm_driver(monkeypatch, facts=demoted)
    with pytest.raises(SystemExit) as e:
        srun.main(w.argv)
    assert "render fact not certified" in str(e.value) and fam in str(e.value)
    with pytest.raises(FirstNetworkCall):
        srun.main(w.argv + [C.FLAG_DIAGNOSTIC])
    arm_driver(monkeypatch, facts=real)                                 # the tree's own facts: admitted
    with pytest.raises(FirstNetworkCall):
        srun.main(w.argv)


def test_stage_prod_reads_certification_through_the_contracts_seam(tmp_path, monkeypatch, capsys):
    """``--stage prod`` calls ``contracts.check_render_fact_certified(family)`` (the seam shipped for
    the dry run and the driver) for every family it plans, and refuses when the seam refuses — never
    by re-deriving the verdict from the fact's fields."""
    seen: list = []
    real = C.check_render_fact_certified

    def spy(family, **kw):
        seen.append(str(family))
        return real(family, **kw)
    monkeypatch.setattr(C, "check_render_fact_certified", spy)
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", certified_facts())
    bank_root = tmp_path / "bank"
    view = bank_root / "qwen35_4b" / "cells_view" / "main"
    _stage_derived(view, M.CELL_OBJECT, M.CELL_OBJECT)
    for lora in (M.CELL_LORA_MIX, M.CELL_LORA_CTL2):      # the overlay lists every cell of the model's rows; the view stages them all
        _stage(view, lora, lora)
    only = ["--only", "qwen35_4b", M.CELL_OBJECT, M.SAMPLING_THINKING_CARD, "--tmp", str(tmp_path / "t"),
            "--bank-root", str(bank_root), "--python", PY]
    assert DRY.main(["--stage", "prod", *only]) == 0
    out = capsys.readouterr().out
    assert seen == ["qwen3_5"] and "VERDICT: PASS stage=prod" in out
    assert "PREFLIGHT_ADMITS: family=qwen3_5 seam=contracts.check_render_fact_certified admits=True" in out
    # the seam refusing IS the stage's refusal, quoted
    seen.clear()
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", {**certified_facts(), "qwen3_5": C.FAMILY_RENDER_FACTS["qwen3_5"]
                                                   ._replace(evidence_status=C.RENDER_EVIDENCE_CANDIDATE)})
    assert DRY.main(["--stage", "prod", *only]) == 2
    out = capsys.readouterr().out
    assert seen == ["qwen3_5"]
    assert "REFUSED: render fact not certified for family qwen3_5" in out and "RenderFactUnverified" in out
    # a tree without the seam cannot certify anything at prod
    seen.clear()
    monkeypatch.delattr(C, "check_render_fact_certified")
    assert DRY.main(["--stage", "prod", *only]) == 2
    assert "no render-fact preflight seam" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the two doors name the same tuple                                            #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def launcher():
    """The Modal launcher loaded against the launcher tests' modal stub (no Modal, no network)."""
    from test_tinycap_launcher import TEST_APP_NAME, _load_launcher, _stub_modal
    return _load_launcher(_stub_modal(), app_name=TEST_APP_NAME, name="tinycap_served_m5t_argv_preflight")


@pytest.mark.parametrize("index", [r["index"] for r in FITTED_ROWS], ids=[r["job_id"] for r in FITTED_ROWS])
def test_the_two_doors_name_the_same_tuple_for_each_fitted_row(index, launcher, monkeypatch):
    """The driver door's shape (the resolved model key, the parsed controls, no reduction mode at the
    door and the served one after the record) and the launcher door's shape (the slug, the control
    tuple, the planned reduction mode and pass) return the row's tuple through the ONE checker; where
    the launcher's model registry reaches (qwen35_4b, gemma4e4b) its REAL planner is run and the
    checker it calls is spied — the other two models have no launcher door (``UnknownModel``: the
    cluster is their only entry; decisions-integration I6)."""
    row = M.M5T_CELLS[index]
    common = {"cell": row["cell"], "mode": row["thinking_mode"], "sampling_mode": row["sampling_mode"], "samples": M.SAMPLES}
    if row["tuple_name"] is None:
        # E2 / P7: a no-switch family's fitted row is IN SUPPORT — no tuple authorises it at either door, and the launcher
        # emits no --allow-out-of-support-thinking for it (test_sdfbase_math500_sbatch pins the argv)
        assert row["thinking_mode"] == M.THINKING_NOT_AVAILABLE and row["model_key"] in M.LLAMA_MODELS
        with pytest.raises(C.OutOfSupportThinkingUnauthorised):
            C.check_out_of_support_thinking_authorised(model=row["model_key"], controls=[M.CONTROL], where="driver run", **common)
        return
    # the driver's FIRST door STATES the served reduction mode read off the standup record beside --routing-table (this world's
    # record is a LENS one, `stacked`) — stated here too, because qwen35_4b's LoRA cells ALSO sit in the seed wave's native `_native`
    # tuple, which differs from the lens-era tuples in the served identity alone (reduction_mode not_applicable): UNSTATED, the pair
    # is ambiguous by construction (the GSM8K property; pinned in test_tinycap_contracts)
    driver_door = C.check_out_of_support_thinking_authorised(
        model=row["model_key"], controls=[M.CONTROL], reduction_mode=C.REDUCTION_MODE_STACKED, where="driver run", **common)
    driver_served = C.check_out_of_support_thinking_authorised(
        model=row["model_key"], controls=[M.CONTROL], reduction_mode=C.REDUCTION_MODE_STACKED, where="driver run",
        **common)
    launcher_shape = C.check_out_of_support_thinking_authorised(
        model=row["model_key"], controls=(M.CONTROL,), reduction_mode=C.REDUCTION_MODE_STACKED, pass_no=1,
        where="launch configuration", **common)
    assert driver_door == driver_served == launcher_shape == row["tuple_name"]
    # the sampling mode is what tells the tuples of one model apart: the neighbouring mode is ANOTHER tuple — for the
    # published object cell; the eight-arm extension's two LoRA cells have NO sensitivity rows (P4: one `_lora2`
    # tuple per model, the primary row only), so under another mode no tuple authorises them
    other = next(m for m in M.SAMPLING_MODES if m != row["sampling_mode"])
    if row["cell"] in M.STEER_CELLS:
        # the steering rung has no sensitivity rows either (one `_steer` tuple per switch model, the primary row only)
        with pytest.raises(C.OutOfSupportThinkingUnauthorised):
            C.check_out_of_support_thinking_authorised(
                model=row["model_key"], controls=[M.CONTROL], where="driver run", **{**common, "sampling_mode": other})
        with pytest.raises(SystemExit, match="thinking_card"):
            M.tuple_name(row["model_key"], other, row["cell"])
    elif row["cell"] in (M.CELL_LORA_MIX, M.CELL_LORA_CTL2):
        with pytest.raises(C.OutOfSupportThinkingUnauthorised):
            C.check_out_of_support_thinking_authorised(
                model=row["model_key"], controls=[M.CONTROL], where="driver run", **{**common, "sampling_mode": other})
        with pytest.raises(SystemExit, match="thinking_card"):
            M.tuple_name(row["model_key"], other, row["cell"])
    elif (row["model_key"], other) in M.SAMPLING_ROWS:
        # the served mode stated, as above: qwen's published object cell under `thinking_card` is ALSO in the seed wave's `_native` tuple
        assert C.check_out_of_support_thinking_authorised(
            model=row["model_key"], controls=[M.CONTROL], reduction_mode=C.REDUCTION_MODE_STACKED, where="driver run", **{**common, "sampling_mode": other}
        ) == M.tuple_name(row["model_key"], other) != row["tuple_name"]
    else:
        with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="sampling_mode differs"):
            C.check_out_of_support_thinking_authorised(
                model=row["model_key"], controls=[M.CONTROL], where="driver run", **{**common, "sampling_mode": other})
    calls = spy_checker(monkeypatch)
    try:
        cfgs = launcher.plan_configurations(
            cells=[row["cell"]], modes=[M.THINKING_MODE], controls=[M.CONTROL], model=row["model_key"],
            allow_out_of_support_thinking=True, sampling_mode=row["sampling_mode"], samples=M.SAMPLES,
            reduction_mode=C.REDUCTION_MODE_STACKED, pass_no=1)
    except C.UnknownModel as e:
        assert row["model_key"] in ("glm46vflash", "nemotron3nano4b") and row["model_key"] in str(e)
        assert calls == []
        return
    assert len(cfgs) == 1 and [name for _, name in calls] == [row["tuple_name"]]
    assert calls[0][0]["where"].startswith("launch configuration") and calls[0][0]["pass_no"] == 1
    assert calls[0][0]["reduction_mode"] == C.REDUCTION_MODE_STACKED
