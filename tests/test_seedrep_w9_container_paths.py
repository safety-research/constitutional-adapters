"""Launch F-G / F-H / preflight check 8: the battery's in-container paths, exercised under the CONTAINER's conditions.

The seedrep ladder module is loaded by path INTO the Modal app process -- the image's system python, which has no torch
and, before this fix, no `sisteer` on `sys.path` (only the child processes got `PYTHONPATH=<tree>/src`). Every earlier test ran
under the repo venv, where both import, so the container's condition was never created: `redraw_messages` pulled
`sisteer.eval.jailbreak` (torch at module top) and `sampler_keys` pulled `sisteer.eval.attacks.bon_served` into an
interpreter that could import neither. These tests hide torch, purge the torch-carrying modules, and drive the producer
side the way the tailer does. Synthetic ids only.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seedrep_testlib as T  # noqa: E402

REPO = T.REPO
DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
_ABSENT = object()
#: the modules whose import chain reaches `import torch` -- purged so the lazy import in the ladder re-executes under the test's conditions
TORCH_CARRIERS = ("sisteer.eval.jailbreak", "sisteer.generate", "sisteer.methods", "sisteer.models", "sisteer.hooks", "sisteer.vectors",
                  "sisteer.eval.system_prefix", "sisteer.eval.attacks.bon_served")


@pytest.fixture
def no_torch():
    """`import torch` raises ImportError (sys.modules[name] = None is the interpreter's own 'this module is absent' marker)."""
    saved = {k: sys.modules.get(k, _ABSENT) for k in ("torch", *TORCH_CARRIERS)}
    for k in TORCH_CARRIERS:
        sys.modules.pop(k, None)
    sys.modules["torch"] = None
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is _ABSENT:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.fixture(scope="module")
def L():
    return T.load_ladder()


class TestFG_LadderProducerSideWithoutTorch:
    def test_torch_is_really_hidden(self, no_torch):
        with pytest.raises(ImportError):
            import torch  # noqa: F401

    def test_redraw_messages_for_a_prompted_arm_composes_without_torch(self, L, no_torch):
        # before the fix: `from sisteer.eval.jailbreak import install_authoritative_system` -> jailbreak's module top imports torch -> ImportError
        msgs = L.redraw_messages("the attack prompt", "the authoritative SI")
        assert msgs == [{"role": "system", "content": "the authoritative SI"}, {"role": "user", "content": "the attack prompt"}]

    def test_static_queue_record_for_a_prompted_arm_builds_without_torch(self, L, no_torch):
        spec = T.static_spec(L)
        ref = L.ProducerRef("0", L.SixAxisKey(T.W7, T.WAVE, "pasttense", "s1", "base", "0"))
        rec = L.static_queue_record(spec, arm="prompted__si_constitution", s=1, family="pasttense", row=T.static_row("g1", 0, seed=1),
                                    base_store="exp/x", producer=ref, producer_attempt_id="a1", vmodel="v", sampling=T.PRESET, system="SI")
        assert rec["redraw"]["messages"][0] == {"role": "system", "content": "SI"}
        assert set(rec["redraw"]["sampling"]) == set(T.PRESET)      # `check_redraw_sampling` ran (its `sampler_keys()` chain is torch-free)

    @staticmethod
    def _enqueue_prompted(L, tmp_path):
        """The tailer's exact motion for one at-cap prompted-arm row: build the record, `Producer.enqueue` it (which re-runs
        `check_redraw_sampling` -> `sampler_keys()` through `_validate_record`). COLD sampler cache: the lazy import executes."""
        L._SAMPLER_KEYS = None
        spec = T.static_spec(L)
        vol = L.SharedVolume().snapshot("a1")
        ref = next(pr for pr in spec.producers if pr.shard == "0")
        prod = L.Producer(vol, spec, ref, "a1", L.producer_attempt_root(ref, "a1"))
        rec = L.static_queue_record(spec, arm="prompted__si_constitution", s=1, family="pasttense", row=T.static_row("g1", 0, seed=1),
                                    base_store="exp/x", producer=ref, producer_attempt_id="a1", vmodel="v", sampling=T.PRESET, system="SI")
        return prod.enqueue(rec), rec

    def test_known_bad_the_app_process_layout_at_w8_cannot_enqueue_a_prompted_record(self, L, no_torch, tmp_path):
        """37762b566's app process: the ladder loaded by path, NO repo on sys.path (only the children got PYTHONPATH), no torch."""
        src = str(REPO / "src")
        saved = list(sys.path)
        saved_mods = {k: v for k, v in sys.modules.items() if k == "sisteer" or k.startswith("sisteer.")}
        try:
            sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(src).resolve()] if src else sys.path
            for k in saved_mods:
                sys.modules.pop(k, None)
            with pytest.raises((ModuleNotFoundError, ImportError)):
                self._enqueue_prompted(L, tmp_path)
        finally:
            sys.path[:] = saved
            sys.modules.update(saved_mods)
            L._SAMPLER_KEYS = None

    def test_the_w9_app_process_layout_enqueues_a_prompted_record_without_torch(self, L, srv, no_torch, tmp_path):
        """The app process: `ensure_tree_src_importable(tree)` ran (the staged tree's src at the head of sys.path), still no torch."""
        src = str(REPO / "src")
        saved = list(sys.path)
        saved_mods = {k: v for k, v in sys.modules.items() if k == "sisteer" or k.startswith("sisteer.")}
        try:
            sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(src).resolve()]
            for k in saved_mods:
                sys.modules.pop(k, None)
            assert srv.ensure_tree_src_importable(REPO) == src and sys.path[0] == src
            rel, rec = self._enqueue_prompted(L, tmp_path)
            assert rel.endswith("/a1.json") and rec["redraw"]["messages"][0] == {"role": "system", "content": "SI"}
            assert tuple(L.sampler_keys()) == tuple(T.PRESET) or set(L.sampler_keys()) == set(T.PRESET)
        finally:
            sys.path[:] = saved
            for k in [k for k in sys.modules if k == "sisteer" or k.startswith("sisteer.")]:
                sys.modules.pop(k, None)
            sys.modules.update(saved_mods)
            L._SAMPLER_KEYS = None

    def test_the_composer_is_a_pure_move_and_jailbreak_still_names_it(self):
        pytest.importorskip("torch")
        from sisteer.eval import jailbreak, system_prefix
        # the property: jailbreak's name IS system_prefix's function (same defining module and code); object identity can churn when
        # another test re-imports `sisteer.eval.*` under a purged sys.modules, which is not what this asserts
        assert jailbreak.install_authoritative_system.__module__ == "sisteer.eval.system_prefix"
        assert jailbreak.install_authoritative_system.__code__.co_code == system_prefix.install_authoritative_system.__code__.co_code
        out = system_prefix.install_authoritative_system([{"role": "system", "content": "carried"}, {"role": "user", "content": "u"}], "SI")
        assert out == [{"role": "system", "content": "SI"}, {"role": "user", "content": "carried\n\nu"}]

    def test_the_composer_module_imports_nothing_but_the_stdlib(self):
        src = (REPO / "src/sisteer/eval/system_prefix.py").read_text()
        imports = [ln.strip() for ln in src.splitlines() if ln.startswith(("import ", "from "))]
        assert imports == ["from __future__ import annotations"], imports


# --- the served app's side: the tree's src reaches the APP interpreter before the ladder loads ------------------------------

class _Chain:
    def __getattr__(self, _name):
        return lambda *a, **k: self

    def __call__(self, *a, **k):
        return self


class _App:
    def __init__(self, name=None, **k):
        self.description = None

    def function(self, **kw):
        return lambda f: f

    def local_entrypoint(self, **kw):
        return lambda f: f


def _stub_modal():
    m = types.ModuleType("modal")
    m.App = _App
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(commit=lambda: None, reload=lambda: None))
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(), debian_slim=lambda *a, **k: _Chain())
    m.Dict = types.SimpleNamespace(from_name=lambda *a, **k: {})
    m.is_local = lambda: False
    m.current_function_call_id = lambda: "fc-test"
    return m


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_w9_container_paths", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal", _ABSENT)
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is _ABSENT:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


class TestFG_AppProcessSysPath:
    def test_ensure_tree_src_importable_puts_the_tree_src_first_and_is_idempotent(self, srv, tmp_path):
        tree = tmp_path / "sisteer-w"
        (tree / "src").mkdir(parents=True)
        saved = list(sys.path)
        try:
            got = srv.ensure_tree_src_importable(tree)
            assert got == str(tree / "src") and sys.path[0] == got
            sys.path.insert(0, "/elsewhere")
            srv.ensure_tree_src_importable(tree)
            assert sys.path[0] == got and sys.path.count(got) == 1
        finally:
            sys.path[:] = saved

    def test_run_standup_row_inserts_the_src_before_the_ladder_loads(self, srv):
        src = inspect.getsource(srv._run_standup_row)
        i, j = src.index("ensure_tree_src_importable(tree)"), src.index("L = load_ladder_lib(tree)")
        assert i < j, "the ladder must load AFTER the staged tree's src is on the app process's sys.path"
        k = src.index("StreamTailer(")
        assert i < k



# --- F-H: the frozen replay corpora reach the drivers; a base producer whose children wrote nothing is refused ------------

class TestFH_HeldoutCorpusReachesTheDrivers:
    def test_container_env_points_the_drivers_at_the_symlinked_corpora(self, srv, monkeypatch):
        monkeypatch.delenv("SISTEER_HELDOUT_CORPUS_DIR", raising=False)
        env = srv.container_env(bank_view="/b", tree="/t", thinking=False, base={})
        assert env["SISTEER_HELDOUT_CORPUS_DIR"] == f"{srv.SHARED_LOCAL}/exp/heldout_corpora" == srv.HELDOUT_CORPUS_LOCAL
        # the very path `_build_shared` symlinks from the volume's corpora
        src = inspect.getsource(srv._build_shared)
        assert 'os.symlink(CORPORA_VOL, exp / "heldout_corpora"' in src and "if Path(CORPORA_VOL).is_dir():\n        os.symlink" not in src

    def test_renellm_reads_its_corpus_under_that_directory(self, srv, monkeypatch, tmp_path):
        pytest.importorskip("torch")
        env = srv.container_env(bank_view="/b", tree="/t", thinking=False, base={})
        monkeypatch.setenv("SISTEER_HELDOUT_CORPUS_DIR", env["SISTEER_HELDOUT_CORPUS_DIR"])
        monkeypatch.setenv("HOME", str(tmp_path))                     # the container's HOME=/root shape: nothing under ~/.local/state
        import importlib
        ren = importlib.import_module("sisteer.eval.attacks.renellm")
        assert ren._corpus_path() == Path(srv.HELDOUT_CORPUS_LOCAL) / "renellm.jsonl"
        monkeypatch.delenv("SISTEER_HELDOUT_CORPUS_DIR")
        empty = tmp_path / "site.env"                                 # an isolated empty site: the neutral default
        empty.write_text("")
        monkeypatch.setenv("SISTEER_CONFIG", str(empty))
        for k in ("SISTEER_JAILBREAK_PRIVATE_ROOT", "SISTEER_STATE_ROOT"):
            monkeypatch.delenv(k, raising=False)
        assert ren._corpus_path() == tmp_path / ".local/state/sisteer-jailbreak/heldout_corpora/renellm.jsonl"   # the earlier silent fall-through

    def test_build_shared_refuses_without_the_corpora_on_the_volume(self, srv, monkeypatch, tmp_path):
        monkeypatch.setattr(srv, "CORPORA_VOL", str(tmp_path / "absent-heldout_corpora"))
        monkeypatch.setattr(srv, "SHARED_LOCAL", str(tmp_path / "shared"))
        monkeypatch.setattr(srv, "REPOS", str(tmp_path / "repos-never-read"))
        with pytest.raises(RuntimeError, match="heldout_corpora is absent on the volume"):
            srv._build_shared("2" * 40)
        assert not (tmp_path / "shared").exists() and not (tmp_path / "repos-never-read").exists()


class TestFH_EmptyInventoryOverEmptyStreamsIsRefused:
    LADDER = {"kind": "static", "cap_next": 8192, "shards": ["0"], "feeder_shards": ["0"], "consumes": 4096, "shard": "0"}

    def _write(self, srv, root, *, entries, streams):
        return srv.write_ladder_enqueued(root, attempt_id="a1", control_key="k", accepted_key="a", ladder=self.LADDER, entries=entries,
                                         streams=streams, complete=True)

    def test_known_bad_an_empty_complete_inventory_over_no_streams_is_refused_by_name(self, srv, tmp_path):
        self._write(srv, tmp_path, entries=[], streams=[])
        with pytest.raises(srv.LadderInventoryIncomplete, match="carried no rows"):
            srv.require_ladder_inventory(tmp_path, require_rows=True)
        # streams present but every one read zero rows: the children ran and wrote nothing -> the same refusal
        self._write(srv, tmp_path, entries=[], streams=[{"path": "exp/x/responses.jsonl", "rows_read": 0, "at_cap": 0, "enqueued": 0}])
        with pytest.raises(srv.LadderInventoryIncomplete, match="carried no rows"):
            srv.require_ladder_inventory(tmp_path, require_rows=True)

    def test_an_empty_inventory_over_streams_that_carried_rows_stands(self, srv, tmp_path):
        self._write(srv, tmp_path, entries=[], streams=[{"path": "exp/x/responses.jsonl", "rows_read": 37, "at_cap": 0, "enqueued": 0}])
        assert srv.require_ladder_inventory(tmp_path, require_rows=True)["n"] == 0    # nothing hit the cap: a legitimate empty rung feed

    def test_the_default_path_is_unchanged_for_rung_rows(self, srv, tmp_path):
        self._write(srv, tmp_path, entries=[], streams=[])
        assert srv.require_ladder_inventory(tmp_path)["n"] == 0
        assert "require_rows" in inspect.signature(srv.finish_attempt).parameters

    def test_run_standup_row_requires_rows_for_a_base_producer_only(self, srv):
        src = inspect.getsource(srv._run_standup_row)
        assert "require_rows=bool(producer_row and not is_rung)" in src


# --- preflight check 8: what the plan's legs read offline is on the volume, at its pin ---------------------------------------

def _plan_with_legs(*legs):
    return {"standups": [{"leg": leg} for leg in legs], "s5_probe_rows": []}


def _payload_name(pattern, suffix):
    """One file name a pins.yaml glob pattern admits (the reader's own rule: suffix by verify format, then fnmatch on the pattern)."""
    name = pattern.replace("*", "00000-of-00001")
    return name if name.endswith(suffix) else name + suffix


def _fake_volume(srv, tmp_path, *, jbb_file=True, drop_pin=None, corpora=True, datasets_cache=True):
    """A /vol shaped like the container's: hub snapshots for every pin this tree names, holding the PAYLOAD files the readers select
    (HarmBench's exact test parquet; each pins.yaml entry's pattern / verify-format files -- a README alone no longer passes),
    the held-out corpora, and the PREPARED `datasets` cache dirs the offline load_dataset readers look for."""
    import yaml
    from sisteer.eval import jailbreak_data as jd
    from sisteer.capeval import corpora as CP
    hf = tmp_path / "vol" / "hf"
    for repo, rev, files in ((jd.HARMBENCH_REPO, jd.HARMBENCH_REVISION, [jd.HARMBENCH_FILES["test"], "README.md"]),
                             (jd.JBB_REPO, jd.JBB_REVISION, [jd.JBB_FILES["harmful"], "README.md"] if jbb_file else ["README.md"])):
        d = srv.dataset_snapshot_dir(str(hf), repo, rev)
        for f in files:
            (d / f).parent.mkdir(parents=True, exist_ok=True)
            (d / f).write_text("x")
    pins = yaml.safe_load((REPO / srv.CORPORA_PINS_REL).read_text())["datasets"]
    for name in (*srv.CONTROLS_PIN_NAMES, *srv.GSM8K_PIN_NAMES):
        if name == drop_pin:
            continue
        e = pins[name]
        d = srv.dataset_snapshot_dir(str(hf), e["repo_id"], e["revision"])
        suffix = CP._VERIFY_SUFFIXES[e["verify"]][0]
        for rel in ([_payload_name(pat, suffix) for pat in e["patterns"]] if e.get("patterns") else [f"data/part{suffix}"]) + ["README.md", ".gitattributes"]:
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text("x")
    corp = tmp_path / "vol" / "heldout_corpora"
    if corpora:
        corp.mkdir(parents=True)
        for f in srv.HELDOUT_CORPUS_FILES:
            (corp / f).write_text("{}")
    dcache = hf / "datasets"
    if datasets_cache:
        for d, cfg in srv.datasets_cache_dirs_needed(REPO, {"datasets": ["gsm8k", srv.MMLU_DATASETS_DIR[0]]}).items():
            (dcache / d / cfg).mkdir(parents=True, exist_ok=True)
    return str(hf), str(corp)


class TestPreflightCheck8_CorporaPinnedSnapshotsPresent:
    ALL = ("pasttense", "codeattack", "renellm", "grok420", "bon", "pair", "crescendo", "actorattack", "controls", "misalign", "gsm8k", "petri")

    def test_needs_are_derived_from_the_plan_legs(self, srv):
        assert srv.corpora_needed_for_plan(_plan_with_legs("misalign", "petri")) == {"legs": ["misalign", "petri"], "goal_sets": False, "heldout": [], "pins": [],
                                                                                        "datasets": []}
        n = srv.corpora_needed_for_plan(_plan_with_legs(*self.ALL))
        assert n["goal_sets"] and n["heldout"] == list(srv.HELDOUT_CORPUS_FILES) and n["pins"] == list(srv.CONTROLS_PIN_NAMES) + list(srv.GSM8K_PIN_NAMES)
        assert n["datasets"] == [srv.MMLU_DATASETS_DIR[0], "gsm8k"]                    # the prepared `datasets` caches the readers need
        assert srv.corpora_needed_for_plan(_plan_with_legs("pasttense"))["heldout"] == []       # only renellm replays a frozen corpus

    def test_a_volume_carrying_every_pin_passes_and_names_what_it_found(self, srv, tmp_path):
        hf, corp = _fake_volume(srv, tmp_path)
        out = srv.preflight_corpora_check(REPO, _plan_with_legs(*self.ALL), hf_vol=hf, corpora_vol=corp, datasets_cache=f"{hf}/datasets")
        assert {"harmbench", "jbb", "xstest_v2", "or_bench", "mmlu", "gsm8k_platinum", "heldout_corpora", "datasets_cache"} <= set(out)
        assert out["harmbench"]["file"].endswith("DirectRequest/test-00000-of-00001.parquet") and out["or_bench"]["files"] == ["or-bench-hard-1k.csv"]
        assert out["heldout_corpora"] == sorted(srv.HELDOUT_CORPUS_FILES)

    def test_each_absence_fails_by_name(self, srv, tmp_path):
        hf, corp = _fake_volume(srv, tmp_path / "a", jbb_file=False)
        with pytest.raises(AssertionError, match="jbb: .*harmful-behaviors.csv missing"):
            srv.preflight_corpora_check(REPO, _plan_with_legs("pasttense"), hf_vol=hf, corpora_vol=corp)
        hf, corp = _fake_volume(srv, tmp_path / "b", drop_pin="or_bench")
        with pytest.raises(AssertionError, match="or_bench: no snapshot"):
            srv.preflight_corpora_check(REPO, _plan_with_legs("controls"), hf_vol=hf, corpora_vol=corp, datasets_cache=f"{hf}/datasets")
        hf, corp = _fake_volume(srv, tmp_path / "c", corpora=False)
        with pytest.raises(AssertionError, match="heldout corpus renellm.jsonl missing"):
            srv.preflight_corpora_check(REPO, _plan_with_legs("renellm"), hf_vol=hf, corpora_vol=corp)
        # the same empty volume passes for legs that read nothing off it
        assert srv.preflight_corpora_check(REPO, _plan_with_legs("misalign", "petri"), hf_vol=hf, corpora_vol=corp) == {"legs": ["misalign", "petri"]}

    def test_the_check_is_registered_in_preflight_and_the_helper_stays_plain(self, srv):
        src = inspect.getsource(srv.preflight)
        assert 'chk("corpora_pinned_snapshots_present", corpora)' in src and "preflight_corpora_check(tree, plan)" in src
        assert src.index('chk("launch_plan_validates_against_table"') < src.index('chk("corpora_pinned_snapshots_present"')
