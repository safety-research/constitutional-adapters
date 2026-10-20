"""Every subprocess the app launches runs under a venv that carries what the script imports.

The defect: the GSM8K assembly helper (`seedrep_gsm8k_assembly.py`) was still launched under the DRIVER venv, and it is
Inspect-dependent -- `inspect_rung.classify_store` -> `store_texts` -> `inspect_served.reusable_log` imports `inspect_ai.log` BEFORE any
escalation decision -> ModuleNotFoundError -> GSM8K_ASSEMBLY_REFUSED after the GPU standup. The lesson of the night: audit EVERY
app-side launch, not the one that bit. `APP_SUBPROCESS_TABLE` (seedrep_served.py) names each launch's script and interpreter; these
tests (a) pin the table to the code, (b) compute each launched script's import closure and require `client` wherever `inspect_served` /
`inspect_rung` is reachable unless the reach is provably unexecuted on that path (two `driver` exceptions, asserted by their guards),
and (c) check every entry's module-level third-party imports against the chosen venv's lock -- the exact driver set from
`uv export --frozen --extra cu128`, the client set from tinycap-client-requirements.txt -- with `inspect_ai` / `inspect_evals` as the
known-bad (absent from the driver set, present in the client set). Synthetic ids only.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _seedrep_testlib as T  # noqa: E402
from test_seedrep_w9_container_paths import _stub_modal  # noqa: E402

REPO = T.REPO
DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
_ABSENT = object()
sys.path.insert(0, str(REPO / "scripts" / "modal"))
import seedrep_launch_plan as lp  # noqa: E402


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_w12_interpreters", DRIVER)
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


# --- the import-closure analyser (follows `import a.b`, `from a import b` AND `from pkg import submodule`) ----------------------------

IMP = re.compile(r"^(\s*)(?:from\s+([\w\.]+)\s+import\s+([^\n#]+)|import\s+([\w\.]+))", re.M)
INSPECT_TOPS = ("inspect_ai", "inspect_evals")
HEAVY = ("tiers/inspect_served.py", "tiers/inspect_rung.py")


def _cands(mod: str, cur: Path):
    if mod.startswith("."):
        base, m = cur.parent, mod
        while m.startswith("."):
            if m[1:2] == ".":
                base = base.parent
            m = m[1:]
        return base.joinpath(*m.split(".")) if m else base
    if mod.startswith("sisteer"):
        return REPO / "src" / Path(*mod.split("."))
    top = mod.split(".")[0]
    for root in (cur.parent, REPO / "scripts"):               # first-party: a sibling of the launched script, or anything on PYTHONPATH=<tree>/scripts
        if (root / (top + ".py")).exists() or (root / top).is_dir():
            return root.joinpath(*mod.split("."))
    return None


_TREE_MODULES = None


def _first_party(top: str, cur: Path) -> bool:
    """A module the launched script reaches through the tree itself (PYTHONPATH=<tree>/src:<tree>/scripts plus the scripts' own sys.path
    inserts): any `<top>.py` or `<top>/` package anywhere under src/ or scripts/, or a sibling of the launched file."""
    global _TREE_MODULES
    if _TREE_MODULES is None:
        _TREE_MODULES = {p.stem for root in (REPO / "src", REPO / "scripts") for p in root.rglob("*.py")} | \
                        {p.name for root in (REPO / "src", REPO / "scripts") for p in root.rglob("*") if p.is_dir() and (p / "__init__.py").exists()}
    return top == "sisteer" or top in _TREE_MODULES or (cur.parent / (top + ".py")).exists()


def _resolve(path):
    if path is None:
        return None
    for p in (path.with_suffix(".py"), path / "__init__.py"):
        if p.exists():
            return p
    return None


def entry_path(script: str) -> Path:
    if script.startswith("-m "):
        return REPO / "src" / Path(*script[3:].split(".")).with_suffix(".py")
    return REPO / script


def closure(entry: Path):
    """(files in the closure, {file: kind} of inspect_* imports, top-level third-party import names of the ENTRY)."""
    seen, inspect_hits, order = set(), {}, [entry]
    entry_top_third_party = set()
    while order:
        f = order.pop(0)
        if f in seen:
            continue
        seen.add(f)
        for m in IMP.finditer(f.read_text(errors="replace")):
            indent, a, names, b = m.groups()
            mod = a or b
            top = mod.split(".")[0]
            if f == entry and not indent and not mod.startswith(".") and top not in sys.stdlib_module_names and not _first_party(top, f):
                entry_top_third_party.add(top)
            if top in INSPECT_TOPS:
                inspect_hits[str(f.relative_to(REPO))] = "lazy" if indent else "top"
                continue
            targets = [_resolve(_cands(mod, f))]
            if a and names:
                base = _cands(a, f)
                for n in names.replace("(", "").replace(")", "").split(","):
                    n = n.strip().split(" as ")[0].strip()
                    if n and base is not None:
                        targets.append(_resolve(base / n))
            order += [r for r in targets if r and r not in seen]
    return seen, inspect_hits, entry_top_third_party


def heavy_in(files) -> set:
    return {str(f.relative_to(REPO)) for f in files if str(f).endswith(HEAVY)}


# --- the delta itself: the assembly helper under the CLIENT venv ----------------------------------------------------------------------

class TestAssemblyHelperUnderTheClientVenv:
    ROW = {"standup_id": "gsm8k__s1__rung16384__shard00", "leg": "gsm8k", "engine": "s1", "stage": 16384,
           "rung_assembly": {"feeder_shards": ["00", "01"], "slots": [{"arm": "none", "s": 1, "cell": "none"}]}}
    CTX = {"results_root": "/vol/seedrep/2222222/exp/x", "W": "2" * 40, "wave_uuid": "22222222-3333-4444-8444-555555555555",
           "dicts": {"accepted": "acc", "results": "res"}}

    def test_the_assembly_argv_runs_under_the_client_python(self, srv, tmp_path):
        argv = srv.gsm8k_assembly_argv(self.ROW, self.CTX, tree=Path("/tree"), client_python="/home/seedrep/venvs/tinycap-client-abcd1234/bin/python",
                                       root=tmp_path, deadline_s=600)
        assert argv[:2] == ["/home/seedrep/venvs/tinycap-client-abcd1234/bin/python", "/tree/scripts/modal/seedrep_gsm8k_assembly.py"]
        assert "driver_python" not in inspect.signature(srv.gsm8k_assembly_argv).parameters

    def test_known_bad_without_a_client_python_it_refuses_by_name(self, srv, tmp_path):
        with pytest.raises(ValueError, match="assembly helper runs under the tinycap CLIENT venv .* no client python was built"):
            srv.gsm8k_assembly_argv(self.ROW, self.CTX, tree=Path("/tree"), client_python="", root=tmp_path, deadline_s=600)

    def test_run_standup_row_threads_the_client_python_to_the_assembly(self, srv):
        src = inspect.getsource(srv._run_standup_row)
        call = src[src.index("run_gsm8k_assembly(row, ctx"):src.index("run_gsm8k_assembly(row, ctx") + 200]
        assert "client_python=client_python" in call and "driver_python=" not in call
        assert src.index("if lp.row_needs_client_venv(row):") < src.index("run_gsm8k_assembly(row, ctx")
        # the helper's own contract says so
        head = (REPO / lp.GSM8K_ASSEMBLY_HELPER_REL).read_text()[:1200]
        assert "UNDER THE tinycap CLIENT VENV" in head and "UNDER THE DRIVER VENV" not in head

    def test_the_helper_is_inspect_dependent_which_is_why(self):
        files, hits, _ = closure(REPO / lp.GSM8K_ASSEMBLY_HELPER_REL)
        assert heavy_in(files) >= {"src/sisteer/capeval/tiers/inspect_rung.py", "src/sisteer/capeval/tiers/inspect_served.py"}
        assert "src/sisteer/capeval/tiers/inspect_served.py" in hits


# --- the audit: every app-side launch, by interpreter ---------------------------------------------------------------------------------

class TestEveryAppSideSubprocess:
    def test_the_table_names_every_launch_in_the_code(self, srv):
        table = {name: (script, kind) for name, script, kind in srv.APP_SUBPROCESS_TABLE}
        src = DRIVER.read_text()
        # every `[driver_python, ...]` / `[client_python, ...]` argv head in served.py is a table row with that interpreter
        for m in re.finditer(r"\[(driver_python|client_python), (?:str\(Path\(tree\) / |\")([^,\]]+)", src):
            var, target = m.group(1), m.group(2).strip().rstrip(")").strip('"').strip()
            kind = "driver" if var == "driver_python" else "client"
            rel = {"lp.GSM8K_ASSEMBLY_HELPER_REL": lp.GSM8K_ASSEMBLY_HELPER_REL, "DETERMINISM_REL": srv.DETERMINISM_REL}.get(target, target)
            if rel.startswith("-m"):
                rel = "-m " + src[m.end():m.end() + 80].split('"')[1] if '"-m"' in src[m.start():m.end() + 10] else rel
            rows = [(s, k) for s, k in table.values() if s == rel or rel.endswith(s) or s.endswith(rel.strip('"'))]
            assert rows, f"{var} launch of {target!r} has no APP_SUBPROCESS_TABLE row"
            assert all(k == kind for _, k in rows), f"{target}: table says {rows}, code launches under {kind}"
        # the plan composers: argv[0]'s placeholder matches the table
        comp = {"lp._static_children": lp._static_children, "lp._multiturn_children": lp._multiturn_children, "lp._bon_children": lp._bon_children,
                "lp._controls_children": getattr(lp, "_controls_children", None), "lp._gsm8k_children": lp._gsm8k_children,
                "lp._misalign_children": getattr(lp, "_misalign_children", None), "lp.rung_consumer_argv": lp.rung_consumer_argv}
        for name, fn in comp.items():
            if fn is None:
                continue
            s = inspect.getsource(fn)
            want = {"driver": "DRIVER_PY_PLACEHOLDER", "client": "CLIENT_PY_PLACEHOLDER"}[table[name][1]]
            assert f"[{want}," in s, f"{name}: argv[0] is not {want}"
        assert "serving_venv_python(" in inspect.getsource(lp.standup_argv) and table["lp.standup_argv"][1] == "serving"
        # the ledger and the helper subcommands
        assert "[driver_python, \"-m\", \"sisteer.capeval.serving.usage_ledger\"" in src and table["gsm8k_ledger_argv"][1] == "driver"

    def test_inspect_heavy_reach_implies_the_client_venv_or_a_proven_guard(self, srv):
        from sisteer.capeval.tinycap import contracts as tc
        for name, script, kind in srv.APP_SUBPROCESS_TABLE:
            files, hits, _ = closure(entry_path(script))
            heavy = heavy_in(files)
            entry_imports_inspect = str(entry_path(script).relative_to(REPO)) in hits
            if not heavy and not entry_imports_inspect:
                continue                                         # the universal lazy leaves (think_split / petri_runner) alone never decide
            if kind == "client":
                continue
            # a `driver` launch whose closure reaches inspect_served / inspect_rung must show the reach is unexecuted on ITS path
            if name == "lp._controls_children":
                # served_capability_run's only inspect_served imports sit behind `fam & set(tc.INSPECT_CONTROLS)` (the seeds block) and the
                # Inspect-atom derivation; the controls legs share no name with the Inspect controls, so neither executes
                assert not (set(lp.CONTROLS_ALL) & set(tc.INSPECT_CONTROLS)), (lp.CONTROLS_ALL, tc.INSPECT_CONTROLS)
                s = (REPO / "scripts/served_capability_run.py").read_text()
                assert "if fam & set(tc.INSPECT_CONTROLS):\n        from sisteer.capeval.tiers import inspect_served" in s
                continue
            if name == "lp.rung_consumer_argv":
                # rung_redraw's inspect_rung import is lazy inside its GSM8K seam; GSM8K rungs never run the consumer (the assembly path)
                s = (REPO / "scripts/sdfbase/seedrep/rung_redraw.py").read_text()
                assert re.search(r"^\s+from sisteer\.capeval\.tiers import inspect_rung", s, re.M) and not re.search(r"^from sisteer\.capeval\.tiers import inspect_rung", s, re.M)
                run_src = inspect.getsource(srv._run_standup_row)
                assert "elif is_gsm8k_rung:" in run_src and run_src.index("elif is_gsm8k_rung:") < run_src.index("elif is_rung:")
                assert "rung_consumer_child(row" in run_src[run_src.index("elif is_rung:"):]
                continue
            pytest.fail(f"{name} ({script}) is launched under the {kind} venv but reaches {sorted(heavy)}"
                        f"{' and imports inspect_* itself' if entry_imports_inspect else ''}: route it to the client python or prove the guard")

    def test_driver_and_client_locks_and_the_inspect_known_bad(self, srv):
        if shutil.which("uv") is None:
            pytest.skip("uv is not on PATH; the driver set is `uv export --frozen --extra cu128`")
        r = subprocess.run(["uv", "export", "--frozen", "--extra", "cu128", "--format", "requirements-txt", "--no-hashes", "--no-emit-project"],
                           cwd=REPO, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-500:]
        driver = {ln.split("==")[0].split("[")[0].strip().lower().replace("_", "-") for ln in r.stdout.splitlines() if "==" in ln and not ln.startswith(("#", "-"))}
        client = {ln.split("==")[0].split("[")[0].split(" @ ")[0].strip().lower().replace("_", "-")
                  for ln in (REPO / srv.CLIENT_LOCK_REL).read_text().splitlines() if re.match(r"^[A-Za-z0-9_.\-]+(==| @ )", ln)}
        # the known-bad: what the launch tree relied on before this change
        assert {"inspect-ai", "inspect-evals"} & driver == set() and {"inspect-ai", "inspect-evals"} <= client
        assert {"datasets", "pandas", "huggingface-hub"} <= driver
        alias = {"yaml": "pyyaml", "huggingface_hub": "huggingface-hub", "PIL": "pillow", "sklearn": "scikit-learn", "attr": "attrs",
                 "dotenv": "python-dotenv", "google": None, "cv2": "opencv-python", "OpenSSL": "pyopenssl"}
        for name, script, kind in srv.APP_SUBPROCESS_TABLE:
            if kind == "serving":
                continue
            lock = driver if kind == "driver" else client | driver     # the client venv carries the driver's deps too (f22_client_venv)
            _, _, tops = closure(entry_path(script))
            for top in sorted(tops):
                dist = alias.get(top, top.lower().replace("_", "-"))
                if dist is None:
                    continue
                assert dist in lock, f"{name} ({script}) imports {top!r} at module level; {dist!r} is not in the {kind} venv's lock"
