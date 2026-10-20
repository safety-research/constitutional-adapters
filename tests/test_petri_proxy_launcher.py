"""The production Petri proxy LAUNCHER's own construction path
(``scripts/sbatch/petri_proxy.sbatch``) — EXECUTED, not reconstructed —
the proxy CLI's defaults, and the source-level classification of every
``RoutingProxy`` construction site.

Why this exists (review 2026-09-02): the declared agent-loop path landed
correct at the adapter, the selector, the verifier and 71 tests, and
DISABLED at the one entry point real audits use — the sbatch launcher
built the proxy with neither a render probe nor the Petri-facing entry, so
a lens cell's first tool round-trip would have refused (contained, but
unservable). Two things fixed it and this module keeps both true:

* the proxy's DEFAULT flipped: the entry is armed unless a site states
  ``petri_agent_loop_entry=False``, and the render probe loads lazily from
  the routing table's ``base_model`` (+ ``base_revision``) — so every
  construction site is correct with no arguments, and the launcher bug
  became impossible rather than tested-for;
* the launcher nonetheless STATES its configuration — the manifest-pinned
  served tokenizer under the pinned chat template, eager, entry on —
  and ``test_launcher_builds_the_proxy_with_the_agent_loop_path_enabled``
  runs the sbatch's EMBEDDED PYTHON (extracted from the heredoc) with the
  GPU seams stubbed (server launch, aiohttp ``run_app``, the tokenizer
  load) and asserts the RoutingProxy it constructs. A future edit to the
  launcher that drops either argument goes red here, against the
  launcher's own code — a test that re-implemented its arguments would
  test the test.

``test_every_proxy_construction_site_is_classified`` enumerates every
``RoutingProxy(`` construction site under ``scripts/`` and ``src/`` and
requires each to be in :data:`SITES` with a one-line reason; a new site
fails until someone classifies it. Under the default-on entry the
classification is a RECORD, not a switch — a site that wants the entry off
must pass ``petri_agent_loop_entry=False`` AND be listed in
:data:`OPTED_OUT` with its reason (none today). All offline and synthetic:
the routing table comes from the proxy tests' canonical cell set; the
manifest is the repo's own.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402

from test_petri_proxy import _cells, _source, _spec  # noqa: E402

SBATCH = REPO / "scripts/sbatch/petri_proxy.sbatch"
MANIFEST = REPO / "configs/capeval/manifests/petri_twosi.yaml"


# --------------------------------------------------------------------------- #
# the launcher, executed                                                       #
# --------------------------------------------------------------------------- #


def _embedded_python(text: str) -> str:
    m = re.search(r"<<'PYEOF'\n(.*?)\nPYEOF\n", text, re.S)
    assert m, ("the launcher's embedded python heredoc (<<'PYEOF' ... PYEOF) "
               "was not found — if the launcher moved its python elsewhere, "
               "point this test at it; the launcher's OWN construction path "
               "must stay under test")
    return m.group(1)


class _StubRender:
    """Stands in for TokenizerAgentLoopRender: records the tokenizer it
    was built from (the real class derives the header from it, which needs
    a real tokenizer — the stub keeps the test offline)."""

    header_ids = (901, 902, 903)

    def __init__(self, tok):
        self.tok = tok

    def render(self, messages, tools):  # pragma: no cover — never called
        raise AssertionError("the launcher test issues no requests")


def _routing_table(tmp_path, spec):
    """A routing table in the manifest's serving mode (the launcher refuses
    a mode mismatch), from the proxy tests' canonical synthetic cells."""
    syn = _spec(serving_mode=spec.serving_mode,
                target_model=spec.target_model,
                target_revision=spec.target_revision)
    source = _source(tmp_path)
    cells = _cells(syn, source)
    path = tmp_path / "routing_table.json"
    px.export_routing_table(
        cells, source, path, base_model=spec.target_model,
        serving_mode=spec.serving_mode,
        exact_replay_strategy=("serialize" if spec.serving_mode
                               == "exact_replay" else None),
        base_revision=spec.target_revision)
    return path


def _run_launcher(monkeypatch, tmp_path, *, did=None, tok_error=None):
    """Execute the sbatch's embedded python with every GPU/network seam
    stubbed; return what it did (``did`` may be passed in so a caller that
    expects the launcher to RAISE can still read the record; ``tok_error``
    makes the tokenizer load raise it)."""
    spec = cm.load_petri_spec(MANIFEST)
    table = _routing_table(tmp_path, spec)
    registry_dir = tmp_path / "serving" / "lens_registry"
    registry_dir.mkdir(parents=True)
    template = tmp_path / "chat_template.jinja"
    template.write_text("{{ PINNED-TEMPLATE-BYTES }}", encoding="utf-8")
    if did is None:
        did = {}
    did.update({"launch_kwargs": None, "server": None, "tok_calls": [],
                "proxies": [], "run_app": None, "terminated": False,
                "order": []})

    fake_launch = SimpleNamespace(base_url="http://127.0.0.1:8801",
                                  api_key="upstream-token",
                                  env={"VLLM_ATTEST": "1"})

    def build_server_launch(spec_, **kw):
        did["launch_kwargs"] = kw
        return fake_launch

    class _Server:
        def terminate(self):
            did["terminated"] = True

        def wait(self, timeout=None):
            return 0

    def launch_server(launch, *, log_path):
        did["server"] = str(log_path)
        did["order"].append("server")
        return _Server()

    def load_served_tokenizer(name, *, revision=None, chat_template=None,
                              **kw):
        did["tok_calls"].append((name, revision, chat_template))
        did["order"].append("tokenizer")
        if tok_error is not None:
            raise tok_error
        return SimpleNamespace(name_or_path=name, revision=revision)

    real_proxy = px.RoutingProxy

    class _Recording(real_proxy):
        def __init__(self, *a, **kw):
            did["proxies"].append(kw)
            super().__init__(*a, **kw)

    import aiohttp.web as web

    def run_app(app, *, host, port):
        did["run_app"] = {"host": host, "port": port, "app": app}

    monkeypatch.setattr(vs, "build_server_launch", build_server_launch)
    monkeypatch.setattr(vs, "launch_server", launch_server)
    monkeypatch.setattr(px, "load_served_tokenizer", load_served_tokenizer)
    monkeypatch.setattr(px, "TokenizerAgentLoopRender", _StubRender)
    monkeypatch.setattr(px, "RoutingProxy", _Recording)
    monkeypatch.setattr(web, "run_app", run_app)
    monkeypatch.chdir(REPO)  # the launcher loads the manifest relatively
    monkeypatch.setattr(sys, "argv", [
        "-", str(table), str(registry_dir), "8801", "8814", str(template)])
    code = _embedded_python(SBATCH.read_text(encoding="utf-8"))
    exec(compile(code, str(SBATCH), "exec"), {"__name__": "__main__"})
    did["spec"] = spec
    return did


def test_launcher_builds_the_proxy_with_the_agent_loop_path_enabled(
        monkeypatch, tmp_path):
    """The production launcher's OWN code constructs the proxy with the
    Petri-facing entry armed and an EXPLICIT render probe built from the
    manifest-pinned served tokenizer under the pinned chat template —
    loaded before the server (eager), and advertised on /v1/models."""
    did = _run_launcher(monkeypatch, tmp_path)
    spec = did["spec"]
    assert len(did["proxies"]) == 1
    kw = did["proxies"][0]
    assert kw["petri_agent_loop_entry"] is True
    render = kw["agent_loop_render"]
    assert isinstance(render, _StubRender)
    # ... built from the manifest pin, under the pinned template bytes
    assert did["tok_calls"] == [(spec.target_model, spec.target_revision,
                                 "{{ PINNED-TEMPLATE-BYTES }}")]
    assert render.tok.revision == spec.target_revision
    # what the audit's client would see on /v1/models
    app_proxy = did["run_app"]["app"]
    assert app_proxy is not None
    assert did["run_app"]["host"] == "127.0.0.1"
    assert did["run_app"]["port"] == 8814
    # the AGENT modality launch (tool calls) and the attestation are intact
    assert did["launch_kwargs"]["tool_calls"] is True
    assert kw["launch_attestation"] == {"VLLM_ATTEST": "1"}
    assert did["terminated"] is True  # the finally block still tears down


def test_launcher_proxy_advertises_the_armed_entry(monkeypatch, tmp_path):
    """The identity block a client (petri_run's ProxyClientEngine.load)
    records: entry on, explicit probe. Asserted on the proxy the launcher
    actually built, via a recording make_app."""
    built = {}
    real_make_app = px.make_app

    def make_app(proxy):
        built["proxy"] = proxy
        return real_make_app(proxy)

    monkeypatch.setattr(px, "make_app", make_app)
    _run_launcher(monkeypatch, tmp_path)
    meta = built["proxy"].models_payload()["petri_proxy"]
    assert meta["petri_agent_loop_entry"] is True
    assert meta["agent_loop_render"]["mode"] == "explicit"
    assert meta["agent_loop_render"]["loaded"] is True


def test_launcher_loads_the_probe_before_the_server(monkeypatch, tmp_path):
    """Eager, on purpose: a bad tokenizer pin must fail at startup, before
    a GPU server is launched — never at the audit's first tool turn."""
    did: dict = {}
    with pytest.raises(OSError, match="no such revision"):
        _run_launcher(monkeypatch, tmp_path, did=did,
                      tok_error=OSError("no such revision"))
    assert did["order"] == ["tokenizer"]  # the server was never launched
    assert did["proxies"] == []


# --------------------------------------------------------------------------- #
# the CLI's defaults                                                           #
# --------------------------------------------------------------------------- #


def _run_cli(monkeypatch, tmp_path, extra=()):
    spec = cm.load_petri_spec(MANIFEST)
    table = _routing_table(tmp_path, spec)
    built = {}
    monkeypatch.setattr(px, "make_app", lambda proxy: built.setdefault(
        "proxy", proxy))
    import aiohttp.web as web
    monkeypatch.setattr(web, "run_app", lambda app, *, host, port: None)
    px.main(["--routing-table", str(table), "--upstream",
             "http://127.0.0.1:1", "--port", "8814", *extra])
    return built["proxy"], spec


def test_cli_entry_defaults_on_with_a_lazy_pinned_probe(monkeypatch,
                                                        tmp_path):
    proxy, spec = _run_cli(monkeypatch, tmp_path)
    meta = proxy.models_payload()["petri_proxy"]
    assert meta["petri_agent_loop_entry"] is True
    assert meta["agent_loop_render"] == {
        "mode": "lazy", "loaded": False,
        "source": f"{spec.target_model}@{spec.target_revision}"}


def test_cli_opt_out_is_explicit(monkeypatch, tmp_path):
    proxy, _ = _run_cli(monkeypatch, tmp_path, ["--no-petri-agent-loop"])
    assert proxy.models_payload()["petri_proxy"][
        "petri_agent_loop_entry"] is False


def test_cli_tokenizer_builds_an_explicit_probe_under_the_pinned_template(
        monkeypatch, tmp_path):
    calls = []

    def load_served_tokenizer(name, *, revision=None, chat_template=None,
                              **kw):
        calls.append((name, revision, chat_template))
        return object()

    monkeypatch.setattr(px, "load_served_tokenizer", load_served_tokenizer)
    monkeypatch.setattr(px, "TokenizerAgentLoopRender", _StubRender)
    tmpl = tmp_path / "t.jinja"
    tmpl.write_text("T", encoding="utf-8")
    proxy, _ = _run_cli(monkeypatch, tmp_path, [
        "--tokenizer", "org/model", "--tokenizer-revision", "abc",
        "--chat-template", str(tmpl)])
    assert calls == [("org/model", "abc", "T")]
    assert proxy.models_payload()["petri_proxy"]["agent_loop_render"][
        "mode"] == "explicit"


# --------------------------------------------------------------------------- #
# every construction site, classified                                          #
# --------------------------------------------------------------------------- #

#: every RoutingProxy construction site under scripts/ and src/, with why
#: its agent-loop configuration is right. The entry is ON everywhere by
#: default; "inert" means the driver's traffic is single-turn (no tool
#: messages — verified by grep, 2026-09-02), so the entry never fires and
#: the lazy probe never loads.
SITES = {
    "scripts/sdfbase/mt_serving_up.py":
        "the sdfbase-mt wave's multi-cell standup (one model's every wave "
        "cell + scale-0 twins): entry ON by default with an EXPLICIT "
        "TokenizerAgentLoopRender probe over the served tokenizer under the "
        "manifest-pinned template (re-hashed before launch); --mode petri "
        "traffic DOES carry agent-loop shapes (the audit's tool-use legs), "
        "--mode attacks traffic never does (chats end at the generation "
        "header) and the entry is inert there; --no-petri-agent-loop turns "
        "it off deliberately",
    "src/sisteer/capeval/serving/native_standup.py":
        "the shared NATIVE serving core (native-serving-default plan §1.2) — "
        "the ONE RoutingProxy construction behind both native fronts, "
        "lora_serving_up.bring_up_lora (the wave; mt_serving_up "
        "--serving-path lora_native) and capctrl_serving_up.bring_up("
        "serving_path=lora_native), neither of which constructs a proxy "
        "itself any more: entry ON by default (petri_agent_loop=True) with "
        "an EXPLICIT TokenizerAgentLoopRender probe over the served tokenizer "
        "under the manifest-pinned template; tool_calls is EXPLICIT per front "
        "(the wave passes eval petri, capctrl its --tool-calls), so petri "
        "traffic carries agent-loop shapes and the static/attacks/capability "
        "evals are single-turn with the entry inert; --no-petri-agent-loop "
        "turns it off deliberately",
    "scripts/sbatch/petri_proxy.sbatch":
        "THE production Petri audit proxy: entry stated ON, EXPLICIT "
        "manifest-pinned probe under the pinned template (tested above)",
    "src/sisteer/capeval/serving/proxy.py":
        "the proxy's own CLI main(): --petri-agent-loop defaults ON, "
        "--tokenizer/--chat-template for an eager probe (tested above)",
    "scripts/capctrl_serving_up.py":
        "served-capability + agent-loop-control standup: explicit probe "
        "from its own tokenizer, entry ON by default (--no-petri-agent-loop "
        "opts out), --tool-calls for the agent modality",
    "scripts/lens_gate_standup.py":
        "the lens-vectorize gate's tree-local standup helper (the body of "
        "capctrl_serving_up.main, run once from the base-commit checkout and "
        "once from the candidate): explicit probe from its own tokenizer, "
        "entry ON by default; the gate's traffic is single-turn; inert",
    "scripts/petri_serving_gate.py":
        "the section-10 serving gate: single-turn fp32-reference probes; "
        "entry ON by default, inert (the reference shim refuses tools, so "
        "agent-loop fidelity is the Modal control's, not this gate's)",
    "scripts/petri_cache_isolation_gate.py":
        "salt-isolation gate: single-turn gate prompts; inert",
    "scripts/petri_families_prompted.py":
        "jailbreak families over the prompted cells: single-turn; inert",
    "scripts/petri_families_served_integration.py":
        "fully-served families integration: single-turn; inert",
    "scripts/petri_bon_prompted.py": "BoN over prompted cells: single-turn; inert",
    "scripts/petri_bon_prompted_remeasure.py":
        "BoN prompted re-measure: single-turn; inert",
    "scripts/petri_bon_fanout.py": "BoN fan-out: single-turn; inert",
    "scripts/petri_bon_pathA.py": "BoN path-A driver: single-turn; inert",
    "scripts/petri_bon_speedup.py": "BoN serving speed-up bench: single-turn; inert",
    "scripts/petri_bon_smoke_integration.py":
        "BoN served smoke: single-turn; inert",
}

#: sites that pass petri_agent_loop_entry=False, each with its reason. A
#: stated opt-out is legitimate; an unlisted one fails below.
OPTED_OUT: dict = {}

_SITE_RE = re.compile(r"\bRoutingProxy\(")
#: an opt-out is a CALL argument; a backticked docstring mention of the
#: spelling (``petri_agent_loop_entry=False``) is documentation, not one
_OPT_OUT_RE = re.compile(r"(?<!`)petri_agent_loop_entry\s*=\s*False\b(?!`)")


def _construction_sites():
    files = (list((REPO / "scripts").rglob("*.py"))
             + list((REPO / "scripts").rglob("*.sbatch"))
             + list((REPO / "scripts").rglob("*.sh"))
             + list((REPO / "src").rglob("*.py")))
    found, opted = set(), set()
    for f in sorted(files):
        code_lines = [ln for ln in f.read_text(encoding="utf-8",
                                               errors="replace").splitlines()
                      if not ln.lstrip().startswith("#")]
        src = "\n".join(code_lines)
        if _SITE_RE.search(src.replace("class RoutingProxy(", "")):
            found.add(str(f.relative_to(REPO)))
        if _OPT_OUT_RE.search(src):
            opted.add(str(f.relative_to(REPO)))
    return found, opted


def test_every_proxy_construction_site_is_classified():
    found, opted = _construction_sites()
    unclassified = sorted(found - set(SITES))
    assert not unclassified, (
        f"RoutingProxy construction site(s) not classified: {unclassified} "
        "— add each to tests/test_petri_proxy_launcher.py::SITES with a "
        "one-line reason its agent-loop configuration is right (the entry "
        "is ON by default; say whether the site's traffic can carry an "
        "agent-loop shape, and pass an explicit probe if it can)")
    stale = sorted(set(SITES) - found)
    assert not stale, (
        f"SITES lists construction site(s) that no longer exist: {stale} "
        "— drop them so the classification stays a record of the code")
    unlisted_opt_outs = sorted(opted - set(OPTED_OUT))
    assert not unlisted_opt_outs, (
        f"site(s) pass petri_agent_loop_entry=False without a stated "
        f"reason: {unlisted_opt_outs} — add them to OPTED_OUT with why the "
        "Petri-facing entry must be off there")
    assert not sorted(set(OPTED_OUT) - opted), "OPTED_OUT lists a site that no longer opts out"


# --------------------------------------------------------------------------- #
# the client-side belt: the audit runner refuses a proxy that cannot serve it  #
# --------------------------------------------------------------------------- #


def _cell(serving_identity, cell_id="si_safety__psr_kl_all"):
    return SimpleNamespace(cell_id=cell_id, serving_identity=serving_identity)


def test_runner_refuses_a_lens_cell_through_an_opted_out_proxy():
    """The grading-host-side runner reads the proxy's identity block (the same
    live source eval_concurrency reads) and refuses to audit a LENS cell
    through a proxy that does not declare agent-loop turns — an opted-out
    launch or a pre-landing build — before the first scenario, instead of
    after a ledger full of contained 500s."""
    from sisteer.capeval.tiers import petri_runner as pr
    lens = _cell("proxy_vllm+lens+interleaved")
    armed = SimpleNamespace(proxy_identity={"version": 1,
                                           "petri_agent_loop_entry": True})
    pr.require_agent_loop_entry(armed, lens)  # passes silently
    for ident in ({"version": 1, "petri_agent_loop_entry": False},
                  {"version": 1}):  # an older proxy build
        with pytest.raises(pr.PetriRunnerError,
                           match="does not declare the audit's agent-loop"):
            pr.require_agent_loop_entry(
                SimpleNamespace(proxy_identity=ident), lens)
    with pytest.raises(pr.PetriRunnerError, match="no proxy identity"):
        pr.require_agent_loop_entry(SimpleNamespace(), lens)
    # non-lens cells have no mask to relocate: never refused
    for ident_str in ("proxy_vllm+passthrough+interleaved",
                      "proxy_vllm+lora+interleaved"):
        pr.require_agent_loop_entry(SimpleNamespace(proxy_identity={}),
                                    _cell(ident_str, "none"))


def test_runner_belt_reads_what_the_launcher_advertises(monkeypatch,
                                                         tmp_path):
    """End to end across the two entry points: the identity block the
    launcher's proxy advertises satisfies the runner's belt."""
    from sisteer.capeval.tiers import petri_runner as pr
    built = {}
    real_make_app = px.make_app
    monkeypatch.setattr(px, "make_app", lambda proxy: (
        built.setdefault("proxy", proxy), real_make_app(proxy))[1])
    _run_launcher(monkeypatch, tmp_path)
    ident = built["proxy"].models_payload()["petri_proxy"]
    pr.require_agent_loop_entry(SimpleNamespace(proxy_identity=ident),
                                _cell("proxy_vllm+lens+interleaved"))
