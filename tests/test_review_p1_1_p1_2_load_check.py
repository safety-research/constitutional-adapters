"""Review round 1 (2026-09-10), P1-1 + P1-2 — the load check's probe is the RUN's, on the RIGHT route.

P1-1  The fullconst launchers sent the full SI as the system message to the `none` vmodel; the production
      proxy refuses a registered prefix on a non-prompted route (si_prefix_unexpected), so every fullconst load
      check was a guaranteed 400 read as a "measured limit". The probe now goes to the PROMPTED route whose
      registered prefix hash matches --system-si, resolved from the standup's routing table. Driven here
      through the REAL RoutingProxy over the committed fullconst manifest (the same H8 machinery production
      runs), never a fake that answers 200 to anything.
P1-2  The probe combined the table's pooled p99 prompt with the table's cap regardless of the window
      (multi-turn 7,496 + 2,048 > 8,704; Petri 17,714 + 1,024 > 16,384) while the run's cap was 512. It now
      derives the cap from the RUN's config, sizes a RENDERED prompt (system + scaffold + filler) to the room the
      window leaves, refuses a probe the window cannot carry as INCOMPATIBLE (exit 4, zero requests) naming the
      geometry the shared budget resolves for it, and does all of that in --preflight before a server exists.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.serving import proxy as px
from sisteer.data.instructions import BY_ID

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))

import load_check as lc  # noqa: E402

MANIFEST = REPO / "configs/capeval/manifests/petri_full22_llama8b_fullconst.yaml"
CFG = REPO / "configs/sdfbase_full22_llama8b_fullconst.yaml"
FULL = BY_ID["si_constitution_full"].text
FULL_WORDS = len(FULL.split())          # 28,977 on the vendored bytes
SCAFFOLD = 17                           # the fake render's chat-scaffold overhead (see _WordTok)


class _WordTok:
    """One token per whitespace word; the served render adds 17 scaffolding tokens (the measured [system,
    minimal user] + generation-header overhead of the pinned Llama-3.1 template)."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] * len(text.split())}

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        return [1] * (sum(len(m["content"].split()) for m in messages) + SCAFFOLD)


class _Backend:
    """The upstream behind the real proxy: records every payload the proxy forwards, answers finish_reason stop
    with a usage block that counts the rendered words (what vLLM's prompt_tokens would be under _WordTok)."""

    def __init__(self):
        self.payloads = []

    async def take_request_failure(self, request_id):
        return None

    async def chat_completions(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        n = sum(len(m["content"].split()) for m in payload["messages"]) + SCAFFOLD
        return {"id": "cmpl-test", "object": "chat.completion", "created": 0, "model": payload["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": n, "completion_tokens": 1, "total_tokens": n + 1}}


class _TailRender:
    """Hermetic agent-loop render probe (as in test_petri_proxy): the proxy never loads a tokenizer here."""

    header_ids = (901, 902, 903, 904)

    def render(self, messages, tools):
        return (1, 2, 3, *self.header_ids)


def _world(tmp_path):
    """The committed fullconst manifest served through the REAL RoutingProxy, with the routing table the standup
    would export beside endpoint.json — the registry carries the registered constitution prefix bytes."""
    from sisteer.capeval.tiers.petri_runner import resolve_prompt_files
    spec = cm.load_petri_spec(MANIFEST)
    source = dc.MappingCellSource({
        "none": dc.RegistryEntry(cell_id="none", kind="none"),
        "prompted__constitution_full": dc.RegistryEntry(
            cell_id="prompted__constitution_full", kind="prompted", si_id="constitution_full",
            si_prefix=FULL.encode("utf-8")),
    })
    behavior_files = {k: (REPO / "configs/capeval/petri" / k / "BEHAVIOR.md").read_bytes()
                      for k in cm.petri_expected_behavior_keys(spec)}
    prompt_files = resolve_prompt_files(spec, override_root=REPO / "configs/capeval/petri/prompts")
    cells = dc.enumerate_cells(spec, source, behavior_files=behavior_files, prompt_files=prompt_files)
    assert not cells.absent
    backend = _Backend()
    proxy = px.RoutingProxy(
        px.routes_from_cells(cells), backend, base_model=spec.target_model, serving_mode=spec.serving_mode,
        prompted_prefixes=px.prompted_prefix_specs(cells, source), exact_replay_strategy=None,
        agent_loop_render=_TailRender())
    table = tmp_path / "routing_table.json"
    px.export_routing_table(cells, source, table, base_model=spec.target_model, serving_mode=spec.serving_mode)

    async def post(body, headers=None):     # the poster contract since the kind tag: (body, headers)
        try:
            d = await proxy.handle_chat_completions(body)
            return 200, json.dumps(d)
        except px.ProxyRefusal as e:
            return e.http_status, json.dumps({"error": {"message": str(e), "code": e.code}})

    return SimpleNamespace(spec=spec, cells=cells, proxy=proxy, backend=backend, table=table, post=post)


def _endpoint(tmp_path, world, *, S=5, L=45056):
    ep = {"proxy_url": "http://127.0.0.1:1/v1", "max_model_len": L, "max_num_seqs": S,
          "model": world.spec.target_model, "model_revision": world.spec.target_revision,
          "routing_table": str(world.table), "pid": None,
          "served_models": [r.virtual_model for r in px.routes_from_cells(world.cells)]}
    p = tmp_path / "endpoint.json"
    p.write_text(json.dumps(ep))
    return p


@pytest.fixture()
def tok(monkeypatch):
    import sisteer.models as sm
    monkeypatch.setattr(sm, "load_tokenizer", lambda name, revision=None, **kw: _WordTok())
    return _WordTok()


def _rec(tmp_path):
    return json.loads((tmp_path / "load_check.json").read_text())


# --------------------------------------------------------------------------- P1-1: the route

def test_p1_1_the_prefixed_probe_hits_the_prompted_route_the_table_registers(tmp_path, tok):
    """With --system-si the probe resolves the PROMPTED vmodel whose si_prefix_hash is the SI's — the real proxy
    verifies the prefix (H8) and forwards all S requests; the record names the vmodel."""
    w = _world(tmp_path)
    ep = _endpoint(tmp_path, w, S=5)
    rc = lc.main(["--endpoint", str(ep), "--eval", "families_served:short", "--cap", "8192",
                  "--system-si", "si_constitution_full", "--model-key", "llama8b"], post=w.post)
    rec = _rec(tmp_path)
    assert rc == 0, rec["verdict"]
    assert rec["vmodel"] == "petri__prompted__constitution_full"
    assert rec["ok"] and rec["n_requests"] == 5 == rec["n_ok"] and rec["n_bad"] == 0
    assert rec["finish_reasons"] == {"stop": 5}
    assert len(w.backend.payloads) == 5
    for p in w.backend.payloads:
        assert p["messages"][0] == {"role": "system", "content": FULL}     # the exact registered prefix
        assert p["max_tokens"] == 8192
    assert rec["max_prompt_tokens"] >= FULL_WORDS + SCAFFOLD


def test_p1_1_the_none_route_refuses_the_registered_prefix_which_is_what_the_old_probe_measured(tmp_path, tok):
    """The defect, reproduced through the real proxy: the same probe forced onto `none` is refused
    si_prefix_unexpected (400) before any upstream call — the old check read exactly this as a capacity limit."""
    w = _world(tmp_path)
    ep = _endpoint(tmp_path, w, S=3)
    rc = lc.main(["--endpoint", str(ep), "--eval", "families_served:short", "--cap", "8192",
                  "--system-si", "si_constitution_full", "--vmodel", "petri__base"], post=w.post)
    rec = _rec(tmp_path)
    assert rc == 3 and rec["ok"] is False and rec["n_bad"] == 3
    assert all(b["status"] == 400 and "si_prefix_unexpected" in b["error"] for b in rec["bad"])
    assert w.backend.payloads == []                       # the proxy refused; nothing reached the engine


def test_p1_1_without_a_prefix_the_probe_hits_none_with_no_system_message(tmp_path, tok):
    w = _world(tmp_path)
    ep = _endpoint(tmp_path, w, S=2)
    rc = lc.main(["--endpoint", str(ep), "--eval", "families_served:short", "--cap", "4096"], post=w.post)
    rec = _rec(tmp_path)
    assert rc == 0 and rec["vmodel"] == "petri__base" and rec["system_si"] is None
    assert all(m["role"] == "user" for p in w.backend.payloads for m in p["messages"])
    assert rec["probe"]["target_prompt_tokens"] == 762                 # the table's p99 fits at this window


def test_p1_1_an_si_the_view_does_not_serve_is_refused_before_any_request(tmp_path, tok):
    """`si_constitution` is a registered SI, but this view registers no prompted route for it: the check refuses
    to guess a vmodel (the proxy would refuse the prefix on every other route)."""
    w = _world(tmp_path)
    ep = _endpoint(tmp_path, w, S=2)
    with pytest.raises(SystemExit, match="no prompted route"):
        lc.main(["--endpoint", str(ep), "--eval", "families_served:short", "--cap", "4096",
                 "--system-si", "si_constitution"], post=w.post)
    assert w.backend.payloads == []


# --------------------------------------------------------------------------- P1-2: the probe's size

def test_p1_2_the_multiturn_probe_takes_the_runs_cap_from_the_config_and_fits_the_window(tmp_path, tok):
    """multi-turn: the table's cap is 2,048 (7,496 + 2,048 > 8,704 — the old probe's guaranteed 400); the RUN's
    cap is the config's OWN multi-turn key (attacks.multiturn_target_max_new_tokens = 4,096) and the window
    served is DERIVED from it (geometry.multiturn_window = 11,648): 7,496 + 4,096 fits it unshrunk, while the
    table's 8,704 can no longer carry the run's cap and refuses INCOMPATIBLE in the preflight — the refusal
    the derived window exists for."""
    from sisteer.capeval.serving.geometry import multiturn_window
    L = multiturn_window(4096)
    # the config's key is a PER-FAMILY map (Crescendo / PAIR 4,096, ActorAttack 512): the probe needs the family the standup serves
    # (--attack, from the sbatch's F22_LOAD_CHECK_ATTACK); without it the preflight refuses rather than guessing a cap
    with pytest.raises(SystemExit, match="pass --attack"):
        lc.main(["--preflight", "--config", str(CFG), "--eval", "multiturn_attacks", "--model-key", "llama8b",
                 "--max-model-len", str(L), "--out", str(tmp_path / "pf0.json")])
    rc = lc.main(["--preflight", "--config", str(CFG), "--eval", "multiturn_attacks", "--model-key", "llama8b", "--attack", "crescendo",
                  "--max-model-len", str(L), "--out", str(tmp_path / "pf.json")])
    rec = json.loads((tmp_path / "pf.json").read_text())
    assert rc == 0, rec["verdict"]
    assert rec["cap"] == 4096 and rec["probe"]["cap_source"] == "config:attacks.multiturn_target_max_new_tokens[crescendo]"
    assert rec["L"] == L == 11648 and rec["probe"]["p99_prompt"] == 7496
    assert rec["probe"]["compatible"] and rec["probe"]["target_prompt_tokens"] == 7496
    assert rec["probe"]["shrunk_from_p99"] is False and rec["n_requests"] == 0 and rec["preflight"] is True
    assert rec["probe"]["rendered_prompt_tokens"] <= 7496 and rec["probe"]["rendered_prompt_tokens"] + 4096 <= L
    # at the table's 8,704 the run's cap leaves only 4,608 tokens: the preflight SHRINKS the probe to that room (it still stands
    # up, but the probe is no longer the p99 prompt) and names the geometry the intended probe needs — >= 7,496 + 4,096, S from the
    # budget — which is what the derived window serves unshrunk
    rc = lc.main(["--preflight", "--config", str(CFG), "--eval", "multiturn_attacks", "--model-key", "llama8b", "--attack", "crescendo",
                  "--max-model-len", "8704", "--out", str(tmp_path / "pf2.json")])
    rec2 = json.loads((tmp_path / "pf2.json").read_text())
    assert rc == 0 and rec2["probe"]["shrunk_from_p99"] is True and rec2["probe"]["rendered_prompt_tokens"] + 4096 <= 8704
    ng = rec2["probe"]["needed_geometry"]
    assert ng["max_model_len"] >= 7496 + 4096 and ng["max_model_len"] <= L and ng["max_num_seqs"] == 475136 // ng["max_model_len"] == 40
    # at the PINNED first-pass window (49,152 x 9 through the measured budget)
    # the probe is the p99 prompt with the whole 4,096 cap and 37,560 tokens to spare — unshrunk, no needed_geometry; the load check
    # sizes its probe from the run's cap and the p99 prompt, so the pin costs it nothing and it refuses nothing
    from sisteer.capeval.serving.geometry import multiturn_geometry
    mg = multiturn_geometry(4096, model_key="llama8b", window=49152, rerun_window=65536)
    assert (mg["window"], mg["S"], mg["rerun_window"], mg["rerun_S"]) == (49152, 9, 65536, 7)
    rc = lc.main(["--preflight", "--config", str(CFG), "--eval", "multiturn_attacks", "--model-key", "llama8b", "--attack", "pair",
                  "--max-model-len", str(mg["window"]), "--out", str(tmp_path / "pf3.json")])
    rec3 = json.loads((tmp_path / "pf3.json").read_text())
    assert rc == 0, rec3["verdict"]
    assert rec3["L"] == 49152 and rec3["cap"] == 4096 and rec3["probe"]["cap_source"] == "config:attacks.multiturn_target_max_new_tokens[pair]"
    assert rec3["probe"]["compatible"] and rec3["probe"]["shrunk_from_p99"] is False and rec3["probe"]["needed_geometry"] is None
    assert rec3["probe"]["target_prompt_tokens"] == 7496 and rec3["probe"]["rendered_prompt_tokens"] + 4096 <= 49152
    # ActorAttack at 4,096 and ITS pinned window 49,152 (the canonical measured Crescendo
    # window; S 9; re-run 65,536 x 7): the probe is the p99 prompt + 4,096, unshrunk, with 37,560 tokens to spare; at PAIR's canonical
    # 16,384 and at the derived 11,648 it still fits unshrunk: the family's cap sizes the probe
    mga = multiturn_geometry(4096, model_key="llama8b", window=49152, rerun_window=65536)
    assert (mga["window"], mga["S"], mga["rerun_window"], mga["rerun_S"], mga["derived_window"], mga["derived_rerun_window"]) == (49152, 9, 65536, 7, 11648, 16768)
    mgp = multiturn_geometry(4096, model_key="llama8b", window=16384, rerun_window=32768)
    assert (mgp["window"], mgp["S"], mgp["rerun_window"], mgp["rerun_S"]) == (16384, 29, 32768, 14)                 # PAIR's canonical window
    for L_a in (49152, 16384, 11648):
        rc = lc.main(["--preflight", "--config", str(CFG), "--eval", "multiturn_attacks", "--model-key", "llama8b", "--attack", "actorattack",
                      "--max-model-len", str(L_a), "--out", str(tmp_path / f"pfa{L_a}.json")])
        ra = json.loads((tmp_path / f"pfa{L_a}.json").read_text())
        assert rc == 0, ra["verdict"]
        assert ra["L"] == L_a and ra["cap"] == 4096 and ra["probe"]["cap_source"] == "config:attacks.multiturn_target_max_new_tokens[actorattack]"
        assert ra["probe"]["compatible"] and ra["probe"]["shrunk_from_p99"] is False and ra["probe"]["needed_geometry"] is None
        assert ra["probe"]["target_prompt_tokens"] == 7496 and ra["probe"]["rendered_prompt_tokens"] <= 7496 and ra["probe"]["rendered_prompt_tokens"] + 4096 <= L_a
    # a COMBINED standup (the `-all` job shape) probes at the MAX cap over its families (4,096) at the max window
    rc = lc.main(["--preflight", "--config", str(CFG), "--eval", "multiturn_attacks", "--model-key", "llama8b", "--attack", "crescendo,pair,actorattack",
                  "--max-model-len", "49152", "--out", str(tmp_path / "pfall.json")])
    rall = json.loads((tmp_path / "pfall.json").read_text())
    assert rc == 0 and rall["cap"] == 4096 and rall["probe"]["cap_source"] == "config:attacks.multiturn_target_max_new_tokens[crescendo] (max over crescendo,pair,actorattack)"
    # a family the map does not name refuses (never a default)
    with pytest.raises(SystemExit, match=r"names no cap for \['bon'\]"):
        lc.main(["--preflight", "--config", str(CFG), "--eval", "multiturn_attacks", "--model-key", "llama8b", "--attack", "bon",
                 "--max-model-len", "32768", "--out", str(tmp_path / "pfbad.json")])


def test_p1_2_the_petri_probe_is_shrunk_to_the_window_and_names_the_geometry_the_budget_resolves(tmp_path, tok):
    """Petri: p99 17,714 + cap 1,024 does not fit 16,384. The probe shrinks the PROMPT to the room the window
    leaves (15,360) instead of sending a guaranteed 400, and records the wider geometry the shared budget resolves
    for the intended probe — S from budget_for('llama8b'), never hand-picked."""
    w = _world(tmp_path)
    ep = _endpoint(tmp_path, w, S=3, L=16384)
    rc = lc.main(["--endpoint", str(ep), "--eval", "petri", "--model-key", "llama8b"], post=w.post)
    rec = _rec(tmp_path)
    assert rc == 0, rec["verdict"]
    pr = rec["probe"]
    assert rec["cap"] == 1024 and pr["cap_source"] == "table:basis.requested_max_tokens"
    assert pr["p99_prompt"] == 17714 and pr["shrunk_from_p99"] is True and pr["target_prompt_tokens"] == 15360
    assert pr["rendered_prompt_tokens"] <= 15360
    ng = pr["needed_geometry"]
    assert ng["needed_tokens"] == 17714 + 1024 and ng["max_model_len"] == 18752
    assert ng["max_num_seqs"] == 475136 // 18752 == 25 and ng["budget_source"] == "measured"   # the 2026-09-10 measured row
    assert all(p["max_tokens"] == 1024 for p in w.backend.payloads) and len(w.backend.payloads) == 3
    assert rec["max_prompt_tokens"] + 1024 <= 16384                      # every served request fit the window


def test_p1_2_a_cap_the_window_cannot_carry_is_incompatible_before_any_request(tmp_path, tok):
    """cap 9,000 at 8,704 leaves no room for a prompt: exit 4 with ZERO requests and a named wider geometry —
    never a 400 diagnosed as capacity."""
    w = _world(tmp_path)
    ep = _endpoint(tmp_path, w, S=3, L=8704)
    rc = lc.main(["--endpoint", str(ep), "--eval", "multiturn_attacks", "--cap", "9000", "--model-key", "llama8b"],
                 post=w.post)
    rec = _rec(tmp_path)
    assert rc == 4 and rec["ok"] is False and rec["n_requests"] == 0 and w.backend.payloads == []
    assert rec["verdict"].startswith("INCOMPATIBLE") and rec["probe"]["compatible"] is False
    ng = rec["probe"]["needed_geometry"]
    assert ng["max_model_len"] > 9000 + SCAFFOLD and ng["max_num_seqs"] == 475136 // ng["max_model_len"]


def test_p1_2_the_fullconst_16384_rung_does_not_fit_45056_and_refuses_in_preflight(tmp_path, tok):
    """The full constitution renders to 28,994 tokens; the 16,384 rung needs 45,378 > 45,056 — the rung's own
    window (51,200, rung table) must be served, and the preflight says so with no server allocated."""
    rc = lc.main(["--preflight", "--config", str(CFG), "--eval", "families_served:short", "--model-key", "llama8b",
                  "--max-model-len", "45056", "--cap", "16384", "--system-si", "si_constitution_full",
                  "--out", str(tmp_path / "pf.json")])
    rec = json.loads((tmp_path / "pf.json").read_text())
    assert rc == 4 and rec["n_requests"] == 0 and rec["verdict"].startswith("INCOMPATIBLE")
    assert rec["probe"]["scaffold_tokens"] == FULL_WORDS + SCAFFOLD
    assert rec["probe"]["needed_geometry"]["max_model_len"] >= FULL_WORDS + SCAFFOLD + 16384
    # and the 8,192 smoke DOES fit the same window
    rc = lc.main(["--preflight", "--config", str(CFG), "--eval", "families_served:short", "--model-key", "llama8b",
                  "--max-model-len", "45056", "--cap", "8192", "--system-si", "si_constitution_full",
                  "--out", str(tmp_path / "pf2.json")])
    rec = json.loads((tmp_path / "pf2.json").read_text())
    assert rc == 0 and rec["probe"]["compatible"] and rec["probe"]["rendered_prompt_tokens"] + 8192 <= 45056


def test_p1_2_the_cap_key_follows_the_leg(tmp_path, tok):
    import yaml
    cfg = yaml.safe_load(CFG.read_text())
    # the leg's own multi-turn key wins — PER FAMILY, so the family the standup serves is needed and a config
    # without a family's entry refuses; a config without the key keeps the shared cap
    assert lc.cap_for("multiturn_attacks", cfg, None, attack="crescendo") == (4096, "config:attacks.multiturn_target_max_new_tokens[crescendo]")
    assert lc.cap_for("multiturn_attacks", cfg, None, attack="pair") == (4096, "config:attacks.multiturn_target_max_new_tokens[pair]")
    assert lc.cap_for("multiturn_attacks", cfg, None, attack="actorattack") == (4096, "config:attacks.multiturn_target_max_new_tokens[actorattack]")
    with pytest.raises(SystemExit, match="pass --attack"):
        lc.cap_for("multiturn_attacks", cfg, None)
    assert lc.cap_for("multiturn_attacks", {"attacks": {"target_max_new_tokens": 512, "multiturn_target_max_new_tokens": 4096}}, None) == (4096, "config:attacks.multiturn_target_max_new_tokens")
    assert lc.cap_for("multiturn_attacks", {"attacks": {"target_max_new_tokens": 512}}, None) == (512, "config:attacks.target_max_new_tokens")
    assert lc.cap_for("families_served:short", cfg, None) == (4096, "config:attacks.static_max_new_tokens")
    # the BoN cap is the caps table's NATIVE 4,096 (bon.max_new_tokens WRITTEN by the generator)
    assert lc.cap_for("bon", cfg, None) == (4096, "config:bon.max_new_tokens")
    assert lc.cap_for("families_prompted", cfg, None) == (4096, "config:bon.max_new_tokens")
    assert lc.cap_for("petri", cfg, None) == (1024, "table:basis.requested_max_tokens")
    assert lc.cap_for("petri", cfg, 8192) == (8192, "flag:--cap")


# --------------------------------------------------------------------------- the wiring

def test_common_sh_preflights_the_probe_before_a_server_is_allocated_and_passes_the_run_config_after():
    common = (REPO / "scripts/sdfbase/full22/common.sh").read_text()
    body = common[common.index("f22_standup(){"):]
    i_pre = body.index("load_check.py --preflight --config \"$CFG\"")
    i_up = body.index("scripts/sdfbase/mt_serving_up.py --repo-dir")
    i_post = body.index("load_check.py --endpoint \"$ep\"")
    assert i_pre < i_up < i_post, "preflight BEFORE the standup is launched; the served probe after"
    pre = body[i_pre:i_pre + 900]
    assert "--model-key \"$KEY\"" in pre and "${gL:+--max-model-len \"$gL\"}" in pre
    assert "${F22_LOAD_CHECK_CAP:+--cap \"$F22_LOAD_CHECK_CAP\"}" in pre
    assert "${F22_LOAD_CHECK_SYSTEM_SI:+--system-si \"$F22_LOAD_CHECK_SYSTEM_SI\"}" in pre
    assert "return 1" in pre and "REFUSING TO STAND UP" in pre
    post = body[i_post:i_post + 400]
    assert "--config \"$CFG\" --model-key \"$KEY\"" in post
