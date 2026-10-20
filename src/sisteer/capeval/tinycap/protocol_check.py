"""``check_protocol_sampling`` — the in-image resolved-parameter check of the
``protocol`` sampling mode.

THE QUESTION. Under ``--sampling protocol`` the provider sends, per pinned leg,
the benchmark's OWN reference decoding — BigCodeBench-200 at temperature 0.0,
BFCL-200 at 0.001 — with ``top_p 1.0`` and ``top_k 0`` spelled explicitly.
Whether those numbers EXECUTE is a property of the served backend, not of the
request: vLLM 0.25.1 clamps every ``0 < T < 0.01`` to 0.01
(``sampling_params._MAX_TEMP``), and ``generation_config: auto`` fills a
MISSING request key from the model's ``generation_config.json``. This module
resolves each task's real request body the way the server does and asserts
the EXECUTED values — never assumes them.

THE TWO-INTERPRETER BRIDGE. The serving image holds TWO venvs: the
CLIENT venv (inspect_ai + ``TinycapAPI``, ``CLIENT_PY``) and the SERVING venv
(vLLM). They are never merged. ``compose`` runs under the client interpreter
and writes ``requests.json`` — per task, the body ``TinycapAPI.sampling_params``
+ inspect's ``openai_completion_params`` would put on the wire (``temperature``,
``top_p``, ``seed``, ``extra_body.top_k`` flattened, ``tools`` on the BFCL
leg). ``resolve`` runs under the serving interpreter: it loads that JSON,
builds ``ChatCompletionRequest(**body)`` and calls
``to_sampling_params(max_tokens, default_sampling_params)`` — the exact server
path (``vllm/entrypoints/openai/chat_completion/protocol.py`` at v0.25.1:
``def to_sampling_params(self, max_tokens: int, default_sampling_params: dict)``)
— TWICE per body: once WITHOUT touching the wheel (the NEGATIVE control: BFCL
must resolve to 0.01, proving the clamp this override lifts is real on THIS
wheel; a wheel that no longer clamps is a different instrument), once WITH
``vllm.sampling_params._MAX_TEMP = 0.001`` set exactly as ``vllm_entry`` sets it
(PRODUCTION: BCB 0.0, BFCL 0.001 EXACTLY, top_p 1.0, top_k 0, min_p 0.0, the
penalties at their no-op values). It also round-trips the resolved params
through msgspec (``decode(encode(params), type=SamplingParams)``) — the
engine-core decode path — and asserts the round trip keeps 0.001 under the
override and clamps to 0.01 without it.

THE DEFAULTS. ``default_sampling_params`` is what the server hands
``to_sampling_params``: ``ModelConfig.get_diff_sampling_param()`` — the model's
``generation_config.json`` diff (``generation_config: auto``) with
``override_generation_config`` merged, filtered to vLLM's ``available_params``.
``resolve`` builds it FROM THE LAUNCH FLAGS the standup would compose
(``--generation-config``, ``--override-generation-config``): first through
``vllm.config.ModelConfig`` itself when it constructs on a CPU container, else
through a faithful replica of ``get_diff_sampling_param`` over
``vllm.transformers_utils.config.try_get_generation_config`` — RECORDED as
``defaults_source`` either way, never silently. Two controls prove the
inheritance the explicit values prevent: (a) at temperature 1.0 (a POSITIVE
temperature — at T < 1e-5 vLLM resets ``top_p`` to 1.0 regardless, so a T-0
control would prove nothing) with ``top_p`` OMITTED, against a synthetic
``generation_config.json`` fixture carrying ``top_p 0.95``, the resolution must
return 0.95, and the SAME request with ``top_p 1.0`` present must return 1.0;
(b) when the launch flags carry an ``--override-generation-config`` (Qwen's
composes ``top_p 0.8``), the T-1.0 top_p-omitted request must resolve to THAT
value, while a protocol request with explicit ``top_p 1.0`` / ``top_k 0`` must
still resolve to 1.0 / 0.

``judge`` is PURE (no vLLM, no inspect_ai) so its expectations are unit-tested
on synthetic records in the grading host venv; the preflight refuses a protocol
launch whose record is not ``ok`` ("no silent substitution").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: the wheel's clamp and the override value (kept as literals here too: ``resolve``
#: runs in the SERVING venv, whose sisteer import is the staged tree's — the
#: contracts are importable there, but the numbers are pinned where they act)
PINNED_MAX_TEMP = 0.01
OVERRIDE_MAX_TEMP = 0.001
#: the RESOLVED terms recorded per task (``contracts.PROTOCOL_EFFECTIVE_FIELDS``)
EFFECTIVE_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty",
                    "frequency_penalty")
#: vLLM 0.25.1 ``ModelConfig.get_diff_sampling_param``'s key list, verbatim
AVAILABLE_PARAMS = ("repetition_penalty", "temperature", "top_k", "top_p", "min_p", "max_new_tokens")
#: the probe request bodies' item id / seed inputs (synthetic; the seed is the
#: provider's own rule over them)
PROBE_ITEM = "protocol-check-probe"
PROBE_MAX_TOKENS = 64
#: a BFCL-shaped tool, so the BFCL body carries ``tools`` the way the leg's do
PROBE_TOOL = {"type": "function",
              "function": {"name": "get_weather", "description": "Weather for a city.",
                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                          "required": ["city"]}}}


# --------------------------------------------------------------------------- #
# step 1 — compose, under the CLIENT interpreter                               #
# --------------------------------------------------------------------------- #

def compose_requests(tasks: Sequence[str], *, model: str) -> Dict[str, Any]:
    """Per protocol task, the request body the provider puts on the wire —
    through the STAGED tree's ``TinycapAPI`` (its ``sampling_params`` selects the
    task's executed row) and inspect_ai's own OpenAI serialiser
    (``completion_params``), with ``extra_body`` flattened into the body as the
    SDK does. Plus the two control bodies: ``inherit`` (T 1.0, ``top_p`` omitted)
    and ``explicit`` (T 1.0, ``top_p`` 1.0, ``top_k`` 0)."""
    from inspect_ai.model import GenerateConfig

    from sisteer.capeval.tiers import inspect_served as isv
    from sisteer.capeval.tinycap import contracts as C

    rows = C.protocol_executed_by_task(list(tasks))
    cls = isv._register_provider()
    route = {"cell_id": "none", "virtual_model": "petri__base",
             "serving_identity": "proxy_vllm+passthrough+interleaved", "cell_class": "control",
             "intervention_hash": "0" * 16}
    api = cls(model_name=C.inspect_model_name("petri__base"), base_url="http://127.0.0.1:1/v1",
              api_key="protocol-check", config=GenerateConfig(), si=None, thinking=C.THINKING_OFF,
              route=route, sampling_by_task=rows)
    out: Dict[str, Any] = {"model": model, "tasks": {}, "controls": {}}
    for task in sorted(rows):
        extra: Dict[str, Any] = {}
        params = api.sampling_params(GenerateConfig(), extra, task=task, item=PROBE_ITEM, idx=0)
        cfg = GenerateConfig(**{k: v for k, v in params.items() if k in ("temperature", "top_p", "presence_penalty")},
                             extra_body=extra or None, seed=isv.sample_seed(PROBE_ITEM, 0),
                             max_tokens=PROBE_MAX_TOKENS)
        # a TOOL-CALLING row task (bfcl_200 and, since capeval-full-tasks, bfcl_full): contracts.TOOL_CALL_CONTROLS
        tool_task = task in {t for fam in C.TOOL_CALL_CONTROLS for t in C.TASKS_OF_CONTROL[fam]}
        wire = api.completion_params(cfg, tools=tool_task)
        body = {k: v for k, v in wire.items() if k != "extra_body"}
        body.update(dict(wire.get("extra_body") or {}))          # the SDK merges extra_body into the JSON body
        body["model"] = model
        body["messages"] = [{"role": "user", "content": "What is the weather in Paris?"}]
        if tool_task:                        # bfcl_200 and, since capeval-full-tasks, bfcl_full
            body["tools"] = [PROBE_TOOL]
            body["tool_choice"] = "auto"
        out["tasks"][task] = {"body": body,
                              "executed": {"temperature": float(rows[task].temperature),
                                           "top_p": float(rows[task].top_p), "top_k": int(rows[task].top_k)}}
    base = {"model": model, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": PROBE_MAX_TOKENS}
    out["controls"]["inherit"] = {**base, "temperature": 1.0}                       # top_p OMITTED on purpose
    out["controls"]["explicit"] = {**base, "temperature": 1.0, "top_p": 1.0, "top_k": 0}
    return out


# --------------------------------------------------------------------------- #
# step 2 — resolve, under the SERVING interpreter                              #
# --------------------------------------------------------------------------- #

def launch_generation_config(launch_flags: Sequence[str]) -> Dict[str, Any]:
    """``{"generation_config": <str>, "override_generation_config": <dict>}`` as
    vLLM's arg parser would read them off the launch argv: ``auto`` when
    ``--generation-config`` is absent (the wheel's default), the JSON dict of
    ``--override-generation-config`` when present (``{}`` otherwise). Both the
    split (``--flag value``) and the ``--flag=value`` spellings are read."""
    flags = list(launch_flags)
    gen, override = "auto", {}

    def _value(name: str) -> Optional[str]:
        for i, tok in enumerate(flags):
            if tok == name:
                if i + 1 >= len(flags):
                    # code review: a trailing valueless flag used to read as ABSENT ("auto" / {}),
                    # where vLLM's own parser rejects the argv — refuse the same way
                    raise ValueError(f"launch flag {name} has no value (trailing token) in {flags!r}")
                return flags[i + 1]
            if tok.startswith(name + "="):
                return tok[len(name) + 1:]
        return None

    g = _value("--generation-config")
    if g is not None:
        gen = g
    o = _value("--override-generation-config")
    if o is not None:
        parsed = json.loads(o)
        if not isinstance(parsed, dict):
            raise ValueError(f"--override-generation-config must be a JSON object, got {o!r}")
        override = parsed
    return {"generation_config": gen, "override_generation_config": override}


def replica_diff_sampling_param(model_path: str, *, revision: Optional[str], generation_config: str = "auto",
                                override_generation_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """A faithful REPLICA of vLLM 0.25.1 ``ModelConfig.get_diff_sampling_param``
    (``vllm/config/model.py``, read at the tag): ``{}`` under ``vllm``, else the
    model's / the named dir's ``GenerationConfig.to_diff_dict()`` through
    ``vllm.transformers_utils.config.try_get_generation_config``; the override
    merged on top; filtered to :data:`AVAILABLE_PARAMS`; ``max_new_tokens``
    renamed ``max_tokens``. Used when ``ModelConfig`` itself cannot be built on
    the CPU container — RECORDED as such by the caller, never silently."""
    from vllm.transformers_utils.config import try_get_generation_config

    if generation_config == "vllm":
        config: Dict[str, Any] = {}
    else:
        src = model_path if generation_config == "auto" else generation_config
        gc = try_get_generation_config(src, trust_remote_code=False,
                                       revision=revision if generation_config == "auto" else None)
        config = dict(gc.to_diff_dict()) if gc is not None else {}
    config.update(dict(override_generation_config or {}))
    if any(p in config for p in AVAILABLE_PARAMS):
        diff = {p: config.get(p) for p in AVAILABLE_PARAMS if config.get(p) is not None}
        if "max_new_tokens" in diff:
            diff["max_tokens"] = diff.pop("max_new_tokens")
    else:
        diff = {}
    return diff


def server_default_sampling_params(model_path: str, *, revision: Optional[str], generation_config: str,
                                   override_generation_config: Mapping[str, Any]) -> Dict[str, Any]:
    """``{"defaults": {...}, "source": ..., "model_config_error": ...}`` — the
    diff sampling defaults the SERVER hands ``to_sampling_params``: through
    ``vllm.config.ModelConfig`` when it constructs (the authoritative path),
    else the replica, with the construction error recorded beside it."""
    try:
        from vllm.config import ModelConfig
        mc = ModelConfig(model=model_path, revision=revision, tokenizer=model_path, dtype="bfloat16",
                         generation_config=generation_config,
                         override_generation_config=dict(override_generation_config))
        return {"defaults": dict(mc.get_diff_sampling_param()), "source": "ModelConfig.get_diff_sampling_param",
                "model_config_error": None}
    except Exception as e:  # noqa: BLE001 — a CPU container may not build a ModelConfig; recorded, not hidden
        return {"defaults": replica_diff_sampling_param(model_path, revision=revision,
                                                        generation_config=generation_config,
                                                        override_generation_config=override_generation_config),
                "source": "replica:get_diff_sampling_param", "model_config_error": f"{type(e).__name__}: {e}"[:600]}


def _params_record(params) -> Dict[str, Any]:
    return {f: (None if getattr(params, f, None) is None else
                (int(getattr(params, f)) if f == "top_k" else float(getattr(params, f))))
            for f in EFFECTIVE_FIELDS}


def _resolve_body(body: Mapping[str, Any], defaults: Mapping[str, Any]) -> Dict[str, Any]:
    """One body through the server's path: ``ChatCompletionRequest(**body)``,
    ``to_sampling_params(max_tokens, default_sampling_params)``, then the msgspec
    round trip (the engine-core decode) — both recorded."""
    import msgspec
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.sampling_params import SamplingParams

    req = ChatCompletionRequest(**dict(body))
    max_tokens = int(body.get("max_tokens") or PROBE_MAX_TOKENS)
    params = req.to_sampling_params(max_tokens, dict(defaults))
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(params), type=SamplingParams)
    return {"resolved": _params_record(params), "round_trip": _params_record(decoded)}


def resolve_requests(doc: Mapping[str, Any], *, model_path: str, revision: Optional[str], fixture_dir: str,
                     launch_flags: Sequence[str] = ()) -> Dict[str, Any]:
    """The whole resolution record (serving interpreter): the wheel's clamp
    value, the model's server defaults (from the launch flags), every task WITH
    and WITHOUT the override, the inheritance controls against the fixture and
    against the model's own defaults, and the override control when the flags
    carry an ``--override-generation-config``."""
    import vllm.sampling_params as sp

    wheel_max_temp = float(sp._MAX_TEMP)
    gen = launch_generation_config(launch_flags)
    model_defaults = server_default_sampling_params(model_path, revision=revision, **gen)
    # the fixture control reads the fixture's generation_config.json THROUGH the model's
    # ModelConfig (`generation_config=<dir>` is vLLM's own "named dir" spelling): passing the
    # fixture dir as `model` could never construct a ModelConfig, so that control silently
    # took the replica path on every run (code review) — now it exercises the same path the
    # model defaults take, and records which one it got
    fixture_defaults = server_default_sampling_params(model_path, revision=revision, generation_config=str(fixture_dir),
                                                      override_generation_config={})
    out: Dict[str, Any] = {"wheel_max_temp": wheel_max_temp, "generation_config": gen,
                           "model_defaults": model_defaults, "fixture_defaults": fixture_defaults,
                           "negative_control": {}, "production": {}, "controls": {}}
    tasks = doc.get("tasks") or {}
    # WITHOUT the override — the wheel as shipped
    for task, entry in tasks.items():
        out["negative_control"][task] = _resolve_body(entry["body"], model_defaults["defaults"])
    ctl = doc.get("controls") or {}
    out["controls"]["inherit_fixture"] = _resolve_body(ctl["inherit"], fixture_defaults["defaults"])
    out["controls"]["explicit_fixture"] = _resolve_body(ctl["explicit"], fixture_defaults["defaults"])
    out["controls"]["inherit_model"] = _resolve_body(ctl["inherit"], model_defaults["defaults"])
    out["controls"]["explicit_model"] = _resolve_body(ctl["explicit"], model_defaults["defaults"])
    # WITH the override, applied in-process exactly as vllm_entry applies it
    sp._MAX_TEMP = OVERRIDE_MAX_TEMP
    try:
        for task, entry in tasks.items():
            out["production"][task] = _resolve_body(entry["body"], model_defaults["defaults"])
    finally:
        sp._MAX_TEMP = wheel_max_temp
    # round 6: a REAL spawned-child control — a fresh interpreter under
    # multiprocessing's spawn context imports the wheel and constructs SamplingParams(T=0.001):
    # with the shim armed it must read 0.001, without it the wheel's 0.01 (the negative control)
    from sisteer.capeval.serving import vllm_entry as VE
    out["spawned_child_control"] = {"armed": VE.spawned_child_control(armed=True),
                                    "unarmed": VE.spawned_child_control(armed=False)}
    out["override_mechanism"] = VE.MECHANISM_SHIM
    return out


# --------------------------------------------------------------------------- #
# the pure judge                                                               #
# --------------------------------------------------------------------------- #

def judge(requests: Mapping[str, Any], resolved: Mapping[str, Any], *, fixture_top_p: float = 0.95,
          pinned_max_temp: float = PINNED_MAX_TEMP, override_max_temp: float = OVERRIDE_MAX_TEMP) -> Dict[str, Any]:
    """The verdict over the two files (no vLLM, no inspect_ai): every problem
    is named; ``ok`` iff none. Assertions:

    * the wheel's ``_MAX_TEMP`` is EXACTLY the pinned 0.01;
    * PRODUCTION (with the override): each task resolved to its EXECUTED
      temperature exactly (BCB 0.0; BFCL 0.001), ``top_p 1.0``, ``top_k 0``,
      ``min_p 0.0``, the penalties at their no-op values, and the msgspec round
      trip keeps them;
    * NEGATIVE control (without it): every task whose executed temperature lies
      strictly inside (0, 0.01) resolved to 0.01 — the clamp is real on this
      wheel — and its round trip stays clamped; a T-0 task stays 0.0;
    * INHERITANCE control: the T-1.0 request with ``top_p`` omitted resolved
      against the fixture to ``fixture_top_p`` (0.95), the same request with
      ``top_p 1.0`` present to 1.0;
    * OVERRIDE control: when the launch flags carry an
      ``--override-generation-config`` naming ``top_p``, the T-1.0 top_p-omitted
      request against the MODEL's defaults resolved to that value, while the
      explicit request resolved to ``top_p 1.0`` / ``top_k 0``.
    The record carries ``production`` as ``{task: resolved}`` — the map the
    driver copies into the manifest's ``by_task.effective``
    (``contracts.check_protocol_effective`` re-asserts the same values)."""
    problems: List[str] = []
    wheel = resolved.get("wheel_max_temp")
    if wheel != pinned_max_temp:
        problems.append(f"the wheel's _MAX_TEMP is {wheel!r}, not the pinned {pinned_max_temp} — another instrument")
    tasks = dict(requests.get("tasks") or {})
    prod_out: Dict[str, Dict[str, Any]] = {}
    neg_out: Dict[str, Dict[str, Any]] = {}
    for task, entry in sorted(tasks.items()):
        want_t = float(entry["executed"]["temperature"])
        prod = (resolved.get("production") or {}).get(task)
        neg = (resolved.get("negative_control") or {}).get(task)
        if not prod or not neg:
            problems.append(f"{task}: missing a production / negative-control resolution")
            continue
        p, pr = prod["resolved"], prod["round_trip"]
        want = {"temperature": want_t, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
                "presence_penalty": 0.0, "frequency_penalty": 0.0}
        for f, v in want.items():
            if p.get(f) != v:
                problems.append(f"{task}: production {f} resolved {p.get(f)!r}, the protocol executes {v!r}")
            if pr.get(f) != v:
                problems.append(f"{task}: production msgspec round trip {f} = {pr.get(f)!r}, not {v!r} "
                                "(the engine-core decode would re-clamp)")
        n, nr = neg["resolved"], neg["round_trip"]
        if 0.0 < want_t < pinned_max_temp:
            for label, rec in (("resolved", n), ("round_trip", nr)):
                if rec.get("temperature") != pinned_max_temp:
                    problems.append(f"{task}: negative control {label} temperature {rec.get('temperature')!r}, "
                                    f"expected the wheel's clamp {pinned_max_temp} — the clamp this override lifts "
                                    "is not real on this wheel (a different instrument)")
        elif n.get("temperature") != want_t:
            problems.append(f"{task}: negative control temperature {n.get('temperature')!r} != {want_t!r}")
        prod_out[task] = dict(p)
        neg_out[task] = dict(n)
    ctl = resolved.get("controls") or {}
    inh = (ctl.get("inherit_fixture") or {}).get("resolved") or {}
    exp = (ctl.get("explicit_fixture") or {}).get("resolved") or {}
    if inh.get("top_p") != fixture_top_p:
        problems.append(f"inheritance control: T 1.0 with top_p OMITTED resolved top_p {inh.get('top_p')!r} against "
                        f"the fixture's generation_config.json (top_p {fixture_top_p}) — expected {fixture_top_p} "
                        "(the inheritance the explicit value prevents)")
    if exp.get("top_p") != 1.0 or exp.get("top_k") != 0:
        problems.append(f"inheritance control: T 1.0 with top_p 1.0 / top_k 0 present resolved top_p "
                        f"{exp.get('top_p')!r} top_k {exp.get('top_k')!r} — expected 1.0 / 0")
    gen = resolved.get("generation_config") or {}
    override = dict(gen.get("override_generation_config") or {})
    inh_m = (ctl.get("inherit_model") or {}).get("resolved") or {}
    exp_m = (ctl.get("explicit_model") or {}).get("resolved") or {}
    if "top_p" in override:
        if inh_m.get("top_p") != float(override["top_p"]):
            problems.append(f"override control (D16): the launch's --override-generation-config top_p "
                            f"{override['top_p']!r} was not inherited by a T-1.0 top_p-omitted request "
                            f"(resolved {inh_m.get('top_p')!r})")
    if exp_m.get("top_p") != 1.0 or exp_m.get("top_k") != 0:
        problems.append(f"override control: the explicit protocol values resolved top_p {exp_m.get('top_p')!r} "
                        f"top_k {exp_m.get('top_k')!r} against the model's server defaults — expected 1.0 / 0")
    # round 6: the spawned-child control — present, the armed child at the override, the unarmed at the wheel's clamp
    spawned = resolved.get("spawned_child_control")
    if not isinstance(spawned, Mapping) or not isinstance(spawned.get("armed"), Mapping) or not isinstance(spawned.get("unarmed"), Mapping):
        problems.append("spawned-child control missing: the check did not run the shim in a real spawned interpreter")
    else:
        armed, unarmed = spawned["armed"], spawned["unarmed"]
        if armed.get("ok") is not True or armed.get("temperature") != override_max_temp:
            problems.append(f"spawned-child control (armed): a spawned interpreter resolved temperature "
                            f"{armed.get('temperature')!r} (report {armed.get('error') or armed.get('sitecustomize')!r}) — "
                            f"expected {override_max_temp}: the sitecustomize shim did not reach the child")
        if unarmed.get("ok") is not True or unarmed.get("temperature") != pinned_max_temp:
            problems.append(f"spawned-child control (unarmed): a spawned interpreter resolved temperature "
                            f"{unarmed.get('temperature')!r} — expected the wheel's clamp {pinned_max_temp} (the negative "
                            "control proves the shim, not the wheel, lifts it)")
    return {"ok": not problems, "problems": problems,
            "spawned_child_control": spawned, "override_mechanism": resolved.get("override_mechanism"),
            "wheel_max_temp": wheel, "override_max_temp": override_max_temp,
            "defaults_source": (resolved.get("model_defaults") or {}).get("source"),
            "model_config_error": (resolved.get("model_defaults") or {}).get("model_config_error"),
            "model_defaults": (resolved.get("model_defaults") or {}).get("defaults"),
            "generation_config": gen,
            "production": prod_out, "negative_control": neg_out,
            "controls": {k: (v or {}).get("resolved") for k, v in ctl.items()},
            "requests": {t: e.get("body") for t, e in tasks.items()},
            "fixture_top_p": fixture_top_p}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="protocol sampling check — compose (client venv) / resolve (serving venv)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compose")
    c.add_argument("--out", required=True)
    c.add_argument("--tasks", required=True, help="comma-separated row tasks (inspect/bigcodebench_200,inspect/bfcl_200; "
                                                 "capeval-full-tasks: inspect/bigcodebench_full,inspect/bfcl_full)")
    c.add_argument("--model", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("--requests", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model-path", required=True)
    r.add_argument("--revision", default=None)
    r.add_argument("--fixture-dir", required=True)
    r.add_argument("--launch-flags", default="[]", help="JSON list of the server launch flags (D16)")
    a = ap.parse_args(argv)
    if a.cmd == "compose":
        doc = compose_requests([t for t in a.tasks.split(",") if t], model=a.model)
        Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True))
        return 0
    doc = json.loads(Path(a.requests).read_text())
    out = resolve_requests(doc, model_path=a.model_path, revision=a.revision, fixture_dir=a.fixture_dir,
                           launch_flags=json.loads(a.launch_flags))
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
