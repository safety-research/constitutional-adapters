"""The closed-loop throughput sampler behind the proxy, factored out of
``scripts/sdfbase/multilora_cert.py`` so the timing harness
(``scripts/sdfbase/native_timing_72.py``) and the
certification driver share ONE sampler: ``FILLER_PROMPT``, ``METRIC_RE``,
``read_metrics`` and ``closed_loop_via_proxy`` are the certification driver's
own text, copied — not rewritten — so the driver can later import them from
here (``tests/test_native_timing_72.py`` pins signature, source and the prompt
bytes against the driver's copies).

What is NEW here, beside the copies:

- :func:`ms_per_step_gate` — the lens gate's ``implied_step_ms`` (the formula
  that produced every §7.2 ms/step number: ``1000 x running_mean / gen_tok_s``,
  ``scripts/lens_vectorize_gate.py`` ``implied_step_ms``): the effective batch
  the ENGINE reports running, not the nominal client count. The gate reads no
  engine step counter; its "step" is implied from the running gauge and the
  generation counter.
- :func:`ms_per_step_nominal` — the certification driver's ``1000 x B / tok_s``
  (``throughput_leg``'s ``ms_per_step``): the nominal client concurrency.
- :func:`ms_per_step_from_counter` — from an engine ITERATION counter when
  ``/metrics`` exposes one (vLLM's ``vllm:iteration_tokens_total`` histogram:
  its ``_count`` series is the number of scheduler steps; ``read_metrics`` sums
  it under ``vllm:iteration_tokens_total_count``). The copied loop does not read
  it over its window, so the harness samples it alongside
  (:class:`StepCounterSampler`) and aligns the span to the loop's gauges.
- :func:`prompt_bound` — binds the prompt the copied loop sends for the length
  of a ``with`` block. The loop reads the module global ``FILLER_PROMPT`` at call
  time (its text is the certification driver's); the §7.2 protocol sends the
  gate's ~800-token passage instead, and this is the ONE seam that changes
  the prompt without touching the copied function (not thread-safe by design:
  the harness runs its windows serially).
- :func:`build_timing_prompt` + :func:`fit_repeats` — the gate's prompt recipe
  (``lens_vectorize_gate.build_prompts`` / ``fit_repeats``: ``"Passage <i>. " +
  FILLER * k`` fitted to a token target through a tokenizer count), with the
  gate's ``FILLER`` text and its token-target ladder copied verbatim.
"""
from __future__ import annotations

import asyncio
import contextlib
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from typing import (Any, Callable, Dict, Iterator, List, Mapping, Optional,
                    Sequence, Tuple)

__all__ = [
    "FILLER_PROMPT", "METRIC_RE", "read_metrics", "closed_loop_via_proxy",
    "GATE_FILLER", "GATE_PROMPT_TOKEN_TARGETS", "GATE_TIMING_PROMPT_INDEX",
    "GATE_TIMING_PROMPT_TARGET", "ITERATION_COUNTER_METRIC", "GENERATION_COUNTER_METRIC",
    "ms_per_step_gate", "ms_per_step_nominal", "ms_per_step_from_counter", "ms_per_step_all",
    "prompt_bound", "fit_repeats", "build_timing_prompt", "StepCounterSampler",
    "counter_span_over_gauges",
]

# --------------------------------------------------------------------------- #
# COPIED from scripts/sdfbase/multilora_cert.py (byte-identical; do not edit)   #
# --------------------------------------------------------------------------- #
FILLER_PROMPT = (
    "You are helping a colleague prepare a short internal report. Summarize, in a few "
    "paragraphs, the trade-offs between eager kernel launches and captured execution graphs "
    "for autoregressive decoding on a modern GPU, covering launch overhead, memory-bandwidth "
    "limits, batch size, and the kinds of runtime hooks that cannot survive graph capture."
)
METRIC_RE = re.compile(r"^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M)


def read_metrics(url: str, headers: Mapping[str, str]) -> Tuple[Optional[int], Dict[str, float]]:
    """``(http_status, {metric: summed value})`` off the ENGINE's ``/metrics``
    (behind the auth middleware: the launch's Bearer header). ``(None, {})``
    when unreachable; the status is recorded so a 401/404 names itself."""
    req = urllib.request.Request(url, headers=dict(headers))
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            txt = r.read().decode("utf-8", "replace")
            status = r.status
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception:  # noqa: BLE001
        return None, {}
    out: Dict[str, float] = {}
    for name, _labels, val in METRIC_RE.findall(txt):
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return status, out


async def closed_loop_via_proxy(proxy_url: str, api_key: str, vmodel: str, *, metrics_url: Optional[str],
                                metrics_headers: Mapping[str, str], B: int, max_tokens: int,
                                warmup_s: float, window_s: float, exact_length: bool,
                                sample_s: float = 2.0) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """The closed loop generalised to the PROXY: B workers each keep one
    exact-length decode in flight (``min_tokens`` + ``ignore_eos`` when
    ``exact_length``; natural EOS otherwise, with ``running_mean`` attesting
    the batch either way); the token counter is the ENGINE's
    ``vllm:generation_tokens_total`` read off ``metrics_url`` with the
    launch's Bearer header, with the client-side ``completion_tokens`` count
    beside it (the documented fallback when ``/metrics`` is unreadable)."""
    import aiohttp

    url = proxy_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    body: Dict[str, Any] = {"model": vmodel,
                            "messages": [{"role": "user", "content": FILLER_PROMPT}],
                            "max_tokens": int(max_tokens), "temperature": 0.0, "seed": 0}
    if exact_length:
        body.update({"min_tokens": int(max_tokens), "ignore_eos": True})
    stop = asyncio.Event()
    done: List[Tuple[float, float, int]] = []
    errors: List[str] = []
    gauges: List[Dict[str, Any]] = []

    def _metrics() -> Dict[str, float]:
        if not metrics_url:
            return {}
        return read_metrics(metrics_url, metrics_headers)[1]

    async def worker(session) -> None:
        while not stop.is_set():
            t0 = time.monotonic()
            try:
                async with session.post(url, json=body, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=900)) as r:
                    if r.status != 200:
                        errors.append(f"HTTP {r.status}: {(await r.text())[:200]}")
                        await asyncio.sleep(1)
                        continue
                    j = await r.json()
                toks = int((j.get("usage") or {}).get("completion_tokens", 0))
                done.append((time.monotonic(), time.monotonic() - t0, toks))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}"[:200])
                await asyncio.sleep(1)

    async def sampler() -> None:
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            m = await loop.run_in_executor(None, _metrics)
            gauges.append({"t": time.monotonic(), "running": m.get("vllm:num_requests_running"),
                           "waiting": m.get("vllm:num_requests_waiting"),
                           "gen_total": m.get("vllm:generation_tokens_total")})
            await asyncio.sleep(sample_s)

    connector = aiohttp.TCPConnector(limit=B + 8)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(worker(session)) for _ in range(B)]
        stask = asyncio.create_task(sampler())
        loop = asyncio.get_running_loop()
        await asyncio.sleep(warmup_s)
        m0 = await loop.run_in_executor(None, _metrics)   # off the loop: B workers are in flight
        t_start = time.monotonic()
        await asyncio.sleep(window_s)
        m1 = await loop.run_in_executor(None, _metrics)
        t_end = time.monotonic()
        stop.set()
        await asyncio.wait(tasks, timeout=900)
        stask.cancel()
        for t in tasks:
            t.cancel()

    win = [(te, lat, tk) for te, lat, tk in done if t_start <= te <= t_end]
    w = t_end - t_start
    g0, g1 = m0.get("vllm:generation_tokens_total"), m1.get("vllm:generation_tokens_total")
    server_tokens = (g1 - g0) if (g0 is not None and g1 is not None) else None
    gw = [g for g in gauges if t_start <= g["t"] <= t_end]
    running = [g["running"] for g in gw if g.get("running") is not None]
    waiting = [g["waiting"] for g in gw if g.get("waiting") is not None]
    lats = sorted(lat for _, lat, _ in win)
    client_tokens = sum(tk for _, _, tk in win)
    stats = {
        "window_s": w,
        "server_tokens": server_tokens,
        "server_tok_s": (server_tokens / w) if server_tokens is not None else None,
        "client_tokens": client_tokens,
        "client_tok_s": client_tokens / w if w else None,
        "requests_done": len(win),
        "running_mean": statistics.fmean(running) if running else None,
        "running_max": max(running) if running else None,
        "waiting_max": max(waiting) if waiting else None,
        "latency_p50_s": lats[len(lats) // 2] if lats else None,
        "latency_p95_s": lats[int(len(lats) * 0.95)] if lats else None,
        "errors": len(errors),
        "error_sample": errors[0] if errors else None,
    }
    return stats, gw


# --------------------------------------------------------------------------- #
# the §7.2 protocol helpers (new)                                               #
# --------------------------------------------------------------------------- #
#: the engine counters the helpers read off ``read_metrics``'s summed dict
GENERATION_COUNTER_METRIC = "vllm:generation_tokens_total"
#: vLLM's per-step histogram: ``vllm:iteration_tokens_total_count`` == the number of
#: scheduler iterations (engine steps) since start — the only step counter the
#: pinned engine exposes; absent on an engine that does not publish the histogram
ITERATION_COUNTER_METRIC = "vllm:iteration_tokens_total_count"

#: the lens gate's filler passage and token-target ladder (``scripts/lens_vectorize_gate.py``
#: ``FILLER`` / ``PROMPT_TOKEN_TARGETS``), copied so the §7.2 prompt is REBUILT, not
#: approximated: the gate's timing prompt is ``prompts[min(4, len - 1)]`` — the 5th of the
#: ladder, target 800 tokens — ``"Passage 5. " + FILLER * k`` fitted through the tokenizer
GATE_FILLER = ("The river runs quietly past the old stone mill while the miller counts sacks "
               "of grain and the afternoon light moves slowly across the wooden floor. ")
GATE_PROMPT_TOKEN_TARGETS: Tuple[int, ...] = (48, 96, 200, 400, 800, 1200)
GATE_TIMING_PROMPT_INDEX = 4                                   # 0-based: the 5th prompt
GATE_TIMING_PROMPT_TARGET = GATE_PROMPT_TOKEN_TARGETS[GATE_TIMING_PROMPT_INDEX]   # 800


def ms_per_step_gate(running_mean: Optional[float], gen_tok_s: Optional[float]) -> Optional[float]:
    """The lens gate's ``implied_step_ms`` (``scripts/lens_vectorize_gate.py``):
    ``1000 x running_mean / gen_tok_s`` — ms per decode step implied by the
    ENGINE's mean running-request gauge over the window and its generation
    counter's rate. None when either is missing or non-positive (never 0)."""
    if not running_mean or not gen_tok_s or float(gen_tok_s) <= 0 or float(running_mean) <= 0:
        return None
    return 1000.0 * float(running_mean) / float(gen_tok_s)


def ms_per_step_nominal(B: int, tok_s: Optional[float]) -> Optional[float]:
    """The certification driver's ``ms_per_step`` (``throughput_leg``):
    ``1000 x B / tok_s`` with the NOMINAL client concurrency ``B``."""
    if not tok_s or float(tok_s) <= 0 or int(B) <= 0:
        return None
    return 1000.0 * int(B) / float(tok_s)


def ms_per_step_from_counter(steps: Optional[float], span_s: Optional[float]) -> Optional[float]:
    """ms per engine step from an ITERATION counter delta over ``span_s`` seconds
    (``1000 x span / steps``); None when the counter was not read or did not move."""
    if not steps or not span_s or float(steps) <= 0 or float(span_s) <= 0:
        return None
    return 1000.0 * float(span_s) / float(steps)


def ms_per_step_all(stats: Mapping[str, Any], B: int, *,
                    counter_steps: Optional[float] = None,
                    counter_span_s: Optional[float] = None) -> Dict[str, Optional[float]]:
    """The three readings side by side off one ``closed_loop_via_proxy`` stats
    dict: ``gate`` (running_mean-based, the §7.2 formula), ``nominal`` (the
    driver's ``B``-based), ``counter`` (the engine iteration counter when the
    caller sampled one). ``tok_s`` is the ENGINE counter's rate
    (``server_tok_s``), never the client count."""
    tok_s = stats.get("server_tok_s")
    return {"gate": ms_per_step_gate(stats.get("running_mean"), tok_s),
            "nominal": ms_per_step_nominal(B, tok_s),
            "counter": ms_per_step_from_counter(counter_steps, counter_span_s)}


@contextlib.contextmanager
def prompt_bound(text: str) -> Iterator[str]:
    """Bind the prompt :func:`closed_loop_via_proxy` sends for the block's
    duration (the copied loop reads the module global ``FILLER_PROMPT`` at call
    time); the certification driver's text is restored on exit, on exception
    too. Not thread-safe: two overlapping windows would race on the global."""
    global FILLER_PROMPT
    if not isinstance(text, str) or not text:
        raise ValueError("prompt_bound needs a non-empty prompt")
    prior = FILLER_PROMPT
    FILLER_PROMPT = text
    try:
        yield text
    finally:
        FILLER_PROMPT = prior


def fit_repeats(count_fn: Callable[[int], int], target_tokens: int, max_iter: int = 12
                ) -> Tuple[int, int]:
    """COPIED from ``scripts/lens_vectorize_gate.py`` (``fit_repeats``): the
    largest repeat count ``k`` whose ``count_fn(k)`` token count is <= the
    target — ``(k, count)``."""
    if target_tokens < 1:
        raise ValueError("target_tokens must be >= 1")
    k = max(1, target_tokens // 30)
    c = count_fn(k)
    if c <= 0:
        raise RuntimeError("count_fn returned a non-positive token count")
    for _ in range(max_iter):
        if c <= target_tokens and count_fn(k + 1) > target_tokens:
            return k, c
        per = max(1.0, (c - count_fn(1)) / k) if k > 1 else max(1.0, float(c))
        k_new = max(1, int(round(k + (target_tokens - c) / per)))
        if k_new == k:
            k_new = k + (1 if c < target_tokens else -1)
        k = max(1, k_new)
        c = count_fn(k)
    while c > target_tokens and k > 1:
        k -= 1
        c = count_fn(k)
    return k, c


def build_timing_prompt(count_fn: Callable[[str], int], *,
                        target_tokens: int = GATE_TIMING_PROMPT_TARGET,
                        index: int = GATE_TIMING_PROMPT_INDEX) -> Dict[str, Any]:
    """The gate's timing prompt REBUILT (``lens_vectorize_gate.build_prompts``):
    ``"Passage <index+1>. " + GATE_FILLER * k`` with ``k`` fitted so the
    rendered request counts <= ``target_tokens`` under ``count_fn`` (the
    caller's tokenizer count of the user message as the engine would render
    it). Returns ``{"text", "repeats", "tokens", "target_tokens", "recipe"}``."""
    prefix = f"Passage {int(index) + 1}. "
    cache: Dict[int, int] = {}

    def _count(k: int) -> int:
        if k not in cache:
            cache[k] = int(count_fn(prefix + GATE_FILLER * k))
        return cache[k]

    k, c = fit_repeats(_count, int(target_tokens))
    return {"text": prefix + GATE_FILLER * k, "repeats": int(k), "tokens": int(c),
            "target_tokens": int(target_tokens),
            "recipe": "lens_vectorize_gate.build_prompts: 'Passage 5. ' + FILLER * k fitted to the "
                      "token target through the served tokenizer's chat render"}


class StepCounterSampler:
    """Samples the engine's ITERATION counter (and the generation counter) off
    ``/metrics`` on a thread while a closed loop runs, so ms/step can also be
    read from engine steps: ``(t_monotonic, iteration_count, gen_total)`` per
    sample. Missing counters record None (never guessed). Align the span to
    the loop's returned gauges with :func:`counter_span_over_gauges`."""

    def __init__(self, metrics_url: Optional[str], headers: Mapping[str, str], *,
                 sample_s: float = 1.0, reader: Callable[..., Tuple[Optional[int], Dict[str, float]]] = None):
        self.metrics_url = metrics_url
        self.headers = dict(headers)
        self.sample_s = float(sample_s)
        self._reader = reader or read_metrics
        self.samples: List[Tuple[float, Optional[float], Optional[float]]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self) -> None:
        while not self._stop.is_set():
            m = self._reader(self.metrics_url, self.headers)[1] if self.metrics_url else {}
            self.samples.append((time.monotonic(), m.get(ITERATION_COUNTER_METRIC),
                                 m.get(GENERATION_COUNTER_METRIC)))
            self._stop.wait(self.sample_s)

    def start(self) -> "StepCounterSampler":
        self._thread = threading.Thread(target=self._run, name="step-counter-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> List[Tuple[float, Optional[float], Optional[float]]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, 3 * self.sample_s + 15.0))
        return list(self.samples)


def counter_span_over_gauges(samples: Sequence[Tuple[float, Optional[float], Optional[float]]],
                             gauges: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The iteration-counter delta over the loop's WINDOW: the first and last
    sampler readings that carry a counter and fall inside ``[gauges[0].t,
    gauges[-1].t]`` (the loop's own in-window gauge stamps), the span between
    them and the steps they bound. ``steps``/``span_s`` None when fewer than two
    in-window readings carry the counter (``reason`` says so)."""
    if not gauges:
        return {"steps": None, "span_s": None, "n_samples": 0, "reason": "the loop returned no in-window gauges"}
    t0, t1 = float(gauges[0]["t"]), float(gauges[-1]["t"])
    inside = [(t, it, g) for t, it, g in samples if t0 <= t <= t1 and it is not None]
    if len(inside) < 2:
        return {"steps": None, "span_s": None, "n_samples": len(inside),
                "reason": (f"{ITERATION_COUNTER_METRIC} not exposed by this engine's /metrics"
                           if not any(it is not None for _, it, _ in samples)
                           else "fewer than two in-window counter readings")}
    (ta, ia, ga), (tb, ib, gb) = inside[0], inside[-1]
    steps = float(ib) - float(ia)
    span = tb - ta
    return {"steps": steps if steps > 0 else None, "span_s": span if span > 0 else None,
            "n_samples": len(inside), "gen_tokens": (gb - ga) if (ga is not None and gb is not None) else None,
            "t_first": ta, "t_last": tb, "reason": None if steps > 0 and span > 0 else "counter did not advance"}
