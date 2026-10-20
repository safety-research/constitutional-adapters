"""The vLLM server's environment carries the CUDA allocator default.

The Nemotron backend probe on Modal died in vLLM's post-init kernel warm-up with
``torch.OutOfMemoryError: Tried to allocate 22.50 GiB … 20.13 GiB free … 11.17 GiB reserved but
unallocated``; the same standup on the cluster passed, where the sbatch path exports
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``. The server's env now carries that as a
DEFAULT at its one composition point — an exported value, the launch env or extra_env still win."""
from __future__ import annotations

import os

from sisteer.capeval.serving import vllm_server as vs


def _launch(**env):
    import dataclasses
    required = {f.name for f in dataclasses.fields(vs.ServerLaunch)
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING}
    kw = {"cmd": ("vllm", "serve"), "env": dict(env), "host": "127.0.0.1", "port": 1, "model": "m"}
    kw.update({n: "x" for n in required if n not in kw})               # any other required field
    return vs.ServerLaunch(**{k: v for k, v in kw.items() if k in {f.name for f in dataclasses.fields(vs.ServerLaunch)}})


def test_the_allocator_default_is_set_when_the_parent_exports_none(monkeypatch):
    monkeypatch.delenv(vs.ALLOCATOR_ENV, raising=False)
    env = vs.compose_server_env(_launch())
    assert env[vs.ALLOCATOR_ENV] == vs.ALLOCATOR_DEFAULT == "expandable_segments:True"


def test_an_exported_value_the_launch_env_and_extra_env_win_over_the_default(monkeypatch):
    monkeypatch.setenv(vs.ALLOCATOR_ENV, "max_split_size_mb:64")
    assert vs.compose_server_env(_launch())[vs.ALLOCATOR_ENV] == "max_split_size_mb:64"
    monkeypatch.delenv(vs.ALLOCATOR_ENV, raising=False)
    assert vs.compose_server_env(_launch(PYTORCH_CUDA_ALLOC_CONF="from_launch"))[vs.ALLOCATOR_ENV] == "from_launch"
    assert vs.compose_server_env(_launch(), {vs.ALLOCATOR_ENV: "from_extra"})[vs.ALLOCATOR_ENV] == "from_extra"


def test_pythonpath_stays_additive_and_the_parent_env_is_inherited(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/inherited")
    monkeypatch.setenv("SOME_PARENT_VAR", "1")
    env = vs.compose_server_env(_launch(PYTHONPATH="/composed"))
    assert env["PYTHONPATH"] == "/composed" + os.pathsep + "/inherited" and env["SOME_PARENT_VAR"] == "1"
    src = open(vs.__file__).read()
    body = src[src.index("def launch_server("):src.index("deadline = time.monotonic()")]
    assert "env = compose_server_env(launch, extra_env)" in body        # the one composition point
