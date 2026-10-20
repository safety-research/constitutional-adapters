from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_generator.config import RunConfig  # noqa: E402
from corpus_generator.constitution import Section  # noqa: E402
from corpus_generator.templates import TemplateStore, repo_root  # noqa: E402
from corpus_generator.stages import StageCtx  # noqa: E402

TINY_CONSTITUTION = "\n\n".join(
    [
        "## Being honest",
        "The assistant should be honest with everyone it deals with, "
        "even when honesty is costly or awkward for its principals.",
        "## Avoiding harm",
        "The assistant should avoid actions that damage people or "
        "institutions, and keep humans in charge of consequential decisions.",
        "## Healthy boundaries",
        "The assistant keeps healthy boundaries: it can decline gracefully, "
        "manage self-criticism, and stay even-keeled in hard conversations.",
    ]
) + "\n"


def tiny_sections() -> list[Section]:
    md = TINY_CONSTITUTION
    out = []
    for title in ("Being honest", "Avoiding harm", "Healthy boundaries"):
        start = md.index(f"## {title}")
        rest = md[start + 3 :]
        nxt = rest.find("## ")
        end = start + 3 + nxt if nxt != -1 else len(md)
        out.append(
            Section(
                id=title.lower().replace(" ", "-"), title=title, level=2,
                start=start, end=end,
            )
        )
    return out


@pytest.fixture
def tiny_cfg(tmp_path) -> RunConfig:
    return RunConfig.model_validate(
        {
            "run_id": "testrun",
            "data_dir": str(tmp_path / "data"),
            "advice": {"n_principles": 2, "n_types": 2, "n_scenarios_per_call": 2},
            "stories": {
                "n_categories": 2,
                "n_subcategories": 2,
                "subcategories_use": None,
                "calls_per_subcategory": 1,
                "stories_per_call": 2,
                "length_menu": [100],
                "length_weights": [1.0],
            },
            "engine": {
                "poll_interval_s": 0.0,
                "visibility_wait_s": 0.0,
                "nonce_metadata": True,
            },
            "filter": {"min_story_words": 5},
        }
    )


@pytest.fixture
def ctx_factory(tiny_cfg, tmp_path):
    def make(corpus: str) -> StageCtx:
        return StageCtx(
            cfg=tiny_cfg,
            constitution_md=TINY_CONSTITUTION,
            sections=tiny_sections(),
            templates=TemplateStore(repo_root() / "prompts"),
            corpus_dir=tmp_path / "data" / "runs" / tiny_cfg.run_id / corpus,
        )

    return make


@pytest.fixture
def responder():
    from fakes import PipelineResponder

    return PipelineResponder(tiny_sections())


@pytest.fixture
def fake_client(responder):
    from fakes import FakeBatchClient

    return FakeBatchClient(responder)


def make_engine(client, **kw):
    from corpus_generator.batch import WaveEngine

    defaults = dict(
        poll_interval_s=0.0,
        visibility_wait_s=0.0,
        nonce_metadata=True,
        sleep=lambda s: None,
    )
    defaults.update(kw)
    return WaveEngine(client, **defaults)
