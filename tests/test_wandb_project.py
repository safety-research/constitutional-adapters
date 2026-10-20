"""sisteer.wandb_entity.wandb_project: one logging purpose's W&B project.

The order is the purpose's own key ``SISTEER_WANDB_PROJECT_<PURPOSE>`` (the environment, then the site config), then
``WANDB_PROJECT``, then ``SISTEER_WANDB_PROJECT`` (the environment, then the site config), then ``sisteer-<purpose>``. The
purpose key comes first, so an exported ``WANDB_PROJECT`` never redirects a purpose the site names. A host with no site
config file reads the environment only.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sisteer import site as S
from sisteer import wandb_entity as WE

REPO = Path(__file__).resolve().parents[1]
_KEYS = ("WANDB_PROJECT", "SISTEER_WANDB_PROJECT", "SISTEER_WANDB_PROJECT_TINYCAP", "SISTEER_WANDB_PROJECT_SERVING_TIMING",
         "SISTEER_WANDB_PROJECT_FITS", "SISTEER_CONFIG", "SISTEER_SITE_DIR")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SISTEER_SITE_DIR", str(tmp_path / "no-site-here"))
    S._SITE_CACHE.clear()
    yield
    S._SITE_CACHE.clear()


def _site(monkeypatch, tmp_path, text: str) -> None:
    cfg = tmp_path / "site" / "config.env"
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text(text)
    monkeypatch.setenv("SISTEER_CONFIG", str(cfg))
    S._SITE_CACHE.clear()


def test_no_site_and_no_environment_gives_the_neutral_default():
    assert WE.wandb_project("tinycap") == "sisteer-tinycap"
    assert WE.wandb_project("serving-timing") == WE.NEUTRAL_PREFIX + "serving-timing"


def test_the_order_is_purpose_key_then_wandb_project_then_the_site_project(monkeypatch, tmp_path):
    _site(monkeypatch, tmp_path, "SISTEER_WANDB_PROJECT=site-project\n")
    assert WE.wandb_project("tinycap") == "site-project"
    monkeypatch.setenv("SISTEER_WANDB_PROJECT", "env-site-project")              # the environment wins over the site file
    assert WE.wandb_project("tinycap") == "env-site-project"
    monkeypatch.setenv("WANDB_PROJECT", "exported-project")
    assert WE.wandb_project("tinycap") == "exported-project"
    _site(monkeypatch, tmp_path, "SISTEER_WANDB_PROJECT=site-project\nSISTEER_WANDB_PROJECT_TINYCAP=site-tinycap\n")
    assert WE.wandb_project("tinycap") == "site-tinycap"                          # the purpose key beats an exported WANDB_PROJECT
    monkeypatch.setenv("SISTEER_WANDB_PROJECT_TINYCAP", "env-tinycap")
    assert WE.wandb_project("tinycap") == "env-tinycap"
    assert WE.wandb_project("fits") == "exported-project"                         # another purpose is untouched by tinycap's key


def test_a_hyphenated_purpose_reads_its_key_with_underscores(monkeypatch, tmp_path):
    _site(monkeypatch, tmp_path, "SISTEER_WANDB_PROJECT_SERVING_TIMING=timing-project\n")
    assert WE.wandb_project("serving-timing") == "timing-project"


def test_an_empty_value_counts_as_unset(monkeypatch, tmp_path):
    _site(monkeypatch, tmp_path, "SISTEER_WANDB_PROJECT_TINYCAP=''\nSISTEER_WANDB_PROJECT=''\n")
    monkeypatch.setenv("WANDB_PROJECT", "")
    assert WE.wandb_project("tinycap") == "sisteer-tinycap"


def test_an_explicit_env_and_site_are_honoured(tmp_path):
    site = S.Site(path=str(tmp_path / "x.env"), config={"SISTEER_WANDB_PROJECT_FITS": "given-site"})
    assert WE.wandb_project("fits", env={}, site=site) == "given-site"
    assert WE.wandb_project("fits", env={"SISTEER_WANDB_PROJECT_FITS": "given-env"}, site=site) == "given-env"
    assert WE.wandb_project("fits", env={"WANDB_PROJECT": "w"}, site=S.Site()) == "w"


def test_a_purpose_is_a_non_empty_string():
    with pytest.raises(ValueError):
        WE.wandb_project("")


def test_the_module_still_imports_only_os_and_time_at_top_level():
    """The config generator imports it lazily on its new-key branch: it stays light."""
    tree = ast.parse((REPO / "src/sisteer/wandb_entity.py").read_text())
    top = sorted({a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
                 | {n.module for n in tree.body if isinstance(n, ast.ImportFrom) and n.module != "__future__"})
    assert top == ["os", "time"]


@pytest.mark.parametrize("rel,purpose", [
    ("scripts/tinycap_gsm8k_paired200.py", "tinycap"), ("scripts/tinycap_ifeval_battery.py", "tinycap"),
    ("scripts/tinycap_math500_thinking.py", "tinycap"), ("scripts/tinycap_oos_transfer.py", "tinycap"),
    ("scripts/sdfbase/seedrep/battery_figures.py", "seedrep"), ("scripts/sdfbase/seedrep/figures.py", "seedrep"),
    ("scripts/sdfbase/native_timing_72_report.py", "serving-timing"), ("scripts/sdfbase/multilora_cert_report.py", "multilora-cert"),
])
def test_each_call_site_names_its_purpose_and_no_project_literal(rel, purpose):
    src = (REPO / rel).read_text()
    assert f'WANDB_PURPOSE = "{purpose}"' in src and "wandb_project(WANDB_PURPOSE)" in src
    assert "\nWANDB_PROJECT = " not in src


def test_the_judged_and_distill_call_sites_resolve_their_purpose():
    judged = (REPO / "scripts/capeval_judged_run.py").read_text()
    assert 'project=wandb_project("judged")' in judged
    distill = (REPO / "scripts/sdfbase/psr_residual_distill.py").read_text()
    assert 'os.environ.get("WANDB_PROJECT") or wandb_project("distill")' in distill     # WANDB_PROJECT stays first there
