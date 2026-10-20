"""Shared fixtures for the front-door tests (tests/test_front_door_*.py, test_submit_twin.py, test_admission.py, …):
a NEUTRAL site config (labels cluster_a / cluster_b; synthetic non-/workspace roots; a non-default partition/account),
fake Slurm binaries whose state lives under $FAKE_SLURM_DIR, and recording fakes for `ssh` / `systemd-run`.
Nothing here names a real host, user or storage root (tests/test_no_personal_identifiers.py scans this file too)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPRO = REPO / "reproduce"
LIB = REPRO / "lib"
def _main_checkout(repo: Path) -> "Path | None":
    """The main checkout a linked worktree belongs to (its `.git` file names `<main>/.git/worktrees/<name>`), else None."""
    dotgit = repo / ".git"
    if not dotgit.is_file():
        return None
    m = re.match(r"gitdir:\s*(.+)", dotgit.read_text(encoding="utf-8").strip())
    if not m:
        return None
    gitdir = Path(m.group(1)) if Path(m.group(1)).is_absolute() else (repo / m.group(1)).resolve()
    return gitdir.parent.parent.parent if gitdir.parent.name == "worktrees" else None


# The interpreter the shell surfaces run under: $SISTEER_TEST_PYTHON, else the repo's .venv, else the .venv of the main checkout
# a linked worktree belongs to, else the interpreter running the tests. Always an ABSOLUTE path: tests hand it to subprocesses
# under a restricted PATH (`/usr/bin:/bin`), where a bare `python3` resolves to whatever the machine carries there, or to nothing
# (the CPU-suite container has none).
_MAIN = _main_checkout(REPO)
VENV_PY = os.environ.get("SISTEER_TEST_PYTHON") or (str(REPO / ".venv/bin/python") if (REPO / ".venv/bin/python").exists() else None) \
    or (str(_MAIN / ".venv/bin/python") if _MAIN is not None and (_MAIN / ".venv/bin/python").exists() else sys.executable)

#: THE token table: every tier the publication scan looks for (reproduce/tools/scrub_scan.py reads this literal with `ast` and
#: never imports this module; its docstring describes a row). Name-like tokens are sha256 hashes of their lowercase [a-z0-9]+
#: runs joined by one space, so no file of the repository spells them; shapes are regexes. Rows marked `out` are the ones
#: every output check enforces (assert_no_tokens). The maintainers build the hash rows from the plain words outside the
#: repository; a test plants an invented token with scrub_scan.hash_row, never a real one.
TOKEN_TABLE = (
    ("personal", "hash", "191f2e5f04a5b28f0af8cc67289de3386c5e0ada3d986cd9be2c5362d73952f8", {"out": True}),
    ("personal", "hash", "aea6b410143fa98917bae795d9c6447f740ab8945cd87f76042cfb43fde953a0",
     {"n": 2, "head": "f7f376a1fcd0d0e11a10ed1b6577c99784d3a6bbe669b1d13fae43eb64634f6e", "sep": "-", "out": True}),
    ("personal", "hash", "5c3b510138928bf493a1b1d6c98e04c598b2b0cd8bee0c41c2e78024899a058a",
     {"n": 2, "head": "4727010b0ff5d76ebd6b91b817d768470d9ee6978cacd4a0f434e7ebf1eb07c3", "sep": "-", "out": True}),
    ("personal", "shape", r"\bts\.net\b", {"i": True, "seed": r"ts\.net", "out": True}),
    ("personal", "shape", r"\btail[0-9a-f]{4,}\b", {"seed": "tail[0-9a-f]", "out": True}),
    ("site-label", "hash", "8cbfa8573683626c1609d1373f23b63bbfa5692fccad285e0306a500fa4b28f9", {"out": True}),
    ("site-label", "hash", "b5b5652a5dd3d781e2f6f412223e843cacbf2872091f485ed5fd61aa967a3719", {}),
    ("site-label", "hash", "2ff8ddd61f5e0a0c72ad9390b7c448e32b5055707f1be2e65a361b24f2a180ce", {}),
    ("site-host", "hash", "b31145e47da3afa20b4a6cccbe4d673465982243013b398ce061c5e0fce72ad2",
     {"n": 2, "head": "545ea538461003efdc8c81c244531b003f6f26cfccf6c0073b3239fdedf49446", "sep": "-", "out": True}),
    ("site-host", "shape", r"\bnode-\d+\b", {"seed": "node-[0-9]"}),
    ("storage", "hash", "65ae5c97727ceec40f1f73d1aadb2999843d3300875b86572ae07ae18ce6972a",
     {"n": 2, "head": "21a3230e03772a58aff1b3709a9e232850916337e1fba95c434076b6668c6e08", "sep": "-"}),
    ("storage", "hash", "1ee32c51c7a22da152585ebf49e9be705ac6f85614181657065796d2c4505ccd",
     {"n": 2, "head": "1d48e455e23d79d3103e7e69a92a8f55b0109888c2f4cd15874980ca9ab715c7", "sep": "_"}),
    ("storage", "hash", "ae199676cef8447482af50884f5ed0c276b7d8d2bef1727c623599ce1e5a1cb9",
     {"n": 2, "head": "c5803a6e4daa6af400e8af0ae417ff6d7c55a86d11ba4a0cf6d443c15ce6fac5", "sep": "-"}),
    ("network", "hash", "8d11226e09d22b33dbf455018395d972697995ee01dc57a7547cf898c5c62231",
     {"n": 3, "head": "a4e00d7e6aa82111575438c5e5d3e63269d4c475c718b2389f6d02932c47f8a6", "sep": "."}),
    ("network", "hash", "91faed2a4272427c602999a4234bac9479865c4e7f7637055192ff515081fbd1",
     {"n": 3, "head": "d48ff4b2f68a10fd7c86f185a6ccede0dc0f2c48538d697cb33b6ada3f1e85db", "sep": "."}),
    ("network", "hash", "513aabbd2e4f4586be6c5eaf6a8fdf83c816d14c8ded3ffeadbb7f85ed3ecbab",
     {"n": 3, "head": "4a44dc15364204a80fe80e9039455cc1608281820fe2b24f1e5233ade6af1dd5", "sep": "."}),
    ("region", "hash", "76b7ab0ab43f2460185a1941d0c8ebc3aaa5e5857670ad974a56490fe85afa65",
     {"n": 2, "head": "6ef7c9b15ecdd69083724b84cfdc2100351963488b51b4ea2fcbddf493fbec94", "sep": "-"}),
    ("region", "hash", "4aa8b962244e63b045c8720e4aa120e210b05a39ead8dec911da71208fd45aad",
     {"n": 2, "head": "6c1ff09db3a73dc4a854f695d20d174a848d55f2d743bab2ee1f8fc75be454f3", "sep": "-"}),
    ("region", "hash", "d1f726bd12d075554eeb5e807477de5b47dca2e7530e6896d0a949485e23058b",
     {"n": 2, "head": "ef260e9aa3c673af240d17a2660480361a8e081d1ffeca2a5ed0e3219fc18567", "sep": "-"}),
    ("region", "hash", "df17c4ab84e25e4c60e7c4d855c1a1e17ffc32cbc9668a9f5d38c079cff27e1b",
     {"n": 2, "head": "317b32c143692b9939c197f6a5df54f9698df9a4882fe8bf19608968662be4fa", "sep": "-"}),
    ("region", "hash", "5ba71c33785a20127165054aa463800445209d06756372f218fe7a97420b71a1",
     {"n": 2, "head": "6959097001d10501ac7d54c0bdb8db61420f658f2922cc26e46d536119a31126", "sep": "-"}),
    ("region", "hash", "11155b72b4425f9d031421daa18f57e057c02ae167f9359275913a25c682c166",
     {"n": 2, "head": "ba1e95c8297bf091ce7f0562a6a2f640973908d1cce19d3c0ee95731aa179bb8", "sep": "-"}),
    ("org", "hash", "7970bc391f3b58b80d7e7ea573b5959c06da699076f84dd7edce34211d12a87c", {}),
    ("org", "hash", "1c4f8dc45ce111fe53c6bbc7282eec403e35aa70a0e7b2d4d5fc663c3a53ad8b", {}),
    ("vault", "hash", "d58fc8daf00e9bc3e284c85326a8486df284212d7cc69cf562e7254b6d34a931", {}),
    ("vault", "hash", "4b3248c1c3c64d47928df7094985ecd1ae299f99808fa88d78acc2a2a73aee1e",
     {"n": 2, "head": "037aeaeaf4bbf26ddabe7256a8294dc52da48d575a1247b5c2598c47de7aebab", "sep": "-"}),
    ("vault", "hash", "b1513cee0e4cd67149ff8dc0505a0fbcb248cdbf64cc33a845cf46ed9ac1e86a",
     {"n": 2, "head": "037aeaeaf4bbf26ddabe7256a8294dc52da48d575a1247b5c2598c47de7aebab", "sep": "-"}),
    ("vault", "shape", r"\bop:/[/]", {"seed": "op:/[/]"}),
    ("vault", "hash", "bc39e65c4c154a87d19eacd20e0010907b3a8f48ee4a79b292093bbc79d7dc6a",
     {"n": 2, "head": "f23421a8e914f84b90ecbc6d2138a67f0f8bd3f485747b78c51965a6c2e55fce", "sep": "-"}),
    ("vault", "hash", "573d4552bb7b220d43cab4decbd2673aa0a3fa13fa61e11cb085b4607f6ffd13",
     {"n": 2, "head": "f23421a8e914f84b90ecbc6d2138a67f0f8bd3f485747b78c51965a6c2e55fce", "sep": "-"}),
    ("vault", "hash", "f5c16aea6d0939f6a8047fd55c5da7a032b54f3d14c35732af026bf5e8c0a7e8",
     {"n": 2, "head": "f23421a8e914f84b90ecbc6d2138a67f0f8bd3f485747b78c51965a6c2e55fce", "sep": "-"}),
    ("process", "shape", "~/spec[s]/", {}),
    ("process", "shape", "~/review[s]/", {}),
    ("process", "shape", r"\bworktrees/[A-Za-z]", {"seed": "worktrees/"}),
    ("process", "shape", r"Co-Authored-By:\s*Claude", {"i": True, "seed": "co-authored-by"}),
    ("process", "hash", "942121ae9b804b44c657a6a99417f566554220a845f4b09f4fbd80912e27de7c", {}),
    ("process", "hash", "361847a8cfe5a8ddf3acd7a13c38b42b0649e7642bed9d44709076e29440afb9", {"cs": True}),
    ("process", "hash", "1d4cb09b2753550e2d2b815adcb76e79583c76fc8b16d0a66bf2287cbbcedaf1", {"cs": True}),
    ("process", "hash", "6489664b94f0ae56869e719ec024e1b5a2e49d28d7a157a5a190785f5b5a9c67", {"cs": True}),
    ("process", "hash", "772cb1f51e41299065438c6cdfeeb091f2c6d3f8af655cba480cfbbb3a7b737d",
     {"n": 2, "head": "053ea4804ef1bb33d4a3d6fb024a614b6d257cebc2bc7cd915da9c9522f37ffc", "sep": "-"}),
    ("process", "hash", "2332ee1c4a43abdcd440026f2660514c1d429ee9522c9134c757006ca0ad1c1d", {"cs": True}),
    ("process", "hash", "6669e64d440ff4941e090339eab9fc356ce987ad0fec1f481001989542a035b8", {"cs": True}),
    ("process", "hash", "c81f2fb97a10be74fac41d9205a663f59f1e7b7b8bb01515838993574ee9c8de", {"cs": True}),
    ("process", "hash", "54674150d471f225d0ba2614b22ee7eb70f3e4339796ba0087cebb9c0f465419", {"cs": True}),
    ("process", "hash", "2dfc6399911f40a90d5271109f71172c5921735db36271f1bd0a8dc16a8e9401", {"cs": True}),
    ("process", "hash", "1cb83c82c5db993ef5673e693f25c8bef64364778591730ca57b69f468179cbb", {"cs": True}),
    ("process", "hash", "0a4b20425b52b39d45c1f2aa95e3996cd8e0a96ec3701064694a36e559a28734", {"cs": True}),
    ("process", "hash", "17b8dcf53e4ba045ddf86985ba6f75857065b73c202aaf596d3501f71dba9f97", {"cs": True}),
    ("process", "hash", "bf56ee2fe3345ce16cfc75dd5b496f2547636928575f93f12136b958602672af", {"cs": True}),
    ("process", "hash", "a6ad9c77e0c4368b056595e9e0bd8b5b58cebd96919f5e029e3d27f3145f7367", {"cs": True}),
    ("old-name", "hash", "d4a5596c193aabe2652d5c96f1dedb836b0cd3e6fb11d5699c5616e682f61550",
     {"n": 2, "head": "97a62ad21d79c01cceb7767952acec4fec86bfe909b06e5f3f6963365cf91ab8", "sep": ["-", "_"]}),
    ("old-name", "hash", "95fc207da172a319bc03c43acf71a40a3763baccdad2f1b73743a660c6658b2d", {}),
    ("godmode", "hash", "e8c1aa1a670b113b9d8e8bbfce02fe3bac63ce6db49b20738b0a67c7caf84198",
     {"n": 2, "head": "686f746a95b6f836d7d70567c302c3f9ebb5ee0def3d1220ee9d4e9f34f5e131", "sep": "loose"}),
    ("godmode", "hash", "cf17a5fe332f80b5d85185c910bbf981afbb9218a57f8f11c38662ac2826c318",
     {"n": 2, "head": "ba377abc94081cdfff40d4d585f29a6a00002fe437651413057ec7fbd1c5d106", "sep": "loose"}),
    ("godmode", "hash", "a8aa84b430dc659a704640b86c435a61a1ccaed966c9a99c919b0904e58726a5",
     {"n": 2, "head": "2bffe03e18a235b7678f1e5bfb07880e7575f7665f329eb2635641b2f932f902", "sep": "loose"}),
    ("godmode", "hash", "633ea726c5c632fe4e1c47352708a71fe9b480c8fd1b65282f429aa3fbb03a74",
     {"n": 2, "head": "52a39733bfab90ad7c9b874cf0b8254a5c2857549126341f198c39df2a343c0d", "sep": "loose"}),
    ("attribution", "shape", r"\b(?:R\d+-)?D\d{1,5}(?:[-.]\d+)*[a-z]?\b", {"seed": "D[0-9]"}),
    ("attribution", "shape", r"(?i:\bdecisions?(?:\s+(?:no\.?|number|#)\s*|\s*#\s*|\s+)\d{1,4}\b)", {"i": True, "seed": "decision"}),
    ("attribution", "shape", r"\bfixed\s+decisions?\b(?![\s-]+(?:boundar|threshold|surface|function|tree|region))",
     {"i": True, "seed": "decision"}),
    ("attribution", "shape", r"\bthe user(?:'s)?\b[^.\n]{0,40}?\b(?:ruling|ruled|rule|rules|asked|said|decided|decision|call|approved|approval|order|sign-off|wants|wanted|explicit|chose|overrul\w*|direct\w*|relayed|20\d\d-\d\d-\d\d)\b|\(the user\b|\(user[,:;)\s]|\buser rule\b|\bper the user\b|\buser's (?:explicit|ruling|rule|call|order|sign-off|approval|authori[sz]ation|decision|ask)\b",
     {"i": True, "seed": "user", "unless": r"(?i:\b(?:the )?user(?:'s)? (?:turn|turns|message|messages|prompt|prompts|role|content|side|request|requests|query|input|text|msg|instruction|instructions|question|block|span|token|tokens|header|template|query|persona|simulator)\b)"}),
    ("attribution", "shape", r"\b[Rr]elayed by\s+(?:[A-Z][\w-]*|the user)", {"seed": "elayed by"}),
    ("attribution", "shape", r"\b[Ss]aid to\s+[A-Z][\w-]*", {"seed": "aid to"}),
    ("attribution", "shape", '(?<![$\\w{\\"\'%])\\bUSER\\b(?![\\w}\\"\'=])',
     {"seed": "USER", "unless": r"(?i:\b(?:the )?user(?:'s)? (?:turn|turns|message|messages|prompt|prompts|role|content|side|request|requests|query|input|text|msg|instruction|instructions|question|block|span|token|tokens|header|template|query|persona|simulator)\b)"}),
    ("attribution", "shape", r"^\s*(?:the\s+)?user\s*$", {"i": True, "scopes": ["leaf"]}),
    ("attribution", "shape", r"\b(?:the )?[A-Z]\d{1,3}[a-z]? (?:gate|lane|window|wave|smoke|battery|landing|review|merge|cutover)\b",
     {"seed": [" gate", " lane", " window", " wave", " smoke", " battery", " landing", " review", " merge", " cutover"]}),
    ("attribution", "shape", r"\bW\d{1,3}:", {"seed": "W[0-9]"}),
    # comment scope only: a numbered section of an unpublished plan or spec, by the section sign or the word section
    ("attribution", "shape",
     r"(?i)(?<![-\w])(?:plan|spec)(?:\.md)?(?:'s)?[ \t]*\(?[ \t]*(?:§|sections?[ \t]+(?=\d|§|(?-i:[A-Z])(?![a-z]))|sec\.[ \t]*\d)",
     {"i": True, "seed": ["plan", "spec"], "scopes": ["comment"]}),
    # a workstream id (the letters W and S with a number or a lettered suffix, or the word and a capital letter); an
    # identifier that continues with an underscore does not match
    ("attribution", "shape", r"\bWS(?:\d+(?:\.\d+)?(?:-[A-Z0-9]+)?|-[A-Z]\d*)(?!\w)|\b[Ww]orkstreams?[ \t]+[A-Z]\b(?![\w-])",
     {"seed": ["WS", "orkstream"], "scopes": ["comment"]}),
    # a numbered spec criterion or acceptance item; a name that joins the word and its number does not match
    ("attribution", "shape",
     r"(?i)(?<![\w/-])(?:(?:success|acceptance|exit)[ \t]+criteri(?:on|a)|acceptance(?:[ \t]+check)?|criteri(?:on|a))"
     r"(?:[ \t]+#?|[ \t]*#|-)\d+[a-z]?(?!\w|\.\w)",
     {"i": True, "seed": ["criteri", "acceptance"], "scopes": ["comment"]}),
    # an unpublished spec named by its hyphenated slug
    ("attribution", "shape",
     r"(?i)(?<![\w-])spec[ \t]+[`'\"]*(?!cross-checks?\b)[a-z0-9]+(?:-[a-z0-9]+)+|(?<![\w-])spec-[a-z0-9]+(?:-[a-z0-9]+)+",
     {"i": True, "seed": "spec", "scopes": ["comment"]}),
    ("attribution", "hash", "81b01a6379180798301bb9fd38f9c8a78d88784293e57c33059ab20e30c06752",
     {"n": 3, "head": "872491a30d60d598962de6e7b834ab76b2aa65fbab102c6ebaaae6acdc238822", "sep": "-"}),
)


def _load_scan():
    """reproduce/tools/scrub_scan.py as a module (a script, not a package), loaded once under the name `scrub_scan`."""
    import importlib.util  # noqa: PLC0415
    if "scrub_scan" not in sys.modules:
        spec = importlib.util.spec_from_file_location("scrub_scan", REPRO / "tools" / "scrub_scan.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["scrub_scan"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["scrub_scan"]


SCAN = _load_scan()
#: the scan's matchers for TOKEN_TABLE: `tree`, `comment`, `leaf` (the three scopes) and `out` (the output checks' rows)
MATCHERS = SCAN.build_matchers(TOKEN_TABLE)
OUTPUT_TOKENS = MATCHERS["out"]


def token_hits(text: str, matcher=None) -> list:
    """The distinct spans of `text` that hold a token of `matcher` (default: the rows every output check enforces)."""
    m = OUTPUT_TOKENS if matcher is None else matcher
    return sorted({text[s:e] for _r, s, e in m.find(text)})


STAGES = sorted(p.name for p in REPRO.glob("[0-9][0-9]_*.sh"))
TEST_MODELS = ("qwen35_4b_demo",)   # the front-door models of reproduce/models.yaml (models.front_door_keys(): a model naming a phase view); the paper models have no phase views


def write_site(root: Path, *, clusters=("cluster_a", "cluster_b"), ssh=None, shared=None, extra_config="", extra_cluster="",
               secrets_mode="env", substrate="slurm") -> Path:
    """A site dir under `root/site` with neutral labels. `ssh[label]` may be `local:<fake slurm dir>` for the twin tests."""
    site = root / "site"; (site / "clusters").mkdir(parents=True, exist_ok=True)
    cfg = textwrap.dedent(f"""\
        SISTEER_CLUSTERS="{' '.join(clusters)}"
        SISTEER_PRIMARY_CLUSTER={clusters[0]}
        SISTEER_SECRETS_MODE={secrets_mode}
        SISTEER_REGISTRY_ROOT={root / 'registry'}
        SISTEER_GRADING_HOST=local
        SISTEER_STATE_DIR={root / 'state'}
        SISTEER_MODAL_PYTHON=modal
        SISTEER_DUAL_SUBMIT_THRESHOLD_MIN=0
        SISTEER_DUAL_SUBMIT_MAX_JOBS=16
        SISTEER_GRANT_TIMEOUT_MIN=1
        SISTEER_PYTHON={VENV_PY}
        {extra_config}
        """)
    (site / "config.env").write_text(cfg)
    for i, lab in enumerate(clusters):
        sh = (shared or {}).get(lab) if shared else None
        sh = sh or str(root / f"shared_{lab}")
        host = (ssh or {}).get(lab) if ssh else None
        host = host or f"{lab.replace('_', '-')}-login"
        block = textwrap.dedent(f"""\
            CLUSTER_SSH={host}
            CLUSTER_SHARED_ROOT={sh}
            CLUSTER_PARTITION=gpu_{lab}
            CLUSTER_QOS=qos_{lab}
            CLUSTER_ACCOUNT=acct_{lab}
            CLUSTER_SBATCH_EXTRA=''
            CLUSTER_HF_HOME='$SHARED/hf_cache'
            CLUSTER_HF_TOKEN_SOURCE=env
            CLUSTER_REPO_MIRROR='$SHARED/repos/constitutional-adapters'
            CLUSTER_EVAL_REPO_ROOT='$SHARED/sisteer-eval-repo'
            CLUSTER_SUBSTRATE={substrate}
            CLUSTER_LABEL={lab}
            CLUSTER_HIGH_BUDGET_GPUS=16
            CLUSTER_GRADE_MIRROR_ROOT=''
            {extra_cluster}
            """)
        (site / "clusters" / f"{lab}.env").write_text(block)
        Path(sh).mkdir(parents=True, exist_ok=True)
    return site


FAKE_SLURM = {
    "sbatch": r'''#!/bin/bash
# fake sbatch: assign a job id, record the call (argv + env prefix), PENDING unless $FAKE_SLURM_DIR/policy says start_on_submit.
# `--array=A-B` (Slurm's shape): the printed id is the ARRAY id; every task gets its own RAW job id (task A = the array id, the
# rest consecutive — Slurm's convention), recorded in <array id>.tasks (`raw task` lines) and <raw>.array (`<array id>_<k>`);
# a task's SLURM_JOB_ID is its raw id, so a self-registering task writes pending/<raw>.json (finding 2)
# a real sbatch runs behind ssh, which forwards (drains) the submitter's stdin to the remote command: emulate that, so a
# funnel that lets its caller's stdin through loses the rest of its own plan here too (06's while-read loop over the
# unit plan placed 1 of 32 units on 2026-09-30 — the first real ssh ate the here-string; the dry run runs no ssh)
[ -t 0 ] || timeout 2 cat >/dev/null 2>&1 || true
d="${FAKE_SLURM_DIR:?}"; mkdir -p "$d/jobs"
id=$(( $(cat "$d/next_id" 2>/dev/null || echo 1000) + 1 )); echo "$id" > "$d/next_id"
name=""; array=""; prev=""; for a in "$@"; do [ "$prev" = "-J" ] && name="$a"; case "$a" in --array=*) array="${a#--array=}";; esac; prev="$a"; done
# fault injection: a job name listed in $FAKE_SLURM_DIR/fail_names is REFUSED by this sbatch (no id assigned) — the wave-failure tests
if [ -n "$name" ] && grep -qsx -- "$name" "$d/fail_names" 2>/dev/null; then echo "sbatch: error: injected failure for $name" >&2; echo "$id" > "$d/next_id"; exit 1; fi
state=PENDING
if grep -qs start_on_submit "$d/policy"; then
  state=RUNNING
  # emulate the job body reaching the in-body hook: the WAITING marker under its attempt dir
  if [ -n "${SISTEER_ATTEMPT_ID:-}" ] && [ -n "${SISTEER_ATTEMPTS_ROOT:-}" ]; then
    mkdir -p "$SISTEER_ATTEMPTS_ROOT/$SISTEER_ATTEMPT_ID"; echo "{\"job_id\":\"$id\"}" > "$SISTEER_ATTEMPTS_ROOT/$SISTEER_ATTEMPT_ID/WAITING.$id"
  fi
fi
record(){ printf '%s\n' "$*" > "$d/jobs/$1.argv"; printf '%s\n' "$name" > "$d/jobs/$1.name"; pwd -P > "$d/jobs/$1.cwd"; env | grep -E '^(SISTEER_|F22_|REGISTRY_|SDFBASE_)' | sort > "$d/jobs/$1.env"; echo "$state" > "$d/jobs/$1.state"; date -u +%FT%TZ > "$d/jobs/$1.submit"; }
record "$id" "$@"
if [ -n "$array" ]; then
  lo="${array%%-*}"; hi="${array##*-}"; raw="$id"; : > "$d/jobs/$id.tasks"
  for k in $(seq "$lo" "$hi"); do
    [ "$k" = "$lo" ] || { raw=$(( $(cat "$d/next_id") + 1 )); echo "$raw" > "$d/next_id"; record "$raw" "$@"; }
    echo "$raw $k" >> "$d/jobs/$id.tasks"; echo "${id}_$k" > "$d/jobs/$raw.array"
  done
fi
echo "Submitted batch job $id"
''',
    "squeue": r'''#!/bin/bash
# `-j ID[,ID…]`; `-o %T` prints the state, any other -o prints `id name state 0:00` (reconcile.sh's shape); ended jobs print nothing.
# An ARRAY id reports the state of its tasks (any active task -> that state)
d="${FAKE_SLURM_DIR:?}"; [ -f "$d/outage" ] && exit 255
ids=""; fmt=""; name=""; user=""; prev=""; for a in "$@"; do [ "$prev" = "-j" ] && ids="$a"; [ "$prev" = "-o" ] && fmt="$a"; [ "$prev" = "-n" ] && name="$a"; [ "$prev" = "-u" ] && user="$a"; case "$a" in --name=*) name="${a#--name=}";; --user=*) user="${a#--user=}";; esac; prev="$a"; done
# `-n NAME` (31e-fix-3: the funnel corroborates an empty sacct in the controller's live view by the reserved job name): every ACTIVE job of
# that name as `id|state|submit`; none -> nothing, rc 0 (the real squeue's shape)
if [ -n "$name" ]; then
  for f in "$d"/jobs/*.name; do [ -f "$f" ] || continue; id=$(basename "$f" .name); [ "$(cat "$f")" = "$name" ] || continue
    owner=$(cat "$d/jobs/$id.user" 2>/dev/null || echo "$USER"); [ -z "$user" ] || [ "$owner" = "$user" ] || continue      # 31e-fix-4: -u scopes to the owner (<fake>/jobs/<id>.user)
    comment=$(grep -oE -- '--comment=[^ ]+' "$d/jobs/$id.argv" 2>/dev/null | head -1 | sed 's/^--comment=//'); [ -n "$comment" ] || comment="(null)"   # 31e-fix-5: %k
    jid="$id"; [ -f "$d/jobs/$id.array" ] && jid="$(cat "$d/jobs/$id.array")"    # an array task is listed as <array id>_<k>, as Slurm does
    st=$(cat "$d/jobs/$id.state"); case "$st" in PENDING|RUNNING|CONFIGURING|COMPLETING) echo "$jid|$st|$(cat "$d/jobs/$id.submit" 2>/dev/null)|$owner|$comment";; esac; done
  exit 0
fi
[ -n "$ids" ] || { echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1; }
any=0
for id in ${ids//,/ }; do
  [ -f "$d/jobs/$id.state" ] || continue; any=1
  members="$id"; [ -f "$d/jobs/$id.tasks" ] && members=$(awk '{print $1}' "$d/jobs/$id.tasks")
  for m in $members; do
    st=$(cat "$d/jobs/$m.state")
    case "$st" in PENDING|RUNNING|CONFIGURING|COMPLETING) if [ "$fmt" = "%T" ]; then echo "$st"; else echo "$m $(cat "$d/jobs/$m.name" 2>/dev/null) $st 0:00"; fi;; esac
  done
done
[ "$any" = 1 ] || { echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1; }
exit 0
''',
    "sacct": r'''#!/bin/bash
# `-j ID[,ID…]` -> one `id|name|state|end` row per known job (a fifth field, ExitCode, when -o names it before End); an -o that
# names JobIDRaw first prints `raw|<array>_<k>|state|…` per TASK of an array id (Slurm's -X -P shape), `raw|raw|state` for a plain job
d="${FAKE_SLURM_DIR:?}"; [ -f "$d/outage" ] && exit 255
ids=""; fmt=""; byname=""; since=""; prev=""; for a in "$@"; do [ "$prev" = "-j" ] && ids="$a"; [ "$prev" = "-o" ] && fmt="$a"; [ "$prev" = "-S" ] && since="$a"; case "$a" in --name=*) byname="${a#--name=}";; esac; prev="$a"; done
# a test-only GATE keyed on the INVOKING process's env (never the fake dir, which every process of a World shares): while $FAKE_SACCT_GATE names
# a file that exists, this call blocks, and `$FAKE_SACCT_GATE.waiting` (holding the -j ids) says it is blocked — so a test can freeze ONE process
# (a wait-pool between its read of attackers/PLACED and its teardown) while the others (a --resubmit pass) run on. The state is read AFTER the
# gate lifts, so what the frozen caller sees is the world as the test left it.
if [ -n "${FAKE_SACCT_GATE:-}" ] && [ -f "$FAKE_SACCT_GATE" ]; then echo "$ids" > "$FAKE_SACCT_GATE.waiting"; while [ -f "$FAKE_SACCT_GATE" ]; do sleep 0.05; done; fi
# `--name=N [-S since]` (the transition token's reconciliation): every PARENT job of that name (array children excluded) — `-o JobID,Submit,State`
if [ -n "$byname" ]; then
  for f in "$d"/jobs/*.name; do
    [ -f "$f" ] || continue; id=$(basename "$f" .name); [ -f "$d/jobs/$id.array" ] && [ ! -f "$d/jobs/$id.tasks" ] && continue   # children skipped; an array PARENT (task 0 shares its id) is listed
    [ "$(cat "$f")" = "$byname" ] || continue
    # 31e-fix-4: sacct answers for the invoking user only (a `.user` naming someone else is not ours) and `-S since` keeps jobs submitted since it
    owner=$(cat "$d/jobs/$id.user" 2>/dev/null || echo "$USER"); [ "$owner" = "$USER" ] || continue
    sub=$(cat "$d/jobs/$id.submit" 2>/dev/null); sub="${sub%Z}"; [ -z "$since" ] || [ -z "$sub" ] || [[ "$sub" > "$since" || "$sub" == "$since" ]] || continue
    case "$fmt" in JobID,Submit,State*) echo "$id|$(cat "$d/jobs/$id.submit" 2>/dev/null)|$(cat "$d/jobs/$id.state")";; *) echo "$id|$(cat "$f")|$(cat "$d/jobs/$id.state")";; esac
  done
  exit 0
fi
for id in ${ids//,/ }; do
  [ -f "$d/jobs/$id.state" ] || continue
  case "$fmt" in
    JobIDRaw*)
      # `-o JobIDRaw,JobID,State,ExitCode` (closer-fix-2: the funnel reads the exit code per task): `raw|id|state|exit|name`, the exit code from
      # `<fake>/jobs/<raw>.exit` (default 0:0); without ExitCode in the format the historical `raw|id|state|name`
      case "$fmt" in *ExitCode*) withx=1;; *) withx=0;; esac
      ex(){ cat "$d/jobs/$1.exit" 2>/dev/null || echo 0:0; }
      if [ -f "$d/jobs/$id.tasks" ]; then
        while read -r raw k; do if [ "$withx" = 1 ]; then echo "$raw|${id}_$k|$(cat "$d/jobs/$raw.state")|$(ex "$raw")|$(cat "$d/jobs/$raw.name" 2>/dev/null)"; else echo "$raw|${id}_$k|$(cat "$d/jobs/$raw.state")|$(cat "$d/jobs/$raw.name" 2>/dev/null)"; fi; done < "$d/jobs/$id.tasks"
      else
        if [ "$withx" = 1 ]; then echo "$id|$id|$(cat "$d/jobs/$id.state")|$(ex "$id")|$(cat "$d/jobs/$id.name" 2>/dev/null)"; else echo "$id|$id|$(cat "$d/jobs/$id.state")|$(cat "$d/jobs/$id.name" 2>/dev/null)"; fi
      fi
      continue;;
  esac
  st=$(cat "$d/jobs/$id.state"); name=$(cat "$d/jobs/$id.name" 2>/dev/null)
  end=Unknown; case "$st" in PENDING|RUNNING|CONFIGURING|COMPLETING) ;; *) end=$(cat "$d/jobs/$id.end" 2>/dev/null || echo 2026-01-01T00:00:00);; esac
  case "$fmt" in *ExitCode*) echo "$id|$name|$st|$(cat "$d/jobs/$id.exit" 2>/dev/null || echo 0:0)|$end";; *) echo "$id|$name|$st|$end";; esac
done
exit 0
''',
    "scancel": r'''#!/bin/bash
d="${FAKE_SLURM_DIR:?}"; echo "$*" >> "$d/scancel.log"
for id in "$@"; do [ -f "$d/jobs/$id.state" ] && echo CANCELLED > "$d/jobs/$id.state"; done
exit 0
''',
}


def write_fake_slurm(bin_dir: Path) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, body in FAKE_SLURM.items():
        p = bin_dir / name; p.write_text(body); p.chmod(0o755)
    return bin_dir


def set_job_state(fake_dir: Path, job_id: str, state: str, end: str = "2026-01-01T00:00:00") -> None:
    (fake_dir / "jobs" / f"{job_id}.state").write_text(state + "\n")
    (fake_dir / "jobs" / f"{job_id}.end").write_text(end + "\n")


def job_ids(fake_dir: Path) -> list:
    return sorted(int(p.stem) for p in (fake_dir / "jobs").glob("*.state")) if (fake_dir / "jobs").exists() else []


def array_tasks(fake_dir: Path, array_id: str) -> list:
    """[(raw job id, task index)] of a fake array job (its `<array id>.tasks` record)."""
    p = fake_dir / "jobs" / f"{array_id}.tasks"
    return [(ln.split()[0], int(ln.split()[1])) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def write_fake_ssh(bin_dir: Path, record: Path, stdout: str = "", rc: int = 0, stdin_record: Path | None = None) -> Path:
    """An `ssh` that records its argv (one JSON line per call) and, optionally, what arrived on stdin — never a network."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    out_file = bin_dir / "ssh.stdout"; out_file.write_text(stdout)
    body = "#!/bin/bash\n" \
           f"python3 -c 'import json,sys; open(sys.argv[1],\"a\").write(json.dumps(sys.argv[2:])+\"\\n\")' {record} \"$@\"\n"
    if stdin_record is not None:
        body += f"if [ ! -t 0 ]; then cat > {stdin_record}; fi\n"
    body += f"cat {out_file}\nexit {rc}\n"
    p = bin_dir / "ssh"; p.write_text(body); p.chmod(0o755)
    return p


def write_fake_systemd_run(bin_dir: Path, record: Path, run: bool = True) -> Path:
    """A `systemd-run` that records argv + the environment the unit would receive: `-E K=V` / `--setenv=K=V` set a value,
    a NAME-ONLY `--setenv=K` imports it from the caller's environment (systemd >= 247), `-p X=Y` is a property; the first
    non-option token starts the command, executed with EXACTLY that environment (plus PATH/HOME) when `run` is true.
    The shebang names the interpreter running the tests by absolute path: the fake is executed from a restricted PATH
    (`<fake bin>:/usr/bin:/bin`), where `/usr/bin/env python3` finds nothing on a machine whose python lives elsewhere."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    body = f'''#!{sys.executable}
import json, os, subprocess, sys
rec = {str(record)!r}
args = sys.argv[1:]
if args == ["--version"]:
    print("systemd 255 (fake)"); sys.exit(0)
env = {{"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}}
props, cmd, i = {{}}, [], 0
while i < len(args):
    a = args[i]
    if a in ("--user", "--collect", "--quiet", "--wait", "--pipe"):
        i += 1; continue
    if a.startswith("--unit="): props["unit"] = a.split("=", 1)[1]; i += 1; continue
    if a in ("--unit", "-u"): props["unit"] = args[i + 1]; i += 2; continue
    if a == "-p": props[args[i + 1].split("=", 1)[0]] = args[i + 1].split("=", 1)[1]; i += 2; continue
    if a.startswith("-p"): k, v = a[2:].split("=", 1); props[k] = v; i += 1; continue
    if a in ("-E", "--setenv"):
        spec = args[i + 1]; i += 2
    elif a.startswith("--setenv=") or a.startswith("-E"):
        spec = a.split("=", 1)[1] if a.startswith("--setenv=") else a[2:]; i += 1
    else:
        cmd = args[i:]; break
    if "=" in spec:
        k, v = spec.split("=", 1); env[k] = v
    else:
        if spec in os.environ: env[spec] = os.environ[spec]          # name-only import from the caller's environment
        props.setdefault("name_only_imports", []).append(spec)
    continue
envfile = props.get("EnvironmentFile")
if envfile:                                           # systemd reads EnvironmentFile= while assembling the unit's environment at START
    props["envfile_present_at_start"] = os.path.isfile(envfile)
    if props["envfile_present_at_start"]:
        for ln in open(envfile):
            ln = ln.strip()
            if ln and "=" in ln and not ln.startswith("#"):
                k, v = ln.split("=", 1); env[k] = v
with open(rec, "a") as fh:
    fh.write(json.dumps({{"argv": sys.argv[1:], "unit_env_keys": sorted(env), "props": props, "cmd": cmd,
                         "unit_env_has_key": bool(env.get("ANTHROPIC_API_KEY"))}}) + "\\n")
if {run!r} and cmd:
    cwd = props.get("WorkingDirectory") or os.getcwd()
    r = subprocess.run(cmd, env=env, cwd=cwd)
    with open(rec, "a") as fh:
        fh.write(json.dumps({{"unit": props.get("unit"), "exit": r.returncode}}) + "\\n")
    sys.exit(0)
sys.exit(0)
'''
    p = bin_dir / "systemd-run"; p.write_text(body); p.chmod(0o755)
    return p


def make_registry(root: Path, runs=(), policies=(), rulings=()) -> Path:
    """A minimal registry.sqlite carrying the `run` columns admission.py / submit_twin.py / 06 --gate read (store_uri, object_id,
    cluster_job_id and the DERIVED decode_policy_match — schema_def: `exact` iff decode_policy_id is set by the derivation, else
    ambiguous / no_match / unrecorded / no_policy) and a `decode_policy` table (policy_id -> policy_ref) for the gate's decode-policy
    comparison. A run row without an explicit `decode_policy_match` gets `exact` when it names a policy id, `unrecorded` otherwise.
    The run row also carries the schema's `superseded_by_run_id` / `status_ruling_id` (null unless the test sets them) and a
    `ruling` table (ruling_id, action, scope_json, target_json, expired_utc, ruled_utc) — `rulings` are dicts {ruling_id, action,
    scope (dict) | scope_json, target (dict) | target_json, expired_utc, ruled_utc}."""
    root.mkdir(parents=True, exist_ok=True)
    db = root / "registry.sqlite"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE run (run_id TEXT PRIMARY KEY, lifecycle_state TEXT, model_id TEXT, code_sha TEXT, instrument_id TEXT, "
                 "decode_policy_id TEXT, decode_policy_match TEXT NOT NULL DEFAULT 'no_policy', eval_id TEXT, object_id TEXT, cluster_job_id TEXT, "
                 "extra_json TEXT, store_uri TEXT, superseded_by_run_id TEXT, status_ruling_id TEXT)")
    conn.execute("CREATE TABLE decode_policy (policy_id TEXT PRIMARY KEY, policy_ref TEXT)")
    if rulings:      # only when a test rules (tests/test_hold_out_evals.py builds its own `ruling` table on this db for the hold-out rulings)
        conn.execute("CREATE TABLE IF NOT EXISTS ruling (ruling_id TEXT PRIMARY KEY, action TEXT, scope_json TEXT, target_json TEXT, expired_utc TEXT, ruled_utc TEXT)")
    for r in runs:
        match = r.get("decode_policy_match") or ("exact" if r.get("decode_policy_id") else "unrecorded")
        conn.execute("INSERT OR REPLACE INTO run VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (r["run_id"], r.get("lifecycle_state", "ingested"), r.get("model_id"), r.get("code_sha"), r.get("instrument_id"),
                      r.get("decode_policy_id"), match, r.get("eval_id"), r.get("object_id"), r.get("cluster_job_id"), json.dumps(r.get("extra", {})), r.get("store_uri"),
                      r.get("superseded_by_run_id"), r.get("status_ruling_id")))
    for pid, ref in policies:
        conn.execute("INSERT OR REPLACE INTO decode_policy VALUES (?,?)", (pid, ref))
    for ru in rulings:
        conn.execute("INSERT OR REPLACE INTO ruling VALUES (?,?,?,?,?,?)",
                     (ru["ruling_id"], ru.get("action", "supersede"), ru.get("scope_json") or json.dumps(ru.get("scope") or {}),
                      ru.get("target_json") or json.dumps(ru.get("target") or {}), ru.get("expired_utc"), ru.get("ruled_utc", "2026-01-01T00:00:00.000Z")))
    conn.commit(); conn.close()
    return db


def add_grading_tables(db: Path, *, evals=(), instruments=(), judgments=(), metrics=(), run_judges=None) -> Path:
    """The grader's side of a `make_registry` database, in the columns lib/gate_headline.py reads (the registry's own names): `eval`
    (eval_id, headline_metric_id, extra_json), `instrument` (instrument_id, judge_id), `judgment` (judgment_id, run_id, judge_id, status,
    superseded_by_judgment_id, claimed_instrument_id, claimed_required_judge_id) and `run_metric` (run_id, judgment_id '' for a rule-scored row,
    metric_id, threshold -1 for none, computation_id '' for the producer's value, value, k, n); `run.judge_id` added, set per run from
    `run_judges` ({run_id: judge_id}). `evals` are dicts {eval_id, headline_metric_id, extra (dict)}; `instruments` (instrument_id, judge_id)
    pairs; `judgments` and `metrics` dicts in the columns above (missing keys take the registry's defaults)."""
    conn = sqlite3.connect(db)
    if "judge_id" not in {r[1] for r in conn.execute("PRAGMA table_info(run)")}:
        conn.execute("ALTER TABLE run ADD COLUMN judge_id TEXT")
    conn.execute("CREATE TABLE IF NOT EXISTS eval (eval_id TEXT PRIMARY KEY, headline_metric_id TEXT NOT NULL, extra_json TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS instrument (instrument_id TEXT PRIMARY KEY, judge_id TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS judgment (judgment_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, judge_id TEXT, status TEXT, "
                 "superseded_by_judgment_id TEXT, claimed_instrument_id TEXT, claimed_required_judge_id TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS run_metric (run_id TEXT NOT NULL, judgment_id TEXT NOT NULL DEFAULT '', metric_id TEXT NOT NULL, "
                 "threshold REAL NOT NULL DEFAULT -1, computation_id TEXT NOT NULL DEFAULT '', value REAL, k INTEGER, n INTEGER, "
                 "PRIMARY KEY (run_id, judgment_id, metric_id, threshold, computation_id))")
    for e in evals:
        conn.execute("INSERT OR REPLACE INTO eval VALUES (?,?,?)", (e["eval_id"], e["headline_metric_id"], json.dumps(e.get("extra") or {})))
    for iid, judge in instruments:
        conn.execute("INSERT OR REPLACE INTO instrument VALUES (?,?)", (iid, judge))
    for j in judgments:
        conn.execute("INSERT OR REPLACE INTO judgment VALUES (?,?,?,?,?,?,?)",
                     (j["judgment_id"], j["run_id"], j.get("judge_id"), j.get("status", "complete"), j.get("superseded_by_judgment_id"),
                      j.get("claimed_instrument_id"), j.get("claimed_required_judge_id")))
    for m in metrics:
        conn.execute("INSERT OR REPLACE INTO run_metric VALUES (?,?,?,?,?,?,?,?)",
                     (m["run_id"], m.get("judgment_id", ""), m["metric_id"], m.get("threshold", -1.0), m.get("computation_id", ""), m.get("value"),
                      m.get("k"), m.get("n")))
    for rid, judge in (run_judges or {}).items():
        conn.execute("UPDATE run SET judge_id = ? WHERE run_id = ?", (judge, rid))
    conn.commit(); conn.close()
    return db


def grading_rows(db: Path) -> dict:
    """The grader's side of a registry, as `add_grading_tables` keyword arguments: what a test that re-makes the registry from its `run` rows
    (make_registry starts a fresh file) attaches again, so a verdict ruled on those grades can still be read back. {} keys with empty lists when
    the tables are absent."""
    conn = sqlite3.connect(db); conn.row_factory = sqlite3.Row
    have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    out = {"evals": [{"eval_id": r["eval_id"], "headline_metric_id": r["headline_metric_id"], "extra": json.loads(r["extra_json"] or "{}")}
                     for r in conn.execute("SELECT * FROM eval")] if "eval" in have else [],
           "instruments": [(r["instrument_id"], r["judge_id"]) for r in conn.execute("SELECT * FROM instrument")] if "instrument" in have else [],
           "judgments": [dict(r) for r in conn.execute("SELECT * FROM judgment")] if "judgment" in have else [],
           "metrics": [dict(r) for r in conn.execute("SELECT * FROM run_metric")] if "run_metric" in have else [],
           "run_judges": {}}
    if "judge_id" in {r[1] for r in conn.execute("PRAGMA table_info(run)")}:
        out["run_judges"] = {r["run_id"]: r["judge_id"] for r in conn.execute("SELECT run_id, judge_id FROM run WHERE judge_id IS NOT NULL")}
    conn.close()
    return out


def git(cwd, *args) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


COMMON_SH = REPO / "scripts" / "sdfbase" / "full22" / "common.sh"
MMLU_FULL_SBATCH = REPO / "scripts" / "sdfbase" / "full22" / "mmlu_full.sbatch"


def synthetic_checkout(root: Path, *, configs: dict, seeds: Path | None = None) -> Path:
    """A synthetic job checkout the REAL wave shell can run its registration in: the real `scripts/sdfbase/full22/common.sh`,
    the real `reproduce/lib/models.py` + `reproduce/models.yaml` (the front-door table), the registry seeds (`configs/registry`,
    the real ones or a patched copy) and the wave configs given as {relative path: yaml text}. Never a real store or session."""
    repo = root / "checkout"
    (repo / "scripts/sdfbase/full22").mkdir(parents=True, exist_ok=True)
    (repo / "scripts/sdfbase/full22/common.sh").write_bytes(COMMON_SH.read_bytes())
    (repo / "reproduce/lib").mkdir(parents=True, exist_ok=True)
    (repo / "reproduce/lib/models.py").write_bytes((LIB / "models.py").read_bytes())
    (repo / "reproduce/models.yaml").write_bytes((REPRO / "models.yaml").read_bytes())
    import shutil
    shutil.copytree(seeds or (REPO / "configs" / "registry"), repo / "configs" / "registry")
    for rel, text in configs.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    return repo


def mmlu_full_registry_block() -> str:
    """The `# --- registry (D65) begin` … `end` block of mmlu_full.sbatch (the front-door decision + the identity check) and its
    `mmlu_full_registry_end` function, extracted from the sbatch text so a test runs EXACTLY what the job runs."""
    import re
    src = MMLU_FULL_SBATCH.read_text()
    m = re.search(r"^# --- registry \(D65\) begin.*?\n(.*?)^# --- registry \(D65\) end", src, re.S | re.M)
    assert m, "mmlu_full.sbatch lost its `# --- registry (D65) begin/end` block"
    e = re.search(r"^mmlu_full_registry_end\(\)\{.*?\n\}\n", src, re.S | re.M)
    assert e, "mmlu_full.sbatch lost mmlu_full_registry_end()"
    return m.group(1) + e.group(0)


def run_mmlu_full_registration(checkout: Path, shared: Path, *, key: str, view: str, cfg_rel: str, job_id: str, sha: str,
                               evals: str | None, src_repo: Path = REPO, extra: str = "", after: str = "") -> subprocess.CompletedProcess:
    """Run mmlu_full.sbatch's registration path FOR REAL: the sbatch's own front-door decision block, then common.sh's
    `f22_registry_register` (the real `sisteer.registry.emit_cli register --from-config`, the registry's planner over the
    checkout's seeds, the pending file + the spool records), then the sbatch's `mmlu_full_registry_check`; `after` runs last
    (e.g. the terminal-state function). `evals` = the job env's F22_REGISTRY_EVALS (None = unset: a paper tag's launch)."""
    import textwrap
    script = textwrap.dedent(f"""
        set -u
        cd {checkout}
        export REPO_DIR={src_repo} SHARED={shared} SISTEER_REGISTRY_SPOOL={shared}/registry/records
        export SISTEER_REGISTRY_SUBSTRATE=slurm SISTEER_REGISTRY_CLUSTER_LABEL=cluster_a CLUSTER=slurm
        export SLURM_JOB_ID={job_id} SLURM_JOB_NAME=mmluF-{key}-{view} WAVE_SHA={sha}
        export KEY={key} VIEW={view} TAG={key if view == "main" else key + "_" + view} CFG={cfg_rel} F22_ROSTER=configs/sdfbase_full22_roster.yaml F22_SERVING_PATH=lora_native
        {"export F22_REGISTRY_EVALS=" + evals if evals is not None else "unset F22_REGISTRY_EVALS"}
        {extra}
        source scripts/sdfbase/full22/common.sh
        f22_log(){{ echo "[log] $*"; }}
        f22_stop_requested(){{ [ -e "$SHARED/STOP" ]; }}
        {mmlu_full_registry_block()}
        f22_registry_register || true
        mmlu_full_registry_check; rc=$?
        echo "CHECK_RC=$rc IDENT=${{SISTEER_REGISTRY_RUN_IDENTITY:-}} EVALS=${{F22_REGISTRY_EVALS:-}}"
        {after}
    """)
    env = registry_emitting_env(PATH=f"{Path(VENV_PY).parent}:{os.environ['PATH']}")     # `python3` = the venv's (PyYAML + sisteer's deps)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=300)


def run_f22_registration(checkout: Path, shared: Path, *, key: str, view: str, cfg_rel: str, job_id: str, job_name: str, sha: str,
                         evals: str, cluster_label: str = "cluster_a", after: str = "", src_repo: Path = REPO) -> subprocess.CompletedProcess:
    """Run common.sh's job-start registration FOR REAL for one wave-shell job (the Petri proxy, a static job, …): the real
    `f22_registry_register` (the real `sisteer.registry.emit_cli register --from-config` over the checkout's seeds — one run per declared
    eval x object of `cfg_rel`'s explicit_cells, the pending file `<shared>/registry/pending/<job_id>.json` with every run's routing
    facts, `run_state running` per run), then `after`. `evals` = the job env's F22_REGISTRY_EVALS."""
    import textwrap
    script = textwrap.dedent(f"""
        set -u
        cd {checkout}
        export REPO_DIR={src_repo} SHARED={shared} SISTEER_REGISTRY_SPOOL={shared}/registry/records
        export SISTEER_REGISTRY_SUBSTRATE=slurm SISTEER_REGISTRY_CLUSTER_LABEL={cluster_label} CLUSTER=slurm
        export SLURM_JOB_ID={job_id} SLURM_JOB_NAME={job_name} WAVE_SHA={sha}
        export KEY={key} VIEW={view} TAG={key if view == "main" else key + "_" + view} CFG={cfg_rel} F22_ROSTER=configs/sdfbase_full22_roster.yaml F22_SERVING_PATH=lora_native
        export F22_REGISTRY_EVALS={evals}
        source scripts/sdfbase/full22/common.sh
        f22_log(){{ echo "[log] $*"; }}
        f22_registry_register || true
        echo "REGISTERED pending=${{SISTEER_REGISTRY_RUN_IDENTITY:-}}"
        {after}
    """)
    env = registry_emitting_env(PATH=f"{Path(VENV_PY).parent}:{os.environ['PATH']}")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=300)


def registry_emitting_env(**extra) -> dict:
    """A child environment in which the results registry EMITS: tests/conftest.py runs every test with
    `SISTEER_REGISTRY_DISABLED=1` (emit.DISABLED_ENV) — the switch every emitter honours — so a fixture that exercises the REAL
    registration / store-hook path clears it (and any inherited pending identity) for its subprocesses."""
    env = dict(os.environ)
    for k in ("SISTEER_REGISTRY_DISABLED", "SISTEER_REGISTRY_RUN_IDENTITY", "REGISTRY_EMIT"):
        env.pop(k, None)
    env.update(extra)
    return env


def read_spool(spool: Path) -> list:
    """Every record under a producer spool (`<spool>/<campaign>/<run dir>/NNNNNN-<type>.jsonl`), as (record_type, payload) in file order."""
    out = []
    for p in sorted(spool.rglob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out.append((rec.get("record_type"), rec.get("payload") or {}))
    return out


def fold_spool_runs(spool: Path, *more_spools: Path, decode_policy_id=None, decode_policy_match=None) -> list:
    """The `run` rows a harvester would fold from a producer spool — a fixture reader standing in for the registry's reducer over
    the REAL emitted records: the `run` payload's identity columns, every `run_patch`'s fields folded in (store_uri, n_items…),
    the LAST `run_state` as the lifecycle, with the reducer's ONE rule for a rule-scored run applied explicitly (`generated` reaches
    `ingested` at the harvester's stamp). `decode_policy_id` / `decode_policy_match` stand in for the harvester's decode derivation
    (decode_match.stamp_run_rows) when given. Never a hand-written identity: a run with no `run` record is not returned. `more_spools`:
    further producers' spools folded AFTER the first, in order — the harvester's one database over every producer (the Petri hand-off:
    the proxy's `run` records in the cluster's spool, the grading-host drive's `run_patch` / `run_state` in the grading host's own)."""
    runs: dict = {}
    for rt, p in [rec for sp in (spool, *more_spools) for rec in read_spool(sp)]:
        rid = p.get("run_id")
        if not rid:
            continue
        if rt == "run":
            runs.setdefault(rid, {}).update({k: p.get(k) for k in ("run_id", "model_id", "code_sha", "instrument_id", "item_set_id", "eval_id", "object_id",
                                                                      "cluster_job_id", "store_uri", "lifecycle_state", "campaign")})
            runs[rid]["_seen_run"] = True
        elif rt == "run_patch":
            runs.setdefault(rid, {}).update({k: v for k, v in (p.get("fields") or {}).items() if k in ("store_uri", "n_items_generated")})
        elif rt == "run_state":
            runs.setdefault(rid, {})["lifecycle_state"] = p.get("state")
    out = []
    for rid, r in runs.items():
        if not r.get("_seen_run"):
            continue
        r.pop("_seen_run", None)
        if r.get("lifecycle_state") == "generated":
            r["lifecycle_state"] = "ingested"           # reducer.py: a rule-scored run reaches ingested from generated
        r["decode_policy_id"] = decode_policy_id
        r["decode_policy_match"] = decode_policy_match or ("exact" if decode_policy_id else "unrecorded")
        out.append(r)
    return out


def run_stage(stage: str, *args: str, site: Path, env: dict | None = None, cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    e = dict(os.environ); e.pop("SISTEER_SITE_DIR", None); e.pop("SISTEER_CONFIG", None)
    e["SISTEER_STATE_DIR"] = str(site.parent / "state")
    if env:
        e.update(env)
    return subprocess.run(["bash", str(REPRO / stage), *args, "--config", str(site / "config.env")], capture_output=True, text=True, env=e,
                          cwd=str(cwd or REPO), timeout=timeout)


def assert_no_tokens(text: str, where: str = "") -> None:
    """A stage's output carries no personal identifier: the `out` rows of the token table, matched as the publication scan
    matches them (a tailnet id separated from its hostname is caught by its shape, in output as in a file)."""
    hits = token_hits(text)
    assert not hits, f"{where}: personal identifier(s) {hits} in output"


def interpreter_venv_root(python: str) -> "str | None":
    """The virtual environment `python` belongs to (`<env>/bin/python` beside `<env>/pyvenv.cfg`), else None: the directory 08_grade.sh's
    Petri install line prints as `UV_PROJECT_ENVIRONMENT=<env>` when the interpreter lacks the petri group."""
    env = Path(python).parent.parent
    return str(env) if (env / "pyvenv.cfg").is_file() else None


def scrub_site(text: str, *, tmp: "Path | str", python: str = VENV_PY) -> str:
    """A stage's output with the RUNNER'S OWN paths normalised before `assert_no_tokens` — this checkout (`<repo>`), the interpreter the
    site's SISTEER_PYTHON names (`<python>`) and the virtual environment it belongs to (`<venv>`: 08_grade.sh's Petri install line prints it
    when the group is absent, which on the cluster gate is a venv under the login's shared root — two gate reds, 2026-10-13), the test's temp
    root (`<tmp>`) and HOME (`<home>`): they are the host's, not identifiers a stage hard-codes. Everything else is scanned as printed. The
    one scrub the dry-run scanners share (test_front_door_dry_run, test_reproduce_readme_committed_sequence)."""
    out = text.replace(str(REPO), "<repo>").replace(python, "<python>")
    env = interpreter_venv_root(python)
    for spelling in dict.fromkeys([env, os.path.realpath(env)] if env else []):   # the install line spells the logical path; a symlinked venv may print either
        out = out.replace(spelling, "<venv>")
    return out.replace(str(tmp), "<tmp>").replace(os.path.expanduser("~"), "<home>")


PETRI_IMPORT_CHECK = "import inspect_ai, petri_bloom"      # grade_daemon.PETRI_IMPORT_CHECK: the drive interpreter's contract


def fake_python(root: Path, *, petri: bool, venv: bool = True) -> str:
    """A stand-in interpreter for a stage's SISTEER_PYTHON: it answers 08_grade.sh's Petri preflight (`-c "import inspect_ai,
    petri_bloom"`) as told — exit 0 with `petri=True`, a ModuleNotFoundError with `petri=False` — and execs the real VENV_PY for
    everything else (the lib's models.py helpers), so a test's outcome never depends on whether the runner's venv carries the
    petri group. Shaped like a virtual environment (`<env>/bin/python` beside `<env>/pyvenv.cfg`) unless `venv=False`, which
    is the interpreter-outside-any-venv case the stage's recovery line distinguishes. Returns the path of the executable."""
    d = root / ("python_with_petri" if petri else "python_without_petri") / ("venv" if venv else "bare")
    bindir = d / "bin" if venv else d; bindir.mkdir(parents=True, exist_ok=True)
    if venv:
        (d / "pyvenv.cfg").write_text(f"home = {Path(VENV_PY).parent}\n")
    answer = "exit 0" if petri else "echo \"ModuleNotFoundError: No module named 'inspect_ai'\" >&2; exit 1"
    p = bindir / "python"
    p.write_text("#!/bin/bash\n"
                 f"if [ \"${{1:-}}\" = -c ] && [ \"${{2:-}}\" = \"{PETRI_IMPORT_CHECK}\" ]; then {answer}; fi\n"
                 f"exec {VENV_PY} \"$@\"\n")
    p.chmod(0o755)
    return str(p)


PETRI_BEHAVIORS = ("cyber_bio_chem_capabilities", "self_preservation", "delusion_sycophancy", "defer_to_users", "reasoning_unfaithfulness", "instruction_evasion")


def petri_bundle(root: Path, n: int = 10) -> Path:
    """A SYNTHETIC frozen Petri scenario bundle in scenarios.freeze_scenarios' shape — `bundle_manifest.json` carrying the committed lock's
    content hash (configs/capeval/petri-scenarios.lock.yaml) and `<behavior>/scenarios/seeds/<scenario id>.md` per behavior (the stems
    are the scenario ids; here invented names, never a real seed's text). stage.py petri_scenario_ids enumerates it through
    $SISTEER_PETRI_BUNDLE_DIR."""
    import yaml
    lock = yaml.safe_load((REPO / "configs" / "capeval" / "petri-scenarios.lock.yaml").read_text())
    d = root / "petri_bundle"; d.mkdir(parents=True, exist_ok=True)
    (d / "bundle_manifest.json").write_text(json.dumps({"version": 1, "content_hash": lock["bundle"]["content_hash"], "behaviors": {}, "files": {}}))
    for b in PETRI_BEHAVIORS:
        seeds = d / b / "scenarios" / "seeds"; seeds.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (seeds / f"{b[:6]}_scn_{i:02d}.md").write_text("synthetic seed\n")
    return d


def petri_scenario_ids(n: int = 10) -> dict:
    return {b: [f"{b[:6]}_scn_{i:02d}" for i in range(n)] for b in PETRI_BEHAVIORS}


def lay_down_petri_store(cell: Path, ids_by_behavior: dict, *, drop=(), epochs: int = 1) -> Path:
    """A Petri per-cell store the way petri_runner writes it (rows.jsonl with the three required metrics per scenario, petri_incomplete.json
    with an empty ledger, results.json) for every `<behavior>/<scenario>` of `ids_by_behavior` except those in `drop` (left unfiled)."""
    rows = []
    for b, ids in ids_by_behavior.items():
        for sid in ids:
            if f"{b}/{sid}" in drop:
                continue
            for ep in range(1, epochs + 1):
                iid = sid if epochs == 1 else f"{sid}#e{ep}"
                for metric in (b, "eval_awareness", "scenario_realism"):
                    rows.append({"task": f"petri/{b}", "item_id": iid, "metric": metric, "value": 4.0, "tier": -1, "domain": "petri_behavior",
                                 "extra_json": json.dumps({"scenario_id": sid, "epoch": ep, "dimension": metric})})
    cell.mkdir(parents=True, exist_ok=True)
    (cell / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (cell / "petri_incomplete.json").write_text(json.dumps({"version": 1, "cell_id": "none", "petri_run_key": cell.name, "unresolved": []}))
    (cell / "results.json").write_text(json.dumps({"cell_id": "none", "petri_run_key": cell.name, "rows_emitted": len(rows),
                                                   "behaviors": {b: {"n_items": len(ids) * epochs, "n_scenarios": len(ids)} for b, ids in ids_by_behavior.items()}}))
    return cell


def sampled_controls_store(cell: Path, items: dict, *, at_cap: dict | None = None, cap: int = 2048, decode: dict | None = None) -> Path:
    """A SAMPLED served-controls cell store the way scripts/served_capability_run.py writes it (modelled on the real smoke stores
    and tests/test_sampled_controls.py: the file names, the spool header line, the meta sidecar bound by text_sha256, gen_stats stamped complete) —
    `items` = {control key: {item_id: text}} for `xstest` (spool gens_xstest_v2.jsonl) and/or `or_bench_hard_1k` (gens_or_bench_hard_1k.jsonl);
    `at_cap` = {control key: [item ids at the cap]}. No rows.jsonl: the judge's file lives elsewhere. Synthetic texts, never a real generation."""
    spools = {"xstest": "gens_xstest_v2.jsonl", "or_bench_hard_1k": "gens_or_bench_hard_1k.jsonl"}
    at_cap = at_cap or {}; decode = decode or {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "cap": cap}
    priv = cell / "private"; priv.mkdir(parents=True, exist_ok=True)
    gs = {}
    for control, texts in items.items():
        name = spools[control]; spool = priv / name; meta = priv / (name[:-len(".jsonl")] + ".meta.jsonl"); caps = priv / (name[:-len(".jsonl")] + ".caps.jsonl")
        with open(spool, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"__spool_header__": {"cell_id": cell.name, "control": control, "cap": cap, "gen_sha": "0" * 16}}, sort_keys=True) + "\n")
            for iid, text in texts.items():
                fh.write(json.dumps({"item_id": iid, "text": text}, sort_keys=True) + "\n")
        with open(meta, "w", encoding="utf-8") as fh, open(caps, "w", encoding="utf-8") as fc:
            for iid, text in texts.items():
                hit = iid in set(at_cap.get(control, []))
                fh.write(json.dumps({"item_id": iid, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "at_cap": hit, "finish_reason": "length" if hit else "stop",
                                     "completion_tokens": cap if hit else 30, "prompt_tokens": 17, "max_tokens": cap, "source_cap": cap, "sampling_seed": 7, "regenerated": False,
                                     "floor_sample": False}, sort_keys=True) + "\n")
                fc.write(json.dumps({"item_id": iid, "cap": cap, "at_cap": hit}) + "\n")
        gs[control] = {"task": f"judged/{name[len('gens_'):-len('.jsonl')]}", "n_new": len(texts), "n_total": len(texts), "spool": str(spool), "meta": str(meta), "decode": dict(decode),
                       "cap": cap, "n_at_cap": len(at_cap.get(control, [])), "at_cap_item_ids": sorted(at_cap.get(control, [])), "n_regenerated": 0, "n_copied": 0, "n_floor_sample": 0}
    (cell / "gen_stats.json").write_text(json.dumps(gs, indent=1, sort_keys=True))
    (cell / "manifest.json").write_text(json.dumps({"cell_id": cell.name, "resolved_config": {"gen": {"max_new_tokens": cap, "do_sample": True, **{k: v for k, v in decode.items() if k != "cap"}}}}))
    (cell / "evidence.json").write_text(json.dumps({"driver": "served_capability_run_v2", "written_at": "2026-10-13T00:00:00Z"}))
    (cell / "cap_ladder.json").write_text(json.dumps({"cap": cap, "escalate": False}))
    return cell
