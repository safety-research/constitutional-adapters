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
    ("process", "hash", "b2757d5c2fbc32d31ea6ad1930d899259f2aab22acd97c55c43d864db4e869c6", {"sep": ["-", "_"], "n": 2, "head": "54d00d867758cef816bc4685f58e327b949712b07ebd17c3485f3ffc9e9f5133"}),
    ("process", "hash", "aeabe2dabcbe2388484d6c2eef3c1dd072432cc59bfc22508f51199574ee0cfc", {"sep": ["-", "_"], "n": 4, "head": "2db16763c475569ce528bdd93ea32b88c03f7aeea12e4cd4c51280e050dd58d2"}),
    ("process", "hash", "3dc8ffc9e11ac8b7126186efebbc221e66e18bde4c5e4ca41371f48dc84e8702", {"sep": ["-", "_"], "n": 3, "head": "eaf9d8e7d58a0077a65490523e8e8cdcc9f96d1904e891fa8e85e8d97a989236"}),
    ("process", "hash", "0bcf36a1f18078e6dd7a9f4a73e3269ec05c9d4ebd8b40dfdb5976bc5fad1ab8", {"sep": ["-", "_"], "n": 3, "head": "eaf9d8e7d58a0077a65490523e8e8cdcc9f96d1904e891fa8e85e8d97a989236"}),
    ("process", "hash", "89dd6b25c5be6cb0b4aa0559ab80f271aa23f84a35874f3eaea96b04037fd226", {"sep": ["-", "_"], "n": 2, "head": "c90795332b95d99e97742b424920d6c6bd5209ea7b522a3448ed2cdac7779809"}),
    ("process", "hash", "f76c792f339261be10eaa61ab3f73599f4bc6b4dca5777de01f04c87644f8595", {"sep": ["-", "_"], "n": 3, "head": "c90795332b95d99e97742b424920d6c6bd5209ea7b522a3448ed2cdac7779809"}),
    ("process", "hash", "03f520139b2db84cc2bb379e602542e00ea1251982e0dc81ce99fc2833826c13", {"sep": ["-", "_"], "n": 2, "head": "c90795332b95d99e97742b424920d6c6bd5209ea7b522a3448ed2cdac7779809"}),
    ("process", "hash", "f2577cfa46f9768c809c19509e895e95831ad933792348e958fdd1495549b9bc", {"sep": ["-", "_"], "n": 3, "head": "91cabb4f0e2af351e051dfb9bda0b485d537d87ed7e5091eebf619ebd05f43ab"}),
    ("process", "hash", "b44d5493834fbd5a80fffe8e8cf9c056a10c916d552301366d248146165c5fa0", {"sep": ["-", "_"], "n": 3, "head": "91cabb4f0e2af351e051dfb9bda0b485d537d87ed7e5091eebf619ebd05f43ab"}),
    ("process", "hash", "df17c8a714841a3cf31739b3538c1bf2aa9baf7f06aef639614abcdfbd0ee6cd", {"sep": ["-", "_"], "n": 2, "head": "683777765dbb85533ecf88e36593789e0d6954b1196738dd04585ea90231172b"}),
    ("process", "hash", "708ce4d75e2196704b0e8192528b190adc9f2bba0ecaccb27db0f1bb522fb59f", {"sep": ["-", "_"], "n": 2, "head": "683777765dbb85533ecf88e36593789e0d6954b1196738dd04585ea90231172b"}),
    ("process", "hash", "4ca633f74b3f471ed19887c57ee7d8d524c509ddd26e35e16be1521bf216b815", {"sep": ["-", "_"], "n": 2, "head": "683777765dbb85533ecf88e36593789e0d6954b1196738dd04585ea90231172b"}),
    ("process", "hash", "15c69dcebf3387c923d18cef051fe4c0c62117d037991a9b5d95ef605eb3ad8e", {"sep": ["-", "_"], "n": 3, "head": "683777765dbb85533ecf88e36593789e0d6954b1196738dd04585ea90231172b"}),
    ("process", "hash", "81de85b0579df830e60dcb785293d9e62f795a6e871c2de1bfeb8776360534ee", {"sep": ["-", "_"], "n": 3, "head": "683777765dbb85533ecf88e36593789e0d6954b1196738dd04585ea90231172b"}),
    ("process", "hash", "7ad69916a6d20ef41cc83b6aacfa8ed4074e35c94e4b78e574774b828ef01ebb", {"sep": ["-", "_"], "n": 3, "head": "683777765dbb85533ecf88e36593789e0d6954b1196738dd04585ea90231172b"}),
    ("process", "hash", "1bc6166d4da9ca27fedbb4706fd49ec18bb5a14a95981098cb7a9f201838d70a", {"sep": ["-", "_"], "n": 2, "head": "683777765dbb85533ecf88e36593789e0d6954b1196738dd04585ea90231172b"}),
    ("process", "hash", "99ff3b31b6c8ed75a5eecd229047d9b1057411c14b0a0b3cf08bcb8e8fecd862", {"sep": ["-", "_"], "n": 2, "head": "683777765dbb85533ecf88e36593789e0d6954b1196738dd04585ea90231172b"}),
    ("process", "hash", "8e7f3df24cb6ef7b459876cc1885060399b75c4818d034058a74fcc52948e52c", {"sep": ["-", "_"], "n": 3, "head": "8345f2d4a6472fced89ee6ed9c3e7f8acb85153b6cab7e759bc442e56b30423b"}),
    ("process", "hash", "dd731d579f2caf84630c26e3f54e988e1b29cc7085541dc095081baa7bd32a37", {"sep": ["-", "_"], "n": 2, "head": "8345f2d4a6472fced89ee6ed9c3e7f8acb85153b6cab7e759bc442e56b30423b"}),
    ("process", "hash", "7f66225cfaeb8e1df842258c648b93de747b38ba139f676b6ef11d1c5f13b093", {"sep": ["-", "_"], "n": 2, "head": "8345f2d4a6472fced89ee6ed9c3e7f8acb85153b6cab7e759bc442e56b30423b"}),
    ("process", "hash", "06b9d67794e2aafc432db9fb5c6dd95851db49f888f99a874468f2b61b0048dd", {"sep": ["-", "_"], "n": 2, "head": "8345f2d4a6472fced89ee6ed9c3e7f8acb85153b6cab7e759bc442e56b30423b"}),
    ("process", "hash", "c98ca052876645c509f285b467380caaee4a5c6408507507cd9ac28a2baa8a84", {"sep": ["-", "_"], "n": 3, "head": "8345f2d4a6472fced89ee6ed9c3e7f8acb85153b6cab7e759bc442e56b30423b"}),
    ("process", "hash", "8ebb0c433336c17322d9262a102a8b77c8136df120f4f538a7cf336037fbf24a", {"sep": ["-", "_"], "n": 2, "head": "8345f2d4a6472fced89ee6ed9c3e7f8acb85153b6cab7e759bc442e56b30423b"}),
    ("process", "hash", "a935ea739f5927cef47bc6b4e4de50ca0a72cc55c2a0fe722952af019a7c02cf", {"sep": ["-", "_"], "n": 3, "head": "8345f2d4a6472fced89ee6ed9c3e7f8acb85153b6cab7e759bc442e56b30423b"}),
    ("process", "hash", "999f81653521eb5a435b36d3d4de906fc8d783b47adb12e2f5bfb2098b60bd50", {"sep": ["-", "_"], "n": 2, "head": "8345f2d4a6472fced89ee6ed9c3e7f8acb85153b6cab7e759bc442e56b30423b"}),
    ("process", "hash", "1e76171fa3bf3f944eeded718939cffb05d1f75f22637ab05900b86b062002e6", {"sep": ["-", "_"], "n": 2, "head": "2ff8ddd61f5e0a0c72ad9390b7c448e32b5055707f1be2e65a361b24f2a180ce"}),
    ("process", "hash", "df67ed21dcfe14b31bc6cf3e497db36a1f963b52fd776047192f263fc1f5b4c9", {"sep": ["-", "_"], "n": 2, "head": "d90ee9ccf6bea1d2942a7b21319338198dec2a746f8a0d0771621f00da2e0864"}),
    ("process", "hash", "598900cb99f52ac8ec1a12fd1bf3e3ec984698847faf2cb43d7fc877b06fc588", {"sep": ["-", "_"], "n": 3, "head": "d90ee9ccf6bea1d2942a7b21319338198dec2a746f8a0d0771621f00da2e0864"}),
    ("process", "hash", "d2b76e7f1d22e9f9caa621170f180636d95ac6a22eef06ac4de93be258fbb224", {"sep": ["-", "_"], "n": 2, "head": "1c6e6c4c02e55178e85890fc9bbed4ce046415ec8122bf38f711b779184ae2a0"}),
    ("process", "hash", "e65e387fce9cc244ecc0671ada30bfe266f96e6ad580c504eb7674f349b63303", {"sep": ["-", "_"], "n": 2, "head": "1c6e6c4c02e55178e85890fc9bbed4ce046415ec8122bf38f711b779184ae2a0"}),
    ("process", "hash", "9dba6990540b266e073b95ad16ee1548c24e8c005f2eb7a9dee30d0c39eecff1", {"sep": ["-", "_"], "n": 3, "head": "1c6e6c4c02e55178e85890fc9bbed4ce046415ec8122bf38f711b779184ae2a0"}),
    ("process", "hash", "fa0b23bb788d1d35ab1790953d2ef4b999998a77135f56e063adf4c5e0858ee5", {"sep": ["-", "_"], "n": 2, "head": "42ff567dff3b465faee2f793db77da8f1c65bcba1debfd4082d021226a47c9bf"}),
    ("process", "hash", "0ab79034985f719d6326d1fd301036513fbe96e10bdb0bc8c26e392abe4b512a", {"sep": ["-", "_"], "n": 2, "head": "42ff567dff3b465faee2f793db77da8f1c65bcba1debfd4082d021226a47c9bf"}),
    ("process", "hash", "fb2eed3040281014e075063869c576c7ee6dc0b54550c93bc7c390ee32f5e431", {"sep": ["-", "_"], "n": 2, "head": "42ff567dff3b465faee2f793db77da8f1c65bcba1debfd4082d021226a47c9bf"}),
    ("process", "hash", "f364160373a50974f13bdff56834aa6cdc78732705006a9690d183476c4b1b71", {"sep": ["-", "_"], "n": 2, "head": "42ff567dff3b465faee2f793db77da8f1c65bcba1debfd4082d021226a47c9bf"}),
    ("process", "hash", "2c1c5790f6cf8279ad600c94f8e0199fd33c993b907449e85e5d03d8021174ea", {"sep": ["-", "_"], "n": 2, "head": "42ff567dff3b465faee2f793db77da8f1c65bcba1debfd4082d021226a47c9bf"}),
    ("process", "hash", "88e4a7b242471fd18be8e6e96d9a64c1094398ae95b0a284e5b38da15e56657b", {"sep": ["-", "_"], "n": 2, "head": "42ff567dff3b465faee2f793db77da8f1c65bcba1debfd4082d021226a47c9bf"}),
    ("process", "hash", "b14cded6e595c8b042b341c2df9eedcf16cc516ef1efddc633c36e22942b61ea", {"sep": ["-", "_"], "n": 2, "head": "42ff567dff3b465faee2f793db77da8f1c65bcba1debfd4082d021226a47c9bf"}),
    ("process", "hash", "35c781b1c2038148f3caaf8613f199ba9511906c758006a9fd5cdd2fa2aeb275", {"sep": ["-", "_"], "n": 2, "head": "42ff567dff3b465faee2f793db77da8f1c65bcba1debfd4082d021226a47c9bf"}),
    ("process", "hash", "1a1654b1b283d008b33b5d0e94b23c155a5beadeb6430106bdf9c1eac1bc58b6", {"sep": ["-", "_"], "n": 2, "head": "42ff567dff3b465faee2f793db77da8f1c65bcba1debfd4082d021226a47c9bf"}),
    ("process", "hash", "1857e35664b3e6cd16b6b41f4e3cafef1bcc0b25db662e549ad571d534e93c82", {"sep": ["-", "_"], "n": 2, "head": "42ff567dff3b465faee2f793db77da8f1c65bcba1debfd4082d021226a47c9bf"}),
    ("process", "hash", "0d426f0d63331fcbc01afbb25c963c608d4704aed17969f5683abe97ae2a4b77", {"sep": ["-", "_"], "n": 2, "head": "42ff567dff3b465faee2f793db77da8f1c65bcba1debfd4082d021226a47c9bf"}),
    ("process", "hash", "e33182d475379634b109f254e4e3b5a3ede16de5b8d5b8057fdedf922016a139", {"sep": ["-", "_"], "n": 2, "head": "42ff567dff3b465faee2f793db77da8f1c65bcba1debfd4082d021226a47c9bf"}),
    ("process", "hash", "01150a792c25151ef6af8c237c5a8a4aa3b08434929885520b0cbeb030b0f765", {"sep": ["-", "_"], "n": 2, "head": "6f84c4ce20e5f2e70fbed4a6be007cb22d703f79892256ad4d2001f7e8021805"}),
    ("process", "hash", "f94ddb5a51d061ce5a7de3456018a7153e104cfc3c2cda2b649752567d08efb0", {"sep": ["-", "_"], "n": 3, "head": "67ad8f41a7bb0a18c45f3efd8f6a32bdcd832bcf8f3f2eb44006c03d53076e18"}),
    ("process", "hash", "d97e7462573ef47dae84b6995817407f721540db55f57cde2dd88268c3d91131", {"sep": ["-", "_"], "n": 3, "head": "67ad8f41a7bb0a18c45f3efd8f6a32bdcd832bcf8f3f2eb44006c03d53076e18"}),
    ("process", "hash", "eeeced5430c3d09191c8124753c58443e7c454e781d7297a365502c7be5d7210", {"sep": ["-", "_"], "n": 2, "head": "9dff50df08c635815f4b19da10f756605a34a79a48d4ba48712782502975a70e"}),
    ("process", "hash", "a249110a79bb83d4f4d503474f0452adfb22c8215bd8c9f169f278ae1bc2978a", {"sep": ["-", "_"], "n": 2, "head": "9dff50df08c635815f4b19da10f756605a34a79a48d4ba48712782502975a70e"}),
    ("process", "hash", "67f671d6758380542e2c073bd0fc02cf063a70a73686af86325f51e4b9b11ea7", {"sep": ["-", "_"], "n": 2, "head": "8148b7fefaf89ebf8e7951994d2601d731220c5460faecc64c69d12bb7ff3213"}),
    ("process", "hash", "d7e729b6b95285ecb2c909a5855321c677fdbe25162f71893860965d11ee22f7", {"sep": ["-", "_"], "n": 3, "head": "58904effcb30c04e5daf2f14baad7f8557733400ef4e1cf4f7d489ca130ec255"}),
    ("process", "hash", "6825469155e0d82f0265def3330ceeb87e9a07a68458c960d236c55bb28a2eb2", {"sep": ["-", "_"], "n": 4, "head": "4ea140588150773ce3aace786aeef7f4049ce100fa649c94fbbddb960f1da942"}),
    ("process", "hash", "680da185c39639a01b68e12b850e377c539bd17a3ad37299f76f1f105ea8f686", {"sep": ["-", "_"], "n": 2, "head": "0811138fc65a7023081b236f43e84f0b848be57d64d83041bf449263b0cf1d80"}),
    ("process", "hash", "65d5510cf636cd7bdd5c0278b11254761fa19031698b069aedbce663f2ac2461", {"sep": ["-", "_"], "n": 2, "head": "a12947dbf7d04ce3a05572d398561b3387440aa01215092ade1f1db4ab652430"}),
    ("process", "hash", "a6e9c84c4d105f2c11748a498cc8469148506c7dab02ac1e72fe65ae5b1a7120", {"sep": ["-", "_"], "n": 2, "head": "a12947dbf7d04ce3a05572d398561b3387440aa01215092ade1f1db4ab652430"}),
    ("process", "hash", "cf969b8bc1202b87c69ced0fa13a7e5b32556b37e6899e86043f299c0ef9e9c6", {"sep": ["-", "_"], "n": 2, "head": "a12947dbf7d04ce3a05572d398561b3387440aa01215092ade1f1db4ab652430"}),
    ("process", "hash", "cceb9eb3225e46b5d67eaed7777f4a511b3dbe83a88c7323c824711d2ce2009b", {"sep": ["-", "_"], "n": 2, "head": "8dfd3780acc9cdcc3c7506c904e8cc28d7b9b8712f35532c56eff7ac07ff4a68"}),
    ("process", "hash", "71933c5947554324f0028884d0d2fb781394b3f139af2b397310ae11b7f00049", {"sep": ["-", "_"], "n": 3, "head": "10e86c6514d40f2a3e861b31847340ee8c8ed181029a17b042f137121d28863e"}),
    ("process", "hash", "3e4aa09992d3358c16332aeb1f884fe12ff10981650333562b8576099beecde5", {"sep": ["-", "_"], "n": 3, "head": "d6709cd49ed879665227b65df1241d46d318b5f29e2ed08835652c9b407ca30b"}),
    ("process", "hash", "b73b15ecc77338cfc7671189902069f2c7308146b9bbea5ec36b0d7270f8de36", {"sep": ["-", "_"], "n": 3, "head": "6c58ab1eeba9037dcf22088b9be59a1788d36ce2a90cb608844e2e23a3a2f925"}),
    ("process", "hash", "b3c17bc3b4fc49a8fe7dad9b6bcf2f779afbbcaee27d31a1f7deee65193d63d3", {"cs": True}),
    ("process", "hash", "9b4df3b4829915b09a16ce819eee2f24ea43e7ae3e71ad2475d14bdb8cf53a83", {"sep": ["-", "_"], "n": 3, "head": "d339f720de1fd92a672df9ef19a8cdbda6171cbf33fcd35ad95c46f8aebaf628"}),
    ("process", "hash", "7073f315590aa022bce90a8824ff3f82f8ec11ac6eb36e33eebea1b24a553dab", {"sep": ["-", "_"], "n": 3, "head": "ebe71c53e9a0a1e6334f2b873867c105b6a37d865839a2d268d04a676d3ca8cb"}),
    ("process", "hash", "14cb8df8f637016224a642b787d96383bf50075658cee751abb2f6c403749687", {"sep": ["-", "_"], "n": 2, "head": "ebe71c53e9a0a1e6334f2b873867c105b6a37d865839a2d268d04a676d3ca8cb"}),
    ("process", "hash", "f32fdcc0cd144239d3bb266853f4b5e17df70131dbb8786310933b80ccccf108", {"sep": ["-", "_"], "n": 2, "head": "ebe71c53e9a0a1e6334f2b873867c105b6a37d865839a2d268d04a676d3ca8cb"}),
    ("process", "hash", "dddc8e1d74a771d272c4616f410e0e47a3d8580fd764d759d8d7a047958a1ca0", {"sep": ["-", "_"], "n": 2, "head": "ebe71c53e9a0a1e6334f2b873867c105b6a37d865839a2d268d04a676d3ca8cb"}),
    ("process", "hash", "08e2e8e9b5ab773714a107650cd1d13b9e4a2d829c3deb2b861f0020df47715c", {"sep": ["-", "_"], "n": 3, "head": "6c733452910a839d5e61f11c809a39e4a962b3ab4c2bcdded4d5294a39a72c45"}),
    ("process", "hash", "761b742c772aa3a0bdb443c09128bbb2cdcb14b7936ff2de1dba6f5b31657298", {"sep": ["-", "_"], "n": 2, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "00feb5f3dce388d24a55a427ff2156beb246603387398d061e64155ba3b08100", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "270711b5f5b1e457cdcc2f8fb18c89ec0a37ecf50e23725e8ff26b485b5cea4b", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "951fd09f46200e3714355532fa711b285f0cda3e7b5da5e6f1ec126100b4d3f7", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "bf6748f60bad51ba98ee453199f6847e2cf7a18f83a4301d9df4f1d5f14aef7c", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "87a31151b90f9a284b9008adcc6ae1ee439ffed1f994ee2ef31f26218dac2cf0", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "27e41887609bc27e6f24effd303a5c636167c26e6e20eacd4ee2df89f92cefd8", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "6dd48f2bd836d29950a771dd2d6336b0494a041174e05b5f00bb6394a970a289", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "9c0e1c7c4188c11242c74b46a0905c1a5cf8657bbb155728c1449f7c89c4bb57", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "e929787efbfaeb8ad08266579c38b020a53da64ac0de07371073ed32ac2dfa44", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "2f4531e82f27b34d8ced0339704a97f451aa6505ade0bfd657b26fe18c5c6ab0", {"sep": ["-", "_"], "n": 3, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "87052582d1f283b167607f53190849d372f6562ce7b2573cc9be23e933293914", {"sep": ["-", "_"], "n": 3, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "9a260a6df0dbd302861e055511834590b55ffe65d37b02ac4cc5a222cc83c4df", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "27ad4561a0f71e86164dfc38c67602e21bc005bf145c706bd943c2b14c1555b7", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "1b8956072516db4d5c0756101a66cdec83faa7c1049b4749bc7c2d4fbfea6fa1", {"sep": ["-", "_"], "n": 5, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "ffa2245fc85e17ce99530d1a2dd0280f0e86cc8bb9d701db23a6ee5ddafc92e9", {"sep": ["-", "_"], "n": 5, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "68b06e05a517509bc02a03c6999b91b48152bc29b3456ed41c150d3ecc696407", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "515231af3f97f9d57b758736dcd88cc5f9a70cab68162fc201005d28166f2dc9", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "95c72d4463be0bd3f8d57ec28b1ee67e41bc07c76a7230e37cd063d5c25d8cf8", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "619a941c13c1f7f5ad5f5770ddbb608a9449e1649288f144604e8cc8765c31a0", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "68d8a39bfd20b2d4d84c5cd3f747a34824d377e7a60034dcc3e122104fa86a1c", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "4476c6469b26d8df306f1a75fbdc697c4c28522d8c073cb017b3d028dd5521b6", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "4596fd891b57927f60136b5afbf2e7ebd782a1c47ce6db74722cae4937515196", {"sep": ["-", "_"], "n": 3, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "6e530152aaf6d2962d651003eb86020a9b2e40c4fc703f786e24386636e3a0d0", {"sep": ["-", "_"], "n": 3, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "ce4dfc36bb1101c26b5ceba6272c4000e628a849630c5af3155c653734eeede0", {"sep": ["-", "_"], "n": 3, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "20972a74d97b6c12dadd30e24174d61d588e616c1139545398da4394d098ba26", {"sep": ["-", "_"], "n": 3, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "97e8c5f86ddc8efa4ea96c8f26b2cbe3efabd9fec64cb62ecdb58193c3b728e9", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "6b367df9d17beed1cf29c1b332e1715f05ba9a80d516394b707288886b74f67e", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "65eddd762aab24c7586f51a841a7ec4b5bc4c78236c84b5fc508d486182b0717", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "743f39b6afc2fe3de548b9bf72144596f81fc451fc8487dc3257173ba5667b49", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "62ef1b7ff91a669d11948c892f72205175dc10c359e4fde40010c92eadddc0b2", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "081b6d02870c40a177e4d466ae70ca3f3b73c0b96f48558622dfd56786b42f97", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "8203747b9317949499e752753de6955ff9c754c9f15641c9ebbbc42e3649ddab", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "efc9931ac91a7c2611be78a21d90b3644d230b86f6cb241fd8ef02d5699ebab7", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "bc8ab9cb0fb378de9f2b285c3a6565bd12ea75e2e16c7ae33883e38b3566c47e", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "e555025ea5ca39f6094e7a043190fde28e0d44447a0b56a10566fa85c3dca30b", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "acba0a699f21615621b9023c90fe9c09510d0ae6677e2f4aa3cafa27b0f06f72", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "a4ddf8ca70aea094e2351cdc5bf55c1c9d7b17370379710586c26e1f835e80a1", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "fd01fab045d0ee6dc60f81d3e0b37c42c998f03757b81bfa93681cea41b564a4", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "f70d7b4d1ee4a1cb787fead3a84e4c750bf07032d3aedcf121fe4f1946b70cdd", {"sep": ["-", "_"], "n": 4, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "37f49d5e05f6da8faa2ebbf284284e4b0abb6ada809f32f91e01f5588790cc4e", {"sep": ["-", "_"], "n": 2, "head": "efa1f375d76194fa51a3556a97e641e61685f914d446979da50a551a4333ffd7"}),
    ("process", "hash", "81bc40e0bed43cb0172d056e960e7d99c0e4dad92d6dc9758dfdf42299342d2e", {"sep": ["-", "_"], "n": 4, "head": "67f2d22514622d1be30c14ee9f3cb104503a159a33a656ca91a1953aa9616429"}),
    ("process", "hash", "15f1515b4f02aa216db5ef4310d490e192acfbcc246ce3dbe9b8ea2c1b618fa7", {"sep": ["-", "_"], "n": 3, "head": "9f929fed50df2f03d868392c64a3db6a785636d6ecb5579030311ee59336e2f5"}),
    ("process", "hash", "6b163b9e0138e8921f03011632cb6473df806e5631cbe7038321edf8cd827a53", {"sep": ["-", "_"], "n": 2, "head": "9f929fed50df2f03d868392c64a3db6a785636d6ecb5579030311ee59336e2f5"}),
    ("process", "hash", "52f8231169e9b35d8ca2699fdf7fc76904111c04123cd4ade394c34371c08c8b", {"sep": ["-", "_"], "n": 2, "head": "9f929fed50df2f03d868392c64a3db6a785636d6ecb5579030311ee59336e2f5"}),
    ("process", "hash", "320291861bd3ea2f13a11534ea6ad87de0aaa8cecef73956cd22a9977a33f61c", {"sep": ["-", "_"], "n": 2, "head": "9f929fed50df2f03d868392c64a3db6a785636d6ecb5579030311ee59336e2f5"}),
    ("process", "hash", "53112dc4794ad2941436474884bd762bd70bcf1d3735e5228a85c1c9b360fe02", {"sep": ["-", "_"], "n": 4, "head": "53c30f54cfa99ebde95098f3aca84f936f91d8c662f25ae4331d430484a4d732"}),
    ("process", "hash", "a247f6bed5cce6d6f80c68651521187de831d39d4f3bca3d3fd0f21a0b750c3a", {"sep": ["-", "_"], "n": 4, "head": "53c30f54cfa99ebde95098f3aca84f936f91d8c662f25ae4331d430484a4d732"}),
    ("process", "hash", "c6ebcb0a65d62244b605f51d2d6dcff09e76b92e0f7304b39fa023aaf09918bb", {"sep": ["-", "_"], "n": 5, "head": "53c30f54cfa99ebde95098f3aca84f936f91d8c662f25ae4331d430484a4d732"}),
    ("process", "hash", "e0dbd2d5d6847687cdf7b5346b8cbaa127cc5232895c7aac2bafdff2697520f7", {"cs": True}),
    ("process", "hash", "16083e1de92139ce08ed92f2c021594f1a59300f44471e62700691d3bc10ac55", {"sep": ["-", "_"], "n": 2, "head": "711a6108ba2ce6ca93dd47d6817f2361db10d8ab6eec89460b2dfc2c325efabe"}),
    ("process", "hash", "ce0197e3326b08bdacb86d8a045fbb3d35ee6214946d5d63c7d7951160dc0555", {"sep": ["-", "_"], "n": 2, "head": "711a6108ba2ce6ca93dd47d6817f2361db10d8ab6eec89460b2dfc2c325efabe"}),
    ("process", "hash", "f1773718c72d6dfa797befd4a5a6a7410c085a0825854ffb74517e9c111c3d48", {"sep": ["-", "_"], "n": 3, "head": "711a6108ba2ce6ca93dd47d6817f2361db10d8ab6eec89460b2dfc2c325efabe"}),
    ("process", "hash", "af72e030992672530dc7d15a42980f80f1cf0e07a1692e86720c243ced1bd8d5", {"sep": ["-", "_"], "n": 2, "head": "a0cecbead27dfb0d580737b7a77b2a0864ab7695ff8bf80b5766c3e9e493797b"}),
    ("process", "hash", "63b07341e5fbcaab2e80f4820c07f029e46323b1a55d1e9cf20d13708663203d", {"sep": ["-", "_"], "n": 3, "head": "a0cecbead27dfb0d580737b7a77b2a0864ab7695ff8bf80b5766c3e9e493797b"}),
    ("process", "hash", "bc7c9cb9a8e1f74251d271ce69ea91ed99109d4241316d8e88c4f1c6f9f0fc27", {"sep": ["-", "_"], "n": 3, "head": "a0cecbead27dfb0d580737b7a77b2a0864ab7695ff8bf80b5766c3e9e493797b"}),
    ("process", "hash", "00682e1d3c1187c6ebaff0820485a2682b0d1f520fcc942787016a96fd60e90f", {"sep": ["-", "_"], "n": 3, "head": "872491a30d60d598962de6e7b834ab76b2aa65fbab102c6ebaaae6acdc238822"}),
    ("process", "hash", "16275f66ef64f596d57218ead3188ca9813999080a591b9c999e611602589cd0", {"sep": ["-", "_"], "n": 2, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "b2213144493a0f61a2024e00c23444326384091c2f016112b0629133b3c97663", {"sep": ["-", "_"], "n": 3, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "06515e791cfcc258edfd50eaa5b5922b501bce0dbd2e91af06015472942609a0", {"sep": ["-", "_"], "n": 3, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "c8d6cd85f0ba05b4fd6e3142bcd788d527796e1ad9f9e7c5e8d3543359ae2aa4", {"sep": ["-", "_"], "n": 2, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "bff91d6792ea61bded14447102afccbbdfd87df4fe74402445599560913a3c41", {"sep": ["-", "_"], "n": 2, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "30843b559e32e8eb0487ba344a4a5a5148959f5cc7924635874ba14c7bca3025", {"sep": ["-", "_"], "n": 2, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "79d33728507dab3c0d82fb6c5d919bbc58852f61b14a8dbd179a3635ee33cd3d", {"sep": ["-", "_"], "n": 2, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "66d0aa694a3c7cead1b1d1574fb0a747e35ee2de2e5d5bda905e651892927866", {"sep": ["-", "_"], "n": 2, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "2729ef6f42f6a94a1eb9f841916862db2d4f9fc7be17e9c34bc72792055633ef", {"sep": ["-", "_"], "n": 3, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "f9a3ffa441fa7c7e79752349901e1d5a8944575aca4560f537c1d1f0365b181c", {"sep": ["-", "_"], "n": 4, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "b55043211122b3a34f19f70a6106a88785634b50200ed051f5161de34d4b32f1", {"sep": ["-", "_"], "n": 3, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "01cbc168b8c5e461022a34f5590737be121bb638ee1ec04c2db89f38847ecd23", {"sep": ["-", "_"], "n": 3, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "a1dc2d2b031b3ad196ea18e6d4af991174633c7d799078826d91b7036a0c9d0a", {"sep": ["-", "_"], "n": 3, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "5dacc4747cc26f679a986ee02a9a5cfcce50460068e5676eda6da6e8f149d49c", {"sep": ["-", "_"], "n": 3, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "4ac1078a6c57a2e604fed64cefebf9cbf464f2ad3f44f85f4eed024b4da91d5c", {"sep": ["-", "_"], "n": 2, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "ca6f1179e54f24f5410d332a951026f7056333f4e6346311b725eb4871a5ce3c", {"sep": ["-", "_"], "n": 2, "head": "2e553bd34105c92fadd21abcbd4a50e5fb9d1d61d4a0d63b845e1e038e3d213c"}),
    ("process", "hash", "fa4446f24c1d31b9b3ba49920b589a02e92f061a801629084aa8a15af338b500", {"sep": ["-", "_"], "n": 3, "head": "68b076be5ba974795f927bf1aa53b0dc2991b90d8d535a9dbaf33bad74a2d3a3"}),
    ("process", "hash", "327dfd6b711f012c844edec638911d949ed39abdf0af6a606b204f21a628e18c", {"sep": ["-", "_"], "n": 3, "head": "68b076be5ba974795f927bf1aa53b0dc2991b90d8d535a9dbaf33bad74a2d3a3"}),
    ("process", "hash", "b48e007a7eff97bc6a21b5b06926eb787aeb1a9d852b26af0f6a8a39065847a9", {"sep": ["-", "_"], "n": 3, "head": "68b076be5ba974795f927bf1aa53b0dc2991b90d8d535a9dbaf33bad74a2d3a3"}),
    ("process", "hash", "3de73b7fccafc98caba4967eedf5601205e9ef9de7e6dd2b12ca0c3c3cea3d40", {"sep": ["-", "_"], "n": 2, "head": "a4d451ec23463726f72c43d64c710968f6b602cd653b4de8adee1b556240a829"}),
    ("process", "hash", "8a61bfb197367bfd445eac5ab5a8e3c6217368cfa31c18fc8da7c9a243f1746b", {"sep": ["-", "_"], "n": 3, "head": "a4d451ec23463726f72c43d64c710968f6b602cd653b4de8adee1b556240a829"}),
    ("process", "hash", "852c0ea1cbd460d7ae7cdec572be4ef19ee4d54cf425618a7e212b89e45687cd", {"sep": ["-", "_"], "n": 2, "head": "49819657d919dc82363519b6f64da1502431fdc9315399b7d42df333bf40ba4a"}),
    ("process", "hash", "9240b3d3cddc45fda939de0bf535cab9831ffffbf97aa0f04a1777f7cf4c14e8", {"sep": ["-", "_"], "n": 3, "head": "e8bc163c82eee18733288c7d4ac636db3a6deb013ef2d37b68322be20edc45cc"}),
    ("process", "hash", "56307f7045ed9575a21f3f6ce0abba5915df804b38845343e838503c35fd1962", {"sep": ["-", "_"], "n": 2, "head": "48045475b3d0281e17cb080c20a337fa8d9e264b55426ef4a3e03c42861cc0fd"}),
    ("process", "hash", "690c29a1e8ef0af890b77a5137e5a0142482711adf703c3b6c93c69e297983eb", {"sep": ["-", "_"], "n": 2, "head": "5a9cb6b54ea56d52a69891af8c21afb73d3841611bbabb5d7c61312d81e6e041"}),
    ("process", "hash", "3af56c32f3d61a931562f00f78aa5b1cb3a4bfe57b2c144ff7c197c5f1b5aa01", {"sep": ["-", "_"], "n": 3, "head": "5a9cb6b54ea56d52a69891af8c21afb73d3841611bbabb5d7c61312d81e6e041"}),
    ("process", "hash", "ab2bde606c0c7eb48349c7075d31d066143354555f9d41fdc040578113df2f8b", {"sep": ["-", "_"], "n": 3, "head": "5a9cb6b54ea56d52a69891af8c21afb73d3841611bbabb5d7c61312d81e6e041"}),
    ("process", "hash", "10473aa6bc8b3489294a77d1ec659339a06a0d727d30be4cc03c2d4dfc11f30d", {"sep": ["-", "_"], "n": 4, "head": "5a9cb6b54ea56d52a69891af8c21afb73d3841611bbabb5d7c61312d81e6e041"}),
    ("process", "hash", "e14e524d2b22e3c3b4384714fd809869ebb4b8bdf4be4cd312a94a833496af3b", {"sep": ["-", "_"], "n": 2, "head": "5a9cb6b54ea56d52a69891af8c21afb73d3841611bbabb5d7c61312d81e6e041"}),
    ("process", "hash", "cd3060c271b4add16fc0167e447077f481c6ee19ebdc308c7d8c13bb7e743c0f", {"sep": ["-", "_"], "n": 3, "head": "97a62ad21d79c01cceb7767952acec4fec86bfe909b06e5f3f6963365cf91ab8"}),
    ("process", "hash", "5973cd83f68af34097e578a25e072333bab8d4824bce11df049d5865e1f3c1f2", {"sep": ["-", "_"], "n": 2, "head": "61b563289c57c244505c10121108d5479ed0cc1a90a5fe07421b4f08c2b95c8b"}),
    ("process", "hash", "9a1bcdfc52b0338720ecc6c7a56139766a1e7ba33a8bf903832791f67e6ec5be", {"sep": ["-", "_"], "n": 2, "head": "61b563289c57c244505c10121108d5479ed0cc1a90a5fe07421b4f08c2b95c8b"}),
    ("process", "hash", "64be9f0758dfcc5792761fbf83b2332f017896a9b75b2fccc04ee4718ba4f9f1", {"sep": ["-", "_"], "n": 3, "head": "61b563289c57c244505c10121108d5479ed0cc1a90a5fe07421b4f08c2b95c8b"}),
    ("process", "hash", "152d6bf8220517c2951e1a51841e87833f9e1849b19c984b42ff50cd8cab1759", {"sep": ["-", "_"], "n": 2, "head": "61b563289c57c244505c10121108d5479ed0cc1a90a5fe07421b4f08c2b95c8b"}),
    ("process", "hash", "d555a9bb6bc026089bd1900a6f8f3c02857db83178fa0ef4c930cf0041039901", {"sep": ["-", "_"], "n": 2, "head": "61b563289c57c244505c10121108d5479ed0cc1a90a5fe07421b4f08c2b95c8b"}),
    ("process", "hash", "f7c019317ad6512f67d93c44fd9d6ddf28f6e25bc7740c7a074411a16d6328a9", {"sep": ["-", "_"], "n": 2, "head": "61b563289c57c244505c10121108d5479ed0cc1a90a5fe07421b4f08c2b95c8b"}),
    ("process", "hash", "53a2067eabae9ec01467187fb5d6b1c24a3d6cf9cf5a27a02f3ba404250bfee1", {"sep": ["-", "_"], "n": 3, "head": "61b563289c57c244505c10121108d5479ed0cc1a90a5fe07421b4f08c2b95c8b"}),
    ("process", "hash", "44e06514b8cc2a5030ada98e3d74adceb776fcf5c63a4db305bf43f3a382aef4", {"sep": ["-", "_"], "n": 2, "head": "61b563289c57c244505c10121108d5479ed0cc1a90a5fe07421b4f08c2b95c8b"}),
    ("process", "hash", "cd4329c1c38f366cbb5cdd6017e590f61f2c8e9d490eb636b3d8883cf369a095", {"sep": ["-", "_"], "n": 2, "head": "d53dc0058f4fc778c354197ec84cce92621463cc48885fe5b56959df7fa2297f"}),
    ("process", "hash", "fe8631e7f0f23adba4c89d4810d629e6723ac60f139afe2541de3f6ae4de234a", {"sep": ["-", "_"], "n": 2, "head": "d53dc0058f4fc778c354197ec84cce92621463cc48885fe5b56959df7fa2297f"}),
    ("process", "hash", "723b30e0962f5ca64d6fc9b9255dc527b2176d593f31cea3167a89c94eb2faa2", {"sep": ["-", "_"], "n": 2, "head": "d53dc0058f4fc778c354197ec84cce92621463cc48885fe5b56959df7fa2297f"}),
    ("process", "hash", "b3ecf7710fd5c934ae7602417eca42b3eb63986b70bce0ddddcd37b22e863798", {"sep": ["-", "_"], "n": 3, "head": "d53dc0058f4fc778c354197ec84cce92621463cc48885fe5b56959df7fa2297f"}),
    ("process", "hash", "f355958056851c2093238caca5ffe4d809c5f019a191c21bc02c13a1e8166249", {"sep": ["-", "_"], "n": 3, "head": "d53dc0058f4fc778c354197ec84cce92621463cc48885fe5b56959df7fa2297f"}),
    ("process", "hash", "148468cadbf6ca5bbd8304ea022b742b19aca1c9ffb9a1922b77f24c72582f2b", {"sep": ["-", "_"], "n": 2, "head": "d53dc0058f4fc778c354197ec84cce92621463cc48885fe5b56959df7fa2297f"}),
    ("process", "hash", "3d16071fb19fdeecc187ceb8cbbe554225dccd05f77a4597f17a63267db6f485", {"sep": ["-", "_"], "n": 2, "head": "d53dc0058f4fc778c354197ec84cce92621463cc48885fe5b56959df7fa2297f"}),
    ("process", "hash", "a26cdc20663cc0e1b60cf9494f62c943135a537e02151fffd09a4f7efe64fe6d", {"sep": ["-", "_"], "n": 3, "head": "d53dc0058f4fc778c354197ec84cce92621463cc48885fe5b56959df7fa2297f"}),
    ("process", "hash", "3e9a9c651aa4df316c927ca9f2fa25e25387b3b0b16bb5764fc3df7387314573", {"sep": ["-", "_"], "n": 3, "head": "5d9e67086933c7798f9404aac570a741f3f50a898caf1fae4f869e59d8f595d7"}),
    ("process", "hash", "b1ed0b392b1ded3cab59592744379d244b4761bc6bbf0581cd7179f4d44259b0", {"sep": ["-", "_"], "n": 3, "head": "db644eceac0e600cb0e3110c6d2c27ec1edfd1613db981a6fa20f0b0468cd6be"}),
    ("process", "hash", "bcf469f1bc141867dbc4658b2074441d1dd192480aa1b6ad0c7ca651ce6ab81a", {"sep": ["-", "_"], "n": 3, "head": "db644eceac0e600cb0e3110c6d2c27ec1edfd1613db981a6fa20f0b0468cd6be"}),
    ("process", "hash", "1b4f6685634b5f86a30fb4d361aa194de87d8baca181a8e0c566646a1943c9a5", {"sep": ["-", "_"], "n": 2, "head": "ab139cf13a42286eb74c8e534d17bdb940350c9eeb29b9aaa78c821b3d6ce11f"}),
    ("process", "hash", "16f28a3250aa63711eb84147ff2bdbdc6d31801a3701a17d6b1d0fe9d5b7202d", {"sep": ["-", "_"], "n": 3, "head": "ab139cf13a42286eb74c8e534d17bdb940350c9eeb29b9aaa78c821b3d6ce11f"}),
    ("process", "hash", "ba5df0b64f8c086490f2d79feb66d45fcb5c5a81d624190badf53bcd6693c964", {"sep": ["-", "_"], "n": 3, "head": "ab139cf13a42286eb74c8e534d17bdb940350c9eeb29b9aaa78c821b3d6ce11f"}),
    ("process", "hash", "c68e38180d67190146a093b4d32294317fbac761bc5901ca5d3ee19e61353e40", {"sep": ["-", "_"], "n": 2, "head": "ab139cf13a42286eb74c8e534d17bdb940350c9eeb29b9aaa78c821b3d6ce11f"}),
    ("process", "hash", "4d05c3e9cae3dfc06211be00621a9b427be5de23973356a47dac9d359710ae49", {"sep": ["-", "_"], "n": 3, "head": "ab139cf13a42286eb74c8e534d17bdb940350c9eeb29b9aaa78c821b3d6ce11f"}),
    ("process", "hash", "28428da1403e54e6d8cf76f17df00925eb40de663b72ef5d47255ee319eb2e0a", {"sep": ["-", "_"], "n": 2, "head": "d98a034f5cb8a0e3f07d5d8b5dc3c43723ea8d8b95199e18a82c52e072eca4dd"}),
    ("process", "hash", "29e373af84edfaa43b24a57259360dbb9d1663759fe9e1935614f8b798467156", {"sep": ["-", "_"], "n": 2, "head": "d98a034f5cb8a0e3f07d5d8b5dc3c43723ea8d8b95199e18a82c52e072eca4dd"}),
    ("process", "hash", "c818c7973a2fe60450fd33f3e36789557b2b7b6ced9f7b79c126063d51cbf4a6", {"sep": ["-", "_"], "n": 3, "head": "d98a034f5cb8a0e3f07d5d8b5dc3c43723ea8d8b95199e18a82c52e072eca4dd"}),
    ("process", "hash", "dd22e3e752dbeac99b04506e2bf5ace9d054ac6ebe78631e58e98df04c823603", {"sep": ["-", "_"], "n": 2, "head": "70ba908aefd9f131ed9537df90a190ee5313ca1cc42e0c73713f150f0be2ce39"}),
    ("process", "hash", "9f5a397306ec449ff58e8b6d38c46aa9974f651f0c57b52730ebad3e32c46e92", {"sep": ["-", "_"], "n": 3, "head": "4b125ec99a74470517f9f51ac1a70eafa3dbd1ddd4bae409fe59a98ed720a3a4"}),
    ("process", "hash", "2d924580b2619b6fa0a5c4c60d45f2f4f1ecb462f3c032b9de7beacf6e51c9ef", {"sep": ["-", "_"], "n": 3, "head": "4b125ec99a74470517f9f51ac1a70eafa3dbd1ddd4bae409fe59a98ed720a3a4"}),
    ("process", "hash", "6f1dcbc2a28244b9e0679ccd8113fc87aa14fe55c1d03730895ae829291a3dfd", {"sep": ["-", "_"], "n": 4, "head": "4b125ec99a74470517f9f51ac1a70eafa3dbd1ddd4bae409fe59a98ed720a3a4"}),
    ("process", "hash", "04a9f99077bfc09872dc79486ddc2e6989902ca35f9c7b82e4685012531061d2", {"sep": ["-", "_"], "n": 3, "head": "4b125ec99a74470517f9f51ac1a70eafa3dbd1ddd4bae409fe59a98ed720a3a4"}),
    ("process", "hash", "0c96b2279a9e8c8a3ffd34b72ca279bbf723718e2d91536de4f6c5c3e97d0f73", {"sep": ["-", "_"], "n": 3, "head": "4b125ec99a74470517f9f51ac1a70eafa3dbd1ddd4bae409fe59a98ed720a3a4"}),
    ("process", "hash", "938586ff812249c7d486e8802195c273db5dca344638387cb304eb0bca1066a3", {"sep": ["-", "_"], "n": 3, "head": "4b125ec99a74470517f9f51ac1a70eafa3dbd1ddd4bae409fe59a98ed720a3a4"}),
    ("process", "hash", "8dea446c17e1cce0acbf02e3b3a9a877ddd4040cffb907f07cf447b6db2c6830", {"sep": ["-", "_"], "n": 3, "head": "4b125ec99a74470517f9f51ac1a70eafa3dbd1ddd4bae409fe59a98ed720a3a4"}),
    ("process", "hash", "c1e49fdc7b9640ac3f387ddb87dcc233dd13e7d238ea8fa21ef859b71d793a86", {"sep": ["-", "_"], "n": 3, "head": "4b125ec99a74470517f9f51ac1a70eafa3dbd1ddd4bae409fe59a98ed720a3a4"}),
    ("process", "hash", "aa77b4c33f51411efec8042cd2c9eb64255173c941c3a4011b2d4a513416ee2c", {"sep": ["-", "_"], "n": 2, "head": "4b125ec99a74470517f9f51ac1a70eafa3dbd1ddd4bae409fe59a98ed720a3a4"}),
    ("process", "hash", "a9c5d4f87f55cb20ccde0648f8f6ac9d10907c16aa531e10606d319c340f4007", {}),
    ("process", "hash", "f9dc41d064b2718e3c1aa275e6c1fe2082e889ca585b739caad79d424d7acde2", {}),
    ("process", "hash", "d8c52d342309e3960ec29cf22c7e75e91dfdf5fff8444eb527a1e0c3771e30bf", {}),
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
