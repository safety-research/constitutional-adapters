"""No vault credential is ever read from a file, read from the environment
outside ``sisteer.vault``, or written into the environment.

Every API key this repository uses comes from :mod:`sisteer.vault` — the
credential's own environment variable, else its item through the secret-manager
CLI that ``SISTEER_SECRET_CMD`` names. This test is a SOURCE-level audit rather than a behavioural one
on purpose: the failure it guards against is a NEW call site, not a behaviour
change in an existing one, and a behavioural test only ever covers the sites
someone remembered to write it for.

WHAT WENT WRONG WITHOUT IT. Every hub read its key from a ``~/.config`` file
and each invented its own handling for a missing one, so removing those files
for the vault-only rule broke the repository in three different directions at
once, all of them quiet:

* ``eval.api_keys.make_client("lowprio")`` built a bare SDK client, which
  picked ``ANTHROPIC_API_KEY`` out of the environment — the HIGH PRIO key. So
  every Low Prio caller silently ran on High Prio, which is the org hop the
  routing rule exists to forbid. The same bare client inherited
  ``ANTHROPIC_BASE_URL``, which under a session manager can point at a local gateway,
  so those requests never reached the API at all.
* ``judge_batch`` left ``api_key=None`` and the SDK did the same thing, so
  Batch API work ran on High Prio — against the pool rule that keeps batch
  off the live orgs.
* ``capeval.judging`` raised instead, so the anthropic judge backend could not
  run at all.

WHAT THE FIRST VERSION OF THIS AUDIT MISSED (2026-09-07). It scanned ``*.py``
under ``src/`` and ``scripts/`` for credential PATH STRINGS, and three live
residues sat outside that net on the day the vault landed:

* ``scripts/realign_judge.py`` had an optional ``--key-file`` flag with NO
  default path — so no banned string appeared — whose handling read the file
  and wrote its contents into ``ANTHROPIC_API_KEY`` in ``os.environ``: a
  credential file read AND a write of the key into the process environment,
  inherited by every child. (The write did not even reach the judge, which
  had long resolved its own key; the coordination post mislabelled the org.)
* ``.nsr-run/judge_one.sh``, a tracked shell driver, ``cat``-ed the key file
  by its banned path — in a file type the audit never opened. Two docs named
  the same path, and stale prose is how someone recreates the file.
* ``scripts/grokmt_analyze.py`` built its own client from a direct
  ``environ.get`` of ``ANTHROPIC_API_KEY`` and labelled the run "low-prio".
  That variable IS the High Prio key, so this was the identical silent
  misroute ``make_client("lowprio")`` had — as a READ rather than a file.

WHAT THE SECOND VERSION GOT WRONG (review, same day). Its ``ALLOWED`` set —
designed for the path-string sweep, so the module that documents the rule
can name the old path — was applied to EVERY check, so
``src/sisteer/eval/api_keys.py`` was exempt from the read, write and
secret-manager-CLI checks too. The reviewer verified it by injection: a direct
read, a write and a direct secret-manager CLI call appended to ``api_keys.py`` all
passed, while the identical read appended to ``scripts/judge_sonnet.py`` was
caught. ``api_keys.py`` is the site of the original misroute, so a "CI
convenience" fallback to ``ANTHROPIC_API_KEY`` in ``make_client`` — the
2026-09-07 bug itself — would have sailed through, and this docstring's
"exactly one sanctioned reader" was false at the file where it mattered most.

So there are two nets with DIFFERENT scopes, and the difference is the point:

* The PATH-STRING sweep opens every TRACKED file (``git ls-files``, not a
  directory walk — an untracked scratch file is not the repository, and a
  peer's cannot fail your run) except a denylist of binary / generated
  suffixes, and skips :data:`ALLOWED`: the module that documents the rule,
  the docstring that records the old code, and the tests that pin the strings
  they forbid — and, for the path-string sweep alone, the files under
  :data:`ALLOWED_DIRS`: verbatim provenance copies that only RECORD a retired
  path. Same for the ``~/.config`` read check (:data:`ALLOWED` only).
* The CODE-PATTERN checks cover every ``.py`` under ``src/`` and ``scripts/``
  EXCEPT ``sisteer/vault.py`` — the one sanctioned reader — and nothing else
  is exempt. They fail on the shapes the residues took, whatever the path: a
  file-taking key flag, a write of a credential variable into the process
  environment, a direct read of an Anthropic org variable, and a read of the
  secret-manager CLI's setting (``SISTEER_SECRET_CMD``: with no default
  command, a call has to read it). Repo code has exactly one sanctioned reader of
  those variables, :func:`sisteer.vault.resolve_secret`, and no writer at
  all; :func:`test_api_keys_is_not_exempt_from_the_code_pattern_checks`
  replays that injection so that stays true.

WHAT A SOURCE REGEX CAN AND CANNOT SEE. These checks match LITERAL shapes:
the variable name spelled out in quotes next to ``environ`` / ``getenv`` /
``putenv`` / ``update`` / ``|=``. The known residual — the near-misses a
peer could still write — is a constant-name form (``environ.get(KEY_ENV)``
with ``KEY_ENV = "ANTHROPIC_API_KEY"`` elsewhere), an f-string, or a
concatenated name. No regex over one line catches those, and pretending to
would be the overclaim above in a new coat; :func:`test_known_residual_shapes_are_not_claimed`
records the boundary. Two consequences for writing code here: name the org
and let ``sisteer.vault`` resolve it (there is then nothing for the regex to
miss), and do not quote a banned shape VERBATIM in a source comment or
docstring — describe it in words — because prose false positives are
accepted rather than special-cased.

The READ check covers the ``ANTHROPIC_API_KEY`` family only (exactly the
three org variables), deliberately. Three direct reads of
``OPENROUTER_API_KEY`` exist and each has a stated reason that needs a
decision rather than a silent exemption here: ``scripts/modal/grokmt_served.py``
reads it at import for the Modal container (no vault CLI there; the secret
arrives in the container env — the non-raising import path of 2825400c) and
again inside the container's wire check, and ``scripts/jailbreak_run.py``
reads a USER-NAMED variable (``--attacker-api-key-env``, default
``OPENROUTER_API_KEY``). Widening the read check to OpenRouter means
resolving those three first. The WRITE check covers OpenRouter too: nothing
may write it.

CLASS-L BEARERS ARE NOT CREDENTIALS. A per-launch bearer token minted locally
for a local vLLM pool (``attacker_pool``'s ``api_key`` file, the served
proxy's ``server_auth.json``) is not a vault secret: it is generated for one
process tree and dies with it. Those paths are deliberately not covered here;
:data:`CLASS_L_FILE_FLAGS` lists the ONE module whose file-taking flag hands
out such a bearer — exempt from the file-flag check only — so an addition is
a decision rather than an accident.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOTS = ("src", "scripts")

#: The one sanctioned reader of a credential variable and the one sanctioned
#: caller of the secret-manager CLI. The ONLY file the code-pattern checks skip.
VAULT = "src/sisteer/vault.py"

#: Suffixes the tracked sweep does NOT open — binary or generated. Every
#: other tracked file is opened, whatever its suffix or lack of one. The
#: content check in :func:`_tracked_inventory` fails the suite if a binary
#: slips this list, so the list is kept honest by the tree itself.
BINARY_OR_GENERATED = (
    ".png", ".jpg", ".jpeg", ".gif", ".pdf",
    ".pt", ".pth", ".safetensors", ".npz", ".npy", ".pkl", ".bin",
    ".whl", ".zip", ".gz", ".tar", ".lock",
    ".eval",   # inspect-ai's zip-packed log (tests/fixtures/tinycap_verification/unscored_12.eval — a synthetic fixture)
    ".parquet",   # the committed paper frame's tables (paper/snapshot/snapshots/<id>/*.parquet)
)
#: A split archive's numbered part (``paper_frame.sqlite.gz.00`` … ``.05``): the
#: LAST suffix is the part number, so the ``.gz`` rule above never sees it.
#: Matched on the file NAME, lower-cased.
_SPLIT_PART = re.compile(r"\.(?:gz|tar|zst)\.\d{2,}$")
#: A path segment that marks generated notebook state.
GENERATED_DIRS = (".ipynb_checkpoints",)
#: Directory PREFIXES (trailing slash required, as for :data:`ALLOWED_DIRS`)
#: whose tracked files are GENERATED data by construction and are not opened
#: by the sweep: the committed paper frame (``10_figures.sh --export-snapshot``)
#: — parquet tables, a split sqlite archive and its MANIFEST. Its
#: one text file is covered by the build's own leak refusal
#: (``snapshot._refuse_manifest_leak``) and the identifier scan, not by this
#: sweep. The frame was committed without running
#: this test, and 36 binaries sat under suffixes the sweep opened.
GENERATED_TREES = ("paper/snapshot/snapshots/",)

#: Fragments that named a vault credential ON DISK. These are PATH-shaped on
#: purpose: ``wandb_key`` alone is also a legitimate W&B panel name, so only
#: ``/wandb_key`` (a filesystem path) is the defect.
BANNED = (
    "anthropic-key",
    "openrouter-key",
    "/wandb_key",
    ".config/anthropic",
    ".config/openrouter",
)

#: Files that may legitimately mention a banned PATH STRING — the module that
#: documents the rule, the docstring that records what the old code did, and
#: the tests that pin the fix by naming the strings they forbid. Applies to
#: the path-string sweep and the ``~/.config`` read check ONLY; the
#: code-pattern checks do not consult it (module docstring, "second version").
#: NOT the place for a doc or a script that still USES the string: fix those.
#: (The site module's one, narrower exemption from the ``~/.config`` read
#: check is :data:`SITE_CONFIG_READER`, not an entry here.)
ALLOWED = {
    VAULT,
    "src/sisteer/eval/api_keys.py",       # docstring: what the old code did
    "tests/test_no_credential_files.py",
    "tests/test_judge_backend_openrouter.py",   # pins the banned strings
    "tests/test_petri_auditor_fallback.py",     # pins the banned strings
}

#: Directory PREFIXES whose tracked files may name a banned PATH STRING because
#: they only RECORD it: verbatim provenance copies of retired, unversioned
#: export builders captured 2026-09-16 (``scripts/r2_export/``). Byte identity
#: with the live files is the point of those copies, so they are never edited;
#: they are never imported or executed from the repo (their README says so);
#: and the paths they name were retired by the vault rule. Same scope as
#: :data:`ALLOWED` minus the ``~/.config`` read check: the PATH-STRING sweep
#: ONLY. The files stay in the tracked inventory (opened, binary-checked), and
#: the code-pattern checks still walk every ``.py`` under these directories
#: (:func:`test_allowed_dirs_do_not_shape_the_code_pattern_scope`). Trailing
#: slash required, so a prefix cannot shadow a sibling directory.
ALLOWED_DIRS = (
    "scripts/r2_export/",
)


def _in_allowed_dir(rel: str) -> bool:
    return rel.startswith(ALLOWED_DIRS)

#: Class-L (module docstring): the one module whose file-taking flag names a
#: LOCALLY MINTED per-launch bearer for a local vLLM pool, not a vault
#: credential. Exempt from the file-flag check ONLY; every other check still
#: applies to it.
CLASS_L_FILE_FLAGS = {
    # `--api-key-file`: the pool's own 0600 bearer (API_KEY_ENV = VLLM_API_KEY),
    # minted by `attacker_pool mint` for one Slurm job and dead with it.
    "src/sisteer/eval/attacks/attacker_pool.py",
}

#: Reading ANY of these from a path under a user's config directory is a
#: credential file read, whatever the filename.
_CONFIG_READ = re.compile(
    r"""\.config[/"']""", re.VERBOSE)

#: The site configuration's ONE reader, and the one ``~/.config`` path it may
#: name: ``sisteer.site`` resolves the site config (``$SISTEER_CONFIG``, else
#: ``$SISTEER_SITE_DIR/config.env``, else ``config.env`` in this directory).
#: It is not a credential file: the site config holds settings (hosts, roots,
#: labels, the command that fetches a secret), never a secret value, and every
#: credential still comes from the vault module. Exempt from the ``~/.config``
#: read check ONLY, and only for this exact directory in this exact file: any
#: other ``~/.config`` path in the module fails, and so does this directory
#: named in any other file, which gets its site values through the module
#: (:func:`test_the_site_config_exemption_is_one_path_in_one_file`).
SITE_CONFIG_READER = ("src/sisteer/site.py", "~/.config/sisteer")
#: The directory as a whole path, never the prefix of a longer one
#: (``~/.config/sisteer2``, or a file under it).
_SITE_CONFIG_DIR = re.compile(re.escape(SITE_CONFIG_READER[1]) + r"(?![\w./-])")


def _config_read_hits(files) -> list:
    """The ``~/.config`` read check over ``(rel, source)`` pairs: every file
    but :data:`ALLOWED`, the site module with its one directory taken out."""
    hits = []
    for rel, src in files:
        if rel in ALLOWED:
            continue
        if rel == SITE_CONFIG_READER[0]:
            src = _SITE_CONFIG_DIR.sub("", src)
        if _CONFIG_READ.search(src):
            hits.append(f"{rel}: reads from a ~/.config path")
    return hits

#: An argparse flag that takes a KEY FILE — with or without its leading
#: dashes, because ``attacker_pool`` builds flags as ``f"--{name}"`` from bare
#: names and a future hub could do the same. Quote-delimited so a docstring
#: that recounts the old flag in backticks is history, not a violation.
_FILE_FLAG = re.compile(
    r"""["'](?:--)?(?:api-|anthropic-)?key-file["']""")

#: The three Anthropic org variables, EXACTLY — nothing wider, so a name that
#: merely starts with one (a hypothetical ``ANTHROPIC_API_KEYS_FILE``) is not
#: a hit. :func:`test_org_variable_alternation_matches_the_vault_exactly`
#: keeps this in step with :data:`sisteer.vault.ANTHROPIC_ORGS`.
_ORG = r"ANTHROPIC_API_KEY(?:_LOWPRIO|_BATCH)?"
#: Every credential variable repo code must never WRITE.
_CRED = r"(?:ANTHROPIC_API_KEY(?:_LOWPRIO|_BATCH)?|OPENROUTER_API_KEY)"
_Q = r"""["']"""

#: A write of a credential variable into the process environment. Four
#: shapes: ``environ["X"] = …``, ``environ.setdefault("X", …)``,
#: ``os.putenv("X", …)``, and a mapping write — ``environ.update({… "X": …})``
#: or ``environ |= {… "X": …}`` — whose dict literal names the variable.
#: ``environ`` with or without the ``os.`` prefix. NOT matched, deliberately:
#: reads, comparisons, ``del``, and a dict literal that is NOT being merged
#: into ``environ`` (a CHILD process's env for a third-party tool that reads
#: only that name is a per-command hand-off, not a write into this process).
_W_SUBSCRIPT = r"(?:os\.)?environ\s*\[\s*" + _Q + _CRED + _Q + r"\s*\]\s*=(?!=)"
_W_SETDEFAULT = r"(?:os\.)?environ\.setdefault\(\s*" + _Q + _CRED + _Q
_W_PUTENV = r"os\.putenv\(\s*" + _Q + _CRED + _Q
_W_MAPPING = (r"(?:os\.)?environ\s*(?:\.update\s*\(|\|=)\s*\{[^}]*"
              + _Q + _CRED + _Q + r"\s*:")
_ENV_WRITE = re.compile("|".join((_W_SUBSCRIPT, _W_SETDEFAULT, _W_PUTENV,
                                  _W_MAPPING)))

#: A direct READ of an Anthropic org variable: ``environ.get("X")``,
#: ``environ["X"]`` (not after ``del`` — a scrub is not a read),
#: ``os.getenv("X")`` or a bare ``getenv("X")``. The only sanctioned reader
#: is ``vault.resolve_secret``; everything else asks the vault module for an
#: ORG and gets that org's key, which is what makes "asked for Low Prio, got
#: High Prio" unreinstateable. A subscript WRITE also matches here (it is a
#: violation under both names).
_R_ACCESS = (r"(?<!\bdel\s)(?<!\bdel\sos\.)(?:os\.)?environ\s*(?:\.get\s*\(|\[)\s*"
             + _Q + _ORG + _Q)
_R_GETENV = r"\bgetenv\s*\(\s*" + _Q + _ORG + _Q
_ENV_READ = re.compile("|".join((_R_ACCESS, _R_GETENV)))

#: A read of the secret-manager CLI's setting, not a mention: the variable's name as a quoted literal (there is no default
#: command, so running the CLI means reading the setting), so a docstring that explains where keys come from is not a
#: violation.
_SECRET_CMD_READ = re.compile(r"""["']SISTEER_SECRET_CMD["']""")


# ------------------------------------------------------------------ scopes


def _code_files(root: pathlib.Path = REPO):
    """Every ``.py`` under ``src/`` and ``scripts/`` EXCEPT :data:`VAULT` —
    the code-pattern checks' scope. :data:`ALLOWED` does NOT apply here.

    A working-tree walk (a superset of the tracked list on a clean checkout),
    on purpose: this is the net that must catch a new call site BEFORE it is
    committed. ``root`` is a parameter so the injection test can run the
    identical walk over a copied tree.
    """
    for r in ROOTS:
        for p in sorted((root / r).rglob("*.py")):
            rel = p.relative_to(root).as_posix()
            if rel == VAULT:
                continue
            yield rel, p.read_text()


class _Inventory:
    """What the tracked sweep did with every ``git ls-files`` entry."""

    def __init__(self):
        self.tracked = 0
        self.opened: list = []          # (rel, text)
        self.denylisted: list = []      # rel, by suffix / generated dir
        self.missing: list = []         # tracked but deleted in this tree
        self.undeclared_binary: list = []   # NUL bytes, suffix not denylisted


def _tracked_inventory() -> _Inventory:
    """Open every tracked file except the binary / generated denylist. A git
    failure is a test failure, never an empty sweep that passes."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "ls-files", "-z"], text=True,
            stderr=subprocess.PIPE, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.fail(f"git ls-files failed, so the tracked-tree sweep cannot "
                    f"run (it must never pass by scanning nothing): {exc}")
    inv = _Inventory()
    rels = sorted(r for r in out.split("\0") if r)
    inv.tracked = len(rels)
    assert inv.tracked > 100, "git ls-files returned suspiciously few files"
    for rel in rels:
        p = REPO / rel
        if not p.is_file():
            inv.missing.append(rel)
            continue
        if (p.suffix.lower() in BINARY_OR_GENERATED
                or _SPLIT_PART.search(p.name.lower())
                or any(seg in GENERATED_DIRS for seg in p.parts)
                or rel.startswith(GENERATED_TREES)):
            inv.denylisted.append(rel)
            continue
        data = p.read_bytes()
        if b"\0" in data[:8192]:
            inv.undeclared_binary.append(rel)
            continue
        inv.opened.append((rel, data.decode("utf-8", errors="replace")))
    return inv


# ------------------------------------------------------------------ checks


def _located(rel: str, src: str, m: re.Match) -> str:
    return f"{rel}:{src.count(chr(10), 0, m.start()) + 1}: {m.group(0).strip()}"


def _flag_hits(files) -> list:
    return [_located(rel, src, m) for rel, src in files
            if rel not in CLASS_L_FILE_FLAGS
            for m in _FILE_FLAG.finditer(src)]


def _read_hits(files) -> list:
    return [_located(rel, src, m) for rel, src in files
            for m in _ENV_READ.finditer(src)]


def _write_hits(files) -> list:
    return [_located(rel, src, m) for rel, src in files
            for m in _ENV_WRITE.finditer(src)]


def _secret_cli_callers(files) -> list:
    return [rel for rel, src in files if _SECRET_CMD_READ.search(src)]


# ------------------------------------------------- PATH-STRING sweep (tracked)


def test_no_tracked_file_names_a_credential_file():
    """Every tracked file the denylist lets through — code, shell, sbatch,
    config, docs, licences, files with no suffix at all. The 2026-09-07
    residues were a tracked ``.sh`` and two ``.md`` files."""
    inv = _tracked_inventory()
    hits = [f"{rel}: names {frag!r}"
            for rel, src in inv.opened
            if rel not in ALLOWED and not _in_allowed_dir(rel)
            for frag in BANNED if frag in src]
    assert not hits, (
        "a vault credential is being named as a FILE again — keys come from "
        "sisteer.vault (env var, else the secret-manager item); fix the file, do "
        "not allowlist it unless it only RECORDS the rule:\n  "
        + "\n  ".join(hits))


def test_every_tracked_file_is_opened_or_declared_binary():
    """The denylist is the ONLY thing that keeps a tracked file out of the
    sweep, and it must be stated: a file with NUL bytes whose suffix is not
    denylisted fails here with the suffix to add. Missing files (tracked but
    deleted in this tree) are reported, not silently skipped."""
    inv = _tracked_inventory()
    assert not inv.undeclared_binary, (
        "binary content under a suffix the sweep opens — add the suffix to "
        "BINARY_OR_GENERATED (or the dir to GENERATED_DIRS / GENERATED_TREES):\n  "
        + "\n  ".join(inv.undeclared_binary))
    assert inv.tracked == (len(inv.opened) + len(inv.denylisted)
                           + len(inv.missing)), "inventory does not add up"
    assert not inv.missing, "tracked files absent from the tree: " + ", ".join(inv.missing)


def test_the_tracked_sweep_opens_the_residue_file_types_and_suffixless_files():
    """The sweep must actually open the kinds of file the first version
    missed, or a green run says nothing. Pins the residue types plus the
    long tail an allowlist of suffixes would have dropped."""
    inv = _tracked_inventory()
    suffixes = {pathlib.Path(rel).suffix for rel, _ in inv.opened}
    for ext in (".sh", ".md", ".py", ".sbatch", ".yaml", ".json", ".tsv", ".txt"):
        assert ext in suffixes, f"tracked sweep opened no {ext} file"
    assert "" in suffixes, "tracked sweep opened no suffixless file (LICENSE…)"
    for ext in (".pkl", ".safetensors", ".png", ".pth", ".parquet"):
        assert not any(rel.endswith(ext) for rel, _ in inv.opened), ext


def test_the_paper_frame_is_declared_generated_and_never_opened():
    """The committed paper frame — 30 ``.parquet`` tables and
    the split ``paper_frame.sqlite.gz.NN`` parts — is binary under names the
    sweep used to open (``.parquet`` was not denylisted; a part's LAST suffix is
    its number, so the ``.gz`` rule never saw it). Declared three ways (the
    suffix, the split-part pattern, the generated tree) and every tracked file
    under the tree is denylisted, none opened."""
    inv = _tracked_inventory()
    frame = [rel for rel in inv.denylisted if rel.startswith(GENERATED_TREES)]
    assert any(rel.endswith(".parquet") for rel in frame), "no parquet table under the paper frame"
    assert any(_SPLIT_PART.search(rel) for rel in frame), "no split sqlite part under the paper frame"
    assert not [rel for rel, _ in inv.opened if rel.startswith(GENERATED_TREES)]
    assert all(t.endswith("/") for t in GENERATED_TREES), "a prefix without its slash can shadow a sibling"
    # the split-part pattern: numbered parts of a gz / tar / zst archive and nothing else
    for name in ("paper_frame.sqlite.gz.05", "frame.tar.00", "X.ZST.12"):
        assert _SPLIT_PART.search(name.lower()), name
    for name in ("notes.05", "archive.gz", "v1.2.gz.x", "a.gz.5", "gz.05"):
        assert not _SPLIT_PART.search(name.lower()), name


# ------------------------------------------- CODE-PATTERN checks (src, scripts)


def test_no_source_file_reads_from_a_user_config_directory():
    hits = _config_read_hits(_code_files())
    assert not hits, (
        "a source file reads from a user config directory; credentials come "
        "from sisteer.vault, site settings through sisteer.site (the site "
        "config's one reader), and nothing else belongs there:\n  "
        + "\n  ".join(hits))


def _planted_tree(root: pathlib.Path, files: dict) -> pathlib.Path:
    for rel, text in files.items():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text)
    return root


def test_the_site_config_exemption_is_one_path_in_one_file(tmp_path):
    """The planted half of the site module's exemption: the unmodified module
    passes; a new script or module that reads the site config directly fails
    (it gets site values through sisteer.site); and the module itself fails
    the moment it names any other ``~/.config`` path, a file under the site
    directory or a longer name sharing its prefix."""
    reader, site_dir = SITE_CONFIG_READER
    real = (REPO / reader).read_text()
    assert _SITE_CONFIG_DIR.search(real), f"{reader} no longer names {site_dir}: the exemption is stale"
    clean = list(_code_files(_planted_tree(tmp_path / "clean", {reader: real})))
    assert [rel for rel, _ in clean] == [reader] and _config_read_hits(clean) == []
    plants = {
        "scripts/plant_site_reader.py": f'CFG = os.path.expanduser("{site_dir}/config.env")\n',
        "src/sisteer/plant_default.py": f'DEFAULT_SITE_DIR = "{site_dir}"\n',
    }
    files = list(_code_files(_planted_tree(tmp_path / "new_reader", {reader: real, **plants})))
    assert sorted(h.split(":")[0] for h in _config_read_hits(files)) == sorted(plants)
    for i, extra in enumerate(('OTHER = "~/.config/sometool/settings"\n', f'UNDER = "{site_dir}/settings"\n',
                               f'LONGER = "{site_dir}2"\n')):
        files = list(_code_files(_planted_tree(tmp_path / f"widened{i}", {reader: real + extra})))
        assert _config_read_hits(files) == [f"{reader}: reads from a ~/.config path"], extra


def test_the_site_config_exemption_names_the_site_modules_own_default():
    """The exempt directory is the one the module actually defaults to, so
    the exemption moves with the module, never stays behind as a stale path."""
    from sisteer import site
    assert site.DEFAULT_SITE_DIR == SITE_CONFIG_READER[1]
    assert re.search(r'^DEFAULT_SITE_DIR = "' + re.escape(SITE_CONFIG_READER[1]) + '"$',
                     (REPO / SITE_CONFIG_READER[0]).read_text(), re.M)


def test_the_vault_module_is_the_only_secret_cli_caller():
    """One entry point, so the timeout, the typed error and the
    never-log-the-value discipline are decided in exactly one place. No file
    is exempt but the vault module itself."""
    callers = _secret_cli_callers(list(_code_files()))
    assert not callers, (
        "the secret-manager CLI's setting is read outside sisteer.vault: " + ", ".join(callers))


def test_no_source_file_takes_a_key_file_flag():
    """``--key-file`` / ``--api-key-file`` / ``--anthropic-key-file``: a flag
    that takes a credential FILE has no banned path string (the path is the
    user's), which is exactly how ``realign_judge`` slipped past the sweep.
    Class-L bearer flags (:data:`CLASS_L_FILE_FLAGS`) are the one exemption."""
    hits = _flag_hits(list(_code_files()))
    assert not hits, (
        "a script takes a credential FILE flag — resolve the org through "
        "sisteer.vault (--org, vault.anthropic_client) instead:\n  "
        + "\n  ".join(hits))


def test_no_source_file_writes_a_credential_into_the_environment():
    """A credential must never be written into the process environment by
    repo code: every child inherits it, and the write bypasses the org
    discipline (``realign_judge`` wrote a file's contents into High Prio's
    variable whatever org the file held). The ONE sanctioned reader is
    ``vault.resolve_secret``; there is no sanctioned writer."""
    hits = _write_hits(list(_code_files()))
    assert not hits, (
        "repo code writes a credential into os.environ — pass the key to "
        "the client directly (sisteer.vault.anthropic_client):\n  "
        + "\n  ".join(hits))


def test_no_source_file_reads_an_anthropic_org_variable_directly():
    """A direct read of an org variable outside ``sisteer.vault`` is a
    misroute waiting to happen: ``ANTHROPIC_API_KEY`` is High Prio and
    nothing else, and the reader has no way to say which org it meant
    (``grokmt_analyze`` read it and called it low-prio). Ask ``sisteer.vault``
    for the ORG."""
    hits = _read_hits(list(_code_files()))
    assert not hits, (
        "repo code reads an Anthropic org variable directly — the one "
        "sanctioned reader is vault.resolve_secret; build the client with "
        "vault.anthropic_client(<org>) / eval.api_keys.make_client(<org>):\n  "
        + "\n  ".join(hits))


def test_class_l_exemption_names_only_files_that_exist_and_mint_bearers():
    """The exemption is a decision about specific files: each must exist, and
    each must actually be a Class-L surface (mint a local bearer), or the
    entry is stale and should go."""
    for rel in CLASS_L_FILE_FLAGS:
        p = REPO / rel
        assert p.is_file(), f"CLASS_L_FILE_FLAGS names a missing file: {rel}"
        src = p.read_text()
        assert "VLLM_API_KEY" in src and "mint" in src, (
            f"{rel} is exempted as a Class-L bearer surface but does not "
            "look like one")


# ------------------------------------------ the scopes are what they claim


#: The injection probe, verbatim in shape: a direct read, a write, and a
#: direct call of the secret-manager CLI.
_INJECTIONS = (
    'key = os.environ.get("ANTHROPIC_API_KEY")\n',
    'os.environ["ANTHROPIC_API_KEY_LOWPRIO"] = key\n',
    'out = subprocess.check_output([os.environ["SISTEER_SECRET_CMD"], "lowprio"], text=True)\n',
)
_API_KEYS = "src/sisteer/eval/api_keys.py"
_CONTROL = "scripts/judge_sonnet.py"


def _copied_tree(root: pathlib.Path, appended: dict) -> pathlib.Path:
    """``api_keys.py``, the control file and the vault module copied under a
    fresh root, each with ``appended`` text added — so the walk under test is
    :func:`_code_files` itself, over a real tree, not a hand-built list."""
    for rel in (_API_KEYS, _CONTROL, VAULT):
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text((REPO / rel).read_text() + appended.get(rel, ""))
    return root


def test_api_keys_is_not_exempt_from_the_code_pattern_checks(tmp_path):
    """The review finding: ``ALLOWED`` had exempted ``api_keys.py`` — the
    site of the original misroute — from every code-pattern check. Replays
    the reviewer's injection: all three shapes appended to ``api_keys.py``
    must be caught by their check, the read appended to ``judge_sonnet.py``
    must be caught (the reviewer's control), and the same three shapes
    appended to the vault module must NOT be, because that file alone is the
    sanctioned reader and caller."""
    root = _copied_tree(tmp_path / "injected", {
        _API_KEYS: "".join(_INJECTIONS),
        _CONTROL: _INJECTIONS[0],
        VAULT: "".join(_INJECTIONS),
    })
    files = list(_code_files(root))
    rels = {rel for rel, _ in files}
    assert _API_KEYS in rels and _CONTROL in rels and VAULT not in rels
    reads, writes, callers = _read_hits(files), _write_hits(files), _secret_cli_callers(files)
    assert any(h.startswith(_API_KEYS + ":") for h in reads), reads
    assert any(h.startswith(_API_KEYS + ":") for h in writes), writes
    assert _API_KEYS in callers, callers
    assert any(h.startswith(_CONTROL + ":") for h in reads), reads
    assert not any(h.startswith(VAULT) for h in reads + writes), (reads, writes)
    assert VAULT not in callers


def test_the_unmodified_files_pass_every_code_pattern_check(tmp_path):
    """The control half: the same walk over UNMODIFIED copies reports nothing
    — including for the vault module, whose real body runs the secret-manager CLI
    and reads the variables, and which the walk must therefore exclude."""
    files = list(_code_files(_copied_tree(tmp_path / "clean", {})))
    assert {rel for rel, _ in files} == {_API_KEYS, _CONTROL}
    assert _read_hits(files) == [] and _write_hits(files) == []
    assert _secret_cli_callers(files) == [] and _flag_hits(files) == []


def test_allowed_does_not_shape_the_code_pattern_scope():
    """Only the vault module is missing from the code-pattern walk; every
    other ``ALLOWED`` entry under ``src/``/``scripts/`` is present in it."""
    rels = {rel for rel, _ in _code_files()}
    assert VAULT not in rels
    for rel in ALLOWED:
        if rel != VAULT and rel.split("/")[0] in ROOTS:
            assert rel in rels, rel
    assert _API_KEYS in rels


def test_allowed_dirs_are_prefixes_of_real_provenance_directories():
    """Each :data:`ALLOWED_DIRS` entry is a directory prefix (trailing slash,
    so ``scripts/r2_export/`` cannot shadow a sibling such as
    ``scripts/r2_export_v2/``), exists in the tree with the README that declares
    the copies' status, and holds tracked files — a stale entry would be a
    silently widened exemption, not a no-op."""
    inv = _tracked_inventory()
    tracked = {rel for rel, _ in inv.opened} | set(inv.denylisted)
    for d in ALLOWED_DIRS:
        assert d.endswith("/"), d
        assert (REPO / d / "README.md").is_file(), f"{d}: no README.md declaring the copies' status"
        assert any(rel.startswith(d) for rel in tracked), f"{d}: no tracked files"
    assert _in_allowed_dir("scripts/r2_export/petri/README.md")
    assert not _in_allowed_dir("scripts/r2_export_v2/README.md")
    assert not _in_allowed_dir("src/scripts/r2_export/x.py")


def test_allowed_dirs_do_not_shape_the_code_pattern_scope():
    """The directory exemption is for the path-string sweep ONLY: every tracked
    ``.py`` under an :data:`ALLOWED_DIRS` entry is still walked by the
    code-pattern checks (file-taking key flag, environment write, direct org
    read, secret-manager CLI call) exactly like any other script."""
    code = {rel for rel, _ in _code_files()}
    copied = [rel for rel, _ in _tracked_inventory().opened
              if _in_allowed_dir(rel) and rel.endswith(".py")]
    assert copied, "no tracked .py under ALLOWED_DIRS — the pin is vacuous"
    missing = sorted(rel for rel in copied if rel not in code)
    assert not missing, missing


def test_org_variable_alternation_matches_the_vault_exactly():
    """The read check names the org variables literally; this keeps that
    literal equal to what ``sisteer.vault`` actually reads, so a new org
    shows up here as a deliberate edit."""
    from sisteer import vault
    envs = {env for env, _item in vault.ANTHROPIC_ORGS.values()}
    assert envs == {"ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_LOWPRIO",
                    "ANTHROPIC_API_KEY_BATCH"}
    org = re.compile(_ORG)
    assert all(org.fullmatch(e) for e in envs)
    assert not org.fullmatch("ANTHROPIC_API_KEYS_FILE")
    assert not org.fullmatch("ANTHROPIC_API_KEY_OTHER")
    assert vault.OPENROUTER_ENV == "OPENROUTER_API_KEY"


# ------------------------------------------- the patterns catch what shipped


@pytest.mark.parametrize("line", [
    'ap.add_argument("--key-file", default=None,',
    "ap.add_argument('--api-key-file')",
    'ap.add_argument("--anthropic-key-file")',
    'for name in ("pool-name", "api-key-file", "job-id"):',   # bare, f"--{name}"
])
def test_file_flag_pattern_catches_each_shape(line):
    assert _FILE_FLAG.search(line), line


@pytest.mark.parametrize("line", [
    'os.environ["ANTHROPIC_API_KEY"] = key_path.read_text().strip()',  # the residue
    "os.environ['ANTHROPIC_API_KEY_LOWPRIO'] = value",
    'os.environ["ANTHROPIC_API_KEY_BATCH"]=value',
    'os.environ.setdefault("OPENROUTER_API_KEY", value)',
    'environ["OPENROUTER_API_KEY"] = value',            # from os import environ
    'os.putenv("ANTHROPIC_API_KEY", value)',
    'os.environ.update({"ANTHROPIC_API_KEY": key})',                 # F2
    'os.environ.update({"HF_HOME": "/vol/hf", "ANTHROPIC_API_KEY_LOWPRIO": k})',
    'os.environ |= {"OPENROUTER_API_KEY": k}',                       # F2
    'environ.update({\n    "ANTHROPIC_API_KEY_BATCH": key,\n})',    # multi-line
])
def test_env_write_pattern_catches_each_shape(line):
    assert _ENV_WRITE.search(line), line


@pytest.mark.parametrize("line", [
    'key = os.environ.get("ANTHROPIC_API_KEY")',                  # a read
    'if os.environ["ANTHROPIC_API_KEY"] == other:',              # a comparison
    'env = {**os.environ, "ANTHROPIC_API_KEY": key}',            # a child env
    'child_env = {"ANTHROPIC_API_KEY": key}; subprocess.run(cmd, env=child_env)',
    'os.environ.update({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"})',
    'os.environ["ANTHROPIC_BASE_URL"] = url',                    # not a credential
    'os.environ["VLLM_API_KEY"] = bearer',                       # Class-L bearer
    'del os.environ["ANTHROPIC_API_KEY"]',                       # a scrub
    'ap.add_argument("--api-key", default=None)',                # a value, not a file
    '"api_key_file": None,',                                     # underscored field
    'the old ``--key-file`` did both',                           # backticked history
])
def test_write_and_flag_patterns_ignore_reads_child_envs_scrubs_and_history(line):
    assert not _ENV_WRITE.search(line), line
    assert not _FILE_FLAG.search(line), line


@pytest.mark.parametrize("line", [
    'key = os.environ.get("ANTHROPIC_API_KEY")',                  # the residue
    "key = os.environ.get('ANTHROPIC_API_KEY_LOWPRIO', '')",
    'key = os.environ["ANTHROPIC_API_KEY_BATCH"]',
    'environ.get("ANTHROPIC_API_KEY")',                          # from os import environ
    'key = os.getenv("ANTHROPIC_API_KEY")',
    'key = getenv("ANTHROPIC_API_KEY")',                         # from os import getenv (F2)
])
def test_env_read_pattern_catches_each_shape(line):
    assert _ENV_READ.search(line), line


@pytest.mark.parametrize("line", [
    'base = os.environ.get("ANTHROPIC_BASE_URL")',               # not a credential
    'key = os.environ.get("VLLM_API_KEY", "")',                  # Class-L bearer
    'key = os.environ.get(key_env, "")',                         # user-named variable
    'env = {**os.environ, "ANTHROPIC_API_KEY": key}',            # a child env
    'from_env = (os.environ.get(env) or "").strip()',            # the vault reader itself
    'del os.environ["ANTHROPIC_API_KEY"]',                       # a scrub (F3)
    'del environ["ANTHROPIC_API_KEY_LOWPRIO"]',                  # a scrub, bare (F3)
    'os.environ.pop("ANTHROPIC_API_KEY", None)',                 # a scrub
    'monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)',    # a test scrub
    'path = os.environ.get("ANTHROPIC_API_KEYS_FILE")',          # wider name (F3)
    'other = os.environ.get("ANTHROPIC_API_KEY_OTHER")',         # not an org
    'the bare env ANTHROPIC_API_KEY is High Prio',               # prose
])
def test_env_read_pattern_ignores_other_variables_scrubs_and_prose(line):
    assert not _ENV_READ.search(line), line


@pytest.mark.parametrize("line", [
    'key = os.environ.get(KEY_ENV)',                             # constant-name form
    'key = os.environ.get(f"ANTHROPIC_API_KEY{suffix}")',        # f-string
    'key = os.environ.get("ANTHROPIC_API" + "_KEY")',            # concatenation
    'os.environ[KEY_ENV] = value',
])
def test_known_residual_shapes_are_not_claimed(line):
    """The documented boundary (module docstring): a one-line source regex
    does not see through a name, an f-string or a concatenation. These stay
    UNCAUGHT on purpose — listing them here is what keeps the claim honest,
    and the module docstring must keep naming them."""
    assert not _ENV_READ.search(line) and not _ENV_WRITE.search(line), line
    assert "constant-name form" in __doc__ and "f-string" in __doc__


def test_prose_that_quotes_the_shape_verbatim_is_flagged_by_design():
    """Chosen policy (F3): prose false positives are accepted, not special-
    cased. A source comment that quotes a banned shape verbatim IS flagged;
    describe it in words instead. The module docstring says so."""
    assert _ENV_READ.search('# it used to do os.environ.get("ANTHROPIC_API_KEY") here')
    assert "do not quote a banned shape VERBATIM" in __doc__
