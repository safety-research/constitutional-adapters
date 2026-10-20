#!/usr/bin/env python3
"""convert_config.py --root ROOT [--dry-run] [--json]  |  convert_config.py --root ROOT --restore BACKUP

The cutover's configuration conversion: ROOT/config.toml loses its
``modal`` table BEFORE the base fast-forwards and before any new-code registry invocation, because the new loader (C10) REFUSES a
configuration that still carries one. Standard library only (``tomllib``, Python >= 3.11): this runs under the OLD code's
interpreter, in the paused window, and must not import the registry package.

What it does, in order:
  1. reads ROOT/config.toml as bytes; no file, or a file without a ``modal`` table -> ``CONVERTED=0`` with the reason, exit 0,
     nothing written (the stage's two validations still run in window.sh);
  2. writes the BACKUP ROOT/state/config.toml.bak-<UTC stamp> (byte-identical, verified by sha256; a backup already at that
     name refuses, exit 3) -- the window's abort path restores it with --restore;
  3. removes the table TEXTUALLY: from its header line through the line before the next table header (``[`` at the start of a
     line) or the end of the file, every other byte untouched (comments, spacing, the other tables' order), writes the new file
     atomically (a temporary beside it, then os.replace), then re-parses it and requires the parsed document to equal the old
     one minus the table (else the original bytes are put back and exit 3);
  4. prints KEY=VALUE lines window.sh reads (CONVERTED, REASON, CONFIG, BACKUP, REMOVED_LINES, SHA_BEFORE, SHA_AFTER), or one JSON
     object with --json.
--restore BACKUP copies the backup's bytes over ROOT/config.toml atomically and verifies the sha256 (RESTORED=1), exit 3 on a
mismatch. --dry-run computes everything and writes nothing (no backup either).
Exit: 0 done (converted or nothing to convert); 2 usage / an unreadable file; 3 a refusal (backup exists, the converted text does
not parse to the old document minus the table, a restore mismatch)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

TABLE = "modal"
CONFIG_NAME = "config.toml"
HEADER_RE = re.compile(r"^\s*\[" + re.escape(TABLE) + r"\]\s*(#.*)?$")
ANY_HEADER_RE = re.compile(r"^\s*\[")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def strip_table(text: str) -> tuple[str, int]:
    """The text without the table's block; (new text, lines removed). The block runs from the header line to the line before
    the next header (any ``[`` at the start of a line) or the end; a header inside a multi-line string is not a table header
    in any configuration this registry writes (render_default), and the re-parse check below is what makes that assumption safe."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if HEADER_RE.match(l)), None)
    if start is None:
        return text, 0
    end = next((j for j in range(start + 1, len(lines)) if ANY_HEADER_RE.match(lines[j])), len(lines))
    return "".join(lines[:start] + lines[end:]), end - start


def atomic_write(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp-convert-{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def emit(rec: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(rec, indent=1, sort_keys=True))
    else:
        for k, v in rec.items():
            print(f"{k}={'' if v is None else v}")


def restore(cfg_path: str, backup: str, as_json: bool) -> int:
    try:
        with open(backup, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        print(f"convert_config: cannot read the backup {backup}: {exc}", file=sys.stderr)
        return 2
    atomic_write(cfg_path, data)
    with open(cfg_path, "rb") as fh:
        now = fh.read()
    ok = sha256_bytes(now) == sha256_bytes(data)
    emit({"RESTORED": 1 if ok else 0, "CONFIG": cfg_path, "BACKUP": backup, "SHA": sha256_bytes(now)}, as_json)
    if not ok:
        print("convert_config: the restored file does not hash as the backup", file=sys.stderr)
        return 3
    return 0


def main(argv=None) -> int:
    import tomllib
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the registry root (ROOT/config.toml; the backup goes under ROOT/state/)")
    ap.add_argument("--restore", metavar="BACKUP", help="put BACKUP's bytes back as ROOT/config.toml and verify")
    ap.add_argument("--dry-run", action="store_true", help="compute and print; write nothing")
    ap.add_argument("--json", action="store_true", help="one JSON object instead of KEY=VALUE lines")
    a = ap.parse_args(argv)
    root = os.path.abspath(os.path.expanduser(a.root))
    cfg_path = os.path.join(root, CONFIG_NAME)
    if a.restore:
        return restore(cfg_path, a.restore, a.json)
    rec: dict = {"CONVERTED": 0, "REASON": None, "CONFIG": cfg_path, "BACKUP": None, "REMOVED_LINES": 0, "SHA_BEFORE": None, "SHA_AFTER": None,
                 "DRY_RUN": 1 if a.dry_run else 0}
    if not os.path.exists(cfg_path):
        rec["REASON"] = "no config file (defaults apply; nothing to convert)"
        emit(rec, a.json)
        return 0
    with open(cfg_path, "rb") as fh:
        before = fh.read()
    rec["SHA_BEFORE"] = sha256_bytes(before)
    try:
        old = tomllib.loads(before.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        print(f"convert_config: {cfg_path} does not parse as TOML: {exc}", file=sys.stderr)
        return 2
    if TABLE not in old:
        rec["REASON"] = f"no {TABLE} table (nothing to convert)"
        rec["SHA_AFTER"] = rec["SHA_BEFORE"]
        emit(rec, a.json)
        return 0
    new_text, removed = strip_table(before.decode("utf-8"))
    try:
        new = tomllib.loads(new_text)
    except tomllib.TOMLDecodeError as exc:
        print(f"convert_config: the converted text does not parse: {exc}; nothing written", file=sys.stderr)
        return 3
    want = {k: v for k, v in old.items() if k != TABLE}
    if new != want:
        print(f"convert_config: the converted document is not the old one minus the {TABLE} table (the textual strip removed or kept "
              f"something else); nothing written", file=sys.stderr)
        return 3
    after = new_text.encode("utf-8")
    rec.update(REMOVED_LINES=removed, SHA_AFTER=sha256_bytes(after))
    if a.dry_run:
        rec["CONVERTED"] = 1
        rec["REASON"] = f"dry run: the {TABLE} table ({removed} line(s)) would be removed"
        emit(rec, a.json)
        return 0
    state_dir = os.path.join(root, "state")
    os.makedirs(state_dir, exist_ok=True)
    backup = os.path.join(state_dir, f"{CONFIG_NAME}.bak-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    if os.path.exists(backup):
        print(f"convert_config: a backup already exists at {backup}; nothing written", file=sys.stderr)
        return 3
    atomic_write(backup, before)
    with open(backup, "rb") as fh:
        if sha256_bytes(fh.read()) != rec["SHA_BEFORE"]:
            print(f"convert_config: the backup at {backup} does not hash as the original; nothing converted", file=sys.stderr)
            return 3
    try:
        os.chmod(backup, 0o444)
    except OSError:
        pass
    atomic_write(cfg_path, after)
    with open(cfg_path, "rb") as fh:
        written = fh.read()
    if sha256_bytes(written) != rec["SHA_AFTER"] or tomllib.loads(written.decode("utf-8")) != want:
        atomic_write(cfg_path, before)
        print(f"convert_config: the written file did not read back as the converted document; the original bytes were put back", file=sys.stderr)
        return 3
    rec.update(CONVERTED=1, REASON=f"the {TABLE} table removed ({removed} line(s)); backup written", BACKUP=backup)
    emit(rec, a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
