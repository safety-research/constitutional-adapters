"""The ``registry`` console script. Help strings are vocabulary-checked by the vocabulary
test (model / object / eval / run / judgment / judge / instrument — no campaign jargon).

Fully working here: ``init``, ``migrate``, ``ingest``, ``rebuild [--verify]``, ``status [--html|--json|--refresh-heavy [--force]]``,
``query``, ``frame``, ``history``, ``mirrors sync|verify`` (local and ssh destinations; the Modal destination was removed),
``snapshot [--scope full|runlevel] [--if-due] [--push]`` (push + commit on every destination), ``paper-head`` and ``config
[--check]`` (the rebuild trigger's inputs; ``--check`` loads ``config.toml`` and the legacy ``config.json`` through the loader and
exits 0 only on a configuration free of the removed Modal table and destination), ``pull [--clusters]``,
``docs``, ``identity resolve``, ``blocked list``, ``reconcile run-patch-extra [--dry-run|--apply]`` (the dropped top-level
``extra`` of legacy ``run_patch`` records re-filed as records, missing keys only, conflicts reported -- reconcile.py; ``--apply`` gated
in this release), ``refusals list|explain|summary`` (the refusal facts read back: every judgment ever made per refused record and judging
stamp, the refused bytes, the current validator's verdict and the record's resolution -- refusals.py; read-only),
``observations summary|list`` (the fold observations read back: a record that restated ``extra.serving_jobs`` as a strict subset of
the set the run's extra_json already held, detected at the fold and never refused -- observations.py; read-only),
``cap-check [--campaign C] [--all] [--json]`` (the token-cap ladder facts of the live runs: the record rule re-applied to
the live rows, the legacy shapes, the instrument cross-check -- capcheck.py; read-only), ``pending-judgment`` (list;
``--claim`` / ``--release`` through pending.py), ``worklist-parity``, ``adapters run`` (loads every concrete adapter —
``adapters.load_all``), ``objects scan``, ``decode-policy export|check``, ``publish-figures
[--publish-to DIR] [--verify-dir DIR]`` (refuses unregistered files), ``register-run``,
``roster load|show|diff|transcribe`` and ``ruling add|list|expire-check`` (adapters/roster.py, roster_transcribe.py),
``revision held|accept`` (the revisions the harvester holds; an operator's declaration bundle for one held on an undeclared
run drop -- a record under ``records/adapters/<adapter>/_activations/<rev>/<stamp>/``, never a database write),
``check-paper-parity`` (parity.py), ``paper-objects``, ``analysis file|list`` (a prefill-only or offline analysis filed
over ingested runs and objects as ledger-fact records, its report and pages registered as artifacts -- analysis.py; the database
is read for the input check and never written). ``export r2-compat`` and ``degeneracy-rebuild`` import
their modules lazily (``r2compat.py`` / ``degeneracy_artifact.py``) and exit 3 naming the branch when absent.
Stubs that print their contract and exit 3 (backend not yet implemented): ``backfill``.
Every run stamps ``state/last_run/<command>.json`` (the status page's timer-health section).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
from typing import List, Optional, Sequence, Tuple

from . import db as DB
from . import records as R
from .config import CONFIG_NAME, ConfigError

STUB_EXIT = 3
#: `adapters run` / `objects scan` on a host with no store access scope: refused by name, nothing read
STORE_READ_REFUSED_EXIT = 5


def _root_unstamped(args) -> str:
    """The root, resolved WITHOUT marking the command for ``stamp_last_run``: for a command whose contract is to write nothing of
    its own under the root (``reconcile run-patch-extra``'s dry run, and its gated ``--apply``, which exits 2 having written nothing). Every
    other command resolves through :func:`_root`, and ``main`` stamps ``state/last_run/<command>.json`` for it (the status page's
    timer-health record); a read-only command that left that file would break its own no-write promise."""
    return os.path.abspath(args.root or os.environ.get("REGISTRY_ROOT", DB.DEFAULT_ROOT))


def _root(args) -> str:
    args._root_resolved = True   # main() stamps state/last_run only for a command that resolved its root (stamp_last_run)
    return _root_unstamped(args)


def _stub(name: str, contract: str) -> int:
    print(f"registry {name}: not available in this tree. Contract: {contract}", file=sys.stderr)
    return STUB_EXIT


def _print_rows(rows, as_json: bool) -> None:
    if as_json:
        print(json.dumps([dict(r) for r in rows], indent=1, default=str))
        return
    rows = list(rows)
    if not rows:
        print("(no rows)")
        return
    keys = list(dict(rows[0]).keys())
    print("\t".join(keys))
    for r in rows:
        d = dict(r)
        print("\t".join("" if d[k] is None else str(d[k]) for k in keys))


# --------------------------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------------------------


def cmd_init(args) -> int:
    from . import harvest as H
    root = _root(args)
    H.ensure_layout(root)
    path = DB.db_path(root)
    if os.path.exists(path):
        print(f"{path} exists; use `registry migrate` to upgrade")
        return 0
    conn = DB.open_fresh(path)
    conn.close()
    print(f"created {path} under {root} (records/, inbox/, rulings/, state/, manifests/, snapshots/)")
    return 0


class SiteAliasesRefused(Exception):
    """The site-alias table a migration would read is refused: unreadable, malformed, a table inside ``config.toml``, or a
    digest other than the declared one."""


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def migrate_site_aliases(args, root: Optional[str]) -> dict:
    """The site-alias table ``registry migrate`` hands the migrations, and where it came from. With ``--site-aliases FILE``: that
    file's ``[site_aliases]`` table. Otherwise, in root mode, ``<root>/site_aliases.toml`` (absent: the empty table). In root mode a
    ``[site_aliases]`` table inside ``<root>/config.toml`` is refused by name whichever file is read, the named one included: every
    later fold over the root refuses that second source, so a migration must not advance the database past it. With ``--db`` and no
    ``--site-aliases`` the table is empty: a database outside a root never borrows a table from the directory it sits in or from
    ``$REGISTRY_ROOT``. The table is validated, and ``--expect-site-aliases-sha256`` (when given) must equal its digest. Returns
    ``{source, path, digest, pairs, table}``; raises :class:`SiteAliasesRefused` before anything is read from the database."""
    from . import site_aliases as SA
    expect = getattr(args, "expect_site_aliases_sha256", None)
    if expect is not None and not _HEX64.match(expect):
        raise SiteAliasesRefused(f"--expect-site-aliases-sha256 must be the full 64-hex digest, got {expect!r}")
    given = getattr(args, "site_aliases", None)
    try:
        if not args.db:
            SA.refuse_config_table(os.path.join(root, CONFIG_NAME))
        if given:
            path = os.path.abspath(given)
            if not os.path.isfile(path):
                raise SiteAliasesRefused(f"--site-aliases {given}: no such file")
            table, source = SA.load_table(path), "file"
        elif args.db:
            path, table, source = None, {}, "none (--db without --site-aliases: the empty table)"
        else:
            path = os.path.join(root, SA.FILE_NAME)
            table = SA.load_root(root)
            source = "root" if os.path.exists(path) else "none (no site_aliases.toml in the root: the empty table)"
        table = SA.validate(table)
    except SiteAliasesRefused:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SiteAliasesRefused(f"the site-alias table is refused: {exc}") from None
    digest = SA.digest(table)
    if expect is not None and digest != expect:
        raise SiteAliasesRefused(f"the site-alias table ({path or 'empty'}) has digest {digest}, not the expected {expect}")
    return {"source": source, "path": path, "digest": digest, "pairs": sorted(table.items()), "table": dict(table)}


def site_alias_plan_lines(info: dict, census: List[dict]) -> Tuple[List[str], int]:
    """The ``--plan`` lines for the site-alias table: its source and digest, every pair with the rows it maps per column
    (``site_aliases.census``), and every stored value that is neither a label of its column nor mapped to one. Returns ``(lines,
    number of values the label migration would refuse)``."""
    lines = [f"site aliases: {info['path'] or '-'} ({info['source']}); digest {info['digest']}; {len(info['pairs'])} pair(s)"]
    for old, new in info["pairs"]:
        hits = [c for c in census if c["value"] == old]
        counts = ", ".join(f"{c['table']}.{c['column']} {c['rows']}" for c in hits) or "no rows"
        bad = [f"{c['table']}.{c['column']}" for c in hits if c["status"] == "alias_not_admitted"]
        lines.append(f"  alias {old} -> {new}: {counts}" + (f"; {new!r} is not admitted by {', '.join(bad)}" if bad else ""))
    for c in census:
        if c["status"] == "unaliased":
            lines.append(f"  legacy label {c['value']!r} in {c['table']}.{c['column']}: {c['rows']} row(s), NO alias -- the label migration "
                         "refuses it; name it in the root's site_aliases.toml (or the file given with --site-aliases)")
    return lines, sum(1 for c in census if c["status"] in ("unaliased", "alias_not_admitted"))


def cmd_migrate(args) -> int:
    """Apply the pending migrations, then verify the shape against the code's DDL (exit 1, naming what still differs,
    when a migration is missing). ``--plan`` reads only: the version state, the pending files and what a new migration
    would have to rebuild; ``--plan --sql`` also renders that migration's body for the next ``NNNN_<slug>.sql``, and with
    ``--map-site-labels`` every site-label column of a rebuilt table is copied through the runner's alias table (the site-label
    rename's form).

    The site-alias table (:func:`migrate_site_aliases`) is handed to the runner, which maps legacy site labels through it in the
    label migration; ``--plan`` prints, after the shape lines, its digest, every pair with the rows it maps per column, and every
    legacy value left without a pair (exit 1 then). ``--plan`` resolves the root without stamping ``state/last_run``: a read-only
    command writes nothing under the root."""
    from . import site_aliases as SA
    from .migrations import (MigrationFailed, apply_migrations, code_version, db_version, pending_migrations, render_shape_migration,
                             shape_plan)
    reading_only = bool(args.plan or args.sql)
    root = None if args.db else (_root_unstamped(args) if reading_only else _root(args))
    path = args.db or DB.db_path(root)
    try:
        aliases = migrate_site_aliases(args, root)
    except SiteAliasesRefused as exc:
        print(f"registry migrate REFUSED: {exc}; nothing was read or applied", file=sys.stderr)
        return 1
    if reading_only:
        if not os.path.exists(path):
            print(f"{path} does not exist; `registry init` creates it", file=sys.stderr)
            return 1
        with DB.reading(path) as conn:
            have = db_version(conn)
            pend = pending_migrations(conn)
            plan = shape_plan(conn)
            body = render_shape_migration(conn, plan, map_site_labels=bool(getattr(args, "map_site_labels", False))) if args.sql else ""
            census = SA.census(conn, aliases["table"])
        print(f"schema version {have}; code {code_version()}; pending: " + (", ".join(f"{v:04d}_{slug}" for v, slug, _ in pend) or "none"))
        if pend:
            print("apply the pending migrations first (`registry migrate`); the plan below is relative to the CURRENT shape")
        drift = plan["drift"]
        print("shape: matches the code's schema" if not drift else "shape differs from the code's schema:")
        for line in drift:
            print(f"  {line}")
        for r in plan["rebuild"]:
            dropped = f"; column(s) not declared by the code, NOT copied: {', '.join(r['dropped_columns'])}" if r["dropped_columns"] else ""
            print(f"rebuild table {r['table']}: {r['rows']} rows to copy ({r['scale']}-scale){dropped}")
        for a in plan.get("add_columns", ()):
            print(f"add column(s) to table {a['table']}: {', '.join(a['columns'])} ({a['rows']} rows, none copied; {a['scale']}-scale)")
        for name in plan["create"]:
            print(f"create table {name}")
        for name in plan["undeclared_tables"]:
            print(f"table {name} is not declared by the code: dropping it is a human decision (not generated)")
        for typ in ("index", "trigger", "view"):
            o = plan["objects"][typ]
            if o["recreate"]:
                print(f"recreate {typ}(s): {', '.join(o['recreate'])}")
            if o["drop"]:
                print(f"drop undeclared {typ}(s): {', '.join(o['drop'])}")
        # the site-alias block follows the shape lines, so the version line and the shape lines keep their places for every reader
        alias_lines, n_unmapped = site_alias_plan_lines(aliases, census)
        for line in alias_lines:
            print(line)
        if args.sql:
            print("-- ---- migration body: save as src/sisteer/registry/migrations/NNNN_<slug>.sql, review, commit; `registry migrate` applies it")
            print(body if body else "-- (nothing: the shape matches the code's schema)")
        return 0 if not pend and not drift and not n_unmapped else 1
    # THE APPLY RUNS UNDER THE REGISTRY'S SINGLE-WRITER LOCK: `registry migrate`
    # rebuilds tables in place, and a harvester cycle, a claim or a landing window's own ingest writing meanwhile would race it. The lock
    # is the harvester's (`<root>/harvester.lock`, the one every writer takes); held by another process it REFUSES by name (exit 4)
    # after `--lock-timeout` seconds (default 0: refuse at once) -- never applies concurrently, never waits silently. `--db` names a file
    # outside a root: the lock then sits beside that file (`<dir>/harvester.lock`), so two migrations of one file still serialise.
    lock_root = os.path.dirname(os.path.abspath(path)) if args.db else root
    print(f"site aliases: {aliases['path'] or '-'} ({aliases['source']}); digest {aliases['digest']}; {len(aliases['pairs'])} pair(s)")
    try:
        with DB.write_lock(lock_root, timeout_s=float(getattr(args, "lock_timeout", 0.0) or 0.0)):
            conn = DB.connect(path)
            notes: List[str] = []
            try:
                try:
                    applied = apply_migrations(conn, notes, site_aliases=aliases["table"])
                except MigrationFailed as exc:
                    print(f"migration failed (rolled back): {exc}", file=sys.stderr)
                    return 1
                drift = DB.schema_drift(conn)
            finally:
                conn.close()
    except DB.LockHeld as exc:
        print(f"registry migrate REFUSED: the registry's write lock is held ({exc}) -- a harvester cycle, a claim or another migration is "
              "writing; stop the writer timers (the landing window) or pass --lock-timeout SECONDS to wait; nothing was applied", file=sys.stderr)
        return 4
    print(f"applied {applied or 'nothing'}; schema version {code_version()}")
    for note in notes:
        print(note)                      # a migration's Python post-step (a data lift) reports what it moved
    if drift:
        print("the shape still differs from the code's schema — a migration is missing (`registry migrate --plan --sql` renders it):",
              file=sys.stderr)
        for line in drift:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("shape matches the code's schema")
    return 0


def _status_page_for_refusal(root: str, exc: "DB.SchemaDrift") -> Optional[str]:
    """Render the status page (its banner names the drift) ONCE per distinct refusal: ``state/schema_refusal.json``
    holds the sha256 of the last message rendered; the same refusal every 20 s must not re-render (and re-register the
    page as an artifact) every cycle. Returns the page path when rendered."""
    import hashlib
    from . import status
    sha = hashlib.sha256(str(exc).encode("utf-8")).hexdigest()
    state = os.path.join(root, "state", "schema_refusal.json")
    try:
        with open(state, "r", encoding="utf-8") as fh:
            if json.load(fh).get("sha256") == sha:
                return None
    except (OSError, ValueError):
        pass
    path = status.render(root)[0]
    os.makedirs(os.path.dirname(state), exist_ok=True)
    tmp = state + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"utc": R.utc_now(), "sha256": sha, "have": exc.have, "want": exc.want, "lines": exc.lines, "page": path}, fh)
    os.replace(tmp, state)
    return path


def cmd_ingest(args) -> int:
    from . import harvest as H
    root = _root(args)
    if args.records:
        # ingest a foreign records directory: copy-free — point the harvester at a root whose records/ is that dir
        print("--records: pass the registry ROOT with --root; the record log lives at <root>/records", file=sys.stderr)
        return 2
    if getattr(args, "honor_maintenance", False):
        # the honest drain: a landing window's state/maintenance.json holds this cycle at the door -- before the
        # cluster pull, before the write lock -- so the drain the window waits for is one cycle at most, and a cycle the timer fires
        # mid-window neither pulls nor folds. Exit 0: the unit is healthy, the window is the reason. The claim mechanism never reads it.
        flag = H.maintenance(root)
        if flag is not None:
            print(json.dumps({"ingest": "skipped: maintenance", "maintenance": flag, "flag": H.maintenance_path(root)}, indent=1, default=str))
            return 0
    try:
        rep = H.ingest_under_lock(root, workers=args.workers, pull=args.pull, budget_s=getattr(args, "budget_s", None),
                                  lock_timeout_s=getattr(args, "lock_timeout", 30.0), retry_refused=bool(getattr(args, "retry_refused", False)),
                                  retry_held=bool(getattr(args, "retry_held", False)))
    except DB.LockHeld as exc:
        # another writer (a claim, a landing window's ingest, an earlier cycle still running) holds harvester.lock past the wait:
        # nothing was folded; refused by name with exit 4 (the `migrate` convention) instead of a traceback in the unit's journal
        # (the 2026-09-30 journal: two cycles in a row ended `status=1/FAILURE` on an uncaught LockHeld). The timer re-fires.
        print(f"refused: the registry's write lock is held ({exc}); nothing was folded this cycle -- the next cycle retries", file=sys.stderr)
        return 4
    except DB.SchemaDrift as exc:
        # the database's version or shape is not the code's: nothing was written. Loud, by name, with the remedy; the
        # status page carries the same lines in its banner (rendered once per distinct refusal, not every cycle)
        print(f"refused: {exc}", file=sys.stderr)
        if args.html:
            page = _status_page_for_refusal(root, exc)
            if page:
                print(page)
        return 3
    d = rep.as_dict()
    d.pop("refusals")
    # the cycle's completion stamp beside its counts: what the status page's harvester row ages (harvest.write_cycle_state);
    # a cycle its unit killed never reaches this line and never moves the stamp
    d["last_completed_cycle_utc"] = rep.completed_utc
    print(json.dumps(d, indent=1))
    if rep.stopped:
        print(f"stopped: {rep.stopped} -- {rep.revisions_deferred_budget} revision director(ies) / bundle(s) and "
              f"{rep.plain_files_deferred_budget} plain file(s) left for the next cycle after {rep.elapsed_s:.1f} s "
              f"(budget {rep.budget_s} s); everything folded before the stop is committed", file=sys.stderr)
    for r in rep.refusals[:50]:
        print("refused:", r, file=sys.stderr)
    if rep.lifecycle_regressions_held:
        print(f"held: {rep.lifecycle_regressions_held} run_state event(s) that would have moved a run's lifecycle backwards "
              "(folded for provenance; lifecycle_state unchanged)", file=sys.stderr)
    if args.html and (rep.refused or rep.blocked or rep.paused or (rep.changed and not rep.self_change_only)):
        # rendered AFTER the write lock is released; a rendering error is this command's to report, never swallowed.
        # A cycle whose only change is the previous render's own page artifact does NOT re-render: that would
        # register another page for the next cycle to ingest, and loop every 20 s (harvest.is_status_page_artifact).
        from . import status
        print(status.render(root)[0])
    return 0


def cmd_rebuild(args) -> int:
    from . import harvest as H
    rep = H.rebuild(args.from_dir, args.into, workers=args.workers)
    d = rep.as_dict()
    d.pop("refusals")
    print(json.dumps(d, indent=1))
    if args.verify:
        live = args.verify_against or DB.db_path(_root(args))
        with DB.reading(live) as a, DB.reading(args.into) as b:
            diff = H.verify_equal(a, b)
        bad = {k: v for k, v in diff.items() if v}
        print("verify:", "0 differing rows on every compared table" if not bad else json.dumps(bad))
        return 0 if not bad else 1
    return 0


def cmd_status(args) -> int:
    from . import status
    root = _root(args)
    if args.refresh_heavy:
        # the cached-with-refresh set recomputed exactly (the 6-hourly registry-status-heavy unit; the window's cache
        # initializer). Event-gated on the ledger head; --force recomputes regardless. Prints the summary JSON.
        out = status.refresh_heavy(root, force=args.force)
        print(json.dumps(out, indent=1, default=str))
        return 0
    if args.html:
        path, timings = status.render(root, out_path=args.out)
        print(path)
        return 0
    ctx = status.status_json(root)
    if args.json:
        print(json.dumps(ctx, indent=1, default=str))
    else:
        behind = (f" (code expects v{ctx['schema_version_code']}: run `registry migrate`)"
                  if ctx["schema_version"] != ctx["schema_version_code"] else "")
        print(f"head ingest_seq: {ctx['head_seq']}  schema v{ctx['schema_version']}{behind}  built {ctx['built_utc']}")
        if ctx["schema_drift"]:
            print("SCHEMA DRIFT — writers refuse this database until `registry migrate` brings its shape to the code's "
                  "(`registry migrate --plan --sql` renders the missing migration):")
            for line in ctx["schema_drift"]:
                print(f"  {line}")
        print(f"in flight: {len(ctx['in_flight'])}  roster gaps: {len(ctx['roster_gaps'])}  refused: {len(ctx['refused'])}  "
              f"blocked: {len(ctx['blocked'])}  pending judgments: {ctx['pending_count']}")
        ms = ctx["multi_source_counts"]
        print(f"multi-source readings: refused {ms['refused']}  stopgap {ms['stopgap']}  folded {ms['folded']}"
              + "".join(f"\n  REFUSED {r['declaration_id']} {r['eval_id']} {r['metric_id']} @{r['threshold']}: {r['reason']}" for r in ctx["multi_source_refused"]))
        cov = ctx["source_coverage"]
        if cov.get("error"):
            print(f"sources without an adapter: not computed ({cov['error']})")
        else:
            print(f"sources without an adapter: {len(cov['without_adapter'])} of {cov['n_declared']} declared"
                  + ("".join(f"\n  {r['source']}  <- {', '.join(r['declared_in'])}" for r in cov["without_adapter"])))
        for t, n in ctx["table_counts"].items():
            print(f"  {t}: {n}")
        if ctx["paused"]:
            print("INGEST PAUSED:", ctx["paused"])
        # the state-file section's headlines (store authority, the replica check, the reads reservation and
        # receipts, the maintenance flag, the removed configuration table) -- the same facts the page's section shows
        so = ctx.get("slurm_only") or {}
        au, rp, rs, rc = (so.get(k) or {} for k in ("authority", "replica", "reservation", "receipts"))
        print(f"cluster side: {so.get('level')} ({so.get('n_problems')} problem(s))")
        print("  store authority: " + (f"checked {au.get('checked_utc')}, {au.get('n_components')} component(s), {au.get('n_problems')} with a problem"
                                       if au.get("present") else f"check-authority has not run ({au.get('n_components') or 0} component(s) known from the mirror state)"))
        print("  replica check: " + (f"{'equal modulo the C1 same-stamp class (' + str(rp.get('n_movers')) + ' movers, declared)' if rp.get('accepted_as') == 'c1_class' else ('equal' if rp.get('equal') else 'UNEQUAL')}"
                                     f"{'' if rp.get('ok') else ' REFUSED (' + str(rp.get('headline')) + ')'} at {rp.get('utc')}, "
                                     f"generation {rp.get('generation')}" if rp.get("present") else "no Slurm live-DB step recorded"))
        print("  reads reservation: " + (f"{rs.get('owner')} {rs.get('op_id')} on {rs.get('host')} pid {rs.get('pid')} since {rs.get('started_utc')} ({rs.get('state')})"
                                         if rs.get("held") else "none held"))
        print(f"  read receipts: {rc.get('n_receipts') or 0}" + (f", latest {rc['rows'][0].get('op_id')} at {rc['rows'][0].get('utc')} "
                                                                  f"(read whole {rc['rows'][0].get('n_read_whole')}, refused {rc['rows'][0].get('n_refused')})" if rc.get("rows") else ""))
        for pr in so.get("problems") or []:
            print(f"  PROBLEM: {pr}")
    return 0


def cmd_query(args) -> int:
    root = _root(args)
    with DB.reading(DB.db_path(root)) as conn:
        if args.parquet:
            import pandas as pd
            df = pd.read_sql_query(args.sql, conn)
            df.to_parquet(args.parquet)
            print(f"{len(df)} rows -> {args.parquet}")
        else:
            _print_rows(conn.execute(args.sql).fetchall(), args.json)
    return 0


def cmd_frame(args) -> int:
    root = _root(args)
    with DB.reading(DB.db_path(root)) as conn:
        import pandas as pd
        sql = f"SELECT * FROM {args.name}"
        if args.where:
            sql += " WHERE " + args.where
        df = pd.read_sql_query(sql, conn)
        if args.parquet:
            df.to_parquet(args.parquet)
            print(f"{len(df)} rows -> {args.parquet}")
        else:
            print(df.to_string(max_rows=args.limit))
    return 0


def cmd_history(args) -> int:
    from . import harvest as H
    root = _root(args)
    with DB.reading(DB.db_path(root)) as conn:
        key = json.loads(args.key_json)
        states = H.history(root, conn, args.table, key)
    if args.json:
        print(json.dumps(states, indent=1, default=str))
    else:
        for s in states:
            if args.table in R.RUN_SCALE_TABLES:
                print(f"{s['written_utc']} {s['record_sha256'][:12]} {s['record_type']:<14} {s['record_uri']}")
                print("   ", json.dumps(s["state"], sort_keys=True))
            else:
                print(f"{s['_written_utc']} {s['_record_sha256'][:12]} {s['_record_uri']}")
                print("   ", json.dumps({k: v for k, v in s.items() if not k.startswith('_')}, sort_keys=True))
    return 0


def cmd_mirrors(args) -> int:
    import time
    from . import mirrors as M
    t0 = time.monotonic()
    prune = args.action == "prune-manifests"
    if not prune and (getattr(args, "dry_run", False) or getattr(args, "keep", None) is not None):
        print("--dry-run and --keep belong to `mirrors prune-manifests`", file=sys.stderr)
        return 2
    # a dry run writes nothing under the root, its state/last_run stamp included (as `reconcile run-patch-extra`'s dry run)
    root = _root_unstamped(args) if (prune and args.dry_run) else _root(args)
    configured = M.destinations(root)
    dests = list(configured)
    if args.dest:
        dests = [d for d in dests if d.name in set(args.dest)]
    if getattr(args, "kind", None):
        dests = [d for d in dests if d.kind == args.kind]
    if getattr(args, "sample", None) is not None:
        if int(args.sample) < 0:
            print(f"--sample must be a non-negative number of files (got {args.sample})", file=sys.stderr)
            return 2
        for d in dests:
            if d.kind == "ssh":
                d.sample = int(args.sample)
    if not dests:
        print("no mirror destinations configured (<root>/config.toml [[clusters]], or config.json mirrors)", file=sys.stderr)
        return 2
    if prune:
        keep = M.MANIFEST_KEEP if args.keep is None else int(args.keep)
        if keep < 1:
            print(f"--keep must be at least 1 (got {args.keep}): the newest manifest is never deleted", file=sys.stderr)
            return 2
        # every configured destination is accounted for, the ones --dest omits included: an unknown pointer holds the grading host
        summary = M.prune_manifests(root, dests, keep=keep, dry_run=bool(args.dry_run), configured=[d.name for d in configured])
        print(M.format_retention(summary))
        if args.json:
            print(json.dumps(summary, default=str))
        return 0 if (args.dry_run or not any(side.get("error") for side in summary["sides"])) else 1
    if args.action == "verify":
        rows = [M.verify_destination(root, d) for d in dests]
        M.write_state(root, rows)
        for r in rows:
            print(json.dumps(r, indent=1, default=str))
        return 0 if all(r["lag_files"] == 0 and not r["error"] for r in rows) else 1
    with DB.reading(DB.db_path(root)) as conn:
        seq = conn.execute("SELECT COALESCE(MAX(ingest_seq), 0) FROM record_ingest").fetchone()[0]
    # the budget counts from process start: the manifest write runs before the destinations' copies
    budget_s = float(args.budget_s) if args.budget_s else M.DEFAULT_SYNC_BUDGET_S
    rows = M.sync_all(root, int(seq), dests, deadline=t0 + budget_s, full=bool(getattr(args, "full", False)))
    for r in rows:
        print(json.dumps(r, default=str))
    # a run the budget stopped made progress (its copied chunks are at their final names) and resumes on the next fire: a success
    return 0 if all(r["switched"] or (r.get("stopped") == "budget" and not r.get("error")) for r in rows) else 1


def cmd_snapshot(args) -> int:
    from . import config as CFG
    from . import harvest as H
    from . import snapshot as SN
    root = _root(args)
    scope = args.scope
    dest_root = getattr(args, "dest_root", None)
    source_root = getattr(args, "source_root", None)
    if scope == "paper_frame" or dest_root:
        # the committed export for the paper's companion repository: read the source registry, write EVERYTHING under
        # --dest-root; no harvester `snapshot` record and no footprint file land under the operational root, and its
        # snapshots/ + CURRENT are never switched or reclaimed. The manifest is printed instead.
        if scope != "paper_frame":
            print(f"snapshot: --dest-root is only for --scope paper_frame (scope {scope} builds under the registry root)", file=sys.stderr)
            return 2
        if not dest_root:
            print("snapshot: --scope paper_frame requires --dest-root DIR: the committed export is written under a directory that is "
                  "never the registry root, so it can never switch or reclaim the registry's own snapshots/ + CURRENT", file=sys.stderr)
            return 2
        if args.push or args.if_due:
            print("snapshot: --push / --if-due are for the operational scopes (full, runlevel); a paper frame is a command run on "
                  "demand and committed, never pushed to a destination", file=sys.stderr)
            return 2
        source = os.path.abspath(source_root) if source_root else root
        try:
            info = SN.build(source, DB.db_path(source), check_free_space=not args.no_space_check, scope="paper_frame",
                            dest_root=os.path.abspath(dest_root), model_ids=list(getattr(args, "model_id", None) or []) or None,
                            precondition_override=getattr(args, "precondition_override", None))
        except SN.SnapshotError as exc:
            print(f"snapshot: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(info.manifest, indent=1, sort_keys=True))
        return 0
    if source_root:
        print("snapshot: --source-root is only for --scope paper_frame (the operational scopes read and write one root, --root)", file=sys.stderr)
        return 2
    cfg = CFG.load(root)
    if args.if_due:
        with DB.reading(DB.db_path(root)) as conn:
            due, reason = SN.should_build(conn, root, cfg["snapshot"])
        if due is None:
            print(json.dumps({"built": False, "reason": reason}))
            return 0
        scope = due
    pushed: List[dict] = []
    with SN.footprint_watcher(root) as w:
        try:
            info = SN.build(root, DB.db_path(root), check_free_space=not args.no_space_check, scope=scope)
        except SN.SnapshotError as exc:
            SN.write_footprint(root, w, None, DB.db_path(root), extra={"error": str(exc), "scope": scope})
            print(f"snapshot: {exc}", file=sys.stderr)
            return 1
        if args.push:
            dests = H.load_config(root).get("snapshot_destinations") or []
            if not dests:
                print("no snapshot destinations configured (config.toml [[clusters]]); built locally only", file=sys.stderr)
            pushed = SN.push(root, info.snapshot_id, dests)
    H.write_harvester_record(root, "snapshot", SN.snapshot_record_payload(info, pushed))
    SN.write_footprint(root, w, info, DB.db_path(root), extra={"pushed_to": pushed})
    print(json.dumps({"built": True, "snapshot_id": info.snapshot_id, "scope": info.scope, "ingest_seq": info.ingest_seq, "dir": info.dir,
                      "runlevel_bytes": info.runlevel_bytes, "parquet_bytes": info.parquet_bytes, "reclaimed": info.reclaimed,
                      "leases_blocking": info.leases_blocking, "peak_bytes": w.peak,
                      "pushed_to": [{k: p[k] for k in ("destination", "ok", "error")} for p in pushed]}, indent=1))
    return 0 if all(p["ok"] for p in pushed) else 1


def cmd_paper_head(args) -> int:
    """The rebuild trigger's inputs: the paper-input head, the open round, and the roster entries with
    neither a judged run nor a disposition (a non-empty list refuses the rebuild)."""
    root = _root(args)
    from . import paper
    from .migrations import db_version
    with DB.reading(DB.db_path(root)) as conn:
        head = conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0] or 0
        # the schema version rides beside the head (migration 0021): a migration that recreates a v_* object changes what the frame
        # reads without moving a rank, so paper_rebuild.sh keys its watermark on both and rebuilds once after `registry migrate`
        schema_version = db_version(conn)
        gaps = [dict(r) for r in conn.execute("SELECT roster_entry_id, comparison_id, model_id, eval_id FROM v_roster_gaps "
                                              "ORDER BY comparison_id, model_id, eval_id").fetchall()]
        indeterminate = [dict(r) for r in conn.execute("SELECT run_id, ruling_ids_json FROM v_paper_current "
                                                       "WHERE selection_state = 'indeterminate'").fetchall()]
        round_id = paper.open_round(conn)
    if args.json:
        print(json.dumps({"head": int(head), "schema_version": int(schema_version), "open_round": round_id, "roster_gaps": len(gaps),
                          "gaps": gaps[:100], "indeterminate": len(indeterminate), "indeterminate_runs": indeterminate[:20]}, indent=1))
    else:
        print(int(head))
    return 0


def cmd_config_check(root: str) -> int:
    """``registry config --check``: ``config.toml`` AND the legacy ``config.json``
    through the loader (``config.load`` + ``harvest.load_config``). Exit 0 with one ``config --check: OK`` line naming what was read
    and how many destinations of which kinds derive from it; exit 1 with ``config --check: REFUSED: <reason>`` on stderr when the
    TOML still carries the removed Modal table, when any destination list of ``config.json`` names a ``kind: modal`` entry, or when
    either file does not parse. The cutover window runs this under the worktree's code right after ``convert_config.py`` and aborts
    on anything but 0; every later window runs it too (a no-op conversion still validates)."""
    import tomllib
    from . import config as CFG
    from . import harvest as H
    path = CFG.config_path(root)
    json_path = os.path.join(root, "config.json")
    try:
        cfg = CFG.load(root)
        merged = H.load_config(root)
    except CFG.ConfigError as exc:
        print(f"config --check: REFUSED: {exc}", file=sys.stderr)
        return 1
    except tomllib.TOMLDecodeError as exc:
        print(f"config --check: REFUSED: {path} does not parse as TOML: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:     # json.JSONDecodeError is a ValueError: a config.json that does not parse
        print(f"config --check: REFUSED: {json_path if os.path.exists(json_path) else path}: {exc}", file=sys.stderr)
        return 1
    lists = {k: list(merged.get(k) or []) for k in CFG.DESTINATION_LISTS}
    kinds = sorted({str(d.get("kind", "local") or "local") for k in ("mirrors", "snapshot_destinations") for d in lists[k] if isinstance(d, dict)})
    print(f"config --check: OK {path} ({'present' if cfg['_present'] else 'absent; defaults apply'}); config.json "
          f"{'present' if os.path.exists(json_path) else 'absent'}; mirrors {len(lists['mirrors'])}, pulls {len(lists['pulls'])}, "
          f"snapshot destinations {len(lists['snapshot_destinations'])}; destination kinds {' '.join(kinds) or 'none'}; "
          f"no `{CFG.REMOVED_TABLE}` table, no `{CFG.REMOVED_KIND}` destination")
    return 0


def cmd_config(args) -> int:
    """Print the merged configuration (or one dotted key); ``--write-default`` lays the file down when absent; ``--check`` validates
    both configuration files through the loader (:func:`cmd_config_check`). A file the loader refuses (the removed Modal table) is
    reported by name on stderr with exit 1 — never a traceback, never a silent default."""
    from . import config as CFG
    root = _root(args)
    path = CFG.config_path(root)
    if args.write_default:
        if os.path.exists(path) and not args.force:
            print(f"{path} exists (use --force to overwrite)", file=sys.stderr)
        else:
            os.makedirs(root, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(CFG.render_default(root, paper_checkout=args.paper_checkout))
            print(path)
        return 0
    if args.check:
        return cmd_config_check(root)
    try:
        cfg = CFG.load(root)
    except CFG.ConfigError as exc:
        print(f"config: REFUSED: {exc}", file=sys.stderr)
        return 1
    if args.key:
        try:
            v = CFG.get(cfg, args.key)
        except KeyError:
            print(f"no such key: {args.key}", file=sys.stderr)
            return 1
        print(json.dumps(v) if not isinstance(v, str) else v)
        return 0
    if args.json:
        print(json.dumps({k: v for k, v in cfg.items()}, indent=1, default=str))
    else:
        print(f"config: {path}{'' if cfg['_present'] else '  (absent; defaults shown)'}")
        for section in ("registry", "paper", "status", "snapshot", "harvest", "repo"):
            for k, v in cfg.get(section, {}).items():
                print(f"  {section}.{k} = {v}")
        for c in cfg.get("clusters", []):
            print(f"  cluster {c['name']}: {c['host']} {c['shared']}")
    return 0


def cmd_docs(args) -> int:
    from . import docsgen
    if args.which in ("schema", "all"):
        # schema.sql only: the migration files are frozen history — a DDL change lands as a NEW migration
        # (`registry migrate --plan --sql` renders it), never as a rewrite of 0001_initial.sql
        print(docsgen.write_schema_sql())
    for name in (("schema", "vocabulary", "records") if args.which == "all" else (args.which,)):
        print(docsgen.write_doc(name))
    return 0


def cmd_identity(args) -> int:
    root = _root(args)
    if args.action != "resolve":
        return 2
    with DB.reading(DB.db_path(root)) as conn:
        sql = "SELECT * FROM v_producer_key WHERE producer_run_key_legacy = ?"
        params: List[object] = [args.producer_key]
        if args.campaign:
            sql += " AND campaign = ?"
            params.append(args.campaign)
        _print_rows(conn.execute(sql, params).fetchall(), args.json)
    return 0


def cmd_blocked(args) -> int:
    root = _root(args)
    with DB.reading(DB.db_path(root)) as conn:
        sql = "SELECT * FROM blocked_records" + (" WHERE stale = 1" if args.stale else "") + " ORDER BY first_seen_utc"
        _print_rows(conn.execute(sql).fetchall(), args.json)
    return 0


def cmd_reconcile(args) -> int:
    """``reconcile run-patch-extra [--dry-run | --apply]`` (:mod:`sisteer.registry.reconcile`).
    The dry run (the default) walks the log's ``run_patch`` files, merges each run's dropped top-level ``extra`` in
    the fold's order and prints the three buckets against the live ``run.extra_json`` -- missing / equal / conflict -- plus the
    loud ones (pending, unaccepted, no_live_row, malformed) and the totals. Only a lost record the fold ACCEPTED (a ``row_history``
    event of the run) is evidence; a refused, blocked or not-yet-ingested one is reported by status and never re-filed. ``--apply``
    is GATED in this release (``reconcile.APPLY_ENABLED`` is False): it prints the same report, lists the conflicts for adjudication
    and exits 2 having written nothing; the landing window lifts the gate. Reads the database read-only; never
    writes it -- and writes nothing of its own under the root either: the root is resolved by :func:`_root_unstamped`, so no
    ``state/last_run`` stamp is left by a command that promises no write. What a read-only open of the WAL-mode database leaves is
    SQLite's, not the command's: the ``-wal`` / ``-shm`` side files, which any ``mode=ro`` connection creates when no other connection
    holds them open (``sqlite3 file:...?mode=ro`` included) and cannot remove on close, the WAL empty, reclaimed by the next writer's
    close. The pass does not open ``immutable=1`` to avoid them: that disables locking and WAL reads, a silently stale snapshot
    against a live writer."""
    import sqlite3
    from . import reconcile as RC
    root = _root_unstamped(args)
    if args.source and args.action not in (RC.ENV_COMMENTS_ACTION, "run-patch-extra"):
        print(f"refused: --source belongs to {RC.ENV_COMMENTS_ACTION} and run-patch-extra (the overrides files a stamp is restated from, and "
              "compared as a reading of)", file=sys.stderr)
        return 1
    if (args.check_sources or args.leave_unsourced) and args.action != RC.ENV_COMMENTS_ACTION:
        print(f"refused: --check-sources and --leave-unsourced belong to {RC.ENV_COMMENTS_ACTION} (the coverage of its --source files over the "
              "stamped run rows, and the operator's word to leave the stamps no file covers as stored)", file=sys.stderr)
        return 1
    if args.action == RC.SCOPE_ACTION:
        return _cmd_reconcile_multi_turn_scope(args, root)
    if args.action == RC.ENV_COMMENTS_ACTION:
        return _cmd_reconcile_correction(args, root, RC.verify_env_comments, RC.reconcile_env_comments, RC.format_overrides_report,
                                         "a record stating each one's run_overrides (this pass's own correction, a writer's clean stamp, or a "
                                         "full run record carrying the stamp) is already in the log and not folded yet, the stamp is no "
                                         "reading of its overrides file (unclassified, refused by name), or no file among the sources has the "
                                         "stamp's sha256 while its values may carry a comment (unsourced: name the file with --source, or "
                                         "leave the stamp as stored by name with --leave-unsourced)",
                                         check_fn=RC.check_env_comments_sources, sources=list(args.source), leave_unsourced=bool(args.leave_unsourced))
    if args.action == RC.BATCH_ACTION:
        return _cmd_reconcile_correction(args, root, RC.verify_fit_effective_batch, RC.reconcile_fit_effective_batch, RC.format_batch_report,
                                         "an object record of each one (this pass's own correction, a native re-emission, an adapter revision) is "
                                         "already in the log and not folded yet, or the row is an unclassified disagreement the pass refuses by name")
    if args.verify:
        print(f"refused: --verify belongs to the correction actions ({RC.SCOPE_ACTION}, {RC.ENV_COMMENTS_ACTION}, {RC.BATCH_ACTION}: the predicate over "
              "the folded database); run-patch-extra has the dry run", file=sys.stderr)
        return 1
    try:
        rep = RC.reconcile_run_patch_extra(root, apply=False, sources=list(args.source))
    except (OSError, ValueError, DB.SchemaDrift, sqlite3.Error) as exc:
        # no record log / no database at the root, a directory that cannot be listed, a record line that is not JSON, a named source
        # file that is not there: by name
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rep.as_dict(), indent=1, default=str))
    else:
        print(RC.format_report(rep, verbose_equal=args.all))
    if args.apply:
        if not RC.APPLY_ENABLED:
            conflicts = [r for r in rep.runs if r.conflict]
            print(f"refused: --apply is gated in this release (reconcile.APPLY_ENABLED is False): the dry run above is the whole output "
                  f"and nothing was written. {len(conflicts)} run(s) carry a conflicting key for the owner's adjudication"
                  + (":" if conflicts else "."), file=sys.stderr)
            for r in conflicts:
                for k, v in r.conflict.items():
                    print(f"  {r.run_id} {k}: lost={R.canonical_json(v['lost'])} live={R.canonical_json(v['live'])}", file=sys.stderr)
            unaccepted = [r for r in rep.runs if r.unaccepted]
            if unaccepted:
                a = rep.records_by_status()
                print(f"{len(unaccepted)} run(s) carry a lost record the fold never accepted (refused {a['refused']} / blocked "
                      f"{a['blocked']} / not_ingested {a['not_ingested']}): excluded from the evidence, never re-filed; a blocked or "
                      f"not-ingested one holds its run until an ingest.", file=sys.stderr)
            return 2
        rep = RC.reconcile_run_patch_extra(root, apply=True, sources=list(args.source))      # pragma: no cover -- the gate is False in this release
        print(f"wrote {len(rep.written)} reconciliation record(s)")
        for p in rep.written:
            print(p)
    return 0


def _cmd_reconcile_multi_turn_scope(args, root: str) -> int:
    """``reconcile multi-turn-scope [--dry-run | --apply | --verify] [--json]`` (:func:`reconcile.reconcile_multi_turn_scope`,
    :func:`reconcile.verify_multi_turn_scope`): the runs the live database shows on a multi-turn, Petri or agentic eval with a steering
    object whose ``multi_turn_steering_scope`` differs from the value derived from their applied-positions mode, the eval's category and the
    object's kind (the job-start default ``single_turn`` the common case; a system-instruction object steers no positions, so its scope is
    ``not_applicable`` whatever mode the row carries), counted by (model, category) and by object kind on the dry run; ``--apply`` writes
    ONE ``run_patch`` per run with the derived scope into the run's own record directory, never the database (the landing window's
    reconcile slot runs the dry run and then the apply, and the records fold in that window's ingest). ``--verify`` is the predicate alone
    over the folded database, read-only and without the hold: the inconsistent runs listed by name. Exit 0 when every selected run was
    written (or on the dry run, or when the verify lists none); exit 2 when ``--apply`` left a run held (a record stating its mode or
    scope -- this pass's own correction, a standup's endpoint patch, a full ``run`` record carrying either column -- is in the log and not
    folded yet) -- the operator asked for a write that did not happen, and the held runs are listed; exit 3 when ``--verify`` lists
    inconsistent runs (the window reconciles once more and verifies again)."""
    import sqlite3
    from . import reconcile as RC
    try:
        rep = RC.verify_multi_turn_scope(root) if args.verify else RC.reconcile_multi_turn_scope(root, apply=bool(args.apply))
    except (OSError, ValueError, DB.SchemaDrift, sqlite3.Error) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rep.as_dict(), indent=1, default=str))
    else:
        print(RC.format_scope_report(rep))
    if args.verify:
        return 3 if rep.runs else 0
    if args.apply:
        for p in rep.written:
            print(p)
        if rep.held:
            print(f"{len(rep.held)} run(s) held: a record stating each one's mode or scope (this pass's own correction, a standup's "
                  f"endpoint patch, or a full run record carrying either column) is already in the log and not folded yet; ingest first, "
                  f"then re-run (nothing was written for them)", file=sys.stderr)
            return 2
    return 0


def _cmd_reconcile_correction(args, root: str, verify_fn, reconcile_fn, format_fn, held_why: str, check_fn=None, **kwargs) -> int:
    """The shared shape of a correction action's command (``env-comments``, and every later action with a verify): the dry run (the
    default) prints the report and writes nothing; ``--apply`` writes one record per selected row into the row's own record directory,
    never the database, exit 2 when a row was HELD behind an unfolded record, or refused by name (the operator asked for a write that
    did not happen; the held rows are listed); ``--verify`` is the predicate alone over the folded database, read-only, exit 3 when it
    lists rows (the landing window reconciles once more and verifies again); ``--check-sources`` (``check_fn``, env-comments) is the
    coverage of the sources alone, read-only, exit 3 when it lists rows. Exit 1 names a refusal of the pass itself (no log, no database,
    a row the pass cannot read, a named source file that is not there). ``kwargs`` ride to the action's functions (env-comments'
    ``sources`` and ``leave_unsourced``)."""
    import sqlite3
    check = bool(getattr(args, "check_sources", False))
    try:
        rep = (check_fn(root, **kwargs) if check else verify_fn(root, **kwargs) if args.verify
               else reconcile_fn(root, apply=bool(args.apply), **kwargs))
    except (OSError, ValueError, DB.SchemaDrift, sqlite3.Error) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rep.as_dict(), indent=1, default=str))
    else:
        print(format_fn(rep))
    if check or args.verify:
        return 3 if rep.listed else 0
    if args.apply:
        for p in rep.written:
            print(p)
        if rep.held:
            print(f"{len(rep.held)} row(s) held: {held_why}; ingest first, then re-run (nothing was written for them)", file=sys.stderr)
            return 2
    return 0


def cmd_pending(args) -> int:
    from . import pending as P
    from . import worklist as W
    root = _root(args)
    session = args.session or W.default_session()
    judge = None
    if args.judge:
        try:
            with DB.reading(DB.db_path(root)) as conn:
                judge = P.resolve_judge_id(conn, args.judge)
        except P.UnknownJudge as exc:
            print(f"pending-judgment: {exc}", file=sys.stderr)
            return 2
    if args.claim:
        if not (args.run_id and judge):
            print("--claim needs --run-id and --judge (--session defaults to $SISTEER_SESSION_NAME, else $USER@host)", file=sys.stderr)
            return 2
        try:
            jid = P.claim(root, args.run_id, judge, session, force=args.force, stale_hours=args.stale_hours)
        except (P.ClaimHeld, P.ClaimNotFolded) as exc:
            # `claim refused: <Class>: <message>` -- the class name (ClaimHeld / ClaimNotFolded / ClaimPendingCorrection / ClaimNotPending) lets a consumer
            # class the refusal by type without parsing the message; the grading daemon prints the same shape (2026-10-04)
            print(f"claim refused: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(jid)
        return 0
    if args.release:
        if not judge:
            print("--release needs --judge", file=sys.stderr)
            return 2
        try:
            ok = P.release(root, args.release, judge, session)
        except P.ClaimNotFolded as exc:
            print(f"release refused: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print("released" if ok else "no in-progress claim by this session")
        return 0 if ok else 1
    with DB.reading(DB.db_path(root)) as conn:
        rows = P.pending(conn, judge, args.campaign)
    if args.json:
        _print_rows(rows, True)
        return 0
    keys = ("run_id", "campaign", "eval_id", "lifecycle_state", "required_judge_id", "pending_reason", "required_coverage",
            "items_missing", "claim_session", "claim_age_s", "store_uri")
    _print_rows([{k: r.get(k) for k in keys} for r in rows], False)
    return 0


def cmd_judgment(args) -> int:
    """``judgment show ID [--json]`` / ``judgment supersede ID (--superseded-by X | --no-successor) --reason TEXT [--ruled-by WHO]
    [--provenance-kind user_direct|relayed --relayed-by S --message-utc T] [--allow-cross-run] [--dry-run] [--json]``
    (:mod:`sisteer.registry.supersede`). ``show`` prints the live row with its stamps, the identity form that mints
    its id, its successor and predecessors and the run's pending state for its judge. ``supersede`` retires a complete or partial judgment
    as a RECORD under ``records/supersedes/<run_dir>/`` (never a database edit): every guard rail refuses by name with exit 1 and nothing
    written, the assembled record is validated before the write, the live rows are re-read under the write lock immediately before it,
    and the record is folded at once -- exit 0 when the live row then reads it, 1 (written, path printed, the note on stderr) when a
    later-stamped producer record won the fold. ``show`` and ``--dry-run`` write nothing, so they resolve the root WITHOUT the
    ``stamp_last_run`` mark: a read-only command must not leave ``state/last_run/<cmd>.json`` under the root."""
    import sqlite3
    from . import supersede as SUP
    writes = args.action == "supersede" and not args.dry_run
    root = _root(args) if writes else os.path.abspath(args.root or os.environ.get("REGISTRY_ROOT", DB.DEFAULT_ROOT))
    try:
        if args.action == "show":
            with DB.reading(DB.db_path(root)) as conn:
                info = SUP.show(conn, args.judgment_id)
            if args.json:
                print(json.dumps(info, indent=1, default=str))
            else:
                for k, v in info.items():
                    print(f"{k}: {v if v is None or isinstance(v, (str, int, float)) else json.dumps(v, default=str)}")
            return 0
        rep = SUP.supersede(root, args.judgment_id, superseded_by=args.superseded_by, no_successor=args.no_successor, reason=args.reason or "",
                            ruled_by=args.ruled_by, provenance_kind=args.provenance_kind, relayed_by=args.relayed_by, message_utc=args.message_utc,
                            allow_cross_run=args.allow_cross_run, dry_run=args.dry_run, session=args.session)
    except SUP.SupersedeRefusal as exc:
        print(f"judgment {args.action} refused ({exc.field}): {exc.reason}", file=sys.stderr)
        return 1
    except (OSError, DB.SchemaDrift, DB.LockHeld, sqlite3.Error) as exc:
        print(f"judgment {args.action}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rep.as_dict(), indent=1, default=str))
    else:
        print("\n".join(rep.lines()))
    if rep.note:
        print(rep.note, file=sys.stderr)
    return 0 if (rep.dry_run or rep.in_force is not False) else 1


def cmd_refusals(args) -> int:
    """``refusals list [--still-refused] [--type KIND] [--limit N] [--json]`` / ``refusals explain <sha | record_uri> [--json]`` /
    ``refusals summary [--json]`` (:mod:`sisteer.registry.refusals`): the refusal facts read back. Read-only --
    the root resolves WITHOUT the ``state/last_run`` stamp, as ``cap-check`` does."""
    import sqlite3
    from . import refusals as RF
    root = _root_unstamped(args)
    try:
        with DB.reading(DB.db_path(root)) as conn:
            if args.action == "list":
                rows = RF.list_facts(conn, still_refused=bool(args.still_refused), record_type=args.type, limit=None if args.all else args.limit)
                out: object = rows
                lines = RF.list_lines(rows)
            elif args.action == "summary":
                rep = RF.summary(conn)
                out, lines = rep, RF.summary_lines(rep)
            else:
                if not args.ident:
                    print("refusals explain: name the record sha (64 hex) or the uri it was refused under", file=sys.stderr)
                    return 2
                rep = RF.explain(root, conn, args.ident)
                out, lines = rep, RF.explain_lines(rep)
    except (OSError, DB.SchemaDrift, sqlite3.Error) as exc:
        print(f"refusals {args.action}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(out, indent=1, default=str))
    else:
        print("\n".join(lines))
    return 0


def cmd_observations(args) -> int:
    """``observations summary [--json]`` / ``observations list [--run RUN] [--kind KIND] [--limit N | --all] [--json]``
    (:mod:`sisteer.registry.observations`; 2026-10-04): the fold observations read back -- what the harvester observed about a record at the
    fold without refusing it (today: ``extra.serving_jobs`` restated as a strict subset of the folded set). Read-only -- the root resolves
    WITHOUT the ``state/last_run`` stamp, as ``refusals`` does."""
    import sqlite3
    from . import observations as OBS
    root = _root_unstamped(args)
    try:
        with DB.reading(DB.db_path(root)) as conn:
            if args.action == "list":
                rows = OBS.list_observations(conn, run_id=args.run, kind=args.kind, limit=None if args.all else args.limit)
                out: object = rows
                lines = OBS.list_lines(rows)
            else:
                rep = OBS.summary(conn)
                out, lines = rep, OBS.summary_lines(rep)
    except (OSError, DB.SchemaDrift, sqlite3.Error) as exc:
        print(f"observations {args.action}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(out, indent=1, default=str))
    else:
        print("\n".join(lines))
    return 0


def cmd_cap_check(args) -> int:
    """``cap-check [--campaign C] [--all] [--json]`` (:mod:`sisteer.registry.capcheck`): the token-cap ladder
    facts of the live runs, read-only. The record rule (``validate.run_cap_ladder_refusal``) re-applied to every live row that states a
    ladder index -- a disagreement means the live table holds a row a replay would refuse, the one finding that exits 1; the legacy
    shapes the validator admits (a ladder stated with no index, an empty ladder with no index, the runs on a multi-level instrument
    stating no index), counted and listed; and the instrument cross-check the fold never does (a cap outside the instrument's ladder, on
    every instrument with a ladder -- one level or several -- the runs on a ladderless instrument counted as unchecked; a run ladder unlike
    the instrument's). Lists are bounded to the first 20 rows per bucket unless ``--all``; counts are whole. Writes
    nothing, so it resolves the root WITHOUT the ``stamp_last_run`` mark, as ``judgment show`` does."""
    import sqlite3
    from . import capcheck as CC
    root = os.path.abspath(args.root or os.environ.get("REGISTRY_ROOT", DB.DEFAULT_ROOT))
    try:
        with DB.reading(DB.db_path(root)) as conn:
            rep = CC.report(conn, campaign=args.campaign, limit=None if args.all else CC.DEFAULT_LIMIT)
    except (OSError, DB.SchemaDrift, sqlite3.Error) as exc:
        print(f"cap-check: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        print("\n".join(CC.lines(rep)))
    return int(rep["exit_code"])


def cmd_worklist_parity(args) -> int:
    """Acceptance "Grading queue": the pending set for one judge against a judging driver's own unit list
    (``units_*.json``: store → object → eval), differences by run_id with the reason; exit 0 iff none."""
    from . import pending as P
    root = _root(args)
    try:
        units = P.load_units(args.units)
    except (OSError, ValueError) as exc:
        print(f"worklist-parity: {exc}", file=sys.stderr)
        return 2
    try:
        with DB.reading(DB.db_path(root)) as conn:
            rep = P.worklist_parity(conn, args.judge, units, campaign=args.campaign)
    except P.UnknownJudge as exc:
        print(f"worklist-parity: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rep.as_dict(), indent=1, default=str))
    else:
        print(f"judge {rep.judge_id}: {len(units)} units, {len(rep.matched)} matched, {rep.n_differences} differences")
        for kind, ident, reason in rep.differences():
            print(f"{kind}\t{ident}\t{reason}")
    return 0 if rep.n_differences == 0 else 1


def parse_shard(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    """``i/K`` -> ``(i, K)`` with ``0 <= i < K``; None for None."""
    if spec is None:
        return None
    m = re.fullmatch(r"(\d+)/(\d+)", str(spec).strip())
    if not m or int(m.group(2)) < 1 or not 0 <= int(m.group(1)) < int(m.group(2)):
        raise ValueError(f"--shard takes i/K with 0 <= i < K, not {spec!r}")
    return int(m.group(1)), int(m.group(2))


@contextlib.contextmanager
def _profiling(path: Optional[str]):
    """cProfile the block into ``path`` (pstats format, ``python -m pstats FILE``), written whether or not the block finished; no
    path, no profiler."""
    if not path:
        yield
        return
    import cProfile
    prof = cProfile.Profile()
    prof.enable()
    try:
        yield
    finally:
        prof.disable()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        prof.dump_stats(path)



def cmd_adapters(args) -> int:
    from . import harvest as H
    from .adapters import load_all
    from .adapters.base import IdentityMap
    from .adapters.runner import ClusterRead, run_pass
    if getattr(args, "force", False) and not args.reread:
        print("adapters run: --force applies to a re-read only (--reread --force); the incremental pass publishes a new revision when "
              "an input or an adapter's EMISSION_VERSION changed, and nothing otherwise", file=sys.stderr)
        return 2
    reader_state_path = getattr(args, "reader_state", None)
    if reader_state_path and args.reread:
        print("adapters run: --reread is refused with --reader-state: a re-read re-activates a committed revision from its committed bytes, "
              "which only the grading host holds (reactivation is the grading host's); run it there", file=sys.stderr)
        return 2
    try:
        shard = parse_shard(getattr(args, "shard", None))
    except ValueError as exc:
        print(f"adapters run: {exc}", file=sys.stderr)
        return 2
    root = _root(args)
    reg = load_all()
    if args.adapter:
        reg = {k: v for k, v in reg.items() if k in args.adapter}
    if not reg:
        print("no adapters registered in this checkout; nothing to run", file=sys.stderr)
        return STUB_EXIT
    from .adapters import dimensions as DIM
    # the adapters read `legacy_spellings["model_ids"]` (HF repo / lane key / tag -> model_id) and `["si_texts"]`
    # (si_id -> sha); built from the seeds the registry was loaded from, never typed in a second place
    legacy = DIM.adapter_legacy_spellings(args.seeds or DIM.DEFAULT_SEEDS_DIR)
    profile_to = getattr(args, "profile", None)
    # identity beside addressing: the environment's home map plus every explicit LOCAL=CANONICAL pair
    pairs = dict(IdentityMap.from_environ().as_dict())
    for spec in getattr(args, "identity_map", None) or []:
        local, sep, canonical = str(spec).partition("=")
        if not sep:
            print(f"adapters run: --identity-map takes LOCAL=CANONICAL, not {spec!r}", file=sys.stderr)
            return 2
        pairs[local] = canonical
    try:
        imap = IdentityMap.build(pairs)
    except ValueError as exc:
        print(f"adapters run: {exc}", file=sys.stderr)
        return 2
    from . import cluster_reads as CR
    cluster = None
    producer = None
    hash_cache = None
    if reader_state_path:
        from . import reader_state as RS
        try:
            sidecar = RS.open_reader_state(reader_state_path, list(reg))
        except RS.ReaderStateError as exc:
            print(f"adapters run: {exc}", file=sys.stderr)
            return 2
        read_copy = {}
        if getattr(args, "read_copy", None):
            with open(args.read_copy, "r", encoding="utf-8") as fh:
                read_copy = json.load(fh)
        cluster = ClusterRead.from_sidecar(sidecar, read_copy)
        producer = CR.cluster_producer(substrate=getattr(args, "substrate", None))
        if getattr(args, "pins", None):
            hash_cache = CR.GenerationHashCache.from_pins_file(args.pins)
        conn_cm = contextlib.closing(sidecar)
    else:
        conn_cm = DB.reading(DB.db_path(root))
    from .adapters.base import GradingHostStoreReadRefused
    try:
        with conn_cm as conn, _profiling(profile_to):       # the pass only, not the fold; the connection closes with the block
            rep = run_pass(root, [f() for f in reg.values()], conn=conn, full=args.full, legacy_spellings=legacy, reread=args.reread,
                           force=getattr(args, "force", False), retry_refused=getattr(args, "retry_refused", False), producer=producer,
                           hash_cache=hash_cache, identity_map_=imap, shard=shard, cluster=cluster, save_cache=cluster is None)
    except GradingHostStoreReadRefused as exc:
        # no store access scope on this host -- adapters read on Slurm; exit 5 by name, nothing read
        print(f"refused: {exc}", file=sys.stderr)
        return STORE_READ_REFUSED_EXIT
    summary = rep.as_dict()
    if profile_to:
        summary["profile"] = profile_to
    if reader_state_path:
        summary["reader_state"] = reader_state_path
    if shard is not None:
        summary["shard"] = {"index": shard[0], "count": shard[1]}
    if imap:
        summary["identity_map"] = imap.as_dict()
    print(json.dumps(summary, indent=1))
    if getattr(args, "summary_out", None):
        CR.write_shard_summary(args.summary_out, rep, summary, adapters=list(reg), shard=shard, root=root, producer=producer or {})
    for r in rep.results:
        if r.refused_known:
            print(f"refused (known to the grading host): {r.adapter} {r.group_identity or r.group}: {r.error}", file=sys.stderr)
        elif r.activation_pending:
            # on the record log, not folded yet: neither held by the harvester nor refused by the pass -- the fold is pending
            print(f"pending fold: {r.adapter} {r.group}: {r.error}", file=sys.stderr)
        elif r.deferred:
            print(f"deferred: {r.adapter} {r.group}: {r.error}", file=sys.stderr)
        elif r.error:
            print(f"{'held' if r.held else 'refused'}: {r.adapter} {r.group}: {r.error}", file=sys.stderr)
        elif r.emitted_nothing:
            # a silent read counts beside the refusals under its own label: the read was ACCEPTED (its status row says ok = 0), so
            # it carries no error, and a summary that folded it into `groups_refreshed` alone is how it read as "never run"
            print(f"emitted nothing (unexplained): {r.adapter} {r.group}: read its store inputs and emitted no record, recording no reason; "
                  f"its status row is filed ok = 0 (revision {r.revision_id})", file=sys.stderr)
    for adapter_name, groups in sorted(rep.refused_standing.items()):
        # a refusal left standing is not re-read on a quiet pass (nothing about it changed); it stays on the page's adapter-pass rows
        named = ", ".join(groups[:5]) + (f" (+{len(groups) - 5} more)" if len(groups) > 5 else "")
        print(f"refusals standing (not re-read; an input change, an EMISSION_VERSION change, --retry-refused or --full reads them again): "
              f"{adapter_name}: {len(groups)} group(s): {named}", file=sys.stderr)
    # the fold: only when this pass published something, and never under --no-ingest (the harvester's own tick folds what the pass
    # wrote; the adapters timer must not hold harvester.lock for a full cycle over nothing). The line stays parseable either way.
    skipped = ("--reader-state (a cluster shard never folds)" if reader_state_path else "--no-ingest" if args.no_ingest
               else "nothing published" if not any(r.files for r in rep.results) else None)
    if skipped:
        print(json.dumps({"ingest_ok": 0, "refused": 0, "blocked": 0, "ingest": f"skipped: {skipped}"}))
    else:
        ing = H.ingest_under_lock(root, workers=args.workers)
        print(json.dumps({"ingest_ok": ing.ok, "refused": ing.refused, "blocked": ing.blocked}))
    return 0


def cmd_reader_state(args) -> int:
    from . import reader_state as RS
    root = _root(args)
    try:
        if args.action == "export":
            if not args.out:
                print("reader-state export: --out FILE is required", file=sys.stderr)
                return 2
            rep = RS.export(root, args.adapter or [], args.out, uri_home=args.uri_home, repo_root=args.repo_root)
            print(json.dumps(rep.as_dict(), indent=1))
            return 0
        if not args.file:
            print("reader-state show: --file FILE is required", file=sys.stderr)
            return 2
        conn = RS.open_reader_state(args.file)
        try:
            print(json.dumps(RS.meta(conn), indent=1))
        finally:
            conn.close()
        return 0
    except RS.ReaderStateError as exc:
        print(f"reader-state {args.action}: {exc}", file=sys.stderr)
        return 2


def cmd_decode_policy(args) -> int:
    from . import decode_policy as DP
    root = _root(args)
    paths = [args.out] if args.out else DP.export_paths()
    with DB.reading(DB.db_path(root)) as conn:
        if args.action == "export":
            h = DP.write_export(conn, paths)
            print(f"{h['n_policies']} policies, {h['n_sampling_modes']} sampling modes -> {', '.join(paths)} "
                  f"(ingest_seq {h['ingest_seq']}, policy_set_sha256 {h['policy_set_sha256'][:16]}…)"
                  + ("  (the table is empty: run `registry dimensions load`)" if h["n_policies"] == 0 else ""))
            return 0
        diff = DP.check_all(conn, paths)
        if not diff:
            print(f"committed export ({', '.join(paths)}) equals the registry")
            return 0
        print("committed export DIFFERS from the registry: " + "; ".join(diff))
        return 1


def cmd_dimensions(args) -> int:
    """Load the dimension seeds in two halves. The disk-bound half — reading the seed YAMLs, re-verifying the SI text shas,
    building and validating every record — runs first and takes no lock. Then, under the harvester lock, the revision
    directory is published by ONE rename and a harvester cycle folds it: the directory goes in ONE transaction, before
    anything else the cycle found (``harvest.Ingestor.apply_dimension_bundle``), so a reader of the paper objects sees the
    previous dimension set or the new one and never a mix. The report carries the timings: ``lock_hold_s`` (publication plus
    the whole cycle, which also folds every other pending record file) and ``dimension_write_s`` (the seconds the dimension
    transaction itself held the database's write lock — the number a reader could ever wait on)."""
    import time
    from . import harvest as H
    from .adapters import dimensions as DIM
    root = _root(args)
    if args.action != "load":
        return 2
    H.ensure_layout(root)
    t_build = time.monotonic()
    try:
        bundle = DIM.build_seed_bundle(root, args.seeds or DIM.DEFAULT_SEEDS_DIR, verify_si=not args.no_verify_si)
    except DIM.SeedRefusal as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    rep = {"revision": bundle.revision_id, "records": bundle.counts(), "warnings": bundle.warnings,
           "build_s": round(time.monotonic() - t_build, 3)}
    try:
        with DB.write_lock(root, timeout_s=args.lock_timeout):
            t_lock = time.monotonic()
            paths = DIM.publish_seed_records(root, bundle, force=args.force)
            rep["files_written"] = len(paths)
            if not paths:
                rep["note"] = "this seed revision is already on the record log; nothing written (--force to rewrite)"
            if not args.no_ingest:
                try:
                    # the same wall-clock budget as `ingest --budget-s` (default none): the dimension bundle itself is never budgeted
                    # (applied first, whole), the cycle's other pending work is -- this cycle runs under the lock with no unit bound
                    # every held revision re-attempts in this cycle when it is the window's UNCAPPED load (no budget: the one cycle with the
                    # timers stopped and the whole backlog in hand) or when --retry-held says so
                    budget = getattr(args, "budget_s", None)
                    ing = H.ingest_once(root, workers=args.workers, budget_s=budget, retry_held=bool(getattr(args, "retry_held", False) or budget is None))
                except DB.SchemaDrift as exc:
                    print(f"refused: {exc}", file=sys.stderr)
                    return 3
                rep["ingest"] = {"ok": ing.ok, "refused": ing.refused, "blocked": ing.blocked,
                                 "dimension_transactions": ing.dimension_transactions,
                                 "dimension_write_s": round(ing.dimension_write_s, 3),
                                 "transactions": ing.transactions, "longest_transaction_s": round(ing.longest_transaction_s, 3),
                                 "stopped": ing.stopped, "revisions_deferred_budget": ing.revisions_deferred_budget,
                                 "plain_files_deferred_budget": ing.plain_files_deferred_budget}
                for r in ing.refusals[:50]:
                    print("refused:", r, file=sys.stderr)
            rep["lock_hold_s"] = round(time.monotonic() - t_lock, 3)
    except DB.LockHeld as exc:
        print(f"refused: {exc} (another writer holds the harvester lock; the seeds were built and nothing was published)", file=sys.stderr)
        return 1
    print(json.dumps(rep, indent=1))
    return 0


def cmd_publish_figures(args) -> int:
    """Register every figure file in ``--dir`` as an artifact (published implies recorded); with
    ``--publish-to`` copy ONLY registered, sha-verified files there and refuse any file without a record;
    with ``--verify-dir`` refuse any file in that directory whose bytes have no artifact row."""
    from . import publish as PUB
    root = _root(args)
    files = PUB.figure_files(args.dir)
    if args.checkout:
        untracked = PUB.untracked_in_git(files, args.checkout)
        if untracked:
            print("refusing: untracked figure file(s): " + ", ".join(untracked), file=sys.stderr)
            return 1
    written = []
    if not args.verify_only:
        written = PUB.register_files(root, files, args.notebook_commit)
    published: List[str] = []
    with DB.reading(DB.db_path(root)) as conn:
        if args.publish_to:
            try:
                rep = PUB.publish_dir(conn, args.dir, args.publish_to, checkout=args.checkout)
            except PUB.Unregistered as exc:
                print(str(exc), file=sys.stderr)
                return 1
            published = rep.published
        if args.verify_dir:
            unregistered = PUB.unregistered_in_dir(conn, args.verify_dir)
            if unregistered:
                print("refusing: published file(s) with no artifact row: " + ", ".join(unregistered), file=sys.stderr)
                return 1
    print(json.dumps({"registered": len(written), "files": files, "published": published,
                      "verified_dir": args.verify_dir}, indent=1))
    return 0


def cmd_register_run(args) -> int:
    """Submit-time registration: validate the plan STRICTLY (an unresolved instrument refuses,
    exit 2, naming the field and what it would need), write ``run`` + ``run_state queued`` (+ a
    ``run_patch`` binding the scheduler job id / name), store the plan beside the job under
    ``--pending-dir/<job-id>.json`` so the job finds its own identity, and push."""
    from . import emit, emit_cli
    return emit_cli.register(plan_path=args.plan, spool=args.spool, job_id=args.job_id, job_name=args.job_name,
                             pending_dir=args.pending_dir, check_only=args.check_only, submit_sha_pin=args.sha_pin,
                             script="registry register-run", push=not args.no_push)


def cmd_inbox_merge(args) -> int:
    """Move what the clusters pushed (``inbox/<host>/…``) into the merged log (``records/…``) atomically —
    the same merge every harvester cycle performs, callable on its own (``scripts/registry/inbox_to_log.sh``)."""
    from . import harvest as H
    root = _root(args)
    moved = H.scan_inbox(root)
    counts = {}
    for _, status in moved:
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({"merged": counts.get("merged", 0), "duplicate": counts.get("duplicate", 0),
                      "conflict": counts.get("conflict", 0), "files": [p for p, _ in moved] if args.verbose else len(moved)}))
    return 0


def cmd_pull(args) -> int:
    """The pull backstop: ``--clusters`` (rsync each cluster's spool into the inbox — the default and the only source; the
    Modal pull was removed with its transport)."""
    from . import harvest as H
    root = _root(args)
    out: dict = {"clusters": H.pull_clusters(root)}
    print(json.dumps(out, indent=1))
    return 0


def cmd_objects(args) -> int:
    """``registry objects scan``: enumerate the object banks reachable from this machine and report, per bank, the
    objects found, the sidecars and weights files present, the refusals and the ``unverified`` field counts.
    ``--emit`` additionally writes the records through the incremental pass into ``--root``."""
    from .adapters import hpo_logs as HPO
    from .adapters import objects_bank as OB
    from .adapters import wandb_runs as WB
    from .adapters.fit_sidecars import load_seeds
    if args.action != "scan":
        return 2
    banks = list(OB.DEFAULT_BANKS)
    if args.bank_root:
        banks = []
        for spec in args.bank_root:
            label, _, rest = spec.partition("=")
            path, _, layout = rest.partition(":")
            if not (label and path and layout):
                print("--bank-root takes LABEL=PATH:LAYOUT (layout full22 | model/obj | bank/model/obj | stage)", file=sys.stderr)
                return 2
            banks.append((label, os.path.expanduser(path), layout))
    repo_configs = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "configs"))
    seeds_dir = args.seeds
    if seeds_dir is None and os.path.isfile(os.path.join(repo_configs, "registry", "legacy_spellings.yaml")):
        seeds_dir = os.path.join(repo_configs, "registry")
    conn = None
    db = DB.db_path(_root(args))
    if os.path.exists(db):
        conn = DB.connect(db, readonly=True)
    seeds = load_seeds(seeds_dir, conn=conn)
    pins = args.pins
    if pins is None and os.path.isfile(os.path.join(repo_configs, "sdfbase_full22_pins.json")):
        pins = os.path.join(repo_configs, "sdfbase_full22_pins.json")
    from .adapters.base import GradingHostStoreReadRefused
    try:
        rep = OB.scan(banks, seeds=seeds, read_blobs=not args.no_blobs, pins_path=pins, conn=conn)
        hpo = HPO.scan(seeds=seeds, conn=conn) if not args.no_hpo else None
    except GradingHostStoreReadRefused as exc:
        # the banks are stores; no scope on this host -> refused by name before any bank is listed
        if conn is not None:
            conn.close()
        print(f"refused: {exc}", file=sys.stderr)
        return STORE_READ_REFUSED_EXIT
    linked = HPO.link_objects(rep.payloads, hpo.searches) if hpo else 0
    wb_ids = [str(p["fit_wandb_run_id"]) for p in rep.payloads if p.get("fit_wandb_run_id")]
    wb_ids += [str(t["wandb_run_id"]) for t in (hpo.trials if hpo else []) if t.get("wandb_run_id")]
    resolved = WB.resolve(wb_ids, use_api=args.wandb, root=_root(args)) if wb_ids else {}     # the W&B entity / projects / mirror: this root's [wandb] table
    wb_filled = WB.apply_to_objects(rep.payloads, resolved) if resolved else 0
    if conn is not None:
        conn.close()
    out = {"objects": rep.as_dict(with_payloads=args.payloads),
           "hpo": ({"by_root": hpo.by_root, "searches": len(hpo.searches), "trials": len(hpo.trials), "refusals": hpo.refusals,
                    "parked": hpo.parked, "objects_linked_to_a_search": linked} if hpo else None),
           "wandb": {"run_ids_named": len(set(wb_ids)), "urls_resolved": sum(1 for r in resolved.values() if r.url), "objects_filled": wb_filled,
                     "api_consulted": bool(args.wandb),
                     "unresolved_reasons": sorted({r.reason for r in resolved.values() if r.reason})[:5]},
           "seeds": seeds.source}
    if args.json and not args.emit:
        print(json.dumps(out, indent=1, default=str))
    elif not args.json:
        print(OB.format_scan(rep))
        if hpo:
            print(f"hyperparameter searches: {len(hpo.searches)} ({len(hpo.trials)} trials; {linked} objects linked to a search; "
                  f"{len(hpo.refusals)} sweep directories refused, {len(hpo.parked)} parked by name (out-of-scope kinds))")
            for label, c in sorted(hpo.by_root.items()):
                print(f"- {label}: sweeps {c['sweeps']}, searches {c['searches']}, trials {c['trials']}, refused {c['refused']}, parked {c['parked']}")
            for r in hpo.refusals:
                print(f"    refused {r['dir']}: {r['why'][:200]}")
            for p in hpo.parked:
                print(f"    parked {p['dir']}: {p['kind_legacy']!r} ({HPO.PARKED_RULING})")
        print(f"W&B: {len(set(wb_ids))} run ids named by sidecars / trial logs, {out['wandb']['urls_resolved']} URLs resolved "
              f"({'API consulted' if args.wandb else 'API not consulted; pass --wandb'}), {wb_filled} objects filled")
        print(f"seeds: {seeds.source}")
    if args.emit:
        from . import harvest as H
        from .adapters.runner import run_pass
        root = _root(args)
        H.ensure_layout(root)
        adapters = [OB.ObjectsBankAdapter(banks, seeds=seeds, read_blobs=not args.no_blobs, pins_path=pins)]
        if not args.no_hpo:
            adapters.append(HPO.HpoLogsAdapter(seeds=seeds))
        conn = DB.connect(DB.db_path(root), readonly=True) if os.path.exists(DB.db_path(root)) else None
        try:
            prep = run_pass(root, adapters, conn=conn, full=args.full)
        finally:
            if conn is not None:
                conn.close()
        out["emitted"] = {"adapters": prep.adapters, "inputs_seen": prep.inputs_seen, "groups_refreshed": prep.groups_refreshed,
                          "groups_refused": prep.groups_refused, "records_written": prep.records_written}
        print(json.dumps(out if args.json else {"emitted": out["emitted"]}, indent=1, default=str))
    return 0 if not rep.refusals or args.allow_refusals else 1


def cmd_roster(args) -> int:
    from .adapters import roster as RO
    root = _root(args)
    if args.action == "transcribe":
        from . import roster_transcribe as T
        if not (args.paper_checkout and args.sha):
            print("roster transcribe needs --paper-checkout DIR and --sha SHA", file=sys.stderr)
            return 2
        try:
            res = T.transcribe(args.paper_checkout, args.sha, args.out or os.path.join("configs", "registry", "transcribed"), args.round or "r2",
                               args.version, python=args.paper_python, previous_roster=args.previous_roster, previous_rulings=args.previous_rulings)
        except (T.TranscriptionRefusal, RO.RosterRefusal) as exc:
            print(f"transcription refused: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({k: v for k, v in res.items() if k != "columns_at_sha"}, indent=1))
        return 0
    roster_path = args.roster or os.path.join("configs", "registry", f"roster_{args.round or 'r2'}_{args.version}.yaml")
    rulings_path = args.rulings or os.path.join("configs", "registry", f"rulings_{args.round or 'r2'}_{args.version}.jsonl")
    if not os.path.exists(rulings_path):
        rulings_path = None
    conn = None
    db = DB.db_path(root)
    if os.path.exists(db):
        conn = DB.connect(db, readonly=True)
    try:
        try:
            load_ = RO.load(roster_path, rulings_path, conn=conn, strict_counts=not args.no_count_check)
        except RO.RosterRefusal as exc:
            print(f"roster refused: {exc}", file=sys.stderr)
            return 1
        if args.action == "show":
            if args.json:
                print(json.dumps(load_.summary(), indent=1, default=str))
            else:
                print(RO.show_text(load_))
            return 0
        if args.action == "diff":
            if args.against:
                other = RO.load(args.against, args.against_rulings, strict_counts=not args.no_count_check)
                d = RO.diff_entries(load_.entries, other.entries)
                label = f"{roster_path} vs {args.against}"
            else:
                if conn is None:
                    print(f"no database at {db}; pass --against OTHER_ROSTER.yaml to diff two files", file=sys.stderr)
                    return 2
                d = RO.diff_entries(load_.entries, RO.db_entries(conn, load_.round_id))
                label = f"{roster_path} vs the database's roster_entry rows"
            print(json.dumps({"compared": label, "n_file": d["n_a"], "n_other": d["n_b"], "only_file": len(d["only_a"]), "only_other": len(d["only_b"]),
                              "changed": len(d["changed"]), "equal": d["equal"], "coarse_key_only_file": len(d["coarse_only_a"]),
                              "coarse_key_only_other": len(d["coarse_only_b"]), "coarse_equal": d["coarse_equal"]}, indent=1))
            if args.json:
                print(json.dumps(d, indent=1, default=str))
            else:
                for k in d["only_a"][:args.limit]:
                    print("only in file:", k["key"])
                for k in d["only_b"][:args.limit]:
                    print("only in other:", k["key"])
                for c in d["changed"][:args.limit]:
                    print("changed:", c["key"], c["diffs"])
            return 0 if d["equal"] else 1
        # load: write the records under <root>/rulings/<round>_<version>/ (the log), then ingest unless --no-ingest
        from . import harvest as H
        H.ensure_layout(root)
        try:
            paths = RO.write_records(root, load_)
        except RO.RosterRefusal as exc:
            print(f"roster load refused: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"written": paths, **{k: v for k, v in load_.summary().items() if k != "counts_by_comparison"}}, indent=1, default=str))
        if conn is not None:
            conn.close()
            conn = None
        if not args.no_ingest:
            rep = H.ingest_under_lock(root, workers=args.workers)
            print(json.dumps({"ingest_ok": rep.ok, "refused": rep.refused, "blocked": rep.blocked, "changed": rep.changed}))
            for r in rep.refusals[:20]:
                print("refused:", r, file=sys.stderr)
        return 0
    finally:
        if conn is not None:
            conn.close()


def cmd_ruling(args) -> int:
    from .adapters import roster as RO
    root = _root(args)
    if args.action == "add":
        if not (args.round and args.ruling_action and args.scope_kind and args.scope_json and args.reason and args.source_ref):
            print("ruling add needs --round --action --scope-kind --scope-json --reason --source-ref [--target-json] [--ruled-utc] [--ruled-by] [--id] [--expires-when-json]", file=sys.stderr)
            return 2
        raw = {"ruling_id": args.id or f"{args.round}-{args.ruling_action}-{R.utc_now().replace(':', '').replace('-', '').replace('.', '')[:15]}",
               "round_id": args.round, "action": args.ruling_action, "scope_kind": args.scope_kind, "scope_json": json.loads(args.scope_json),
               "target_json": json.loads(args.target_json) if args.target_json else {}, "reason": args.reason, "ruled_utc": args.ruled_utc or R.utc_now(),
               "ruled_by": args.ruled_by or "the user", "source_ref": args.source_ref,
               "expires_when_json": json.loads(args.expires_when_json) if args.expires_when_json else None, "extra_json": {"added_by": "registry ruling add"}}
        # who the ruling came from; `relayed` names the session and the message time so the reading can be checked against its source
        prov = {"kind": args.provenance_kind}
        if args.relayed_by:
            prov["relayed_by"] = args.relayed_by
        if args.message_utc:
            prov["message_utc"] = args.message_utc
        raw["provenance"] = prov
        try:
            payload = RO.normalise_ruling(raw, None, where="ruling add")
        except RO.RosterRefusal as exc:
            print(f"ruling refused: {exc}", file=sys.stderr)
            return 1
        from . import harvest as H
        from .identity import adapter_record_id
        H.ensure_layout(root)
        now = R.utc_now()
        rec = R.make_record("ruling", payload, RO.producer("registry ruling add"), adapter_record_id(payload, "ruling"), now)
        base = os.path.join(H.rulings_dir(root), "added", f"{now[:10]}-ruling-{payload['ruling_id']}")
        path = base + ".jsonl"
        if os.path.exists(path) or os.path.exists(path + ".gz"):
            # a record file is immutable once written: a second `ruling add` of the same id on the same day -- the idempotent
            # re-add, a corrected reason -- goes into a NEW file named by its record's sha, never appended to the one the harvester may
            # already have folded (the harvester refuses an appended ingested file by name); the fold keeps one row per ruling id
            path = f"{base}-{R.record_sha(rec)[:12]}.jsonl"
        R.write_records(path, [rec], exclusive=True)
        print(path)
        return 0
    with DB.reading(DB.db_path(root)) as conn:
        if args.action == "list":
            sql = "SELECT ruling_id, round_id, action, scope_kind, evidence_state, expired_utc, ruled_utc, ruled_by FROM ruling"
            conds, params = [], []
            if args.round:
                conds.append("round_id = ?"); params.append(args.round)
            if args.ruling_action:
                conds.append("action = ?"); params.append(args.ruling_action)
            if conds:
                sql += " WHERE " + " AND ".join(conds)
            _print_rows(conn.execute(sql + " ORDER BY ruled_utc, ruling_id", params).fetchall(), args.json)
            return 0
        # expire-check: evaluate every self-retiring ruling against source_state without writing a record (dry run)
        from . import paper
        rows = conn.execute("SELECT * FROM ruling WHERE expires_when_json IS NOT NULL OR action = 'supersede'" + (" AND round_id = ?" if args.round else ""),
                            ([args.round] if args.round else [])).fetchall()
        out = []
        for r in rows:
            ruling = dict(r)
            ev = paper.evaluate_expiry(conn, ruling)
            stale = paper.stale_check(conn, ruling)
            out.append({"ruling_id": ruling["ruling_id"], "action": ruling["action"], "current_evidence_state": ruling["evidence_state"],
                        "current_expired_utc": ruling["expired_utc"], "evaluation": (ev.evidence_state if ev else "ok"),
                        "would_expire": bool(ev and ev.expired), "stale": stale, "evidence": (ev.evidence if ev else {})})
        print(json.dumps(out, indent=1, default=str) if args.json else "\n".join(
            f"{o['ruling_id']}: {o['action']} state={o['current_evidence_state']} -> {o['evaluation']}{' EXPIRED' if o['would_expire'] else ''}{' STALE' if o['stale'] else ''}" for o in out))
        return 0


def cmd_analysis(args) -> int:
    """``analysis file --spec <json> [--dry-run] [--json]`` / ``analysis list [--name N] [--json]`` (:mod:`sisteer.registry.analysis`):
    file a prefill-only or offline analysis over ingested runs and objects as ledger-fact records under records/analyses/<day>/ (the
    report and page files registered as artifacts first), or read the filed analyses back. The database is READ for the input check and
    the listing, never written: the harvester folds the records on its next cycle. A dry run prints the plan and writes nothing (and
    stamps no state/last_run, as a read-only command must not)."""
    import sqlite3
    from . import analysis as AN
    if args.action == "list":
        root = _root_unstamped(args)
        try:
            with DB.reading(DB.db_path(root)) as conn:
                rows = AN.list_analyses(conn, name=args.name)
                detail = {r["analysis_id"]: {"inputs": AN.inputs_of(conn, str(r["analysis_id"])), "artifacts": AN.artifacts_of(conn, str(r["analysis_id"]))}
                          for r in rows} if args.json else {}
        except (OSError, DB.SchemaDrift, sqlite3.Error) as exc:
            print(f"analysis list: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps([dict(r, **detail[r["analysis_id"]]) for r in rows], indent=1, default=str))
        else:
            print("\n".join(AN.list_lines(rows)))
        return 0
    if not args.spec:
        print("analysis file needs --spec <json>", file=sys.stderr)
        return 2
    root = _root_unstamped(args) if args.dry_run else _root(args)
    now = R.utc_now()
    try:
        raw = AN.load_spec(args.spec)
        try:
            with DB.reading(DB.db_path(root)) as conn:
                plan = AN.build_plan(raw, now, conn)
        except (OSError, DB.SchemaDrift, sqlite3.Error) as exc:
            print(f"analysis file: cannot read the database under {root} for the input check: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        lines = AN.plan_lines(plan)
        if plan.missing_inputs:
            print("\n".join(lines), file=sys.stderr)
            print(f"analysis file REFUSED: {len(plan.missing_inputs)} input(s) not in this registry -- nothing written", file=sys.stderr)
            return 1
        if args.dry_run:
            print("\n".join(lines + ["dry run: nothing written"]))
            _print_abandoned_stagings(root)
            return 0
        written = AN.write_filing(root, plan, AN.producer_block(), now)
    except AN.AnalysisRefusal as exc:
        print(f"analysis file refused by name: {exc}", file=sys.stderr)
        _print_abandoned_stagings(root)
        return 2
    # the filing's own directory under records/analyses/<day>/ (published whole by one rename) and the header sha its children carry
    header = next(p for p in written if os.path.basename(p).endswith("-analysis.jsonl"))
    filing_sha = AN.filing_sha_of(header)
    if args.json:
        print(json.dumps({"analysis_id": plan.analysis_id, "name": plan.spec["name"], "written": written,
                          "filing_sha256": filing_sha, "filing_dir": os.path.dirname(header),
                          "inputs": [{"kind": k, "id": i} for k, i in plan.inputs],
                          "artifacts": [{"role": p.role, "kind": p.kind, "artifact_id": p.sha256, "uri": p.uri, "remote": p.remote,
                                         "known": p.known, "record_written": p.payload is not None, "same_bytes_as": p.same_bytes_as}
                                        for p in plan.artifacts]},
                         indent=1))
    else:
        print("\n".join(lines + [f"written {len(written)} record file(s) (filing {AN.filing_dir_name(filing_sha)}, published whole):"]
                        + [f"  {p}" for p in written]
                        + ["the harvester folds them on its next cycle (`registry ingest`); `registry analysis list` reads them back"]))
    _print_abandoned_stagings(root)
    return 0


def _print_abandoned_stagings(root: str) -> None:
    """Name, on stderr, every filing staged under ``staging/analyses/`` and never published (an interrupted filing -- or one another
    process is writing now): nothing of them is in the record log, and a dry run or a filing that said nothing about them would leave the
    interruption invisible. Refuses nothing: the operator removes the directory by hand."""
    from . import analysis as AN
    lines = AN.abandoned_lines(AN.abandoned_stagings(root))
    if lines:
        print("\n".join(lines), file=sys.stderr)


def cmd_revision(args) -> int:
    """``revision held``: the adapter revision markers the harvester holds -- the ``source_revision`` rows of ``blocked_records``
    (a pending revision's own directory, or a re-activation bundle of an accepted non-head revision), each with its kind
    (``blocked`` / ``refused`` / ``run_drop``) and reason; ``--all`` includes the stale ones (superseded, or older than seven days).
    ``revision accept --revision-id X --reason ...``: release ONE marker held for dropping run ids of the accepted head without a
    declaration (``kind = run_drop``) by publishing an activation bundle ``records/adapters/<adapter>/_activations/<rev>/<stamp>/``
    -- the held marker's own ``observation`` / ``withdrawal`` / ``source_state`` / ``source_watermark`` records re-emitted beside a
    fresh ``source_revision`` marker carrying the operator's declaration (``extra.declared_run_drops``: the run ids, or ``"*"``, the
    reason, who) and the hold it releases (``extra.operator_accept``). An append-only record, like ``ruling add``: the next harvester
    cycle folds the bundle with the files the held attempt already staged (they share the revision), the guard sees the declaration,
    the revision activates and the head moves. Refused, with the reason: a revision not held for a run drop (a declaration releases
    nothing else), or a pending one a later accepted revision superseded (``--force`` makes it the head)."""
    from . import harvest as H
    root = _root(args)
    held_sql = ("SELECT json_extract(b.extra_json, '$.revision_id') AS revision_id, sr.source_id, sr.adapter, sr.accepted, "
                "json_extract(b.extra_json, '$.kind') AS kind, b.missing_parent_table, b.missing_parent_id, b.first_seen_utc, b.stale, "
                "sr.superseded_by_revision_id, b.record_uri, json_extract(b.extra_json, '$.blocked_on') AS reason "
                "FROM blocked_records b LEFT JOIN source_revision sr ON sr.revision_id = json_extract(b.extra_json, '$.revision_id') "
                "WHERE b.record_type = 'source_revision'")
    if args.action == "held":
        with DB.reading(DB.db_path(root)) as conn:
            rows = conn.execute(held_sql + ("" if args.all else " AND b.stale = 0") + " ORDER BY b.first_seen_utc, b.record_uri").fetchall()
            _print_rows(rows, args.json)
        return 0
    if not (args.revision_id and args.reason and args.reason.strip()):
        print("revision accept needs --revision-id and --reason [--run-id ID ...] [--accepted-by NAME] [--force]", file=sys.stderr)
        return 2
    rev = args.revision_id
    with DB.reading(DB.db_path(root)) as conn:
        row = conn.execute("SELECT revision_id, source_id, adapter, observed_utc, accepted, refusal, superseded_by_revision_id "
                           "FROM source_revision WHERE revision_id = ?", (rev,)).fetchone()
        holds = [dict(r) for r in conn.execute(held_sql + " AND json_extract(b.extra_json, '$.revision_id') = ? ORDER BY b.first_seen_utc DESC, b.record_uri DESC",
                                               (rev,)).fetchall()]
    if row is None:
        print(f"revision accept refused: {rev} has no source_revision row (not published, or not folded yet)", file=sys.stderr)
        return 1
    row = dict(row)
    if not holds:
        state = "accepted" if int(row["accepted"]) else f"pending: {row['refusal'] or 'no refusal recorded'}"
        print(f"revision accept refused: the harvester holds no marker of {rev} ({state})", file=sys.stderr)
        return 1
    hold = holds[0]
    if hold["kind"] != H.RUN_SET_DROP_KIND:
        print(f"revision accept refused: {rev} is held on {hold['kind'] or 'a missing marker'} -- {hold['missing_parent_table']} "
              f"{hold['missing_parent_id']} ({hold['reason'] or hold['record_uri']}); a declaration releases nothing there", file=sys.stderr)
        return 1
    if not int(row["accepted"]) and row["superseded_by_revision_id"] and not args.force:
        print(f"revision accept refused: {rev} was superseded by the later accepted revision {row['superseded_by_revision_id']} of "
              f"{row['source_id']}; accepting it now would make it the head again -- pass --force if that is the intent", file=sys.stderr)
        return 1
    marker_uri = str(hold["record_uri"])
    marker_path = H.resolve_uri(root, marker_uri)
    if not marker_path or not os.path.isfile(marker_path):
        print(f"revision accept refused: the held marker file of {rev} is not on disk ({marker_uri})", file=sys.stderr)
        return 1
    recs = [rec for _, rec, _ in R.read_record_file(marker_path)]
    mrec = next((r for r in recs if r.get("record_type") == "source_revision" and (r.get("payload") or {}).get("revision_id") == rev), None)
    if mrec is None:
        print(f"revision accept refused: {marker_uri} carries no source_revision record for {rev}", file=sys.stderr)
        return 1
    from .adapters import roster as RO
    from .identity import adapter_record_id
    now = R.utc_now()
    payload = dict(mrec["payload"])
    extra = dict(payload.get("extra") or {})
    extra[H.DECLARED_RUN_DROPS_KEY] = {"run_ids": sorted(set(args.run_id)) if args.run_id else "*", "reason": args.reason.strip(),
                                      "declared_by": ["operator:registry revision accept"], "accepted_by": args.accepted_by or "the user"}
    extra["operator_accept"] = {"of_marker": marker_uri, "held_reason": str(hold["reason"] or ""), "accepted_utc": now}
    payload.update(observed_utc=now, accepted=1, extra=extra)
    payload.pop("refusal", None)
    out = [R.make_record("source_revision", payload, RO.producer("registry revision accept"), adapter_record_id(payload, "source_revision"), now)]
    # the marker's own membership and bookkeeping records, re-emitted with a fresh stamp (a re-activation bundle's way)
    out += [dict(r, written_utc=now) for r in recs if r.get("record_type") != "source_revision"]
    stamp = "".join(ch for ch in now if ch.isdigit())
    bundle = os.path.join(H.records_dir(root), "adapters", str(row["adapter"]), "_activations", rev, stamp)
    path = R.write_records(os.path.join(bundle, R.record_file_name(1, "source_revision")), out, exclusive=True)
    print(path)
    return 0


def cmd_check_paper_parity(args) -> int:
    from . import parity as P
    root = _root(args)
    levels = P.LEVELS if args.level in (None, "all") else tuple(l.strip() for l in args.level.split(","))
    try:
        res = P.check_paper_parity(args.paper_checkout, args.pin or P.checkout_sha(args.paper_checkout), root, round_id=args.round, levels=levels,
                                   paper_python=args.paper_python, report_dir=args.report, work_dir=args.work_dir, skip_sha_check=args.pin is None)
    except P.ParityRefused as exc:
        print(f"parity gate refused: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(res.summary(), indent=1, default=str))
    if res.refused:
        return 3
    return 0 if all(res.result_of(l) == "pass" for l in levels if l in res.levels) else 1


def cmd_paper_objects(args) -> int:
    from . import paper
    root = _root(args)
    with DB.reading(DB.db_path(root)) as conn:
        df = paper.paper_objects(conn, args.round)
        if args.json:
            print(df.to_json(orient="records", indent=1))
        else:
            print(df.to_string(max_rows=args.limit))
    return 0


def _lazy_module(module: str, fn: str, args) -> int:
    try:
        mod = __import__(f"sisteer.registry.{module}", fromlist=[fn])
    except ImportError as exc:
        print(f"registry: sisteer.registry.{module} cannot be imported in this checkout ({exc}); nothing to run", file=sys.stderr)
        return STUB_EXIT
    return int(getattr(mod, fn)(args) or 0)


# --------------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="registry", description="The results registry: one queryable source of truth for every "
                                "measurement (models, objects, evals, instruments, runs, judgments) and its provenance.")
    p.add_argument("--root", help="registry root (default $REGISTRY_ROOT or ~/registry)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create the registry root layout and an empty database")
    s.set_defaults(fn=cmd_init)
    s = sub.add_parser("migrate", help="apply pending schema migrations under the registry's write lock, then verify the database's shape against the code's DDL")
    s.add_argument("--lock-timeout", type=float, default=0.0, help="seconds to wait for the write lock (default 0: refuse at once, exit 4, when a writer holds it)")
    s.add_argument("--db")
    s.add_argument("--plan", action="store_true", help="read-only: version state, pending files, and what a new migration must rebuild")
    s.add_argument("--sql", action="store_true", help="with --plan: render that migration's SQL body for the next NNNN_<slug>.sql")
    s.add_argument("--map-site-labels", action="store_true",
                   help="with --sql: copy every site-label column of a rebuilt table through the migration runner's alias table (the form of "
                        "the site-label rename)")
    s.add_argument("--site-aliases", metavar="FILE",
                   help="the TOML file whose [site_aliases] table maps legacy site labels to the registry's labels. Default: <root>/site_aliases.toml "
                        "in root mode; with --db, only this file is read (absent: the empty table, and a legacy label is refused by name)")
    s.add_argument("--expect-site-aliases-sha256", metavar="SHA256",
                   help="the digest the site-alias table must have (site_aliases.digest); any other refuses before anything is read or applied")
    s.set_defaults(fn=cmd_migrate)
    s = sub.add_parser("ingest", help="one harvester cycle under the write lock: inbox merge, parse, fold, blocked queue, harvester facts")
    s.add_argument("--since", help="(accepted for compatibility; the harvester ingests every new or changed record file)")
    s.add_argument("--records", help="(the record log is <root>/records; pass --root)")
    s.add_argument("--once", action="store_true", default=True)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--pull", action="store_true", help="also run the cluster pull backstops (rsync into the inbox), BEFORE the write lock is taken")
    s.add_argument("--lock-timeout", type=float, default=30.0, metavar="SECONDS",
                   help="seconds to wait for the write lock before refusing with exit 4 (default 30)")
    s.add_argument("--retry-refused", action="store_true",
                   help="re-attempt every revision held on a refused constituent this cycle, even one whose refusing file and judging "
                        "code are unchanged (the standing rule skips such an exact hold); for a refusal that depended on other rows "
                        "rather than on the record's bytes or the code")
    s.add_argument("--retry-held", action="store_true",
                   help="re-attempt every held revision directory this cycle, even one whose attempt identity (its files, marker, the judging code "
                        "and its parent's presence) has not moved since the attempt that recorded it -- the standing rule skips such a hold "
                        "until something moves or [harvest] held_sweep_s passes; the landing window's uncapped cycles set this")
    s.add_argument("--html", action="store_true", help="re-render the status page when anything changed")
    s.add_argument("--honor-maintenance", action="store_true",
                   help="exit 0 at once, before the cluster pull and before the write lock, while <root>/state/maintenance.json exists "
                        "(a landing window's drain flag; the harvest unit's mode). The disk-pressure flag paused.json is the cycle's own")
    s.add_argument("--budget-s", type=float, default=None, metavar="SECONDS",
                   help="wall-clock budget for the cycle's fold, from the cycle's start: once it is spent the cycle takes no NEW plain "
                        "chunk or revision directory / activation bundle (the one in hand finishes), runs its closing steps and reports "
                        "`stopped: budget`; everything committed stays committed and the next cycle continues. Set it below the unit's "
                        "TimeoutStartSec so a long backlog is folded in bounded, durable cycles instead of being killed mid-fold (default: none)")
    s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("pull", help="run the pull backstop on demand: the clusters' spools (rsync)")
    s.add_argument("--clusters", action="store_true", help="rsync each cluster's record spool into the inbox (the default and only source)")
    s.set_defaults(fn=cmd_pull)
    s = sub.add_parser("paper-head", help="the paper-input head (max ingest_seq the paper reads) and the roster entries still without a "
                                          "judged run or a disposition — the rebuild trigger's inputs")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_paper_head)
    s = sub.add_parser("config", help="show the operating configuration (<root>/config.toml), one dotted key, write the default file, "
                                      "or --check both configuration files through the loader")
    s.add_argument("key", nargs="?", help="a dotted key, e.g. paper.checkout")
    s.add_argument("--json", action="store_true")
    s.add_argument("--write-default", action="store_true", help="write config.toml with the defaults when absent")
    s.add_argument("--force", action="store_true")
    s.add_argument("--paper-checkout", help="with --write-default: the paper checkout to name")
    s.add_argument("--check", action="store_true",
                   help="load config.toml AND the legacy config.json through the loader and exit 0 only on a clean configuration: "
                        "the removed Modal table, a destination of the removed kind anywhere in config.json's lists, or a file that "
                        "does not parse is a named refusal on stderr with exit 1 (the cutover window's validation after the conversion)")
    s.set_defaults(fn=cmd_config)
    s = sub.add_parser("rebuild", help="replay the record log into a fresh database (acceptance 1)")
    s.add_argument("--from", dest="from_dir", required=True)
    s.add_argument("--into", required=True)
    s.add_argument("--verify", action="store_true", help="diff every table (except the arrival ledger) against the live database")
    s.add_argument("--verify-against", help="path of the live database (default <root>/registry.sqlite)")
    s.add_argument("--workers", type=int, default=3)
    s.set_defaults(fn=cmd_rebuild)
    s = sub.add_parser("status", help="registry status: runs in flight, roster gaps, refused and blocked records, snapshots")
    s.add_argument("--html", action="store_true")
    s.add_argument("--json", action="store_true")
    s.add_argument("--out")
    s.add_argument("--refresh-heavy", action="store_true",
                   help="recompute the page's cached-with-refresh sections exactly (state/status_heavy.json; each under [status] "
                        "heavy_budget_s) into state/status_sections/; skipped while the ledger head has not moved since the last "
                        "refresh; the first run computes every section and derives the set from the measured times")
    s.add_argument("--force", action="store_true", help="with --refresh-heavy: refresh even when the ledger head has not moved")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("query", help="run one SQL statement against the database")
    s.add_argument("sql")
    s.add_argument("--json", action="store_true")
    s.add_argument("--parquet")
    s.set_defaults(fn=cmd_query)
    s = sub.add_parser("frame", help="a table or a view as a frame (optionally to parquet)")
    s.add_argument("name")
    s.add_argument("--where")
    s.add_argument("--parquet")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(fn=cmd_frame)
    s = sub.add_parser("history", help="every historical state of one natural key (run-scale from row_history; item-scale folded from the log)")
    s.add_argument("table")
    s.add_argument("key_json")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_history)
    s = sub.add_parser("mirrors", help="record-log fan-out: sync to a destination, verify its file set and checksums, or prune the "
                                        "checksum manifests (prune-manifests)")
    s.add_argument("action", choices=["sync", "verify", "prune-manifests"])
    s.add_argument("--dest", action="append", metavar="NAME",
                   help="only this destination; repeatable (--dest cluster_a --dest cluster_b runs both off ONE manifest write)")
    s.add_argument("--kind", choices=["local", "ssh"], help="only destinations of this kind (the weekly full verify: --kind ssh)")
    s.add_argument("--full", action="store_true", help="sync: rsync the WHOLE records/ and rulings/ trees and hash EVERY entry on the "
                                                        "ssh destinations (under srun when the cluster entry has slurm = true) instead "
                                                        "of the incremental copy and verify against the destination's CHECKSUMS.CURRENT "
                                                        "(the delta since it copied and hashed, plus gz twins and a random sample hashed)")
    s.add_argument("--sample", type=int, metavar="N", help="sync: carried entries re-hashed per incremental verify on the ssh "
                                                            "destinations for this run (default: [[clusters]] verify_sample, else 2000)")
    s.add_argument("--budget-s", type=float, help="sync: wall-clock budget in seconds from process start; an ssh destination's "
                                                   "incremental copy starts no new chunk when less than 1.5x the last chunk's time is left, "
                                                   "exits 0 and resumes on the next run (default 7200 = the unit's TimeoutStartSec)")
    s.add_argument("--keep", type=int, metavar="K", help="prune-manifests: manifests kept per side -- the newest K by ingest_seq, besides "
                                                          "each side's CHECKSUMS.CURRENT and every manifest a destination's pointer names "
                                                          "(default 96, about two days of half-hourly passes; the same rule every sync pass applies)")
    s.add_argument("--dry-run", action="store_true", help="prune-manifests: print the plan (counts and bytes per side, the oldest and "
                                                             "newest kept, the referenced manifests and why) and delete nothing; exit 0")
    s.add_argument("--json", action="store_true", help="prune-manifests: the plan as one JSON document after the text")
    s.set_defaults(fn=cmd_mirrors)
    s = sub.add_parser("stores", help="store copies for the cluster-read adapters: carry each canonical root to cluster A as an immutable, "
                                      "manifest-verified generation (mirror), publish one component as its producer with the PUBLISHED record and the "
                                      "directory seal (push, also --on-cluster), compare a cluster-A-origin mirror with its original (check), show the "
                                      "state (status), re-validate a canonical copy against a pinned generation (revalidate), check every registered "
                                      "adapter's declared roots against the components without opening a store (coverage), read every `current` in one "
                                      "batch (check-authority), verify a component's generation against its grading host copy for the one-time move "
                                      "(verify-authority), or re-attest a published generation's directory listings (attest)")
    from . import stores as ST
    ST.add_arguments(s)
    s.set_defaults(fn=lambda a: ST.cmd_stores(a, _root(a)))
    s = sub.add_parser("replica", help="the cluster replica of the registry database: a quiescent "
                                       "replayable head of the live database taken at a cursor-free boundary (head), the bootstrap replay of exactly "
                                       "that head's allowlist (build), the increment (ingest), canonical equality through frame_projection/1 (check), "
                                       "generations switched by compare-and-swap (publish, rollback, migrate, status), the landing job (land) and "
                                       "the C1 refold measurement (refold-compare)")
    from . import replica as RP
    RP.add_arguments(s)
    s.set_defaults(fn=lambda a: RP.cmd_replica(a, _root(a) if (a.action in ("head", "residual-class") and not a.db) else (os.path.abspath(a.root) if a.root else None)))
    s = sub.add_parser("snapshot", help="build a point-in-time snapshot (per-table parquet + run-level sqlite) and switch CURRENT")
    s.add_argument("--push", action="store_true", help="push to every configured destination (the clusters over rsync+ssh) and commit there")
    s.add_argument("--scope", choices=["full", "runlevel", "paper_frame"], default="full",
                   help="full = every table's parquet (daily / on demand); runlevel = the run-level sqlite + run-scale parquet only; "
                        "paper_frame = the committed export for the paper's companion repository (the paper's model set, locations "
                        "scrubbed, parquet + a gzip-split sqlite; requires --dest-root; never touches the registry's own snapshots/ + CURRENT)")
    s.add_argument("--if-due", action="store_true", help="build only when the cadence policy says so (in flight: run-level every 5 min; "
                                                          "full daily; idle hourly when the head moved), else exit 0")
    s.add_argument("--no-space-check", action="store_true")
    s.add_argument("--dest-root", metavar="DIR", help="paper_frame only, REQUIRED there: the directory the export is written under "
                                                       "(<DIR>/CURRENT + <DIR>/snapshots/<id>/); never the registry root")
    s.add_argument("--source-root", metavar="ROOT", help="paper_frame only: the registry root to read (default: --root)")
    s.add_argument("--model-id", action="append", default=[], metavar="ID",
                   help="paper_frame only: also export this model's runs (repeatable; the paper's four, their fit checkpoints and the "
                        "pipeline-test models are always in)")
    s.add_argument("--precondition-override", metavar="REASON",
                   help="paper_frame only: skip the tree precondition (relu_native's PAIRED_ROOT is a real directory and the commit "
                        "that moved it is an ancestor of HEAD); the reason is recorded in MANIFEST.precondition")
    s.set_defaults(fn=cmd_snapshot)
    s = sub.add_parser("docs", help="regenerate schema.sql and docs/registry/{schema,vocabulary,records}.md")
    s.add_argument("which", choices=["schema", "vocabulary", "records", "all"])
    s.set_defaults(fn=cmd_docs)
    s = sub.add_parser("identity", help="resolve a producer key to the run ids behind it")
    s.add_argument("action", choices=["resolve"])
    s.add_argument("producer_key")
    s.add_argument("--campaign")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_identity)
    s = sub.add_parser("blocked", help="dependency-blocked records")
    s.add_argument("action", choices=["list"])
    s.add_argument("--stale", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_blocked)
    s = sub.add_parser("reconcile", help="record-only reconciliations from what the log already holds (never a database edit): "
                       "`run-patch-extra` re-files, per run, the keys a legacy run_patch carried under a top-level `extra` (dropped at "
                       "the fold) that the live run.extra_json still lacks -- missing keys re-filed, equal skipped, a "
                       "conflict (the live value differs) reported for adjudication and never overwritten; a run_overrides stamp that "
                       "differs from the live one is equal when both are readings of one overrides file among the --source files (the old "
                       "reader's or the fixed one's), a conflict otherwise; only a record the fold "
                       "accepted is evidence (a refused, blocked or not-yet-ingested one is reported, never re-filed). Dry run by "
                       "default; --apply is gated in this release (exit 2, the conflicts listed); leaves no state/last_run stamp. "
                       "`multi-turn-scope` corrects the runs the live database shows on a multi-turn, Petri or agentic eval with a steering "
                       "object whose multi_turn_steering_scope differs from the value derived from its applied-positions mode, the eval's "
                       "category and the object's kind (the job-start default single_turn the common case; a system-instruction object "
                       "steers no positions: not_applicable): the dry run counts them by (model, category) and by object kind; --apply "
                       "writes one run_patch per run with the derived scope into the run's own record directory (exit 2 when a run is held "
                       "behind an unfolded record stating its mode or scope); --verify lists the inconsistent runs over the folded "
                       "database, read-only (exit 3 when there are any; the landing window reconciles once more and verifies again). "
                       "`env-comments` restates, per run row whose extra_json.run_overrides stamp differs from the reading of its own "
                       "overrides file (the files named by --source, matched to a stamp by the file's sha256; the repository's "
                       "reproduce/overrides is always read), the whole stamp from the file -- the values as the shell reads them, a literal "
                       "`#` inside quotes kept -- as one run_patch into the run's own record directory with a marker naming the replaced "
                       "values and the file; the dry run counts the runs by overrides file and by key, lists the stamps whose file was not "
                       "given and whose values may carry a comment (ambiguous, never corrected) and the other tables' rows (full statements, "
                       "listed, not corrected); --apply exit 2 when a run is held behind an unfolded record stating its stamp, its stamp is "
                       "no reading of its file (unclassified, refused by name: a key set that is not the file's, or a value that is neither "
                       "the file's reading nor what the old reader made of its line -- a later statement, never restated), or no file among "
                       "the sources has its stamp's sha256 while its values may carry a comment (unsourced: refused by name, since a pass that "
                       "completed past every stamp it could not decide would correct nothing and exit 0); --verify lists the run rows whose "
                       "stamp still differs from the file over the folded database, the unsourced rows among them (exit 3 when there are any); "
                       "--leave-unsourced is the operator's word to leave the unsourced stamps as stored (listed, not a failure); "
                       "--check-sources is the coverage alone (exit 3 when a stamped run row whose values may carry a comment has no file "
                       "among the sources), which the landing window runs before anything moves. "
                       "`fit-effective-batch` restates, per object row whose fit_effective_batch is the micro-batch a sidecar hook read "
                       "(resolved_batch_size) where the row's own fit_hyperparameters_json says the optimizer width (batch_size x accum_steps = "
                       "effective_batch), that recorded value as one same-id object record into the registry owner's record directory, with the "
                       "column's confidence entry and a marker naming the replaced value; the dry run groups the rows by kind, role and carrier "
                       "and lists the ambiguous rows (fit_effective_batch NULL beside a recorded value, or a dose copy's NULL stated by the adapter's own entry "
                       "with no recorded value at all: a dose copy the fit-sidecars adapter has not restated as its parent's batch yet, or left NULL by "
                       "name; every row by object id with the adapter's reason, under its group) "
                       "and any unclassified disagreement, neither corrected; reads every dose copy carrying the adapter's statement against its OWN "
                       "row in Python, in its own type (the statement check, read-only: a derived value that is not the block's own integer -- the "
                       "block recording 8.0 beside a derived 8, which SQL reads as equal, or no effective_batch at all -- or a value standing beside "
                       "the NULL-by-name entry, is listed by object id and never written for); and cross-checks every derived dose copy against its PARENT object "
                       "row's recorded effective batch (read-only: a disagreement is listed by object id and never written for, a parent value that "
                       "is not an integer in its own type -- 8.0, true, \"8\" -- among them, named with its type, nothing coerced; a parent row absent "
                       "from the registry is not a disagreement, said so by name); --apply exit 2 when a row is held behind an unfolded object record "
                       "or refused as unclassified; --verify lists the disagreeing rows, the statement inconsistencies and the parent disagreements over "
                       "the folded database (exit 3 when there are any)")
    s.add_argument("action", choices=["run-patch-extra", "multi-turn-scope", "env-comments", "fit-effective-batch"])
    g = s.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="print the three buckets per run and the totals; write nothing (the default)")
    g.add_argument("--apply", action="store_true", help="write ONE run_patch per run carrying only its missing keys into the run's own record "
                                                           "directory (gated: refuses with exit 2 until the landing window lifts it on the owner's word)")
    g.add_argument("--verify", action="store_true", help="the correction actions only: the action's predicate alone over the folded database "
                                                            "(multi-turn-scope: every run of the class whose (applied_positions_mode, "
                                                            "multi_turn_steering_scope) disagrees with the derivation; env-comments: every run row "
                                                            "whose run_overrides stamp still differs from its overrides file's reading, the same "
                                                            "--source files; fit-effective-batch: every object row "
                                                            "whose fit_effective_batch disagrees with its own recorded effective_batch); exit 0 when "
                                                            "none, 3 when some (read-only, no hold, nothing written)")
    g.add_argument("--check-sources", action="store_true", help="env-comments only: the coverage of the --source files alone over the folded run "
                                                                   "table -- every stamped run row whose values may carry a comment has its overrides "
                                                                   "file among them (exit 0), or the uncovered stamps are listed by overrides id and "
                                                                   "file sha with their keys (exit 3); read-only, no walk, nothing written -- what the "
                                                                   "landing window runs before anything moves when env-comments is listed for apply")
    s.add_argument("--leave-unsourced", action="store_true",
                   help="env-comments only: leave the stamps no --source file covers as stored, by name -- they stay listed (ambiguous) but "
                        "--apply, --verify and --check-sources complete past them; without it an unsourced stamp whose values may carry a comment "
                        "is refused by the apply (exit 2) and listed by the verify (exit 3), so a source list that covers nothing corrects "
                        "nothing loudly rather than reading a clean class")
    s.add_argument("--all", action="store_true", help="list the runs whose lost keys are all equal too (the default lists only runs with something to act on)")
    s.add_argument("--source", action="append", default=[], metavar="PATH",
                   help="env-comments and run-patch-extra (repeatable): an overrides file, or a directory of .env files, offered as the "
                        "evidence a stamp is restated from (env-comments) or compared as a reading of (run-patch-extra) -- matched to a stamp "
                        "by the file's sha256, so a file that is not a stamp's original matches nothing; the repository's reproduce/overrides "
                        "directory is read without being named; a path that is not there is refused")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_reconcile)
    s = sub.add_parser("pending-judgment", help="the judging work queue: runs with no complete judgment by the requested judge "
                                                "(default: each run's instrument judge), with the reason, coverage and any live claim")
    s.add_argument("--judge", help="a judge_id or its seeded readable id (configs/registry/judges.yaml)")
    s.add_argument("--campaign")
    s.add_argument("--claim", action="store_true", help="register an in-progress judgment (needs --run-id and --judge)")
    s.add_argument("--release", metavar="RUN_ID", help="abandon one's own claim (needs --judge)")
    s.add_argument("--session", help="the claiming session (default $SISTEER_SESSION_NAME, else $USER@host)")
    s.add_argument("--run-id")
    s.add_argument("--force", action="store_true", help="take over a stale claim")
    s.add_argument("--stale-hours", type=float, default=12.0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_pending)
    s = sub.add_parser("judgment", help="one judgment by id: `show` its live row (status, successor, stamps, the identity form that mints "
                       "its id, the run's pending state for its judge); `supersede` retires a complete or partial judgment as a RECORD in the "
                       "log (records/supersedes/<run_dir>/, never a database edit): snapshot-checked, minimal, identity-bound")
    s.add_argument("action", choices=["show", "supersede"])
    s.add_argument("judgment_id")
    s.add_argument("--superseded-by", metavar="JUDGMENT_ID", help="supersede: the successor, a complete or partial judgment of the same run that has folded")
    s.add_argument("--no-successor", action="store_true", help="supersede: a null link (a re-judge is wanted; the readers queue the run again)")
    s.add_argument("--reason", help="supersede: why the judgment is retired (free text, kept on the record; required)")
    s.add_argument("--ruled-by", help="supersede: who ruled it (default: the user, or the user via the relaying session)")
    s.add_argument("--provenance-kind", choices=["user_direct", "relayed"], default="user_direct",
                   help="supersede: the user ruled it in this session, or a peer session relayed the user's words (then --relayed-by and "
                        "--message-utc are required)")
    s.add_argument("--relayed-by", help="supersede: the relaying session (provenance relayed)")
    s.add_argument("--message-utc", help="supersede: the relayed message's time, date or ISO-8601 UTC (provenance relayed)")
    s.add_argument("--allow-cross-run", action="store_true", help="supersede: admit a successor that judges ANOTHER run (refused otherwise, at "
                   "the CLI and at the fold); the record carries extra.allow_cross_run")
    s.add_argument("--session", help="supersede: the producer session stamped on the record (default $SISTEER_SESSION_NAME, else $USER@host)")
    s.add_argument("--dry-run", action="store_true", help="supersede: print the assembled record and the consequences; write nothing")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_judgment)
    s = sub.add_parser("refusals", help="the refusal FACTS read back (read-only): `list` every refusal judgment ever made (one row per refused "
                       "record and judging stamp, beside the record's current judgment; --still-refused keeps the current ones), `explain` one "
                       "record by sha or by the uri it was refused under (its facts per stamp, the refused bytes re-read by sha, the current "
                       "validator's verdict, and where the record's key stands now: a later row folded, an earlier row stands, no row, unknown), "
                       "`summary` the counts; writes nothing, not even a state/last_run stamp")
    s.add_argument("action", choices=["list", "explain", "summary"])
    s.add_argument("ident", nargs="?", help="explain: the record sha (64 hex) or the record_uri it was refused under (`<file>` or `<file>#<line>`)")
    s.add_argument("--still-refused", action="store_true", help="list: only the facts that are the record's CURRENT judgment")
    s.add_argument("--type", help="list: one record kind only")
    s.add_argument("--limit", type=int, default=200, help="list: the newest N facts (default 200)")
    s.add_argument("--all", action="store_true", help="list: every fact")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_refusals)
    s = sub.add_parser("observations", help="the harvester's fold OBSERVATIONS read back (read-only): what it saw about a record at the fold "
                       "without refusing it -- today one kind, `serving_jobs_strict_subset`: a run or run_patch record whose extra.serving_jobs "
                       "names a strict subset of the set the run's extra_json already held (the fold replaced the key as it always does; the "
                       "folded set, the restated set and the dropped members are recorded). `summary` the counts by kind and the runs with the "
                       "most; `list` the observations, newest restatement first, each with the arrival of its record's file (the cycle that folded "
                       "it); writes nothing, not even a state/last_run stamp. Not the `observation` table (a source revision's membership of a key)")
    s.add_argument("action", choices=["summary", "list"])
    s.add_argument("--run", help="list: one run_id")
    s.add_argument("--kind", help="list: one observation kind")
    s.add_argument("--limit", type=int, default=200, help="list: the newest N observations (default 200)")
    s.add_argument("--all", action="store_true", help="list: every observation")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_observations)
    s = sub.add_parser("cap-check", help="the token-cap ladder facts of the live runs, read-only: the record rule re-applied to every run "
                       "that states its ladder index (a disagreement exits 1: the live table holds a row a replay would refuse; 0 live), the "
                       "legacy shapes the validator admits (a ladder stated with no index; an empty ladder with no index, split by the "
                       "instrument's ladder: one level, several, none) and the instrument cross-check (a cap outside the instrument's ladder, on "
                       "every instrument with a ladder; a run ladder unlike the instrument's); writes nothing, not even a state/last_run stamp")
    s.add_argument("--campaign", help="one campaign only (default: every campaign)")
    s.add_argument("--all", action="store_true", help="list every run of each bucket (default: the first 20; counts are always whole)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_cap_check)
    s = sub.add_parser("worklist-parity", help="compare the pending set for one judge to a judging driver's unit list "
                                               "(units_*.json: store, object, eval) set-wise; differences by run_id with reasons")
    s.add_argument("--judge", required=True, help="a judge_id or its seeded readable id")
    s.add_argument("--units", action="append", required=True, metavar="UNITS_JSON", help="repeatable")
    s.add_argument("--campaign")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_worklist_parity)
    s = sub.add_parser("adapters", help="the incremental adapter pass over registered source adapters, then ingest what it published "
                                        "(nothing published, or --no-ingest: no fold, harvester.lock never taken)")
    s.add_argument("action", choices=["run"])
    s.add_argument("--incremental", action="store_true", default=True)
    s.add_argument("--full", action="store_true", help="read every group again and re-hash every input's bytes, bypassing the stat-identity "
                   "hash cache (state/input_hash_cache.json) -- the answer to an in-place edit that kept size, mtime, inode and ctime")
    s.add_argument("--no-ingest", action="store_true", help="publish records only and never take harvester.lock: the harvester's own tick "
                   "folds them (the adapters timer's mode; a landing window runs one ingest after every adapter)")
    s.add_argument("--retry-refused", action="store_true", help="read again every group whose last read REFUSED although nothing about it "
                   "changed (a reader fix landed): without it a standing refusal is left alone until an input changes, the adapter's "
                   "EMISSION_VERSION changes, or --full; the landing window passes it once, the quiet timer never")
    s.add_argument("--profile", metavar="FILE", help="cProfile the pass (run_pass only, not the fold) and write the stats to FILE in pstats "
                   "format; read with `python -m pstats FILE` -- states where a pass's time goes instead of guessing")
    s.add_argument("--reread", action="store_true", help="read every committed source again and publish its status facts as an "
                   "activation bundle (a current head whose keys the read reproduces has no measurement re-emitted). An adapter "
                   "whose EMISSION_VERSION was bumped publishes a NEW revision instead -- as the incremental pass also does")
    s.add_argument("--force", action="store_true", help="with --reread: publish every read as a NEW revision whatever its inputs; "
                   "the pass stamp is folded into each revision id as a nonce and recorded in the marker's extra_json.forced. The "
                   "operational path when an adapter's code changed what it parks and nobody bumped its EMISSION_VERSION")
    s.add_argument("--adapter", nargs="*")
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seeds", help="seed directory the adapters' legacy spellings (model_ids, si_texts) are built from (default configs/registry)")
    s.add_argument("--reader-state", metavar="FILE", help="CLUSTER mode: compare against this reader-state sidecar (registry reader-state export) "
                   "instead of the live database; implies --no-ingest; a group whose minted revision id the grading host already knows is refused by "
                   "name (reactivation is the grading host's); --reread is refused")
    s.add_argument("--shard", metavar="i/K", help="read only the comparison groups shard i of K owns: the sorted union of the present and the "
                   "prior-only groups, by canonical identity, dealt round-robin (a reader_traversal adapter admits K = 1 only)")
    s.add_argument("--identity-map", action="append", metavar="LOCAL=CANONICAL", help="spell identities under LOCAL as CANONICAL (repeatable; "
                   "SISTEER_REGISTRY_URI_HOME in the environment maps this home onto the named one)")
    s.add_argument("--pins", metavar="FILE", help="CLUSTER mode: the pinned generations ({gen_dir, pins: {slug: sha8}}); every input under one is "
                   "hashed and checked against its verified manifest (a mismatch refuses the adapter by name)")
    s.add_argument("--read-copy", metavar="FILE", help="CLUSTER mode: the JSON recorded on every marker as extra.read_copy (which copy of which "
                   "component was read, under which attestation)")
    s.add_argument("--substrate", choices=["cluster_a", "cluster_b", "grading_host"], help="CLUSTER mode: the producer substrate stamped on every record "
                   "(default: detected from the shared root that exists)")
    s.add_argument("--summary-out", metavar="FILE", help="write the shard summary JSON (groups seen / changed / published / held / refused / "
                   "refused-known / not owned, the revision directories published with their files, wall, host)")
    s.set_defaults(fn=cmd_adapters)
    s = sub.add_parser("reader-state", help="the reader-state sidecar a cluster shard compares against: one SQLite with the named adapters' "
                       "source_revision / source_watermark / source_state / source_status / current observation rows, the source_revision "
                       "marker queue, the accepted heads' item membership keys, the dimension tables, the anchor inputs, and the revision ids "
                       "published on this grading host's disk (read-only on the live database)")
    s.add_argument("action", choices=["export", "show"])
    s.add_argument("--adapter", nargs="*", help="export: the adapters to export (at least one)")
    s.add_argument("--out", metavar="FILE", help="export: where to write the sidecar (replaced atomically)")
    s.add_argument("--uri-home", help="export: the canonical home spelling recorded for the cluster task (default: this home)")
    s.add_argument("--repo-root", help="export: the canonical checkout spelling recorded for the cluster task (default: this checkout)")
    s.add_argument("--file", metavar="FILE", help="show: the sidecar to describe")
    s.set_defaults(fn=cmd_reader_state)
    s = sub.add_parser("cluster-reads", help="the Slurm-sharded reads of the heavy adapters: plan a window "
                       "(admission of every pinned generation on the grading host), run one shard on the cluster, re-validate after the array, pull "
                       "the published revisions marker-last with the failure barrier, and name the adapters the grading host reads itself")
    from . import cluster_reads as CRM
    CRM.add_arguments(s)
    s.set_defaults(fn=lambda a: CRM.cmd_cluster_reads(a, _root(a)))
    s = sub.add_parser("decode-policy", help="export the decode_policy authority to configs/registry/decode_policy.json (hash-pinned, "
                       "stamped with the registry ingest_seq), or check that the committed file equals the registry")
    s.add_argument("action", choices=["export", "check"])
    s.add_argument("--out", help="one export path (default: BOTH committed copies, configs/registry/decode_policy.json and the "
                       "package copy src/sisteer/registry/decode_policy.json, written and checked together)")
    s.set_defaults(fn=cmd_decode_policy)
    s = sub.add_parser("dimensions", help="load the dimension seeds (configs/registry/*.yaml: model, eval, item_set, judge, instrument, "
                       "si_text, decode_policy, sampling_mode_policy, plus the `none` and prompted `si:<sha>` object rows) as records, then ingest")
    s.add_argument("action", choices=["load"])
    s.add_argument("--seeds", help="seed directory (default configs/registry)")
    s.add_argument("--no-ingest", action="store_true", help="write the records only")
    s.add_argument("--no-verify-si", action="store_true", help="skip recomputing the SI text shas from sisteer.data.instructions")
    s.add_argument("--force", action="store_true", help="rewrite a seed revision already on the record log (its records are appended to the "
                       "revision's files under the harvester lock)")
    s.add_argument("--workers", type=int, default=0)
    s.add_argument("--lock-timeout", type=float, default=30.0, help="seconds to wait for the harvester lock before refusing (default 30)")
    s.add_argument("--budget-s", type=float, default=None, metavar="SECONDS",
                   help="wall-clock budget for the harvester cycle the load runs under the lock (as `ingest --budget-s`): the seed "
                        "revision is applied whole regardless; the cycle's other pending files and revision directories are taken "
                        "only while the budget lasts, and `ingest.stopped` says `budget` when it ran out (default: none)")
    s.add_argument("--retry-held", action="store_true",
                   help="re-attempt every held revision directory in the load's cycle whatever its attempt identity; an UNCAPPED load "
                        "(no --budget-s, the landing window's) does so regardless")
    s.set_defaults(fn=cmd_dimensions)
    s = sub.add_parser("publish-figures", help="register every figure file in a directory as an artifact (published implies recorded); "
                                               "optionally publish only registered files and refuse any file without a record")
    s.add_argument("--dir", required=True)
    s.add_argument("--notebook-commit")
    s.add_argument("--checkout", help="refuse files untracked in this git checkout")
    s.add_argument("--publish-to", help="copy registered, sha-verified files here; refuse unregistered files in either directory")
    s.add_argument("--verify-dir", help="refuse when this (already published) directory holds a file with no artifact row")
    s.add_argument("--verify-only", action="store_true", help="register nothing; only run the --publish-to / --verify-dir checks")
    s.set_defaults(fn=cmd_publish_figures)
    s = sub.add_parser("register-run", help="submit-time registration: validate the planned run strictly, write the run + "
                       "queued state records (+ the scheduler job binding), store the plan beside the job, push")
    s.add_argument("plan", help="planned-run JSON (the resolved identity fields; see docs/registry/records.md)")
    s.add_argument("--spool", help="record spool (default: the resolved producer spool)")
    s.add_argument("--job-id", help="scheduler job id (Slurm) to bind to the run")
    s.add_argument("--job-name")
    s.add_argument("--sha-pin", help="the submit-time code sha pin (--comment=sha:<sha>)")
    s.add_argument("--pending-dir", help="store the plan (+ run_id) as <pending-dir>/<job-id>.json for the job to find")
    s.add_argument("--check-only", action="store_true", help="validate only; write nothing (exit 2 on refusal)")
    s.add_argument("--no-push", action="store_true")
    s.set_defaults(fn=cmd_register_run)
    s = sub.add_parser("inbox-merge", help="move pushed record files from inbox/<host>/ into records/ (the harvester's merge step, alone)")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_inbox_merge)
    s = sub.add_parser("objects", help="the object banks on this machine: per bank, objects found, sidecars, weights files, refusals, unverified fields")
    s.add_argument("action", choices=["scan"])
    s.add_argument("--bank-root", nargs="*", metavar="LABEL=PATH:LAYOUT",
                   help="replace the default banks (layout: full22 | model/obj | bank/model/obj | stage)")
    s.add_argument("--seeds", help="directory holding legacy_spellings.yaml, si_texts.yaml, models.yaml (read-only; default configs/registry when present)")
    s.add_argument("--pins", help="the full22 pins JSON that names the cluster copies (default configs/sdfbase_full22_pins.json when present)")
    s.add_argument("--no-blobs", action="store_true", help="do not open weights files (blob-only fields become unverified)")
    s.add_argument("--no-hpo", action="store_true", help="skip the hyperparameter-search logs")
    s.add_argument("--wandb", action="store_true", help="consult the W&B API for the run ids the sidecars name (read-only)")
    s.add_argument("--emit", action="store_true", help="also write the records through the incremental pass into --root")
    s.add_argument("--full", action="store_true", help="with --emit: refresh every group, not only changed inputs")
    s.add_argument("--payloads", action="store_true", help="with --json: include every object payload")
    s.add_argument("--allow-refusals", action="store_true", help="exit 0 even when some objects were refused")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_objects)

    # stubs with no backend in this tree
    s = sub.add_parser("backfill", help="run every adapter into a fresh database")
    s.add_argument("--adapter", nargs="*")
    s.add_argument("--into")
    s.add_argument("--resume", action="store_true")
    s.set_defaults(fn=lambda a: _stub("backfill", "registry backfill [--adapter NAME ...] [--into PATH] [--resume]: adapters -> records -> "
                                      "fresh database with indexes deferred; refuses below 2x the projected write (backfill.backfill)"))
    s = sub.add_parser("roster", help="the paper roster of a round: load it into the record log, show it, diff it, or transcribe it from a pinned paper checkout")
    s.add_argument("action", choices=["load", "show", "diff", "transcribe"])
    s.add_argument("--round", default="r2")
    s.add_argument("--version", default="v1")
    s.add_argument("--roster", help="roster yaml (default configs/registry/roster_<round>_<version>.yaml)")
    s.add_argument("--rulings", help="rulings jsonl (default configs/registry/rulings_<round>_<version>.jsonl)")
    s.add_argument("--against", help="diff: another roster yaml (default: the database's rows)")
    s.add_argument("--against-rulings")
    s.add_argument("--paper-checkout", help="transcribe: the pinned paper checkout (read-only)")
    s.add_argument("--sha", help="transcribe: the commit the checkout must be at")
    s.add_argument("--paper-python", help="transcribe: the interpreter that imports the pinned paperlib (default: this one)")
    s.add_argument("--previous-roster", help="transcribe: the previous version's roster (its pins and disagreement prose are kept)")
    s.add_argument("--previous-rulings", help="transcribe: the previous version's rulings (provenance kept; written rulings carried forward)")
    s.add_argument("--out", help="transcribe: output directory")
    s.add_argument("--no-count-check", action="store_true", help="do not refuse when the expansion differs from meta.expanded_counts_by_comparison")
    s.add_argument("--no-ingest", action="store_true", help="load: write the records only")
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--limit", type=int, default=40)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_roster)
    s = sub.add_parser("ruling", help="add a ruling as a record under <root>/rulings/ (never the database), list rulings, or dry-run their expiry evaluation")
    s.add_argument("action", choices=["add", "list", "expire-check"])
    s.add_argument("--round")
    s.add_argument("--action", dest="ruling_action", help="include | exclude | supersede | coexist | operating_point | descope | drop_eval | reference | hold_out")
    s.add_argument("--scope-kind")
    s.add_argument("--scope-json", help="JSON; `match` lists (model_ids, paper_labels, eval_ids, ...) are what the flag matches on")
    s.add_argument("--target-json")
    s.add_argument("--expires-when-json")
    s.add_argument("--reason")
    s.add_argument("--source-ref")
    s.add_argument("--ruled-utc", help="date or ISO-8601 UTC (default now)")
    s.add_argument("--ruled-by")
    s.add_argument("--provenance-kind", choices=["user_direct", "relayed"], default="user_direct",
                   help="who the ruling came from: the user in this session, or a peer session relaying the user's words "
                        "(then --relayed-by and --message-utc are required; the ruling is shown for the user's confirmation)")
    s.add_argument("--relayed-by", help="the relaying session (provenance relayed)")
    s.add_argument("--message-utc", help="the relayed message's time, date or ISO-8601 UTC (provenance relayed)")
    s.add_argument("--id", help="ruling_id (default <round>-<action>-<stamp>)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_ruling)
    s = sub.add_parser("analysis", help="a prefill-only or offline ANALYSIS over ingested runs and objects: `file --spec <json>` checks the "
                       "spec's shape, READ-checks that every input run and object is in this registry (a missing one refuses by name), "
                       "registers the report and the page directories' files as artifacts, and writes the analysis, its inputs and its "
                       "artifact links as records under records/analyses/<day>/ (the harvester folds them; the database is never written "
                       "here); `--dry-run` prints the plan and writes nothing; `list` reads the filed analyses back (read-only)")
    s.add_argument("action", choices=["file", "list"])
    s.add_argument("--spec", help="file: the analysis spec (JSON; sisteer.registry.analysis names its fields)")
    s.add_argument("--dry-run", action="store_true", help="file: print the plan (the id, the inputs and their presence, the files to register) and write nothing")
    s.add_argument("--name", help="list: one analysis name only")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_analysis)
    s = sub.add_parser("revision", help="adapter revision markers the harvester HOLDS (published, not activated): list them with their kind and "
                       "reason, or ACCEPT one held for dropping run ids of the accepted head without a declaration by publishing a "
                       "declaration bundle -- an append-only record the next harvester cycle folds, never a database write")
    s.add_argument("action", choices=["held", "accept"])
    s.add_argument("--revision-id", help="accept: the held revision")
    s.add_argument("--reason", help="accept: why the drop is legitimate; recorded on the marker under extra_json.declared_run_drops")
    s.add_argument("--run-id", action="append", help="accept: declare only these dropped run ids (repeatable; default every dropped id, '*')")
    s.add_argument("--accepted-by", help="accept: who decided (default 'the user')")
    s.add_argument("--force", action="store_true", help="accept: release a held revision a later accepted revision has since superseded (it becomes the head)")
    s.add_argument("--all", action="store_true", help="held: include the stale holds (superseded by a later accepted revision, or older than seven days)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_revision)
    s = sub.add_parser("check-paper-parity", help="the parity gate against a pinned paper checkout: raw rows vs the readers (L0), assembled vs assembled (L1), "
                       "the selected keys (L1b), watermarks (L1c), figure data (L2); pins the input set and refuses when it moves")
    s.add_argument("--paper-checkout", required=True)
    s.add_argument("--pin", help="the commit the checkout must be at (default: its HEAD, recorded)")
    s.add_argument("--round")
    s.add_argument("--level", default="all", help="all or a comma list of L0,L1,L1b,L1c,L2")
    s.add_argument("--paper-python", help="the paper venv's interpreter that runs the checkout's own readers (default ~/venvs/sdfbase-r2-nb/bin/python)")
    s.add_argument("--report", help="report directory (default <root>/state/parity/<gate_id>/)")
    s.add_argument("--work-dir")
    s.set_defaults(fn=cmd_check_paper_parity)
    s = sub.add_parser("paper-objects", help="object-level paper membership of the open round (in_paper, comparisons, excluding rulings)")
    s.add_argument("--round")
    s.add_argument("--limit", type=int, default=200)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_paper_objects)
    s = sub.add_parser("export", help="compatibility exports: the deprecated round-2 parquet rendered from the registry (r2compat.py)")
    s.add_argument("what", choices=["r2-compat"])
    s.add_argument("out")
    s.add_argument("--compare", help="an existing export parquet to diff against, per join key and column")
    s.set_defaults(fn=lambda a: _lazy_module("r2compat", "cli_export", a))
    s = sub.add_parser("degeneracy-rebuild", help="regenerate the 2026-09-14 degeneration tables from v_run_degeneration_wide and diff them against the CSV oracles (degeneracy_artifact.py)")
    s.add_argument("--out", required=True)
    s.add_argument("--csv-dir", help="directory of the degen_*_over1pct.csv oracles (default ~/figs/degeneracy/2026-09-14)")
    s.set_defaults(fn=lambda a: _lazy_module("degeneracy_artifact", "cli_rebuild", a))
    return p


def stamp_last_run(root: str, cmd: str, rc: int, seconds: float, argv: Sequence[str]) -> None:
    """``state/last_run/<cmd>.json`` — when the command last ran, its exit code and duration (the status
    page's timer-health section). Never fails the command. Written only for a command that resolved a root
    (``_root``): ``docs`` or ``register-run --spool …`` touch no registry, so a test or a shell that never named
    one must not leave a stamp in the operator's ``~/registry`` (the live install found `docs` and a pytest's
    `register-run` there, 2026-09-17)."""
    try:
        if not os.path.isdir(root):
            return
        d = os.path.join(root, "state", "last_run")
        os.makedirs(d, exist_ok=True)
        payload = {"command": cmd, "utc": R.utc_now(), "rc": int(rc), "seconds": round(seconds, 3), "argv": list(argv)}
        tmp = os.path.join(d, f"{cmd}.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, os.path.join(d, f"{cmd}.json"))
    except Exception:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    import time
    from .adapters.base import install_store_audit_from_env
    install_store_audit_from_env()      # REGISTRY_STORE_AUDIT=<log>: every store access under the named roots recorded (inert otherwise)
    p = build_parser()
    args = p.parse_args(argv)
    t0 = time.monotonic()
    rc = 1
    try:
        rc = int(args.fn(args) or 0)
        return rc
    except ConfigError as exc:
        # the loader refused the configuration by name (the removed Modal table or destination): the
        # command did not run -- fail-closed, reported on stderr, exit 1, never a traceback and never a silent default
        print(f"registry {args.cmd}: configuration REFUSED: {exc}", file=sys.stderr)
        rc = 1
        return rc
    finally:
        name = str(args.cmd) + (f"-{args.action}" if getattr(args, "action", None) else "")
        if getattr(args, "_root_resolved", False):
            stamp_last_run(_root(args), name, rc, time.monotonic() - t0, list(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
