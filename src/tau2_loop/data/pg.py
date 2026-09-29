"""Postgres: the connections, the schema, and what is in it.

The database is `tau2` on the central Postgres (nmp-central-ai, PLATFORM.md D13);
`infra/roles.sql` is applied once as the platform's superuser by `make db-migrate`,
and everything else connects as `tau2_owner` (writes) or `tau2_ro` (the served
viewer). Nothing here reads `os.environ`: the URLs come from `config.settings()`.

**The database is optional.** It holds app state only — a human review of a scored
conversation, a record of a prepared submission — and never a benchmark fact. Every
number the viewer shows is read from a committed file, so `reachable()` is what the
API asks before offering a write route, and the whole app boots with Postgres
stopped (s04 R-3, decision Q2-A).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from tau2_loop.config import ROOT, settings

if TYPE_CHECKING:
    import psycopg

SCHEMA = "tau2_loop"
ROLES_SQL = ROOT / "infra" / "roles.sql"


def connect(url: str | None = None, autocommit: bool = True) -> psycopg.Connection[Any]:
    """As `tau2_owner` unless told otherwise."""
    import psycopg

    return psycopg.connect(url or settings().database_url, autocommit=autocommit)


def connect_ro() -> psycopg.Connection[Any]:
    """As `tau2_ro`: SELECT only, 15 s statement timeout (infra/roles.sql)."""
    import psycopg

    return psycopg.connect(settings().ro_database_url, autocommit=True)


def migrate(path: Path = ROLES_SQL) -> None:
    """Apply `infra/roles.sql` as the platform superuser. Idempotent."""
    import psycopg

    with psycopg.connect(settings().pg_superuser_url, autocommit=True) as con:
        con.execute(path.read_text())  # type: ignore[arg-type,unused-ignore]


def reachable(url: str | None = None) -> bool:
    """Can we talk to it at all? A `False` is a legitimate state, never an error."""
    try:
        with connect(url) as con:
            con.execute("select 1")
        return True
    except Exception:  # noqa: BLE001 - the caller prints the one-line remedy
        return False


def tables() -> list[tuple[str, int]]:
    """(table, rows) in our schema — what `make db-smoke` prints."""
    with connect() as con:
        names = [
            r[0]
            for r in con.execute(
                "select table_name from information_schema.tables "
                "where table_schema = %s and table_type = 'BASE TABLE' order by table_name",
                (SCHEMA,),
            ).fetchall()
        ]
        out = []
        for name in names:
            row = con.execute(f'select count(*) from {SCHEMA}."{name}"').fetchone()
            out.append((name, int(row[0]) if row else 0))
        return out
