"""The database is optional: these run with or without it.

The central Postgres holds app state only — a human review, a submission record —
so every test here either needs no server at all, or skips when there is none.
"""

from __future__ import annotations

import pytest

from tau2_loop.config import settings
from tau2_loop.data import pg


def test_the_urls_come_from_settings_and_point_at_the_central_server() -> None:
    s = settings()
    for url in (s.database_url, s.ro_database_url, s.pg_superuser_url):
        assert url.startswith("postgresql://") and url.endswith("/tau2")
    # never a local file store, never another port (PLATFORM.md rule 1 and its db rule 7)
    assert "sqlite" not in s.database_url and "5432" in s.database_url


def test_an_unreachable_database_is_a_state_not_an_error() -> None:
    # the viewer asks this before offering a write route; it must never raise
    assert pg.reachable("postgresql://nobody:nobody@127.0.0.1:1/none") is False


@pytest.mark.skipif(not pg.reachable(), reason="central Postgres not running")
def test_migrate_leaves_every_table_owned_by_tau2_owner_and_visible_to_it() -> None:
    # a table owned by the platform superuser is invisible to tau2_owner: migrate()
    # must reclaim ownership before creating anything, or this regresses silently
    pg.migrate()
    with pg.connect() as con:
        owners = con.execute(
            "select tablename, tableowner from pg_tables where schemaname = %s",
            (pg.SCHEMA,),
        ).fetchall()
        assert {name for name, _ in owners} == {"review", "submission"}
        assert all(owner == "tau2_owner" for _, owner in owners)

        visible = {
            r[0]
            for r in con.execute(
                "select table_name from information_schema.tables where table_schema = %s",
                (pg.SCHEMA,),
            ).fetchall()
        }
        assert visible == {"review", "submission"}

    # idempotent: a second migrate() used to fail with "must be owner of table
    # review" on CREATE INDEX IF NOT EXISTS once the superuser owned the tables
    pg.migrate()


@pytest.mark.skipif(not pg.reachable(), reason="central Postgres not running")
def test_both_tables_exist_and_the_read_only_role_cannot_write() -> None:
    import psycopg

    assert {name for name, _ in pg.tables()} == {"review", "submission"}
    with pg.connect_ro() as con:
        con.execute("select count(*) from tau2_loop.review")
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            con.execute(
                "insert into tau2_loop.review (run_id, task_id, trial, judge_reward, verdict) "
                "values ('x', 'y', 1, 0, 'agree')"
            )
