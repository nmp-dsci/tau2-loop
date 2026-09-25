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


def test_the_schema_file_creates_its_tables_as_the_owner() -> None:
    sql = pg.ROLES_SQL.read_text()
    # a table owned by the superuser is invisible to tau2_owner: the DDL must act as it
    assert "SET ROLE tau2_owner;" in sql
    assert sql.index("SET ROLE tau2_owner;") < sql.index(
        "CREATE TABLE IF NOT EXISTS tau2_loop.review"
    )
    assert "RESET ROLE;" in sql
    # and the read-only role must never be able to write
    assert "default_transaction_read_only = on" in sql


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
