-- tau2-loop: one schema, two roles, on the central Postgres (database `tau2`).
-- Applied as the cluster superuser by `make db-migrate`; idempotent.
--
--   tau2_owner  the migration and the write routes: owns the schema.
--   tau2_ro     what the served viewer reads as: SELECT only, 15 s statement
--               timeout, read-only transactions, no privilege anywhere else.
--
-- Nothing here is a benchmark fact. Every number the viewer shows is read from a
-- committed file under runs/, agents/, loop/ or data/; this schema holds only what
-- a person typed after a run, which cannot be a committed file without a commit.
-- The app must boot and serve with this database stopped (s04 R-3).
--
-- Passwords are the platform's local-only defaults (DATABASE_URL / RO_DATABASE_URL
-- override them; `make -C ../nmp-central-ai db-init` creates the roles).

CREATE SCHEMA IF NOT EXISTS tau2_loop AUTHORIZATION tau2_owner;

GRANT USAGE ON SCHEMA tau2_loop TO tau2_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA tau2_loop TO tau2_ro;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA tau2_loop TO tau2_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE tau2_owner IN SCHEMA tau2_loop
  GRANT SELECT ON TABLES TO tau2_ro;

ALTER ROLE tau2_ro SET search_path = tau2_loop;
ALTER ROLE tau2_ro SET statement_timeout = '15s';
ALTER ROLE tau2_ro SET default_transaction_read_only = on;
REVOKE CREATE ON SCHEMA public FROM tau2_ro;
REVOKE ALL ON SCHEMA public FROM tau2_ro;
ALTER ROLE tau2_owner SET search_path = tau2_loop, public;

-- Reclaim anything a previous migration created as the superuser, before acting as
-- the owner below: a CREATE INDEX IF NOT EXISTS on a table owned by `nmp` is refused
-- to `tau2_owner`, so this must run first and as the superuser.
DO $$
DECLARE t text;
BEGIN
  FOR t IN SELECT tablename FROM pg_tables WHERE schemaname = 'tau2_loop' LOOP
    EXECUTE format('ALTER TABLE tau2_loop.%I OWNER TO tau2_owner', t);
  END LOOP;
END
$$;

-- The tables below are created as `tau2_owner`, not as the superuser running this
-- file: a table owned by `nmp` is invisible to the owner role (it holds no privilege
-- on it, so it does not even appear in information_schema), and the default
-- privileges granted above are declared FOR ROLE tau2_owner, so only tables it
-- actually owns hand `tau2_ro` its SELECT.
SET ROLE tau2_owner;

-- ── the human verdict on one scored conversation ─────────────────────────────
-- Append-only: the newest row per (run_id, task_id, trial) is the current review,
-- and a changed mind stays readable. `judge_reward` is copied from the run at the
-- time of review, so a later re-score cannot quietly rewrite what was disagreed with.
CREATE TABLE IF NOT EXISTS tau2_loop.review (
  id            bigserial PRIMARY KEY,
  run_id        text        NOT NULL,
  task_id       text        NOT NULL,
  trial         integer     NOT NULL,
  judge_reward  numeric     NOT NULL,
  verdict       text        NOT NULL CHECK (verdict IN ('agree', 'disagree', 'task_broken')),
  reason        text        NOT NULL DEFAULT '',
  author        text        NOT NULL DEFAULT '',
  code_sha      text        NOT NULL DEFAULT '',
  created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS review_conversation_idx
  ON tau2_loop.review (run_id, task_id, trial, created_at DESC);

-- ── one row per `submit prepare` for the τ²-bench leaderboard ────────────────
CREATE TABLE IF NOT EXISTS tau2_loop.submission (
  id            bigserial PRIMARY KEY,
  domain        text        NOT NULL,
  run_ids       text[]      NOT NULL,
  agent_model   text        NOT NULL,
  user_model    text        NOT NULL,
  trials        integer     NOT NULL,
  pass_hat_k    jsonb       NOT NULL,
  avg_cost_usd  numeric,
  note          text        NOT NULL DEFAULT '',
  pr_url        text,
  created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS submission_domain_idx
  ON tau2_loop.submission (domain, created_at DESC);

RESET ROLE;

GRANT SELECT ON ALL TABLES IN SCHEMA tau2_loop TO tau2_ro;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA tau2_loop TO tau2_ro;
