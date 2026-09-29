import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  type Health,
  type RunMeta,
  type TaskResult,
  domainLabel,
  fmtPct,
  post,
  shortRun,
  shortTask,
  useGet,
  when,
} from '../lib/api';
import { Kpi, Loading } from '../lib/ui';
import { reviewPath, runPath, trialId, trialPath, useLens } from '../lib/url';

/**
 * The human verdict on a conversation the judge already scored — the one fact in
 * this project that cannot be a committed file, because a person types it after
 * the run. It is therefore the only write route, and the only page that needs the
 * central Postgres.
 *
 * With the database stopped, or in the demo image, the tab still renders: the
 * list is empty, the form is disabled, and the reason is on the page (s04 R-3).
 */

type ReviewRow = {
  id: number;
  run_id: string;
  task_id: string;
  trial: number;
  judge_reward: number;
  verdict: Verdict;
  reason: string;
  author: string;
  code_sha: string;
  created_at: string;
};
type Verdict = 'agree' | 'disagree' | 'task_broken';
type ReviewIndex = {
  writable: boolean;
  reason: string;
  current: Record<string, ReviewRow>;
  tally: Partial<Record<Verdict, number>>;
};
type ReviewOne = { writable: boolean; reason: string; history: ReviewRow[] };

const VERDICTS: [Verdict, string, string][] = [
  ['agree', 'the judge was right', 'the score matches what the transcript shows'],
  ['disagree', 'the judge was wrong', 'the agent did the job, or failed for a reason the score does not name'],
  ['task_broken', 'the task is broken', 'no agent could satisfy it as written — the scenario or the criteria are at fault'],
];

const LABEL: Record<Verdict, string> = {
  agree: 'agrees',
  disagree: 'disagrees',
  task_broken: 'task broken',
};

export function Review() {
  const [lens, setLens] = useLens();
  const { runId: openRun, taskId: openTask, trial: openTrial } = useParams();
  const nav = useNavigate();
  const [nonce, setNonce] = useState(0);
  const { data: health } = useGet<Health>('/healthz');
  const { data: runs } = useGet<RunMeta[]>('/api/runs');
  const scored = (runs ?? []).filter((r) => !r.dry_run && r.summary?.n_scored);
  const run = lens.get('run') || openRun || scored[scored.length - 1]?.run_id || '';
  const { data: index, error } = useGet<ReviewIndex>(
    run ? `/api/review?run_id=${encodeURIComponent(run)}` : '/api/review',
    nonce,
  );
  const { data: detail } = useGet<{ meta: RunMeta; results: TaskResult[] }>(
    run ? `/api/runs/${encodeURIComponent(run)}` : null,
  );
  const only = lens.get('only') ?? 'fail';

  if (!index) return <Loading error={error} />;
  const meta = detail?.meta;
  const rows = (detail?.results ?? []).filter((r) =>
    only === 'fail' ? r.correct === false : only === 'unreviewed' ? !index.current[trialId(r)] : true,
  );
  const reviewed = Object.keys(index.current).length;
  const disagreed = (index.tally.disagree ?? 0) + (index.tally.task_broken ?? 0);

  return (
    <>
      <p className="label">Review</p>
      <h1>
        The NL judge is a <em>model</em>. Without a human label set, a regression and a judge flake
        look the same
      </h1>
      <p className="lead">
        Four of τ²'s five checks are deterministic — the database hash, the expected actions, the
        things that must be said, the environment assertions. The fifth asks a model whether a
        sentence is true of the transcript, and it can be wrong twice: by missing a real failure, and
        by inventing one. This page records what a person thought, beside what the judge scored.
      </p>

      {!index.writable && (
        <div className="empty">
          <b>Read only.</b> {index.reason || health?.mode === 'demo' ? index.reason : ''} Reviews are
          the one thing here that is not a committed file, so they need the central Postgres; every
          other page works without it.
        </div>
      )}

      <div className="kpis">
        <Kpi n={String(reviewed)} b={`conversations reviewed${run ? ` in ${shortRun(run)}` : ''}`} />
        <Kpi
          n={String(index.tally.agree ?? 0)}
          b="where the judge was right — the ordinary case, and the one worth counting"
          tone="ok"
        />
        <Kpi
          n={String(index.tally.disagree ?? 0)}
          b="where the judge was wrong: the agent's score understates it"
          tone="warn"
        />
        <Kpi
          n={String(index.tally.task_broken ?? 0)}
          b="where no agent could pass the task as written"
          tone="warn"
        />
      </div>
      {reviewed > 0 && (
        <p className="small muted">
          {fmtPct(disagreed / reviewed)} of reviewed conversations were scored in a way a person
          disputed. Until that number is small, a change of a point or two in a pass rate is not
          evidence of anything.
        </p>
      )}

      <div className="filters">
        <label className="pick">
          <span className="label">run</span>
          <select
            value={run}
            onChange={(e) => {
              setLens({ run: e.target.value });
              nav(reviewPath(undefined, undefined, { run: e.target.value }));
            }}
          >
            {scored.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {shortRun(r.run_id)} · {domainLabel(r.domain)} · {r.agent}
              </option>
            ))}
          </select>
        </label>
        <label className="pick">
          <span className="label">show</span>
          <select value={only} onChange={(e) => setLens({ only: e.target.value })}>
            <option value="fail">failed conversations</option>
            <option value="unreviewed">not yet reviewed</option>
            <option value="all">all</option>
          </select>
        </label>
        <span className="count">{rows.length} conversations</span>
      </div>

      {openRun && openTask && openTrial && (
        <Editor
          runId={openRun}
          taskId={openTask}
          trial={openTrial}
          writable={index.writable}
          reason={index.reason}
          judge={(detail?.results ?? []).find(
            (r) => r.task_id === openTask && `t${r.trial}` === openTrial,
          )}
          onSaved={() => setNonce((n) => n + 1)}
          onClose={() => nav(reviewPath(undefined, undefined, { run }))}
        />
      )}

      {meta && (
        <div className="tw">
          <table>
            <caption>
              {domainLabel(meta.domain)} · {meta.agent} · {meta.split}. Pick a conversation to read
              the judge's verdict and record your own.
            </caption>
            <thead>
              <tr>
                <th>task</th>
                <th>judge</th>
                <th>which check</th>
                <th>human</th>
                <th>reason</th>
                <th>trace</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const cur = index.current[trialId(r)];
                const open = openTask === r.task_id && openTrial === `t${r.trial}`;
                return (
                  <tr
                    key={trialId(r)}
                    className={`click ${open ? 'pro' : ''}`}
                    onClick={() => nav(reviewPath(run, trialId(r)))}
                  >
                    <td className="sub mono small" title={r.task_id}>
                      {shortTask(r.task_id, 28)}
                      {meta.trials > 1 && <span className="path">trial {r.trial}</span>}
                    </td>
                    <td>
                      {r.correct ? (
                        <span className="status ok">pass</span>
                      ) : (
                        <span className="status err">fail</span>
                      )}
                    </td>
                    <td className="mono small">
                      {[
                        r.db_check === false ? 'DB ✗' : null,
                        r.action_checks ? `actions ${r.action_checks}` : null,
                        r.nl_assertions ? `NL ${r.nl_assertions}` : null,
                      ]
                        .filter(Boolean)
                        .join(' · ') || '—'}
                    </td>
                    <td>
                      {cur ? (
                        <span
                          className={`status ${cur.verdict === 'agree' ? 'ok' : 'warn'}`}
                          title={`${cur.author || 'anonymous'} · ${when(cur.created_at)}`}
                        >
                          {LABEL[cur.verdict]}
                        </span>
                      ) : (
                        <span className="status no">not reviewed</span>
                      )}
                    </td>
                    <td className="wrap small muted">{cur?.reason ?? ''}</td>
                    <td>
                      <Link to={trialPath(run, trialId(r))} onClick={(e) => e.stopPropagation()}>
                        open
                      </Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {meta && rows.length === 0 && (
        <div className="empty">
          Nothing matches this filter — <Link to={runPath(run)}>the run itself</Link> has{' '}
          {detail?.results.length ?? 0} conversations.
        </div>
      )}
    </>
  );
}

function Editor({
  runId,
  taskId,
  trial,
  writable,
  reason: why,
  judge,
  onSaved,
  onClose,
}: {
  runId: string;
  taskId: string;
  trial: string;
  writable: boolean;
  reason: string;
  judge: TaskResult | undefined;
  onSaved: () => void;
  onClose: () => void;
}) {
  const [nonce, setNonce] = useState(0);
  const { data } = useGet<ReviewOne>(
    `/api/review/${encodeURIComponent(runId)}/${encodeURIComponent(taskId)}/${encodeURIComponent(trial)}`,
    nonce,
  );
  const current = data?.history[0];
  const [verdict, setVerdict] = useState<Verdict | ''>('');
  const [reason, setReason] = useState('');
  const [author, setAuthor] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const save = async () => {
    if (!verdict) return;
    setSaving(true);
    setError(null);
    try {
      await post(
        `/api/review/${encodeURIComponent(runId)}/${encodeURIComponent(taskId)}/${encodeURIComponent(trial)}`,
        { verdict, reason, author },
      );
      setReason('');
      setVerdict('');
      setNonce((n) => n + 1);
      onSaved();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="card hi round-open">
      <p className="crumbs">
        <Link to={runPath(runId)}>{shortRun(runId)}</Link> /{' '}
        <span className="mono">{shortTask(taskId, 40)}</span> / {trial}
        <button type="button" className="linkish" onClick={onClose}>
          {' '}
          close
        </button>
      </p>
      <h3 style={{ marginTop: 0 }}>
        The judge scored this {judge?.correct ? 'a pass' : 'a failure'}
        {judge ? ` (reward ${judge.reward.toFixed(2)})` : ''} — was it right?
      </h3>
      {judge && (
        <p className="small muted">
          {[
            judge.db_check != null ? `database ${judge.db_check ? '✓' : '✗'}` : null,
            judge.action_checks ? `actions ${judge.action_checks}` : null,
            judge.communicate_checks ? `said ${judge.communicate_checks}` : null,
            judge.nl_assertions ? `NL assertions ${judge.nl_assertions}` : null,
          ]
            .filter(Boolean)
            .join(' · ')}{' '}
          · ended {judge.termination_reason} ·{' '}
          <Link to={trialPath(runId, `${taskId}/${trial}`)}>read the conversation</Link>
        </p>
      )}

      <fieldset disabled={!writable}>
        <div className="verdicts">
          {VERDICTS.map(([key, title, sub]) => (
            <label key={key} className={`opt ${verdict === key ? 'on' : ''}`}>
              <input
                type="radio"
                name="verdict"
                value={key}
                checked={verdict === key}
                onChange={() => setVerdict(key)}
              />
              <span className="t">
                <b>{title}</b>
                <span className="small muted">{sub}</span>
              </span>
            </label>
          ))}
        </div>
        <div className="row">
          <textarea
            placeholder="why — the sentence a future reader needs"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
          />
        </div>
        <div className="row">
          <input
            type="text"
            placeholder="who (optional)"
            value={author}
            onChange={(e) => setAuthor(e.target.value)}
            style={{ maxWidth: '16rem' }}
          />
          <button type="button" className="btn" disabled={!verdict || saving} onClick={save}>
            {saving ? 'saving…' : 'record this verdict'}
          </button>
          {error && <span className="v-warn small">{error}</span>}
        </div>
      </fieldset>
      {!writable && <p className="small v-warn">{why || 'read only here'}</p>}

      {current && (
        <>
          <h4 style={{ marginTop: 'var(--s5)' }}>
            {data?.history.length === 1 ? 'One review' : `${data?.history.length} reviews`}, newest
            first — the top one is current
          </h4>
          <div className="tw">
            <table>
              <thead>
                <tr>
                  <th>verdict</th>
                  <th>reason</th>
                  <th>who</th>
                  <th>judge said</th>
                  <th>when</th>
                </tr>
              </thead>
              <tbody>
                {data?.history.map((h, i) => (
                  <tr key={h.id} className={i ? 'dim' : ''}>
                    <td className={h.verdict === 'agree' ? 'v-ok' : 'v-warn'}>{LABEL[h.verdict]}</td>
                    <td className="wrap small">{h.reason || '—'}</td>
                    <td className="small">{h.author || 'anonymous'}</td>
                    <td className="num">{h.judge_reward.toFixed(2)}</td>
                    <td className="small nw">{when(h.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted">
            Rows are append-only and the judge's reward is copied at the time of review, so a later
            re-score cannot quietly rewrite what was disagreed with.
          </p>
        </>
      )}
    </section>
  );
}
