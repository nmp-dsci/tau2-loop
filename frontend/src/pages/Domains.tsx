import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  type DomainDetail,
  type DomainSummary,
  type TaskRow,
  domainLabel,
  shortTask,
  useGet,
} from '../lib/api';
import { DomainChips, Loading } from '../lib/ui';
import { domainPath, taskId as makeTaskId, taskPath, useLens } from '../lib/url';

/**
 * The benchmark, one tab: the four domains, then a domain's tasks, then one
 * task. s04 M3 merged the old Data and Tasks pages, which were two views of the
 * same object reached from two places in the nav.
 */

export function Domains() {
  const { data: domains, error } = useGet<DomainSummary[]>('/api/domains');
  if (!domains) return <Loading error={error} />;
  return (
    <>
      <p className="label">The benchmark</p>
      <h1>
        Four domains, each a policy, a toolset and a database; nothing is <em>held out</em>, so the
        test split is ours
      </h1>
      <p className="lead">
        τ²-bench ships every task with its answer key. The loop therefore cuts its own split per
        domain — twenty train, twenty test, drawn once with seed 300 from the public{' '}
        <code>base</code> set and committed under <code>data/splits/</code> — and the agent reads
        only the policy and the tools; the scenario and the expected actions stay with the simulator
        and the evaluator.
      </p>
      <div className="tw">
        <table>
          <caption>Pick a domain to read its policy, its tools and its forty split tasks.</caption>
          <thead>
            <tr>
              <th>domain</th>
              <th className="num">base tasks</th>
              <th className="num">train</th>
              <th className="num">test</th>
              <th className="num">reserve</th>
              <th className="num">policy words</th>
              <th className="num">agent tools</th>
              <th>reward basis (tasks)</th>
            </tr>
          </thead>
          <tbody>
            {domains.map((d) => (
              <tr key={d.domain}>
                <td className="sub">
                  <Link to={domainPath(d.domain)}>{domainLabel(d.domain)}</Link>
                </td>
                <td className="num">{d.base_n}</td>
                <td className="num">{d.train}</td>
                <td className="num">{d.test}</td>
                <td className="num">{d.reserve_n}</td>
                <td className="num">{d.policy_words?.toLocaleString()}</td>
                <td className="num">{d.n_tools}</td>
                <td className="small wrap">
                  {Object.entries(d.reward_bases)
                    .map(([k, v]) => `${k} ${v}`)
                    .join(' · ')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        Telecom's base set is 114 of its 2,285 generated tasks (tau2's own <code>base</code> split);
        banking_knowledge has no tau2 split, so its base is all 97. The reserve is never run in this
        build.
      </p>
    </>
  );
}

type TaskFull = {
  id: string;
  split: string;
  description?: { purpose?: string; relevant_policies?: string; notes?: string };
  user_scenario?: { instructions?: Record<string, unknown> | string; persona?: string };
  evaluation_criteria?: {
    actions?: { name: string; arguments: Record<string, unknown>; info?: string }[];
    communicate_info?: string[];
    nl_assertions?: string[];
    env_assertions?: unknown[];
    reward_basis?: string[];
  };
};

type SortKey = 'id' | 'split' | 'n_actions' | 'n_communicate' | 'n_nl_assertions';

export function Domain() {
  const { domain = 'airline', taskId } = useParams();
  const nav = useNavigate();
  const [lens, setLens] = useLens();
  const { data, error } = useGet<DomainDetail>(`/api/domains/${encodeURIComponent(domain)}`);
  const { data: task } = useGet<TaskFull>(
    taskId ? `/api/domains/${encodeURIComponent(domain)}/tasks/${encodeURIComponent(taskId)}` : null,
  );
  const [open, setOpen] = useState(false);

  const split = lens.get('split') ?? '';
  const q = (lens.get('q') ?? '').toLowerCase();
  const sort = (lens.get('sort') as SortKey | null) ?? 'id';
  const desc = lens.get('desc') === '1';

  if (!data) return <Loading error={error} />;

  const rows = data.tasks
    .filter((t) => (split ? t.split === split : true))
    .filter((t) =>
      q ? `${t.id} ${t.purpose ?? ''} ${t.relevant_policies ?? ''}`.toLowerCase().includes(q) : true,
    )
    .slice()
    .sort((a, b) => cmp(a, b, sort) * (desc ? -1 : 1));

  const head = (key: SortKey, label: string, num = false) => (
    <th
      className={`sortable ${num ? 'num' : ''} ${sort === key ? 'sorted' : ''}`}
      onClick={() => setLens({ sort: key, desc: sort === key && !desc ? '1' : '' })}
      aria-sort={sort === key ? (desc ? 'descending' : 'ascending') : 'none'}
    >
      {label}
    </th>
  );

  return (
    <>
      <DomainChips current={domain} />
      <p className="label">Tasks</p>
      <h1>
        Forty tasks per domain, split <em>once</em>; the agent never sees a scenario or an expected
        action
      </h1>
      <p className="lead">
        Each task is a user scenario the simulator plays, a purpose the annotator wrote, the policy
        clauses it exercises, and the evaluation criteria the score is a product of. Train tasks feed
        the optimiser; test tasks are run once per promotion and only reported.
      </p>

      {task && (
        <section className="card hi task-open">
          <p className="crumbs">
            <Link to={domainPath(domain)}>{domainLabel(domain)}</Link> /{' '}
            <span className="mono">{shortTask(task.id, 60)}</span> ·{' '}
            {task.split === 'train' ? (
              <span className="status ok">train</span>
            ) : (
              <span className="status warn">{task.split}</span>
            )}
            <button type="button" className="linkish" onClick={() => nav(domainPath(domain))}>
              {' '}
              close
            </button>
          </p>
          <div className="grid2">
            <div>
              <h3>What the simulated user was told</h3>
              <div className="code">
                <pre>
                  {typeof task.user_scenario?.instructions === 'string'
                    ? task.user_scenario.instructions
                    : JSON.stringify(task.user_scenario?.instructions ?? {}, null, 1)}
                </pre>
              </div>
              {task.user_scenario?.persona && (
                <p className="small muted">
                  persona: {String(task.user_scenario.persona).slice(0, 300)}
                </p>
              )}
            </div>
            <div>
              <h3>What the evaluator checks — the answer key</h3>
              <p className="small">
                <b>purpose</b> {task.description?.purpose ?? '—'}
              </p>
              <p className="small">
                <b>policies</b> {task.description?.relevant_policies ?? '—'}
              </p>
              {task.description?.notes && (
                <p className="small">
                  <b>notes</b> {task.description.notes}
                </p>
              )}
              <p className="small">
                <b>reward basis</b>{' '}
                {(task.evaluation_criteria?.reward_basis ?? []).join(' × ') ||
                  'DB × COMMUNICATE (default)'}
              </p>
              <div className="code">
                <pre>
                  {(task.evaluation_criteria?.actions ?? [])
                    .map(
                      (a) =>
                        `${a.name}(${JSON.stringify(a.arguments)})${a.info ? `  # ${a.info}` : ''}`,
                    )
                    .join('\n') || '(no expected actions)'}
                </pre>
              </div>
              {(task.evaluation_criteria?.communicate_info ?? []).length > 0 && (
                <p className="small">
                  <b>must say</b> {(task.evaluation_criteria?.communicate_info ?? []).join(' · ')}
                </p>
              )}
              {(task.evaluation_criteria?.nl_assertions ?? []).length > 0 && (
                <p className="small">
                  <b>NL assertions</b>{' '}
                  {(task.evaluation_criteria?.nl_assertions ?? []).join(' · ')}
                </p>
              )}
            </div>
          </div>
          <p className="small muted">
            This is the answer key, and it is public: it ships in <code>tasks.json</code>. The
            optimiser is shown it only for <b>failed train tasks</b>, which is why the test split is
            the number worth reporting.
          </p>
        </section>
      )}

      <div className="filters">
        <label className="pick">
          <span className="label">split</span>
          <select value={split} onChange={(e) => setLens({ split: e.target.value })}>
            <option value="">all</option>
            <option value="train">train</option>
            <option value="test">test</option>
            <option value="reserve">reserve</option>
          </select>
        </label>
        <input
          type="search"
          placeholder="search id, purpose, policies"
          value={lens.get('q') ?? ''}
          onChange={(e) => setLens({ q: e.target.value })}
        />
        <span className="count">
          {rows.length} of {data.tasks.length} tasks
        </span>
      </div>

      <div className="tw">
        <table>
          <thead>
            <tr>
              {head('id', 'task')}
              {head('split', 'split')}
              <th>purpose</th>
              <th>policies</th>
              {head('n_actions', 'actions', true)}
              {head('n_communicate', 'say', true)}
              {head('n_nl_assertions', 'NL', true)}
              <th>basis</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((t) => (
              <tr
                key={t.id}
                className={`click ${t.id === taskId ? 'pro' : ''}`}
                onClick={() => nav(taskPath(makeTaskId(domain, t.id)))}
              >
                <td className="sub mono small" title={t.id}>
                  {shortTask(t.id, 30)}
                </td>
                <td>
                  {t.split === 'train' ? (
                    <span className="status ok">train</span>
                  ) : (
                    <span className="status warn">{t.split}</span>
                  )}
                </td>
                <td className="wrap small">{t.purpose ?? t.reason_for_call ?? '—'}</td>
                <td className="wrap small muted">{t.relevant_policies ?? '—'}</td>
                <td className="num">{t.n_actions}</td>
                <td className="num">{t.n_communicate}</td>
                <td className="num">{t.n_nl_assertions}</td>
                <td className="mono small">{(t.reward_basis ?? []).join('+')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>
        {domainLabel(domain)} — the policy the agent is given, verbatim (
        {data.policy_words.toLocaleString()} words)
      </h2>
      <p className="small muted">
        Split method: <code>{data.split.method}</code>, seed {data.split.seed}, {data.split.base_n}{' '}
        base tasks.
      </p>
      <details>
        <summary>{data.tools.length} tools the harness exposes to the agent</summary>
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>tool</th>
                <th>description</th>
              </tr>
            </thead>
            <tbody>
              {data.tools.map((t) => (
                <tr key={t.name}>
                  <td className="sub mono">{t.name}</td>
                  <td className="wrap small">{t.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <details open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
        <summary>policy.md</summary>
        <div className="code">
          <pre>{data.policy}</pre>
        </div>
      </details>
    </>
  );
}

function cmp(a: TaskRow, b: TaskRow, key: SortKey): number {
  if (key === 'id' || key === 'split') return String(a[key]).localeCompare(String(b[key]));
  return (a[key] ?? 0) - (b[key] ?? 0);
}
