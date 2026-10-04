import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  type DomainDetail,
  type DomainSummary,
  type KbDocument,
  type TaskRow,
  byTask,
  domainLabel,
  shortTask,
  useGet,
} from '../lib/api';
import { RequiredDocs } from '../lib/docreads';
import { Loading, Points } from '../lib/ui';
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
      <p className="label">Evals · the benchmark</p>
      <h1>
        Four domains, each a policy, a toolset and a database; nothing is <em>held out</em>, so the
        test split is ours
      </h1>
      <Points
        lead
        items={[
          <>
            <b>τ²-bench ships every answer key</b>, so the loop cuts its own split: half of{' '}
            <code>base</code> for train, half for test.
          </>,
          <>
            <b>No test task was ever trained on</b>: cut v2 keeps v1's 20 train and 20 test per domain
            on the same sides.
          </>,
          <>
            <b>Drawn with seed 300</b>, committed under <code>data/splits/</code>; the agent reads only
            the policy and the tools.
          </>,
        ]}
      />
      <div className="tw">
        <table>
          <caption>Pick a domain for its policy, tools and tasks.</caption>
          <thead>
            <tr>
              <th>domain</th>
              <th className="num">base tasks</th>
              <th className="num">train</th>
              <th className="num">test</th>
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
      <Points
        className="small muted"
        items={[
          <>
            <b>Every base task is train or test</b>; none is held in reserve.
          </>,
          <>
            <b>Telecom's base is 114 of its 2,285 tasks</b>, tau2's own <code>base</code> split.
          </>,
          <>
            <b>banking_knowledge has no tau2 split</b>: its base is all 97, the odd one on the test side.
          </>,
        ]}
      />
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
  /** banking: the documents the agent needs, as tau2 lists them, and with their titles and sizes */
  required_documents?: string[];
  documents?: KbDocument[];
};

type SortKey = 'id' | 'split' | 'n_actions' | 'n_communicate' | 'n_nl_assertions' | 'n_documents';

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

  // an open task is the scope bar's task: the table narrows to it; "close" shows them all again
  const rows = data.tasks
    .filter((t) => !taskId || t.id === taskId)
    .filter((t) => (split ? t.split === split : true))
    .filter((t) =>
      q
        ? `${t.id} ${t.purpose ?? ''} ${t.goal ?? ''} ${t.relevant_policies ?? ''}`.toLowerCase().includes(q)
        : true,
    )
    .slice()
    .sort((a, b) => cmp(a, b, sort) * (desc ? -1 : 1));
  // banking's purposes are all tau2's placeholder, so the column is the goal the scenario states
  const goals = data.tasks.filter((t) => t.goal).length;
  const allGoals = goals > 0 && goals === data.tasks.length;
  const hasDocs = data.tasks.some((t) => (t.n_documents ?? 0) > 0);
  const ht = data.harness_tools;
  const tools = ht?.tools ?? data.tools;
  const knowledge = new Set(ht?.retrieval_info?.tools ?? []);

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
      <p className="label">Evals · the answering agent · {domainLabel(domain)}</p>
      <h1>
        Every base task, split in half <em>once</em>; the agent never sees a scenario or an expected
        action
      </h1>
      <Points
        lead
        items={[
          <>
            <b>Each task is a scenario the simulator plays</b>, with a purpose, policy clauses and the
            criteria the score multiplies.
          </>,
          <>
            <b>Train tasks feed the optimiser</b>; test tasks run once per challenger and are only
            reported.
          </>,
        ]}
      />

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
              {(task.documents ?? []).length > 0 && (
                <RequiredDocs domain={domain} taskId={task.id} docs={task.documents ?? []} />
              )}
            </div>
          </div>
          <Points
            className="small muted"
            items={[
              <>
                <b>The optimiser sees this key only for failed train tasks</b>, so test is the number
                worth reporting.
              </>,
              <>
                <b>The answer key is public</b>: it ships in <code>tasks.json</code>.
              </>,
            ]}
          />
        </section>
      )}

      <div className="filters">
        <label className="pick">
          <span className="label">split</span>
          <select value={split} onChange={(e) => setLens({ split: e.target.value })}>
            <option value="">all</option>
            <option value="train">train</option>
            <option value="test">test</option>
          </select>
        </label>
        <input
          type="search"
          placeholder={allGoals ? 'search id, goal' : 'search id, purpose, policies'}
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
              {allGoals ? (
                <th>
                  customer goal<span className="path">derived from the scenario, not tau2's</span>
                </th>
              ) : (
                <th>purpose</th>
              )}
              <th>policies</th>
              {head('n_actions', 'actions', true)}
              {head('n_communicate', 'say', true)}
              {head('n_nl_assertions', 'NL', true)}
              {hasDocs && head('n_documents', 'docs', true)}
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
                <td className="wrap small">
                  {t.goal ?? t.purpose ?? t.reason_for_call ?? '—'}
                  {t.goal && !allGoals && <span className="path">goal, derived from the scenario</span>}
                </td>
                <td className="wrap small muted">{t.relevant_policies ?? '—'}</td>
                <td className="num">{t.n_actions}</td>
                <td className="num">{t.n_communicate}</td>
                <td className="num">{t.n_nl_assertions}</td>
                {hasDocs && <td className="num">{t.n_documents ?? 0}</td>}
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
        Split method (v{data.split.version ?? 1}): <code>{data.split.method}</code>, seed {data.split.seed},{' '}
        {data.split.base_n} base tasks.
      </p>
      <details>
        <summary>
          <span>
            {tools.length} tools the harness exposes to{' '}
            {ht?.version ? `${domainLabel(domain)}'s champion, ${ht.version}` : 'the agent'}
            {ht?.retrieval ? ` · retrieval ${ht.retrieval}` : ''}
          </span>
        </summary>
        {ht?.retrieval && <ToolsNote ht={ht} />}
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>tool</th>
                <th>description</th>
              </tr>
            </thead>
            <tbody>
              {tools.map((t) => (
                <tr key={t.name}>
                  <td className="sub mono">
                    {t.name}
                    {ht?.retrieval && knowledge.has(t.name) && <span className="path">{ht.retrieval}</span>}
                  </td>
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

/** Which tools the list shows and why: the extract's, with banking's knowledge tools swapped for
 *  the champion's retrieval variant's, or the extract's alone where the variant cannot be read. */
function ToolsNote({ ht }: { ht: NonNullable<DomainDetail['harness_tools']> }) {
  const info = ht.retrieval_info;
  const dense = info?.dense_model ? info.dense_model.replace(/^local:/, '').replace(/^sentence-transformers\//, '') : null;
  if (!ht.known) {
    return (
      <p className="small muted">
        This image ships without tau2, so {ht.version}'s <code>{ht.retrieval}</code> tools cannot be read; these are the task
        extract's <code>{ht.extract_retrieval}</code> tools.
      </p>
    );
  }
  if (!ht.replaced.length) {
    return (
      <p className="small muted">
        {ht.version} runs tau2's <code>{ht.retrieval}</code> retrieval, the one the task extract lists.
      </p>
    );
  }
  return (
    <p className="small muted">
      The task extract lists <code>{ht.extract_retrieval}</code>'s {ht.replaced.map((n, i) => (
        <span key={n}>
          {i > 0 && ', '}
          <code>{n}</code>
        </span>
      ))}
      ; {ht.version} runs <code>{ht.retrieval}</code>, so its knowledge tools are{' '}
      {(info?.tools ?? []).map((n, i) => (
        <span key={n}>
          {i > 0 && ', '}
          <code>{n}</code>
        </span>
      ))}
      {dense ? ` (dense search on ${dense})` : ''}, read from tau2's variant spec.
    </p>
  );
}

/** The column a person picked, then always task order, so equal rows never shuffle. */
function cmp(a: TaskRow, b: TaskRow, key: SortKey): number {
  if (key === 'id') return byTask(a.id, b.id);
  const k = key === 'split' ? String(a.split).localeCompare(String(b.split)) : (a[key] ?? 0) - (b[key] ?? 0);
  return k || byTask(a.id, b.id);
}
