import { Fragment, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { type GoldenEntry, type GoldenSet, domainLabel, useGet } from '../lib/api';
import { useTask } from '../lib/scope';
import { Kpi, Loading, Points } from '../lib/ui';
import { taskId, taskPath, useLens, workflowEvalsPath } from '../lib/url';

/**
 * workflow_rag (s16): banking's second agent, which will research each question into a workflow
 * the answering agent looks up. Only its golden set exists yet, so Evals is its one built view: for
 * each train question, the three things the RAG agent is scored on (the workflows applied, the
 * information the customer must supply, the documents the task requires) and gold's calls.
 * Every question is listed with its split. Test is held out of optimisation, not hidden: a test
 * question's workflows are read off gold's calls by the rules that reproduce the train labels, and it
 * has no hand-labelled facts, summary or flag. Read from `data/workflows/<domain>/labels.json` (a
 * draft until a person confirms it), the tasks, and `test_counts.json`. Its tables stack into labelled
 * rows at phone width (`.tw.stack`, each cell's `data-label`).
 */

const enc = encodeURIComponent;
const AREAS = ['credit cards', 'bank accounts', 'debit cards', 'transfers'];
const taskNum = (id: string) => Number(id.replace(/\D/g, '')) || 0;
/** A workflow's name, free to wrap after each underscore and nowhere else. */
const breakable = (s: string) => s.split(/(?<=_)/).flatMap((part, i) => (i ? [<wbr key={i} />, part] : [part]));

export function NoAgent({ view, domain }: { view: string; domain: string }) {
  return (
    <>
      <p className="label">
        {view} · workflow_rag · {domainLabel(domain)}
      </p>
      <h1>{domainLabel(domain)} has no workflow_rag agent</h1>
      <div className="empty">Every agent lives under a dataset; workflow_rag is banking’s. Pick Banking in the dataset bar.</div>
    </>
  );
}

function Calls({ e }: { e: GoldenEntry }) {
  return (
    <ol className="small">
      {e.calls.map((c, i) => (
        <li key={i}>
          <span className={c.by === 'customer' ? 'v-ok' : ''}>{c.by}</span> · <code>{c.tool}</code>
          {Object.keys(c.args).length > 0 && (
            <span className="muted">
              {' '}
              {Object.entries(c.args)
                .map(([k, v]) => `${k}=${v}`)
                .join(', ')}
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}

/** The facts a question's workflows need, grouped by workflow, the held-back ones marked. */
function Info({ e }: { e: GoldenEntry }) {
  if (!e.info.length) return <p className="small muted">not labelled yet</p>;
  const groups = [...new Set(e.info.map((i) => i.workflow))].filter((g) => g !== 'verification');
  const verify = e.info.filter((i) => i.workflow === 'verification').map((i) => i.field);
  return (
    <>
      {verify.length > 0 && <p className="small muted">verification, when asked: {verify.join(', ')}</p>}
      {groups.map((g) => (
        <div key={g}>
          <p className="small">
            <b>{g}</b>
          </p>
          <ul className="small">
            {e.info
              .filter((i) => i.workflow === g)
              .map((i, k) => (
                <li key={k}>
                  <code>{i.field}</code> {i.value}
                  {i.held_back && <span className="v-warn"> · only if asked</span>}
                </li>
              ))}
          </ul>
        </div>
      ))}
      {e.distractors.length > 0 && (
        <p className="small muted">Red herrings the customer raises: {e.distractors.join('; ')}.</p>
      )}
    </>
  );
}

/** A question's golden entry in one row, opened into its calls, facts and documents. A test question's
 *  workflows are mapped by rule and it has no hand labels, so those cells say so. */
function QuestionRow({ e, domain, wf, open, toggle }: { e: GoldenEntry; domain: string; wf: string; open: boolean; toggle: () => void }) {
  const test = e.split === 'test';
  const own = e.info.filter((i) => i.workflow !== 'verification');
  const held = own.filter((i) => i.held_back).length;
  const newTools = e.new_tools ?? [];
  // a test question has no hand-written summary, so its cell names gold's writes and transfers
  const writes: [string, number][] = [];
  for (const c of e.calls) {
    if (c.tool.startsWith('get_') || c.tool === 'log_verification') continue;
    const last = writes[writes.length - 1];
    if (last && last[0] === c.tool) last[1] += 1;
    else writes.push([c.tool, 1]);
  }
  return (
    <>
      <tr>
        <td className="nw">
          <button type="button" className="more" aria-expanded={open} onClick={toggle}>
            {open ? '▾' : '▸'}
          </button>{' '}
          <Link to={taskPath(taskId(domain, e.task_id))} className="mono" title="the task, as the answering agent sees it">
            {e.task_id}
          </Link>
        </td>
        <td data-label="split">
          {test ? <span className="status warn">test</span> : <span className="status ok">train</span>}
        </td>
        <td data-label="workflows">
          {e.workflows.length > 0 ? (
            <span className="chips flat">
              {e.workflows.map((w) => (
                <span key={w} className={`chip ${w === wf ? 'ok' : ''}`}>
                  {breakable(w)}
                </span>
              ))}
            </span>
          ) : (
            <span className="small muted">none of train’s</span>
          )}
          {test && <span className="path">mapped by rule from gold’s calls</span>}
        </td>
        <td className={`num ${test ? 'muted' : ''}`} title={test ? 'facts are labelled on train only' : 'facts the workflows need · of which held back until asked'} data-label="information">
          {test ? '—' : own.length}
          {held > 0 && <span className="path v-warn">{held} if asked</span>}
        </td>
        <td className="num" data-label="documents">
          {e.required_documents.length}
        </td>
        <td className="wrap" data-label="gold did">
          {test ? (
            <>
              <span className="mono small">
                {writes.length
                  ? writes.map(([w, n], i) => (
                      <Fragment key={i}>
                        {i > 0 && ' · '}
                        {breakable(w)}
                        {n > 1 && ` ×${n}`}
                      </Fragment>
                    ))
                  : 'reads only'}
              </span>
              {newTools.length > 0 && (
                <span className="path">
                  <span className="v-warn">no train answer uses {newTools.join(', ')}</span>
                </span>
              )}
            </>
          ) : (
            e.gold
          )}
        </td>
        <td className={`wrap ${e.flag ? 'v-warn' : 'muted'}`} data-label="flag">
          {e.flag || '—'}
        </td>
      </tr>
      {open && (
        <tr>
          <td colSpan={7} className="wrap" data-label="">
            <p className="small">{e.situation}</p>
            <div className="cards">
              <div className="card">
                <h3>Workflows applied · {e.workflows.length}</h3>
                <p className="small">{e.workflows.join(' → ')}</p>
                <p className="label">gold’s calls · {e.calls.length}</p>
                <Calls e={e} />
              </div>
              <div className="card">
                <h3>Information required{test ? '' : ` · ${own.length}`}</h3>
                {test ? <p className="small muted">Labelled on train only: test is held out of optimisation, so no one writes its labels.</p> : <Info e={e} />}
              </div>
              <div className="card">
                <h3>Documents required · {e.required_documents.length}</h3>
                <ul className="small">
                  {e.required_documents.map((d) => (
                    <li key={d.id} title={d.id}>
                      {d.title ?? d.id}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

/** How the RAG agent will be scored on each question: three parts, and all three at once. */
const SCORES: [string, string, string][] = [
  ['workflow applied', 'the jobs it names for the question', 'every golden workflow named, and no other'],
  ['information required', 'the facts its workflow says to gather', 'recall of the golden facts, the held-back ones above all'],
  ['documents referenced', 'the documents its workflow cites', 'recall of the documents the task requires'],
  ['match', 'all three for the question', 'a question counts only when all three are right'],
];

export function WorkflowEvals() {
  const { domain = 'banking_knowledge' } = useParams();
  const [lens, setLens] = useLens();
  const task = useTask();
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const { data, error } = useGet<GoldenSet>(`/api/workflows/${enc(domain)}/golden`);
  if (domain !== 'banking_knowledge' && error) return <NoAgent view="Evals" domain={domain} />;
  if (!data) return <Loading error={error} />;

  const wf = lens.get('wf') ?? '';
  const flagged = lens.get('show') === 'flagged';
  const q = (lens.get('q') ?? '').trim().toLowerCase();
  const multi = data.entries.filter((e) => e.workflows.length > 1).length;
  const facts = data.entries.reduce((n, e) => n + e.info.filter((i) => i.workflow !== 'verification').length, 0);
  const held = data.entries.reduce((n, e) => n + e.info.filter((i) => i.workflow !== 'verification' && i.held_back).length, 0);
  const docs = data.entries.reduce((n, e) => n + e.required_documents.length, 0);
  const t = data.test;
  const split = lens.get('split') ?? '';
  const testEntries = data.test_entries ?? [];
  const splitOf = (e: GoldenEntry) => e.split ?? 'train';
  const questions = [...data.entries, ...testEntries].sort((a, b) => taskNum(a.task_id) - taskNum(b.task_id));
  const text = (e: GoldenEntry) =>
    `${e.task_id} ${splitOf(e)} ${e.gold} ${e.situation} ${e.workflows.join(' ')} ${(e.new_tools ?? []).join(' ')} ${e.info.map((i) => i.value).join(' ')}`;
  const rows = questions
    .filter((e) => !task || e.task_id === task)
    .filter((e) => !split || splitOf(e) === split)
    .filter((e) => !wf || e.workflows.includes(wf))
    .filter((e) => !flagged || !!e.flag)
    .filter((e) => !q || text(e).toLowerCase().includes(q));
  const byArea = [...data.workflows].sort((a, b) => AREAS.indexOf(a.area) - AREAS.indexOf(b.area) || b.train - a.train);

  return (
    <>
      <p className="label">Evals · workflow_rag · {domainLabel(domain)} · golden set, {data.status}</p>
      <h1>
        {data.entries.length} train questions need {data.workflows.length} workflows, and <em>{multi}</em> need more than one
      </h1>
      <Points
        lead
        items={[
          <>
            <b>Each question says three things:</b> the workflows to apply, the information the customer must supply, and the
            documents the task requires. A workflow is a rubric: what to ask, what qualifies, how to decide. The customer’s
            answers make it right for them, such as a fee they will not pay or a spend limit.
          </>,
          <>
            <b>{held} of {facts} facts are held back</b> until the agent asks. A workflow that does not ask for them recommends
            the wrong card or account.
          </>,
          t && (
            <>
              <b>Test is held out of optimisation, not hidden:</b> {t.covered} of {t.n_test} test questions use only these
              workflows and {t.needs_new} need a tool no train answer uses.
            </>
          ),
        ]}
      />
      <div className="kpis">
        <Kpi n={`${multi} / ${data.entries.length}`} b="questions that need more than one workflow" />
        <Kpi n={`${held} / ${facts}`} b="facts the customer gives only when asked" tone="warn" />
        <Kpi n={`${docs}`} b={`documents required across ${data.entries.length} questions`} />
        {t && <Kpi n={`${t.covered} / ${t.n_test}`} b="test questions covered by train’s workflows" tone={t.covered === t.n_test ? 'ok' : 'warn'} />}
      </div>

      <h2>How the RAG agent will be scored on each question</h2>
      <div className="tw fit stack">
        <table>
          <thead>
            <tr>
              <th>score</th>
              <th>the RAG agent’s</th>
              <th>against the golden set</th>
              <th>result</th>
            </tr>
          </thead>
          <tbody>
            {SCORES.map(([s, mine, gold]) => (
              <tr key={s}>
                <td className="sub">{s}</td>
                <td className="wrap" data-label="RAG agent’s">
                  {mine}
                </td>
                <td className="wrap" data-label="golden set">
                  {gold}
                </td>
                <td className="wrap muted" data-label="result">
                  no RAG-agent run yet
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>The workflows</h2>
      <div className="tw fit stack">
        <table>
          <thead>
            <tr>
              <th>workflow</th>
              <th className="num">train</th>
              <th className="num">test</th>
              <th>what it does</th>
              <th>information it needs</th>
              <th>defined by (knowledge base)</th>
            </tr>
          </thead>
          <tbody>
            {byArea.map((w) => (
              <tr key={w.name}>
                <td>
                  <button type="button" className={`chip nav ${wf === w.name ? 'on' : ''}`} onClick={() => setLens({ wf: wf === w.name ? '' : w.name })}>
                    {breakable(w.name)}
                  </button>
                  <span className="path">{w.area}</span>
                </td>
                <td className="num" data-label="train">
                  {w.train}
                </td>
                <td className={`num ${w.test === 0 ? 'muted' : ''}`} data-label="test">
                  {w.test}
                </td>
                <td className="wrap" data-label="does">
                  {w.does}
                </td>
                <td className="wrap small" data-label="needs">
                  {w.info_fields.map((f, i) => (
                    <span key={f.field} title={f.about}>
                      {i ? ', ' : ''}
                      <code>{f.field}</code>
                    </span>
                  ))}
                </td>
                <td className="wrap" data-label="defined by">
                  {w.kb_docs.length ? (
                    w.kb_docs.map((d) => (
                      <div key={d.id} className="small" title={d.id}>
                        {d.title ?? d.id}
                      </div>
                    ))
                  ) : (
                    <span className="small muted">no procedure document: each card’s own terms</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {t && t.needs_new > 0 && (
        <p className="small v-warn">
          Not in train · {t.needs_new} test questions need a tool no train answer uses:{' '}
          {Object.entries(t.new_tools)
            .sort((a, b) => b[1] - a[1])
            .map(([k, n]) => `${k} (${n})`)
            .join(', ')}
          .
        </p>
      )}
      {t && (
        <p className="small muted">
          Test counts come from fixed rules that reproduce {t.train_reproduced} of {data.entries.length} train labels; the{' '}
          {t.train_mismatches.length} they miss ({t.train_mismatches.join(', ')}) are workflows with no write of their own, so test
          can undercount those. Every question also verifies identity: {data.verification_fields.map((f) => f.field).join(', ')}.
        </p>
      )}

      <h2>
        All {questions.length} questions: {data.entries.length} train labelled by hand, {testEntries.length} test mapped by rule
      </h2>
      <div className="filters">
        <label className="pick">
          <span className="label">split</span>
          <select value={split} onChange={(e) => setLens({ split: e.target.value })}>
            <option value="">all · {questions.length}</option>
            <option value="train">train · {data.entries.length}</option>
            <option value="test">test · {testEntries.length}</option>
          </select>
        </label>
        <label className="pick">
          <span className="label">workflow</span>
          <select value={wf} onChange={(e) => setLens({ wf: e.target.value })}>
            <option value="">all</option>
            {byArea.map((w) => (
              <option key={w.name} value={w.name}>
                {w.name} · {w.train}
              </option>
            ))}
          </select>
        </label>
        <label className="pick">
          <span className="label">show</span>
          <select value={flagged ? 'flagged' : ''} onChange={(e) => setLens({ show: e.target.value })}>
            <option value="">all</option>
            <option value="flagged">flagged only</option>
          </select>
        </label>
        <input type="search" placeholder="task, split, workflow, fact, situation" value={lens.get('q') ?? ''} onChange={(e) => setLens({ q: e.target.value })} aria-label="search" />
        <span className="count">
          {rows.length} / {questions.length}
        </span>
      </div>
      {testEntries.length > 0 && t && (
        <p className="small muted">
          Test is held out of optimisation: no optimiser or RAG agent reads it. Its workflows are read off gold’s calls by the
          rules that give {t.train_reproduced} of {data.entries.length} train labels, and its facts, summary and flag are
          labelled on train only.
        </p>
      )}
      {task && (
        <p className="small muted">
          The scope bar narrows this to {task}. <Link to={workflowEvalsPath(domain)}>Show every question</Link>.
        </p>
      )}
      <div className="tw fit stack">
        <table>
          <thead>
            <tr>
              <th>task</th>
              <th>split</th>
              <th>workflows applied, in gold’s order</th>
              <th className="num">information required</th>
              <th className="num">documents required</th>
              <th>what gold did</th>
              <th>flag</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((e) => (
              <QuestionRow
                key={e.task_id}
                e={e}
                domain={domain}
                wf={wf}
                open={!!open[e.task_id]}
                toggle={() => setOpen((o) => ({ ...o, [e.task_id]: !o[e.task_id] }))}
              />
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">{data.source}</p>
    </>
  );
}

const PENDING: Record<string, { label: string; what: string }> = {
  runs: { label: 'Runs', what: 'Each RAG-agent run, on train and on held-out test, with its job, document, action and database scores.' },
  optimise: { label: 'Optimise', what: 'Each RAG-agent version and the train misses that changed it; misses are studied, never patched.' },
  review: { label: 'Review', what: 'The golden-set review: confirm or correct each train question’s workflows before they are golden.' },
};

/** workflow_rag's views not built yet (Runs, Optimise, Review): each says what it will show. */
export function WorkflowPending({ view, domain: fixed }: { view: keyof typeof PENDING; domain?: string }) {
  const params = useParams();
  const domain = fixed ?? params.domain ?? 'banking_knowledge';
  const p = PENDING[view];
  if (domain !== 'banking_knowledge') return <NoAgent view={p.label} domain={domain} />;
  return (
    <>
      <p className="label">
        {p.label} · workflow_rag · {domainLabel(domain)}
      </p>
      <h1>workflow_rag has no {p.label.toLowerCase()} yet; its golden set is in Evals</h1>
      <div className="empty">
        {p.what} Nothing of workflow_rag is built except its golden set: <Link to={workflowEvalsPath(domain)}>review it in Evals</Link>.
      </div>
    </>
  );
}
