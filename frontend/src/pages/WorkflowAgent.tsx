import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  type GoldenSet,
  type RagEvent,
  type RagJob,
  type RagLibrary,
  type RagLock,
  type RagPayload,
  type RagScore,
  type RagSession,
  type RagSessionMeta,
  type RagVersion,
  domainLabel,
  fmtK,
  fmtS,
  get,
  post,
  shortModel,
  useGet,
} from '../lib/api';
import { Kpi, Loading, Points } from '../lib/ui';
import { taskId, taskPath, useLens, workflowEvalsPath } from '../lib/url';
import { NoAgent } from './Workflow';

/**
 * Agent, with the scope bar on workflow_rag (s16): the RAG agent's live demo. Pick a train question
 * and the agent researches the knowledge base with the answering agent's own tools (v4's
 * `alltools_minilm`: BM25, dense search, the read-only shell), turn by turn, then writes the
 * workflow the answering agent would follow. Each session runs on the server in the background
 * (`POST /api/workflows/<domain>/rag/sessions`) and this page polls it. The score beside the
 * workflow reads gold only after the session; the agent never sees it. Test questions are sealed
 * and cannot be picked.
 */

const enc = encodeURIComponent;
const MODEL_CHOICES = ['opus', 'sonnet', 'haiku'];
const EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'];
const DOC_ID = /\bdoc_[A-Za-z0-9_()-]+/g;

/** Polls a session every 1.5 s while it runs, keeping the last state on screen between polls. */
function useSession(url: string | null): { s: RagSession | null; err: string | null } {
  const [s, setS] = useState<RagSession | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setS(null);
    setErr(null);
    if (!url) return;
    let alive = true;
    let timer: number | undefined;
    const tick = () =>
      get<RagSession>(url)
        .then((d) => {
          if (!alive) return;
          setS(d);
          setErr(null);
          if (d.meta.status === 'running') timer = window.setTimeout(tick, 1500);
        })
        .catch((e: Error) => alive && setErr(e.message));
    tick();
    return () => {
      alive = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [url]);
  return { s, err };
}

const statusClass = (st: string) => (st === 'done' ? 'ok' : st === 'running' ? 'warn' : 'err');

function sessionLabel(m: RagSessionMeta): string {
  return `${m.started} · ${m.task_id} · ${m.agent} · ${shortModel(m.model)} ${m.effort} · ${m.status} · ${m.steps} steps`;
}

/** The query or command a call carries, as one line. */
function callLine(args: Record<string, unknown> | undefined): string {
  if (!args) return '';
  const v = args.query ?? args.command ?? JSON.stringify(args);
  return String(v);
}

function ToolBlock({ e }: { e: RagEvent }) {
  const docs = [...new Set((e.content ?? '').match(DOC_ID) ?? [])];
  return (
    <div className={`blk ${e.refused ? 'miss' : ''}`}>
      <span className="label">
        {e.name}
        {e.refused ? ' · refused' : ''}
        {e.cut ? ' · cut' : ''}
      </span>
      <p className="msg mono">{callLine(e.args)}</p>
      {docs.length > 0 && (
        <p className="sub">
          {docs.length} {docs.length === 1 ? 'document' : 'documents'}: {docs.slice(0, 6).join(', ')}
          {docs.length > 6 ? ` and ${docs.length - 6} more` : ''}
        </p>
      )}
      <details className="part">
        <summary>
          result <span className="muted">· {(e.content ?? '').length.toLocaleString('en-GB')} characters</span>
        </summary>
        <pre>{e.content}</pre>
      </details>
    </div>
  );
}

/** s21: what a concurrent merge decided, waited for, held and committed, one line each. */
const LOCK_KINDS = new Set(['decided', 'wait', 'locked', 'committed']);
function lockLine(e: RagEvent): string {
  if (e.kind === 'decided')
    return `Decided what it writes: ${Object.entries(e.decision ?? {})
      .map(([k, v]) => `${k} → ${v.join(', ') || 'a new job'}`)
      .join(' · ')}`;
  if (e.kind === 'wait')
    return `Waiting for ${(e.jobs ?? []).join(', ')}: ${(e.in_the_way ?? [])
      .map((w) => `${w.task ?? w.session} ${w.state === 'held' ? 'is merging' : 'is ahead in line for'} ${w.jobs.join(', ')}`)
      .join('; ')}`;
  if (e.kind === 'locked')
    return `Holds ${(e.jobs ?? []).join(', ')} (lock ${e.token}) after waiting ${fmtS(e.waited_ms ?? 0)} · read ${Object.entries(e.versions ?? {})
      .map(([j, v]) => `${j} v${v}`)
      .join(', ') || 'no library job'}${e.changed?.length ? ` · changed since it first read them: ${e.changed.join(', ')}` : ''}`;
  const wrote = Object.entries(e.versions ?? {});
  return `Committed ${wrote.map(([j, v]) => `${j} v${v}`).join(', ') || 'nothing'} under lock ${e.token}, then released it`;
}

/** s21: who is updating which workflow in a concurrent version's library, read every 3 s. */
function LibraryPanel({ domain, version, open }: { domain: string; version: string; open: (session: string, q: string | null) => void }) {
  const [lib, setLib] = useState<RagLibrary | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const tick = () =>
      get<RagLibrary>(`/api/workflows/${enc(domain)}/rag/library?version=${enc(version)}`)
        .then((d) => {
          if (!alive) return;
          setLib(d);
          setErr(null);
          timer = window.setTimeout(tick, 3000);
        })
        .catch((e: Error) => alive && setErr(e.message));
    tick();
    return () => {
      alive = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [domain, version]);
  if (err) return <p className="empty">Could not read {version}’s library: {err}</p>;
  if (!lib) return null;
  const heldBy = new Map<string, RagLock>();
  lib.held.forEach((h) => h.jobs.forEach((j) => heldBy.set(j, h)));
  const waitFor = new Map<string, RagLock[]>();
  lib.waiting.forEach((w) => w.jobs.forEach((j) => waitFor.set(j, [...(waitFor.get(j) ?? []), w])));
  const busy = (j: string) => heldBy.has(j) || waitFor.has(j);
  const rows = [...lib.jobs].sort(
    (a, b) => Number(busy(b.job)) - Number(busy(a.job)) || (b.version ?? 0) - (a.version ?? 0) || a.job.localeCompare(b.job),
  );
  const fresh = [...new Set([...heldBy.keys(), ...waitFor.keys()])].filter((j) => !lib.jobs.some((x) => x.job === j));
  const active = rows.filter((r) => busy(r.job) || (r.version ?? 0) > 0);
  const rest = rows.filter((r) => !active.includes(r));
  const now = (j: string) => {
    const h = heldBy.get(j);
    const w = waitFor.get(j) ?? [];
    return (
      <span className="chips flat">
        {h && <span className="chip lock">merging · {h.task} · {fmtS((h.held_s ?? 0) * 1000)}</span>}
        {w.length > 0 && (
          <span className="chip warn">
            {w.length} waiting · {w.map((x) => x.task).join(', ')}
          </span>
        )}
        {!h && !w.length && <span className="chip ok">free</span>}
      </span>
    );
  };
  const row = (r: RagLibrary['jobs'][number]) => (
    <tr key={r.job}>
      <td className="mono small">{r.job}</td>
      <td className="num">{r.version === 0 ? 'seed' : `v${r.version}`}</td>
      <td className="small">
        <button type="button" className="linkish" onClick={() => open(r.session, r.task)}>
          {r.task ?? r.session}
        </button>
      </td>
      <td>{now(r.job)}</td>
    </tr>
  );
  const changed = lib.jobs.filter((j) => (j.version ?? 0) > 0).length;
  return (
    <div className="agent-panel rag-library">
      <h3>
        Library · {lib.version} · {lib.jobs.length} workflows
      </h3>
      <p className="small muted">
        {lib.held.length} of {lib.workers} workers merging · {lib.waiting.length} waiting · {lib.commits ?? 0} commits · {changed} of {lib.jobs.length} workflows
        changed since the seed. Research takes no lock; a worker locks only the workflows its merge writes.
      </p>
      {fresh.length + active.length === 0 && <p className="small muted">No worker is merging and nothing has been committed yet.</p>}
      {fresh.length + active.length > 0 && (
      <div className="tw flat">
        <table>
          <thead>
            <tr>
              <th>workflow</th>
              <th className="num">version</th>
              <th>last change</th>
              <th>now</th>
            </tr>
          </thead>
          <tbody>
            {fresh.map((j) => (
              <tr key={j}>
                <td className="mono small">{j}</td>
                <td className="num">new</td>
                <td />
                <td>{now(j)}</td>
              </tr>
            ))}
            {active.map(row)}
          </tbody>
        </table>
      </div>
      )}
      {rest.length > 0 && (
        <details className="part">
          <summary>
            {rest.length} more, unchanged since the seed <span className="muted">· {lib.version} starts from them</span>
          </summary>
          <div className="tw flat">
            <table>
              <tbody>{rest.map(row)}</tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}

/** The research, one block per model turn, with each tool call it made and what came back. */
function Research({ s }: { s: RagSession }) {
  const start = s.events.find((e) => e.kind === 'start');
  const turns = s.events.filter((e) => e.kind === 'model');
  const running = s.meta.status === 'running';
  return (
    <div className="agent-panel rag-log">
      <h3>Research, step by step</h3>
      {start?.question && (
        <details className="part">
          <summary>
            The customer’s script <span className="muted">· all the agent reads about this question</span>
          </summary>
          <div className="blk">
            <pre className="tall">{start.question}</pre>
          </div>
        </details>
      )}
      {turns.map((m) => {
        const tools = s.events.filter((e) => e.kind === 'tool' && e.step === m.step);
        const said = s.events.filter((e) => e.kind === 'user' && e.step === m.step);
        const final = !m.calls?.length;
        return (
          <div key={m.step} className="step">
            {said.map((e, i) => (
              <details key={i} className="part">
                <summary>
                  The harness’s message <span className="muted">· before step {m.step}</span>
                </summary>
                <div className="blk">
                  <pre className="tall">{e.text}</pre>
                </div>
              </details>
            ))}
            <div className="hd">
              <span>
                step {m.step}
                {final ? ' · writes' : ` · ${tools.length} ${tools.length === 1 ? 'call' : 'calls'}`}
              </span>
              <span className="n">
                {fmtS(m.ms)} · {fmtK(m.input_tokens)} in · {fmtK(m.output_tokens)} out
              </span>
            </div>
            <div className="io">
              {m.text && (
                <div className="blk out">
                  <span className="label">{final ? 'its reply' : 'its note'}</span>
                  <pre className={final ? 'tall' : ''}>{m.text}</pre>
                </div>
              )}
              {tools.map((t) => (
                <ToolBlock key={t.id} e={t} />
              ))}
            </div>
          </div>
        );
      })}
      {s.events
        .filter((e) => e.kind === 'note' || e.kind === 'error' || LOCK_KINDS.has(e.kind))
        .map((e, i) => (
          <p key={i} className={`small ${e.kind === 'error' ? 'v-warn' : e.kind === 'wait' ? 'v-warn' : 'muted'}`}>
            {LOCK_KINDS.has(e.kind) ? lockLine(e) : e.text}
          </p>
        ))}
      {running && (
        <p className="small muted rag-wait">
          step {s.meta.steps + 1}: waiting for the model · the session has run {fmtS(Date.now() - Date.parse(isoOf(s.meta.started)))}
        </p>
      )}
    </div>
  );
}

/** `20261006T101500Z` → an ISO time the browser parses. */
function isoOf(stamp: string): string {
  const m = stamp.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/);
  return m ? `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}Z` : stamp;
}

function Found({ q, found }: { q: string | undefined; found: Map<string, boolean> }) {
  if (!q) return <span className="muted">—</span>;
  const f = found.get(q);
  return (
    <span className={f ? 'v-ok' : 'v-warn'} title={q}>
      {f ? '✓ quoted' : '✕ not in its document'}
    </span>
  );
}

function Job({ j, found }: { j: RagJob; found: Map<string, boolean> }) {
  return (
    <div className="card rag-job">
      <h3 className="mono">{j.job}</h3>
      {j.when?.quote && (
        <p className="small">
          <b>When:</b> “{j.when.quote}” <span className="muted">{j.when.doc}</span> <Found q={j.when.quote} found={found} />
        </p>
      )}
      {!!j.info?.length && (
        <>
          <p className="label">information it needs · {j.info.length}</p>
          <ul className="small">
            {j.info.map((i, k) => (
              <li key={k}>
                <code>{i.field}</code> {i.about} <span className="muted">· from {i.from ?? '—'}</span>
                {i.may_hold_back && <span className="v-warn"> · ask: “{i.ask}”</span>}
              </li>
            ))}
          </ul>
        </>
      )}
      {!!j.steps?.length && (
        <>
          <p className="label">steps · {j.steps.length}</p>
          <div className="tw flat">
            <table>
              <thead>
                <tr>
                  <th>step</th>
                  <th>by</th>
                  <th>call</th>
                  <th>source</th>
                </tr>
              </thead>
              <tbody>
                {j.steps.map((st) => (
                  <tr key={st.id}>
                    <td className="wrap">
                      <span className="mono">{st.id}</span> {st.do}
                      {st.if && <span className="muted"> · if {st.if}</span>}
                    </td>
                    <td className={st.by === 'harness' ? 'v-ok' : st.by === 'customer' ? 'v-warn' : ''}>{st.by}</td>
                    <td className="mono wrap">
                      {st.call ?? '—'}
                      {st.args && Object.keys(st.args).length > 0 && (
                        <span className="path">
                          {Object.entries(st.args)
                            .map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)
                            .join(' · ')}
                        </span>
                      )}
                    </td>
                    <td className="small">
                      <Found q={st.quote} found={found} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {!!j.rules?.length && (
        <>
          <p className="label">rules · {j.rules.length}</p>
          <ul className="small">
            {j.rules.map((r, k) => (
              <li key={k}>
                {r.rule} <Found q={r.quote} found={found} />
              </li>
            ))}
          </ul>
        </>
      )}
      {j.done_when && (
        <p className="small">
          <b>Done when:</b> {j.done_when}
        </p>
      )}
    </div>
  );
}

function ScoreKpis({ sc }: { sc: RagScore }) {
  const dr = sc.documents_referenced;
  const ds = sc.documents_seen;
  return (
    <div className="kpis">
      {dr && <Kpi n={`${dr.hit.length} of ${dr.required}`} b="required documents it cites" tone={dr.hit.length === dr.required ? 'ok' : 'warn'} />}
      <Kpi n={`${ds.hit.length} of ${ds.required}`} b={`required documents its searches surfaced (of ${ds.n} seen)`} tone={ds.hit.length === ds.required ? 'ok' : 'warn'} />
      {sc.quotes && <Kpi n={`${sc.quotes.found} of ${sc.quotes.n}`} b="quotes found in the document they cite" tone={sc.quotes.found === sc.quotes.n ? 'ok' : 'warn'} />}
      {sc.tools && <Kpi n={`${sc.tools.hit.length} of ${sc.tools.gold.length}`} b="tools in gold's calls that its steps call" tone={sc.tools.hit.length === sc.tools.gold.length ? 'ok' : 'warn'} />}
    </div>
  );
}

/** The workflow it wrote, checked: each quote looked up, and gold beside it for comparison. */
function Result({ s, domain }: { s: RagSession; domain: string }) {
  const sc = s.score;
  const wf = s.output;
  const found = useMemo(() => new Map((sc?.quotes?.rows ?? []).map((r) => [r.quote, r.found])), [sc]);
  const required = new Set((sc?.golden?.required_documents ?? []).map((d) => d.id));
  if (s.meta.status === 'running')
    return (
      <div className="agent-panel">
        <h3>The workflow</h3>
        <p className="small muted">It appears here when the agent writes it, with its score against the golden entry.</p>
      </div>
    );
  return (
    <div className="agent-panel">
      <h3>{wf ? `The workflow · ${wf.jobs.length} ${wf.jobs.length === 1 ? 'job' : 'jobs'}` : 'No workflow'}</h3>
      {!wf && <p className="small v-warn">{s.meta.error ?? 'The session ended without a workflow.'}</p>}
      {sc && <ScoreKpis sc={sc} />}
      {sc?.golden && (
        <p className="small">
          Jobs it named: <span className="mono">{(sc.jobs ?? []).join(' → ') || '—'}</span>. The golden set’s:{' '}
          <Link className="mono" to={workflowEvalsPath(domain, { q: s.meta.task_id })}>
            {sc.golden.workflows.join(' → ')}
          </Link>
          .
        </p>
      )}
      {wf?.jobs.map((j, i) => <Job key={i} j={j} found={found} />)}
      {!!wf?.documents?.length && (
        <>
          <p className="label">documents it cites · {wf.documents.length}</p>
          <ul className="small">
            {wf.documents.map((d) => (
              <li key={d.id}>
                <span className={required.has(d.id) ? 'v-ok' : ''}>{d.id}</span> <span className="muted">{d.why}</span>
              </li>
            ))}
          </ul>
        </>
      )}
      {!!wf?.open_questions?.length && (
        <>
          <p className="label">open questions</p>
          <ul className="small">
            {wf.open_questions.map((q, i) => (
              <li key={i}>{q}</li>
            ))}
          </ul>
        </>
      )}
      {sc?.golden && (
        <details className="part">
          <summary>
            Gold for {s.meta.task_id} <span className="muted">· read by the score only, never by the agent</span>
          </summary>
          <div className="blk">
            <p className="label">gold’s calls · {sc.golden.calls.length}</p>
            <ol className="small">
              {sc.golden.calls.map((c, i) => (
                <li key={i}>
                  {c.by} · <code>{c.tool}</code>
                  {sc.tools && (sc.tools.hit.includes(c.tool.replace(/^give /, '')) ? <span className="v-ok"> ✓</span> : <span className="v-warn"> ✕</span>)}
                </li>
              ))}
            </ol>
            <p className="label">required documents · {sc.golden.required_documents.length}</p>
            <ul className="small">
              {sc.golden.required_documents.map((d) => (
                <li key={d.id} title={d.id}>
                  {d.title ?? d.id}
                  {sc.documents_referenced?.hit.includes(d.id) ? <span className="v-ok"> · cited</span> : sc.documents_seen.hit.includes(d.id) ? <span className="v-warn"> · seen, not cited</span> : <span className="muted"> · not found</span>}
                </li>
              ))}
            </ul>
          </div>
        </details>
      )}
      {wf && (
        <details className="part">
          <summary>The workflow as JSON</summary>
          <div className="blk">
            <pre className="tall">{JSON.stringify(wf, null, 1)}</pre>
          </div>
        </details>
      )}
    </div>
  );
}

function VersionDetails({ v }: { v: RagVersion }) {
  return (
    <details>
      <summary>
        {v.name}’s prompt and tools <span className="muted">· rag_agents/banking_knowledge/{v.name}/</span>
      </summary>
      <p className="small">
        It calls {v.tools.map((t) => t.name).join(', ')}: the knowledge tools of the answering agent’s retrieval, <code>{v.retrieval}</code>. The answering agent’s other tools and the customer’s are listed in its prompt for it to name, never to call. Each reply is one SDK session on the subscription; a tool result over {v.result_chars.toLocaleString('en-GB')} characters is cut.
      </p>
      <div className="code">
        <pre>{v.prompt}</pre>
      </div>
    </details>
  );
}

export function WorkflowAgent() {
  const { domain = 'banking_knowledge' } = useParams();
  const [lens, setLens] = useLens();
  const [nonce, setNonce] = useState(0);
  const has = domain === 'banking_knowledge';
  const { data, error } = useGet<RagPayload>(has ? `/api/workflows/${enc(domain)}/rag` : null, nonce);
  const { data: golden } = useGet<GoldenSet>(has ? `/api/workflows/${enc(domain)}/golden` : null);
  const sid = lens.get('session') ?? '';
  const { s, err } = useSession(sid ? `/api/workflows/${enc(domain)}/rag/sessions/${enc(sid)}` : null);
  const v = data?.versions.find((x) => x.name === (lens.get('version') ?? '')) ?? data?.versions[data.versions.length - 1];
  const [model, setModel] = useState('');
  const [effort, setEffort] = useState('');
  const [busy, setBusy] = useState(false);
  const [startErr, setStartErr] = useState<string | null>(null);
  const q = lens.get('q') ?? s?.meta.task_id ?? 'task_019';
  const status = s?.meta.status;

  // the session list's status goes stale while one runs: read it again when that one ends
  useEffect(() => {
    if (status && status !== 'running') setNonce((n) => n + 1);
  }, [status]);

  if (!has) return <NoAgent view="Agent" domain={domain} />;
  if (!data || !v) return <Loading error={error} />;
  const entries = golden?.entries ?? [];
  const entry = entries.find((e) => e.task_id === q);

  const run = async () => {
    setBusy(true);
    setStartErr(null);
    try {
      const r = await post<{ id: string }>(`/api/workflows/${enc(domain)}/rag/sessions`, {
        task_id: q,
        agent: v.name,
        model: model || v.model,
        effort: effort || v.effort,
      });
      setLens({ session: r.id, q });
      setNonce((n) => n + 1);
    } catch (e) {
      setStartErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const running = data.sessions.some((m) => m.status === 'running');
  return (
    <>
      <p className="label">
        Agent · workflow_rag · {domainLabel(domain)} · {v.name} · {shortModel(v.model)} · {v.effort} effort · {v.retrieval} retrieval · max {v.max_steps} steps
      </p>
      <h1>
        workflow_rag {v.name} researches one train question with the answering agent’s own search tools, and writes its <em>workflow</em>
      </h1>
      <Points
        lead
        items={[
          <>
            <b>It reads the customer’s script and nothing else about the question.</b> Gold never reaches it; test questions cannot be picked.
          </>,
          <>
            <b>It searches iteratively</b> with BM25, dense search and the read-only shell, the tools v4 answers with, until every step has a source.
          </>,
          <>
            <b>It writes jobs, steps marked harness, model or customer, the facts to ask for, and the rules</b>, each with a quote from its document.
          </>,
          <>
            <b>The score reads gold only afterwards</b>: required documents cited and surfaced, quotes found, gold’s tools named.
          </>,
        ]}
      />

      <div className="filters agent-filters">
        <label className="pick">
          <span className="label">question</span>
          <select aria-label="train question" value={q} onChange={(e) => setLens({ q: e.target.value })}>
            {entries.map((e) => (
              <option key={e.task_id} value={e.task_id}>
                {e.task_id} · {e.workflows.join(' + ')} · {e.situation.slice(0, 70)}
                {e.situation.length > 70 ? '…' : ''}
              </option>
            ))}
          </select>
        </label>
        <label className="pick">
          <span className="label">model</span>
          <select aria-label="model" value={model || v.model} onChange={(e) => setModel(e.target.value)}>
            {MODEL_CHOICES.map((m) => (
              <option key={m} value={m}>
                {shortModel(m)}
                {m === v.model ? ` · ${v.name}'s` : ''}
              </option>
            ))}
          </select>
        </label>
        <label className="pick">
          <span className="label">effort</span>
          <select aria-label="effort" value={effort || v.effort} onChange={(e) => setEffort(e.target.value)}>
            {EFFORTS.map((x) => (
              <option key={x} value={x}>
                {x}
                {x === v.effort ? ` · ${v.name}'s` : ''}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="btn" onClick={run} disabled={busy || !data.live}>
          {busy ? 'Starting…' : `Research ${q}`}
        </button>
        <span className="count">{data.sessions.length} sessions{running ? ' · one running' : ''}</span>
      </div>
      {!data.live && <p className="empty">{data.reason}</p>}
      {startErr && <p className="empty v-warn">Could not start: {startErr}</p>}
      {entry && (
        <p className="small muted rag-q">
          <Link to={taskPath(taskId(domain, q))} className="mono">
            {q}
          </Link>{' '}
          · {entry.situation}
          {entry.situation.length >= 700 ? '…' : ''}
        </p>
      )}

      {data.sessions.length > 0 && (
        <div className="filters agent-filters">
          <label className="pick">
            <span className="label">session</span>
            <select aria-label="session" value={sid} onChange={(e) => setLens({ session: e.target.value, q: data.sessions.find((m) => m.id === e.target.value)?.task_id ?? q })}>
              {!sid && <option value="">pick a session</option>}
              {data.sessions.map((m) => (
                <option key={m.id} value={m.id}>
                  {sessionLabel(m)}
                </option>
              ))}
            </select>
          </label>
        </div>
      )}

      {v.merge === 'concurrent' && <LibraryPanel domain={domain} version={v.name} open={(session, task) => setLens({ session, q: task ?? q })} />}
      {err && <p className="empty">Could not load this session: {err}</p>}
      {sid && !s && !err && <Loading error={null} />}
      {s && (
        <>
          <p className="trialline small">
            <span className={`status ${statusClass(s.meta.status)}`}>{s.meta.status}</span> · {s.meta.task_id} · {shortModel(s.meta.model)} {s.meta.effort} · {s.meta.steps} steps · {s.meta.tool_calls} tool calls ·{' '}
            <span className="mono">{fmtK(s.meta.input_tokens)}</span> in · <span className="mono">{fmtK(s.meta.output_tokens)}</span> out
            {s.meta.status !== 'running' && <> · {fmtS(s.meta.duration_ms)}</>} · paid $0 (subscription)
            <span className="path rag-path">rag_agent_runs/{s.meta.id}/</span>
          </p>
          <div className="rag-cols">
            <Research s={s} />
            <Result s={s} domain={domain} />
          </div>
        </>
      )}
      {!sid && (
        <p className="empty">
          Pick a question and press Research. A session takes a few minutes: each step is one model reply, and the page shows it as it lands.
        </p>
      )}
      <VersionDetails v={v} />
    </>
  );
}

