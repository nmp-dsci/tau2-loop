import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  type LiveCall,
  type LiveEvent,
  type LiveMessage,
  type LivePayload,
  type LiveState,
  type RagSession,
  fmtS,
  get,
  post,
  shortModel,
  useGet,
} from '../lib/api';
import { ActionsDone } from '../lib/ui';
import { useLens } from '../lib/url';

/**
 * A live conversation on the Agent tab (s16): pick a train task, press Play, and the version
 * talks to the customer while the page polls `GET /api/live/conversations/<id>?since=` and
 * appends each message as the orchestrator adds it. A version with workflow tools (v6) also
 * shows each `find_workflow` lookup and each `request_workflow` research session from inside its
 * turn, with workflow_rag's own steps streaming beneath it. tau2 scores it at the end, as a run
 * would; it is not a run (`live_runs/<id>/`).
 */

const enc = encodeURIComponent;
const CUSTOMERS = ['haiku', 'sonnet', 'opus'];

/** Polls a conversation every 1.2 s while it runs, fetching only the events it has not seen. */
function useLive(id: string): { st: LiveState | null; err: string | null } {
  const [st, setSt] = useState<LiveState | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setSt(null);
    setErr(null);
    if (!id) return;
    let alive = true;
    let timer: number | undefined;
    let next = 0;
    let all: LiveEvent[] = [];
    const tick = () =>
      get<LiveState>(`/api/live/conversations/${enc(id)}?since=${next}`)
        .then((d) => {
          if (!alive) return;
          all = [...all, ...d.events];
          next = d.next;
          setSt({ meta: d.meta, events: all, next });
          setErr(null);
          if (d.meta.status === 'running') timer = window.setTimeout(tick, 1200);
        })
        .catch((e: Error) => alive && setErr(e.message));
    tick();
    return () => {
      alive = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [id]);
  return { st, err };
}

/** `20261006T082734Z` → an ISO time the browser parses. */
const isoOf = (stamp: string) => stamp.replace(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/, '$1-$2-$3T$4:$5:$6Z');

const argLine = (a: Record<string, unknown> | undefined) =>
  Object.entries(a ?? {})
    .map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`)
    .join(', ');

function Call({ c, result, who }: { c: LiveCall; result: LiveMessage | undefined; who: 'agent' | 'customer' }) {
  return (
    <div className={`live-call ${who}`}>
      <p className="small">
        <span className="label">{who === 'agent' ? 'agent calls' : 'customer calls'}</span> <code>{c.name}</code>
        <span className="muted mono"> {argLine(c.arguments)}</span>
      </p>
      {result ? (
        <details>
          <summary className="small">
            result{result.error ? <span className="v-warn"> · error</span> : null} <span className="muted">· {(result.content ?? '').length.toLocaleString('en-GB')} characters</span>
          </summary>
          <pre className="live-pre">{result.content}</pre>
        </details>
      ) : (
        <p className="small muted">running…</p>
      )}
    </div>
  );
}

/** workflow_rag's research for a `request_workflow`, polled while it runs. */
function Research({ domain, session }: { domain: string; session: string }) {
  const [s, setS] = useState<RagSession | null>(null);
  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const tick = () =>
      get<RagSession>(`/api/workflows/${enc(domain)}/rag/sessions/${enc(session)}`)
        .then((d) => {
          if (!alive) return;
          setS(d);
          if (d.meta.status === 'running') timer = window.setTimeout(tick, 2000);
        })
        .catch(() => undefined);
    tick();
    return () => {
      alive = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [domain, session]);
  const tools = (s?.events ?? []).filter((e) => e.kind === 'tool');
  return (
    <div className="live-research">
      <p className="small">
        <span className="label">workflow_rag researching</span>{' '}
        {s ? (
          <>
            {s.meta.status} · step {s.meta.steps} · {s.meta.tool_calls} searches
          </>
        ) : (
          'starting…'
        )}{' '}
        · <Link to={`/agent/${enc(domain)}/workflow_rag?session=${enc(session)}`}>open the session</Link>
      </p>
      {tools.length > 0 && (
        <ol className="small live-steps">
          {tools.slice(-8).map((t) => (
            <li key={t.id}>
              <code>{t.name}</code> <span className="muted mono">{String(t.args?.query ?? t.args?.command ?? '')}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

type Wf = { id: string; name: string; args?: Record<string, unknown>; session?: string | null; content?: string; at: number };

function WorkflowCard({ w, domain }: { w: Wf; domain: string }) {
  const ask = String(w.args?.request ?? '');
  const lookup = w.name === 'find_workflow';
  const best = lookup && w.content ? (w.content.match(/\n- ([a-z0-9_]+) \(match ([\d.]+)\)/) ?? null) : null;
  return (
    <div className="live-wf">
      <p className="small">
        <span className="label">{lookup ? 'agent looks up the workflow library' : 'agent asks workflow_rag to research the procedure'}</span>
      </p>
      <p className="small">
        <code>{w.name}</code> “{ask}”
        {w.args?.job ? <span className="muted"> · job {String(w.args.job)}</span> : null}
      </p>
      {w.session && <Research domain={domain} session={w.session} />}
      {w.content === undefined ? (
        <p className="small muted">{lookup ? 'looking up…' : 'workflow_rag is researching; the customer waits…'}</p>
      ) : (
        <>
          {best && (
            <p className="small">
              best match <b className="mono">{best[1]}</b> <span className="muted">· score {best[2]}</span>
            </p>
          )}
          {lookup && !best && <p className="small v-warn">no workflow in the library fits</p>}
          <details>
            <summary className="small">
              what it returned <span className="muted">· {w.content.length.toLocaleString('en-GB')} characters</span>
            </summary>
            <pre className="live-pre">{w.content}</pre>
          </details>
        </>
      )}
    </div>
  );
}

/** The conversation in order: messages, each call with its result beneath it, and the
 *  workflow lookups placed before the reply they fed. */
function Timeline({ st, domain }: { st: LiveState; domain: string }) {
  const msgs = st.events.filter((e) => e.kind === 'message' && e.message).map((e) => e.message as LiveMessage);
  const results = new Map<string, LiveMessage>();
  for (const m of msgs) if (m.role === 'tool' && m.id) results.set(m.id, m);
  const wfs = new Map<string, Wf>();
  for (const e of st.events) {
    if (!e.kind.startsWith('workflow_') || !e.id) continue;
    const w = wfs.get(e.id) ?? { id: e.id, name: e.name ?? '', at: e.at ?? 0 };
    if (e.kind === 'workflow_call') w.args = e.args;
    if (e.kind === 'workflow_research') w.session = e.session;
    if (e.kind === 'workflow_result') {
      w.content = e.content ?? '';
      w.session = w.session ?? e.session;
    }
    wfs.set(e.id, w);
  }
  const byAt = new Map<number, Wf[]>();
  for (const w of wfs.values()) byAt.set(w.at, [...(byAt.get(w.at) ?? []), w]);
  const out: JSX.Element[] = [];
  const pushWfs = (i: number) => (byAt.get(i) ?? []).forEach((w) => out.push(<WorkflowCard key={`wf-${w.id}`} w={w} domain={domain} />));
  msgs.forEach((m, i) => {
    pushWfs(i);
    if (m.role === 'tool') return; // shown under its call
    const who = m.role === 'assistant' ? 'agent' : 'customer';
    if (m.content)
      out.push(
        <div key={`m-${i}`} className={`live-msg ${who}`}>
          <span className="label">{who === 'agent' ? 'agent' : 'customer'}</span>
          <p>{m.content}</p>
        </div>,
      );
    for (const c of m.tool_calls ?? []) out.push(<Call key={`c-${i}-${c.id}`} c={c} result={results.get(c.id)} who={who} />);
  });
  pushWfs(msgs.length);
  const last = msgs[msgs.length - 1];
  const running = st.meta.status === 'running';
  const next = !last ? 'the agent' : last.role === 'assistant' && !last.tool_calls?.length ? 'the customer' : 'the agent';
  return (
    <div className="live-chat">
      {out}
      {running && (
        <p className="small muted live-wait">
          {next} is writing… · the conversation has run {fmtS(Date.now() - Date.parse(isoOf(st.meta.started)))}
        </p>
      )}
    </div>
  );
}

function Outcome({ st }: { st: LiveState }) {
  const done = st.events.find((e) => e.kind === 'done');
  const err = st.events.find((e) => e.kind === 'error');
  const m = st.meta;
  if (err) return <p className="empty v-warn">The conversation failed: {err.text}</p>;
  if (!done) return null;
  const checks = done.action_checks ?? [];
  const lookups = st.events.filter((e) => e.kind === 'workflow_call');
  return (
    <div className="live-done">
      <h3>
        <span className={`status ${m.correct ? 'ok' : 'err'}`}>{m.correct ? 'passed' : 'failed'}</span> · reward {m.reward ?? '—'} · ended {m.termination}
      </h3>
      {checks.length > 0 && <ActionsDone frac={m.actions_done} count={m.action_checks ?? ''} title="gold's expected actions this conversation made" />}
      <p className="small">
        {lookups.length} workflow {lookups.length === 1 ? 'call' : 'calls'} ({lookups.map((e) => e.name).join(', ') || 'none'}) · database {done.db_match == null ? 'not checked' : done.db_match ? 'matches gold' : 'differs from gold'} · {fmtS(m.duration_ms)}
      </p>
      {checks.length > 0 && (
        <ol className="small">
          {checks.map((c, i) => (
            <li key={i}>
              <span className={c.matched ? 'v-ok' : 'v-warn'}>{c.matched ? '✓' : '✕'}</span> {c.requestor === 'user' ? 'customer' : 'agent'} · <code>{c.name}</code> <span className="muted mono">{argLine(c.arguments).slice(0, 140)}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export function LiveConversation({ domain, agent, workflows }: { domain: string; agent: string; workflows: string | null }) {
  const [lens, setLens] = useLens();
  const [nonce, setNonce] = useState(0);
  const { data } = useGet<LivePayload>(`/api/live/${enc(domain)}/${enc(agent)}`, nonce);
  const id = lens.get('live') ?? '';
  const { st, err } = useLive(id);
  const [task, setTask] = useState('');
  const [customer, setCustomer] = useState('haiku');
  const [busy, setBusy] = useState(false);
  const [startErr, setStartErr] = useState<string | null>(null);
  const status = st?.meta.status;
  useEffect(() => {
    if (status && status !== 'running') setNonce((n) => n + 1);
  }, [status]);
  if (!data) return null;
  const pick = task || st?.meta.task_id || (data.tasks.some((t) => t.id === 'task_019') ? 'task_019' : (data.tasks[0]?.id ?? ''));
  const goal = data.tasks.find((t) => t.id === pick)?.goal;

  const play = async () => {
    setBusy(true);
    setStartErr(null);
    try {
      const r = await post<{ id: string }>(`/api/live/${enc(domain)}/${enc(agent)}`, { task_id: pick, user_model: customer });
      setLens({ live: r.id });
      setNonce((n) => n + 1);
    } catch (e) {
      setStartErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section id="live">
      <h2>
        Live · {agent} plays a train task now, each message shown as it lands{workflows ? `, with its workflow_rag ${workflows} lookups` : ''}
      </h2>
      <div className="filters agent-filters">
        <label className="pick">
          <span className="label">task</span>
          <select aria-label="train task" value={pick} onChange={(e) => setTask(e.target.value)}>
            {data.tasks.map((t) => (
              <option key={t.id} value={t.id}>
                {t.id}
                {t.goal ? ` · ${t.goal.slice(0, 80)}${t.goal.length > 80 ? '…' : ''}` : ''}
              </option>
            ))}
          </select>
        </label>
        <label className="pick">
          <span className="label">customer</span>
          <select aria-label="customer model" value={customer} onChange={(e) => setCustomer(e.target.value)}>
            {CUSTOMERS.map((m) => (
              <option key={m} value={m}>
                {shortModel(m)}
                {m === 'haiku' ? ' · the runs’' : ''}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="btn" onClick={play} disabled={busy || !data.live || !pick}>
          {busy ? 'Starting…' : `Play ${pick}`}
        </button>
        {data.conversations.length > 0 && (
          <label className="pick">
            <span className="label">earlier</span>
            <select aria-label="live conversation" value={id} onChange={(e) => setLens({ live: e.target.value })}>
              {!id && <option value="">pick one</option>}
              {data.conversations.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.started} · {c.task_id} · {c.status}
                  {c.correct == null ? '' : c.correct ? ' · passed' : ' · failed'}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {!data.live && <p className="empty">{data.reason}</p>}
      {startErr && <p className="empty v-warn">Could not start: {startErr}</p>}
      {goal && !id && <p className="small muted rag-q">{goal}</p>}
      {err && <p className="empty">Could not load the conversation: {err}</p>}
      {st && (
        <>
          <p className="trialline small">
            <span className={`status ${st.meta.status === 'done' ? (st.meta.correct ? 'ok' : 'err') : st.meta.status === 'running' ? 'warn' : 'err'}`}>{st.meta.status}</span> · {st.meta.task_id} · agent {shortModel(st.meta.model)} · customer {shortModel(st.meta.user_model)} · {st.events.filter((e) => e.kind === 'message').length} messages
            <span className="path rag-path">live_runs/{st.meta.id}/ · not a run: never logged, gated or graded into a ledger</span>
          </p>
          <Timeline st={st} domain={domain} />
          <Outcome st={st} />
        </>
      )}
    </section>
  );
}
