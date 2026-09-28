import { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom';
import {
  type AgentInfo,
  DOMAINS,
  type Health,
  type Registries,
  type RunAgent,
  type RunMeta,
  type TaskResult,
  type TrialPayload,
  domainLabel,
  fmtK,
  shortModel,
  shortRun,
  useGet,
} from '../lib/api';
import {
  AgentGraph,
  type NodeKey,
  NodePanel,
  type PgRequest,
  Replay,
  RoundLink,
  apiBehind,
  legacyNode,
  outcome,
} from '../lib/agentgraph';
import { Loading } from '../lib/ui';
import { agentPath, optimisePath, parseTrialId, trialId, trialPath, useLens } from '../lib/url';

/**
 * The agent, doing something (s05). Pick a version, one of its runs and one
 * conversation; the graph draws the agent in its harness, filled in from that
 * conversation, and every node opens a panel of what it received and produced.
 * Tools carry a playground that re-runs a call against the database as it stood.
 *
 * The lens is `?run=&trial=&node=&step=`. An address missing a part is completed
 * in place: the version's champion or latest run, then its first failed
 * conversation, then the agent node. `/agent` alone reopens the last view in this
 * browser tab (sessionStorage, as DataAgentBench's AGENT_VIEW), else the airline
 * champion. The figure this tab drew before (how a call reaches the subscription)
 * is now the model node's panel; its `?node=` keys still land (`legacyNode`).
 */

type AgentsPayload = { versions: AgentInfo[]; registry: Registries };
type RunPayload = { meta: RunMeta; results: TaskResult[] };

const VIEW_KEY = 'tau2loop.agent_view';

function readView(): string | null {
  try {
    return sessionStorage.getItem(VIEW_KEY);
  } catch {
    return null;
  }
}

function writeView(v: string): void {
  try {
    sessionStorage.setItem(VIEW_KEY, v);
  } catch {
    // private windows refuse storage; the address still carries the view
  }
}

/** The run a version opens on: the one the registry names for it, else its latest. */
function defaultRun(runs: RunMeta[], reg: Registries, domain: string, name: string): string | null {
  const r = reg[domain];
  for (const e of [r?.champion, r?.challenger]) if (e?.agent === name && runs.some((m) => m.run_id === e.run_id)) return e.run_id;
  const mine = runs.filter((m) => m.agent === name && !m.dry_run && m.summary).sort((a, b) => b.started_at.localeCompare(a.started_at));
  return mine[0]?.run_id ?? null;
}

/** Failures first, then the run's own task order. */
function ordered(results: TaskResult[], taskIds: string[]): TaskResult[] {
  const pos = (id: string) => (taskIds.includes(id) ? taskIds.indexOf(id) : 1 << 30);
  return results.slice().sort((a, b) => (a.correct === b.correct ? pos(a.task_id) - pos(b.task_id) || a.trial - b.trial : a.correct === false ? -1 : 1));
}

function role(reg: Registries, domain: string, name: string): string {
  if (reg[domain]?.champion?.agent === name) return 'champion';
  if (reg[domain]?.challenger?.agent === name) return 'challenger';
  return 'held';
}

export function Agent() {
  const { domain, name } = useParams();
  const [lens, setLens] = useLens();
  const nav = useNavigate();
  const loc = useLocation();
  const [pg, setPg] = useState<PgRequest | null>(null);

  const runId = lens.get('run');
  const trialLens = lens.get('trial');
  const rawNode = lens.get('node');
  const node: NodeKey | null = legacyNode(rawNode);
  const step = lens.get('step') != null && lens.get('step') !== '' ? Number(lens.get('step')) : null;
  const parsed = trialLens ? parseTrialId(trialLens) : null;

  const { data: agents, error } = useGet<AgentsPayload>('/api/agents');
  const { data: health } = useGet<Health>('/healthz');
  const { data: runs } = useGet<RunMeta[]>(domain ? `/api/runs?domain=${encodeURIComponent(domain)}` : null);
  const enc = encodeURIComponent;
  const { data: run } = useGet<RunPayload>(runId ? `/api/runs/${enc(runId)}` : null);
  const { data: snap } = useGet<RunAgent>(runId ? `/api/runs/${enc(runId)}/agent` : null);
  const { data: t, error: tErr } = useGet<TrialPayload>(runId && parsed ? `/api/runs/${enc(runId)}/${enc(parsed.task)}/t${parsed.trial}` : null);

  // `/agent` alone: the last view, else the first domain's champion
  useEffect(() => {
    if ((domain && name) || !agents) return;
    const last = readView();
    if (last?.startsWith('/agent/')) {
      nav(last, { replace: true });
      return;
    }
    const d = DOMAINS.find((x) => agents.registry[x]?.champion);
    const ch = d ? agents.registry[d]?.champion : null;
    if (d && ch) nav(agentPath(d, ch.agent, { run: ch.run_id }), { replace: true });
    else if (agents.versions[0]) nav(agentPath(agents.versions[0].domain, agents.versions[0].name), { replace: true });
  }, [domain, name, agents, nav]);

  // complete the address in one step: run, then conversation, then node
  useEffect(() => {
    if (!domain || !name || !agents || !runs) return;
    const patch: Record<string, string> = {};
    const rid = runId ?? defaultRun(runs, agents.registry, domain, name);
    if (!runId && rid) patch.run = rid;
    if (rawNode && legacyNode(rawNode) !== rawNode) patch.node = legacyNode(rawNode) ?? '';
    if (!trialLens && run && run.meta.run_id === rid) {
      const first = ordered(run.results, run.meta.task_ids)[0];
      if (first) patch.trial = trialId(first);
      if (!rawNode) patch.node = 'agent';
    }
    if (Object.keys(patch).length) setLens(patch);
    // setLens is rebuilt every render, so it is not a dependency; these decide the patch
  }, [domain, name, agents, runs, run, runId, trialLens, rawNode]);

  // remember a complete view, so `/agent` reopens it
  useEffect(() => {
    if (domain && name && runId && trialLens) writeView(`${loc.pathname}${loc.search}`);
  }, [domain, name, runId, trialLens, loc.pathname, loc.search]);

  // a playground request belongs to one conversation
  useEffect(() => setPg(null), [runId, trialLens]);

  if (!agents) return <Loading error={error} />;
  if (!domain || !name) return <Loading error={null} />;

  const version = agents.versions.find((v) => v.domain === domain && v.name === name);
  if (!version) {
    return (
      <>
        <p className="label">The agent</p>
        <h1>
          There is no <em>{`${domain}/${name}`}</em> agent to draw
        </h1>
        <p>
          <Link to="/agent">Open the champion instead</Link>.
        </p>
      </>
    );
  }
  const versionRuns = (runs ?? []).filter((m) => m.agent === name).sort((a, b) => b.started_at.localeCompare(a.started_at));
  const pick = (n: NodeKey) => setLens({ node: node === n ? '' : n, step: '' });
  const onRow = (n: NodeKey, i: number) => setLens({ node: n, step: String(i) });
  const onRun = (r: Omit<PgRequest, 'nonce'>) => {
    setPg({ ...r, nonce: Date.now() });
    if (node !== r.node) setLens({ node: r.node, step: '' });
  };

  const filters = (
    <div className="filters agent-filters">
      <label className="pick">
        <span className="label">agent</span>
        <select aria-label="agent version" value={`${domain}/${name}`} onChange={(e) => nav(agentPath(...(e.target.value.split('/') as [string, string])))}>
          {DOMAINS.map((d) => (
            <optgroup key={d} label={domainLabel(d)}>
              {agents.versions
                .filter((v) => v.domain === d)
                .map((v) => (
                  <option key={v.ref} value={`${v.domain}/${v.name}`}>
                    {domainLabel(v.domain)}/{v.name} · {role(agents.registry, v.domain, v.name)} · {v.fingerprint.slice(0, 7)} · {v.runs.length} {v.runs.length === 1 ? 'run' : 'runs'}
                  </option>
                ))}
            </optgroup>
          ))}
        </select>
      </label>
      <label className="pick">
        <span className="label">run</span>
        <select aria-label="run" value={runId ?? ''} onChange={(e) => setLens({ run: e.target.value, trial: '', step: '' })} disabled={!versionRuns.length}>
          {!versionRuns.length && <option value="">no runs</option>}
          {versionRuns.map((m) => (
            <option key={m.run_id} value={m.run_id}>
              {shortRun(m.run_id)} · {m.summary ? `${m.summary.passed}/${m.summary.n_scored} passed` : 'not scored'}
            </option>
          ))}
        </select>
      </label>
      {run && (
        <label className="pick">
          <span className="label">conversation</span>
          <select aria-label="conversation" value={trialLens ?? ''} onChange={(e) => setLens({ trial: e.target.value, step: '' })}>
            {ordered(run.results, run.meta.task_ids).map((r) => (
              <option key={trialId(r)} value={trialId(r)}>
                {domainLabel(domain)}/{r.task_id.length > 40 ? `${r.task_id.slice(0, 39)}…` : r.task_id} t{r.trial} · {r.correct ? '✓ pass' : '✕ fail'} · {r.n_agent_turns} turns · {r.n_tool_calls} tool calls
              </option>
            ))}
          </select>
        </label>
      )}
      {run?.meta.summary && (
        <span className="count">
          {run.meta.summary.passed}/{run.meta.summary.n_scored} passed · {run.meta.summary.n_scored - run.meta.summary.passed} failed
        </span>
      )}
    </div>
  );

  const behind = apiBehind(t);
  const head = t && !behind ? outcome(t) : null;
  const meta = run?.meta;
  return (
    <>
      <p className="label">
        The agent · {domainLabel(domain)}/{name}@{version.fingerprint.slice(0, 7)} · {role(agents.registry, domain, name)} · {shortModel(String(version.config.model))} · {String(version.config.tool_mode)} tools · max {String(version.config.max_steps ?? '—')} steps
        {version.diagnosis && (
          <>
            {' '}
            · <RoundLink domain={domain} name={name} />
          </>
        )}
      </p>
      {head ? (
        <h1 className="agent-h1">
          {head.lead}
          <em>{head.em}</em>
          {head.tail}
        </h1>
      ) : (
        <h1>
          Pick a run and a conversation, and every node shows <em>what went in and what came out</em>
        </h1>
      )}
      {filters}

      {!versionRuns.length && runs && (
        <p className="empty">
          {domainLabel(domain)}/{name} has no committed run, so there is no conversation to draw. Its files and the reasoning that wrote it are on <Link to={optimisePath(domain, name)}>Optimise</Link>.
        </p>
      )}
      {tErr && <p className="empty">Could not load this conversation: {tErr}</p>}
      {behind && (
        <p className="empty">
          The API behind this page is older than the page: it sent this conversation without its messages, so there is nothing to draw. Restart it with <code>make dev</code>, which now reloads itself when the code changes.
        </p>
      )}

      {t && meta && !behind && (
        <>
          <p className="trialline small">
            <span className={`status ${t.result.correct ? 'ok' : 'err'}`}>{t.result.correct ? 'pass' : 'fail'}</span> ·{' '}
            <span className="mono">{(t.result.agent_input_tokens + t.result.agent_output_tokens + t.result.user_input_tokens + t.result.user_output_tokens).toLocaleString('en-GB')}</span> tokens (agent {fmtK(t.result.agent_input_tokens + t.result.agent_output_tokens)}, user {fmtK(t.result.user_input_tokens + t.result.user_output_tokens)}) · {t.messages.length} messages · ended{' '}
            <span className="mono">{t.termination_reason}</span> · cost est. ${(t.result.cost_usd_est ?? 0).toFixed(3)}, paid $0 · <Link to={trialPath(meta.run_id, trialLens ?? '')}>full trace</Link>
          </p>
          <div className="agent-cols">
            <div className="graph-wrap">
              <AgentGraph t={t} agent={snap} meta={meta} node={node} onPick={pick} />
            </div>
            <NodePanel node={node} step={step} t={t} agent={snap} meta={meta} playground={!!health?.playground} pg={pg} onGo={(n) => setLens({ node: n, step: '' })} onRun={onRun} onClose={() => setLens({ node: '', step: '' })} />
          </div>
          <Replay t={t} node={node} step={step} onRow={onRow} />
        </>
      )}
      {runId && trialLens && !t && !tErr && <Loading error={null} />}
    </>
  );
}
