/**
 * The Agent tab's working parts (s05): one conversation drawn as the agent in its
 * harness, a panel that shows what any node received and produced, a playground
 * that re-runs a tool call against the database as it stood, and the replay.
 *
 * The graph is hand-laid SVG under the `.dia` contract. Its rows come from data:
 * the domain's tools (split by tau2's own read/write type), the customer's tools
 * when the conversation used any, and the task's reward basis. Nothing is drawn
 * that the trace, the task spec or the run's agent snapshot does not say.
 */

import { type KeyboardEvent, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  type DiffRecord,
  type PlaygroundResult,
  type RewardInfo,
  type RunAgent,
  type RunMeta,
  type Scenario,
  type ToolSpec,
  type TraceCall,
  type TraceMessage,
  type TrialPayload,
  domainLabel,
  fmtK,
  fmtS,
  post,
  shortModel,
} from './api';
import { optimisePath } from './url';

// ── nodes ─────────────────────────────────────────────────────────────────
/** `task`, `user`, `orch`, `agent`, `prompt`, `helper`, `model`, `reward`,
 *  `tool:<name>` (the agent's), `utool:<name>` (the customer's), `check:<id>`. */
export type NodeKey = string;

/** The eight boxes of the figure this tab drew before s05, and where each now lives,
 *  so a bookmarked `?node=` still opens something. The plumbing went into `model`. */
export const LEGACY_NODES: Record<string, NodeKey> = {
  user: 'user',
  judge: 'check:nl',
  factory: 'agent',
  generate: 'model',
  provider: 'model',
  sdk: 'model',
  prompting: 'model',
  runs: 'model',
};

export function legacyNode(key: string | null): NodeKey | null {
  if (!key) return null;
  return LEGACY_NODES[key] ?? key;
}

// ── small formatting ──────────────────────────────────────────────────────
const fmtInt = (n: number | null | undefined) => (n == null ? '—' : n.toLocaleString('en-GB'));
const plural = (n: number, w: string) => `${n} ${n === 1 ? w : /[^aeiou]y$/.test(w) ? `${w.slice(0, -1)}ies` : `${w}s`}`;
const pretty = (s: string | null) => {
  if (!s) return '';
  try {
    return JSON.stringify(JSON.parse(s), null, 1);
  } catch {
    return s;
  }
};
const argsShort = (a: Record<string, unknown>) => {
  const s = Object.entries(a)
    .map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`)
    .join(', ');
  return s.length > 90 ? `${s.slice(0, 89)}…` : s;
};
const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
const onKey = (go: () => void) => (e: KeyboardEvent) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    go();
  }
};

// ── what one conversation says ────────────────────────────────────────────
export type AgentStep = { i: number; m: TraceMessage; inputs: TraceMessage[]; earlier: number };
export type UserStep = { i: number; m: TraceMessage; inputs: TraceMessage[] };
export type CallRow = { i: number; k: number; by: 'agent' | 'user'; call: TraceCall; result: TraceMessage | null };

/** The agent's model calls. τ²'s opening greeting is a fixed string, not a model
 *  call: it has no usage, so it is not a step. */
export function agentSteps(msgs: TraceMessage[]): AgentStep[] {
  const out: AgentStep[] = [];
  let last = -1;
  msgs.forEach((m, i) => {
    if (m.role !== 'assistant') return;
    if (i === 0 && m.usage == null) {
      last = i;
      return;
    }
    out.push({ i, m, inputs: msgs.slice(last + 1, i), earlier: last + 1 });
    last = i;
  });
  return out;
}

/** The simulated customer's turns, each with the agent message it answered. */
export function userSteps(msgs: TraceMessage[]): UserStep[] {
  const out: UserStep[] = [];
  msgs.forEach((m, i) => {
    if (m.role !== 'user') return;
    let j = i - 1;
    while (j >= 0 && !(msgs[j].role === 'assistant' && msgs[j].content)) j--;
    out.push({ i, m, inputs: j >= 0 ? [msgs[j]] : [] });
  });
  return out;
}

/** Every tool call, the agent's and the customer's, with the result that answered it
 *  (paired by id, which is how τ² pairs them). */
export function toolCalls(msgs: TraceMessage[]): CallRow[] {
  const results = new Map<string, TraceMessage>();
  for (const m of msgs) if (m.role === 'tool' && m.id) results.set(m.id, m);
  const out: CallRow[] = [];
  msgs.forEach((m, i) =>
    m.tool_calls.forEach((call, k) =>
      out.push({
        i,
        k,
        by: (call.requestor || m.role) === 'user' ? 'user' : 'agent',
        call,
        result: call.id ? (results.get(call.id) ?? null) : null,
      }),
    ),
  );
  return out;
}

export function callNode(by: 'agent' | 'user', name: string): NodeKey {
  return by === 'user' ? `utool:${name}` : `tool:${name}`;
}

/** The node a replay row belongs to. */
export function nodeOfMessage(msgs: TraceMessage[], m: TraceMessage): NodeKey {
  if (m.tool_calls.length) return callNode(m.role === 'user' ? 'user' : 'agent', m.tool_calls[0].name);
  if (m.role === 'assistant') return 'agent';
  if (m.role === 'user') return 'user';
  if (m.role === 'tool') {
    const c = toolCalls(msgs).find((x) => x.call.id === m.id);
    return c ? callNode(c.by, c.call.name) : 'orch';
  }
  return 'orch';
}

export type CheckId = 'db' | 'action' | 'communicate' | 'nl' | 'env';
export type Check = { id: `check:${CheckId}`; key: string; label: string; sub: string; cls: 'ok' | 'err' | 'ext'; inBasis: boolean };

/** The five checks τ² can grade on, each coloured by this conversation's verdict and
 *  drawn dashed when the task's reward basis does not count it. */
export function checks(ri: RewardInfo | null, basis: string[]): Check[] {
  const bd = ri?.reward_breakdown ?? {};
  const verdict = (k: string, fallback: boolean | null): Check['cls'] =>
    !basis.includes(k) ? 'ext' : bd[k] != null ? (bd[k] >= 1 ? 'ok' : 'err') : fallback == null ? 'ext' : fallback ? 'ok' : 'err';
  const off = (k: string) => (basis.includes(k) ? '' : ' · off');
  const acts = ri?.action_checks ?? [];
  const made = acts.filter((a) => a.action_match).length;
  const nl = ri?.nl_assertions ?? [];
  const com = ri?.communicate_checks ?? [];
  const env = ri?.env_assertions ?? [];
  const frac = (xs: { met: boolean }[], none: string) => (xs.length ? `${xs.filter((x) => x.met).length}/${xs.length}` : none);
  const all = (xs: { met: boolean }[]) => (xs.length ? xs.every((x) => x.met) : null);
  const row = (id: CheckId, key: string, label: string, sub: string, fallback: boolean | null): Check => ({
    id: `check:${id}`,
    key,
    label,
    sub: sub + off(key),
    cls: verdict(key, fallback),
    inBasis: basis.includes(key),
  });
  return [
    row('db', 'DB', 'DB', ri?.db_check ? (ri.db_check.db_match ? '✓ match' : '✕ differs') : 'not run', ri?.db_check ? ri.db_check.db_match : null),
    row('action', 'ACTION', 'actions', acts.length ? `${made}/${acts.length}` : 'none', acts.length ? made === acts.length : null),
    row('communicate', 'COMMUNICATE', 'say', frac(com, 'none'), all(com)),
    row('nl', 'NL_ASSERTION', 'NL judge', frac(nl, 'not run'), all(nl)),
    row('env', 'ENV_ASSERTION', 'env', frac(env, 'none'), all(env)),
  ];
}

export function basisOf(t: TrialPayload): string[] {
  return t.reward_info?.reward_basis ?? t.result.reward_basis ?? t.task?.evaluation_criteria?.reward_basis ?? [];
}

/** The conversation's outcome as a claim: the first thing in the basis that failed. */
/**
 * Vite reloads the page when its code changes; an API started before s05 does not,
 * and answers a conversation with events but no messages. Everything this tab draws
 * reads the messages, so such a payload gets an explanation instead of a crash.
 */
export function apiBehind(t: unknown): boolean {
  return !!t && typeof t === 'object' && !Array.isArray((t as { messages?: unknown }).messages);
}

export function outcome(t: TrialPayload): { lead: string; em: string; tail: string } {
  // telecom's ids run past 200 unbroken characters; the full id is in the picker and the task panel
  const id = clip(`${domainLabel(t.domain)}/${t.task_id}`, 44);
  const r = t.result;
  const ri = t.reward_info;
  if (r.correct) {
    return { lead: `${id} passed: `, em: `reward 1 on ${basisOf(t).join(' × ') || 'its basis'}`, tail: ` after ${plural(r.n_agent_turns, 'agent turn')} and ${plural(r.n_tool_calls, 'tool call')}` };
  }
  if (!['agent_stop', 'user_stop'].includes(r.termination_reason)) {
    return { lead: `${id} failed: `, em: `it ended on ${r.termination_reason}`, tail: `, so nothing was graded, after ${plural(r.n_agent_turns, 'agent turn')}` };
  }
  const acts = ri?.action_checks ?? [];
  if (acts.length && acts.some((a) => !a.action_match)) {
    const made = acts.filter((a) => a.action_match).length;
    return { lead: `${id} failed: `, em: `${made} of ${acts.length} expected actions made`, tail: ri?.db_check ? `, and the database ${ri.db_check.db_match ? 'matched' : 'did not match'} gold` : '' };
  }
  if (ri?.db_check && !ri.db_check.db_match) return { lead: `${id} failed: `, em: 'the database did not match gold', tail: ` after ${plural(r.n_tool_calls, 'tool call')}` };
  const com = ri?.communicate_checks ?? [];
  if (com.some((c) => !c.met)) return { lead: `${id} failed: `, em: `${com.filter((c) => !c.met).length} of ${com.length} required facts never said`, tail: '' };
  const nl = ri?.nl_assertions ?? [];
  if (nl.some((c) => !c.met)) return { lead: `${id} failed: `, em: `the judge found ${nl.filter((c) => !c.met).length} of ${nl.length} assertions unmet`, tail: '' };
  return { lead: `${id} failed: `, em: `reward ${ri?.reward ?? r.reward}`, tail: ` after ${plural(r.n_agent_turns, 'agent turn')}` };
}

// ── the graph ─────────────────────────────────────────────────────────────
type GraphProps = { t: TrialPayload; agent: RunAgent | null; meta: RunMeta; node: NodeKey | null; onPick: (n: NodeKey) => void };

type Row = { name: string; spec: ToolSpec | null; count: number; errs: number; expected: number; matched: number };

function tally(t: TrialPayload, by: 'agent' | 'user', names: string[]): Row[] {
  const calls = toolCalls(t.messages).filter((c) => c.by === by);
  const acts = (t.reward_info?.action_checks ?? []).filter((a) => ((a.action.requestor ?? 'assistant') === 'user' ? 'user' : 'agent') === by);
  const specs = by === 'agent' ? t.tools : t.user_tools;
  return names.map((name) => ({
    name,
    spec: specs.find((s) => s.name === name) ?? null,
    count: calls.filter((c) => c.call.name === name).length,
    errs: calls.filter((c) => c.call.name === name && c.result?.error).length,
    expected: acts.filter((a) => a.action.name === name).length,
    matched: acts.filter((a) => a.action.name === name && a.action_match).length,
  }));
}

export function AgentGraph({ t, agent, meta, node, onPick }: GraphProps) {
  const W = 540;
  const msgs = t.messages;
  const calls = toolCalls(msgs);
  const aSteps = agentSteps(msgs);
  const uSteps = userSteps(msgs);
  const basis = basisOf(t);
  const ck = checks(t.reward_info, basis);

  // the agent's tools: the domain's list, plus anything the trace called that is not on it
  const listed = t.tools.map((x) => x.name);
  const strays = [...new Set(calls.filter((c) => c.by === 'agent' && !listed.includes(c.call.name)).map((c) => c.call.name))];
  const agentRows = tally(t, 'agent', [...listed, ...strays]);
  const left = agentRows.filter((r) => r.spec?.type !== 'write');
  const right = agentRows.filter((r) => r.spec?.type === 'write');

  // the customer's tools (τ²'s `env.user_tools`, beside the agent's `env.tools`): only the
  // ones this conversation used or the task expects, since telecom's customer has 30
  const expectedUser = (t.reward_info?.action_checks ?? []).filter((a) => a.action.requestor === 'user').map((a) => a.action.name);
  const usedUser = [...new Set([...calls.filter((c) => c.by === 'user').map((c) => c.call.name), ...expectedUser])];
  const hasUserSide = t.user_tools.length > 0 && (usedUser.length > 0 || (t.task?.user_tools ?? []).length > 0);
  const userRows = tally(t, 'user', usedUser);

  const toolRow = (r: Row, by: 'agent' | 'user', x: number, y: number, w: number) => {
    const key = callNode(by, r.name);
    const cls = r.expected ? (r.matched === r.expected ? 'ok' : 'miss') : r.errs ? 'err' : r.count ? '' : 'idle';
    const label = r.expected ? `${r.matched}/${r.expected}` : r.count ? `×${r.count}` : '';
    const lcls = r.expected ? (r.matched === r.expected ? 'ok' : 'warn') : r.errs ? 'warn' : '';
    const title = `${r.name} (${r.spec?.type ?? 'not in the tool list'}) — ${r.count ? plural(r.count, 'call') : 'not called'}${r.errs ? `, ${r.errs} errored` : ''}${r.expected ? `; the task expects ${r.expected}, ${r.matched} matched` : ''}. ${r.spec?.description ?? ''}`;
    return (
      <g key={key} className="pick" id={`n-${key.replace(/[^a-z0-9_-]/gi, '-')}`} role="button" tabIndex={0} aria-label={title} aria-pressed={node === key} onClick={() => onPick(key)} onKeyDown={onKey(() => onPick(key))}>
        <title>{title}</title>
        <rect className={`nd ${cls} ${node === key ? 'sel' : ''}`} x={x} y={y} width={w} height={22} rx={6} />
        <text className={`tx t ${r.count || r.expected ? '' : 'idle'}`} x={x + 8} y={y + 15.5}>
          {clip(r.name, label ? 27 : 31)}
        </text>
        {label && (
          <text className={`tx n ${lcls}`} x={x + w - 8} y={y + 15.5} textAnchor="end">
            {label}
          </text>
        )}
      </g>
    );
  };

  const box = (id: NodeKey, x: number, y: number, w: number, h: number, label: string, sub: string, cls: string, title: string) => (
    <g key={id} className="pick" id={`n-${id.replace(/[^a-z0-9_-]/gi, '-')}`} role="button" tabIndex={0} aria-label={title} aria-pressed={node === id} onClick={() => onPick(id)} onKeyDown={onKey(() => onPick(id))}>
      <title>{title}</title>
      <rect className={`nd ${cls} ${node === id ? 'sel' : ''}`} x={x} y={y} width={w} height={h} rx={8} />
      <text className="tx" x={x + 12} y={y + (sub ? 22 : h / 2 + 5)}>
        {label}
      </text>
      {sub && (
        <text className="tx s" x={x + 12} y={y + 41}>
          {sub}
        </text>
      )}
    </g>
  );

  // layout, top to bottom; every y follows from the one above it
  const rowH = 27;
  const topY = 12;
  const orchY = topY + 80;
  const grpY = orchY + 74;
  const envY = grpY + 196;
  const envRows = Math.max(left.length, right.length);
  const userKeyY = envY + 58 + envRows * rowH + 24;
  const userRowsN = Math.max(1, Math.ceil(userRows.length / 2));
  const envH = 58 + envRows * rowH + 6 + (hasUserSide ? 36 + userRowsN * rowH : 0);
  const gradeY = envY + envH + 30;
  const H = gradeY + 150;
  const helper = agent?.files['helper.py'];
  const hooksOn = Object.values(agent?.hooks ?? {}).filter(Boolean).length;
  const bd = t.reward_info?.reward_breakdown ?? {};
  const ckW = 96;
  const ckGap = (516 - 5 * ckW) / 4;
  const agentCalls = calls.filter((c) => c.by === 'agent').length;

  return (
    <svg className="dia graph agentgraph" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`The ${meta.agent} agent in one conversation: the task briefs the user simulator; the orchestrator passes messages between the simulator and the agent and runs the agent's tool calls against the ${t.domain} environment; five checks grade the result into a reward.`}>
      <defs>
        <marker id="agarr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0L10 5L0 10z" fill="currentColor" />
        </marker>
      </defs>

      {box('task', 12, topY, 252, 56, `task · ${clip(`${t.domain}/${t.task_id}`, 26)}`, 'scenario · agent never sees it', '', 'The task: the scenario the user simulator plays, and the criteria the grade uses')}
      {box('user', 276, topY, 252, 56, 'user simulator', `${shortModel(meta.user_model)} · ${plural(uSteps.length, 'turn')}`, '', 'The user simulator: plays the customer from the scenario')}
      <path className="ed" d={`M264 ${topY + 28} H274`} />

      {box('orch', 12, orchY, 516, 50, 'orchestrator · τ² half-duplex', `${msgs.length} messages · ended ${t.termination_reason}`, '', 'The orchestrator: owns the turn loop and runs every tool call')}
      <path className="ed two" d={`M402 ${topY + 58} V${orchY - 2}`} />

      <rect className="grp" x={12} y={grpY} width={516} height={172} rx={10} />
      <text className="tx k" x={24} y={grpY + 20}>
        agent · {t.domain}/{meta.agent} · {meta.fingerprint.slice(0, 7)}
      </text>
      {box('agent', 24, grpY + 30, 492, 52, 'agent turns', `${plural(aSteps.length, 'model call')} · ${fmtK(t.result.agent_input_tokens)} in · ${fmtK(t.result.agent_output_tokens)} out`, '', 'The agent: one model call per turn, each carrying the system prompt and the whole history')}
      {box('prompt', 24, grpY + 96, 152, 58, 'system prompt', agent ? `${fmtK(agent.prompt.text.length)} chars` : '…', '', 'The composed system prompt: system.md, the policy in its slot, and extra_context()')}
      {box('helper', 194, grpY + 96, 152, 58, 'helper.py', helper ? `${hooksOn} of 3 hooks` : 'none', helper ? '' : 'ext', 'helper.py: the deterministic hooks around the model')}
      {box('model', 364, grpY + 96, 152, 58, 'model', shortModel(meta.model), '', 'The model, and the one path every call takes to the subscription')}
      <path className="ed two" d={`M440 ${orchY + 52} V${grpY + 28}`} />
      <path className="ed dash" d={`M100 ${grpY + 96} V${grpY + 82}`} />
      <path className="ed dash" d={`M270 ${grpY + 96} V${grpY + 82}`} />
      <path className="ed dash" d={`M440 ${grpY + 96} V${grpY + 82}`} />

      <path className="ed two" d={`M270 ${grpY + 174} V${envY - 2}`} />
      <text className="cap" x={282} y={grpY + 189}>
        {plural(agentCalls, 'tool call')}, run by the orchestrator
      </text>

      <rect className="grp" x={12} y={envY} width={516} height={envH} rx={10} />
      <text className="tx k" x={24} y={envY + 22}>
        environment · {t.domain} · {t.tools.length} tools
      </text>
      <text className="tx k" x={24} y={envY + 46}>
        read · {left.length}
      </text>
      <text className="tx k" x={272} y={envY + 46}>
        write · {right.length}
      </text>
      {left.map((r, i) => toolRow(r, 'agent', 24, envY + 58 + i * rowH, 236))}
      {right.map((r, i) => toolRow(r, 'agent', 272, envY + 58 + i * rowH, 244))}
      {hasUserSide && (
        <>
          <text className="tx k" x={24} y={userKeyY}>
            customer's device · {t.user_tools.length} tools · {usedUser.length} used
          </text>
          {userRows.length === 0 && (
            <text className="cap" x={24} y={userKeyY + 24}>
              called by the user simulator; none in this conversation
            </text>
          )}
          {userRows.map((r, i) => toolRow(r, 'user', i % 2 ? 272 : 24, userKeyY + 12 + Math.floor(i / 2) * rowH, i % 2 ? 244 : 236))}
        </>
      )}

      <path className="ed" d={`M270 ${envY + envH + 2} V${gradeY - 2}`} />
      <text className="tx k" x={12} y={gradeY + 16}>
        grade · reward = product of the basis
      </text>
      {ck.map((c, i) => box(c.id, 12 + i * (ckW + ckGap), gradeY + 28, ckW, 54, c.label, c.sub, c.cls, `${c.label}: ${c.sub}${c.inBasis ? '' : " (not in this task's reward basis)"}`))}
      {box('reward', 12, gradeY + 96, 516, 50, `reward ${t.result.reward} · ${t.result.correct ? 'pass' : 'fail'}`, basis.map((b) => `${b} ${bd[b] ?? '—'}`).join(' × ') || 'no basis', t.result.correct ? 'ok' : 'err', "The reward: the product of the components in the task's reward basis")}
    </svg>
  );
}

// ── the node panel ────────────────────────────────────────────────────────
function Msg({ t, m, label, cls = '' }: { t: TrialPayload; m: TraceMessage; label: string; cls?: string }) {
  if (m.role === 'tool') {
    const c = toolCalls(t.messages).find((x) => x.call.id === m.id);
    return (
      <div className={`blk ${cls}`}>
        <span className="label">
          {label} · {c?.call.name ?? '?'}
          {m.error ? ' · error' : ''}
        </span>
        <pre>{pretty(m.content)}</pre>
      </div>
    );
  }
  return (
    <div className={`blk ${cls}`}>
      <span className="label">{label}</span>
      {m.tool_calls.length ? (
        m.tool_calls.map((c, k) => (
          <div key={k}>
            <p className="msg mono">{c.name}</p>
            <pre>{JSON.stringify(c.arguments, null, 1)}</pre>
          </div>
        ))
      ) : (
        <p className="msg">{m.content}</p>
      )}
    </div>
  );
}

function StepHead({ what, k, m }: { what: string; k: number; m: TraceMessage }) {
  return (
    <div className="hd">
      <span>
        {what} {k} · message {m.i}
      </span>
      <span className="n">
        {m.usage ? `${fmtInt(m.usage.prompt_tokens)} in · ${fmtInt(m.usage.completion_tokens)} out` : ''}
        {m.seconds != null ? ` · ${m.seconds.toFixed(1)} s` : ''}
      </span>
    </div>
  );
}

/** A request to prefill (and maybe run) the playground on a tool's panel. */
export type PgRequest = { node: NodeKey; at: number; args: Record<string, unknown>; after_calls: number; nonce: number };

type PanelProps = {
  node: NodeKey | null;
  step: number | null;
  t: TrialPayload;
  agent: RunAgent | null;
  meta: RunMeta;
  playground: boolean;
  pg: PgRequest | null;
  onGo: (n: NodeKey) => void;
  onRun: (r: Omit<PgRequest, 'nonce'>) => void;
  onClose: () => void;
};

export function NodePanel(props: PanelProps) {
  const { node, step } = props;
  const ref = useRef<HTMLElement>(null);
  // bring the selected step to the top of the panel; with none selected, start at the top
  useEffect(() => {
    const p = ref.current;
    if (!p) return;
    const el = step != null ? p.querySelector<HTMLElement>(`#step-${step}`) : null;
    if (!el) {
      p.scrollTop = 0;
      return;
    }
    if (p.scrollHeight > p.clientHeight + 1) p.scrollTop += el.getBoundingClientRect().top - p.getBoundingClientRect().top - 12;
    else el.scrollIntoView({ block: 'nearest' });
  }, [node, step]);
  return (
    <aside className="agent-panel" ref={ref} aria-live="polite">
      <PanelBody {...props} />
    </aside>
  );
}

function Crumbs({ where, onClose }: { where: string; onClose: () => void }) {
  return (
    <div className="crumbs">
      <span className="label">{where}</span>
      <button type="button" className="linkish" onClick={onClose}>
        close
      </button>
    </div>
  );
}

function PanelBody({ node, step, t, agent, meta, playground, pg, onGo, onRun, onClose }: PanelProps) {
  const src = `runs/${meta.run_id}/traces`;
  const r = t.result;
  if (!node) {
    return (
      <>
        <h3>Nothing selected</h3>
        <p className="small">
          Click a node in the graph, or a row in the replay. Every panel shows what that part received and what it produced in{' '}
          <b>
            {t.domain}/{t.task_id}
          </b>
          .
        </p>
        <dl className="diff-sum">
          <dt>version</dt>
          <dd>
            {meta.domain}/{meta.agent} · {meta.fingerprint}
          </dd>
          <dt>run</dt>
          <dd className="mono">{meta.run_id}</dd>
          <dt>note</dt>
          <dd>{meta.note || '—'}</dd>
          <dt>split</dt>
          <dd>
            {meta.split} · seed {meta.seed} · τ² {meta.tau2_sha.slice(0, 7)}
          </dd>
        </dl>
      </>
    );
  }

  if (node === 'task') {
    const sc = t.task?.user_scenario?.instructions;
    const acts = t.reward_info?.action_checks ?? [];
    const crit = t.task?.evaluation_criteria;
    const parts: [string, string][] =
      typeof sc === 'string'
        ? [['instructions', sc]]
        : (['reason_for_call', 'known_info', 'unknown_info', 'task_instructions'] as (keyof Scenario)[])
            .filter((k) => sc?.[k])
            .map((k) => [k.replace(/_/g, ' '), String(sc?.[k])]);
    return (
      <>
        <Crumbs where={`data/tasks/${t.domain}.json · tasks[id=${t.task_id}]`} onClose={onClose} />
        <h3>
          Task {t.domain}/{t.task_id}: what the customer wants, which the agent never sees
        </h3>
        {t.task?.description?.purpose && <p className="small">{t.task.description.purpose}</p>}
        <h4>Input to the user simulator</h4>
        <div className="io">
          {parts.length ? (
            parts.map(([k, v]) => (
              <div className="blk" key={k}>
                <span className="label">{k}</span>
                <p className="msg">{v}</p>
              </div>
            ))
          ) : (
            <p className="small muted">This task has no scenario in the extract.</p>
          )}
        </div>
        <h4>What the grade expects · basis {basisOf(t).join(' × ') || '—'}</h4>
        <div className="io">
          {acts.length ? (
            acts.map((a) => (
              <div key={a.action.action_id} className={`blk ${a.action_match ? 'okb' : 'miss'}`}>
                <span className="label">
                  {a.action_match ? '✓ made' : '✕ not made'} · {a.action.action_id}
                  {a.tool_type ? ` · ${a.tool_type}` : ''}
                  <RunIt onClick={() => onRun({ node: callNode(a.action.requestor === 'user' ? 'user' : 'agent', a.action.name), at: t.messages.length, args: a.action.arguments, after_calls: 0 })} label="▶ run it" />
                </span>
                <p className="msg mono">{a.action.name}</p>
                <pre>{JSON.stringify(a.action.arguments, null, 1)}</pre>
              </div>
            ))
          ) : (
            <p className="small muted">No expected actions: the grade rests on the other checks.</p>
          )}
        </div>
        {(crit?.communicate_info ?? []).length > 0 && (
          <>
            <p className="small">
              <b>Must be said</b> (graded when <code>COMMUNICATE</code> is in the basis):
            </p>
            <ul className="small">
              {crit!.communicate_info!.map((x) => (
                <li key={x}>{x}</li>
              ))}
            </ul>
          </>
        )}
        {(crit?.nl_assertions ?? []).length > 0 && (
          <>
            <p className="small">
              <b>NL assertions</b> (graded when <code>NL_ASSERTION</code> is in the basis):
            </p>
            <ul className="small">
              {crit!.nl_assertions!.map((x) => (
                <li key={x}>{x}</li>
              ))}
            </ul>
          </>
        )}
      </>
    );
  }

  if (node === 'user') {
    const st = userSteps(t.messages);
    return (
      <>
        <Crumbs where={`${src} · messages[role=user]`} onClose={onClose} />
        <h3>User simulator: {plural(st.length, 'turn')}, played from the task's scenario</h3>
        <p className="small">
          τ²'s own simulator, on <span className="mono">{shortModel(meta.user_model)}</span>. It gets the scenario as instructions (see{' '}
          <button type="button" className="linkish" onClick={() => onGo('task')}>
            task
          </button>
          ) and answers each agent message. It ends the conversation with <code>###STOP###</code>. This conversation used {fmtInt(r.user_input_tokens)} input and {fmtInt(r.user_output_tokens)} output tokens.
        </p>
        {st.map((s, k) => (
          <div key={s.i} id={`step-${s.i}`} className={`step ${step === s.i ? 'cur' : ''}`}>
            <StepHead what="turn" k={k + 1} m={s.m} />
            <div className="io">
              {s.inputs.map((x) => (
                <Msg key={x.i} t={t} m={x} label="input · the agent said" />
              ))}
              <Msg t={t} m={s.m} label={s.m.tool_calls.length ? 'output · on their own device' : 'output · the customer says'} cls="out" />
            </div>
          </div>
        ))}
      </>
    );
  }

  if (node === 'orch') {
    const by: Record<string, number> = {};
    t.messages.forEach((m) => {
      by[m.role] = (by[m.role] ?? 0) + 1;
    });
    return (
      <>
        <Crumbs where={`${src} · termination_reason`} onClose={onClose} />
        <h3>
          Orchestrator: {t.messages.length} messages, ended by <span className="mono">{t.termination_reason}</span>
        </h3>
        <p className="small">
          τ²'s turn loop, unmodified. Half-duplex: one party speaks per step. It passes each message to the other side, runs every tool call against the environment and returns the results. It stops on <code>###STOP###</code>, a transfer to a human, or the step cap.
        </p>
        <dl className="diff-sum">
          <dt>messages</dt>
          <dd>
            {Object.entries(by)
              .map(([k, v]) => `${k} ${v}`)
              .join(' · ')}
          </dd>
          <dt>agent turns</dt>
          <dd>{r.n_agent_turns}</dd>
          <dt>tool calls</dt>
          <dd>
            {r.n_tool_calls} · {r.n_tool_errors} errored
          </dd>
          <dt>step cap</dt>
          <dd>{String(agent?.config.max_steps ?? '—')} (agent.yaml · max_steps)</dd>
          <dt>wall clock</dt>
          <dd>{fmtS(r.duration_ms)} (includes waits for the subscription's rate window)</dd>
        </dl>
      </>
    );
  }

  if (node === 'agent') {
    const st = agentSteps(t.messages);
    const withUsage = st.filter((s) => s.m.usage);
    const secs = st.reduce((a, s) => a + (s.m.seconds ?? 0), 0);
    return (
      <>
        <Crumbs where={`${src} · messages[role=assistant]`} onClose={onClose} />
        <h3>Agent: {plural(st.length, 'model call')}, each answering with one message or tool calls</h3>
        <p className="small">
          Each call is <code>LoopAgent.generate_next_message</code>: the composed system prompt ({agent ? fmtK(agent.prompt.text.length) : '…'} chars) plus the whole history. That is why input grows every turn
          {withUsage.length > 1 ? `, from ${fmtInt(withUsage[0].m.usage?.prompt_tokens)} to ${fmtInt(withUsage[withUsage.length - 1].m.usage?.prompt_tokens)} tokens here` : ''}. Below, <b>input</b> is only what is new since the agent last spoke. In total: {fmtInt(r.agent_input_tokens)} in, {fmtInt(r.agent_output_tokens)} out, {secs.toFixed(0)} s of generation.
        </p>
        {st.map((s, k) => (
          <div key={s.i} id={`step-${s.i}`} className={`step ${step === s.i ? 'cur' : ''}`}>
            <StepHead what="call" k={k + 1} m={s.m} />
            <div className="io">
              {s.inputs.map((x) => (
                <Msg key={x.i} t={t} m={x} label={x.role === 'user' ? 'input · the customer said' : 'input · tool result'} />
              ))}
              <div className="blk">
                <span className="label">also carried</span>
                <p className="sub">system prompt + {plural(s.earlier, 'earlier message')}</p>
              </div>
              <Msg t={t} m={s.m} label={s.m.tool_calls.length ? 'output · tool call' : 'output · to the customer'} cls="out" />
            </div>
          </div>
        ))}
      </>
    );
  }

  if (node === 'prompt') {
    if (!agent) return <p className="small muted">Loading the run's agent…</p>;
    const p = agent.prompt;
    return (
      <>
        <Crumbs where={`runs/${meta.run_id}/agent/system.md · trace.policy`} onClose={onClose} />
        <h3>System prompt: {fmtInt(p.text.length)} characters, sent unchanged on every call</h3>
        <p className="small">
          <code>system.md</code> ({fmtInt(p.system_md_chars)} chars) with the domain policy ({fmtInt(p.policy_words)} words) {p.slotted ? <>in its <code>{'{policy}'}</code> slot</> : 'appended, since it has no slot'}
          {p.extra_context ? (
            <>
              , then <code>extra_context()</code>'s output ({fmtInt(p.extra_context.length)} chars)
            </>
          ) : (
            ''
          )}
          . Composed by the same function the agent uses, from the run's own snapshot. Only the optimiser edits <code>system.md</code>.
        </p>
        <details className="part" open>
          <summary>
            system.md <span className="muted">· {fmtInt(p.system_md_chars)} chars · written by the optimiser</span>
          </summary>
          <div className="blk">
            <pre className="tall">{agent.files['system.md']}</pre>
          </div>
        </details>
        <details className="part">
          <summary>
            policy <span className="muted">· {fmtInt(p.policy.length)} chars · τ²'s, as the trace recorded it</span>
          </summary>
          <div className="blk">
            <pre className="tall">{p.policy}</pre>
          </div>
        </details>
        {p.extra_context ? (
          <details className="part" open>
            <summary>
              extra_context() <span className="muted">· {fmtInt(p.extra_context.length)} chars · from helper.py</span>
            </summary>
            <div className="blk out">
              <pre>{p.extra_context}</pre>
            </div>
          </details>
        ) : (
          <p className="small muted">No helper output, so nothing is appended.</p>
        )}
      </>
    );
  }

  if (node === 'helper') {
    const h = agent?.files['helper.py'];
    if (!agent) return <p className="small muted">Loading the run's agent…</p>;
    if (!h) {
      return (
        <>
          <Crumbs where={`runs/${meta.run_id}/agent/`} onClose={onClose} />
          <h3>
            No helper.py: {meta.domain}/{meta.agent} is <code>system.md</code> alone
          </h3>
          <p className="small">
            A version may define three hooks: <code>extra_context(policy)</code>, <code>on_tool_call(name, args)</code> and <code>on_reply(text)</code>. This one defines none, so every model output reaches the orchestrator unchanged.
          </p>
        </>
      );
    }
    const hookRow = (k: string, what: string) => (
      <tr key={k}>
        <td className="mono">{k}</td>
        <td>{agent.hooks[k] ? <span className="status ok">defined</span> : <span className="status no">absent</span>}</td>
        <td className="wrap">{what}</td>
      </tr>
    );
    const nCalls = toolCalls(t.messages).filter((c) => c.by === 'agent').length;
    return (
      <>
        <Crumbs where={`runs/${meta.run_id}/agent/helper.py`} onClose={onClose} />
        <h3>helper.py: {Object.values(agent.hooks).filter(Boolean).length} of 3 hooks, all deterministic</h3>
        <div className="tw flat">
          <table>
            <thead>
              <tr>
                <th>hook</th>
                <th>here</th>
                <th>in this conversation</th>
              </tr>
            </thead>
            <tbody>
              {hookRow('extra_context', agent.hooks.extra_context ? `Ran once, when the prompt was built: ${fmtInt(agent.prompt.extra_context?.length ?? 0)} chars appended.` : 'Nothing appended.')}
              {hookRow('on_tool_call', agent.hooks.on_tool_call ? `Saw ${plural(nCalls, 'call')}; could rewrite any.` : 'Tool calls pass through unchanged.')}
              {hookRow('on_reply', agent.hooks.on_reply ? 'Saw every text reply.' : 'Text replies pass through unchanged.')}
            </tbody>
          </table>
        </div>
        <details className="part">
          <summary>
            helper.py <span className="muted">· {fmtInt(h.length)} chars</span>
          </summary>
          <div className="blk">
            <pre className="tall">{h}</pre>
          </div>
        </details>
      </>
    );
  }

  if (node === 'model') {
    return (
      <>
        <Crumbs where={`runs/${meta.run_id}/run.json`} onClose={onClose} />
        <h3>Model: every call runs on the subscription, through one choke point</h3>
        <dl className="diff-sum">
          <dt>agent</dt>
          <dd className="mono">{meta.model}</dd>
          <dt>user</dt>
          <dd className="mono">{meta.user_model}</dd>
          <dt>NL judge</dt>
          <dd className="mono">{meta.judge_model}</dd>
          <dt>sampling</dt>
          <dd>
            {meta.sampling}: the SDK exposes no temperature, so trials vary and <code>pass^k</code> is reported
          </dd>
          <dt>tool mode</dt>
          <dd>{meta.tool_mode}: tools travel as a JSON contract in the prompt, not as native tool calls</dd>
          <dt>cost est.</dt>
          <dd>${(r.cost_usd_est ?? 0).toFixed(3)} for this conversation had it been billed per token; paid $0</dd>
        </dl>
        <h4>One agent call, end to end</h4>
        <ol className="chain">
          <li>
            <code>LoopAgent.generate_next_message</code> · src/tau2_loop/agent/factory.py
          </li>
          <li>
            τ²'s <code>generate()</code>, the one function every model call goes through (the agent's, the simulator's, the judge's) · llm_utils.py
          </li>
          <li>
            litellm, with the <code>claude-sdk/</code> prefix routed to our provider
          </li>
          <li>
            the <code>claude-sdk/</code> provider · src/tau2_loop/llm/sdk_provider.py
          </li>
          <li>history → one prompt; the reply parsed back into a message or tool_calls · prompting.py</li>
          <li>
            one Agent SDK <code>query()</code> on the subscription; <code>require_live()</code> refuses a per-token key
          </li>
        </ol>
        <p className="small">
          One CLI process per model call, so a twenty-turn conversation is about twenty agent calls and twenty simulator calls. The demo image sets <code>DEMO_MODE=1</code> and cannot reach a model. The run folder is the record; MLflow indexes it
          {meta.mlflow_url ? (
            <>
              {' '}
              (<a href={meta.mlflow_url}>this run in MLflow</a>)
            </>
          ) : (
            ''
          )}
          .
        </p>
      </>
    );
  }

  if (node.startsWith('tool:') || node.startsWith('utool:')) {
    const by: 'agent' | 'user' = node.startsWith('utool:') ? 'user' : 'agent';
    const name = node.slice(node.indexOf(':') + 1);
    const spec = (by === 'agent' ? t.tools : t.user_tools).find((x) => x.name === name);
    const cs = toolCalls(t.messages).filter((c) => c.by === by && c.call.name === name);
    const exp = (t.reward_info?.action_checks ?? []).filter((a) => a.action.name === name && ((a.action.requestor ?? 'assistant') === 'user') === (by === 'user'));
    return (
      <>
        <Crumbs where={`${src} · tool_calls[name=${name}]`} onClose={onClose} />
        <h3>
          <span className="mono">{name}</span>: {cs.length ? `called ${plural(cs.length, 'time')}` : 'not called'}
          {exp.length ? `, ${exp.filter((a) => a.action_match).length} of ${exp.length} expected calls made` : ''}
        </h3>
        <p className="small">
          <span className={`chip ${spec?.type === 'write' ? 'warn' : 'no'}`}>{spec?.type ?? 'not in the tool list'}</span> {by === 'user' ? "The customer's own tool. " : ''}
          {spec?.description ?? ''}
        </p>
        {exp.length > 0 && (
          <div className="io">
            {exp.map((a) => (
              <div key={a.action.action_id} className={`blk ${a.action_match ? 'okb' : 'miss'}`}>
                <span className="label">
                  {a.action_match ? '✓ matched' : '✕ never made'} · expected by the task · {a.action.action_id}
                  <RunIt onClick={() => onRun({ node, at: t.messages.length, args: a.action.arguments, after_calls: 0 })} label="▶ run it" />
                </span>
                <pre>{JSON.stringify(a.action.arguments, null, 1)}</pre>
              </div>
            ))}
          </div>
        )}
        {cs.length ? (
          cs.map((c, k) => (
            <div key={`${c.i}-${c.k}`} id={`step-${c.i}`} className={`step ${step === c.i || (c.result && step === c.result.i) ? 'cur' : ''}`}>
              <div className="hd">
                <span>
                  call {k + 1} · message {c.i}
                </span>
                <span className="n">
                  {c.result?.error ? 'error · ' : ''}
                  <RunIt onClick={() => onRun({ node, at: c.i, args: c.call.arguments, after_calls: c.k })} label="▶ edit and re-run" />
                </span>
              </div>
              <div className="io">
                <div className="blk">
                  <span className="label">input · arguments</span>
                  <pre>{JSON.stringify(c.call.arguments, null, 1)}</pre>
                </div>
                {c.result ? (
                  <Msg t={t} m={c.result} label="output · result" cls="out" />
                ) : (
                  <div className="blk out">
                    <span className="label">output</span>
                    <p className="sub">no result recorded</p>
                  </div>
                )}
              </div>
            </div>
          ))
        ) : (
          <p className="small muted">{exp.length ? 'Never called, which is the failure.' : 'Not called in this conversation.'}</p>
        )}
        <Playground key={`${meta.run_id}|${t.task_id}|${t.result.trial}|${node}`} t={t} runId={meta.run_id} name={name} by={by} available={playground} req={pg && pg.node === node ? pg : null} />
      </>
    );
  }

  if (node.startsWith('check:')) {
    const ri = t.reward_info;
    const c = checks(ri, basisOf(t)).find((x) => x.id === node);
    if (!c) return null;
    const bdv = ri?.reward_breakdown?.[c.key];
    const head = (
      <>
        <Crumbs where={`${src} · reward_info`} onClose={onClose} />
        <h3>
          {c.label}: {c.sub.replace(' · off', '')}
          {c.inBasis ? `, in the basis (${bdv ?? '—'})` : ", not in this task's basis"}
        </h3>
      </>
    );
    if (node === 'check:db') {
      const writes = toolCalls(t.messages).filter((x) => x.by === 'agent' && t.tools.find((s) => s.name === x.call.name)?.type === 'write').length;
      const want = (ri?.action_checks ?? []).filter((a) => a.tool_type === 'write').length;
      return (
        <>
          {head}
          <p className="small">
            τ² replays the task's expected actions on a fresh copy of the database and compares that state's hash with the one this conversation left behind. The agent made {plural(writes, 'write')}; the task expects {plural(want, 'write')}.
          </p>
          <dl className="diff-sum">
            <dt>db_match</dt>
            <dd>{ri?.db_check ? String(ri.db_check.db_match) : '—'}</dd>
            <dt>db_reward</dt>
            <dd>{ri?.db_check ? ri.db_check.db_reward : '—'}</dd>
          </dl>
        </>
      );
    }
    const items: { ok: boolean; label: string; body: string; sub?: string; mono?: boolean }[] =
      node === 'check:action'
        ? (ri?.action_checks ?? []).map((a) => ({ ok: a.action_match, label: `${a.action_match ? '✓' : '✕'} ${a.action.action_id} · ${a.action.name}`, body: JSON.stringify(a.action.arguments, null, 1), mono: true }))
        : node === 'check:communicate'
          ? (ri?.communicate_checks ?? []).map((x) => ({ ok: x.met, label: x.met ? '✓ said' : '✕ not said', body: x.info }))
          : node === 'check:nl'
            ? (ri?.nl_assertions ?? []).map((x) => ({ ok: x.met, label: x.met ? '✓ met' : '✕ not met', body: x.nl_assertion, sub: x.justification }))
            : (ri?.env_assertions ?? []).map((x) => ({ ok: x.met, label: `${x.met ? '✓' : '✕'} ${x.env_assertion.func_name ?? 'assertion'}`, body: JSON.stringify(x.env_assertion.arguments ?? {}, null, 1), mono: true }));
    const why: Record<string, string> = {
      'check:action': 'Each expected call is matched on name and arguments. When ACTION is not in the basis it does not change the reward, but it is the quickest way to read a failure.',
      'check:communicate': 'Things the agent must tell the customer, matched as strings in its messages.',
      'check:nl': `The only check with a model in it: ${shortModel(meta.judge_model)} reads the transcript and decides whether each assertion holds. A judge can be wrong; the Review tab records a person's verdict.`,
      'check:env': "Assertions about the environment's state after the conversation, run as Python functions.",
    };
    return (
      <>
        {head}
        <p className="small">{why[node]}</p>
        <div className="io">
          {items.length ? (
            items.map((x, i) => (
              <div key={i} className={`blk ${x.ok ? 'okb' : 'miss'}`}>
                <span className="label">{x.label}</span>
                {x.mono ? <pre>{x.body}</pre> : <p className="msg">{x.body}</p>}
                {x.sub && <p className="sub">{x.sub}</p>}
              </div>
            ))
          ) : (
            <p className="small muted">Nothing to check for this task.</p>
          )}
        </div>
      </>
    );
  }

  if (node === 'reward') {
    const bd = t.reward_info?.reward_breakdown ?? {};
    const basis = basisOf(t);
    return (
      <>
        <Crumbs where={`runs/${meta.run_id}/results.jsonl · ${t.task_id}`} onClose={onClose} />
        <h3>
          Reward {r.reward}: {basis.map((b) => `${b} ${bd[b] ?? '—'}`).join(' × ') || 'no basis'}
        </h3>
        <p className="small">
          τ² multiplies the components named in the task's <code>reward_basis</code>, so one zero fails the conversation. A reward of 1 is a pass, and that is what <code>pass^k</code> and the gate count.
        </p>
        <dl className="diff-sum">
          {basis.map((b) => [<dt key={`${b}t`}>{b}</dt>, <dd key={`${b}d`}>{bd[b] ?? '—'}</dd>])}
          <dt>verdict</dt>
          <dd>
            <span className={`status ${r.correct ? 'ok' : 'err'}`}>{r.correct ? 'pass' : 'fail'}</span>
          </dd>
        </dl>
      </>
    );
  }
  return <p className="small muted">No node called {node} in this graph.</p>;
}

function RunIt({ onClick, label }: { onClick: () => void; label: string }) {
  return (
    <button type="button" className="linkish runit" onClick={onClick}>
      {label}
    </button>
  );
}

// ── the playground ────────────────────────────────────────────────────────
type PgProps = { t: TrialPayload; runId: string; name: string; by: 'agent' | 'user'; available: boolean; req: PgRequest | null };

/** One tool call against the database as it stood at a chosen message. The route
 *  rebuilds τ²'s environment in memory for each call and drops it: nothing is saved,
 *  and no model is involved. */
function Playground({ t, runId, name, by, available, req }: PgProps) {
  const calls = toolCalls(t.messages);
  const initial = (): { at: number; args: string; after: number } => {
    const miss = (t.reward_info?.action_checks ?? []).find((a) => a.action.name === name && !a.action_match);
    if (miss) return { at: t.messages.length, args: JSON.stringify(miss.action.arguments, null, 1), after: 0 };
    const mine = calls.filter((c) => c.by === by && c.call.name === name);
    const last = mine[mine.length - 1];
    if (last) return { at: last.i, args: JSON.stringify(last.call.arguments, null, 1), after: last.k };
    return { at: t.messages.length, args: '{}', after: 0 };
  };
  const [form, setForm] = useState(initial);
  const [res, setRes] = useState<PlaygroundResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const run = async (f = form) => {
    setErr(null);
    let args: Record<string, unknown>;
    try {
      args = JSON.parse(f.args) as Record<string, unknown>;
    } catch (e) {
      setRes(null);
      setErr(`The arguments are not valid JSON: ${(e as Error).message}`);
      return;
    }
    setBusy(true);
    try {
      const url = `/api/runs/${encodeURIComponent(runId)}/${encodeURIComponent(t.task_id)}/t${t.result.trial}/tool`;
      setRes(await post<PlaygroundResult>(url, { name, arguments: args, at: f.at, after_calls: f.after, requestor: by === 'user' ? 'user' : 'assistant' }));
    } catch (e) {
      setRes(null);
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  // a "run it" button elsewhere in the panel fills the form and runs it
  useEffect(() => {
    if (!req) return;
    const f = { at: req.at, args: JSON.stringify(req.args, null, 1), after: req.after_calls };
    setForm(f);
    if (available) void run(f);
    ref.current?.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }, [req?.nonce]);

  const points = [...new Set(calls.map((c) => c.i))];
  const label = (i: number) => `message ${i} · before ${t.messages[i].tool_calls.map((c) => c.name).join(', ')}`;
  return (
    <div className="pg" ref={ref} id="playground">
      <h4>
        Playground <span className="label">run {name} yourself</span>
      </h4>
      <p className="small">
        One call against the database as it stood at the chosen message: the task's initial state, then every earlier write replayed through τ²'s own <code>set_state</code>. No model is called and nothing is saved.
      </p>
      {!available && (
        <p className="small warn-note">
          The playground needs τ², which the demo image does not ship. Run the viewer from a checkout (<code>make dev</code>) to use it.
        </p>
      )}
      <fieldset disabled={!available || busy}>
        <div className="row">
          <label className="label" htmlFor="pg-at">
            database as of
          </label>
          <select id="pg-at" value={form.at} onChange={(e) => setForm({ ...form, at: Number(e.target.value), after: 0 })}>
            {points.map((i) => (
              <option key={i} value={i}>
                {label(i)}
              </option>
            ))}
            <option value={t.messages.length}>message {t.messages.length} · after the conversation</option>
          </select>
        </div>
        <textarea id="pg-args" className="mono" spellCheck={false} aria-label="arguments, as JSON" value={form.args} onChange={(e) => setForm({ ...form, args: e.target.value })} />
        <div className="row">
          <button type="button" className="btn" onClick={() => void run()}>
            {busy ? 'Running…' : 'Run'}
          </button>
          <span className="small muted mono">
            POST …/{clip(t.task_id, 24)}/t{t.result.trial}/tool · at {form.at}
          </span>
        </div>
      </fieldset>
      {err && (
        <div className="blk miss">
          <span className="label">not run</span>
          <p className="msg">{err}</p>
        </div>
      )}
      {res && <PgResult res={res} t={t} />}
    </div>
  );
}

function PgResult({ res, t }: { res: PlaygroundResult; t: TrialPayload }) {
  const tag = res.same_as_recorded == null ? '' : res.same_as_recorded ? ' · same as recorded ✓' : ' · differs from recorded';
  const result = (
    <div className={`blk ${res.error ? 'miss' : 'out'}`}>
      <span className="label">
        output · {res.error ? 'error' : 'result'}
        {tag}
      </span>
      <pre>{pretty(res.content)}</pre>
    </div>
  );
  const spec = [...t.tools, ...t.user_tools].find((s) => s.name === res.name);
  const db = res.wrote ? (
    <div className="blk">
      <span className="label">database · {plural(res.diff.length, 'record')} changed · in memory only</span>
      <Diff diff={res.diff} />
    </div>
  ) : (
    <div className="blk">
      <span className="label">database · unchanged</span>
      <p className="sub">{res.error ? 'The call raised before writing anything.' : spec?.mutates ? 'The call returned without changing a record.' : 'A read: nothing to change.'}</p>
    </div>
  );
  return (
    <div className="io pg-out">
      {/* a write's diff says more than the record it returns, so it goes first */}
      {res.wrote ? (
        <>
          {db}
          {result}
        </>
      ) : (
        <>
          {result}
          {db}
        </>
      )}
      <p className="sub">
        Replayed {plural(res.replayed_writes, 'earlier write')} to reach message {res.at}. A tool enforces only what its own code checks, never the policy, so an accepted call does not mean the agent was allowed to make it.
      </p>
    </div>
  );
}

function Diff({ diff }: { diff: DiffRecord[] }) {
  return (
    <div className="dl">
      {diff.map((d) => (
        <div key={d.record}>
          <div className="rec">{d.record}</div>
          {d.fields.map((f) => (
            <div className="fld" key={f.field}>
              <div className="f">{f.field}</div>
              <div className="b">{f.before ?? '(absent)'}</div>
              <div className="a">{f.after ?? '(removed)'}</div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

// ── the replay ────────────────────────────────────────────────────────────
const FIRST_ROWS = 60;

export function Replay({ t, node, step, onRow }: { t: TrialPayload; node: NodeKey | null; step: number | null; onRow: (n: NodeKey, i: number) => void }) {
  const [onlyTools, setOnlyTools] = useState(false);
  const [all, setAll] = useState(false);
  const msgs = t.messages;
  const calls = toolCalls(msgs);
  const rows = msgs.map((m) => ({ m, node: nodeOfMessage(msgs, m) })).filter(({ m }) => (onlyTools ? m.tool_calls.length > 0 || m.role === 'tool' : true));
  const shown = all ? rows : rows.slice(0, FIRST_ROWS);
  const who = (m: TraceMessage) => (m.role === 'assistant' ? 'agent' : m.role === 'user' ? 'user' : m.role === 'tool' ? 'tool' : 'system');
  const mine = (n: NodeKey, m: TraceMessage) => node != null && (n === node || (node === 'agent' && m.role === 'assistant') || (node === 'user' && m.role === 'user'));
  const errored = calls.filter((c) => c.result?.error).length;
  return (
    <>
      <h2>
        Replay — {msgs.length} messages, {plural(calls.length, 'tool call')}, {errored} errored
      </h2>
      <div className="filters">
        <button type="button" className={`tog ${onlyTools ? '' : 'on'}`} aria-pressed={!onlyTools} onClick={() => setOnlyTools(false)}>
          every message
        </button>
        <button type="button" className={`tog ${onlyTools ? 'on' : ''}`} aria-pressed={onlyTools} onClick={() => setOnlyTools(true)}>
          tool calls only
        </button>
        <span className="count">a row selects its node · the selected node's rows are marked</span>
      </div>
      <div className="tw replay">
        <table>
          <thead>
            <tr>
              <th className="num">#</th>
              <th>who</th>
              <th>message</th>
              <th className="num">tokens in / out</th>
              <th className="num">s</th>
            </tr>
          </thead>
          <tbody>
            {shown.map(({ m, node: n }) => (
              <tr key={m.i} className={`click ${mine(n, m) ? 'mine' : ''} ${step === m.i ? 'cur' : ''} ${m.error ? 'err' : ''}`} tabIndex={0} onClick={() => onRow(n, m.i)} onKeyDown={onKey(() => onRow(n, m.i))}>
                <td className="num">{m.i}</td>
                <td className={`who ${who(m)}`}>{who(m)}</td>
                <td className="msgc">
                  {m.tool_calls.length ? (
                    m.tool_calls.map((c, k) => (
                      <div key={k} className="call">
                        → {c.name}({argsShort(c.arguments)})
                      </div>
                    ))
                  ) : m.role === 'tool' ? (
                    <span className="cell">
                      {calls.find((c) => c.call.id === m.id)?.call.name ?? ''} ← {(m.content ?? '').slice(0, 260)}
                    </span>
                  ) : (
                    <span className="cell">{m.content}</span>
                  )}
                </td>
                <td className="num">{m.usage ? `${fmtInt(m.usage.prompt_tokens)} / ${fmtInt(m.usage.completion_tokens)}` : '—'}</td>
                <td className="num">{m.seconds != null ? m.seconds.toFixed(1) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > FIRST_ROWS && (
        <p className="small">
          <button type="button" className="linkbtn" onClick={() => setAll((v) => !v)}>
            {all ? `show the first ${FIRST_ROWS}` : `show all ${rows.length} rows`}
          </button>
        </p>
      )}
    </>
  );
}

/** Where "why this version exists" now lives (s05 Q4: the versions table is gone). */
export function RoundLink({ domain, name }: { domain: string; name: string }) {
  return <Link to={optimisePath(domain, name)}>why {name} exists</Link>;
}
