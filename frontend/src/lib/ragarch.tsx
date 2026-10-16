/**
 * workflow_rag's version view on the Agent tab: what the answering agent's s13 §2 draws for a vN,
 * drawn for an rN. One version from its folder (`rag_agents/<domain>/<rN>/`) with every layer that
 * differs from its parent marked, a panel for whichever part is picked, and every version side by
 * side. A version's parent is `based_on` in its change record (r5 on), else its seed (r2–r4), else
 * none (r1). When a session of the version is open, the figure carries that session's counts too.
 *
 * Everything here reads `GET /api/workflows/<d>/rag` (each version's settings, prompts, tools and
 * library), `GET /api/agents` (which answering versions read it) and `GET /api/versions` (their
 * runs). Nothing is drawn that those do not say.
 */

import { type KeyboardEvent, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { type AgentInfo, type HRun, type RagEvent, type RagSession, type RagVersion, type VersionNode, domainLabel, fmtS } from './api';
import { modelName } from './architecture';
import { agentPath, runPath } from './url';

export type RagLayerKey = 'model' | 'tool_mode' | 'retrieval' | 'tool_docs' | 'research' | 'turns' | 'seed' | 'merge' | 'decide' | 'merge_md' | 'merge_write' | 'rubric';
export type RagLayer = { key: RagLayerKey; label: string; value: string; detail?: string; cmp: string };

/** The three prompts a version may hold, one per kind of turn, with the layer each one is. */
export const RAG_FILES: { file: string; layer: RagLayerKey; turn: string; text: (v: RagVersion) => string }[] = [
  { file: 'rag_agent.md', layer: 'research', turn: 'research', text: (v) => v.prompt },
  { file: 'decide.md', layer: 'decide', turn: 'decide', text: (v) => v.decide_prompt ?? '' },
  { file: 'merge.md', layer: 'merge_md', turn: 'merge', text: (v) => v.merge_prompt ?? '' },
];
/** The merge gate every merging version has (rubric.py `gate()`), before a version's own checks. */
const GATE = ['D1', 'D8'];
export const RUBRIC: Record<string, string> = {
  D1: 'grounded: every quote is found in the document it cites',
  D8: 'nothing lost: a merge keeps every step, tool and quote of the version it merged, or its changelog names the change',
  D9: 'parameters: every write step names every required parameter of its tool',
  D10: 'statuses: a write a document allows only in some statuses is not made on a card the job can meet in another',
  D12: 're-read: a merge accounts for every step and rule of the versions it merged (kept, rewritten or removed)',
  D13: 'pointers: a job that calls or rules out a tool whose description points to the knowledge base cites a document naming that tool',
};

const num = (n: string) => Number(n.replace(/\D/g, '')) || 0;
const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, Math.max(1, n - 1))}…` : s);
const onKey = (go: () => void) => (e: KeyboardEvent) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    go();
  }
};

export function ragParent(v: RagVersion, all: RagVersion[]): RagVersion | null {
  const p = v.based_on ?? v.seed ?? null;
  return p && p !== v.name ? (all.find((x) => x.name === p) ?? null) : null;
}

/** The earliest version (by number) whose `file` has these bytes: a copied prompt reads as its source's. */
export function fileOrigin(v: RagVersion, file: string, all: RagVersion[]): string | null {
  const sha = v.files?.[file]?.sha;
  if (!sha) return null;
  return [...all].sort((a, b) => num(a.name) - num(b.name)).find((x) => x.files?.[file]?.sha === sha)?.name ?? v.name;
}

function fileValue(v: RagVersion, file: string, all: RagVersion[]): string {
  const o = fileOrigin(v, file, all);
  if (!o) return v.files ? 'none' : '—';
  return o === v.name ? `its own · ${(v.files?.[file]?.chars ?? 0).toLocaleString('en-GB')} chars` : `${o}'s, byte for byte`;
}

export const knowledgeTools = (v: RagVersion) => v.tools.filter((t) => !t.harness).map((t) => t.name);
const toolModeText = (m: string) => (m === 'native_parallel' ? 'native · every call of a reply runs' : m);
const mergeText = (v: RagVersion) =>
  v.merge === 'concurrent'
    ? `concurrent · ${v.workers ?? 1} workers · ${v.order === 'shuffled' ? `shuffled (seed ${v.order_seed ?? 0})` : 'in id order'}`
    : v.merge === 'sequential'
      ? 'sequential · one question at a time, in id order'
      : 'none: a job replaces its namesake';
const docsText = (v: RagVersion) =>
  v.tool_docs ? `tool_description · ${v.describe_named ? 'every tool a job names' : 'every tool a job calls'} described in decide and merge` : 'none';

/** The layers a version is compared on, in the table's order. */
export function ragLayers(v: RagVersion, all: RagVersion[]): RagLayer[] {
  const merges = !!v.merge;
  const kt = knowledgeTools(v);
  const parentName = v.based_on ?? v.seed ?? '';
  return [
    { key: 'model', label: 'model · effort', value: `${modelName(v.model)} · ${v.effort}`, cmp: `${v.model}|${v.effort}` },
    { key: 'tool_mode', label: 'tool calls', value: toolModeText(v.tool_mode), cmp: v.tool_mode },
    { key: 'retrieval', label: 'knowledge tools', value: `${v.retrieval}: ${kt.join(', ') || 'not readable here'}`, cmp: `${v.retrieval}|${kt.join(',')}` },
    { key: 'tool_docs', label: 'tool descriptions', value: docsText(v), cmp: `${!!v.tool_docs}|${!!v.describe_named}` },
    { key: 'research', label: 'rag_agent.md · research', value: fileValue(v, 'rag_agent.md', all), cmp: v.files?.['rag_agent.md']?.sha ?? v.prompt },
    {
      key: 'turns',
      label: 'turn caps',
      value: `research ${v.max_steps}${merges ? ` · merge ${v.merge_steps ?? 12}` : ''}`,
      detail: `a result over ${v.result_chars.toLocaleString('en-GB')} chars is cut`,
      cmp: `${v.max_steps}|${merges ? (v.merge_steps ?? 12) : ''}|${v.result_chars}`,
    },
    {
      // compared by `seedChanged`, not `cmp`: whether it starts from its parent's library
      key: 'seed',
      label: 'seed library',
      value: v.seed ? `${v.seed}'s` : 'none: a fresh library',
      detail: [v.seed && v.seed !== parentName ? `not its parent's (${parentName})` : '', v.queue === 'after_seed' ? "the seed's questions are not researched again" : ''].filter(Boolean).join(' · ') || undefined,
      cmp: `${!v.seed ? '' : v.seed === parentName ? 'parent' : v.seed}|${v.queue ?? ''}`,
    },
    { key: 'merge', label: 'merge', value: mergeText(v), cmp: `${v.merge ?? ''}|${v.workers ?? 1}|${v.order ?? ''}|${v.order_seed ?? 0}` },
    { key: 'decide', label: 'decide.md · decide', value: fileValue(v, 'decide.md', all), cmp: v.files?.['decide.md']?.sha ?? '' },
    { key: 'merge_md', label: 'merge.md · merge', value: fileValue(v, 'merge.md', all), cmp: v.files?.['merge.md']?.sha ?? '' },
    {
      key: 'merge_write',
      label: 'merge written as',
      value: merges ? (v.merge_write === 'edits' ? 'edits on the newest version' : 'whole jobs') : 'none',
      detail: merges && v.keep_tools ? 'write turns keep the tools listed' : undefined,
      cmp: merges ? `${v.merge_write ?? 'full'}|${!!v.keep_tools}` : '',
    },
    { key: 'rubric', label: 'merge gate', value: merges ? [...GATE, ...(v.rubric ?? [])].join(' · ') : 'none', cmp: merges ? (v.rubric ?? []).join(',') : '-' },
  ];
}

/** Every rN from r2 starts from the newest library, its parent's, so the seed is marked only when
 *  a version does not (r5 started from r3's: r4 never ran), gains one, or skips other questions. */
const seedChanged = (v: RagVersion, parent: RagVersion) =>
  !!v.seed !== !!parent.seed || (!!v.seed && v.seed !== parent.name) || (v.queue ?? '') !== (parent.queue ?? '');

export function ragChanged(v: RagVersion, parent: RagVersion | null, all: RagVersion[]): Set<RagLayerKey> {
  if (!parent) return new Set();
  const p = new Map(ragLayers(parent, all).map((l) => [l.key, l.cmp]));
  return new Set(ragLayers(v, all).filter((l) => (l.key === 'seed' ? seedChanged(v, parent) : p.get(l.key) !== l.cmp)).map((l) => l.key));
}

/** The assertion for a version's section: what changed from its parent, with the denominator. */
export function ragClaim(v: RagVersion, all: RagVersion[]): string {
  const parent = ragParent(v, all);
  if (!parent) return `${v.name} is the first version: nothing to compare`;
  const n = ragLayers(v, all).length;
  const ch = ragChanged(v, parent, all);
  return ch.size ? `${v.name} differs from ${parent.name} in ${ch.size} of ${n} layers` : `${v.name} matches ${parent.name} in all ${n} layers`;
}

/** The answering versions whose workflow tools read this version's library. */
export function readersOf(v: RagVersion, agents: AgentInfo[]): AgentInfo[] {
  return agents.filter((a) => (a.config as { workflows?: string | null }).workflows === v.name).sort((a, b) => num(a.name) - num(b.name));
}

// ── one session's counts ──────────────────────────────────────────────────
export type SessionCounts = {
  task: string;
  research: number;
  mergeTurns: number;
  calls: Map<string, RagEvent[]>;
  decided: number | null;
  waitedMs: number | null;
  sentBack: string[];
  taken: string[];
  rejected: string[];
  committed: Record<string, number> | null;
  merged: boolean;
};

/** What a session did at each part of the figure: its research turns (the model turns before the
 *  merge began), its merge turns, each tool's calls, and the decide, lock, gate and commit events. */
export function sessionCounts(s: RagSession): SessionCounts {
  const ev = s.events;
  const mergeAt = ev.find((e) => e.kind === 'merge')?.t ?? Number.POSITIVE_INFINITY;
  const models = ev.filter((e) => e.kind === 'model');
  const calls = new Map<string, RagEvent[]>();
  for (const e of ev) if (e.kind === 'tool' && e.name) calls.set(e.name, [...(calls.get(e.name) ?? []), e]);
  const dec = ev.find((e) => e.kind === 'decided');
  const lock = ev.find((e) => e.kind === 'locked');
  const done = ev.find((e) => e.kind === 'merged');
  const com = ev.find((e) => e.kind === 'committed');
  return {
    task: s.meta.task_id,
    research: models.filter((e) => e.t < mergeAt).length,
    mergeTurns: models.filter((e) => e.t >= mergeAt).length,
    calls,
    decided: dec ? Object.keys(dec.decision ?? {}).length : null,
    waitedMs: lock?.waited_ms ?? null,
    sentBack: ev.filter((e) => e.kind === 'note' && /failed its checks/i.test(e.text ?? '')).map((e) => e.text ?? ''),
    taken: done?.taken ?? [],
    rejected: done?.rejected ?? [],
    committed: com?.versions ?? null,
    merged: mergeAt !== Number.POSITIVE_INFINITY,
  };
}

// ── the figure ────────────────────────────────────────────────────────────
export type RagNode = string; // input · seed · prompts · file:<f> · model · tool_mode · turns · merge · env · tool:<n> · gate · library · readers

type FigProps = { domain: string; v: RagVersion; all: RagVersion[]; readers: AgentInfo[]; s: RagSession | null; node: RagNode | null; onPick: (n: RagNode) => void };

/** One workflow_rag version as its folder draws it: what it reads, the agent (its prompts, one per
 *  turn, model, tool calls, turn caps and merge), the read-only tools it calls, then the merge gate,
 *  the library it writes and the answering versions that read it. */
export function RagFigure({ domain, v, all, readers, s, node, onPick }: FigProps) {
  const W = 540;
  const parent = ragParent(v, all);
  const ch = ragChanged(v, parent, all);
  const ne = parent ? `≠ ${parent.name}` : '';
  const sc = s && s.meta.agent === v.name ? sessionCounts(s) : null;
  const seedLib = v.seed ? all.find((x) => x.name === v.seed)?.library?.workflows : undefined;
  const files = RAG_FILES.filter((f) => (v.files ? !!v.files[f.file] : f.file === 'rag_agent.md'));
  const parentTools = new Set(parent?.tools.map((t) => t.name) ?? []);

  // a box: label, sub, `≠ parent` top right, a session count bottom right
  const box = (id: RagNode, x: number, y: number, w: number, label: string, sub: string, o: { hi?: boolean; ext?: boolean; n?: string; tone?: 'ok' | 'warn'; title: string }) => {
    const subMax = Math.floor((w - 24 - (o.n ? o.n.length * 7.8 + 8 : 0)) / 6.9);
    return (
      <g key={id} className="pick" id={`rn-${id.replace(/[^a-z0-9_-]/gi, '-')}`} role="button" tabIndex={0} aria-label={o.title} aria-pressed={node === id} onClick={() => onPick(id)} onKeyDown={onKey(() => onPick(id))}>
        <title>{`${o.title}${o.hi ? ` — differs from ${parent?.name}` : ''}`}</title>
        <rect className={`nd ${o.hi ? 'hi' : o.ext ? 'ext' : ''} ${node === id ? 'sel' : ''}`} x={x} y={y} width={w} height={50} rx={8} />
        <text className="tx" x={x + 12} y={y + 22}>
          {clip(label, Math.floor((w - 24 - (o.hi ? ne.length * 8 + 6 : 0)) / 7.6))}
        </text>
        <text className="tx s" x={x + 12} y={y + 41}>
          {clip(sub, subMax)}
        </text>
        {o.hi && (
          <text className="tx n hi" x={x + w - 10} y={y + 22} textAnchor="end">
            {ne}
          </text>
        )}
        {o.n && (
          <text className={`tx n ${o.tone ?? ''}`} x={x + w - 10} y={y + 41} textAnchor="end">
            {o.n}
          </text>
        )}
      </g>
    );
  };
  // a row: one line, the mark (`≠ parent` or `new`) and a session count at the right
  const row = (id: RagNode, x: number, y: number, w: number, label: string, o: { hi?: boolean; mark?: string; idle?: boolean; n?: string; title: string }) => {
    const mark = o.hi ? (o.mark ?? ne) : '';
    const right = [o.n, mark].filter(Boolean).join(' ');
    return (
      <g key={id} className="pick" id={`rn-${id.replace(/[^a-z0-9_-]/gi, '-')}`} role="button" tabIndex={0} aria-label={o.title} aria-pressed={node === id} onClick={() => onPick(id)} onKeyDown={onKey(() => onPick(id))}>
        <title>{`${o.title}${o.hi ? (mark === 'new' ? ` — new since ${parent?.name}` : ` — differs from ${parent?.name}`) : ''}`}</title>
        <rect className={`nd ${o.hi ? 'hi' : o.idle ? 'idle' : ''} ${node === id ? 'sel' : ''}`} x={x} y={y} width={w} height={22} rx={6} />
        <text className={`tx t ${o.idle ? 'idle' : ''}`} x={x + 8} y={y + 15.5}>
          {clip(label, Math.floor((w - 16 - (right ? right.length * 7.8 + 6 : 0)) / 6.8))}
        </text>
        {right && (
          <text className={`tx n ${o.hi ? 'hi' : ''}`} x={x + w - 8} y={y + 15.5} textAnchor="end">
            {right}
          </text>
        )}
      </g>
    );
  };

  // layout, top to bottom; every y follows from the one above it
  const rowH = 27;
  const topY = 12;
  const grpY = topY + 50 + 28;
  const leftH = 30 + 50 + 10 + files.length * rowH;
  const rightH = 30 + 4 * 62 - 12;
  const grpH = Math.max(leftH, rightH) + 14;
  const envY = grpY + grpH + 44;
  const envH = 58 + v.tools.length * rowH + 52;
  const chainH = 3 * 50 + 2 * 28;
  const H = envY + Math.max(envH, chainH) + 12;
  const dl = domainLabel(domain);
  const merges = !!v.merge;
  const lib = v.library;
  const gateN = sc?.merged
    ? sc.rejected.length
      ? { n: '✕ kept old', tone: 'warn' as const }
      : sc.taken.length
        ? { n: sc.sentBack.length ? '✓ on resend' : '✓ passed', tone: 'ok' as const }
        : { n: '' }
    : { n: '' };

  return (
    <svg
      className="dia graph agentgraph archgraph raggraph"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`${dl} workflow_rag ${v.name} drawn from its folder: ${modelName(v.model)} at ${v.effort} effort, ${files.length} ${files.length === 1 ? 'prompt' : 'prompts'}, ${mergeText(v)}${parent ? `; ${ch.size} of its layers differ from ${parent.name}` : ''}${sc ? `; counts from session ${sc.task}` : ''}.`}
    >
      <defs>
        <marker id="ragarr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0L10 5L0 10z" fill="currentColor" />
        </marker>
      </defs>

      {box('input', 12, topY, 252, 'input · the customer’s script', 'never gold, never an outcome', { n: sc?.task, title: "What a session reads about its question: the customer's whole script (input: script)" })}
      {box('seed', 276, topY, 252, v.seed ? `seed · ${v.seed}'s library` : 'seed · none', v.seed ? `${seedLib ?? '—'} workflows${v.queue === 'after_seed' ? ' · not re-researched' : ''}` : 'writes a fresh library', {
        hi: ch.has('seed'),
        ext: !v.seed,
        title: v.seed ? `The library this version starts from: ${v.seed}'s` : 'No seed: the library is the newest session per job',
      })}
      <path className="ed rag" d={`M138 ${topY + 52} V${grpY - 2}`} />
      <path className="ed rag" d={`M402 ${topY + 52} V${grpY - 2}`} />

      <rect className="grp" x={12} y={grpY} width={516} height={grpH} rx={10} />
      <text className="tx k" x={24} y={grpY + 20}>
        agent · {dl}/workflow_rag/{v.name}
      </text>
      {box('prompts', 24, grpY + 30, 236, 'prompts', files.map((f) => f.turn).join(' · ') + (files.length === 1 ? ' turns' : ''), {
        hi: ch.has('research') || ch.has('decide') || ch.has('merge_md'),
        title: 'The prompts, one per kind of turn: research and draft, decide which library jobs to write, merge',
      })}
      {files.map((f, i) => {
        const o = fileOrigin(v, f.file, all);
        const n = sc ? (f.layer === 'research' ? `${sc.research} turns` : f.layer === 'decide' ? (sc.decided != null ? `${sc.decided} jobs` : '') : sc.merged ? `${sc.mergeTurns} turns` : '') : '';
        // with a session's counts at the right, the row names the file alone; its origin is in the panel
        const label = sc ? f.file : `${f.file} · ${!o || o === v.name ? 'its own' : `${o}'s`}`;
        return row(`file:${f.file}`, 36, grpY + 30 + 60 + i * rowH, 224, label, {
          hi: ch.has(f.layer),
          mark: parent && !parent.files?.[f.file] ? 'new' : ne,
          n: n || undefined,
          title: `${f.file}: the ${f.turn} turn's prompt${f.layer !== 'decide' && sc ? ` (${f.layer === 'research' ? sc.research : sc.mergeTurns} turns in this session)` : ''}`,
        });
      })}
      {box('model', 272, grpY + 30, 244, 'model', `${modelName(v.model)} · ${v.effort} effort`, { hi: ch.has('model'), title: 'The model and its effort, frozen in rag_agent.yaml; a session may override them' })}
      {box('tool_mode', 272, grpY + 92, 244, 'tool calls', v.tool_mode === 'native_parallel' ? 'native · every call runs' : v.tool_mode, { hi: ch.has('tool_mode'), n: sc ? `${[...sc.calls.values()].reduce((a, x) => a + x.length, 0)}` : undefined, title: 'How the model calls its tools: natively, and every call of a reply runs' })}
      {box('turns', 272, grpY + 154, 244, 'turn caps', `research ${v.max_steps}${merges ? ` · merge ${v.merge_steps ?? 12}` : ''}`, { hi: ch.has('turns'), title: `Model turns before it must write what it has; a tool result over ${v.result_chars.toLocaleString('en-GB')} characters is cut` })}
      {box(
        'merge',
        272,
        grpY + 216,
        244,
        'merge',
        v.merge === 'concurrent' ? `${v.workers ?? 1} workers · ${v.merge_write === 'edits' ? 'as edits' : 'whole jobs'}` : v.merge === 'sequential' ? 'one at a time · whole jobs' : 'none',
        {
          hi: ch.has('merge') || ch.has('merge_write'),
          ext: !merges,
          n: sc?.waitedMs != null ? `waited ${fmtS(sc.waitedMs)}` : undefined,
          title: `How a draft joins the library: ${mergeText(v)}`,
        },
      )}

      <path className="ed two rag" d={`M148 ${grpY + grpH + 2} V${envY - 2}`} />
      <text className="cap" x={160} y={grpY + grpH + 27}>
        run by the harness
      </text>
      <path className="ed rag" d={`M412 ${grpY + grpH + 2} V${envY - 2}`} />
      <text className="cap" x={424} y={grpY + grpH + 27}>
        {merges ? 'its merge' : 'its draft'}
      </text>

      <rect className="grp" x={12} y={envY} width={272} height={envH} rx={10} />
      <text className="tx k" x={24} y={envY + 22}>
        tools · read only
      </text>
      <text className={`tx k ${ch.has('retrieval') ? 'hi' : ''}`} x={24} y={envY + 46}>
        {clip(v.retrieval, 24)}
        {ch.has('retrieval') && <tspan className="mk">{` · ${ne}`}</tspan>}
      </text>
      {v.tools.map((t, i) => {
        const n = sc?.calls.get(t.name)?.length;
        return row(`tool:${t.name}`, 24, envY + 58 + i * rowH, 248, t.harness ? `${t.name} · harness` : t.name, {
          hi: (!!parent && !parentTools.has(t.name)) || (!!t.harness && ch.has('tool_docs')),
          mark: parent && !parentTools.has(t.name) ? 'new' : ne,
          idle: !!sc && !n,
          n: n ? `×${n}` : undefined,
          title: `${t.name}: ${t.harness ? "workflow_rag's own research tool" : `a knowledge tool the ${v.retrieval} retrieval gives`}. ${t.description ?? ''}`,
        });
      })}
      <text className="cap" x={24} y={envY + 58 + v.tools.length * rowH + 18}>
        the answering agent’s tools and
      </text>
      <text className="cap" x={24} y={envY + 58 + v.tools.length * rowH + 36}>
        the customer’s: named, never called
      </text>

      {box('gate', 296, envY, 232, 'merge gate', merges ? [...GATE, ...(v.rubric ?? [])].join(' · ') : 'none: no merge to check', {
        hi: ch.has('rubric'),
        ext: !merges,
        ...gateN,
        title: merges ? "rubric.py checks each merge without a model; a failure is sent back once, and a merge that still fails keeps the library's version" : 'No merge, so nothing to gate',
      })}
      <path className="ed rag" d={`M412 ${envY + 52} V${envY + 76}`} />
      {box('library', 296, envY + 78, 232, `library · ${v.name}`, lib ? `${lib.workflows} workflows${lib.commits != null ? ` · ${lib.commits} commits` : ''}` : '—', {
        n: sc?.committed ? `+${Object.keys(sc.committed).length}` : undefined,
        tone: 'ok',
        title: merges ? 'The library: the seed’s jobs, then every commit' : 'The library: the newest session’s version of each job',
      })}
      <path className="ed rag" d={`M412 ${envY + 130} V${envY + 154}`} />
      {box('readers', 296, envY + 156, 232, 'read by', readers.length ? readers.map((a) => a.name).join(' · ') : 'no answering version yet', {
        ext: !readers.length,
        title: 'The answering versions whose find_workflow looks jobs up in this library',
      })}
    </svg>
  );
}

/** The figure's caption: the takeaway, then where it came from. */
export function RagCaption({ domain, v, all, s }: { domain: string; v: RagVersion; all: RagVersion[]; s: RagSession | null }) {
  const parent = ragParent(v, all);
  const ch = [...ragChanged(v, parent, all)];
  const labels = new Map(ragLayers(v, all).map((l) => [l.key, l.label]));
  return (
    <figcaption>
      {parent ? (
        ch.length ? (
          <>
            Marked boxes differ from {parent.name}: {ch.map((k) => labels.get(k)).join(', ')}. The rest is {parent.name}&rsquo;s.
          </>
        ) : (
          <>Nothing differs from {parent.name}: the same prompts, settings and tools; only the library it starts from moved on.</>
        )
      ) : (
        <>No parent to mark against: every box is this version&rsquo;s own.</>
      )}{' '}
      {s && s.meta.agent === v.name ? <>The counts at the right are session {s.meta.task_id}&rsquo;s: turns, jobs decided, tool calls, the gate and the jobs it committed.</> : <>Open a session of {v.name} and its counts join the figure.</>}
      <span className="path">
        rag_agents/{domain}/{v.name}/ ({[...Object.keys(v.files ?? { 'rag_agent.md': 0 }), 'rag_agent.yaml', ...(v.changes ? ['changes.json'] : [])].join(', ')}) · GET /api/workflows/{domain}/rag
      </span>
    </figcaption>
  );
}

// ── the panel ─────────────────────────────────────────────────────────────
type PanelProps = {
  domain: string;
  node: RagNode | null;
  v: RagVersion;
  all: RagVersion[];
  readers: AgentInfo[];
  hist: VersionNode[] | null;
  s: RagSession | null;
  onVersion: (name: string) => void;
  onClose: () => void;
};

function frac(r: HRun | null | undefined): ReactNode {
  if (!r) return <span className="muted">not run</span>;
  return r.run_id ? (
    <Link to={runPath(r.run_id)} className="mono">
      {r.passed} / {r.n}
    </Link>
  ) : (
    <span className="mono">
      {r.passed} / {r.n}
    </span>
  );
}

function Kv({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <div className="tw flat">
      <table>
        <tbody>
          {rows.map(([k, val]) => (
            <tr key={k}>
              <td className="mono small">{k}</td>
              <td className="wrap small">{val}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** The change record's lines for one file, when this version changed it. */
function Changed({ v, file }: { v: RagVersion; file: string }) {
  const c = v.changes?.changes.filter((x) => x.file === file) ?? [];
  if (!c.length) return null;
  return (
    <>
      <h4>What {v.name} changed here</h4>
      {c.map((x, i) => (
        <div key={i} className="blk okb">
          <p className="msg">{x.what}</p>
          <p className="small muted">{x.why}</p>
        </div>
      ))}
    </>
  );
}

/** What a picked part of the figure is: its settings, its prompt, its change, and what the open session did there. */
export function RagNodePanel({ domain, node, v, all, readers, hist, s, onVersion, onClose }: PanelProps) {
  const parent = ragParent(v, all);
  const ch = ragChanged(v, parent, all);
  const sc = s && s.meta.agent === v.name ? sessionCounts(s) : null;
  const layer = new Map(ragLayers(v, all).map((l) => [l.key, l]));
  const val = (k: RagLayerKey) => layer.get(k)?.value ?? '—';
  const differs = (...k: RagLayerKey[]) => (parent && k.some((x) => ch.has(x)) ? <p className="small v-warn">Differs from {parent.name}: {parent.name} had {k.map((x) => ragLayers(parent, all).find((l) => l.key === x)?.value).join(' · ')}.</p> : null);
  const head = (what: string, title: string) => (
    <>
      <div className="crumbs">
        <span className="label">
          {domainLabel(domain)}/workflow_rag/{v.name} · {what}
        </span>
        <button type="button" className="linkish small" onClick={onClose}>
          close
        </button>
      </div>
      <h3>{title}</h3>
    </>
  );
  if (!node)
    return (
      <div className="agent-panel">
        <h3>Pick a part of {v.name}</h3>
        <p className="small muted">Each box opens what it is in {v.name}: its prompt or setting, what {v.name} changed there{parent ? ` from ${parent.name}` : ''}, and what the open session did there.</p>
      </div>
    );

  let body: ReactNode;
  if (node.startsWith('file:')) {
    const f = RAG_FILES.find((x) => `file:${x.file}` === node);
    const text = f ? f.text(v) : '';
    const o = f ? fileOrigin(v, f.file, all) : null;
    body = (
      <>
        {head(f?.turn ?? 'prompt', `${f?.file ?? node.slice(5)}: the ${f?.turn ?? ''} turn’s prompt`)}
        <p className="small">
          {o && o !== v.name ? `${o}'s, byte for byte` : `${v.name}'s own`} · {(v.files?.[f?.file ?? '']?.chars ?? text.length).toLocaleString('en-GB')} characters
          {sc && f && <> · this session: {f.layer === 'research' ? `${sc.research} research turns` : f.layer === 'decide' ? (sc.decided != null ? `${sc.decided} jobs decided` : 'no decide turn') : `${sc.mergeTurns} merge turns`}</>}
        </p>
        {f && differs(f.layer)}
        {f && <Changed v={v} file={f.file} />}
        <div className="blk">
          <pre className="tall">{text || 'This version has no such prompt.'}</pre>
        </div>
      </>
    );
  } else if (node.startsWith('tool:')) {
    const t = v.tools.find((x) => `tool:${x.name}` === node);
    const calls = sc?.calls.get(t?.name ?? '') ?? [];
    body = (
      <>
        {head('tool', t?.name ?? node.slice(5))}
        <p className="small">{t?.harness ? "workflow_rag's own research tool (tool_docs: true), answered by its harness from τ²'s tool descriptions" : `A knowledge tool the ${v.retrieval} retrieval gives, run read-only in τ²'s banking environment`}.</p>
        <div className="blk">
          <pre>{t?.description ?? 'τ² gives no description for it here.'}</pre>
        </div>
        {sc && (
          <>
            <h4>
              This session · {calls.length} {calls.length === 1 ? 'call' : 'calls'}
            </h4>
            {calls.length ? (
              <ol className="small">
                {calls.map((e, i) => (
                  <li key={i} className="mono">
                    {String(e.args?.query ?? e.args?.command ?? e.args?.tool_name ?? JSON.stringify(e.args ?? {}))}
                    {e.refused ? ' · refused' : ''}
                  </li>
                ))}
              </ol>
            ) : (
              <p className="small muted">Not called.</p>
            )}
          </>
        )}
      </>
    );
  } else if (node === 'input') {
    const q = s && s.meta.agent === v.name ? s.events.find((e) => e.kind === 'start')?.question : null;
    body = (
      <>
        {head('input', 'The customer’s script, and nothing else about the question')}
        <p className="small">
          A session reads the customer&rsquo;s whole script (<code>input: {v.input ?? 'script'}</code>): their details and how they will react. Gold never reaches it, and neither does any outcome.
        </p>
        {q ? (
          <div className="blk">
            <span className="label">this session · {sc?.task}</span>
            <pre className="tall">{q}</pre>
          </div>
        ) : (
          <p className="small muted">Open a session of {v.name} to read its script here.</p>
        )}
      </>
    );
  } else if (node === 'seed') {
    const seed = v.seed ? all.find((x) => x.name === v.seed) : null;
    body = (
      <>
        {head('seed', v.seed ? `${v.name} starts from ${v.seed}'s library` : `${v.name} starts from nothing`)}
        <Kv
          rows={[
            ['seed', v.seed ? <button type="button" className="linkish" onClick={() => onVersion(v.seed!)}>{v.seed}</button> : 'none'],
            ["seed's workflows", seed?.library ? String(seed.library.workflows) : '—'],
            ['queue', v.queue === 'after_seed' ? "after_seed: a question the seed merged is not researched again" : 'every question'],
          ]}
        />
        {differs('seed')}
        <Changed v={v} file="rag_agent.yaml" />
      </>
    );
  } else if (node === 'model' || node === 'tool_mode' || node === 'turns' || node === 'merge') {
    const rows: Record<string, [string, ReactNode][]> = {
      model: [
        ['model', `${v.model} (${modelName(v.model)})`],
        ['effort', v.effort],
      ],
      tool_mode: [['tool_mode', `${v.tool_mode}: ${toolModeText(v.tool_mode)}`]],
      turns: [
        ['max_steps', `${v.max_steps} research turns, then it must write what it has`],
        ...(v.merge ? ([['merge_steps', `${v.merge_steps ?? 12} turns for the merge, its research included`]] as [string, ReactNode][]) : []),
        ['result_chars', `${v.result_chars.toLocaleString('en-GB')}: a longer tool result is cut, and says so`],
      ],
      merge: [
        ['merge', v.merge ?? 'none'],
        ...(v.merge === 'concurrent'
          ? ([
              ['workers', String(v.workers ?? 1)],
              ['order', v.order === 'shuffled' ? `shuffled, seed ${v.order_seed ?? 0}` : 'id order'],
            ] as [string, ReactNode][])
          : []),
        ...(v.merge
          ? ([
              ['merge_write', v.merge_write === 'edits' ? 'edits: a merge into one job may be written as edits on its newest version' : 'full: every merged job written whole'],
              ['keep_tools', v.keep_tools ? 'true: a write turn keeps the tools listed, a call is refused' : 'false'],
            ] as [string, ReactNode][])
          : []),
      ],
    };
    const keys: Record<string, RagLayerKey[]> = { model: ['model'], tool_mode: ['tool_mode'], turns: ['turns'], merge: ['merge', 'merge_write'] };
    const titles: Record<string, string> = { model: `${modelName(v.model)} at ${v.effort} effort`, tool_mode: 'Native tool calls, every call of a reply run', turns: val('turns'), merge: mergeText(v) };
    body = (
      <>
        {head(node.replace('_', ' '), titles[node])}
        <Kv rows={rows[node]} />
        {node === 'merge' && sc?.waitedMs != null && (
          <p className="small">
            This session waited {fmtS(sc.waitedMs)} for its lock{sc.decided != null ? ` on the ${sc.decided} jobs it decided to write` : ''}.
          </p>
        )}
        {differs(...keys[node])}
        {keys[node].some((k) => ch.has(k)) && <Changed v={v} file="rag_agent.yaml" />}
      </>
    );
  } else if (node === 'gate') {
    const rubric = v.merge ? [...GATE, ...(v.rubric ?? [])] : [];
    body = (
      <>
        {head('merge gate', v.merge ? `${rubric.length} checks on every merge, made without a model` : 'No merge, so no gate')}
        {v.merge && (
          <>
            <ul className="small">
              {rubric.map((d) => (
                <li key={d}>
                  <b>{d}</b> {RUBRIC[d] ?? ''}
                </li>
              ))}
            </ul>
            <p className="small muted">A merge that fails is sent back once; one that still fails is not taken, and the library keeps the version it had.</p>
          </>
        )}
        {differs('rubric')}
        {sc?.merged && (
          <>
            <h4>This session</h4>
            {sc.sentBack.map((t, i) => (
              <div key={i} className="blk miss">
                <pre>{t}</pre>
              </div>
            ))}
            <p className="small">
              {sc.taken.length ? `Taken: ${sc.taken.join(', ')}.` : 'Nothing taken.'} {sc.rejected.length ? `Kept the library's version of ${sc.rejected.join(', ')}.` : ''}
            </p>
          </>
        )}
      </>
    );
  } else if (node === 'library') {
    body = (
      <>
        {head('library', `${v.name}'s library holds ${v.library?.workflows ?? '—'} workflows`)}
        <Kv
          rows={[
            ['workflows', String(v.library?.workflows ?? '—')],
            ['commits', v.library?.commits != null ? String(v.library.commits) : 'not a concurrent build'],
            ['starts from', v.seed ? `${v.seed}'s library` : 'nothing'],
          ]}
        />
        {sc?.committed && (
          <p className="small">
            This session committed{' '}
            {Object.entries(sc.committed)
              .map(([j, n]) => `${j} v${n}`)
              .join(', ')}
            .
          </p>
        )}
        {v.merge === 'concurrent' && <p className="small muted">The Library panel below lists each workflow, its version, the question that last changed it and who holds it now.</p>}
      </>
    );
  } else if (node === 'readers') {
    const node_ = (n: string) => hist?.find((h) => h.version === n) ?? null;
    body = (
      <>
        {head('read by', readers.length ? `${readers.length} answering ${readers.length === 1 ? 'version reads' : 'versions read'} ${v.name}'s library` : `No answering version reads ${v.name}'s library yet`)}
        {readers.length > 0 && (
          <div className="tw flat">
            <table>
              <thead>
                <tr>
                  <th>version</th>
                  <th>train</th>
                  <th>test</th>
                </tr>
              </thead>
              <tbody>
                {readers.map((a) => (
                  <tr key={a.name}>
                    <td>
                      <Link to={agentPath(a.domain, a.name)}>{a.name}</Link>
                    </td>
                    <td className="nw">{hist ? frac(node_(a.name)?.train) : '…'}</td>
                    <td className="nw">{hist ? frac(node_(a.name)?.test) : '…'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="small muted">Each looks a job up with find_workflow, one search over the whole library, on train and test alike.</p>
      </>
    );
  } else {
    body = (
      <>
        {head('prompts', `${Object.keys(v.files ?? {}).length || 1} prompts, one per kind of turn`)}
        <ul className="small">
          {RAG_FILES.filter((f) => v.files?.[f.file] || f.file === 'rag_agent.md').map((f) => (
            <li key={f.file}>
              <code>{f.file}</code> · the {f.turn} turn · {val(f.layer)}
            </li>
          ))}
        </ul>
        {differs('research', 'decide', 'merge_md')}
      </>
    );
  }
  return <div className="agent-panel">{body}</div>;
}

// ── the comparison table ──────────────────────────────────────────────────
type TableProps = {
  versions: RagVersion[];
  readers: (v: RagVersion) => AgentInfo[];
  hist: VersionNode[] | null;
  sessions: (v: RagVersion) => number;
  current: string;
  onVersion: (name: string) => void;
};

/** Every workflow_rag version, one column each, a row per layer; a cell that differs from that
 *  version's parent is marked. Then why it was made, its library, its sessions and who reads it. */
export function RagVersionTable({ versions, readers, hist, sessions, current, onVersion }: TableProps) {
  const rows = versions.length ? ragLayers(versions[0], versions).map((l) => [l.key, l.label] as const) : [];
  const per = new Map(versions.map((v) => [v.name, new Map(ragLayers(v, versions).map((l) => [l.key, l]))]));
  const chg = new Map(versions.map((v) => [v.name, ragChanged(v, ragParent(v, versions), versions)]));
  const node = (n: string) => hist?.find((h) => h.version === n) ?? null;
  return (
    <div className="tw">
      <table className="vtable ragtable">
        <thead>
          <tr>
            <th>layer</th>
            {versions.map((v) => {
              const p = ragParent(v, versions);
              return (
                <th key={v.name} aria-current={v.name === current ? 'true' : undefined}>
                  <button type="button" className="linkish" onClick={() => onVersion(v.name)}>
                    {v.name}
                  </button>
                  {v.name === current ? ' · this one' : ''}
                  <span className="path">{p ? `against ${p.name}` : 'no parent'}</span>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {rows.map(([key, label]) => (
            <tr key={key}>
              <td className="sub">{label}</td>
              {versions.map((v) => {
                const l = per.get(v.name)?.get(key);
                const hi = chg.get(v.name)?.has(key) ?? false;
                return (
                  <td key={v.name} className={`wrap ${hi ? 'chg' : ''}`}>
                    {l?.value ?? '—'}
                    {l?.detail && <span className="path">{l.detail}</span>}
                    {hi && <span className="path chg-mark">≠ {ragParent(v, versions)?.name}</span>}
                  </td>
                );
              })}
            </tr>
          ))}
          <tr>
            <td className="sub">
              why · defect<span className="path">changes.json, r5 on</span>
            </td>
            {versions.map((v) => (
              <td key={v.name} className="wrap">
                {v.changes ? (
                  <>
                    {v.changes.defect ?? '—'} · {v.changes.changes.map((c) => c.file).join(', ')}
                    <span className="path" title={v.changes.why}>
                      {clip(v.changes.why, 140)}
                    </span>
                  </>
                ) : (
                  <span className="muted">no change record{v.written ? ` · ${v.written}` : ''}</span>
                )}
              </td>
            ))}
          </tr>
          <tr>
            <td className="sub">library</td>
            {versions.map((v) => (
              <td key={v.name} className="nw">
                {v.library ? `${v.library.workflows} workflows` : '—'}
                {v.library?.commits != null && <span className="path">{v.library.commits} commits</span>}
              </td>
            ))}
          </tr>
          <tr>
            <td className="sub">sessions</td>
            {versions.map((v) => (
              <td key={v.name} className="nw">
                {sessions(v)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="sub">
              read by · train · test<span className="path">answering versions, passed / conversations</span>
            </td>
            {versions.map((v) => {
              const rs = readers(v);
              return (
                <td key={v.name} className="wrap">
                  {rs.length ? (
                    rs.map((a) => (
                      <span key={a.name} className="rdr">
                        <Link to={agentPath(a.domain, a.name)}>{a.name}</Link>{' '}
                        {hist ? (
                          <>
                            {frac(node(a.name)?.train)} · {frac(node(a.name)?.test)}
                          </>
                        ) : (
                          '…'
                        )}
                      </span>
                    ))
                  ) : (
                    <span className="muted">none</span>
                  )}
                </td>
              );
            })}
          </tr>
        </tbody>
      </table>
    </div>
  );
}
