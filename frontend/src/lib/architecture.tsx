/**
 * The Agent tab's version view (s13 §2): any version's architecture, drawn from its own
 * folder with no run, and every version of a dataset side by side. Each layer is compared
 * with the version's parent (the one it was made from, else the one before it) and a layer
 * that differs is marked, in the figure (`.nd.hi`) and in the table (`td.chg`), always with
 * the words `≠ <parent>` beside the colour.
 *
 * Everything here reads `GET /api/agents` (a version's `agent.yaml`, surfaces, retrieval
 * and prompt layers), `GET /api/domains/<d>` (the dataset's tools) and `GET /api/versions`
 * (the train and test runs). Nothing is drawn that those do not say.
 */

import { Fragment, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { type AgentInfo, type HRun, type RetrievalInfo, type ToolSpec, type TrialPayload, type VersionNode, domainLabel } from './api';
import { MODELS, modelFamily } from './scope';
import { agentPath, runPath } from './url';

/** Every knowledge tool a banking retrieval variant can give, in `eval/retrieval.py`'s order
 *  (`variant_tools`): a dataset extract lists the ones its own variant gave. */
export const KNOWLEDGE_TOOLS = ['KB_search', 'KB_search_bm25', 'KB_search_dense', 'grep', 'shell'];
/** The hooks `helper.py` may define (agent/compose.py `HELPER_HOOKS`). */
const HELPER_HOOKS = ['extra_context', 'on_tool_call', 'on_reply'];

/** `sonnet`, `claude-sonnet-5` → `Sonnet 5`, as the scope bar names it. */
export const modelName = (m: string): string => MODELS.find(([k]) => k === modelFamily(m))?.[1] ?? m;

/** A version's tool mode, short (a table cell) and whole (the model panel). */
export function toolModeText(mode: string | null | undefined): string {
  return mode === 'native' ? 'native tool calls' : 'JSON in the reply';
}
export function toolModeLong(mode: string | null | undefined): string {
  return mode === 'native'
    ? 'native: the model is given the tools and calls them; each call returns to tau2 unrun'
    : `${mode || 'json'}: tools travel as a JSON contract in the prompt, not as native tool calls`;
}

const cfgOf = (v: AgentInfo) => v.config as { model?: string; effort?: string; tool_mode?: string; max_steps?: number; identity_note?: boolean; workflows?: string | null };
const modeOf = (v: AgentInfo) => v.tool_mode ?? cfgOf(v).tool_mode ?? 'json';
const sha = (v: AgentInfo, f: string) => v.surfaces?.[f]?.sha ?? null;
const codeOf = (v: AgentInfo) => (v.surfaces_present ?? (v.has_helper ? ['system.md', 'helper.py'] : ['system.md'])).filter((f) => f !== 'system.md');
/** A prompt layer as the table compares it: the template has its own row, so the policy is one word here. */
const layerName = (l: string) => (l.startsWith('policy: ') ? 'policy' : l);

/** The earliest version (in build order) whose `f` has these bytes: a fork's `system.md` reads as its source's. */
function origin(v: AgentInfo, f: string, all: AgentInfo[]): string {
  const s = sha(v, f);
  if (!s) return v.name;
  return all.find((x) => x.domain === v.domain && sha(x, f) === s)?.name ?? v.name;
}

/** How a version was made, in words: `loop cycle 1 from v1`, `model swap from v0`, `base`. */
export function madeText(v: AgentInfo): { value: string; detail: string } {
  const m = v.made_by;
  if (!m) return { value: '—', detail: '' };
  if (m.kind === 'base') return { value: 'base', detail: m.detail };
  const lead = m.kind === 'loop cycle' && m.cycle != null ? `loop cycle ${m.cycle}` : m.kind;
  return { value: m.from ? `${lead} from ${m.from}` : lead, detail: m.detail };
}

export type LayerKey = 'model' | 'tool_mode' | 'retrieval' | 'dense' | 'template' | 'layers' | 'system' | 'code' | 'workflows' | 'steps';
export type Layer = { key: LayerKey; label: string; value: string; detail?: string; cmp: string };

/** The layers a version is compared on, in the table's order. `withRetrieval` adds the three a
 *  retrieval variant decides (banking's); a dataset without one shows none of them. */
export function layers(v: AgentInfo, all: AgentInfo[], withRetrieval = !!v.retrieval): Layer[] {
  const c = cfgOf(v);
  const info: RetrievalInfo | null = v.retrieval_info ?? null;
  const tools = info?.tools ?? [];
  const sys = origin(v, 'system.md', all);
  const code = codeOf(v);
  const hooks = v.helper_functions.filter((f) => HELPER_HOOKS.includes(f));
  const out: Layer[] = [
    { key: 'model', label: 'model · effort', value: `${modelName(String(c.model ?? '—'))} · ${c.effort ?? 'medium'}`, cmp: `${c.model}|${c.effort ?? 'medium'}` },
    { key: 'tool_mode', label: 'tool calls', value: toolModeText(modeOf(v)), cmp: modeOf(v) },
  ];
  if (withRetrieval) {
    out.push(
      {
        key: 'retrieval',
        label: 'retrieval',
        value: v.retrieval ? `${v.retrieval}: ${tools.length ? tools.join(', ') : 'tools not readable here'}` : 'none',
        cmp: `${v.retrieval}|${tools.join(',')}`,
      },
      { key: 'dense', label: 'dense model', value: info?.dense_model ?? 'none', cmp: info?.dense_model ?? '' },
      { key: 'template', label: 'policy template', value: info?.template ?? (v.retrieval ? '—' : "τ²'s domain policy"), cmp: info?.template ?? '' },
    );
  }
  const pl = (v.prompt_layers ?? []).map(layerName);
  out.push(
    { key: 'layers', label: 'prompt, in order', value: pl.length ? pl.join(' + ') : '—', cmp: pl.join('+') },
    {
      key: 'system',
      label: 'system.md',
      value: sys === v.name ? `its own · ${(v.surfaces?.['system.md']?.chars ?? 0).toLocaleString('en-GB')} chars` : `${sys}'s, byte for byte`,
      cmp: sha(v, 'system.md') ?? v.fingerprint,
    },
    {
      key: 'code',
      label: 'code surfaces',
      value: code.length ? code.map((f) => (origin(v, f, all) === v.name ? f : `${f} (${origin(v, f, all)}'s)`)).join(' + ') : 'none',
      detail: hooks.length ? `hooks: ${hooks.join(', ')}` : undefined,
      cmp: code.map((f) => `${f}:${sha(v, f)}`).join(','),
    },
  );
  // s16: workflow_rag's library and research as two harness tools; a row only where a version has them
  if (all.some((x) => x.domain === v.domain && cfgOf(x).workflows)) {
    out.push({
      key: 'workflows',
      label: 'workflow tools',
      value: c.workflows ? `find_workflow + request_workflow → workflow_rag ${c.workflows}` : 'none',
      cmp: c.workflows ?? '',
    });
  }
  out.push({ key: 'steps', label: 'turn cap', value: `max ${c.max_steps ?? 200} steps`, cmp: String(c.max_steps ?? 200) });
  return out;
}

export function parentOf(v: AgentInfo, all: AgentInfo[]): AgentInfo | null {
  return v.parent ? (all.find((x) => x.domain === v.domain && x.name === v.parent) ?? null) : null;
}

/** The layers where `v` differs from `parent`; none without one. */
export function changed(v: AgentInfo, parent: AgentInfo | null, all: AgentInfo[], withRetrieval = !!(v.retrieval || parent?.retrieval)): Set<LayerKey> {
  if (!parent) return new Set();
  const p = new Map(layers(parent, all, withRetrieval).map((l) => [l.key, l.cmp]));
  return new Set(layers(v, all, withRetrieval).filter((l) => p.get(l.key) !== l.cmp).map((l) => l.key));
}

/** The assertion for a version's section: what changed from its parent, with the denominator. */
export function versionClaim(v: AgentInfo, all: AgentInfo[]): string {
  const parent = parentOf(v, all);
  if (!parent) {
    const from = v.made_by?.from;
    return from && from !== v.name ? `${v.name} has no parent here: ${from} was retired` : `${v.name} is the first version: nothing to compare`;
  }
  const n = layers(v, all, !!(v.retrieval || parent.retrieval)).length;
  const ch = changed(v, parent, all);
  return ch.size ? `${v.name} differs from ${parent.name} in ${ch.size} of ${n} layers` : `${v.name} matches ${parent.name} in all ${n} layers`;
}

/**
 * The trial's tool list as the version's retrieval gives it. The dataset extract was made under
 * one variant, so a version on another would still list that variant's knowledge tools: here
 * they are swapped for the version's own (`retrieval_info.tools`). A knowledge tool the extract
 * does not describe gets the variant's name as its description, never an invented one.
 */
export function withKnowledge(t: TrialPayload, info: RetrievalInfo | null | undefined): TrialPayload {
  if (!info?.tools.length) return t;
  const have = t.tools.filter((x) => KNOWLEDGE_TOOLS.includes(x.name)).map((x) => x.name);
  if (have.join(',') === info.tools.join(',')) return t;
  const spec = (n: string): ToolSpec =>
    t.tools.find((x) => x.name === n) ?? { name: n, description: `A knowledge tool the ${info.variant} retrieval gives (tau2's variant spec).`, type: 'read', mutates: false };
  return { ...t, tools: [...info.tools.map(spec), ...t.tools.filter((x) => !KNOWLEDGE_TOOLS.includes(x.name))] };
}

// ── the figure ────────────────────────────────────────────────────────────
const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
const gid = (s: string) => `arch-${s.replace(/[^a-z0-9_-]/gi, '-')}`;

type FigProps = { v: AgentInfo; all: AgentInfo[]; tools: ToolSpec[] | null };

/** One version, as the Agent tab's graph draws a conversation but from its folder alone: the
 *  orchestrator, the agent (prompt layers, model, tool calls, code surfaces) and the environment
 *  (the retrieval's knowledge tools, then the dataset's own). */
export function ArchitectureFigure({ v, all, tools }: FigProps) {
  const W = 540;
  const parent = parentOf(v, all);
  const ch = changed(v, parent, all);
  const ne = parent ? `≠ ${parent.name}` : '';
  const c = cfgOf(v);
  const info = v.retrieval_info ?? null;
  const mode = modeOf(v);
  const code = codeOf(v);
  const sys = origin(v, 'system.md', all);
  const parentLayers = new Set((parent?.prompt_layers ?? []).map(layerName));

  const box = (id: string, x: number, y: number, w: number, h: number, label: string, sub: string, hi: boolean, title: string, ext = false) => (
    <g key={id} id={gid(id)}>
      <title>{`${title}${hi ? ` — differs from ${parent?.name}` : ''}`}</title>
      <rect className={`nd ${hi ? 'hi' : ext ? 'ext' : ''}`} x={x} y={y} width={w} height={h} rx={8} />
      <text className="tx" x={x + 12} y={y + 22}>
        {label}
      </text>
      <text className="tx s" x={x + 12} y={y + 41}>
        {sub}
      </text>
      {hi && (
        <text className="tx n hi" x={x + w - 10} y={y + 22} textAnchor="end">
          {ne}
        </text>
      )}
    </g>
  );
  // a row's mark is `≠ <parent>`, or `new` for a tool the parent did not have
  const row = (id: string, x: number, y: number, w: number, label: string, hi: boolean, title: string, idle = false, mark = ne) => (
    <g key={id} id={gid(id)}>
      <title>{`${title}${hi ? (mark === 'new' ? ` — new since ${parent?.name}` : ` — differs from ${parent?.name}`) : ''}`}</title>
      <rect className={`nd ${hi ? 'hi' : idle ? 'idle' : ''}`} x={x} y={y} width={w} height={22} rx={6} />
      <text className={`tx t ${idle ? 'idle' : ''}`} x={x + 8} y={y + 15.5}>
        {clip(label, Math.floor((w - 16 - (hi ? mark.length * 8 + 6 : 0)) / 6.8))}
      </text>
      {hi && (
        <text className="tx n hi" x={x + w - 8} y={y + 15.5} textAnchor="end">
          {mark}
        </text>
      )}
    </g>
  );

  // the prompt's layers, each marked when the parent's prompt lacks it (or its bytes or template differ)
  const prompt = (v.prompt_layers ?? []).map((l) => {
    if (l === 'system.md') return { id: 'system', label: sys === v.name ? 'system.md · its own' : `system.md · ${sys}'s`, hi: ch.has('system'), title: 'system.md: the optimiser’s surface, with the policy in its {policy} slot' };
    if (l.startsWith('policy: ')) return { id: 'policy', label: l.replace(': ', ' · '), hi: ch.has('template'), title: `The policy in the slot: ${l.slice(8)}` };
    return { id: l, label: l, hi: !!parent && !parentLayers.has(l), title: `${l}: joined after the policy by compose()` };
  });

  const rowH = 27;
  const orchY = 12;
  const grpY = orchY + 50 + 28;
  const leftH = 30 + 50 + 10 + prompt.length * rowH;
  const rightH = 30 + (c.workflows ? 4 : 3) * 62 - 12;
  const grpH = Math.max(leftH, rightH) + 14;
  const envY = grpY + grpH + 44;
  const knowledge = v.retrieval ? (info?.tools ?? []) : [];
  const parentTools = new Set(parent?.retrieval_info?.tools ?? []);
  const kW = knowledge.length ? (492 - (knowledge.length - 1) * 12) / knowledge.length : 0;
  const kH = v.retrieval ? 58 + 22 + 30 : 30;
  const rest = (tools ?? []).filter((x) => !KNOWLEDGE_TOOLS.includes(x.name));
  const left = rest.filter((x) => x.type !== 'write');
  const right = rest.filter((x) => x.type === 'write');
  const envRows = Math.max(left.length, right.length);
  const envH = kH + (tools ? 28 + envRows * rowH : 0) + 8;
  const H = envY + envH + 12;
  const nTools = knowledge.length + rest.length;
  const dl = domainLabel(v.domain);

  return (
    <svg
      className="dia graph agentgraph archgraph"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`${dl}/${v.name} drawn from its folder alone: ${modelName(String(c.model))} at ${c.effort ?? 'medium'} effort, tool calls ${toolModeText(mode)}, ${v.retrieval ? `retrieval ${v.retrieval}, ` : ''}a prompt of ${prompt.length} layers and ${code.length ? code.join(' and ') : 'no code surfaces'}${parent ? `; ${ch.size} of its layers differ from ${parent.name}` : ''}.`}
    >
      <defs>
        <marker id="archarr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0L10 5L0 10z" fill="currentColor" />
        </marker>
      </defs>

      {box('orch', 12, orchY, 516, 50, 'orchestrator · τ² half-duplex', `max ${c.max_steps ?? 200} steps · τ²'s user simulator plays the customer`, ch.has('steps'), 'The orchestrator: owns the turn loop and runs every tool call')}
      <path className="ed two" d={`M270 ${orchY + 52} V${grpY - 2}`} />

      <rect className="grp" x={12} y={grpY} width={516} height={grpH} rx={10} />
      <text className="tx k" x={24} y={grpY + 20}>
        agent · {dl}/{v.name} · {v.fingerprint.slice(0, 7)}
      </text>
      {box('prompt', 24, grpY + 30, 236, 50, 'system prompt', `${prompt.length} layers, joined in order`, ch.has('layers'), 'The composed system prompt, layer by layer as compose() joins them')}
      {prompt.map((p, i) => row(`layer-${p.id}`, 36, grpY + 30 + 60 + i * rowH, 224, p.label, p.hi, p.title))}
      {box('model', 272, grpY + 30, 244, 50, 'model', `${modelName(String(c.model ?? '—'))} · ${c.effort ?? 'medium'} effort`, ch.has('model'), 'The model and its effort, frozen in agent.yaml')}
      {box('tool-mode', 272, grpY + 92, 244, 50, 'tool calls', toolModeText(mode), ch.has('tool_mode'), toolModeLong(mode))}
      {box('code', 272, grpY + 154, 244, 50, 'code surfaces', code.length ? clip(code.join(' + '), 30) : 'none', ch.has('code'), 'helper.py, checks.py, memory.py and guidance.py: the deterministic hooks around the model', !code.length)}
      {c.workflows &&
        box(
          'workflows',
          272,
          grpY + 216,
          244,
          50,
          'workflow tools',
          `find · request → workflow_rag ${c.workflows}`,
          ch.has('workflows'),
          'find_workflow looks the job up in the workflows workflow_rag wrote; request_workflow has it research one now. The harness runs both inside the turn: τ² never sees them.',
        )}

      <path className="ed two" d={`M270 ${grpY + grpH + 2} V${envY - 2}`} />
      <text className="cap" x={282} y={grpY + grpH + 27}>
        {mode === 'native' ? 'native calls, returned to τ² unrun' : 'calls parsed from the reply text'}
      </text>

      <rect className="grp" x={12} y={envY} width={516} height={envH} rx={10} />
      <text className="tx k" x={24} y={envY + 22}>
        environment · {dl} · {tools ? `${nTools} tools` : 'tools loading'}
      </text>
      {v.retrieval && (
        <>
          <text className={`tx k ${ch.has('retrieval') ? 'hi' : ''}`} x={24} y={envY + 46}>
            knowledge · {v.retrieval}
            {ch.has('retrieval') && <tspan className="mk">{` · ${ne}`}</tspan>}
          </text>
          {knowledge.length === 0 && (
            <text className="cap" x={24} y={envY + 72}>
              its tools are read from τ²'s spec, which this server cannot load
            </text>
          )}
          {knowledge.map((name, i) => row(`k-${name}`, 24 + i * (kW + 12), envY + 58, kW, name, !!parent && !parentTools.has(name), `${name}: a knowledge tool the ${v.retrieval} retrieval gives`, false, 'new'))}
          <text className="cap" x={24} y={envY + 58 + 22 + 22}>
            dense model · {info?.dense_model ?? 'none'}
            {ch.has('dense') ? ` · ${ne}` : ''}
          </text>
        </>
      )}
      {tools && (
        <>
          <text className="tx k" x={24} y={envY + kH + 16}>
            read · {left.length}
          </text>
          <text className="tx k" x={272} y={envY + kH + 16}>
            write · {right.length}
          </text>
          {left.map((x, i) => row(`t-${x.name}`, 24, envY + kH + 28 + i * rowH, 236, x.name, false, `${x.name} (${x.type}): the dataset's own tool, the same in every version`, true))}
          {right.map((x, i) => row(`t-${x.name}`, 272, envY + kH + 28 + i * rowH, 244, x.name, false, `${x.name} (${x.type}): the dataset's own tool, the same in every version`, true))}
        </>
      )}
    </svg>
  );
}

// ── the comparison table ──────────────────────────────────────────────────
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

type TableProps = { versions: AgentInfo[]; hist: VersionNode[] | null; current: string };

/** Every version of one dataset, one column each, a row per layer; a cell that differs from
 *  that version's parent is marked. The judge's side-by-side is the pattern (`JudgeAgent.tsx`). */
export function VersionTable({ versions, hist, current }: TableProps) {
  const withRetrieval = versions.some((v) => v.retrieval);
  const rows = versions.length ? layers(versions[0], versions, withRetrieval).map((l) => [l.key, l.label] as const) : [];
  const per = new Map(versions.map((v) => [v.name, new Map(layers(v, versions, withRetrieval).map((l) => [l.key, l]))]));
  const chg = new Map(versions.map((v) => [v.name, changed(v, parentOf(v, versions), versions, withRetrieval)]));
  const node = (name: string) => hist?.find((h) => h.version === name) ?? null;
  return (
    <div className="tw">
      <table className="vtable">
        <thead>
          <tr>
            <th>layer</th>
            {versions.map((v) => (
              <th key={v.name} aria-current={v.name === current ? 'true' : undefined}>
                <Link to={agentPath(v.domain, v.name)}>{v.name}</Link>
                {v.name === current ? ' · this one' : ''}
                <span className="path">{v.parent ? `against ${v.parent}` : 'no parent'}</span>
              </th>
            ))}
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
                    {hi && <span className="path chg-mark">≠ {v.parent}</span>}
                  </td>
                );
              })}
            </tr>
          ))}
          <tr>
            <td className="sub">made by · kind</td>
            {versions.map((v) => {
              const m = madeText(v);
              return (
                <td key={v.name} className="wrap">
                  {m.value}
                  {m.detail && <span className="path">{m.detail}</span>}
                </td>
              );
            })}
          </tr>
          <tr>
            <td className="sub">
              train · test<span className="path">passed / conversations</span>
            </td>
            {versions.map((v) => {
              const h = node(v.name);
              return (
                <td key={v.name} className="nw">
                  {hist ? (
                    <>
                      {frac(h?.train)}
                      {' · '}
                      {frac(h?.test)}
                    </>
                  ) : (
                    '…'
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

/** The figure's caption: the takeaway, then where it came from. */
export function ArchitectureCaption({ v, all }: { v: AgentInfo; all: AgentInfo[] }) {
  const parent = parentOf(v, all);
  const ch = [...changed(v, parent, all)];
  const labels = new Map(layers(v, all, !!(v.retrieval || parent?.retrieval)).map((l) => [l.key, l.label]));
  return (
    <figcaption>
      {parent ? (
        ch.length ? (
          <>
            Marked boxes differ from {parent.name}:{' '}
            {ch.map((k, i) => (
              <Fragment key={k}>
                {i ? ', ' : ''}
                {labels.get(k)}
              </Fragment>
            ))}
            . The rest is {parent.name}&rsquo;s.
          </>
        ) : (
          <>Nothing differs from {parent.name}: the same model, tools, prompt and code.</>
        )
      ) : (
        <>No parent to mark against: every box is this version&rsquo;s own.</>
      )}{' '}
      Drawn without a run: no environment is built, no sandbox started, no embedding model loaded.
      <span className="path">
        agents/{v.domain}/{v.name}/ ({(v.surfaces_present ?? []).concat('agent.yaml').join(', ')}) · τ²&rsquo;s retrieval spec via src/tau2_loop/eval/retrieval.py
      </span>
    </figcaption>
  );
}
