/**
 * The viewer's scope (s11 review): which dataset, which agent, which model. Two agents live here
 * and are kept apart: the answering agent, which talks to the customer and calls the tools, and
 * the LLM judge, which reviews the answering agent's plans. The bar at the top sets all three, and
 * every tab honours them.
 *
 * The scope is not new state with its own URL: each tab already names its dataset (a path segment
 * or a `?domain=` lens) and, where both agents have a view, which agent (`/optimise/<d>/judge`,
 * `/review/golden/<d>`, `/agent/<d>/judge`, `?agent=judge` on Runs). The bar reads the scope back
 * from the address, so a shared link carries it, and remembers the last one for tabs that say
 * nothing (Overview, Rubric). Changing it moves the current tab to its address under the new scope.
 */

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { type AgentInfo, DOMAINS, type Registries, domainLabel, useGet } from './api';

export type AgentKind = 'answering' | 'judge';
export type Scope = { dataset: string; agent: AgentKind; model: string };
export type Tab = 'overview' | 'evals' | 'rubric' | 'leaderboard' | 'runs' | 'optimise' | 'agent' | 'review';

export const AGENTS: [AgentKind, string, string][] = [
  ['answering', 'answering agent', 'talks to the customer and calls the tools'],
  ['judge', 'LLM judge', 'reviews the answering agent’s plans before the customer sees them'],
];
/** Model families, by what every model string here contains. */
export const MODELS: [string, string][] = [
  ['haiku', 'Haiku 4.5'],
  ['sonnet', 'Sonnet 5'],
  ['opus', 'Opus 5.5'],
];
/** The tabs where both agents have a view, so Agent and Model filter what is shown. */
export const AGENT_TABS: Tab[] = ['evals', 'runs', 'optimise', 'agent', 'review'];

const DEFAULT: Scope = { dataset: 'airline', agent: 'answering', model: '' };
const STORE = 'tau2loop.scope';

/** `claude-sdk/claude-haiku-4-5`, `claude-sonnet-5`, `opus` → `haiku`, `sonnet`, `opus`. */
export function modelFamily(m: string | null | undefined): string {
  const s = (m ?? '').toLowerCase();
  return MODELS.find(([k]) => s.includes(k))?.[0] ?? s;
}

export function tabOf(pathname: string): Tab {
  const seg = pathname.split('/')[1] ?? '';
  const known: Record<string, Tab> = {
    '': 'overview',
    evals: 'evals',
    domains: 'evals',
    rubric: 'rubric',
    leaderboard: 'leaderboard',
    runs: 'runs',
    optimise: 'optimise',
    agent: 'agent',
    review: 'review',
  };
  return known[seg] ?? 'overview';
}

/** What the address says about the scope; a key is absent where the address says nothing. */
export function scopeFromLocation(pathname: string, search: string): Partial<Scope> {
  const parts = pathname.split('/').map((p) => decodeURIComponent(p));
  const q = new URLSearchParams(search);
  const out: Partial<Scope> = {};
  const ds = (d: string | undefined) => {
    if (d && (DOMAINS as readonly string[]).includes(d)) out.dataset = d;
  };
  const tab = tabOf(pathname);
  if (tab === 'evals' || tab === 'optimise' || tab === 'agent') ds(parts[2]);
  if (tab === 'leaderboard' || tab === 'runs' || tab === 'review') ds(q.get('domain') ?? undefined);
  if (tab === 'runs' && parts[2]) ds(DOMAINS.find((d) => parts[2].includes(`_${d}_`)));
  if (tab === 'review' && parts[2] === 'golden') ds(parts[3]);
  if (tab === 'review' && parts[2] && parts[2] !== 'golden') ds(DOMAINS.find((d) => parts[2].includes(`_${d}_`)));
  if ((tab === 'optimise' || tab === 'evals') && parts[2]) out.agent = parts[3] === 'judge' ? 'judge' : 'answering';
  if (tab === 'agent' && parts[2]) out.agent = parts[3] === 'judge' ? 'judge' : 'answering';
  if (tab === 'review') out.agent = parts[2] === 'golden' ? 'judge' : 'answering';
  if (tab === 'runs') out.agent = !parts[2] && q.get('agent') === 'judge' ? 'judge' : 'answering';
  if (AGENT_TABS.includes(tab) && q.has('model')) out.model = q.get('model') ?? '';
  return out;
}

function qs(o: Record<string, string>): string {
  const p = Object.entries(o).filter(([, v]) => v);
  return p.length ? `?${p.map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&')}` : '';
}

/** The Agent tab's version for a dataset and model: the champion if it fits, else the newest that does. */
export function agentVersion(s: Scope, versions: AgentInfo[], registry: Registries): string | null {
  const mine = versions.filter((v) => v.domain === s.dataset && (!s.model || modelFamily(String(v.config.model)) === s.model));
  const champ = registry[s.dataset]?.champion?.agent;
  if (champ && mine.some((v) => v.name === champ)) return champ;
  const byNum = mine.slice().sort((a, b) => Number(b.name.slice(1)) - Number(a.name.slice(1)));
  return byNum[0]?.name ?? null;
}

/** Where `tab` opens under `s`. */
export function scopeHref(tab: Tab, s: Scope, agents?: { versions: AgentInfo[]; registry: Registries } | null): string {
  const d = encodeURIComponent(s.dataset);
  const judge = s.agent === 'judge';
  switch (tab) {
    case 'overview':
      return '/';
    case 'rubric':
      return '/rubric';
    case 'evals':
      return judge ? `/evals/${d}/judge${qs({ model: s.model })}` : `/evals/${d}`;
    case 'leaderboard':
      return `/leaderboard${qs({ domain: s.dataset })}`;
    case 'runs':
      return `/runs${qs({ domain: s.dataset, agent: judge ? 'judge' : '', model: s.model })}`;
    case 'optimise':
      return `/optimise/${d}${judge ? '/judge' : ''}${qs({ model: s.model })}`;
    case 'review':
      return judge ? `/review/golden/${d}` : `/review${qs({ domain: s.dataset, model: s.model })}`;
    case 'agent': {
      if (judge) return `/agent/${d}/judge${qs({ model: s.model })}`;
      const v = agents ? agentVersion(s, agents.versions, agents.registry) : null;
      return v ? `/agent/${d}/${encodeURIComponent(v)}` : '/agent';
    }
  }
}

type Ctx = { scope: Scope; set: (patch: Partial<Scope>) => void; href: (tab: Tab) => string; tab: Tab };
const ScopeCtx = createContext<Ctx | null>(null);

function readStored(): Scope {
  try {
    const raw = window.localStorage.getItem(STORE);
    return raw ? { ...DEFAULT, ...(JSON.parse(raw) as Partial<Scope>) } : DEFAULT;
  } catch {
    return DEFAULT;
  }
}

export function ScopeProvider({ children }: { children: ReactNode }) {
  const loc = useLocation();
  const nav = useNavigate();
  const { data: agents } = useGet<{ versions: AgentInfo[]; registry: Registries }>('/api/agents');
  const [stored, setStored] = useState<Scope>(readStored);
  const fromLoc = useMemo(() => scopeFromLocation(loc.pathname, loc.search), [loc.pathname, loc.search]);
  const scope: Scope = { ...stored, ...fromLoc };
  const tab = tabOf(loc.pathname);

  // remember what the address said, for the tabs that say nothing
  useEffect(() => {
    const next = { ...stored, ...fromLoc };
    if (next.dataset !== stored.dataset || next.agent !== stored.agent || next.model !== stored.model) {
      setStored(next);
      try {
        window.localStorage.setItem(STORE, JSON.stringify(next));
      } catch {
        /* a private window: the scope still works for this page */
      }
    }
    // `stored` is what this updates, so it is not a dependency
  }, [fromLoc]);

  const value: Ctx = {
    scope,
    tab,
    href: (t) => scopeHref(t, scope, agents),
    set: (patch) => {
      const next = { ...scope, ...patch };
      if (patch.agent && patch.agent !== scope.agent) next.model = '';
      setStored(next);
      try {
        window.localStorage.setItem(STORE, JSON.stringify(next));
      } catch {
        /* see above */
      }
      if (tab !== 'overview' && tab !== 'rubric') nav(scopeHref(tab, next, agents));
    },
  };
  return <ScopeCtx.Provider value={value}>{children}</ScopeCtx.Provider>;
}

export function useScope(): Ctx {
  const c = useContext(ScopeCtx);
  if (!c) throw new Error('useScope outside ScopeProvider');
  return c;
}

/** The bar under the tabs: Dataset everywhere it applies; Agent and Model where both agents have a view. */
export function ScopeBar() {
  const { scope, set, tab } = useScope();
  const { data: runs } = useGet<{ domain: string; model: string; dry_run: boolean }[]>('/api/runs');
  const { data: replays } = useGet<{ model: string }[]>(scope.agent === 'judge' ? `/api/judge/${encodeURIComponent(scope.dataset)}/replays` : null);
  if (tab === 'overview' || tab === 'rubric') return null;
  const used = new Set(
    scope.agent === 'judge' && tab === 'evals'
      ? (runs ?? []).filter((r) => r.domain === scope.dataset && !r.dry_run).map((r) => modelFamily(r.model))
      : scope.agent === 'judge'
      ? (replays ?? []).map((r) => modelFamily(r.model))
      : (runs ?? []).filter((r) => r.domain === scope.dataset && !r.dry_run).map((r) => modelFamily(r.model)),
  );
  const withAgent = AGENT_TABS.includes(tab);
  return (
    <div className="in scopebar" role="group" aria-label="scope">
      <label className="pick">
        <span className="label">dataset</span>
        <select value={scope.dataset} onChange={(e) => set({ dataset: e.target.value })}>
          {DOMAINS.map((d) => (
            <option key={d} value={d}>
              {domainLabel(d)}
            </option>
          ))}
        </select>
      </label>
      {withAgent && (
        <div className="seg" role="radiogroup" aria-label="agent">
          <span className="label">agent</span>
          {AGENTS.map(([k, label, what]) => (
            <button
              key={k}
              type="button"
              role="radio"
              aria-checked={scope.agent === k}
              className={`chip nav ${scope.agent === k ? 'on' : ''}`}
              title={what}
              onClick={() => set({ agent: k })}
            >
              {label}
            </button>
          ))}
        </div>
      )}
      {withAgent && !(tab === 'evals' && scope.agent === 'answering') && (
        <label className="pick">
          <span className="label">model</span>
          <select value={scope.model} onChange={(e) => set({ model: e.target.value })}>
            <option value="">all</option>
            {MODELS.map(([k, label]) => (
              <option key={k} value={k} disabled={!used.has(k) && scope.model !== k}>
                {label}
                {used.has(k) ? '' : ' · none'}
              </option>
            ))}
          </select>
        </label>
      )}
    </div>
  );
}
