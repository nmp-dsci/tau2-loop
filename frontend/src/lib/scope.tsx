/**
 * The viewer's scope (s11 review): which dataset, which agent, which experiment. Two agents live
 * here and are kept apart: the answering agent, which talks to the customer and calls the tools,
 * and the LLM judge, which reviews its writes and transfers. An experiment is a version of the
 * answering agent (`v0`, `v1`, … : the `<agent>` in every run id), so it filters whatever that
 * version produced: its runs, the round that made it, its conversations in Review and in the
 * judge's eval set. The bar at the top sets all three, and every tab honours them.
 *
 * The scope is not new state with its own URL: each tab already names its dataset (a path segment
 * or a `?domain=` lens) and, where both agents have a view, which agent (`/optimise/<d>/judge`,
 * `/review/golden/<d>`, `/agent/<d>/judge`, `?agent=judge` on Runs). The bar reads the scope back
 * from the address, so a shared link carries it, and remembers the last one for tabs that say
 * nothing (Overview, Rubric). Changing it moves the current tab to its address under the new scope.
 * The experiment is remembered the same way, so it carries from tab to tab until it is set back to
 * all.
 */

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { type AgentInfo, DOMAINS, type Registries, byTask, domainLabel, useGet } from './api';

export type AgentKind = 'answering' | 'judge';
export type Scope = { dataset: string; agent: AgentKind; exp: string };
export type Tab = 'overview' | 'evals' | 'rubric' | 'leaderboard' | 'runs' | 'optimise' | 'agent' | 'review';

export const AGENTS: [AgentKind, string, string][] = [
  ['answering', 'answering agent', 'talks to the customer and calls the tools'],
  ['judge', 'LLM judge', 'reviews the answering agent’s writes and transfers before they run'],
];
/** Model families, by what every model string here contains: for naming a model, not for the scope. */
export const MODELS: [string, string][] = [
  ['haiku', 'Haiku 4.5'],
  ['sonnet', 'Sonnet 5'],
  ['opus', 'Opus 5.5'],
];
/** The tabs where both agents have a view, so the Agent switch and the experiment apply. */
export const AGENT_TABS: Tab[] = ['evals', 'runs', 'optimise', 'agent', 'review'];

/** Where an experiment filters what is shown: what a version of the answering agent produced. The
 *  answering agent's tasks are the same for every version, and the judge's replays, loop and
 *  versions are the judge's own, so there it is carried but not applied. */
export function expApplies(tab: Tab, agent: AgentKind): boolean {
  if (tab === 'evals') return agent === 'judge';
  if (tab === 'review') return true;
  return agent === 'answering' && (tab === 'runs' || tab === 'optimise' || tab === 'agent');
}

/** A dataset's experiments: every version of its answering agent, made or run, in order. */
export function experimentsOf(dataset: string, versions: AgentInfo[], runs: { domain: string; agent: string; dry_run?: boolean }[]): string[] {
  const names = new Set([
    ...versions.filter((v) => v.domain === dataset).map((v) => v.name),
    ...runs.filter((r) => r.domain === dataset && !r.dry_run && r.agent).map((r) => r.agent),
  ]);
  return [...names].sort(byTask);
}

const DEFAULT: Scope = { dataset: 'airline', agent: 'answering', exp: '' };
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
  if (AGENT_TABS.includes(tab) && q.has('exp')) out.exp = q.get('exp') ?? '';
  return out;
}

function qs(o: Record<string, string>): string {
  const p = Object.entries(o).filter(([, v]) => v);
  return p.length ? `?${p.map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&')}` : '';
}

/** The Agent tab's version for a dataset: the experiment when one is set, else the champion, else the newest. */
export function agentVersion(s: Scope, versions: AgentInfo[], registry: Registries): string | null {
  const mine = versions.filter((v) => v.domain === s.dataset);
  if (s.exp && mine.some((v) => v.name === s.exp)) return s.exp;
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
      return judge ? `/evals/${d}/judge${qs({ exp: s.exp })}` : `/evals/${d}`;
    case 'leaderboard':
      return `/leaderboard${qs({ domain: s.dataset })}`;
    case 'runs':
      return judge ? `/runs${qs({ domain: s.dataset, agent: 'judge' })}` : `/runs${qs({ domain: s.dataset, exp: s.exp })}`;
    case 'optimise':
      return judge ? `/optimise/${d}/judge` : `/optimise/${d}${qs({ exp: s.exp })}`;
    case 'review':
      return judge ? `/review/golden/${d}${qs({ exp: s.exp })}` : `/review${qs({ domain: s.dataset, exp: s.exp })}`;
    case 'agent': {
      if (judge) return `/agent/${d}/judge`;
      const v = agents ? agentVersion(s, agents.versions, agents.registry) : null;
      return v ? `/agent/${d}/${encodeURIComponent(v)}` : '/agent';
    }
  }
}

type RunRow = { domain: string; agent: string; dry_run: boolean };
type Ctx = {
  scope: Scope;
  set: (patch: Partial<Scope>) => void;
  href: (tab: Tab) => string;
  tab: Tab;
  /** the dataset's experiments, for the bar's list */
  experiments: (dataset: string) => string[];
  champion: (dataset: string) => string | null;
};
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
  const { data: runs } = useGet<RunRow[]>('/api/runs');
  const experiments = (d: string) => experimentsOf(d, agents?.versions ?? [], runs ?? []);
  const [stored, setStored] = useState<Scope>(readStored);
  const tab = tabOf(loc.pathname);
  const fromLoc = useMemo(() => {
    const out = scopeFromLocation(loc.pathname, loc.search);
    // on the Agent tab, a chosen experiment follows the version opened; with none chosen, opening
    // the champion chooses nothing
    const opened = tab === 'agent' ? decodeURIComponent(loc.pathname.split('/')[3] ?? '') : '';
    if (stored.exp && opened && opened !== 'judge' && opened !== stored.exp && out.exp === undefined) out.exp = opened;
    return out;
    // `stored` changes only through this, so it is not a dependency
  }, [loc.pathname, loc.search]);
  const scope: Scope = { ...stored, ...fromLoc };

  // remember what the address said, for the tabs that say nothing
  useEffect(() => {
    const next = { ...stored, ...fromLoc };
    if (next.dataset !== stored.dataset || next.agent !== stored.agent || next.exp !== stored.exp) {
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
    experiments,
    champion: (d) => agents?.registry[d]?.champion?.agent ?? null,
    set: (patch) => {
      const next = { ...scope, ...patch };
      // another dataset keeps the experiment only if it ran one of that name
      if (patch.dataset && patch.dataset !== scope.dataset && next.exp && !experiments(patch.dataset).includes(next.exp)) next.exp = '';
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

/** The experiment the scope bar has set, or '' for all: what a page filters on. Read from the
 *  scope, so a page opened without `?exp=` still honours the one carried from the last tab. */
export function useExp(): string {
  return useContext(ScopeCtx)?.scope.exp ?? '';
}

/** The bar under the tabs: Dataset everywhere it applies; Agent where both agents have a view; the
 *  experiment where it filters what the tab shows. */
export function ScopeBar() {
  const { scope, set, tab, experiments, champion } = useScope();
  if (tab === 'overview' || tab === 'rubric') return null;
  const withAgent = AGENT_TABS.includes(tab);
  const exps = experiments(scope.dataset);
  const champ = champion(scope.dataset);
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
      {withAgent && expApplies(tab, scope.agent) && (
        <label className="pick" title="a version of the answering agent: its runs, the round that made it, its conversations">
          <span className="label">experiment</span>
          <select value={scope.exp} onChange={(e) => set({ exp: e.target.value })}>
            <option value="">all</option>
            {scope.exp && !exps.includes(scope.exp) && <option value={scope.exp}>{scope.exp} · none here</option>}
            {exps.map((x) => (
              <option key={x} value={x}>
                {x}
                {x === champ ? ' · champion' : ''}
              </option>
            ))}
          </select>
        </label>
      )}
    </div>
  );
}
