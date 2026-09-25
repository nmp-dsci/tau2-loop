import { NavLink, Outlet } from 'react-router-dom';
import { type Health, useGet } from './lib/api';

/**
 * Eight tabs, in the order the story runs: what the benchmark is → how it is
 * judged → who else has tried → what we ran → how we improve → what the agent
 * is → what we know by hand. The same eight slots as DataAgentBench's explorer
 * (s04 M3–M6).
 */
const NAV: [string, string][] = [
  ['/', 'Overview'],
  ['/domains', 'Domains & tasks'],
  ['/rubric', 'Rubric'],
  ['/leaderboard', 'Leaderboard'],
  ['/runs', 'Runs'],
  ['/optimise', 'Optimise'],
  ['/agent', 'Agent'],
  ['/review', 'Review'],
];

export function Shell() {
  const { data: health } = useGet<Health>('/healthz');
  return (
    <>
      <header className="top">
        <div className="in">
          <NavLink to="/" className="brand">
            <img src="/favicon.svg" alt="" width="22" height="22" />
            <span>
              tau2<b>-loop</b>
            </span>
          </NavLink>
          <nav aria-label="Pages">
            {NAV.map(([to, label]) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) => (isActive ? 'on' : '')}
              >
                {label}
              </NavLink>
            ))}
          </nav>
          <span
            className={`mode ${health?.mode === 'live' ? 'live' : ''}`}
            title="demo serves committed runs and never calls a model"
          >
            {health ? (health.mode === 'demo' ? 'demo · read only' : 'live · dev') : '…'}
          </span>
        </div>
      </header>
      <main>
        <Outlet context={health} />
      </main>
      <footer>
        Every number on these pages is read from a committed file in the repo: <code>runs/</code>,{' '}
        <code>agents/</code>, <code>loop/&lt;domain&gt;/ledger.jsonl</code>,{' '}
        <code>data/splits/</code>. Build <code>{health?.code_sha ?? '…'}</code>.{' '}
        <a href="https://github.com/nmp-dsci/tau2-loop">Source</a> ·{' '}
        <a href="https://github.com/sierra-research/tau2-bench">τ²-bench</a>.
      </footer>
    </>
  );
}
