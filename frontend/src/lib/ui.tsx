/**
 * The pieces every page shares, so DESIGN.md's number rules live in one place
 * instead of in eleven. Ported from DataAgentBench's `lib/ui.tsx` (s04 M0).
 */

import { type ReactNode, useState } from 'react';
import { Link } from 'react-router-dom';
import { DOMAINS, type DomainSummary, domainLabel, fmtPct, useGet } from './api';
import { domainPath, taskPath } from './url';

/** A paragraph as its points (DESIGN.md rule 8): the most important first, at most three, each a
 *  bold lead and a short clause. `lead` sizes it as the page's lead; in a caption it takes the
 *  caption's size. An empty item is dropped, so a conditional point needs no wrapper. */
export function Points({ items, lead = false, className = '' }: { items: ReactNode[]; lead?: boolean; className?: string }) {
  const shown = items.filter((x) => x != null && x !== false && x !== '');
  return (
    <ul className={['points', lead ? 'lead' : '', className].filter(Boolean).join(' ')}>
      {shown.map((x, i) => (
        <li key={i}>{x}</li>
      ))}
    </ul>
  );
}

/** A pass rate with its denominator in the same cell, per DESIGN.md: never a bare
 *  percentage. `--ok` fills the track, `--amber` when almost nothing passes. */
export function Rate({
  passed,
  n,
  digits = 0,
}: {
  passed: number | null | undefined;
  n: number | null | undefined;
  digits?: number;
}) {
  if (passed == null || !n) return <span className="muted">not scored</span>;
  const r = passed / n;
  return (
    <span className="ratecell" title={`${passed} of ${n} conversations`}>
      <span className="track">
        <i className={r < 0.1 ? 'warn' : ''} style={{ width: `${Math.max(1, r * 100)}%` }} />
      </span>
      {/* breaks only after the ·, where a .tw.fit column is too narrow for one line */}
      <span className="mono">
        {fmtPct(r, digits)}&nbsp;· {passed}/{n}
      </span>
    </span>
  );
}

/** Expected actions made, 0 to 1, with what it counts in the same cell (DESIGN.md: never a bare
 *  number). For one conversation `count` is "5/6"; for a run it is the conversations averaged. */
export function ActionsDone({ frac, count, title }: { frac: number | null | undefined; count: string; title?: string }) {
  if (frac == null) return <span className="muted">—</span>;
  return (
    <span className="ratecell" title={title ?? `${count} expected actions made`}>
      <span className="track">
        <i className={frac < 0.5 ? 'warn' : ''} style={{ width: `${Math.max(1, frac * 100)}%` }} />
      </span>
      <span className="mono">
        {frac.toFixed(2)}&nbsp;· {count.replace(/ /g, '\u00a0')}
      </span>
    </span>
  );
}

export function Kpi({ n, b, tone }: { n: string; b: string; tone?: 'ok' | 'warn' }) {
  return (
    <div className="kpi">
      <div className={`n ${tone ?? ''}`}>{n}</div>
      <div className="b">{b}</div>
    </div>
  );
}

export function Loading({ error }: { error: string | null }) {
  return <p className="empty">{error ? `Could not load: ${error}` : 'Loading…'}</p>;
}

/** Every domain as a lozenge, one click into it. `current` is unset on the index. */
export function DomainChips({ current }: { current?: string }) {
  const { data } = useGet<DomainSummary[]>('/api/domains');
  const counts = Object.fromEntries((data ?? []).map((d) => [d.domain, d.base_n]));
  return (
    <nav className="chips domainbar" aria-label="domains">
      {DOMAINS.map((d) => (
        <Link
          key={d}
          to={domainPath(d)}
          className={`chip nav ${d === current ? 'on' : ''}`}
          aria-current={d === current ? 'page' : undefined}
        >
          {domainLabel(d)}
          {counts[d] != null && <span className="n">{counts[d]}</span>}
        </Link>
      ))}
    </nav>
  );
}

/** A task id as a link to the task, in mono, cut to `n` characters: telecom's run to 90. */
export function TaskLink({ id, n = 36 }: { id: string; n?: number }) {
  const short = id.length > n ? `${id.slice(0, n - 1)}…` : id;
  return (
    <Link to={taskPath(id)} className="mono" title={id}>
      {short}
    </Link>
  );
}

/** Long text clipped at `at` characters, with a toggle that opens the rest in place. */
export function Clip({ text, at }: { text: string; at: number }) {
  const [open, setOpen] = useState(false);
  if (text.length <= at) return <>{text}</>;
  return (
    <>
      {open ? text : `${text.slice(0, at).trimEnd()}…`}
      <button type="button" className="more" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        {open ? 'less' : 'more'}
      </button>
    </>
  );
}
