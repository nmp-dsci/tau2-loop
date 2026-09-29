/**
 * The pieces every page shares, so DESIGN.md's number rules live in one place
 * instead of in eleven. Ported from DataAgentBench's `lib/ui.tsx` (s04 M0).
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { DOMAINS, type DomainSummary, domainLabel, fmtPct, useGet } from './api';
import { domainPath, taskPath } from './url';

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
      <span className="mono">
        {fmtPct(r, digits)} · {passed}/{n}
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
