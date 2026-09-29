/**
 * The version history as figures (s09), ported from DataAgentBench's `lib/versions.tsx`:
 * every version of a domain in build order with how it was made, the champion across
 * the domains, and every cycle's champion against its challenger. One source,
 * `/api/versions`, which joins the agents, the ledger, the registry and the run folders.
 *
 * Colour follows DESIGN.md: `--ok` is a champion or a promotion (passed), `--amber` a
 * held challenger (a caveat), grey everything else; every mark also carries its word.
 */
import type { KeyboardEvent, ReactNode } from 'react';
import { type HRun, type Made, type Reign, type VersionHistory, type VersionNode, domainLabel, shortModel } from './api';

export const share = (r: HRun | null | undefined): number | null => (r && r.n ? r.passed / r.n : null);
export const frac = (r: HRun | null | undefined): string => (r ? `${r.passed}/${r.n}` : '—');

const onKey = (go: () => void) => (e: KeyboardEvent) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    go();
  }
};

/** The gate's verdict as the figures word it, with its tone: `ok` passed, `warn` held. */
export function verdictWords(v: VersionNode): { text: string; tone: 'ok' | 'warn' | '' } {
  const moved = v.fixed != null ? ` · +${v.fixed} −${v.broke ?? 0}` : '';
  switch (v.verdict) {
    case 'promote':
      return { text: `promoted${moved}`, tone: 'ok' };
    case 'hold':
      return { text: `held${moved}`, tone: 'warn' };
    case 'first':
      return { text: 'first champion', tone: 'ok' };
    case 'by hand':
      return { text: 'promoted by hand', tone: 'ok' };
    case null:
      return { text: 'not scored', tone: '' };
    default:
      return { text: v.verdict, tone: '' };
  }
}

/** `model: sonnet → opus` reads `sonnet → opus` under a column; a second agent.yaml line is cut. */
function madeWords(m: Made): { kind: string; detail: string; from: string } {
  const first = m.detail.split('; ')[0].replace(/^model: /, '');
  const detail = m.detail.includes('; ') || first.length > 22 ? `${first.slice(0, 21)}…` : first;
  const from = [m.cycle != null ? `cycle ${m.cycle}` : null, m.source ? `from ${m.source}` : null].filter(Boolean).join(' · ');
  return { kind: m.kind, detail, from };
}

/** `sonnet-5 · medium`; older runs recorded no effort. */
export function modelWords(v: VersionNode): string {
  return [v.model ? shortModel(v.model) : null, v.effort].filter(Boolean).join(' · ');
}

/** Contiguous runs of versions on the same cut, so no line joins numbers over different tasks. */
function cuts(vs: VersionNode[]): { from: number; to: number; cut: number | null; n: number | null }[] {
  const out: { from: number; to: number; cut: number | null; n: number | null }[] = [];
  vs.forEach((v, i) => {
    const last = out[out.length - 1];
    const c = v.train?.cut ?? last?.cut ?? null;
    if (last && last.cut === c) {
      last.to = i;
      last.n ??= v.train?.n ?? null;
    } else out.push({ from: i, to: i, cut: c, n: v.train?.n ?? null });
  });
  return out;
}

/** The champion's version and the newest scored version after it, if any. */
export function standing(h: VersionHistory): { champ: VersionNode | null; chal: VersionNode | null; base: Reign | null } {
  const i = h.versions.findIndex((v) => v.version === h.champion);
  const champ = h.versions[i] ?? null;
  const newer = i < 0 ? [] : h.versions.slice(i + 1).filter((v) => v.train);
  const first = h.reigns[0];
  return { champ, chal: newer[newer.length - 1] ?? null, base: first && first.version !== h.champion && first.passed != null && first.n ? first : null };
}

/** a column per version; the figure's min-width keeps 13px labels at ≥ 12.8px (--t-2) when rendered */
const COL = 132;

/** The width `VersionsFig` draws a domain at, so a row of figures can size each to fit. */
export const versionsWidth = (h: VersionHistory): number => Math.max(480, 84 + h.versions.length * COL);

/**
 * Fig · every version of one domain in build order, one column each; under each, how it
 * was made and what the gate did with it.
 * `champion` (the Runs tab): train pass^1, filled where the version held the title, and
 * a line joining the champions in the order they took it — a re-baseline is a second
 * point in the same column. `rounds` (the Optimise tab): train (filled) and test (open),
 * each joined within a cut, and a tick at the champion's train score each challenger faced.
 */
export function VersionsFig({
  h,
  mode,
  open = null,
  onPick,
}: {
  h: VersionHistory;
  mode: 'champion' | 'rounds';
  open?: string | null;
  onPick: (v: VersionNode) => void;
}) {
  const vs = h.versions;
  const lx = 60;
  const W = versionsWidth(h);
  const top = 64;
  const bot = 244;
  const H = bot + 148;
  const X = (i: number) => lx + 12 + COL / 2 + i * COL;
  const Y = (s: number) => bot - (bot - top) * s;
  const col = new Map(vs.map((v, i) => [v.version, i]));
  const segs = cuts(vs);
  const train = vs.map((v) => share(v.train));
  const test = vs.map((v) => share(v.test));
  const pts = (vals: (number | null)[], from: number, to: number) =>
    vals
      .map((s, i) => (i >= from && i <= to && s != null ? `${X(i)},${Y(s)}` : null))
      .filter(Boolean)
      .join(' ');
  // the champions, in the order they took the title
  const rp = h.reigns.flatMap((r) => {
    const i = col.get(r.version);
    return i != null && r.passed != null && r.n ? [{ r, x: X(i), y: Y(r.passed / r.n) }] : [];
  });
  const lastReignOf = new Map(rp.map((p, k) => [p.r.version, k]));
  /** Where the champion line crosses column `i`, if it does: a label goes on the other side. */
  const lineAt = (i: number): number | null => {
    const x = X(i);
    for (let k = 1; k < rp.length; k++) {
      const [a, b] = [rp[k - 1], rp[k]];
      if (a.x < x && x < b.x) return a.y + ((b.y - a.y) * (x - a.x)) / (b.x - a.x);
    }
    return null;
  };
  const aria =
    mode === 'champion'
      ? `Every ${domainLabel(h.domain)} version in build order with its train pass^1; a line joins the versions that held the title, from ${rp[0]?.r.version ?? '—'} to ${h.champion ?? '—'}; under each version, how it was made and the gate's verdict`
      : `Every ${domainLabel(h.domain)} version in build order with its train pass^1 (filled) and test pass^1 (open), and a tick at the champion's train score each challenger was gated against; under each version, how it was made and the gate's verdict`;
  const hit = (v: VersionNode, i: number): ReactNode => {
    const go = () => onPick(v);
    const w = verdictWords(v);
    return (
      <g
        key={v.version}
        id={`ver-${h.domain}-${v.version}`}
        className={`pick${open === v.version ? ' on' : ''}`}
        role="button"
        tabIndex={0}
        aria-label={`${v.version}: train ${frac(v.train)}, test ${frac(v.test)}; ${v.made.kind}; ${w.text}`}
        onClick={go}
        onKeyDown={onKey(go)}
      >
        <title>
          {v.version} ({modelWords(v) || 'no run'}): train {frac(v.train)} · test {frac(v.test)}
          {v.vs ? ` · gated against ${v.vs.version} at ${frac(v.vs.train)} train${v.vs.test ? `, ${frac(v.vs.test)} test` : ''}` : ''} · {v.made.kind}
          {v.made.detail ? `, ${v.made.detail}` : ''} · {w.text}
          {v.p != null ? `, p = ${v.p.toFixed(3)}` : ''}
          {v.train?.run_id ? `\ntrain run ${v.train.run_id}` : ''}
          {v.test?.run_id ? `\ntest run ${v.test.run_id}` : ''}
        </title>
        <rect className="hit" x={X(i) - COL / 2 + 4} y={top - 22} width={COL - 8} height={H - top + 14} rx={6} />
      </g>
    );
  };
  return (
    <div className="figscroll">
      <svg className="dia" viewBox={`0 0 ${W} ${H}`} style={{ maxWidth: W, minWidth: Math.ceil(W * 0.985) }} role="img" aria-label={aria}>
        <title>{mode === 'champion' ? `The ${domainLabel(h.domain)} champion over the versions` : `Every ${domainLabel(h.domain)} version, train and test`}</title>
        {/* hit areas first, so a hovered column's band sits under the marks */}
        {vs.map(hit)}
        <g style={{ pointerEvents: 'none' }}>
          {[0, 0.25, 0.5, 0.75, 1].map((g) => (
            <g key={g}>
              <line className="gl" x1={lx} x2={W - 12} y1={Y(g)} y2={Y(g)} style={{ strokeDasharray: g ? '2 4' : undefined }} />
              <text className="tx k" x={lx - 8} y={Y(g) + 4} textAnchor="end">
                {g * 100}%
              </text>
            </g>
          ))}
          {segs.map((s, k) => (
            <g key={k}>
              {k > 0 && <line className="ax" x1={X(s.from) - COL / 2} x2={X(s.from) - COL / 2} y1={top - 44} y2={bot + 6} style={{ strokeDasharray: '4 3' }} />}
              <text className="tx s" x={X(s.from) - COL / 2 + 10} y={top - 30}>
                {s.cut ? `split v${s.cut}` : 'split —'}
                {s.n ? ` · ${s.n} train tasks` : ''}
              </text>
            </g>
          ))}
          {mode === 'rounds' &&
            segs.map((s, k) => (
              <g key={k}>
                <polyline fill="none" style={{ stroke: 'var(--line-3)', strokeWidth: 1.5, strokeDasharray: '5 4' }} points={pts(test, s.from, s.to)} />
                <polyline fill="none" style={{ stroke: 'var(--ink-2)', strokeWidth: 2 }} points={pts(train, s.from, s.to)} />
              </g>
            ))}
          {mode === 'champion' &&
            rp.slice(1).map((p, k) => (
              <line key={k} x1={rp[k].x} y1={rp[k].y} x2={p.x} y2={p.y} style={{ stroke: 'var(--ok)', strokeWidth: 3, strokeDasharray: p.r.cut !== rp[k].r.cut ? '6 5' : undefined }} />
            ))}
          {mode === 'champion' &&
            rp.map((p, k) =>
              lastReignOf.get(p.r.version) === k ? null : (
                // an earlier reign of a version that took the title again (a re-baseline)
                <g key={`r${k}`}>
                  <circle cx={p.x} cy={p.y} r={5} style={{ fill: 'var(--ok)' }} />
                  <text className="tx s" x={p.x} y={p.y + 22} textAnchor="middle">
                    {p.r.passed}/{p.r.n} · {p.r.kind}
                  </text>
                </g>
              ),
            )}
          {vs.map((v, i) => {
            const t = train[i];
            const s = test[i];
            const champ = v.version === h.champion;
            const m = madeWords(v.made);
            const w = verdictWords(v);
            const hand = v.made.kind !== 'optimise' && v.made.kind !== 'base';
            const trainAbove = s == null || t == null || t >= s;
            const faced = mode === 'rounds' ? share(v.vs?.train) : null;
            const crossing = mode === 'champion' ? lineAt(i) : null;
            return (
              <g key={v.version}>
                {faced != null && <line x1={X(i) - 18} x2={X(i) + 18} y1={Y(faced)} y2={Y(faced)} style={{ stroke: 'var(--ok)', strokeWidth: 3 }} />}
                {mode === 'rounds' && s != null && (
                  <>
                    <circle cx={X(i)} cy={Y(s)} r={7.5} style={{ fill: 'var(--panel)', stroke: 'var(--ink-2)', strokeWidth: 2 }} />
                    <text className="tx s" x={X(i)} y={trainAbove ? Y(s) + 26 : Y(s) - 15} textAnchor="middle">
                      test {frac(v.test)}
                    </text>
                  </>
                )}
                {t != null &&
                  (mode === 'rounds' ? (
                    <>
                      <circle cx={X(i)} cy={Y(t)} r={5.5} style={{ fill: v.held_title ? 'var(--ok)' : 'var(--ink-2)' }} />
                      <text className={`tx${v.held_title ? ' ok' : ''}`} x={X(i)} y={trainAbove ? Y(t) - 15 : Y(t) + 26} textAnchor="middle">
                        {frac(v.train)}
                      </text>
                    </>
                  ) : v.held_title ? (
                    <>
                      <circle cx={X(i)} cy={Y(t)} r={8} style={{ fill: 'var(--ok)' }} />
                      <text className="tx ok" x={X(i)} y={Y(t) - 16} textAnchor="middle">
                        {frac(v.train)}
                      </text>
                    </>
                  ) : (
                    <>
                      <circle cx={X(i)} cy={Y(t)} r={6} style={{ fill: 'var(--panel)', stroke: 'var(--ink-2)', strokeWidth: 2 }} />
                      <text className="tx s" x={X(i)} y={(crossing != null ? crossing > Y(t) : t < 0.12) ? Y(t) - 14 : Y(t) + 26} textAnchor="middle">
                        {frac(v.train)}
                      </text>
                    </>
                  ))}
                <text className="tx" x={X(i)} y={bot + 28} textAnchor="middle" style={champ || open === v.version ? { fontWeight: 700 } : undefined}>
                  {v.version}
                  {champ ? ' ★' : ''}
                </text>
                <text className="tx s" x={X(i)} y={bot + 46} textAnchor="middle">
                  {modelWords(v) || 'no run'}
                </text>
                {v.made.kind !== 'base' && (
                  <line x1={X(i) - 40} x2={X(i) + 40} y1={bot + 60} y2={bot + 60} style={{ stroke: hand ? 'var(--line-3)' : 'var(--ink-2)', strokeDasharray: hand ? '4 3' : undefined, strokeWidth: 2 }} />
                )}
                <text className="tx" x={X(i)} y={bot + 80} textAnchor="middle">
                  {m.kind}
                </text>
                <text className="tx s" x={X(i)} y={bot + 98} textAnchor="middle">
                  {m.detail}
                </text>
                {m.from && (
                  <text className="tx s" x={X(i)} y={bot + 116} textAnchor="middle">
                    {m.from}
                  </text>
                )}
                <text className={`tx${w.tone ? ` ${w.tone}` : ''}`} x={X(i)} y={bot + 136} textAnchor="middle">
                  {w.text}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}

/**
 * Fig · per domain, the champion and the newest challenger after it: train pass^1 as a
 * bar, test as an open ring on the same track, and a dashed tick at the domain's first
 * champion. A row opens the run behind its bar.
 */
export function DomainBars({ hs, onPick }: { hs: VersionHistory[]; onPick: (domain: string, v: VersionNode) => void }) {
  const W = 1000;
  const x0 = 16;
  const lab = 250;
  const tx0 = 262;
  const tw = 440;
  const rx = tx0 + tw + 16;
  const rowH = 28;
  const gap = 20;
  const top = 34;
  let y = top;
  const groups = hs.flatMap((h) => {
    const s = standing(h);
    if (!s.champ) return [];
    const rows = [
      { role: 'champion', v: s.champ },
      ...(s.chal ? [{ role: 'challenger', v: s.chal }] : []),
    ];
    const g = { h, rows, base: s.base, y };
    y += Math.max(rows.length, 2) * rowH + gap;
    return [g];
  });
  const H = y - gap + 14;
  const X = (s: number) => tx0 + tw * s;
  return (
    <div className="figscroll">
      <svg className="dia" viewBox={`0 0 ${W} ${H}`} style={{ minWidth: Math.ceil(W * 0.985) }} role="img" aria-label="Per domain, the champion's and the newest challenger's train pass^1 as bars and test pass^1 as open rings, with the domain's first champion as a dashed tick">
        <title>Champion and challenger by domain</title>
        {groups.flatMap((g) =>
          g.rows.map((row, k) => {
            const go = () => onPick(g.h.domain, row.v);
            return (
              <g key={`${g.h.domain}-${row.role}`} className="pick" role="button" tabIndex={0} aria-label={`${domainLabel(g.h.domain)} ${row.role} ${row.v.version}: train ${frac(row.v.train)}, test ${frac(row.v.test)}`} onClick={go} onKeyDown={onKey(go)}>
                <title>
                  {domainLabel(g.h.domain)} · {row.role} {row.v.version} ({modelWords(row.v)}): train {frac(row.v.train)} · test {frac(row.v.test)} · {verdictWords(row.v).text}
                  {row.v.train?.run_id ? `\n${row.v.train.run_id}` : ''}
                </title>
                <rect className="hit" x={8} y={g.y + k * rowH - 4} width={W - 16} height={rowH} rx={6} />
              </g>
            );
          }),
        )}
        <g style={{ pointerEvents: 'none' }}>
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <g key={f}>
              <line className="gl" x1={X(f)} x2={X(f)} y1={top - 8} y2={H - 8} />
              <text className="tx k" x={X(f)} y={top - 14} textAnchor="middle">
                {f * 100}%
              </text>
            </g>
          ))}
          {groups.map((g) => {
            const c = g.rows[0].v;
            return (
              <g key={g.h.domain} id={`dom-${g.h.domain}`}>
                <text className="tx" x={x0} y={g.y + 14} style={{ fontWeight: 600 }}>
                  {domainLabel(g.h.domain)}
                </text>
                <text className="tx s" x={x0} y={g.y + 14 + rowH}>
                  split v{c.train?.cut ?? '—'} · {c.train?.n ?? '—'} train
                </text>
                {g.rows.map((row, k) => {
                  const yy = g.y + k * rowH;
                  const t = share(row.v.train);
                  const s = share(row.v.test);
                  const w = verdictWords(row.v);
                  return (
                    <g key={row.role}>
                      <text className="tx" x={lab} y={yy + 14} textAnchor="end">
                        {row.role} {row.v.version}
                      </text>
                      <rect className="trk" x={tx0} y={yy + 3} width={tw} height={14} rx={3} />
                      {t != null && <rect className={`bar${row.role === 'champion' ? ' ok' : ''}`} x={tx0} y={yy + 3} width={Math.max(2, t * tw)} height={14} rx={3} />}
                      {s != null && <circle cx={X(s)} cy={yy + 10} r={6} style={{ fill: 'var(--panel)', stroke: 'var(--ink)', strokeWidth: 2 }} />}
                      <text className="tx s" x={rx} y={yy + 14}>
                        train {frac(row.v.train)} · test {row.v.test ? frac(row.v.test) : 'not run'}
                        {row.role === 'challenger' && <tspan style={{ fill: w.tone === 'ok' ? 'var(--ok)' : w.tone === 'warn' ? 'var(--amber)' : undefined }}> · {w.text.split(' · ')[0]}</tspan>}
                      </text>
                    </g>
                  );
                })}
                {g.base && g.base.passed != null && g.base.n ? (
                  <line x1={X(g.base.passed / g.base.n)} x2={X(g.base.passed / g.base.n)} y1={g.y - 2} y2={g.y + g.rows.length * rowH - 6} style={{ stroke: 'var(--line-3)', strokeWidth: 2, strokeDasharray: '3 3' }} />
                ) : null}
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}

/**
 * Fig · every cycle in every domain: the champion's train pass^1 (open) against the
 * challenger's (filled), and the same pair on test below it when both ran it. A green
 * edge promoted, an amber one held. A row opens the round.
 */
export function CycleDumbbell({
  hs,
  open = null,
  onPick,
}: {
  hs: VersionHistory[];
  open?: { domain: string; version: string } | null;
  onPick: (domain: string, v: VersionNode) => void;
}) {
  const W = 1000;
  const x0 = 16;
  const lab = 262;
  const tx0 = 276;
  const tw = 440;
  const rx = tx0 + tw + 16;
  const rowH = 46;
  const gap = 16;
  const top = 34;
  const X = (s: number) => tx0 + tw * s;
  let y = top;
  const groups = hs.map((h) => {
    const rows = h.versions.filter((v) => v.made.cycle != null).sort((a, b) => (a.made.cycle ?? 0) - (b.made.cycle ?? 0));
    const g = { h, rows, y };
    y += Math.max(rows.length, 1) * rowH + gap;
    return g;
  });
  const H = y - gap + 10;
  const tone = (v: VersionNode) => (v.verdict === 'promote' ? 'var(--ok)' : v.verdict === 'hold' ? 'var(--amber)' : 'var(--ink-2)');
  return (
    <div className="figscroll">
      <svg className="dia" viewBox={`0 0 ${W} ${H}`} style={{ minWidth: Math.ceil(W * 0.985) }} role="img" aria-label="Every loop cycle in every domain: the champion's train pass^1 as an open dot against the challenger's as a filled dot, and the same pair on test below when both ran it; green where the challenger was promoted, amber where it was held">
        <title>Every cycle, champion against challenger</title>
        {groups.flatMap((g) =>
          g.rows.map((v, k) => {
            const go = () => onPick(g.h.domain, v);
            const on = open?.domain === g.h.domain && open.version === v.version;
            return (
              <g key={`${g.h.domain}-${v.version}`} id={`cyc-${g.h.domain}-${v.made.cycle}`} className={`pick${on ? ' on' : ''}`} role="button" tabIndex={0} aria-label={`${domainLabel(g.h.domain)} cycle ${v.made.cycle}, ${v.vs?.version ?? '—'} against ${v.version}: train ${frac(v.vs?.train)} to ${frac(v.train)}; ${verdictWords(v).text}`} onClick={go} onKeyDown={onKey(go)}>
                <title>
                  {domainLabel(g.h.domain)} cycle {v.made.cycle} · {v.vs?.version ?? '—'} → {v.version} ({v.made.kind}
                  {v.made.detail ? `, ${v.made.detail}` : ''}): train {frac(v.vs?.train)} → {frac(v.train)} · test {frac(v.vs?.test)} → {frac(v.test)} · {verdictWords(v).text}
                  {v.p != null ? `, p = ${v.p.toFixed(3)}` : ''}
                </title>
                <rect className="hit" x={8} y={g.y + k * rowH - 6} width={W - 16} height={rowH - 2} rx={6} />
              </g>
            );
          }),
        )}
        <g style={{ pointerEvents: 'none' }}>
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <g key={f}>
              <line className="gl" x1={X(f)} x2={X(f)} y1={top - 8} y2={H - 8} />
              <text className="tx k" x={X(f)} y={top - 14} textAnchor="middle">
                {f * 100}%
              </text>
            </g>
          ))}
          {groups.map((g) => (
            <g key={g.h.domain} id={`cycles-${g.h.domain}`}>
              <text className="tx" x={x0} y={g.y + 14} style={{ fontWeight: 600 }}>
                {domainLabel(g.h.domain)}
              </text>
              {g.rows.length === 0 && (
                <>
                  <text className="tx s" x={lab} y={g.y + 14} textAnchor="end">
                    no cycle yet
                  </text>
                  <text className="tx s" x={rx} y={g.y + 14}>
                    {g.h.champion ?? '—'} at {frac(standing(g.h).champ?.train)} train, the first champion
                  </text>
                </>
              )}
              {g.rows.map((v, k) => {
                const yy = g.y + k * rowH;
                const a = share(v.vs?.train);
                const b = share(v.train);
                const ta = share(v.vs?.test);
                const tb = share(v.test);
                const c = tone(v);
                const w = verdictWords(v);
                const m = madeWords(v.made);
                return (
                  <g key={v.version}>
                    <text className="tx" x={lab} y={yy + 14} textAnchor="end">
                      cycle {v.made.cycle} · {v.vs?.version ?? '—'} → {v.version}
                    </text>
                    <text className="tx s" x={lab} y={yy + 32} textAnchor="end">
                      {v.made.kind === 'optimise' ? m.detail : `${m.kind}: ${m.detail}`}
                    </text>
                    {a != null && b != null && <line x1={X(a)} x2={X(b)} y1={yy + 10} y2={yy + 10} style={{ stroke: c, strokeWidth: 3 }} />}
                    {a != null && <circle cx={X(a)} cy={yy + 10} r={6} style={{ fill: 'var(--panel)', stroke: 'var(--ink-2)', strokeWidth: 2 }} />}
                    {b != null && <circle cx={X(b)} cy={yy + 10} r={6} style={{ fill: c }} />}
                    {ta != null && tb != null && <line x1={X(ta)} x2={X(tb)} y1={yy + 28} y2={yy + 28} style={{ stroke: c, strokeWidth: 1.5, strokeDasharray: '4 3' }} />}
                    {ta != null && <circle cx={X(ta)} cy={yy + 28} r={4.5} style={{ fill: 'var(--panel)', stroke: 'var(--ink-2)', strokeWidth: 1.5 }} />}
                    {tb != null && <circle cx={X(tb)} cy={yy + 28} r={4.5} style={{ fill: c }} />}
                    <text className="tx" x={rx} y={yy + 14}>
                      {v.vs?.train?.passed ?? '—'} → {v.train?.passed ?? '—'} of {v.train?.n ?? '—'} train ·{' '}
                      <tspan style={{ fill: w.tone === 'ok' ? 'var(--ok)' : w.tone === 'warn' ? 'var(--amber)' : undefined }}>{w.text}</tspan>
                    </text>
                    <text className="tx s" x={rx} y={yy + 32}>
                      {v.test ? `test ${v.vs?.test ? `${v.vs.test.passed} → ` : ''}${v.test.passed} of ${v.test.n}` : 'no test run'}
                    </text>
                  </g>
                );
              })}
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}
