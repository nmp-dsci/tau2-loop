import { renderToStaticMarkup } from 'react-dom/server';
import { StaticRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import { type Cycle, JudgeCycles } from './JudgeCycles';

const RUN = '20260915T022857Z_airline_v0_train';
const CYCLE: Cycle = {
  cycle: 1,
  at: '2026-10-01T10:00:00Z',
  champion: { name: 'j1', fingerprint: 'aaa', replay: 'r1' },
  challenger: { name: 'j2', fingerprint: 'bbb', replay: 'r2' },
  optimiser: { model: 'opus', effort: 'high', turns: 30, input_tokens: 400000, duration_ms: 600000, error: null },
  read: { disagreements: 61, now_agree: 20, ids: { d01: `${RUN}/11/t1#6` } },
  diagnosis: 'Most false blocks are replies that mention a tool problem.',
  patterns: [],
  because: { add: [['d01']], edit: [], remove: [] },
  lessons: { added: ['Replies that are not plans: a hedged remark is not a falsehood.'], edited: [], removed: [] },
  verdict: 'held',
  reason: 'balanced accuracy 0.41 → 0.45 (rises) · McNemar one-sided: fixed 3, broke 1, p = 0.312 ≥ α = 0.05',
  gate: {
    promote: false,
    rule: 'none',
    conversations: 58,
    fixed: [{ key: `${RUN}/12/t1`, kind: 'pass' }],
    broken: [],
    p_value: 0.3125,
    balanced_accuracy: { champion: 0.41, challenger: 0.45 },
    read: { champion: 0.47, challenger: 0.55 },
    bars: { balanced_accuracy: { champion: false, challenger: false }, reason_agreement: { champion: null, challenger: true } },
  },
};

const html = (cycles: Cycle[]) =>
  renderToStaticMarkup(
    <StaticRouter location="/optimise/airline/judge">
      <JudgeCycles cycles={cycles} registry={{ champion: 'j1', versions: [] }} />
    </StaticRouter>,
  );

describe('the judge loop’s ledger', () => {
  it('heads with the last cycle’s verdict and its gate numbers, and links each lesson to what it came from', () => {
    const h = html([CYCLE]);
    expect(h).toContain('cycle 1 held j2: gate balanced accuracy 0.41 → 0.45');
    expect(h).toContain('20 of 61 disagreements it read now agree');
    expect(h).toContain('fixed 1, broke 0 of 58');
    expect(h).toContain(`href="/runs/${RUN}/11/t1"`);
    expect(h).toContain('task 11 · message 6');
    expect(h).toContain('The champion now is j1.');
  });

  it('draws nothing before the first cycle, and a rejected cycle without a gate', () => {
    expect(html([])).toBe('');
    const rejected: Cycle = { ...CYCLE, verdict: 'rejected', gate: undefined, reason: 'lesson 1 names a flight number' };
    const h = html([rejected]);
    expect(h).toContain('cycle 1 rejected j2');
    expect(h).toContain('lesson 1 names a flight number');
    expect(h).not.toContain('the gate’s pairs');
  });

  it('marks a cycle that learned from v0’s traces, and states each bar on the gate half’s own n', () => {
    const bars = { passes_interrupted: { op: '<=' as const, limit: 2 / 50 } };
    const draw = (c: Cycle) =>
      renderToStaticMarkup(
        <StaticRouter location="/optimise/airline/judge">
          <JudgeCycles cycles={[c]} registry={null} bars={bars} gateN={{ passes_interrupted: 50 }} />
        </StaticRouter>,
      );
    const gate = { ...CYCLE.gate!, bars: { passes_interrupted: { champion: false, challenger: true } } };
    expect(draw({ ...CYCLE, gate })).toContain('before the optimised-agents rule');
    const clean = draw({ ...CYCLE, gate, agents: ['v1', 'v2', 'v3'] });
    expect(clean).not.toContain('before the optimised-agents rule');
    expect(clean).toContain('passes interrupted ≤ 2 / 50');
  });
});
