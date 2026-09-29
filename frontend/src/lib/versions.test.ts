/**
 * The version figures' arithmetic, pinned: which version is the open challenger, what a
 * verdict reads as, and the width a domain is drawn at. The fixture is the shape
 * `GET /api/versions` returns.
 */
import { describe, expect, it } from 'vitest';
import type { HRun, VersionHistory, VersionNode } from './api';
import { modelWords, standing, verdictWords, versionsWidth } from './versions';

const run = (passed: number, n = 25): HRun => ({ run_id: `r${passed}`, passed, n, cut: 2, model: 'claude-sdk/claude-sonnet-5', effort: 'medium' });
const node = (version: string, extra: Partial<VersionNode> = {}): VersionNode => ({
  version,
  model: 'claude-sdk/claude-sonnet-5',
  effort: 'medium',
  made: { kind: 'optimise', cycle: 1, source: 'v0', detail: 'opus optimiser' },
  train: run(20),
  test: null,
  verdict: 'hold',
  fixed: 2,
  broke: 1,
  p: 0.5,
  vs: null,
  held_title: false,
  ...extra,
});
const hist = (versions: VersionNode[], champion: string): VersionHistory => ({
  domain: 'airline',
  champion,
  versions,
  reigns: [{ version: 'v0', run_id: 'r12', passed: 12, n: 20, cut: 1, kind: 'first', at: null }],
});

describe('standing', () => {
  it('pairs the champion with the newest scored version after it, and the first champion as the baseline', () => {
    const h = hist([node('v0', { verdict: 'first' }), node('v1', { verdict: 'by hand' }), node('v2'), node('v3', { train: null, verdict: null })], 'v1');
    const s = standing(h);
    expect(s.champ?.version).toBe('v1');
    expect(s.chal?.version).toBe('v2');
    expect(s.base?.version).toBe('v0');
  });
  it('has no challenger when the champion is the newest version, and no baseline when it is the first', () => {
    const s = standing(hist([node('v0', { verdict: 'first' })], 'v0'));
    expect(s.chal).toBeNull();
    expect(s.base).toBeNull();
  });
});

describe('verdictWords', () => {
  it('names the tasks fixed and broken beside a cycle verdict, and tones it', () => {
    expect(verdictWords(node('v1'))).toEqual({ text: 'held · +2 −1', tone: 'warn' });
    expect(verdictWords(node('v1', { verdict: 'promote', fixed: 5, broke: 0 }))).toEqual({ text: 'promoted · +5 −0', tone: 'ok' });
    expect(verdictWords(node('v3', { verdict: 'by hand', fixed: null, broke: null })).text).toBe('promoted by hand');
    expect(verdictWords(node('v4', { verdict: null })).text).toBe('not scored');
  });
});

describe('the figure', () => {
  it('draws a model without the provider prefix, and effort only when the run recorded it', () => {
    expect(modelWords(node('v0'))).toBe('sonnet-5 · medium');
    expect(modelWords(node('v0', { model: 'claude-sdk/claude-haiku-4-5', effort: null }))).toBe('haiku-4-5');
  });
  it('widens by a column per version, from a floor', () => {
    expect(versionsWidth(hist([node('v0')], 'v0'))).toBe(480);
    expect(versionsWidth(hist(['v0', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'].map((v) => node(v)), 'v3'))).toBe(84 + 7 * 132);
  });
});
