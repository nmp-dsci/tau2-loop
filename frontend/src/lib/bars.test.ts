import { describe, expect, it } from 'vitest';
import { barCount, barText, meetsBar } from './bars';

describe('§9’s bars, read from the gate', () => {
  it('states a bar as the count of the half it allows, so it moves with the data', () => {
    const passes = { op: '<=' as const, limit: 2 / 50 };
    expect(barText(passes, 50)).toBe('≤ 2 / 50');
    expect(barCount(passes, 55)).toBe(2);
    const stopped = { op: '>=' as const, limit: 0.5 };
    expect(barText(stopped, 3)).toBe('≥ 2 / 3');
    expect(barText({ op: '>=', limit: 0.73 }, null)).toBe('≥ 0.73');
  });

  it('meets a bar exactly at its limit and misses just past it', () => {
    expect(meetsBar({ op: '<=', limit: 2 / 50 }, 2 / 50)).toBe(true);
    expect(meetsBar({ op: '<=', limit: 2 / 50 }, 3 / 50)).toBe(false);
    expect(meetsBar({ op: '>=', limit: 0.5 }, null)).toBeNull();
  });
});
