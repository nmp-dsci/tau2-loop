import { describe, expect, it } from 'vitest';
import { countFrac, parseCount, showsActionsDone } from './api';

describe('banking’s second accuracy metric: expected actions done (6 Oct 2026)', () => {
  it('reads a conversation’s count as a fraction between 0 and 1', () => {
    // task_019 in v4's train run: 5 of 6 expected actions, though its reward is 0
    expect(countFrac('5/6')).toBeCloseTo(0.833, 3);
    expect(countFrac('6/6')).toBe(1);
    expect(countFrac('0/4')).toBe(0);
    expect(parseCount('5/6')).toEqual([5, 6]);
  });

  it('has no fraction where nothing was expected or there is no count', () => {
    expect(countFrac('0/0')).toBeNull();
    expect(countFrac(null)).toBeNull();
    expect(countFrac('n/a')).toBeNull();
  });

  it('is shown for banking only', () => {
    expect(showsActionsDone('banking_knowledge')).toBe(true);
    expect(['airline', 'retail', 'telecom', undefined].map(showsActionsDone)).toEqual([false, false, false, false]);
  });
});
