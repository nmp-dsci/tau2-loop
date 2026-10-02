import { describe, expect, it } from 'vitest';
import type { Check } from './goldcheck';
import { type Conv, WHOLE, firstBlock, stateOf, toTick, toUntick } from './judgeevals';

const conv = (over: Partial<Conv> = {}): Conv => ({
  key: 'r/6/t1',
  run: 'r',
  version: 'v4',
  model: null,
  task: '6',
  trial: 1,
  passed: false,
  mode: 'wrong_write',
  suspect: false,
  checkpoints: 2,
  writes: 2,
  golden_blocks: 1,
  calls: [
    { msg: 20, kinds: ['write'], names: ['update_reservation_flights'], golden: 'allow' },
    { msg: 50, kinds: ['write'], names: ['book_reservation'], golden: 'block' },
  ],
  first_wrong: { msg: 28, kind: 'reply', blame: 'agent', why: '' },
  has_gold: true,
  review: [20, 50],
  ...over,
});

const check = (item_id: string, verdict: Check['verdict'], correction: Check['correction'] = {}): Check => ({
  id: 1,
  item_id,
  conv_key: item_id.split('#')[0],
  msg: Number(item_id.split('#')[1]),
  verdict,
  correction,
  note: '',
  author: '',
  created_at: '2026-10-02 10:00:00+00',
});

describe('the eval set’s states', () => {
  it('confirms a pass by the grader, allowing every call, with nothing to tick', () => {
    const c = conv({ passed: true, mode: 'pass', review: [] });
    expect(stateOf(c, {}).state).toBe('passed');
    expect(firstBlock(c, {})).toBeNull();
    expect(toTick(c, {})).toEqual([]);
  });

  it('finds the first call to block, and a person’s correction moves it', () => {
    const c = conv();
    expect(firstBlock(c, {})?.msg).toBe(50);
    const moved = { 'r/6/t1#20': check('r/6/t1#20', 'correct', { verdict: 'block' }) };
    expect(firstBlock(c, moved)?.msg).toBe(20);
    const cleared = { 'r/6/t1#50': check('r/6/t1#50', 'correct', { verdict: 'allow' }) };
    expect(firstBlock(c, cleared)).toBeNull();
  });

  it('ticks every unchecked call, and taking it back withdraws agrees but keeps a correction', () => {
    const c = conv();
    const one = { 'r/6/t1#50': check('r/6/t1#50', 'correct', { verdict: 'block', check: 3 }) };
    expect(stateOf(c, one)).toEqual({ state: 'partial', checked: 1, of: 2 });
    expect(toTick(c, one)).toEqual([20]);
    const both = { ...one, 'r/6/t1#20': check('r/6/t1#20', 'agree') };
    expect(stateOf(c, both).state).toBe('confirmed');
    expect(toUntick(c, both)).toEqual([20]);
  });

  it('confirms a failure with no write or transfer as a whole', () => {
    const c = conv({ calls: [], checkpoints: 0, writes: 0, golden_blocks: 0, review: [WHOLE] });
    expect(stateOf(c, {}).state).toBe('todo');
    expect(toTick(c, {})).toEqual([WHOLE]);
    expect(stateOf(c, { 'r/6/t1#-1': check('r/6/t1#-1', 'agree') }).state).toBe('confirmed');
  });

  it('waits on a failure with no golden answer yet', () => {
    expect(stateOf(conv({ has_gold: false, review: [] }), {}).state).toBe('nogold');
  });
});
