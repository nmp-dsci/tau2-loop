import type { Check } from './goldcheck';

/**
 * Where each conversation of the LLM judge's eval set stands (s11): the passed rule, a person's
 * ticks, and the first call the judge must block. Pure, so Evals' table and its open card agree.
 *
 * The passed rule: a passed conversation is confirmed by the grader. tau2 matched its database to
 * gold's, so every write and transfer in it was right, and no person checks it. A failed one waits
 * for a person at each call, or as a whole (message −1) when it has none, since the judge is never
 * called there.
 */

/** One message holding a write or a transfer the judge reviews, and its golden verdict as frozen. */
export type Call = { msg: number; kinds: string[]; names: string[]; golden: 'allow' | 'block' | null };

export type Conv = {
  key: string;
  run: string;
  version: string;
  model: string | null;
  task: string;
  trial: number;
  passed: boolean;
  mode: string;
  suspect: boolean;
  /** the writes and transfers the judge reviews */
  checkpoints: number;
  writes: number;
  golden_blocks: number;
  calls: Call[];
  first_wrong: { msg: number; kind: string; blame: string; why: string } | null;
  has_gold: boolean;
  /** what a person confirms: each call in a failure, or [−1], the failure as a whole */
  review: number[];
};

/** The message number of a check on the conversation as a whole. */
export const WHOLE = -1;

export const itemId = (c: Conv, msg: number) => `${c.key}#${msg}`;

/** The golden verdict at a call, with a person's correction still only in Postgres laid over it. */
export function verdictAt(c: Conv, call: Call, current: Record<string, Check>): 'allow' | 'block' | null {
  if (c.passed) return 'allow';
  const x = current[itemId(c, call.msg)];
  const v = x?.verdict === 'correct' ? x.correction.verdict : undefined;
  return v === 'allow' || v === 'block' ? v : call.golden;
}

/** The first call the golden answer blocks: where the judge must stop this failure, or null. */
export function firstBlock(c: Conv, current: Record<string, Check>): Call | null {
  return c.calls.find((call) => verdictAt(c, call, current) === 'block') ?? null;
}

export type State = 'passed' | 'nogold' | 'confirmed' | 'partial' | 'todo';

/** Where a conversation stands: passed (the grader confirmed it), or a person's progress on it. */
export function stateOf(c: Conv, current: Record<string, Check>): { state: State; checked: number; of: number } {
  const of = c.review.length;
  if (c.passed) return { state: 'passed', checked: 0, of };
  if (!c.has_gold) return { state: 'nogold', checked: 0, of };
  const checked = c.review.filter((m) => current[itemId(c, m)]).length;
  return { state: checked === of ? 'confirmed' : checked ? 'partial' : 'todo', checked, of };
}

/** What a tick writes: an agree at every message not yet checked. */
export const toTick = (c: Conv, current: Record<string, Check>) => c.review.filter((m) => !current[itemId(c, m)]);

/** What taking a tick back withdraws: the agrees only; a person's correction stays. */
export const toUntick = (c: Conv, current: Record<string, Check>) =>
  c.review.filter((m) => current[itemId(c, m)]?.verdict === 'agree');
