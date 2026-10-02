/**
 * s11 §9's bars as the judge loop's gate applies them (`tooljudge/loop.py` BARS, served on
 * `GET /api/judge/<domain>` as `bars`), so the viewer never keeps a copy that can drift.
 */

export type BarRule = { op: '<=' | '>='; limit: number };
export type BarRules = Record<string, BarRule>;

/** The most (≤) or fewest (≥) of n a bar allows: the gate's own `bar_count`. */
export function barCount(rule: BarRule, n: number): number {
  return rule.op === '<=' ? Math.floor(rule.limit * n + 1e-9) : Math.ceil(rule.limit * n - 1e-9);
}

/** The bar as a person reads it: "≤ 2 / 50" given the half's n, else the rate or the number. */
export function barText(rule: BarRule, n: number | null): string {
  const op = rule.op === '<=' ? '≤' : '≥';
  if (n) return `${op} ${barCount(rule, n)} / ${n}`;
  return rule.limit <= 1 && n === 0 ? `${op} ${Math.round(rule.limit * 100)}%` : `${op} ${rule.limit}`;
}

export function meetsBar(rule: BarRule, value: number | null): boolean | null {
  if (value == null) return null;
  return rule.op === '<=' ? value <= rule.limit + 1e-12 : value >= rule.limit - 1e-12;
}
