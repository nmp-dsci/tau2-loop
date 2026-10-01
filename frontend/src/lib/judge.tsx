/**
 * The tool judge on one conversation (plan s11): J0's labels, J1's golden answer, J2's replayed
 * verdicts. Each arrives from `GET /api/runs/<run>/<task>/t<n>` as `judge`, read from committed
 * files; a part a slice has not produced yet is null and draws nothing.
 */

import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { goldReviewPath } from './url';

/** `20260928T075613Z_airline_v4_train` → `075613Z · v4`: the run tag the s11 page prints. */
export function runTag(run: string): string {
  const m = /^\d{8}T(\d{6}Z)_.+?_(v\d+)_/.exec(run);
  return m ? `${m[1]} · ${m[2]}` : run;
}

export const CHECKS: Record<number, string> = {
  1: 'identity and ownership',
  2: 'arguments trace to the transcript',
  3: 'the policy allows it',
  4: 'the user confirmed exactly this',
  5: 'the user asked for it',
};

export type Checkpoint = {
  msg: number;
  kind: 'plan' | 'reply' | 'write' | 'transfer';
  judge: 'plan' | 'call' | 'direction';
  trigger: boolean;
  live: boolean;
  label: string;
  pinned: boolean;
  writes?: number[];
  name?: string;
  call_id?: string;
};
export type ConvLabels = {
  key: string;
  run: string;
  version: string;
  task: string;
  passed: boolean;
  mode: string;
  fold: number | null;
  half: 'read' | 'gate' | null;
  suspect: boolean;
  checkpoints: Checkpoint[];
  first_wrong: { msg: number; kind: string } | null;
};
type Evidence = { msg: number; quote: string };
export type GoldAnswer = {
  id: string;
  msg: number | null;
  kind: string | null;
  verdict: 'allow' | 'block';
  check: number | null;
  rule: string | null;
  evidence: Evidence[];
  detectable: boolean | null;
  why: string;
  fix: string | null;
};
type HumanCheck = { verdict: 'agree' | 'correct'; correction: { verdict?: string; check?: number }; note: string };
export type GoldRecord = {
  key: string;
  answer: {
    summary: string;
    first_wrong_step: { msg: number; kind: string; blame: string; why: string } | null;
    checkpoints: GoldAnswer[];
  } | null;
  human?: Record<string, HumanCheck>;
  annotator: { model: string; effort: string };
};
export type JudgeVerdict = {
  msg: number;
  kind: string;
  verdict: 'allow' | 'block';
  note: string | null;
  error: string | null;
  raw: {
    is_plan: boolean;
    verdict: 'allow' | 'block';
    confidence: number;
    check: number | null;
    rule: string | null;
    evidence: Evidence[];
    fix: string | null;
    why: string;
  } | null;
};
export type JudgeView = {
  labels: ConvLabels;
  gold: GoldRecord | null;
  verdicts: {
    replay_id: string;
    judge: string;
    model: string;
    threshold: number;
    finished: boolean;
    items: JudgeVerdict[];
  } | null;
};

const MODE: Record<string, string> = {
  pass: 'passed',
  wrong_write: 'a wrong write ran',
  missed_write_transfer: 'a gold write never ran, and the agent transferred',
  missed_write_other: 'a gold write never ran',
  communicate: 'every gold write ran, but something was not said',
};

/** Each label's status glyph and what it means, per DESIGN.md: never colour alone. */
const LABEL: Record<string, [tone: 'ok' | 'err' | 'warn' | 'no', meaning: string]> = {
  'plan:right': ['ok', 'every write it led to matched gold'],
  'plan:wrong': ['err', 'it led to a bad write, and the error was already in it'],
  'plan:slip': ['warn', 'the plan was right; the call departed from it'],
  'reply:allow': ['ok', 'flagged by the plan trigger, not a plan; the conversation passed'],
  'reply:unlabelled': ['no', 'flagged by the plan trigger, not a plan; the golden answer decides it'],
  'write:good': ['ok', 'matches a gold write'],
  'write:bad': ['err', 'matches no gold write'],
  'write:errored': ['no', 'the tool refused it, so nothing changed'],
  'transfer:allow': ['ok', 'the conversation passed, so the hand-off was right'],
  'transfer:candidate': ['no', 'a gold write never ran; a block candidate for J7'],
  'transfer:unlabelled': ['no', 'left to J7'],
};

const WHO: Record<Checkpoint['judge'], string> = {
  plan: 'plan judge',
  call: 'checks.py · call judge later',
  direction: 'J7',
};

/** The golden verdict at a checkpoint, with the person's correction where there is one. */
function golden(j: JudgeView, cp: Checkpoint): GoldAnswer | null {
  const a = j.gold?.answer?.checkpoints.find((c) => c.msg === cp.msg && c.kind === cp.kind) ?? null;
  const h = j.gold?.human?.[String(cp.msg)];
  if (a && h?.verdict === 'correct' && (h.correction.verdict === 'allow' || h.correction.verdict === 'block')) {
    return { ...a, verdict: h.correction.verdict, check: h.correction.check ?? a.check };
  }
  return a;
}

function verdictOf(j: JudgeView, cp: Checkpoint): JudgeVerdict | null {
  return cp.live ? (j.verdicts?.items.find((v) => v.msg === cp.msg) ?? null) : null;
}

function Status({ v }: { v: 'allow' | 'block' }) {
  return <span className={`status ${v === 'block' ? 'err' : 'ok'}`}>{v}</span>;
}

function LabelLine({ cp, first }: { cp: Checkpoint; first: boolean }) {
  const [tone, meaning] = LABEL[`${cp.kind}:${cp.label}`] ?? ['no', ''];
  return (
    <p className="small judge-line">
      <span className="chip">{cp.kind}</span> <span className={`status ${tone}`}>{cp.label}</span> · {meaning}
      {cp.kind === 'plan' && cp.writes?.length ? ` · led to the write at message ${cp.writes.join(', ')}` : ''}
      {cp.kind === 'plan' && !cp.trigger ? ' · the plan trigger missed it' : ''}
      {first ? <b> · the first wrong step</b> : ''}
      <span className="muted"> · J0 label · {WHO[cp.judge]}</span>
    </p>
  );
}

function GoldLine({ a, first }: { a: GoldAnswer; first: boolean }) {
  return (
    <p className="small judge-line">
      <span className="chip">golden answer</span> <Status v={a.verdict} />
      {a.check ? ` · check ${a.check}, ${CHECKS[a.check]}` : ''} · {a.why}
      {a.verdict === 'block' && a.rule ? (
        <span className="muted"> · rule: {a.rule === 'transcript' ? 'the transcript' : `“${a.rule}”`}</span>
      ) : null}
      {first ? <b> · the golden first wrong step</b> : ''}
    </p>
  );
}

function VerdictLine({ v, g, judge }: { v: JudgeVerdict; g: GoldAnswer | null; judge: string }) {
  const r = v.raw;
  const name = judge.split('/').pop();
  return (
    <p className="small judge-line">
      <span className="chip">plan judge {name}</span> <Status v={v.verdict} />
      {v.error ? <span className="v-warn"> · {v.error}</span> : null}
      {r && v.verdict === 'block' ? (
        <>
          {r.check ? ` · check ${r.check}, ${CHECKS[r.check]}` : ''}
          {r.rule ? ` · rule: ${r.rule === 'transcript' ? 'the transcript' : `“${r.rule}”`}` : ''}
          {r.fix ? ` · fix: ${r.fix}` : ''}
        </>
      ) : r ? (
        ` · ${r.why}`
      ) : null}
      {r && v.note ? <span className="muted"> · a block turned into a note: {v.note}</span> : null}
      {r ? <span className="muted"> · confidence {r.confidence.toFixed(2)}</span> : null}
      {g ? (
        <span className={g.verdict === v.verdict ? 'v-ok' : 'v-warn'}>
          {' '}
          · {g.verdict === v.verdict ? 'agrees with' : 'differs from'} the golden answer
        </span>
      ) : null}
    </p>
  );
}

/** The lines on each checkpoint message: J0's label, the golden answer, the judge's verdict. */
export function judgeMarks(j: JudgeView | null | undefined): Record<number, ReactNode> {
  if (!j) return {};
  const out: Record<number, ReactNode[]> = {};
  const fw = j.labels.first_wrong;
  const gfw = j.gold?.answer?.first_wrong_step;
  j.labels.checkpoints.forEach((cp, k) => {
    const g = golden(j, cp);
    const v = verdictOf(j, cp);
    (out[cp.msg] ??= []).push(
      <LabelLine key={`l${k}`} cp={cp} first={!!fw && fw.msg === cp.msg && fw.kind === cp.kind} />,
      g ? <GoldLine key={`g${k}`} a={g} first={!!gfw && gfw.msg === cp.msg} /> : null,
      v && j.verdicts ? <VerdictLine key={`v${k}`} v={v} g={g} judge={j.verdicts.judge} /> : null,
    );
  });
  return Object.fromEntries(Object.entries(out).map(([m, nodes]) => [m, <>{nodes}</>]));
}

function count(cps: Checkpoint[], kind: Checkpoint['kind'], label: string): number {
  return cps.filter((c) => c.kind === kind && c.label === label).length;
}

/** The headline claim for this conversation, from the most-built slice that has run. */
function claim(j: JudgeView): string {
  const l = j.labels;
  const live = l.checkpoints.filter((c) => c.live);
  if (j.verdicts && j.verdicts.items.length) {
    const first = live.find((cp) => verdictOf(j, cp)?.verdict === 'block');
    const gfw = j.gold?.answer?.first_wrong_step;
    if (l.passed) return first ? `the judge would interrupt this pass at message ${first.msg}` : `the judge leaves this pass alone, ${live.length} of ${live.length} allowed`;
    if (first && gfw && first.msg === gfw.msg) return `the judge stops the first wrong step, message ${first.msg}`;
    if (first) return `the judge first blocks message ${first.msg}${gfw ? `; it went wrong at ${gfw.msg}` : ''}`;
    return gfw ? `the judge misses the first wrong step, message ${gfw.msg}` : 'the judge blocks nothing here';
  }
  if (j.gold?.answer?.first_wrong_step) {
    const f = j.gold.answer.first_wrong_step;
    return `golden first wrong step: the ${f.kind} at message ${f.msg}`;
  }
  if (l.first_wrong) return `first wrong step: the ${l.first_wrong.kind} at message ${l.first_wrong.msg}`;
  return l.passed ? `${live.length} plan-judge checkpoints in a pass, all to allow` : `no wrong plan or write: ${MODE[l.mode] ?? l.mode}`;
}

export function JudgePanel({ j, domain }: { j: JudgeView; domain: string }) {
  const l = j.labels;
  const cps = l.checkpoints;
  const plans = cps.filter((c) => c.kind === 'plan');
  const writes = cps.filter((c) => c.kind === 'write');
  const live = cps.filter((c) => c.live);
  const a = j.gold?.answer;
  return (
    <>
      <h2>Tool judge — {claim(j)}</h2>
      <p className="small muted">
        J0 labels every place a judge could stop this conversation, from the task’s gold actions, which no judge
        sees.
        {a ? ' J1’s annotator, who saw gold, wrote the golden answer at each one.' : ''}
        {j.verdicts ? ` J2 replayed the plan judge (${j.verdicts.judge}, ${j.verdicts.model}) at each one it would see live, without gold.` : ''}{' '}
        {MODE[l.mode] ? `This conversation: ${MODE[l.mode]}.` : ''}
      </p>
      {a && (
        <div className="card gold-card">
          <span className="label">golden answer · {j.gold?.annotator.model} at {j.gold?.annotator.effort} effort, with gold in view</span>
          <p className="small">{a.summary}</p>
          {a.first_wrong_step && (
            <p className="small">
              <b>First wrong step:</b> message {a.first_wrong_step.msg}, a {a.first_wrong_step.kind}, blamed on the{' '}
              {a.first_wrong_step.blame.replace(/_/g, ' ')} · {a.first_wrong_step.why}
            </p>
          )}
          <p className="small muted">
            A person checks the cases structure could not pin in <Link to={goldReviewPath(domain)}>Review › golden answers</Link>.
          </p>
        </div>
      )}
      <figure>
        <div className="label fig-title">every checkpoint in this conversation: its label, its golden answer, the judge</div>
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th className="num">message</th>
                <th>checkpoint</th>
                <th>J0 label</th>
                <th>golden answer</th>
                <th>plan judge</th>
              </tr>
            </thead>
            <tbody>
              {cps.map((cp, k) => {
                const [tone] = LABEL[`${cp.kind}:${cp.label}`] ?? ['no'];
                const g = golden(j, cp);
                const v = verdictOf(j, cp);
                return (
                  <tr key={k}>
                    <td className="num">{cp.msg}</td>
                    <td className="sub">
                      {cp.kind}
                      <span className="path">{cp.name ?? WHO[cp.judge]}</span>
                    </td>
                    <td>
                      <span className={`status ${tone}`}>{cp.label}</span>
                    </td>
                    <td>{g ? <Status v={g.verdict} /> : <span className="dim">—</span>}</td>
                    <td className="small">
                      {v ? (
                        <>
                          <Status v={v.verdict} />
                          {g ? <span className={g.verdict === v.verdict ? 'v-ok' : 'v-warn'}> · {g.verdict === v.verdict ? 'agrees' : 'differs'}</span> : null}
                        </>
                      ) : (
                        <span className="dim">{cp.live ? (j.verdicts ? 'not replayed yet' : '—') : 'not its checkpoint'}</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <figcaption>
          {live.length} of {cps.length} checkpoints reach the plan judge; {count(plans, 'plan', 'wrong')} of {plans.length}{' '}
          plans are wrong by J0 and {count(writes, 'write', 'bad')} of {writes.length} writes bad. Task {l.task} sits in fold F
          {l.fold}, the {l.half} half
          {l.half === 'gate' ? ', which the judge loop never reads' : ', which the judge loop reads'}
          {l.suspect ? '; s08 lists it as suspect' : ''}.
          <span className="path">
            data/judge/{domain}.json{a ? ` · data/judge/${domain}_gold.jsonl` : ''}
            {j.verdicts ? ` · judge_runs/${j.verdicts.replay_id}/verdicts.jsonl` : ''}
          </span>
        </figcaption>
      </figure>
    </>
  );
}
