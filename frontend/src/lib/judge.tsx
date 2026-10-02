/**
 * The tool judge on one conversation (plan s11): J0's labels, J1's golden answer, J2's replayed
 * verdicts. Each arrives from `GET /api/runs/<run>/<task>/t<n>` as `judge`, read from committed
 * files; a part a slice has not produced yet is null and draws nothing.
 */

import { type ReactNode, useState } from 'react';
import { Link } from 'react-router-dom';
import { CHECKS, type Check, GoldCheck } from './goldcheck';
import { Points } from './ui';
import { goldReviewPath } from './url';

/** `20260928T075613Z_airline_v4_train` → `075613Z · v4`: the run tag the s11 page prints. */
export function runTag(run: string): string {
  const m = /^\d{8}T(\d{6}Z)_.+?_(v\d+)_/.exec(run);
  return m ? `${m[1]} · ${m[2]}` : run;
}

/** `20260928T075613Z_airline_v4_train` → `v4`: the experiment, the answering agent's version. */
export function runVersion(run: string): string | null {
  return /_(v\d+)_[^_]+$/.exec(run)?.[1] ?? null;
}

export { CHECKS };

export type Checkpoint = {
  msg: number;
  kind: 'plan' | 'reply' | 'write' | 'transfer';
  judge: 'plan' | 'call' | 'direction';
  /** the judge reviews it, before the call runs: a write or a transfer (s11's second rule) */
  judged?: boolean;
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
  /** set when the passed rule decides it, not the annotator or a person */
  by?: 'passed';
};
/** A person's check, as frozen into the gold file (`human`, by message) or live from Postgres. */
type HumanCheck = Pick<Check, 'verdict' | 'correction' | 'note' | 'created_at'>;
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
  /** A person's checks still only in Postgres, by message; the gold file has the frozen ones. */
  checks?: Record<string, Check>;
  writable?: boolean;
  reason?: string;
};
/** What the judge's bar needs to let a person correct a golden answer where they read it. */
export type GoldEdit = { domain: string; onSaved: (c: Check) => void };

export const MODE: Record<string, string> = {
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
  plan: 'the retired plan judge',
  call: 'the tool-call judge',
  direction: 'the tool-call judge',
};

/** A person's newest check at a message: a live one beats the frozen one; `frozen` says the gold
 *  file already holds it. */
export function humanAt(j: JudgeView, msg: number): { h: HumanCheck; frozen: boolean } | null {
  const live = j.checks?.[String(msg)];
  const fz = j.gold?.human?.[String(msg)];
  if (live) return { h: live, frozen: !!fz && fz.created_at === live.created_at };
  return fz ? { h: fz, frozen: true } : null;
}

/** The annotator's answer at a checkpoint, before any person's correction. */
export function annotated(j: JudgeView, cp: Checkpoint): GoldAnswer | null {
  return j.gold?.answer?.checkpoints.find((c) => c.msg === cp.msg && c.kind === cp.kind) ?? null;
}

export const PASSED_WHY = 'the conversation passed: tau2 matched its database to gold’s, so every write and transfer in it was right';

/** The golden verdict at a checkpoint, with the person's correction where there is one; at a write
 *  or transfer of a passed conversation, `allow` by the passed rule, whatever the annotator said. */
export function golden(j: JudgeView, cp: Checkpoint): GoldAnswer | null {
  const a = annotated(j, cp);
  if (j.labels.passed && cp.judged) {
    const base: GoldAnswer = a ?? { id: '', msg: cp.msg, kind: cp.kind, verdict: 'allow', check: null, rule: null, evidence: [], detectable: null, why: '', fix: null };
    return { ...base, verdict: 'allow', check: null, rule: null, evidence: [], fix: null, why: PASSED_WHY, by: 'passed' };
  }
  const c = humanAt(j, cp.msg)?.h;
  const v = c?.verdict === 'correct' ? c.correction.verdict : undefined;
  if (a && (v === 'allow' || v === 'block')) {
    return { ...a, verdict: v, check: v === 'block' ? (c?.correction.check ?? a.check) : null };
  }
  return a;
}

type FirstWrong = { msg: number; kind: string; blame: string; why: string };

/** The golden first wrong step, moved or cleared by a person's newest correction of it, as
 *  `effective_first_wrong` reads the gold file. */
export function goldFirstWrong(j: JudgeView): (FirstWrong & { corrected: boolean }) | null {
  const fw = j.gold?.answer?.first_wrong_step ?? null;
  const all = Object.values({ ...(j.gold?.human ?? {}), ...(j.checks ?? {}) });
  const c = all
    .filter((x) => x.verdict === 'correct' && 'first_wrong_msg' in x.correction)
    .sort((x, y) => y.created_at.localeCompare(x.created_at))[0];
  if (!c) return fw ? { ...fw, corrected: false } : null;
  const m = c.correction.first_wrong_msg;
  if (m == null) return null;
  if (fw && fw.msg === m) return { ...fw, corrected: true };
  const cp = j.labels.checkpoints.find((x) => x.msg === m);
  return { msg: m, kind: cp?.kind ?? 'other', blame: fw?.blame ?? 'agent', why: '', corrected: true };
}

function verdictOf(j: JudgeView, cp: Checkpoint): JudgeVerdict | null {
  return cp.judged ? (j.verdicts?.items.find((v) => v.msg === cp.msg) ?? null) : null;
}

export function Status({ v }: { v: 'allow' | 'block' }) {
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

export function GoldLine({
  a,
  was,
  human,
  first,
  edit,
}: {
  a: GoldAnswer;
  was: GoldAnswer;
  human: { h: HumanCheck; frozen: boolean } | null;
  first: boolean;
  edit: { domain: string; itemId: string; firstWrongNow: number | null; writable: boolean; reason: string; onSaved: (c: Check) => void } | null;
}) {
  const [open, setOpen] = useState(false);
  const h = human?.h;
  const flipped = a.verdict !== was.verdict;
  if (a.by === 'passed') {
    return (
      <p className="small judge-line">
        <span className="chip">golden answer</span> <Status v="allow" /> · the passed rule: {a.why}
        {was.why && was.why !== a.why ? (
          <span className="muted">
            {' '}
            · the annotator said {was.verdict} · {was.why}
          </span>
        ) : null}
      </p>
    );
  }
  return (
    <>
      <p className="small judge-line">
        <span className="chip">golden answer</span> <Status v={a.verdict} />
        {a.check ? ` · check ${a.check}, ${CHECKS[a.check]}` : ''}
        {flipped ? (
          <>
            {' '}
            · <b>a person’s correction</b>
            {h?.note ? `: ${h.note}` : ''}
            <span className="muted">
              {' '}
              · the annotator said {was.verdict} · {was.why}
            </span>
          </>
        ) : (
          <>
            {' '}
            · {a.why}
            {a.verdict === 'block' && a.rule ? (
              <span className="muted"> · rule: {a.rule === 'transcript' ? 'the transcript' : `“${a.rule}”`}</span>
            ) : null}
            {h?.verdict === 'agree' ? <span className="v-ok"> · a person agrees</span> : null}
            {h?.verdict === 'correct' ? <b> · a person’s correction{h.note ? `: ${h.note}` : ''}</b> : null}
          </>
        )}
        {first ? <b> · the golden first wrong step</b> : ''}
        {human && !human.frozen ? <span className="muted"> · in Postgres, not yet in the gold file (make judge-gold-freeze)</span> : null}
        {edit ? (
          <>
            {' '}
            <button type="button" className="linkish gold-edit" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
              {open ? 'close' : h ? 'check it again' : 'correct it'}
            </button>
          </>
        ) : null}
      </p>
      {open && edit ? (
        <GoldCheck
          domain={edit.domain}
          itemId={edit.itemId}
          verdictNow={a.verdict}
          firstWrongNow={edit.firstWrongNow}
          writable={edit.writable}
          reason={edit.reason}
          startCorrecting
          onSaved={(c) => {
            setOpen(false);
            edit.onSaved(c);
          }}
        />
      ) : null}
    </>
  );
}

function VerdictLine({ v, g }: { v: JudgeVerdict; g: GoldAnswer | null }) {
  const r = v.raw;
  return (
    <p className="small judge-line">
      <span className="chip">verdict</span> <Status v={v.verdict} />
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

/** The LLM judge's bar after each checkpoint message: its verdict first, in its own voice, then the
 *  golden answer it is scored against and J0's label. The judge is a second agent, so it gets a bar
 *  of its own in `--judge` purple, never lines inside the answering agent's message. */
export function judgeMarks(j: JudgeView | null | undefined, edit?: GoldEdit): Record<number, ReactNode> {
  if (!j) return {};
  const out: Record<number, { nodes: ReactNode[]; judged: boolean }> = {};
  const fw = j.labels.first_wrong;
  const gfw = goldFirstWrong(j);
  const annotatorFw = j.gold?.answer?.first_wrong_step?.msg ?? null;
  j.labels.checkpoints.forEach((cp, k) => {
    // the judge is called only at a write or a transfer, before it runs; never at a text reply
    if (!cp.judged) return;
    const g = golden(j, cp);
    const was = annotated(j, cp);
    const v = j.verdicts ? verdictOf(j, cp) : null;
    const bar = (out[cp.msg] ??= { nodes: [], judged: false });
    bar.judged ||= !!v;
    const firstNow = gfw?.msg ?? annotatorFw;
    bar.nodes.push(
      v ? <VerdictLine key={`v${k}`} v={v} g={g} /> : null,
      g ? (
        <GoldLine
          key={`g${k}`}
          a={g}
          was={was ?? g}
          human={humanAt(j, cp.msg)}
          first={!!gfw && gfw.msg === cp.msg}
          edit={
            edit && !g.by
              ? {
                  domain: edit.domain,
                  itemId: `${j.labels.key}#${cp.msg}`,
                  // the first-wrong field shows where the annotator put it, so it can be moved back
                  firstWrongNow: annotatorFw === cp.msg || firstNow === cp.msg ? firstNow : null,
                  writable: !!j.writable,
                  reason: j.reason ?? 'no database: corrections need the central Postgres',
                  onSaved: edit.onSaved,
                }
              : null
          }
        />
      ) : null,
      <LabelLine key={`l${k}`} cp={cp} first={!!fw && fw.msg === cp.msg && fw.kind === cp.kind} />,
    );
  });
  const name = j.verdicts?.judge.split('/').pop();
  const who = (judged: boolean) =>
    judged
      ? `tool-call judge ${name}, before the call runs`
      : j.verdicts
        ? 'no verdict at this call'
        : 'before the call runs · no tool-call judge has run yet';
  return Object.fromEntries(
    Object.entries(out).map(([m, bar]) => [
      m,
      <div key={`judge-${m}`} className="ev judge">
        <div className="label">
          LLM judge · {who(bar.judged)}
          <span className="muted"> · on message {m}</span>
        </div>
        {bar.nodes}
      </div>,
    ]),
  );
}

function count(cps: Checkpoint[], kind: Checkpoint['kind'], label: string): number {
  return cps.filter((c) => c.kind === kind && c.label === label).length;
}

/** The headline claim for this conversation: what the judge reviews here, and what gold says of it. */
function claim(j: JudgeView): string {
  const l = j.labels;
  const judged = l.checkpoints.filter((c) => c.judged);
  if (!judged.length) return 'no write or transfer, so the judge is never called';
  const golden_blocks = judged.filter((cp) => golden(j, cp)?.verdict === 'block');
  if (j.verdicts && j.verdicts.items.length) {
    const first = judged.find((cp) => verdictOf(j, cp)?.verdict === 'block');
    if (l.passed) return first ? `the judge would interrupt this pass at message ${first.msg}` : `the judge lets every call through in a pass, ${judged.length} of ${judged.length}`;
    const missed = golden_blocks.filter((cp) => verdictOf(j, cp)?.verdict !== 'block');
    return missed.length ? `the judge misses ${missed.length} of ${golden_blocks.length} calls gold blocks` : `the judge stops all ${golden_blocks.length} calls gold blocks`;
  }
  const n = `${judged.length} call${judged.length === 1 ? '' : 's'}`;
  if (l.passed) return `the grader confirmed this pass, so gold allows the ${n} the judge would review`;
  return golden_blocks.length ? `gold blocks ${golden_blocks.length} of the ${n} the judge would review` : `gold allows the ${n} the judge would review`;
}

export function JudgePanel({ j, domain }: { j: JudgeView; domain: string }) {
  const l = j.labels;
  const cps = l.checkpoints.filter((c) => c.judged);
  const writes = cps.filter((c) => c.kind === 'write');
  const a = j.gold?.answer;
  const fw = goldFirstWrong(j);
  return (
    <>
      <h2>Tool judge — {claim(j)}</h2>
      <Points
        className="small muted"
        items={[
          <>
            <b>The judge reviews each write and transfer before it runs</b>, never a text reply
            {MODE[l.mode] ? `; this conversation: ${MODE[l.mode]}` : ''}.
          </>,
          l.passed ? (
            <>
              <b>The grader confirmed this pass</b>, so by the passed rule the golden answer allows every write and transfer;
              no person checks it.
            </>
          ) : a ? (
            <>
              <b>J1’s annotator wrote the golden answer</b> at each, with gold in view; J0 labels each from the task’s gold actions.
            </>
          ) : null,
          j.verdicts ? (
            <>
              <b>The tool-call judge</b> ({j.verdicts.judge}, {j.verdicts.model}) answered at each, without gold.
            </>
          ) : (
            <>
              <b>No tool-call judge has run yet.</b>
            </>
          ),
        ]}
      />
      {a && (
        <div className="card gold-card">
          <span className="label">
            {j.gold?.annotator.model === 'rule'
              ? 'golden answer · by the passed rule, no model'
              : `golden answer · ${j.gold?.annotator.model} at ${j.gold?.annotator.effort} effort, with gold in view`}
          </span>
          <p className="small">{a.summary}</p>
          {fw && fw.corrected ? (
            <p className="small">
              <b>First wrong step:</b> message {fw.msg}, a {fw.kind}, by a person’s correction
              {a.first_wrong_step ? <span className="muted"> · the annotator said message {a.first_wrong_step.msg}</span> : null}
            </p>
          ) : fw ? (
            <p className="small">
              <b>First wrong step:</b> message {fw.msg}, a {fw.kind}, blamed on the {fw.blame.replace(/_/g, ' ')} · {fw.why}
            </p>
          ) : a.first_wrong_step ? (
            <p className="small">
              <b>First wrong step:</b> none, by a person’s correction
              <span className="muted"> · the annotator said message {a.first_wrong_step.msg}</span>
            </p>
          ) : null}
          {!l.passed && (
            <p className="small muted">
              Correct any golden answer with “correct it” in the judge’s bar below; the queue is in{' '}
              <Link to={goldReviewPath(domain)}>Review › golden answers</Link>.
            </p>
          )}
        </div>
      )}
      <figure>
        <div className="label fig-title">every write and transfer in this conversation: its label, its golden answer, the judge</div>
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th className="num">message</th>
                <th>call</th>
                <th>J0 label</th>
                <th>golden answer</th>
                <th>judge</th>
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
                        <span className="dim">{j.verdicts ? 'not replayed yet' : 'not run yet'}</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <figcaption>
          <Points
            items={[
              <>
                <b>
                  {cps.length} calls the judge reviews
                </b>
                ; by J0, {count(writes, 'write', 'bad')} of {writes.length} writes are bad.
              </>,
              l.suspect ? (
                <>
                  <b>s08 lists it as suspect.</b>
                </>
              ) : null,
            ]}
          />
          <span className="path">
            data/judge/{domain}.json{a ? ` · data/judge/${domain}_gold.jsonl` : ''}
            {j.verdicts ? ` · judge_runs/${j.verdicts.replay_id}/verdicts.jsonl` : ''}
          </span>
        </figcaption>
      </figure>
    </>
  );
}
