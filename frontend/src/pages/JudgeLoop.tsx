import { Link, useParams } from 'react-router-dom';
import { domainLabel, fmtK, useGet } from '../lib/api';
import { CHECKS, runTag } from '../lib/judge';
import { Loading } from '../lib/ui';
import { modelFamily } from '../lib/scope';
import { goldReviewPath, judgeLoopPath, trialPath, useLens } from '../lib/url';
import { type Cycle, JudgeCycles, type Registry } from './JudgeCycles';

/**
 * Optimise › judge loop (plan s11, Fig 24): the tool judge's own calibration, beside the agent's
 * loop. J0's labels, J1's golden answers, and every J2 replay scored on them as they stand now, read
 * from committed files (`data/judge/`, `judge_runs/`) so the demo image draws it too. J3's loop adds
 * its ledger (`judges/<domain>/plan/ledger.jsonl`) above the bar; `?replay=` picks the replay the bar
 * and the disagreements are read from.
 */

type Rate = { k: number; n: number; rate: number | null };
type Scores = {
  balanced_accuracy: number | null;
  passes_interrupted: Rate;
  wrong_plans_stopped: Rate;
  wrong_plans_blocked_early: Rate;
  right_plans_blocked: Rate;
  pass_replies_blocked: Rate;
  wrong_plans_blocked: Rate;
  synthetic_blocked: Rate;
  golden_blocks_caught: Rate;
  golden_allows_blocked: Rate;
  detectable_blocks_caught: Rate;
  checkpoint_balanced_accuracy: number | null;
  reason_agreement: Rate;
  confusion: { tp: number; fn: number; fp: number; tn: number };
  checkpoints: number;
  with_gold: number;
  failed_open: number;
  blocks_noted: Rate;
  noted_golden_blocks: Rate;
};
type Disagreement = {
  key: string;
  run: string;
  task: string;
  trial: number;
  msg: number;
  half: string;
  passed: boolean;
  kind: string;
  label: string;
  golden: 'allow' | 'block';
  judge: 'allow' | 'block';
  judge_why: string | null;
  golden_why: string | null;
  judge_check: number | null;
  confidence: number | null;
};
type Replay = {
  replay_id: string;
  judge: string;
  name: string;
  model: string;
  effort: string;
  threshold: number;
  fingerprint: string;
  structured: boolean;
  items: number | { checkpoints: number; synthetic: number };
  started_at: string;
  finished_at: string | null;
  scores: Record<'read' | 'gate' | 'all', Scores>;
  tokens: Record<string, number>;
  gold_records: number;
  gold_human: number;
  disagreements: Disagreement[];
};
type Overview = {
  domain: string;
  labels: {
    summary: { train: { conversations: number; passed: number; plans: Record<string, number>; trigger_fires: Record<string, number> }; synthetic: Record<string, number> };
    folds: Record<string, string[]>;
    halves: Record<string, string[]>;
  } | null;
  gold: {
    records: number;
    conversations: number;
    machine_checks: { pass: number; fail: number };
    review_queue: { items: number };
    human: { reviewed: number; agree: number; correct: number };
    first_wrong_step: Record<string, number>;
    tokens: Record<string, number>;
    annotator: { model: string; effort: string };
  } | null;
  probe: { holds: boolean; sdk: string; model: string } | null;
  replays: Replay[];
  registry: Registry | null;
  cycles: Cycle[];
};

/** One `k / n` with its rate, per DESIGN.md's denominator rule. */
function kn(r: Rate | undefined): string {
  if (!r || !r.n) return '—';
  return `${r.k} / ${r.n} · ${Math.round((r.k / r.n) * 100)}%`;
}

type Bar = { label: string; def: string; get: (s: Scores) => string; meets?: (s: Scores) => boolean | null; bar: string };
const BARS: Bar[] = [
  {
    label: 'balanced accuracy against the golden answers',
    def: 'the mean of passes left alone and wrong-plan conversations stopped at the plan',
    get: (s) => (s.balanced_accuracy == null ? '—' : s.balanced_accuracy.toFixed(2)),
    meets: (s) => (s.balanced_accuracy == null ? null : s.balanced_accuracy >= 0.73),
    bar: '≥ 0.73',
  },
  {
    label: 'passing conversations interrupted',
    def: 'passes in which the judge blocks anything, walked in order',
    get: (s) => kn(s.passes_interrupted),
    meets: (s) => (s.passes_interrupted.n ? s.passes_interrupted.k / s.passes_interrupted.n <= 2 / 55 : null),
    bar: '≤ 2 / 55',
  },
  {
    label: 'wrong plans stopped at the plan',
    def: 'failures whose golden first wrong step is a flagged plan, blocked there first',
    get: (s) => kn(s.wrong_plans_stopped),
    meets: (s) => (s.wrong_plans_stopped.n ? s.wrong_plans_stopped.k / s.wrong_plans_stopped.n >= 0.5 : null),
    bar: '≥ 2 / 4',
  },
  {
    label: 'right plans blocked',
    def: 'plans that led only to good writes, or were right before the call slipped',
    get: (s) => kn(s.right_plans_blocked),
    meets: (s) => (s.right_plans_blocked.n ? s.right_plans_blocked.k / s.right_plans_blocked.n <= 2 / 41 : null),
    bar: '≤ 2 / 41',
  },
  {
    label: 'flagged replies blocked in passes',
    def: 'trigger fires that are not plans, in passing conversations',
    get: (s) => kn(s.pass_replies_blocked),
    meets: (s) => (s.pass_replies_blocked.n ? s.pass_replies_blocked.k / s.pass_replies_blocked.n <= 0.05 : null),
    bar: '≤ 5%',
  },
  {
    label: 'wrong plans blocked',
    def: 'plans J0 labels wrong, each judged on its own',
    get: (s) => kn(s.wrong_plans_blocked),
    meets: (s) => (s.wrong_plans_blocked.n ? s.wrong_plans_blocked.k / s.wrong_plans_blocked.n >= 0.6 : null),
    bar: '≥ 3 / 5',
  },
  {
    label: 'reason agreement',
    def: 'both block: the same check and an evidence message in common',
    get: (s) => kn(s.reason_agreement),
    meets: (s) => (s.reason_agreement.n ? s.reason_agreement.k / s.reason_agreement.n >= 0.7 : null),
    bar: '≥ 70%',
  },
  { label: 'synthetic positives blocked', def: 'right plans with one detail changed so the transcript contradicts it', get: (s) => kn(s.synthetic_blocked), bar: 'reported' },
  { label: 'golden blocks caught, every checkpoint', def: 'flagged replies and plans the golden answer blocks', get: (s) => kn(s.golden_blocks_caught), bar: 'reported' },
  { label: 'golden allows blocked, every checkpoint', def: 'flagged replies and plans the golden answer allows', get: (s) => kn(s.golden_allows_blocked), bar: 'reported' },
  {
    label: 'blocks turned into notes',
    def: 'a raw block under the threshold, or citing a rule not verbatim in the policy, lets the reply through',
    get: (s) => `${s.blocks_noted.k} / ${s.blocks_noted.n}${s.blocks_noted.k ? ` · ${s.noted_golden_blocks.k} golden blocks` : ''}`,
    bar: 'reported',
  },
  { label: 'failed open', def: 'an error or an unparsable verdict, so the reply went through', get: (s) => `${s.failed_open} / ${s.checkpoints}`, bar: 'reported' },
];

export function JudgeLoop() {
  const { domain = 'airline' } = useParams();
  const [lens] = useLens();
  const model = lens.get('model') ?? '';
  const { data, error } = useGet<Overview>(`/api/judge/${encodeURIComponent(domain)}`);
  if (!data) return <Loading error={error} />;
  // the scope bar's model picks the judge's replays on it; `?replay=` names one
  const reps = data.replays.filter((r) => !model || modelFamily(r.model) === model);
  const rep = reps.find((r) => r.replay_id === lens.get('replay')) ?? reps.find((r) => r.finished_at) ?? reps[0];
  const g = rep?.scores.gate;
  const gold = data.gold;
  const lab = data.labels;
  return (
    <>
      <p className="label">The loop · the LLM judge · {domainLabel(domain)}</p>
      {!lab ? (
        <>
          <h1>No tool judge has been calibrated on {domainLabel(domain)}</h1>
          <div className="empty">
            <code>make judge-labels DOMAIN={domain}</code> labels its train conversations from gold; s11’s J8 brings the
            judge here after airline.
          </div>
        </>
      ) : (
        <>
          <h1>
            {g && rep?.finished_at ? (
              <>
                On the gate half, the plan judge stops <em>{g.wrong_plans_stopped.k} of {g.wrong_plans_stopped.n}</em> wrong
                plans and interrupts {g.passes_interrupted.k} of {g.passes_interrupted.n} passes
              </>
            ) : (
              <>
                The plan judge is calibrated on <em>{lab.summary.train.conversations}</em> train conversations before it
                sees a live one
              </>
            )}
          </h1>
          <p className="lead">
            An independent reviewer reads the agent’s plan before the customer does. It is tuned offline: J0 labels every
            checkpoint from gold, J1’s annotator writes a golden answer for each conversation, and each judge version is
            replayed on the same checkpoints without gold and scored against those answers. The gate half (folds{' '}
            {lab.halves.gate.join(', ')}) is the one the judge loop never reads.
          </p>
          <div className="cards">
            <div className="card">
              <span className="label">labels · J0</span>
              <h3>
                {(lab.summary.train.trigger_fires.plans ?? 0) + (lab.summary.train.trigger_fires.replies ?? 0)} checkpoints
              </h3>
              <p className="small">
                the plan trigger’s fires in {lab.summary.train.conversations} train conversations ({lab.summary.train.passed} passed):{' '}
                {lab.summary.train.plans.right ?? 0} right plans, {lab.summary.train.plans.wrong ?? 0} wrong,{' '}
                {lab.summary.train.plans.slip ?? 0} slips, plus{' '}
                {Object.values(lab.summary.synthetic).reduce((a, b) => a + b, 0)} synthetic positives.
              </p>
            </div>
            {gold && (
              <div className="card">
                <span className="label">golden answers · J1</span>
                <h3>
                  {gold.records} of {gold.conversations}
                </h3>
                <p className="small">
                  conversations, by {gold.annotator.model} at {gold.annotator.effort} effort with gold in view;{' '}
                  {gold.machine_checks.pass} of {gold.records} pass every machine check. A person has checked{' '}
                  {gold.human.reviewed} of {gold.review_queue.items} cases in{' '}
                  <Link to={goldReviewPath(domain)}>Review</Link>.
                </p>
              </div>
            )}
            {rep && (
              <div className="card">
                <span className="label">plan judge · {rep.name}</span>
                <h3>{rep.model.replace(/^claude-/, '')}</h3>
                <p className="small">
                  {rep.effort} effort, a block stands at confidence ≥ {rep.threshold}; fingerprint{' '}
                  <span className="mono">{rep.fingerprint}</span>. Verdicts schema-enforced by the SDK
                  {data.probe ? ` (the probe ${data.probe.holds ? 'held' : 'failed'} on ${data.probe.sdk})` : ''}.
                </p>
              </div>
            )}
          </div>

          <JudgeCycles cycles={data.cycles ?? []} registry={data.registry} />

          {reps.length > 1 && (
            <nav className="chips" aria-label="replays">
              {reps.map((r) => (
                <Link
                  key={r.replay_id}
                  className={`chip nav ${r.replay_id === rep?.replay_id ? 'on' : ''}`}
                  to={judgeLoopPath(domain, { replay: r.replay_id, ...(model ? { model } : {}) })}
                >
                  {r.name}
                  <span className="n">
                    {r.name === data.registry?.champion ? 'champion · ' : ''}
                    {r.replay_id.slice(9, 16)}
                  </span>
                </Link>
              ))}
            </nav>
          )}
          {rep && (
            <>
              <h2>
                The bar — {rep.finished_at ? `replay ${rep.replay_id.slice(0, 16)} against §9’s gate-half bars` : `replay ${rep.replay_id.slice(0, 16)} still running`}
              </h2>
              <figure>
                <div className="label fig-title">
                  every bar the judge must clear, read on the gate half; the read half and all train reported beside it
                </div>
                <div className="tw">
                  <table>
                    <thead>
                      <tr>
                        <th>metric</th>
                        <th className="num">gate half</th>
                        <th>bar</th>
                        <th className="num">read half</th>
                        <th className="num">all train</th>
                      </tr>
                    </thead>
                    <tbody>
                      {BARS.map((b) => {
                        const meets = b.meets?.(rep.scores.gate) ?? null;
                        return (
                          <tr key={b.label}>
                            <td className="sub">
                              {b.label}
                              <span className="path">{b.def}</span>
                            </td>
                            <td className="num">{b.get(rep.scores.gate)}</td>
                            <td className="nw">
                              {b.bar}
                              {meets != null && (
                                <>
                                  {' · '}
                                  <span className={`status ${meets ? 'ok' : 'err'}`}>{meets ? 'meets' : 'misses'}</span>
                                </>
                              )}
                            </td>
                            <td className="num">{b.get(rep.scores.read)}</td>
                            <td className="num">{b.get(rep.scores.all)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <figcaption>
                  {rep.name === 'j1' ? 'J2’s first verdicts: j1, untuned. ' : `${rep.name}, from the judge loop. `}
                  {rep.scores.all.checkpoints} checkpoints replayed
                  {typeof rep.items === 'number' ? ` of ${rep.items} items` : ''}, scored on {rep.gold_records} golden answers
                  {rep.gold_human ? ` with ${rep.gold_human} of a person’s checks frozen in` : ''}; input{' '}
                  {fmtK(rep.tokens.input ?? 0)} tokens, {fmtK(rep.tokens.cache_read ?? 0)} of them cache reads. J3’s loop
                  starts from here and is promoted only on the gate half.
                  <span className="path">judge_runs/{rep.replay_id}/verdicts.jsonl · data/judge/{domain}_gold.jsonl → /api/judge/{domain}</span>
                </figcaption>
              </figure>

              <h2>Disagreements — {rep.disagreements.length} checkpoints where the judge and the golden answer differ</h2>
              <figure>
                <div className="label fig-title">each checkpoint the judge got wrong; the read half’s are what J3’s optimiser will read</div>
                <div className="tw">
                  <table>
                    <thead>
                      <tr>
                        <th>conversation</th>
                        <th className="num">message</th>
                        <th>checkpoint</th>
                        <th>golden</th>
                        <th>judge</th>
                        <th>the judge said</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rep.disagreements.map((d) => (
                        <tr key={`${d.key}#${d.msg}`}>
                          <td className="sub mono small">
                            <Link to={trialPath(d.run, `${d.task}/t${d.trial}`)}>
                              {runTag(d.run)} · task {d.task}
                            </Link>
                            <span className="path">
                              {d.half} half · {d.passed ? 'passed' : 'failed'}
                            </span>
                          </td>
                          <td className="num">{d.msg}</td>
                          <td className="small">
                            {d.kind} · {d.label}
                          </td>
                          <td>
                            <span className={`status ${d.golden === 'block' ? 'err' : 'ok'}`}>{d.golden}</span>
                          </td>
                          <td>
                            <span className={`status ${d.judge === 'block' ? 'err' : 'ok'}`}>{d.judge}</span>
                          </td>
                          <td className="wrap small">
                            {d.judge_check ? `check ${d.judge_check}, ${CHECKS[d.judge_check]} · ` : ''}
                            {d.judge_why}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <figcaption>
                  A false block (golden allow, judge block) in a passing conversation interrupts a pass; a miss (golden block,
                  judge allow) lets a wrong step through. Each row opens the conversation, where the label, the golden answer and
                  the verdict sit on the message.
                  <span className="path">judge_runs/{rep.replay_id}/verdicts.jsonl</span>
                </figcaption>
              </figure>
            </>
          )}
        </>
      )}
    </>
  );
}
