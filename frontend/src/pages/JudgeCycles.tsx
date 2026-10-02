import { Link } from 'react-router-dom';
import { fmtK } from '../lib/api';
import { runTag } from '../lib/judge';
import { trialPath } from '../lib/url';

/**
 * Optimise › LLM judge, J3 (s11 §7): the judge loop's ledger. Each cycle is one line of
 * `judges/<domain>/plan/ledger.jsonl`: the lessons a fenced optimiser wrote from the read half's
 * disagreements, and the gate's paired verdict on the gate half, which it never read.
 */

type Bar = { champion: boolean | null; challenger: boolean | null };
export type Cycle = {
  cycle: number;
  at: string;
  champion: { name: string; fingerprint: string; replay: string };
  challenger: { name: string; fingerprint: string | null; replay: string | null };
  optimiser: { model: string; effort: string; turns: number; input_tokens: number; duration_ms: number; error: string | null };
  read: { disagreements: number; now_agree?: number; ids: Record<string, string> };
  diagnosis: string | null;
  patterns: { pattern: string; direction: string; disagreements: string[] }[];
  because: { add: (string[] | null)[]; edit: (string[] | null)[]; remove: (string[] | null)[] };
  lessons: { added: string[]; edited: { n: number; from: string; to: string }[]; removed: { n: number; lesson: string }[] };
  verdict: 'promoted' | 'held' | 'rejected';
  reason: string;
  recheck?: { of: string; was: string; why: string | null };
  gate?: {
    promote: boolean;
    rule: string;
    conversations: number;
    fixed: { key: string; kind: string }[];
    broken: { key: string; kind: string }[];
    p_value: number;
    balanced_accuracy: { champion: number | null; challenger: number | null };
    read: { champion: number | null; challenger: number | null };
    bars: Record<string, Bar>;
  };
};
export type Registry = { champion: string; versions: { name: string; made: string; cycle: number | null; verdict: string | null }[] };

const BAR_NAMES: Record<string, string> = {
  balanced_accuracy: 'balanced accuracy ≥ 0.73',
  passes_interrupted: 'passes interrupted ≤ 2 / 55',
  wrong_plans_stopped: 'wrong plans stopped ≥ 2 / 4',
  right_plans_blocked: 'right plans blocked ≤ 2 / 41',
  pass_replies_blocked: 'flagged replies blocked in passes ≤ 5%',
  wrong_plans_blocked: 'wrong plans blocked ≥ 3 / 5',
  reason_agreement: 'reason agreement ≥ 70%',
};
const TONE = { promoted: 'ok', held: 'warn', rejected: 'err' } as const;
const ba = (x: number | null | undefined) => (x == null ? '—' : x.toFixed(2));

/** `run/task/tN` or `run/task/tN#msg` → a link to that conversation. */
function Conv({ ref_ }: { ref_: string }) {
  const [key, msg] = ref_.split('#');
  const [run, task, trial] = key.split('/');
  return (
    <Link to={trialPath(run, `${task}/${trial}`)}>
      {runTag(run)} · task {task}
      {msg ? ` · message ${msg}` : ''}
    </Link>
  );
}

function Meets({ v }: { v: boolean | null }) {
  if (v == null) return <span className="dim">—</span>;
  return <span className={`status ${v ? 'ok' : 'err'}`}>{v ? 'meets' : 'misses'}</span>;
}

export function JudgeCycles({ cycles, registry }: { cycles: Cycle[]; registry: Registry | null }) {
  if (!cycles.length) return null;
  const c = cycles[cycles.length - 1];
  const g = c.gate;
  const added = c.lessons.added.map((text, i) => ({ text, because: c.because.add[i] ?? [] }));
  return (
    <>
      <h2>
        The loop — cycle {c.cycle} {c.verdict} {c.challenger.name}
        {g ? `: gate balanced accuracy ${ba(g.balanced_accuracy.champion)} → ${ba(g.balanced_accuracy.challenger)}` : ''}
      </h2>
      <p>
        Each cycle an {c.optimiser.model === 'opus' ? 'Opus 5.5' : c.optimiser.model} session, fenced from every replay and the
        gate half, reads the champion’s disagreements with the golden answers on the read half and adds, edits or removes
        numbered lessons. The challenger is replayed on every train checkpoint and paired with the champion conversation by
        conversation on the gate half. It is promoted only when its balanced accuracy there rises, the paired test agrees, and
        it loses no bar the champion meets.{registry ? ` The champion now is ${registry.champion}.` : ''}
      </p>
      <figure>
        <div className="label fig-title">the ledger: one row per cycle</div>
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th className="num">cycle</th>
                <th>judges</th>
                <th>lessons</th>
                <th>read half · what it read</th>
                <th className="num">gate half</th>
                <th>gate conversations</th>
                <th>verdict</th>
              </tr>
            </thead>
            <tbody>
              {cycles.map((e, i) => (
                <tr key={i} className={e.verdict === 'promoted' ? 'pro' : ''}>
                  <td className="num">
                    {e.cycle}
                    {e.recheck && <span className="path">re-check</span>}
                  </td>
                  <td className="mono nw">
                    {e.champion.name} → {e.challenger.name}
                  </td>
                  <td className="small nw">
                    +{e.lessons.added.length} · ~{e.lessons.edited.length} · −{e.lessons.removed.length}
                  </td>
                  <td className="small">
                    {e.gate ? `${ba(e.gate.read.champion)} → ${ba(e.gate.read.challenger)}` : '—'}
                    <span className="path">
                      {e.read.now_agree != null
                        ? `${e.read.now_agree} of ${e.read.disagreements} disagreements it read now agree`
                        : `${e.read.disagreements} disagreements read`}
                    </span>
                  </td>
                  <td className="num">
                    {e.gate ? `${ba(e.gate.balanced_accuracy.champion)} → ${ba(e.gate.balanced_accuracy.challenger)}` : '—'}
                  </td>
                  <td className="small">
                    {e.gate ? (
                      <>
                        fixed {e.gate.fixed.length}, broke {e.gate.broken.length} of {e.gate.conversations}
                        <span className="path">p = {e.gate.p_value.toFixed(3)}</span>
                      </>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td className="wrap small">
                    <span className={`status ${TONE[e.verdict]}`}>{e.verdict}</span>
                    <span className="path">{e.reason}</span>
                    {e.recheck && (
                      <span className="path">
                        the same changes.json, no new session: {e.recheck.why ?? 'after a fix to the harness'}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <figcaption>
          Balanced accuracy is the mean of passing conversations left alone and wrong-plan failures stopped at the plan. A gate
          conversation is fixed when the challenger gets it right and the champion did not. The optimiser{' '}
          {c.optimiser.error ? `ended with an error (${c.optimiser.error}); ` : ''}took {c.optimiser.turns} turns and{' '}
          {fmtK(c.optimiser.input_tokens)} input tokens in cycle {c.cycle}.
          <span className="path">judges/airline/plan/ledger.jsonl · registry.json · src/tau2_loop/tooljudge/loop.py</span>
        </figcaption>
      </figure>

      <h3>
        What cycle {c.cycle} changed — {c.lessons.added.length} lessons added
        {c.lessons.edited.length ? `, ${c.lessons.edited.length} edited` : ''}
        {c.lessons.removed.length ? `, ${c.lessons.removed.length} removed` : ''}
      </h3>
      {c.diagnosis && (
        <details>
          <summary>The optimiser’s diagnosis, in its own words</summary>
          <p className="small">{c.diagnosis}</p>
        </details>
      )}
      {added.length > 0 && (
        <ol className="lessons">
          {added.map((l, i) => (
            <li key={i}>
              {l.text}
              {l.because.length > 0 && (
                <span className="path">
                  from {l.because.length} disagreement{l.because.length > 1 ? 's' : ''}:{' '}
                  {l.because.map((id, k) => (
                    <span key={id}>
                      {k ? ', ' : ''}
                      {c.read.ids[id] ? <Conv ref_={c.read.ids[id]} /> : id}
                    </span>
                  ))}
                </span>
              )}
            </li>
          ))}
        </ol>
      )}
      {c.lessons.edited.map((e) => (
        <p key={`e${e.n}`} className="small">
          <b>lesson {e.n} edited</b> · {e.to}
          <span className="path">was: {e.from}</span>
        </p>
      ))}
      {c.lessons.removed.map((e) => (
        <p key={`r${e.n}`} className="small">
          <b>lesson {e.n} removed</b> · {e.lesson}
        </p>
      ))}

      {g && (
        <div className="grid2">
          <figure>
            <div className="label fig-title">§9’s bars on the gate half, champion and challenger</div>
            <div className="tw">
              <table>
                <thead>
                  <tr>
                    <th>bar</th>
                    <th>{c.champion.name}</th>
                    <th>{c.challenger.name}</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(g.bars).map(([k, b]) => (
                    <tr key={k}>
                      <td className="small">{BAR_NAMES[k] ?? k}</td>
                      <td>
                        <Meets v={b.champion} />
                      </td>
                      <td>
                        <Meets v={b.challenger} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </figure>
          <figure>
            <div className="label fig-title">
              the gate’s pairs: {g.fixed.length} fixed, {g.broken.length} broken of {g.conversations}
            </div>
            <div className="tw">
              <table>
                <thead>
                  <tr>
                    <th>conversation</th>
                    <th>was</th>
                    <th>now</th>
                  </tr>
                </thead>
                <tbody>
                  {[...g.fixed.map((f) => ({ ...f, fixed: true })), ...g.broken.map((f) => ({ ...f, fixed: false }))].map((f) => (
                    <tr key={f.key}>
                      <td className="small mono">
                        <Conv ref_={f.key} />
                        <span className="path">{f.kind === 'pass' ? 'a pass' : 'a wrong-plan failure'}</span>
                      </td>
                      <td>
                        <span className={`status ${f.fixed ? 'err' : 'ok'}`}>{f.fixed ? 'wrong' : 'right'}</span>
                      </td>
                      <td>
                        <span className={`status ${f.fixed ? 'ok' : 'err'}`}>{f.fixed ? 'right' : 'wrong'}</span>
                      </td>
                    </tr>
                  ))}
                  {!g.fixed.length && !g.broken.length && (
                    <tr>
                      <td colSpan={3} className="dim">
                        no conversation changed
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </figure>
        </div>
      )}
    </>
  );
}
