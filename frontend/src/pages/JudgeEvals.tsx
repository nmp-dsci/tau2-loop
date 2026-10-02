import { Link, useParams } from 'react-router-dom';
import { domainLabel, useGet } from '../lib/api';
import { runTag } from '../lib/judge';
import { MODELS, modelFamily } from '../lib/scope';
import { Kpi, Loading } from '../lib/ui';
import { goldReviewPath, judgeLoopPath, trialPath, useLens } from '../lib/url';

/**
 * Evals, with the scope bar on the LLM judge (s11): what the judge is scored on. Not tasks, as for
 * the answering agent, but the agent's own train conversations: every checkpoint the judge would see
 * live, each with a golden answer written with gold in view, dealt into five task folds so the
 * judge loop reads one half and is gated on the other. Test conversations never appear here.
 * Read from `data/judge/<domain>.json` and `data/judge/<domain>_gold.jsonl`.
 */

type Conv = {
  key: string;
  run: string;
  version: string;
  model: string | null;
  task: string;
  trial: number;
  fold: number;
  half: 'read' | 'gate';
  passed: boolean;
  mode: string;
  suspect: boolean;
  checkpoints: number;
  plans: number;
  golden_blocks: number;
  first_wrong: { msg: number; kind: string; blame: string; why: string } | null;
  has_gold: boolean;
  human: number;
};
type Synth = {
  id: string;
  key: string;
  msg: number;
  task: string;
  fold: number;
  half: string;
  kind: string;
  what: string;
  change: { from: string; to: string };
};
type Evals = {
  domain: string;
  conversations: Conv[] | null;
  folds?: Record<string, string[]>;
  halves?: Record<string, string[]>;
  synthetic?: Synth[];
  gold?: { records: number; review_queue: { items: number }; human: { reviewed: number } } | null;
};

const MODE: Record<string, string> = {
  pass: 'passed',
  wrong_write: 'a wrong write ran',
  missed_write_transfer: 'missed a write, transferred',
  missed_write_other: 'missed a write',
  communicate: 'did not say something',
};

function convPath(c: { run: string; task: string; trial?: number }): string {
  return trialPath(c.run, `${c.task}/t${c.trial ?? 1}`);
}

export function JudgeEvals() {
  const { domain = 'airline' } = useParams();
  const [lens, setLens] = useLens();
  const model = lens.get('model') ?? '';
  const { data, error } = useGet<Evals>(`/api/judge/${encodeURIComponent(domain)}/evals`);
  if (!data) return <Loading error={error} />;
  if (!data.conversations) {
    return (
      <>
        <p className="label">Evals · the LLM judge · {domainLabel(domain)}</p>
        <h1>No LLM judge has an eval set on {domainLabel(domain)} yet</h1>
        <div className="empty">
          <code>make judge-labels DOMAIN={domain}</code> labels its train conversations from gold, and{' '}
          <code>make judge-gold DOMAIN={domain}</code> writes their golden answers. s11’s J8 brings the judge here after airline.
        </div>
      </>
    );
  }
  const half = lens.get('half') ?? '';
  const show = lens.get('show') ?? '';
  const q = (lens.get('q') ?? '').trim();
  const all = data.conversations.filter((c) => !model || modelFamily(c.model) === model);
  const rows = all
    .filter((c) => !half || c.half === half)
    .filter((c) => (show === 'passed' ? c.passed : show === 'failed' ? !c.passed : show === 'blocks' ? c.golden_blocks > 0 : true))
    .filter((c) => !q || c.task === q || c.version === q || c.key.includes(q));
  const cps = all.reduce((n, c) => n + c.checkpoints, 0);
  const blocks = all.reduce((n, c) => n + c.golden_blocks, 0);
  const plans = all.reduce((n, c) => n + c.plans, 0);
  const passed = all.filter((c) => c.passed).length;
  const synth = (data.synthetic ?? []).filter((s) => !half || s.half === half);
  const label = MODELS.find(([k]) => k === model)?.[1];
  const folds = Object.entries(data.folds ?? {});

  return (
    <>
      <p className="label">Evals · the LLM judge · {domainLabel(domain)}</p>
      <h1>
        The LLM judge is scored on <em>{cps}</em> checkpoints in {all.length} train conversations, each with a golden answer
      </h1>
      <p className="lead">
        The answering agent is evaluated on tasks; the judge is evaluated on the answering agent’s conversations. Every
        checkpoint it would see live, a plan or another reply the plan trigger flags, carries a golden answer that an
        annotator wrote with gold in view. Only train conversations are here: the test split is never shown to the judge or
        its loop.{label ? ` Filtered to conversations ${label} wrote.` : ''}
      </p>
      <div className="kpis">
        <Kpi n={`${passed} / ${all.length}`} b="conversations that passed; the judge must leave these alone" tone="ok" />
        <Kpi n={`${plans} / ${cps}`} b="checkpoints that are plans; the rest are other replies the plan trigger flags" />
        <Kpi n={`${blocks} / ${cps}`} b="checkpoints the golden answer blocks: what a perfect judge would stop" tone="warn" />
        <Kpi
          n={String((data.synthetic ?? []).length)}
          b="synthetic positives: right plans with one detail changed so the conversation contradicts them"
        />
      </div>
      {data.gold && (
        <p className="small muted">
          {data.gold.records} golden answers; a person has checked {data.gold.human.reviewed} of the{' '}
          {data.gold.review_queue.items} cases structure could not pin, in{' '}
          <Link to={goldReviewPath(domain)}>Review</Link>. The judge’s score on this set is under{' '}
          <Link to={judgeLoopPath(domain)}>Optimise</Link>.
        </p>
      )}

      <h2>The folds — the judge loop reads one half and is gated on the other</h2>
      <figure>
        <div className="label fig-title">five folds by task, so a task never sits on both sides</div>
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>fold</th>
                <th>half</th>
                <th>tasks</th>
                <th className="num">conversations passed · failed</th>
                <th className="num">checkpoints</th>
                <th className="num">golden blocks</th>
                <th className="num">first wrong step at a plan</th>
              </tr>
            </thead>
            <tbody>
              {folds.map(([f, tasks]) => {
                const n = Number(f.slice(1));
                const fc = all.filter((c) => c.fold === n);
                const h = fc[0]?.half ?? (data.halves?.gate.includes(f) ? 'gate' : 'read');
                return (
                  <tr key={f} className={half && h !== half ? 'dim' : ''}>
                    <td className="sub mono">{f}</td>
                    <td>{h === 'gate' ? 'gate · never read by the loop' : 'read'}</td>
                    <td className="mono small">{tasks.join(', ')}</td>
                    <td className="num">
                      {fc.filter((c) => c.passed).length} · {fc.filter((c) => !c.passed).length}
                    </td>
                    <td className="num">{fc.reduce((k, c) => k + c.checkpoints, 0)}</td>
                    <td className="num">{fc.reduce((k, c) => k + c.golden_blocks, 0)}</td>
                    <td className="num">{fc.filter((c) => c.first_wrong?.kind === 'plan').length}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <figcaption>
          Tasks were dealt to folds by their count of wrong plans and slips, most first, in snake order. A task recurs in
          up to 10 runs, so splitting by run would put it on both sides.
          <span className="path">data/judge/{domain}.json (folds, halves) · make judge-labels</span>
        </figcaption>
      </figure>

      <h2>The conversations — {rows.length} of {all.length} shown</h2>
      <div className="filters">
        <label className="pick">
          <span className="label">half</span>
          <select value={half} onChange={(e) => setLens({ half: e.target.value })}>
            <option value="">both</option>
            <option value="read">read (F1–F3)</option>
            <option value="gate">gate (F4–F5)</option>
          </select>
        </label>
        <label className="pick">
          <span className="label">show</span>
          <select value={show} onChange={(e) => setLens({ show: e.target.value })}>
            <option value="">all</option>
            <option value="passed">passed</option>
            <option value="failed">failed</option>
            <option value="blocks">with a golden block</option>
          </select>
        </label>
        <input type="search" placeholder="task id, version or run" value={q} onChange={(e) => setLens({ q: e.target.value })} />
        <span className="count">
          {rows.length} of {all.length} conversations
        </span>
      </div>
      <div className="tw">
        <table>
          <caption>Each conversation opens with its checkpoints, golden answers and the judge’s verdicts on its messages.</caption>
          <thead>
            <tr>
              <th>conversation</th>
              <th>fold</th>
              <th>outcome</th>
              <th className="num">checkpoints</th>
              <th className="num">golden blocks</th>
              <th>golden first wrong step</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => (
              <tr key={c.key}>
                <td className="sub mono small">
                  <Link to={convPath(c)}>
                    {runTag(c.run)} · task {c.task}
                  </Link>
                  <span className="path">
                    {(c.model ?? '').replace(/^claude-sdk\//, '').replace(/^claude-/, '')}
                    {c.suspect ? ' · s08 suspect' : ''}
                  </span>
                </td>
                <td className="small">
                  F{c.fold} · {c.half}
                </td>
                <td>
                  <span className={`status ${c.passed ? 'ok' : 'err'}`}>{MODE[c.mode] ?? c.mode}</span>
                </td>
                <td className="num">{c.checkpoints}</td>
                <td className="num">{c.golden_blocks}</td>
                <td className="wrap small">
                  {c.first_wrong ? (
                    <>
                      message {c.first_wrong.msg}, a {c.first_wrong.kind}
                      <span className="path">blamed on the {c.first_wrong.blame.replace(/_/g, ' ')}</span>
                    </>
                  ) : c.passed ? (
                    <span className="dim">none: it passed</span>
                  ) : (
                    <span className="dim">no golden answer yet</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {synth.length > 0 && (
        <>
          <h2>Synthetic positives — {synth.length} right plans, each with one detail changed</h2>
          <figure>
            <div className="label fig-title">each a real plan whose conversation now contradicts it; the golden answer is block</div>
            <div className="tw">
              <table>
                <thead>
                  <tr>
                    <th>id</th>
                    <th>source</th>
                    <th>what changed</th>
                    <th>from → to</th>
                    <th>fold</th>
                  </tr>
                </thead>
                <tbody>
                  {synth.map((s) => {
                    const [run, task, trial] = s.key.split('/');
                    return (
                      <tr key={s.id}>
                        <td className="sub mono">{s.id}</td>
                        <td className="mono small">
                          <Link to={convPath({ run, task, trial: Number(trial.slice(1)) })}>
                            {runTag(run)} · task {task}
                          </Link>
                          <span className="path">message {s.msg}</span>
                        </td>
                        <td className="small">{s.what}</td>
                        <td className="wrap small mono">
                          {s.change.from || '—'} → {s.change.to || '(dropped)'}
                        </td>
                        <td className="small">
                          F{s.fold} · {s.half}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <figcaption>
              Real wrong plans are few, so these add catches to score. They are reported
              apart from the real ones and kept in their source task’s fold.
              <span className="path">data/judge/{domain}.json (synthetic) · tooljudge/labels.py (synthetic_positives)</span>
            </figcaption>
          </figure>
        </>
      )}
    </>
  );
}
