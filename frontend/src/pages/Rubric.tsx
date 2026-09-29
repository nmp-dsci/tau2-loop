import { Link } from 'react-router-dom';
import { domainLabel, fmtPct, useGet } from '../lib/api';
import { Kpi, Loading } from '../lib/ui';
import { domainPath } from '../lib/url';

/**
 * How τ² decides a conversation passed — and, across every scored run, which
 * check was the one that failed. DataAgentBench's Validators tab, for a
 * benchmark whose validator is a product of four things rather than a file.
 */

type DomainRubric = {
  domain: string;
  n_tasks: number;
  uses: Record<string, number>;
  reward_bases: Record<string, number>;
};
type Rubric = {
  by_domain: DomainRubric[];
  failures: { failed: number; by_check: Record<string, number>; hit_turn_cap: number };
};

const CHECKS: [string, string, string][] = [
  [
    'db_check',
    'the database',
    'the environment database after the conversation, hashed and compared with the one the annotator recorded. It cannot be argued with, and it is the only check that sees what the agent did rather than what it said.',
  ],
  [
    'actions',
    'expected actions',
    'the write calls the annotator expected, matched by name and arguments. A read-only conversation has none, so the check is vacuous there.',
  ],
  [
    'communicate_info',
    'things that must be said',
    'literal strings that have to appear in what the agent told the user — a refund amount, a confirmation code. Substring matching, so phrasing is free but the value is not.',
  ],
  [
    'nl_assertions',
    'NL assertions',
    'a model judges whether a sentence is true of the transcript. This is the only check with an LLM inside it, and the only one that can be wrong twice — once by missing a real failure, once by inventing one.',
  ],
  [
    'env_assertions',
    'environment assertions',
    'a function run against the final environment. Telecom is judged almost entirely this way — it has no database check at all — which is why its failures look different from the other three.',
  ],
];

export function Rubric() {
  const { data, error } = useGet<Rubric>('/api/rubric');
  if (!data) return <Loading error={error} />;
  const f = data.failures;
  const nl = f.by_check.nl_assertions ?? 0;
  const db = f.by_check.db_check ?? 0;
  const total = data.by_domain.reduce((n, d) => n + d.n_tasks, 0);
  return (
    <>
      <p className="label">The rubric</p>
      <h1>
        A conversation's reward is a <em>product</em>: every check the task names must pass, so one
        miss is a zero
      </h1>
      <p className="lead">
        τ² scores a simulation with <code>evaluate_simulation()</code>. The task's{' '}
        <code>reward_basis</code> names which of the checks below count for it, and the reward is
        their product — which is why a reward is almost always 1.0 or 0.0, and why the interesting
        question is never "what was the score" but "which check failed".
      </p>

      <div className="kpis">
        <Kpi
          n={String(total)}
          b="split tasks across the four domains (40 each, train + test), every one naming its own basis"
        />
        <Kpi n={String(f.failed)} b="failed conversations across every scored run" />
        <Kpi
          n={f.failed ? fmtPct(db / f.failed) : '—'}
          b={`of failures left the database wrong — the check that cannot be argued with`}
          tone="warn"
        />
        <Kpi
          n={f.failed ? fmtPct(nl / f.failed) : '—'}
          b={`of failures missed an NL assertion — the only check with a model inside it`}
        />
      </div>

      <h2>1 · The five checks, and what each can tell you</h2>
      <div className="tw">
        <table>
          <caption>
            How many tasks per domain carry each check. A task with no expected actions is not judged
            on actions at all — and telecom, alone, is judged on environment assertions rather than a
            database hash.
          </caption>
          <thead>
            <tr>
              <th>check</th>
              {data.by_domain.map((d) => (
                <th key={d.domain} className="num">
                  {domainLabel(d.domain)}
                </th>
              ))}
              <th className="num">failed conversations</th>
            </tr>
          </thead>
          <tbody>
            {CHECKS.map(([key, label]) => (
              <tr key={key}>
                <td className="sub">
                  {label}
                  <span className="path">{key}</span>
                </td>
                {data.by_domain.map((d) => (
                  <td key={d.domain} className="num">
                    {d.uses[key] ?? 0}
                    <span className="path">of {d.n_tasks}</span>
                  </td>
                ))}
                <td className="num">
                  {key in f.by_check ? f.by_check[key] : '—'}
                  {key in f.by_check && f.failed > 0 && (
                    <span className="path">{fmtPct((f.by_check[key] ?? 0) / f.failed)}</span>
                  )}
                </td>
              </tr>
            ))}
            <tr className="dim">
              <td className="sub">
                the harness itself
                <span className="path">error</span>
              </td>
              {data.by_domain.map((d) => (
                <td key={d.domain} className="num">
                  —
                </td>
              ))}
              <td className="num">{f.by_check.error ?? 0}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div className="cards">
        {CHECKS.map(([key, label, what]) => (
          <div className="card" key={key}>
            <h3>{label}</h3>
            <p className="small">{what}</p>
          </div>
        ))}
      </div>

      <h2>2 · What the reward is a product of, per domain</h2>
      <p>
        The basis is the task's, not the domain's: within one domain some tasks are judged on the
        database and the conversation, others on the conversation alone.
      </p>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>domain</th>
              <th className="num">tasks</th>
              <th>reward basis (tasks)</th>
            </tr>
          </thead>
          <tbody>
            {data.by_domain.map((d) => (
              <tr key={d.domain}>
                <td className="sub">
                  <Link to={domainPath(d.domain)}>{domainLabel(d.domain)}</Link>
                </td>
                <td className="num">{d.n_tasks}</td>
                <td className="small wrap">
                  {Object.entries(d.reward_bases)
                    .sort((a, b) => b[1] - a[1])
                    .map(([k, v]) => `${k} ${v}`)
                    .join(' · ') || '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>3 · A conversation can miss more than one check</h2>
      <p>
        The counts above do not sum to {f.failed}: a conversation that never made the booking both
        leaves the database wrong and fails to say the confirmation code. What the mix does say is
        which check is worth reading first — and here it is the database, on {fmtPct(db / (f.failed || 1))} of
        failures.{' '}
        {f.hit_turn_cap > 0 ? (
          <>
            {f.hit_turn_cap} conversations never got to answer at all: they ran out of turns.
          </>
        ) : (
          <>No conversation ran out of turns, so no failure here is the turn cap's fault.</>
        )}
      </p>
    </>
  );
}
