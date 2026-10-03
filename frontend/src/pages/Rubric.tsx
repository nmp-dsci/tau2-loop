import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { domainLabel, fmtPct, useGet } from '../lib/api';
import { Kpi, Loading, Points } from '../lib/ui';
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

const CHECKS: [string, string, ReactNode[]][] = [
  [
    'db_check',
    'the database',
    [
      <>
        <b>The final database, hashed</b> and compared with the annotator's; it cannot be argued
        with.
      </>,
      <>
        <b>The only check that sees what the agent did</b>, not what it said.
      </>,
    ],
  ],
  [
    'actions',
    'expected actions',
    [
      <>
        <b>The write calls the annotator expected</b>, matched by name and arguments.
      </>,
      <>
        <b>Vacuous on a read-only conversation</b>, which expects none.
      </>,
    ],
  ],
  [
    'communicate_info',
    'things that must be said',
    [
      <>
        <b>Literal strings the agent must tell the user</b>: a refund amount, a confirmation code.
      </>,
      <>
        <b>A substring match</b>: the phrasing is free, the value is not.
      </>,
    ],
  ],
  [
    'nl_assertions',
    'NL assertions',
    [
      <>
        <b>A model judges whether a sentence is true</b> of the transcript: the only check with an
        LLM inside.
      </>,
      <>
        <b>It can be wrong twice</b>: by missing a real failure, or by inventing one.
      </>,
    ],
  ],
  [
    'env_assertions',
    'environment assertions',
    [
      <>
        <b>A function run against the final environment.</b>
      </>,
      <>
        <b>Telecom is judged almost entirely this way</b>, with no database check, so its failures
        look different.
      </>,
    ],
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
      <Points
        lead
        items={[
          <>
            <b>The question is which check failed</b>, never what the score was: a reward is almost
            always 1.0 or 0.0.
          </>,
          <>
            <b>
              The task's <code>reward_basis</code> names the checks that count
            </b>
            ; τ²'s <code>evaluate_simulation()</code> multiplies them.
          </>,
        ]}
      />

      <div className="kpis">
        <Kpi
          n={String(total)}
          b="split tasks across the four domains, train + test, each naming its own basis"
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
            Tasks per domain that carry each check; telecom alone is judged on environment assertions,
            not a database hash.
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
            <Points className="small" items={what} />
          </div>
        ))}
      </div>

      <h2>2 · What the reward is a product of, per domain</h2>
      <p>
        The basis is the task's, not the domain's: in one domain, some tasks count the database and
        the conversation, others the conversation alone.
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
      <Points
        items={[
          <>
            <b>Read the database first</b>: it is wrong in {db} of {f.failed} failures
            ({fmtPct(db / (f.failed || 1))}).
          </>,
          <>
            <b>The counts above do not sum to {f.failed}</b>: a missed booking leaves the database
            wrong and the confirmation code unsaid.
          </>,
          f.hit_turn_cap > 0 ? (
            <>
              <b>
                {f.hit_turn_cap} of {f.failed} failures ran out of turns
              </b>{' '}
              before answering at all.
            </>
          ) : (
            <>
              <b>No conversation ran out of turns</b>, so no failure here is the turn cap's fault.
            </>
          ),
        ]}
      />
    </>
  );
}
