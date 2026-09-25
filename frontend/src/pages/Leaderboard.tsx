import { Link } from 'react-router-dom';
import { DOMAINS, domainLabel, fmtPct, shortModel, shortRun, useGet, when } from '../lib/api';
import { Kpi, Loading } from '../lib/ui';
import { domainPath, runPath, useLens } from '../lib/url';

/**
 * The published τ²-bench board, and our own runs on the same axis. s04 M6.
 *
 * Nothing here is fetched from a site: the board is a folder of submission.json
 * files in the pinned submodule, ingested to data/index/leaderboard.json by
 * `make leaderboard` and committed, so the app works from a bare clone.
 */

type Score = {
  pass_1: number | null;
  pass_2: number | null;
  pass_3: number | null;
  pass_4: number | null;
  cost: number | null;
  retrieval_config: string | null;
};
type Entry = {
  id: string;
  model: string;
  model_org: string;
  submitted_by: string;
  date: string;
  type: string;
  modality: string;
  user_simulator: string | null;
  tau2_version: string | null;
  notes: string | null;
  modified_prompts: boolean | null;
  omitted_questions: boolean | null;
  trajectories: boolean;
  scores: Record<string, Score>;
  domains: string[];
};
type Ours = {
  run_id: string;
  domain: string;
  agent: string;
  fingerprint: string;
  split: string;
  n_tasks: number;
  trials: number;
  model: string;
  user_model: string;
  pass_hat_k: Record<string, number>;
  passed: number | null;
  n_scored: number | null;
  cost_usd_est: number | null;
  started_at: string;
};
type Board = {
  source: string | null;
  tau2_sha: string | null;
  listed: number;
  entries: Entry[];
  best: Record<string, { pass_1: number; id: string; model: string }>;
  ours: Ours[];
  our_caveats: Record<string, string>;
};

const pct = (x: number | null | undefined) => (x == null ? '—' : `${x.toFixed(1)}%`);

export function Leaderboard() {
  const [lens, setLens] = useLens();
  const { data, error } = useGet<Board>('/api/leaderboard');
  const domain = lens.get('domain') ?? 'airline';
  if (!data) return <Loading error={error} />;
  if (!data.entries.length) {
    return (
      <div className="empty">
        No leaderboard ingested. <code>make leaderboard</code> reads the pinned submodule's
        submissions folder and writes <code>data/index/leaderboard.json</code>.
      </div>
    );
  }

  const rows = data.entries
    .filter((e) => e.scores[domain]?.pass_1 != null)
    .sort((a, b) => (b.scores[domain].pass_1 ?? 0) - (a.scores[domain].pass_1 ?? 0));
  const best = data.best[domain];
  const ours = data.ours
    .filter((o) => o.domain === domain)
    .sort((a, b) => (b.pass_hat_k['pass^1'] ?? 0) - (a.pass_hat_k['pass^1'] ?? 0));
  const bestOurs = ours[0];
  const gap =
    best && bestOurs?.pass_hat_k['pass^1'] != null
      ? best.pass_1 - bestOurs.pass_hat_k['pass^1'] * 100
      : null;

  return (
    <>
      <p className="label">The leaderboard</p>
      <h1>
        Every entry is <em>self-reported</em>: a team runs the harness itself and opens a pull
        request
      </h1>
      <p className="lead">
        There is no holdout server and no submission endpoint. The board at taubench.com renders a
        folder of <code>submission.json</code> files in the upstream repo, one per entry, and a
        maintainer merges the pull request that adds one. That is why{' '}
        <code>methodology.verification</code> matters more than the number beside it: it is the only
        thing separating a standard run from one with rewritten prompts.
      </p>

      <nav className="chips domainbar" aria-label="domains">
        {DOMAINS.map((d) => (
          <button
            key={d}
            type="button"
            className={`chip nav ${d === domain ? 'on' : ''}`}
            onClick={() => setLens({ domain: d })}
          >
            {domainLabel(d)}
            <span className="n">{data.entries.filter((e) => e.scores[d]?.pass_1 != null).length}</span>
          </button>
        ))}
      </nav>

      <div className="kpis">
        <Kpi
          n={best ? pct(best.pass_1) : '—'}
          b={best ? `best published pass^1 on ${domainLabel(domain)} · ${best.model}` : 'no entry'}
        />
        <Kpi
          n={bestOurs ? fmtPct(bestOurs.pass_hat_k['pass^1'], 1) : 'not run'}
          b={
            bestOurs
              ? `our best on ${domainLabel(domain)} · ${bestOurs.agent} · ${bestOurs.n_scored} conversations of our own ${bestOurs.split} split`
              : `no scored run on ${domainLabel(domain)} yet`
          }
          tone={bestOurs ? 'ok' : undefined}
        />
        <Kpi
          n={gap != null ? `${gap > 0 ? '−' : '+'}${Math.abs(gap).toFixed(1)} pts` : '—'}
          b="the distance to the top of the board, on a different task set — indicative, not comparable"
          tone="warn"
        />
        <Kpi n={String(rows.length)} b={`published entries scoring ${domainLabel(domain)}`} />
      </div>

      <h2>1 · The published board on {domainLabel(domain)}</h2>
      <div className="tw">
        <table>
          <caption>
            Ingested from <code>{data.source}</code> at <code>{data.tau2_sha}</code> · {data.listed}{' '}
            submissions listed in the manifest. pass^k is the probability that all k trials of a task
            pass, as the site reports it.
          </caption>
          <thead>
            <tr>
              <th className="num">#</th>
              <th>model</th>
              <th>submitted by</th>
              <th className="num">pass^1</th>
              <th className="num">pass^2</th>
              <th className="num">pass^4</th>
              <th>user simulator</th>
              <th>verification</th>
              <th>date</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((e, i) => (
              <tr key={e.id}>
                <td className="num">{i + 1}</td>
                <td className="sub">
                  {e.model}
                  <span className="path">
                    {e.model_org}
                    {e.scores[domain].retrieval_config
                      ? ` · ${e.scores[domain].retrieval_config}`
                      : ''}
                  </span>
                </td>
                <td className="small">{e.submitted_by}</td>
                <td className="num">{pct(e.scores[domain].pass_1)}</td>
                <td className="num">{pct(e.scores[domain].pass_2)}</td>
                <td className="num">{pct(e.scores[domain].pass_4)}</td>
                <td className="mono small">{e.user_simulator ?? '—'}</td>
                <td className="small">
                  {e.modified_prompts ? (
                    <span className="status warn">modified prompts</span>
                  ) : e.type === 'standard' ? (
                    <span className="status ok">standard</span>
                  ) : (
                    <span className="status no">{e.type}</span>
                  )}
                  {e.trajectories ? <span className="path">trajectories published</span> : null}
                </td>
                <td className="small nw">{e.date}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>2 · Our runs on {domainLabel(domain)}, and why they are not comparable yet</h2>
      {ours.length === 0 ? (
        <div className="empty">
          Nothing scored on {domainLabel(domain)}. <code>make eval DOMAIN={domain}</code> runs it.
        </div>
      ) : (
        <div className="tw">
          <table>
            <caption>
              Our own <code>pass^k</code>, from each run's summary — the same statistic the board
              reports, over a different set of tasks.
            </caption>
            <thead>
              <tr>
                <th>run</th>
                <th>agent</th>
                <th>split</th>
                <th className="num">tasks × trials</th>
                <th className="num">pass^1</th>
                <th className="num">pass^2</th>
                <th className="num">pass^4</th>
                <th>agent · user model</th>
              </tr>
            </thead>
            <tbody>
              {ours.map((o) => (
                <tr key={o.run_id}>
                  <td className="sub">
                    <Link to={runPath(o.run_id)}>{shortRun(o.run_id)}</Link>
                    <span className="path">{when(o.started_at)}</span>
                  </td>
                  <td className="mono">
                    {o.agent}
                    <span className="path">{o.fingerprint}</span>
                  </td>
                  <td>{o.split}</td>
                  <td className="num">
                    {o.n_tasks} × {o.trials}
                  </td>
                  <td className="num">{fmtPct(o.pass_hat_k['pass^1'], 1)}</td>
                  <td className="num">{fmtPct(o.pass_hat_k['pass^2'], 1)}</td>
                  <td className="num">{fmtPct(o.pass_hat_k['pass^4'], 1)}</td>
                  <td className="mono small">
                    {shortModel(o.model)}
                    <span className="path">{shortModel(o.user_model)}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <h2>3 · What a submission of ours would have to declare</h2>
      <p>
        Four differences, each of which the form asks about. Three of them would make the entry
        unverified, and the fourth means the numbers above are not measuring the same thing as the
        board.
      </p>
      <div className="cards">
        <div className="card">
          <h3>The user simulator</h3>
          <p className="small">
            Ours is <code>{data.our_caveats.user_simulator}</code>. The board's entries mostly use
            gpt-5.2. The simulator is half the conversation, so this alone moves a pass rate.
          </p>
        </div>
        <div className="card">
          <h3>Tool calling</h3>
          <p className="small">{data.our_caveats.tool_calling}. The published baselines use native tool calls.</p>
        </div>
        <div className="card">
          <h3>The prompts</h3>
          <p className="small">
            {data.our_caveats.prompts} — which is the point of this project, and which the form
            records as <code>modified_prompts: true</code>.
          </p>
        </div>
        <div className="card">
          <h3>The task set</h3>
          <p className="small">
            {data.our_caveats.split}. A submission must run the whole <code>base</code> set; see{' '}
            <Link to={domainPath(domain)}>{domainLabel(domain)}</Link> for how big that is.
          </p>
        </div>
      </div>
      <p className="small muted">
        The plan for closing that gap is <code>.lavish/s03_eval-submission-plan.html</code>: score
        one domain's full base set at four trials, then prepare and validate a submission.
      </p>
    </>
  );
}
