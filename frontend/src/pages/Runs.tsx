import { Link } from 'react-router-dom';
import { type Check, type Checks, DOMAINS, type RunMeta, domainLabel, fmtK, fmtPct, fmtS, shortModel, shortRun, useGet, when } from '../lib/api';
import { Rate } from '../lib/ui';
import { runPath, useLens } from '../lib/url';

/** The reward checks, in the order τ² multiplies them, with the name a column shows. */
const CHECKS: [keyof Checks, string, string][] = [
  ['db', 'DB', 'the final database equals the gold one'],
  ['actions', 'actions', 'every expected action was called with the expected arguments'],
  ['communicate', 'said', 'the agent told the user each fact the task requires'],
  ['nl', 'NL', "an LLM judge finds the task's NL assertions met"],
  ['env', 'env', "the environment's assertions hold at the end (telecom)"],
];

/** passes over conversations that carry the check; muted when this domain's score does not use it */
function CheckCell({ c, what }: { c?: Check; what: string }) {
  if (!c) return <td className="num muted">—</td>;
  const counted = c.scored > 0;
  const title = `${what}: ${c.passed} of ${c.n} conversations met every item (${c.items_met} of ${c.items} items) · ${
    counted ? `counts toward the score in ${c.scored} of ${c.n}` : 'recorded, not scored in this domain'
  }`;
  return (
    <td className={`num mono${counted ? '' : ' muted'}`} title={title}>
      {c.passed}/{c.n}
    </td>
  );
}

function PassK({ r, k }: { r: RunMeta; k: number }) {
  const v = r.summary?.pass_hat_k[`pass^${k}`];
  if (v == null) return <td className="num muted">—</td>;
  if (k === 1) {
    return (
      <td className="num">
        <Rate passed={r.summary?.passed} n={r.summary?.n_scored} />
      </td>
    );
  }
  return <td className="num mono">{fmtPct(v)}</td>;
}

type Snapshot = { experiment: string | null; tracking_uri?: string; runs: { name: string; tags: Record<string, string>; metrics: Record<string, number>; params: Record<string, string> }[] };

export function Runs() {
  const [lens, setLens] = useLens();
  const { data: runs } = useGet<RunMeta[]>('/api/runs');
  const { data: snap } = useGet<Snapshot>('/api/experiments');
  const domain = lens.get('domain') ?? '';
  const split = lens.get('split') ?? '';
  const q = (lens.get('q') ?? '').toLowerCase();
  const all = (runs ?? []).filter((r) => !r.dry_run).slice().reverse();
  const real = all
    .filter((r) => (domain ? r.domain === domain : true))
    .filter((r) => (split ? r.split === split : true))
    .filter((r) => (q ? `${r.run_id} ${r.agent} ${r.note}`.toLowerCase().includes(q) : true));
  return (
    <>
      <p className="label">Evaluation runs</p>
      <h1>
        A run is a <em>folder</em>: one row per conversation, every trace, the exact agent files
      </h1>
      <p className="lead">
        Each row is <code>runs/&lt;id&gt;/</code> in the repo: <code>results.jsonl</code> with the reward and each component's verdict,{' '}
        <code>traces/</code> with every message, and tau2's own <code>tau2_results.json</code> so the run can be re-scored offline. MLflow indexes
        the same folders; the snapshot below is what the tracker holds, exported for the public demo.
      </p>
      <div className="filters">
        <label className="pick">
          <span className="label">domain</span>
          <select value={domain} onChange={(e) => setLens({ domain: e.target.value })}>
            <option value="">all</option>
            {DOMAINS.map((d) => (
              <option key={d} value={d}>
                {domainLabel(d)}
              </option>
            ))}
          </select>
        </label>
        <label className="pick">
          <span className="label">split</span>
          <select value={split} onChange={(e) => setLens({ split: e.target.value })}>
            <option value="">all</option>
            <option value="train">train</option>
            <option value="test">test</option>
            <option value="custom">custom</option>
          </select>
        </label>
        <input type="search" placeholder="search run, agent, note" value={lens.get('q') ?? ''} onChange={(e) => setLens({ q: e.target.value })} />
        <span className="count">
          {real.length} of {all.length} runs
        </span>
      </div>
      <p className="small muted">
        pass^1 · 2 · 3 are the leaderboard's statistic: the chance that all k trials of a task pass, averaged over its tasks; a run
        of one trial per task has only pass^1. A check's cell is conversations that met every item of it over those that carry
        it; a muted cell is a check τ² records in this domain but does not multiply into the score, and hovering says how many
        items passed. Under a split, "25 × 1 trial · v2" is tasks × trials and the cut: v1 is 20 train / 20 test, v2 half of each
        base set, and two runs on different cuts are not the same tasks.
      </p>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>run</th>
              <th>domain</th>
              <th>agent</th>
              <th>split</th>
              <th className="num" title="the probability that one trial of a task passes, averaged over tasks">
                pass^1
              </th>
              <th className="num" title="the probability that 2 trials of a task all pass; needs a run of 2 or more trials">
                pass^2
              </th>
              <th className="num" title="the probability that 3 trials of a task all pass; needs a run of 3 or more trials">
                pass^3
              </th>
              {CHECKS.map(([key, label, what]) => (
                <th key={key} className="num" title={what}>
                  {label}
                </th>
              ))}
              <th className="num">errors</th>
              <th className="num">turns/conv</th>
              <th className="num">tokens/conv</th>
              <th className="num">time</th>
              <th>note</th>
            </tr>
          </thead>
          <tbody>
            {real.map((r) => (
              <tr key={r.run_id}>
                <td className="sub nw">
                  <Link to={runPath(r.run_id)}>{shortRun(r.run_id)}</Link>
                  <span className="path">{when(r.started_at)}</span>
                  <span className="path">
                    {shortModel(r.model)}
                    {r.agent_route?.startsWith('service') ? ' via service' : ''} · user {shortModel(r.user_model)}
                  </span>
                </td>
                <td>{domainLabel(r.domain)}</td>
                <td className="mono">
                  {r.agent} · {r.fingerprint}
                </td>
                <td className="sub nw">
                  {r.split}
                  <span className="path">
                    {r.n_tasks} × {r.trials} trial{r.trials > 1 ? 's' : ''}
                    {r.split_version ? ` · v${r.split_version}` : ''}
                  </span>
                </td>
                {[1, 2, 3].map((k) => (
                  <PassK key={k} r={r} k={k} />
                ))}
                {CHECKS.map(([key, , what]) => (
                  <CheckCell key={key} c={r.checks?.[key]} what={what} />
                ))}
                <td className="num">{r.summary?.errored_ids.length ?? '—'}</td>
                <td className="num">{r.summary?.mean_agent_turns ?? '—'}</td>
                <td
                  className="num"
                  title={r.tokens_per_conversation ? `agent ${fmtK(r.tokens_per_conversation.agent)} · user ${fmtK(r.tokens_per_conversation.all - r.tokens_per_conversation.agent)}` : undefined}
                >
                  {fmtK(r.tokens_per_conversation?.all)}
                </td>
                <td className="num">{fmtS(r.summary?.duration_ms)}</td>
                <td className="wrap small muted">{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {real.length === 0 && (
        <div className="empty">{all.length ? 'No run matches this filter.' : 'No runs committed yet.'}</div>
      )}

      <h2>
        MLflow snapshot — {snap?.runs.length ?? 0} tracked runs in experiment {snap?.experiment ?? '—'}
      </h2>
      {snap && snap.runs.length > 0 ? (
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>name</th>
                <th>kind</th>
                <th>domain</th>
                <th>agent</th>
                <th className="num">pass_rate</th>
                <th className="num">cost est.</th>
                <th className="num">agent tokens</th>
                <th className="num">user tokens</th>
              </tr>
            </thead>
            <tbody>
              {snap.runs.map((r) => (
                <tr key={r.name}>
                  <td className="sub mono">{shortRun(r.name)}</td>
                  <td>{r.tags.kind ?? '—'}</td>
                  <td>{r.tags.domain ? domainLabel(r.tags.domain) : '—'}</td>
                  <td className="mono">{r.tags.agent ?? r.tags.challenger ?? '—'}</td>
                  <td className="num">{r.metrics.pass_rate != null ? `${Math.round(r.metrics.pass_rate * 100)}%` : '—'}</td>
                  <td className="num">{r.metrics.cost_usd_est != null ? `$${r.metrics.cost_usd_est.toFixed(2)}` : '—'}</td>
                  <td className="num">{r.metrics.agent_input_tokens != null ? `${Math.round(r.metrics.agent_input_tokens).toLocaleString()} / ${Math.round(r.metrics.agent_output_tokens ?? 0).toLocaleString()}` : '—'}</td>
                  <td className="num">{r.metrics.user_input_tokens != null ? `${Math.round(r.metrics.user_input_tokens).toLocaleString()} / ${Math.round(r.metrics.user_output_tokens ?? 0).toLocaleString()}` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty">
          No snapshot committed. <code>make snapshot</code> exports the experiment.
        </div>
      )}
      <p className="small muted">"cost est." is litellm's per-token estimate for the model id; every call ran on the subscription, so the amount paid was $0.</p>
    </>
  );
}
