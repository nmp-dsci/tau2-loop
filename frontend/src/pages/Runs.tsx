import { Link } from 'react-router-dom';
import { type RunMeta, domainLabel, fmtS, shortModel, shortRun, useGet, when } from '../lib/api';

type Snapshot = { experiment: string | null; tracking_uri?: string; runs: { name: string; tags: Record<string, string>; metrics: Record<string, number>; params: Record<string, string> }[] };

export function Runs() {
  const { data: runs } = useGet<RunMeta[]>('/api/runs');
  const { data: snap } = useGet<Snapshot>('/api/experiments');
  const real = (runs ?? []).filter((r) => !r.dry_run).slice().reverse();
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
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>run</th>
              <th>domain</th>
              <th>agent</th>
              <th>split</th>
              <th className="num">pass</th>
              <th className="num">pass^k</th>
              <th className="num">errors</th>
              <th className="num">turns/conv</th>
              <th className="num">time</th>
              <th>note</th>
            </tr>
          </thead>
          <tbody>
            {real.map((r) => (
              <tr key={r.run_id}>
                <td className="sub">
                  <Link to={`/runs/${r.run_id}`}>{shortRun(r.run_id)}</Link>
                  <span className="path">
                    {when(r.started_at)} · {shortModel(r.model)} · user {shortModel(r.user_model)}
                  </span>
                </td>
                <td>{domainLabel(r.domain)}</td>
                <td className="mono">
                  {r.agent} · {r.fingerprint}
                </td>
                <td>
                  {r.split}
                  {r.trials > 1 && <span className="path">{r.trials} trials</span>}
                </td>
                <td className="num">{r.summary?.n_scored ? `${r.summary.passed}/${r.summary.n_scored}` : '—'}</td>
                <td className="num small">
                  {r.summary
                    ? Object.entries(r.summary.pass_hat_k)
                        .map(([k, v]) => `${k} ${Math.round(v * 100)}%`)
                        .join(' · ')
                    : '—'}
                </td>
                <td className="num">{r.summary?.errored_ids.length ?? '—'}</td>
                <td className="num">{r.summary?.mean_agent_turns ?? '—'}</td>
                <td className="num">{fmtS(r.summary?.duration_ms)}</td>
                <td className="wrap small muted">{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {real.length === 0 && <div className="empty">No runs committed yet.</div>}

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
