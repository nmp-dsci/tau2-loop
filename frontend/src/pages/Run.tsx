import { Link, useParams } from 'react-router-dom';
import { type RunMeta, type TaskResult, domainLabel, fmtS, shortModel, shortTask, useGet, when } from '../lib/api';
import { agentPath, trialId, trialPath } from '../lib/url';

function Verdict({ r }: { r: TaskResult }) {
  if (r.correct === true) return <span className="status ok">pass</span>;
  if (r.correct === false) return <span className="status err">fail</span>;
  return <span className="status no">unscored</span>;
}

function Components({ r }: { r: TaskResult }) {
  const parts: string[] = [];
  if (r.db_check != null) parts.push(`DB ${r.db_check ? '✓' : '✗'}`);
  if (r.env_assertions) parts.push(`env ${r.env_assertions}`);
  if (r.action_checks) parts.push(`actions ${r.action_checks}`);
  if (r.communicate_checks) parts.push(`said ${r.communicate_checks}`);
  if (r.nl_assertions) parts.push(`NL ${r.nl_assertions}`);
  return <span className="mono small">{parts.join(' · ') || '—'}</span>;
}

export function Run() {
  const { runId = '' } = useParams();
  const { data, error } = useGet<{ meta: RunMeta; results: TaskResult[] }>(`/api/runs/${runId}`);
  if (error) return <div className="empty">{error}</div>;
  if (!data) return <p className="muted">loading…</p>;
  const { meta, results } = data;
  const s = meta.summary;
  const aIn = results.reduce((n, r) => n + r.agent_input_tokens, 0);
  const aOut = results.reduce((n, r) => n + r.agent_output_tokens, 0);
  const uIn = results.reduce((n, r) => n + r.user_input_tokens, 0);
  const uOut = results.reduce((n, r) => n + r.user_output_tokens, 0);
  return (
    <>
      <p className="label">
        <Link to="/runs">runs</Link> / {domainLabel(meta.domain)} / {meta.agent}
      </p>
      <h1>
        {meta.agent} on {domainLabel(meta.domain)} {meta.split}: <em>{s?.n_scored ? `${s.passed} of ${s.n_scored}` : `${meta.n_tasks} run`}</em>
      </h1>
      <p className="lead">
        {shortModel(meta.model)} agent · {shortModel(meta.user_model)} user · {meta.trials} trial{meta.trials > 1 ? 's' : ''} · concurrency {meta.concurrency} · seed{' '}
        {meta.seed} · started {when(meta.started_at)} · {fmtS(s?.duration_ms)} of conversation · fingerprint <code>{meta.fingerprint}</code>
        {meta.note ? ` · ${meta.note}` : ''}
      </p>
      <div className="chips">
        {Object.entries(s?.pass_hat_k ?? {}).map(([k, v]) => (
          <span key={k} className="chip ok">
            {k} {Math.round(v * 100)}%
          </span>
        ))}
        {Object.entries(s?.by_termination ?? {}).map(([k, v]) => (
          <span key={k} className={`chip ${k === 'user_stop' || k === 'agent_stop' ? '' : 'warn'}`}>
            {k} {v}
          </span>
        ))}
        <span className={`chip ${s?.errored_ids.length ? 'warn' : ''}`}>errors {s?.errored_ids.length ?? 0}</span>
        <span className="chip">agent tokens {aIn.toLocaleString()} in / {aOut.toLocaleString()} out</span>
        <span className="chip">user tokens {uIn.toLocaleString()} in / {uOut.toLocaleString()} out</span>
        {s?.partial_action_mean != null && <span className="chip">partial actions {Math.round(s.partial_action_mean * 100)}%</span>}
      </div>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>task</th>
              <th>verdict</th>
              <th>components</th>
              <th>purpose</th>
              <th className="num">agent turns</th>
              <th className="num">tools</th>
              <th className="num">time</th>
              <th>ended</th>
              <th>trace</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={`${r.task_id}#${r.trial}`}>
                <td className="sub mono small" title={r.task_id}>
                  {shortTask(r.task_id, 28)}
                  {meta.trials > 1 && <span className="path">trial {r.trial}</span>}
                </td>
                <td>
                  <Verdict r={r} />
                </td>
                <td>
                  <Components r={r} />
                </td>
                <td className="wrap small muted">{r.purpose}</td>
                <td className="num">{r.n_agent_turns}</td>
                <td className="num">
                  {r.n_tool_calls}
                  {r.n_tool_errors ? <span className="path v-warn">{r.n_tool_errors} err</span> : null}
                </td>
                <td className="num">{fmtS(r.duration_ms)}</td>
                <td className="small">
                  {r.termination_reason}
                  {r.error && r.error !== r.termination_reason && <span className="path v-warn">{r.error.slice(0, 80)}</span>}
                </td>
                <td>
                  <Link to={trialPath(runId, trialId(r))}>open</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <details>
        <summary>the agent files this run used</summary>
        <p className="small muted">
          Copied into <code>runs/{runId}/agent/</code> at run time; see{' '}
          <Link to={agentPath(meta.domain, meta.agent)}>
            {domainLabel(meta.domain)}/{meta.agent}
          </Link>{' '}
          for the version.
        </p>
      </details>
    </>
  );
}
