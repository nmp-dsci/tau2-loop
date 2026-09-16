import { Link, useParams } from 'react-router-dom';
import { type Event, fmtS, shortRun, shortTask, useGet } from '../lib/api';

type RewardInfo = {
  reward: number;
  reward_basis?: string[];
  reward_breakdown?: Record<string, number>;
  db_check?: { db_match: boolean } | null;
  action_checks?: { action: { name: string; arguments: Record<string, unknown> }; action_match: boolean }[] | null;
  communicate_checks?: { info: string; met: boolean }[] | null;
  nl_assertions?: { nl_assertion: string; met: boolean; justification: string }[] | null;
  env_assertions?: { env_assertion: { func_name?: string; arguments?: unknown }; met: boolean }[] | null;
  info?: Record<string, unknown> | null;
};
type TracePayload = { task_id: string; trial: number | null; termination_reason: string; duration: number; agent_cost: number | null; user_cost: number | null; reward_info: RewardInfo | null; events: Event[]; policy_words: number };

export function EventList({ events }: { events: Event[] }) {
  return (
    <>
      {events.map((e, i) => (
        <div key={i} className={`ev ${e.type === 'assistant' ? 'text' : e.type === 'user' ? 'user' : e.type === 'tool_call' ? 'tool_use' : e.type === 'tool_result' ? (e.error ? 'error' : 'tool_result') : e.type}`}>
          <div className="label">
            {e.type === 'assistant' ? 'agent' : e.type === 'user' ? 'user (simulated)' : e.type === 'tool_call' ? `${e.by === 'user' ? 'user' : 'agent'} → ${String(e.name)}` : e.type === 'tool_result' ? (e.error ? 'tool result · error' : 'tool result') : e.type}
          </div>
          {e.type === 'tool_call' && <pre>{JSON.stringify(e.arguments, null, 1)}</pre>}
          {(e.type === 'tool_result' || e.type === 'assistant' || e.type === 'user') && <pre>{String(e.text)}</pre>}
        </div>
      ))}
    </>
  );
}

export function Trace() {
  const { runId = '', name = '' } = useParams();
  const { data, error } = useGet<TracePayload>(`/api/runs/${runId}/traces/${name}`);
  if (error) return <div className="empty">{error}</div>;
  if (!data) return <p className="muted">loading…</p>;
  const ri = data.reward_info;
  return (
    <>
      <p className="label">
        <Link to="/runs">runs</Link> / <Link to={`/runs/${runId}`}>{shortRun(runId)}</Link> / {shortTask(data.task_id, 40)}
      </p>
      <h1>
        {shortTask(data.task_id, 50)}: <em>reward {ri ? ri.reward.toFixed(2) : '—'}</em> after {data.events.filter((e) => e.type === 'assistant' || (e.type === 'tool_call' && e.by !== 'user')).length} agent turns
      </h1>
      <p className="lead">
        ended by {data.termination_reason} · {fmtS(data.duration * 1000)} · basis {(ri?.reward_basis ?? []).join(' × ') || '—'} · policy of {data.policy_words.toLocaleString()} words in the system prompt
      </p>
      {ri && (
        <div className="cards">
          {ri.db_check != null && (
            <div className="card">
              <h3>Database</h3>
              <p className={ri.db_check.db_match ? 'v-ok' : 'v-warn'}>{ri.db_check.db_match ? 'final DB equals the gold DB' : 'final DB differs from the gold DB'}</p>
            </div>
          )}
          {ri.action_checks && ri.action_checks.length > 0 && (
            <div className="card">
              <h3>Expected actions</h3>
              {ri.action_checks.map((a, i) => (
                <p key={i} className={`small ${a.action_match ? 'v-ok' : 'v-warn'}`}>
                  {a.action_match ? '✓' : '✗'} <span className="mono">{a.action.name}</span>({JSON.stringify(a.action.arguments)})
                </p>
              ))}
            </div>
          )}
          {ri.communicate_checks && ri.communicate_checks.length > 0 && (
            <div className="card">
              <h3>Must be said to the user</h3>
              {ri.communicate_checks.map((c, i) => (
                <p key={i} className={`small ${c.met ? 'v-ok' : 'v-warn'}`}>
                  {c.met ? '✓' : '✗'} {c.info}
                </p>
              ))}
            </div>
          )}
          {ri.nl_assertions && ri.nl_assertions.length > 0 && (
            <div className="card">
              <h3>NL assertions (Haiku judge)</h3>
              {ri.nl_assertions.map((a, i) => (
                <p key={i} className={`small ${a.met ? 'v-ok' : 'v-warn'}`} title={a.justification}>
                  {a.met ? '✓' : '✗'} {a.nl_assertion}
                </p>
              ))}
            </div>
          )}
          {ri.env_assertions && ri.env_assertions.length > 0 && (
            <div className="card">
              <h3>Environment assertions</h3>
              {ri.env_assertions.map((a, i) => (
                <p key={i} className={`small ${a.met ? 'v-ok' : 'v-warn'}`}>
                  {a.met ? '✓' : '✗'} <span className="mono">{a.env_assertion.func_name}</span> {JSON.stringify(a.env_assertion.arguments ?? {})}
                </p>
              ))}
            </div>
          )}
          {ri.info && typeof ri.info.note === 'string' && (
            <div className="card">
              <h3>Note</h3>
              <p className="small v-warn">{ri.info.note}</p>
            </div>
          )}
        </div>
      )}
      <h2>The conversation, in order</h2>
      <EventList events={data.events} />
    </>
  );
}
