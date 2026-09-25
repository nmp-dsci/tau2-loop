import { Link, useNavigate, useParams } from 'react-router-dom';
import { DOMAINS, type DomainDetail, domainLabel, shortTask, useGet } from '../lib/api';
import { domainPath } from '../lib/url';

type TaskFull = {
  id: string;
  split: string;
  description?: { purpose?: string; relevant_policies?: string; notes?: string };
  user_scenario?: { instructions?: Record<string, unknown> | string; persona?: string };
  evaluation_criteria?: {
    actions?: { name: string; arguments: Record<string, unknown>; info?: string }[];
    communicate_info?: string[];
    nl_assertions?: string[];
    env_assertions?: unknown[];
    reward_basis?: string[];
  };
};

export function Tasks() {
  const { domain = 'airline', taskId } = useParams();
  const nav = useNavigate();
  const { data } = useGet<DomainDetail>(`/api/domains/${domain}`);
  const { data: task } = useGet<TaskFull>(taskId ? `/api/domains/${domain}/tasks/${encodeURIComponent(taskId)}` : null);
  return (
    <>
      <p className="label">Tasks</p>
      <h1>
        Forty tasks per domain, split <em>once</em>; the agent never sees a scenario or an expected action
      </h1>
      <p className="lead">
        Each task is a user scenario the simulator plays, a purpose the annotator wrote, the policy clauses it exercises, and the
        evaluation criteria the score is a product of. Train tasks feed the optimiser; test tasks are run once per promotion and only
        reported.
      </p>
      <div className="row">
        {DOMAINS.map((d) => (
          <Link key={d} to={domainPath(d)} className={`chip ${d === domain ? 'ok' : ''}`}>
            {domainLabel(d)}
          </Link>
        ))}
      </div>
      {data && (
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>task</th>
                <th>split</th>
                <th>purpose</th>
                <th>policies</th>
                <th className="num">actions</th>
                <th className="num">say</th>
                <th className="num">NL</th>
                <th>basis</th>
              </tr>
            </thead>
            <tbody>
              {data.tasks.map((t) => (
                <tr key={t.id} className={`click ${t.id === taskId ? 'pro' : ''}`} onClick={() => nav(`/tasks/${domain}/${encodeURIComponent(t.id)}`)}>
                  <td className="sub mono small" title={t.id}>
                    {shortTask(t.id, 30)}
                  </td>
                  <td>{t.split === 'train' ? <span className="status ok">train</span> : <span className="status warn">test</span>}</td>
                  <td className="wrap small">{t.purpose ?? t.reason_for_call ?? '—'}</td>
                  <td className="wrap small muted">{t.relevant_policies ?? '—'}</td>
                  <td className="num">{t.n_actions}</td>
                  <td className="num">{t.n_communicate}</td>
                  <td className="num">{t.n_nl_assertions}</td>
                  <td className="mono small">{(t.reward_basis ?? []).join('+')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {task && (
        <>
          <h2>
            {domainLabel(domain)} · {shortTask(task.id, 60)} · {task.split}
          </h2>
          <div className="grid2">
            <div className="card">
              <h3>What the simulated user was told</h3>
              <div className="code">
                <pre>{typeof task.user_scenario?.instructions === 'string' ? task.user_scenario.instructions : JSON.stringify(task.user_scenario?.instructions ?? {}, null, 1)}</pre>
              </div>
              {task.user_scenario?.persona && <p className="small muted">persona: {String(task.user_scenario.persona).slice(0, 300)}</p>}
            </div>
            <div className="card">
              <h3>What the evaluator checks</h3>
              <p className="small">
                <b>purpose</b> {task.description?.purpose ?? '—'}
              </p>
              <p className="small">
                <b>policies</b> {task.description?.relevant_policies ?? '—'}
              </p>
              {task.description?.notes && (
                <p className="small">
                  <b>notes</b> {task.description.notes}
                </p>
              )}
              <p className="small">
                <b>reward basis</b> {(task.evaluation_criteria?.reward_basis ?? []).join(' × ') || 'DB × COMMUNICATE (default)'}
              </p>
              <div className="code">
                <pre>
                  {(task.evaluation_criteria?.actions ?? []).map((a) => `${a.name}(${JSON.stringify(a.arguments)})${a.info ? `  # ${a.info}` : ''}`).join('\n') || '(no expected actions)'}
                </pre>
              </div>
              {(task.evaluation_criteria?.communicate_info ?? []).length > 0 && (
                <p className="small">
                  <b>must say</b> {(task.evaluation_criteria?.communicate_info ?? []).join(' · ')}
                </p>
              )}
              {(task.evaluation_criteria?.nl_assertions ?? []).length > 0 && (
                <p className="small">
                  <b>NL assertions</b> {(task.evaluation_criteria?.nl_assertions ?? []).join(' · ')}
                </p>
              )}
            </div>
          </div>
        </>
      )}
    </>
  );
}
