import { useNavigate } from 'react-router-dom';
import { domainLabel, fmtK, shortRun, useGet, when } from '../lib/api';
import { MODELS, modelFamily } from '../lib/scope';
import { Loading } from '../lib/ui';
import { judgeLoopPath } from '../lib/url';

/**
 * Runs, with the scope bar on the LLM judge (s11): the judge's replays, each a folder under
 * `judge_runs/`, scored live against the golden answers as they stand. The answering agent's runs
 * are the same tab with the agent switched back.
 */

type Rate = { k: number; n: number };
type Replay = {
  replay_id: string;
  judge: string;
  name: string;
  model: string;
  effort: string;
  threshold: number;
  started_at: string;
  finished_at: string | null;
  limit: number | null;
  scores: Record<'gate' | 'read' | 'all', {
    balanced_accuracy: number | null;
    passes_interrupted: Rate;
    wrong_plans_stopped: Rate;
    synthetic_blocked: Rate;
    checkpoints: number;
    failed_open: number;
  }>;
  tokens: Record<string, number>;
};

const kn = (r: Rate) => (r.n ? `${r.k} / ${r.n}` : '—');

export function JudgeRuns({ domain, model }: { domain: string; model: string }) {
  const nav = useNavigate();
  const { data, error } = useGet<{ replays: Replay[]; labels: unknown }>(`/api/judge/${encodeURIComponent(domain)}`);
  if (!data) return <Loading error={error} />;
  const reps = data.replays.filter((r) => !model || modelFamily(r.model) === model);
  const label = MODELS.find(([k]) => k === model)?.[1];
  return (
    <>
      <p className="label">Judge replays · {domainLabel(domain)}</p>
      <h1>
        A judge replay is a <em>folder</em>: every verdict at every checkpoint, scored against the golden answers
      </h1>
      <p className="lead">
        Each row is <code>judge_runs/&lt;id&gt;/</code>: the LLM judge asked at every checkpoint of every train conversation it
        would see live, with the real conversation up to that point and no gold. The verdicts never change; the scores are
        read live, so they move when a person’s review of the golden answers is frozen in.
      </p>
      {!data.labels ? (
        <div className="empty">
          No LLM judge has been calibrated on {domainLabel(domain)}: <code>make judge-labels DOMAIN={domain}</code> starts it.
        </div>
      ) : reps.length === 0 ? (
        <div className="empty">
          No replay of the {domainLabel(domain)} judge {label ? `on ${label}` : ''} yet. <code>make judge-replay DOMAIN={domain}</code>{' '}
          runs one.
        </div>
      ) : (
        <div className="tw fit">
          <table>
            <caption>Pick a replay to open its bar and its disagreements under Optimise.</caption>
            <thead>
              <tr>
                <th>replay</th>
                <th>judge</th>
                <th>model</th>
                <th className="num">checkpoints</th>
                <th className="num">
                  balanced accuracy<span className="path">gate half</span>
                </th>
                <th className="num">
                  passes interrupted<span className="path">gate half</span>
                </th>
                <th className="num">
                  wrong plans stopped<span className="path">gate half</span>
                </th>
                <th className="num">
                  synthetic blocked<span className="path">both halves</span>
                </th>
                <th className="num">input tokens</th>
              </tr>
            </thead>
            <tbody>
              {reps.map((r) => (
                <tr key={r.replay_id} className="click" onClick={() => nav(judgeLoopPath(domain, { replay: r.replay_id }))}>
                  <td className="sub mono small" title={`judge_runs/${r.replay_id}/`}>
                    {shortRun(r.replay_id)}
                    <span className="path">
                      {r.finished_at ? `finished ${when(r.finished_at)}` : <span className="v-warn">running since {when(r.started_at)}</span>}
                    </span>
                    {r.limit ? <span className="path">a trial on {r.limit} conversations</span> : null}
                  </td>
                  <td className="mono">{r.name}</td>
                  <td className="small">
                    {r.model.replace(/^claude-/, '')} · {r.effort}
                    <span className="path">blocks at confidence ≥ {r.threshold}</span>
                  </td>
                  <td className="num">
                    {r.scores.all.checkpoints}
                    {r.scores.all.failed_open ? <span className="path">{r.scores.all.failed_open} failed open</span> : null}
                  </td>
                  <td className="num">{r.scores.gate.balanced_accuracy?.toFixed(2) ?? '—'}</td>
                  <td className="num">{kn(r.scores.gate.passes_interrupted)}</td>
                  <td className="num">{kn(r.scores.gate.wrong_plans_stopped)}</td>
                  <td className="num">{kn(r.scores.all.synthetic_blocked)}</td>
                  <td className="num">
                    {fmtK(r.tokens.input ?? 0)}
                    <span className="path">{fmtK(r.tokens.cache_read ?? 0)} cache reads</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
