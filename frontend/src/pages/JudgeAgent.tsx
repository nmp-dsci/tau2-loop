import { Link, useParams } from 'react-router-dom';
import { type AgentInfo, type Registries, domainLabel, useGet } from '../lib/api';
import { MODELS, modelFamily } from '../lib/scope';
import { Loading, Points } from '../lib/ui';
import { agentPath, judgeLoopPath, runsPath, useLens } from '../lib/url';

/**
 * Agent, with the scope bar on the LLM judge (s11): the judge as an agent of its own. It is a second
 * sealed session through `llm/core.py` with its own rubric, model and effort, and it shares nothing
 * with the answering agent but the policy and the tools' names. Read from `judges/<domain>/` and
 * `data/judge/probe.json`.
 */

type JudgeVersion = {
  ref: string;
  kind: string;
  name: string;
  model: string;
  effort: string;
  threshold: number;
  structured: boolean;
  fingerprint: string;
  config: Record<string, unknown>;
  files: Record<string, string>;
  replays: string[];
};
type Payload = { versions: JudgeVersion[]; probe: { holds: boolean; sdk: string; model: string; at: string } | null };

/** `sonnet`, `claude-sonnet-5` → `Sonnet 5`: one name per model family, as the scope bar shows it. */
const short = (m: string) => MODELS.find(([k]) => k === modelFamily(m))?.[1] ?? m;

export function JudgeAgent() {
  const { domain = 'airline' } = useParams();
  const [lens, setLens] = useLens();
  const { data, error } = useGet<Payload>(`/api/judge/${encodeURIComponent(domain)}/versions`);
  const { data: agents } = useGet<{ versions: AgentInfo[]; registry: Registries }>('/api/agents');
  if (!data) return <Loading error={error} />;
  const vs = data.versions;
  const j = vs.find((v) => v.name === lens.get('version')) ?? vs[vs.length - 1];
  const champ = agents?.registry[domain]?.champion?.agent;
  const champInfo = agents?.versions.find((v) => v.domain === domain && v.name === champ);

  return (
    <>
      <p className="label">Agent · the LLM judge · {domainLabel(domain)}</p>
      <h1>
        The LLM judge is a second agent that only <em>reads</em>: it never talks to the customer or calls a tool
      </h1>
      <Points
        lead
        items={[
          <>
            <b>The judge reads each plan before the customer does</b> and says allow or block.
          </>,
          <>
            <b>It is a separate sealed session</b> with its own rubric, model and effort.
          </>,
          <>
            <b>The answering agent</b> talks to the customer and calls the tools.
          </>,
        ]}
      />
      {!j ? (
        <div className="empty">
          No LLM judge for {domainLabel(domain)} yet. Airline’s is <code>judges/airline/plan/j1/</code>;
          s11’s J8 brings one here.
        </div>
      ) : (
        <>
          {vs.length > 1 && (
            <nav className="chips" aria-label="judge versions">
              {vs.map((v) => (
                <button key={v.ref} type="button" className={`chip nav ${v.name === j.name ? 'on' : ''}`} onClick={() => setLens({ version: v.name })}>
                  {v.name}
                </button>
              ))}
            </nav>
          )}
          <h2>Two agents — one answers, one reviews, sharing nothing but the policy</h2>
          <figure>
            <div className="label fig-title">the answering agent and the LLM judge, side by side</div>
            <div className="tw">
              <table>
                <thead>
                  <tr>
                    <th></th>
                    <th>answering agent</th>
                    <th>LLM judge</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="sub">version</td>
                    <td>
                      {champ ? (
                        <Link to={agentPath(domain, champ)} className="mono">
                          {champ}
                        </Link>
                      ) : (
                        '—'
                      )}{' '}
                      <span className="muted">the champion</span>
                    </td>
                    <td className="mono">
                      {j.name} <span className="muted">· {j.fingerprint}</span>
                    </td>
                  </tr>
                  <tr>
                    <td className="sub">model</td>
                    <td>{champInfo ? `${short(String(champInfo.config.model))} · ${String(champInfo.config.effort ?? 'medium')}` : '—'}</td>
                    <td>
                      {short(j.model)} · {j.effort}
                    </td>
                  </tr>
                  <tr>
                    <td className="sub">its job</td>
                    <td className="wrap">serve the customer within the policy: read, plan, ask for a yes, write</td>
                    <td className="wrap">check each plan against the policy and the conversation before the customer sees it</td>
                  </tr>
                  <tr>
                    <td className="sub">reads</td>
                    <td className="wrap">its system prompt with the policy, the tools’ schemas, the conversation</td>
                    <td className="wrap">its rubric, the policy, the tools’ names, the conversation so far, the proposed reply</td>
                  </tr>
                  <tr>
                    <td className="sub">never reads</td>
                    <td className="wrap">gold, the grader, the user simulator’s instructions</td>
                    <td className="wrap">the same, and the answering agent’s own session</td>
                  </tr>
                  <tr>
                    <td className="sub">answers with</td>
                    <td className="wrap">a message, or tool calls (the JSON reply contract)</td>
                    <td className="wrap">allow or block, with confidence, check, policy rule, evidence and a fix, schema-enforced</td>
                  </tr>
                  <tr>
                    <td className="sub">improved by</td>
                    <td className="wrap">
                      the agent loop: <Link to={`/optimise/${encodeURIComponent(domain)}`}>Optimise › answering agent</Link>
                    </td>
                    <td className="wrap">
                      the judge loop: <Link to={judgeLoopPath(domain)}>Optimise › LLM judge</Link>
                    </td>
                  </tr>
                  <tr>
                    <td className="sub">its runs</td>
                    <td className="wrap">
                      <Link to={runsPath({ domain })}>runs/</Link>, one conversation per task
                    </td>
                    <td className="wrap">
                      <Link to={runsPath({ domain, agent: 'judge' })}>judge_runs/</Link>: {j.replays.length} replay{j.replays.length === 1 ? '' : 's'} of
                      the train checkpoints
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <figcaption>
              <Points
                items={[
                  <>
                    <b>A block stands only at confidence ≥ {j.threshold}</b> with a rule verbatim in the policy; an error lets the
                    reply through.
                  </>,
                  <>
                    <b>The same sealed core, its own session</b>: <code>llm/core.py</code>, no tools, no settings, a temporary
                    directory.
                  </>,
                  data.probe ? (
                    <>
                      <b>Schema-enforced output held</b> under that core (probe on {data.probe.sdk}, {data.probe.at.slice(0, 10)}).
                    </>
                  ) : null,
                ]}
              />
              <span className="path">
                judges/{domain}/{j.kind}/{j.name}/ · agents/{domain}/{champ ?? 'vN'}/ · src/tau2_loop/tooljudge/
              </span>
            </figcaption>
          </figure>

          <h2>Its rubric — five generic checks, no task, id or customer</h2>
          <figure>
            <div className="label fig-title">judge.md, the judge’s whole system prompt before the policy and tools are added</div>
            <pre className="filebody">{j.files['judge.md']}</pre>
            <figcaption>
              J3’s loop edits only this file, as numbered lessons, gated on the train half it never reads.
              <span className="path">judges/{domain}/{j.kind}/{j.name}/judge.md</span>
            </figcaption>
          </figure>
          <figure>
            <div className="label fig-title">judge.yaml, its model, effort and decision rule</div>
            <pre className="filebody">{j.files['judge.yaml']}</pre>
          </figure>
        </>
      )}
    </>
  );
}
