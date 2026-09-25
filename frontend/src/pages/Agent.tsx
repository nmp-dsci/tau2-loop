import { Link, useParams } from 'react-router-dom';
import {
  type AgentInfo,
  type Health,
  type Registries,
  domainLabel,
  shortRun,
  useGet,
} from '../lib/api';
import { Loading } from '../lib/ui';
import { agentPath, agentsPath, optimisePath, runPath, useLens } from '../lib/url';

/**
 * What the agent is, in one tab. s04 M3 merged the old Architecture page (the
 * figure: who calls the model, and how) with the old Agents page (the version
 * folders), because they answered the same question at two zoom levels and the
 * figure had no way to show you what was inside any of its boxes.
 *
 * Every box in the figure is a button: it sets `?node=`, and the panel below
 * shows what that box actually contains for the selected version.
 */

type AgentsPayload = { versions: AgentInfo[]; registry: Registries };
type AgentFiles = {
  domain: string;
  name: string;
  fingerprint: string;
  config: Record<string, unknown>;
  files: Record<string, string>;
};

type NodeKey =
  | 'user'
  | 'judge'
  | 'factory'
  | 'generate'
  | 'provider'
  | 'sdk'
  | 'prompting'
  | 'runs';

const NODES: Record<NodeKey, { title: string; where: string }> = {
  user: { title: 'The user simulator', where: 'vendor/tau2-bench · user_simulator.py' },
  judge: { title: 'The NL judge', where: 'vendor/tau2-bench · evaluator_nl_assertions.py' },
  factory: { title: 'Our agent factory', where: 'src/tau2_loop/agent/factory.py' },
  generate: { title: "tau2's one choke point", where: 'vendor/tau2-bench · llm_utils.generate()' },
  provider: { title: 'The claude-sdk provider', where: 'src/tau2_loop/llm/sdk_provider.py' },
  sdk: { title: 'The Agent SDK', where: 'claude-agent-sdk · the subscription' },
  prompting: { title: 'History → prompt', where: 'src/tau2_loop/llm/prompting.py' },
  runs: { title: 'The run folder is the record', where: 'runs/<id>/' },
};

export function Agent() {
  const { domain, name } = useParams();
  const [lens, setLens] = useLens();
  const node = (lens.get('node') as NodeKey | null) ?? null;
  const { data, error } = useGet<AgentsPayload>('/api/agents');
  const { data: health } = useGet<Health>('/healthz');
  const { data: files } = useGet<AgentFiles>(
    domain && name ? `/api/agents/${encodeURIComponent(domain)}/${encodeURIComponent(name)}` : null,
  );
  if (!data) return <Loading error={error} />;

  const selected = data.versions.find((v) => v.domain === domain && v.name === name);
  const role = (d: string, n: string) => {
    const r = data.registry[d];
    if (r?.champion?.agent === n) return <span className="status ok">champion</span>;
    if (r?.challenger?.agent === n) return <span className="status warn">challenger</span>;
    return <span className="status no">held</span>;
  };
  const pick = (key: NodeKey) => () => setLens({ node: node === key ? '' : key });

  return (
    <>
      <p className="label">The agent</p>
      <h1>
        One choke point carries every model call to the subscription; the harness is{' '}
        <em>unmodified</em>
      </h1>
      <p className="lead">
        tau2 sends every model call — the agent's, the user simulator's, the NL judge's — through one
        function, <code>generate()</code>, which calls litellm. A custom litellm provider registered
        under the prefix <code>claude-sdk/</code> answers those calls with one Claude Agent SDK query
        each, so <code>--agent-llm claude-sdk/claude-haiku-4-5</code> is a Haiku call on the
        subscription with nothing in tau2 touched. <b>Pick a box to see what is inside it.</b>
      </p>

      <figure>
        <div className="label fig-title">fig 1 · the subscription adapter — one choke point, three callers</div>
        <svg
          className="dia graph"
          viewBox="0 0 1000 300"
          role="img"
          aria-label="Three callers — tau2's user simulator, tau2's NL judge and our agent factory — call tau2's generate(), which calls litellm; the claude-sdk provider turns each request into one Agent SDK query on the subscription."
        >
          <defs>
            <marker
              id="garr"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M0 0L10 5L0 10z" fill="currentColor" />
            </marker>
          </defs>
          {(
            [
              ['user', 'user simulator', 'tau2 · user_simulator.py', 20, 30],
              ['judge', 'NL judge', 'tau2 · evaluator_nl_assertions.py', 20, 120],
              ['factory', 'agent factory (ours)', 'tools owned by tau2', 20, 210],
            ] as [NodeKey, string, string, number, number][]
          ).map(([key, t, s, x, y]) => (
            <g key={key} className="pick" role="button" tabIndex={0} onClick={pick(key)}>
              <title>{NODES[key].title}</title>
              <rect
                className={`nd ${key === 'factory' ? 'hi' : ''} ${node === key ? 'on' : ''}`}
                x={x}
                y={y}
                width="230"
                height="60"
                rx="10"
              />
              <text className="tx" x={x + 14} y={y + 26}>
                {t}
              </text>
              <text className="tx s" x={x + 14} y={y + 46}>
                {s}
              </text>
            </g>
          ))}
          <g className="pick" role="button" tabIndex={0} onClick={pick('generate')}>
            <title>{NODES.generate.title}</title>
            <rect className={`nd ${node === 'generate' ? 'on' : ''}`} x="330" y="110" width="190" height="80" rx="10" />
            <text className="tx" x="346" y="140">
              generate()
            </text>
            <text className="tx s" x="346" y="162">
              llm_utils.py · litellm
            </text>
          </g>
          <g className="pick" role="button" tabIndex={0} onClick={pick('provider')}>
            <title>{NODES.provider.title}</title>
            <rect className={`nd hi ${node === 'provider' ? 'on' : ''}`} x="600" y="110" width="200" height="80" rx="10" />
            <text className="tx" x="616" y="140">
              claude-sdk/ provider
            </text>
            <text className="tx s" x="616" y="162">
              sdk_provider.py
            </text>
          </g>
          <g className="pick" role="button" tabIndex={0} onClick={pick('sdk')}>
            <title>{NODES.sdk.title}</title>
            <rect className={`nd ${node === 'sdk' ? 'on' : ''}`} x="860" y="110" width="126" height="80" rx="10" />
            <text className="tx" x="874" y="140">
              Agent SDK
            </text>
            <text className="tx s" x="874" y="162">
              subscription
            </text>
          </g>
          <g className="pick" role="button" tabIndex={0} onClick={pick('prompting')}>
            <title>{NODES.prompting.title}</title>
            <rect className={`nd ext ${node === 'prompting' ? 'on' : ''}`} x="600" y="220" width="200" height="60" rx="10" />
            <text className="tx" x="616" y="246">
              prompting.py
            </text>
            <text className="tx s" x="616" y="266">
              history → prompt · JSON reply
            </text>
          </g>
          <g className="pick" role="button" tabIndex={0} onClick={pick('runs')}>
            <title>{NODES.runs.title}</title>
            <rect className={`nd ${node === 'runs' ? 'on' : ''}`} x="330" y="220" width="190" height="60" rx="10" />
            <text className="tx" x="346" y="246">
              runs/&lt;id&gt;/
            </text>
            <text className="tx s" x="346" y="266">
              the record · MLflow indexes it
            </text>
          </g>
          <path className="ed" d="M250 60 L328 130" />
          <path className="ed" d="M250 150 L328 150" />
          <path className="ed hi" d="M250 240 L328 170" />
          <path className="ed" d="M520 150 L598 150" />
          <text className="cap" x="528" y="140">
            claude-sdk/*
          </text>
          <path className="ed" d="M800 150 L858 150" />
          <path className="ed dash" d="M700 218 L700 192" />
          <path className="ed dash" d="M425 218 L425 192" />
        </svg>
        <figcaption>
          Text roles are a plain query; the agent's tools travel as a JSON contract in the prompt and
          come back as <code>tool_calls</code> the same way an OpenAI response would, so tau2's
          orchestrator executes them unchanged.
          <span className="path">
            src/tau2_loop/llm/{'{'}sdk_provider,prompting{'}'}.py ·
            vendor/tau2-bench/src/tau2/utils/llm_utils.py
          </span>
        </figcaption>
      </figure>

      {node && (
        <section className="card hi nodepanel">
          <div className="crumbs">
            <span className="label">{NODES[node].where}</span>
            <button type="button" className="linkish" onClick={() => setLens({ node: '' })}>
              {' '}
              close
            </button>
          </div>
          <h3>{NODES[node].title}</h3>
          <NodePanel node={node} selected={selected} files={files} health={health} />
        </section>
      )}

      <h2>What a turn costs, and what it cannot do</h2>
      <div className="cards">
        <div className="card">
          <h3>One CLI process per model call</h3>
          <p className="small">
            Each request is a fresh <code>query()</code>: the system prompt carries the policy and
            the tool schemas, the user prompt carries the transcript so far. A twenty-turn
            conversation is roughly twenty agent calls and twenty user-simulator calls.
          </p>
        </div>
        <div className="card">
          <h3>Sampling is the CLI default</h3>
          <p className="small">
            The SDK exposes no temperature, so tau2's <code>temperature: 0.0</code> does not apply;
            every run records <code>sampling: cli-default</code>. Variance across trials is real and
            is why <code>pass^k</code> is reported.
          </p>
        </div>
        <div className="card">
          <h3>Billing cannot surprise</h3>
          <p className="small">
            <code>require_live()</code> refuses to start when a key is exported alongside{' '}
            <code>BILLING=subscription</code>, scrubs a key that tau2's dotenv search injects from{' '}
            <code>~/.env</code>, and the SDK child gets a blanked key. The demo image sets{' '}
            <code>DEMO_MODE=1</code> and has no route that calls a model.
          </p>
        </div>
      </div>

      <h2>Every version is a folder per domain; the champion is the one the gate kept</h2>
      <p>
        <code>agents/&lt;domain&gt;/vN/</code> holds a system prompt with a <code>{'{policy}'}</code>{' '}
        slot, an optional helper with three hooks, and a frozen config. The optimiser writes the next
        folder and a <code>diagnosis.json</code> beside it; the gate decides which folder the
        domain's registry points at — read a round on{' '}
        <Link to={optimisePath(domain ?? 'airline')}>Optimise</Link>.
      </p>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>domain</th>
              <th>version</th>
              <th>role</th>
              <th>fingerprint</th>
              <th>model</th>
              <th>helper hooks</th>
              <th>runs</th>
            </tr>
          </thead>
          <tbody>
            {data.versions.map((v) => (
              <tr
                key={v.ref}
                className={data.registry[v.domain]?.champion?.agent === v.name ? 'pro' : ''}
              >
                <td>{domainLabel(v.domain)}</td>
                <td className="sub">
                  <Link to={agentPath(v.domain, v.name, { node })}>{v.name}</Link>
                </td>
                <td>{role(v.domain, v.name)}</td>
                <td className="mono">{v.fingerprint}</td>
                <td className="mono">
                  {String(v.config.model)} · {String(v.config.tool_mode)}
                </td>
                <td className="wrap mono small">{v.helper_functions.join(', ') || '—'}</td>
                <td className="small">
                  {v.runs.map((r) => (
                    <div key={r}>
                      <Link to={runPath(r)}>{shortRun(r)}</Link>
                    </div>
                  ))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {domain && name && files && (
        <>
          <h2>
            {domainLabel(domain)} / {name} — the files, verbatim
          </h2>
          <p className="small muted">
            fingerprint <code>{files.fingerprint}</code> ·{' '}
            <Link to={agentsPath({ node })}>every version</Link>
          </p>
          {Object.entries(files.files).map(([fname, text]) => (
            <details key={fname} open={fname !== 'agent.yaml'}>
              <summary>
                {fname} <span className="muted small">({text.length.toLocaleString()} chars)</span>
              </summary>
              <div className="code">
                <pre>{text}</pre>
              </div>
            </details>
          ))}
          {selected?.diagnosis && (
            <>
              <h3>diagnosis.json — why this version exists</h3>
              <div className="code">
                <pre>{JSON.stringify(selected.diagnosis, null, 1)}</pre>
              </div>
            </>
          )}
        </>
      )}
    </>
  );
}

function NodePanel({
  node,
  selected,
  files,
  health,
}: {
  node: NodeKey;
  selected: AgentInfo | undefined;
  files: AgentFiles | null;
  health: Health | null;
}) {
  switch (node) {
    case 'user':
      return (
        <>
          <p className="small">
            τ²'s own simulator plays the customer from the task's scenario, which the agent never
            sees. It is a model call like any other — it goes through the same{' '}
            <code>generate()</code> — so it is on the subscription too, and it is roughly half the
            token bill of every run.
          </p>
          <p className="small muted">
            Our runs use Haiku as the simulator. The published leaderboard uses gpt-5.2, which is one
            of the three reasons our entry would be marked unverified.
          </p>
        </>
      );
    case 'judge':
      return (
        <>
          <p className="small">
            The only check with a model inside it: it decides whether each NL assertion is true of
            the transcript. The other four checks — the database hash, expected actions, things that
            must be said, environment assertions — are deterministic.
          </p>
          <p className="small">
            Which means a failed conversation is not automatically a bad agent, and that is what the
            Review tab exists to record.
          </p>
        </>
      );
    case 'factory':
      return (
        <>
          <p className="small">
            Registered in tau2's own agent registry under <code>tau2_loop</code>, so the harness
            constructs it exactly as it constructs its own. It supplies a system prompt with a{' '}
            <code>{'{policy}'}</code> slot and, optionally, a helper with three hooks. The tools stay
            tau2's.
          </p>
          {selected ? (
            <dl className="diff-sum">
              <dt>version</dt>
              <dd>
                {domainLabel(selected.domain)} / {selected.name} · {selected.fingerprint}
              </dd>
              <dt>model</dt>
              <dd className="mono">
                {String(selected.config.model)} · {String(selected.config.tool_mode)}
              </dd>
              <dt>helper</dt>
              <dd className="mono">{selected.helper_functions.join(', ') || 'none'}</dd>
              <dt>system.md</dt>
              <dd>
                {files ? `${(files.files['system.md'] ?? '').length.toLocaleString()} chars` : '—'}
              </dd>
            </dl>
          ) : (
            <p className="small muted">Pick a version in the table below to see its prompt here.</p>
          )}
        </>
      );
    case 'generate':
      return (
        <p className="small">
          Every model call in τ² funnels through this one function before litellm sees it, which is
          the whole reason this project needs no fork of the harness: register a provider under a
          prefix, and all three callers are redirected at once.
        </p>
      );
    case 'provider':
      return (
        <p className="small">
          A litellm custom provider registered under <code>claude-sdk/</code>. It turns one
          OpenAI-shaped chat request into one Agent SDK query and turns the reply back, including
          tool calls. Nothing in the harness knows the difference.
        </p>
      );
    case 'sdk':
      return (
        <>
          <p className="small">
            One <code>query()</code> per model call, on the Claude subscription rather than an API
            key. No temperature is exposed, so every run records{' '}
            <code>sampling: cli-default</code>.
          </p>
          <p className="small muted">
            Cost figures anywhere in this viewer are <code>cost_usd_est</code>: what the same tokens
            would have cost on the API. Nothing here is billed per token.
          </p>
        </>
      );
    case 'prompting':
      return (
        <p className="small">
          Serialises the chat history into a single prompt, and — when tools are present — asks for a
          JSON reply of either a message or a list of tool calls. That JSON contract is the second
          reason our entry would be unverified: the published baselines use native tool calling.
        </p>
      );
    case 'runs':
      return (
        <>
          <p className="small">
            Every number in this viewer is read from a committed file. MLflow is an index over those
            files, never the record: if the tracking server is down the eval still completes and the
            run folder is complete.
          </p>
          <p className="small">
            central MLflow:{' '}
            {health?.mlflow_url ? (
              <a href={health.mlflow_url}>{health.mlflow_url}</a>
            ) : (
              'not configured'
            )}{' '}
            · {health?.mlflow_embeddable ? 'answering' : 'not reachable from here'}
          </p>
        </>
      );
  }
}
