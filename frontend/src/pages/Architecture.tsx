function Arrow() {
  return (
    <defs>
      <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0L10 5L0 10z" fill="currentColor" />
      </marker>
    </defs>
  );
}

export function Architecture() {
  return (
    <>
      <p className="label">Architecture</p>
      <h1>
        One choke point carries every model call to the subscription; the harness is <em>unmodified</em>
      </h1>
      <p className="lead">
        tau2 sends every model call — the agent's, the user simulator's, the retail judge's — through one function,
        <code>generate()</code>, which calls litellm. A custom litellm provider registered under the prefix <code>claude-sdk/</code>
        answers those calls with one Claude Agent SDK query each, so <code>--agent-llm claude-sdk/claude-haiku-4-5</code> is a Haiku call on
        the subscription with nothing in tau2 touched.
      </p>

      <figure>
        <div className="label fig-title">fig 1 · the subscription adapter — one choke point, three callers</div>
        <svg className="dia" viewBox="0 0 1000 300" role="img" aria-label="Three callers — tau2's user simulator, tau2's NL judge and our agent factory — call tau2's generate(), which calls litellm; the claude-sdk provider turns each request into one Agent SDK query on the subscription, serialising the chat history into the prompt and, when tools are present, asking for a JSON reply of either a message or tool calls.">
          <Arrow />
          {[
            ['user simulator', 'tau2 · user_simulator.py', 20, 30],
            ['retail NL judge', 'tau2 · evaluator_nl_assertions.py', 20, 120],
            ['agent factory (ours)', 'tools owned by tau2', 20, 210],
          ].map(([t, s, x, y]) => (
            <g key={String(t)}>
              <rect className={String(t).includes('ours') ? 'nd hi' : 'nd'} x={Number(x)} y={Number(y)} width="230" height="60" rx="10" />
              <text className="tx" x={Number(x) + 14} y={Number(y) + 26}>
                {String(t)}
              </text>
              <text className="tx s" x={Number(x) + 14} y={Number(y) + 46}>
                {String(s)}
              </text>
            </g>
          ))}
          <rect className="nd" x="330" y="110" width="190" height="80" rx="10" />
          <text className="tx" x="346" y="140">generate()</text>
          <text className="tx s" x="346" y="162">llm_utils.py · litellm</text>
          <rect className="nd hi" x="600" y="110" width="200" height="80" rx="10" />
          <text className="tx" x="616" y="140">claude-sdk/ provider</text>
          <text className="tx s" x="616" y="162">sdk_provider.py</text>
          <rect className="nd" x="860" y="110" width="126" height="80" rx="10" />
          <text className="tx" x="874" y="140">Agent SDK</text>
          <text className="tx s" x="874" y="162">subscription</text>
          <rect className="nd ext" x="600" y="220" width="200" height="60" rx="10" />
          <text className="tx" x="616" y="246">prompting.py</text>
          <text className="tx s" x="616" y="266">history → prompt · JSON reply</text>
          <path className="ed" d="M250 60 L328 130" />
          <path className="ed" d="M250 150 L328 150" />
          <path className="ed hi" d="M250 240 L328 170" />
          <path className="ed" d="M520 150 L598 150" />
          <text className="cap" x="528" y="140">claude-sdk/*</text>
          <path className="ed" d="M800 150 L858 150" />
          <path className="ed dash" d="M700 218 L700 192" />
        </svg>
        <figcaption>
          Text roles are a plain query; the agent's tools travel as a JSON contract in the prompt and come back as <code>tool_calls</code> the same way an OpenAI response would, so tau2's orchestrator executes them unchanged.
          <span className="path">src/tau2_loop/llm/{'{'}sdk_provider,prompting{'}'}.py · vendor/tau2-bench/src/tau2/utils/llm_utils.py</span>
        </figcaption>
      </figure>

      <h2>What a turn costs, and what it cannot do</h2>
      <div className="cards">
        <div className="card">
          <h3>One CLI process per model call</h3>
          <p className="small">
            Each request is a fresh <code>query()</code>: the system prompt carries the policy and the tool schemas, the user prompt carries the
            transcript so far. A twenty-turn conversation is roughly twenty agent calls and twenty user-simulator calls.
          </p>
        </div>
        <div className="card">
          <h3>Sampling is the CLI default</h3>
          <p className="small">
            The SDK exposes no temperature, so tau2's <code>temperature: 0.0</code> does not apply; every run records{' '}
            <code>sampling: cli-default</code>. Variance across trials is real and is why <code>pass^k</code> is reported.
          </p>
        </div>
        <div className="card">
          <h3>Billing cannot surprise</h3>
          <p className="small">
            <code>require_live()</code> refuses to start when a key is exported alongside <code>BILLING=subscription</code>, scrubs a key that tau2's
            dotenv search injects from <code>~/.env</code>, and the SDK child gets a blanked key. The demo image sets <code>DEMO_MODE=1</code> and
            has no route that calls a model.
          </p>
        </div>
      </div>

      <h2>The repository, by responsibility</h2>
      <div className="code">
        <pre>{`tau2-loop/
  vendor/tau2-bench/        τ³-bench v1.0.1 at 2174a60 (submodule): harness, domains, data, evaluator
  data/splits/<domain>.json 20 train / 20 test ids, seed 300, committed
  data/tasks/<domain>.json  the forty tasks, policy and tool list, extracted for the viewer
  agents/<domain>/vN/       system.md · helper.py (editable) · agent.yaml (frozen) · diagnosis.json
  runs/<ts>_<domain>_<vN>_<split>/  run.json · results.jsonl · traces/ · tau2_results.json · agent/
  loop/<domain>/            ledger.jsonl · registry.json
  src/tau2_loop/
    llm/                    __init__ (models, billing) · sdk_provider (litellm → Agent SDK) · prompting
    agent/                  versions · factory (registered as "tau2_loop")
    data/splits.py          cut, extract, read
    eval/                   runner · results · compare (McNemar) · rescore (offline replay)
    loop/                   run · optimiser · ledger
    tracking/               registry · mlflow_log · snapshot · gate (CI)
    serving/app.py          this viewer's API
  frontend/                 Vite + React on tokens.css`}</pre>
      </div>
    </>
  );
}
