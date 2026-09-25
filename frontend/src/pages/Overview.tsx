import { Link } from 'react-router-dom';
import { type DomainSummary, type LedgerEntry, type RunMeta, domainLabel, fmtK, shortRun, useGet } from '../lib/api';
import { Rate } from '../lib/ui';
import { domainPath, optimisePath, rubricPath, runPath, runsPath } from '../lib/url';

/** One object the whole strip is read from (`/api/stats`). */
type Stats = {
  domains: { domain: string; base_n: number | null; champion: string | null; passed: number | null; n_scored: number | null; cycles: number }[];
  base_total: number;
  runs: number;
  runs_scored: number;
  conversations: number;
  cost_usd_est: number;
  cycles: number;
};

function Arrow() {
  return (
    <defs>
      <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0L10 5L0 10z" fill="currentColor" />
      </marker>
    </defs>
  );
}

export function Overview() {
  const { data: stats } = useGet<Stats>('/api/stats');
  const { data: domains } = useGet<DomainSummary[]>('/api/domains');
  const { data: runs } = useGet<RunMeta[]>('/api/runs');
  const { data: ledger } = useGet<Record<string, LedgerEntry[]>>('/api/ledger');
  const real = (runs ?? []).filter((r) => !r.dry_run && r.summary?.n_scored);
  const cycles = Object.values(ledger ?? {}).flat();
  const promoted = cycles.filter((e) => e.outcome?.verdict === 'promote').length;
  const optTokens = cycles.reduce((n, e) => n + (e.tokens?.optimiser_in ?? 0) + (e.tokens?.optimiser_out ?? 0), 0);
  const scoredDomains = (domains ?? []).filter((d) => d.champion);
  const champPassed = scoredDomains.reduce((n, d) => n + (d.champion?.passed ?? 0), 0);
  const champScored = scoredDomains.reduce((n, d) => n + (d.champion?.n_scored ?? 0), 0);
  const conversations = real.reduce((n, r) => n + (r.summary?.n ?? 0), 0);
  const testRuns = real.filter((r) => r.split === 'test');

  return (
    <>
      <p className="label">τ²-bench · Claude Agent SDK on the subscription · Haiku 4.5</p>
      <h1>
        A policy agent on four domains, and the loop that <em>learns</em> from the conversations it failed
      </h1>
      <p className="lead">
        The agent is a system prompt and an optional helper on Haiku 4.5; tau2's harness owns the tools, the simulated
        customer and the score. The loop around it is one optimiser session per cycle that reads every failed
        conversation, edits those two files, and hands the result to a gate. Every diagnosis and every verdict is
        kept per domain, so the next cycle starts from what the last one learnt.
      </p>

      <div className="kpis">
        <div className="kpi">
          <div className="label">champions on train</div>
          <div className="n">{champScored ? `${champPassed}/${champScored}` : '—'}</div>
          <div className="b">
            {scoredDomains.length ? `${scoredDomains.length} of 4 domains have a champion · 20 train conversations each` : 'no champion yet — run `make eval DOMAIN=…` and promote'}
          </div>
        </div>
        <div className="kpi">
          <div className="label">conversations scored</div>
          <div className="n">{conversations || '—'}</div>
          <div className="b">
            {real.length} runs · {testRuns.length} on the held-out test split · of {stats?.base_total ?? '—'} base tasks in the benchmark
          </div>
        </div>
        <div className="kpi">
          <div className="label">loop cycles</div>
          <div className="n">{cycles.length}</div>
          <div className="b">
            {promoted} promoted · {cycles.length - promoted} held or rejected · {fmtK(optTokens)} optimiser tokens
          </div>
        </div>
        <div className="kpi">
          <div className="label">model calls billed</div>
          <div className="n ok">$0</div>
          <div className="b">every role runs through the Claude subscription; `cost est.` on the runs page is the SDK's per-token estimate</div>
        </div>
      </div>

      <h2>1 · Where each domain stands — train is what the gate sees, test is reported once per promotion</h2>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>domain</th>
              <th>base</th>
              <th>split</th>
              <th>scored by</th>
              <th>champion</th>
              <th className="num">train</th>
              <th className="num">test</th>
              <th className="num">versions</th>
              <th className="num">cycles</th>
            </tr>
          </thead>
          <tbody>
            {(domains ?? []).map((d) => {
              const test = real.filter((r) => r.domain === d.domain && r.split === 'test' && r.agent === d.champion?.agent).slice(-1)[0];
              return (
                <tr key={d.domain} className={d.champion ? '' : ''}>
                  <td className="sub">
                    <Link to={domainPath(d.domain)}>{domainLabel(d.domain)}</Link>
                  </td>
                  <td className="num">{d.base_n ?? '—'}</td>
                  <td className="mono small">
                    {d.train} / {d.test} · seed {d.seed}
                  </td>
                  <td className="small wrap">{Object.entries(d.reward_bases).map(([k, v]) => `${k} ${v}`).join(' · ') || '—'}</td>
                  <td className="mono">{d.champion ? `${d.champion.agent} · ${d.champion.fingerprint}` : <span className="muted">none</span>}</td>
                  <td className="num">
                    <Rate passed={d.champion?.passed} n={d.champion?.n_scored} />
                  </td>
                  <td className="num">
                    <Rate passed={test?.summary?.passed} n={test?.summary?.n_scored} />
                  </td>
                  <td className="num">{d.versions.length}</td>
                  <td className="num">{d.cycles}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        A champion's train number is the run its registry entry points at; the test number is the last test-split run of that version. Both are folders under <code>runs/</code>. The benchmark is {stats?.base_total ?? '—'} base tasks in total; {stats?.cycles ?? 0} loop cycles have run, at an estimated ${stats?.cost_usd_est?.toFixed(2) ?? '0.00'} of tokens had they been billed.
      </p>

      <h2>2 · One conversation — tau2 runs the environment, our agent is one factory in its registry</h2>
      <figure>
        <div className="label fig-title">fig 1 · a simulated conversation and where our code sits</div>
        <svg className="dia" viewBox="0 0 1000 340" role="img" aria-label="tau2's orchestrator passes messages between a Haiku user simulator, our agent and the environment; the agent loads system.md and helper.py from the version folder and calls Haiku through the subscription adapter; the evaluator scores the final database, the actions and the communication and writes the result folder.">
          <Arrow />
          <rect className="nd ext" x="14" y="14" width="650" height="312" rx="12" />
          <text className="tx k" x="30" y="38">tau2-bench · dependency, unmodified</text>
          <rect className="nd ext" x="690" y="14" width="296" height="312" rx="12" />
          <text className="tx k" x="706" y="38">tau2_loop · this repo</text>
          <g>
            <title>User simulator — tau2's own, Haiku via the adapter</title>
            <rect className="nd" x="40" y="70" width="170" height="70" rx="10" />
            <text className="tx" x="56" y="98">user simulator</text>
            <text className="tx s" x="56" y="120">haiku 4.5 via adapter</text>
          </g>
          <g>
            <title>Orchestrator — half-duplex turns, 200 steps max</title>
            <rect className="nd hi" x="250" y="70" width="180" height="70" rx="10" />
            <text className="tx" x="266" y="98">orchestrator</text>
            <text className="tx s" x="266" y="120">turns · executes tools</text>
          </g>
          <g>
            <title>Environment — the domain's tools over its database</title>
            <rect className="nd" x="470" y="70" width="170" height="70" rx="10" />
            <text className="tx" x="486" y="98">environment</text>
            <text className="tx s" x="486" y="120">tools · db.json</text>
          </g>
          <g>
            <title>Evaluator — DB hash, actions, communicate, NL judge</title>
            <rect className="nd" x="250" y="220" width="180" height="80" rx="10" />
            <text className="tx" x="266" y="248">evaluator</text>
            <text className="tx s" x="266" y="270">DB hash · actions · said</text>
            <text className="tx s" x="266" y="290">retail: NL judge (haiku)</text>
          </g>
          <g>
            <title>Our agent factory</title>
            <rect className="nd hi" x="716" y="70" width="244" height="70" rx="10" />
            <text className="tx" x="732" y="98">agent factory · tau2_loop</text>
            <text className="tx s" x="732" y="120">JSON tool contract per turn</text>
          </g>
          <g>
            <title>The version folder</title>
            <rect className="nd" x="716" y="170" width="244" height="60" rx="10" />
            <text className="tx" x="732" y="196">agents/&lt;domain&gt;/vN/</text>
            <text className="tx s" x="732" y="216">system.md · helper.py · agent.yaml</text>
          </g>
          <g>
            <title>The subscription adapter</title>
            <rect className="nd" x="716" y="256" width="244" height="56" rx="10" />
            <text className="tx" x="732" y="282">claude-sdk/ provider</text>
            <text className="tx s" x="732" y="302">Agent SDK · no API key</text>
          </g>
          <path className="ed" d="M210 105 L248 105" />
          <path className="ed" d="M430 105 L468 105" />
          <path className="ed hi" d="M430 90 L714 90" />
          <path className="ed" d="M714 120 L432 120" />
          <text className="cap" x="470" y="140">generate_next_message()</text>
          <path className="ed" d="M555 140 L555 260 L432 260" />
          <text className="cap" x="565" y="200">final db</text>
          <path className="ed dash" d="M838 170 L838 142" />
          <path className="ed dash" d="M838 230 L838 254" />
        </svg>
        <figcaption>
          The harness executes tools and scores; our only code in the conversation is the factory that turns <code>system.md</code> and <code>helper.py</code> into the next reply, and the provider that carries every model call to the subscription.
          <span className="path">src/tau2_loop/agent/factory.py · src/tau2_loop/llm/sdk_provider.py · vendor/tau2-bench/src/tau2/orchestrator/</span>
        </figcaption>
      </figure>

      <h2>3 · The learning loop — a failed conversation becomes a diagnosis, a diff, and a verdict</h2>
      <figure>
        <div className="label fig-title">fig 2 · one cycle on one domain, and what feeds the next</div>
        <svg className="dia" viewBox="0 0 1000 300" role="img" aria-label="The loop: the champion's train run yields failed conversations; one optimiser session reads them with the ledger history and writes the next version's two files plus a diagnosis; the challenger is scored on the same train tasks; the gate promotes when a one-sided McNemar test clears p < 0.05; a promotion runs the test split once; the outcome is written back to the ledger the next optimiser reads.">
          <Arrow />
          {[
            ['champion · train', '20 conversations', 20, 'nd'],
            ['failures', 'reward < 1 · errors', 215, 'nd'],
            ['optimiser session', 'sonnet · writes vN+1', 410, 'nd hi'],
            ['challenger · train', 'same 20 tasks', 605, 'nd'],
            ['gate', 'McNemar p < .05', 800, 'nd hi'],
          ].map(([t, s, x, cls]) => (
            <g key={String(t)}>
              <rect className={String(cls)} x={Number(x)} y="60" width="175" height="70" rx="10" />
              <text className="tx" x={Number(x) + 14} y="88">
                {String(t)}
              </text>
              <text className="tx s" x={Number(x) + 14} y="110">
                {String(s)}
              </text>
            </g>
          ))}
          <path className="ed" d="M195 95 L213 95" />
          <path className="ed" d="M390 95 L408 95" />
          <path className="ed" d="M585 95 L603 95" />
          <path className="ed" d="M780 95 L798 95" />
          <rect className="nd" x="20" y="190" width="955" height="60" rx="10" />
          <text className="tx" x="36" y="216">loop/&lt;domain&gt;/ledger.jsonl</text>
          <text className="tx s" x="36" y="238">entry written before the challenger runs · outcome filled after the gate · read by the next optimiser · promotion → one test-split run</text>
          <path className="ed hi" d="M497 130 L497 188" />
          <path className="ed" d="M887 130 L887 188" />
          <path className="ed dash" d="M300 188 L300 130" />
        </svg>
        <figcaption>
          The optimiser never sees the gate's verdict except through the ledger on the next cycle: what worked is a file it reads, not a memory it keeps.
          <span className="path">src/tau2_loop/loop/{'{'}run,optimiser,ledger{'}'}.py · src/tau2_loop/eval/compare.py</span>
        </figcaption>
      </figure>

      <h2>4 · Latest runs</h2>
      <p className="small muted">
        <Link to={runsPath()}>every run, filterable</Link> · <Link to={optimisePath('airline')}>the rounds that produced them</Link> ·{' '}
        <Link to={rubricPath()}>how a conversation is judged</Link>
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
              <th className="num">errors</th>
              <th>note</th>
            </tr>
          </thead>
          <tbody>
            {real
              .slice()
              .reverse()
              .slice(0, 8)
              .map((r) => (
                <tr key={r.run_id}>
                  <td className="sub">
                    <Link to={runPath(r.run_id)}>{shortRun(r.run_id)}</Link>
                  </td>
                  <td>{domainLabel(r.domain)}</td>
                  <td className="mono">{r.agent}</td>
                  <td>{r.split}</td>
                  <td className="num">
                    <Rate passed={r.summary?.passed} n={r.summary?.n_scored} />
                  </td>
                  <td className="num">{r.summary?.errored_ids.length ?? 0}</td>
                  <td className="wrap small muted">{r.note}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      {real.length === 0 && <div className="empty">No scored runs committed yet.</div>}
    </>
  );
}
