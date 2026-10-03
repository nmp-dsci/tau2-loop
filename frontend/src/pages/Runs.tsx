import { Link, useNavigate } from 'react-router-dom';
import { type Check, type Checks, DOMAINS, type RunMeta, type VersionHistory, type VersionNode, domainLabel, fmtK, fmtPct, fmtS, shortModel, shortRun, useGet, when } from '../lib/api';
import { Loading, Points, Rate } from '../lib/ui';
import { runPath, trialPath, useLens } from '../lib/url';
import { useExp, useScope, useTask } from '../lib/scope';
import { JudgeRuns } from './JudgeRuns';
import { DomainBars, VersionsFig, standing, versionsWidth } from '../lib/versions';

/** The reward checks, in the order τ² multiplies them, with the name a column shows. */
const CHECKS: [keyof Checks, string, string][] = [
  ['db', 'DB', 'the final database equals the gold one'],
  ['actions', 'actions', 'every expected action was called with the expected arguments'],
  ['communicate', 'said', 'the agent told the user each fact the task requires'],
  ['nl', 'NL', "an LLM judge finds the task's NL assertions met"],
  ['env', 'env', "the environment's assertions hold at the end (telecom)"],
];

/** `banking_knowledge_v2_train`, breakable after each underscore, so a long run name wraps instead of widening its column. */
const breakable = (s: string) => s.split(/(?<=_)/).flatMap((part, i) => (i ? [<wbr key={i} />, part] : [part]));

const PASS_K: Record<number, string> = {
  1: 'the probability that one trial of a task passes, averaged over tasks',
  2: 'the probability that 2 trials of a task all pass; needs a run of 2 or more trials',
  3: 'the probability that 3 trials of a task all pass; needs a run of 3 or more trials',
};

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
  // a run is scored when it finishes: until then it has no summary to show
  if (k === 1 && !r.summary && !r.finished_at) {
    return (
      <td className="num muted" title="scored when the run finishes">
        running
      </td>
    );
  }
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

type TaskConv = {
  run_id: string;
  agent: string;
  split: string;
  model: string;
  started_at: string;
  trial: number;
  correct: boolean | null;
  reward: number;
  db_check: boolean | null;
  action_checks: string | null;
  communicate_checks: string | null;
  n_agent_turns: number;
  n_tool_calls: number;
  termination_reason: string;
  duration_ms: number;
};

/** The scope bar's task in every run in scope: its conversation in each, one click from the trace. */
function TaskRuns({ domain, task, runs }: { domain: string; task: string; runs: RunMeta[] }) {
  const { data, error } = useGet<TaskConv[]>(`/api/domains/${encodeURIComponent(domain)}/conversations?task=${encodeURIComponent(task)}`);
  if (!data) return <Loading error={error} />;
  const inScope = new Set(runs.map((r) => r.run_id));
  const rows = data.filter((c) => inScope.has(c.run_id));
  const passed = rows.filter((c) => c.correct).length;
  return (
    <>
      <h2>
        Task {task} — passed in {passed} of {rows.length} conversations across the runs in scope
      </h2>
      <div className="tw fit">
        <table>
          <caption>Newest run first; a row opens that conversation’s trace.</caption>
          <thead>
            <tr>
              <th>run</th>
              <th>agent</th>
              <th>verdict</th>
              <th>DB</th>
              <th className="num">actions</th>
              <th className="num">agent turns</th>
              <th>ended</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => (
              <tr key={`${c.run_id}#${c.trial}`}>
                <td className="sub mono small">
                  <Link to={trialPath(c.run_id, `${task}/t${c.trial}`)}>{breakable(c.run_id.replace(/^\d{8}T\d{6}Z_/, ''))}</Link>
                  <span className="path">
                    {when(c.started_at)} · t{c.trial}
                  </span>
                </td>
                <td className="small">
                  {c.agent}
                  <span className="path">{shortModel(c.model)}</span>
                </td>
                <td>
                  <span className={`status ${c.correct ? 'ok' : c.correct === false ? 'err' : 'no'}`}>
                    {c.correct ? 'pass' : c.correct === false ? 'fail' : 'unscored'}
                  </span>
                </td>
                <td className="small">{c.db_check == null ? <span className="dim">—</span> : c.db_check ? 'matches gold' : 'differs'}</td>
                <td className="num">{c.action_checks ?? '—'}</td>
                <td className="num">{c.n_agent_turns}</td>
                <td className="small">{c.termination_reason}</td>
              </tr>
            ))}
            {!rows.length && (
              <tr>
                <td colSpan={7} className="dim">
                  no run in scope played task {task}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}

const names = (hs: VersionHistory[]) => hs.map((h) => domainLabel(h.domain)).join(', ');
const sentence = (s: string) => (s ? `${s[0].toUpperCase()}${s.slice(1)}.` : '');

/** 01 and 02: the champion and its challenger by domain, then each domain's champion over its versions. */
function ChampionFigs({ hs }: { hs: VersionHistory[] }) {
  const nav = useNavigate();
  const open = (v: VersionNode) => v.train?.run_id && nav(runPath(v.train.run_id));
  const gated = hs.flatMap((h) => h.versions.filter((v) => v.made.cycle != null));
  const promoted = gated.filter((v) => v.verdict === 'promote').length;
  const by = (verdict: string) => hs.filter((h) => standing(h).champ?.verdict === verdict);
  const [gate, hand, first] = [by('promote'), by('by hand'), by('first')];
  const moving = hs.filter((h) => h.versions.length > 1);
  const still = hs.filter((h) => h.versions.length <= 1);
  const titles = (h: VersionHistory) => h.reigns.map((r) => r.version).filter((v, i, xs) => v !== xs[i - 1]);
  const changed = hs.filter((h) => titles(h).length > 1);
  const viaGate = hs.filter((h) => h.reigns.some((r) => r.kind === 'gate')).length;
  return (
    <>
      <h2>
        01 · By domain — {promoted} of {gated.length} gated challenger{gated.length === 1 ? '' : 's'} took the title
      </h2>
      <figure>
        <div className="label fig-title">fig 1 · pass^1 by domain: the champion, and the newest challenger after it, on train and test</div>
        <DomainBars hs={hs} onPick={(_, v) => open(v)} />
        <figcaption>
          <Points
            items={[
              <b key="who">
                {sentence(
                  [
                    gate.length ? `${names(gate)} took ${gate.length === 1 ? 'its' : 'their'} champion through the gate` : null,
                    hand.length ? `${names(hand)} crowned ${hand.length === 1 ? 'its' : 'their'} champion by hand` : null,
                    first.length ? `${names(first)} still ${first.length === 1 ? 'runs its' : 'run their'} first version` : null,
                  ]
                    .filter(Boolean)
                    .join('; '),
                )}
              </b>,
              <>
                <b>A bar is train pass^1</b>, the split the gate reads; the ring is test, never gated on.
              </>,
              <>
                <b>Green is the champion</b>, grey a challenger with its verdict. A row opens its run.
              </>,
            ]}
          />
          <span className="path">runs/*/run.json · loop/&lt;domain&gt;/registry.json · loop/&lt;domain&gt;/ledger.jsonl → /api/versions</span>
        </figcaption>
      </figure>

      <h2>
        02 · The champion over time — {changed.length} of {hs.length} domain{hs.length === 1 ? '' : 's'} changed champion, {viaGate} through the gate
      </h2>
      <div className="figrow">
        {moving.map((h, i) => (
          <figure key={h.domain} style={{ flex: `1 1 ${versionsWidth(h)}px` }}>
            <div className="label fig-title">
              fig {i + 2} · {domainLabel(h.domain)}: {titles(h).join(' → ')} — train pass^1 of every version, the line joining the champions
            </div>
            <VersionsFig h={h} mode="champion" onPick={open} />
          </figure>
        ))}
      </div>
      <Points
        className="small muted"
        items={[
          <>
            <b>One column per version</b>, in build order; the green line joins the champions as they took the title.
          </>,
          <>
            <b>Under each version</b>: its model, how it was made (solid: the loop, dashed: by hand) and the gate's verdict, tasks
            fixed (+) and broken (−).
          </>,
          <>
            <b>A column opens its train run.</b>
            {still.length ? ` ${names(still)}: one version, no cycle yet.` : ''}
          </>,
        ]}
      />
    </>
  );
}

type Snapshot = { experiment: string | null; tracking_uri?: string; runs: { mlflow_run_id?: string; name: string; tags: Record<string, string>; metrics: Record<string, number>; params: Record<string, string> }[] };

export function Runs() {
  const [lens, setLens] = useLens();
  const { data: runs } = useGet<RunMeta[]>('/api/runs');
  const { data: snap } = useGet<Snapshot>('/api/experiments');
  const { data: hist, error: histError } = useGet<Record<string, VersionHistory>>('/api/versions');
  const domain = lens.get('domain') ?? '';
  const exp = useExp();
  const task = useTask();
  const { scope } = useScope();
  // the scope bar's dataset narrows the figures to it; with no dataset in the address, all four
  const hs = DOMAINS.filter((d) => !domain || d === domain).flatMap((d) => (hist?.[d] ? [hist[d]] : []));
  const split = lens.get('split') ?? '';
  const q = (lens.get('q') ?? '').toLowerCase();
  const all = (runs ?? []).filter((r) => !r.dry_run).slice().reverse();
  const real = all
    .filter((r) => (domain ? r.domain === domain : true))
    // the scope bar's experiment: that version's runs, train, test and custom
    .filter((r) => (exp ? r.agent === exp : true))
    // the scope bar's task, which belongs to one dataset: the runs that played it
    .filter((r) => !task || (r.domain === (domain || scope.dataset) && (r.task_ids ?? []).includes(task)))
    .filter((r) => (split ? r.split === split : true))
    .filter((r) => (q ? `${r.run_id} ${r.agent} ${r.note}`.toLowerCase().includes(q) : true));
  // the MLflow snapshot under the same scope: an eval by its agent, a loop cycle by either side of it
  const tracked = (snap?.runs ?? [])
    .filter((r) => !domain || r.tags.domain === domain)
    .filter((r) => !exp || [r.tags.agent, r.tags.challenger, r.tags.champion].includes(exp));
  // the scope bar on the LLM judge: its replays, not the answering agent's runs
  if (lens.get('agent') === 'judge') return <JudgeRuns domain={domain || 'airline'} />;
  // so the table fits the page: a column with nothing to show for the runs listed is left out
  const ks = [1, 2, 3].filter((k) => k === 1 || real.some((r) => r.summary?.pass_hat_k[`pass^${k}`] != null));
  const checks = CHECKS.filter(([key]) => real.some((r) => r.checks?.[key]));
  return (
    <>
      <p className="label">Evaluation runs · the answering agent</p>
      <h1>
        A run is a <em>folder</em>: one row per conversation, every trace, the exact agent files
      </h1>
      <Points
        lead
        items={[
          <>
            <b>
              Each row is a folder, <code>runs/&lt;id&gt;/</code>
            </b>
            : every conversation's reward, every trace, the agent's exact files.
          </>,
          <>
            <b>It re-scores offline</b> from tau2's own <code>tau2_results.json</code>.
          </>,
          <>
            <b>MLflow indexes the same folders</b>; its snapshot is at the foot of the page.
          </>,
        ]}
      />
      {/* the scope bar's task first: its conversation in each run, one click from the trace */}
      {task && <TaskRuns domain={domain || scope.dataset} task={task} runs={real} />}
      {hist ? <ChampionFigs hs={hs} /> : <Loading error={histError} />}

      <h2>
        03 · Every run — {all.length} folders under <code>runs/</code>, each re-scorable offline
      </h2>
      <div className="filters">
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
      <Points
        className="small muted"
        items={[
          <>
            <b>pass^k</b> is the leaderboard's statistic: the chance that all k trials of a task pass.
          </>,
          <>
            <b>A check's cell</b> is conversations that met all of it; muted when τ² does not score it here.
          </>,
          <>
            <b>"25 × 1 trial · cut v2"</b> is tasks × trials and the split's version; different cuts are different tasks.
          </>,
        ]}
      />
      <div className="tw fit">
        <table>
          <caption>Empty columns are left out; each run's name starts with its dataset.</caption>
          <thead>
            <tr>
              <th>run</th>
              <th>agent</th>
              <th>split</th>
              {ks.map((k) => (
                <th key={k} className="num" title={PASS_K[k]}>
                  pass^{k}
                </th>
              ))}
              {checks.map(([key, label, what]) => (
                <th key={key} className="num" title={what}>
                  {label}
                </th>
              ))}
              <th className="num">errors</th>
              <th className="num">turns / conv</th>
              <th className="num">tokens / conv</th>
              <th className="num">time</th>
              <th>note</th>
            </tr>
          </thead>
          <tbody>
            {real.map((r) => (
              <tr key={r.run_id}>
                <td className="sub">
                  <Link to={runPath(r.run_id)}>{breakable(shortRun(r.run_id))}</Link>
                  <span className="path nw">{when(r.started_at)}</span>
                  <span className="path">
                    {shortModel(r.model)}
                    {r.agent_route?.startsWith('service') ? ' via service' : ''}
                  </span>
                  <span className="path">user {shortModel(r.user_model)}</span>
                </td>
                <td className="mono">
                  {r.agent}
                  <span className="path">{r.fingerprint}</span>
                </td>
                <td className="sub">
                  {r.split}
                  <span className="path nw">
                    {r.n_tasks} × {r.trials} trial{r.trials > 1 ? 's' : ''}
                  </span>
                  {r.split_version ? <span className="path">cut v{r.split_version}</span> : null}
                </td>
                {ks.map((k) => (
                  <PassK key={k} r={r} k={k} />
                ))}
                {checks.map(([key, , what]) => (
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
        04 · MLflow snapshot — {tracked.length} of {snap?.runs.length ?? 0} tracked runs in experiment {snap?.experiment ?? '—'}
      </h2>
      {snap && tracked.length > 0 ? (
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
              {tracked.map((r) => (
                <tr key={r.mlflow_run_id ?? r.name}>
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
          {snap && snap.runs.length > 0 ? (
            'No tracked run in this scope.'
          ) : (
            <>
              No snapshot committed. <code>make snapshot</code> exports the experiment.
            </>
          )}
        </div>
      )}
      <p className="small muted">"cost est." is litellm's estimate; on the subscription the amount paid was $0.</p>
    </>
  );
}
