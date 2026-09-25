import { Link, useParams } from 'react-router-dom';
import {
  type RunMeta,
  type TaskResult,
  domainLabel,
  fmtK,
  fmtPct,
  fmtS,
  shortModel,
  shortRun,
  shortTask,
  useGet,
  when,
} from '../lib/api';
import { Loading, Rate } from '../lib/ui';
import { agentPath, runPath, runsPath, trialId, trialPath, useLens } from '../lib/url';

/**
 * One run, and — with `?vs=<run>` — the gate against another. s04 M3 folded the
 * old Gate tab in here: a promotion verdict is a fact about a pair of runs, so
 * it belongs at the run's address rather than at a tab with no subject.
 */

type Profile = {
  n: number;
  metrics: Record<string, { mean: number; p50: number; p95: number; max: number; sum: number }>;
  cost_per_pass: number | null;
  cost_per_conversation: number | null;
  tokens_per_pass: number | null;
  error_rate: number | null;
  fail_rate: number | null;
  hit_turn_cap: number;
  hit_turn_cap_rate: number | null;
};
type RunPayload = { meta: RunMeta; results: TaskResult[]; profile: Profile };
type Verdict = {
  promote: boolean;
  champion_passed: number;
  challenger_passed: number;
  n: number;
  fixed: string[];
  broken: string[];
  p_value: number;
  alpha: number;
  reason: string;
};
type ComparePayload = {
  verdict: Verdict;
  rows: { task_id: string; trial: number; purpose: string; a: TaskResult | null; b: TaskResult | null }[];
  a: RunMeta;
  b: RunMeta;
  profiles: { a: Profile; b: Profile };
};

/** The metrics worth a row, and what each one answers. */
const PROFILE_ROWS: [string, string][] = [
  ['turns', 'agent turns'],
  ['tool_calls', 'tool calls'],
  ['wall_s', 'seconds'],
  ['agent_in', 'agent tokens in'],
  ['agent_out', 'agent tokens out'],
  ['user_in', 'user tokens in'],
  ['cost_usd', 'cost est. $'],
];

function Verdict_({ r }: { r: TaskResult }) {
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
  const [lens, setLens] = useLens();
  const vs = lens.get('vs') ?? '';
  const verdictFilter = lens.get('v') ?? '';
  const q = (lens.get('q') ?? '').toLowerCase();
  const { data, error } = useGet<RunPayload>(`/api/runs/${encodeURIComponent(runId)}`);
  const { data: runs } = useGet<RunMeta[]>('/api/runs');
  const { data: cmp } = useGet<ComparePayload>(
    vs ? `/api/compare?a=${encodeURIComponent(vs)}&b=${encodeURIComponent(runId)}` : null,
  );
  if (!data) return <Loading error={error} />;
  const { meta, results, profile } = data;
  const s = meta.summary;

  const rows = results
    .filter((r) =>
      verdictFilter === 'pass'
        ? r.correct === true
        : verdictFilter === 'fail'
          ? r.correct === false
          : true,
    )
    .filter((r) => (q ? `${r.task_id} ${r.purpose}`.toLowerCase().includes(q) : true));

  // only runs of the same domain and split can be paired task by task
  const pairable = (runs ?? []).filter(
    (r) => r.run_id !== runId && r.domain === meta.domain && r.split === meta.split && r.summary?.n_scored,
  );

  return (
    <>
      <p className="crumbs">
        <Link to={runsPath()}>runs</Link> / {domainLabel(meta.domain)} / {meta.agent}
      </p>
      <h1>
        {meta.agent} on {domainLabel(meta.domain)} {meta.split}:{' '}
        <em>{s?.n_scored ? `${s.passed} of ${s.n_scored}` : `${meta.n_tasks} run`}</em>
      </h1>
      <p className="lead">
        {shortModel(meta.model)} agent · {shortModel(meta.user_model)} user · {meta.trials} trial
        {meta.trials > 1 ? 's' : ''} · concurrency {meta.concurrency} · seed {meta.seed} · started{' '}
        {when(meta.started_at)} · {fmtS(s?.duration_ms)} of conversation · fingerprint{' '}
        <code>{meta.fingerprint}</code>
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
        <span className={`chip ${s?.errored_ids.length ? 'warn' : ''}`}>
          errors {s?.errored_ids.length ?? 0}
        </span>
        {meta.mlflow_url && (
          <a className="chip" href={meta.mlflow_url}>
            MLflow run
          </a>
        )}
      </div>

      <h2>1 · What it cost, as a distribution</h2>
      <p>
        Totals hide the tail that decides whether a full-split run fits a subscription window. The
        median conversation here takes {profile.metrics.turns.p50} agent turns; the 95th takes{' '}
        {profile.metrics.turns.p95}.
        {profile.hit_turn_cap > 0
          ? ` ${profile.hit_turn_cap} never got to answer: they ran out of turns.`
          : ' None ran out of turns.'}
      </p>
      <div className="tw">
        <table>
          <caption>
            {profile.n} conversations ({meta.n_tasks} tasks × {meta.trials} trial
            {meta.trials > 1 ? 's' : ''}). Nearest-rank percentiles, so every figure is a real
            conversation's.
          </caption>
          <thead>
            <tr>
              <th>metric</th>
              <th className="num">mean</th>
              <th className="num">p50</th>
              <th className="num">p95</th>
              <th className="num">max</th>
              <th className="num">total</th>
            </tr>
          </thead>
          <tbody>
            {PROFILE_ROWS.map(([key, label]) => {
              const m = profile.metrics[key];
              if (!m) return null;
              const fmt = (x: number) =>
                key === 'cost_usd'
                  ? `$${x.toFixed(x < 1 ? 3 : 2)}`
                  : key.includes('_in') || key.includes('_out')
                    ? fmtK(x)
                    : Math.round(x * 100) / 100;
              return (
                <tr key={key}>
                  <td className="sub">{label}</td>
                  <td className="num">{fmt(m.mean)}</td>
                  <td className="num">{fmt(m.p50)}</td>
                  <td className="num">{fmt(m.p95)}</td>
                  <td className="num">{fmt(m.max)}</td>
                  <td className="num">{fmt(m.sum)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        cost per passing conversation{' '}
        {profile.cost_per_pass != null ? `$${profile.cost_per_pass.toFixed(3)}` : '—'} · per
        conversation attempted{' '}
        {profile.cost_per_conversation != null
          ? `$${profile.cost_per_conversation.toFixed(3)}`
          : '—'}{' '}
        · tokens per pass {fmtK(profile.tokens_per_pass)}. Every call ran on the subscription, so the
        amount paid was $0; these are litellm's per-token estimates.
      </p>

      <h2>2 · Every conversation</h2>
      <div className="filters">
        <label className="pick">
          <span className="label">verdict</span>
          <select value={verdictFilter} onChange={(e) => setLens({ v: e.target.value })}>
            <option value="">all</option>
            <option value="pass">pass</option>
            <option value="fail">fail</option>
          </select>
        </label>
        <input
          type="search"
          placeholder="search task or purpose"
          value={lens.get('q') ?? ''}
          onChange={(e) => setLens({ q: e.target.value })}
        />
        <label className="pick">
          <span className="label">against</span>
          <select value={vs} onChange={(e) => setLens({ vs: e.target.value })}>
            <option value="">— no comparison —</option>
            {pairable.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {shortRun(r.run_id)} · {r.agent}
              </option>
            ))}
          </select>
        </label>
        <span className="count">
          {rows.length} of {results.length}
        </span>
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
            {rows.map((r) => (
              <tr key={`${r.task_id}#${r.trial}`}>
                <td className="sub mono small" title={r.task_id}>
                  {shortTask(r.task_id, 28)}
                  {meta.trials > 1 && <span className="path">trial {r.trial}</span>}
                </td>
                <td>
                  <Verdict_ r={r} />
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
                  {r.error && r.error !== r.termination_reason && (
                    <span className="path v-warn">{r.error.slice(0, 80)}</span>
                  )}
                </td>
                <td>
                  <Link to={trialPath(runId, trialId(r))}>open</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {vs && cmp && <Gate cmp={cmp} focus={runId} />}

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

function Gate({ cmp, focus }: { cmp: ComparePayload; focus: string }) {
  const v = cmp.verdict;
  const sameShape = cmp.a.domain === cmp.b.domain && cmp.a.split === cmp.b.split;
  const discordant = v.fixed.length + v.broken.length;
  return (
    <>
      <h2>
        3 · The gate — {shortRun(cmp.a.run_id)} vs {shortRun(cmp.b.run_id)}:{' '}
        <span className={v.promote ? 'v-ok' : 'v-warn'}>{v.promote ? 'promote' : 'hold'}</span>
      </h2>
      <p>
        Two runs of the same tasks, paired by task; only the discordant pairs count — {v.fixed.length}{' '}
        fixed, {v.broken.length} broken of {discordant}. Under "no real difference" each is a coin
        flip, so p = P(breaks ≤ {v.broken.length} | {discordant}, ½) = {v.p_value.toFixed(3)}, and the
        gate promotes at p &lt; {v.alpha}. With twenty tasks that is blunt by construction: five fixes
        and no breaks is the smallest result that clears it.
      </p>
      {!sameShape && (
        <p className="warn-note v-warn">
          These two runs are not the same domain and split, so the pairing below is not a gate.
        </p>
      )}
      <div className="kpis">
        <div className="kpi">
          <div className="label">{cmp.a.agent} · champion</div>
          <div className="n">
            {v.champion_passed}/{v.n}
          </div>
          <div className="b">
            <Link to={runPath(cmp.a.run_id)}>{shortRun(cmp.a.run_id)}</Link>
          </div>
        </div>
        <div className="kpi">
          <div className="label">{cmp.b.agent} · challenger</div>
          <div className={`n ${v.challenger_passed > v.champion_passed ? 'ok' : ''}`}>
            {v.challenger_passed}/{v.n}
          </div>
          <div className="b">
            <Link to={runPath(cmp.b.run_id)}>{shortRun(cmp.b.run_id)}</Link>
          </div>
        </div>
        <div className="kpi">
          <div className="label">p-value</div>
          <div className={`n ${v.promote ? 'ok' : 'warn'}`}>{v.p_value.toFixed(3)}</div>
          <div className="b">{v.reason}</div>
        </div>
        <div className="kpi">
          <div className="label">cost, champion → challenger</div>
          <div className="n">
            {cmp.profiles.a.cost_per_pass != null && cmp.profiles.b.cost_per_pass != null
              ? `$${cmp.profiles.a.cost_per_pass.toFixed(3)} → $${cmp.profiles.b.cost_per_pass.toFixed(3)}`
              : '—'}
          </div>
          <div className="b">per passing conversation</div>
        </div>
      </div>
      <div className="tw">
        <table>
          <caption>
            One row per task. A row where the two disagree is what the test is computed from.
          </caption>
          <thead>
            <tr>
              <th>task</th>
              <th>purpose</th>
              <th>{shortRun(cmp.a.run_id)}</th>
              <th>{shortRun(cmp.b.run_id)}</th>
              <th>change</th>
            </tr>
          </thead>
          <tbody>
            {cmp.rows.map((r) => {
              const change =
                r.a?.correct === r.b?.correct
                  ? r.a?.correct
                    ? 'held'
                    : 'still failing'
                  : r.b?.correct
                    ? 'fixed'
                    : 'broken';
              return (
                <tr key={`${r.task_id}#${r.trial}`} className={change === 'broken' ? 'warnrow' : ''}>
                  <td className="sub mono small" title={r.task_id}>
                    {r.b ? (
                      <Link to={trialPath(focus, trialId(r.b))}>{shortTask(r.task_id, 26)}</Link>
                    ) : (
                      shortTask(r.task_id, 26)
                    )}
                  </td>
                  <td className="wrap small muted">{r.purpose}</td>
                  <td>{r.a ? <Verdict_ r={r.a} /> : <span className="muted">—</span>}</td>
                  <td>{r.b ? <Verdict_ r={r.b} /> : <span className="muted">—</span>}</td>
                  <td
                    className={
                      change === 'fixed'
                        ? 'v-ok'
                        : change === 'broken'
                          ? 'v-warn'
                          : change === 'held'
                            ? 'v-no'
                            : 'muted'
                    }
                  >
                    {change}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        pass rate {fmtPct((v.champion_passed || 0) / (v.n || 1))} →{' '}
        {fmtPct((v.challenger_passed || 0) / (v.n || 1))} ·{' '}
        <Rate passed={v.challenger_passed} n={v.n} />
      </p>
    </>
  );
}
