import { Link, useNavigate } from 'react-router-dom';
import { type Check, type Checks, DOMAINS, type RunMeta, type VersionHistory, type VersionNode, domainLabel, fmtK, fmtPct, fmtS, shortModel, shortRun, useGet, when } from '../lib/api';
import { Loading, Rate } from '../lib/ui';
import { runPath, useLens } from '../lib/url';
import { modelFamily } from '../lib/scope';
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
          {sentence(
            [
              gate.length ? `${names(gate)} took ${gate.length === 1 ? 'its' : 'their'} champion through the gate` : null,
              hand.length ? `${names(hand)} by hand` : null,
              first.length ? `${names(first)} still ${first.length === 1 ? 'runs its' : 'run their'} first version` : null,
            ]
              .filter(Boolean)
              .join('; '),
          )}{' '}
          A bar is train pass^1, the split the gate reads; the open ring on its track is test, reported and never gated on. The champion's bar is green, a challenger's grey with its verdict after its numbers; the dashed tick is the domain's first champion, on its own cut. A row opens its run.
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
      <p className="small muted" style={{ maxWidth: 'var(--measure)' }}>
        One column per version in build order. A green point held the title, and the green line joins the champions in the order they took it; a dashed stretch crosses from one split to another, so its ends are different tasks. A second green point in one column is a re-run that re-baselined the champion. Under each version: the model its train run used, how it was made (a solid rule is a loop cycle's optimiser, a dashed one a hand-made fork) and the gate's verdict, with the tasks it fixed (+) and broke (−) against the champion, paired by task.
        {still.length ? ` ${names(still)}: one version, no cycle yet.` : ''} A column opens its train run.
      </p>
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
  const model = lens.get('model') ?? '';
  // the scope bar's dataset narrows the figures to it; with no dataset in the address, all four
  const hs = DOMAINS.filter((d) => !domain || d === domain).flatMap((d) => (hist?.[d] ? [hist[d]] : []));
  const split = lens.get('split') ?? '';
  const q = (lens.get('q') ?? '').toLowerCase();
  const all = (runs ?? []).filter((r) => !r.dry_run).slice().reverse();
  const real = all
    .filter((r) => (domain ? r.domain === domain : true))
    .filter((r) => (model ? modelFamily(r.model) === model : true))
    .filter((r) => (split ? r.split === split : true))
    .filter((r) => (q ? `${r.run_id} ${r.agent} ${r.note}`.toLowerCase().includes(q) : true));
  // the scope bar on the LLM judge: its replays, not the answering agent's runs
  if (lens.get('agent') === 'judge') return <JudgeRuns domain={domain || 'airline'} model={model} />;
  // so the table fits the page: a column with nothing to show for the runs listed is left out
  const ks = [1, 2, 3].filter((k) => k === 1 || real.some((r) => r.summary?.pass_hat_k[`pass^${k}`] != null));
  const checks = CHECKS.filter(([key]) => real.some((r) => r.checks?.[key]));
  return (
    <>
      <p className="label">Evaluation runs · the answering agent</p>
      <h1>
        A run is a <em>folder</em>: one row per conversation, every trace, the exact agent files
      </h1>
      <p className="lead">
        Each row is <code>runs/&lt;id&gt;/</code> in the repo: <code>results.jsonl</code> with the reward and each component's verdict,{' '}
        <code>traces/</code> with every message, and tau2's own <code>tau2_results.json</code> so the run can be re-scored offline. MLflow indexes
        the same folders; the snapshot below is what the tracker holds, exported for the public demo.
      </p>
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
      <p className="small muted">
        pass^1 · 2 · 3 are the leaderboard's statistic: the chance that all k trials of a task pass, averaged over its tasks; a run
        of one trial per task has only pass^1. A check's cell is conversations that met every item of it over those that carry
        it; a muted cell is a check τ² records in this domain but does not multiply into the score, and hovering says how many
        items passed. Under a split, "25 × 1 trial" is tasks × trials, and the cut is v1, 20 train / 20 test, or v2, half
        of each base set; two runs on different cuts are not the same tasks.
      </p>
      <div className="tw fit">
        <table>
          <caption>
            A column with nothing to show for the runs listed is left out: pass^2 and pass^3 need a run of 2 or more trials, and a
            check shows only where a listed run carries it. Each run's name starts with its dataset.
          </caption>
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
        04 · MLflow snapshot — {snap?.runs.length ?? 0} tracked runs in experiment {snap?.experiment ?? '—'}
      </h2>
      {snap && snap.runs.length > 0 ? (
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
              {snap.runs.map((r) => (
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
          No snapshot committed. <code>make snapshot</code> exports the experiment.
        </div>
      )}
      <p className="small muted">"cost est." is litellm's per-token estimate for the model id; every call ran on the subscription, so the amount paid was $0.</p>
    </>
  );
}
