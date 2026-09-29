import { useEffect, useRef, useState } from 'react';
import { Link, Outlet, useNavigate, useParams } from 'react-router-dom';
import {
  DOMAINS,
  type AgentInfo,
  type Diagnosis,
  type LedgerEntry,
  type Registries,
  type RunMeta,
  type VersionHistory,
  domainLabel,
  fmtK,
  fmtS,
  shortRun,
  shortTask,
  useGet,
} from '../lib/api';
import { Loading } from '../lib/ui';
import { agentPath, optimisePath, runPath, useLens } from '../lib/url';
import { CycleDumbbell, VersionsFig, frac, standing } from '../lib/versions';

/**
 * The loop, as rounds. s04 M3 merged the old Loop tab (the ledger) and the old
 * Evolution tab (the diff between two versions), which were the two halves of
 * one story told in two places: a round is a diagnosis, the change it produced,
 * and what the gate then decided.
 *
 * Rule 3 of the grammar: a round opens inside the list, at
 * `/optimise/<domain>/<version>`, and the list stays mounted behind it.
 */

type Side = { name: string; fingerprint: string; runs: RunMeta[] };
type DiffFile = {
  name: string;
  changed: boolean;
  added: number;
  removed: number;
  diff: string[];
  before: string;
  after: string;
};
type Change = { file: string; anchor: string; what: string; why: string; task_ids: string[] };
type DiffPayload = {
  domain: string;
  a: Side;
  b: Side;
  files: DiffFile[];
  closing_account: string | null;
  diagnosis: {
    diagnoses: Diagnosis[];
    prompt_diff_summary: string;
    helper_diff_summary: string;
    expected_to_fix: string[];
    risks: string[];
    changes?: Change[];
    /** a `fork`: `model swap`, the version it copied, the agent.yaml lines that changed */
    kind?: string;
    forked_from?: string;
    agent_yaml?: string[];
  } | null;
  cycles: LedgerEntry[];
};

type Hunk = { header: string; lines: string[] };

function hunks(diff: string[]): Hunk[] {
  const out: Hunk[] = [];
  for (const line of diff.slice(2)) {
    if (line.startsWith('@@')) out.push({ header: line, lines: [] });
    else out[out.length - 1]?.lines.push(line);
  }
  return out;
}

const norm = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, ' ')
    .trim();

function commentaryFor(h: Hunk, changes: Change[], file: string): Change[] {
  const body = norm(h.lines.filter((l) => l.startsWith('+') || l.startsWith('-')).join(' '));
  return changes.filter(
    (c) => c.file === file && c.anchor && body.includes(norm(c.anchor).split(' ').slice(0, 5).join(' ')),
  );
}

function best(runs: RunMeta[], split: string): RunMeta | undefined {
  return runs.filter((r) => r.split === split && r.summary?.n_scored).slice(-1)[0];
}

/** A cycle no optimiser wrote (`make challenge`), named for what it changed:
 * `model swap v3 → v5 (Claude Opus 5.5, medium)`. Null for a loop cycle. */
function handMade(e: LedgerEntry | undefined): string | null {
  if (!e?.kind) return null;
  const model = [e.challenger_model, e.challenger_effort].filter(Boolean).join(', ');
  return `${e.kind} ${e.forked_from ?? e.champion} → ${e.challenger ?? '—'}${model ? ` (${model})` : ''}`;
}

/** What the gate did with one task, by the ledger's own three lists. */
function taskOutcome(e: LedgerEntry | undefined, taskId: string): string {
  const o = e?.outcome;
  if (!o) return 'pending';
  if (o.fixed?.includes(taskId)) return 'fixed';
  if (o.broken?.includes(taskId)) return 'broken';
  if (o.still_failed?.includes(taskId)) return 'still failed';
  return o.verdict === 'pending' ? 'pending' : 'unchanged';
}

/** Fig 1 · one domain's versions in build order, train and test, and the champion each faced. */
function VersionsSection({ h, open }: { h: VersionHistory; open: string | null }) {
  const nav = useNavigate();
  const { champ } = standing(h);
  const cyc = h.versions.filter((v) => v.made.cycle != null);
  const held = cyc.filter((v) => v.verdict === 'hold').length;
  const aboveHeld = cyc.filter((v) => v.verdict === 'hold' && v.train && v.vs?.train && v.train.passed > v.vs.train.passed).length;
  return (
    <>
      <h2>
        The versions — {champ ? `${champ.version} holds ${domainLabel(h.domain)} at ${frac(champ.train)} train, ${frac(champ.test)} test` : 'no champion yet'}
        {cyc.length ? `; ${held} of ${cyc.length} challengers held` : ''}
      </h2>
      <figure>
        <div className="label fig-title">
          fig 1 · every {domainLabel(h.domain)} version in build order: train (filled), test (open), and the champion each challenger faced (green tick)
        </div>
        <VersionsFig
          h={h}
          mode="rounds"
          open={open}
          onPick={(v) => nav(v.made.kind === 'base' ? agentPath(h.domain, v.version) : optimisePath(h.domain, v.version))}
        />
        <figcaption>
          {aboveHeld
            ? `${aboveHeld} of ${cyc.length} challengers beat the champion's train count and were still held: the gate reads the tasks fixed and broken, not the count. `
            : ''}
          The solid line is train pass^1, what the gate reads; the dashed one is test, run once per challenger and reported only. Neither line crosses from one split to the next. The green tick is the champion's train score each challenger was gated against; + is tasks fixed and − tasks broken against it. A column opens its round.
          <span className="path">agents/{h.domain}/v*/ · loop/{h.domain}/ledger.jsonl · loop/{h.domain}/registry.json · runs/*/run.json → /api/versions</span>
        </figcaption>
      </figure>
    </>
  );
}

function DiagnosisTable({ rows, cycle }: { rows: Diagnosis[]; cycle?: LedgerEntry }) {
  return (
    <div className="tw">
      <table>
        <caption>
          One row per failed conversation the optimiser read. `outcome` is what the gate found
          afterwards — the only column the optimiser did not write.
        </caption>
        <thead>
          <tr>
            <th>task</th>
            <th>symptom</th>
            <th>root cause</th>
            <th>surface</th>
            <th>change</th>
            <th>verified</th>
            <th>outcome</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((d, i) => {
            const res = taskOutcome(cycle, d.task_id);
            return (
              <tr key={d.task_id + d.surface + i}>
                <td className="sub mono small wrap" title={d.task_id} style={{ maxWidth: '18ch' }}>
                  {shortTask(d.task_id, 28)}
                </td>
                <td className="wrap">{d.symptom}</td>
                <td className="wrap">{d.root_cause}</td>
                <td className="mono">{d.surface}</td>
                <td className="wrap">{d.change}</td>
                <td>
                  {d.verified_in_session ? (
                    <span className="v-ok">yes</span>
                  ) : (
                    <span className="v-warn">no</span>
                  )}
                </td>
                <td
                  className={
                    res === 'fixed' ? 'v-ok' : res === 'unchanged' || res === 'pending' ? 'v-no' : 'v-warn'
                  }
                >
                  {res}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function Optimise() {
  const { domain = 'airline', version } = useParams();
  const nav = useNavigate();
  const { data: ledger, error } = useGet<Record<string, LedgerEntry[]>>('/api/ledger');
  const { data: hist } = useGet<Record<string, VersionHistory>>('/api/versions');
  if (!ledger) return <Loading error={error} />;
  const entries = ledger[domain] ?? [];
  const counts = Object.fromEntries(DOMAINS.map((d) => [d, (ledger[d] ?? []).length]));
  const hs = DOMAINS.flatMap((d) => (hist?.[d] ? [hist[d]] : []));
  const gated = hs.flatMap((h) => h.versions.filter((v) => v.made.cycle != null));
  const promoted = gated.filter((v) => v.verdict === 'promote').length;
  const heldN = gated.filter((v) => v.verdict === 'hold').length;
  const mixed = gated.filter((v) => v.verdict === 'hold' && (v.fixed ?? 0) > 0 && (v.broke ?? 0) > 0).length;
  return (
    <>
      <nav className="chips domainbar" aria-label="domains">
        {DOMAINS.map((d) => (
          <Link
            key={d}
            to={optimisePath(d)}
            className={`chip nav ${d === domain ? 'on' : ''}`}
            aria-current={d === domain ? 'page' : undefined}
          >
            {domainLabel(d)}
            <span className="n">{counts[d] ?? 0}</span>
          </Link>
        ))}
      </nav>
      <p className="label">The loop</p>
      <h1>
        A failed conversation becomes a diagnosis, a diff, and a <em>verdict</em> — and the ledger
        keeps all three
      </h1>
      <p className="lead">
        This is <code>loop/{domain}/ledger.jsonl</code>, rendered as rounds. The optimiser writes the
        diagnoses and the change; the harness writes the outcome after the gate. The next cycle's
        optimiser is shown this page's contents before it proposes anything, which is what stops the
        loop repeating a fix that already failed.
      </p>

      {hist?.[domain] && hist[domain].versions.length > 1 && <VersionsSection h={hist[domain]} open={version ?? null} />}

      {entries.length === 0 && (
        <div className="empty">
          No cycle has run on {domainLabel(domain)}. <code>make loop DOMAIN={domain}</code> starts
          one.
        </div>
      )}

      <Outlet />

      {entries.length > 0 && (
        <div className="tw">
          <table>
            <caption>Pick a round to read its diagnosis, its diff and its outcome.</caption>
            <thead>
              <tr>
                <th className="num">cycle</th>
                <th>champion → challenger</th>
                <th>verdict</th>
                <th className="num">train</th>
                <th className="num">test</th>
                <th>what changed</th>
                <th>optimiser</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e) => (
                <tr
                  key={e.cycle}
                  className={`click ${e.challenger === version ? 'pro' : ''}`}
                  onClick={() => e.challenger && nav(optimisePath(domain, e.challenger))}
                >
                  <td className="num">{e.cycle}</td>
                  <td className="sub mono">
                    {e.champion} → {e.challenger ?? '—'}
                  </td>
                  <td
                    className={
                      e.outcome?.verdict === 'promote'
                        ? 'v-ok'
                        : e.outcome?.verdict === 'hold'
                          ? 'v-warn'
                          : 'v-no'
                    }
                  >
                    {e.outcome?.verdict ?? 'pending'}
                  </td>
                  <td className="num">{e.outcome?.passes ?? '—'}</td>
                  <td className="num">{e.outcome?.test_passes ?? '—'}</td>
                  <td className="wrap small">
                    {e.kind && e.agent_yaml?.length
                      ? `agent.yaml: ${e.agent_yaml.join('; ')}`
                      : e.prompt_diff_summary || '—'}
                  </td>
                  {/* a hand-made row names its fork in the path line, which must wrap, not clip */}
                  <td className={`small ${handMade(e) ? 'wrap' : 'nw'}`}>
                    {handMade(e) ? 'none' : (e.optimiser_model ?? '—')}
                    {handMade(e) && <span className="path">{handMade(e)}</span>}
                    {e.optimiser && (
                      <span className="path">
                        {e.optimiser.turns} turns · {fmtS(e.optimiser.duration_ms)}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {gated.length > 0 && (
        <>
          <h2>
            Every cycle, every domain — {promoted} of {gated.length} challengers took the title
          </h2>
          <figure>
            <div className="label fig-title">
              fig 2 · every loop cycle by domain: the champion's pass^1 (open) against the challenger's (filled), train above test
            </div>
            <CycleDumbbell hs={hs} open={version ? { domain, version } : null} onPick={(d, v) => nav(optimisePath(d, v.version))} />
            <figcaption>
              {promoted} of {gated.length} cycles promoted
              {mixed ? `; ${mixed} of the ${heldN} held challengers fixed some of the champion's failures and broke others of its passes` : ''}. Each row is one cycle: the open dot is the champion it faced, the filled dot the challenger, green where it was promoted and amber where it was held. The thin dashed pair under it is the same two versions on test, drawn only when both ran it. A row opens its round.
              <span className="path">loop/&lt;domain&gt;/ledger.jsonl · runs/*/run.json → /api/versions</span>
            </figcaption>
          </figure>
        </>
      )}
    </>
  );
}

const STEPS: [string, string, string][] = [
  ['diagnose', '01 · Diagnose', 'every failed train conversation, read and explained'],
  ['propose', '02 · Propose', 'the two files the optimiser may edit, and why each line changed'],
  ['outcome', '03 · Outcome', "what the gate found, task by task"],
];

export function OptimiseRound() {
  const { domain = 'airline', version = '' } = useParams();
  const [lens, setLens] = useLens();
  const step = lens.get('step') ?? 'diagnose';
  // null until a file tab is picked: the round opens on the first file it changed
  const [file, setFile] = useState<string | null>(null);
  const nav = useNavigate();
  const { data: agents } = useGet<{ versions: AgentInfo[]; registry: Registries }>(
    `/api/agents?domain=${encodeURIComponent(domain)}`,
  );
  const { data: ledger } = useGet<Record<string, LedgerEntry[]>>('/api/ledger');
  const cycle = (ledger?.[domain] ?? []).find((e) => e.challenger === version);
  // the round's "before" is the champion it was measured against; failing that, the version before it
  const names = (agents?.versions ?? []).map((v) => v.name);
  const parent = cycle?.champion ?? names[Math.max(0, names.indexOf(version) - 1)] ?? version;
  const { data, error } = useGet<DiffPayload>(
    version ? `/api/agents/diff?domain=${encodeURIComponent(domain)}&a=${parent}&b=${version}` : null,
  );
  // a round opened from below (the cycle figure, a row far down the table) comes into view
  const ref = useRef<HTMLElement>(null);
  const ready = !!data;
  useEffect(() => {
    const top = ref.current?.getBoundingClientRect().top;
    if (top != null && (top < 0 || top > window.innerHeight * 0.75)) ref.current?.scrollIntoView({ block: 'start' });
  }, [domain, version, ready]);
  if (!data) return <Loading error={error} />;

  const ra = best(data.a.runs, 'train');
  const rb = best(data.b.runs, 'train');
  const ta = best(data.a.runs, 'test');
  const tb = best(data.b.runs, 'test');
  const cur =
    data.files.find((f) => f.name === (file ?? data.files.find((x) => x.changed)?.name)) ??
    data.files[0];
  const changes = data.diagnosis?.changes ?? [];
  const added = data.files.reduce((n, f) => n + f.added, 0);
  const removed = data.files.reduce((n, f) => n + f.removed, 0);

  return (
    <section className="round-open" ref={ref}>
      <p className="crumbs">
        <Link to={optimisePath(domain)}>{domainLabel(domain)} rounds</Link> /{' '}
        <span className="mono">
          {parent} → {version}
        </span>
        <button type="button" className="linkish" onClick={() => nav(optimisePath(domain))}>
          {' '}
          close
        </button>
      </p>
      <h2 style={{ marginTop: 0 }}>
        {cycle ? `Cycle ${cycle.cycle}` : 'A version'} — {handMade(cycle) ?? `${parent} → ${version}`}:{' '}
        {cycle?.outcome?.verdict ?? 'not produced by the loop'}
      </h2>

      <div className="kpis">
        <div className="kpi">
          <div className="label">{data.a.name} · train / test</div>
          <div className="n">
            {ra?.summary ? `${ra.summary.passed}/${ra.summary.n_scored}` : '—'} ·{' '}
            {ta?.summary ? `${ta.summary.passed}/${ta.summary.n_scored}` : '—'}
          </div>
          <div className="b">
            {ra ? <Link to={runPath(ra.run_id)}>{shortRun(ra.run_id)}</Link> : 'no scored run'} ·{' '}
            {data.a.fingerprint}
          </div>
        </div>
        <div className="kpi">
          <div className="label">{data.b.name} · train / test</div>
          <div
            className={`n ${rb && ra && (rb.summary?.passed ?? 0) > (ra.summary?.passed ?? 0) ? 'ok' : ''}`}
          >
            {rb?.summary ? `${rb.summary.passed}/${rb.summary.n_scored}` : '—'} ·{' '}
            {tb?.summary ? `${tb.summary.passed}/${tb.summary.n_scored}` : '—'}
          </div>
          <div className="b">
            {rb ? <Link to={runPath(rb.run_id)}>{shortRun(rb.run_id)}</Link> : 'no scored run'} ·{' '}
            {data.b.fingerprint}
          </div>
        </div>
        <div className="kpi">
          <div className="label">lines changed</div>
          <div className="n">
            <span className="v-ok">+{added}</span> <span className="v-warn">−{removed}</span>
          </div>
          <div className="b">
            {data.files
              .filter((f) => f.changed)
              .map((f) => f.name)
              .join(', ') || 'identical'}
          </div>
        </div>
        <div className="kpi">
          <div className="label">gate</div>
          <div
            className={`n ${cycle?.outcome?.verdict === 'promote' ? 'ok' : cycle?.outcome?.verdict === 'hold' ? 'warn' : ''}`}
          >
            {cycle?.outcome?.verdict ?? '—'}
          </div>
          <div className="b">{cycle?.outcome?.reason ?? 'no ledger entry for this version'}</div>
        </div>
      </div>

      <div className="steps3" role="tablist">
        {STEPS.map(([key, title, sub], i) => (
          <button
            key={key}
            type="button"
            role="tab"
            aria-selected={step === key}
            className={`step ${step === key ? 'on' : ''}`}
            onClick={() => setLens({ step: key })}
          >
            <span className="label">{title}</span>
            <span className="small">{sub}</span>
            {i < 2 && (
              <span className="arrow" aria-hidden="true">
                →
              </span>
            )}
          </button>
        ))}
      </div>

      {step === 'diagnose' &&
        (data.diagnosis ? (
          <>
            <dl className="diff-sum">
              <dt>prompt</dt>
              <dd>{data.diagnosis.prompt_diff_summary}</dd>
              <dt>helper</dt>
              <dd>{data.diagnosis.helper_diff_summary}</dd>
            </dl>
            {data.diagnosis.kind && !data.diagnosis.diagnoses.length ? (
              <div className="empty">
                {data.b.name} is a {data.diagnosis.kind} of {data.diagnosis.forked_from ?? parent}
                {data.diagnosis.agent_yaml?.length ? ` (${data.diagnosis.agent_yaml.join('; ')})` : ''}:
                no optimiser read the failures, so nothing was diagnosed.{' '}
                {cycle ? (
                  <>
                    The gate scored it against {cycle.champion} on the same train tasks, as it
                    scores a loop challenger.
                  </>
                ) : (
                  <>
                    No gate has scored it; <code>make challenge</code> scores a fork against the
                    champion with the loop's own gate.
                  </>
                )}
              </div>
            ) : (
              <DiagnosisTable rows={data.diagnosis.diagnoses} cycle={cycle} />
            )}
            {data.diagnosis.risks?.length ? (
              <details>
                <summary>
                  {data.diagnosis.kind
                    ? `risks named for the ${data.diagnosis.kind}`
                    : 'risks the optimiser named'}
                </summary>
                <ul>
                  {data.diagnosis.risks.map((r, i) => (
                    <li key={i} className="small">
                      {r}
                    </li>
                  ))}
                </ul>
              </details>
            ) : null}
          </>
        ) : (
          <div className="empty">
            {data.b.name} has no <code>diagnosis.json</code> — it was not written by the optimiser
            (v0 is tau2's own agent instruction, verbatim).
          </div>
        ))}

      {step === 'propose' && (
        <>
          <div className="chips" role="tablist">
            {data.files.map((f) => (
              <button
                key={f.name}
                type="button"
                role="tab"
                aria-selected={cur?.name === f.name}
                className={`chip ${cur?.name === f.name ? 'acc' : f.changed ? '' : 'no'}`}
                onClick={() => setFile(f.name)}
                style={{ cursor: 'pointer' }}
              >
                {f.name} {f.changed ? `+${f.added} −${f.removed}` : 'unchanged'}
              </button>
            ))}
          </div>
          {cur && (
            <div className="code diff">
              {cur.changed ? (
                hunks(cur.diff).map((h, hi) => (
                  <div key={hi}>
                    {commentaryFor(h, changes, cur.name).map((c, ci) => (
                      <div key={ci} className="note">
                        <span className="label">
                          why · tasks {c.task_ids.map((t) => shortTask(t, 20)).join(', ') || '—'}
                        </span>
                        <b>{c.what}</b> {c.why}
                      </div>
                    ))}
                    <pre>
                      <span className="hunk">
                        {h.header}
                        {'\n'}
                      </span>
                      {h.lines.map((line, i) => (
                        <span
                          key={i}
                          className={
                            line.startsWith('+') ? 'add' : line.startsWith('-') ? 'del' : 'ctx'
                          }
                        >
                          {line}
                          {'\n'}
                        </span>
                      ))}
                    </pre>
                  </div>
                ))
              ) : (
                <pre className="muted">
                  {cur.name} is identical in {data.a.name} and {data.b.name}
                  {cur.name === 'agent.yaml' ? ' — frozen by design' : ''}.
                </pre>
              )}
            </div>
          )}
          {changes.length > 0 && (
            <>
              <h3>Every edit, what it does, and the evidence for it</h3>
              <div className="tw">
                <table>
                  <thead>
                    <tr>
                      <th>file</th>
                      <th>anchor</th>
                      <th>what</th>
                      <th>why</th>
                      <th>tasks</th>
                    </tr>
                  </thead>
                  <tbody>
                    {changes.map((c, i) => (
                      <tr key={i}>
                        <td className="mono">{c.file}</td>
                        <td className="mono small wrap">{c.anchor}</td>
                        <td className="wrap">{c.what}</td>
                        <td className="wrap">{c.why}</td>
                        <td className="mono small">
                          {c.task_ids.map((t) => shortTask(t, 20)).join(', ') || '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          {data.closing_account && (
            <>
              <h3>In the optimiser's own words</h3>
              <div className="code">
                <pre>{data.closing_account}</pre>
              </div>
              <p className="small muted">
                The last message of{' '}
                <code>
                  agents/{domain}/{data.b.name}/optimiser_transcript.json
                </code>
                , verbatim.
              </p>
            </>
          )}
        </>
      )}

      {step === 'outcome' &&
        (cycle ? (
          <>
            <div className="chips">
              {cycle.outcome?.fixed?.length ? (
                <span className="chip ok">
                  fixed {cycle.outcome.fixed.map((t) => shortTask(t, 20)).join(', ')}
                </span>
              ) : null}
              {cycle.outcome?.broken?.length ? (
                <span className="chip warn">
                  broken {cycle.outcome.broken.map((t) => shortTask(t, 20)).join(', ')}
                </span>
              ) : null}
              {cycle.outcome?.still_failed?.length ? (
                <span className="chip no">still failed {cycle.outcome.still_failed.length}</span>
              ) : null}
              {/* two chips, not one: a chip does not wrap, and one long one overflows a phone */}
              {cycle.tokens && (
                <span className="chip">
                  {cycle.tokens.optimiser_in != null
                    ? `tokens opt ${fmtK(cycle.tokens.optimiser_in)}/${fmtK(cycle.tokens.optimiser_out)}`
                    : 'no optimiser tokens'}
                </span>
              )}
              {cycle.tokens && (
                <span className="chip">
                  tokens eval agent {fmtK(cycle.tokens.eval_agent_in)}/
                  {fmtK(cycle.tokens.eval_agent_out)}
                </span>
              )}
              {cycle.outcome?.challenger_run && (
                <Link
                  className="chip"
                  to={runPath(cycle.champion_run ?? '', { vs: cycle.outcome.challenger_run })}
                >
                  the gate, paired by task
                </Link>
              )}
              {cycle.outcome?.test_run && (
                <Link className="chip" to={runPath(cycle.outcome.test_run)}>
                  test run · {shortRun(cycle.outcome.test_run)}
                </Link>
              )}
            </div>
            <dl className="diff-sum">
              <dt>verdict</dt>
              <dd>
                {cycle.outcome?.verdict ?? 'pending'}
                {cycle.outcome?.p_value != null ? ` · p = ${cycle.outcome.p_value}` : ''}
              </dd>
              <dt>reason</dt>
              <dd>{cycle.outcome?.reason ?? '—'}</dd>
              <dt>train</dt>
              <dd>{cycle.outcome?.passes ?? '—'}</dd>
              <dt>test</dt>
              <dd>{cycle.outcome?.test_passes ?? 'not run'}</dd>
              {cycle.optimiser?.error && (
                <>
                  <dt>harness</dt>
                  <dd className="v-warn">{cycle.optimiser.error}</dd>
                </>
              )}
            </dl>
            {cycle.expected_to_fix?.length ? (
              <p className="small">
                <b>expected to fix</b>{' '}
                {cycle.expected_to_fix.map((t) => shortTask(t, 24)).join(', ')} — of which{' '}
                {cycle.expected_to_fix.filter((t) => taskOutcome(cycle, t) === 'fixed').length} did.
              </p>
            ) : null}
          </>
        ) : (
          <div className="empty">
            {version} has no ledger entry: it was registered by hand rather than produced by a loop
            cycle, so no gate ran on it.
          </div>
        ))}
    </section>
  );
}
