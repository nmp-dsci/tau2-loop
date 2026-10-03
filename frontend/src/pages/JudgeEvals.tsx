import { useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { type Event, byTask, domainLabel, post, useGet } from '../lib/api';
import { DbDiff } from '../lib/dbdiff';
import { type Check, GoldCheck } from '../lib/goldcheck';
import { GoldLine, type JudgeView, MODE, annotated, goldFirstWrong, golden, humanAt, runTag } from '../lib/judge';
import { type Conv, WHOLE, firstBlock, itemId, stateOf, toTick, toUntick, verdictAt } from '../lib/judgeevals';
import { useExp, useTask } from '../lib/scope';
import { Kpi, Loading, Points } from '../lib/ui';
import { judgeConvPath, judgeEvalsPath, judgeLoopPath, trialPath, useLens } from '../lib/url';
import { EventList } from './Trace';

/**
 * Evals, with the scope bar on the LLM judge (s11): every conversation it is scored against, laid
 * out like the answering agent's tasks. The judge is called only when the agent issues a write or a
 * transfer, before it runs. A passed conversation is confirmed by the grader (the passed rule): tau2
 * matched its database to gold's, so every call in it was right. A person ticks each failed one,
 * agreeing with its golden answers, above all the first call the judge must block; a failure with no
 * write or transfer is confirmed as a whole, since the judge is never called in it. Checks go to the
 * central Postgres; `make judge-gold-freeze` copies them into the gold file every score reads. Read
 * from `data/judge/<domain>.json` and `data/judge/<domain>_gold.jsonl`.
 */

type Synth = { id: string; key: string; msg: number; task: string; kind: string; what: string; change: { from: string; to: string } };
type Pending = { runs: { run: string; agent: string; traces: number }[]; no_gold: { failed: number; passed: number } };
type Evals = { domain: string; conversations: Conv[] | null; synthetic?: Synth[]; pending?: Pending };
type GoldIndex = { items: { id: string }[]; current: Record<string, Check>; writable: boolean; reason: string };
type TracePayload = { events: Event[]; judge?: JudgeView | null; reward_info?: { db_check?: { db_match: boolean } | null } | null };
type Record_ = (c: Conv, verdict: 'agree' | 'withdraw', msgs: number[]) => Promise<void>;

const enc = encodeURIComponent;
const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? '' : 's'}`;

export function JudgeEvals() {
  const { domain = 'airline', runId, taskId, trial } = useParams();
  const openKey = runId && taskId && trial ? `${runId}/${taskId}/${trial}` : null;
  const nav = useNavigate();
  const [lens, setLens] = useLens();
  const exp = useExp();
  const task = useTask();
  const { data, error } = useGet<Evals>(`/api/judge/${enc(domain)}/evals`);
  const { data: gold } = useGet<GoldIndex>(`/api/judge/${enc(domain)}/gold`);
  // checks made on this page, by case id, laid over the server's so the table follows at once;
  // null is a check taken back
  const [mine, setMine] = useState<Record<string, Check | null>>({});
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [err, setErr] = useState<string | null>(null);
  if (!data) return <Loading error={error} />;
  if (!data.conversations) {
    return (
      <>
        <p className="label">Evals · the LLM judge · {domainLabel(domain)}</p>
        <h1>No LLM judge has an eval set on {domainLabel(domain)} yet</h1>
        <div className="empty">
          <code>make judge-labels DOMAIN={domain}</code> then <code>make judge-gold DOMAIN={domain}</code> build it; s11’s J8
          brings it here after airline.
        </div>
      </>
    );
  }

  const current: Record<string, Check> = { ...(gold?.current ?? {}) };
  for (const [k, v] of Object.entries(mine)) {
    if (v) current[k] = v;
    else delete current[k];
  }
  const saved = (x: Check) => setMine((m) => ({ ...m, [x.item_id]: x.verdict === 'withdraw' ? null : x }));
  const record: Record_ = async (c, verdict, msgs) => {
    setBusy((b) => ({ ...b, [c.key]: true }));
    setErr(null);
    try {
      for (const m of msgs) {
        saved(await post<Check>(`/api/review/golden/${enc(domain)}`, { item_id: itemId(c, m), verdict, correction: {}, note: '', author: '' }));
      }
    } catch (e) {
      setErr(`task ${c.task}, ${c.version}: ${(e as Error).message}`);
    } finally {
      setBusy((b) => ({ ...b, [c.key]: false }));
    }
  };
  const tick = (c: Conv, on: boolean) => record(c, on ? 'agree' : 'withdraw', on ? toTick(c, current) : toUntick(c, current));

  const writable = !!gold?.writable;
  const show = lens.get('show') ?? '';
  const q = (lens.get('q') ?? '').trim();
  const all = data.conversations
    .filter((c) => !exp || c.version === exp)
    .filter((c) => !task || c.task === task)
    .sort((a, b) => byTask(a.task, b.task) || byTask(a.version, b.version) || a.key.localeCompare(b.key));
  const st = (c: Conv) => stateOf(c, current).state;
  const corrected = (c: Conv) => c.review.filter((m) => current[itemId(c, m)]?.verdict === 'correct').length;
  const rows = all
    .filter((c) => {
      if (show === 'todo') return !c.passed && c.has_gold && st(c) !== 'confirmed';
      if (show === 'confirmed') return st(c) === 'confirmed';
      if (show === 'passed') return c.passed;
      if (show === 'failed') return !c.passed;
      if (show === 'block') return !c.passed && !!firstBlock(c, current);
      if (show === 'none') return !c.passed && c.has_gold && !firstBlock(c, current);
      if (show === 'corrected') return corrected(c) > 0;
      return true;
    })
    .filter((c) => !q || c.task === q || c.version === q || c.key.includes(q));
  const open = openKey ? all.find((c) => c.key === openKey) : undefined;
  const at = open ? rows.indexOf(open) : -1;
  const next = [...rows.slice(at + 1), ...rows.slice(0, Math.max(at, 0))].find(
    (c) => c !== open && !c.passed && c.has_gold && st(c) !== 'confirmed',
  );
  const passed = all.filter((c) => c.passed);
  const failed = all.filter((c) => !c.passed);
  const confirmed = failed.filter((c) => st(c) === 'confirmed');
  const catchable = failed.filter((c) => firstBlock(c, current));
  const callless = failed.filter((c) => c.has_gold && !c.calls.length);
  const pending = data.pending;
  const waiting = pending ? pending.runs.length + pending.no_gold.failed + pending.no_gold.passed : 0;
  const synth = (data.synthetic ?? [])
    .slice()
    .sort((a, b) => byTask(a.task, b.task) || a.key.localeCompare(b.key) || a.msg - b.msg);

  return (
    <>
      <p className="label">Evals · the LLM judge · {domainLabel(domain)}</p>
      <h1>
        {passed.length} of {all.length} conversations passed, so the grader confirms them; a <em>person</em> confirms the
        other {failed.length}
      </h1>
      <Points
        lead
        items={[
          <>
            <b>The judge is called only when the agent issues a write or a transfer</b>, before it runs. A failure it can catch
            has a call the golden answer blocks; the first such call is where it must stop.
            {exp || task ? ` Filtered by the scope bar to ${[exp && `${exp}’s conversations`, task && `task ${task}`].filter(Boolean).join(' on ')}.` : ''}
          </>,
          <>
            <b>A pass needs no one</b>: tau2 matched its database to gold’s, so every write and transfer in it was right.
          </>,
          <>
            <b>Tick a failed conversation</b> to agree with its golden answers, or open it to correct one; once frozen with{' '}
            <code>make judge-gold-freeze DOMAIN={domain}</code>, your answers stand in every score. The judge’s score is under{' '}
            <Link to={judgeLoopPath(domain)}>Optimise</Link>.
          </>,
        ]}
      />
      <div className="kpis">
        <Kpi n={`${confirmed.length} / ${failed.length}`} b="failed conversations you have confirmed" tone={confirmed.length === failed.length ? 'ok' : 'warn'} />
        <Kpi n={`${catchable.length} / ${failed.length}`} b="failed conversations with a call the judge must block" />
        <Kpi n={`${passed.length} / ${passed.length}`} b="passes confirmed by the grader: every call allowed" tone="ok" />
      </div>
      <p className="small muted">
        {failed.length - catchable.length} of {failed.length} failures have no call to block: {callless.length} issue no write or
        transfer, so the judge is never called, and in the rest every call was right.
      </p>
      {pending && waiting > 0 ? (
        <p className="small v-warn">
          {pending.runs.length > 0 && (
            <>
              {plural(pending.runs.length, 'newer run')} of an optimised agent {pending.runs.length === 1 ? 'is' : 'are'} not in the
              set yet ({pending.runs.map((r) => `${runTag(r.run)}, ${r.traces} train traces`).join('; ')}):{' '}
              <code>make judge-labels DOMAIN={domain}</code> adds {pending.runs.length === 1 ? 'its' : 'their'} conversations, no
              model.{' '}
            </>
          )}
          {pending.no_gold.failed + pending.no_gold.passed > 0 && (
            <>
              {plural(pending.no_gold.failed + pending.no_gold.passed, 'conversation')} {pending.no_gold.failed + pending.no_gold.passed === 1 ? 'waits' : 'wait'} for a golden answer:{' '}
            </>
          )}
          <code>make judge-gold DOMAIN={domain}</code> writes the failures’ with Opus 5.5; a pass is written by the passed rule,
          no model.
        </p>
      ) : (
        <p className="small muted">Every finished train run of an optimised agent is in the set; a new one shows here with the commands that bring it in.</p>
      )}
      {gold && !writable && <p className="small v-warn">Read only: {gold.reason}</p>}
      {err && <p className="small v-warn">{err}</p>}

      {open && (
        <GoldConversation
          key={open.key}
          domain={domain}
          c={open}
          current={current}
          queued={new Set((gold?.items ?? []).map((i) => i.id))}
          writable={writable}
          reason={gold?.reason ?? ''}
          busy={!!busy[open.key]}
          next={next ? judgeConvPath(domain, next.key, { exp, show, q }) : null}
          onSaved={saved}
          onRecord={record}
          onClose={() => nav(judgeEvalsPath(domain, { exp, show, q }))}
        />
      )}

      <div className="filters">
        <label className="pick">
          <span className="label">show</span>
          <select value={show} onChange={(e) => setLens({ show: e.target.value })}>
            <option value="">all</option>
            <option value="todo">failed, not yet confirmed</option>
            <option value="confirmed">confirmed by you</option>
            <option value="failed">failed</option>
            <option value="block">failed, with a call to block</option>
            <option value="none">failed, no call to block</option>
            <option value="passed">passed</option>
            <option value="corrected">with a correction</option>
          </select>
        </label>
        <input type="search" placeholder="task id, version or run" value={q} onChange={(e) => setLens({ q: e.target.value })} />
        <span className="count">
          {rows.length} of {all.length} conversations
        </span>
      </div>
      <div className="tw fit">
        <table>
          <caption>Tick a failed conversation to agree with its golden answers; a row opens the conversation here.</caption>
          <thead>
            <tr>
              <th>task</th>
              <th>agent</th>
              <th>outcome</th>
              <th className="num">writes · transfers</th>
              <th>first call to block</th>
              <th>you</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => (
              <tr key={c.key} className={`click ${c.key === openKey ? 'pro' : ''}`} onClick={() => nav(judgeConvPath(domain, c.key, { exp, show, q }))}>
                <td className="sub mono">
                  task {c.task}
                  {c.suspect ? <span className="path">s08 suspect</span> : null}
                </td>
                <td className="small">
                  {c.version}
                  <span className="path nw">
                    {runTag(c.run).split(' · ')[0]} · t{c.trial}
                  </span>
                </td>
                <td>
                  <span className={`status ${c.passed ? 'ok' : 'err'}`}>{MODE[c.mode] ?? c.mode}</span>
                </td>
                <td className="num">
                  {c.writes} · {c.checkpoints - c.writes}
                </td>
                <td className="wrap small">
                  <FirstBlock c={c} current={current} />
                </td>
                <td className="small">
                  <Tick c={c} current={current} writable={writable} busy={!!busy[c.key]} onTick={(on) => tick(c, on)} />
                  {corrected(c) ? <span className="path">{corrected(c)} corrected</span> : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {synth.length > 0 && (
        <details>
          <summary>{synth.length} synthetic positives: right plans with one detail changed, each a golden block</summary>
          <div className="tw">
            <table>
              <thead>
                <tr>
                  <th>id</th>
                  <th>source</th>
                  <th>what changed</th>
                  <th>from → to</th>
                </tr>
              </thead>
              <tbody>
                {synth.map((s) => {
                  const [run, task, tn] = s.key.split('/');
                  return (
                    <tr key={s.id}>
                      <td className="sub mono">{s.id}</td>
                      <td className="mono small">
                        <Link to={trialPath(run, `${task}/${tn}`)}>
                          {runTag(run)} · task {task}
                        </Link>
                        <span className="path">message {s.msg}</span>
                      </td>
                      <td className="small">{s.what}</td>
                      <td className="wrap small mono">
                        {s.change.from || '—'} → {s.change.to || '(dropped)'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="small muted">
            Reported apart from the real conversations. <span className="path">data/judge/{domain}.json (synthetic)</span>
          </p>
        </details>
      )}
    </>
  );
}

/** The first call the golden answer blocks: where the judge must stop this failure, or why there is none. */
function FirstBlock({ c, current }: { c: Conv; current: Record<string, Check> }) {
  if (c.passed) return <span className="dim">none: it passed</span>;
  if (!c.has_gold) return <span className="dim">no golden answer yet</span>;
  const b = firstBlock(c, current);
  if (!c.calls.length) {
    return (
      <>
        none: no write or transfer<span className="path">the judge is never called</span>
      </>
    );
  }
  if (!b) {
    return (
      <>
        none: every call was right<span className="path">the judge cannot catch this failure</span>
      </>
    );
  }
  const blocks = c.calls.filter((x) => verdictAt(c, x, current) === 'block').length;
  const fw = c.first_wrong;
  const earlier = fw && fw.msg < b.msg && !c.calls.some((x) => x.msg === fw.msg);
  return (
    <>
      message {b.msg} · <span className="mono">{b.names.join(', ')}</span>
      <span className="path">
        {blocks} of {plural(c.calls.length, 'call')} blocked
        {earlier ? ` · it first went wrong at message ${fw.msg}, a ${fw.kind}` : ''}
      </span>
    </>
  );
}

/** The person's tick on a row: a pass is ticked by the grader; a failure agrees with every golden
 *  answer in it, and unticking takes the agrees back (a correction stays). */
function Tick({
  c,
  current,
  writable,
  busy,
  onTick,
}: {
  c: Conv;
  current: Record<string, Check>;
  writable: boolean;
  busy: boolean;
  onTick: (on: boolean) => void;
}) {
  const s = stateOf(c, current);
  if (s.state === 'nogold') return <span className="dim">—</span>;
  const what = `task ${c.task}, ${c.version}`;
  if (s.state === 'passed') {
    return (
      <label className="tick">
        <input type="checkbox" checked disabled readOnly aria-label={`${what}: confirmed by the grader`} />
        <span className="muted">passed</span>
      </label>
    );
  }
  // the tick is not the row: clicking it never opens the conversation
  return (
    <label className="tick" onClick={(e) => e.stopPropagation()}>
      <input
        type="checkbox"
        checked={s.state === 'confirmed'}
        ref={(el) => {
          if (el) el.indeterminate = s.state === 'partial';
        }}
        disabled={!writable || busy}
        onChange={(e) => onTick(e.target.checked)}
        aria-label={`${what}: confirm its golden answers`}
      />
      <span>{s.state === 'confirmed' ? 'confirmed' : s.state === 'partial' ? `${s.checked} of ${s.of}` : 'to check'}</span>
    </label>
  );
}

/** One conversation, open for review: the whole conversation as Runs shows it, the judge's bar
 *  after each write and transfer with its golden answer, and the person's confirmation of it. */
function GoldConversation({
  domain,
  c,
  current,
  queued,
  writable,
  reason,
  busy,
  next,
  onSaved,
  onRecord,
  onClose,
}: {
  domain: string;
  c: Conv;
  current: Record<string, Check>;
  queued: Set<string>;
  writable: boolean;
  reason: string;
  busy: boolean;
  next: string | null;
  onSaved: (x: Check) => void;
  onRecord: Record_;
  onClose: () => void;
}) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    ref.current?.scrollIntoView?.({ block: 'start', behavior: 'smooth' });
  }, []);
  const { data, error } = useGet<TracePayload>(`/api/runs/${enc(c.run)}/${enc(c.task)}/t${c.trial}`);
  const [again, setAgain] = useState<Record<number, boolean>>({});
  if (!data?.judge) {
    return (
      <section ref={ref} className="card hi task-open">
        <Loading error={error} />
      </section>
    );
  }
  const here = Object.fromEntries(
    Object.values(current)
      .filter((x) => x.conv_key === c.key)
      .map((x) => [String(x.msg), x]),
  );
  const j: JudgeView = { ...data.judge, checks: here };
  const a = j.gold?.answer;
  const fw = goldFirstWrong(j);
  const annotFw = a?.first_wrong_step?.msg ?? null;
  const s = stateOf(c, current);
  const block = firstBlock(c, current);
  const unticked = toTick(c, current);
  const agrees = toUntick(c, current);
  const whole = c.review.length === 1 && c.review[0] === WHOLE;
  const items = c.calls.map((call) => ({ msg: call.msg, cps: j.labels.checkpoints.filter((x) => x.msg === call.msg && x.judged) }));
  const id = (msg: number) => itemId(c, msg);

  // the judge's bar after each write or transfer: its golden answer, and the person's check of it
  const marks = Object.fromEntries(
    items.map(({ msg, cps }) => {
      const answers = cps.map((cp) => ({ cp, g: golden(j, cp), was: annotated(j, cp) }));
      const h = humanAt(j, msg);
      const verdictNow = answers.some((x) => x.g?.verdict === 'block') ? 'block' : answers.length ? 'allow' : null;
      return [
        msg,
        <div key={`review-${msg}`} className={`ev judge review-bar ${h || c.passed ? 'done' : ''}`}>
          <div className="label">
            LLM judge · reviews this {cps.map((cp) => cp.kind).join(' and ')} before it runs
            <span className="muted">
              {' '}
              · on message {msg} · J0 {cps.map((cp) => cp.label).join(', ')}
            </span>
            {block?.msg === msg ? <span className="chip">the first call to block</span> : queued.has(id(msg)) ? <span className="chip">queued for review</span> : null}
          </div>
          {answers.map(({ cp, g, was }) =>
            g ? (
              <div key={cp.call_id ?? cp.kind}>
                <GoldLine a={g} was={was ?? g} human={h} first={fw?.msg === msg} edit={null} />
                {g.verdict === 'block' && (g.evidence.length > 0 || g.fix) && (
                  <p className="small muted gold-evidence">
                    {g.evidence.length > 0 ? `evidence: ${g.evidence.map((e) => `message ${e.msg}, “${e.quote}”`).join('; ')}` : ''}
                    {g.fix ? `${g.evidence.length ? ' · ' : ''}fix it would send: ${g.fix}` : ''}
                  </p>
                )}
              </div>
            ) : null,
          )}
          {c.passed ? null : h && !again[msg] ? (
            <button type="button" className="linkish gold-edit" onClick={() => setAgain((x) => ({ ...x, [msg]: true }))}>
              check it again
            </button>
          ) : (
            <GoldCheck
              domain={domain}
              itemId={id(msg)}
              verdictNow={verdictNow}
              firstWrongNow={annotFw === msg || fw?.msg === msg ? (fw?.msg ?? annotFw) : null}
              writable={writable}
              reason={reason}
              quiet
              onSaved={(x) => {
                setAgain((y) => ({ ...y, [msg]: false }));
                onSaved(x);
              }}
            />
          )}
        </div>,
      ];
    }),
  );

  const question = c.passed ? (
    <>
      <b>Confirmed by the grader.</b> tau2 matched the database to gold’s, so the golden answer allows{' '}
      {c.calls.length ? `all ${plural(c.calls.length, 'call')} here` : 'everything; it issued no write or transfer'}.
    </>
  ) : !c.has_gold ? (
    <>
      <b>No golden answer yet:</b> <code>make judge-gold DOMAIN={domain}</code> writes it.
    </>
  ) : block ? (
    <>
      <b>
        The first call to block: message {block.msg}, {block.names.join(', ')}
      </b>{' '}
      ({block.kinds.join(' and ')}). Could the judge stop it there, before the database goes wrong? Agree, or correct the
      golden answer at that call.
    </>
  ) : c.calls.length ? (
    <>
      <b>No call to block:</b> the golden answer allows all {plural(c.calls.length, 'call')}, so the judge cannot catch this
      failure. Agree, or correct the call it should have blocked.
    </>
  ) : (
    <>
      <b>No call to block:</b> the agent issued no write or transfer, so the judge is never called and cannot catch this failure.
    </>
  );

  const actions = (
    <div className="row">
      {c.passed ? (
        <span className="status ok">nothing to check: the grader confirmed it</span>
      ) : !c.has_gold ? null : s.state === 'confirmed' ? (
        <>
          <span className="status ok">confirmed</span>
          {agrees.length > 0 && (
            <button type="button" className="linkish" disabled={!writable || busy} onClick={() => onRecord(c, 'withdraw', agrees)}>
              take it back
            </button>
          )}
        </>
      ) : (
        <button type="button" className="btn" disabled={!writable || busy} onClick={() => onRecord(c, 'agree', unticked)}>
          {whole ? 'Confirm: nothing for the judge to block' : `Confirm: agree with ${unticked.length === c.review.length ? 'all' : 'the'} ${plural(unticked.length, 'unchecked call')}`}
        </button>
      )}
      {next && (
        <Link to={next} className="linkish">
          next to confirm →
        </Link>
      )}
      {!writable && reason && !c.passed && <span className="small v-warn">{reason}</span>}
    </div>
  );

  return (
    <section ref={ref} className="card hi task-open gold-conv">
      <p className="crumbs">
        task {c.task} · {c.version} · {runTag(c.run).split(' · ')[0]} · t{c.trial} ·{' '}
        <Link to={trialPath(c.run, `${c.task}/t${c.trial}`)}>open in Runs</Link>
        <button type="button" className="linkish" onClick={onClose}>
          {' '}
          close
        </button>
      </p>
      <h3 style={{ marginTop: 0 }}>
        Task {c.task}, {c.version}: {MODE[c.mode] ?? c.mode} —{' '}
        {c.passed
          ? 'confirmed by the grader'
          : !c.has_gold
            ? 'no golden answer yet'
            : s.state === 'confirmed'
              ? 'confirmed'
              : whole
                ? 'to confirm'
                : `${s.checked} of ${plural(s.of, 'call')} checked`}
      </h3>
      <p className="small">{question}</p>
      {a?.summary && j.gold?.annotator.model !== 'rule' && <p className="small muted">{a.summary}</p>}
      {fw && block && fw.msg < block.msg && !c.calls.some((x) => x.msg === fw.msg) && (
        <p className="small muted">
          It first went wrong at message {fw.msg}, a {fw.kind}, which the judge never sees: it is not a write or a transfer.
        </p>
      )}
      {data.reward_info?.db_check && !data.reward_info.db_check.db_match && (
        <DbDiff url={`/api/runs/${enc(c.run)}/${enc(c.task)}/t${c.trial}/db`} />
      )}
      {actions}
      <div className="label fig-title gold-conv-title">
        the conversation, in order{c.calls.length ? '; the judge’s bar follows each write and transfer' : ''}
      </div>
      <EventList events={data.events} marks={marks} />
      {data.events.length > 12 && actions}
    </section>
  );
}
