import { useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { post, useGet, when } from '../lib/api';
import { CHECKS, runTag } from '../lib/judge';
import { Kpi, Loading } from '../lib/ui';
import { goldReviewPath, reviewPath, trialPath, useLens } from '../lib/url';

/**
 * Review › golden answers (plan s11, J1). The tool judge is scored against an answer key an
 * annotator wrote with gold in view. Structure pins most of it; a person checks the rest here —
 * every case structure could not pin, plus 20 pinned ones drawn at random — and the person's
 * answer stands. Checks are rows in the central Postgres (`tau2_loop.gold_review`), the one thing
 * here a person types after the run; `make judge-gold-freeze` copies them into the committed
 * gold file, which is what every score reads.
 */

type Evidence = { msg: number; quote: string };
type Answer = {
  id: string;
  verdict: 'allow' | 'block';
  check: number | null;
  rule: string | null;
  evidence: Evidence[];
  detectable: boolean | null;
  why: string;
  fix: string | null;
};
type FirstWrong = { msg: number; kind: string; blame: string; why: string };
type Item = {
  id: string;
  key: string;
  run: string;
  task: string;
  trial: number;
  msg: number;
  asks: string[];
  fold: number;
  half: string;
  passed: boolean;
  mode: string;
  checkpoint: { id: string; kind: string; label: string; pinned: boolean } | null;
  answer: Answer | null;
  first_wrong_step: FirstWrong | null;
  is_first_wrong: boolean;
  summary: string | null;
  problems: string[];
};
type Check = {
  id: number;
  item_id: string;
  verdict: 'agree' | 'correct';
  correction: { verdict?: string; check?: number | null; first_wrong_msg?: number | null };
  note: string;
  author: string;
  created_at: string;
};
type GoldSummary = {
  records: number;
  conversations: number;
  machine_checks: { pass: number; pass_first_time: number; fail: number };
  slip_cross_check: { agree?: number; disagree?: number };
  first_wrong_step: Record<string, number>;
  blame: Record<string, number>;
  review_queue: { items: number; structural: number; pinned_sample: number };
  tokens: Record<string, number>;
  calls: number;
  annotator: { model: string; effort: string };
};
type GoldIndex = {
  summary: GoldSummary | null;
  items: Item[];
  current: Record<string, Check>;
  writable: boolean;
  reason: string;
};
type ItemDetail = Item & { messages: Record<string, { role: string; text: string }>; history: Check[] };

/** The two things Review checks: the NL judge's scores, and the tool judge's answer key. */
export function ReviewTabs({ current }: { current: 'conversations' | 'golden' }) {
  return (
    <nav className="chips domainbar" aria-label="what to review">
      <Link to={reviewPath()} className={`chip nav ${current === 'conversations' ? 'on' : ''}`}>
        conversations · the NL judge
      </Link>
      <Link to={goldReviewPath('airline')} className={`chip nav ${current === 'golden' ? 'on' : ''}`}>
        golden answers · the tool judge
      </Link>
    </nav>
  );
}

function verdictLine(a: Answer | null) {
  if (!a) return <span className="muted">no answer</span>;
  return (
    <>
      <span className={`status ${a.verdict === 'block' ? 'err' : 'ok'}`}>{a.verdict}</span>
      {a.check ? ` · check ${a.check}, ${CHECKS[a.check] ?? ''}` : ''}
    </>
  );
}

export function GoldReview() {
  const { domain = 'airline', runId, taskId, trial, msg } = useParams();
  const open = runId && taskId && trial && msg ? `${runId}/${taskId}/${trial}#${msg}` : null;
  const [lens, setLens] = useLens();
  const nav = useNavigate();
  const [nonce, setNonce] = useState(0);
  const { data, error } = useGet<GoldIndex>(`/api/judge/${encodeURIComponent(domain)}/gold`, nonce);
  if (!data) return <Loading error={error} />;
  const s = data.summary;
  const show = lens.get('show') ?? 'todo';
  const checked = data.items.filter((i) => data.current[i.id]);
  const agreed = checked.filter((i) => data.current[i.id].verdict === 'agree').length;
  const rows = data.items.filter((i) =>
    show === 'todo' ? !data.current[i.id] : show === 'checked' ? !!data.current[i.id] : show === 'corrected' ? data.current[i.id]?.verdict === 'correct' : true,
  );

  return (
    <>
      <p className="label">Review · golden answers</p>
      <ReviewTabs current="golden" />
      <h1>
        The tool judge is scored against an answer key a <em>person</em> checks
      </h1>
      <p className="lead">
        An annotator ({s?.annotator.model ?? 'Opus 5.5'} at {s?.annotator.effort ?? 'high'} effort) read every train
        conversation with the task’s gold actions and the grader’s verdict in view, and wrote whether each plan, write
        and transfer should be allowed or blocked, and why. Structure pins most of those verdicts. You check the rest,
        and where you disagree your answer stands.
      </p>
      {!data.writable && (
        <div className="empty">
          <b>Read only.</b> {data.reason}
        </div>
      )}
      {s && (
        <div className="kpis">
          <Kpi n={`${checked.length} / ${data.items.length}`} b="cases checked by a person: every case structure could not pin, plus 20 pinned ones at random" />
          {checked.length > 0 && (
            <Kpi
              n={`${agreed} / ${checked.length}`}
              b="where the person agrees with the annotator; the bar is 90% before the judge is scored on it"
              tone={checked.length && agreed / checked.length < 0.9 ? 'warn' : undefined}
            />
          )}
          <Kpi
            n={`${s.machine_checks.pass} / ${s.records}`}
            b={`answers that pass every machine check: pinned verdicts unchanged, quotes verbatim in the policy and the cited message (${s.machine_checks.pass_first_time} first time)`}
            tone={s.machine_checks.fail ? 'warn' : 'ok'}
          />
          <Kpi
            n={`${s.slip_cross_check.agree ?? 0} / ${(s.slip_cross_check.agree ?? 0) + (s.slip_cross_check.disagree ?? 0)}`}
            b="plans that led to a bad write where the annotator and the hand reading agree: wrong plan, or a slip"
          />
        </div>
      )}
      {s && s.records < s.conversations && (
        <p className="small v-warn">
          {s.records} of {s.conversations} conversations have a golden answer so far; the queue grows as the rest arrive.
        </p>
      )}

      {open && (
        <ItemEditor
          domain={domain}
          id={open}
          writable={data.writable}
          onSaved={() => setNonce((n) => n + 1)}
          onClose={() => nav(goldReviewPath(domain, undefined, { show }))}
        />
      )}

      <div className="filters">
        <label className="pick">
          <span className="label">show</span>
          <select value={show} onChange={(e) => setLens({ show: e.target.value })}>
            <option value="todo">not yet checked</option>
            <option value="checked">checked</option>
            <option value="corrected">corrected</option>
            <option value="all">all</option>
          </select>
        </label>
        <span className="count">
          {rows.length} of {data.items.length} cases
        </span>
      </div>
      <div className="tw">
        <table>
          <caption>
            Each case is one message in one conversation. Pick one to read the annotator’s answer and check it.
          </caption>
          <thead>
            <tr>
              <th>conversation</th>
              <th className="num">message</th>
              <th>what to check</th>
              <th>the annotator says</th>
              <th>person</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((i) => {
              const cur = data.current[i.id];
              return (
                <tr key={i.id} className={`click ${open === i.id ? 'pro' : ''}`} onClick={() => nav(goldReviewPath(domain, i.id, { show }))}>
                  <td className="sub mono small">
                    {runTag(i.run)} · task {i.task}
                    <span className="path">
                      F{i.fold} · {i.half} half · {i.passed ? 'passed' : i.mode.replace(/_/g, ' ')}
                    </span>
                  </td>
                  <td className="num">{i.msg >= 0 ? i.msg : '—'}</td>
                  <td className="wrap small">{i.asks.join(' · ')}</td>
                  <td className="small">
                    {i.checkpoint ? verdictLine(i.answer) : null}
                    {i.is_first_wrong && i.first_wrong_step ? (
                      <span className="path">first wrong step · {i.first_wrong_step.kind} · {i.first_wrong_step.blame.replace(/_/g, ' ')}</span>
                    ) : null}
                  </td>
                  <td>
                    {cur ? (
                      <span className={`status ${cur.verdict === 'agree' ? 'ok' : 'warn'}`} title={`${cur.author || 'anonymous'} · ${when(cur.created_at)}`}>
                        {cur.verdict === 'agree' ? 'agrees' : 'corrected'}
                      </span>
                    ) : (
                      <span className="status no">not checked</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        Checks are append-only rows in the central Postgres; <code>make judge-gold-freeze DOMAIN={domain}</code> copies the
        current ones into <code>data/judge/{domain}_gold.jsonl</code>, which every score reads.
      </p>
    </>
  );
}

function ItemEditor({ domain, id, writable, onSaved, onClose }: { domain: string; id: string; writable: boolean; onSaved: () => void; onClose: () => void }) {
  const [nonce, setNonce] = useState(0);
  const { data, error } = useGet<ItemDetail>(`/api/judge/${encodeURIComponent(domain)}/gold/item?id=${encodeURIComponent(id)}`, nonce);
  const [correcting, setCorrecting] = useState(false);
  const [verdict, setVerdict] = useState<'allow' | 'block' | ''>('');
  const [check, setCheck] = useState<string>('');
  const [firstWrong, setFirstWrong] = useState<string>('');
  const [note, setNote] = useState('');
  const [author, setAuthor] = useState('');
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  if (!data) return <Loading error={error} />;
  const a = data.answer;
  const current = data.history[0];

  const save = async (kind: 'agree' | 'correct') => {
    setSaving(true);
    setErr(null);
    try {
      const correction: Record<string, unknown> = {};
      if (kind === 'correct') {
        if (verdict) correction.verdict = verdict;
        if (verdict === 'block' && check) correction.check = Number(check);
        if (data.is_first_wrong && firstWrong !== '') correction.first_wrong_msg = firstWrong === 'none' ? null : Number(firstWrong);
      }
      await post(`/api/review/golden/${encodeURIComponent(domain)}`, { item_id: id, verdict: kind, correction, note, author });
      setCorrecting(false);
      setNote('');
      setNonce((n) => n + 1);
      onSaved();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="card hi round-open">
      <p className="crumbs">
        <Link to={trialPath(data.run, `${data.task}/t${data.trial}`)}>
          {runTag(data.run)} · task {data.task} · t{data.trial}
        </Link>{' '}
        · message {data.msg} · F{data.fold}, {data.half} half
        <button type="button" className="linkish" onClick={onClose}>
          {' '}
          close
        </button>
      </p>
      <h3 style={{ marginTop: 0 }}>Check: {data.asks.join(' · ')}</h3>
      {Object.entries(data.messages).map(([n, m]) => (
        <div key={n} className={`ev ${m.role === 'assistant' ? 'text' : m.role === 'user' ? 'user' : 'tool_result'}`}>
          <div className="label">
            message {n}
            {Number(n) === data.msg ? ' · the case' : ' · cited as evidence'}
          </div>
          <pre>{m.text}</pre>
        </div>
      ))}
      {data.checkpoint && a && (
        <div className="gold-answer">
          <p className="small">
            <b>The annotator says:</b> <span className="chip">{data.checkpoint.kind}</span> {verdictLine(a)} · {a.why}
          </p>
          {a.verdict === 'block' && (
            <>
              {a.rule && (
                <p className="small">
                  <b>Rule:</b> {a.rule === 'transcript' ? 'the transcript contradicts it' : `“${a.rule}”`}
                </p>
              )}
              {a.evidence.length > 0 && (
                <p className="small">
                  <b>Evidence:</b> {a.evidence.map((e) => `message ${e.msg}, “${e.quote}”`).join('; ')}
                </p>
              )}
              {a.fix && (
                <p className="small">
                  <b>Fix it would send:</b> {a.fix}
                </p>
              )}
              <p className="small muted">
                {a.detectable ? 'Detectable from the policy and transcript alone.' : 'Only gold reveals it: a judge without gold cannot be expected to catch it.'} J0’s
                label: {data.checkpoint.label}
                {data.checkpoint.pinned ? ' (pinned)' : ''}.
              </p>
            </>
          )}
        </div>
      )}
      {data.is_first_wrong && data.first_wrong_step && (
        <p className="small">
          <b>First wrong step:</b> message {data.first_wrong_step.msg}, a {data.first_wrong_step.kind}, blamed on the{' '}
          {data.first_wrong_step.blame.replace(/_/g, ' ')} · {data.first_wrong_step.why}
        </p>
      )}
      {data.summary && <p className="small muted">{data.summary}</p>}
      {data.problems.length > 0 && <p className="small v-warn">Machine checks still failing: {data.problems.join('; ')}</p>}

      <fieldset disabled={!writable || saving}>
        <div className="row">
          <button type="button" className="btn" onClick={() => save('agree')}>
            Agree
          </button>{' '}
          <button type="button" className="linkish" aria-expanded={correcting} onClick={() => setCorrecting((v) => !v)}>
            Correct it
          </button>
          <input type="text" placeholder="who (optional)" value={author} onChange={(e) => setAuthor(e.target.value)} style={{ maxWidth: '14rem' }} />
          {err && <span className="v-warn small">{err}</span>}
        </div>
        {correcting && (
          <div className="correct-form">
            {data.checkpoint && (
              <div className="row">
                <label className="pick">
                  <span className="label">verdict</span>
                  <select value={verdict} onChange={(e) => setVerdict(e.target.value as 'allow' | 'block' | '')}>
                    <option value="">unchanged</option>
                    <option value="allow">allow</option>
                    <option value="block">block</option>
                  </select>
                </label>
                {verdict === 'block' && (
                  <label className="pick">
                    <span className="label">check</span>
                    <select value={check} onChange={(e) => setCheck(e.target.value)}>
                      <option value="">—</option>
                      {Object.entries(CHECKS).map(([k, v]) => (
                        <option key={k} value={k}>
                          {k} · {v}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>
            )}
            {data.is_first_wrong && (
              <div className="row">
                <label className="pick">
                  <span className="label">first wrong step at message</span>
                  <input type="text" inputMode="numeric" placeholder="a number, or none" value={firstWrong} onChange={(e) => setFirstWrong(e.target.value.trim())} style={{ maxWidth: '10rem' }} />
                </label>
              </div>
            )}
            <div className="row">
              <textarea placeholder="why — the sentence a future reader needs" value={note} onChange={(e) => setNote(e.target.value)} />
            </div>
            <div className="row">
              <button type="button" className="linkish" disabled={!note && !verdict && firstWrong === ''} onClick={() => save('correct')}>
                record the correction
              </button>
            </div>
          </div>
        )}
      </fieldset>
      {current && (
        <p className="small muted">
          Current: {current.verdict === 'agree' ? 'agrees' : 'corrected'}
          {current.verdict === 'correct' && Object.keys(current.correction).length ? ` (${JSON.stringify(current.correction)})` : ''}
          {current.note ? ` · ${current.note}` : ''} · {current.author || 'anonymous'} · {when(current.created_at)}
          {data.history.length > 1 ? ` · ${data.history.length - 1} earlier` : ''}
        </p>
      )}
    </section>
  );
}
