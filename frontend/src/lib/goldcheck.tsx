import { useState } from 'react';
import { post } from './api';

/**
 * A person's check of one golden answer (s11 J1): Agree, or Correct it with a verdict, a check, a
 * first-wrong message and the why. Review › golden answers and the LLM judge's bar on a conversation
 * share it, so a golden answer is corrected wherever it is read. A check is a row in the central
 * Postgres; `make judge-gold-freeze` copies it into the gold file.
 */

export type Check = {
  id: number;
  item_id: string;
  conv_key: string;
  msg: number;
  verdict: 'agree' | 'correct' | 'withdraw';
  correction: { verdict?: string; check?: number | null; first_wrong_msg?: number | null };
  note: string;
  author: string;
  created_at: string;
};

/** The plan judge's five checks (s11 §5), by number. */
export const CHECKS: Record<number, string> = {
  1: 'identity and ownership',
  2: 'arguments trace to the transcript',
  3: 'the policy allows it',
  4: 'the user confirmed exactly this',
  5: 'the user asked for it',
};

export function GoldCheck({
  domain,
  itemId,
  verdictNow,
  firstWrongNow,
  writable,
  reason = '',
  startCorrecting = false,
  quiet = false,
  onSaved,
}: {
  domain: string;
  itemId: string;
  /** The golden verdict at this message, or null when the case is only the first wrong step. */
  verdictNow: 'allow' | 'block' | null;
  /** The golden first wrong step's message when it is this one, else null. */
  firstWrongNow: number | null;
  writable: boolean;
  reason?: string;
  startCorrecting?: boolean;
  /** Agree as a link and no name box, for a page that lists many answers under one solid button. */
  quiet?: boolean;
  onSaved: (c: Check) => void;
}) {
  const [correcting, setCorrecting] = useState(startCorrecting);
  const [verdict, setVerdict] = useState<'allow' | 'block' | ''>('');
  const [check, setCheck] = useState('');
  const [firstWrong, setFirstWrong] = useState('');
  const [note, setNote] = useState('');
  const [author, setAuthor] = useState('');
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const isFirst = firstWrongNow != null;
  // the golden first wrong step cannot be right and still be where it went wrong
  const needsFirst = isFirst && verdict === 'allow' && firstWrong === '';

  const save = async (kind: 'agree' | 'correct') => {
    setSaving(true);
    setErr(null);
    try {
      const correction: Record<string, unknown> = {};
      if (kind === 'correct') {
        if (verdict) correction.verdict = verdict;
        if (verdict === 'block' && check) correction.check = Number(check);
        if (isFirst && firstWrong !== '') correction.first_wrong_msg = firstWrong === 'none' ? null : Number(firstWrong);
      }
      const row = await post<Check>(`/api/review/golden/${encodeURIComponent(domain)}`, { item_id: itemId, verdict: kind, correction, note, author });
      setCorrecting(false);
      setNote('');
      onSaved(row);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <fieldset className="gold-check" disabled={!writable || saving}>
      <div className="row">
        <button type="button" className={correcting || quiet ? 'linkish' : 'btn'} onClick={() => save('agree')}>
          Agree
        </button>{' '}
        <button type="button" className="linkish" aria-expanded={correcting} onClick={() => setCorrecting((v) => !v)}>
          Correct it
        </button>
        {!quiet && <input type="text" placeholder="who (optional)" value={author} onChange={(e) => setAuthor(e.target.value)} style={{ maxWidth: '14rem' }} />}
        {!writable && reason && <span className="v-warn small">{reason}</span>}
        {err && <span className="v-warn small">{err}</span>}
      </div>
      {correcting && (
        <div className="correct-form">
          {verdictNow && (
            <div className="row">
              <label className="pick">
                <span className="label">verdict</span>
                <select value={verdict} onChange={(e) => setVerdict(e.target.value as 'allow' | 'block' | '')}>
                  <option value="">unchanged ({verdictNow})</option>
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
          {isFirst && (
            <div className="row">
              <label className="pick">
                <span className="label">first wrong step at message</span>
                <input
                  type="text"
                  inputMode="numeric"
                  placeholder={`now ${firstWrongNow}; a number, or none`}
                  value={firstWrong}
                  onChange={(e) => setFirstWrong(e.target.value.trim())}
                  style={{ maxWidth: '14rem' }}
                />
              </label>
              {needsFirst && <span className="small v-warn">This is the golden first wrong step: say where it first went wrong instead, or none.</span>}
            </div>
          )}
          <div className="row">
            <textarea placeholder="why — the sentence a future reader needs" value={note} onChange={(e) => setNote(e.target.value)} />
          </div>
          <div className="row">
            <button type="button" className="btn" disabled={needsFirst || (!note && !verdict && firstWrong === '')} onClick={() => save('correct')}>
              Record the correction
            </button>
          </div>
        </div>
      )}
    </fieldset>
  );
}
