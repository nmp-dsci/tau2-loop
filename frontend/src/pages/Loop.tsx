import { Link, useSearchParams } from 'react-router-dom';
import { DOMAINS, type LedgerEntry, domainLabel, fmtK, fmtS, shortRun, shortTask, useGet } from '../lib/api';

export function Loop() {
  const [sp, setSp] = useSearchParams();
  const domain = sp.get('domain') ?? 'airline';
  const { data: ledger } = useGet<Record<string, LedgerEntry[]>>('/api/ledger');
  const entries = ledger?.[domain] ?? [];
  const counts = Object.fromEntries(DOMAINS.map((d) => [d, (ledger?.[d] ?? []).length]));
  return (
    <>
      <p className="label">The ledger</p>
      <h1>
        Every diagnosis is kept per domain, so the loop does not <em>repeat</em> a failed fix
      </h1>
      <p className="lead">
        This is <code>loop/{domain}/ledger.jsonl</code>, rendered. The optimiser writes the diagnoses; the harness writes the outcome after the
        gate. The next cycle's optimiser is shown this page's contents before it proposes anything.
      </p>
      <div className="row">
        {DOMAINS.map((d) => (
          <button key={d} className={`chip ${d === domain ? 'ok' : ''}`} style={{ cursor: 'pointer' }} onClick={() => setSp({ domain: d })}>
            {domainLabel(d)} · {counts[d] ?? 0}
          </button>
        ))}
      </div>
      {entries.length === 0 && (
        <div className="empty">
          No cycle has run on {domainLabel(domain)}. <code>make loop DOMAIN={domain}</code> starts one.
        </div>
      )}
      {entries.map((e) => (
        <section key={e.cycle} className="band" style={{ marginTop: 'var(--s6)' }}>
          <h2 style={{ marginTop: 0 }}>
            cycle {e.cycle} — {e.champion} → {e.challenger ?? '—'}: {e.outcome?.verdict ?? 'pending'}
            {e.outcome?.passes ? ` (${e.outcome.passes} on train)` : ''}
            {e.outcome?.test_passes ? ` · test ${e.outcome.test_passes}` : ''}
          </h2>
          <div className="chips">
            {e.optimiser_model && <span className="chip">optimiser {e.optimiser_model}</span>}
            {e.optimiser && (
              <span className="chip">
                {e.optimiser.turns} turns · {fmtS(e.optimiser.duration_ms)}
              </span>
            )}
            {e.tokens && (
              <span className="chip">
                tokens opt {fmtK(e.tokens.optimiser_in)}/{fmtK(e.tokens.optimiser_out)} · eval agent {fmtK(e.tokens.eval_agent_in)}/{fmtK(e.tokens.eval_agent_out)}
              </span>
            )}
            {e.outcome?.fixed?.length ? <span className="chip ok">fixed {e.outcome.fixed.map((t) => shortTask(t, 20)).join(', ')}</span> : null}
            {e.outcome?.broken?.length ? <span className="chip warn">broken {e.outcome.broken.map((t) => shortTask(t, 20)).join(', ')}</span> : null}
            {e.outcome?.still_failed?.length ? <span className="chip no">still failed {e.outcome.still_failed.length}</span> : null}
            {e.outcome?.challenger_run && (
              <Link className="chip" to={`/compare?a=${e.champion_run ?? ''}&b=${e.outcome.challenger_run}`}>
                gate view · {shortRun(e.outcome.challenger_run)}
              </Link>
            )}
            {e.outcome?.test_run && (
              <Link className="chip" to={`/runs/${e.outcome.test_run}`}>
                test run · {shortRun(e.outcome.test_run)}
              </Link>
            )}
            {e.challenger && (
              <Link className="chip" to={`/evolution?domain=${domain}&a=${e.champion}&b=${e.challenger}`}>
                diff {e.champion} → {e.challenger}
              </Link>
            )}
          </div>
          <dl className="diff-sum">
            <dt>prompt</dt>
            <dd>{e.prompt_diff_summary || '—'}</dd>
            <dt>helper</dt>
            <dd>{e.helper_diff_summary || '—'}</dd>
            {e.outcome?.reason && (
              <>
                <dt>reason</dt>
                <dd>{e.outcome.reason}</dd>
              </>
            )}
            {e.optimiser?.error && (
              <>
                <dt>harness</dt>
                <dd className="v-warn">{e.optimiser.error}</dd>
              </>
            )}
          </dl>
          {e.diagnoses?.length ? (
            <div className="tw">
              <table>
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
                  {e.diagnoses.map((d, i) => {
                    const o = e.outcome ?? { verdict: 'pending' };
                    const res = o.fixed?.includes(d.task_id) ? 'fixed' : o.broken?.includes(d.task_id) ? 'broken' : o.still_failed?.includes(d.task_id) ? 'still failed' : o.verdict === 'pending' ? 'pending' : 'unchanged';
                    return (
                      <tr key={d.task_id + d.surface + i}>
                        <td className="sub mono small wrap" title={d.task_id} style={{ maxWidth: '18ch' }}>
                          {shortTask(d.task_id, 28)}
                        </td>
                        <td className="wrap">{d.symptom}</td>
                        <td className="wrap">{d.root_cause}</td>
                        <td className="mono">{d.surface}</td>
                        <td className="wrap">{d.change}</td>
                        <td>{d.verified_in_session ? <span className="v-ok">yes</span> : <span className="v-warn">no</span>}</td>
                        <td className={res === 'fixed' ? 'v-ok' : res === 'broken' || res === 'still failed' ? 'v-warn' : 'v-no'}>{res}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : null}
          {e.risks?.length ? (
            <details>
              <summary>risks the optimiser named</summary>
              <ul>
                {e.risks.map((r, i) => (
                  <li key={i} className="small">
                    {r}
                  </li>
                ))}
              </ul>
            </details>
          ) : null}
        </section>
      ))}
    </>
  );
}
