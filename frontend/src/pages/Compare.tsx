import { useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { type Registries, type RunMeta, type TaskResult, domainLabel, shortRun, shortTask, useGet } from '../lib/api';
import { trialId, trialPath } from '../lib/url';

type Verdict = { promote: boolean; champion_passed: number; challenger_passed: number; n: number; fixed: string[]; broken: string[]; p_value: number; alpha: number; reason: string };
type ComparePayload = { verdict: Verdict; rows: { task_id: string; trial: number; purpose: string; a: TaskResult | null; b: TaskResult | null }[]; a: RunMeta; b: RunMeta };

export function Compare() {
  const [sp, setSp] = useSearchParams();
  const { data: runs } = useGet<RunMeta[]>('/api/runs');
  const { data: regs } = useGet<Registries>('/api/registry');
  const real = (runs ?? []).filter((r) => !r.dry_run && r.summary?.n_scored);
  const a = sp.get('a') ?? '';
  const b = sp.get('b') ?? '';
  useEffect(() => {
    if (!a && !b && regs && real.length) {
      const withChamp = Object.values(regs).find((r) => r.champion);
      const champ = withChamp?.champion?.run_id ?? real[0].run_id;
      const other = withChamp?.challenger?.run_id ?? real.filter((r) => r.run_id !== champ && r.domain === (withChamp?.domain ?? real[0].domain)).slice(-1)[0]?.run_id ?? champ;
      setSp({ a: champ, b: other }, { replace: true });
    }
  }, [a, b, regs, real, setSp]);
  const { data, error } = useGet<ComparePayload>(a && b ? `/api/compare?a=${a}&b=${b}` : null);
  const v = data?.verdict;
  const sameDomain = data ? data.a.domain === data.b.domain && data.a.split === data.b.split : true;
  return (
    <>
      <p className="label">Promotion gate</p>
      <h1>
        A challenger is promoted when a one-sided McNemar test says it <em>beats</em> the champion
      </h1>
      <p className="lead">
        Two runs of the same twenty train tasks are paired by task; only the discordant tasks count: b fixed, c broken. Under "no real
        difference" each is a coin flip, so p = P(breaks ≤ c | b + c, ½), and the gate promotes at p &lt; 0.05. With twenty tasks that
        is blunt by construction: five fixes and no breaks is the smallest result that clears it (p = 0.031); one break costs three extra fixes.
      </p>
      <div className="row">
        <label style={{ flex: '1 1 320px' }}>
          <span className="label">champion run</span>
          <select value={a} onChange={(e) => setSp({ a: e.target.value, b })}>
            <option value="">—</option>
            {real.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {domainLabel(r.domain)} · {shortRun(r.run_id)} · {r.summary?.passed}/{r.summary?.n_scored}
              </option>
            ))}
          </select>
        </label>
        <label style={{ flex: '1 1 320px' }}>
          <span className="label">challenger run</span>
          <select value={b} onChange={(e) => setSp({ a, b: e.target.value })}>
            <option value="">—</option>
            {real.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {domainLabel(r.domain)} · {shortRun(r.run_id)} · {r.summary?.passed}/{r.summary?.n_scored}
              </option>
            ))}
          </select>
        </label>
      </div>
      {error && <div className="empty">{error}</div>}
      {data && !sameDomain && <div className="empty">These runs are not on the same domain and split; the pairing below is by task id only and means nothing.</div>}
      {v && (
        <div className="kpis">
          <div className="kpi">
            <div className="label">verdict</div>
            <div className={`n ${v.promote ? 'ok' : 'warn'}`}>{v.promote ? 'promote' : 'hold'}</div>
            <div className="b">{v.reason}</div>
          </div>
          <div className="kpi">
            <div className="label">p-value</div>
            <div className={`n ${v.p_value < v.alpha ? 'ok' : 'warn'}`}>{v.p_value.toFixed(3)}</div>
            <div className="b">one-sided exact McNemar · α = {v.alpha}</div>
          </div>
          <div className="kpi">
            <div className="label">passes</div>
            <div className="n">
              {v.champion_passed} → {v.challenger_passed}
            </div>
            <div className="b">of {v.n} paired conversations</div>
          </div>
          <div className="kpi">
            <div className="label">fixed</div>
            <div className="n ok">{v.fixed.length}</div>
            <div className="b mono small">{v.fixed.map((t) => shortTask(t, 24)).join(', ') || '—'}</div>
          </div>
          <div className="kpi">
            <div className="label">broken</div>
            <div className={`n ${v.broken.length ? 'warn' : ''}`}>{v.broken.length}</div>
            <div className="b mono small">{v.broken.map((t) => shortTask(t, 24)).join(', ') || '—'}</div>
          </div>
        </div>
      )}
      {data && (
        <div className="tw">
          <table>
            <thead>
              <tr>
                <th>task</th>
                <th>purpose</th>
                <th>champion</th>
                <th>challenger</th>
                <th>change</th>
                <th>champion components</th>
                <th>challenger components</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((r) => {
                const ca = r.a?.correct;
                const cb = r.b?.correct;
                const change = ca === cb ? '' : cb ? 'fixed' : 'broken';
                return (
                  <tr key={`${r.task_id}#${r.trial}`} className={change === 'fixed' ? 'pro' : ''}>
                    <td className="sub mono small" title={r.task_id}>
                      {r.b ? <Link to={trialPath(b, trialId(r.b))}>{shortTask(r.task_id, 26)}</Link> : shortTask(r.task_id, 26)}
                    </td>
                    <td className="wrap small muted">{r.purpose}</td>
                    <td>{ca == null ? '—' : ca ? <span className="status ok">pass</span> : <span className="status err">fail</span>}</td>
                    <td>{cb == null ? '—' : cb ? <span className="status ok">pass</span> : <span className="status err">fail</span>}</td>
                    <td className={change === 'fixed' ? 'v-ok' : change === 'broken' ? 'v-warn' : 'v-no'}>{change || 'same'}</td>
                    <td className="mono small">{comps(r.a)}</td>
                    <td className="mono small">{comps(r.b)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function comps(r: TaskResult | null): string {
  if (!r) return '—';
  const p: string[] = [];
  if (r.db_check != null) p.push(`DB ${r.db_check ? '✓' : '✗'}`);
  if (r.action_checks) p.push(`act ${r.action_checks}`);
  if (r.communicate_checks) p.push(`said ${r.communicate_checks}`);
  if (r.nl_assertions) p.push(`NL ${r.nl_assertions}`);
  if (r.env_assertions) p.push(`env ${r.env_assertions}`);
  if (r.error) p.push(r.error);
  return p.join(' · ');
}
