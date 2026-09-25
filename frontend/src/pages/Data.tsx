import { useState } from 'react';
import { Link } from 'react-router-dom';
import { type DomainDetail, type DomainSummary, domainLabel, useGet } from '../lib/api';
import { domainPath } from '../lib/url';

export function Data() {
  const { data: domains } = useGet<DomainSummary[]>('/api/domains');
  const [open, setOpen] = useState<string>('airline');
  const { data: detail } = useGet<DomainDetail>(`/api/domains/${open}`);
  return (
    <>
      <p className="label">The benchmark</p>
      <h1>
        Four domains, each a policy, a toolset and a database; nothing is <em>held out</em>, so the test split is ours
      </h1>
      <p className="lead">
        τ²-bench ships every task with its answer key. The loop therefore cuts its own split per domain — twenty train, twenty
        test, drawn once with seed 300 from the public <code>base</code> set and committed under <code>data/splits/</code> — and the
        agent reads only the policy and the tools; the scenario and the expected actions stay with the simulator and the evaluator.
      </p>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>domain</th>
              <th className="num">base tasks</th>
              <th className="num">train</th>
              <th className="num">test</th>
              <th className="num">reserve</th>
              <th className="num">policy words</th>
              <th className="num">agent tools</th>
              <th>reward basis (tasks)</th>
            </tr>
          </thead>
          <tbody>
            {(domains ?? []).map((d) => (
              <tr key={d.domain} className={`click ${d.domain === open ? 'pro' : ''}`} onClick={() => setOpen(d.domain)}>
                <td className="sub">{domainLabel(d.domain)}</td>
                <td className="num">{d.base_n}</td>
                <td className="num">{d.train}</td>
                <td className="num">{d.test}</td>
                <td className="num">{d.reserve_n}</td>
                <td className="num">{d.policy_words?.toLocaleString()}</td>
                <td className="num">{d.n_tools}</td>
                <td className="small wrap">{Object.entries(d.reward_bases).map(([k, v]) => `${k} ${v}`).join(' · ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        Telecom's base set is 114 of its 2,285 generated tasks (tau2's own <code>base</code> split); banking_knowledge has no tau2 split, so its base is all 97. The reserve is never run in this build.
      </p>

      {detail && (
        <>
          <h2>
            {domainLabel(open)} — the policy the agent is given, verbatim ({detail.policy_words.toLocaleString()} words)
          </h2>
          <p className="small muted">
            Split method: <code>{detail.split.method}</code>. Task list: <Link to={domainPath(open)}>Tasks · {domainLabel(open)}</Link>.
          </p>
          <details>
            <summary>{detail.tools.length} tools the harness exposes to the agent</summary>
            <div className="tw">
              <table>
                <thead>
                  <tr>
                    <th>tool</th>
                    <th>description</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.tools.map((t) => (
                    <tr key={t.name}>
                      <td className="sub mono">{t.name}</td>
                      <td className="wrap small">{t.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
          <details open>
            <summary>policy.md</summary>
            <div className="code">
              <pre>{detail.policy}</pre>
            </div>
          </details>
        </>
      )}
    </>
  );
}
