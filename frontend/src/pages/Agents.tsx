import { Link, useParams } from 'react-router-dom';
import { type AgentInfo, type Registries, domainLabel, shortRun, useGet } from '../lib/api';
import { agentPath, runPath } from '../lib/url';

type AgentsPayload = { versions: AgentInfo[]; registry: Registries };
type AgentFiles = { domain: string; name: string; fingerprint: string; config: Record<string, unknown>; files: Record<string, string> };

export function Agents() {
  const { domain, name } = useParams();
  const { data } = useGet<AgentsPayload>('/api/agents');
  const { data: files } = useGet<AgentFiles>(domain && name ? `/api/agents/${domain}/${name}` : null);
  const role = (d: string, n: string) => {
    const r = data?.registry[d];
    if (r?.champion?.agent === n) return <span className="status ok">champion</span>;
    if (r?.challenger?.agent === n) return <span className="status warn">challenger</span>;
    return <span className="status no">held</span>;
  };
  return (
    <>
      <p className="label">Versions</p>
      <h1>
        Every version is a folder per domain; the champion is the one the <em>gate</em> kept
      </h1>
      <p className="lead">
        <code>agents/&lt;domain&gt;/vN/</code> holds a system prompt with a <code>{'{policy}'}</code> slot, an optional helper with three hooks, and a
        frozen config. The optimiser writes the next folder and a <code>diagnosis.json</code> beside it; the gate decides which folder the
        domain's registry points at. <Link to="/evolution">Evolution</Link> shows the diff between two versions with that reasoning.
      </p>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th>domain</th>
              <th>version</th>
              <th>role</th>
              <th>fingerprint</th>
              <th>model</th>
              <th>helper hooks</th>
              <th>runs</th>
            </tr>
          </thead>
          <tbody>
            {data?.versions.map((v) => (
              <tr key={v.ref} className={data.registry[v.domain]?.champion?.agent === v.name ? 'pro' : ''}>
                <td>{domainLabel(v.domain)}</td>
                <td className="sub">
                  <Link to={agentPath(v.domain, v.name)}>{v.name}</Link>
                </td>
                <td>{role(v.domain, v.name)}</td>
                <td className="mono">{v.fingerprint}</td>
                <td className="mono">
                  {String(v.config.model)} · {String(v.config.tool_mode)}
                </td>
                <td className="wrap mono small">{v.helper_functions.join(', ') || '—'}</td>
                <td className="small">
                  {v.runs.map((r) => (
                    <div key={r}>
                      <Link to={runPath(r)}>{shortRun(r)}</Link>
                    </div>
                  ))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {domain && name && files && (
        <>
          <h2>
            {domainLabel(domain)} / {name} — the files, verbatim
          </h2>
          {Object.entries(files.files).map(([fname, text]) => (
            <details key={fname} open={fname !== 'agent.yaml'}>
              <summary>
                {fname} <span className="muted small">({text.length.toLocaleString()} chars)</span>
              </summary>
              <div className="code">
                <pre>{text}</pre>
              </div>
            </details>
          ))}
          {data?.versions.find((v) => v.domain === domain && v.name === name)?.diagnosis && (
            <>
              <h3>diagnosis.json — why this version exists</h3>
              <div className="code">
                <pre>{JSON.stringify(data.versions.find((v) => v.domain === domain && v.name === name)?.diagnosis, null, 1)}</pre>
              </div>
            </>
          )}
        </>
      )}
    </>
  );
}
