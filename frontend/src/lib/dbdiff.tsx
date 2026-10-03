import { useGet } from './api';
import { Loading, Points } from './ui';

/**
 * Where a conversation's final database differs from gold's: both sides of tau2's DB check,
 * rebuilt offline from the trace and the task (`GET /api/runs/<run>/<task>/t<n>/db`). Each
 * differing field shows its value before the conversation, what the agent left and what gold
 * expects; each record names the agent's calls and the expected actions that changed it, so a
 * missed write, a wrong write and a write with the wrong arguments read apart.
 */

type Field = { field: string; before: string | null; agent: string | null; gold: string | null };
type AgentCall = { msg: number; name: string; arguments: Record<string, unknown>; requestor: string };
type GoldAction = { action_id: string; name: string; arguments: Record<string, unknown> };
export type DbRecord = {
  record: string;
  kind: 'changed' | 'agent_only' | 'gold_only';
  /** why it differs, as `replay.record_verdict` classes it: the optimiser reads the same word */
  verdict: 'missed write' | 'wrong write' | 'wrong arguments' | 'differs';
  fields: Field[];
  agent_calls: AgentCall[];
  gold_actions: GoldAction[];
};
type DbDiffPayload = {
  match: boolean;
  graded: boolean | null;
  records: DbRecord[];
  fields: number;
  shown: number;
  gold_errors: { action_id: string; name: string; error: string }[];
};

const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? '' : 's'}`;
const call = (c: AgentCall) => `${c.name} at message ${c.msg}`;
const action = (a: GoldAction) => `${a.action_id} ${a.name}`;

/** One record's difference in a sentence: who changed it, and who did not. */
export function verdictOf(r: DbRecord): { kind: string; text: string } {
  const mine = `the agent’s ${[...new Set(r.agent_calls.map(call))].join(', ')}`;
  const gold = `gold’s ${r.gold_actions.map(action).join(', ')}`;
  const text = {
    'missed write': `${gold} ${r.kind === 'gold_only' ? 'creates it' : 'changes it'}; the agent never wrote to it.`,
    'wrong write': `${mine} ${r.kind === 'agent_only' ? 'created it' : 'changed it'}; gold leaves it as it was.`,
    'wrong arguments': `both change it, and the results differ: ${mine}; ${gold}.`,
    differs: 'no recorded write on either side changed it directly.',
  }[r.verdict];
  return { kind: r.verdict, text };
}

/** A whole record, indented; one cut at its length cap is shown as it came. */
function pretty(x: string): string {
  try {
    return JSON.stringify(JSON.parse(x), null, 1);
  } catch {
    return x;
  }
}

/** A JSON value as a person reads it: a plain string without its quotes, nothing as "not set". */
function Value({ v }: { v: string | null }) {
  if (v == null) return <span className="dim">not set</span>;
  try {
    const x: unknown = JSON.parse(v);
    if (typeof x === 'string') return <>{x}</>;
  } catch {
    /* a value cut at its length cap is not JSON any more: show it as it is */
  }
  return <>{v}</>;
}

function Calls({ r }: { r: DbRecord }) {
  const rows = [
    ...r.agent_calls.map((c, i) => ({ key: `a${i}`, who: `answering agent · message ${c.msg}`, name: c.name, args: c.arguments })),
    ...r.gold_actions.map((a) => ({ key: a.action_id, who: `expected action ${a.action_id}`, name: a.name, args: a.arguments })),
  ];
  if (!rows.length) return null;
  return (
    <div className="db-calls">
      {rows.map((x) => (
        <details key={x.key}>
          <summary className="small">
            {x.who} · <span className="mono">{x.name}</span>
          </summary>
          <pre>{JSON.stringify(x.args, null, 1)}</pre>
        </details>
      ))}
    </div>
  );
}

export function DbDiff({ url }: { url: string }) {
  const { data, error } = useGet<DbDiffPayload>(url);
  if (!data) return <Loading error={error} />;
  const recs = data.records;
  return (
    <section className="db-diff">
      <h2>
        {data.match
          ? 'The final database equals gold’s'
          : `The final database differs from gold’s in ${plural(data.fields, 'field')} of ${plural(recs.length, 'record')}`}
      </h2>
      <Points
        className="small muted"
        items={[
          <>
            <b>Both sides of the DB check, rebuilt from the trace</b>: the conversation’s writes on the task’s initial state,
            against every expected action on it.
          </>,
          data.graded != null && data.graded !== data.match ? (
            <span className="v-warn">
              <b>The rebuild disagrees with the grade</b>: the run was graded {data.graded ? 'a match' : 'a mismatch'}.
            </span>
          ) : null,
          data.shown < data.fields ? <>The first {data.shown} of {data.fields} differing fields are listed.</> : null,
          data.gold_errors.length ? <>{plural(data.gold_errors.length, 'expected action')} raised on gold’s side and changed nothing.</> : null,
        ]}
      />
      {recs.map((r) => {
        const v = verdictOf(r);
        return (
          <figure key={r.record} className="db-record">
            <h3 className="db-title">
              <span className="mono">{r.record}</span> · <span className="status err">{v.kind}</span>
            </h3>
            <p className="small">{v.text}</p>
            <div className="tw">
              <table>
                <thead>
                  <tr>
                    <th>field</th>
                    <th>before the conversation</th>
                    <th>the agent left</th>
                    <th>gold expects</th>
                  </tr>
                </thead>
                <tbody>
                  {r.fields.map((f) => (
                    <tr key={f.field}>
                      <td className="mono small">{f.field}</td>
                      {f.field === '(record)' ? (
                        [f.before, f.agent, f.gold].map((x, i) => (
                          <td key={i} className="small">
                            {x == null ? (
                              <span className="dim">no such record</span>
                            ) : (
                              <details>
                                <summary>the whole record</summary>
                                <pre>{pretty(x)}</pre>
                              </details>
                            )}
                          </td>
                        ))
                      ) : (
                        <>
                          <td className="mono small db-val">
                            <Value v={f.before} />
                          </td>
                          <td className="mono small db-val">
                            <Value v={f.agent} />
                          </td>
                          <td className="mono small db-val">
                            <Value v={f.gold} />
                          </td>
                        </>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Calls r={r} />
          </figure>
        );
      })}
    </section>
  );
}
