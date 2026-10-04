/**
 * Evals › a banking task: the documents tau2 says the agent needs (`required_documents`), each by
 * its title with its full text one click away, and, per agent version, whether that version's
 * newest scored conversation of the task read it whole, only saw it named, or never reached it.
 * Titles come from the committed index (`data/tasks/banking_knowledge_documents.json`), the text
 * from tau2's files (`GET /api/domains/<d>/documents/<id>`), the reads from the run's harness
 * health (`GET /api/domains/<d>/tasks/<id>/reads`, `src/tau2_loop/eval/health.py`).
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { type KbDocument, type KbDocumentText, type TaskReads, type VersionReads, useGet } from './api';
import { Points } from './ui';
import { trialId, trialPath } from './url';

export type DocState = 'read' | 'seen' | 'none';

/** What one version did with one document: read it whole, saw it named, or never reached it. */
export function docState(v: Pick<VersionReads, 'read' | 'seen'>, id: string): DocState {
  if (v.read.includes(id)) return 'read';
  if (v.seen.includes(id)) return 'seen';
  return 'none';
}

/** A version's bottom-row counts, each with its denominator: read `1 / 4`, seen `0 / 4`. Only
 *  the required documents count, whatever else the conversation read. */
export function readTally(v: Pick<VersionReads, 'read' | 'seen'>, required: string[]): { read: string; seen: string } {
  const n = required.length;
  const read = required.filter((id) => v.read.includes(id)).length;
  const seen = required.filter((id) => !v.read.includes(id) && v.seen.includes(id)).length;
  return { read: `${read} / ${n}`, seen: `${seen} / ${n}` };
}

const STATE_WORD: Record<DocState, string> = { read: 'read', seen: 'seen', none: '—' };

export function RequiredDocs({ domain, taskId, docs }: { domain: string; taskId: string; docs: KbDocument[] }) {
  const { data: reads, error } = useGet<TaskReads>(
    `/api/domains/${encodeURIComponent(domain)}/tasks/${encodeURIComponent(taskId)}/reads`,
  );
  const versions = reads?.versions ?? [];
  const required = docs.map((d) => d.id);
  return (
    <section className="docreads" aria-labelledby="docreads-h">
      <h3 id="docreads-h">
        Documents the agent needs to find — {docs.length}{' '}
        <span className="h-src">
          · tau2's <code>required_documents</code>
        </span>
      </h3>
      <div className="tw fit">
        <table>
          <thead>
            <tr>
              <th>document</th>
              {versions.map((v) => (
                <th key={v.version} className="st">
                  <span className="ver">{v.version}</span>
                  <span className="path">
                    <Link to={trialPath(v.run_id, trialId({ task_id: taskId, trial: v.trial }))} title={`${v.run_id}, trial ${v.trial}`}>
                      <span className={v.passed ? 'v-ok' : 'v-no'}>{v.passed ? '✓ passed' : '✗ failed'}</span>
                    </Link>
                  </span>
                  <span className="path">{v.split}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {docs.map((d) => (
              <tr key={d.id}>
                <td className="doc">
                  <span className="t">{d.title ?? 'untitled'}</span>
                  <span className="path">{d.id}</span>
                  <DocText domain={domain} doc={d} />
                </td>
                {versions.map((v) => {
                  const s = docState(v, d.id);
                  return (
                    <td key={v.version} className="st">
                      <span className={`dr ${s}`}>{STATE_WORD[s]}</span>
                    </td>
                  );
                })}
              </tr>
            ))}
            {versions.length > 0 &&
              (['read', 'seen'] as const).map((k) => (
                <tr key={k} className={`tot ${k}`}>
                  <td>{k === 'read' ? 'read' : 'seen only'}</td>
                  {versions.map((v) => (
                    <td key={v.version} className="st mono">
                      {readTally(v, required)[k]}
                    </td>
                  ))}
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      {error && <p className="small muted">Could not load the versions' reads: {error}</p>}
      {reads && versions.length === 0 && <p className="small muted">No scored run of any version has played this task yet.</p>}
      <Points
        className="small muted"
        items={[
          <>
            <b>read</b>: the whole document came back in a search or <code>grep</code> result, or the shell printed it by name
            (<code>cat</code>, <code>head</code>, …).
          </>,
          <>
            <b>seen</b>: only its id or file name came back (a listing, a <code>grep</code> line, or a file the shell printed
            through a wildcard, which the count cannot name); <b>—</b>: never reached.
          </>,
          <>
            <b>Each column is the version's newest scored run with this task</b>; the heading opens that conversation.
          </>,
        ]}
      />
    </section>
  );
}

/** A document's full text, fetched when opened. The demo image ships without tau2: there the
 *  route answers 404 and the index's title and size are all there is. */
function DocText({ domain, doc }: { domain: string; doc: KbDocument }) {
  const [open, setOpen] = useState(false);
  const { data, error } = useGet<KbDocumentText>(
    open ? `/api/domains/${encodeURIComponent(domain)}/documents/${encodeURIComponent(doc.id)}` : null,
  );
  return (
    <details className="doctext" onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
      <summary>
        <span>full text{doc.chars != null ? ` · ${doc.chars.toLocaleString('en-GB')} characters` : ''}</span>
      </summary>
      {open &&
        (data ? (
          <div className="code">
            <pre>{data.content}</pre>
          </div>
        ) : (
          <p className="small muted">
            {error
              ? error.startsWith('404')
                ? "The text is not in this image: tau2's documents ship only with the submodule."
                : `Could not load: ${error}`
              : 'Loading…'}
          </p>
        ))}
    </details>
  );
}
