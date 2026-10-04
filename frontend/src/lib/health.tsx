/**
 * A run's harness health on Runs (s13): whether the agent's tools and tool calls did what they
 * are for. Each row is one of s13's milestone checks, read from `GET /api/runs/<id>`'s `health`
 * (`src/tau2_loop/eval/health.py`, arithmetic over `tau2_results.json`). Every number carries
 * its denominator in the same cell.
 */

import type { RunHealth } from './api';

const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;
const int = (n: number) => n.toLocaleString('en-GB');

export type HealthRow = { key: string; label: string; value: string; what: string };

/** Whether the knowledge-tool rows apply: a run that recorded a retrieval variant, or used one. */
export function usesKnowledge(h: RunHealth, retrieval: string | null | undefined): boolean {
  return !!retrieval || h.bm25_calls + h.dense_calls + h.grep_calls + h.shell_calls > 0;
}

export function healthRows(h: RunHealth, knowledge: boolean): HealthRow[] {
  const n = h.conversations;
  const rows: HealthRow[] = [
    {
      key: 'slipped',
      label: 'slipped calls',
      value: `${int(h.slipped)} of ${int(h.replies)} replies (${pct(h.slipped_rate, 1)}), in ${h.slipped_conversations} of ${n} conversations`,
      what: 'a text reply that was really a tool call; native tool calls should make it 0',
    },
    {
      key: 'calls',
      label: 'calls a tool turn',
      value: `${h.calls_per_turn.toFixed(2)}, mean of ${n} conversations (${int(h.tool_calls)} tool calls)`,
      what: 'more than 1.0 means the agent batches calls',
    },
  ];
  if (knowledge) {
    rows.push(
      {
        key: 'shell',
        label: 'shell used',
        value: `${h.shell_conversations} of ${n} conversations, ${int(h.shell_calls)} commands`,
        what: 'the read-only shell over the knowledge base',
      },
      {
        key: 'index',
        label: 'INDEX.md read',
        value: `${h.index_read_conversations} of ${n} conversations`,
        what: "the knowledge base's own table of contents, through the shell",
      },
      {
        key: 'search',
        label: 'searches',
        value: `BM25 ${int(h.bm25_calls)} · dense ${int(h.dense_calls)} · grep ${int(h.grep_calls)}, of ${int(h.tool_calls)} tool calls`,
        what: 'knowledge searches by kind',
      },
      {
        key: 'lookups',
        label: 'look-ups before the first action',
        value: `${h.lookups_before_first_action.toFixed(2)} a conversation, mean of ${n}`,
        what: 'searches before the first verification, unlock, discoverable call or transfer',
      },
      {
        key: 'bare',
        label: 'bare discoverable calls',
        value: `${int(h.bare_discoverable_calls)} in ${h.bare_discoverable_conversations} of ${n} conversations`,
        what: 'a discoverable tool called by its own name: it runs, and the grader counts nothing',
      },
    );
  }
  rows.push({
    key: 'email',
    label: 'account-email calls',
    value: `${int(h.account_email_calls)} in ${h.account_email_conversations} of ${n} conversations`,
    what: "a call carrying the CLI's account address instead of the customer's",
  });
  if (knowledge) {
    rows.push(
      {
        key: 'failures',
        label: 'sandbox · dense failures',
        value: `${h.sandbox_failures} sandbox · ${h.dense_errors} dense · ${h.shell_blocked} shell blocked, of ${int(h.tool_errors)} tool errors`,
        what: 'the harness failing, not the agent',
      },
      {
        key: 'docs',
        label: 'required documents read',
        value: h.required_docs_read_share == null ? '—' : `${pct(h.required_docs_read_share)} of a task's required documents, mean over the conversations whose task lists any`,
        what: 'documents the agent had in front of it in full: search results, or a file printed through the shell',
      },
    );
  } else {
    rows.push({ key: 'errors', label: 'tool errors', value: `${int(h.tool_errors)} of ${int(h.tool_calls)} tool calls`, what: 'calls the environment answered with an error' });
  }
  return rows;
}

/** The section's assertion, at most ten words, with its denominator. */
export function healthClaim(h: RunHealth): string {
  return h.slipped ? `${int(h.slipped)} of ${int(h.replies)} replies were slipped tool calls` : `none of ${int(h.replies)} replies was a slipped tool call`;
}

export function HarnessHealth({ h, retrieval, runId }: { h: RunHealth; retrieval: string | null | undefined; runId: string }) {
  const rows = healthRows(h, usesKnowledge(h, retrieval));
  return (
    <figure>
      <div className="label fig-title">
        harness health · {h.conversations} conversations, {h.passed} passed
      </div>
      <div className="tw">
        <table className="health">
          <thead>
            <tr>
              <th>measure</th>
              <th>this run</th>
              <th>what it checks</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key}>
                <td className="sub">{r.label}</td>
                <td className="mono wrap">{r.value}</td>
                <td className="wrap small muted">{r.what}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figcaption>
        Counted from the run&rsquo;s own transcripts: no model, no environment, no tracking server.
        <span className="path">runs/{runId}/tau2_results.json · src/tau2_loop/eval/health.py</span>
      </figcaption>
    </figure>
  );
}
