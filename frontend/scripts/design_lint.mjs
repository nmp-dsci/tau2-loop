#!/usr/bin/env node
/**
 * DESIGN.md's never-do list as a check: a hex colour outside `tokens.css`
 * fails the build; so does a font-size under 12.8px (--t-2) declared in CSS.
 * No dependencies; runs in CI from the frontend job.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const srcDir = join(root, 'src');

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.(tsx?|css)$/.test(name)) out.push(p);
  }
  return out;
}

const HEX = /(?<!&)#[0-9a-fA-F]{6}\b|(?<!&)#[0-9a-fA-F]{3}\b(?![0-9a-fA-F])/g;
const SMALL_PX = /font-size:\s*(\d+(?:\.\d+)?)px/g;
const BANNED_FONTS = /font-family:[^;]*\b(Inter|Roboto|Helvetica)\b/g;

const problems = [];
for (const file of walk(srcDir)) {
  const rel = relative(root, file);
  const text = readFileSync(file, 'utf8');
  if (!rel.endsWith('tokens.css')) {
    for (const m of text.matchAll(HEX)) problems.push(`${rel}: hex colour ${m[0]} — use a token`);
  }
  for (const m of text.matchAll(SMALL_PX)) {
    if (Number.parseFloat(m[1]) < 12.8) problems.push(`${rel}: font-size ${m[1]}px is under --t-2`);
  }
  for (const m of text.matchAll(BANNED_FONTS)) problems.push(`${rel}: banned font ${m[1]}`);
}
for (const p of problems) console.log(`  ✗ ${p}`);
if (problems.length) {
  console.log(`design lint failed: ${problems.length}`);
  process.exit(1);
}
console.log('design lint: clean');
