# DESIGN.md — the brief tau2-loop is built to

The source of truth for how anything visual in this project looks and reads:
Lavish review artifacts in `.lavish/`, the run viewer, README figures, and the
case study this project will eventually become on nmp-dsci.github.io. It is a
port of the portfolio site's brief (`nmp-dsci.github.io/DESIGN.md`, direction
"Field Guide", chosen 31 Aug 2026) so that every surface the author ships reads
as one system. If this file and the site's `assets/css/tokens.css` disagree,
the site wins — fix this file.

---

## 1. Who it is for and what it has to do

**Reader:** the author reviewing a plan before code exists, and later a hiring
manager or senior engineer scanning the case study for 60–90 seconds. Both
scan headline to headline and read a quarter of the words.

**The one job:** make the decision, risk or number obvious at a glance, with
its evidence directly under it. A plan artifact exists to be annotated; a
viewer exists to find the failed task and its trace in two clicks.

**What we are not:** a dashboard template, a pitch deck, or a generated-looking
page. Engineered, not generated.

---

## 2. The rules

### Presentation mode
Every page is a presentation. The unit is a slide:

```
ASSERTION HEADLINE  — a claim, with its number
VISUAL              — the chart, diagram or table that IS the evidence
ONE INSIGHT LINE    — what it means, once, in under 25 words
```

- **Headings state claims, never topics.** A section heading keeps a stable
  label, then carries the claim after an em dash, at most ten words:
  `## 3 · Labels — ten gold tasks are all anyone gets`. Banned heading forms:
  `Architecture`, `Cost`, `Overview`, `Setup`, `Results`, `Background`,
  `Approach`, or any bare noun.
- **Order, always:** outcome (with baseline and caveat) → the visual → the
  mechanism → the setup. A reader who leaves after the first screen still has
  the result.
- **One insight, not a bullet pile.** A list that restates the figure is the
  AI-slop signature. The exception is a numbered slide with three or four
  question-and-answer bullets that say what the figure cannot.
- **Every figure is introduced before it is read:** a mono uppercase
  `fig-title` naming what it is, the frame, then a caption stating the
  takeaway first and the source path second.
- **Every number carries its baseline or denominator in the same sentence.**
  "89.95% hard, against 19.84% for the plain Sonnet 4 ReAct baseline (378
  tasks)" — never a bare "89.95%".

### Type
- Three voices from one superfamily, each with one job. **IBM Plex Serif** is
  the anchor display face: `h1` and lead paragraphs. **IBM Plex Sans** is the
  interface: `h2`, `h3`, labels, buttons, table cells. **IBM Plex Mono** is
  data: numbers, paths, keys, status words. Mono is never prose or a heading;
  serif is never a control.
- **One decorated keyword per page, only inside the `h1`:** a single `<em>` in
  serif italic 400 in `--accent`. Two is scatter.
- One scale, 16px × 1.25, declared as `--t-2 … --t6`. **Nothing smaller than
  `--t-2` (12.8px)**, including captions, legends and SVG labels at rendered
  size.
- Reading measure `--measure: 68ch`, line-height 1.6, paragraphs under 80
  words. `text-wrap: balance` on headings, `pretty` on paragraphs.
- Letter-spacing exists in one place: `--track-label` on uppercase mono labels.

### Colour
- 60/30/10: surfaces (`--bg --panel --band`), ink (four steps), accent.
- **The accent means "act on this".** Primary buttons, links, the recommended
  option, the brand. Never decoration, never borders that carry no status,
  never a diagram fill without meaning. `--accent` here is red, not the
  portfolio's green: this project's one deliberate divergence from the site
  palette, recorded 2026-09-25 (plan `.lavish/s04_…`, decision Q3-A).
- **`--ok` means exactly one thing: passed** — a passed gate, a correct
  conversation, a promoted champion (`.v-ok`, `.chip.ok`, `.status.ok`,
  `tr.pro`). It is this project's extension to the site palette, not a site
  token, and exists because `--accent` is red here: a pass must never render
  as a failure.
- **`--amber` means exactly one thing: partial** (a warning, a caveat, a
  provisional number). `--line-3` outlines mean **designed, not built**.
- Status is never colour alone; every glyph is accompanied by the word.
- Light is the reference palette; dark redefines colour only. Every text and
  surface pair holds 4.5:1 in both themes (3:1 for meaningful borders).

### Space and shape
- One 4/8-point ladder (`--s1 … --s10`); one section rhythm.
- **One radius** (`--r: 10px`), one small radius for chips and cells
  (`--r-sm: 6px`), one shadow. Mismatched radii are the tell of no system.
- Sections alternate `--bg` and `--band`. Cards sit on the band without
  borders; space separates things, not lines.

### Motion
- Micro only: 300ms `--dur` with `--ease` on hover, focus and reveal. No
  scroll-jacking, parallax or autoplay. `prefers-reduced-motion` zeroes `--dur`.
- Focus is always visible: a 2px `--focus` ring with a 2px offset.

### Diagrams and charts (the `.dia` contract)
- Hand-authored inline SVG with `viewBox` and `width: 100%`, never fixed
  pixels; every element inside the viewBox.
- Colour through `currentColor` and the tokens so one file is correct in both
  themes. Class vocabulary: `.nd` node, `.nd.hi` promoted node, `.nd.ext`
  external (dashed `--line-3`), `.ed` edge, `.ed.dash`, `.tx` label, `.tx.s`
  small label, `.tx.k` key label, `.bar`, `.bar.hi`, `.bar.pro`, `.ax`, `.gl`,
  `.ci`, `.pt`, `.cap` caption.
- A `<title>` in every meaningful `<g>` and a stable `id` so reviewers can
  annotate; an `aria-label` describing the whole figure.
- **Never draw a number that is not in a repo.** A chart with no committed
  source looks like evidence and is not.

### Tables
- Semantic `<table>` inside a `.tw` scroll wrapper; mono uppercase `thead`;
  `td.num` for numbers (tabular figures), `td.sub` for the row identity, a
  `.path` line under it for the file it came from.
- Status words in the cell (`holds`, `partial`, `fails`), tinted with `.v-ok`,
  `.v-warn`, `.v-no`; the promoted row gets `tr.pro`.

### Voice
Write like an engineer explaining to another engineer. British/Australian
spelling. No "seamless", "robust", "cutting-edge", "leverage". Inline `code`
for paths, identifiers and commands. A smaller true number always beats a
bigger unverified one.

---

## 3. Never do this

1. **No ghost primary action.** One solid `--accent` button; the secondary is
   an underlined link. Never two outlined buttons side by side.
2. **No text under `--t-2`.**
3. **No accent outside action, link, recommended and passed.**
4. **No decorative background pattern, gradient card, or pastel panel.**
5. **No placeholder for a result that does not exist.** No "score coming".
6. **No headline metric strip without stakes.** A number appears with the
   thing it measures and the baseline it beats, or not at all.
7. **No purple, no Inter, no wide letter-spacing on display type.**
8. **No paragraph over 80 words; no heading that names a topic.** Two or more
    distinct points are a bold-led list, not a paragraph: structure, not blobs of text.
9. **No published figure with no committed source.**
10. **No second decorated keyword.**
11. **No standalone control under 24×24px**; inline links in prose are exempt.
12. **No link distinguished by colour alone inside prose.** Underline it.
13. **No `display:grid` or `display:flex` on an element whose children are
    mixed inline nodes** (a `<b>` followed by bare text, a label with text
    beside an input). Every inline run becomes its own grid item and the text
    wraps under the marker. Numbered markers use `position:absolute` on a
    padded block; controls wrap their text in a `<span>`.
14. **No side-scrolling page body.** Only tables, code and diagrams may be
    wider, each inside its own `overflow-x: auto` wrapper.
15. **No hand-maintained duplicate of a list that exists in data.** The viewer
    reads `runs/` and `agents/`; it never hard-codes a run name.

---

## 4. Where it applies

| Surface | What it is | How the brief applies |
|---|---|---|
| `.lavish/sNN_*.html` | the review artifact written before each change | full system: tokens block copied verbatim, assertion headings, `.dia` figures, `form.q` decision controls |
| `frontend/` | the React + Vite demo site (served from FastAPI, deployed to App Runner as read-only) | `frontend/src/tokens.css` is the token block below, including its two recorded divergences (`--accent` red, `--ok` added); `frontend/scripts/design_lint.mjs` hard-fails on hex outside it; three voices; status words with glyphs; tables per §2; every page led by an assertion headline; no metric strip without a baseline |
| `README.md`, `docs/` | the repo's public face | assertion headings; every number with its denominator; figures as committed SVG with a title and caption |
| `nmp-dsci.github.io/_projects/tau2-loop.md` | the eventual case study | the site's own contract applies, run `scripts/lint_case_study.py` there |

## 5. The tokens

The light palette below is the site's
(`nmp-dsci.github.io/assets/css/tokens.css`), with this project's two recorded
divergences: `--accent` is red, not the site's green, and `--ok` carries the
"passed" meaning the site's green accent carried (§2 Colour). Everything else
is pasted into every artifact unchanged, with the dark override that follows
it. Fonts load from Google Fonts in artifacts (the site self-hosts).

```css
:root{
  --bg:#F7F6F2; --panel:#FFFFFF; --band:#EFEEE8;
  --ink:#1A1D1B; --ink-2:#3B403D; --muted:#4A514D; --faint:#5C635E;
  --line:#DCDAD2; --line-2:#C7C5BC; --line-3:#8A8880;
  --accent:#8A2B1D; --accent-ink:#FFFFFF; --accent-soft:#F5DCD7; /* divergence: red, not the site's #0A6552 */
  --amber:#8A5006; --amber-soft:#F6E6CF;
  --ok:#0A6552; --ok-soft:#DFEFE8; /* project extension: passed (§2 Colour) */
  --sans:'IBM Plex Sans',system-ui,-apple-system,sans-serif;
  --serif:'IBM Plex Serif',Georgia,'Times New Roman',serif;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  --t-2:0.8rem; --t-1:0.875rem; --t0:1rem; --t1:1.125rem; --t2:1.406rem; --t3:1.758rem; --t4:2.197rem;
  --s1:4px; --s2:8px; --s3:12px; --s4:16px; --s5:24px; --s6:32px; --s7:48px; --s8:64px; --s9:96px;
  --r:10px; --r-sm:6px;
  --shadow:0 1px 2px rgb(20 25 22 / .05), 0 12px 28px -18px rgb(20 25 22 / .22);
  --dur:300ms; --ease:cubic-bezier(.2,.7,.2,1); --focus:var(--accent);
  --w:1140px; --measure:68ch; --track-label:.08em;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --bg:#121614; --panel:#191E1B; --band:#161A18;
  --ink:#E8EBE7; --ink-2:#C9CFC9; --muted:#AEB6B0; --faint:#98A09A;
  --line:#28302B; --line-2:#3A443E; --line-3:#727E77;
  --accent:#E08A7A; --accent-ink:#1A0F0C; --accent-soft:#3A1E1A;
  --amber:#D99A4E; --amber-soft:#3A2A14;
  --ok:#43C29A; --ok-soft:#17382C;
  --shadow:0 1px 2px rgb(0 0 0 / .4), 0 12px 28px -18px rgb(0 0 0 / .65);
}}
:root[data-theme="dark"]{ /* same values as the dark block above */ }
@media (prefers-reduced-motion: reduce){ :root{ --dur:0.01ms; } }
```

## 6. Checks

| Check | How |
|---|---|
| Contrast, both themes | `uv run --no-project python ../nmp-dsci.github.io/scripts/contrast_audit.py` against the tokens block (same values, same result) |
| Layout at 390 / 900 / 1280px, light and dark | screenshot the artifact before serving it; Lavish's layout warnings are fixed before the human sees the page |
| Headings | every `h2` reads `NN · label — claim`, claim ≤ 10 words; one `<em>` on the page, in the `h1` |
| Numbers | every number in prose has its baseline or denominator in the same sentence |
