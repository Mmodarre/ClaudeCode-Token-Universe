# Claude Code Token Universe

> A local, single-file HTML dashboard for [Claude Code](https://claude.com/claude-code) usage and **estimated cost** — token totals across all classes (input, output, cache reads, cache writes), per-model **USD spend** at list API prices, **live current-day data** (the built-in `/usage` cache only refreshes through yesterday), per-model breakdowns, MCP server activity, web research history, programming-language frequency, and an AI-clustered word cloud of your search themes.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Runs with uv](https://img.shields.io/badge/runs%20with-uv-DE5FE9.svg)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Local-first](https://img.shields.io/badge/local--first-yes-success.svg)](#privacy)
[![No install](https://img.shields.io/badge/install-not%20required-brightgreen.svg)](#quick-start)

![Claude Code usage dashboard — token universe view](docs/dashboard-preview.png)

The whole product is one Python script — `claude_usage.py` — runnable directly
with [uv](https://docs.astral.sh/uv/) so dependencies are resolved
automatically. No `pip install`, no virtualenv, no server, no account.

---

## Why this exists

Anthropic's `/usage` command in Claude Code reports raw input + output tokens,
but does not surface **cache reads and cache writes** — which on a heavy Opus
session are typically 5-50× larger than the headline figure. This tool reads
the local `~/.claude/stats-cache.json` and your transcript JSONL files, then
renders the **full picture** as a single self-contained HTML file you can
open, share, or export to PDF.

It is useful if you want to:

- **Estimate Claude Code cost in USD** at Anthropic's list API prices — per model, per day, per month, and as a single all-time total — without waiting for an invoice from the Admin API.
- **See how much you've actually spent in tokens** across all classes (input, output, cache read, cache create) — not just the partial figure shown in `/usage`.
- **See today's usage**, not just data through yesterday. The built-in `/usage` cache is stale by design; this tool walks your local JSONL transcripts to extend the picture through the current moment.
- **Compare Claude Code model usage** (Opus 4.5 / 4.6 / 4.7, Sonnet 4.5 / 4.6, Haiku 4.5) and see which sessions burned which model.
- **Audit MCP server activity** — which tools your agents called, how often, on which days.
- **Review your web research history** — top domains visited via WebFetch, top keywords searched via WebSearch, themes clustered via Claude Haiku.
- **Catch heavy-language drift** — which file types Claude reads and edits most often, by extension.
- **Produce a shareable PDF** for retros, blog posts, or expense reports — without uploading anything to a third-party.

## Features

- **Zero install.** Single `uv run claude_usage.py ~/.claude` produces `claude_usage_dashboard.html` in the working directory.
- **Token universe view.** All four token classes (input, output, cache read, cache create) per model, with daily and monthly aggregates.
- **Estimated cost.** USD cost at list API prices (Opus / Sonnet / Haiku, version-aware) on a dedicated KPI card and as tooltips on the daily / monthly charts, plus a $ overlay line on the monthly chart. Prices are pulled from [Anthropic's public pricing page](https://platform.claude.com/docs/en/about-claude/pricing) and date-stamped; edit the constants table to refresh.
- **MCP server analytics.** Per-server tool-call counts, distinct tools used, and daily activity charts. Auto-detects top-N servers or accepts an explicit list.
- **Web research map.** Top WebFetch domains, top WebSearch keywords, daily research volume, and AI-clustered themes via 4 parallel `claude -p --model haiku` calls.
- **Programming-language frequency.** Bar chart of reads / edits by language, derived from file extensions in your transcript JSONLs.

  ![Reads vs edits by programming language](docs/languages-preview.png)
- **PDF export that actually works.** Uses the browser's native print engine (Chrome / Edge / Safari / Firefox) with print CSS tuned for A4 — vector text, sharp at any zoom, no clipped charts.
- **Privacy-first by design.** All parsing is local. With `--no-ai`, nothing leaves your machine.

## Quick start

```bash
# Default — generates ./claude_usage_dashboard.html
uv run claude_usage.py ~/.claude

# Pick the output path and open it in your browser when done
uv run claude_usage.py ~/.claude --output /tmp/me.html --open

# Skip the AI step (fully offline)
uv run claude_usage.py ~/.claude --no-ai

# Pick which MCP servers to show (default: top 3 by call count)
uv run claude_usage.py ~/.claude --mcp-servers databricks,serena

# Override the header title
uv run claude_usage.py ~/.claude --title "My team — Claude Code usage"
```

The final line of stdout is the path to the generated HTML, so you can pipe it:

```bash
dashboard=$(uv run claude_usage.py ~/.claude)
open "$dashboard"
```

> ⚠️ **Heads-up about the default (AI) mode.** Without `--no-ai`, the script
> calls the local `claude` CLI repeatedly: four parallel Haiku calls to cluster
> WebSearch themes, one Haiku call to discover topic categories, and **one
> Haiku call per session** to classify it. With a `.claude/` folder that holds
> a few thousand sessions, that is a few thousand Haiku calls.
>
> Haiku is cheap per call, but the call count adds up: expect **tens of
> millions of input tokens** end-to-end on a heavy account, and a runtime of
> **several minutes to tens of minutes** depending on how many sessions you
> have and your Claude Code rate limits. The tokens count against your
> Anthropic usage.
>
> If you just want the token universe / model mix / MCP / language charts,
> use `--no-ai` — it skips the theme cloud and the Topics section, runs in
> ~2-3 seconds, and costs nothing.

## Requirements

- `uv` installed
  ([install](https://docs.astral.sh/uv/getting-started/installation/))
- A `.claude/` folder produced by Claude Code (defaults to `~/.claude`)
- **Optional:** the `claude` CLI on `PATH` if you want the AI-clustered word
  cloud of your search queries. Pass `--no-ai` to skip this step entirely.

## What it reads

| Path                                  | Required? | Used for                                                                      |
| ------------------------------------- | --------- | ----------------------------------------------------------------------------- |
| `<root>/stats-cache.json`             | yes       | KPIs, per-model token totals, daily I+O, hour-of-day counts                   |
| `<root>/projects/**/*.jsonl`          | yes       | Tool calls (MCP servers, WebFetch, WebSearch, research-expert subagent calls) |

Everything else under `.claude/` is ignored. The script never writes to your
`.claude/` folder.

## Privacy

- **All parsing is local.** No data is uploaded.
- **With `--no-ai`, nothing leaves your machine.** The script reads files and
  writes one HTML file. That's it.
- **Without `--no-ai`,** the script invokes the local `claude` CLI many times
  (model: `haiku`): four parallel calls to cluster WebSearch queries into
  themes, one call to discover topic categories, and one call per session
  to classify it. On accounts with thousands of sessions this is **thousands
  of Haiku calls** sending the first user prompt of each session to
  Anthropic. Raw queries and prompts are not written into the rendered HTML.
  See the heads-up under [Quick start](#quick-start) for runtime and cost
  expectations.
- **The generated HTML loads three CDN scripts at view time**: Chart.js, d3,
  and d3-cloud, all from `jsdelivr.net`. The HTML itself does not phone home.

## Exporting to PDF

The dashboard has an **Export PDF** button in the top-right of the header.
Clicking it opens your browser's print dialog with the layout already tuned
for A4 (charts shrink to fit, cards don't split across pages, the button
itself is hidden). Choose "Save as PDF" in the destination dropdown and the
PDF lands on disk.

This uses the browser's native print engine (Chrome, Edge, Safari, Firefox)
rather than a JS-based renderer like html2pdf. Trade-offs:

- Vector text — sharp at any zoom, much smaller files
- Charts render at their final values (no animation freeze, no clipped bars)
- Page breaks respect card boundaries
- No CDN dependency for PDF generation
- Requires one extra click in the print dialog ("Save as PDF")

For programmatic / scripted PDF generation, you can also do:

```bash
chrome --headless=new --no-pdf-header-footer --print-to-pdf=out.pdf \
  "file:///path/to/claude_usage_dashboard.html"
```

## FAQ

### How do I estimate my Claude Code cost in USD?

Run `uv run claude_usage.py ~/.claude`. The "Estimated cost" KPI card shows
total USD at list API prices (per-class, per-model). The daily and monthly
charts include $ cost in their tooltips, and the monthly chart adds a $
overlay line on a secondary axis. See [How accurate is the "Estimated
cost" figure?](#how-accurate-is-the-estimated-cost-figure) for the caveats.

### How do I see Claude Code token usage including cache reads?

Run `uv run claude_usage.py ~/.claude`. The KPI cards at the top show the
full total across all four token classes; the "Per-model token classes" bar
chart breaks them out by model.

### Why does my Claude Code `/usage` show different numbers from this dashboard?

`/usage` only refreshes its on-disk cache when you open it in the **All**
tab and keep it open. That cache also stops at *yesterday* — today's usage
is never persisted. This dashboard reads the same cache, then walks your
local JSONL transcripts to add a live delta through the current day, so
totals are consistently higher and more current than `/usage`.

### Does this work without internet access?

Generation works fully offline if you pass `--no-ai`. Viewing the generated
HTML requires internet for the three CDN-loaded chart libraries on first
view (they cache after that). A future flag will vendor them inline.

### Does it work with Claude Code on Windows / Linux / WSL?

Yes. Paths use `pathlib`, `webbrowser.open` is cross-platform, and the
generated HTML opens in any modern browser.

### Where is `stats-cache.json` if it isn't in `~/.claude`?

Pass the actual path as the first argument: `uv run claude_usage.py /path/to/your/.claude`.
The script accepts any directory containing a `stats-cache.json` plus a
`projects/` folder.

### How accurate is the "Estimated cost" figure?

The KPI card and per-model totals are exact at list API prices: they multiply
each model's per-class token counts (input, output, cache read, cache write)
by the rates in [Anthropic's pricing page](https://platform.claude.com/docs/en/about-claude/pricing).
Two systematic gaps to be aware of:

1. **Cache-write TTL.** The cache stores only one `cache_creation_input_tokens`
   number per model; it does not distinguish the 5-minute and 1-hour TTLs.
   The dashboard assumes 5-minute (Claude Code's default). If you opted into
   1-hour caches, real cache-write spend is ≈60% higher per token (2× input
   vs. 1.25× input).
2. **Per-day and per-month allocation.** The cache only records daily *input +
   output* tokens per model (no per-day cache split). To attribute cost to a
   day, the dashboard uses each model's all-time `$/IO-token` rate; the cache
   portion is therefore pro-rated across days rather than counted on the
   day it was actually written. Sum-of-daily ≈ total.

Enterprise discounts, Batch-API rebates, and any pricing private offer are
not reflected — these are list-price estimates. For invoice-accurate
numbers, use the Anthropic Admin API
(`/v1/organizations/usage_report/claude_code`).

To update prices when Anthropic changes them, edit `MODEL_PRICING` and
`PRICING_FETCHED_ON` near the top of `claude_usage.py`.

### Is this affiliated with Anthropic?

No. It is a third-party tool. See the disclaimer below.

### Can I use it for a team?

The dashboard is per-user. To produce a team view, generate one HTML per
member and concatenate the KPIs by hand — multi-user aggregation is not in
scope for v1.

## Disclaimer

Third-party tool. Not affiliated with or endorsed by Anthropic. The dashboard
mirrors Anthropic's published design language (Inter/Tiempos typography,
cream/coral palette) and includes the "Anthropic / Claude Code · Usage"
header used in their official Claude Code surface so the report feels like
the dashboards users already recognize. If you intend to redistribute output
publicly, replace the header text via `--title "Your name · Claude usage"`.

## Out of scope (for now)

- Offline HTML (vendoring Chart.js / d3 / d3-cloud inline)
- Per-project cost breakdown from `cost_cache.json`
- `--watch` mode / auto-refresh
- Reconciled cost figures from the Anthropic Admin API (the dashboard's $ values are list-price estimates, not billed amounts — they ignore enterprise discounts, batch-API rebates, and the 1-hour cache-write TTL premium)
- Multi-user team aggregation

## License

MIT — see [`LICENSE`](LICENSE).

## Keywords

Claude · Claude Code · Anthropic · Claude Code usage · Claude Code dashboard ·
Claude Code cost · Claude Code spend · estimate Claude API cost · LLM cost tracking ·
Claude Code stats · Claude Code analytics · token usage · cache token cost ·
LLM analytics · LLM observability · MCP · MCP server · Model Context Protocol ·
WebSearch · WebFetch · self-hosted · privacy-first · local-first ·
uv · Python · Chart.js · d3 · Opus 4.7 cost · Sonnet 4.6 cost · Haiku 4.5 cost
