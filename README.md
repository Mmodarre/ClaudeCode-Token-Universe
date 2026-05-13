# cc_universe — `claude_usage.py`

Generate a local, single-file HTML dashboard from your `~/.claude/` folder
showing token usage, tool-call patterns, MCP server activity, and an
AI-clustered word cloud of your WebSearch research themes.

The whole product is one Python script — `claude_usage.py` — runnable directly
with [uv](https://docs.astral.sh/uv/) so dependencies are resolved
automatically. No `pip install`, no virtualenv.

## Requirements

- `uv` installed
  ([install](https://docs.astral.sh/uv/getting-started/installation/))
- A `.claude/` folder produced by Claude Code (defaults to `~/.claude`)
- **Optional:** the `claude` CLI on `PATH` if you want the AI-clustered word
  cloud of your search queries. Pass `--no-ai` to skip this step entirely.

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
- **Without `--no-ai`,** your WebSearch query strings are sent to Anthropic
  via the local `claude` CLI in four parallel `claude -p` calls (model:
  `haiku`). Queries are clustered into themes; raw queries do not appear in
  the rendered HTML.
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

- ✅ Vector text — sharp at any zoom, much smaller files
- ✅ Charts render at their final values (no animation freeze, no clipped bars)
- ✅ Page breaks respect card boundaries
- ✅ No CDN dependency for PDF generation
- ⚠️ Requires one extra click in the print dialog ("Save as PDF")

For programmatic / scripted PDF generation, you can also do:

```bash
chrome --headless=new --no-pdf-header-footer --print-to-pdf=out.pdf \
  "file:///path/to/claude_usage_dashboard.html"
```

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
- File-edit / file-read frequency by language
- `--watch` mode / auto-refresh
- Cost figures from the Anthropic Admin API
