#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "jinja2>=3.1",
# ]
# ///
"""
claude_usage.py — Portable Claude Code usage dashboard.

Reads a local ``.claude/`` folder and emits a single self-contained HTML
dashboard with token usage, MCP server activity, web-research patterns, and an
optional AI-clustered word cloud of WebSearch queries.

Run with `uv` so deps are resolved automatically:

    uv run claude_usage.py ~/.claude
    uv run claude_usage.py ~/.claude --output /tmp/report.html --open
    uv run claude_usage.py ~/.claude --no-ai
    uv run claude_usage.py ~/.claude --mcp-servers databricks,serena

Third-party tool. Not affiliated with or endorsed by Anthropic.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.parse
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Template


# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# Model display names — extend here when new models appear. Anything not in
# this map falls back to the raw model id, which still renders fine but loses
# the friendly label.
MODEL_SHORT: dict[str, str] = {
    "claude-opus-4-7": "Opus 4.7",
    "claude-opus-4-6": "Opus 4.6",
    "claude-opus-4-5-20251101": "Opus 4.5",
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-sonnet-4-5-20250929": "Sonnet 4.5",
}

# Canonicalization rules for the AI-extracted theme list. Each entry is
# (canonical_name, [substring_patterns]) — case-insensitive. The first matching
# rule wins; anything that matches nothing passes through unchanged.
# Heavily Databricks-flavored — extend per audience.
THEME_CANON: list[tuple[str, list[str]]] = [
    ("Delta Live Tables / Lakeflow", [
        "dlt", "delta live table", "lakeflow", "apply_changes", "auto loader",
        "autoloader", "@dlt", "streaming table", "materialized view",
    ]),
    ("Databricks Apps (OBO auth)", [
        "databricks app", "obo ", "on-behalf-of", "on behalf of",
    ]),
    ("Databricks Model Serving", [
        "model serving", "serving endpoint", "serving-endpoint",
    ]),
    ("Databricks Asset Bundles", [
        "asset bundle", "databricks bundle", "dab ", "dabs ",
    ]),
    ("Databricks metric views", [
        "metric view", "measure(", "uc metric",
    ]),
    ("Databricks VARIANT type", [
        "variant ", "variant type", "variant streaming",
    ]),
    ("Delta Lake Change Data Feed", [
        "change data feed", "cdf ", "readchangefeed",
    ]),
    ("Databricks Repos / Git API", [
        "repos api", "databricks repos", "git api",
    ]),
    ("Databricks structured streaming", [
        "structured streaming", "spark streaming", "foreachbatch",
    ]),
    ("Databricks pipeline frameworks", [
        "dbt vs", "vs dbt", "pipeline framework",
    ]),
    ("OpenCode CLI", [
        "opencode",
    ]),
    ("Monaco editor (web IDE)", [
        "monaco-editor", "monaco editor",
    ]),
    ("Git OAuth & credentials", [
        "github oauth", "git oauth", "git credential", "azure devops oauth",
        "gho_", "oauth app",
    ]),
    ("YAML / config-as-code", [
        "yaml template", "yaml config", "config-as-code", "jinja2", "jinja ",
    ]),
    ("uv / Python tooling", [
        "uv python", "uv package", "astral uv", "pyproject.toml",
    ]),
    ("Documentation platforms & frameworks", [
        "readthedocs", "sphinx", "mkdocs", "docusaurus", "gitbook", "netlify",
        "vercel", "cloudflare pages",
    ]),
    ("Documentation structure best practices", [
        "diataxis", "documentation best practice", "docs structure",
    ]),
    ("MCP servers", [
        "mcp server", "serena mcp", "playwright mcp",
    ]),
    ("Anthropic tooling", [
        "claude code settings", "claude_code_", "anthropic_auth", "anthropic admin",
        "anthropic sdk",
    ]),
    ("Lakehouse Plumber", [
        "lakehouse plumber", "lhp ", "lhp.yaml",
    ]),
    ("Gmail / email", [
        "gmail", "imap", "himalaya email",
    ]),
    ("Secrets / encryption", [
        "secret scope", "fernet", "aes encryption", "token storage",
    ]),
    ("Workspace / multi-tenant", [
        "multi-tenant", "per-user", "concurrent editing",
    ]),
    ("Pipeline testing", [
        "pytest", "unit test", "dbt unit test", "dry-run", "dry run pipeline",
    ]),
    ("DLT data quality & quarantine", [
        "data quality", "dqx", "apply_checks", "quarantine",
    ]),
]


# --------------------------------------------------------------------------- #
# Stats cache                                                                 #
# --------------------------------------------------------------------------- #

def parse_stats_cache(claude_dir: Path) -> dict[str, Any]:
    """Read ``stats-cache.json`` and shape it for the dashboard.

    The cache holds aggregates Claude Code maintains itself: per-model token
    totals, daily activity, hour-of-day counts. We pass most of it through
    verbatim and derive a ``monthlyIO`` rollup from ``dailyModelTokens``.
    """
    cache_path = claude_dir / "stats-cache.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Expected {cache_path}; is this a .claude folder? "
            f"Pass the path to a Claude Code state directory."
        )
    raw = json.loads(cache_path.read_text())

    monthly: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for d in raw.get("dailyModelTokens", []):
        month = d["date"][:7]
        for model, tokens in d.get("tokensByModel", {}).items():
            monthly[month][model] += tokens

    return {
        "firstSessionDate": raw.get("firstSessionDate", ""),
        "lastComputedDate": raw.get("lastComputedDate", ""),
        "totalSessions": raw.get("totalSessions", 0),
        "totalMessages": raw.get("totalMessages", 0),
        "longestSession": raw.get("longestSession", {}),
        "modelUsage": raw.get("modelUsage", {}),
        "dailyActivity": raw.get("dailyActivity", []),
        "dailyModelTokens": raw.get("dailyModelTokens", []),
        "hourCounts": raw.get("hourCounts", {}),
        "monthlyIO": {m: dict(v) for m, v in monthly.items()},
    }


# --------------------------------------------------------------------------- #
# Transcript walk                                                             #
# --------------------------------------------------------------------------- #

_MCP_TOOL_RE = re.compile(r"^mcp__([a-zA-Z0-9_\-]+)__(.+)$")


def _iter_tool_uses(claude_dir: Path):
    """Yield ``(date, tool_name, tool_input)`` for every unique tool_use.

    Claude Code writes assistant turns as a stream of delta rows: each row
    carries exactly one block of a logical message and all rows share the
    same ``message.id``. So deduping by message id would silently drop most
    blocks. We dedupe at the block level instead, by ``block.id`` (the
    ``toolu_*`` identifier the API assigns to every tool_use), which is
    globally unique and survives resumed/branched sessions.
    """
    seen_block_ids: set[str] = set()
    projects = claude_dir / "projects"
    if not projects.exists():
        return
    for jsonl in projects.glob("**/*.jsonl"):
        try:
            with jsonl.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("type") != "assistant":
                        continue
                    msg = rec.get("message") or {}
                    ts = rec.get("timestamp") or msg.get("timestamp") or ""
                    date = ts[:10] if isinstance(ts, str) else ""
                    for block in msg.get("content") or []:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_use":
                            continue
                        bid = block.get("id")
                        if bid:
                            if bid in seen_block_ids:
                                continue
                            seen_block_ids.add(bid)
                        yield date, block.get("name") or "", block.get("input") or {}
        except OSError:
            continue


_LANG_BY_EXT: dict[str, str] = {
    ".py": "Python",
    ".yaml": "YAML", ".yml": "YAML",
    ".sql": "SQL",
    ".md": "Markdown", ".mdx": "Markdown",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".rst": "RST",
    ".json": "JSON", ".jsonl": "JSON",
    ".txt": "Plain text",
    ".toml": "TOML",
    ".cfg": "Config", ".ini": "Config", ".conf": "Config", ".properties": "Config",
    ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".html": "HTML", ".htm": "HTML",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell", ".fish": "Shell",
    ".rs": "Rust",
    ".css": "CSS", ".scss": "CSS", ".sass": "CSS",
    ".csv": "CSV", ".tsv": "CSV",
    ".png": "Image", ".jpg": "Image", ".jpeg": "Image", ".gif": "Image",
    ".svg": "Image", ".webp": "Image", ".bmp": "Image", ".ico": "Image",
    ".ipynb": "Python",
    ".go": "Go",
    ".java": "Java",
    ".c": "C", ".h": "C",
    ".cpp": "C++", ".cc": "C++", ".hpp": "C++",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".lua": "Lua",
    ".r": "R",
    ".scala": "Scala",
}


def _lang_for(path: str) -> str:
    """Classify a file path as a language label, mirroring the reference dashboard."""
    if not isinstance(path, str) or not path:
        return "Other / ?"
    name = path.rsplit("/", 1)[-1]
    # Special cases that aren't extension-based
    if name == "Dockerfile" or name.startswith("Dockerfile."):
        return "Dockerfile"
    if name == "Makefile" or name.endswith(".mk"):
        return "Makefile"
    dot = name.rfind(".")
    if dot < 0:
        return "Plain text" if "." not in name else "Other / ?"
    ext = name[dot:].lower()
    return _LANG_BY_EXT.get(ext, "Other / ?")


def walk_transcripts(
    claude_dir: Path,
    mcp_servers_filter: list[str] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any], list[str], dict[str, Any], int]:
    """Single pass over ``projects/**/*.jsonl``.

    Returns:
      mcp_block        — per-server stats for the chosen MCP server (None if no
                         MCP usage). Single block, matching the reference's
                         single-section layout. If --mcp-servers is given, the
                         first matching server wins; otherwise top-1 by calls.
      research         — WebFetch / WebSearch / research-expert aggregates
      queries          — every WebSearch query string in order (input to AI)
      langs            — Read/Edit/Write/NotebookEdit/MultiEdit by file ext
      total_tool_calls — denominator for "% of all tool calls" KPIs
    """
    mcp_by_server: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {"tools": Counter(), "daily": Counter()}
    )

    webfetch_total = 0
    websearch_total = 0
    research_total = 0
    webfetch_by_day: Counter = Counter()
    websearch_by_day: Counter = Counter()
    research_by_day: Counter = Counter()
    domain_counts: Counter = Counter()
    queries: list[str] = []
    keyword_counter: Counter = Counter()
    total_tool_calls = 0

    # Language tracking: per-language reads/edits + unique paths sets.
    lang_reads: Counter = Counter()
    lang_edits: Counter = Counter()
    lang_unique_read: dict[str, set[str]] = defaultdict(set)
    lang_unique_edit: dict[str, set[str]] = defaultdict(set)

    READ_TOOLS = {"Read"}
    EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}

    for date, name, inp in _iter_tool_uses(claude_dir):
        total_tool_calls += 1

        m = _MCP_TOOL_RE.match(name)
        if m:
            server, tool = m.group(1), m.group(2)
            mcp_by_server[server]["tools"][tool] += 1
            if date:
                mcp_by_server[server]["daily"][date] += 1
            continue

        if name == "WebFetch":
            webfetch_total += 1
            if date:
                webfetch_by_day[date] += 1
            url = inp.get("url") or ""
            if isinstance(url, str) and url:
                try:
                    netloc = urllib.parse.urlparse(url).netloc.lower()
                    if netloc.startswith("www."):
                        netloc = netloc[4:]
                    if netloc:
                        domain_counts[netloc] += 1
                except ValueError:
                    pass
        elif name == "WebSearch":
            websearch_total += 1
            if date:
                websearch_by_day[date] += 1
            q = inp.get("query") or ""
            if isinstance(q, str) and q.strip():
                queries.append(q.strip())
                for word in re.findall(r"[A-Za-z][A-Za-z0-9_+]{2,}", q.lower()):
                    if word not in _STOPWORDS:
                        keyword_counter[word] += 1
        elif name in ("Task", "Agent"):
            if (inp.get("subagent_type") or "") == "research-expert":
                research_total += 1
                if date:
                    research_by_day[date] += 1
        elif name in READ_TOOLS:
            path = inp.get("file_path") or inp.get("notebook_path") or inp.get("path") or ""
            if path:
                lang = _lang_for(path)
                lang_reads[lang] += 1
                lang_unique_read[lang].add(path)
        elif name in EDIT_TOOLS:
            path = inp.get("file_path") or inp.get("notebook_path") or inp.get("path") or ""
            if path:
                lang = _lang_for(path)
                lang_edits[lang] += 1
                lang_unique_edit[lang].add(path)

    # --- MCP filtering / top-1 ----------------------------------------------
    mcp_block: dict[str, Any] | None = None
    if mcp_by_server:
        if mcp_servers_filter:
            wanted = [s.strip() for s in mcp_servers_filter if s.strip()]
            chosen = next((s for s in wanted if s in mcp_by_server), None)
        else:
            chosen = max(
                mcp_by_server.items(),
                key=lambda kv: sum(kv[1]["tools"].values()),
            )[0]
        if chosen:
            info = mcp_by_server[chosen]
            tool_counts = dict(info["tools"])
            daily = dict(info["daily"])
            mcp_block = {
                "name": chosen,
                "displayName": chosen[:1].upper() + chosen[1:],
                "totalCalls": sum(tool_counts.values()),
                "distinctTools": len(tool_counts),
                "toolCounts": tool_counts,
                "dailyTotal": daily,
            }

    research = {
        "webfetchTotal": webfetch_total,
        "websearchTotal": websearch_total,
        "researchTotal": research_total,
        "uniqueDomains": len(domain_counts),
        "uniqueQueries": len(set(queries)),
        "domainCounts": dict(domain_counts.most_common(20)),
        "queryKeywords": dict(keyword_counter.most_common(20)),
        "webfetchByDay": dict(webfetch_by_day),
        "websearchByDay": dict(websearch_by_day),
        "researchByDay": dict(research_by_day),
    }

    # Build LANGS payload — sorted by total activity (reads + edits) descending.
    all_langs = set(lang_reads) | set(lang_edits)
    rows = []
    for lang in all_langs:
        rows.append({
            "lang": lang,
            "reads": lang_reads.get(lang, 0),
            "edits": lang_edits.get(lang, 0),
            "uniqRead": len(lang_unique_read.get(lang, ())),
            "uniqEdit": len(lang_unique_edit.get(lang, ())),
        })
    rows.sort(key=lambda r: r["reads"] + r["edits"], reverse=True)

    langs = {
        "totalReads": sum(lang_reads.values()),
        "totalWrites": sum(lang_edits.values()),
        "uniqueRead": sum(len(s) for s in lang_unique_read.values()),
        "uniqueWrite": sum(len(s) for s in lang_unique_edit.values()),
        "rows": rows,
    }

    return mcp_block, research, queries, langs, total_tool_calls


# Small stopword set for keyword extraction — only used as a back-up signal
# in the research payload; the real "what you researched" view is the AI
# theme cloud. Kept short on purpose.
_STOPWORDS: set[str] = {
    "the", "and", "for", "with", "from", "this", "that", "are", "how", "what",
    "when", "where", "which", "who", "why", "can", "use", "using", "into",
    "via", "not", "but", "all", "any", "you", "your", "out", "get", "got",
    "off", "set", "new", "old", "between", "after", "before", "than", "then",
    "also", "have", "has", "had", "will", "would", "could", "should", "make",
    "made", "does", "did", "doing", "been", "being", "about", "over", "under",
    "without", "within", "still", "just", "only", "more", "most", "some",
    "such", "very", "much", "many", "few", "own", "way", "two", "one", "three",
}


# --------------------------------------------------------------------------- #
# AI theme extraction                                                         #
# --------------------------------------------------------------------------- #

_AI_PROMPT_HEADER = textwrap.dedent("""\
    You will be given a list of web search queries. Cluster them into SPECIFIC
    NOUN-PHRASE THEMES suitable for a word cloud.

    Output JSON ONLY. No prose. No markdown fences. Schema:
      {"themes": [{"text": "<theme name>", "weight": <int>, "examples": ["q1","q2","q3"]}]}

    Rules:
      - Theme names MUST be specific noun phrases. Good: "Delta Live Tables /
        Lakeflow", "GitHub OAuth", "Monaco editor", "Databricks Apps".
        Bad: "documentation", "errors", "apis", "configuration", "best practices".
      - Each query belongs to exactly one theme.
      - weight = number of queries in that theme.
      - Include 2-5 representative example query strings, verbatim, per theme.
      - Drop one-off queries that do not cluster with at least one other.
      - Aim for 8-20 themes total per chunk.

    Queries:
    """)


def _extract_chunk(
    chunk: list[str], claude_bin: str, timeout: int = 180
) -> list[dict[str, Any]]:
    """Run a single ``claude -p`` call against one query chunk."""
    prompt = _AI_PROMPT_HEADER + "\n".join(f"- {q}" for q in chunk)
    try:
        result = subprocess.run(
            [claude_bin, "-p", prompt, "--model", "haiku", "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        print(f"  [warn] claude CLI chunk failed: {exc}", file=sys.stderr)
        return []
    out = (result.stdout or "").strip()
    if not out:
        return []
    # Strip ```json ... ``` fences the model sometimes adds despite instructions.
    if out.startswith("```"):
        out = re.sub(r"^```(?:json)?\s*", "", out)
        out = re.sub(r"\s*```\s*$", "", out)
    # Find the JSON object — it might have leading prose despite instructions.
    brace = out.find("{")
    if brace > 0:
        out = out[brace:]
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError as exc:
        print(f"  [warn] chunk JSON parse failed: {exc}", file=sys.stderr)
        return []
    themes = parsed.get("themes")
    if not isinstance(themes, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for t in themes:
        if not isinstance(t, dict):
            continue
        text = (t.get("text") or "").strip()
        try:
            weight = int(t.get("weight") or 0)
        except (TypeError, ValueError):
            weight = 0
        examples = [e for e in (t.get("examples") or []) if isinstance(e, str)]
        if text and weight > 0:
            cleaned.append({"text": text, "weight": weight, "examples": examples[:5]})
    return cleaned


def extract_themes(
    queries: list[str],
    claude_bin: str | None,
    chunks: int = 4,
) -> list[dict[str, Any]] | None:
    """Cluster WebSearch queries via 4 parallel ``claude -p`` calls.

    Returns ``None`` if no ``claude`` binary is available or every chunk failed
    — the caller then skips the word-cloud section. A subset of failed chunks
    is tolerated (we keep whatever themes came back successfully).
    """
    if not claude_bin or not queries:
        return None

    # Split into ``chunks`` roughly-equal pieces. We dedupe to keep the prompt
    # short — a query that ran 12 times still gets a single line.
    unique = sorted(set(queries))
    if not unique:
        return None
    n = max(1, min(chunks, len(unique)))
    size = (len(unique) + n - 1) // n
    pieces = [unique[i : i + size] for i in range(0, len(unique), size)]
    print(
        f"  Running {len(pieces)} parallel `claude -p` calls "
        f"over {len(unique)} unique queries…",
        file=sys.stderr,
    )

    all_themes: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=min(4, len(pieces))) as ex:
        futures = [ex.submit(_extract_chunk, p, claude_bin) for p in pieces]
        for fut in cf.as_completed(futures):
            all_themes.extend(fut.result())

    if not all_themes:
        return None
    return all_themes


def merge_themes(themes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonicalize and merge per-chunk themes into a single ranked list."""
    bucket: dict[str, dict[str, Any]] = {}

    def canonicalize(label: str) -> str:
        low = label.lower()
        for canon, patterns in THEME_CANON:
            if any(p in low for p in patterns):
                return canon
        return label

    for t in themes:
        key = canonicalize(t["text"])
        slot = bucket.setdefault(key, {"text": key, "weight": 0, "examples": []})
        slot["weight"] += t["weight"]
        for ex in t["examples"]:
            if ex not in slot["examples"] and len(slot["examples"]) < 5:
                slot["examples"].append(ex)

    merged = sorted(bucket.values(), key=lambda r: r["weight"], reverse=True)
    return merged


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #

def render_html(
    *,
    title: str,
    stats: dict[str, Any],
    mcp_block: dict[str, Any] | None,
    research: dict[str, Any],
    wordcloud: list[dict[str, Any]] | None,
    langs: dict[str, Any],
    total_tool_calls: int,
    generated_at: str,
) -> str:
    """Render ``HTML_TEMPLATE`` with the prepared payloads.

    The template carries the entire CSS+JS dashboard inline; we only inject
    JSON-serialized payloads and a few scalar strings.
    """
    return Template(HTML_TEMPLATE).render(
        title=title,
        data_json=json.dumps(stats, default=str),
        mcp_json=json.dumps(mcp_block, default=str) if mcp_block else "null",
        mcp_present=mcp_block is not None,
        mcp_server_name=mcp_block["name"] if mcp_block else "",
        mcp_display_name=mcp_block["displayName"] if mcp_block else "",
        research_json=json.dumps(research, default=str),
        wordcloud_json=json.dumps(wordcloud) if wordcloud else "null",
        langs_json=json.dumps(langs, default=str),
        total_tool_calls=total_tool_calls,
        websearch_count=research["websearchTotal"],
        generated_at=generated_at,
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate a local HTML usage dashboard from a .claude folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("claude_dir", type=Path, help="Path to a .claude folder")
    p.add_argument(
        "--output", "-o", type=Path,
        default=Path("claude_usage_dashboard.html"),
        help="Output HTML path (default: ./claude_usage_dashboard.html)",
    )
    p.add_argument(
        "--mcp-servers", type=str, default=None,
        help="Comma-separated MCP server names to render. Default: top 3 by call count.",
    )
    p.add_argument(
        "--no-ai", action="store_true",
        help="Skip the AI-generated word cloud section.",
    )
    p.add_argument(
        "--open", dest="open_browser", action="store_true",
        help="Open the generated HTML in the default browser.",
    )
    p.add_argument(
        "--title", type=str, default="Claude Code · Usage",
        help='Header title (default: "Claude Code · Usage").',
    )
    args = p.parse_args(argv)

    claude_dir = args.claude_dir.expanduser().resolve()
    if not claude_dir.is_dir():
        print(f"error: {claude_dir} is not a directory.", file=sys.stderr)
        return 2

    print(f"Reading stats-cache from {claude_dir}…", file=sys.stderr)
    try:
        stats = parse_stats_cache(claude_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    mcp_filter: list[str] | None = None
    if args.mcp_servers:
        mcp_filter = [s.strip() for s in args.mcp_servers.split(",") if s.strip()]

    projects = claude_dir / "projects"
    if projects.exists():
        transcript_count = sum(1 for _ in projects.glob("**/*.jsonl"))
    else:
        transcript_count = 0
    print(f"Walking {transcript_count} transcripts…", file=sys.stderr)
    mcp_block, research, queries, langs, total_tool_calls = walk_transcripts(
        claude_dir, mcp_filter
    )
    mcp_label = mcp_block["name"] if mcp_block else "none"
    print(
        f"  → {total_tool_calls} unique tool calls; "
        f"MCP server: {mcp_label}; "
        f"{research['websearchTotal']} WebSearch queries; "
        f"{langs['totalReads']} reads, {langs['totalWrites']} edits "
        f"across {len(langs['rows'])} languages.",
        file=sys.stderr,
    )

    wordcloud: list[dict[str, Any]] | None = None
    if args.no_ai:
        print("Skipping AI theme extraction (--no-ai).", file=sys.stderr)
    elif not queries:
        print("No WebSearch queries found — skipping word cloud.", file=sys.stderr)
    else:
        claude_bin = shutil.which("claude")
        if not claude_bin:
            print(
                "warning: `claude` CLI not found on PATH — skipping word cloud. "
                "Install Claude Code or pass --no-ai to silence this warning.",
                file=sys.stderr,
            )
        else:
            print("Extracting themes via claude CLI…", file=sys.stderr)
            raw_themes = extract_themes(queries, claude_bin)
            if raw_themes:
                wordcloud = merge_themes(raw_themes)
                print(f"  → {len(wordcloud)} themes after merge.", file=sys.stderr)
            else:
                print(
                    "warning: AI theme extraction returned nothing — "
                    "skipping word cloud.",
                    file=sys.stderr,
                )

    html = render_html(
        title=args.title,
        stats=stats,
        mcp_block=mcp_block,
        research=research,
        wordcloud=wordcloud,
        langs=langs,
        total_tool_calls=total_tool_calls,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )

    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes).", file=sys.stderr)

    if args.open_browser:
        webbrowser.open(out_path.as_uri())

    print(out_path)
    return 0


# --------------------------------------------------------------------------- #
# HTML template                                                               #
# --------------------------------------------------------------------------- #

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{{ title }}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/d3-cloud@1.2.7/build/d3.layout.cloud.min.js"></script>
  <style>
    :root {
      --bg:        #F0EEE6;
      --surface:   #FAF9F5;
      --card:      #FFFFFF;
      --border:    #E5E1D4;
      --text:      #1F1E1D;
      --text-2:    #5B5A56;
      --text-3:    #8E8C82;
      --coral:     #CC785C;
      --coral-2:   #A55A40;
      --opus-7:    #CC785C;
      --opus-6:    #7A4F37;
      --opus-5:    #5C4838;
      --haiku-5:   #D4A27F;
      --sonnet-6:  #B07F5C;
      --sonnet-5:  #E5C7AF;
      --grid:      #EDE9DC;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); }
    body {
      font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
      font-feature-settings: 'ss01','ss02','cv11';
      -webkit-font-smoothing: antialiased;
      line-height: 1.5;
    }
    .serif {
      font-family: 'Tiempos Headline','Times New Roman', Charter, Georgia, 'Iowan Old Style', serif;
      font-weight: 500;
      letter-spacing: -0.01em;
    }

    /* ==== Top bar ==== */
    .topbar {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 18px 36px;
      display: flex; align-items: center; justify-content: space-between;
    }
    .brand { display: flex; align-items: center; gap: 14px; }
    .brand-name { font-weight: 600; font-size: 15.5px; letter-spacing: -0.01em; }
    .brand-sep  { color: var(--text-3); margin: 0 8px; }
    .brand-page { color: var(--text-2); font-size: 15.5px; }
    .pill {
      font-size: 11.5px; letter-spacing: 0.04em; text-transform: uppercase;
      padding: 4px 10px; border-radius: 999px; border: 1px solid var(--border);
      color: var(--text-2); background: var(--card);
    }

    /* ==== Hero ==== */
    .hero {
      padding: 56px 36px 24px;
      max-width: 1280px; margin: 0 auto;
    }
    .eyebrow { font-size: 12.5px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--coral); font-weight: 600; }
    .hero h1 {
      font-family: 'Tiempos Headline', Charter, Georgia, serif;
      font-size: 44px; line-height: 1.1; font-weight: 500;
      margin: 8px 0 6px; letter-spacing: -0.02em;
    }
    .hero h1 em { font-style: italic; color: var(--coral-2); font-weight: 500; }
    .hero p { margin: 12px 0 0; color: var(--text-2); font-size: 15.5px; max-width: 720px; }

    /* ==== Layout ==== */
    .container { max-width: 1280px; margin: 0 auto; padding: 16px 36px 80px; }
    .grid { display: grid; gap: 16px; }
    .grid.cols-4 { grid-template-columns: repeat(4, 1fr); }
    .grid.cols-2 { grid-template-columns: 1.1fr 0.9fr; }
    @media (max-width: 1020px) {
      .grid.cols-4 { grid-template-columns: repeat(2, 1fr); }
      .grid.cols-2 { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 22px 24px;
    }
    .card h3 { margin: 0 0 4px; font-size: 13.5px; font-weight: 600; color: var(--text-2); letter-spacing: 0.01em; }
    .card .sub { font-size: 12px; color: var(--text-3); margin: 0; }
    .card .big {
      font-family: 'Tiempos Headline', Charter, Georgia, serif;
      font-size: 32px; line-height: 1; font-weight: 500; margin: 12px 0 0;
      letter-spacing: -0.02em;
    }
    .card .big small { font-size: 13px; color: var(--text-3); margin-left: 6px; font-family: 'Inter', system-ui, sans-serif; font-weight: 400; }
    .delta { font-size: 12px; color: var(--text-3); margin-top: 6px; }

    .chart-card { padding: 22px 24px 16px; }
    .chart-head { display:flex; justify-content:space-between; align-items: baseline; margin-bottom: 12px; }
    .chart-title { font-family: 'Tiempos Headline', Charter, Georgia, serif; font-size: 20px; font-weight: 500; letter-spacing: -0.01em; }
    .chart-sub { font-size: 12.5px; color: var(--text-3); }
    .chart-wrap { position: relative; height: 320px; }
    .chart-wrap.tall { height: 380px; }

    .section-title {
      font-family: 'Tiempos Headline', Charter, Georgia, serif;
      font-size: 26px; letter-spacing: -0.015em; font-weight: 500;
      margin: 36px 0 14px;
    }

    /* ==== Heatmap ==== */
    .heat-wrap { overflow-x: auto; padding-bottom: 4px; }
    .heat-grid { display: grid; grid-auto-flow: column; grid-template-rows: repeat(7, 14px); gap: 3px; }
    .heat-cell { width: 14px; height: 14px; border-radius: 3px; background: var(--grid); }
    .heat-cell.l1 { background: #F1D4C0; }
    .heat-cell.l2 { background: #E2AA85; }
    .heat-cell.l3 { background: #CC785C; }
    .heat-cell.l4 { background: #8E4A30; }
    .heat-legend { display:flex; gap:6px; align-items:center; font-size: 11px; color: var(--text-3); margin-top: 12px; }
    .heat-legend span { display: inline-block; width: 14px; height: 14px; border-radius: 3px; }

    /* ==== Table ==== */
    table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
    th { text-align: right; font-weight: 600; color: var(--text-2); padding: 12px 8px; border-bottom: 1px solid var(--border); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
    th:first-child, td:first-child { text-align: left; }
    td { padding: 12px 8px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }
    tr:last-child td { border-bottom: none; }
    .model-chip { display: inline-flex; align-items: center; gap: 8px; }
    .model-chip i { width: 8px; height: 8px; border-radius: 2px; background: var(--coral); }
    tfoot td { font-weight: 600; border-top: 1px solid var(--border); }

    /* ==== Footer note ==== */
    .note {
      padding: 14px 16px; background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; color: var(--text-2); font-size: 13px;
      margin-top: 22px;
    }
    .note strong { color: var(--text); }
    .note .tag {
      display: inline-block; font-size: 10.5px; letter-spacing: 0.08em; text-transform: uppercase;
      color: var(--coral); font-weight: 700; margin-right: 8px;
    }
    .source-pill {
      display: inline-block; padding: 2px 8px; border-radius: 6px;
      background: #F4ECE3; color: var(--coral-2); font-size: 11.5px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    /* ==== Export PDF button ==== */
    .export-btn {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 7px 14px; border-radius: 8px;
      background: var(--text); color: var(--surface);
      border: none; cursor: pointer;
      font: inherit; font-size: 13px; font-weight: 600;
      letter-spacing: 0.01em;
      transition: background 0.15s ease, transform 0.05s ease;
      box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    }
    .export-btn:hover { background: var(--coral-2); }
    .export-btn:active { transform: translateY(1px); }
    .export-btn svg { width: 14px; height: 14px; }

    /* Native browser print (Cmd+P → Save as PDF). The Export PDF button just
       invokes window.print(); these rules are what give the resulting PDF a
       proper page layout (charts shrink, cards don't split, no shadows). */
    @media print {
      .export-btn, .topbar .pill { display: none !important; }
      html, body { background: #FFFFFF; }
      .topbar { border-bottom-color: #D4D0C2; }
      .hero { padding-top: 24px; padding-bottom: 12px; }
      .card { break-inside: avoid; page-break-inside: avoid; box-shadow: none; }
      .section-title { break-after: avoid; page-break-after: avoid; margin-top: 22px; }
      .grid { gap: 10px; }
      .chart-wrap, .chart-wrap.tall { height: 280px; }
      #wordcloud { height: 420px; }
      .footnote { display: none !important; }
      a { color: inherit; text-decoration: none; }
      @page { size: A4; margin: 12mm 10mm; }
    }

    .footnote {
      max-width: 1280px; margin: 24px auto 60px; padding: 0 36px;
      color: var(--text-3); font-size: 12px; line-height: 1.6;
    }
  </style>
</head>
<body>

<div class="topbar">
  <div class="brand">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path d="M12 2.5L14.5 9H20.5L15.7 13L17.5 19.5L12 15.7L6.5 19.5L8.3 13L3.5 9H9.5L12 2.5Z" fill="#1F1E1D"/>
    </svg>
    <div class="brand-name">Anthropic</div>
    <div class="brand-sep">/</div>
    <div class="brand-page">{{ title }}</div>
  </div>
  <div style="display:flex; align-items:center; gap:12px;">
    <div class="pill" id="windowLabel">Window: —</div>
    <button class="export-btn" id="exportPdfBtn" type="button" aria-label="Export to PDF">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M12 3v12"></path>
        <path d="m7 10 5 5 5-5"></path>
        <path d="M5 21h14"></path>
      </svg>
      Export PDF
    </button>
  </div>
</div>

<section class="hero">
  <div class="eyebrow">Local data · stats-cache.json</div>
  <h1>Your token <em>universe</em>, visualized.</h1>
  <p>A dashboard drawn entirely from <span class="source-pill">~/.claude/stats-cache.json</span> and your local
    transcripts. Token totals include input, output, cache reads and cache writes — the full picture, not just
    the input+output figure shown in <span class="source-pill">/usage</span>.</p>
</section>

<main class="container">

  <!-- KPI cards -->
  <div class="grid cols-4">
    <div class="card"><h3>Total tokens</h3><p class="sub">All classes incl. cache</p>
      <div class="big" id="kpiTokens">—</div>
      <div class="delta" id="kpiTokensDelta">—</div>
    </div>
    <div class="card"><h3>Sessions</h3><p class="sub">Distinct Claude Code sessions</p>
      <div class="big" id="kpiSessions">—</div>
      <div class="delta" id="kpiSessionsDelta">—</div>
    </div>
    <div class="card"><h3>Messages</h3><p class="sub">Total messages exchanged</p>
      <div class="big" id="kpiMessages">—</div>
      <div class="delta" id="kpiMessagesDelta">—</div>
    </div>
    <div class="card"><h3>Active days</h3><p class="sub">Days with at least one session</p>
      <div class="big" id="kpiDays">—</div>
      <div class="delta" id="kpiDaysDelta">—</div>
    </div>
  </div>

  <h2 class="section-title">Tokens over time</h2>
  <div class="card chart-card">
    <div class="chart-head">
      <div>
        <div class="chart-title">Daily input + output by model</div>
        <div class="chart-sub">Cache tokens are not stored per-day in this cache — see model breakdown below for the full picture.</div>
      </div>
    </div>
    <div class="chart-wrap tall"><canvas id="dailyChart"></canvas></div>
  </div>

  <h2 class="section-title">Model mix</h2>
  <div class="grid cols-2">
    <div class="card chart-card">
      <div class="chart-head">
        <div>
          <div class="chart-title">Share of total tokens</div>
          <div class="chart-sub">Including cache reads / writes</div>
        </div>
      </div>
      <div class="chart-wrap"><canvas id="donutChart"></canvas></div>
    </div>
    <div class="card chart-card">
      <div class="chart-head">
        <div>
          <div class="chart-title">Per-model token classes</div>
          <div class="chart-sub">Input · Output · Cache read · Cache create</div>
        </div>
      </div>
      <div class="chart-wrap"><canvas id="classChart"></canvas></div>
    </div>
  </div>

  <h2 class="section-title">By month</h2>
  <div class="card chart-card">
    <div class="chart-head">
      <div>
        <div class="chart-title">Monthly input + output tokens, by model</div>
        <div class="chart-sub">Note: this is the only per-month breakdown available in stats-cache (cache tokens are aggregate-only).</div>
      </div>
    </div>
    <div class="chart-wrap"><canvas id="monthlyChart"></canvas></div>
  </div>

  <h2 class="section-title">Activity</h2>
  <div class="grid cols-2">
    <div class="card chart-card">
      <div class="chart-head">
        <div>
          <div class="chart-title">Daily activity heatmap</div>
          <div class="chart-sub">Color intensity = number of sessions on that day</div>
        </div>
      </div>
      <div class="heat-wrap" id="heatmap"></div>
      <div class="heat-legend">
        Fewer
        <span class="heat-cell"></span>
        <span class="heat-cell l1"></span>
        <span class="heat-cell l2"></span>
        <span class="heat-cell l3"></span>
        <span class="heat-cell l4"></span>
        More
      </div>
    </div>
    <div class="card chart-card">
      <div class="chart-head">
        <div>
          <div class="chart-title">Sessions by hour of day</div>
          <div class="chart-sub">Local time, aggregated across the full window</div>
        </div>
      </div>
      <div class="chart-wrap"><canvas id="hourChart"></canvas></div>
    </div>
  </div>

  {% if mcp_present %}
  <h2 class="section-title">{{ mcp_display_name }} MCP</h2>
  <div class="grid cols-4" style="margin-bottom:16px">
    <div class="card"><h3>Total MCP calls</h3><p class="sub">via mcp__{{ mcp_server_name }}__*</p>
      <div class="big" id="dbxTotal">—</div>
      <div class="delta" id="dbxTotalDelta">—</div>
    </div>
    <div class="card"><h3>Distinct tools</h3><p class="sub">Different {{ mcp_server_name }} tools used</p>
      <div class="big" id="dbxDistinct">—</div>
      <div class="delta" id="dbxDistinctDelta">—</div>
    </div>
    <div class="card"><h3>Active days</h3><p class="sub">Days with at least one call</p>
      <div class="big" id="dbxActiveDays">—</div>
      <div class="delta" id="dbxActiveDaysDelta">—</div>
    </div>
    <div class="card"><h3>Peak day</h3><p class="sub">Most calls in a single day</p>
      <div class="big" id="dbxPeak">—</div>
      <div class="delta" id="dbxPeakDelta">—</div>
    </div>
  </div>
  <div class="grid cols-2">
    <div class="card chart-card">
      <div class="chart-head">
        <div>
          <div class="chart-title">Tools, by call count</div>
          <div class="chart-sub">Top tools used on this server</div>
        </div>
      </div>
      <div class="chart-wrap tall"><canvas id="dbxToolsChart"></canvas></div>
    </div>
    <div class="card chart-card">
      <div class="chart-head">
        <div>
          <div class="chart-title">Daily {{ mcp_display_name }} MCP activity</div>
          <div class="chart-sub">Calls per day across the full window</div>
        </div>
      </div>
      <div class="chart-wrap tall"><canvas id="dbxDailyChart"></canvas></div>
    </div>
  </div>
  {% endif %}

  <h2 class="section-title">Web research</h2>
  <div class="grid cols-4" style="margin-bottom:16px">
    <div class="card"><h3>Total research signal</h3><p class="sub">WebFetch + WebSearch + research-expert</p>
      <div class="big" id="resTotal">—</div>
      <div class="delta" id="resTotalDelta">—</div>
    </div>
    <div class="card"><h3>WebFetch</h3><p class="sub">URLs pulled directly</p>
      <div class="big" id="resWF">—</div>
      <div class="delta" id="resWFDelta">—</div>
    </div>
    <div class="card"><h3>WebSearch</h3><p class="sub">Queries to the search tool</p>
      <div class="big" id="resWS">—</div>
      <div class="delta" id="resWSDelta">—</div>
    </div>
    <div class="card"><h3>research-expert</h3><p class="sub">Delegated research tasks</p>
      <div class="big" id="resRE">—</div>
      <div class="delta" id="resREDelta">—</div>
    </div>
  </div>
  <div class="grid cols-2">
    <div class="card chart-card">
      <div class="chart-head">
        <div>
          <div class="chart-title">Top domains visited</div>
          <div class="chart-sub">via WebFetch — where you actually got the answer from</div>
        </div>
      </div>
      <div class="chart-wrap tall"><canvas id="resDomainsChart"></canvas></div>
    </div>
    <div class="card chart-card">
      <div class="chart-head">
        <div>
          <div class="chart-title">Daily research activity</div>
          <div class="chart-sub">Stacked by signal type</div>
        </div>
      </div>
      <div class="chart-wrap tall"><canvas id="resDailyChart"></canvas></div>
    </div>
  </div>

  {% if wordcloud_json != 'null' %}
  <h2 class="section-title">What you researched</h2>
  <div class="card chart-card" style="padding-bottom:24px">
    <div class="chart-head" style="margin-bottom:6px">
      <div>
        <div class="chart-title">Topic word map</div>
        <div class="chart-sub">All {{ websearch_count }} WebSearch queries clustered into themes by 4 parallel Haiku subagents · size = number of queries</div>
      </div>
    </div>
    <div id="wordcloud" style="width:100%; height:520px;"></div>
  </div>
  <div class="card" style="margin-top:16px">
    <table>
      <thead>
        <tr>
          <th>Theme</th>
          <th>Weight</th>
          <th>Example query</th>
        </tr>
      </thead>
      <tbody id="themeTable"></tbody>
    </table>
  </div>
  {% endif %}

  <h2 class="section-title">Programming languages</h2>
  <div class="grid cols-4" style="margin-bottom:16px">
    <div class="card"><h3>Reads</h3><p class="sub">Read tool calls with a file path</p>
      <div class="big" id="langReads">—</div>
      <div class="delta" id="langReadsDelta">—</div>
    </div>
    <div class="card"><h3>Edits</h3><p class="sub">Edit / Write / MultiEdit calls</p>
      <div class="big" id="langEdits">—</div>
      <div class="delta" id="langEditsDelta">—</div>
    </div>
    <div class="card"><h3>Unique files read</h3><p class="sub">Distinct file paths</p>
      <div class="big" id="langUniqRead">—</div>
      <div class="delta" id="langUniqReadDelta">—</div>
    </div>
    <div class="card"><h3>Unique files edited</h3><p class="sub">Distinct file paths</p>
      <div class="big" id="langUniqEdit">—</div>
      <div class="delta" id="langUniqEditDelta">—</div>
    </div>
  </div>
  <div class="card chart-card">
    <div class="chart-head">
      <div>
        <div class="chart-title">Reads vs Edits, by language</div>
        <div class="chart-sub">Detected from file extensions on Read / Edit / Write / NotebookEdit / MultiEdit tool calls</div>
      </div>
    </div>
    <div class="chart-wrap tall" style="height:440px"><canvas id="langChart"></canvas></div>
  </div>

  <h2 class="section-title">Full breakdown</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Input</th>
          <th>Output</th>
          <th>Cache read</th>
          <th>Cache create</th>
          <th>Total</th>
          <th>% of total</th>
        </tr>
      </thead>
      <tbody id="modelTable"></tbody>
      <tfoot id="modelTableFoot"></tfoot>
    </table>
  </div>

  <div class="note">
    <span class="tag">Source note</span>
    Generated from <strong>~/.claude/stats-cache.json</strong> and local Claude Code transcripts. Both live entirely on this machine.
    Per-day model tokens cover input+output only; full token-class breakdown is available as an all-time total per model.
    For numbers your finance team can reconcile against an Anthropic invoice, use the Admin API
    (<code>/v1/organizations/usage_report/claude_code</code>) — only that source ties out to billing.
  </div>

</main>

<div class="footnote">
  Generated by <code>claude_usage.py</code> · third-party tool · not affiliated with or endorsed by Anthropic ·
  branding loosely inspired by Anthropic's design language but contains no Anthropic marks · {{ generated_at }} ·
  Export PDF uses the browser's built-in print → Save as PDF (Chrome/Edge: instant; Safari: "Save as PDF" in the dialog).
</div>

<script>
const DATA = {{ data_json|safe }};
const DBX  = {{ mcp_json|safe }};
const RESEARCH = {{ research_json|safe }};
const WORDCLOUD = {{ wordcloud_json|safe }};
const LANGS = {{ langs_json|safe }};
const TOTAL_TOOL_CALLS = {{ total_tool_calls }};

// ============== HELPERS ==============
const fmt = n => new Intl.NumberFormat('en-US').format(Math.round(n));
const compact = n => {
  if (n >= 1e9) return (n/1e9).toFixed(2)+'B';
  if (n >= 1e6) return (n/1e6).toFixed(2)+'M';
  if (n >= 1e3) return (n/1e3).toFixed(1)+'k';
  return String(n);
};
const MODEL_COLOR = {
  'claude-opus-4-7':            getComputedStyle(document.documentElement).getPropertyValue('--opus-7').trim(),
  'claude-opus-4-6':            getComputedStyle(document.documentElement).getPropertyValue('--opus-6').trim(),
  'claude-opus-4-5-20251101':   getComputedStyle(document.documentElement).getPropertyValue('--opus-5').trim(),
  'claude-haiku-4-5-20251001':  getComputedStyle(document.documentElement).getPropertyValue('--haiku-5').trim(),
  'claude-sonnet-4-6':          getComputedStyle(document.documentElement).getPropertyValue('--sonnet-6').trim(),
  'claude-sonnet-4-5-20250929': getComputedStyle(document.documentElement).getPropertyValue('--sonnet-5').trim(),
};
const MODEL_SHORT = {
  'claude-opus-4-7':            'Opus 4.7',
  'claude-opus-4-6':            'Opus 4.6',
  'claude-opus-4-5-20251101':   'Opus 4.5',
  'claude-haiku-4-5-20251001':  'Haiku 4.5',
  'claude-sonnet-4-6':          'Sonnet 4.6',
  'claude-sonnet-4-5-20250929': 'Sonnet 4.5',
};
const colorOf = m => MODEL_COLOR[m] || '#888';
const shortOf = m => MODEL_SHORT[m] || m;

Chart.defaults.font.family = "'Inter', ui-sans-serif, system-ui, sans-serif";
Chart.defaults.color = '#5B5A56';
Chart.defaults.borderColor = '#EDE9DC';
// Disable bar/line/donut animations globally. Animations look fine on
// screen but Chrome's print/PDF capture frequently fires mid-animation,
// freezing bars partway from 0 to their final value — which is what made
// every chart look "shrunk" in the html2pdf output and showed up again
// in headless --print-to-pdf. Static charts are also cheaper to draw and
// avoid a layout-shift flicker as data loads.
Chart.defaults.animation = false;
Chart.defaults.animations = {};
Chart.defaults.transitions = { active: { animation: { duration: 0 } } };
Chart.defaults.plugins.legend.labels.boxWidth = 10;
Chart.defaults.plugins.legend.labels.boxHeight = 10;
Chart.defaults.plugins.legend.labels.padding = 14;
Chart.defaults.plugins.tooltip.backgroundColor = '#1F1E1D';
Chart.defaults.plugins.tooltip.titleColor = '#FAF9F5';
Chart.defaults.plugins.tooltip.bodyColor = '#FAF9F5';
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 6;
Chart.defaults.plugins.tooltip.boxPadding = 4;

// ============== KPIs ==============
function totalGrandTokens() {
  let t = 0;
  for (const m of Object.values(DATA.modelUsage)) {
    t += m.inputTokens + m.outputTokens + m.cacheReadInputTokens + m.cacheCreationInputTokens;
  }
  return t;
}
function totalIOOnly() {
  let t = 0;
  for (const m of Object.values(DATA.modelUsage)) t += m.inputTokens + m.outputTokens;
  return t;
}
const firstDate = (DATA.firstSessionDate || '').slice(0,10);
const lastDate  = DATA.lastComputedDate || firstDate;
const totalCalendarDays = firstDate && lastDate
  ? Math.round((new Date(lastDate) - new Date(firstDate)) / 86400000) + 1
  : 0;
const activeDays = (DATA.dailyActivity || []).filter(d => d.messageCount > 0).length;

document.getElementById('windowLabel').textContent = firstDate
  ? `${firstDate}  →  ${lastDate}`
  : 'Window: —';
document.getElementById('kpiTokens').innerHTML = `${compact(totalGrandTokens())}<small>${fmt(totalGrandTokens())} total</small>`;
document.getElementById('kpiTokensDelta').textContent = `${compact(totalIOOnly())} input+output`;
document.getElementById('kpiSessions').textContent = fmt(DATA.totalSessions);
if (DATA.longestSession && DATA.longestSession.duration) {
  document.getElementById('kpiSessionsDelta').textContent =
    `longest: ${(DATA.longestSession.duration/3600000).toFixed(1)} h on ${(DATA.longestSession.timestamp||'').slice(0,10)}`;
}
document.getElementById('kpiMessages').textContent = fmt(DATA.totalMessages);
if (DATA.totalSessions > 0) {
  document.getElementById('kpiMessagesDelta').textContent =
    `${(DATA.totalMessages / DATA.totalSessions).toFixed(0)} per session avg`;
}
document.getElementById('kpiDays').innerHTML = `${activeDays}<small>of ${totalCalendarDays} days</small>`;
if (totalCalendarDays > 0) {
  document.getElementById('kpiDaysDelta').textContent =
    `${(activeDays/totalCalendarDays*100).toFixed(0)}% of calendar days`;
}

// ============== Daily chart (stacked bar by model) ==============
(function() {
  const dates = DATA.dailyModelTokens.map(d => d.date);
  const modelsInUse = new Set();
  DATA.dailyModelTokens.forEach(d => Object.keys(d.tokensByModel).forEach(m => modelsInUse.add(m)));
  const orderedKnown = ['claude-opus-4-7','claude-opus-4-6','claude-opus-4-5-20251101',
                        'claude-haiku-4-5-20251001','claude-sonnet-4-6','claude-sonnet-4-5-20250929'];
  const orderedModels = orderedKnown.filter(m => modelsInUse.has(m))
    .concat([...modelsInUse].filter(m => !orderedKnown.includes(m)));
  const datasets = orderedModels.map(model => ({
    label: shortOf(model),
    data: DATA.dailyModelTokens.map(d => d.tokensByModel[model] || 0),
    backgroundColor: colorOf(model),
    borderWidth: 0, borderRadius: 2, stack: 's',
  }));
  new Chart(document.getElementById('dailyChart'), {
    type: 'bar',
    data: { labels: dates, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { stacked: true, grid: { display: false }, ticks: { maxRotation: 0, autoSkipPadding: 18 } },
        y: { stacked: true, grid: { color: '#EDE9DC' }, ticks: { callback: v => compact(v) } },
      },
      plugins: {
        legend: { position: 'bottom' },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw)} tokens` } },
      },
    },
  });
})();

// ============== Donut ==============
(function() {
  const entries = Object.entries(DATA.modelUsage)
    .map(([m, v]) => [m, v.inputTokens + v.outputTokens + v.cacheReadInputTokens + v.cacheCreationInputTokens])
    .sort((a,b) => b[1] - a[1]);
  const total = entries.reduce((s, [,v]) => s + v, 0);
  new Chart(document.getElementById('donutChart'), {
    type: 'doughnut',
    data: {
      labels: entries.map(([m]) => shortOf(m)),
      datasets: [{
        data: entries.map(([,v]) => v),
        backgroundColor: entries.map(([m]) => colorOf(m)),
        borderColor: '#FFFFFF', borderWidth: 2, hoverOffset: 6,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '62%',
      plugins: {
        legend: { position: 'right' },
        tooltip: { callbacks: { label: ctx => `${ctx.label}: ${compact(ctx.raw)} (${(ctx.raw/total*100).toFixed(1)}%)` } },
      },
    },
  });
})();

// ============== Per-model class breakdown ==============
(function() {
  const models = Object.entries(DATA.modelUsage)
    .sort((a,b) => (b[1].inputTokens+b[1].outputTokens+b[1].cacheReadInputTokens+b[1].cacheCreationInputTokens)
                 - (a[1].inputTokens+a[1].outputTokens+a[1].cacheReadInputTokens+a[1].cacheCreationInputTokens));
  const labels = models.map(([m]) => shortOf(m));
  const palette = { input: '#1F1E1D', output: '#CC785C', cacheRead: '#D4A27F', cacheCreate: '#7A4F37' };
  const datasets = [
    { label: 'Input',        data: models.map(([,v]) => v.inputTokens),               backgroundColor: palette.input,       stack: 's', borderRadius: 2 },
    { label: 'Output',       data: models.map(([,v]) => v.outputTokens),              backgroundColor: palette.output,      stack: 's', borderRadius: 2 },
    { label: 'Cache read',   data: models.map(([,v]) => v.cacheReadInputTokens),      backgroundColor: palette.cacheRead,   stack: 's', borderRadius: 2 },
    { label: 'Cache create', data: models.map(([,v]) => v.cacheCreationInputTokens),  backgroundColor: palette.cacheCreate, stack: 's', borderRadius: 2 },
  ];
  new Chart(document.getElementById('classChart'), {
    type: 'bar',
    data: { labels, datasets },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { color: '#EDE9DC' }, ticks: { callback: v => compact(v) } },
        y: { stacked: true, grid: { display: false } },
      },
      plugins: {
        legend: { position: 'bottom' },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw)}` } },
      },
    },
  });
})();

// ============== Monthly ==============
(function() {
  const months = Object.keys(DATA.monthlyIO).sort();
  const modelsInUse = new Set();
  for (const m of months) Object.keys(DATA.monthlyIO[m]).forEach(k => modelsInUse.add(k));
  const orderedKnown = ['claude-opus-4-7','claude-opus-4-6','claude-opus-4-5-20251101',
                        'claude-haiku-4-5-20251001','claude-sonnet-4-6','claude-sonnet-4-5-20250929'];
  const orderedModels = orderedKnown.filter(m => modelsInUse.has(m))
    .concat([...modelsInUse].filter(m => !orderedKnown.includes(m)));
  const datasets = orderedModels.map(model => ({
    label: shortOf(model),
    data: months.map(m => DATA.monthlyIO[m][model] || 0),
    backgroundColor: colorOf(model),
    borderWidth: 0, borderRadius: 4, stack: 's',
  }));
  new Chart(document.getElementById('monthlyChart'), {
    type: 'bar',
    data: { labels: months, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { display: false } },
        y: { stacked: true, grid: { color: '#EDE9DC' }, ticks: { callback: v => compact(v) } },
      },
      plugins: {
        legend: { position: 'bottom' },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw)} tokens` } },
      },
    },
  });
})();

// ============== Heatmap ==============
(function() {
  if (!firstDate) return;
  const byDate = new Map((DATA.dailyActivity || []).map(d => [d.date, d.sessionCount]));
  const start = new Date(firstDate);
  const end   = new Date(lastDate);
  const cur = new Date(start); cur.setDate(cur.getDate() - cur.getDay());
  const cells = [];
  while (cur <= end) {
    const iso = cur.toISOString().slice(0,10);
    cells.push({ date: iso, dow: cur.getDay(), sessions: byDate.get(iso) || 0, before: cur < start });
    cur.setDate(cur.getDate() + 1);
  }
  const grid = document.getElementById('heatmap');
  const g = document.createElement('div'); g.className = 'heat-grid';
  const counts = [...byDate.values()].filter(v => v > 0).sort((a,b) => a-b);
  const q = p => counts[Math.floor(counts.length * p)] || 1;
  const t1 = q(0.25), t2 = q(0.55), t3 = q(0.85);
  cells.forEach(c => {
    const cell = document.createElement('div');
    let cls = 'heat-cell';
    if (c.sessions > 0) {
      if (c.sessions >= t3) cls += ' l4';
      else if (c.sessions >= t2) cls += ' l3';
      else if (c.sessions >= t1) cls += ' l2';
      else cls += ' l1';
    }
    cell.className = cls;
    cell.title = `${c.date} — ${c.sessions} sessions`;
    cell.style.gridRow = (c.dow + 1).toString();
    g.appendChild(cell);
  });
  grid.appendChild(g);
})();

// ============== Hour-of-day ==============
(function() {
  const hours = Array.from({length:24}, (_,i) => i);
  const data = hours.map(h => DATA.hourCounts[h] || DATA.hourCounts[String(h)] || 0);
  new Chart(document.getElementById('hourChart'), {
    type: 'bar',
    data: {
      labels: hours.map(h => h.toString().padStart(2,'0')+':00'),
      datasets: [{
        label: 'Sessions', data,
        backgroundColor: '#CC785C', hoverBackgroundColor: '#A55A40', borderRadius: 3,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkipPadding: 10 } },
        y: { grid: { color: '#EDE9DC' }, ticks: { precision: 0 } },
      },
      plugins: { legend: { display: false } },
    },
  });
})();

// ============== MCP server (single section) ==============
(function() {
  if (!DBX) return;
  const days = Object.keys(DBX.dailyTotal).sort();
  const peakDay = days.reduce((best, d) => DBX.dailyTotal[d] > DBX.dailyTotal[best] ? d : best, days[0] || '');
  document.getElementById('dbxTotal').textContent = fmt(DBX.totalCalls);
  if (TOTAL_TOOL_CALLS > 0) {
    document.getElementById('dbxTotalDelta').textContent =
      `${(DBX.totalCalls/TOTAL_TOOL_CALLS*100).toFixed(1)}% of all tool calls in transcripts`;
  }
  document.getElementById('dbxDistinct').textContent = DBX.distinctTools;
  document.getElementById('dbxDistinctDelta').textContent = `tools exercised on this server`;
  document.getElementById('dbxActiveDays').textContent = days.length;
  if (days.length) {
    document.getElementById('dbxActiveDaysDelta').textContent = `${days[0]} → ${days[days.length-1]}`;
  }
  if (peakDay) {
    document.getElementById('dbxPeak').innerHTML = `${fmt(DBX.dailyTotal[peakDay])}<small>calls</small>`;
    document.getElementById('dbxPeakDelta').textContent = `on ${peakDay}`;
  }

  const entries = Object.entries(DBX.toolCounts).sort((a,b) => b[1] - a[1]);
  const TOP_N = 12;
  const top = entries.slice(0, TOP_N);
  const restSum = entries.slice(TOP_N).reduce((s,[,n]) => s + n, 0);
  const labels = top.map(([k]) => k);
  const values = top.map(([,v]) => v);
  if (restSum > 0) { labels.push(`+ ${entries.length - TOP_N} other`); values.push(restSum); }
  const palette = [
    '#CC785C','#BB6E54','#A8624C','#945744','#80503E',
    '#6D4737','#7A5A48','#8A6A55','#9B7B66','#AB8B77',
    '#BB9C88','#C9AC99','#9E8B7C'
  ];
  new Chart(document.getElementById('dbxToolsChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: labels.map((_,i) => palette[i % palette.length]),
        borderWidth: 0, borderRadius: 3,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#EDE9DC' }, ticks: { callback: v => compact(v) } },
        y: { grid: { display: false }, ticks: { font: { size: 11.5 } } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `${fmt(ctx.raw)} calls` } },
      },
    },
  });

  if (days.length) {
    const minD = new Date(days[0]);
    const maxD = new Date(days[days.length-1]);
    const allDays = [];
    for (let d = new Date(minD); d <= maxD; d.setDate(d.getDate()+1)) {
      allDays.push(d.toISOString().slice(0,10));
    }
    const dailyValues = allDays.map(d => DBX.dailyTotal[d] || 0);
    new Chart(document.getElementById('dbxDailyChart'), {
      type: 'bar',
      data: {
        labels: allDays,
        datasets: [{
          label: `${DBX.displayName} MCP calls`,
          data: dailyValues,
          backgroundColor: '#CC785C', hoverBackgroundColor: '#A55A40',
          borderWidth: 0, borderRadius: 2,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkipPadding: 22 } },
          y: { grid: { color: '#EDE9DC' }, ticks: { precision: 0 } },
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => `${fmt(ctx.raw)} calls on ${ctx.label}` } },
        },
      },
    });
  }
})();

// ============== Web research ==============
(function() {
  const total = RESEARCH.webfetchTotal + RESEARCH.websearchTotal + RESEARCH.researchTotal;
  document.getElementById('resTotal').textContent = fmt(total);
  if (TOTAL_TOOL_CALLS > 0) {
    document.getElementById('resTotalDelta').textContent =
      `~${(total/TOTAL_TOOL_CALLS*100).toFixed(1)}% of all tool calls in transcripts`;
  }
  document.getElementById('resWF').textContent = fmt(RESEARCH.webfetchTotal);
  document.getElementById('resWFDelta').textContent = `${RESEARCH.uniqueDomains} unique domains visited`;
  document.getElementById('resWS').textContent = fmt(RESEARCH.websearchTotal);
  document.getElementById('resWSDelta').textContent = `${fmt(RESEARCH.uniqueQueries)} distinct queries`;
  document.getElementById('resRE').textContent = fmt(RESEARCH.researchTotal);
  document.getElementById('resREDelta').textContent = `Spawned subagent for deep research`;

  const entries = Object.entries(RESEARCH.domainCounts).sort((a,b) => b[1] - a[1]);
  const labels = entries.map(([k]) => k);
  const values = entries.map(([,v]) => v);
  if (labels.length) {
    const palette = ['#CC785C','#BD6F54','#AE664D','#A05E47','#915541','#834D3B','#7A4F37','#6D4737','#604138','#544039','#4A3E37','#5C4838','#7F6650','#8A745E','#9A8770','#AA9A82','#B7AB95','#C3BCA8','#CECCBA','#D8D9CB'];
    new Chart(document.getElementById('resDomainsChart'), {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: labels.map((_,i) => palette[i % palette.length]),
          borderWidth: 0, borderRadius: 3,
        }],
      },
      options: {
        indexAxis: 'y',
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { grid: { color: '#EDE9DC' }, ticks: { callback: v => compact(v) } },
          y: { grid: { display: false }, ticks: { font: { size: 11 } } },
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => `${fmt(ctx.raw)} fetches` } },
        },
      },
    });
  }

  const allDayKeys = new Set([
    ...Object.keys(RESEARCH.webfetchByDay),
    ...Object.keys(RESEARCH.websearchByDay),
    ...Object.keys(RESEARCH.researchByDay),
  ]);
  if (allDayKeys.has('')) allDayKeys.delete('');
  const days = [...allDayKeys].sort();
  if (days.length) {
    const minD = new Date(days[0]);
    const maxD = new Date(days[days.length-1]);
    const fullDays = [];
    for (let d = new Date(minD); d <= maxD; d.setDate(d.getDate()+1)) {
      fullDays.push(d.toISOString().slice(0,10));
    }
    const ds = [
      { label: 'WebFetch',        data: fullDays.map(d => RESEARCH.webfetchByDay[d]  || 0), backgroundColor: '#CC785C', stack:'r', borderWidth:0, borderRadius:2 },
      { label: 'WebSearch',       data: fullDays.map(d => RESEARCH.websearchByDay[d] || 0), backgroundColor: '#7A4F37', stack:'r', borderWidth:0, borderRadius:2 },
      { label: 'research-expert', data: fullDays.map(d => RESEARCH.researchByDay[d]  || 0), backgroundColor: '#D4A27F', stack:'r', borderWidth:0, borderRadius:2 },
    ];
    new Chart(document.getElementById('resDailyChart'), {
      type: 'bar',
      data: { labels: fullDays, datasets: ds },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        scales: {
          x: { stacked: true, grid: { display: false }, ticks: { maxRotation: 0, autoSkipPadding: 22 } },
          y: { stacked: true, grid: { color: '#EDE9DC' }, ticks: { precision: 0 } },
        },
        plugins: {
          legend: { position: 'bottom' },
          tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw)} on ${ctx.label}` } },
        },
      },
    });
  }
})();

// ============== Word cloud ==============
(function() {
  if (!WORDCLOUD || !window.d3 || !d3.layout || !d3.layout.cloud) return;
  const container = document.getElementById('wordcloud');
  if (!container) return;
  const W = container.clientWidth || 1100;
  const H = 520;
  const items = WORDCLOUD.slice().sort((a,b) => b.weight - a.weight);
  if (!items.length) return;
  const maxW = items[0].weight;
  const minW = items[items.length-1].weight;
  const palette = ['#CC785C','#A55A40','#7A4F37','#6D4737','#5C4838','#B07F5C','#D4A27F'];
  const sizeFor = w => 14 + (60 - 14) * Math.sqrt((w - minW) / Math.max(1, (maxW - minW)));

  const layout = d3.layout.cloud()
    .size([W, H])
    .words(items.map((t, i) => ({
      text: t.text, size: sizeFor(t.weight),
      color: palette[i % palette.length],
      weight: t.weight, examples: t.examples,
    })))
    .padding(6)
    .rotate(() => (Math.random() < 0.75) ? 0 : 90)
    .font("'Inter', system-ui, sans-serif")
    .fontSize(d => d.size)
    .on('end', draw);
  layout.start();

  function draw(words) {
    const svg = d3.select('#wordcloud')
      .append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('width', '100%')
      .attr('height', H);
    const g = svg.append('g').attr('transform', `translate(${W/2},${H/2})`);
    g.selectAll('text')
      .data(words).enter().append('text')
      .style('font-family', "'Inter', system-ui, sans-serif")
      .style('font-weight', d => d.size > 36 ? 600 : 500)
      .style('fill', d => d.color)
      .style('cursor', 'default')
      .attr('text-anchor', 'middle')
      .attr('transform', d => `translate(${d.x},${d.y}) rotate(${d.rotate})`)
      .style('font-size', d => `${d.size}px`)
      .text(d => d.text)
      .append('title')
      .text(d => `${d.text}\nWeight: ${d.weight}\nExample: ${(d.examples||[])[0] || ''}`);
  }

  const tbody = document.getElementById('themeTable');
  if (tbody) {
    items.forEach((t, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><span class="model-chip"><i style="background:${palette[i % palette.length]}"></i>${t.text}</span></td>
        <td>${t.weight}</td>
        <td style="color:var(--text-2); font-size:12.5px">${(t.examples||[])[0] || ''}</td>
      `;
      tbody.appendChild(tr);
    });
  }
})();

// ============== Programming languages ==============
(function() {
  if (!LANGS || !LANGS.rows || !LANGS.rows.length) return;
  document.getElementById('langReads').textContent       = fmt(LANGS.totalReads);
  if (LANGS.totalReads + LANGS.totalWrites > 0) {
    document.getElementById('langReadsDelta').textContent  =
      `${(LANGS.totalReads/(LANGS.totalReads+LANGS.totalWrites)*100).toFixed(0)}% of file activity`;
  }
  document.getElementById('langEdits').textContent       = fmt(LANGS.totalWrites);
  if (LANGS.totalReads > 0) {
    document.getElementById('langEditsDelta').textContent  =
      `${(LANGS.totalWrites/LANGS.totalReads).toFixed(2)}× ratio vs reads`;
  }
  document.getElementById('langUniqRead').textContent    = fmt(LANGS.uniqueRead);
  if (LANGS.uniqueRead > 0) {
    document.getElementById('langUniqReadDelta').textContent = `${(LANGS.totalReads/LANGS.uniqueRead).toFixed(1)} avg reads per file`;
  }
  document.getElementById('langUniqEdit').textContent    = fmt(LANGS.uniqueWrite);
  if (LANGS.uniqueWrite > 0) {
    document.getElementById('langUniqEditDelta').textContent = `${(LANGS.totalWrites/LANGS.uniqueWrite).toFixed(1)} avg edits per file`;
  }

  const rows = LANGS.rows
    .filter(r => r.lang !== 'Other / ?')
    .slice(0, 14);
  const labels = rows.map(r => r.lang);
  new Chart(document.getElementById('langChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Reads', data: rows.map(r => r.reads), backgroundColor: '#CC785C', borderWidth: 0, borderRadius: 3 },
        { label: 'Edits', data: rows.map(r => r.edits), backgroundColor: '#7A4F37', borderWidth: 0, borderRadius: 3 },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { grid: { color: '#EDE9DC' }, ticks: { callback: v => compact(v) } },
        y: { grid: { display: false }, ticks: { font: { size: 12 } } },
      },
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw)} calls`,
            afterBody: items => {
              const row = rows[items[0].dataIndex];
              return `Unique files — read: ${fmt(row.uniqRead)} · edit: ${fmt(row.uniqEdit)}`;
            },
          },
        },
      },
    },
  });
})();

// ============== Export to PDF (via browser print) ==============
// The reference dashboard tried to use html2pdf.js, which captures the page
// via html2canvas at desktop width and then places it at native pixel size
// on an A4 page — meaning right-side content gets clipped, KPI cards collapse
// to single-letter wrapping, and the output is split across ~10 ragged pages.
// Chrome's native print engine handles all of this correctly via @media print:
// vector text, real layout reflow, page-break-inside on cards. So the button
// just triggers window.print() and the @media print rules take care of the
// rest. One extra dialog click vs a direct download, but the resulting PDF
// is sharp, paginates cleanly, and works on every browser without a CDN.
//
// Chart.js charts need an explicit resize() before the print snapshot is
// taken, otherwise their canvas bitmap stays at desktop dimensions and the
// bars end up scaled to a fraction of the new container width. The
// beforeprint hook redraws each chart at print-target size; afterprint
// restores screen size when the dialog closes.
(function() {
  const allCharts = () => (typeof Chart !== 'undefined' && Chart.instances)
    ? Object.values(Chart.instances)
    : [];

  // Chart.js's responsive resize is async (it schedules a redraw via RAF),
  // but Chrome's print snapshot is taken synchronously right after the
  // beforeprint handlers return. So a bare resize() leaves the canvas
  // bitmap stale and the bars come out at the old desktop scale.
  // We force a synchronous redraw via update('none') — and pass the parent's
  // *current* CSS box to resize() so Chart.js can't fall back to a cached size.
  function redrawAll() {
    for (const c of allCharts()) {
      try {
        const parent = c.canvas.parentElement;
        if (parent && parent.clientWidth > 0 && parent.clientHeight > 0) {
          c.resize(parent.clientWidth, parent.clientHeight);
        } else {
          c.resize();
        }
        c.update('none');
      } catch (_) { /* chart was destroyed; ignore */ }
    }
  }

  window.addEventListener('beforeprint', redrawAll);
  window.addEventListener('afterprint', redrawAll);

  // Belt-and-suspenders: matchMedia fires synchronously when the print media
  // state flips on, before beforeprint in some browsers (older Safari).
  const mql = (typeof window.matchMedia === 'function') && window.matchMedia('print');
  if (mql) {
    const cb = (e) => { if (e.matches) redrawAll(); };
    if (typeof mql.addEventListener === 'function') mql.addEventListener('change', cb);
    else if (typeof mql.addListener === 'function') mql.addListener(cb);
  }

  const btn = document.getElementById('exportPdfBtn');
  if (!btn) return;
  btn.addEventListener('click', () => {
    // Trigger a redraw before opening the dialog so charts are already at
    // print dimensions by the time the user clicks "Save as PDF".
    redrawAll();
    requestAnimationFrame(() => requestAnimationFrame(() => window.print()));
  });
})();

// ============== Model table ==============
(function() {
  const tbody = document.getElementById('modelTable');
  const tfoot = document.getElementById('modelTableFoot');
  const grand = totalGrandTokens();
  const rows = Object.entries(DATA.modelUsage)
    .map(([m, v]) => ({
      model: m,
      input: v.inputTokens,
      output: v.outputTokens,
      cacheRead: v.cacheReadInputTokens,
      cacheCreate: v.cacheCreationInputTokens,
      total: v.inputTokens + v.outputTokens + v.cacheReadInputTokens + v.cacheCreationInputTokens,
    }))
    .sort((a,b) => b.total - a.total);
  for (const r of rows) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="model-chip"><i style="background:${colorOf(r.model)}"></i>${shortOf(r.model)}</span></td>
      <td>${fmt(r.input)}</td>
      <td>${fmt(r.output)}</td>
      <td>${fmt(r.cacheRead)}</td>
      <td>${fmt(r.cacheCreate)}</td>
      <td>${fmt(r.total)}</td>
      <td>${grand > 0 ? (r.total/grand*100).toFixed(1) : '0.0'}%</td>
    `;
    tbody.appendChild(tr);
  }
  const sum = (k) => rows.reduce((s, r) => s + r[k], 0);
  tfoot.innerHTML = `<tr>
    <td>Total</td>
    <td>${fmt(sum('input'))}</td>
    <td>${fmt(sum('output'))}</td>
    <td>${fmt(sum('cacheRead'))}</td>
    <td>${fmt(sum('cacheCreate'))}</td>
    <td>${fmt(sum('total'))}</td>
    <td>100%</td>
  </tr>`;
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    sys.exit(main())
