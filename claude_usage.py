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
_MODEL_ID_RE = re.compile(r"^claude-([a-z]+)-(\d+)-(\d+)(?:-\d+)?$")

# Pseudo-models that Claude Code writes into JSONL transcripts for assistant
# turns that never hit the API: harness-injected refusal messages, no-op
# "no response requested" placeholders, and replayed state on resumed sessions.
# Every record carries zero tokens in every class. We drop them so they don't
# clutter the model mix, donut chart, daily/monthly breakdown, or full table.
_NON_BILLABLE_MODELS: frozenset[str] = frozenset({"<synthetic>"})


def model_short_name(model_id: str) -> str:
    """Derive ``'Opus 4.7'`` from ``'claude-opus-4-7-20251101'`` at runtime.

    Anthropic's model IDs follow ``claude-<family>-<major>-<minor>[-<date>]``,
    so no static dict is needed. Unknown shapes pass through unchanged so a
    new family or naming convention won't break the dashboard.
    """
    if not model_id:
        return ""
    m = _MODEL_ID_RE.match(model_id)
    if m:
        family, major, minor = m.group(1), m.group(2), m.group(3)
        return f"{family.title()} {major}.{minor}"
    return model_id


# --------------------------------------------------------------------------- #
# Model pricing                                                               #
# --------------------------------------------------------------------------- #
#
# Source: https://platform.claude.com/docs/en/about-claude/pricing
# Fetched on 2026-05-14. Prices are in USD per million (1e6) tokens, standard
# (non-batch, non-fast-mode) rates.
#
# `cache_write_5m` is the default Claude Code uses for the 5-minute TTL; the
# JSONL `cache_creation_input_tokens` field does not distinguish 5m vs 1h, so
# we apply this rate uniformly. If you opted into 1h caches the displayed cost
# will under-estimate cache-write spend (real rate is 2× input instead of
# 1.25×). Anthropic's stated multipliers: cache_write_5m = 1.25× input,
# cache_write_1h = 2× input, cache_read = 0.1× input.
#
# Keys are ``(family, major, minor)`` tuples matching the parsed model id.
# Edit this table when Anthropic publishes new rates.
MODEL_PRICING: dict[tuple[str, str, str], dict[str, float]] = {
    ("opus", "4", "7"):   {"input":  5.00, "output": 25.00, "cache_write_5m":  6.25, "cache_read": 0.50},
    ("opus", "4", "6"):   {"input":  5.00, "output": 25.00, "cache_write_5m":  6.25, "cache_read": 0.50},
    ("opus", "4", "5"):   {"input":  5.00, "output": 25.00, "cache_write_5m":  6.25, "cache_read": 0.50},
    ("opus", "4", "1"):   {"input": 15.00, "output": 75.00, "cache_write_5m": 18.75, "cache_read": 1.50},
    ("opus", "4", "0"):   {"input": 15.00, "output": 75.00, "cache_write_5m": 18.75, "cache_read": 1.50},
    ("sonnet", "4", "6"): {"input":  3.00, "output": 15.00, "cache_write_5m":  3.75, "cache_read": 0.30},
    ("sonnet", "4", "5"): {"input":  3.00, "output": 15.00, "cache_write_5m":  3.75, "cache_read": 0.30},
    ("sonnet", "4", "0"): {"input":  3.00, "output": 15.00, "cache_write_5m":  3.75, "cache_read": 0.30},
    ("sonnet", "3", "7"): {"input":  3.00, "output": 15.00, "cache_write_5m":  3.75, "cache_read": 0.30},
    ("haiku", "4", "5"):  {"input":  1.00, "output":  5.00, "cache_write_5m":  1.25, "cache_read": 0.10},
    ("haiku", "3", "5"):  {"input":  0.80, "output":  4.00, "cache_write_5m":  1.00, "cache_read": 0.08},
    ("opus", "3", "0"):   {"input": 15.00, "output": 75.00, "cache_write_5m": 18.75, "cache_read": 1.50},
    ("haiku", "3", "0"):  {"input":  0.25, "output":  1.25, "cache_write_5m":  0.30, "cache_read": 0.03},
}
PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICING_FETCHED_ON = "2026-05-14"


def model_price(model_id: str) -> dict[str, float] | None:
    """Return the per-MTok price dict for a model id, or None if unknown.

    Falls back through (major, minor) → (major, "0") → None. Anthropic's
    "Opus 4" and "Sonnet 4" entries are encoded with minor=0 above; an
    explicit `claude-opus-4-20250514` id without a minor still matches.
    """
    if not model_id:
        return None
    m = _MODEL_ID_RE.match(model_id)
    if not m:
        return None
    family, major, minor = m.group(1), m.group(2), m.group(3)
    return MODEL_PRICING.get((family, major, minor))


def model_cost(model_id: str, usage: dict[str, int]) -> float:
    """Apply the per-MTok rates to a usage dict; returns USD.

    Usage dict uses the cache's own field names (``inputTokens`` etc.) so
    this can be called against both ``stats['modelUsage'][m]`` entries
    and the delta-style dicts produced by ``compute_live_delta``.
    """
    p = model_price(model_id)
    if not p:
        return 0.0
    return (
        usage.get("inputTokens", 0) * p["input"]
        + usage.get("outputTokens", 0) * p["output"]
        + usage.get("cacheCreationInputTokens", 0) * p["cache_write_5m"]
        + usage.get("cacheReadInputTokens", 0) * p["cache_read"]
    ) / 1_000_000.0


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

    model_usage = {
        m: v for m, v in (raw.get("modelUsage") or {}).items()
        if m not in _NON_BILLABLE_MODELS
    }
    daily_model_tokens: list[dict[str, Any]] = []
    for d in raw.get("dailyModelTokens", []):
        filtered_tbm = {
            m: t for m, t in (d.get("tokensByModel") or {}).items()
            if m not in _NON_BILLABLE_MODELS
        }
        if filtered_tbm:
            daily_model_tokens.append({"date": d.get("date", ""), "tokensByModel": filtered_tbm})

    monthly: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for d in daily_model_tokens:
        month = d["date"][:7]
        for model, tokens in d["tokensByModel"].items():
            monthly[month][model] += tokens

    return {
        "firstSessionDate": raw.get("firstSessionDate", ""),
        "lastComputedDate": raw.get("lastComputedDate", ""),
        "totalSessions": raw.get("totalSessions", 0),
        "totalMessages": raw.get("totalMessages", 0),
        "longestSession": raw.get("longestSession", {}),
        "modelUsage": model_usage,
        "dailyActivity": raw.get("dailyActivity", []),
        "dailyModelTokens": daily_model_tokens,
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


def _iter_assistant_usage(claude_dir: Path, after_date: str | None):
    """Yield ``(date, raw_model_id, usage_dict, session_id)`` per assistant
    message whose date > ``after_date``. Deduped by ``message.id``.

    Sibling of ``_iter_tool_uses``. ``usage`` is recorded **once per
    assistant message** (not per content block), so we dedupe by
    ``message.id`` here rather than by ``block.id``. The ``after_date``
    filter is the cache's ``lastComputedDate``; we walk only the tail.
    """
    seen_message_ids: set[str] = set()
    projects = claude_dir / "projects"
    if not projects.exists():
        return
    for jsonl in projects.glob("**/*.jsonl"):
        session_id = jsonl.stem
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
                    mid = msg.get("id")
                    if not mid or mid in seen_message_ids:
                        continue
                    seen_message_ids.add(mid)
                    ts = rec.get("timestamp") or msg.get("timestamp") or ""
                    date = ts[:10] if isinstance(ts, str) else ""
                    if after_date and date and date <= after_date:
                        continue
                    usage = msg.get("usage") or {}
                    yield date, msg.get("model") or "", usage, session_id
        except OSError:
            continue


def compute_live_delta(
    claude_dir: Path, last_computed_date: str | None
) -> dict[str, Any]:
    """Compute the post-cache delta from raw JSONL transcripts.

    ``stats-cache.json`` only stores data through ``last_computed_date``
    (set by Claude Code itself, typically yesterday at best). This walks
    every assistant turn dated strictly after that and returns a
    delta-shaped dict that mirrors the cache's own keys, ready to merge.
    """
    delta_by_model: dict[str, dict[str, int]] = defaultdict(lambda: {
        "inputTokens": 0,
        "outputTokens": 0,
        "cacheReadInputTokens": 0,
        "cacheCreationInputTokens": 0,
    })
    delta_daily: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sessions_seen: set[str] = set()
    msg_count = 0
    for date, model, usage, sess in _iter_assistant_usage(
        claude_dir, last_computed_date
    ):
        sessions_seen.add(sess)
        msg_count += 1
        if model in _NON_BILLABLE_MODELS:
            # Harness-injected turn — zero tokens by definition; still count
            # it toward newMessages/newSessions so totals stay consistent
            # with stats-cache.json's own accounting.
            continue
        b = delta_by_model[model]
        b["inputTokens"] += usage.get("input_tokens", 0)
        b["outputTokens"] += usage.get("output_tokens", 0)
        b["cacheReadInputTokens"] += usage.get("cache_read_input_tokens", 0)
        b["cacheCreationInputTokens"] += usage.get("cache_creation_input_tokens", 0)
        if date:
            delta_daily[date][model] += (
                usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            )
    return {
        "modelUsage": {m: dict(v) for m, v in delta_by_model.items()},
        "dailyModelTokens": [
            {"date": d, "tokensByModel": dict(v)}
            for d, v in sorted(delta_daily.items())
        ],
        "newSessions": len(sessions_seen),
        "newMessages": msg_count,
    }


def compute_costs(stats: dict[str, Any]) -> dict[str, Any]:
    """Roll the per-model token counts into USD estimates.

    Per-model and grand totals are computed exactly from
    ``stats['modelUsage']`` (which has the full input/output/cacheRead/
    cacheCreate breakdown). Per-day and per-month costs are approximated
    by allocating the daily ``input+output`` total at each model's
    **effective $/IO-token** rate — i.e. ``model_cost / (in+out)`` taken
    from the all-time numbers. That way the cache portion is implicitly
    pro-rated across the days that produced it (which is the best we can
    do given the cache only stores per-day in+out tokens per model, not
    a full token-class breakdown). Sum of daily ≈ total.
    """
    cost_by_model: dict[str, float] = {}
    effective_per_io_token: dict[str, float] = {}
    missing: list[str] = []
    total_cost = 0.0
    for mid, usage in (stats.get("modelUsage") or {}).items():
        if not model_price(mid):
            if mid:
                missing.append(mid)
            continue
        c = model_cost(mid, usage)
        cost_by_model[mid] = c
        total_cost += c
        io = (usage.get("inputTokens", 0) or 0) + (usage.get("outputTokens", 0) or 0)
        if io > 0:
            effective_per_io_token[mid] = c / io

    daily_cost: list[dict[str, Any]] = []
    for d in stats.get("dailyModelTokens") or []:
        c = 0.0
        for mid, toks in (d.get("tokensByModel") or {}).items():
            rate = effective_per_io_token.get(mid)
            if rate is None:
                continue
            c += toks * rate
        daily_cost.append({"date": d.get("date", ""), "cost": c})

    monthly_cost: dict[str, float] = defaultdict(float)
    for entry in daily_cost:
        month = (entry.get("date") or "")[:7]
        if month:
            monthly_cost[month] += entry["cost"]

    return {
        "totalCost": total_cost,
        "costByModel": cost_by_model,
        "dailyCost": daily_cost,
        "monthlyCost": dict(monthly_cost),
        "effectivePerIoToken": effective_per_io_token,
        "missingModels": sorted(set(missing)),
        "source": PRICING_SOURCE,
        "fetchedOn": PRICING_FETCHED_ON,
        "cacheWriteAssumption": "5min TTL",
    }


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


# --------------------------------------------------------------------------- #
# Row-level events + sessions (for in-browser filtering)                      #
# --------------------------------------------------------------------------- #

_COMMAND_TAG_RE = re.compile(r"<command-name>\s*/?([\w\-:]+)\s*</command-name>", re.IGNORECASE)
_SLASH_CMD_RE = re.compile(r"^/([\w\-:]+)\b")

# Claude Code built-in slash commands. Detected the same way as user-defined
# skills (via <command-name> tags), but we don't surface them as "skills" in
# the dashboard. This list is intentionally a superset of the official docs
# so future-Claude-Code additions stay filtered out by default.
_SLASH_BUILTINS: frozenset[str] = frozenset({
    "clear", "compact", "model", "init", "exit", "help", "status", "memory",
    "resume", "quit", "save", "logout", "mcp", "doctor", "ide", "terminal-setup",
    "context", "simplify", "cost", "config", "login", "feedback", "bug",
    "fast", "loop", "remember", "permissions", "agents", "output-style",
    "continue", "release-notes", "usage", "rewind", "add-dir",
})


def _first_user_text(rec: dict) -> str:
    """Pull the text out of a user record, ignoring tool_result blocks."""
    msg = rec.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for blk in content:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") == "text":
                return blk.get("text") or ""
    return ""


def _detect_skill(text: str) -> str:
    """Detect a skill / slash-command name from a user message body."""
    if not text:
        return ""
    m = _COMMAND_TAG_RE.search(text)
    if m:
        return m.group(1).strip().lower()
    head = text.strip().split("\n", 1)[0]
    m2 = _SLASH_CMD_RE.match(head)
    if m2:
        return m2.group(1).strip().lower()
    return ""


def walk_with_events(
    claude_dir: Path,
    mcp_servers_filter: list[str] | None,
) -> dict[str, Any]:
    """Single pass over ``projects/**/*.jsonl`` returning rich row-level data.

    Returns a dict with the same keys ``walk_transcripts`` returns *plus*:

      events   — list of ``[dayIdx, projIdx, modelIdx, serverIdx, toolIdx,
                 skillIdx, langReadIdx, langEditIdx, sessIdx, kind]``
                 ``kind`` encodes the event family:
                   0 = generic, 1 = MCP, 2 = WebFetch, 3 = WebSearch,
                   4 = research-expert subagent, 5 = Read, 6 = Edit/Write.
      sessions — list of ``{id, proj, start, prompt}`` (one row per JSONL).
      dims     — interned dimension arrays the browser uses to populate the
                 filter dropdowns and to decode integer-encoded event rows.
    """
    # --- Interning helpers ----------------------------------------------------
    dims: dict[str, list[str]] = {
        "projects": [],
        "models": [],
        "servers": ["(none)"],
        "tools": ["(none)"],
        "skills": ["(none)"],
        "langs": ["(none)"],
        "dates": [],            # sorted at the end
        "classifications": ["(unclassified)"],
    }
    idx_maps: dict[str, dict[str, int]] = {
        "projects": {},
        "models": {},
        "servers": {"(none)": 0},
        "tools": {"(none)": 0},
        "skills": {"(none)": 0},
        "langs": {"(none)": 0},
    }

    def intern(dim: str, key: str) -> int:
        if not key:
            return 0 if dim in idx_maps and "(none)" in idx_maps[dim] else -1
        m = idx_maps[dim]
        if key in m:
            return m[key]
        m[key] = len(dims[dim])
        dims[dim].append(key)
        return m[key]

    # --- Aggregation state (mirrors walk_transcripts) ------------------------
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
    lang_reads: Counter = Counter()
    lang_edits: Counter = Counter()
    lang_unique_read: dict[str, set[str]] = defaultdict(set)
    lang_unique_edit: dict[str, set[str]] = defaultdict(set)

    READ_TOOLS = {"Read"}
    EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}

    events: list[list[int]] = []
    sessions: list[dict[str, Any]] = []
    seen_block_ids: set[str] = set()
    dates_set: set[str] = set()

    projects_root = claude_dir / "projects"
    if not projects_root.exists():
        # No transcripts — return empties.
        return {
            "mcp_block": None,
            "research": _empty_research(),
            "queries": [],
            "langs": _empty_langs(),
            "total_tool_calls": 0,
            "events": [],
            "sessions": [],
            "dims": dims,
        }

    for jsonl in sorted(projects_root.glob("**/*.jsonl")):
        session_id = jsonl.stem
        proj_name: str | None = None
        first_prompt: str = ""
        start_date: str = ""
        # Skills are inferred per-message; persist the most recent one for
        # subsequent tool_use events in the same logical turn.
        current_skill: str = ""

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

                    cwd = rec.get("cwd")
                    if proj_name is None and isinstance(cwd, str) and cwd:
                        proj_name = Path(cwd).name or "(unknown)"

                    ts = rec.get("timestamp") or ""
                    date = ts[:10] if isinstance(ts, str) else ""
                    if date:
                        dates_set.add(date)
                        if not start_date:
                            start_date = date

                    rtype = rec.get("type")

                    if rtype == "user":
                        text = _first_user_text(rec)
                        if text and not first_prompt:
                            first_prompt = text[:400]
                        skill_name = _detect_skill(text)
                        if skill_name:
                            current_skill = skill_name
                        continue

                    if rtype != "assistant":
                        continue

                    msg = rec.get("message") or {}
                    model_raw = msg.get("model") or ""
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

                        name = block.get("name") or ""
                        inp = block.get("input") or {}

                        # ---- existing per-event aggregates ----
                        total_tool_calls += 1
                        server_name = ""
                        mcp_tool_name = ""
                        kind = 0
                        lang_read = ""
                        lang_edit = ""

                        mmatch = _MCP_TOOL_RE.match(name)
                        if mmatch:
                            server_name, mcp_tool_name = mmatch.group(1), mmatch.group(2)
                            mcp_by_server[server_name]["tools"][mcp_tool_name] += 1
                            if date:
                                mcp_by_server[server_name]["daily"][date] += 1
                            kind = 1
                        elif name == "WebFetch":
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
                            kind = 2
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
                            kind = 3
                        elif name in ("Task", "Agent"):
                            if (inp.get("subagent_type") or "") == "research-expert":
                                research_total += 1
                                if date:
                                    research_by_day[date] += 1
                                kind = 4
                            else:
                                continue   # uninteresting agent call
                        elif name in READ_TOOLS:
                            path = inp.get("file_path") or inp.get("notebook_path") or inp.get("path") or ""
                            if path:
                                lang_read = _lang_for(path)
                                lang_reads[lang_read] += 1
                                lang_unique_read[lang_read].add(path)
                            kind = 5
                        elif name in EDIT_TOOLS:
                            path = inp.get("file_path") or inp.get("notebook_path") or inp.get("path") or ""
                            if path:
                                lang_edit = _lang_for(path)
                                lang_edits[lang_edit] += 1
                                lang_unique_edit[lang_edit].add(path)
                            kind = 6
                        else:
                            # generic tool — still emit an event row so it can
                            # be filtered by project/model/skill.
                            kind = 0

                        # ---- intern dimensions and emit a row ----
                        if proj_name is None:
                            proj_name = "(unknown)"
                        proj_idx = intern("projects", proj_name)
                        model_short = model_short_name(model_raw) or "(unknown)"
                        model_idx = intern("models", model_short)
                        server_idx = intern("servers", server_name) if server_name else 0
                        tool_idx = intern("tools", mcp_tool_name) if mcp_tool_name else 0
                        skill_idx = intern("skills", current_skill) if current_skill else 0
                        lr_idx = intern("langs", lang_read) if lang_read else 0
                        le_idx = intern("langs", lang_edit) if lang_edit else 0
                        # day idx is finalized after the walk completes (we
                        # sort the dates dimension); store the date string for
                        # now in a parallel list, then translate.
                        events.append([
                            date,         # placeholder; index applied below
                            proj_idx,
                            model_idx,
                            server_idx,
                            tool_idx,
                            skill_idx,
                            lr_idx,
                            le_idx,
                            len(sessions),  # session row index (current file)
                            kind,
                        ])
        except OSError:
            continue

        if proj_name is None:
            proj_name = "(unknown)"
        sess_proj_idx = intern("projects", proj_name)
        sessions.append({
            "id": session_id,
            "proj": sess_proj_idx,
            "start": start_date,
            "prompt": first_prompt,
            "cls": 0,  # classification fills in later
        })

    # --- Finalize date dimension and replace date strings with indices -------
    dims["dates"] = sorted(dates_set)
    date_idx: dict[str, int] = {d: i for i, d in enumerate(dims["dates"])}
    for ev in events:
        ev[0] = date_idx.get(ev[0], -1)

    # --- MCP filtering / top-1 (matches existing semantics) ------------------
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

    # Languages payload (same shape as existing) ------------------------------
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

    return {
        "mcp_block": mcp_block,
        "research": research,
        "queries": queries,
        "langs": langs,
        "total_tool_calls": total_tool_calls,
        "events": events,
        "sessions": sessions,
        "dims": dims,
    }


def _empty_research() -> dict[str, Any]:
    return {
        "webfetchTotal": 0, "websearchTotal": 0, "researchTotal": 0,
        "uniqueDomains": 0, "uniqueQueries": 0,
        "domainCounts": {}, "queryKeywords": {},
        "webfetchByDay": {}, "websearchByDay": {}, "researchByDay": {},
    }


def _empty_langs() -> dict[str, Any]:
    return {"totalReads": 0, "totalWrites": 0, "uniqueRead": 0, "uniqueWrite": 0, "rows": []}


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
    chunks: int = 10,
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


# Static fallback categories — used ONLY when (a) the user explicitly
# disables AI via --no-ai, or (b) no `claude` CLI is on PATH, or (c) the
# discovery call fails. These are intentionally generic to coding work, not
# to any particular stack or audience. The default path discovers categories
# from the user's actual data via Haiku, so these are never the user-facing
# taxonomy unless one of those fallback conditions hits.
_FALLBACK_CATEGORIES: tuple[str, ...] = (
    "Bug fix",
    "New feature",
    "Refactor",
    "Question / Q&A",
    "Exploration / Research",
    "Documentation",
    "Setup / Config",
    "Testing",
    "Operations",
    "Other",
)


_DISCOVER_PROMPT = textwrap.dedent("""\
    You are looking at first-messages from a coding-assistant user's sessions.
    Your job is to PROPOSE a taxonomy of 8-12 SHORT CATEGORY LABELS that
    together cover the messages below. The labels will be used to classify
    every session's first message; one label per session.

    Output JSON ONLY. No prose. No markdown fences. Schema:
      {"categories": ["<label1>", "<label2>", ...]}

    Rules:
      - Labels are short noun phrases (1-4 words), title-cased.
      - Aim for 8-12 categories.
      - Include 'Other' as the final fallback for messages that fit nothing.
      - Labels should be distinct from each other; no near-duplicates.
      - Pick labels that match THIS user's actual work, not generic ones.
        Example: if many messages are about Rust ECS, "Rust ECS" is a fine
        label; if many are about Databricks pipelines, "Databricks Pipelines"
        is fine.

    Messages (representative sample):
    """)


def discover_topic_categories(
    sessions: list[dict[str, Any]],
    claude_bin: str,
    sample_size: int = 40,
    timeout: int = 120,
) -> list[str] | None:
    """Sample session first-prompts and let Haiku propose a taxonomy.

    Returns ``None`` if the discovery call fails — caller falls back to
    ``_FALLBACK_CATEGORIES``.
    """
    prompts = [s.get("prompt") or "" for s in sessions if (s.get("prompt") or "").strip()]
    if not prompts:
        return None
    # Sample deterministically (evenly spaced) so a re-run produces the same
    # sample. Random sampling would make categories non-deterministic across
    # runs which makes tracking trends harder.
    n = min(sample_size, len(prompts))
    if n == 0:
        return None
    step = max(1, len(prompts) // n)
    sample = prompts[::step][:n]
    body = "\n".join(f"- {p[:300].replace(chr(10), ' ').strip()}" for p in sample)
    prompt = _DISCOVER_PROMPT + body
    try:
        result = subprocess.run(
            [claude_bin, "-p", prompt, "--model", "haiku", "--output-format", "text"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        print(f"  [warn] discover_topic_categories failed: {exc}", file=sys.stderr)
        return None
    out = (result.stdout or "").strip()
    if out.startswith("```"):
        out = re.sub(r"^```(?:json)?\s*", "", out)
        out = re.sub(r"\s*```\s*$", "", out)
    brace = out.find("{")
    if brace > 0:
        out = out[brace:]
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        return None
    cats = parsed.get("categories")
    if not isinstance(cats, list):
        return None
    cleaned = [c.strip() for c in cats if isinstance(c, str) and c.strip()]
    # Always make sure 'Other' is present so the classifier has a fallback bucket.
    if not any(c.lower() == "other" for c in cleaned):
        cleaned.append("Other")
    return cleaned[:12] if cleaned else None


def _classify_chunk(
    chunk: list[tuple[int, str]],
    claude_bin: str,
    categories: list[str],
    timeout: int = 120,
) -> dict[int, str]:
    """Run one ``claude -p`` call over a batch of (index, text) pairs."""
    cat_block = "\n".join(f"      {c}" for c in categories)
    prompt_header = textwrap.dedent("""\
        You will be given a numbered list of first messages from coding sessions.
        Classify each message into EXACTLY ONE of these categories:

        """) + cat_block + textwrap.dedent("""\

        Output JSON ONLY. No prose. No markdown fences. Schema:
          {"results": [{"i": <int>, "c": "<category>"}]}

        Rules:
          - i is the message number as given.
          - c MUST be one of the categories above, spelled exactly.
          - One result per input message.
          - If the message is a slash command (starts with /), classify by what
            the command name implies.

        Messages:
        """)
    body = "\n".join(f"{i}. {text[:300].replace(chr(10), ' ').strip()}" for i, text in chunk)
    prompt = prompt_header + body
    try:
        result = subprocess.run(
            [claude_bin, "-p", prompt, "--model", "haiku", "--output-format", "text"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        print(f"  [warn] classify chunk failed: {exc}", file=sys.stderr)
        return {}
    out = (result.stdout or "").strip()
    if out.startswith("```"):
        out = re.sub(r"^```(?:json)?\s*", "", out)
        out = re.sub(r"\s*```\s*$", "", out)
    brace = out.find("{")
    if brace > 0:
        out = out[brace:]
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        return {}
    results = parsed.get("results")
    if not isinstance(results, list):
        return {}
    valid = set(categories)
    other_label = next((c for c in categories if c.lower() == "other"), "Other")
    mapping: dict[int, str] = {}
    for r in results:
        if not isinstance(r, dict):
            continue
        try:
            i = int(r.get("i"))
        except (TypeError, ValueError):
            continue
        c = (r.get("c") or "").strip()
        if c not in valid:
            c = other_label
        mapping[i] = c
    return mapping


def classify_sessions(
    sessions: list[dict[str, Any]],
    claude_bin: str | None,
    categories: list[str],
    chunk_size: int = 60,
    max_workers: int = 20,
) -> list[str]:
    """Classify each session's first prompt via parallel Haiku calls.

    Returns a list aligned with ``sessions`` mapping each session to one of the
    given ``categories`` (or ``"(unclassified)"`` if Haiku failed).
    """
    out: list[str] = ["(unclassified)"] * len(sessions)
    if not claude_bin or not sessions or not categories:
        return out

    indexed: list[tuple[int, str]] = [
        (i, s.get("prompt") or "") for i, s in enumerate(sessions) if (s.get("prompt") or "").strip()
    ]
    if not indexed:
        return out

    chunks = [indexed[i : i + chunk_size] for i in range(0, len(indexed), chunk_size)]
    print(
        f"  Running {len(chunks)} parallel `claude -p` calls "
        f"over {len(indexed)} session first-prompts…",
        file=sys.stderr,
    )

    with cf.ThreadPoolExecutor(max_workers=min(max_workers, len(chunks))) as ex:
        futures = [ex.submit(_classify_chunk, c, claude_bin, categories) for c in chunks]
        for fut in cf.as_completed(futures):
            for i, cat in fut.result().items():
                if 0 <= i < len(out):
                    out[i] = cat
    return out


def merge_themes(themes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge per-chunk themes into one ranked list by case-insensitive text.

    Previously this used a hand-curated table of canonical names and substring
    patterns. That table was Databricks-flavored and useless to anyone else,
    so it's gone. The 4 parallel Haiku calls already produce fairly consistent
    names within a single dataset; case-insensitive dedupe handles the rest.
    Truly redundant themes (e.g. "Databricks Apps" + "databricks apps") merge;
    everything else passes through. Some users will see a few near-duplicates
    in the wordcloud — that's the price of portability.
    """
    bucket: dict[str, dict[str, Any]] = {}
    for t in themes:
        key = (t["text"] or "").strip()
        if not key:
            continue
        lookup = key.lower()
        slot = bucket.setdefault(lookup, {"text": key, "weight": 0, "examples": []})
        slot["weight"] += t["weight"]
        for ex in t["examples"]:
            if ex not in slot["examples"] and len(slot["examples"]) < 5:
                slot["examples"].append(ex)
    return sorted(bucket.values(), key=lambda r: r["weight"], reverse=True)


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
    events: list[list[int]],
    sessions: list[dict[str, Any]],
    dims: dict[str, list[str]],
    generated_at: str,
    last_computed_date: str = "",
    delta_last_date: str = "",
    delta_new_sessions: int = 0,
    costs: dict[str, Any] | None = None,
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
        events_json=json.dumps(events, default=str, separators=(",", ":")),
        sessions_json=json.dumps(sessions, default=str, separators=(",", ":")),
        dims_json=json.dumps(dims, default=str, separators=(",", ":")),
        total_tool_calls=total_tool_calls,
        websearch_count=research["websearchTotal"],
        # Hide the Topics section if classification didn't run (e.g., --no-ai
        # or no claude CLI). dims.classifications always starts with the
        # placeholder "(unclassified)"; real categories appear after that.
        has_classifications=len(dims.get("classifications", [])) > 1,
        # Hide the Skills section unless at least one user-defined skill was
        # detected. dims.skills[0] is the "(none)" placeholder; some of the
        # rest are Claude Code built-ins (/clear, /compact, /model, etc.)
        # which we don't surface as "skills".
        has_skills=any(
            sk and sk not in _SLASH_BUILTINS for sk in dims.get("skills", [])
        ),
        generated_at=generated_at,
        last_computed_date=last_computed_date,
        delta_last_date=delta_last_date,
        delta_new_sessions=delta_new_sessions,
        costs_json=json.dumps(costs or {}, default=str),
        pricing_source=(costs or {}).get("source", ""),
        pricing_fetched_on=(costs or {}).get("fetchedOn", ""),
        cost_missing_models=(costs or {}).get("missingModels", []),
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
    p.add_argument(
        "--topics", type=str, default=None,
        help=(
            "Comma-separated list of classification categories for the "
            '"What you worked on" section. Overrides the AI-discovered '
            "taxonomy. Example: --topics \"Bug fix,New feature,Refactor,Other\"."
        ),
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
    walked = walk_with_events(claude_dir, mcp_filter)
    mcp_block = walked["mcp_block"]
    research = walked["research"]
    queries = walked["queries"]
    langs = walked["langs"]
    total_tool_calls = walked["total_tool_calls"]
    events = walked["events"]
    sessions_rows = walked["sessions"]
    dims = walked["dims"]
    mcp_label = mcp_block["name"] if mcp_block else "none"
    print(
        f"  → {total_tool_calls} unique tool calls; "
        f"MCP server: {mcp_label}; "
        f"{research['websearchTotal']} WebSearch queries; "
        f"{langs['totalReads']} reads, {langs['totalWrites']} edits "
        f"across {len(langs['rows'])} languages; "
        f"{len(events)} event rows; {len(sessions_rows)} sessions; "
        f"{len(dims['projects'])} projects, {len(dims['skills'])-1} skills.",
        file=sys.stderr,
    )

    # --- Live delta: extend the cache with today + any post-cache days ---
    # stats-cache.json only refreshes when /usage is opened in the "all" tab
    # AND covers data through lastComputedDate (typically yesterday). Walk
    # the JSONL tail and add a delta so the dashboard isn't stale.
    last_computed_date = stats.get("lastComputedDate") or ""
    print(
        f"Computing live delta after {last_computed_date or '(no cache date)'}…",
        file=sys.stderr,
    )
    delta = compute_live_delta(claude_dir, last_computed_date or None)
    for m, v in delta["modelUsage"].items():
        base = stats["modelUsage"].setdefault(m, {
            "inputTokens": 0,
            "outputTokens": 0,
            "cacheReadInputTokens": 0,
            "cacheCreationInputTokens": 0,
        })
        for k in (
            "inputTokens",
            "outputTokens",
            "cacheReadInputTokens",
            "cacheCreationInputTokens",
        ):
            base[k] = base.get(k, 0) + v[k]
    existing_dates = {d["date"] for d in stats["dailyModelTokens"]}
    for entry in delta["dailyModelTokens"]:
        if entry["date"] not in existing_dates:
            stats["dailyModelTokens"].append(entry)
    stats["dailyModelTokens"].sort(key=lambda d: d["date"])
    stats["totalSessions"] = stats.get("totalSessions", 0) + delta["newSessions"]
    stats["totalMessages"] = stats.get("totalMessages", 0) + delta["newMessages"]
    monthly_rebuilt: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for d in stats["dailyModelTokens"]:
        month = d["date"][:7]
        for m, t in d.get("tokensByModel", {}).items():
            monthly_rebuilt[month][m] += t
    stats["monthlyIO"] = {m: dict(v) for m, v in monthly_rebuilt.items()}
    delta_last_date = (
        delta["dailyModelTokens"][-1]["date"] if delta["dailyModelTokens"] else ""
    )
    if delta_last_date:
        stats["liveLastDate"] = delta_last_date
    print(
        f"  → live delta: +{delta['newMessages']} messages, "
        f"+{delta['newSessions']} sessions, "
        f"{len(delta['dailyModelTokens'])} new daily rows "
        f"(through {delta_last_date or '—'}).",
        file=sys.stderr,
    )

    costs = compute_costs(stats)
    print(
        f"Estimated cost: ${costs['totalCost']:,.2f} across {len(costs['costByModel'])} priced model(s)"
        + (
            f"; unpriced models skipped: {', '.join(costs['missingModels'])}"
            if costs["missingModels"]
            else ""
        )
        + ".",
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

    # Optionally classify session first-prompts via Haiku and attach the
    # classification to the corresponding event rows (the events table uses a
    # 0-based index into dims.classifications; we wrote 0 as a placeholder).
    if args.no_ai:
        print("Skipping session classification (--no-ai).", file=sys.stderr)
    elif not sessions_rows:
        pass
    else:
        claude_bin = shutil.which("claude")
        if not claude_bin:
            pass  # warning already printed for themes step
        else:
            # Decide the taxonomy: explicit --topics > AI-discovered from this
            # user's data > static fallback. The default path is discovery,
            # which makes the dashboard portable: a Rust dev sees Rust-flavored
            # categories, a Databricks dev sees Databricks-flavored ones.
            if args.topics:
                categories = [
                    s.strip() for s in args.topics.split(",") if s.strip()
                ]
                if not any(c.lower() == "other" for c in categories):
                    categories.append("Other")
                print(
                    f"Using {len(categories)} user-supplied topic categories: "
                    f"{', '.join(categories)}",
                    file=sys.stderr,
                )
            else:
                print("Discovering topic categories from session sample…", file=sys.stderr)
                discovered = discover_topic_categories(sessions_rows, claude_bin)
                if discovered:
                    categories = discovered
                    print(
                        f"  → discovered {len(categories)} categories: "
                        f"{', '.join(categories)}",
                        file=sys.stderr,
                    )
                else:
                    categories = list(_FALLBACK_CATEGORIES)
                    print(
                        f"  [warn] discovery failed — using static fallback "
                        f"({len(categories)} categories).",
                        file=sys.stderr,
                    )
            print("Classifying session first-prompts via Haiku…", file=sys.stderr)
            cats = classify_sessions(sessions_rows, claude_bin, categories)
            cls_idx_map: dict[str, int] = {
                name: i for i, name in enumerate(dims["classifications"])
            }

            def intern_cls(name: str) -> int:
                if name in cls_idx_map:
                    return cls_idx_map[name]
                cls_idx_map[name] = len(dims["classifications"])
                dims["classifications"].append(name)
                return cls_idx_map[name]

            for i, cat in enumerate(cats):
                sessions_rows[i]["cls"] = intern_cls(cat)
            counts: Counter = Counter(cats)
            top = ", ".join(f"{k}={v}" for k, v in counts.most_common(5))
            print(f"  → classified {len(sessions_rows)} sessions ({top}).", file=sys.stderr)

    html = render_html(
        title=args.title,
        stats=stats,
        mcp_block=mcp_block,
        research=research,
        wordcloud=wordcloud,
        langs=langs,
        total_tool_calls=total_tool_calls,
        events=events,
        sessions=sessions_rows,
        dims=dims,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        last_computed_date=last_computed_date,
        delta_last_date=delta_last_date,
        delta_new_sessions=delta["newSessions"],
        costs=costs,
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
    .grid.cols-5 { grid-template-columns: repeat(5, 1fr); }
    .grid.cols-2 { grid-template-columns: 1.1fr 0.9fr; }
    @media (max-width: 1200px) {
      .grid.cols-5 { grid-template-columns: repeat(3, 1fr); }
    }
    @media (max-width: 1020px) {
      .grid.cols-4 { grid-template-columns: repeat(2, 1fr); }
      .grid.cols-5 { grid-template-columns: repeat(2, 1fr); }
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

    /* ==== Filter bar ==== */
    .filterbar {
      position: sticky; top: 0; z-index: 50;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 10px 36px;
      display: flex; flex-wrap: wrap; align-items: center;
      gap: 10px;
      box-shadow: 0 1px 0 rgba(0,0,0,0.02);
    }
    .filterbar .fb-label {
      font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
      color: var(--text-3); font-weight: 600; margin-right: 4px;
    }
    .filterbar select, .filterbar input[type="date"] {
      font: inherit; font-size: 12.5px;
      padding: 5px 9px; border-radius: 6px;
      background: var(--card); color: var(--text);
      border: 1px solid var(--border); cursor: pointer;
      font-family: 'Inter', system-ui, sans-serif;
    }
    .filterbar select:focus, .filterbar input[type="date"]:focus {
      outline: none; border-color: var(--coral);
    }
    .filter-group {
      position: relative;
      display: inline-flex; align-items: center; gap: 4px;
    }
    .fb-btn {
      font: inherit; font-size: 12.5px;
      padding: 5px 11px; border-radius: 6px;
      background: var(--card); color: var(--text);
      border: 1px solid var(--border); cursor: pointer;
      display: inline-flex; align-items: center; gap: 6px;
      font-family: 'Inter', system-ui, sans-serif;
    }
    .fb-btn:hover { border-color: var(--coral); }
    .fb-btn .count {
      display: inline-block; min-width: 18px; padding: 0 5px;
      background: var(--coral); color: #FFF; border-radius: 999px;
      font-size: 11px; font-weight: 600; text-align: center;
      line-height: 16px; height: 16px;
    }
    .fb-btn .count.zero { background: var(--border); color: var(--text-2); }
    .fb-btn .chev { color: var(--text-3); font-size: 10px; }
    .fb-popover {
      position: absolute; top: calc(100% + 4px); left: 0;
      background: var(--card); border: 1px solid var(--border);
      border-radius: 8px; padding: 8px;
      min-width: 220px; max-height: 320px; overflow-y: auto;
      box-shadow: 0 4px 16px rgba(0,0,0,0.08);
      display: none; z-index: 60;
    }
    .fb-popover.open { display: block; }
    .fb-popover label {
      display: flex; align-items: center; gap: 8px;
      padding: 5px 6px; font-size: 12.5px; color: var(--text);
      cursor: pointer; border-radius: 4px;
    }
    .fb-popover label:hover { background: var(--bg); }
    .fb-popover input[type="checkbox"] { margin: 0; cursor: pointer; }
    .fb-popover .fb-search {
      width: 100%; padding: 6px 8px; margin-bottom: 6px;
      border: 1px solid var(--border); border-radius: 5px;
      font: inherit; font-size: 12.5px; font-family: 'Inter', system-ui, sans-serif;
    }
    .fb-popover .fb-actions {
      display: flex; gap: 6px; padding: 6px 4px 2px;
      border-top: 1px solid var(--border); margin-top: 6px;
    }
    .fb-popover .fb-actions button {
      flex: 1; padding: 4px 8px; font-size: 11.5px;
      background: transparent; border: 1px solid var(--border);
      border-radius: 4px; cursor: pointer; color: var(--text-2);
      font-family: 'Inter', system-ui, sans-serif;
    }
    .fb-popover .fb-actions button:hover { color: var(--text); border-color: var(--coral); }
    .fb-reset {
      margin-left: auto;
      font: inherit; font-size: 12px;
      padding: 5px 11px; border-radius: 6px;
      background: transparent; color: var(--coral-2);
      border: 1px solid transparent; cursor: pointer;
      font-family: 'Inter', system-ui, sans-serif;
    }
    .fb-reset:hover { background: var(--card); border-color: var(--border); }
    .filter-warn {
      display: inline-flex; align-items: center; gap: 4px;
      font-size: 11px; padding: 2px 8px; border-radius: 999px;
      background: rgba(204,120,92,0.10); color: var(--coral-2);
      font-weight: 500; letter-spacing: 0.02em;
      margin-left: 6px;
    }
    .filter-warn.hidden { display: none; }
    @media print {
      .filterbar { display: none !important; }
    }
    @media (max-width: 760px) {
      .filterbar { padding: 8px 14px; gap: 6px; }
      .fb-popover { left: auto; right: 0; }
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

<div class="filterbar" id="filterbar">
  <span class="fb-label">From</span>
  <input type="date" id="fbDateMin" />
  <span class="fb-label">to</span>
  <input type="date" id="fbDateMax" />

  <div class="filter-group" data-dim="models">
    <button class="fb-btn" data-popover="popModels">
      Model <span class="count zero" id="cntModels">0</span> <span class="chev">▾</span>
    </button>
    <div class="fb-popover" id="popModels"></div>
  </div>
  <div class="filter-group" data-dim="projects">
    <button class="fb-btn" data-popover="popProjects">
      Project <span class="count zero" id="cntProjects">0</span> <span class="chev">▾</span>
    </button>
    <div class="fb-popover" id="popProjects"></div>
  </div>
  <div class="filter-group" data-dim="servers">
    <button class="fb-btn" data-popover="popServers">
      MCP <span class="count zero" id="cntServers">0</span> <span class="chev">▾</span>
    </button>
    <div class="fb-popover" id="popServers"></div>
  </div>
  <div class="filter-group" data-dim="skills">
    <button class="fb-btn" data-popover="popSkills">
      Skill <span class="count zero" id="cntSkills">0</span> <span class="chev">▾</span>
    </button>
    <div class="fb-popover" id="popSkills"></div>
  </div>
  <div class="filter-group" data-dim="classes">
    <button class="fb-btn" data-popover="popClasses">
      Topic <span class="count zero" id="cntClasses">0</span> <span class="chev">▾</span>
    </button>
    <div class="fb-popover" id="popClasses"></div>
  </div>

  <button class="fb-reset" id="fbReset" type="button" title="Clear all filters">Reset</button>
</div>

<section class="hero">
  <div class="eyebrow">Local data · stats-cache.json</div>
  <h1>Your token <em>universe</em>, visualized.</h1>
  <p>A dashboard drawn entirely from <span class="source-pill">~/.claude/stats-cache.json</span> and your local
    transcripts.
    <span class="filter-warn hidden" id="filterWarn">Filtered view</span></p>
</section>

<main class="container">

  <!-- KPI cards -->
  <div class="grid cols-5">
    <div class="card"><h3>Total tokens</h3><p class="sub">All classes incl. cache</p>
      <div class="big" id="kpiTokens">—</div>
      <div class="delta" id="kpiTokensDelta">—</div>
    </div>
    <div class="card"><h3>Estimated cost</h3><p class="sub">At list API prices · see source note</p>
      <div class="big" id="kpiCost">—</div>
      <div class="delta" id="kpiCostDelta">—</div>
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
        <div class="chart-sub">Hover for estimated $ cost per day. Cache tokens are not stored per-day in this cache — see model breakdown below for the full picture.</div>
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
        <div class="chart-sub">Right axis: estimated $ cost per month (brown line). Note: this is the only per-month breakdown available in stats-cache (cache tokens are aggregate-only).</div>
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

  {% if has_classifications %}
  <h2 class="section-title">What you worked on</h2>
  <div class="grid cols-4" style="margin-bottom:16px">
    <div class="card"><h3>Sessions classified</h3><p class="sub">First-prompt → Haiku</p>
      <div class="big" id="topicSessions">—</div>
      <div class="delta" id="topicSessionsDelta">—</div>
    </div>
    <div class="card"><h3>Dominant topic</h3><p class="sub">Most-frequent category</p>
      <div class="big" id="topicTop" style="font-size:24px;">—</div>
      <div class="delta" id="topicTopDelta">—</div>
    </div>
    <div class="card"><h3>Unique topics</h3><p class="sub">Active categories in view</p>
      <div class="big" id="topicCount">—</div>
      <div class="delta" id="topicCountDelta">—</div>
    </div>
    <div class="card"><h3>Unclassified</h3><p class="sub">First prompt empty or unreadable</p>
      <div class="big" id="topicUnc">—</div>
      <div class="delta" id="topicUncDelta">—</div>
    </div>
  </div>
  <div class="card chart-card">
    <div class="chart-head">
      <div>
        <div class="chart-title">Sessions and tool calls by topic</div>
        <div class="chart-sub">Each session's first prompt classified by Haiku into one of 10 buckets · chart ignores the topic filter so the distribution stays visible while you slice</div>
      </div>
    </div>
    <div class="chart-wrap tall" style="height:380px"><canvas id="topicChart"></canvas></div>
  </div>
  {% endif %}

  {% if has_skills %}
  <h2 class="section-title">Skills you used</h2>
  <div class="grid cols-4" style="margin-bottom:16px">
    <div class="card"><h3>Skill invocations</h3><p class="sub">Tool calls under an active skill</p>
      <div class="big" id="skillCalls">—</div>
      <div class="delta" id="skillCallsDelta">—</div>
    </div>
    <div class="card"><h3>Unique skills</h3><p class="sub">Distinct skill names invoked</p>
      <div class="big" id="skillUnique">—</div>
      <div class="delta" id="skillUniqueDelta">—</div>
    </div>
    <div class="card"><h3>Most used</h3><p class="sub">Skill with the most tool calls</p>
      <div class="big" id="skillTop" style="font-size:24px;">—</div>
      <div class="delta" id="skillTopDelta">—</div>
    </div>
    <div class="card"><h3>Skill-using sessions</h3><p class="sub">Sessions that invoked any skill</p>
      <div class="big" id="skillSess">—</div>
      <div class="delta" id="skillSessDelta">—</div>
    </div>
  </div>
  <div class="card chart-card">
    <div class="chart-head">
      <div>
        <div class="chart-title">Tool calls and sessions by skill</div>
        <div class="chart-sub">Custom skills only — Claude Code built-ins (<code>/clear</code>, <code>/compact</code>, <code>/model</code>, etc.) are excluded · chart ignores the skill filter to keep the distribution visible while you slice</div>
      </div>
    </div>
    <div class="chart-wrap tall" style="height:420px"><canvas id="skillChart"></canvas></div>
  </div>
  {% endif %}

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
    <div style="margin-top:8px;">
      <strong>Cost estimate:</strong> rates pulled from
      <a href="{{ pricing_source }}" target="_blank" rel="noopener" style="color:var(--coral-2);">{{ pricing_source }}</a>
      on <code>{{ pricing_fetched_on }}</code>. Cache writes assume the 5-minute TTL (Claude Code default; 1-hour TTL would be ≈60% higher per cache-write token). Per-day and per-month costs apply each model's all-time effective $ / IO-token rate, so the cache portion is pro-rated across days. Per-model totals and the KPI card use the exact per-class breakdown.
      {% if cost_missing_models %}
      Models without a known price (skipped from cost totals): <code>{{ cost_missing_models|join(', ') }}</code>.
      {% endif %}
    </div>
    {% if delta_last_date %}
    <div style="margin-top:8px;">
      Cache last refreshed by Claude Code: <code>{{ last_computed_date or '—' }}</code> · supplemented with
      live transcripts through <code>{{ delta_last_date }}</code> ({{ delta_new_sessions }} new session{{ '' if delta_new_sessions == 1 else 's' }}).
    </div>
    {% endif %}
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
const EVENTS = {{ events_json|safe }};
const SESSIONS = {{ sessions_json|safe }};
const DIMS = {{ dims_json|safe }};
const COSTS = {{ costs_json|safe }};
const COST_EFF = (COSTS && COSTS.effectivePerIoToken) || {};
function fmtUsd(x) {
  if (!isFinite(x) || x === null || x === undefined) return '—';
  if (x >= 100000) return '$' + Math.round(x).toLocaleString();
  if (x >= 1000)   return '$' + x.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (x >= 100)    return '$' + x.toFixed(0);
  if (x >= 10)     return '$' + x.toFixed(1);
  if (x >= 1)      return '$' + x.toFixed(2);
  if (x > 0)       return '$' + x.toFixed(3);
  return '$0.00';
}

// ============== HELPERS ==============
const fmt = n => new Intl.NumberFormat('en-US').format(Math.round(n));
const compact = n => {
  if (n >= 1e9) return (n/1e9).toFixed(2)+'B';
  if (n >= 1e6) return (n/1e6).toFixed(2)+'M';
  if (n >= 1e3) return (n/1e3).toFixed(1)+'k';
  return String(n);
};

// Anthropic model IDs follow `claude-<family>-<major>-<minor>[-<date>]`.
// Deriving the short label at runtime means a new model id (e.g. when
// Opus 4.8 ships) lands in the dashboard without a code change.
const _MODEL_ID_RE = /^claude-([a-z]+)-(\d+)-(\d+)(?:-\d+)?$/;
const _SHORT_CACHE = new Map();
function shortOf(modelId) {
  if (!modelId) return '';
  if (_SHORT_CACHE.has(modelId)) return _SHORT_CACHE.get(modelId);
  const m = _MODEL_ID_RE.exec(modelId);
  const label = m
    ? `${m[1][0].toUpperCase() + m[1].slice(1)} ${m[2]}.${m[3]}`
    : modelId;
  _SHORT_CACHE.set(modelId, label);
  return label;
}

// Colors are generated from the model id so new models get a stable,
// deterministic hue without anyone editing a palette. The family steers
// the base hue (Opus = coral, Sonnet = warm brown, Haiku = light tan,
// anything else = neutral) and the version offsets lightness so newer
// versions render slightly brighter than older ones.
const _COLOR_CACHE = new Map();
const _FAMILY_HUES = {
  opus:   { h: 14,  s: 52 },
  sonnet: { h: 22,  s: 38 },
  haiku:  { h: 32,  s: 48 },
};
function colorOf(modelId) {
  if (!modelId) return '#888';
  if (_COLOR_CACHE.has(modelId)) return _COLOR_CACHE.get(modelId);
  const m = _MODEL_ID_RE.exec(modelId);
  let color;
  if (m) {
    const family = m[1].toLowerCase();
    const minor = parseInt(m[3], 10);
    const base = _FAMILY_HUES[family];
    if (base) {
      // Map minor version into a 28-62 lightness band — newer minors are
      // lighter / more brand-coral, older minors are darker brown. Mirrors
      // the original Anthropic palette where Opus 4.7 was the brand coral.
      const lum = Math.max(28, Math.min(62, 30 + (minor - 4) * 8));
      color = `hsl(${base.h}, ${base.s}%, ${lum}%)`;
    }
  }
  if (!color) {
    // Unknown family — hash the id into a coral-adjacent palette so the page
    // stays on-brand. djb2 hash → palette index.
    const palette = ['#CC785C','#A55A40','#7A4F37','#B07F5C','#D4A27F','#5C4838'];
    let h = 5381;
    for (let i = 0; i < modelId.length; i++) h = ((h << 5) + h) + modelId.charCodeAt(i);
    color = palette[Math.abs(h) % palette.length];
  }
  _COLOR_CACHE.set(modelId, color);
  return color;
}

// Inverse: short label → raw id. Built incrementally as shortOf() runs;
// looked up in places that need to map from a filter selection (which
// stores short names) back to a raw model id (used as a key in
// DATA.modelUsage / dailyModelTokens.tokensByModel).
const SHORT_TO_RAW = {};
function rememberShort(rawId) {
  if (rawId) SHORT_TO_RAW[shortOf(rawId)] = rawId;
}
for (const raw of Object.keys(DATA.modelUsage || {})) rememberShort(raw);
for (const d of (DATA.dailyModelTokens || [])) {
  for (const raw of Object.keys(d.tokensByModel || {})) rememberShort(raw);
}

// ============== Chart.js defaults ==============
Chart.defaults.font.family = "'Inter', ui-sans-serif, system-ui, sans-serif";
Chart.defaults.color = '#5B5A56';
Chart.defaults.borderColor = '#EDE9DC';
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

// ============== Filter state ==============
const TOTAL_DATES = DIMS.dates.length;
const state = {
  dateMin: 0,
  dateMax: Math.max(0, TOTAL_DATES - 1),
  models: new Set(),
  projects: new Set(),
  servers: new Set(),
  skills: new Set(),
  classes: new Set(),
};
const sessCls = SESSIONS.map(s => s.cls);  // session_index -> classification_id

function filtersActive() {
  return state.dateMin !== 0
      || state.dateMax !== Math.max(0, TOTAL_DATES - 1)
      || state.models.size > 0
      || state.projects.size > 0
      || state.servers.size > 0
      || state.skills.size > 0
      || state.classes.size > 0;
}

function eventMatches(ev) {
  // ev = [d, p, m, s, t, skill, lr, le, sess, kind]
  if (ev[0] < state.dateMin || ev[0] > state.dateMax) return false;
  if (state.models.size   && !state.models.has(ev[2]))   return false;
  if (state.projects.size && !state.projects.has(ev[1])) return false;
  if (state.servers.size  && !state.servers.has(ev[3]))  return false;
  if (state.skills.size   && !state.skills.has(ev[5]))   return false;
  if (state.classes.size) {
    const cls = sessCls[ev[8]];
    if (cls === undefined || !state.classes.has(cls)) return false;
  }
  return true;
}

function sessionMatches(s) {
  // s = {id, proj, start, prompt, cls}
  // Date check: session is "in range" if its start date index is in range.
  if (s.start) {
    const di = DATES_INDEX.get(s.start);
    if (di !== undefined && (di < state.dateMin || di > state.dateMax)) return false;
  }
  if (state.projects.size && !state.projects.has(s.proj)) return false;
  if (state.classes.size  && !state.classes.has(s.cls))   return false;
  return true;
}
const DATES_INDEX = new Map(DIMS.dates.map((d, i) => [d, i]));

// ============== Date string helpers ==============
function currentDateRange() {
  const min = DIMS.dates[state.dateMin] || '';
  const max = DIMS.dates[state.dateMax] || '';
  return { min, max };
}

// ============== Chart registry ==============
const charts = {};

function destroyChart(id) {
  if (charts[id]) {
    try { charts[id].destroy(); } catch (_) {}
    delete charts[id];
  }
}

// ============== KPI rendering ==============
function renderTopKPIs() {
  const { min: dMin, max: dMax } = currentDateRange();
  document.getElementById('windowLabel').textContent = dMin
    ? `${dMin}  →  ${dMax}` : 'Window: —';

  // dailyModelTokens clipped to date range (always input+output only).
  const wantedShort = state.models.size
    ? new Set([...state.models].map(i => DIMS.models[i]))
    : null;

  let tokensIO = 0;
  let dailyClipped = [];
  for (const d of (DATA.dailyModelTokens || [])) {
    if (d.date < dMin || d.date > dMax) continue;
    let kept = {};
    let dayTotal = 0;
    for (const [m, v] of Object.entries(d.tokensByModel || {})) {
      if (wantedShort && !wantedShort.has(shortOf(m))) continue;
      kept[m] = v;
      dayTotal += v;
    }
    if (Object.keys(kept).length) {
      dailyClipped.push({ date: d.date, tokensByModel: kept });
      tokensIO += dayTotal;
    }
  }

  let tokensTotalDisplay, tokensSubtitle;
  if (filtersActive()) {
    tokensTotalDisplay = tokensIO;
    tokensSubtitle = `${compact(tokensIO)} input+output`;
  } else {
    // Original behavior: show grand total including cache.
    let grand = 0, io = 0;
    for (const m of Object.values(DATA.modelUsage || {})) {
      grand += (m.inputTokens||0) + (m.outputTokens||0)
             + (m.cacheReadInputTokens||0) + (m.cacheCreationInputTokens||0);
      io    += (m.inputTokens||0) + (m.outputTokens||0);
    }
    tokensTotalDisplay = grand;
    tokensSubtitle = `${compact(io)} input+output`;
  }
  document.getElementById('kpiTokens').innerHTML =
    `${compact(tokensTotalDisplay)}<small>${fmt(tokensTotalDisplay)} total</small>`;
  document.getElementById('kpiTokensDelta').textContent = tokensSubtitle;

  // --- Estimated cost card ---
  // Filtered view: apply each model's effective $/IO-token to clipped daily
  // totals (the only per-day numbers available). Unfiltered: use the exact
  // per-class total computed server-side. The two paths agree at full range.
  let costDisplay, costSubtitle;
  if (filtersActive()) {
    let costFiltered = 0;
    for (const d of dailyClipped) {
      for (const [m, v] of Object.entries(d.tokensByModel || {})) {
        const rate = COST_EFF[m];
        if (rate) costFiltered += v * rate;
      }
    }
    costDisplay = costFiltered;
    costSubtitle = 'in this filtered view';
  } else {
    costDisplay = (COSTS && COSTS.totalCost) || 0;
    const missing = (COSTS && COSTS.missingModels) || [];
    costSubtitle = missing.length
      ? `unpriced models skipped: ${missing.map(shortOf).join(', ')}`
      : 'list API prices, all models priced';
  }
  document.getElementById('kpiCost').innerHTML =
    `${fmtUsd(costDisplay)}<small>${COSTS && COSTS.cacheWriteAssumption ? 'assumes ' + COSTS.cacheWriteAssumption + ' cache' : ''}</small>`;
  document.getElementById('kpiCostDelta').textContent = costSubtitle;

  // Sessions / Messages / Active days — filterable derived from SESSIONS + filteredEvents.
  let filteredSessions = SESSIONS.filter(sessionMatches);
  if (filtersActive()) {
    document.getElementById('kpiSessions').textContent = fmt(filteredSessions.length);
    document.getElementById('kpiSessionsDelta').textContent = `Sessions in this filtered view`;
  } else {
    document.getElementById('kpiSessions').textContent = fmt(DATA.totalSessions);
    if (DATA.longestSession && DATA.longestSession.duration) {
      document.getElementById('kpiSessionsDelta').textContent =
        `longest: ${(DATA.longestSession.duration/3600000).toFixed(1)} h on ${(DATA.longestSession.timestamp||'').slice(0,10)}`;
    } else {
      document.getElementById('kpiSessionsDelta').textContent = '';
    }
  }

  if (filtersActive()) {
    // Use filtered-event count as "Messages" proxy.
    const fEvCount = window._filteredEvents ? window._filteredEvents.length : 0;
    document.getElementById('kpiMessages').textContent = fmt(fEvCount);
    document.getElementById('kpiMessagesDelta').textContent = `tool calls in filtered view`;
  } else {
    document.getElementById('kpiMessages').textContent = fmt(DATA.totalMessages);
    if (DATA.totalSessions > 0) {
      document.getElementById('kpiMessagesDelta').textContent =
        `${(DATA.totalMessages / DATA.totalSessions).toFixed(0)} per session avg`;
    }
  }

  if (filtersActive()) {
    // Active days = unique dates in filtered events.
    const fEv = window._filteredEvents || [];
    const dayIdSet = new Set();
    for (const ev of fEv) dayIdSet.add(ev[0]);
    const calRange = state.dateMax - state.dateMin + 1;
    document.getElementById('kpiDays').innerHTML =
      `${dayIdSet.size}<small>of ${calRange} days</small>`;
    document.getElementById('kpiDaysDelta').textContent =
      calRange > 0 ? `${(dayIdSet.size/calRange*100).toFixed(0)}% of selected window` : '';
  } else {
    const firstDate = (DATA.firstSessionDate || '').slice(0,10);
    const lastDate  = DATA.lastComputedDate || firstDate;
    const totalCalendarDays = firstDate && lastDate
      ? Math.round((new Date(lastDate) - new Date(firstDate)) / 86400000) + 1
      : 0;
    const activeDays = (DATA.dailyActivity || []).filter(d => d.messageCount > 0).length;
    document.getElementById('kpiDays').innerHTML = `${activeDays}<small>of ${totalCalendarDays} days</small>`;
    document.getElementById('kpiDaysDelta').textContent =
      totalCalendarDays > 0 ? `${(activeDays/totalCalendarDays*100).toFixed(0)}% of calendar days` : '';
  }

  // Cache results for downstream chart renders.
  return { dailyClipped, dMin, dMax, wantedShort };
}

// ============== Daily chart ==============
function renderDailyChart(dailyClipped) {
  destroyChart('daily');
  const dates = dailyClipped.map(d => d.date);
  const modelsInUse = new Set();
  dailyClipped.forEach(d => Object.keys(d.tokensByModel).forEach(m => modelsInUse.add(m)));
  // Order by family then version descending, so the user sees their newest
  // models first. Unknown families sort alphabetically at the end.
  const FAMILY_ORDER = { opus: 0, sonnet: 1, haiku: 2 };
  const orderedModels = [...modelsInUse].sort((a, b) => {
    const ma = /^claude-([a-z]+)-(\d+)-(\d+)/.exec(a);
    const mb = /^claude-([a-z]+)-(\d+)-(\d+)/.exec(b);
    if (!ma && !mb) return a.localeCompare(b);
    if (!ma) return 1; if (!mb) return -1;
    const fa = FAMILY_ORDER[ma[1]] ?? 99;
    const fb = FAMILY_ORDER[mb[1]] ?? 99;
    if (fa !== fb) return fa - fb;
    const va = parseInt(ma[2]) * 100 + parseInt(ma[3]);
    const vb = parseInt(mb[2]) * 100 + parseInt(mb[3]);
    return vb - va;   // newer first
  });
  const datasets = orderedModels.map(model => ({
    label: shortOf(model),
    data: dailyClipped.map(d => d.tokensByModel[model] || 0),
    backgroundColor: colorOf(model),
    borderWidth: 0, borderRadius: 2, stack: 's',
  }));
  charts.daily = new Chart(document.getElementById('dailyChart'), {
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
        tooltip: {
          callbacks: {
            label: ctx => {
              const mid = orderedModels[ctx.datasetIndex];
              const rate = COST_EFF[mid];
              const cost = rate ? ctx.raw * rate : null;
              return cost !== null
                ? `${ctx.dataset.label}: ${fmt(ctx.raw)} tokens · ${fmtUsd(cost)}`
                : `${ctx.dataset.label}: ${fmt(ctx.raw)} tokens`;
            },
            footer: items => {
              let total = 0;
              for (const it of items) {
                const mid = orderedModels[it.datasetIndex];
                const rate = COST_EFF[mid];
                if (rate) total += it.raw * rate;
              }
              return total > 0 ? `Estimated cost: ${fmtUsd(total)}` : '';
            },
          },
        },
      },
    },
  });
}

// ============== Donut & Per-model class breakdown ==============
function computeModelUsageFiltered(dailyClipped, wantedShort) {
  // When unfiltered, return DATA.modelUsage as-is (preserves cache totals).
  if (!filtersActive()) return DATA.modelUsage;
  // When filtered, recompute model totals from clipped daily I+O only.
  const out = {};
  for (const d of dailyClipped) {
    for (const [m, v] of Object.entries(d.tokensByModel)) {
      if (!out[m]) out[m] = { inputTokens: 0, outputTokens: 0, cacheReadInputTokens: 0, cacheCreationInputTokens: 0 };
      // We don't have an input/output split per-day; treat as combined.
      out[m].outputTokens += v;  // pile onto output for the donut/class chart proportionality
    }
  }
  return out;
}

function renderDonut(usage) {
  destroyChart('donut');
  const entries = Object.entries(usage)
    .map(([m, v]) => [m, (v.inputTokens||0)+(v.outputTokens||0)+(v.cacheReadInputTokens||0)+(v.cacheCreationInputTokens||0)])
    .filter(([,v]) => v > 0)
    .sort((a,b) => b[1] - a[1]);
  const total = entries.reduce((s, [,v]) => s + v, 0);
  charts.donut = new Chart(document.getElementById('donutChart'), {
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
        tooltip: { callbacks: { label: ctx => `${ctx.label}: ${compact(ctx.raw)} (${total ? (ctx.raw/total*100).toFixed(1) : 0}%)` } },
      },
    },
  });
}

function renderClassChart(usage) {
  destroyChart('classChart');
  const models = Object.entries(usage)
    .filter(([,v]) => (v.inputTokens||0)+(v.outputTokens||0)+(v.cacheReadInputTokens||0)+(v.cacheCreationInputTokens||0) > 0)
    .sort((a,b) => (b[1].inputTokens+b[1].outputTokens+b[1].cacheReadInputTokens+b[1].cacheCreationInputTokens)
                 - (a[1].inputTokens+a[1].outputTokens+a[1].cacheReadInputTokens+a[1].cacheCreationInputTokens));
  const labels = models.map(([m]) => shortOf(m));
  const palette = { input: '#1F1E1D', output: '#CC785C', cacheRead: '#D4A27F', cacheCreate: '#7A4F37' };
  const datasets = [
    { label: 'Input',        data: models.map(([,v]) => v.inputTokens||0),               backgroundColor: palette.input,       stack: 's', borderRadius: 2 },
    { label: 'Output',       data: models.map(([,v]) => v.outputTokens||0),              backgroundColor: palette.output,      stack: 's', borderRadius: 2 },
    { label: 'Cache read',   data: models.map(([,v]) => v.cacheReadInputTokens||0),      backgroundColor: palette.cacheRead,   stack: 's', borderRadius: 2 },
    { label: 'Cache create', data: models.map(([,v]) => v.cacheCreationInputTokens||0),  backgroundColor: palette.cacheCreate, stack: 's', borderRadius: 2 },
  ];
  charts.classChart = new Chart(document.getElementById('classChart'), {
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
}

// ============== Monthly chart ==============
function renderMonthly(dailyClipped) {
  destroyChart('monthly');
  // Re-roll monthly from dailyClipped so date filters apply.
  const byMonth = {};
  for (const d of dailyClipped) {
    const month = d.date.slice(0,7);
    if (!byMonth[month]) byMonth[month] = {};
    for (const [m, v] of Object.entries(d.tokensByModel)) {
      byMonth[month][m] = (byMonth[month][m] || 0) + v;
    }
  }
  const months = Object.keys(byMonth).sort();
  const modelsInUse = new Set();
  for (const m of months) Object.keys(byMonth[m]).forEach(k => modelsInUse.add(k));
  // Order by family then version descending, so the user sees their newest
  // models first. Unknown families sort alphabetically at the end.
  const FAMILY_ORDER = { opus: 0, sonnet: 1, haiku: 2 };
  const orderedModels = [...modelsInUse].sort((a, b) => {
    const ma = /^claude-([a-z]+)-(\d+)-(\d+)/.exec(a);
    const mb = /^claude-([a-z]+)-(\d+)-(\d+)/.exec(b);
    if (!ma && !mb) return a.localeCompare(b);
    if (!ma) return 1; if (!mb) return -1;
    const fa = FAMILY_ORDER[ma[1]] ?? 99;
    const fb = FAMILY_ORDER[mb[1]] ?? 99;
    if (fa !== fb) return fa - fb;
    const va = parseInt(ma[2]) * 100 + parseInt(ma[3]);
    const vb = parseInt(mb[2]) * 100 + parseInt(mb[3]);
    return vb - va;   // newer first
  });
  const datasets = orderedModels.map(model => ({
    label: shortOf(model),
    data: months.map(m => byMonth[m][model] || 0),
    backgroundColor: colorOf(model),
    borderWidth: 0, borderRadius: 4, stack: 's',
  }));
  // Build an overlay line of estimated $ cost per month (right axis).
  const monthlyCost = months.map(month => {
    let c = 0;
    for (const [mid, v] of Object.entries(byMonth[month])) {
      const rate = COST_EFF[mid];
      if (rate) c += v * rate;
    }
    return c;
  });
  const costLine = {
    type: 'line',
    label: 'Estimated cost (USD)',
    data: monthlyCost,
    yAxisID: 'yCost',
    borderColor: '#7A4F37',
    backgroundColor: '#7A4F37',
    borderWidth: 2,
    tension: 0.25,
    pointRadius: 3,
    pointHoverRadius: 5,
    stack: 'cost',
  };
  charts.monthly = new Chart(document.getElementById('monthlyChart'), {
    type: 'bar',
    data: { labels: months, datasets: [...datasets, costLine] },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { stacked: true, grid: { display: false } },
        y: { stacked: true, grid: { color: '#EDE9DC' }, ticks: { callback: v => compact(v) } },
        yCost: {
          position: 'right', stacked: false,
          grid: { display: false },
          ticks: { callback: v => fmtUsd(v) },
          title: { display: true, text: 'USD', color: '#7A4F37', font: { size: 11 } },
        },
      },
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: ctx => {
              if (ctx.dataset.yAxisID === 'yCost') {
                return `${ctx.dataset.label}: ${fmtUsd(ctx.raw)}`;
              }
              const mid = orderedModels[ctx.datasetIndex];
              const rate = COST_EFF[mid];
              const cost = rate ? ctx.raw * rate : null;
              return cost !== null
                ? `${ctx.dataset.label}: ${fmt(ctx.raw)} tokens · ${fmtUsd(cost)}`
                : `${ctx.dataset.label}: ${fmt(ctx.raw)} tokens`;
            },
            footer: items => {
              let total = 0;
              for (const it of items) {
                if (it.dataset.yAxisID === 'yCost') continue;
                const mid = orderedModels[it.datasetIndex];
                const rate = COST_EFF[mid];
                if (rate) total += it.raw * rate;
              }
              return total > 0 ? `Estimated total: ${fmtUsd(total)}` : '';
            },
          },
        },
      },
    },
  });
}

// ============== Heatmap (date filter only) ==============
function renderHeatmap(dMin, dMax) {
  const grid = document.getElementById('heatmap');
  grid.innerHTML = '';
  if (!dMin) return;
  const byDate = new Map((DATA.dailyActivity || [])
    .filter(d => d.date >= dMin && d.date <= dMax)
    .map(d => [d.date, d.sessionCount]));
  const start = new Date(dMin);
  const end   = new Date(dMax);
  const cur = new Date(start); cur.setDate(cur.getDate() - cur.getDay());
  const cells = [];
  while (cur <= end) {
    const iso = cur.toISOString().slice(0,10);
    cells.push({ date: iso, dow: cur.getDay(), sessions: byDate.get(iso) || 0, before: cur < start });
    cur.setDate(cur.getDate() + 1);
  }
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
}

// ============== Hour-of-day ==============
function renderHourChart() {
  destroyChart('hour');
  // hourCounts has no date attribution in stats-cache; filter only works for the full window.
  const hours = Array.from({length:24}, (_,i) => i);
  const data = hours.map(h => (DATA.hourCounts||{})[h] || (DATA.hourCounts||{})[String(h)] || 0);
  charts.hour = new Chart(document.getElementById('hourChart'), {
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
}

// ============== MCP server (single section) ==============
function aggregateMCPFromEvents(filteredEvents) {
  if (!DBX) return null;
  // Match the primary server selected on Python side.
  const primaryServerName = DBX.name;
  const serverIdx = DIMS.servers.indexOf(primaryServerName);
  if (serverIdx < 0) return null;

  const toolCounts = {};
  const dailyTotal = {};
  let totalCalls = 0;
  for (const ev of filteredEvents) {
    if (ev[9] !== 1) continue;     // kind != MCP
    if (ev[3] !== serverIdx) continue;
    const toolName = DIMS.tools[ev[4]] || '(none)';
    toolCounts[toolName] = (toolCounts[toolName] || 0) + 1;
    const date = DIMS.dates[ev[0]];
    if (date) dailyTotal[date] = (dailyTotal[date] || 0) + 1;
    totalCalls++;
  }
  return {
    name: primaryServerName,
    displayName: DBX.displayName,
    totalCalls,
    distinctTools: Object.keys(toolCounts).length,
    toolCounts,
    dailyTotal,
  };
}

function renderMCP(mcpAgg, filteredTotalCalls) {
  if (!mcpAgg) return;
  const days = Object.keys(mcpAgg.dailyTotal).sort();
  const peakDay = days.reduce((best, d) => mcpAgg.dailyTotal[d] > (mcpAgg.dailyTotal[best]||0) ? d : best, days[0] || '');
  document.getElementById('dbxTotal').textContent = fmt(mcpAgg.totalCalls);
  document.getElementById('dbxTotalDelta').textContent = filteredTotalCalls > 0
    ? `${(mcpAgg.totalCalls/filteredTotalCalls*100).toFixed(1)}% of tool calls in view`
    : '—';
  document.getElementById('dbxDistinct').textContent = mcpAgg.distinctTools;
  document.getElementById('dbxDistinctDelta').textContent = `tools exercised on this server`;
  document.getElementById('dbxActiveDays').textContent = days.length;
  document.getElementById('dbxActiveDaysDelta').textContent = days.length
    ? `${days[0]} → ${days[days.length-1]}` : '';
  if (peakDay && mcpAgg.dailyTotal[peakDay]) {
    document.getElementById('dbxPeak').innerHTML = `${fmt(mcpAgg.dailyTotal[peakDay])}<small>calls</small>`;
    document.getElementById('dbxPeakDelta').textContent = `on ${peakDay}`;
  } else {
    document.getElementById('dbxPeak').textContent = '0';
    document.getElementById('dbxPeakDelta').textContent = '';
  }

  destroyChart('dbxTools');
  const entries = Object.entries(mcpAgg.toolCounts).sort((a,b) => b[1] - a[1]);
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
  charts.dbxTools = new Chart(document.getElementById('dbxToolsChart'), {
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

  destroyChart('dbxDaily');
  if (days.length) {
    const minD = new Date(days[0]);
    const maxD = new Date(days[days.length-1]);
    const allDays = [];
    for (let d = new Date(minD); d <= maxD; d.setDate(d.getDate()+1)) {
      allDays.push(d.toISOString().slice(0,10));
    }
    const dailyValues = allDays.map(d => mcpAgg.dailyTotal[d] || 0);
    charts.dbxDaily = new Chart(document.getElementById('dbxDailyChart'), {
      type: 'bar',
      data: {
        labels: allDays,
        datasets: [{
          label: `${mcpAgg.displayName} MCP calls`,
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
}

// ============== Web research ==============
function aggregateResearchFromEvents(filteredEvents) {
  let wf = 0, ws = 0, re_ = 0;
  const wfByDay = {}, wsByDay = {}, reByDay = {};
  for (const ev of filteredEvents) {
    const kind = ev[9];
    if (kind !== 2 && kind !== 3 && kind !== 4) continue;
    const date = DIMS.dates[ev[0]];
    if (kind === 2) { wf++; if (date) wfByDay[date] = (wfByDay[date]||0)+1; }
    else if (kind === 3) { ws++; if (date) wsByDay[date] = (wsByDay[date]||0)+1; }
    else if (kind === 4) { re_++; if (date) reByDay[date] = (reByDay[date]||0)+1; }
  }
  return { webfetchTotal: wf, websearchTotal: ws, researchTotal: re_,
           webfetchByDay: wfByDay, websearchByDay: wsByDay, researchByDay: reByDay };
}

function renderResearch(agg, filteredTotalCalls) {
  const total = agg.webfetchTotal + agg.websearchTotal + agg.researchTotal;
  document.getElementById('resTotal').textContent = fmt(total);
  document.getElementById('resTotalDelta').textContent = filteredTotalCalls > 0
    ? `~${(total/filteredTotalCalls*100).toFixed(1)}% of tool calls in view` : '—';
  document.getElementById('resWF').textContent = fmt(agg.webfetchTotal);
  document.getElementById('resWFDelta').textContent = filtersActive()
    ? `(domains list reflects full corpus)`
    : `${RESEARCH.uniqueDomains} unique domains visited`;
  document.getElementById('resWS').textContent = fmt(agg.websearchTotal);
  document.getElementById('resWSDelta').textContent = filtersActive()
    ? `(query themes reflect full corpus)`
    : `${fmt(RESEARCH.uniqueQueries)} distinct queries`;
  document.getElementById('resRE').textContent = fmt(agg.researchTotal);
  document.getElementById('resREDelta').textContent = `Spawned subagent for deep research`;

  // Domain chart is from the pre-aggregated RESEARCH.domainCounts (not filterable
  // without shipping the URL strings — that would double the HTML size).
  destroyChart('resDomains');
  const entries = Object.entries(RESEARCH.domainCounts).sort((a,b) => b[1] - a[1]);
  const labels = entries.map(([k]) => k);
  const values = entries.map(([,v]) => v);
  if (labels.length) {
    const palette = ['#CC785C','#BD6F54','#AE664D','#A05E47','#915541','#834D3B','#7A4F37','#6D4737','#604138','#544039','#4A3E37','#5C4838','#7F6650','#8A745E','#9A8770','#AA9A82','#B7AB95','#C3BCA8','#CECCBA','#D8D9CB'];
    charts.resDomains = new Chart(document.getElementById('resDomainsChart'), {
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

  destroyChart('resDaily');
  const allDayKeys = new Set([
    ...Object.keys(agg.webfetchByDay),
    ...Object.keys(agg.websearchByDay),
    ...Object.keys(agg.researchByDay),
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
      { label: 'WebFetch',        data: fullDays.map(d => agg.webfetchByDay[d]  || 0), backgroundColor: '#CC785C', stack:'r', borderWidth:0, borderRadius:2 },
      { label: 'WebSearch',       data: fullDays.map(d => agg.websearchByDay[d] || 0), backgroundColor: '#7A4F37', stack:'r', borderWidth:0, borderRadius:2 },
      { label: 'research-expert', data: fullDays.map(d => agg.researchByDay[d]  || 0), backgroundColor: '#D4A27F', stack:'r', borderWidth:0, borderRadius:2 },
    ];
    charts.resDaily = new Chart(document.getElementById('resDailyChart'), {
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
}

// ============== Word cloud (static — run once) ==============
function renderWordcloud() {
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
}

// ============== Topics (Haiku-classified first prompts) ==============
// Like eventMatches but ignores the classification filter — the topics chart
// shows the distribution under all OTHER filters, so users can see how
// projects/dates/etc. shift the topic mix.
function eventMatchesIgnoreClass(ev) {
  if (ev[0] < state.dateMin || ev[0] > state.dateMax) return false;
  if (state.models.size   && !state.models.has(ev[2]))   return false;
  if (state.projects.size && !state.projects.has(ev[1])) return false;
  if (state.servers.size  && !state.servers.has(ev[3]))  return false;
  if (state.skills.size   && !state.skills.has(ev[5]))   return false;
  return true;
}
function sessionMatchesIgnoreClass(s) {
  if (s.start) {
    const di = DATES_INDEX.get(s.start);
    if (di !== undefined && (di < state.dateMin || di > state.dateMax)) return false;
  }
  if (state.projects.size && !state.projects.has(s.proj)) return false;
  return true;
}

function aggregateTopics() {
  const sessionsByTopic = {};
  const eventsByTopic = {};
  for (const s of SESSIONS) {
    if (!sessionMatchesIgnoreClass(s)) continue;
    const t = DIMS.classifications[s.cls] || '(unclassified)';
    sessionsByTopic[t] = (sessionsByTopic[t] || 0) + 1;
  }
  for (const ev of EVENTS) {
    if (!eventMatchesIgnoreClass(ev)) continue;
    const t = DIMS.classifications[sessCls[ev[8]] || 0] || '(unclassified)';
    eventsByTopic[t] = (eventsByTopic[t] || 0) + 1;
  }
  return { sessionsByTopic, eventsByTopic };
}

function renderTopics() {
  if (!document.getElementById('topicChart')) return;   // section hidden
  destroyChart('topicChart');
  const { sessionsByTopic, eventsByTopic } = aggregateTopics();
  const all = new Set([...Object.keys(sessionsByTopic), ...Object.keys(eventsByTopic)]);
  all.delete('(unclassified)');
  const rows = [...all].map(t => ({
    topic: t,
    sessions: sessionsByTopic[t] || 0,
    events: eventsByTopic[t] || 0,
  })).sort((a, b) => b.sessions - a.sessions);

  const classifiedSessions = rows.reduce((s, r) => s + r.sessions, 0);
  const unc = sessionsByTopic['(unclassified)'] || 0;
  const totalSessionsInView = classifiedSessions + unc;

  document.getElementById('topicSessions').textContent = fmt(classifiedSessions);
  document.getElementById('topicSessionsDelta').textContent =
    `of ${fmt(totalSessionsInView)} sessions in view`;
  if (rows.length) {
    document.getElementById('topicTop').textContent = rows[0].topic;
    document.getElementById('topicTopDelta').textContent = classifiedSessions > 0
      ? `${fmt(rows[0].sessions)} sessions (${(rows[0].sessions/classifiedSessions*100).toFixed(0)}%)`
      : '—';
  } else {
    document.getElementById('topicTop').textContent = '—';
    document.getElementById('topicTopDelta').textContent = '';
  }
  document.getElementById('topicCount').textContent = String(rows.length);
  const totalCats = (DIMS.classifications.length - 1);
  document.getElementById('topicCountDelta').textContent = `of ${totalCats} categories`;
  document.getElementById('topicUnc').textContent = fmt(unc);
  document.getElementById('topicUncDelta').textContent = totalSessionsInView > 0
    ? `${(unc/totalSessionsInView*100).toFixed(0)}% of sessions in view`
    : '—';

  if (!rows.length) return;
  const labels = rows.map(r => r.topic);
  charts.topicChart = new Chart(document.getElementById('topicChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        // Sessions and tool-call counts differ by ~30× so they share a Y axis
        // but use independent X scales — otherwise the session bars vanish
        // beside the tool-call bars.
        { label: 'Sessions',   data: rows.map(r => r.sessions), xAxisID: 'xSessions', backgroundColor: '#CC785C', borderWidth: 0, borderRadius: 3 },
        { label: 'Tool calls', data: rows.map(r => r.events),   xAxisID: 'xCalls',    backgroundColor: '#7A4F37', borderWidth: 0, borderRadius: 3 },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        xSessions: {
          position: 'bottom',
          grid: { display: false },
          ticks: { callback: v => compact(v), color: '#CC785C' },
          title: { display: true, text: 'Sessions', color: '#CC785C', font: { size: 11, weight: '600' } },
        },
        xCalls: {
          position: 'top',
          grid: { color: '#EDE9DC' },
          ticks: { callback: v => compact(v), color: '#7A4F37' },
          title: { display: true, text: 'Tool calls', color: '#7A4F37', font: { size: 11, weight: '600' } },
        },
        y: { grid: { display: false }, ticks: { font: { size: 12 } } },
      },
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw)}`,
            afterBody: items => {
              const row = rows[items[0].dataIndex];
              const ratio = row.sessions > 0 ? (row.events/row.sessions).toFixed(1) : '—';
              return `${ratio} tool calls per session`;
            },
          },
        },
      },
    },
  });
}

// ============== Skills (parsed from <command-name>) ==============
// Like eventMatches but ignores the skills filter so the chart keeps its
// shape when the user picks specific skills.
function eventMatchesIgnoreSkill(ev) {
  if (ev[0] < state.dateMin || ev[0] > state.dateMax) return false;
  if (state.models.size   && !state.models.has(ev[2]))   return false;
  if (state.projects.size && !state.projects.has(ev[1])) return false;
  if (state.servers.size  && !state.servers.has(ev[3]))  return false;
  if (state.classes.size) {
    const cls = sessCls[ev[8]];
    if (cls === undefined || !state.classes.has(cls)) return false;
  }
  return true;
}

// Claude Code's built-in slash commands. Detected exactly the same way as
// user-defined skills (via the <command-name> tag) so we filter them out at
// render time. The filter UI keeps them so users can still slice the rest of
// the dashboard by "what was I doing while clearing context".
const SLASH_BUILTINS = new Set([
  'clear', 'compact', 'model', 'init', 'exit', 'help', 'status', 'memory',
  'resume', 'quit', 'save', 'logout', 'mcp', 'doctor', 'ide', 'terminal-setup',
  'context', 'simplify', 'cost', 'config', 'login', 'feedback', 'bug',
  'fast', 'loop', 'remember', 'permissions', 'agents', 'output-style',
  'continue', 'release-notes', 'usage', 'rewind', 'add-dir',
]);

function aggregateSkills() {
  const callsBySkill = {};
  const sessionsBySkill = {};       // skill -> Set<session_idx>
  let totalSkillCalls = 0;
  for (const ev of EVENTS) {
    if (ev[5] === 0) continue;            // no skill in scope
    if (!eventMatchesIgnoreSkill(ev)) continue;
    const sk = DIMS.skills[ev[5]] || '(none)';
    if (SLASH_BUILTINS.has(sk)) continue;  // hide built-ins from the chart
    callsBySkill[sk] = (callsBySkill[sk] || 0) + 1;
    if (!sessionsBySkill[sk]) sessionsBySkill[sk] = new Set();
    sessionsBySkill[sk].add(ev[8]);
    totalSkillCalls++;
  }
  return { callsBySkill, sessionsBySkill, totalSkillCalls };
}

function renderSkills() {
  if (!document.getElementById('skillChart')) return;     // section hidden
  destroyChart('skillChart');
  const { callsBySkill, sessionsBySkill, totalSkillCalls } = aggregateSkills();
  const rows = Object.keys(callsBySkill).map(sk => ({
    skill: sk,
    calls: callsBySkill[sk],
    sessions: sessionsBySkill[sk]?.size || 0,
  })).sort((a, b) => b.calls - a.calls);

  // KPI cards
  const unique = rows.length;
  const allSkillSessions = new Set();
  for (const s of Object.values(sessionsBySkill)) for (const id of s) allSkillSessions.add(id);
  const totalEventsInView = (window._filteredEvents || []).length;

  document.getElementById('skillCalls').textContent = fmt(totalSkillCalls);
  document.getElementById('skillCallsDelta').textContent = totalEventsInView > 0
    ? `${(totalSkillCalls / totalEventsInView * 100).toFixed(1)}% of tool calls in view`
    : '—';
  document.getElementById('skillUnique').textContent = String(unique);
  const totalUserSkills = DIMS.skills
    .slice(1)                              // drop the "(none)" placeholder
    .filter(sk => !SLASH_BUILTINS.has(sk))
    .length;
  document.getElementById('skillUniqueDelta').textContent =
    `of ${totalUserSkills} user-defined skills detected`;
  if (rows.length) {
    document.getElementById('skillTop').textContent = rows[0].skill;
    document.getElementById('skillTopDelta').textContent =
      `${fmt(rows[0].calls)} calls · ${rows[0].sessions} sessions`;
  } else {
    document.getElementById('skillTop').textContent = '—';
    document.getElementById('skillTopDelta').textContent = '';
  }
  document.getElementById('skillSess').textContent = fmt(allSkillSessions.size);
  document.getElementById('skillSessDelta').textContent = `Sessions invoking at least one skill`;

  if (!rows.length) return;
  const TOP_N = 15;
  const top = rows.slice(0, TOP_N);
  const labels = top.map(r => r.skill);

  charts.skillChart = new Chart(document.getElementById('skillChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        // Tool calls (primary, dark brown) on the larger top axis,
        // sessions (coral) on the smaller bottom axis — same dual-axis
        // pattern as the Topics chart so the two read consistently.
        { label: 'Tool calls', data: top.map(r => r.calls),    xAxisID: 'xCalls',    backgroundColor: '#7A4F37', borderWidth: 0, borderRadius: 3 },
        { label: 'Sessions',   data: top.map(r => r.sessions), xAxisID: 'xSessions', backgroundColor: '#CC785C', borderWidth: 0, borderRadius: 3 },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        xCalls: {
          position: 'top',
          grid: { color: '#EDE9DC' },
          ticks: { callback: v => compact(v), color: '#7A4F37' },
          title: { display: true, text: 'Tool calls', color: '#7A4F37', font: { size: 11, weight: '600' } },
        },
        xSessions: {
          position: 'bottom',
          grid: { display: false },
          ticks: { callback: v => compact(v), color: '#CC785C' },
          title: { display: true, text: 'Sessions', color: '#CC785C', font: { size: 11, weight: '600' } },
        },
        y: { grid: { display: false }, ticks: { font: { size: 12 } } },
      },
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw)}`,
            afterBody: items => {
              const row = top[items[0].dataIndex];
              const ratio = row.sessions > 0 ? (row.calls/row.sessions).toFixed(1) : '—';
              return `${ratio} tool calls per session using this skill`;
            },
          },
        },
      },
    },
  });
}

// ============== Programming languages ==============
function aggregateLangsFromEvents(filteredEvents) {
  const reads = {}, edits = {};
  for (const ev of filteredEvents) {
    const kind = ev[9];
    if (kind === 5 && ev[6] > 0) {
      const lang = DIMS.langs[ev[6]];
      reads[lang] = (reads[lang] || 0) + 1;
    } else if (kind === 6 && ev[7] > 0) {
      const lang = DIMS.langs[ev[7]];
      edits[lang] = (edits[lang] || 0) + 1;
    }
  }
  const all = new Set([...Object.keys(reads), ...Object.keys(edits)]);
  const rows = [];
  for (const lang of all) {
    rows.push({
      lang, reads: reads[lang]||0, edits: edits[lang]||0,
      uniqRead: 0, uniqEdit: 0,   // not derivable from row-level events
    });
  }
  rows.sort((a,b) => (b.reads+b.edits) - (a.reads+a.edits));
  return {
    totalReads: Object.values(reads).reduce((s,v) => s+v, 0),
    totalWrites: Object.values(edits).reduce((s,v) => s+v, 0),
    uniqueRead: 0,   // hidden in filter mode
    uniqueWrite: 0,
    rows,
  };
}

function renderLangs(agg) {
  destroyChart('langChart');
  // For unfiltered view, use the original LANGS payload (so unique-file counts show).
  const useOriginal = !filtersActive();
  const view = useOriginal ? LANGS : agg;

  document.getElementById('langReads').textContent       = fmt(view.totalReads);
  if (view.totalReads + view.totalWrites > 0) {
    document.getElementById('langReadsDelta').textContent  =
      `${(view.totalReads/(view.totalReads+view.totalWrites)*100).toFixed(0)}% of file activity`;
  } else {
    document.getElementById('langReadsDelta').textContent = '—';
  }
  document.getElementById('langEdits').textContent       = fmt(view.totalWrites);
  if (view.totalReads > 0) {
    document.getElementById('langEditsDelta').textContent  =
      `${(view.totalWrites/view.totalReads).toFixed(2)}× ratio vs reads`;
  } else {
    document.getElementById('langEditsDelta').textContent = '—';
  }
  if (useOriginal) {
    document.getElementById('langUniqRead').textContent    = fmt(view.uniqueRead);
    document.getElementById('langUniqReadDelta').textContent = view.uniqueRead > 0
      ? `${(view.totalReads/view.uniqueRead).toFixed(1)} avg reads per file` : '—';
    document.getElementById('langUniqEdit').textContent    = fmt(view.uniqueWrite);
    document.getElementById('langUniqEditDelta').textContent = view.uniqueWrite > 0
      ? `${(view.totalWrites/view.uniqueWrite).toFixed(1)} avg edits per file` : '—';
  } else {
    document.getElementById('langUniqRead').textContent = '—';
    document.getElementById('langUniqReadDelta').textContent = 'not available in filtered view';
    document.getElementById('langUniqEdit').textContent = '—';
    document.getElementById('langUniqEditDelta').textContent = 'not available in filtered view';
  }

  const rows = (view.rows || [])
    .filter(r => r.lang !== 'Other / ?' && r.lang !== '(none)')
    .slice(0, 14);
  if (!rows.length) return;
  const labels = rows.map(r => r.lang);
  charts.langChart = new Chart(document.getElementById('langChart'), {
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
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw)} calls` } },
      },
    },
  });
}

// ============== Model breakdown table ==============
function renderModelTable(usage) {
  const tbody = document.getElementById('modelTable');
  const tfoot = document.getElementById('modelTableFoot');
  tbody.innerHTML = '';
  tfoot.innerHTML = '';
  const rows = Object.entries(usage)
    .map(([m, v]) => ({
      model: m,
      input: v.inputTokens||0,
      output: v.outputTokens||0,
      cacheRead: v.cacheReadInputTokens||0,
      cacheCreate: v.cacheCreationInputTokens||0,
      total: (v.inputTokens||0)+(v.outputTokens||0)+(v.cacheReadInputTokens||0)+(v.cacheCreationInputTokens||0),
    }))
    .filter(r => r.total > 0)
    .sort((a,b) => b.total - a.total);
  const grand = rows.reduce((s, r) => s + r.total, 0);
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
}

// ============== Apply filters orchestrator ==============
function applyFilters() {
  // 1) Compute filtered events once and cache for downstream use.
  window._filteredEvents = EVENTS.filter(eventMatches);
  const filteredEvents = window._filteredEvents;

  // 2) KPI block also computes the date-clipped DATA view we re-use.
  const { dailyClipped, dMin, dMax } = renderTopKPIs();

  // 3) Model usage / token charts.
  const usage = computeModelUsageFiltered(dailyClipped, state.models);
  renderDailyChart(dailyClipped);
  renderDonut(usage);
  renderClassChart(usage);
  renderMonthly(dailyClipped);

  // 4) Activity charts.
  renderHeatmap(dMin, dMax);
  renderHourChart();

  // 5) MCP / Research / Languages from EVENTS.
  renderMCP(aggregateMCPFromEvents(filteredEvents), filteredEvents.length);
  renderResearch(aggregateResearchFromEvents(filteredEvents), filteredEvents.length);
  renderTopics();
  renderSkills();
  renderLangs(aggregateLangsFromEvents(filteredEvents));

  // 6) Model breakdown table.
  renderModelTable(usage);

  // 7) Filter UI count badges + warning banner.
  updateFilterBadges();
  const warn = document.getElementById('filterWarn');
  warn.classList.toggle('hidden', !filtersActive());
  warn.textContent = state.projects.size > 0
    ? "Filtered view · token charts ignore project (stats-cache has no per-project attribution)"
    : "Filtered view";
}

// ============== Filter UI population ==============
function buildPopover(popId, options, stateSet, allowSearch) {
  const pop = document.getElementById(popId);
  pop.innerHTML = '';
  if (allowSearch && options.length > 8) {
    const inp = document.createElement('input');
    inp.type = 'text'; inp.className = 'fb-search';
    inp.placeholder = 'Search…';
    inp.addEventListener('input', e => {
      const q = e.target.value.toLowerCase();
      pop.querySelectorAll('label').forEach(l => {
        const text = l.dataset.text || '';
        l.style.display = text.toLowerCase().includes(q) ? 'flex' : 'none';
      });
    });
    pop.appendChild(inp);
  }
  for (const opt of options) {
    const lbl = document.createElement('label');
    lbl.dataset.text = opt.label;
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = String(opt.value);
    cb.checked = stateSet.has(opt.value);
    cb.addEventListener('change', () => {
      if (cb.checked) stateSet.add(opt.value);
      else stateSet.delete(opt.value);
      applyFilters();
    });
    lbl.appendChild(cb);
    const span = document.createElement('span'); span.textContent = opt.label;
    lbl.appendChild(span);
    pop.appendChild(lbl);
  }
  const actions = document.createElement('div'); actions.className = 'fb-actions';
  const selAll = document.createElement('button'); selAll.textContent = 'Select all';
  selAll.addEventListener('click', () => {
    stateSet.clear();
    options.forEach(o => stateSet.add(o.value));
    pop.querySelectorAll('input[type=checkbox]').forEach(cb => cb.checked = true);
    applyFilters();
  });
  const clr = document.createElement('button'); clr.textContent = 'Clear';
  clr.addEventListener('click', () => {
    stateSet.clear();
    pop.querySelectorAll('input[type=checkbox]').forEach(cb => cb.checked = false);
    applyFilters();
  });
  actions.appendChild(selAll); actions.appendChild(clr);
  pop.appendChild(actions);
}

function updateFilterBadges() {
  const upd = (id, set) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = String(set.size);
    el.classList.toggle('zero', set.size === 0);
  };
  upd('cntModels',   state.models);
  upd('cntProjects', state.projects);
  upd('cntServers',  state.servers);
  upd('cntSkills',   state.skills);
  upd('cntClasses',  state.classes);
}

function populateAllPopovers() {
  // Models: labels derived at runtime by shortOf() inside walk_with_events,
  // already in DIMS.models as short names.
  const modelOpts = DIMS.models.map((label, i) => ({ label, value: i }))
    .sort((a, b) => a.label.localeCompare(b.label));
  buildPopover('popModels', modelOpts, state.models, false);

  const projectOpts = DIMS.projects.map((label, i) => ({ label, value: i }))
    .sort((a, b) => a.label.localeCompare(b.label));
  buildPopover('popProjects', projectOpts, state.projects, projectOpts.length > 8);

  // Servers: skip index 0 which is "(none)".
  const serverOpts = DIMS.servers.slice(1).map((label, i) => ({ label, value: i + 1 }))
    .sort((a, b) => a.label.localeCompare(b.label));
  buildPopover('popServers', serverOpts, state.servers, false);

  // Skills: skip the "(none)" placeholder at index 0, then drop Claude
  // Code built-in slash commands so the dropdown only lists user-defined
  // skills (matches the chart). Built-ins are still detected and stored in
  // DIMS.skills so any cached state remains consistent; we just hide them
  // from the filter UI.
  const skillOpts = DIMS.skills.map((label, i) => ({ label, value: i }))
    .filter(o => o.value !== 0 && !SLASH_BUILTINS.has(o.label))
    .sort((a, b) => a.label.localeCompare(b.label));
  buildPopover('popSkills', skillOpts, state.skills, skillOpts.length > 8);

  // Classifications: skip index 0 which is "(unclassified)".
  const clsOpts = DIMS.classifications.slice(1).map((label, i) => ({ label, value: i + 1 }))
    .sort((a, b) => a.label.localeCompare(b.label));
  buildPopover('popClasses', clsOpts, state.classes, false);
}

// ============== Date input wiring ==============
function setupDateInputs() {
  const minInput = document.getElementById('fbDateMin');
  const maxInput = document.getElementById('fbDateMax');
  const first = DIMS.dates[0] || '';
  const last  = DIMS.dates[DIMS.dates.length - 1] || '';
  minInput.value = first; minInput.min = first; minInput.max = last;
  maxInput.value = last;  maxInput.min = first; maxInput.max = last;
  minInput.addEventListener('change', e => {
    const v = e.target.value;
    // Find closest date index >= v
    let idx = DIMS.dates.findIndex(d => d >= v);
    if (idx < 0) idx = 0;
    state.dateMin = idx;
    if (state.dateMin > state.dateMax) state.dateMax = state.dateMin;
    applyFilters();
  });
  maxInput.addEventListener('change', e => {
    const v = e.target.value;
    // Find closest date index <= v
    let idx = -1;
    for (let i = DIMS.dates.length - 1; i >= 0; i--) {
      if (DIMS.dates[i] <= v) { idx = i; break; }
    }
    if (idx < 0) idx = DIMS.dates.length - 1;
    state.dateMax = idx;
    if (state.dateMax < state.dateMin) state.dateMin = state.dateMax;
    applyFilters();
  });
}

// ============== Popover open/close ==============
function setupPopoverToggles() {
  document.querySelectorAll('.fb-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const targetId = btn.dataset.popover;
      const target = document.getElementById(targetId);
      // Close all others.
      document.querySelectorAll('.fb-popover').forEach(p => {
        if (p !== target) p.classList.remove('open');
      });
      target.classList.toggle('open');
    });
  });
  document.addEventListener('click', e => {
    if (!e.target.closest('.filter-group')) {
      document.querySelectorAll('.fb-popover').forEach(p => p.classList.remove('open'));
    }
  });
}

function setupReset() {
  document.getElementById('fbReset').addEventListener('click', () => {
    state.dateMin = 0; state.dateMax = Math.max(0, TOTAL_DATES - 1);
    state.models.clear(); state.projects.clear(); state.servers.clear();
    state.skills.clear(); state.classes.clear();
    document.getElementById('fbDateMin').value = DIMS.dates[0] || '';
    document.getElementById('fbDateMax').value = DIMS.dates[DIMS.dates.length-1] || '';
    populateAllPopovers();
    applyFilters();
  });
}

// ============== Export to PDF ==============
(function() {
  const allCharts = () => (typeof Chart !== 'undefined' && Chart.instances)
    ? Object.values(Chart.instances) : [];
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
      } catch (_) {}
    }
  }
  window.addEventListener('beforeprint', redrawAll);
  window.addEventListener('afterprint', redrawAll);
  const mql = (typeof window.matchMedia === 'function') && window.matchMedia('print');
  if (mql) {
    const cb = (e) => { if (e.matches) redrawAll(); };
    if (typeof mql.addEventListener === 'function') mql.addEventListener('change', cb);
    else if (typeof mql.addListener === 'function') mql.addListener(cb);
  }
  const btn = document.getElementById('exportPdfBtn');
  if (btn) {
    btn.addEventListener('click', () => {
      redrawAll();
      requestAnimationFrame(() => requestAnimationFrame(() => window.print()));
    });
  }
})();

// ============== Boot ==============
populateAllPopovers();
setupDateInputs();
setupPopoverToggles();
setupReset();
renderWordcloud();
applyFilters();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    sys.exit(main())
