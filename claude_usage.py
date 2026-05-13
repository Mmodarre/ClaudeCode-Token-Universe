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


# --------------------------------------------------------------------------- #
# Row-level events + sessions (for in-browser filtering)                      #
# --------------------------------------------------------------------------- #

_COMMAND_TAG_RE = re.compile(r"<command-name>\s*/?([\w\-:]+)\s*</command-name>", re.IGNORECASE)
_SLASH_CMD_RE = re.compile(r"^/([\w\-:]+)\b")


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
                        model_short = MODEL_SHORT.get(model_raw, model_raw or "(unknown)")
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


_CLASSIFICATION_CATEGORIES: list[str] = [
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
]


_CLASSIFY_PROMPT_HEADER = textwrap.dedent("""\
    You will be given a numbered list of first messages from coding sessions.
    Classify each message into EXACTLY ONE of these categories:

      Bug fix
      New feature
      Refactor
      Question / Q&A
      Exploration / Research
      Documentation
      Setup / Config
      Testing
      Operations
      Other

    Output JSON ONLY. No prose. No markdown fences. Schema:
      {"results": [{"i": <int>, "c": "<category>"}]}

    Rules:
      - i is the message number as given.
      - c MUST be one of the categories above, spelled exactly.
      - One result per input message.
      - If the message is a slash command (starts with /), classify by what the
        command name implies (e.g. /review → Question / Q&A, /init → Setup / Config).

    Messages:
    """)


def _classify_chunk(
    chunk: list[tuple[int, str]],
    claude_bin: str,
    timeout: int = 120,
) -> dict[int, str]:
    """Run one ``claude -p`` call over a batch of (index, text) pairs."""
    body = "\n".join(f"{i}. {text[:300].replace(chr(10), ' ').strip()}" for i, text in chunk)
    prompt = _CLASSIFY_PROMPT_HEADER + body
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
    valid = set(_CLASSIFICATION_CATEGORIES)
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
            c = "Other"
        mapping[i] = c
    return mapping


def classify_sessions(
    sessions: list[dict[str, Any]],
    claude_bin: str | None,
    chunk_size: int = 60,
    max_workers: int = 10,
) -> list[str]:
    """Classify each session's first prompt via parallel Haiku calls.

    Returns a list aligned with ``sessions`` mapping each to a category from
    ``_CLASSIFICATION_CATEGORIES`` (or ``"(unclassified)"`` if Haiku failed).
    """
    out: list[str] = ["(unclassified)"] * len(sessions)
    if not claude_bin or not sessions:
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
        futures = [ex.submit(_classify_chunk, c, claude_bin) for c in chunks]
        for fut in cf.as_completed(futures):
            for i, cat in fut.result().items():
                if 0 <= i < len(out):
                    out[i] = cat
    return out


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
    events: list[list[int]],
    sessions: list[dict[str, Any]],
    dims: dict[str, list[str]],
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
        events_json=json.dumps(events, default=str, separators=(",", ":")),
        sessions_json=json.dumps(sessions, default=str, separators=(",", ":")),
        dims_json=json.dumps(dims, default=str, separators=(",", ":")),
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
            print("Classifying session first-prompts via Haiku…", file=sys.stderr)
            cats = classify_sessions(sessions_rows, claude_bin)
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
    transcripts. Token totals include input, output, cache reads and cache writes — the full picture, not just
    the input+output figure shown in <span class="source-pill">/usage</span>.
    <span class="filter-warn hidden" id="filterWarn">Filtered view</span></p>
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
const EVENTS = {{ events_json|safe }};
const SESSIONS = {{ sessions_json|safe }};
const DIMS = {{ dims_json|safe }};

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
const MODEL_SHORT_JS = {
  'claude-opus-4-7':            'Opus 4.7',
  'claude-opus-4-6':            'Opus 4.6',
  'claude-opus-4-5-20251101':   'Opus 4.5',
  'claude-haiku-4-5-20251001':  'Haiku 4.5',
  'claude-sonnet-4-6':          'Sonnet 4.6',
  'claude-sonnet-4-5-20250929': 'Sonnet 4.5',
};
const SHORT_TO_RAW = Object.fromEntries(
  Object.entries(MODEL_SHORT_JS).map(([raw, short]) => [short, raw])
);
const colorOf = m => MODEL_COLOR[m] || '#888';
const shortOf = m => MODEL_SHORT_JS[m] || m;

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
  const orderedKnown = ['claude-opus-4-7','claude-opus-4-6','claude-opus-4-5-20251101',
                        'claude-haiku-4-5-20251001','claude-sonnet-4-6','claude-sonnet-4-5-20250929'];
  const orderedModels = orderedKnown.filter(m => modelsInUse.has(m))
    .concat([...modelsInUse].filter(m => !orderedKnown.includes(m)));
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
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw)} tokens` } },
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
  const orderedKnown = ['claude-opus-4-7','claude-opus-4-6','claude-opus-4-5-20251101',
                        'claude-haiku-4-5-20251001','claude-sonnet-4-6','claude-sonnet-4-5-20250929'];
  const orderedModels = orderedKnown.filter(m => modelsInUse.has(m))
    .concat([...modelsInUse].filter(m => !orderedKnown.includes(m)));
  const datasets = orderedModels.map(model => ({
    label: shortOf(model),
    data: months.map(m => byMonth[m][model] || 0),
    backgroundColor: colorOf(model),
    borderWidth: 0, borderRadius: 4, stack: 's',
  }));
  charts.monthly = new Chart(document.getElementById('monthlyChart'), {
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
  // Models: known short names from MODEL_SHORT_JS first, then any extra dims.
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

  // Skills: skip index 0.
  const skillOpts = DIMS.skills.slice(1).map((label, i) => ({ label, value: i + 1 }))
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
