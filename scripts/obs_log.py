#!/usr/bin/env python3
"""
Obsidian logging utility for AeroStrike agent.

Usage:
    python3 scripts/obs_log.py log "action title" --what "..." --files "a.py,b.py" --result "success"
    python3 scripts/obs_log.py read "Sessions/2025-01-15.md"
    python3 scripts/obs_log.py search "raycast sensor"
    python3 scripts/obs_log.py task-done "Implement velocity control"
    python3 scripts/obs_log.py problem "CUDA OOM" --severity blocker --desc "..."
"""

import sys
import os
import argparse
import json
import requests
from datetime import datetime, date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

API_BASE = f"http://localhost:{os.getenv('OBSIDIAN_PORT', '27123')}"
API_KEY  = os.getenv("OBSIDIAN_API_KEY", "")
HEADERS  = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "text/markdown",
}


# ──────────────────────────────────────────────────────────────────────────────
# Core API calls
# ──────────────────────────────────────────────────────────────────────────────

def _get(path: str) -> str:
    """Read a note."""
    r = requests.get(f"{API_BASE}/vault/{path}", headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.text


def _put(path: str, content: str) -> None:
    """Create or overwrite a note."""
    r = requests.put(
        f"{API_BASE}/vault/{path}",
        data=content.encode("utf-8"),
        headers=HEADERS,
        timeout=10,
    )
    r.raise_for_status()


def _patch(path: str, content: str) -> None:
    """Append to a note (creates if missing)."""
    try:
        existing = _get(path)
        _put(path, existing + content)
    except requests.HTTPError:
        _put(path, content)


def _search(query: str) -> list[dict]:
    """Full-text search across vault."""
    r = requests.post(
        f"{API_BASE}/search/simple/",
        params={"query": query},
        headers={**HEADERS, "Content-Type": "application/json"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# ──────────────────────────────────────────────────────────────────────────────
# High-level operations
# ──────────────────────────────────────────────────────────────────────────────

def log_action(title: str, what: str = "", files: str = "",
               result: str = "", problems: str = "", next_step: str = "") -> None:
    """Append a timestamped action entry to today's session log."""
    today     = date.today().isoformat()
    now       = datetime.now().strftime("%H:%M")
    session   = f"Sessions/{today}.md"
    file_list = ", ".join(f"`{f.strip()}`" for f in files.split(",") if f.strip())

    # Ensure session file header exists
    try:
        _get(session)
    except requests.HTTPError:
        header = (
            f"---\ntags: [session]\ndate: {today}\n---\n\n"
            f"# Session {today}\n\n"
        )
        _put(session, header)

    entry = f"""
## {now} — {title}

**Task:** {what}
**Files changed:** {file_list or '—'}
**Result:** {result or '—'}
**Problems:** {problems or 'none'}
**Next:** {next_step or '—'}

---
"""
    _patch(session, entry)
    print(f"[obs_log] ✓ Logged: {title}")


def read_note(path: str) -> str:
    """Print a note to stdout."""
    content = _get(path)
    print(content)
    return content


def search_vault(query: str) -> None:
    """Search and print results."""
    results = _search(query)
    if not results:
        print("No results found.")
        return
    for item in results[:10]:
        print(f"  • {item.get('filename', '')}  (score: {item.get('score', 0):.2f})")


def mark_task_done(task_text: str) -> None:
    """Move a task from active to done in Tasks/active.md."""
    today   = date.today().isoformat()
    content = _get("Tasks/active.md")
    updated = content.replace(
        f"- [ ] {task_text}",
        f"- [x] {task_text} ✅ {today}"
    )
    if updated == content:
        print(f"[obs_log] ⚠ Task not found: '{task_text}'")
    else:
        _put("Tasks/active.md", updated)
        print(f"[obs_log] ✓ Marked done: {task_text}")


def create_problem(title: str, severity: str = "major",
                   desc: str = "", error: str = "") -> None:
    """Create a new problem note."""
    today = date.today().isoformat()
    slug  = title.lower().replace(" ", "-")[:40]
    path  = f"Problems/{slug}.md"
    content = (
        f"---\ntags: [problem]\nstatus: open\nseverity: {severity}\n"
        f"created: {today}\n---\n\n"
        f"# Problem: {title}\n\n"
        f"## Description\n{desc}\n\n"
        f"## Error\n```\n{error}\n```\n\n"
        f"## Attempted solutions\n\n## Solution\n"
    )
    _put(path, content)
    # Also log reference in today's session
    log_action(
        f"Problem logged: {title}",
        what="Encountered blocker",
        result="blocked",
        problems=f"See [[Problems/{slug}]]",
    )
    print(f"[obs_log] ✓ Problem created: {path}")


def update_environment(component: str, version: str, notes: str = "") -> None:
    """Update version entry in Environment/stack.md."""
    content = _get("Environment/stack.md")
    marker  = f"| {component} |"
    new_row = f"| {component} | {version} | {notes} |"
    if marker in content:
        lines   = content.split("\n")
        updated = "\n".join(
            new_row if marker in line else line for line in lines
        )
        _put("Environment/stack.md", updated)
        print(f"[obs_log] ✓ Updated {component} → {version}")
    else:
        # Append new row before the last empty line
        _patch("Environment/stack.md", f"\n{new_row}\n")
        print(f"[obs_log] ✓ Added {component} = {version}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="AeroStrike Obsidian log utility")
    sub = p.add_subparsers(dest="cmd", required=True)

    # log
    lg = sub.add_parser("log")
    lg.add_argument("title")
    lg.add_argument("--what",     default="")
    lg.add_argument("--files",    default="")
    lg.add_argument("--result",   default="")
    lg.add_argument("--problems", default="")
    lg.add_argument("--next",     default="")

    # read
    rd = sub.add_parser("read")
    rd.add_argument("path")

    # search
    sr = sub.add_parser("search")
    sr.add_argument("query")

    # task-done
    td = sub.add_parser("task-done")
    td.add_argument("task")

    # problem
    pb = sub.add_parser("problem")
    pb.add_argument("title")
    pb.add_argument("--severity", default="major")
    pb.add_argument("--desc",     default="")
    pb.add_argument("--error",    default="")

    # env
    ev = sub.add_parser("env")
    ev.add_argument("component")
    ev.add_argument("version")
    ev.add_argument("--notes", default="")

    args = p.parse_args()

    if args.cmd == "log":
        log_action(args.title, args.what, args.files,
                   args.result, args.problems, args.next)
    elif args.cmd == "read":
        read_note(args.path)
    elif args.cmd == "search":
        search_vault(args.query)
    elif args.cmd == "task-done":
        mark_task_done(args.task)
    elif args.cmd == "problem":
        create_problem(args.title, args.severity, args.desc, args.error)
    elif args.cmd == "env":
        update_environment(args.component, args.version, args.notes)


if __name__ == "__main__":
    main()