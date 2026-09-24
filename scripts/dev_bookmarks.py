#!/usr/bin/env python3
"""Rebuild Chrome's "Dev apps (local)" bookmark folder from what is actually listening.

The portfolio runs a dozen local servers whose ports drift (a Vite dev server
takes the next free port, a compose stack is restarted elsewhere), so a folder
typed by hand goes stale within a week — the tau2-loop MLflow bookmark pointed
at :5601 long after tracking moved to the central platform on :5000.

This walks `lsof`, names each listener from its process working directory
(`KNOWN` below maps a repo to a label), and writes the folder into Chrome's
`Bookmarks` JSON grouped by project. Chrome rewrites that file from memory when
it exits, so it must not be running: `--quit-chrome` quits it with an Apple
Event (a clean quit, so the session is saved) and relaunches afterwards, which
restores the open tabs as long as "Continue where you left off" is set.

    uv run python scripts/dev_bookmarks.py --dry-run
    uv run python scripts/dev_bookmarks.py --quit-chrome

Nothing here is tau2-loop specific except that this is where the script lives.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

CHROME_DEFAULT = Path.home() / "Library/Application Support/Google/Chrome/Default"
FOLDER = "Dev apps (local)"

# A listener is named by the repo its process sits in. The value is the folder
# label; ports inside a project keep the order they are listed here.
KNOWN: dict[str, str] = {
    "tau2-loop": "tau2-loop",
    "DABStep-loop": "DABStep-loop",
    "ConvFinQA-agent": "ConvFinQA-agent",
    "DataAgentBench": "DataAgentBench",
    "transcript-rag-agent": "transcript-rag-agent",
    "productivity-tracker": "productivity-tracker",
    "value-invest-agent": "value-invest-agent",
    "nmp-dsci.github.io": "portfolio site",
    "nmp-central-ai": "nmp-central-ai (platform)",
}

# Containers report the Docker VM as their working directory, so they are named
# by compose project instead; `docker ps` supplies the mapping at run time.
DOCKER_LABELS: dict[str, str] = {
    "nmp-central": "nmp-central-ai (platform)",
    "data-qa-agent": "data-qa-agent (Docker)",
}

# Ports that are not browsable (databases, object storage, MCP stdio servers,
# editor helpers). Bookmarking them only produces dead tabs.
SKIP_PORTS = {5432, 9000, 9001, 8000}

# What a port is, when the title does not say it. Falls back to the page title.
HINTS: dict[int, str] = {
    5000: "MLflow",
    5050: "DbGate",
    3000: "Grafana",
    9090: "Prometheus",
}

# Ports whose project cannot be read from the process: a shared tool started
# from whichever repo happened to be current, or an API that answers only under
# a path. (label, name, path) — any field may be "" to keep the detected value.
PORT_RULES: dict[int, tuple[str, str, str]] = {
    4387: ("Lavish", "Lavish Editor", ""),
    8010: ("", "backend API docs", "docs"),
    8100: ("", "data-agent API docs", "docs"),
    8081: ("", "tau2-loop API docs", "docs"),
}

# Extra links that are not local listeners but belong beside them.
EXTRA: dict[str, list[tuple[str, str]]] = {
    "tau2-loop": [("MLflow · tau2-loop experiment", "http://localhost:5000/#/experiments")],
    "portfolio site": [("nmp-dsci.github.io (live)", "https://nmp-dsci.github.io/")],
}


def _sh(cmd: list[str], timeout: int = 15) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return out.stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def listeners() -> list[tuple[int, int]]:
    """(port, pid) for every TCP listener owned by this user, lowest port first."""
    found: dict[int, int] = {}
    for line in _sh(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"]).splitlines()[1:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        m = re.search(r":(\d+)$", parts[8])
        if not m:
            continue
        port = int(m.group(1))
        if port >= 10000 or port in SKIP_PORTS:
            continue
        found.setdefault(port, int(parts[1]))
    return sorted(found.items())


def cwd_of(pid: int) -> str:
    for line in _sh(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"]).splitlines():
        if line.startswith("n"):
            return line[1:]
    return ""


def docker_projects() -> dict[int, str]:
    """host port → compose project, for containers publishing a port."""
    fmt = "{{.Label \"com.docker.compose.project\"}}\t{{.Ports}}"
    out: dict[int, str] = {}
    for line in _sh(["docker", "ps", "--format", fmt]).splitlines():
        project, _, ports = line.partition("\t")
        if not project:
            continue
        for m in re.finditer(r"0\.0\.0\.0:(\d+)->", ports):
            out[int(m.group(1))] = project
    return out


def title(port: int) -> str:
    """The page's <title>. `localhost` first: a Vite dev server binds IPv6 only."""
    for host in ("localhost", "127.0.0.1"):
        html = _sh(["curl", "-s", "-m", "3", "-L", f"http://{host}:{port}/"], timeout=8)
        m = re.search(r"<title>([^<]{1,60})", html)
        if m:
            return " ".join(m.group(1).split())
    return ""


def collect() -> dict[str, list[tuple[str, str]]]:
    """{project label: [(bookmark name, url)]} for everything listening now."""
    dockers = docker_projects()
    groups: dict[str, list[tuple[str, str]]] = {}
    for port, pid in listeners():
        label = ""
        if port in dockers:
            label = DOCKER_LABELS.get(dockers[port], f"{dockers[port]} (Docker)")
        else:
            cwd = cwd_of(pid)
            for repo, name in KNOWN.items():
                if f"/{repo}" in cwd:
                    label = name
                    break
        rule_label, rule_name, rule_path = PORT_RULES.get(port, ("", "", ""))
        label = rule_label or label
        if not label:
            continue
        name = rule_name or HINTS.get(port) or title(port) or f"port {port}"
        url = f"http://localhost:{port}/{rule_path}"
        groups.setdefault(label, []).append((f"{name} (:{port})", url))
    for label, links in EXTRA.items():
        if label in groups:
            groups[label].extend(links)
    return groups


# --- Chrome's Bookmarks JSON -------------------------------------------------
# Every node carries an id unique across the file and a Chrome-epoch timestamp
# (microseconds since 1601). The top-level `checksum` is dropped: Chrome
# recomputes it, and a stale one makes it treat the file as corrupt.


def _chrome_now() -> str:
    return str(int(time.time() * 1_000_000) + 11_644_473_600 * 1_000_000)


def _next_id(doc: dict) -> "callable[[], str]":  # type: ignore[valid-type]
    def walk(node: dict):
        yield node
        for child in node.get("children", []):
            yield from walk(child)

    top = max(int(n["id"]) for root in doc["roots"].values() for n in walk(root) if "id" in n)
    counter = [top]

    def nxt() -> str:
        counter[0] += 1
        return str(counter[0])

    return nxt


def build_folder(groups: dict[str, list[tuple[str, str]]], nxt) -> dict:
    def url_node(name: str, href: str) -> dict:
        return {
            "type": "url",
            "name": name,
            "url": href,
            "id": nxt(),
            "date_added": _chrome_now(),
            "guid": str(uuid.uuid4()),
        }

    def folder_node(name: str, children: list[dict]) -> dict:
        return {
            "type": "folder",
            "name": name,
            "children": children,
            "id": nxt(),
            "date_added": _chrome_now(),
            "date_modified": _chrome_now(),
            "guid": str(uuid.uuid4()),
        }

    order = list(KNOWN.values()) + list(DOCKER_LABELS.values())
    rank = {name: i for i, name in enumerate(order)}
    subfolders = [
        folder_node(label, [url_node(n, u) for n, u in links])
        for label, links in sorted(groups.items(), key=lambda kv: (rank.get(kv[0], 99), kv[0]))
    ]
    return folder_node(FOLDER, subfolders)


def write(groups: dict[str, list[tuple[str, str]]], profile: Path) -> Path:
    path = profile / "Bookmarks"
    backup = path.with_name(f"Bookmarks.bak-{datetime.now(UTC):%Y%m%d-%H%M%S}")
    shutil.copy2(path, backup)
    doc = json.loads(path.read_text())
    doc.pop("checksum", None)
    folder = build_folder(groups, _next_id(doc))
    bar = doc["roots"]["bookmark_bar"]["children"]
    bar[:] = [c for c in bar if c.get("name") != FOLDER]
    bar.insert(0, folder)
    path.write_text(json.dumps(doc, indent=3))
    return backup


def chrome_running() -> bool:
    return bool(_sh(["pgrep", "-x", "Google Chrome"]).strip())


def quit_chrome() -> None:
    subprocess.run(["osascript", "-e", 'quit app "Google Chrome"'], check=False, timeout=30)
    for _ in range(30):
        if not chrome_running():
            return
        time.sleep(1)
    raise SystemExit("Chrome did not quit; nothing was written")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print the folder, write nothing")
    ap.add_argument("--quit-chrome", action="store_true", help="quit Chrome, write, relaunch")
    ap.add_argument("--profile", type=Path, default=CHROME_DEFAULT, help="Chrome profile dir")
    args = ap.parse_args()

    groups = collect()
    for label, links in groups.items():
        print(f"{label}")
        for name, href in links:
            print(f"   {name:<44} {href}")
    if args.dry_run:
        return
    if chrome_running():
        if not args.quit_chrome:
            raise SystemExit("Chrome is running: rerun with --quit-chrome (tabs are restored)")
        quit_chrome()
        relaunch = True
    else:
        relaunch = False
    backup = write(groups, args.profile)
    print(f"\nwrote {args.profile / 'Bookmarks'} (backup: {backup.name})")
    if relaunch:
        subprocess.run(["open", "-a", "Google Chrome"], check=False, timeout=30)


if __name__ == "__main__":
    main()
