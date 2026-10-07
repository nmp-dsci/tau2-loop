"""A workflow library several workers build at once (s21, r3): versions, locks and commits.

Research needs no lock. A merge names the jobs it will write, locks them all at once (first come,
first served per job), reads their newest versions, and commits only while its lock is current and
every version it read is still the head, so no worker ever writes over another's merge. A worker
waiting for a job sees who holds it; the viewer and `make rag-locks` read the same table.

A FileStore keeps one version's library in `rag_library/<domain>/<rag>/`:

  commits.jsonl            one line per committed version: seq, job, version, the versions it
                           read, the names it retired, session, task, lock token, time. Append-only
                           and kept in git like rag_agent_runs/: the library at any commit is this
                           log up to that line (`entries(upto=)`).
  versions/<job>/<v>.json  each committed workflow as written
  seed.json                the seed library this version started from (job, session), pinned the
                           first time it is read, so later sessions of the seed never change it
  locks.json               held and waiting locks; runtime only, git-ignored
  library.lock             every call holds it (flock, plus a thread lock) for its few milliseconds

A lock is a lease: it expires LEASE_S after its last heartbeat, so a worker that dies frees its
jobs; its token rises with every grant, and a commit under a token that is no longer held is
refused (a fencing token), so a worker that comes back late cannot write. Nothing here calls a
model.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tau2_loop.config import ROOT

LIBRARY_DIR = ROOT / "rag_library"
LEASE_S = 120.0  # a lock not renewed for this long is free
HEARTBEAT_S = 30.0  # a holder renews its lease this often
POLL_S = 1.0  # a waiting worker checks again this often
WAITER_STALE_S = 30.0  # a waiting entry not refreshed for this long belonged to a worker that died

Entry = dict[str, Any]  # {"job", "workflow", "session", "task_id", "version"}, as library.entries


class LockLostError(RuntimeError):
    """The lease ran out and the jobs may be another worker's now."""


class CommitRefusedError(RuntimeError):
    """A write the lock does not cover, or a version that is no longer the head."""


@dataclass(frozen=True)
class Lock:
    jobs: tuple[str, ...]  # every job the holder may write, sorted
    holder: str  # session id
    task: str  # the train question, for the viewer
    token: int  # rises with every grant; a commit must carry a current one
    since: float
    expires: float


@dataclass
class Holding:
    """A lock while it is held: `lost` turns true if a heartbeat finds the lease gone."""

    lock: Lock
    lost: bool = False
    beats: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)


_proc_locks: dict[str, threading.Lock] = {}
_proc_locks_guard = threading.Lock()


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class FileStore:
    """One RAG version's library on disk. `seed` returns the seed version's entries (read once
    and pinned); `load(session, job)` reads a pinned seed job back from its session."""

    def __init__(
        self,
        domain: str,
        rag: str,
        *,
        seed: Callable[[], list[Entry]] | None = None,
        load: Callable[[str, str], dict[str, Any] | None] | None = None,
        root: Path | None = None,
    ) -> None:
        self.domain, self.rag = domain, rag
        self.dir = (root or LIBRARY_DIR) / domain / rag
        self._seed, self._load = seed, load
        with _proc_locks_guard:
            self._plock = _proc_locks.setdefault(str(self.dir), threading.Lock())

    # ── the mutex every call holds ──

    @contextmanager
    def _mutex(self) -> Iterator[None]:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self._plock, (self.dir / "library.lock").open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _write(self, name: str, text: str) -> None:
        tmp = self.dir / f".{name}.{os.getpid()}.{threading.get_ident()}.tmp"
        tmp.write_text(text)
        tmp.replace(self.dir / name)

    # ── versions ──

    def commits(self) -> list[dict[str, Any]]:
        p = self.dir / "commits.jsonl"
        if not p.is_file():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def _seed_entries(self) -> list[Entry]:
        """The seed's library as it was the first time this version read it."""
        p = self.dir / "seed.json"
        if not p.is_file():
            if self._seed is None:
                return []
            got = self._seed()
            pinned = [
                {"job": e["job"], "session": e["session"], "task_id": e.get("task_id")} for e in got
            ]
            self.dir.mkdir(parents=True, exist_ok=True)
            if not p.is_file():  # another process may have pinned it meanwhile
                self._write("seed.json", json.dumps(pinned, indent=1) + "\n")
            return [{**e, "version": 0} for e in got]
        out = []
        for ref in json.loads(p.read_text()):
            wf = self._load(ref["session"], ref["job"]) if self._load else None
            if wf is not None:
                out.append({**ref, "workflow": wf, "version": 0})
        return out

    def entries(self, upto: int | None = None) -> list[Entry]:
        """The library: the seed's jobs at version 0, then every commit in order (up to and
        including commit `upto`, when given). A commit replaces its job and retires the names
        it merged."""
        out: dict[str, Entry] = {e["job"]: e for e in self._seed_entries()}
        for c in self.commits():
            if upto is not None and c["seq"] > upto:
                break
            for old in c.get("retired") or []:
                out.pop(old, None)
            wf = json.loads((self.dir / "versions" / c["job"] / f"{c['version']}.json").read_text())
            out[c["job"]] = {
                "job": c["job"],
                "workflow": wf,
                "session": c["session"],
                "task_id": c.get("task"),
                "version": c["version"],
            }
        return list(out.values())

    # ── locks ──

    def _state(self) -> dict[str, Any]:
        p = self.dir / "locks.json"
        st = json.loads(p.read_text()) if p.is_file() else {}
        st.setdefault("next_token", 0)
        st.setdefault("held", [])
        st.setdefault("waiting", [])
        now = time.time()
        st["held"] = [h for h in st["held"] if h["expires"] > now]
        st["waiting"] = [w for w in st["waiting"] if now - w["seen"] < WAITER_STALE_S]
        return st

    def _save(self, st: dict[str, Any]) -> None:
        self._write("locks.json", json.dumps(st, indent=1) + "\n")

    def lock(
        self,
        jobs: list[str] | tuple[str, ...] | set[str],
        holder: str,
        task: str = "",
        *,
        on_wait: Callable[[list[dict[str, Any]]], None] | None = None,
        timeout: float | None = None,
    ) -> Lock:
        """Every job or none: a worker never holds one while waiting for another, so no two can
        each hold what the other needs. It waits behind any holder of one of its jobs and behind
        any earlier waiter for one (first come, first served). `on_wait` gets who is in the way
        each time that changes."""
        want = tuple(sorted(set(jobs)))
        started = time.time()
        shown: tuple[tuple[str, int | float], ...] | None = None
        while True:
            with self._mutex():
                st = self._state()
                now = time.time()
                mine = next((w for w in st["waiting"] if w["holder"] == holder), None)
                if mine is None:
                    mine = {"jobs": list(want), "holder": holder, "task": task, "arrived": now}
                    st["waiting"].append(mine)
                mine["seen"], mine["jobs"] = now, list(want)
                held = [h for h in st["held"] if set(h["jobs"]) & set(want)]
                ahead = [
                    w
                    for w in st["waiting"]
                    if w is not mine
                    and w["arrived"] < mine["arrived"]
                    and set(w["jobs"]) & set(want)
                ]
                if not held and not ahead:
                    st["next_token"] += 1
                    lk = Lock(want, holder, task, st["next_token"], now, now + LEASE_S)
                    st["waiting"].remove(mine)
                    st["held"].append(
                        {
                            "jobs": list(want),
                            "holder": holder,
                            "task": task,
                            "token": lk.token,
                            "since": now,
                            "expires": lk.expires,
                        }
                    )
                    self._save(st)
                    return lk
                self._save(st)
            way = [{**h, "state": "held"} for h in held] + [
                {**w, "state": "waiting"} for w in ahead
            ]
            key = tuple((w["holder"], w.get("token", w.get("arrived", 0))) for w in way)
            if on_wait is not None and key != shown:
                on_wait(way)
            shown = key
            if timeout is not None and time.time() - started > timeout:
                with self._mutex():
                    st = self._state()
                    st["waiting"] = [w for w in st["waiting"] if w["holder"] != holder]
                    self._save(st)
                raise TimeoutError(f"{holder} waited {timeout:.0f} s for {', '.join(want)}")
            time.sleep(POLL_S)

    def _held(self, st: dict[str, Any], lock: Lock) -> dict[str, Any] | None:
        return next((h for h in st["held"] if h["token"] == lock.token), None)

    def heartbeat(self, lock: Lock) -> Lock:
        with self._mutex():
            st = self._state()
            h = self._held(st, lock)
            if h is None:
                raise LockLostError(f"lock {lock.token} on {', '.join(lock.jobs)} expired")
            h["expires"] = time.time() + LEASE_S
            self._save(st)
            return Lock(lock.jobs, lock.holder, lock.task, lock.token, lock.since, h["expires"])

    def release(self, lock: Lock) -> None:
        with self._mutex():
            st = self._state()
            st["held"] = [h for h in st["held"] if h["token"] != lock.token]
            self._save(st)

    @contextmanager
    def holding(self, lock: Lock) -> Iterator[Holding]:
        """Hold `lock` with a heartbeat thread renewing it; release it however the block ends."""
        hold = Holding(lock)

        def beat() -> None:
            while not hold._stop.wait(HEARTBEAT_S):
                try:
                    hold.lock = self.heartbeat(hold.lock)
                    hold.beats += 1
                except LockLostError:
                    hold.lost = True
                    return

        t = threading.Thread(target=beat, daemon=True, name=f"lease-{lock.token}")
        t.start()
        try:
            yield hold
        finally:
            hold._stop.set()
            t.join(timeout=5)
            self.release(hold.lock)

    # ── commits ──

    def commit(self, lock: Lock, writes: list[dict[str, Any]]) -> dict[str, int]:
        """Write each job's next version. A write is {job, workflow, reads: {name: version},
        session, task}: `reads` are the library jobs it merged, at the versions it read. Refused
        unless the lock is current, covers every job named, and every read is still the head.
        Returns each written job's new version."""
        with self._mutex():
            st = self._state()
            if self._held(st, lock) is None:
                raise LockLostError(f"lock {lock.token} on {', '.join(lock.jobs)} expired")
            now = {e["job"]: e for e in self.entries()}
            mine = set(lock.jobs)
            for w in writes:
                names = {w["job"], *w.get("reads", {})}
                if names - mine:
                    raise CommitRefusedError(
                        f"{', '.join(sorted(names - mine))} not locked; the lock holds "
                        f"{', '.join(lock.jobs)}"
                    )
                for name, v in (w.get("reads") or {}).items():
                    cur = now.get(name)
                    if cur is None or cur["version"] != v:
                        raise CommitRefusedError(
                            f"{name} is at {cur['version'] if cur else 'none'}, not the {v} read"
                        )
                if w["job"] in now and w["job"] not in (w.get("reads") or {}):
                    raise CommitRefusedError(f"{w['job']} exists and the write did not read it")
            log = self.commits()
            seq = log[-1]["seq"] + 1 if log else 1
            out: dict[str, int] = {}
            lines = []
            for w in writes:
                reads = dict(w.get("reads") or {})
                version = 1 + max(reads.values(), default=0)
                d = self.dir / "versions" / w["job"]
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{version}.json").write_text(
                    json.dumps(w["workflow"], indent=1, ensure_ascii=False) + "\n"
                )
                lines.append(
                    {
                        "seq": seq,
                        "job": w["job"],
                        "version": version,
                        "reads": reads,
                        "retired": sorted(n for n in reads if n != w["job"]),
                        "session": w.get("session"),
                        "task": w.get("task"),
                        "token": lock.token,
                        "at": _now_iso(),
                    }
                )
                out[w["job"]] = version
                seq += 1
            with (self.dir / "commits.jsonl").open("a", encoding="utf-8") as f:
                for line in lines:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
            return out

    # ── what the viewer reads ──

    def locks(self) -> dict[str, Any]:
        """Held and waiting locks now, with how long each has held or waited."""
        if not (self.dir / "locks.json").is_file():
            return {"held": [], "waiting": []}
        with self._mutex():
            st = self._state()
        now = time.time()
        return {
            "held": [
                {
                    **h,
                    "held_s": round(now - h["since"], 1),
                    "expires_s": round(h["expires"] - now, 1),
                }
                for h in st["held"]
            ],
            "waiting": [{**w, "waited_s": round(now - w["arrived"], 1)} for w in st["waiting"]],
        }

    @contextmanager
    def build_guard(self) -> Iterator[None]:
        """One build process per version: a second `make rag-build` is refused, not run twice."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / "build.lock").open("a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as e:
                raise RuntimeError(f"a build of {self.rag} is already running") from e
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
