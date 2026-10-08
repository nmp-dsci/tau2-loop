"""s21: the store a concurrent workflow_rag build shares: locks, leases, commits, replay. No model."""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tau2_loop.workflows import store

D = "banking_knowledge"


@pytest.fixture(autouse=True)
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store, "POLL_S", 0.02)


def job(name: str, *steps: str) -> dict[str, Any]:
    return {"job": name, "steps": [{"id": s} for s in steps]}


def seeded(tmp_path: Path, *names: str) -> store.FileStore:
    seed = [
        {"job": n, "workflow": job(n, "verify"), "session": f"s_{n}", "task_id": "task_001"}
        for n in names
    ]
    return store.FileStore(
        D,
        "r9",
        seed=lambda: seed,
        load=lambda session, name: job(name, "verify"),
        root=tmp_path,
    )


def write(
    st: store.FileStore, lk: store.Lock, name: str, reads: dict[str, int], *steps: str
) -> dict[str, int]:
    return st.commit(
        lk,
        [
            {
                "job": name,
                "workflow": job(name, *steps),
                "reads": reads,
                "session": lk.holder,
                "task": lk.task,
            }
        ],
    )


def test_the_second_worker_waits_then_merges_the_first_ones_version(tmp_path: Path) -> None:
    st = seeded(tmp_path, "dispute")
    a = st.lock(["dispute"], "A", "task_019")
    seen: list[list[dict[str, Any]]] = []
    got: dict[str, Any] = {}

    def b() -> None:
        lk = st.lock(["dispute"], "B", "task_020", on_wait=seen.append)
        head = next(e for e in st.entries() if e["job"] == "dispute")
        got["read"] = head["version"], [s["id"] for s in head["workflow"]["steps"]]
        got["wrote"] = write(st, lk, "dispute", {"dispute": head["version"]}, "verify", "a", "b")
        st.release(lk)

    t = threading.Thread(target=b)
    t.start()
    time.sleep(0.2)
    assert st.locks()["waiting"][0]["task"] == "task_020"  # anyone can see B waiting for A
    assert seen and seen[0][0]["holder"] == "A" and seen[0][0]["state"] == "held"
    assert write(st, a, "dispute", {"dispute": 0}, "verify", "a") == {"dispute": 1}
    st.release(a)
    t.join(5)
    assert got["read"] == (1, ["verify", "a"])  # B merged A's version, not the one it saw first
    assert got["wrote"] == {"dispute": 2}
    assert [s["id"] for s in st.entries()[0]["workflow"]["steps"]] == ["verify", "a", "b"]


def test_a_commit_of_a_version_that_is_no_longer_the_head_is_refused(tmp_path: Path) -> None:
    st = seeded(tmp_path, "dispute", "closure")
    lk = st.lock(["dispute"], "A", "task_019")
    write(st, lk, "dispute", {"dispute": 0}, "verify", "a")
    with pytest.raises(store.CommitRefusedError, match="not the 0 read"):
        write(st, lk, "dispute", {"dispute": 0}, "verify", "b")
    with pytest.raises(store.CommitRefusedError, match="closure not locked"):
        write(st, lk, "dispute", {"dispute": 1, "closure": 0}, "verify")
    with pytest.raises(store.CommitRefusedError, match="did not read it"):
        write(st, lk, "dispute", {}, "verify")


def test_an_expired_lease_frees_the_jobs_and_its_holder_cannot_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    st = seeded(tmp_path, "dispute")
    monkeypatch.setattr(store, "LEASE_S", 0.1)
    a = st.lock(["dispute"], "A", "task_019")
    time.sleep(0.2)  # A stalls past its lease
    monkeypatch.setattr(store, "LEASE_S", 60.0)
    b = st.lock(["dispute"], "B", "task_020")  # granted: A's lease is gone
    assert b.token > a.token
    with pytest.raises(store.LockLostError):
        write(st, a, "dispute", {"dispute": 0}, "verify", "late")
    with pytest.raises(store.LockLostError):
        st.heartbeat(a)
    assert write(st, b, "dispute", {"dispute": 0}, "verify", "b") == {"dispute": 1}


def test_a_heartbeat_keeps_a_long_merge_its_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    st = seeded(tmp_path, "dispute")
    monkeypatch.setattr(store, "LEASE_S", 0.3)
    monkeypatch.setattr(store, "HEARTBEAT_S", 0.05)
    with st.holding(st.lock(["dispute"], "A", "task_019")) as hold:
        time.sleep(0.6)  # twice the lease
        assert not hold.lost and hold.beats >= 5
        write(st, hold.lock, "dispute", {"dispute": 0}, "verify", "a")
    assert st.locks()["held"] == []  # released on the way out


def test_workers_locking_random_sets_all_finish(tmp_path: Path) -> None:
    names = ["a", "b", "c", "d", "e"]
    st = seeded(tmp_path, *names)
    order: list[str] = []
    rng = random.Random(7)
    sets = [rng.sample(names, rng.randint(1, 3)) for _ in range(16)]

    def worker(i: int, want: list[str]) -> None:
        lk = st.lock(want, f"w{i}", f"task_{i:03d}", timeout=20)
        heads = {e["job"]: e["version"] for e in st.entries()}
        for n in sorted(want):
            write(st, lk, n, {n: heads[n]}, "verify", f"w{i}")
        order.append(f"w{i}")
        st.release(lk)

    threads = [threading.Thread(target=worker, args=(i, s)) for i, s in enumerate(sets)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(order) == 16  # no deadlock: a worker asks for all its jobs at once
    heads = {e["job"]: e["version"] for e in st.entries()}
    # every write landed on the version before it: each job's version counts its writers
    assert heads == {n: sum(n in s for s in sets) for n in names}


def test_the_library_replays_from_its_log_and_a_merged_name_retires(tmp_path: Path) -> None:
    st = seeded(tmp_path, "retention_a", "retention_b")
    lk = st.lock(["retention_a", "retention_b"], "A", "task_046")
    write(st, lk, "retention_a", {"retention_a": 0, "retention_b": 0}, "verify", "offer")
    assert [e["job"] for e in st.entries()] == ["retention_a"]  # b merged into a
    assert {e["job"] for e in st.entries(upto=0)} == {"retention_a", "retention_b"}  # before it
    c = st.commits()[0]
    assert (c["version"], c["retired"], c["task"]) == (1, ["retention_b"], "task_046")


def test_a_retired_name_recreated_gets_a_new_version_and_keeps_the_old_file(
    tmp_path: Path,
) -> None:
    st = seeded(tmp_path, "a")
    lk = st.lock(["b"], "A", "task_001")
    write(st, lk, "b", {}, "first")
    st.release(lk)
    lk = st.lock(["a", "b"], "A", "task_002")
    write(st, lk, "a", {"a": 0, "b": 1}, "verify", "merged")
    st.release(lk)
    first = st.commits()[0]["seq"]
    lk = st.lock(["b"], "A", "task_003")
    assert write(st, lk, "b", {}, "second") == {"b": 2}
    assert [s["id"] for s in st.entries(upto=first)[-1]["workflow"]["steps"]] == ["first"]
    assert [e["job"] for e in st.entries(upto=first)] == ["a", "b"]
    assert [s["id"] for s in st.entries()[-1]["workflow"]["steps"]] == ["second"]


def test_the_seed_is_pinned_the_first_time_it_is_read(tmp_path: Path) -> None:
    seed = [
        {"job": "dispute", "workflow": job("dispute", "verify"), "session": "s1", "task_id": "t"}
    ]
    st = store.FileStore(
        D, "r9", seed=lambda: list(seed), load=lambda s, n: job(n, "verify"), root=tmp_path
    )
    assert [e["job"] for e in st.entries()] == ["dispute"]
    seed.append({"job": "late", "workflow": job("late"), "session": "s2", "task_id": "t"})
    assert [e["job"] for e in st.entries()] == [
        "dispute"
    ]  # the seed's later sessions never reach it


def test_a_second_build_of_the_same_version_is_refused(tmp_path: Path) -> None:
    st = seeded(tmp_path, "dispute")
    with (
        st.build_guard(),
        pytest.raises(RuntimeError, match="already running"),
        seeded(tmp_path, "dispute").build_guard(),
    ):
        pass
