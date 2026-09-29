"""pytest plugin: make the macOS 3.14t hang report itself.

Four CI jobs have gone silent immediately before
test/with_dummyserver/test_https.py (and its test/contrib/test_pyopenssl.py copy)
::TestHTTPS_TLSv1_3::test_http2_probe_blocked_per_thread
and were then killed by the 10-minute job timeout, leaving nothing in the log.

This records every HTTP/2 probe acquire and release and every exception that comes out
of HTTPSConnection.connect(), and arms a watchdog for each test. If a test runs longer
than PROBE_HANG_WATCHDOG seconds the watchdog writes the recording, the state of every
probe lock and the traceback of every thread, then exits the process so the surrounding
loop can continue.

The report is written to a file AND to a copy of the real stderr taken before pytest
installs its capture, because pytest would otherwise swallow it and os._exit() skips
the capture teardown that would have replayed it.

Environment:
  PROBE_HANG_WATCHDOG   seconds before the watchdog fires (default 45)
  PROBE_HANG_OUT        report file (default probe_hang_report.txt)
  PROBE_HANG_LIGHT      only arm the watchdog: do not wrap acquire_and_get,
                        set_and_release or connect(). The wrappers add a lock and a
                        list append to the very code whose timing is in question, so
                        half of the CI jobs run light in case the instrumentation
                        itself hides the race.
  PROBE_HANG_INJECT     self-test: raise while holding the probe lock, which is the
                        failure this plugin exists to catch, to prove it gets reported

Load with: -p probe_hang  (with this directory on PYTHONPATH)
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
import traceback
import typing

import pytest

WATCHDOG = float(os.environ.get("PROBE_HANG_WATCHDOG", "45"))
OUT = os.environ.get("PROBE_HANG_OUT", "probe_hang_report.txt")
INJECT = bool(os.environ.get("PROBE_HANG_INJECT"))
LIGHT = bool(os.environ.get("PROBE_HANG_LIGHT")) and not INJECT

# Taken at import time, before pytest's capture replaces fd 2.
_STDERR_FD = os.dup(2)

_events: list[tuple[float, int, str, str]] = []
_events_lock = threading.Lock()
state: dict[str, typing.Any] = {
    "test": "<none>",
    "patched": False,
    "injected": 0,
    "entered": 0,
}


def _rec(text: str) -> None:
    with _events_lock:
        _events.append(
            (round(time.monotonic(), 4), threading.get_ident(), state["test"], text)
        )
        if len(_events) > 600:
            del _events[:300]


def _patch() -> None:
    """Patch what the test run has already imported.

    Nothing is imported here: urllib3 and the dummyserver decide at import time whether
    IPv6 is usable, so importing them earlier than the run does changes what is measured.
    """
    probe = sys.modules.get("urllib3.http2.probe")
    conn = sys.modules.get("urllib3.connection")
    if probe is None or conn is None or state["patched"]:
        return
    state["patched"] = True
    state["probe"] = probe
    if LIGHT:
        _rec(
            f"watching (light), pid={os.getpid()} python={sys.version.split()[0]} "
            f"watchdog={WATCHDOG}s"
        )
        return

    real_acquire = probe.acquire_and_get
    real_release = probe.set_and_release

    def acquire_and_get(host: str, port: int) -> bool | None:
        state["entered"] = nth = state.get("entered", 0) + 1
        _rec(f"acquire_and_get({host}:{port}) enter (entry {nth})")
        injecting = INJECT and "blocked_per_thread" in state["test"]
        if injecting and nth == 1:
            # Hold the first caller back so a later one wins the lock; otherwise the
            # first item's future fails and the test fails instead of hanging.
            time.sleep(0.3)
        value = real_acquire(host, port)
        held = " [this thread now holds the lock]" if value is None else ""
        _rec(f"acquire_and_get({host}:{port}) -> {value!r}{held}")
        if injecting and value is None and nth != 1 and not state["injected"]:
            state["injected"] = 1
            _rec("INJECT: raising while holding the probe lock")
            raise AssertionError(
                "injected failure, as if an assert in _connect_callback"
            )
        return value

    def set_and_release(host: str, port: int, supports_http2: bool | None) -> None:
        _rec(f"set_and_release({host}:{port}, {supports_http2!r})")
        return real_release(host=host, port=port, supports_http2=supports_http2)

    probe.acquire_and_get = acquire_and_get
    probe.set_and_release = set_and_release

    real_connect = conn.HTTPSConnection.connect

    def connect(self: typing.Any) -> None:
        try:
            return real_connect(self)
        except BaseException as e:
            _rec(f"connect() raised {type(e).__name__}: {str(e)[:150]}")
            raise

    conn.HTTPSConnection.connect = connect
    _rec(
        f"watching, pid={os.getpid()} python={sys.version.split()[0]} "
        f"watchdog={WATCHDOG}s inject={INJECT}"
    )


def _lock_states() -> list[str]:
    probe = state.get("probe")
    if probe is None:
        return ["<urllib3.http2.probe was never imported>"]
    cache = probe._HTTP2_PROBE_CACHE
    out = [f"values: {cache._cache_values!r}"]
    for key, lock in list(cache._cache_locks.items()):
        try:
            free = lock.acquire(blocking=False)
            if free:
                lock.release()
        except BaseException as e:  # pragma: no cover - defensive
            out.append(f"lock {key}: could not be probed: {e!r}")
            continue
        out.append(f"lock {key}: repr={lock!r} held_by_another_thread={not free}")
    return out


def _report(nodeid: str) -> str:
    lines = [
        "",
        f"===== probe_hang: {nodeid} has been running for {WATCHDOG}s =====",
        "--- probe cache",
    ]
    try:
        lines += [f"  {line}" for line in _lock_states()]
    except BaseException:
        lines.append("  <failed to read the probe cache>")
        lines += ["  " + line for line in traceback.format_exc().splitlines()]
    lines.append("--- recorded probe events (monotonic, thread, test, what)")
    with _events_lock:
        for t, tid, test, text in _events:
            lines.append(f"  {t:12.4f} {tid} {test} :: {text}")
    lines.append("--- live threads")
    for th in threading.enumerate():
        lines.append(f"  {th.ident} {th.name} daemon={th.daemon}")
    lines.append("--- traceback of every thread follows")
    return "\n".join(lines) + "\n"


def _fire(nodeid: str) -> None:
    try:
        text = _report(nodeid)
    except BaseException:  # pragma: no cover - defensive
        text = "probe_hang: building the report failed\n" + traceback.format_exc()
    try:
        with open(OUT, "a") as f:
            f.write(text)
            f.flush()
            faulthandler.dump_traceback(file=f, all_threads=True)
            f.write("===== probe_hang: end of report =====\n")
    except BaseException:
        pass
    try:
        os.write(_STDERR_FD, text.encode("utf-8", "replace"))
        faulthandler.dump_traceback(file=_STDERR_FD, all_threads=True)
        os.write(
            _STDERR_FD, b"===== probe_hang: exiting 99 so the loop continues =====\n"
        )
    except BaseException:
        pass
    os._exit(99)


def _monitor() -> None:
    """One daemon thread, so the shutdown phase is watched too.

    A per-test timer would be cancelled when the test ends, and a leaked probe lock can
    instead hang the interpreter at exit, where concurrent.futures joins its workers.
    """
    while True:
        time.sleep(0.5)
        deadline = state.get("deadline")
        if deadline is not None and time.monotonic() > deadline:
            _fire(str(state.get("phase", "?")))


def pytest_configure(config: pytest.Config) -> None:
    t = threading.Thread(target=_monitor, name="probe_hang-monitor", daemon=True)
    t.start()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(
    item: pytest.Item,
) -> typing.Generator[None, object, object]:
    state["test"] = item.nodeid
    state["phase"] = item.nodeid
    state["entered"] = 0
    _patch()
    state["deadline"] = time.monotonic() + WATCHDOG
    try:
        return (yield)
    finally:
        state["deadline"] = None


def pytest_sessionfinish(session: pytest.Session, exitstatus: object) -> None:
    # Keep watching: a leaked lock can hang the interpreter at exit instead.
    state["phase"] = "session shutdown"
    state["deadline"] = time.monotonic() + WATCHDOG
