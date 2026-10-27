"""MacFlowgraphDeps: a GTK-free stand-in for GRC-Agent's NativeFlowgraphProxy.

GRC-Agent's tools reach the flowgraph through `ctx.deps`. In the desktop app that is
`grc_agent.native_canvas.NativeFlowgraphProxy`, which forwards every attribute to the live
`FlowGraph` and adds five capabilities the tools check with runtime protocols
(`grc_agent.deps`): notify_edit, save_graph, run_flowgraph, get_run_log, save_block. This class
provides the same five on one headless page, so all GRC-Agent tools work unmodified.

Same as the desktop app:
  - the error texts of the run and save gates (never saved, already running, invalid graph,
    no project directory) and the result keys the model reads;
  - saving: GRC-Agent's own `resolve_save_target` naming (untitled -> <project>/<id>.grc with
    the SAVE_AS id rename), the hash-equal no-op, GRC's serializer, atomic temp + os.replace under
    the `.grc_agent/<name>.lock` flock;
  - running: GRC's own generator writes the program next to the saved graph and GRC's own run
    command (`{python} -u {file}`) starts it.

Different, because there is no GTK window and no sandbox on macOS:
  M1  The run is a plain child process (own process group, cwd = the graph's folder), not GRC's
      Executor thread. Its environment drops every *_API_KEY / *_TOKEN / *_SECRET variable and
      sets QT_QPA_PLATFORM=offscreen, so Qt GUI flowgraphs draw off-screen. stdin is an open pipe
      nobody writes to (a terminal nobody types into, as in grc-agent-headless), so a no_gui
      "Prompt for Exit" program runs until it is stopped, as in GRC's terminal window.
  M2  The run log is the program's own output (stdout + stderr, UTF-8, at most 512 KB, the
      oldest text dropped first and flagged), not GRC's console messages.
  M3  Every run has a hard wall-clock cap (`max_run_seconds`, default 300 s): the process group
      is killed when it is reached, also for background (wait=False) runs.
  M4  The invalid-graph gate formats GRC's (element, message) pairs as "<element>: <message>";
      GRC-Agent's desktop proxy joins them as strings, which raises TypeError on a real graph.
  M5  Canvas-only state (redraw, fit-to-view, undo stack, recent files, tabs) does not exist.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import os
import re
import shlex
import signal
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

LOG_LIMIT_BYTES = 512 * 1024
_SECRET_ENV = re.compile(r"(API_KEY|_TOKEN|_SECRET|ACCESS_KEY)", re.I)


def _sha256_file(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()
    except OSError:
        return None


def _error_lines(fg: Any, limit: int = 5) -> list[str]:
    out = []
    for item in fg.iter_error_messages():
        if isinstance(item, tuple) and len(item) == 2:
            element, message = item
            out.append(f"{element}: {message}")
        else:
            out.append(str(item))
        if len(out) >= limit:
            break
    return out


def run_environment() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONUNBUFFERED"] = "1"
    return env


class MacFlowgraphDeps:
    """The agent's deps: one headless page around a live GRC FlowGraph (module docstring)."""

    def __init__(self, flow_graph: Any, *, max_run_seconds: float = 300.0) -> None:
        object.__setattr__(self, "_fg", flow_graph)
        object.__setattr__(self, "_max_run_seconds", float(max_run_seconds))
        object.__setattr__(self, "_proc", None)
        object.__setattr__(self, "_reader", None)
        object.__setattr__(self, "_killer", None)
        object.__setattr__(self, "_log", bytearray())
        object.__setattr__(self, "_log_dropped", 0)
        object.__setattr__(self, "_last_run", None)
        object.__setattr__(self, "_graph_modified", False)
        object.__setattr__(self, "runs", [])

    # -- the page -------------------------------------------------------------------------------
    def _get_target(self) -> Any:
        return object.__getattribute__(self, "_fg")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_target(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._get_target(), name, value)

    @property
    def file_path(self) -> str:
        return str(getattr(self._get_target(), "grc_file_path", "") or "")

    def _set(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def _get(self, name: str) -> Any:
        return object.__getattribute__(self, name)

    # -- notify_edit ----------------------------------------------------------------------------
    async def notify_edit(self, relayout: bool = False) -> dict:
        self._set("_graph_modified", True)
        return {"ok": True}

    # -- save_graph -----------------------------------------------------------------------------
    async def save_graph(self) -> dict:
        from grc_agent import fs_tools
        from grc_agent.adapter import flow_graph_content_hash
        from grc_agent.adapter.graph import resolve_save_target

        from . import grc_platform

        fg = self._get_target()
        options_id = fg.options_block.params["id"].get_value()
        target, id_stem = resolve_save_target(fs_tools.active_project_dir(), options_id,
                                              self.file_path or None)
        target = Path(target)
        if target.exists() and not os.access(target, os.W_OK):
            raise ValueError(f"Could not save {target.name}: the file is read-only")
        if id_stem is not None and options_id != id_stem:
            fg.options_block.params["id"].set_value(id_stem)
            fg.rewrite()
            fg.validate()
        if target.exists() and flow_graph_content_hash(fg) == _sha256_file(target):
            fg.grc_file_path = str(target)
            return {"path": str(target), "page": target.stem}

        lock = target.parent / ".grc_agent" / (target.name + ".lock")
        try:
            lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with lock.open("a", encoding="utf-8") as lock_file:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError(f"{target.name} is locked by another writer. "
                                     "Retry the save in a moment.") from None
                try:
                    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp",
                                                    dir=str(target.parent))
                    os.close(fd)
                    tmp = Path(tmp_name)
                    try:
                        grc_platform.install().save_flow_graph(str(tmp), fg)
                        if target.exists():
                            os.chmod(tmp, stat.S_IMODE(os.stat(target).st_mode))
                        else:
                            umask = os.umask(0)
                            os.umask(umask)
                            os.chmod(tmp, 0o666 & ~umask)
                        os.replace(tmp, target)
                    finally:
                        with contextlib.suppress(OSError):
                            tmp.unlink()
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError as e:
            raise ValueError(f"Could not save {target.name}: {e}") from e
        fg.grc_file_path = str(target)
        return {"path": str(target), "page": target.stem}

    # -- save_block -----------------------------------------------------------------------------
    async def save_block(self, instance_name: str, block_id: str | None = None,
                         label: str | None = None, category: str | None = None,
                         overwrite: bool = False) -> dict:
        from grc_agent.adapter.block_library import save_block_to_library
        return save_block_to_library(self._get_target(), instance_name, block_id=block_id,
                                     label=label, category=category, overwrite=overwrite)

    # -- run_flowgraph --------------------------------------------------------------------------
    def _running(self) -> bool:
        proc = self._get("_proc")
        return proc is not None and proc.returncode is None

    async def run_flowgraph(self, action: Literal["start", "stop"] = "start", wait: bool = True,
                            timeout_seconds: float = 60.0,
                            stop_after_seconds: float | None = None) -> dict:
        if action == "stop":
            return await self.stop_flowgraph()
        if action != "start":
            raise ValueError(f"Invalid action {action!r}: must be 'start' or 'stop'")
        if stop_after_seconds is not None:
            if not wait:
                raise ValueError("stop_after_seconds requires wait=True (a bounded run blocks "
                                 "until it is stopped).")
            if stop_after_seconds <= 0:
                raise ValueError("stop_after_seconds must be a positive number of seconds.")
        if self._running():
            raise ValueError("A flowgraph execution is already in progress. Stop it with "
                             "run_flowgraph(action='stop') first, or wait for it to finish.")
        if not self.file_path:
            raise ValueError("The flowgraph has never been saved. GRC generates into the saved "
                             "graph's directory and executes from there, so save it first: call "
                             "save_graph to write this flowgraph into the project directory, "
                             "then retry run_flowgraph.")
        fg = self._get_target()
        fg.validate()
        if not fg.is_valid():
            raise ValueError(f"The flowgraph is invalid, so GRC will refuse to execute it: "
                             f"{'; '.join(_error_lines(fg))}. Fix the graph with change_graph first.")

        from gnuradio.grc.core.generator.Generator import Generator
        out_dir = Path(self.file_path).parent
        gen = Generator(fg, str(out_dir))
        gen.write()
        command = fg.get_run_command(gen.file_path)
        argv = shlex.split(command)

        self._get("_log").clear()
        self._set("_log_dropped", 0)
        self._set("_graph_modified", False)
        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(out_dir), env=run_environment(), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True)
        record = {"command": command, "started": time.time(), "return_code": None,
                  "seconds": None, "stopped": False, "hard_cap_hit": False}
        self.runs.append(record)
        self._set("_proc", proc)
        self._set("_last_run", record)
        self._set("_reader", asyncio.ensure_future(self._read_output(proc, record, started)))
        self._set("_killer", asyncio.ensure_future(self._hard_cap(proc, record)))

        if not wait:
            return {"status": "started",
                    "note": ("The flowgraph is running in the background. GUI flowgraphs run "
                             "until stopped: call run_flowgraph(action='stop') when it should "
                             "end. Read output with get_run_log (its run_in_progress field "
                             "tells you when the run has finished).")}

        budget = stop_after_seconds if stop_after_seconds is not None else timeout_seconds
        try:
            await asyncio.wait_for(asyncio.shield(self._get("_reader")), timeout=budget)
        except asyncio.TimeoutError:
            if stop_after_seconds is not None:
                await self.stop_flowgraph()
                return self._result("stopped_after_timeout",
                                    f"Stopped after the {stop_after_seconds:g} s budget. Read "
                                    "the output with get_run_log.")
            return {"status": "still_running",
                    "note": (f"Still running after {timeout_seconds:g} s. Stop it with "
                             "run_flowgraph(action='stop') or check get_run_log later.")}
        rc = record["return_code"]
        note = ("The flowgraph finished. Read the output with get_run_log."
                if rc == 0 else "The flowgraph exited with an error: read the output with "
                "get_run_log to see why.")
        return self._result("finished", note)

    def _result(self, status: str, note: str) -> dict:
        record = self._get("_last_run") or {}
        rc = record.get("return_code")
        return {"status": status, "return_code": rc,
                "ran_successfully": status == "finished" and rc == 0,
                "hard_cap_hit": bool(record.get("hard_cap_hit")), "note": note}

    async def _read_output(self, proc: Any, record: dict, started: float) -> None:
        log = self._get("_log")
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            log.extend(chunk)
            excess = len(log) - LOG_LIMIT_BYTES
            if excess > 0:
                del log[:excess]
                self._set("_log_dropped", self._get("_log_dropped") + excess)
        record["return_code"] = await proc.wait()
        record["seconds"] = round(time.monotonic() - started, 2)
        killer = self._get("_killer")
        if killer is not None and not killer.done():
            killer.cancel()

    async def _hard_cap(self, proc: Any, record: dict) -> None:
        await asyncio.sleep(self._get("_max_run_seconds"))
        if proc.returncode is None:
            record["hard_cap_hit"] = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)

    async def stop_flowgraph(self) -> dict:
        proc = self._get("_proc")
        if not self._running():
            return {"status": "not_running", "note": "No flowgraph is running."}
        record = self._get("_last_run")
        record["stopped"] = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.shield(self._get("_reader")), timeout=5)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            await self._get("_reader")
        return {"status": "stopped", "return_code": record["return_code"],
                "note": "Stopped (SIGTERM, then SIGKILL if the shutdown stalled)."}

    def get_run_log(self) -> dict | None:
        record = self._get("_last_run")
        if record is None:
            return None
        data = {
            "log_text": bytes(self._get("_log")).decode("utf-8", errors="replace"),
            "command": record["command"],
            "return_code": record["return_code"],
            "run_in_progress": self._running(),
            "graph_modified_since_run": self._get("_graph_modified"),
            "output_truncated": self._get("_log_dropped") > 0,
            "dropped_bytes": self._get("_log_dropped"),
        }
        with contextlib.suppress(Exception):
            if self._get_target().get_option("generate_options") == "no_gui":
                data["generate_options"] = "no_gui"
        return data

    async def aclose(self) -> None:
        """Stop a run still going at the end of a call (the next call starts with none)."""
        if self._running():
            await self.stop_flowgraph()
        for name in ("_reader", "_killer"):
            task = self._get(name)
            if task is not None and not task.done():
                task.cancel()
