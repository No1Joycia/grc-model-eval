"""One call of a session: restore the session state, run GRC-Agent's executor, write the outputs.

Layout (`<work>` defaults to grc-agent-adapter/runs):
    <work>/<session>/session.json        every call in order
    <work>/<session>/project/            the live project folder (the agent's file and save root)
    <work>/<session>/call-NN/            one folder per call (README.md, "Outputs")

The project folder keeps one path for the whole session, so paths in the conversation stay
valid. Each call ends with a snapshot of it in call-NN/project/; continuing from a call restores
that snapshot first, so any call can be continued (a session can branch).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from . import build, grc_platform, trace
from .deps import MacFlowgraphDeps, _error_lines, _sha256_file

SHELL_TOOLS = {"run_command", "start_command"}


@dataclasses.dataclass
class ModelSpec:
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None      # the variable's name, never its value


@dataclasses.dataclass
class Policy:
    allow_graph_edits: bool = True
    allow_run: bool = True
    allow_shell: bool = False
    allow_web: bool = False


@dataclasses.dataclass
class Caps:
    requests: int | None = 40
    episode_seconds: float = 1800.0
    flowgraph_seconds: float = 300.0
    compaction: bool = True


def work_root() -> Path:
    home = Path(os.environ.get("GRC_ADAPTER_HOME") or Path(__file__).resolve().parents[2])
    return Path(os.environ.get("GRC_ADAPTER_WORK") or home / "runs")


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _hashes(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and ".grc_agent" not in rel.parts:
            out[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _file_changes(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {"added": sorted(set(after) - set(before)),
            "changed": sorted(k for k in set(after) & set(before) if after[k] != before[k]),
            "removed": sorted(set(before) - set(after))}


def reply_text(output: Any) -> str:
    from grc_agent.agent import GrcAgentResponse
    if isinstance(output, GrcAgentResponse):
        text = output.explanation.strip()
        if output.actions_taken:
            text += "\n\nActions taken:\n" + "\n".join(f"- {a}" for a in output.actions_taken)
        return text
    return "" if output is None else str(output)


def _tool_calls(messages: list[Any]) -> list[dict[str, Any]]:
    from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart
    results: dict[str, Any] = {}
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, (ToolReturnPart, RetryPromptPart)) and part.tool_call_id:
                    results.setdefault(part.tool_call_id, part)
    calls = []
    for msg in messages:
        if not isinstance(msg, ModelResponse):
            continue
        for part in msg.parts:
            if not isinstance(part, ToolCallPart):
                continue
            ret = results.get(part.tool_call_id or "")
            if isinstance(ret, RetryPromptPart):
                outcome, seen = "retry", ret.model_response()
            elif isinstance(ret, ToolReturnPart):
                outcome, seen = str(getattr(ret, "outcome", "success")), ret.content
            else:
                outcome, seen = "unanswered", None
            calls.append({"tool": part.tool_name, "tool_call_id": part.tool_call_id,
                          "arguments": part.args, "outcome": outcome, "result": seen})
    return calls


def _decide(requests: Any, policy: Policy, log: list[dict[str, Any]]) -> Any:
    """Answer GRC-Agent's approval requests the way the policy says (its sidebar's role)."""
    from pydantic_ai.tools import DeferredToolResults, ToolApproved, ToolDenied
    if getattr(requests, "calls", None):
        names = ", ".join(c.tool_name for c in requests.calls)
        raise RuntimeError(f"the agent asked for external tool results ({names}); not supported")
    approvals: dict[str, Any] = {}
    for call in requests.approvals:
        name = call.tool_name
        if name == "change_graph":
            ok, why = policy.allow_graph_edits, "flowgraph edits are turned off for this run"
        elif name == "run_flowgraph":
            ok, why = policy.allow_run, "flowgraph runs are turned off for this run"
        elif name in SHELL_TOOLS:
            ok, why = policy.allow_shell, "shell commands are turned off for this run"
        else:
            ok, why = False, f"the adapter does not approve {name!r}"
        approvals[call.tool_call_id] = (ToolApproved() if ok else
                                        ToolDenied(f"Denied by the grc-agent-adapter approval policy: {why}."))
        log.append({"tool": name, "tool_call_id": call.tool_call_id,
                    "decision": "approved" if ok else "denied"})
    return DeferredToolResults(approvals=approvals)


class _RunFailed(Exception):
    def __init__(self, kind: str, end_reason: str, error: str, messages: list[Any]) -> None:
        super().__init__(error)
        self.kind, self.end_reason, self.error, self.messages = kind, end_reason, error, messages


async def _run_once(agent: Any, deadline: float, **kwargs: Any) -> Any:
    from pydantic_ai import capture_run_messages
    from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UsageLimitExceeded
    with capture_run_messages() as captured:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            return await asyncio.wait_for(agent.run(**kwargs), timeout=remaining)
        except UsageLimitExceeded as e:
            raise _RunFailed("cap_failure", "cap:requests", str(e), list(captured)) from e
        except asyncio.TimeoutError as e:
            raise _RunFailed("cap_failure", "cap:episode_seconds", "episode time limit reached",
                             list(captured)) from e
        except (ModelHTTPError, ModelAPIError, OSError) as e:
            raise _RunFailed("infrastructure_error", "model server error",
                             f"{type(e).__name__}: {e}", list(captured)) from e
        except Exception as e:
            raise _RunFailed("agent_error", "agent run failed", f"{type(e).__name__}: {e}",
                             list(captured)) from e


async def _turns(agent: Any, deps: Any, prompts: list[str], history: list[Any], policy: Policy,
                 caps: Caps, conversation_id: str) -> dict[str, Any]:
    from pydantic_ai import UsageLimits
    from pydantic_ai.tools import DeferredToolRequests
    from pydantic_ai.usage import RunUsage

    usage = RunUsage()
    limits = UsageLimits(request_limit=caps.requests) if caps.requests else None
    deadline = time.monotonic() + caps.episode_seconds
    common = {"deps": deps, "usage": usage, "usage_limits": limits, "conversation_id": conversation_id}
    turns: list[dict[str, Any]] = []
    approvals: list[dict[str, Any]] = []
    out = {"status": "completed", "end_reason": "agent_finished", "error": None}
    for prompt in prompts:
        try:
            res = await _run_once(agent, deadline, user_prompt=prompt, message_history=history or None, **common)
            while isinstance(res.output, DeferredToolRequests):
                results = _decide(res.output, policy, approvals)
                res = await _run_once(agent, deadline, message_history=res.all_messages(),
                                      deferred_tool_results=results, **common)
        except _RunFailed as e:
            if len(e.messages) > len(history):
                history = e.messages
            out.update(status=e.kind, end_reason=e.end_reason, error=e.error)
            turns.append({"user": prompt, "reply": None})
            break
        except RuntimeError as e:
            out.update(status="agent_error", end_reason="agent run failed", error=str(e))
            turns.append({"user": prompt, "reply": None})
            break
        history = list(res.all_messages())
        turns.append({"user": prompt, "reply": reply_text(res.output)})
    out.update(history=history, turns=turns, approvals=approvals)
    return out


def _restore(session_dir: Path, *, parent: Path | None, graph: Path | None, files: Path | None) -> tuple[Any, list[Any]]:
    """Set up the live project folder and the flowgraph; return (flow_graph, history)."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter
    project = session_dir / "project"
    if parent is not None:
        shutil.rmtree(project, ignore_errors=True)
        shutil.copytree(parent / "project", project, symlinks=True)
        summary = _read_json(parent / "summary.json")
        in_project = (summary.get("graph") or {}).get("file_in_project")
        fg = grc_platform.load_flow_graph(parent / "graph.grc",
                                          file_path=project / in_project if in_project else None)
        history = list(ModelMessagesTypeAdapter.validate_json((parent / "messages.json").read_bytes()))
        return fg, history
    project.mkdir(parents=True, exist_ok=True)
    if files is not None:
        shutil.copytree(files, project, dirs_exist_ok=True, symlinks=True)
    if graph is not None:
        target = project / graph.name
        if graph.resolve() != target.resolve():
            shutil.copy2(graph, target)
        return grc_platform.load_flow_graph(target, file_path=target), []
    return grc_platform.new_flow_graph(), []


def _attach_log(path: Path) -> logging.Handler:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    return handler


def run_call(session_dir: Path, message: str, *, spec: ModelSpec, policy: Policy, caps: Caps,
             then: list[str] = (), parent: Path | None = None, graph: Path | None = None,
             files: Path | None = None, model: Any = None) -> dict[str, Any]:
    """Run one call and write its folder; returns summary.json's content."""
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    session_file = session_dir / "session.json"
    session = _read_json(session_file) if session_file.exists() else {"name": session_dir.name, "calls": []}
    call_no = len(session["calls"]) + 1
    call_dir = session_dir / f"call-{call_no:02d}"
    call_dir.mkdir()
    request = {"message": message, "then": list(then), "model": dataclasses.asdict(spec),
               "policy": dataclasses.asdict(policy), "caps": dataclasses.asdict(caps),
               "graph": str(graph) if graph else None, "files": str(files) if files else None,
               "parent": str(parent) if parent else None}
    _write_json(call_dir / "request.json", request)
    handler = _attach_log(call_dir / "agent.log")
    try:
        return asyncio.run(_run_call(session_dir, session, call_no, call_dir, message, list(then),
                                     spec, policy, caps, parent, graph, files, model))
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


async def _run_call(session_dir: Path, session: dict, call_no: int, call_dir: Path, message: str,
                    then: list[str], spec: ModelSpec, policy: Policy, caps: Caps, parent: Path | None,
                    graph: Path | None, files: Path | None, model: Any) -> dict[str, Any]:
    from grc_agent.adapter import flow_graph_content_hash, inspect_graph
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    started = time.monotonic()
    grc_platform.install()
    fg, history = _restore(session_dir, parent=parent, graph=graph, files=files)
    project = (session_dir / "project").resolve()
    deps = MacFlowgraphDeps(fg, max_run_seconds=caps.flowgraph_seconds)
    before = _hashes(project)
    start_len = len(history)
    agent_info: dict[str, Any] = {}
    tools: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="grc-adapter-env-") as env_dir:
        run_env = build.write_run_env(Path(env_dir) / ".env", provider=spec.provider,
                                      model=spec.model, base_url=spec.base_url)
        try:
            agent, agent_info = build.build_executor(deps=deps, project_dir=project, run_env=run_env,
                                                     allow_web=policy.allow_web,
                                                     compaction=caps.compaction, model=model)
            try:
                tools = await build.freeze_tools(agent, deps)
            except Exception as e:  # private pydantic-ai API: record, do not fail the call
                agent_info["tools_snapshot_error"] = f"{type(e).__name__}: {e}"
            result = await _turns(agent, deps, [message, *then], history, policy, caps,
                                  conversation_id=f"grc-adapter-{session_dir.name}")
        except Exception as e:
            result = {"status": "infrastructure_error", "end_reason": "setup failed",
                      "error": f"{type(e).__name__}: {e}", "history": history,
                      "turns": [{"user": message, "reply": None}], "approvals": []}
        finally:
            await deps.aclose()
            os.environ.pop("GRC_AGENT_ENV", None)

    history = result["history"]
    new_messages = history[start_len:]
    (call_dir / "messages.json").write_bytes(ModelMessagesTypeAdapter.dump_json(history, indent=2))
    _write_json(call_dir / "tools.json", tools)

    platform = grc_platform.install()
    platform.save_flow_graph(str(call_dir / "graph.grc"), fg)
    fg.validate()
    view = inspect_graph(fg)
    view["valid"] = fg.is_valid()
    view["errors"] = _error_lines(fg, limit=50)
    _write_json(call_dir / "graph_view.json", view)

    file_path = Path(deps.file_path) if deps.file_path else None
    in_project = None
    if file_path is not None:
        with contextlib.suppress(ValueError):
            in_project = str(file_path.resolve().relative_to(project))
    matches = (flow_graph_content_hash(fg) == _sha256_file(file_path)) if file_path and file_path.exists() else None
    shutil.copytree(project, call_dir / "project", symlinks=True)

    model_info = agent_info.get("model") or {"provider": spec.provider, "model_name": spec.model}
    usage = trace.sum_usage(new_messages)
    totals: dict[str, int] = {}
    for c in session["calls"]:
        for k, v in (c.get("usage") or {}).items():
            totals[k] = totals.get(k, 0) + v
    for k, v in usage.items():
        totals[k] = totals.get(k, 0) + v
    final_reply = next((t["reply"] for t in reversed(result["turns"]) if t["reply"] is not None), None)
    summary = {
        "session": session_dir.name, "call": call_no, "run_dir": str(call_dir),
        "parent": str(parent) if parent else None,
        "status": result["status"], "end_reason": result["end_reason"], "error": result["error"],
        "final_reply": final_reply, "turns": result["turns"],
        "tool_calls": _tool_calls(new_messages), "approvals": result["approvals"],
        "flowgraph_runs": deps.runs,
        "graph": {"file_in_project": in_project, "file_matches_graph": matches,
                  "valid": view["valid"], "errors": view["errors"]},
        "files": _file_changes(before, _hashes(project)),
        "usage": usage, "session_totals": totals,
        "model": model_info, "policy": dataclasses.asdict(policy), "caps": dataclasses.asdict(caps),
        "agent": agent_info, "versions": grc_platform.versions(),
        "seconds": round(time.monotonic() - started, 1),
    }
    _write_json(call_dir / "summary.json", summary)

    meta = {"status": summary["status"], "end_reason": summary["end_reason"],
            "policy": summary["policy"], "caps": summary["caps"],
            "grc_agent_commit": summary["versions"].get("grc_agent_commit"),
            "gnuradio": summary["versions"].get("gnuradio")}
    row = trace.session_row(session=session_dir.name, call=call_no, status=summary["status"],
                            model=model_info, messages=history, tools=tools, meta=meta)
    trace.write_trace(call_dir / "trace.jsonl", row)
    (call_dir / "transcript.md").write_text(
        trace.transcript(row["messages"], f"{session_dir.name}, call {call_no}"), encoding="utf-8")

    session["calls"].append({"call": call_no, "run_dir": str(call_dir),
                             "parent": str(parent) if parent else None, "message": message,
                             "status": summary["status"], "end_reason": summary["end_reason"],
                             "usage": usage, "model": model_info})
    _write_json(session_dir / "session.json", session)
    return summary
