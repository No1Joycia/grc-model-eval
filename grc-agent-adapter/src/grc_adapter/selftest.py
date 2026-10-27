"""`./agent selftest`: GRC-Agent's real executor driven by a scripted model (no API key, no
network), through every part of the adapter a real run uses."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import grc_platform, session as sess
from .session import Caps, ModelSpec, Policy

FAKE_KEY = "sk-grc-adapter-selftest-0123456789"


def _scripted(steps: list[Any]) -> Any:
    """A FunctionModel that plays `steps` in order within each user turn."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart
    from pydantic_ai.models.function import FunctionModel

    def fn(messages: list[Any], _info: Any) -> Any:
        last_user = max(i for i, m in enumerate(messages)
                        if isinstance(m, ModelRequest) and any(isinstance(p, UserPromptPart) for p in m.parts))
        n = sum(1 for m in messages[last_user:] if isinstance(m, ModelResponse))
        return ModelResponse(parts=steps[min(n, len(steps) - 1)])

    return FunctionModel(fn, model_name="selftest-script")


def _call(name: str, **args: Any) -> Any:
    from pydantic_ai.messages import ToolCallPart
    return ToolCallPart(tool_name=name, args=args)


def _text(text: str) -> Any:
    from pydantic_ai.messages import TextPart
    return TextPart(content=text)


def selftest(*, keep: bool = False) -> int:
    results: list[tuple[bool, str]] = []

    def check(ok: bool, label: str) -> None:
        results.append((bool(ok), label))
        print(f"{'ok  ' if ok else 'FAIL'}  {label}", flush=True)

    work = Path(tempfile.mkdtemp(prefix="grc-adapter-selftest-"))
    os.environ["OPENAI_COMPATIBLE_API_KEY"] = FAKE_KEY       # must reach no output file
    try:
        platform = grc_platform.install()
        check(len(platform.blocks) > 100, f"GNU Radio platform built headless ({len(platform.blocks)} blocks)")

        fixture = work / "fixture"
        fixture.mkdir()
        fg = grc_platform.new_flow_graph()
        fg.options_block.params["id"].set_value("selftest")
        fg.options_block.params["generate_options"].set_value("no_gui")
        fg.options_block.params["run_options"].set_value("run")       # run to completion
        fg.rewrite()
        platform.save_flow_graph(str(fixture / "selftest.grc"), fg)

        steps1 = [
            [_call("change_graph", reason="Build a short tone chain.",
                   add_blocks=[{"block_id": "analog_sig_source_x", "instance_name": "src", "params": {"type": "float"}},
                               {"block_id": "blocks_head", "instance_name": "head",
                                "params": {"type": "float", "num_items": "48000"}},
                               {"block_id": "blocks_file_sink", "instance_name": "sink",
                                "params": {"type": "float", "file": "selftest.dat"}}],
                   add_connections=["src:0->head:0", "head:0->sink:0"])],
            [_call("save_graph")],
            [_call("run_flowgraph", action="start", wait=True, timeout_seconds=60)],
            [_call("get_run_log")],
            [_call("query_knowledge", query="signal source frequency", domain="catalog")],
            [_call("run_command", command="echo should-be-denied")],
            [_text("Built and ran the tone chain; selftest.dat was written.")],
        ]
        spec = ModelSpec(provider="openai_compatible", model="selftest-script",
                         base_url="http://127.0.0.1:9/v1")
        s1 = sess.run_call(work / "selftest", "Build a tone into a file sink, save it and run it.",
                           spec=spec, policy=Policy(), caps=Caps(), graph=fixture / "selftest.grc",
                           model=_scripted(steps1))
        check(s1["status"] == "completed", f"GRC-Agent ran headless (status {s1['status']}, "
              f"{s1.get('error') or s1['end_reason']})")
        by_tool = {c["tool"]: c for c in s1["tool_calls"]}
        ok_tools = all(by_tool.get(t, {}).get("outcome") == "success"
                       for t in ("change_graph", "save_graph", "run_flowgraph", "get_run_log"))
        check(ok_tools, "change_graph, save_graph, run_flowgraph and get_run_log executed")
        runs = s1["flowgraph_runs"]
        check(bool(runs) and runs[-1]["return_code"] == 0,
              f"the flowgraph ran in GNU Radio (return code {runs[-1]['return_code'] if runs else None})")
        dat = Path(s1["run_dir"]) / "project" / "selftest.dat"
        check(dat.exists() and dat.stat().st_size == 48000 * 4,
              f"the flowgraph wrote project/selftest.dat ({dat.stat().st_size if dat.exists() else 0} bytes)")
        kn = by_tool.get("query_knowledge", {})
        check(kn.get("outcome") == "success" and "analog_sig_source_x" in json.dumps(kn.get("result")),
              "query_knowledge answered from GRC-Agent's local catalog")
        denied = [a for a in s1["approvals"] if a["tool"] == "run_command"]
        check(denied and denied[0]["decision"] == "denied", "shell command denied by the default policy")
        g = s1["graph"]
        check(g["valid"] and g["file_in_project"] == "selftest.grc" and g["file_matches_graph"],
              "graph.grc / graph_view.json: valid, the saved file matches the flowgraph in memory")
        run_dir = Path(s1["run_dir"])
        names = ("summary.json", "transcript.md", "trace.jsonl", "messages.json", "tools.json",
                 "graph.grc", "graph_view.json", "request.json")
        check(all((run_dir / n).exists() for n in names), "all output files written")
        row = json.loads((run_dir / "trace.jsonl").read_text())
        check(len(row.get("tools") or []) >= 15 and row["system"],
              f"trace.jsonl has the system prompt and {len(row.get('tools') or [])} tool schemas")

        steps2 = [[_call("inspect_graph")], [_text("The flowgraph has three blocks.")]]
        s2 = sess.run_call(work / "selftest", "How many blocks are there now?", spec=spec,
                           policy=Policy(), caps=Caps(), parent=run_dir, model=_scripted(steps2))
        insp = next((c for c in s2["tool_calls"] if c["tool"] == "inspect_graph"), {})
        check(s2["status"] == "completed" and s2["call"] == 2 and "head" in json.dumps(insp.get("result")),
              "continue: the next call has the conversation and the same flowgraph")
        users = [m for m in json.loads((Path(s2["run_dir"]) / "trace.jsonl").read_text())["messages"]
                 if m["role"] == "user"]
        check(len(users) == 2, "continue: trace.jsonl holds both user messages")

        # A "Prompt for Exit" graph runs until stopped: a bounded run, then a background run + stop.
        fg.options_block.params["id"].set_value("forever")
        fg.options_block.params["run_options"].set_value("prompt")
        fg.rewrite()
        platform.save_flow_graph(str(fixture / "forever.grc"), fg)
        steps3 = [
            [_call("change_graph", reason="A tone that never ends.",
                   add_blocks=[{"block_id": "analog_sig_source_x", "instance_name": "src", "params": {"type": "float"}},
                               {"block_id": "blocks_throttle2", "instance_name": "thr", "params": {"type": "float"}},
                               {"block_id": "blocks_null_sink", "instance_name": "sink", "params": {"type": "float"}}],
                   add_connections=["src:0->thr:0", "thr:0->sink:0"])],
            [_call("save_graph")],
            [_call("run_flowgraph", action="start", wait=True, stop_after_seconds=2)],
            [_call("run_flowgraph", action="start", wait=False)],
            [_call("run_flowgraph", action="stop")],
            [_text("Ran it bounded, then in the background, then stopped it.")],
        ]
        s3 = sess.run_call(work / "selftest-stop", "Run it for two seconds, then start and stop it.",
                           spec=spec, policy=Policy(), caps=Caps(), graph=fixture / "forever.grc",
                           model=_scripted(steps3))
        statuses = [json.loads(c["result"]).get("status") for c in s3["tool_calls"]
                    if c["tool"] == "run_flowgraph" and c["outcome"] == "success"]
        check(statuses == ["stopped_after_timeout", "started", "stopped"],
              f"bounded run, background run and stop ({', '.join(map(str, statuses)) or s3.get('error')})")

        leaked = [str(p) for p in work.rglob("*") if p.is_file() and FAKE_KEY.encode() in p.read_bytes()]
        check(not leaked, "the API key is in no output file" + (f" (found in {leaked})" if leaked else ""))
    except Exception as e:  # report, do not crash
        import traceback
        traceback.print_exc()
        check(False, f"self-test crashed: {type(e).__name__}: {e}")
    finally:
        os.environ.pop("OPENAI_COMPATIBLE_API_KEY", None)
        if keep:
            print(f"kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    passed = sum(ok for ok, _ in results)
    print(f"self-test: {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1
