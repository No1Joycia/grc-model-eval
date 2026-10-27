"""Command line: run, continue, batch, show, selftest (README.md)."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from . import build, session as sess
from .session import Caps, ModelSpec, Policy

REFUSED, FAILED = 2, 1


class Refused(Exception):
    """A request the adapter will not run (bad option, missing file, missing API key)."""


# -- options ------------------------------------------------------------------------------------
def _model_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--provider", help="GRC-Agent provider id, e.g. cohere, openrouter, "
                   "openai_compatible (default), ollama_local, ollama_cloud")
    g.add_argument("--model", help="model name (default: GRC-Agent's default for the provider)")
    g.add_argument("--base-url", help="server URL for openai_compatible / ollama_local "
                   "(default http://127.0.0.1:8080/v1 / http://localhost:11434)")
    g.add_argument("--api-key-env", help="openai_compatible only: the environment variable that "
                   "holds the key (its value is never written anywhere)")
    g.add_argument("--env-file", help="where API keys are read from (default: grc-agent-adapter/.env)")


def _policy_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("approvals (macOS has no sandbox: README.md, Safety)")
    g.add_argument("--allow-shell", action="store_true", default=None,
                   help="approve the agent's shell commands (default: denied)")
    g.add_argument("--allow-web", action="store_true", default=None,
                   help="real web search/fetch (default: the tools answer 'offline')")
    g.add_argument("--no-graph-edits", action="store_true", default=None,
                   help="deny change_graph (read-only runs)")
    g.add_argument("--no-run", action="store_true", default=None, help="deny run_flowgraph")
    g = p.add_argument_group("limits")
    g.add_argument("--max-requests", help="model requests per call, or 'none' (default 40)")
    g.add_argument("--episode-seconds", type=float, help="wall-clock limit per call (default 1800)")
    g.add_argument("--flowgraph-seconds", type=float, help="hard limit per flowgraph run (default 300)")
    g.add_argument("--no-compaction", action="store_true", default=None,
                   help="turn GRC-Agent's history compaction off")


def _resolve(args: argparse.Namespace, base: dict[str, Any] | None = None) -> tuple[ModelSpec, Policy, Caps]:
    """Options on top of `base` (the continued call's request.json), on top of the defaults."""
    base = base or {}
    m = dict(base.get("model") or {})
    if args.provider:
        if args.provider != m.get("provider"):
            m = {}
        m["provider"] = args.provider
    provider = m.get("provider") or "openai_compatible"
    if provider not in build.valid_providers():
        raise Refused(f"unknown provider {provider!r}; one of: {', '.join(build.valid_providers())}")
    spec = ModelSpec(provider=provider,
                     model=args.model or m.get("model") or build.default_model(provider),
                     base_url=args.base_url or m.get("base_url"),
                     api_key_env=args.api_key_env or m.get("api_key_env"))
    if spec.provider == "openai_compatible" and not spec.base_url:
        spec.base_url = "http://127.0.0.1:8080/v1"

    pol = Policy(**(base.get("policy") or {}))
    if args.allow_shell:
        pol.allow_shell = True
    if args.allow_web:
        pol.allow_web = True
    if args.no_graph_edits:
        pol.allow_graph_edits = False
    if args.no_run:
        pol.allow_run = False

    caps = Caps(**(base.get("caps") or {}))
    if args.max_requests is not None:
        caps.requests = None if args.max_requests.lower() == "none" else int(args.max_requests)
    if args.episode_seconds is not None:
        caps.episode_seconds = args.episode_seconds
    if args.flowgraph_seconds is not None:
        caps.flowgraph_seconds = args.flowgraph_seconds
    if args.no_compaction:
        caps.compaction = False
    return spec, pol, caps


def _load_keys(spec: ModelSpec, env_file: str | None) -> None:
    """Put the API key the provider needs into this process's environment, from the env file
    unless the shell already has it. Only *_API_KEY variables (and --api-key-env) are read."""
    from dotenv import dotenv_values
    path = Path(env_file) if env_file else Path(os.environ.get("GRC_ADAPTER_HOME", ".")) / ".env"
    if env_file and not path.exists():
        raise Refused(f"env file {path} does not exist")
    if path.exists():
        for k, v in dotenv_values(path).items():
            if v and (k.endswith("_API_KEY") or k == spec.api_key_env):
                os.environ.setdefault(k, v)
    if spec.provider == "openai_compatible":
        if spec.api_key_env:
            if not os.environ.get(spec.api_key_env):
                raise Refused(f"environment variable {spec.api_key_env} is not set (shell or {path})")
            os.environ["OPENAI_COMPATIBLE_API_KEY"] = os.environ[spec.api_key_env]
        return
    var = build.key_env_var(spec.provider)
    if var and spec.provider != "ollama_local" and not os.environ.get(var):
        raise Refused(f"provider {spec.provider} needs {var}: put it in {path} or export it")


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "session"


def _print_report(summary: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return
    g = summary["graph"]
    u = summary["usage"]
    print(f"session {summary['session']}, call {summary['call']}: {summary['status']} ({summary['end_reason']})")
    print(f"run folder: {summary['run_dir']}")
    print(f"graph: {'valid' if g['valid'] else 'INVALID'}, "
          f"{('project/' + g['file_in_project']) if g['file_in_project'] else 'untitled'}")
    changed = [f"{k} {', '.join(v)}" for k, v in summary["files"].items() if v]
    print(f"files: {'; '.join(changed) or 'no changes'}")
    print(f"requests {u.get('requests', 0)}, tokens in {u.get('input_tokens', 0)} / out "
          f"{u.get('output_tokens', 0)}, tool calls {len(summary['tool_calls'])}, {summary['seconds']} s")
    if summary.get("error"):
        print(f"error: {summary['error']}")
    print("\nfinal reply:")
    print(summary.get("final_reply") or "(none)")


# -- commands -----------------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    spec, pol, caps = _resolve(args)
    _load_keys(spec, args.env_file)
    graph = Path(args.graph).expanduser().resolve() if args.graph else None
    files = Path(args.files).expanduser().resolve() if args.files else None
    for label, p, want_dir in (("--graph", graph, False), ("--files", files, True)):
        if p is not None and (not p.exists() or p.is_dir() != want_dir):
            raise Refused(f"{label} {p}: not an existing {'folder' if want_dir else 'file'}")
    root = Path(args.work).expanduser() if args.work else sess.work_root()
    name = _safe_name(args.name) if args.name else time.strftime("%Y%m%d-%H%M%S")
    if (root / name).exists():
        raise Refused(f"session {name} already exists in {root}: use continue, or another --name")
    summary = sess.run_call(root / name, args.message, spec=spec, policy=pol, caps=caps,
                            then=args.then or [], graph=graph, files=files)
    _print_report(summary, args.json)
    return FAILED if summary["status"] == "infrastructure_error" else 0


def cmd_continue(args: argparse.Namespace) -> int:
    parent = Path(args.run_dir).expanduser().resolve()
    if not (parent / "summary.json").exists():
        raise Refused(f"{parent} is not a call folder (no summary.json)")
    base = json.loads((parent / "request.json").read_text(encoding="utf-8"))
    spec, pol, caps = _resolve(args, base)
    _load_keys(spec, args.env_file)
    summary = sess.run_call(parent.parent, args.message, spec=spec, policy=pol, caps=caps,
                            then=args.then or [], parent=parent)
    _print_report(summary, args.json)
    return FAILED if summary["status"] == "infrastructure_error" else 0


def cmd_batch(args: argparse.Namespace) -> int:
    """One fresh session per JSONL record; one answer row per record (run_trial.py's format)."""
    spec, pol, caps = _resolve(args)
    _load_keys(spec, args.env_file)
    src = Path(args.questions).expanduser()
    out = Path(args.out).expanduser()
    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = records[: args.limit] if args.limit else records
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    root = (Path(args.work).expanduser() if args.work else sess.work_root()) / _safe_name(args.name or out.stem)
    todo = [r for r in records if str(r[args.id_key]) not in done]
    print(f"{len(records)} questions | {len(done)} already answered | {len(todo)} to go | "
          f"{spec.provider}:{spec.model} | sessions in {root}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    infra_errors = 0
    with open(out, "a", encoding="utf-8") as fh:
        for i, rec in enumerate(todo, 1):
            qid = str(rec[args.id_key])
            session_dir = root / _safe_name(qid)
            if session_dir.exists():
                shutil.rmtree(session_dir)        # an interrupted earlier attempt at this question
            summary = sess.run_call(session_dir, rec[args.question_key], spec=spec, policy=pol, caps=caps)
            u = summary["usage"]
            row = {"id": qid, "model": f"{spec.provider}:{spec.model}",
                   "answer": summary.get("final_reply") or "",
                   "finish_reason": "stop" if summary["status"] == "completed" else summary["status"],
                   "usage": {"prompt_tokens": u.get("input_tokens", 0),
                             "completion_tokens": u.get("output_tokens", 0),
                             "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
                             "reasoning_tokens": u.get("reasoning_tokens", 0),
                             "requests": u.get("requests", 0)},
                   "seconds": summary["seconds"], "status": summary["status"],
                   "end_reason": summary["end_reason"], "error": summary.get("error"),
                   "tool_calls": [c["tool"] for c in summary["tool_calls"]],
                   "run_dir": summary["run_dir"]}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)}] {qid} {summary['status']:<20} {u.get('requests', 0):>2} req "
                  f"{u.get('output_tokens', 0):>6} tok out  {len(summary['tool_calls']):>2} tools  "
                  f"{summary['seconds']:7.1f}s", flush=True)
            if summary["status"] == "infrastructure_error":
                infra_errors += 1
                if infra_errors >= args.stop_after_errors:
                    print(f"stopping: {infra_errors} infrastructure errors in a row "
                          f"(last: {summary.get('error')})", file=sys.stderr)
                    return FAILED
            else:
                infra_errors = 0
    print(f"DONE: answers in {out}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser()
    path = run_dir / ("transcript.md" if args.transcript else "summary.json")
    if not path.exists():
        raise Refused(f"{path} not found")
    print(path.read_text(encoding="utf-8"))
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from .selftest import selftest
    return selftest(keep=args.keep)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent", description="GRC-Agent without its window, on macOS (README.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="start a session")
    p.add_argument("message")
    p.add_argument("--graph", help="a .grc file, copied into the project folder and opened")
    p.add_argument("--files", help="a folder copied into the project folder")
    p.add_argument("--name", help="session name (default: date and time)")
    p.add_argument("--then", action="append", help="a follow-up message (repeatable)")
    p.add_argument("--work", help="where sessions are written (default: grc-agent-adapter/runs)")
    p.add_argument("--json", action="store_true", help="print summary.json")
    _model_options(p)
    _policy_options(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("continue", help="the next message of a session")
    p.add_argument("run_dir", help="the call folder to continue from")
    p.add_argument("message")
    p.add_argument("--then", action="append", help="a follow-up message (repeatable)")
    p.add_argument("--json", action="store_true", help="print summary.json")
    _model_options(p)
    _policy_options(p)
    p.set_defaults(fn=cmd_continue)

    p = sub.add_parser("batch", help="one fresh session per question in a JSONL file")
    p.add_argument("questions", help="JSONL file, one object per line")
    p.add_argument("--out", required=True, help="answers JSONL (appended; reruns skip done ids)")
    p.add_argument("--id-key", default="id")
    p.add_argument("--question-key", default="question")
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.add_argument("--name", help="batch folder name under the work folder (default: --out's stem)")
    p.add_argument("--work", help="where sessions are written (default: grc-agent-adapter/runs)")
    p.add_argument("--stop-after-errors", type=int, default=3,
                   help="stop after N infrastructure errors in a row (default 3)")
    _model_options(p)
    _policy_options(p)
    p.set_defaults(fn=cmd_batch)

    p = sub.add_parser("show", help="print a call's summary.json (or --transcript)")
    p.add_argument("run_dir")
    p.add_argument("--transcript", action="store_true")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("selftest", help="check the setup with a scripted model (no API key)")
    p.add_argument("--keep", action="store_true", help="keep the self-test's session folder")
    p.set_defaults(fn=cmd_selftest)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return REFUSED
