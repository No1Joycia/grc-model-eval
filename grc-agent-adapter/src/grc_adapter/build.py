"""Build GRC-Agent's real executor, headless, for one call.

Like grc-agent-headless (agent_build.py), this calls GRC-Agent's own
`agent_factory.build_agents_from_cfg` and keeps the executor role, so the system prompt, output
types and validator, capabilities and tools are GRC-Agent's own objects. The factory runs with
these scoped substitutions, each undone when it returns:

  1. `grc_agent.native_canvas` imports GTK; the factory only takes `NativeFlowgraphProxy` from it
     as the deps type. A GTK-free stand-in module exposes MacFlowgraphDeps under that name.
  2. Web search / fetch (default): the same two tools (name, description, schema), local-only,
     whose body returns OFFLINE_MESSAGE. `allow_web=True` keeps GRC-Agent's real ones.
  3. `compaction=False`: GRC-Agent's compaction capability is replaced by one with no hooks, so
     the history reaches the model unmodified (grc-agent-headless's "benchmark" preset).
  4. `model` (tests, selftest): the given pydantic-ai model instead of GRC-Agent's own build.

Settings come from a per-call `.env` (GRC_AGENT_ENV) that names the provider, model and base URL
only. API keys are never written there: GRC-Agent's `settings.resolve_key` falls back to the
process environment, which the CLI fills from the adapter's `.env` file.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic_ai.capabilities import AbstractCapability

OFFLINE_MESSAGE = "offline: web access is turned off in this grc-agent-adapter run"
NATIVE_CANVAS = "grc_agent.native_canvas"

# Providers whose model GRC-Agent builds from an OpenAI-style base URL the user chooses.
URL_PROVIDERS = {"openai_compatible": "OPENAI_COMPATIBLE_BASE_URL", "ollama_local": "OLLAMA_BASE_URL"}


@dataclasses.dataclass
class CompactionOff(AbstractCapability[Any]):
    """Stands in for GRC-Agent's compaction capability: no hooks, history unmodified."""


def key_env_var(provider: str) -> str | None:
    """The environment variable GRC-Agent reads the provider's API key from."""
    from grc_agent.ui.providers import PROVIDER_API_KEY
    return PROVIDER_API_KEY.get(provider) or None


def default_model(provider: str) -> str:
    from grc_agent import settings as s
    return s._DEFAULT_MODELS[s._PROVIDER_MODEL_KEY[provider]]


def valid_providers() -> tuple[str, ...]:
    from grc_agent import settings as s
    return tuple(p for p in s._VALID_PROVIDERS if p != "openai_codex")


def write_run_env(path: Path, *, provider: str, model: str, base_url: str | None) -> Path:
    """The per-call GRC-Agent settings file (no secrets)."""
    from grc_agent import settings as s
    lines = [f"GRC_PROVIDER={provider}", "GRC_EMBED_BACKEND=lexical",
             f"{s._PROVIDER_ENV_VAR[provider]}={model}"]
    if provider in URL_PROVIDERS and base_url:
        lines.append(f"{URL_PROVIDERS[provider]}={base_url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _offline_copy(tool: Any) -> Any:
    async def offline(*_args: Any, **_kwargs: Any) -> str:
        return OFFLINE_MESSAGE

    clone = copy.copy(tool)
    clone.function = offline
    clone.function_schema = dataclasses.replace(tool.function_schema, function=offline, is_async=True)
    return clone


def offline_web_capabilities() -> tuple[Any, Any]:
    from pydantic_ai.capabilities import WebFetch, WebSearch

    from grc_agent.agent import web_fetch_cap, web_search_cap
    return (WebSearch(native=False, local=_offline_copy(web_search_cap.local)),
            WebFetch(native=False, local=_offline_copy(web_fetch_cap.local)))


@contextlib.contextmanager
def _swapped(obj: Any, **attrs: Any) -> Iterator[None]:
    old = {name: getattr(obj, name) for name in attrs}
    for name, value in attrs.items():
        setattr(obj, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(obj, name, value)


@contextlib.contextmanager
def gtk_free_native_canvas(proxy_cls: type) -> Iterator[None]:
    stub = types.ModuleType(NATIVE_CANVAS, "grc-agent-adapter stand-in for grc_agent.native_canvas")
    stub.NativeFlowgraphProxy = proxy_cls          # type: ignore[attr-defined]

    def _missing(name: str) -> Any:
        raise AttributeError(f"{NATIVE_CANVAS} is a GTK-free stand-in; {name!r} needs GTK")

    stub.__getattr__ = _missing                    # type: ignore[attr-defined]
    saved = sys.modules.get(NATIVE_CANVAS)
    sys.modules[NATIVE_CANVAS] = stub
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop(NATIVE_CANVAS, None)
        else:
            sys.modules[NATIVE_CANVAS] = saved


async def freeze_tools(agent: Any, deps: Any = None) -> list[dict[str, Any]]:
    """The agent's tool snapshot as {name, description, parameters}, sorted by name, output tools
    left out: the `tools` field of Qoherent's SFT row format (grc-agent-headless agent_build)."""
    from pydantic_ai import RunContext
    from pydantic_ai.agent import _build_run_capabilities
    from pydantic_ai.usage import RunUsage
    ctx = RunContext(deps=deps, model=agent.model, usage=RunUsage(),
                     capabilities=_build_run_capabilities(agent._root_capability))
    tools = await agent._get_toolset().get_tools(ctx)
    frozen = []
    for name, tool in sorted(tools.items()):
        tool_def = getattr(tool, "tool_def", None)
        if getattr(tool_def, "kind", None) == "output":
            continue
        frozen.append({"name": str(getattr(tool_def, "name", name) or name),
                       "description": str(getattr(tool_def, "description", "") or ""),
                       "parameters": getattr(tool_def, "parameters_json_schema", None)})
    return frozen


def build_executor(*, deps: Any, project_dir: Path, run_env: Path, allow_web: bool = False,
                   compaction: bool = True, model: Any = None) -> tuple[Any, dict[str, Any]]:
    """GRC-Agent's executor for one call; returns (agent, info). `run_env` must already exist."""
    os.environ["GRC_AGENT_ENV"] = str(run_env)
    from grc_agent import agent_factory as af
    from grc_agent import fs_tools
    from grc_agent import settings as grc_settings

    if grc_settings.env_path().resolve() != Path(run_env).resolve():
        raise RuntimeError(f"GRC-Agent reads {grc_settings.env_path()}, not {run_env}")
    cfg = grc_settings.load_settings()

    project_dir = Path(project_dir).resolve()
    fs_tools.set_active_graph_providers(grc_path_fn=lambda: deps.file_path or None,
                                        flow_graph_fn=lambda: deps._get_target(),
                                        project_dir_fn=lambda: project_dir)

    substitutions = [f"{NATIVE_CANVAS}: GTK-free stand-in (NativeFlowgraphProxy = {type(deps).__name__})"]
    patches: dict[str, Any] = {}
    if model is not None:
        patches["_build_model"] = lambda _cfg, _http_client: model
        substitutions.append(f"model: given {type(model).__name__}")
    if not compaction:
        patches["_build_compaction_capability"] = lambda _cfg, agent_name="grc_chat": CompactionOff()
        substitutions.append("compaction: off")
    if not allow_web:
        search_cap, fetch_cap = offline_web_capabilities()
        patches["web_search_cap"] = search_cap
        patches["web_fetch_cap"] = fetch_cap
        substitutions.append("web_search/web_fetch: offline")

    with gtk_free_native_canvas(type(deps)), _swapped(af, **patches):
        bundle = af.build_agents_from_cfg(cfg)
    if bundle.model_build_error:
        raise RuntimeError(f"GRC-Agent could not build the model: {bundle.model_build_error}")
    agent = bundle.executor
    provider, base_url, model_name = af.describe_model(agent.model)
    info = {
        "factory": "grc_agent.agent_factory.build_agents_from_cfg (executor role)",
        "agent_name": agent.name,
        "model": {"class": type(agent.model).__name__, "provider": provider or cfg["provider"],
                  "base_url": base_url, "model_name": model_name or cfg.get("model")},
        "substitutions": substitutions,
        "web": "real" if allow_web else "offline",
        "compaction": "grc_agent" if compaction else "off",
    }
    return agent, info
