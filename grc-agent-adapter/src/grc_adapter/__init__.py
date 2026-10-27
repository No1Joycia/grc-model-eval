"""grc-agent-adapter: GRC-Agent's real executor, headless, on macOS.

The macOS counterpart of grc-agent-headless (GRC-Bench Tier-A runner, Linux only). It drives the
same `agent_factory.build_agents_from_cfg` executor of the local GRC-Agent checkout, with a
GTK-free stand-in for the desktop canvas, and writes every result as files. There is no
bubblewrap sandbox on macOS, so safety comes from the approval policy instead (README.md).
"""

__version__ = "0.1.0"
