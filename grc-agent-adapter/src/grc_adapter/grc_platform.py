"""GNU Radio's headless platform, built the way this Mac's GNU Radio needs.

GRC-Agent's `adapter.graph.get_platform()` builds `gnuradio.grc.core.platform.Platform` without
`install_prefix`. Ubuntu's GNU Radio accepts that; the conda-forge GNU Radio 3.10.12 used on macOS
does not (`Config.__init__() missing 1 required positional argument: 'install_prefix'`). The GUI
path (`get_gui_platform`) already passes `install_prefix=gr.prefix()`, so we build the core
platform the same way and hand it to GRC-Agent's module-level cache before anything asks for it.
The grc-agent repository itself is not modified.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

_PLATFORM: Any = None


def install() -> Any:
    """Build (once) and install the core Platform into GRC-Agent's cache; return it."""
    global _PLATFORM
    if _PLATFORM is not None:
        return _PLATFORM
    from gnuradio import gr
    from gnuradio.grc.core.platform import Platform

    from grc_agent.adapter import graph as grc_graph

    kwargs = dict(name="grc_agent", prefs=gr.prefs(), version=gr.version(),
                  version_parts=(gr.major_version(), gr.api_version(), gr.minor_version()))
    try:
        platform = Platform(install_prefix=gr.prefix(), **kwargs)
    except TypeError:
        platform = Platform(**kwargs)
    platform.build_library()
    grc_graph._PLATFORM = platform
    _PLATFORM = platform
    return platform


def new_flow_graph() -> Any:
    """GRC's File > New: options id 'default' plus the samp_rate variable, never saved."""
    platform = install()
    fg = platform.make_flow_graph()
    fg.import_data(platform.parse_flow_graph(""))
    fg.rewrite()
    fg.grc_file_path = ""
    return fg


def load_flow_graph(path: Path | str, *, file_path: Path | str | None = None) -> Any:
    """Open a .grc file. `file_path` is the path the page should report as its own (a restored
    snapshot opens from one place but belongs to another); None means untitled."""
    platform = install()
    fg = platform.make_flow_graph()
    fg.import_data(platform.parse_flow_graph(str(path)))
    fg.rewrite()
    fg.grc_file_path = str(Path(file_path).resolve()) if file_path else ""
    return fg


def versions() -> dict[str, Any]:
    """GNU Radio version and the GRC-Agent checkout's commit, for the run record."""
    from gnuradio import gr
    repo = os.environ.get("GRC_AGENT_REPO", "")
    commit = None
    if repo:
        try:
            commit = subprocess.run(["git", "-C", repo, "rev-parse", "--short=7", "HEAD"],
                                    capture_output=True, text=True, timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            commit = None
    return {"gnuradio": gr.version(), "grc_agent_repo": repo, "grc_agent_commit": commit}
