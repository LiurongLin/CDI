#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow direct execution via `python3 scripts/coronagraph/slm_gui.py`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coronagraph.html_slm_gui import launch_html_gui


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the browser SLM optimization GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    launch_html_gui(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
