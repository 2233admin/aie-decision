"""Public JSON process entry point for the integrated Fermi kernel."""

from __future__ import annotations

import os
from typing import Sequence

from .agent_cli import main as agent_main


def main(argv: Sequence[str] | None = None) -> int:
    previous = os.environ.get("AIE_AGENT_KERNEL")
    if previous is None:
        os.environ["AIE_AGENT_KERNEL"] = "aie_decision.fermi_kernel:build"
    try:
        return agent_main(argv)
    finally:
        if previous is None:
            os.environ.pop("AIE_AGENT_KERNEL", None)
        else:
            os.environ["AIE_AGENT_KERNEL"] = previous


if __name__ == "__main__":
    raise SystemExit(main())
