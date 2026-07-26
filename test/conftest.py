"""Make the repository root importable from `test/`.

The suite imports both the installed package (`multiplex.*`) and root-level
scripts that are deliberately not part of it (`try_engine`, `try_vision`). The
latter only resolve when the repo root is on `sys.path`, which it is when pytest
runs from the root but is not when a test file is run directly
(`python test/test_scheduler.py`) — that puts `test/` on the path instead.
Inserting it here covers both.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
