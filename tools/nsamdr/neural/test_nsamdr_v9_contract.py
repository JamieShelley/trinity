#!/usr/bin/env python3
"""Historical command shim: run the V14 production contract tests."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from test_nsamdr_v14 import NSAMDRV14ContractTests


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NSAMDRV14ContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
