#!/usr/bin/env python3
import sys
from pathlib import Path
import unittest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    loader = unittest.TestLoader()
    suite = loader.discover(str(project_root / 'tests'))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    main()
