"""
Entry point for command line interface via `python -m <prog>`.
"""

import sys

from . import main

if __name__ == "__main__":
    sys.exit(main())
