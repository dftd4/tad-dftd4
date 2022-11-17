"""
Entry point for command line interface via `python -m <prog>`.
"""

import sys

from .__version__ import __version__
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
