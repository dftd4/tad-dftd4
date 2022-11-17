"""
Command line interface and argparser.
"""

from __future__ import annotations
import argparse


from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence | None = None) -> int:
    # get command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("number", type=float, help="Number to square.")
    args = parser.parse_args(argv)

    # print result

    return 0
