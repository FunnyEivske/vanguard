"""Module entrypoint for `python -m vanguard`."""

import sys

from vanguard.cli import main

if __name__ == "__main__":
    sys.exit(main())

