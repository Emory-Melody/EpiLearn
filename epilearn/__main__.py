"""Entry point for running benchmark as a module: python -m epilearn.benchmark"""

import sys

from epilearn.benchmark import main

if __name__ == "__main__":
    sys.exit(main())
