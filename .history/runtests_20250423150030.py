# runtests.py
# Run GeoJax test suite with optional coverage report

import subprocess
import sys

def run_tests(with_coverage=True):
    cmd = ["pytest"]
    if with_coverage:
        cmd += ["--cov=GeoJax", "--cov-report=term-missing"]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    with_coverage = "--no-cov" not in sys.argv
    run_tests(with_coverage)
