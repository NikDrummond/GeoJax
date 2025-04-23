# runtests.py
import subprocess
import sys
import os

def run_tests(with_coverage=True):
    cmd = [sys.executable, "-m", "pytest"]  # 👈 Use `python -m pytest`
    if with_coverage:
        cmd += ["--cov=GeoJax", "--cov-report=term-missing"]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    with_coverage = "--no-cov" not in sys.argv
    run_tests(with_coverage)
