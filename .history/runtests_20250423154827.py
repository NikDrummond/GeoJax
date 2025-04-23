# runtests.py
import sys
import subprocess

def run_tests(with_coverage: bool = True):
    """
    Run the test suite using pytest. Optionally includes coverage.

    Parameters
    ----------
    with_coverage : bool
        If True, includes coverage report.
    """
    # Determine the base pytest command
    cmd = [sys.executable, "-m", "pytest"]

    # Add coverage arguments if requested
    if with_coverage:
        cmd += ["--cov=GeoJax", "--cov-report=term-missing"]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    # Default behavior is to include coverage unless --no-cov is passed
    use_cov = "--no-cov" not in sys.argv
    run_tests(with_coverage=use_cov)
