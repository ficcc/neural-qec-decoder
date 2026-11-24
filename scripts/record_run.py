"""Record a run:  python -m scripts.record_run --label NAME [--notes TEXT] [--run-tests] [--extra '{"cost_usd": 3.7}']"""
import argparse
import json
import re
import subprocess
import sys

from src.runlog import record_run


def run_tests() -> dict:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    n = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|error|errors|skipped|deselected)", tail)}
    return {"summary_line": tail, "passed": n.get("passed", 0), "failed": n.get("failed", 0),
            "errors": n.get("error", 0) + n.get("errors", 0), "exit_code": r.returncode}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--extra", default="{}", help="JSON dict of free-form facts (cost, hardware, provenance)")
    ap.add_argument("--run-tests", action="store_true")
    ap.add_argument("--root", default="runs")
    a = ap.parse_args(argv)
    out = record_run(a.label, root=a.root, notes=a.notes, extra=json.loads(a.extra),
                     test_result=run_tests() if a.run_tests else None)
    print("recorded ->", out)


if __name__ == "__main__":
    main()
