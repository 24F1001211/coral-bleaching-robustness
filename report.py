"""Build all charts and an HTML report from the finished runs.

  python report.py --runs runs --out report      # then open report/index.html
"""
import argparse

import viz

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    viz.save_all(a.runs, a.out)
