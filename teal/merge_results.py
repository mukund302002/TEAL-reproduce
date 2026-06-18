"""
Merge two comparison_summary CSVs (different tasks, same methods/columns).

Usage:
  python merge_results.py <csv1> <csv2> <output_csv>

Example:
  python merge_results.py \\
      ../analysis/comparison/alpaca/comparison_summary_alpaca_sp50.csv \\
      ../analysis/comparison/alpaca_extra/comparison_summary_alpaca_sp50.csv \\
      ../analysis/comparison/alpaca_combined/comparison_summary_alpaca_sp50.csv
"""

import csv
import sys
from pathlib import Path


def merge_csv(file1, file2, output):
    with open(file1) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows1 = list(reader)

    with open(file2) as f:
        reader = csv.reader(f)
        next(reader)          # skip header (same columns expected)
        rows2 = list(reader)

    all_rows = sorted(rows1 + rows2, key=lambda r: r[0])

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    print(f"Merged {len(rows1)} + {len(rows2)} rows → {output}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_results.py <csv1> <csv2> <output_csv>")
        sys.exit(1)
    merge_csv(sys.argv[1], sys.argv[2], sys.argv[3])
