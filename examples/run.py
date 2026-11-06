import argparse
import csv
import json
import sys
from pathlib import Path

from deflated_sharpe import summarize

def main():
    parser = argparse.ArgumentParser(description="Deflated Sharpe summary from trial excess returns")
    parser.add_argument("input", nargs="?", type=Path, default=Path(__file__).resolve().parent / "input.json")
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text())
        rows = summarize(**data)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    if args.format == "json":
        print(json.dumps(rows, indent=2, allow_nan=False))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    else:
        trial_fields = ["trial", "sharpe", "skewness", "pearson_kurtosis", "cluster",
                        "representative_sharpe", "psr", "dsr"]
        shared = [(key, rows[0][key]) for key in rows[0] if key not in trial_fields]
        print("Search inputs and estimates")
        print_table(["field", "value"], shared)
        print("\nTrial results")
        print_table(trial_fields, [[row[key] for key in trial_fields] for row in rows])

def print_table(headers, records):
    rendered = [[f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
                for row in records]
    widths = [max(len(key), *(len(row[i]) for row in rendered)) for i,key in enumerate(headers)]
    print(" | ".join(key.ljust(width) for key,width in zip(headers,widths)))
    print("-+-".join("-"*width for width in widths))
    for row in rendered:
        print(" | ".join(value.ljust(width) for value,width in zip(row,widths)))

if __name__ == "__main__":
    main()
