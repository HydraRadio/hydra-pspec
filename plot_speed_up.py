import argparse
from pathlib import Path
import json

parser = argparse.ArgumentParser("Combine timing files and plot speed up.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--results_dir", type=str, help="Directory containing output from multiple runs (in subdirectories)")
group.add_argument("--summary_file", type=str, help="File containing timings for all runs")

args = parser.parse_args()

if args.results_dir:
    # Combine files from multiple runs
    timings = []
    results_dir = Path(args.results_dir).resolve()
    for subdir in results_dir.iterdir():
        file = subdir.joinpath("timings.json")
        with open(file) as f:
            data = json.load(f)
        timings.append(data)

    with open(results_dir.joinpath("combined_timings.json"), "w") as f:
        json.dump(timings, f)

if args.summary_file:
    with open(Path(args.summary_file).resolve()) as f:
        data = json.load(f)
