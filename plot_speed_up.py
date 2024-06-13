"""
Plot speed up for fixed problem size, varying number of baselines/rank.
Usage:
python plot_speed_up.py --results_dir=dir_containing_multiple_runs_as_subdirs
python plot_speed_up.py --summary_file=file_containing_timings_from_all_runs.json
"""

import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Combine timing files and plot speed up.")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--results_dir", type=str, help="Directory containing output from multiple runs (in subdirectories)")
group.add_argument("--summary_file", type=str, help="File containing timings for all runs")
parser.add_argument("--timer", type=str, help="Which timer to use")


args = parser.parse_args()

if args.results_dir:
    # Combine files from multiple runs
    timings = []
    results_dir = Path(args.results_dir).resolve()
    for dir_item in results_dir.iterdir():
        if dir_item.is_dir():
            file = dir_item.joinpath("timings.json")
            with open(file) as f:
                data = json.load(f)
            timings.append(data)

    with open(results_dir.joinpath("combined_timings.json"), "w") as f:
        # Save summary file
        json.dump(timings, f, indent=2)

if args.summary_file:
    with open(Path(args.summary_file).resolve()) as f:
        timings = json.load(f)
    results_dir = Path(args.summary_file).parent.resolve()


def plot_speed_up_ranks(speed_up: list, n_ranks: list):
    """Plot speed up vs number of ranks"""
    fig, ax = plt.subplots()
    ax.plot(n_ranks, speed_up, "o--", label=timer)
    slope = 1/n_ranks[0]
    ax.axline((n_ranks[0], speed_up[0]), slope=slope, linestyle=":", color="k", label="Ideal")
    ax.set_ylabel("Speed up")
    ax.set_xlabel("Number of ranks")
    plt.legend()
    plt.savefig(results_dir.joinpath("speed_up.svg"))


def process_timings(data: list[dict], timer: str):
    "Extract execution time and number of ranks"
    n_ranks = []
    ex_time = []
    for d in data:
        n_ranks.append(d["num_ranks"])
        ex_time.append((d["rank_0_timers"][timer]))

    sorted_indices = sorted(range(len(n_ranks)), key=lambda i: n_ranks[i])
    ex_time = [ex_time[i] for i in sorted_indices]
    n_ranks.sort()
    speed_up = [ex_time[0]/t for t in ex_time]

    return ex_time, n_ranks, speed_up


def plot_time_vs_ranks(ex_time: list, x: list):
    """Plot speed up vs variable x (e.g. number of ranks)"""
    fig, ax = plt.subplots()
    ax.plot(x, ex_time, "o--", label=timer)
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Number of ranks")
    plt.legend()
    plt.savefig(results_dir.joinpath("time_vs_ranks.svg"))

if args.timer:
    timer = args.timer
else:
    timer = "total"

ex_time, n_ranks, speed_up = process_timings(timings, timer)
plot_time_vs_ranks(ex_time, n_ranks)
plot_speed_up_ranks(speed_up, n_ranks)
