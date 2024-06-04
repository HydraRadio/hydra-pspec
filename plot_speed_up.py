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


def get_speed_up_data(data: list[dict], timer: str):
    "Extract execution time and baselines/rank"
    bl_per_rank = []
    ex_time = []
    for d in data:
        bl_per_rank.append(d["num_baselines"]/d["num_ranks"])
        ex_time.append((d["rank_0_timers"][timer]))

    sorted_indices = sorted(range(len(bl_per_rank)), key=lambda i: bl_per_rank[i], reverse=True)
    ex_time = [ex_time[i] for i in sorted_indices]
    bl_per_rank.sort(reverse=True)
    speed_up = [ex_time[0]/t for t in ex_time]
    return speed_up, bl_per_rank


def plot_speed_up(speed_up: list, x: list):
    """Plot speed up vs variable x (e.g. baselines/rank)"""
    fig, ax = plt.subplots()
    ax.plot(x, speed_up, "o--", label="Results")
    ax.axline((x[0], speed_up[0]), slope=-1, linestyle=":", color="k", label="1:1")
    ax.set_ylabel("Speed up")
    ax.set_xlabel("Baselines/rank")
    ax.xaxis.set_inverted(True)
    plt.legend()
    plt.savefig(results_dir.joinpath("speed_up.svg"))


def get_time_and_ranks(data: list[dict], timer: str):
    "Extract execution time and number of rank"
    n_ranks = []
    ex_time = []
    for d in data:
        n_ranks.append(d["num_ranks"])
        ex_time.append((d["rank_0_timers"][timer]))

    sorted_indices = sorted(range(len(n_ranks)), key=lambda i: n_ranks[i])
    ex_time = [ex_time[i] for i in sorted_indices]
    n_ranks.sort()

    return ex_time, n_ranks


def plot_time_vs_ranks(ex_time: list, x: list):
    """Plot speed up vs variable x (e.g. number of ranks)"""
    fig, ax = plt.subplots()
    ax.plot(x, ex_time, "o--", label="Total run time")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Number of ranks")
    plt.legend()
    plt.savefig(results_dir.joinpath("time_vs_ranks.svg"))

speed_up, bl_per_rank = get_speed_up_data(timings, "total")
plot_speed_up(speed_up, bl_per_rank)
ex_time, n_ranks = get_time_and_ranks(timings, "total")
plot_time_vs_ranks(ex_time, n_ranks)
