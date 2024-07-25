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
parser.add_argument("--timer", type=str, help="Which timer to compare with ideal scaling")


args = parser.parse_args()

if args.results_dir:
    # Combine files from multiple runs
    timing_logs = []
    results_dir = Path(args.results_dir).resolve()
    for dir_item in results_dir.iterdir():
        if dir_item.is_dir():
            file = dir_item.joinpath("timings.json")
            if file.is_file():
                with open(file) as f:
                    data = json.load(f)
                timing_logs.append(data)

    with open(results_dir.joinpath("combined_timings.json"), "w") as f:
        # Save summary file
        json.dump(timing_logs, f, indent=2)

if args.summary_file:
    with open(Path(args.summary_file).resolve()) as f:
        timing_logs = json.load(f)
    results_dir = Path(args.summary_file).parent.resolve()


def process_timings(data: list[dict]):
    "Extract execution time and number of ranks"
    n_ranks = []
    time_load = []
    time_scatter = []
    time_process = []
    time_barrier = []
    time_total = []

    for d in data:
        n_ranks.append(d["num_ranks"])
        time_load.append((d["rank_0_timers"]["load_data"]))
        time_scatter.append((d["rank_0_timers"]["scatter"]))
        time_process.append((d["rank_0_timers"]["process"]))
        time_barrier.append((d["rank_0_timers"]["barrier"]))
        time_total.append((d["rank_0_timers"]["total"]))

    sorted_indices = sorted(range(len(n_ranks)), key=lambda i: n_ranks[i])
    time_load = [time_load[i] for i in sorted_indices]
    time_scatter = [time_scatter[i] for i in sorted_indices]
    time_process = [time_process[i] for i in sorted_indices]
    time_barrier = [time_barrier[i] for i in sorted_indices]
    time_total = [time_total[i] for i in sorted_indices]
    n_ranks.sort()


    timings = {"load": time_load,
               "scatter": time_scatter,
               "process": time_process,
               "barrier": time_barrier,
               "total": time_total,
               "n_ranks": n_ranks,
              }

    time_for_speedup = timings[timer]
    speed_up = [time_for_speedup[0] / t for t in time_for_speedup]
    timings["speed_up"] = speed_up

    return timings


def plot_speed_up_ranks(speed_up: list, n_ranks: list, key_timer: str):
    """Plot speed up vs number of ranks

    <key_timer> is the timer to compare with ideal scaling
    """
    fig, ax = plt.subplots()
    ax.plot(n_ranks, speed_up, "o--", label=key_timer)
    slope = 1 / n_ranks[0]
    ax.axline((n_ranks[0], speed_up[0]), slope=slope, linestyle=":", color="k", label="Ideal " + key_timer)
    ax.set_ylabel("Speed up")
    ax.set_xlabel("Number of ranks")
    ax.set_ylim(None, n_ranks[-1]*slope)
    plt.legend()
    plt.savefig(results_dir.joinpath(f"speed_up-{key_timer}.svg"))


def plot_time_vs_ranks(timings: dict, key_timer: str):
    """Plot speed up vs number of ranks

    <key_timer> is the timer to compare with ideal scaling
    """
    fig, ax = plt.subplots()
    t_total = timings["total"]
    n_ranks = timings["n_ranks"]
    t_process = timings["process"]
    t_barrier = timings["barrier"]
    t_scatter = timings["scatter"]
    t_load = timings["load"]
    ax.plot(n_ranks, t_load, label="load")
    ax.plot(n_ranks, t_scatter, label="scatter")
    ax.plot(n_ranks, t_barrier, label="barrier")
    ax.plot(n_ranks, t_process, "+-", label="process")
    ax.plot(n_ranks, t_total, "o--", label="total")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Number of ranks")
    t_key = timings[key_timer]
    ideal_time = [t_key[0] * n_ranks[0]/val for val in n_ranks]
    ax.plot(n_ranks, ideal_time, ":", label="ideal " + key_timer, color="k")
    plt.legend()
    plt.savefig(results_dir.joinpath(f"time_vs_ranks-{key_timer}.svg"))


if args.timer:
    timer = args.timer
else:
    timer = "total"

timings = process_timings(timing_logs)
plot_time_vs_ranks(timings, key_timer=timer)
plot_speed_up_ranks(timings["speed_up"], timings["n_ranks"], key_timer=timer)
