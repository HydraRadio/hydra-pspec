"""Copy data for a single baseline, renaming the antenna pair string for however many copies are required"""
import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser("Make duplicate directories linking back to one baseline for testing scaling")
parser.add_argument("--data_dir", type=str, help="Directory containing baseline data to copy")
parser.add_argument("--num_baselines", type=int, help="Number of copies to make")
parser.add_argument("--dest_dir", type=str, help="Destination directory to copy data into")
parser.add_argument("--baseline_pair", type=str, help="Baseline antenna pair to copy")

args = parser.parse_args()

data_dir = Path(args.data_dir)
num_baselines = args.num_baselines
dest_dir = Path(args.dest_dir)

dest_dir.mkdir()

for n in range(args.num_baselines):
    for data_type in ["eor-cov", "fgmodes", "noise"]:
        dest = Path(dest_dir, data_type, "0-" + str(n + 1))
        dest.mkdir(parents=True)

        for src in Path(data_dir, data_type, args.baseline_pair).glob("*.npy"):
            shutil.copy2(src, dest)
vis_data_name = "vis-eor-fgs.uvh5"
shutil.copy2(Path(data_dir, vis_data_name), Path(dest_dir, vis_data_name))