"""Set up data directory, populated with identical baseline data."""
import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser("Make data directory for scaling tests")
parser.add_argument("--data_dir", type=str, help="Directory containing baseline data (all files in one dir)")
parser.add_argument("--num_baselines", type=int, help="Number of baseline directories to create")
parser.add_argument("--dest_dir", type=str, help="Destination directory to copy data into")

args = parser.parse_args()

data_dir = Path(args.data_dir)
num_baselines = args.num_baselines
dest_dir = Path(args.dest_dir)

dest_dir.mkdir()

for n in range(args.num_baselines):
    for data_type in ["eor-cov", "fgmodes", "noise"]:
        dest = Path(dest_dir, data_type, "0-"+str(n + 1))
        dest.mkdir(parents=True)

        match data_type:
            case "eor-cov":
                shutil.copy2(Path(data_dir, "eor-cov.npy"), dest)
            case "fgmodes":
                shutil.copy2(Path(data_dir, "fgmodes.npy"), dest)
            case "noise":
                shutil.copy2(Path(data_dir, "noise.npy"), dest)
                shutil.copy2(Path(data_dir, "noise-cov.npy"), dest)

vis_data_name = "vis-eor-fgs.uvh5"
shutil.copy2(Path(data_dir, vis_data_name), Path(dest_dir, vis_data_name))
