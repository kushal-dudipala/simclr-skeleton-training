import argparse
from utils.write_ffcv import write_dataset_to_ffcv
import os

parser = argparse.ArgumentParser(
    description="Convert image dataset to FFCV .beton format"
)
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Path to image folder",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Output .beton file path (e.g., imagenet_train.beton)",
)
parser.add_argument(
    "--resolution", type=int, default=224, help="Max image resolution (default: 224)"
)
parser.add_argument(
    "--max_images", type=int, default=None, help="Optionally limit number of images"
)

args = parser.parse_args()

assert os.path.isdir(args.data_dir), f"Data directory {args.data_dir} does not exist."
assert (not os.path.exists(args.output) or
        os.path.isfile(args.output)), f"Output path {args.output} must be a file."
assert args.resolution > 0, "Resolution must be a positive integer."
assert args.max_images is None or args.max_images > 0, "Max images must be a positive integer or None."

write_dataset_to_ffcv(
    data_dir=args.data_dir,
    output_path=args.output,
    resolution=args.resolution,
    max_images=args.max_images,
)
