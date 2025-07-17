import argparse
from utils.write_ffcv import write_dataset_to_ffcv
import os
from torchvision.datasets import MNIST
from torchvision.transforms import ToPILImage
import shutil


def download_and_prepare_mnist(target_dir):
    if os.path.exists(target_dir):
        print(f"MNIST directory {target_dir} already exists.")
        return target_dir

    print(f"Downloading MNIST into {target_dir}...")

    dataset = MNIST(root="./temp_mnist", download=True, train=True)

    for label in range(10):
        os.makedirs(os.path.join(target_dir, str(label)), exist_ok=True)

    for idx, (img, label) in enumerate(dataset):
        img_path = os.path.join(target_dir, str(label), f"{idx}.png")
        ToPILImage(img).save(img_path)

    shutil.rmtree("./temp_mnist")
    print("MNIST downloaded and prepared.")
    return target_dir



parser = argparse.ArgumentParser(
    description="Convert image dataset to FFCV .beton format"
)

parser.add_argument(
    "--data_dir", type=str, default=None, help="Path to image folder. If not provided, MNIST will be downloaded."
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Output .beton file path",
)
parser.add_argument(
    "--resolution",
    type=int,
    default=28,
    help="Max image resolution (default: 28 for MNIST)",
)
parser.add_argument(
    "--max_images", type=int, default=None, help="Optionally limit number of images"
)

args = parser.parse_args()

if args.data_dir is None:
    args.data_dir = download_and_prepare_mnist("./mnist_imagefolder")
    print(f"\033[92mNo data directory provided. Downloaded MNIST to {args.data_dir}.\033[0m")
else:
    assert os.path.isdir(
        args.data_dir
    ), f"Data directory {args.data_dir} does not exist."

print(f"\033[92mUsing data directory: {args.data_dir}\033[0m")

assert not os.path.exists(args.output) or os.path.isfile(
    args.output
), f"Output path {args.output} must be a file."
assert args.resolution > 0, "Resolution must be a positive integer."
assert (
    args.max_images is None or args.max_images > 0
), "Max images must be a positive integer or None."


write_dataset_to_ffcv(
    data_dir=args.data_dir,
    output_path=args.output,
    resolution=args.resolution,
    max_images=args.max_images,
)
