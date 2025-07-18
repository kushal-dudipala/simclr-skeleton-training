import argparse
from utils.write_ffcv import write_dataset_to_ffcv
import os
import torchvision
from torchvision.transforms import ToPILImage
import shutil


def load_dataset(
    data_dir: str,
    output_path: str,
    resolution: int = 28,
    max_images: int = None,
    transform=None,
):

    if data_dir is None:
        dataset = torchvision.datasets.CIFAR10(
            "datasets/cifar10", download=True, transform=transform
        )
        print(
            f"\033[92mNo data directory provided. Downloaded MNIST to {data_dir}.\033[0m"
        )
    else:
        assert os.path.isdir(data_dir), f"Data directory {data_dir} does not exist."
        dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)

    print(f"\033[92mUsing data directory: {data_dir}\033[0m")

    assert not os.path.exists(output_path) or os.path.isfile(
        output_path
    ), f"Output path {output_path} must be a file."
    assert resolution > 0, "Resolution must be a positive integer."
    assert (
        max_images is None or max_images > 0
    ), "Max images must be a positive integer or None."

    return dataset
