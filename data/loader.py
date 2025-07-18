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

def load_dataset(
    data_dir: str,
    output_path: str,
    resolution: int = 28,
    max_images: int = None,
):

    if data_dir is None:
        data_dir = download_and_prepare_mnist("./mnist_imagefolder")
        print(f"\033[92mNo data directory provided. Downloaded MNIST to {data_dir}.\033[0m")
    else:
        assert os.path.isdir(
            data_dir
        ), f"Data directory {data_dir} does not exist."

    print(f"\033[92mUsing data directory: {data_dir}\033[0m")

    assert not os.path.exists(output_path) or os.path.isfile(
        output_path
    ), f"Output path {output_path} must be a file."
    assert resolution > 0, "Resolution must be a positive integer."
    assert (
        max_images is None or max_images > 0
    ), "Max images must be a positive integer or None."


    write_dataset_to_ffcv(
        data_dir=data_dir,
        output_path=output_path,
        resolution=resolution,
        max_images=max_images,
    )
