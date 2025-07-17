from torchvision.datasets import ImageFolder
import torch
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField


def write_dataset_to_ffcv(
    data_dir: str,
    output_path: str,
    resolution: tuple = (224, 224),
    max_images: int = None,
):
    dataset = ImageFolder(data_dir)
    if max_images:
        dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), max_images)))

    writer = DatasetWriter(
        output_path,
        {
            "image": RGBImageField(write_mode="proportion", max_resolution=resolution),
            "label": IntField(),
        },
        num_workers=4,
    )

    print(f"Writing {len(dataset)} images from {data_dir} to {output_path}")
    writer.from_indexed_dataset(dataset)
    print("Done.")
