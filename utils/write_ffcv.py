import torch
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField


def write_dataset_to_ffcv(
    dataset: torch.utils.data.Dataset,
    output_path: str,
    resolution: tuple = (224, 224),
    max_images: int = None,
    num_workers: int = 4,
):
    """
    Takes a pytorch dataset and writes it to an FFCV format file as a .ffcv file.
    
    """
    
    if max_images:
        dataset = torch.utils.data.Subset(dataset, range(min(len(dataset), max_images)))

    writer = DatasetWriter(
        output_path,
        {
            "image": RGBImageField(write_mode="proportion", max_resolution=resolution),
            "label": IntField(),
        },
        num_workers=num_workers,
    )

    print(f"Writing {len(dataset)} images to {output_path} in ffcv format...")
    writer.from_indexed_dataset(dataset)
    print("Done.")
