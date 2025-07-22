from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder
from typing import List
from ffcv.pipeline.operation import Operation


def ffcv_dataloader(
    baton_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    order: OrderOption = OrderOption.RANDOM,
    transform=None,    
):
    # Below is from FFCV documentaion on Order Options:
    # ---------------------------------------------------------
    # Truly random shuffling (shuffle=True in PyTorch) -> high memory usage
    # ORDERING = OrderOption.RANDOM

    # Unshuffled (i.e., served in the order the dataset was written) -> medium memory usage
    # ORDERING = OrderOption.SEQUENTIAL

    # Memory-efficient but not truly random loading -> low memory usage
    # ORDERING = OrderOption.QUASI_RANDOM
    
    # pipeline: List[Operation] = [
        
    # ]
    
    loader = Loader(
        baton_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
    )
    raise NotImplementedError("FFCV dataloader is not implemented yet.")
        