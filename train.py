import torch
import torchvision
from torch import nn
from model.resnet import ResnetSimCLR    
from data.augment import SimCLRAugmentation
from data.loader import load_dataset
from utils.write_ffcv import write_dataset_to_ffcv

from utils.seed import seed_everything

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = 'resnet18'
image_size = 32
num_workers = 4

model = ResnetSimCLR(backbone=backbone, output_dim=128).to(device)

sim_clr_transform = SimCLRAugmentation(image_size=image_size)

raw_data = load_dataset(
    data_dir=None,  
    output_path="datasets/cifar10",
    resolution=image_size,
    max_images=None,
    transform=sim_clr_transform.transform
)

write_dataset_to_ffcv(
    dataset=raw_data,
    output_path="datasets/cifar10/simclr_train.ffcv",
    resolution=image_size,
    max_images=None,
    num_workers=num_workers
)


