import torch
import torchvision
from torch import nn
from model.resnet import ResnetSimCLR
from data.augment import SimCLRAugmentation
from data.loader import load_dataset
from utils.write_ffcv import write_dataset_to_ffcv
from model.loss import SimCLRLoss
from utils.seed import seed_everything
from torch.optim import Adam
from tqdm import tqdm

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = "resnet18"
image_size = 32
num_workers = 4
optimizer = "adam"
learning_rate = 0.001
weight_decay = 1e-4
momentum = 0.9
epochs = 10

model = ResnetSimCLR(backbone=backbone, output_dim=128).to(device)

# simclr data augmentation, follows SimCLR paper
sim_clr_transform = SimCLRAugmentation(image_size=image_size)

# load raw dataset, either from local directory or download CIFAR-10 dataset
raw_data = load_dataset(
    data_dir=None,
    output_path="datasets/cifar10",
    resolution=image_size,
    max_images=None,
)

####################################################################################
#                                                                                  #
#          moved all ffcv files to ffcv.py for self containment of ffcv.           #
#                                                                                  #
####################################################################################


# convert raw dataset to FFCV format
write_dataset_to_ffcv(
    dataset=raw_data,
    output_path="datasets/cifar10/simclr_train.ffcv",
    resolution=image_size,
    max_images=None,
    num_workers=num_workers,
)

# load dataset from FFCV


ffcv_dataset = torchvision.datasets.FFCVDataset(
    "datasets/cifar10/simclr_train.ffcv",
    num_workers=num_workers,
    transform=sim_clr_transform.transform,
)

# loss function
criterion = SimCLRLoss().to(device)

# optimizer
if optimizer == "adam":
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(),
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
elif optimizer == "muon":
    raise NotImplementedError("Muon optimizer is not implemented yet.")

for epoch in tqdm(range(epochs), desc="Training Epochs"):
    total_loss = 0.0
    for batch in ffcv_dataset:
        x0, x1 = batch[0]
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(ffcv_dataset)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
