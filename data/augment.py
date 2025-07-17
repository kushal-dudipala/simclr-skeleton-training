from torchvision import transforms
from utils.seed import seed_everything

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SimCLRAugmentation:
    def __init__(
        self,
        image_size: int,
        color_jitter: float = 0.4,
        gaussian_blur: float = 0.1,
        horizontal_flip: bool = True,
        seed: int = 42,
    ):
        """
        defaulted to initial args of SimCLR paper:
        https://arxiv.org/abs/2002.05709
        """

        self.image_size = image_size
        assert 0 <= color_jitter <= 1, "Color jitter should be between 0 and 1"
        self.color_jitter = color_jitter
        assert 0 <= gaussian_blur <= 1, "Gaussian blur should be between 0 and 1"
        self.gaussian_blur = gaussian_blur
        self.horizontal_flip = horizontal_flip

        seed_everything(seed)

        # Apply transforms step by step as in original SimCLR paper
        self.transform = transforms.Compose(
            [
                # Step 1: Random resized crop
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                # Step 2: Random horizontal flip
                (
                    transforms.RandomHorizontalFlip(p=0.5)
                    if horizontal_flip
                    else transforms.Lambda(lambda x: x)
                ),
                # Step 3: Color jitter (applied with probability 0.8)
                (
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                brightness=color_jitter,
                                contrast=color_jitter,
                                saturation=color_jitter,
                                hue=0.1,
                            )
                        ],
                        p=0.8,
                    )
                    if color_jitter > 0
                    else transforms.Lambda(lambda x: x)
                ),
                # Step 4: Random grayscale conversion
                transforms.RandomGrayscale(p=0.2),
                # Step 5: Gaussian blur (applied with probability based on gaussian_blur param)
                (
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                        p=gaussian_blur,
                    )
                    if gaussian_blur > 0
                    else transforms.Lambda(lambda x: x)
                ),
                # Step 6: Convert to tensor and normalize
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
