from torchvision import transforms
from utils.seed import seed_everything

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SimCLRAugmentation:
    def __init__(
        self,
        image_size: int,
        color_jitter: float = 0.4,
        gaussian_blur_prob: float = 0.5,
        min_scale: float = 0.08,
        horizontal_flip: bool = True,
        random_gray_prob: float = 0.2,
    ):
        """
        defaulted to initial args of SimCLR paper:
        https://arxiv.org/abs/2002.05709

        """
        assert 0.0 <= color_jitter <= 1.0, "Color jitter should be between 0 and 1"
        assert (
            0.0 <= gaussian_blur_prob <= 1.0
        ), "Gaussian blur should be between 0 and 1"
        assert (
            0.0 <= random_gray_prob <= 1.0
        ), "Random gray probability should be between 0 and 1"
        assert 0.0 <= min_scale <= 1.0, "Min scale should be between 0 and 1"

        # kernel size as in SimCLR: 0.1 * image_size (must be odd)
        ksize = int(0.1 * image_size)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize

        # Apply transforms step by step as in original SimCLR paper
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(min_scale, 1.0)),
                (
                    transforms.RandomHorizontalFlip()
                    if horizontal_flip
                    else transforms.Lambda(lambda x: x)
                ),
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
                ),
                transforms.RandomGrayscale(p=random_gray_prob),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=ksize, sigma=(0.1, 2.0))],
                    p=gaussian_blur_prob,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, image):
        """
        Apply the SimCLR augmentation to the input image.
        """
        return self.transform(image)
