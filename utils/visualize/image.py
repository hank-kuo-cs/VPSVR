import torch
from PIL import Image
from torchvision.transforms import transforms


def concat_images(images: list) -> Image:
    to_pil_image = transforms.ToPILImage()
    images = [to_pil_image(img.detach().cpu()) for img in images]
    w, h = images[0].width, images[0].height

    result = Image.new('RGB', (w * len(images), h))

    for i, img in enumerate(images):
        result.paste(img, (i * w, 0))

    return result


def denormlize_image(image: torch.Tensor):
    assert image.ndimension() == 3  # (3, H, W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)

    return image * std + mean
