from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class AugmentationConfig:
    """Central configuration for all augmentation pipelines."""

    image_size: int = 224

    # Replace these with your dataset mean/std if you compute them.
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Physical/object-level augmentation.
    hflip_prob: float = 0.5
    rotation_degrees: float = 10.0
    translate: Tuple[float, float] = (0.08, 0.08)
    scale: Tuple[float, float] = (0.85, 1.15)
    shear_degrees: float = 5.0
    perspective_prob: float = 0.15
    perspective_distortion: float = 0.15

    # Crop strength.
    standard_crop_scale: Tuple[float, float] = (0.75, 1.0)
    strong_crop_scale: Tuple[float, float] = (0.60, 1.0)

    # Color and lighting augmentation.
    color_jitter_prob: float = 0.8
    brightness: float = 0.25
    contrast: float = 0.25
    saturation: float = 0.20
    hue: float = 0.04
    grayscale_prob: float = 0.05

    # Blur and noise.
    gaussian_blur_prob: float = 0.15
    gaussian_blur_kernel_size: int = 3
    gaussian_blur_sigma: Tuple[float, float] = (0.1, 1.2)
    noise_prob: float = 0.20
    noise_std: float = 0.03

    # Background/context robustness through mild cutout.
    random_erasing_prob: float = 0.25
    erasing_scale: Tuple[float, float] = (0.02, 0.12)
    erasing_ratio: Tuple[float, float] = (0.3, 3.3)

    # Stronger robustness settings.
    strong_color_jitter_prob: float = 0.9
    strong_brightness: float = 0.35
    strong_contrast: float = 0.35
    strong_saturation: float = 0.30
    strong_hue: float = 0.06
    strong_grayscale_prob: float = 0.08
    strong_gaussian_blur_prob: float = 0.25
    strong_noise_prob: float = 0.30
    strong_noise_std: float = 0.045
    strong_random_erasing_prob: float = 0.35
    strong_erasing_scale: Tuple[float, float] = (0.02, 0.16)

    # Kept intentionally rare because these can alter fine texture cues.
    posterize_prob: float = 0.05
    solarize_prob: float = 0.05

    # Approximate foreground/background augmentation. These are intentionally
    # moderate because the mask is heuristic, not a semantic segmentation model.
    background_delete_prob: float = 0.06
    background_replace_prob: float = 0.10
    strong_background_delete_prob: float = 0.10
    strong_background_replace_prob: float = 0.22
    stress_background_replace_prob: float = 0.18


class Compose:
    def __init__(self, transforms: Iterable[Callable]) -> None:
        self.transforms = list(transforms)

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image

    def __repr__(self) -> str:
        lines = [f"    {transform}" for transform in self.transforms]
        return "Compose(\n" + "\n".join(lines) + "\n)"


class RandomApply:
    def __init__(self, transform: Callable, probability: float) -> None:
        self.transform = transform
        self.probability = probability

    def __call__(self, image):
        if random.random() < self.probability:
            return self.transform(image)
        return image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability}, transform={self.transform})"


class Resize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= height:
            new_width = self.size
            new_height = round(height * self.size / width)
        else:
            new_height = self.size
            new_width = round(width * self.size / height)
        return image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class CenterCrop:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        left = max(0, (width - self.size) // 2)
        top = max(0, (height - self.size) // 2)
        return image.crop((left, top, left + self.size, top + self.size))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class RandomResizedCrop:
    def __init__(
        self,
        size: int,
        scale: Tuple[float, float],
        ratio: Tuple[float, float],
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        area = width * height

        for _ in range(10):
            target_area = area * random.uniform(*self.scale)
            aspect_ratio = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            crop_width = round(math.sqrt(target_area * aspect_ratio))
            crop_height = round(math.sqrt(target_area / aspect_ratio))

            if 0 < crop_width <= width and 0 < crop_height <= height:
                left = random.randint(0, width - crop_width)
                top = random.randint(0, height - crop_height)
                image = image.crop((left, top, left + crop_width, top + crop_height))
                return image.resize((self.size, self.size), Image.Resampling.BILINEAR)

        return CenterCrop(min(width, height))(image).resize((self.size, self.size), Image.Resampling.BILINEAR)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, scale={self.scale}, ratio={self.ratio})"


class RandomHorizontalFlip:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.probability:
            return ImageOps.mirror(image)
        return image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability})"


class RandomAffine:
    def __init__(
        self,
        degrees: float,
        translate: Tuple[float, float],
        scale: Tuple[float, float],
        shear: float,
    ) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        angle = random.uniform(-self.degrees, self.degrees)
        max_dx = self.translate[0] * width
        max_dy = self.translate[1] * height
        tx = random.uniform(-max_dx, max_dx)
        ty = random.uniform(-max_dy, max_dy)
        scale = random.uniform(*self.scale)
        shear = math.tan(math.radians(random.uniform(-self.shear, self.shear)))

        cx = width * 0.5
        cy = height * 0.5
        angle = math.radians(angle)
        cos_angle = math.cos(angle) / scale
        sin_angle = math.sin(angle) / scale

        a = cos_angle + shear * sin_angle
        b = sin_angle
        d = shear * cos_angle - sin_angle
        e = cos_angle
        c = cx - a * cx - b * cy - tx
        f = cy - d * cx - e * cy - ty

        return image.transform(
            image.size,
            Image.Transform.AFFINE,
            (a, b, c, d, e, f),
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(degrees={self.degrees}, translate={self.translate}, "
            f"scale={self.scale}, shear={self.shear})"
        )


class RandomPerspective:
    def __init__(self, probability: float, distortion_scale: float) -> None:
        self.probability = probability
        self.distortion_scale = distortion_scale

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image

        width, height = image.size
        max_dx = self.distortion_scale * width
        max_dy = self.distortion_scale * height
        coeffs = _find_perspective_coeffs(
            [
                (random.uniform(0, max_dx), random.uniform(0, max_dy)),
                (width - random.uniform(0, max_dx), random.uniform(0, max_dy)),
                (width - random.uniform(0, max_dx), height - random.uniform(0, max_dy)),
                (random.uniform(0, max_dx), height - random.uniform(0, max_dy)),
            ],
            [(0, 0), (width, 0), (width, height), (0, height)],
        )
        return image.transform(
            image.size,
            Image.Transform.PERSPECTIVE,
            coeffs,
            resample=Image.Resampling.BILINEAR,
            fillcolor=0,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability}, distortion_scale={self.distortion_scale})"


class ColorJitter:
    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: Image.Image) -> Image.Image:
        operations = [
            lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(1 - self.brightness, 1 + self.brightness)),
            lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(1 - self.contrast, 1 + self.contrast)),
            lambda img: ImageEnhance.Color(img).enhance(random.uniform(1 - self.saturation, 1 + self.saturation)),
            self._adjust_hue,
        ]
        random.shuffle(operations)
        for operation in operations:
            image = operation(image)
        return image

    def _adjust_hue(self, image: Image.Image) -> Image.Image:
        if self.hue <= 0:
            return image
        hsv = image.convert("HSV")
        h, s, v = hsv.split()
        shift = int(random.uniform(-self.hue, self.hue) * 255)
        h = h.point(lambda pixel: (pixel + shift) % 256)
        return Image.merge("HSV", (h, s, v)).convert("RGB")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(brightness={self.brightness}, contrast={self.contrast}, "
            f"saturation={self.saturation}, hue={self.hue})"
        )


class RandomGrayscale:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.probability:
            return ImageOps.grayscale(image).convert("RGB")
        return image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability})"


class GaussianBlur:
    def __init__(self, sigma: Tuple[float, float]) -> None:
        self.sigma = sigma

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.sigma)))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma})"


class RandomPosterize:
    def __init__(self, bits: int = 5) -> None:
        self.bits = bits

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.posterize(image, self.bits)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bits={self.bits})"


class RandomSolarize:
    def __init__(self, threshold: int = 192) -> None:
        self.threshold = threshold

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.solarize(image, self.threshold)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"


class RandomBackgroundChange:
    """Approximate online background deletion/replacement for centered objects."""

    def __init__(self, delete_probability: float, replace_probability: float) -> None:
        self.delete_probability = delete_probability
        self.replace_probability = replace_probability

    def __call__(self, image: Image.Image) -> Image.Image:
        action = random.random()
        if action < self.delete_probability:
            return delete_background(image)
        if action < self.delete_probability + self.replace_probability:
            return replace_background(image)
        return image

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(delete_p={self.delete_probability}, "
            f"replace_p={self.replace_probability})"
        )


class ToTensor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = data.view(image.height, image.width, 3).permute(2, 0, 1).to(dtype=torch.float32)
        return tensor.div(255.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AddGaussianNoise:
    """Add mild Gaussian noise to a tensor image in [0, 1]."""

    def __init__(self, std: float = 0.03) -> None:
        self.std = std

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(image) * self.std
        return torch.clamp(image + noise, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"


class RandomErasing:
    def __init__(
        self,
        probability: float,
        scale: Tuple[float, float],
        ratio: Tuple[float, float],
    ) -> None:
        self.probability = probability
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return image

        _, height, width = image.shape
        area = height * width
        for _ in range(10):
            target_area = area * random.uniform(*self.scale)
            aspect_ratio = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            erase_height = round(math.sqrt(target_area / aspect_ratio))
            erase_width = round(math.sqrt(target_area * aspect_ratio))

            if 0 < erase_width < width and 0 < erase_height < height:
                top = random.randint(0, height - erase_height)
                left = random.randint(0, width - erase_width)
                image = image.clone()
                image[:, top : top + erase_height, left : left + erase_width] = torch.rand(
                    (image.shape[0], erase_height, erase_width),
                    dtype=image.dtype,
                    device=image.device,
                )
                return image

        return image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability}, scale={self.scale}, ratio={self.ratio})"


class Normalize:
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean.to(image.device)) / self.std.to(image.device)

    def __repr__(self) -> str:
        mean = tuple(float(value) for value in self.mean.flatten())
        std = tuple(float(value) for value in self.std.flatten())
        return f"{self.__class__.__name__}(mean={mean}, std={std})"


class BaseTransformBuilder:
    """Shared helpers for constructing transform pipelines."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def _to_tensor_and_normalize(self) -> list[Callable]:
        return [
            ToTensor(),
            Normalize(mean=self.config.mean, std=self.config.std),
        ]

    def _resize_for_eval(self) -> list[Callable]:
        size = self.config.image_size
        return [
            Resize(int(size * 1.14)),
            CenterCrop(size),
        ]

    def _color_jitter(
        self,
        probability: float,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
    ) -> RandomApply:
        return RandomApply(
            ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            ),
            probability=probability,
        )

    def _affine(self) -> RandomAffine:
        return RandomAffine(
            degrees=self.config.rotation_degrees,
            translate=self.config.translate,
            scale=self.config.scale,
            shear=self.config.shear_degrees,
        )

    def _perspective(self) -> RandomPerspective:
        return RandomPerspective(
            distortion_scale=self.config.perspective_distortion,
            probability=self.config.perspective_prob,
        )

    def _random_erasing(
        self,
        probability: float,
        scale: Tuple[float, float],
    ) -> RandomErasing:
        return RandomErasing(
            probability=probability,
            scale=scale,
            ratio=self.config.erasing_ratio,
        )


class ValidationTransformBuilder(BaseTransformBuilder):
    """Build clean validation/test transforms."""

    def build(self) -> Compose:
        return Compose(
            [
                *self._resize_for_eval(),
                *self._to_tensor_and_normalize(),
            ]
        )


class StandardTrainingTransformBuilder(BaseTransformBuilder):
    """Build moderate training-time augmentations."""

    def build(self) -> Compose:
        cfg = self.config

        return Compose(
            [
                RandomResizedCrop(
                    cfg.image_size,
                    scale=cfg.standard_crop_scale,
                    ratio=(0.85, 1.15),
                ),
                RandomHorizontalFlip(probability=cfg.hflip_prob),
                self._affine(),
                self._perspective(),
                RandomBackgroundChange(
                    delete_probability=cfg.background_delete_prob,
                    replace_probability=cfg.background_replace_prob,
                ),
                self._color_jitter(
                    probability=cfg.color_jitter_prob,
                    brightness=cfg.brightness,
                    contrast=cfg.contrast,
                    saturation=cfg.saturation,
                    hue=cfg.hue,
                ),
                RandomGrayscale(probability=cfg.grayscale_prob),
                RandomApply(
                    GaussianBlur(sigma=cfg.gaussian_blur_sigma),
                    probability=cfg.gaussian_blur_prob,
                ),
                RandomApply(RandomPosterize(bits=5), probability=cfg.posterize_prob),
                ToTensor(),
                RandomApply(AddGaussianNoise(std=cfg.noise_std), probability=cfg.noise_prob),
                self._random_erasing(
                    probability=cfg.random_erasing_prob,
                    scale=cfg.erasing_scale,
                ),
                Normalize(mean=cfg.mean, std=cfg.std),
            ]
        )


class StrongRobustnessTrainingTransformBuilder(BaseTransformBuilder):
    """Build stronger but still label-preserving robustness augmentations."""

    def build(self) -> Compose:
        cfg = self.config

        return Compose(
            [
                RandomResizedCrop(
                    cfg.image_size,
                    scale=cfg.strong_crop_scale,
                    ratio=(0.80, 1.25),
                ),
                RandomHorizontalFlip(probability=cfg.hflip_prob),
                RandomApply(self._affine(), probability=0.85),
                RandomPerspective(
                    distortion_scale=min(cfg.perspective_distortion * 1.4, 0.30),
                    probability=min(cfg.perspective_prob * 1.5, 0.35),
                ),
                RandomBackgroundChange(
                    delete_probability=cfg.strong_background_delete_prob,
                    replace_probability=cfg.strong_background_replace_prob,
                ),
                self._color_jitter(
                    probability=cfg.strong_color_jitter_prob,
                    brightness=cfg.strong_brightness,
                    contrast=cfg.strong_contrast,
                    saturation=cfg.strong_saturation,
                    hue=cfg.strong_hue,
                ),
                RandomGrayscale(probability=cfg.strong_grayscale_prob),
                RandomApply(
                    GaussianBlur(sigma=cfg.gaussian_blur_sigma),
                    probability=cfg.strong_gaussian_blur_prob,
                ),
                RandomApply(RandomPosterize(bits=5), probability=cfg.posterize_prob),
                RandomApply(RandomSolarize(threshold=192), probability=cfg.solarize_prob),
                ToTensor(),
                RandomApply(AddGaussianNoise(std=cfg.strong_noise_std), probability=cfg.strong_noise_prob),
                self._random_erasing(
                    probability=cfg.strong_random_erasing_prob,
                    scale=cfg.strong_erasing_scale,
                ),
                Normalize(mean=cfg.mean, std=cfg.std),
            ]
        )


class StressTestTransformBuilder(BaseTransformBuilder):
    """Build augmented validation transforms for robustness evaluation."""

    def build(self) -> Compose:
        cfg = self.config

        return Compose(
            [
                Resize(int(cfg.image_size * 1.14)),
                RandomResizedCrop(
                    cfg.image_size,
                    scale=(0.80, 1.0),
                    ratio=(0.90, 1.10),
                ),
                RandomHorizontalFlip(probability=cfg.hflip_prob * 0.5),
                RandomApply(self._affine(), probability=0.5),
                RandomPerspective(
                    distortion_scale=cfg.perspective_distortion,
                    probability=cfg.perspective_prob,
                ),
                RandomBackgroundChange(
                    delete_probability=0.0,
                    replace_probability=cfg.stress_background_replace_prob,
                ),
                self._color_jitter(
                    probability=0.7,
                    brightness=cfg.brightness,
                    contrast=cfg.contrast,
                    saturation=cfg.saturation,
                    hue=cfg.hue,
                ),
                RandomGrayscale(probability=cfg.grayscale_prob),
                RandomApply(
                    GaussianBlur(sigma=cfg.gaussian_blur_sigma),
                    probability=cfg.gaussian_blur_prob,
                ),
                ToTensor(),
                RandomApply(AddGaussianNoise(std=cfg.noise_std), probability=cfg.noise_prob),
                self._random_erasing(
                    probability=0.15,
                    scale=(0.01, 0.08),
                ),
                Normalize(mean=cfg.mean, std=cfg.std),
            ]
        )


class AugmentationFactory:
    """Main entry point for creating all transform pipelines."""

    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()

    def validation(self) -> Compose:
        return ValidationTransformBuilder(self.config).build()

    def standard_training(self) -> Compose:
        return StandardTrainingTransformBuilder(self.config).build()

    def strong_training(self) -> Compose:
        return StrongRobustnessTrainingTransformBuilder(self.config).build()

    def stress_test(self) -> Compose:
        return StressTestTransformBuilder(self.config).build()


def delete_background(image: Image.Image) -> Image.Image:
    mask = foreground_mask(image)
    background = Image.new("RGB", image.size, (242, 242, 238))
    return Image.composite(image, background, mask)


def replace_background(image: Image.Image) -> Image.Image:
    mask = foreground_mask(image)
    background_factory = random.choice(
        [
            make_sky_background,
            make_grass_background,
            make_water_background,
            make_studio_background,
        ]
    )
    background = background_factory(image.size)
    return Image.composite(image, background, mask)


def foreground_mask(image: Image.Image) -> Image.Image:
    """
    Dependency-free approximate mask for centered object photos.

    This is intentionally heuristic. A true segmentation model would produce
    cleaner masks, but would also add an external dependency to training.
    """

    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    corner_colors = sample_corner_colors(rgb)

    mask = Image.new("L", rgb.size, 0)
    mask_pixels = mask.load()
    center_x = (width - 1) / 2
    center_y = (height - 1) / 2
    max_radius = (center_x * center_x + center_y * center_y) ** 0.5
    edge_scale = max(1, min(width, height) * 0.08)

    for y in range(height):
        for x in range(width):
            color = pixels[x, y]
            distance_from_background = min(color_distance(color, background) for background in corner_colors)
            center_distance = (((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5) / max_radius
            center_bonus = max(0.0, 1.0 - center_distance) * 70.0
            edge_distance = min(x, y, width - 1 - x, height - 1 - y)
            edge_bonus = min(edge_distance / edge_scale, 1.0) * 25.0
            score = distance_from_background + center_bonus + edge_bonus
            mask_pixels[x, y] = 255 if score > 78 else 0

    mask = mask.filter(ImageFilter.MedianFilter(size=5))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2.2))
    return ImageEnhance.Contrast(mask).enhance(1.8)


def sample_corner_colors(image: Image.Image) -> list[tuple[int, int, int]]:
    width, height = image.size
    patch = max(4, min(width, height) // 12)
    boxes = [
        (0, 0, patch, patch),
        (width - patch, 0, width, patch),
        (0, height - patch, patch, height),
        (width - patch, height - patch, width, height),
    ]
    return [average_color(image.crop(box)) for box in boxes]


def average_color(image: Image.Image) -> tuple[int, int, int]:
    pixels = list(image.getdata())
    count = max(1, len(pixels))
    return tuple(sum(pixel[channel] for pixel in pixels) // count for channel in range(3))


def color_distance(first: tuple[int, int, int], second: tuple[int, int, int]) -> float:
    return sum((a - b) ** 2 for a, b in zip(first, second)) ** 0.5


def make_sky_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    top = (95, 174, 236)
    bottom = (225, 242, 255)

    for y in range(height):
        blend = y / max(1, height - 1)
        color = tuple(round(top[channel] * (1 - blend) + bottom[channel] * blend) for channel in range(3))
        draw.line((0, y, width, y), fill=color)

    for _ in range(random.randint(4, 8)):
        cloud_x = random.randint(-width // 5, width)
        cloud_y = random.randint(0, max(1, height * 2 // 3))
        cloud_width = random.randint(max(1, width // 7), max(2, width // 3))
        cloud_height = random.randint(max(1, height // 14), max(2, height // 7))
        for _ in range(random.randint(3, 6)):
            offset_x = random.randint(0, cloud_width)
            offset_y = random.randint(-cloud_height // 3, cloud_height // 3)
            box = (
                cloud_x + offset_x,
                cloud_y + offset_y,
                cloud_x + offset_x + random.randint(max(1, cloud_width // 3), cloud_width),
                cloud_y + offset_y + random.randint(max(1, cloud_height // 2), cloud_height),
            )
            draw.ellipse(box, fill=(248, 250, 252))

    return image.filter(ImageFilter.GaussianBlur(radius=0.4))


def make_grass_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, (72, 139, 74))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        blend = y / max(1, height - 1)
        color = (
            round(92 * (1 - blend) + 48 * blend),
            round(160 * (1 - blend) + 112 * blend),
            round(86 * (1 - blend) + 50 * blend),
        )
        draw.line((0, y, width, y), fill=color)
    for _ in range(width * 2):
        x = random.randint(0, width - 1)
        y = random.randint(height // 3, height - 1)
        draw.line((x, y, x + random.randint(-2, 2), y - random.randint(3, 12)), fill=(45, 105, 48))
    return image.filter(ImageFilter.GaussianBlur(radius=0.8))


def make_water_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, (58, 141, 180))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        blend = y / max(1, height - 1)
        color = (
            round(73 * (1 - blend) + 24 * blend),
            round(164 * (1 - blend) + 95 * blend),
            round(205 * (1 - blend) + 145 * blend),
        )
        draw.line((0, y, width, y), fill=color)
    for _ in range(max(1, height // 5)):
        y = random.randint(0, height - 1)
        x = random.randint(0, width - 1)
        draw.arc((x - 30, y - 4, x + 30, y + 8), 0, 180, fill=(165, 218, 235), width=1)
    return image.filter(ImageFilter.GaussianBlur(radius=0.7))


def make_studio_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    base = random.choice([(232, 229, 222), (224, 231, 232), (229, 224, 235), (235, 233, 224)])
    image = Image.new("RGB", size, base)
    draw = ImageDraw.Draw(image)
    for y in range(height):
        shade = round((y / max(1, height - 1)) * 28)
        color = tuple(max(0, channel - shade) for channel in base)
        draw.line((0, y, width, y), fill=color)
    return image


def _find_perspective_coeffs(startpoints, endpoints) -> list[float]:
    matrix = []
    for point, endpoint in zip(startpoints, endpoints):
        x, y = endpoint
        u, v = point
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])

    a = torch.tensor(matrix, dtype=torch.float64)
    b = torch.tensor(startpoints, dtype=torch.float64).reshape(8)
    return torch.linalg.solve(a, b).tolist()


if __name__ == "__main__":
    config = AugmentationConfig(image_size=224)
    factory = AugmentationFactory(config)

    train_transform = factory.standard_training()
    strong_train_transform = factory.strong_training()
    val_transform = factory.validation()
    stress_test_transform = factory.stress_test()

    print("Standard training transform:")
    print(train_transform)
    print("\nStrong robustness training transform:")
    print(strong_train_transform)
    print("\nClean validation transform:")
    print(val_transform)
    print("\nStress-test validation transform:")
    print(stress_test_transform)
