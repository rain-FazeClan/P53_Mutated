import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random


class MultiChannelTransforms:
    """多通道医学图像的数据增强类，适用于(C, H, W)格式的数据"""

    def __init__(self, mode='simple', **kwargs):
        self.mode = mode

        if mode == 'full':
            # 完整增强模式
            self.transforms = T.Compose([
                RandomFlip(p=kwargs.get('p_flip', 0.5)),
                RandomRotation(degrees=kwargs.get('degrees', 15), p=kwargs.get('p_rotate', 0.5)),
                RandomAffine(
                    degrees=0,
                    translate=kwargs.get('translate', (0.1, 0.1)),
                    scale=kwargs.get('scale', (0.9, 1.1)),
                    p=kwargs.get('p_affine', 0.3)
                ),
                RandomNoise(std=kwargs.get('noise_std', 0.05), p=kwargs.get('p_noise', 0.2)),
                RandomBrightness(factor=kwargs.get('brightness_factor', 0.2), p=kwargs.get('p_brightness', 0.3)),
                RandomContrast(factor=kwargs.get('contrast_factor', 0.2), p=kwargs.get('p_contrast', 0.3))
            ])
        elif mode == 'simple':
            # 简单增强模式
            self.transforms = T.Compose([
                RandomFlip(p=kwargs.get('p_flip', 0.5)),
                RandomRotation(degrees=kwargs.get('degrees', 10), p=kwargs.get('p_rotate', 0.3)),
                RandomAffine(
                    degrees=0,
                    translate=kwargs.get('translate', (0.05, 0.05)),
                    scale=kwargs.get('scale', (0.95, 1.05)),
                    p=kwargs.get('p_affine', 0.2)
                )
            ])
        else:
            # 无增强
            self.transforms = T.Compose([])

    def __call__(self, image):
        """
        Args:
            image: torch.Tensor of shape (C, H, W)
        Returns:
            augmented image: torch.Tensor of shape (C, H, W)
        """
        return self.transforms(image)


class RandomFlip:
    """随机翻转"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            image = TF.hflip(image)
        if random.random() < self.p:
            image = TF.vflip(image)
        return image


class RandomRotation:
    """随机旋转"""
    def __init__(self, degrees=15, p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            angle = random.uniform(-self.degrees, self.degrees)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=[0.0])
        return image


class RandomAffine:
    """随机仿射变换"""
    def __init__(self, degrees=0, translate=None, scale=None, p=0.3):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            # 获取仿射变换参数
            angle = random.uniform(-self.degrees, self.degrees) if self.degrees > 0 else 0

            if self.translate:
                max_dx = float(self.translate[0] * image.shape[-1])
                max_dy = float(self.translate[1] * image.shape[-2])
                tx = int(random.uniform(-max_dx, max_dx))
                ty = int(random.uniform(-max_dy, max_dy))
            else:
                tx, ty = 0, 0

            if self.scale:
                scale = random.uniform(self.scale[0], self.scale[1])
            else:
                scale = 1.0

            image = TF.affine(
                image,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=[0.0]
            )
        return image


class RandomNoise:
    """随机高斯噪声"""
    def __init__(self, std=0.05, p=0.2):
        self.std = std
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            noise = torch.randn_like(image) * self.std
            image = image + noise
        return image


class RandomBrightness:
    """随机亮度调整 - 支持多通道"""
    def __init__(self, factor=0.2, p=0.3):
        self.factor = factor
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            brightness_factor = random.uniform(1-self.factor, 1+self.factor)
            # 直接对张量进行数学运算，避免使用torchvision的adjust_brightness
            image = image * brightness_factor
            image = torch.clamp(image, 0, 1)  # 确保值在合理范围内
        return image


class RandomContrast:
    """随机对比度调整 - 支持多通道"""
    def __init__(self, factor=0.2, p=0.3):
        self.factor = factor
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            contrast_factor = random.uniform(1-self.factor, 1+self.factor)
            # 对每个通道分别计算均值并调整对比度
            mean = torch.mean(image, dim=(-2, -1), keepdim=True)
            image = (image - mean) * contrast_factor + mean
        return image


def transforms(mode='simple', **kwargs):
    """创建数据增强变换函数

    Args:
        mode: 'full', 'simple', 'none'
        **kwargs: 传递给MultiChannelTransforms的参数

    Returns:
        transform function or None
    """
    if mode == 'none':
        return None

    return MultiChannelTransforms(mode=mode, **kwargs)
