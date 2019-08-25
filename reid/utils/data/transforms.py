from __future__ import absolute_import

import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Normalize, ToTensor
from PIL import Image
import random
import math
import numpy as np
import cv2


class RectScale(object):
    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

        # straight using Image
        # return img.resize((self.width, self.height), self.interpolation)  # Image.BILINEAR

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return Image.fromarray(
                cv2.resize(np.array(img), (self.width, self.height), interpolation=self.interpolation))


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        # Tricky!! random.random() can not reproduce the score of np.random.random(),
        # dropping ~1% for both Market1501 and Duke GlobalPool.
        # if random.random() < self.p:
        if np.random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)



class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.width, self.height), self.interpolation)

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(img)
