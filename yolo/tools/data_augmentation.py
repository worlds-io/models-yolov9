from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


class AugmentationComposer:
    """Composes several transforms together."""

    def __init__(self, transforms, image_size: int = [640, 640]):
        self.transforms = transforms
        # TODO: handle List of image_size [640, 640]
        self.pad_resize = PadAndResize(image_size)

        for transform in self.transforms:
            if hasattr(transform, "set_parent"):
                transform.set_parent(self)

    def __call__(self, image, boxes=torch.zeros(0, 5)):
        for transform in self.transforms:
            image, boxes = transform(image, boxes)
        image, boxes, rev_tensor = self.pad_resize(image, boxes)
        image = TF.to_tensor(image)
        return image, boxes, rev_tensor


class RemoveOutliers:
    """Removes outlier bounding boxes that are too small or have invalid dimensions."""

    def __init__(self, min_box_area=1e-8):
        """
        Args:
            min_box_area (float): Minimum area for a box to be kept, as a fraction of the image area.
        """
        self.min_box_area = min_box_area

    def __call__(self, image, boxes):
        """
        Args:
            image (PIL.Image): The cropped image.
            boxes (torch.Tensor): Bounding boxes in normalized coordinates (x_min, y_min, x_max, y_max).
        Returns:
            PIL.Image: The input image (unchanged).
            torch.Tensor: Filtered bounding boxes.
        """
        box_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])

        valid_boxes = (box_areas > self.min_box_area) & (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 4] > boxes[:, 2])

        return image, boxes[valid_boxes]


class PadAndResize:
    def __init__(self, image_size, background_color=(0, 0, 0)):
        """Initialize the object with the target image size."""
        self.target_width, self.target_height = image_size
        self.background_color = background_color

    def set_size(self, image_size: List[int]):
        self.target_width, self.target_height = image_size

    def __call__(self, image: Image, boxes):
        img_width, img_height = image.size
        scale = min(self.target_width / img_width, self.target_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        pad_left = 0  # (self.target_width - new_width) // 2
        pad_top = 0  # (self.target_height - new_height) // 2

        padded_image = Image.new("RGB", (self.target_width, self.target_height), self.background_color)
        padded_image.paste(resized_image, (pad_left, pad_top))

        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * new_width + pad_left) / self.target_width
        boxes[:, [2, 4]] = (boxes[:, [2, 4]] * new_height + pad_top) / self.target_height

        transform_info = torch.tensor([scale, pad_left, pad_top, pad_left, pad_top])
        return padded_image, boxes, transform_info


class HorizontalFlip:
    """Randomly horizontally flips the image along with the bounding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image = TF.hflip(image)
            boxes[:, [1, 3]] = 1 - boxes[:, [3, 1]]
        return image, boxes


class VerticalFlip:
    """Randomly vertically flips the image along with the bounding boxes."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image = TF.vflip(image)
            boxes[:, [2, 4]] = 1 - boxes[:, [4, 2]]
        return image, boxes


class MixUp:
    """Applies the MixUp augmentation to a pair of images and their corresponding boxes."""

    def __init__(self, prob=0.5, alpha=1.0):
        self.alpha = alpha
        self.prob = prob
        self.parent = None

    def set_parent(self, parent):
        """Set the parent dataset object for accessing dataset methods."""
        self.parent = parent

    def __call__(self, image, boxes):
        if torch.rand(1) >= self.prob:
            return image, boxes

        assert self.parent is not None, "Parent is not set. MixUp cannot retrieve additional data."

        # Retrieve another image and its boxes randomly from the dataset
        image2, boxes2 = self.parent.get_more_data()[0]

        # Resize the second image to be the same size as the first
        image2 = image2.resize((image.width, image.height), Image.Resampling.LANCZOS)

        # Calculate the mixup lambda parameter
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 0.5

        # Mix images
        image1, image2 = TF.to_tensor(image), TF.to_tensor(image2)
        mixed_image = lam * image1 + (1 - lam) * image2

        # Merge bounding boxes
        merged_boxes = torch.cat((boxes, boxes2))

        return TF.to_pil_image(mixed_image), merged_boxes


class RandomCrop:
    """Randomly crops the image along with adjusting the bounding boxes."""

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): Probability of applying the crop.
        """
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            original_width, original_height = image.size
            crop_height, crop_width = int(original_height / np.random.uniform(1.5, 4)), int(original_width / np.random.uniform(1.5, 4))
            top = torch.randint(0, original_height - crop_height + 1, (1,)).item()
            left = torch.randint(0, original_width - crop_width + 1, (1,)).item()

            image = TF.crop(image, top, left, crop_height, crop_width)

            boxes[:, [1, 3]] = boxes[:, [1, 3]] * original_width - left
            boxes[:, [2, 4]] = boxes[:, [2, 4]] * original_height - top

            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, crop_width)
            boxes[:, [2, 4]] = boxes[:, [2, 4]].clamp(0, crop_height)

            boxes[:, [1, 3]] /= crop_width
            boxes[:, [2, 4]] /= crop_height

        return image, boxes


class RandomPad:
    """Randomly pads the image along with adjusting the bounding boxes."""

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): Probability of applying the pad.
        """
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            original_width, original_height = image.size
            padded_height, padded_width = int(original_height * np.random.uniform(1.125, 1.375)), int(original_width * np.random.uniform(1.125, 1.375))

            padded_image = Image.new("RGB", (padded_width, padded_height), (0, 0, 0))

            top = torch.randint(0, padded_height - original_height, (1,)).item()
            left = torch.randint(0, padded_width - original_width, (1,)).item()

            padded_image.paste(image, (left, top))

            boxes[:, [1, 3]] += left
            boxes[:, [2, 4]] += top

            image = padded_image

        return image, boxes


class Colorspace:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes, hue=10, saturation=1.5, exposure=1.5):
        if torch.rand(1) < self.prob:
            image = np.array(image)

            dhue        = np.random.uniform(-hue,         hue)
            dsaturation = np.random.uniform(1/saturation, saturation)
            dexposure   = np.random.uniform(1/exposure,   1)

            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float64)
            hsv_image[:, :, 1] *= dsaturation
            hsv_image[:, :, 2] *= dexposure

            hsv_image[:, :, 0] += dhue
            hsv_image[:, :, 0] -= (hsv_image[:, :, 0] > 180) * 180
            hsv_image[:, :, 0] += (hsv_image[:, :, 0] < 0)   * 180

            image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)
            image = np.clip(image, 0, 255)

            image = Image.fromarray(image)

        return image, boxes


class Occlude:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes, per_box_probability=0.5, min_occlusion_width=0.25, max_occlusion_width=0.65, min_occlusion_height=0.25, max_occlusion_height=0.65):
        if torch.rand(1) < self.prob:
            image = np.array(image)

            for i in range(boxes.shape[0]):
                if torch.rand(1) < per_box_probability:
                    x1, y1, x2, y2 = boxes[i, 1:5]

                    width  = x2 - x1
                    height = y2 - y1

                    min_width  = int(min_occlusion_width  * width)
                    max_width  = int(max_occlusion_width  * width)
                    min_height = int(min_occlusion_height * height)
                    max_height = int(max_occlusion_height * height)

                    if (min_width < max_width) and (min_height < max_height):
                        w = np.random.randint(min_width,  max_width)
                        h = np.random.randint(min_height, max_height)

                        max_x = x2 - w
                        max_y = y2 - h

                        if (x1 < max_x) and (y1 < max_y):
                            x = np.random.randint(x1, max_x)
                            y = np.random.randint(y1, max_y)

                            image[y:y+h, x:x+w, ...] = np.random.random(image[y:y+h, x:x+w, ...].shape) * 255

            image = Image.fromarray(image)

        return image, boxes


class Blur:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if torch.rand(1) < self.prob:
            image = np.array(image)

            diameter    = np.random.randint(7, 13)
            sigma_color = np.random.randint(13, 55)
            sigma_space = np.random.randint(13, 55)

            image = cv2.bilateralFilter(image.astype(np.uint8), diameter, sigma_color, sigma_space)
            image = Image.fromarray(image)

        return image, boxes


class JPEG:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes, min_quality=15, max_quality=70):
        if torch.rand(1) < self.prob:
            image = np.array(image)

            quality = np.random.randint(min_quality, max_quality)
            _, encoded_image = cv2.imencode('.jpg', image.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])

            image = cv2.imdecode(encoded_image, cv2.IMREAD_UNCHANGED)
            image = Image.fromarray(image)

        return image, boxes
