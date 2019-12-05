import random

from torchvision.transforms import functional

from myutils.common import file_util
from myutils.pytorch import tensor_util
import torch


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target['boxes']
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target['boxes'] = bbox
            if 'masks' in target:
                target['masks'] = target['masks'].flip(-1)
            if 'keypoints' in target:
                keypoints = target['keypoints']
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target['keypoints'] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = functional.to_tensor(image)
        return image, target


class DataSizeLogger(object):
    def __init__(self, num_bits=8):
        self.num_bits4quant = num_bits
        self.data_size_list = list()
        self.quantized_data_size_list = list()

    def get_data(self):
        return self.data_size_list.copy(), self.quantized_data_size_list.copy()

    def clear(self):
        self.data_size_list.clear()
        self.quantized_data_size_list.clear()

    def __call__(self, z):
        data_size = file_util.get_binary_object_size(z)
        quantized_data_size = None if not isinstance(z, torch.Tensor)\
            else file_util.get_binary_object_size(tensor_util.quantize_tensor(z, num_bits=self.num_bits4quant))
        self.data_size_list.append(data_size)
        self.quantized_data_size_list.append(quantized_data_size)
        return z
