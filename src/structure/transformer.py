import os
import random

import torch
from PIL import Image
from torchvision.transforms import functional

from myutils.common import file_util
from myutils.pytorch import tensor_util


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


class DataLogger(object):
    def __init__(self, num_bits=8):
        self.num_bits4quant = num_bits
        self.data_size_list = list()
        self.fp16_data_size_list = list()
        self.quantized_data_size_list = list()
        self.tensor_shape_list = list()

    def get_data(self):
        return self.data_size_list.copy(), self.fp16_data_size_list,\
               self.quantized_data_size_list.copy(), self.tensor_shape_list.copy()

    def clear(self):
        self.data_size_list.clear()
        self.fp16_data_size_list.clear()
        self.quantized_data_size_list.clear()
        self.tensor_shape_list.clear()

    def __call__(self, z, target):
        if z is None:
            data_size = 0.0
            fp16_data_size = 0.0
            quantized_data_size = 0.0
        else:
            data_size = file_util.get_binary_object_size(z)
            fp16_data_size = None if not isinstance(z, torch.Tensor) else file_util.get_binary_object_size(z.short())
            quantized_data_size = None if not isinstance(z, torch.Tensor)\
                else file_util.get_binary_object_size(tensor_util.quantize_tensor(z, num_bits=self.num_bits4quant))

        self.data_size_list.append(data_size)
        self.fp16_data_size_list.append(fp16_data_size)
        self.quantized_data_size_list.append(quantized_data_size)
        self.tensor_shape_list.append([0, 0, 0] if z is None else [z.shape[1], z.shape[2], z.shape[3]])
        return z, target


class JpegCompressor(object):
    def __init__(self, jpeg_quality=95, tmp_dir_path='./tmp/'):
        self.jpeg_quality = jpeg_quality
        self.tmp_dir_path = tmp_dir_path
        file_util.make_dirs(tmp_dir_path)

    def save_image(self, z, output_file_path):
        qz = tensor_util.quantize_tensor(z)
        img = Image.fromarray(qz.tensor.permute(1, 2, 0).cpu().numpy())
        img.save(output_file_path, format='jpeg', quality=self.jpeg_quality)
        return qz

    def __call__(self, z, target):
        if (z.dim() == 3 and z.shape[0] == 3) or (z.dim() == 4 and z.shape[0] == 1 and z.shape[1] == 3):
            if z.dim() == 4:
                z = z.squeeze(0)

            file_path = os.path.join(self.tmp_dir_path, '{}.jpg'.format(hash(z)))
            qz = self.save_image(z, file_path)
            return (file_path, qz), target
        return z, target


class JpegDecompressor(object):
    def __init__(self, tmp_dir_path='./tmp/', target_dim=4):
        self.tmp_dir_path = tmp_dir_path
        self.target_dim = target_dim

    def __call__(self, z, target):
        if isinstance(z, tuple) and isinstance(z[0], str):
            img = Image.open(z[0]).convert('RGB')
            qz = z[1]
            img = qz.scale * (functional.to_tensor(img) * 255.0 - qz.zero_point)
            return img if self.target_dim != 4 else img.unsqueeze(0), target
        return z, target


class Quantizer(object):
    def __init__(self, num_bits=8):
        self.num_bits = num_bits

    def __call__(self, z, target):
        if self.num_bits == 16:
            return z.half(), target

        qz = tensor_util.quantize_tensor(z, num_bits=self.num_bits)
        return qz, target


class Dequantizer(object):
    def __init__(self, num_bits=8):
        # num_bits should be the same as Quantizer
        self.num_bits = num_bits

    def __call__(self, qz, target):
        if self.num_bits == 16:
            return qz.float(), target

        z = tensor_util.dequantize_tensor(qz)
        return z, target


TRANSFORMER_CLASS_DICT = {
    'jpeg_compressor': JpegCompressor,
    'jpeg_decompressor': JpegDecompressor,
    'quantizer': Quantizer,
    'dequantizer': Dequantizer
}


def get_bottleneck_transformer(transformer_config):
    component_list = list()
    components_config = transformer_config['components']
    for component_name in transformer_config['order']:
        param_config = components_config[component_name]['params']
        if component_name not in TRANSFORMER_CLASS_DICT:
            raise KeyError('transformer `{}` is not expected'.format(component_name))

        obj_class = TRANSFORMER_CLASS_DICT[component_name]
        component_list.append(obj_class(**param_config))
    return Compose(component_list) if len(component_list) > 0 else None
