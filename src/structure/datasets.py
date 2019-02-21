import csv
import os
import sys

import cv2
import numpy as np
import skimage
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from utils import yolo_util


class CocoDataset(Dataset):
    def __init__(self, annotation_file_path, img_root_dir_path, transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotation_file_path = annotation_file_path
        self.img_root_dir_path = img_root_dir_path
        self.transform = transform
        self.coco = COCO(annotation_file_path)
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        self.coco_idx2id_dict = dict()
        self.coco_id2idx_dict = dict()
        self.class_idx2name_dict = dict()
        self.class_name2idx_dict = dict()
        for category in categories:
            coco_idx = len(self.class_name2idx_dict)
            coco_id = category['id']
            self.coco_idx2id_dict[coco_idx] = coco_id
            self.coco_id2idx_dict[coco_id] = coco_idx
            class_idx = len(self.class_name2idx_dict)
            class_name = category['name']
            self.class_idx2name_dict[class_idx] = class_name
            self.class_name2idx_dict[class_name] = class_idx

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.img_root_dir_path, image_info['file_name'])
        img = skimage.io.imread(path)
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))
        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, annotation_dict in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if annotation_dict['bbox'][2] < 1 or annotation_dict['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = annotation_dict['bbox']
            annotation[0, 4] = self.coco_label_to_label(annotation_dict['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_id2idx_dict[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_idx2id_dict[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return len(self.class_idx2name_dict)


class CSVDataset(Dataset):
    def __init__(self, data_file_path, class_list, transform=None):
        self.data_file_path = data_file_path
        self.class_list = class_list
        self.transform = transform
        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.data_file_path) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            ValueError('invalid CSV annotations file: {}: {}'.format(self.data_file_path, e))
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, func, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `func(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return func(value)
        except ValueError as e:
            ValueError(fmt.format(e))

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1
            try:
                class_name, class_id = row
            except ValueError:
                ValueError('line {}: format should be \'class_name,class_id\''.format(line))

            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))
            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2
            annotation[0, 4] = self.name_to_label(a['class'])
            annotations = np.append(annotations, annotation, axis=0)
        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1
            img_file_path, x1, y1, x2, y2, class_name = row[:6]
            if img_file_path not in result:
                result[img_file_path] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file_path].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


class CocoDataset4Yolo(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, annotation_file_path, img_root_dir_path, img_size=416,
                 augment_config=None, min_size=1, debug=False):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
            debug (bool): if True, only one data id is selected from the dataset
        """
        self.img_root_dir_path = img_root_dir_path
        self.coco = COCO(annotation_file_path)
        self.ids = self.coco.getImgIds()
        if debug:
            self.ids = self.ids[1:2]
            print("debug mode...", self.ids)

        self.class_ids = sorted(self.coco.getCatIds())
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        self.lrflip = augment_config['lrflip']
        self.jitter = augment_config['jitter']
        self.random_placing = augment_config['random_placing']
        self.hue = augment_config['hue']
        self.saturation = augment_config['saturation']
        self.exposure = augment_config['exposure']
        self.random_distort = augment_config['random_distort']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        id_ = self.ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        lrflip = False
        if np.random.rand() > 0.5 and self.lrflip:
            lrflip = True

        # load image and preprocess
        img_file = os.path.join(self.img_root_dir_path, '{:012}'.format(id_) + '.jpg')
        img = cv2.imread(img_file)
        assert img is not None

        img, info_img = yolo_util.preprocess(img, self.img_size, jitter=self.jitter, random_placing=self.random_placing)

        if self.random_distort:
            img = yolo_util.random_distort(img, self.hue, self.saturation, self.exposure)

        img = np.transpose(img / 255., (2, 0, 1))
        if lrflip:
            img = np.flip(img, axis=2).copy()

        # load labels
        labels = []
        for anno in annotations:
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
                labels.append([])
                labels[-1].append(self.class_ids.index(anno['category_id']))
                labels[-1].extend(anno['bbox'])

        padded_labels = np.zeros((self.max_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels)
            labels = yolo_util.label2yolobox(labels, info_img, self.img_size, lrflip)
            padded_labels[range(len(labels))[:self.max_labels]] = labels[:self.max_labels]
        padded_labels = torch.from_numpy(padded_labels.astype(np.float32))
        return img, padded_labels, info_img, id_
