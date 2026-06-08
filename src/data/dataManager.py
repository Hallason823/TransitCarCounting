import os
import csv
import numpy as np
from torchvision import transforms

from .carCountDataset import CarCountDataset

class DataManager:
    """
    Loads images and car-count labels, applies transforms, and splits into
    train / val / test sets.

    Dataset layout supported:
      Option A — subfolders named by count:
        images/0/img1.jpg, images/1/img2.jpg, images/3/img3.jpg ...
      Option B — CSV annotations file:
        images/annotations.csv  (columns: path, count)
        where 'path' is relative to the images/ folder.
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, folder_path='../images/', split_mode='Hold-out', weights=[0.7, 0.1], k_groups=5, img_size=(224, 224)):
        self.folder_path = folder_path
        self.split_mode = split_mode
        self.weights = weights
        self.k_groups = k_groups
        self.img_size = img_size
        self.k_val = None
        self.transform_train = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        self.transform_eval = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])
        self.loadData()
        self.findAllCategories()
        self.splitSamples()

    def loadData(self):
        annotations_path = os.path.join(self.folder_path, 'annotations.csv')
        if os.path.exists(annotations_path):
            self._loadFromCSV(annotations_path)
        else:
            self._loadFromFolders()

    def _loadFromCSV(self, csv_path):
        self.all_image_paths = []
        self.all_labels = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.all_image_paths.append(os.path.join(self.folder_path, row['path']))
                self.all_labels.append(int(row['count']))
        self.num_classes = max(self.all_labels) + 1

    def _loadFromFolders(self):
        self.all_image_paths = []
        self.all_labels = []
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        for class_folder in sorted(os.listdir(self.folder_path)):
            class_path = os.path.join(self.folder_path, class_folder)
            if not os.path.isdir(class_path):
                continue
            try:
                label = int(class_folder)
            except ValueError:
                continue
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(valid_exts):
                    self.all_image_paths.append(os.path.join(class_path, img_file))
                    self.all_labels.append(label)
        self.num_classes = max(self.all_labels) + 1 if self.all_labels else 0

    def findAllCategories(self):
        self.categories = sorted(list(set(self.all_labels)))
        self.categories_name = [str(c) for c in self.categories]

    def splitSamples(self):
        length = len(self.all_image_paths)
        idx = list(range(length))
        np.random.shuffle(idx)
        if self.split_mode == 'Hold-out':
            lengths = [int(np.floor(sum(self.weights[:i + 1]) * length)) for i in range(len(self.weights))]
            self.train_idx = idx[:lengths[0]]
            self.val_idx   = idx[lengths[0]:lengths[1]]
            self.test_idx  = idx[lengths[1]:]
        elif self.split_mode == 'K-fold cross-validation':
            self.k_val = 0
            test_len       = int((1 - sum(self.weights)) * length)
            train_val_len  = length - test_len
            fold_boundaries = [test_len + int(np.floor(i / self.k_groups * train_val_len)) for i in range(self.k_groups + 1)]
            self.test_idx  = idx[:test_len]
            self.folds_idx = [idx[fold_boundaries[i]:fold_boundaries[i + 1]] for i in range(self.k_groups)]
            self._updateFolds()
        else:
            print('\nInvalid split mode!\n')
        self._buildDatasets()

    def _updateFolds(self):
        self.val_idx   = self.folds_idx[self.k_val]
        self.train_idx = []
        for i in range(self.k_groups):
            if i != self.k_val:
                self.train_idx += self.folds_idx[i]
        self.k_val += 1

    def updateTheDistributionFolds(self):
        self._updateFolds()
        self._buildDatasets()

    def _buildDatasets(self):
        paths  = self.all_image_paths
        labels = self.all_labels
        self.train_ds = CarCountDataset([paths[i] for i in self.train_idx], [labels[i] for i in self.train_idx], self.transform_train)
        self.val_ds   = CarCountDataset([paths[i] for i in self.val_idx], [labels[i] for i in self.val_idx], self.transform_eval)
        self.test_ds  = CarCountDataset([paths[i] for i in self.test_idx], [labels[i] for i in self.test_idx], self.transform_eval)
        self.train_labels = [labels[i] for i in self.train_idx]
        self.val_labels   = [labels[i] for i in self.val_idx]
        self.test_labels  = [labels[i] for i in self.test_idx]