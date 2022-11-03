import os
import pandas as pd
import torch
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms, ops
from torch.utils.data import Dataset, DataLoader
from utils import torch_distributed_zero_first, xywh2xyxy, xyxy2xywh
import random
import math


def random_perspective(img, targets=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    targets[:, 1:] = xywh2xyxy(targets[:, 1:], width)
    # Transform label coordinates
    n = len(targets)
    if n:
        new = np.zeros((n, 4))

        # warp boxes
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
    targets[:, 1:] = xyxy2xywh(new, width)
    return img, targets[0]

def cutout(img, target, ratio=0.5):
    if random.random() < ratio:
        height, width = img.shape[:2]
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(height * s))  # create random masks
            mask_w = random.randint(1, int(width * s))

            # box
            xmin = max(0, random.randint(0, width) - mask_w // 2)
            ymin = max(0, random.randint(0, height) - mask_h // 2)
            xmax = min(width, xmin + mask_w)
            ymax = min(height, ymin + mask_h)

            # apply random color mask
            img[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # # return unobscured labels
            # if len(targets) and s > 0.03:
            #     box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            #     ioa = bbox_ioa(box, targets[:, 1:5])  # intersection over area
            #     targets = targets[ioa < 0.60]  # remove >60% obscured labels

    return img, target

def bbox_ioa(box1, box2, eps=1E-7):
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, imgsz, transform=None, augment=False):
        self.img_labels = np.loadtxt(annotations_file)
        self.img_dirs = np.loadtxt(img_dir, dtype=str)
        self.transform = transform
        self.imgsz = imgsz
        self.augment = augment

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dirs[idx]
        # image = Image.open(img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.img_labels[idx]
        image = cv2.resize(image, (self.imgsz, self.imgsz))
        if self.augment:
            image, label = random_perspective(image, label[None])
            image, label = cutout(image, label, ratio=0.3)

        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label), img_path


def create_dataloader(annot_dir, img_dir, imgsz, batch_size, augment=False, shuffle=False, rank=-1, world_size=1, workers=8):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    tsfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    with torch_distributed_zero_first(rank):
        dataset = CustomImageDataset(annot_dir, img_dir, imgsz, transform=tsfm, augment=augment)  # augment images)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=(shuffle and rank == -1),
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True)
    return dataloader, dataset


if __name__ == '__main__':
    # print(1)
    # print(os.listdir('tiny_vid/bird'))
    dataset = CustomImageDataset('label/test.txt', 'imgset/test.txt', 128)
    print(next(iter(dataset)))