import logging
import os
import math
import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager
from torch.utils import tensorboard
import torchvision
from torchvision.utils import draw_bounding_boxes, save_image, make_grid
import torchvision.ops as ops
import cv2

class_mapping = {
    0 : 'bird',
    1 : 'car',
    2 : 'dog',
    3 : 'lizard',
    4 : 'turtle'
}

def get_logger(args):
    logger = logging.getLogger()
    if args.local_rank in [-1, 0]:
        out_path = args.working_dir
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        logging.basicConfig(filename=os.path.join(out_path, 'run_log.log'),
                            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.WARN)
    logger.warning("Process rank: %s, , n_gpu: %s, distributed training: %s",
                   args.local_rank, args.num_gpu, bool(args.local_rank != -1))
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

    
def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    is_parallel = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    return model.module if is_parallel else model


def visualize(imgs_path, outputs, tb_writer, epoch, save_dir='', wrong_idx=None):
    xyxy_outputs = ops.box_convert(outputs[:,1:], in_fmt='cxcywh', out_fmt='xyxy').clamp(0, 1)
    class_labels = list(map(lambda output: class_mapping[output], outputs[:, 0].tolist()))
    imgs = []
    for idx, path in enumerate(imgs_path):
        img = torchvision.io.read_image(path)
        colors = ['blue']
        if wrong_idx is not None:
            if idx in wrong_idx:
                colors = (255, 0, 0)
        loc_img = draw_bounding_boxes(img, (xyxy_outputs[idx] * img.shape[1]).unsqueeze(0), [class_labels[idx]], colors=colors, font_size=8, width=3)
        if len(save_dir):
            loc_img_numpy = loc_img.permute(1, 2, 0).numpy()
            save_img = cv2.cvtColor(loc_img_numpy, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f'{idx}.jpg'), save_img)
        imgs.append(loc_img)
    grid = make_grid(imgs)
    tb_writer.add_image('loc_images', grid, epoch)


def xyxy2xywh(xyxy, imgsz):
    x_center = (xyxy[:, 0] + xyxy[:, 2]) / 2
    y_center = (xyxy[:, 1] + xyxy[:, 3]) / 2
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    xywh = np.column_stack([x_center, y_center, w, h]) / imgsz
    return xywh

def xywh2xyxy(xywh, imgsz):
    x_min = xywh[:, 0] - xywh[:, 2] / 2
    x_max = xywh[:, 0] + xywh[:, 2] / 2
    y_min = xywh[:, 1] - xywh[:, 3] / 2
    y_max = xywh[:, 1] + xywh[:, 3] / 2
    xyxy = np.column_stack([x_min, y_min, x_max, y_max]) * imgsz
    return xyxy
