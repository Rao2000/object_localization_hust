import math
import torchvision.ops as ops
import numpy as np
import torch
# from utils import xywh2xyxy


def metric(output, ground_truth):
    idx = output[:, 0] == ground_truth[:, 0]
    o_xyxy = ops.box_convert(output[:, 1:], in_fmt='cxcywh', out_fmt='xyxy').clamp(0, 1)
    g_xyxy = ops.box_convert(ground_truth[:, 1:], in_fmt='cxcywh', out_fmt='xyxy')
    iou = cal_iou(o_xyxy, g_xyxy)
    acc = len(output[(iou > 0.5) & idx]) / output.shape[0]
    cls_acc = len(output[idx]) / output.shape[0]
    wrong_idx = torch.arange(len(output))[iou < 0.5]
    return acc, cls_acc, wrong_idx

    
def cal_iou(boxes1, boxes2):
    inter, union = cal_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou

def cal_inter_union(boxes1, boxes2):
    area1 = ops.box_area(boxes1)
    area2 = ops.box_area(boxes2)

    left_top = torch.max(boxes1[:, :2], boxes2[:, :2])
    right_bottom = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wid_hgt = (right_bottom - left_top).clamp(min=0)
    inter = wid_hgt[:, 0] * wid_hgt[:, 1]
    union = area1 + area2 - inter

    return inter, union


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


if __name__ == '__main__':
    a = np.loadtxt('label/test.txt')
    b = a.copy()
    a = torch.tensor(a)
    b = torch.tensor(b)
    print(cal_iou(a[:, 1:], b))
    
