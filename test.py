import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import visualize
from custom_dataset import create_dataloader
from model import Model
from metric import metric


@torch.no_grad()
def test(dataloader,
         model,
         tb_writer,
         epoch=0,
         conf_thres=0.001, # 0.001
         iou_thres=0.6,  # for NMS, 0.6
         args=None):
    model.eval()
    device = next(model.parameters()).device
    mean_acc = 0.0
    mean_cls_acc = 0.0
    is_train = args is None
    for iter, (imgs, labels, imgs_path) in tqdm(enumerate(dataloader)):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device)
        preds = model(imgs)
        class_ids = torch.argmax(preds[:, 4:], dim=-1, keepdim=True)
        outputs = torch.cat([class_ids, preds[:, 0:4]], dim=-1)
        acc, cls_acc, wrong_idx = metric(outputs, labels)
        mean_acc = (mean_acc * iter + acc) / (iter + 1)
        mean_cls_acc = (mean_cls_acc * iter + cls_acc) / (iter + 1)
        
        save_dir = os.path.join(args.working_dir, f'results_{iter}') if args is not None else ''
        if len(save_dir):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        if is_train:
            if iter == 0:
                visualize(imgs_path, outputs.detach().cpu(), tb_writer, epoch, save_dir, wrong_idx=wrong_idx)
        else:
            visualize(imgs_path, outputs.detach().cpu(), tb_writer, iter, save_dir, wrong_idx=wrong_idx)
    
    return round(mean_acc, 6), round(mean_cls_acc, 6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='obj_det/best.pth', type=str, help='path for pretrained model')
    parser.add_argument('--val_path', default='imgset/test.txt', type=str, help='path for val set')
    parser.add_argument('--val_label_path', default='label/test.txt', type=str, help='path for val labels')
    parser.add_argument('--working_dir', default='log_test', type=str, help='dir name for logs and outputs')
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--num_classes', default=5, type=int)
    parser.add_argument('--batch_size', default=30, type=int)
    parser.add_argument('--img_size', default=128, type=int)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if args.use_gpu else 'cpu')
    tb_writer = SummaryWriter(args.working_dir)  # Tensorboard

    ckpt = torch.load(args.weights, map_location=device)
    model = Model(args).to(device)
    model.load_state_dict(ckpt)
    loader, _ = create_dataloader(args.val_label_path, args.val_path, args.img_size, args.batch_size, augment=False,
                                rank=-1, world_size=1, workers=args.workers)

    acc, cls_acc = test(loader, model, tb_writer, args=args)
    print(f'Test: acc: {acc}, cls_acc: {cls_acc}')
    tb_writer.close()