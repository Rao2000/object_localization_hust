import os
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import get_logger, one_cycle, intersect_dicts, de_parallel
from custom_dataset import create_dataloader
from model import Model
from loss import LossFunctionClass
from test import test


def train(args, device, logger, tb_writer=None):
    rank = args.global_rank
    
    pretrained = args.weights.endswith('.pth')
    if pretrained:
        ckpt = torch.load(args.weights, map_location=device)
        model = Model(args).to(device)
        state_dict = intersect_dicts(ckpt, model.backbone.state_dict())
        model.backbone.load_state_dict(state_dict, strict=False)
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), args.weights))
    else:
        model = Model(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
    lrf = one_cycle(1, args.gamma, args.epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lrf)

    if args.use_gpu and rank == -1 and args.num_gpu > 1:
        model = torch.nn.DataParallel(model)

    trainloader, _ = create_dataloader(args.train_label_path, args.train_path, args.img_size, args.batch_size, augment=True,
                                        shuffle=True, rank=rank, world_size=args.world_size, workers=args.workers)
    if rank in [-1, 0]:
        testloader, _ = create_dataloader(args.val_label_path, args.val_path, args.img_size, args.batch_size * 2, augment=False,
                                            rank=-1, world_size=args.world_size, workers=args.workers)

    if args.use_gpu and rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    criterion = LossFunctionClass(model)
    best_metric = 0.0
    logger.info(f'Image sizes {args.img_size} train\n'
                f'Using {trainloader.num_workers} dataloader workers\n'
                f'Logging results to {args.working_dir}\n'
                f'Starting training for {args.epochs} epochs...')
    for epo in range(args.epochs):
        model.train()
        if rank != -1:
            trainloader.sampler.set_epoch(epo)
        pbar = enumerate(trainloader)
        if rank in [-1, 0]:
            pbar = tqdm(pbar)
        optimizer.zero_grad()
        mean_loss = torch.zeros(3)
        for iter, (imgs, labels, _) in pbar:
            imgs = imgs.to(device, non_blocking=True)
            preds = model(imgs)
            loss, loss_items = criterion(preds, labels.to(device))

            if rank != -1:
                loss *= args.world_size
            if args.accumulate > 1:
                loss /= float(args.accumulate)
            
            loss.backward()
            if (iter + 1) % args.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if rank in [-1, 0]:
                mean_loss = (mean_loss * iter + loss_items) / (iter + 1)
        scheduler.step()
        if rank in [-1, 0]:
            ckpt = de_parallel(model).state_dict()
            torch.save(ckpt, os.path.join(args.working_dir, 'last.pth'))
            acc, cls_acc = test(testloader, model, tb_writer, epo)
            if acc > best_metric:
                best_metric = acc
                torch.save(ckpt, os.path.join(args.working_dir, 'best.pth'))
            tags = ['train/obj_loss', 'train/box_loss', 'train/cls_loss',  # train loss
                    'acc', 'classify_acc']  # params
            for x, tag in zip(list(mean_loss) + [acc, cls_acc], tags):
                tb_writer.add_scalar(tag, x, epo)  # tensorboard
            logger.info(f'Epoch: {epo}, obj_loss: {mean_loss[0].item()}, box_loss: {mean_loss[1].item()}, cls_loss: {mean_loss[2].item()}, acc: {acc}, cls_acc: {cls_acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='resnet50.pth', type=str, help='path for pretrained model')
    parser.add_argument('--train_path', default='imgset/train.txt', type=str, help='path for train set')
    parser.add_argument('--train_label_path', default='label/train.txt', type=str, help='path for train labels')
    parser.add_argument('--val_path', default='imgset/test.txt', type=str, help='path for val set')
    parser.add_argument('--val_label_path', default='label/test.txt', type=str, help='path for val labels')
    parser.add_argument('--working_dir', default='log_anchor_augment', type=str, help='dir name for logs and outputs')
    parser.add_argument('--gpus', default='0', type=str)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--num_classes', default=5, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--accumulate', default=1, type=int, help='gradient accumulate steps')
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='learning rate')
    parser.add_argument('--gamma', default=0.2, type=float, help='lr scheduler factor')
    parser.add_argument('--local_rank', default=-1, type=int)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.use_gpu = torch.cuda.is_available()
    args.num_gpu = torch.cuda.device_count()
    device = torch.device('cuda' if args.use_gpu else 'cpu')
    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    # DDP
    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert args.batch_size % args.world_size == 0, '--batch-size must be multiple of CUDA device count'
        args.batch_size = args.batch_size // args.world_size
    
    tb_writer = None  # init loggers
    if args.global_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.working_dir)  # Tensorboard
    logger = get_logger(args)
    train(args, device, logger, tb_writer)
    if args.global_rank in [-1, 0]:
        tb_writer.close()
