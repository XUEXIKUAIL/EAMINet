#os 库是Python标准库，包含几百个函数，常用的有路径操作、进程管理、环境参数等。
import os
import shutil
import json
import time
# from apex import amp
from torch.cuda import amp
import tqdm
# import apex
import numpy as np
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader

from .config import MscCrossEntropyLoss
from .config import get_dataset
from .config import get_logger
from .config import get_model_t
from .config import averageMeter, runningScore
from .config import ClassWeight, save_ckpt, load_ckpt
torch.manual_seed(123)

cudnn.benchmark = True
from .L_contrast import ContrastCELoss
def run(args):

    with open(args.config, 'r') as fp:
        cfg = json.load(fp)


    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}_Model_XXX'
    #logdir = 'run/2020-12-23-18-38'

    args.logdir = logdir

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    if args.local_rank == 0:
        logger.info(f'Conf | use logdir {logdir}')

    model = get_model_t(cfg)
    ######################Need Your Pth!###################
    model.load_pre_b2('//MiT-b2.pth')
    # model.load_pre_b0('//MiT-b0.pth')

    trainset, *testset = get_dataset(cfg)
    device = torch.device('cuda:0')
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.local_rank == 0:
            print(f"WORLD_SIZE is {os.environ['WORLD_SIZE']}")

    train_sampler = None
    # Distributed Training! #
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()

        # model = apex.parallel.convert_syncbn_model(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    model.to(device)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=(train_sampler is None),
                              num_workers=cfg['num_workers'], pin_memory=True, sampler=train_sampler, drop_last=True)
    #                                             drop_last=True解决照片留单然后导致batch变成1
    val_loader = DataLoader(testset[0], batch_size=cfg['ims_per_gpu'], shuffle=False,num_workers=cfg['num_workers'],pin_memory=True, drop_last=True)
    params_list = model.parameters()
    # wd_params, non_wd_params = model.get_params()
    # params_list = [{'params': wd_params, },
    #                {'params': non_wd_params, 'weight_decay': 0}]
    # optimizer = torch.optim.Adam(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    optimizer = torch.optim.SGD(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])
    Scaler = amp.GradScaler()
    #optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay']
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)

    # model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    # if args.distributed:
    #   model = torch.nn.parallel.DistributedDataParallel(model)

    # class weight 计算
    if hasattr(trainset, 'class_weight'):
        print('using classweight in dataset')
        class_weight = trainset.class_weight
    else:
        classweight = ClassWeight(cfg['class_weight'])
        class_weight = classweight.get_weight(train_loader, cfg['n_classes'])

    # classweight = ClassWeight(cfg['class_weight'])
    # class_weight = classweight.get_weight(train_loader, cfg['n_classes'])

    class_weight = torch.from_numpy(class_weight).float().to(device)
    # print(class_weight)
    class_weight[cfg['id_unlabel']] = 0


    # 损失函数 & 类别权重平衡 & 训练时ignore unlabel
    criterion = MscCrossEntropyLoss(weight=class_weight).to(device)
    contra_loss = ContrastCELoss().to(device)

    # 指标 包含unlabel
    train_loss_meter = averageMeter()
    val_loss_meter = averageMeter()
    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    # 每个epoch迭代循环

    flag = True #为了先保存一次模型做的判断
    #设置一个初始miou
    miou = 0
    for ep in range(cfg['epochs']):
        if args.distributed:
            train_sampler.set_epoch(ep)

        # training
        model.train()
        train_loss_meter.reset()
        # teacher.eval()

        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零


            ################### train edit #######################
            depth = sample['depth'].to(device)
            image = sample['image'].to(device)
            label = sample['label'].to(device)
            # edge = sample['edge'].to(device)
            with amp.autocast():
                predict = model(image, depth)

                Seg_loss = criterion(predict[0], label) + criterion(predict[1][0], label)+criterion(predict[1][1], label)+criterion(predict[1][2],label)

                Con_loss = contra_loss(predict[0], predict[2], label)

                loss = Seg_loss + Con_loss
            ####################################################

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()
            # optimizer.step()

            if args.distributed:
                reduced_loss = loss.clone()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss /= args.world_size
            else:
                reduced_loss = loss
            train_loss_meter.update(reduced_loss.item())

        scheduler.step(ep)

        # val
        with torch.no_grad():
            model.eval()
            running_metrics_val.reset()

            val_loss_meter.reset()
            ################### val edit #######################
            for i, sample in enumerate(val_loader):
                depth = sample['depth'].to(device)
                image = sample['image'].to(device)
                label = sample['label'].to(device)

                predict = model(image, depth)

                loss = criterion(predict[0], label)

                val_loss_meter.update(loss.item())

                predict = predict[0].max(1)[1].cpu().numpy()  # [1, h, w]
                label = label.cpu().numpy()

            ###################edit end#########################
                running_metrics_val.update(label, predict)


        if args.local_rank == 0:
            logger.info(
                f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] train/val loss={train_loss_meter.avg:.5f}/{val_loss_meter.avg:.5f}, '
                f'miou={running_metrics_val.get_scores()[0]["mIou: "]:.4f}, '
                f'PA={running_metrics_val.get_scores()[0]["pixel_acc: "]:.4f}, '
                f'CA={running_metrics_val.get_scores()[0]["class_acc: "]:.4f}, '
                f'best_miou={miou:.4f}')
            save_ckpt(logdir, model, kind='end')
            newmiou = running_metrics_val.get_scores()[0]["mIou: "]
            # newPA = running_metrics_val.get_scores()[0]["pixel_acc: "]
            # newCA = running_metrics_val.get_scores()[0]["class_acc: "]


            if newmiou > miou:
                save_ckpt(logdir, model, kind='best')
                miou = newmiou


    # save_ckpt(logdir, model, kind='end')  #保存最后一个模型参数

if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nyuv2.json",
        # default="configs/sunrgbd.json",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--opt_level",
        type=str,
        default='O1',
    )

    args = parser.parse_args()
    run(args)