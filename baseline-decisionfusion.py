# from sklearn.ensemble import GradientBoostingClassifier
import logging
import os
import time
import datetime
import numpy as np
from datetime import timedelta

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from timm.utils import accuracy, AverageMeter

from core.utils import create_logging, save_checkpoint, Distribution, accuracy_twoclass
from core.data import get_loader
from models.baseline_model import ExponentialLoss, Baseline_DecisionFusion
from core.param import args
from core.scheduler import WarmupLinearSchedule, WarmupCosineSchedule

Views_dim = {'gabor': 48, 'wavelet_moments': 40, 'cenhist': 254, 'hog': 1984, 'gist': 512, 'lbp': 928}

# get logger
creat_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())  # 获取训练创建时间
args.path_log = os.path.join(args.save_dir, f'{args.data}', f'{args.name}')  # 确定训练log保存路径
os.makedirs(args.path_log, exist_ok=True)  # 创建训练log保存路径
logger = create_logging(os.path.join(args.path_log, '%s-%s-train.log' % (creat_time, args.name)))  # 创建训练保存log文件
# tensorboard为acc和loss画图
tb_writer = SummaryWriter(log_dir=args.path_log)


def main(args):
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    # get datasets
    train_loader, train_dataset_len = get_loader(data_name=args.data, mode='train', TwoClass=args.TwoClass, args=args)
    val_loader, val_dataset_len = get_loader(data_name=args.data, mode='val', TwoClass=args.TwoClass, args=args)
    # get net
    logger.info(f"Creating model: BMM_v1")
    model = Baseline_DecisionFusion(num_classes=args.num_classes, feature_dict=Views_dim)
    model.to(args.device)

    # print args
    for param in sorted(vars(args).keys()):  # 遍历args的属性对象
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # get optimizer and scheduler
    # args.train_dataset_len = train_dataset_len
    loss_function = ExponentialLoss(args)
    # loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9)

    # Train concat model
    start_epoch = 1
    args.max_accuracy = 0.0
    logger.info("Start baseline_concat_model training")
    model.zero_grad()
    for epoch in range(start_epoch, args.epochs + 1):
        # train
        train_loss, train_acc = train_one_epoch_local_data(train_loader, val_loader, model, loss_function, optimizer, epoch, args)
        # save_checkpoint(epoch, model, optimizer, args.max_accuracy, args, logger, save_name='Latest'+'-epoch'+str(epoch))
        
        # validate
        logger.info(f"**********Latest val***********")
        val_loss, val_acc = validate(val_loader, model, loss_function, epoch, args)
        # 保存最好效果
        if val_acc > args.max_accuracy:
            args.max_accuracy = val_acc
            logger.info(f'Max accuracy: {args.max_accuracy:.4f}')
            # save_checkpoint(epoch, model, optimizer, args.max_accuracy, args, logger, save_name='Best')
        logger.info('Exp path: %s' % args.path_log)


def train_one_epoch_local_data(train_loader, val_loader, model, loss_function, optimizer, epoch, args):
    model.train()

    num_steps = len(train_loader)
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    for iter, (data, target) in enumerate(train_loader):
        data = [data[i].to(args.device) for i in range(len(data))]  # 选择对应维度
        target = target.to(args.device)
        optimizer.zero_grad()

        output = model(data).to(args.device)
        loss = loss_function(output, target).requires_grad_(True)
        acc1 = accuracy_twoclass(output, target)
        # loss.requires_grad_(True)
        # print(f"loss: {loss}")
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # 储存loss和acc
        loss_meter.update(loss.item(), output.size(0))
        acc1_meter.update(acc1.item(), output.size(0))
        tb_writer.add_scalar('train_loss', loss.item(), (epoch-1) * num_steps + iter)
        tb_writer.add_scalar('train_acc', acc1.item(), (epoch-1) * num_steps + iter)
        # log输出训练参数
        if iter % 5 == 0:
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{iter}/{num_steps}]\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})')
    logger.info(f"Train: [{epoch}/{args.epochs}], loss_meter.avg: {loss_meter.avg:.7f}, acc1_meter.avg: {acc1_meter.avg:.7f}")
    return loss_meter.avg, acc1_meter.avg

@torch.no_grad()
def validate(val_loader, model, loss_function, epoch, args):
    model.eval()
    logger.info('eval epoch {}'.format(epoch))

    num_steps = len(val_loader)
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    for iter, (data, target) in enumerate(val_loader):
        data = [data[i].to(args.device) for i in range(len(data))]  # 选择对应维度
        target = target.to(args.device)

        output = model(data).to(args.device)
        loss = loss_function(output, target)
        acc1 = accuracy_twoclass(output, target)

        loss_meter.update(loss.item(), output.size(0))
        acc1_meter.update(acc1.item(), output.size(0))
        tb_writer.add_scalar('val_loss', loss.item(), (epoch-1) * num_steps + iter)
        tb_writer.add_scalar('val_acc', acc1.item(), (epoch-1) * num_steps + iter)
        # log输出测试参数
        if iter % 5 == 0:
            logger.info(
                f'Test: [{iter}/{len(val_loader)}]\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t')
    logger.info(f"Eval: [{epoch}/{args.epochs}], loss_meter.avg: {loss_meter.avg:.7f}, acc1_meter.avg: {acc1_meter.avg:.7f}")
    return loss_meter.avg, acc1_meter.avg

if __name__ == "__main__":
    main(args)