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
from models.model import ExponentialLoss, BMM_v1
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
    models = []
    for view_name in Views_dim:
        logger.info(f"View: {view_name}, dim: {Views_dim[view_name]}")
        tmpmodel = BMM_v1(num_classes=args.num_classes, feature_dim=Views_dim[view_name])
        tmpmodel.to(args.device)
        models.append(tmpmodel)

    # print args
    for param in sorted(vars(args).keys()):  # 遍历args的属性对象
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # get optimizer and scheduler
    args.train_dataset_len = train_dataset_len
    loss_function = ExponentialLoss(args)
    # loss_function = torch.nn.CrossEntropyLoss()
    optimizers = [torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=0.9) for model in models]
    # optimizers = [torch.optim.Adam(model.parameters(),
    #                             lr=1e-2) for model in models]

    # if args.decay_type == "cosine":
    #     scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=len(train_loader) * args.epochs)
    # else:
    #     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=len(train_loader) * args.epochs)


    D = Distribution(train_dataset_len, logger)
    alpha_final = []
    for view_iter, view in enumerate(Views_dim.keys()):
        logger.info(f"********** {view_iter+1} view ({view}) with {Views_dim[view]} dims Start **********")
        # if view_iter+1 != 1:
        #     alpha_final.append(0.0)
        #     continue
        # step1: D_0 ---> h_t(warm up), P_{k,t}
        P = Distribution(train_dataset_len, logger)
        P, targetcollect = Train_Uniview(view_iter, train_loader, val_loader, P, models[view_iter], loss_function, optimizers[view_iter], args)

        # step2: D_{t}, P_{k,t} ---> D_{t}'
        D.mul(P)
        D.paint(targetcollect, name=f"step2: view_{view_iter+1}", save_pth=os.path.join(args.path_log, f"step2_view_{view_iter+1}.png"))
        # step3: D_{t}' ---> h_t
        alpha_t, D = Train_Multiview(view_iter, train_loader, val_loader, D, models[view_iter], loss_function, optimizers[view_iter], args)
        # step4: h_t ---> Alpha_t, D_{t+1}
        alpha_final.append(alpha_t)

    # 组合强分类器
    strong_acc, strong_outputcollect, strong_targetcollect = boost_model(train_loader, val_loader, alpha_final, models, args)
    strong_error_rate = 0.0
    for i in range(len(strong_outputcollect)):
        strong_outputcollect[i] = -1 if strong_outputcollect[i] < 0 else 1
        strong_error_rate = strong_error_rate + (1 if strong_outputcollect[i] != strong_targetcollect[i] else 0)
    strong_error_rate = strong_error_rate / len(strong_outputcollect)
    print("strong_error_rate:", strong_error_rate)


# 根据view_iter模态下的特征，得到单模态下的分类器（使用指数损失，5epoch），计算损失得到分布P,并返回该模态下的分布P_{k,t}
def Train_Uniview(view_iter, train_loader, val_loader, P, model, loss_function, optimizer, args):
    start_epoch = 1
    logger.info("Start Uni-View training")
    model.zero_grad()
    for epoch in range(start_epoch, args.warmup_epochs+1):
        # train
        train_loss, train_acc, outputcollect, targetcollect = train_one_epoch_local_data(view_iter, train_loader, val_loader, P, model, loss_function, optimizer, epoch, args)
        # validate
        val_acc = validate(view_iter, val_loader, model, loss_function, epoch, args)
    # 计算误差率
    error_rate = 0.0
    for i in range(len(outputcollect)):
        outputcollect[i] = -1 if outputcollect[i] < 0 else 1
        error_rate = error_rate + (1 if outputcollect[i] != targetcollect[i] else 0)
    error_rate = error_rate / len(outputcollect)
    print("error_rate:", error_rate)
    # alpha
    error_rate = error_rate + 1e-4
    alpha = 0.5 * np.log((1 - error_rate) / error_rate)
    print("alpha:", alpha)
    # 计算分布P
    for i in range(len(outputcollect)):
        P.weights[i] = P.weights[i] * np.exp(-alpha * outputcollect[i] * targetcollect[i])
    P.norm()
    return P, targetcollect


def Train_Multiview(view_iter, train_loader, val_loader, D, model, loss_function, optimizer, args):
    start_epoch = args.warmup_epochs+1
    logger.info("Start Multi-View training")
    model.zero_grad()
    for epoch in range(start_epoch, args.epochs+1):
        # train
        train_loss, train_acc, outputcollect, targetcollect = train_one_epoch_local_data(view_iter, train_loader, val_loader, D, model, loss_function, optimizer, epoch, args)
        # save_checkpoint(epoch, model, optimizer, args.max_accuracy, args, save_name='Latest'+'-epoch'+str(epoch))
        
        # validate
        # logger.info(f"**********Latest val***********")
        # val_acc = validate(view_iter, val_loader, model, loss_function, epoch, args)

        # # 保存最好效果
        # if val_acc > args.max_accuracy:
        #     args.max_accuracy = val_acc
        #     logger.info(f"**********Best val***********")
        #     save_checkpoint(epoch, model, optimizer, args.max_accuracy, args, save_name='Best'+'-epoch'+str(epoch))
    # 计算误差率
    error_rate = 0.0
    for i in range(len(outputcollect)):
        outputcollect[i] = -1 if outputcollect[i] < 0 else 1
        error_rate = error_rate + (1 if outputcollect[i] != targetcollect[i] else 0)
    error_rate = error_rate / len(outputcollect)
    print("error_rate:", error_rate)
    # alpha
    error_rate = error_rate + 1e-4
    alpha = 0.5 * np.log((1 - error_rate) / error_rate)
    print("alpha:", alpha)
    # 计算分布P
    for i in range(len(outputcollect)):
        D.weights[i] = D.weights[i] * np.exp(-alpha * outputcollect[i] * targetcollect[i])
    D.norm()
    return alpha, D


def train_one_epoch_local_data(view_iter, train_loader, val_loader, D, model, loss_function, optimizer, epoch, args):
    model.train()
    optimizer.zero_grad()

    num_steps = len(train_loader)
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    outputcollect = []
    targetcollect = []
    for iter, (data, target) in enumerate(train_loader):
        data = data[view_iter].to(args.device)  # 选择对应维度
        target = target.to(args.device)  # [32]
        
        output = model(data).to(args.device)  # [32]
        # if (iter == 0):
        #     print("output:", output.shape)
        #     print("target:", target.shape)
        #     print("output:", output)
        #     print("target:", target)
        outputcollect.extend(output.tolist())  # [32*26+30=862]
        targetcollect.extend(target.tolist())  # [32*26+30=862]

        this_D = D.weights
        this_D = this_D[iter*args.train_batch_size: min((iter+1)*args.train_batch_size, this_D.shape[0])]  # 选取对应batch的D
        this_D = torch.from_numpy(this_D).cuda()
        loss = loss_function(output, target, this_D)
        acc1 = accuracy_twoclass(output, target)
        loss.requires_grad_(True)
        # print(f"loss: {loss}")
        loss.backward()
        # 解决梯度爆炸！！！
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # 储存loss和acc
        loss_meter.update(loss.item(), output.size(0))
        acc1_meter.update(acc1.item(), output.size(0))
        tb_writer.add_scalar('train_loss', loss.item(), (epoch-1) * num_steps + iter)
        tb_writer.add_scalar('train_acc', acc1.item(), (epoch-1) * num_steps + iter)
        # tb_writer.add_scalar('train_lr', scheduler.get_lr()[0], (epoch-1) * num_steps + iter)
        # log输出训练参数
        # if iter % 5 == 0:
        #     logger.info(
        #         f'Train: [{epoch}/{args.epochs}][{iter}/{num_steps}]\t'
        #         f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        #         f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t')
        # if iter % 200 == 0 and iter > 0:
        #     val_loss, val_acc = validate(val_loader, model, loss_function, epoch, args)
        #     if val_acc > args.max_accuracy:
        #         args.max_accuracy = val_acc
        #         logger.info(f'Max accuracy: {args.max_accuracy:.4f}')
        #         save_checkpoint(epoch, model, optimizer, args.max_accuracy, args, save_name='Best')
        #     model.train()
    logger.info(f"Train: [{epoch}/{args.epochs}], loss_meter.avg: {loss_meter.avg:.7f}, acc1_meter.avg: {acc1_meter.avg:.7f}")
    return loss_meter.avg, acc1_meter.avg, outputcollect, targetcollect

@torch.no_grad()
def validate(view_iter, val_loader, model, loss_function, epoch, args):
    # switch to evaluate mode
    # logger.info('eval epoch {}'.format(epoch))
    model.eval()

    num_steps = len(val_loader)
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    for iter, (data, target) in enumerate(val_loader):
        data = data[view_iter].to(args.device)  # 选择对应维度
        target = target.to(args.device)

        output = model(data).to(args.device)
        # loss = loss_function(output, target)
        acc1 = accuracy_twoclass(output, target)

        # loss_meter.update(loss.item(), output.size(0))
        acc1_meter.update(acc1.item(), output.size(0))
        # tb_writer.add_scalar('val_loss', loss.item(), (epoch-1) * num_steps + iter)
        tb_writer.add_scalar('val_acc', acc1.item(), (epoch-1) * num_steps + iter)
        # log输出测试参数
        # if iter % 5 == 0:
        #     logger.info(
        #         f'Test: [{iter}/{len(val_loader)}]\t'
        #         f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        #         f'acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t')
    logger.info(f"Eval: [{epoch}/{args.epochs}], acc1_meter.avg: {acc1_meter.avg:.7f}")
    return acc1_meter.avg

def boost_model(train_loader, val_loader, alpha_final, models, args):    
    acc1_meter = AverageMeter()
    outputcollect = []
    targetcollect = []
    for iter, (data, target) in enumerate(train_loader):
        strong_output = np.array([0 for i in range(len(target))])
        target = target.to(args.device)
        for view_iter, alpha_t in enumerate(alpha_final):
            choose_data = data[view_iter].to(args.device)
            output_ = models[view_iter](choose_data).to(args.device)
            output_ = output_.detach().cpu().numpy()

            for i in range(len(output_)):
                output_[i] = -1.0*alpha_t if output_[i] < 0 else alpha_t
            strong_output = strong_output + output_
        # 计算强分类器输出
        strong_output = np.sign(strong_output)
        outputcollect.extend(strong_output.tolist())  # [32*26+30=862]
        targetcollect.extend(target.tolist())  # [32*26+30=862]

        acc1 = accuracy_twoclass(torch.tensor(strong_output), target)
        acc1_meter.update(acc1.item(), torch.tensor(strong_output).size(0))
        # if iter % 5 == 0:
        #     logger.info(f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t')
    logger.info(f"Train Boost: acc1_meter.avg: {acc1_meter.avg:.7f}")
    # validate
    acc1_meter.reset()
    outputcollect = []
    targetcollect = []
    for iter, (data, target) in enumerate(val_loader):
        strong_output = np.array([0 for i in range(len(target))])
        target = target.to(args.device)
        for view_iter, alpha_t in enumerate(alpha_final):
            choose_data = data[view_iter].to(args.device)
            output_ = models[view_iter](choose_data).to(args.device)
            output_ = output_.detach().cpu().numpy()

            for i in range(len(output_)):
                output_[i] = -1.0*alpha_t if output_[i] < 0 else alpha_t
            strong_output = strong_output + output_
        # 计算强分类器输出
        strong_output = np.sign(strong_output)
        outputcollect.extend(strong_output.tolist())  # [32*26+30=862]
        targetcollect.extend(target.tolist())  # [32*26+30=862]

        acc1 = accuracy_twoclass(torch.tensor(strong_output), target)
        acc1_meter.update(acc1.item(), torch.tensor(strong_output).size(0))
        # if iter % 5 == 0:
        #     logger.info(f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t')
    logger.info(f"Eval Boost: acc1_meter.avg: {acc1_meter.avg:.7f}")
    return acc1_meter.avg, outputcollect, targetcollect


if __name__ == "__main__":
    main(args)