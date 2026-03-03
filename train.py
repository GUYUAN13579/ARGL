# coding:utf-8
# ----------------------------------------------------------------------------
# Pytorch multi-GPU YOLOV5 based UMT
# Licensed under The MIT License [see LICENSE for details]
# Written by Huayi Zhou, based on code from
# https://github.com/kinredon/umt
# https://github.com/ultralytics/yolov5
# ----------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import math
import os
import random
import sys
import time
import warnings
import yaml
import numpy as np
from copy import deepcopy
from pathlib import Path
from threading import Thread
from tqdm import tqdm

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data

from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

import test as test  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model

from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.datasets_single import create_dataloader_single
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss

from utils.torch_utils import ModelEMA, WeightEMA, select_device, intersect_dicts, torch_distributed_zero_first, \
    de_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.metrics import fitness

from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr, \
    non_max_suppression, check_dataset_umt, xyxy2xywhn
import cv2

logger = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


# 在检测到水平线后添加坐标转换
def adjust_horizon_coord(orig_h, new_h, y):
    scale = new_h / orig_h
    return int(y * scale)


def detect_horizon(img):
    if img is None:
        return 0, 0, 0

    # if isinstance(img,Image.Image):
    #    #img=np.array(img)
    #     imgray=np.array(img.convert('L'))
    # else:#default:cv2
    #   if len(img.shape)==3:
    # imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #  else:
    #     imgray=img

    if len(img.shape) == 3:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        imgray = img
    # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray, (3, 3), 0, 0)
    thesd = 0.0
    Mkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thesd, imggtem = cv2.threshold(imgray, thesd, 255, cv2.THRESH_OTSU)
    dilatedgray = cv2.dilate(imggtem, kernel=Mkernel)

    # imgsrc = img.copy()
    # cv2.imshow('dt', dilatedgray)
    # cv2.imshow('gt', imggtem)
    # cv2.waitKey(1)

    imgmask = cv2.reduce(dilatedgray, 1, cv2.REDUCE_AVG, cv2.CV_16S)  # //reduce to single column average
    row, col = imgray.shape
    # row = imgray.rows, col = imgray.cols;
    kuan_hight = round(row / 20)

    horizon_top = 0
    horizon_bottom = row - 1  # ;//区域上下界
    thesd, imgtem = cv2.threshold(imgmask, thesd, 255, cv2.THRESH_OTSU)
    imgtemd = np.abs(cv2.filter2D(imgtem, cv2.CV_16S, np.array([[-1], [0], [1]])))

    bottom_temp = row - 1  # ;//区域下界
    flagContinue = False
    for i in range(kuan_hight, row - 1):  # (int i = kuan_hight; i < row; i++)//获得海天线上下界
        ppre = imgtemd[i, 0]
        # paft=imgtem[i+1,0]

        if ppre == 0:  # 寻找跳变，先验认为天空255,当没有河流则可能被认为255，增加0->255的判断,抓住第一个跳变
            continue
        top_temp = i - 1.5 * kuan_hight  # //海天线上界
        horizon_top = 0 if top_temp < 0 else top_temp
        bottom_temp = i + 1.5 * kuan_hight
        horizon_bottom = row - 1 if bottom_temp >= row else bottom_temp  # 海天线下界
        break

    # horizonLine=round((horizon_bottom + horizon_top) / 2)
    # imgsrc=cv2.line(imgsrc,(0,horizonLine),(img.shape[1]-1,horizonLine),(0,0,255),2)
    # cv2.imshow('line',imgsrc)
    # cv2.waitKey(1)

    return (horizon_bottom + horizon_top) / 2  # 返回水平线

def calculate_weights(y_centers_pseudo, pred_tf_nms, horizon_ys, img_height):
    # 水平线位置均值
    mean_h = torch.mean(horizon_ys)
    # 伪标签中心点方差（加平滑项）
    var_y = torch.var(y_centers_pseudo, unbiased=False) if len(y_centers_pseudo) > 1 else 0.1
    var_y = torch.clamp(var_y, min=1e-3)  # 防止方差过小
    weights=[]
    for img_id in range(len(pred_tf_nms)):
        # 转换为图像尺度坐标
        y_img = y_centers_pseudo[img_id] * img_height
        mean_h_img = mean_h.item() * img_height
        var_y_img = var_y.item() * (img_height ** 2)  # 方差随尺度平方变化

        # 动态高斯权重计算
        weight = torch.exp(-(y_img - mean_h_img) ** 2 / (2 * var_y_img))

        # 标准化到[0.2, 1.0]范围
        weight = 0.8 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-6) + 0.2
        weights.append(weight)
    return weights
'''
# 在生成伪标签时计算水平线权重
def generate_pseudo_labels(pred_tf_nms, horizon_ys, img_height):
    pseudo_labels = []
    y_centers=[]
    for img_id in range(len(pred_tf_nms)):
        img_pred = pred_tf_nms[img_id]
        if len(img_pred) == 0:
            continue

        # 获取当前图像水平线位置
        horizon_y = horizon_ys[img_id] * img_height  # 反归一化

        # 计算每个预测框的垂直中心
        y_centers = (img_pred[:, 1] + img_pred[:, 3]) / 2 * img_height

        # 高斯权重计算
        sigma = img_height * 0.1  # 10%图像高度为影响范围
        weights = np.exp(-(y_centers.cpu().numpy() - horizon_y) ** 2 / (2 * sigma ** 2))
        weights = torch.from_numpy(weights).to(img_pred.device)

        # 融合到伪标签
        weighted_pred = torch.cat([img_pred, weights.unsqueeze(1)], dim=1)
        pseudo_labels.append(weighted_pred)

    return torch.cat(pseudo_labels) if pseudo_labels else None
'''
def generate_pseudo_labels(pred_tf_nms, horizon_ys, img_height, per_batch_size, device):
    pseudo_labels = []
    
    # 计算整个batch的水平线均值（实际像素坐标）
    all_horizon_ys = (horizon_ys.float()) * img_height
    batch_horizon_mean = torch.mean(all_horizon_ys)
    
    for img_id in range(per_batch_size):
        #img_pred = pred_tf_nms[img_id]
        
        # 安全获取当前图像的预测结果
        img_pred = pred_tf_nms[img_id] if img_id < len(pred_tf_nms) else torch.zeros((0,6), device=device)
        
        # 确保始终是二维张量 [N,6]
        if img_pred.dim() == 1:
            img_pred = img_pred.view(-1, 6)
        
        if len(img_pred) == 0:
             filtered_pred = torch.zeros((0,6), device=device)
        else:
            #print(len(img_pred))
            # 获取当前图像水平线位置
            horizon_y = horizon_ys[img_id].item() * img_height
        
            # 计算中心点y坐标并过滤
            #y_centers = (img_pred[:,1] + img_pred[:,3]) * img_height / 2
            y_centers = (img_pred[:,1] + img_pred[:,3]) / 2
            valid_mask = y_centers < (horizon_y / img_height)  # 直接使用中心点比较
            filtered_pred = img_pred[valid_mask]

            # 提前检查空张量
            #if len(filtered_pred) == 0:
            #    continue  # 跳过后续处理
            
            if len(filtered_pred) > 0:
                # 计算当前图像伪标签中心点的方差（实际像素坐标）
                y_centers_pixels = y_centers[valid_mask] * img_height  # 转实际坐标
                var_y = torch.var(y_centers_pixels, unbiased=False)     # 样本方差
                var_y = torch.clamp(var_y, min=1e-3)                   # 防止过小
                # 高斯权重计算
                batch_horizon_mean = torch.mean(all_horizon_ys)
                
                distances = torch.abs(y_centers_pixels - batch_horizon_mean)
                weights = torch.exp(-distances**2 / (2 * var_y**2))

                #distances = torch.abs(y_centers[valid_mask] - horizon_y)  # 使用过滤后的中心点
                #sigma = img_height * 0.1
                #weights = torch.exp(-(distances**2) / (2 * sigma**2))
                weights = 0.5 + 0.5*(weights - weights.min())/(weights.max() - weights.min() + 1e-6)
        
                # 调整置信度并过滤
                filtered_pred[:,4] *= weights
                conf_mask = filtered_pred[:,4] >= 0.15
                filtered_pred = filtered_pred[conf_mask]
            
            #print(len(filtered_pred))
            #if len(filtered_pred) > 0:
        if len(filtered_pred) == 0:
            filtered_pred = torch.zeros((0,6), device=device)
        pseudo_labels.append(filtered_pred if filtered_pred.dim() == 2 else filtered_pred.view(-1,6))

    #return torch.cat(pseudo_labels) if pseudo_labels else None
    return pseudo_labels

# hyp means path/to/hyp.yaml or hyp dictionary
def train(hyp, opt, device):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, notest, nosave, workers, = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
            opt.resume, opt.notest, opt.nosave, opt.workers

    teacher_alpha, conf_thres, iou_thres, max_gt_boxes, lambda_weight, student_weight, teacher_weight = \
        opt.teacher_alpha, opt.conf_thres, opt.iou_thres, opt.max_gt_boxes, opt.lambda_weight, \
            opt.student_weight, opt.teacher_weight

    all_shift = opt.consistency_loss

    # Directories
    save_dir = Path(save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last_student, last_teacher = wdir / 'last_student.pt', wdir / 'last_teacher.pt'
    best_student, best_teacher = wdir / 'best_student.pt', wdir / 'best_teacher.pt'
    results_file = save_dir / 'results.txt'

    # 新建viewer目录
    viewer_dir = save_dir / 'viewer'
    (viewer_dir / 'weights').mkdir(parents=True, exist_ok=True)  # viewer下的权重目录
    (viewer_dir / 'visualizations').mkdir(parents=True, exist_ok=True)  # viewer下的可视化目录

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp) as f:  # default path data/hyps/hyp.scratch.yaml
            hyp = yaml.safe_load(f)  # load hyps dict
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with open(data) as f:
        data_dict = yaml.safe_load(f)  # data dict

    # Loggers
    loggers = {'wandb': None, 'tb': None}  # loggers dict
    if RANK in [-1, 0]:
        # TensorBoard
        if not evolve:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            loggers['tb'] = SummaryWriter(str(save_dir))

        # W&B
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        run_id = run_id if opt.resume else None  # start fresh run if transfer learning
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        if loggers['wandb']:
            data_dict = wandb_logger.data_dict
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # may update weights, epochs if resuming

    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data)  # check
    is_coco = data.endswith('coco.yaml') and nc == 80  # COCO dataset

    # Model
    pretrained = weights.endswith('.pt')
    # torch.cuda.empty_cache()
    # strip_optimizer(weights)  # strip optimizers, this will apparently reduce the model size
    if pretrained:
        with torch_distributed_zero_first(RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        # model_student
        model_student = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        # model_teacher
        model_teacher = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model_student.state_dict(), exclude=exclude)  # intersect
        model_student.load_state_dict(state_dict, strict=False)  # load
        # model_teacher.load_state_dict(state_dict, strict=False)  # load
        model_teacher.load_state_dict(state_dict.copy(), strict=False)  # load
        logger.info(
            'Transferred %g/%g items from %s' % (len(state_dict), len(model_student.state_dict()), weights))  # report
    else:
        model_student = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        model_teacher = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    # Update models weights [only by this way, we can resume the old training normally...][ref models.experimental.attempt_load()]
    if student_weight != "None" and teacher_weight != "None":  # update model_student and model_teacher
        torch.cuda.empty_cache()
        ckpt_student = torch.load(student_weight, map_location=device)  # load checkpoint
        state_dict_student = ckpt_student[
            'ema' if ckpt_student.get('ema') else 'model'].float().half().state_dict()  # to FP32
        model_student.load_state_dict(state_dict_student, strict=False)  # load
        del ckpt_student, state_dict_student

        ckpt_teacher = torch.load(teacher_weight, map_location=device)  # load checkpoint
        state_dict_teacher = ckpt_teacher[
            'ema' if ckpt_teacher.get('ema') else 'model'].float().half().state_dict()  # to FP32
        model_teacher.load_state_dict(state_dict_teacher, strict=False)  # load
        del ckpt_teacher, state_dict_teacher

    # Dataset
    with torch_distributed_zero_first(RANK):
        # check_dataset(data_dict)  # check, need to be re-write or command out
        check_dataset_umt(data_dict)  # check, need to be re-write or command out
    train_path_source_real = data_dict['train_source_real']  # training source dataset w labels
    train_path_source_fake = data_dict['train_source_fake']  # training target-like dataset w labels
    train_path_target_real = data_dict['train_target_real']  # training target dataset w/o labels
    train_path_target_fake = data_dict['train_target_fake']  # training source-like dataset w/o labels
    test_path_target_real = data_dict[
        'test_target_real']  # test on target dataset w labels, should not use testset to train
    # test_path_target_real = data_dict['train_target_real']  # test on target dataset w labels, remember val in 'test_target_real'

    # Freeze
    freeze_student = []  # parameter names to freeze (full or partial)
    for k, v in model_student.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze_student):
            print('freezing %s' % k)
            v.requires_grad = False
    freeze_teacher = []  # parameter names to freeze (full or partial)
    for k, v in model_teacher.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze_teacher):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model_student.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.adam:
        student_optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        student_optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    student_optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    student_optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # UMT algorithm
    student_detection_params = []
    for key, value in model_student.named_parameters():
        if value.requires_grad:
            student_detection_params += [value]
    teacher_detection_params = []
    for key, value in model_teacher.named_parameters():
        if value.requires_grad:
            teacher_detection_params += [value]
            value.requires_grad = False
    teacher_optimizer = WeightEMA(teacher_detection_params, student_detection_params, alpha=teacher_alpha)

    # For debugging
    # for k, v in model_student.named_parameters():
    # print(k, v.requires_grad)
    # for k, v in model_teacher.named_parameters():
    # print(k, v.requires_grad)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(student_optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA (exponential moving average)
    ema = ModelEMA(model_student) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        '''
        if ckpt['optimizer'] is not None:
            student_optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        '''
        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        ckpt['epoch'] = 0
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epoches

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model_student.stride.max()), 32)  # grid size (max stride)
    nl = model_student.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model_student = torch.nn.DataParallel(model_student)
        model_teacher = torch.nn.DataParallel(model_teacher)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model_student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_student).to(device)
        model_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_teacher).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    '''
    dataloader_sr, dataset_sr = create_dataloader(train_path_source_real, train_path_source_fake, imgsz, batch_size // WORLD_SIZE, 
        gs, single_cls, hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=RANK, workers=workers,
        image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train_source_real_fake: '))
    dataloader_tr, dataset_tr = create_dataloader(train_path_target_real, train_path_target_fake, imgsz, batch_size // WORLD_SIZE, 
        gs, single_cls, hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=RANK, workers=workers,
        image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train_target_real_fake: '))
    '''
    dataloader_sr, dataset_sr = create_dataloader(train_path_source_real, train_path_source_fake, imgsz,
                                                  batch_size // WORLD_SIZE,
                                                  gs, single_cls, hyp=hyp, augment=True, cache=opt.cache_images,
                                                  rect=opt.rect, rank=RANK, workers=workers,
                                                  image_weights=opt.image_weights, quad=opt.quad,
                                                  prefix=colorstr('train_source_real_fake: '))
    dataloader_tr, dataset_tr = create_dataloader(train_path_target_real, train_path_target_fake, imgsz,
                                                  batch_size // WORLD_SIZE,
                                                  gs, single_cls, hyp=hyp, augment=True, cache=opt.cache_images,
                                                  rect=opt.rect, rank=RANK, workers=workers,
                                                  image_weights=opt.image_weights, quad=opt.quad,
                                                  prefix=colorstr('train_target_real_fake: '))

    mlc = np.concatenate(dataset_sr.labels, 0)[:, 0].max()  # max label class
    # nb = len(dataloader_sr)  # number of batches. (For knowledge distillation, shall we calculate iters_per_epoch like this?)
    # nb = max(len(dataloader_sr), len(dataloader_tr))  # number of batches. [This way will lead to larger dataset dominanting train]
    nb = (len(dataloader_sr) + len(
        dataloader_tr)) // 2  # number of batches. [This way will keep a balance between double datasets]
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, data, nc - 1)

    # Process 0
    if RANK in [-1, 0]:
        testloader = create_dataloader_single(test_path_target_real, imgsz_test, batch_size // WORLD_SIZE * 2,
                                              gs, single_cls, hyp=hyp, cache=opt.cache_images and not notest, rect=True,
                                              rank=-1,
                                              workers=workers, pad=0.5, prefix=colorstr('val_source_real: '))[0]

        if not resume:
            labels = np.concatenate(dataset_sr.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir, loggers)  # ./labels.jpg, ./labels_correlogram.jpg
                if loggers['tb']:
                    loggers['tb'].add_histogram('classes', c, 0)  # TensorBoard

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset_sr, model=model_student, thr=hyp['anchor_t'], imgsz=imgsz)
                check_anchors(dataset_tr, model=model_teacher, thr=hyp['anchor_t'], imgsz=imgsz)
            model_student.half().float()  # pre-reduce anchor precision
            model_teacher.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and RANK != -1:
        # model_student = DDP(model_student, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        '''
        # https://www.wangt.cc/2021/06/one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/
        # https://discuss.pytorch.org/t/ddp-sync-batch-norm-gradient-computation-modified/82847/5
        # for fixing bug: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation [2021-12-14]
        '''
        model_student = DDP(model_student, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, broadcast_buffers=False)
        # model_teacher = DDP(model_teacher, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model_student.nc = nc  # attach number of classes to model
    model_student.hyp = hyp  # attach hyperparameters to model
    model_student.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model_student.class_weights = labels_to_class_weights(dataset_sr.labels, nc).to(device) * nc  # attach class weights
    model_student.names = names
    model_teacher.nc = nc  # attach number of classes to model
    model_teacher.hyp = hyp  # attach hyperparameters to model
    model_teacher.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model_teacher.class_weights = labels_to_class_weights(dataset_tr.labels, nc).to(device) * nc  # attach class weights
    model_teacher.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    # results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    results = (
    0, 0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.75, mAP@.5-.95, val_loss(box, obj, cls)  # Added in 2021-10-01
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model_student)  # init loss class. [Not used in knowledge distillation based UMT]
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader_sr.num_workers} dataloader_sr workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # start epoch ------------------------------------------------------------
        model_student.train()
        model_teacher.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if RANK in [-1, 0]:
                cw = model_student.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset_sr.labels, nc=nc, class_weights=cw)  # image weights
                dataset_sr.indices = random.choices(range(dataset_sr.n), weights=iw,
                                                    k=dataset_sr.n)  # rand weighted idx
            # Broadcast if DDP
            if RANK != -1:
                indices = (torch.tensor(dataset_sr.indices) if RANK == 0 else torch.zeros(dataset_sr.n)).int()
                dist.broadcast(indices, 0)
                if RANK != 0:
                    dataset_sr.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset_sr.mosaic_border = [b - imgsz, -b]  # height, width borders

        # mloss = torch.zeros(4, device=device)  # mean losses
        if all_shift:
            mloss = torch.zeros((4 + all_shift), device=device)  # mean losses
        else:
            mloss = torch.zeros(4, device=device)  # mean losses

        # if RANK != -1 and False:  # load dats sequentially in UMT
        if RANK != -1:  # load dats sequentially in UMT
            dataloader_sr.sampler.set_epoch(epoch)  # For DistributedSampler, this will shuffle dataset
            dataloader_tr.sampler.set_epoch(epoch)
            # dataloader_sr.sampler.set_epoch(epoch+random.random())  # For DistributedSampler, this will shuffle dataset
            # dataloader_tr.sampler.set_epoch(epoch+random.random())

        # pbar = enumerate(dataloader)
        pbar = enumerate([ind for ind in range(nb)])
        # source and target dataset have different images number
        data_iter_sr = iter(dataloader_sr)
        data_iter_tr = iter(dataloader_tr)
        # logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if all_shift:
            log_list = ['Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size']
            if opt.consistency_loss: log_list = log_list[:6] + ['cons'] + log_list[6:]
            shift = opt.consistency_loss
            # if opt.sem_gcn: log_list = log_list[:6+shift] + ['sem'] + log_list[6+shift:]
            # shift += opt.sem_gcn
            logger.info(('\n' + '%10s' * (8 + shift)) % tuple(log_list))
        else:
            logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))

        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        student_optimizer.zero_grad()

        # for i, (imgs, targets, paths, _) in pbar:
        for i, ind in pbar:
            # start batch -------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            # imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # for model_student, source real and fake images, with using labels
            try:
                imgs_sr, imgs_sf, targets_sr, paths_sr, paths_sf, _ = next(data_iter_sr)
            except:
                data_iter_sr = iter(dataloader_sr)
                imgs_sr, imgs_sf, targets_sr, paths_sr, paths_sf, _ = next(data_iter_sr)
            imgs_sr = imgs_sr.to(device, non_blocking=True).float() / 255.0
            imgs_sf = imgs_sf.to(device, non_blocking=True).float() / 255.0

            # for model_student and model_teacher, target real and fake images, without using labels
            try:
                imgs_tr, imgs_tf, targets_tr, paths_tr, paths_tf, _ = next(data_iter_tr)
            except:
                data_iter_tr = iter(dataloader_tr)
                imgs_tr, imgs_tf, targets_tr, paths_tr, paths_tf, _ = next(data_iter_tr)
            imgs_tr = imgs_tr.to(device, non_blocking=True).float() / 255.0
            imgs_tf = imgs_tf.to(device, non_blocking=True).float() / 255.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(student_optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                temp_imgs_list = [imgs_sr, imgs_sf, imgs_tr, imgs_tf]
                for i, imgs in enumerate(temp_imgs_list):
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in
                              imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                    temp_imgs_list[i] = imgs
                [imgs_sr, imgs_sf, imgs_tr, imgs_tf] = temp_imgs_list

            # Forward
            with amp.autocast(enabled=cuda):

                # [branch 1] for model_student, with using labels
                model_student.zero_grad()
                pred_sr = model_student(imgs_sr)  # forward
                # 生成 targets_sr 并添加置信度
                # 确保 targets_sr 在 GPU 上
                #targets_sr = targets_sr.to(device)
                # targets_sr = torch.cat([targets_sr, torch.ones((targets_sr.shape[0], 1), device=targets_sr.device)], dim=1)  # 添加置信度
                loss_sr, loss_items_sr = compute_loss(pred_sr, targets_sr.to(device))  # loss scaled by batch_size

                # [branch 2] for model_student, with using labels
                pred_sf = model_student(imgs_sf)  # forward
                loss_sf, loss_items_sf = compute_loss(pred_sf, targets_sr.to(device))  # loss scaled by batch_size

                # [branch 3] for model_teacher, without using labels
                model_teacher.eval()
                pred_tf, train_out = model_teacher(imgs_tf)  # forward. when eval(), the output is (x1, x2) in yolo.py
                
                
                # 计算水平线权重
                horizon_ys_tf = []  # 类似上述步骤计算
                for path in paths_tf:
                    orig_img = cv2.imread(str(path))
                    if orig_img is None:
                        orig_h = imgs_sf.shape[2]  # 预处理后的高度
                        horizon_y = orig_h // 2  # 默认居中
                    else:
                        orig_h = orig_img.shape[0]
                        horizon_y = detect_horizon(orig_img)
                    new_h = imgs_tr.shape[2]
                    adjusted_y = adjust_horizon_coord(orig_h, new_h, horizon_y)
                    #horizon_ys_tf.append(adjusted_y / new_h)  # 归一化
                    horizon_ys_tf.append(adjusted_y)  # 归一化
                horizon_ys_tf = torch.tensor(horizon_ys_tf, device=device)
                
                per_batch_size, channels, height, width = imgs_tf.shape
                
                #pred_tf = generate_pseudo_labels(pred_tf, horizon_ys_tf, imgs_tf.shape[2], per_batch_size)
                
                pred_tf_nms = non_max_suppression(pred_tf, conf_thres=conf_thres, iou_thres=iou_thres,
                                                  max_det=max_gt_boxes, multi_label=True,
                                                  agnostic=single_cls)  # pred_tf_nms type is list with batch_size length
                
                
                #weights_horizon = generate_pseudo_labels(pred_tf_nms, horizon_ys_tf, imgs_tf.shape[2])
                pred_tf_nms = generate_pseudo_labels(pred_tf_nms, horizon_ys_tf, imgs_tf.shape[2], per_batch_size, device)
                
                if pred_tf_nms is None:
                    pred_tf_nms = [torch.zeros((0,6), device=device) for _ in range(per_batch_size)]
                
                # [branch 4] for model_student, without using labels
                pred_tr = model_student(imgs_tr)
                
                # print(ni, len(pred_tr), pred_tr[0].shape, len(pred_tf_nms), pred_tf_nms[0].shape, pred_tf_nms[0], "\n",
                # imgs_tr.shape, targets_tr.shape, "\n", targets_tr, "\n", paths_tr)
                #per_batch_size, channels, height, width = imgs_tf.shape
                # print(type(targets_tr), targets_tr.shape, type(targets_tr[0]),
                # targets_tr[0].shape, targets_tr[0].cpu().numpy(), targets_tr[-1].cpu().numpy())
                # output: <class 'torch.Tensor'> torch.Size([49, 6]) <class 'torch.Tensor'>
                # torch.Size([6]) [0,6,0.91104,0.40758,0.13493,0.2201] [3,4,0.55561,0.9577,0.080059,0.084592]
                pred_labels_out_batch = []
                #y_centers_pseudo = [] #伪标签中心点

                for img_id in range(per_batch_size):
                    current_pred = pred_tf_nms[img_id]
    
                    # 维度修正
                    if current_pred.dim() == 1:
                        current_pred = current_pred.view(-1, 6)  # 转换为 [N,6]
                    labels_num = current_pred.shape[
                        0]  # pred_tf_nms prediction shape is (bs,n,6), per image [xyxy, conf, cls]
                    #labels_num = pred_tf_nms[img_id].shape[
                        #0]  # pred_tf_nms prediction shape is (bs,n,6), per image [xyxy, conf, cls]
                    #y_center_pseudo = (pred_tf_nms[img_id][:, 1]*imgs_tf.shape[2] + pred_tf_nms[img_id][:, 3]*imgs_tf.shape[2])/2
                    #y_centers_pseudo.append(y_center_pseudo)
                    if labels_num:
                        labels_list = torch.cat((pred_tf_nms[img_id][:, 5].unsqueeze(-1),
                                                 pred_tf_nms[img_id][:, 0:4]),
                                                dim=1)  # remove predicted conf, new format [cls x y x y]
                        labels_list[:, 1:5] = xyxy2xywhn(labels_list[:, 1:5], w=width,
                                                         h=height)  # xyxy to xywh normalized


                        pred_labels_out = torch.cat(((torch.ones(labels_num) * img_id).unsqueeze(-1).to(device),
                                                     labels_list),
                                                    dim=1)  # pred_labels_out shape is (labels_num, 6), per label format [img_id cls x y x y]

                        # else:
                        # pred_labels_out = pred_tf_nms[img_id]  # in this condition, pred_tf_nms[img_id] tensor size is [0,6]
                        '''[BUG] When training, nan can appear in batchnorm when all the values are the same, and thus std = 0'''
                        # pred_labels_out = torch.from_numpy(np.array([[img_id,0,0,0,0,0]])).to(device)
                        '''If no bboxes have been detected, we set a [0,0,w,h](xyxy) or [0.5,0.5,1,1](xywh) bounding-box for the image'''
                        # pred_labels_out = torch.from_numpy(np.array([[img_id,0,0.5,0.5,1,1]])).to(device)
                        pred_labels_out_batch.append(pred_labels_out)
                #weights_horizon = calculate_weights(y_centers_pseudo, pred_tf_nms, horizon_ys_tf, imgs_tf.shape[2])

                if len(pred_labels_out_batch) != 0:
                    pred_labels = torch.cat(pred_labels_out_batch, dim=0)
                else:
                    pred_labels = torch.from_numpy(np.array([[0, 0, 0.5, 0.5, 1, 1]])).to(device)
                pred_labels = torch.from_numpy(np.array([[0,0, 0.5, 0.5, random.uniform(0.2,0.8), random.uniform(0.2,0.8)]])).to(device)
                #pred_labels[:, 1] = torch.clamp(pred_labels[:, 1], 0, nc - 1)  # 限制类别在 [0, nc-1]
                '''
                horizon_ys_tr = []  # 类似上述步骤计算
                for path in paths_tr:
                    orig_img = cv2.imread(str(path))
                    if orig_img is None:
                        orig_h = imgs_sr.shape[2]  # 预处理后的高度
                        horizon_y = orig_h // 2  # 默认居中
                    else:
                        orig_h = orig_img.shape[0]
                        horizon_y = detect_horizon(orig_img)
                    new_h = imgs_tr.shape[2]
                    adjusted_y = adjust_horizon_coord(orig_h, new_h, horizon_y)
                    horizon_ys_tr.append(adjusted_y / new_h)  # 归一化
                '''


                loss_distillation, loss_items_distillation = compute_loss(pred_tr, pred_labels.to(device),
                                                                          horizon_ys=horizon_ys_tf)  # loss scaled by batch_size

                # consistency loss (source_real and source_fake should have similarly outputs)
                if opt.consistency_loss:
                    # loss_cons = torch.abs(loss_sr - loss_sf) * opt.alpha_weight  # L1 loss
                    loss_cons = torch.abs(loss_sr - loss_sf) ** 2 * opt.alpha_weight  # L2 loss

                # combine all losses
                loss = loss_sr + loss_sf + loss_distillation * lambda_weight
                loss_items = loss_items_sr + loss_items_sf + loss_items_distillation * lambda_weight
                if opt.consistency_loss:
                    loss += loss_cons
                    # print(loss_items.shape, loss_cons.shape)  # torch.Size([4]) torch.Size([1])
                    loss_items[3] += loss_cons.detach()[0]  # (lbox, lobj, lcls, total_loss)
                    loss_items = torch.cat((loss_items, loss_cons.detach()), 0)

                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            try:
                # Backward
                scaler.scale(loss).backward()
                # Optimizer
                if ni - last_opt_step >= accumulate:
                    scaler.step(student_optimizer)  # optimizer.step
                    scaler.update()
                    student_optimizer.zero_grad()
                    if ema:
                        ema.update(model_student)
                    last_opt_step = ni

                    model_teacher.zero_grad()
                    teacher_optimizer.step()
            except:
                # for possible bug when running scaler.scale(loss).backward()
                print("RuntimeError: Function 'CudnnConvolutionBackward0' returned nan values in its 1th output")
                print(targets_sr, "\n", paths_sr, "\n", targets_tr, "\n", paths_tr)
                print(pred_tf_nms, "\n", pred_labels, "\n", loss, "\n", loss_items)
                print(
                    "Currently, we have not been able to find the bug. Please resume training from the last running...")
                # continue

            # Print
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * (6 + all_shift)) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets_sr.shape[0], imgs_sr.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f'train_sr_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs_sr, targets_sr, paths_sr, f),
                           daemon=True).start()  # ./train_sr_batch[0,1,2].jpg
                    f = save_dir / f'train_sf_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs_sf, targets_sr, paths_sf, f),
                           daemon=True).start()  # ./train_sf_batch[0,1,2].jpg
                    f = save_dir / f'train_tr_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs_tr, targets_tr, paths_tr, f),
                           daemon=True).start()  # ./train_tr_batch[0,1,2].jpg
                    f = save_dir / f'train_tf_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs_tf, targets_tr, paths_tf, f),
                           daemon=True).start()  # ./train_tf_batch[0,1,2].jpg
                    if loggers['tb'] and ni == 0:  # TensorBoard
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')  # suppress jit trace warning
                            loggers['tb'].add_graph(
                                torch.jit.trace(de_parallel(model_student), imgs_sr[0:1], strict=False), [])
                elif plots and ni == 10 and loggers['wandb']:
                    wandb_logger.log({'Mosaics': [loggers['wandb'].Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in student_optimizer.param_groups]  # for loggers
        scheduler.step()

        # DDP process 0 or single-GPU
        if RANK in [-1, 0]:
            # mAP
            ema.update_attr(model_student, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            current_epoch = epoch + 1
            if not notest or final_epoch or (40 <= current_epoch <= 60):  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                
                # 创建当前epoch的保存目录
                if 1 <= current_epoch <= 100:
                    # current_save_dir = Path(save_dir) / f'epoch{current_epoch}_vis'
                    # current_save_dir.mkdir(parents=True, exist_ok=True)
                    current_save_dir = viewer_dir / 'visualizations' / f'epoch{current_epoch}_vis'  # 修改后的路径
                    current_save_dir.mkdir(parents=True, exist_ok=True)
                else:
                    current_save_dir = Path(save_dir)
                
                results, maps, _ = test.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz_test,
                                            model=ema.ema,
                                            single_cls=single_cls,
                                            dataloader=testloader,
                                            # save_dir=save_dir,
                                            save_dir=current_save_dir,
                                            save_json=is_coco and final_epoch,
                                            verbose=nc < 50 and final_epoch,
                                            plots=plots and final_epoch or (40 <= current_epoch <= 50),
                                            # ./test_batch[0,1,2]_labels.jpg ./test_batch[0,1,2]_pred.jpg  ...
                                            wandb_logger=wandb_logger,
                                            compute_loss=compute_loss)

            # Write
            with open(results_file, 'a') as f:
                # f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
                f.write(s + '%10.4g' * 8 % results + '\n')  # append metrics, val_loss  # Added in 2021-10-01
                # former 8 values ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size')
                # latter 8 values (P, R, mAP@.5, mAP@.75, mAP@.5-.95, val_loss(box, obj, cls))

            # Log
            # tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/total_loss',  # Added in 2022-04-04
                    # 'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.75',
                    'metrics/mAP_0.5:0.95',  # Added in 2021-10-01
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            if opt.consistency_loss: tags = tags[:4] + ['train/cons_loss'] + tags[4:]  # Added in 2022-04-04
            # for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
            for x, tag in zip(list(mloss) + list(results) + lr, tags):  # Changed in 2022-04-04
                if loggers['tb']:
                    loggers['tb'].add_scalar(tag, x, epoch)  # TensorBoard
                if loggers['wandb']:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            # fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi = fitness(np.array(results).reshape(1,
                                                   -1))  # weighted combination of [P, R, mAP@.5, mAP@.75, mAP@.5-.95]  # Added in 2021-10-01
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt_student = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),
                    'model': deepcopy(de_parallel(model_student)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': student_optimizer.state_dict(),
                    'wandb_id': wandb_logger.wandb_run.id if loggers['wandb'] else None}

                ckpt_teacher = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model_teacher)).half(),
                    'wandb_id': wandb_logger.wandb_run.id if loggers['wandb'] else None}

                # 按epoch保存学生和教师模型
                if (epoch + 1) % 10 == 0:
                    # epoch_student = wdir / f'epoch{epoch + 1}_student.pt'
                    # epoch_teacher = wdir / f'epoch{epoch + 1}_teacher.pt'
                    # torch.save(ckpt_student, epoch_student)
                    # torch.save(ckpt_teacher, epoch_teacher)
                    # 保存到viewer/weights目录
                    epoch_student = viewer_dir / 'weights' / f'epoch{epoch + 1}_student.pt'
                    epoch_teacher = viewer_dir / 'weights' / f'epoch{epoch + 1}_teacher.pt'
                    torch.save(ckpt_student, epoch_student)
                    torch.save(ckpt_teacher, epoch_teacher)

                # Save last, best and delete for model_student and model_teacher
                torch.save(ckpt_student, last_student)
                torch.save(ckpt_teacher, last_teacher)
                if best_fitness == fi:
                    torch.save(ckpt_student, best_student)
                    torch.save(ckpt_teacher, best_teacher)
                if loggers['wandb']:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt_student, ckpt_teacher

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    if RANK in [-1, 0]:
        logger.info(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png, confusion_matrix.png
            if loggers['wandb']:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [loggers['wandb'].Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})

        if not evolve:
            if is_coco:  # COCO dataset
                for m in [last_student, best_student] if best_student.exists() else [last_student]:  # speed, mAP tests
                    results, _, _ = test.run(data_dict,
                                             batch_size=batch_size // WORLD_SIZE * 2,
                                             imgsz=imgsz_test,
                                             conf_thres=0.001,
                                             iou_thres=0.7,
                                             model=attempt_load(m, device).half(),
                                             single_cls=single_cls,
                                             dataloader=testloader,
                                             save_dir=save_dir,
                                             save_json=True,
                                             plots=False)

            # Strip optimizers
            for f in last_student, best_student:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers, this will apparently reduce the model size
            if loggers['wandb']:  # Log the stripped model
                loggers['wandb'].log_artifact(str(best_student if best_student.exists() else last_student),
                                              type='model',
                                              name='run_' + wandb_logger.wandb_run.id + '_model',
                                              aliases=['latest', 'best', 'stripped'])
        wandb_logger.finish_run()

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', type=str, default='/workspace/SSDA-YOLO/runs/train/model_with_copypaste/weights/best_teacher.pt', help='initial weights path')
    parser.add_argument('--weights', type=str, default='/workspace/SSDA-YOLO/weights/yolov5s.pt',
                        help='initial weights path')
    parser.add_argument('--cfg', type=str, default='/workspace/SSDA-YOLO/data/models/yolov5s.yaml',
                        help='model.yaml path')
    parser.add_argument('--data', type=str, default='/workspace/SSDA-YOLO/data/yamls_sda/marine.yaml',
                        help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='/workspace/SSDA-YOLO/data/hyps/hyp.scratch.yaml',
                        help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters [removed]')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket [removed]')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='model_with_copypaste', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    parser.add_argument('--teacher_alpha', type=float, default=0.99, help='Teacher EMA alpha (decay) in UMT')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='Confidence threshold for pseudo label in UMT')
    parser.add_argument('--iou_thres', type=float, default=0.3,
                        help='Overlap threshold used for non-maximum suppression in UMT')
    parser.add_argument('--max_gt_boxes', type=int, default=20,
                        help='Maximal number of gt rois in an image during training in UMT')
    parser.add_argument('--lambda_weight', type=float, default=0.005, help='The weight for distillation loss in UMT')

    parser.add_argument('--consistency_loss', action='store_true',
                        help='Whether use the consistency loss (newly added)')
    parser.add_argument('--alpha_weight', type=float, default=2.0,
                        help='The weight for the consistency loss (newly added)')

    parser.add_argument('--student_weight', type=str, default='None',
                        help='Resuming weights path of student model in UMT')
    parser.add_argument('--teacher_weight', type=str, default='None',
                        help='Resuming weights path of teacher model in UMT')
    parser.add_argument('--save_dir', type=str, default='None', help='Resuming project path in UMT')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    set_logging(RANK)
    if RANK in [-1, 0]:
        print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
        check_git_status()
        check_requirements(exclude=['thop'])

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        if opt.save_dir == "None":
            opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        from datetime import timedelta
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        # dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=60))
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                                timeout=timedelta(seconds=600))  # fixed bug in 2021-07-13
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device)
        if WORLD_SIZE > 1 and RANK == 0:
            _ = [print('Destroying process group... ', end=''), dist.destroy_process_group(), print('Done.')]


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

