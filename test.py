# coding:utf-8
# ----------------------------------------------------------------------------
# Pytorch multi-GPU YOLOV5 based UMT
# Licensed under The MIT License [see LICENSE for details]
# Written by Huayi Zhou, based on code from
# https://github.com/kinredon/umt
# https://github.com/ultralytics/yolov5
# ----------------------------------------------------------------------------

"""Test a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/test.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import argparse
import json
import os
import sys
import torch
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from threading import Thread

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
# from utils.datasets import create_dataloader
from utils.datasets_single import create_dataloader_single
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr, \
    check_dataset_umt
import cv2
import numpy as np

# 在检测到水平线后添加坐标转换
def adjust_horizon_coord(orig_h, new_h, y):
    scale = new_h / orig_h
    return int(y * scale)

def detect_horizon(img):

    if img is None:
        return 0,0,0
    
    #if isinstance(img,Image.Image):
    #    #img=np.array(img)
   #     imgray=np.array(img.convert('L'))
    #else:#default:cv2
     #   if len(img.shape)==3:
            #imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      #  else:
       #     imgray=img
    
    if len(img.shape) == 3:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        imgray = img
    #imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    return (horizon_bottom + horizon_top) / 2 #返回水平线

'''
#检测水平线
def detect_horizon(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    horizon_y = image.shape[0]  # 默认底部
    y_values = []
    #
    if lines is not None:
        max_y = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(np.arctan2(y2-y1, x2-x1)) < np.deg2rad(10):
                current_max = max(y1, y2)
                if current_max > max_y:
                    max_y = current_max
        if max_y > 0:
            horizon_y = max_y
    #
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算直线的斜率
            if x2 - x1 != 0:  # 避免除以零
                slope = (y2 - y1) / (x2 - x1)
                # 判断是否为水平线
                if abs(slope) < 0.5:  # 斜率接近于0，表示水平线
                    # 绘制检测到的水平线
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 存储 y 坐标的均值
                    y_values.append((y1 + y2) / 2)
    if y_values:
        horizon_y = int(np.mean(y_values))
    
    return horizon_y
'''

@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        # task='val',  # train, val, test, speed or study
        task='test_target_real',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a cocoapi-compatible JSON results file
        project='runs/test',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        wandb_logger=None,
        compute_loss=None,
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # Data
        with open(data) as f:
            data = yaml.safe_load(f)
        # check_dataset(data)  # check
        check_dataset_umt(data)  # check, need to be re-write or command out

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    # is_coco = type(data['val']) is str and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    is_coco = type(data['test_target_real']) is str and data['test_target_real'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        # task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        task = task if task in ('train_source_real', 'train_source_fake', 'train_target_real', 
            'train_target_fake', 'test_target_real') else 'test_target_real'
        dataloader = create_dataloader_single(data[task], imgsz, batch_size, gs, single_cls, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    
    # s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    # p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    s = ('%20s' + '%11s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.75', 'mAP@.5:.95')  # Added in 2021-10-01
    p, r, f1, mp, mr, map50, map75, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.  # Added in 2021-10-01
    
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    
    # 新增CSV文件路径
    csv_path = save_dir / 'batch_metrics.csv'
    
    # 初始化CSV文件并写入标题
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Batch', 'Precision', 'Recall', 'mAP_50', 'mAP_75', 'mAP'])
        
    prev_stats_len = 0  # 用于跟踪统计信息位置

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        # 在数据加载后显式转移数据到 GPU
        img = img.to(device, non_blocking=True).float()  # 明确指定类型和位置
        # 检测每个图像的水平线
        horizon_ys = []
        path_idxs=[]
        for path_idx, path in enumerate(paths):
            # 读取原始图像
            orig_img = cv2.imread(str(path))
            if orig_img is None:
                orig_h = 640  # 默认尺寸需与预处理尺寸一致
                horizon_y = orig_h // 2  # 默认水平线居中
            else:
                orig_h = orig_img.shape[0]
                # 检测水平线
                horizon_y = detect_horizon(orig_img)

            # 获取预处理后的尺寸（假设shapes格式为[[h,w], [h,w], ...]）
            new_h = shapes[path_idx][0][0]  # 嵌套列表时需双重索引  # 当前图像的预处理后高度
            # 坐标转换
            adjusted_y = adjust_horizon_coord(orig_h, new_h, horizon_y)
            horizon_ys.append(adjusted_y)
            path_idxs.append(path_idx)
        #device = img.device  # 获取当前批次所在的设备（如 cuda:0）
        #horizon_ys = torch.tensor(horizon_ys, dtype=torch.float32, device=device)
        #print("horizon_ys:", horizon_ys)
        '''
        # 检测每个图像的水平线
        horizon_ys = []
        for path in paths:
            image = cv2.imread(str(path))  # 读取原始图像
            if image is None:
                horizon_y = image.shape[0]  # 默认底部
            else:
                horizon_y = detect_horizon(image)
            horizon_ys.append(horizon_y)
        horizon_ys = torch.tensor(horizon_ys, device=device)
        '''
        #可视化
        # 转换为归一化坐标（相对于图像高度）
        img_heights = [cv2.imread(str(path)).shape[0] for path in paths]
        horizon_norm = [y / h for y, h in zip(horizon_ys, img_heights)]
        
        
        t_ = time_synchronized()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        #targets = torch.cat([targets, torch.ones((targets.shape[0], 1), device=targets.device)], dim=1)  # 添加置信度列
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t = time_synchronized()
        t0 += t - t_

        # Run model
        device = select_device('0')  # 使用 GPU 0
        model = model.to(device)
        out, train_out = model(img, augment=augment)  # inference and training outputs
        t1 += time_synchronized() - t
        
        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets, horizon_ys=None, path_idxs=None)[1][:3]  # box, obj, cls

        # Run NMS
        #targets = targets[:, :6]  # 保留前6列，去除第7列
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time_synchronized()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        t2 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
            
            
            '''   
        # 在批次处理结束后添加以下代码 -------------------------
        current_stats_len = len(stats)
        batch_stats = stats[prev_stats_len:current_stats_len]
        prev_stats_len = current_stats_len

        # 计算当前批次指标
        if len(batch_stats) > 0:
            # 提取当前批次的统计信息
            batch_correct = torch.cat([s[0] for s in batch_stats]).cpu().numpy()
            batch_conf = torch.cat([s[1] for s in batch_stats]).cpu().numpy()
            batch_pcls = torch.cat([s[2] for s in batch_stats]).cpu().numpy()
            batch_tcls = np.concatenate([s[3] for s in batch_stats], axis=0)

            if len(batch_correct) > 0:
                # 计算精度指标
                batch_p, batch_r, batch_ap, _, _ = ap_per_class(
                    batch_correct, 
                    batch_conf,
                    batch_pcls,
                    batch_tcls,
                    plot=False
                )
                # 计算平均指标
                batch_p_mean = batch_p.mean()
                batch_r_mean = batch_r.mean()
                map50 = batch_ap[:, 0].mean()
                map75 = batch_ap[:, batch_ap.shape[1]//2].mean()
                map = batch_ap.mean()
            else:
                batch_p_mean = batch_r_mean = map50 = map75 = map = 0.0
        else:
            batch_p_mean = batch_r_mean = map50 = map75 = map = 0.0

        # 写入CSV文件
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                batch_i,
                round(float(batch_p_mean), 4),
                round(float(batch_r_mean), 4),
                round(float(map50), 4),
                round(float(map75), 4),
                round(float(map), 4)
            ])
            ''' 
            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()
          
        # 可视化预测结果时传递水平线参数
        if plots and batch_i < 3:  # 可视化前3个批次
            plot_images(img, output_to_target(out), paths, 
                       save_dir/f'batch{batch_i}_pred.jpg', 
                       names, horizon_lines=horizon_norm)
       

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)  # will plot and save (F1,PR,P,R)_curve.png
        
        ap75 = ap[:, ap.shape[1]//2]  # Added in 2021-10-01, ap.shape is [len(classes), 101]
        
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        
        # mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()  # Added in 2021-10-01
        
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    # pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    # print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 5  # print format  # Added in 2021-10-01
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map75, map))  # Added in 2021-10-01

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            # print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap75[i], ap[i]))  # Added in 2021-10-01

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            # map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            map, map50, map75 = eval.stats[:3]  # update results (mAP@0.5:0.95, mAP@0.5)  # Added in 2021-10-01
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    # return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
    return (mp, mr, map50, map75, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t  # Added in 2021-10-01


def parse_opt():
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--data', type=str, default='/workspace/SSDA-YOLO/data/yamls_sda/marine.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/copy+horizon/viewer/weights/epoch50_teacher.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    opt.data = check_file(opt.data)  # check file
    return opt


def main(opt):
    set_logging()
    print(colorstr('test: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                (f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
