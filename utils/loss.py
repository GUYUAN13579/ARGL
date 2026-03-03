# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False

        # 新增水平线参数初始化
        self.horizon_sigma = 1  # 方差平滑项
        self.horizon_gamma = 1  # 权重缩放系数
        self.base_sigma = 0  # 基础方差系数

        # 获取检测层
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]

        # 保存各层stride和anchor
        self.stride = det.stride  # 形状为[3]，例如tensor([ 8., 16., 32.])
        self.anchors = det.anchors  # 形状为[3,3,2]

        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, horizon_ys=None, path_idxs=None, weights_horizon=None):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # 新增：动态sigma调整
        #sigma_factor = 1.0 - min(self.hyp['epoch'] / self.hyp['warmup_epochs'], 1.0)
        sigma_factor = 0
        current_sigma = self.base_sigma * sigma_factor

        # 计算每个目标的垂直位置权重
        if horizon_ys is not None and targets.shape[0] > 0:
            img_h = p[0].shape[2] * self.stride[0]  # 原图高度
            # 权重计算使用indices中的网格坐标
            if indices:
                gj = indices[0][2]  # 取第一个检测层的网格y坐标
                cy = (gj.float() + 0.5) #/ p[0].shape[2]  # 归一化中心坐标
            #cy = targets[:, 3] * img_h  # 反归一化中心y坐标
            #cy_normalized = (indices[0][2].float() + 0.5) / p[0].shape[2]  # 第0层特征图高度
            #mu_batch = horizon_ys.mean()
            mu_batch = horizon_ys.float().mean()
            var_batch = torch.var(cy) if len(cy) > 1 else torch.tensor(0.1)

            # 高斯权重计算
            weights = torch.exp(-(cy - mu_batch) ** 2 / (2 * (var_batch + current_sigma)))
            if weights.numel() != 0:     
                weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-7)
            #weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-7)  # 标准化到[0,1]
            weights = weights * self.horizon_gamma + 0.3  # 缩放并加基底权重
        else:
            weights = torch.ones(len(targets), device=device)
        #print(weights)
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                '''
                # 在 loss.py 的 ComputeLoss.__call__ 方法中
                for i in range(self.nl):
                    if len(tcls[i]) > 0:
                        assert (tcls[i] < self.nc).all(), f"类别标签越界: 最大标签 {tcls[i].max()} >= nc={self.nc}"
                '''
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox = lbox + (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio
                # print("sort_id:",sort_id)
                # Classification

                # 边界框回归损失加权
                #iou_loss = 1.0 - iou
                #weighted_iou = (iou_loss * weights).sum() / (weights.sum() + 1e-7)
                #lbox += weighted_iou.mean()

                if self.nc > 1 :  # cls loss (only if multiple classes)
                    cls_loss = self.BCEcls(ps[:, 5:], t)
                    weighted_cls = (cls_loss * weights.view(-1, 1)).sum() / (weights.sum() + 1e-7)
                    lcls += weighted_cls.mean()
                    if self.nc == 1:
                        # 单类别处理: 目标全为positive (假设所有目标属于第0类)
                        t = torch.full_like(ps[:, 5:6], self.cn, device=device)
                        t[:, 0] = self.cp  # 正样本平滑
                        lcls = self.BCEcls(ps[:, 5:6], t)
                    else:
                        t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                        t[range(n), tcls[i]] = self.cp
                        lcls = lcls + self.BCEcls(ps[:, 5:], t)  # BCE
                    # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

                obji = self.BCEobj(pi[..., 4], tobj)
                lobj = lobj + obji * self.balance[i]  # obj loss
                '''
                # 目标置信度损失加权
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou
                obj_loss = self.BCEobj(pi[..., 4], tobj)
                weighted_obj = obj_loss * weights.mean()
                lobj += weighted_obj.mean()
                '''
                if self.autobalance:
                    self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
              
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox = lbox * self.hyp['box']
        lobj = lobj * self.hyp['obj']
        lcls = lcls * self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        # 强制转换为1D张量
        #lbox = lbox.view(-1) if lbox.dim() == 0 else lbox
        #lobj = lobj.view(-1) if lobj.dim() == 0 else lobj
        #lcls = lcls.view(-1) if lcls.dim() == 0 else lcls

        #loss = (lbox * weights.mean()) + lobj + lcls
        loss = lbox + lobj + lcls
        #loss = loss 
        #print(loss)
        return loss * bs, torch.cat((lbox, lobj, lcls, loss.unsqueeze(
            0) if loss.dim() == 0 else loss)).detach()  # torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        #gain = torch.ones(8, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        # 将 anchor 索引添加到 targets 的最后一列
        # targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # shape: [na, nt, 8]

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
        # 在返回信息中添加中心y坐标
        #gain = torch.ones(8, device=targets.device)  # 增加一列存储中心y坐标
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            #gain[7] = 1.0  # 新增第8列增益
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices

            # 保存索引、边界框、锚点和类别
            # a = t[:, 7].long()  # 锚点索引（第8列）

            # indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, gain[3].long() - 1),
                            gi.clamp_(0, gain[2].long() - 1)))  # image, anchor, grid indice
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            # 保存中心y坐标
            #cy = gxy[:, 1] / gain[3]  # 归一化y坐标
            #t = torch.cat((t, cy.unsqueeze(1)), dim=1)  # 添加第7列

        return tcls, tbox, indices, anch
