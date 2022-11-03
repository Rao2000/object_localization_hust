import torch
import torch.nn as nn
from metric import bbox_iou


# class LossFunctionClass:

#     def __init__(self, model):
#         self.cls_loss = nn.CrossEntropyLoss()
#         self.box_loss = nn.MSELoss()
#         self.obj_loss = nn.BCEWithLogitsLoss()
#         self.anchors = model.anchors
#         self.num_anchors = model.num_anchors


#     def __call__(self, output, target):
#         # output: [bs, num_anchors, (x, y, w, h, conf, labels)]

#         # l_cls = self.cls_loss(output[:, 5:], target[:, 0].long())
#         # pxy = output[:, 0:2].sigmoid()
#         # pwh = output[:, 2:4].sigmoid() ** 2
#         # pbox = torch.cat((pxy, pwh), dim=1)
#         # l_box = self.box_loss(pbox, target[:, 1:])
#         # iou = bbox_iou(pbox.T, target, x1y1x2y2=False, CIoU=True).detach().clamp(0)
#         # loss = l_cls + l_box
#         target = target.repeat(self.num_anchors, 1, 1).permute(1, 0, 2)  # [bs, num_anchors, 5]
#         pxy = output[..., 0:2].sigmoid()
#         pwh = output[..., 2:4].sigmoid() ** 2 * self.anchors
#         pbox = torch.cat((pxy, pwh), dim=-1)
#         tbox = target[..., 1:]
#         box_iou = bbox_iou(pbox.T, tbox, x1y1x2y2=False, CIoU=True).detach().clamp(0).T
#         box_iou = torch.where(box_iou > 0.5, 1.0, 0.0)
#         l_obj = self.obj_loss(output[..., 4], box_iou)
#         l_cls = self.cls_loss(output[..., 5:].permute(0, 2, 1), target[..., 0].long())
#         l_box = self.box_loss(pbox, tbox)
#         loss = l_obj + l_cls + l_box

#         return loss, torch.stack((l_obj, l_box, l_cls)).detach().cpu()

class LossFunctionClass:

    def __init__(self, model):
        self.cls_loss = nn.CrossEntropyLoss()
        self.box_loss = nn.MSELoss()
        self.obj_loss = nn.BCEWithLogitsLoss()
        self.anchors = model.anchors
        self.num_anchors = model.num_anchors


    def __call__(self, output, target):
        # output: [bs, num_anchors, ny, nx, (x, y, w, h, conf, labels)]

        # l_cls = self.cls_loss(output[:, 5:], target[:, 0].long())
        # pxy = output[:, 0:2].sigmoid()
        # pwh = output[:, 2:4].sigmoid() ** 2
        # pbox = torch.cat((pxy, pwh), dim=1)
        # l_box = self.box_loss(pbox, target[:, 1:])
        # iou = bbox_iou(pbox.T, target, x1y1x2y2=False, CIoU=True).detach().clamp(0)
        # loss = l_cls + l_box
        g = 0.5  # bias
        off = torch.tensor([[0, 0]], device=target.device).float() * g  # offsets
        target = torch.cat((torch.arange(output.shape[0], device=output.device).view(-1, 1), target), 1)
        anchor_idx = torch.arange(self.num_anchors, device=target.device).float().view(self.num_anchors, 1).repeat(1, target.shape[0])
        target = torch.cat((target.repeat(self.num_anchors, 1, 1), anchor_idx[:, :, None]), 2)  # [num_anchors, bs, 6(img_idx, cls, x, y, w, h, anchor_idx)]
        gain = torch.ones(7, device=output.device)
        gain[2:6] = torch.tensor(output.shape)[[3, 2, 3, 2]]
        target = target * gain  # normalize (x, y, w, h) to grid scale
        r = target[:, :, 4:6] / self.anchors[:, None]  # wh ratio
        j = torch.max(r, 1. / r).max(2)[0] < 4.0  # compare
        # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
        t = target[j]  # filter
        # Offsets
        gxy = t[:, 2:4]  # grid xy
        gxi = gain[[2, 3]] - gxy  # inverse
        j, k = ((gxy % 1. < g) & (gxy > 1.)).T
        l, m = ((gxi % 1. < g) & (gxi > 1.)).T
        j = torch.stack((torch.ones_like(j),))
        t = t.repeat((off.shape[0], 1, 1))[j]
        offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

        b, c = t[:, :2].long().T
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]
        gij = (gxy - offsets).long()
        gi, gj = gij.T
        
        a = t[:, 6].long()  # anchor indices
        gj.clamp_(0, gain[3] - 1)
        gi.clamp_(0, gain[2] - 1)  # image, anchor, grid indices
        tbox = torch.cat((gxy - gij, gwh), 1)  # box
        anchors = self.anchors[a]  # anchors
        tcls = c  # class

        tobj = torch.zeros_like(output[..., 0], device=output.device)
        ps = output[b, a, gj, gi]  # prediction subset corresponding to targets

        # Regression
        pxy = ps[:, :2].sigmoid() * 2. - 0.5
        pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors
        pbox = torch.cat((pxy, pwh), 1)  # predicted box
        iou = bbox_iou(pbox.T, tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
        l_box = (1.0 - iou).mean()  # iou loss

        # Objectness
        tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)  # iou ratio
        l_obj = self.obj_loss(output[..., 4], tobj)

        # Classification
        l_cls = self.cls_loss(ps[:, 5:], tcls)  # BCE
        loss = l_obj + l_cls + l_box
        # pxy = output[..., 0:2].sigmoid()
        # pwh = output[..., 2:4].sigmoid() ** 2 * self.anchors
        # pbox = torch.cat((pxy, pwh), dim=-1)
        # tbox = target[..., 1:]
        # box_iou = bbox_iou(pbox.T, tbox, x1y1x2y2=False, CIoU=True).detach().clamp(0).T
        # box_iou = torch.where(box_iou > 0.5, 1.0, 0.0)
        # l_obj = self.obj_loss(output[..., 4], box_iou)
        # l_cls = self.cls_loss(output[..., 5:].permute(0, 2, 1), target[..., 0].long())
        # l_box = self.box_loss(pbox, tbox)
        # loss = l_obj + l_cls + l_box

        return loss, torch.stack((l_obj, l_box, l_cls)).detach().cpu()

