import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
import skimage
from scipy.ndimage import distance_transform_edt
###################################################################
# ########################## iou loss #############################
###################################################################
class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def _iou(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)

        return iou.mean()

    def forward(self, pred, target):
        return self._iou(pred, target)


class Thickness_Aware_Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(Thickness_Aware_Loss, self).__init__()
        self.eps = eps

    def estimate_thin_vessel_weight(self, lab):
        weight_maps = []
        for b in range(lab.shape[0]):
            mask = lab[b, 0].detach().cpu().numpy() > 0.5
            dist = distance_transform_edt(mask)
            skeleton = skimage.morphology.skeletonize(mask)
            thickness = dist * skeleton
            weight = 1.0 / (thickness + self.eps)
            weight = torch.tensor(weight).clamp(max=10.0)
            weight_maps.append(weight.unsqueeze(0).unsqueeze(0))
        weight_map = torch.cat(weight_maps, dim=0).to(lab.device)  # (B,1,H,W)
        return weight_map

    def smoothness_penalty(self, tensor, weight=None):
        grad_x = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
        grad_y = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
        if weight is not None:
            weight_x = weight[:, :, :, 1:] * weight[:, :, :, :-1]
            weight_y = weight[:, :, 1:, :] * weight[:, :, :-1, :]
            smoothness_loss = (grad_x * weight_x).mean() + (grad_y * weight_y).mean()
        else:
            smoothness_loss = grad_x.mean() + grad_y.mean()
        return smoothness_loss

    def forward(self, img, lab):
        kernel = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=img.dtype).cuda()

        img = K.morphology.opening(img, kernel)
        lab = K.morphology.opening(lab, kernel)

        img = torch.sigmoid(10 * (img - 0.5))
        lab = torch.sigmoid(10 * (lab - 0.5))

        weight_map = self.estimate_thin_vessel_weight(lab)
        smoothness_img = self.smoothness_penalty(img, weight_map)
        smoothness_lab = self.smoothness_penalty(lab, weight_map)

        C = torch.abs(smoothness_img - smoothness_lab) / (torch.count_nonzero(lab) + self.eps)
        return C
