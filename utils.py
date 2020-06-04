import os
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import morphology

class AvgMeter(object):
    """
    Acc meter class, use the update to add the current acc 
    and self.avg to get the avg acc 
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dice_coeff(seg, gt, ratio=0.5):
    """
    function to calculate the dice score
    """
    seg = seg.flatten()
    gt = gt.flatten()
    seg[seg > ratio] = np.float32(1)
    seg[seg < ratio] = np.float32(0)
    dice = float(2 * (gt * seg).sum())/float(gt.sum() + seg.sum())
    return dice


def check_dir(path):              # if folder does not exist, create it
    if not os.path.exists(path):
        os.mkdir(path)


# class DiceLoss(nn.Module):
#     def __init__(self, epsilon=1e-5):
#         super(DiceLoss, self).__init__()
#         # smooth factor
#         self.epsilon = epsilon
#
#     def forward(self, targets, logits):
#         batch_size = targets.size(0)
#         # log_prob = torch.sigmoid(logits)
#         logits = logits.view(batch_size, -1).type(torch.FloatTensor)
#         targets = targets.view(batch_size, -1).type(torch.FloatTensor)
#         intersection = (logits * targets).sum(-1)
#         dice_score = 2. * intersection / ((logits + targets).sum(-1) + self.epsilon)
#         # dice_score = 1 - dice_score.sum() / batch_size
#         return torch.mean(1. - dice_score)


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

def rel_abs_vol_diff(y_true, y_pred):
    return np.abs((y_pred.sum() / y_true.sum() - 1) * 100)


def get_boundary(data, img_dim=2, shift=-1):
    data = data > 0
    edge = np.zeros_like(data)
    for nn in range(img_dim):
        edge += ~(data ^ np.roll(~data, shift=shift, axis=nn))
    return edge.astype(int)


def surface_dist(input1, input2, sampling=1, connectivity=1):
    input1 = np.squeeze(input1)
    input2 = np.squeeze(input2)

    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    ##    S = input_1 - morphology.binary_erosion(input_1, conn)
    ##    Sprime = input_2 - morphology.binary_erosion(input_2, conn)

    S = np.bitwise_xor(input_1, morphology.binary_erosion(input_1, conn))
    Sprime = np.bitwise_xor(input_2, morphology.binary_erosion(input_2, conn))

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return sds

if __name__ == "__main__":
    import numpy as np
    yt = np.random.random(size=(2, 1, 3, 3, 3))
    # print(yt)
    yt = torch.from_numpy(yt)
    yp = np.zeros(shape=(2, 1, 3, 3, 3))
    yp = yp + 1
    yp = torch.from_numpy(yp)
    # print(yp)
    dl = BinaryDiceLoss()
    print(dl(yp, yt).item())
