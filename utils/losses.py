import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def one_hot_encode(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input_tensor == i) * torch.ones_like(input_tensor)
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, target, one_hot):
        x_shape = list(target.shape)
        if (len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N * D, C, H, W]
            target = torch.transpose(target, 1, 2)
            target = torch.reshape(target, new_shape)

        if one_hot:
            target = self.one_hot_encode(target)
        assert inputs.shape == target.shape, 'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:, i, :, :], target[:, i, :, :])
            class_wise_dice.append(diceloss)
            loss += diceloss
        return loss / self.n_classes


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1) / \
         torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1,
                         keepdim=True) / torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2) ** 2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def one_hot_encode(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input_tensor == i) * torch.ones_like(input_tensor)
            tensor_list.append(tmp)
        output_tensor = torch.stack(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, target, one_hot=True):
        if one_hot:
            target = self.one_hot_encode(target)
        print(inputs.shape, target.shape)
        assert inputs.shape == target.shape, 'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:, i, :, :], target[:, i, :, :])
            class_wise_dice.append(diceloss)
            loss += diceloss
        return loss / self.n_classes


def align_loss(source_features, target_features, tau=0.1):
    sim_matrix = torch.matmul(source_features, target_features.T) / tau  # 这里面进行矩阵运算已经包含了余弦相似度
    exp_sim_matrix = torch.exp(sim_matrix)
    loss = -torch.mean(torch.log(exp_sim_matrix / exp_sim_matrix.sum(dim=1, keepdim=True)))
    return loss


def soft_similarity_minimization_flattened(unlabeled_features, labeled_features, temperature=1.0):
    """
    Calculates soft similarity minimization loss, flattening feature maps.
    """
    # Flatten the feature maps
    unlabeled_features = unlabeled_features.view(unlabeled_features.size(0), -1)
    labeled_features = labeled_features.view(labeled_features.size(0), -1)

    # Normalize and calculate loss (same as before)
    unlabeled_features = F.normalize(unlabeled_features, dim=1)
    labeled_features = F.normalize(labeled_features, dim=1)
    similarities = torch.mm(unlabeled_features, labeled_features.t()) / temperature
    probabilities = F.softmax(similarities, dim=1)
    log_probabilities = F.log_softmax(similarities, dim=1)
    loss = -torch.mean(torch.sum(probabilities * log_probabilities, dim=1))
    return loss


class ConfidenceBasedSelfTrainingLoss(nn.Module):
    """Self-training loss with confidence threshold."""

    def __init__(self, threshold: float):
        super(ConfidenceBasedSelfTrainingLoss, self).__init__()
        self.threshold = threshold
        self.criterion = CrossEntropyLoss()

    def forward(self, y, y_target):
        confidence, pseudo_labels = F.softmax(y_target.detach(), dim=1).max(dim=1)
        mask = (confidence > self.threshold).float()
        self_training_loss = (self.criterion(y, pseudo_labels) * mask).mean()

        return self_training_loss, mask, pseudo_labels


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


def consistency_loss(logits_s, logits_w, mask, name='ce', T=0.5, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()

    pseudo_label = torch.softmax(logits_w, dim=-1)
    _, max_idx = torch.max(pseudo_label, dim=-1)
    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask.float()  # 过滤不靠谱的标签
    else:
        pseudo_label = torch.softmax(logits_w / T, dim=-1)  # 把弱增强的预测当伪标签，对应到debias中就是另一个头作为伪标签
        masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask.float()
    return masked_loss.mean(), mask


def soft_similarity_minimization_averaged(unlabeled_features, labeled_features, temperature=1.0):
    """
    Calculates soft similarity minimization loss, averaging spatial dimensions.
    """
    # Average across spatial dimensions
    unlabeled_features = torch.mean(unlabeled_features, dim=(1, 2))
    labeled_features = torch.mean(labeled_features, dim=(1, 2))

    # Normalize features
    unlabeled_features = F.normalize(unlabeled_features, dim=1)
    labeled_features = F.normalize(labeled_features, dim=1)

    # Cosine similarity and loss calculation (same as before)
    similarities = torch.mm(unlabeled_features, labeled_features.t()) / temperature
    probabilities = F.softmax(similarities, dim=1)
    log_probabilities = F.log_softmax(similarities, dim=1)
    loss = -torch.mean(torch.sum(probabilities * log_probabilities, dim=1))

    return loss


def entropy_minmization(p):
    y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1,
                             keepdim=True)
    return ent_map


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
