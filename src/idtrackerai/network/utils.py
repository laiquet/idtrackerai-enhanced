import torch
from scipy.optimize import linear_sum_assignment as hungarian
from sklearn.metrics.cluster import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from torch import nn


def Class2Simi(x: torch.Tensor, mode="cls", mask=None):
    """
    Give a 1d torch tensor with classes in dense format, returns the pairwise similarity matrix liniarized. A mask can
    be applied to discard some elements of the similarity matrix.

    :param x: 1d torch tensor with classes in dense format
    :param mode: 'cls' for classification 'hinge' for clustering
    :param mask: 2d torch tensor with the mask to be applied to the pairwise similarity matrix
    :return: 1d torch tensor with the elements to be considered
    """
    # Convert class label to pairwise similarity
    n = x.nelement()
    assert (n - x.ndimension() + 1) == n, "Dimension of Label is not right"
    expand1 = x.view(-1, 1).expand(n, n)
    expand2 = x.view(1, -1).expand(n, n)
    out = expand1 - expand2
    out[out != 0] = -1  # dissimilar pair: label=-1
    out[out == 0] = 1  # Similar pair: label=1
    if mode == "cls":
        out[out == -1] = 0  # dissimilar pair: label=0
    if mode == "hinge":
        out = out.float()  # hingeloss require float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out


def weights_xavier_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight.data)


def fc_weights_reinit(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


class Normalize:
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    # TODO: This is kind of a batch normalization but not trained. Explore using real BN in idCNN.

    def __init__(self, inplace=False):
        self.inplace = inplace  # TODO is inplace used?

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        mean = torch.tensor([tensor.mean()])
        std = torch.tensor([tensor.std()])
        return tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        # return F.normalize(tensor, tensor.mean(), tensor.std(), self.inplace)


def prepare_task_target(target, args, mask=None):
    # Prepare the target for different criterion/tasks
    if args.loss == "CE":  # For standard classification
        if "semi" in args.dataset:
            one_hot_targets = target[:, :-1].reshape(-1)
            pairwise_targets = Class2Simi(target[:, -1], mode="hinge", mask=mask)
            train_target = torch.cat((one_hot_targets, pairwise_targets), 0)
            eval_target = target[:, -1]
        else:
            train_target = eval_target = target
    elif args.loss == "MCL":  # For clustering
        if "semi" in args.dataset:
            one_hot_targets = target[:, :-1].reshape(-1)
            pairwise_targets = Class2Simi(target[:, -1], mode="hinge", mask=mask)
            train_target = torch.cat((one_hot_targets, pairwise_targets), 0)
            eval_target = target[:, -1]
        else:
            train_target = Class2Simi(target, mode="hinge", mask=mask)
            eval_target = target
    elif args.loss in ("CEMCL", "CEMCL_weighted"):  # For semi-supervised clustering
        one_hot_targets = target[:, :-1].reshape(-1)
        pairwise_targets = Class2Simi(target[:, -1], mode="hinge", mask=mask)
        train_target = torch.cat((one_hot_targets, pairwise_targets), 0)
        eval_target = target[:, -1]
    return train_target, eval_target


class Confusion:
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index
    """

    def __init__(self, k, normalized=False):
        self.k = k
        self.conf = torch.LongTensor(k, k)
        self.normalized = normalized
        self.reset()

    def reset(self):
        self.conf.fill_(0)
        self.gt_n_cluster = None

    def cuda(self):
        self.conf = self.conf.cuda()

    def add(self, output, target):
        if target.size(0) > 1:
            output = output.squeeze_()
            target = target.squeeze_()
        assert output.size(0) == target.size(
            0
        ), "number of targets and outputs do not match"
        if output.ndimension() > 1:  # it is the raw probabilities over classes
            assert output.size(1) == self.conf.size(
                0
            ), "number of outputs does not match size of confusion matrix"

            _, pred = output.max(1)  # find the predicted class
        else:  # it is already the predicted class
            pred = output
        indices = (
            target * self.conf.stride(0) + pred.squeeze_().type_as(target)
        ).type_as(self.conf)
        ones = torch.ones(1).type_as(self.conf).expand(indices.size(0))
        self._conf_flat = self.conf.view(-1)
        self._conf_flat.index_add_(0, indices, ones)

    def classIoU(self, ignore_last=False):
        confusion_tensor = self.conf
        if ignore_last:
            confusion_tensor = self.conf.narrow(0, 0, self.k - 1).narrow(
                1, 0, self.k - 1
            )
        union = (
            confusion_tensor.sum(0).view(-1)
            + confusion_tensor.sum(1).view(-1)
            - confusion_tensor.diag().view(-1)
        )
        acc = confusion_tensor.diag().float().view(-1).div(union.float() + 1)
        return acc

    def recall(self, clsId):
        i = clsId
        TP = self.conf[i, i].sum().item()
        TPuFN = self.conf[i, :].sum().item()
        if TPuFN == 0:
            return 0
        return float(TP) / TPuFN

    def precision(self, clsId):
        i = clsId
        TP = self.conf[i, i].sum().item()
        TPuFP = self.conf[:, i].sum().item()
        if TPuFP == 0:
            return 0
        return float(TP) / TPuFP

    def f1score(self, clsId):
        r = self.recall(clsId)
        p = self.precision(clsId)
        if (p + r) == 0:
            return 0
        return 2 * float(p * r) / (p + r)

    def acc(self):
        TP = self.conf.diag().sum().item()
        total = self.conf.sum().item()
        if total == 0:
            return 0.0
        return float(TP) / total

    def optimal_assignment(self, gt_n_cluster=None, assign=None):
        if assign is None:
            mat = -self.conf.cpu().numpy()  # hungaian finds the minimum cost
            r, assign = hungarian(mat)
        self.conf = self.conf[:, assign]
        self.gt_n_cluster = gt_n_cluster
        return assign

    def show(self, width=6, row_labels=None, column_labels=None):
        print("Confusion Matrix:")
        conf = self.conf
        rows = self.gt_n_cluster or conf.size(0)
        cols = conf.size(1)
        if column_labels is not None:
            print(("%" + str(width) + "s") % "", end="")
            for c in column_labels:
                print(("%" + str(width) + "s") % c, end="")
            print("")
        for i in range(0, rows):
            if row_labels is not None:
                print(("%" + str(width) + "s|") % row_labels[i], end="")
            for j in range(0, cols):
                print(("%" + str(width) + ".d") % conf[i, j], end="")
            print("")

    def conf2label(self):
        conf = self.conf
        gt_classes_count = conf.sum(1).squeeze()
        n_sample = gt_classes_count.sum().item()
        gt_label = torch.zeros(n_sample)
        pred_label = torch.zeros(n_sample)
        cur_idx = 0
        for c in range(conf.size(0)):
            if gt_classes_count[c] > 0:
                gt_label[cur_idx : cur_idx + gt_classes_count[c]].fill_(c)
            for p in range(conf.size(1)):
                if conf[c][p] > 0:
                    pred_label[cur_idx : cur_idx + conf[c][p]].fill_(p)
                cur_idx = cur_idx + conf[c][p]
        return gt_label, pred_label

    def clusterscores(self):
        target, pred = self.conf2label()
        NMI = normalized_mutual_info_score(target, pred, average_method="arithmetic")
        ARI = adjusted_rand_score(target, pred)
        AMI = adjusted_mutual_info_score(target, pred, average_method="arithmetic")
        return {"NMI": NMI, "ARI": ARI, "AMI": AMI}
