import torch
import torch.nn.functional as F

eps = 1e-7


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2):
        """Focal loss implementation.
        :param gamma: Exponent of focal factor. Increase to give more weight to
                      lower represented classes.
        """

        super(FocalLoss, self).__init__()
        self._gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """Computation of the focal loss.
        :param logits: Logits, i.e. network output before soft-max (B, C, N)
        :param labels: Labels, each label indicates the class index (B, N)
        :return: The loss.
        """

        B, C, N = logits.size()
        # one-hot encoding of labels
        y_true = torch.eye(C, device=labels.device)[labels].transpose(-1, -2)
        y_true = y_true.clamp(eps, 1.0 - eps)
        # soft-maxing the model output
        y_pred = F.softmax(logits, dim=-2)
        y_pred = y_pred.clamp(eps, 1.0 - eps)

        cross_entropy = -y_true * torch.log(y_pred)
        focal_loss = cross_entropy * (1 - y_pred) ** self._gamma
        return focal_loss.sum() / (B * N)


class FocalTverskyLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.7,
        gamma: float = 4.0 / 3.0,
        neglect_background: bool = True,
    ):
        """Focal Tversky loss implementation. This is generalization of the
        (i) Dice loss when alpha=0.5 and gamma=1 and the (ii) regular Tversky
        loss when gamma=1.
        :param alpha: Tversky factor, increase to give more weight to false
                      negatives.
        :param gamma: Exponent of focal factor. Increase to give more weight to
                      lower represented classes.
        :param neglect_background: Boolean indicating to exclude background
                                   (unlabeled) class from loss function.
        """
        super(FocalTverskyLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._neglect_background = neglect_background

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """Computation of the focal Tversky loss.
        :param logits: Logits, i.e. network output before soft-max (B, C, N)
        :param labels: Labels, each label indicates the class index (B, N)
        :return: The loss.
        """

        C = logits.size(-2)
        # one-hot encoding of labels
        y_true = torch.eye(C, device=labels.device)[labels].transpose(-1, -2)
        # soft-maxing the model output
        y_pred = F.softmax(logits, dim=-2)
        y_true = y_true.permute(1, 0, 2).flatten(1)  # C x rest
        y_pred = y_pred.permute(1, 0, 2).flatten(1)  # C x rest
        if self._neglect_background:
            y_true = y_true[1:, :]
            y_pred = y_pred[1:, :]

        true_pos = torch.sum(y_true * y_pred, dim=1)
        false_neg = torch.sum(y_true * (1 - y_pred), dim=1)
        false_pos = torch.sum((1 - y_true) * y_pred, dim=1)
        tversky_index = (true_pos + eps) / (
            true_pos
            + self._alpha * false_neg
            + (1 - self._alpha) * false_pos
            + eps
        )
        focal_tversky_loss = (1 - tversky_index) ** self._gamma
        return focal_tversky_loss.mean()
