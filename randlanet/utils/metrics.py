from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import torch


def accuracy(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[float, List[float]]:
    """Compute the overall accuracy, and per-class accuracies
    :param logits: Logits, i.e. network output before soft-max (B?, C, N)
    :param labels: Labels, each label indicates the class index (B?, N)
    :return: Overall accuracy and a list of per-class accuracies
    """

    # use -2 instead of 1 to enable arbitrary batch dimensions
    n_classes = logits.size(-2)
    predictions = torch.max(logits, dim=-2).indices
    accuracy_mask = predictions == labels
    overall_accuracy = accuracy_mask.float().mean().cpu().item()
    per_class_accuracies: List[float] = []
    for label in range(n_classes):
        label_mask = labels == label
        per_class_accuracy = (accuracy_mask & label_mask).float().sum()
        n_labels = label_mask.float().sum()
        if n_labels == 0:
            per_class_accuracy = (per_class_accuracy == 0).float()
        else:
            per_class_accuracy /= label_mask.float().sum()
        per_class_accuracies.append(per_class_accuracy.cpu().item())
    return overall_accuracy, per_class_accuracies


def iou(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[float, List[float]]:
    """Compute the mean IoU and the per-class IoUs
    :param logits: Logits, i.e. network output before soft-max (B?, C, N)
    :param labels: Labels, each label indicates the class index (B?, N)
    :return: Mean IoU and per-class IoUs
    """

    # use -2 instead of 1 to enable arbitrary batch dimensions
    n_classes = logits.size(-2)
    predictions = torch.max(logits, dim=-2).indices
    per_class_ious: List[float] = []
    for label in range(n_classes):
        label_mask = labels == label
        pred_mask = predictions == label
        intersection = (pred_mask & label_mask).float().sum()
        union = (pred_mask | label_mask).float().sum()
        if union == 0:
            iou = 1.0
        else:
            iou = (intersection / union).cpu().item()
        per_class_ious.append(iou)
    miou = np.nanmean(per_class_ious)
    return miou, per_class_ious


class MetricCollector:
    """Collects all metrics of an evaluation on a dataset."""

    def __init__(self, class_names: Optional[List[str]] = None):
        self._class_names = class_names
        self.reset()

    def reset(self):
        """Reset all metrics in the collector."""

        self._losses: List[float] = []
        self._overall_accuracies: List[float] = []
        self._per_class_accuracies: List[np.ndarray] = []
        self._mious: List[float] = []
        self._per_class_ious: List[np.ndarray] = []

    def push(
        self,
        loss: float,
        overall_accuracy: float,
        per_class_accuracies: List[float],
        miou: float,
        per_class_ious: List[float],
    ) -> None:
        """Push new metrics for one sample to the collector.
        :param loss: Loss function.
        :param overall_accuracy: Overall accuracy; what portion of points is
                                 classified correctly.
        :param per_class_accuracies: Point accuracy per class.
        :param miou: Mean Intersection-over-Union.
        :param per_class_ious: Intersection-over-Union per class.
        """

        self._losses.append(loss)
        self._overall_accuracies.append(overall_accuracy)
        self._per_class_accuracies.append(np.array(per_class_accuracies))
        self._mious.append(miou)
        self._per_class_ious.append(np.array(per_class_ious))

    def as_dict(self, tag: str = "") -> OrderedDict:
        """Convert averaged metrics to dictionary.
        :param tag: Tag to put as suffix to the dictionary keys.
        :return: Dictionary summarizing the averaged metrics.
        """

        prefix = "" if tag == "" else f"{tag}_"
        dct = OrderedDict(
            {
                f"{prefix}loss": self.loss,
                f"{prefix}OA": self.overall_accuracy,
                f"{prefix}mAcc": self.mean_class_accuracy,
                f"{prefix}mIoU": self.miou,
            }
        )
        for class_idx, iou in enumerate(self.per_class_ious):
            key = (
                prefix + self._class_names[class_idx]
                if self._class_names
                else f"class {class_idx}"
            )
            key += " IoU"
            dct[key] = iou
        return dct

    @property
    def loss(self) -> float:
        """Returns average loss."""
        return float(np.mean(self._losses))

    @property
    def overall_accuracy(self) -> float:
        """Returns averaged overall accuracy."""
        return float(np.nanmean(self._overall_accuracies))

    @property
    def per_class_accuracies(self) -> List[float]:
        """Returns averaged per-class accuracies."""
        return list(np.nanmean(self._per_class_accuracies, axis=0))

    @property
    def mean_class_accuracy(self) -> float:
        """Returns mean class accuracy. The mean is taken over the different
        per-class accuracies.
        """
        return float(np.mean(self.per_class_accuracies))

    @property
    def miou(self) -> float:
        """Returns mean intersection over union."""
        return float(np.nanmean(self._mious))

    @property
    def per_class_ious(self) -> List[float]:
        """Returns averaged per-class ious."""
        return list(np.nanmean(self._per_class_ious, axis=0))


class MetricCollectorBag:
    """Bundles different MetricCollectors and reports averages and stdev results.
    Idea is to run multiple evaluations on RandLA-Net with different random
    seeds and report average results.
    """

    def __init__(
        self,
        metric_collectors: List[MetricCollector],
        class_names: Optional[List[str]] = None,
    ):
        self._class_names = class_names
        self._mcs = metric_collectors

    def as_dict(
        self, tag: str = "", include_stdev: bool = False
    ) -> OrderedDict:
        """Convert averaged metrics to dictionary.
        :param tag: Tag to put as suffix to the dictionary keys.
        :param include_stdev: Add stdev as part of a dict value.
        :return: Dictionary summarizing the averaged metrics.
        """

        prefix = "" if tag == "" else f"{tag}_"
        dct = OrderedDict(
            {
                f"{prefix}loss": self.loss,
                f"{prefix}OA": self.overall_accuracy,
                f"{prefix}mAcc": self.mean_class_accuracy,
                f"{prefix}mIoU": self.miou,
            }
        )
        for class_idx, iou in enumerate(self.per_class_ious):
            key = (
                prefix + self._class_names[class_idx]
                if self._class_names
                else f"class {class_idx}"
            )
            key += " IoU"
            dct[key] = iou
        if not include_stdev:
            dct_f = OrderedDict()
            for key in dct.keys():
                dct_f[key] = dct[key][0]
            return dct_f
        else:
            return dct

    @property
    def loss(self) -> Tuple[float, float]:
        """Returns mean and stdev loss."""
        losses = [mc.loss for mc in self._mcs]
        return np.mean(losses), np.std(losses)

    @property
    def overall_accuracy(self) -> Tuple[float, float]:
        """Returns mean and stdev overall accuracy."""
        overall_accuracies = [mc.overall_accuracy for mc in self._mcs]
        return np.mean(overall_accuracies), np.std(overall_accuracies)

    @property
    def mean_class_accuracy(self) -> Tuple[float, float]:
        """Returns mean and stdev mean class accuracy."""
        mean_class_accuracies = [mc.mean_class_accuracy for mc in self._mcs]
        return np.mean(mean_class_accuracies), np.std(mean_class_accuracies)

    @property
    def per_class_accuracies(self) -> List[Tuple[float, float]]:
        """Returns mean and stdev per class accuracies."""
        per_class_accuracies = [mc.per_class_accuracies for mc in self._mcs]
        if len(per_class_accuracies) == 0:
            return []
        n_classes = len(per_class_accuracies[0])
        pcas = [
            [pca[class_idx] for pca in per_class_accuracies]
            for class_idx in range(n_classes)
        ]
        return [(np.mean(pca), np.std(pca)) for pca in pcas]

    @property
    def miou(self) -> Tuple[float, float]:
        """Returns mean and stdev mean intersection over union."""
        mious = [mc.miou for mc in self._mcs]
        return np.mean(mious), np.std(mious)

    @property
    def per_class_ious(self) -> List[Tuple[float, float]]:
        """Returns mean and stdev per-class intersection over union."""
        per_class_ious = [mc.per_class_ious for mc in self._mcs]
        if len(per_class_ious) == 0:
            return []
        n_classes = len(per_class_ious[0])
        pcious = [
            [pciou[class_idx] for pciou in per_class_ious]
            for class_idx in range(n_classes)
        ]
        return [(np.mean(pciou), np.std(pciou)) for pciou in pcious]
