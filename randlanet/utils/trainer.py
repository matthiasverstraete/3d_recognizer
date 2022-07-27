import logging
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .early_stopper import EarlyStopper
from .losses import FocalLoss, FocalTverskyLoss
from .metrics import MetricCollector, MetricCollectorBag, accuracy, iou
from .modules import RandLANet, UpSampler

logger = logging.getLogger("trainer")
logger.setLevel(logging.DEBUG)


@dataclass
class TrainingSettings:
    #: Number of epochs to train
    epochs: int = 150
    #: Size of minibatches used during training
    batch_size: int = 8
    #: Base learning rate
    learning_rate: float = 1e-2
    #: Exponential decay for learning rate (per epoch).
    learning_rate_decay: float = 0.9
    #: Loss function to use
    #: ("cross_entropy", "focal", "dice", "tversky", "focal_tversky")
    loss_function: str = "dice"
    #: Early stopping
    early_stopping: bool = True
    #: Patience for early stopping
    early_stopping_patience: int = 20


class Trainer:
    def __init__(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        log_dir: Optional[Path] = None,
        class_names: Optional[List[str]] = None,
    ):
        """Class managing the training process.
        :param train_dataloader: Torch data loader for training data.
        :param validation_dataloader: Torch data loader for validation data.
        :param log_dir: Optional directory path to store logging data.
        :param class_names: Optional list with class names (for pretty logging).
        """

        self._train_dataloader = train_dataloader
        self._validation_dataloader = validation_dataloader
        self._log_dir = log_dir
        self._class_names = class_names

    def train(
        self,
        model: RandLANet,
        settings: TrainingSettings,
        callbacks: List[Callable[[int, Dict[str, float]], None]] = [],
    ) -> RandLANet:
        """Train a given model.
        :param model: Model to train.
        :param settings: Training settings to use.
        :param callbacks: List of callback functions that are evaluated every
                          epoch.
        :return: Trained model.
        """

        # define optimizer, scheduler, loss and early stopper
        device = model.device
        optimizer = torch.optim.Adam(
            model.parameters(), lr=settings.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=settings.learning_rate_decay
        )
        criterion = self._get_loss(settings.loss_function)
        patience = (
            settings.early_stopping_patience
            if settings.early_stopping
            else settings.epochs
        )
        early_stopper = EarlyStopper(patience=patience, metric="val_mIoU")

        # bring in train mode
        model.train()
        n_train = len(self._train_dataloader.dataset)  # type: ignore
        n_val = len(self._validation_dataloader.dataset)  # type: ignore
        logger.info(
            f"Training on {n_train} training samples and {n_val} "
            "validation samples."
        )
        writer: Optional[SummaryWriter] = None
        if self._log_dir is not None:
            writer = SummaryWriter(str(self._log_dir))
        for epoch in range(1, settings.epochs + 1):
            # metrics to collect
            train_metrics = MetricCollector(self._class_names)
            # iterate over batches
            for input, labels, _ in tqdm(
                self._train_dataloader, desc="Training", leave=False
            ):
                # move data to device
                input = input.to(device)
                labels = labels.to(device)
                # forward step: get logits & evaluate loss
                logits = model(input)
                loss = criterion(logits, labels)
                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # save loss and metrics
                overall_accuracy, per_class_accuracies = accuracy(
                    logits, labels
                )
                miou, per_class_ious = iou(logits, labels)
                train_metrics.push(
                    loss.cpu().item(),
                    overall_accuracy,
                    per_class_accuracies,
                    miou,
                    per_class_ious,
                )
            # update learning rate
            scheduler.step()
            # evaluate on validation set
            validation_metrics = Trainer.evaluate(
                model,
                self._validation_dataloader,
                class_names=self._class_names,
                loss_function=settings.loss_function,
            )
            # early stopping check
            metrics = train_metrics.as_dict()
            metrics.update(validation_metrics.as_dict("val"))
            continue_training = early_stopper.check(metrics, model)
            # log everything
            self._log(
                epoch,
                settings.epochs,
                optimizer,
                train_metrics.as_dict(),
                validation_metrics.as_dict(include_stdev=True),
                writer,
            )
            # run callbacks
            for callback in callbacks:
                callback(epoch, metrics)
            if not continue_training:
                break
        if writer is not None:
            writer.close()
        # save best model
        trained_model = early_stopper.load_best_model_weights(model)
        if trained_model is None:
            logger.warning("Model did not improve during training!")
            trained_model = model
        model.eval()
        trained_model.eval()
        return trained_model

    def _log(
        self,
        epoch: int,
        total_epochs: int,
        optimizer: torch.optim.Optimizer,
        train_metrics: OrderedDict,
        validation_metrics: OrderedDict,
        writer: Optional[SummaryWriter],
    ) -> None:
        """Log training data.
        :param epoch: Current epoch number.
        :param total_epochs: Total number of epochs.
        :param optimizer: Optimizer instance.
        :param train_metrics: Metrics on the training set.
        :param validation_metrics: Metrics on the validation set.
        :param writer: Optional tensorboard writer.
        """

        # console logging
        log_line = f"Epoch {epoch:3d}/{total_epochs:3d} - "
        for key in ["loss"]:
            log_line += "%s: %.4f - val_%s: %.4f (s: %.4f) - " % (
                key,
                train_metrics[key],
                key,
                validation_metrics[key][0],
                validation_metrics[key][1],
            )
        for key in ["mAcc", "mIoU"]:
            log_line += "%s: %.2f%% - val_%s: %.2f%% (s: %.2f%%) - " % (
                key,
                train_metrics[key] * 100,
                key,
                validation_metrics[key][0] * 100,
                validation_metrics[key][1] * 100,
            )
        logger.info(log_line[:-2])
        all_metrics = {
            "Training": train_metrics,
            "Validation": validation_metrics,
        }
        for mode, metrics in all_metrics.items():
            log_line = f"{'':15s} {mode + ' IoU:':16s}"
            keys = [k for k in metrics.keys() if k.endswith(" IoU")]
            for key in keys:
                log_line += key.split(" IoU")[0]
                metric = metrics[key]
                if isinstance(metric, tuple):
                    log_line += ": %5.2f%% (s: %5.2f%%)" % (
                        metric[0] * 100,
                        metric[1] * 100,
                    )
                elif isinstance(metric, float):
                    log_line += ": %5.2f%% %11s" % (metric * 100, "")
                if key != keys[-1]:
                    log_line += " - "
            logger.info(log_line)
        if writer is not None:
            # tensorboard logging
            writer.add_scalar(
                "Learning rate", optimizer.param_groups[0]["lr"], epoch
            )
            tb_data: Dict[str, Dict] = {
                "Train": train_metrics,
                "Validation": validation_metrics,
            }
            for mode, metric_set in tb_data.items():
                for key, metric in metric_set.items():
                    writer.add_scalar(
                        f"{key}/{mode}",
                        metric[0] if isinstance(metric, tuple) else metric,
                        epoch,
                    )

    @staticmethod
    def _get_loss(loss_function: str) -> torch.nn.Module:
        """Get torch Module loss function from string value.
        :param loss_function: One of ("cross_entropy", "focal", "dice",
                                      "tversky", "focal_tversky")
        :return: Torch Module representing the loss function
                 with standard parameters.
        """
        if loss_function == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        elif loss_function == "focal":
            return FocalLoss(gamma=2)
        elif loss_function == "dice":
            return FocalTverskyLoss(
                alpha=0.5, gamma=1.0, neglect_background=True
            )
        elif loss_function == "tversky":
            return FocalTverskyLoss(
                alpha=0.7, gamma=1.0, neglect_background=True
            )
        elif loss_function == "focal_tversky":
            return FocalTverskyLoss(
                alpha=0.7, gamma=(4.0 / 3.0), neglect_background=True
            )
        else:
            raise ValueError(f"Loss function {loss_function} not known!")

    @staticmethod
    def evaluate(
        model: RandLANet,
        data_loader: DataLoader,
        class_names: Optional[List[str]] = None,
        loss_function: str = "dice",
        postprocess: bool = False,
        n_evaluations: int = 10,
    ) -> MetricCollectorBag:
        """Evaluate model on data.
        :param model: Model to evaluate.
        :param data_loader: Loader with data to evaluate on.
        :param class_names: Optional list with class names (for pretty logging).
        :param loss_function: Loss function to use ("cross_entropy", "focal",
                              "dice", "tversky", "focal_tversky")
        :param postprocess: Include postprocessing (upsampling) for evaluation.
        :param n_evaluations: Number of evaluations to average over.
        :return: MetricCollectorBag instance collecting all metrics.
        """

        @contextmanager
        def eval_context(model: torch.nn.Module):
            # switch to eval mode and go back to original mode afterwards
            training_mode = model.training
            model.eval()
            yield
            model.train(training_mode)

        criterion = Trainer._get_loss(loss_function)
        device = model.device
        # use fixed seeds to have a consistent evaluation
        seeds = [100 * i for i in range(n_evaluations)]
        rnd_state = np.random.get_state()
        metric_collectors: List[MetricCollector] = []
        if postprocess:
            assert (
                data_loader.batch_size == 1
            ), "Batch size 1 required when evaluating with postprocessing!"
        upsampler = UpSampler("nni", device)
        with eval_context(model):
            with torch.no_grad():
                for eval_idx, seed in enumerate(seeds):
                    np.random.seed(seed)
                    evaluation_metrics = MetricCollector()
                    for input, labels, indices in tqdm(
                        data_loader, desc="Evaluation", leave=False
                    ):
                        # move data to device
                        input = input.to(device)
                        labels = labels.to(device)
                        # predict
                        logits = model(input)
                        # compute loss
                        loss = criterion(logits, labels).cpu().item()
                        # compute metrics
                        if postprocess:
                            (
                                input_upsampled,
                                labels_upsampled,
                                _,
                            ) = data_loader.dataset.__getitem__(  # type: ignore
                                int(indices[0]), preprocess=False
                            )
                            xyz_upsampled = input_upsampled[:, :3].unsqueeze(0)
                            labels_upsampled = labels_upsampled.unsqueeze(
                                0
                            ).to(device)
                            xyz = input[:, :, :3]
                            confidences_upsampled = upsampler(
                                torch.softmax(logits, dim=1).unsqueeze(-1),
                                xyz,
                                xyz_upsampled,
                            ).squeeze(-1)
                            overall_accuracy, per_class_accuracies = accuracy(
                                confidences_upsampled, labels_upsampled
                            )
                            miou, per_class_ious = iou(
                                confidences_upsampled, labels_upsampled
                            )
                        else:
                            overall_accuracy, per_class_accuracies = accuracy(
                                logits, labels
                            )
                            miou, per_class_ious = iou(logits, labels)
                        evaluation_metrics.push(
                            loss,
                            overall_accuracy,
                            per_class_accuracies,
                            miou,
                            per_class_ious,
                        )
                    metric_collectors.append(evaluation_metrics)
        metric_collector_bag = MetricCollectorBag(
            metric_collectors, class_names
        )
        np.random.set_state(rnd_state)
        return metric_collector_bag
