import copy
import logging
from typing import Dict, Optional

import numpy as np

from .modules import RandLANet

logger = logging.getLogger("early stopper")


class EarlyStopper:
    def __init__(self, patience: int, metric: str, mode: str = "max"):
        """Class to manage early stopping.
        :param patience: Patience (in epochs) for early stopping.
        :param metric: Metric to monitor for early stopping.
        :param mode: Maximizing or minimizing the metric?
        """
        self._patience = patience
        self._metric = metric
        self._mode = mode
        assert self._mode in ("max", "min"), "mode should be max or min!"
        self.reset()

    def reset(self):
        """Reset the counts, reference metric and remove best model weights."""

        self._count = 0
        self._best_model_weights = None
        if self._mode == "max":
            self._reference = -1
        else:
            self._reference = np.inf

    def check(self, metrics: Dict[str, float], model: RandLANet) -> bool:
        """Do the early stopping check. When a better metric is detected, save
        the weights. When no improvement is detected for x iterations, stop
        training.

        :param metrics: Dictionary with all metrics.
        :param model: Model currently being trained.
        :return: Boolean indicating to continue training.
        """

        if self._metric not in metrics.keys():
            logger.warning(f"Metric {self._metric} not known!")
            return True
        if self._mode == "max":
            improvement = metrics[self._metric] >= self._reference
        elif self._mode == "min":
            improvement = metrics[self._metric] <= self._reference
        else:
            logger.warning(f"Mode {self._mode} not known for early stopping!")
            improvement = True
        if improvement:
            self._count = 0
            self._reference = metrics[self._metric]
            self._best_model_weights = copy.deepcopy(model.state_dict())
        else:
            self._count += 1
            logger.info(
                f"No improvement in metric {self._metric} "
                f"({self._reference:.3f}) detected for "
                f"{self._count}/{self._patience} epochs."
            )
        continue_training: bool = self._count < self._patience
        if not continue_training:
            logger.info(
                f"Stopping training as no improvement in {self._metric} was "
                f"detected for {self._patience} consecutive test runs."
            )
        return continue_training

    def load_best_model_weights(self, model: RandLANet) -> Optional[RandLANet]:
        """Load given model with best detected model weights.

        :param model: Model to load weights.
        :return: Model loaded with weights or None when no best model
                 is available.
        """

        if self._best_model_weights is None:
            return None
        model.load_state_dict(self._best_model_weights)
        logger.info(
            f"Returning model with {self._metric}: {self._reference:.3f}"
        )
        return model
