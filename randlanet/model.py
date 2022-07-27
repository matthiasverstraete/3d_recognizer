import json
import logging
import os
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .utils.augmentation import AugmentationSettings
from .utils.dataset import get_data_loader
from .utils.modules import RandLANet, RandLANetSettings, UpSampler
from .utils.preprocessing import sample_points
from .utils.trainer import Trainer, TrainingSettings


class Model:
    """Wrapper around detectron2 model building that allow for easy weight saving,
    loading and predicting.
    """

    def __init__(
        self,
        settings: RandLANetSettings,
        weights: Optional[OrderedDict] = None,
        use_gpu: bool = True,
    ):
        """
        :param settings: Model settings defining RandLANet.
        :param weights: Optional weights to initialize the model with.
        :param use_gpu: Boolean indicating if GPU should be used when available.
        """

        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
        )
        self._model = RandLANet(settings, self.device)
        if weights is not None:
            self._model.load_state_dict(weights)
        self._model.eval()
        self._upsampler = UpSampler(settings.upsampling, self.device)

    def __del__(self):
        """Free pytorch's cuda cache. Note that nvidia-smi can still report
        taken memory but that's expected: pytorch reuses memory that was
        previously allocated.
        """

        try:
            torch.cuda.empty_cache()
        except AttributeError:
            pass

    def __str__(self) -> str:
        """
        :return: A string explaining the model structure
        """

        return str(self._model)

    @property
    def settings(self) -> RandLANetSettings:
        """The model settings."""

        return self._model.settings

    @property
    def module(self) -> torch.nn.Module:
        """Underlying torch Module."""
        return self._model

    @staticmethod
    def load(path: Path, use_gpu: bool = True, **kwargs) -> "Model":
        """Load Model from serialized file on disk

        :param path: Path where model file is saved.
        :param use_gpu: Boolean indicating if GPU should be used when available.
        :return: Loaded model.
        """

        assert path.is_file(), f"Could not find model file at {path}!"
        device = torch.device(
            "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
        )
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            shutil.unpack_archive(str(path), tmp, format="zip")
            with (tmp / "config").open("r") as f:
                config = json.load(f)
            # loading config
            settings = RandLANetSettings(**config)
            model_path = tmp / "model"
            state_dict = torch.load(model_path, device)
            if "model" in state_dict.keys():
                state_dict = state_dict["model"]
        # update settings with possible given arguments
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        model = Model(settings, weights=state_dict, use_gpu=use_gpu)
        return model

    def save(self, path: Path) -> None:
        """Save current model weights

        :param path: Path to save weights to
        """

        os.makedirs(path.parent, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            with (tmp / "config").open("w") as f:
                json.dump(asdict(self.settings), f)
            torch.save(self._model.state_dict(), tmp / "model")
            with tempfile.TemporaryDirectory() as tmp2:
                shutil.make_archive(str(Path(tmp2) / "file"), "zip", tmp)
                shutil.move(str(Path(tmp2) / "file.zip"), path)

    def upsample(
        self,
        logits: torch.Tensor,
        xyz: torch.Tensor,
        xyz_upsampled: torch.Tensor,
    ) -> torch.Tensor:
        """Upsample predictions, starting from given logits.
        :param logits: Logits, i.e. network output before soft-max (B, C, N1).
        :param xyz: Point coordinates (B, N1, 3).
        :param xyz_upsampled: Point coordinates of upsampled point cloud (B, N2, 3).
        :return: Upsampled predictions (B, N2)
        """

        # confidences as softmax of logits
        confidences = torch.softmax(logits, dim=-2).unsqueeze(3)
        # upsample confidences
        confidences_upsampled = self._upsampler(
            confidences, xyz, xyz_upsampled
        ).squeeze(-1)
        # take max to define classes
        predictions = torch.max(confidences_upsampled, dim=-2).indices
        return predictions

    def predict(
        self,
        xyz: np.ndarray,
        features: Optional[np.ndarray] = None,
        prepostprocess: bool = True,
    ) -> np.ndarray:
        """Predict on one or a batch of point clouds.
        :param xyz: Point coordinates (B, N, 3) or (N, 3).
        :param features: Optional point features (B, F, 3) or (F, 3).
        :param prepostprocess: Include preprocessing (downsampling) and
                               postprocessing (upsampling) of point cloud.
        :return: Predicted point classes (B, N) or (N,).
        """

        # warning if better knn approach could be used
        if self.settings.n_points > 20000:
            if self.settings.n_neighbors < 32:
                if self.settings.knn != "kdtree":
                    logging.warning(
                        "For improved performance, it is recommended to "
                        'use knn="kdtree" when N > 20000 and K < 32.'
                    )
            else:
                if self.settings.knn != "approximate":
                    logging.warning(
                        "For improved performance, it is recommended to "
                        'use knn="approximate" when N > 20000 and K > 32.'
                    )
            if self.settings.knn == "naive":
                logging.warning(
                    'Using knn="naive" for N > 20000 potentially has '
                    "very low performance or will reach an OOM!"
                )
        else:
            if self.settings.knn != "naive":
                logging.warning(
                    "For improved performance, it is recommended to "
                    'use knn="naive" when N < 20000.'
                )

        assert xyz.shape[-1] == 3, "xyz should have shape (B) x N x 3!"
        # add batch dimension if necessary
        batched = True
        if len(xyz.shape) == 2:
            xyz = np.expand_dims(xyz, 0)
            batched = False
        if features is not None and len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        input = xyz
        if features is not None:
            # concatenate xyz and features
            assert (
                xyz.shape[0] == features.shape[0]
            ), "xyz and features should have same batch size!"
            assert (
                xyz.shape[1] == features.shape[1]
            ), "xyz and features should have same number of points!"
            input = np.concatenate((xyz, features), axis=-1)

        if self.settings.upsampling == "none":
            prepostprocess = False

        with torch.no_grad():
            input_t = torch.from_numpy(input.astype(np.float32))
            if prepostprocess:
                # sample points
                indices = sample_points(
                    input.shape[1], self.settings.n_points, consistent=True
                )
                input_t_sampled = input_t[:, indices, :]
                # predict logits
                logits = self._model(input_t_sampled.to(self._model.device))
                # upsample predictions
                predictions = (
                    self.upsample(
                        logits, input_t_sampled[:, :, :3], input_t[:, :, :3]
                    )
                    .cpu()
                    .numpy()
                )
            else:
                # predict logits
                logits = self._model(input_t.to(self._model.device))
                # take max to define classes
                predictions = torch.max(logits, dim=-2).indices.cpu().numpy()
        if not batched:
            predictions = predictions[0]
        return predictions

    def train(
        self,
        dataset_train: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        dataset_validation: Sequence[
            Tuple[np.ndarray, np.ndarray, np.ndarray]
        ],
        training_settings: TrainingSettings = TrainingSettings(),
        augmentation_settings: AugmentationSettings = AugmentationSettings(),
        log_dir: Optional[Path] = None,
        class_names: Optional[List[str]] = None,
        callbacks: List[Callable[[int, Dict[str, float]], None]] = [],
    ):

        """Train this model, starting from the current loaded weights. This
        function will update those weights after training.

        :param dataset_train: The Dataset used for training as a sequence returning
                              the point coordinates (N, 3), the point
                              features (N, F) and the point class indices (N,)
        :param dataset_validation: The Dataset used for validation, using the
                                   same format as the dataset_train.
        :param training_settings: Settings defining the training process.
        :param augmentation_settings: Parameters for augmentations to use during
                                      training.
        :param log_dir: Path to the directory to dump checkpoints and
                        tensorboard events.
        :param class_names: Optional list of class names for prettier logging.
        :param callbacks: List of callback functions that are evaluated every
                          epoch.
        """

        assert (
            class_names is not None
            and len(class_names) == self.settings.n_classes
        ), (
            "The length of given class names should correspond to the "
            "n_classes setting of the model"
        )

        # generate batched data loaders
        train_dataloader = get_data_loader(
            dataset_train,
            self.settings.n_points,
            training_settings.batch_size,
            shuffle=True,
            consistent_sampling=False,
            augmentation_settings=augmentation_settings,
        )
        validation_dataloader = get_data_loader(
            dataset_validation,
            self.settings.n_points,
            training_settings.batch_size,
            shuffle=False,
            consistent_sampling=True,
        )

        trainer = Trainer(
            train_dataloader, validation_dataloader, log_dir, class_names
        )
        trained_model = trainer.train(
            self._model, training_settings, callbacks=callbacks
        )
        self._model = trained_model

    def evaluate(
        self,
        dataset: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        class_names: Optional[List[str]] = None,
        batch_size: int = 16,
        loss_function: str = "dice",
        postprocess: bool = False,
        include_stdev: bool = False,
    ) -> Dict:
        """Run the evaluation of the model on a given dataset.
        :param dataset: The Dataset used for evaluation as a sequence returning
                        the point coordinates (N, 3), the point
                        features (N, F) and the point class indices (N,)
        :param class_names: Optional list of class names for prettier logging.
        :param batch_size: The size of minibatches to process data during
                           evaluation.
        :param loss_function: The loss function to return the value from.
        :param postprocess: Include postprocessing (upsampling) for evaluation.
        :param include_stdev: Include the standard deviation in the metrics.
        :return: Dictionary with the different metrics. When include_stdev is
                 True, each metric is represented as a tuple of the mean and
                 standard deviation.
        """

        # generate batched data loaders
        dataloader = get_data_loader(
            dataset,
            self.settings.n_points,
            batch_size,
            shuffle=False,
            consistent_sampling=True,
        )
        metric_collector_bag = Trainer.evaluate(
            self._model, dataloader, class_names, loss_function, postprocess
        )
        return metric_collector_bag.as_dict(include_stdev=include_stdev)
