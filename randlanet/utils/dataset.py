from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import preprocessing
from .augmentation import AugmentationSettings, perturbate_point_cloud


class PointCloudPreprocessor(Dataset):
    def __init__(
        self,
        dataset: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        n_sample_points: int,
        consistent_sampling: bool = True,
        augmentation_settings: Optional[AugmentationSettings] = None,
        normalization: Optional[str] = None,
    ) -> None:
        """Torch dataset that preprocesses (sample, normalize, augment) the
        point clouds and labels.
        :param dataset: Sequence returning point coordinates (N, 3), point
                        features (N, F), and point labels (N, 1).
        :param n_sample_points: Number of points to sample.
        :param consistent_sampling: Make sampling reproducable.
        :param augmentation_settings: Settings for augmentation.
        :param normalization: Optional string with the normalization approach
                              to use (mean, max or stdev). If None, don't
                              normalize.
        """

        self._dataset = dataset
        self._n_sample_points = n_sample_points
        self._consistent_sampling = consistent_sampling
        self._augmentation_settings = augmentation_settings
        self._normalization = normalization
        self._epoch = 0

    def __getitem__(
        self, idx: int, preprocess: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return torch tensors of point coordinates, features and labels,
        after preprocessing, normalizing and augmenting.
        :param idx: Integer index indicating the sample in the dataset.
        :return: Tuple of input and labels torch Tensors and index.
        """
        if preprocess:
            xyz, features, labels = self.preprocess(*self._dataset[idx])
        else:
            xyz, features, labels = self._dataset[idx]
        xyz_t = torch.from_numpy(xyz).float()
        features_t = torch.from_numpy(features).float()
        input_t = torch.cat((xyz_t, features_t), dim=1)
        labels_t = torch.from_numpy(labels).long()
        return input_t, labels_t, idx

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self._dataset)

    def preprocess(
        self, xyz: np.ndarray, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess one sample and labels.
        :param xyz: Point coordinates (N, 3).
        :param features: Point features (N, F).
        :param labels: Point labels (N,).
        :return: Preprocessed coordinates, features and labels.
        """
        N = xyz.shape[0]
        assert xyz.shape[1] == 3, "Point coordinates should have shape (N, 3)!"
        assert features.shape[0] == N, "Features should have shape (N, F)!"
        assert labels.shape == (N,), "Labels should have shape (N,)!"

        sample_indices = preprocessing.sample_points(
            N, self._n_sample_points, consistent=self._consistent_sampling
        )
        sampled_xyz = xyz[sample_indices]
        sampled_features = features[sample_indices]
        sampled_labels = labels[sample_indices]
        if self._normalization is not None:
            center = np.mean(sampled_xyz, axis=0, keepdims=True)
            sampled_xyz = sampled_xyz - center
            if self._normalization == "mean":
                radius = np.mean(np.linalg.norm(sampled_xyz, axis=1))
            elif self._normalization == "max":
                radius = np.max(np.linalg.norm(sampled_xyz, axis=1))
            elif self._normalization == "stdev":
                radius = np.std(np.linalg.norm(sampled_xyz, axis=1))
            else:
                radius = 1.0
            sampled_xyz /= radius
        if self._augmentation_settings:
            sampled_xyz = perturbate_point_cloud(
                sampled_xyz, self._augmentation_settings
            )
        return sampled_xyz, sampled_features, sampled_labels


def get_data_loader(
    dataset: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_sample_points: int,
    batch_size: int,
    shuffle: bool = False,
    consistent_sampling: bool = True,
    augmentation_settings: Optional[AugmentationSettings] = None,
    normalization: Optional[str] = None,
) -> DataLoader:
    """Get torch dataloader to load preprocessed data.
    :param dataset: Sequence returning point coordinates (N, 3), point
                    features (N, F), and point labels (N, 1).
    :param n_sample_points: Number of points to sample.
    :param batch_size: Number of samples per minibatch.
    :param consistent_sampling: Make sampling reproducable.
    :param augmentation_settings: Settings for augmentation.
    :param normalization: Optional string with the normalization approach
                          to use (mean, max or stdev). If None, don't
                          normalize.
    :return: Torch DataLoader instance.
    """

    preprocessed_dataset = PointCloudPreprocessor(
        dataset,
        n_sample_points,
        consistent_sampling=consistent_sampling,
        augmentation_settings=augmentation_settings,
        normalization=normalization,
    )
    return DataLoader(
        preprocessed_dataset, batch_size=batch_size, shuffle=shuffle
    )
