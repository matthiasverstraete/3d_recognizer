from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AugmentationSettings:
    #: Variance of random perturbation of single points
    jitter_variance: float = 0.01
    #: Clip value of random perturbation of single points
    jitter_limit: float = 0.05
    #: Maximum scale for random scaling of the point cloud.
    #: Scale will be randomly selected from [1 - scale_limit, 1 + scale_limit]
    scale_limit: float = 0.2
    #: Maximum translation for random shifting the point cloud
    shift_limit: float = 0.1
    #: Variances of random rotation of the point cloud around x,y and z-axis respectively
    rotation_angle_variances: Tuple[float, float, float] = (0.06, 0.06, 0.06)
    #: Clip values of random rotation of the point cloud around x,y and z-axis respectively
    rotation_angle_limits: Tuple[float, float, float] = (0.18, 0.18, 0.18)


def get_mean_radius(xyz: np.ndarray) -> float:
    """Get mean radius of point cloud, as the mean distance to the center.
    :param xyz: Point cloud coordinates (N, 3).
    :return: Mean radius of point cloud.
    """

    center = np.mean(xyz, axis=0, keepdims=True)
    radius = np.mean(np.linalg.norm(xyz - center, axis=1))
    return float(radius)


def jitter_point_cloud(
    xyz: np.ndarray, variance: float = 0.01, limit: float = 0.05
) -> np.ndarray:
    """Randomly perturb single point positions. Scale perturbation with mean
    point cloud radius.
    :param xyz: Point cloud coordinates (N, 3).
    :param variance: Variance of random perturbation.
    :param limit: Limit for random perturbation.
    :return: Perturbed point cloud coordinates (N, 3).
    """

    radius = get_mean_radius(xyz)
    N, C = xyz.shape
    xyz_perturbed = np.clip(
        radius * variance * np.random.randn(xyz.shape[0], xyz.shape[1]),
        -limit,
        limit,
    )
    xyz_perturbed += xyz
    return xyz_perturbed


def random_scale_point_cloud(
    xyz: np.ndarray, scale_limit: float = 0.2
) -> np.ndarray:
    """Randomly scale the point cloud. Scale is per point cloud and is with
    respect to the point cloud center.
    :param xyz: Point cloud coordinates (N, 3).
    :param scale_limit: Maximum scale for random scaling of the point cloud.
                        Scale will be randomly selected from
                        [1 - scale_limit, 1 + scale_limit]
    :param min_scale: Minimum scaling alowed.
    :param max_scale: Maximum scaling alowed.
    :return: Scaled point cloud coordinates (N, 3).
    """

    scale = np.random.uniform(1 - scale_limit, 1 + scale_limit)
    center = np.mean(xyz, axis=0, keepdims=True)
    # scale with respect to center, otherwise point cloud will be translated
    # as well.
    xyz_scaled = (xyz - center) * scale + center
    return xyz_scaled


def random_rotate_point_cloud(
    xyz: np.ndarray,
    angle_variances: Tuple[float, float, float] = (0.06, 0.06, 0.06),
    angle_limits: Tuple[float, float, float] = (0.18, 0.18, 0.18),
) -> np.ndarray:
    """Randomly perturb the point cloud by small rotations. Rotations are
    around the center of the point cloud.
    :param xyz: Point cloud coordinates (N, 3).
    :param angle_variances: Variance of random angle selection around x,y,z axis (rad).
    :param angle_limits: Min/max angles around x,y,z axis (rad).
    :return: Perturbed point cloud coordinates (N, 3).
    """

    assert len(angle_variances) == 3, "angle_sigmas should have length 3"
    assert len(angle_limits) == 3, "angle_clips should have length 3"

    angles = [
        np.clip(angle_sigma * np.random.randn(), -angle_limit, angle_limit)
        for angle_sigma, angle_limit in zip(angle_variances, angle_limits)
    ]
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])],
        ]
    )
    Ry = np.array(
        [
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])],
        ]
    )
    Rz = np.array(
        [
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )
    R = Rz @ Ry @ Rx
    center = np.mean(xyz, axis=0, keepdims=True)
    # rotation around center of the point cloud
    xyz_rotated = (xyz - center) @ R.T + center
    return xyz_rotated


def random_shift_point_cloud(
    xyz: np.ndarray, shift_limit: float = 0.1
) -> np.ndarray:
    """Randomly shift the point cloud. Shift is per point cloud and is scaled
    with the mean point cloud radius.
    :param xyz: Point cloud coordinates (N, 3).
    :param shift_limit: Max translation alowed in any directions
    :return: shifted point cloud coordinates (N, 3)
    """

    radius = get_mean_radius(xyz)
    shifts = radius * np.random.uniform(-shift_limit, shift_limit, 3)
    xyz_shifted = xyz + shifts
    return xyz_shifted


def perturbate_point_cloud(
    xyz: np.ndarray, settings: AugmentationSettings
) -> np.ndarray:
    """
    Perturb point cloud in order to do some data augmentation
    :param xyz: Point cloud coordinates (N, 3).
    :param settings: The augmentation settings.
    :return: Perturbated point cloud coordinates (N, 3).
    """

    xyz_perturbed = jitter_point_cloud(
        xyz, settings.jitter_variance, settings.jitter_limit
    )
    xyz_perturbed = random_scale_point_cloud(
        xyz_perturbed, settings.scale_limit
    )
    xyz_perturbed = random_rotate_point_cloud(
        xyz_perturbed,
        settings.rotation_angle_variances,
        settings.rotation_angle_limits,
    )
    xyz_perturbed = random_shift_point_cloud(
        xyz_perturbed, settings.shift_limit
    )
    return xyz_perturbed
