from typing import Optional

import numpy as np


def random_choice(
    a: int,
    size: int,
    replace: bool = True,
    p: Optional[np.ndarray] = None,
    consistent: bool = False,
) -> np.ndarray:
    """Wrapper for random choice, resets seed when consistent sampling is
    requested.
    :param a: Maximum value to choose.
    :param size: Size of output array.
    :param replace: Sample with replacement.
    :param p: Probabities to sample.
    :param consistent: Boolean to make sampling consistent by setting the seed
                       to a fixed value first.
    :return: Sampled array.
    """
    if consistent:
        # save state of randomize
        rnd_state = np.random.get_state()
        # set seed to consistent value
        np.random.seed(0)
    value = np.random.choice(a, size, replace, p)
    if consistent:
        # reset state of randomizer
        np.random.set_state(rnd_state)
    return value


def sample_points(
    n_points: int, n_sample_points: int, consistent: bool = False
) -> np.ndarray:
    """Random sub- or upsample points.
    When the number of sampled points is larger than the curren points,
    randomly duplicate points.
    :param n_points: Number of points in the current point cloud.
    :param n_sample_points: Number of points to sample.
    :param consistent: Boolean to make sampling consistent by setting the seed
                       to a fixed value first.
    :return: Point indices for sampling.
    """

    # sample without replacement as long as n_sample_points <= n_points
    sample_indices = random_choice(
        n_points,
        min(n_sample_points, n_points),
        replace=False,
        consistent=consistent,
    )
    if n_sample_points > n_points:
        n_points_dupl = n_sample_points - n_points
        # sample with replacement as we can have n_points_dupl > n_points
        sample_indices_dupl = random_choice(
            n_points, n_points_dupl, replace=True, consistent=consistent
        )
        sample_indices = np.r_[sample_indices, sample_indices_dupl]
    return sample_indices


def sample_points_balanced(
    labels, n_sample_points: int, consistent: bool = False
) -> np.ndarray:
    """Random sub- or upsample points giving more weight to the under
    represented classes. Not a good idea to use this during training as
    it gives a wrong representation of reality.
    :param labels: Labels, each label indicates the class index (N,)
    :param n_sample_points: Number of points to sample.
    :param consistent: Boolean to make sampling consistent by setting the seed
                       to a fixed value first.
    :return: Point indices for sampling.
    """

    n_points = len(labels)
    # express labels as one hot encoded
    n_classes = len(np.unique(labels))
    one_hot_encoded = np.eye(n_classes)[labels]
    # subsample based on class occurences
    inverse_annotation = 1 - one_hot_encoded
    normalized_inverse_annotation = inverse_annotation / np.sum(
        inverse_annotation, axis=-1, keepdims=True
    )
    # average over points: expresses for each class the chance of not occuring
    p_global = np.sum(normalized_inverse_annotation, axis=0) / np.sum(
        normalized_inverse_annotation
    )
    if 0 in p_global:  # only one class available
        sample_indices = random_choice(
            n_points, n_sample_points, consistent=consistent
        )
    else:
        # for each point: take class in-occuring chance
        p_local = np.dot(one_hot_encoded, p_global.T)
        # normalize to express chance for each point to be sampled
        p_local_normalized = p_local / np.sum(p_local)
        p_local_normalized = np.squeeze(p_local_normalized)
        sample_indices = random_choice(
            n_points,
            n_sample_points,
            p=p_local_normalized,
            consistent=consistent,
        )
    return sample_indices


def sample_points_equal(
    labels, n_sample_points: int, ratio: float = 1.0, consistent: bool = False
) -> np.ndarray:
    """Random sub- or upsample points making sure that each class is
    represented equal. Not a good idea to use this during training as
    it gives a wrong representation of reality.
    :param labels: Labels, each label indicates the class index (N,)
    :param n_sample_points: Number of points to sample.
    :param consistent: Boolean to make sampling consistent by setting the seed
                       to a fixed value first.
    :return: Point indices for sampling.
    """

    n_points = len(labels)
    indices = np.arange(0, n_points)
    sample_indices = np.array([], dtype=int)
    unique_labels = np.unique(labels)
    n_points_per_class = [
        np.count_nonzero(labels == lbl) for lbl in unique_labels
    ]
    n_sample_points_per_class_equal = int(
        np.round(n_sample_points / len(unique_labels))
    )
    n_sample_points_per_class = np.round(
        [(n_sample_points / n_points) * np for np in n_points_per_class]
    ).astype(np.int32)
    n_sample_points_per_class = np.round(
        [
            ratio * n_sample_points_per_class_equal + (1 - ratio) * np
            for np in n_sample_points_per_class
        ]
    ).astype(np.int32)

    total_sample_points = np.sum(n_sample_points_per_class)
    if total_sample_points != n_sample_points:
        idx = np.argmax(n_sample_points_per_class)
        n_sample_points_per_class[idx] += n_sample_points - total_sample_points

    for idx, cls_idx in enumerate(unique_labels):
        selected_indices = indices[labels == cls_idx]

        ind = random_choice(
            n_points_per_class[idx],
            min(n_sample_points_per_class[idx], n_points_per_class[idx]),
            replace=False,
            consistent=consistent,
        )
        if n_sample_points_per_class[idx] > n_points_per_class[idx]:
            n_points_dupl = (
                n_sample_points_per_class[idx] - n_points_per_class[idx]
            )
            # sample with replacement as we can have n_points_dupl > n_points_per_class
            ind_dupl = random_choice(
                n_points_per_class[idx],
                n_points_dupl,
                replace=True,
                consistent=consistent,
            )
            ind = np.r_[ind, ind_dupl]

        sample_indices = np.append(sample_indices, selected_indices[ind])
    return sample_indices


def sample_points_factor(
    n_points: int, factor: float = 0.25, consistent: bool = False
) -> np.ndarray:
    """Random sub- or upsample points by a factor.
    :param n_points: Number of points in the current point cloud.
    :param factor: Factor of sampling;
        number of sampling points = factor * n_points.
    :param consistent: Boolean to make sampling consistent by setting the seed
                       to a fixed value first.
    :return: Point indices for sampling.
    """

    # sample without replacement as long as n_sample_points <= n_points
    n_sample_points = int(n_points * factor)
    sample_indices = random_choice(
        n_points,
        min(n_sample_points, n_points),
        replace=False,
        consistent=consistent,
    )
    if n_sample_points > n_points:
        n_points_dupl = n_sample_points - n_points
        # sample with replacement as we can have n_points_dupl > n_points
        sample_indices_dupl = random_choice(
            n_points, n_points_dupl, replace=True, consistent=consistent
        )
        sample_indices = np.r_[sample_indices, sample_indices_dupl]
    return sample_indices
