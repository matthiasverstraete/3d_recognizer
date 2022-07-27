from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

# TODO: remove comments
from .knn import knn_approximate, knn_naive  # , knn_kdtree


@dataclass
class RandLANetSettings:
    #: Number of classes (including background)
    n_classes: int
    #: Number of points used during training
    n_points: int = 10000
    #: Number of point features (excluding xyz coordinates)
    n_features: int = 0
    #: Number of neighbors to observer during local aggregation.
    n_neighbors: int = 32
    #: Downsampling factor for each encoder layer.
    decimation: int = 4
    #: Output sizes of each layer in decoder, is the "d_out" from paper (half size)
    layer_sizes: List[int] = field(
        default_factory=(lambda: [16, 64, 128, 256])
    )
    #: KNN approach to use. Is one of the following choices:
    #: - kdtree: KNN search using k-d tree space partitioning (on CPU)
    #: - approximate: Approximate KNN search using the FAISS implementation (on CPU)
    #: - naive: KNN search by computing the distances to each point. Based on
    #:          matrix multiplications that can be heavily parallellized on GPU.
    #:          Performance is quasi independent of K, but reduces quadratically
    #:          with N. Takes up a lot of memory for large point clouds.
    knn: str = "approximate"
    #: Upsampling approach to use during postprocessing. Is one of the following choices:
    #: - none: Do not use upsampling.
    #: - nni: Nearest neighbor interpolation.
    #: - nna: Nearest neighbors average.
    #: - idw: Nearest neighbors inverse distance weighting.
    #: - isdw: Nearest neighbors inverse squared distance weighting.
    upsampling: str = "nni"

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        RandLANetSettings.__init__(obj, *args, **kwargs)
        assert obj.knn in ["kdtree", "approximate", "naive"], (
            f"knn value \"{kwargs['knn']}\" not understood, "
            'should be "kdtree", "approximate" or "naive"'
        )
        assert obj.upsampling in ["none", "nni", "nna", "idw", "isdw"], (
            f"upsampling value \"{kwargs['upsampling']}\" not understood, "
            'should be "none", "nni", "nna", "idw", or "isdw"'
        )
        return obj

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


class SharedMLP(torch.nn.Module):
    """Shared MLP block. An MLP is shared accross the points, implemented
    as a 2D convolution."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        transpose: bool = False,
        bn: bool = True,
        activation: Optional[torch.nn.Module] = None,
    ):
        """
        :param n_in: Number of input feature dimensions.
        :param n_out: Number of output feature dimensions.
        :param transpose: Boolean indicating to take transpose convolution
                          (in decoder layers).
        :param bn: Use batch norm layer.
        :param activation: Activation function to use or None.
        """

        super(SharedMLP, self).__init__()
        conv_fn = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
        self.conv = conv_fn(
            n_in, n_out, kernel_size=1, stride=1, padding_mode="zeros"
        )
        self.batch_norm = (
            torch.nn.BatchNorm2d(n_out, eps=1e-6, momentum=0.99)
            if bn
            else None
        )
        self.activation = activation

    def forward(self, input: torch.Tensor):
        """Forward pass.
        :param input: Tensor of shape (B, n_in, N, 1)
        :return: Tensor of shape (B, n_out, N, 1)
        """

        features = self.conv(input)
        if self.batch_norm:
            features = self.batch_norm(features)
        if self.activation:
            features = self.activation(features)
        return features


class KNN(torch.nn.Module):
    """K-Nearest Neighbors block."""

    def __init__(self, device: torch.device):
        """
        :param device: Device to run the network on.
        """

        super(KNN, self).__init__()
        self._device = device

    def forward(
        self,
        xyz: torch.Tensor,
        xyz_query: torch.Tensor,
        n_neighbors: int,
        approach: str = "approximate",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Searches for each point in xyz_query the n_neighbors
        closest points in xyz.
        :param xyz: Support point coordinates (B, N', 3).
        :param xyz_query: Query point coordinates (B, N, 3).
        :param n_neighbors: Number of neighbors to search.
        :param approach: KNN approach to use ("kdtree", "approximate" or "naive").
        :return: Tensor with neighbor indices and tensor with corresponding
                 distances (B, N, K).
        """

        # if approach == "kdtree":
        #     neighbors, distances_sq = knn_kdtree(xyz, xyz_query, n_neighbors)
        #     neighbors = neighbors.to(self._device)
        #     distances_sq = distances_sq.to(self._device)
        if approach == "approximate":
            neighbors, distances_sq = knn_approximate(
                xyz, xyz_query, n_neighbors
            )
            neighbors = neighbors.to(self._device)
            distances_sq = distances_sq.to(self._device)
        elif approach == "naive":
            neighbors, distances_sq = knn_naive(xyz, xyz_query, n_neighbors)
        else:
            raise ValueError(f"KNN approach {approach} not understood!")
        distances = torch.sqrt(distances_sq)
        return neighbors, distances


class RelativePositionEncoding(torch.nn.Module):
    """Relative position encoding block."""

    def forward(
        self,
        xyz: torch.Tensor,
        neighbors: torch.Tensor,
        distances: torch.Tensor,
    ):
        """Forward pass. Does NOT apply the MLP to the relative position
        encoding!
        :param xyz: Point coordinates (B, N, 3).
        :param neighbors: Neighbor indices (B, N, K).
        :param distances: Neighbor distances (B, N, K).
        :return: Tensor of relative position encodings (B, 10, N, K).
        """

        B, N, K = neighbors.size()

        # get neighboring xyz
        expanded_neighbors = neighbors.unsqueeze(1).expand(B, 3, N, K)
        expanded_xyz = xyz.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
        neighbor_xyz = torch.gather(expanded_xyz, 2, expanded_neighbors)
        # concat all position information
        relative_position_encoding = torch.cat(
            (
                expanded_xyz,
                neighbor_xyz,
                expanded_xyz - neighbor_xyz,
                distances.unsqueeze(-3),
            ),
            dim=-3,
        )
        return relative_position_encoding


class PointFeatureAugmentation(torch.nn.Module):
    """Point feature augmentation block, combining relative position
    encodings with neighbor features.
    """

    def forward(
        self,
        relative_position_encoding: torch.Tensor,
        features: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        :param relative_position_encoding: Relative position encodings
                                           (B, n_out/2, N, K).
        :param features: Point features (B, n_out/2, N, 1).
        :param neighbors: Neighbor indices (B, N, K).
        :return: Tensor of concatenated relative position encodings and
                 neighbor features (B, n_out, N, K)
        """

        B, N, K = neighbors.size()
        n_features = features.size(1)

        # get neighboring features
        expanded_neighbors = neighbors.unsqueeze(1).expand(B, n_features, N, K)
        expanded_features = features.expand(B, -1, N, K)
        neighbor_features = torch.gather(
            expanded_features, 2, expanded_neighbors
        )
        # concat relative position encoding with neighbor features
        return torch.cat(
            (relative_position_encoding, neighbor_features), dim=-3
        )


class AttentivePooling(torch.nn.Module):
    """Attentive pooling block."""

    def __init__(self, n_in: int, n_out: int):
        """
        :param n_in: Number of input feature dimensions.
        :param n_out: Number of output feature dimensions.
        """

        super(AttentivePooling, self).__init__()
        self.score_fn = torch.nn.Sequential(
            torch.nn.Linear(n_in, n_in, bias=False), torch.nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(n_in, n_out, activation=torch.nn.ReLU())

    def forward(self, input: torch.Tensor):
        """Forward pass.
        :param input: Tensor of shape (B, n_in, N, K)
        :return: Tensor of shape (B, n_out, N, 1)
        """

        # computing attention scores
        scores = (
            self.score_fn(input.permute(0, 2, 3, 1))
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        # sum over the neighbors
        features = torch.sum(scores * input, dim=-1, keepdim=True)
        return self.mlp(features)


class LocalFeatureAggregation(torch.nn.Module):
    """Local feature aggregation block."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_neighbors: int,
        device: torch.device,
    ):
        """
        :param n_in: Number of input feature dimensions.
        :param n_out: Number of output feature dimensions.
        :param n_neighbors: Number of neighbors to observe during
                            local aggregation.
        :param device: Device to run the network on.
        """

        super(LocalFeatureAggregation, self).__init__()
        self._n_neighbors = n_neighbors
        self._device = device

        self.mlp1 = SharedMLP(
            n_in, n_out // 2, activation=torch.nn.LeakyReLU(0.2)
        )
        self.mlp2 = SharedMLP(n_out, 2 * n_out)
        self.shortcut = SharedMLP(n_in, 2 * n_out)

        self.knn = KNN(device)
        self.rpe = RelativePositionEncoding()
        self.pfa = PointFeatureAugmentation()

        self.mlp_rpe1 = SharedMLP(10, n_out // 2, activation=torch.nn.ReLU())
        self.mlp_rpe2 = SharedMLP(
            n_out // 2, n_out // 2, activation=torch.nn.ReLU()
        )

        self.pool1 = AttentivePooling(n_out, n_out // 2)
        self.pool2 = AttentivePooling(n_out, n_out)

        self.lrelu = torch.nn.LeakyReLU()

    def forward(
        self, xyz: torch.Tensor, input: torch.Tensor, knn_approach: str
    ):
        """Forward pass.
        :param xyz: Point coordinates (B, N, 3).
        :param input: Point features (B, n_in, N, 1).
        :param knn_approach: KNN approach to use ("kdtree", "approximate" or "naive").
        :return: Aggregated point features (B, 2*n_out, N, 1)
        """

        # find nearest neighbors
        neighbors, distances = self.knn(
            xyz, xyz, self._n_neighbors, knn_approach
        )
        xyz_device = xyz.to(self._device)
        # input transformation
        features = self.mlp1(input)
        # first local spatial encoding + attentive pooling
        relative_position_encoding = self.rpe(xyz_device, neighbors, distances)
        relative_position_encoding = self.mlp_rpe1(relative_position_encoding)
        features = self.pfa(relative_position_encoding, features, neighbors)
        features = self.pool1(features)
        # second local spatial encoding + attentive pooling
        relative_position_encoding = self.mlp_rpe2(relative_position_encoding)
        features = self.pfa(relative_position_encoding, features, neighbors)
        features = self.pool2(features)

        return self.lrelu(self.mlp2(features) + self.shortcut(input))


class UpSampler(torch.nn.Module):
    """Feature upsampler block."""

    def __init__(self, upsampling_approach: str, device: torch.device):
        """
        :param upsampling_approach: Upsampling approach to use
            ("none", "nni", "nna", "idw", "isdw")
        :param device: Device to run the network on.
        """

        super(UpSampler, self).__init__()
        self._upsampling_approach = upsampling_approach
        self._device = device
        self.knn = KNN(device)

    def nearest_neighbor_interpolation(
        self,
        features: torch.Tensor,
        xyz: torch.Tensor,
        xyz_upsampled: torch.Tensor,
    ) -> torch.Tensor:
        """Nearest neighboring interpolation:  upsampled point takes value of
        nearest neighbor in original point cloud.
        :param features: Point features to upsample (B, F, N1, 1).
        :param xyz: Point coordinates corresponding to features (B, N1, 3).
        :param xyz_upsampled: Upsampled point coordinates (B, N2, 3).
        :return: Upsampled features (B, F, N2, 1)
        """

        # search nearest neighbor
        neighbors, _ = self.knn(xyz, xyz_upsampled, 1)
        expanded_neighbors = neighbors.unsqueeze(1).expand(
            -1, features.size(1), -1, 1
        )
        # find corresponding features
        upsampled_features = torch.gather(features, -2, expanded_neighbors)
        return upsampled_features

    def nearest_neighbors_averaging(
        self,
        features: torch.Tensor,
        xyz: torch.Tensor,
        xyz_upsampled: torch.Tensor,
        n_neighbors: int = 8,
        inverse_distance_weighting: bool = True,
        distance_power: float = 1.0,
    ) -> torch.Tensor:
        """Interpolation by averaging the features of the K nearest neighbors
        in the original point cloud.
        :param features: Point features to upsample (B, F, N1, 1).
        :param xyz: Point coordinates corresponding to features (B, N1, 3).
        :param xyz_upsampled: Upsampled point coordinates (B, N2, 3).
        :param n_neighbors: Number of neighbors to consider for averaging.
        :param inverse_distance_weighting: Use the inverse distance as weights
                                           for averaging.
        :param distance_power: Power for the distance in the inverse distance
                               weight.
        :return: Upsampled features (B, F, N2, 1)
        """

        # search k nearest neighbors
        neighbors, distances = self.knn(xyz, xyz_upsampled, n_neighbors)
        expanded_neighbors = neighbors.unsqueeze(1).expand(
            -1, features.size(1), -1, n_neighbors
        )
        expanded_features = features.expand(-1, -1, -1, n_neighbors)
        # find corresponding features and average over neighbors
        neighbor_features = torch.gather(
            expanded_features, 2, expanded_neighbors
        )
        if inverse_distance_weighting:
            eps = 1e-7
            weights = (1.0 + eps) / (distances ** distance_power + eps)
            weights /= torch.sum(weights, dim=-1, keepdim=True)
            expanded_weights = weights.unsqueeze(1).expand(
                -1, features.size(1), -1, n_neighbors
            )
            # inverse distance weighting
            upsampled_features = torch.sum(
                expanded_weights * neighbor_features, dim=-1, keepdim=True
            )
        else:
            # regular averaging
            upsampled_features = torch.mean(
                neighbor_features, dim=-1, keepdim=True
            )
        return upsampled_features

    def forward(
        self,
        features: torch.Tensor,
        xyz: torch.Tensor,
        xyz_upsampled: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. Upsample features from xyz point coordinates to
        xyz_upsampled point coordinates.
        :param features: Point features to upsample (B, F, N1, 1).
        :param xyz: Point coordinates corresponding to features (B, N1, 3).
        :param xyz_upsampled: Upsampled point coordinates (B, N2, 3).
        :return: Upsampled features (B, F, N2, 1)
        """

        if self._upsampling_approach == "nni":
            return self.nearest_neighbor_interpolation(
                features, xyz, xyz_upsampled
            )
        elif self._upsampling_approach == "nna":
            return self.nearest_neighbors_averaging(
                features, xyz, xyz_upsampled
            )
        elif self._upsampling_approach == "idw":
            return self.nearest_neighbors_averaging(
                features, xyz, xyz_upsampled, inverse_distance_weighting=True
            )
        elif self._upsampling_approach == "isdw":
            return self.nearest_neighbors_averaging(
                features,
                xyz,
                xyz_upsampled,
                inverse_distance_weighting=True,
                distance_power=2,
            )
        elif self._upsampling_approach == "none":
            return features
        else:
            raise ValueError(
                f"Upsampling approach {self._upsampling_approach} "
                "not understood!"
            )


class RandLANet(torch.nn.Module):
    """Implementation of the RandLA-Net model."""

    def __init__(
        self,
        settings: RandLANetSettings,
        device: Optional[torch.device] = None,
    ):
        """
        :param settings: All model settings.
        :param device: Optional device to run the network on. If None given,
                       take GPU when available, otherwise CPU.
        """

        super(RandLANet, self).__init__()
        self._settings = settings
        n_neighbors = self._settings.n_neighbors
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        self._device = device

        # Minimum number of points in the point cloud determined by
        # (1) At least K points should be available for KNN. In encoder, KNN is
        #     applied up till before the last layer.
        # (2) Fully downsampled encoede embeddings should have at least
        #     2 points left.
        n_layers = len(self._settings.layer_sizes)
        self._min_n_points = max(
            n_neighbors * (self._settings.decimation ** (n_layers - 1)),
            2 * (self._settings.decimation ** n_layers),
        )
        n_in = self._settings.n_features + 3

        # input transformation
        self.fc_start = torch.nn.Linear(n_in, 8)
        self.bn_start = torch.nn.Sequential(
            torch.nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            torch.nn.LeakyReLU(0.2),
        )
        # encoder
        n_outs = self._settings.layer_sizes
        self.encoder = torch.nn.ModuleList([])
        n_in = self.fc_start.out_features
        for n_out in n_outs:
            self.encoder.append(
                LocalFeatureAggregation(n_in, n_out, n_neighbors, device)
            )
            n_in = 2 * n_out
        self.mlp = SharedMLP(n_in, n_in, activation=torch.nn.ReLU())
        # decoder
        self.upsampling = UpSampler("nni", device)
        self.decoder = torch.nn.ModuleList([])
        n_in *= 2  # due to concatenation with intermediate feature map
        for n_out in n_outs[::-1][1:]:
            self.decoder.append(
                SharedMLP(
                    n_in, 2 * n_out, transpose=True, activation=torch.nn.ReLU()
                )
            )
            n_in = 4 * n_out
        self.decoder.append(
            SharedMLP(n_in, 8, transpose=True, activation=torch.nn.ReLU())
        )
        # final fully connected layers
        self.fc_end = torch.nn.Sequential(
            SharedMLP(8, 64, activation=torch.nn.ReLU()),
            SharedMLP(64, 32, activation=torch.nn.ReLU()),
            torch.nn.Dropout(),
            SharedMLP(32, self._settings.n_classes, bn=False),
        )
        # put model on device
        self = self.to(self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def settings(self) -> RandLANetSettings:
        return self._settings

    def forward(self, input: torch.Tensor):
        """Forward pass.
        :param input: Tensor of shape (B, N, 3 + F) combining xyz coordinates
                      and point features (F).
        :return: Logits, i.e. network output before soft-max (B, C, N).
        """

        B, N, dim = input.size()
        assert (
            dim == 3 + self._settings.n_features
        ), "Input should have shape (B, N, 3 + F)!"
        assert (
            N >= self._min_n_points
        ), f"Input point cloud should have at least {self._min_n_points} points!"

        # save xyz coordinates for the random sampling
        if self._settings.knn in ["kdtree", "approximate"]:
            # save to cpu
            xyz = input[..., :3].float().cpu()
        else:
            xyz = input[..., :3].float()

        # input transformation
        features = self.fc_start(input).transpose(-2, -1).unsqueeze(-1)
        features = self.bn_start(features)  # (B, 8, N, 1)

        # doing random permutation as part of random sampling
        # use numpy's random generator in stead of torch's as it allows better
        # control of the seed/state
        permutation = torch.from_numpy(np.random.permutation(N))
        xyz = xyz[:, permutation]
        features = features[:, :, permutation]

        xyz_sampled = xyz
        features_sampled = features
        decimation = self._settings.decimation

        # encoder
        decimation_ratio = 1
        features_stack = []
        for lfa in self.encoder:
            # local feature aggregation
            features = lfa(xyz_sampled, features_sampled, self._settings.knn)
            features_stack.append(features)
            # random sampling
            decimation_ratio *= decimation
            xyz_sampled = xyz[:, : (N // decimation_ratio)]
            features_sampled = features[:, :, : (N // decimation_ratio)]

        features = self.mlp(features_sampled)

        # decoder
        for mlp in self.decoder:
            # upsample features
            xyz_from = xyz[:, : (N // decimation_ratio)]
            xyz_to = xyz[:, : (decimation * N // decimation_ratio)]
            upsampled_features = self.upsampling(features, xyz_from, xyz_to)
            # stack with intermediate encoder features
            features = torch.cat(
                (upsampled_features, features_stack.pop()), dim=1
            )
            # pass through shared mlp
            features = mlp(features)
            decimation_ratio //= decimation

        # inverse permutation: bring feature order back to original point order
        features = features[:, :, torch.argsort(permutation)]
        # final fully connected layers
        logits = self.fc_end(features)
        return logits.squeeze(-1)
