from typing import Tuple

import faiss
import torch
# TODO: remove comments
# from .utils.knn_tpk import knn as knn_tpk


def knn_naive(
    xyz: torch.Tensor,
    xyz_query: torch.Tensor,
    n_neighbors: int,
    partition_size: int = 4000,
    n_parts_max: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """K Nearest Neighbor search based on distance matrix computations on GPU.
    Searches for each point in xyz_query the n_neighbors closest points
    in xyz.
    :param xyz: Support point coordinates (B, N', 3).
    :param xyz_query: Query point coordinates (B, N, 3).
    :param n_neighbors: Number of neighbors to search.
    :param partition_size: Number of query points to handle in one pass.
    :param n_parts_max: Maximum number of parts to divide the query points.
    :return: Tensor with neighbor indices and tensor with corresponding
             distances (B, N, K).
    """

    B, N, _ = xyz_query.size()
    n_parts = N // partition_size
    if n_parts > n_parts_max:
        n_parts = n_parts_max
    if n_parts == 0:
        n_parts = 1
    # number of query points to handle in one pass.
    n = N // n_parts
    neighbors = torch.empty(
        (B, N, n_neighbors), dtype=torch.int64, device=xyz.device
    )
    distances_sq = torch.empty(
        (B, N, n_neighbors), dtype=torch.float32, device=xyz.device
    )
    for i in range(n_parts):
        idx_start, idx_end = i * n, (i + 1) * n
        if i == (n_parts - 1):
            idx_end = N
        xyz_query_part = xyz_query[:, idx_start:idx_end]
        # matrix with pairwise distances between xyz_query_part and xyz
        pairwise_distances = (
            torch.sum(xyz_query_part ** 2, dim=2, keepdim=True)
            + torch.sum(xyz ** 2, dim=2, keepdim=True).transpose(2, 1)
            - 2 * torch.matmul(xyz_query_part, xyz.transpose(2, 1))
        )
        # lowest K distances
        ret = pairwise_distances.topk(k=n_neighbors, dim=2, largest=False)
        neighbors[:, idx_start:idx_end, :] = ret.indices
        distances_sq[:, idx_start:idx_end, :] = ret.values.clamp(min=0)
    return neighbors, distances_sq


# def knn_kdtree(
#     xyz: torch.Tensor, xyz_query: torch.Tensor, n_neighbors: int
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """K Nearest Neighbor search based on k-d tree space partitioning.
#     This is the implementation from torch-points-kernels
#     Uses the implementation of torch_points_kernels on CPU.
#     Searches for each point in xyz_query the n_neighbors closest points
#     in xyz.
#     :param xyz: Support point coordinates (B, N', 3).
#     :param xyz_query: Query point coordinates (B, N, 3).
#     :param n_neighbors: Number of neighbors to search.
#     :return: Tensor with neighbor indices and tensor with corresponding
#              distances (B, N, K).
#     """
#
#     neighbors, distances_sq = knn_tpk(
#         xyz.cpu().contiguous(), xyz_query.cpu().contiguous(), n_neighbors
#     )
#     return neighbors, distances_sq


def knn_approximate(
    xyz: torch.Tensor, xyz_query: torch.Tensor, n_neighbors: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Approximate K Nearest Neighbor search using the FAISS implementation.
    Searches for each point in xyz_query the n_neighbors closest points
    in xyz.
    :param xyz: Support point coordinates (B, N', 3).
    :param xyz_query: Query point coordinates (B, N, 3).
    :param n_neighbors: Number of neighbors to search.
    :return: Tensor with neighbor indices and tensor with corresponding
             distances (B, N, K).
    """
    B = xyz.size(0)
    N = xyz_query.size(1)
    xyz_cpu = xyz.cpu()
    xyz_query_cpu = xyz_query.cpu()

    def knn_faiss_single_batch(
        xyz: torch.Tensor, xyz_query: torch.Tensor, n_neighbors: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single batch faiss evaluation.
        :param xyz: Support point coordinates (N', 3).
        :param xyz_query: Query point coordinates (N, 3).
        :param n_neighbors: Number of neighbors to search.
        :return: Tensor with neighbor indices and tensor with corresponding
                 distances (N, K).
        """
        xyz_n = xyz.float().numpy()
        xyz_query_n = xyz_query.float().numpy()
        index = faiss.IndexFlatL2(3)
        # split up metric space in number of Voranoid cells
        index = faiss.IndexIVFFlat(index, 3, max(xyz_n.shape[0] // 400, 1))
        # KNN is performed in 2 cells
        index.nprobe = 2
        index.train(xyz_n)
        index.add(xyz_n)
        ret = index.search(xyz_query_n, n_neighbors)
        assert -1 not in ret[1], "Could not determine KNN!"
        return torch.from_numpy(ret[1]), torch.from_numpy(ret[0])

    if B == 1:
        neighbors, distances_sq = knn_faiss_single_batch(
            xyz_cpu[0], xyz_query_cpu[0], n_neighbors
        )
        return neighbors.unsqueeze(0), distances_sq.unsqueeze(0)
    else:
        # split up per batch (FAISS is single batch only)
        neighbors = torch.empty(
            (B, N, n_neighbors), dtype=torch.int64, device=xyz_cpu.device
        )
        distances_sq = torch.empty(
            (B, N, n_neighbors), dtype=torch.float32, device=xyz_cpu.device
        )
        for b in range(B):
            ret = knn_faiss_single_batch(
                xyz_cpu[b], xyz_query_cpu[b], n_neighbors
            )
            neighbors[b] = ret[0]
            distances_sq[b] = ret[1]
        return neighbors, distances_sq
