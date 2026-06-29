from __future__ import annotations

import numpy as np
import torch


def voxel_downsample_indices(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """Return indices for the first point that falls into each voxel."""
    if voxel_size <= 0 or points.shape[0] == 0:
        return torch.arange(points.shape[0], dtype=torch.long, device=points.device)

    coords = torch.floor(points.detach().cpu() / voxel_size).numpy().astype(np.int64)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    unique_indices.sort()
    return torch.as_tensor(unique_indices, dtype=torch.long, device=points.device)


def voxel_downsample(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """Average points that fall into each voxel, matching Open3D voxel_down_sample."""
    if voxel_size <= 0 or points.shape[0] == 0:
        return points

    coords = torch.floor(points / voxel_size).long()
    _, inverse = torch.unique(coords, dim=0, return_inverse=True)
    downsampled = torch.zeros(
        (int(inverse.max().item()) + 1, points.shape[1]),
        dtype=points.dtype,
        device=points.device,
    )
    downsampled.index_add_(0, inverse, points)
    counts = torch.bincount(inverse, minlength=downsampled.shape[0]).to(
        dtype=points.dtype,
        device=points.device,
    )
    return downsampled / counts.unsqueeze(1)


def farthest_point_indices(points: torch.Tensor, num_points: int) -> torch.Tensor:
    """Return pure-Torch farthest point sample indices."""
    if points.shape[0] <= num_points:
        return torch.arange(points.shape[0], dtype=torch.long, device=points.device)

    selected = torch.empty(num_points, dtype=torch.long, device=points.device)
    min_dist = torch.full((points.shape[0],), torch.inf, device=points.device)
    farthest = torch.tensor(0, dtype=torch.long, device=points.device)

    for i in range(num_points):
        selected[i] = farthest
        centroid = points[farthest].view(1, -1)
        dist = torch.sum((points - centroid) ** 2, dim=1)
        min_dist = torch.minimum(min_dist, dist)
        farthest = torch.argmax(min_dist)

    return selected


def farthest_point_sample(points: torch.Tensor, num_points: int) -> torch.Tensor:
    """Pure-Torch farthest point sampling with a deterministic first point."""
    return points[farthest_point_indices(points, num_points=num_points)]


def downsample_pointcloud_indices(
    points: torch.Tensor,
    num_points: int = 4096,
    voxel_size: float = 0.01,
    prefilter_factor: int = 8,
) -> torch.Tensor:
    """Return source indices for voxel filtering and fixed-size sampling."""
    voxel_indices = voxel_downsample_indices(points, voxel_size=voxel_size)
    candidates = points[voxel_indices]
    if candidates.shape[0] <= num_points:
        return voxel_indices

    max_candidates = num_points * prefilter_factor
    if candidates.shape[0] > max_candidates:
        prefilter_indices = torch.linspace(
            0,
            candidates.shape[0] - 1,
            max_candidates,
            dtype=torch.long,
            device=points.device,
        )
        voxel_indices = voxel_indices[prefilter_indices]
        candidates = candidates[prefilter_indices]

    sampled_indices = farthest_point_indices(candidates.float(), num_points=num_points)
    return voxel_indices[sampled_indices]


def downsample_pointcloud(
    points: torch.Tensor,
    num_points: int = 4096,
    voxel_size: float = 0.01,
    prefilter_factor: int = 8,
) -> torch.Tensor:
    """Voxel-filter then sample a fixed-size point cloud."""
    candidates = voxel_downsample(points, voxel_size=voxel_size)
    if candidates.shape[0] <= num_points:
        return candidates

    max_candidates = num_points * prefilter_factor
    if candidates.shape[0] > max_candidates:
        prefilter_indices = torch.linspace(
            0,
            candidates.shape[0] - 1,
            max_candidates,
            dtype=torch.long,
            device=points.device,
        )
        candidates = candidates[prefilter_indices]

    return farthest_point_sample(candidates.float(), num_points=num_points).to(points.dtype)


def voxelize_points(
    points: torch.Tensor,
    center: torch.Tensor,
    scale: float,
    resolution: int = 64,
) -> np.ndarray:
    """Convert world-space points to normalized unique voxel centers."""
    vertices = torch.clamp((points - center) / scale, -0.5 + 1e-6, 0.5 - 1e-6)
    indices = torch.floor((vertices + 0.5) * resolution).long()
    indices = torch.clamp(indices, 0, resolution - 1)
    unique_indices = torch.unique(indices, dim=0)
    voxel_centers = (unique_indices.float() + 0.5) / resolution - 0.5
    return voxel_centers.detach().cpu().numpy()
