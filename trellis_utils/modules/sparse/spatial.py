from typing import *
import torch
import torch.nn as nn
from . import SparseTensor

__all__ = [
    'SparseDownsample',
    'SparseUpsample',
    'SparseSubdivide',
    'sparse_patchify',
]


class SparseDownsample(nn.Module):
    """
    Downsample a sparse tensor by a factor of `factor`.
    Implemented as average pooling.
    """
    def __init__(self, factor: Union[int, Tuple[int, ...], List[int]]):
        super(SparseDownsample, self).__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert DIM == len(factor), 'Input coordinates must have the same dimension as the downsample factor.'

        coord = list(input.coords.unbind(dim=-1))
        for i, f in enumerate(factor):
            coord[i+1] = coord[i+1] // f

        MAX = [coord[i+1].max().item() + 1 for i in range(DIM)]
        OFFSET = torch.cumprod(torch.tensor(MAX[::-1]), 0).tolist()[::-1] + [1]
        code = sum([c * o for c, o in zip(coord, OFFSET)])
        code, idx = code.unique(return_inverse=True)

        new_feats = torch.scatter_reduce(
            torch.zeros(code.shape[0], input.feats.shape[1], device=input.feats.device, dtype=input.feats.dtype),
            dim=0,
            index=idx.unsqueeze(1).expand(-1, input.feats.shape[1]),
            src=input.feats,
            reduce='mean'
        )
        new_coords = torch.stack(
            [code // OFFSET[0]] +
            [(code // OFFSET[i+1]) % MAX[i] for i in range(DIM)],
            dim=-1
        )
        out = SparseTensor(new_feats, new_coords, input.shape,)
        out._scale = tuple([s // f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache

        out.register_spatial_cache(f'upsample_{factor}_coords', input.coords)
        out.register_spatial_cache(f'upsample_{factor}_layout', input.layout)
        out.register_spatial_cache(f'upsample_{factor}_idx', idx)

        return out


class SparseUpsample(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as nearest neighbor interpolation.
    """
    def __init__(self, factor: Union[int, Tuple[int, int, int], List[int]]):
        super(SparseUpsample, self).__init__()
        self.factor = tuple(factor) if isinstance(factor, (list, tuple)) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        DIM = input.coords.shape[-1] - 1
        factor = self.factor if isinstance(self.factor, tuple) else (self.factor,) * DIM
        assert DIM == len(factor), 'Input coordinates must have the same dimension as the upsample factor.'

        new_coords = input.get_spatial_cache(f'upsample_{factor}_coords')
        new_layout = input.get_spatial_cache(f'upsample_{factor}_layout')
        idx = input.get_spatial_cache(f'upsample_{factor}_idx')
        if any([x is None for x in [new_coords, new_layout, idx]]):
            raise ValueError('Upsample cache not found. SparseUpsample must be paired with SparseDownsample.')
        new_feats = input.feats[idx]
        out = SparseTensor(new_feats, new_coords, input.shape, new_layout)
        out._scale = tuple([s * f for s, f in zip(input._scale, factor)])
        out._spatial_cache = input._spatial_cache
        return out
    
class SparseSubdivide(nn.Module):
    """
    Upsample a sparse tensor by a factor of `factor`.
    Implemented as nearest neighbor interpolation.
    """
    def __init__(self):
        super(SparseSubdivide, self).__init__()

    def forward(self, input: SparseTensor) -> SparseTensor:
        DIM = input.coords.shape[-1] - 1
        # upsample scale=2^DIM
        n_cube = torch.ones([2] * DIM, device=input.device, dtype=torch.int)
        n_coords = torch.nonzero(n_cube)
        n_coords = torch.cat([torch.zeros_like(n_coords[:, :1]), n_coords], dim=-1)
        factor = n_coords.shape[0]
        assert factor == 2 ** DIM
        # print(n_coords.shape)
        new_coords = input.coords.clone()
        new_coords[:, 1:] *= 2
        new_coords = new_coords.unsqueeze(1) + n_coords.unsqueeze(0).to(new_coords.dtype)
        
        new_feats = input.feats.unsqueeze(1).expand(input.feats.shape[0], factor, *input.feats.shape[1:])
        out = SparseTensor(new_feats.flatten(0, 1), new_coords.flatten(0, 1), input.shape)
        out._scale = input._scale * 2
        out._spatial_cache = input._spatial_cache
        return out


def sparse_patchify(x: SparseTensor, patch_size: int) -> SparseTensor:
    """
    Patchify a sparse tensor.
    
    Args:
        x (SparseTensor): coords (N, 4), feats (N, C)
        patch_size (int): Patch size
    """

    DIM = x.coords.shape[-1] - 1
    nc = x.feats.shape[1]
    batch_idx = x.coords[:, 0]
    x_coords = x.coords[:, 1:]

    # split coords into patch idx + intra-patch offset
    patch_coord = x_coords // patch_size
    sub_coord = x_coords % patch_size

    # hash patch_coord to a unique 1-D code (grouping key)
    MAX = (patch_coord.max(0).values + 1).tolist()
    # OFFSET = torch.cumprod(torch.tensor(MAX[::-1], device=x.device), 0).flip(0)
    OFFSET = torch.cumprod(torch.tensor(MAX[::-1], device=x.device), 0).tolist()[::-1] + [1]
    # code = batch_idx * OFFSET[0] + (patch_coord * OFFSET[1:]).sum(-1) # (N, )
    code = batch_idx * OFFSET[0] + sum([patch_coord[:, i] * OFFSET[i+1] for i in range(DIM)]) # (N, )
    code, idx = code.unique(return_inverse=True)

    # linearise sub-patch position into channel offset
    offset = sub_coord[:, 0]
    for d in range(1, DIM):
        offset = offset * patch_size + sub_coord[:, d]
    channel_offset = offset * nc

    patch_dim = nc * patch_size ** DIM
    patch_num = code.shape[0]
    linear_idx = idx.unsqueeze(1) * patch_dim + channel_offset.unsqueeze(1) + torch.arange(nc, device=x.device).unsqueeze(0)
    linear_idx = linear_idx.reshape(-1)

    # write features
    out_feats_flat = torch.zeros(patch_num * patch_dim, device=x.device, dtype=x.feats.dtype)
    out_feats_flat.index_add_(0, linear_idx, x.feats.reshape(-1))
    out_feats = out_feats_flat.view(patch_num, patch_dim)

    # decode spatial part back to coordinate grid
    patch_coords = torch.cat(
        [(code // OFFSET[0]).unsqueeze(1), 
         torch.stack([(code // OFFSET[i+1]) % MAX[i] for i in range(DIM)], dim=1)],
        dim=1
    )
    out = SparseTensor(out_feats, patch_coords)
    return out
