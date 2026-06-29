
from typing import *
import torch

from ....modules.transformer import AbsolutePositionEmbedder
from ....modules.spatial import patchify


class InterpConditionedMixin:

    def __init__(self, *args, ss_cond_resolution, ss_cond_channel, ss_cond_patch_size, **kwargs):
        super().__init__(*args, **kwargs)

        self.ss_cond_patch_size = ss_cond_patch_size

        pos_embedder = AbsolutePositionEmbedder(2 * ss_cond_channel)
        coords = torch.meshgrid(*[torch.arange(res) for res in [ss_cond_resolution // ss_cond_patch_size] * 3], indexing='ij')
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        self.pos_emb = pos_embedder(coords).float()

        
    @torch.no_grad()
    def generate_conds(self, conds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate conditioning data for interpolation.
        """
        # conds: [B, 16, 16, 16, 16]
        conds = patchify(conds, self.ss_cond_patch_size)
        conds = conds.view(*conds.shape[:2], -1).permute(0, 2, 1).contiguous() # [B, L, 16]
        neg_conds = torch.zeros_like(conds)
        self.pos_emb = self.pos_emb.to(conds.device)
        conds = conds + self.pos_emb.unsqueeze(0)
        neg_conds = neg_conds + self.pos_emb.unsqueeze(0)
        return conds, neg_conds

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        conds, neg_conds = self.generate_conds(cond)
        kwargs['neg_cond'] = neg_conds
        cond = super().get_cond(conds, **kwargs)
        return cond

    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        conds, neg_conds = self.generate_conds(cond)
        kwargs['neg_cond'] = neg_conds
        cond = super().get_inference_cond(conds, **kwargs)
        return cond

