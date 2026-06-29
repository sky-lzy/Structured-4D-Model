from typing import *
import torch
from easydict import EasyDict as edict


class RepaintMixin:
    """
    A mixin class for samplers that support repainting.
    
    This implements the Repaint technique where known regions (gt_x) are progressively
    noised according to the current timestep and combined with the sampled regions.
    
    The repainting happens AFTER each denoising step, not during model inference.
    """

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond=None,
        gt_x=None,
        gt_mask=None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method with repainting.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            gt_x: Ground truth data for known regions
            gt_mask: Boolean mask indicating known regions (True = known, False = unknown)
                    Shape: (batch_size, channels) where each element indicates if that channel is known
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1} with repainting applied.
            - 'pred_x_0': a prediction of x_0.
        """
        # First, do the normal denoising step
        out = super().sample_once(model, x_t, t, t_prev, cond, **kwargs)
        pred_x_prev = out.pred_x_prev
        
        # Then apply repainting if ground truth is provided
        if gt_x is not None and gt_mask is not None and torch.any(gt_mask):
            pred_x_prev = self._apply_repaint(pred_x_prev, t_prev, gt_x, gt_mask)
        
        # Return the result with repainting applied
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": out.pred_x_0})
    
    def _apply_repaint(self, x_t, t, gt_x, gt_mask):
        """
        Apply repainting by replacing known regions with appropriately noised ground truth.
        
        For flow matching models, this adds noise according to the current timestep t:
        x_t = (1 - t) * x_0 + (sigma_min + (1 - sigma_min) * t) * noise
        
        Args:
            x_t: Current sample at timestep t
            t: Current timestep (scalar between 0 and 1)
            gt_x: Ground truth data  
            gt_mask: Boolean mask for known regions
                    Shape: (batch_size, channels)
            
        Returns:
            Repainted sample with known regions replaced
        """
        if not torch.any(gt_mask):
            return x_t
            
        # Get sigma_min from the sampler (flow matching parameter)
        sigma_min = getattr(self, 'sigma_min', 1e-5)
        
        # Generate noise for all ground truth at once
        noise = torch.randn_like(gt_x)
        
        # Apply forward flow process to ground truth at current timestep
        # x_t = (1 - t) * x_0 + (sigma_min + (1 - sigma_min) * t) * noise
        gt_noised = (1 - t) * gt_x + (sigma_min + (1 - sigma_min) * t) * noise
        
        # Expand gt_mask to match x_t dimensions for broadcasting
        # gt_mask: (batch_size, channels) -> (batch_size, channels, 1, 1, ...)
        mask_shape = gt_mask.shape + (1,) * (len(x_t.shape) - len(gt_mask.shape))
        gt_mask_expanded = gt_mask.view(mask_shape)
        
        # Replace channels where gt_mask is True with noised ground truth
        x_t_repainted = torch.where(gt_mask_expanded, gt_noised, x_t)
        
        return x_t_repainted

