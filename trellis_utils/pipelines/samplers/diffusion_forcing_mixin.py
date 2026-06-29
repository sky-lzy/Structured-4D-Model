from typing import *
import torch
import numpy as np
from easydict import EasyDict as edict


class InterpolationMixin:
    """
    A mixin class for samplers that support interpolation by diffusion forcing.
    """

    def _apply_conditioning(self, x_t, context_frames, context_mask):
        """
        Apply context conditioning to the input tensor x_t based on the provided context frames and mask.
        
        Args:
            x_t: The input tensor to condition.
            context_frames: Known frames for conditioning.
            context_mask: Boolean mask indicating which frames are known.
        
        Returns:
            The conditioned tensor.
        """
        x_t_cond = x_t.clone()
        if context_frames is not None and context_mask is not None:
            # Replace known frames with their corresponding context frames
            x_t_cond[context_mask] = context_frames[context_mask]
        return x_t_cond

    def _get_model_prediction(self, model, x_t, context_mask, cond_flag, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, context_mask=context_mask, cond_flag=cond_flag, cond=cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    def _inference_model(self, model, x_t, t, context_mask, cond_flag, cond=None, **kwargs):
        """
        Perform inference with the model, applying context conditioning if necessary.
        
        Args:
            model: The model to use for inference.
            x_t: The input tensor at time t.
            t: The current timestep.
            context_mask: Boolean mask indicating which frames are known.
            cond_flag: Whether to apply conditional guidance.
            cond: Additional conditional information.
            **kwargs: Additional arguments for model inference.
        
        Returns:
            Model predictions (x_0, eps, velocity).
        """
        t = torch.tensor([[1000 * t]], device=x_t.device, dtype=torch.float32).repeat(x_t.shape[0], x_t.shape[1])
        if cond_flag: # with condition
            t[context_mask] = 0
        else: # without condition
            t[context_mask] = 1000
        return model(x_t, t, cond, **kwargs)

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond=None,
        context_mask=None,
        context_frames=None,
        history_guidance=True,
        history_weight=0.1,
        **kwargs
    ):
        """
        Sample with proper History Guidance using diffusion forcing.
        
        This implements the actual History Guidance from the DFoT paper:
        - Multiple model evaluations with different history masks
        - CFG-style composition of conditional and unconditional scores
        
        Args:
            model: The model to sample from.
            x_t: The [N x T x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            context_mask: [N x T] boolean mask indicating which timesteps are known.
            context_frames: [N x T x C x ...] known frames (usually start and end).
            history_guidance: whether to apply history guidance for temporal consistency.
            history_weight: guidance scale for CFG-style composition (like CFG scale).
            **kwargs: Additional arguments for model inference.
        """
        
        # Default context mask: first and last frames are known
        if context_mask is None:
            context_mask = torch.zeros(x_t.shape[0], x_t.shape[1], dtype=torch.bool, device=x_t.device)
            context_mask[:, [0, -1]] = True  # Assume first and last timesteps are known

        if history_guidance and history_weight > 0 and torch.any(context_mask):
            # PROPER HISTORY GUIDANCE: CFG-style composition
            
            # 1. x_t_cond: history frames are conditioned (already done above)
            x_t_cond = self._apply_conditioning(x_t, context_frames, context_mask)

            # 2. x_t_uncond: fully mask the history frames (unconditional on history)
            full_noise = torch.randn_like(x_t)
            # x_t_uncond = self._make_fully_masked(x_t, context_mask, t)
            x_t_uncond = self._apply_conditioning(x_t, full_noise, context_mask)
            
            # 3. Get model predictions for both inputs
            pred_x_0_cond, pred_eps_cond, pred_v_cond = self._get_model_prediction(model, x_t_cond, context_mask, True, t, cond=cond, **kwargs)
            pred_x_0_uncond, pred_eps_uncond, pred_v_uncond = self._get_model_prediction(model, x_t_uncond, context_mask, False, t, cond=cond, **kwargs)

            # 4. CFG-style composition (this is the actual History Guidance)
            guidance_scale = float(history_weight)
            pred_v = pred_v_uncond + guidance_scale * (pred_v_cond - pred_v_uncond)
            pred_x_0 = pred_x_0_uncond + guidance_scale * (pred_x_0_cond - pred_x_0_uncond)
            pred_eps = pred_eps_uncond + guidance_scale * (pred_eps_cond - pred_eps_uncond)
            
        else:
            # No history guidance: single model evaluation
            dummy_mask = torch.zeros_like(context_mask) if context_mask is not None else torch.zeros(x_t.shape[0], x_t.shape[1], dtype=torch.bool, device=x_t.device)
            pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, dummy_mask, True, t, cond=cond, **kwargs)

        # Euler step
        pred_x_prev = x_t - (t - t_prev) * pred_v
        
        # Re-apply context conditioning to the result to ensure known frames remain fixed
        if context_frames is not None:
            pred_x_prev = self._apply_conditioning(pred_x_prev, context_frames, context_mask)

        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})
