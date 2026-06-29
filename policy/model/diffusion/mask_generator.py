from __future__ import annotations

import torch

from policy.model.common.module_attr_mixin import ModuleAttrMixin


class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        max_n_obs_steps: int = 2,
        fix_obs_steps: bool = True,
        action_visible: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape: tuple[int, int, int], seed: int | None = None) -> torch.Tensor:
        device = self.device
        batch_size, horizon, dim = shape
        if dim != self.action_dim + self.obs_dim:
            raise ValueError(f"Expected dim {self.action_dim + self.obs_dim}, got {dim}.")

        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(seed)

        dim_mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[..., : self.action_dim] = True
        is_obs_dim = ~is_action_dim

        if self.fix_obs_steps:
            obs_steps = torch.full((batch_size,), fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1,
                high=self.max_n_obs_steps + 1,
                size=(batch_size,),
                generator=rng,
                device=device,
            )

        steps = torch.arange(0, horizon, device=device).reshape(1, horizon).expand(batch_size, horizon)
        obs_mask = (steps.T < obs_steps).T.reshape(batch_size, horizon, 1).expand(batch_size, horizon, dim)
        mask = obs_mask & is_obs_dim

        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1,
                torch.tensor(0, dtype=obs_steps.dtype, device=obs_steps.device),
            )
            action_mask = (steps.T < action_steps).T.reshape(batch_size, horizon, 1).expand(batch_size, horizon, dim)
            mask = mask | (action_mask & is_action_dim)

        return mask
