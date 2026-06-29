from typing import Dict
import torch
from policy.model.common.module_attr_mixin import ModuleAttrMixin
from policy.model.common.normalizer import LinearNormalizer

class BaseImagePolicy(ModuleAttrMixin):
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def reset(self):
        pass

    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
