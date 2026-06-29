from typing import *
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import torch
import numpy as np
from transformers import AutoTokenizer, CLIPTextModel

from ....utils import dist_utils
from ....modules.transformer import AbsolutePositionEmbedder
from ....modules.sparse import sparse_patchify, SparseTensor


def add_random_noise(x: Union[torch.Tensor, SparseTensor], sigma_min=1e-5, sigma_max=0.25) -> Union[torch.Tensor, SparseTensor]:
    t = torch.rand(1, device=x.device) * sigma_max + sigma_min
    if isinstance(x, torch.Tensor):
        noise = torch.randn_like(x)
        return (1 - t) * x + t * noise
    elif isinstance(x, SparseTensor):
        noise = torch.randn_like(x.feats)
        noisy_feats = (1 - t) * x.feats + t * noise
        return SparseTensor(
            coords=x.coords,
            feats=noisy_feats,
        )


def drop_random_coords(x: SparseTensor, max_drop_ratio: float = 1.) -> SparseTensor:
    drop_ratio = np.random.rand() * max_drop_ratio
    num_coords = x.coords.shape[0]
    keep_idx = np.random.rand(num_coords) > drop_ratio
    if keep_idx.sum() == 0:
        keep_idx[0] = True  # ensure at least one coordinate is kept
    keep_idx = torch.from_numpy(keep_idx).to(x.coords.device)
    return SparseTensor(
        coords=x.coords[keep_idx],
        feats=x.feats[keep_idx],
    )


class SlatConditionedMixin:
    """
    Mixin for slat-conditioned models.
    
    Args:
        slat_cond_model: The slat conditioning model.
    """
    def __init__(self, *args, slat_patch_size, base_channel, keep_sparse=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = slat_patch_size
        self.patched_channel = base_channel * self.patch_size ** 3
        self.pos_embedder = AbsolutePositionEmbedder(self.patched_channel)
        self.keep_sparse = keep_sparse

    @torch.no_grad()
    def encode_slat(self, slats: List[SparseTensor]) -> Union[torch.Tensor, List[SparseTensor]]:
        # return [sparse_patchify(slat, self.patch_size) for slat in slats]
        if self.keep_sparse:
            conds = []
            for slat in slats:
                slat_patched = sparse_patchify(slat.cuda(), self.patch_size)
                cond_feats = slat_patched.feats + self.pos_embedder(slat_patched.coords[:, 1:]).type(slat_patched.feats.dtype)
                conds.append(SparseTensor(
                    coords=slat_patched.coords,
                    feats=cond_feats,
                ))
            return conds
        else: # pad to the same length
            conds = []
            for slat in slats:
                slat_patched = sparse_patchify(slat.cuda(), self.patch_size)
                cond = slat_patched.feats + self.pos_embedder(slat_patched.coords[:, 1:]).type(slat_patched.feats.dtype)
                conds.append(cond)
            max_len = max([cond.shape[0] for cond in conds])
            conds_pad = torch.zeros(len(conds), max_len, self.patched_channel, device=conds[0].device, dtype=conds[0].dtype)
            for i, cond in enumerate(conds):
                conds_pad[i, :cond.shape[0], :] = cond
            return conds_pad

    @torch.no_grad()
    def generate_neg_cond(self, conds: Union[torch.Tensor, List[SparseTensor]]) -> Union[torch.Tensor, List[SparseTensor]]:
        if self.keep_sparse: # conds: SparseTensor
            batch_num = len(conds)
            neg_coords = torch.stack(torch.meshgrid([torch.arange(0, 8, device=conds[0].device) * 2 for _ in range(3)], indexing='ij'), dim=-1).reshape(-1, 3).int()
            neg_feats = torch.zeros([neg_coords.shape[0], self.patched_channel], device=conds[0].device, dtype=conds[0].feats.dtype)
            neg_conds = [
                SparseTensor(
                    coords=torch.cat([torch.full((neg_coords.shape[0], 1), 0, dtype=torch.int32).cuda(), neg_coords.clone()], dim=-1),
                    feats=neg_feats.clone(),
                ) for _ in range(batch_num)
            ]
            return neg_conds
        else: # conds: torch.Tensor
            return torch.zeros_like(conds)

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.encode_slat(cond)
        kwargs['neg_cond'] = self.generate_neg_cond(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_slat(cond)
        kwargs['neg_cond'] = self.generate_neg_cond(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond


class SlatTextConditionedMixin:
    """
    Mixin for slat-text-conditioned models.
    """
    def __init__(self, *args, text_cond_model: str = "openai/clip-vit-large-patch14", keep_slat_coords: bool = False, noisy_cond: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_slat_coords = keep_slat_coords
        self.noisy_cond = noisy_cond
        self.text_cond_model_name = text_cond_model
        self.text_cond_model = None     # the model is init lazily

    def _init_text_cond_model(self):
        """
        Initialize the text conditioning model.
        """
        with dist_utils.local_master_first():
            model = CLIPTextModel.from_pretrained(self.text_cond_model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.text_cond_model_name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and isinstance(text[0], str), "TextConditionedMixin only supports list of strings as cond"
        if self.text_cond_model is None:
            self._init_text_cond_model()
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings

    @torch.no_grad()
    def generate_conds(self, conds: List, inference_mode=False) -> List:
        slat_conds = [iter_conds[0].cuda() for iter_conds in conds]
        text_conds = [iter_conds[1] for iter_conds in conds]
        if not inference_mode and self.noisy_cond:
            for i in range(len(slat_conds)):
                if text_conds[i] == 'Complete the scene with partial observations.':
                    slat_conds[i] = drop_random_coords(slat_conds[i], max_drop_ratio=0.75)
                    slat_conds[i] = add_random_noise(slat_conds[i], sigma_max=1.)
                # elif not inference_mode:
                else:
                    slat_conds[i] = drop_random_coords(slat_conds[i], max_drop_ratio=0.25)
                    slat_conds[i] = add_random_noise(slat_conds[i], sigma_max=0.5)
        generated_text_conds = list(self.encode_text(text_conds).unbind(0))
        return slat_conds + generated_text_conds
    
    @torch.no_grad()
    def generate_neg_conds(self, conds: List) -> List:
        batch_size = len(conds) // 2
        slat_conds = conds[:batch_size]
        text_conds = conds[batch_size:]
        neg_text_conds = list(self.text_cond_model['null_cond'].repeat(batch_size, 1, 1).unbind(0))
        if self.keep_slat_coords:
            neg_slat_conds = [SparseTensor(coords=slat.coords, feats=torch.zeros_like(slat.feats)) for slat in slat_conds]
        else:
            channel_dim = slat_conds[0].feats.shape[1]
            neg_coords = torch.stack(torch.meshgrid([torch.arange(0, 4, device=conds[0].device) * 16 for _ in range(3)], indexing='ij'), dim=-1).reshape(-1, 3).int()
            neg_feats = torch.zeros([neg_coords.shape[0], channel_dim], device=conds[0].device, dtype=conds[0].feats.dtype)
            neg_slat_conds = [
                SparseTensor(
                    coords=torch.cat([torch.full((neg_coords.shape[0], 1), 0, dtype=torch.int32).cuda(), neg_coords.clone()], dim=-1),
                    feats=neg_feats.clone(),
                ) for _ in range(batch_size)
            ]
        return neg_slat_conds + neg_text_conds
    
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.generate_conds(cond)
        kwargs['neg_cond'] = self.generate_neg_conds(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.generate_conds(cond, inference_mode=True)
        kwargs['neg_cond'] = self.generate_neg_conds(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond
