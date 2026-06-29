import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import json
from typing import *
import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPTextModel, AutoTokenizer

from .base import Pipeline
from . import samplers
from .. import models
from ..modules import sparse as sp

SUPPORTED_DECODE_FORMATS = ("gaussian",)

class TrellisSDM3DGenPipeline(Pipeline):
    """
    Pipeline for unrolling Trellis slat-conditioned models.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_text_cond_model(text_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisSDM3DGenPipeline":
        """
        Load a pretrained model.
        """
        pipeline = super(TrellisSDM3DGenPipeline, TrellisSDM3DGenPipeline).from_pretrained(path)
        new_pipeline = TrellisSDM3DGenPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']
        new_pipeline.rembg_session = None
        new_pipeline._init_text_cond_model(args.get('text_cond_model'))

        return new_pipeline
    
    @staticmethod
    def from_config(config: dict) -> "TrellisSDM3DGenPipeline":
        """
        Load a pipeline from a config.
        """
        assert config['name'] == 'TrellisSDM3DGenPipeline', "Pipeline name mismatch."
        pipeline_args = config['args']

        # load models
        _models = {}
        for k, v in pipeline_args['models'].items():
            if isinstance(v, str): # load from pretrained
                _models[k] = models.from_pretrained(v)
            elif isinstance(v, dict): # create and load params from path
                _model = getattr(models, v['name'])(**v['args'])
                _model.load_state_dict(torch.load(v['load_from'], map_location='cpu', weights_only=True))
                _models[k] = _model

        pipeline = TrellisSDM3DGenPipeline(_models, text_cond_model=pipeline_args['text_cond_model'])

        pipeline.sparse_structure_sampler = getattr(samplers, pipeline_args['sparse_structure_sampler']['name'])(**pipeline_args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = pipeline_args['sparse_structure_sampler']['params']

        pipeline.slat_sampler = getattr(samplers, pipeline_args['slat_sampler']['name'])(**pipeline_args['slat_sampler']['args'])
        pipeline.slat_sampler_params = pipeline_args['slat_sampler']['params']

        pipeline.slat_normalization = pipeline_args['slat_normalization']

        return pipeline

    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        if name is None:
            return
        
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
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
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        return embeddings

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given slat conditioning.
        """
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=verbose
        ).samples

        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()

        return coords
    
    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: Sequence[str] = SUPPORTED_DECODE_FORMATS,
    ) -> dict:
        """
        Decode the structured latent.
        """
        unsupported = tuple(format for format in formats if format not in SUPPORTED_DECODE_FORMATS)
        if unsupported:
            supported = ", ".join(SUPPORTED_DECODE_FORMATS)
            requested = ", ".join(unsupported)
            raise ValueError(f"Unsupported decode format(s): {requested}. Supported format(s): {supported}.")

        ret = {}
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
        verbose: bool = True,
    ) -> sp.SparseTensor:
        """
        Sample structured latents with the given slat conditioning.
        """
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=verbose
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    @staticmethod
    def load_data(data_path) -> sp.SparseTensor:
        cond_data = np.load(data_path)
        coords = torch.tensor(cond_data['coords']).int()
        feats = torch.tensor(cond_data['feats']).float()
        slat = sp.SparseTensor(
            feats=feats,
            coords=torch.cat([torch.full((coords.shape[0], 1), 0, dtype=torch.int32), coords], dim=-1),
        ).cuda()
        return slat
    
    @torch.no_grad()
    def get_cond(self, slat_cond: sp.SparseTensor, text_instruction: str) -> dict:
        """
        Get the slat condition for sampling.
        """
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat_cond.device)
        std = torch.tensor(self.slat_normalization['std'])[None].to(slat_cond.device)
        slat_cond = (slat_cond - mean) / std

        neg_slat_cond_keepcoords = sp.SparseTensor(
            coords=slat_cond.coords.clone(),
            feats=torch.zeros_like(slat_cond.feats),
        )

        channel_dim = slat_cond.feats.shape[-1]
        neg_coords = torch.stack(torch.meshgrid([torch.arange(0, 4, device=slat_cond.device) * 16 for _ in range(3)], indexing='ij'), dim=-1).reshape(-1, 3).int()
        neg_feats = torch.zeros([neg_coords.shape[0], channel_dim], device=slat_cond.device, dtype=slat_cond.feats.dtype)
        neg_slat_cond_randcoords = sp.SparseTensor(
            coords=torch.cat([torch.full((neg_coords.shape[0], 1), 0, dtype=torch.int32).cuda(), neg_coords], dim=-1),
            feats=neg_feats,
        )
        
        text_cond = self.encode_text([text_instruction])
        neg_text_cond = self.text_cond_model['null_cond'].expand(text_cond.shape[0], -1, -1)

        return {
            'cond': [slat_cond, text_cond],
            'neg_cond_keepcoords': [neg_slat_cond_keepcoords, neg_text_cond],
            'neg_cond_randcoords': [neg_slat_cond_randcoords, neg_text_cond],
        }

    def generate_onestep(self, prev_slat: sp.SparseTensor, ss_sampler_params: dict = {}, slat_sampler_params: dict = {}, text_instruction: str = None, verbose: bool = True) -> sp.SparseTensor:
        """
        One-step generation of the pipeline.
        """
        cond = self.get_cond(prev_slat, text_instruction)
        cond_keepcoords = {
            'cond': cond['cond'],
            'neg_cond': cond['neg_cond_keepcoords'],
        }
        cond_randcoords = {
            'cond': cond['cond'],
            'neg_cond': cond['neg_cond_randcoords'],
        }

        coords = self.sample_sparse_structure(cond_randcoords, 1, ss_sampler_params, verbose=verbose)
        slat = self.sample_slat(cond_keepcoords, coords, slat_sampler_params, verbose=verbose)
        return slat
        
    def generate_unroll(self, base_slat: sp.SparseTensor, num_step: int = 4, ss_sampler_params: dict = {}, slat_sampler_params: dict = {}, text_instruction: str = None, verbose: bool = True):
        """
        Unroll generation of the pipeline.
        """
        generated_slats = []
        slat = base_slat
        for i in range(num_step):
            if verbose:
                print(f"Sampling step {i+1:02d}/{num_step:02d} ...")
            slat = self.generate_onestep(slat, ss_sampler_params, slat_sampler_params, text_instruction=text_instruction, verbose=verbose)
            generated_slats.append(slat)
        return generated_slats
    
    @torch.no_grad()
    def run(
        self,
        data_path: str,
        num_step: int = 4,
        seed: int = 0,
        ss_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: Sequence[str] = SUPPORTED_DECODE_FORMATS,
        instruction: str = None,
    ) -> list:
        """
        Run the pipeline.
        """
        base_slat = self.load_data(data_path)
        torch.manual_seed(seed)
        generated_slats = self.generate_unroll(base_slat, num_step, ss_sampler_params, slat_sampler_params, text_instruction=instruction)
        decoded_slats = []
        for slat in generated_slats:
            decoded_slats.append(self.decode_slat(slat, formats))
        decoded_gt_slat = self.decode_slat(base_slat, formats)
        
        return decoded_slats, decoded_gt_slat
