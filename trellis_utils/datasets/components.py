from typing import *
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import default_collate
from ..modules.sparse import SparseTensor


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
    ):
        super().__init__()
        self.roots = roots.split(',')
        self.instances = []
        self.metadata = pd.DataFrame()
        
        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
            self._stats[key]['Total'] = len(metadata)
            metadata, stats = self.filter_metadata(metadata)
            self._stats[key].update(stats)
            self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
            metadata.set_index('sha256', inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    @abstractmethod
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class TextConditionedMixin:
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        self.captions = {}
        for instance in self.instances:
            sha256 = instance[1]
            self.captions[sha256] = json.loads(self.metadata.loc[sha256]['captions'])
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['captions'].notna()]
        stats['With captions'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        text = np.random.choice(self.captions[instance])
        pack['cond'] = text
        return pack
    
    
class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root, 'renders_cond', instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
       
        return pack
    

class CustomDatasetBase(Dataset):
    """
    Dataset structure:
    ├── dataset_1
    │   ├── scene_0000
    │   │   ├── frame_0000
    │   │   │   ├── images
    │   │   │   │   ├── view_0000.png
    │   │   │   │   ├── view_0001.png
    │   │   │   │   ├── ...
    │   │   │   ├── metadata.json
    │   │   │   ├── voxels.ply
    │   │   │   ├── ss_latents.npz
    │   │   │   ├── latents.npz
    │   │   │   ├── feature_dinov2_vitl14_reg.npz
    │   │   ├── frame_0001
    │   │   ├── ...
    │   ├── scene_0001
    │   ├── ...
    ├── dataset_2
    ├── ...
    """

    def __init__(self, roots: str,):
        super().__init__()
        # self.roots = roots.split(',')
        roots = roots.split(',')
        self.roots = []
        self.instances = []
        self.has_complete_task = False # TODO: make this configurable
        
        def _get_subtask(basedir):
            subtask_names = [subtask_name for subtask_name in sorted(os.listdir(basedir)) if subtask_name.startswith('task_')]
            if len(subtask_names) == 0:
                return [basedir]
            else:
                return [os.path.join(basedir, subtask_name) for subtask_name in subtask_names]

        for root in roots:
            self.roots.extend(_get_subtask(root))

        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            if key.startswith('task_'):
                key = os.path.basename(os.path.dirname(root)) + '_' + key
            self._stats[key] = {}

            packed_ids = []
            scene_ids = sorted(os.listdir(root))
            if True: # hold_out for validation
                hold_out = 0.05
                num_keep = int(len(scene_ids) * (1 - hold_out))
                scene_ids = scene_ids[:num_keep]
            for scene_id in scene_ids:
                # packed_ids.extend([(scene_id, frame_id) for frame_id in sorted(os.listdir(os.path.join(root, scene_id)))])
                frame_ids = [frame_id for frame_id in sorted(os.listdir(os.path.join(root, scene_id))) if frame_id.startswith('frame_')]
                packed_ids.extend([(scene_id, frame_ids[i+1], frame_ids[i]) for i in range(len(frame_ids)-1)])
                # packed_ids.extend([(scene_id, frame_ids[i+4], frame_ids[i]) for i in range(len(frame_ids)-4)])
                self.instances.extend([(root, scene_id, frame_ids[i+1], frame_ids[i]) for i in range(len(frame_ids)-1)])
                # self.instances.extend([(root, scene_id, frame_ids[i+4], frame_ids[i]) for i in range(len(frame_ids)-4)])
            
            self._stats[key]['Total'] = len(packed_ids)
    
            if self.has_complete_task:
                self._stats[key+'_complete'] = {}
                packed_ids_complete = []
                for scene_id in scene_ids:
                    frame_ids = [frame_id for frame_id in sorted(os.listdir(os.path.join(root, scene_id))) if frame_id.startswith('frame_')]
                    frame_id = np.random.choice(frame_ids)
                    packed_ids_complete.append((root, scene_id, frame_id, frame_id))
                self.instances.extend(packed_ids_complete)
                self._stats[key+'_complete']['Total'] = len(packed_ids_complete)

    @abstractmethod
    def filter_metadata(self, metadata):
        raise NotImplementedError("filter_metadata method not suitable for CustomDatasetBase")
    
    @abstractmethod
    def get_instance(self, root: str, scene_id: str, frame_id: str, frame_cond_id: str = None) -> Dict[str, Any]:
        pass

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, scene_id, frame_id, frame_cond_id = self.instances[index]
            return self.get_instance(root, scene_id, frame_id, frame_cond_id)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class SlatConditionedMixin:
    def __init__(self, roots, cond_normalization: Optional[Dict] = None, **kwargs):
        
        self.cond_normalization = cond_normalization
        if cond_normalization is not None:
            self.cond_mean = torch.tensor(cond_normalization['mean']).reshape(1, -1)
            self.cond_std = torch.tensor(cond_normalization['std']).reshape(1, -1)
        super().__init__(roots, **kwargs)
        
    def get_instance(self, root, scene_id, frame_id, frame_cond_id):
        pack = super().get_instance(root, scene_id, frame_id, frame_cond_id)

        slat_cond_path = os.path.join(root, scene_id, frame_cond_id, 'latents.npz')
        data = np.load(slat_cond_path)
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()

        if self.cond_normalization is not None:
            feats = (feats - self.cond_mean) / self.cond_std

        if self.has_complete_task and (frame_id == frame_cond_id):
            raise NotImplementedError("Complete task not supported for SlatConditionedMixin, please use a mixin with text conditioning.")

        pack['cond'] = SparseTensor(
            feats=feats,
            coords=torch.cat([torch.full((coords.shape[0], 1), 0, dtype=torch.int32), coords], dim=-1),
        )
        return pack

    @staticmethod
    def collate_fn(batch):
        # only consider SparseTensor case
        packs = {}
        keys = batch[0].keys()
        for key in keys:
            if isinstance(batch[0][key], SparseTensor):
                packs[key] = [iter_batch[key] for iter_batch in batch]
            else:
                packs[key] = default_collate([iter_batch[key] for iter_batch in batch])
        return packs
        

class SlatTextConditionedMixin:
    def __init__(self, roots, cond_normalization: Optional[Dict] = None, **kwargs):
        
        self.cond_normalization = cond_normalization
        if cond_normalization is not None:
            self.cond_mean = torch.tensor(cond_normalization['mean']).reshape(1, -1)
            self.cond_std = torch.tensor(cond_normalization['std']).reshape(1, -1)
        super().__init__(roots, **kwargs)
    
    def get_instance(self, root, scene_id, frame_id, frame_cond_id):
        pack = super().get_instance(root, scene_id, frame_id, frame_cond_id)

        # load slat condition
        slat_cond_path = os.path.join(root, scene_id, frame_cond_id, 'latents.npz')
        data = np.load(slat_cond_path)
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()

        if self.cond_normalization is not None:
            feats = (feats - self.cond_mean) / self.cond_std

        slat_cond = SparseTensor(
            feats=feats,
            coords=torch.cat([torch.full((coords.shape[0], 1), 0, dtype=torch.int32), coords], dim=-1),
        )

        # load text condition
        text_cond_path = os.path.join(root, scene_id, 'instruction.txt')
        if not os.path.exists(text_cond_path):
            raise FileNotFoundError(f"Text condition file not found: {text_cond_path}")
        with open(text_cond_path, 'r') as f:
            lines = f.readlines()
        text_cond = np.random.choice(lines)
        text_cond = text_cond.replace('\n', '')
        if self.has_complete_task and (frame_id == frame_cond_id):
            text_cond = 'Complete the scene with partial observations.'
        pack['cond'] = [slat_cond, text_cond]

        return pack
    
    @staticmethod
    def collate_fn(batch):
        packs = {}
        keys = batch[0].keys()
        for key in keys:
            if key not in ['cond']:
                packs[key] = default_collate([iter_batch[key] for iter_batch in batch])
        packs['cond'] = [iter_batch['cond'] for iter_batch in batch]
        return packs


class InterpSparseStructureDataset(Dataset):
    def __init__(self, roots,):
        super().__init__()
        self.roots = roots.split(',')

        self.instances = []
        for root in self.roots:
            scene_ids = [scene_id for scene_id in sorted(os.listdir(root)) if scene_id.startswith('scene_')]
            if True: # hold_out for validation
                hold_out = 0.05
                num_keep = int(len(scene_ids) * (1 - hold_out))
                scene_ids = scene_ids[:num_keep]
            for scene_id in scene_ids:
                frame_ids = [frame_id for frame_id in sorted(os.listdir(os.path.join(root, scene_id))) if frame_id.startswith('frame_')]
                self.instances.append((root, scene_id, frame_ids))
        
    @abstractmethod
    def filter_metadata(self, metadata):
        raise NotImplementedError("filter_metadata method not suitable for InterpConditionedDataset")
    
    @abstractmethod
    def get_instance(self, root: str, scene_id: str, frame_ids: List[str]):
        pass

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, scene_id, frame_ids = self.instances[index]
            return self.get_instance(root, scene_id, frame_ids)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for root in self.roots:
            lines.append(f'    - {os.path.basename(root)}')
        return '\n'.join(lines)
