import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import json
import argparse
import torch
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import trellis_utils.models as models


torch.set_grad_enabled(False)


def get_voxels(voxel_path):
    # position = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{instance}.ply'))[0]
    position = utils3d.io.read_ply(voxel_path)[0]
    coords = ((torch.tensor(position) + 0.5) * opt.resolution).int().contiguous()
    ss = torch.zeros(1, opt.resolution, opt.resolution, opt.resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--enc_pretrained', type=str, default='microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str, default='results',
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint to load')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Resolution')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    if opt.enc_model is None:
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.enc_model}_{opt.ckpt}'
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')

    pack_ids = []
    task_names = [task_name for task_name in sorted(os.listdir(opt.output_dir)) if task_name.startswith("task_")]
    if len(task_names) == 0:
        print(f"No task directories found in {opt.output_dir}, directly process the scenes.")
        task_names = [""]
    else:
        print(f"Processing {len(task_names)} tasks.")
    
    for task_name in task_names:
        task_dir = os.path.join(opt.output_dir, task_name)
        scene_names = [scene_name for scene_name in sorted(os.listdir(task_dir)) if scene_name.startswith("scene_")]
        for scene_name in scene_names:
            scene_dir = os.path.join(task_dir, scene_name)
            frame_names = [frame_name for frame_name in sorted(os.listdir(scene_dir)) if frame_name.startswith("frame_")]
            for frame_name in frame_names:
                frame_dir = os.path.join(scene_dir, frame_name)
                target_path = os.path.join(frame_dir, "ss_latents.npz")
                if not os.path.exists(target_path):
                    pack_ids.append(os.path.join(task_name, scene_name, frame_name))

    # encode latents
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=32) as loader_executor, \
            ThreadPoolExecutor(max_workers=32) as saver_executor:
            def loader(pack_id):
                try:
                    # ss = get_voxels(sha256)[None].float()
                    frame_dir = pack_id
                    voxel_path = os.path.join(opt.output_dir, frame_dir, "voxels.ply")
                    ss = get_voxels(voxel_path)[None].float()
                    load_queue.put((frame_dir, ss))
                except Exception as e:
                    print(f"Error loading features for {frame_dir}: {e}")
            loader_executor.map(loader, pack_ids)
            
            def saver(frame_dir, pack):
                # save_path = os.path.join(opt.output_dir, 'ss_latents', latent_name, f'{sha256}.npz')
                save_path = os.path.join(opt.output_dir, frame_dir, "ss_latents.npz")
                np.savez_compressed(save_path, **pack)
                # records.append({'sha256': sha256, f'ss_latent_{latent_name}': True})
                
            for _ in tqdm(range(len(pack_ids)), desc="Extracting latents"):
                frame_dir, ss = load_queue.get()
                ss = ss.cuda().float()
                latent = encoder(ss, sample_posterior=False)
                assert torch.isfinite(latent).all(), "Non-finite latent"
                pack = {
                    'mean': latent[0].cpu().numpy(),
                }
                saver_executor.submit(saver, frame_dir, pack)
                
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
