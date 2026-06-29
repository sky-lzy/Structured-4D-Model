import os
import copy
import sys
import json
import importlib
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from torchvision import transforms
from PIL import Image


torch.set_grad_enabled(False)


def get_data(frames, frame_dir):
    with ThreadPoolExecutor(max_workers=16) as executor:
        def worker(frame):
            image_path = os.path.join(opt.output_dir, frame_dir, "images", f"view_{frame['camera_id']:04d}.png")
            try:
                image = Image.open(image_path)
            except:
                print(f"Error loading image {image_path}")
                return None
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            if image.shape[2] == 4:  # If image has an alpha channel
                image = image[:, :, :3] * image[:, :, 3:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            intrinsics = torch.tensor(frame['intrinsic_matrix'])
            extrinsics = torch.tensor(frame['extrinsic_matrix'])
            if extrinsics.shape[0] == 3:
                extrinsics = torch.cat([extrinsics, torch.tensor([[0, 0, 0, 1]])], dim=0)

            # scale = frame['scale']
            # center = torch.tensor(frame['center'])

            return {
                'image': image,
                'extrinsics': extrinsics,
                'intrinsics': intrinsics,
                # 'scale': scale,
                # 'center': center,
            }
        
        datas = executor.map(worker, frames)
        for data in datas:
            if data is not None:
                yield data
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    feature_name = opt.model
    # os.makedirs(os.path.join(opt.output_dir, 'features', feature_name), exist_ok=True)

    # load model
    dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
    dinov2_model.eval().cuda()
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    n_patch = 518 // 14

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
                target_paths = [os.path.join(frame_dir, f"feature_{feature_name}.npz"), os.path.join(frame_dir, "latents.npz")]
                if not any(os.path.exists(target_path) for target_path in target_paths):
                    pack_ids.append(os.path.join(task_name, scene_name, frame_name))

    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=8) as loader_executor, \
            ThreadPoolExecutor(max_workers=8) as saver_executor:
            def loader(pack_id):
                # scene_id, frame_id = pack_id
                frame_dir = pack_id
                try:
                    with open(os.path.join(opt.output_dir, frame_dir, "metadata.json"), 'r') as f:
                        metadata = json.load(f)
                    frames = metadata['frames']
                    data = []
                    for datum in get_data(frames, frame_dir):
                        datum['image'] = transform(datum['image'])
                        data.append(datum)
                    # positions = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply'))[0]
                    positions = utils3d.io.read_ply(os.path.join(opt.output_dir, frame_dir, "voxels.ply"))[0]
                    scale = metadata['scale']
                    center = np.array(metadata['center'])
                    image_size = metadata['image_size']
                    # rescale_factor = max(image_size)
                    rescale_factor = torch.tensor(image_size)[[1, 0]]
                    # load_queue.put((sha256, data, positions))
                    load_queue.put((frame_dir, data, positions, scale, center, rescale_factor))
                except Exception as e:
                    print(f"Error loading data for {frame_dir}: {e}")

            loader_executor.map(loader, pack_ids)
            
            def saver(frame_dir, pack, patchtokens, uv):
                pack['patchtokens'] = F.grid_sample(
                    patchtokens,
                    uv.unsqueeze(1),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(2).permute(0, 2, 1).cpu().numpy()
                pack['patchtokens'] = np.mean(pack['patchtokens'], axis=0).astype(np.float16)
                # save_path = os.path.join(opt.output_dir, 'features', feature_name, f'{sha256}.npz')
                save_path = os.path.join(opt.output_dir, frame_dir, f"feature_{feature_name}.npz")
                np.savez_compressed(save_path, **pack)
                # records.append({'sha256': sha256, f'feature_{feature_name}' : True})
                
            for _ in tqdm(range(len(pack_ids)), desc="Extracting features"):
                frame_dir, data, positions, scale, center, rescale_factor = load_queue.get()
                positions = torch.from_numpy(positions).float().cuda()
                center = torch.from_numpy(center).float().cuda()
                indices = ((positions + 0.5) * 64).long()
                positions = positions * scale + center
                assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"
                n_views = len(data)
                N = positions.shape[0]
                pack = {
                    'indices': indices.cpu().numpy().astype(np.uint8),
                }
                patchtokens_lst = []
                uv_lst = []
                for i in range(0, n_views, opt.batch_size):
                    batch_data = data[i:i+opt.batch_size]
                    bs = len(batch_data)
                    batch_images = torch.stack([d['image'] for d in batch_data]).cuda()
                    batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data]).cuda()
                    batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data]).cuda()
                    features = dinov2_model(batch_images, is_training=True)
                    uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] / rescale_factor.view(1, 1, 2).cuda() * 2 - 1
                    patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
                    patchtokens_lst.append(patchtokens)
                    uv_lst.append(uv)
                patchtokens = torch.cat(patchtokens_lst, dim=0)
                uv = torch.cat(uv_lst, dim=0)

                # save features
                saver_executor.submit(saver, frame_dir, pack, patchtokens, uv)
                
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    # records = pd.DataFrame.from_records(records)
    # records.to_csv(os.path.join(opt.output_dir, f'feature_{feature_name}_{opt.rank}.csv'), index=False)
        