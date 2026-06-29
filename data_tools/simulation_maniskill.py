
import os
import json
import h5py
import torch
import imageio
import argparse
import numpy as np
import open3d as o3d

import gymnasium as gym
from PIL import Image
from plyfile import PlyData, PlyElement

# multi-threaded
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.tasks.tabletop import PegInsertionSideEnv, StackCubeEnv, PullCubeToolEnv
from mani_skill.trajectory.utils import dict_to_list_of_dicts
# from mani_skill.utils.geometry.trimesh_utils import get_actor_visual_mesh, get_component_mesh, merge_meshes

torch.set_grad_enabled(False)

ENV_NAMES = {
    "PegInsertionSide-v1": PegInsertionSideEnv,
    "StackCube-v1": StackCubeEnv,
    "PullCubeTool-v1": PullCubeToolEnv,
}

def create_custom_env(base_env_name="PegInsertionSide-v1"):

    BASE_ENV = ENV_NAMES.get(base_env_name)
    if BASE_ENV is None:
        supported = ", ".join(sorted(ENV_NAMES))
        raise ValueError(f"Unsupported ManiSkill task '{base_env_name}'. Supported tasks: {supported}")

    @register_env("MyDefineEnv", max_episode_steps=1000)
    class MyDefineEnv(BASE_ENV):
        def __init__(self, *args, custom_render_cameras=None, **kwargs):
            self.custom_render_cameras = custom_render_cameras
            self._activate_obs = True
            super().__init__(*args, **kwargs)
            self._activate_obs = False
        
        def activate_obs(self):
            self._activate_obs = True
        
        def deactivate_obs(self):
            self._activate_obs = False

        def get_obs(self, *args, **kwargs):
            if not self._activate_obs:
                return dict()
            return super().get_obs(*args, **kwargs)

        @property
        def _default_sensor_configs(self):
            return self.custom_render_cameras if self.custom_render_cameras else super()._default_sensor_configs

    return MyDefineEnv

def get_render_camera_configs(num_cameras, target_point=[0., 0., 0.], image_size=[512, 512], radius=2, fov=40/180*np.pi, near=0.01, far=100):

    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    def radical_inverse(base, n):
        val = 0
        inv_base = 1.0 / base
        inv_base_n = inv_base
        while n > 0:
            digit = n % base
            val += digit * inv_base_n
            n //= base
            inv_base_n *= inv_base
        return val

    def halton_sequence(dim, n):
        return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

    def hammersley_sequence(dim, n, num_samples):
        return [n / num_samples] + halton_sequence(dim - 1, n)

    def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
        u, v = hammersley_sequence(2, n, num_samples)
        u += offset[0] / num_samples
        v += offset[1]
        u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
        theta = np.arccos(1 - 2 * u) - np.pi / 2
        phi = v * 2 * np.pi
        return [phi, theta]

    camera_configs = []

    offset = (np.random.rand(), np.random.rand())
    # radius = 2
    # fov = 40 / 180 * np.pi
    for i in range(num_cameras):
        y, p = sphere_hammersley_sequence(i, num_cameras, offset)
        # only keeps cameras above the xy plane
        if p < 0:
            p = -p
        cam_pos = [
            radius * np.cos(p) * np.cos(y) + target_point[0],
            radius * np.cos(p) * np.sin(y) + target_point[1],
            radius * np.sin(p) + target_point[2],
        ]
        camera_configs.append(CameraConfig(
            uid=f"render_camera_{i}",
            pose=sapien_utils.look_at(
                eye=cam_pos,
                target=target_point,
            ),
            fov=fov,
            width=image_size[1],
            height=image_size[0],
            near=near,
            far=far,
            ))

    return camera_configs

def unproject_images(rgbs, depths, intrinsic_mats, extrinsic_mats, image_size=[512, 512], bbox=[-1., -1., -1., 1., 1., 1.]):
    # rgbs: [n_env, n_camera, h, w, 3]
    # depths, segmentations: [n_env, n_camera, h, w, 1]
    # intrinsic_mats: [n_env, n_camera, 3, 3]
    # extrinsic_mats: [n_env, n_camera, 3, 4]

    n_env, n_camera, h, w, _ = rgbs.shape
    device = rgbs.device
    intrinsic_mats = intrinsic_mats.to(device)
    extrinsic_mats = extrinsic_mats.to(device)

    grid_x = torch.arange(image_size[1], dtype=torch.float32, device=device)
    grid_y = torch.arange(image_size[0], dtype=torch.float32, device=device)
    grid_xy = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1)
    grid_xy = grid_xy.unsqueeze(0).unsqueeze(0).repeat(n_env, 1, 1, 1, 1).repeat(1, n_camera, 1, 1, 1)
    ones = torch.ones_like(grid_xy[..., :1])
    grid_xy = torch.cat([grid_xy, ones], dim=-1).view(n_env * n_camera, h*w, 3)  # [n_env * n_camera, h * w, 3]

    R, t = extrinsic_mats[:, :, :3, :3].view(n_env * n_camera, 3, 3), extrinsic_mats[:, :, :3, 3].view(n_env * n_camera, 1, 3)  # [n_env * n_camera, 3, 3], [n_env * n_camera, 1, 3]
    K = intrinsic_mats.view(n_env * n_camera, 3, 3)  # [n_env * n_camera, 3, 3]
    K_inv = torch.linalg.inv(K)  # [n_env * n_camera, 3, 3]
    R_inv = torch.linalg.inv(R)  # [n_env * n_camera, 3, 3]

    xyz_cam = torch.bmm(grid_xy, K_inv.transpose(1, 2))  # [n_env * n_camera, h * w, 3]
    xyz_cam = xyz_cam * depths.view(n_env * n_camera, h*w, 1) / 1000. # [n_env * n_camera, h * w, 3]
    xyz_world = torch.bmm(xyz_cam - t, R_inv.transpose(1, 2))  # [n_env * n_camera, h * w, 3]

    flag_inbbox = (xyz_world[..., 0] >= bbox[0]) & (xyz_world[..., 0] <= bbox[3]) & \
        (xyz_world[..., 1] >= bbox[1]) & (xyz_world[..., 1] <= bbox[4]) & \
        (xyz_world[..., 2] >= bbox[2]) & (xyz_world[..., 2] <= bbox[5])
    flag_inbbox = flag_inbbox.view(n_env, n_camera, h, w, 1)  # [n_env, n_camera, h, w, 1]

    return xyz_world.view(n_env, n_camera, h, w, 3), flag_inbbox


def save_voxels(xyzs, flags, output_dir, scene_idxs, scale=None, center=None):

    def _voxelize(xyz, center, scale):
        # aabb = torch.stack([xyz.min(dim=0)[0], xyz.max(dim=0)[0]], dim=0)  # [2, 3]
        # center = (aabb[0] + aabb[1]) / 2.0
        # scale = (aabb[1] - aabb[0]).max().item()
        vertices = (xyz - center) / scale
        vertices = torch.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        _pcd = o3d.geometry.PointCloud()
        _pcd.points = o3d.utility.Vector3dVector(vertices.cpu().numpy())
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            _pcd,
            voxel_size=1/64,
            min_bound=(-0.5, -0.5, -0.5),
            max_bound=(0.5, 0.5, 0.5),
        )
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
        vertices = (vertices + 0.5) / 64 - 0.5
        # return vertices, center.cpu().numpy(), scale
        return vertices

    def _write_ply(vertices, output_path):
        vertex = np.empty(vertices.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        vertex["x"] = vertices[:, 0]
        vertex["y"] = vertices[:, 1]
        vertex["z"] = vertices[:, 2]
        el = PlyElement.describe(vertex, "vertex")
        PlyData([el], text=True).write(output_path)

    # xyzs: [n_env, n_camera, h, w, 3]
    # flags: [n_env, n_camera, h, w, 1]

    xyz = xyzs.view(-1, 3)  # [n_env * n_camera * h * w, 3]
    xyz_filtered = xyz[flags.view(-1)]
    aabb = torch.stack([xyz_filtered.min(dim=0)[0], xyz_filtered.max(dim=0)[0]], dim=0)  # [2, 3]
    # aabb = torch.tensor([bbox[:3], bbox[3:]], dtype=torch.float32, device=xyz.device)  # [2, 3]
    center = (aabb[0] + aabb[1]) / 2.0 if center is None else torch.tensor(center, dtype=torch.float32, device=xyz.device)
    scale = (aabb[1] - aabb[0]).max().item() if scale is None else scale

    for iter_env in range(xyzs.shape[0]):
        xyz = xyzs[iter_env].view(-1, 3)  # [n_cam * h * w, 3]
        flag = flags[iter_env].view(-1)  # [n_cam * h * w]
        xyz_filtered = xyz[flag]  # [N, 3]
        # if export_voxels:
        vertices = _voxelize(xyz_filtered, center, scale)  # [N, 3]
        # else:
        #     vertices = xyz_filtered.cpu().numpy()
        #     _pcd = o3d.geometry.PointCloud()
        #     _pcd.points = o3d.utility.Vector3dVector(vertices)
        #     _pcd = _pcd.voxel_down_sample(voxel_size=0.01)
        #     vertices = np.array(_pcd.points)
        output_env_dir = os.path.join(output_dir, f"frame_{scene_idxs[iter_env]:04d}")
        os.makedirs(output_env_dir, exist_ok=True)
        output_path = os.path.join(output_env_dir, f"voxels.ply")
        _write_ply(vertices, output_path)
    
    return center.cpu().numpy(), scale

def save_images(rgbs, output_dir, scene_idxs, output_type="images", max_workers=32):
    num_envs, num_cams, h, w, _ = rgbs.shape
    assert num_envs == len(scene_idxs), "Number of environments must match the number of scene indices."
    os.makedirs(output_dir, exist_ok=True)

    # multi-threaded save
    def save_image_func(image, save_path):
        Image.fromarray(image).save(save_path)
    
    def save_depth_func(depth, save_path):
        # np.save(save_path, depth[..., 0])
        Image.fromarray(depth, mode='I;16').save(save_path)

    # suffix = ".png" if output_type == "images" else ".npy"
    suffix = ".png"

    rgbs = rgbs.reshape(num_envs * num_cams, h, w, -1)
    save_paths = []
    for iter_env in range(num_envs):
        output_scene_dir = os.path.join(output_dir, f"frame_{scene_idxs[iter_env]:04d}", output_type)
        os.makedirs(output_scene_dir, exist_ok=True)
        save_scene_paths = [os.path.join(output_scene_dir, f"view_{i:04d}"+suffix) for i in range(num_cams)]
        save_paths.extend(save_scene_paths)

    # print(f"Saving {num_envs * num_cams} images to {output_dir}...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(num_envs * num_cams):
            image = rgbs[i]
            save_path = save_paths[i]
            if output_type == "images":
                future = executor.submit(save_image_func, image, save_path)
            else:
                future = executor.submit(save_depth_func, image, save_path)
            futures.append(future)

        # for future in tqdm(futures):
        for future in futures:
            future.result()

def save_metadata(output_dir, center, scale, scene_idxs, intrinsic_mats, extrinsic_mats, image_size, scene_seed, control_mode):

    # center: [3]
    # scale: [1]
    # scene_idxs: [n_env]
    # intrinsic_mats: [n_env, n_cam, 3, 3]
    # extrinsic_mats: [n_env, n_cam, 3, 4]
    num_envs, num_cams, _, _ = intrinsic_mats.shape
    os.makedirs(output_dir, exist_ok=True)
    for iter_env in range(num_envs):
        output_scene_dir = os.path.join(output_dir, f"frame_{scene_idxs[iter_env]:04d}")
        os.makedirs(output_scene_dir, exist_ok=True)
        metadata = {
            "seed": scene_seed,
            "control_mode": control_mode,
            "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            "center": center.tolist() if center is not None else None,
            "scale": scale,
            "image_size": image_size,
            "frames": [],
        }
        for iter_cam in range(num_cams):
            metadata["frames"].append({
                "camera_id": iter_cam,
                "intrinsic_matrix": intrinsic_mats[iter_env, iter_cam].cpu().numpy().tolist(),
                "extrinsic_matrix": extrinsic_mats[iter_env, iter_cam].cpu().numpy().tolist(),
            })
        metadata_path = os.path.join(output_scene_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

def get_obs(obs):
    rgbs = torch.stack([obs["sensor_data"][camera_key]["rgb"] for camera_key in obs["sensor_data"].keys() if "render_camera" in camera_key], dim=1)  # [n_env, n_camera, h, w, 3]
    depths = torch.stack([obs["sensor_data"][camera_key]["depth"] for camera_key in obs["sensor_data"].keys() if "render_camera" in camera_key], dim=1)  # [n_env, n_camera, h, w, 1]
    intrinsic_mats = torch.stack([obs["sensor_param"][camera_key]["intrinsic_cv"] for camera_key in obs["sensor_param"].keys() if "render_camera" in camera_key], dim=1)  # [n_env, n_camera, 3, 3]
    extrinsic_mats = torch.stack([obs["sensor_param"][camera_key]["extrinsic_cv"] for camera_key in obs["sensor_param"].keys() if "render_camera" in camera_key], dim=1)  # [n_env, n_camera, 3, 4]
    return rgbs, depths, intrinsic_mats, extrinsic_mats

def replay(env, traj, seed, num_timesteps=5):
    env.reset(seed=seed)
    env_states = dict_to_list_of_dicts(traj["env_states"])
    num_actions = len(traj["actions"])
    # num_actions = sum(traj["success"][()] == False)
    if num_timesteps == -1: # calculate num_timesteps
        # num_timesteps = num_actions // 30
        start_step = 4
        # sample every 4 steps
        sample_steps = np.arange(start_step, num_actions, 4).astype(int)
    elif num_timesteps == -2: # get all observations
        sample_steps = np.arange(0, num_actions).astype(int)
    else:
        start_step = 5
        sample_steps = np.linspace(start_step, num_actions - 1, num_timesteps).astype(int)

    rgbs, depths, intrinsic_mats, extrinsic_mats = [], [], [], []

    agentviews = []
    for i in range(num_actions):
        action = traj["actions"][i]
        if i in sample_steps:
            env.unwrapped.activate_obs()
            obs, _, _, _, _ = env.step(action)
            env.unwrapped.deactivate_obs()
        else:
            obs, _, _, _, _ = env.step(action)
        env.unwrapped.set_state_dict(env_states[i])
        agentviews.append(env.unwrapped.render_rgb_array()[0].cpu().numpy())
        if i in sample_steps:
            rgbs_i, depths_i, intrinsic_mats_i, extrinsic_mats_i = get_obs(obs)
            rgbs.append(rgbs_i)
            depths.append(depths_i)
            intrinsic_mats.append(intrinsic_mats_i)
            extrinsic_mats.append(extrinsic_mats_i)
    rgbs = torch.cat(rgbs, dim=0)  # [n_env, n_camera, h, w, 3]
    depths = torch.cat(depths, dim=0)  # [n_env, n_camera, h, w, 1]
    intrinsic_mats = torch.cat(intrinsic_mats, dim=0)  # [n_env, n_camera, 3, 3]
    extrinsic_mats = torch.cat(extrinsic_mats, dim=0)  # [n_env, n_camera, 3, 4]

    return rgbs, depths, intrinsic_mats, extrinsic_mats, sample_steps, agentviews

def process_single_scene(env, base_env, traj, seed, output_dir, args):

    # replay
    rgbs, depths, intrinsic_mats, extrinsic_mats, sample_steps, agentviews = replay(env, traj, seed, num_timesteps=args.num_timesteps)
    # unproject
    xyzs, flags = unproject_images(rgbs, depths, intrinsic_mats, extrinsic_mats, image_size=args.image_size, bbox=args.bbox)
    
    # save agent views
    if args.export_agentview:
        os.makedirs(output_dir, exist_ok=True)
        agent_video_path = os.path.join(output_dir, "agentview.mp4")
        imageio.mimsave(agent_video_path, agentviews, fps=30)

    # save images
    if args.export_images:
        rgbs = (rgbs * flags).to(torch.uint8)  # [n_env, n_camera, h, w, 3]
        alphas = (torch.ones_like(rgbs[..., :1]) * flags * 255).to(torch.uint8)  # [n_env, n_camera, h, w, 1]
        rgba = torch.cat([rgbs, alphas], dim=-1).cpu().numpy()  # [n_env, n_camera, h, w, 4]
        # rgba = rgbs.cpu().numpy()  # [n_env, n_camera, h, w, 3]
        save_images(rgba, output_dir=output_dir, scene_idxs=sample_steps, max_workers=args.max_workers, output_type="images")

        if args.save_depth:
            depths = (depths * flags).cpu().numpy().astype(np.uint16)  # [n_env, n_camera, h, w, 1]
            save_images(depths, output_dir=output_dir, scene_idxs=sample_steps, max_workers=args.max_workers, output_type="depth")

    # save voxels
    center, scale = None, None
    if args.export_voxels:
        center, scale = save_voxels(xyzs, flags, output_dir=output_dir, scene_idxs=sample_steps, scale=args.scale, center=args.center)

    # save metadata
    save_metadata(output_dir=output_dir, center=center, scale=scale, scene_idxs=sample_steps, intrinsic_mats=intrinsic_mats, extrinsic_mats=extrinsic_mats, image_size=args.image_size, scene_seed=seed, control_mode=args.control_mode)

    # save text instruction
    with open(os.path.join(output_dir, "instruction.txt"), "w") as f:
        f.write(args.instruction)

def create_env(args):
    camera_configs = get_render_camera_configs(
        num_cameras=args.num_cameras,
        target_point=args.target_point,
        image_size=args.image_size,
        radius=args.radius,
        fov=args.fov,
    )

    MyDefineEnv = create_custom_env(args.base_env_name)
    env = gym.make("MyDefineEnv", num_envs=1, custom_render_cameras=camera_configs, obs_mode="rgb+depth", control_mode=args.control_mode, reward_mode="none")
    base_env = gym.make("MyDefineEnv", num_envs=1, obs_mode="none", control_mode=args.control_mode, reward_mode="none")
    return env, base_env

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--demo_path", type=str)
    parser.add_argument("--base_env_name", type=str)
    parser.add_argument("--control_mode", type=str, default="pd_joint_pos", choices=["pd_joint_pos", "pd_ee_delta_pose"])
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_scenes", type=int, default=1)
    parser.add_argument("--start_scene", type=int, default=0)
    parser.add_argument("--instruction", type=str, default="")

    parser.add_argument("--num_cameras", type=int, default=40)
    parser.add_argument("--target_point", type=float, nargs="+", default=[0., 0., 0.])
    parser.add_argument("--image_size", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--fov", type=float, default=1)
    parser.add_argument("--bbox", type=float, nargs="+", default=[-1., -1., -1., 1., 1., 1.])
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--center", type=float, nargs="+", default=None)
    parser.add_argument("--num_timesteps", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--export_images", action="store_false", default=True)
    parser.add_argument("--export_agentview", action="store_true", default=True)
    parser.add_argument("--export_voxels", action="store_true", default=True)
    parser.add_argument("--save_depth", action="store_true", default=False)

    args = parser.parse_args()

    # update from config file
    cfg = json.load(open(args.config, "r"))
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    # print(f"Args: {args}")
    return args


if __name__ == "__main__":

    args = get_args()
    env, base_env = create_env(args)
    
    traj_path = args.demo_path
    # traj_path = os.path.join(demo_path, "trajectory.h5")
    if args.control_mode != "pd_joint_pos":
        assert args.control_mode in traj_path, f"Control mode {args.control_mode} not found in trajectory path {traj_path}. Please check the trajectory file."
    h5_file = h5py.File(traj_path, "r")
    json_path = traj_path.replace(".h5", ".json")
    json_data = json.load(open(json_path, "r"))
    episodes = json_data["episodes"]

    # num_scenes = len(episodes)
    for iter_scene in tqdm(range(args.start_scene, min(args.num_scenes + args.start_scene, len(episodes)))):

        output_dir = os.path.join(args.output_dir, f"scene_{iter_scene:04d}")
        os.makedirs(output_dir, exist_ok=True)

        # if the output dir is not empty, print a warning and skip
        if os.listdir(output_dir):
            print(f"!!!Warning!!!: Output directory {output_dir} is not empty. Skipping this scene.")
            continue

        episode = episodes[iter_scene]
        episode_id = episode["episode_id"]
        traj = h5_file[f"traj_{episode_id}"]
        seed = episode["episode_seed"]

        process_single_scene(env, base_env, traj, seed, output_dir, args)

    base_env.close()
    env.close()
