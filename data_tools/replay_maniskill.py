"""Replay ManiSkill trajectories stored in HDF5 (.h5) format

The replayed trajectory can use different observation modes and control modes.

We support translating actions from certain controllers to a limited number of controllers.

The script is only tested for Panda, and may include some Panda-specific hardcode.
"""

import copy
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Annotated, Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import tyro
from tqdm import tqdm
import pytorch3d.ops
import open3d as o3d
import open3d.core as o3d_core

import mani_skill.envs
from mani_skill.envs.utils.system.backend import CPU_SIM_BACKENDS
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.trajectory.utils.actions import conversion as action_conversion
from mani_skill.utils import common, io_utils, wrappers
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.wrappers.record import RecordEpisode

from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop import PegInsertionSideEnv, StackCubeEnv, PullCubeToolEnv

torch.set_grad_enabled(False)

ENV_NAMES = {
    "PegInsertionSide-v1": PegInsertionSideEnv,
    "StackCube-v1": StackCubeEnv,
    "PullCubeTool-v1": PullCubeToolEnv,
}

def create_custom_env(base_env_name: str):
    BASE_ENV = ENV_NAMES.get(base_env_name, None)
    if BASE_ENV is None:
        raise NotImplementedError(f"Custom environment for {base_env_name} is not implemented. ")
    
    @register_env("MyDefineEnv")
    class MyDefineEnv(BASE_ENV):
        def __init__(self, *args, num_extra_cameras=0, fix_camera=False, **kwargs):
            self.custom_render_cameras = None
            if num_extra_cameras > 0:
                if fix_camera:
                    self.custom_render_cameras = get_fixed_camera_configs(num_extra_cameras)
                else:
                    self.custom_render_cameras = get_render_camera_configs(
                        num_cameras=num_extra_cameras,
                        target_point=[-0.3, 0., 0.2],
                        image_size=[512, 512],
                        radius=1.5,
                        fov=1,
                    )

            super().__init__(*args, **kwargs)

        @property
        def _default_sensor_configs(self):
            return self.custom_render_cameras if self.custom_render_cameras else super()._default_sensor_configs
    
    return MyDefineEnv


def get_render_camera_configs(num_cameras, target_point=[0., 0., 0.], image_size=[512, 512], radius=2., fov=40/180*np.pi, near=0.01, far=100):

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


def get_fixed_camera_configs(num_cameras: int, img_size: int = 512):
    assert num_cameras <= 4, "Only support up to 4 cameras now."
    camera_configs = [
        CameraConfig("render_camera_0", sapien_utils.look_at([1, 0, 0.6], [-0.3, 0, 0.2]), img_size, img_size, 1, 0.01, 100),
        CameraConfig("render_camera_1", sapien_utils.look_at([0.5, -0.5, 0.8], [-0.1, 0, 0.2]), img_size, img_size, 1, 0.01, 100),
        CameraConfig("render_camera_2", sapien_utils.look_at([-0.2, 1, 0.6], [-0.1, 0, 0.2]), img_size, img_size, 1, 0.01, 100),
        CameraConfig("render_camera_3", sapien_utils.look_at([-0.2, -1, 0.6], [-0.1, 0, 0.2]), img_size, img_size, 1, 0.01, 100),
    ]
    return camera_configs[:num_cameras]


class ProcessPointcloudWrapper(gym.Wrapper):
    """
    A wrapper to process the point cloud observation from the environment.
    """
    def __init__(
        self,
        env: mani_skill.envs.sapien_env.BaseEnv,
        bbox: tuple[float, float, float, float, float, float] = (-0.5, -0.5, -0.5, 0.5, 0.5, 0.5),
        downsample_num: int = 1024,
        output_rgb: bool = False,
    ):
        super().__init__(env)
        self.bbox = bbox
        self.downsample_num = downsample_num
        self.output_rgb = output_rgb

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        obs = self.preprocess_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self.preprocess_obs(obs)
        return obs, reward, terminated, truncated, info

    def filter_points_with_bbox(self, xyzw: torch.Tensor, rgb: torch.Tensor):
        """
        Filter points within the bounding box.
        xyzw: Tensor of shape (N, 4) where N is the number of points.
        rgb: Tensor of shape (N, 3) where N is the number of points.
        return: Filtered points within the bounding box of shape (M, 3) where M <= N.
        """
        rgb = rgb[xyzw[..., -1] > 0.5]  # filter out background points
        xyzw = xyzw[xyzw[..., -1] > 0.5][..., :3]  # filter out background points
        bbox_min = torch.tensor(self.bbox[:3], device=xyzw.device)
        bbox_max = torch.tensor(self.bbox[3:], device=xyzw.device)
        valid_mask = torch.all((xyzw >= bbox_min) & (xyzw <= bbox_max), dim=-1)
        return xyzw[valid_mask], rgb[valid_mask]

    def downsample_pointclouds(self, xyz: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        xyz, rgb = self.downsample_by_voxels(xyz, rgb, voxel_size=0.01)
        downsample_num = torch.tensor([self.downsample_num], device=xyz.device)
        _, sampled_indices = pytorch3d.ops.sample_farthest_points(points=xyz[None], K=downsample_num)
        return xyz[sampled_indices[0]], rgb[sampled_indices[0]]

    def downsample_by_voxels(self, xyz: torch.Tensor, rgb: torch.Tensor, voxel_size: float = 0.001):
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(rgb.cpu().numpy())
        # pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        # downsampled_xyz = torch.tensor(np.asarray(pcd.points), device=xyz.device)
        # downsampled_rgb = torch.tensor(np.asarray(pcd.colors), device=rgb.device)

        # Create Open3D tensor point cloud with CUDA device
        device = o3d_core.Device("CUDA:0") if xyz.is_cuda else o3d_core.Device("CPU:0")
        pcd = o3d.t.geometry.PointCloud(device)
        
        # Convert PyTorch tensors to Open3D tensors
        # Make sure tensors are contiguous and in the right dtype
        xyz_o3d = o3d_core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(xyz.contiguous().float()))
        rgb_o3d = o3d_core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(rgb.contiguous().float()))
        
        pcd.point["positions"] = xyz_o3d
        pcd.point["colors"] = rgb_o3d
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        
        # Convert back to PyTorch tensors using DLPack for zero-copy
        downsampled_xyz = torch.utils.dlpack.from_dlpack(pcd.point["positions"].to_dlpack()).to(torch.float32)
        downsampled_rgb = torch.utils.dlpack.from_dlpack(pcd.point["colors"].to_dlpack()).to(torch.float32)
        
        return downsampled_xyz, downsampled_rgb

    def preprocess_obs(self, obs):
        xyzw = obs["pointcloud"]["xyzw"].float()
        assert xyzw.shape[0] == 1, "Only support single environment for now"
        xyzw = xyzw[0]
        rgb = obs["pointcloud"]["rgb"][0].float() / 255.

        xyz, rgb = self.filter_points_with_bbox(xyzw, rgb)
        if xyz.shape[0] > self.downsample_num:
            xyz, rgb = self.downsample_pointclouds(xyz, rgb)

        if self.output_rgb:
            xyz = torch.cat([xyz, rgb], dim=-1)

        obs["pointcloud"] = xyz.unsqueeze(0).float()  # add batch dimension
        obs_keep_keys = ['agent', 'extra', 'pointcloud']
        obs = {k: v for k, v in obs.items() if k in obs_keep_keys}
        return obs


class ProcessRGBDWrapper(gym.ObservationWrapper):
    def __init__(
        self, env,
        bbox: tuple[float, float, float, float, float, float] = (-0.5, -0.5, -0.5, 0.5, 0.5, 0.5),
    ) -> None:
        self.base_env = env.unwrapped
        super().__init__(env)
        self.bbox = bbox

        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def filter_background(self, rgb: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor):
        n_env, n_camera, h, w, _ = rgb.shape
        device = rgb.device
        intrinsic_mats = intrinsics.to(device)
        extrinsic_mats = extrinsics.to(device)

        grid_x = torch.arange(w, dtype=torch.float32, device=device)
        grid_y = torch.arange(h, dtype=torch.float32, device=device)
        grid_xy = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(0).repeat(n_env, 1, 1, 1, 1).repeat(1, n_camera, 1, 1, 1)
        ones = torch.ones_like(grid_xy[..., :1])
        grid_xy = torch.cat([grid_xy, ones], dim=-1).view(n_env * n_camera, h*w, 3)  # [n_env * n_camera, h * w, 3]

        R, t = extrinsic_mats[:, :, :3, :3].view(n_env * n_camera, 3, 3), extrinsic_mats[:, :, :3, 3].view(n_env * n_camera, 1, 3)  # [n_env * n_camera, 3, 3], [n_env * n_camera, 1, 3]
        K = intrinsic_mats.view(n_env * n_camera, 3, 3)  # [n_env * n_camera, 3, 3]
        K_inv = torch.linalg.inv(K)  # [n_env * n_camera, 3, 3]
        R_inv = torch.linalg.inv(R)  # [n_env * n_camera, 3, 3]

        xyz_cam = torch.bmm(grid_xy, K_inv.transpose(1, 2))  # [n_env * n_camera, h * w, 3]
        xyz_cam = xyz_cam * depth.view(n_env * n_camera, h*w, 1) / 1000. # [n_env * n_camera, h * w, 3]
        xyz_world = torch.bmm(xyz_cam - t, R_inv.transpose(1, 2))  # [n_env * n_camera, h * w, 3]

        flag_inbbox = (xyz_world[..., 0] >= self.bbox[0]) & (xyz_world[..., 0] <= self.bbox[3]) & \
            (xyz_world[..., 1] >= self.bbox[1]) & (xyz_world[..., 1] <= self.bbox[4]) & \
            (xyz_world[..., 2] >= self.bbox[2]) & (xyz_world[..., 2] <= self.bbox[5])
        flag_inbbox = flag_inbbox.view(n_env, n_camera, h, w, 1)  # [n_env, n_camera, h, w, 1]

        # return xyz_world.view(n_env, n_camera, h, w, 3), flag_inbbox
        # make all background pixels black
        rgb = rgb * flag_inbbox
        depth = depth * flag_inbbox
        return rgb, depth

    def extract_obs(self, observation, camera_keys): 
        rgbs = torch.stack([observation["sensor_data"][camera_key]["rgb"] for camera_key in camera_keys], dim=1)  # [n_env, n_camera, h, w, 3]
        depths = torch.stack([observation["sensor_data"][camera_key]["depth"] for camera_key in camera_keys], dim=1)  # [n_env, n_camera, h, w, 1]
        intrinsic_mats = torch.stack([observation["sensor_param"][camera_key]["intrinsic_cv"] for camera_key in camera_keys], dim=1)  # [n_env, n_camera, 3, 3]
        extrinsic_mats = torch.stack([observation["sensor_param"][camera_key]["extrinsic_cv"] for camera_key in camera_keys], dim=1)  # [n_env, n_camera, 3, 4]
        return rgbs, depths, intrinsic_mats, extrinsic_mats

    def observation(self, observation):

        camera_keys = sorted([k for k in observation["sensor_data"].keys() if "render_camera" in k])

        rgbs, depths, intrinsic_mats, extrinsic_mats = self.extract_obs(observation, camera_keys)
        rgbs, depths = self.filter_background(rgbs, depths, intrinsic_mats, extrinsic_mats)
        
        # update for each camera
        for i, camera_key in enumerate(camera_keys):
            observation["sensor_data"][camera_key]["rgb"] = rgbs[:, i]
            observation["sensor_data"][camera_key]["depth"] = depths[:, i]
        
        return observation


@dataclass
class Args:
    traj_path: str
    """Path to the trajectory .h5 file to replay"""
    sim_backend: Annotated[Optional[str], tyro.conf.arg(aliases=["-b"])] = None
    """Which simulation backend to use. Can be 'physx_cpu', 'physx_gpu'. If not specified the backend used is the same as the one used to collect the trajectory data."""
    obs_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-o"])] = None
    """Target observation mode to record in the trajectory. See
    https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html for a full list of supported observation modes."""
    target_control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Target control mode to convert the demonstration actions to.
    Note that not all control modes can be converted to others successfully and not all robots have easy to convert control modes.
    Currently the Panda robots are the best supported when it comes to control mode conversion. Furthermore control mode conversion is not supported in GPU parallelized environments.
    """
    verbose: bool = False
    """Whether to print verbose information during trajectory replays"""
    save_traj: bool = False
    """Whether to save trajectories to disk. This will not override the original trajectory file."""
    save_video: bool = False
    """Whether to save videos"""
    max_retry: int = 0
    """Maximum number of times to try and replay a trajectory until the task reaches a success state at the end."""
    discard_timeout: bool = False
    """Whether to discard episodes that timeout and are truncated (depends on the max_episode_steps parameter of task)"""
    allow_failure: bool = False
    """Whether to include episodes that fail in saved videos and trajectory data based on the environment's evaluation returned "success" label"""
    vis: bool = False
    """Whether to visualize the trajectory replay via the GUI."""
    use_env_states: bool = False
    """Whether to replay by environment states instead of actions. This guarantees that the environment will look exactly
    the same as the original trajectory at every step."""
    use_first_env_state: bool = False
    """Use the first env state in the trajectory to set initial state. This can be useful for trying to replay
    demonstrations collected in the CPU simulation in the GPU simulation by first starting with the same initial
    state as GPU simulated tasks will randomize initial states differently despite given the same seed compared to CPU sim."""
    count: Optional[int] = None
    """Number of demonstrations to replay before exiting. By default will replay all demonstrations"""
    reward_mode: Optional[str] = None
    """Specifies the reward type that the env should use. By default it will pick the first supported reward mode. Most environments
    support 'sparse', 'none', and some further support 'normalized_dense' and 'dense' reward modes"""
    record_rewards: bool = False
    """Whether the replayed trajectory should include rewards"""
    shader: Optional[str] = None
    """Change shader used for rendering for all cameras. Default is none meaning it will use whatever was used in the original data collection or the environment default.
    Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    video_fps: Optional[int] = None
    """The FPS of saved videos. Defaults to the control frequency"""
    render_mode: str = "rgb_array"
    """The render mode used for saving videos. Typically there is also 'sensors' and 'all' render modes which further render all sensor outputs like cameras."""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run to replay trajectories. With CPU backends typically this is parallelized via python multiprocessing.
    For parallelized simulation backends like physx_gpu, this is parallelized within a single python process by leveraging the GPU."""

    num_extra_cams: int = 0
    """Number of extra cameras to use for recording point clouds."""
    fix_camera: bool = False
    """Whether to use fixed cameras or randomly generated cameras."""
    # point_cloud_res: int = 64
    # """Voxel resolution of the point cloud to record."""
    downsample_num: int = 1024
    """Number of points to downsample the point cloud to. This is only used if use_env_states is True and the environment supports point clouds."""
    bbox: tuple[float, float, float, float, float, float] = (-0.5, -0.5, -0.5, 0.5, 0.5, 0.5)
    """Bounding box to use for recording point clouds. This is in the format (xmin, ymin, zmin, xmax, ymax, zmax) in meters."""
    output_rgb: bool = False
    """Whether to output RGB along with point cloud positions during replay."""
    output_dir: str = None

    world_size: int = 1
    rank: int = 0


@dataclass
class ReplayResult:
    num_replays: int
    successful_replays: int


def sanity_check_and_format_seed(episode):
    """sanity checks the trajectory seed aligns with the episode seed. reformats the reset kwargs seed if missing or formatted wrong"""
    if "seed" in episode["reset_kwargs"]:
        if isinstance(episode["reset_kwargs"]["seed"], list):

            assert (
                len(episode["reset_kwargs"]["seed"]) == 1
            ), f"found multiple seeds for one trajectory (id={episode['episode_id']}) in the reset kwargs which means it is ambiguous which seed to use"
            episode["reset_kwargs"]["seed"] = episode["reset_kwargs"]["seed"][0]
        assert (
            episode["reset_kwargs"]["seed"] == episode["episode_seed"]
        ), f"found mismatch between trajectory seed and episode seed (id={episode['episode_id']})"
    else:
        episode["reset_kwargs"]["seed"] = episode["episode_seed"]


def replay_parallelized_sim(
    args: Args, env: RecordEpisode, pbar, episodes, trajectories
):
    pbar.reset(total=len(episodes))
    warned_reset_kwargs_options = False
    # split all episodes into batches of args.num_envs environments and process each batch in parallel, truncating where necessary
    # add fake episode padding to the end of the episodes to make sure all batches are the same size
    episode_pad = (args.num_envs - len(episodes) % args.num_envs) % args.num_envs
    batches = np.pad(
        np.array(episodes),
        (0, episode_pad),
        mode="constant",
        constant_values=episodes[-1],
    ).reshape(-1, args.num_envs)

    successful_replays = 0
    if pbar is not None:
        pbar.reset(total=len(episodes))
    for episode_batch_index, episode_batch in enumerate(batches):
        trajectory_ids = [episode["episode_id"] for episode in episode_batch]
        episode_lens = np.array([episode["elapsed_steps"] for episode in episode_batch])
        ori_control_mode = episode_batch[0]["control_mode"]
        assert all(
            [episode["control_mode"] == ori_control_mode for episode in episode_batch]
        ), "Replay trajectory with parallelized environments is only supported for trajectories with the same control mode"
        episode_batch_max_len = max(episode_lens)
        seeds = torch.tensor(
            [episode["episode_seed"] for episode in episode_batch],
            device=env.base_env.device,
        )
        env.reset(seed=seeds)

        # generate batched env states and actions
        env_states_list = []
        original_actions_batch = []
        env_states_batch = []  # list of batched env states shape (max_steps, D)
        for i, trajectory_id in enumerate(trajectory_ids):

            # sanity check seeds and warn user if reset kwargs includes options (which are not supported in GPU sim replay)
            traj = trajectories[f"traj_{trajectory_id}"]
            episode = episode_batch[i]
            sanity_check_and_format_seed(episode)
            if not warned_reset_kwargs_options and "options" in episode["reset_kwargs"]:
                logger.warning(
                    f"Reset kwargs includes options, which are not supported in GPU sim replay and will be ignored."
                )
                warned_reset_kwargs_options = True

            # note (stao): this code to reformat the trajectories into a list of batched dicts can be optimized
            env_states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
            actions = np.array(traj["actions"])

            # padding
            for _ in range(episode_batch_max_len + 1 - len(env_states)):
                env_states.append(env_states[-1])
            if len(actions) < episode_batch_max_len:
                actions = np.concatenate(
                    [
                        actions,
                        np.zeros(
                            (episode_batch_max_len - len(actions), actions.shape[1])
                        ),
                    ],
                    axis=0,
                )
            env_states_list.append(env_states)
            original_actions_batch.append(actions)
        for t in range(episode_batch_max_len + 1):
            env_states_batch.append(
                trajectory_utils.list_of_dicts_to_dict(
                    [env_states_list[i][t] for i in range(len(env_states_list))]
                )
            )

        original_actions_batch = np.stack(original_actions_batch, axis=1)
        if args.use_first_env_state or args.use_env_states:
            # set the first environment state to the first states in the trajectories given
            env.base_env.set_state_dict(env_states_batch[0])
            if args.save_traj:
                # replace the first saved env state
                # since we set state earlier and RecordEpisode will save the reset to state.
                def recursive_replace(x, y):
                    if isinstance(x, np.ndarray):
                        x[-1, :] = y[-1, :]
                    else:
                        for k in x.keys():
                            recursive_replace(x[k], y[k])

                recursive_replace(
                    env._trajectory_buffer.state, common.batch(env_states_batch[0])
                )
                recursive_replace(
                    env._trajectory_buffer.observation,
                    common.to_numpy(common.batch(env.base_env.get_obs())),
                )

        # replay with env states / actions
        if (
            args.target_control_mode is None
            or ori_control_mode == args.target_control_mode
        ):
            flushed_trajectories = np.zeros(len(episode_batch), dtype=bool)
            # mark the fake padding trajectories as flushed
            if episode_batch_index == len(batches) - 1 and episode_pad > 0:
                flushed_trajectories[-episode_pad:] = True
            for t, a in enumerate(original_actions_batch):
                _, _, _, truncated, info = env.step(a)
                if args.use_env_states:
                    # NOTE (stao): due to the high precision nature of some tasks even taking a single step in GPU simulation (in e.g. PushT-v1) can lead
                    # to some non-deterministic behaviors leading to some steps labeled with slightly wrong observations/rewards/success/fail data (1e-4 error).
                    # I unfortunately do not have a good solution for this apart from using the same number of parallel environments to replay demos as the original trajectory collection.
                    env.base_env.set_state_dict(env_states_batch[t])
                if args.vis:
                    env.base_env.render_human()
                # if the elapsed_steps mark saved in the trajectory is reached for any env, flush that trajectory buffer

                if args.save_traj:
                    envs_to_flush = (t >= episode_lens - 1) & (~flushed_trajectories)
                    flushed_trajectories |= envs_to_flush
                    if envs_to_flush.sum() > 0:
                        pbar.update(n=envs_to_flush.sum())
                        if not args.allow_failure:
                            if "success" in info:
                                envs_to_flush &= (info["success"] == True).cpu().numpy()
                        if args.discard_timeout:
                            envs_to_flush &= (truncated == False).cpu().numpy()
                        successful_replays += envs_to_flush.sum()
                        env.flush_trajectory(
                            env_idxs_to_flush=np.where(envs_to_flush)[0]
                        )
        else:
            raise NotImplementedError(
                "Replay with different control modes are not supported when replaying on GPU parallelized environments"
            )
    return ReplayResult(
        num_replays=len(episodes), successful_replays=successful_replays
    )


def replay_cpu_sim(
    args: Args, env: RecordEpisode, ori_env, pbar, episodes, trajectories
):
    successful_replays = 0
    for episode in episodes:
        sanity_check_and_format_seed(episode)
        episode_id = episode["episode_id"]
        traj_id = f"traj_{episode_id}"
        reset_kwargs = episode["reset_kwargs"]
        ori_control_mode = episode["control_mode"]
        if pbar is not None:
            pbar.set_description(f"Replaying {traj_id}")
        if traj_id not in trajectories:
            tqdm.write(f"{traj_id} does not exist in {args.traj_path}")
            continue

        for _ in range(args.max_retry + 1):
            # Each trial for each trajectory to replay, we reset the environment
            # and optionally set the first environment state
            env.reset(**reset_kwargs)
            if ori_env is not None:
                ori_env.reset(**reset_kwargs)

            # set first environment state and update recorded env state
            if args.use_first_env_state or args.use_env_states:
                ori_env_states = trajectory_utils.dict_to_list_of_dicts(
                    trajectories[traj_id]["env_states"]
                )
                if ori_env is not None:
                    ori_env.set_state_dict(ori_env_states[0])
                env.base_env.set_state_dict(ori_env_states[0])
                ori_env_states = ori_env_states[1:]
                if args.save_traj:
                    # replace the first saved env state
                    # since we set state earlier and RecordEpisode will save the reset to state.
                    def recursive_replace(x, y):
                        if isinstance(x, np.ndarray):
                            x[-1, :] = y[-1, :]
                        else:
                            for k in x.keys():
                                recursive_replace(x[k], y[k])

                    recursive_replace(
                        env._trajectory_buffer.state, common.batch(ori_env_states[0])
                    )
                    fixed_obs = env.base_env.get_obs()
                    recursive_replace(
                        env._trajectory_buffer.observation,
                        common.to_numpy(common.batch(fixed_obs)),
                    )
            # Original actions to replay
            ori_actions = trajectories[traj_id]["actions"][:]
            info = {}

            # Without conversion between control modes
            assert (
                args.target_control_mode is None
                or ori_control_mode == args.target_control_mode
                or not args.use_env_states
            ), "Cannot use env states when trying to \
                convert from one control mode to another. This is because control mode conversion causes there to be changes \
                in how many actions are taken to achieve the same states"
            if (
                args.target_control_mode is None
                or ori_control_mode == args.target_control_mode
            ):
                n = len(ori_actions)
                if pbar is not None:
                    pbar.reset(total=n)
                for t, a in enumerate(ori_actions):
                    if pbar is not None:
                        pbar.update()
                    _, _, _, truncated, info = env.step(a)
                    if args.use_env_states:
                        env.base_env.set_state_dict(ori_env_states[t])
                    if args.vis:
                        env.base_env.render_human()

            # From joint position to others
            elif ori_control_mode == "pd_joint_pos":
                info = action_conversion.from_pd_joint_pos(
                    args.target_control_mode,
                    ori_actions,
                    ori_env,
                    env,
                    render=args.vis,
                    pbar=pbar,
                    verbose=args.verbose,
                )

            # From joint delta position to others
            elif ori_control_mode == "pd_joint_delta_pos":
                info = action_conversion.from_pd_joint_delta_pos(
                    args.target_control_mode,
                    ori_actions,
                    ori_env,
                    env,
                    render=args.vis,
                    pbar=pbar,
                    verbose=args.verbose,
                )
            else:
                raise NotImplementedError(
                    f"Script currently does not support converting {ori_control_mode} to {args.target_control_mode}"
                )

            success = info.get("success", False)
            if args.discard_timeout:
                success = success and (not truncated)

            if success or args.allow_failure:
                successful_replays += 1
                if args.save_traj:
                    env.flush_trajectory()
                if args.save_video:
                    env.flush_video(ignore_empty_transition=False)
                break
            else:
                if args.verbose:
                    print("info", info)
        else:
            env.flush_video(save=False)
            tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")

    return ReplayResult(
        num_replays=len(episodes), successful_replays=successful_replays
    )


def _main_helper(x):
    return _main(*x)


def _main(
    args: Args,
    use_cpu_backend,
    env_id,
    env_kwargs,
    ori_env_kwargs,
    record_episode_kwargs,
    proc_id: int = 0,
    num_procs=1,
):
    pbar = tqdm(position=proc_id, leave=None, unit="step", dynamic_ncols=True)

    # Load HDF5 containing trajectories
    traj_path = args.traj_path
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    # env = gym.make(env_id, **env_kwargs)
    create_custom_env(env_id)
    env = gym.make("MyDefineEnv", num_extra_cameras=args.num_extra_cams, fix_camera=args.fix_camera, **env_kwargs)
    # TODO (support adding wrappers to the recorded data?)

    if args.obs_mode == "pointcloud":
        env = ProcessPointcloudWrapper(
            env,
            bbox=args.bbox,
            downsample_num=args.downsample_num,
            output_rgb=args.output_rgb,
        )
    elif args.obs_mode in ["rgbd", "rgb+depth"]:
        env = ProcessRGBDWrapper(env, bbox=args.bbox)
    elif args.obs_mode in ["rgb"]:
        pass
    else:
        raise NotImplementedError(f"obs_mode {args.obs_mode} not supported for current recording")

    # if pbar is not None:
    #     pbar.set_postfix(
    #         {
    #             "control_mode": env_kwargs.get("control_mode"),
    #             "obs_mode": env_kwargs.get("obs_mode"),
    #         }
    #     )

    ### Prepare for recording ###

    # note for maniskill trajectory datasets the general naming format is <trajectory_name>.<obs_mode>.<control_mode>.<sim_backend>.h5
    # If it is called <file_name>.h5 then we assume obs_mode=None, control_mode=pd_joint_pos, and sim_backend=physx_cpu
    output_dir = os.path.dirname(traj_path) if args.output_dir is None else args.output_dir
    ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
    parts = ori_traj_name.split(".")
    if len(parts) > 1:
        ori_traj_name = parts[0]
    if args.obs_mode == "pointcloud":
        suffix = "{}.{}.{}.{}cams.{}points".format(
            env.unwrapped.obs_mode if not args.output_rgb else env.unwrapped.obs_mode + "_rgb",
            env.unwrapped.control_mode,
            env.unwrapped.backend.sim_backend,
            args.num_extra_cams,
            args.downsample_num,
        )
    elif args.obs_mode in ["rgbd", "rgb+depth"]:
        suffix = "{}.{}.{}.{}cams.filtered".format(
            env.unwrapped.obs_mode,
            env.unwrapped.control_mode,
            env.unwrapped.backend.sim_backend,
            args.num_extra_cams,
        )
    elif args.obs_mode in ["rgb"]:
        suffix = "{}.{}.{}.{}cams".format(
            env.unwrapped.obs_mode,
            env.unwrapped.control_mode,
            env.unwrapped.backend.sim_backend,
            args.num_extra_cams,
        )
    else:
        raise NotImplementedError(f"obs_mode {args.obs_mode} not supported for current recording")
    
    if args.world_size > 1:
        suffix = suffix + f".ws{args.world_size}.rank{args.rank}"
    
    new_traj_name = ori_traj_name + "." + suffix
    if use_cpu_backend:
        if num_procs > 1:
            new_traj_name = new_traj_name + "." + str(proc_id)
        if args.target_control_mode is not None:
            ori_env = gym.make(env_id, **ori_env_kwargs)
            # ori_env = gym.make("MyDefineEnv", **ori_env_kwargs)
        else:
            ori_env = None
    else:
        pass

    env = wrappers.RecordEpisode(
        env,
        output_dir,
        trajectory_name=new_traj_name,
        video_fps=(
            args.video_fps if args.video_fps is not None else env.unwrapped.control_freq
        ),
        **record_episode_kwargs,
    )

    if env.save_trajectory:
        output_h5_path = env._h5_file.filename
        assert not os.path.samefile(output_h5_path, traj_path)
    else:
        output_h5_path = None

    episodes = json_data["episodes"]
    
    # Split episodes across multiple GPUs/ranks (world_size > 1)
    if args.world_size > 1:
        all_episode_indices = np.array_split(np.arange(len(episodes)), args.world_size)[args.rank]
        episodes = [episodes[i] for i in all_episode_indices]
    
    # Apply count limit after rank splitting
    if args.count is not None:
        episodes = episodes[:args.count]
    
    if use_cpu_backend:
        inds = np.arange(len(episodes))
        inds = np.array_split(inds, num_procs)[proc_id]
        replay_result = replay_cpu_sim(
            args, env, ori_env, pbar, [episodes[index] for index in inds], ori_h5_file
        )
    else:
        replay_result = replay_parallelized_sim(args, env, pbar, episodes, ori_h5_file)

    env.close()
    ori_h5_file.close()
    return output_h5_path, replay_result


def parse_args(args=None):
    return tyro.cli(Args, args=args)


def main(args: Args):
    traj_path = args.traj_path
    # Load trajectory metadata json
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]
    env_kwargs = ori_env_kwargs.copy()

    ### Checks and setting up env kwargs ###
    # First we determine how to setup the environment to replay demonstrations and raise relevant warnings to the user
    if (
        "sim_backend" in ori_env_kwargs
        and ori_env_kwargs["sim_backend"] != args.sim_backend
        and args.use_env_states
    ):
        logger.warning(
            f"Warning: Using different backend ({args.sim_backend}) than the original used to collect the trajectory data "
            f"({ori_env_kwargs['sim_backend']}). This may cause replay failures due to "
            f"differences in simulation/physics engine backend. Use the same backend by passing -b {ori_env_kwargs['sim_backend']} "
            f"or replay by environment states by passing --use-env-states instead."
        )
    if args.sim_backend is None:
        # try to guess which sim backend to use
        if "sim_backend" not in ori_env_kwargs:
            args.sim_backend = "physx_cpu"
        else:
            args.sim_backend = ori_env_kwargs["sim_backend"]

    ori_env_kwargs["sim_backend"] = args.sim_backend
    env_kwargs["sim_backend"] = args.sim_backend

    # modify the env kwargs according to the users inputs
    target_obs_mode = args.obs_mode
    target_control_mode = args.target_control_mode
    if target_obs_mode is not None:
        env_kwargs["obs_mode"] = target_obs_mode
    if target_control_mode is not None:
        env_kwargs["control_mode"] = target_control_mode
    if args.shader is not None:
        env_kwargs["shader_dir"] = args.shader  # change all shaders
    env_kwargs["reward_mode"] = args.reward_mode
    env_kwargs[
        "render_mode"
    ] = (
        args.render_mode
    )  # note this only affects the videos saved as RecordEpisode wrapper calls env.render

    record_episode_kwargs = dict(
        save_on_reset=False,
        save_trajectory=args.save_traj,
        save_video=args.save_video,
        record_reward=args.record_rewards,
    )

    if args.count is not None and args.count > len(json_data["episodes"]):
        logger.warning(
            f"Warning: Requested to replay {args.count} demos but there are only {len(json_data['episodes'])} demos collected, replaying all demos now"
        )
        args.count = len(json_data["episodes"])
    elif args.count is None:
        args.count = len(json_data["episodes"])

    # Calculate total episodes for this rank for progress bar
    total_episodes = len(json_data["episodes"])
    if args.world_size > 1:
        rank_episode_count = len(np.array_split(np.arange(total_episodes), args.world_size)[args.rank])
        if args.count is not None:
            rank_episode_count = min(rank_episode_count, args.count)
        logger.info(f"Rank {args.rank}/{args.world_size}: Will process ~{rank_episode_count} episodes")
    else:
        rank_episode_count = args.count
    
    pbar = tqdm(total=rank_episode_count, unit="step", dynamic_ncols=True, desc=f"Rank {args.rank}" if args.world_size > 1 else None)

    # if missing info or auto sim backend is provided, we try to infer which backend is being used
    if "sim_backend" not in env_kwargs or (
        env_kwargs["sim_backend"] == "auto"
        and ("num_envs" not in env_kwargs or env_kwargs["num_envs"] == 1)
    ):
        env_kwargs["sim_backend"] = "physx_cpu"
    env_kwargs["num_envs"] = args.num_envs
    if env_kwargs["sim_backend"] not in CPU_SIM_BACKENDS:
        record_episode_kwargs["max_steps_per_video"] = env_info["max_episode_steps"]
        _, replay_result = _main(
            args,
            use_cpu_backend=False,
            env_id=env_id,
            env_kwargs=env_kwargs,
            ori_env_kwargs=ori_env_kwargs,
            record_episode_kwargs=record_episode_kwargs,
            proc_id=0,
            num_procs=1,
        )

    else:
        env_kwargs["num_envs"] = 1
        ori_env_kwargs["num_envs"] = 1
        if args.num_envs > 1:
            pool = mp.Pool(args.num_envs)
            proc_args = [
                (
                    copy.deepcopy(args),
                    True,
                    env_id,
                    env_kwargs,
                    ori_env_kwargs,
                    record_episode_kwargs,
                    i,
                    args.num_envs,
                )
                for i in range(args.num_envs)
            ]
            # res = pool.starmap(_main, proc_args)
            res = list(tqdm(pool.imap(_main_helper, proc_args), total=args.count))
            replay_results_list = [x[1] for x in res]
            trajectory_paths = [x[0] for x in res]
            pool.close()
            if args.save_traj:
                # A hack to find the path
                output_path = trajectory_paths[0][: -len("0.h5")] + "h5"
                merge_trajectories(output_path, trajectory_paths)
                for h5_path in trajectory_paths:
                    tqdm.write(f"Remove {h5_path}")
                    os.remove(h5_path)
                    json_path = h5_path.replace(".h5", ".json")
                    tqdm.write(f"Remove {json_path}")
                    os.remove(json_path)
            replay_result = ReplayResult(
                num_replays=sum([x.num_replays for x in replay_results_list]),
                successful_replays=sum(
                    [x.successful_replays for x in replay_results_list]
                ),
            )
        else:
            _, replay_result = _main(
                args,
                use_cpu_backend=True,
                env_id=env_id,
                env_kwargs=env_kwargs,
                ori_env_kwargs=ori_env_kwargs,
                record_episode_kwargs=record_episode_kwargs,
                proc_id=0,
                num_procs=1,
            )

    pbar.close()
    rank_info = f"[Rank {args.rank}/{args.world_size}] " if args.world_size > 1 else ""
    print(
        f"{rank_info}Replayed {replay_result.num_replays} episodes, "
        f"{replay_result.successful_replays}/{replay_result.num_replays}={replay_result.successful_replays/replay_result.num_replays*100:.2f}% demos saved"
    )


if __name__ == "__main__":
    # spawn is needed due to warp init issue
    mp.set_start_method("spawn")
    main(parse_args())
