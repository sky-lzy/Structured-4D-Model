
import numpy as np
import torch
import gymnasium as gym

from mani_skill.utils import common
from mani_skill.envs.tasks.tabletop import StackCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig

from policy.common.pointcloud_utils import downsample_pointcloud_indices

ENV_NAMES = {
    "StackCube-v1": StackCubeEnv,
}

def rotate_up(angle):
    angle = np.deg2rad(angle)
    z = np.cos(angle)
    y = np.sin(angle)
    return (0, y, z)

def create_custom_env(base_env_name: str):
    BASE_ENV = ENV_NAMES.get(base_env_name, None)
    if BASE_ENV is None:
        raise NotImplementedError(f"Custom environment for {base_env_name} is not implemented. ")
    
    @register_env("MyDefineEnv")
    class MyDefineEnv(BASE_ENV):
        def __init__(self, *args, num_extra_cameras=0, fix_camera=False, _clearance=None, ambient_light=None, _rotate_angle=0, **kwargs):
            self.custom_render_cameras = None
            self._activate_obs = True # default to True
            if _clearance is not None:
                self._clearance = _clearance
            if num_extra_cameras > 0:
                if fix_camera:
                    assert num_extra_cameras == 4, "Only support 4 fixed cameras for now"
                    self.custom_render_cameras = [
                        CameraConfig("render_camera_0", sapien_utils.look_at([1, 0, 0.6], [-0.3, 0, 0.2], rotate_up(_rotate_angle)), 512, 512, 1, 0.01, 100),
                        CameraConfig("render_camera_1", sapien_utils.look_at([0.5, -0.5, 0.8], [-0.1, 0, 0.2], rotate_up(_rotate_angle)), 512, 512, 1, 0.01, 100),
                        CameraConfig("render_camera_2", sapien_utils.look_at([-0.2, 1, 0.6], [-0.1, 0, 0.2], rotate_up(_rotate_angle)), 512, 512, 1, 0.01, 100),
                        CameraConfig("render_camera_3", sapien_utils.look_at([-0.2, -1, 0.6], [-0.1, 0, 0.2], rotate_up(_rotate_angle)), 512, 512, 1, 0.01, 100),
                    ]
                else:
                    raise ValueError("Release eval only supports fixed StackCube render cameras.")
            self.ambient_light = ambient_light

            super().__init__(*args, **kwargs)

        def activate_obs(self):
            self._activate_obs = True

        def deactivate_obs(self):
            self._activate_obs = False
        
        def get_obs(self, *args, **kwargs):
            if not self._activate_obs:
                return dict()
            return super().get_obs(*args, **kwargs)
        
        def _load_lighting(self, options: dict):

            if self.ambient_light is None:
                return super()._load_lighting(options)

            for scene in self.scene.sub_scenes:
                scene.ambient_light = [self.ambient_light] * 3
                scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
                scene.add_directional_light([0, 0, -1], [1, 1, 1])

        @property
        def _default_sensor_configs(self):
            return self.custom_render_cameras if self.custom_render_cameras else super()._default_sensor_configs
    
    return MyDefineEnv

class FilterPointcloudObservationWrapper(gym.ObservationWrapper):
    def __init__(
        self, env,
        bbox: tuple[float, float, float, float, float, float] = (-1.0, -0.6, -0.2, 0.2, 0.6, 1.0),
        downsample_num: int = 1024,
        output_rgb: bool = False,
    ) -> None:
        self.base_env = env.unwrapped
        super().__init__(env)

        self.bbox = bbox
        self.downsample_num = downsample_num
        self.output_rgb = output_rgb

        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

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

    def downsample_pointclouds(self, xyz: torch.Tensor, rgb: torch.Tensor):
        indices = downsample_pointcloud_indices(xyz, num_points=self.downsample_num, voxel_size=0.01)
        return xyz[indices], rgb[indices]

    def observation(self, observation):

        if "pointcloud" not in observation:
            return observation

        pointcloud = observation.pop("pointcloud")
        del observation["sensor_param"]
        del observation["sensor_data"]
        xyzw = pointcloud["xyzw"].float().cuda()
        rgbs = pointcloud["rgb"].float().cuda() / 255.
        pointclouds = []

        for iter_xyzw, iter_rgb in zip(xyzw, rgbs):
            xyz, rgb = self.filter_points_with_bbox(iter_xyzw, iter_rgb)  # (M, 3)
            if xyz.shape[0] > self.downsample_num:
                xyz, rgb = self.downsample_pointclouds(xyz, rgb)
            else:
                raise NotImplementedError("Downsampling not implemented for less than downsample_num points")
            
            if self.output_rgb:
                xyz = torch.cat([xyz, rgb], dim=-1)

            pointclouds.append(xyz)
        
        observation["pointcloud"] = torch.stack(pointclouds).float()  # (B, M, 3)

        return observation


class FlattenPointcloudObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        self.base_env = env.unwrapped
        super().__init__(env)

        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation):
        if "pointcloud" not in observation:
            return observation
        pointcloud = observation.pop("pointcloud")
        observation = common.flatten_state_dict(
            observation, use_torch=True, device=self.base_env.device
        )
        return {
            "state": observation,
            "pointcloud": pointcloud,
        }
