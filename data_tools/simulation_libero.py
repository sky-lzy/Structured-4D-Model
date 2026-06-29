
import robosuite.macros as macros
macros.IMAGE_CONVENTION = "opencv"

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import h5py
import json
import re
import torch
import imageio
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
import xml.etree.ElementTree as ET
from plyfile import PlyData, PlyElement
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.transform import Rotation

import libero.libero.utils.utils as libero_utils
from libero.libero import get_libero_path
from libero.libero.envs import *
from libero.libero.envs.bddl_base_domain import register_problem, BDDLBaseDomain
from robosuite.utils.camera_utils import get_real_depth_map, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix

TARGET_POINT = {
    "libero_floor_manipulation": [-0.3, 0., 0.3],
    "libero_kitchen_tabletop_manipulation": [-0.3, 0., 1.2],
    "libero_living_room_tabletop_manipulation": [-0.25, 0., 0.7],
    "libero_study_tabletop_manipulation": [-0.4, 0., 1.2],
    "libero_tabletop_manipulation": [-0.3, 0., 1.2],
}

BBOX = {
    "libero_floor_manipulation": [-0.8, -0.5, -0.5, 0.25, 0.5, 1.],
    "libero_kitchen_tabletop_manipulation": [-1., -0.5, 0.85, 0.25, 0.5, 2],
    "libero_living_room_tabletop_manipulation": [-0.7, -0.45, 0.38, 0.25, 0.45, 1.5],
    "libero_study_tabletop_manipulation": [-1., -0.45, 0.85, 0, 0.45, 2.],
    "libero_tabletop_manipulation": [-1., -0.5, 0.85, 0.25, 0.5, 2],
}


def get_custom_env(base_env: BDDLBaseDomain):

    # @register_problem
    class CustomEnv(base_env):
        def __init__(self, custom_camera_list, *args, **kwargs):
            self.custom_camera_list = custom_camera_list
            super().__init__(*args, **kwargs)

        def _setup_camera(self, mujoco_arena):
            for custom_camera in self.custom_camera_list:
                custom_camera_name = custom_camera["name"]
                custom_camera_pos = custom_camera["pos"]
                custom_camera_quat = custom_camera["quat"]
                mujoco_arena.set_camera(
                    camera_name=custom_camera_name,
                    pos=custom_camera_pos,
                    quat=custom_camera_quat,
                )
            # super()._setup_camera(mujoco_arena)

        def _assert_problem_name(self):
            assert (
                self.parsed_problem["problem_name"] == self.__class__.__bases__[0].__name__.lower()
            )

    register_problem(CustomEnv)
    return CustomEnv


def get_render_camera_configs(num_cameras, target_point=[0., 0., 0.], radius=2):

    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    def look_at_to_q(eye, target, up=(0, 0, 1)):
        eye = np.array(eye)
        target = np.array(target)
        up = np.array(up)

        forward = target - eye
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        up = np.cross(right, forward)

        rot_mat = np.column_stack((right, up, -forward))

        quat = Rotation.from_matrix(rot_mat).as_quat()
        # return quat.tolist()
        return [quat[3], quat[0], quat[1], quat[2]]

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

    offset = (np.random.rand(), np.random.rand())
    camera_list = []
    for i in range(num_cameras):
        y, p = sphere_hammersley_sequence(i, num_cameras, offset)
        if p < 0:
            p = -p

        cam_pos = [
            radius * np.cos(p) * np.cos(y) + target_point[0],
            radius * np.cos(p) * np.sin(y) + target_point[1],
            radius * np.sin(p) + target_point[2],
        ]
        # cam_quat = lookat_to_pq(cam_pos, target_point)
        cam_quat = look_at_to_q(cam_pos, target_point)
        camera_list.append(
            {
                "name": f"render_camera_{i}",
                "pos": cam_pos,
                "quat": cam_quat,
            }
        )

    return camera_list

def add_cameras_to_xml(xml_str, cameras_list=[]):

    def find_element_root(root, tag):
        for child in root:
            if child.tag == tag:
                return root
            
        for child in root:
            result = find_element_root(child, tag)
            if result is not None:
                return result
        
        return None
    
    tree = ET.fromstring(xml_str)
    sensor_el = find_element_root(tree, "camera")
    for camera in cameras_list:
        camera_el = ET.SubElement(sensor_el, "camera")
        camera_el.set("name", camera["name"])
        pos_str = " ".join(map(str, camera["pos"]))
        quat_str = " ".join(map(str, camera["quat"]))
        camera_el.set("pos", pos_str)
        camera_el.set("quat", quat_str)

    return ET.tostring(tree, encoding="utf8").decode("utf8")


def rewrite_legacy_benchmark_paths(model_xml, benchmark_root):
    legacy_roots = [
        r"[^\"'\s<>]+/(?:workspace|Desktop)/libero-dev/chiliocosm",
    ]
    for legacy_root in legacy_roots:
        model_xml = re.sub(legacy_root, benchmark_root, model_xml)
    return model_xml


def create_env(hdf5_path, camera_list, image_size=(512, 512)):

    f = h5py.File(hdf5_path, "r")

    env_name = f["data"].attrs["env_name"].lower()
    env_args = json.loads(f["data"].attrs["env_args"])
    env_kwargs = env_args["env_kwargs"]

    bddl_file_name = f["data"].attrs["bddl_file_name"]
    bddl_folder = os.path.basename(os.path.dirname(bddl_file_name))
    bddl_basename = os.path.basename(bddl_file_name)
    bddl_file_dir = get_libero_path("bddl_files")
    bddl_file_path = os.path.join(bddl_file_dir, bddl_folder, bddl_basename)

    camera_names = [camera["name"] for camera in camera_list]
    camera_names.append("agentview")

    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_path,
    has_offscreen_renderer=True,
    ignore_done=True,
    use_camera_obs=True,
    camera_depths=True,
    camera_names=camera_names,
    reward_shaping=True,
    control_freq=20,
    camera_heights=image_size[0],
    camera_widths=image_size[1],
    camera_segmentations=None,
    render_gpu_device_id=0,
    )

    # create env
    custom_env = get_custom_env(TASK_MAPPING[env_name])
    env = custom_env(
        custom_camera_list=camera_list,
        **env_kwargs,
    )

    return env, f

def replay_one_scene(env, demo_data, num_samples=5):

    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except:
            continue

    camera_list = env.custom_camera_list

    benchmark_root = get_libero_path("benchmark_root")

    model_xml = demo_data.attrs["model_file"]
    model_xml = libero_utils.postprocess_model_xml(model_xml, {})
    model_xml = add_cameras_to_xml(model_xml, camera_list)
    model_xml = rewrite_legacy_benchmark_paths(model_xml, benchmark_root)

    states = demo_data["states"][()]
    actions = demo_data["actions"][()]
    dones = demo_data["dones"][()]

    done_idx = np.where(dones == 1)[0][0]

    num_actions = min(actions.shape[0], done_idx)
    if num_samples == -1: # calculate num_samples
        num_samples = num_actions // 30 # sample every 30 steps
        num_samples = max(num_samples, 4) # at least 4 samples

    init_idx = 0
    env.reset_from_xml_string(model_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(states[init_idx])
    env.sim.forward()
    # model_xml = env.sim.model.get_xml()

    camera_names = [camera["name"] for camera in camera_list]
    agent_images = []
    rgbs = []
    depths = []

    cap_index = 5
    sample_idxs = np.linspace(cap_index, num_actions - 2, num_samples).astype(int)

    for j, action in enumerate(actions):

        obs, _, _, _ = env.step(action)

        # if j >= cap_index and j < num_actions - 1:
        #     state_playback = env.sim.get_state().flatten()
        #     err = np.linalg.norm(states[j + 1] - state_playback)
        #     if err > 1e-2:
        #         print(f"[warning] playback diverged by {err:.2f} at step {j}")
        
        if j < num_actions - 1:
            env.sim.set_state_from_flattened(states[j + 1])

        agent_images.append(obs["agentview_image"])
        if j in sample_idxs:
            sample_images = [obs[f"{camera_name}_image"] for camera_name in camera_names]
            sample_depths = [obs[f"{camera_name}_depth"] for camera_name in camera_names]
            rgbs.append(np.stack(sample_images, axis=0))
            depths.append(np.stack(sample_depths, axis=0))
        
    rgbs = np.stack(rgbs, axis=0)
    depths = np.stack(depths, axis=0)

    depths = get_real_depth_map(env.sim, depths)

    return rgbs, depths, agent_images, sample_idxs

def unproject_images(depths, intrinsic_mats, extrinsic_mats, bbox=[-1., -1., -1., 1., 1., 1.], device="cuda:0"):
    # depths: np.array [n_frame, n_cam, h, w, 1]
    # intrinsic_mats: np.array [n_cam, 3, 3]
    # extrinsic_mats: np.array [n_cam, 4, 4]

    n_frame, n_cam, h, w, _ = depths.shape
    depths = torch.tensor(depths, dtype=torch.float32, device=device)
    intrinsic_mats = torch.tensor(intrinsic_mats, dtype=torch.float32, device=device)
    extrinsic_mats = torch.tensor(extrinsic_mats, dtype=torch.float32, device=device)

    grid_x = torch.arange(w, dtype=torch.float32, device=device)
    grid_y = torch.arange(h, dtype=torch.float32, device=device)
    grid_xy = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='xy'), dim=-1)
    grid_xy = grid_xy.unsqueeze(0).unsqueeze(0).repeat(n_frame, n_cam, 1, 1, 1)
    ones = torch.ones_like(grid_xy[..., :1])
    grid_xy = torch.cat([grid_xy, ones], dim=-1).view(n_frame*n_cam, h*w, 3)  # [n_frame * n_cam, h * w, 3]

    R, t = extrinsic_mats[None, :, :3, :3].repeat(n_frame, 1, 1, 1).view(n_frame*n_cam, 3, 3), extrinsic_mats[None, :, :3, 3].repeat(n_frame, 1, 1, 1).view(n_frame*n_cam, 1, 3)  # [n_frame * n_cam, 3, 3], [n_frame * n_cam, 1, 3]
    K = intrinsic_mats[None].repeat(n_frame, 1, 1, 1).view(n_frame*n_cam, 3, 3)  # [n_frame * n_cam, 3, 3]
    K_inv = torch.linalg.inv(K)  # [n_frame * n_cam, 3, 3]
    R_inv = torch.linalg.inv(R)  # [n_frame * n_cam, 3, 3]

    xyz_cam = torch.bmm(grid_xy, K_inv.transpose(1, 2))  # [n_frame * n_cam, h * w, 3]
    xyz_cam = xyz_cam * depths.view(n_frame*n_cam, h*w, 1)  # [n_frame * n_cam, h * w, 3]
    xyz_world = torch.bmm(xyz_cam - t, R_inv.transpose(1, 2))  # [n_frame * n_cam, h * w, 3]

    flag_inbbox = (xyz_world[..., 0] >= bbox[0]) & (xyz_world[..., 0] <= bbox[3]) & \
        (xyz_world[..., 1] >= bbox[1]) & (xyz_world[..., 1] <= bbox[4]) & \
        (xyz_world[..., 2] >= bbox[2]) & (xyz_world[..., 2] <= bbox[5])
    flag_inbbox = flag_inbbox.view(n_frame, n_cam, h, w, 1)

    return xyz_world.view(n_frame, n_cam, h, w, 3), flag_inbbox

def get_camera_poses(env, image_size=(512, 512)):
    camera_list = env.custom_camera_list
    camera_names = [camera["name"] for camera in camera_list]
    intrinsic_mat = [get_camera_intrinsic_matrix(env.sim, camera_name, image_size[0], image_size[1]) for camera_name in camera_names]
    extrinsic_mat = [get_camera_extrinsic_matrix(env.sim, camera_name) for camera_name in camera_names]
    intrinsic_mat = np.stack(intrinsic_mat, axis=0)
    extrinsic_mat = np.stack(extrinsic_mat, axis=0)
    extrinsic_mat = np.linalg.inv(extrinsic_mat)
    return intrinsic_mat, extrinsic_mat

def save_voxels(xyzs, flags, output_dir, sample_idxs, export_voxels=True):

    assert xyzs.shape[0] == len(sample_idxs)

    def _voxelize(xyz, center, scale):
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
        return vertices

    def _write_ply(vertices, output_path):
        vertex = np.empty(vertices.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex['x'] = vertices[:, 0]
        vertex['y'] = vertices[:, 1]
        vertex['z'] = vertices[:, 2]
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=True).write(output_path)
    
    xyz = xyzs.view(-1, 3)
    xyz_filtered = xyz[flags.view(-1)]
    aabb = torch.stack([xyz_filtered.min(dim=0)[0], xyz_filtered.max(dim=0)[0]], dim=0)
    center = (aabb[0] + aabb[1]) / 2.0
    scale = (aabb[1] - aabb[0]).max().item()

    for i in range(len(sample_idxs)):
        xyz = xyzs[i].view(-1, 3)
        flag = flags[i].view(-1)
        xyz_filtered = xyz[flag]
        if export_voxels:
            vertices = _voxelize(xyz_filtered, center, scale)
        else:
            vertices = xyz_filtered.cpu().numpy()
            _pcd = o3d.geometry.PointCloud()
            _pcd.points = o3d.utility.Vector3dVector(vertices)
            _pcd = _pcd.voxel_down_sample(voxel_size=0.01)
            vertices = np.array(_pcd.points)
        output_frame_dir = os.path.join(output_dir, f"frame_{sample_idxs[i]:04d}")
        os.makedirs(output_frame_dir, exist_ok=True)
        output_path = os.path.join(output_frame_dir, "voxels.ply")
        _write_ply(vertices, output_path)
        
    return center.cpu().numpy(), scale

def save_images(rgbs, flags, output_dir, sample_idxs, max_workers=32):

    n_frame, n_cam, h, w, _ = rgbs.shape
    rgbs = torch.tensor(rgbs, dtype=torch.uint8, device=flags.device)
    rgbs = (rgbs * flags).to(torch.uint8)
    alphas = (torch.ones_like(rgbs[..., :1]) * flags * 255).to(torch.uint8)
    rgbas = torch.cat([rgbs, alphas], dim=-1).cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)
    def _save_image_func(image, save_path):
        Image.fromarray(image).save(save_path)

    rgbas = rgbas.reshape(n_frame*n_cam, h, w, 4)
    save_paths = []
    for i in range(n_frame):
        output_frame_dir = os.path.join(output_dir, f"frame_{sample_idxs[i]:04d}", "images")
        os.makedirs(output_frame_dir, exist_ok=True)
        save_frame_paths = [os.path.join(output_frame_dir, f"view_{j:04d}.png") for j in range(n_cam)]
        save_paths.extend(save_frame_paths)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(n_frame*n_cam):
            image = rgbas[i]
            save_path = save_paths[i]
            future = executor.submit(_save_image_func, image, save_path)
            futures.append(future)
        for future in futures:
            future.result()

def save_metadata(output_dir, center, scale, sample_idxs, intrinsic_mats, extrinsic_mats, image_size):
    # center: [3]
    # scale: [1]
    # sample_idxs: [n_frame]
    # intrinsic_mats: [n_cam, 3, 3]
    # extrinsic_mats: [n_cam, 4, 4]

    n_frame = len(sample_idxs)
    n_cam = intrinsic_mats.shape[0]

    os.makedirs(output_dir, exist_ok=True)
    for i in range(n_frame):
        output_frame_dir = os.path.join(output_dir, f"frame_{sample_idxs[i]:04d}")
        os.makedirs(output_frame_dir, exist_ok=True)
        metadata = {
            "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            "center": center.tolist(),
            "scale": scale,
            "image_size": image_size,
            "frames": [],
        }
        for iter_cam in range(n_cam):
            metadata["frames"].append({
                "camera_id": iter_cam,
                "intrinsic_matrix": intrinsic_mats[iter_cam].tolist(),
                "extrinsic_matrix": extrinsic_mats[iter_cam].tolist(),
            })
        metadata_path = os.path.join(output_frame_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

def save_one_scene(env, demo_data, output_dir, instruction, num_samples=5, bbox=[-1., -1., -1., 1., 1., 1.], export_voxels=True):

    rgbs, depths, agent_images, sample_idxs = replay_one_scene(env, demo_data, num_samples=num_samples)
    intrinsics, extrinsics = get_camera_poses(env, image_size=(512, 512))
    xyz_world, flag_inbbox = unproject_images(depths, intrinsics, extrinsics, device="cuda:0", bbox=bbox)

    center, scale = save_voxels(xyz_world, flag_inbbox, output_dir, sample_idxs, export_voxels=export_voxels)
    save_images(rgbs, flag_inbbox, output_dir, sample_idxs)
    save_metadata(output_dir, center, scale, sample_idxs, intrinsics, extrinsics, image_size=(512, 512))
    with open(os.path.join(output_dir, "instruction.txt"), "w") as f:
        f.write(instruction)

    agent_video_path = os.path.join(output_dir, "agentview.mp4")
    imageio.mimsave(agent_video_path, agent_images, fps=30)

def get_target_and_bbox(hdf5_path):
    f = h5py.File(hdf5_path, "r")
    problem_info = json.loads(f["data"].attrs["problem_info"])
    problem_name = problem_info["problem_name"]
    target_point = TARGET_POINT.get(problem_name, None)
    bbox = BBOX.get(problem_name, None)
    if target_point is None or bbox is None:
        raise ValueError(f"Cannot find target point or bbox for problem {problem_name}")
    return target_point, bbox

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--libero_type", type=str, default="libero_goal", choices=["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"])
    # parser.add_argument("--base_env_name", type=str)
    parser.add_argument("--output_dir", type=str)
    # parser.add_argument("--num_scenes", type=int, default=1)
    parser.add_argument("--num_tasks", type=int, default=1)
    # parser.add_argument("--start_scene", type=int, default=0)
    parser.add_argument("--start_task", type=int, default=0)
    parser.add_argument("--scenes_per_task", type=int, default=50)

    parser.add_argument("--num_cameras", type=int, default=120)
    # parser.add_argument("--target_point", type=float, nargs="+", default=[0., 0., 0.])
    parser.add_argument("--image_size", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--radius", type=float, default=2.0)
    # parser.add_argument("--fov", type=float, default=1)
    # parser.add_argument("--bbox", type=float, nargs="+", default=[-1., -1., -1., 1., 1., 1.])
    parser.add_argument("--num_timesteps", type=int, default=5)
    # parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--export_voxels", action="store_true", default=True)

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
    dataset_dir = os.path.join(get_libero_path("datasets"), args.libero_type)
    all_h5_files = sorted(os.listdir(dataset_dir))
    all_h5_files = [os.path.join(dataset_dir, f) for f in all_h5_files if f.endswith(".hdf5")]

    # scene_idx = 0
    
    with tqdm(total=args.num_tasks*args.scenes_per_task) as pbar:
        # for hdf5_path in all_h5_files:
        for task_idx in range(args.start_task, args.start_task + args.num_tasks):
            
            scene_idx = 0
            
            if task_idx >= len(all_h5_files):
                print(f"Task index {task_idx} exceeds the number of available tasks. Stopping.")
                break
            hdf5_path = all_h5_files[task_idx]

            target_point, bbox = get_target_and_bbox(hdf5_path)

            camera_list = get_render_camera_configs(args.num_cameras, target_point=target_point, radius=args.radius)
            env, f = create_env(hdf5_path, camera_list, image_size=args.image_size)

            problem_info = json.loads(f["data"].attrs["problem_info"])
            language_instruction = problem_info["language_instruction"]

            output_task_dir = os.path.join(args.output_dir, f"task_{task_idx:04d}")
            os.makedirs(output_task_dir, exist_ok=True)

            demos = list(f["data"].keys())
            for i, ep in enumerate(demos):
                # if scene_idx < args.start_scene:
                #     scene_idx += 1
                #     continue
                # if scene_idx >= args.start_scene + args.num_scenes:
                #     break
                if scene_idx >= args.scenes_per_task:
                    print(f"Reached the limit of {args.scenes_per_task} scenes for task {task_idx}. Stopping.")
                    break
                
                demo_data = f["data"][ep]
                output_scene_dir = os.path.join(output_task_dir, f"scene_{scene_idx:04d}")
                os.makedirs(output_scene_dir, exist_ok=True)

                try:
                    save_one_scene(env, demo_data, output_scene_dir, language_instruction, num_samples=args.num_timesteps, bbox=bbox, export_voxels=args.export_voxels)
                except Exception as e:
                    print(f"Error when playing back task {task_idx}")
                    print(f"Skipping task {task_idx} ...")
                    # remove task directory
                    if os.path.exists(output_task_dir):
                        import shutil
                        shutil.rmtree(output_task_dir)
                    break

                scene_idx += 1
                pbar.update(1)

                # print(f"Only save one scene now for quick test!!!")
                # break

            env.close()

            # if scene_idx >= args.start_scene + args.num_scenes:
            #     break

                
    print(f"Finished saving {scene_idx} scenes to {args.output_dir}")
