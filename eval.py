from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np

import imageio
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import gymnasium as gym
import utils3d

import trellis_utils.models
from trellis_utils.modules import sparse as sp
from trellis_utils.pipelines import TrellisSDM3DGenPipeline

from policy.common.pointcloud_utils import downsample_pointcloud, voxelize_points
from policy.env_utils.env_wrappers import create_custom_env
from policy.workspace.robotworkspace import RobotWorkspace

torch.set_grad_enabled(False)

PROJECT_ROOT = Path(__file__).resolve().parent
POLICY_DIR = PROJECT_ROOT / "policy"
DEFAULT_MODEL_PATH = Path(".ckpt/structured-4d-model")
DEFAULT_CHECKPOINT_PATH = DEFAULT_MODEL_PATH / "inverse_dynamics" / "inverse_dynamics.ckpt"
DEFAULT_RENDER_DIR = Path("render/policy")
DEFAULT_INSTRUCTION = (
    "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling."
)
DEFAULT_BBOX = [-0.85, -0.6, -0.2, 0.35, 0.6, 1.0]
DEFAULT_CENTER = [-0.25, 0.0, 0.3]
DEFAULT_SCALE = 1.2
DEFAULT_FEATURE_ENCODER = "microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16"

_RUNTIME_LOADED = False


@dataclass(frozen=True)
class EvalConfig:
    env_id: str
    config_name: str
    checkpoint_path: Path
    model_path: str
    checkpoint_iter: int
    pipeline_config: Path
    render_dir: Path | None
    instruction: str
    seed_start: int
    num_seeds: int
    num_trials: int
    num_extra_cameras: int
    pred_horizon: int
    act_horizon: int
    num_preds: int
    num_replans: int
    max_task_steps: int
    hard_gripper: bool
    mode: str
    bbox: list[float]
    center: list[float]
    scale: float
    clearance: float | None


def parse_args(
    argv: Iterable[str] | None = None,
    default_env_id: str = "StackCube-v1",
    default_config_name: str = "inverse_dynamics",
    default_checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
) -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate the planner with inverse dynamics.")
    parser.add_argument("--env-id", default=default_env_id, choices=["StackCube-v1"])
    parser.add_argument("--config-name", default=default_config_name)
    parser.add_argument(
        "--checkpoint-path",
        default=str(default_checkpoint_path),
        help="Policy checkpoint file. Defaults to the downloaded inverse dynamics checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Compatibility override for an internal training-run directory containing checkpoints/<iter>.ckpt.",
    )
    parser.add_argument("--checkpoint-iter", type=int, default=20000)
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Local directory or Hugging Face repo id containing the generation pipeline.",
    )
    parser.add_argument("--pipeline-config", default=str(PROJECT_ROOT / "configs/pipeline.json"))
    parser.add_argument("--render-dir", default=str(DEFAULT_RENDER_DIR))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--num-extra-cameras", type=int, default=4)
    parser.add_argument("--pred-horizon", type=int, default=64)
    parser.add_argument("--act-horizon", type=int, default=32)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--num-replans", type=int, default=10)
    parser.add_argument("--max-task-steps", type=int, default=300)
    parser.add_argument("--hard-gripper", action="store_true")
    parser.add_argument("--mode", choices=["closed", "once"], default="closed")
    parser.add_argument("--clearance", type=float, default=None)
    args = parser.parse_args(argv)

    render_dir = None
    if not args.no_render:
        render_dir = Path(args.render_dir).expanduser()

    checkpoint_path = Path(args.checkpoint_path).expanduser()
    if args.checkpoint_dir is not None:
        checkpoint_path = Path(args.checkpoint_dir).expanduser() / "checkpoints" / f"{args.checkpoint_iter}.ckpt"

    return EvalConfig(
        env_id=args.env_id,
        config_name=args.config_name,
        checkpoint_path=checkpoint_path,
        model_path=args.model_path,
        checkpoint_iter=args.checkpoint_iter,
        pipeline_config=Path(args.pipeline_config).expanduser(),
        render_dir=render_dir,
        instruction=args.instruction,
        seed_start=args.seed_start,
        num_seeds=args.num_seeds,
        num_trials=args.num_trials,
        num_extra_cameras=args.num_extra_cameras,
        pred_horizon=args.pred_horizon,
        act_horizon=args.act_horizon,
        num_preds=args.num_preds,
        num_replans=args.num_replans,
        max_task_steps=args.max_task_steps,
        hard_gripper=args.hard_gripper,
        mode=args.mode,
        bbox=DEFAULT_BBOX,
        center=DEFAULT_CENTER,
        scale=DEFAULT_SCALE,
        clearance=args.clearance,
    )


def load_policy_config(config_name: str):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = str(POLICY_DIR / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name=config_name)


def clamp_gripper(action_sequence: np.ndarray) -> np.ndarray:
    action_sequence = action_sequence.copy()
    action_sequence[:, 7] = np.where(action_sequence[:, 7] > 0, 1.0, -1.0)
    return action_sequence


def info_success(info: dict) -> bool:
    success = info["success"]
    if hasattr(success, "item"):
        return bool(success.item())
    return bool(success)


class ReplayAgent:
    def __init__(self, cfg: EvalConfig, policy_workspace: RobotWorkspace):
        self.cfg = cfg
        self.policy_workspace = policy_workspace
        self.device = torch.device("cuda")

        print(f"Creating environment: {cfg.env_id} with {cfg.num_extra_cameras} fixed cameras")
        create_custom_env(cfg.env_id)
        self.env = gym.make(
            "MyDefineEnv",
            num_extra_cameras=cfg.num_extra_cameras,
            obs_mode="rgb+depth+segmentation",
            control_mode="pd_joint_pos",
            _clearance=cfg.clearance,
            fix_camera=True,
        )

        if not cfg.checkpoint_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {cfg.checkpoint_path}")
        print(f"Loading policy checkpoint: {cfg.checkpoint_path}")
        self.policy_workspace.load_checkpoint(cfg.checkpoint_path)
        self.policy_workspace.model = self.policy_workspace.model.cuda().eval()

        stat_path = cfg.checkpoint_path.parent / f"{cfg.env_id}_data_stat.json"
        if stat_path.exists():
            with stat_path.open("r") as f:
                data_stat = json.load(f)
        elif cfg.env_id == "StackCube-v1":
            from policy.dataset.hdf5_dataset import DATA_STAT

            print(f"Policy data statistics not found: {stat_path}; using built-in StackCube-v1 release statistics.")
            data_stat = DATA_STAT
        else:
            raise FileNotFoundError(f"Policy data statistics not found: {stat_path}")
        self.data_stat = {
            key: torch.tensor(value, dtype=torch.float32, device=self.device)
            for key, value in data_stat.items()
        }

        if cfg.model_path:
            print(f"Loading generation pipeline from pretrained source: {cfg.model_path}")
            self.pipeline = TrellisSDM3DGenPipeline.from_pretrained(cfg.model_path)
        else:
            if not cfg.pipeline_config.exists():
                raise FileNotFoundError(f"Pipeline config not found: {cfg.pipeline_config}")
            print(f"Loading generation pipeline: {cfg.pipeline_config}")
            with cfg.pipeline_config.open("r") as f:
                pipeline_cfg = json.load(f)
            self.pipeline = TrellisSDM3DGenPipeline.from_config(pipeline_cfg)
        self.pipeline.cuda()

        print("Loading DINOv2 model")
        self.dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
        self.dinov2_model.eval().cuda()

        print(f"Loading feature encoder: {DEFAULT_FEATURE_ENCODER}")
        self.feature_encoder = trellis_utils.models.from_pretrained(DEFAULT_FEATURE_ENCODER).eval().cuda()

    def take_action_once(self, seed: int, instruction: str, num_step: int, render_dir: Path | None) -> bool:
        obs, info = self.env.reset(seed=seed)
        self._ensure_render_dir(render_dir)
        target_pcs = self.get_predictions_all(obs, instruction, num_step=num_step)

        agentviews = []
        total = target_pcs.shape[0] * self.cfg.num_preds * self.cfg.act_horizon
        pbar = tqdm(total=total, desc="Sampling actions from generated pointclouds")
        terminated = False
        truncated = False

        for target_pc in target_pcs:
            for _ in range(self.cfg.num_preds):
                obs = self.process_obs_for_policy(obs)
                action_seq = self.predict_action(obs["pointcloud"], target_pc, obs["agent"]["qpos"])
                obs, info, terminated, truncated = self.run_action_sequence(action_seq, agentviews, pbar)
                if terminated or truncated:
                    break
            if terminated or truncated:
                break

        pbar.close()
        self.env.unwrapped.activate_obs()
        self.write_video(render_dir, f"eval_{seed}.mp4", agentviews)
        print(info)
        return info_success(info)

    def take_action_closed(self, seed: int, instruction: str, num_step: int, render_dir: Path | None) -> bool:
        self.env.unwrapped.activate_obs()
        obs, info = self.env.reset(seed=seed)
        self._ensure_render_dir(render_dir)

        agentviews = []
        total_step_num = min(
            num_step * self.cfg.num_preds * self.cfg.act_horizon + self.cfg.act_horizon,
            self.cfg.max_task_steps,
        )
        pbar = tqdm(total=total_step_num, desc="Sampling actions from generated pointclouds")
        terminated = False
        truncated = False

        for iter_step in range(num_step):
            curr_pc, target_pc = self.get_single_prediction(obs, instruction)
            num_preds = self.cfg.num_preds if iter_step < num_step - 1 else self.cfg.num_preds + 1
            for iter_pred in range(num_preds):
                if iter_pred > 0:
                    curr_pc = self.process_obs_for_policy(obs)["pointcloud"]
                action_seq = self.predict_action(curr_pc, target_pc, obs["agent"]["qpos"])
                obs, info, terminated, truncated = self.run_action_sequence(
                    action_seq,
                    agentviews,
                    pbar,
                    max_elapsed_steps=total_step_num,
                )

                if terminated or truncated:
                    break
            if terminated or truncated:
                break

        pbar.close()
        self.env.unwrapped.activate_obs()
        self.write_video(render_dir, f"eval_closed_{seed}.mp4", agentviews)
        print(info)
        return info_success(info)

    def predict_action(self, current_pc: torch.Tensor, target_pc: torch.Tensor, qpos: torch.Tensor) -> np.ndarray:
        obs_seq = {
            "pointcloud": torch.stack([current_pc.cuda(), target_pc.cuda()], dim=0).unsqueeze(0),
            "agent_pos": self.normalize_states(qpos.cuda())[:, :8].unsqueeze(0),
        }
        actions = self.policy_workspace.model.predict_action(obs_seq)["action"][0]
        actions = self.denormalize_actions(actions)[: self.cfg.act_horizon].cpu().numpy()
        if self.cfg.hard_gripper:
            actions = clamp_gripper(actions)
        return actions

    def run_action_sequence(
        self,
        actions: np.ndarray,
        agentviews: list[np.ndarray],
        pbar: tqdm,
        max_elapsed_steps: int | None = None,
    ) -> tuple[dict, dict, bool, bool]:
        self.env.unwrapped.deactivate_obs()
        obs = {}
        info = {}
        terminated = False
        truncated = False
        for action_idx, action in enumerate(actions):
            if action_idx >= len(actions) - 1:
                self.env.unwrapped.activate_obs()
            obs, _, terminated, truncated, info = self.env.step(action)
            agentviews.append(self.env.unwrapped.render_rgb_array().cpu().numpy()[0])
            pbar.update(1)
            elapsed_steps = info.get("elapsed_steps")
            if max_elapsed_steps is not None and elapsed_steps is not None:
                if hasattr(elapsed_steps, "item"):
                    elapsed_steps = elapsed_steps.item()
                if elapsed_steps >= max_elapsed_steps:
                    terminated = True
            if terminated or truncated:
                break
        return obs, info, terminated, truncated

    def get_predictions_all(self, obs: dict, instruction: str, num_step: int) -> torch.Tensor:
        obs_processed = self.process_obs(obs)
        generated_slats = self.pipeline.generate_unroll(
            obs_processed["latent"],
            num_step=num_step,
            ss_sampler_params={"steps": 25, "cfg_strength": 7.5},
            slat_sampler_params={"steps": 25, "cfg_strength": 3.0},
            text_instruction=instruction,
        )
        decoded = [self.pipeline.decode_slat(slat, ["gaussian"]) for slat in generated_slats]
        target_pcs = [item["gaussian"][0].get_xyz.detach() for item in decoded]
        target_pcs = [downsample_pointcloud(pc, num_points=4096) for pc in target_pcs]
        target_pcs = [
            pc * obs_processed["scale"] + obs_processed["center"].view(1, 3)
            for pc in target_pcs
        ]
        return torch.stack(target_pcs, dim=0)

    def get_single_prediction(self, obs: dict, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        obs_processed = self.process_obs(obs)
        generated_slat = self.pipeline.generate_onestep(
            obs_processed["latent"],
            ss_sampler_params={"steps": 25, "cfg_strength": 7.5},
            slat_sampler_params={"steps": 25, "cfg_strength": 3.0},
            text_instruction=instruction,
        )
        decoded = self.pipeline.decode_slat(generated_slat, ["gaussian"])
        target_pc = downsample_pointcloud(decoded["gaussian"][0].get_xyz.detach(), num_points=4096)
        target_pc = target_pc * obs_processed["scale"] + obs_processed["center"].view(1, 3)
        return obs_processed["pointcloud"], target_pc

    def process_obs_for_policy(self, obs: dict) -> dict:
        rgbs, depths, intrinsic_mats, extrinsic_mats = self.get_obs(obs)
        xyzs, flags = self.unproject_images(rgbs, depths, intrinsic_mats, extrinsic_mats)
        _, pointcloud, _, _ = self.get_voxels(xyzs, flags)
        return {
            "pointcloud": downsample_pointcloud(pointcloud, num_points=4096),
            "agent": obs["agent"],
        }

    def process_obs(self, obs: dict) -> dict:
        rgbs, depths, intrinsic_mats, extrinsic_mats = self.get_obs(obs)
        xyzs, flags = self.unproject_images(rgbs, depths, intrinsic_mats, extrinsic_mats)
        voxels, pointcloud, center, scale = self.get_voxels(
            xyzs,
            flags,
            center=torch.tensor(self.cfg.center, dtype=torch.float32, device=self.device),
            scale=self.cfg.scale,
        )

        rgbs = (rgbs * flags).to(torch.uint8)
        alphas = (torch.ones_like(rgbs[..., :1]) * flags * 255).to(torch.uint8)
        images = torch.cat([rgbs, alphas], dim=-1)

        pack = self.extract_visual_features(
            self.dinov2_model,
            voxels,
            scale,
            center,
            image_size=[512, 512],
            images=images[0],
            intrinsics=intrinsic_mats[0],
            extrinsics=extrinsic_mats[0],
        )
        latent = self.encode_latent(self.feature_encoder, pack)

        return {
            "latent": latent,
            "pointcloud": downsample_pointcloud(pointcloud, num_points=4096),
            "center": center,
            "scale": scale,
        }

    def get_obs(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sensor_keys = [key for key in obs["sensor_data"] if "render_camera" in key]
        sensor_keys.sort()
        rgbs = torch.stack([obs["sensor_data"][key]["rgb"] for key in sensor_keys], dim=1)
        depths = torch.stack([obs["sensor_data"][key]["depth"] for key in sensor_keys], dim=1)
        intrinsic_mats = torch.stack([obs["sensor_param"][key]["intrinsic_cv"] for key in sensor_keys], dim=1)
        extrinsic_mats = torch.stack([obs["sensor_param"][key]["extrinsic_cv"] for key in sensor_keys], dim=1)
        return rgbs.cuda(), depths.cuda(), intrinsic_mats.cuda(), extrinsic_mats.cuda()

    def unproject_images(
        self,
        rgbs: torch.Tensor,
        depths: torch.Tensor,
        intrinsic_mats: torch.Tensor,
        extrinsic_mats: torch.Tensor,
        image_size: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_size = image_size or [512, 512]
        n_env, n_camera, h, w, _ = rgbs.shape
        device = rgbs.device
        grid_x = torch.arange(image_size[1], dtype=torch.float32, device=device)
        grid_y = torch.arange(image_size[0], dtype=torch.float32, device=device)
        grid_xy = torch.stack(torch.meshgrid(grid_x, grid_y, indexing="xy"), dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(0).repeat(n_env, n_camera, 1, 1, 1)
        grid_xy = torch.cat([grid_xy, torch.ones_like(grid_xy[..., :1])], dim=-1)
        grid_xy = grid_xy.view(n_env * n_camera, h * w, 3)

        rotation = extrinsic_mats[:, :, :3, :3].view(n_env * n_camera, 3, 3)
        translation = extrinsic_mats[:, :, :3, 3].view(n_env * n_camera, 1, 3)
        intrinsics = intrinsic_mats.view(n_env * n_camera, 3, 3)

        xyz_cam = torch.bmm(grid_xy, torch.linalg.inv(intrinsics).transpose(1, 2))
        xyz_cam = xyz_cam * depths.view(n_env * n_camera, h * w, 1) / 1000.0
        xyz_world = torch.bmm(
            xyz_cam - translation,
            torch.linalg.inv(rotation).transpose(1, 2),
        )

        bbox = self.cfg.bbox
        in_bbox = (
            (xyz_world[..., 0] >= bbox[0])
            & (xyz_world[..., 0] <= bbox[3])
            & (xyz_world[..., 1] >= bbox[1])
            & (xyz_world[..., 1] <= bbox[4])
            & (xyz_world[..., 2] >= bbox[2])
            & (xyz_world[..., 2] <= bbox[5])
        )
        flags = in_bbox.view(n_env, n_camera, h, w, 1)
        return xyz_world.view(n_env, n_camera, h, w, 3), flags

    def get_voxels(
        self,
        xyzs: torch.Tensor,
        flags: torch.Tensor,
        center: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, float]:
        points = xyzs.view(-1, 3)[flags.view(-1)]
        if center is None or scale is None:
            aabb = torch.stack([points.min(dim=0).values, points.max(dim=0).values], dim=0)
            center = (aabb[0] + aabb[1]) / 2.0 if center is None else center
            scale = (aabb[1] - aabb[0]).max().item() if scale is None else scale
        voxels = voxelize_points(points, center=center, scale=scale, resolution=64)
        return voxels, points, center, scale

    @staticmethod
    def extract_visual_features(
        dinov2_model,
        voxels: np.ndarray,
        scale: float,
        center: torch.Tensor,
        image_size: list[int],
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> dict:
        image_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        processed_images = []
        for image in images:
            image = Image.fromarray(image.cpu().numpy()).resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255.0
            if image.shape[2] == 4:
                image = image[:, :, :3] * image[:, :, 3:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            processed_images.append(image_transform(image))
        images = torch.stack(processed_images, dim=0).cuda()

        if extrinsics.shape[1] == 3:
            bottom = torch.tensor([[0, 0, 0, 1]], device=extrinsics.device).expand(extrinsics.shape[0], -1, -1)
            extrinsics = torch.cat([extrinsics, bottom], dim=1).cuda()

        positions = torch.from_numpy(voxels).float().cuda()
        indices = ((positions + 0.5) * 64).long()
        positions = positions * scale + center.float().cuda()

        n_patch = 518 // 14
        features = dinov2_model(images, is_training=True)
        image_size_tensor = torch.tensor(image_size, dtype=torch.float32, device=images.device)
        uv = utils3d.torch.project_cv(positions, extrinsics, intrinsics)[0]
        uv = uv / image_size_tensor.view(1, 1, 2).cuda() * 2 - 1
        patchtokens = features["x_prenorm"][:, dinov2_model.num_register_tokens + 1 :]
        patchtokens = patchtokens.permute(0, 2, 1).reshape(images.shape[0], 1024, n_patch, n_patch)
        patchtokens = F.grid_sample(
            patchtokens,
            uv.unsqueeze(1),
            mode="bilinear",
            align_corners=False,
        ).squeeze(2).permute(0, 2, 1)

        return {
            "indices": indices,
            "patchtokens": torch.mean(patchtokens, dim=0),
        }

    @staticmethod
    def encode_latent(feature_encoder, pack: dict):
        feats = sp.SparseTensor(
            feats=pack["patchtokens"].float(),
            coords=torch.cat(
                [
                    torch.zeros(pack["patchtokens"].shape[0], 1).int().cuda(),
                    pack["indices"].int(),
                ],
                dim=1,
            ),
        ).cuda()
        return feature_encoder(feats, sample_posterior=False)

    def normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        return (states - self.data_stat["state_min"]) / (
            self.data_stat["state_max"] - self.data_stat["state_min"]
        ) * 2 - 1

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions + 1) / 2 * (
            self.data_stat["action_max"] - self.data_stat["action_min"]
        ) + self.data_stat["action_min"]

    @staticmethod
    def _ensure_render_dir(render_dir: Path | None) -> None:
        if render_dir is not None:
            render_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_video(render_dir: Path | None, name: str, frames: list[np.ndarray]) -> None:
        if render_dir is not None and frames:
            imageio.mimsave(render_dir / name, frames, fps=30)


def run_evaluation(cfg: EvalConfig) -> None:

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for policy evaluation but is not visible.")

    policy_cfg = load_policy_config(cfg.config_name)
    if cfg.pred_horizon != policy_cfg.horizon:
        raise ValueError(f"Prediction horizon mismatch: {cfg.pred_horizon} != {policy_cfg.horizon}")

    policy_workspace = RobotWorkspace(policy_cfg)
    agent = ReplayAgent(cfg, policy_workspace)

    success_list = []
    count_success = 0
    seeds = range(cfg.seed_start, cfg.seed_start + cfg.num_seeds)
    for count_all, seed in enumerate(seeds, start=1):
        print(f"Running seed {seed}...")
        success = False
        for trial in range(cfg.num_trials):
            trial_seed = seed * 100 + trial
            torch.manual_seed(trial_seed)
            np.random.seed(trial_seed)
            if cfg.mode == "closed":
                success = agent.take_action_closed(
                    seed=seed,
                    instruction=cfg.instruction,
                    num_step=cfg.num_replans,
                    render_dir=cfg.render_dir,
                )
            else:
                success = agent.take_action_once(
                    seed=seed,
                    instruction=cfg.instruction,
                    num_step=cfg.num_replans,
                    render_dir=cfg.render_dir,
                )
            if success:
                break

        success_list.append(success)
        count_success += int(success)
        print(
            f"Current process - Total {count_all} seeds, success {count_success} seeds, "
            f"success rate: {count_success / count_all:.2f}"
        )

    print(f"Total {cfg.num_seeds} seeds, success {count_success} seeds, success rate: {count_success / cfg.num_seeds:.2f}")
    if cfg.render_dir is not None:
        cfg.render_dir.mkdir(parents=True, exist_ok=True)
        with (cfg.render_dir / "success_list.txt").open("w") as f:
            f.write(str(success_list) + "\n")
            f.write(f"Total {cfg.num_seeds}, Success {count_success}, Success rate {count_success / cfg.num_seeds:.3f}\n")


def main(
    argv: Iterable[str] | None = None,
    *,
    default_env_id: str = "StackCube-v1",
    default_config_name: str = "inverse_dynamics",
    default_checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
) -> None:
    cfg = parse_args(argv, default_env_id, default_config_name, default_checkpoint_path)
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
