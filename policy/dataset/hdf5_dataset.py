from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import psutil
import torch
from scipy.interpolate import interp1d
from tqdm import tqdm


DATA_STAT = {
    "state_min": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0],
    "state_max": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04],
    "action_min": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.0],
    "action_max": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0],
}


def interpolate_action_sequence(action_sequence: np.ndarray, target_size: int) -> np.ndarray:
    old_indices = np.arange(action_sequence.shape[0])
    new_indices = np.linspace(0, action_sequence.shape[0] - 1, target_size)
    return interp1d(old_indices, action_sequence, kind="linear", axis=0, assume_sorted=True)(new_indices)


def resolve_data_path(path: str) -> Path:
    data_path = Path(path).expanduser()
    if data_path.exists():
        return data_path

    repo_relative = Path(__file__).resolve().parents[2] / data_path
    if repo_relative.exists():
        return repo_relative

    return data_path


class HDF5Dataset:
    def __init__(
        self,
        task: str,
        data_paths: str,
        output_dir: str,
        chunk_size: int = 32,
        interp_ratio: int = 1,
        inv_dyn: bool = True,
        obs_mode: str = "pointcloud",
        num_success_demos: int = 1000,
        num_failure_demos: int = 0,
    ) -> None:
        if task != "StackCube-v1":
            raise ValueError(f"Only StackCube-v1 is supported in the release policy dataset, got {task}.")
        if obs_mode != "pointcloud":
            raise ValueError(f"Only pointcloud observations are supported, got {obs_mode}.")

        self.task = task
        self.data_paths = [resolve_data_path(path.strip()) for path in data_paths.split(",") if path.strip()]
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.interp_ratio = interp_ratio
        self.inv_dyn = inv_dyn
        self.state: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.pointcloud: list[np.ndarray] = []

        print(f"Loading {len(self.data_paths)} StackCube pointcloud dataset file(s): {self.data_paths}")
        for data_path in self.data_paths:
            self._load_file(data_path, num_success_demos, num_failure_demos)

        if not self.state:
            raise RuntimeError("No demonstrations were loaded for policy training.")

        self.state_min = np.array(DATA_STAT["state_min"], dtype=np.float32)
        self.state_max = np.array(DATA_STAT["state_max"], dtype=np.float32)
        self.action_min = np.array(DATA_STAT["action_min"], dtype=np.float32)
        self.action_max = np.array(DATA_STAT["action_max"], dtype=np.float32)
        self._write_data_stats()
        print(f"Loaded {len(self.state)} StackCube episodes.")

    def _load_file(self, data_path: Path, num_success_demos: int, num_failure_demos: int) -> None:
        if self.task not in str(data_path):
            raise ValueError(f"Task {self.task} does not match dataset path {data_path}.")
        if "pd_joint_pos" not in str(data_path):
            raise ValueError("Only pd_joint_pos control data is supported.")

        metadata_path = data_path.with_suffix(".json")
        with metadata_path.open("r") as f:
            metadata = json.load(f)

        count_success_demo = 0
        count_failure_demo = 0
        with h5py.File(data_path, "r") as f:
            trajectories = sorted(f.keys(), key=lambda name: int(name.split("_")[-1]))
            progress = tqdm(enumerate(trajectories), total=len(trajectories), desc="Loading policy data")
            for episode_idx, trajectory_name in progress:
                memory = psutil.virtual_memory()
                progress.set_postfix({"Memory": f"{memory.used / (1024**3):.1f}GB ({memory.percent:.1f}%)"})

                if metadata["episodes"][episode_idx]["success"]:
                    if count_success_demo >= num_success_demos:
                        continue
                    count_success_demo += 1
                else:
                    if count_failure_demo >= num_failure_demos:
                        continue
                    count_failure_demo += 1

                trajectory = f[trajectory_name]
                if "pointcloud" not in trajectory["obs"]:
                    raise KeyError(f"Dataset trajectory {trajectory_name} does not contain obs/pointcloud.")

                self.state.append(trajectory["obs"]["agent"]["qpos"][:])
                self.action.append(trajectory["actions"][:])
                self.pointcloud.append(trajectory["obs"]["pointcloud"][:])

        total = count_success_demo + count_failure_demo
        print(
            f"Loaded {total} / {len(trajectories)} demos from {data_path}, "
            f"{count_success_demo} successful, {count_failure_demo} failed."
        )

    def _write_data_stats(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        stat_path = Path(self.output_dir) / f"{self.task}_data_stat.json"
        with stat_path.open("w") as f:
            json.dump(
                {
                    "state_min": self.state_min.tolist(),
                    "state_max": self.state_max.tolist(),
                    "action_min": self.action_min.tolist(),
                    "action_max": self.action_max.tolist(),
                },
                f,
            )

    def __len__(self) -> int:
        return len(self.state)

    def __getitem__(self, index: int) -> dict:
        return self.get_item(index)

    def get_item(self, index: int | None = None) -> dict:
        if index is None:
            index = np.random.randint(0, len(self))
        return self.parse_hdf5_file(index)

    def parse_hdf5_file(self, index: int) -> dict:
        num_steps = len(self.action[index])
        step_index = np.random.randint(0, num_steps)

        states = (self.state[index] - self.state_min) / (self.state_max - self.state_min) * 2 - 1
        states = states[:, :-1]
        actions = (self.action[index] - self.action_min) / (self.action_max - self.action_min) * 2 - 1

        runtime_chunk_size = self.chunk_size // self.interp_ratio
        if self.inv_dyn and runtime_chunk_size > 2 and np.random.rand() < 0.5:
            runtime_chunk_size = np.random.randint(2, runtime_chunk_size)

        action_sequence = actions[step_index : step_index + runtime_chunk_size]
        base_chunk_size = self.chunk_size // self.interp_ratio
        if action_sequence.shape[0] < base_chunk_size:
            padding = np.tile(action_sequence[-1:], (base_chunk_size - action_sequence.shape[0], 1))
            action_sequence = np.concatenate([action_sequence, padding], axis=0)
        if self.interp_ratio > 1:
            action_sequence = interpolate_action_sequence(action_sequence, self.chunk_size)

        if action_sequence.shape[0] != self.chunk_size:
            raise RuntimeError(f"Expected action chunk length {self.chunk_size}, got {action_sequence.shape[0]}.")

        start_obs_idx = step_index if self.inv_dyn else max(0, step_index - 1)
        end_obs_idx = min(step_index + runtime_chunk_size, self.pointcloud[index].shape[0] - 1) if self.inv_dyn else step_index
        pointclouds = self.pointcloud[index][[start_obs_idx, end_obs_idx]]

        return {
            "obs": {
                "agent_pos": torch.tensor(states[step_index : step_index + 1], dtype=torch.float32),
                "pointcloud": torch.tensor(pointclouds, dtype=torch.float32),
            },
            "action": torch.tensor(action_sequence, dtype=torch.float32),
        }

    def normalize_states(self, states: np.ndarray) -> np.ndarray:
        return (states - self.state_min) / (self.state_max - self.state_min) * 2 - 1

    def denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        return (actions + 1) / 2 * (self.action_max - self.action_min) + self.action_min
