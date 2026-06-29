from __future__ import annotations

import hydra
from omegaconf import OmegaConf

from policy.workspace.robotworkspace import RobotWorkspace


DESCRIPTION = "Train the StackCube inverse dynamics model."
DEFAULT_CONFIG_NAME = "inverse_dynamics"


OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path="config", config_name=DEFAULT_CONFIG_NAME)
def main(cfg: OmegaConf) -> None:
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
