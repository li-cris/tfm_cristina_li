import logging
import subprocess
from typing import Optional

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def log_config(cfg: DictConfig) -> None:
    """Logs the configuration in YAML format."""
    log.info("")
    log.info("Configuration:")
    log.info("--------------")
    yaml_str = OmegaConf.to_yaml(cfg=cfg)
    for line in yaml_str.splitlines():
        log.info(f"{line}")
    log.info("")


def get_git_root() -> Optional[str]:
    """Returns the root directory of the current Git repository."""
    try:
        return subprocess.check_output(
            args=["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as e:
        log.error(f"Failed to get Git root: {e}")
        return None
