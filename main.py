import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from causal_hts_modeling.utils import log_config, get_git_root

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    try:
        log_config(cfg=cfg)
        log.info(f"User: {cfg.info.user}")
        log.info(f"Git root: {get_git_root()}")
    except Exception as e:
        log.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
