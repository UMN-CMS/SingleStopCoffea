import os
import logging.config
import yaml
import analyzer.resources
import importlib.resources
from pathlib import Path


def setup_logging(
    default_path=None,
    default_level=logging.INFO,
    env_key="LOG_CFG",
):
    if default_path is None:
        with importlib.resources.as_file(
            importlib.resources.files(analyzer.resources)
        ) as f:
            default_path = Path(f) / "logging_config.yaml"
    path = Path(default_path)
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
