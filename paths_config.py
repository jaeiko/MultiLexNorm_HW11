"""Centralized project path configurations and dynamic import management.

This module defines absolute paths for key directories and resources in the
MultiLexNorm2026 workspace and configures the sys.path environment to ensure
smooth, error-free dynamic imports across different modules and runtimes.
"""

import sys
import logging
from pathlib import Path

# Configure a standard logger for path setup
logger = logging.getLogger("paths_config")
logging.basicConfig(level=logging.INFO, format="[PathsConfig] %(levelname)s: %(message)s")

# 1. Project Root Directory Determination
# Finds the parent directory of this config file, which is the repository root
ROOT_DIR: Path = Path(__file__).parent.resolve()

# 2. Key Directories and Resource Paths
DATASET_DIR: Path = ROOT_DIR / "multilexnorm2026-dataset"
MFR_STATS_PATH: Path = ROOT_DIR / "mfr_stats.pkl.gz"
TRIGRAM_STATS_PATH: Path = ROOT_DIR / "outputs" / "trigram_stats.pkl.gz"
PROMPT_MFR_DICT_DIR: Path = ROOT_DIR / "prompt_mfr_dictionary"
COMMON_PROMPT_DIR: Path = PROMPT_MFR_DICT_DIR / "common_prompt_v2_package" / "prompts"


def setup_imports() -> None:
    """Configures system paths to support modular project imports dynamically.

    This function inserts the project root directory and custom prompt package
    paths into `sys.path` so that nested baseline or pipeline scripts can load
    shared dependencies seamlessly without relative path errors.
    """
    paths_to_add = [
        str(ROOT_DIR),
        str(COMMON_PROMPT_DIR)
    ]
    
    for path_str in paths_to_add:
        if path_str not in sys.path:
            # Insert at the beginning to prioritize local workspace packages
            sys.path.insert(0, path_str)
            logger.info(f"Registered path to sys.path: {path_str}")
        else:
            logger.debug(f"Path already exists in sys.path: {path_str}")
