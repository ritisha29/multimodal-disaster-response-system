"""
Kaggle and Local Path Configuration Module for CrisisMMD Disaster Project.

Provides automatic environment detection (Kaggle vs. Local) and configurable
paths for datasets, model checkpoints, and evaluation results.

Features:
- Exact confirmed Kaggle dataset and model input paths:
    KAGGLE_MODELS_PATH = "/kaggle/input/datasets/ritishaburugupalli/urop-models/models/"
    KAGGLE_DATA_PATH = "/kaggle/input/datasets/ritishaburugupalli/urop-dataset/data/"
- Auto-detects Kaggle environment (/kaggle, KAGGLE_KERNEL_RUN_TYPE) or local setup.
- Directs writable outputs (newly trained models, results) to /kaggle/working/ on Kaggle.
- Falls back gracefully to local project directories (data/, models/, results/).
- Supports manual overrides via environment variables or function arguments.
"""

import os
from typing import Optional, List


# Confirmed real Kaggle input paths
KAGGLE_MODELS_PATH: str = "/kaggle/input/datasets/ritishaburugupalli/urop-models/models/"
KAGGLE_DATA_PATH: str = "/kaggle/input/datasets/ritishaburugupalli/urop-dataset/data/"


def is_kaggle_environment() -> bool:
    """
    Determines if the current process is executing within a Kaggle Notebook.

    Returns:
        bool: True if on Kaggle, False otherwise.
    """
    # Explicit user overrides via environment variable
    if os.environ.get("FORCE_KAGGLE", "").strip().lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("FORCE_LOCAL", "").strip().lower() in ("1", "true", "yes"):
        return False

    # Standard Kaggle environment markers
    return (
        os.path.exists("/kaggle") or
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ or
        "KAGGLE_URL_BASE" in os.environ
    )


def _get_project_root() -> str:
    """Returns the local project root directory (parent of src/)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def get_data_dir(custom_path: Optional[str] = None) -> str:
    """
    Resolves the dataset directory containing CrisisMMD TSV splits and images.

    Search priority:
    1. custom_path (if provided and exists)
    2. DATA_DIR or KAGGLE_DATA_DIR environment variable
    3. Confirmed Kaggle dataset path (KAGGLE_DATA_PATH)
    4. Kaggle input directory auto-discovery (/kaggle/input/<dataset-name>)
    5. Local project data directory (<project_root>/data)
    6. Working directory fallback (./data)

    Args:
        custom_path (str, optional): User-supplied path override.

    Returns:
        str: Resolved absolute path to the dataset directory.
    """
    if custom_path and os.path.exists(custom_path):
        return os.path.abspath(custom_path)

    # Check environment variable overrides
    env_data = os.environ.get("DATA_DIR") or os.environ.get("KAGGLE_DATA_DIR")
    if env_data and os.path.exists(env_data):
        return os.path.abspath(env_data)

    # Kaggle environment resolution
    if is_kaggle_environment():
        # 1. Primary: Exact confirmed Kaggle dataset path
        if os.path.exists(KAGGLE_DATA_PATH):
            return KAGGLE_DATA_PATH

        # 2. Check specific dataset name if configured
        kaggle_input = "/kaggle/input"
        named_dataset = os.environ.get("KAGGLE_DATASET_NAME", "").strip()
        if named_dataset:
            cand = f"{kaggle_input}/{named_dataset}"
            if os.path.exists(cand):
                return cand

        # 3. Auto-discover datasets mounted inside /kaggle/input
        if os.path.exists(kaggle_input):
            try:
                subdirs = [
                    f"{kaggle_input}/{d}"
                    for d in os.listdir(kaggle_input)
                    if os.path.isdir(os.path.join(kaggle_input, d))
                ]
            except Exception:
                subdirs = []

            for path in subdirs:
                bname = os.path.basename(path).lower()
                if "crisis" in bname or "disaster" in bname or "urop" in bname:
                    return path

            for path in subdirs:
                try:
                    contents = [c.lower() for c in os.listdir(path)]
                    if any("crisismmd" in c or "data_image" in c or c.endswith(".tsv") for c in contents):
                        return path
                except Exception:
                    continue

            if len(subdirs) == 1:
                return subdirs[0]

        # Default Kaggle fallback
        return KAGGLE_DATA_PATH

    # Local environment resolution
    project_root = _get_project_root()
    local_data = os.path.join(project_root, "data")
    if os.path.exists(local_data):
        return os.path.abspath(local_data)

    return os.path.abspath("data")


def get_models_dir(custom_path: Optional[str] = None, writeable: bool = False) -> str:
    """
    Resolves the directory for saving and loading model checkpoints (.pt).

    On Kaggle:
    - If writeable=True (saving new weights): returns /kaggle/working/models.
    - If loading pre-trained weights: prefers KAGGLE_MODELS_PATH if it exists,
      otherwise defaults to /kaggle/working/models.
    Locally:
    - Defaults to <project_root>/models.

    Args:
        custom_path (str, optional): User-supplied path override.
        writeable (bool): If True, returns a writeable directory (e.g. for saving).

    Returns:
        str: Resolved path to the models directory.
    """
    if custom_path:
        return os.path.abspath(custom_path)

    env_models = os.environ.get("MODELS_DIR") or os.environ.get("KAGGLE_MODELS_DIR")
    if env_models:
        return os.path.abspath(env_models)

    if is_kaggle_environment():
        if writeable:
            return "/kaggle/working/models"
        if os.path.exists(KAGGLE_MODELS_PATH):
            return KAGGLE_MODELS_PATH
        return "/kaggle/working/models"

    project_root = _get_project_root()
    return os.path.abspath(os.path.join(project_root, "models"))


def get_results_dir(custom_path: Optional[str] = None) -> str:
    """
    Resolves the directory for saving evaluation figures, plots, and logs.

    On Kaggle, defaults to /kaggle/working/results.
    Locally, defaults to <project_root>/results.

    Args:
        custom_path (str, optional): User-supplied path override.

    Returns:
        str: Resolved path to the results directory.
    """
    if custom_path:
        return os.path.abspath(custom_path)

    env_results = os.environ.get("RESULTS_DIR") or os.environ.get("KAGGLE_RESULTS_DIR")
    if env_results:
        return os.path.abspath(env_results)

    if is_kaggle_environment():
        return "/kaggle/working/results"

    project_root = _get_project_root()
    return os.path.abspath(os.path.join(project_root, "results"))


def resolve_checkpoint_path(filename: str, models_dir: Optional[str] = None) -> str:
    """
    Finds the exact path for a model checkpoint file.

    Checks:
    1. Inside models_dir if explicitly provided.
    2. Inside exact confirmed Kaggle models path (KAGGLE_MODELS_PATH).
    3. Inside /kaggle/working/models/ (if trained in the current session).
    4. Inside local <project_root>/models/ if running locally.
    5. On Kaggle: searches /kaggle/input recursively as fallback.

    Args:
        filename (str): Name of the checkpoint file, e.g. 'fusion_model.pt'.
        models_dir (str, optional): Base models directory to search first.

    Returns:
        str: Path to the checkpoint (existing if found, or expected target path).
    """
    # 1. Check explicit models_dir if supplied
    if models_dir:
        direct_path = os.path.join(models_dir, filename)
        if os.path.exists(direct_path):
            return direct_path

    # 2. Check exact confirmed Kaggle models input path
    kaggle_exact = os.path.join(KAGGLE_MODELS_PATH, filename)
    if os.path.exists(kaggle_exact):
        return kaggle_exact

    # 3. Check writable /kaggle/working/models/
    working_path = os.path.join("/kaggle/working/models", filename)
    if os.path.exists(working_path):
        return working_path

    # 4. Check standard get_models_dir()
    base_dir = models_dir or get_models_dir()
    direct_path = os.path.join(base_dir, filename)
    if os.path.exists(direct_path):
        return direct_path

    # 5. If running on Kaggle, search /kaggle/input recursively for the checkpoint
    if is_kaggle_environment():
        kaggle_input = "/kaggle/input"
        if os.path.exists(kaggle_input):
            for root, _, files in os.walk(kaggle_input):
                if filename in files:
                    return os.path.join(root, filename)
        return kaggle_exact

    return direct_path


# Module-level convenience constants
IS_KAGGLE: bool = is_kaggle_environment()
DATA_DIR: str = get_data_dir()
MODELS_DIR: str = get_models_dir()
RESULTS_DIR: str = get_results_dir()


def print_path_summary():
    """Prints active runtime environment and resolved directory paths."""
    print("=" * 65)
    print(" CrisisMMD Runtime Environment Configuration")
    print("=" * 65)
    print(f" Environment             : {'KAGGLE NOTEBOOK' if is_kaggle_environment() else 'LOCAL SYSTEM'}")
    print(f" Confirmed Kaggle Data   : {KAGGLE_DATA_PATH}")
    print(f" Confirmed Kaggle Models : {KAGGLE_MODELS_PATH}")
    print(f" Active Data Dir         : {get_data_dir()}")
    print(f" Active Models Dir       : {get_models_dir()}")
    print(f" Active Results Dir      : {get_results_dir()}")
    print("=" * 65)


if __name__ == "__main__":
    print_path_summary()
