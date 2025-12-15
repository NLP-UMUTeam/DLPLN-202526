"""
ATLAS v2 – Utility functions for Deep Learning experiments
-----------------------------------------------------------

This module groups common helper functions used across different
training and inference scripts when working on the ATLAS cluster.

The goal is to:
- Avoid code duplication
- Standardize good HPC practices (scratch usage, reproducibility)
- Make scripts easier to read and maintain
- Provide clear, pedagogical explanations for students

@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
@author Ronghao Pan <ronghao.pan@um.es>
@author Rafael Valencia-García <valencia@um.es>
"""

import os
import random
from pathlib import Path

import numpy as np
import torch


def ensure_dir (p: Path):
    """
    Ensure that a directory exists on the filesystem.

    This function ensures that a directory exists before using it.
    It is commonly used to prepare scratch, cache, and output folders.

    If the directory (or any of its parents) does not exist,
    it will be created. If it already exists, nothing happens.

    This helper avoids repetitive checks such as:
        if not os.path.exists(...)

    Args:
        p (Path): Path object representing the directory to create.

    Returns:
        Path: The same Path object, for convenience and chaining.
    """
    p.mkdir (parents = True, exist_ok = True)
    return p


def setup_hf_caches ():
    """
    Prepare cache directories for Hugging Face, PyTorch, and temporary files.

    In ATLAS, disk quota in $HOME is very limited, so all heavy caches
    (models, datasets, tokenizers) should live in /scratch/<user>.

    This function:
    - Reads environment variables if already defined (recommended)
    - Falls back to /scratch/<user> if not
    - Creates all required directories safely

    Returns:
        tuple:
            - hf_home (Path): Base directory for Hugging Face caches
            - scratch_base (Path): Base scratch directory for the user
    """
    user = os.getenv ("USER", "user")

    scratch_base = Path (
        os.getenv ("SCRATCH_DIR", f"/scratch/{user}")
    )

    hf_home = Path (
        os.getenv ("HF_HOME", scratch_base / ".hf")
    )

    xdg_cache = Path (
        os.getenv ("XDG_CACHE_HOME", scratch_base / ".cache")
    )

    torch_home = Path (
        os.getenv ("TORCH_HOME", scratch_base / ".torch")
    )

    tmpdir = Path (
        os.getenv ("TMPDIR", scratch_base / "tmp")
    )

    # Create directories following Hugging Face recommendations
    ensure_dir (hf_home / "hub")
    ensure_dir (hf_home / "datasets")
    ensure_dir (hf_home / "transformers")

    # Generic caches
    ensure_dir (xdg_cache)
    ensure_dir (torch_home)
    ensure_dir (tmpdir)

    # Output directory for experiments
    ensure_dir (scratch_base / "out")

    return hf_home, scratch_base


def set_seed (seed: int = 42):
    """
    This function enforces reproducibility by fixing random seeds
    across Python, NumPy, and PyTorch (CPU and GPU).
    
    Fix random seeds to improve reproducibility of experiments.

    This function affects:
    - Python's built-in random module
    - NumPy
    - PyTorch (CPU and CUDA)

    It also disables certain CUDA optimizations that introduce
    non-determinism.

    Args:
        seed (int): Seed value to use. Default is 42.
    """
    random.seed (seed)
    np.random.seed (seed)

    torch.manual_seed (seed)
    torch.cuda.manual_seed_all (seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device ():
    """
    Determine the computation device (CPU or GPU).

    This function selects the correct computation device (CPU or GPU)
    and prints useful diagnostic information.

    If CUDA is available, the function selects the GPU corresponding
    to LOCAL_RANK (useful for multi-GPU or distributed setups).
    Otherwise, it falls back to CPU.

    The function also prints diagnostic information to help
    debugging in HPC environments.

    Returns:
        torch.device: Selected device object.
    """
    local_rank = int (os.environ.get ("LOCAL_RANK", 0))

    if torch.cuda.is_available ():
        torch.cuda.set_device (local_rank)
        dev = torch.device (f"cuda:{local_rank}")
    else:
        dev = torch.device ("cpu")

    # Informative logging (very useful in clusters)
    print (
        f"[CUDA] is_available={torch.cuda.is_available()} "
        f"| device={dev} "
        f"| count={torch.cuda.device_count()}"
    )

    if torch.cuda.is_available ():
        print (
            f"[CUDA] name={torch.cuda.get_device_name (dev.index)} "
            f"| torch.version.cuda={getattr (torch.version, 'cuda', None)}"
        )

    return dev
