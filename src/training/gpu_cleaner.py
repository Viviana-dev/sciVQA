import gc
import sys
import time
from os import path

import torch
from accelerate import Accelerator

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.logging_utils import setup_logger

logger = setup_logger(__name__)


def clear_memory(accelerator: Accelerator | None = None):
    """Clear GPU memory and print memory usage statistics."""
    if accelerator:
        accelerator.free_memory()

    gc.collect()
    time.sleep(1)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(1)
    gc.collect()
    time.sleep(1)

    logger.info(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    logger.info(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
