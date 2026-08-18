import os
import torch
import numpy as np
import random
import logging
import warnings
warnings.filterwarnings("ignore")

# Set before the first CUDA op, i.e. before python starts. Kept here as the
# single source of truth so the shell script and the assertion agree.
REQUIRED_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # deterministic cuBLAS GEMM workspaces
    "PYTHONHASHSEED": "0",                 # str hashing -> set/dict iteration order
    "TOKENIZERS_PARALLELISM": "false",     # rust tokenizer thread pool
}


def assert_determinism_env(strict=None):
    """Fail loudly if the process was not launched with a deterministic env.

    CUBLAS_WORKSPACE_CONFIG and PYTHONHASHSEED are read by cuBLAS and by the
    interpreter at startup respectively. Setting them from inside Python (as
    the old set_seed did, via os.environ.setdefault) is too late to have any
    effect, and the run looks deterministic while not being so. Export them in
    the SLURM script instead; this only checks.
    """
    if strict is None:
        # "2" is stricter than "1", not a different mode -- it must not turn the
        # env check OFF.
        strict = os.environ.get("STRICT_DETERMINISM", "1") in ("1", "2")
    bad = {k: (os.environ.get(k), v) for k, v in REQUIRED_ENV.items()
           if os.environ.get(k) != v}
    if not bad:
        return
    msg = ("determinism env not set before interpreter start (got, want): "
           f"{bad}\nExport these in the launcher, e.g.\n"
           + "\n".join(f'  export {k}="{v}"' for k, v in REQUIRED_ENV.items()))
    if strict:
        raise RuntimeError(msg)
    print("[determinism] WARNING: " + msg)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # TF32 and reduced-precision accumulation change results depending on which
    # kernel autotuning picks, which varies with free memory and shape. Pin them
    # off: slower matmuls, identical numbers across runs and across nodes.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    # warn_only=False is the real guarantee, but gpt-oss's MoE routing uses
    # scatter/index_add kernels with no deterministic implementation, so it
    # raises. Default to warn_only=True and let STRICT_DETERMINISM=2 opt into
    # the hard failure when you want to find out exactly which op is at fault.
    hard = os.environ.get("STRICT_DETERMINISM") == "2"
    torch.use_deterministic_algorithms(True, warn_only=not hard)


def configure_logging():
    logging.basicConfig(level=logging.ERROR)
