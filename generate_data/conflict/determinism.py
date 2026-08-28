"""Pin determinism for anything that runs the model, and record what it ran on.

Dataset CONSTRUCTION is already deterministic: hierarchy_common uses no RNG, and
two builds under different PYTHONHASHSEED values are byte-identical. What is not
deterministic by default is everything downstream of a forward pass -- and QC
decides which colour pairs enter the corpus from model logits, so an unpinned QC
run makes the corpus itself irreproducible even though the builder is not.

Three env vars have to be set BEFORE the interpreter starts, because cuBLAS
reads its workspace config at init and Python reads PYTHONHASHSEED at startup.
Setting them from inside Python is too late, which is why this raises instead of
setting them. run.py enforces the same three; this keeps the auxiliary scripts
on the same footing.

    import determinism
    fp = determinism.enforce(seed=0)     # before loading the model
    ...
    json.dump({"fingerprint": fp, ...}, f)

The fingerprint belongs in every output file. Bitwise reproducibility is
conditional on the hardware and library stack -- a different GPU count changes
reduction order, and a different transformers version can change the MXFP4
dequantisation path -- so a result that cannot be reproduced later is only
diagnosable if you recorded what produced it.
"""

import os
import sys
import random
import hashlib

REQUIRED_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
}


def check_env(strict=True):
    """Verify the pre-start environment. Returns the list of problems."""
    bad = []
    for k, want in REQUIRED_ENV.items():
        got = os.environ.get(k)
        if got != want:
            bad.append((k, got, want))
    if bad and strict:
        lines = "\n".join(f"  export {k}=\"{want}\"    (currently {got!r})"
                          for k, got, want in bad)
        raise RuntimeError(
            "determinism env not set before interpreter start:\n" + lines +
            "\n\nThese must be exported in the shell, not set in Python: cuBLAS "
            "reads its\nworkspace config at initialisation and Python reads "
            "PYTHONHASHSEED at startup.\nPass strict=False only if you accept "
            "that the run is not reproducible.")
    return bad


def enforce(seed=0, strict=True, allow_nondeterministic=False):
    """Seed everything, pin deterministic kernels, return a fingerprint dict."""
    problems = check_env(strict=strict)

    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # warn_only=True lets an op without a deterministic kernel run anyway. Some
    # MoE routing and scatter ops do not have one, so a hard failure here is a
    # real possibility -- but it is recorded in the fingerprint rather than
    # silently tolerated, because a run with warn_only is not bitwise
    # reproducible and should not be reported as if it were.
    try:
        torch.use_deterministic_algorithms(True, warn_only=allow_nondeterministic)
        det_mode = "warn_only" if allow_nondeterministic else "strict"
    except Exception as e:                                    # noqa: BLE001
        det_mode = f"unavailable: {type(e).__name__}"

    fp = {
        "seed": seed,
        "deterministic_algorithms": det_mode,
        "env_problems": [{"var": k, "got": g, "want": w} for k, g, w in problems],
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_names": ([torch.cuda.get_device_name(i)
                          for i in range(torch.cuda.device_count())]
                         if torch.cuda.is_available() else []),
    }
    try:
        import transformers
        fp["transformers"] = transformers.__version__
    except Exception:                                          # noqa: BLE001
        pass

    if det_mode != "strict":
        print(f"[determinism] WARNING: deterministic algorithms are "
              f"'{det_mode}'. This run is not guaranteed bitwise reproducible.")
    return fp


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hash_dir(path, suffix=".jsonl"):
    """{filename: sha256} for every matching file, sorted by name.

    Lets a later run prove it rebuilt the same corpus rather than assuming it,
    which matters because the builder is deterministic but the QC gate that
    feeds it is not.
    """
    out = {}
    for name in sorted(os.listdir(path)):
        if name.endswith(suffix):
            out[name] = sha256_file(os.path.join(path, name))
    return out
