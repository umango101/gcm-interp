"""Determinism hardening and run fingerprinting for the layer-level pipeline.

``determinism.py`` covers the generation path: it pins cuBLAS workspace, cuDNN
algorithm choice, and calls ``use_deterministic_algorithms(True, warn_only=True)``.
That was enough for the head pipeline's forward-only steering runs. It is not
enough here, for two reasons specific to this experiment.

1. SDPA BACKWARD. The localization pass differentiates through attention.
   ModelHandler only forces ``attn_implementation="eager"`` on the pyreft path,
   so everywhere else transformers picks SDPA, and SDPA's flash and
   mem-efficient backward kernels accumulate with atomics -- run-to-run
   nondeterministic by construction. Under ``warn_only=True`` that does not even
   raise; it silently perturbs every gradient, which is to say every attribution
   value, which is to say the layer ranking itself. ``harden()`` pins SDPA to the
   math backend, whose backward is deterministic.

2. warn_only=True IS A SILENT FAILURE MODE. It was the right call for generation
   -- fail-open beats aborting on an op that may not affect the output. For a
   ranking derived from gradients it is the wrong default, because the thing that
   goes wrong produces plausible numbers rather than an error. ``--strict_determinism``
   flips it to fail-closed so an offending op raises and names itself.

Neither of these changes the head pipeline: nothing here is imported by run.py.
"""

import hashlib
import json
import os


def harden(sdp_backend='math', strict=False, verbose=True):
    """Pin the kernel choices that ``determinism.py`` leaves open.

    sdp_backend='math'    -- disable flash and mem-efficient SDPA, forcing the
                             math backend. Slower and heavier on memory, but its
                             backward is deterministic. Required for localization.
    sdp_backend='default' -- leave SDPA dispatch alone. Only appropriate for
                             forward-only work you have separately convinced
                             yourself is deterministic.
    strict=True           -- use_deterministic_algorithms(..., warn_only=False),
                             so a nondeterministic op raises instead of warning.
    """
    import torch

    if strict:
        torch.use_deterministic_algorithms(True, warn_only=False)

    applied = {}
    if sdp_backend == 'math':
        for name, value in (('enable_flash_sdp', False),
                            ('enable_mem_efficient_sdp', False),
                            ('enable_math_sdp', True)):
            fn = getattr(torch.backends.cuda, name, None)
            if fn is not None:
                fn(value)
                applied[name] = value
    elif sdp_backend != 'default':
        raise ValueError(f"Unknown sdp_backend: {sdp_backend!r}")

    if verbose:
        print(f"[determinism] sdp_backend={sdp_backend} applied={applied} "
              f"strict={strict} "
              f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}",
              flush=True)
    return applied


def _version(module_name):
    try:
        mod = __import__(module_name)
        return getattr(mod, '__version__', 'unknown')
    except Exception:
        return 'absent'


def env_fingerprint(config):
    """Everything that can move a bit between two runs of the same command.

    Written next to the results so a diff between two runs can be explained
    rather than guessed at. Package versions are in here because `syc` has
    drifted transformers between 4.53.3 and 4.57.1 more than once, and an
    attribution map built under one is not comparable to one built under the
    other.
    """
    import torch

    device_name = 'cpu'
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    def _sdp(name):
        fn = getattr(torch.backends.cuda, name, None)
        try:
            return bool(fn()) if fn is not None else None
        except Exception:
            return None

    return {
        'torch': _version('torch'),
        'transformers': _version('transformers'),
        'nnsight': _version('nnsight'),
        'numpy': _version('numpy'),
        'cuda': getattr(torch.version, 'cuda', None),
        'device_name': device_name,
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'flash_sdp': _sdp('flash_sdp_enabled'),
        'mem_efficient_sdp': _sdp('mem_efficient_sdp_enabled'),
        'math_sdp': _sdp('math_sdp_enabled'),
        'seed': config.args.seed,
        'batch_size': config.args.batch_size,
        'strict_determinism': getattr(config.args, 'strict_determinism', None),
        'sdp_backend': getattr(config.args, 'sdp_backend', None),
        'no_deterministic': config.args.no_deterministic,
        'model_id': config.args.model_id,
    }


def write_fingerprint(config, out_dir, name='determinism.json'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fp = env_fingerprint(config)
    with open(path, 'w') as f:
        json.dump(fp, f, indent=2, sort_keys=True)
    if fp['no_deterministic']:
        print("[determinism] WARNING: --no_deterministic is set. Generation is "
              "nondeterministic; arms produced with and without it are not comparable.")
    elif fp['flash_sdp'] or fp['mem_efficient_sdp']:
        print("[determinism] WARNING: a fused SDPA backend is still enabled. Its "
              "backward accumulates with atomics, so attribution values will move "
              "between runs. Pass --sdp_backend math for the localization pass.")
    return fp


def fingerprint_tensor(t):
    """Stable content hash of a tensor, for run-to-run comparison."""
    import torch
    if isinstance(t, torch.Tensor):
        arr = t.detach().to('cpu').to(torch.float32).contiguous().numpy()
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    return hashlib.sha256(repr(t).encode()).hexdigest()[:16]


def fingerprint_json(obj):
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
