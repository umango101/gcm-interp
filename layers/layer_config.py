"""Config for the layer-level pipeline.

Subclasses the existing ``Config`` rather than editing it, so nothing in the
head-level pipeline changes. The layer-specific flags are stripped from argv by
a pre-parser, the untouched remainder is handed to ``Config.parse_arguments``,
and the extras are merged back onto the resulting namespace.

The output prefix is rerooted at ``--results_root`` (default
``./results_layers``) so layer runs never collide with the head-level tree under
``./results``. Everything below the root keeps the head pipeline's exact layout
--- ``{model}/from_{source}_to_{base}/{algo}/{eval}_eval/{steer}_steer/eval/`` ---
so ``eval_pipeline_bias.py`` can score these runs with only its ``RESULTS_DIR``
constant repointed.
"""

import argparse
import sys

from config import Config


def _int_list(s):
    return [int(x) for x in str(s).replace(",", " ").split()]


class LayerConfig(Config):

    def parse_arguments(self):
        # allow_abbrev=False is load-bearing, not tidiness. With abbreviation on,
        # the base parser's `--steering` flag is a prefix of both --steering_scale
        # and --steering_norm_batch here, so argparse would reject it as an
        # ambiguous option before it ever reached Config.parse_arguments -- i.e.
        # every eval run would die at startup.
        pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        pre.add_argument('--results_root', type=str, default='./results_layers',
                         help='Root for layer-level results. Kept separate from ./results '
                              'so head-level and layer-level runs cannot overwrite each other.')
        pre.add_argument('--topk_layers', type=_int_list, default=[1, 2, 3, 5, 7, 9, 10],
                         help='Layer counts to sweep (counts, not fractions).')
        pre.add_argument('--n_vals', type=_int_list, default=[1, 2, 4, 5, 6, 8, 10],
                         help='Steering multipliers to sweep. Kept integral because the '
                              'scorer regex parses N as an integer out of the gen filename.')
        pre.add_argument('--n_scale', type=float, default=0.1,
                         help='Effective coefficient is N * n_scale. With the default '
                              'relative scale and n_scale=0.1, N=1..10 sweeps 0.1x..1.0x of '
                              'the typical residual norm.')
        pre.add_argument('--steering_scale', type=str, default='relative',
                         choices=['relative', 'raw', 'unit'],
                         help='relative: unit vector rescaled to alpha * mean residual norm '
                              'at that layer. raw: the diff-in-means vector as computed '
                              '(CAA convention). unit: unit norm, matching the head '
                              'pipeline -- negligible against a full residual stream, kept '
                              'only for parity checks.')
        pre.add_argument('--rank_by', type=str, default='cumulative',
                         choices=['cumulative', 'marginal', 'cumulative_abs', 'marginal_abs'],
                         help='cumulative: raw ATP effect at the layer output (includes all '
                              'upstream contributions). marginal: effect[L] - effect[L-1], '
                              'the increment this layer adds.')
        pre.add_argument('--steer_positions', type=str, default='all',
                         choices=['all', 'prompt'],
                         help='Sequence positions the steering vector is written to.')
        pre.add_argument('--steering_norm_batch', type=int, default=8,
                         help='Batch size for the residual-norm reference pass.')
        pre.add_argument('--force_reduce', action='store_true',
                         help='Rebuild the reduced attribution map even if a cache exists.')
        pre.add_argument('--strict_determinism', action='store_true',
                         help='use_deterministic_algorithms(warn_only=False): a '
                              'nondeterministic op raises and names itself instead of '
                              'silently perturbing results. Recommended for localization, '
                              'where the failure mode is plausible-looking numbers.')
        pre.add_argument('--sdp_backend', type=str, default='math',
                         choices=['math', 'default'],
                         help="math (default): disable flash/mem-efficient SDPA so the "
                              "attention backward is deterministic. ATP differentiates "
                              "through attention, and the fused backward kernels "
                              "accumulate with atomics. 'default' leaves dispatch alone -- "
                              "only safe for forward-only runs.")
        pre.add_argument('--limit_items', type=int, default=None,
                         help='Truncate the dataset to this many items. For smoke tests '
                              'and the determinism verification harness.')
        pre.add_argument('--steering_type_override', type=str, default=None,
                         choices=['last_token', 'all_tokens'],
                         help='Override the position the steering vector is read from. '
                              'Config hardcodes last_token; all_tokens takes a '
                              'padding-masked mean over the prompt instead.')

        extra, remaining = pre.parse_known_args()
        saved_argv = sys.argv
        sys.argv = [saved_argv[0]] + remaining
        try:
            args = super().parse_arguments()
        finally:
            sys.argv = saved_argv

        for key, value in vars(extra).items():
            setattr(args, key, value)

        # Config sets steering_type='last_token' inside its own eval branch, so the
        # override has to be applied after -- and only when actually supplied,
        # otherwise a None default would wipe out the base value.
        if args.steering_type_override is not None:
            args.steering_type = args.steering_type_override

        if args.steering_scale == 'unit' and args.n_scale != 1.0:
            print(f"[layers] note: steering_scale=unit with n_scale={args.n_scale}; "
                  f"effective coefficients are {[n * args.n_scale for n in args.n_vals]}")
        return args

    def set_output_prefix(self):
        model = self.args.model_id.split('/')[-1]
        root = self.args.results_root.rstrip('/')
        eval_test_dir = self.args.eval_test.split('/')[-2] if isinstance(self.args.eval_test, str) else ''
        steering_dir = self.args.steering_add_path.split('/')[-2] if self.args.steering_add_path else ''
        if self.args.patch_model:
            self.output_prefix = f"{root}/{model}/from_{self.args.source}_to_{self.args.base}/{self.args.patch_algo}/"
        if self.args.eval_model:
            self.output_prefix = (f"{root}/{model}/from_{self.args.source}_to_{self.args.base}/"
                                  f"{self.args.patch_algo}/{eval_test_dir}_eval/{steering_dir}_steer/")
        print("[layers] output prefix ", self.output_prefix)
        return self.output_prefix

    def localization_dir(self):
        """Where the attribution shards and the reduced map live.

        This is the patch-mode prefix, derived directly from (results_root, model,
        source, base, patch_algo) rather than by slicing the eval prefix backwards
        --- so cross-localization eval runs read the right map, and a change to the
        directory depth can't silently point it somewhere else.
        """
        model = self.args.model_id.split('/')[-1]
        root = self.args.results_root.rstrip('/')
        algo = 'atp' if self.args.patch_algo == 'random' else self.args.patch_algo
        return f"{root}/{model}/from_{self.args.source}_to_{self.args.base}/{algo}/"
