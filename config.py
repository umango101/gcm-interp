from ast import parse
import torch
import os
import random
import datetime
import time
import argparse
import sys
import yaml
from dataclasses import dataclass
import json
from eval.setup import set_seed, assert_determinism_env
class Config:
    def __init__(self):
        self.args = self.parse_arguments()


        self.args.data_path = f"./data/{self.args.model_id.split('/')[-1]}/"

        # The corpus directory. It used to be implied by --source, which forced
        # the directory name to equal the source file prefix -- so several arms
        # holding identically named dev-single-*/user-single-* files could not
        # be addressed separately. --data_dir names the directory; --source and
        # --base keep naming the file prefixes inside it. Defaults to --source,
        # so every existing invocation resolves exactly as before.
        if getattr(self.args, 'data_dir', None) in (None, ''):
            self.args.data_dir = self.args.source

        if self.args.patch_algo == None:
            self.args.patch_algo = 'atp'
        self.setup_environment(seed=self.args.seed)

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Patching')
        parser.add_argument('-d', '--device', type=str, default='cuda:1', required=True, help='Device to run the model on')
        parser.add_argument('-model_id', '--model_id', type=str, required=True, help='Model ID for the model')
        parser.add_argument('-batch_size', '--batch_size', type=int, default=8, required=True, help='Batch size for patching')
        parser.add_argument('-seed', '--seed', type=int, default=42, help='Random seed for reproducibility')
        parser.add_argument('-ablation', '--ablation', type=str, default='steer', help='Apply steering ablation')
        parser.add_argument('-patch_model', '--patch_model', action='store_true', help='Patch the model')
        parser.add_argument('-eval_model', '--eval_model', action='store_true', help='Evaluate the model')
        parser.add_argument("--eval_test", nargs="?", const=True, default=None, help="Evaluate on test set. Optionally provide a test set name/path.")
        parser.add_argument('-eval_train', '--eval_train', action='store_true', help='Evaluate the model on train set')
        parser.add_argument('-eval_transfer', '--eval_transfer', type=str, help='Path to the test dataset for evaluation')
        parser.add_argument('--steering', action='store_true', help='Steering Eval mode')
        parser.add_argument('--pyreft', action='store_true', help='Use PyReFT Eval Mode')
        parser.add_argument('-max_new_tokens', '--max_new_tokens', type=int, default=256, help='Max new tokens to generate during eval')
        parser.add_argument('-patch_algo', '--patch_algo', type=str, help='acp/atp? acp for activation patching, atp for attribution patching')
        parser.add_argument('--data_dir', type=str, default=None,
                            help='Corpus subdirectory under data_path, e.g. '
                                 'devuser. Defaults to --source, which is the '
                                 'previous behaviour.')
        parser.add_argument('-source', '--source', type=str, help='Patch from source')
        parser.add_argument('-base', '--base', type=str, help='Patch to base')
        parser.add_argument('-steering_add_path', '--steering_add_path', type=str, help='steering reps to add')
        parser.add_argument('-steering_sub_path', '--steering_sub_path', type=str, help='steering reps to subtract')
        parser.add_argument('--full_precision', action='store_true',
                            help='Load model in full bfloat16 with device_map=auto (no quantization). '
                                 'Required for very large models (e.g. 72B) that exceed single-GPU memory.')
        parser.add_argument('--head_site', type=str, default='auto',
                            choices=['auto', 'o_proj_input', 'o_proj_output'],
                            help='Where per-head activations are read/written. auto -> '
                                 'o_proj_output when num_heads*head_dim == hidden_size, '
                                 'else o_proj_input (required for gpt-oss, gemma-3).')
        parser.add_argument('--kv_caching', action='store_true', help='Steer prefill only using KV cache; decoding steps are not re-steered')
        parser.add_argument('--results_root', type=str, default='./results',
                            help='Root of the results tree. Was hardcoded to '
                                 './results, which meant two runs differing only '
                                 'in their head list (e.g. a differenced vs an '
                                 'undifferenced localization) resolved to the '
                                 'same output prefix and overwrote each other. '
                                 'Give each condition its own root.')

        args = parser.parse_args()
        if not (args.patch_model or args.eval_model):
            parser.error("At least one of -patch_model, -eval_model is required")
        if args.patch_model or args.eval_model:
            if not args.patch_algo:
                parser.error("-patch_algo argument is required when --patch_model is set")
            if not args.source:
                parser.error("-source argument is required when --patch_model is set")
            if not args.base:
                parser.error("-base argument is required when --patch_model is set")

        if args.eval_model:
            if not args.eval_test:
                args.eval_train = True
            if isinstance(args.eval_test, str) and not os.path.exists(args.eval_test):
                parser.error(f"The provided eval_test path '{args.eval_test}' does not exist.")
            if isinstance(args.eval_test, str):
                args.test_dataset = args.eval_test.split('/')[-2]
                print(f"Steering dataset set to: {args.test_dataset}")
            elif isinstance(args.eval_test, bool) and args.steering:
                args.test_dataset = args.source

            if 'single' in args.test_dataset:
                # 24 (not 3): enough to reach the option letter when a model prefaces its
                # answer ("The answer is (B)"), which the scorer's parse_letter recovers.
                args.max_new_tokens = 24
            elif 'long' in args.test_dataset:
                args.max_new_tokens = 256
            
            args.steering_type = 'last_token'

        return args

    def save_to_yaml(self, file_path, args):
        args_dict = vars(args)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(args_dict, yaml_file, default_flow_style=False)
            
    def setup_environment(self, seed=42):
        # Checked first: CUBLAS_WORKSPACE_CONFIG / PYTHONHASHSEED only take
        # effect if they were exported before the interpreter started.
        assert_determinism_env()
        set_seed(seed)

        os.makedirs(f'{self.set_output_prefix()}', exist_ok=True)
        self.save_to_yaml(f"{self.output_prefix}/config.yml", self.args)
        print(f'Saved config to file {self.get_output_prefix()}/config.yml')

    def get_output_prefix(self):
        return self.output_prefix
    
    def set_output_prefix(self):
        model = self.args.model_id.split('/')[-1]
        # Include the test FILE STEM, not just its parent directory:
        # dev-single-test and devNaive-single-test live in the same arm
        # directory, so keying on the directory alone makes the in-distribution
        # and transfer evals overwrite each other. Kept as ONE path component
        # so load_logits' split('/')[:-3] walk still lands on the algo dir.
        if isinstance(self.args.eval_test, str):
            _parts = self.args.eval_test.split('/')
            eval_test_dir = f"{_parts[-2]}-{_parts[-1].replace('.jsonl', '')}"
        else:
            eval_test_dir = ''
        steering_dir = self.args.steering_add_path.split('/')[-2] if self.args.steering_add_path else ''
        # rstrip('/') so load_logits' split('/')[:-3] walk lands on the algo dir
        # regardless of whether the caller passed a trailing slash.
        root = getattr(self.args, 'results_root', './results').rstrip('/')
        # data_dir goes in the pair component. Without it every arm writes to
        # from_user-single_to_dev-single/ and the localizations overwrite one
        # another. Depth is unchanged, so nothing that walks this path breaks.
        pair = f"{self.args.data_dir}__from_{self.args.source}_to_{self.args.base}"
        if self.args.patch_model:
            self.output_prefix = f"{root}/{model}/{pair}/{self.args.patch_algo}/"
        if self.args.eval_model:
            self.output_prefix = f"{root}/{model}/{pair}/{self.args.patch_algo}/{eval_test_dir}_eval/{steering_dir}_steer/"
        print("op prefix ", self.output_prefix)
        return self.output_prefix
    
    def update_config(self, key, value):
        setattr(self.args, key, value)
        self.save_to_yaml(f"{self.output_prefix}/config.yml", self.args)
