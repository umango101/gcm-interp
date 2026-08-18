import sys
from determinism import set_cublas_env, enable_determinism
set_cublas_env()
from config import Config
from model_handler import ModelHandler
from data_handler import DataHandler
from experiment import Experiment
from patching_utils import PatchingUtils
from patching import Patching
import os
from eval.eval_runner import *
import logging
logging.basicConfig(level=logging.WARNING)
from batch_handler import BatchHandler
from patching import Patching
def main():
    print('Parsing config...')
    config = Config()
    if not config.args.no_deterministic:
        enable_determinism()
    print('Loading model...')
    model_handler = ModelHandler(config)
    config.args.batch_size = 5
    data_handler = DataHandler(config, model_handler)
    if config.args.patch_algo == 'acp':
        config.args.batch_size = config.args.batch_size = max(x for x in range(64, 0, -1) if data_handler.LEN % x != 1)
    else:
        config.args.batch_size = 1

    if config.args.patch_model:
        if config.args.patch_algo == 'probes':
            probes_experiment = Experiment(config, data_handler, model_handler, 'heads')
            probes_experiment.run_probes()
        else:
            print('Running patching on heads...')
            heads_experiment = Experiment(config, data_handler, model_handler, 'heads')
            heads_experiment.run()

    if config.args.eval_model:
        config.args.batch_size = max(x for x in range(16, 0, -1) if data_handler.LEN % x != 1)
        if config.args.model_id == 'allenai/OLMo-2-1124-13B-DPO':
            if config.args.source == 'hate':
                config.args.batch_size = max(x for x in range(8, 0, -1) if data_handler.LEN % x != 1)
        batch_size = config.args.batch_size
        batch_handler = BatchHandler(config, data_handler, 0, min(batch_size, data_handler.LEN))
        patching = Patching(model_handler, batch_handler, config)
        patching_utils = PatchingUtils(patching)
        if config.args.pyreft:
            run_eval_pyreft(config, data_handler, model_handler, batch_handler)
        elif config.args.steering:
            run_eval(config, data_handler, model_handler, batch_handler, patching_utils, 'heads')
        elif config.args.eval_transfer:
            data_handler.LEN = min(data_handler.LEN, 100)
            run_eval_transfer(config, data_handler, model_handler, batch_handler, patching_utils)

if __name__ == "__main__":
    main()
