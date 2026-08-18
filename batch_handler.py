import sys
sys.path.insert(1, '../atp/')

class BatchHandler:
    def __init__(self, config, data_handler, start=None, stop=None):
        self.config = config
        self.data_handler = data_handler
        self.batch_size = config.args.batch_size

        # `not start` is True for start=0, so an explicit (0, LEN) window was
        # silently replaced by (0, batch_size).
        if start is None or stop is None:
            start = 0
            stop = self.batch_size

        self.start = start
        self.stop = stop

        if self.config.args.patch_model:
            self.base_toks = {
                key: { "input_ids": self.data_handler.base_toks[key]["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.base_toks[key]["attention_mask"][self.start:self.stop]} for key in self.data_handler.base_toks
            }

            self.base_qs_toks = {
                key: { "input_ids": self.data_handler.base_qs_toks[key]["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.base_qs_toks[key]["attention_mask"][self.start:self.stop]} for key in self.data_handler.base_qs_toks
            }

            self.source_qs_toks = {
                key: { "input_ids": self.data_handler.source_qs_toks[key]["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.source_qs_toks[key]["attention_mask"][self.start:self.stop]} for key in self.data_handler.source_qs_toks
            }

            self.response_start_positions = {
                "base": {
                    key: self.data_handler.response_start_positions["base"][key][self.start:self.stop] for key in self.data_handler.response_start_positions["base"]
                }
            }

        elif self.config.args.eval_model:
            if self.config.args.ablation == 'pyreft':
                self.pyreft_toks = {
                    "input_ids": self.data_handler.pyreft_toks["input_ids"][self.start:self.stop],
                    "attention_mask": self.data_handler.pyreft_toks["attention_mask"][self.start:self.stop]
                }

                self.response_start_positions['pyreft'] = self.data_handler.response_start_positions['pyreft'][self.start:self.stop]

            if self.config.args.eval_test:
                self.base_qs_toks = {
                    'test': { "input_ids": self.data_handler.base_qs_toks['test']["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.base_qs_toks['test']["attention_mask"][self.start:self.stop]}
                }
            if self.config.args.eval_transfer:
                self.eval_transfer = {
                    "queries": {
                        'input_ids': self.data_handler.eval_transfer["queries"]['input_ids'][self.start:self.stop],
                        'attention_mask': self.data_handler.eval_transfer["queries"]['attention_mask'][self.start:self.stop]
                    },
                    "answers": self.data_handler.eval_transfer["answers"][self.start:self.stop] if self.data_handler.eval_transfer["answers"] is not None else None
                }

    def update(self, start=None, stop=None):
        if start is None or stop is None:
            self.start = self.start + self.batch_size
            self.stop = self.start + self.batch_size
        else:
            self.start = start
            self.stop = stop
        print(f"Updating batch handler: start={self.start}, stop={self.stop}")

        if self.config.args.patch_model:
            self.base_toks = {
                key: { "input_ids": self.data_handler.base_toks[key]["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.base_toks[key]["attention_mask"][self.start:self.stop]} for key in self.base_toks
            }

            self.source_qs_toks = {
                key: { "input_ids": self.data_handler.source_qs_toks[key]["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.source_qs_toks[key]["attention_mask"][self.start:self.stop]} for key in self.source_qs_toks
            }
            self.response_start_positions = {
                "base": {
                    key: self.data_handler.response_start_positions["base"][key][self.start:self.stop] for key in self.response_start_positions["base"]
                }
            }

            self.base_qs_toks = {
                key: { "input_ids": self.data_handler.base_qs_toks[key]["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.base_qs_toks[key]["attention_mask"][self.start:self.stop]} for key in self.base_qs_toks
            }

        elif self.config.args.eval_model:
            if self.config.args.ablation == 'pyreft':
                self.pyreft_toks = {
                    "input_ids": self.data_handler.pyreft_toks["input_ids"][self.start:self.stop],
                    "attention_mask": self.data_handler.pyreft_toks["attention_mask"][self.start:self.stop]
                }

                self.response_start_positions['pyreft'] = self.data_handler.response_start_positions['pyreft'][self.start:self.stop]
            if self.config.args.eval_test:
                self.base_qs_toks = {
                    'test': { "input_ids": self.data_handler.base_qs_toks['test']["input_ids"][self.start:self.stop], "attention_mask": self.data_handler.base_qs_toks['test']["attention_mask"][self.start:self.stop]}
                }

            if self.config.args.eval_transfer:
                self.eval_transfer = {
                    "queries": {
                        'input_ids': self.data_handler.eval_transfer["queries"]['input_ids'][self.start:self.stop],
                        'attention_mask': self.data_handler.eval_transfer["queries"]['attention_mask'][self.start:self.stop]
                    },
                    "answers": self.data_handler.eval_transfer["answers"][self.start:self.stop] if self.data_handler.eval_transfer["answers"] is not None else None
                }
        