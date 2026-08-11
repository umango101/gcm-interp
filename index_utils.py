from ctypes import alignment
import token
import torch

class IndexUtils:
    def __init__(self, model_handler, data_handler):
        self.model_handler = model_handler
        self.data_handler = data_handler
        self.align_toks = self.data_handler.align_toks
        
    def get_indices(self, base_toks, source_toks, token_variable, span):
        token_indices = None
        if token_variable == 'diff':
            token_indices = torch.tensor(self.get_differing_positions(base_toks, source_toks))
        elif token_variable == 'asst':
            token_indices = torch.tensor(self.get_assistant_tokens(base_toks, source_toks))
        elif token_variable == 'asst_mn1':
            token_indices = torch.tensor(self.get_assistant_tokens(base_toks, source_toks))
            token_indices = token_indices - 1
        elif token_variable == 'asst_mn2':
            token_indices = torch.tensor(self.get_assistant_tokens(base_toks, source_toks))
            token_indices = token_indices + 1
        elif token_variable == 'user_start':
            token_indices = torch.tensor(self.get_user_tokens(base_toks, source_toks))
        elif token_variable == 'user_mn1':
            token_indices = torch.tensor(self.get_user_tokens(base_toks, source_toks))
            token_indices = token_indices - 1
        elif token_variable == 'user_mn2':
            token_indices = torch.tensor(self.get_user_tokens(base_toks, source_toks))
            token_indices = token_indices + 1
        elif token_variable == 'user_end':
            token_indices = torch.tensor(self.get_assistant_tokens(base_toks, source_toks))
            token_indices = token_indices - 3
        elif token_variable == 'sys_end':
            token_indices = torch.tensor(self.get_user_tokens(base_toks, source_toks))
            token_indices = token_indices - 2
        elif token_variable == 'sys_mn':
            token_indices = torch.tensor(self.get_user_tokens(base_toks, source_toks))
            token_indices = token_indices - 3
        else:
            print(token_variable)
            token_indices = torch.tensor(list(token_variable))
        
        print('SPAN IS ', span)
        if span is not None:
            offset = torch.arange(span)
            token_indices = token_indices[:, None] + offset
            print(token_indices.size())

        assert token_indices is not None, "Token index not found"
        print(token_indices.size())
        return token_indices
    
    def get_differing_positions(self, base_toks, source_toks):
        tokenizer = self.model_handler.tokenizer
        source_toks = self.align_toks(source_toks, base_toks)
        if source_toks['input_ids'].shape != base_toks['input_ids'].shape:
            raise ValueError("Both batches must have the same shape.")
        
        differing_positions = []
        for source, base in zip(source_toks['input_ids'], base_toks['input_ids']):
            differing_indices = (source != base).nonzero()[0].item()
            print('---------------------------------------------------------------')
            print('DIFFERENT TOKENS: #SOURCE ', tokenizer.decode(source[differing_indices]), ' #BASE ', tokenizer.decode(base[differing_indices]))
            print('---------------------------------------------------------------')
            differing_positions.append(differing_indices)
        return differing_positions

    def get_assistant_tokens(self, base_toks, source_toks):
        source_toks = self.align_toks(source_toks, base_toks)
        source_toks = source_toks['input_ids']
        base_toks = base_toks['input_ids']
        tokenizer = self.model_handler.tokenizer
        # print('newline ', tokenizer.encode('\n'))
        alignment_tokens = self.model_handler.alignment_tokens
        pad_token_id = self.model_handler.tokenizer.pad_token_id
        asst_indices = []
        print(alignment_tokens)
        for idx, source in enumerate(source_toks):
            # print('source ', source.shape)
            len_eoq = len(asst_indices)
            for i in range(len(source) - 1, -1, -1):
                if source[i] == alignment_tokens[1]:
                    print('---------------------------------------------------------')
                    print('ASST TOKENS: SOURCE ', tokenizer.decode(source[i]), ' BASE ', tokenizer.decode(base_toks[idx][i]))
                    print('---------------------------------------------------------')
                    asst_indices.append(i)
                    break
            if len(asst_indices) == len_eoq:
                assert False, "End of query not found"
        
        assert len(asst_indices) == len(base_toks), "Length mismatch"
        assert torch.equal(base_toks[torch.arange(base_toks.size(0)), asst_indices], source_toks[torch.arange(source_toks.size(0)), asst_indices]), f"Mismatch in assistant tokens: {base_toks[torch.arange(base_toks.size(0)), asst_indices]} vs {source_toks[torch.arange(source_toks.size(0)), asst_indices]}"
        return asst_indices
        
    def get_user_tokens(self, base_toks, source_toks):
        source_toks = self.align_toks(source_toks, base_toks)
        source_toks = source_toks['input_ids']
        base_toks = base_toks['input_ids']
        tokenizer = self.model_handler.tokenizer
        # The user-turn marker is model-specific; it was hardcoded to Qwen's.
        # ModelHandler.user_marker carries the right one per chat format.
        user_marker = getattr(self.model_handler, 'user_marker', "<|im_start|>user\n")
        user_tokens = tokenizer(user_marker, return_tensors="pt")["input_ids"][0]
        # get_user_tokens matches on user_tokens[1] (the role word), so the
        # marker must tokenize to [<start-of-turn>, <role word>, ...].
        print(user_tokens)
        pad_token_id = self.model_handler.tokenizer.pad_token_id
        user_indices = []
        for idx, source in enumerate(source_toks):
            # print('source ', source.shape)
            len_eoq = len(user_indices)
            for i in range(len(source) - 1, -1, -1):
                if source[i] == user_tokens[1]:
                    print('---------------------------------------------------------')
                    print('USER TOKENS: SOURCE ', tokenizer.decode(source[i]), ' BASE ', tokenizer.decode(base_toks[idx][i]))
                    print('---------------------------------------------------------')
                    user_indices.append(i)
                    break
            if len(user_indices) == len_eoq:
                assert False, "End of query not found"
        
        assert len(user_indices) == len(base_toks), "Length mismatch"
        assert torch.equal(base_toks[torch.arange(base_toks.size(0)), user_indices], source_toks[torch.arange(source_toks.size(0)), user_indices]), f"Mismatch in assistant tokens: {base_toks[torch.arange(base_toks.size(0)), user_indices]} vs {source_toks[torch.arange(source_toks.size(0)), user_indices]}"
        return user_indices
    