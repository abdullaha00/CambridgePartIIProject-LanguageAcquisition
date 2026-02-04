import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

TOK_Q = "<Q>"
TOK_A = "<A>"
TOK_Y = "<Y>"
TOK_N = "<N>"
TOK_G = "<G>"

SPECIAL_TOKS = [TOK_Q, TOK_A, TOK_Y, TOK_N, TOK_G]

class LMKTQG(nn.Module):
    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        added = self.tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKS})
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        emb_dim = self.model.config.n_embd
        
        # scalar -> embedding is done via a linear layer
        self.diff_embd = nn.Linear(1, emb_dim)

        self.g_id = self.tokenizer.convert_tokens_to_ids(TOK_G)
        assert self.g_id != self.tokenizer.unk_token_id, f"Special token {TOK_G} not found in tokenizer."

        self.to(self.device)
    
    def forward(self, input_ids, difficulty, labels=None):

        #=== move to device
        input_ids = input_ids.to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        difficulty = difficulty.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        #=== token embds

        input_embs = self.model.get_input_embeddings(input_ids)

        g_mask = (input_ids == self.g_id)
        g_pos = g_mask.argmax(dim=1) 

        diff_embs = self.diff_embd(difficulty)  # (B, emb_dim)
        
        #scatter add (REPLACE)
        for i in range(input_ids.size(0)):
            input_embs[i, g_pos[i], :] += diff_embs[i]

        out = self.model(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return out

