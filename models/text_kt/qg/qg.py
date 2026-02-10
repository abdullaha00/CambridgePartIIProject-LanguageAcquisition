import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.text_kt.common.tokens import TOK_G, TOK_Q, TOK_A, SPECIAL_TOKS

class LMKTQG(nn.Module):
    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        added = self.tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKS})
        
        # PAD GPT2
        self.tokenizer.pad_token = self.tokenizer.eos_token

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
    
    def forward(self, input_ids, attention_mask, difficulty, labels=None):

        #=== move to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        difficulty = difficulty.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        #=== token embds

        input_embs = self.model.get_input_embeddings()(input_ids)

        g_mask = (input_ids == self.g_id).int()  # (B, T)
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

    #=== G

    @torch.no_grad()
    def generate_question(
        self,
        prefix_text: str,
        difficulty: float,
        max_new_tokens: int = 40,
    ):
        self.eval()

        tok = self.tokenizer
        device = self.device

        # Build input ids: prefix + <G>
        if prefix_text:
            prompt = f"{prefix_text} {TOK_G}"
        else:
            prompt = f"{TOK_G}"

        input_ids = tok.encode(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

        attention_mask = torch.ones_like(input_ids)

        # Build embeddings
        input_embs = self.model.get_input_embeddings()(input_ids)

        # Inject difficulty at <G>
        g_pos = (input_ids == self.g_id).nonzero(as_tuple=False)[0, 1]
        diff = torch.tensor([[difficulty]], dtype=torch.float, device=device)
        diff_emb = self.diff_embd(diff)

        input_embs[0, g_pos, :] += diff_emb[0]

        # Generate
        gen_ids = self.model.generate(
            inputs_embeds=input_embs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

        # Decode only the generated part (keep special tokens for parsing)
        new_ids = gen_ids[0, input_ids.shape[1]:]
        raw_text = tok.decode(new_ids, skip_special_tokens=False)
        
        # DEBUG: Log raw generated text
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Raw generated text: {raw_text[:200]}")

        # Expected format: <Q> question_text <A> (possibly more)
        # Extract question_text between <Q> and <A>
        if TOK_Q in raw_text and TOK_A in raw_text:
            # Find text between <Q> and <A>
            start = raw_text.find(TOK_Q) + len(TOK_Q)
            end = raw_text.find(TOK_A)
            if start < end:
                text = raw_text[start:end]
            else:
                text = ""
        elif TOK_A in raw_text:
            # Just stop at <A>
            text = raw_text.split(TOK_A)[0]
        else:
            # No special tokens found - return raw text up to eos
            text = raw_text
        
        # Clean up any remaining special tokens
        for tok_str in SPECIAL_TOKS:
            text = text.replace(tok_str, "")

        return text.strip()
