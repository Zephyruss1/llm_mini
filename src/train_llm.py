import os
import numpy as np
import torch  # type: ignore
import math
from torch.distributed.optim import ZeroRedundancyOptimizer  # type: ignore
from torch import nn  # type: ignore
from torch.nn import functional as F  # type: ignore
from dataclasses import dataclass
import inspect

torch.manual_seed(1333)

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12  # Renamed from d_head to n_head for clarity
    d_model: int = 768  # Consistent naming across the model

class NewGelu(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "Number of heads must divide output dimension"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Ensure each head has equal dimensionality

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape for multi-head attention
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # (b, num_heads, num_tokens, head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn_scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)  # (b, num_heads, num_tokens, num_tokens)

        # Apply causal mask
        mask_bool = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores = attn_scores.masked_fill(mask_bool.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute attention output
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu = NewGelu()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1  # Flag for weight initialization
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)  # Changed from n_embd to d_model
        self.attn = MultiHeadAttention(config.d_model, config.d_model, config.block_size, 0.1, config.n_head)
        self.ln_2 = nn.LayerNorm(config.d_model)  # Changed from n_embd to d_model

        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            NewGelu(),
            nn.Linear(4 * config.d_model, config.d_model)
        )
        # Removed duplicate initialization of self.attn

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.d_model),
            'wpe': nn.Embedding(config.block_size, config.d_model),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.d_model)
        })  # Changed '==' to '='
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1  # Flag for weight initialization
        self.transformer.wte.weight = self.lm_head.weight

        self.init_rng = torch.Generator()  # Fixed naming from init_rn to init_rng
        self.init_rng.manual_seed(1333)  # Corrected from init_rng.manual_seed
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02 / math.sqrt(2 * self.config.n_layer)
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)
    
    def forward(self, idx, targets=None, return_logits=True):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward, model block size is exhausted. Got {t} tokens, but configured for {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Token and position embeddings
        tok_emb = self.transformer.wte(idx)  # (b, t, d_model)
        pos_emb = self.transformer.wpe(pos)  # (t, d_model)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if not return_logits:
            logits = None

        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt3-small', 'gpt3-medium', 'gpt3-large', 'gpt3-xl'}, f"Model type {model_type} not implemented"
        from transformers import GPT2LMHeadModel
        print(f"Loading {model_type} from transformers library")

        # Define configuration parameters for each model type
        config_args = {
            'gpt3-small': {'n_layer': 12, 'n_head': 12, 'd_model': 768},
            'gpt3-medium': {'n_layer': 24, 'n_head': 16, 'd_model': 1024},
            'gpt3-large': {'n_layer': 24, 'n_head': 32, 'd_model': 1280},
            'gpt3-xl': {'n_layer': 24, 'n_head': 64, 'd_model': 1600}
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # Initialize Hugging Face model and load its state dict
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch for {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Check if fused AdamW is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda' and not zero_stage
        print(f"using fused AdamW: {use_fused}")

        if zero_stage == 1:
            optimizer = ZeroRedundancyOptimizer(optim_groups, optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        else:
            print("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    # --- Test MultiHeadAttention --- #
    config = GPTConfig()
    context_length = config.block_size

    mha = MultiHeadAttention(config.d_model, config.d_model, context_length, 0.1, config.n_head)
    batch = torch.randn(3, context_length, config.d_model)
    context_vectors = mha(batch)
    print(f"MultiHeadAttention output shape: {context_vectors.shape}")  # Expected: (3, 1024, 768)

    # --- Train GPT model --- #
    model = GPT(config=config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)

    # Optionally, you can add a dummy training step to verify everything works
    dummy_input = torch.randint(0, config.vocab_size, (2, config.block_size))  # (batch_size, sequence_length)
    logits, loss = model(dummy_input, targets=dummy_input)
    loss.backward()
    optimizer.step()
    print(f"Dummy training step completed with loss: {loss.item()}")
