import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

# 1. RE-DEFINE ARCHITECTURE (Must match training exactly)
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=1024, d_model=128, n_layers=2, n_heads=4, max_len=128):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, 
                dim_feedforward=d_model * 4, dropout=0.1,
                activation='gelu', batch_first=True, norm_first=True
            ) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.pos_embedding(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(idx.device)
        mask = mask == -float('inf')
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = F.linear(x, self.token_embedding.weight)
        return logits

# 2. TOKENIZER LOADER
class PrimitiveTokenizer:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.itos = {int(k): v for k, v in json.load(f).items()}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.vocab_size = len(self.itos)

    def encode(self, text):
        return [self.stoi.get(t.upper(), 3) for t in text.split()]

    def decode(self, ids):
        return " ".join([self.itos.get(i, "[UNK]") for i in ids])

# 3. INFERENCE ENGINE
def generate(model, tokenizer, prompt, max_new_tokens=10):
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)])
    
    for _ in range(max_new_tokens):
        # Crop context if it exceeds max_len (128)
        idx_cond = idx[:, -128:]
        
        with torch.no_grad():
            logits = model(idx_cond)
            # Focus on the last token's predictions
            logits = logits[:, -1, :]
            # Greedy decoding
            probs = F.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            # Append to the sequence
            idx = torch.cat((idx, next_token), dim=1)
            
            # Stop if we hit [EOS] (if implemented) or [PAD]
            if next_token.item() == 0: break 

    return tokenizer.decode(idx[0].tolist())

def main():
    model_path = "st_rootmod_500k.pt"
    config_path = "tokenizer_config.json"
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print("Error: Model or Tokenizer config not found. Run train_st_rootmod.py first.")
        return

    # Initialize Tokenizer
    tokenizer = PrimitiveTokenizer(config_path)
    
    # Initialize and Load Model
    model = TinyTransformer(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded {model_path} successfully.")

    # Test generation
    prompts = [
        "THE CHILD [PL] BE",
        "THE DOG [PL] BE",
        "HE BE A",
        "SHE BE"
    ]

    print("\nStarting ST-RootMod Generation Demo:")
    print("-" * 40)
    for p in prompts:
        completion = generate(model, tokenizer, p)
        print(f"Prompt: {p}")
        print(f"Model:  {completion}")
        print()

if __name__ == "__main__":
    main()
