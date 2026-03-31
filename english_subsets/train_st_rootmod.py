import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import json

# 1. ARCHITECTURE DEFINITION (525k Parameters)
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=1024, d_model=128, n_layers=2, n_heads=4, max_len=128):
        super().__init__()
        self.d_model = d_model
        # Shared Embedding and Output Head (Weight Tying)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # Transformer Blocks
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=n_heads, 
                dim_feedforward=d_model * 4, # 4x expansion
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        # Note: head weight is tied to token_embedding.weight
        self.max_len = max_len

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, device=idx.device).unsqueeze(0)
        
        x = self.token_embedding(idx) + self.pos_embedding(pos)
        
        # Correct PyTorch method for causal masking
        mask = nn.Transformer.generate_square_subsequent_mask(t).to(idx.device)
        # Modern PyTorch prefers bool masks
        mask = mask == -float('inf')
        
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
            
        x = self.ln_f(x)
        
        # Project back to vocab using tied weights
        logits = F.linear(x, self.token_embedding.weight)
        return logits

# 2. TOKENIZER DEFINITION
class PrimitiveTokenizer:
    def __init__(self, words, modifiers):
        self.itos = {0: "[PAD]", 1: "[SOS]", 2: "[EOS]", 3: "[UNK]"}
        # Map words and modifiers to unique IDs
        all_tokens = sorted(list(set(words + modifiers)))
        for i, token in enumerate(all_tokens):
            self.itos[i + 4] = token.upper()
        self.stoi = {v: k for k, v in self.itos.items()}
        self.vocab_size = len(self.itos)

    def encode(self, text):
        tokens = text.upper().split()
        return [self.stoi.get(t, 3) for t in tokens]

    def decode(self, ids):
        return " ".join([self.itos.get(i, "[UNK]") for i in ids])

# 3. TRAINING UTILITIES
def train():
    # Load Vocab (Limiting to 1024 for this specific target)
    df = pd.read_csv("vocab.csv")
    core_words = df["word"].dropna().str.upper().unique()[:1000].tolist()
    modifiers = ["[PL]", "[ER]", "[ING]", "[ED]", "[LY]", "[UN]", "[CMP]", "[SUP]"]
    
    tokenizer = PrimitiveTokenizer(core_words, modifiers)
    print(f"Vocab Size: {tokenizer.vocab_size}")
    
    model = TinyTransformer(vocab_size=tokenizer.vocab_size)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {param_count:,}")

    # Mock Data Loading with diverse examples
    mock_sentences = [
        "THE CHILD [PL] BE PLAY [ING] IN THE SUN .",
        "THE DOG [PL] BE RUN [ING] FAST .",
        "HE BE A GOOD TEACH [ER] .",
        "SHE BE TALL [CMP] THAN I .",
        "THEY BE [ED] HAPPY [LY] PLAY [ING] .",
        "THIS MACHINE BE A COLD BOX ."
    ]
    
    # Pad to equal length
    max_len = 12
    encoded = []
    for s in mock_sentences:
        tokens = tokenizer.encode(s)
        # Simple padding
        tokens = tokens + [0] * (max_len - len(tokens))
        encoded.append(tokens[:max_len])
        
    train_data = torch.tensor(encoded)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    print("\nStarting Training...")
    for epoch in range(100):
        # Shift inputs/targets for next-token prediction
        inputs = train_data[:, :-1]
        targets = train_data[:, 1:]
        
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, tokenizer.vocab_size), targets.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    print("\nTraining Complete.")
    
    # Save Model and Tokenizer
    torch.save(model.state_dict(), "st_rootmod_500k.pt")
    with open("tokenizer_config.json", "w") as f:
        json.dump(tokenizer.itos, f)
    print("Model saved to 'st_rootmod_500k.pt'")

if __name__ == "__main__":
    train()
