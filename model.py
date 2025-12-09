# model.py
import torch
import torch.nn as nn

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=3, n_heads=4, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, n_heads, dim*4, dropout=0.1, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.max_len = max_len
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        x = self.tok_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.head(self.ln(x))

if __name__ == "__main__":
    model = TinyLLM(vocab_size=100)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
