import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        scaling_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))

        pe[:, 0::2] = torch.sin(position * scaling_term)
        pe[:, 1::2] = torch.cos(position * scaling_term)

        self.pe = pe.unsqueeze(0)  # shape (1, max_len, dim)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes, num_layers, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),                  
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)                  # (batch_size, seq_len, embed_dim)
        x = self.pos_encoder(x)                # (batch_size, seq_len, embed_dim)
        x = x.transpose(0, 1)                  # Transformer expects (seq_len, batch_size, embed_dim)
        x = self.transformer(x)                # Output: (seq_len, batch_size, embed_dim)
        x = x.mean(dim=0)                      # (batch_size, embed_dim)
        return self.classifier(x)              # (batch_size, num_classes)
