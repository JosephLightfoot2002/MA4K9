import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.nn.attention import sdpa_kernel, SDPBackend
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # Compute the positional encodings once in log space.
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # Add positional encoding to input embeddings.
        x = x + self.pe[:, :seq_len, :]
        return x
    
# ----- Modified Dataset for Variable Length Sequences ----- #
class ReverseDataset(Dataset):
    def __init__(self, num_samples=1000, min_seq_len=5, max_seq_len=15, vocab_size=50):
        self.data = []
        self.vocab_size = vocab_size
        for _ in range(num_samples):
            seq_len = np.random.randint(min_seq_len, max_seq_len + 1)
            # Generate a random sequence with values in 1 ... vocab_size-1 (reserve 0 for padding)
            seq = torch.randint(1, vocab_size, (seq_len,))
            self.data.append(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = torch.flip(x.clone(), dims=[0])
        return x, y

# ----- Custom Collate Function for Padding ----- #
def collate_fn(batch):
    xs, ys = zip(*batch)
    # Determine the maximum length in the batch
    max_len = max([len(x) for x in xs])
    
    # Pad sequences with 0 (assumed pad token) to max_len
    padded_xs = torch.stack([torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)]) for x in xs])
    padded_ys = torch.stack([torch.cat([y, torch.zeros(max_len - len(y), dtype=torch.long)]) for y in ys])
    
    return padded_xs, padded_ys


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=4, nhead=2, num_layers=1, ff=32, max_len=50):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                batch_first=True, 
                activation='relu', 
                norm_first=True,
                dim_feedforward=ff
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_key_padding_mask=None):
        """
        src: (batch_size, seq_len) of token indices.
        src_key_padding_mask: (batch_size, seq_len) where True indicates padded positions.
        """
        # Embedding: (batch_size, seq_len, d_model)
        embedded = self.embedding(src)
        # Apply positional encoding
        embedded = self.pos_encoding(embedded)
        # Pass through transformer layers (each can use the padding mask)
        for layer in self.attention_layers:
            embedded = layer(embedded, src_key_padding_mask=src_key_padding_mask)
        # Output layer: (batch_size, seq_len, vocab_size)
        output = self.fc_out(embedded)
        return output

def create_padding_mask(batch):
    # Here, padded tokens are 0. Return mask of shape (batch_size, seq_len)
    return batch == 0
