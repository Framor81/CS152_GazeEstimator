# Imports
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np

# Embedding
class SwinEmbedding(nn.Module):
    # Converts input images into embedded patches and flattens them for transformer processing.
    def __init__(self, patch_size=4, emb_size=96):
        super().__init__()
        # Convolutional layer to create patch embeddings
        self.linear_embedding = nn.Conv2d(3, emb_size, kernel_size=patch_size, stride=patch_size)
        # Rearrange layer to flatten the patches
        self.rearrange = Rearrange('b c h w -> b (h w) c')
        
    def forward(self, x):
        # Apply convolution to create patch embeddings
        x = self.linear_embedding(x)
        # Flatten the patches into a sequence
        x = self.rearrange(x)
        return x
    
# Patch Merging
class PatchMerging(nn.Module):
    # Merges adjacent patches to reduce spatial dimensions and increase channel dimensions. Used for hierarchical representation in Swin Transformer.
    def __init__(self, emb_size):
        super().__init__()
        # Linear layer to project merged patches to a new dimension
        self.linear = nn.Linear(4 * emb_size, 2 * emb_size)
        
    def forward(self, x):
        # Get batch size, sequence length, and channels
        B, L, C = x.shape
        # Calculate height and width of the feature map
        H = W = int(np.sqrt(L)/2)
        # Rearrange and merge patches
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s1 s2 c)', s1=2, s2=2, h=H, w=W)
        # Apply linear transformation
        x = self.linear(x)
        return x
    
# Shifted Window Attention
class ShiftedWindowMSA(nn.Module):
    def __init__(self, emb_size, num_heads, window_size=7, shifted=True):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shifted = shifted
        # Linear layers for query, key, and value projections
        self.linear1 = nn.Linear(emb_size, 3 * emb_size)
        self.linear2 = nn.Linear(emb_size, emb_size)
        # Positional embeddings for relative position encoding
        self.pos_embeddings = nn.Parameter(torch.randn(window_size * 2 - 1, window_size * 2 - 1))
        # Indices for relative position calculation
        self.indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
        self.relative_indices = self.indices[None, :, :] - self.indices[:, None, :]
        self.relative_indices += self.window_size - 1
        
    def forward(self, x):
        # Calculate dimension per head
        h_dim = self.emb_size / self.num_heads
        # Calculate height and width of the feature map
        height = width = int(np.sqrt(x.shape[1]))
        # Project input to query, key, and value
        x = self.linear1(x)
        
        # Rearrange for multi-head attention
        x = rearrange(x, 'b (h w) (c k) -> b h w c k', h=height, w=width, k=3, c=self.emb_size)
        
        if self.shifted:
            # Shift the feature map for shifted window attention
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1, 2))
            
        # Rearrange for window-based attention
        x = rearrange(x, 'b (Wh w1) (Ww w2) (e H) k -> b H Wh Ww (w1 w2) e k', w1=self.window_size, w2=self.window_size, H=self.num_heads)            

        # Split into query, key, and value
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        # Calculate attention weights
        wei = (Q @ K.transpose(4,5)) / np.sqrt(h_dim)
        
        # Add relative positional encoding
        rel_pos_embedding = self.pos_embeddings[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        wei += rel_pos_embedding
        
        if self.shifted:
            # Apply masks for shifted windows
            row_mask = torch.zeros((self.window_size**2, self.window_size**2)).cuda()
            row_mask[-self.window_size * (self.window_size//2):, 0:-self.window_size * (self.window_size//2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size)
            wei[:, :, -1, :] += row_mask
            wei[:, :, :, -1] += column_mask
        
        # Apply softmax to get attention scores and multiply by value
        wei = F.softmax(wei, dim=-1) @ V
        
        # Rearrange back to original shape
        x = rearrange(wei, 'b H Wh Ww (w1 w2) e -> b (Wh w1) (Ww w2) (H e)', w1=self.window_size, w2=self.window_size, H=self.num_heads)
        x = rearrange(x, 'b h w c -> b (h w) c')
        
        # Final linear projection
        return self.linear2(x)
    
# MLP
class MLP(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        # Feed-forward network with GELU activation
        self.ff = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GELU(),
            nn.Linear(4 * emb_size, emb_size),
        )
    def forward(self, x):
        return self.ff(x)

# Swin Encoder Block
class SwinEncoder(nn.Module):
    def __init__(self, emb_size, num_heads, window_size=7):
        super().__init__()
        # Window-based multi-head self-attention
        self.WMSA = ShiftedWindowMSA(emb_size, num_heads, window_size, shifted=False)
        # Shifted window-based multi-head self-attention
        self.SWMSA = ShiftedWindowMSA(emb_size, num_heads, window_size, shifted=True)
        # Layer normalization
        self.ln = nn.LayerNorm(emb_size)
        # MLP layer
        self.MLP = MLP(emb_size)
        
    def forward(self, x):
        # Apply window-based attention and MLP
        x = x + self.WMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))
        # Apply shifted window-based attention and MLP
        x = x + self.SWMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))
        
        return x
    
# Swin Transformer
class Swin(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial embedding layer
        self.Embedding = SwinEmbedding()
        # Patch merging layers
        self.PatchMerging = nn.ModuleList()
        emb_size = 96
        num_class = 5
        for i in range(3):
            # Append patch merging layers
            self.PatchMerging.append(PatchMerging(emb_size))
            emb_size *= 2
            
        # Encoder stages with increasing complexity
        self.stage1 = SwinEncoder(96, 3)
        self.stage2 = SwinEncoder(192, 6)
        self.stage3 = nn.ModuleList([SwinEncoder(384, 12), SwinEncoder(384, 12), SwinEncoder(384, 12)])
        self.stage4 = SwinEncoder(768, 24)
        
        # Pooling and classification layers
        self.avgpool1d = nn.AdaptiveAvgPool1d(output_size=1)
        self.avg_pool_layer = nn.AvgPool1d(kernel_size=49)
        self.layer = nn.Linear(768, num_class)
        
    def forward(self, x):
        # Apply embedding
        x = self.Embedding(x)
        # Pass through encoder stages
        x = self.stage1(x)
        x = self.PatchMerging[0](x)
        x = self.stage2(x)
        x = self.PatchMerging[1](x)
        for stage in self.stage3:
            x = stage(x)
            
        x = self.PatchMerging[2](x)
        x = self.stage4(x)
        # Apply final classification layer
        x = self.layer(self.avgpool1d(x.transpose(1, 2)).squeeze(2))
        return x