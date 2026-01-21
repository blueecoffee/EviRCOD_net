import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleDeformableEncoder(nn.Module):
    def __init__(self, dim=1024, num_heads=16, depths=[2, 2, 3, 4], mlp_ratio=4.):
        super().__init__()
        
        self.encoders = nn.ModuleList([
            DeformableEncoderBlock(dim, num_heads, depth, mlp_ratio) 
            for depth in depths
        ])

        self.cross_scale_fusion = CrossScaleFusion(dim, num_heads)

    def forward(self, x0, x1, x2, x3):
        x2 = self.cross_scale_fusion(x2, x3)
        x1 = self.cross_scale_fusion(x1, x2) 
        x0 = self.cross_scale_fusion(x0, x1)

        s3 = self.encoders[0](x3, mask=None)
        s2 = self.encoders[1](x2, mask=s3)
        s1 = self.encoders[2](x1, mask=s2)
        s0 = self.encoders[3](x0, mask=s1)
        
        return s0, s1, s2, s3


class DeformableEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, depth, mlp_ratio):
        super().__init__()
        self.depth = depth
        self.block = DeformableAttentionBlock(dim, num_heads, mlp_ratio)

    def forward(self, x, mask):
        for _ in range(self.depth):
            x = self.block(x, mask)
        return x


class DeformableAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.attn = DeformableMultiheadAttention(dim, num_heads)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim)

    def forward(self, x, mask):
        x_attn = self.attn(self.norm1(x), mask)
        x = x + x_attn
        
        x_mlp = self.mlp(self.norm2(x))
        x = x + x_mlp
        
        return x

class DeformableMultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.high_q = nn.Linear(dim, dim)
        self.high_k = nn.Linear(dim, dim)
        self.high_v = nn.Linear(dim, dim)
        self.mask_q = nn.Linear(dim, dim)
        
        self.proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

        self.offset_net = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim // 2),
            nn.GELU(),
            nn.Linear(self.head_dim // 2, 2) 
        )
    def apply_deformable_offset(self, k, offsets):
        B, H, N, D = k.shape
        offsets_expanded = offsets.unsqueeze(-1).expand(-1, -1, -1, -1, D)  # (B, H, N, 2, D)
        k_deformed = k.unsqueeze(3) + offsets_expanded * 0.1  # (B, H, N, 2, D)
        k_deformed = k_deformed.mean(dim=3)  # (B, H, N, D)
        return k_deformed

    def forward(self, high_fea, mask):
        B, N, C = high_fea.shape
        high_q = self.high_q(high_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        high_k = self.high_k(high_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        high_v = self.high_v(high_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        offsets = self.offset_net(high_q.reshape(-1, self.head_dim)).reshape(B, self.num_heads, N, 2)
        high_k = self.apply_deformable_offset(high_k, offsets)
        
        if mask is None:
            high_attn = torch.matmul(high_q, high_k.transpose(-2, -1)) * self.scale
            high_attn = high_attn.softmax(dim=-1)
            high_attn = (torch.matmul(high_attn, high_v)).transpose(2, 1).reshape(B, N, C)
        else:
            mask_q = self.mask_q(mask).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            high_attn = torch.matmul(mask_q, high_k.transpose(-2, -1)) * self.scale
            high_attn = high_attn.softmax(dim=-1)
            high_attn = (torch.matmul(high_attn, high_v)).transpose(2, 1).reshape(B, N, C)

        return high_attn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
class CrossScaleFusion(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, low_res, high_res):
        fused, _ = self.cross_attn(low_res, high_res, high_res)
        return self.norm(low_res + fused)