"""
Representative RAMs Fusion Architecture for NPC Prognostic Modeling.
This script demonstrates tensor flow and model structure using synthetic inputs.
"""
 
from __future__ import annotations
import torch
import torch.nn as nn
 
 
class RAMsFusionNetwork(nn.Module):
    """Representative dual-level recursive attention fusion network."""
    def __init__(self, radiomics_dim: int = 13, clinical_dim: int = 5, pathomics_dim: int = 3, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.30):
        super().__init__()
        self.rad_proj = nn.Linear(radiomics_dim, hidden_dim)
        self.clin_proj = nn.Linear(clinical_dim, hidden_dim)
        self.path_proj = nn.Linear(pathomics_dim, hidden_dim)
        self.attn_level1 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_level2 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())
 
    def forward(self, radiomics, clinical, pathomics):
        q_rad = self.rad_proj(radiomics).unsqueeze(1)
        kv_clin = self.clin_proj(clinical).unsqueeze(1)
        kv_path = self.path_proj(pathomics).unsqueeze(1)
        clin_rad_repr, _ = self.attn_level1(query=q_rad, key=kv_clin, value=kv_clin)
        joint_repr = q_rad + clin_rad_repr
        path_recalibrated, _ = self.attn_level2(query=joint_repr, key=kv_path, value=kv_path)
        fused_repr = (joint_repr + path_recalibrated).squeeze(1)
        return self.classifier(fused_repr)
 
 
if __name__ == "__main__":
    model = RAMsFusionNetwork()
    batch_size = 8
    radiomics = torch.randn(batch_size, 13)
    clinical = torch.randn(batch_size, 5)
    pathomics = torch.randn(batch_size, 3)
    output = model(radiomics, clinical, pathomics)
    print("Output shape:", output.shape)
    print("Example RAMs scores:", output.detach().cpu().numpy().round(4).ravel())
