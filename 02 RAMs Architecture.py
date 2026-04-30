RAMs: Recursive Attention Mechanisms for Multimodal Fusion
----------------------------------------------------------
This module defines the core PyTorch architecture of the RAMs fusion network, 
designed for prognostic risk stratification in Nasopharyngeal Carcinoma (NPC).
It hierarchically integrates macroscopic radiomics, clinical baseline factors, 
and microscopic pathomics (spatial topology) via a dual-level cross-attention mechanism.
"""

import torch
import torch.nn as nn

class RAMsFusionNetwork(nn.Module):
    """
    Multimodal Recursive Attention Fusion Network.
    
    Architecture logic:
    - Level-1: Radiomics features act as Query to attend to Clinical factors (Key/Value).
    - Level-2: The joint representation from Level-1 acts as Query to attend to Pathomics spatial topology (Key/Value).
    """
    def __init__(self, dim_clin: int = 5, dim_rad: int = 13, dim_path: int = 11, 
                 hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.3):
        super(RAMsFusionNetwork, self).__init__()
        
        # 1. Linear Projection Layers (Embedding to a unified latent space)
        self.proj_clin = nn.Linear(dim_clin, hidden_dim)
        self.proj_rad  = nn.Linear(dim_rad, hidden_dim)
        self.proj_path = nn.Linear(dim_path, hidden_dim)
        
        # 2. Level-1 Cross-Attention Module (Clinical-Radiomics Alignment)
        self.level1_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, 
                                                 dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # 3. Level-2 Recursive Attention Module (Spatial Topology Adaptive Reweighting)
        self.level2_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, 
                                                 dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # 4. Final Risk Scoring Head (Outputs probability 0 to 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )
        
    def forward(self, x_clin: torch.Tensor, x_rad: torch.Tensor, x_path: torch.Tensor) -> torch.Tensor:
        # Embed and add Sequence dimension: (batch_size, seq_len=1, hidden_dim)
        emb_clin = self.proj_clin(x_clin).unsqueeze(1) 
        emb_rad  = self.proj_rad(x_rad).unsqueeze(1)   
        emb_path = self.proj_path(x_path).unsqueeze(1) 
        
        # Level-1 Attention: Radiomics (Query) attends to Clinical (Key/Value)
        # Output shape: (batch_size, 1, hidden_dim)
        attn_out_1, _ = self.level1_attn(query=emb_rad, key=emb_clin, value=emb_clin)
        joint_rep_1 = self.norm1(emb_rad + self.dropout1(attn_out_1))
        
        # Level-2 Attention: Joint Rep 1 (Query) attends to Pathomics (Key/Value)
        attn_out_2, _ = self.level2_attn(query=joint_rep_1, key=emb_path, value=emb_path)
        joint_rep_2 = self.norm2(joint_rep_1 + self.dropout2(attn_out_2))
        
        # Squeeze sequence dimension to shape (batch_size, hidden_dim) and classify
        final_vector = joint_rep_2.squeeze(1) 
        risk_score = self.classifier(final_vector)
        
        return risk_score

# =====================================================================
# Self-Verification Module (Run this script to test)
# =====================================================================
if __name__ == "__main__":
    print("-" * 50)
    print("Testing PyTorch RAMs Architecture Forward Pass...")
    
    # Enable hardware acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using computation device: {device}")
    
    # Simulate a batch of 4 patients with your exact feature dimensions
    batch_size = 4
    mock_clin = torch.rand(batch_size, 5).to(device)   # 5 Independent Clinical factors
    mock_rad  = torch.rand(batch_size, 13).to(device)  # 13 Core Radiomics features
    mock_path = torch.rand(batch_size, 11).to(device)  # 11 Pathomics features
    
    # Initialize the model and move to device
    model = RAMsFusionNetwork().to(device)
    model.eval() # Set to evaluation mode for deterministic output
    
    with torch.no_grad():
        mock_fusion_scores = model(mock_clin, mock_rad, mock_path)
    
    print("[SUCCESS] RAMs Network instantiated and forward pass executed flawlessly!")
    print(f"-> Input Batch Size: {batch_size} patients")
    print(f"-> Output Tensor Shape: {list(mock_fusion_scores.shape)} (Expected: [4, 1])")
    print(f"-> Calculated RAMs Fusion Scores (Probabilities):\n{mock_fusion_scores.cpu().numpy()}")
    print("-" * 50)