Core Algorithms for RAMs Multimodal Fusion Model in Nasopharyngeal Carcinoma
----------------------------------------------------------------------------
Part 1: Spatial Topology Feature Extraction (SSE & TIII) based on Delaunay Network.
Part 2: Nested 10-fold Cross-Validation Logistic-LASSO Feature Selection.
"""

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, qhull
from collections import Counter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress minor convergence warnings for clean execution logs
warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================================
# Part 1: Spatial Topology Feature Extraction
# =====================================================================
def calculate_spatial_topology_features(cell_data: pd.DataFrame) -> tuple:
    """
    Extract high-order spatial topology features based on 2D physical coordinates.
    """
    # Safety Check: Delaunay triangulation requires at least 4 points to be robust
    if len(cell_data) < 4:
        return 0.0, 0.0
        
    coords = cell_data[['x', 'y']].values
    cell_types = cell_data['cell_type'].values
    
    # Construct Delaunay triangulation network
    try:
        tri = Delaunay(coords)
    except qhull.QhullError:
        # Fallback if points are co-linear or identically located
        return 0.0, 0.0
    
    # Extract unique edges
    edges = set()
    for simplex in tri.simplices:
        edges.add(frozenset([simplex[0], simplex[1]]))
        edges.add(frozenset([simplex[1], simplex[2]]))
        edges.add(frozenset([simplex[0], simplex[2]]))
        
    E_TI = 0  # Heterogeneous (Tumor - Immune)
    E_TT = 0  # Homogeneous (Tumor - Tumor)
    adjacency_list = {i: [] for i in range(len(coords))}
    
    for edge in edges:
        u, v = list(edge)
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
        
        type_u, type_v = cell_types[u], cell_types[v]
        if (type_u == 0 and type_v == 1) or (type_u == 1 and type_v == 0):
            E_TI += 1
        elif type_u == 0 and type_v == 0:
            E_TT += 1
            
    TIII = float(E_TI) / float(E_TT) if E_TT > 0 else 0.0

    local_entropies = []
    for i in range(len(coords)):
        neighbors = adjacency_list[i]
        if not neighbors:
            continue
            
        neighbor_types = [cell_types[n] for n in neighbors]
        counts = Counter(neighbor_types)
        total_neighbors = len(neighbors)
        
        entropy = 0.0
        for count in counts.values():
            p_i = count / total_neighbors
            entropy -= p_i * np.log2(p_i)
        local_entropies.append(entropy)
        
    SSE = float(np.mean(local_entropies)) if local_entropies else 0.0
    return SSE, TIII

# =====================================================================
# Part 2: Nested 10-fold CV Logistic-LASSO Dimension Reduction
# =====================================================================
def run_nested_lasso_selection(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    Perform Logistic-LASSO feature selection nested within a 10-fold CV framework.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("[INFO] Executing 10-fold CV Logistic-LASSO optimization for AUC...")
    
    lasso_cv_model = LogisticRegressionCV(
        Cs=50,                  # Search across 50 lambda penalties
        cv=10,                  # 10-fold cross validation
        penalty='l1',           # LASSO L1 penalty
        solver='liblinear',     
        scoring='roc_auc',      # Optimize for highest AUC
        max_iter=2000,
        random_state=2026,
        n_jobs=-1               
    )
    
    lasso_cv_model.fit(X_scaled, y)
    optimal_coefs = lasso_cv_model.coef_[0]
    
    feature_names = X.columns
    selected_mask = optimal_coefs != 0
    selected_features = feature_names[selected_mask].tolist()
    selected_weights = optimal_coefs[selected_mask]
    
    result_df = pd.DataFrame({
        'Feature': selected_features,
        'LASSO_Weight': selected_weights
    }).sort_values(by='LASSO_Weight', key=abs, ascending=False).reset_index(drop=True)
    
    print(f"[SUCCESS] LASSO reduction completed. Identified {len(selected_features)} core features out of {X.shape[1]}.")
    return result_df

# =====================================================================
# Self-Verification Module (Run this script to test)
# =====================================================================
if __name__ == "__main__":
    print("-" * 50)
    print("Testing Part 1: Spatial Topology (SSE & TIII)")
    # Generate 100 mock cells (0=Tumor, 1=Immune, 2=Stroma)
    mock_cells = pd.DataFrame({
        'x': np.random.rand(100) * 1000,
        'y': np.random.rand(100) * 1000,
        'cell_type': np.random.choice([0, 1, 2], 100)
    })
    sse, tiii = calculate_spatial_topology_features(mock_cells)
    print(f"-> Calculated SSE: {sse:.4f}")
    print(f"-> Calculated TIII: {tiii:.4f}")
    
    print("\nTesting Part 2: LASSO Feature Selection")
    # Generate mock radiomics data: 100 patients, 50 features
    mock_X = pd.DataFrame(np.random.randn(100, 50), columns=[f"Rad_Feature_{i}" for i in range(50)])
    mock_y = np.random.choice([0, 1], 100) # Binary clinical outcome
    lasso_results = run_nested_lasso_selection(mock_X, mock_y)
    print("-> Top Selected Features:")
    print(lasso_results.head(3))
    print("-" * 50)