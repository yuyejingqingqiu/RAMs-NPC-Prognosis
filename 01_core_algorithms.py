"""
Core Algorithms for RAMs Multimodal Fusion Model in Nasopharyngeal Carcinoma
 
Part 1: Spatial topology feature extraction, including:
- Spatial Shannon Entropy (SSE)
- Tumor-Immune Interaction Index (TIII)
- Nearest Immune Distance (NID)
 
Part 2: Demonstration Logistic-LASSO feature selection with cross-validation.
This script is intended for code demonstration using synthetic data.
"""
 
from __future__ import annotations
from collections import Counter
from typing import Tuple
 
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError, distance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
 
 
def _normalize_cell_type(x):
    """Map common cell labels to tumor=0, immune=1, stroma=2."""
    if isinstance(x, str):
        x_lower = x.strip().lower()
        if x_lower in {"tumor", "tumour", "cancer", "0"}:
            return 0
        if x_lower in {"immune", "lymphocyte", "til", "1"}:
            return 1
        if x_lower in {"stroma", "stromal", "2"}:
            return 2
    return int(x)
 
 
def calculate_spatial_topology_features(cell_data: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Calculate SSE, TIII, and NID from cell coordinates and cell types.
    Required columns: x, y, cell_type.
    """
    required_cols = {"x", "y", "cell_type"}
    if not required_cols.issubset(cell_data.columns):
        raise ValueError(f"cell_data must contain columns: {required_cols}")
 
    df = cell_data.copy()
    df["cell_type"] = df["cell_type"].apply(_normalize_cell_type)
    coords = df[["x", "y"]].to_numpy(dtype=float)
    cell_types = df["cell_type"].to_numpy()
 
    tumor_coords = coords[cell_types == 0]
    immune_coords = coords[cell_types == 1]
    if len(tumor_coords) > 0 and len(immune_coords) > 0:
        dist_matrix = distance.cdist(tumor_coords, immune_coords, metric="euclidean")
        nid = float(np.mean(np.min(dist_matrix, axis=1)))
    else:
        nid = 0.0
 
    if len(coords) < 4:
        return 0.0, 0.0, nid
 
    try:
        tri = Delaunay(coords)
    except QhullError:
        return 0.0, 0.0, nid
 
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[0], simplex[2]))))
 
    e_ti = 0
    e_tt = 0
    adjacency_list = {i: [] for i in range(len(coords))}
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
        type_u, type_v = cell_types[u], cell_types[v]
        if {type_u, type_v} == {0, 1}:
            e_ti += 1
        elif type_u == 0 and type_v == 0:
            e_tt += 1
 
    tiii = float(e_ti) / float(e_tt) if e_tt > 0 else 0.0
 
    local_entropies = []
    for _, neighbors in adjacency_list.items():
        if not neighbors:
            continue
        neighbor_types = [cell_types[n] for n in neighbors]
        counts = Counter(neighbor_types)
        total = len(neighbor_types)
        entropy = 0.0
        for count in counts.values():
            p_i = count / total
            entropy -= p_i * np.log2(p_i)
        local_entropies.append(entropy)
 
    sse = float(np.mean(local_entropies)) if local_entropies else 0.0
    return sse, tiii, nid
 
 
def run_logistic_lasso_selection(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegressionCV(Cs=50, cv=10, penalty="l1", solver="liblinear", scoring="roc_auc", max_iter=2000, random_state=2026, n_jobs=-1)
    model.fit(X_scaled, y)
    coefs = model.coef_[0]
    selected_mask = coefs != 0
    result = pd.DataFrame({"Feature": X.columns[selected_mask], "LASSO_Weight": coefs[selected_mask]})
    return result.sort_values("LASSO_Weight", key=abs, ascending=False).reset_index(drop=True)
 
 
if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    mock_cells = pd.DataFrame({"x": rng.random(100) * 1000, "y": rng.random(100) * 1000, "cell_type": rng.choice([0, 1, 2], 100)})
    sse, tiii, nid = calculate_spatial_topology_features(mock_cells)
    print(f"SSE = {sse:.4f}, TIII = {tiii:.4f}, NID = {nid:.4f}")
    mock_X = pd.DataFrame(rng.normal(size=(100, 50)), columns=[f"Feature_{i}" for i in range(50)])
    mock_y = rng.choice([0, 1], 100)
    print(run_logistic_lasso_selection(mock_X, mock_y).head())
