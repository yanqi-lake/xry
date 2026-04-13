#!/usr/bin/env python3
"""
Mixture Similarity Heatmap with Hierarchical Clustering
=============================================
Based on: Snitz et al. (2013) PLoS Comput Biol 9(9): e1003184
Method: Cosine similarity between mixture total vectors + hierarchical clustering

USAGE:
======
python mixture_similarity_heatmap.py

OUTPUT:
=======
- Console: Pairwise similarity matrix
- Heatmap with dendrogram: mixture_similarity_heatmap.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


def normalize_descriptors(desc_matrix):
    """Normalize each descriptor to [0,1] range"""
    desc_matrix = np.array(desc_matrix, dtype=float)
    desc_matrix[desc_matrix == -999] = np.nan
    
    col_min = np.nanmin(desc_matrix, axis=0)
    col_max = np.nanmax(desc_matrix, axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    
    normalized = (desc_matrix - col_min) / col_range
    normalized = np.nan_to_num(normalized, nan=0.0)
    
    return normalized


def create_mixture_vector(indices, desc_matrix):
    """Create mixture vector by summing normalized component descriptors"""
    if max(indices) >= desc_matrix.shape[0]:
        indices = [i for i in indices if i < desc_matrix.shape[0]]
    
    mix_vec = np.sum(desc_matrix[indices], axis=0)
    norm = np.linalg.norm(mix_vec)
    if norm > 0:
        mix_vec = mix_vec / norm
    return mix_vec


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    dot = np.dot(vec1, vec2)
    n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def main():
    print("=" * 70)
    print("Mixture Similarity Heatmap with Clustering")
    print("Based on Snitz et al. (2013) PLoS Comput Biology")
    print("=" * 70)
    
    # =========================================================================
    # Load Data
    # =========================================================================
    DESCRIPTOR_FILE = '13-descriptors-dragon.xlsx'
    MIXTURE_FILE = 'mixture-components.xlsx'
    
    print("\n[1] Loading molecular descriptors...")
    df_desc = pd.read_excel(DESCRIPTOR_FILE, header=None, skiprows=3)
    df_desc.columns = ['No', 'MOL_ID'] + [f'D{i}' for i in range(df_desc.shape[1]-2)]
    df_desc = df_desc.drop(columns=['No'])
    
    molecules = df_desc['MOL_ID'].values
    mol_to_idx = {name: idx for idx, name in enumerate(molecules)}
    n_molecules = len(molecules)
    n_descriptors = df_desc.shape[1] - 1
    
    print(f"  Loaded: {n_molecules} molecules, {n_descriptors} descriptors")
    
    print("\n[2] Loading mixture composition...")
    df_mix = pd.read_excel(MIXTURE_FILE, header=0)
    df_mix.columns = ['Similarity Level', 'Mixture', 'Component 1', 'Component 2', 'Component 3', 'Component 4']
    
    # Parse mixtures
    mixtures = {}
    component_cols = ['Component 1', 'Component 2', 'Component 3', 'Component 4']
    
    for idx, row in df_mix.iterrows():
        mix_name = row['Mixture']
        sim_level = row['Similarity Level'] if pd.notna(row['Similarity Level']) else 'N/A'
        components = []
        component_names = []
        for col in component_cols:
            if col in df_mix.columns and pd.notna(row[col]):
                mol_name = str(row[col]).strip()
                component_names.append(mol_name)
                if mol_name in mol_to_idx:
                    components.append(mol_to_idx[mol_name])
        
        if components:
            mixtures[mix_name] = {
                'components': components,
                'component_names': component_names,
                'level': sim_level
            }
    
    print(f"  Loaded {len(mixtures)} mixtures:")
    for name, data in mixtures.items():
        print(f"    {name} ({data['level']}): {data['component_names']}")
    
    # =========================================================================
    # Normalize Descriptors
    # =========================================================================
    print("\n[3] Normalizing descriptors...")
    
    desc_cols = df_desc.columns[1:]
    desc_matrix = df_desc[desc_cols].values.astype(float)
    norm_matrix = normalize_descriptors(desc_matrix)
    print(f"  Normalized: {norm_matrix.shape}")
    
    # =========================================================================
    # Create Mixture Vectors
    # =========================================================================
    print("\n[4] Creating mixture vectors...")
    
    mix_names = list(mixtures.keys())
    mix_vectors = {}
    
    for mix_name, data in mixtures.items():
        mix_vec = create_mixture_vector(data['components'], norm_matrix)
        mix_vectors[mix_name] = mix_vec
    
    # Create vector matrix
    vector_matrix = np.array([mix_vectors[name] for name in mix_names])
    print(f"  Mixture vector matrix: {vector_matrix.shape}")
    
    # =========================================================================
    # Compute Similarity Matrix
    # =========================================================================
    print("\n[5] Computing pairwise similarity...")
    
    n = len(mix_names)
    similarity_matrix = np.zeros((n, n))
    
    for i, name1 in enumerate(mix_names):
        for j, name2 in enumerate(mix_names):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity_matrix[i, j] = cosine_similarity(mix_vectors[name1], mix_vectors[name2])
    
    # Create DataFrame
    sim_df = pd.DataFrame(similarity_matrix, index=mix_names, columns=mix_names)
    
    # Print matrix
    print("\n" + "=" * 70)
    print("COSINE SIMILARITY MATRIX")
    print("=" * 70)
    
    print(f"\n{'Mixture':<12}", end="")
    for name in mix_names:
        print(f"{name:<12}", end="")
    print()
    print("-" * (12 * (n + 1)))
    
    for i, name1 in enumerate(mix_names):
        print(f"{name1:<12}", end="")
        for j, name2 in enumerate(mix_names):
            sim = similarity_matrix[i, j]
            print(f"{sim:>10.4f}  ", end="")
        print()
    
    # =========================================================================
    # Hierarchical Clustering
    # =========================================================================
    print("\n[6] Performing hierarchical clustering...")
    
    # Convert similarity to distance
    dist_matrix = 1 - similarity_matrix
    
    # Perform clustering using Ward's method
    linkage = sch.linkage(ssd.squareform(dist_matrix), method='ward')
    
    # Get cluster labels at different cut points
    print("\n  Cluster dendrogram cut points:")
    
    for n_clusters in [2, 3, 4]:
        labels = sch.fcluster(linkage, n_clusters, criterion='maxclust')
        print(f"\n  {n_clusters} clusters:")
        for cluster_id in range(1, n_clusters + 1):
            members = [mix_names[i] for i in range(len(mix_names)) if labels[i] == cluster_id]
            levels = [mixtures[m]['level'] for m in members]
            print(f"    Cluster {cluster_id}: {members} ({levels})")
    
    # =========================================================================
    # Create Heatmap with Clustering
    # =========================================================================
    print("\n[7] Creating clustered heatmap...")
    
    # Create clustermap (includes dendrogram)
    g = sns.clustermap(
        sim_df,
        method='ward',
        metric='euclidean',
        cmap='RdYlBu_r',
        vmin=0,
        vmax=1,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        figsize=(12, 10),
        dendrogram_ratio=(15, 15),
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    # Add title
    g.fig.suptitle('Mixture Similarity Heatmap with Hierarchical Clustering\n(Ward\'s method)', 
                  fontsize=14, fontweight='bold', y=1.02)
    
    # Save figure
    output_file = 'mixture_similarity_heatmap.png'
    g.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Heatmap saved to: {output_file}")
    
    # Also save as regular heatmap (no clustering)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        sim_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Cosine Similarity'},
        ax=ax
    )
    
    ax.set_title('Mixture Similarity Heatmap', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plain_heatmap_file = 'mixture_heatmap_plain.png'
    plt.savefig(plain_heatmap_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Plain heatmap saved to: {plain_heatmap_file}")
    
    # Save similarity matrix
    excel_file = 'mixture_similarity_matrix.xlsx'
    sim_df.to_excel(excel_file)
    print(f"  Similarity matrix saved to: {excel_file}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    upper_tri = similarity_matrix[np.triu_indices(n, k=1)]
    print(f"\n  Mean similarity: {np.mean(upper_tri):.4f}")
    print(f"  Std similarity: {np.std(upper_tri):.4f}")
    print(f"  Min similarity: {np.min(upper_tri):.4f}")
    print(f"  Max similarity: {np.max(upper_tri):.4f}")
    
    # Most similar pairs
    print("\n  Top 5 most similar mixture pairs:")
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((mix_names[i], mix_names[j], similarity_matrix[i, j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    for m1, m2, sim in pairs[:5]:
        print(f"    {m1} - {m2}: {sim:.4f}")
    
    # Most dissimilar pairs
    print("\n  Top 5 most dissimilar mixture pairs:")
    for m1, m2, sim in pairs[-5:]:
        print(f"    {m1} - {m2}: {sim:.4f}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()