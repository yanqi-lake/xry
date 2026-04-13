#!/usr/bin/env python3
"""
Molecular Similarity Heatmap using Dragon Descriptors
=========================================================
Based on: Snitz et al. (2013) PLoS Comput Biol 9(9): e1003184
Method: Cosine similarity between normalized Dragon descriptors

Paper URL: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003184

USAGE:
======
python molecular_similarity_heatmap.py

OUTPUT:
=======
- Console: Pairwise cosine similarity matrix
- Heatmap: similarity_heatmap.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


def normalize_descriptors(desc_matrix):
    """
    Normalize each descriptor to [0,1] range using formula:
    vn = (v - min(ld)) / (max(ld) - min(ld))
    
    Handles missing values (-999 or NaN)
    """
    # Handle missing values
    desc_matrix = np.array(desc_matrix, dtype=float)
    desc_matrix[desc_matrix == -999] = np.nan
    
    # Get min and max for each column (descriptor)
    col_min = np.nanmin(desc_matrix, axis=0)
    col_max = np.nanmax(desc_matrix, axis=0)
    
    # Avoid division by zero
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    
    # Normalize
    normalized = (desc_matrix - col_min) / col_range
    
    # Replace NaN with 0
    normalized = np.nan_to_num(normalized, nan=0.0)
    
    return normalized


def cosine_similarity_matrix(matrix):
    """
    Compute pairwise cosine similarity between all rows in a matrix.
    
    cosine_similarity = (A · B) / (||A|| × ||B||)
    """
    # Normalize each row (molecule vector)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = matrix / norms
    
    # Compute cosine similarity matrix
    # S = normalized @ normalized.T
    similarity = normalized @ normalized.T
    
    # Ensure diagonal is 1 (self-similarity)
    np.fill_diagonal(similarity, 1.0)
    
    return similarity


def main():
    print("=" * 70)
    print("Molecular Similarity Heatmap")
    print("Based on Snitz et al. (2013) PLoS Comput Biology")
    print("=" * 70)
    
    # =========================================================================
    # Load Data
    # =========================================================================
    DESCRIPTOR_FILE = '13-descriptors-dragon.xlsx'
    
    print("\n[1] Loading molecular descriptors...")
    df = pd.read_excel(DESCRIPTOR_FILE, header=None, skiprows=3)
    df.columns = ['No', 'MOL_ID'] + [f'D{i}' for i in range(df.shape[1]-2)]
    df = df.drop(columns=['No'])
    
    molecules = df['MOL_ID'].values
    n_molecules = len(molecules)
    n_descriptors = df.shape[1] - 1
    
    print(f"  Loaded: {n_molecules} molecules, {n_descriptors} descriptors")
    print(f"  Molecules: {list(molecules)}")
    
    # =========================================================================
    # Normalize Descriptors
    # =========================================================================
    print("\n[2] Normalizing descriptors to [0,1]...")
    
    desc_cols = df.columns[1:]
    desc_matrix = df[desc_cols].values.astype(float)
    
    norm_matrix = normalize_descriptors(desc_matrix)
    print(f"  Normalized matrix shape: {norm_matrix.shape}")
    print(f"  Value range: [{norm_matrix.min():.4f}, {norm_matrix.max():.4f}]")
    
    # =========================================================================
    # Compute Cosine Similarity
    # =========================================================================
    print("\n[3] Computing pairwise cosine similarity...")
    
    similarity_matrix = cosine_similarity_matrix(norm_matrix)
    
    # Print similarity matrix
    print("\n" + "=" * 70)
    print("COSINE SIMILARITY MATRIX")
    print("=" * 70)
    
    # Header
    print(f"\n{'Molecule':<25}", end="")
    for mol in molecules:
        # Truncate long names
        name = mol[:8] if len(mol) > 8 else mol
        print(f"{name:<12}", end="")
    print()
    print("-" * (12 * (n_molecules + 1)))
    
    # Matrix rows
    for i, mol1 in enumerate(molecules):
        name = mol1[:20] if len(mol1) > 20 else mol1
        print(f"{name:<25}", end="")
        for j, mol2 in enumerate(molecules):
            sim = similarity_matrix[i, j]
            print(f"{sim:>10.4f}  ", end="")
        print()
    
    # =========================================================================
    # Create Heatmap
    # =========================================================================
    print("\n[4] Creating heatmap...")
    
    # Create DataFrame for heatmap
    sim_df = pd.DataFrame(similarity_matrix, index=molecules, columns=molecules)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(
        sim_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',  # Red-Yellow-Blue reversed (red=high similarity)
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.8},
        ax=ax
    )
    
    # Customize
    ax.set_title('Molecular Similarity Heatmap\n(Based on Normalized Dragon Descriptors)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Molecules', fontsize=12)
    ax.set_ylabel('Molecules', fontsize=12)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_file = 'similarity_heatmap.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Heatmap saved to: {output_file}")
    
    # Also save similarity matrix to Excel
    excel_output = 'molecular_similarity_matrix.xlsx'
    sim_df.to_excel(excel_output)
    print(f"  Similarity matrix saved to: {excel_output}")
    
    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Get upper triangle (excluding diagonal)
    upper_tri = similarity_matrix[np.triu_indices(n_molecules, k=1)]
    
    print(f"\n  Number of molecule pairs: {len(upper_tri)}")
    print(f"  Mean similarity: {np.mean(upper_tri):.4f}")
    print(f"  Std similarity: {np.std(upper_tri):.4f}")
    print(f"  Min similarity: {np.min(upper_tri):.4f}")
    print(f"  Max similarity: {np.max(upper_tri):.4f}")
    
    # Most similar pairs
    print("\n  Top 5 most similar pairs:")
    pairs = []
    for i in range(n_molecules):
        for j in range(i+1, n_molecules):
            pairs.append((molecules[i], molecules[j], similarity_matrix[i, j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    for mol1, mol2, sim in pairs[:5]:
        print(f"    {mol1} - {mol2}: {sim:.4f}")
    
    # Most dissimilar pairs
    print("\n  Top 5 most dissimilar pairs:")
    for mol1, mol2, sim in pairs[-5:]:
        print(f"    {mol1} - {mol2}: {sim:.4f}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()