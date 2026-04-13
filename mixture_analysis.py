#!/usr/bin/env python3
"""
Mixture Similarity & Complexity Analysis
=====================================
- Heatmap of mixture pairwise similarity
- Publication-quality clustering figure
- Information entropy / complexity bar chart

USAGE:
======
python mixture_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import warnings
warnings.filterwarnings('ignore')

# Style settings for publication
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

sns.set_style("ticks")


def normalize_descriptors(desc_matrix):
    """Normalize descriptors to [0,1]"""
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
    """Create mixture total vector"""
    mix_vec = np.sum(desc_matrix[indices], axis=0)
    norm = np.linalg.norm(mix_vec)
    if norm > 0:
        mix_vec = mix_vec / norm
    return mix_vec


def cosine_similarity(vec1, vec2):
    """Cosine similarity"""
    dot = np.dot(vec1, vec2)
    n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def information_entropy(vector):
    """Calculate Shannon entropy of a vector"""
    p = np.abs(vector)
    total = np.sum(p)
    if total == 0:
        return 0.0
    p = p / total
    return -np.sum(p[p > 0] * np.log2(p[p > 0]))


def rel_complexity(entropy, n_descriptors):
    """Relative perceptual complexity"""
    max_entropy = np.log2(n_descriptors)
    return entropy / max_entropy


def main():
    print("=" * 70)
    print("Mixture Similarity & Complexity Analysis")
    print("=" * 70)
    
    # Load data
    DESCRIPTOR_FILE = '13-descriptors-dragon.xlsx'
    MIXTURE_FILE = 'mixture-components.xlsx'
    
    # Load descriptors
    df_desc = pd.read_excel(DESCRIPTOR_FILE, header=None, skiprows=3)
    df_desc.columns = ['No', 'MOL_ID'] + [f'D{i}' for i in range(df_desc.shape[1]-2)]
    df_desc = df_desc.drop(columns=['No'])
    
    molecules = df_desc['MOL_ID'].values
    mol_to_idx = {name: idx for idx, name in enumerate(molecules)}
    n_descriptors = df_desc.shape[1] - 1
    
    # Load mixtures
    df_mix = pd.read_excel(MIXTURE_FILE, header=0)
    df_mix.columns = ['Similarity Level', 'Mixture', 'Component 1', 'Component 2', 'Component 3', 'Component 4']
    
    # Parse mixtures
    mixtures = {}
    component_cols = ['Component 1', 'Component 2', 'Component 3', 'Component 4']
    
    for idx, row in df_mix.iterrows():
        mix_name = row['Mixture']
        sim_level = row['Similarity Level'] if pd.notna(row['Similarity Level']) else 'N/A'
        components = []
        for col in component_cols:
            if col in df_mix.columns and pd.notna(row[col]):
                mol_name = str(row[col]).strip()
                if mol_name in mol_to_idx:
                    components.append(mol_to_idx[mol_name])
        if components:
            mixtures[mix_name] = {'components': components, 'level': sim_level}
    
    # Normalize descriptors
    desc_cols = df_desc.columns[1:]
    desc_matrix = df_desc[desc_cols].values.astype(float)
    norm_matrix = normalize_descriptors(desc_matrix)
    
    # Create mixture vectors
    mix_names = list(mixtures.keys())
    mix_vectors = {name: create_mixture_vector(data['components'], norm_matrix) 
                  for name, data in mixtures.items()}
    
    # Compute similarity matrix
    n = len(mix_names)
    similarity_matrix = np.zeros((n, n))
    for i, name1 in enumerate(mix_names):
        for j, name2 in enumerate(mix_names):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity_matrix[i, j] = cosine_similarity(mix_vectors[name1], mix_vectors[name2])
    
    sim_df = pd.DataFrame(similarity_matrix, index=mix_names, columns=mix_names)
    
    # =========================================================================
    # FIGURE 1: Clean heatmap (for publication)
    # =========================================================================
    print("\n[1] Creating heatmap...")
    
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    
    # Create heatmap with better aesthetics
    hm = sns.heatmap(
        sim_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        vmin=0.8,
        vmax=1.0,
        square=True,
        linewidths=1,
        linecolor='white',
        cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.8},
        ax=ax1,
        annot_kws={'size': 9}
    )
    
    ax1.set_title('Perceptual Similarity Matrix\nof Odorant Mixtures', fontsize=14, fontweight='bold', pad=15)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0)
    
    plt.tight_layout()
    fig1.savefig('mixture_similarity_heatmap_pub.png', dpi=300, facecolor='white')
    print("  Saved: mixture_similarity_heatmap_pub.png")
    plt.close()
    
    # =========================================================================
    # FIGURE 2: Clustering dendrogram (for publication)
    # =========================================================================
    print("\n[2] Creating clustering figure...")
    
    # Prepare clustering
    dist_matrix = 1 - similarity_matrix
    condensed = ssd.squareform(dist_matrix)
    linkage = sch.linkage(condensed, method='ward')
    
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 6), 
                                      gridspec_kw={'width_ratios': [1.5, 1]})
    
    # Dendrogram
    dendro = sch.dendrogram(
        linkage,
        labels=mix_names,
        orientation='left',
        ax=ax2a,
        leaf_font_size=11,
        color_threshold=0.3,
        above_threshold_color='gray'
    )
    ax2a.set_xlabel('Distance (Ward)', fontsize=11)
    ax2a.set_title('Hierarchical Clustering\nDendrogram', fontsize=13, fontweight='bold')
    ax2a.spines['top'].set_visible(False)
    ax2a.spines['right'].set_visible(False)
    
    # Reorder similarity matrix by cluster
    order = dendro['leaves']
    ordered_names = [mix_names[i] for i in order]
    ordered_sim = similarity_matrix[np.ix_(order, order)]
    ordered_df = pd.DataFrame(ordered_sim, index=ordered_names, columns=ordered_names)
    
    # Clustered heatmap
    sns.heatmap(
        ordered_df,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        vmin=0.8,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Similarity', 'shrink': 0.6},
        ax=ax2b,
        annot_kws={'size': 8}
    )
    ax2b.set_title('Clustered\nSimilarity', fontsize=13, fontweight='bold')
    plt.setp(ax2b.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    fig2.savefig('mixture_clustering_pub.png', dpi=300, facecolor='white')
    print("  Saved: mixture_clustering_pub.png")
    plt.close()
    
    # =========================================================================
    # FIGURE 3: Complexity bar chart
    # =========================================================================
    print("\n[3] Creating complexity bar chart...")
    
    # Calculate complexity for each mixture
    complexities = {}
    entropies = {}
    
    for name in mix_names:
        vec = mix_vectors[name]
        ent = information_entropy(vec)
        comp = rel_complexity(ent, n_descriptors)
        entropies[name] = ent
        complexities[name] = comp
    
    # Order: EGM CGA AMI PCE NOP BCM MIC IME OCP
    order = ['EGM', 'CGA', 'AMI', 'PCE', 'NOP', 'BCM', 'MIC', 'IME', 'OCP']
    comp_values = [complexities[name] for name in order]
    entropy_values = [entropies[name] for name in order]
    
    # Create bar chart
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(order))
    width = 0.6
    
    # Use relative complexity (more interpretable)
    bars = ax3.bar(x, comp_values, width, color='steelblue', edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for bar, val in zip(bars, comp_values):
        height = bar.get_height()
        ax3.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9)
    
    ax3.set_xlabel('Mixture', fontsize=12)
    ax3.set_ylabel('Relative Complexity', fontsize=12)
    ax3.set_title('Chemical Complexity of Odorant Mixtures\n(Based on Relative Perceptual Complexity)', 
                fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(order, rotation=45, ha='right')
    ax3.set_ylim(0, 1.1)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.axhline(y=np.mean(comp_values), color='red', linestyle='--', linewidth=1, 
               label=f'Mean = {np.mean(comp_values):.3f}')
    ax3.legend(loc='upper right')
    
    # Add grid
    ax3.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax3.set_axisbelow(True)
    
    plt.tight_layout()
    fig3.savefig('mixture_complexity_bar.png', dpi=300, facecolor='white')
    print("  Saved: mixture_complexity_bar.png")
    plt.close()
    
    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPLEXITY VALUES")
    print("=" * 70)
    print(f"\n{'Mixture':<12}{'Entropy':<12}{'Complexity':<12}")
    print("-" * 36)
    for name in order:
        print(f"{name:<12}{entropies[name]:<12.4f}{complexities[name]:<12.4f}")
    
    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)
    print("  1. mixture_similarity_heatmap_pub.png - Similarity heatmap")
    print("  2. mixture_clustering_pub.png - Clustering figure")  
    print("  3. mixture_complexity_bar.png - Complexity bar chart")
    print("=" * 70)


if __name__ == "__main__":
    main()